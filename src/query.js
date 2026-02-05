import { readColumnIndex, readOffsetIndex } from './indexes.js'
import { parquetMetadataAsync, parquetSchema } from './metadata.js'
import { getSchemaPath } from './schema.js'
import { decodeDataPage, decodeDictionaryPage, readColumn } from './column.js'
import { parquetReadObjects } from './index.js'
import { DEFAULT_PARSERS } from './convert.js'
import { concat, equals } from './utils.js'
import {
  createColumnIndexMap,
  createNestedColumnIndexMap,
  createPredicates,
  extractFilterColumns,
  extractVariantFilterColumns,
  getRowGroupFullRange,
} from './plan.js'

/**
 * @import {AsyncBuffer, FileMetaData, ColumnChunk, SchemaElement, ColumnIndex, OffsetIndex, CompressionCodec, Compressors, ParquetParsers, RowGroup, DecodedArray, ParquetReadOptions, ParquetQueryFilter, VariantShredConfig} from './types.js'
 */

// Hoisted to module level to avoid creating new Set on every filter evaluation
const FILTER_OPERATORS = new Set(['$and', '$or', '$nor', '$not', '$comment'])

/**
 * Query parquet file with predicate pushdown.
 * This is a parquet-aware query engine that can read a subset of rows and columns.
 * Uses Parquet statistics to skip data that doesn't match filters.
 * Accepts optional filter object to filter the results and orderBy column name to sort the results.
 * Note that using orderBy may SIGNIFICANTLY increase the query time.
 *
 * Supports Variant shredding for predicate pushdown on nested Variant fields.
 * Pass variantConfig to enable dot-notation filters like { '$index.titleType': 'movie' }
 * which map to statistics on $index.typed_value.titleType.typed_value columns.
 *
 * @param {ParquetReadOptions & { filter?: ParquetQueryFilter, variantConfig?: VariantShredConfig[], orderBy?: string, desc?: boolean, offset?: number, limit?: number }} options
 * @returns {Promise<Record<string, any>[]>} resolves when all requested rows and columns are parsed
 */
export async function parquetQuery(options) {
  const { file, filter, columns, orderBy, desc = false, variantConfig = [] } = options

  // Fast path: delegate to parquetReadObjects when no query features are needed
  // This avoids the overhead of query infrastructure for simple reads
  if (!filter && !orderBy && variantConfig.length === 0) {
    const rowStart = options.offset ?? options.rowStart ?? 0
    const rowEnd = options.limit !== undefined
      ? rowStart + options.limit
      : options.rowEnd
    return parquetReadObjects({
      ...options,
      rowStart,
      rowEnd,
    })
  }

  const metadata = options.metadata || await parquetMetadataAsync(file)

  // Support both APIs since users might use either style
  const offset = options.offset ?? options.rowStart ?? 0
  if (offset < 0) throw new Error('parquet rowStart must be positive')
  const limit = options.limit ?? (options.rowEnd !== undefined ? options.rowEnd - offset : undefined)

  // Get schema once since we'll reference it multiple times
  const schema = parquetSchema(metadata)
  const allColumns = schema.children.map((c) => c.element.name)

  // Need both output columns and filter columns for evaluation
  // Use Variant-aware extraction if variantConfig is provided
  let filterColumns, statsColumns
  if (filter && variantConfig.length > 0) {
    const extracted = extractVariantFilterColumns(filter, variantConfig)
    filterColumns = extracted.readColumns
    statsColumns = extracted.statsColumns
  } else {
    filterColumns = filter ? extractFilterColumns(filter) : []
    statsColumns = filterColumns
  }

  const outputColumns = columns || allColumns

  // Build required columns set efficiently without intermediate arrays
  const requiredSet = new Set(outputColumns)
  for (const col of filterColumns) {
    if (col) requiredSet.add(col)
  }
  if (orderBy) requiredSet.add(orderBy)
  const requiredColumns = [...requiredSet]

  // Convert filter to predicates that can test min/max statistics
  // For Variant shredding, predicates use the stats column paths
  const predicates = filter ? createVariantPredicates(filter, variantConfig) : new Map()

  // Validate columns exist (only check top-level read columns)
  if (filter) {
    const missingColumns = filterColumns.filter((col) => !allColumns.includes(col))
    if (missingColumns.length) {
      throw new Error(`parquet filter columns not found: ${missingColumns.join(', ')}`)
    }
  }
  if (orderBy && !allColumns.includes(orderBy)) {
    throw new Error(`parquet orderBy column not found: ${orderBy}`)
  }

  /** @type {Record<string, any>[]} */
  const rows = []
  let groupStart = 0

  // Main processing loop
  for (let rgIndex = 0; rgIndex < metadata.row_groups.length; rgIndex++) {
    const rowGroup = metadata.row_groups[rgIndex]
    const groupRows = Number(rowGroup.num_rows)

    // Skip row groups that can't contain matches based on min/max statistics
    if (!canRowGroupMatch(rowGroup, predicates)) {
      groupStart += groupRows
      continue
    }

    // Apply 4MB optimization
    const { size } = getRowGroupFullRange(rowGroup)
    const useSmallGroupOptimization = size < 4 * 1024 * 1024

    /** @type {typeof readSmallRowGroup | typeof readLargeRowGroup} */
    const groupDataFn = useSmallGroupOptimization ? readSmallRowGroup : readLargeRowGroup
    const groupData = await groupDataFn(file, metadata, rgIndex, predicates, requiredColumns, options, groupStart)

    // Apply filter and accumulate rows
    // When filtering, push matching rows directly to avoid intermediate array allocation
    if (filter) {
      for (let i = 0; i < groupData.length; i++) {
        if (matchesFilter(groupData[i], filter)) {
          rows.push(groupData[i])
        }
      }
    } else {
      concat(rows, groupData)
    }

    // Early exit for limited queries without sorting
    if (!orderBy && limit !== undefined && rows.length >= offset + limit) {
      const sliced = rows.slice(offset, offset + limit)
      return columns ? sliced.map((row) => projectRow(row, columns)) : sliced
    }

    groupStart += groupRows
  }

  // Handle sorting
  if (orderBy) {
    if (!filter) {
      // Add stable sort indexes
      rows.forEach((row, idx) => {
        // @ts-ignore
        row.__index__ = idx
      })
    }
    sortRows(rows, orderBy, desc)
  }

  // Final slice and projection
  const sliced = filter || orderBy ? rows.slice(offset, limit ? offset + limit : undefined) : rows
  return columns ? sliced.map((row) => projectRow(row, columns)) : sliced
}

/**
 * Read small row group with buffering optimization
 * @param {AsyncBuffer} file
 * @param {FileMetaData} metadata
 * @param {number} rgIndex
 * @param {Map<string, (min: any, max: any) => boolean>} predicates
 * @param {string[]} columns
 * @param {{parsers?: ParquetParsers, compressors?: Compressors, utf8?: boolean}} options
 * @param {number} groupStart
 * @returns {Promise<object[]>}
 */
export async function readSmallRowGroup(file, metadata, rgIndex, predicates, columns, options, groupStart) {
  const rowGroup = metadata.row_groups[rgIndex]
  const { start, end } = getRowGroupFullRange(rowGroup)

  // Read entire row group at once
  const rgBuffer = await file.slice(start, end)

  // Create buffered file
  /** @type {AsyncBuffer & {sliceAll: (ranges: ([number, number] | null)[]) => Promise<ArrayBuffer[]>}} */
  const bufferedFile = {
    byteLength: file.byteLength,
    /**
     * @param {number} sliceStart
     * @param {number} sliceEnd
     * @returns {Promise<ArrayBuffer>}
     */
    slice(sliceStart, sliceEnd) {
      if (sliceStart >= start && sliceEnd <= end) {
        return Promise.resolve(rgBuffer.slice(sliceStart - start, sliceEnd - start))
      }
      return Promise.resolve(file.slice(sliceStart, sliceEnd))
    },
    /**
     * @param {([number, number] | null)[]} ranges
     * @returns {Promise<ArrayBuffer[]>}
     */
    sliceAll(ranges) {
      const allInBuffer = ranges.every((range) => !range || range[0] >= start && range[1] <= end)
      if (allInBuffer) {
        return Promise.resolve(
          ranges.map((range) => range ? rgBuffer.slice(range[0] - start, range[1] - start) : new ArrayBuffer(0))
        )
      }
      return sliceAll(file, ranges)
    },
  }

  // Use page filtering if available
  const hasIndexes = rowGroup.columns.some((col) => col.column_index_offset)
  if (hasIndexes && predicates.size > 0) {
    return readRowGroupWithPageFilter(bufferedFile, metadata, rgIndex, predicates, columns, options)
  }

  // Otherwise read normally
  return parquetReadObjects({
    ...options,
    file: bufferedFile,
    metadata,
    columns,
    rowStart: groupStart,
    rowEnd: groupStart + Number(rowGroup.num_rows),
  })
}

/**
 * Read large row group without buffering
 * @param {AsyncBuffer} file
 * @param {FileMetaData} metadata
 * @param {number} rgIndex
 * @param {Map<string, (min: any, max: any) => boolean>} predicates
 * @param {string[]} columns
 * @param {{parsers?: ParquetParsers, compressors?: Compressors, utf8?: boolean}} options
 * @param {number} groupStart
 * @returns {Promise<object[]>}
 */
export async function readLargeRowGroup(file, metadata, rgIndex, predicates, columns, options, groupStart) {
  const rowGroup = metadata.row_groups[rgIndex]
  const hasIndexes = rowGroup.columns.some((col) => col.column_index_offset)

  if (hasIndexes && predicates.size > 0) {
    return readRowGroupWithPageFilter(file, metadata, rgIndex, predicates, columns, options)
  }

  return await parquetReadObjects({
    ...options,
    file,
    metadata,
    columns,
    rowStart: groupStart,
    rowEnd: groupStart + Number(rowGroup.num_rows),
  })
}

/**
 * Check if row group can contain matching rows based on statistics.
 * Supports both top-level columns and nested Variant shredded columns.
 *
 * @param {RowGroup} rowGroup
 * @param {Map<string, (min: any, max: any) => boolean>} predicates
 * @returns {boolean}
 */
function canRowGroupMatch(rowGroup, predicates) {
  // Build both top-level and nested column maps
  const columnIndexMap = createColumnIndexMap(rowGroup)
  const nestedIndexMap = createNestedColumnIndexMap(rowGroup)

  for (const [columnPath, predicate] of predicates) {
    // Try nested path first (for Variant shredding)
    let colIndex = nestedIndexMap.get(columnPath)
    // Fall back to top-level column
    if (colIndex === undefined) {
      colIndex = columnIndexMap.get(columnPath)
    }
    if (colIndex === undefined) continue

    const column = rowGroup.columns[colIndex]
    const stats = column?.meta_data?.statistics

    if (stats?.min_value !== undefined && stats?.max_value !== undefined) {
      if (!predicate(stats.min_value, stats.max_value)) {
        return false
      }
    }
  }

  return true
}

/**
 * Read row group with page-level filtering.
 * Supports both top-level and nested Variant shredded columns.
 *
 * @param {AsyncBuffer} file
 * @param {FileMetaData} metadata
 * @param {number} rgIndex
 * @param {Map<string, (min: any, max: any) => boolean>} predicates
 * @param {string[]} columns
 * @param {{parsers?: ParquetParsers, compressors?: Compressors, utf8?: boolean}} options
 * @returns {Promise<object[]>}
 */
export async function readRowGroupWithPageFilter(file, metadata, rgIndex, predicates, columns, options) {
  const rowGroup = metadata.row_groups[rgIndex]
  const columnIndexMap = createColumnIndexMap(rowGroup)
  const nestedIndexMap = createNestedColumnIndexMap(rowGroup)

  // Find pages that might contain matching data
  const selectedPages = await selectPages(file, metadata, rowGroup, predicates, columnIndexMap, nestedIndexMap)
  if (!selectedPages || selectedPages.size === 0) return []

  // Read column data from selected pages
  const columnData = await readSelectedPages(
    file, metadata, rowGroup, columns, selectedPages, columnIndexMap, options
  )

  // Assemble into rows
  return assembleRows(columnData, columns)
}

/**
 * Select pages that might contain matching rows.
 * Supports both top-level and nested Variant shredded columns.
 *
 * @param {AsyncBuffer} file
 * @param {FileMetaData} metadata
 * @param {RowGroup} rowGroup
 * @param {Map<string, (min: any, max: any) => boolean>} predicates
 * @param {Map<string, number>} columnIndexMap
 * @param {Map<string, number>} [nestedIndexMap] - Optional nested column index map for Variant shredding
 * @returns {Promise<Set<number>|null>}
 */
export async function selectPages(file, metadata, rowGroup, predicates, columnIndexMap, nestedIndexMap) {
  const pageSelections = new Map()
  let hasAnyPages = false

  for (const [columnPath, predicate] of predicates) {
    // Try nested path first (for Variant shredding)
    let colIndex = nestedIndexMap?.get(columnPath)
    // Fall back to top-level column
    if (colIndex === undefined) {
      colIndex = columnIndexMap.get(columnPath)
    }
    if (colIndex === undefined) continue

    const column = rowGroup.columns[colIndex]
    if (!column.column_index_offset) continue

    // Read indexes
    const indexes = await readIndexes(file, column, metadata.schema)

    // Find matching pages
    const matchingPages = new Set()
    for (let i = 0; i < indexes.columnIndex.min_values.length; i++) {
      const matches = predicate(indexes.columnIndex.min_values[i], indexes.columnIndex.max_values[i])
      if (matches) {
        matchingPages.add(i)
      }
    }

    if (matchingPages.size > 0) {
      hasAnyPages = true
      pageSelections.set(colIndex, matchingPages)
    }
  }

  if (!hasAnyPages) return null

  // AND semantics: intersect all column selections
  let selectedPages = null
  for (const pages of pageSelections.values()) {
    if (selectedPages === null) {
      selectedPages = new Set(pages)
    } else {
      selectedPages = new Set([...selectedPages].filter((/** @type {number} */ p) => pages.has(p)))
    }
  }

  return selectedPages
}

/**
 * Read indexes for a column
 * @param {AsyncBuffer} file
 * @param {ColumnChunk} column
 * @param {SchemaElement[]} schema
 * @returns {Promise<{columnIndex: ColumnIndex, offsetIndex: OffsetIndex}>}
 */
export async function readIndexes(file, column, schema) {
  /** @type {[number, number][]} */
  const ranges = [
    [Number(column.column_index_offset), Number(column.column_index_offset) + Number(column.column_index_length)],
    [Number(column.offset_index_offset), Number(column.offset_index_offset) + Number(column.offset_index_length)],
  ]

  const [colIndexData, offsetIndexData] = await sliceAll(file, ranges)

  const schemaPath = getSchemaPath(schema, column.meta_data?.path_in_schema || [])
  const element = schemaPath[schemaPath.length - 1]?.element

  return {
    columnIndex: readColumnIndex({ view: new DataView(colIndexData), offset: 0 }, element),
    offsetIndex: readOffsetIndex({ view: new DataView(offsetIndexData), offset: 0 }),
  }
}

/**
 * Read selected pages for columns
 * @param {AsyncBuffer} file
 * @param {FileMetaData} metadata
 * @param {RowGroup} rowGroup
 * @param {string[]} columns
 * @param {Set<number>} selectedPages
 * @param {Map<string, number>} columnIndexMap
 * @param {{parsers?: ParquetParsers, compressors?: Compressors, utf8?: boolean}} options
 * @returns {Promise<Map<string, any[]>>}
 */
export async function readSelectedPages(file, metadata, rowGroup, columns, selectedPages, columnIndexMap, options) {
  const columnData = new Map()

  for (const columnName of columns) {
    const colIndex = columnIndexMap.get(columnName)
    if (colIndex === undefined) continue

    const column = rowGroup.columns[colIndex]
    if (!column.meta_data) continue

    // Read full column if no page selection
    if (!selectedPages || !column.offset_index_offset) {
      const data = await readFullColumn(file, metadata, rowGroup, column, options)
      columnData.set(columnName, data)
      continue
    }

    // Read indexes
    const indexes = await readIndexes(file, column, metadata.schema)
    const selectedPagesList = Array.from(selectedPages).sort((a, b) => a - b)

    // Collect page ranges
    /** @type {[number, number][]} */
    const pageRanges = []
    let needsDictionary = false

    if (column.meta_data.dictionary_page_offset) {
      needsDictionary = true
      pageRanges.push([
        Number(column.meta_data.dictionary_page_offset),
        Number(column.meta_data.data_page_offset),
      ])
    }

    for (const pageIdx of selectedPagesList) {
      const location = indexes.offsetIndex.page_locations[pageIdx]
      pageRanges.push([
        Number(location.offset),
        Number(location.offset) + location.compressed_page_size,
      ])
    }

    // Read all pages
    const pageBuffers = await sliceAll(file, pageRanges)
    const dictionaryBuffer = needsDictionary ? pageBuffers.shift() : null

    // Decode pages
    const schemaPath = getSchemaPath(metadata.schema, column.meta_data.path_in_schema)
    const columnDecoder = {
      columnName: column.meta_data.path_in_schema.join('.'),
      type: column.meta_data.type,
      element: schemaPath[schemaPath.length - 1].element,
      schemaPath,
      codec: column.meta_data.codec,
      parsers: options?.parsers || DEFAULT_PARSERS,
      compressors: options?.compressors,
      utf8: options?.utf8 !== false,
    }

    // Decode dictionary once if present
    const dictionary = dictionaryBuffer
      ? decodeDictionaryPage(dictionaryBuffer, columnDecoder)
      : undefined

    const allValues = []
    for (let i = 0; i < selectedPagesList.length; i++) {
      const pageIdx = selectedPagesList[i]
      const pageBuffer = pageBuffers[i]
      const location = indexes.offsetIndex.page_locations[pageIdx]

      // Calculate page row count
      const pageFirstRow = Number(location.first_row_index)
      const nextLocation = indexes.offsetIndex.page_locations[pageIdx + 1]
      const pageRowCount = nextLocation
        ? Number(nextLocation.first_row_index) - pageFirstRow
        : Number(column.meta_data.num_values) - pageFirstRow

      // Decode the data page
      const pageValues = decodeDataPage(pageBuffer, columnDecoder, dictionary, pageRowCount)

      // Collect values - pageValues should always be an array
      if (Array.isArray(pageValues)) {
        concat(allValues, pageValues)
      } else {
        allValues.push(pageValues)
      }
    }

    columnData.set(columnName, allValues)
  }

  return columnData
}

/**
 * Read full column without page filtering
 * @param {AsyncBuffer} file
 * @param {FileMetaData} metadata
 * @param {RowGroup} rowGroup
 * @param {ColumnChunk} column
 * @param {{parsers?: ParquetParsers, compressors?: Compressors, utf8?: boolean}} options
 * @returns {Promise<DecodedArray>}
 */
export async function readFullColumn(file, metadata, rowGroup, column, options) {
  const start = Number(column.meta_data?.dictionary_page_offset || column.meta_data?.data_page_offset)
  const size = Number(column.meta_data?.total_compressed_size)
  const buffer = await file.slice(start, start + size)

  const schemaPath = getSchemaPath(metadata.schema, column.meta_data?.path_in_schema || [])
  const reader = { view: new DataView(buffer), offset: 0 }

  const values = readColumn(
    reader,
    {
      groupStart: 0,
      selectStart: 0,
      selectEnd: Number(column.meta_data?.num_values),
      groupRows: Number(rowGroup.num_rows),
    },
    {
      columnName: column.meta_data?.path_in_schema?.join('.') || '',
      type: column.meta_data?.type || 'BYTE_ARRAY',
      element: schemaPath[schemaPath.length - 1].element,
      schemaPath,
      codec: column.meta_data?.codec || 'UNCOMPRESSED',
      parsers: options?.parsers || DEFAULT_PARSERS,
      compressors: options?.compressors,
      utf8: options?.utf8 !== false,
    }
  )

  /** @type {DecodedArray} Flattened values if needed */
  const flatValues = []
  if (Array.isArray(values)) {
    for (const chunk of values) {
      concat(flatValues, chunk)
    }
  }
  return flatValues
}

/**
 * Batch read multiple byte ranges
 * @param {AsyncBuffer & {sliceAll?: (ranges: ([number, number] | null)[]) => Promise<ArrayBuffer[]>}} file
 * @param {Array<[number, number]|null>} ranges
 * @returns {Promise<ArrayBuffer[]>}
 */
export function sliceAll(file, ranges) {
  return (
    file.sliceAll?.(ranges) ??
    Promise.all(ranges.map((range) => range ? file.slice(range[0], range[1]) : Promise.resolve(new ArrayBuffer(0))))
  )
}

/**
 * Create predicates for Variant-aware filtering.
 * Maps user filter paths to Parquet stats column paths.
 *
 * For filter { '$index.titleType': 'movie' }, creates predicate for:
 * '$index.typed_value.titleType.typed_value'
 *
 * @param {object} filter - User filter object
 * @param {VariantShredConfig[]} variantConfig - Variant shredding configuration
 * @returns {Map<string, (min: any, max: any) => boolean>}
 */
function createVariantPredicates(filter, variantConfig = []) {
  const predicates = new Map()

  /**
   * @param {any} f - Filter object
   */
  function processFilter(f) {
    if (f.$and) {
      f.$and.forEach(processFilter)
    } else if (f.$or) {
      // OR predicates across different columns can't use statistics effectively
    } else {
      // Process column-level conditions
      for (const [col, cond] of Object.entries(f)) {
        // Skip known operators only (allow $ prefixed column names like $index_category)
        if (FILTER_OPERATORS.has(col)) continue

        // Check for dot-notation (Variant field access)
        const dotIndex = col.indexOf('.')
        if (dotIndex > 0 && variantConfig.length > 0) {
          const columnName = col.slice(0, dotIndex)
          const fieldPath = col.slice(dotIndex + 1)

          // Check if this is a shredded Variant column
          const config = variantConfig.find(c => c.column === columnName)
          if (config) {
            const fieldName = fieldPath.split('.')[0]
            if (config.fields.includes(fieldName)) {
              // Map to Parquet Variant shredding statistics path
              const statsPath = `${columnName}.typed_value.${fieldPath}.typed_value`
              const pred = createRangePredicate(cond)
              if (pred) predicates.set(statsPath, pred)
              continue
            }
          }
        }

        // Regular column
        const pred = createRangePredicate(cond)
        if (pred) predicates.set(col, pred)
      }
    }
  }

  processFilter(filter)
  return predicates
}

/**
 * Create range predicate from condition.
 * Returns a function that tests if a [min,max] range could contain matching values.
 *
 * Supports both MongoDB-style operators ($lt, $gt, etc.) and shorthand (<, >, etc.)
 *
 * @param {any} condition - filter condition (value or operators object)
 * @returns {((min: any, max: any) => boolean)|null} predicate function or null
 */
function createRangePredicate(condition) {
  // Handle direct value comparison
  if (typeof condition !== 'object' || condition === null) {
    return (min, max) => min <= condition && condition <= max
  }

  // Support both MongoDB-style ($lt) and shorthand (<) operators
  const $eq = condition.$eq ?? condition['=']
  const $gt = condition.$gt ?? condition['>']
  const $gte = condition.$gte ?? condition['>=']
  const $lt = condition.$lt ?? condition['<']
  const $lte = condition.$lte ?? condition['<=']
  const $in = condition.$in ?? condition.in

  // Test if statistics range could contain values matching the condition
  return (min, max) => {
    if ($eq !== undefined) {
      return min <= $eq && $eq <= max
    }

    if ($in && Array.isArray($in)) {
      return $in.some((v) => min <= v && v <= max)
    }

    let possible = true

    if ($gt !== undefined) {
      possible = possible && max > $gt
    }
    if ($gte !== undefined) {
      possible = possible && max >= $gte
    }
    if ($lt !== undefined) {
      possible = possible && min < $lt
    }
    if ($lte !== undefined) {
      possible = possible && min <= $lte
    }

    return possible
  }
}

/**
 * Get nested value from row using dot-notation path.
 * Supports paths like '$index.titleType' to access nested objects.
 *
 * @param {{[key: string]: any}} row
 * @param {string} path
 * @returns {any}
 */
function getNestedValue(row, path) {
  const parts = path.split('.')
  let current = row

  for (const part of parts) {
    if (current === null || current === undefined) return undefined
    if (typeof current !== 'object') return undefined
    current = current[part]
  }

  return current
}

/**
 * Assemble column data into rows
 * @param {Map<string, any[]>} columnData
 * @param {string[]} columns
 * @returns {object[]}
 */
export function assembleRows(columnData, columns) {
  if (columnData.size === 0) return []
  const numRows = [...columnData.values()].reduce((max, d) => Math.max(max, d.length), 0)
  const rows = []

  for (let i = 0; i < numRows; i++) {
    /** @type {{[key: string]: any}} */
    const row = {}
    for (const col of columns) {
      const data = columnData.get(col)
      row[col] = data?.[i] ?? null
    }
    rows.push(row)
  }

  return rows
}

/**
 * Check if row matches filter.
 * Supports dot-notation for accessing nested values (e.g., '$index.titleType').
 *
 * @param {{[key: string]: any}} row
 * @param {any} filter
 * @returns {boolean}
 */
export function matchesFilter(row, filter) {
  if (filter.$and) {
    return filter.$and.every((/** @type {any} */ f) => matchesFilter(row, f))
  }

  if (filter.$or) {
    return filter.$or.some((/** @type {any} */ f) => matchesFilter(row, f))
  }

  if (filter.$nor) {
    return !filter.$nor.some((/** @type {any} */ f) => matchesFilter(row, f))
  }

  if (filter.$not) {
    return !matchesFilter(row, filter.$not)
  }

  // Evaluate each column's condition
  for (const [col, cond] of Object.entries(filter)) {
    // Skip known operators only
    if (FILTER_OPERATORS.has(col)) continue

    // Use dot-notation access for nested paths, direct access otherwise
    const value = col.includes('.') ? getNestedValue(row, col) : row[col]
    if (!matchesCondition(value, cond)) {
      return false
    }
  }

  return true
}

/**
 * Check if value matches condition.
 * Supports both MongoDB-style operators ($lt, $gt, etc.) and shorthand (<, >, etc.)
 *
 * @param {any} value
 * @param {any} condition
 * @returns {boolean}
 */
export function matchesCondition(value, condition) {
  // Handle direct value comparison
  if (typeof condition !== 'object' || condition === null || Array.isArray(condition)) {
    return equals(value, condition)
  }

  // All operators must be satisfied
  for (const [op, target] of Object.entries(condition)) {
    switch (op) {
    case '$eq':
    case '=':
      if (!equals(value, target)) return false
      break
    case '$ne':
    case '!=':
      if (equals(value, target)) return false
      break
    case '$gt':
    case '>':
      if (!(value > target)) return false
      break
    case '$gte':
    case '>=':
      if (!(value >= target)) return false
      break
    case '$lt':
    case '<':
      if (!(value < target)) return false
      break
    case '$lte':
    case '<=':
      if (!(value <= target)) return false
      break
    case '$in':
    case 'in':
      if (!Array.isArray(target) || !target.includes(value)) return false
      break
    case '$nin':
      if (!Array.isArray(target) || target.includes(value)) return false
      break
    case '$not':
      if (matchesCondition(value, target)) return false
      break
    }
  }

  return true
}

/**
 * Sort rows by column
 * @param {Record<string, any>[]} rows
 * @param {string} orderBy
 * @param {boolean} desc
 */
export function sortRows(rows, orderBy, desc) {
  rows.sort((a, b) => {
    const aVal = a[orderBy]
    const bVal = b[orderBy]

    if (aVal === bVal) {
      // Use __index__ for stable sort
      const aIndex = a.__index__
      const bIndex = b.__index__
      if (aIndex !== undefined && bIndex !== undefined) {
        return aIndex - bIndex
      }
      return 0
    }
    if (aVal === null || aVal === undefined) return desc ? -1 : 1
    if (bVal === null || bVal === undefined) return desc ? 1 : -1

    const cmp = aVal < bVal ? -1 : 1
    return desc ? -cmp : cmp
  })
}

/**
 * Project row to selected columns
 * @param {{[key: string]: any}} row
 * @param {string[]} columns
 * @returns {{[key: string]: any}}
 */
export function projectRow(row, columns) {
  /** @type {{[key: string]: any}} */
  const projected = {}
  for (const col of columns) {
    projected[col] = row[col]
  }
  return projected
}

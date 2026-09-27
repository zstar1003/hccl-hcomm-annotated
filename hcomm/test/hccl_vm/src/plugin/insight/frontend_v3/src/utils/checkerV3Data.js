/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

import { decode } from '@msgpack/msgpack'
import { normalizeMsId } from './msMemory.js'

const DEFAULT_API_ROOT = ''
const DEFAULT_STAGE_NAME = 'input_graph'

const binaryCache = new Map()
const decodedCache = new Map()

// Returns a plain object or the provided fallback.
function ensureObject(value, fallback = {}) {
  return value && typeof value === 'object' && !Array.isArray(value) ? value : fallback
}

// Returns an array or an empty array fallback.
function ensureArray(value) {
  return Array.isArray(value) ? value : []
}

// Converts finite numeric input to Number with a fallback.
function toNumber(value, fallback = null) {
  return Number.isFinite(value) ? Number(value) : fallback
}

// Converts integer-like values, including bigint loop metadata, to Number.
function toIntegerValue(value, fallback = null) {
  if (typeof value === 'bigint') {
    const numeric = Number(value)
    return Number.isSafeInteger(numeric) ? numeric : fallback
  }
  return Number.isInteger(value) ? Number(value) : fallback
}

// Normalizes ids from string / number / bigint to the string keys used by the UI.
function toIdString(value) {
  if (typeof value === 'string' && value) return value
  if (typeof value === 'number' && Number.isFinite(value)) return String(value)
  if (typeof value === 'bigint') return String(value)
  return ''
}

// Joins the configured API root with a relative API path.
function joinApiPath(apiRoot, path) {
  const normalizedRoot = apiRoot.endsWith('/') ? apiRoot.slice(0, -1) : apiRoot
  return `${normalizedRoot}${path}`
}

// Parses error payloads from the dataset API when possible.
async function readJsonResponse(response) {
  const text = await response.text()
  if (!text) return {}

  try {
    return JSON.parse(text)
  } catch {
    throw new Error(`Failed to parse JSON response from ${response.url}`)
  }
}

// Loads a binary API response and memoizes it by URL.
async function requestArrayBuffer(path, { apiRoot = DEFAULT_API_ROOT, fetchImpl = fetch } = {}) {
  const url = joinApiPath(apiRoot, path)
  if (!binaryCache.has(url)) {
    binaryCache.set(
      url,
      (async () => {
        const response = await fetchImpl(url)
        if (!response.ok) {
          let message = `Request failed: ${response.status}`
          try {
            const payload = await readJsonResponse(response)
            message = payload.error || message
          } catch {
            // Keep the status-only fallback when the response is not JSON.
          }
          throw new Error(message)
        }
        return response.arrayBuffer()
      })(),
    )
  }
  return binaryCache.get(url)
}

// Fetches a dataset file through the shared backend file endpoint.
async function fetchDatasetBinary(datasetName, relativePath, options) {
  const encodedPath = encodeURIComponent(relativePath)
  return requestArrayBuffer(`/api/file?dataset=${encodeURIComponent(datasetName)}&path=${encodedPath}`, options)
}

// Decodes msgpack dataset files and memoizes the decoded payload.
async function fetchDecodedMsgpack(datasetName, relativePath, options) {
  const cacheKey = `${datasetName}:${relativePath}:${options?.apiRoot ?? DEFAULT_API_ROOT}`
  if (!decodedCache.has(cacheKey)) {
    decodedCache.set(
      cacheKey,
      (async () => decode(new Uint8Array(await fetchDatasetBinary(datasetName, relativePath, options))))(),
    )
  }
  return decodedCache.get(cacheKey)
}

// Sort helper that pushes null / undefined / NaN values to the end.
function sortByNullableNumber(left, right) {
  const lhs = Number.isFinite(left) ? Number(left) : Number.POSITIVE_INFINITY
  const rhs = Number.isFinite(right) ? Number(right) : Number.POSITIVE_INFINITY
  if (lhs !== rhs) return lhs - rhs
  return 0
}

// Builds parent/child adjacency maps from the raw edge list.
function buildEdgeMaps(graphDoc) {
  const parentsByNodeId = new Map()
  const childrenByNodeId = new Map()

  for (const edge of ensureArray(graphDoc.edges)) {
    const srcNodeId = toIdString(edge?.src)
    const dstNodeId = toIdString(edge?.dst)
    if (!srcNodeId || !dstNodeId) continue

    const children = childrenByNodeId.get(srcNodeId) ?? []
    children.push(dstNodeId)
    childrenByNodeId.set(srcNodeId, children)

    const parents = parentsByNodeId.get(dstNodeId) ?? []
    parents.push(srcNodeId)
    parentsByNodeId.set(dstNodeId, parents)
  }

  return { parentsByNodeId, childrenByNodeId }
}

// Normalizes a single memory slice into the UI-friendly shape.
function normalizeMemorySlice(entry) {
  const slice = ensureObject(entry, null)
  if (!slice) return null

  const bufferType = typeof slice.mem_type === 'string' ? slice.mem_type : ''
  const rankId = toNumber(slice.rank_id)
  if (!bufferType || !Number.isInteger(rankId)) return null

  const rawOffset = toNumber(slice.offset)
  const offset = Number.isFinite(rawOffset) ? rawOffset : 0
  const msId = normalizeMsId(slice.ms_id ?? slice.msId, {
    bufferType,
    offset: rawOffset,
  })

  const normalizedSlice = {
    rank_id: rankId,
    buffer_type: bufferType,
    offset,
    size: toNumber(slice.len, 0),
  }
  if (Number.isInteger(msId)) normalizedSlice.ms_id = msId
  return normalizedSlice
}

// Normalizes and filters a flat memory-slice list.
function normalizeMemorySliceArray(entries) {
  return ensureArray(entries).map(normalizeMemorySlice).filter(Boolean)
}

// Normalizes nested memory-slice groups.
function normalizeMemorySliceGroups(entries) {
  return ensureArray(entries)
    .map((group) => normalizeMemorySliceArray(group))
    .filter((group) => group.length)
}

// Flattens grouped reads into the list view expected by the inspector.
function flattenMemorySliceGroups(groups) {
  return groups.flatMap((group) => group)
}

// Converts raw checker task payloads into the compact task_data shape used across Insight V3.
function buildTaskData(rawNode, lane) {
  const taskType = String(rawNode?.task_type ?? '').toUpperCase()
  const srcSlice = normalizeMemorySlice(rawNode?.src_slice)
  const dstSlice = normalizeMemorySlice(rawNode?.dst_slice)
  const srcSlices = normalizeMemorySliceArray(rawNode?.src_slices)
  const dstSlices = normalizeMemorySliceArray(rawNode?.dst_slices)
  const srcGroups = normalizeMemorySliceGroups(rawNode?.src_groups)
  const mergedSrcSlices = normalizeMemorySliceArray(rawNode?.merged_src_slices)
  const mergedDstSlices = normalizeMemorySliceArray(rawNode?.merged_dst_slices)
  const mergedSrcGroups = normalizeMemorySliceGroups(rawNode?.merged_src_groups)
  const mergedGroupReads = flattenMemorySliceGroups(mergedSrcGroups)
  const taskData = {}

  const notifyId = toNumber(rawNode?.notify?.notify_id)
  if (Number.isInteger(notifyId)) taskData.notify_id = notifyId
  if (typeof rawNode?.notify?.kind === 'string' && rawNode.notify.kind) {
    taskData.notify_kind = rawNode.notify.kind
  }
  if (typeof rawNode?.notify_role === 'string' && rawNode.notify_role) {
    taskData.notify_role = rawNode.notify_role
  }
  if (typeof rawNode?.protocol_type === 'string' && rawNode.protocol_type) {
    taskData.protocol_type = rawNode.protocol_type
  }
  if (Number.isInteger(toNumber(rawNode?.reduce_op))) {
    taskData.reduce_op = toNumber(rawNode.reduce_op)
  }
  if (Number.isInteger(toNumber(rawNode?.data_type))) {
    taskData.data_type = toNumber(rawNode.data_type)
  }
  if (rawNode?.ccu_trace && typeof rawNode.ccu_trace === 'object') {
    taskData.ccu_trace = rawNode.ccu_trace
  }

  const rankId = toNumber(lane?.rank_id)
  const queueId = toNumber(lane?.queue_id)
  const streamId = toNumber(lane?.stream_id)
  if (Number.isInteger(rankId)) taskData.rank_id = rankId
  if (Number.isInteger(queueId)) taskData.queue_id = queueId
  if (Number.isInteger(streamId)) taskData.stream_id = streamId

  if (taskType === 'TRANS_MEM') {
    if (srcSlice) taskData.src_slice = srcSlice
    if (dstSlice) taskData.dst_slice = dstSlice
    if (srcSlices.length) taskData.src_slices = srcSlices
    if (dstSlices.length) taskData.dst_slices = dstSlices
  } else if (taskType === 'REDUCE') {
    if (srcSlice) taskData.src_slice = srcSlice
    if (srcSlices.length) taskData.src_slices = srcSlices
    if (dstSlice) taskData.dst_slice = dstSlice
  } else if (taskType === 'BATCH_TRANS_MEM') {
    const displayReads = mergedSrcSlices.length ? mergedSrcSlices : srcSlices
    const displayWrites = mergedDstSlices.length ? mergedDstSlices : dstSlices
    if (displayReads.length) taskData.src_slices = displayReads
    if (displayWrites.length) taskData.dst_slices = displayWrites
    taskData.pair_count = Math.min(displayReads.length, displayWrites.length)
  } else if (taskType === 'BATCH_REDUCE') {
    const displayReads = mergedGroupReads.length ? mergedGroupReads : flattenMemorySliceGroups(srcGroups)
    const displayWrites = mergedDstSlices.length ? mergedDstSlices : dstSlices
    if (displayReads.length) taskData.src_slices = displayReads
    if (displayWrites.length) taskData.dst_slices = displayWrites
    taskData.group_count = displayWrites.length
  } else {
    if (srcSlice) taskData.src_slice = srcSlice
    if (dstSlice) taskData.dst_slice = dstSlice
    if (srcSlices.length) taskData.src_slices = srcSlices
    if (dstSlices.length) taskData.dst_slices = dstSlices
  }

  return taskData
}

// Generates the short task subtitle shown in the DAG and inspector cards.
function buildTaskSubtitle(taskType, taskData) {
  if (taskData?.src_slice?.buffer_type && taskData?.dst_slice?.buffer_type) {
    return `${taskData.src_slice.buffer_type} -> ${taskData.dst_slice.buffer_type}`
  }
  if (Array.isArray(taskData?.src_slices) && taskData?.dst_slice?.buffer_type) {
    return `${taskData.src_slices.length} inputs -> ${taskData.dst_slice.buffer_type}`
  }
  if (Array.isArray(taskData?.src_slices) && Array.isArray(taskData?.dst_slices) && taskData.dst_slices.length) {
    return `${taskData.src_slices.length} src / ${taskData.dst_slices.length} dst`
  }
  if (Number.isInteger(taskData?.notify_id)) {
    return `notify ${taskData.notify_id}`
  }
  return String(taskType || 'TASK')
}

// Normalizes task labels for display while keeping the original taskType for logic and filtering.
function formatDisplayTaskLabel(taskType) {
  return String(taskType ?? 'TASK')
    .replace(/^BATCH_/i, '')
    .replaceAll('_', ' ')
    .trim() || 'TASK'
}

// Preserves the raw checker node payload for drill-down and debug views.
function buildRawTaskNode(rawNode, parents, children, taskData) {
  return {
    ...rawNode,
    parents,
    children,
    task: {
      task_type: rawNode?.task_type ?? 'TASK',
      task_data: taskData,
    },
  }
}

// Resolves loop boundary role from trace metadata first, then from structural fallback hints.
function resolveLoopRole(rawNode, trace) {
  const traceRole = typeof trace?.node_role === 'string' ? trace.node_role.trim().toLowerCase() : ''
  if (traceRole === 'loop_start') return 'start'
  if (traceRole === 'loop_end') return 'end'

  const boundaryType = String(rawNode?.boundary_type ?? '').trim().toUpperCase()
  const taskType = String(rawNode?.task_type ?? '').trim().toUpperCase()
  if (boundaryType !== 'LOOP') return ''
  if (taskType === 'START') return 'start'
  if (taskType === 'END') return 'end'
  return ''
}

// Extracts loop-specific fields from ccu_trace so later stages do not need raw-trace branching.
function buildLoopNodeFields(rawNode, nodeId, taskData) {
  const trace = ensureObject(taskData?.ccu_trace, null)
  const loopRole = resolveLoopRole(rawNode, trace)
  const parentLoopStartNodeId = toIdString(trace?.loop_start_node_id)
  const ownLoopStartNodeId = loopRole === 'start' && nodeId ? nodeId : ''
  const loopEndNodeId = loopRole === 'start' ? toIdString(trace?.loop_end_node_id) : ''
  const instrIdStart = toIntegerValue(trace?.loop_instr_id_start)
  const instrIdEnd = toIntegerValue(trace?.loop_instr_id_end)
  const loopCnt = toIntegerValue(trace?.loop_cnt)
  const loopExpandCnt = toIntegerValue(trace?.loop_expand_cnt)

  return {
    loopRole,
    ownLoopStartNodeId,
    parentLoopStartNodeId:
      parentLoopStartNodeId && parentLoopStartNodeId !== ownLoopStartNodeId ? parentLoopStartNodeId : '',
    loopEndNodeId,
    loopInstrRange:
      Number.isInteger(instrIdStart) && Number.isInteger(instrIdEnd)
        ? {
            start: instrIdStart,
            end: instrIdEnd,
            label: `I${instrIdStart}-${instrIdEnd}`,
          }
        : null,
    loopCnt,
    loopExpandCnt,
  }
}

// Normalizes one raw graph node into the task-node model consumed by DAG, inspector, and overlays.
function normalizeCheckerV3TaskNode(rawNode, lane, nodeLayout, parents, children, stageName, laneRowIndex) {
  const taskType = typeof rawNode?.task_type === 'string' ? rawNode.task_type : 'TASK'
  const taskData = buildTaskData(rawNode, lane)
  const nodeId = toIdString(rawNode?.id)
  const slotIndex = toNumber(nodeLayout?.col, toNumber(rawNode?.stream_seq, 0))
  const semanticStep = toNumber(rawNode?.semantic_step)
  const loopNodeFields = buildLoopNodeFields(rawNode, nodeId, taskData)

  return {
    id: nodeId,
    actualNodeId: nodeId,
    lookupIds: nodeId ? [nodeId] : [],
    mappingKey: nodeId,
    nodeMappings: [],
    stageName,
    label: formatDisplayTaskLabel(taskType),
    subtitle: buildTaskSubtitle(taskType, taskData),
    taskType,
    taskData,
    notifyId: taskData.notify_id ?? null,
    rankId: toNumber(lane?.rank_id),
    streamId: toNumber(lane?.stream_id, toNumber(rawNode?.stream_id, 0)),
    queueId: toNumber(lane?.queue_id, toNumber(rawNode?.queue_id, 0)),
    orderIndex: toNumber(rawNode?.stream_seq, 0),
    slotIndex,
    globalStep: semanticStep,
    localStep: semanticStep,
    pos: slotIndex,
    posInfo: '',
    hasSubgraph: false,
    parents,
    children,
    laneId: toIdString(rawNode?.lane_id),
    logicalRow: Number.isInteger(laneRowIndex) ? laneRowIndex : null,
    finalCol: slotIndex,
    streamSeq: toNumber(rawNode?.stream_seq, 0),
    ...loopNodeFields,
    innermostLoopStartNodeId: '',
    loopChainStartNodeIds: [],
    loopContexts: [],
    ownLoopGroup: null,
    loopSummary: null,
    rawNode: buildRawTaskNode(rawNode, parents, children, taskData),
  }
}

// Narrows a loop group to the subset of fields that inspector and overlay code actually need.
function summarizeLoopGroup(loopGroup) {
  if (!loopGroup || typeof loopGroup !== 'object') return null
  return {
    startNodeId: loopGroup.startNodeId,
    endNodeId: loopGroup.endNodeId,
    parentLoopStartNodeId: loopGroup.parentLoopStartNodeId,
    depth: loopGroup.depth,
    instrIdStart: loopGroup.instrIdStart,
    instrIdEnd: loopGroup.instrIdEnd,
    instrRangeLabel: loopGroup.instrRangeLabel,
    loopCnt: loopGroup.loopCnt,
    loopExpandCnt: loopGroup.loopExpandCnt,
    nodeCount: loopGroup.nodeCount,
    rankId: loopGroup.rankId,
    streamId: loopGroup.streamId,
    queueId: loopGroup.queueId,
    laneId: loopGroup.laneId,
  }
}

// Builds explicit loop groups from normalized task nodes.
// Call chain: buildCheckerV3DagScene -> buildLoopGroups -> scene.loopGroups / node.loopContexts
// -> MemViewDagSigmaCanvas overlay + MemViewInspectorPanel loop sections.
function buildLoopGroups(taskNodes) {
  const nodeById = new Map(taskNodes.map((node) => [node.id, node]))
  const nodeOrderIndexById = new Map(taskNodes.map((node, index) => [node.id, index]))
  const loopStartNodes = taskNodes.filter((node) => node.loopRole === 'start' && node.ownLoopStartNodeId)
  const loopStartParentById = new Map(
    loopStartNodes.map((node) => [node.ownLoopStartNodeId, node.parentLoopStartNodeId || '']),
  )
  const loopChainCache = new Map()

  // Resolves the nesting chain from the innermost loop start to its ancestors.
  function resolveLoopChain(startNodeId, visited = new Set()) {
    if (!startNodeId) return []
    if (loopChainCache.has(startNodeId)) return loopChainCache.get(startNodeId)
    if (visited.has(startNodeId)) return [startNodeId]

    visited.add(startNodeId)
    const parentStartNodeId = loopStartParentById.get(startNodeId) ?? ''
    const chain = [startNodeId]
    if (parentStartNodeId) {
      chain.push(...resolveLoopChain(parentStartNodeId, visited))
    }
    loopChainCache.set(startNodeId, chain)
    return chain
  }

  for (const node of taskNodes) {
    const innermostLoopStartNodeId =
      node.loopRole === 'start' && node.ownLoopStartNodeId ? node.ownLoopStartNodeId : node.parentLoopStartNodeId
    node.innermostLoopStartNodeId = innermostLoopStartNodeId || ''
    node.loopChainStartNodeIds = innermostLoopStartNodeId ? resolveLoopChain(innermostLoopStartNodeId) : []
  }

  const loopGroups = []
  const loopGroupByStartNodeId = new Map()
  for (const startNode of loopStartNodes) {
    const endNodeId = startNode.loopEndNodeId
    const endNode = endNodeId ? nodeById.get(endNodeId) ?? null : null
    if (!endNode || !startNode.loopInstrRange) continue

    const loopChain = resolveLoopChain(startNode.id)
    const group = {
      startNodeId: startNode.id,
      endNodeId,
      memberNodeIds: [],
      bodyNodeIds: [],
      parentLoopStartNodeId: loopStartParentById.get(startNode.id) ?? '',
      depth: Math.max(0, loopChain.length - 1),
      instrIdStart: startNode.loopInstrRange.start,
      instrIdEnd: startNode.loopInstrRange.end,
      instrRangeLabel: startNode.loopInstrRange.label,
      loopCnt: Number.isInteger(startNode.loopCnt) ? startNode.loopCnt : null,
      loopExpandCnt: Number.isInteger(startNode.loopExpandCnt) ? startNode.loopExpandCnt : null,
      nodeCount: 0,
      laneId: startNode.laneId,
      rankId: startNode.rankId,
      streamId: startNode.streamId,
      queueId: startNode.queueId,
      _memberNodeIdSet: new Set([startNode.id, endNodeId]),
      _bodyNodeIdSet: new Set(),
    }
    loopGroups.push(group)
    loopGroupByStartNodeId.set(startNode.id, group)
  }

  for (const node of taskNodes) {
    for (const startNodeId of node.loopChainStartNodeIds) {
      const loopGroup = loopGroupByStartNodeId.get(startNodeId)
      if (!loopGroup) continue
      loopGroup._memberNodeIdSet.add(node.id)
      if (node.id !== loopGroup.startNodeId && node.id !== loopGroup.endNodeId) {
        loopGroup._bodyNodeIdSet.add(node.id)
      }
    }
  }

  // Keeps members stable by the rendered task order instead of raw insertion order.
  const compareNodeOrder = (leftId, rightId) =>
    sortByNullableNumber(nodeOrderIndexById.get(leftId), nodeOrderIndexById.get(rightId)) || leftId.localeCompare(rightId)

  for (const loopGroup of loopGroups) {
    loopGroup.memberNodeIds = [...loopGroup._memberNodeIdSet].sort(compareNodeOrder)
    loopGroup.bodyNodeIds = [...loopGroup._bodyNodeIdSet].sort(compareNodeOrder)
    loopGroup.nodeCount = loopGroup.memberNodeIds.length
    delete loopGroup._memberNodeIdSet
    delete loopGroup._bodyNodeIdSet
  }

  const loopSummaryByStartNodeId = new Map(
    loopGroups.map((loopGroup) => [loopGroup.startNodeId, summarizeLoopGroup(loopGroup)]),
  )
  for (const node of taskNodes) {
    node.loopContexts = node.loopChainStartNodeIds
      .map((startNodeId) => loopSummaryByStartNodeId.get(startNodeId) ?? null)
      .filter(Boolean)
    node.ownLoopGroup = node.loopRole === 'start' ? loopSummaryByStartNodeId.get(node.id) ?? null : null
    node.loopSummary = node.ownLoopGroup ?? node.loopContexts[0] ?? null
  }

  return {
    loopGroups: loopGroups.sort(
      (left, right) =>
        sortByNullableNumber(nodeOrderIndexById.get(left.startNodeId), nodeOrderIndexById.get(right.startNodeId)) ||
        left.startNodeId.localeCompare(right.startNodeId),
    ),
    loopGroupByStartNodeId,
  }
}

// Selects and orders the lanes currently visible in the DAG.
function buildLaneSelection(graphDoc, layoutDoc, selectedRankIds) {
  const selectedRankSet = new Set((selectedRankIds ?? []).filter((rankId) => Number.isInteger(rankId)))
  const laneById = new Map(
    ensureArray(graphDoc.lanes).map((lane) => [toIdString(lane?.id), ensureObject(lane)]),
  )

  const laneRows = ensureArray(layoutDoc.lane_layout)
    .map((item) => {
      const layout = ensureObject(item)
      const laneId = toIdString(layout.lane_id)
      return {
        layout,
        lane: laneById.get(laneId) ?? {},
      }
    })
    .filter(({ lane }) => {
      const rankId = toNumber(lane?.rank_id)
      return selectedRankSet.has(rankId)
    })
    .sort((left, right) => {
      const byRankOrder = sortByNullableNumber(left.layout.rank_order, right.layout.rank_order)
      if (byRankOrder) return byRankOrder
      const byRow = sortByNullableNumber(left.layout.row, right.layout.row)
      if (byRow) return byRow
      const byStreamOrder = sortByNullableNumber(left.layout.stream_order, right.layout.stream_order)
      if (byStreamOrder) return byStreamOrder
      return toIdString(left.lane?.id).localeCompare(toIdString(right.lane?.id))
    })

  const laneOrder = []
  const laneRowIndexById = new Map()
  laneRows.forEach(({ lane }, index) => {
    const laneId = toIdString(lane?.id)
    if (!laneId) return
    const rankId = toNumber(lane?.rank_id)
    const streamId = toNumber(lane?.stream_id, 0)
    const queueId = toNumber(lane?.queue_id, 0)
    laneRowIndexById.set(laneId, index)
    laneOrder.push({
      laneId,
      rowIndex: index,
      rankId,
      streamId,
      queueId,
      label: `Rank ${Number.isInteger(rankId) ? rankId : '--'} / Stream ${streamId} / Queue ${queueId}`,
    })
  })

  return { laneOrder, laneRowIndexById }
}

// Resolves the full set of rank ids available to the current graph.
function resolveAvailableRankIds(graphDoc, laneOrder) {
  const graphRanks = ensureArray(graphDoc.ranks)
    .map((rankId) => toNumber(rankId))
    .filter((rankId) => Number.isInteger(rankId))
  if (graphRanks.length) {
    return [...new Set(graphRanks)].sort((left, right) => left - right)
  }

  const laneRanks = laneOrder
    .map((lane) => lane.rankId)
    .filter((rankId) => Number.isInteger(rankId))
  return [...new Set(laneRanks)].sort((left, right) => left - right)
}

// Computes the semantic-step upper bound used by the step scrubber.
function resolveTotalSteps(graphDoc, taskNodes) {
  const statsStepCount = toNumber(graphDoc?.stats?.semantic_step_count)
  const maxNodeStep = taskNodes.reduce((max, node) => {
    if (!Number.isInteger(node?.globalStep)) return max
    return Math.max(max, node.globalStep)
  }, 0)

  return Math.max(1, (Number.isInteger(statsStepCount) ? statsStepCount + 1 : 0), maxNodeStep + 1)
}

// Builds the full Checker V3 DAG scene model from graph/layout datasets.
// Call chain: page loader -> buildCheckerV3DagScene -> MemViewDagSigmaCanvas / MemViewInspectorPanel props.
export async function buildCheckerV3DagScene(datasetName, selectedRankIds, options = {}) {
  const stageName =
    typeof options?.stageName === 'string' && options.stageName ? options.stageName : DEFAULT_STAGE_NAME
  const requestOptions = {
    apiRoot: options?.apiRoot,
    fetchImpl: options?.fetchImpl,
  }

  const [graphDocRaw, layoutDocRaw] = await Promise.all([
    fetchDecodedMsgpack(datasetName, 'graph/graph.msgpack', requestOptions),
    fetchDecodedMsgpack(datasetName, 'graph/layout.msgpack', requestOptions),
  ])

  const graphDoc = ensureObject(graphDocRaw)
  const layoutDoc = ensureObject(layoutDocRaw)
  const laneById = new Map(
    ensureArray(graphDoc.lanes).map((lane) => [toIdString(lane?.id), ensureObject(lane)]),
  )
  const nodeLayoutById = new Map(
    ensureArray(layoutDoc.node_layout).map((item) => [toIdString(item?.node_id), ensureObject(item)]),
  )
  const { parentsByNodeId, childrenByNodeId } = buildEdgeMaps(graphDoc)
  const { laneOrder, laneRowIndexById } = buildLaneSelection(graphDoc, layoutDoc, selectedRankIds)
  const selectedLaneIds = new Set(laneOrder.map((lane) => lane.laneId))

  const taskNodes = ensureArray(graphDoc.nodes)
    .filter((rawNode) => selectedLaneIds.has(toIdString(rawNode?.lane_id)))
    .map((rawNode) => {
      const nodeId = toIdString(rawNode?.id)
      const laneId = toIdString(rawNode?.lane_id)
      const lane = laneById.get(laneId) ?? {}
      const nodeLayout = nodeLayoutById.get(nodeId) ?? {}
      const parents = parentsByNodeId.get(nodeId) ?? []
      const children = childrenByNodeId.get(nodeId) ?? []
      const laneRowIndex = laneRowIndexById.get(laneId)
      return normalizeCheckerV3TaskNode(rawNode, lane, nodeLayout, parents, children, stageName, laneRowIndex)
    })
    .filter((node) => node.id)
    .sort((left, right) => {
      const byRow = sortByNullableNumber(laneRowIndexById.get(left.laneId), laneRowIndexById.get(right.laneId))
      if (byRow) return byRow
      const byCol = sortByNullableNumber(left.finalCol, right.finalCol)
      if (byCol) return byCol
      return left.id.localeCompare(right.id)
    })

  const { loopGroups, loopGroupByStartNodeId } = buildLoopGroups(taskNodes)
  const nodeById = new Map(taskNodes.map((node) => [node.id, node]))
  const edgeRecords = ensureArray(graphDoc.edges)
    .map((edge) => ({
      edgeId: toIdString(edge?.id),
      srcNodeId: toIdString(edge?.src),
      dstNodeId: toIdString(edge?.dst),
      edgeKind: typeof edge?.kind === 'string' ? edge.kind : 'dependency',
      isPrimaryAlignEdge: Boolean(edge?.primary_align),
    }))
    .filter((edge) => nodeById.has(edge.srcNodeId) && nodeById.has(edge.dstNodeId))

  return {
    stageName,
    graphStageName: typeof graphDoc?.stage === 'string' ? graphDoc.stage : '',
    totalSteps: resolveTotalSteps(graphDoc, taskNodes),
    availableRankIds: resolveAvailableRankIds(graphDoc, laneOrder),
    laneOrder,
    taskNodes,
    loopGroups,
    loopGroupByStartNodeId,
    edgeRecords,
    nodeById,
  }
}

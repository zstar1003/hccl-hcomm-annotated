<!--
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
-->

<script setup>
import { computed, nextTick, ref, shallowRef, triggerRef, watch } from 'vue'
import {
  RefreshRight,
  Close,
  ZoomIn,
  ZoomOut,
} from '@element-plus/icons-vue'
import { fetchMsgpack, parseGraphIndex, parseGraphRank, parseMemoryManifest } from '../../utils/insightData'
import { taskOpsAtStep } from '../../utils/memviewIndex.js'
import { buildCenteredFocusStreamWindow } from '../../utils/memviewDagFocus.js'
import {
  DEFAULT_MEMVIEW_STAGE,
  normalizeStreamCollection,
  normalizeTopLevelStreams,
} from '../../utils/memviewDag.js'
import { useMemViewTimeline } from '../../composables/useMemViewTimeline'
import MemViewDagPixiCanvas from './MemViewDagPixiCanvas.vue'

const STREAM_ROW_HEIGHT = 42
const STREAM_GAP = 3
const STREAM_LABEL_WIDTH = 150
const STREAM_CONTENT_GAP = 10
const NODE_WIDTH = 74
const TRACK_PADDING_X = 6
const TRACK_PADDING_Y = 4
const GRAPH_SIDE_PADDING = 18
const GRAPH_OUTER_PADDING_X = 40
const GRAPH_OUTER_PADDING_Y = 180
const NODE_HEIGHT = STREAM_ROW_HEIGHT - TRACK_PADDING_Y * 2
const NODE_GAP = 24
const TRACK_OFFSET_X = STREAM_LABEL_WIDTH + STREAM_CONTENT_GAP
const EDGE_VERTICAL_PADDING = 12
const MIN_ZOOM = 0.75
const MAX_ZOOM = 1.6
const ZOOM_STEP = 0.15
const FOCUS_WINDOW_RADIUS = 10

const props = defineProps({
  datasetName: {
    type: String,
    default: '',
  },
  graphGroups: {
    type: Array,
    default: () => [],
  },
  stageName: {
    type: String,
    default: DEFAULT_MEMVIEW_STAGE,
  },
  selectedRankKeys: {
    type: Array,
    required: true,
  },
  currentStep: {
    type: Number,
    default: 0,
  },
  totalSteps: {
    type: Number,
    default: 0,
  },
  displayStepItems: {
    type: Array,
    default: () => [],
  },
  selectedNodeId: {
    type: String,
    default: '',
  },
  focusStep: {
    type: Number,
    default: null,
  },
  issueNodeIds: {
    type: Object,
    default: null,
  },
  subgraphNode: {
    type: Object,
    default: null,
  },
  emptyText: {
    type: String,
    default: 'No data',
  },
})

const emit = defineEmits([
  'node-select',
  'update:stage-name',
  'update:currentStep',
  'update:totalSteps',
  'nodes-loaded',
  'graph-node-ids-change',
  'close-focus-view',
])

const { indexes, nodeRegistry } = useMemViewTimeline()
const loadingIndex = ref(false)
const loadingRanks = ref(false)
const errorMessage = ref('')
const graphIndex = ref(null)
const rankGraphStates = shallowRef([])
const playbackTotalSteps = ref(0)
const zoomScale = ref(1)

const rankRequestCache = new Map()

const subgraphLayout = computed(() => {
  const node = props.subgraphNode
  if (!node?.rawNode?.task?.task_data?.sub_graph) return null
  const streams = normalizeStreamCollection(node.rawNode.task.task_data.sub_graph, node.rankId, {
    stageName: resolvedStageName.value || props.stageName || DEFAULT_MEMVIEW_STAGE,
    topLevel: false,
  })
  const graphLayout = buildGraphLayout(streams)
  return {
    streams,
    graphLayout,
    nodeCount: streams.reduce((sum, stream) => sum + stream.nodes.length, 0),
  }
})

const selectedRankIds = computed(() =>
  props.selectedRankKeys
    .map((key) => {
      const match = String(key).match(/^rank-(\d+)$/)
      return match ? Number(match[1]) : null
    })
    .filter((rankId) => Number.isInteger(rankId))
    .sort((left, right) => left - right),
)

const stageOptions = computed(() =>
  props.graphGroups.map((group) => ({
    label: group.name,
    value: group.name,
  })),
)

const resolvedStageName = computed(() => {
  const options = stageOptions.value
  if (!options.length) return ''
  if (options.some((option) => option.value === props.stageName)) {
    return props.stageName
  }
  if (options.some((option) => option.value === DEFAULT_MEMVIEW_STAGE)) {
    return DEFAULT_MEMVIEW_STAGE
  }
  return options[0]?.value ?? ''
})

const activeRankIds = computed(() => {
  const available = new Set(graphIndex.value?.ranks ?? [])
  return selectedRankIds.value.filter((rankId) => available.has(rankId))
})

const unifiedGraphState = computed(() => {
  if (subgraphLayout.value) {
    const { streams, graphLayout, nodeCount } = subgraphLayout.value
    return {
      mode: 'subgraph',
      streams,
      nodeCount,
      graphLayout,
    }
  }

  const loadedStates = rankGraphStates.value
    .filter((state) => state?.loaded)
    .sort((left, right) => left.rankId - right.rankId)
  const streams = loadedStates.flatMap((state) =>
    state.streams.map((stream) => ({
      ...stream,
      rankId: state.rankId,
      streamKey: `rank_${state.rankId}:stream_${stream.streamId}`,
      label: `Rank ${state.rankId} / Stream ${stream.streamId}`,
      nodes: stream.nodes.map((node) => ({
        ...node,
        rankId: Number.isInteger(node.rankId) ? node.rankId : state.rankId,
        streamKey: `rank_${state.rankId}:stream_${stream.streamId}`,
      })),
    })),
  )
  const graphLayout = buildGraphLayout(streams)

  return {
    mode: 'root',
    streams,
    nodeCount: streams.reduce((sum, stream) => sum + stream.nodes.length, 0),
    graphLayout,
  }
})

const unifiedNodeLookup = computed(() => {
  const lookup = new Map()
  for (const node of unifiedGraphState.value?.graphLayout?.nodesFlat ?? []) {
    if (!node?.id || node.isPlaceholder || lookup.has(node.id)) continue
    lookup.set(node.id, node)
  }
  return lookup
})

const focusCandidateNodes = computed(() => {
  if (!Number.isInteger(props.focusStep) || !indexes.value) return []

  const candidates = []
  const seen = new Set()
  for (const op of taskOpsAtStep(indexes.value, props.focusStep)) {
    const taskNodeId = typeof op?.taskNodeId === 'string' ? op.taskNodeId : ''
    if (!taskNodeId || seen.has(taskNodeId)) continue
    seen.add(taskNodeId)
    const node = nodeRegistry.value?.get(taskNodeId) ?? unifiedNodeLookup.value.get(taskNodeId)
    if (node && !node.isPlaceholder) {
      candidates.push(node)
    }
  }

  return candidates.sort((left, right) => {
    const leftRank = Number.isInteger(left?.rankId) ? left.rankId : Number.POSITIVE_INFINITY
    const rightRank = Number.isInteger(right?.rankId) ? right.rankId : Number.POSITIVE_INFINITY
    if (leftRank !== rightRank) return leftRank - rightRank

    const leftStream = Number.isInteger(left?.streamId) ? left.streamId : Number.POSITIVE_INFINITY
    const rightStream = Number.isInteger(right?.streamId) ? right.streamId : Number.POSITIVE_INFINITY
    if (leftStream !== rightStream) return leftStream - rightStream

    const leftOrder = Number.isFinite(left?.layoutIndex) ? left.layoutIndex : unifiedNodeLookup.value.get(left.id)?.layoutIndex
    const rightOrder = Number.isFinite(right?.layoutIndex) ? right.layoutIndex : unifiedNodeLookup.value.get(right.id)?.layoutIndex
    const leftIndex = Number.isFinite(leftOrder) ? leftOrder : Number.POSITIVE_INFINITY
    const rightIndex = Number.isFinite(rightOrder) ? rightOrder : Number.POSITIVE_INFINITY
    if (leftIndex !== rightIndex) return leftIndex - rightIndex

    return String(left.id).localeCompare(String(right.id))
  })
})

const focusGraphState = computed(() => {
  if (!Number.isInteger(props.focusStep)) return null
  const candidates = focusCandidateNodes.value
  if (!candidates.length) return null

  const streams = unifiedGraphState.value?.streams ?? []
  const focusStreams = []
  const focusAnchorNodeIds = []
  const seenTaskIds = new Set()

  candidates.forEach((candidate, index) => {
    if (!candidate?.id || seenTaskIds.has(candidate.id)) return
    seenTaskIds.add(candidate.id)

    const sourceStream = streams.find((stream) => stream.streamKey === candidate.streamKey)
      ?? streams.find((stream) => stream.rankId === candidate.rankId && stream.streamId === candidate.streamId)
      ?? null
    if (!sourceStream) return

    const focusWindowKey = `focus_${props.focusStep}_${index}_${candidate.id}`
    const focusStream = buildCenteredFocusStreamWindow(sourceStream, candidate.id, FOCUS_WINDOW_RADIUS, {
      focusWindowKey,
    })
    if (!focusStream) return

    focusStreams.push(focusStream)
    if (focusStream.focusAnchorNodeId) {
      focusAnchorNodeIds.push(focusStream.focusAnchorNodeId)
    }
  })

  if (!focusStreams.length) return null

  const graphLayout = buildGraphLayout(focusStreams)

  return {
    mode: 'focus',
    streams: focusStreams,
    nodeCount: focusStreams.reduce(
      (sum, stream) => sum + stream.nodes.filter((node) => !node?.isPlaceholder).length,
      0,
    ),
    graphLayout,
    focusAnchorNodeIds,
  }
})

const renderGraphState = computed(() => focusGraphState.value ?? unifiedGraphState.value)
const isFocusView = computed(() => Number.isInteger(props.focusStep))
const focusAnchorNodeIds = computed(() => focusGraphState.value?.focusAnchorNodeIds ?? [])

const currentGraphNodeIds = computed(() => {
  const ids = []
  const seen = new Set()
  for (const node of unifiedGraphState.value?.graphLayout?.nodesFlat ?? []) {
    if (node?.isPlaceholder) continue
    const lookupIds = Array.isArray(node?.lookupIds) && node.lookupIds.length ? node.lookupIds : [node?.id]
    for (const lookupId of lookupIds) {
      if (!lookupId || seen.has(lookupId)) continue
      seen.add(lookupId)
      ids.push(lookupId)
    }
  }
  return ids
})

const hasContent = computed(() => (renderGraphState.value?.nodeCount ?? 0) > 0)

const effectiveTotalSteps = computed(() => {
  if (Number.isFinite(props.totalSteps) && props.totalSteps > 0) return props.totalSteps
  return playbackTotalSteps.value
})

const highlightedStep = computed(() => Math.min(Math.max(props.currentStep, 0), Math.max(0, effectiveTotalSteps.value - 1)))

const stepHighlightedNodeIds = computed(() => {
  if (!indexes.value || !effectiveTotalSteps.value) return null
  const highlightIds = new Set()

  for (const op of taskOpsAtStep(indexes.value, highlightedStep.value)) {
    const taskNodeId = typeof op?.taskNodeId === 'string' ? op.taskNodeId : ''
    if (!taskNodeId) continue
    highlightIds.add(taskNodeId)
    const mappedNode = nodeRegistry.value?.get(taskNodeId)
    if (mappedNode?.mappingKey) {
      highlightIds.add(mappedNode.mappingKey)
    }
  }

  return highlightIds.size ? highlightIds : null
})

watch(
  stageOptions,
  (options) => {
    if (!options.length) {
      return
    }

    if (!options.some((option) => option.value === props.stageName)) {
      const fallbackStage = options.some((option) => option.value === DEFAULT_MEMVIEW_STAGE)
        ? DEFAULT_MEMVIEW_STAGE
        : options[0].value
      if (fallbackStage && fallbackStage !== props.stageName) {
        emit('update:stage-name', fallbackStage)
      }
    }
  },
  { immediate: true },
)

watch(
  () => [props.datasetName, resolvedStageName.value, selectedRankIds.value.join(',')],
  async ([datasetName, stage, rankKey]) => {
    await loadGraphIndex()
    await loadPlaybackInfo()
    await nextTick()
    await ensureVisibleRanksLoaded()
  },
  { immediate: true },
)

watch(
  subgraphLayout,
  (next) => {
    if (!next) return
    const flattened = next.streams.flatMap((stream) => stream.nodes)
    if (flattened.length) emit('nodes-loaded', flattened)
  },
  { immediate: false },
)

watch(
  currentGraphNodeIds,
  (nodeIds) => {
    emit('graph-node-ids-change', nodeIds)
  },
  { immediate: true },
)

watch(
  () => [activeRankIds.value.join(','), resolvedStageName.value, props.datasetName, Boolean(props.subgraphNode)],
  () => {
    ensureVisibleRanksLoaded()
  },
)

watch(
  () => resolvedStageName.value,
  () => {
    zoomScale.value = 1
  },
)

function isMissingResourceError(error) {
  const message = error instanceof Error ? error.message : String(error ?? '')
  return /not found|404|missing/i.test(message)
}

async function loadGraphIndex() {
  rankGraphStates.value = []
  graphIndex.value = null
  errorMessage.value = ''
  rankRequestCache.clear()

  if (!props.datasetName || !resolvedStageName.value) {
    return
  }

  loadingIndex.value = true

  try {
    const buffer = await fetchMsgpack({
      dataset: props.datasetName,
      kind: 'graph',
      group: resolvedStageName.value,
      file: 'index.msgpack',
    })
    const parsedIndex = parseGraphIndex(buffer)
    graphIndex.value = parsedIndex
    const availableRanks = new Set(parsedIndex.ranks ?? [])
    rankGraphStates.value = selectedRankIds.value
      .filter((rankId) => availableRanks.has(rankId))
      .map((rankId) => ({
        rankId,
        loaded: false,
        streams: [],
        nodeCount: 0,
        graphLayout: null,
      }))
    await nextTick()
  } catch (error) {
    if (!isMissingResourceError(error)) {
      errorMessage.value = error instanceof Error ? error.message : 'Failed to load graph index.'
    }
  } finally {
    loadingIndex.value = false
  }
}

async function loadPlaybackInfo() {
  playbackTotalSteps.value = 0
  emit('update:totalSteps', 0)

  const targetRank = selectedRankIds.value[0]
  if (!props.datasetName || !Number.isInteger(targetRank)) {
    return
  }

  try {
    const buffer = await fetchMsgpack({
      dataset: props.datasetName,
      kind: 'memory',
      rank: targetRank,
      file: 'manifest.msgpack',
    })
    const manifest = parseMemoryManifest(buffer)
    const lastChunk = manifest.chunks.at(-1)
    playbackTotalSteps.value = lastChunk ? Number(lastChunk.logicalStepEnd) + 1 : 0
    emit('update:totalSteps', playbackTotalSteps.value)
    emit('update:currentStep', Math.min(props.currentStep, Math.max(0, playbackTotalSteps.value - 1)))
  } catch {
    playbackTotalSteps.value = 0
    emit('update:totalSteps', 0)
  }
}

async function ensureVisibleRanksLoaded() {
  if (props.subgraphNode || !props.datasetName || !resolvedStageName.value) {
    return
  }

  if (!activeRankIds.value.length) {
    return
  }

  const pending = activeRankIds.value
    .map((rankId) => rankGraphStates.value.find((state) => state.rankId === rankId))
    .filter((state) => state && !state.loaded)
    .map((state) => loadRankGraph(state.rankId))

  if (pending.length) {
    loadingRanks.value = true
    await Promise.all(pending)
    loadingRanks.value = false
  }
}

async function loadRankGraph(rankId) {
  const key = `${props.datasetName}:${resolvedStageName.value}:${rankId}`
  if (rankRequestCache.has(key)) {
    return rankRequestCache.get(key)
  }

  const request = (async () => {
    try {
      const buffer = await fetchMsgpack({
        dataset: props.datasetName,
        kind: 'graph',
        group: resolvedStageName.value,
        rank: rankId,
      })
      const parsed = parseGraphRank(buffer)
      const state = rankGraphStates.value.find((item) => item.rankId === rankId)
      if (!state) {
        return
      }

      state.streams = normalizeTopLevelStreams(parsed, resolvedStageName.value)
      state.nodeCount = state.streams.reduce((sum, stream) => sum + stream.nodes.length, 0)
      state.graphLayout = null
      state.loaded = true
      triggerRef(rankGraphStates)
      const flattened = state.streams.flatMap((stream) => stream.nodes)
      if (flattened.length) emit('nodes-loaded', flattened)
    } catch (error) {
      const state = rankGraphStates.value.find((item) => item.rankId === rankId)
      if (isMissingResourceError(error) && state) {
        state.streams = []
        state.nodeCount = 0
        state.graphLayout = null
        state.loaded = true
        triggerRef(rankGraphStates)
        return
      }

      if (!errorMessage.value) {
        errorMessage.value = error instanceof Error ? error.message : 'Failed to load rank graph.'
      }
    } finally {
      rankRequestCache.delete(key)
    }
  })()

  rankRequestCache.set(key, request)
  return request
}

function buildGraphLayout(streams) {
  const nodeMap = new Map()
  let maxNodeRight = 0
  const visibleStreams = streams.filter((stream) => Array.isArray(stream?.nodes) && stream.nodes.length > 0)

  const streamRows = visibleStreams.map((stream, streamIndex) => {
    const baseY = EDGE_VERTICAL_PADDING + GRAPH_OUTER_PADDING_Y + streamIndex * (STREAM_ROW_HEIGHT + STREAM_GAP)
    const labelY = baseY
    const trackY = baseY
    const streamKey = stream.streamKey ?? `stream_${stream.streamId}`
    return {
      streamKey,
      streamId: stream.streamId,
      rankId: stream.rankId,
      label: stream.label ?? `Stream ${stream.streamId}`,
      streamIndex,
      labelY,
      trackY,
      trackX: TRACK_OFFSET_X,
      trackHeight: STREAM_ROW_HEIGHT,
      nodes: [],
    }
  })

  visibleStreams.forEach((stream) => {
    const streamKey = stream.streamKey ?? `stream_${stream.streamId}`
    const streamState = streamRows.find((row) => row.streamKey === streamKey)
    if (!streamState) {
      return
    }

    stream.nodes.forEach((node, nodeIndex) => {
      const x = TRACK_OFFSET_X + TRACK_PADDING_X + GRAPH_SIDE_PADDING + GRAPH_OUTER_PADDING_X + nodeIndex * (NODE_WIDTH + NODE_GAP)
      const y = streamState.trackY + TRACK_PADDING_Y
      const positionedNode = {
        ...node,
        layoutIndex: nodeIndex,
        streamKey,
        x,
        y,
        width: NODE_WIDTH,
        height: NODE_HEIGHT,
      }

      streamState.nodes.push(positionedNode)
      nodeMap.set(node.id, positionedNode)
      maxNodeRight = Math.max(maxNodeRight, x + NODE_WIDTH)
    })
  })

  const nodesFlat = streamRows.flatMap((stream) =>
    stream.nodes.map((node) => ({
      ...node,
      minX: node.x,
      minY: node.y,
      maxX: node.x + node.width,
      maxY: node.y + node.height,
    })),
  )

  nodesFlat.forEach((node) => {
    nodeMap.set(node.id, node)
  })

  const trackBottom = streamRows.length
    ? streamRows[streamRows.length - 1].trackY + STREAM_ROW_HEIGHT
    : EDGE_VERTICAL_PADDING
  const maxSameStreamBendDepth = Math.max(12, EDGE_VERTICAL_PADDING + 8)

  const edges = []
  let edgeMinY = Infinity
  let edgeMaxY = -Infinity
  let edgeMaxX = 0

  nodesFlat.forEach((node) => {
    node.children.forEach((childId) => {
      const child = nodeMap.get(childId)
      if (!child) {
        return
      }

      const startX = node.x + node.width
      const startY = node.y + node.height / 2
      const endX = child.x
      const endY = child.y + child.height / 2
      const sameStream = node.streamKey === child.streamKey
      const distance = Math.max(1, Math.abs((child.layoutIndex ?? 0) - (node.layoutIndex ?? 0)))
      const horizontalGap = Math.abs(endX - startX)
      const controlX1 = startX + Math.max(24, horizontalGap * 0.28)
      const controlX2 = endX - Math.max(24, horizontalGap * 0.28)
      let edge

      if (sameStream) {
        const isForwardEdge = (child.layoutIndex ?? 0) > (node.layoutIndex ?? 0)
        if (distance === 1 && isForwardEdge) {
          edge = {
            kind: 'line',
            startX,
            startY,
            endX,
            endY,
            arrowBaseX: endX - 6,
            arrowBaseY: endY,
          }
        } else {
          const desiredLift = 16 + distance * 10
          const actualLift = Math.min(desiredLift, maxSameStreamBendDepth)
          const bendY = Math.min(startY, endY) - actualLift
          edge = {
            kind: 'bezier',
            startX,
            startY,
            endX,
            endY,
            cp1x: controlX1,
            cp1y: bendY,
            cp2x: controlX2,
            cp2y: bendY,
            arrowBaseX: controlX2,
            arrowBaseY: bendY,
          }
        }
      } else {
        const middleX = (startX + endX) / 2
        const desiredLift = 20 + distance * 8
        const maxCrossLift = Math.max(36, Math.min(startY, endY) - 8)
        const lift = Math.min(desiredLift, maxCrossLift)
        edge = {
          kind: 'bezier',
          startX,
          startY,
          endX,
          endY,
          cp1x: middleX,
          cp1y: startY - lift,
          cp2x: middleX,
          cp2y: endY - lift,
          arrowBaseX: middleX,
          arrowBaseY: endY - lift,
        }
      }

      const minX = Math.min(startX, endX, edge.cp1x ?? startX, edge.cp2x ?? endX)
      const minY = Math.min(startY, endY, edge.cp1y ?? startY, edge.cp2y ?? endY) - 6
      const maxX = Math.max(startX, endX, edge.cp1x ?? startX, edge.cp2x ?? endX) + 6
      const maxY = Math.max(startY, endY, edge.cp1y ?? startY, edge.cp2y ?? endY) + 6

      if (minY < edgeMinY) edgeMinY = minY
      if (maxY > edgeMaxY) edgeMaxY = maxY
      if (maxX > edgeMaxX) edgeMaxX = maxX

      edges.push({
        id: `${node.id}->${child.id}`,
        crossStream: !sameStream,
        minX,
        minY,
        maxX,
        maxY,
        ...edge,
      })
    })
  })

  const canvasWidth = Math.max(
    TRACK_OFFSET_X + TRACK_PADDING_X * 2 + GRAPH_SIDE_PADDING * 2 + NODE_WIDTH,
    maxNodeRight + TRACK_PADDING_X + GRAPH_SIDE_PADDING + GRAPH_OUTER_PADDING_X,
    edgeMaxX + TRACK_PADDING_X + GRAPH_SIDE_PADDING + GRAPH_OUTER_PADDING_X,
  )

  const bottomSlack = EDGE_VERTICAL_PADDING + GRAPH_OUTER_PADDING_Y
  const canvasHeight = Math.max(
    trackBottom + bottomSlack,
    Number.isFinite(edgeMaxY) ? edgeMaxY + bottomSlack : trackBottom + bottomSlack,
  )

  const trackWidth = Math.max(160, canvasWidth - TRACK_OFFSET_X)

  return {
    canvasWidth,
    canvasHeight,
    trackWidth,
    streamRows,
    nodesFlat,
    edges,
  }
}

function formatTaskLabel(node) {
  const taskType = String(node?.taskType ?? node?.task?.task_type ?? node?.task_type ?? 'TASK')
  return taskType.replaceAll('_', ' ')
}

function formatTaskSubtitle(node) {
  const data = node?.taskData ?? node?.task?.task_data ?? node?.task_data ?? {}
  if (data.src_slice?.buffer_type && data.dst_slice?.buffer_type) {
    return `${data.src_slice.buffer_type} -> ${data.dst_slice.buffer_type}`
  }
  if (Array.isArray(data.src_slices) && data.dst_slice?.buffer_type) {
    return `${data.src_slices.length} inputs -> ${data.dst_slice.buffer_type}`
  }
  if (typeof data.remote_rank === 'number') {
    return `remote rank ${data.remote_rank}`
  }
  return String(node?.id ?? node?.node_id ?? '')
}

function resolveNodeLookupIds(node) {
  return Array.isArray(node?.lookupIds) && node.lookupIds.length ? node.lookupIds : [node?.id].filter(Boolean)
}

function resolveOriginalNode(node) {
  if (!node?.id) return null
  const lookupIds = resolveNodeLookupIds(node)
  for (const lookupId of lookupIds) {
    const original = nodeRegistry.value?.get(lookupId) ?? unifiedNodeLookup.value.get(lookupId)
    if (original && !original.isPlaceholder) {
      return original
    }
  }

  const mappingKey = typeof node.mappingKey === 'string' && node.mappingKey ? node.mappingKey : ''
  if (mappingKey) {
    const mapped = nodeRegistry.value?.get(mappingKey) ?? unifiedNodeLookup.value.get(mappingKey)
    if (mapped && !mapped.isPlaceholder) {
      return mapped
    }
  }

  return node.isPlaceholder ? null : node
}

function handleNodeSelect(node) {
  emit('node-select', resolveOriginalNode(node) ?? node)
}

function zoomOut() {
  zoomScale.value = Math.max(MIN_ZOOM, Number((zoomScale.value - ZOOM_STEP).toFixed(2)))
}

function zoomIn() {
  zoomScale.value = Math.min(MAX_ZOOM, Number((zoomScale.value + ZOOM_STEP).toFixed(2)))
}

function resetZoom() {
  zoomScale.value = 1
}

</script>

<template>
  <section class="dashboard-panel memview-dag-panel">
    <div class="dashboard-section-title">
      <div>
        <p class="dashboard-eyebrow">MemView</p>
        <h2>DAG Task Graph</h2>
      </div>

      <div class="memview-dag-toolbar">
        <el-select
          :model-value="resolvedStageName"
          class="memview-stage-select"
          placeholder="Stage"
          @update:model-value="emit('update:stage-name', $event)"
        >
          <el-option
            v-for="option in stageOptions"
            :key="option.value"
            :label="option.label"
            :value="option.value"
          />
        </el-select>
      </div>
    </div>

    <div v-if="errorMessage" class="dashboard-empty-wrap">
      <el-empty :description="errorMessage" />
    </div>

    <div v-else-if="!hasContent" class="dashboard-empty-wrap">
      <el-empty :description="emptyText" />
    </div>

    <div v-else class="memview-dag-stage memview-dag-stage--full" :class="{ 'is-focus-view': isFocusView }">
      <button
        v-if="isFocusView"
        type="button"
        class="memview-dag-focus-exit"
        aria-label="Exit focus view"
        @click="emit('close-focus-view')"
      >
        <el-icon><Close /></el-icon>
      </button>
      <MemViewDagPixiCanvas
        :key="`${resolvedStageName}-${props.subgraphNode?.id ?? 'root'}-${activeRankIds.join('-')}-${isFocusView ? `focus-${props.focusStep}` : 'normal'}`"
        :graph-layout="renderGraphState.graphLayout"
        :stage-name="resolvedStageName"
        :selected-node-id="selectedNodeId"
        :issue-node-ids="issueNodeIds"
        :highlighted-node-ids="stepHighlightedNodeIds"
        :is-focus-mode="isFocusView"
        :focus-node-ids="focusAnchorNodeIds"
        :zoom-scale="zoomScale"
        @node-select="handleNodeSelect"
      />
      <div class="memview-dag-zoom-controls" aria-label="DAG zoom controls">
        <button
          type="button"
          class="memview-dag-zoom-button"
          aria-label="Zoom out DAG"
          :disabled="zoomScale <= MIN_ZOOM"
          @click="zoomOut"
        >
          <el-icon><ZoomOut /></el-icon>
        </button>
        <button
          type="button"
          class="memview-dag-zoom-button memview-dag-zoom-button--meta"
          aria-label="Reset DAG zoom"
          @click="resetZoom"
        >
          <span>{{ Math.round(zoomScale * 100) }}%</span>
          <el-icon><RefreshRight /></el-icon>
        </button>
        <button
          type="button"
          class="memview-dag-zoom-button"
          aria-label="Zoom in DAG"
          :disabled="zoomScale >= MAX_ZOOM"
          @click="zoomIn"
        >
          <el-icon><ZoomIn /></el-icon>
        </button>
      </div>
    </div>

  </section>
</template>

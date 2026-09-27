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
import { computed, nextTick, onBeforeUnmount, onMounted, ref, shallowRef, watch } from 'vue'
import Graph from 'graphology'
import Sigma from 'sigma'
import { EdgeArrowProgram, EdgeLineProgram } from 'sigma/rendering'
import { EdgeCurvedArrowProgram } from '@sigma/edge-curve'
import SigmaRoundedRectNodeProgram from './sigmaRoundedRectNodeProgram'

const TASK_NODE_WIDTH = 104
const TASK_NODE_HEIGHT = 42
const NODE_GAP = 90
const ROW_SPACING = TASK_NODE_HEIGHT + NODE_GAP
const COL_SPACING = TASK_NODE_WIDTH + NODE_GAP
const LABEL_OFFSET_X = 190
const CONTROL_NODE_SIZE = 0.001
const STRAIGHT_EDGE_EPSILON = 0.001
const TASK_NODE_CORNER_RADIUS = 8
const LABEL_MOVE_HIDE_NODE_LIMIT = 1500
const EDGE_MOVE_HIDE_NODE_LIMIT = 4000
const ZOOM_EVENT_EPSILON = 0.01
const NODE_FOCUS_ANIMATION_DURATION = 320
const NODE_FOCUS_POSITION_EPSILON = 0.0005
const LOOP_OVERLAY_PADDING_X = 16
const LOOP_OVERLAY_PADDING_Y = 12
const LOOP_OVERLAY_LABEL_HEIGHT = 26
const LOOP_OVERLAY_TOP_PADDING_EXTRA = 8
const LOOP_OVERLAY_LABEL_OUTER_GAP = 10
const LOOP_OVERLAY_LABEL_WIDTH_LIMIT = 188
const LOOP_OVERLAY_SCREEN_MIN = 8
const LOOP_OVERLAY_LABEL_INSET_X = 10
const LOOP_OVERLAY_BOX_BORDER_WIDTH = 2
const LOOP_OVERLAY_BOX_BORDER_RADIUS = 14
const LOOP_OVERLAY_LABEL_MAX_WIDTH = 240
const LOOP_OVERLAY_MIN_SCALE = 0.32
const LOOP_OVERLAY_MAX_SCALE = 1.08

const props = defineProps({
  scene: {
    type: Object,
    default: null,
  },
  selectedNodeId: {
    type: String,
    default: '',
  },
  issueNodeIds: {
    type: Object,
    default: null,
  },
  highlightedNodeIds: {
    type: Object,
    default: null,
  },
  focusNodeIds: {
    type: Array,
    default: () => [],
  },
  showLabels: {
    type: Boolean,
    default: true,
  },
  showEdges: {
    type: Boolean,
    default: true,
  },
})

const emit = defineEmits(['node-select', 'zoom-change'])

const hostRef = ref(null)
const renderRootRef = ref(null)
const sigmaRef = shallowRef(null)
const hoveredNodeId = ref('')
const hoveredLoopStartNodeId = ref('')
const loopOverlayItems = ref([])
const initialCameraRatioRef = shallowRef(null)
const lastEmittedZoomScaleRef = shallowRef(null)
let resizeObserver = null
let pendingRebuildFrame = 0
let pendingZoomEmitFrame = 0
let pendingOverlayRefreshFrame = 0

const selectedNodeSet = computed(() => {
  const set = new Set()
  if (props.selectedNodeId) set.add(props.selectedNodeId)
  return set
})

const highlightedNodeSet = computed(() =>
  props.highlightedNodeIds instanceof Set ? props.highlightedNodeIds : new Set(),
)

const issueNodeSet = computed(() =>
  props.issueNodeIds instanceof Set ? props.issueNodeIds : new Set(),
)

const focusNodeSet = computed(() =>
  new Set((props.focusNodeIds ?? []).filter((nodeId) => typeof nodeId === 'string' && nodeId)),
)

const loopGroupByStartNodeId = computed(() =>
  props.scene?.loopGroupByStartNodeId instanceof Map ? props.scene.loopGroupByStartNodeId : new Map(),
)

// Tracks every loop that should be emphasized because a related node is selected, focused, hovered, or flagged.
const activeLoopStartNodeSet = computed(() => {
  const set = new Set()
  const appendLoopContexts = (nodeId) => {
    if (!nodeId) return
    const node = props.scene?.nodeById instanceof Map ? props.scene.nodeById.get(nodeId) ?? null : null
    if (!node) return
    for (const loopStartNodeId of node.loopChainStartNodeIds ?? []) {
      if (loopGroupByStartNodeId.value.has(loopStartNodeId)) set.add(loopStartNodeId)
    }
    if (node.loopRole === 'start' && node.id && loopGroupByStartNodeId.value.has(node.id)) {
      set.add(node.id)
    }
  }

  appendLoopContexts(props.selectedNodeId)
  appendLoopContexts(hoveredNodeId.value)

  for (const nodeId of highlightedNodeSet.value) appendLoopContexts(nodeId)
  for (const nodeId of issueNodeSet.value) appendLoopContexts(nodeId)
  for (const nodeId of focusNodeSet.value) appendLoopContexts(nodeId)

  if (hoveredLoopStartNodeId.value && loopGroupByStartNodeId.value.has(hoveredLoopStartNodeId.value)) {
    set.add(hoveredLoopStartNodeId.value)
  }

  return set
})

// Maps task types to the visual accent colors used by node borders.
function colorForTaskType(taskType) {
  const value = String(taskType ?? '').toUpperCase()
  if (value === 'START' || value === 'END') return '#6e7c95'
  if (value === 'BATCH_TRANS_MEM') return '#739dff'
  if (value === 'BATCH_REDUCE') return '#f2bc63'
  if (value === 'TRANS_MEM') return '#4e9cff'
  if (value === 'REDUCE') return '#f0a53a'
  if (value === 'RECORD') return '#15a46a'
  if (value === 'WAIT') return '#c25cff'
  if (value === 'CCU_GRAPH') return '#ff6b88'
  if (value === 'AIV_GRAPH') return '#4dc8c8'
  return '#8ba0c9'
}

// Blends two hex colors for hover / active states without introducing extra color utilities.
function mixHexColors(colorA, colorB, weightOfB = 0.5) {
  const hexA = String(colorA ?? '').replace('#', '')
  const hexB = String(colorB ?? '').replace('#', '')
  if (hexA.length !== 6 || hexB.length !== 6) return colorA || colorB || '#173052'

  const weight = Math.max(0, Math.min(1, weightOfB))
  const mixChannel = (index) => {
    const a = Number.parseInt(hexA.slice(index, index + 2), 16)
    const b = Number.parseInt(hexB.slice(index, index + 2), 16)
    return Math.round(a * (1 - weight) + b * weight)
      .toString(16)
      .padStart(2, '0')
  }

  return `#${mixChannel(0)}${mixChannel(2)}${mixChannel(4)}`
}

// Maps edge kinds to the base stroke color used by Sigma.
function edgeColor(edgeKind, isPrimaryAlignEdge) {
  if (isPrimaryAlignEdge) return '#f0a53a'
  if (edgeKind === 'notify') return '#c25cff'
  if (edgeKind === 'lane_order') return '#6ea9ff'
  return '#7b8aa8'
}

// Converts task identifiers into the uppercase label style used on DAG nodes.
function formatTaskLabel(label) {
  const normalized = String(label ?? '')
    .replace(/^BATCH_/i, '')
    .replaceAll('_', ' ')
    .trim()
  return normalized || 'TASK'
}

// Returns the rendered task-node dimensions shared by Sigma and the loop overlay.
function taskNodeDimensions() {
  const width = TASK_NODE_WIDTH
  const height = TASK_NODE_HEIGHT
  const hitSize = Math.max(width, height) / 2
  return { width, height, hitSize }
}

// Builds the color bundle for loop boxes and pills for default vs active/hovered states.
function resolveLoopOverlayColor(depth, state = 'default') {
  const palette = ['#68a6ff', '#7fd4ff', '#9ce7d9', '#f2d27b', '#ffb16e']
  const baseColor = palette[Math.max(0, depth) % palette.length]
  if (state === 'active') {
    return {
      stroke: mixHexColors(baseColor, '#ffffff', 0.22),
      fill: hexToRgba(baseColor, 0.06),
      labelBackground: hexToRgba(mixHexColors(baseColor, '#12213a', 0.5), 0.94),
      labelBorder: hexToRgba(mixHexColors(baseColor, '#ffffff', 0.22), 0.6),
      labelText: '#f6fbff',
    }
  }
  return {
    stroke: hexToRgba(baseColor, 0.92),
    fill: hexToRgba(baseColor, 0.018 + Math.min(depth, 4) * 0.006),
    labelBackground: hexToRgba(mixHexColors(baseColor, '#12213a', 0.56), 0.9),
    labelBorder: hexToRgba(baseColor, 0.35),
    labelText: '#e6f3ff',
  }
}

// Produces the default pill label shown next to each loop frame.
function formatLoopLabel(loopGroup) {
  return 'Loop'
}

// Falls back to a compact label when the loop frame is too narrow.
function formatLoopCompactLabel(loopGroup) {
  return 'Loop'
}

// Matches the inspector behavior by exposing one aggregated expansion count instead of raw checker counters.
function resolveLoopTotalExpandCount(loopGroup) {
  const loopCnt = Number.isInteger(loopGroup?.loopCnt) ? Number(loopGroup.loopCnt) : null
  const expandCnt = Number.isInteger(loopGroup?.loopExpandCnt) ? Number(loopGroup.loopExpandCnt) : null
  if (!Number.isInteger(loopCnt) || !Number.isInteger(expandCnt)) return null
  return loopCnt * expandCnt
}

// Clamps overlay scaling so tiny graphs and huge graphs stay readable.
function clampValue(value, min, max) {
  return Math.min(max, Math.max(min, value))
}

// Filters out Sigma helper nodes that should not participate in task interactions.
function isTaskNodeId(nodeId) {
  return typeof nodeId === 'string' && !nodeId.startsWith('lane-label:') && !nodeId.startsWith('control:')
}

// Converts a lane row index into the graph-space Y coordinate used by Sigma.
function graphYForRow(rowIndex) {
  return -rowIndex * ROW_SPACING
}

// Rebuilds stable graph-space positions from the scene model before feeding Sigma.
function buildSceneGeometry(scene) {
  const laneRowById = new Map(
    (scene?.laneOrder ?? []).map((lane) => [lane.laneId, Number.isFinite(lane.rowIndex) ? Number(lane.rowIndex) : 0]),
  )
  const taskPositions = new Map()

  for (const node of scene?.taskNodes ?? []) {
    const rowIndex = laneRowById.get(node.laneId) ?? 0
    const colIndex = Number.isFinite(node.finalCol) ? Number(node.finalCol) : Number(node.pos ?? 0)
    taskPositions.set(node.id, {
      x: colIndex * COL_SPACING,
      y: graphYForRow(rowIndex),
      rowIndex,
      colIndex,
    })
  }

  return { laneRowById, taskPositions }
}

// Adds a node only when it is not already present in the graphology instance.
function addNode(graph, key, attributes) {
  if (graph.hasNode(key)) return
  graph.addNode(key, attributes)
}

// Adds an edge only when both endpoints exist and the key is still unused.
function addEdge(graph, key, source, target, attributes) {
  if (!graph.hasNode(source) || !graph.hasNode(target)) return
  if (graph.hasEdge(key)) return
  graph.addEdgeWithKey(key, source, target, attributes)
}

// Detects whether an edge should react to the same active states as its endpoint nodes.
function isEdgeRelatedToActiveState(edgeLike) {
  return (
    selectedNodeSet.value.has(edgeLike.sourceTaskId) ||
    selectedNodeSet.value.has(edgeLike.targetTaskId) ||
    highlightedNodeSet.value.has(edgeLike.sourceTaskId) ||
    highlightedNodeSet.value.has(edgeLike.targetTaskId) ||
    issueNodeSet.value.has(edgeLike.sourceTaskId) ||
    issueNodeSet.value.has(edgeLike.targetTaskId) ||
    focusNodeSet.value.has(edgeLike.sourceTaskId) ||
    focusNodeSet.value.has(edgeLike.targetTaskId)
  )
}

// Applies highlight emphasis to edges without mutating the original edge record.
function resolveEdgeVisual(edgeLike) {
  const relatedToSelection = isEdgeRelatedToActiveState(edgeLike)
  return {
    ...edgeLike,
    size: relatedToSelection ? Math.max(edgeLike.size * 1.35, 1.8) : edgeLike.size,
    color: relatedToSelection ? '#f6d365' : edgeLike.color,
    zIndex: relatedToSelection ? 4 : 1,
  }
}

// Converts the normalized scene model into the graphology structure consumed by Sigma.
function buildGraph(scene) {
  const graph = new Graph({ multi: true })
  const { taskPositions } = buildSceneGeometry(scene)
  const taskNodeById = new Map((scene?.taskNodes ?? []).map((node) => [node.id, node]))

  for (const lane of scene?.laneOrder ?? []) {
    addNode(graph, `lane-label:${lane.laneId}`, {
      x: -LABEL_OFFSET_X,
      y: graphYForRow(Number(lane.rowIndex ?? 0)),
      size: CONTROL_NODE_SIZE,
      color: 'rgba(0,0,0,0)',
      label: lane.label,
      laneId: lane.laneId,
      rankId: lane.rankId,
      isLaneLabel: true,
    })
  }

  for (const node of scene?.taskNodes ?? []) {
    const position = taskPositions.get(node.id)
    if (!position) continue
    const dimensions = taskNodeDimensions()
    const maxDimension = Math.max(dimensions.width, dimensions.height, 1)
    addNode(graph, node.id, {
      x: position.x,
      y: position.y,
      size: dimensions.hitSize,
      color: '#173052',
      type: 'roundedRect',
      label: formatTaskLabel(node.label || node.taskType || node.id),
      forceLabel: true,
      shortLabel: formatTaskLabel(node.label || node.taskType || node.id),
      rectWidth: dimensions.width,
      rectHeight: dimensions.height,
      rectWidthFactor: dimensions.width / maxDimension,
      rectHeightFactor: dimensions.height / maxDimension,
      rectCornerRadius: TASK_NODE_CORNER_RADIUS,
      subtitle: node.subtitle || '',
      rankId: node.rankId,
      streamId: node.streamId,
      queueId: node.queueId,
      laneId: node.laneId,
      taskType: node.taskType,
      isTaskNode: true,
      taskNodeId: node.id,
      taskNode: node,
    })
  }

  if (props.showEdges) {
    for (const edge of scene?.edgeRecords ?? []) {
      const source = taskPositions.get(edge.srcNodeId)
      const target = taskPositions.get(edge.dstNodeId)
      if (!source || !target) continue
      const sourceNode = taskNodeById.get(edge.srcNodeId) ?? null
      const targetNode = taskNodeById.get(edge.dstNodeId) ?? null
      const isSameLane = Boolean(
        sourceNode &&
        targetNode &&
        sourceNode.laneId &&
        sourceNode.laneId === targetNode.laneId,
      )
      const streamSeqGap =
        sourceNode && targetNode && Number.isFinite(sourceNode.streamSeq) && Number.isFinite(targetNode.streamSeq)
          ? Math.abs(Number(sourceNode.streamSeq) - Number(targetNode.streamSeq))
          : 0
      const colGap = Math.abs(source.x - target.x) / COL_SPACING
      const shouldUseArcRoute = isSameLane && (streamSeqGap > 1 || colGap > 1 + STRAIGHT_EDGE_EPSILON)
      const stroke = edgeColor(edge.edgeKind, edge.isPrimaryAlignEdge)
      const baseSize = edge.isPrimaryAlignEdge ? 1.8 : edge.edgeKind === 'lane_order' ? 1.3 : 1.15
      const curvature =
        shouldUseArcRoute
          ? Math.min(0.55, 0.22 + Math.max(0, colGap - 2) * 0.035)
          : 0

      addEdge(graph, edge.edgeId, edge.srcNodeId, edge.dstNodeId, {
        type: shouldUseArcRoute ? 'curved' : 'arrow',
        size: baseSize,
        color: stroke,
        curvature,
        edgeId: edge.edgeId,
        edgeKind: edge.edgeKind,
        isPrimaryAlignEdge: Boolean(edge.isPrimaryAlignEdge),
        sourceTaskId: edge.srcNodeId,
        targetTaskId: edge.dstNodeId,
        routeMode: shouldUseArcRoute ? 'arc' : 'straight',
        zIndex: 1,
      })
    }
  }

  return { graph }
}

// Creates the Sigma node reducer used for selection, issue, focus, and hover styling.
function makeNodeReducer() {
  return (nodeKey, data) => {
    if (data.isControl) {
      return {
        ...data,
        label: '',
        size: CONTROL_NODE_SIZE,
        color: 'rgba(0,0,0,0)',
      }
    }

    if (data.isLaneLabel) {
      return {
        ...data,
        label: props.showLabels ? data.label : '',
        forceLabel: props.showLabels,
        size: CONTROL_NODE_SIZE,
        color: 'rgba(0,0,0,0)',
      }
    }

    const isSelected = selectedNodeSet.value.has(nodeKey)
    const isStepHighlighted = highlightedNodeSet.value.has(nodeKey)
    const isIssue = issueNodeSet.value.has(nodeKey)
    const isFocused = focusNodeSet.value.has(nodeKey)
    const isHovered = hoveredNodeId.value === nodeKey
    const emphasized = isSelected || isStepHighlighted || isIssue || isFocused || isHovered
    const visual = resolveTaskNodeVisual(nodeKey, data)

    return {
      ...data,
      forceLabel: props.showLabels,
      label: props.showLabels ? data.label : '',
      size: data.size,
      color: '#173052',
      fillColor: hexToRgba(visual.fillColor, visual.fillAlpha),
      borderColor: visual.strokeColor,
      rectBorderWidth: visual.strokeWidth,
      rectCornerRadius: TASK_NODE_CORNER_RADIUS,
      isIssue,
      zIndex: emphasized ? 5 : 1,
    }
  }
}

// Creates the Sigma edge reducer used for active-state emphasis.
function makeEdgeReducer() {
  return (edgeKey, data) => ({
    ...data,
    ...resolveEdgeVisual(data),
  })
}

// Converts a hex color into the rgba string format expected by CSS/Sigma helpers.
function hexToRgba(hexColor, alpha) {
  const hex = String(hexColor ?? '').replace('#', '')
  if (hex.length !== 6) return `rgba(23,48,82,${alpha})`
  const rgb = [
    Number.parseInt(hex.slice(0, 2), 16),
    Number.parseInt(hex.slice(2, 4), 16),
    Number.parseInt(hex.slice(4, 6), 16),
  ]
  return `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${alpha})`
}

// Resolves the final node fill/border styling for every interactive state.
function resolveTaskNodeVisual(nodeKey, data) {
  const isSelected = selectedNodeSet.value.has(nodeKey)
  const isStepHighlighted = highlightedNodeSet.value.has(nodeKey)
  const isIssue = issueNodeSet.value.has(nodeKey)
  const isFocused = focusNodeSet.value.has(nodeKey)
  const isHovered = hoveredNodeId.value === nodeKey
  const hasSubgraph = Boolean(data?.taskNode?.hasSubgraph)
  const typeAccent = colorForTaskType(data?.taskType)
  const baseFillColor = mixHexColors(typeAccent, '#132238', 0.35)
  const highlightFillColor = mixHexColors(typeAccent, '#ffffff', 0.14)

  return {
    isIssue,
    label: String(data?.shortLabel ?? '').trim(),
    fillColor: isIssue
      ? '#4a1820'
      : hasSubgraph
        ? mixHexColors(typeAccent, '#3b2810', 0.28)
        : isStepHighlighted
          ? highlightFillColor
          : baseFillColor,
    fillAlpha: isHovered || isStepHighlighted || isIssue ? 0.98 : 0.92,
    strokeColor: isIssue
      ? '#ff5d6c'
      : isSelected
        ? '#9bd1ff'
        : isStepHighlighted
          ? '#fff1a8'
          : isHovered
            ? mixHexColors(typeAccent, '#ffffff', 0.35)
            : hasSubgraph
              ? '#e3a64e'
              : isFocused
                ? '#76e6a1'
                : typeAccent,
    strokeWidth: isIssue ? 2.6 : isSelected ? 2.2 : isStepHighlighted ? 1.9 : isHovered ? 1.7 : 1.2,
  }
}

// Keeps the host cursor aligned with whichever interactive overlay is currently hovered.
function syncInteractiveCursor() {
  const host = hostRef.value
  if (!host) return
  host.style.cursor = hoveredNodeId.value || hoveredLoopStartNodeId.value ? 'pointer' : 'default'
}

// Emits a user-facing zoom scale derived from Sigma's camera ratio.
function emitZoomScaleFromCamera() {
  const sigma = sigmaRef.value
  const initialRatio = initialCameraRatioRef.value
  if (!sigma || !Number.isFinite(initialRatio) || initialRatio <= 0) return

  const camera = sigma.getCamera()
  const currentState = camera.getState()
  const ratio = Number(currentState?.ratio)
  if (!Number.isFinite(ratio) || ratio <= 0) return
  const scale = Number((initialRatio / ratio).toFixed(3))
  if (!Number.isFinite(scale) || scale <= 0) return
  if (
    Number.isFinite(lastEmittedZoomScaleRef.value) &&
    Math.abs(scale - Number(lastEmittedZoomScaleRef.value)) < ZOOM_EVENT_EPSILON
  ) {
    return
  }

  if (pendingZoomEmitFrame) cancelAnimationFrame(pendingZoomEmitFrame)
  pendingZoomEmitFrame = requestAnimationFrame(() => {
    pendingZoomEmitFrame = 0
    lastEmittedZoomScaleRef.value = scale
    emit('zoom-change', scale)
  })
}

// Captures the initial camera ratio once the renderer is ready.
function captureInitialCameraRatio() {
  const sigma = sigmaRef.value
  if (!sigma) return

  const state = sigma.getCamera().getState()
  if (!Number.isFinite(state?.ratio) || state.ratio <= 0) return

  initialCameraRatioRef.value = state.ratio
  emitZoomScaleFromCamera()
}

// Hides labels during camera movement on very large graphs to keep interaction smooth.
function shouldHideLabelsOnMove(scene) {
  if (!props.showLabels) return false
  const nodeCount = scene?.taskNodes?.length ?? 0
  return nodeCount > LABEL_MOVE_HIDE_NODE_LIMIT
}

// Hides edge rendering during movement on very large graphs for performance.
function shouldHideEdgesOnMove(scene) {
  if (!props.showEdges) return false
  const nodeCount = scene?.taskNodes?.length ?? 0
  return nodeCount > EDGE_MOVE_HIDE_NODE_LIMIT
}

// Resizes Sigma and then schedules an overlay refresh in the same viewport.
function refreshRendererSize() {
  const sigma = sigmaRef.value
  const root = renderRootRef.value
  if (!root) return

  if (root.offsetWidth <= 1 || root.offsetHeight <= 1) {
    if (!sigma && props.scene?.taskNodes?.length) scheduleRebuildRenderer()
    return
  }

  if (!sigma) {
    if (props.scene?.taskNodes?.length) scheduleRebuildRenderer()
    return
  }

  sigma.resize(true)
  sigma.refresh()
  if (!Number.isFinite(initialCameraRatioRef.value) || initialCameraRatioRef.value <= 0) {
    captureInitialCameraRatio()
  }
  scheduleOverlayRefresh()
  syncInteractiveCursor()
}

// Defers expensive renderer rebuilds to the next animation frame.
function scheduleRebuildRenderer() {
  if (pendingRebuildFrame) return
  pendingRebuildFrame = requestAnimationFrame(() => {
    pendingRebuildFrame = 0
    rebuildRenderer()
  })
}

// Watches the Sigma host size so both the renderer and overlay stay aligned.
function ensureResizeObserver() {
  if (resizeObserver || typeof ResizeObserver !== 'function') return
  resizeObserver = new ResizeObserver(() => {
    refreshRendererSize()
  })

  if (hostRef.value) resizeObserver.observe(hostRef.value)
  if (renderRootRef.value && renderRootRef.value !== hostRef.value) {
    resizeObserver.observe(renderRootRef.value)
  }
}

// Cleans up the shared ResizeObserver on teardown.
function teardownResizeObserver() {
  if (!resizeObserver) return
  resizeObserver.disconnect()
  resizeObserver = null
}

// Cancels any queued overlay refresh before rebuilding or unmounting.
function cancelOverlayRefreshFrame() {
  if (!pendingOverlayRefreshFrame) return
  cancelAnimationFrame(pendingOverlayRefreshFrame)
  pendingOverlayRefreshFrame = 0
}

// Fully tears down Sigma state and the derived loop overlay state.
function resetRenderer() {
  if (sigmaRef.value) {
    sigmaRef.value.kill()
    sigmaRef.value = null
  }
  initialCameraRatioRef.value = null
  lastEmittedZoomScaleRef.value = null
  hoveredNodeId.value = ''
  hoveredLoopStartNodeId.value = ''
  loopOverlayItems.value = []
  cancelOverlayRefreshFrame()
  syncInteractiveCursor()
}

// Derives loop-overlay scaling from current node screen width instead of raw camera ratio.
function resolveLoopOverlayScale(referenceNodeWidth) {
  const width = Number(referenceNodeWidth)
  if (!Number.isFinite(width) || width <= 0) return 1
  return clampValue(width / TASK_NODE_WIDTH, LOOP_OVERLAY_MIN_SCALE, LOOP_OVERLAY_MAX_SCALE)
}

// Builds one loop overlay box + pill from the current Sigma frame.
// Call chain: camera/selection updates -> scheduleOverlayRefresh -> refreshLoopOverlay -> buildLoopOverlayItem.
function buildLoopOverlayItem(loopGroup, sigma) {
  const taskNodeIds = Array.isArray(loopGroup?.memberNodeIds) ? loopGroup.memberNodeIds : []
  if (!taskNodeIds.length) return null

  let minX = Number.POSITIVE_INFINITY
  let maxX = Number.NEGATIVE_INFINITY
  let minY = Number.POSITIVE_INFINITY
  let maxY = Number.NEGATIVE_INFINITY
  let totalRectWidth = 0
  let visibleNodeCount = 0

  for (const nodeId of taskNodeIds) {
    const displayData = sigma.getNodeDisplayData(nodeId)
    if (!displayData || !displayData.isTaskNode) continue

    const viewportCenter = sigma.framedGraphToViewport(displayData)
    const scaledNodeSize = sigma.scaleSize(Number(displayData.size ?? 0))
    const rectWidth =
      Math.max(1, scaledNodeSize * 2 * Math.max(0.02, Number(displayData.rectWidthFactor ?? 1)))
    const rectHeight =
      Math.max(1, scaledNodeSize * 2 * Math.max(0.02, Number(displayData.rectHeightFactor ?? 1)))

    const left = viewportCenter.x - rectWidth / 2
    const right = viewportCenter.x + rectWidth / 2
    const top = viewportCenter.y - rectHeight / 2
    const bottom = viewportCenter.y + rectHeight / 2

    minX = Math.min(minX, left)
    maxX = Math.max(maxX, right)
    minY = Math.min(minY, top)
    maxY = Math.max(maxY, bottom)
    totalRectWidth += rectWidth
    visibleNodeCount += 1
  }

  if (!visibleNodeCount || ![minX, maxX, minY, maxY].every(Number.isFinite)) return null

  const averageRectWidth = totalRectWidth / visibleNodeCount
  const overlayScale = resolveLoopOverlayScale(averageRectWidth)
  const overlayPaddingX = LOOP_OVERLAY_PADDING_X * overlayScale
  const overlayPaddingY = LOOP_OVERLAY_PADDING_Y * overlayScale
  const overlayTopPaddingExtra = LOOP_OVERLAY_TOP_PADDING_EXTRA * overlayScale
  const overlayLabelOuterGap = LOOP_OVERLAY_LABEL_OUTER_GAP * overlayScale
  const overlayLabelInsetX = LOOP_OVERLAY_LABEL_INSET_X * overlayScale
  const overlayBorderWidth = Math.max(1, LOOP_OVERLAY_BOX_BORDER_WIDTH * overlayScale)
  const overlayBorderRadius = LOOP_OVERLAY_BOX_BORDER_RADIUS * overlayScale
  const labelFontSize = Math.max(10, 12.5 * overlayScale)
  const labelPaddingY = 5 * overlayScale
  const labelPaddingX = 10 * overlayScale
  const labelHeight = Math.max(LOOP_OVERLAY_LABEL_HEIGHT * overlayScale, labelFontSize * 1.35 + labelPaddingY * 2 + 2)

  const host = hostRef.value
  const viewportWidth = Number(host?.clientWidth ?? 0)
  const viewportHeight = Number(host?.clientHeight ?? 0)
  if (viewportWidth <= 0 || viewportHeight <= 0) return null

  const boxLeft = minX - overlayPaddingX
  const boxTop = minY - overlayPaddingY - overlayTopPaddingExtra
  const boxWidth = maxX - minX + overlayPaddingX * 2
  const boxHeight = maxY - minY + overlayPaddingY * 2 + overlayTopPaddingExtra
  if (boxWidth < LOOP_OVERLAY_SCREEN_MIN || boxHeight < LOOP_OVERLAY_SCREEN_MIN) return null
  if (boxLeft > viewportWidth || boxTop > viewportHeight || boxLeft + boxWidth < 0 || boxTop + boxHeight < 0) {
    return null
  }

  const isActive = activeLoopStartNodeSet.value.has(loopGroup.startNodeId)
  const isHovered = hoveredLoopStartNodeId.value === loopGroup.startNodeId
  const visual = resolveLoopOverlayColor(loopGroup.depth ?? 0, isActive || isHovered ? 'active' : 'default')
  const label = formatLoopLabel(loopGroup)
  const compactLabel = formatLoopCompactLabel(loopGroup)
  const showCompactLabel = boxWidth < LOOP_OVERLAY_LABEL_WIDTH_LIMIT * overlayScale

  return {
    id: loopGroup.startNodeId,
    startNodeId: loopGroup.startNodeId,
    x: boxLeft,
    y: boxTop,
    width: boxWidth,
    height: boxHeight,
    label,
    compactLabel,
    visibleLabel: showCompactLabel ? compactLabel : label,
    showCompactLabel,
    tooltipRows: [
      ['Total expands', Number.isInteger(resolveLoopTotalExpandCount(loopGroup)) ? String(resolveLoopTotalExpandCount(loopGroup)) : '--'],
      [
        'Instr range',
        Number.isInteger(loopGroup.instrIdStart) && Number.isInteger(loopGroup.instrIdEnd)
          ? `${loopGroup.instrIdStart}-${loopGroup.instrIdEnd}`
          : '--',
      ],
      ['Node count', Number.isInteger(loopGroup.nodeCount) ? String(loopGroup.nodeCount) : '--'],
    ],
    visual,
    depth: loopGroup.depth ?? 0,
    isActive,
    isHovered,
    scale: overlayScale,
    borderWidth: overlayBorderWidth,
    borderRadius: overlayBorderRadius,
    labelX: boxLeft + overlayLabelInsetX,
    labelY: boxTop - labelHeight - overlayLabelOuterGap,
    labelMaxWidth: LOOP_OVERLAY_LABEL_MAX_WIDTH * overlayScale,
    labelPaddingX,
    labelPaddingY,
    labelFontSize,
  }
}

// Recomputes all loop overlay geometry from the latest Sigma display cache.
function refreshLoopOverlay() {
  const sigma = sigmaRef.value
  const scene = props.scene
  if (!sigma || !scene?.taskNodes?.length || !Array.isArray(scene?.loopGroups) || !scene.loopGroups.length) {
    loopOverlayItems.value = []
    syncInteractiveCursor()
    return
  }

  const items = scene.loopGroups
    .map((loopGroup) => buildLoopOverlayItem(loopGroup, sigma))
    .filter(Boolean)
    .sort((left, right) => {
      if (left.depth !== right.depth) return left.depth - right.depth
      if (left.isActive !== right.isActive) return left.isActive ? 1 : -1
      return left.startNodeId.localeCompare(right.startNodeId)
    })

  loopOverlayItems.value = items
  syncInteractiveCursor()
}

// Batches repeated overlay updates into a single animation-frame refresh.
function scheduleOverlayRefresh() {
  if (pendingOverlayRefreshFrame) return
  pendingOverlayRefreshFrame = requestAnimationFrame(() => {
    pendingOverlayRefreshFrame = 0
    refreshLoopOverlay()
  })
}

// Starts loop hover state from the pill interaction layer.
function handleLoopOverlayEnter(startNodeId) {
  hoveredLoopStartNodeId.value = typeof startNodeId === 'string' ? startNodeId : ''
  scheduleOverlayRefresh()
}

// Clears loop hover state when the pointer leaves the pill.
function handleLoopOverlayLeave(startNodeId) {
  if (hoveredLoopStartNodeId.value !== startNodeId) return
  hoveredLoopStartNodeId.value = ''
  scheduleOverlayRefresh()
}

// Reuses node selection flow when a loop pill is clicked.
function handleLoopOverlayClick(startNodeId) {
  if (!startNodeId) return
  const taskNode = props.scene?.nodeById instanceof Map ? props.scene.nodeById.get(startNodeId) ?? null : null
  if (taskNode) emit('node-select', taskNode)
}

// Zooms the Sigma camera in using the built-in animation helper.
async function zoomIn() {
  const sigma = sigmaRef.value
  if (!sigma) return
  await sigma.getCamera().animatedZoom()
}

// Zooms the Sigma camera out using the built-in animation helper.
async function zoomOut() {
  const sigma = sigmaRef.value
  if (!sigma) return
  await sigma.getCamera().animatedUnzoom()
}

// Resets the Sigma camera to the default view.
async function resetZoom() {
  const sigma = sigmaRef.value
  if (!sigma) return
  await sigma.getCamera().animatedReset()
}

// Recenters the selected node in the current viewport without changing the zoom level.
async function focusSelectedNode() {
  const sigma = sigmaRef.value
  const nodeId = props.selectedNodeId
  if (!sigma || !isTaskNodeId(nodeId)) return

  const nodeDisplayData = sigma.getNodeDisplayData(nodeId)
  if (!nodeDisplayData) return

  const camera = sigma.getCamera()
  const cameraState = camera.getState()
  const viewportCenter = {
    x: sigma.getDimensions().width / 2,
    y: sigma.getDimensions().height / 2,
  }
  const nodeViewportPosition = sigma.framedGraphToViewport(nodeDisplayData)

  if (
    !Number.isFinite(nodeViewportPosition?.x) ||
    !Number.isFinite(nodeViewportPosition?.y) ||
    !Number.isFinite(viewportCenter.x) ||
    !Number.isFinite(viewportCenter.y)
  ) {
    return
  }

  const viewportDelta = {
    x: nodeViewportPosition.x - viewportCenter.x,
    y: nodeViewportPosition.y - viewportCenter.y,
  }

  if (
    Math.abs(viewportDelta.x) < 1 &&
    Math.abs(viewportDelta.y) < 1
  ) {
    return
  }

  const graphOrigin = sigma.viewportToFramedGraph({ x: 0, y: 0 }, { cameraState })
  const graphDelta = sigma.viewportToFramedGraph(viewportDelta, { cameraState })
  const offsetX = Number(graphDelta.x) - Number(graphOrigin.x)
  const offsetY = Number(graphDelta.y) - Number(graphOrigin.y)
  const nextX = Number(cameraState?.x ?? 0) + offsetX
  const nextY = Number(cameraState?.y ?? 0) + offsetY

  if (!Number.isFinite(nextX) || !Number.isFinite(nextY)) return
  if (
    Math.abs(Number(cameraState?.x ?? 0) - nextX) < NODE_FOCUS_POSITION_EPSILON &&
    Math.abs(Number(cameraState?.y ?? 0) - nextY) < NODE_FOCUS_POSITION_EPSILON
  ) {
    return
  }

  await camera.animate(
    {
      x: nextX,
      y: nextY,
    },
    {
      duration: NODE_FOCUS_ANIMATION_DURATION,
    },
  )
}

defineExpose({
  zoomIn,
  zoomOut,
  resetZoom,
})

// Rebuilds the Sigma renderer and reconnects the overlay refresh pipeline.
// Call chain: scene/settings watchers -> scheduleRebuildRenderer -> rebuildRenderer -> camera/overlay subscriptions.
async function rebuildRenderer() {
  const root = renderRootRef.value
  if (!root || !props.scene?.taskNodes?.length) {
    resetRenderer()
    return
  }

  if (root.offsetWidth <= 1 || root.offsetHeight <= 1) {
    scheduleRebuildRenderer()
    return
  }

  const { graph } = buildGraph(props.scene)
  const hideLabelsOnMove = shouldHideLabelsOnMove(props.scene)
  const hideEdgesOnMove = shouldHideEdgesOnMove(props.scene)
  let sigma = sigmaRef.value

  if (!sigma) {
    sigma = new Sigma(graph, root, {
      renderLabels: props.showLabels,
      renderEdgeLabels: false,
      hideEdgesOnMove,
      hideLabelsOnMove,
      minCameraRatio: null,
      maxCameraRatio: null,
      labelRenderedSizeThreshold: 99999,
      labelDensity: 0.05,
      defaultNodeType: 'circle',
      defaultEdgeType: 'arrow',
      itemSizesReference: 'positions',
      zoomToSizeRatioFunction: (ratio) => ratio,
      nodeReducer: makeNodeReducer(),
      edgeReducer: makeEdgeReducer(),
      nodeProgramClasses: {
        roundedRect: SigmaRoundedRectNodeProgram,
      },
      edgeProgramClasses: {
        line: EdgeLineProgram,
        arrow: EdgeArrowProgram,
        curved: EdgeCurvedArrowProgram,
      },
      enableEdgeEvents: false,
      minEdgeThickness: 1,
      labelFont: 'SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, monospace',
      labelWeight: '600',
      defaultDrawNodeHover: () => {},
      defaultDrawNodeLabel: (context, data, settings) => {
        if (!data.isLaneLabel || !data.label) return
        context.save()
        context.font = `600 11px ${settings.labelFont}`
        context.fillStyle = '#9aaccc'
        context.textAlign = 'right'
        context.textBaseline = 'middle'
        context.fillText(data.label, data.x - 12, data.y)
        context.restore()
      },
    })
    sigma.getCamera().on('updated', () => {
      emitZoomScaleFromCamera()
      scheduleOverlayRefresh()
    })
    sigma.on('enterNode', ({ node }) => {
      if (!isTaskNodeId(node)) return
      hoveredNodeId.value = node
      sigma.refresh()
      scheduleOverlayRefresh()
      syncInteractiveCursor()
    })
    sigma.on('leaveNode', ({ node }) => {
      if (hoveredNodeId.value !== node) return
      hoveredNodeId.value = ''
      sigma.refresh()
      scheduleOverlayRefresh()
      syncInteractiveCursor()
    })
    sigma.on('clickNode', ({ node }) => {
      if (!isTaskNodeId(node)) return
      const taskNode = props.scene?.taskNodes?.find((item) => item.id === node)
      if (taskNode) emit('node-select', taskNode)
    })
    sigmaRef.value = sigma
  } else {
    sigma.setGraph(graph)
    sigma.setSetting('renderLabels', props.showLabels)
    sigma.setSetting('hideLabelsOnMove', hideLabelsOnMove)
    sigma.setSetting('hideEdgesOnMove', hideEdgesOnMove)
  }

  await nextTick()
  sigma.resize(true)
  sigma.refresh()
  captureInitialCameraRatio()
  scheduleOverlayRefresh()
  await focusSelectedNode()
}

onMounted(() => {
  ensureResizeObserver()
  syncInteractiveCursor()
  scheduleRebuildRenderer()
})

onBeforeUnmount(() => {
  teardownResizeObserver()
  if (pendingRebuildFrame) {
    cancelAnimationFrame(pendingRebuildFrame)
    pendingRebuildFrame = 0
  }
  if (pendingZoomEmitFrame) {
    cancelAnimationFrame(pendingZoomEmitFrame)
    pendingZoomEmitFrame = 0
  }
  cancelOverlayRefreshFrame()
  resetRenderer()
})

watch(
  () => props.scene,
  () => {
    scheduleRebuildRenderer()
  },
  { deep: false, immediate: true },
)

watch(
  () => props.showEdges,
  () => {
    scheduleRebuildRenderer()
  },
)

watch(
  () => [
    props.selectedNodeId,
    props.issueNodeIds instanceof Set ? props.issueNodeIds.size : 0,
    props.highlightedNodeIds instanceof Set ? props.highlightedNodeIds.size : 0,
    props.focusNodeIds?.join(',') ?? '',
    props.showLabels ? 1 : 0,
  ],
  () => {
    const sigma = sigmaRef.value
    if (!sigma) return
    sigma.setSetting('renderLabels', props.showLabels)
    sigma.setSetting('hideLabelsOnMove', shouldHideLabelsOnMove(props.scene))
    sigma.setSetting('hideEdgesOnMove', shouldHideEdgesOnMove(props.scene))
    sigma.setSetting('nodeReducer', makeNodeReducer())
    sigma.setSetting('edgeReducer', makeEdgeReducer())
    sigma.refresh()
    scheduleOverlayRefresh()
    void focusSelectedNode()
  },
)
</script>

<template>
  <div ref="hostRef" class="memview-dag-pixi-shell memview-dag-sigma-shell">
    <div ref="renderRootRef" class="memview-dag-sigma-surface" />
    <div class="memview-dag-loop-overlay" aria-hidden="true">
      <div class="memview-dag-loop-overlay__boxes">
        <div
          v-for="item in loopOverlayItems"
          :key="`${item.id}-box`"
          class="memview-dag-loop-overlay__box"
          :class="{ 'is-active': item.isActive, 'is-hovered': item.isHovered }"
          :style="{
            left: `${item.x}px`,
            top: `${item.y}px`,
            width: `${item.width}px`,
            height: `${item.height}px`,
            backgroundColor: item.visual.fill,
            borderColor: item.visual.stroke,
            borderWidth: `${item.borderWidth}px`,
            borderRadius: `${item.borderRadius}px`,
          }"
        />
      </div>
      <div class="memview-dag-loop-overlay__labels">
        <div
          v-for="item in loopOverlayItems"
          :key="`${item.id}-label`"
          class="memview-dag-loop-label"
          :class="{ 'is-active': item.isActive, 'is-hovered': item.isHovered, 'is-compact': item.showCompactLabel }"
          :style="{
            left: `${item.labelX}px`,
            top: `${item.labelY}px`,
            backgroundColor: item.visual.labelBackground,
            borderColor: item.visual.labelBorder,
            color: item.visual.labelText,
            maxWidth: `${item.labelMaxWidth}px`,
            padding: `${item.labelPaddingY}px ${item.labelPaddingX}px`,
            fontSize: `${item.labelFontSize}px`,
          }"
          @mouseenter="handleLoopOverlayEnter(item.startNodeId)"
          @mouseleave="handleLoopOverlayLeave(item.startNodeId)"
          @click.stop="handleLoopOverlayClick(item.startNodeId)"
        >
          <span class="memview-dag-loop-label__summary">{{ item.visibleLabel }}</span>
          <div v-if="item.isHovered" class="memview-dag-loop-label__tooltip">
            <div
              v-for="([label, value], index) in item.tooltipRows"
              :key="`${item.id}-tip-${index}`"
              class="memview-dag-loop-label__tooltip-row"
            >
              <span>{{ label }}</span>
              <strong>{{ value }}</strong>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.memview-dag-sigma-shell {
  position: relative;
  height: 100%;
  min-height: 0;
  user-select: none;
}

.memview-dag-sigma-surface {
  position: absolute;
  inset: 0;
  overflow: hidden;
  pointer-events: auto;
  z-index: 2;
}

.memview-dag-sigma-surface :deep(canvas) {
  display: block;
  max-width: none;
  max-height: none;
  pointer-events: auto;
}

.memview-dag-loop-overlay {
  position: absolute;
  inset: 0;
  overflow: hidden;
  pointer-events: none;
  z-index: 3;
}

.memview-dag-loop-overlay__boxes,
.memview-dag-loop-overlay__labels {
  position: absolute;
  inset: 0;
}

.memview-dag-loop-overlay__box {
  position: absolute;
  pointer-events: none;
  border: 2px dashed rgba(104, 166, 255, 0.92);
  border-radius: 14px;
  box-sizing: border-box;
  box-shadow: inset 0 0 0 1px rgba(7, 16, 28, 0.22);
  transition:
    border-width 120ms ease,
    background-color 120ms ease,
    border-color 120ms ease,
    box-shadow 120ms ease;
}

.memview-dag-loop-overlay__box.is-active,
.memview-dag-loop-overlay__box.is-hovered {
  box-shadow:
    inset 0 0 0 1px rgba(255, 255, 255, 0.08),
    0 0 0 1px rgba(104, 166, 255, 0.12);
}

.memview-dag-loop-label {
  position: absolute;
  display: inline-flex;
  align-items: center;
  min-width: 0;
  padding: 5px 10px;
  border-radius: 999px;
  border: 1px solid rgba(104, 166, 255, 0.34);
  box-shadow: 0 10px 24px rgba(5, 10, 18, 0.28);
  font-family: 'SFMono-Regular', Menlo, Monaco, Consolas, Liberation Mono, monospace;
  line-height: 1.35;
  letter-spacing: 0.02em;
  pointer-events: auto;
  cursor: pointer;
  z-index: 2;
}

.memview-dag-loop-label.is-active,
.memview-dag-loop-label.is-hovered {
  box-shadow: 0 12px 28px rgba(7, 16, 28, 0.38);
}

.memview-dag-loop-label.is-compact {
  max-width: none;
}

.memview-dag-loop-label__summary {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.memview-dag-loop-label__tooltip {
  position: absolute;
  left: calc(100% + 10px);
  top: 50%;
  transform: translateY(-50%);
  display: grid;
  grid-template-columns: 92px minmax(0, 1fr);
  gap: 6px 10px;
  width: 256px;
  padding: 12px 14px;
  border-radius: 14px;
  border: 1px solid rgba(104, 166, 255, 0.3);
  background: rgba(58, 102, 171, 0.94);
  box-shadow: 0 16px 32px rgba(5, 10, 18, 0.32);
  pointer-events: auto;
  font-size: 12px;
  line-height: 1.45;
  z-index: 3;
}

.memview-dag-loop-label__tooltip-row {
  display: contents;
}

.memview-dag-loop-label__tooltip-row span {
  color: rgba(235, 242, 255, 0.78);
}

.memview-dag-loop-label__tooltip-row strong {
  min-width: 0;
  color: #f7fbff;
  font-weight: 600;
  white-space: normal;
  overflow-wrap: anywhere;
}
</style>

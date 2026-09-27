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
import Flatbush from 'flatbush'
import { Application, Container, Graphics, Rectangle, Text } from 'pixi.js'
import { bezierTangentAt } from '../../utils/memviewIndex.js'

const STREAM_LABEL_RENDER_SCALE = 2
const NODE_LABEL_RENDER_SCALE = 2
const VIEWPORT_OVERSCAN = 160
const BYPASS_FLATBUSH_RENDER = false
const BYPASS_VIEWPORT_WINDOW = false
const STREAM_LABEL_LEFT_INSET = 24
const STREAM_LABEL_RIGHT_GAP = 10
const STREAM_LABEL_MIN_WIDTH = 132
const STREAM_LABEL_FIXED_X = STREAM_LABEL_LEFT_INSET
const STREAM_LABEL_BAR_HEIGHT = 24

const props = defineProps({
  graphLayout: {
    type: Object,
    required: true,
  },
  stageName: {
    type: String,
    default: '',
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
  isFocusMode: {
    type: Boolean,
    default: false,
  },
  focusNodeIds: {
    type: Array,
    default: () => [],
  },
  focusNodeId: {
    type: String,
    default: '',
  },
  zoomScale: {
    type: Number,
    default: 1,
  },
})

const emit = defineEmits(['node-select'])
const hoveredNodeId = ref('')

const shellRef = ref(null)
const viewportRef = ref(null)
const hostRef = ref(null)
const appRef = shallowRef(null)
const rootSceneRef = shallowRef(null)
const streamLayerRef = shallowRef(null)
const edgeLayerRef = shallowRef(null)
const nodeLayerRef = shallowRef(null)
const labelLayerRef = shallowRef(null)

const viewportWidth = ref(0)
const viewportHeight = ref(0)
const scrollLeft = ref(0)
const scrollTop = ref(0)
const devicePixelRatioRef = ref(typeof window !== 'undefined' ? (window.devicePixelRatio || 1) : 1)
const isDragging = ref(false)

const nodeIndexRef = shallowRef(null)
const edgeIndexRef = shallowRef(null)
const nodeListRef = shallowRef([])
const edgeListRef = shallowRef([])
const neighborEdgesByNodeRef = shallowRef(new Map())
const edgeByIdRef = shallowRef(new Map())
const nodeViewMap = new Map()
const edgeViewMap = new Map()
const streamTrackMap = new Map()

function getNodeIdentityIds(node) {
  const ids = []
  const push = (value) => {
    if (typeof value !== 'string' || !value) return
    if (!ids.includes(value)) {
      ids.push(value)
    }
  }

  push(node?.id)
  for (const lookupId of Array.isArray(node?.lookupIds) ? node.lookupIds : []) {
    push(lookupId)
  }
  push(node?.mappingKey)
  push(node?.actualNodeId)
  push(node?.focusOriginalNodeId)
  return ids
}

function nodeMatchesIdentity(node, targetId) {
  if (!node || typeof targetId !== 'string' || !targetId) return false
  return getNodeIdentityIds(node).includes(targetId)
}

function findMatchingNodes(targetId) {
  if (typeof targetId !== 'string' || !targetId) return []
  return nodeListRef.value.filter((node) => nodeMatchesIdentity(node, targetId))
}

const highlightEdgeIdSet = computed(() => {
  const ids = new Set()
  const neighbors = neighborEdgesByNodeRef.value
  const collect = (targetId) => {
    for (const node of findMatchingNodes(targetId)) {
      const list = neighbors.get(node.id)
      if (!list) continue
      for (const edgeId of list) ids.add(edgeId)
    }
  }
  if (hoveredNodeId.value) collect(hoveredNodeId.value)
  if (props.selectedNodeId) collect(props.selectedNodeId)
  return ids
})

let resizeObserver = null
let renderFrame = 0
let resizeFrame = 0
let resizeCommitFrame = 0
let syncFrame = 0
let dragState = null
let suppressNextClick = false
let highlightFollowFrame = 0
let highlightFollowToken = 0
let renderGeneration = 0

const logicalWidth = computed(() => Math.max(1, Math.round((props.graphLayout?.canvasWidth ?? 1) * props.zoomScale)))
const logicalHeight = computed(() => Math.max(1, Math.round((props.graphLayout?.canvasHeight ?? 1) * props.zoomScale)))
const spacerWidth = computed(() => Math.max(logicalWidth.value, viewportWidth.value || 0, 1))
const spacerHeight = computed(() => Math.max(logicalHeight.value, viewportHeight.value || 0, 1))

const screenX = (worldX) => (worldX * props.zoomScale) - scrollLeft.value
const screenY = (worldY) => (worldY * props.zoomScale) - scrollTop.value
const screenSize = (worldSize) => worldSize * props.zoomScale

onMounted(async () => {
  await ensureApp()
  await nextTick()
  updateViewportMetrics()
  rebuildSpatialIndex()
  await renderVisibleScene(++renderGeneration)
  await nextTick()
  followActiveContent()

  resizeObserver = new ResizeObserver(() => {
    scheduleViewportResizeRefresh()
  })
  if (shellRef.value) {
    resizeObserver.observe(shellRef.value)
  }
  if (viewportRef.value && viewportRef.value !== shellRef.value) {
    resizeObserver.observe(viewportRef.value)
  }
  window.addEventListener('resize', handleWindowResize, { passive: true })
  startViewportSyncLoop()
})

onBeforeUnmount(() => {
  renderGeneration += 1
  if (renderFrame) {
    cancelAnimationFrame(renderFrame)
    renderFrame = 0
  }
  if (resizeFrame) {
    cancelAnimationFrame(resizeFrame)
    resizeFrame = 0
  }
  if (resizeCommitFrame) {
    cancelAnimationFrame(resizeCommitFrame)
    resizeCommitFrame = 0
  }
  if (syncFrame) {
    cancelAnimationFrame(syncFrame)
    syncFrame = 0
  }
  if (highlightFollowFrame) {
    cancelAnimationFrame(highlightFollowFrame)
    highlightFollowFrame = 0
  }
  resizeObserver?.disconnect()
  window.removeEventListener('resize', handleWindowResize)
  destroyApp()
})

watch(
  () => props.graphLayout,
  async (newLayout, oldLayout) => {
    const hadContent = (oldLayout?.nodesFlat?.length ?? 0) > 0
    if (!hadContent) {
      resetSceneCaches()
    }
    rebuildSpatialIndex()
    await nextTick()
    if (!hadContent) {
      resetScrollPosition()
    }
    updateViewportMetrics()
    if (!hadContent) {
      followActiveContent()
    }
    queueRender()
  },
  { deep: false },
)

watch(
  () => props.zoomScale,
  async (nextZoom, previousZoom) => {
    preserveViewportCenterOnZoom(nextZoom, previousZoom)
    await nextTick()
    updateViewportMetrics()
    queueRender()
  },
)

watch(
  () => props.selectedNodeId,
  () => {
    updateNodeSelectionState()
    queueRender()
    scrollSelectedIntoView()
  },
)

watch(
  () => props.issueNodeIds,
  () => {
    updateNodeSelectionState()
    queueRender()
  },
)

watch(
  () => props.highlightedNodeIds,
  () => {
    updateNodeSelectionState()
    queueRender()
    scheduleHighlightedFollow()
  },
)

watch(
  () => [props.isFocusMode, props.focusNodeIds.join('|'), props.focusNodeId],
  () => {
    queueRender()
    scheduleHighlightedFollow()
  },
)

watch(
  highlightEdgeIdSet,
  () => {
    queueRender()
  },
)

watch(
  hoveredNodeId,
  () => {
    queueRender()
  },
)

function getRenderResolution(multiplier = 1) {
  const baseDpr = Math.max(1, devicePixelRatioRef.value || 1)
  const zoomCompensation = props.zoomScale < 1 ? 1 / Math.max(props.zoomScale, 0.5) : 1
  const resolution = baseDpr * multiplier * zoomCompensation
  return Math.min(3, Number(resolution.toFixed(2)))
}

function handleWindowResize() {
  const nextDpr = typeof window !== 'undefined' ? (window.devicePixelRatio || 1) : 1
  if (nextDpr !== devicePixelRatioRef.value) {
    devicePixelRatioRef.value = nextDpr
  }
  scheduleViewportResizeRefresh()
}

function scheduleViewportResizeRefresh() {
  if (resizeFrame) {
    cancelAnimationFrame(resizeFrame)
  }
  if (resizeCommitFrame) {
    cancelAnimationFrame(resizeCommitFrame)
  }
  resizeFrame = requestAnimationFrame(() => {
    resizeFrame = 0
    resizeCommitFrame = requestAnimationFrame(() => {
      resizeCommitFrame = 0
      updateViewportMetrics()
      queueRender()
    })
  })
}

function preserveViewportCenterOnZoom(nextZoom, previousZoom) {
  const viewport = viewportRef.value
  if (!viewport || !Number.isFinite(nextZoom) || !Number.isFinite(previousZoom) || previousZoom <= 0) return

  const worldCenterX = (viewport.scrollLeft + viewport.clientWidth / 2) / previousZoom
  const worldCenterY = (viewport.scrollTop + viewport.clientHeight / 2) / previousZoom
  const nextScrollLeft = Math.max(0, worldCenterX * nextZoom - viewport.clientWidth / 2)
  const nextScrollTop = Math.max(0, worldCenterY * nextZoom - viewport.clientHeight / 2)

  viewport.scrollLeft = nextScrollLeft
  viewport.scrollTop = nextScrollTop
  scrollLeft.value = nextScrollLeft
  scrollTop.value = nextScrollTop
}

async function ensureApp() {
  if (appRef.value) {
    return appRef.value
  }

  const app = new Application()
  await app.init({
    width: 1,
    height: 1,
    backgroundAlpha: 0,
    antialias: true,
    autoDensity: true,
    resolution: getRenderResolution(),
    roundPixels: true,
  })

  const rootScene = new Container()
  rootScene.roundPixels = true
  const streamLayer = new Container()
  const edgeLayer = new Container()
  const nodeLayer = new Container()
  const labelLayer = new Container()
  rootScene.addChild(streamLayer, edgeLayer, nodeLayer, labelLayer)
  app.stage.addChild(rootScene)

  rootSceneRef.value = rootScene
  streamLayerRef.value = streamLayer
  edgeLayerRef.value = edgeLayer
  nodeLayerRef.value = nodeLayer
  labelLayerRef.value = labelLayer

  hostRef.value?.appendChild(app.canvas)
  app.canvas.style.display = 'block'
  app.canvas.style.pointerEvents = 'none'
  appRef.value = app
  return app
}

function destroyApp() {
  resetSceneCaches()
  nodeIndexRef.value = null
  edgeIndexRef.value = null
  nodeListRef.value = []
  edgeListRef.value = []

  if (!appRef.value) {
    return
  }

  appRef.value.destroy({ removeView: true }, { children: true })
  appRef.value = null
  rootSceneRef.value = null
  streamLayerRef.value = null
  edgeLayerRef.value = null
  nodeLayerRef.value = null
  labelLayerRef.value = null
}

function resetSceneCaches() {
  nodeViewMap.forEach((container) => container.destroy({ children: true }))
  nodeViewMap.clear()

  edgeViewMap.forEach((graphic) => {
    graphic.line.destroy()
    graphic.arrow.destroy()
  })
  edgeViewMap.clear()

  streamTrackMap.forEach((track) => {
    track.background.destroy()
    track.labelShell.destroy()
    track.labelText.destroy()
  })
  streamTrackMap.clear()

  streamLayerRef.value?.removeChildren()
  edgeLayerRef.value?.removeChildren()
  nodeLayerRef.value?.removeChildren()
  labelLayerRef.value?.removeChildren()
}

function resetScrollPosition() {
  if (!viewportRef.value) {
    return
  }
  viewportRef.value.scrollLeft = 0
  viewportRef.value.scrollTop = 0
  scrollLeft.value = 0
  scrollTop.value = 0
}

function readElementPixelSize(element) {
  if (!element) return { width: 0, height: 0 }
  const rect = element.getBoundingClientRect()
  return {
    width: Math.round(element.clientWidth || element.offsetWidth || rect.width || 0),
    height: Math.round(element.clientHeight || element.offsetHeight || rect.height || 0),
  }
}

function resolveViewportPixelSize() {
  const viewportSize = readElementPixelSize(viewportRef.value)
  const shellSize = readElementPixelSize(shellRef.value)
  return {
    width: Math.max(1, viewportSize.width || shellSize.width),
    height: Math.max(1, viewportSize.height || shellSize.height),
  }
}

function updateViewportMetrics() {
  const size = resolveViewportPixelSize()
  viewportWidth.value = size.width
  viewportHeight.value = size.height
}

function handleScroll(event) {
  scrollLeft.value = event.target.scrollLeft
  scrollTop.value = event.target.scrollTop
  queueRender()
}

function syncViewportScrollState() {
  if (!viewportRef.value) return
  scrollLeft.value = viewportRef.value.scrollLeft
  scrollTop.value = viewportRef.value.scrollTop
}

function startViewportSyncLoop() {
  const tick = () => {
    const nextViewportSize = resolveViewportPixelSize()
    if (nextViewportSize.width !== viewportWidth.value || nextViewportSize.height !== viewportHeight.value) {
      viewportWidth.value = nextViewportSize.width
      viewportHeight.value = nextViewportSize.height
      queueRender()
    }
    if (viewportRef.value) {
      const nextLeft = viewportRef.value.scrollLeft
      const nextTop = viewportRef.value.scrollTop
      if (nextLeft !== scrollLeft.value || nextTop !== scrollTop.value) {
        scrollLeft.value = nextLeft
        scrollTop.value = nextTop
        queueRender()
      }
    }
    syncFrame = requestAnimationFrame(tick)
  }
  syncFrame = requestAnimationFrame(tick)
}

function viewportPointToWorld(clientX, clientY) {
  if (!viewportRef.value) return null
  const rect = viewportRef.value.getBoundingClientRect()
  const localX = clientX - rect.left + scrollLeft.value
  const localY = clientY - rect.top + scrollTop.value
  return {
    x: localX / props.zoomScale,
    y: localY / props.zoomScale,
  }
}

function hitTestNode(clientX, clientY) {
  const worldPoint = viewportPointToWorld(clientX, clientY)
  if (!worldPoint) return null

  const candidateNodes = BYPASS_FLATBUSH_RENDER
    ? nodeListRef.value
    : (nodeIndexRef.value?.search(worldPoint.x, worldPoint.y, worldPoint.x, worldPoint.y) ?? [])

  for (let index = candidateNodes.length - 1; index >= 0; index -= 1) {
    const node = BYPASS_FLATBUSH_RENDER
      ? candidateNodes[index]
      : nodeListRef.value[candidateNodes[index]]
    if (!node) continue
    if (
      worldPoint.x >= node.x &&
      worldPoint.x <= node.x + node.width &&
      worldPoint.y >= node.y &&
      worldPoint.y <= node.y + node.height
    ) {
      return node
    }
  }
  return null
}

function handleViewportPointerMove(event) {
  if (dragState?.active && dragState.pointerId === event.pointerId && viewportRef.value) {
    const nextLeft = Math.max(0, dragState.startScrollLeft - (event.clientX - dragState.startClientX))
    const nextTop = Math.max(0, dragState.startScrollTop - (event.clientY - dragState.startClientY))
    if (Math.abs(event.clientX - dragState.startClientX) > 3 || Math.abs(event.clientY - dragState.startClientY) > 3) {
      dragState.moved = true
      isDragging.value = true
    }
    viewportRef.value.scrollLeft = nextLeft
    viewportRef.value.scrollTop = nextTop
    scrollLeft.value = nextLeft
    scrollTop.value = nextTop
    hoveredNodeId.value = ''
    queueRender()
    return
  }

  const node = hitTestNode(event.clientX, event.clientY)
  hoveredNodeId.value = node?.id ?? ''
}

function handleViewportPointerLeave() {
  if (dragState?.active) return
  hoveredNodeId.value = ''
}

function handleViewportPointerDown(event) {
  if (!viewportRef.value) return
  if (event.button !== 0 && event.button !== 1) return
  if (isPointerOnNativeScrollbar(event)) return

  dragState = {
    active: true,
    pointerId: event.pointerId,
    captureTarget: event.currentTarget,
    startClientX: event.clientX,
    startClientY: event.clientY,
    startScrollLeft: viewportRef.value.scrollLeft,
    startScrollTop: viewportRef.value.scrollTop,
    moved: false,
  }
  suppressNextClick = false
  event.currentTarget?.setPointerCapture?.(event.pointerId)
}

function isPointerOnNativeScrollbar(event) {
  const viewport = viewportRef.value
  if (!viewport) return false

  const rect = viewport.getBoundingClientRect()
  const verticalScrollbarWidth = Math.max(0, viewport.offsetWidth - viewport.clientWidth)
  const horizontalScrollbarHeight = Math.max(0, viewport.offsetHeight - viewport.clientHeight)
  const scrollableX = viewport.scrollWidth > viewport.clientWidth + 1
  const scrollableY = viewport.scrollHeight > viewport.clientHeight + 1
  const edgeThreshold = 14
  const onVerticalScrollbar = verticalScrollbarWidth > 0
    ? event.clientX >= rect.right - verticalScrollbarWidth - 1
    : scrollableX && event.clientX >= rect.right - edgeThreshold
  const onHorizontalScrollbar = horizontalScrollbarHeight > 0
    ? event.clientY >= rect.bottom - horizontalScrollbarHeight - 1
    : scrollableY && event.clientY >= rect.bottom - edgeThreshold

  return onVerticalScrollbar || onHorizontalScrollbar
}

function handleViewportPointerUp(event) {
  if (!dragState || dragState.pointerId !== event.pointerId) return
  dragState.captureTarget?.releasePointerCapture?.(event.pointerId)
  suppressNextClick = dragState.moved
  isDragging.value = false
  dragState = null
}

function handleViewportPointerCancel(event) {
  if (!dragState || dragState.pointerId !== event.pointerId) return
  dragState.captureTarget?.releasePointerCapture?.(event.pointerId)
  isDragging.value = false
  dragState = null
}

function handleViewportClick(event) {
  if (suppressNextClick) {
    suppressNextClick = false
    return
  }
  const node = hitTestNode(event.clientX, event.clientY)
  if (node) emit('node-select', node)
}

function queueRender() {
  const generation = ++renderGeneration
  if (renderFrame) {
    cancelAnimationFrame(renderFrame)
  }
  renderFrame = requestAnimationFrame(() => {
    renderFrame = 0
    renderVisibleScene(generation)
  })
}

function scheduleHighlightedFollow() {
  highlightFollowToken += 1
  const token = highlightFollowToken
  if (highlightFollowFrame) {
    cancelAnimationFrame(highlightFollowFrame)
  }
  highlightFollowFrame = requestAnimationFrame(() => {
    highlightFollowFrame = 0
    if (token !== highlightFollowToken) return
    scrollHighlightedIntoView()
  })
}

function rebuildSpatialIndex() {
  const nodes = []
  const edges = []
  const nodeCount = props.graphLayout.nodesFlat.filter((node) => !node?.isPlaceholder).length
  const edgeCount = props.graphLayout.edges.length

  if (!nodeCount) {
    nodeListRef.value = []
    edgeListRef.value = []
    nodeIndexRef.value = null
    edgeIndexRef.value = null
    neighborEdgesByNodeRef.value = new Map()
    edgeByIdRef.value = new Map()
    return
  }

  const nodeIndex = new Flatbush(nodeCount)
  const edgeIndex = edgeCount ? new Flatbush(edgeCount) : null

  props.graphLayout.nodesFlat.forEach((node) => {
    if (node?.isPlaceholder) {
      return
    }
    nodes.push(node)
    nodeIndex.add(node.minX, node.minY, node.maxX, node.maxY)
  })
  nodeIndex.finish()

  const neighborMap = new Map()
  const edgeLookup = new Map()
  props.graphLayout.edges.forEach((edge) => {
    edges.push(edge)
    edgeIndex?.add(edge.minX, edge.minY, edge.maxX, edge.maxY)
    edgeLookup.set(edge.id, edge)
    const [srcId, dstId] = edge.id.split('->')
    if (srcId) {
      const list = neighborMap.get(srcId) ?? []
      list.push(edge.id)
      neighborMap.set(srcId, list)
    }
    if (dstId) {
      const list = neighborMap.get(dstId) ?? []
      list.push(edge.id)
      neighborMap.set(dstId, list)
    }
  })
  edgeIndex?.finish()

  nodeListRef.value = nodes
  edgeListRef.value = edges
  nodeIndexRef.value = nodeIndex
  edgeIndexRef.value = edgeIndex
  neighborEdgesByNodeRef.value = neighborMap
  edgeByIdRef.value = edgeLookup
}

async function renderVisibleScene(generation = renderGeneration) {
  const app = await ensureApp()
  if (generation !== renderGeneration || !appRef.value || !rootSceneRef.value || !hostRef.value) {
    return
  }
  syncViewportScrollState()
  const viewportSize = resolveViewportPixelSize()
  const renderWidth = Math.max(
    1,
    BYPASS_VIEWPORT_WINDOW ? logicalWidth.value : viewportSize.width,
  )
  const renderHeight = Math.max(
    1,
    BYPASS_VIEWPORT_WINDOW ? logicalHeight.value : viewportSize.height,
  )
  viewportWidth.value = viewportSize.width
  viewportHeight.value = viewportSize.height
  const resolution = getRenderResolution()

  await app.renderer.resize(renderWidth, renderHeight, resolution)
  if (generation !== renderGeneration || !appRef.value || !rootSceneRef.value || !hostRef.value || !app.canvas) {
    return
  }

  const host = hostRef.value
  host.style.left = '0'
  host.style.top = '0'
  host.style.transform = 'none'
  host.style.width = '100%'
  host.style.height = '100%'
  app.canvas.style.width = '100%'
  app.canvas.style.height = '100%'

  const worldLeft = BYPASS_VIEWPORT_WINDOW ? 0 : scrollLeft.value / props.zoomScale
  const worldTop = BYPASS_VIEWPORT_WINDOW ? 0 : scrollTop.value / props.zoomScale
  const worldRight = BYPASS_VIEWPORT_WINDOW ? (renderWidth / props.zoomScale) : (scrollLeft.value + renderWidth) / props.zoomScale
  const worldBottom = BYPASS_VIEWPORT_WINDOW ? (renderHeight / props.zoomScale) : (scrollTop.value + renderHeight) / props.zoomScale
  const queryLeft = Math.max(0, worldLeft - VIEWPORT_OVERSCAN / props.zoomScale)
  const queryTop = Math.max(0, worldTop - VIEWPORT_OVERSCAN / props.zoomScale)
  const queryRight = worldRight + VIEWPORT_OVERSCAN / props.zoomScale
  const queryBottom = worldBottom + VIEWPORT_OVERSCAN / props.zoomScale

  rootSceneRef.value.position.set(0, 0)
  rootSceneRef.value.scale.set(1)

  updateVisibleStreams(queryTop, queryBottom)
  updateVisibleEdges(queryLeft, queryTop, queryRight, queryBottom)
  updateVisibleNodes(queryLeft, queryTop, queryRight, queryBottom)
  updateNodeSelectionState()
  app.render()
}

function updateVisibleStreams(viewTop, viewBottom) {
  const activeTrackIds = new Set()
  const streamLayer = streamLayerRef.value
  const labelLayer = labelLayerRef.value
  const minimumTrackWidth = Math.max(
    props.graphLayout.trackWidth,
    (viewportWidth.value || 0) / Math.max(props.zoomScale, 0.001),
  )

  props.graphLayout.streamRows.forEach((stream) => {
    if (!BYPASS_VIEWPORT_WINDOW && (stream.trackY > viewBottom || stream.trackY + stream.trackHeight < viewTop)) {
      return
    }

    const streamKey = stream.streamKey ?? stream.streamId
    activeTrackIds.add(streamKey)

    let track = streamTrackMap.get(streamKey)
    if (!track) {
      track = createStreamTrackView(stream)
      streamTrackMap.set(streamKey, track)
      streamLayer.addChild(track.background)
      labelLayer?.addChild(track.labelShell, track.labelText)
    }

    const trackWidth = Math.max(props.graphLayout.trackWidth, minimumTrackWidth - stream.trackX - 8)
    const trackX = screenX(stream.trackX)
    const trackY = screenY(stream.trackY)
    const scaledTrackWidth = screenSize(trackWidth)
    const scaledTrackHeight = screenSize(stream.trackHeight)
    const labelShellX = screenX(STREAM_LABEL_FIXED_X)
    const labelShellY = screenY(stream.trackY + 2)
    const labelShellWidth = screenSize(Math.max(STREAM_LABEL_MIN_WIDTH, stream.trackX - STREAM_LABEL_LEFT_INSET - STREAM_LABEL_RIGHT_GAP))
    const labelShellHeight = screenSize(Math.min(stream.trackHeight - 4, STREAM_LABEL_BAR_HEIGHT))

    track.background.clear()
    track.background.roundRect(trackX, trackY, scaledTrackWidth, scaledTrackHeight, screenSize(8))
    track.background.fill({ color: stream.streamIndex % 2 === 0 ? 0x181d25 : 0x141922, alpha: 0.96 })
    track.background.stroke({ color: 0x2f4c73, alpha: 0.18, width: 1 })

    track.labelShell.clear()
    track.labelShell.roundRect(labelShellX, labelShellY, labelShellWidth, labelShellHeight, screenSize(8))
    track.labelShell.fill({ color: 0x101722, alpha: 1 })
    track.labelShell.stroke({ color: 0x456d9d, alpha: 0.4, width: 1 })

    track.labelText.text = stream.label ?? `Rank ${stream.rankId ?? '-'} / Stream ${stream.streamId}`
    track.labelText.scale.set(props.zoomScale / STREAM_LABEL_RENDER_SCALE)
    track.labelText.x = labelShellX + 8
    track.labelText.y = labelShellY + Math.max(2, Math.round((labelShellHeight - track.labelText.height / props.zoomScale) / 2))
  })

  streamTrackMap.forEach((track, streamId) => {
    const visible = activeTrackIds.has(streamId)
    track.background.visible = visible
    track.labelShell.visible = visible
    track.labelText.visible = visible
  })
}

function createStreamTrackView(stream) {
  const background = new Graphics()
  const labelShell = new Graphics()
  const labelText = new Text({
    text: stream.label ?? `Rank ${stream.rankId ?? '-'} / Stream ${stream.streamId}`,
    resolution: getRenderResolution(STREAM_LABEL_RENDER_SCALE),
    roundPixels: true,
    style: {
      fill: 0xd8e9ff,
      fontSize: 10 * STREAM_LABEL_RENDER_SCALE,
      fontWeight: '700',
      fontFamily: 'Segoe UI, PingFang SC, Microsoft YaHei, sans-serif',
      wordWrap: true,
      wordWrapWidth: Math.max(STREAM_LABEL_MIN_WIDTH - 16, stream.trackX - 32) * STREAM_LABEL_RENDER_SCALE,
      breakWords: true,
      lineHeight: 11 * STREAM_LABEL_RENDER_SCALE,
      padding: STREAM_LABEL_RENDER_SCALE,
    },
  })
  labelText.scale.set(1 / STREAM_LABEL_RENDER_SCALE)
  return { background, labelShell, labelText }
}

function updateVisibleEdges(viewLeft, viewTop, viewRight, viewBottom) {
  const edgeLayer = edgeLayerRef.value
  const activeEdgeIds = new Set()
  const highlightSet = highlightEdgeIdSet.value

  const edgeQueue = BYPASS_FLATBUSH_RENDER
    ? [...edgeListRef.value]
    : (edgeIndexRef.value?.search(viewLeft, viewTop, viewRight, viewBottom) ?? []).reduce((result, edgeIndex) => {
      const edge = edgeListRef.value[edgeIndex]
      if (edge) result.push(edge)
      return result
    }, [])

  if (highlightSet.size) {
    const lookup = edgeByIdRef.value
    highlightSet.forEach((edgeId) => {
      if (!activeEdgeIds.has(edgeId)) {
        const edge = lookup.get(edgeId)
        if (edge) edgeQueue.push(edge)
      }
    })
  }

  edgeQueue.sort((a, b) => Number(highlightSet.has(a.id)) - Number(highlightSet.has(b.id)))

  edgeQueue.forEach((edge) => {
    activeEdgeIds.add(edge.id)
    const isHighlighted = highlightSet.has(edge.id)

    let edgeGraphic = edgeViewMap.get(edge.id)
    if (!edgeGraphic) {
      edgeGraphic = {
        line: new Graphics(),
        arrow: new Graphics(),
      }
      edgeViewMap.set(edge.id, edgeGraphic)
      edgeLayer.addChild(edgeGraphic.line, edgeGraphic.arrow)
    }

    if (isHighlighted) {
      edgeLayer.setChildIndex(edgeGraphic.line, edgeLayer.children.length - 1)
      edgeLayer.setChildIndex(edgeGraphic.arrow, edgeLayer.children.length - 1)
    }

    edgeGraphic.line.visible = true
    edgeGraphic.line.clear()
    edgeGraphic.line.moveTo(screenX(edge.startX), screenY(edge.startY))
    if (edge.kind === 'line') {
      edgeGraphic.line.lineTo(screenX(edge.endX), screenY(edge.endY))
    } else {
      edgeGraphic.line.bezierCurveTo(
        screenX(edge.cp1x),
        screenY(edge.cp1y),
        screenX(edge.cp2x),
        screenY(edge.cp2y),
        screenX(edge.endX),
        screenY(edge.endY),
      )
    }

    const baseColor = edge.crossStream ? 0xf5b04f : 0x69a9ff
    const strokeColor = isHighlighted ? 0xfff1a8 : baseColor
    const strokeAlpha = isHighlighted ? 0.92 : edge.crossStream ? 0.2 : 0.24
    const strokeWidth = (isHighlighted ? 1.8 : edge.crossStream ? 1 : 1.1) * props.zoomScale

    edgeGraphic.line.stroke({
      color: strokeColor,
      alpha: strokeAlpha,
      width: strokeWidth,
      cap: 'round',
      join: 'round',
    })

    const arrowLength = isHighlighted ? 6 : edge.crossStream ? 5 : 4
    const arrowHalf = isHighlighted ? 2.6 : edge.crossStream ? 2.1 : 1.8

    let angle
    if (edge.kind === 'line') {
      angle = Math.atan2(edge.endY - edge.startY, edge.endX - edge.startX)
    } else {
      const tangent = bezierTangentAt(
        edge.cp1x, edge.cp1y, edge.cp2x, edge.cp2y,
        edge.endX, edge.endY, edge.startX, edge.startY,
        0.95,
      )
      angle = Math.atan2(tangent.dy, tangent.dx)
    }

    edgeGraphic.arrow.visible = true
    edgeGraphic.arrow.clear()
    const endX = screenX(edge.endX)
    const endY = screenY(edge.endY)
    const scaledArrowLength = arrowLength * props.zoomScale
    const scaledArrowHalf = arrowHalf * props.zoomScale
    edgeGraphic.arrow.moveTo(endX, endY)
    edgeGraphic.arrow.lineTo(
      endX - Math.cos(angle) * scaledArrowLength + Math.sin(angle) * scaledArrowHalf,
      endY - Math.sin(angle) * scaledArrowLength - Math.cos(angle) * scaledArrowHalf,
    )
    edgeGraphic.arrow.lineTo(
      endX - Math.cos(angle) * scaledArrowLength - Math.sin(angle) * scaledArrowHalf,
      endY - Math.sin(angle) * scaledArrowLength + Math.cos(angle) * scaledArrowHalf,
    )
    edgeGraphic.arrow.closePath()
    edgeGraphic.arrow.fill({
      color: strokeColor,
      alpha: isHighlighted ? 0.95 : edge.crossStream ? 0.3 : 0.26,
    })
  })

  edgeViewMap.forEach((graphic, edgeId) => {
    const visible = activeEdgeIds.has(edgeId)
    graphic.line.visible = visible
    graphic.arrow.visible = visible
  })
}

function updateVisibleNodes(viewLeft, viewTop, viewRight, viewBottom) {
  const nodeLayer = nodeLayerRef.value
  const activeNodeIds = new Set()

  const nodesToRender = BYPASS_FLATBUSH_RENDER
    ? nodeListRef.value.filter((node) => !node?.isPlaceholder)
    : (nodeIndexRef.value?.search(viewLeft, viewTop, viewRight, viewBottom) ?? []).map(
      (nodeIndex) => nodeListRef.value[nodeIndex],
    )

  nodesToRender.forEach((node) => {
    if (!node) {
      return
    }
    if (node.isPlaceholder) {
      return
    }
    activeNodeIds.add(node.id)

    let container = nodeViewMap.get(node.id)
    if (!container) {
      container = createNodeView(node)
      nodeViewMap.set(node.id, container)
      nodeLayer.addChild(container)
    }

    container.visible = true
    container.x = Math.round(screenX(node.x))
    container.y = Math.round(screenY(node.y))
    container.scale.set(props.zoomScale)
  })

  nodeViewMap.forEach((container, nodeId) => {
    container.visible = activeNodeIds.has(nodeId)
  })
}

function createNodeView(node) {
  const container = new Container()
  container.x = Math.round(node.x)
  container.y = Math.round(node.y)
  container.roundPixels = true
  container.eventMode = 'static'
  container.cursor = 'pointer'
  container.hitArea = new Rectangle(0, 0, node.width, node.height)

  const shadow = new Graphics()
  shadow.y = 1
  const body = new Graphics()
  const issueMarker = new Graphics()
  container.addChild(shadow, body, issueMarker)

  const text = new Text({
    text: node.label,
    resolution: getRenderResolution(NODE_LABEL_RENDER_SCALE),
    roundPixels: true,
    style: {
      fill: 0xf3f7ff,
      fontSize: 9 * NODE_LABEL_RENDER_SCALE,
      fontWeight: '700',
      fontFamily: 'Segoe UI, PingFang SC, Microsoft YaHei, sans-serif',
      wordWrap: true,
      wordWrapWidth: (node.width - 10) * NODE_LABEL_RENDER_SCALE,
      breakWords: true,
      align: 'left',
      lineHeight: 10 * NODE_LABEL_RENDER_SCALE,
      padding: NODE_LABEL_RENDER_SCALE,
      letterSpacing: 0.2 * NODE_LABEL_RENDER_SCALE,
    },
  })
  text.scale.set(1 / NODE_LABEL_RENDER_SCALE)
  text.x = 6
  text.y = Math.max(4, Math.round((node.height - text.height / NODE_LABEL_RENDER_SCALE) / 2))
  container.addChild(text)

  container.__nodeMeta = { body, shadow, issueMarker, node }

  applyNodeState(node.id, false)
  return container
}

function updateNodeSelectionState() {
  nodeViewMap.forEach((_, nodeId) => {
    applyNodeState(nodeId, hoveredNodeId.value === nodeId)
  })
}

function applyNodeState(nodeId, hovered) {
  const container = nodeViewMap.get(nodeId)
  if (!container?.__nodeMeta) {
    return
  }

  const { body, shadow, issueMarker, node } = container.__nodeMeta
  const nodeIdentityIds = getNodeIdentityIds(node)
  const isSelected = nodeIdentityIds.includes(props.selectedNodeId)
  const isIssueHighlighted = Boolean(
    props.issueNodeIds && nodeIdentityIds.some((identityId) => props.issueNodeIds.has?.(identityId)),
  )
  const isStepHighlighted = Boolean(
    props.highlightedNodeIds && nodeIdentityIds.some((identityId) => props.highlightedNodeIds.has?.(identityId)),
  )
  const hasSubgraph = Boolean(node.hasSubgraph)
  const fillColor = isIssueHighlighted ? 0x4a1820 : hasSubgraph ? 0x4d3410 : isStepHighlighted ? 0x1f3d62 : 0x173052
  const fillAlpha = hovered || isStepHighlighted || isIssueHighlighted ? 0.96 : 0.82
  const strokeColor = isIssueHighlighted
    ? 0xff5d6c
    : isSelected
      ? 0x9bd1ff
      : isStepHighlighted
        ? 0xfff1a8
        : hovered
          ? 0x7ec0ff
          : hasSubgraph
            ? 0xe3a64e
            : 0x4e82bc
  const strokeWidth = isIssueHighlighted ? 2.6 : isSelected ? 2.2 : isStepHighlighted ? 1.8 : hovered ? 1.6 : 1
  const glowColor = isIssueHighlighted
    ? 0xff3648
    : isSelected
      ? 0x67a9ff
      : isStepHighlighted
        ? 0xf5b04f
        : hovered
          ? 0x4d9fff
          : 0x000000

  body.clear()
  body.roundRect(0, 0, node.width, node.height, 8)
  body.fill({ color: fillColor, alpha: fillAlpha })
  body.stroke({ color: strokeColor, alpha: 0.95, width: strokeWidth })

  shadow.clear()
  shadow.roundRect(-2, -2, node.width + 4, node.height + 4, 10)
  shadow.fill({
    color: glowColor,
    alpha: isIssueHighlighted ? 0.26 : isSelected ? 0.2 : isStepHighlighted ? 0.22 : hovered ? 0.14 : 0.08,
  })
  shadow.y = 1

  issueMarker.visible = isIssueHighlighted
  issueMarker.clear()
  if (isIssueHighlighted) {
    const markerX = Math.max(10, node.width - 10)
    const markerY = 10
    issueMarker.circle(markerX, markerY, 5)
    issueMarker.fill({ color: 0xff5d6c, alpha: 1 })
    issueMarker.stroke({ color: 0xffffff, alpha: 0.96, width: 1.5 })
    issueMarker.moveTo(markerX, markerY - 2)
    issueMarker.lineTo(markerX, markerY + 1)
    issueMarker.moveTo(markerX, markerY + 3)
    issueMarker.lineTo(markerX, markerY + 3.2)
    issueMarker.stroke({ color: 0xffffff, alpha: 0.96, width: 1.2, cap: 'round' })
  }
}

function scrollSelectedIntoView() {
  if (!props.selectedNodeId || !viewportRef.value) return
  if (props.isFocusMode && (props.focusNodeIds.length || props.focusNodeId)) return
  const node = findMatchingNodes(props.selectedNodeId)[0] ?? null
  if (!node) return

  scrollNodeIntoView(node, { smooth: false, align: 'center' })
}

function scrollHighlightedIntoView() {
  const focusIds = props.focusNodeIds.length ? props.focusNodeIds : (props.focusNodeId ? [props.focusNodeId] : [])
  if (props.isFocusMode && focusIds.length) {
    const nodes = focusIds.flatMap((nodeId) => findMatchingNodes(nodeId))
    if (nodes.length) {
      scrollBoundsIntoView(getNodeBounds(nodes), { smooth: false, align: 'center' })
      return
    }
  }

  const highlighted = props.highlightedNodeIds
  if (!highlighted?.size || !viewportRef.value) return
  const nodes = nodeListRef.value.filter((item) => highlighted.has(item.id))
  if (!nodes.length) return
  scrollBoundsIntoView(getNodeBounds(nodes), { smooth: false, align: 'center' })
}

function followActiveContent() {
  const focusIds = props.focusNodeIds.length ? props.focusNodeIds : (props.focusNodeId ? [props.focusNodeId] : [])
  if (props.isFocusMode && focusIds.length) {
    const nodes = focusIds.flatMap((nodeId) => findMatchingNodes(nodeId))
    if (nodes.length) {
      scrollBoundsIntoView(getNodeBounds(nodes), { smooth: false, align: 'center' })
      return
    }
  }

  if (props.selectedNodeId) {
    scrollSelectedIntoView()
    return
  }

  if (props.highlightedNodeIds?.size) {
    scheduleHighlightedFollow()
    return
  }

  scrollDefaultContentIntoView()
}

function scrollDefaultContentIntoView() {
  const host = viewportRef.value
  if (!host || !nodeListRef.value.length) return

  const firstNode = [...nodeListRef.value].sort((left, right) => (
    left.y - right.y ||
    left.x - right.x ||
    String(left.id).localeCompare(String(right.id))
  ))[0]

  if (!firstNode) return

  const zoom = props.zoomScale || 1
  const anchorX = Math.min(145, Math.max(72, host.clientWidth * 0.38))
  const anchorY = Math.min(80, Math.max(48, host.clientHeight * 0.42))
  const nodeCenterX = (firstNode.x + firstNode.width / 2) * zoom
  const nodeCenterY = (firstNode.y + firstNode.height / 2) * zoom
  const maxLeft = Math.max(0, host.scrollWidth - host.clientWidth)
  const maxTop = Math.max(0, host.scrollHeight - host.clientHeight)
  const nextLeft = Math.min(maxLeft, Math.max(0, nodeCenterX - anchorX))
  const nextTop = Math.min(maxTop, Math.max(0, nodeCenterY - anchorY))

  if (nextLeft !== host.scrollLeft || nextTop !== host.scrollTop) {
    host.scrollLeft = nextLeft
    host.scrollTop = nextTop
    syncViewportScrollState()
    queueRender()
  }
}

function scrollNodeIntoView(node, { smooth = true, align = 'contain' } = {}) {
  if (!node) return
  scrollBoundsIntoView(getNodeBounds([node]), { smooth, align })
}

function getNodeBounds(nodes) {
  if (!Array.isArray(nodes) || !nodes.length) return null
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity

  for (const node of nodes) {
    if (!node) continue
    minX = Math.min(minX, node.x)
    minY = Math.min(minY, node.y)
    maxX = Math.max(maxX, node.x + node.width)
    maxY = Math.max(maxY, node.y + node.height)
  }

  if (!Number.isFinite(minX) || !Number.isFinite(minY) || !Number.isFinite(maxX) || !Number.isFinite(maxY)) {
    return null
  }

  return { minX, minY, maxX, maxY }
}

function scrollBoundsIntoView(bounds, { smooth = true, align = 'contain' } = {}) {
  if (!bounds || !viewportRef.value) return
  const host = viewportRef.value
  const zoom = props.zoomScale || 1
  const targetX = bounds.minX * zoom
  const targetY = bounds.minY * zoom
  const targetRight = bounds.maxX * zoom
  const targetBottom = bounds.maxY * zoom
  const targetWidth = Math.max(1, targetRight - targetX)
  const targetHeight = Math.max(1, targetBottom - targetY)
  const behavior = smooth ? 'smooth' : 'auto'

  let nextLeft = host.scrollLeft
  let nextTop = host.scrollTop

  if (align === 'center') {
    nextLeft = Math.max(0, targetX - host.clientWidth / 2 + targetWidth / 2)
    nextTop = Math.max(0, targetY - host.clientHeight / 2 + targetHeight / 2)
  } else {
    if (targetX < host.scrollLeft + 24 || targetRight > host.scrollLeft + host.clientWidth - 24) {
      nextLeft = Math.max(0, targetX - 48)
    }
    if (targetY < host.scrollTop + 20 || targetBottom > host.scrollTop + host.clientHeight - 20) {
      nextTop = Math.max(0, targetY - 40)
    }
  }

  if (nextLeft !== host.scrollLeft || nextTop !== host.scrollTop) {
    host.scrollTo({ left: nextLeft, top: nextTop, behavior })
    syncViewportScrollState()
    queueRender()
  }
}
</script>

<template>
  <div
    ref="shellRef"
    class="memview-dag-pixi-shell"
    :class="{ 'is-dragging': isDragging }"
    @pointerdown="handleViewportPointerDown"
    @pointermove="handleViewportPointerMove"
    @pointerup="handleViewportPointerUp"
    @pointercancel="handleViewportPointerCancel"
    @pointerleave="handleViewportPointerLeave"
    @click="handleViewportClick"
  >
    <div
      ref="viewportRef"
      class="memview-dag-pixi-viewport"
      @scroll="handleScroll"
    >
      <div
        class="memview-dag-pixi-spacer"
        :style="{
          width: `${spacerWidth}px`,
          height: `${spacerHeight}px`,
        }"
      />
    </div>
    <div ref="hostRef" class="memview-dag-pixi memview-dag-pixi-overlay" />
  </div>
</template>

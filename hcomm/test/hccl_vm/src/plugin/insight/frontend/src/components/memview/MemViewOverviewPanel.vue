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
import { computed, onBeforeUnmount, ref, shallowRef, triggerRef, watch } from 'vue'
import { DataLine, ZoomIn, ZoomOut } from '@element-plus/icons-vue'
import { fetchMsgpack, parseGraphRank } from '../../utils/insightData'
import {
  BUFFER_SLOTS,
  buildMemviewIndex,
  bufferOpsAtStep,
  rankColorFor,
  taskOpsAtStep,
} from '../../utils/memviewIndex.js'
import { postWorker } from '../../workers/insightDataClient.js'
import { useMemViewTimeline } from '../../composables/useMemViewTimeline'
import MemViewTimelineCanvas from './MemViewTimelineCanvas.vue'

const ARROW_WINDOW = 0 // Only current step ops drawn, simpler + fewer arrows

const props = defineProps({
  datasetName: {
    type: String,
    default: '',
  },
  currentStep: {
    type: Number,
    default: 0,
  },
  selectedRankKeys: {
    type: Array,
    required: true,
  },
  displayStepItems: {
    type: Array,
    default: () => [],
  },
  emptyText: {
    type: String,
    default: 'No data',
  },
})

const emit = defineEmits(['update:totalSteps', 'update:currentStep', 'buffer-select'])

const { setIndexes } = useMemViewTimeline()

const loading = ref(false)
const errorMessage = ref('')
const manifestsLoaded = ref(false)
const rankBundles = shallowRef([])
const taskNodeLookup = shallowRef(new Map())
const indexes = shallowRef(null)
const canvasRef = ref(null)

const pendingChunkRequests = new Map()
let indexBuildHandle = 0
let loadToken = 0

const selectedRankIds = computed(() =>
  props.selectedRankKeys
    .map((key) => {
      const match = String(key).match(/^rank-(\d+)$/)
      return match ? Number(match[1]) : null
    })
    .filter((rankId) => Number.isInteger(rankId))
    .sort((a, b) => a - b),
)

const totalSteps = computed(() => {
  let max = 0
  for (const bundle of rankBundles.value) {
    for (const chunk of bundle.manifest?.chunks ?? []) {
      if (chunk.logicalStepEnd + 1 > max) max = chunk.logicalStepEnd + 1
    }
  }
  return max
})

const hasContent = computed(
  () =>
    manifestsLoaded.value &&
    totalSteps.value > 0 &&
    rankBundles.value.length > 0 &&
    rankBundles.value.some((bundle) => bundle.loadedChunks.size > 0),
)

const activeStep = computed(() => Math.min(Math.max(props.currentStep, 0), Math.max(0, totalSteps.value - 1)))

const currentStepBufferOps = computed(() => {
  if (!indexes.value) return []
  if (ARROW_WINDOW === 0) return bufferOpsAtStep(indexes.value, activeStep.value)
  const result = []
  for (let offset = -ARROW_WINDOW; offset <= ARROW_WINDOW; offset += 1) {
    const step = activeStep.value + offset
    if (step < 0 || step >= totalSteps.value) continue
    result.push(...bufferOpsAtStep(indexes.value, step))
  }
  return result
})

const currentStepTaskOps = computed(() => {
  if (!indexes.value) return []
  return taskOpsAtStep(indexes.value, activeStep.value)
})

const visibleBufferSlots = computed(() => {
  if (!indexes.value) return null

  const usedBufferIds = new Set()
  const rankIdSet = new Set(selectedRankIds.value)
  const presenceMap = indexes.value.bufferPresence

  if (presenceMap instanceof Map) {
    const ranksToScan = rankIdSet.size ? selectedRankIds.value : [...presenceMap.keys()]
    for (const rankId of ranksToScan) {
      const presence = presenceMap.get(rankId)
      if (!presence) continue
      for (const mask of presence) {
        if (!mask) continue
        for (const slot of BUFFER_SLOTS) {
          if ((mask & (1 << slot.id)) !== 0) {
            usedBufferIds.add(slot.id)
          }
        }
      }
    }
  }

  const layoutItems = indexes.value.layoutBuffers?.items
  if (Array.isArray(layoutItems)) {
    for (const item of layoutItems) {
      if (!Number.isInteger(item?.bufferId)) continue
      if (rankIdSet.size && !rankIdSet.has(item.rankId)) continue
      usedBufferIds.add(item.bufferId)
    }
  }

  return BUFFER_SLOTS.filter((slot) => usedBufferIds.has(slot.id))
})

const legendItems = computed(() => {
  const rankOrder = selectedRankIds.value
  return rankOrder.map((rankId) => ({
    id: rankId,
    label: `R${rankId}`,
    color: rankColorFor(rankId, rankOrder),
  }))
})

const loadingLabel = computed(() => {
  if (loading.value && !manifestsLoaded.value) return 'Loading manifests...'
  if (loading.value) return 'Streaming chunks...'
  return `${rankBundles.value.length} ranks · ${totalSteps.value} steps`
})

watch(
  () => [props.datasetName, selectedRankIds.value.join(',')],
  async () => {
    await loadManifests()
  },
  { immediate: true },
)

watch(
  totalSteps,
  (value) => {
    emit('update:totalSteps', value)
  },
  { immediate: true },
)

watch(
  () => [props.datasetName, selectedRankIds.value.join(','), activeStep.value, manifestsLoaded.value],
  () => {
    ensureVisibleChunksLoaded()
  },
)

onBeforeUnmount(() => {
  pendingChunkRequests.clear()
  loadToken += 1
})

async function loadManifests() {
  const datasetName = props.datasetName
  const rankIds = selectedRankIds.value
  const token = ++loadToken

  pendingChunkRequests.clear()
  manifestsLoaded.value = false
  errorMessage.value = ''

  if (!datasetName || !rankIds.length) {
    rankBundles.value = []
    taskNodeLookup.value = new Map()
    indexes.value = null
    setIndexes(null)
    loading.value = false
    return
  }

  loading.value = true

  try {
    const bundles = await Promise.all(
      rankIds.map(async (rankId) => {
        const manifestRequest = fetchMsgpack({
          dataset: datasetName,
          kind: 'memory',
          rank: rankId,
          file: 'manifest.msgpack',
        }).then((buffer) => postWorker('decodeMemoryManifest', { buffer }, [buffer]))

        const graphRequest = fetchMsgpack({
          dataset: datasetName,
          kind: 'graph',
          group: 'input_graph',
          rank: rankId,
        })
          .then((buffer) => parseGraphRank(buffer))
          .catch(() => null)

        const [manifest, graphRank] = await Promise.all([manifestRequest, graphRequest])
        return {
          rankId,
          manifest,
          loadedChunks: new Map(),
          taskNodes: flattenTaskNodesFromGraphRank(graphRank, rankId),
        }
      }),
    )

    if (token !== loadToken) return

    rankBundles.value = bundles.sort((a, b) => a.rankId - b.rankId)
    taskNodeLookup.value = buildTaskNodeLookup(rankBundles.value)
    manifestsLoaded.value = true
  } catch (error) {
    if (token !== loadToken) return
    rankBundles.value = []
    taskNodeLookup.value = new Map()
    errorMessage.value = error instanceof Error ? error.message : 'Failed to load memory manifests.'
  } finally {
    if (token === loadToken) loading.value = false
  }
}

function ensureVisibleChunksLoaded() {
  const datasetName = props.datasetName
  if (!datasetName || !manifestsLoaded.value) return

  const windowSize = Math.max(200, Math.ceil(totalSteps.value / 20))
  const startStep = Math.max(0, activeStep.value - windowSize)
  const endStep = Math.min(totalSteps.value - 1, activeStep.value + windowSize)

  rankBundles.value.forEach((bundle) => {
    for (const chunk of bundle.manifest.chunks) {
      if (chunk.logicalStepEnd < startStep || chunk.logicalStepBegin > endStep) continue
      if (bundle.loadedChunks.has(chunk.chunkId)) continue
      loadChunk(datasetName, bundle.rankId, chunk)
    }
  })
}

async function loadChunk(datasetName, rankId, chunk) {
  const requestKey = `${datasetName}:${rankId}:${chunk.chunkId}`
  if (pendingChunkRequests.has(requestKey)) return pendingChunkRequests.get(requestKey)

  loading.value = true
  const token = loadToken

  const request = (async () => {
    try {
      const buffer = await fetchMsgpack({
        dataset: datasetName,
        kind: 'memory',
        rank: rankId,
        file: chunk.path.split('/').pop(),
      })
      const decoded = await postWorker('decodeMemoryChunk', { buffer }, [buffer])
      if (token !== loadToken) return

      const bundle = rankBundles.value.find((item) => item.rankId === rankId)
      if (!bundle) return
      bundle.loadedChunks.set(chunk.chunkId, decoded.snapshots)
      triggerRef(rankBundles)
      scheduleIndexRebuild()
    } catch (error) {
      if (!errorMessage.value) {
        errorMessage.value = error instanceof Error ? error.message : 'Failed to load memory chunk.'
      }
    } finally {
      pendingChunkRequests.delete(requestKey)
      if (!pendingChunkRequests.size) loading.value = false
    }
  })()

  pendingChunkRequests.set(requestKey, request)
  return request
}

function scheduleIndexRebuild() {
  if (indexBuildHandle) cancelAnimationFrame(indexBuildHandle)
  indexBuildHandle = requestAnimationFrame(() => {
    indexBuildHandle = 0
    rebuildIndex()
  })
}

function rebuildIndex() {
  if (!rankBundles.value.length) {
    indexes.value = null
    setIndexes(null)
    return
  }

  const bundles = rankBundles.value.map((bundle) => ({
    rankId: bundle.rankId,
    snapshots: Array.from(bundle.loadedChunks.values()).flat(),
  }))

  const built = buildMemviewIndex(bundles, {
    totalSteps: totalSteps.value,
    taskNodeLookup: taskNodeLookup.value,
  })
  indexes.value = built
  setIndexes(built)
}

function flattenTaskNodesFromGraphRank(graphRank, fallbackRankId) {
  const collected = []

  function walkNodes(nodes, rankId, depth = 0) {
    if (!Array.isArray(nodes)) return
    for (const rawNode of nodes) {
      if (!rawNode || typeof rawNode !== 'object') continue
      const nodeId = typeof rawNode.node_id === 'string' ? rawNode.node_id : ''
      if (nodeId) {
        const rankCandidate = Number.isFinite(rawNode.rank_id) ? Number(rawNode.rank_id) : rankId
        collected.push({
          id: nodeId,
          rankId: Number.isInteger(rankCandidate) ? rankCandidate : fallbackRankId,
          taskType: rawNode?.task?.task_type ?? '',
          taskData: rawNode?.task?.task_data ?? {},
          parents: Array.isArray(rawNode.parents) ? rawNode.parents : [],
          children: Array.isArray(rawNode.children) ? rawNode.children : [],
          depth,
        })
      }

      const subGraph = rawNode?.task?.task_data?.sub_graph
      if (Array.isArray(subGraph)) {
        for (const streamNodes of subGraph) {
          walkNodes(streamNodes, rankId, depth + 1)
        }
      }
    }
  }

  const rankId = Number.isInteger(graphRank?.rankId) ? graphRank.rankId : fallbackRankId
  walkNodes(graphRank?.nodes ?? [], rankId, 0)
  return collected
}

function taskNodeHasFlowSlices(node) {
  const data = node?.taskData
  if (!data || typeof data !== 'object') return false
  return Boolean(
    data.src_slice ||
      data.dst_slice ||
      (Array.isArray(data.src_slices) && data.src_slices.length) ||
      (Array.isArray(data.dst_slices) && data.dst_slices.length) ||
      (data.local_slice && data.remote_slice),
  )
}

function shouldReplaceTaskLookupNode(currentNode, nextNode) {
  if (!currentNode) return true
  const currentHasFlow = taskNodeHasFlowSlices(currentNode)
  const nextHasFlow = taskNodeHasFlowSlices(nextNode)
  if (currentHasFlow !== nextHasFlow) {
    return nextHasFlow
  }

  const currentDepth = Number.isInteger(currentNode?.depth) ? currentNode.depth : 0
  const nextDepth = Number.isInteger(nextNode?.depth) ? nextNode.depth : 0
  if (currentDepth !== nextDepth) {
    return nextDepth > currentDepth
  }

  return false
}

function buildTaskNodeLookup(bundles) {
  const lookup = new Map()
  for (const bundle of bundles) {
    for (const node of bundle.taskNodes ?? []) {
      if (!node?.id) continue
      const currentNode = lookup.get(node.id) ?? null
      if (shouldReplaceTaskLookupNode(currentNode, node)) {
        lookup.set(node.id, node)
      }
    }
  }
  return lookup
}

function handleCanvasStepUpdate(step) {
  emit('update:currentStep', step)
}

function handleBufferSelect(selection) {
  emit('buffer-select', selection)
}

function zoomOut() {
  canvasRef.value?.adjustZoom(-1)
}

function zoomIn() {
  canvasRef.value?.adjustZoom(1)
}
</script>

<template>
  <section class="dashboard-panel memview-overview-panel memview-overview-panel--unified">
    <div class="memview-overview-hero">
      <div class="memview-overview-hero__identity">
        <div class="memview-overview-hero__badge">
          <el-icon><DataLine /></el-icon>
        </div>
        <div>
          <h2>{{ datasetName || 'No dataset loaded' }}</h2>
          <div class="memview-overview-stats">
            <div class="memview-overview-stat">
              <span>Timeline</span>
              <strong>Step {{ activeStep }}</strong>
            </div>
            <div class="memview-overview-stat">
              <span>Ranks</span>
              <strong>{{ rankBundles.length }}</strong>
            </div>
            <div class="memview-overview-stat">
              <span>Total steps</span>
              <strong>{{ totalSteps }}</strong>
            </div>
          </div>
        </div>
      </div>

      <div class="memview-overview-hero__meta">
        <div class="memview-overview-meta-block">
          <span>Ranks</span>
          <div class="memview-overview-rank-badges">
            <span
              v-for="item in legendItems"
              :key="item.id"
              class="memview-overview-rank-badge"
              :style="{ '--memview-rank-badge-color': item.color }"
            >
              {{ item.label }}
            </span>
          </div>
        </div>
      </div>

      <div class="memview-overview-zoom-controls" aria-label="Timeline zoom">
        <button type="button" class="memview-overview-zoom-button" @click="zoomOut">
          <el-icon><ZoomOut /></el-icon>
        </button>
        <button type="button" class="memview-overview-zoom-button" @click="zoomIn">
          <el-icon><ZoomIn /></el-icon>
        </button>
      </div>
    </div>

    <div v-if="errorMessage" class="dashboard-empty-wrap">
      <el-empty :description="errorMessage" />
    </div>

    <div v-else-if="loading && !hasContent" class="dashboard-empty-wrap">
      <el-empty description="Loading memory overview..." />
    </div>

    <div v-else-if="!hasContent && !loading" class="dashboard-empty-wrap">
      <el-empty :description="emptyText" />
    </div>

    <div v-else class="memview-overview-timeline-wrap">
      <p class="memview-overview-loading-label">{{ loadingLabel }}</p>
            <MemViewTimelineCanvas
              ref="canvasRef"
              :rank-ids="selectedRankIds"
              :total-steps="totalSteps"
              :current-step="activeStep"
              :display-items="props.displayStepItems"
              :buffer-presence="indexes?.bufferPresence ?? null"
              :buffer-changed-flags="indexes?.bufferChangedFlags ?? null"
              :layout-buffers-index="indexes?.layoutBuffers ?? null"
              :buffer-ops-slice="currentStepBufferOps"
              :task-ops-slice="currentStepTaskOps"
              :buffer-slots="visibleBufferSlots"
        @update:currentStep="handleCanvasStepUpdate"
        @buffer-select="handleBufferSelect"
      />
    </div>
  </section>
</template>

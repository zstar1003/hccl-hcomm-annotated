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
import { computed, ref, shallowRef, watch } from 'vue'
import {
  RefreshRight,
  ZoomIn,
  ZoomOut,
} from '@element-plus/icons-vue'
import { buildCheckerV3DagScene } from '../../utils/checkerV3Data.js'
import { DEFAULT_MEMVIEW_STAGE } from '../../utils/memviewDag.js'
import MemViewDagSigmaCanvas from './MemViewDagSigmaCanvas.vue'

const props = defineProps({
  datasetName: {
    type: String,
    default: '',
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
  'update:currentStep',
  'update:totalSteps',
  'nodes-loaded',
  'graph-node-ids-change',
])

const loadingScene = ref(false)
const errorMessage = ref('')
const checkerV3Scene = shallowRef(null)
const sigmaCanvasRef = ref(null)
const zoomScale = ref(1)
let latestLoadToken = 0

const selectedRankIds = computed(() =>
  props.selectedRankKeys
    .map((key) => {
      const match = String(key).match(/^rank-(\d+)$/)
      return match ? Number(match[1]) : null
    })
    .filter((rankId) => Number.isInteger(rankId))
    .sort((left, right) => left - right),
)
const resolvedStageName = DEFAULT_MEMVIEW_STAGE

const currentGraphNodeIds = computed(() => {
  const ids = []
  const seen = new Set()
  for (const node of checkerV3Scene.value?.taskNodes ?? []) {
    const lookupIds = Array.isArray(node?.lookupIds) && node.lookupIds.length ? node.lookupIds : [node?.id]
    for (const lookupId of lookupIds) {
      if (!lookupId || seen.has(lookupId)) continue
      seen.add(lookupId)
      ids.push(lookupId)
    }
  }
  return ids
})

const hasContent = computed(() => (checkerV3Scene.value?.taskNodes?.length ?? 0) > 0)

function resolveNodesAtStep(step) {
  if (!Number.isInteger(step)) return null

  const ids = new Set()
  for (const node of checkerV3Scene.value?.taskNodes ?? []) {
    if (node?.globalStep === step || node?.localStep === step) {
      ids.add(node.id)
    }
  }

  return ids.size ? ids : null
}

const stepHighlightedNodeIds = computed(() => resolveNodesAtStep(props.currentStep))

watch(
  currentGraphNodeIds,
  (nodeIds) => {
    emit('graph-node-ids-change', nodeIds)
  },
  { immediate: true },
)

watch(
  () => [props.datasetName, selectedRankIds.value.join(',')],
  async () => {
    await loadScene()
  },
  { immediate: true },
)

async function loadScene() {
  const loadToken = ++latestLoadToken
  checkerV3Scene.value = null
  errorMessage.value = ''
  zoomScale.value = 1
  emit('update:totalSteps', 0)

  if (!props.datasetName) {
    return
  }

  loadingScene.value = true

  try {
    const scene = await buildCheckerV3DagScene(props.datasetName, selectedRankIds.value, {
      stageName: resolvedStageName,
    })
    if (loadToken !== latestLoadToken) return

    checkerV3Scene.value = scene
    emit('update:totalSteps', scene.totalSteps)
    emit('update:currentStep', Math.min(props.currentStep, Math.max(0, scene.totalSteps - 1)))
    if (scene.taskNodes.length) {
      emit('nodes-loaded', scene.taskNodes)
    }
  } catch (error) {
    if (loadToken !== latestLoadToken) return
    errorMessage.value = error instanceof Error ? error.message : 'Failed to load V3 DAG scene.'
  } finally {
    if (loadToken === latestLoadToken) {
      loadingScene.value = false
    }
  }
}

function handleNodeSelect(node) {
  emit('node-select', node ?? null)
}

function zoomOut() {
  sigmaCanvasRef.value?.zoomOut?.()
}

function zoomIn() {
  sigmaCanvasRef.value?.zoomIn?.()
}

function resetZoom() {
  sigmaCanvasRef.value?.resetZoom?.()
}

function handleSigmaZoomChange(nextZoomScale) {
  if (!Number.isFinite(nextZoomScale)) return
  zoomScale.value = Number(nextZoomScale)
}
</script>

<template>
  <section class="dashboard-panel memview-dag-panel">
    <div class="dashboard-section-title">
      <div>
        <p class="dashboard-eyebrow">MemView</p>
        <h2>DAG Task Graph</h2>
      </div>
    </div>

    <div v-if="errorMessage" class="dashboard-empty-wrap">
      <el-empty :description="errorMessage" />
    </div>

    <div v-else-if="loadingScene && !hasContent" class="dashboard-empty-wrap">
      <el-empty description="Loading data..." />
    </div>

    <div v-else-if="!hasContent" class="dashboard-empty-wrap">
      <el-empty :description="emptyText" />
    </div>

    <div v-else class="memview-dag-stage memview-dag-stage--full">
      <MemViewDagSigmaCanvas
        ref="sigmaCanvasRef"
        :key="`checker-v3-${props.datasetName}-${selectedRankIds.join('-')}`"
        :scene="checkerV3Scene"
        :selected-node-id="selectedNodeId"
        :issue-node-ids="issueNodeIds"
        :highlighted-node-ids="stepHighlightedNodeIds"
        @node-select="handleNodeSelect"
        @zoom-change="handleSigmaZoomChange"
      />
      <div class="memview-dag-zoom-controls" aria-label="DAG zoom controls">
        <button
          type="button"
          class="memview-dag-zoom-button"
          aria-label="Zoom out DAG"
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
          @click="zoomIn"
        >
          <el-icon><ZoomIn /></el-icon>
        </button>
      </div>
    </div>
  </section>
</template>

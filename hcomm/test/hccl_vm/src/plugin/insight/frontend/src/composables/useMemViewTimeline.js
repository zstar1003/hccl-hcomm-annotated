/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

import { computed, ref, shallowRef, triggerRef } from 'vue'
import { bufferOpsAtStep, taskOpsAtStep, layoutBuffersAtStep } from '../utils/memviewIndex.js'

const currentStep = ref(0)
const totalSteps = ref(0)
const selectedNode = shallowRef(null)
const hoveredNode = shallowRef(null)
const subgraphNode = shallowRef(null)
const indexes = shallowRef(null)
const nodeRegistry = shallowRef(new Map())
const navigationRequest = shallowRef({ nodeId: '', follow: false, tick: 0 })
const stepFocusRequest = shallowRef({ step: 0, smoothScroll: false, tick: 0 })

const clampedCurrentStep = computed(() => {
  const max = Math.max(0, totalSteps.value - 1)
  return Math.min(Math.max(currentStep.value, 0), max)
})

const activeStepBufferOps = computed(() => {
  if (!indexes.value) return []
  return bufferOpsAtStep(indexes.value, clampedCurrentStep.value)
})

const activeStepTaskOps = computed(() => {
  if (!indexes.value) return []
  return taskOpsAtStep(indexes.value, clampedCurrentStep.value)
})

const activeStepLayoutBuffers = computed(() => {
  if (!indexes.value) return []
  return layoutBuffersAtStep(indexes.value, clampedCurrentStep.value)
})

const selectedNodeBufferOps = computed(() => {
  const node = selectedNode.value
  if (!node || !indexes.value?.bufferOpsByNode) return []
  return indexes.value.bufferOpsByNode.get(node.id) ?? []
})

function setTotalSteps(value) {
  if (!Number.isFinite(value)) return
  totalSteps.value = Math.max(0, Math.floor(value))
  if (currentStep.value >= totalSteps.value) {
    currentStep.value = Math.max(0, totalSteps.value - 1)
  }
}

function setIndexes(next) {
  indexes.value = next
  if (next?.totalSteps && next.totalSteps > totalSteps.value) {
    totalSteps.value = next.totalSteps
  }
}

function registerNodes(nodes) {
  if (!Array.isArray(nodes) || !nodes.length) return
  const registry = nodeRegistry.value
  let mutated = false
  for (const node of nodes) {
    if (!node?.id) continue
    for (const lookupId of Array.isArray(node.lookupIds) ? node.lookupIds : [node.id]) {
      if (!lookupId) continue
      if (registry.get(lookupId) !== node) {
        registry.set(lookupId, node)
        mutated = true
      }
    }
  }
  if (mutated) {
    triggerRef(nodeRegistry)
  }
}

function clearNodeRegistry() {
  nodeRegistry.value = new Map()
}

function focusStep(step, { smoothScroll = false } = {}) {
  if (!Number.isFinite(step)) return
  const clamped = Math.min(Math.max(Math.floor(step), 0), Math.max(0, totalSteps.value - 1))
  currentStep.value = clamped
  stepFocusRequest.value = { step: clamped, smoothScroll, tick: stepFocusRequest.value.tick + 1 }
}

function focusNode(nodeOrId, { follow = true, sync = true } = {}) {
  let node = null
  if (nodeOrId && typeof nodeOrId === 'object') {
    node = nodeOrId
  } else if (typeof nodeOrId === 'string' && nodeOrId) {
    node = nodeRegistry.value.get(nodeOrId) ?? null
  }

  selectedNode.value = node

  if (!node) return
  if (sync && indexes.value?.stepByNodeId) {
    const step = indexes.value.stepByNodeId.get(node.id)
    if (Number.isInteger(step)) {
      focusStep(step, { smoothScroll: true })
    }
  }
  if (follow) {
    navigationRequest.value = { nodeId: node.id, follow: true, tick: navigationRequest.value.tick + 1 }
  }
}

function setHoveredNode(node) {
  hoveredNode.value = node && typeof node === 'object' ? node : null
}

function enterSubgraph(node) {
  if (!node) return
  subgraphNode.value = node
}

function exitSubgraph() {
  subgraphNode.value = null
}

function reset() {
  currentStep.value = 0
  totalSteps.value = 0
  selectedNode.value = null
  hoveredNode.value = null
  subgraphNode.value = null
  indexes.value = null
  clearNodeRegistry()
}

export function useMemViewTimeline() {
  return {
    currentStep,
    clampedCurrentStep,
    totalSteps,
    selectedNode,
    hoveredNode,
    subgraphNode,
    indexes,
    nodeRegistry,
    activeStepBufferOps,
    activeStepTaskOps,
    activeStepLayoutBuffers,
    selectedNodeBufferOps,
    navigationRequest,
    stepFocusRequest,
    setTotalSteps,
    setIndexes,
    registerNodes,
    clearNodeRegistry,
    focusStep,
    focusNode,
    setHoveredNode,
    enterSubgraph,
    exitSubgraph,
    reset,
  }
}

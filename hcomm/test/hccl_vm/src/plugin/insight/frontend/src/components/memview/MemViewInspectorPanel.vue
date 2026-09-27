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
import { computed, ref, watch } from 'vue'
import { ArrowDown, Cpu, List, Odometer, Share } from '@element-plus/icons-vue'
import { BUFFER_LABEL_BY_ID, BUFFER_SLOTS } from '../../utils/memviewIndex.js'

const props = defineProps({
  selectedNode: {
    type: Object,
    default: null,
  },
  subgraphActive: {
    type: Boolean,
    default: false,
  },
  subgraphNode: {
    type: Object,
    default: null,
  },
  currentStep: {
    type: Number,
    default: 0,
  },
  nodeStep: {
    type: Number,
    default: null,
  },
  bufferOps: {
    type: Array,
    default: () => [],
  },
  nodeTaskOps: {
    type: Array,
    default: () => [],
  },
  stepBufferOps: {
    type: Array,
    default: () => [],
  },
  stepTaskOps: {
    type: Array,
    default: () => [],
  },
  stepLayoutBuffers: {
    type: Array,
    default: () => [],
  },
  selectedRankKeys: {
    type: Array,
    default: () => [],
  },
  canResolveRelation: {
    type: Function,
    default: null,
  },
  emptyText: {
    type: String,
    default: '暂无数据',
  },
})

const emit = defineEmits(['enter-subgraph', 'exit-subgraph', 'navigate', 'focus-step'])

const activeMemorySections = ref(['summary', 'task-ops', 'buffer-layout'])
const activeDagSections = ref(['node-summary', 'node-mapping', 'subgraph', 'node-ops', 'more-info'])
const selectedLayoutRankId = ref(null)
const selectedLayoutBufferId = ref(null)
const expandedLayoutSliceKeys = ref([])

const hasSelectedNode = computed(() => Boolean(props.selectedNode))
const hasRenderableData = computed(
  () =>
    hasSelectedNode.value ||
    props.stepTaskOps.length > 0 ||
    props.stepBufferOps.length > 0 ||
    props.currentStep > 0,
)

const stepStats = computed(() => ({
  tasks: props.stepTaskOps.length,
  buffers: props.stepBufferOps.length,
}))

const stepSummaryRows = computed(() => [
  ['Task Memory Ops', String(props.stepTaskOps.length)],
  ['变更 buffer', String(props.stepBufferOps.length)],
  ['当前节点', formatDisplayNodeId(props.selectedNode?.id)],
])

const stepTaskRankGroups = computed(() => {
  const groups = new Map()
  for (const op of props.stepTaskOps) {
    const rankId = Number.isInteger(op?.rankId) ? op.rankId : -1
    if (!groups.has(rankId)) groups.set(rankId, [])
    groups.get(rankId).push(op)
  }

  return Array.from(groups.entries())
    .sort((left, right) => left[0] - right[0])
    .map(([rankId, ops]) => ({
      rankId,
      ops: [...ops].sort(
        (left, right) =>
          compareNumber(left.dstBufferId, right.dstBufferId) ||
          compareNumber(left.dstOffset, right.dstOffset) ||
          compareNumber(left.srcRankId, right.srcRankId) ||
          String(left.taskNodeId ?? '').localeCompare(String(right.taskNodeId ?? '')),
      ),
    }))
})

const stepBufferRankGroups = computed(() => {
  const byRank = new Map()

  for (const op of props.stepLayoutBuffers) {
    const rankId = Number.isInteger(op?.rankId) ? op.rankId : -1
    if (!byRank.has(rankId)) byRank.set(rankId, [])
    byRank.get(rankId).push(op)
  }

  return Array.from(byRank.entries())
    .sort((left, right) => left[0] - right[0])
    .map(([rankId, ops]) => ({
      rankId,
      buffers: [...ops].sort((left, right) => compareNumber(left.bufferId, right.bufferId)),
    }))
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

const layoutRankOptions = computed(() => {
  const availableRankIds = new Set(stepBufferRankGroups.value.map((group) => group.rankId))
  return selectedRankIds.value.map((rankId) => ({
    label: `Rank ${rankId}`,
    value: rankId,
    disabled: !availableRankIds.has(rankId),
  }))
})

const selectedLayoutRankGroup = computed(() => {
  if (!stepBufferRankGroups.value.length) return null
  return stepBufferRankGroups.value.find((group) => group.rankId === selectedLayoutRankId.value) ?? null
})

const layoutBufferOptions = computed(() => {
  const availableBufferIds = new Set((selectedLayoutRankGroup.value?.buffers ?? []).map((op) => op.bufferId))
  return BUFFER_SLOTS.map((slot) => ({
    label: slot.label,
    value: slot.id,
    disabled: !availableBufferIds.has(slot.id),
  }))
})

const selectedLayoutOp = computed(() => {
  const buffers = selectedLayoutRankGroup.value?.buffers ?? []
  if (!buffers.length) return null
  return buffers.find((op) => op.bufferId === selectedLayoutBufferId.value) ?? buffers[0]
})

const hasSelectedNodeMemoryOps = computed(
  () => Number.isInteger(props.nodeStep) && props.nodeTaskOps.length > 0,
)

const selectedNodeMappings = computed(() =>
  Array.isArray(props.selectedNode?.nodeMappings)
    ? props.selectedNode.nodeMappings.filter(
        (item) => item && typeof item === 'object' && typeof item.lookupId === 'string' && item.lookupId,
      )
    : [],
)

const selectedNodeSummaryRows = computed(() => {
  if (!props.selectedNode) return []
  return [
    ['Node id', formatDisplayNodeId(props.selectedNode.id)],
    ['视图', props.selectedNode.stageName ?? '--'],
    ['Task 类型', props.selectedNode.taskType ?? '--'],
    ['rank / stream', `R${props.selectedNode.rankId ?? '--'} / Q${props.selectedNode.streamId ?? '--'}`],
    ['父 / 子节点', `${props.selectedNode.parents?.length ?? 0} / ${props.selectedNode.children?.length ?? 0}`],
    ['Step', hasSelectedNodeMemoryOps.value ? `Step ${props.nodeStep}` : '--'],
    ['位置', props.selectedNode.posInfo || formatMaybeNumber(props.selectedNode.pos)],
  ]
})

const nodeTaskSummary = computed(() =>
  props.nodeTaskOps.map((op, index) => ({
    key: `${op.taskNodeId ?? 'task'}-${op.dstRankId ?? 'x'}-${index}`,
    title: formatTaskOpTitle(op),
    stepLabel: hasSelectedNodeMemoryOps.value ? `Step ${props.nodeStep}` : '--',
    taskNodeId: op.taskNodeId ?? '',
    srcFields: taskFieldRows(op.srcRankId, op.srcBufferId, op.srcOffset, op.srcLen),
    dstFields: taskFieldRows(op.dstRankId, op.dstBufferId, op.dstOffset, op.dstLen),
    nodeIdText: formatTaskNodeId(op.taskNodeId),
    targetStageName: '',
    containerNodeId: props.subgraphNode?.id ?? '',
  })),
)

const selectedNodeSubgraph = computed(() => {
  const collection = props.selectedNode?.rawNode?.task?.task_data?.sub_graph
  if (!Array.isArray(collection)) return null
  const normalizedStreams = collection.filter((stream) => Array.isArray(stream))
  const streamCount = normalizedStreams.length
  const taskCount = normalizedStreams.reduce(
    (sum, stream) => sum + stream.filter((item) => item && typeof item === 'object').length,
    0,
  )
  return {
    streamCount,
    taskCount,
    queueNum: Number.isFinite(props.selectedNode?.taskData?.queue_num)
      ? Number(props.selectedNode.taskData.queue_num)
      : null,
  }
})

const selectedNodeDebug = computed(() => {
  if (!props.selectedNode) return ''
  const rawNode = props.selectedNode.rawNode ?? {}
  const taskData = { ...(props.selectedNode.taskData ?? {}) }
  if (Array.isArray(taskData.sub_graph)) {
    taskData.sub_graph = `[${taskData.sub_graph.length} streams hidden]`
  }

  return JSON.stringify(
    {
      id: props.selectedNode.id,
      taskType: props.selectedNode.taskType,
      rankId: props.selectedNode.rankId,
      streamId: props.selectedNode.streamId,
      localStep: props.selectedNode.localStep,
      globalStep: props.selectedNode.globalStep,
      parents: rawNode.parents ?? props.selectedNode.parents ?? [],
      children: rawNode.children ?? props.selectedNode.children ?? [],
      taskData,
    },
    null,
    2,
  )
})

watch(
  () => [selectedRankIds.value.join(','), stepBufferRankGroups.value.map((group) => group.rankId).join(',')],
  () => {
    if (!selectedRankIds.value.length) {
      selectedLayoutRankId.value = null
      selectedLayoutBufferId.value = null
      return
    }

    const enabledRank = layoutRankOptions.value.find((option) => !option.disabled)?.value ?? null
    const hasSelectedRank = layoutRankOptions.value.some(
      (option) => option.value === selectedLayoutRankId.value && !option.disabled,
    )
    if (!hasSelectedRank) {
      selectedLayoutRankId.value = enabledRank
    }
  },
  { immediate: true },
)

watch(
  selectedLayoutRankGroup,
  (group) => {
    if (!group) {
      selectedLayoutBufferId.value = null
      return
    }

    const enabledBuffer = layoutBufferOptions.value.find((option) => !option.disabled)?.value ?? null
    const hasSelectedBuffer = layoutBufferOptions.value.some(
      (option) => option.value === selectedLayoutBufferId.value && !option.disabled,
    )
    if (!hasSelectedBuffer) {
      selectedLayoutBufferId.value = enabledBuffer
    }
  },
  { immediate: true },
)

watch(
  selectedLayoutOp,
  () => {
    expandedLayoutSliceKeys.value = []
  },
)

function compareNumber(left, right) {
  const leftValue = Number.isFinite(left) ? Number(left) : Number.POSITIVE_INFINITY
  const rightValue = Number.isFinite(right) ? Number(right) : Number.POSITIVE_INFINITY
  return leftValue - rightValue
}

function formatMaybeNumber(value) {
  return Number.isFinite(value) ? String(value) : '--'
}

function formatBytes(value) {
  if (!Number.isFinite(value)) return '--'
  const size = Math.abs(Number(value))
  if (size >= 1024 * 1024 * 1024) return `${(value / (1024 * 1024 * 1024)).toFixed(2)} GiB`
  if (size >= 1024 * 1024) return `${(value / (1024 * 1024)).toFixed(2)} MiB`
  if (size >= 1024) return `${(value / 1024).toFixed(2)} KiB`
  return `${value} B`
}

function bufferLabel(bufferId) {
  if (BUFFER_LABEL_BY_ID.has(bufferId)) return BUFFER_LABEL_BY_ID.get(bufferId)
  return Number.isInteger(bufferId) ? `BUFFER_${bufferId}` : '--'
}

function formatOffsetLen(offset, len) {
  const offsetLabel = Number.isFinite(offset) ? `offset ${offset}` : 'offset --'
  const lenLabel = Number.isFinite(len) ? `len ${len}` : 'len --'
  return `${offsetLabel} | ${lenLabel}`
}

function formatBufferRef(rankId, bufferId, offset, len) {
  const rankLabel = Number.isInteger(rankId) ? `R${rankId}` : 'R?'
  return `${rankLabel} | ${bufferLabel(bufferId)} | ${formatOffsetLen(offset, len)}`
}

function formatTaskOpTitle(op) {
  return String(op?.taskType ?? 'TASK').replaceAll('_', ' ')
}

function taskFieldRows(rankId, bufferId, offset, len) {
  return [
    { label: 'rank', value: Number.isInteger(rankId) ? `R${rankId}` : '--' },
    { label: 'buffer', value: bufferLabel(bufferId) },
    { label: 'offset', value: Number.isFinite(offset) ? String(offset) : '--' },
    { label: 'len', value: Number.isFinite(len) ? String(len) : '--' },
  ]
}

function formatTaskNodeId(nodeId) {
  return formatDisplayNodeId(nodeId)
}

function formatDisplayNodeId(nodeId) {
  if (!nodeId) return '--'
  return String(nodeId)
}

function sourceFieldRows(sourceRef) {
  return taskFieldRows(sourceRef?.rankId, sourceRef?.bufferId, sourceRef?.addr, sourceRef?.len)
}

function isSourceRefNavigable(sourceRef) {
  return Number.isInteger(sourceRef?.step)
}

function offsetLenFields(offset, len) {
  return [
    { label: 'offset', value: Number.isFinite(offset) ? String(offset) : '--' },
    { label: 'len', value: Number.isFinite(len) ? String(len) : '--' },
  ]
}

function sliceSummaryFields(segment, fallbackBufferId = null) {
  const bufferType = Number.isInteger(segment?.bufferType)
    ? segment.bufferType
    : Number.isInteger(segment?.buffer_type)
      ? segment.buffer_type
      : Number.isInteger(segment?.bufferId)
        ? segment.bufferId
        : fallbackBufferId
  const offset = Number.isFinite(segment?.startAddr) ? segment.startAddr : Number.isFinite(segment?.offset) ? segment.offset : null
  const size = Number.isFinite(segment?.size) ? segment.size : null

  return [
    { label: 'bufferType', value: bufferLabel(bufferType) },
    { label: 'offset', value: formatMaybeNumber(offset) },
    { label: 'size', value: formatMaybeNumber(size) },
  ]
}

function isLayoutSliceExpanded(sliceKey) {
  return expandedLayoutSliceKeys.value.includes(sliceKey)
}

function toggleLayoutSlice(sliceKey) {
  if (!sliceKey) return
  if (isLayoutSliceExpanded(sliceKey)) {
    expandedLayoutSliceKeys.value = expandedLayoutSliceKeys.value.filter((key) => key !== sliceKey)
    return
  }
  expandedLayoutSliceKeys.value = [...expandedLayoutSliceKeys.value, sliceKey]
}

function resolveRelationItems(nodeIds) {
  return (Array.isArray(nodeIds) ? nodeIds : [])
    .filter((nodeId) => typeof nodeId === 'string' && nodeId)
    .map((nodeId) => ({
      id: nodeId,
      resolvable: typeof props.canResolveRelation === 'function' ? Boolean(props.canResolveRelation(nodeId)) : true,
    }))
}

function sliceRows(op) {
  const currentSegments = Array.isArray(op?.segments) ? op.segments : []
  const previousSegments = Array.isArray(op?.previousSegments) ? op.previousSegments : []
  const baseSegments = currentSegments.length ? currentSegments : previousSegments

  return baseSegments.map((segment, index) => ({
    key: segment?.key ?? `${op.rankId}-${op.bufferId}-${index}`,
    current: currentSegments[index] ?? (!currentSegments.length ? segment : null),
    previous: findPreviousSegment(previousSegments, currentSegments[index] ?? segment),
  }))
}

function findPreviousSegment(previousSegments, currentSegment) {
  if (!Array.isArray(previousSegments) || !currentSegment) return null
  return (
    previousSegments.find(
      (segment) =>
        segment?.startAddr === currentSegment.startAddr &&
        segment?.size === currentSegment.size &&
        Boolean(segment?.isReduce) === Boolean(currentSegment.isReduce),
    ) ??
    previousSegments.find((segment) => segment?.startAddr === currentSegment.startAddr) ??
    previousSegments.find((segment) => overlaps(segment, currentSegment)) ??
    null
  )
}

function overlaps(left, right) {
  if (!left || !right) return false
  const leftStart = Number(left.startAddr)
  const leftEnd = leftStart + Number(left.size)
  const rightStart = Number(right.startAddr)
  const rightEnd = rightStart + Number(right.size)
  if (![leftStart, leftEnd, rightStart, rightEnd].every((value) => Number.isFinite(value))) return false
  return leftStart < rightEnd && rightStart < leftEnd
}

function sourceRefs(segment) {
  return Array.isArray(segment?.sourceRefs) ? segment.sourceRefs : []
}

function formatSourceRef(sourceRef) {
  return formatBufferRef(sourceRef?.rankId, sourceRef?.bufferId, sourceRef?.addr, sourceRef?.len)
}

function formatSourceStep(sourceRef) {
  return Number.isInteger(sourceRef?.step) ? `Step ${sourceRef.step}` : 'Step --'
}

function sectionMeta(label) {
  return label ? String(label) : ''
}

function navigateToNode(nodeId, containerNodeId = '') {
  if (!nodeId) return
  if (containerNodeId) {
    emit('navigate', {
      lookupId: nodeId,
      containerLookupId: containerNodeId,
    })
    return
  }
  emit('navigate', nodeId)
}

function navigateToMappedNode(mapping) {
  if (!mapping?.lookupId) return
  emit('navigate', {
    stageName: typeof mapping.stageName === 'string' ? mapping.stageName : '',
    lookupId: mapping.lookupId,
  })
}

function focusStep(step) {
  if (!Number.isInteger(step)) return
  emit('focus-step', step)
}

function selectLayout(rankId, bufferId) {
  if (!Number.isInteger(rankId) || !Number.isInteger(bufferId)) return false

  const rankOption = layoutRankOptions.value.find((option) => option.value === rankId && !option.disabled)
  if (!rankOption) return false

  selectedLayoutRankId.value = rankId

  const bufferOption = layoutBufferOptions.value.find((option) => option.value === bufferId && !option.disabled)
  if (!bufferOption) return false

  selectedLayoutBufferId.value = bufferId
  activeMemorySections.value = [...new Set([...activeMemorySections.value, 'buffer-layout'])]
  return true
}

defineExpose({ selectLayout })
</script>

<template>
  <aside class="dashboard-panel dashboard-record-panel memview-inspector-panel">
    <template v-if="hasRenderableData">
      <div class="dashboard-section-title">
        <div>
          <p class="dashboard-eyebrow">详情面板</p>
          <h2>详情</h2>
        </div>
        <span class="dashboard-status-pill">
          {{ selectedNode ? 'DAG + 内存' : '内存 Step' }}
        </span>
      </div>

      <el-collapse
        v-if="!hasSelectedNode"
        v-model="activeMemorySections"
        class="dashboard-sidebar-collapse memview-inspector-collapse"
      >
        <el-collapse-item name="summary" class="dashboard-subpanel">
          <template #title>
            <div class="dashboard-panel-toggle">
              <span class="dashboard-panel-toggle__title">当前 Step</span>
              <span class="dashboard-panel-toggle__meta">{{ sectionMeta(`Step ${currentStep}`) }}</span>
            </div>
          </template>

          <div class="memview-inspector-section-body">
            <div class="dashboard-record-card">
              <div class="dashboard-record-card__header">
                <div class="dashboard-record-card__icon">
                  <el-icon><Odometer /></el-icon>
                </div>
                <div>
                  <p class="dashboard-eyebrow">当前 Step</p>
                  <h2>Step {{ currentStep }}</h2>
                </div>
              </div>

              <div class="dashboard-tag-row">
                <el-tag type="primary">{{ stepStats.tasks }} 个 task ops</el-tag>
                <el-tag type="success">{{ stepStats.buffers }} 个变更 buffer</el-tag>
                <el-tag v-if="selectedNode" type="warning">节点 {{ formatDisplayNodeId(selectedNode.id) }}</el-tag>
              </div>

              <div v-if="subgraphActive" class="dashboard-action-row memview-inspector-actions">
                <el-button plain type="primary" @click="emit('exit-subgraph')">
                  退出子图
                </el-button>
              </div>

              <div class="dashboard-kv-grid">
                <div
                  v-for="([label, value], index) in stepSummaryRows"
                  :key="`${label}-${index}`"
                  class="dashboard-kv-card"
                  :class="{ 'is-wide': index === stepSummaryRows.length - 1 && stepSummaryRows.length % 2 === 1 }"
                >
                  <span>{{ label }}</span>
                  <strong>{{ value }}</strong>
                </div>
              </div>
            </div>
          </div>
        </el-collapse-item>

        <el-collapse-item v-if="stepTaskRankGroups.length" name="task-ops" class="dashboard-subpanel">
          <template #title>
            <div class="dashboard-panel-toggle">
              <span class="dashboard-panel-toggle__title">Task Memory Ops</span>
              <span class="dashboard-panel-toggle__meta">{{ sectionMeta(`${stepTaskRankGroups.length} 个 ranks`) }}</span>
            </div>
          </template>

          <div class="memview-inspector-section-body">
            <section class="dashboard-card-group">
              <div class="memview-inspector-task-groups">
                <section
                  v-for="group in stepTaskRankGroups"
                  :key="`step-rank-${group.rankId}`"
                  class="memview-inspector-task-rank"
                >
                  <header class="memview-inspector-task-rank__header">
                    <strong>Rank {{ group.rankId }}</strong>
                    <span>{{ group.ops.length }} 个 ops</span>
                  </header>

                  <ol class="memview-inspector-task-op-list">
                    <li
                      v-for="(op, opIndex) in group.ops"
                      :key="`${group.rankId}-${op.taskNodeId ?? 'task'}-${opIndex}`"
                      class="memview-inspector-task-op-item"
                    >
                      <div class="memview-inspector-task-op-flow memview-inspector-task-op-flow--split">
                        <div class="memview-inspector-task-op-src">
                          <span class="memview-inspector-table-tag memview-inspector-table-tag--src">SRC</span>
                          <div class="memview-inspector-task-op-grid">
                            <div
                              v-for="field in taskFieldRows(op.srcRankId, op.srcBufferId, op.srcOffset, op.srcLen)"
                              :key="`src-${field.label}`"
                              class="memview-inspector-task-op-cell"
                            >
                              <span>{{ field.label }}</span>
                              <strong>{{ field.value }}</strong>
                            </div>
                          </div>
                        </div>

                        <div class="memview-inspector-task-op-dst">
                          <span class="memview-inspector-table-tag memview-inspector-table-tag--dst">DST</span>
                          <div class="memview-inspector-task-op-grid">
                            <div
                              v-for="field in taskFieldRows(op.dstRankId, op.dstBufferId, op.dstOffset, op.dstLen)"
                              :key="`dst-${field.label}`"
                              class="memview-inspector-task-op-cell"
                            >
                              <span>{{ field.label }}</span>
                              <strong>{{ field.value }}</strong>
                            </div>
                          </div>
                        </div>
                      </div>

                      <div class="memview-inspector-task-op-meta">
                        <span class="memview-inspector-task-op-task">{{ formatTaskOpTitle(op) }}</span>
                        <span class="memview-inspector-task-op-common">{{ formatTaskNodeId(op.taskNodeId) }}</span>
                        <el-button
                          v-if="op.taskNodeId"
                          size="small"
                          plain
                          type="primary"
                          class="memview-inspector-action-button memview-inspector-action-button--inline"
                          @click="navigateToNode(op.taskNodeId)"
                        >
                          定位 Task
                        </el-button>
                      </div>
                    </li>
                  </ol>
                </section>
              </div>
            </section>
          </div>
        </el-collapse-item>

        <el-collapse-item name="buffer-layout" class="dashboard-subpanel">
          <template #title>
            <div class="dashboard-panel-toggle">
              <span class="dashboard-panel-toggle__title">Buffer Layout</span>
              <span class="dashboard-panel-toggle__meta">{{ sectionMeta(`${layoutRankOptions.length} 个 ranks`) }}</span>
            </div>
          </template>

          <div class="memview-inspector-section-body">
            <section class="dashboard-card-group">
              <div v-if="layoutRankOptions.length" class="memview-inspector-buffer-groups">
                <div class="dashboard-select-block memview-inspector-layout-filters">
                  <div class="memview-inspector-layout-filter">
                    <span class="dashboard-muted-label">Rank</span>
                    <el-select
                      v-model="selectedLayoutRankId"
                      class="dashboard-dark-select memview-inspector-layout-select"
                      placeholder="Rank"
                    >
                      <el-option
                        v-for="option in layoutRankOptions"
                        :key="`layout-rank-${option.value}`"
                        :label="option.label"
                        :value="option.value"
                        :disabled="option.disabled"
                      />
                    </el-select>
                  </div>

                  <div class="memview-inspector-layout-filter">
                    <span class="dashboard-muted-label">Buffer Type</span>
                    <el-select
                      v-model="selectedLayoutBufferId"
                      class="dashboard-dark-select memview-inspector-layout-select"
                      placeholder="Buffer Type"
                    >
                      <el-option
                        v-for="option in layoutBufferOptions"
                        :key="`layout-buffer-${option.value}`"
                        :label="option.label"
                        :value="option.value"
                        :disabled="option.disabled"
                      />
                    </el-select>
                  </div>
                </div>

                <section v-if="selectedLayoutOp" class="memview-inspector-buffer-layout">
                  <div class="dashboard-tag-row memview-inspector-buffer-layout__summary">
                    <el-tag size="small" type="info">{{ bufferLabel(selectedLayoutOp.bufferId) }}</el-tag>
                    <el-tag size="small" type="primary">{{ selectedLayoutOp.semanticsCount }} 个 slices</el-tag>
                    <el-tag size="small" type="success">{{ formatBytes(selectedLayoutOp.totalSize) }}</el-tag>
                    <el-tag v-if="selectedLayoutOp.isReduce" size="small" type="warning">reduce</el-tag>
                  </div>

                  <ol class="memview-inspector-buffer-slice-list">
                    <li
                      v-for="slice in sliceRows(selectedLayoutOp)"
                      :key="slice.key"
                      class="memview-inspector-buffer-slice"
                      :class="{ 'is-expanded': isLayoutSliceExpanded(slice.key) }"
                    >
                      <div class="memview-inspector-buffer-slice__meta">
                        <div class="memview-inspector-slice-summary-grid">
                          <div
                            v-for="field in offsetLenFields(
                              slice.current?.startAddr ?? slice.previous?.startAddr,
                              slice.current?.size ?? slice.previous?.size,
                            )"
                            :key="`${slice.key}-${field.label}`"
                            class="memview-inspector-task-op-cell memview-inspector-slice-summary-cell"
                          >
                            <span>{{ field.label }}</span>
                            <strong>{{ field.value }}</strong>
                          </div>
                        </div>
                        <button
                          type="button"
                          class="memview-inspector-slice-toggle"
                          :aria-expanded="isLayoutSliceExpanded(slice.key)"
                          :title="isLayoutSliceExpanded(slice.key) ? '收起 slice 详情' : '展开 slice 详情'"
                          @click="toggleLayoutSlice(slice.key)"
                        >
                          <el-icon
                            class="memview-inspector-slice-toggle__icon"
                            :class="{ 'is-expanded': isLayoutSliceExpanded(slice.key) }"
                          >
                            <ArrowDown />
                          </el-icon>
                        </button>
                      </div>

                      <div v-if="isLayoutSliceExpanded(slice.key)" class="memview-inspector-slice-details">
                        <div class="memview-inspector-slice-grid">
                          <section class="memview-inspector-slice-panel">
                            <header>
                              <strong>当前 Slice</strong>
                              <span>3 fields</span>
                            </header>
                            <div class="memview-inspector-slice-field-grid">
                              <div
                                v-for="field in sliceSummaryFields(slice.current ?? slice.previous, selectedLayoutOp.bufferId)"
                                :key="`${slice.key}-current-${field.label}`"
                                class="memview-inspector-slice-field"
                              >
                                <span>{{ field.label }}</span>
                                <strong>{{ field.value }}</strong>
                              </div>
                            </div>
                          </section>

                          <section class="memview-inspector-slice-panel">
                            <header>
                              <strong>上一 Slice</strong>
                              <span>3 fields</span>
                            </header>
                            <div class="memview-inspector-slice-field-grid">
                              <div
                                v-for="field in sliceSummaryFields(slice.previous ?? slice.current, selectedLayoutOp.bufferId)"
                                :key="`${slice.key}-previous-${field.label}`"
                                class="memview-inspector-slice-field"
                              >
                                <span>{{ field.label }}</span>
                                <strong>{{ field.value }}</strong>
                              </div>
                            </div>
                          </section>
                        </div>

                        <section class="memview-inspector-slice-panel">
                          <header>
                            <strong>来源语义</strong>
                            <span>{{ sourceRefs(slice.current).length }} 个 refs</span>
                          </header>
                          <div v-if="sourceRefs(slice.current).length" class="memview-inspector-source-list">
                            <component
                              v-for="(sourceRef, sourceIndex) in sourceRefs(slice.current)"
                              :is="isSourceRefNavigable(sourceRef) ? 'button' : 'div'"
                              :key="`${slice.key}-current-${sourceIndex}`"
                              class="memview-inspector-source-item memview-inspector-source-item--tasklike"
                              :class="{ 'is-clickable': isSourceRefNavigable(sourceRef) }"
                              :type="isSourceRefNavigable(sourceRef) ? 'button' : null"
                              :title="isSourceRefNavigable(sourceRef) ? `跳转到 ${formatSourceStep(sourceRef)}` : null"
                              @click="isSourceRefNavigable(sourceRef) ? focusStep(sourceRef.step) : null"
                            >
                              <div
                                v-if="isSourceRefNavigable(sourceRef)"
                                class="memview-inspector-source-item__meta"
                              >
                                <span>{{ formatSourceStep(sourceRef) }}</span>
                              </div>
                              <div class="memview-inspector-task-op-grid">
                                <div
                                  v-for="field in sourceFieldRows(sourceRef)"
                                  :key="`${slice.key}-current-${sourceIndex}-${field.label}`"
                                  class="memview-inspector-task-op-cell"
                                >
                                  <span>{{ field.label }}</span>
                                  <strong>{{ field.value }}</strong>
                                </div>
                              </div>
                            </component>
                          </div>
                          <p v-else class="memview-inspector-empty-inline">当前没有记录来源语义。</p>
                        </section>

                        <section class="memview-inspector-slice-panel">
                          <header>
                            <strong>上一步语义</strong>
                            <span>{{ sourceRefs(slice.previous).length }} 个 refs</span>
                          </header>
                          <div v-if="slice.previous && sourceRefs(slice.previous).length" class="memview-inspector-source-list">
                            <component
                              v-for="(sourceRef, sourceIndex) in sourceRefs(slice.previous)"
                              :is="isSourceRefNavigable(sourceRef) ? 'button' : 'div'"
                              :key="`${slice.key}-previous-${sourceIndex}`"
                              class="memview-inspector-source-item memview-inspector-source-item--tasklike"
                              :class="{ 'is-clickable': isSourceRefNavigable(sourceRef) }"
                              :type="isSourceRefNavigable(sourceRef) ? 'button' : null"
                              :title="isSourceRefNavigable(sourceRef) ? `跳转到 ${formatSourceStep(sourceRef)}` : null"
                              @click="isSourceRefNavigable(sourceRef) ? focusStep(sourceRef.step) : null"
                            >
                              <div
                                v-if="isSourceRefNavigable(sourceRef)"
                                class="memview-inspector-source-item__meta"
                              >
                                <span>{{ formatSourceStep(sourceRef) }}</span>
                              </div>
                              <div class="memview-inspector-task-op-grid">
                                <div
                                  v-for="field in sourceFieldRows(sourceRef)"
                                  :key="`${slice.key}-previous-${sourceIndex}-${field.label}`"
                                  class="memview-inspector-task-op-cell"
                                >
                                  <span>{{ field.label }}</span>
                                  <strong>{{ field.value }}</strong>
                                </div>
                              </div>
                            </component>
                          </div>
                          <p v-else class="memview-inspector-empty-inline">这个 slice 没有上一步语义。</p>
                        </section>
                      </div>
                    </li>
                  </ol>
                </section>
                <p v-else class="memview-inspector-empty-inline">
                  当前选中的 rank 在这个快照里没有 buffer layout 数据。
                </p>
              </div>
              <p v-else class="memview-inspector-empty-inline">
                当前 Step 下，所选 ranks 没有可用的 buffer layout 数据。
              </p>
            </section>
          </div>
        </el-collapse-item>

      </el-collapse>

      <el-collapse
        v-else
        v-model="activeDagSections"
        class="dashboard-sidebar-collapse memview-inspector-collapse"
      >
        <el-collapse-item name="node-summary" class="dashboard-subpanel">
          <template #title>
            <div class="dashboard-panel-toggle">
              <span class="dashboard-panel-toggle__title">节点概览</span>
              <span class="dashboard-panel-toggle__meta">
                {{ sectionMeta(selectedNode?.taskType ?? selectedNode?.id ?? '--') }}
              </span>
            </div>
          </template>

          <div class="memview-inspector-section-body">
            <div class="dashboard-record-card">
              <div class="dashboard-record-card__header">
                <div class="dashboard-record-card__icon">
                  <el-icon><Cpu /></el-icon>
                </div>
                <div>
                  <p class="dashboard-eyebrow">当前节点</p>
                  <h2>{{ selectedNode.label || selectedNode.taskType || formatDisplayNodeId(selectedNode.id) }}</h2>
                </div>
              </div>

              <div class="dashboard-tag-row">
                <el-tag type="primary">Rank {{ selectedNode.rankId }}</el-tag>
                <el-tag type="info">Queue {{ selectedNode.streamId }}</el-tag>
                <el-tag type="success">{{ hasSelectedNodeMemoryOps ? `Step ${nodeStep}` : 'Step --' }}</el-tag>
                <el-tag v-if="selectedNode.hasSubgraph" type="warning">子图</el-tag>
              </div>

              <div class="dashboard-action-row memview-inspector-actions">
                <el-button
                  v-if="selectedNode.hasSubgraph && !subgraphActive"
                  type="primary"
                  @click="emit('enter-subgraph')"
                >
                  打开子图
                </el-button>
                <el-button v-if="subgraphActive" plain @click="emit('exit-subgraph')">
                  退出子图
                </el-button>
                <el-button
                  v-if="hasSelectedNodeMemoryOps"
                  plain
                  type="primary"
                  @click="focusStep(nodeStep)"
                >
                  跳转到 Step
                </el-button>
              </div>

              <div class="dashboard-kv-grid">
                <div
                  v-for="([label, value], index) in selectedNodeSummaryRows"
                  :key="`${label}-${index}`"
                  class="dashboard-kv-card"
                  :class="{ 'is-wide': index === selectedNodeSummaryRows.length - 1 && selectedNodeSummaryRows.length % 2 === 1 }"
                >
                  <span>{{ label }}</span>
                  <strong>{{ value }}</strong>
                </div>
              </div>
            </div>

            <article class="memview-inspector-relation-shell">
              <div class="memview-inspector-relation-block">
                <span class="memview-inspector-relation-label">父节点</span>
                <template v-if="resolveRelationItems(selectedNode.parents).length">
                  <button
                    v-for="item in resolveRelationItems(selectedNode.parents)"
                    :key="`parent-${item.id}`"
                    type="button"
                    class="memview-inspector-relation-chip"
                    :class="{ 'is-unresolved': !item.resolvable }"
                    :disabled="!item.resolvable"
                    @click="navigateToNode(item.id)"
                  >
                    {{ item.id }}
                  </button>
                </template>
                <span v-else class="memview-inspector-relation-empty">没有父节点。</span>
              </div>

              <div class="memview-inspector-relation-block">
                <span class="memview-inspector-relation-label">子节点</span>
                <template v-if="resolveRelationItems(selectedNode.children).length">
                  <button
                    v-for="item in resolveRelationItems(selectedNode.children)"
                    :key="`child-${item.id}`"
                    type="button"
                    class="memview-inspector-relation-chip"
                    :class="{ 'is-unresolved': !item.resolvable }"
                    :disabled="!item.resolvable"
                    @click="navigateToNode(item.id)"
                  >
                    {{ item.id }}
                  </button>
                </template>
                <span v-else class="memview-inspector-relation-empty">没有子节点。</span>
              </div>
            </article>
          </div>
        </el-collapse-item>

        <el-collapse-item v-if="selectedNodeMappings.length" name="node-mapping" class="dashboard-subpanel">
          <template #title>
            <div class="dashboard-panel-toggle">
              <span class="dashboard-panel-toggle__title">节点映射</span>
              <span class="dashboard-panel-toggle__meta">{{ sectionMeta(selectedNodeMappings.length) }}</span>
            </div>
          </template>

          <div class="memview-inspector-section-body">
            <section class="dashboard-card-group">
              <div class="memview-inspector-mapping-list">
                <button
                  v-for="mapping in selectedNodeMappings"
                  :key="`${mapping.stageName}-${mapping.lookupId}`"
                  type="button"
                  class="memview-inspector-mapping-item"
                  @click="navigateToMappedNode(mapping)"
                >
                  <span>{{ mapping.stageName }}</span>
                  <strong>{{ mapping.label || mapping.lookupId }}</strong>
                  <em v-if="mapping.taskType">{{ mapping.taskType }}</em>
                </button>
              </div>
            </section>
          </div>
        </el-collapse-item>

        <el-collapse-item v-if="selectedNodeSubgraph" name="subgraph" class="dashboard-subpanel">
          <template #title>
            <div class="dashboard-panel-toggle">
              <span class="dashboard-panel-toggle__title">子图概览</span>
              <span class="dashboard-panel-toggle__meta">{{ sectionMeta(`${selectedNodeSubgraph.streamCount} 个 streams`) }}</span>
            </div>
          </template>

          <div class="memview-inspector-section-body">
            <section class="dashboard-card-group">
              <div class="dashboard-stats-grid">
                <div class="dashboard-stat-card">
                  <div class="dashboard-stat-card__head">
                    <el-icon class="dashboard-stat-card__icon"><Share /></el-icon>
                    <strong>{{ selectedNodeSubgraph.streamCount }}</strong>
                  </div>
                  <div class="dashboard-stat-card__body">
                    <span>Streams</span>
                  </div>
                </div>
                <div class="dashboard-stat-card">
                  <div class="dashboard-stat-card__head">
                    <el-icon class="dashboard-stat-card__icon"><Cpu /></el-icon>
                    <strong>{{ selectedNodeSubgraph.taskCount }}</strong>
                  </div>
                  <div class="dashboard-stat-card__body">
                    <span>Tasks</span>
                  </div>
                </div>
                <div class="dashboard-stat-card">
                  <div class="dashboard-stat-card__head">
                    <el-icon class="dashboard-stat-card__icon"><List /></el-icon>
                    <strong>{{ selectedNodeSubgraph.queueNum ?? '--' }}</strong>
                  </div>
                  <div class="dashboard-stat-card__body">
                    <span>Queue Num</span>
                  </div>
                </div>
              </div>
            </section>
          </div>
        </el-collapse-item>

        <el-collapse-item name="node-ops" class="dashboard-subpanel">
          <template #title>
            <div class="dashboard-panel-toggle">
              <span class="dashboard-panel-toggle__title">节点 Memory Ops</span>
              <span class="dashboard-panel-toggle__meta">{{ sectionMeta(nodeTaskSummary.length) }}</span>
            </div>
          </template>

          <div class="memview-inspector-section-body">
            <section class="dashboard-card-group">
              <div v-if="nodeTaskSummary.length" class="memview-inspector-impact-list">
                <article v-for="item in nodeTaskSummary" :key="item.key" class="memview-inspector-impact-item">
                  <header>
                    <strong>{{ item.title }}</strong>
                    <em>{{ item.stepLabel }}</em>
                  </header>
                  <div class="memview-inspector-task-op-flow memview-inspector-task-op-flow--split">
                    <div class="memview-inspector-task-op-src">
                      <span class="memview-inspector-table-tag memview-inspector-table-tag--src">SRC</span>
                      <div class="memview-inspector-task-op-grid">
                        <div
                          v-for="field in item.srcFields"
                          :key="`${item.key}-src-${field.label}`"
                          class="memview-inspector-task-op-cell"
                        >
                          <span>{{ field.label }}</span>
                          <strong>{{ field.value }}</strong>
                        </div>
                      </div>
                    </div>

                    <div class="memview-inspector-task-op-dst">
                      <span class="memview-inspector-table-tag memview-inspector-table-tag--dst">DST</span>
                      <div class="memview-inspector-task-op-grid">
                        <div
                          v-for="field in item.dstFields"
                          :key="`${item.key}-dst-${field.label}`"
                          class="memview-inspector-task-op-cell"
                        >
                          <span>{{ field.label }}</span>
                          <strong>{{ field.value }}</strong>
                        </div>
                      </div>
                    </div>
                  </div>
                  <footer class="memview-inspector-task-op-meta">
                    <span class="memview-inspector-task-op-common">{{ item.nodeIdText }}</span>
                    <el-button
                      v-if="item.taskNodeId"
                      size="small"
                      plain
                      type="primary"
                      class="memview-inspector-action-button memview-inspector-action-button--inline"
                      @click="item.targetStageName ? navigateToMappedNode({ stageName: item.targetStageName, lookupId: item.taskNodeId }) : navigateToNode(item.taskNodeId, item.containerNodeId)"
                    >
                      定位节点
                    </el-button>
                    <el-button
                      v-if="hasSelectedNodeMemoryOps"
                      size="small"
                      plain
                      type="primary"
                      class="memview-inspector-action-button memview-inspector-action-button--inline"
                      @click="focusStep(nodeStep)"
                    >
                      跳转到 Step
                    </el-button>
                  </footer>
                </article>
              </div>
              <p v-else class="memview-inspector-empty-inline">当前节点没有解析到直接关联的 task memory ops。</p>
            </section>
          </div>
        </el-collapse-item>

        <el-collapse-item name="more-info" class="dashboard-subpanel">
          <template #title>
            <div class="dashboard-panel-toggle">
              <span class="dashboard-panel-toggle__title">更多信息</span>
              <span class="dashboard-panel-toggle__meta">{{ sectionMeta('诊断视图') }}</span>
            </div>
          </template>

          <div class="memview-inspector-section-body">
            <section class="dashboard-card-group">
              <pre class="memview-inspector-json">{{ selectedNodeDebug }}</pre>
            </section>
          </div>
        </el-collapse-item>
      </el-collapse>
    </template>

    <div v-else class="dashboard-empty-wrap">
      <el-empty :description="emptyText" />
    </div>
  </aside>
</template>




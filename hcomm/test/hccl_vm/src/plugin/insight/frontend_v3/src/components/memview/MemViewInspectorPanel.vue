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
import { ArrowDown, Cpu, DocumentCopy, List, Odometer, Share } from '@element-plus/icons-vue'
import { BUFFER_LABEL_BY_ID, BUFFER_SLOTS } from '../../utils/memviewIndex.js'
import { normalizeMsId, resolveMemoryAddressDisplay } from '../../utils/msMemory.js'

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
  resolveRelationMeta: {
    type: Function,
    default: null,
  },
  emptyText: {
    type: String,
    default: '暂无数据',
  },
})

const emit = defineEmits(['enter-subgraph', 'exit-subgraph', 'navigate', 'focus-step'])

const DEFAULT_DAG_SECTIONS = ['node-summary', 'node-mapping', 'subgraph', 'node-ops']

const activeMemorySections = ref(['summary', 'task-ops', 'buffer-layout'])
const activeDagSections = ref([...DEFAULT_DAG_SECTIONS])
const selectedLayoutRankId = ref(null)
const selectedLayoutBufferId = ref(null)
const expandedLayoutSliceKeys = ref([])
const jsonCopied = ref(false)

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
    ['原始节点', formatDisplayNodeId(props.selectedNode.actualNodeId)],
    ['Task 类型', props.selectedNode.taskType ?? '--'],
    ['父 / 子节点', `${props.selectedNode.parents?.length ?? 0} / ${props.selectedNode.children?.length ?? 0}`],
    ['语义 Step', formatSemanticStep(props.selectedNode.globalStep)],
  ]
})

const selectedNodeBriefText = computed(() => {
  const subtitle = normalizeInlineText(props.selectedNode?.subtitle)
  if (!subtitle) return ''
  if (subtitle === props.selectedNode?.taskType) return ''
  if (subtitle.length > 72) return ''
  if (subtitle.includes('{')) return ''
  return subtitle
})

const selectedNodeDetailBlock = computed(() => null)

const selectedNodeDetailRows = computed(() =>
  (selectedNodeDetailBlock.value?.rows ?? []).filter(([label]) => isMeaningfulDetailLabel(label)),
)

const selectedNodeContextRows = computed(() => {
  if (!props.selectedNode) return []

  return [
    ['Rank', formatRankLabel(props.selectedNode.rankId)],
    ['Stream', formatMaybeNumber(props.selectedNode.streamId)],
    ['Queue', formatMaybeNumber(props.selectedNode.queueId)],
    ['Memory Step', hasSelectedNodeMemoryOps.value ? `Step ${props.nodeStep}` : '--'],
  ].filter(([, value]) => value !== '--')
})

// Returns the loop group owned by the selected node when the node itself is a loop start.
const selectedNodeOwnLoopGroup = computed(() => {
  const loopGroup = props.selectedNode?.ownLoopGroup
  return loopGroup && typeof loopGroup === 'object' ? loopGroup : null
})

// Returns every loop context that contains the selected node from inner to outer.
const selectedNodeLoopContexts = computed(() =>
  Array.isArray(props.selectedNode?.loopContexts)
    ? props.selectedNode.loopContexts.filter((item) => item && typeof item === 'object')
    : [],
)

// Picks the most relevant loop to summarize in the inspector.
const selectedNodePrimaryLoop = computed(() =>
  selectedNodeOwnLoopGroup.value ?? selectedNodeLoopContexts.value[0] ?? null,
)

// Builds the summary cards shown in the loop inspector section.
const selectedNodeLoopSummaryRows = computed(() => {
  const node = props.selectedNode
  const loopGroup = selectedNodePrimaryLoop.value
  if (!node || !loopGroup) return []

  return [
    ['节点角色', formatLoopRoleLabel(node.loopRole, Boolean(selectedNodeOwnLoopGroup.value))],
    ['总展开次数', formatLoopTotalExpandCount(loopGroup)],
    ['Instr range', formatLoopInstrRange(loopGroup)],
    ['Loop 节点数', formatMaybeNumber(loopGroup.nodeCount)],
    ['Loop 边界', formatLoopBoundary(loopGroup)],
  ].filter(([, value]) => value !== '--')
})

// Builds the clickable loop ancestry chips shown below the summary cards.
const selectedNodeLoopChainItems = computed(() =>
  selectedNodeLoopContexts.value.map((loopGroup, index) => ({
    key: `${loopGroup.startNodeId}-${index}`,
    startNodeId: loopGroup.startNodeId,
    title: formatLoopGroupTitle(loopGroup),
    subtitle: formatLoopBoundary(loopGroup),
    detail: formatLoopGroupDetail(loopGroup),
    isCurrent:
      loopGroup.startNodeId === selectedNodePrimaryLoop.value?.startNodeId ||
      loopGroup.startNodeId === props.selectedNode?.innermostLoopStartNodeId,
  })),
)

// Guards the loop section so non-loop nodes do not render empty scaffolding.
const hasSelectedNodeLoopInfo = computed(
  () => selectedNodeLoopSummaryRows.value.length > 0 || selectedNodeLoopChainItems.value.length > 0,
)

const selectedNodeNotifyRows = computed(() => {
  const notify = props.selectedNode?.rawNode?.notify
  if (!notify || typeof notify !== 'object') return []

  const rows = [
    ['notify_role', formatTaskValue(props.selectedNode?.rawNode?.notify_role)],
    ['kind', formatTaskValue(notify.kind)],
    ['record_rank_id', formatMaybeNumber(notify.record_rank_id)],
    ['wait_rank_id', formatMaybeNumber(notify.wait_rank_id)],
    ['notify_id', formatMaybeNumber(notify.notify_id)],
    ['channel_id', formatMaybeNumber(notify.channel_id)],
    ['die_id', formatMaybeNumber(notify.die_id)],
    ['cke_id', formatMaybeNumber(notify.cke_id)],
    ['cke_mask', formatMaskValue(notify.cke_mask)],
  ]

  return rows.filter(([, value]) => value !== '--')
})

const selectedNodeMetaRows = computed(() => {
  const taskData = props.selectedNode?.taskData
  if (!taskData || typeof taskData !== 'object') return []

  const rows = [
    ['协议', formatTaskValue(taskData.protocol_type)],
    ['Reduce Op', formatReduceType(taskData.reduce_op)],
    ['数据类型', formatDataType(taskData.data_type)],
    ['Boundary', formatTaskValue(props.selectedNode?.rawNode?.boundary_type)],
    ['pair_count', formatMaybeNumber(taskData.pair_count)],
    ['group_count', formatMaybeNumber(taskData.group_count)],
  ]

  return rows.filter(([, value]) => value !== '--')
})

const selectedNodeSliceGroups = computed(() => {
  const taskData = props.selectedNode?.taskData
  if (!taskData || typeof taskData !== 'object') return []

  const groups = [
    buildSliceGroup('src', '主输入 Slice', taskData.src_slice, false),
    buildSliceGroup('dst', '主输出 Slice', taskData.dst_slice, false),
    buildSliceGroup('src-list', 'SRC', taskData.src_slices, true),
    buildSliceGroup('dst-list', 'DST', taskData.dst_slices, true),
  ].filter(Boolean)

  return groups
})

const selectedNodeTrace = computed(() => {
  const trace = props.selectedNode?.taskData?.ccu_trace
  return trace && typeof trace === 'object' && !Array.isArray(trace) ? trace : null
})

const selectedNodeTraceRows = computed(() => {
  const trace = selectedNodeTrace.value
  if (!trace) return []

  const rows = [
    ['op_name', formatTraceValue(trace.op_name)],
    ['node_role', formatTraceValue(trace.node_role)],
    ['instr_id', formatTraceValue(trace.instr_id)],
    ['queue_id', formatTraceValue(trace.queue_id)],
    ['die_id', formatTraceValue(trace.die_id)],
    ['mission_id', formatTraceValue(trace.mission_id)],
    ['location.rank_id', formatTraceValue(trace.location?.rank_id)],
    ['location.stream_id', formatTraceValue(trace.location?.stream_id)],
    ['location.queue_id', formatTraceValue(trace.location?.queue_id)],
  ]

  return rows.filter(([, value]) => value !== '--')
})

const selectedNodeTraceRegisterGroups = computed(() => {
  const registerState = selectedNodeTrace.value?.register_state
  if (!registerState || typeof registerState !== 'object' || Array.isArray(registerState)) return []

  return [
    buildTraceRegisterGroup('xn', 'XN', registerState.xn),
    buildTraceRegisterGroup('gsa', 'GSA', registerState.gsa),
    buildTraceRegisterGroup('cke', 'CKE', registerState.cke),
  ].filter(Boolean)
})

const selectedNodeTraceRegisterCount = computed(() =>
  selectedNodeTraceRegisterGroups.value.reduce((sum, group) => sum + group.items.length, 0),
)

const selectedNodeTraceMetaCount = computed(
  () => selectedNodeTraceRows.value.length + selectedNodeTraceRegisterCount.value,
)

const hasSelectedNodeTrace = computed(() => selectedNodeTraceMetaCount.value > 0)

const nodeTaskSummary = computed(() =>
  props.nodeTaskOps.map((op, index) => ({
    key: `${op.taskNodeId ?? 'task'}-${op.dstRankId ?? 'x'}-${index}`,
    title: formatTaskOpTitle(op),
    stepLabel: hasSelectedNodeMemoryOps.value ? `Step ${props.nodeStep}` : '--',
    taskNodeId: op.taskNodeId ?? '',
    srcFields: taskFieldRows(op.srcRankId, op.srcBufferId, op.srcOffset, op.srcLen, { msId: op.srcMsId }),
    dstFields: taskFieldRows(op.dstRankId, op.dstBufferId, op.dstOffset, op.dstLen, { msId: op.dstMsId }),
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

const selectedNodeDebugObject = computed(() => {
  if (!props.selectedNode) return null
  const rawNode = { ...(props.selectedNode.rawNode ?? {}) }
  const displayTaskData = { ...(props.selectedNode.taskData ?? {}) }
  if (Array.isArray(displayTaskData.sub_graph)) {
    displayTaskData.sub_graph = `[${displayTaskData.sub_graph.length} streams hidden]`
  }
  if (rawNode.task && typeof rawNode.task === 'object' && rawNode.task.task_data && typeof rawNode.task.task_data === 'object') {
    rawNode.task = {
      ...rawNode.task,
      task_data: { ...rawNode.task.task_data },
    }
    if (Array.isArray(rawNode.task.task_data.sub_graph)) {
      rawNode.task.task_data.sub_graph = `[${rawNode.task.task_data.sub_graph.length} streams hidden]`
    }
  }

  return JSON.parse(
    JSON.stringify({
      id: props.selectedNode.id,
      taskType: props.selectedNode.taskType,
      rankId: props.selectedNode.rankId,
      streamId: props.selectedNode.streamId,
      localStep: props.selectedNode.localStep,
      globalStep: props.selectedNode.globalStep,
      ownLoopGroup: props.selectedNode.ownLoopGroup ?? null,
      loopContexts: props.selectedNode.loopContexts ?? [],
      rawNode,
      displayTaskData,
    }),
  )
})

const selectedNodeDebug = computed(() =>
  selectedNodeDebugObject.value ? JSON.stringify(selectedNodeDebugObject.value, null, 2) : '',
)

const selectedNodeDebugTree = computed(() => {
  if (!selectedNodeDebugObject.value || typeof selectedNodeDebugObject.value !== 'object') return []
  return buildJsonTreeChildren(selectedNodeDebugObject.value, 'root')
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

watch(
  () => props.selectedNode?.id ?? '',
  () => {
    activeDagSections.value = [...DEFAULT_DAG_SECTIONS]
    jsonCopied.value = false
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

function formatOffsetLen(offset, len, { bufferType = '', bufferId = null, msId = null } = {}) {
  const addressField = resolveMemoryAddressDisplay({ bufferType, bufferId, offset, msId })
  const offsetLabel = `${addressField.label} ${addressField.value}`
  const lenLabel = Number.isFinite(len) ? `len ${len}` : 'len --'
  return `${offsetLabel} | ${lenLabel}`
}

function formatBufferRef(rankId, bufferId, offset, len, { msId = null } = {}) {
  const rankLabel = Number.isInteger(rankId) ? `R${rankId}` : 'R?'
  return `${rankLabel} | ${bufferLabel(bufferId)} | ${formatOffsetLen(offset, len, { bufferId, msId })}`
}

function formatTaskOpTitle(op) {
  return String(op?.taskType ?? 'TASK')
    .replace(/^BATCH_/i, '')
    .replaceAll('_', ' ')
    .trim()
}

function taskFieldRows(rankId, bufferId, offset, len, { msId = null } = {}) {
  const addressField = resolveMemoryAddressDisplay({ bufferId, offset, msId })
  return [
    { label: 'rank', value: Number.isInteger(rankId) ? `R${rankId}` : '--' },
    { label: 'buffer', value: bufferLabel(bufferId) },
    { label: addressField.label, value: addressField.value },
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

function normalizeInlineText(value) {
  return typeof value === 'string' ? value.trim() : ''
}

function formatRankLabel(rankId) {
  return Number.isInteger(rankId) ? `R${rankId}` : '--'
}

function formatSemanticStep(step) {
  return Number.isInteger(step) ? String(step) : '--'
}

function formatTaskValue(value) {
  if (typeof value === 'string' && value) return value
  if (Number.isFinite(value)) return String(value)
  return '--'
}

function formatDataType(value) {
  if (value === 2 || value === '2') return 'INT32'
  return typeof value === 'string' && value ? value.toUpperCase() : Number.isFinite(value) ? String(value) : '--'
}

function formatReduceType(value) {
  if (value === 0 || value === '0') return 'SUM'
  return typeof value === 'string' && value ? value.toUpperCase() : Number.isFinite(value) ? String(value) : '--'
}

function formatTraceValue(value) {
  if (typeof value === 'string' && value) return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  if (value && typeof value === 'object') return JSON.stringify(value)
  return '--'
}

// Formats loop counters while keeping missing metadata explicit.
function formatLoopCounter(value) {
  return Number.isInteger(value) ? String(value) : '--'
}

// Collapses checker loop counters into one number so the UI reflects total expansions directly.
function resolveLoopTotalExpandCount(loopGroup) {
  const loopCnt = Number.isInteger(loopGroup?.loopCnt) ? Number(loopGroup.loopCnt) : null
  const expandCnt = Number.isInteger(loopGroup?.loopExpandCnt) ? Number(loopGroup.loopExpandCnt) : null
  if (!Number.isInteger(loopCnt) || !Number.isInteger(expandCnt)) return null
  return loopCnt * expandCnt
}

function formatLoopTotalExpandCount(loopGroup) {
  const totalExpandCount = resolveLoopTotalExpandCount(loopGroup)
  return Number.isInteger(totalExpandCount) ? String(totalExpandCount) : '--'
}

// Formats the instruction range shown in loop summaries and chips.
function formatLoopInstrRange(loopGroup) {
  if (!loopGroup || typeof loopGroup !== 'object') return '--'
  if (typeof loopGroup.instrRangeLabel === 'string' && loopGroup.instrRangeLabel) return loopGroup.instrRangeLabel
  if (Number.isInteger(loopGroup.instrIdStart) && Number.isInteger(loopGroup.instrIdEnd)) {
    return `I${loopGroup.instrIdStart}-${loopGroup.instrIdEnd}`
  }
  return '--'
}

// Formats the loop boundary node ids for display in the inspector.
function formatLoopBoundary(loopGroup) {
  if (!loopGroup || typeof loopGroup !== 'object') return '--'
  const startNodeId = formatDisplayNodeId(loopGroup.startNodeId)
  const endNodeId = formatDisplayNodeId(loopGroup.endNodeId)
  if (startNodeId === '--' && endNodeId === '--') return '--'
  return `${startNodeId} -> ${endNodeId}`
}

// Resolves how the selected node participates in the current loop.
function formatLoopRoleLabel(loopRole, hasOwnLoopGroup = false) {
  if (loopRole === 'start') return 'Loop Start'
  if (loopRole === 'end') return 'Loop End'
  if (hasOwnLoopGroup) return 'Loop Node'
  if (props.selectedNode?.innermostLoopStartNodeId) return 'Loop Body'
  return '--'
}

// Produces the short title used by loop ancestry chips.
function formatLoopGroupTitle(loopGroup) {
  return 'Loop'
}

// Produces the secondary detail line used by loop ancestry chips.
function formatLoopGroupDetail(loopGroup) {
  if (!loopGroup || typeof loopGroup !== 'object') return ''
  const parts = []
  const totalExpandCount = resolveLoopTotalExpandCount(loopGroup)
  if (Number.isInteger(totalExpandCount)) {
    parts.push(`Total ${totalExpandCount}`)
  }
  if (Number.isInteger(loopGroup.nodeCount)) {
    parts.push(`${loopGroup.nodeCount} nodes`)
  }
  if (Number.isInteger(loopGroup.depth)) {
    parts.push(`Depth ${loopGroup.depth}`)
  }
  const rangeLabel = formatLoopInstrRange(loopGroup)
  if (rangeLabel !== '--') parts.push(rangeLabel)
  return parts.join(' · ')
}

function formatRegisterValue(value) {
  if (typeof value === 'string' && value) {
    const normalized = value.trim()
    if (/^-?\d+$/.test(normalized)) {
      try {
        const bigint = BigInt(normalized)
        const negative = bigint < 0n
        const abs = negative ? -bigint : bigint
        const hex = `${negative ? '-' : ''}0x${abs.toString(16)}`
        return `${hex} (${normalized})`
      } catch {
        return normalized
      }
    }
    return normalized
  }
  if (Number.isFinite(value)) {
    const numeric = Number(value)
    const decimal = String(value)
    if (Number.isSafeInteger(numeric)) {
      const abs = Math.abs(numeric)
      const hex = `${numeric < 0 ? '-' : ''}0x${abs.toString(16)}`
      return `${hex} (${decimal})`
    }
    return decimal
  }
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  return '--'
}

function formatMaskValue(value) {
  if (typeof value === 'string' && value) return value
  if (Number.isFinite(value)) return String(value)
  return '--'
}

function formatJsonLeafValue(value) {
  const serialized = JSON.stringify(value)
  return typeof serialized === 'string' ? serialized : String(value)
}

function formatJsonSummary(value) {
  if (Array.isArray(value)) {
    const total = value.filter((item) => item !== undefined).length
    return `[${total} item${total === 1 ? '' : 's'}]`
  }

  if (value && typeof value === 'object') {
    const total = Object.entries(value).filter(([, child]) => child !== undefined).length
    return `{${total} field${total === 1 ? '' : 's'}}`
  }

  return formatJsonLeafValue(value)
}

function buildJsonTreeChildren(value, path) {
  if (Array.isArray(value)) {
    return value
      .map((item, index) => buildJsonTreeNode(`[${index}]`, item, `${path}.${index}`))
      .filter(Boolean)
  }

  if (value && typeof value === 'object') {
    return Object.entries(value)
      .map(([key, child]) => buildJsonTreeNode(key, child, `${path}.${key}`))
      .filter(Boolean)
  }

  return []
}

function buildJsonTreeNode(label, value, path) {
  if (value === undefined) return null

  if (Array.isArray(value)) {
    if (!value.length) {
      return {
        id: path,
        label,
        valueText: '[]',
        type: 'array',
        isLeaf: true,
      }
    }

    return {
      id: path,
      label,
      valueText: formatJsonSummary(value),
      type: 'array',
      isLeaf: false,
      children: buildJsonTreeChildren(value, path),
    }
  }

  if (value && typeof value === 'object') {
    const children = buildJsonTreeChildren(value, path)
    if (!children.length) {
      return {
        id: path,
        label,
        valueText: '{}',
        type: 'object',
        isLeaf: true,
      }
    }

    return {
      id: path,
      label,
      valueText: formatJsonSummary(value),
      type: 'object',
      isLeaf: false,
      children,
    }
  }

  return {
    id: path,
    label,
    valueText: formatJsonLeafValue(value),
    type: value === null ? 'null' : typeof value,
    isLeaf: true,
  }
}

function splitTopLevelSegments(text) {
  const source = normalizeInlineText(text)
  if (!source) return []

  const segments = []
  let current = ''
  let depth = 0

  for (const char of source) {
    if (char === ',' && depth === 0) {
      if (current.trim()) segments.push(current.trim())
      current = ''
      continue
    }

    if (char === '{' || char === '[' || char === '(') depth += 1
    if ((char === '}' || char === ']' || char === ')') && depth > 0) depth -= 1
    current += char
  }

  if (current.trim()) segments.push(current.trim())
  return segments
}

function parseNodeDetailFields(text, prefix = '') {
  return splitTopLevelSegments(text).flatMap((segment) => {
    const separatorIndex = segment.indexOf('=')
    if (separatorIndex <= 0) {
      const label = prefix ? `${prefix}.detail` : 'detail'
      return [[label, segment]]
    }

    const key = segment.slice(0, separatorIndex).trim()
    const value = segment.slice(separatorIndex + 1).trim()
    const label = prefix ? `${prefix}.${key}` : key

    if (value.startsWith('{') && value.endsWith('}')) {
      const nestedRows = parseNodeDetailFields(value.slice(1, -1), label)
      return nestedRows.length ? nestedRows : [[label, value]]
    }

    return [[label, value || '--']]
  })
}

function parseNodeDetail(text) {
  const raw = normalizeInlineText(text)
  if (!raw) return null

  const match = raw.match(/^\[([^\]]+)\]\s*(.*)$/)
  const title = match?.[1] ?? ''
  const content = match?.[2] ?? raw
  const rows = parseNodeDetailFields(content)

  return {
    title,
    raw,
    rows,
  }
}

function isMeaningfulDetailLabel(label) {
  const normalized = String(label ?? '').trim().toLowerCase()
  if (!normalized) return false
  if (normalized.startsWith('ccutrace')) return false

  const leaf = normalized.split('.').pop() ?? normalized
  return !new Set([
    'node',
    'rank',
    'stream',
    'queue',
    'protocol',
    'recordrank',
    'waitrank',
    'notifyid',
    'channelid',
    'dieid',
    'ckeid',
    'ckemask',
    'instrid',
    'queueid',
    'role',
    'op',
    'valid',
    'location',
  ]).has(leaf)
}

function normalizeTaskSlice(slice) {
  if (!slice || typeof slice !== 'object') return null
  const bufferType = typeof slice.buffer_type === 'string' ? slice.buffer_type : ''
  const offset = Number.isFinite(slice.offset) ? slice.offset : null
  return {
    rankId: Number.isInteger(slice.rank_id) ? slice.rank_id : null,
    bufferType,
    offset,
    size: Number.isFinite(slice.size) ? slice.size : null,
    msId: normalizeMsId(slice.ms_id ?? slice.msId, { bufferType, offset }),
  }
}

// Resolves the address field shown in Memory Slices so MS-backed slices read as ids instead of offsets.
function sliceAddressField(slice) {
  return resolveMemoryAddressDisplay({
    bufferType: slice?.bufferType,
    offset: slice?.offset,
    msId: slice?.msId,
  })
}

// Formats one slice card using the compact two-row layout: rank/type then id-or-offset/size.
function sliceDisplayRows(slice) {
  if (!slice) return []
  const addressField = sliceAddressField(slice)
  return [
    { label: 'rank', value: formatRankLabel(slice.rankId) },
    { label: 'type', value: slice.bufferType || '--' },
    { label: addressField.label, value: addressField.value },
    { label: 'size', value: formatMaybeNumber(slice.size) },
  ]
}

function slicePrimaryRows(rows) {
  return Array.isArray(rows) ? rows.slice(0, 2) : []
}

function sliceSecondaryRows(rows) {
  return Array.isArray(rows) ? rows.slice(2, 4) : []
}

function canMergeDisplaySlice(slice) {
  return Number.isInteger(slice?.rankId) &&
    typeof slice?.bufferType === 'string' &&
    slice.bufferType &&
    Number.isFinite(slice?.offset) &&
    Number.isFinite(slice?.size)
}

function mergeDisplaySlices(source) {
  const slices = source.map((slice) => ({
    ...slice,
    mergeCount: 1,
  }))
  if (slices.length <= 1) return slices

  slices.sort((left, right) =>
    compareNumber(left.rankId, right.rankId) ||
    String(left.bufferType || '').localeCompare(String(right.bufferType || '')) ||
    compareNumber(left.offset, right.offset) ||
    compareNumber(left.size, right.size),
  )

  const merged = []
  for (const slice of slices) {
    const last = merged[merged.length - 1]
    if (!last || !canMergeDisplaySlice(last) || !canMergeDisplaySlice(slice)) {
      merged.push(slice)
      continue
    }

    const sameRange = last.rankId === slice.rankId &&
      last.bufferType === slice.bufferType &&
      last.offset === slice.offset &&
      last.size === slice.size
    if (sameRange) {
      last.mergeCount += slice.mergeCount
      continue
    }

    merged.push(slice)
  }

  return merged
}

// Creates the shared slice-card payload consumed by the Memory Slices template.
function buildSliceCard(key, name, slice) {
  return {
    key,
    name,
    rows: sliceDisplayRows(slice),
    mergeCount: slice.mergeCount ?? 1,
    dropdownLabel: '',
    dropdownEntries: [],
  }
}

// Collapses a batch of MS slices into one expandable card while preserving each id/size pair in the dropdown.
function buildMsBatchSliceCard(key, title, slices) {
  if (!Array.isArray(slices) || !slices.length) return null

  const rankId = slices.every((slice) => slice.rankId === slices[0].rankId) ? slices[0].rankId : null
  const bufferType = slices.every((slice) => slice.bufferType === slices[0].bufferType) ? slices[0].bufferType : 'MS_CCU'
  const totalSize = slices.reduce((sum, slice) => sum + (Number.isFinite(slice.size) ? Number(slice.size) : 0), 0)

  return {
    key,
    name: title,
    rows: [
      { label: 'rank', value: formatRankLabel(rankId) },
      { label: 'type', value: bufferType || '--' },
      { label: 'id', value: `共计使用${slices.length}个MSID` },
      { label: 'size', value: formatMaybeNumber(totalSize) },
    ],
    mergeCount: 1,
    dropdownLabel: 'MSID 明细',
    dropdownEntries: slices
      .map((slice, index) => {
        const addressField = sliceAddressField(slice)
        return {
          key: `${key}-${index}`,
          label: addressField.label,
          value: addressField.value,
          size: formatMaybeNumber(slice.size),
          mergeCount: slice.mergeCount ?? 1,
        }
      })
      .sort((left, right) => compareNumber(Number(left.value), Number(right.value)) || compareNumber(left.size, right.size)),
  }
}

function isMsDisplaySlice(slice) {
  return sliceAddressField(slice).label === 'id'
}

function sliceItemName(title, index) {
  return `${title.replace(/s$/, '')} ${index + 1}`
}

function buildSliceGroup(key, title, source, multiple) {
  if (multiple) {
    const slices = Array.isArray(source) ? source.map(normalizeTaskSlice).filter(Boolean) : []
    if (!slices.length) return null
    const mergedSlices = mergeDisplaySlices(slices)
    const msSlices = mergedSlices.filter(isMsDisplaySlice)
    const regularSlices = mergedSlices.filter((slice) => !isMsDisplaySlice(slice))
    const items = []

    if (msSlices.length > 1) {
      const msBatchCard = buildMsBatchSliceCard(`${key}-ms`, 'MSID 列表', msSlices)
      if (msBatchCard) items.push(msBatchCard)
    } else if (msSlices.length === 1) {
      items.push(buildSliceCard(`${key}-ms-0`, sliceItemName(title, 0), msSlices[0]))
    }

    regularSlices.forEach((slice, index) => {
      items.push(buildSliceCard(`${key}-${index}`, sliceItemName(title, index + (msSlices.length === 1 ? 1 : 0)), slice))
    })

    return {
      key,
      title,
      total: items.length,
      totalLabel: items.length === slices.length ? String(slices.length) : `${items.length} / ${slices.length}`,
      items,
    }
  }

  const slice = normalizeTaskSlice(source)
  if (!slice) return null
  return {
    key,
    title,
    total: 1,
    totalLabel: '1',
    items: [buildSliceCard(`${key}-0`, title, slice)],
  }
}

function buildTraceRegisterGroup(key, title, source) {
  const entries = Array.isArray(source) ? source : []
  const items = entries
    .map((entry, index) => normalizeTraceRegisterEntry(key, title, entry, index))
    .filter(Boolean)
  if (!items.length) return null
  return {
    key,
    title,
    total: items.length,
    items,
  }
}

function normalizeTraceRegisterEntry(groupKey, title, entry, index) {
  if (!entry || typeof entry !== 'object' || Array.isArray(entry)) return null
  if (Number.isFinite(entry.id)) {
    const id = Number(entry.id)
    return {
      key: `${groupKey}-${id}`,
      label: `${title}[${id}]`,
      value: formatRegisterValue(entry.value),
    }
  }

  const keys = Object.keys(entry)
  if (keys.length !== 1) return null
  const rawId = keys[0]
  const id = Number.isFinite(Number(rawId)) ? Number(rawId) : null
  return {
    key: `${groupKey}-${id ?? index}`,
    label: id === null ? `${title}[${index}]` : `${title}[${id}]`,
    value: formatRegisterValue(entry[rawId]),
  }
}

async function copySelectedNodeDebug() {
  if (!selectedNodeDebug.value) return

  try {
    if (navigator?.clipboard?.writeText) {
      await navigator.clipboard.writeText(selectedNodeDebug.value)
    } else {
      const textarea = document.createElement('textarea')
      textarea.value = selectedNodeDebug.value
      textarea.setAttribute('readonly', 'readonly')
      textarea.style.position = 'absolute'
      textarea.style.left = '-9999px'
      document.body.appendChild(textarea)
      textarea.select()
      document.execCommand('copy')
      document.body.removeChild(textarea)
    }
    jsonCopied.value = true
    window.setTimeout(() => {
      jsonCopied.value = false
    }, 1600)
  } catch (error) {
    console.warn('[MemViewInspectorPanel] Failed to copy raw json.', error)
  }
}

function sourceFieldRows(sourceRef) {
  return taskFieldRows(sourceRef?.rankId, sourceRef?.bufferId, sourceRef?.addr, sourceRef?.len, {
    msId: sourceRef?.msId,
  })
}

function isSourceRefNavigable(sourceRef) {
  return Number.isInteger(sourceRef?.step)
}

function offsetLenFields(offset, len, { bufferType = '', bufferId = null, msId = null } = {}) {
  const addressField = resolveMemoryAddressDisplay({ bufferType, bufferId, offset, msId })
  return [
    { label: addressField.label, value: addressField.value },
    { label: 'len', value: Number.isFinite(len) ? String(len) : '--' },
  ]
}

function sliceSummaryFields(segment, fallbackBufferId = null) {
  const bufferId = Number.isInteger(segment?.bufferType)
    ? segment.bufferType
    : Number.isInteger(segment?.buffer_type)
      ? segment.buffer_type
      : Number.isInteger(segment?.bufferId)
        ? segment.bufferId
        : fallbackBufferId
  const offset = Number.isFinite(segment?.startAddr) ? segment.startAddr : Number.isFinite(segment?.offset) ? segment.offset : null
  const size = Number.isFinite(segment?.size) ? segment.size : null
  const addressField = resolveMemoryAddressDisplay({
    bufferId,
    offset,
    msId: normalizeMsId(segment?.ms_id ?? segment?.msId, { bufferId, offset }),
  })

  return [
    { label: 'bufferType', value: bufferLabel(bufferId) },
    { label: addressField.label, value: addressField.value },
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
    .map((nodeId) => {
      const meta = typeof props.resolveRelationMeta === 'function' ? props.resolveRelationMeta(nodeId) ?? null : null
      const taskType = normalizeInlineText(meta?.taskType ?? '')
      return {
        id: nodeId,
        taskType,
        label: [taskType, nodeId].filter(Boolean).join(' · '),
        resolvable: typeof props.canResolveRelation === 'function' ? Boolean(props.canResolveRelation(nodeId)) : true,
      }
    })
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
  return formatBufferRef(sourceRef?.rankId, sourceRef?.bufferId, sourceRef?.addr, sourceRef?.len, {
    msId: sourceRef?.msId,
  })
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
                              v-for="field in taskFieldRows(op.srcRankId, op.srcBufferId, op.srcOffset, op.srcLen, { msId: op.srcMsId })"
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
                              v-for="field in taskFieldRows(op.dstRankId, op.dstBufferId, op.dstOffset, op.dstLen, { msId: op.dstMsId })"
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
                              { bufferId: selectedLayoutOp.bufferId },
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
                  <p v-if="selectedNodeBriefText" class="memview-inspector-node-brief">{{ selectedNodeBriefText }}</p>
                </div>
              </div>

              <div class="dashboard-tag-row">
                <el-tag type="primary">Rank {{ selectedNode.rankId }}</el-tag>
                <el-tag type="info">Stream {{ selectedNode.streamId ?? '--' }}</el-tag>
                <el-tag type="info">Queue {{ selectedNode.queueId ?? '--' }}</el-tag>
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

            <section v-if="selectedNodeDetailRows.length" class="dashboard-card-group">
              <div class="memview-inspector-section-head">
                <h3>节点摘要</h3>
                <span class="memview-inspector-heading-meta">
                  {{ sectionMeta(selectedNodeDetailBlock?.title || '结构化详情') }}
                </span>
              </div>

              <div class="memview-inspector-detail-grid">
                <div
                  v-for="([label, value], index) in selectedNodeDetailRows"
                  :key="`detail-${label}-${index}`"
                  class="memview-inspector-detail-row"
                >
                  <span>{{ label }}</span>
                  <strong>{{ value }}</strong>
                </div>
              </div>
            </section>

            <section class="dashboard-card-group">
              <div class="memview-inspector-section-head">
                <h3>运行上下文</h3>
                <span class="memview-inspector-heading-meta">{{ sectionMeta('rank / stream / queue') }}</span>
              </div>

              <div class="dashboard-kv-grid">
                <div
                  v-for="([label, value], index) in selectedNodeContextRows"
                  :key="`context-${label}-${index}`"
                  class="dashboard-kv-card"
                >
                  <span>{{ label }}</span>
                  <strong>{{ value }}</strong>
                </div>
              </div>
            </section>

            <section v-if="hasSelectedNodeLoopInfo" class="dashboard-card-group">
              <div class="memview-inspector-section-head">
                <h3>Loop 信息</h3>
                <span class="memview-inspector-heading-meta">
                  {{ sectionMeta(selectedNodeLoopChainItems.length ? `${selectedNodeLoopChainItems.length} loops` : '当前 loop') }}
                </span>
              </div>

              <div v-if="selectedNodeLoopSummaryRows.length" class="dashboard-kv-grid">
                <div
                  v-for="([label, value], index) in selectedNodeLoopSummaryRows"
                  :key="`loop-summary-${label}-${index}`"
                  class="dashboard-kv-card"
                >
                  <span>{{ label }}</span>
                  <strong>{{ value }}</strong>
                </div>
              </div>

              <div v-if="selectedNodeLoopChainItems.length" class="memview-inspector-loop-groups">
                <button
                  v-for="item in selectedNodeLoopChainItems"
                  :key="item.key"
                  type="button"
                  class="memview-inspector-loop-chip"
                  :class="{ 'is-current': item.isCurrent }"
                  @click="navigateToNode(item.startNodeId)"
                >
                  <span>{{ item.title }}</span>
                  <strong>{{ item.subtitle }}</strong>
                  <em v-if="item.detail">{{ item.detail }}</em>
                </button>
              </div>
            </section>

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
                    {{ item.label || item.id }}
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
                    {{ item.label || item.id }}
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
                  <span>{{ mapping.stageName || '当前视图' }}</span>
                  <strong>{{ mapping.label || mapping.lookupId }}</strong>
                  <em>{{ [mapping.taskType, mapping.lookupId].filter(Boolean).join(' · ') }}</em>
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
              <span class="dashboard-panel-toggle__title">节点语义</span>
              <span class="dashboard-panel-toggle__meta">{{ sectionMeta(nodeTaskSummary.length || selectedNodeSliceGroups.length) }}</span>
            </div>
          </template>

          <div class="memview-inspector-section-body">
            <section v-if="selectedNodeNotifyRows.length" class="dashboard-card-group">
              <div class="memview-inspector-section-head">
                <h3>Notify</h3>
                <span class="memview-inspector-heading-meta">{{ sectionMeta('dump_v3 notify') }}</span>
              </div>

              <div class="dashboard-kv-grid">
                <div
                  v-for="([label, value], index) in selectedNodeNotifyRows"
                  :key="`notify-${label}-${index}`"
                  class="dashboard-kv-card"
                >
                  <span>{{ label }}</span>
                  <strong>{{ value }}</strong>
                </div>
              </div>
            </section>

            <section v-if="selectedNodeMetaRows.length" class="dashboard-card-group">
              <div class="memview-inspector-section-head">
                <h3>Task 元数据</h3>
                <span class="memview-inspector-heading-meta">{{ sectionMeta('protocol / reduce / boundary') }}</span>
              </div>

              <div class="dashboard-kv-grid">
                <div
                  v-for="([label, value], index) in selectedNodeMetaRows"
                  :key="`meta-${label}-${index}`"
                  class="dashboard-kv-card"
                >
                  <span>{{ label }}</span>
                  <strong>{{ value }}</strong>
                </div>
              </div>
            </section>

            <section v-if="selectedNodeSliceGroups.length" class="dashboard-card-group">
              <div class="memview-inspector-section-head">
                <h3>Memory Slices</h3>
                <span class="memview-inspector-heading-meta">{{ sectionMeta('src / dst') }}</span>
              </div>

              <div class="memview-inspector-slice-groups">
                <article
                  v-for="group in selectedNodeSliceGroups"
                  :key="group.key"
                  class="memview-inspector-slice-group"
                >
                  <header class="memview-inspector-slice-group__header">
                    <strong>{{ group.title }}</strong>
                    <span>{{ group.totalLabel || group.total }}</span>
                  </header>

                  <div class="memview-inspector-slice-group__body">
                    <section
                      v-for="item in group.items"
                      :key="item.key"
                      class="memview-inspector-slice-card"
                    >
                      <div class="memview-inspector-slice-card__head">
                        <div class="memview-inspector-slice-card__title">{{ item.name }}</div>
                        <span v-if="item.mergeCount > 1" class="memview-inspector-slice-card__badge">x{{ item.mergeCount }}</span>
                      </div>
                      <div class="memview-inspector-slice-card__rows memview-inspector-slice-card__rows--primary">
                        <div
                          v-for="field in slicePrimaryRows(item.rows)"
                          :key="`${item.key}-${field.label}`"
                          class="memview-inspector-slice-card__cell"
                        >
                          <span>{{ field.label }}</span>
                          <strong>{{ field.value }}</strong>
                        </div>
                      </div>
                      <div class="memview-inspector-slice-card__rows memview-inspector-slice-card__rows--secondary">
                        <div
                          v-for="field in sliceSecondaryRows(item.rows)"
                          :key="`${item.key}-${field.label}`"
                          class="memview-inspector-slice-card__cell"
                        >
                          <span>{{ field.label }}</span>
                          <strong>{{ field.value }}</strong>
                        </div>
                      </div>

                      <details v-if="item.dropdownEntries?.length" class="memview-inspector-slice-card__details">
                        <summary class="memview-inspector-slice-card__details-summary">
                          <span>{{ item.dropdownLabel }}</span>
                          <strong>{{ item.dropdownEntries.length }}</strong>
                        </summary>

                        <div class="memview-inspector-slice-card__details-list">
                          <div
                            v-for="entry in item.dropdownEntries"
                            :key="entry.key"
                            class="memview-inspector-slice-card__details-item"
                          >
                            <div class="memview-inspector-slice-card__details-cell">
                              <span>{{ entry.label }}</span>
                              <strong>{{ entry.value }}</strong>
                            </div>
                            <div class="memview-inspector-slice-card__details-cell">
                              <span>size</span>
                              <strong>{{ entry.size }}</strong>
                            </div>
                            <span v-if="entry.mergeCount > 1" class="memview-inspector-slice-card__badge">x{{ entry.mergeCount }}</span>
                          </div>
                        </div>
                      </details>
                    </section>
                  </div>
                </article>
              </div>
            </section>

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
              <p v-else-if="!selectedNodeSliceGroups.length" class="memview-inspector-empty-inline">当前节点没有解析到直接关联的 task memory ops。</p>
            </section>
          </div>
        </el-collapse-item>

        <el-collapse-item v-if="hasSelectedNodeTrace" name="node-trace" class="dashboard-subpanel">
          <template #title>
            <div class="dashboard-panel-toggle">
              <span class="dashboard-panel-toggle__title">CCU Trace</span>
              <span class="dashboard-panel-toggle__meta">{{ sectionMeta(selectedNodeTraceMetaCount) }}</span>
            </div>
          </template>

          <div class="memview-inspector-section-body">
            <section v-if="selectedNodeTraceRows.length" class="dashboard-card-group">
              <div class="dashboard-kv-grid">
                <div
                  v-for="([label, value], index) in selectedNodeTraceRows"
                  :key="`trace-${label}-${index}`"
                  class="dashboard-kv-card"
                >
                  <span>{{ label }}</span>
                  <strong>{{ value }}</strong>
                </div>
              </div>
            </section>

            <section v-if="selectedNodeTraceRegisterGroups.length" class="dashboard-card-group">
              <div class="memview-inspector-section-head">
                <h3>寄存器快照</h3>
                <span class="memview-inspector-heading-meta">{{ sectionMeta(selectedNodeTraceRegisterCount) }}</span>
              </div>

              <div class="memview-inspector-trace-groups">
                <article
                  v-for="group in selectedNodeTraceRegisterGroups"
                  :key="group.key"
                  class="memview-inspector-trace-group"
                >
                  <header class="memview-inspector-trace-group__header">
                    <strong>{{ group.title }}</strong>
                    <span>{{ group.total }}</span>
                  </header>

                  <div class="dashboard-kv-grid memview-inspector-trace-register-grid">
                    <div
                      v-for="item in group.items"
                      :key="item.key"
                      class="dashboard-kv-card memview-inspector-trace-register-card"
                    >
                      <span>{{ item.label }}</span>
                      <strong>{{ item.value }}</strong>
                    </div>
                  </div>
                </article>
              </div>
            </section>
          </div>
        </el-collapse-item>

        <el-collapse-item name="more-info" class="dashboard-subpanel">
          <template #title>
            <div class="dashboard-panel-toggle">
              <span class="dashboard-panel-toggle__title">原始 JSON</span>
              <span class="dashboard-panel-toggle__meta">{{ sectionMeta('完整节点数据') }}</span>
            </div>
          </template>

          <div class="memview-inspector-section-body">
            <section class="dashboard-card-group">
              <div class="memview-inspector-json-actions">
                <el-button
                  size="small"
                  plain
                  type="primary"
                  class="memview-inspector-action-button memview-inspector-action-button--inline"
                  @click="copySelectedNodeDebug"
                >
                  <el-icon><DocumentCopy /></el-icon>
                  <span>{{ jsonCopied ? '已复制' : '复制 JSON' }}</span>
                </el-button>
              </div>
              <el-tree
                v-if="selectedNodeDebugTree.length"
                :data="selectedNodeDebugTree"
                node-key="id"
                :props="{ label: 'label', children: 'children' }"
                :expand-on-click-node="false"
                :indent="16"
                class="dashboard-tree memview-inspector-json-tree"
              >
                <template #default="{ data }">
                  <div class="memview-inspector-json-node" :class="{ 'is-leaf': data.isLeaf }">
                    <span class="memview-inspector-json-node__key">{{ data.label }}</span>
                    <span class="memview-inspector-json-node__colon">:</span>
                    <span
                      class="memview-inspector-json-node__value"
                      :class="`is-${data.type}`"
                    >{{ data.valueText }}</span>
                  </div>
                </template>
              </el-tree>
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




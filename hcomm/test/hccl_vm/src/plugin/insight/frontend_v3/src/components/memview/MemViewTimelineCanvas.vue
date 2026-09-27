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
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useVirtualizer } from '@tanstack/vue-virtual'
import { BUFFER_SLOTS, rankColorFor } from '../../utils/memviewIndex.js'

const GUTTER_WIDTH = 66
const PANEL_TOP = 4
const RANK_CONTAINER_GAP = 8
const PANEL_HEADER_HEIGHT = 26
const PANEL_PADDING_TOP = 8
const PANEL_PADDING_BOTTOM = 10
const PANEL_PADDING_X = 10
const PANEL_GAP = 10
const MIN_CELL = 14
const MAX_CELL = 42
const DEFAULT_CELL = 24
const CELL_GAP = 4
const RANK_CONTAINER_PADDING_Y = 6
const RANK_CONTAINER_PADDING_X = 8
const MAX_BUFFER_COLUMNS = 5
const ARROW_DRAW_LIMIT = 384
const OVERSCAN_PANELS = 6
const ELLIPSIS_WIDTH = 40
const MIN_PANEL_WIDTH = 96

const props = defineProps({
  rankIds: {
    type: Array,
    required: true,
  },
  totalSteps: {
    type: Number,
    default: 0,
  },
  currentStep: {
    type: Number,
    default: 0,
  },
  displayItems: {
    type: Array,
    default: () => [],
  },
  bufferPresence: {
    type: Object,
    default: null,
  },
  bufferChangedFlags: {
    type: Object,
    default: null,
  },
  bufferOpsSlice: {
    type: Array,
    default: () => [],
  },
  layoutBuffersIndex: {
    type: Object,
    default: null,
  },
  taskOpsSlice: {
    type: Array,
    default: () => [],
  },
  bufferSlots: {
    type: [Array, null],
    default: null,
  },
})

const emit = defineEmits(['update:currentStep', 'buffer-select'])

const hostRef = ref(null)
const bodyWrapperRef = ref(null)
const scrollTop = ref(0)
const cellWidth = ref(DEFAULT_CELL)
const isScrubbing = ref(false)

const sortedRankIds = computed(() => [...props.rankIds].sort((a, b) => a - b))
const visibleBufferSlots = computed(() => {
  if (!Array.isArray(props.bufferSlots)) return BUFFER_SLOTS
  const seen = new Set()
  const result = []
  for (const slot of props.bufferSlots) {
    if (!Number.isInteger(slot?.id) || seen.has(slot.id)) continue
    if (!BUFFER_SLOTS.some((item) => item.id === slot.id)) continue
    seen.add(slot.id)
    result.push(slot)
  }
  return result
})

const cellStride = computed(() => cellWidth.value + CELL_GAP)
const bufferColumnCount = computed(() =>
  Math.max(1, Math.min(MAX_BUFFER_COLUMNS, visibleBufferSlots.value.length || 1)),
)
const bufferRowCount = computed(() => {
  if (visibleBufferSlots.value.length <= MAX_BUFFER_COLUMNS) return 1
  return Math.ceil(visibleBufferSlots.value.length / MAX_BUFFER_COLUMNS)
})
const bufferGridWidth = computed(() =>
  visibleBufferSlots.value.length
    ? bufferColumnCount.value * cellStride.value - CELL_GAP
    : 0,
)
const panelWidth = computed(
  () =>
    Math.max(
      MIN_PANEL_WIDTH,
      bufferGridWidth.value +
        PANEL_PADDING_X * 2 +
        RANK_CONTAINER_PADDING_X * 2,
    ),
)
const panelStride = computed(() => panelWidth.value + PANEL_GAP)
const rankContainerHeight = computed(
  () =>
    bufferRowCount.value * cellWidth.value +
    Math.max(0, bufferRowCount.value - 1) * CELL_GAP +
    RANK_CONTAINER_PADDING_Y * 2,
)
const ellipsisHeight = computed(() => PANEL_HEADER_HEIGHT + PANEL_PADDING_TOP + Math.max(20, cellWidth.value) + PANEL_PADDING_BOTTOM)
const panelHeight = computed(() => {
  const ranks = Math.max(1, sortedRankIds.value.length)
  const ranksHeight = ranks * rankContainerHeight.value + Math.max(0, ranks - 1) * RANK_CONTAINER_GAP
  return PANEL_HEADER_HEIGHT + PANEL_PADDING_TOP + ranksHeight + PANEL_PADDING_BOTTOM
})

const totalBodyHeight = computed(() => Math.max(panelHeight.value, ellipsisHeight.value) + 10)
const clampedCurrentStep = computed(() => Math.min(Math.max(0, props.currentStep), Math.max(0, props.totalSteps - 1)))
const normalizedDisplayItems = computed(() => {
  const items = Array.isArray(props.displayItems) ? props.displayItems : []
  if (items.length) return items
  return Array.from({ length: Math.max(0, props.totalSteps) }, (_, step) => ({ type: 'step', step }))
})
const displayItemSizeMap = computed(() =>
  normalizedDisplayItems.value.map((item) => (item?.type === 'ellipsis' ? ELLIPSIS_WIDTH + PANEL_GAP : panelStride.value)),
)
const displayItemOffsets = computed(() => {
  const offsets = new Array(normalizedDisplayItems.value.length)
  let cursor = PANEL_GAP
  for (let index = 0; index < normalizedDisplayItems.value.length; index += 1) {
    offsets[index] = cursor
    cursor += displayItemSizeMap.value[index] ?? panelStride.value
  }
  return offsets
})
const totalDisplayWidth = computed(() => {
  if (!normalizedDisplayItems.value.length) return 1
  const lastIndex = normalizedDisplayItems.value.length - 1
  return Math.max(1, displayItemOffsets.value[lastIndex] + displayItemSizeMap.value[lastIndex] + PANEL_GAP)
})

const rankRows = computed(() =>
  sortedRankIds.value.map((rankId, index) => ({
    rankId,
    index,
    color: rankColorFor(rankId, sortedRankIds.value),
    top:
      PANEL_HEADER_HEIGHT +
      PANEL_PADDING_TOP +
      index * (rankContainerHeight.value + RANK_CONTAINER_GAP),
  })),
)

const stepVirtualizer = useVirtualizer(
  computed(() => ({
    count: normalizedDisplayItems.value.length,
    getScrollElement: () => bodyWrapperRef.value,
    estimateSize: (index) => displayItemSizeMap.value[index] ?? panelStride.value,
    getItemKey: (index) => {
      const item = normalizedDisplayItems.value[index]
      if (item?.type === 'ellipsis') {
        return `ellipsis-${item.fromStep}-${item.toStep}`
      }
      return `step-${item?.step ?? index}`
    },
    horizontal: true,
    overscan: OVERSCAN_PANELS,
    paddingStart: PANEL_GAP,
    paddingEnd: PANEL_GAP,
  })),
)

const virtualItems = computed(() => stepVirtualizer.value.getVirtualItems())
const totalBodyWidth = computed(() => Math.max(stepVirtualizer.value.getTotalSize(), totalDisplayWidth.value, 1))
const displayIndexByStep = computed(() => {
  const result = new Map()
  normalizedDisplayItems.value.forEach((item, index) => {
    if (item?.type === 'step' && Number.isInteger(item.step)) {
      result.set(item.step, index)
    }
  })
  return result
})
const visibleBufferIndexById = computed(() => {
  const result = new Map()
  visibleBufferSlots.value.forEach((slot, index) => {
    result.set(slot.id, index)
  })
  return result
})
const visibleStepMap = computed(() => {
  const result = new Map()
  for (const item of virtualItems.value) {
    const displayItem = normalizedDisplayItems.value[item.index]
    if (displayItem?.type === 'step' && Number.isInteger(displayItem.step)) {
      result.set(displayItem.step, item.start)
    }
  }
  return result
})

const visibleLineSegments = computed(() => {
  const segments = []
  const currentStep = clampedCurrentStep.value
  const panelX = visibleStepMap.value.get(currentStep)
  if (panelX === undefined) return segments

  const rowsByRank = new Map(rankRows.value.map((row) => [row.rankId, row]))
  const ops = Array.isArray(props.taskOpsSlice) ? props.taskOpsSlice : []
  const seen = new Set()

  let drawn = 0
  for (const taskOp of ops) {
    if (drawn >= ARROW_DRAW_LIMIT) break
    if (!Number.isInteger(taskOp?.dstRankId) || !Number.isInteger(taskOp?.dstBufferId)) continue
    if (!Number.isInteger(taskOp?.srcRankId) || !Number.isInteger(taskOp?.srcBufferId)) continue
    if (Number.isInteger(taskOp?.step) && taskOp.step !== currentStep) continue
    if (!isBufferPresent(currentStep, taskOp.dstRankId, taskOp.dstBufferId)) continue
    if (!isBufferPresent(currentStep, taskOp.srcRankId, taskOp.srcBufferId)) continue

    if (!resolveVisibleBufferLayout(taskOp.srcBufferId) || !resolveVisibleBufferLayout(taskOp.dstBufferId)) continue
    const src = resolveCellCenterInStep(panelX, taskOp.srcRankId, taskOp.srcBufferId, rowsByRank)
    const dst = resolveCellCenterInStep(panelX, taskOp.dstRankId, taskOp.dstBufferId, rowsByRank)
    if (!src || !dst) continue

    const semanticStyle = createSemanticStyle(resolveTaskOpSemanticRanks(taskOp), sortedRankIds.value)
    const key = [
      currentStep,
      taskOp.taskNodeId ?? '',
      taskOp.srcRankId,
      taskOp.srcBufferId,
      taskOp.srcOffset ?? 'x',
      taskOp.srcLen ?? 'x',
      taskOp.dstRankId,
      taskOp.dstBufferId,
      taskOp.dstOffset ?? 'x',
      taskOp.dstLen ?? 'x',
    ].join(':')
    if (seen.has(key)) continue
    seen.add(key)

    const srcKey = cellKey(currentStep, taskOp.srcRankId, taskOp.srcBufferId)
    const dstKey = cellKey(currentStep, taskOp.dstRankId, taskOp.dstBufferId)
    const isSelf = srcKey === dstKey
    segments.push({
      id: key,
      path: isSelf ? buildSelfArrowPath(src) : buildArrowPath(src, dst),
      stroke: semanticStyle.stroke,
      alpha: isSelf ? 0.88 : 0.8,
      isSelf,
    })
    drawn += 1
  }

  return segments
})

const visibleStepNumbers = computed(() =>
  virtualItems.value
    .map((item) => normalizedDisplayItems.value[item.index])
    .filter((item) => item?.type === 'step' && Number.isInteger(item.step))
    .map((item) => item.step),
)

const cellSemanticStyles = computed(() => {
  const styles = new Map()
  const visibleSteps = visibleStepNumbers.value
  if (visibleSteps.length) {
    for (const step of visibleSteps) {
      applyLayoutStylesForStep(styles, step)
    }
  }
  applyTaskOpOverrides(styles)
  return styles
})

const currentStepTaskTargetKeys = computed(() => {
  const keys = new Set()
  const step = clampedCurrentStep.value
  for (const taskOp of props.taskOpsSlice ?? []) {
    if (!Number.isInteger(taskOp?.dstRankId) || !Number.isInteger(taskOp?.dstBufferId)) continue
    keys.add(cellKey(step, taskOp.dstRankId, taskOp.dstBufferId))
  }
  return keys
})

watch(
  () => props.currentStep,
  async () => {
    await nextTick()
    scrollCurrentIntoView(isScrubbing.value ? 'auto' : 'smooth')
  },
)

watch(
  () => props.totalSteps,
  async () => {
    await nextTick()
    stepVirtualizer.value.measure()
    scrollCurrentIntoView('auto')
  },
  { immediate: true },
)

onMounted(async () => {
  await nextTick()
  stepVirtualizer.value.measure()
  scrollCurrentIntoView('auto')
})

onBeforeUnmount(() => {
  window.removeEventListener('pointermove', handleBodyPointerMove)
})

function resolveCellCenterInStep(panelX, rankId, bufferId, rowsByRank) {
  const row = rowsByRank.get(rankId)
  if (!row) return null
  const bufferLayout = resolveVisibleBufferLayout(bufferId)
  if (!bufferLayout) return null

  return {
    x:
      panelX +
      PANEL_PADDING_X +
      RANK_CONTAINER_PADDING_X +
      bufferLayout.column * cellStride.value +
      cellWidth.value / 2,
    y:
      PANEL_TOP +
      row.top +
      RANK_CONTAINER_PADDING_Y +
      bufferLayout.row * cellStride.value +
      cellWidth.value / 2,
  }
}

function resolveVisibleBufferLayout(bufferId) {
  if (!Number.isInteger(bufferId)) return null
  const index = visibleBufferIndexById.value.get(bufferId)
  if (!Number.isInteger(index)) return null
  return {
    index,
    column: index % bufferColumnCount.value,
    row: Math.floor(index / bufferColumnCount.value),
  }
}

function getPresenceMask(rankId, step) {
  if (step < 0) return 0
  const presenceMap = props.bufferPresence
  if (!(presenceMap instanceof Map)) return 0
  const presence = presenceMap.get(rankId)
  if (!presence || step >= presence.length) return 0
  return presence[step] ?? 0
}

function getChangedMask(rankId, step) {
  if (step < 0) return 0
  const changedMap = props.bufferChangedFlags
  if (!(changedMap instanceof Map)) return 0
  const changedFlags = changedMap.get(rankId)
  if (!changedFlags || step >= changedFlags.length) return 0
  return changedFlags[step] ?? 0
}

function buildArrowPath(src, dst) {
  const dx = dst.x - src.x
  const dyDiff = dst.y - src.y
  const lift = Math.max(16, Math.min(56, Math.abs(dyDiff) * 0.42 + 20))
  const cp1x = src.x + dx * 0.35
  const cp1y = src.y - lift
  const cp2x = src.x + dx * 0.65
  const cp2y = dst.y - lift
  return `M ${src.x} ${src.y} C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${dst.x} ${dst.y}`
}

function buildSelfArrowPath(center) {
  const radius = Math.max(14, Math.min(30, cellWidth.value * 0.72))
  const startX = center.x + radius * 0.5
  const startY = center.y - radius * 0.15
  const endX = center.x
  const endY = center.y - radius * 0.62
  const cp1x = center.x + radius * 1.6
  const cp1y = center.y - radius * 1.45
  const cp2x = center.x + radius * 0.45
  const cp2y = center.y - radius * 1.85
  return `M ${startX} ${startY} C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${endX} ${endY}`
}

function isBufferPresent(step, rankId, bufferId) {
  const presenceMap = props.bufferPresence
  if (!(presenceMap instanceof Map)) return false
  const presence = presenceMap.get(rankId)
  if (!presence || step < 0 || step >= presence.length) return false
  return (presence[step] & (1 << bufferId)) !== 0
}

function isBufferChanged(step, rankId, bufferId) {
  const changedMap = props.bufferChangedFlags
  if (!(changedMap instanceof Map)) return false
  const changedFlags = changedMap.get(rankId)
  if (!changedFlags || step < 0 || step >= changedFlags.length) return false
  return (changedFlags[step] & (1 << bufferId)) !== 0
}

function panelStyle(item) {
  const displayItem = normalizedDisplayItems.value[item.index]
  return {
    width: `${displayItem?.type === 'ellipsis' ? ELLIPSIS_WIDTH : panelWidth.value}px`,
    height: `${displayItem?.type === 'ellipsis' ? ellipsisHeight.value : panelHeight.value}px`,
    top: `${PANEL_TOP}px`,
    transform: `translateX(${item.start}px)`,
  }
}

function rowStyle(row) {
  return {
    top: `${row.top}px`,
    height: `${rankContainerHeight.value}px`,
    '--memview-rank-color': row.color,
    '--memview-rank-color-soft': `${row.color}20`,
  }
}

function cellStyle(step, row, slot) {
  const semanticStyle = cellSemanticStyles.value.get(cellKey(step, row.rankId, slot.id))
  const bufferLayout = resolveVisibleBufferLayout(slot.id)
  const left = bufferLayout
    ? RANK_CONTAINER_PADDING_X + bufferLayout.column * cellStride.value
    : RANK_CONTAINER_PADDING_X
  const top = bufferLayout
    ? RANK_CONTAINER_PADDING_Y + bufferLayout.row * cellStride.value
    : RANK_CONTAINER_PADDING_Y
  return {
    left: `${left}px`,
    width: `${cellWidth.value}px`,
    height: `${cellWidth.value}px`,
    top: `${top}px`,
    '--memview-rank-color': row.color,
    '--memview-rank-color-soft': `${row.color}33`,
    '--memview-rank-color-strong': `${row.color}d9`,
    '--memview-cell-gradient-soft': semanticStyle?.softGradient ?? null,
    '--memview-cell-gradient-strong': semanticStyle?.strongGradient ?? null,
    '--memview-cell-border-color': semanticStyle?.borderColor ?? row.color,
  }
}

function cellClass(step, row, slot) {
  const key = cellKey(step, row.rankId, slot.id)
  const currentStepHasTaskOps = Array.isArray(props.taskOpsSlice) && props.taskOpsSlice.length > 0
  const shouldShowChanged =
    step === clampedCurrentStep.value && currentStepHasTaskOps
      ? currentStepTaskTargetKeys.value.has(key)
      : isBufferChanged(step, row.rankId, slot.id)

  return {
    'is-present': isBufferPresent(step, row.rankId, slot.id),
    'is-changed': shouldShowChanged,
    'is-current-step': step === clampedCurrentStep.value,
  }
}

function handleBodyScroll(event) {
  scrollTop.value = event.target.scrollTop
}

function scrollCurrentIntoView(behavior = 'smooth') {
  if (!bodyWrapperRef.value || props.totalSteps <= 0) return
  const displayIndex = displayIndexByStep.value.get(clampedCurrentStep.value)
  if (!Number.isInteger(displayIndex)) return
  stepVirtualizer.value.scrollToIndex(displayIndex, {
    align: 'center',
    behavior,
  })
}

function handleWheel(event) {
  if (!bodyWrapperRef.value) return
  if (event.shiftKey || Math.abs(event.deltaX) > Math.abs(event.deltaY)) return
  const host = bodyWrapperRef.value
  const canScrollX = host.scrollWidth > host.clientWidth + 1
  if (!canScrollX) return
  event.preventDefault()
  host.scrollBy({ left: event.deltaY, top: 0, behavior: 'auto' })
}

function pickStepFromClientX(clientX) {
  if (!bodyWrapperRef.value || !normalizedDisplayItems.value.length) return null
  const rect = bodyWrapperRef.value.getBoundingClientRect()
  const localX = clientX - rect.left + bodyWrapperRef.value.scrollLeft

  let left = 0
  let right = normalizedDisplayItems.value.length - 1
  let matchedIndex = -1

  while (left <= right) {
    const middle = Math.floor((left + right) / 2)
    const start = displayItemOffsets.value[middle]
    const size = displayItemSizeMap.value[middle] ?? panelStride.value
    if (localX < start) {
      right = middle - 1
      continue
    }
    if (localX > start + size) {
      left = middle + 1
      continue
    }
    matchedIndex = middle
    break
  }

  if (matchedIndex < 0) return null

  const item = normalizedDisplayItems.value[matchedIndex]
  if (item?.type === 'step' && Number.isInteger(item.step)) {
    return item.step
  }

  if (item?.type === 'ellipsis') {
    const start = displayItemOffsets.value[matchedIndex]
    const midpoint = start + (displayItemSizeMap.value[matchedIndex] ?? ELLIPSIS_WIDTH) / 2
    return localX <= midpoint ? item.fromStep : item.toStep
  }

  return null
}

function handleBodyPointerDown(event) {
  if (event.button !== 0) return
  if (event.target?.closest?.('.memview-timeline-panel-cell')) return
  const step = pickStepFromClientX(event.clientX)
  if (step === null) return
  isScrubbing.value = true
  emit('update:currentStep', step)
  window.addEventListener('pointermove', handleBodyPointerMove)
  window.addEventListener('pointerup', handleBodyPointerUp, { once: true })
}

function handleBodyPointerMove(event) {
  if (!isScrubbing.value) return
  const step = pickStepFromClientX(event.clientX)
  if (step === null) return
  emit('update:currentStep', step)
}

function handleBodyPointerUp() {
  isScrubbing.value = false
  window.removeEventListener('pointermove', handleBodyPointerMove)
}

function handleBufferCellClick(step, row, slot) {
  if (!Number.isInteger(step) || !Number.isInteger(row?.rankId) || !Number.isInteger(slot?.id)) return
  emit('update:currentStep', step)
  emit('buffer-select', {
    step,
    rankId: row.rankId,
    bufferId: slot.id,
  })
}

function cellKey(step, rankId, bufferId) {
  return `${step}:${rankId}:${bufferId}`
}

function withAlpha(hexColor, alphaHex) {
  if (typeof hexColor !== 'string' || !hexColor.startsWith('#')) return hexColor
  return `${hexColor}${alphaHex}`
}

function uniqueSemanticRanks(rankIds) {
  return [...new Set((Array.isArray(rankIds) ? rankIds : []).filter((rankId) => Number.isInteger(rankId)))]
}

function buildGradient(colors, alphaHex) {
  if (!colors.length) return null
  if (colors.length === 1) return withAlpha(colors[0], alphaHex)
  const lastIndex = Math.max(colors.length - 1, 1)
  const stops = colors.map((color, index) => `${withAlpha(color, alphaHex)} ${Math.round((index / lastIndex) * 100)}%`)
  return `linear-gradient(135deg, ${stops.join(', ')})`
}

function createSemanticStyle(rankIds, rankOrder = []) {
  const uniqueRanks = uniqueSemanticRanks(rankIds)
  const colors = uniqueRanks.length
    ? uniqueRanks.map((rankId) => rankColorFor(rankId, rankOrder))
    : [rankColorFor(0, rankOrder)]
  const borderColor = colors[0]

  return {
    stroke: colors.length === 1 ? colors[0] : borderColor,
    borderColor,
    softGradient: buildGradient(colors, '33'),
    strongGradient: buildGradient(colors, 'd9'),
    gradient:
      colors.length > 1
        ? {
            colors,
          }
        : null,
  }
}

function resolveTaskOpSemanticRanks(taskOp) {
  if (!taskOp || typeof taskOp !== 'object') return []
  if (taskOp.isReduce) {
    return uniqueSemanticRanks([taskOp.srcRankId, taskOp.dstRankId])
  }
  return uniqueSemanticRanks([Number.isInteger(taskOp.srcRankId) ? taskOp.srcRankId : taskOp.dstRankId])
}

function resolveLayoutSemanticRanks(bufferLayout) {
  if (!bufferLayout || typeof bufferLayout !== 'object') return []
  const srcRanks = uniqueSemanticRanks(bufferLayout.srcRanks)
  if (bufferLayout.isReduce) {
    return uniqueSemanticRanks([bufferLayout.rankId, ...srcRanks])
  }
  return srcRanks.length ? srcRanks : uniqueSemanticRanks([bufferLayout.rankId])
}

function applyLayoutStylesForStep(styles, step) {
  const layoutIndex = props.layoutBuffersIndex
  const offsets = layoutIndex?.offsets
  const items = layoutIndex?.items
  if (!offsets || !items || step < 0 || step >= offsets.length - 1) return

  for (let index = offsets[step]; index < offsets[step + 1]; index += 1) {
    const entry = items[index]
    if (!Number.isInteger(entry?.rankId) || !Number.isInteger(entry?.bufferId)) continue
    styles.set(
      cellKey(step, entry.rankId, entry.bufferId),
      createSemanticStyle(resolveLayoutSemanticRanks(entry), sortedRankIds.value),
    )
  }
}

function mergeStyles(existingStyle, nextRanks) {
  if (!existingStyle) {
    return createSemanticStyle(nextRanks, sortedRankIds.value)
  }

  const existingRanks = uniqueSemanticRanks(existingStyle.gradient?.colors?.map((color) => colorToRankId(color)) ?? [])
  const mergedRanks = uniqueSemanticRanks([...existingRanks, ...nextRanks])
  return createSemanticStyle(mergedRanks.length ? mergedRanks : nextRanks, sortedRankIds.value)
}

function colorToRankId(color) {
  const index = sortedRankIds.value.findIndex((rankId) => rankColorFor(rankId, sortedRankIds.value) === color)
  return index >= 0 ? sortedRankIds.value[index] : null
}

function applyTaskOpOverrides(styles) {
  const step = clampedCurrentStep.value
  const semanticRanksByCell = new Map()

  for (const taskOp of props.taskOpsSlice ?? []) {
    if (!Number.isInteger(taskOp?.dstRankId) || !Number.isInteger(taskOp?.dstBufferId)) continue
    const semanticRanks = resolveTaskOpSemanticRanks(taskOp)
    if (!semanticRanks.length) continue

    const dstKey = cellKey(step, taskOp.dstRankId, taskOp.dstBufferId)
    semanticRanksByCell.set(dstKey, uniqueSemanticRanks([...(semanticRanksByCell.get(dstKey) ?? []), ...semanticRanks]))
  }

  for (const [key, semanticRanks] of semanticRanksByCell.entries()) {
    styles.set(key, createSemanticStyle(semanticRanks, sortedRankIds.value))
  }
}

function adjustZoom(delta) {
  const next = Math.min(MAX_CELL, Math.max(MIN_CELL, cellWidth.value + delta))
  if (next === cellWidth.value) return
  cellWidth.value = next
  nextTick(() => {
    stepVirtualizer.value.measure()
    scrollCurrentIntoView('auto')
  })
}

defineExpose({ adjustZoom })
</script>

<template>
  <div ref="hostRef" class="memview-timeline" @wheel="handleWheel">
    <div class="memview-timeline-grid" :style="{ gridTemplateColumns: `${GUTTER_WIDTH}px 1fr` }">
      <div class="memview-timeline-gutter memview-timeline-gutter--panels">
        <div
          class="memview-timeline-gutter-inner"
          :style="{
            height: `${totalBodyHeight}px`,
            transform: `translateY(-${scrollTop}px)`,
          }"
        >
          <div
            v-for="row in rankRows"
            :key="`rank-${row.rankId}`"
            class="memview-timeline-gutter-row"
            :style="{
              top: `${PANEL_TOP + row.top}px`,
              height: `${rankContainerHeight}px`,
              borderLeftColor: row.color,
            }"
          >
            Rank {{ row.rankId }}
          </div>
        </div>
      </div>

      <div
        ref="bodyWrapperRef"
        class="memview-timeline-body"
        @scroll="handleBodyScroll"
        @pointerdown="handleBodyPointerDown"
      >
        <div
          class="memview-timeline-body-spacer"
          :style="{
            width: `${totalBodyWidth}px`,
            height: `${totalBodyHeight}px`,
          }"
        >
          <svg
            v-if="visibleLineSegments.length"
            class="memview-timeline-overlay"
            :width="totalBodyWidth"
            :height="totalBodyHeight"
            :viewBox="`0 0 ${totalBodyWidth} ${totalBodyHeight}`"
            aria-hidden="true"
          >
            <g v-for="segment in visibleLineSegments" :key="segment.id">
              <path
                class="memview-timeline-ops-line"
                :class="{ 'is-self': segment.isSelf }"
                :d="segment.path"
                :stroke="segment.stroke"
                :stroke-opacity="segment.alpha"
              />
            </g>
          </svg>

          <div
            v-for="item in virtualItems"
            :key="item.key"
            class="memview-timeline-panel"
            :class="{
              'is-current': normalizedDisplayItems[item.index]?.type === 'step'
                && normalizedDisplayItems[item.index]?.step === clampedCurrentStep,
              'is-ellipsis': normalizedDisplayItems[item.index]?.type === 'ellipsis',
            }"
            :style="panelStyle(item)"
            :data-step="normalizedDisplayItems[item.index]?.type === 'step' ? normalizedDisplayItems[item.index]?.step : null"
          >
            <template v-if="normalizedDisplayItems[item.index]?.type === 'step'">
              <span
                class="memview-timeline-step-label"
                :class="{ 'is-current': normalizedDisplayItems[item.index]?.step === clampedCurrentStep }"
              >
                Step {{ normalizedDisplayItems[item.index]?.step }}
              </span>

              <div class="memview-timeline-panel-grid">
                <div
                  v-for="row in rankRows"
                  :key="`step-${normalizedDisplayItems[item.index]?.step}-rank-${row.rankId}`"
                  class="memview-timeline-rank-container"
                  :style="rowStyle(row)"
                >
                  <div
                    v-for="slot in visibleBufferSlots"
                    :key="`step-${normalizedDisplayItems[item.index]?.step}-rank-${row.rankId}-buffer-${slot.id}`"
                    class="memview-timeline-panel-cell"
                    :class="cellClass(normalizedDisplayItems[item.index]?.step, row, slot)"
                    :style="cellStyle(normalizedDisplayItems[item.index]?.step, row, slot)"
                    :data-step="normalizedDisplayItems[item.index]?.step"
                    :data-rank="row.rankId"
                    :data-buffer="slot.id"
                    :title="slot.label"
                    @click.stop="handleBufferCellClick(normalizedDisplayItems[item.index]?.step, row, slot)"
                  />
                </div>
              </div>
            </template>

            <div v-else class="memview-timeline-ellipsis" :title="`Folded ${normalizedDisplayItems[item.index]?.hiddenCount} steps`">
              <span class="memview-timeline-ellipsis__dots">...</span>
              <span class="memview-timeline-ellipsis__meta">
                {{ normalizedDisplayItems[item.index]?.fromStep + 1 }}-{{ normalizedDisplayItems[item.index]?.toStep - 1 }}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

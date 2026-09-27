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
import { computed, ref, watch, nextTick } from 'vue'
import { getInstrSpace, getEntriesForCcu, getEntriesForInstr, getLoopRanges, getLoopColorForInstr } from '../../utils/ccuTraceData'

const props = defineProps({
  traceData: Object,
  currentRankId: Number,
  currentDieId: Number,
  currentEntry: Object,
  currentStepIndex: Number,
  breakpoints: Array,
})

const emit = defineEmits(['toggle-breakpoint', 'has-breakpoint', 'jump-to-step'])

const listRef = ref(null)

// 当前 CCU 的指令空间
const instructions = computed(() => {
  if (!props.traceData) return []
  return getInstrSpace(props.traceData, props.currentRankId, props.currentDieId)
})

// 当前 CCU 的 entries
const ccuEntries = computed(() => {
  if (!props.traceData) return []
  return getEntriesForCcu(props.traceData, props.currentRankId, props.currentDieId)
})

// 当前 CCU 的 Loop 循环体范围
const loopRanges = computed(() => {
  if (!props.traceData) return []
  return getLoopRanges(props.traceData, props.currentRankId, props.currentDieId)
})

// 获取指令所属 Loop 的内联样式（背景色 + 左边框）
function getLoopStyle(instr) {
  const color = getLoopColorForInstr(loopRanges.value, instr.instrId)
  if (!color) return {}
  return {
    backgroundColor: color.bg,
    borderLeft: `3px solid ${color.border}`,
  }
}

// 获取 Loop 标签（如 "Loop #0"）用于 tooltip
function getLoopLabel(instr) {
  const color = getLoopColorForInstr(loopRanges.value, instr.instrId)
  if (!color) return null
  return `Loop #${color.loopIndex} [${loopRanges.value[color.loopIndex].startId}-${loopRanges.value[color.loopIndex].endId}]`
}

// 每条指令的执行状态
function getInstrStatus(instr) {
  const entries = getEntriesForInstr(props.traceData, props.currentRankId, props.currentDieId, instr.instrId)
  if (!entries || entries.length === 0) return null

  // 检查是否是当前 entry
  const isCurrent = props.currentEntry &&
    props.currentEntry.rankId === props.currentRankId &&
    props.currentEntry.dieId === props.currentDieId &&
    props.currentEntry.instrId === instr.instrId

  // 获取最近一次执行（在当前步骤之前的）
  const executed = entries.filter(e => e.globalSeqId <= (props.currentEntry?.globalSeqId ?? Infinity))

  if (executed.length === 0 && !isCurrent) return null

  const last = executed.length > 0 ? executed[executed.length - 1] : null

  if (isCurrent) {
    return {
      type: 'current',
      seqId: props.currentEntry.globalSeqId,
      count: entries.length,
      lastEntry: props.currentEntry,
    }
  }

  if (last?.errorInfo?.hasError) {
    return { type: 'fail', seqId: last.globalSeqId, count: executed.length, lastEntry: last }
  }

  if (last?.waitInfo?.hadWait && last?.execState === 2) {
    return { type: 'hold', seqId: last.globalSeqId, count: executed.length, lastEntry: last }
  }

  return {
    type: 'done',
    seqId: last?.globalSeqId ?? '--',
    round: last?.execRound ?? '--',
    count: executed.length,
    lastEntry: last,
  }
}

function isBreakpoint(instr) {
  return props.breakpoints.some(
    bp => bp.rankId === props.currentRankId && bp.dieId === props.currentDieId && bp.instrId === instr.instrId
  )
}

function isCurrent(instr) {
  return props.currentEntry &&
    props.currentEntry.rankId === props.currentRankId &&
    props.currentEntry.dieId === props.currentDieId &&
    props.currentEntry.instrId === instr.instrId
}

function statusText(instr) {
  const st = getInstrStatus(instr)
  if (!st) return ''
  if (st.type === 'current') return `← #${st.seqId}`
  if (st.type === 'fail') return 'FAIL'
  if (st.type === 'hold') return 'HOLD'
  return `✓ #${st.seqId} R${st.round}`
}

function statusClass(instr) {
  const st = getInstrStatus(instr)
  if (!st) return ''
  return `status--${st.type}`
}

function scrollToCurrent() {
  if (!props.currentEntry || !listRef.value) return
  if (props.currentEntry.rankId !== props.currentRankId || props.currentEntry.dieId !== props.currentDieId) return

  const row = listRef.value.querySelector('.instr-row.is-current')
  if (row) {
    row.scrollIntoView({ block: 'center', behavior: 'smooth' })
  }
}

watch(() => props.currentStepIndex, () => {
  nextTick(scrollToCurrent)
})

watch(() => [props.currentRankId, props.currentDieId], () => {
  nextTick(scrollToCurrent)
})
</script>

<template>
  <div class="instr-list" ref="listRef">
    <div class="instr-list__header">
      <span class="col-bp">BP</span>
      <span class="col-id">instrId</span>
      <span class="col-desc">指令描述</span>
      <span class="col-status">执行状态</span>
    </div>
    <div class="instr-list__body">
      <div
        v-for="instr in instructions"
        :key="instr.instrId"
        class="instr-row"
        :class="{
          'is-current': isCurrent(instr),
          'has-breakpoint': isBreakpoint(instr),
        }"
        :style="!isCurrent(instr) ? getLoopStyle(instr) : {}"
      >
        <span class="col-bp" @click="emit('toggle-breakpoint', currentRankId, currentDieId, instr.instrId)">
          <span v-if="isCurrent(instr)" class="bp-current">▶</span>
          <span v-else-if="isBreakpoint(instr)" class="bp-active">●</span>
          <span v-else class="bp-inactive">○</span>
        </span>
        <span class="col-id">{{ instr.instrId }}</span>
        <span class="col-desc" :title="[getLoopLabel(instr), instr.instrDescribe].filter(Boolean).join(' | ')">{{ instr.instrDescribe }}</span>
        <span class="col-status" :class="statusClass(instr)">{{ statusText(instr) }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.instr-list {
  flex: 1;
  overflow-y: auto;
  font-family: 'JetBrains Mono', 'Consolas', monospace;
  font-size: 12px;
}

.instr-list__header {
  display: flex;
  padding: 6px 8px;
  background: #1a1a1a;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  font-weight: 600;
  font-size: 11px;
  color: rgba(255, 255, 255, 0.5);
  position: sticky;
  top: 0;
  z-index: 1;
}

.instr-list__body {
  overflow-y: auto;
}

.instr-row {
  display: flex;
  padding: 3px 8px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.03);
  transition: background 0.1s;
}

.instr-row:hover {
  background: rgba(255, 255, 255, 0.05);
}

.instr-row.is-current {
  background: rgba(64, 158, 255, 0.15);
  border-left: 3px solid #409eff;
}

.instr-row.has-breakpoint {
  background: rgba(231, 76, 60, 0.08);
}

.instr-row.is-current.has-breakpoint {
  background: rgba(64, 158, 255, 0.15);
}

.col-bp {
  width: 36px;
  min-width: 36px;
  text-align: center;
  cursor: pointer;
}

.col-id {
  width: 56px;
  min-width: 56px;
  color: rgba(255, 255, 255, 0.5);
}

.col-desc {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: rgba(255, 255, 255, 0.85);
}

.col-status {
  width: 120px;
  min-width: 120px;
  text-align: right;
  font-size: 11px;
}

.bp-current {
  color: #409eff;
  font-size: 14px;
}

.bp-active {
  color: #e74c3c;
  font-size: 14px;
}

.bp-inactive {
  color: rgba(255, 255, 255, 0.2);
}

.status--current {
  color: #409eff;
  font-weight: 600;
}

.status--done {
  color: rgba(255, 255, 255, 0.4);
}

.status--hold {
  color: #e6a23c;
  font-weight: 600;
}

.status--fail {
  color: #f56c6c;
  font-weight: 600;
}
</style>

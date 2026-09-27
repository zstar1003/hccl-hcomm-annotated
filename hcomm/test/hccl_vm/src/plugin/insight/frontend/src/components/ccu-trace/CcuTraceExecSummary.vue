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
import { computed } from 'vue'
import { getEntriesForCcu } from '../../utils/ccuTraceData'

const props = defineProps({
  traceData: Object,
  currentStepIndex: Number,
  currentRankId: Number,
  currentDieId: Number,
})

const ccuEntries = computed(() => {
  if (!props.traceData) return []
  return getEntriesForCcu(props.traceData, props.currentRankId, props.currentDieId)
})

// 当前步骤之前的、属于当前 CCU 的 entry 数量
const executedCount = computed(() => {
  if (!props.traceData || !props.traceData.globalEntries[props.currentStepIndex]) return 0
  const currentSeqId = props.traceData.globalEntries[props.currentStepIndex].globalSeqId
  return ccuEntries.value.filter(e => e.globalSeqId <= currentSeqId).length
})

// 当前 CCU 涉及的 round 集合
const rounds = computed(() => {
  const s = new Set()
  for (const e of ccuEntries.value) {
    if (e.globalSeqId <= (props.traceData?.globalEntries[props.currentStepIndex]?.globalSeqId ?? Infinity)) {
      s.add(e.execRound)
    }
  }
  return [...s].sort((a, b) => a - b)
})

// 指令类型统计
const typeStats = computed(() => {
  const counts = {}
  for (const e of ccuEntries.value) {
    if (e.globalSeqId > (props.traceData?.globalEntries[props.currentStepIndex]?.globalSeqId ?? Infinity)) break
    const t = e.detail?.typeName || 'Unknown'
    counts[t] = (counts[t] || 0) + 1
  }
  return Object.entries(counts).sort((a, b) => b[1] - a[1])
})
</script>

<template>
  <div class="exec-summary">
    <div class="exec-summary__header">执行摘要</div>

    <div class="summary-stat">
      <span class="stat-label">已执行</span>
      <span class="stat-value">{{ executedCount }} 条</span>
    </div>
    <div class="summary-stat">
      <span class="stat-label">CCU 总指令</span>
      <span class="stat-value">{{ ccuEntries.length }} 条</span>
    </div>
    <div class="summary-stat">
      <span class="stat-label">Rounds</span>
      <span class="stat-value">{{ rounds.join(', ') || '--' }}</span>
    </div>

    <div v-if="typeStats.length > 0" class="summary-types">
      <div class="summary-types__title">指令类型分布</div>
      <div v-for="[type, count] in typeStats" :key="type" class="type-row">
        <span class="type-name">{{ type }}</span>
        <span class="type-count">{{ count }}</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.exec-summary {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.exec-summary__header {
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: rgba(255, 255, 255, 0.6);
  margin-bottom: 4px;
}

.summary-stat {
  display: flex;
  justify-content: space-between;
  font-size: 11px;
  padding: 1px 0;
}

.stat-label {
  color: rgba(255, 255, 255, 0.5);
}

.stat-value {
  color: rgba(255, 255, 255, 0.85);
  font-family: monospace;
}

.summary-types {
  margin-top: 8px;
}

.summary-types__title {
  font-size: 11px;
  font-weight: 600;
  color: rgba(255, 255, 255, 0.5);
  margin-bottom: 4px;
}

.type-row {
  display: flex;
  justify-content: space-between;
  font-size: 10px;
  padding: 1px 0;
}

.type-name {
  color: rgba(255, 255, 255, 0.7);
}

.type-count {
  color: rgba(255, 255, 255, 0.5);
  font-family: monospace;
}
</style>

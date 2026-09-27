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

const props = defineProps({
  traceData: Object,
  currentStepIndex: Number,
  currentRankId: Number,
  currentDieId: Number,
  ccuList: Array,
  progressPercent: Number,
  currentEntry: Object,
})

const emit = defineEmits(['ccu-change', 'jump-to-step'])

const rankOptions = computed(() => {
  const ranks = new Set()
  for (const ccu of props.ccuList) ranks.add(ccu.rankId)
  return [...ranks].sort((a, b) => a - b)
})

const dieOptions = computed(() => {
  return props.ccuList
    .filter((c) => c.rankId === props.currentRankId)
    .map((c) => c.dieId)
    .sort((a, b) => a - b)
})

const currentRound = computed(() => props.currentEntry?.execRound ?? '--')
const currentSeqId = computed(() => props.currentEntry?.globalSeqId ?? '--')

function onRankChange(rankId) {
  const dies = props.ccuList.filter((c) => c.rankId === rankId).map((c) => c.dieId)
  const dieId = dies.length > 0 ? dies[0] : 0
  emit('ccu-change', { rankId, dieId })
}

function onDieChange(dieId) {
  emit('ccu-change', { rankId: props.currentRankId, dieId })
}

function onSliderChange(value) {
  emit('jump-to-step', value)
}
</script>

<template>
  <div class="progress-bar">
    <div class="progress-bar__selectors">
      <span class="selector-label">Rank</span>
      <el-select size="small" :model-value="currentRankId" @change="onRankChange" style="width: 80px">
        <el-option v-for="r in rankOptions" :key="r" :label="'R' + r" :value="r" />
      </el-select>

      <span class="selector-label">Die</span>
      <el-select size="small" :model-value="currentDieId" @change="onDieChange" style="width: 80px">
        <el-option v-for="d in dieOptions" :key="d" :label="'D' + d" :value="d" />
      </el-select>

      <span class="progress-meta">
        #{{ currentSeqId }} &middot; Round {{ currentRound }}
      </span>
    </div>

    <div class="progress-bar__slider">
      <el-slider
        :model-value="currentStepIndex"
        :min="0"
        :max="(traceData?.totalEntries || 1) - 1"
        :show-tooltip="true"
        @change="onSliderChange"
        :format-tooltip="(v) => `#${v + 1}`"
      />
      <span class="progress-percent">{{ progressPercent }}%</span>
    </div>
  </div>
</template>

<style scoped>
.progress-bar {
  padding: 8px 16px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.progress-bar__selectors {
  display: flex;
  align-items: center;
  gap: 8px;
}

.selector-label {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.5);
}

.progress-meta {
  margin-left: auto;
  font-size: 12px;
  font-family: monospace;
  color: rgba(255, 255, 255, 0.6);
}

.progress-bar__slider {
  display: flex;
  align-items: center;
  gap: 12px;
}

.progress-bar__slider :deep(.el-slider) {
  flex: 1;
}

.progress-percent {
  font-size: 12px;
  font-family: monospace;
  color: rgba(255, 255, 255, 0.5);
  min-width: 36px;
  text-align: right;
}
</style>

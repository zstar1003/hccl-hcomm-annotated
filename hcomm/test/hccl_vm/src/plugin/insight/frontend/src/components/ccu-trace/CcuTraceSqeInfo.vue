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
import { getSqeTask, formatHex } from '../../utils/ccuTraceData'

const props = defineProps({
  traceData: Object,
  currentEntry: Object,
})

const sqeTask = computed(() => {
  if (!props.traceData || !props.currentEntry) return null
  return getSqeTask(props.traceData, props.currentEntry.sqeTaskId)
})

// 当前指令 ID（动态，随回放变化）
const curInstrId = computed(() => props.currentEntry?.instrId ?? '--')

// SQE 参数列表
const sqeArgs = computed(() => {
  if (!sqeTask.value || !sqeTask.value.args) return []
  return sqeTask.value.args
})
</script>

<template>
  <div class="sqe-info">
    <div class="sqe-info__header">SQE 任务</div>

    <div v-if="!currentEntry || !sqeTask" class="sqe-empty">
      {{ currentEntry ? '无 SQE 信息' : '无选中指令' }}
    </div>

    <template v-else>
      <div class="sqe-row">
        <span class="sqe-label">sqeTaskId</span>
        <span class="sqe-value">{{ sqeTask.sqeTaskId }}</span>
      </div>
      <div class="sqe-row">
        <span class="sqe-label">missionId</span>
        <span class="sqe-value">{{ sqeTask.missionId }}</span>
      </div>
      <div class="sqe-row">
        <span class="sqe-label">instStartId</span>
        <span class="sqe-value">{{ sqeTask.instStartId }}</span>
      </div>
      <div class="sqe-row sqe-row--highlight">
        <span class="sqe-label">curInstrId</span>
        <span class="sqe-value sqe-value--current">{{ curInstrId }}</span>
      </div>
      <div class="sqe-row">
        <span class="sqe-label">instCnt</span>
        <span class="sqe-value">{{ sqeTask.instCnt }}</span>
      </div>
      <div class="sqe-row">
        <span class="sqe-label">simulator</span>
        <span class="sqe-value sqe-value--mono">{{ sqeTask.simulatorPtr }}</span>
      </div>
      <div class="sqe-row">
        <span class="sqe-label">firstRound</span>
        <span class="sqe-value">{{ sqeTask.firstExecRound }}</span>
      </div>

      <!-- SQE 参数列表 -->
      <div v-if="sqeArgs.length > 0" class="sqe-args">
        <div class="sqe-args__header">参数列表 ({{ sqeArgs.length }})</div>
        <div v-for="(arg, idx) in sqeArgs" :key="idx" class="sqe-arg-row">
          <span class="sqe-arg-idx">args[{{ idx }}]</span>
          <span class="sqe-arg-val">{{ formatHex(arg) }}</span>
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.sqe-info {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.sqe-info__header {
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: rgba(255, 255, 255, 0.6);
}

.sqe-empty {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.3);
}

.sqe-row {
  display: flex;
  justify-content: space-between;
  font-size: 11px;
  padding: 1px 0;
}

.sqe-row--highlight {
  background: rgba(64, 158, 255, 0.1);
  padding: 2px 4px;
  border-radius: 3px;
}

.sqe-label {
  color: rgba(255, 255, 255, 0.5);
}

.sqe-value {
  color: rgba(255, 255, 255, 0.85);
}

.sqe-value--current {
  color: #409eff;
  font-weight: 600;
}

.sqe-value--mono {
  font-family: monospace;
  font-size: 10px;
  color: rgba(255, 255, 255, 0.7);
}

/* SQE 参数列表 */
.sqe-args {
  margin-top: 8px;
  padding-top: 6px;
  border-top: 1px solid rgba(255, 255, 255, 0.08);
}

.sqe-args__header {
  font-size: 11px;
  font-weight: 600;
  color: rgba(255, 255, 255, 0.5);
  margin-bottom: 4px;
}

.sqe-arg-row {
  display: flex;
  justify-content: space-between;
  font-size: 10px;
  padding: 1px 4px;
  font-family: monospace;
}

.sqe-arg-idx {
  color: rgba(255, 255, 255, 0.5);
}

.sqe-arg-val {
  color: rgba(255, 255, 255, 0.8);
}
</style>

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
import { buildCumulativeState, formatHex } from '../../utils/ccuTraceData'

const props = defineProps({
  traceData: Object,
  currentRankId: Number,
  currentDieId: Number,
  currentStepIndex: Number,
  currentEntry: Object,
})

const cumState = computed(() => {
  if (!props.traceData || !props.currentEntry) return null
  return buildCumulativeState(props.traceData, props.currentRankId, props.currentDieId, props.currentEntry.globalSeqId)
})

// 显示前 32 个 XN 寄存器
const xnSlice = computed(() => {
  if (!cumState.value) return []
  const result = []
  const xn = cumState.value.xn
  // 收集所有有值的 key，取前 32
  const keys = [...xn.keys()].sort((a, b) => a - b)
  for (const k of keys.slice(0, 32)) {
    result.push({ id: k, value: xn.get(k) })
  }
  return result
})

const gsaSlice = computed(() => {
  if (!cumState.value) return []
  const result = []
  const gsa = cumState.value.gsa
  const keys = [...gsa.keys()].sort((a, b) => a - b)
  for (const k of keys.slice(0, 32)) {
    result.push({ id: k, value: gsa.get(k) })
  }
  return result
})

const ckeSlice = computed(() => {
  if (!cumState.value) return []
  const result = []
  const cke = cumState.value.cke
  const keys = [...cke.keys()].sort((a, b) => a - b)
  for (const k of keys.slice(0, 16)) {
    result.push({ id: k, value: cke.get(k) })
  }
  return result
})
</script>

<template>
  <div class="resource-overview">
    <div class="panel-header">CCU 资源概览</div>
    <div class="panel-subheader">Rank{{ currentRankId }}:Die{{ currentDieId }} 累积状态</div>

    <div v-if="!cumState" class="panel-empty">无数据</div>

    <template v-else>
      <!-- XN -->
      <div v-if="xnSlice.length > 0" class="overview-section">
        <div class="overview-section__title">XN ({{ cumState.xn.size }} 项)</div>
        <div class="reg-grid">
          <div v-for="r in xnSlice" :key="r.id" class="reg-cell">
            <span class="reg-id">{{ r.id }}</span>
            <span class="reg-val">{{ formatHex(r.value) }}</span>
          </div>
        </div>
      </div>

      <!-- GSA -->
      <div v-if="gsaSlice.length > 0" class="overview-section">
        <div class="overview-section__title">GSA ({{ cumState.gsa.size }} 项)</div>
        <div class="reg-grid">
          <div v-for="r in gsaSlice" :key="r.id" class="reg-cell">
            <span class="reg-id">{{ r.id }}</span>
            <span class="reg-val">{{ formatHex(r.value) }}</span>
          </div>
        </div>
      </div>

      <!-- CKE -->
      <div v-if="ckeSlice.length > 0" class="overview-section">
        <div class="overview-section__title">CKE ({{ cumState.cke.size }} 项)</div>
        <div class="reg-grid">
          <div v-for="r in ckeSlice" :key="r.id" class="reg-cell">
            <span class="reg-id">{{ r.id }}</span>
            <span class="reg-val">{{ r.value }}</span>
          </div>
        </div>
      </div>

      <div v-if="xnSlice.length === 0 && gsaSlice.length === 0 && ckeSlice.length === 0" class="panel-empty">
        当前 CCU 无资源变化记录
      </div>
    </template>
  </div>
</template>

<style scoped>
.resource-overview {
  padding: 8px 12px;
}

.panel-header {
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: rgba(255, 255, 255, 0.6);
  margin-bottom: 2px;
}

.panel-subheader {
  font-size: 11px;
  color: rgba(255, 255, 255, 0.4);
  margin-bottom: 8px;
}

.panel-empty {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.3);
}

.overview-section {
  margin-bottom: 12px;
}

.overview-section__title {
  font-size: 11px;
  font-weight: 600;
  color: rgba(255, 255, 255, 0.5);
  margin-bottom: 4px;
}

.reg-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
  gap: 2px;
}

.reg-cell {
  display: flex;
  gap: 4px;
  font-family: monospace;
  font-size: 10px;
  padding: 1px 4px;
  background: rgba(255, 255, 255, 0.03);
  border-radius: 2px;
}

.reg-id {
  color: rgba(255, 255, 255, 0.4);
  min-width: 24px;
}

.reg-val {
  color: rgba(255, 255, 255, 0.8);
  overflow: hidden;
  text-overflow: ellipsis;
}
</style>

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
import { formatHex } from '../../utils/ccuTraceData'

const props = defineProps({
  currentEntry: Object,
})

const delta = computed(() => props.currentEntry?.resourceDelta || {})
const xnChanges = computed(() => delta.value?.xnChanges || [])
const gsaChanges = computed(() => delta.value?.gsaChanges || [])
const ckeChanges = computed(() => delta.value?.ckeChanges || [])

const hasChanges = computed(() =>
  xnChanges.value.length > 0 || gsaChanges.value.length > 0 || ckeChanges.value.length > 0
)
</script>

<template>
  <div class="resource-delta">
    <div class="panel-header">资源变化</div>

    <div v-if="!currentEntry" class="panel-empty">无选中指令</div>
    <div v-else-if="!hasChanges" class="panel-empty">本次执行无资源变化</div>

    <template v-else>
      <!-- XN Changes -->
      <div v-if="xnChanges.length > 0" class="delta-section">
        <div class="delta-section__title">XN Changes</div>
        <div v-for="ch in xnChanges" :key="ch.id" class="delta-row">
          <span class="delta-id">Xn[{{ ch.id }}]</span>
          <span class="delta-before">{{ formatHex(ch.valueBefore) }}</span>
          <span class="delta-arrow">→</span>
          <span class="delta-after">{{ formatHex(ch.valueAfter) }}</span>
        </div>
      </div>

      <!-- GSA Changes -->
      <div v-if="gsaChanges.length > 0" class="delta-section">
        <div class="delta-section__title">GSA Changes</div>
        <div v-for="ch in gsaChanges" :key="ch.id" class="delta-row">
          <span class="delta-id">GSA[{{ ch.id }}]</span>
          <span class="delta-before">{{ formatHex(ch.valueBefore) }}</span>
          <span class="delta-arrow">→</span>
          <span class="delta-after">{{ formatHex(ch.valueAfter) }}</span>
        </div>
      </div>

      <!-- CKE Changes -->
      <div v-if="ckeChanges.length > 0" class="delta-section">
        <div class="delta-section__title">CKE Changes</div>
        <div v-for="ch in ckeChanges" :key="ch.id" class="delta-row">
          <span class="delta-id">CKE[{{ ch.id }}]</span>
          <span class="delta-before">{{ ch.valueBefore }}</span>
          <span class="delta-arrow">→</span>
          <span class="delta-after">{{ ch.valueAfter }}</span>
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.resource-delta {
  padding: 8px 12px;
}

.panel-header {
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: rgba(255, 255, 255, 0.6);
  margin-bottom: 8px;
}

.panel-empty {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.3);
}

.delta-section {
  margin-bottom: 12px;
}

.delta-section__title {
  font-size: 11px;
  font-weight: 600;
  color: rgba(255, 255, 255, 0.5);
  margin-bottom: 4px;
}

.delta-row {
  display: flex;
  align-items: center;
  gap: 6px;
  font-family: monospace;
  font-size: 11px;
  padding: 2px 0;
}

.delta-id {
  color: rgba(255, 255, 255, 0.7);
  min-width: 64px;
}

.delta-before {
  color: rgba(255, 255, 255, 0.4);
}

.delta-arrow {
  color: rgba(255, 255, 255, 0.3);
}

.delta-after {
  color: #67c23a;
}
</style>

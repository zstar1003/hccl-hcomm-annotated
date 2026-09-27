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
  traceData: Object,
})

const detail = computed(() => props.currentEntry?.detail || {})
const typeName = computed(() => detail.value?.typeName || '--')
const args = computed(() => detail.value?.args || {})

const context = computed(() => props.currentEntry?.context || {})
const waitInfo = computed(() => props.currentEntry?.waitInfo || {})
const errorInfo = computed(() => props.currentEntry?.errorInfo || {})

const contextEntries = computed(() => {
  const ctx = context.value
  if (!ctx || typeof ctx !== 'object') return []
  return Object.entries(ctx).filter(([k, v]) => v !== 0 && v !== false)
})
</script>

<template>
  <div class="instr-detail">
    <div class="panel-header">指令细节</div>

    <div v-if="!currentEntry" class="panel-empty">无选中指令</div>

    <template v-else>
      <!-- 基本信息 -->
      <div class="detail-section">
        <div class="detail-row">
          <span class="detail-label">指令类型</span>
          <span class="detail-value">{{ typeName }}</span>
        </div>
        <div class="detail-row">
          <span class="detail-label">CCU</span>
          <span class="detail-value">Rank{{ currentEntry.rankId }}:Die{{ currentEntry.dieId }}</span>
        </div>
        <div class="detail-row">
          <span class="detail-label">instrId</span>
          <span class="detail-value">{{ currentEntry.instrId }}</span>
        </div>
        <div class="detail-row">
          <span class="detail-label">Round</span>
          <span class="detail-value">{{ currentEntry.execRound }}</span>
        </div>
        <div class="detail-row">
          <span class="detail-label">类别</span>
          <span class="detail-value">{{ currentEntry.category }}</span>
        </div>
      </div>

      <!-- 动态参数 -->
      <div v-if="Object.keys(args).length > 0" class="detail-section">
        <div class="detail-section__title">运行时参数</div>
        <div v-for="(val, key) in args" :key="key" class="detail-row">
          <span class="detail-label">{{ key }}</span>
          <span class="detail-value detail-value--mono">{{ formatHex(val) }}</span>
        </div>
      </div>

      <!-- 执行上下文 -->
      <div v-if="contextEntries.length > 0" class="detail-section">
        <div class="detail-section__title">执行上下文</div>
        <div v-for="[key, val] in contextEntries" :key="key" class="detail-row">
          <span class="detail-label">{{ key }}</span>
          <span class="detail-value detail-value--mono">{{ formatHex(val) }}</span>
        </div>
      </div>

      <!-- CKE Wait 信息 -->
      <div v-if="waitInfo.hadWait" class="detail-section">
        <div class="detail-section__title">CKE Wait 信息</div>
        <div class="detail-row">
          <span class="detail-label">waitCKE</span>
          <span class="detail-value">CKE[{{ waitInfo.waitCKEId }}]&0x{{ waitInfo.waitCKEMask?.toString(16) }}</span>
        </div>
        <div class="detail-row">
          <span class="detail-label">自旋次数</span>
          <span class="detail-value">{{ waitInfo.waitRetryCount }} 轮</span>
        </div>
        <div class="detail-row">
          <span class="detail-label">首次CKE</span>
          <span class="detail-value">{{ waitInfo.ckeValueOnFirstCheck }}</span>
        </div>
        <div class="detail-row">
          <span class="detail-label">通过CKE</span>
          <span class="detail-value">{{ waitInfo.ckeValueOnPass }}</span>
        </div>
      </div>

      <!-- 错误信息 -->
      <div v-if="errorInfo.hasError" class="detail-section detail-section--error">
        <div class="detail-section__title">错误信息</div>
        <div class="detail-row">
          <span class="detail-label">message</span>
          <span class="detail-value">{{ errorInfo.errorMessage }}</span>
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.instr-detail {
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

.detail-section {
  margin-bottom: 12px;
}

.detail-section__title {
  font-size: 11px;
  font-weight: 600;
  color: rgba(255, 255, 255, 0.5);
  margin-bottom: 4px;
  padding-bottom: 2px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.detail-section--error .detail-section__title {
  color: #f56c6c;
}

.detail-row {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  font-size: 11px;
  padding: 2px 0;
}

.detail-label {
  color: rgba(255, 255, 255, 0.5);
}

.detail-value {
  color: rgba(255, 255, 255, 0.85);
  font-family: monospace;
}

.detail-value--mono {
  color: #67c23a;
}
</style>

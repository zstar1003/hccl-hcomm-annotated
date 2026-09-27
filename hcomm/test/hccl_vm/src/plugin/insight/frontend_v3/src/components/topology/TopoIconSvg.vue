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
const props = defineProps({
  type: {
    type: String,
    required: true,
    validator: (v) => ['npu', 'cpu', 'switch', 'server', 'superpod'].includes(v),
  },
  size: {
    type: Number,
    default: 40,
  },
})
</script>

<template>
  <svg
    :width="size"
    :height="size"
    viewBox="0 0 40 40"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    class="topo-icon"
    :class="`topo-icon--${type}`"
  >
    <template v-if="type === 'npu'">
      <rect x="4" y="8" width="32" height="24" rx="3" stroke="currentColor" stroke-width="1.5" />
      <rect x="8" y="12" width="24" height="16" rx="1.5" stroke="currentColor" stroke-width="1" opacity="0.6" />
      <line x1="12" y1="6" x2="12" y2="8" stroke="currentColor" stroke-width="1.5" />
      <line x1="20" y1="6" x2="20" y2="8" stroke="currentColor" stroke-width="1.5" />
      <line x1="28" y1="6" x2="28" y2="8" stroke="currentColor" stroke-width="1.5" />
      <line x1="12" y1="32" x2="12" y2="34" stroke="currentColor" stroke-width="1.5" />
      <line x1="20" y1="32" x2="20" y2="34" stroke="currentColor" stroke-width="1.5" />
      <line x1="28" y1="32" x2="28" y2="34" stroke="currentColor" stroke-width="1.5" />
      <rect x="14" y="16" width="12" height="8" rx="1" fill="currentColor" opacity="0.25" />
    </template>

    <template v-else-if="type === 'cpu'">
      <rect x="6" y="6" width="28" height="28" rx="3" stroke="currentColor" stroke-width="1.5" />
      <rect x="10" y="10" width="20" height="20" rx="1.5" stroke="currentColor" stroke-width="1" opacity="0.6" />
      <line x1="12" y1="4" x2="12" y2="6" stroke="currentColor" stroke-width="1.5" />
      <line x1="20" y1="4" x2="20" y2="6" stroke="currentColor" stroke-width="1.5" />
      <line x1="28" y1="4" x2="28" y2="6" stroke="currentColor" stroke-width="1.5" />
      <line x1="12" y1="34" x2="12" y2="36" stroke="currentColor" stroke-width="1.5" />
      <line x1="20" y1="34" x2="20" y2="36" stroke="currentColor" stroke-width="1.5" />
      <line x1="28" y1="34" x2="28" y2="36" stroke="currentColor" stroke-width="1.5" />
      <line x1="4" y1="12" x2="6" y2="12" stroke="currentColor" stroke-width="1.5" />
      <line x1="4" y1="20" x2="6" y2="20" stroke="currentColor" stroke-width="1.5" />
      <line x1="4" y1="28" x2="6" y2="28" stroke="currentColor" stroke-width="1.5" />
      <line x1="34" y1="12" x2="36" y2="12" stroke="currentColor" stroke-width="1.5" />
      <line x1="34" y1="20" x2="36" y2="20" stroke="currentColor" stroke-width="1.5" />
      <line x1="34" y1="28" x2="36" y2="28" stroke="currentColor" stroke-width="1.5" />
    </template>

    <template v-else-if="type === 'switch'">
      <rect x="4" y="12" width="32" height="16" rx="3" stroke="currentColor" stroke-width="1.5" />
      <circle cx="10" cy="20" r="2" fill="currentColor" opacity="0.5" />
      <circle cx="16" cy="20" r="2" fill="currentColor" opacity="0.5" />
      <circle cx="22" cy="20" r="2" fill="currentColor" opacity="0.5" />
      <circle cx="28" cy="20" r="2" fill="currentColor" opacity="0.5" />
      <line x1="10" y1="10" x2="10" y2="12" stroke="currentColor" stroke-width="1.5" />
      <line x1="16" y1="10" x2="16" y2="12" stroke="currentColor" stroke-width="1.5" />
      <line x1="22" y1="10" x2="22" y2="12" stroke="currentColor" stroke-width="1.5" />
      <line x1="28" y1="10" x2="28" y2="12" stroke="currentColor" stroke-width="1.5" />
      <line x1="10" y1="28" x2="10" y2="30" stroke="currentColor" stroke-width="1.5" />
      <line x1="16" y1="28" x2="16" y2="30" stroke="currentColor" stroke-width="1.5" />
      <line x1="22" y1="28" x2="22" y2="30" stroke="currentColor" stroke-width="1.5" />
      <line x1="28" y1="28" x2="28" y2="30" stroke="currentColor" stroke-width="1.5" />
    </template>

    <template v-else-if="type === 'server'">
      <rect x="6" y="4" width="28" height="32" rx="3" stroke="currentColor" stroke-width="1.5" />
      <line x1="10" y1="12" x2="30" y2="12" stroke="currentColor" stroke-width="0.8" opacity="0.4" />
      <line x1="10" y1="20" x2="30" y2="20" stroke="currentColor" stroke-width="0.8" opacity="0.4" />
      <line x1="10" y1="28" x2="30" y2="28" stroke="currentColor" stroke-width="0.8" opacity="0.4" />
      <circle cx="13" cy="8" r="1.5" fill="currentColor" opacity="0.4" />
      <circle cx="13" cy="16" r="1.5" fill="currentColor" opacity="0.4" />
      <circle cx="13" cy="24" r="1.5" fill="currentColor" opacity="0.4" />
      <rect x="18" y="6" width="12" height="4" rx="1" fill="currentColor" opacity="0.15" />
      <rect x="18" y="14" width="12" height="4" rx="1" fill="currentColor" opacity="0.15" />
      <rect x="18" y="22" width="12" height="4" rx="1" fill="currentColor" opacity="0.15" />
    </template>

    <template v-else-if="type === 'superpod'">
      <rect x="2" y="2" width="36" height="36" rx="4" stroke="currentColor" stroke-width="1.5" stroke-dasharray="4 2" />
      <rect x="8" y="8" width="10" height="10" rx="2" stroke="currentColor" stroke-width="1" opacity="0.5" />
      <rect x="22" y="8" width="10" height="10" rx="2" stroke="currentColor" stroke-width="1" opacity="0.5" />
      <rect x="8" y="22" width="10" height="10" rx="2" stroke="currentColor" stroke-width="1" opacity="0.5" />
      <rect x="22" y="22" width="10" height="10" rx="2" stroke="currentColor" stroke-width="1" opacity="0.5" />
    </template>
  </svg>
</template>

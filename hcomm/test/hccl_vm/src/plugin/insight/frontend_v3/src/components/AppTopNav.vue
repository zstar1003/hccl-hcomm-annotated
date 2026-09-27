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
import { DataAnalysis, Odometer, Share, WarningFilled } from '@element-plus/icons-vue'

defineProps({
  pages: {
    type: Array,
    required: true,
  },
  activePage: {
    type: String,
    required: true,
  },
})

const emit = defineEmits(['change'])

function handleChange(pageId) {
  emit('change', pageId)
}

const iconMap = {
  dashboard: Odometer,
  'mem-view': Share,
  analytic: WarningFilled,
}
</script>

<template>
  <header class="top-nav">
    <div class="top-nav__brand">
      <div class="top-nav__logo">
        <el-icon><DataAnalysis /></el-icon>
      </div>
      <h1 class="top-nav__title">HVRM Insight</h1>
    </div>

    <nav class="top-nav__tabs" aria-label="Primary">
      <button
        v-for="page in pages"
        :key="page.id"
        type="button"
        class="top-nav__tab"
        :class="{ 'is-active': activePage === page.id }"
        @click="handleChange(page.id)"
      >
        <el-icon><component :is="iconMap[page.id]" /></el-icon>
        <span>{{ page.label }}</span>
      </button>
    </nav>
  </header>
</template>

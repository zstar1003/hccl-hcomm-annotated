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
import { computed, ref } from 'vue'
import AppTopNav from './components/AppTopNav.vue'
import AnalyticPage from './pages/AnalyticPage.vue'
import DashboardPage from './pages/DashboardPage.vue'
import MemViewPage from './pages/MemViewPage.vue'

const pages = [
  { id: 'dashboard', label: '总览' },
  { id: 'mem-view', label: '关联' },
  { id: 'analytic', label: '报错' },
]

const activePage = ref('dashboard')

const pageComponents = {
  dashboard: DashboardPage,
  'mem-view': MemViewPage,
  analytic: AnalyticPage,
}

const activeComponent = computed(() => pageComponents[activePage.value] ?? DashboardPage)
</script>

<template>
  <div class="app-shell">
    <AppTopNav :pages="pages" :active-page="activePage" @change="activePage = $event" />
    <main class="app-main">
      <component :is="activeComponent" @navigate-page="activePage = $event" />
    </main>
  </div>
</template>

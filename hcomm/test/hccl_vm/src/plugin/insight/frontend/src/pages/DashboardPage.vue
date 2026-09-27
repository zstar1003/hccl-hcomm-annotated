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
import { computed, onMounted, ref } from 'vue'
import sidebarHandle from '../assets/sidebar-handle.svg'
import sidebarHandleExpand from '../assets/sidebar-handle-expand.svg'
import { useInsightDatasetState } from '../composables/useInsightDatasetState'
import DashboardCurrentRecord from '../components/dashboard/DashboardCurrentRecord.vue'
import DashboardDataTablePanel from '../components/dashboard/DashboardDataTablePanel.vue'
import DashboardSidebar from '../components/dashboard/DashboardSidebar.vue'
import DashboardTopologyPanel from '../components/dashboard/DashboardTopologyPanel.vue'

const CENTER_TOP_MIN_SIZE = 250
const CENTER_TOP_MAX_SIZE = 620
const CENTER_TOP_DEFAULT_RATIO = 0.6

const emit = defineEmits(['navigate-page'])

const sidebarCollapsed = ref(false)
const recordCollapsed = ref(false)
const leftPanelSize = ref(376)
const rightPanelSize = ref(540)
const centerTopSize = ref(resolveDefaultCenterTopSize())

const {
  EMPTY_TEXT,
  selectedDatasetName,
  selectedRankKeys,
  datasetDetail,
  loadingList,
  loadingDetail,
  usingFallback,
  selectedDataset,
  hasData,
  selectedRankCount,
  fileStats,
  summaryRows,
  overviewStats,
  tableRows,
  treeData,
  ensureDatasetsLoaded,
  selectDataset,
  setSelectedRankKeys,
  formatCount,
} = useInsightDatasetState()

const renderedLeftPanelSize = computed(() => (sidebarCollapsed.value ? 28 : leftPanelSize.value))
const renderedRightPanelSize = computed(() => (recordCollapsed.value ? 28 : rightPanelSize.value))
const renderedCenterTopSize = computed(() => clampCenterTopSize(centerTopSize.value))

function handleTreeCheck(checkedKeys) {
  setSelectedRankKeys(checkedKeys)
}

function handleTableRowSelect(row) {
  if (!row?.datasetKey || row.datasetKey === selectedDatasetName.value) {
    return
  }

  selectDataset(row.datasetKey)
}

function handleLeftPanelResize(size) {
  if (!sidebarCollapsed.value && typeof size === 'number') {
    leftPanelSize.value = size
  }
}

function handleRightPanelResize(size) {
  if (!recordCollapsed.value && typeof size === 'number') {
    rightPanelSize.value = size
  }
}

function handleCenterTopResize(size) {
  if (typeof size === 'number') {
    centerTopSize.value = clampCenterTopSize(size)
  }
}

function toggleSidebar() {
  sidebarCollapsed.value = !sidebarCollapsed.value
}

function toggleRecordPanel() {
  recordCollapsed.value = !recordCollapsed.value
}

function handleNavigatePage(pageId) {
  if (typeof pageId !== 'string' || !pageId) return
  emit('navigate-page', pageId)
}

function clampCenterTopSize(size) {
  return Math.min(CENTER_TOP_MAX_SIZE, Math.max(CENTER_TOP_MIN_SIZE, Math.round(Number(size) || 0)))
}

function resolveDefaultCenterTopSize() {
  if (typeof window === 'undefined') return 360
  const layoutHeight = Math.max(0, window.innerHeight - 62)
  return clampCenterTopSize(layoutHeight * CENTER_TOP_DEFAULT_RATIO)
}

onMounted(() => {
  centerTopSize.value = resolveDefaultCenterTopSize()
  ensureDatasetsLoaded()
})
</script>

<template>
  <section class="dashboard-layout">
    <el-splitter class="dashboard-layout-splitter" lazy>
      <el-splitter-panel
        class="dashboard-split-panel dashboard-split-panel--sidebar"
        :size="renderedLeftPanelSize"
        :min="sidebarCollapsed ? 28 : 280"
        :resizable="!sidebarCollapsed"
        @update:size="handleLeftPanelResize"
      >
        <aside class="dashboard-sidebar-shell" :class="{ 'is-collapsed': sidebarCollapsed }">
          <DashboardSidebar
            :selected-rank-count="selectedRankCount"
            :selected-dataset="selectedDataset"
            :tree-data="treeData"
            :checked-keys="selectedRankKeys"
            :format-count="formatCount"
            :empty-text="loadingList ? 'Loading...' : EMPTY_TEXT"
            @tree-check="handleTreeCheck"
          />

          <button
            v-if="!sidebarCollapsed"
            type="button"
            class="dashboard-sidebar-toggle dashboard-sidebar-toggle--collapse"
            aria-label="Collapse sidebar"
            @click="toggleSidebar"
          >
            <img :src="sidebarHandle" alt="" class="dashboard-sidebar-toggle__image" />
          </button>

          <button
            v-else
            type="button"
            class="dashboard-sidebar-toggle dashboard-sidebar-toggle--expand"
            aria-label="Expand sidebar"
            @click="toggleSidebar"
          >
            <img :src="sidebarHandleExpand" alt="" class="dashboard-sidebar-toggle__image" />
          </button>
        </aside>
      </el-splitter-panel>

      <el-splitter-panel class="dashboard-split-panel dashboard-split-panel--main" min="420">
        <div class="dashboard-main-grid">
          <el-splitter class="dashboard-main-splitter" layout="vertical" lazy>
            <el-splitter-panel
              class="dashboard-split-panel dashboard-split-panel--topology"
              :size="renderedCenterTopSize"
              min="250"
              @update:size="handleCenterTopResize"
            >
              <DashboardTopologyPanel />
            </el-splitter-panel>

            <el-splitter-panel class="dashboard-split-panel dashboard-split-panel--table" min="220">
              <DashboardDataTablePanel
                :table-rows="tableRows"
                :selected-row-key="selectedDatasetName"
                :empty-text="loadingList || loadingDetail ? 'Loading...' : EMPTY_TEXT"
                @row-select="handleTableRowSelect"
              />
            </el-splitter-panel>
          </el-splitter>
        </div>
      </el-splitter-panel>

      <el-splitter-panel
        class="dashboard-split-panel dashboard-split-panel--record"
        :size="renderedRightPanelSize"
        :min="recordCollapsed ? 28 : 360"
        :resizable="!recordCollapsed"
        @update:size="handleRightPanelResize"
      >
        <aside class="dashboard-record-shell" :class="{ 'is-collapsed': recordCollapsed }">
          <DashboardCurrentRecord
            :dataset-detail="datasetDetail"
            :selected-dataset="selectedDataset"
            :selected-rank-count="selectedRankCount"
            :using-fallback="usingFallback"
            :summary-rows="summaryRows"
            :overview-stats="overviewStats"
            :file-stats="fileStats"
            :empty-text="loadingList || loadingDetail ? 'Loading...' : EMPTY_TEXT"
            @navigate-page="handleNavigatePage"
          />

          <button
            v-if="!recordCollapsed"
            type="button"
            class="dashboard-record-toggle dashboard-record-toggle--collapse"
            aria-label="Collapse detail panel"
            @click="toggleRecordPanel"
          >
            <img :src="sidebarHandle" alt="" class="dashboard-record-toggle__image" />
          </button>

          <button
            v-else
            type="button"
            class="dashboard-record-toggle dashboard-record-toggle--expand"
            aria-label="Expand detail panel"
            @click="toggleRecordPanel"
          >
            <img :src="sidebarHandleExpand" alt="" class="dashboard-record-toggle__image" />
          </button>
        </aside>
      </el-splitter-panel>
    </el-splitter>
  </section>
</template>

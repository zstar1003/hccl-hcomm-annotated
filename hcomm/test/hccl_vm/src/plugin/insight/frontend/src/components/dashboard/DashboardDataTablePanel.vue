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

const props = defineProps({
  tableRows: {
    type: Array,
    required: true,
  },
  selectedRowKey: {
    type: String,
    default: '',
  },
  emptyText: {
    type: String,
    default: 'No data',
  },
})

const emit = defineEmits(['row-select'])

const datasetKeyword = ref('')
const datasetSortOrder = ref('default')
const datasetSearchVisible = ref(false)

const filteredRows = computed(() => {
  const keyword = datasetKeyword.value.trim().toLowerCase()
  const rows = keyword
    ? props.tableRows.filter((row) =>
        String(row.datasetName ?? '').toLowerCase().includes(keyword),
      )
    : props.tableRows.slice()

  if (datasetSortOrder.value !== 'default') {
    rows.sort((left, right) => {
      const leftName = String(left.datasetName ?? '').toLowerCase()
      const rightName = String(right.datasetName ?? '').toLowerCase()
      const result = leftName.localeCompare(rightName, 'en')
      return datasetSortOrder.value === 'asc' ? result : -result
    })
  }

  return rows.map((row, index) => ({
    ...row,
    index: index + 1,
  }))
})

function toggleDatasetSort() {
  if (datasetSortOrder.value === 'default') {
    datasetSortOrder.value = 'asc'
    return
  }

  if (datasetSortOrder.value === 'asc') {
    datasetSortOrder.value = 'desc'
    return
  }

  datasetSortOrder.value = 'default'
}

function toggleDatasetSearch() {
  datasetSearchVisible.value = !datasetSearchVisible.value
  if (!datasetSearchVisible.value) {
    datasetKeyword.value = ''
  }
}

function handleRowClick(row) {
  emit('row-select', row)
}
</script>

<template>
  <section class="dashboard-panel dashboard-table-panel">
    <el-table
      :data="filteredRows"
      height="100%"
      class="dashboard-dark-table"
      :empty-text="emptyText"
      row-key="datasetKey"
      highlight-current-row
      :current-row-key="selectedRowKey"
      @row-click="handleRowClick"
    >
      <el-table-column prop="index" label="No." width="72" align="center" header-align="center" />
      <el-table-column prop="datasetName" min-width="260">
        <template #header>
          <div class="dashboard-table-header-cell">
            <span>Dataset Name</span>

            <div class="dashboard-table-header-actions">
              <button
                type="button"
                class="dashboard-table-icon-button"
                aria-label="Filter dataset name"
                @click="toggleDatasetSearch"
              >
                <svg viewBox="0 0 16 16" aria-hidden="true" class="dashboard-table-icon-svg">
                  <path
                    d="M2.5 3.25H13.5L9.35 7.9V11.8L6.65 13.3V7.9L2.5 3.25Z"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="1.3"
                    stroke-linejoin="round"
                  />
                </svg>
              </button>

              <transition name="dashboard-inline-search">
                <el-input
                  v-if="datasetSearchVisible"
                  v-model="datasetKeyword"
                  class="dashboard-inline-search-input"
                  placeholder="Search"
                  clearable
                />
              </transition>

              <button
                type="button"
                class="dashboard-table-icon-button"
                :aria-label="
                  datasetSortOrder === 'default'
                    ? 'Sort ascending'
                    : datasetSortOrder === 'asc'
                      ? 'Sort descending'
                      : 'Reset sort'
                "
                @click="toggleDatasetSort"
              >
                <svg
                  v-if="datasetSortOrder === 'default'"
                  viewBox="0 0 16 16"
                  aria-hidden="true"
                  class="dashboard-table-icon-svg"
                >
                  <path
                    d="M5 3.25V12.75M5 3.25L3.2 5.05M5 3.25L6.8 5.05M11 12.75V3.25M11 12.75L9.2 10.95M11 12.75L12.8 10.95"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="1.3"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  />
                </svg>
                <svg
                  v-else-if="datasetSortOrder === 'asc'"
                  viewBox="0 0 16 16"
                  aria-hidden="true"
                  class="dashboard-table-icon-svg"
                >
                  <path
                    d="M5 12.75V3.25M5 3.25L3.2 5.05M5 3.25L6.8 5.05"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="1.3"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  />
                  <path
                    d="M9 5.5H12.75M9 8H11.4M9 10.5H10.1"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="1.3"
                    stroke-linecap="round"
                  />
                </svg>
                <svg
                  v-else
                  viewBox="0 0 16 16"
                  aria-hidden="true"
                  class="dashboard-table-icon-svg"
                >
                  <path
                    d="M5 3.25V12.75M5 12.75L3.2 10.95M5 12.75L6.8 10.95"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="1.3"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                  />
                  <path
                    d="M9 5.5H10.1M9 8H11.4M9 10.5H12.75"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="1.3"
                    stroke-linecap="round"
                  />
                </svg>
              </button>
            </div>
          </div>
        </template>
      </el-table-column>
      <el-table-column prop="operator" label="Operator" min-width="180" />
      <el-table-column prop="rank" label="Rank" min-width="140" />
      <el-table-column prop="type" label="Type" min-width="140" />
    </el-table>
  </section>
</template>

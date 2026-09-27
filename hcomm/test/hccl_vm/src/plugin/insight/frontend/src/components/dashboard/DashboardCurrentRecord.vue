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
import {
  Cpu,
  Files,
  Postcard,
  Share,
  TrendCharts,
  WarningFilled,
} from '@element-plus/icons-vue'

defineProps({
  datasetDetail: {
    type: Object,
    default: null,
  },
  selectedDataset: {
    type: Object,
    default: null,
  },
  selectedRankCount: {
    type: Number,
    required: true,
  },
  usingFallback: {
    type: Boolean,
    required: true,
  },
  summaryRows: {
    type: Array,
    required: true,
  },
  overviewStats: {
    type: Array,
    required: true,
  },
  fileStats: {
    type: Object,
    required: true,
  },
  emptyText: {
    type: String,
    default: '暂无数据',
  },
})

const emit = defineEmits(['navigate-page'])

const statIcons = {
  warning: WarningFilled,
  files: Files,
  cpu: Cpu,
  share: Share,
  trend: TrendCharts,
}

function navigateTo(pageId) {
  emit('navigate-page', pageId)
}
</script>

<template>
  <aside class="dashboard-panel dashboard-record-panel">
    <template v-if="selectedDataset">
      <div class="dashboard-record-card">
        <div class="dashboard-record-card__header">
          <div class="dashboard-record-card__icon">
            <el-icon><Postcard /></el-icon>
          </div>
          <div>
            <p class="dashboard-eyebrow">当前记录</p>
            <h2>{{ datasetDetail?.dataId ?? selectedDataset?.dataId ?? '--' }}</h2>
          </div>
        </div>

        <div class="dashboard-tag-row">
          <el-tag type="success">{{ selectedDataset?.success === false ? 'failed' : 'success' }}</el-tag>
          <el-tag type="info">算子 ReduceScatter</el-tag>
          <el-tag type="primary">Rank {{ selectedRankCount }}</el-tag>
          <el-tag v-if="usingFallback" type="warning">preview fallback</el-tag>
        </div>

        <div class="dashboard-action-row">
          <el-button type="primary" @click="navigateTo('mem-view')">进入关联视图</el-button>
          <el-button plain @click="navigateTo('analytic')">查看诊断</el-button>
        </div>

        <div class="dashboard-kv-grid">
          <div
            v-for="([label, value], index) in summaryRows"
            :key="`${label}-${index}`"
            class="dashboard-kv-card"
            :class="{ 'is-wide': index === summaryRows.length - 1 && summaryRows.length % 2 === 1 }"
          >
            <span>{{ label }}</span>
            <strong>{{ value }}</strong>
          </div>
        </div>
      </div>

      <section class="dashboard-card-group">
        <h3>计数统计</h3>
        <div class="dashboard-stats-grid">
          <div v-for="item in overviewStats" :key="item.label" class="dashboard-stat-card">
            <div class="dashboard-stat-card__head">
              <el-icon class="dashboard-stat-card__icon">
                <component :is="statIcons[item.icon]" />
              </el-icon>
              <strong>{{ item.value }}</strong>
            </div>
            <div class="dashboard-stat-card__body">
              <span>{{ item.label }}</span>
            </div>
          </div>
        </div>
      </section>

      <section class="dashboard-card-group">
        <h3>文件计数</h3>
        <div class="dashboard-file-grid">
          <div class="dashboard-file-card">
            <strong>{{ fileStats.origin }}</strong>
            <span>原图</span>
          </div>
          <div class="dashboard-file-card">
            <strong>{{ fileStats.translated }}</strong>
            <span>翻译图</span>
          </div>
          <div class="dashboard-file-card">
            <strong>{{ fileStats.rebuilt }}</strong>
            <span>改造图</span>
          </div>
          <div class="dashboard-file-card">
            <strong>{{ fileStats.memory }}</strong>
            <span>内存快照</span>
          </div>
          <div class="dashboard-file-card">
            <strong>{{ fileStats.errors }}</strong>
            <span>错误信息</span>
          </div>
        </div>
      </section>
    </template>

    <div v-else class="dashboard-empty-wrap">
      <el-empty :description="emptyText" />
    </div>
  </aside>
</template>

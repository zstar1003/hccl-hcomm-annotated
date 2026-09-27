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
import { ref, watch, nextTick, computed } from 'vue'
import {
  Collection,
  CollectionTag,
  Cpu,
  Finished,
  FolderOpened,
  Memo,
  Minus,
  Odometer,
  Plus,
  Switch,
} from '@element-plus/icons-vue'
import {
  formatCountWithUnit,
  formatDataType,
  formatOperatorName,
  formatReduceType,
} from '../../composables/useInsightDatasetState'

const props = defineProps({
  selectedRankCount: {
    type: Number,
    required: true,
  },
  datasetDetail: {
    type: Object,
    default: null,
  },
  selectedDataset: {
    type: Object,
    default: null,
  },
  treeData: {
    type: Array,
    required: true,
  },
  checkedKeys: {
    type: Array,
    required: true,
  },
  emptyText: {
    type: String,
    default: '暂无数据',
  },
})

const emit = defineEmits(['tree-check'])

const treeRef = ref(null)
const activeGroups = ref(['summary', 'rank'])

const hasTreeData = computed(() => props.treeData.length > 0)
const summaryOpParam = computed(() => props.datasetDetail?.opParam ?? props.selectedDataset?.opParam ?? {})

const metricCards = computed(() => [
  { label: '算子', value: formatOperatorName(summaryOpParam.value?.cmd_type), icon: Cpu },
  { label: 'REDUCE OP', value: formatReduceType(summaryOpParam.value?.reduce_type), icon: Switch },
  { label: '数据类型', value: formatDataType(summaryOpParam.value?.data_type), icon: CollectionTag },
  { label: '数据量', value: formatCountWithUnit(summaryOpParam.value?.data_count), icon: Odometer },
])

watch(
  () => props.treeData,
  async () => {
    await nextTick()
    expandAll()
    syncCheckedKeys()
  },
  { deep: true },
)

watch(
  () => props.checkedKeys,
  async () => {
    await nextTick()
    syncCheckedKeys()
  },
  { deep: true },
)

function handleTreeCheck(_node, payload) {
  emit('tree-check', payload.checkedKeys)
}

function getAllNodeKeys(nodes, keys = []) {
  nodes.forEach((node) => {
    keys.push(node.id)
    if (node.children?.length) {
      getAllNodeKeys(node.children, keys)
    }
  })
  return keys
}

function forEachTreeNode(visitor) {
  const storeNodes = treeRef.value?.store?.nodesMap ?? {}
  Object.values(storeNodes).forEach((node) => {
    if (node.level > 0) {
      visitor(node)
    }
  })
}

function expandAll() {
  forEachTreeNode((node) => {
    node.expanded = true
  })
}

function collapseAll() {
  forEachTreeNode((node) => {
    if (node.childNodes?.length) {
      node.expanded = false
    }
  })
}

function selectAllRanks() {
  const rankKeys = getAllNodeKeys(props.treeData).filter((key) => String(key).startsWith('rank-'))
  treeRef.value?.setCheckedKeys(rankKeys)
  emit('tree-check', rankKeys)
}

function syncCheckedKeys() {
  if (!treeRef.value) return
  treeRef.value.setCheckedKeys(props.checkedKeys)
}
</script>

<template>
  <aside class="dashboard-sidebar">
    <slot name="top" />

    <template v-if="selectedDataset">
      <el-collapse v-model="activeGroups" class="dashboard-sidebar-collapse">
        <el-collapse-item name="summary" class="dashboard-subpanel dashboard-summary-panel">
          <template #title>
            <div class="dashboard-panel-toggle">
              <span class="dashboard-panel-toggle__title">
                <el-icon><Memo /></el-icon>
                <span>算子摘要</span>
              </span>
            </div>
          </template>

          <div class="dashboard-metric-grid">
            <div v-for="card in metricCards" :key="card.label" class="dashboard-info-card">
              <div class="dashboard-info-card__head">
                <el-icon><component :is="card.icon" /></el-icon>
                <span>{{ card.label }}</span>
              </div>
              <strong>{{ card.value }}</strong>
            </div>
          </div>
        </el-collapse-item>

        <el-collapse-item name="rank" class="dashboard-subpanel dashboard-rank-panel">
          <template #title>
            <div class="dashboard-panel-toggle">
              <span class="dashboard-panel-toggle__title">
                <el-icon><Collection /></el-icon>
                <span>Rank</span>
              </span>
              <span class="dashboard-panel-toggle__meta">
                <el-tag type="success">{{ selectedRankCount }} selected</el-tag>
              </span>
            </div>
          </template>

          <template v-if="hasTreeData">
            <div class="dashboard-tool-row">
              <button type="button" class="dashboard-tool-button" title="全部展开" @click.stop="expandAll">
                <el-icon><Plus /></el-icon>
              </button>
              <button type="button" class="dashboard-tool-button" title="全部折叠" @click.stop="collapseAll">
                <el-icon><Minus /></el-icon>
              </button>
              <button type="button" class="dashboard-tool-button" title="全选 Rank" @click.stop="selectAllRanks">
                <el-icon><Finished /></el-icon>
              </button>
              <button type="button" class="dashboard-tool-button" title="文件视图" disabled>
                <el-icon><FolderOpened /></el-icon>
              </button>
            </div>

            <el-tree
              ref="treeRef"
              :data="treeData"
              show-checkbox
              node-key="id"
              default-expand-all
              :default-checked-keys="checkedKeys"
              :props="{ label: 'label', children: 'children' }"
              class="dashboard-tree"
              @check="handleTreeCheck"
            >
              <template #default="{ data }">
                <div class="dashboard-tree-node">
                  <span class="dashboard-tree-node__label">{{ data.label }}</span>
                  <span v-if="data.caption" class="dashboard-tree-node__caption">{{ data.caption }}</span>
                </div>
              </template>
            </el-tree>
          </template>

          <div v-else class="dashboard-empty-wrap dashboard-empty-wrap--compact">
            <el-empty :description="emptyText" />
          </div>
        </el-collapse-item>
      </el-collapse>

      <slot name="after-rank" />
    </template>

    <div v-else class="dashboard-empty-wrap">
      <el-empty :description="emptyText" />
    </div>
  </aside>
</template>

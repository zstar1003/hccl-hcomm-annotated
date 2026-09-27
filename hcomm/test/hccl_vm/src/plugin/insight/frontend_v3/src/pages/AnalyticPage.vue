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
import { computed, onMounted, ref, watch } from 'vue'
import { Connection, Cpu, InfoFilled, Share, WarningFilled } from '@element-plus/icons-vue'
import sidebarHandle from '../assets/sidebar-handle.svg'
import sidebarHandleExpand from '../assets/sidebar-handle-expand.svg'
import DashboardSidebar from '../components/dashboard/DashboardSidebar.vue'
import DashboardTopologyPanel from '../components/dashboard/DashboardTopologyPanel.vue'
import { usePageNavigationState } from '../composables/usePageNavigationState'
import { useInsightDatasetState } from '../composables/useInsightDatasetState'
import { fetchValidationIssues } from '../utils'

const emit = defineEmits(['navigate-page'])

const sidebarCollapsed = ref(false)
const recordCollapsed = ref(false)
const leftPanelSize = ref(376)
const rightPanelSize = ref(420)
const centerTopSize = ref(324)
const loadingIssues = ref(false)
const issueLoadError = ref('')
const issuePayload = ref({ issueCount: 0, issues: [] })
const selectedIssueId = ref(null)

const {
  EMPTY_TEXT,
  datasetDetail,
  selectedDataset,
  selectedDatasetName,
  selectedRankKeys,
  loadingList,
  selectedRankCount,
  treeData,
  ensureDatasetsLoaded,
  setSelectedRankKeys,
} = useInsightDatasetState()

const { queueMemViewNavigation } = usePageNavigationState()

const renderedLeftPanelSize = computed(() => (sidebarCollapsed.value ? 28 : leftPanelSize.value))
const renderedRightPanelSize = computed(() => (recordCollapsed.value ? 28 : rightPanelSize.value))
const issues = computed(() => issuePayload.value?.issues ?? [])
const issueCount = computed(() => issuePayload.value?.issueCount ?? issues.value.length)
const selectedIssue = computed(
  () => issues.value.find((issue) => issue.issueId === selectedIssueId.value) ?? issues.value[0] ?? null,
)

const selectedAnchorRows = computed(() => selectedIssue.value?.anchorRows ?? [])

const selectedTaskNodeRows = computed(() => {
  const taskNode = selectedIssue.value?.taskNode
  if (!taskNode) return []

  return [
    ['Node', taskNode.nodeId || '--'],
    ['Rank', Number.isInteger(taskNode.rankId) ? `Rank ${taskNode.rankId}` : '--'],
    ['Queue', Number.isInteger(taskNode.queueId) ? `Queue ${taskNode.queueId}` : '--'],
    ['Slot', Number.isInteger(taskNode.pos) ? `Task ${taskNode.pos}` : '--'],
    ['Task Type', taskNode.taskType || '--'],
    ['位置', taskNode.posInfo || '--'],
    ['父节点数', String(taskNode.parents.length)],
    ['子节点数', String(taskNode.children.length)],
  ].filter(([, value]) => value !== '--')
})

watch(
  issues,
  (nextIssues) => {
    if (!nextIssues.length) {
      selectedIssueId.value = null
      return
    }

    const currentIssueExists = nextIssues.some((issue) => issue.issueId === selectedIssueId.value)
    if (!currentIssueExists) {
      selectedIssueId.value = nextIssues[0].issueId
    }
  },
  { immediate: true },
)

watch(
  selectedDatasetName,
  async (datasetName) => {
    if (!datasetName) {
      issuePayload.value = { issueCount: 0, issues: [] }
      selectedIssueId.value = null
      issueLoadError.value = ''
      return
    }

    loadingIssues.value = true
    issueLoadError.value = ''
    issuePayload.value = await fetchValidationIssues(datasetName, {
      onError: (error) => {
        issueLoadError.value = formatIssueLoadError(error)
      },
    })
    loadingIssues.value = false
  },
  { immediate: true },
)

function handleTreeCheck(checkedKeys) {
  setSelectedRankKeys(checkedKeys)
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
    centerTopSize.value = size
  }
}

function toggleSidebar() {
  sidebarCollapsed.value = !sidebarCollapsed.value
}

function toggleRecordPanel() {
  recordCollapsed.value = !recordCollapsed.value
}

function selectIssue(issueId) {
  selectedIssueId.value = issueId
}

function formatIssueLoadError(error) {
  const detail = error instanceof Error ? error.message : String(error ?? '').trim()
  return detail ? `无法读取 issues.msgpack：${detail}` : '无法读取 issues.msgpack。'
}

function resolveNavigationRankIds(issue) {
  if (issue?.relatedRankIds?.length) {
    return issue.relatedRankIds
  }

  return selectedRankKeys.value
    .map((key) => {
      const match = String(key).match(/^rank-(\d+)$/)
      return match ? Number(match[1]) : null
    })
    .filter((rankId) => Number.isInteger(rankId))
}

function resolveIssueNavigationStep(issue) {
  const primaryStep = issue?.primaryRelatedStep
  if (Number.isInteger(primaryStep?.snapshotStep)) return primaryStep.snapshotStep

  const taskNode = issue?.taskNode
  if (Number.isInteger(taskNode?.globalStep)) return taskNode.globalStep
  if (Number.isInteger(taskNode?.localStep)) return taskNode.localStep
  return null
}

function resolveIssueNodeTarget(issue) {
  const relatedNode = issue?.primaryRelatedNode ?? issue?.taskNode
  if (relatedNode?.nodeId || (Number.isInteger(relatedNode?.rankId) && Number.isInteger(relatedNode?.queueId))) {
    return {
      nodeId: relatedNode?.nodeId ?? '',
      rankId: Number.isInteger(relatedNode?.rankId) ? relatedNode.rankId : issue?.primaryRankId ?? null,
      queueId: Number.isInteger(relatedNode?.queueId) ? relatedNode.queueId : issue?.queueId ?? null,
      slotIndex: Number.isInteger(relatedNode?.pos) ? relatedNode.pos : null,
      taskType: relatedNode?.taskType ?? issue?.taskType ?? '',
    }
  }

  if (!issue?.dagTarget) {
    return {
      nodeId: '',
      rankId: null,
      queueId: null,
      slotIndex: null,
      taskType: '',
    }
  }

  return {
    nodeId: issue.dagTarget.nodeId ?? '',
    rankId:
      Number.isInteger(issue?.dagTarget?.queueId) || Number.isInteger(issue?.dagTarget?.slotIndex) || issue?.dagTarget?.nodeId
        ? issue.dagTarget.rankId ?? issue?.primaryRankId ?? null
        : null,
    queueId: issue.dagTarget.queueId ?? issue?.queueId ?? null,
    slotIndex: issue?.dagTarget?.slotIndex ?? null,
    taskType: issue?.dagTarget?.taskType ?? issue?.taskType ?? '',
  }
}

function navigateToMemView(issue) {
  if (!selectedDatasetName.value) return

  const target = resolveIssueNodeTarget(issue)
  queueMemViewNavigation({
    source: 'analytic',
    targetView: 'mem',
    datasetName: selectedDatasetName.value,
    rankIds: resolveNavigationRankIds(issue),
    stageName: issue?.dagTarget?.stageName ?? issue?.memTarget?.stageName ?? '',
    step: resolveIssueNavigationStep(issue),
    nodeId: target.nodeId,
    rankId: target.rankId,
    queueId: target.queueId,
    slotIndex: target.slotIndex,
    taskType: target.taskType,
  })
  emit('navigate-page', 'mem-view')
}

function navigateToDagView(issue) {
  if (!selectedDatasetName.value) return

  const target = resolveIssueNodeTarget(issue)
  queueMemViewNavigation({
    source: 'analytic',
    targetView: 'dag',
    datasetName: selectedDatasetName.value,
    rankIds: resolveNavigationRankIds(issue),
    stageName: issue?.dagTarget?.stageName ?? issue?.memTarget?.stageName ?? '',
    nodeId: target.nodeId,
    rankId: target.rankId,
    queueId: target.queueId,
    slotIndex: target.slotIndex,
    taskType: target.taskType,
  })
  emit('navigate-page', 'mem-view')
}

onMounted(() => {
  ensureDatasetsLoaded()
})
</script>

<template>
  <section class="dashboard-layout analytic-layout">
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
            :dataset-detail="datasetDetail"
            :selected-dataset="selectedDataset"
            :tree-data="treeData"
            :checked-keys="selectedRankKeys"
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
              :size="centerTopSize"
              min="220"
              @update:size="handleCenterTopResize"
            >
              <DashboardTopologyPanel
                :loading-detail="loadingIssues"
                :has-data="Boolean(selectedDataset)"
                :empty-text="EMPTY_TEXT"
                eyebrow="可视化"
                title="相关 Rank DAG"
                ready-text="暂未实现"
                placeholder-title="相关 Rank DAG 占位"
                placeholder-copy="这里暂时复用 dashboard 同款占位区域，后续再接入报错前后相关 Rank DAG 的真实可视化。"
              />
            </el-splitter-panel>

            <el-splitter-panel class="dashboard-split-panel dashboard-split-panel--table" min="220">
              <section class="dashboard-panel analytic-list-panel">
                <div class="dashboard-section-title">
                  <div>
                    <p class="dashboard-eyebrow">Issue</p>
                    <h2>报错列表</h2>
                  </div>
                  <span class="dashboard-status-pill">{{ issueCount }} 条</span>
                </div>

                <el-alert
                  v-if="issueLoadError"
                  class="analytic-issue-load-alert"
                  type="error"
                  :title="issueLoadError"
                  show-icon
                  :closable="false"
                />

                <div v-if="loadingIssues" class="dashboard-empty-wrap analytic-list-empty">
                  <el-empty description="正在读取 issues.msgpack..." />
                </div>

                <div v-else-if="issues.length" class="analytic-issue-list">
                  <button
                    v-for="issue in issues"
                    :key="issue.issueId"
                    type="button"
                    class="analytic-issue-row"
                    :class="{ 'is-active': selectedIssue?.issueId === issue.issueId }"
                    @click="selectIssue(issue.issueId)"
                  >
                    <span class="analytic-issue-row__icon" :class="`is-${issue.severityType}`">
                      <el-icon><WarningFilled /></el-icon>
                    </span>
                    <div class="analytic-issue-row__content">
                      <div class="analytic-issue-row__topline">
                        <strong>{{ issue.title }}</strong>
                        <el-tag size="small" :type="issue.severityType">{{ issue.severityLabel }}</el-tag>
                      </div>
                      <div v-if="issue.listFieldRows.primaryRow.length" class="analytic-issue-row__field-grid">
                        <div
                          v-for="field in issue.listFieldRows.primaryRow"
                          :key="`primary-${issue.issueId}-${field.label}`"
                          class="analytic-issue-row__field"
                        >
                          <span>{{ field.label }}</span>
                          <strong>{{ field.value }}</strong>
                        </div>
                      </div>
                      <div v-if="issue.listFieldRows.secondaryRow.length" class="analytic-issue-row__field-grid is-secondary">
                        <div
                          v-for="field in issue.listFieldRows.secondaryRow"
                          :key="`secondary-${issue.issueId}-${field.label}`"
                          class="analytic-issue-row__field"
                        >
                          <span>{{ field.label }}</span>
                          <strong>{{ field.value }}</strong>
                        </div>
                      </div>
                    </div>
                    <span class="analytic-issue-row__code">{{ issue.code }}</span>
                  </button>
                </div>

                <div v-else class="dashboard-empty-wrap analytic-list-empty">
                  <el-empty description="当前数据集没有 issue 记录。" />
                </div>
              </section>
            </el-splitter-panel>
          </el-splitter>
        </div>
      </el-splitter-panel>

      <el-splitter-panel
        class="dashboard-split-panel dashboard-split-panel--record"
        :size="renderedRightPanelSize"
        :min="recordCollapsed ? 28 : 340"
        :resizable="!recordCollapsed"
        @update:size="handleRightPanelResize"
      >
        <aside class="dashboard-record-shell" :class="{ 'is-collapsed': recordCollapsed }">
          <section class="dashboard-panel dashboard-record-panel analytic-detail-panel">
            <template v-if="selectedIssue">
              <div class="dashboard-section-title">
                <div>
                  <p class="dashboard-eyebrow">Issue Detail</p>
                  <h2>报错详情</h2>
                </div>
                <span class="dashboard-status-pill">{{ selectedIssue.stageLabel }}</span>
              </div>

              <div class="dashboard-record-card analytic-detail-hero">
                <div class="dashboard-record-card__header">
                  <div class="dashboard-record-card__icon">
                    <el-icon><Connection /></el-icon>
                  </div>
                  <div>
                    <p class="dashboard-eyebrow">当前报错</p>
                    <h2>{{ selectedIssue.title }}</h2>
                  </div>
                </div>

                <div class="dashboard-tag-row">
                  <el-tag :type="selectedIssue.severityType">{{ selectedIssue.severityLabel }}</el-tag>
                  <el-tag type="primary">{{ selectedIssue.code }}</el-tag>
                  <el-tag v-if="selectedIssue.primaryRankId !== null" type="info">
                    Rank {{ selectedIssue.primaryRankId }}
                  </el-tag>
                  <el-tag v-if="selectedIssue.nodeDisplayId !== '--'" type="info">
                    {{ selectedIssue.nodeDisplayId }}
                  </el-tag>
                </div>

                <div class="dashboard-action-row analytic-detail-actions">
                  <el-button type="primary" @click="navigateToMemView(selectedIssue)">查看 memView</el-button>
                  <el-button plain type="primary" @click="navigateToDagView(selectedIssue)">跳转</el-button>
                  <el-button plain type="primary" @click="navigateToDagView(selectedIssue)">查看 dagView</el-button>
                </div>

                <div class="dashboard-kv-grid">
                  <div
                    v-for="([label, value], index) in selectedIssue.overviewRows"
                    :key="`${label}-${index}`"
                    class="dashboard-kv-card"
                    :class="{ 'is-wide': index === selectedIssue.overviewRows.length - 1 && selectedIssue.overviewRows.length % 2 === 1 }"
                  >
                    <span>{{ label }}</span>
                    <strong>{{ value }}</strong>
                  </div>
                </div>
              </div>

              <section v-if="selectedAnchorRows.length" class="dashboard-card-group">
                <div class="dashboard-subpanel__header">
                  <div class="dashboard-subpanel__title">
                    <el-icon><Cpu /></el-icon>
                    <span>关联节点</span>
                  </div>
                </div>
                <div class="dashboard-kv-grid">
                  <div
                    v-for="([label, value], index) in selectedAnchorRows"
                    :key="`${label}-${index}`"
                    class="dashboard-kv-card"
                    :class="{ 'is-wide': index === selectedAnchorRows.length - 1 && selectedAnchorRows.length % 2 === 1 }"
                  >
                    <span>{{ label }}</span>
                    <strong>{{ value }}</strong>
                  </div>
                </div>
              </section>

              <section v-if="selectedIssue.sliceCards.length" class="dashboard-card-group">
                <div class="dashboard-subpanel__header">
                  <div class="dashboard-subpanel__title">
                    <el-icon><Share /></el-icon>
                    <span>Slice 详情</span>
                  </div>
                </div>
                <div class="analytic-slice-grid">
                  <article v-for="slice in selectedIssue.sliceCards" :key="slice.label" class="analytic-slice-card">
                    <h3>{{ slice.label }}</h3>
                    <div class="dashboard-kv-grid">
                      <div
                        v-for="([label, value], index) in slice.fields"
                        :key="`${slice.label}-${label}-${index}`"
                        class="dashboard-kv-card"
                      >
                        <span>{{ label }}</span>
                        <strong>{{ value }}</strong>
                      </div>
                    </div>
                  </article>
                </div>
              </section>

              <section v-if="selectedIssue.detailRows.length" class="dashboard-card-group">
                <div class="dashboard-subpanel__header">
                  <div class="dashboard-subpanel__title">
                    <el-icon><InfoFilled /></el-icon>
                    <span>补充信息</span>
                  </div>
                </div>
                <div class="dashboard-kv-grid">
                  <div
                    v-for="([label, value], index) in selectedIssue.detailRows"
                    :key="`${label}-${index}`"
                    class="dashboard-kv-card"
                    :class="{ 'is-wide': index === selectedIssue.detailRows.length - 1 && selectedIssue.detailRows.length % 2 === 1 }"
                  >
                    <span>{{ label }}</span>
                    <strong>{{ value }}</strong>
                  </div>
                </div>
              </section>

              <section class="dashboard-card-group">
                <div class="dashboard-subpanel__header">
                  <div class="dashboard-subpanel__title">
                    <el-icon><InfoFilled /></el-icon>
                    <span>原始 detail</span>
                  </div>
                </div>
                <pre class="analytic-detail-json">{{ JSON.stringify(selectedIssue.rawDetail, null, 2) }}</pre>
              </section>
            </template>

            <div v-else class="dashboard-empty-wrap">
              <el-empty :description="loadingIssues ? '正在读取 issue 详情...' : '请选择一条报错记录。'" />
            </div>
          </section>

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

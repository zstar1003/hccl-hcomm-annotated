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
import { computed, nextTick, onMounted, onBeforeUnmount, ref, watch } from 'vue'
import {
  CaretLeft,
  CaretRight,
  DArrowLeft,
  DArrowRight,
  Search,
  VideoPause,
  VideoPlay,
  WarningFilled,
} from '@element-plus/icons-vue'
import sidebarHandle from '../assets/sidebar-handle.svg'
import sidebarHandleExpand from '../assets/sidebar-handle-expand.svg'
import { useInsightDatasetState } from '../composables/useInsightDatasetState'
import { usePageNavigationState } from '../composables/usePageNavigationState'
import { useMemViewTimeline } from '../composables/useMemViewTimeline'
import { taskOpsAtStep } from '../utils/memviewIndex.js'
import { fetchValidationIssues } from '../utils/analyticIssues.js'
import {
  DEFAULT_MEMVIEW_STAGE,
  normalizeStreamCollection,
} from '../utils/memviewDag.js'
import DashboardSidebar from '../components/dashboard/DashboardSidebar.vue'
import MemViewDagPanel from '../components/memview/MemViewDagPanel.vue'
import MemViewInspectorPanel from '../components/memview/MemViewInspectorPanel.vue'
import MemViewOverviewPanel from '../components/memview/MemViewOverviewPanel.vue'

const CENTER_TOP_MIN_SIZE = 250
const CENTER_TOP_MAX_SIZE = 620
const CENTER_TOP_DEFAULT_RATIO = 0.6
const CENTER_BOTTOM_MIN_SIZE = 220
const CENTER_COLLAPSED_SIZE = 36
const CENTER_COLLAPSE_THRESHOLD = 72
const SEARCH_FIELD_OPTIONS = [
  { label: 'taskId', value: 'taskId' },
  { label: 'taskType', value: 'taskType' },
  { label: 'notifyId', value: 'notifyId' },
]

const sidebarCollapsed = ref(false)
const recordCollapsed = ref(false)
const leftPanelSize = ref(376)
const rightPanelSize = ref(420)
const centerPaneState = ref('normal')
const centerAvailableHeight = ref(0)
const centerTopSize = ref(360)
const currentGraphNodeIds = ref([])
const currentGraphStage = ref(DEFAULT_MEMVIEW_STAGE)
const pendingStageNavigation = ref(null)
const pendingExternalNavigation = ref(null)
const issueHighlightedNodeIds = ref(null)
const datasetIssuePayload = ref({ issueCount: 0, issues: [] })
const selectedIssueId = ref(null)
const inspectorRef = ref(null)
const centerSplitRef = ref(null)
const layoutRefreshTimers = []
const playbackPlaying = ref(false)
const issuePanelSections = ref(['issues'])
const searchPanelSections = ref(['search'])
const searchField = ref('taskId')
const searchQuery = ref('')
let centerResizeObserver = null
let playbackTimer = 0
let issueLoadToken = 0

const hasMemoryData = computed(() => (datasetDetail.value?.memory?.length ?? 0) > 0)

const overviewEmptyText = computed(() => {
  if (loadingDetail.value) return 'Loading data...'
  if (selectedDataset.value && !hasMemoryData.value) {
    return '当前数据集只有 DAG 数据，内存时间线暂不可用。'
  }
  return EMPTY_TEXT
})

const dagEmptyText = computed(() => {
  if (loadingDetail.value) return 'Loading data...'
  if (selectedDataset.value && !datasetDetail.value?.graph?.length) {
    return '当前数据集没有 DAG 数据。'
  }
  return EMPTY_TEXT
})

const {
  EMPTY_TEXT,
  datasetDetail,
  selectedDataset,
  selectedDatasetName,
  selectedRankKeys,
  loadingList,
  loadingDetail,
  hasData,
  memorySnapshotCount,
  selectedRankCount,
  treeData,
  ensureDatasetsLoaded,
  selectDataset,
  setSelectedRankKeys,
} = useInsightDatasetState()

const { pendingMemViewNavigation, clearMemViewNavigation } = usePageNavigationState()

const {
  clampedCurrentStep,
  totalSteps,
  selectedNode,
  subgraphNode,
  indexes,
  nodeRegistry,
  activeStepBufferOps,
  activeStepTaskOps,
  activeStepLayoutBuffers,
  selectedNodeBufferOps,
  setTotalSteps,
  registerNodes,
  clearNodeRegistry,
  focusStep,
  focusNode,
  enterSubgraph,
  exitSubgraph,
  reset: resetTimeline,
} = useMemViewTimeline()

const renderedLeftPanelSize = computed(() => (sidebarCollapsed.value ? 28 : leftPanelSize.value))
const renderedRightPanelSize = computed(() => (recordCollapsed.value ? 28 : rightPanelSize.value))
const topPaneCollapsed = computed(() => centerPaneState.value === 'top-collapsed')
const bottomPaneCollapsed = computed(() => centerPaneState.value === 'bottom-collapsed')
const issues = computed(() => datasetIssuePayload.value?.issues ?? [])
const issueCount = computed(() => datasetIssuePayload.value?.issueCount ?? issues.value.length)
const selectedIssue = computed(
  () => issues.value.find((issue) => issue.issueId === selectedIssueId.value) ?? issues.value[0] ?? null,
)
const renderedCenterTopSize = computed(() => {
  if (topPaneCollapsed.value) return CENTER_COLLAPSED_SIZE
  if (bottomPaneCollapsed.value) {
    return Math.max(CENTER_COLLAPSED_SIZE, resolveCenterAvailableHeight() - CENTER_COLLAPSED_SIZE)
  }
  return clampCenterTopSize(centerTopSize.value)
})
const displayStepItems = computed(() => {
  const resolvedTotalSteps = Math.max(0, totalSteps.value || 0)
  if (resolvedTotalSteps <= 0) return []

  const stepByNodeId = indexes.value?.stepByNodeId
  if (!(stepByNodeId instanceof Map) || !currentGraphNodeIds.value.length) {
    return Array.from({ length: resolvedTotalSteps }, (_, step) => ({ type: 'step', step }))
  }

  const visibleSteps = new Set()
  if (subgraphNode.value) {
    visibleSteps.add(clampedCurrentStep.value)
  } else {
    visibleSteps.add(0)
    visibleSteps.add(resolvedTotalSteps - 1)
  }

  for (const nodeId of currentGraphNodeIds.value) {
    if (!nodeId) continue
    const step = stepByNodeId.get(nodeId)
    if (Number.isInteger(step) && step >= 0 && step < resolvedTotalSteps) {
      visibleSteps.add(step)
    }
  }

  const sortedSteps = [...visibleSteps].sort((left, right) => left - right)
  const items = []
  for (let index = 0; index < sortedSteps.length; index += 1) {
    const step = sortedSteps[index]
    if (index > 0) {
      const previous = sortedSteps[index - 1]
      if (step - previous > 1) {
        items.push({
          type: 'ellipsis',
          fromStep: previous,
          toStep: step,
          hiddenCount: step - previous - 1,
        })
      }
    }
    items.push({ type: 'step', step })
  }

  return items
})
const displayVisibleSteps = computed(() =>
  displayStepItems.value
    .filter((item) => item?.type === 'step' && Number.isInteger(item.step))
    .map((item) => item.step),
)
const playbackVisibleSteps = computed(() => {
  if (displayVisibleSteps.value.length) return displayVisibleSteps.value
  return Array.from({ length: Math.max(0, totalSteps.value) }, (_, step) => step)
})
const playbackVisibleIndex = computed({
  get: () => {
    const currentIndex = playbackVisibleSteps.value.indexOf(clampedCurrentStep.value)
    if (currentIndex >= 0) return currentIndex
    let fallbackIndex = 0
    for (let index = 0; index < playbackVisibleSteps.value.length; index += 1) {
      if (playbackVisibleSteps.value[index] <= clampedCurrentStep.value) fallbackIndex = index
    }
    return fallbackIndex
  },
  set: (value) => {
    const upper = Math.max(0, playbackVisibleSteps.value.length - 1)
    const nextIndex = Math.min(Math.max(Number(value) || 0, 0), upper)
    const nextStep = playbackVisibleSteps.value[nextIndex]
    if (Number.isInteger(nextStep)) {
      focusStep(nextStep, { smoothScroll: false })
    }
  },
})
const playbackStep = computed(() => {
  const current = playbackVisibleSteps.value[playbackVisibleIndex.value]
  if (Number.isInteger(current)) return current
  return clampedCurrentStep.value
})

const availableGraphStages = computed(() =>
  Array.isArray(datasetDetail.value?.graph)
    ? datasetDetail.value.graph
        .map((group) => (typeof group?.name === 'string' ? group.name : ''))
        .filter(Boolean)
    : [],
)

watch(
  () => [subgraphNode.value?.id ?? '', displayVisibleSteps.value.join(','), clampedCurrentStep.value],
  ([subgraphId, visibleStepsKey, currentStep]) => {
    if (subgraphId || !visibleStepsKey) return
    if (displayVisibleSteps.value.includes(currentStep)) return

    let nearestStep = displayVisibleSteps.value[0]
    let nearestDistance = Math.abs((nearestStep ?? 0) - currentStep)
    for (const step of displayVisibleSteps.value) {
      const distance = Math.abs(step - currentStep)
      if (distance < nearestDistance) {
        nearestStep = step
        nearestDistance = distance
      }
    }

    if (Number.isInteger(nearestStep) && nearestStep !== currentStep) {
      focusStep(nearestStep, { smoothScroll: false })
    }
  },
)

watch(
  availableGraphStages,
  (stages) => {
    if (!stages.length) {
      currentGraphStage.value = ''
      return
    }

    if (stages.includes(currentGraphStage.value)) return
    currentGraphStage.value = stages.includes(DEFAULT_MEMVIEW_STAGE) ? DEFAULT_MEMVIEW_STAGE : stages[0]
  },
  { immediate: true },
)

watch(
  selectedDatasetName,
  async (datasetName) => {
    const token = ++issueLoadToken
    if (!datasetName) {
      datasetIssuePayload.value = { issueCount: 0, issues: [] }
      return
    }

    const payload = await fetchValidationIssues(datasetName)
    if (token === issueLoadToken) {
      datasetIssuePayload.value = payload
    }
  },
  { immediate: true },
)

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
  () => pendingMemViewNavigation.value?.token ?? 0,
  async () => {
    const request = pendingMemViewNavigation.value
    if (!request) return

    if (!(request.source === 'analytic' && (request.targetView === 'dag' || request.targetView === 'mem'))) {
      issueHighlightedNodeIds.value = null
    }

    if (request.datasetName && request.datasetName !== selectedDatasetName.value) {
      await selectDataset(request.datasetName)
    }

    if (Array.isArray(request.rankIds) && request.rankIds.length) {
      setSelectedRankKeys(request.rankIds.map((rankId) => `rank-${rankId}`))
    }

    if (Number.isInteger(request.step)) {
      focusStep(request.step, { smoothScroll: false })
    }

    pendingExternalNavigation.value = normalizeExternalNavigationRequest(request)

    if (requestTargetsNode(pendingExternalNavigation.value) && bottomPaneCollapsed.value) {
      expandCenterPane('bottom')
    }

    clearMemViewNavigation()

    if (pendingExternalNavigation.value?.stageName && pendingExternalNavigation.value.stageName !== currentGraphStage.value) {
      currentGraphStage.value = pendingExternalNavigation.value.stageName
      return
    }

    resolvePendingExternalNavigation()
  },
  { immediate: true },
)

const selectedNodeStep = computed(() => {
  const node = selectedNode.value
  return resolveStepForNode(node)
})

const selectedRankIdSet = computed(() => {
  const ids = new Set()
  for (const key of selectedRankKeys.value) {
    const match = String(key).match(/^rank-(\d+)$/)
    if (match) {
      ids.add(Number(match[1]))
    }
  }
  return ids
})

const selectedNodeTaskOps = computed(() => {
  const node = selectedNode.value
  const step = selectedNodeStep.value
  if (!node?.id || !indexes.value || !Number.isInteger(step)) return []
  return taskOpsAtStep(indexes.value, step).filter((op) => taskOpMatchesSelectedNode(node, op))
})

const selectedSearchFieldLabel = computed(
  () => SEARCH_FIELD_OPTIONS.find((option) => option.value === searchField.value)?.label ?? searchField.value,
)

const selectedRankSearchNodes = computed(() => {
  if (!selectedRankIdSet.value.size) return []

  const registry = nodeRegistry.value
  if (!(registry instanceof Map) || !registry.size) return []

  const seen = new Map()
  for (const node of registry.values()) {
    if (!node?.id || seen.has(node.id)) continue
    if (Number.isInteger(node.rankId) && !selectedRankIdSet.value.has(node.rankId)) {
      continue
    }
    if (!Number.isInteger(node.rankId)) {
      continue
    }
    seen.set(node.id, node)
  }

  return [...seen.values()].sort((left, right) => {
    const leftRank = Number.isInteger(left.rankId) ? left.rankId : Number.POSITIVE_INFINITY
    const rightRank = Number.isInteger(right.rankId) ? right.rankId : Number.POSITIVE_INFINITY
    if (leftRank !== rightRank) return leftRank - rightRank

    const leftQueue = Number.isInteger(left.queueId) ? left.queueId : Number.POSITIVE_INFINITY
    const rightQueue = Number.isInteger(right.queueId) ? right.queueId : Number.POSITIVE_INFINITY
    if (leftQueue !== rightQueue) return leftQueue - rightQueue

    const leftPos = Number.isInteger(left.pos) ? left.pos : Number.isInteger(left.slotIndex) ? left.slotIndex : Number.POSITIVE_INFINITY
    const rightPos = Number.isInteger(right.pos) ? right.pos : Number.isInteger(right.slotIndex) ? right.slotIndex : Number.POSITIVE_INFINITY
    if (leftPos !== rightPos) return leftPos - rightPos

    return String(left.id).localeCompare(String(right.id))
  })
})

const normalizedSearchQuery = computed(() => searchQuery.value.trim())

function collectSearchNodesFromRegistry(registry) {
  if (!(registry instanceof Map) || !registry.size) return []

  const roots = [...registry.values()].filter((node) => node?.id && !node.isPlaceholder && Number.isInteger(node.rankId))

  const collected = []
  const seenIds = new Set()

  const pushNode = (node, context = {}) => {
    if (!node?.id || seenIds.has(node.id)) return
    if (!Number.isInteger(node.rankId)) return

    seenIds.add(node.id)
    collected.push({
      ...node,
      searchParentId: context.parentId ?? '',
      searchDepth: Number.isInteger(context.depth) ? context.depth : 0,
      searchPath: Array.isArray(context.path) ? context.path : [],
      searchRootId: context.rootId ?? node.id,
    })
  }

  const walkSubgraph = (entries, context) => {
    if (!Array.isArray(entries)) return
    for (const entry of entries) {
      if (!entry || typeof entry !== 'object') continue
      const nestedNodes = normalizeStreamCollection([entry], context.rankId, {
        stageName: currentGraphStage.value || DEFAULT_MEMVIEW_STAGE,
        topLevel: false,
      }).flatMap((stream) => stream.nodes)

      for (const nestedNode of nestedNodes) {
        const nestedContext = {
          rootId: context.rootId,
          parentId: context.parentId,
          depth: context.depth + 1,
          path: [...context.path, entry.node_id ?? entry.id ?? entry.task_id ?? nestedNode.id],
        }
        pushNode(nestedNode, nestedContext)
        const nestedSubGraph = nestedNode?.rawNode?.task?.task_data?.sub_graph
        if (Array.isArray(nestedSubGraph) && nestedSubGraph.length) {
          walkSubgraph(nestedSubGraph.flat(), {
            rankId: nestedNode.rankId,
            rootId: context.rootId,
            parentId: nestedNode.id,
            depth: nestedContext.depth,
            path: nestedContext.path,
          })
        }
      }
    }
  }

  for (const rootNode of roots) {
    pushNode(rootNode, {
      rootId: rootNode.id,
      parentId: '',
      depth: 0,
      path: [rootNode.id],
    })

    const subGraph = rootNode?.rawNode?.task?.task_data?.sub_graph
    if (Array.isArray(subGraph) && subGraph.length) {
      walkSubgraph(subGraph.flat(), {
        rankId: rootNode.rankId,
        rootId: rootNode.id,
        parentId: rootNode.id,
        depth: 0,
        path: [rootNode.id],
      })
    }
  }

  return collected
}

function isExactSearchMatch(node, field, query) {
  if (!node || !query) return false
  const normalizedQuery = String(query).trim().toLowerCase()
  if (!normalizedQuery) return false

  if (field === 'taskType') {
    const value = String(node.taskType ?? node.rawNode?.task?.task_type ?? node.rawNode?.task_type ?? '').trim().toLowerCase()
    return value === normalizedQuery
  }

  if (field === 'notifyId') {
    const value = normalizeSearchFieldValue(
      node.notifyId ??
        node.notify_id ??
        node.taskData?.notifyId ??
        node.taskData?.notify_id ??
        node.rawNode?.notifyId ??
        node.rawNode?.notify_id ??
        node.rawNode?.task?.notifyId ??
        node.rawNode?.task?.notify_id,
    )
    return value === normalizedQuery
  }

  const values = resolveSearchNodeIds(node)
  return values.some((value) => String(value).trim().toLowerCase() === normalizedQuery)
}

function resolveSearchNodeIds(node) {
  const ids = new Set()
  const push = (value) => {
    if (typeof value === 'string' && value.trim()) {
      ids.add(value.trim())
    }
  }

  push(node?.id)
  push(node?.rawNode?.node_id)
  push(node?.rawNode?.id)
  push(node?.rawNode?.task_id)
  return [...ids]
}

function sortSearchNodes(nodes) {
  return [...nodes].sort((left, right) => {
    const leftRank = Number.isInteger(left.rankId) ? left.rankId : Number.POSITIVE_INFINITY
    const rightRank = Number.isInteger(right.rankId) ? right.rankId : Number.POSITIVE_INFINITY
    if (leftRank !== rightRank) return leftRank - rightRank

    const leftQueue = Number.isInteger(left.queueId) ? left.queueId : Number.POSITIVE_INFINITY
    const rightQueue = Number.isInteger(right.queueId) ? right.queueId : Number.POSITIVE_INFINITY
    if (leftQueue !== rightQueue) return leftQueue - rightQueue

    const leftPos = Number.isInteger(left.pos) ? left.pos : Number.isInteger(left.slotIndex) ? left.slotIndex : Number.POSITIVE_INFINITY
    const rightPos = Number.isInteger(right.pos) ? right.pos : Number.isInteger(right.slotIndex) ? right.slotIndex : Number.POSITIVE_INFINITY
    if (leftPos !== rightPos) return leftPos - rightPos

    return String(left.id).localeCompare(String(right.id))
  })
}

const searchResults = computed(() => {
  const query = normalizedSearchQuery.value
  if (!query) return []

  const field = searchField.value
  const allSearchNodes = collectSearchNodesFromRegistry(nodeRegistry.value)
  const matched = allSearchNodes.filter((node) => isExactSearchMatch(node, field, query))

  return sortSearchNodes(matched)
    .map((node) => ({
      node,
      taskId: node.id,
      taskType: String(node.taskType ?? '--').replaceAll('_', ' '),
      rank: Number.isInteger(node.rankId) ? node.rankId : '--',
      queue: Number.isInteger(node.queueId) ? node.queueId : '--',
      pos: Number.isInteger(node.pos) ? node.pos : Number.isInteger(node.slotIndex) ? node.slotIndex : '--',
      formatLoc: formatSearchLocation(node),
      notifyId: resolveSearchFieldDisplayValue(node, 'notifyId'),
    }))
})

const canResolveRelation = computed(() => (nodeId) => Boolean(resolveNodeNavigationTarget(nodeId)))
const resolveRelationMeta = computed(() => (nodeId) => {
  if (typeof nodeId !== 'string' || !nodeId) return null

  const resolvedTarget = resolveNodeNavigationTarget(nodeId)
  const node = resolvedTarget?.node ?? nodeRegistry.value?.get(nodeId) ?? null
  if (!node) return null

  return {
    taskType: typeof node.taskType === 'string' ? node.taskType.replaceAll('_', ' ') : '',
  }
})
const datasetIssueNodeIds = computed(() => {
  const issues = datasetIssuePayload.value?.issues ?? []
  const registry = nodeRegistry.value
  if (!(registry instanceof Map) || !registry.size || !issues.length) return null

  const ids = new Set()
  const nodes = [...new Set([...registry.values()].filter(Boolean))]
  for (const issue of issues) {
    for (const node of nodes) {
      if (issueMatchesNode(issue, node)) {
        ids.add(node.id)
      }
    }
  }

  return ids.size ? ids : null
})

const dagIssueNodeIds = computed(() => {
  const ids = new Set()
  datasetIssueNodeIds.value?.forEach((nodeId) => ids.add(nodeId))
  issueHighlightedNodeIds.value?.forEach?.((nodeId) => ids.add(nodeId))
  return ids.size ? ids : null
})

function handleTreeCheck(checkedKeys) {
  setSelectedRankKeys(checkedKeys)
}

function issueMatchesNode(issue, node) {
  if (!issue || !node?.id) return false

  const targetIds = issueTargetNodeIds(issue)
  const lookupIds = new Set(resolveNodeLookupIds(node))
  if (targetIds.some((nodeId) => lookupIds.has(nodeId) || node.mappingKey === nodeId)) {
    return true
  }

  const target = issue.dagTarget ?? {}
  const detail = issue.rawDetail ?? {}
  const rankId = firstIssueInteger(target.rankId, issue.primaryRankId, detail.rank_id, detail.task_rank, detail.peer_rank)
  const queueId = firstIssueInteger(target.queueId, issue.queueId, detail.queue_id)
  const slotIndex = firstIssueInteger(target.slotIndex, detail.task_pos, detail.pos)
  if (!Number.isInteger(rankId) || !Number.isInteger(queueId) || !Number.isInteger(slotIndex)) return false

  const nodeSlot = Number.isInteger(node.pos) ? node.pos : node.slotIndex
  const nodeQueue = Number.isInteger(node.queueId) ? node.queueId : node.streamId
  return node.rankId === rankId && nodeQueue === queueId && nodeSlot === slotIndex
}

function issueTargetNodeIds(issue) {
  const ids = []
  const push = (value) => {
    if (typeof value === 'string' && value.trim()) ids.push(value.trim())
  }

  push(issue?.dagTarget?.nodeId)
  push(issue?.taskNode?.nodeId)
  push(issue?.rawDetail?.task_id)
  issue?.relatedNodeAnchors?.forEach((anchor) => push(anchor?.nodeId))

  return [...new Set(ids)]
}

function firstIssueInteger(...values) {
  for (const value of values) {
    if (Number.isInteger(value)) return value
    if (typeof value === 'string' && value.trim()) {
      const numeric = Number(value)
      if (Number.isInteger(numeric)) return numeric
    }
  }
  return null
}

function handleLeftPanelResize(size) {
  if (!sidebarCollapsed.value && typeof size === 'number') {
    leftPanelSize.value = size
    notifyDagLayoutChange()
  }
}

function handleRightPanelResize(size) {
  if (!recordCollapsed.value && typeof size === 'number') {
    rightPanelSize.value = size
    notifyDagLayoutChange()
  }
}

function handleCenterTopResize(size) {
  if (typeof size !== 'number') return

  const availableHeight = resolveCenterAvailableHeight()
  if (size <= CENTER_COLLAPSE_THRESHOLD) {
    centerPaneState.value = 'top-collapsed'
  } else if (availableHeight - size <= CENTER_COLLAPSE_THRESHOLD) {
    centerPaneState.value = 'bottom-collapsed'
  } else {
    centerPaneState.value = 'normal'
    centerTopSize.value = clampCenterTopSize(size)
  }
  notifyDagLayoutChange()
}

function toggleSidebar() {
  sidebarCollapsed.value = !sidebarCollapsed.value
  notifyDagLayoutChange()
}

function toggleRecordPanel() {
  recordCollapsed.value = !recordCollapsed.value
  notifyDagLayoutChange()
}

function expandCenterPane(target) {
  centerPaneState.value = 'normal'
  if (target === 'top') {
    centerTopSize.value = clampCenterTopSize(resolveDefaultCenterTopSize())
  } else if (target === 'bottom') {
    centerTopSize.value = clampCenterTopSize(resolveCenterAvailableHeight() * CENTER_TOP_DEFAULT_RATIO)
  }
  notifyDagLayoutChange()
}

function handleNodeSelect(node) {
  focusNode(node, { follow: true, sync: true })
}

async function handleIssueSelect(issue) {
  if (!issue) return

  selectedIssueId.value = issue.issueId

  const target = resolveIssueNodeTarget(issue)
  const step = resolveIssueNavigationStep(issue)
  const rankIds = resolveNavigationRankIds(issue)
  const stageName = issue?.dagTarget?.stageName ?? issue?.memTarget?.stageName ?? ''
  const hasNodeTarget = Boolean(target?.nodeId)
  const hasStepTarget = Number.isInteger(step)

  if (!hasNodeTarget && !hasStepTarget && !rankIds.length) {
    return
  }

  if (rankIds.length) {
    setSelectedRankKeys(rankIds.map((rankId) => `rank-${rankId}`))
  }

  if (stageName && stageName !== currentGraphStage.value) {
    handleStageUpdate(stageName)
  }

  if (hasNodeTarget) {
    pendingExternalNavigation.value = {
      source: 'analytic',
      targetView: 'dag',
      stageName: stageName || currentGraphStage.value,
      lookupId: target.nodeId,
      nodeId: target.nodeId,
      rankId: target.rankId,
      queueId: target.queueId,
      slotIndex: target.slotIndex,
      taskType: target.taskType,
    }

    await nextTick()
    resolvePendingExternalNavigation()
    return
  }

  if (hasStepTarget) {
    focusStep(step, { smoothScroll: false })
  }
}

function handleSearchResultSelect(result) {
  if (!result?.node) return
  handleNavigate({
    stageName: result.node.stageName || currentGraphStage.value || '',
    lookupId: result.node.id,
    containerLookupId:
      result.node.searchParentId && result.node.searchParentId !== result.node.id
        ? result.node.searchParentId
        : result.node.searchRootId && result.node.searchRootId !== result.node.id
          ? result.node.searchRootId
          : '',
  })
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

  const stage = typeof issue?.stage === 'string' ? issue.stage : ''
  const stageMatch = stage.match(/^step_(\d+)_/)
  if (stageMatch) return Number(stageMatch[1])

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

function resolveSearchFieldValue(node, field) {
  if (!node) return ''
  const rawNode = node.rawNode ?? {}
  const taskData = node.taskData ?? rawNode?.task?.task_data ?? rawNode?.task_data ?? {}

  switch (field) {
    case 'taskType':
      return String(node.taskType ?? rawNode?.task?.task_type ?? rawNode?.task_type ?? '').toLowerCase()
    case 'notifyId':
      return normalizeSearchFieldValue(
        node.notifyId ??
          node.notify_id ??
          taskData.notifyId ??
          taskData.notify_id ??
          rawNode.notifyId ??
          rawNode.notify_id ??
          rawNode?.task?.notifyId ??
          rawNode?.task?.notify_id,
      )
    case 'taskId':
    default:
      return String(node.id ?? rawNode?.node_id ?? rawNode?.id ?? '').toLowerCase()
  }
}

function resolveSearchFieldDisplayValue(node, field) {
  if (!node) return '--'
  const rawNode = node.rawNode ?? {}
  const taskData = node.taskData ?? rawNode?.task?.task_data ?? rawNode?.task_data ?? {}

  switch (field) {
    case 'taskType':
      return String(node.taskType ?? rawNode?.task?.task_type ?? rawNode?.task_type ?? '--').replaceAll('_', ' ')
    case 'notifyId':
      return formatSearchFieldDisplayValue(
        node.notifyId ??
          node.notify_id ??
          taskData.notifyId ??
          taskData.notify_id ??
          rawNode.notifyId ??
          rawNode.notify_id ??
          rawNode?.task?.notifyId ??
          rawNode?.task?.notify_id,
      )
    case 'taskId':
    default:
      return formatSearchFieldDisplayValue(node.id ?? rawNode?.node_id ?? rawNode?.id)
  }
}

function formatSearchLocation(node) {
  const rank = Number.isInteger(node?.rankId) ? `rank ${node.rankId}` : 'rank --'
  const queue = Number.isInteger(node?.queueId) ? `queue ${node.queueId}` : 'queue --'
  const posValue = Number.isInteger(node?.pos)
    ? node.pos
    : Number.isInteger(node?.slotIndex)
      ? node.slotIndex
      : null
  const pos = Number.isInteger(posValue) ? `pos ${posValue}` : 'pos --'
  return `${rank} / ${queue} / ${pos}`
}

function normalizeSearchFieldValue(value) {
  if (value === null || value === undefined || value === '') return ''
  return String(value).toLowerCase()
}

function formatSearchFieldDisplayValue(value) {
  if (value === null || value === undefined || value === '') return '--'
  return String(value)
}

function handleStageUpdate(nextStage) {
  if (typeof nextStage !== 'string' || !nextStage || nextStage === currentGraphStage.value) return

  const mappedTarget = selectedNode.value?.nodeMappings?.find((item) => item?.stageName === nextStage) ?? null
  pendingStageNavigation.value = mappedTarget
    ? {
        stageName: nextStage,
        lookupId: mappedTarget.lookupId,
      }
    : null

  if (selectedNode.value) {
    selectedNode.value = null
  }
  if (subgraphNode.value) {
    exitSubgraph()
  }

  currentGraphStage.value = nextStage
}

function handleEnterSubgraph() {
  const node = selectedNode.value
  if (!node?.hasSubgraph) return
  enterSubgraph(node)
}

function handleExitSubgraph() {
  exitSubgraph()
}

function handleNavigate(target) {
  const targetInfo = normalizeNavigationTarget(target)
  if (!targetInfo?.lookupId) return

  if (targetInfo.stageName && targetInfo.stageName !== currentGraphStage.value) {
    pendingStageNavigation.value = {
      stageName: targetInfo.stageName,
      lookupId: targetInfo.lookupId,
      containerLookupId: targetInfo.containerLookupId,
    }
    selectedNode.value = null
    exitSubgraph()
    currentGraphStage.value = targetInfo.stageName
    return
  }

  const resolvedTarget = resolveNavigationTargetWithContainer(targetInfo)
  if (!resolvedTarget?.node) {
    pendingStageNavigation.value = {
      stageName: currentGraphStage.value,
      lookupId: targetInfo.lookupId,
      containerLookupId: targetInfo.containerLookupId,
    }
    return
  }

  if (resolvedTarget.containerNode) {
    enterSubgraph(resolvedTarget.containerNode)
  } else if (subgraphNode.value) {
    exitSubgraph()
  }

  registerNodes([...resolvedTarget.path, resolvedTarget.node])
  focusNode(resolvedTarget.node, { follow: true, sync: true })
}

function handleFocusStep(step) {
  focusStep(step, { smoothScroll: true })
}

function handleStepUpdate(step) {
  if (!Number.isFinite(step)) return
  focusStep(step, { smoothScroll: false })
}

function handleOverviewStepUpdate(step) {
  if (!Number.isFinite(step)) return
  selectedNode.value = null
  focusStep(step, { smoothScroll: false })
}

async function handleBufferSelect(selection) {
  const step = Number(selection?.step)
  const rankId = Number(selection?.rankId)
  const bufferId = Number(selection?.bufferId)
  if (!Number.isInteger(step) || !Number.isInteger(rankId) || !Number.isInteger(bufferId)) return

  selectedNode.value = null
  focusStep(step, { smoothScroll: false })
  await nextTick()
  inspectorRef.value?.selectLayout?.(rankId, bufferId)
}

function handleTotalStepsUpdate(total) {
  setTotalSteps(total)
}

function stepBackward(amount = 1) {
  playbackVisibleIndex.value = Math.max(0, playbackVisibleIndex.value - Math.max(1, amount))
}

function stepForward(amount = 1) {
  playbackVisibleIndex.value = Math.min(
    Math.max(0, playbackVisibleSteps.value.length - 1),
    playbackVisibleIndex.value + Math.max(1, amount),
  )
}

function togglePlayback() {
  if (!playbackVisibleSteps.value.length) {
    playbackPlaying.value = false
    return
  }

  if (!playbackPlaying.value && playbackVisibleIndex.value >= playbackVisibleSteps.value.length - 1) {
    playbackVisibleIndex.value = 0
  }

  playbackPlaying.value = !playbackPlaying.value
}

function clearPlaybackTimer() {
  if (!playbackTimer) return
  clearInterval(playbackTimer)
  playbackTimer = 0
}

function stopPlayback() {
  playbackPlaying.value = false
  clearPlaybackTimer()
}

function startPlayback() {
  clearPlaybackTimer()

  if (!playbackVisibleSteps.value.length) {
    playbackPlaying.value = false
    return
  }

  playbackTimer = window.setInterval(() => {
    if (playbackVisibleIndex.value >= playbackVisibleSteps.value.length - 1) {
      stopPlayback()
      return
    }
    stepForward()
  }, 1000)
}

function handleNodesLoaded(nodes) {
  registerNodes(nodes)
  registerNodeStepAliases(nodes)
  resolvePendingStageNavigation()
  resolvePendingExternalNavigation()
}

function handleGraphNodeIdsChange(nodeIds) {
  currentGraphNodeIds.value = Array.isArray(nodeIds) ? nodeIds : []
}

async function notifyDagLayoutChange() {
  if (typeof window === 'undefined') return

  while (layoutRefreshTimers.length) {
    clearTimeout(layoutRefreshTimers.pop())
  }

  await nextTick()
  window.dispatchEvent(new Event('resize'))

  for (const delay of [80, 180, 280]) {
    const timer = window.setTimeout(() => {
      window.dispatchEvent(new Event('resize'))
      const index = layoutRefreshTimers.indexOf(timer)
      if (index >= 0) layoutRefreshTimers.splice(index, 1)
    }, delay)
    layoutRefreshTimers.push(timer)
  }
}

function clampCenterTopSize(size) {
  const availableHeight = resolveCenterAvailableHeight()
  const upperBound = Math.max(
    CENTER_TOP_MIN_SIZE,
    Math.min(CENTER_TOP_MAX_SIZE, availableHeight - CENTER_BOTTOM_MIN_SIZE),
  )
  return Math.min(upperBound, Math.max(CENTER_TOP_MIN_SIZE, Math.round(Number(size) || 0)))
}

function resolveDefaultCenterTopSize() {
  const layoutHeight =
    resolveCenterAvailableHeight() ||
    (typeof window === 'undefined' ? 0 : Math.max(0, window.innerHeight - 62))
  return clampCenterTopSize(layoutHeight * CENTER_TOP_DEFAULT_RATIO)
}

function updateCenterAvailableHeight() {
  centerAvailableHeight.value = centerSplitRef.value?.clientHeight ?? 0
  if (centerPaneState.value === 'normal') {
    centerTopSize.value = clampCenterTopSize(centerTopSize.value || resolveDefaultCenterTopSize())
  }
}

function resolveCenterAvailableHeight() {
  if (centerAvailableHeight.value > 0) return centerAvailableHeight.value
  if (centerSplitRef.value?.clientHeight) return centerSplitRef.value.clientHeight
  if (typeof window === 'undefined') return CENTER_TOP_MIN_SIZE + CENTER_BOTTOM_MIN_SIZE
  return Math.max(CENTER_TOP_MIN_SIZE + CENTER_BOTTOM_MIN_SIZE, window.innerHeight - 62)
}

function resolveNodeLookupIds(node) {
  if (!node) return []
  const lookupIds = Array.isArray(node.lookupIds) && node.lookupIds.length ? node.lookupIds : [node.id]
  return lookupIds.filter((lookupId) => typeof lookupId === 'string' && lookupId)
}

function resolveStepForNode(node) {
  const stepByNodeId = indexes.value?.stepByNodeId
  if (!(stepByNodeId instanceof Map) || !node) return null

  for (const lookupId of resolveNodeLookupIds(node)) {
    const step = stepByNodeId.get(lookupId)
    if (Number.isInteger(step)) return step
  }

  return null
}

function taskOpMatchesSelectedNode(node, op) {
  if (!node) return false
  const taskNodeId = typeof op?.taskNodeId === 'string' ? op.taskNodeId : ''
  if (!taskNodeId) return false

  const lookupIds = new Set(resolveNodeLookupIds(node))
  if (lookupIds.has(taskNodeId)) return true

  const opNode = nodeRegistry.value?.get(taskNodeId)
  return Boolean(node.mappingKey && opNode?.mappingKey && node.mappingKey === opNode.mappingKey)
}

function normalizeNavigationTarget(target) {
  if (typeof target === 'string' && target) {
    return {
      stageName: '',
      lookupId: target,
    }
  }

  if (target && typeof target === 'object') {
    const lookupId = typeof target.lookupId === 'string' && target.lookupId
      ? target.lookupId
      : typeof target.nodeId === 'string' && target.nodeId
        ? target.nodeId
        : ''
    const stageName = typeof target.stageName === 'string' ? target.stageName : ''
    const containerLookupId =
      typeof target.containerLookupId === 'string' && target.containerLookupId
        ? target.containerLookupId
        : typeof target.containerNodeId === 'string' && target.containerNodeId
          ? target.containerNodeId
          : ''
    return lookupId ? { stageName, lookupId, containerLookupId } : null
  }

  return null
}

function normalizeExternalNavigationRequest(request) {
  if (!request || typeof request !== 'object') return null

  const stageName = typeof request.stageName === 'string' ? request.stageName : ''
  const lookupId =
    typeof request.lookupId === 'string' && request.lookupId
      ? request.lookupId
      : typeof request.nodeId === 'string' && request.nodeId
        ? request.nodeId
        : ''

  return {
    source: typeof request.source === 'string' ? request.source : '',
    stageName,
    lookupId,
    rankId: Number.isInteger(request.rankId) ? request.rankId : null,
    queueId: Number.isInteger(request.queueId) ? request.queueId : null,
    slotIndex: Number.isInteger(request.slotIndex) ? request.slotIndex : null,
    taskType: typeof request.taskType === 'string' ? request.taskType : '',
    targetView: typeof request.targetView === 'string' ? request.targetView : '',
  }
}

function requestTargetsNode(target) {
  if (!target || typeof target !== 'object') return false
  return Boolean(
    target.lookupId ||
      Number.isInteger(target.slotIndex) ||
      (Number.isInteger(target.rankId) && Number.isInteger(target.queueId)),
  )
}

function shouldMarkIssueNode(target) {
  return (target?.targetView === 'dag' || target?.targetView === 'mem') && target?.source === 'analytic'
}

function updateIssueHighlightForTarget(target, resolvedTarget = null) {
  if (!shouldMarkIssueNode(target) || !resolvedTarget?.node?.id) {
    issueHighlightedNodeIds.value = null
    return
  }

  issueHighlightedNodeIds.value = new Set([resolvedTarget.node.id])
}

function registerNodeStepAliases(nodes) {
  const stepByNodeId = indexes.value?.stepByNodeId
  if (!(stepByNodeId instanceof Map) || !Array.isArray(nodes) || !nodes.length) return

  for (const node of nodes) {
    if (!node?.mappingKey) continue

    for (const lookupId of resolveNodeLookupIds(node)) {
      const step = stepByNodeId.get(lookupId)
      if (Number.isInteger(step)) {
        stepByNodeId.set(node.mappingKey, step)
        break
      }
    }
  }
}

function resolvePendingStageNavigation() {
  const pending = pendingStageNavigation.value
  if (!pending?.lookupId || pending.stageName !== currentGraphStage.value) return

  const resolvedTarget = resolveNavigationTargetWithContainer(pending)
  if (!resolvedTarget?.node) return

  pendingStageNavigation.value = null
  if (resolvedTarget.containerNode) {
    enterSubgraph(resolvedTarget.containerNode)
  } else if (subgraphNode.value) {
    exitSubgraph()
  }
  registerNodes([...resolvedTarget.path, resolvedTarget.node])
  focusNode(resolvedTarget.node, { follow: true, sync: true })
}

function resolvePendingExternalNavigation() {
  const pending = pendingExternalNavigation.value
  if (!pending) return
  if (pending.stageName && pending.stageName !== currentGraphStage.value) return

  const resolvedTarget =
    resolveNavigationTargetWithContainer(pending) ??
    resolveNodeNavigationBySelector(pending)

  if (!resolvedTarget?.node) {
    if (requestTargetsNode(pending)) return
    issueHighlightedNodeIds.value = null
    pendingExternalNavigation.value = null
    return
  }

  updateIssueHighlightForTarget(pending, resolvedTarget)
  pendingExternalNavigation.value = null
  if (resolvedTarget.containerNode) {
    enterSubgraph(resolvedTarget.containerNode)
  } else if (subgraphNode.value) {
    exitSubgraph()
  }
  registerNodes([...resolvedTarget.path, resolvedTarget.node])
  focusNode(resolvedTarget.node, { follow: true, sync: true })
}

function resolveNodeNavigationBySelector(target) {
  if (!target) return null
  const registry = nodeRegistry.value
  if (!(registry instanceof Map)) return null

  const candidates = [...registry.values()].filter((node) => {
    if (!node) return false
    if (target.stageName && node.stageName && node.stageName !== target.stageName) return false
    if (Number.isInteger(target.rankId) && node.rankId !== target.rankId) return false
    if (Number.isInteger(target.queueId) && node.queueId !== target.queueId && node.streamId !== target.queueId) return false
    if (Number.isInteger(target.slotIndex) && node.slotIndex !== target.slotIndex && node.pos !== target.slotIndex) return false
    if (target.taskType && node.taskType && node.taskType !== target.taskType) return false
    return true
  })

  const node = candidates[0]
  if (!node?.id) return null
  return resolveNodeNavigationTarget(node.id)
}

function resolveNavigationTargetWithContainer(target) {
  if (!target?.lookupId) return null

  if (target.containerLookupId) {
    const containerTarget = resolveNodeNavigationTarget(target.containerLookupId)
    if (containerTarget?.node) {
      const nestedTarget = findNodeInDescendants(
        containerTarget.node,
        target.lookupId,
        new WeakSet(),
        containerTarget.path,
      )
      if (nestedTarget) {
        return nestedTarget
      }

      if (nodeMatchesLookupId(containerTarget.node, target.lookupId)) {
        return containerTarget
      }
    }
  }

  return resolveNodeNavigationTarget(target.lookupId)
}

function resolveNodeNavigationTarget(nodeId) {
  if (typeof nodeId !== 'string' || !nodeId) return null

  const registry = nodeRegistry.value
  const directNode = registry?.get(nodeId) ?? null
  const searchCandidates = registry instanceof Map ? [...registry.values()] : []
  const visited = new WeakSet()

  for (const rootNode of searchCandidates) {
    const match = findNodeInHierarchy(rootNode, nodeId, visited)
    if (match) return match
  }

  if (directNode) {
    return {
      node: directNode,
      containerNode: null,
      path: [],
    }
  }

  return null
}

function nodeMatchesLookupId(node, targetId) {
  if (!node?.id || typeof targetId !== 'string' || !targetId) return false
  const lookupIds = Array.isArray(node.lookupIds) && node.lookupIds.length ? node.lookupIds : [node.id]
  return lookupIds.includes(targetId)
}

function findNodeInHierarchy(rootNode, targetId, visited, path = []) {
  if (!rootNode?.id || visited.has(rootNode)) return null
  visited.add(rootNode)

  if (nodeMatchesLookupId(rootNode, targetId)) {
    return {
      node: rootNode,
      containerNode: path.at(-1) ?? null,
      path,
    }
  }

  for (const entry of iterateSubgraphEntries(rootNode?.rawNode?.task?.task_data?.sub_graph, rootNode.rankId)) {
    const nextPath = [...path, rootNode]
    if (nodeMatchesLookupId(entry.node, targetId)) {
      return {
        node: entry.node,
        containerNode: rootNode,
        path: nextPath,
      }
    }

    const nestedMatch = findNodeInHierarchy(entry.node, targetId, visited, nextPath)
    if (nestedMatch) return nestedMatch
  }

  return null
}

function findNodeInDescendants(rootNode, targetId, visited, path = []) {
  if (!rootNode?.id || visited.has(rootNode)) return null
  visited.add(rootNode)

  for (const entry of iterateSubgraphEntries(rootNode?.rawNode?.task?.task_data?.sub_graph, rootNode.rankId)) {
    const nextPath = [...path, rootNode]
    if (nodeMatchesLookupId(entry.node, targetId)) {
      return {
        node: entry.node,
        containerNode: rootNode,
        path: nextPath,
      }
    }

    const nestedMatch = findNodeInDescendants(entry.node, targetId, visited, nextPath)
    if (nestedMatch) return nestedMatch
  }

  return null
}

function iterateSubgraphEntries(streamCollection, fallbackRankId) {
  return normalizeStreamCollection(streamCollection, fallbackRankId, {
    stageName: DEFAULT_MEMVIEW_STAGE,
    topLevel: false,
  }).flatMap((stream) =>
    stream.nodes.map((node) => ({
      rawNode: node.rawNode,
      node,
    })),
  )
}

watch(
  () => [selectedDatasetName.value, selectedRankKeys.value.join(',')],
  () => {
    resetTimeline()
    clearNodeRegistry()
    currentGraphNodeIds.value = []
    pendingStageNavigation.value = null
    issueHighlightedNodeIds.value = null
    if (pendingExternalNavigation.value?.targetView !== 'dag') {
      pendingExternalNavigation.value = null
    }
  },
)

watch(
  () => [currentGraphStage.value, nodeRegistry.value?.size ?? 0],
  () => {
    resolvePendingExternalNavigation()
  },
)

watch(
  () => playbackPlaying.value,
  (playing) => {
    if (playing) {
      startPlayback()
    } else {
      clearPlaybackTimer()
    }
  },
)

watch(
  () => [playbackVisibleSteps.value.length, playbackVisibleIndex.value],
  ([stepCount, currentIndex]) => {
    if (!playbackPlaying.value) return
    if (!stepCount || currentIndex >= stepCount - 1) {
      stopPlayback()
    }
  },
)

onMounted(() => {
  centerTopSize.value = resolveDefaultCenterTopSize()
  ensureDatasetsLoaded()
  updateCenterAvailableHeight()
  centerResizeObserver = new ResizeObserver(() => {
    updateCenterAvailableHeight()
    notifyDagLayoutChange()
  })
  if (centerSplitRef.value) {
    centerResizeObserver.observe(centerSplitRef.value)
  }
})

onBeforeUnmount(() => {
  centerResizeObserver?.disconnect()
  clearPlaybackTimer()
  while (layoutRefreshTimers.length) {
    clearTimeout(layoutRefreshTimers.pop())
  }
  resetTimeline()
})
</script>

<template>
  <section class="dashboard-layout memview-layout">
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
          >
            <template #after-rank>
              <el-collapse
                v-model="issuePanelSections"
                class="dashboard-sidebar-collapse memview-issue-collapse"
              >
                <el-collapse-item name="issues" class="dashboard-subpanel analytic-list-panel">
                  <template #title>
                    <div class="dashboard-panel-toggle">
                      <span class="dashboard-panel-toggle__title">
                        <el-icon><WarningFilled /></el-icon>
                        <span>报错列表</span>
                      </span>
                      <span class="dashboard-panel-toggle__meta">
                        <el-tag size="small" type="danger">{{ issueCount }} 条</el-tag>
                      </span>
                    </div>
                  </template>

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
                      @click="handleIssueSelect(issue)"
                    >
                      <span class="analytic-issue-row__icon" :class="`is-${issue.severityType}`">
                        <el-icon><WarningFilled /></el-icon>
                      </span>
                      <div class="analytic-issue-row__content">
                        <div class="analytic-issue-row__head">
                          <strong>{{ issue.title }}</strong>
                          <div class="analytic-issue-row__head-tags">
                            <el-tag size="small" :type="issue.severityType">{{ issue.severityLabel }}</el-tag>
                          </div>
                        </div>
                        <div class="analytic-issue-row__body">
                          <div v-if="issue.listFieldRows.primaryRow.length" class="analytic-issue-row__field-stack">
                            <div
                              v-for="field in issue.listFieldRows.primaryRow"
                              :key="`primary-${issue.issueId}-${field.label}`"
                              class="analytic-issue-row__field"
                            >
                              <span>{{ field.label }}</span>
                              <strong>{{ field.value }}</strong>
                            </div>
                          </div>
                          <div
                            v-if="issue.listFieldRows.secondaryRow.length"
                            class="analytic-issue-row__field-stack is-secondary"
                          >
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
                      </div>
                    </button>
                  </div>

                  <div v-else class="dashboard-empty-wrap analytic-list-empty">
                    <el-empty description="当前数据集没有 issue 记录。" />
                  </div>
                </el-collapse-item>
              </el-collapse>

              <el-collapse
                v-model="searchPanelSections"
                class="dashboard-sidebar-collapse memview-search-collapse"
              >
                <el-collapse-item name="search" class="dashboard-subpanel memview-search-panel">
                  <template #title>
                    <div class="dashboard-panel-toggle">
                      <span class="dashboard-panel-toggle__title">
                        <el-icon><Search /></el-icon>
                        <span>搜索</span>
                      </span>
                      <span class="dashboard-panel-toggle__meta">
                        <el-tag size="small" type="info">{{ selectedSearchFieldLabel }}</el-tag>
                        <el-tag size="small" type="success">
                          {{ normalizedSearchQuery ? `${searchResults.length} results` : `${selectedRankCount} ranks` }}
                        </el-tag>
                      </span>
                    </div>
                  </template>

                  <section class="memview-sidebar-search">
                    <div class="memview-sidebar-search__controls">
                      <el-select
                        v-model="searchField"
                        class="memview-sidebar-search__field-select"
                        aria-label="Search field"
                      >
                        <el-option
                          v-for="option in SEARCH_FIELD_OPTIONS"
                          :key="option.value"
                          :label="option.label"
                          :value="option.value"
                        />
                      </el-select>

                      <el-input
                        v-model="searchQuery"
                        class="memview-sidebar-search__input"
                        clearable
                        placeholder="Enter keyword"
                        aria-label="Search keyword"
                      />
                    </div>

                    <div class="memview-sidebar-search__meta">
                      <span>Selected ranks: {{ selectedRankCount }}</span>
                      <span>Results: {{ searchResults.length }}</span>
                    </div>

                    <div
                      v-if="normalizedSearchQuery && searchResults.length"
                      class="memview-sidebar-search__results"
                    >
                      <button
                        v-for="result in searchResults"
                        :key="result.node.id"
                        type="button"
                        class="memview-sidebar-search-card"
                        :class="{ 'is-active': selectedNode?.id === result.node.id }"
                        @click="handleSearchResultSelect(result)"
                      >
                        <div class="memview-sidebar-search-card__head">
                          <strong>{{ result.taskId }}</strong>
                          <el-tag size="small" type="info">{{ result.taskType }}</el-tag>
                        </div>

                        <div class="memview-sidebar-search-card__grid">
                          <div class="memview-sidebar-search-card__field memview-sidebar-search-card__field--wide">
                            <span>rank / queue / pos</span>
                            <strong>{{ result.formatLoc }}</strong>
                          </div>
                          <div
                            v-if="searchField === 'notifyId'"
                            class="memview-sidebar-search-card__field memview-sidebar-search-card__field--wide"
                          >
                            <span>notifyId</span>
                            <strong>{{ result.notifyId }}</strong>
                          </div>
                        </div>
                      </button>
                    </div>

                    <div v-else-if="normalizedSearchQuery" class="memview-sidebar-search__empty">
                      <el-empty description="No matching tasks" :image-size="72" />
                    </div>

                    <div v-else class="memview-sidebar-search__hint">
                      Choose taskId, taskType, or notifyId and enter a keyword to search within the selected ranks.
                    </div>
                  </section>
                </el-collapse-item>
              </el-collapse>
            </template>
          </DashboardSidebar>

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
        <div class="dashboard-main-grid memview-main-grid">
          <div ref="centerSplitRef" class="memview-content-area">
            <el-splitter class="dashboard-main-splitter memview-content-splitter" layout="vertical" lazy>
              <el-splitter-panel
                class="dashboard-split-panel dashboard-split-panel--topology"
                :size="renderedCenterTopSize"
                :min="CENTER_COLLAPSED_SIZE"
                @update:size="handleCenterTopResize"
              >
                <button
                  v-if="topPaneCollapsed"
                  type="button"
                  class="dashboard-collapsed-pane dashboard-collapsed-pane--top"
                  @click="expandCenterPane('top')"
                >
                  MemView 内存视图
                </button>
                <MemViewOverviewPanel
                  v-else
                  :dataset-name="selectedDatasetName"
                  :enabled="hasMemoryData"
                  :selected-rank-keys="selectedRankKeys"
                  :current-step="clampedCurrentStep"
                  :display-step-items="displayStepItems"
                  :empty-text="overviewEmptyText"
                  @update:current-step="handleOverviewStepUpdate"
                  @update:total-steps="handleTotalStepsUpdate"
                  @buffer-select="handleBufferSelect"
                />
              </el-splitter-panel>

              <el-splitter-panel class="dashboard-split-panel dashboard-split-panel--table" :min="CENTER_COLLAPSED_SIZE">
                <button
                  v-if="bottomPaneCollapsed"
                  type="button"
                  class="dashboard-collapsed-pane dashboard-collapsed-pane--bottom"
                  @click="expandCenterPane('bottom')"
                >
                  DAGView 任务视图
                </button>
                <MemViewDagPanel
                  v-else
                  :dataset-name="selectedDatasetName"
                  :selected-rank-keys="selectedRankKeys"
                  :selected-node-id="selectedNode?.id ?? ''"
                  :issue-node-ids="dagIssueNodeIds"
                  :subgraph-node="subgraphNode"
                  :current-step="clampedCurrentStep"
                  :total-steps="totalSteps"
                  :display-step-items="displayStepItems"
                  :empty-text="dagEmptyText"
                  @node-select="handleNodeSelect"
                  @update:current-step="handleStepUpdate"
                  @update:total-steps="handleTotalStepsUpdate"
                  @nodes-loaded="handleNodesLoaded"
                  @graph-node-ids-change="handleGraphNodeIdsChange"
                />
              </el-splitter-panel>
            </el-splitter>
          </div>

          <footer class="memview-dag-playback memview-page-playback">
            <div class="memview-dag-playback__controls">
              <button type="button" class="memview-playback-button" aria-label="Jump to start" @click="stepBackward(playbackVisibleSteps.length || 1)">
                <el-icon><DArrowLeft /></el-icon>
              </button>
              <button type="button" class="memview-playback-button" aria-label="Previous step" @click="stepBackward()">
                <el-icon><CaretLeft /></el-icon>
              </button>
              <button type="button" class="memview-playback-button memview-playback-button--primary" :aria-label="playbackPlaying ? 'Pause playback' : 'Play playback'" @click="togglePlayback">
                <el-icon><component :is="playbackPlaying ? VideoPause : VideoPlay" /></el-icon>
              </button>
              <button type="button" class="memview-playback-button" aria-label="Next step" @click="stepForward()">
                <el-icon><CaretRight /></el-icon>
              </button>
              <button type="button" class="memview-playback-button" aria-label="Jump to end" @click="stepForward(playbackVisibleSteps.length || 1)">
                <el-icon><DArrowRight /></el-icon>
              </button>
            </div>

            <div class="memview-dag-playback__timeline">
              <span class="memview-dag-playback__meta">Step {{ playbackStep }}</span>
              <el-slider
                v-model="playbackVisibleIndex"
                class="memview-dag-slider"
                :min="0"
                :max="Math.max(0, playbackVisibleSteps.length - 1)"
                :show-tooltip="false"
              />
              <span class="memview-dag-playback__meta">
                {{ playbackVisibleSteps.length }} / {{ totalSteps }} steps
              </span>
            </div>
          </footer>
        </div>
      </el-splitter-panel>

      <el-splitter-panel
        class="dashboard-split-panel dashboard-split-panel--record"
        :size="renderedRightPanelSize"
        :min="recordCollapsed ? 28 : 320"
        :resizable="!recordCollapsed"
        @update:size="handleRightPanelResize"
      >
        <aside class="dashboard-record-shell" :class="{ 'is-collapsed': recordCollapsed }">
          <MemViewInspectorPanel
            ref="inspectorRef"
            :selected-node="selectedNode"
            :subgraph-active="Boolean(subgraphNode)"
            :subgraph-node="subgraphNode"
            :current-step="clampedCurrentStep"
            :node-step="selectedNodeStep"
            :buffer-ops="selectedNodeBufferOps"
            :node-task-ops="selectedNodeTaskOps"
            :step-buffer-ops="activeStepBufferOps"
            :step-task-ops="activeStepTaskOps"
            :step-layout-buffers="activeStepLayoutBuffers"
            :selected-rank-keys="selectedRankKeys"
            :can-resolve-relation="canResolveRelation"
            :resolve-relation-meta="resolveRelationMeta"
            :empty-text="hasData ? 'Select a node to inspect.' : EMPTY_TEXT"
            @enter-subgraph="handleEnterSubgraph"
            @exit-subgraph="handleExitSubgraph"
            @navigate="handleNavigate"
            @focus-step="handleFocusStep"
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


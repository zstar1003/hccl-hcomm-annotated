/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

import { computed, ref } from 'vue'
import { fetchDatasetDetail, fetchDatasetList } from '../utils'
import { buildDefaultSelectedRankKeys } from '../utils/rankSelection.js'

const EMPTY_TEXT = '暂无数据'

const datasets = ref([])
const selectedDatasetName = ref('')
const selectedRankKeys = ref([])
const datasetDetail = ref(null)
const loadingList = ref(false)
const loadingDetail = ref(false)
const usingFallback = ref(false)

let loadPromise = null

const selectedDataset = computed(
  () => datasets.value.find((item) => item.name === selectedDatasetName.value) ?? null,
)

const hasData = computed(() => datasets.value.length > 0 && Boolean(selectedDataset.value))

const selectedRankCount = computed(
  () => selectedRankKeys.value.filter((key) => key.startsWith('rank-')).length,
)

const fileStats = computed(() => {
  const detail = datasetDetail.value
  const graphFiles = detail?.graph?.reduce((sum, group) => sum + group.files.length, 0) ?? 0
  const memoryFiles = detail?.memory?.reduce((sum, rank) => sum + rank.files.length, 0) ?? 0
  const validationFiles = detail?.validation?.length ?? 0

  return {
    origin: graphFiles,
    translated: validationFiles,
    rebuilt: detail?.graph?.length ?? 0,
    memory: memoryFiles,
    errors: validationFiles,
    total: graphFiles + memoryFiles + validationFiles,
  }
})

const memorySnapshotCount = computed(() => {
  const detail = datasetDetail.value
  return (
    detail?.memory?.reduce(
      (sum, rank) => sum + rank.files.filter((file) => file.startsWith('snapshots_')).length,
      0,
    ) ?? 0
  )
})

const summaryRows = computed(() => {
  const detail = datasetDetail.value
  const dataset = selectedDataset.value
  const opParam = detail?.opParam ?? dataset?.opParam ?? {}

  return [
    ['数据 ID', detail?.dataId ?? dataset?.dataId ?? '--'],
    ['状态', dataset?.success === true ? 'success' : dataset?.success === false ? 'failed' : '--'],
    ['数据集', detail?.name ?? dataset?.name ?? '--'],
    ['算子', 'ReduceScatter'],
    ['Reduce Op', formatReduceType(opParam.reduce_type)],
    ['数据类型', formatDataType(opParam.data_type)],
    ['数据量', formatCount(opParam.data_count)],
    ['文件数', String(fileStats.value.total)],
    ['算子展开类型', formatCmdType(opParam.cmd_type)],
  ]
})

const overviewStats = computed(() => {
  const detail = datasetDetail.value
  const graphStageStats = detail?.graphStageStats ?? {}
  return [
    { label: '错误记录数', value: detail?.errorCount ?? selectedDataset.value?.errorCount ?? 0, icon: 'warning' },
    // { label: '内存快照数', value: memorySnapshotCount.value, icon: 'files' },
    { label: '原图总任务数', value: graphStageStats.inputTaskQueues?.taskCount ?? 0, icon: 'cpu' },
    { label: '翻译图总任务数', value: graphStageStats.inputGraph?.nodeCount ?? 0, icon: 'share' },
    { label: '改造图总任务数', value: graphStageStats.revampGraph?.nodeCount ?? 0, icon: 'trend' },
  ]
})

const tableRows = computed(() =>
  datasets.value.map((dataset) => ({
    datasetKey: dataset.name,
    datasetName: dataset.name || '--',
    operator: 'ReduceScatter',
    rank: Number.isInteger(dataset.opParam?.rank_size) ? String(dataset.opParam.rank_size) : '--',
    type: formatDataType(dataset.opParam?.data_type),
  })),
)

const treeData = computed(() => {
  const detail = datasetDetail.value
  const rankLabels = buildRankLabels(detail)

  if (!rankLabels.length) {
    return []
  }

  return [
    {
      id: 'superpod-0',
      label: 'Superpod',
      caption: 'Dataset',
      children: [
        {
          id: 'server-0',
          label: 'server',
          caption: 'Ranks',
          children: rankLabels.map((rankLabel, index) => ({
            id: `rank-${index}`,
            label: rankLabel,
            caption: rankSummary(detail, index),
          })),
        },
      ],
    },
  ]
})

async function ensureDatasetsLoaded(force = false) {
  if (!force && (loadPromise || datasets.value.length)) {
    return loadPromise ?? Promise.resolve()
  }

  loadPromise = loadDatasetsInternal()
  try {
    await loadPromise
  } finally {
    loadPromise = null
  }
}

async function loadDatasetsInternal() {
  loadingList.value = true

  try {
    const response = await fetchDatasetList()
    datasets.value = normalizeDatasets(response.datasets)
  } catch {
    datasets.value = []
  } finally {
    loadingList.value = false
  }

  if (!selectedDatasetName.value || !datasets.value.some((item) => item.name === selectedDatasetName.value)) {
    selectedDatasetName.value = datasets.value[0]?.name ?? ''
  }

  if (!selectedDatasetName.value) {
    datasetDetail.value = null
    selectedRankKeys.value = []
    return
  }

  if (!datasetDetail.value || datasetDetail.value.name !== selectedDatasetName.value) {
    await selectDataset(selectedDatasetName.value)
  }
}

async function selectDataset(datasetName) {
  if (!datasetName) {
    selectedDatasetName.value = ''
    datasetDetail.value = null
    selectedRankKeys.value = []
    return
  }

  selectedDatasetName.value = datasetName
  loadingDetail.value = true

  try {
    const response = await fetchDatasetDetail(datasetName)
    datasetDetail.value = normalizeDetail(response)
  } catch {
    datasetDetail.value = null
  } finally {
    loadingDetail.value = false
  }

  selectedRankKeys.value = buildDefaultSelectedRankKeys(datasetDetail.value)
}

function setSelectedRankKeys(checkedKeys) {
  selectedRankKeys.value = checkedKeys
}

function normalizeDatasets(source) {
  return source.map((item) => ({
    ...item,
    opParam: {
      ...item.opParam,
      data_type: formatDataType(item.opParam?.data_type),
      reduce_type: formatReduceType(item.opParam?.reduce_type),
      cmd_type: formatCmdType(item.opParam?.cmd_type),
    },
  }))
}

function normalizeDetail(detail) {
  return {
    ...detail,
    opParam: {
      ...detail.opParam,
      data_type: formatDataType(detail.opParam?.data_type),
      reduce_type: formatReduceType(detail.opParam?.reduce_type),
      cmd_type: formatCmdType(detail.opParam?.cmd_type),
    },
  }
}

function buildRankLabels(detail) {
  const ranks = detail?.memory?.map((item) => item.rank).filter((item) => Number.isInteger(item)) ?? []
  if (ranks.length) {
    return ranks.map((rank) => `rank ${rank}`)
  }

  const rankSize = detail?.opParam?.rank_size
  if (Number.isInteger(rankSize) && rankSize > 0) {
    return Array.from({ length: rankSize }, (_, index) => `rank ${index}`)
  }
  return []
}

function rankSummary(detail, index) {
  const rankData = detail?.memory?.find((item) => item.rank === index)
  if (!rankData) {
    return EMPTY_TEXT
  }
  const snapshotCount = rankData.files.filter((file) => file.startsWith('snapshots_')).length
  return `${snapshotCount} snapshots`
}

function formatDataType(value) {
  if (value === 2 || value === '2') return 'INT32'
  return typeof value === 'string' && value ? value.toUpperCase() : '--'
}

function formatReduceType(value) {
  if (value === 0 || value === '0') return 'SUM'
  return typeof value === 'string' && value ? value.toUpperCase() : '--'
}

function formatCmdType(value) {
  const cmdTypeMap = {
    0: 'INVALID',
    1: 'BROADCAST',
    2: 'ALLREDUCE',
    3: 'REDUCE',
    4: 'SEND',
    5: 'RECEIVE',
    6: 'ALLGATHER',
    7: 'REDUCE_SCATTER',
    8: 'ALLTOALLV',
    9: 'ALLTOALLVC',
    10: 'ALLTOALL',
    11: 'GATHER',
    12: 'SCATTER',
    13: 'BATCH_SEND_RECV',
    14: 'BATCH_PUT',
    15: 'BATCH_GET',
    16: 'ALLGATHER_V',
    17: 'REDUCE_SCATTER_V',
    18: 'BATCH_WRITE',
    20: 'HALF_ALLTOALLV',
    21: 'ALL',
    100: 'FINALIZE',
    101: 'INTER_GROUP_SYNC',
    102: 'INIT',
    103: 'BARRIER',
    104: 'MAX',
  }

  const numericValue = Number(value)
  if (Number.isInteger(numericValue) && Object.prototype.hasOwnProperty.call(cmdTypeMap, numericValue)) {
    return cmdTypeMap[numericValue]
  }

  if (typeof value === 'string' && value) {
    const trimmed = value.trim()
    if (!trimmed) return '--'
    if (/^\d+$/.test(trimmed)) {
      const parsed = Number(trimmed)
      return Object.prototype.hasOwnProperty.call(cmdTypeMap, parsed) ? cmdTypeMap[parsed] : `CMD_${parsed}`
    }
    return trimmed.toUpperCase()
  }

  if (Number.isInteger(numericValue)) {
    return `CMD_${numericValue}`
  }

  return '--'
}

function formatCount(value) {
  if (!Number.isFinite(value)) {
    return '--'
  }
  return new Intl.NumberFormat('en-US').format(value)
}

export function useInsightDatasetState() {
  return {
    EMPTY_TEXT,
    datasets,
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
    memorySnapshotCount,
    summaryRows,
    overviewStats,
    tableRows,
    treeData,
    ensureDatasetsLoaded,
    selectDataset,
    setSelectedRankKeys,
    formatCount,
  }
}

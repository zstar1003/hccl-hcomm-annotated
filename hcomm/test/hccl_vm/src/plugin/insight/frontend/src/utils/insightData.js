/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

import { decode } from '@msgpack/msgpack'

const DEFAULT_API_ROOT = ''

function isObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function ensureObject(value, fallback = {}) {
  return isObject(value) ? value : fallback
}

function ensureArray(value) {
  return Array.isArray(value) ? value : []
}

function toNumber(value, fallback = null) {
  return Number.isFinite(value) ? value : fallback
}

function parseRankList(value) {
  return ensureArray(value)
    .map((item) => toNumber(item))
    .filter((item) => item !== null)
}

function parseDatasetItemData(rawDataset = {}) {
  const dataset = ensureObject(rawDataset)

  return {
    name: typeof dataset.name === 'string' ? dataset.name : '',
    dataId: typeof dataset.data_id === 'string' ? dataset.data_id : '',
    success: typeof dataset.success === 'boolean' ? dataset.success : null,
    opParam: ensureObject(dataset.op_param),
    retCode: toNumber(dataset.ret_code),
    errorCount: toNumber(dataset.error_count),
    hasManifest: Boolean(dataset.has_manifest),
    manifestError: typeof dataset.manifest_error === 'string' ? dataset.manifest_error : null,
  }
}

function parseGraphGroupData(rawGroup = {}) {
  const group = ensureObject(rawGroup)

  return {
    name: typeof group.name === 'string' ? group.name : '',
    files: ensureArray(group.files).filter((item) => typeof item === 'string'),
    hasIndex: Boolean(group.has_index),
    ranks: parseRankList(group.ranks),
  }
}

function parseGraphStageStatData(rawStat = null) {
  if (!isObject(rawStat)) {
    return null
  }

  const stat = rawStat

  return {
    rankCount: toNumber(stat.rank_count, 0),
    nodeCount: toNumber(stat.node_count, 0),
    edgeCount: toNumber(stat.edge_count, 0),
    taskCount: toNumber(stat.task_count, 0),
    queueCount: toNumber(stat.queue_count, 0),
    ranks: ensureArray(stat.ranks).map((rank) => ({
      rankId: toNumber(rank?.rank_id),
      nodeCount: toNumber(rank?.node_count, 0),
      edgeCount: toNumber(rank?.edge_count, 0),
      taskCount: toNumber(rank?.task_count, 0),
      queueCount: toNumber(rank?.queue_count, 0),
    })),
  }
}

function parseGraphStageStatsData(rawStats = {}) {
  const stats = ensureObject(rawStats, null)
  if (!stats) {
    return null
  }

  return {
    inputGraph: parseGraphStageStatData(stats.input_graph),
    inputTaskQueues: parseGraphStageStatData(stats.input_task_queues),
    originGraph: parseGraphStageStatData(stats.origin_graph),
    revampGraph: parseGraphStageStatData(stats.revamp_graph),
  }
}

function parseMemoryRankData(rawRank = {}) {
  const rank = ensureObject(rawRank)

  return {
    name: typeof rank.name === 'string' ? rank.name : '',
    rank: toNumber(rank.rank),
    files: ensureArray(rank.files).filter((item) => typeof item === 'string'),
  }
}

function parseDatasetListData(rawPayload = {}) {
  const payload = ensureObject(rawPayload)

  return {
    datasets: ensureArray(payload.datasets).map(parseDatasetItemData),
  }
}

function parseDatasetDetailData(rawManifest = {}) {
  const manifest = ensureObject(rawManifest)
  const runManifest = ensureObject(manifest.run_manifest, null)
  const graphStageStats = manifest.graph_stage_stats ?? runManifest?.graph_stage_stats

  return {
    name: typeof manifest.name === 'string' ? manifest.name : '',
    dataId: typeof manifest.data_id === 'string' ? manifest.data_id : '',
    children: ensureArray(manifest.children).filter((item) => typeof item === 'string'),
    runManifest,
    manifestError: typeof manifest.manifest_error === 'string' ? manifest.manifest_error : null,
    opParam: ensureObject(manifest.op_param, runManifest?.op_param ?? {}),
    success: typeof manifest.success === 'boolean' ? manifest.success : runManifest?.success ?? null,
    retCode: toNumber(manifest.ret_code, toNumber(runManifest?.ret_code)),
    errorCount: toNumber(manifest.error_count, toNumber(runManifest?.error_count)),
    graphStageStats: parseGraphStageStatsData(graphStageStats),
    graph: ensureArray(manifest.graph).map(parseGraphGroupData),
    memory: ensureArray(manifest.memory).map(parseMemoryRankData),
    validation: ensureArray(manifest.validation).filter((item) => typeof item === 'string'),
  }
}

function parseGraphIndexData(rawData = {}) {
  const data = ensureObject(rawData)

  return {
    stage: typeof data.stage === 'string' ? data.stage : '',
    rankCount: toNumber(data.rank_count, 0),
    ranks: parseRankList(data.ranks),
  }
}

function parseGraphRankData(rawData = {}) {
  const data = ensureObject(rawData)

  return {
    type: typeof data.type === 'string' ? data.type : '',
    stage: typeof data.stage === 'string' ? data.stage : '',
    rankId: toNumber(data.rank_id),
    nodes: ensureArray(data.nodes),
    edges: ensureArray(data.edges),
    streams: ensureArray(data.streams),
  }
}

function parseMemoryChunkMeta(rawChunk) {
  const chunk = ensureArray(rawChunk)

  return {
    chunkId: toNumber(chunk[0]),
    path: typeof chunk[1] === 'string' ? chunk[1] : '',
    snapshotCount: toNumber(chunk[2], 0),
    logicalStepBegin: toNumber(chunk[3], 0),
    logicalStepEnd: toNumber(chunk[4], 0),
  }
}

function parseMemoryManifestData(rawData = {}) {
  const data = ensureObject(rawData)

  return {
    rankId: toNumber(data.rank_id),
    chunkCount: toNumber(data.chunk_count, 0),
    chunks: ensureArray(data.chunks).map(parseMemoryChunkMeta),
  }
}

function parseMemoryChunkData(rawData = {}) {
  const data = ensureObject(rawData)
  const snapshots = ensureObject(data.snapshots)

  return {
    rankId: toNumber(data.rank_id),
    chunkId: toNumber(data.chunk_id),
    snapshotCount: toNumber(data.snapshot_count, 0),
    snapshots,
    orderedSnapshots: Object.entries(snapshots)
      .map(([logicalStep, snapshot]) => ({
        logicalStep: Number(logicalStep),
        data: ensureObject(snapshot),
      }))
      .filter((item) => Number.isFinite(item.logicalStep))
      .sort((left, right) => left.logicalStep - right.logicalStep),
  }
}

function joinApiPath(apiRoot, path) {
  const normalizedRoot = apiRoot.endsWith('/') ? apiRoot.slice(0, -1) : apiRoot
  return `${normalizedRoot}${path}`
}

async function readJsonResponse(response) {
  const text = await response.text()
  if (!text) {
    return {}
  }

  try {
    return JSON.parse(text)
  } catch {
    throw new Error(`Failed to parse JSON response from ${response.url}`)
  }
}

async function requestJson(path, { apiRoot = DEFAULT_API_ROOT, fetchImpl = fetch } = {}) {
  const response = await fetchImpl(joinApiPath(apiRoot, path))
  const payload = await readJsonResponse(response)

  if (!response.ok) {
    throw new Error(payload.error || `Request failed: ${response.status}`)
  }

  return payload
}

async function requestArrayBuffer(path, { apiRoot = DEFAULT_API_ROOT, fetchImpl = fetch } = {}) {
  const response = await fetchImpl(joinApiPath(apiRoot, path))

  if (!response.ok) {
    let message = `Request failed: ${response.status}`
    try {
      const payload = await readJsonResponse(response)
      message = payload.error || message
    } catch {
      // Keep the status-based fallback when the error body is not JSON.
    }
    throw new Error(message)
  }

  return response.arrayBuffer()
}

export function buildDatasetUrl(datasetName) {
  return `/api/datasets/${encodeURIComponent(datasetName)}`
}

export function buildMsgpackUrl({ dataset, kind, group, rank, file } = {}) {
  const searchParams = new URLSearchParams()

  if (dataset) searchParams.set('dataset', dataset)
  if (kind) searchParams.set('kind', kind)
  if (group) searchParams.set('group', group)
  if (Number.isInteger(rank)) searchParams.set('rank', String(rank))
  if (file) searchParams.set('file', file)

  return `/api/msgpack?${searchParams.toString()}`
}

export async function fetchDatasetList(options) {
  const payload = await requestJson('/api/data-files', options)
  return parseDatasetListData(payload)
}

export async function fetchDatasetDetail(datasetName, options) {
  const payload = await requestJson(buildDatasetUrl(datasetName), options)
  return parseDatasetDetailData(payload)
}

export async function fetchMsgpack(request, options) {
  return requestArrayBuffer(buildMsgpackUrl(request), options)
}

export function decodeMsgpack(arrayBuffer) {
  return decode(new Uint8Array(arrayBuffer))
}

export async function fetchDecodedMsgpack(request, options) {
  const arrayBuffer = await fetchMsgpack(request, options)
  return decodeMsgpack(arrayBuffer)
}

export function parseGraphIndex(arrayBuffer) {
  return parseGraphIndexData(decodeMsgpack(arrayBuffer))
}

export function parseGraphRank(arrayBuffer) {
  return parseGraphRankData(decodeMsgpack(arrayBuffer))
}

export function parseMemoryManifest(arrayBuffer) {
  return parseMemoryManifestData(decodeMsgpack(arrayBuffer))
}

export function parseMemoryChunk(arrayBuffer) {
  return parseMemoryChunkData(decodeMsgpack(arrayBuffer))
}

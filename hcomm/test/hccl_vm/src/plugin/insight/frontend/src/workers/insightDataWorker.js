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
import { buildMemviewIndex, normalizeSnapshotEntry } from '../utils/memviewIndex.js'

function ensureObject(value, fallback = {}) {
  return value && typeof value === 'object' && !Array.isArray(value) ? value : fallback
}
function ensureArray(value) {
  return Array.isArray(value) ? value : []
}
function toNumber(value, fallback = null) {
  return Number.isFinite(value) ? value : fallback
}
function decodeBuffer(arrayBuffer) {
  return decode(new Uint8Array(arrayBuffer))
}

function decodeMemoryManifest(arrayBuffer) {
  const raw = ensureObject(decodeBuffer(arrayBuffer))
  return {
    rankId: toNumber(raw.rank_id),
    chunkCount: toNumber(raw.chunk_count, 0),
    chunks: ensureArray(raw.chunks).map((chunk) => {
      const tuple = ensureArray(chunk)
      return {
        chunkId: toNumber(tuple[0]),
        path: typeof tuple[1] === 'string' ? tuple[1] : '',
        snapshotCount: toNumber(tuple[2], 0),
        logicalStepBegin: toNumber(tuple[3], 0),
        logicalStepEnd: toNumber(tuple[4], 0),
      }
    }),
  }
}

function decodeMemoryChunk(arrayBuffer) {
  const raw = ensureObject(decodeBuffer(arrayBuffer))
  const snapshots = ensureObject(raw.snapshots)
  const ordered = Object.entries(snapshots)
    .map(([step, data]) => ({
      logicalStep: Number(step),
      data: ensureObject(data),
    }))
    .filter((item) => Number.isFinite(item.logicalStep))
    .sort((l, r) => l.logicalStep - r.logicalStep)

  return {
    rankId: toNumber(raw.rank_id),
    chunkId: toNumber(raw.chunk_id),
    snapshots: ordered.map((item) => normalizeSnapshotEntry(item)),
  }
}

function decodeGraphIndex(arrayBuffer) {
  const raw = ensureObject(decodeBuffer(arrayBuffer))
  return {
    stage: typeof raw.stage === 'string' ? raw.stage : '',
    rankCount: toNumber(raw.rank_count, 0),
    ranks: ensureArray(raw.ranks).map((item) => toNumber(item)).filter((item) => item !== null),
  }
}

function decodeGraphRank(arrayBuffer) {
  const raw = ensureObject(decodeBuffer(arrayBuffer))
  return {
    stage: typeof raw.stage === 'string' ? raw.stage : '',
    rankId: toNumber(raw.rank_id),
    nodes: ensureArray(raw.nodes),
    edges: ensureArray(raw.edges),
  }
}

function handleMessage(message) {
  const { id, kind, payload } = message

  try {
    if (kind === 'decodeGraphIndex') {
      const result = decodeGraphIndex(payload.buffer)
      return { id, ok: true, result }
    }
    if (kind === 'decodeGraphRank') {
      const result = decodeGraphRank(payload.buffer)
      return { id, ok: true, result }
    }
    if (kind === 'decodeMemoryManifest') {
      const result = decodeMemoryManifest(payload.buffer)
      return { id, ok: true, result }
    }
    if (kind === 'decodeMemoryChunk') {
      const result = decodeMemoryChunk(payload.buffer)
      return { id, ok: true, result }
    }
    if (kind === 'buildMemviewIndex') {
      const { rankBundles, totalSteps } = payload
      const index = buildMemviewIndex(rankBundles, { totalSteps })
      const transferables = []
      const presenceSerialized = []
      index.bufferPresence.forEach((array, rankId) => {
        presenceSerialized.push([rankId, array])
        transferables.push(array.buffer)
      })
      const changedSerialized = []
      index.bufferChangedFlags.forEach((array, rankId) => {
        changedSerialized.push([rankId, array])
        transferables.push(array.buffer)
      })
      transferables.push(index.memoryTaskIds.offsets.buffer, index.bufferOps.offsets.buffer)
      if (index.taskOps?.offsets?.buffer) {
        transferables.push(index.taskOps.offsets.buffer)
      }

      return {
        id,
        ok: true,
        result: {
          totalSteps: index.totalSteps,
          memoryTaskIds: index.memoryTaskIds,
          stepByNodeId: Array.from(index.stepByNodeId.entries()),
          bufferPresence: presenceSerialized,
          bufferChangedFlags: changedSerialized,
          bufferOps: index.bufferOps,
          taskOps: index.taskOps,
          bufferOpsByNode: Array.from(index.bufferOpsByNode.entries()),
        },
        transferables,
      }
    }

    return { id, ok: false, error: `Unknown worker kind: ${kind}` }
  } catch (error) {
    return { id, ok: false, error: error instanceof Error ? error.message : String(error) }
  }
}

self.addEventListener('message', (event) => {
  const response = handleMessage(event.data)
  const transferables = response.transferables ?? []
  delete response.transferables
  self.postMessage(response, transferables)
})

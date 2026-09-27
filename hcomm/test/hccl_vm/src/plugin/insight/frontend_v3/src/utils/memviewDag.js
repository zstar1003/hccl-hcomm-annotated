/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

export const DEFAULT_MEMVIEW_STAGE = 'input_graph'
export const INPUT_TASK_QUEUE_STAGE = 'input_task_queues'

export function resolveCounterpartMemviewStage(stageName) {
  if (stageName === DEFAULT_MEMVIEW_STAGE) return INPUT_TASK_QUEUE_STAGE
  if (stageName === INPUT_TASK_QUEUE_STAGE) return DEFAULT_MEMVIEW_STAGE
  return ''
}

function ensureArray(value) {
  return Array.isArray(value) ? value : []
}

function ensureObject(value, fallback = {}) {
  return value && typeof value === 'object' && !Array.isArray(value) ? value : fallback
}

function uniqueStrings(values) {
  return [...new Set(ensureArray(values).filter((value) => typeof value === 'string' && value))]
}

function compactStreams(streams) {
  return ensureArray(streams).filter((stream) => Array.isArray(stream?.nodes) && stream.nodes.length > 0)
}

function sortNodesByPosThenOrder(nodes) {
  return [...ensureArray(nodes)].sort((left, right) => {
    const leftPos = Number.isFinite(left?.pos) ? Number(left.pos) : Number.POSITIVE_INFINITY
    const rightPos = Number.isFinite(right?.pos) ? Number(right.pos) : Number.POSITIVE_INFINITY
    if (leftPos !== rightPos) {
      return leftPos - rightPos
    }

    const leftOrder = Number.isFinite(left?.orderIndex) ? Number(left.orderIndex) : Number.POSITIVE_INFINITY
    const rightOrder = Number.isFinite(right?.orderIndex) ? Number(right.orderIndex) : Number.POSITIVE_INFINITY
    return leftOrder - rightOrder
  })
}

export function resolveGraphTaskType(entry) {
  return String(entry?.task?.task_type ?? entry?.task_type ?? 'TASK')
}

export function resolveGraphTaskData(entry) {
  return ensureObject(entry?.task?.task_data ?? entry?.task_data)
}

export function resolveStreamId(entry, fallback = 0) {
  if (Number.isFinite(entry?.queue_id)) return Number(entry.queue_id)
  if (Number.isFinite(entry?.stream_id)) return Number(entry.stream_id)
  if (Number.isFinite(entry?.streamId)) return Number(entry.streamId)
  if (Number.isFinite(entry?.pipe_idx) && Number(entry.pipe_idx) >= 0) return Number(entry.pipe_idx)
  return fallback
}

function resolveRankId(entry, fallbackRankId = null) {
  const taskData = resolveGraphTaskData(entry)
  if (Number.isFinite(taskData?.rank_id)) return Number(taskData.rank_id)
  if (Number.isFinite(entry?.rank_id)) return Number(entry.rank_id)
  if (Number.isFinite(entry?.rankId)) return Number(entry.rankId)
  return Number.isFinite(fallbackRankId) ? Number(fallbackRankId) : null
}

function resolveSlotIndex(entry, orderIndex) {
  if (Number.isFinite(entry?.pos)) {
    return Number(entry.pos)
  }
  if (Number.isFinite(entry?.pipe_pos) && Number(entry.pipe_pos) >= 0) {
    return Number(entry.pipe_pos)
  }
  return Number(orderIndex) || 0
}

function resolveNodeId(entry) {
  const directId =
    entry?.node_id ??
    entry?.id ??
    entry?.task_id ??
    entry?.task?.node_id ??
    entry?.task?.id ??
    entry?.task?.task_id

  if (directId !== undefined && directId !== null && String(directId).trim()) {
    return String(directId)
  }
  return ''
}

function formatTaskLabel(entry) {
  return resolveGraphTaskType(entry)
    .replace(/^BATCH_/i, '')
    .replaceAll('_', ' ')
    .trim()
}

function formatTaskSubtitle(entry, slotIndex) {
  const data = resolveGraphTaskData(entry)
  if (data.src_slice?.buffer_type && data.dst_slice?.buffer_type) {
    return `${data.src_slice.buffer_type} -> ${data.dst_slice.buffer_type}`
  }
  if (Array.isArray(data.src_slices) && data.dst_slice?.buffer_type) {
    return `${data.src_slices.length} inputs -> ${data.dst_slice.buffer_type}`
  }
  if (typeof data.remote_rank === 'number') {
    return `remote rank ${data.remote_rank}`
  }
  if (Number.isInteger(slotIndex)) {
    return `slot ${slotIndex}`
  }
  return resolveNodeId(entry)
}

function createNodeMappings({ stageName, lookupId, streamId, slotIndex, taskType }) {
  if (!lookupId) return []

  const mappedStage = resolveCounterpartMemviewStage(stageName)
  if (!mappedStage) return []

  return [
    {
      stageName: mappedStage,
      lookupId,
      label: `Queue ${streamId} / Task ${slotIndex}`,
      taskType: String(taskType)
        .replace(/^BATCH_/i, '')
        .replaceAll('_', ' ')
        .trim(),
    },
  ]
}

export function normalizeGraphNode(entry, fallbackRankId, streamId, orderIndex, options = {}) {
  const stageName = typeof options.stageName === 'string' && options.stageName ? options.stageName : DEFAULT_MEMVIEW_STAGE
  const rankId = resolveRankId(entry, fallbackRankId)
  const slotIndex = resolveSlotIndex(entry, orderIndex)
  const taskType = resolveGraphTaskType(entry)
  const taskData = resolveGraphTaskData(entry)
  const id = resolveNodeId(entry)
  const directParents = ensureArray(entry?.parents).filter((value) => typeof value === 'string' && value)
  const directChildren = ensureArray(entry?.children).filter((value) => typeof value === 'string' && value)

  return {
    id,
    stageName,
    actualNodeId: id,
    lookupIds: uniqueStrings([id]),
    mappingKey: id,
    nodeMappings: createNodeMappings({ stageName, lookupId: id, streamId, slotIndex, taskType }),
    label: formatTaskLabel(entry),
    subtitle: formatTaskSubtitle(entry, slotIndex),
    taskType,
    taskData,
    rankId,
    streamId,
    queueId: Number.isFinite(entry?.queue_id) ? Number(entry.queue_id) : streamId,
    orderIndex,
    slotIndex,
    globalStep: Number.isFinite(entry?.global_step) ? Number(entry.global_step) : null,
    localStep: Number.isFinite(entry?.local_step) ? Number(entry.local_step) : null,
    pos: Number.isFinite(entry?.pos) ? Number(entry.pos) : null,
    posInfo: typeof entry?.pos_info === 'string' ? entry.pos_info : '',
    hasSubgraph: Array.isArray(taskData?.sub_graph),
    parents: directParents,
    children: directChildren,
    rawNode: entry,
  }
}

export function normalizeNodeList(streamEntries, rankId, streamId, options = {}) {
  const nodes = ensureArray(streamEntries)
    .filter(Boolean)
    .map((entry, orderIndex) => normalizeGraphNode(entry, rankId, streamId, orderIndex, options))
  return sortNodesByPosThenOrder(nodes)
}

export function normalizeTopLevelStreams(rankGraph, stageName = DEFAULT_MEMVIEW_STAGE) {
  if (Array.isArray(rankGraph?.streams) && rankGraph.streams.length > 0) {
    return compactStreams(
      ensureArray(rankGraph?.streams)
        .filter(Boolean)
        .map((stream, streamIndex) => {
          const streamId = Number.isFinite(stream?.queue_id) ? Number(stream.queue_id) : streamIndex
          return {
            streamId,
            nodes: normalizeNodeList(stream?.tasks, rankGraph?.rankId, streamId, {
              stageName,
              topLevel: true,
            }),
          }
        })
        .sort((left, right) => left.streamId - right.streamId),
    )
  }

  const groupedStreams = new Map()
  ensureArray(rankGraph?.nodes)
    .filter(Boolean)
    .forEach((entry) => {
      const streamId = resolveStreamId(entry, 0)
      if (!groupedStreams.has(streamId)) {
        groupedStreams.set(streamId, [])
      }
      groupedStreams.get(streamId).push(entry)
    })

  return compactStreams(
    [...groupedStreams.entries()]
      .sort((left, right) => left[0] - right[0])
      .map(([streamId, streamEntries]) => ({
        streamId,
        nodes: normalizeNodeList(streamEntries, rankGraph?.rankId, streamId, {
          stageName,
          topLevel: true,
        }),
      })),
  )
}

export function normalizeStreamCollection(streamCollection, rankId, options = {}) {
  const stageName = typeof options.stageName === 'string' && options.stageName ? options.stageName : DEFAULT_MEMVIEW_STAGE
  const rawCollection = ensureArray(streamCollection).filter(Boolean)
  if (!rawCollection.length) return []

  const nestedStreams = rawCollection.filter((entry) => Array.isArray(entry))
  if (nestedStreams.length === rawCollection.length) {
    return compactStreams(
      nestedStreams.map((streamEntries, streamIndex) => {
        const streamId = resolveStreamId(streamEntries[0], streamIndex)
        return {
          streamId,
          nodes: normalizeNodeList(streamEntries, rankId, streamId, {
            stageName,
            topLevel: false,
          }),
        }
      }),
    )
  }

  const groupedStreams = new Map()
  rawCollection
    .flatMap((entry) => (Array.isArray(entry) ? entry : [entry]))
    .filter(Boolean)
    .forEach((entry) => {
      const streamId = resolveStreamId(entry, 0)
      if (!groupedStreams.has(streamId)) groupedStreams.set(streamId, [])
      groupedStreams.get(streamId).push(entry)
    })

  return compactStreams(
    [...groupedStreams.entries()]
      .sort((left, right) => left[0] - right[0])
      .map(([streamId, streamEntries]) => ({
        streamId,
        nodes: normalizeNodeList(streamEntries, rankId, streamId, {
          stageName,
          topLevel: false,
        }),
      })),
  )
}

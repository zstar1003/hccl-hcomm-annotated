/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

const FOCUS_PLACEHOLDER_PREFIX = '__focus_placeholder__'

function toIntegerOrNull(value) {
  return Number.isInteger(value) ? value : null
}

function uniqueStrings(values) {
  return [...new Set((Array.isArray(values) ? values : []).filter((value) => typeof value === 'string' && value))]
}

function remapNodeIds(ids, idMap) {
  if (!(idMap instanceof Map) || !idMap.size) return []
  return uniqueStrings(
    (Array.isArray(ids) ? ids : [])
      .map((id) => idMap.get(id))
      .filter((id) => typeof id === 'string' && id),
  )
}

export function createFocusPlaceholderNode({
  index = 0,
  offset = 0,
  rankId = null,
  streamId = null,
  streamKey = '',
} = {}) {
  const safeIndex = Math.max(0, Math.floor(Number(index) || 0))
  const safeStreamKey = typeof streamKey === 'string' && streamKey ? streamKey : 'stream'
  return {
    id: `${FOCUS_PLACEHOLDER_PREFIX}${safeStreamKey}_${safeIndex}`,
    lookupIds: [],
    mappingKey: '',
    label: '',
    subtitle: '',
    taskType: '',
    taskData: {},
    rankId: toIntegerOrNull(rankId),
    streamId: toIntegerOrNull(streamId),
    queueId: toIntegerOrNull(streamId),
    orderIndex: safeIndex,
    slotIndex: safeIndex,
    globalStep: null,
    localStep: null,
    pos: null,
    posInfo: '',
    hasSubgraph: false,
    parents: [],
    children: [],
    rawNode: null,
    isPlaceholder: true,
    focusWindowIndex: safeIndex,
    focusWindowOffset: Math.floor(Number(offset) || 0),
  }
}

export function buildCenteredNodeWindow(nodes, anchorIndex, radius = 10, context = {}) {
  const sourceNodes = Array.isArray(nodes) ? nodes : []
  const safeAnchorIndex = Math.max(0, Math.floor(Number(anchorIndex) || 0))
  const safeRadius = Math.max(0, Math.floor(Number(radius) || 0))
  const windowNodes = []

  for (let offset = -safeRadius; offset <= safeRadius; offset += 1) {
    const sourceNode = sourceNodes[safeAnchorIndex + offset]
    if (sourceNode) {
      windowNodes.push({
        ...sourceNode,
        isPlaceholder: Boolean(sourceNode?.isPlaceholder),
        focusWindowIndex: offset + safeRadius,
        focusWindowOffset: offset,
      })
      continue
    }

    windowNodes.push(
      createFocusPlaceholderNode({
        index: offset + safeRadius,
        offset,
        rankId: context.rankId ?? null,
        streamId: context.streamId ?? null,
        streamKey: context.streamKey ?? '',
      }),
    )
  }

  return windowNodes
}

export function buildCenteredFocusStreamWindow(sourceStream, anchorNodeId, radius = 10, context = {}) {
  const sourceNodes = Array.isArray(sourceStream?.nodes) ? sourceStream.nodes : []
  if (!sourceNodes.length || typeof anchorNodeId !== 'string' || !anchorNodeId) return null

  const anchorIndex = sourceNodes.findIndex((node) => node?.id === anchorNodeId)
  if (anchorIndex < 0) return null

  const sourceStreamKey = sourceStream.streamKey ?? `stream_${sourceStream.streamId ?? 'unknown'}`
  const safeWindowKey = typeof context.focusWindowKey === 'string' && context.focusWindowKey
    ? context.focusWindowKey
    : `${sourceStreamKey}::focus_${anchorNodeId}`
  const windowNodes = buildCenteredNodeWindow(sourceNodes, anchorIndex, radius, {
    rankId: sourceStream.rankId ?? context.rankId ?? null,
    streamId: sourceStream.streamId ?? context.streamId ?? null,
    streamKey: safeWindowKey,
  })

  const cloneIdMap = new Map()
  for (const node of windowNodes) {
    if (!node || node.isPlaceholder || typeof node.id !== 'string' || !node.id) continue
    cloneIdMap.set(node.id, `${safeWindowKey}::${node.id}`)
  }

  const focusNodes = windowNodes.map((node, index) => {
    if (!node) return node
    if (node.isPlaceholder) {
      return {
        ...node,
        streamKey: safeWindowKey,
        focusWindowKey: safeWindowKey,
        focusWindowIndex: index,
        focusWindowOffset: node.focusWindowOffset ?? 0,
      }
    }

    const cloneId = cloneIdMap.get(node.id)
    const lookupIds = uniqueStrings([...(Array.isArray(node.lookupIds) ? node.lookupIds : []), node.id])

    return {
      ...node,
      id: cloneId ?? node.id,
      lookupIds,
      mappingKey: node.mappingKey ?? node.id,
      streamKey: safeWindowKey,
      parents: remapNodeIds(node.parents, cloneIdMap),
      children: remapNodeIds(node.children, cloneIdMap),
      focusOriginalNodeId: node.id,
      focusWindowKey: safeWindowKey,
      focusWindowIndex: index,
      focusWindowOffset: node.focusWindowOffset ?? 0,
    }
  })

  const anchorCloneId = cloneIdMap.get(anchorNodeId) ?? ''
  const anchorNode = windowNodes[anchorIndex] ?? null
  return {
    ...sourceStream,
    streamKey: safeWindowKey,
    label: `Rank ${sourceStream.rankId ?? '-'} / Stream ${sourceStream.streamId ?? '-'}`,
    nodes: focusNodes,
    focusAnchorNodeId: anchorCloneId,
    focusAnchorSourceNodeId: anchorNodeId,
    focusAnchorIndex: anchorIndex,
    focusWindowRadius: radius,
    focusWindowKey: safeWindowKey,
  }
}

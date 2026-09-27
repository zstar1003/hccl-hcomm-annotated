/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

import { MS_BUFFER_ID, normalizeMsId } from './msMemory.js'

export const BUFFER_SLOTS = [
  { id: 0, label: 'INPUT' },
  { id: 1, label: 'OUTPUT' },
  { id: 2, label: 'CCL' },
  { id: 3, label: 'SCRATCH' },
  { id: 4, label: 'INPUT_AIV' },
  { id: 5, label: 'OUTPUT_AIV' },
  { id: 6, label: 'AIV_COMMINFO' },
  { id: 7, label: 'USERBUF_AIV' },
  { id: 8, label: 'MS' },
]

export const BUFFER_LABEL_BY_ID = new Map(BUFFER_SLOTS.map((slot) => [slot.id, slot.label]))
const BUFFER_ID_BY_LABEL = new Map(BUFFER_SLOTS.map((slot) => [slot.label.toUpperCase(), slot.id]))
BUFFER_ID_BY_LABEL.set('MS_CCU', MS_BUFFER_ID)
const TASK_PARENT_SEARCH_DEPTH = 6

export const RANK_PALETTE = [
  '#ff9f36',
  '#7357ff',
  '#33b8c7',
  '#18b275',
  '#f3bc2d',
  '#2ba5ff',
  '#b35cff',
  '#ff4e98',
]

export function rankColorFor(rankId, rankOrder = []) {
  const index = rankOrder.indexOf(rankId)
  const resolvedIndex = index >= 0 ? index : Number.isFinite(rankId) ? rankId : 0
  return RANK_PALETTE[((resolvedIndex % RANK_PALETTE.length) + RANK_PALETTE.length) % RANK_PALETTE.length]
}

export function fnv1a(str) {
  let hash = 0x811c9dc5
  for (let i = 0; i < str.length; i += 1) {
    hash ^= str.charCodeAt(i)
    hash = Math.imul(hash, 0x01000193)
  }
  return hash >>> 0
}

function parseRankFromNodeId(nodeId) {
  const value = String(nodeId ?? '')
  let match = value.match(/^rank_(\d+)_/i)
  if (match) return Number(match[1])
  match = value.match(/^r_(\d+)_/i)
  if (match) return Number(match[1])
  match = value.match(/^r(\d+)_/i)
  if (match) return Number(match[1])
  return null
}

function toFiniteNumber(value, fallback = null) {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : fallback
}

function compareNullableNumber(left, right) {
  if (left === right) return 0
  if (left === null || left === undefined) return 1
  if (right === null || right === undefined) return -1
  return left - right
}

function compareSourceRefs(left, right) {
  return (
    compareNullableNumber(left.rankId, right.rankId) ||
    compareNullableNumber(left.bufferId, right.bufferId) ||
    compareNullableNumber(left.addr, right.addr) ||
    compareNullableNumber(left.len, right.len) ||
    compareNullableNumber(left.step, right.step) ||
    String(left.nodeId).localeCompare(String(right.nodeId))
  )
}

function sourceRefSignature(sourceRef) {
  return `${sourceRef.rankId ?? 'x'}:${sourceRef.bufferId ?? 'x'}:${sourceRef.addr ?? 'x'}:${sourceRef.len ?? 'x'}@${sourceRef.step ?? 'x'}#${sourceRef.nodeId ?? ''}`
}

function normalizeBufferId(value) {
  if (Number.isInteger(value)) return value >= 0 ? Number(value) : null
  if (typeof value === 'string') {
    const normalized = value.trim().toUpperCase()
    if (!normalized) return null
    if (BUFFER_ID_BY_LABEL.has(normalized)) return BUFFER_ID_BY_LABEL.get(normalized)
    if (/^\d+$/.test(normalized)) {
      const asNumber = Number(normalized)
      return asNumber >= 0 ? asNumber : null
    }
  }
  return null
}

function normalizeSlice(slice, fallbackRankId = null) {
  if (!slice || typeof slice !== 'object') return null
  const bufferId = normalizeBufferId(slice.buffer_type ?? slice.bufferType ?? slice.buffer)
  if (!Number.isInteger(bufferId)) return null
  const rankCandidate = toFiniteNumber(slice.rank_id ?? slice.rankId ?? slice.rank, null)
  const addr = toFiniteNumber(slice.offset ?? slice.addr ?? slice.start_addr, null)
  return {
    rankId: Number.isInteger(rankCandidate) ? rankCandidate : fallbackRankId,
    bufferId,
    addr,
    len: toFiniteNumber(slice.size ?? slice.len, null),
    msId: normalizeMsId(slice.ms_id ?? slice.msId, {
      bufferType: slice.buffer_type ?? slice.bufferType ?? slice.buffer,
      bufferId,
      offset: addr,
    }),
  }
}

function normalizeSourceRefs(rawSources) {
  if (!Array.isArray(rawSources) || !rawSources.length) return []

  const dedup = new Map()
  for (const source of rawSources) {
    const tuple = Array.isArray(source?.buf) ? source.buf : []
    const rankFromBuf = Number.isInteger(tuple[0]) ? Number(tuple[0]) : null
    const bufferFromBuf = Number.isInteger(tuple[1]) ? Number(tuple[1]) : null
    const addrFromBuf = toFiniteNumber(tuple[2], null)
    const lenFromBuf = toFiniteNumber(tuple[3], null)
    const nodeId = typeof source?.node === 'string' ? source.node : ''
    const rankFromNodeId = parseRankFromNodeId(nodeId)

    const sourceRef = {
      rankId: rankFromBuf ?? rankFromNodeId,
      bufferId: bufferFromBuf,
      addr: addrFromBuf,
      len: lenFromBuf,
      msId: normalizeMsId(source?.ms_id ?? source?.msId, {
        bufferId: bufferFromBuf,
        offset: addrFromBuf,
      }),
      step: Number.isFinite(source?.snapshot) ? Number(source.snapshot) : null,
      nodeId,
    }
    dedup.set(sourceRefSignature(sourceRef), sourceRef)
  }

  return Array.from(dedup.values()).sort(compareSourceRefs)
}

function normalizeBufferItem(bufferItem) {
  const semantics = Array.isArray(bufferItem?.semantics) ? bufferItem.semantics : []
  const bufferType = Number.isInteger(bufferItem?.buffer_type) ? Number(bufferItem.buffer_type) : -1
  if (bufferType < 0) return null

  let totalSize = 0
  const segmentDedup = new Map()
  const segmentCollisionCount = new Map()

  for (const semantic of semantics) {
    const startAddr = toFiniteNumber(semantic?.start_addr, 0)
    const size = toFiniteNumber(semantic?.size, 0)
    if (Number.isFinite(size)) totalSize += size
    const isReduce = Number(semantic?.is_reduce) === 1
    const reduceType = Number.isFinite(semantic?.reduce_type) ? Number(semantic.reduce_type) : null
    const sourceRefs = normalizeSourceRefs(semantic?.srcs)
    const sourceSignature = sourceRefs.map((sourceRef) => sourceRefSignature(sourceRef)).join(',')

    const baseSegmentKey = `${bufferType}:${startAddr}:${size}`
    const collisionIndex = segmentCollisionCount.get(baseSegmentKey) ?? 0
    segmentCollisionCount.set(baseSegmentKey, collisionIndex + 1)
    const segmentKey = collisionIndex === 0 ? baseSegmentKey : `${baseSegmentKey}#${collisionIndex}`
    const segmentSignature = `${Number(isReduce)}|${reduceType ?? 'x'}|${sourceSignature}`

    segmentDedup.set(segmentKey, {
      key: segmentKey,
      startAddr,
      size,
      isReduce,
      reduceType,
      sourceRefs,
      signature: segmentSignature,
    })
  }

  const segments = Array.from(segmentDedup.values()).sort(
    (left, right) => left.startAddr - right.startAddr || left.size - right.size || left.key.localeCompare(right.key),
  )

  const mergedSourceRefs = new Map()
  for (const segment of segments) {
    for (const sourceRef of segment.sourceRefs) {
      mergedSourceRefs.set(sourceRefSignature(sourceRef), sourceRef)
    }
  }
  const srcRefs = Array.from(mergedSourceRefs.values()).sort(compareSourceRefs)

  const srcRanks = Array.from(
    new Set(srcRefs.filter((sourceRef) => Number.isInteger(sourceRef.rankId)).map((sourceRef) => sourceRef.rankId)),
  ).sort((left, right) => left - right)

  const srcNodes = Array.from(
    new Map(
      srcRefs
        .filter((sourceRef) => typeof sourceRef.nodeId === 'string' && sourceRef.nodeId.length > 0)
        .map((sourceRef) => [`${sourceRef.nodeId}@${sourceRef.step ?? 'x'}`, { nodeId: sourceRef.nodeId, step: sourceRef.step }]),
    ).values(),
  ).sort((left, right) => compareNullableNumber(left.step, right.step) || left.nodeId.localeCompare(right.nodeId))

  const signatureSource = segments.map((segment) => `${segment.key}|${segment.signature}`).join(';')
  const hasReduce = segments.some((segment) => segment.isReduce)

  return {
    bufferType,
    semanticsCount: segments.length,
    totalSize: Math.max(0, Math.round(totalSize)),
    isReduce: hasReduce,
    segments,
    srcNodes,
    srcRefs,
    srcRanks,
    signature: fnv1a(signatureSource),
  }
}

export function normalizeSnapshotEntry(snapshotEntry) {
  const data = snapshotEntry?.data ?? snapshotEntry ?? {}
  const layout = Array.isArray(data.layout) ? data.layout : []
  const buffers = []
  const memoryTaskIds = Array.isArray(data.memory_task_ids)
    ? data.memory_task_ids.filter((id) => typeof id === 'string')
    : []

  for (const bufferItem of layout) {
    const normalized = normalizeBufferItem(bufferItem)
    if (normalized) buffers.push(normalized)
  }

  return {
    logicalStep: Number(snapshotEntry?.logicalStep ?? data?.logicalStep ?? -1),
    memoryTaskIds,
    buffers,
  }
}

function buildPresenceMask(bufferState) {
  let mask = 0
  for (const bufferType of bufferState.keys()) {
    if (bufferType >= 0 && bufferType < 16) {
      mask |= 1 << bufferType
    }
  }
  return mask
}

function buildBufferState(snapshot) {
  const state = new Map()
  for (const buffer of snapshot?.buffers ?? []) {
    if (!Number.isInteger(buffer?.bufferType) || buffer.bufferType < 0) continue
    state.set(buffer.bufferType, buffer)
  }
  return state
}

function getTaskNode(taskNodeLookup, nodeId) {
  if (!nodeId) return null
  if (taskNodeLookup instanceof Map) return taskNodeLookup.get(nodeId) ?? null
  if (taskNodeLookup && typeof taskNodeLookup === 'object') return taskNodeLookup[nodeId] ?? null
  return null
}

function pushTaskFlow(flows, srcSlice, dstSlice, node, { isReduce = false, reduceOp = null } = {}) {
  if (!srcSlice || !dstSlice) return
  if (!Number.isInteger(dstSlice.rankId) || !Number.isInteger(dstSlice.bufferId)) return
  if (!Number.isInteger(srcSlice.bufferId)) return
  flows.push({
    srcRankId: Number.isInteger(srcSlice.rankId) ? srcSlice.rankId : node.rankId,
    srcBufferId: srcSlice.bufferId,
    srcAddr: srcSlice.addr,
    srcLen: srcSlice.len,
    srcMsId: srcSlice.msId,
    dstRankId: dstSlice.rankId,
    dstBufferId: dstSlice.bufferId,
    dstAddr: dstSlice.addr,
    dstLen: dstSlice.len,
    dstMsId: dstSlice.msId,
    nodeId: node.id,
    taskType: node.taskType,
    isReduce,
    reduceOp: Number.isFinite(reduceOp) ? Number(reduceOp) : null,
  })
}

function flowsFromTaskNode(node) {
  if (!node?.taskData || typeof node.taskData !== 'object') return []
  const data = node.taskData
  const flows = []
  const taskType = String(node.taskType ?? '').toUpperCase()
  const reduceOp = Number.isFinite(data.reduce_op) ? Number(data.reduce_op) : null
  const fallbackRank = Number.isInteger(node.rankId) ? node.rankId : parseRankFromNodeId(node.id)

  const srcSlice = normalizeSlice(data.src_slice, fallbackRank)
  const dstSlice = normalizeSlice(data.dst_slice, fallbackRank)
  const srcSlices = Array.isArray(data.src_slices)
    ? data.src_slices.map((item) => normalizeSlice(item, fallbackRank)).filter(Boolean)
    : []
  const dstSlices = Array.isArray(data.dst_slices)
    ? data.dst_slices.map((item) => normalizeSlice(item, fallbackRank)).filter(Boolean)
    : []
  const localSlice = normalizeSlice(data.local_slice, fallbackRank)
  const remoteRank = Number.isInteger(data.remote_rank) ? data.remote_rank : null
  const remoteSlice = normalizeSlice(data.remote_slice, remoteRank ?? fallbackRank)

  if (srcSlice && dstSlice) {
    pushTaskFlow(flows, srcSlice, dstSlice, node, {
      isReduce: taskType.includes('REDUCE'),
      reduceOp,
    })
  }
  if (srcSlices.length && dstSlice) {
    for (const item of srcSlices) {
      pushTaskFlow(flows, item, dstSlice, node, {
        isReduce: taskType.includes('REDUCE'),
        reduceOp,
      })
    }
  }
  if (srcSlice && dstSlices.length) {
    for (const item of dstSlices) {
      pushTaskFlow(flows, srcSlice, item, node, {
        isReduce: taskType.includes('REDUCE'),
        reduceOp,
      })
    }
  }
  if (remoteSlice && localSlice) {
    if (taskType.includes('READ')) {
      pushTaskFlow(flows, remoteSlice, localSlice, node, { isReduce: false })
    } else if (taskType.includes('WRITE')) {
      pushTaskFlow(flows, localSlice, remoteSlice, node, { isReduce: false })
    }
  }

  const dedup = new Map()
  for (const flow of flows) {
    const key = `${flow.srcRankId ?? 'x'}:${flow.srcBufferId}->${flow.dstRankId}:${flow.dstBufferId}:${flow.nodeId}`
    dedup.set(key, flow)
  }
  return Array.from(dedup.values())
}

function collectTaskFlowsFromTaskNode(taskNodeId, taskNodeLookup) {
  if (!taskNodeId || !taskNodeLookup) return []
  const directNode = getTaskNode(taskNodeLookup, taskNodeId)
  if (!directNode) return []

  const directFlows = flowsFromTaskNode(directNode)
  if (directFlows.length) return directFlows

  const descendantFlows = collectTaskFlowsInDirection(taskNodeId, taskNodeLookup, 'children')
  if (descendantFlows.length) return descendantFlows

  const parentFlows = collectTaskFlowsInDirection(taskNodeId, taskNodeLookup, 'parents')
  if (parentFlows.length) return parentFlows

  return []
}

function collectTaskFlowsInDirection(taskNodeId, taskNodeLookup, direction) {
  const relationKey = direction === 'children' ? 'children' : 'parents'
  const visited = new Set([taskNodeId])
  const queue = [{ nodeId: taskNodeId, depth: 0 }]
  const flows = []

  while (queue.length) {
    const { nodeId, depth } = queue.shift()
    const node = getTaskNode(taskNodeLookup, nodeId)
    if (!node) continue

    if (depth > 0) {
      const nodeFlows = flowsFromTaskNode(node)
      if (nodeFlows.length) {
        flows.push(...nodeFlows)
        continue
      }
    }

    if (depth >= TASK_PARENT_SEARCH_DEPTH) continue
    const relations = Array.isArray(node[relationKey]) ? node[relationKey] : []
    for (const relatedNodeId of relations) {
      if (typeof relatedNodeId !== 'string' || !relatedNodeId || visited.has(relatedNodeId)) continue
      visited.add(relatedNodeId)
      queue.push({ nodeId: relatedNodeId, depth: depth + 1 })
    }
  }

  return flows
}

function buildTaskOpsForStep(step, memoryTaskIds, taskNodeLookup) {
  if (!taskNodeLookup || !Array.isArray(memoryTaskIds) || !memoryTaskIds.length) return []

  const entries = []
  const dedup = new Map()

  for (const taskNodeId of memoryTaskIds) {
    const flows = collectTaskFlowsFromTaskNode(taskNodeId, taskNodeLookup)
    for (const flow of flows) {
      if (!Number.isInteger(flow.dstRankId) || !Number.isInteger(flow.dstBufferId)) continue
      const srcRankId = Number.isInteger(flow.srcRankId) ? flow.srcRankId : flow.dstRankId
      const key = [
        step,
        flow.nodeId ?? '',
        srcRankId,
        flow.srcBufferId ?? 'x',
        flow.srcAddr ?? 'x',
        flow.srcLen ?? 'x',
        flow.dstRankId,
        flow.dstBufferId,
        flow.dstAddr ?? 'x',
        flow.dstLen ?? 'x',
        flow.reduceOp ?? 'x',
      ].join('|')
      if (dedup.has(key)) continue
      const op = {
        step,
        rankId: flow.dstRankId,
        srcRankId,
        srcBufferId: flow.srcBufferId,
        srcOffset: flow.srcAddr,
        srcLen: flow.srcLen,
        srcMsId: flow.srcMsId,
        dstRankId: flow.dstRankId,
        dstBufferId: flow.dstBufferId,
        dstOffset: flow.dstAddr,
        dstLen: flow.dstLen,
        dstMsId: flow.dstMsId,
        isReduce: Boolean(flow.isReduce),
        reduceOp: flow.reduceOp,
        taskNodeId: flow.nodeId ?? null,
        taskType: flow.taskType ?? '',
      }
      dedup.set(key, op)
      entries.push(op)
    }
  }

  return entries.sort(
    (left, right) =>
      left.rankId - right.rankId ||
      left.dstBufferId - right.dstBufferId ||
      compareNullableNumber(left.dstOffset, right.dstOffset) ||
      compareNullableNumber(left.srcRankId, right.srcRankId) ||
      compareNullableNumber(left.srcBufferId, right.srcBufferId),
  )
}

function buildTaskSourcesByDestinationFromTaskOps(taskOps) {
  if (!Array.isArray(taskOps) || !taskOps.length) return new Map()
  const byDestination = new Map()

  for (const taskOp of taskOps) {
    if (!Number.isInteger(taskOp.dstRankId) || !Number.isInteger(taskOp.dstBufferId)) continue
    if (!Number.isInteger(taskOp.srcBufferId)) continue
    const destinationKey = `${taskOp.dstRankId}:${taskOp.dstBufferId}`
    const list = byDestination.get(destinationKey) ?? []
    list.push({
      rankId: Number.isInteger(taskOp.srcRankId) ? taskOp.srcRankId : taskOp.dstRankId,
      bufferId: taskOp.srcBufferId,
      addr: taskOp.srcOffset ?? null,
      len: taskOp.srcLen ?? null,
      msId: normalizeMsId(taskOp.srcMsId, {
        bufferId: taskOp.srcBufferId,
        offset: taskOp.srcOffset ?? null,
      }),
      step: null,
      nodeId: taskOp.taskNodeId ?? '',
    })
    byDestination.set(destinationKey, list)
  }

  for (const [destinationKey, list] of byDestination.entries()) {
    const dedup = new Map()
    for (const sourceRef of list) {
      const coarseKey = `${sourceRef.rankId ?? 'x'}:${sourceRef.bufferId ?? 'x'}`
      if (!dedup.has(coarseKey)) dedup.set(coarseKey, sourceRef)
    }
    byDestination.set(destinationKey, Array.from(dedup.values()).sort(compareSourceRefs))
  }

  return byDestination
}

function applyTaskSourcesToOps(ops, sourcesByDestination) {
  if (!Array.isArray(ops) || !ops.length || !(sourcesByDestination instanceof Map) || !sourcesByDestination.size) return
  for (const op of ops) {
    if (!Number.isInteger(op.rankId) || !Number.isInteger(op.bufferId)) continue
    const destinationKey = `${op.rankId}:${op.bufferId}`
    const sourceRefs = sourcesByDestination.get(destinationKey)
    if (!sourceRefs?.length) continue

    op.srcRefs = sourceRefs
    op.srcRanks = Array.from(
      new Set(sourceRefs.filter((sourceRef) => Number.isInteger(sourceRef.rankId)).map((sourceRef) => sourceRef.rankId)),
    ).sort((left, right) => left - right)
    op.srcNodes = Array.from(
      new Map(
        sourceRefs
          .filter((sourceRef) => typeof sourceRef.nodeId === 'string' && sourceRef.nodeId.length > 0)
          .map((sourceRef) => [`${sourceRef.nodeId}@${sourceRef.step ?? 'x'}`, { nodeId: sourceRef.nodeId, step: sourceRef.step }]),
      ).values(),
    )
    op.isTaskDerived = true
  }
}

function diffBufferStates(previousState, currentState, rankId, step) {
  const changedBufferTypes = new Set([...previousState.keys(), ...currentState.keys()])
  const orderedBufferTypes = Array.from(changedBufferTypes).sort((left, right) => left - right)

  let changedMask = 0
  const ops = []

  for (const bufferType of orderedBufferTypes) {
    const previousBuffer = previousState.get(bufferType) ?? null
    const currentBuffer = currentState.get(bufferType) ?? null

    let changeType = ''
    if (!previousBuffer && currentBuffer) changeType = 'added'
    else if (previousBuffer && !currentBuffer) changeType = 'removed'
    else if (previousBuffer && currentBuffer && previousBuffer.signature !== currentBuffer.signature) changeType = 'modified'
    else continue

    if (bufferType >= 0 && bufferType < 16) {
      changedMask |= 1 << bufferType
    }

    const renderedBuffer = currentBuffer ?? previousBuffer
    ops.push({
      step,
      rankId,
      bufferId: bufferType,
      changeType,
      totalSize: currentBuffer ? currentBuffer.totalSize : 0,
      previousTotalSize: previousBuffer ? previousBuffer.totalSize : 0,
      isReduce: Boolean(currentBuffer?.isReduce),
      srcRanks: currentBuffer?.srcRanks ?? [],
      srcNodes: currentBuffer?.srcNodes ?? [],
      srcRefs: currentBuffer?.srcRefs ?? [],
      semanticsCount: currentBuffer?.semanticsCount ?? 0,
      segments: currentBuffer?.segments ?? [],
      previousSegments: previousBuffer?.segments ?? [],
      signature: renderedBuffer?.signature ?? null,
    })
  }

  return { changedMask, ops }
}

export function buildMemviewIndex(rankSnapshotBundles, { totalSteps, taskNodeLookup = null } = {}) {
  const bundles = Array.isArray(rankSnapshotBundles) ? rankSnapshotBundles : []
  let resolvedTotalSteps = Number.isFinite(totalSteps) && totalSteps > 0 ? Math.floor(totalSteps) : 0

  for (const bundle of bundles) {
    for (const snapshot of bundle.snapshots ?? []) {
      if (Number.isFinite(snapshot.logicalStep)) {
        resolvedTotalSteps = Math.max(resolvedTotalSteps, snapshot.logicalStep + 1)
      }
    }
  }

  const memoryTaskIdsByStepRaw = new Array(resolvedTotalSteps)
  for (let i = 0; i < resolvedTotalSteps; i += 1) memoryTaskIdsByStepRaw[i] = []
  const stepByNodeId = new Map()

  const bufferPresence = new Map()
  const bufferChangedFlags = new Map()

  const bufferOpsByStepRaw = new Array(resolvedTotalSteps)
  for (let i = 0; i < resolvedTotalSteps; i += 1) bufferOpsByStepRaw[i] = []
  const layoutBuffersByStepRaw = new Array(resolvedTotalSteps)
  for (let i = 0; i < resolvedTotalSteps; i += 1) layoutBuffersByStepRaw[i] = []
  const taskOpsByStepRaw = new Array(resolvedTotalSteps)
  for (let i = 0; i < resolvedTotalSteps; i += 1) taskOpsByStepRaw[i] = []
  const bufferOpsByNode = new Map()

  for (const bundle of bundles) {
    const rankId = bundle.rankId
    if (!Number.isInteger(rankId)) continue

    let presence = bufferPresence.get(rankId)
    if (!presence) {
      presence = new Uint16Array(resolvedTotalSteps)
      bufferPresence.set(rankId, presence)
    }
    let changedFlags = bufferChangedFlags.get(rankId)
    if (!changedFlags) {
      changedFlags = new Uint16Array(resolvedTotalSteps)
      bufferChangedFlags.set(rankId, changedFlags)
    }

    const snapshotsByStep = new Map()
    const sortedSnapshots = [...(bundle.snapshots ?? [])].sort((left, right) => left.logicalStep - right.logicalStep)
    for (const snapshot of sortedSnapshots) {
      const step = snapshot.logicalStep
      if (!Number.isInteger(step) || step < 0 || step >= resolvedTotalSteps) continue
      snapshotsByStep.set(step, snapshot)
    }

    const snapshotStates = Array.from(snapshotsByStep.keys())
      .sort((left, right) => left - right)
      .map((step) => {
        const snapshot = snapshotsByStep.get(step)
        return {
          step,
          memoryTaskIds: Array.isArray(snapshot?.memoryTaskIds) ? snapshot.memoryTaskIds : [],
          state: buildBufferState(snapshot),
        }
      })
    const snapshotStateByStep = new Map(snapshotStates.map((snapshotState) => [snapshotState.step, snapshotState]))

    const emptyState = new Map()
    let cursor = 0
    let currentState = emptyState
    let previousState = emptyState

    for (let step = 0; step < resolvedTotalSteps; step += 1) {
      while (cursor < snapshotStates.length && snapshotStates[cursor].step <= step) {
        currentState = snapshotStates[cursor].state
        cursor += 1
      }

      presence[step] = buildPresenceMask(currentState)

      const diff = diffBufferStates(previousState, currentState, rankId, step)
      changedFlags[step] = diff.changedMask

      const layoutEntries = Array.from(currentState.values())
        .map((buffer) => ({
          step,
          rankId,
          bufferId: buffer.bufferType,
          totalSize: buffer.totalSize,
          isReduce: Boolean(buffer.isReduce),
          srcRanks: buffer.srcRanks ?? [],
          srcNodes: buffer.srcNodes ?? [],
          srcRefs: buffer.srcRefs ?? [],
          semanticsCount: buffer.semanticsCount ?? 0,
          segments: buffer.segments ?? [],
          previousSegments: previousState.get(buffer.bufferType)?.segments ?? [],
          signature: buffer.signature ?? null,
        }))
        .sort((left, right) => left.bufferId - right.bufferId)

      if (layoutEntries.length) {
        layoutBuffersByStepRaw[step].push(...layoutEntries)
      }

      const snapshotState = snapshotStateByStep.get(step)
      const stepTaskOps = buildTaskOpsForStep(step, snapshotState?.memoryTaskIds ?? [], taskNodeLookup)
      if (stepTaskOps.length) {
        taskOpsByStepRaw[step].push(...stepTaskOps)
        const taskSources = buildTaskSourcesByDestinationFromTaskOps(stepTaskOps)
        applyTaskSourcesToOps(diff.ops, taskSources)
        for (const taskOp of stepTaskOps) {
          if (typeof taskOp?.taskNodeId !== 'string' || !taskOp.taskNodeId) continue
          if (!stepByNodeId.has(taskOp.taskNodeId)) stepByNodeId.set(taskOp.taskNodeId, step)
        }
      }

      if (diff.ops.length) {
        bufferOpsByStepRaw[step].push(...diff.ops)
      }

      if (snapshotState) {
        for (const taskNodeId of snapshotState.memoryTaskIds) {
          if (typeof taskNodeId !== 'string' || !taskNodeId) continue
          memoryTaskIdsByStepRaw[step].push(taskNodeId)
          if (!stepByNodeId.has(taskNodeId)) stepByNodeId.set(taskNodeId, step)

          if (diff.ops.length) {
            const list = bufferOpsByNode.get(taskNodeId) ?? []
            for (const op of diff.ops) list.push(op)
            bufferOpsByNode.set(taskNodeId, list)
          }
        }
      }

      previousState = currentState
    }
  }

  const memoryTaskOffsets = new Uint32Array(resolvedTotalSteps + 1)
  let runningOffset = 0
  for (let i = 0; i < resolvedTotalSteps; i += 1) {
    memoryTaskOffsets[i] = runningOffset
    runningOffset += memoryTaskIdsByStepRaw[i].length
  }
  memoryTaskOffsets[resolvedTotalSteps] = runningOffset
  const memoryTaskIdsFlat = new Array(runningOffset)
  let writeCursor = 0
  for (let i = 0; i < resolvedTotalSteps; i += 1) {
    const stepNodes = memoryTaskIdsByStepRaw[i]
    for (let j = 0; j < stepNodes.length; j += 1) memoryTaskIdsFlat[writeCursor++] = stepNodes[j]
  }

  const bufferOpOffsets = new Uint32Array(resolvedTotalSteps + 1)
  let opCursor = 0
  for (let i = 0; i < resolvedTotalSteps; i += 1) {
    bufferOpOffsets[i] = opCursor
    opCursor += bufferOpsByStepRaw[i].length
  }
  bufferOpOffsets[resolvedTotalSteps] = opCursor
  const bufferOpsFlat = new Array(opCursor)
  let writeOp = 0
  for (let i = 0; i < resolvedTotalSteps; i += 1) {
    const list = bufferOpsByStepRaw[i]
    for (let j = 0; j < list.length; j += 1) bufferOpsFlat[writeOp++] = list[j]
  }

  const taskOpOffsets = new Uint32Array(resolvedTotalSteps + 1)
  let taskOpCursor = 0
  for (let i = 0; i < resolvedTotalSteps; i += 1) {
    taskOpOffsets[i] = taskOpCursor
    taskOpCursor += taskOpsByStepRaw[i].length
  }
  taskOpOffsets[resolvedTotalSteps] = taskOpCursor
  const taskOpsFlat = new Array(taskOpCursor)
  let writeTaskOp = 0
  for (let i = 0; i < resolvedTotalSteps; i += 1) {
    const list = taskOpsByStepRaw[i]
    for (let j = 0; j < list.length; j += 1) taskOpsFlat[writeTaskOp++] = list[j]
  }

  const layoutOffsets = new Uint32Array(resolvedTotalSteps + 1)
  let layoutCursor = 0
  for (let i = 0; i < resolvedTotalSteps; i += 1) {
    layoutOffsets[i] = layoutCursor
    layoutCursor += layoutBuffersByStepRaw[i].length
  }
  layoutOffsets[resolvedTotalSteps] = layoutCursor
  const layoutFlat = new Array(layoutCursor)
  let writeLayout = 0
  for (let i = 0; i < resolvedTotalSteps; i += 1) {
    const list = layoutBuffersByStepRaw[i]
    for (let j = 0; j < list.length; j += 1) layoutFlat[writeLayout++] = list[j]
  }

  return {
    totalSteps: resolvedTotalSteps,
    memoryTaskIds: { offsets: memoryTaskOffsets, items: memoryTaskIdsFlat },
    stepByNodeId,
    bufferPresence,
    bufferChangedFlags,
    bufferOps: { offsets: bufferOpOffsets, items: bufferOpsFlat },
    layoutBuffers: { offsets: layoutOffsets, items: layoutFlat },
    taskOps: { offsets: taskOpOffsets, items: taskOpsFlat },
    bufferOpsByNode,
  }
}

export function bufferOpsAtStep(indexes, step) {
  if (!indexes?.bufferOps) return []
  const { offsets, items } = indexes.bufferOps
  if (step < 0 || step >= offsets.length - 1) return []
  return items.slice(offsets[step], offsets[step + 1])
}

export function taskOpsAtStep(indexes, step) {
  if (!indexes?.taskOps) return []
  const { offsets, items } = indexes.taskOps
  if (step < 0 || step >= offsets.length - 1) return []
  return items.slice(offsets[step], offsets[step + 1])
}

export function layoutBuffersAtStep(indexes, step) {
  if (!indexes?.layoutBuffers) return []
  const { offsets, items } = indexes.layoutBuffers
  if (step < 0 || step >= offsets.length - 1) return []
  return items.slice(offsets[step], offsets[step + 1])
}

export function bezierTangentAt(cp1x, cp1y, cp2x, cp2y, endX, endY, startX, startY, t) {
  const mt = 1 - t
  const dx = 3 * mt * mt * (cp1x - startX) + 6 * mt * t * (cp2x - cp1x) + 3 * t * t * (endX - cp2x)
  const dy = 3 * mt * mt * (cp1y - startY) + 6 * mt * t * (cp2y - cp1y) + 3 * t * t * (endY - cp2y)
  const length = Math.hypot(dx, dy) || 1
  return { dx: dx / length, dy: dy / length }
}

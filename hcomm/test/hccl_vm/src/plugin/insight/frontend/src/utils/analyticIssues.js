/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

import { ElMessage } from 'element-plus'
import { fetchDecodedMsgpack } from './insightData'
import { DEFAULT_MEMVIEW_STAGE } from './memviewDag.js'

function ensureObject(value, fallback = {}) {
  return value && typeof value === 'object' && !Array.isArray(value) ? value : fallback
}

function ensureArray(value) {
  return Array.isArray(value) ? value : []
}

function toInteger(value, fallback = null) {
  return Number.isFinite(value) ? Number(value) : fallback
}

function uniqueIntegers(values) {
  return [...new Set(values.filter((value) => Number.isInteger(value)))].sort((left, right) => left - right)
}

function formatTitleCase(value) {
  return String(value)
    .split('_')
    .filter(Boolean)
    .map((part) => part.charAt(0) + part.slice(1).toLowerCase())
    .join(' ')
}

function formatStageLabel(stage) {
  if (!stage) return '--'
  const match = String(stage).match(/^step_(\d+)_(.+)$/)
  if (!match) return String(stage)
  return `step_${match[1]} / ${match[2]}`
}

function formatRankLabel(rankId) {
  return Number.isInteger(rankId) ? `Rank ${rankId}` : '--'
}

function formatValue(value) {
  if (value === null || value === undefined || value === '') return '--'
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  return JSON.stringify(value)
}

function normalizeTaskNode(rawTaskNode) {
  const taskNode = ensureObject(rawTaskNode, null)
  if (!taskNode) return null

  const task = ensureObject(taskNode.task)
  return {
    nodeId: typeof taskNode.node_id === 'string' ? taskNode.node_id : '',
    rankId: toInteger(taskNode.rank_id),
    queueId: toInteger(taskNode.queue_id),
    pos: toInteger(taskNode.pos),
    posInfo: typeof taskNode.pos_info === 'string' ? taskNode.pos_info : '',
    taskType:
      typeof task.task_type === 'string'
        ? task.task_type
        : typeof taskNode.task_type === 'string'
          ? taskNode.task_type
          : '',
    globalStep: toInteger(taskNode.global_step),
    localStep: toInteger(taskNode.local_step),
    parents: ensureArray(taskNode.parents),
    children: ensureArray(taskNode.children),
    raw: taskNode,
  }
}

function toList(value) {
  if (Array.isArray(value)) return value
  if (value === null || value === undefined) return []
  return [value]
}

function firstInteger(...values) {
  for (const value of values) {
    const integer = toInteger(value)
    if (Number.isInteger(integer)) return integer
  }
  return null
}

function firstString(...values) {
  for (const value of values) {
    if (typeof value === 'string' && value) return value
  }
  return ''
}

function normalizeRelatedNodeAnchor(rawNode) {
  if (typeof rawNode === 'string') {
    return {
      nodeId: rawNode,
      rankId: null,
      queueId: null,
      pos: null,
      nodePtr: '',
      taskType: '',
      raw: rawNode,
    }
  }

  const node = ensureObject(rawNode, null)
  if (!node) return null

  return {
    nodeId: firstString(node.node_id, node.nodeId),
    rankId: firstInteger(node.rank_id, node.rankId),
    queueId: firstInteger(node.queue_id, node.queueId),
    pos: firstInteger(node.pos),
    nodePtr: firstString(node.node_ptr, node.nodePtr),
    taskType: firstString(node.task_type, node.taskType),
    raw: node,
  }
}

function normalizeRelatedQueueAnchor(rawQueue) {
  const queue = ensureObject(rawQueue, null)
  if (!queue) return null

  return {
    rankId: firstInteger(queue.rank_id, queue.rankId),
    queueId: firstInteger(queue.queue_id, queue.queueId),
    pos: firstInteger(queue.pos),
    raw: queue,
  }
}

function normalizeRelatedStepAnchor(rawStep) {
  if (Number.isInteger(rawStep)) {
    return {
      snapshotStep: Number(rawStep),
      eventId: '',
      slicePair: null,
      raw: rawStep,
    }
  }

  const step = ensureObject(rawStep, null)
  if (!step) return null

  return {
    snapshotStep: firstInteger(step.snapshot_step, step.snapshotStep, step.step),
    eventId: firstString(step.event_id, step.eventId),
    slicePair: step.slice_pair ?? step.slicePair ?? null,
    raw: step,
  }
}

function formatRelatedNodeValue(anchor, { includeCoords = false } = {}) {
  if (!anchor) return null

  const parts = []
  if (anchor.nodeId) parts.push(anchor.nodeId)

  if (includeCoords) {
    const coords = []
    if (Number.isInteger(anchor.rankId)) coords.push(`rank_id=${anchor.rankId}`)
    if (Number.isInteger(anchor.queueId)) coords.push(`queue_id=${anchor.queueId}`)
    if (Number.isInteger(anchor.pos)) coords.push(`pos=${anchor.pos}`)
    if (anchor.nodePtr) coords.push(`node_ptr=${anchor.nodePtr}`)
    if (coords.length) {
      parts.push(coords.join(' '))
    }
  }

  if (!parts.length) {
    const coords = []
    if (Number.isInteger(anchor.rankId)) coords.push(`rank_id=${anchor.rankId}`)
    if (Number.isInteger(anchor.queueId)) coords.push(`queue_id=${anchor.queueId}`)
    if (Number.isInteger(anchor.pos)) coords.push(`pos=${anchor.pos}`)
    if (anchor.nodePtr) coords.push(`node_ptr=${anchor.nodePtr}`)
    if (!coords.length) return null
    return coords.join(' ')
  }

  return parts.join(includeCoords ? ' | ' : '')
}

function formatRelatedQueueValue(anchor) {
  if (!anchor) return null
  const parts = []
  if (Number.isInteger(anchor.rankId)) parts.push(`rank_id=${anchor.rankId}`)
  if (Number.isInteger(anchor.queueId)) parts.push(`queue_id=${anchor.queueId}`)
  if (Number.isInteger(anchor.pos)) parts.push(`pos=${anchor.pos}`)
  return parts.length ? parts.join(' ') : null
}

function formatRelatedStepValue(anchor) {
  if (!anchor) return null
  const parts = []
  if (Number.isInteger(anchor.snapshotStep)) parts.push(`snapshot_step=${anchor.snapshotStep}`)
  if (anchor.eventId) parts.push(`event_id=${anchor.eventId}`)
  if (anchor.slicePair) parts.push(`slice_pair=${formatValue(anchor.slicePair)}`)
  return parts.length ? parts.join(' ') : null
}

function appendField(target, label, value) {
  if (value === null || value === undefined) return
  const text = String(value).trim()
  if (!text || text === '--') return
  target.push({ label, value: text })
}

function buildIssueListFieldRows(issue) {
  const rows = []
  const relatedNode = issue.relatedNodeAnchors?.[0] ?? null
  const relatedQueue = issue.relatedQueueAnchors?.[0] ?? null
  const relatedStep = issue.relatedStepAnchors?.[0] ?? null

  appendField(rows, 'related_node', issue.relatedNodeText || formatRelatedNodeValue(relatedNode))
  appendField(rows, 'related_rank', issue.relatedRankText)
  appendField(rows, 'related_queue', issue.relatedQueueText || formatRelatedQueueValue(relatedQueue))
  appendField(rows, 'related_step', issue.relatedStepText || formatRelatedStepValue(relatedStep))

  return {
    primaryRow: rows,
    secondaryRow: [],
  }
}

function buildRelatedRankIds(detail, relatedNodeAnchors, relatedQueueAnchors) {
  return uniqueIntegers([
    ...toList(detail.related_rank).map((item) => toInteger(item)),
    ...toList(detail.rank_id).map((item) => toInteger(item)),
    ...toList(detail.task_rank).map((item) => toInteger(item)),
    ...toList(detail.peer_rank).map((item) => toInteger(item)),
    ...relatedNodeAnchors.map((anchor) => toInteger(anchor?.rankId)),
    ...relatedQueueAnchors.map((anchor) => toInteger(anchor?.rankId)),
  ])
}

function buildDagTarget(detail, taskNode, relatedNodeAnchors, relatedQueueAnchors) {
  const primaryNode = relatedNodeAnchors[0] ?? taskNode ?? null
  const primaryQueue = relatedQueueAnchors[0] ?? null
  const taskType =
    firstString(primaryNode?.taskType, taskNode?.taskType, detail.expected_task_type, detail.first_task_type, detail.op_type)

  if (primaryNode?.nodeId) {
    return {
      stageName: DEFAULT_MEMVIEW_STAGE,
      nodeId: primaryNode.nodeId,
      rankId: Number.isInteger(primaryNode.rankId) ? primaryNode.rankId : firstInteger(detail.rank_id, detail.task_rank, detail.peer_rank),
      queueId: Number.isInteger(primaryNode.queueId) ? primaryNode.queueId : firstInteger(detail.queue_id, primaryQueue?.queueId),
      slotIndex: Number.isInteger(primaryNode.pos) ? primaryNode.pos : firstInteger(detail.task_id, primaryQueue?.pos),
      taskType,
    }
  }

  if (primaryQueue?.queueId !== null && primaryQueue?.queueId !== undefined) {
    return {
      stageName: DEFAULT_MEMVIEW_STAGE,
      nodeId: '',
      rankId: Number.isInteger(primaryQueue.rankId) ? primaryQueue.rankId : firstInteger(detail.rank_id, detail.task_rank, detail.peer_rank),
      queueId: primaryQueue.queueId,
      slotIndex: Number.isInteger(primaryQueue.pos) ? primaryQueue.pos : firstInteger(detail.task_id),
      taskType,
    }
  }

  const primaryRankId = firstInteger(detail.rank_id, detail.task_rank, detail.peer_rank)
  const queueId = firstInteger(detail.queue_id, detail.task_id)

  if (Number.isInteger(primaryRankId) && Number.isInteger(queueId)) {
    return {
      stageName: DEFAULT_MEMVIEW_STAGE,
      nodeId: '',
      rankId: primaryRankId,
      queueId,
      slotIndex: Number.isInteger(detail.task_id) ? detail.task_id : null,
      taskType,
    }
  }

  return null
}

function buildOverviewRows(issue) {
  const rows = [
    ['issue_id', issue.issueId ?? '--'],
    ['severity', issue.severity],
    ['stage', issue.stage || '--'],
  ]

  return rows.filter(([, value]) => value !== '--' && value !== '')
}

function buildDetailRows(detail) {
  return Object.entries(detail)
    .filter(
      ([key, value]) =>
        key !== 'task_node' &&
        key !== 'related_node' &&
        key !== 'related_step' &&
        key !== 'related_rank' &&
        key !== 'related_queue' &&
        key !== 'ret_code' &&
        !Array.isArray(value) &&
        (typeof value !== 'object' || value === null),
    )
    .map(([key, value]) => [key, formatValue(value)])
}

function buildSliceCards(detail) {
  return [
    ['slice_a', ensureObject(detail.slice_a, null)],
    ['slice_b', ensureObject(detail.slice_b, null)],
  ]
    .filter(([, slice]) => slice)
    .map(([label, slice]) => ({
      label,
      fields: [
        ['buffer_type', formatValue(slice.buffer_type)],
        ['offset', Number.isFinite(slice.offset) ? String(slice.offset) : '--'],
        ['size', Number.isFinite(slice.size) ? String(slice.size) : '--'],
      ],
    }))
}

export function normalizeValidationIssue(rawIssue, index = 0) {
  const issue = ensureObject(rawIssue)
  const detail = ensureObject(issue.detail)
  const taskNode = normalizeTaskNode(detail.task_node)
  const code = typeof issue.code === 'string' ? issue.code : 'UNKNOWN_ISSUE'
  const severity = typeof issue.severity === 'string' ? issue.severity : 'UNKNOWN'
  const severityLabel = formatTitleCase(severity)
  const relatedNodeAnchors = toList(detail.related_node)
    .map((item) => normalizeRelatedNodeAnchor(item))
    .filter(Boolean)
  if (!relatedNodeAnchors.length && taskNode) {
    relatedNodeAnchors.push(normalizeRelatedNodeAnchor(taskNode))
  }

  const relatedQueueAnchors = toList(detail.related_queue)
    .map((item) => normalizeRelatedQueueAnchor(item))
    .filter(Boolean)

  const relatedStepAnchors = toList(detail.related_step)
    .map((item) => normalizeRelatedStepAnchor(item))
    .filter(Boolean)

  const relatedRankIds = buildRelatedRankIds(detail, relatedNodeAnchors, relatedQueueAnchors)
  const primaryRankId =
    relatedRankIds[0] ??
    relatedNodeAnchors[0]?.rankId ??
    relatedQueueAnchors[0]?.rankId ??
    toInteger(detail.rank_id ?? detail.task_rank ?? detail.peer_rank)
  const primaryNode = relatedNodeAnchors[0] ?? null
  const primaryQueue = relatedQueueAnchors[0] ?? null
  const primaryStep = relatedStepAnchors[0] ?? null
  const queueId =
    primaryQueue?.queueId ??
    primaryNode?.queueId ??
    toInteger(detail.queue_id)
  const taskType = firstString(primaryNode?.taskType, taskNode?.taskType, detail.first_task_type, detail.expected_task_type, detail.op_type)
  const dagTarget = buildDagTarget(detail, taskNode, relatedNodeAnchors, relatedQueueAnchors)
  const relatedNodeText = relatedNodeAnchors.length
    ? relatedNodeAnchors.map((anchor) => formatRelatedNodeValue(anchor, { includeCoords: false })).filter(Boolean).join(', ')
    : '--'
  const relatedQueueText = relatedQueueAnchors.length
    ? relatedQueueAnchors.map((anchor) => formatRelatedQueueValue(anchor)).filter(Boolean).join(', ')
    : '--'
  const relatedStepText = relatedStepAnchors.length
    ? relatedStepAnchors.map((anchor) => formatRelatedStepValue(anchor)).filter(Boolean).join(', ')
    : '--'
  const normalized = {
    issueId: toInteger(issue.issue_id, index + 1),
    code,
    title: code,
    severity,
    severityLabel,
    severityType: severity === 'ERROR' ? 'danger' : severity === 'WARNING' ? 'warning' : 'info',
    stage: typeof issue.stage === 'string' ? issue.stage : '',
    stageLabel: formatStageLabel(issue.stage),
    retCode: toInteger(detail.ret_code),
    primaryRankId,
    relatedRankIds,
    relatedRankText: relatedRankIds.length ? relatedRankIds.map((rankId) => `Rank ${rankId}`).join(', ') : '--',
    relatedNodeAnchors,
    relatedQueueAnchors,
    relatedStepAnchors,
    relatedNodeText,
    relatedQueueText,
    relatedStepText,
    queueId,
    nodeDisplayId: primaryNode?.nodeId || taskNode?.nodeId || '--',
    taskType,
    taskNode,
    primaryRelatedNode: primaryNode,
    primaryRelatedQueue: primaryQueue,
    primaryRelatedStep: primaryStep,
    dagTarget,
    memTarget: {
      rankIds: relatedRankIds,
      stageName: DEFAULT_MEMVIEW_STAGE,
    },
    anchorRows: [],
    listFieldRows: {
      primaryRow: [],
      secondaryRow: [],
    },
    overviewRows: [],
    detailRows: buildDetailRows(detail),
    sliceCards: buildSliceCards(detail),
    rawDetail: detail,
  }

  normalized.anchorRows = [
    ['related_node', normalized.relatedNodeText || '--'],
    ['related_rank', normalized.relatedRankText],
    ['related_queue', normalized.relatedQueueText || '--'],
    ['related_step', normalized.relatedStepText || '--'],
  ].filter(([, value]) => value !== '--' && value !== '')
  normalized.listFieldRows = buildIssueListFieldRows(normalized)
  normalized.overviewRows = buildOverviewRows(normalized)
  return normalized
}

export function parseValidationIssues(rawPayload) {
  const payload = ensureObject(rawPayload)
  const issues = ensureArray(payload.issues).map((issue, index) => normalizeValidationIssue(issue, index))

  return {
    issueCount: Number.isFinite(payload.issue_count) ? Number(payload.issue_count) : issues.length,
    issues,
    type: typeof payload.type === 'string' ? payload.type : 'validation_issues',
  }
}

export async function fetchValidationIssues(datasetName, { onError } = {}) {
  if (!datasetName) {
    return parseValidationIssues({ issues: [] })
  }

  try {
    const payload = await fetchDecodedMsgpack({
      dataset: datasetName,
      kind: 'validation',
      file: 'issues.msgpack',
    })
    return parseValidationIssues(payload)
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : '无法读取 issues.msgpack。'
    ElMessage.error(`读取 issues.msgpack 失败：${errorMessage}`)
    if (typeof onError === 'function') {
      onError(error)
    }
    return parseValidationIssues({ issues: [] })
  }
}

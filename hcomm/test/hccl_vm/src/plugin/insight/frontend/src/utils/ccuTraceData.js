/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * CCU Trace 数据加载与索引构建工具
 *
 * 负责加载 trace JSON 文件并构建前端回放所需的索引结构。
 */

/**
 * 从 File 对象加载 trace JSON 数据
 * @param {File} file
 * @returns {Promise<CcuTraceData>}
 */
export async function loadTraceFromFile(file) {
  const text = await file.text()
  const json = JSON.parse(text)
  return buildTraceIndex(json)
}

/**
 * 从 URL 加载 trace JSON 数据
 * @param {string} url
 * @returns {Promise<CcuTraceData>}
 */
export async function loadTraceFromUrl(url) {
  const resp = await fetch(url)
  if (!resp.ok) throw new Error(`Failed to fetch trace: ${resp.status}`)
  const json = await resp.json()
  return buildTraceIndex(json)
}

/**
 * 从指令空间的指令列表中提取 Loop 循环体范围。
 * 解析 instrDescribe 字符串中 "Loop From startInstrId[X] to endInstrId[Y]" 的格式。
 * 返回按 loopIndex 排序的 loop 范围列表，每个 loop 分配一个稳定的 loopIndex。
 *
 * @param {Array} instructions - 指令空间中的指令列表
 * @returns {Array<{startId: number, endId: number, loopInstrId: number, loopIndex: number}>}
 */
export function extractLoopRanges(instructions) {
  const ranges = []
  // Loop 指令匹配: "Loop From startInstrId[X] to endInstrId[Y] ..."
  const loopRegex = /^Loop From startInstrId\[(\d+)\] to endInstrId\[(\d+)\]/

  for (const instr of instructions || []) {
    const match = instr.instrDescribe?.match(loopRegex)
    if (match) {
      ranges.push({
        startId: parseInt(match[1], 10),
        endId: parseInt(match[2], 10),
        loopInstrId: instr.instrId,
      })
    }
  }

  // 按 startId 排序，然后分配 loopIndex
  ranges.sort((a, b) => a.startId - b.startId || a.endId - b.endId)
  for (let i = 0; i < ranges.length; i++) {
    ranges[i].loopIndex = i
  }

  return ranges
}

/**
 * 预定义的 Loop 高亮颜色方案（背景色 + 左边框色）。
 * 选用柔和的低饱和度色，与回放蓝色高亮和断点红色高亮明显区分。
 */
export const LOOP_COLORS = [
  { bg: 'rgba(46, 204, 113, 0.10)', border: '#2ecc71' },  // 翠绿
  { bg: 'rgba(155, 89, 182, 0.10)', border: '#9b59b6' },  // 紫色
  { bg: 'rgba(241, 196, 15, 0.10)', border: '#f1c40f' },  // 金黄
  { bg: 'rgba(26, 188, 156, 0.10)', border: '#1abc9c' },  // 青色
  { bg: 'rgba(230, 126, 34, 0.10)', border: '#e67e22' },  // 橙色
  { bg: 'rgba(52, 152, 219, 0.10)', border: '#3498db' },  // 天蓝
  { bg: 'rgba(231, 76, 60, 0.08)', border: '#e74c3c' },   // 红色（备用）
  { bg: 'rgba(149, 165, 166, 0.10)', border: '#95a5a6' },  // 灰蓝
]

/**
 * 获取指定 instrId 所属的 loop 颜色。如果不在任何 loop 中，返回 null。
 * @param {Array} loopRanges - extractLoopRanges 的输出
 * @param {number} instrId
 * @returns {{bg: string, border: string, loopIndex: number}|null}
 */
export function getLoopColorForInstr(loopRanges, instrId) {
  if (!loopRanges || loopRanges.length === 0) return null

  for (const range of loopRanges) {
    if (instrId >= range.startId && instrId <= range.endId) {
      return {
        ...LOOP_COLORS[range.loopIndex % LOOP_COLORS.length],
        loopIndex: range.loopIndex,
      }
    }
  }
  return null
}

/**
 * 构建 trace 数据的索引结构
 * @param {object} raw - 原始 JSON 数据
 * @returns {CcuTraceData}
 */
export function buildTraceIndex(raw) {
  // 指令空间索引: key = "rankId:dieId" → instructions[]
  const instrSpaceMap = new Map()
  // Loop 循环体范围索引: key = "rankId:dieId" → loopRanges[]
  const loopRangesMap = new Map()

  for (const space of raw.instrSpaces || []) {
    const key = ccuKey(space.rankId, space.dieId)
    const instrs = space.instructions || []
    instrSpaceMap.set(key, instrs)
    loopRangesMap.set(key, extractLoopRanges(instrs))
  }

  // Channel 空间索引: key = "rankId:dieId" → channels[]
  const channelSpaceMap = new Map()
  for (const space of raw.channelSpaces || []) {
    const key = ccuKey(space.rankId, space.dieId)
    channelSpaceMap.set(key, space.channels || [])
  }

  // CCU 注册表索引: key = "rankId:dieId" → registry entry
  const ccuRegistryMap = new Map()
  for (const ccu of raw.ccuRegistry || []) {
    ccuRegistryMap.set(ccuKey(ccu.rankId, ccu.dieId), ccu)
  }

  // SQE 任务注册表索引: sqeTaskId → sqe task
  const sqeTaskMap = new Map()
  for (const sqe of raw.sqeTaskRegistry || []) {
    sqeTaskMap.set(sqe.sqeTaskId, sqe)
  }

  // 按 CCU 分组的 trace entries: key = "rankId:dieId" → entries[]
  const entriesByCcu = new Map()
  // instrId → 在该 CCU 内的 entry 索引列表
  const entriesByInstr = new Map()

  for (const entry of raw.globalEntries || []) {
    const key = ccuKey(entry.rankId, entry.dieId)
    if (!entriesByCcu.has(key)) entriesByCcu.set(key, [])
    entriesByCcu.get(key).push(entry)

    const instrKey = `${key}:${entry.instrId}`
    if (!entriesByInstr.has(instrKey)) entriesByInstr.set(instrKey, [])
    entriesByInstr.get(instrKey).push(entry)
  }

  // 提取所有 CCU 列表
  const ccuList = []
  for (const ccu of raw.ccuRegistry || []) {
    ccuList.push({ rankId: ccu.rankId, dieId: ccu.dieId })
  }

  return {
    raw,
    runMetadata: raw.runMetadata || {},
    ccuList,
    instrSpaceMap,
    loopRangesMap,
    channelSpaceMap,
    ccuRegistryMap,
    sqeTaskMap,
    entriesByCcu,
    entriesByInstr,
    globalEntries: raw.globalEntries || [],
    totalEntries: (raw.globalEntries || []).length,
  }
}

/**
 * 获取指定 CCU 的指令空间
 * @param {CcuTraceData} data
 * @param {number} rankId
 * @param {number} dieId
 * @returns {Array}
 */
export function getInstrSpace(data, rankId, dieId) {
  return data.instrSpaceMap.get(ccuKey(rankId, dieId)) || []
}

/**
 * 获取指定 CCU 的 Loop 循环体范围列表
 * @param {CcuTraceData} data
 * @param {number} rankId
 * @param {number} dieId
 * @returns {Array<{startId: number, endId: number, loopInstrId: number, loopIndex: number}>}
 */
export function getLoopRanges(data, rankId, dieId) {
  return data.loopRangesMap?.get(ccuKey(rankId, dieId)) || []
}

/**
 * 获取指定 CCU 的 Channel 空间
 * @param {CcuTraceData} data
 * @param {number} rankId
 * @param {number} dieId
 * @returns {Array}
 */
export function getChannelSpace(data, rankId, dieId) {
  return data.channelSpaceMap.get(ccuKey(rankId, dieId)) || []
}

/**
 * 获取指定 CCU 的 trace entries
 * @param {CcuTraceData} data
 * @param {number} rankId
 * @param {number} dieId
 * @returns {Array}
 */
export function getEntriesForCcu(data, rankId, dieId) {
  return data.entriesByCcu.get(ccuKey(rankId, dieId)) || []
}

/**
 * 获取指定指令的所有 trace entries
 * @param {CcuTraceData} data
 * @param {number} rankId
 * @param {number} dieId
 * @param {number} instrId
 * @returns {Array}
 */
export function getEntriesForInstr(data, rankId, dieId, instrId) {
  return data.entriesByInstr.get(`${ccuKey(rankId, dieId)}:${instrId}`) || []
}

/**
 * 获取 SQE 任务信息
 * @param {CcuTraceData} data
 * @param {number} sqeTaskId
 * @returns {object|null}
 */
export function getSqeTask(data, sqeTaskId) {
  return data.sqeTaskMap.get(sqeTaskId) || null
}

/**
 * 构建当前 CCU 的累积资源状态（应用所有 delta 到指定 entry 为止）
 * @param {CcuTraceData} data
 * @param {number} rankId
 * @param {number} dieId
 * @param {number} upToSeqId - 全局序号（包含）
 * @returns {object} { xn: Map, gsa: Map, cke: Map }
 */
export function buildCumulativeState(data, rankId, dieId, upToSeqId) {
  const xn = new Map()
  const gsa = new Map()
  const cke = new Map()

  const entries = getEntriesForCcu(data, rankId, dieId)
  for (const entry of entries) {
    if (entry.globalSeqId > upToSeqId) break

    const delta = entry.resourceDelta || {}
    for (const ch of delta.xnChanges || []) {
      xn.set(ch.id, ch.valueAfter)
    }
    for (const ch of delta.gsaChanges || []) {
      gsa.set(ch.id, ch.valueAfter)
    }
    for (const ch of delta.ckeChanges || []) {
      cke.set(ch.id, ch.valueAfter)
    }
  }

  return { xn, gsa, cke }
}

/**
 * 格式化 hex 值
 */
export function formatHex(value) {
  if (value == null) return '--'
  if (typeof value === 'string') {
    if (value.startsWith('0x') || value.startsWith('0X')) return value
    const num = parseInt(value, 10)
    return isNaN(num) ? value : '0x' + num.toString(16)
  }
  return '0x' + value.toString(16)
}

/**
 * 格式化数值
 */
export function formatValue(value) {
  if (value == null) return '--'
  if (typeof value === 'string') return value
  return String(value)
}

function ccuKey(rankId, dieId) {
  return `${rankId}:${dieId}`
}

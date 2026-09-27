/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

let workerSingleton = null
let nextRequestId = 0
const pending = new Map()

function ensureWorker() {
  if (workerSingleton) return workerSingleton
  workerSingleton = new Worker(new URL('./insightDataWorker.js', import.meta.url), { type: 'module' })
  workerSingleton.addEventListener('message', (event) => {
    const { id, ok, result, error } = event.data ?? {}
    const entry = pending.get(id)
    if (!entry) return
    pending.delete(id)
    if (ok) entry.resolve(result)
    else entry.reject(new Error(error || 'Worker request failed'))
  })
  workerSingleton.addEventListener('error', () => {
    pending.forEach(({ reject }) => reject(new Error('Insight worker crashed')))
    pending.clear()
  })
  return workerSingleton
}

export function postWorker(kind, payload, transferables = []) {
  const worker = ensureWorker()
  const id = ++nextRequestId
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject })
    worker.postMessage({ id, kind, payload }, transferables)
  })
}

export function rehydrateMemviewIndex(serialized) {
  if (!serialized) return null
  return {
    totalSteps: serialized.totalSteps,
    memoryTaskIds: serialized.memoryTaskIds,
    stepByNodeId: new Map(serialized.stepByNodeId),
    bufferPresence: new Map(serialized.bufferPresence),
    bufferChangedFlags: new Map(serialized.bufferChangedFlags),
    bufferOps: serialized.bufferOps,
    taskOps: serialized.taskOps ?? null,
    bufferOpsByNode: new Map(serialized.bufferOpsByNode),
  }
}

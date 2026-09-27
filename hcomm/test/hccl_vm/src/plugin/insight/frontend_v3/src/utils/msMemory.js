/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

export const MS_BUFFER_ID = 8
export const MS_SLICE_BYTES = 4 * 1024

function toSafeNonNegativeInteger(value) {
  if (typeof value === 'number') {
    return Number.isSafeInteger(value) && value >= 0 ? value : null
  }

  if (typeof value === 'bigint') {
    if (value < 0n || value > BigInt(Number.MAX_SAFE_INTEGER)) return null
    return Number(value)
  }

  if (typeof value === 'string') {
    const normalized = value.trim()
    if (!/^\d+$/.test(normalized)) return null
    const parsed = Number(normalized)
    return Number.isSafeInteger(parsed) ? parsed : null
  }

  return null
}

// Matches both checkerV3 graph labels and memview buffer labels for MS-backed slices.
export function isMsBufferTypeLabel(bufferType) {
  const normalized = typeof bufferType === 'string' ? bufferType.trim().toUpperCase() : ''
  return normalized === 'MS' || normalized === 'MS_CCU'
}

// Matches the normalized memview buffer id used for MS.
export function isMsBufferId(bufferId) {
  return Number.isInteger(bufferId) && Number(bufferId) === MS_BUFFER_ID
}

// Resolves an MS register id from explicit dump metadata first, then from the synthetic offset fallback.
export function normalizeMsId(msId, { bufferType = '', bufferId = null, offset = null } = {}) {
  const explicitMsId = toSafeNonNegativeInteger(msId)
  if (Number.isInteger(explicitMsId)) {
    return explicitMsId
  }

  if (!isMsBufferTypeLabel(bufferType) && !isMsBufferId(bufferId)) {
    return null
  }

  const numericOffset = toSafeNonNegativeInteger(offset)
  if (!Number.isInteger(numericOffset) || numericOffset % MS_SLICE_BYTES !== 0) {
    return null
  }

  const derivedMsId = numericOffset / MS_SLICE_BYTES
  return Number.isSafeInteger(derivedMsId) ? derivedMsId : null
}

// Chooses the address field that should be shown in the UI for one memory slice.
export function resolveMemoryAddressDisplay({ bufferType = '', bufferId = null, offset = null, msId = null } = {}) {
  const resolvedMsId = normalizeMsId(msId, { bufferType, bufferId, offset })
  if (Number.isInteger(resolvedMsId)) {
    return {
      label: 'id',
      value: String(resolvedMsId),
      msId: resolvedMsId,
    }
  }

  return {
    label: 'offset',
    value: Number.isFinite(offset) ? String(Number(offset)) : '--',
    msId: null,
  }
}

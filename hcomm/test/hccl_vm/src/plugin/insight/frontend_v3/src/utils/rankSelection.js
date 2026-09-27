/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

function resolveRankCount(detail) {
  const ranks = detail?.memory?.map((item) => item.rank).filter((item) => Number.isInteger(item)) ?? []
  if (ranks.length) {
    return ranks.length
  }

  const rankSize = detail?.opParam?.rank_size
  if (Number.isInteger(rankSize) && rankSize > 0) {
    return rankSize
  }

  return 0
}

export const DEFAULT_SELECTED_RANK_LIMIT = 8

export function buildDefaultSelectedRankKeys(detail, maxDefaultRanks = DEFAULT_SELECTED_RANK_LIMIT) {
  const maxRanks = Math.max(0, Math.floor(Number(maxDefaultRanks) || 0))
  const selectedCount = Math.min(resolveRankCount(detail), maxRanks)
  return Array.from({ length: selectedCount }, (_, index) => `rank-${index}`)
}

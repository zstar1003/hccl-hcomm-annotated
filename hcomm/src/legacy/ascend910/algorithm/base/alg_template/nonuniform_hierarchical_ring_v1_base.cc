/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "nonuniform_hierarchical_ring_v1_base.h"

namespace hccl {

RingInfo::RingInfo(u32 rankSize) : rankSize_(rankSize)
{
    sqrtRankSize_ = static_cast<u32>(std::sqrt(rankSize));

    u32 left = sqrtRankSize_;
    u32 right = sqrtRankSize_;

    while (left > 0 && right < rankSize) {
        if (left * right < rankSize) {
            right++;
        } else if (left * right > rankSize) {
            left--;
        } else {
            break;
        }
    }

    if ((right - left) < threshold) {
        colSize_ = left;
        rowSize_ = right;
        extraRowSize_ = 0;
        extraColSize_ = 0;
        rankOffset_ = colSize_ * rowSize_;
    } else {
        // 保证除了最后一列之外的其他列size相等
        u32 extraSize = rankSize - sqrtRankSize_ * sqrtRankSize_;
        extraRowSize_ = (extraSize >= sqrtRankSize_) ? sqrtRankSize_ : 0;
        extraColSize_ = extraSize - extraRowSize_;

        colSize_ = sqrtRankSize_ + ((extraRowSize_ == 0) ? 0 : 1);
        rowSize_ = sqrtRankSize_;

        // 当rank < rankOffset_时，rank所处的行size为sqrtRankSize_ + 1
        // 当rank >= rankOffset_时，rank所处的行size为sqrtRankSize_
        if (extraColSize_ == 0) {
            rankOffset_ = colSize_ * rowSize_;
        } else {
            rankOffset_ = (sqrtRankSize_ + 1) * extraColSize_;
        }
    }
}

RingInfo::~RingInfo() {}

u32 RingInfo::GetRankSize() const { return rankSize_; }

u32 RingInfo::GetRankOffset() const { return rankOffset_; }

u32 RingInfo::GetSqrtRankSize() const { return sqrtRankSize_; }

u32 RingInfo::GetRowSize() const { return rowSize_; }

u32 RingInfo::GetColSize() const { return colSize_; }

u32 RingInfo::GetVIndex(u32 rank) const
{
    if (rank < rankOffset_) {
        return rank / (rowSize_ + ((extraColSize_ == 0) ? 0 : 1));
    } else {
        return extraColSize_ + (rank - rankOffset_) / sqrtRankSize_;
    }
}

u32 RingInfo::GetHIndex(u32 rank) const
{
    if (rank < rankOffset_) {
        return rank % (rowSize_ + ((extraColSize_ == 0) ? 0 : 1));
    } else {
        return (rank - rankOffset_) % sqrtRankSize_;
    }
}

u32 RingInfo::GetVSizeByHIndex(u32 hIndex) const
{
    if (rankOffset_ == rankSize_) {
        return colSize_;
    } else if (hIndex == sqrtRankSize_) {
        return extraColSize_;
    } else {
        return sqrtRankSize_ + static_cast<u32>(hIndex < extraRowSize_);
    }
}

u32 RingInfo::GetVSizeByRank(u32 rank) const { return GetVSizeByHIndex(GetHIndex(rank)); }

u32 RingInfo::GetHSizeByVIndex(u32 vIndex) const
{
    if (rankOffset_ == rankSize_) {
        return rowSize_;
    } else {
        return sqrtRankSize_ + static_cast<u32>(vIndex < extraColSize_);
    }
}

u32 RingInfo::GetHSizeByRank(u32 rank) const { return GetHSizeByVIndex(GetVIndex(rank)); }

u32 RingInfo::GetRank(u32 vIndex, u32 hIndex) const
{
    if (rankOffset_ == rankSize_) {
        return vIndex * rowSize_ + hIndex;
    } else if (vIndex < extraColSize_) {
        return (sqrtRankSize_ + 1) * vIndex + hIndex;
    } else {
        return rankOffset_ + sqrtRankSize_ * (vIndex - extraColSize_) + hIndex;
    }
}

NHRV1Base::NHRV1Base(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

NHRV1Base::~NHRV1Base() {}

RingInfo NHRV1Base::GetRingInfo(u32 rankSize) { return RingInfo(rankSize); }

} // namespace hccl

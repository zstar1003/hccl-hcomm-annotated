/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NONUNIFORM_HIERARCHICAL_RING_V1_BASE_PUB_H
#define NONUNIFORM_HIERARCHICAL_RING_V1_BASE_PUB_H

#include <cmath>
#include "alg_template_base_pub.h"

namespace hccl {

class RingInfo {
public:
    explicit RingInfo(u32 rankSize);
    ~RingInfo();

    u32 GetRankSize() const;
    u32 GetRankOffset() const;
    u32 GetSqrtRankSize() const;
    u32 GetRowSize() const;
    u32 GetColSize() const;
    u32 GetVIndex(u32 rank) const;
    u32 GetHIndex(u32 rank) const;
    u32 GetVSizeByRank(u32 rank) const;
    u32 GetVSizeByHIndex(u32 hIndex) const;
    u32 GetHSizeByRank(u32 rank) const;
    u32 GetHSizeByVIndex(u32 vIndex) const;
    u32 GetRank(u32 vIndex, u32 hIndex) const;

    u32 threshold = 5;

private:
    u32 rankSize_{};
    u32 sqrtRankSize_{};
    u32 extraColSize_{};
    u32 extraRowSize_{};
    u32 rankOffset_{};
    u32 colSize_{};
    u32 rowSize_{};
};

class NHRV1Base : public AlgTemplateBase {
public:
    explicit NHRV1Base(const HcclDispatcher dispatcher);
    ~NHRV1Base() override;

    static RingInfo GetRingInfo(u32 rankSize);

private:
};
} // namespace hccl

#endif /* NONUNIFORM_HIERARCHICAL_RING_V1_BASE_PUB_H */

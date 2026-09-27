/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NONUNIFORM_HIERARCHICAL_RING_BASE_PUB_H
#define NONUNIFORM_HIERARCHICAL_RING_BASE_PUB_H

#include <cmath>
#include "alg_template_base_pub.h"

namespace hccl {

using InterServerAlgoStep = struct InterServerAlgoStepDef {
    u32 step = 0;
    u32 myRank = 0;

    u32 nSlices;
    u32 toRank = 0;
    u32 fromRank = 0;
    std::vector<u32> txSliceIdxs;
    std::vector<u32> rxSliceIdxs;

    InterServerAlgoStepDef() : nSlices(0) {}
};

class NHRBase : public AlgTemplateBase {
public:
    explicit NHRBase(const HcclDispatcher dispatcher);
    ~NHRBase() override;

    void GetRankMapping(const u32 rankSize, bool keepOrder = false);

    void FetchRankMapping(std::vector<u32>& sliceMap);

    void MergeSlices(std::vector<Slice>& slices);

protected:
    void ReorderSequence(u32 start, u32 end, u32 len, std::vector<u32>& tree, std::vector<u32>& tmp);

    u32 GetStepNumInterServer(u32 rankSize);

    virtual HcclResult GetStepInfo(u32 step, u32 nSteps, u32 rank, u32 rankSize, InterServerAlgoStep& stepInfo);

    HcclResult ExecuteBarrier(const std::shared_ptr<Transport>& preLink, const std::shared_ptr<Transport>& aftLink);

    std::vector<u32> sliceMap_;

    bool isNeedMerge = false;

private:
};
} // namespace hccl

#endif /* NONUNIFORM_HIERARCHICAL_RING_BASE_PUB_H */

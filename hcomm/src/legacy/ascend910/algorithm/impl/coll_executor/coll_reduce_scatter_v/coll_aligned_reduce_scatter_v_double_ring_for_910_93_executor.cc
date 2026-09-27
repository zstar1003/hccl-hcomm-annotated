/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_aligned_reduce_scatter_v_double_ring_for_910_93_executor.h"
#include <numeric>

namespace hccl {

CollAlignedReduceScatterVDoubleRingFor91093Executor::CollAlignedReduceScatterVDoubleRingFor91093Executor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAlignedReduceScatterDoubleRingFor91093Executor(dispatcher, topoMatcher)
{
    isReduceScatterV_ = true;
    desc_.level1SupportedAlgos
        = {AlgTypeLevel1::ALG_LEVEL1_NHR, AlgTypeLevel1::ALG_LEVEL1_NB, AlgTypeLevel1::ALG_LEVEL1_RING};
}

bool CollAlignedReduceScatterVDoubleRingFor91093Executor::IsUnifiedMarch(const OpParam& param) const
{
    (void)param;
    return false;
}

u64 CollAlignedReduceScatterVDoubleRingFor91093Executor::CalcLoopMaxCount(const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count，这里不除以RankSize，因为每次循环可能会减少需要参与通信的Rank
    return inCCLbufferSize_ / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN / unitSize;
}

bool CollAlignedReduceScatterVDoubleRingFor91093Executor::IsHugeData(const u64 curSize, OpParam* param)
{
    u32 level2RankSize;
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC
        || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE) {
        // AHC非对称场景下没有L2
        level2RankSize = 1;
    } else {
        // 多QP哈希散列开启且RDMA通信下，强制刷新子图
        // 这里如果CheckCommSize返回ERROR，相当于HugeData true，防止GetSubCommInfo越界
        CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
        SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
        level2RankSize = level2CommInfo.localRankSize;
    }

    const HcclDataType dataType = param->GetDataType();
    const u64 TBE_REDUCE_MAX_COUNT = INT32_MAX;
    u64 curCount = curSize / SIZE_TABLE[dataType];
    bool issupportRDMAInlineReduce = IsSupportRDMAReduce(dataType, param->reduceType);
    bool hugeData = (curSize * level2RankSize > RDMA_SEND_MAX_SIZE) || (curSize > SDMA_SEND_MAX_SIZE)
                    || ((!isSupportSDMAReduce_) && (curCount > TBE_REDUCE_MAX_COUNT))
                    || ((!issupportRDMAInlineReduce) && (curCount * level2RankSize > TBE_REDUCE_MAX_COUNT));
    return hugeData;
}

REGISTER_EXEC(
    "AlignedReduceScatterVDoubleRingFor91093Executor", AlignedReduceScatterVDoubleRingFor91093,
    CollAlignedReduceScatterVDoubleRingFor91093Executor);
} // namespace hccl

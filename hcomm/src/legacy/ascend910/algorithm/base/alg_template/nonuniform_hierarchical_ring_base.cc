/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include "nonuniform_hierarchical_ring_base.h"

namespace hccl {

NHRBase::NHRBase(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {}

NHRBase::~NHRBase() {}

void NHRBase::GetRankMapping(const u32 rankSize, bool keepOrder)
{
    std::vector<u32> tree;
    for (u32 i = 0; i < rankSize; i++) {
        tree.push_back(i);
    }

    if (keepOrder) {
        HCCL_DEBUG("[NHRBase][GetRankMapping] keep order and disable tree mapping, just return");
        sliceMap_ = tree;
        return;
    }

    // 其他的再进行计算
    std::vector<u32> tmp(rankSize);
    u32 nSteps = GetStepNumInterServer(rankSize);

    u32 len = rankSize;

    for (u32 step = 0; step < nSteps; step++) {
        u32 nSlices = (rankSize - 1 + (1 << step)) / (1 << (step + 1));
        if (nSlices <= 1) {
            break;
        }

        bool endFlag = false;

        for (u32 part = 0; part * len < rankSize; part++) {
            u32 start = part * len;
            u32 end = std::min(start + len, rankSize);
            ReorderSequence(start, end, len, tree, tmp);

            if (((end - start) & 1) == 1) {
                endFlag = true;
            }
        }

        for (u32 i = 0; i < rankSize; i++) {
            tree[i] = tmp[i];
        }

        if (endFlag) {
            break;
        }

        len >>= 1;
    }

    // 因为取的是tree中rank的idx，所以直接返回反向的映射
    sliceMap_.resize(rankSize);
    for (u32 i = 0; i < rankSize; i++) {
        sliceMap_[tree[i]] = i;
    }

    return;
}

void NHRBase::FetchRankMapping(std::vector<u32>& sliceMap) { sliceMap = sliceMap_; }

void NHRBase::ReorderSequence(u32 start, u32 end, u32 len, std::vector<u32>& tree, std::vector<u32>& tmp)
{
    const u32 DIVIDE_TWO = 2;

    for (u32 i = start; i < end; i++) {
        u32 offset = i - start;
        if ((offset & 1) == 0) {
            tmp[start + offset / DIVIDE_TWO] = tree[i];
        } else {
            tmp[start + (offset + len) / DIVIDE_TWO] = tree[i];
        }
    }
}

// 合并连续的内存块，slice数量可能会因此减少
void NHRBase::MergeSlices(std::vector<Slice>& slices)
{
    if (!isNeedMerge) {
        return;
    }

    if (slices.size() <= 1) {
        return;
    }

    std::sort(slices.begin(), slices.end(), [](const Slice& s1, const Slice& s2) {
        return s1.offset == s2.offset ? s1.size < s2.size : s1.offset < s2.offset;
    });

    u32 mergedIdx = 0;
    u64 tmpSliceOffset = slices[0].offset;
    u64 tmpSliceSize = slices[0].size;
    for (u32 i = 1; i < slices.size(); i++) {
        if (tmpSliceOffset + tmpSliceSize == slices[i].offset) {
            // 合并到上一块中
            tmpSliceSize += slices[i].size;
        } else {
            // 上一块先存储
            slices[mergedIdx].size = tmpSliceSize;
            slices[mergedIdx].offset = tmpSliceOffset;
            mergedIdx += 1;

            // 记录当前新的一块
            tmpSliceSize = slices[i].size;
            tmpSliceOffset = slices[i].offset;
        }
    }

    // 可能有size为0的片段，不判断slicesize > 0
    slices[mergedIdx].size = tmpSliceSize;
    slices[mergedIdx].offset = tmpSliceOffset;
    mergedIdx += 1;

    // 原地清理
    slices.erase(slices.begin() + mergedIdx, slices.end());

    return;
}

// NHR的算法步数
u32 NHRBase::GetStepNumInterServer(u32 rankSize)
{
    u32 nSteps = 0;
    for (u32 tmp = rankSize - 1; tmp != 0; tmp >>= 1, nSteps++) {
    }
    HCCL_DEBUG("[NHRBase][GetStepNumInterServer] rankSize[%u] nSteps[%u]", rankSize, nSteps);

    return nSteps;
}

// NHR每步的算法描述原理函数
HcclResult NHRBase::GetStepInfo(u32 step, u32 nSteps, u32 rank, u32 rankSize, InterServerAlgoStep& stepInfo)
{
    (void)step;
    (void)nSteps;
    (void)rank;
    (void)rankSize;
    (void)stepInfo;
    return HCCL_SUCCESS;
}

HcclResult NHRBase::ExecuteBarrier(const std::shared_ptr<Transport>& preLink, const std::shared_ptr<Transport>& aftLink)
{
    CHK_RET(preLink->TxAck(stream_));
    CHK_RET(aftLink->RxAck(stream_));

    CHK_RET(aftLink->TxDataSignal(stream_));
    CHK_RET(preLink->RxDataSignal(stream_));

    CHK_RET(preLink->PostFinAck(stream_));
    CHK_RET(aftLink->WaitFinAck(stream_));

    return HCCL_SUCCESS;
}
} // namespace hccl

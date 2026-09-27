/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_gather_v_ring_for_910_93_executor.h"
#include <numeric>

namespace hccl {
CollAllGatherVRingFor91093Executor::CollAllGatherVRingFor91093Executor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAllGatherRingFor91093Executor(dispatcher, topoMatcher)
{
    isAllGatherV_ = true;
    desc_.level1SupportedAlgos
        = {AlgTypeLevel1::ALG_LEVEL1_NHR, AlgTypeLevel1::ALG_LEVEL1_NB, AlgTypeLevel1::ALG_LEVEL1_RING};
}

bool CollAllGatherVRingFor91093Executor::IsSmallData(const u64 size)
{
    (void)size;
    return false;
}

u64 CollAllGatherVRingFor91093Executor::CalcDstMemOffset(const OpParam& param, u32 perDataSize, u64 inputMemSize) const
{
    (void)inputMemSize;
    const auto* counts = static_cast<const u64*>(param.VDataDes.counts);
    const u64 offset = std::accumulate(counts, counts + topoAttr_.userRank, 0ULL);
    return offset * perDataSize;
}

HcomCollOpInfo CollAllGatherVRingFor91093Executor::GetHcomCollOpInfo(const OpParam& param, const ExecMem& execMem) const
{
    HcomCollOpInfo opInfo = {
        "", execMem.inputPtr, execMem.outputPtr, execMem.count, param.VDataDes.dataType, param.root, param.reduceType,
        0 // 暂不支持MC2的strideCount特性
    };
    return opInfo;
}

std::vector<Slice> CollAllGatherVRingFor91093Executor::PrepareSlicesL2(
    const OpParam& param, const SubCommInfo& level2CommInfo, const SubCommInfo& level1CommInfo,
    const SubCommInfo& level0CommInfo, u32 perDataSize, u64 inputMemSize) const
{
    (void)inputMemSize;
    std::vector<Slice> level2DataSegsSlice;
    const auto* counts = static_cast<u64*>(param.VDataDes.counts);
    const u32 level0RankSize = level0CommInfo.localRankSize;
    const u32 level0ServerIndex = level0CommInfo.localRank;
    const u32 level1RankSize = level1CommInfo.localRankSize;
    const u32 level1ServerIndex = level1CommInfo.localRank;
    const u32 level2RankSize = level2CommInfo.localRankSize;
    for (u32 i = 0; i < level2RankSize; i++) {
        const u32 rank = i * level1RankSize * level0RankSize + level1ServerIndex * level0RankSize + level0ServerIndex;
        Slice sliceTemp;
        sliceTemp.size = counts[rank] * perDataSize;
        const u64 offset = std::accumulate(counts, counts + rank, 0ULL);
        sliceTemp.offset = offset * perDataSize;
        level2DataSegsSlice.push_back(sliceTemp);
    }
    return level2DataSegsSlice;
}

std::vector<Slice> CollAllGatherVRingFor91093Executor::PrepareSlicesL1(
    const OpParam& param, const SubCommInfo& level2CommInfo, const SubCommInfo& level1CommInfo,
    const SubCommInfo& level0CommInfo, u32 perDataSize, u64 inputMemSize) const
{
    (void)inputMemSize;
    const auto* counts = static_cast<u64*>(param.VDataDes.counts);
    const u32 level0RankSize = level0CommInfo.localRankSize;
    const u32 level0ServerIndex = level0CommInfo.localRank;
    const u32 level1RankSize = level1CommInfo.localRankSize;
    const u32 level2RankSize = level2CommInfo.localRankSize;
    std::vector<Slice> level1DataSegsSlice;
    for (u32 j = 0; j < level1RankSize; j++) {
        for (u32 i = 0; i < level2RankSize; i++) {
            Slice level1Slice;
            const u32 rank = i * level1RankSize * level0RankSize + j * level0RankSize + level0ServerIndex;
            level1Slice.size = counts[rank] * perDataSize;
            const u64 offset = std::accumulate(counts, counts + rank, 0ULL);
            level1Slice.offset = offset * perDataSize;
            level1DataSegsSlice.push_back(level1Slice);
        }
    }
    return level1DataSegsSlice;
}

HcclResult CollAllGatherVRingFor91093Executor::PrepareSlicesL0(
    std::vector<std::vector<Slice>>& multRingsSlice, const OpParam& param, const SubCommInfo& level2CommInfo,
    const SubCommInfo& level1CommInfo, const SubCommInfo& level0CommInfo, u32 perDataSize, u64 inputMemSize)
{
    (void)inputMemSize;
    HCCL_CONFIG_INFO(
        HCCL_ALG, "[CollAllGatherVRingFor91093Executor][PrepareSlicesL0] userRank[%u] starts.", topoAttr_.userRank);
    const auto* counts = static_cast<u64*>(param.VDataDes.counts);
    const u32 level0RankSize = level0CommInfo.localRankSize;
    const u32 level1RankSize = level1CommInfo.localRankSize;
    const u32 level2RankSize = level2CommInfo.localRankSize;
    std::vector<Slice> dataSegsSlice;
    for (u32 k = 0; k < level0RankSize; k++) { // 根据数据量计算每个环上数据的偏移和大小
        for (u32 i = 0; i < level2RankSize; i++) {
            for (u32 j = 0; j < level1RankSize; j++) {
                Slice sliceTemp;
                const u32 rank = i * level1RankSize * level0RankSize + j * level0RankSize + k;
                sliceTemp.size = counts[rank] * perDataSize;
                const u64 offset = std::accumulate(counts, counts + rank, 0ULL);
                sliceTemp.offset = offset * perDataSize; // no displs
                dataSegsSlice.push_back(sliceTemp);
            }
        }
    }
    multRingsSlice.push_back(dataSegsSlice);
    return HCCL_SUCCESS;
}

// AGV不支持MC2的strideCount特性
HcclResult CollAllGatherVRingFor91093Executor::PrepareUserMemSlices(
    std::vector<std::vector<Slice>>& userMemSlices, const std::vector<std::vector<Slice>>& multRingsSlice,
    const OpParam& param, const SubCommInfo& level2CommInfo, const SubCommInfo& level1CommInfo,
    const SubCommInfo& level0CommInfo, u32 perDataSize, u64 inputMemSize)
{
    (void)inputMemSize;
    (void)multRingsSlice;
    const auto* counts = static_cast<u64*>(param.VDataDes.counts);
    const auto* displs = static_cast<u64*>(param.VDataDes.displs);
    std::vector<Slice> dataSegsSlice;
    const u32 level0RankSize = level0CommInfo.localRankSize;
    const u32 level1RankSize = level1CommInfo.localRankSize;
    const u32 level2RankSize = level2CommInfo.localRankSize;
    for (u32 k = 0; k < level0RankSize; k++) { // 根据数据量计算每个环上数据的偏移和大小
        for (u32 i = 0; i < level2RankSize; i++) {
            for (u32 j = 0; j < level1RankSize; j++) {
                Slice sliceTemp;
                const u32 rank = i * level1RankSize * level0RankSize + j * level0RankSize + k;
                sliceTemp.size = counts[rank] * perDataSize;
                sliceTemp.offset = displs[rank] * perDataSize; // with displs
                dataSegsSlice.push_back(sliceTemp);
            }
        }
    }
    userMemSlices.push_back(dataSegsSlice);
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllGatherVRingFor91093Executor", AllGatherVRingFor91093, CollAllGatherVRingFor91093Executor);

} // namespace hccl

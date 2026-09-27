/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_aligned_all_gather_v_double_ring_for_910_93_executor.h"
#include <numeric>

namespace hccl {
CollAlignedAllGatherVDoubleRingFor91093Executor::CollAlignedAllGatherVDoubleRingFor91093Executor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAlignedAllGatherDoubleRingFor91093Executor(dispatcher, topoMatcher)
{
    isAllGatherV_ = true;
    desc_.level1SupportedAlgos
        = {AlgTypeLevel1::ALG_LEVEL1_NHR, AlgTypeLevel1::ALG_LEVEL1_NB, AlgTypeLevel1::ALG_LEVEL1_RING};
}

bool CollAlignedAllGatherVDoubleRingFor91093Executor::IsSmallData(const u64 size)
{
    (void)size;
    return false;
}

u64 CollAlignedAllGatherVDoubleRingFor91093Executor::CalcDstMemOffset(
    const OpParam& param, u32 perDataSize, u64 inputMemSize) const
{
    (void)inputMemSize;
    const auto* counts = static_cast<const u64*>(param.VDataDes.counts);
    const u64 offset = std::accumulate(counts, counts + topoAttr_.userRank, 0ULL);
    return offset * perDataSize;
}

HcomCollOpInfo
CollAlignedAllGatherVDoubleRingFor91093Executor::GetHcomCollOpInfo(const OpParam& param, const ExecMem& execMem) const
{
    HcomCollOpInfo opInfo = {
        "", execMem.inputPtr, execMem.outputPtr, execMem.count, param.VDataDes.dataType, param.root, param.reduceType,
        0 // 暂不支持MC2的strideCount特性
    };
    if (!DMAReduceFlag_) {
        opInfo.inputAddr = execMem.inputMem.ptr();
        opInfo.outputAddr = execMem.outputMem.ptr();
    }
    return opInfo;
}

std::vector<Slice> CollAlignedAllGatherVDoubleRingFor91093Executor::PrepareSlicesL2(
    const OpParam& param, const SubCommInfo& level2CommInfo, const SubCommInfo& level1CommInfo,
    const SubCommInfo& level0CommInfo, u32 perDataSize, u64 inputMemSize) const
{
    (void)inputMemSize;
    const auto* counts = static_cast<u64*>(param.VDataDes.counts);
    const u32 level0RankSize = level0CommInfo.localRankSize;
    const u32 level0ServerIndex = level0CommInfo.localRank;
    const u32 level1RankSize = level1CommInfo.localRankSize;
    const u32 level1ServerIndex = level1CommInfo.localRank;
    const u32 level2RankSize = level2CommInfo.localRankSize;
    std::vector<Slice> level2DataSegsSlice;
    for (u32 i = 0; i < level2RankSize; i++) {
        Slice sliceTemp;
        const u32 rank = i * level1RankSize * level0RankSize + level1ServerIndex * level0RankSize + level0ServerIndex;
        sliceTemp.size = counts[rank] * perDataSize;
        const u64 offset = std::accumulate(counts, counts + rank, 0ULL);
        sliceTemp.offset = offset * perDataSize;
        level2DataSegsSlice.push_back(sliceTemp);
    }
    return level2DataSegsSlice;
}

std::vector<Slice> CollAlignedAllGatherVDoubleRingFor91093Executor::PrepareSlicesL1(
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

HcclResult CollAlignedAllGatherVDoubleRingFor91093Executor::PrepareSlicesL0(
    std::vector<std::vector<Slice>>& multRingsSlice, const OpParam& param, const SubCommInfo& level2CommInfo,
    const SubCommInfo& level1CommInfo, const SubCommInfo& level0CommInfo, u32 perDataSize, u64 inputMemSize)
{
    (void)inputMemSize;
    HCCL_CONFIG_INFO(
        HCCL_ALG, "[CollAlignedAllGatherVDoubleRingFor91093Executor][PrepareSlicesL0] userRank[%u] starts.",
        topoAttr_.userRank);
    const auto* counts = static_cast<u64*>(param.VDataDes.counts);
    const u32 level0RankSize = level0CommInfo.localRankSize;
    const u32 level1RankSize = level1CommInfo.localRankSize;
    const u32 level2RankSize = level2CommInfo.localRankSize;
    u32 ringSize = 0;
    std::vector<std::vector<std::vector<Slice>>> multRingsSliceZeroServers;
    for (u32 i = 0; i < level2RankSize; i++) {
        for (u32 j = 0; j < level1RankSize; j++) {
            std::vector<Slice> dataSegsSlice;
            for (u32 k = 0; k < level0RankSize; k++) { // 根据数据量计算每个环上数据的偏移和大小
                Slice sliceTemp;
                const u32 rank = i * level1RankSize * level0RankSize + j * level0RankSize + k;
                sliceTemp.size = counts[rank] * perDataSize;
                const u64 offset = std::accumulate(counts, counts + rank, 0ULL);
                sliceTemp.offset = offset * perDataSize; // no displs
                dataSegsSlice.push_back(sliceTemp);
            }
            // 机内多环数据切分
            auto multRingsSliceZeroServer = PrepareMultiRingSlice(dataSegsSlice, param.tag, false, topoAttr_.nicList);
            if (ringSize == 0) {
                ringSize = multRingsSliceZeroServer.size();
            } else {
                CHK_PRT_RET(
                    multRingsSliceZeroServer.size() != ringSize,
                    HCCL_ERROR(
                        "[CollAlignedAllGatherVDoubleRingFor91093Executor][PrepareSlicesL0]mismatch "
                        "ringSize[%u], expect[%u]",
                        multRingsSliceZeroServer.size(), ringSize),
                    HCCL_E_PARA);
            }
            multRingsSliceZeroServers.push_back(multRingsSliceZeroServer);
        }
    }
    multRingsSlice.resize(ringSize);
    for (u32 k = 0; k < level0RankSize; k++) {
        for (u32 i = 0; i < level2RankSize; i++) {
            for (u32 j = 0; j < level1RankSize; j++) { // 按照机内rank的顺序调整数据分片的排布
                const u32 serverIndex = i * level1RankSize + j;
                const auto& multRingsSliceZeroServer = multRingsSliceZeroServers[serverIndex];
                for (u32 ringIndex = 0; ringIndex < ringSize; ringIndex++) {
                    multRingsSlice[ringIndex].push_back(multRingsSliceZeroServer[ringIndex][k]);
                }
            }
        }
    }
    return HCCL_SUCCESS;
}

// AGV不支持MC2的strideCount特性
HcclResult CollAlignedAllGatherVDoubleRingFor91093Executor::PrepareUserMemSlices(
    std::vector<std::vector<Slice>>& userMemSlices, const std::vector<std::vector<Slice>>& multRingsSlice,
    const OpParam& param, const SubCommInfo& level2CommInfo, const SubCommInfo& level1CommInfo,
    const SubCommInfo& level0CommInfo, u32 perDataSize, u64 inputMemSize)
{
    (void)multRingsSlice;
    (void)inputMemSize;
    const auto* counts = static_cast<u64*>(param.VDataDes.counts);
    const auto* displs = static_cast<u64*>(param.VDataDes.displs);
    const u32 level0RankSize = level0CommInfo.localRankSize;
    const u32 level1RankSize = level1CommInfo.localRankSize;
    const u32 level2RankSize = level2CommInfo.localRankSize;
    u32 ringSize = 0;
    std::vector<std::vector<std::vector<Slice>>> userMemSlicesServers;
    for (u32 i = 0; i < level2RankSize; i++) {
        for (u32 j = 0; j < level1RankSize; j++) {
            std::vector<Slice> dataSegsSlice;
            for (u32 k = 0; k < level0RankSize; k++) { // 根据数据量计算每个环上数据的偏移和大小
                Slice sliceTemp;
                const u32 rank = i * level1RankSize * level0RankSize + j * level0RankSize + k;
                sliceTemp.size = counts[rank] * perDataSize;
                sliceTemp.offset = displs[rank] * perDataSize; // with displs
                dataSegsSlice.push_back(sliceTemp);
            }
            // 多环数据切分
            auto userMemSlicesServer = PrepareMultiRingSlice(dataSegsSlice, param.tag, false, topoAttr_.nicList);
            if (ringSize == 0) {
                ringSize = userMemSlicesServer.size();
            } else {
                CHK_PRT_RET(
                    userMemSlicesServer.size() != ringSize,
                    HCCL_ERROR(
                        "[CollAlignedAllGatherVDoubleRingFor91093Executor][PrepareUserMemSlices]mismatch "
                        "ringSize[%u], expect[%u]",
                        userMemSlicesServer.size(), ringSize),
                    HCCL_E_PARA);
            }
            userMemSlicesServers.push_back(userMemSlicesServer);
        }
    }
    userMemSlices.resize(ringSize);
    for (u32 k = 0; k < level0RankSize; k++) {
        for (u32 i = 0; i < level2RankSize; i++) {
            for (u32 j = 0; j < level1RankSize; j++) { // 按照机内rank的顺序调整数据分片的排布
                const u32 serverIndex = i * level1RankSize + j;
                const auto& userMemSlicesServer = userMemSlicesServers[serverIndex];
                for (u32 ringIndex = 0; ringIndex < ringSize; ringIndex++) {
                    userMemSlices[ringIndex].push_back(userMemSlicesServer[ringIndex][k]);
                }
            }
        }
    }
    return HCCL_SUCCESS;
}

REGISTER_EXEC(
    "AlignedAllGatherVDoubleRingFor91093Executor", AlignedAllGatherVDoubleRingFor91093,
    CollAlignedAllGatherVDoubleRingFor91093Executor);

} // namespace hccl

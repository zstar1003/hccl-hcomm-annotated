/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_COMMON_EXECUTOR_H
#define COLL_COMMON_EXECUTOR_H

#include "coll_native_executor_base.h"
#include "coll_alg_exec_registry.h"
#include "profiler_base_pub.h"
#include "send_receive_pub.h"
#include "alg_template_register.h"
#include "alltoallv_staged_calculator_pub.h"

namespace hccl {
constexpr u32 NSLBDP_MIN_COUNT = 128;
class CollCommExecutor : public CollNativeExecutorBase {
public:
    CollCommExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollCommExecutor() override = default;

    // CCL Op Share
    HcclResult MultiRingAllReduce(
        const std::string& tag, DeviceMem& inputMem, DeviceMem& outputMem, const u64 count, const HcclDataType dataType,
        const HcclReduceOp reductionOp, const std::vector<std::vector<Slice>>& multRingsSliceZero, Stream stream,
        s32 profStage, const u64 baseOffset = 0);
    HcclResult CollectMultiRingsUserMemSlices(
        u32 ringNum, const HcclDataType dataType, const HcomCollOpInfo* opInfo,
        const std::vector<std::vector<Slice>>& multRingsSliceZero, const std::vector<std::vector<u32>>& multiRingsOrder,
        const std::vector<std::vector<Slice>>& multRingsUserMemSlice,
        std::vector<std::vector<Slice>>& userMemSlicesOfMultiRings);
    HcclResult CollectMultiRingsRankOrder(
        u32 ringNum, const std::vector<std::vector<u32>>& multiRingsOrder, std::vector<std::vector<u32>>& rankOrders);
    u32 CalcOptimalIntraRingsize(u64 count, HcclDataType dataType, HcclCMDType opType);

    HcclResult MultiRingReduceScatter(
        const std::string& tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count, const HcclDataType dataType,
        const HcclReduceOp reductionOp, const std::vector<std::vector<Slice>> multRingsSliceZero, Stream stream,
        s32 profStage, const u64 baseOffset = 0, const HcomCollOpInfo* opInfo = nullptr,
        const std::vector<std::vector<Slice>> multRingsUserMemSlice = std::vector<std::vector<Slice>>(0),
        const CommPlane levelIndex = COMM_LEVEL0);

    HcclResult MultiRingReduceScatterConcurrent(
        const std::string& tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count, const HcclDataType dataType,
        const HcclReduceOp reductionOp, const std::vector<std::pair<bool, std::vector<Slice>>> multRingsSliceZero,
        Stream stream, s32 profStage, const u64 baseOffset = 0, const HcomCollOpInfo* opInfo = nullptr,
        const std::vector<std::pair<bool, std::vector<Slice>>> multRingsUserMemSlice
        = std::vector<std::pair<bool, std::vector<Slice>>>(0));

    HcclResult Level1ReduceScatterConcurrent(
        DeviceMem inputMem, DeviceMem scratchMem, const u64 count, const HcclDataType dataType,
        const HcclReduceOp reductionOp, Stream stream, s32 profStage, std::vector<Slice>& level1DataSegsSlice,
        u32 syncTrans, u64 reduceAttr);

    HcclResult
    UpdateOffsetBasedOnStrideCount(const OpParam& param, std::vector<std::vector<Slice>>& multRingsUserMemSlice) const;

    HcclResult MultiRingAllGather(
        const std::string& tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count, const HcclDataType dataType,
        const std::vector<std::vector<Slice>> multRingsSliceZero, Stream stream, s32 profStage,
        const u64 baseOffset = 0, const HcomCollOpInfo* opInfo = nullptr,
        const std::vector<std::vector<Slice>> multRingsUserMemSlice = std::vector<std::vector<Slice>>(0),
        const CommPlane leveIndex = COMM_LEVEL0);

    HcclResult MultiRingAllGatherConcurrent(
        const std::string& tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count, const HcclDataType dataType,
        const std::vector<std::pair<bool, std::vector<Slice>>> multRingsSliceZero, Stream stream, s32 profStage,
        const u64 baseOffset = 0, const HcomCollOpInfo* opInfo = nullptr,
        const std::vector<std::pair<bool, std::vector<Slice>>> multRingsUserMemSlice
        = std::vector<std::pair<bool, std::vector<Slice>>>(0));

    HcclResult Level1AllGatherConcurrent(
        DeviceMem inputMem, DeviceMem outputMem, const u64 count, const HcclDataType dataType, Stream stream,
        s32 profStage, std::vector<Slice>& level1DataSegsSlice, u32 syncTrans);

    HcclResult MultiRingMultiRootScatter(
        const std::string& tag, DeviceMem& inputMem, DeviceMem& outputMem, const u64 count, const HcclDataType dataType,
        const std::vector<std::vector<Slice>>& multRingsSliceZero, u32 root, Stream stream, const u64 baseOffset);

    HcclResult MultiStreamReduceScatterMesh(
        const std::string& tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count, const HcclDataType dataType,
        const HcclReduceOp reductionOp, const std::vector<std::vector<Slice>>& multStreamsSlice, Stream stream,
        const CommPlane commLevelIndex, const u64 baseOffset = 0);

    HcclResult MultiRingGather(
        const std::string& tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count, const HcclDataType dataType,
        const std::vector<std::vector<Slice>> multRingsSliceZero, HcclReduceOp op, u32 root, Stream stream,
        s32 profStage);

    HcclResult MultiStreamReduceScatterMeshAtomic(
        const std::string& tag, DeviceMem& inputMem, DeviceMem& outputMem, const u64 count, const HcclDataType dataType,
        const HcclReduceOp reductionOp, const std::vector<Slice>& dataSliceVct, Stream& stream,
        const CommPlane commLevelIndex, const u64 baseOffset = 0, HcomCollOpInfo* opInfo = nullptr);
    HcclResult
    PrepareReduceScatterSliceData(u64 dataCount, u32 unitSize, u32 sliceNum, std::vector<Slice>& dataSlice) const;

    HcclResult MultiRingScatter(
        const std::string& tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count, const HcclDataType dataType,
        const std::vector<std::vector<Slice>> multRingsSliceZero, u32 root, Stream stream, const HcomCollOpInfo* opInfo,
        const u64 baseOffset = 0);
    std::vector<std::vector<u32>>
    GetRingsOrderByTopoType(u32 ranksSize, TopoType topoType, std::vector<u32>& nicList) const;
    HcclResult MutliSegSlicePrepare(
        const std::vector<Slice>& dataSegsSlice, std::vector<std::vector<Slice>>& mutliSegsSlices, u32 ringCount) const;
    HcclResult MutliSegSlicePrepareAvoidCceRewrite(
        const std::vector<Slice>& dataSegsSlice, std::vector<std::vector<Slice>>& mutliSegsSlices, u32 ringCount) const;
    void NicSendSizeCal(
        const std::vector<std::vector<Slice>>& mutliSegsSlices, u32 ringCount, u32 chunkSize,
        const std::vector<u32>& nicList, const std::string& tag);
    std::vector<std::vector<Slice>> PrepareMultiRingSlice(
        const std::vector<Slice>& dataSegsSlice, const std::string& tag, bool avoidCceRewrite = false,
        std::vector<u32> nicList = {0, 1, 2, 3, 4, 5, 6, 7}, CommPlane commLevelIndex = COMM_LEVEL0);
    // AnyPath特性使用
    std::vector<std::vector<u32>>
    GetRingsOrderForAnyPath(u32 ranksSize, TopoType topoType, std::vector<u32>& nicList) const;
    std::vector<std::vector<Slice>> AnyPathPrepareMultiRingSlice(
        const std::vector<Slice>& dataSegsSlice, const std::string& tag, bool avoidCceRewrite = false,
        std::vector<u32> nicList = {0, 1, 2, 3, 4, 5, 6, 7});

    bool Is2U2PInfer();
    bool Is910BSingleMesh();
    bool NeedCreateSingleMeshPlane(const bool isInlineReduce);
    bool SingleMeshInlineReduce(void* inputPtr, void* outputPtr, HcclDataType dataType, HcclReduceOp op);
    bool IsMultiMeshInlineReduce(void* inputPtr, void* outputPtr, HcclDataType dataType, HcclReduceOp op);

    u64 GetReduceAttr(DeviceMem& inputMem, DeviceMem& outputMem, HcclDataType dataType, HcclReduceOp op);
    HcclResult PrepareLevel1CommInfo(
        u32& segmentIdx, u32& commIndex, u64& hdSize, const SubCommInfo& commInfo,
        const std::vector<std::vector<Slice>>& multRingsSliceZero, const std::string& tag);
    HcclResult GetAdjInfo(AlgResourceResponse& algRes, AdjInfo& adjInfo) override;

protected:
    virtual HcclResult GetSubStreamInfoOnOneRing(
        const u32 ringIndex, std::vector<Stream>& subStreamsInOneRing,
        std::vector<std::shared_ptr<LocalNotify>>& mainSignalsInOneRing,
        std::vector<std::shared_ptr<LocalNotify>>& subSignalsInOneRing);
    virtual u32 GetLevel0RingNum() const;
    HcclResult CalUserMemSlices(
        const HcclDataType dataType, const HcomCollOpInfo* opInfo, const std::vector<Slice>& singleRingSliceZero,
        u32 ringIndex, const std::vector<std::vector<u32>>& multiRingsOrder, std::vector<Slice>& userMemSlices);
    HcclResult GetRankOrder(
        const std::vector<std::vector<u32>>& multiRingsOrder, u32 ringIndex, std::vector<u32>& rankOrder) const;
    HcclResult SetRingNics(const std::string& tag, const std::vector<std::vector<u32>>& ringNics);
    HcclResult GetRingNics(const std::string& tag, std::vector<std::vector<u32>>& ringNics);
    HcclResult SetNicSendSize(const std::string& tag, std::vector<u64>& sizeList);

    // 用于ZerocopyExecutor
    HcclResult CalcIntraServerDataSlicesDiscontinuous(
        const OpParam& param, const ExecMem& execMem, u32 level0RankSize, u32 level1RankSize, u32 level2RankSize,
        std::vector<Slice>& dataSegsSlice);
    HcclResult CalcIntraServerDataSlicesContinuous(
        const OpParam& param, const ExecMem& execMem, u32 level0RankSize, u32 level1RankSize, u32 level2RankSize,
        std::vector<Slice>& dataSegsSlice) const;
    void CalcLevel1DataSlices(
        u64 sliceSize, u32 level1RankSize, u32 level2RankSize, std::vector<Slice>& level1DataSegsSlice) const;
    HcclResult GetCommRankInfoNormal(
        u32& level0Rank, u32& level0RankSize, u32& level1Rank, u32& level1RankSize, u32& level2Rank,
        u32& level2RankSize, bool isAHCAlgo = false);

    // 用于ExchangeExecutor
    HcclResult CalExchangeRemoteRankForReduceScatter(u32& remoteRankSend, u32& remoteRankRecv);
    HcclResult GetTransportForExchange(u32 remoteUserRank, LINK& targetLink);
    bool IsLevel0Neighbor(u32 remoteRank, u32 level0RankSize);

    std::mutex ringNicListLock_;
    std::map<std::string, std::vector<std::vector<u32>>> ringNicList_;
    std::mutex nicSendSizeListLock_;
    std::map<std::string, std::vector<u64>> nicSendSizeList_;
};
} // namespace hccl

#endif /** __COLL_COMMON_EXECUTOR_H__ */

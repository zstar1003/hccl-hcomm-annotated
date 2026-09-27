/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_ring_for_910_93_executor.h"
#include <numeric>
#include "alg_template_register.h"

namespace hccl {

CollReduceScatterRingFor91093Executor::CollReduceScatterRingFor91093Executor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollReduceScatterExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    desc_.deterministic = 1;
    desc_.level1SupportedAlgos
        = {AlgTypeLevel1::ALG_LEVEL1_NHR, AlgTypeLevel1::ALG_LEVEL1_NB, AlgTypeLevel1::ALG_LEVEL1_RING,
           AlgTypeLevel1::ALG_LEVEL1_AHC, AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE};
    desc_.level2SupportedAlgos
        = {AlgTypeLevel2::ALG_LEVEL2_NHR, AlgTypeLevel2::ALG_LEVEL2_NB, AlgTypeLevel2::ALG_LEVEL2_RING};
}

bool CollReduceScatterRingFor91093Executor::IsUnifiedMarch(const OpParam& param) const
{
    return IsSupportUnifiedMarch(param, topoType_, topoAttr_.serverNum, topoAttr_.superPodNum);
}

u64 CollReduceScatterRingFor91093Executor::CalcTotalCount(const OpParam& param) const
{
    if (isReduceScatterV_) {
        const auto* counts = static_cast<const u64*>(param.VDataDes.counts);
        return std::accumulate(counts, counts + topoAttr_.userRankSize, 0ULL);
    }
    return param.DataDes.count * topoAttr_.userRankSize;
}

void CollReduceScatterRingFor91093Executor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;

    const HcclDataType dataType = param.GetDataType();
    // 是否需要scratch memory
    if ((workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) && isSupportSDMAReduce_
        && IsSupportRDMAReduce(dataType, param.reduceType)) {
        scratchMemFlag_ = false;
    } else {
        scratchMemFlag_ = true;
    }

    HCCL_DEBUG(
        "[CollReduceScatterRingFor91093Executor][ParseParam] tag[%s] isSupportSDMAReduce_[%u] "
        "scratchMemFlag_[%u] workflowMode_[%u]",
        tag_.c_str(), isSupportSDMAReduce_, scratchMemFlag_, workflowMode_);

    // 记录图模式总数据量
    totalSize_ = CalcTotalCount(param) * SIZE_TABLE[dataType];
    aicpuUnfoldMode_ = param.aicpuUnfoldMode;
    isZeroCopy_ = param.isZeroCopy;
}

HcclResult CollReduceScatterRingFor91093Executor::CalcScratchMemSize(u64& scratchMemSize)
{
    if (scratchMemFlag_) {
        if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            scratchMemSize = inCCLbufferSize_;
        } else {
            scratchMemSize = totalSize_;
        }
    } else {
        scratchMemSize = 0U;
    }
    HCCL_INFO(
        "[CollReduceScatterRingFor91093Executor][CalcScratchMemSize] tag[%s] scratchMemSize[%llu] "
        "scratchMemFlag_[%u] workflowMode_[%u]",
        tag_.c_str(), scratchMemSize, scratchMemFlag_, workflowMode_);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingFor91093Executor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum
        = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING ? LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE :
                                                             LEVEL0_PLANE_NUM_IN_NPRING_SINGLE);
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        totalStreamNum *= STREAM_NUM_FOR_DMAREDUCE_ONE_RING;
    }
    streamNum = totalStreamNum - 1;
    HCCL_INFO("[CollReduceScatterRingFor91093Executor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingFor91093Executor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel2CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingFor91093Executor::CalcTransportMemType(
    TransportMemType& inputType, TransportMemType& outputType) const
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        if (scratchMemFlag_) {
            outputType = TransportMemType::SCRATCH;
        } else {
            outputType = TransportMemType::CCL_OUTPUT;
        }
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        if (scratchMemFlag_) {
            outputType = TransportMemType::SCRATCH;
        } else {
            outputType = TransportMemType::PARAM_OUTPUT;
        }
    }
    HCCL_INFO(
        "[CollReduceScatterRingFor91093Executor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingFor91093Executor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingFor91093Executor::CalcLevel2CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC
        || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE) {
        HCCL_INFO(
            "[CollReduceScatterRingFor91093Executor][CalcLevel2CommInfo] select AHC bypass level2 comm calculate");
        return HCCL_SUCCESS;
    }

    CommParaInfo commParaLevel2(COMM_LEVEL2, CommType::COMM_TAG_MAX);
    if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
        commParaLevel2.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING;
        HCCL_INFO("[%s]Calc NHRCommInfo", __func__);
    } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
        commParaLevel2.commType = CommType::COMM_TAG_NONUNIFORM_BRUCK;
        HCCL_INFO("[%s]Calc NBCommInfo", __func__);
    } else {
        commParaLevel2.commType = CommType::COMM_TAG_RING_INNER;
        HCCL_INFO("[%s]Calc RingCommInfo", __func__);
    }
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel2, opTransport[COMM_LEVEL2], inputType, outputType));
    return HCCL_SUCCESS;
}

u64 CollReduceScatterRingFor91093Executor::CalcLoopMaxCount(const u32 unitSize)
{
    // 中转内存单次最多能够接受的output count，放开ranksize限制
    u64 maxCountPerLoop
        = inCCLbufferSize_ / topoAttr_.userRankSize / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN / unitSize;
    return maxCountPerLoop;
}

bool CollReduceScatterRingFor91093Executor::IsHugeData(const u64 curSize, OpParam* param)
{
    u32 level2RankSize;
    if ((algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC
         || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE)) {
        // AHC非对称场景下没有L2
        level2RankSize = 1;
    } else {
        // 多QP哈希散列开启且RDMA通信下，强制刷新子图
        // 这里如果CheckCommSize返回ERROR，相当于HugeData true，防止GetSubCommInfo越界
        CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
        SubCommInfo level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
        level2RankSize = level2CommInfo.localRankSize;
    }

    const u64 TBE_REDUCE_MAX_COUNT = INT32_MAX;

    u64 curCount = curSize / SIZE_TABLE[param->DataDes.dataType];
    bool issupportRDMAInlineReduce = IsSupportRDMAReduce(param->DataDes.dataType, param->reduceType);
    bool hugeData = (curSize * level2RankSize / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE)
                    || (curSize > SDMA_SEND_MAX_SIZE) || ((!isSupportSDMAReduce_) && (curCount > TBE_REDUCE_MAX_COUNT))
                    || ((!issupportRDMAInlineReduce)
                        && (curCount * level2RankSize / HCCL_INTERNODE_MAX_DATA_RATE > TBE_REDUCE_MAX_COUNT));
    return hugeData;
}

HcclResult CollReduceScatterRingFor91093Executor::RunIntraSeverReduceScatter(
    const std::string& tag, DeviceMem& inputMem, DeviceMem& outputMem, const u64 count, const HcclDataType& dataType,
    const HcclReduceOp& reductionOp, const std::vector<std::vector<Slice>>& multRingsSliceZero, const Stream& stream,
    s32 profStage, const u64 baseOffset, const HcomCollOpInfo* opInfo,
    const std::vector<std::vector<Slice>>& multRingsUserMemSlice, [[maybe_unused]] const bool disableDMAReduce)
{
    CHK_RET(MultiRingReduceScatter(
        tag, inputMem, outputMem, count, dataType, reductionOp, multRingsSliceZero, stream, profStage, baseOffset,
        opInfo, multRingsUserMemSlice, logicalLevel0plane_));
    return HCCL_SUCCESS;
}

void CollReduceScatterRingFor91093Executor::FillMultiRingSlice(
    const ExecMem& execMem, const std::vector<std::vector<Slice>>& multiStreamSlice, u32 sliceNum, u32 level1RankSize,
    u32 level2RankSize, const u32 ringIndex, std::vector<Slice>& dataSlice)
{
    for (u32 level0Idx = 0; level0Idx < sliceNum; level0Idx++) {
        Slice sliceTemp;
        for (u32 level2Idx = 0; level2Idx < level2RankSize; level2Idx++) {
            for (u32 level1Idx = 0; level1Idx < level1RankSize; level1Idx++) {
                sliceTemp.size = multiStreamSlice[ringIndex][level0Idx].size;
                sliceTemp.offset = multiStreamSlice[ringIndex][level0Idx].offset
                                   + level1Idx * sliceNum * execMem.outputMem.size()
                                   + level2Idx * sliceNum * level1RankSize * execMem.outputMem.size();
                dataSlice.push_back(sliceTemp);
                HCCL_DEBUG(
                    "rank[%u] sliceTemp.size[%zu], sliceTemp.offset[%llu]", topoAttr_.userRank, sliceTemp.size,
                    sliceTemp.offset);
            }
        }
    }
}

HcclResult CollReduceScatterRingFor91093Executor::CalLevel0DataSegsSlice(
    const ExecMem& execMem, std::vector<std::vector<Slice>>& multiStreamSlice, const OpParam& param, u32 ringNum,
    u32 sliceNum, u32 level1RankSize, u32 level2RankSize, HcclDataType dataType,
    std::vector<std::vector<Slice>>& level0DataSegsSlice)
{
    if (isReduceScatterV_) {
        return CalLevel0DataSegsSliceV(
            execMem, multiStreamSlice, param, ringNum, sliceNum, level1RankSize, level2RankSize, dataType,
            level0DataSegsSlice);
    }
    bool isInlineReduce
        = IsSupportSDMAReduce(execMem.inputMem.ptr(), execMem.scratchMem.ptr(), dataType, param.reduceType);
    bool useInlineReduce = isInlineReduce && algoAttr_.inlineReduceSwitchOn;
    std::vector<Slice> dataSegsSlice; // 数据分成ranksize份，每份的起始偏移和大小
    multiStreamSlice = ReduceScatterRingSlicePrepare(
        ringNum, sliceNum, useInlineReduce, execMem.outputMem, dataSegsSlice, param.tag);

    for (u32 ringIndex = 0; ringIndex < multiStreamSlice.size(); ringIndex++) {
        std::vector<Slice> dataSlice;
        FillMultiRingSlice(execMem, multiStreamSlice, sliceNum, level1RankSize, level2RankSize, ringIndex, dataSlice);
        level0DataSegsSlice.push_back(dataSlice);
    }
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingFor91093Executor::CalUserMemDataSegsSlice(
    const ExecMem& execMem, const std::vector<std::vector<Slice>>& level0DataSegsSlice,
    const std::vector<std::vector<Slice>>& multiStreamSlice, const OpParam& param, u32 ringNum, u32 sliceNum,
    u32 level1RankSize, u32 level2RankSize, HcclDataType dataType, u32 perDataSize, HcomCollOpInfo* opInfoPtr,
    bool disableDMAReduce, std::vector<std::vector<Slice>>& multRingsUserMemSlice)
{
    if (isReduceScatterV_) {
        return CalUserMemDataSegsSliceV(
            execMem, param, ringNum, sliceNum, level1RankSize, level2RankSize, dataType, multRingsUserMemSlice);
    }
    CHK_PRT_RET(
        0 < param.DataDes.strideCount && param.DataDes.strideCount < param.DataDes.count,
        HCCL_ERROR(
            "[CollReduceScatterRingFor91093Executor][KernelRun]strideCount[%llu] is smaller than opCount[%llu]",
            param.DataDes.strideCount, param.DataDes.count),
        HCCL_E_PARA);
    HCCL_DEBUG(
        "[CollReduceScatterRingFor91093Executor][KernelRun]strideCount[%llu], opCount[%llu]", param.DataDes.strideCount,
        param.DataDes.count);

    u32 level0RankSize = logicalLevel0CommInfo_.localRankSize;
    bool ARSFlag = topoMatcher_->GetARSFlag();
    bool ARSDoubleRing = (ARSFlag && (level0RankSize > FACTOR_TWO) && topoAttr_.isARSDoubleRing);

    if (opInfoPtr == nullptr
        && (!(
            (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING || ARSDoubleRing)
            && (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB || disableDMAReduce)))) {
        multRingsUserMemSlice = level0DataSegsSlice;
        // 图模式，根据strideCount更新slice的offset
        if (param.DataDes.strideCount != 0) {
            CHK_RET(UpdateOffsetBasedOnStrideCount(param, multRingsUserMemSlice));
        }
    } else {
        for (u32 ringIndex = 0; ringIndex < level0DataSegsSlice.size(); ringIndex++) {
            std::vector<Slice> level1UserMemSlice;
            for (auto& cclSlice : level0DataSegsSlice[ringIndex]) {
                Slice tmpSlice;
                u64 count = (param.DataDes.strideCount == 0) ? param.DataDes.count : param.DataDes.strideCount;
                tmpSlice.size = cclSlice.size;
                CHK_PRT_RET(
                    execMem.outputMem.size() == 0,
                    HCCL_ERROR(
                        "[CollReduceScatterRingFor91093Executor][KernelRun]cclout memsize[%llu] is zero",
                        execMem.outputMem.size()),
                    HCCL_E_PARA);
                tmpSlice.offset = (cclSlice.offset / execMem.outputMem.size()) * count * perDataSize
                                  + multiStreamSlice[ringIndex][0].offset;
                level1UserMemSlice.push_back(tmpSlice);
                HCCL_DEBUG(
                    "rank[%u], ringIndex[%u], tmpSlice.offset=[%llu], size=[%llu]", topoAttr_.userRank, ringIndex,
                    tmpSlice.offset, tmpSlice.size);
            }
            multRingsUserMemSlice.push_back(level1UserMemSlice);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingFor91093Executor::CalLevel1DataSegsSlice(
    const ExecMem& execMem, const OpParam& param, CommPlane commPlaneLevel, const u32& commIndex, u32 sliceNum,
    u32 level1RankSize, u32 level2RankSize, u32 perDataSize, std::vector<Slice>& level1DataSegsSlice)
{
    if (isReduceScatterV_) {
        return CalLevel1DataSegsSliceV(
            param, commPlaneLevel, commIndex, sliceNum, level1RankSize, level2RankSize, perDataSize,
            level1DataSegsSlice);
    }
    for (u32 i = 0; i < level1RankSize; i++) {
        Slice sliceTemp;
        u32 level1UserRank;
        CHK_RET(GetUserRankByRank(commPlaneLevel, commIndex, i, level1UserRank));
        if (level2RankSize <= 1) {
            sliceTemp.size = execMem.outputMem.size();
            sliceTemp.offset = level1UserRank * execMem.outputMem.size();
            level1DataSegsSlice.push_back(sliceTemp);
            HCCL_DEBUG(
                "rank[%u], level1DataSegsSlice[%u].offset=%llu, size=[%llu]", topoAttr_.userRank, i, sliceTemp.offset,
                sliceTemp.size);
        } else {
            for (u32 level2Idx = 0; level2Idx < level2RankSize; level2Idx++) {
                sliceTemp.size = execMem.outputMem.size();
                sliceTemp.offset = (level1UserRank % (level1RankSize * sliceNum)) * execMem.outputMem.size()
                                   + level2Idx * sliceNum * level1RankSize * execMem.outputMem.size();
                level1DataSegsSlice.push_back(sliceTemp);
                HCCL_DEBUG(
                    "rank[%u], level1DataSegsSlice[%u].offset=%llu, size=[%llu]", topoAttr_.userRank, i,
                    sliceTemp.offset, sliceTemp.size);
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingFor91093Executor::GetLevelCommInfo()
{
    logicalLevel0plane_ = COMM_LEVEL0;
    CHK_RET(CheckCommSize(logicalLevel0plane_, COMM_INDEX_0 + 1));
    logicalLevel0CommInfo_ = GetSubCommInfo(logicalLevel0plane_, COMM_INDEX_0);
    u32 commIndex = logicalLevel0CommInfo_.localRank;
    bool isSelectAHC
        = (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC
           || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE);
    logicalLevel1plane_ = isSelectAHC ? COMM_LEVEL1_AHC : COMM_LEVEL1;
    CHK_RET(CheckCommSize(logicalLevel1plane_, commIndex + 1));
    logicalLevel1CommInfo_ = GetSubCommInfo(logicalLevel1plane_, commIndex);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingFor91093Executor::CalLevel2DataSegsSlice(
    const ExecMem& execMem, const OpParam& param, u32 level2RankSize, u32 perDataSize,
    std::vector<Slice>& level2DataSegsSlice)
{
    if (isReduceScatterV_) {
        return CalLevel2DataSegsSliceV(param, level2RankSize, perDataSize, level2DataSegsSlice);
    }
    Slice sliceTemp;
    for (u32 i = 0; i < level2RankSize; i++) {
        sliceTemp.size = execMem.outputMem.size();
        u32 level2UserRank;
        CHK_RET(GetUserRankByRank(COMM_LEVEL2, COMM_INDEX_0, i, level2UserRank));
        sliceTemp.offset = level2UserRank * execMem.outputMem.size();
        level2DataSegsSlice.push_back(sliceTemp);
        HCCL_DEBUG(
            "rank[%u], level2DataSegsSlice[%u].offset=%llu, size=[%llu], level2RankSize[%u]", topoAttr_.userRank, i,
            sliceTemp.offset, sliceTemp.size, level2RankSize);
    }
    return HCCL_SUCCESS;
}

void CollReduceScatterRingFor91093Executor::PrepareLevel0Slices(
    const OpParam& param, u32 sliceNum, u32 level1RankSize, u32 level1Index, u32 level2Index, u32 perDataSize,
    std::vector<Slice>& cclSegSlices)
{
    const auto* counts = static_cast<u64*>(param.VDataDes.counts);
    // 根据counts和displace计算每个rank的数据范围
    // cclSlices里的offset是cclBuffer范围内的偏移，就地计算得出，不考虑displs
    const u32 level1Rank = level2Index * level1RankSize * sliceNum + level1Index * sliceNum;
    u64 displace = std::accumulate(counts, counts + level1Rank, 0ULL);
    for (auto rank = 0U; rank < sliceNum; ++rank) {
        const u32 idx = level1Rank + rank;
        Slice slice;
        slice.size = counts[idx] * perDataSize;
        slice.offset = displace * perDataSize;
        cclSegSlices.emplace_back(slice);
        displace += counts[idx];
    }
}

void CollReduceScatterRingFor91093Executor::PrepareLevel0UserSlices(
    const OpParam& param, u32 sliceNum, u32 level1RankSize, u32 level1Index, u32 level2Index, u32 perDataSize,
    std::vector<Slice>& userSegSlices)
{
    const auto* counts = static_cast<u64*>(param.VDataDes.counts);
    const auto* displsPtr = static_cast<const u64*>(param.VDataDes.displs);
    const u32 level1Rank = level2Index * level1RankSize * sliceNum + level1Index * sliceNum;
    // 根据counts和displace计算每个rank的数据范围
    // userSlices里的offset是user input的偏移，使用传入的displs算得
    for (auto rank = 0U; rank < sliceNum; ++rank) {
        const u32 idx = level1Rank + rank;
        Slice slice;
        slice.size = counts[idx] * perDataSize;
        slice.offset = displsPtr[idx] * perDataSize;
        userSegSlices.emplace_back(std::move(slice));
    }
}

bool CollReduceScatterRingFor91093Executor::IsCceReduceAligned(const std::vector<Slice>& dataSlices) const
{
    for (const auto& slice : dataSlices) {
        if (slice.size % CCE_REDUCE_ALIGN_SIZE != 0) {
            return false;
        }
    }
    return true;
}

HcclResult CollReduceScatterRingFor91093Executor::FillMultiRingSliceV(
    const ExecMem& execMem, const OpParam& param, u32 ringNum, u32 sliceNum, u32 level1RankSize, u32 level2RankSize,
    HcclDataType dataType, std::vector<std::vector<Slice>>& level0DataSegsSlice,
    std::vector<std::vector<std::vector<Slice>>>& serverSlices, const Level0SlicesCalculator& calcLevel0Slices)
{
    bool isInlineReduce
        = IsSupportSDMAReduce(execMem.inputMem.ptr(), execMem.scratchMem.ptr(), dataType, param.reduceType);
    bool useInlineReduce = isInlineReduce && algoAttr_.inlineReduceSwitchOn;
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(dataType, perDataSize));
    for (u32 i = 0; i < level2RankSize; i++) {
        for (u32 j = 0; j < level1RankSize; j++) {
            std::vector<Slice> dataSegsSlice; // 数据分成rank size份，每份的起始偏移和大小
            calcLevel0Slices(param, sliceNum, level1RankSize, j, i, perDataSize, dataSegsSlice);

            std::vector<std::vector<Slice>> multiStreamSlices;
            // 再将每个 slice 划分为 ringNum 份
            if (ringNum == LEVEL0_PLANE_NUM_IN_8PRING) {
                if (useInlineReduce) {
                    multiStreamSlices = PrepareMultiRingSlice(dataSegsSlice, param.tag);
                } else if (IsCceReduceAligned(dataSegsSlice)) {
                    multiStreamSlices = PrepareMultiRingSlice(dataSegsSlice, param.tag);
                } else {
                    multiStreamSlices = PrepareMultiRingSlice(dataSegsSlice, param.tag, true);
                }
            } else if (ringNum == LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE) {
                // 双环场景，需要传入正确的 niclist (不涉及网口裁剪)
                if (useInlineReduce) {
                    multiStreamSlices = PrepareMultiRingSlice(dataSegsSlice, param.tag, false, topoAttr_.nicList);
                } else if (IsCceReduceAligned(dataSegsSlice)) {
                    multiStreamSlices = PrepareMultiRingSlice(dataSegsSlice, param.tag, false, topoAttr_.nicList);
                } else {
                    multiStreamSlices = PrepareMultiRingSlice(dataSegsSlice, param.tag, true, topoAttr_.nicList);
                }
            } else {
                multiStreamSlices.push_back(dataSegsSlice);
            }
            serverSlices.push_back(multiStreamSlices);
        }
    }
    level0DataSegsSlice.resize(ringNum);
    for (u32 level0Idx = 0; level0Idx < sliceNum; level0Idx++) {
        for (u32 level2Idx = 0; level2Idx < level2RankSize; level2Idx++) {
            for (u32 level1Idx = 0; level1Idx < level1RankSize; level1Idx++) {
                u32 serverIdx = level2Idx * level1RankSize + level1Idx;
                const auto& multiStreamSlices = serverSlices[serverIdx];
                for (u32 ringIndex = 0; ringIndex < multiStreamSlices.size(); ringIndex++) {
                    const auto& slice = multiStreamSlices[ringIndex][level0Idx];
                    level0DataSegsSlice[ringIndex].push_back(slice);
                    HCCL_DEBUG(
                        "[RSV]rank[%u], level0[%u]level2[%u]level1[%u], ringIndex[%u] slice.offset=[%llu], "
                        "size=[%llu]",
                        topoAttr_.userRank, level0Idx, level2Idx, level1Idx, ringIndex, slice.offset, slice.size);
                }
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingFor91093Executor::CalUserMemDataSegsSliceV(
    const ExecMem& execMem, const OpParam& param, u32 ringNum, u32 sliceNum, u32 level1RankSize, u32 level2RankSize,
    HcclDataType dataType, std::vector<std::vector<Slice>>& multRingsUserMemSlice)
{
    std::vector<std::vector<std::vector<Slice>>> serverSlices;
    CHK_RET(FillMultiRingSliceV(
        execMem, param, ringNum, sliceNum, level1RankSize, level2RankSize, dataType, multRingsUserMemSlice,
        serverSlices, PrepareLevel0UserSlices));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingFor91093Executor::CalLevel0DataSegsSliceV(
    const ExecMem& execMem, std::vector<std::vector<Slice>>& multiStreamSlice, const OpParam& param, u32 ringNum,
    u32 sliceNum, u32 level1RankSize, u32 level2RankSize, HcclDataType dataType,
    std::vector<std::vector<Slice>>& level0DataSegsSlice)
{
    std::vector<std::vector<std::vector<Slice>>> serverSlices;
    CHK_RET(FillMultiRingSliceV(
        execMem, param, ringNum, sliceNum, level1RankSize, level2RankSize, dataType, level0DataSegsSlice, serverSlices,
        PrepareLevel0Slices));
    multiStreamSlice = serverSlices[0];
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingFor91093Executor::CalLevel1DataSegsSliceV(
    const OpParam& param, CommPlane commPlaneLevel, const u32& commIndex, u32 sliceNum, u32 level1RankSize,
    u32 level2RankSize, u32 perDataSize, std::vector<Slice>& level1DataSegsSlice)
{
    const auto* counts = static_cast<u64*>(param.VDataDes.counts);
    for (u32 i = 0; i < level1RankSize; i++) {
        Slice sliceTemp;
        u32 level1UserRank;
        CHK_RET(GetUserRankByRank(commPlaneLevel, commIndex, i, level1UserRank));
        if (level2RankSize <= 1) {
            sliceTemp.size = counts[level1UserRank] * perDataSize;
            sliceTemp.offset = std::accumulate(counts, counts + level1UserRank, 0ULL) * perDataSize;
            level1DataSegsSlice.push_back(sliceTemp);
            HCCL_DEBUG(
                "[RSV]rank[%u], level1UserRank[%u], level1DataSegsSlice[%u].offset=%llu, size=[%llu]",
                topoAttr_.userRank, level1UserRank, i, sliceTemp.offset, sliceTemp.size);
        } else {
            for (u32 level2Idx = 0; level2Idx < level2RankSize; level2Idx++) {
                const u32 ranksPerServer = level1RankSize * sliceNum;
                const u32 level2UserRank = level2Idx * ranksPerServer + level1UserRank % ranksPerServer;
                sliceTemp.size = counts[level2UserRank] * perDataSize;
                sliceTemp.offset = std::accumulate(counts, counts + level2UserRank, 0ULL) * perDataSize;
                level1DataSegsSlice.push_back(sliceTemp);
                HCCL_DEBUG(
                    "[RSV]rank[%u], level2UserRank[%u], level1DataSegsSlice[%u].offset=%llu, size=[%llu]",
                    topoAttr_.userRank, level2UserRank, i, sliceTemp.offset, sliceTemp.size);
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingFor91093Executor::CalLevel2DataSegsSliceV(
    const OpParam& param, u32 level2RankSize, u32 perDataSize, std::vector<Slice>& level2DataSegsSlice)
{
    const auto* counts = static_cast<u64*>(param.VDataDes.counts);
    Slice sliceTemp;
    for (u32 i = 0; i < level2RankSize; i++) {
        u32 level2UserRank;
        CHK_RET(GetUserRankByRank(COMM_LEVEL2, COMM_INDEX_0, i, level2UserRank));
        sliceTemp.size = counts[level2UserRank] * perDataSize;
        sliceTemp.offset = std::accumulate(counts, counts + level2UserRank, 0ULL) * perDataSize;
        level2DataSegsSlice.push_back(sliceTemp);
        HCCL_DEBUG(
            "[RSV]rank[%u], level2UserRank[%u], level2DataSegsSlice[%u].offset=%llu, size=[%llu]", topoAttr_.userRank,
            level2UserRank, i, sliceTemp.offset, sliceTemp.size);
    }
    return HCCL_SUCCESS;
}

HcomCollOpInfo
CollReduceScatterRingFor91093Executor::GetHcomCollOpInfo(const OpParam& param, const ExecMem& execMem) const
{
    const u64 count = param.GetDataCount(topoAttr_.userRank);
    const HcclDataType dataType = param.GetDataType();
    const u64 strideCount = param.GetStrideCount();
    HcomCollOpInfo opInfo
        = {"", execMem.inputPtr, execMem.outputPtr, count, dataType, param.root, param.reduceType, strideCount};
    HCCL_DEBUG(
        "[CollReduceScatterRingFor91093Executor][KernelRun] execMem.inputPtr[%p], execMem.outputPtr[%p], "
        "execMem.inputMem[%p], execMem.outputMem[%p], strideCount[%llu]",
        execMem.inputPtr, execMem.outputPtr, execMem.inputMem.ptr(), execMem.outputMem.ptr(), strideCount);
    return opInfo;
}

u64 CollReduceScatterRingFor91093Executor::CalcSrcMemOffset(
    const ExecMem& execMem, const OpParam& param, u32 perDataSize) const
{
    if (isReduceScatterV_) {
        const auto* counts = static_cast<u64*>(param.VDataDes.counts);
        return std::accumulate(counts, counts + topoAttr_.userRank, 0ULL) * perDataSize;
    }
    return topoAttr_.userRank * execMem.outputMem.size();
}

HcclResult CollReduceScatterRingFor91093Executor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] executor starts, rsv[%u]", __func__, isReduceScatterV_);
    CHK_RET(GetLevelCommInfo()); // 获取通信域
    u32 perDataSize = 0;
    const HcclDataType dataType = param.GetDataType();
    CHK_RET(SalGetDataTypeSize(dataType, perDataSize));

    u32 ringNum;
    u32 level0RankSize = logicalLevel0CommInfo_.localRankSize;
    bool ARSFlag = topoMatcher_->GetARSFlag();
    bool ARSDoubleRing = (ARSFlag && (level0RankSize > FACTOR_TWO) && topoAttr_.isARSDoubleRing);
    u32 sliceNum = logicalLevel0CommInfo_.localRankSize;
    u32 commIndex = logicalLevel0CommInfo_.localRank;

    if ((topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING && !IsUnifiedMarch(param) && !ARSFlag) || ARSDoubleRing) {
        ringNum = LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE;
    } else {
        ringNum = LEVEL0_PLANE_NUM_IN_NPRING_SINGLE;
    }

    bool isSelectAHC
        = (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC
           || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE);

    SubCommInfo level2CommInfo;
    if (isSelectAHC) {
        level2CommInfo = logicalLevel1CommInfo_;
        level2CommInfo.localRankSize = 1; // AHC bypass level2
    } else {
        CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
        level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
    }
    const u32 level2RankSize = level2CommInfo.localRankSize;
    const u32 level1RankSize = logicalLevel1CommInfo_.localRankSize;

    // 节点内reduce scatter
    CHK_RET(ActiveSlaveStreams(param.stream));

    // 计算slice
    std::vector<std::vector<Slice>> multiStreamSlice; // 每个stream使用的数据基于用户buffer的偏移
    std::vector<std::vector<Slice>> level0DataSegsSlice;
    CalLevel0DataSegsSlice(
        execMem, multiStreamSlice, param, ringNum, sliceNum, level1RankSize, level2RankSize, dataType,
        level0DataSegsSlice);

    HcomCollOpInfo opInfo = GetHcomCollOpInfo(param, execMem);
    HcomCollOpInfo* opInfoPtr = nullptr;
    if (DMAReduceFlag_) {
        opInfoPtr = &opInfo;
    }

    bool disableDMAReduce = algOpContext_.opRetryHandler.retryEnable
                            && (algOpContext_.opRetryHandler.inPlaceSupportRetryStatus
                                    == InplaceSupportRetryStatus::RETRY_1_ALLOW_NO_DMA_REDUCE_CASE1
                                || algOpContext_.opRetryHandler.inPlaceSupportRetryStatus
                                       == InplaceSupportRetryStatus::RETRY_1_ALLOW_NO_DMA_REDUCE_CASE2);
    std::vector<std::vector<Slice>> multRingsUserMemSlice;
    CalUserMemDataSegsSlice(
        execMem, level0DataSegsSlice, multiStreamSlice, param, ringNum, sliceNum, level1RankSize, level2RankSize,
        dataType, perDataSize, opInfoPtr, disableDMAReduce, multRingsUserMemSlice);

    // 区分消减拷贝场景
    if ((topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING || ARSDoubleRing)
        && (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB)) {
        // 图模式opinfo不为空
        HcomCollOpInfo graphModeOpInfo
            = {"",
               execMem.inputMem.ptr(),
               nullptr,
               param.GetDataCount(topoAttr_.userRank),
               dataType,
               param.root,
               param.reduceType,
               param.GetStrideCount()};
        CHK_RET(RunIntraSeverReduceScatter(
            param.tag, execMem.inputMem, execMem.scratchMem, execMem.count, dataType, param.reduceType,
            level0DataSegsSlice, param.stream, PROF_STAGE_1, 0, &graphModeOpInfo, multRingsUserMemSlice,
            disableDMAReduce));
    } else if (opInfoPtr != nullptr && (level1RankSize > 1 || level2RankSize > 1)) {
        HcomCollOpInfo opInfoByReduceScatterDMAreduce = *opInfoPtr;
        opInfoByReduceScatterDMAreduce.outputAddr = nullptr;
        CHK_RET(RunIntraSeverReduceScatter(
            param.tag, execMem.inputMem, execMem.scratchMem, execMem.count, dataType, param.reduceType,
            level0DataSegsSlice, param.stream, PROF_STAGE_1, 0, &opInfoByReduceScatterDMAreduce, multRingsUserMemSlice,
            disableDMAReduce));
    } else {
        CHK_RET(RunIntraSeverReduceScatter(
            param.tag, execMem.inputMem, execMem.scratchMem, execMem.count, dataType, param.reduceType,
            level0DataSegsSlice, param.stream, PROF_STAGE_1, 0, opInfoPtr, multRingsUserMemSlice, disableDMAReduce));
    }
    // 对于单server图模式的最后一步需要把数据从ccl input拷贝到ccl output上
    if (level1RankSize == 1 && level2RankSize == 1 && opInfoPtr == nullptr) {
        const u64 offset = CalcSrcMemOffset(execMem, param, perDataSize);
        DeviceMem srcMem = execMem.inputMem.range(offset, execMem.outputMem.size());
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, execMem.outputMem, srcMem, const_cast<Stream&>(param.stream)));
    }

    if (level1RankSize > 1) {
        // 节点间做reduce scatter(ring/NHR/NB)
        u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.scratchMem, dataType, param.reduceType);
        std::unique_ptr<AlgTemplateBase> level1TempAlg;

        // 计算slice
        std::vector<Slice> level1DataSegsSlice;
        CHK_RET(CalLevel1DataSegsSlice(
            execMem, param, logicalLevel1plane_, commIndex, sliceNum, level1RankSize, level2RankSize, perDataSize,
            level1DataSegsSlice));

        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
            CHK_SMART_PTR_NULL(level1TempAlg);
            CHK_RET(level1TempAlg->Prepare(reduceAttr));
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_RING in COMM_LEVEL1", __func__);
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NB, dispatcher_);
            CHK_SMART_PTR_NULL(level1TempAlg);
            CHK_RET(level1TempAlg->Prepare(reduceAttr));
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_NB in COMM_LEVEL1", __func__);
        } else if (isSelectAHC) {
            // 获取通信域分组信息
            std::vector<std::vector<std::vector<u32>>> globalSubGroups;
            std::map<AHCConcOpType, TemplateType> ahcAlgOption;
            CHK_RET(topoMatcher_->GetGlobalSubGroups(logicalLevel1plane_, globalSubGroups));
            topoMatcher_->GetAHCAlgOption(ahcAlgOption);
            if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC) {
                level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_REDUCESCATTER_AHC, dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_AHC in COMM_LEVEL1", __func__);
            } else {
                level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_REDUCESCATTER_AHC_BROKE, dispatcher_);
                HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_AHC_BROKE in COMM_LEVEL1", __func__);
            }
            HCCL_DEBUG("[CollReduceScatterRingFor91093Executor]runAsync for COMM_LEVEL1 ends");
            CHK_SMART_PTR_NULL(level1TempAlg);
            CHK_RET(level1TempAlg->Prepare(execMem.count, globalSubGroups, ahcAlgOption));
            CHK_RET(level1TempAlg->Prepare(reduceAttr));
        } else {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NHR, dispatcher_);
            CHK_SMART_PTR_NULL(level1TempAlg);
            CHK_RET(level1TempAlg->Prepare(reduceAttr, false));
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_NHR in COMM_LEVEL1", __func__);
        }

        CHK_RET(level1TempAlg->Prepare(
            execMem.inputMem, execMem.inputMem, execMem.scratchMem, execMem.count, dataType, param.stream,
            param.reduceType, LEVEL0_BRIDGE_RANK_ID, level1DataSegsSlice));
        CHK_RET(level1TempAlg->RegisterProfiler(
            (level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + logicalLevel1CommInfo_.localRank, PROF_STAGE_2,
            HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(RunTemplate(level1TempAlg, logicalLevel1CommInfo_));
    }

    if (level2RankSize > 1) {
        /* ****************** 超节点间 reducescatter *******************************/
        u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.scratchMem, dataType, param.reduceType);

        // 计算slice
        std::vector<Slice> level2DataSegsSlice;
        CHK_RET(CalLevel2DataSegsSlice(execMem, param, level2RankSize, perDataSize, level2DataSegsSlice));

        std::unique_ptr<AlgTemplateBase> level2TempAlg;
        if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
            level2TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NB, dispatcher_);
            CHK_SMART_PTR_NULL(level2TempAlg);
            CHK_RET(level2TempAlg->Prepare(reduceAttr));
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_NB in COMM_LEVEL2", __func__);
        } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
            level2TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NHR, dispatcher_);
            CHK_SMART_PTR_NULL(level2TempAlg);
            CHK_RET(level2TempAlg->Prepare(reduceAttr, false));
            if (algoAttr_.isSupportAtomicWrite) {
                level2TempAlg->CloseBarrier();
            }
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_NHR in COMM_LEVEL2", __func__);
        } else {
            level2TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
            CHK_SMART_PTR_NULL(level2TempAlg);
            CHK_RET(level2TempAlg->Prepare(reduceAttr));
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_RING in COMM_LEVEL2", __func__);
        }

        CHK_RET(level2TempAlg->Prepare(
            execMem.inputMem, execMem.inputMem, execMem.scratchMem, execMem.count, dataType, param.stream,
            param.reduceType, LEVEL0_BRIDGE_RANK_ID, level2DataSegsSlice));
        CHK_RET(level2TempAlg->RegisterProfiler(
            (level2RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank, PROF_STAGE_2,
            HCCL_EXEC_STEP_NOT_SET, param.stream));
        CHK_RET(RunTemplate(level2TempAlg, level2CommInfo));
    }

    if (level1RankSize > 1 || level2RankSize > 1) {
        // 区分消减拷贝场景（消减拷贝数据需要拷贝到user output上）
        const u64 offset = CalcSrcMemOffset(execMem, param, perDataSize);
        DeviceMem srcMem = execMem.inputMem.range(offset, execMem.outputMem.size());
        if (opInfoPtr != nullptr) {
            DeviceMem dstMem = DeviceMem::create(static_cast<u8*>(opInfoPtr->outputAddr), execMem.outputMem.size());
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
        } else {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, execMem.outputMem, srcMem, const_cast<Stream&>(param.stream)));
        }
    }

    HCCL_INFO("ReduceScatter ring run success, rsv[%u]", isReduceScatterV_);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingFor91093Executor::Getlevel1CommRank(SubCommInfo& level1CommInfo)
{
    bool isSelectAHC
        = (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC
           || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE);

    if (isSelectAHC) {
        CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
        SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

        u32 commIndex = level0CommInfo.localRank;

        CommPlane commPlaneLevel1 = isSelectAHC ? COMM_LEVEL1_AHC : COMM_LEVEL1;
        CHK_RET(CheckCommSize(commPlaneLevel1, commIndex + 1));
        level1CommInfo = GetSubCommInfo(commPlaneLevel1, commIndex);
        return HCCL_SUCCESS;
    }

    if (CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1) != HCCL_SUCCESS) {
        return HCCL_E_UNAVAIL;
    }
    level1CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);

    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterRingFor91093Executor::SelectTempAlg(
    std::unique_ptr<AlgTemplateBase>& level1TempAlg, u32 level1RankSize)
{
    bool isSelectAHC
        = (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC
           || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE);
    HCCL_DEBUG("[CollReduceScatterRingFor91093Executor]SelectTempAlg begins");
    if (isSelectAHC) {
        CommPlane commPlaneLevel1 = COMM_LEVEL1_AHC;
        // 获取通信域分组信息
        std::vector<std::vector<std::vector<u32>>> globalSubGroups;
        std::map<AHCConcOpType, TemplateType> ahcAlgOption;
        CHK_RET(topoMatcher_->GetGlobalSubGroups(commPlaneLevel1, globalSubGroups));
        topoMatcher_->GetAHCAlgOption(ahcAlgOption);
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_AHC, dispatcher_);
            HCCL_INFO("reducescatter ring: using ahc algo inter-server.");
        } else {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_AHC_BROKE, dispatcher_);
            HCCL_INFO("reducescatter ring: using ahc-broke algo inter-server.");
        }
        CHK_SMART_PTR_NULL(level1TempAlg);
        CHK_RET(level1TempAlg->Prepare(NSLBDP_MIN_COUNT, globalSubGroups, ahcAlgOption));
        return HCCL_SUCCESS;
    }
    if (level1RankSize > 1) {
        if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NB, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_NB in COMM_LEVEL2", __func__);
            CHK_SMART_PTR_NULL(level1TempAlg);
        } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NHR, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_NHR in COMM_LEVEL2", __func__);
            CHK_SMART_PTR_NULL(level1TempAlg);
        } else {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_RING in COMM_LEVEL2", __func__);
            CHK_SMART_PTR_NULL(level1TempAlg);
        }
        return HCCL_SUCCESS;
    }
    return HCCL_E_UNAVAIL;
}

REGISTER_EXEC("ReduceScatterRingFor91093Executor", ReduceScatterRingFor91093, CollReduceScatterRingFor91093Executor);
} // namespace hccl

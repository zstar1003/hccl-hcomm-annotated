/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_mix_executor.h"
#include "alg_template_register.h"

namespace hccl {
CollReduceScatterMixExecutor::CollReduceScatterMixExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollReduceScatterExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
                     && topoAttr_.deviceType == DevType::DEV_TYPE_910_93;
}

void CollReduceScatterMixExecutor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;
    aicpuUnfoldMode_ = param.aicpuUnfoldMode;

    // 是否需要scratch memory
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && isSupportSDMAReduce_
        && IsSupportRDMAReduce(param.DataDes.dataType, param.reduceType)) {
        scratchMemFlag_ = false;
    } else {
        scratchMemFlag_ = true;
    }

    // 记录图模式总数据量
    totalSize_ = topoAttr_.userRankSize * param.DataDes.count * SIZE_TABLE[param.DataDes.dataType];

    // 910B 图模式非确定计算，inlineReduce使能，MESH拓扑场景下，创建一个mesh平面
    bool isInlineReduce
        = IsSupportSDMAReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType);
    meshSinglePlane_ = (topoAttr_.deviceType == DevType::DEV_TYPE_910B)
                       && topoMatcher_->GetExternalInputHcclDeterministic() == DETERMINISTIC_DISABLE && isInlineReduce
                       && (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    bool isAlsoSupportDMAReduce = topoAttr_.deviceType == DevType::DEV_TYPE_910B && isInlineReduce
                                  && workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
                                  && topoMatcher_->GetExternalInputHcclDeterministic() == DETERMINISTIC_DISABLE
                                  && param.DataDes.dataType != HCCL_DATA_TYPE_INT64
                                  && param.reduceType != HCCL_REDUCE_PROD;
    if (isAlsoSupportDMAReduce) {
        DMAReduceFlag_ = true;
    }
}

HcclResult CollReduceScatterMixExecutor::CalcScratchMemSize(u64& scratchMemSize)
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
        "[CollReduceScatterMixExecutor][CalcScratchMemSize] tag[%s] scratchMemSize[%llu]", tag_.c_str(),
        scratchMemSize);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMixExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = 0;
    if (topoAttr_.deviceType == DevType::DEV_TYPE_910B) { // mesh
        totalStreamNum = topoAttr_.deviceNumPerAggregation;
        HCCL_DEBUG("[CollReduceScatterMixExecutor][CalcStreamNum]totalStreamNum is %u", totalStreamNum);
    } else if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93) { // dbring
        totalStreamNum
            = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING ? LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE :
                                                                 LEVEL0_PLANE_NUM_IN_NPRING_SINGLE);
        if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            totalStreamNum *= STREAM_NUM_FOR_DMAREDUCE_ONE_RING;
        }
    }

    streamNum = totalStreamNum - 1U;
    HCCL_INFO("[CollReduceScatterMixExecutor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

bool CollReduceScatterMixExecutor::IsHugeData(const u64 curSize, OpParam* param)
{
    const u64 TBE_REDUCE_MAX_COUNT = INT32_MAX;

    u64 curCount = curSize / SIZE_TABLE[param->DataDes.dataType];
    bool issupportRDMAInlineReduce = IsSupportRDMAReduce(param->DataDes.dataType, param->reduceType);
    // 这里如果CheckCommSize返回ERROR，相当于HugeData true，防止GetSubCommInfo越界
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 level0RankSize = level0CommInfo.localRankSize;

    bool hugeData = (curSize * level0RankSize / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE)
                    || (curSize > SDMA_SEND_MAX_SIZE) || ((!isSupportSDMAReduce_) && (curCount > TBE_REDUCE_MAX_COUNT))
                    || ((!issupportRDMAInlineReduce)
                        && (curCount * level0RankSize / HCCL_INTERNODE_MAX_DATA_RATE > TBE_REDUCE_MAX_COUNT));

    return hugeData;
}

bool CollReduceScatterMixExecutor::IsSmallData(const u64 totalSize, [[maybe_unused]] const u64 curSize)
{
    bool smallData = totalSize <= HCCL_SMALL_COUNT_32_KB;
    return smallData;
}

HcclResult CollReduceScatterMixExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));

    // mix在server间使用NHR通信域，并在多机A+X场景下当未设置使用RDMA时，默认使用RDMA
    std::vector<SingleSubCommTransport>& commTransportLevel1 = opTransport[COMM_LEVEL1];
    for (u32 ringIndex = 0; ringIndex < commTransportLevel1.size(); ringIndex++) {
        for (auto& transportRequest : commTransportLevel1[ringIndex].transportRequests) {
            transportRequest.isUsedRdma = true;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMixExecutor::CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType)
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
        "[CollReduceScatterMixExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]", tag_.c_str(),
        inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMixExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    if (topoAttr_.deviceType == DevType::DEV_TYPE_910B) {
        CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
        commParaLevel0.meshSinglePlane = meshSinglePlane_;
        CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    } else if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
        CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
        CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    }

    return HCCL_SUCCESS;
}

u64 CollReduceScatterMixExecutor::CalcLoopMaxCount(const u32 unitSize)
{
    u64 maxCountPerLoop = inCCLbufferSize_ / (topoAttr_.userRankSize * unitSize);
    return maxCountPerLoop;
}

void CollReduceScatterMixExecutor::CalLevel0DataSegsSlice(
    const ExecMem& execMem, const std::vector<std::vector<Slice>>& multiStreamSlice, u32 sliceNum, u32 level1RankSize,
    std::vector<std::vector<Slice>>& level0DataSegsSlice)
{
    for (u32 ringIndex = 0; ringIndex < multiStreamSlice.size(); ringIndex++) {
        std::vector<Slice> dataSlice;
        FillMultiRingSlice(execMem, multiStreamSlice, sliceNum, level1RankSize, ringIndex, dataSlice);
        level0DataSegsSlice.push_back(dataSlice);
    }
}

void CollReduceScatterMixExecutor::FillMultiRingSlice(
    const ExecMem& execMem, const std::vector<std::vector<Slice>>& multiStreamSlice, u32 sliceNum, u32 level1RankSize,
    const u32 ringIndex, std::vector<Slice>& dataSlice)
{
    for (u32 level0Idx = 0; level0Idx < sliceNum; level0Idx++) {
        Slice sliceTemp;
        for (u32 level1Idx = 0; level1Idx < level1RankSize; level1Idx++) {
            sliceTemp.size = multiStreamSlice[ringIndex][level0Idx].size;
            sliceTemp.offset
                = multiStreamSlice[ringIndex][level0Idx].offset + level1Idx * sliceNum * execMem.outputMem.size();
            dataSlice.push_back(sliceTemp);
            HCCL_DEBUG(
                "rank[%u] sliceTemp.size[%zu], sliceTemp.offset[%llu]", topoAttr_.userRank, sliceTemp.size,
                sliceTemp.offset);
        }
    }
}

HcclResult CollReduceScatterMixExecutor::CalLevel1DataSegsSlice(
    const ExecMem& execMem, const u32& commIndex, u32 sliceNum, u32 level1RankSize,
    std::vector<Slice>& level1DataSegsSlice)
{
    (void)sliceNum;
    for (u32 i = 0; i < level1RankSize; i++) {
        Slice sliceTemp;
        u32 level1UserRank;
        CHK_RET(GetUserRankByRank(COMM_LEVEL1, commIndex, i, level1UserRank));
        sliceTemp.size = execMem.outputMem.size();
        sliceTemp.offset = level1UserRank * execMem.outputMem.size();
        level1DataSegsSlice.push_back(sliceTemp);
        HCCL_DEBUG(
            "rank[%u], level1DataSegsSlice[%u].offset=%llu, size=[%llu]", topoAttr_.userRank, i, sliceTemp.offset,
            sliceTemp.size);
    }
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterMixExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollReduceScatterMixExecutor][KernelRun] The ReduceScatterMixExecutor starts.");
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 level0RankSize = level0CommInfo.localRankSize;
    u32 commIndex = level0CommInfo.localRank; // 找到rank所在的节点间平面

    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
    u32 serverIndex = level1CommInfo.localRank;
    u32 level1RankSize = level1CommInfo.localRankSize;
    HCCL_DEBUG(
        "ReduceScatterMixExecutor inputSize=%llu, level0RankSize=%u,commIndex=%u, level1RankSize=%u, serverIndex=%u",
        execMem.inputMem.size(), level0RankSize, commIndex, level1RankSize, serverIndex);

    HcomCollOpInfo opInfo = {"",
                             execMem.inputPtr,
                             execMem.outputPtr,
                             param.DataDes.count,
                             param.DataDes.dataType,
                             param.root,
                             param.reduceType,
                             0};
    HCCL_DEBUG(
        "[CollReduceScatterMixExecutor][KernelRun] execMem.inputPtr[%p], execMem.outputPtr[%p], "
        "execMem.inputMem[%p], execMem.outputMem[%p]",
        execMem.inputPtr, execMem.outputPtr, execMem.inputMem.ptr(), execMem.outputMem.ptr());
    HcomCollOpInfo* opInfoPtr = nullptr;
    if (DMAReduceFlag_) {
        opInfoPtr = &opInfo;
    }

    //  第一步，AI server内reduce scatter mesh/dbring
    u32 sliceNum = level0CommInfo.localRankSize;
    if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
        u32 ringNum;
        if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
            ringNum = LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE;
        } else {
            ringNum = LEVEL0_PLANE_NUM_IN_NPRING_SINGLE;
        }

        Slice sliceTemp;

        std::vector<Slice> dataSegsSlice;                 // 数据分成ranksize份，每份的起始偏移和大小
        std::vector<std::vector<Slice>> multiStreamSlice; // 每个stream使用的数据基于用户buffer的偏移

        CHK_RET(ActiveSlaveStreams(param.stream));

        // 计算slice
        std::vector<std::vector<Slice>> level0DataSegsSlice;
        bool useInlineRduce = false;
        bool isInlineReduce = IsSupportSDMAReduce(
            execMem.inputMem.ptr(), execMem.scratchMem.ptr(), param.DataDes.dataType, param.reduceType);
        useInlineRduce = isInlineReduce && algoAttr_.inlineReduceSwitchOn;
        multiStreamSlice = ReduceScatterRingSlicePrepare(
            ringNum, sliceNum, useInlineRduce, execMem.outputMem, dataSegsSlice,
            param.tag); // 2个ring，每条ring上数据的偏移和大小

        CalLevel0DataSegsSlice(execMem, multiStreamSlice, sliceNum, level1RankSize, level0DataSegsSlice);

        std::vector<std::vector<Slice>> multRingsUserMemSlice;

        if (opInfoPtr == nullptr
            && (!(
                topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING
                && workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB))) {
            multRingsUserMemSlice = level0DataSegsSlice;
        } else {
            for (u32 ringIndex = 0; ringIndex < level0DataSegsSlice.size(); ringIndex++) {
                std::vector<Slice> level1UserMemSlice;
                for (auto& cclSlice : level0DataSegsSlice[ringIndex]) {
                    Slice tmpSlice;
                    tmpSlice.size = cclSlice.size;
                    tmpSlice.offset = (cclSlice.offset / execMem.outputMem.size()) * param.DataDes.count * perDataSize
                                      + multiStreamSlice[ringIndex][0].offset;
                    level1UserMemSlice.push_back(tmpSlice);
                    HCCL_DEBUG(
                        "rank[%u], ringIndex[%u], tmpSlice.offset=[%llu], size=[%llu]", topoAttr_.userRank, ringIndex,
                        tmpSlice.offset, tmpSlice.size);
                }
                multRingsUserMemSlice.push_back(level1UserMemSlice);
            }
        }
        // 区分消减拷贝场景
        if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING
            && workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
            // 图模式opinfo不为空
            HcomCollOpInfo graphModeOpInfo
                = {"",         execMem.inputMem.ptr(), nullptr, param.DataDes.count, param.DataDes.dataType,
                   param.root, param.reduceType,       0};
            CHK_RET(MultiRingReduceScatter(
                param.tag, execMem.inputMem, execMem.scratchMem, execMem.count, param.DataDes.dataType,
                param.reduceType, level0DataSegsSlice, param.stream, PROF_STAGE_0, 0, &graphModeOpInfo,
                multRingsUserMemSlice));
        } else if (opInfoPtr != nullptr) {
            HcomCollOpInfo opInfoByReduceScatterDMAreduce = *opInfoPtr;
            opInfoByReduceScatterDMAreduce.outputAddr = nullptr;
            CHK_RET(MultiRingReduceScatter(
                param.tag, execMem.inputMem, execMem.scratchMem, execMem.count, param.DataDes.dataType,
                param.reduceType, level0DataSegsSlice, param.stream, PROF_STAGE_0, 0, &opInfoByReduceScatterDMAreduce,
                multRingsUserMemSlice));
        } else {
            CHK_RET(MultiRingReduceScatter(
                param.tag, execMem.inputMem, execMem.scratchMem, execMem.count, param.DataDes.dataType,
                param.reduceType, level0DataSegsSlice, param.stream, PROF_STAGE_0, 0, opInfoPtr,
                multRingsUserMemSlice));
        }
    } else if (topoAttr_.deviceType == DevType::DEV_TYPE_910B) {
        CHK_RET(ActiveSlaveStreams(param.stream));

        // 根据数据量算每个环上数据的偏移和大小，把做完hd的slice均分成RankSize份
        std::vector<Slice> dataSegsSlice;
        CHK_RET(PrepareReduceScatterSliceData(execMem.count, perDataSize, sliceNum, dataSegsSlice));

        if (opInfoPtr != nullptr) {
            u64 reduceAttr
                = GetReduceAttr(execMem.inputMem, execMem.scratchMem, param.DataDes.dataType, param.reduceType);
            std::unique_ptr<AlgTemplateBase> level0Executor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_REDUCESCATTER_MESH_MIX, dispatcher_);

            CHK_SMART_PTR_NULL(level0Executor);
            CHK_RET(level0Executor->Prepare(
                execMem.inputMem, execMem.inputMem, execMem.scratchMem, execMem.count, param.DataDes.dataType,
                param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID, dataSegsSlice, 0, reduceAttr,
                algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux, serverIndex,
                level1RankSize, opInfoPtr));
            CHK_RET(level0Executor->RegisterProfiler(
                (level0RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commIndex, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET,
                param.stream));
            CHK_RET(RunTemplate(level0Executor, level0CommInfo));
        } else {
            std::vector<std::vector<Slice>> multiStreamSlice; // 每个stream使用的数据基于用户buffer的偏移
            // mesh算法stream数量为rank数减1
            CHK_RET(AlgTemplateBase::PrepareSliceMeshStreams(dataSegsSlice, sliceNum - 1, multiStreamSlice));

            // 计算slice
            std::vector<std::vector<Slice>> level0DataSegsSlice;
            CalLevel0DataSegsSlice(execMem, multiStreamSlice, sliceNum, level1RankSize, level0DataSegsSlice);

            CHK_RET(MultiStreamReduceScatterMesh(
                param.tag, execMem.inputMem, execMem.scratchMem, execMem.count, param.DataDes.dataType,
                param.reduceType, level0DataSegsSlice, param.stream, COMM_LEVEL0, 0));
        }
    }

    //  第二步，节点间reduce scatter
    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.scratchMem, param.DataDes.dataType, param.reduceType);

    std::unique_ptr<AlgTemplateBase> level1Executor;

    // 计算slice
    std::vector<Slice> level1DataSegsSlice;

    CHK_RET(CalLevel1DataSegsSlice(execMem, commIndex, sliceNum, level1RankSize, level1DataSegsSlice));

    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
        level1Executor
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_RING, dispatcher_);
        CHK_SMART_PTR_NULL(level1Executor);
        CHK_RET(level1Executor->Prepare(reduceAttr));
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_RING in COMM_LEVEL1", __func__);
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
        level1Executor
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NHR, dispatcher_);
        CHK_SMART_PTR_NULL(level1Executor);
        CHK_RET(level1Executor->Prepare(reduceAttr, false));
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_REDUCESCATTER_NHR in COMM_LEVEL1", __func__);
    } else {
        HCCL_ERROR(
            "[CollReduceScatterMixExecutor][KernelRun]ReduceScatter mix: algType[%u] is not supported.",
            algType_.algoLevel1);
        return HCCL_E_NOT_SUPPORT;
    }

    CHK_RET(level1Executor->Prepare(
        execMem.inputMem, execMem.inputMem, execMem.scratchMem, execMem.count, param.DataDes.dataType, param.stream,
        param.reduceType, LEVEL0_BRIDGE_RANK_ID, level1DataSegsSlice));
    CHK_RET(level1Executor->RegisterProfiler(
        (level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank, PROF_STAGE_1,
        HCCL_EXEC_STEP_NOT_SET, param.stream));
    CHK_RET(RunTemplate(level1Executor, level1CommInfo));

    // 区分消减拷贝场景（消减拷贝数据需要拷贝到user output上）
    DeviceMem srcMem = execMem.inputMem.range(topoAttr_.userRank * execMem.outputMem.size(), execMem.outputMem.size());
    if (opInfoPtr != nullptr) {
        DeviceMem dstMem = DeviceMem::create(static_cast<u8*>(opInfoPtr->outputAddr), execMem.outputMem.size());
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream)));
    } else {
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, execMem.outputMem, srcMem, const_cast<Stream&>(param.stream)));
    }

    HCCL_INFO("[CollReduceScatterMixExecutor][KernelRun]ReduceScatter mix run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceScatterMixExecutor", ReduceScatterMix, CollReduceScatterMixExecutor);
} // namespace hccl

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_gather_mix_executor.h"

namespace hccl {
CollAllGatherMixExecutor::CollAllGatherMixExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAllGatherExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
}

HcclResult CollAllGatherMixExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = 0;
    if (topoAttr_.deviceType == DevType::DEV_TYPE_910B) { // mesh
        totalStreamNum = topoAttr_.deviceNumPerAggregation;
    } else if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93) { // dbring
        totalStreamNum
            = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING ? LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE :
                                                                 LEVEL0_PLANE_NUM_IN_NPRING_SINGLE);
        if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            totalStreamNum *= STREAM_NUM_FOR_DMAREDUCE_ONE_RING;
        }
    }

    streamNum = totalStreamNum - 1U;
    HCCL_INFO("[CollAllGatherMixExecutor][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherMixExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
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

HcclResult CollAllGatherMixExecutor::CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO(
        "[CollAllGatherMixExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]", tag_.c_str(),
        inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherMixExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    if (topoAttr_.deviceType == DevType::DEV_TYPE_910B) {
        CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
        commParaLevel0.meshSinglePlane = topoMatcher_->GetExternalInputHcclDeterministic() == DETERMINISTIC_DISABLE
                                         && (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
        CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    } else if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
        CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
        CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    }

    return HCCL_SUCCESS;
}

u64 CollAllGatherMixExecutor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    u64 maxCountPerLoop = cclBuffSize / (topoAttr_.userRankSize * unitSize);
    return maxCountPerLoop;
}

HcclResult CollAllGatherMixExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAllGatherMixExecutor][KernelRun] The AllGatherMixExecutor starts.");
    u32 perDataSize = SIZE_TABLE[param.DataDes.dataType];

    // 获取子通信域信息
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 level0RankSize = level0CommInfo.localRankSize;
    u32 commIndex = level0CommInfo.localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
    u32 serverIndex = level1CommInfo.localRank;
    u32 level1RankSize = level1CommInfo.localRankSize;
    HCCL_DEBUG(
        "inputSize=%llu, level0RankSize=%u, commIndex=%u, level1RankSize=%u, serverIndex=%u", execMem.inputMem.size(),
        level0RankSize, commIndex, level1RankSize, serverIndex);

    u64 inputMemSize = execMem.inputMem.size();
    u64 baseOffset = serverIndex * inputMemSize * level0RankSize;
    u64 level0Offset = commIndex * inputMemSize;
    DeviceMem dstMem = execMem.outputMem.range(baseOffset + level0Offset, inputMemSize);
    CHK_SMART_PTR_NULL(dstMem);

    HcomCollOpInfo opInfo = {
        "", execMem.inputPtr, execMem.outputPtr, param.DataDes.count, param.DataDes.dataType, 0, HCCL_REDUCE_RESERVED,
        0};
    HcomCollOpInfo* opInfoPtr = nullptr;
    // 图模式opinfo不为空，但需要将数据从ccl input拷贝到ccl output上
    HcclResult ret = HCCL_SUCCESS;
    if (!DMAReduceFlag_) {
        ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, execMem.inputMem, const_cast<Stream&>(param.stream));
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[CollAllGatherMixExecutor][KernelRun]AllGather mix memcpy Failed, Offset[%llu], Size[%llu]",
                baseOffset + level0Offset, inputMemSize),
            ret);
    } else {
        opInfoPtr = &opInfo;
        // 先做server间算法，带有消减拷贝场景数据需要从user input取，拷贝到ccl output上
        DeviceMem srcMem = DeviceMem::create(static_cast<u8*>(execMem.inputPtr), inputMemSize);
        ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream));
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[CollAllGatherMixExecutor][KernelRun]AllGather mix user memcpy "
                "Failed, Offset[%llu], Size[%llu]",
                baseOffset + level0Offset, inputMemSize),
            ret);
    }

    //  第一步，AI server间AllGather
    std::unique_ptr<AlgTemplateBase> level1Executor;
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
        level1Executor
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
        HCCL_INFO("[CollAllGatherMixExecutor][KernelRun]AllGather mix: using ring algo inter-server.");
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
        level1Executor
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
        HCCL_INFO("[CollAllGatherMixExecutor][KernelRun]AllGather mix: using nhr algo inter-server.");
    } else {
        HCCL_ERROR(
            "[CollAllGatherMixExecutor][KernelRun]AllGather mix: algType_[%u] is not supported.", algType_.algoLevel1);
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_SMART_PTR_NULL(level1Executor);

    // 计算slice
    std::vector<Slice> level1DataSegsSlice;
    for (u32 i = 0; i < level1RankSize; i++) {
        Slice level1Slice;
        level1Slice.size = inputMemSize;
        level1Slice.offset = (i * level0RankSize + commIndex) * inputMemSize;
        level1DataSegsSlice.push_back(level1Slice);
    }
    CHK_RET(level1Executor->Prepare(
        execMem.outputMem, execMem.outputMem, execMem.inputMem, execMem.count, param.DataDes.dataType, param.stream,
        HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, level1DataSegsSlice, 0));

    CHK_RET(level1Executor->RegisterProfiler(
        (level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + serverIndex, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET,
        param.stream));

    CHK_RET(RunTemplate(level1Executor, level1CommInfo));
    HCCL_INFO("[CollAllGatherMixExecutor][KernelRun]AllGather mix level1 AllGather run success");

    //  第二步，节点内做AllGather mesh/dbring
    if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
        std::vector<Slice> dataSegsSlice; // 数据分成ranksize份，每份的起始偏移和大小
        CHK_RET(PrepareAllgatherSlice(level0RankSize, inputMemSize, dataSegsSlice));
        std::vector<std::vector<Slice>> multRingsSliceZero; // 数据基于该rank上环0的偏移
        //  多环数据切分
        if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
            multRingsSliceZero = PrepareMultiRingSlice(dataSegsSlice, param.tag, false, topoAttr_.nicList);
        } else {
            multRingsSliceZero.push_back(dataSegsSlice);
        }
        std::vector<std::vector<Slice>> multRingsSlice;
        CHK_RET(CalculateLevel1AllgatherSlice(
            inputMemSize, level0RankSize, level1RankSize, multRingsSliceZero, multRingsSlice));

        std::vector<std::vector<Slice>> multRingsUserMemSlice;
        if (!DMAReduceFlag_) {
            multRingsUserMemSlice = multRingsSlice;
        } else {
            for (u32 ringIndex = 0; ringIndex < multRingsSlice.size(); ringIndex++) {
                std::vector<Slice> userMemSlice;
                for (auto& cclSlice : multRingsSlice[ringIndex]) {
                    Slice tmpSlice;
                    tmpSlice.size = cclSlice.size;
                    tmpSlice.offset = (cclSlice.offset / inputMemSize) * opInfo.count * perDataSize
                                      + multRingsSliceZero[ringIndex][0].offset;
                    userMemSlice.push_back(tmpSlice);
                }
                multRingsUserMemSlice.push_back(userMemSlice);
            }
        }
        CHK_RET(ActiveSlaveStreams(param.stream));
        if (DMAReduceFlag_) {
            // allgather输入放在CCL buffer上，通过设置nullptr指示要从CCL buffer获取输入
            opInfo.inputAddr = nullptr;
        }
        CHK_RET(MultiRingAllGather(
            param.tag, execMem.inputMem, execMem.outputMem, execMem.count, param.DataDes.dataType, multRingsSlice,
            param.stream, PROF_STAGE_1, 0, opInfoPtr, multRingsUserMemSlice));
    } else if (topoAttr_.deviceType == DevType::DEV_TYPE_910B) {
        CHK_RET(ActiveSlaveStreams(param.stream));

        std::unique_ptr<AlgTemplateBase> level0Executor
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_MESH_MIX, dispatcher_);
        CHK_SMART_PTR_NULL(level0Executor);
        CHK_RET(level0Executor->Prepare(
            algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux, 0, opInfoPtr, serverIndex,
            level1RankSize));
        std::vector<Slice> emptySlices;
        CHK_RET(level0Executor->Prepare(
            execMem.outputMem, execMem.outputMem, execMem.inputMem, execMem.count, param.DataDes.dataType, param.stream,
            HCCL_REDUCE_RESERVED, LEVEL0_BRIDGE_RANK_ID, emptySlices, 0));
        CHK_RET(level0Executor->RegisterProfiler(
            (level0RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commIndex, PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET,
            param.stream));
        CHK_RET(RunTemplate(level0Executor, level0CommInfo));
    }

    HCCL_INFO("[CollAllGatherMixExecutor][KernelRun]AllGather mix run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllGatherMixExecutor", AllGatherMix, CollAllGatherMixExecutor);
} // namespace hccl

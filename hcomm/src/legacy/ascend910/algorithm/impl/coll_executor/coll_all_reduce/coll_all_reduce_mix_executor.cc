/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_reduce_mix_executor.h"

namespace hccl {

CollAllReduceMixExecutor::CollAllReduceMixExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAllReduceExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
                     && topoAttr_.deviceType == DevType::DEV_TYPE_910_93;
}

void CollAllReduceMixExecutor::ParseParam(const OpParam& param)
{
    tag_ = param.tag;
    bool isInlineReduce
        = IsSupportSDMAReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType);
    meshSinglePlane_ = (topoAttr_.deviceType == DevType::DEV_TYPE_910B)
                       && topoMatcher_->GetExternalInputHcclDeterministic() == DETERMINISTIC_DISABLE && isInlineReduce
                       && (workflowMode_ != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
}

HcclResult CollAllReduceMixExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = 0;

    if (topoAttr_.deviceType == DevType::DEV_TYPE_910B) {
        totalStreamNum = topoAttr_.deviceNumPerAggregation;
    } else if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
        totalStreamNum
            = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING ? LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE :
                                                                 LEVEL0_PLANE_NUM_IN_NPRING_SINGLE);
        if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            totalStreamNum *= STREAM_NUM_FOR_DMAREDUCE_ONE_RING;
        }
    }

    streamNum = totalStreamNum - 1;
    HCCL_INFO("[CollAllReduceMixExecutor][CalcStreamNum] tag[%s] streamNum[%u].", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMixExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
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

HcclResult
CollAllReduceMixExecutor::CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO(
        "[CollAllReduceMixExecutor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d].", tag_.c_str(),
        inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllReduceMixExecutor::CalcLevel0CommInfo(
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
    HCCL_DEBUG("[CollAllReduceMixExecutor][CalcLevel0CommInfo]Calculate for Level0CommInfo success");
    return HCCL_SUCCESS;
}

bool CollAllReduceMixExecutor::IsSmallData([[maybe_unused]] const u64 totalSize, const u64 curSize)
{
    bool smallData = IsAllReduceSmallData(curSize);
    return smallData;
}

bool CollAllReduceMixExecutor::IsHugeData(const u64 curSize)
{
    bool hugeData = curSize / topoAttr_.deviceNumPerAggregation / HCCL_INTERNODE_MAX_DATA_RATE > RDMA_SEND_MAX_SIZE
                    || curSize > SDMA_SEND_MAX_SIZE;
    return hugeData;
}

HcclResult CollAllReduceMixExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAllReduceMixExecutor][Run]The CollAllReduceMixExecutor starts.");
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 sliceNum = level0CommInfo.localRankSize;

    std::vector<Slice> dataSegsSlice; // 数据分成ranksize份，每份的起始偏移和大小
    // 根据数据量计算每个环上数据的偏移和大小
    CHK_RET(AlgTemplateBase::PrepareSliceData(execMem.count, perDataSize, sliceNum, 0, dataSegsSlice));
    std::vector<std::vector<Slice>> multRingsSliceZero; // 910_93数据基于该rank上环0的偏移

    /* 三步算法step1：外层 - 节点内 reduce-scatter */
    CHK_RET(ActiveSlaveStreams(param.stream));

    if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
        //  多环数据切分
        if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
            multRingsSliceZero = PrepareMultiRingSlice(dataSegsSlice, param.tag, false, topoAttr_.nicList);
        } else {
            multRingsSliceZero.push_back(dataSegsSlice);
        }

        // 第一步的reducescatter输出放在CCL buffer上，通过设置nullptr指示不做最后一步的DMA削减动作
        HcomCollOpInfo reduceScatterOpInfo
            = {"", execMem.inputPtr, nullptr, execMem.count, param.DataDes.dataType, param.root, param.reduceType, 0};
        HcomCollOpInfo reduceScatterGraphModeOpInfo
            = {"",         execMem.inputMem.ptr(), nullptr, execMem.count, param.DataDes.dataType,
               param.root, param.reduceType,       0};
        HcomCollOpInfo* reduceScatterOpInfoPtr = nullptr;
        if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
            reduceScatterOpInfoPtr = &reduceScatterGraphModeOpInfo;
        }
        if (DMAReduceFlag_) {
            reduceScatterOpInfoPtr = &reduceScatterOpInfo;
        }
        const std::vector<std::vector<Slice>> multRingsUserMemSliceDefault = std::vector<std::vector<Slice>>(0);
        CHK_RET(MultiRingReduceScatter(
            param.tag, execMem.inputMem, execMem.outputMem, execMem.count, param.DataDes.dataType, param.reduceType,
            multRingsSliceZero, param.stream, PROF_STAGE_0, 0, reduceScatterOpInfoPtr, multRingsUserMemSliceDefault));
    } else if (topoAttr_.deviceType == DevType::DEV_TYPE_910B) {
        if (topoMatcher_->GetExternalInputHcclDeterministic() == DETERMINISTIC_DISABLE
            && (param.DataDes.dataType != HCCL_DATA_TYPE_INT64)
            && (topoAttr_.deviceType == DevType::DEV_TYPE_910B && param.reduceType != HCCL_REDUCE_PROD)) {
            CHK_RET(MultiStreamReduceScatterMeshAtomic(
                param.tag, execMem.inputMem, execMem.outputMem, execMem.count, param.DataDes.dataType, param.reduceType,
                dataSegsSlice, const_cast<Stream&>(param.stream), COMM_LEVEL0));
        } else {
            std::vector<std::vector<Slice>> multiStreamSlice; // 每个stream使用的数据基于用户buffer的偏移
            CHK_RET(AlgTemplateBase::PrepareSliceMeshStreams(dataSegsSlice, sliceNum - 1, multiStreamSlice));
            CHK_RET(MultiStreamReduceScatterMesh(
                param.tag, execMem.inputMem, execMem.outputMem, execMem.count, param.DataDes.dataType, param.reduceType,
                multiStreamSlice, const_cast<Stream&>(param.stream), COMM_LEVEL0));
        }
    }

    HCCL_INFO("[CollAllReduceMixExecutor][KernelRun]AllReduce mix stage0 run success");

    /* 三步算法step2: 内层 - 节点间 allreduce */
    u32 commIndex = level0CommInfo.localRank;
    CHK_PRT_RET(
        commIndex >= dataSegsSlice.size(),
        HCCL_ERROR(
            "[CollAllReduceMixExecutor][Run]commIndex[%u] >= dataSegsSlice size[%zu]", commIndex, dataSegsSlice.size()),
        HCCL_E_INTERNAL);

    DeviceMem allreduceInput = execMem.inputMem.range(dataSegsSlice[commIndex].offset, dataSegsSlice[commIndex].size);
    CHK_SMART_PTR_NULL(allreduceInput);
    DeviceMem allreduceOutput = execMem.outputMem.range(dataSegsSlice[commIndex].offset, dataSegsSlice[commIndex].size);
    CHK_SMART_PTR_NULL(allreduceOutput);

    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    u64 reduceAttr = GetReduceAttr(allreduceInput, allreduceOutput, param.DataDes.dataType, param.reduceType);

    std::unique_ptr<AlgTemplateBase> level1Executor;
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
        level1Executor
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_RING, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_RING in COMM_LEVEL1", __func__);
        CHK_SMART_PTR_NULL(level1Executor);
        CHK_RET(level1Executor->Prepare(reduceAttr));
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
        u64 curSize = execMem.count * perDataSize; // 单位 byte
        HCCL_DEBUG(
            "[CollAllReduceMixExecutor][KernelRun] curSize[%llu] deviceNumPerAggregation[%u] commLevel0Size[%u]",
            curSize, topoAttr_.deviceNumPerAggregation, level0CommInfo.localRankSize);
        if (curSize / topoAttr_.deviceNumPerAggregation <= NHR_ALLREDUCE_SMALL_SIZE) {
            level1Executor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_REDUCE_NHR_ONESHOT, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_REDUCE_NHR_ONESHOT in COMM_LEVEL1", __func__);
        } else {
            level1Executor
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_REDUCE_NHR, dispatcher_);
            HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_NHR in COMM_LEVEL1", __func__);
        }
        CHK_SMART_PTR_NULL(level1Executor);
        CHK_RET(level1Executor->Prepare(reduceAttr));
    } else {
        HCCL_ERROR(
            "[CollAllReduceMixExecutor][KernelRun]AllReduce mix: algType[%u] is not supported.", algType_.algoLevel1);
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_SMART_PTR_NULL(level1Executor);
    u32 rankSize = level1CommInfo.localRankSize;

    u64 level1Count = dataSegsSlice[commIndex].size / perDataSize;
    CHK_RET(level1Executor->Prepare(
        allreduceInput, allreduceOutput, allreduceOutput, level1Count, param.DataDes.dataType, param.stream,
        param.reduceType, LEVEL0_BRIDGE_RANK_ID, std::vector<Slice>(0), dataSegsSlice[commIndex].offset));
    CHK_RET(level1Executor->RegisterProfiler(
        (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank, PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET,
        param.stream));
    CHK_RET(RunTemplate(level1Executor, level1CommInfo));

    HCCL_INFO("[CollAllReduceMixExecutor][KernelRun]AllReduce mix stage1 run success.");

    /* 三步算法step3：外层 - 节点内 allgather */
    // 第三步的allgather输入放在CCL buffer上，通过设置nullptr指示要从CCL buffer获取输入
    if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
        HcomCollOpInfo allgatherOpInfo
            = {"", nullptr, execMem.outputPtr, execMem.count, param.DataDes.dataType, param.root, param.reduceType, 0};
        HcomCollOpInfo allgatherOpInfoGraphModeOpInfo = {
            "", nullptr, execMem.outputMem.ptr(), execMem.count, param.DataDes.dataType, param.root, param.reduceType,
            0};
        HcomCollOpInfo* allgatherOpInfoPtr = nullptr;
        if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
            allgatherOpInfoPtr = &allgatherOpInfoGraphModeOpInfo;
        }
        if (DMAReduceFlag_) {
            allgatherOpInfoPtr = &allgatherOpInfo;
        }
        CHK_RET(MultiRingAllGather(
            param.tag, execMem.inputMem, execMem.outputMem, level1Count, param.DataDes.dataType, multRingsSliceZero,
            param.stream, PROF_STAGE_2, 0, allgatherOpInfoPtr));
    } else if (topoAttr_.deviceType == DevType::DEV_TYPE_910B) {
        std::unique_ptr<AlgTemplateBase> level0Executor = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_GATHER_MESH_ATOMIC, dispatcher_);
        HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_GATHER_MESH_ATOMIC in COMM_LEVEL0", __func__);
        CHK_SMART_PTR_NULL(level0Executor);
        CHK_RET(level0Executor->Prepare(
            algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux, topoAttr_.userRank, nullptr,
            level0CommInfo.localRank, level0CommInfo.localRankSize));

        u32 rankSize = level0CommInfo.localRankSize;
        CHK_RET(level0Executor->Prepare(
            execMem.outputMem, execMem.outputMem, execMem.inputMem, execMem.count, param.DataDes.dataType, param.stream,
            param.reduceType, LEVEL0_BRIDGE_RANK_ID, dataSegsSlice, 0));

        CHK_RET(level0Executor->RegisterProfiler(
            (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank, PROF_STAGE_2,
            HCCL_EXEC_STEP_NOT_SET, param.stream));

        CHK_RET(RunTemplate(level0Executor, level0CommInfo));
    }

    HCCL_INFO("[CollAllReduceMixExecutor][KernelRun]AllReduce mix stage2 run success.");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllReduceMixExecutor", AllReduceMix, CollAllReduceMixExecutor);

} // namespace hccl

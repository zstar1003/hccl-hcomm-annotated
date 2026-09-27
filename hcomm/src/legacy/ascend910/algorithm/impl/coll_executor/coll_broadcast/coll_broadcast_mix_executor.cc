/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_broadcast_mix_executor.h"
#include "alg_template_register.h"

namespace hccl {

CollBroadCastMix::CollBroadCastMix(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollBroadcastExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
                     && topoAttr_.deviceType == DevType::DEV_TYPE_910_93;
}

HcclResult CollBroadCastMix::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = 0U;

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
    HCCL_INFO("[CollBroadCastMix][CalcStreamNum] tag[%s] streamNum_[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollBroadCastMix::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
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

HcclResult CollBroadCastMix::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    if (topoAttr_.deviceType == DevType::DEV_TYPE_910B) {
        CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
        commParaLevel0.meshSinglePlane = true;
        CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    } else if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
        CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
        CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    }
    return HCCL_SUCCESS;
}

HcclResult CollBroadCastMix::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollBroadCastMix][KernelRun] The CollBroadCastMix starts.");
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(param.DataDes.dataType, perDataSize));
    CHK_PRT_RET(
        perDataSize == 0,
        HCCL_ERROR(
            "[CollBroadCastMix][KernelRun]errNo[0x%016llx] datatype[%d] is invalid", HCCL_ERROR_CODE(HCCL_E_PARA),
            param.DataDes.dataType),
        HCCL_E_PARA);

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    std::vector<Slice> dataSegsSlice; // 数据分成ranksize份，每份的起始偏移和大小
    // 按ranksize得到内存切分slice数
    u32 sliceNum = level0CommInfo.localRankSize;
    HCCL_DEBUG("[CollBroadCastMix][KernelRun]sliceNum[%u]", sliceNum);
    // 将根节点数据切分成sliceNum份
    CHK_RET(AlgTemplateBase::PrepareSliceData(execMem.count, perDataSize, sliceNum, 0, dataSegsSlice));

    std::vector<std::vector<Slice>> mulRingSlice; // 910_93数据基于该rank上环0的偏移

    /* step 1: 节点内 scatter */
    if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
        u32 ringNum = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) ? LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE :
                                                                          LEVEL0_PLANE_NUM_IN_NPRING_SINGLE;

        HCCL_DEBUG("[CollBroadCastMix][KernelRun]ringNum[%u]", ringNum);
        CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));

        if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
            // 将每slice再切分成2份，按各ring的dev顺序排列
            mulRingSlice = PrepareMultiRingSlice(dataSegsSlice, param.tag, false, topoAttr_.nicList);
            CHK_PRT_RET(
                mulRingSlice.size() != ringNum,
                HCCL_ERROR(
                    "[CollBroadCastMix][KernelRun] ringNum[%u] != mulRingSlice size[%zu]", ringNum,
                    mulRingSlice.size()),
                HCCL_E_INTERNAL);
        } else {
            mulRingSlice.push_back(dataSegsSlice); // 应该offset全为0，而大小和dataSegsSlice中一样,里面的offset不使用
        }

        HcomCollOpInfo* scatterOpInfoPtr = nullptr;
        HcomCollOpInfo scatterOpInfo
            = {"",         execMem.inputPtr, nullptr, param.DataDes.count, param.DataDes.dataType,
               param.root, param.reduceType, 0};

        if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) {
            scatterOpInfoPtr = &scatterOpInfo;
            CHK_RET(ActiveSlaveStreams(param.stream));
            CHK_RET(DoubleRingScatter(
                param.tag, execMem.inputMem, execMem.outputMem, execMem.count, param.DataDes.dataType, mulRingSlice,
                param.root, param.stream, scatterOpInfoPtr));
        } else {
            if (DMAReduceFlag_) {
                scatterOpInfoPtr = &scatterOpInfo;
            }
            CHK_RET(MultiRingScatter(
                param.tag, execMem.inputMem, execMem.outputMem, execMem.count, param.DataDes.dataType, mulRingSlice,
                param.root, param.stream, scatterOpInfoPtr));
        }
    } else if (topoAttr_.deviceType == DevType::DEV_TYPE_910B) {
        std::unique_ptr<AlgTemplateBase> level0Executor
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_SCATTER_MESH, dispatcher_);
        CHK_SMART_PTR_NULL(level0Executor);
        CHK_RET(level0Executor->Prepare(level0CommInfo.localRank, level0CommInfo.localRankSize));
        level0Executor->CloseBarrier();

        /* 节点内执行器 stage0 */
        u32 rootRank = 0;
        HcclResult ret = GetRankByUserRank(COMM_LEVEL0, COMM_INDEX_0, param.root, rootRank);

        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[CollBroadCastMix][KernelRun]errNo[0x%016llx]invalid root[%u] to get userrank", HCCL_ERROR_CODE(ret),
                param.root),
            ret);

        CHK_RET(level0Executor->Prepare(
            execMem.inputMem, execMem.outputMem, execMem.outputMem, execMem.count, param.DataDes.dataType, param.stream,
            HCCL_REDUCE_RESERVED, rootRank, dataSegsSlice));

        u32 rankSize = level0CommInfo.localRankSize;
        CHK_RET(level0Executor->RegisterProfiler(
            (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank, PROF_STAGE_0,
            HCCL_EXEC_STEP_NOT_SET, param.stream));

        CHK_RET(RunTemplate(level0Executor, level0CommInfo));
    }
    HCCL_INFO("[CollBroadCastMix][KernelRun] level0-scatter run success");

    /* step 2: server间 broadcast */
    u32 commIndex = level0CommInfo.localRank;
    u64 level1DataSize = dataSegsSlice[commIndex].size;
    u64 level1DataCount = level1DataSize / perDataSize;
    HCCL_DEBUG(
        "[CollBroadCastMix][KernelRun]usrRank[%u] level1 use level1DataCount[%llu]", topoAttr_.userRank,
        level1DataCount);

    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
    HCCL_DEBUG(
        "[CollBroadCastMix][KernelRun]commIdx:%u TagCommInfo[%s].commLevel1.size():%u", commIndex, param.tag.c_str(),
        level1CommInfo.localRankSize);

    std::unique_ptr<AlgTemplateBase> level1Executor;
    u64 curSize = execMem.count * perDataSize;
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
        HCCL_DEBUG(
            "[CollBroadCastMix][KernelRun] curSize[%llu] deviceNumPerAggregation[%u] commLevel0Size[%u]", curSize,
            topoAttr_.deviceNumPerAggregation, level0CommInfo.localRankSize);
        if (curSize / topoAttr_.deviceNumPerAggregation <= NHR_BCAST_SMALL_SIZE) {
            level1Executor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_BROADCAST_NHR_ONESHOT, dispatcher_);
        } else {
            level1Executor
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_NHR, dispatcher_);
        }
        HCCL_INFO("[CollBroadCastMix][KernelRun]broadcast mix: using nhr algo inter-server.");
    } else {
        HCCL_ERROR("[CollBroadCastMix][KernelRun]broadcast mix: algType[%u] is not supported.", algType_.algoLevel1);
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_SMART_PTR_NULL(level1Executor);

    u32 subUserrankRoot = topoMatcher_->GetSubRootUserRank(topoAttr_.userRank, param.root);
    CHK_PRT_RET(
        subUserrankRoot == INVALID_VALUE_RANKID,
        HCCL_ERROR(
            "[CollBroadCastMix][KernelRun]subUserrankRoot[%u] is invalid,userRank[%u],root[%u]", subUserrankRoot,
            topoAttr_.userRank, param.root),
        HCCL_E_INTERNAL);

    u32 planeRoot = 0;
    CHK_RET(GetRankByUserRank(COMM_LEVEL1, commIndex, subUserrankRoot, planeRoot));

    CHK_RET(level1Executor->Prepare(
        execMem.inputMem, execMem.inputMem, execMem.outputMem, level1DataCount, param.DataDes.dataType, param.stream,
        HCCL_REDUCE_RESERVED, planeRoot, std::vector<Slice>(0), dataSegsSlice[commIndex].offset));

    u32 rankSize = level1CommInfo.localRankSize;
    CHK_RET(level1Executor->RegisterProfiler(
        (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level1CommInfo.localRank, PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET,
        param.stream));

    CHK_RET(RunTemplate(level1Executor, level1CommInfo));

    HCCL_INFO("[CollBroadCastMix][KernelRun]Broadcast mix stage1 run success");

    /* step 3: 节点内 allgather */
    if (topoAttr_.deviceType == DevType::DEV_TYPE_910_93) {
        HcomCollOpInfo allgatherOpInfo = {
            "", nullptr, execMem.outputPtr, param.DataDes.count, param.DataDes.dataType, param.root, param.reduceType,
            0};
        HcomCollOpInfo* allgatherOpInfoPtr = DMAReduceFlag_ ? (&allgatherOpInfo) : (nullptr);

        CHK_RET(MultiRingAllGather(
            param.tag, execMem.inputMem, execMem.outputMem, level1DataCount, param.DataDes.dataType, mulRingSlice,
            param.stream, PROF_STAGE_2, 0, allgatherOpInfoPtr));
    } else if (topoAttr_.deviceType == DevType::DEV_TYPE_910B) {
        std::unique_ptr<AlgTemplateBase> level0Executor = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_GATHER_MESH_ATOMIC, dispatcher_);
        CHK_SMART_PTR_NULL(level0Executor);
        CHK_RET(level0Executor->Prepare(
            algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux, topoAttr_.userRank, nullptr,
            level0CommInfo.localRank, level0CommInfo.localRankSize));

        if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) { // offline
            for (u32 streamIndex = 0; streamIndex < algResResp_->slaveStreams.size(); streamIndex++) {
                CHK_RET(StreamActiveManager::GetInstance(topoAttr_.deviceLogicId)
                            .StreamActive(algResResp_->slaveStreams[streamIndex].ptr(), param.stream.ptr()));
            }
        }
        CHK_RET(level0Executor->Prepare(
            execMem.outputMem, execMem.outputMem, execMem.outputMem, execMem.count, param.DataDes.dataType,
            param.stream, HCCL_REDUCE_RESERVED, LEVEL0_BRIDGE_RANK_ID, dataSegsSlice));

        u32 level0RankSize = level0CommInfo.localRankSize;
        CHK_RET(level0Executor->RegisterProfiler(
            (0 << PROF_RINGINDEX_OFFSET_OF_PLANEID) + (level0RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID)
                + level0CommInfo.localRank,
            PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, param.stream));

        CHK_RET(RunTemplate(level0Executor, level0CommInfo));
    }
    HCCL_INFO("[CollBroadCastMix][KernelRun]Broadcast mix stage2 run success");

    return HCCL_SUCCESS;
}

HcclResult CollBroadCastMix::DoubleRingScatter(
    const std::string& tag, DeviceMem inputMem, DeviceMem outputMem, const u64 count, const HcclDataType dataType,
    const std::vector<std::vector<Slice>> multRingsSliceZero, u32 root, Stream stream, const HcomCollOpInfo* opInfo,
    const u64 baseOffset)
{
    HCCL_INFO("[CollBroadCastMix][DoubleRingScatter] DoubleRingScatter starts.");
    HcclResult ret = HCCL_SUCCESS;
    u32 ringNum = multRingsSliceZero.size();

    CHK_RET(CheckCommSize(COMM_LEVEL0, ringNum));

    std::vector<std::vector<u32>> ringNics;
    CHK_RET(GetRingNics(tag, ringNics));

    // 拿到ring环映射关系
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    SubCommInfo level0CommInfo1 = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_1);

    auto nicList = topoAttr_.nicList;
    std::vector<std::vector<u32>> multiRingsOrder
        = GetRingsOrderByTopoType(level0CommInfo.localRankSize, topoType_, nicList);

    std::vector<std::vector<u32>> doubleRingsOrders;
    std::vector<std::vector<Slice>> doubleRingUserMemInputSlices;
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> singleRingSliceZero = multRingsSliceZero[ringIndex];
        CHK_PRT_RET(
            singleRingSliceZero.empty(),
            HCCL_ERROR("[CollBroadCastMix][DoubleRingScatter]singleRingSliceZero is empty"), HCCL_E_INTERNAL);

        // 生成userMemIn_上对应的slices
        std::vector<Slice> userMemInputSlices;
        CHK_RET(
            CalUserMemSlices(dataType, opInfo, singleRingSliceZero, ringIndex, multiRingsOrder, userMemInputSlices));
        doubleRingUserMemInputSlices.push_back(userMemInputSlices);
        std::vector<u32> rankOrder;
        CHK_RET(GetRankOrder(multiRingsOrder, ringIndex, rankOrder));
        doubleRingsOrders.push_back(rankOrder);
    }
    std::unique_ptr<AlgTemplateBase> executor = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_SCATTER_DOUBLE_RING_DIRECT, dispatcher_);
    CHK_SMART_PTR_NULL(executor);

    CHK_RET(executor->Prepare(
        const_cast<HcomCollOpInfo*>(opInfo), topoAttr_.userRank, level0CommInfo1.localRank, algResResp_->slaveStreams,
        algResResp_->notifiesMain, algResResp_->notifiesAux, doubleRingsOrders, multRingsSliceZero,
        doubleRingUserMemInputSlices));

    u32 rootRank = 0;
    ret = GetRankByUserRank(COMM_LEVEL0, COMM_INDEX_0, root, rootRank);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[CollBroadCastMix][DoubleRingScatter]invalid root [%u] to get userrank", root),
        ret);
    ret = executor->Prepare(
        inputMem, inputMem, outputMem, count, dataType, stream, HCCL_REDUCE_RESERVED, rootRank, std::vector<Slice>(0),
        baseOffset);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollBroadCastMix][DoubleRingScatter]scatter(ring) prepare failed, return[%d]", ret), ret);

    u32 rankSize = level0CommInfo.localRankSize;
    ret = executor->RegisterProfiler(
        (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level0CommInfo.localRank, PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET,
        stream);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollBroadCastMix][DoubleRingScatter]scatter(ring) register profiler failed,return[%d]", ret), ret);

    ret = RunTemplate(executor, level0CommInfo);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollBroadCastMix][DoubleRingScatter]scatter(ring) run failed, return[%d]", ret), ret);

    HCCL_INFO("[CollBroadCastMix] double ring scatter run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("BroadCastMixExecutor", BroadCastMix, CollBroadCastMix);

} // namespace hccl

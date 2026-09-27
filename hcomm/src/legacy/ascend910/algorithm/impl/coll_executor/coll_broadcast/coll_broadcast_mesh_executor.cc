/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_broadcast_mesh_executor.h"

namespace hccl {

CollBroadcastMeshExecutor::CollBroadcastMeshExecutor(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollBroadcastExecutor(dispatcher, topoMatcher)
{}

HcclResult CollBroadcastMeshExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = 0U;
    if (algType_.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_4P_MESH) {
        totalStreamNum = LEVEL0_PLANE_NUM_IN_4PMESH;
    } else {
        if ((workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)
            && (topoAttr_.deviceType == DevType::DEV_TYPE_910B) && topoAttr_.isSingleMeshAggregation) {
            totalStreamNum = topoAttr_.deviceNumPerAggregation;
        } else {
            totalStreamNum = topoAttr_.deviceNumPerAggregation - 1;
        }
    }
    streamNum = totalStreamNum > 0 ? totalStreamNum - 1 : 0;

    HCCL_INFO("[CollBroadcastMeshExecutor][CalcStreamNum] tag[%s] streamNum_[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastMeshExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastMeshExecutor::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    commParaLevel0.meshSinglePlane = true;
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastMeshExecutor::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollBroadcastMeshExecutor][KernelRun] userRank[%u] starts.", topoAttr_.userRank);
    u32 perDataSize = SIZE_TABLE[param.DataDes.dataType];

    bool isUsedRegister = false;
    std::unique_ptr<AlgTemplateBase> level0TempAlg1;
    std::unique_ptr<AlgTemplateBase> level1TempAlg;
    std::unique_ptr<AlgTemplateBase> level0TempAlg2;

    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 commIndex = level0CommInfo.localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));

    level0TempAlg1 = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_SCATTER_MESH, dispatcher_);
    CHK_SMART_PTR_NULL(level0TempAlg1);
    CHK_RET(level0TempAlg1->Prepare(level0CommInfo.localRank, level0CommInfo.localRankSize));
    level0TempAlg1->CloseBarrier();

    /* 内层topo:all_reduce */
    /* 外层所有rank均参与内层的broadcast计算，所以此处对rank不作限制，但是每个rank需找到自己所在的内层通信域 */
    std::vector<Slice> slice;
    CHK_RET(GetRankSliceSize(param.DataDes.dataType, execMem.count, level0CommInfo.localRankSize, slice));

    CHK_PRT_RET(
        slice.empty(), HCCL_ERROR("[BroadCastOperator][BroadCastMeshExecutor]got slice is empty"), HCCL_E_INTERNAL);

    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);

    u64 curSize = execMem.count * SIZE_TABLE[param.DataDes.dataType];
    if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
        HCCL_DEBUG(
            "broadcast mesh: curSize[%llu] deviceNumPerAggregation[%u] commLevel0Size[%u]", curSize,
            topoAttr_.deviceNumPerAggregation, level0CommInfo.localRankSize);
        if (curSize / topoAttr_.deviceNumPerAggregation <= NHR_BCAST_SMALL_SIZE) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_BROADCAST_NHR_ONESHOT, dispatcher_);
        } else {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_NHR, dispatcher_);
        }
        HCCL_INFO("broadcast mesh: using nhr algo inter-server.");
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
        isUsedRegister = true;
        level1TempAlg
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_NHR_V1, dispatcher_);
        HCCL_INFO("broadcast mesh: using nhr_v1 algo inter-server.");
    } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
        const u32 level1RankSize = level1CommInfo.localRankSize;
        HCCL_DEBUG("[CollBroadcastMeshExecutor][KernelRun]level1RankSize is %u", level1RankSize);
        if (ShouldUseBinaryBroadcastOfNB(
                curSize / topoAttr_.deviceNumPerAggregation, level1RankSize, topoAttr_.userRankSize,
                topoAttr_.deviceNumPerAggregation)) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_BROADCAST_NB_BINARY, dispatcher_);
        } else {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_NB, dispatcher_);
        }
        HCCL_INFO("broadcast mesh: using nonuniform-bruck algo inter-server.");
    } else {
        level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_BROADCAST_RECURSIVE_HD, dispatcher_);
        HCCL_INFO("broadcast mesh: using Recursive halving-doubling algo inter-server.");
    }
    CHK_SMART_PTR_NULL(level1TempAlg);

    /* 外层topo:all_gather */
    if (topoAttr_.deviceType == DevType::DEV_TYPE_910B) {
        level0TempAlg2 = AlgTemplateRegistry::Instance().GetAlgTemplate(
            TemplateType::TEMPLATE_ALL_GATHER_MESH_ATOMIC, dispatcher_);
    } else {
        level0TempAlg2
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_MESH, dispatcher_);
    }
    CHK_SMART_PTR_NULL(level0TempAlg2);
    CHK_RET(level0TempAlg2->Prepare(
        algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux, topoAttr_.userRank, nullptr,
        level0CommInfo.localRank, level0CommInfo.localRankSize));

    /* 节点内执行器 stage0 */
    u32 rootRank = 0;
    HcclResult ret = GetRankByUserRank(COMM_LEVEL0, COMM_INDEX_0, param.root, rootRank);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[BroadCastOperator][BroadCastMeshExecutor]invalid root[%u] to get userrank", param.root), ret);

    if (ret == HCCL_SUCCESS) {
        CHK_RET(level0TempAlg1->Prepare(
            execMem.inputMem, execMem.outputMem, execMem.outputMem, execMem.count, param.DataDes.dataType, param.stream,
            HCCL_REDUCE_RESERVED, rootRank, slice));

        u32 rankSize = level0CommInfo.localRankSize;
        CHK_RET(level0TempAlg1->RegisterProfiler(
            (0 << PROF_RINGINDEX_OFFSET_OF_PLANEID) + (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID)
                + level0CommInfo.localRank,
            PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));

        CHK_RET(RunTemplate(level0TempAlg1, level0CommInfo));
    } else {
        HCCL_ERROR("[BroadCastOperator][BroadCastMeshExecutor]invalid root[%u] to get userrank", param.root);
    }
    HCCL_INFO("[BroadCastOperator][BroadCastMeshExecutor] stage0 run success");
    u64 hdCount = slice[level0CommInfo.localRank].size / perDataSize;
    /* 节点间执行器 stage1 */

    u32 subUserrankRoot = topoMatcher_->GetSubRootUserRank(topoAttr_.userRank, param.root);
    CHK_PRT_RET(
        subUserrankRoot == INVALID_VALUE_RANKID,
        HCCL_ERROR(
            "[BroadCastOperator][BroadCastMeshExecutor]subUserrankRoot[%u] is invalid,userRank[%u],root[%u]",
            subUserrankRoot, topoAttr_.userRank, param.root),
        HCCL_E_INTERNAL);

    u32 subRoot = 0;
    CHK_RET(GetRankByUserRank(COMM_LEVEL1, commIndex, subUserrankRoot, subRoot));

    // 增加偏移参数
    if (isUsedRegister) {
        PrepareData prepareData;
        prepareData.inputMem = execMem.inputMem;
        prepareData.outputMem = execMem.outputMem;
        prepareData.scratchMem = execMem.outputMem;
        prepareData.count = hdCount;
        prepareData.dataType = param.DataDes.dataType;
        prepareData.stream = param.stream;
        prepareData.reductionOp = HCCL_REDUCE_RESERVED;
        prepareData.root = subRoot;
        prepareData.baseOffset = slice[level0CommInfo.localRank].offset;
        CHK_RET(level1TempAlg->Prepare(prepareData));
    } else {
        CHK_RET(level1TempAlg->Prepare(
            execMem.inputMem, execMem.outputMem, execMem.outputMem, hdCount, param.DataDes.dataType, param.stream,
            HCCL_REDUCE_RESERVED, subRoot, std::vector<Slice>(0), slice[level0CommInfo.localRank].offset));
    }

    u32 rankSize = level1CommInfo.localRankSize;
    CHK_RET(level1TempAlg->RegisterProfiler(
        (0 << PROF_RINGINDEX_OFFSET_OF_PLANEID) + (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID)
            + level1CommInfo.localRank,
        PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));
    HCCL_INFO("[BroadCastOperator][BroadCastMeshExecutor] stage1 run success");

    /* 节点内执行器 stage2 */
    {
        if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) { // offline
            for (u32 streamIndex = 0; streamIndex < algResResp_->slaveStreams.size(); streamIndex++) {
                CHK_RET(StreamActiveManager::GetInstance(topoAttr_.deviceLogicId)
                            .StreamActive(algResResp_->slaveStreams[streamIndex].ptr(), param.stream.ptr()));
            }
        }
        CHK_RET(level0TempAlg2->Prepare(
            execMem.outputMem, execMem.outputMem, execMem.outputMem, execMem.count, param.DataDes.dataType,
            param.stream, HCCL_REDUCE_RESERVED, LEVEL0_BRIDGE_RANK_ID, slice));

        u32 rankSize = level0CommInfo.localRankSize;
        CHK_RET(level0TempAlg2->RegisterProfiler(
            (0 << PROF_RINGINDEX_OFFSET_OF_PLANEID) + (rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID)
                + level0CommInfo.localRank,
            PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, param.stream));

        CHK_RET(RunTemplate(level0TempAlg2, level0CommInfo));
    }

    HCCL_INFO("[BroadCastOperator][BroadCastMeshExecutor] stage2 run success");
    return HCCL_SUCCESS;
}
HcclResult CollBroadcastMeshExecutor::Getlevel1CommRank(SubCommInfo& level1CommInfo)
{
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    u32 commIndex = level0CommInfo.localRank; // 找到rank所在的节点间平面
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));

    level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastMeshExecutor::SelectTempAlg(std::unique_ptr<AlgTemplateBase>& level1TempAlg, u32 level1RankSize)
{
    if (level1RankSize > 1) {
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_NHR, dispatcher_);
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_NHR_V1, dispatcher_);
            HCCL_INFO("broadcast mesh: using nhr_v1 algo inter-server.");
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            level1TempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_NB, dispatcher_);
            HCCL_INFO("broadcast mesh: using nonuniform-bruck algo inter-server.");
        } else {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_BROADCAST_RECURSIVE_HD, dispatcher_);
            HCCL_INFO("broadcast mesh: using Recursive halving-doubling algo inter-server.");
        }
        CHK_SMART_PTR_NULL(level1TempAlg);
    }
    return HCCL_SUCCESS;
}
REGISTER_EXEC("BroadCastMeshExecutor", BroadcastMesh, CollBroadcastMeshExecutor);

} // namespace hccl

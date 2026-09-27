/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_broadcast_plus_broadcast.h"

namespace hccl {
CollBroadcastPlusBroadcast::CollBroadcastPlusBroadcast(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollBroadcastExecutor(dispatcher, topoMatcher)
{}

HcclResult CollBroadcastPlusBroadcast::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastPlusBroadcast::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    HCCL_INFO("[CollBroadcastPlusBroadcast][CalcLevel0CommInfo]tag[%s] start", tag_.c_str());
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    HCCL_INFO("[CollBroadcastPlusBroadcast][CalcLevel0CommInfo]tag[%s] Calc MeshComm finish", tag_.c_str());
    return HCCL_SUCCESS;
}

HcclResult CollBroadcastPlusBroadcast::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollBroadcastPlusBroadcast]310p userRank[%u] starts.", topoAttr_.userRank);
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);

    bool rootIsDevPhyZero = false;
    if (topoAttr_.userRank == param.root && topoAttr_.devicePhyId == 0) {
        rootIsDevPhyZero = true;
    }
    // 第一步，如果root不在dev 0上，先将数据bcast到设备0上，在进行server间bcast，设备0调度网卡更快
    if (!rootIsDevPhyZero) {
        u32 rootRank = 0;
        CHK_RET(GetRankByUserRank(COMM_LEVEL0, COMM_INDEX_0, param.root, rootRank));
        std::unique_ptr<AlgTemplateBase> bCastRingInNode
            = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_RING, dispatcher_);
        CHK_SMART_PTR_NULL(bCastRingInNode);
        CHK_RET(bCastRingInNode->Prepare(
            execMem.inputMem, execMem.outputMem, execMem.inputMem, execMem.count, param.DataDes.dataType, param.stream,
            HCCL_REDUCE_RESERVED, rootRank));
        CHK_RET(RunTemplate(bCastRingInNode, level0CommInfo));
    }
    // 第二步，进行server间bcast
    if (topoAttr_.devicePhyId == 0) {
        bool isUsedRegister = false;
        std::unique_ptr<AlgTemplateBase> broadcastTempAlg = nullptr;
        SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, COMM_INDEX_0);
        u64 curSize = execMem.count * SIZE_TABLE[param.DataDes.dataType];
        if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
            broadcastTempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_RING, dispatcher_);
            HCCL_INFO("broadcast ring: using ring algo inter-server.");
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
            HCCL_DEBUG(
                "broadcast ring: curSize[%llu] deviceNumPerAggregation[%u] commLevel0Size[%u]", curSize,
                topoAttr_.deviceNumPerAggregation, level0CommInfo.localRankSize);
            if (curSize / topoAttr_.deviceNumPerAggregation <= NHR_BCAST_SMALL_SIZE) {
                broadcastTempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_BROADCAST_NHR_ONESHOT, dispatcher_);
            } else {
                broadcastTempAlg
                    = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_NHR, dispatcher_);
            }
            HCCL_INFO("broadcast ring: using nhr algo inter-server.");
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
            isUsedRegister = true;
            broadcastTempAlg
                = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_NHR_V1, dispatcher_);
            HCCL_INFO("broadcast ring: using nhr_v1 algo inter-server.");
        } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
            const u32 level1RankSize = level1CommInfo.localRankSize;
            if (ShouldUseBinaryBroadcastOfNB(
                    curSize / topoAttr_.deviceNumPerAggregation, level1RankSize, topoAttr_.userRankSize,
                    topoAttr_.deviceNumPerAggregation)) {
                broadcastTempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_BROADCAST_NB_BINARY, dispatcher_);
            } else {
                broadcastTempAlg
                    = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_NB, dispatcher_);
            }
            HCCL_INFO("broadcast ring: using nonuniform-bruck algo inter-server.");
        } else {
            broadcastTempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_BROADCAST_RECURSIVE_HD, dispatcher_);
            HCCL_INFO("broadcast recursive hd: using halving-doubling algo inter-server.");
        }
        CHK_SMART_PTR_NULL(broadcastTempAlg);
        // 获取root所在的server的device0的userRank
        u32 level1RootUserRank = level1CommInfo.localRank;
        CHK_RET(CheckCommSize(COMM_LEVEL1, COMM_INDEX_0 + 1));
        u32 planeRoot = 0;
        CHK_RET(GetRankByUserRank(COMM_LEVEL1, COMM_INDEX_0, level1RootUserRank, planeRoot));

        if (isUsedRegister) {
            PrepareData prepareData;
            prepareData.inputMem = execMem.inputMem;
            prepareData.outputMem = execMem.outputMem;
            prepareData.scratchMem = execMem.outputMem;
            prepareData.count = execMem.count;
            prepareData.dataType = param.DataDes.dataType;
            prepareData.stream = param.stream;
            prepareData.reductionOp = HCCL_REDUCE_RESERVED;
            prepareData.root = planeRoot;
            prepareData.baseOffset = 0;
            CHK_RET(broadcastTempAlg->Prepare(prepareData));
        } else {
            CHK_RET(broadcastTempAlg->Prepare(
                execMem.inputMem, execMem.outputMem, execMem.outputMem, execMem.count, param.DataDes.dataType,
                param.stream, HCCL_REDUCE_RESERVED, planeRoot));
        }

        CHK_RET(RunTemplate(broadcastTempAlg, level1CommInfo));
    }
    // 第三步，执行server内broadcast（从设备0到设备1）
    std::unique_ptr<AlgTemplateBase> bcastTempAlg
        = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_BROADCAST_RING, dispatcher_);
    CHK_SMART_PTR_NULL(bcastTempAlg);
    // 获取本rank所在server上device0的UserRank
    u32 level0RootUserRank = level0CommInfo.localRank;
    u32 rootRank = 0;
    CHK_RET(GetRankByUserRank(COMM_LEVEL0, COMM_INDEX_0, level0RootUserRank, rootRank));
    CHK_RET(bcastTempAlg->Prepare(
        execMem.outputMem, execMem.outputMem, execMem.inputMem, execMem.count, param.DataDes.dataType, param.stream,
        HCCL_REDUCE_RESERVED, rootRank));
    CHK_RET(RunTemplate(bcastTempAlg, level0CommInfo));

    return HCCL_SUCCESS;
}

REGISTER_EXEC("BroadcastPlusBroadcast", BroadcastPlusBroadcast, CollBroadcastPlusBroadcast);
} // namespace hccl

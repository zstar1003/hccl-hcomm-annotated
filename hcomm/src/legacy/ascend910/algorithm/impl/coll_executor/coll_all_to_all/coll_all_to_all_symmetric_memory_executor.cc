/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_to_all_symmetric_memory_executor.h"

namespace hccl {

CollRunAlltoAllFullMeshSymmetricMemory::CollRunAlltoAllFullMeshSymmetricMemory(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAlltoAllExecutor(dispatcher, topoMatcher)
{
    desc_.isZeroCopy = true;
}

HcclResult CollRunAlltoAllFullMeshSymmetricMemory::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    HcclResult ret = HCCL_SUCCESS;
    tag_ = param.tag;
    algResResp_ = &algRes;
    AlltoAllVParam_ = param;

    ExecMem execMem;
    execMem.count = 0;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;
    execMem.inputMem = algRes.paramInputMem;
    execMem.outputMem = algRes.paramOutputMem;
    ret = KernelRun(param, execMem);

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollRunAlltoAllFullMeshSymmetricMemory][Orchestrate]errNo[0x%016llx]executor run failed",
            HCCL_ERROR_CODE(ret)),
        ret);

    HCCL_INFO(
        "tag[%s], AlltoAllFullMeshSymmetricMemory tempAlg orchestrate success, take time [%lld]us.", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllFullMeshSymmetricMemory::GetLocalSDMAGroupInfo(u32& devNumInlocalPod, u32& rankIdxInPod) const
{
    CHK_RET(topoMatcher_->GetLocalSuperPodRankSize(topoAttr_.userRank, devNumInlocalPod, rankIdxInPod));
    CHK_PRT_RET(
        devNumInlocalPod == INVALID_VALUE_RANKSIZE,
        HCCL_ERROR(
            "[CollRunAlltoAllFullMeshSymmetricMemory][GetLocalSDMAGroupInfo]get local superPod total ranksize failed."),
        HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllFullMeshSymmetricMemory::CalcStreamNum(u32& streamNum)
{
    // 每个超节点内的卡数
    u32 devNumInlocalPod = INVALID_VALUE_RANKSIZE;
    u32 rankIdxInPod = INVALID_VALUE_RANKID;
    CHK_RET(GetLocalSDMAGroupInfo(devNumInlocalPod, rankIdxInPod));

    // 单超节点场景需要的从流数量，待确认是否需要减去一条主流
    streamNum = (devNumInlocalPod > ALLTOALLV_DIRECT_FULLMESH_SDMA_CONCURRENT_SIZE) ?
                    (ALLTOALLV_DIRECT_FULLMESH_SDMA_CONCURRENT_SIZE) :
                    (devNumInlocalPod);

    HCCL_INFO("[CollRunAlltoAllFullMeshSymmetricMemory][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

// level0-level1 打平fullmesh
// 超节点内建SDMA链路；超节点间建RDMA链路
HcclResult CollRunAlltoAllFullMeshSymmetricMemory::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commCombinePara(COMM_COMBINE_ORDER, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commCombinePara, opTransport[COMM_COMBINE_ORDER], inputType, outputType));
    LevelNSubCommTransport& commTransportLevel0 = opTransport[COMM_COMBINE_ORDER];
    for (u32 subCommIndex = 0; subCommIndex < commTransportLevel0.size(); subCommIndex++) {
        commTransportLevel0[subCommIndex].isZeroCopy = true;
    }
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllFullMeshSymmetricMemory::CalcTransportMemType(
    TransportMemType& inputType, TransportMemType& outputType) const
{
    inputType = TransportMemType::CCL_INPUT;
    outputType = TransportMemType::CCL_OUTPUT;

    HCCL_INFO(
        "[CollRunAlltoAllFullMeshSymmetricMemory][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllFullMeshSymmetricMemory::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;

    CHK_RET(CalcTransportMemType(inputType, outputType));
    // level0 - level1 全连接通信域
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllFullMeshSymmetricMemory::GetLocalSendRecvInfoforAlltoall(const OpParam& param)
{
    u64 curRecvOffset = 0;
    for (u32 j = 0; j < topoAttr_.userRankSize; j++) {
        u64 curSendCounts = param.All2AllDataDes.sendCount;
        u64 curSendLength = curSendCounts * SIZE_TABLE[param.All2AllDataDes.sendType];
        sendRecvInfo_.remoteSendOffset[j] = curSendLength * topoAttr_.userRank;

        u64 curRecvCounts = param.All2AllDataDes.sendCount;
        u64 curRecvLength = curRecvCounts * SIZE_TABLE[param.All2AllDataDes.recvType];
        sendRecvInfo_.localRecvLength[j] = curRecvLength;
        sendRecvInfo_.localRecvOffset[j] = curRecvOffset;
        curRecvOffset += curRecvLength;
        HCCL_DEBUG(
            "GetLocalSendRecvInfoforAlltoall rank[%u], remoteSendOffset[j][%llu], localRecvLength[j][%llu] "
            "localRecvOffset[j][%llu]",
            topoAttr_.userRank, sendRecvInfo_.remoteSendOffset[j], sendRecvInfo_.localRecvLength[j],
            sendRecvInfo_.localRecvOffset[j]);
    }
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllFullMeshSymmetricMemory::GetLocalSendRecvInfoforAlltoallVC(const OpParam& param)
{
    u64 rankSize = topoAttr_.userRankSize;
    u64 usrRank = topoAttr_.userRank;
    for (u32 j = 0; j < topoAttr_.userRankSize; j++) {
        sendRecvInfo_.remoteSendOffset[j] = 0;
        for (u32 i = 0; i < usrRank; i++) {
            u64 sendCounts = *(static_cast<const u64*>(param.All2AllDataDes.sendCountMatrix) + i + rankSize * j);
            u64 sendLength = sendCounts * SIZE_TABLE[param.All2AllDataDes.recvType];
            sendRecvInfo_.remoteSendOffset[j] += sendLength;
        }
        sendRecvInfo_.localRecvOffset[j] = 0;
        for (u32 i = 0; i < j; i++) {
            u64 recvCounts = *(static_cast<const u64*>(param.All2AllDataDes.sendCountMatrix) + usrRank + rankSize * i);
            u64 recvLength = recvCounts * SIZE_TABLE[param.All2AllDataDes.sendType];
            sendRecvInfo_.localRecvOffset[j] += recvLength;
        }
        u64 curRecvCounts = *(static_cast<const u64*>(param.All2AllDataDes.sendCountMatrix) + usrRank + rankSize * j);
        sendRecvInfo_.localRecvLength[j] = curRecvCounts * SIZE_TABLE[param.All2AllDataDes.recvType];

        HCCL_DEBUG(
            "GetLocalSendRecvInfoforAlltoallVC rank[%u], remoteSendOffset[%llu], "
            "localRecvLength[%llu], localRecvOffset[%llu]",
            topoAttr_.userRank, sendRecvInfo_.remoteSendOffset[j], sendRecvInfo_.localRecvLength[j],
            sendRecvInfo_.localRecvOffset[j]);
    }
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllFullMeshSymmetricMemory::GetAlltoAllTmpRankSendRecvInfo(const OpParam& param)
{
    sendRecvInfo_.remoteSendOffset.resize(topoAttr_.userRankSize, 0);
    sendRecvInfo_.localRecvLength.resize(topoAttr_.userRankSize, 0);
    sendRecvInfo_.localRecvOffset.resize(topoAttr_.userRankSize, 0);

    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        CHK_RET(GetLocalSendRecvInfoforAlltoall(param));
    } else if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        CHK_RET(GetLocalSendRecvInfoforAlltoallVC(param));
    } else {
        HCCL_ERROR("Only support optype AllToAll and AllToAllVC !");
    }
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllFullMeshSymmetricMemory::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] AllToAll fullmesh start.", __func__);

    // 准备数据
    CHK_RET(ActiveSlaveStreams(param.stream));
    CHK_RET(GetAlltoAllTmpRankSendRecvInfo(param));

    // 获取当前超节点内总卡数
    u32 devNumInlocalPod = INVALID_VALUE_RANKSIZE;
    u32 rankIdxInPod = INVALID_VALUE_RANKID;
    CHK_RET(GetLocalSDMAGroupInfo(devNumInlocalPod, rankIdxInPod));

    // 获取通信域
    CHK_RET(CheckCommSize(COMM_COMBINE_ORDER, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_COMBINE_ORDER, COMM_INDEX_0);
    // isSuPodAsym 表示A2A3卡数不一致场景或者A3多超节点server数不同场景
    bool isSuPodAsym = false;

    // 执行
    std::unique_ptr<AlgTemplateBase> tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_ALL_2_ALL_FULL_MESH_SYMMETRIC_MEMORY, dispatcher_);
    HCCL_CONFIG_INFO(
        HCCL_ALG, "[%s] Run TEMPLATE_ALL_2_ALL_FULL_MESH_SYMMETRIC_MEMORY in COMM_COMBINE_ORDER", __func__);
    CHK_SMART_PTR_NULL(tempAlg);

    PrepareData prepareData;
    prepareData.stream = param.stream;
    prepareData.userRank = topoAttr_.userRank;
    prepareData.userRankSize = topoAttr_.userRankSize;
    prepareData.linksPtr = &level0CommInfo.links;
    prepareData.sendRecvInfoPtr = &sendRecvInfo_;
    prepareData.devNumInlocalPod = devNumInlocalPod;
    prepareData.rankIdxInPod = rankIdxInPod;

    prepareData.inputMem = algResResp_->paramInputMem;
    prepareData.outputMem = algResResp_->paramOutputMem;
    prepareData.cclInMem = execMem.inputMem;
    prepareData.cclOutMem = execMem.outputMem;
    prepareData.workMode = workflowMode_;
    prepareData.subStreamsPtr = &algResResp_->slaveStreams;
    prepareData.signalPtr = &algResResp_->notifiesMain;
    prepareData.signalAuxPtr = &algResResp_->notifiesAux;
    prepareData.isSuPodAsym = isSuPodAsym;
    prepareData.opType = param.opType;
    prepareData.algOpContext = algOpContext_;

    CHK_RET(tempAlg->Prepare(prepareData));

    CHK_RET(tempAlg->RunAsync());

    HCCL_INFO("[CollRunAlltoAllFullMeshSymmetricMemory] executor run success.");
    if (algOpContext_.opRetryHandler.isPostSync == true) {
        OpParam postSyncParam = param;
        if ((*prepareData.subStreamsPtr).size() == 0) {
            CHK_RET(PostSyncWithoutSubstream(postSyncParam, execMem));
        } else {
            PrepareData postSyncPrepareData = prepareData;
            CHK_RET(PostSyncWithSubstream(postSyncParam, execMem, postSyncPrepareData));
        }
    }
    return HCCL_SUCCESS;
}

REGISTER_EXEC(
    "RunAlltoAllFullMeshSymmetricMemory", AlltoAllFullMeshSymmetricMemory, CollRunAlltoAllFullMeshSymmetricMemory);
} // namespace hccl

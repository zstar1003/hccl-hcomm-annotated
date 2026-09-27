/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_to_all_v_direct_fullmesh_executor.h"

namespace hccl {

CollRunAlltoAllDirectFullmesh::CollRunAlltoAllDirectFullmesh(
    const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
    : CollAlltoAllExecutor(dispatcher, topoMatcher)
{}

HcclResult CollRunAlltoAllDirectFullmesh::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
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
    execMem.inputMem = algRes.cclInputMem;
    execMem.outputMem = algRes.cclOutputMem;
    ret = KernelRun(param, execMem);

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[CollRunAlltoAllDirectFullmesh][Orchestrate]errNo[0x%016llx]executor run failed", HCCL_ERROR_CODE(ret)),
        ret);

    // Enforce task launch at the end of Orchestrate
    // 注意: 不要删除这里的强制launch, 否则会导致aicpu cache功能问题
    HCCL_INFO("%s: enforce task launch at the end of Orchestrate", __func__);
    CHK_RET(LaunchTaskExtend(dispatcher_, param.stream, algResResp_->slaveStreams));

    HCCL_INFO(
        "tag[%s], AlltoAllDirectFullmesh tempAlg orchestrate success, take time [%lld]us.", param.tag.c_str(),
        DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllDirectFullmesh::GetAdjInfo(AlgResourceResponse& algRes, AdjInfo& adjInfo)
{
    HCCL_INFO("[CollRunAlltoAllDirectFullmesh] GetAdjInfo.");
    algResResp_ = &algRes;
    SubCommInfo levelCommInfo = {};
    AdjInfo nslbAdjInfo = {};
    u32 devNumInlocalPod = INVALID_VALUE_RANKSIZE;

    u32 localRank = topoAttr_.userRank;
    u32 localRankSize = topoAttr_.userRankSize;

    std::unique_ptr<AlgTemplateBase> levelTempAlg;

    HCCL_INFO("[CollRunAlltoAllDirectFullmesh] SelectTempAlg.");
    levelTempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_ALL_2_ALL_V_DIRECT_FULL_MESH, dispatcher_);
    CHK_SMART_PTR_NULL(levelTempAlg);
    u32 rankIdxInPod = INVALID_VALUE_RANKID;
    CHK_RET(GetLocalSDMAGroupInfo(topoAttr_.userRank, devNumInlocalPod, rankIdxInPod));

    if (devNumInlocalPod == INVALID_VALUE_RANKSIZE) {
        HCCL_INFO("[CollRunAlltoAllDirectFullmesh] devNumInlocalPod == INVALID_VALUE_RANKSIZE.");
        return HCCL_SUCCESS;
    }

    nslbAdjInfo.dstRankNum = devNumInlocalPod;
    CHK_RET(levelTempAlg->GetNslbAdjInfo(localRank, localRankSize, levelCommInfo.links, nslbAdjInfo));

    adjInfo.dstRankNum = nslbAdjInfo.dstRankNum;
    HCCL_INFO("[CollRunAlltoAllDirectFullmesh] adjInfo.dstRankNum[%u].", adjInfo.dstRankNum);

    for (size_t i = 0; i < nslbAdjInfo.nsAdjInfo.size(); i++) {
        NslbDpAdjInfo dpAdjInfo = {};
        dpAdjInfo.dstLocalRankId = nslbAdjInfo.nsAdjInfo[i].dstLocalRankId;
        dpAdjInfo.phaseId = nslbAdjInfo.nsAdjInfo[i].phaseId;
        dpAdjInfo.rev = 0;
        adjInfo.nsAdjInfo.push_back(dpAdjInfo);
        HCCL_INFO(
            "[CollRunAlltoAllDirectFullmesh][GetAdjInfo] dstLocalRankId[%u], phaseId[%u].",
            nslbAdjInfo.nsAdjInfo[i].dstLocalRankId, nslbAdjInfo.nsAdjInfo[i].phaseId);
    }
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllDirectFullmesh::MarkNeedAlltoallvCache()
{
    needAlltoallvCache_ = true;
    HCCL_INFO(
        "[CollRunAlltoAllDirectFullmesh][MarkNeedAlltoallvCache] set needAlltoallvCache_[%u]"
        "for alltoallv aicpu cache",
        needAlltoallvCache_);
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllDirectFullmesh::GetHcclOffsetDstRanksMap(
    std::unordered_map<uint64_t, std::vector<uint32_t>>& hcclOffsetDstRanksMap) const
{
    hcclOffsetDstRanksMap.clear();
    hcclOffsetDstRanksMap = hcclOffsetDstRanksMap_; // Deep copy

    return HCCL_SUCCESS;
}

HcclOpMetaInfo CollRunAlltoAllDirectFullmesh::GetOpMeta(HcclCMDType opType, const u64 size)
{
    (void)opType;
    HcclOpMetaInfoDef opMeta = HcclOpMetaInfo::GetOneForAllToAllV(CopyPattern::ZCOPY, size, true);
    return opMeta;
}

HcclResult
CollRunAlltoAllDirectFullmesh::GetLocalSDMAGroupInfo(const u32 userRank, u32& devNumInlocalPod, u32& rankIdxInPod) const
{
    (void)userRank;
    bool isA2MultiModule = topoAttr_.deviceType == DevType::DEV_TYPE_910B && !topoAttr_.isSingleMeshAggregation;
    if (static_cast<bool>(topoMatcher_->GetExternalInputInterHccsDisable()) || isA2MultiModule) {
        CHK_RET(topoMatcher_->GetLocalServerRankSize(topoAttr_.userRank, devNumInlocalPod, rankIdxInPod));
    } else {
        CHK_RET(topoMatcher_->GetLocalSuperPodRankSize(topoAttr_.userRank, devNumInlocalPod, rankIdxInPod));
    }
    CHK_PRT_RET(
        devNumInlocalPod == INVALID_VALUE_RANKSIZE,
        HCCL_ERROR("[CollRunAlltoAllDirectFullmesh][GetLocalSDMAGroupInfo]get local superPod total ranksize failed."),
        HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllDirectFullmesh::CalcStreamNum(u32& streamNum)
{
    // 每个超节点内的卡数
    u32 devNumInlocalPod = INVALID_VALUE_RANKSIZE;
    u32 rankIdxInPod = INVALID_VALUE_RANKID;
    CHK_RET(GetLocalSDMAGroupInfo(topoAttr_.userRank, devNumInlocalPod, rankIdxInPod));

    // 单超节点场景需要的从流数量
    streamNum = (devNumInlocalPod > ALLTOALLV_DIRECT_FULLMESH_SDMA_CONCURRENT_SIZE) ?
                    (ALLTOALLV_DIRECT_FULLMESH_SDMA_CONCURRENT_SIZE * RANK_SET_COMPUTE_CONST) :
                    (devNumInlocalPod * RANK_SET_COMPUTE_CONST);

    // 多超节点场景下，RDMA会设置独立的并发度
    if ((topoAttr_.userRankSize - devNumInlocalPod) > 0) {
        streamNum += 1; // 一条从流专门用来管理超节点间的RDMA通信
        u32 totalRdmaRankNum = topoAttr_.userRankSize - devNumInlocalPod;
        streamNum += (totalRdmaRankNum > ALLTOALLV_DIRECT_FULLMESH_RDMA_CONCURRENT_SIZE) ?
                         (ALLTOALLV_DIRECT_FULLMESH_RDMA_CONCURRENT_SIZE) :
                         (totalRdmaRankNum);
    }

    HCCL_INFO("[CollRunAlltoAllDirectFullmesh][CalcStreamNum] tag[%s] streamNum[%u]", tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

// level0-level1 打平fullmesh
// 超节点内建SDMA链路；超节点间建RDMA链路
HcclResult CollRunAlltoAllDirectFullmesh::CalcLevel0CommInfo(
    TransportMemType inputType, TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commCombinePara(COMM_COMBINE_ORDER, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commCombinePara, opTransport[COMM_COMBINE_ORDER], inputType, outputType));

    LevelNSubCommTransport& commTransportLevel0 = opTransport[COMM_COMBINE_ORDER];
    for (u32 subCommIndex = 0; subCommIndex < commTransportLevel0.size(); subCommIndex++) {
        for (auto& transportRequest : commTransportLevel0[subCommIndex].transportRequests) {
            transportRequest.isUsedRdma = topoAttr_.isUsedRdmaMap.at(transportRequest.remoteUserRank);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult
CollRunAlltoAllDirectFullmesh::CalcTransportMemType(TransportMemType& inputType, TransportMemType& outputType) const
{
    inputType = TransportMemType::CCL_INPUT;
    outputType = TransportMemType::CCL_OUTPUT;

    HCCL_INFO(
        "[CollRunAlltoAllDirectFullmesh][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]", tag_.c_str(),
        inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllDirectFullmesh::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;

    CHK_RET(CalcTransportMemType(inputType, outputType));
    // level0 - level1 全连接通信域
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllDirectFullmesh::GetLocalSendRecvInfoforAlltoallV(const OpParam& param)
{
    // 注意: 如果send/recv info的计算逻辑发生变化, 需要同步修改framework下的IsSmallDataAlltoallv()函数
    for (u32 j = 0; j < topoAttr_.userRankSize; j++) {
        u64 curSendCounts = *(static_cast<const u64*>(param.All2AllDataDes.sendCounts) + j);
        u64 curSendDispls = *(static_cast<const u64*>(param.All2AllDataDes.sdispls) + j);
        localSendRecvInfo_.sendCounts[j] = curSendCounts;
        localSendRecvInfo_.sendDispls[j] = curSendDispls;
        localSendRecvInfo_.sendLength[j] = curSendCounts * SIZE_TABLE[param.All2AllDataDes.sendType];
        localSendRecvInfo_.sendOffset[j] = curSendDispls * SIZE_TABLE[param.All2AllDataDes.sendType];

        u64 curRecvCounts = *(static_cast<const u64*>(param.All2AllDataDes.recvCounts) + j);
        u64 curRecvDispls = *(static_cast<const u64*>(param.All2AllDataDes.rdispls) + j);
        localSendRecvInfo_.recvCounts[j] = curRecvCounts;
        localSendRecvInfo_.recvDispls[j] = curRecvDispls;
        localSendRecvInfo_.recvLength[j] = curRecvCounts * SIZE_TABLE[param.All2AllDataDes.recvType];
        localSendRecvInfo_.recvOffset[j] = curRecvDispls * SIZE_TABLE[param.All2AllDataDes.recvType];

        HCCL_DEBUG(
            "GetLocalSendRecvInfoforAlltoallV rank[%u], sendCounts[%llu], sendDispls[%llu] "
            "recvCounts[%llu], recvDispls[%llu]",
            topoAttr_.userRank, localSendRecvInfo_.sendCounts[j], localSendRecvInfo_.sendDispls[j],
            localSendRecvInfo_.recvCounts[j], localSendRecvInfo_.recvDispls[j]);
    }
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllDirectFullmesh::GetLocalSendRecvInfoforAlltoall(const OpParam& param)
{
    u64 curSendDispls = 0;
    u64 curSendOffset = 0;
    u64 curRecvDispls = 0;
    u64 curRecvOffset = 0;
    for (u32 j = 0; j < topoAttr_.userRankSize; j++) {
        u64 curSendCounts = param.All2AllDataDes.sendCount;
        u64 curSendLength = curSendCounts * SIZE_TABLE[param.All2AllDataDes.sendType];
        localSendRecvInfo_.sendCounts[j] = curSendCounts;
        localSendRecvInfo_.sendDispls[j] = curSendDispls;
        localSendRecvInfo_.sendLength[j] = curSendLength;
        localSendRecvInfo_.sendOffset[j] = curSendOffset;
        curSendDispls += curSendCounts;
        curSendOffset += curSendLength;

        u64 curRecvCounts = param.All2AllDataDes.sendCount;
        u64 curRecvLength = curRecvCounts * SIZE_TABLE[param.All2AllDataDes.recvType];
        localSendRecvInfo_.recvCounts[j] = curRecvCounts;
        localSendRecvInfo_.recvDispls[j] = curRecvDispls;
        localSendRecvInfo_.recvLength[j] = curRecvLength;
        localSendRecvInfo_.recvOffset[j] = curRecvOffset;
        curRecvDispls += curRecvCounts;
        curRecvOffset += curRecvLength;
        HCCL_DEBUG(
            "GetLocalSendRecvInfoforAlltoAll rank[%u], sendCounts[%llu], sendDispls[%llu] "
            "recvCounts[%llu], recvDispls[%llu]",
            topoAttr_.userRank, localSendRecvInfo_.sendCounts[j], localSendRecvInfo_.sendDispls[j],
            localSendRecvInfo_.recvCounts[j], localSendRecvInfo_.recvDispls[j]);
    }
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllDirectFullmesh::GetLocalSendRecvInfoforAlltoallVC(const OpParam& param)
{
    u64 curSendDispls = 0;
    u64 curSendOffset = 0;
    u64 curRecvDispls = 0;
    u64 curRecvOffset = 0;
    u64 rankSize = topoAttr_.userRankSize;
    u64 usrRank = topoAttr_.userRank;
    for (u32 j = 0; j < topoAttr_.userRankSize; j++) {
        u64 curSendCounts = *(static_cast<const u64*>(param.All2AllDataDes.sendCountMatrix) + usrRank * rankSize + j);
        u64 curSendLength = curSendCounts * SIZE_TABLE[param.All2AllDataDes.sendType];
        localSendRecvInfo_.sendCounts[j] = curSendCounts;
        localSendRecvInfo_.sendDispls[j] = curSendDispls;
        localSendRecvInfo_.sendLength[j] = curSendLength;
        localSendRecvInfo_.sendOffset[j] = curSendOffset;
        curSendDispls += curSendCounts;
        curSendOffset += curSendLength;

        u64 curRecvCounts = *(static_cast<const u64*>(param.All2AllDataDes.sendCountMatrix) + usrRank + rankSize * j);
        u64 curRecvLength = curRecvCounts * SIZE_TABLE[param.All2AllDataDes.recvType];
        localSendRecvInfo_.recvCounts[j] = curRecvCounts;
        localSendRecvInfo_.recvDispls[j] = curRecvDispls;
        localSendRecvInfo_.recvLength[j] = curRecvLength;
        localSendRecvInfo_.recvOffset[j] = curRecvOffset;
        curRecvDispls += curRecvCounts;
        curRecvOffset += curRecvLength;
        HCCL_DEBUG(
            "GetLocalSendRecvInfoforAlltoallVC rank[%u], sendCounts[%llu], sendDispls[%llu] "
            "recvCounts[%llu], recvDispls[%llu]",
            topoAttr_.userRank, localSendRecvInfo_.sendCounts[j], localSendRecvInfo_.sendDispls[j],
            localSendRecvInfo_.recvCounts[j], localSendRecvInfo_.recvDispls[j]);
    }
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllDirectFullmesh::GetAlltoAllvTmpRankSendRecvInfo(const OpParam& param)
{
    localSendRecvInfo_.sendCounts.resize(topoAttr_.userRankSize, 0);
    localSendRecvInfo_.sendDispls.resize(topoAttr_.userRankSize, 0);
    localSendRecvInfo_.sendLength.resize(topoAttr_.userRankSize, 0);
    localSendRecvInfo_.sendOffset.resize(topoAttr_.userRankSize, 0);

    localSendRecvInfo_.recvCounts.resize(topoAttr_.userRankSize, 0);
    localSendRecvInfo_.recvDispls.resize(topoAttr_.userRankSize, 0);
    localSendRecvInfo_.recvLength.resize(topoAttr_.userRankSize, 0);
    localSendRecvInfo_.recvOffset.resize(topoAttr_.userRankSize, 0);
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        CHK_RET(GetLocalSendRecvInfoforAlltoallV(param));
    } else if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        CHK_RET(GetLocalSendRecvInfoforAlltoall(param));
    } else if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        CHK_RET(GetLocalSendRecvInfoforAlltoallVC(param));
    } else {
        HCCL_ERROR("Only support optype AllToAll , AllToAllV and AllToAllVC !");
    }
    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllDirectFullmesh::KernelRun(const OpParam& param, ExecMem& execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] AllToAll fullmesh start.", __func__);

    // 准备数据
    CHK_RET(ActiveSlaveStreams(param.stream));
    CHK_RET(GetAlltoAllvTmpRankSendRecvInfo(param));

    // 获取当前超节点内总卡数
    u32 devNumInlocalPod = INVALID_VALUE_RANKSIZE;
    u32 rankIdxInPod = INVALID_VALUE_RANKID;
    CHK_RET(GetLocalSDMAGroupInfo(topoAttr_.userRank, devNumInlocalPod, rankIdxInPod));

    // 获取通信域
    CHK_RET(CheckCommSize(COMM_COMBINE_ORDER, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_COMBINE_ORDER, COMM_INDEX_0);
    bool isA2MultiModule = topoAttr_.deviceType == DevType::DEV_TYPE_910B && !topoAttr_.isSingleMeshAggregation;
    // isSuPodAsym 表示A2A3卡数不一致场景或者A3多超节点server数不同场景
    bool isSuPodAsym = false;
    if (topoAttr_.superPodNum > 1) {
        isSuPodAsym
            = (static_cast<bool>(topoAttr_.multiModuleDiffDeviceNumMode)
               || static_cast<bool>(topoAttr_.multiSuperPodDiffServerNumMode));
    } else {
        isSuPodAsym = (static_cast<bool>(topoMatcher_->GetExternalInputInterHccsDisable()) || isA2MultiModule)
                      && static_cast<bool>(topoAttr_.multiModuleDiffDeviceNumMode);
    }

    // 执行
    // 注意: 如果使用了非AlltoAllVDirectFullMesh的算法模板, 需要同步修改framework中的NeedOpUnfoldCache()函数
    std::unique_ptr<AlgTemplateBase> tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_ALL_2_ALL_V_DIRECT_FULL_MESH, dispatcher_);
    HCCL_CONFIG_INFO(HCCL_ALG, "[%s] Run TEMPLATE_ALL_2_ALL_V_DIRECT_FULL_MESH in COMM_COMBINE_ORDER", __func__);
    CHK_SMART_PTR_NULL(tempAlg);

    PrepareData prepareData;
    prepareData.stream = param.stream;
    prepareData.userRank = topoAttr_.userRank;
    prepareData.userRankSize = topoAttr_.userRankSize;
    prepareData.linksPtr = &level0CommInfo.links;
    prepareData.localSendRecvInfoPtr = &localSendRecvInfo_;
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

    // 如果使能alltoallv aicpu cache
    if (needAlltoallvCache_) {
        // 注意: 一定是alltoallv类算子才有可能设置needAlltoallvCache_, 让alltoallv temp alg感知cache并保存算法中间结果
        CHK_PRT_RET(
            !(param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC),
            HCCL_ERROR(
                "[CollRunAlltoAllDirectFullmesh][KernelRun] needAlltoallvCache_[%u] opType[%u]", needAlltoallvCache_,
                param.opType),
            HCCL_E_INTERNAL);

        // 使能alltoallv temp alg感知alltoallv aicpu cache
        prepareData.needAlltoallvCache = true;
    } else {
        prepareData.needAlltoallvCache = false;
    }

    CHK_RET(tempAlg->Prepare(prepareData));

    CHK_RET(tempAlg->RunAsync());

    if (needAlltoallvCache_) {
        // 在tempAlg被销毁前保存hcclOffset-dstRank之间的mapping信息
        // 注意: CollRunAlltoAllDirectFullmesh executor使用的一定是AlltoAllVDirectFullMesh template
        hcclOffsetDstRanksMap_.clear();
        HCCL_INFO(
            "[CollRunAlltoAllDirectFullmesh][KernelRun] get hcclOffset-dstRanks mapping for AlltoAllVDirectFullMesh");
        CHK_RET(tempAlg->GetHcclOffsetDstRanksMap(hcclOffsetDstRanksMap_));
    }

    HCCL_INFO("[CollRunAlltoAllDirectFullmesh] executor run success.");
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
HcclResult CollRunAlltoAllDirectFullmesh::Getlevel1CommRank(SubCommInfo& level1CommInfo)
{
    HCCL_INFO(
        "[CollRunAlltoAllDirectFullmesh] Getlevel1CommRank userRank[%u]--userRankSize[%u].", topoAttr_.userRank,
        topoAttr_.userRankSize);
    level1CommInfo.localRank = topoAttr_.userRank;
    level1CommInfo.localRankSize = topoAttr_.userRankSize;
    return HCCL_SUCCESS;
}

HcclResult
CollRunAlltoAllDirectFullmesh::SelectTempAlg(std::unique_ptr<AlgTemplateBase>& level1TempAlg, u32 level1RankSize)
{
    (void)level1RankSize;
    HCCL_INFO("[CollRunAlltoAllDirectFullmesh] SelectTempAlg.");
    level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_ALL_2_ALL_V_DIRECT_FULL_MESH, dispatcher_);
    CHK_SMART_PTR_NULL(level1TempAlg);

    return HCCL_SUCCESS;
}

HcclResult CollRunAlltoAllDirectFullmesh::GetDevNumInlocalPod(u32& devNumInlocalPod)
{
    HCCL_INFO("[CollRunAlltoAllDirectFullmesh] GetDevNumInlocalPod.");
    // 获取当前超节点内总卡数
    u32 rankIdxInPod = INVALID_VALUE_RANKID;
    CHK_RET(GetLocalSDMAGroupInfo(topoAttr_.userRank, devNumInlocalPod, rankIdxInPod));

    return HCCL_SUCCESS;
}
REGISTER_EXEC("RunAlltoAllDirectFullmesh", AlltoAllVDirectFullMesh, CollRunAlltoAllDirectFullmesh);
} // namespace hccl

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include "hccl_impl.h"
#include "alltoall_operator.h"
#include "all_reduce_operator.h"
#include "coll_alg_op_registry.h"
#include "topo_matcher.h"
#include "topo_info_extractor.h"
#include "alg_configurator.h"
#include "hccl_alg.h"

namespace hccl {
constexpr u32 TINY_MEMORY_SIZE = 32; // sendBuff或recvBuff为空时, 使用的DeviceMem大小

HcclAlg::HcclAlg(CCLBufferManager& cclBufferManager, const HcclDispatcher dispatcher, const HcclDispatcher vDispatcher)
    : cclBufferManager_(cclBufferManager),
      dispatcher_(dispatcher),
      vDispatcher_(vDispatcher)
{}

HcclAlg::~HcclAlg()
{
#ifndef OPEN_HCCL_TEST
    pimpl_ = nullptr;
#endif
}

HcclResult HcclAlg::Init(
    std::unique_ptr<WorkspaceResource>& workSpaceRes, const std::unique_ptr<NotifyPool>& notifyPool,
    std::map<HcclIpAddress, HcclNetDevCtx>& netDevCtxMap, const std::unique_ptr<QueueNotifyManager>& queueNotifyManager,
    HcclAlgoAttr& algoAttr, HcclTopoAttr& topoAttr, bool isHeterogComm)
{
    CHK_RET(Init(algoAttr, topoAttr, isHeterogComm));

#ifndef OPEN_HCCL_TEST
    // 老流程使用，新流程的LLT不编译相关的代码
    pimpl_.reset((new (std::nothrow) hcclImpl(
        dispatcher_, notifyPool, netDevCtxMap, queueNotifyManager, workSpaceRes, cclBufferManager_, algoAttr_,
        topoAttr_, algConfigurator_, topoInfoEx_)));
    CHK_SMART_PTR_NULL(pimpl_);
    CHK_RET(pimpl_->Init(isHeterogComm));
#endif
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::Init(HcclAlgoAttr& algoAttr, HcclTopoAttr& topoAttr, bool isHeterogComm)
{
    algoAttr_ = algoAttr;
    topoAttr_ = topoAttr;
    algConfigurator_.reset(new (std::nothrow) AlgConfigurator(algoAttr_, topoAttr_));
    CHK_SMART_PTR_NULL(algConfigurator_);
    CHK_RET(algConfigurator_->Init(isHeterogComm));

    TopoType topoType = TopoType::TOPO_TYPE_RESERVED;
    algConfigurator_->GetTopoType(topoType);
    topoInfoEx_.reset(new (std::nothrow) TopoInfoExtractor(algoAttr_, topoAttr_, topoType));
    CHK_SMART_PTR_NULL(topoInfoEx_);
    CHK_RET(topoInfoEx_->Init(algoAttr_.commAlgoConfig));

    std::vector<std::vector<std::vector<u32>>> CommPlaneRanks;
    CHK_RET(topoInfoEx_->GetCommPlaneRanks(CommPlaneRanks));

    std::vector<bool> isBridgeVector;
    topoInfoEx_->GetIsBridgeVector(isBridgeVector);

    std::vector<std::vector<std::vector<u32>>> serverAndsuperPodToRank;
    CHK_RET(topoInfoEx_->GetRankVecInfo(serverAndsuperPodToRank));

    HcclTopoInfo topoInfo;
    CHK_RET(InitTopoInfo(topoInfo, topoAttr_));

    HcclAlgoInfo algoInfo;
    CHK_RET(InitAlgoInfo(algoInfo, algoAttr_));

    HcclExternalEnable externalEnable;
    CHK_RET(InitExternalEnable(externalEnable));

    topoMatcher_.reset((new (std::nothrow) TopoMatcher(
        CommPlaneRanks, isBridgeVector, topoInfo, algoInfo, externalEnable, serverAndsuperPodToRank)));
    CHK_SMART_PTR_NULL(topoMatcher_);

    parallelTaskLoader_.reset(
        static_cast<ParallelTaskLoader*>(new (std::nothrow) ParallelTaskLoader(topoAttr_.deviceLogicId, dispatcher_)));
    CHK_SMART_PTR_NULL(parallelTaskLoader_);

#ifndef OPEN_HCCL_TEST
    if (static_cast<s32>(topoAttr_.devicePhyId) != HOST_DEVICE_ID) {
        CHK_RET(DeviceMem::alloc(tinySendRecvMem_, TINY_MEMORY_SIZE));
    }
#endif
    return HCCL_SUCCESS;
}

std::unique_ptr<CollAlgOperator> HcclAlg::GetAlgOperator(const HcclCMDType& opType, HcclWorkflowMode workflowMode)
{
    (void)workflowMode;
    if (!topoMatcher_) {
        HCCL_ERROR("[HcclAlg][GetAlgOperator] topoMatcher ptr is null, get algorithm operator failed.");
        return nullptr;
    }
    std::unique_ptr<CollAlgOperator> operation = CollAlgOpRegistry::Instance().GetAlgOp(
        opType, algConfigurator_.get(), cclBufferManager_, dispatcher_, topoMatcher_);
    CHK_PRT_RET(
        operation == nullptr, HCCL_ERROR("[HcclAlg][GetAlgOperator] GetAlgOp return nullptr, opType[%d]", opType),
        nullptr);
    if (opType == HcclCMDType::HCCL_CMD_ALLTOALL || opType == HcclCMDType::HCCL_CMD_ALLTOALLV
        || opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        AlltoAllOperator* alltoAllOperator = dynamic_cast<AlltoAllOperator*>(operation.get());
        alltoAllOperator->SetVirtualDispatcher(vDispatcher_);
        alltoAllOperator->SetParallelTaskLoader(parallelTaskLoader_.get());
    }
    HCCL_INFO("[AIG][GetAlgOperator] GetAlgOperator done");
    return operation;
}

HcclResult
HcclAlg::GetAlltoAllStagedWorkSpaceMemSize(std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo, u64& memSize)
{
    AlltoAllOperator operation(algConfigurator_.get(), cclBufferManager_, dispatcher_, topoMatcher_);
    operation.SetVirtualDispatcher(vDispatcher_);
    operation.SetParallelTaskLoader(parallelTaskLoader_.get());
    return operation.GetAlltoAllStagedWorkSpaceMemSize(allMeshAggregationSendRecvInfo, memSize);
}

HcclResult HcclAlg::GetAllReduceScratchSize(const u64 count, const HcclDataType dataType, u64& scratchSize)
{
    AllReduceOperator operation(algConfigurator_.get(), cclBufferManager_, dispatcher_, topoMatcher_);
    return operation.GetAllReduceScratchSize(count, dataType, scratchSize);
}

HcclResult HcclAlg::GetTopoType(TopoType& topoType)
{
    algConfigurator_->GetTopoType(topoType);
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::SetAlgType(AlgType algType, HcclCMDType opType)
{
    return algConfigurator_->SetAlgType(algType, opType);
}

HcclResult HcclAlg::GetAlgType(AlgType& algType, HcclCMDType opType)
{
    return algConfigurator_->GetAlgType(algType, opType);
}

HcclResult HcclAlg::SupportDeterministicOptim(bool& isDeterministicOptim)
{
    isDeterministicOptim = algConfigurator_->SupportDeterministicOptim();
    return HCCL_SUCCESS;
}

u8 HcclAlg::GetDeterministicConfig() const { return topoMatcher_->GetDeterministicConfig(); }

HcclResult HcclAlg::SetDeterministicConfig(const u8 deterministic)
{
    CHK_RET(topoMatcher_->SetDeterministicConfig(deterministic));
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::SetAivModeConfig(const bool aivMode)
{
    CHK_RET(topoMatcher_->SetAivModeConfig(aivMode));
    return HCCL_SUCCESS;
}

bool HcclAlg::GetAicpuUnfoldConfig() const { return topoMatcher_->GetAicpuUnfoldConfig(); }

bool HcclAlg::GetAivModeConfig() const { return topoMatcher_->GetAivModeConfig(); }

HcclResult HcclAlg::SetAicpuUnfoldConfig(const bool aicpuUnfold)
{
    CHK_RET(topoMatcher_->SetAicpuUnfoldConfig(aicpuUnfold));
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::SetExecTimeOutConfig(const s32 execTimeOut)
{
    CHK_RET(topoMatcher_->SetExecTimeOutConfig(execTimeOut));
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::SetAlgoConfig(const std::map<HcclCMDType, std::vector<HcclAlgoType>>& algoMap)
{
    CHK_RET(topoMatcher_->SetAlgoConfig(algoMap));
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::GetRankVecInfo(std::vector<std::vector<std::vector<u32>>>& serverAndsuperPodToRank)
{
    CHK_RET(topoInfoEx_->GetRankVecInfo(serverAndsuperPodToRank));
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::GetIsBridgeVector(std::vector<bool>& isBridgeVector)
{
    topoInfoEx_->GetIsBridgeVector(isBridgeVector);
    return HCCL_SUCCESS;
}
HcclResult HcclAlg::GetCommPlaneRanks(std::vector<std::vector<std::vector<u32>>>& commPlaneRanks)
{
    CHK_RET(topoInfoEx_->GetCommPlaneRanks(commPlaneRanks));
    return HCCL_SUCCESS;
}

void HcclAlg::GetCommPlaneVector(std::vector<std::vector<std::vector<RankInfo>>>& commPlaneVector)
{
    topoInfoEx_->GetCommPlaneVector(commPlaneVector);
}

HcclResult HcclAlg::SetOnlyAivModeConfig(const bool isOnlyAiv)
{
    CHK_RET(topoMatcher_->SetOnlyAivModeConfig(isOnlyAiv));
    return HCCL_SUCCESS;
}

HcclResult
HcclAlg::GetCommPlaneSubGroupVector(std::vector<std::vector<std::vector<std::vector<u32>>>>& commPlaneSubGroupVector)
{
    topoMatcher_->GetCommPlaneSubGroupVector(commPlaneSubGroupVector);
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::GetAHCAlgOption(std::map<AHCConcOpType, TemplateType>& ahcAlgOption)
{
    topoMatcher_->GetAHCAlgOption(ahcAlgOption);
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::GetIsUsedRdmaMap(std::unordered_map<u32, bool>& isUsedRdmaMap)
{
    CHK_RET(topoInfoEx_->GetIsUsedRdmaMap(isUsedRdmaMap));
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::GetTinyMem(DeviceMem& tinySendRecvMem)
{
    tinySendRecvMem = tinySendRecvMem_;
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::InitExternalEnable(HcclExternalEnable& externalEnable)
{
    externalEnable.enableFfts = GetExternalInputHcclEnableFfts();
    externalEnable.deterministic = GetExternalInputHcclDeterministicV2();
    externalEnable.intraRoceSwitch = GetExternalInputIntraRoceSwitch();
    externalEnable.dumpDebug = GetExternalInputHcclDumpDebug();
    externalEnable.aivMode = GetExternalInputHcclAivMode();
    externalEnable.aicpuUnfold = GetExternalInputHcclAicpuUnfold();
    externalEnable.execTimeOut = GetInternalExecTimeOut();
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::InitTopoInfo(HcclTopoInfo& topoInfo, HcclTopoAttr& topoAttr)
{
    topoInfo.userRank = topoAttr.userRank;
    topoInfo.userRankSize = topoAttr.userRankSize;
    topoInfo.devicePhyId = topoAttr.devicePhyId;
    topoInfo.deviceLogicId = topoAttr.deviceLogicId;
    topoInfo.nicList = topoAttr.nicList;
    topoInfo.isSingleMeshAggregation = topoAttr.isSingleMeshAggregation;
    topoInfo.deviceNumPerAggregation = topoAttr.deviceNumPerAggregation;
    topoInfo.superPodNum = topoAttr.superPodNum;
    topoInfo.deviceType = topoAttr.deviceType;
    topoInfo.serverNum = topoAttr.serverNum;
    topoInfo.meshAggregationRankSize = topoAttr.meshAggregationRankSize;
    topoInfo.multiModuleDiffDeviceNumMode = topoAttr.multiModuleDiffDeviceNumMode;
    topoInfo.multiSuperPodDiffServerNumMode = topoAttr.multiSuperPodDiffServerNumMode;
    topoInfo.multiSuperPodDiffDeviceNumMode = topoAttr.multiSuperPodDiffDeviceNumMode;
    topoInfo.isDiffDeviceType = topoAttr.isDiffDeviceType;
    topoInfo.gcdDeviceNumPerAggregation = topoAttr.gcdDeviceNumPerAggregation;
    topoInfo.pairLinkCounter = topoAttr.pairLinkCounter;
    topoInfo.isDiffDeviceModule = topoAttr.isDiffDeviceModule;
    topoInfo.realUserRank = topoAttr.realUserRank;
    topoInfo.moduleNum = topoAttr.moduleNum;
    topoInfo.useSuperPodMode = topoAttr.useSuperPodMode;
    topoInfo.isARSDoubleRing = topoAttr.isARSDoubleRing;

    topoInfoEx_->GetCommPlaneSubGroupVector(topoInfo.CommPlaneSubGroupVector);
    topoInfoEx_->GetAHCAlgOption(topoInfo.ahcAlgOption);

    algConfigurator_->GetTopoType(topoInfo.topoType);
    topoInfo.is310P3Common = Is310P3Common(algoAttr_.isHaveCpuRank, topoAttr_.deviceType);
    std::unordered_map<u32, bool> isUsedRdmaMap;
    CHK_RET(topoInfoEx_->GetIsUsedRdmaMap(isUsedRdmaMap));
    topoInfo.isUsedRdmaMap = isUsedRdmaMap;
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::InitAlgoInfo(HcclAlgoInfo& algoInfo, HcclAlgoAttr& algoAttr)
{
    algoInfo.identifier = algoAttr.identifier;
    algoInfo.inlineReduceSwitchOn = algoAttr.inlineReduceSwitchOn;
    algoInfo.isUsedRdmaLevel0 = algoAttr.isUsedRdmaLevel0;
    algoInfo.isSupportAtomicWrite = false; // 涉及到任务编排，当前不能只判断本机驱动版本是否支持
    return HCCL_SUCCESS;
}

#ifndef OPEN_HCCL_TEST
// 上层保证，以下方法在初始化成功后才会调用，所以未对pimpl_进行保护判断
HcclResult HcclAlg::ReleaseCommInfos() { return pimpl_->ReleaseCommInfos(); }

HcclResult HcclAlg::ClearOpResource(const std::string& tag) { return pimpl_->ClearOpResource(tag); }

HcclResult HcclAlg::CreateMutiStreamRes(
    const std::string& tag, Stream& stream, level1StreamInfo_t& streamInfo, AlgType algType, bool isAicpuModeEn)
{
    return pimpl_->CreateMutiStreamRes(tag, stream, streamInfo, algType, isAicpuModeEn);
}

HcclResult HcclAlg::CreateComm(
    const std::string& tag, DeviceMem& inputMem, DeviceMem& outputMem, AlgType algType,
    std::unique_ptr<CommInfo>& commInfo, u32 root, bool isP2p, bool isAicpuModeEn)
{
    return pimpl_->CreateComm(tag, inputMem, outputMem, algType, commInfo, root, isP2p, isAicpuModeEn);
}

HcclResult HcclAlg::CreateComm(
    const std::string& tag, DeviceMem& inputMem, DeviceMem& outputMem, AlgType algType, u32 root, bool isP2p)
{
    return pimpl_->CreateComm(tag, inputMem, outputMem, algType, root, isP2p);
}

void HcclAlg::Break() { pimpl_->Break(); }

HcclResult HcclAlg::SetHDCModeInfo(
    std::unordered_map<std::string, std::map<u32, HcclIpAddress>>& rankDevicePhyIdNicInfoMap,
    std::vector<u32>& ranksPort, std::vector<u32>& vnicRanksPort, bool isSetHDCModeInfo, bool isUseRankPort)
{
    pimpl_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap, ranksPort, vnicRanksPort, isSetHDCModeInfo, isUseRankPort);
    return HCCL_SUCCESS;
}
#endif
} // namespace hccl

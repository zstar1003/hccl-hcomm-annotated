/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_alg.h"
#include "topo_matcher.h"

namespace hccl {
constexpr u32 TINY_MEMORY_SIZE = 32; // sendBuff或recvBuff为空时, 使用的DeviceMem大小

HcclAlg::HcclAlg(CCLBufferManager& cclBufferManager, const HcclDispatcher dispatcher, const HcclDispatcher vDispatcher)
    : cclBufferManager_(cclBufferManager),
      dispatcher_(dispatcher),
      vDispatcher_(vDispatcher)
{}

HcclAlg::~HcclAlg() {}

HcclResult HcclAlg::Init(
    std::unique_ptr<WorkspaceResource>& workSpaceRes, const std::unique_ptr<NotifyPool>& notifyPool,
    std::map<HcclIpAddress, HcclNetDevCtx>& netDevCtxMap, const std::unique_ptr<QueueNotifyManager>& queueNotifyManager,
    HcclAlgoAttr& algoAttr, HcclTopoAttr& topoAttr, bool isHeterogComm)
{
    (void)workSpaceRes;
    (void)notifyPool;
    (void)netDevCtxMap;
    (void)queueNotifyManager;
    (void)algoAttr;
    (void)topoAttr;
    (void)isHeterogComm;
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::Init(HcclAlgoAttr& algoAttr, HcclTopoAttr& topoAttr, bool isHeterogComm)
{
    (void)algoAttr;
    (void)topoAttr;
    (void)isHeterogComm;
    return HCCL_SUCCESS;
}

HcclResult
HcclAlg::GetAlltoAllStagedWorkSpaceMemSize(std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo, u64& memSize)
{
    (void)allMeshAggregationSendRecvInfo;
    (void)memSize;
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::GetAllReduceScratchSize(const u64 count, const HcclDataType dataType, u64& scratchSize)
{
    (void)count;
    (void)dataType;
    (void)scratchSize;
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::GetTopoType(TopoType& topoType)
{
    (void)topoType;
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::SetAlgType(AlgType algType, HcclCMDType opType)
{
    (void)algType;
    (void)opType;
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::GetAlgType(AlgType& algType, HcclCMDType opType)
{
    (void)algType;
    (void)opType;
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::SupportDeterministicOptim(bool& isDeterministicOptim)
{
    (void)isDeterministicOptim;
    return HCCL_SUCCESS;
}

u8 HcclAlg::GetDeterministicConfig() const { return 0; }

HcclResult HcclAlg::SetDeterministicConfig(const u8 deterministic)
{
    (void)deterministic;
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::SetAivModeConfig(const bool aivMode)
{
    (void)aivMode;
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::SetOnlyAivModeConfig(const bool isOnlyAiv)
{
    (void)isOnlyAiv;
    return HCCL_SUCCESS;
}

bool HcclAlg::GetAicpuUnfoldConfig() const { return false; }

HcclResult HcclAlg::SetAicpuUnfoldConfig(const bool aicpuUnfold)
{
    (void)aicpuUnfold;
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::SetExecTimeOutConfig(const s32 execTimeOut)
{
    (void)execTimeOut;
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::SetAlgoConfig(const std::map<HcclCMDType, std::vector<HcclAlgoType>>& algoMap)
{
    (void)algoMap;
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::GetRankVecInfo(std::vector<std::vector<std::vector<u32>>>& serverAndsuperPodToRank)
{
    (void)serverAndsuperPodToRank;
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::GetIsBridgeVector(std::vector<bool>& isBridgeVector)
{
    (void)isBridgeVector;
    return HCCL_SUCCESS;
}
HcclResult HcclAlg::GetCommPlaneRanks(std::vector<std::vector<std::vector<u32>>>& commPlaneRanks)
{
    (void)commPlaneRanks;
    return HCCL_SUCCESS;
}

void HcclAlg::GetCommPlaneVector(std::vector<std::vector<std::vector<RankInfo>>>& commPlaneVector)
{
    (void)commPlaneVector;
}

HcclResult
HcclAlg::GetCommPlaneSubGroupVector(std::vector<std::vector<std::vector<std::vector<u32>>>>& commPlaneSubGroupVector)
{
    (void)commPlaneSubGroupVector;
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::GetAHCAlgOption(std::map<AHCConcOpType, TemplateType>& ahcAlgOption)
{
    (void)ahcAlgOption;
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::GetIsUsedRdmaMap(std::unordered_map<u32, bool>& isUsedRdmaMap)
{
    (void)isUsedRdmaMap;
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::GetTinyMem(DeviceMem& tinySendRecvMem)
{
    (void)tinySendRecvMem;
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::InitExternalEnable(HcclExternalEnable& externalEnable)
{
    (void)externalEnable;
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::InitTopoInfo(HcclTopoInfo& topoInfo, HcclTopoAttr& topoAttr)
{
    (void)topoInfo;
    (void)topoAttr;
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::InitAlgoInfo(HcclAlgoInfo& algoInfo, HcclAlgoAttr& algoAttr)
{
    (void)algoInfo;
    (void)algoAttr;
    return HCCL_SUCCESS;
}

#ifndef OPEN_HCCL_TEST
// 上层保证，以下方法在初始化成功后才会调用，所以未对pimpl_进行保护判断
HcclResult HcclAlg::ReleaseCommInfos() { return HCCL_SUCCESS; }

HcclResult HcclAlg::ClearOpResource(const std::string& tag)
{
    (void)tag;
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::CreateMutiStreamRes(
    const std::string& tag, Stream& stream, level1StreamInfo_t& streamInfo, AlgType algType, bool isAicpuModeEn)
{
    (void)tag;
    (void)stream;
    (void)streamInfo;
    (void)algType;
    (void)isAicpuModeEn;
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::CreateComm(
    const std::string& tag, DeviceMem& inputMem, DeviceMem& outputMem, AlgType algType,
    std::unique_ptr<CommInfo>& commInfo, u32 root, bool isP2p, bool isAicpuModeEn)
{
    (void)tag;
    (void)inputMem;
    (void)outputMem;
    (void)algType;
    (void)commInfo;
    (void)root;
    (void)isP2p;
    (void)isAicpuModeEn;
    return HCCL_SUCCESS;
}

HcclResult HcclAlg::CreateComm(
    const std::string& tag, DeviceMem& inputMem, DeviceMem& outputMem, AlgType algType, u32 root, bool isP2p)
{
    (void)tag;
    (void)inputMem;
    (void)outputMem;
    (void)algType;
    (void)root;
    (void)isP2p;
    return HCCL_SUCCESS;
}

void HcclAlg::CancelCommRes(const std::string& tag) { (void)tag; }

void HcclAlg::Break() {}

HcclResult HcclAlg::SetHDCModeInfo(
    std::unordered_map<std::string, std::map<u32, HcclIpAddress>>& rankDevicePhyIdNicInfoMap,
    std::vector<u32>& ranksPort, std::vector<u32>& vnicRanksPort, bool isSetHDCModeInfo, bool isUseRankPort)
{
    (void)rankDevicePhyIdNicInfoMap;
    (void)ranksPort;
    (void)vnicRanksPort;
    (void)isSetHDCModeInfo;
    (void)isUseRankPort;
    return HCCL_SUCCESS;
}
#endif
} // namespace hccl

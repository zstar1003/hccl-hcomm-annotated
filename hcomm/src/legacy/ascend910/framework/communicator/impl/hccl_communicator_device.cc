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
#include <atomic>
#include <chrono>
#include <thread>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <sys/time.h>
#include "hccl_aiv.h"
#include "hccl_communicator.h"

using namespace std;

constexpr u32 MODULE_NUM_FOUR = 4;

namespace hccl {
HcclCommunicator::HcclCommunicator()
    : dispatcher_(nullptr),
      vDispatcher_(nullptr),
      notifyPool_(nullptr),
      initializedFlag_(ATOMIC_FLAG_INIT),
      userRank_(INVALID_VALUE_RANKID),
      realUserRank_(INVALID_VALUE_RANKID),
      userRankSize_(INVALID_VALUE_RANKSIZE),
      drvInit_(false),
      inlineReduceSwitchOn_(true),
      nicDeployment_(NICDeployment::NIC_DEPLOYMENT_DEVICE),
      devicePhyId_(INVALID_UINT),
      deviceLogicId_(-1),
      localRank_(INVALID_VALUE_RANKID),
      hostSocketHandle_(nullptr),
      isUsedRdmaLevel0_(false),
      nicInitialized_(0),
      hcomGroupNicInit_(false),
      profilingMode_(HcomProfilingMode::PROFILING_CLOSE),
      raResourceInit_(false),
      interServer_(false),
      isSingleMeshAggregation_(false),
      cclBufferManager_(CCLBufferManager()),
      isExecuteProfilingInit_(false),
      deviceType_(DevType::DEV_TYPE_COUNT),
      commHandle_(nullptr),
      commWorkMode_(WorkMode::HCCL_MODE_NORMAL),
      meshAggregationRankSize_(0),
      isHaveCpuRank_(false),
      ranktableCrc_(0),
      multiModuleDiffDeviceNumMode_(false),
      multiSuperPodDiffServerNumMode_(false),
      isStandardCard_(false),
      is310PDuoCard_(false),
      hccsPortNum_(-1),
      loopBackIp_(HcclIpAddress(COMM_LOOPBACK_IP)),
      profilingInitiated_(false),
      callbackThreadId_(INVALID_U64),
      role_(SERVER_ROLE_SOCKET),
      isHostUseDevNic_(false),
      isAllRankSamePlane_(false),
      serverNum_(0),
      moduleNum_(0)
{}

HcclCommunicator::HcclCommunicator(const CommConfig& commConfig)
    : dispatcher_(nullptr),
      vDispatcher_(nullptr),
      notifyPool_(nullptr),
      initializedFlag_(ATOMIC_FLAG_INIT),
      userRank_(INVALID_VALUE_RANKID),
      realUserRank_(INVALID_VALUE_RANKID),
      userRankSize_(INVALID_VALUE_RANKSIZE),
      drvInit_(false),
      inlineReduceSwitchOn_(true),
      nicDeployment_(NICDeployment::NIC_DEPLOYMENT_DEVICE),
      devicePhyId_(INVALID_UINT),
      deviceLogicId_(-1),
      localRank_(INVALID_VALUE_RANKID),
      hostSocketHandle_(nullptr),
      isUsedRdmaLevel0_(false),
      nicInitialized_(0),
      hcomGroupNicInit_(false),
      profilingMode_(HcomProfilingMode::PROFILING_CLOSE),
      raResourceInit_(false),
      interServer_(false),
      isSingleMeshAggregation_(false),
      cclBufferManager_(CCLBufferManager()),
      isExecuteProfilingInit_(false),
      deviceType_(DevType::DEV_TYPE_COUNT),
      commHandle_(nullptr),
      commWorkMode_(WorkMode::HCCL_MODE_NORMAL),
      meshAggregationRankSize_(0),
      isHaveCpuRank_(false),
      ranktableCrc_(0),
      multiModuleDiffDeviceNumMode_(false),
      multiSuperPodDiffServerNumMode_(false),
      isStandardCard_(false),
      is310PDuoCard_(false),
      hccsPortNum_(-1),
      loopBackIp_(HcclIpAddress(COMM_LOOPBACK_IP)),
      profilingInitiated_(false),
      callbackThreadId_(INVALID_U64),
      role_(SERVER_ROLE_SOCKET),
      isHostUseDevNic_(false),
      isAllRankSamePlane_(false),
      serverNum_(0),
      moduleNum_(0)
{
    commConfig_ = commConfig;
}

HcclCommunicator::~HcclCommunicator() {}

HcclResult
HcclCommunicator::Init([[maybe_unused]] HcclCommParams& params, [[maybe_unused]] const RankTable_t& rankTable)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Init(
    [[maybe_unused]] HcclCommParams& params, [[maybe_unused]] const std::vector<RankInfo>& rankList,
    [[maybe_unused]] WorldGroupInfo& groupCommonData)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitOneSidedService([[maybe_unused]] const RankTable_t& rankTable) { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::InitOneSidedServiceNetDevCtx([[maybe_unused]] u32 remoteRankId) { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::DeInitOneSidedServiceNetDevCtx() { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::GetOneSidedService([[maybe_unused]] IHcclOneSidedService** service)
{
    return HCCL_SUCCESS;
}

HcclResult
HcclCommunicator::OneSidedServiceStartListen([[maybe_unused]] NicType nicType, [[maybe_unused]] HcclNetDevCtx netDevCtx)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetOneSidedServiceDevIpAndPort(
    [[maybe_unused]] NicType nicType, [[maybe_unused]] HcclIpAddress& ipAddress, [[maybe_unused]] u32& port)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::DeinitOneSidedService() { return HCCL_SUCCESS; }

bool HcclCommunicator::IsSupportSymmetricMemory([[maybe_unused]] HcclCMDType opType, [[maybe_unused]] OpParam& opParam)
{
    return false;
}

bool HcclCommunicator::IsSupportZeroCopy([[maybe_unused]] const OpParam& opParam) const { return false; }

HcclResult HcclCommunicator::PrepareZeroCopy(
    [[maybe_unused]] const std::string& algName, [[maybe_unused]] const AlgDesc& algDesc,
    [[maybe_unused]] OpParam& opParam)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::UpdateZeroCopy(
    [[maybe_unused]] const OpParam& opParam, [[maybe_unused]] const AlgResourceResponse& algResource)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildZeroCopyParam() { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::InitCommParams([[maybe_unused]] HcclCommParams& params) { return HCCL_SUCCESS; }

bool HcclCommunicator::Is310PDuoCard() { return false; }

// 910B A+X 在RDMA未启用情况下，两模块间的device数目需要一致且两模块中使用的卡都在同一平面上
HcclResult HcclCommunicator::CheckSingleServerComm([[maybe_unused]] const std::vector<RankInfo_t>& rankList) const
{
    return HCCL_SUCCESS;
}

HcclResult
HcclCommunicator::CheckDataType([[maybe_unused]] const HcclDataType dataType, [[maybe_unused]] bool needReduce)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitZeroCopyMemoryAgent() { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::DeinitZeroCopyMemoryAgent([[maybe_unused]] bool inDestructor) { return HCCL_SUCCESS; }

u8 HcclCommunicator::GetConfigAclGraphZeroCopyEnable() { return 0; }

HcclResult HcclCommunicator::ClearResMap(
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] bool& findTag, bool aclGraphDestroyCbk)
{
    (void)aclGraphDestroyCbk;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ClearOpResource([[maybe_unused]] const std::string& tag, bool aclGraphDestroyCbk)
{
    (void)aclGraphDestroyCbk;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CreateOpBasedResources(
    [[maybe_unused]] const HcclCMDType& opType, [[maybe_unused]] const std::string& tag,
    [[maybe_unused]] const HcomCollOpInfo& opInfo)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult
HcclCommunicator::CreateRemoteOpBasedResources([[maybe_unused]] u64 memSize, [[maybe_unused]] const std::string& tag)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclCommunicator::DestroyRemoteOpBasedMem([[maybe_unused]] const std::string& tag)
{
    return HCCL_E_NOT_SUPPORT;
}

bool HcclCommunicator::IsAtomicInit() { return false; }

bool HcclCommunicator::IsNeedNicInit() { return false; }

HcclResult HcclCommunicator::GetBandWidthPerNPU([[maybe_unused]] u32 level, [[maybe_unused]] float& bandWidth)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclCommunicator::GetDeviceNumPerAggregation([[maybe_unused]] u32& deviceNumPerAggregation)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitHccpChannel() { return HCCL_SUCCESS; }

std::vector<RankInfo> HcclCommunicator::GetRankLists() { return {}; }

HcclResult HcclCommunicator::CheckReduceDataType(
    [[maybe_unused]] const HcclDataType dataType, [[maybe_unused]] const HcclReduceOp op)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetAlgType([[maybe_unused]] AlgType& algType, [[maybe_unused]] HcclCMDType opType)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclCommunicator::GetCommParams([[maybe_unused]] HcclCommParams& params) { return HCCL_E_NOT_SUPPORT; }

HcclResult HcclCommunicator::GetCommRankTable([[maybe_unused]] RankTable_t& rankTable) { return HCCL_E_NOT_SUPPORT; }

HcclResult HcclCommunicator::InitPara() { return HCCL_SUCCESS; }

bool HcclCommunicator::IsStandardCard() { return false; }

HcclResult HcclCommunicator::InitOpRetry() { return HCCL_SUCCESS; }

bool HcclCommunicator::CompareWithServerId(
    [[maybe_unused]] const ServerInfo_t& left, [[maybe_unused]] const ServerInfo_t& right)
{
    return false;
}

bool HcclCommunicator::CompareWithNicName(
    [[maybe_unused]] const NetworkInfo_t& left, [[maybe_unused]] const NetworkInfo_t& right)
{
    return false;
}

bool HcclCommunicator::CompareWithUserRank(
    [[maybe_unused]] const RankInfo& left, [[maybe_unused]] const RankInfo& right)
{
    return false;
}

HcclResult HcclCommunicator::InitPreResource([[maybe_unused]] const RankTable_t& rankTable) { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::InitTcpMode([[maybe_unused]] const RankTable_t& rankTable) const { return HCCL_SUCCESS; }

bool HcclCommunicator::IsEnableBackupLink() { return false; }

HcclResult HcclCommunicator::InitRaResource() { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::DisablePreResource() { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::GetWorkspaceSubStreamNum(
    [[maybe_unused]] u64 count, [[maybe_unused]] HcclDataType dataType, [[maybe_unused]] HcclReduceOp op,
    [[maybe_unused]] const std::string& algName, [[maybe_unused]] u64& streamNum, [[maybe_unused]] u64 dataSize,
    [[maybe_unused]] bool ifAiv, [[maybe_unused]] HcclCMDType opType)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::DestroyNetworkResources() { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::SetWorkspaceResource(
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] void* memPtr, [[maybe_unused]] u64& maxSize,
    [[maybe_unused]] std::vector<rtStream_t>& stream)
{
    return HCCL_E_NOT_SUPPORT;
}

void HcclCommunicator::DestroyWorkspaceResource([[maybe_unused]] const std::string& tag) {}

HcclResult HcclCommunicator::AtomicInitSet() { return HCCL_SUCCESS; }

void HcclCommunicator::AtomicInitClear() {}

u32 HcclCommunicator::GetUserRank() const { return 0; }

u32 HcclCommunicator::GetGroupRank() const { return 0; }

u32 HcclCommunicator::GetRankSize() const { return 0; }

bool HcclCommunicator::GetNicInitialized() { return false; }

/*
    1. 选择算法
    2. 计算resource，存到request内
    3. 创建和分配资源
*/
HcclResult HcclCommunicator::HcclSelectAlg(
    [[maybe_unused]] HcclCMDType opType, [[maybe_unused]] u64 count, [[maybe_unused]] void* counts,
    [[maybe_unused]] HcclDataType dataType, [[maybe_unused]] HcclReduceOp op, [[maybe_unused]] int32_t aivCoreLimit,
    [[maybe_unused]] bool& ifAiv, [[maybe_unused]] std::string& algName)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::HcclCalcNumBlocks(
    [[maybe_unused]] HcclCMDType opType, [[maybe_unused]] u64 count, [[maybe_unused]] void* counts,
    [[maybe_unused]] HcclDataType dataType, [[maybe_unused]] int32_t aivCoreLimit,
    [[maybe_unused]] std::string& algName, [[maybe_unused]] u32& numBlocks)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::HcclGetAlgExecParam(
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] HcclCMDType opType, [[maybe_unused]] u64 count,
    [[maybe_unused]] void* inputPtr, [[maybe_unused]] void* outputPtr, [[maybe_unused]] bool clearEnable,
    [[maybe_unused]] HcclDataType dataType, [[maybe_unused]] HcclReduceOp op, [[maybe_unused]] void*& commContext,
    [[maybe_unused]] u64& len, [[maybe_unused]] u32 aivCoreLimit)
{
    return HCCL_SUCCESS;
}

HcclResult
HcclCommunicator::GetAivTag([[maybe_unused]] s32 tagNum, [[maybe_unused]] bool isCapture, [[maybe_unused]] s32& aivTag)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CheckDeviceType([[maybe_unused]] const DevType deviceType) const { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::CheckReductionOp([[maybe_unused]] const HcclReduceOp op) const { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::CheckUserRank([[maybe_unused]] const u32 userRank) const { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::CheckCount([[maybe_unused]] const u64 count) const { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::GetGroupRanksInfo(
    [[maybe_unused]] const std::vector<u32>& groupRanks, [[maybe_unused]] std::vector<RankInfo>& ranksInfo)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetGroupCommonData([[maybe_unused]] WorldGroupInfo& groupCommonData) const
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetWorkspaceMemSize(
    [[maybe_unused]] const std::string& opType, [[maybe_unused]] u64 count, [[maybe_unused]] HcclDataType dataType,
    [[maybe_unused]] u32& rankSize, [[maybe_unused]] u64& memSize, [[maybe_unused]] DevType& deviceType) const
{
    return HCCL_E_NOT_SUPPORT;
}

DeviceMem
HcclCommunicator::GetWorkspaceScracthMem([[maybe_unused]] const std::string& tag, [[maybe_unused]] u64 allocMemSize)
{
    return DeviceMem();
}

std::vector<Stream>
HcclCommunicator::GetWorkspaceSubStreams([[maybe_unused]] const std::string& tag, [[maybe_unused]] u32 num)
{
    return {};
}

HcclResult HcclCommunicator::InitProfiling() { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::DeinitProfiling() { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::RegistTaskExceptionHandler() const { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::UnRegistTaskExceptionHandler() const { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::GetInCCLbuffer([[maybe_unused]] void*& buffer, [[maybe_unused]] u64& size)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclCommunicator::GetOutCCLbuffer([[maybe_unused]] void*& buffer, [[maybe_unused]] u64& size)
{
    return HCCL_E_NOT_SUPPORT;
}

void HcclCommunicator::ReleaseCommCCLbuffer() {}

HcclResult HcclCommunicator::ReleaseCommInfos() { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::InitProfiler() { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::CreateCommCCLbuffer() { return HCCL_E_NOT_SUPPORT; }

HcclResult HcclCommunicator::InitCCLbuffer([[maybe_unused]] u64 inCCLbufferSize, [[maybe_unused]] u64 outCCLbufferSize)
{
    return HCCL_E_NOT_SUPPORT;
}

u32 HcclCommunicator::GetLocalNicPort([[maybe_unused]] NicType nicType) { return 0; }

HcclResult HcclCommunicator::InitNic([[maybe_unused]] bool isMC2ReInit) { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::DeinitNic() { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::RegisterRanksToDca() { return HCCL_E_NOT_SUPPORT; }

HcclResult HcclCommunicator::RegisterToHeartBeat() { return HCCL_E_NOT_SUPPORT; }

HcclResult HcclCommunicator::AddOpInfoToHeartBeat(
    [[maybe_unused]] const OpInfoDesc& opInfo, [[maybe_unused]] const std::string& tag)
{
    return HCCL_E_NOT_SUPPORT;
}

void HcclCommunicator::DeleteOpInfoToHeartBeat() {}

HcclResult HcclCommunicator::RegisterToHeartBeat([[maybe_unused]] u32 peerRankId, [[maybe_unused]] string& tag)
{
    return HCCL_E_NOT_SUPPORT;
}

void HcclCommunicator::UnRegisterToHeartBeat() {}

void HcclCommunicator::UnRegisterToCommConfiger() {}

HcclResult HcclCommunicator::SetGlobalWorkSpace([[maybe_unused]] std::vector<void*>& globalWorkSpaceAddr)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetandClearOverFlowTasks([[maybe_unused]] std::vector<HcclDumpInfo>& hcclDumpInfo)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetDeviceId([[maybe_unused]] s32& deviceId) const { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::GetCqeError([[maybe_unused]] HcclResult& result) { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::GetOpInconsistentError([[maybe_unused]] HcclResult& result) { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::SupportDeterministicOptim([[maybe_unused]] bool& isDeterministicOptim)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetHccsLinkNum([[maybe_unused]] u32& numHccsLink) { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::AllGather(
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] void* inputPtr, [[maybe_unused]] void* outputPtr,
    [[maybe_unused]] u64 inputCount, [[maybe_unused]] HcclDataType dataType, [[maybe_unused]] HcclRtStream stream,
    [[maybe_unused]] HcomCollOpInfo* opInfo)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllGatherV(
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] const void* sendBuf, [[maybe_unused]] u64 sendCount,
    [[maybe_unused]] const void* recvBuf, [[maybe_unused]] const void* recvCounts, [[maybe_unused]] const void* rdispls,
    [[maybe_unused]] HcclDataType dataType, [[maybe_unused]] HcclRtStream stream)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AicpuUnfold(
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] void* inputPtr, [[maybe_unused]] void* outputPtr,
    [[maybe_unused]] u64 count, [[maybe_unused]] HcclDataType dataType, [[maybe_unused]] HcclReduceOp op,
    [[maybe_unused]] HcclRtStream stream, [[maybe_unused]] HcclCMDType cmdType)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllGatherOutPlace(
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] void* inputPtr, [[maybe_unused]] void* outputPtr,
    [[maybe_unused]] u64 inputCount, [[maybe_unused]] HcclDataType dataType, [[maybe_unused]] HcclRtStream stream)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllGatherVOutPlace(
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] void* inputPtr, [[maybe_unused]] void* outputPtr,
    [[maybe_unused]] u64 inputCount, [[maybe_unused]] const void* outputCounts,
    [[maybe_unused]] const void* outputDispls, [[maybe_unused]] HcclDataType dataType,
    [[maybe_unused]] HcclRtStream stream)
{
    return HCCL_SUCCESS;
}

void HcclCommunicator::GetAndSetSyncMode([[maybe_unused]] SyncMode& preSyncMode, [[maybe_unused]] SyncMode newSyncMode)
{}

void HcclCommunicator::RestorePreSyncMode([[maybe_unused]] SyncMode preSyncMode, [[maybe_unused]] SyncMode newSyncMode)
{}

HcclResult HcclCommunicator::AllReduce(
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] void* inputPtr, [[maybe_unused]] void* outputPtr,
    [[maybe_unused]] u64 count, [[maybe_unused]] HcclDataType dataType, [[maybe_unused]] HcclReduceOp op,
    [[maybe_unused]] HcclRtStream stream, [[maybe_unused]] SyncMode syncMode,
    [[maybe_unused]] const HcomCollOpInfo* opInfo)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllReduceAicpuUnfold(
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] void* inputPtr, [[maybe_unused]] void* outputPtr,
    [[maybe_unused]] u64 count, [[maybe_unused]] HcclDataType dataType, [[maybe_unused]] HcclReduceOp op,
    [[maybe_unused]] HcclRtStream stream)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllReduceOutPlace(
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] void* inputPtr, [[maybe_unused]] void* outputPtr,
    [[maybe_unused]] u64 count, [[maybe_unused]] HcclDataType dataType, [[maybe_unused]] HcclReduceOp op,
    [[maybe_unused]] HcclRtStream stream, [[maybe_unused]] SyncMode syncMode)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AlltoAllV(
    [[maybe_unused]] const void* sendBuf, [[maybe_unused]] const void* sendCounts, [[maybe_unused]] const void* sdispls,
    [[maybe_unused]] HcclDataType sendType, [[maybe_unused]] const void* recvBuf,
    [[maybe_unused]] const void* recvCounts, [[maybe_unused]] const void* rdispls,
    [[maybe_unused]] HcclDataType recvType, [[maybe_unused]] rtStream_t stream, [[maybe_unused]] const std::string& tag)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AlltoAllVOutPlace(
    [[maybe_unused]] const void* sendBuf, [[maybe_unused]] const void* sendCounts, [[maybe_unused]] const void* sdispls,
    [[maybe_unused]] HcclDataType sendType, [[maybe_unused]] const void* recvBuf,
    [[maybe_unused]] const void* recvCounts, [[maybe_unused]] const void* rdispls,
    [[maybe_unused]] HcclDataType recvType, [[maybe_unused]] rtStream_t stream, [[maybe_unused]] const std::string& tag)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AlltoAllVC(
    [[maybe_unused]] const void* sendBuf, [[maybe_unused]] const void* sendCountMatrix,
    [[maybe_unused]] HcclDataType sendType, [[maybe_unused]] const void* recvBuf,
    [[maybe_unused]] HcclDataType recvType, [[maybe_unused]] rtStream_t stream, [[maybe_unused]] const std::string& tag)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AlltoAllVCOutPlace(
    [[maybe_unused]] const void* sendBuf, [[maybe_unused]] const void* sendCountMatrix,
    [[maybe_unused]] HcclDataType sendType, [[maybe_unused]] const void* recvBuf,
    [[maybe_unused]] HcclDataType recvType, [[maybe_unused]] rtStream_t stream, [[maybe_unused]] const std::string& tag)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AlltoAll(
    [[maybe_unused]] const void* sendBuf, [[maybe_unused]] u64 sendCount, [[maybe_unused]] HcclDataType sendType,
    [[maybe_unused]] const void* recvBuf, [[maybe_unused]] u64 recvCount, [[maybe_unused]] HcclDataType recvType,
    [[maybe_unused]] rtStream_t stream, [[maybe_unused]] const std::string& tag)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Broadcast(
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] void* ptr, [[maybe_unused]] u64 count,
    [[maybe_unused]] HcclDataType dataType, [[maybe_unused]] u32 root, [[maybe_unused]] HcclRtStream stream)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BroadcastOutPlace(
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] void* ptr, [[maybe_unused]] u64 count,
    [[maybe_unused]] HcclDataType dataType, [[maybe_unused]] u32 root, [[maybe_unused]] HcclRtStream stream)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Scatter(
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] void* inputPtr, [[maybe_unused]] void* outputPtr,
    [[maybe_unused]] u64 recvCount, [[maybe_unused]] HcclDataType dataType, [[maybe_unused]] u32 root,
    [[maybe_unused]] HcclRtStream stream)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ScatterOutPlace(
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] void* inputPtr, [[maybe_unused]] void* outputPtr,
    [[maybe_unused]] u64 recvCount, [[maybe_unused]] HcclDataType dataType, [[maybe_unused]] u32 root,
    [[maybe_unused]] HcclRtStream stream)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Reduce(
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] void* inputPtr, [[maybe_unused]] void* outputPtr,
    [[maybe_unused]] u64 count, [[maybe_unused]] HcclDataType dataType, [[maybe_unused]] HcclReduceOp op,
    [[maybe_unused]] u32 root, [[maybe_unused]] HcclRtStream stream)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ReduceOutPlace(
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] void* inputPtr, [[maybe_unused]] void* outputPtr,
    [[maybe_unused]] u64 count, [[maybe_unused]] HcclDataType dataType, [[maybe_unused]] HcclReduceOp op,
    [[maybe_unused]] u32 root, [[maybe_unused]] HcclRtStream stream)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ReduceScatter(
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] void* inputPtr, [[maybe_unused]] void* outputPtr,
    [[maybe_unused]] u64 count, [[maybe_unused]] HcclDataType dataType, [[maybe_unused]] HcclReduceOp op,
    [[maybe_unused]] HcclRtStream stream, [[maybe_unused]] HcomCollOpInfo* opInfo)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ReduceScatterOutPlace(
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] void* inputPtr, [[maybe_unused]] void* outputPtr,
    [[maybe_unused]] u64 count, [[maybe_unused]] HcclDataType dataType, [[maybe_unused]] HcclReduceOp op,
    [[maybe_unused]] HcclRtStream stream)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ReduceScatterV(
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] void* inputPtr, [[maybe_unused]] const void* inputCounts,
    [[maybe_unused]] const void* inputDispls, [[maybe_unused]] void* outputPtr, [[maybe_unused]] u64 outputCount,
    [[maybe_unused]] HcclDataType dataType, [[maybe_unused]] HcclReduceOp op, [[maybe_unused]] HcclRtStream stream,
    [[maybe_unused]] HcomCollOpInfo* opInfo)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ReduceScatterVOutPlace(
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] void* inputPtr, [[maybe_unused]] void* outputPtr,
    [[maybe_unused]] const void* inputCounts, [[maybe_unused]] const void* inputDispls,
    [[maybe_unused]] u64 outputCount, [[maybe_unused]] HcclDataType dataType, [[maybe_unused]] HcclReduceOp op,
    [[maybe_unused]] HcclRtStream stream)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BatchSendRecv(
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] HcclSendRecvItem* sendRecvItemsPtr,
    [[maybe_unused]] u32 itemNum, [[maybe_unused]] rtStream_t stream)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Send(
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] void* inputPtr, [[maybe_unused]] u64 count,
    [[maybe_unused]] HcclDataType dataType, [[maybe_unused]] u32 destRank, [[maybe_unused]] rtStream_t stream,
    [[maybe_unused]] u32 srTag, [[maybe_unused]] u32 localGroupRank)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SendOutPlace(
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] void* inputPtr, [[maybe_unused]] u64 count,
    [[maybe_unused]] HcclDataType dataType, [[maybe_unused]] u32 destRank, [[maybe_unused]] rtStream_t stream)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Receive(
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] void* outputPtr, [[maybe_unused]] u64 count,
    [[maybe_unused]] HcclDataType dataType, [[maybe_unused]] u32 srcRank, [[maybe_unused]] rtStream_t stream,
    [[maybe_unused]] u32 srTag, [[maybe_unused]] u32 localGroupRank)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ReceiveOutPlace(
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] void* outputPtr, [[maybe_unused]] u64 count,
    [[maybe_unused]] HcclDataType dataType, [[maybe_unused]] u32 srcRank, [[maybe_unused]] rtStream_t stream)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::RegressCalPreOp(
    [[maybe_unused]] AlltoAllOperator*& alltoAllOperator, [[maybe_unused]] const OpParam& opParam,
    [[maybe_unused]] std::unique_ptr<PreProcessMetaInfo>& preMetaInfo)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::RegressCalPreOp(
    [[maybe_unused]] AlltoAllOperator*& alltoAllOperator, [[maybe_unused]] const OpParam& opParam,
    [[maybe_unused]] std::unique_ptr<PreProcessMetaInfo>& preMetaInfo, [[maybe_unused]] Stream& preProcessStream)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ExecOp(
    [[maybe_unused]] HcclCMDType opType, [[maybe_unused]] OpParam& opParam, [[maybe_unused]] bool isCustom)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::FreeScratchMemOnOpBaseMode(
    [[maybe_unused]] DeviceMem& scratchMem, [[maybe_unused]] const OpParam& opParam,
    [[maybe_unused]] const HcclCMDType& opType)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ExecOpAlltoAll(
    [[maybe_unused]] HcclCMDType opType, [[maybe_unused]] OpParam& opParam, [[maybe_unused]] bool isCustom)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::HandleAclGraphFirstOpAivBuff([[maybe_unused]] rtStream_t mainStream)
{
    return HCCL_SUCCESS;
}

void HcclCommunicator::EraseCaptureModelId([[maybe_unused]] u64 modelId) {}

bool HcclCommunicator::StreamIsCapture([[maybe_unused]] rtStream_t mainStream) const
{
    bool isCapture = false;
    return isCapture;
}

HcclResult HcclCommunicator::CaptureSlaveStreams(
    [[maybe_unused]] rtStream_t mainStream, [[maybe_unused]] vector<Stream>& slaveStreams)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildOpLocalScratchMemResParam(
    [[maybe_unused]] const AlgResourceResponse& algResource, [[maybe_unused]] const std::string& newTag,
    [[maybe_unused]] LocalResInfoV2* localResHostPtr)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CheckSetRetryStateToWaitResume() { return HCCL_SUCCESS; }

bool HcclCommunicator::HasRoceTransportLinks(OpCommTransport& opTransportReq) const
{
    (void)opTransportReq;
    return false;
}

HcclResult HcclCommunicator::AicpuKfcClearOpResLaunch(const std::unordered_set<std::string>& tags)
{
    (void)tags;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ClearAclgraphHostLinks(const std::unordered_set<std::string>& tags)
{
    (void)tags;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildOpLocalResParam(
    [[maybe_unused]] const AlgResourceResponse& algResource, [[maybe_unused]] const std::string& newTag)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitAndCheckAicpuOrderNotify([[maybe_unused]] u8& orderLaunchMode) { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::AllocAndGetStreamContextBuff(
    [[maybe_unused]] u32 streamId, [[maybe_unused]] u64& addr, [[maybe_unused]] u64& size)
{
    return HCCL_SUCCESS;
}

u32 HcclCommunicator::UpdateOpIndex([[maybe_unused]] const OpParam& opParam) { return 0; }

HcclResult HcclCommunicator::BuildAicpuCustomParam() { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::BuildAiRmaInfoParam(
    [[maybe_unused]] const std::string& newTag, [[maybe_unused]] const std::string& algName,
    [[maybe_unused]] const HcclCMDType opType)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildAicpuOrderLaunchNotify() { return HCCL_SUCCESS; }

template <typename T>
HcclResult HcclCommunicator::CopyVectorToDeviceMem(const u64 len, DeviceMem& dstDeviceMem, const std::vector<T>& srcVec)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildOpTopoResTlvParam(
    [[maybe_unused]] const std::string& algName,
    [[maybe_unused]] const std::vector<std::vector<std::vector<u32>>>& inputVectorInfo,
    [[maybe_unused]] DeviceMem& dstTlvDeviceMem, [[maybe_unused]] u64& tlvLen)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildOpTopoResVectorTlvParam(
    [[maybe_unused]] const std::string& algName,
    [[maybe_unused]] const std::vector<std::vector<std::vector<std::vector<u32>>>>& inputVectorInfo,
    [[maybe_unused]] DeviceMem& dstTlvDeviceMem, [[maybe_unused]] u64& tlvLen)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildPairLinkCounter([[maybe_unused]] const std::string& algName) { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::BuildIsUsedRdmaRank([[maybe_unused]] const std::string& algName) { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::BuildNicList([[maybe_unused]] const std::string& algName) { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::BuildBridgeRank([[maybe_unused]] const std::string& algName) { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::BuildCommPlanRank([[maybe_unused]] const std::string& algName) { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::BuildServerAndsuperPodRank([[maybe_unused]] const std::string& algName)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildOpRetryParam(
    [[maybe_unused]] const AlgResourceResponse& algResource, [[maybe_unused]] const std::string& newTag)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildCommPlaneSubGroupRank([[maybe_unused]] const std::string& algName)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildHierarchicalAlgOption([[maybe_unused]] u32* ahcConfInfo) { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::BuildOpTopoResParam(
    [[maybe_unused]] const std::string& algName, [[maybe_unused]] const AlgResourceResponse& algResource)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildOpRemoteLinkP2pResParam(
    [[maybe_unused]] const LINK& link, [[maybe_unused]] HccltagRemoteResV3& tagRemoteRes,
    [[maybe_unused]] TransportLinkType linkType)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildOpRemoteLinkRoceResParam(
    [[maybe_unused]] const LINK& link, [[maybe_unused]] HccltagRemoteResV3& tagRemoteRes,
    [[maybe_unused]] bool isBackup, [[maybe_unused]] bool isRetry, [[maybe_unused]] bool isSecondBuild)
{
    return HCCL_SUCCESS;
}

template <typename T>
HcclResult HcclCommunicator::CreateListNode(T** resHostPtr, T** resDevicePtr)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildRemoteResByTag(
    [[maybe_unused]] const std::string& newTag, [[maybe_unused]] const u32& usrRankId,
    [[maybe_unused]] HcclRankRelationResV2*& rankRelationResHostPtr,
    [[maybe_unused]] HcclRankRelationResV2*& rankRelationResDevicePtr, [[maybe_unused]] bool isBackup,
    [[maybe_unused]] bool isRetry)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildRelationResByRemoteRankId(
    [[maybe_unused]] const TransportRequest& transportRequest, [[maybe_unused]] const LINK& link,
    [[maybe_unused]] HcclRankRelationResV2*& rankRelationResHostPtr,
    [[maybe_unused]] HcclRankRelationResV2*& rankRelationResDevicePtr)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ParseRemoteDataToMem(
    [[maybe_unused]] const OpCommTransport& opTransportResponse, [[maybe_unused]] const std::string& newTag,
    [[maybe_unused]] const HcclCMDType opType, [[maybe_unused]] bool isBackup, [[maybe_unused]] bool isRetry)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildOpRemoteResParam(
    [[maybe_unused]] const AlgResourceResponse& algResource, [[maybe_unused]] const std::string& newTag,
    [[maybe_unused]] const HcclCMDType opType, [[maybe_unused]] bool isRetry)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CopyHostListResToDeviceParam(
    [[maybe_unused]] const std::string& newTag, [[maybe_unused]] const ListCommon* headHostList,
    [[maybe_unused]] const u64 size)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CopyHostAirmaInfoToDeviceParam(
    [[maybe_unused]] const std::string& newTag, [[maybe_unused]] const HcclCMDType opType,
    [[maybe_unused]] const rtStream_t aiCpuStream)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CopyHostOpResToDeviceParam([[maybe_unused]] const std::string& newTag)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildOpResParam(
    [[maybe_unused]] const std::string& algName, [[maybe_unused]] const AlgResourceResponse& algResource,
    [[maybe_unused]] const std::string& newTag, [[maybe_unused]] const HcclCMDType opType,
    [[maybe_unused]] const rtStream_t aicpuStream)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildCustomOpResParam() { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::RegisterDfxInfo(
    [[maybe_unused]] const OpParam& param, [[maybe_unused]] AlgType algType,
    [[maybe_unused]] const std::vector<Stream>& slaveStreams, [[maybe_unused]] bool isAiv,
    [[maybe_unused]] const std::string& newTag)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetReportHcclMC2Info(
    [[maybe_unused]] const Stream& kfcStream, [[maybe_unused]] const std::vector<Stream>& aicpuStreams)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::OrchestrateAicpu(
    [[maybe_unused]] const HcclCMDType& opType, [[maybe_unused]] const std::string& algName,
    [[maybe_unused]] const OpParam& param, [[maybe_unused]] const AlgResourceResponse& algResource,
    [[maybe_unused]] const std::string& newTag, [[maybe_unused]] AlgType algType, [[maybe_unused]] bool isCustom,
    [[maybe_unused]] bool needIncreLink, [[maybe_unused]] bool needRecreateAlltoallComm)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CalcTinySendRecvMem(
    [[maybe_unused]] const OpParam& opParam, [[maybe_unused]] AlgResourceResponse& algResResponse,
    [[maybe_unused]] DeviceMem& tinySendRecvMem) const
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllocAlgNotifys(
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] const NotifyLoadType notifyLoadType,
    [[maybe_unused]] const u32 notifyNum, [[maybe_unused]] std::vector<std::shared_ptr<LocalNotify>>& notifiesMain,
    [[maybe_unused]] std::vector<std::shared_ptr<LocalNotify>>& notifiesAux)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllocAlgResource(
    [[maybe_unused]] const std::string& newTag, [[maybe_unused]] HcclCMDType opType,
    [[maybe_unused]] const OpParam& opParam, [[maybe_unused]] AlgResourceRequest& resRequest,
    AlgResourceResponse& algResResponse, [[maybe_unused]] bool selectAivAlg)
{
    SaveLinkRes(algResResponse.opTransportResponse);
    SaveLinkRes(algResResponse.opTransportResponseBackUp);

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::IncreAllocLink(
    [[maybe_unused]] const std::string& newTag, [[maybe_unused]] const OpParam& opParam,
    [[maybe_unused]] AlgResourceRequest& resRequest, AlgResourceResponse& algResResponse)
{
    SaveLinkRes(algResResponse.opTransportResponse);
    SaveLinkRes(algResResponse.opTransportResponseBackUp);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetDevicePid([[maybe_unused]] s32 devicePid) { return HCCL_SUCCESS; }

void HcclCommunicator::ReleaseWorkSpacebuffer() {}

HcclResult HcclCommunicator::AllocAndClearDeviceMem(
    [[maybe_unused]] u64 size, [[maybe_unused]] std::shared_ptr<DeviceMem>& bufferPtr) const
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllocAndClearHostMem(
    [[maybe_unused]] u64 size, [[maybe_unused]] std::shared_ptr<HostMem>& bufferPtr) const
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CreateWorkSpace([[maybe_unused]] u64 size, [[maybe_unused]] DeviceMem& buffer) const
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetWorkSpace([[maybe_unused]] u64* workSpaceSize, [[maybe_unused]] u64* workSpace) const
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitWorkSpace() { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::FillOpParam(
    [[maybe_unused]] const HcclCMDType commType, [[maybe_unused]] OpParam& opParam,
    [[maybe_unused]] const uint64_t count, [[maybe_unused]] void* pCount, [[maybe_unused]] void* pDispls) const
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllocComResource(
    [[maybe_unused]] const string& newTag, [[maybe_unused]] const string& algName,
    [[maybe_unused]] const HcclCMDType commType, [[maybe_unused]] const OpParam& opParam,
    [[maybe_unused]] rtStream_t stream, [[maybe_unused]] bool isNeedHostSlaveStream)
{
    return HCCL_SUCCESS;
}

HcclResult
HcclCommunicator::AllocComResourceByTiling([[maybe_unused]] const std::string& algConfig, [[maybe_unused]] void* param)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CreateCommResource(
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] rtStream_t aiCpuStream,
    [[maybe_unused]] bool isOpbaseMode, [[maybe_unused]] void** commContext,
    [[maybe_unused]] const std::string& algConfig)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Mc2CreateAndLaunchContext(
    [[maybe_unused]] rtStream_t aiCpuStream, [[maybe_unused]] bool isOpbaseMode, [[maybe_unused]] void** commContext,
    [[maybe_unused]] const string& tag)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetAiCpuNotifyData(
    [[maybe_unused]] const std::shared_ptr<LocalNotify>& localNotify, [[maybe_unused]] HcclSignalInfo& notifyInfo) const
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CreateAndGetAiCpuNotify(
    [[maybe_unused]] std::shared_ptr<LocalNotify>& localNotify, [[maybe_unused]] HcclSignalInfo& notifyInfo)
{
    return HCCL_SUCCESS;
}

HcclResult
HcclCommunicator::Mc2AiCpuStreamAllocAndGet([[maybe_unused]] u32 streamMode, [[maybe_unused]] rtStream_t& aiCpuStream)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Mc2AiCpuInitStreamAllocAndGet(
    [[maybe_unused]] u32 streamMode, [[maybe_unused]] rtStream_t& aiCpuStream)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AicpuResourceInit(
    [[maybe_unused]] const std::string& algName, [[maybe_unused]] const AlgResourceResponse& algResource,
    [[maybe_unused]] const std::string& newTag, [[maybe_unused]] const rtStream_t& aicpuStream,
    [[maybe_unused]] const HcclCMDType opType, [[maybe_unused]] bool isCustom)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AiCpuKernelLaunch(
    [[maybe_unused]] const rtStream_t stm, [[maybe_unused]] u64 addr, [[maybe_unused]] const std::string& kernelName)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::LoadAICPUKernel(void) { return HCCL_SUCCESS; }

void HcclCommunicator::UnloadAICPUKernel(void) { return; }

HcclResult HcclCommunicator::LoadCustomKernel(void) { return HCCL_SUCCESS; }

void HcclCommunicator::UnloadCustomKernel(void) { return; }

HcclResult HcclCommunicator::AicpuKfcTilingDataLaunch(
    [[maybe_unused]] const OpParam& opParam, [[maybe_unused]] const HcclCMDType& opType,
    [[maybe_unused]] const DeviceMem& deviceContext, [[maybe_unused]] const std::string& kernelName,
    [[maybe_unused]] const AicpuOpTiling opTilingInfo)
{
    return HCCL_SUCCESS;
}

HcclResult AicpuInitOpTilingDataAicpuCache(
    [[maybe_unused]] const OpParam& opParam, [[maybe_unused]] const HcclCMDType& opType,
    [[maybe_unused]] struct OpTilingData* opTilingData)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AicpuInitOpTilingDataBuf(
    [[maybe_unused]] const OpParam& opParam, [[maybe_unused]] const HcclCMDType& opType,
    [[maybe_unused]] const std::string& kernelName, [[maybe_unused]] const AicpuOpTiling opTilingInfo,
    [[maybe_unused]] u64 dynamicDataSize)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AicpuKfcTilingDataLaunchIn(
    [[maybe_unused]] const OpParam& opParam, [[maybe_unused]] const DeviceMem& deviceContext,
    [[maybe_unused]] const std::string& kernelName, [[maybe_unused]] const AicpuOpTiling opTilingInfo,
    [[maybe_unused]] u64 opTilingDataSize, [[maybe_unused]] bool isCustom)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AicpuKfcTilingDataLaunchExt(
    [[maybe_unused]] const OpParam& opParam, [[maybe_unused]] const HcclCMDType& opType,
    [[maybe_unused]] const DeviceMem& deviceContext, [[maybe_unused]] const std::string& kernelName,
    [[maybe_unused]] const AicpuOpTiling opTilingInfo, [[maybe_unused]] bool isCustom)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AicpuUnfoldKernelLaunch(
    [[maybe_unused]] void* inputPtr, [[maybe_unused]] void* outputPtr, [[maybe_unused]] const rtStream_t stm,
    [[maybe_unused]] u64 addr, [[maybe_unused]] void* tilingDataPtr, [[maybe_unused]] u32 tilingDataSize,
    [[maybe_unused]] const std::string& kernelName, [[maybe_unused]] HcclWorkflowMode mode,
    [[maybe_unused]] const std::string& tag)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AicpuUnfoldKernelLaunchV2(
    [[maybe_unused]] void* inputPtr, [[maybe_unused]] void* outputPtr, [[maybe_unused]] const rtStream_t stm,
    [[maybe_unused]] u64 addr, [[maybe_unused]] void* tilingDataPtr, [[maybe_unused]] u32 tilingDataSize,
    [[maybe_unused]] const std::string& kernelName, [[maybe_unused]] HcclWorkflowMode mode,
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] bool isCustom)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitCombinOpara() { return HCCL_SUCCESS; }

bool HcclCommunicator::GetCommResource([[maybe_unused]] const std::string& tag, [[maybe_unused]] void** commContext)
{
    return false;
}

bool HcclCommunicator::GetCommResource([[maybe_unused]] void*& commContext) { return false; }

HcclResult HcclCommunicator::GetAicpuOpStreamNotify(
    [[maybe_unused]] HcclRtStream* opStream, [[maybe_unused]] u8 aicpuNotifyNum, [[maybe_unused]] void** aicpuNotify)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetAicpuOpStreamAndNotify(
    [[maybe_unused]] HcclRtStream* opStream, [[maybe_unused]] u8 aicpuNotifyNum, [[maybe_unused]] void** aicpuNotify)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetAicpuNotifyInvalid() { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::ReplaceCommInfoByTag(
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] std::unique_ptr<CommInfo>& commInfo)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CreateMutiStreamResFor310P(
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] level1StreamInfo_t& streamInfo)
{
    return HCCL_SUCCESS;
}

HcclResult
HcclCommunicator::CreateCommAndStreamRes([[maybe_unused]] const std::string& tag, [[maybe_unused]] Stream& stream)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetComm([[maybe_unused]] const std::string& tag, [[maybe_unused]] CommBase** comm)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetCommResource(
    [[maybe_unused]] u64 commBufferSize, [[maybe_unused]] void* commInPtr, [[maybe_unused]] void* commOutPtr,
    [[maybe_unused]] void* commExpPtr, [[maybe_unused]] CommBase* comm, [[maybe_unused]] level1StreamInfo_t& streamInfo,
    [[maybe_unused]] Stream& stream)
{
    return HCCL_SUCCESS;
}

void HcclCommunicator::ReleaseCommContextbuffer() {}

HcclResult
HcclCommunicator::CreateDeviceCommContext([[maybe_unused]] u64 size, [[maybe_unused]] DeviceMem& buffer) const
{
    return HCCL_SUCCESS;
}

void HcclCommunicator::Break() { return; }

HcclResult HcclCommunicator::GetAlltoAllStagedWorkSpaceMemSize(
    [[maybe_unused]] u64* sendCounts, [[maybe_unused]] u64* sdispls, [[maybe_unused]] HcclDataType sendType,
    [[maybe_unused]] u64* recvCounts, [[maybe_unused]] u64* rdispls, [[maybe_unused]] HcclDataType recvType,
    [[maybe_unused]] u64& memSize)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclCommunicator::GetAlltoAllStagedWorkSpaceMemSize(
    [[maybe_unused]] std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo, [[maybe_unused]] u64& memSize)
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclCommunicator::GetAllReduceScratchSize(
    [[maybe_unused]] const u64 count, [[maybe_unused]] const HcclDataType dataType,
    [[maybe_unused]] u64& scratchSize) const
{
    return HCCL_E_NOT_SUPPORT;
}

HcclResult HcclCommunicator::SetWorldGroupInfo(
    [[maybe_unused]] std::unordered_map<std::string, std::map<u32, HcclIpAddress>> phyIdNicInfoMap,
    [[maybe_unused]] vector<RankInfo> worldRankInfoList, [[maybe_unused]] vector<u32>& nicRanksPort,
    [[maybe_unused]] vector<u32>& vnicRanksPort)
{
    return HCCL_SUCCESS;
}

HcclResult
HcclCommunicator::GetTopoDesc([[maybe_unused]] HcclTopoDescs* topoDescs, [[maybe_unused]] uint32_t topoSize) const
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetAivModeConfig([[maybe_unused]] const bool aivMode) { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::SetOnlyAivModeConfig(const bool isOnlyAiv)
{
    (void)isOnlyAiv;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetAicpuUnfoldConfig([[maybe_unused]] const bool aicpuUnfold) { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::SetExecTimeOutConfig([[maybe_unused]] const s32 execTimeOut) { return HCCL_SUCCESS; }

HcclResult
HcclCommunicator::SetAlgoConfig([[maybe_unused]] const std::map<HcclCMDType, std::vector<HcclAlgoType>>& algoMap)
{
    return HCCL_SUCCESS;
}

bool HcclCommunicator::GetAivModeConfig() { return false; }

bool HcclCommunicator::GetConfigIsOnlyAivMode() { return false; }

bool HcclCommunicator::GetAicpuUnfoldConfig() { return false; }

void HcclCommunicator::SetQpQosAttr(u32 trafficClass, u32 serviceLevel)
{
    transportManager_->SetQpQosAttr(trafficClass, serviceLevel);
    indptOpTransportManager_->SetQpQosAttr(trafficClass, serviceLevel);
}

HcclResult HcclCommunicator::CheckExitWaitResumeState([[maybe_unused]] bool& isChangedLink) { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::SetMemoryRange(
    [[maybe_unused]] void* baseVirPtr, [[maybe_unused]] size_t size, [[maybe_unused]] size_t alignment,
    [[maybe_unused]] uint64_t flags)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::UnsetMemoryRange([[maybe_unused]] void* baseVirPtr) { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::ActivateCommMemory(
    [[maybe_unused]] void* virPtr, [[maybe_unused]] size_t size, [[maybe_unused]] size_t offset,
    [[maybe_unused]] void* handle, [[maybe_unused]] uint64_t flags)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::DeactivateCommMemory([[maybe_unused]] void* virPtr) { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::SetSingleLinkInfo(
    [[maybe_unused]] std::unordered_map<u32, bool>& switchRanks, [[maybe_unused]] u32 remoteRankId,
    [[maybe_unused]] ChangeLinkInfo& changeLinkInfo)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetAttachedStream(
    [[maybe_unused]] u32 graphId, [[maybe_unused]] const std::vector<rtStream_t>& streams)
{
    return HCCL_SUCCESS;
}

u8 HcclCommunicator::GetOrderLaunchMode([[maybe_unused]] bool isCapture) { return 0; }

HcclResult HcclCommunicator::SetRemoteRankLinkInfo(
    [[maybe_unused]] std::unordered_map<u32, bool>& switchRanks, [[maybe_unused]] ChangeLinkInfo& changeLinkInfo)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ActiveStoppedLink(
    [[maybe_unused]] std::map<u32, bool>& remoteRankPortMap, [[maybe_unused]] OpCommTransport& opTransportResponse,
    [[maybe_unused]] bool isBackup)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::PrepareLinkForSwitchNic(
    [[maybe_unused]] std::unordered_map<u32, bool>& switchRanks, [[maybe_unused]] ChangeLinkInfo& changeLinkInfo)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ParseSwitchRanks(
    [[maybe_unused]] uint32_t nRanks, [[maybe_unused]] uint32_t* ranks, [[maybe_unused]] bool* useBackup,
    [[maybe_unused]] std::unordered_map<u32, bool>& switchRanks)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SwitchNic(
    [[maybe_unused]] uint32_t nRanks, [[maybe_unused]] uint32_t* ranks, [[maybe_unused]] bool* useBackup,
    [[maybe_unused]] std::shared_ptr<HDCommunicate>& controlH2D,
    [[maybe_unused]] std::shared_ptr<HDCommunicate>& statusD2H)
{
    HcclResult ret = HCCL_SUCCESS;
    return ret;
}

HcclResult HcclCommunicator::GetSwitchRanks(
    [[maybe_unused]] u32* distSwitchRankList, [[maybe_unused]] bool* distSwitchUseBackup,
    [[maybe_unused]] u32& distSwitchRankNum, [[maybe_unused]] u8* distRemoteRankNicStatus,
    [[maybe_unused]] u32& distNicStatusNum, [[maybe_unused]] bool& needCheckDefaultNic,
    [[maybe_unused]] bool& needCheckBackupNic)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::LoadCustomFile(
    [[maybe_unused]] const char* binPath, [[maybe_unused]] aclrtBinaryLoadOptionType optionType,
    [[maybe_unused]] uint32_t cpuKernelMode, [[maybe_unused]] aclrtBinHandle& binHandle)
{
    return HCCL_SUCCESS;
}

void HcclCommunicator::UnloadBinary([[maybe_unused]] aclrtBinHandle& binHandle) { return; }

HcclResult HcclCommunicator::GetCacheMap(
    [[maybe_unused]] std::unique_ptr<CollAlgOperator>& algOperator, [[maybe_unused]] OpParam& opParam,
    [[maybe_unused]] AlgType& algType, [[maybe_unused]] bool selectAivAlg, [[maybe_unused]] std::string& newTag)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ExecOpCache(
    [[maybe_unused]] HcclCMDType opType, [[maybe_unused]] OpParam& opParam, [[maybe_unused]] HcclCacheInfo& cacheInfo)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetLocalCCLBuf([[maybe_unused]] void** addr, [[maybe_unused]] uint64_t* size)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetRemoteCCLBuf(
    [[maybe_unused]] uint32_t remoteRank, [[maybe_unused]] void** addr, [[maybe_unused]] uint64_t* size)
{
    return HCCL_SUCCESS;
}
HcclResult HcclCommunicator::GetKFCWorkSpace([[maybe_unused]] void** addr, [[maybe_unused]] uint64_t* size)
{
    return HCCL_SUCCESS;
}
HcclResult IndOpTransportAlloc(
    [[maybe_unused]] const std::string& tag, [[maybe_unused]] OpCommTransport& opCommTransport,
    [[maybe_unused]] TransportIOMem& transMem, [[maybe_unused]] bool isAicpuModeEn)
{
    return HCCL_SUCCESS;
}
HcclTopoAttr HcclCommunicator::GetTopoAttr() { return {}; }
aclrtBinHandle HcclCommunicator::GetBinHandle() { return nullptr; }
HcclResult HcclCommunicator::GetHDCommunicate(
    [[maybe_unused]] HDCommunicateParams& kfcControlTransferH2DParams,
    [[maybe_unused]] HDCommunicateParams& kfcStatusTransferD2HParams)
{
    return HCCL_SUCCESS;
}
HcclResult HcclCommunicator::SetGetAicpuCommState([[maybe_unused]] std::function<bool()> getAicpuCommState)
{
    return HCCL_SUCCESS;
}

HcclResult
HcclCommunicator::CommGetNetLayers([[maybe_unused]] uint32_t** netLayers, [[maybe_unused]] uint32_t* netLayerNum)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CommGetInstSizeByNetLayer(
    [[maybe_unused]] uint32_t netLayer, [[maybe_unused]] uint32_t* rankNum) const
{
    return HCCL_SUCCESS;
}
HcclResult HcclCommunicator::CommGetInstTopoTypeByNetLayer(
    [[maybe_unused]] uint32_t netLayer, [[maybe_unused]] u32* topoType) const
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetNetLayers([[maybe_unused]] uint32_t** netLayers, [[maybe_unused]] uint32_t* netLayerNum)
{
    return HCCL_SUCCESS;
}

HcclResult
HcclCommunicator::GetInstSizeByNetLayer([[maybe_unused]] uint32_t netLayer, [[maybe_unused]] uint32_t* rankNum)
{
    return HCCL_SUCCESS;
}

HcclResult
HcclCommunicator::GetInstTopoTypeByNetLayer([[maybe_unused]] uint32_t netLayer, [[maybe_unused]] CommTopo* topoType)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetInstRanksByNetLayer(
    [[maybe_unused]] uint32_t netLayer, [[maybe_unused]] uint32_t** rankList, [[maybe_unused]] uint32_t* rankNum)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetInstSizeListByNetLayer(
    [[maybe_unused]] uint32_t netLayer, [[maybe_unused]] uint32_t** instSizeList, [[maybe_unused]] uint32_t* listSize)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetRankGraph(
    [[maybe_unused]] GraphType type, [[maybe_unused]] void** graph, [[maybe_unused]] uint32_t* len)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetLinks(
    [[maybe_unused]] uint32_t netLayer, [[maybe_unused]] uint32_t srcRank, [[maybe_unused]] uint32_t dstRank,
    [[maybe_unused]] CommLink** linkList, [[maybe_unused]] uint32_t* listSize)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetTopoInstsByLayer(
    [[maybe_unused]] uint32_t netLayer, [[maybe_unused]] uint32_t** topoInsts, [[maybe_unused]] uint32_t* topoInstNum)
{
    return HCCL_SUCCESS;
}
HcclResult HcclCommunicator::GetTopoType(
    [[maybe_unused]] uint32_t netLayer, [[maybe_unused]] uint32_t topoInstId, [[maybe_unused]] CommTopo* topoType)
{
    return HCCL_SUCCESS;
}
HcclResult HcclCommunicator::GetRanksByTopoInst(
    [[maybe_unused]] uint32_t netLayer, [[maybe_unused]] uint32_t topoInstId, [[maybe_unused]] uint32_t** ranks,
    [[maybe_unused]] uint32_t* rankNum)
{
    return HCCL_SUCCESS;
}
HcclResult HcclCommunicator::GetEndpointNum(
    [[maybe_unused]] uint32_t netLayer, [[maybe_unused]] uint32_t topoInstId, [[maybe_unused]] uint32_t* num)
{
    return HCCL_SUCCESS;
}
HcclResult HcclCommunicator::GetEndpointDesc(
    [[maybe_unused]] uint32_t netLayer, [[maybe_unused]] uint32_t topoInstId, [[maybe_unused]] uint32_t* descNum,
    [[maybe_unused]] EndpointDesc* endpointDesc)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetHeterogMode([[maybe_unused]] HcclHeterogMode* mode) { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::SnapshotCheckPreProcess() { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::SnapshotCheckPostProcess() { return HCCL_SUCCESS; }

void HcclCommunicator::SetReleaseChannel([[maybe_unused]] std::function<HcclResult()> releaseChannel) { return; }

CCLBufferManager& HcclCommunicator::GetCCLbufferManager() { return cclBufferManager_; }

void HcclCommunicator::SetHcclQos(u32 hcclQos)
{
    HCCL_INFO("[HcclCommunicator][device][SetHcclQos] hcclQos[%u]", hcclQos);
    hcclQos_ = hcclQos;
}

u32 HcclCommunicator::GetHcclQos() const
{
    HCCL_INFO("[HcclCommunicator][device][GetHcclQos] hcclQos[%u]", hcclQos_);
    return hcclQos_;
}

HcclResult HcclCommunicator::InitSymmetricMemory() { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::RegisterWindow(
    [[maybe_unused]] void* ptr, [[maybe_unused]] size_t size, [[maybe_unused]] HcclCommSymWindow* winHandle)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::DeregisterWindow([[maybe_unused]] HcclCommSymWindow winHandle) { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::GetCommSymWin(
    [[maybe_unused]] void* ptr, [[maybe_unused]] size_t size, [[maybe_unused]] HcclCommSymWindow* winHandle,
    [[maybe_unused]] size_t* offset)
{
    return HCCL_SUCCESS;
}

bool HcclCommunicator::EnableAicpuUnfold(bool isCapture)
{
    (void)isCapture;
    return false;
}

HcclResult ReAllocScratchMemForAlltoall(
    [[maybe_unused]] HcclCMDType opType, [[maybe_unused]] const OpParam& opParam,
    [[maybe_unused]] AlgResourceRequest& resRequest, [[maybe_unused]] AlgResourceResponse& algResResponse)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::HandleExistAlgResource(
    [[maybe_unused]] const std::string& newTag, [[maybe_unused]] const std::string& algName,
    [[maybe_unused]] HcclCMDType opType, [[maybe_unused]] const OpParam& opParam,
    [[maybe_unused]] std::unique_ptr<CollAlgOperator>& algOperator, [[maybe_unused]] bool selectAivAlg,
    [[maybe_unused]] bool aicpuUnfoldModeFor910B, [[maybe_unused]] bool needRecreateAlltoallComm)
{
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitMyRankConnectMode(
    [[maybe_unused]] const HcclCommParams& params, [[maybe_unused]] const RankTable_t& rankTable)
{
    return HCCL_SUCCESS;
}
uint32_t HcclCommunicator::GetConnectMode() const { return HCCL_SUCCESS; }
} // namespace hccl

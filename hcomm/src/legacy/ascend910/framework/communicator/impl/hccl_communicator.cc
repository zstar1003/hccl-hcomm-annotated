/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_communicator.h"
#include <atomic>
#include <chrono>
#include <thread>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <memory>
#include "externalinput_pub.h"
#include "opexecounter_pub.h"
// ltm指定config路径
#include "common/src/config.h"
#include "stream_active_manager.h"
#include "profiling_manager_pub.h"
#include "hccl_aiv.h"
#include "adapter_rts_common.h"
#include "coll_alg_utils.h"
#include "../common/src/state_guard.h"
#include "alg_profiling.h"
#include "preempt_port_manager.h"
#include "../nslbdp/hccl_nslbdp.h"
#include "dispatcher_ctx.h"

using namespace std;

constexpr u32 MODULE_NUM_FOUR = 4;

namespace hccl {
static std::mutex g_hcomInitMutex;
std::mutex HcclCommunicator::linkResMapMutex_;
std::unordered_map<Transport*, LinkInfo> HcclCommunicator::linkResMap_;
constexpr u32 MEMORY_CAPACITY = 256 * 1024;
constexpr u32 WAIT_PREPARE_SLEEP_TIME = 5000;
constexpr u32 SINGLE_SERVER_NUM = 1;
constexpr u32 CONN_LIMIT = 4096;
constexpr u32 COMM_DEV_TYPE_DIGIT_NUM = 8;
constexpr u32 TILINGDATA_BUF_SIZE = 32 * 1024; // 单位：字节
constexpr u32 ALLTOALL_INFO_MATRIX_SIZE = 4;
constexpr u32 AICPU_RETRY_LINKROCE_DEFAULT = 0;
constexpr u32 AICPU_RETRY_LINKROCE_BACKUP = 1;
constexpr u32 SINGLE_PROCESS_MIN_PORT = 1024;
constexpr u32 SINGLE_PROCESS_MAX_PORT = 65535;

enum TransferMemInfoIdx {
    TRANSFER_MEM_INFO_KEY_IDX = 0,
    TRANSFER_MEM_INFO_VALUE_IDX = 1,
    TRANSFER_MEM_INFO_RDMA_ENVELOPE_IDX = 2,
    TRANSFER_MEM_INFO_IDX_NUM = 3
};

unordered_map<std::string, std::string> ALGCFG_TO_NAME
    = {{"AllGather=level0:ring", "AllGatherRingFor91093Executor"},
       {"AllGather=level0:fullmesh", "AllGatherMeshOpbaseExecutor"},
       {"AllGather=level0:doublering", "AlignedAllGatherDoubleRingFor91093Executor"},
       {"ReduceScatter=level0:ring", "ReduceScatterRingFor91093Executor"},
       {"ReduceScatter=level0:fullmesh", "ReduceScatterMeshDmaEliminationExecutor"},
       {"ReduceScatter=level0:doublering", "AlignedReduceScatterDoubleRingFor91093Executor"},
       {"AllReduce=level0:ring", "AllReduceRingFor91093Executor"},
       {"AllReduce=level0:fullmesh", "AllReduceMeshOpbaseLoopExecutor"},
       {"AllReduce=level0:doublering", "AlignedAllReduceDoubleRingFor91093Executor"},
       {"AlltoAll=level0:fullmesh;level1:pairwise", "RunAlltoAllDirectFullmesh"},
       {"AlltoAll=level1:hierarchy", "RunAlltoAllAivDirect"},
       {"BatchWrite=level0:fullmesh", "BatchWriteBySdma"},
       {"BatchWrite=level1:fullmesh", "DispatchCombineFullmesh"},
       {"BatchWrite=level1:hierarchy", "DispatchCombineHierarchy"}};

struct HcclCMDTypeHash {
    size_t operator()(HcclCMDType t) const { return static_cast<size_t>(t); }
};

unordered_map<HcclCMDType, std::string, HcclCMDTypeHash> CMDTYPE_TO_KEYWORD
    = {{HcclCMDType::HCCL_CMD_ALLGATHER, "AllGather"},   {HcclCMDType::HCCL_CMD_REDUCE_SCATTER, "ReduceScatter"},
       {HcclCMDType::HCCL_CMD_ALLREDUCE, "AllReduce"},   {HcclCMDType::HCCL_CMD_ALLTOALLV, "AlltoAll"},
       {HcclCMDType::HCCL_CMD_ALLTOALLVC, "AlltoAll"},   {HcclCMDType::HCCL_CMD_ALLTOALL, "AlltoAll"},
       {HcclCMDType::HCCL_CMD_BATCH_WRITE, "BatchWrite"}};

bool HcclCommunicator::IsEnableCustom() const { return binCustomHandle_ != nullptr; }

HcclResult HcclCommunicator::InitOpResPara()
{
    CHK_SAFETY_FUNC_RET(
        memset_s(reinterpret_cast<void*>(&opResPara_), sizeof(HcclOpResParam), 0, sizeof(HcclOpResParam)));
    ListCommonInit(&opResDeviceParaPtr_->localRes.nextTagRes, &opResPara_.localRes.nextTagRes);
    opResPara_.remoteResNum = 0;
    CHK_RET(GetOpCountInfo(opResPara_.opCounterInfo));
    if (deviceType_ == DevType::DEV_TYPE_910B && GetAicpuUnfoldConfig() == false && IsOneSidedIdentifier(identifier_)) {
        // A2单边通信域在非aicpu展开场景下不初始化host与device侧的数据同步内存
        return HCCL_SUCCESS;
    }
    CHK_RET(CreateWorkSpace(sizeof(HcclOpResParam), opResDevicePara_));

    opResDeviceParaPtr_ = static_cast<HcclOpResParam*>(opResDevicePara_.ptr());

    hostDeviceLock_.reset(new (std::nothrow) PetersonLock(PetersonLock::DEFAULT_LOCK_TIMEOUT_SEC));
    CHK_SMART_PTR_NULL(hostDeviceLock_);
    CHK_RET(hostDeviceLock_->Init());
    if (aiRMAInfoMem_ == nullptr) {
        CHK_RET(AllocAndClearHostMem(sizeof(HcclAiRMAInfo), aiRMAInfoMem_));
    }
    CHK_PTR_NULL(aiRMAInfoMem_);
    CHK_PTR_NULL(aiRMAInfoMem_->ptr());

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitRankInfo(const RankTable_t& rankTable)
{
    CHK_RET(InitTcpMode(rankTable));
    SetAttrs();
    localRank_ = attrCollector_.GetLocalRank();
    deviceLogicId_ = attrCollector_.GetDeviceLogicId();
    // 按通信域配置是否使用算子级重执行
    HcclIpAddress serverIp = !rankInfoList_.empty() ? rankInfoList_[0].hostIp : HcclIpAddress();
    HcclIpAddress localIp = rankInfoList_.size() > userRank_ ? rankInfoList_[userRank_].hostIp : HcclIpAddress();
    bool isAivMode = GetAivModeConfig() || GetConfigIsOnlyAivMode();
    SetRetryEnable(
        deviceType_, superPodNum_, serverNum_, deviceNumPerAggregation_, isDiffDeviceType_, isAivMode, serverIp,
        localIp, retryEnable_, commConfig_.GetConfigInterServerRetryEnable(),
        commConfig_.GetConfigInterSuperPodRetryEnable());
    // 校验A+X单机双module场景下通信能否建立
    CHK_RET(CheckSingleServerComm(rankTable.rankList));
    // 解析rank和port的映射信息
    CHK_RET(SetRanksPort(rankTable.rankList));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetRanksPort(const std::vector<RankInfo_t>& rankList)
{
    bool devicePortSwitchOn = commPortConfig_.devPortSwitchOn;
    if (devicePortSwitchOn) {
        nicRanksPort_.resize(userRankSize_, HCCL_INVALID_PORT);
        vnicRanksPort_.resize(userRankSize_, HCCL_INVALID_PORT);
        for (auto& rankInfo : rankList) {
            nicRanksPort_[rankInfo.rankId]
                = rankInfo.deviceInfo.port == HCCL_INVALID_PORT ? HETEROG_CCL_PORT : rankInfo.deviceInfo.port;
            vnicRanksPort_[rankInfo.rankId]
                = rankInfo.deviceInfo.vnicPort == HCCL_INVALID_PORT ? HETEROG_CCL_PORT : rankInfo.deviceInfo.vnicPort;
        }
    } else {
        nicRanksPort_.resize(userRankSize_, HCCL_INVALID_PORT);
        for (auto& rankInfo : rankList) {
            nicRanksPort_[rankInfo.rankId] = rankInfo.deviceInfo.port == HCCL_INVALID_PORT
                                                     || rankInfo.deviceInfo.port < SINGLE_PROCESS_MIN_PORT
                                                     || rankInfo.deviceInfo.port > SINGLE_PROCESS_MAX_PORT ?
                                                 HETEROG_CCL_PORT :
                                                 rankInfo.deviceInfo.port;
        }
    }
    isUseRankPort_
        = ((devicePortSwitchOn && nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_DEVICE) || isHaveCpuRank_) ?
              true :
              isUseRankPort_;
    HCCL_INFO(
        "[HcclCommunicator][SetRanksPort] devicePortSwitchOn[%u], isHaveCpuRank[%u], isUseRankPort[%u], "
        "nicRanksPort size[%u], vnicRanksPort size[%u].",
        devicePortSwitchOn, isHaveCpuRank_, isUseRankPort_, nicRanksPort_.size(), vnicRanksPort_.size());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitNetResource(const RankTable_t& rankTable)
{
    CHK_RET(InitPreResource(rankTable));
    CHK_RET(InitRaResource());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitDebug()
{
    CHK_RET(InitProfiling());
    CHK_RET(InitATraceInfo());
    return HCCL_SUCCESS;
}

std::string HcclCommunicator::GetSupportDataType(bool needReduce)
{
    std::vector<HcclDataType> supportList
        = {HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_INT16, HCCL_DATA_TYPE_INT32, HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_FP32};
    if (needReduce) {
        if (!Is310P3Common(isHaveCpuRank_, deviceType_)) {
            supportList.insert(supportList.end(), {HCCL_DATA_TYPE_BFP16, HCCL_DATA_TYPE_INT64});
        }
    } else {
        supportList.insert(
            supportList.end(), {HCCL_DATA_TYPE_INT64, HCCL_DATA_TYPE_UINT8, HCCL_DATA_TYPE_UINT16,
                                HCCL_DATA_TYPE_UINT32, HCCL_DATA_TYPE_UINT64, HCCL_DATA_TYPE_FP64});
        if (!Is310P3Common(isHaveCpuRank_, deviceType_)) {
            supportList.push_back(HCCL_DATA_TYPE_BFP16);
        }
    }

    std::string supportInfo = "";
    for (u32 i = 0; i < supportList.size(); i++) {
        if (i != 0) {
            supportInfo += ", ";
        }
        supportInfo += GetDataTypeEnumStr(supportList[i]);
    }

    return supportInfo;
}

HcclResult HcclCommunicator::InitATraceInfo()
{
    /* 申请trace资源信息 */
    std::string logInfo = "HCCL_";
    logInfo.append(to_string(SalGetTid()));
    logInfo.append("_");
    logInfo.append(to_string(deviceLogicId_));
    opBaseAtraceInfo_.reset(new (std::nothrow) HcclTraceInfo());
    CHK_PTR_NULL(opBaseAtraceInfo_);
    CHK_RET(opBaseAtraceInfo_->Init(logInfo));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitDebugSubGroup()
{
    CHK_RET(InitATraceInfo());
    CHK_RET(InitProfiler());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitNotifyManager()
{
    queueNotifyManager_.reset(new (std::nothrow) QueueNotifyManager());
    CHK_SMART_PTR_NULL(queueNotifyManager_);
    CHK_RET(queueNotifyManager_->Init());
    queueNotifyManagerRefac_.reset(new (std::nothrow) QueueNotifyManager());
    CHK_SMART_PTR_NULL(queueNotifyManagerRefac_);
    CHK_RET(queueNotifyManagerRefac_->Init());

    return HCCL_SUCCESS;
}

void TaskProfilerCallBack(void* userPtr, void* param, u32 length)
{
    static_cast<ProfilerManager*>(userPtr)->TaskProfilerHandle(param, length);
}

void TaskAivProfilerCallBack(void* userPtr, void* param, u32 length)
{
    static_cast<ProfilerManager*>(userPtr)->TaskAivProfilerHandle(param, length);
}

HcclResult HcclCommunicator::InitDispatcher()
{
    // 根据设备ID创建dispatcher
    if ((deviceType_ == DevType::DEV_TYPE_910B) && GetExternalInputHcclEnableFfts()) {
        CHK_PRT_CONT(
            GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !GetAicpuUnfoldConfig(),
            HCCL_RUN_INFO("Will use ffts mode."));
    } else {
        // 不满足ffts+特性开启条件。
        SetFftsSwitch(false);
    }
    CHK_RET(HcclDispatcherInit(DispatcherType::DISPATCHER_NORMAL, devicePhyId_, &dispatcher_));
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_RET(HcclSetExecTimeOut(dispatcher_, commConfig_.GetConfigExecTimeOut()));

    if (!FindDispatcherByCommId(&dispatcherCtx_, identifier_.c_str())) {
        CHK_RET(CreateDispatcherCtx(&dispatcherCtx_, devicePhyId_, identifier_.c_str()));
    }
    CHK_PTR_NULL(dispatcherCtx_);

    hccl::DispatcherCtx* Ctx_tmp = static_cast<DispatcherCtx*>(dispatcherCtx_);
    HCCL_INFO(
        "[%s] RegisterLoadTaskCallBack Dispatcher = [%p], Ctx_tmp = [%p]", __func__, Ctx_tmp->GetDispatcher(),
        static_cast<void*>(Ctx_tmp));
    (void)RegisterLoadTaskCallBack(
        Ctx_tmp->GetDispatcher(), static_cast<void*>(profilerManager_.get()), TaskProfilerCallBack);

    CHK_RET(HcclDispatcherInit(DispatcherType::DISPATCHER_VIRTURAL, devicePhyId_, &vDispatcher_));
    CHK_SMART_PTR_NULL(vDispatcher_);
    CHK_RET(HcclSetExecTimeOut(vDispatcher_, commConfig_.GetConfigExecTimeOut()));

    (void)RegisterLoadTaskCallBack(dispatcher_, static_cast<void*>(profilerManager_.get()), TaskProfilerCallBack);
    // 此时要确保identify已经全部构造完成
    AlgWrap::GetInstance().RegisterAlgCallBack(
        identifier_, static_cast<void*>(profilerManager_.get()), TaskAivProfilerCallBack, deviceLogicId_);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitStreamManager()
{
    opStreamManager_.reset(static_cast<OpBaseStreamManager*>(new (std::nothrow) OpBaseStreamManager));
    CHK_SMART_PTR_NULL(opStreamManager_);
    CHK_RET(StreamActiveManager::GetInstance(deviceLogicId_).Init());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitSocketManager()
{
    socketManager_.reset(new (std::nothrow) HcclSocketManager(nicDeployment_, deviceLogicId_, devicePhyId_, userRank_));
    CHK_PTR_NULL(socketManager_);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitTransportManager()
{
    std::vector<u32>& nicRanksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    std::vector<u32>& vnicRanksPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
    transportManager_.reset(static_cast<TransportManager*>(new (std::nothrow) TransportManager(
        cclBufferManager_, socketManager_, dispatcher_, notifyPool_, rankInfoList_, userRank_, identifier_,
        deviceLogicId_, nicDeployment_, isHaveCpuRank_, isUseRankPort_, isUsedRdmaLevel0_, nicRanksPorts,
        vnicRanksPorts, useSuperPodMode_, devIpAddr_, hostIp_, localVnicIp_, netDevCtxMap_)));
    CHK_SMART_PTR_NULL(transportManager_);
    (void)transportManager_->SetPortConfig(commPortConfig_.devPortSwitchOn);
    (void)transportManager_->SetIsStandardCard(isStandardCard_);

    DispatcherCtx* ctx = static_cast<DispatcherCtx*>(dispatcherCtx_);
    CHK_PTR_NULL(ctx);
    indptOpTransportManager_.reset(static_cast<TransportManager*>(new (std::nothrow) TransportManager(
        cclBufferManager_, socketManager_, ctx->GetDispatcher(), notifyPool_, rankInfoList_, userRank_, identifier_,
        deviceLogicId_, nicDeployment_, isHaveCpuRank_, isUseRankPort_, isUsedRdmaLevel0_, nicRanksPorts,
        vnicRanksPorts, useSuperPodMode_, devIpAddr_, hostIp_, localVnicIp_, netDevCtxMap_)));
    CHK_SMART_PTR_NULL(indptOpTransportManager_);
    (void)indptOpTransportManager_->SetPortConfig(commPortConfig_.devPortSwitchOn);
    (void)indptOpTransportManager_->SetIsStandardCard(isStandardCard_);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitHcclAlg()
{
    CHK_RET(OpExeCounter::GetInstance(deviceLogicId_).InitCounter());

    notifyPool_.reset(new (std::nothrow) NotifyPool());
    CHK_SMART_PTR_NULL(notifyPool_);
    CHK_RET(notifyPool_->Init(devicePhyId_));

    callbackTask_.reset(new (std::nothrow) HcclCallbackTask(devicePhyId_, deviceLogicId_, dispatcher_, nicDeployment_));
    CHK_SMART_PTR_NULL(callbackTask_);

    workSpaceRes_.reset(new (std::nothrow) WorkspaceResource(devicePhyId_, deviceLogicId_));
    CHK_SMART_PTR_NULL(workSpaceRes_);

    CHK_RET(InitAlgResource());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitAlgResource()
{
    HcclTopoAttr topoAttr{};
    attrCollector_.GetTopoAttr(topoAttr);

    HcclAlgoAttr algoAttr{};
    attrCollector_.GetAlgoAttr(algoAttr);

    implAlg_.reset(new (std::nothrow) HcclAlg(cclBufferManager_, dispatcher_, vDispatcher_));
    CHK_SMART_PTR_NULL(implAlg_);
    CHK_RET(implAlg_->Init(workSpaceRes_, notifyPool_, netDevCtxMap_, queueNotifyManager_, algoAttr, topoAttr, false));
    return HCCL_SUCCESS;
}

void HcclCommunicator::SetAttrs()
{
    serverId_ = attrCollector_.GetServerId();
    superPodId_ = attrCollector_.GetSuperPodId();
    superDeviceId_ = attrCollector_.GetSuperDeviceId();
    // GetServerNum
    serverNum_ = attrCollector_.GetServerNum();
    // IsSuperPodMode
    useSuperPodMode_ = attrCollector_.GetSuperPodMode();
    // GetSuperPodNum
    superPodNum_ = attrCollector_.GetSuperPodNums();
    // GetInnerServerAverageDevice
    deviceNumPerAggregation_ = attrCollector_.GetDeviceNumPerAggregation();
    deviceNumPerServer_ = attrCollector_.GetDeviceNumPerServer();
    isHaveCpuRank_ = attrCollector_.GetHaveCpuRank();
    // TransformRankInfoByServerId
    servRankInfo_ = attrCollector_.GetServRankInfo();
    // GetModuleInfo
    isDiffDeviceModule_ = attrCollector_.GetDiffDeviceModule();
    isDiffDeviceType_ = attrCollector_.GetDiffDeviceType();
    gcdDeviceNumPerAggregation_ = attrCollector_.GetGcdDeviceNumPerAggregation();
    moduleNum_ = attrCollector_.GetModuleNum();
    multiModuleDiffDeviceNumMode_ = attrCollector_.GetMultiModuleDiffDeviceNumMode();
    multiSuperPodDiffServerNumMode_ = attrCollector_.GetMultiSuperPodDiffServerNumMode();
    multiSuperPodDiffDeviceNumMode_ = attrCollector_.GetmultiSuperPodDiffDeviceNumMode();
    isARSDoubleRing_ = attrCollector_.GetSupportARS();
    // 生成nicList
    nicList_ = attrCollector_.GetNicList();
    // InitTopoInfo
    isSingleMeshAggregation_ = attrCollector_.GetSingleMeshAggregation();
    isAllRankSamePlane_ = attrCollector_.GetAllRankSamePlane();
    isStandardCard_ = attrCollector_.GetStandardCard();
    is310PDuoCard_ = attrCollector_.Get310PDuoCard();
    isCommon310P3DUO_ = attrCollector_.GetIsCommon310P3DUO();
    hccsPortNum_ = attrCollector_.GetHccsPortNum();
    attrCollector_.GetPairLinkCounter(pairLinkCounter_);
    attrCollector_.GetPairLinkInfo(pairLinkInfo_);
    // SetInterModeInSuperPod
    isUsedInterHccsMode_ = attrCollector_.GetUsedInterHccsMode();
    // GetRankInfoList
    rankInfoList_ = attrCollector_.GetRankInfoList();
    // Localinfo
    devIpAddr_ = attrCollector_.GetDevIpAddr();
    devBackupIpAddr_ = attrCollector_.GetDevBackupIpAddr();
    devBackupPort_ = attrCollector_.GetBackupDevPort();
    devBackupPort_ = devBackupPort_ == HCCL_INVALID_PORT ? AICPU_RETRY_BACKUP_PORT : devBackupPort_;
    devicePhyId_ = attrCollector_.GetDevicePhyId();
    hostIp_ = attrCollector_.GetHostIp();
    hostPort_ = attrCollector_.GetHostPort();

    interServer_ = attrCollector_.GetInterServe();
    nicDeployment_ = attrCollector_.GetNicDeployment();
}

void HcclCommunicator::ForceProf(bool isForce) { ForceProfOn(dispatcher_, isForce); }

HcclResult HcclCommunicator::InitRankInfoSubGroup(WorldGroupInfo& groupCommonData)
{
    SetAttrs();
    // inline reduce 开关
    inlineReduceSwitchOn_ = attrCollector_.GetInlineReduceSwitchOn();
    // CalAndSetMeshAggRankSize
    meshAggregationRankSize_ = attrCollector_.GetMeshAggregationRankSize();
    // IsUsedRdmaLevel0AndIpInvalid
    isUsedRdmaLevel0_ = attrCollector_.GetUsedRdmaLevel0();

    CHK_RET(SetWorldGroupInfo(
        groupCommonData.phyIdNicInfoMap, groupCommonData.worldRankInfoList, groupCommonData.ranksPort,
        groupCommonData.vnicRanksPort));
    for (auto& rankInfo : worldRankInfoList_) {
        if (rankInfo.devicePhyId == HOST_DEVICE_ID) {
            isUseRankPort_ = true;
            break;
        }
    }
    CHK_RET(IsHostUseDevNic(isHostUseDevNic_));
    // 按通信域配置是否使用算子级重执行
    HcclIpAddress serverIp = !rankInfoList_.empty() ? rankInfoList_[0].hostIp : HcclIpAddress();
    HcclIpAddress localIp = rankInfoList_.size() > userRank_ ? rankInfoList_[userRank_].hostIp : HcclIpAddress();
    bool isAivMode = GetAivModeConfig() || GetConfigIsOnlyAivMode();
    SetRetryEnable(
        deviceType_, superPodNum_, serverNum_, deviceNumPerAggregation_, isDiffDeviceType_, isAivMode, serverIp,
        localIp, retryEnable_, commConfig_.GetConfigInterServerRetryEnable(),
        commConfig_.GetConfigInterSuperPodRetryEnable());
    groupNicRanksPort_.resize(rankInfoList_.size(), HCCL_INVALID_PORT);
    if (nicRanksPort_.size() != 0) {
        for (auto& rankInfo : rankInfoList_) {
            groupNicRanksPort_[rankInfo.userRank] = nicRanksPort_[rankInfo.worldRank];
            HCCL_INFO(
                "hostIp[%s], nicIp[%s], rankInfo.userRank[%u], rankInfo.worldRank[%u], "
                "nic port[%u], devicePhyId[%d]",
                rankInfo.hostIp.GetReadableAddress(), rankInfo.nicIp[0].GetReadableAddress(), rankInfo.userRank,
                rankInfo.worldRank, groupNicRanksPort_[rankInfo.userRank], rankInfo.devicePhyId);
        }
    }
    commPortConfig_.devPortSwitchOn = groupCommonData.devPortSwitchOn;
    if (commPortConfig_.devPortSwitchOn) {
        groupVnicRanksPort_.resize(rankInfoList_.size(), HCCL_INVALID_PORT);
        if (vnicRanksPort_.size() != 0) {
            for (auto& rankInfo : rankInfoList_) {
                groupVnicRanksPort_[rankInfo.userRank] = vnicRanksPort_[rankInfo.worldRank];
                HCCL_INFO(
                    "hostIp[%s], nicIp[%s], rankInfo.userRank[%u], rankInfo.worldRank[%u], "
                    "vnic port[%u], devicePhyId[%d]",
                    rankInfo.hostIp.GetReadableAddress(), rankInfo.nicIp[0].GetReadableAddress(), rankInfo.userRank,
                    rankInfo.worldRank, groupVnicRanksPort_[rankInfo.userRank], rankInfo.devicePhyId);
            }
        }
    }
    isUseRankPort_ = ((commPortConfig_.devPortSwitchOn && nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_DEVICE)
                      || isHaveCpuRank_) ?
                         true :
                         isUseRankPort_;
    for (auto& rank : rankInfoList_) {
        if (hostIp_ != rank.hostIp) {
            isServerInter_ = true;
            HCCL_DEBUG(" isServerInter_ is true");
            break;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetClearAivSyncBuf(bool aivClearEnable)
{
    aivClearEnable_ = aivClearEnable;
    return HCCL_SUCCESS;
}

u32 HcclCommunicator::GetRankTableCrc() const { return ranktableCrc_; }

u32 HcclCommunicator::GetServerNum() const { return serverNum_; }

u32 HcclCommunicator::GetRealUserRank() const { return realUserRank_; }

u32 HcclCommunicator::GetModuleNum() const { return moduleNum_; }

bool HcclCommunicator::GetSupportHDCommunicate()
{
    HCCL_INFO(
        "%s aicpuUnfold[%d], deviceType_[%d], isHaveCpuRank_[%d]", __func__, GetAicpuUnfoldConfig(), deviceType_,
        isHaveCpuRank_);
    if (deviceType_ == DevType::DEV_TYPE_910B && GetAicpuUnfoldConfig() == false && IsOneSidedIdentifier(identifier_)) {
        // A2单边通信域在非aicpu展开场景下不初始化HDC资源
        return false;
    }
    return (GetAicpuUnfoldConfig() == true)
           || ((deviceType_ == DevType::DEV_TYPE_910_93) || (deviceType_ == DevType::DEV_TYPE_910B)
               || Is310P3Common(isHaveCpuRank_, deviceType_));
}

HcclResult HcclCommunicator::InitHDCommunicate()
{
    if (GetSupportHDCommunicate()) {
        // 初始化aicpu进程host-device共享内存
        EXCEPTION_CATCH(
            (kfcControlTransferH2D_
             = std::make_shared<hccl::HDCommunicate>(deviceLogicId_, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl))),
            return HCCL_E_PTR);
        CHK_RET(kfcControlTransferH2D_->InitHost());

        EXCEPTION_CATCH(
            (kfcStatusTransferD2H_
             = std::make_shared<hccl::HDCommunicate>(deviceLogicId_, HCCL_HDC_TYPE_D2H, sizeof(KfcExecStatus))),
            return HCCL_E_PTR);
        CHK_RET(kfcStatusTransferD2H_->InitHost());

        if (IsEnableCustom()) {
            // 初始化custom进程host-device共享内存
            EXCEPTION_CATCH(
                (customControlTransferH2D_
                 = std::make_shared<hccl::HDCommunicate>(deviceLogicId_, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl))),
                return HCCL_E_PTR);
            CHK_RET(customControlTransferH2D_->InitHost());

            EXCEPTION_CATCH(
                (customStatusTransferD2H_
                 = std::make_shared<hccl::HDCommunicate>(deviceLogicId_, HCCL_HDC_TYPE_D2H, sizeof(KfcExecStatus))),
                return HCCL_E_PTR);
            CHK_RET(customStatusTransferD2H_->InitHost());
        }
    }
    return HCCL_SUCCESS;
}

bool HcclCommunicator::IsEnableRoce() { return attrCollector_.IsEnableRoce(); }

u32 HcclCommunicator::LargestPowerOfTwoLessThan(const u32 localRankSize)
{
    return (1 << static_cast<int>(std::floor(SalLog2(localRankSize))));
}

u32 HcclCommunicator::CalcStreamNumForReduceOrderPreservation()
{
    // Level0RankSize条流给alltoall，剩下的流给LocalReduce使用
    u32 level0StreamNum = deviceNumPerAggregation_ - 1 + LargestPowerOfTwoLessThan(deviceNumPerAggregation_);
    // level1主流分给alltoall，从流给LocalReduce使用
    u32 level1StreamNum = LargestPowerOfTwoLessThan(moduleNum_);
    // 总流数上限：7（alltoall使用，提前的本地拷贝任务不需要并行）+ 4（LocalReduce使用）
    u32 streamNum
        = std::min(std::max(level0StreamNum - 1, level1StreamNum), DEVICE_EIGHT + DEVICE_EIGHT / FACTOR_NUM_TWO - 1);

    HCCL_INFO(
        "[%s]level0StreamNum[%u], level1StreamNum[%u], streamNum[%u]", __func__, level0StreamNum, level1StreamNum,
        streamNum);
    return streamNum;
}

void HcclCommunicator::DestroyOpTransportResponse(OpCommTransport& opTransportResponse, bool aclGraphDestroyCbk)
{
    std::unique_lock<std::mutex> commLock(linkResMapMutex_);
    for (auto& levelNSubCommTransport : opTransportResponse) {
        for (auto& singleSubCommTransport : levelNSubCommTransport) {
            for (u32 i = 0; i < singleSubCommTransport.virtualLinks.size(); i++) {
                if (singleSubCommTransport.virtualLinks[i] != nullptr
                    && (!aclGraphDestroyCbk || singleSubCommTransport.transportRequests[i].isUsedRdma)) {
                    linkResMap_.erase(singleSubCommTransport.virtualLinks[i].get());
                }
            }
            for (u32 i = 0; i < singleSubCommTransport.links.size(); i++) {
                if (singleSubCommTransport.transportRequests[i].isValid && singleSubCommTransport.links[i] != nullptr
                    && (!aclGraphDestroyCbk || singleSubCommTransport.transportRequests[i].isUsedRdma)) {
                    linkResMap_.erase(singleSubCommTransport.links[i].get());
                }
            }
        }
    }
    commLock.unlock();
    for (auto& levelNSubCommTransport : opTransportResponse) {
        for (auto& singleSubCommTransport : levelNSubCommTransport) {
            for (u32 i = 0; i < singleSubCommTransport.virtualLinks.size(); i++) {
                if (singleSubCommTransport.virtualLinks[i] != nullptr
                    && (!aclGraphDestroyCbk || singleSubCommTransport.transportRequests[i].isUsedRdma)) {
                    singleSubCommTransport.virtualLinks[i]->DeInit();
                    singleSubCommTransport.virtualLinks[i] = nullptr;
                }
            }
            for (u32 i = 0; i < singleSubCommTransport.links.size(); i++) {
                if (singleSubCommTransport.transportRequests[i].isValid && singleSubCommTransport.links[i] != nullptr
                    && (!aclGraphDestroyCbk || singleSubCommTransport.transportRequests[i].isUsedRdma)) {
                    singleSubCommTransport.links[i]->DeInit();
                    singleSubCommTransport.links[i] = nullptr;
                }
            }
            if (!aclGraphDestroyCbk) {
                singleSubCommTransport.virtualLinks.clear();
                singleSubCommTransport.links.clear();
            }
        }
    }
}

void HcclCommunicator::DestroyAlgResource(AlgResourceResponse& res, bool aclGraphDestroyCbk)
{
    DestroyOpTransportResponse(res.opTransportResponse, aclGraphDestroyCbk);
    if (IsEnableBackupLink()) {
        DestroyOpTransportResponse(res.opTransportResponseBackUp, aclGraphDestroyCbk);
        HCCL_INFO("[%s]finish DestroyOpTransportResponse", __func__);
    }
}

HcclResult HcclCommunicator::ReleasePreemptSocket()
{
    if (commPortConfig_.devNicListen.first) {
        CHK_RET(PreemptPortManager::GetInstance(deviceLogicId_).Release(commPortConfig_.devNicListen.first));
        commPortConfig_.devNicListen.first.reset();
        if (commPortConfig_.devNicListen.second) {
            HcclNetCloseDev(commPortConfig_.devNicListen.second);
            commPortConfig_.devNicListen.second = nullptr;
        }
        CHK_RET(HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, devicePhyId_, deviceLogicId_));
        HCCL_INFO(
            "[HcclCommunicator][ReleasePreemptSocket] release preempt socket of device nic success, "
            "comm id[%s].",
            identifier_.c_str());
    }

    if (commPortConfig_.devVnicListen.first) {
        CHK_RET(PreemptPortManager::GetInstance(deviceLogicId_).Release(commPortConfig_.devVnicListen.first));
        commPortConfig_.devVnicListen.first.reset();
        if (commPortConfig_.devVnicListen.second) {
            HcclNetCloseDev(commPortConfig_.devVnicListen.second);
            commPortConfig_.devVnicListen.second = nullptr;
        }
        CHK_RET(HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, devicePhyId_, deviceLogicId_));
        HCCL_INFO(
            "[HcclCommunicator][ReleasePreemptSocket] release preempt socket of device vnic success, "
            "comm id[%s].",
            identifier_.c_str());
    }

    if (commPortConfig_.backupDevNicListen.first) {
        CHK_RET(hrtGetPairDevicePhyId(devicePhyId_, deviceBackUpPhyId_));
        if (hrtGetDeviceIndexByPhyId(deviceBackUpPhyId_, deviceBackUpLogicId_) == HCCL_SUCCESS) {
            CHK_RET(PreemptPortManager::GetInstance(deviceBackUpLogicId_)
                        .Release(commPortConfig_.backupDevNicListen.first));
            commPortConfig_.backupDevNicListen.first.reset();
            if (commPortConfig_.backupDevNicListen.second) {
                HcclNetCloseDev(commPortConfig_.backupDevNicListen.second);
                commPortConfig_.backupDevNicListen.second = nullptr;
            }
            CHK_RET(
                HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, deviceBackUpPhyId_, deviceBackUpLogicId_, true));
            HCCL_INFO(
                "[HcclCommunicator][ReleasePreemptSocket] release preempt socket of backup nic success, "
                "comm id[%s].",
                identifier_.c_str());
        }
    }

    HCCL_INFO(
        "[HcclCommunicator][ReleasePreemptSocket] release all preempt socket success, comm id[%s].",
        identifier_.c_str());

    return HCCL_SUCCESS;
}

ErrorMessageReport HcclCommunicator::GetAicpuTaskException()
{
    HcclResult ret = HCCL_SUCCESS;
    ErrorMessageReport errorMessage;
    if (kfcStatusTransferD2H_ != nullptr) {
        CHK_PRT_RET(
            isInvalidComm_,
            HCCL_ERROR(
                "[HcclCommunicator][%s] comm[%s], rank[%u], devId[%d], snapshot recovering, "
                "this comm is invalid.",
                __func__, identifier_.c_str(), userRank_, deviceLogicId_),
            errorMessage);
        ret = kfcStatusTransferD2H_->Get(
            sizeof(HcclOpIdentifier) + sizeof(ExecStatusDef), sizeof(errorMessage),
            reinterpret_cast<uint8_t*>(&errorMessage));
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("GetAicpuTaskException get aicpu task exception failed.ret[%u]", ret);
        }
    }
    return errorMessage;
}

HcclResult HcclCommunicator::UnRegisterBackGroundThread()
{
    CHK_PRT_RET(
        isInvalidComm_,
        HCCL_ERROR(
            "[HcclCommunicator][%s] comm[%s], rank[%u], devId[%d], snapshot recovering, "
            "this comm is invalid.",
            __func__, identifier_.c_str(), userRank_, deviceLogicId_),
        HCCL_E_UNAVAIL);
    CHK_RET(UnRegisterBackGroundThread(kfcControlTransferH2D_, kfcStatusTransferD2H_));
    if (IsEnableCustom()) {
        CHK_RET(UnRegisterBackGroundThread(customControlTransferH2D_, customStatusTransferD2H_));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::UnRegisterBackGroundThread(
    std::shared_ptr<HDCommunicate>& controlH2D, std::shared_ptr<HDCommunicate>& statusD2H)
{
    HCCL_INFO("start to stop the backGround Thread");
    if (deviceType_ == DevType::DEV_TYPE_910 || (deviceType_ == DevType::DEV_TYPE_910B && !GetAicpuUnfoldFlag())) {
        if (GetMC2EnvFlag()) {
            if (controlH2D != nullptr) {
                BackgroundCommand request = BackgroundCommand::kStop;
                CHK_RET(controlH2D->Put(
                    sizeof(KfcCommand), sizeof(BackgroundCommand),
                    reinterpret_cast<uint8_t*>(&request))); // 下的停止命令仅仅只修改BackGroundCommand
                auto waitStopExecCmdTimeoutMs = HcclGetCmdTimeout();
                auto waitStopExecCmdTimeout = std::chrono::milliseconds(waitStopExecCmdTimeoutMs);
                auto startTime = std::chrono::steady_clock::now();
                while (true) {
                    if ((std::chrono::steady_clock::now() - startTime) >= waitStopExecCmdTimeout) {
                        HCCL_ERROR("[NsRecovery]~HcclCommunicator is timeout [%u ms]", waitStopExecCmdTimeoutMs);
                        return HCCL_E_INTERNAL;
                    }
                    KfcExecStatus status;
                    CHK_RET(statusD2H->Get(0, sizeof(KfcExecStatus), reinterpret_cast<uint8_t*>(&status)));
                    if (status.execStatus.backgroundStatus == BackgroundStatus::kStop) {
                        break;
                    }
                }
            }
        }
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::DestroyAicpuComm()
{
    CHK_PRT_RET(
        isInvalidComm_,
        HCCL_ERROR(
            "[HcclCommunicator][%s] comm[%s], rank[%u], devId[%d], snapshot recovering, "
            "this comm is invalid.",
            __func__, identifier_.c_str(), userRank_, deviceLogicId_),
        HCCL_E_UNAVAIL);
    CHK_RET(DestroyAicpuComm(kfcControlTransferH2D_, kfcStatusTransferD2H_));
    if (IsEnableCustom()) {
        CHK_RET(DestroyAicpuComm(customControlTransferH2D_, customStatusTransferD2H_));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::DestroyAicpuComm(
    std::shared_ptr<HDCommunicate>& controlH2D, std::shared_ptr<HDCommunicate>& statusD2H)
{
    HCCL_INFO("[HcclCommunicator][%s]start to destroy the aicpu comm, group[%s].", __func__, identifier_.c_str());
    if (deviceType_ != DevType::DEV_TYPE_910_93 && !(deviceType_ == DevType::DEV_TYPE_910B && GetAicpuUnfoldFlag())
        && (myRankConnectMode_ == 0)) {
        HCCL_INFO("[HcclCommunicator][%s]Device type[%d] no needs to destroy the aicpu comm.", __func__, deviceType_);
        return HCCL_SUCCESS;
    }
    if (controlH2D == nullptr) {
        HCCL_WARNING("[HcclCommunicator][%s]controlH2D is nullptr, can not destroy the aicpu comm.", __func__);
        return HCCL_SUCCESS;
    }
    if (!GetMC2EnvFlag() && (getAicpuCommState_ == nullptr || !getAicpuCommState_())) {
        HCCL_INFO(
            "[HcclCommunicator][%s]Not mc2 or aicpu environment, "
            "no needs to destroy the aicpu comm.",
            __func__);
        return HCCL_SUCCESS;
    }
    KfcCommand destroyCmd = KfcCommand::kDestroyComm;
    CHK_RET(controlH2D->Put(0, sizeof(KfcCommand), reinterpret_cast<uint8_t*>(&destroyCmd)));
    KfcExecStatus status;
    auto waitCmdTimeoutMs = HcclGetCmdTimeout();
    auto waitCmdTimeout = std::chrono::milliseconds(waitCmdTimeoutMs);
    auto startTime = std::chrono::steady_clock::now();
    while (true) {
        CHK_RET(statusD2H->Get(0, sizeof(KfcExecStatus), reinterpret_cast<uint8_t*>(&status)));
        if (status.execStatus.kfcStatus == KfcStatus::kDestroyComm) {
            HCCL_RUN_INFO("[HcclCommunicator][%s]ExecStatus[%d]", __func__, status.execStatus.kfcStatus);
            return HCCL_SUCCESS;
        } else {
            if ((std::chrono::steady_clock::now() - startTime) >= waitCmdTimeout) {
                HCCL_ERROR(
                    "[HcclCommunicator][%s]Wait DestroyExec response status timeout[%u ms] and get the "
                    "ExecState is [%d].",
                    __func__, waitCmdTimeoutMs, status.execStatus.kfcStatus);
                return HCCL_E_INTERNAL;
            }
        }
    }

    return HCCL_SUCCESS;
}

u32 HcclCommunicator::GetHostPort(s32 devicePhyId)
{
    if (GetExternalInputHcclIfBasePort() == HCCL_INVALID_PORT) {
        return (devicePhyId + HOST_PARA_BASE_PORT);
    } else {
        return (devicePhyId + GetExternalInputHcclIfBasePort() + HCCL_AISERVER_DEVICE_NUM);
    }
}

HcclResult HcclCommunicator::setVnicIpToRankInfoList()
{
    // 单卡场景不需要获取
    if (userRankSize_ <= 1) {
        HCCL_INFO("user rank size <= 1, ra is not needed for single device.");
        return HCCL_SUCCESS;
    }

    HcclIpAddress vnicIp;
    for (auto& rankInfo : rankInfoList_) {
        if (useSuperPodMode_ && superPodId_ == rankInfo.superPodId) {
            CHK_RET(hrtRaGetSingleSocketVnicIpInfo(
                devicePhyId_, DeviceIdType::DEVICE_ID_TYPE_SDID, rankInfo.superDeviceId, vnicIp));
            rankInfo.deviceVnicIp = vnicIp;
        } else if (serverId_ == rankInfo.serverId) {
            if (rankInfo.devicePhyId != HOST_DEVICE_ID) {
                CHK_RET(hrtRaGetSingleSocketVnicIpInfo(
                    devicePhyId_, DeviceIdType::DEVICE_ID_TYPE_PHY_ID, rankInfo.devicePhyId, vnicIp));
                rankInfo.deviceVnicIp = vnicIp;
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetInfoToDevice(const std::unique_ptr<PreProcessMetaInfo>& preMetaInfo, Stream& stream)
{
    auto inAlltoAllvParaBuffer = cclBufferManager_.GetInAlltoAllvParaBuffer();
    auto outAlltoAllvParaBuffer = cclBufferManager_.GetOutAlltoAllvParaBuffer();
    if ((inAlltoAllvParaBuffer.ptr() == nullptr) || (outAlltoAllvParaBuffer.ptr() == nullptr)) {
        CHK_RET(cclBufferManager_.InitAlltoAllvParaBuffer(preMetaInfo->inputSize, preMetaInfo->outputSize));
        inAlltoAllvParaBuffer = cclBufferManager_.GetInAlltoAllvParaBuffer();
        outAlltoAllvParaBuffer = cclBufferManager_.GetOutAlltoAllvParaBuffer();
    }

    auto inCCLbuffer = cclBufferManager_.GetInCCLbuffer();
    auto outCCLbuffer = cclBufferManager_.GetOutCCLbuffer();
    auto expBuffer = cclBufferManager_.GetCommExpBuffer();
    if ((inCCLbuffer.ptr() == nullptr) || (outCCLbuffer.ptr() == nullptr) || (expBuffer.ptr() == nullptr)) {
        CHK_RET(CreateCommCCLbuffer());
        inCCLbuffer = cclBufferManager_.GetInCCLbuffer();
        outCCLbuffer = cclBufferManager_.GetOutCCLbuffer();
        expBuffer = cclBufferManager_.GetCommExpBuffer();
    }

    CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
    CHK_RET(hcclStreamSynchronize(stream.ptr(), commConfig_.GetConfigExecTimeOut()));
    CHK_RET(hrtMemSyncCopy(
        inAlltoAllvParaBuffer.ptr(), preMetaInfo->inputSize, preMetaInfo->inputData.data(), preMetaInfo->inputSize,
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetInfoFromDevice(
    const std::unique_ptr<PreProcessMetaInfo>& preMetaInfo, const HcclWorkflowMode& mode, HostMem& hostCollectBuffer)
{
    CHK_RET(hrtMemSyncCopy(
        hostCollectBuffer.ptr(), preMetaInfo->outputSize, cclBufferManager_.GetOutAlltoAllvParaBuffer().ptr(),
        preMetaInfo->outputSize, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST));

    // 非单算子场景，中转内存使用完之后直接释放
    if (mode != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        cclBufferManager_.ReleaseAlltoAllvParaBuffer();
    }

    return HCCL_SUCCESS;
}

DevType HcclCommunicator::NslbGetDeviceType() const { return deviceType_; }

u32 HcclCommunicator::NslbGetServerNum() const { return serverNum_; }

HcclResult HcclCommunicator::NslbDp_CollectOperTable(
    HcclCMDType opType, OpParam& opParam, AlgType nslbAlgType, std::string& algName)
{
    HCCL_INFO("NSLBDP-HCCL try to collect Table NSLBDP_TYPE_TBL_OPER.");
    u32 srcLocalRankId = userRank_;
    u32 rootRank = opParam.root;
    if (opParam.root == INVALID_VALUE_RANKID) {
        rootRank = 0;
    }
    std::string nslb_identifier = identifier_;

    HCCL_INFO("NSLBDP-SWK NslbDp_CollectOperTable nslb_identifier[%s] .", nslb_identifier.c_str());
    u32 rankSize = userRankSize_;
    u64 count = opParam.outputSize;
    if (opParam.outputSize == 0) {
        count = opParam.inputSize;
    }

    if (opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
        u32 perDataSize = SIZE_TABLE[opParam.BatchSendRecvDataDes.sendRecvItemsPtr->dataType];
        count = opParam.BatchSendRecvDataDes.sendRecvItemsPtr->count * perDataSize;
    }

    /* NSLB 填充表2  */
    AlgTypeLevel1 algValue = nslbAlgType.algoLevel1;
    uint8_t nslbAlg = hcclNslbDp::GetInstance().GetNslbLevel1AlgType(algValue);
    if (algValue != AlgTypeLevel1::ALG_LEVEL1_AHC && algValue != AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE) {
        if (NslbGetDeviceType() == DevType::DEV_TYPE_910_93 && NslbGetServerNum() > 1) {
            AlgTypeLevel2 algValue2 = nslbAlgType.algoLevel2;
            nslbAlg = hcclNslbDp::GetInstance().GetNslbLevel2AlgType(algValue2);
        }
    }

    HCCL_INFO("NSLB-HCCL algValue:[%u],nslbAlg[%u], count:[%llu].", algValue, nslbAlg, count);
    if (hcclNslbDp::GetInstance().CheckAlgoConsistency(opType, algName) == true) {
        hcclNslbDp::GetInstance().GenerateOpAndAdjTable(
            opType, rootRank, srcLocalRankId, nslbAlg, nslb_identifier, count, rankSize);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::NslbDp_CollectSendAdjTable(
    HcclCMDType opType, const OpParam& opParam, AlgType nslbAlgType, AdjInfo& nslbAdjInfo)
{
    HCCL_INFO("NSLBDP-HCCL try to collect Table NSLBDP_TYPE_TBL_ADJ.");
    u32 srcLocalRankId = userRank_;
    u32 rootRank = opParam.root;
    if (opParam.root == INVALID_VALUE_RANKID) {
        rootRank = 0;
    }

    std::string nslb_identifier = identifier_;
    AlgTypeLevel1 algValue = nslbAlgType.algoLevel1;
    uint8_t nslbAlg = hcclNslbDp::GetInstance().GetNslbLevel1AlgType(algValue);
    if (algValue != AlgTypeLevel1::ALG_LEVEL1_AHC && algValue != AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE) {
        if (NslbGetDeviceType() == DevType::DEV_TYPE_910_93 && NslbGetServerNum() > 1) {
            AlgTypeLevel2 algValue2 = nslbAlgType.algoLevel2;
            nslbAlg = hcclNslbDp::GetInstance().GetNslbLevel2AlgType(algValue2);
        }
    }
    HCCL_INFO(
        "NSLBDP-SHEN opType:[%u],srcLocalRankId[%u],rootRank[%u],algValue[%u],rankSize[%u]-commDesc[%s].", opType,
        srcLocalRankId, rootRank, algValue, userRankSize_, nslb_identifier.c_str());
    if (opType == HcclCMDType::HCCL_CMD_SEND || opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
        if (nslbAdjInfo.dstRankNum == 0) {
            u32 ringNextRank = (srcLocalRankId + userRankSize_ / 2) % userRankSize_;
            nslbAdjInfo.dstRankNum = 1;
            NslbDpAdjInfo adjInfoStep = {};
            adjInfoStep.dstLocalRankId = ringNextRank;
            adjInfoStep.phaseId = 1;
            adjInfoStep.rev = 0;
            nslbAdjInfo.nsAdjInfo.push_back(adjInfoStep);
        }
    }
    // 填充表3
    hcclNslbDp::GetInstance().GetAlgAdjacencyTable(
        opType, srcLocalRankId, rootRank, nslbAlg, nslb_identifier, nslbAdjInfo);
    hcclNslbDp::GetInstance().SendAlgorithmInfoTable();
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::updateList() const { return HCCL_SUCCESS; }

HcclResult HcclCommunicator::SetMC2EnvFlag()
{
    isNsRecovery_ = true;
    return HCCL_SUCCESS;
}

bool HcclCommunicator::GetMC2EnvFlag() const { return isNsRecovery_; }

bool HcclCommunicator::GetAicpuCommEngine() const { return isAicpuCommEngine_; }

HcclResult HcclCommunicator::SetAicpuCommEngine(bool isAicpuCommEngine)
{
    HCCL_INFO("SetAicpuCommEngine isAicpuCommEngine_[%u]", isAicpuCommEngine);
    isAicpuCommEngine_ = isAicpuCommEngine;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetAicpuUnfoldFlag()
{
    isAicpuUnfold_ = true;
    return HCCL_SUCCESS;
}

bool HcclCommunicator::GetAicpuUnfoldFlag() const { return isAicpuUnfold_; }

HcclResult HcclCommunicator::SetStopFlag(bool value)
{
    if (socketManager_ != nullptr) {
        CHK_RET(socketManager_->SetStopFlag(value));
    }

    if (transportManager_ != nullptr) {
        CHK_RET(transportManager_->SetStopFlag(value));
    }

    if (indptOpTransportManager_ != nullptr) {
        CHK_RET(indptOpTransportManager_->SetStopFlag(value));
    }

    for (auto& entry : resMap_) {                                                           // map
        for (auto& levelNSubCommTransport : entry.second.opTransportResponse) {             // vector
            for (auto& singleSubCommTransport : levelNSubCommTransport) {                   // vector
                for (u32 i = 0; i < singleSubCommTransport.transportRequests.size(); i++) { // vector
                    if (singleSubCommTransport.transportRequests[i].isValid
                        && i < singleSubCommTransport.links.size()) {
                        auto transport = singleSubCommTransport.links[i];
                        if (transport != nullptr) {
                            CHK_RET(transport->SetStopFlag(value));
                        }
                    }
                }
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetState(HcclCommState state)
{
    state_.store(state);
    return HCCL_SUCCESS;
}

HcclCommState HcclCommunicator::GetState() { return state_.load(); }

u32 HcclCommunicator ::HcclGetCmdTimeout() const { return HCCL_AICPU_HOST_BASE_TIME_MS; }

HcclResult HcclCommunicator::Suspend()
{
    CHK_PRT_RET(
        isInvalidComm_,
        HCCL_ERROR(
            "[HcclCommunicator][%s] comm[%s], rank[%u], devId[%d], snapshot recovering, "
            "this comm is invalid.",
            __func__, identifier_.c_str(), userRank_, deviceLogicId_),
        HCCL_E_UNAVAIL);
    return Suspend(kfcControlTransferH2D_, kfcStatusTransferD2H_);
}

HcclResult
HcclCommunicator::Suspend(std::shared_ptr<HDCommunicate>& controlH2D, std::shared_ptr<HDCommunicate>& statusD2H)
{
    isSuspending = true;
    if (GetAicpuUnfoldFlag() || GetAicpuCommEngine()) {
        HCCL_DEBUG("[NsRecovery]MC2 OR AICPU ENVIRONMENT TO RECOVERY");
        KfcExecControl execCommand;
        execCommand.kfcCmd = KfcCommand::NsStopLaunch;
        execCommand.bgCmd = BackgroundCommand::kNone;
        execCommand.suspendingStatus = HcclComSuspendingFlag::isSuspending;
        HCCL_RUN_INFO(
            "[NsRecovery][SetOpExecCmd]set the suspending flag [%d] and set KfcCommand [%d], group[%s]",
            execCommand.suspendingStatus, execCommand.kfcCmd, identifier_.c_str());

        CHK_RET(CheckSetRetryStateToWaitResume());

        CHK_RET(controlH2D->Put(0, sizeof(KfcExecControl), reinterpret_cast<uint8_t*>(&execCommand)));
        KfcExecStatus opInfo;
        auto waitStopExecCmdTimeoutMs = HcclGetCmdTimeout();
        auto waitStopExecCmdTimeout = std::chrono::milliseconds(waitStopExecCmdTimeoutMs);
        auto startTime = std::chrono::steady_clock::now();
        while (true) {
            CHK_RET(statusD2H->Get(0, sizeof(KfcExecStatus), reinterpret_cast<uint8_t*>(&opInfo)));
            if (opInfo.execStatus.kfcStatus == KfcStatus::kStoplaunch) {
                HCCL_RUN_INFO("[NsRecovery]opExecState[%d], opId[%u]", opInfo.execStatus.kfcStatus, opInfo.opId.index);
                return HCCL_E_SUSPENDING;
            } else if (opInfo.execStatus.kfcStatus == KfcStatus::kError) {
                return HCCL_E_INTERNAL;
            } else {
                if ((std::chrono::steady_clock::now() - startTime) >= waitStopExecCmdTimeout) {
                    HCCL_ERROR(
                        "[NsRecovery]Wait suspend response status timeout[%u ms] and get the opExecState is [%u] and "
                        "opId[%u].",
                        waitStopExecCmdTimeoutMs, opInfo.execStatus.kfcStatus, opInfo.opId.index);

                    return HCCL_E_INTERNAL;
                }
                continue;
            }
        }
    } else {
        HCCL_RUN_INFO("[NsRecovery] not mc2 or aicpu ENVIRONMENT, group[%s]", identifier_.c_str());
        return HCCL_SUCCESS;
    }
}

HcclResult HcclCommunicator::StopExec()
{
    CHK_PRT_RET(
        isInvalidComm_,
        HCCL_ERROR(
            "[HcclCommunicator][%s] comm[%s], rank[%u], devId[%d], snapshot recovering, "
            "this comm is invalid.",
            __func__, identifier_.c_str(), userRank_, deviceLogicId_),
        HCCL_E_UNAVAIL);
    return StopExec(kfcControlTransferH2D_, kfcStatusTransferD2H_);
}

HcclResult
HcclCommunicator::StopExec(std::shared_ptr<HDCommunicate>& controlH2D, std::shared_ptr<HDCommunicate>& statusD2H)
{
    isSuspending = true;
    if (GetAicpuUnfoldFlag() || GetAicpuCommEngine()) {
        HCCL_DEBUG("[NsRecovery]MC2 OR AICPU ENVIRONMENT TO RECOVERY");
        KfcExecStatus opInfo;
        CHK_RET(statusD2H->Get(0, sizeof(KfcExecStatus), reinterpret_cast<uint8_t*>(&opInfo)));
        HCCL_DEBUG(
            "[NsRecovery][GetOpExecInfo] opExeState[%d], opId[%u]", opInfo.execStatus.kfcStatus, opInfo.opId.index);
        if (opInfo.execStatus.kfcStatus == KfcStatus::kStoplaunch) {
            KfcCommand opCmd = KfcCommand::NsStopExec;
            HCCL_RUN_INFO("[NsRecovery][SetOpExecCmd]set KfcCommand [%d], group[%s]", opCmd, identifier_.c_str());
            CHK_RET(controlH2D->Put(0, sizeof(KfcCommand), reinterpret_cast<uint8_t*>(&opCmd)));
            auto waitStopExecCmdTimeoutMs = HcclGetCmdTimeout();
            auto waitStopExecCmdTimeout = std::chrono::milliseconds(waitStopExecCmdTimeoutMs);
            auto startTime = std::chrono::steady_clock::now();
            while (true) {
                CHK_RET(statusD2H->Get(0, sizeof(KfcExecStatus), reinterpret_cast<uint8_t*>(&opInfo)));
                if (opInfo.execStatus.kfcStatus == KfcStatus::kStopExec) {
                    HCCL_RUN_INFO(
                        "[NsRecovery]opExecState[%d], opId[%u]", opInfo.execStatus.kfcStatus, opInfo.opId.index);
                    return HCCL_E_SUSPENDING;
                } else if (opInfo.execStatus.kfcStatus == KfcStatus::kEnd) {
                    HCCL_RUN_INFO(
                        "[NsRecovery]opExecState[%d], opId[%u]", opInfo.execStatus.kfcStatus, opInfo.opId.index);
                    return HCCL_SUCCESS;
                } else if (opInfo.execStatus.kfcStatus == KfcStatus::kError) {
                    return HCCL_E_INTERNAL;
                } else {
                    if ((std::chrono::steady_clock::now() - startTime) >= waitStopExecCmdTimeout) {
                        HCCL_ERROR(
                            "[NsRecovery]Wait stopExec response status timeout[%u ms] and get the opExecState is [%u] "
                            "and opId[%u].",
                            waitStopExecCmdTimeoutMs, opInfo.execStatus.kfcStatus, opInfo.opId.index);
                        return HCCL_E_INTERNAL;
                    }
                    continue;
                }
            }
        } else {
            return HCCL_SUCCESS;
        }
    } else {
        HCCL_DEBUG("[NsRecovery] not mc2 or aicpu ENVIRONMENT");
        return HCCL_SUCCESS;
    }
}

HcclResult HcclCommunicator::Clean() { return Clean(kfcControlTransferH2D_, kfcStatusTransferD2H_); }

HcclResult
HcclCommunicator::Clean(std::shared_ptr<HDCommunicate>& controlH2D, std::shared_ptr<HDCommunicate>& statusD2H)
{
    isSuspending = true;
    if (GetAicpuUnfoldFlag() || GetAicpuCommEngine()) {
        HCCL_DEBUG("[NsRecovery]MC2 OR AICPU ENVIRONMENT TO RECOVERY");
        KfcExecStatus opInfo;
        CHK_RET(statusD2H->Get(0, sizeof(KfcExecStatus), reinterpret_cast<uint8_t*>(&opInfo)));
        HCCL_DEBUG(
            "[NsRecovery][GetOpExecInfo] opExeState[%d], opId[%u]", opInfo.execStatus.kfcStatus, opInfo.opId.index);
        if (opInfo.execStatus.kfcStatus == KfcStatus::kStopExec || opInfo.execStatus.kfcStatus == KfcStatus::kEnd) {
            KfcCommand opCmd = KfcCommand::NsClear;
            HCCL_RUN_INFO("[NsRecovery][SetOpExecCmd]set KfcCommand [%d], group[%s]", opCmd, identifier_.c_str());
            CHK_RET(controlH2D->Put(0, sizeof(KfcCommand), reinterpret_cast<uint8_t*>(&opCmd)));
            auto waitStopExecCmdTimeoutMs = HcclGetCmdTimeout();
            auto waitStopExecCmdTimeout = std::chrono::milliseconds(waitStopExecCmdTimeoutMs);
            auto startTime = std::chrono::steady_clock::now();
            while (true) {
                CHK_RET(statusD2H->Get(0, sizeof(KfcExecStatus), reinterpret_cast<uint8_t*>(&opInfo)));
                if (opInfo.execStatus.kfcStatus == KfcStatus::kClear) {
                    HCCL_RUN_INFO(
                        "[NsRecovery]opExecState[%d], opId[%u]", opInfo.execStatus.kfcStatus, opInfo.opId.index);
                    return HCCL_E_SUSPENDING;
                } else if (opInfo.execStatus.kfcStatus == KfcStatus::kEnd) {
                    HCCL_RUN_INFO(
                        "[NsRecovery]opExecState[%d], opId[%u]", opInfo.execStatus.kfcStatus, opInfo.opId.index);
                    return HCCL_SUCCESS;
                } else if (opInfo.execStatus.kfcStatus == KfcStatus::kError) {
                    return HCCL_E_INTERNAL;
                } else {
                    if ((std::chrono::steady_clock::now() - startTime) >= waitStopExecCmdTimeout) {
                        HCCL_ERROR(
                            "[NsRecovery]Wait clean response status timeout[%u ms] and get the opExecState is [%u] and "
                            "opId[%u].",
                            waitStopExecCmdTimeoutMs, opInfo.execStatus.kfcStatus, opInfo.opId.index);
                        return HCCL_E_INTERNAL;
                    }
                    continue;
                }
            }
        } else {
            return HCCL_SUCCESS;
        }
    } else {
        HCCL_DEBUG("[NsRecovery] not mc2 or aicpu ENVIRONMENT");
        return HCCL_SUCCESS;
    }
}

HcclResult HcclCommunicator::CheckNotifyOrQPMaxNum(u64& existNum, const u64& MaxNum, const bool& isNotifyRes)
{
    std::string resType = isNotifyRes ? "Notify" : "QP";
    if (existNum + 1 > MaxNum) {
        HCCL_ERROR(
            "[%s]%s resources are insufficient, existNum[%llu], MaxNum is [%llu]", __func__, resType.c_str(), existNum,
            MaxNum);
        return HCCL_E_INTERNAL;
    }
    HCCL_DEBUG(
        "[%s]%s resources are sufficient, existNum[%llu], MaxNum is [%llu]", __func__, resType.c_str(), existNum,
        MaxNum);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CopyHostOpRemoteResToDeviceParam(const std::string& newTag)
{
    HCCL_DEBUG("[%s] remote resource, tag[%s]", __func__, newTag.c_str());
    for (u32 userRankIdx = 0; userRankIdx < AICPU_MAX_RANK_NUM; userRankIdx++) {
        if (opResPara_.remoteRes[userRankIdx].nextHostPtr == 0
            && opResPara_.remoteRes[userRankIdx].nextDevicePtr == 0) {
            continue;
        }
        // 1、将rank公共资源，H2D到device
        HcclRankRelationResV2* remoteResHostPtr
            = reinterpret_cast<HcclRankRelationResV2*>(opResPara_.remoteRes[userRankIdx].nextHostPtr);
        HcclRankRelationResV2* remoteResDevicePtr
            = reinterpret_cast<HcclRankRelationResV2*>(opResPara_.remoteRes[userRankIdx].nextDevicePtr);
        CHK_RET(hrtMemSyncCopy(
            static_cast<void*>(remoteResDevicePtr), sizeof(HcclRankRelationResV2), static_cast<void*>(remoteResHostPtr),
            sizeof(HcclRankRelationResV2), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
        HCCL_DEBUG(
            "[%s] remote resource, tag[%s], userRankIx[%u], "
            "cclinbuffer[%p], ccloutbuffer[%p], opResPara_.remoteRes[userRankIdx].nextDevicePtr[%p], "
            "opResPara_.remoteRes[userRankIdx].nextHostPtr[%p]",
            __func__, newTag.c_str(), userRankIdx, remoteResHostPtr->windowsIn, remoteResHostPtr->windowsOut,
            reinterpret_cast<HcclRankRelationResV2*>(opResPara_.remoteRes[userRankIdx].nextDevicePtr),
            reinterpret_cast<HcclRankRelationResV2*>(opResPara_.remoteRes[userRankIdx].nextHostPtr));
        CHK_RET(CopyHostListResToDeviceParam(
            newTag, reinterpret_cast<ListCommon*>(&remoteResHostPtr->nextTagRes), sizeof(HccltagRemoteResV2)));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AicpuResourceRefresh(
    const AlgResourceResponse& algResource, const std::string& newTag, const HcclCMDType opType)
{
    HCCL_INFO(
        "[HcclCommunicator][AicpuResourceRefresh] start refresh aicpu resources newTag[%s] local rankId[%u]",
        newTag.c_str(), userRank_);
    LocalResInfoV2* localResHostPtr = &opResPara_.localRes;
    opResPara_.winSize = algResource.cclInputMem.size();
    opResPara_.localWindowsIn = reinterpret_cast<u64>(algResource.cclInputMem.ptr());
    opResPara_.localWindowsOut = reinterpret_cast<u64>(algResource.cclOutputMem.ptr());
    CHK_RET(BuildOpLocalScratchMemResParam(algResource, newTag, localResHostPtr));
    CHK_RET(BuildOpRemoteResParam(algResource, newTag, opType));
    CHK_RET(BuildZeroCopyParam());
    CHK_RET(CopyHostOpResToDeviceParam(newTag));
    newTagResAlloced_.insert(newTag);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AddGroupTagInfo([[maybe_unused]] const std::string& tag, bool isAiv)
{
    HCCL_PROFILER_ADD_GROUPRANK(identifier_, userRankSize_, userRank_);
    if (isAiv) {
        HCCL_PROFILER_ADD_TAG_AIV(tag, identifier_, GetWorkflowMode());
    } else {
        HCCL_PROFILER_ADD_TAG(tag, identifier_, GetWorkflowMode());
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::UnRegisterDfxInfo(const OpParam& param, const std::vector<Stream>& slaveStreams)
{
    HCCL_PROFILER_DEL_TAG(param.tag);
    HCCL_PROFILER_DEL_GROUPRANK(identifier_);
    HCCL_PROFILER_DEL_STREAM_BY_STREAMID(param.stream.id());
    HCCL_PROFILER_DEL_OPDATA(param.tag);
    if (((GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)
         && !hccl::ProfilingManagerPub::GetAdditionInfoState() && !hccl::ProfilingManagerPub::GetTaskApiState())
        && !param.isCapture) {
        return HCCL_SUCCESS;
    }
    for (auto subStream : slaveStreams) {
        HCCL_PROFILER_DEL_STREAM_BY_STREAMID(subStream.id());
    }
    return HCCL_SUCCESS;
}

// 判断AICPU展开是否需要都走OpBase模式
bool HcclCommunicator::IsForceAicpuOpBaseMode(const OpParam& opParam, const HcclCMDType& opType)
{
    // 目前alltoall系列算子在aicpu展开场景下仍走原有的OpBase模式
    // ZeroCopy特性也强制走OpBase流程
    if (opParam.aicpuUnfoldMode
        && (opType == HcclCMDType::HCCL_CMD_ALLTOALL || opType == HcclCMDType::HCCL_CMD_ALLTOALLV
            || opType == HcclCMDType::HCCL_CMD_ALLTOALLVC || opParam.isZeroCopy)) {
        return true;
    }

    return false;
}

HcclResult HcclCommunicator::AllocOpBaseModeScratchMem(
    HcclCMDType opType, const OpParam& opParam, AlgResourceRequest& resRequest, AlgResourceResponse& algResResponse)
{
    if (resRequest.scratchMemSize == 0) {
        return HCCL_SUCCESS;
    }

    if (opParam.isZeroCopy) {
        if (opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
            // 零拷贝场景不需要进行scratchMem申请
            DeviceMem tmpBuffer
                = DeviceMem::create(opParam.inputPtr, resRequest.scratchMemSize + CCE_REDUCE_ALIGN_SIZE);
            // cce reduce地址32字节对齐，截取32字节对齐后的内存地址
            u32 addOffset = (reinterpret_cast<uintptr_t>(tmpBuffer.ptr())) % CCE_REDUCE_ALIGN_SIZE;
            u64 totalSize = userRankSize_ * opParam.DataDes.count * SIZE_TABLE[opParam.DataDes.dataType];
            algResResponse.scratchMem = addOffset == 0 ? tmpBuffer.range(addOffset, totalSize) :
                                                         tmpBuffer.range(CCE_REDUCE_ALIGN_SIZE - addOffset, totalSize);
            deviceResOrigMem_.emplace_back(std::move(tmpBuffer));
        } else {
            algResResponse.scratchMem = DeviceMem::create(opParam.inputPtr, resRequest.scratchMemSize);
        }
    } else {
        if (opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
            DeviceMem tmpBuffer;
            CHK_RET(DeviceMem::alloc(tmpBuffer, resRequest.scratchMemSize + CCE_REDUCE_ALIGN_SIZE));
            // cce reduce地址32字节对齐，截取32字节对齐后的内存地址
            u32 addOffset = (reinterpret_cast<uintptr_t>(tmpBuffer.ptr())) % CCE_REDUCE_ALIGN_SIZE;
            algResResponse.scratchMem
                = addOffset == 0 ?
                      tmpBuffer.range(addOffset, cclBufferManager_.GetInCCLbufferSize()) :
                      tmpBuffer.range(CCE_REDUCE_ALIGN_SIZE - addOffset, cclBufferManager_.GetInCCLbufferSize());
            deviceResOrigMem_.emplace_back(std::move(tmpBuffer));
        } else {
            CHK_RET(DeviceMem::alloc(algResResponse.scratchMem, resRequest.scratchMemSize));
        }
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetAlgInfo(
    const std::string& algConfig, const std::string& tag, HcclCMDType commType, std::string& algName,
    std::string& newTag)
{
    // 查表
    CHK_PRT_RET(
        (ALGCFG_TO_NAME.find(algConfig) == ALGCFG_TO_NAME.end()),
        HCCL_ERROR("[%s] invalid algConfig=[%s]", __func__, algConfig.c_str()), HCCL_E_PARA);

    auto iter = CMDTYPE_TO_KEYWORD.find(commType);
    CHK_PRT_RET(
        (iter == CMDTYPE_TO_KEYWORD.end()),
        HCCL_ERROR("[%s] invalid commType=[%d]", __func__, static_cast<int>(commType)), HCCL_E_PARA);
    CHK_PRT_RET(
        (algConfig.find(iter->second) == algConfig.npos),
        HCCL_ERROR(
            "[%s] commType=[%d] not support algConfig=[%s]", __func__, static_cast<int>(commType), algConfig.c_str()),
        HCCL_E_PARA);

    algName = ALGCFG_TO_NAME[algConfig];

    TopoType topoType;
    CHK_RET(implAlg_->GetTopoType(topoType));
    if (topoType == TopoType::TOPO_TYPE_NP_SINGLE_RING) {
        if (algConfig == "AllGather=level0:doublering" || algConfig == "ReduceScatter=level0:doublering"
            || algConfig == "AllReduce=level0:doublering") {
            std::size_t found = algConfig.find(":");
            std::string algConfigTmp = algConfig.substr(0, found + 1) + "ring";
            algName = ALGCFG_TO_NAME[algConfigTmp];
        }
    }
    newTag = tag + algName + "_device";
    HCCL_INFO(
        "[%s] tag=[%s], algName=[%s], newTag=[%s], topoType=[%d]", __func__, tag.c_str(), algName.c_str(),
        newTag.c_str(), topoType);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CreateAndGetAiCpuNotifyWithNotifyRes(HcclSignalInfo& notifyInfo)
{
    if (localAiCpuNotifyRes_.size() > 0) {
        CHK_RET(CreateAndGetAiCpuNotify(localAiCpuNotifyRes_[0], notifyInfo));
    } else {
        std::shared_ptr<LocalNotify> localNotify = {nullptr};
        CHK_RET(CreateAndGetAiCpuNotify(localNotify, notifyInfo));
        localAiCpuNotifyRes_.push_back(localNotify);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetDynamicTilingDataAlltoall(const OpParam& opParam, HostMem& dynamicDataMem)
{
    struct OpTilingAllToAllDataDes* a2ADataPtr
        = reinterpret_cast<struct OpTilingAllToAllDataDes*>(dynamicDataMem.ptr());
    a2ADataPtr->sendType = static_cast<u8>(opParam.All2AllDataDes.sendType);
    a2ADataPtr->recvType = static_cast<u8>(opParam.All2AllDataDes.recvType);
    a2ADataPtr->sendCount = opParam.All2AllDataDes.sendCount;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetDynamicTilingDataAlltoallv(
    const OpParam& opParam, HostMem& dynamicDataMem, const std::string& algName)
{
    struct OpTilingAlltoallvDataDes* alltoallvDataPtr
        = reinterpret_cast<struct OpTilingAlltoallvDataDes*>(dynamicDataMem.ptr());
    alltoallvDataPtr->sendType = static_cast<u8>(opParam.All2AllDataDes.sendType);
    alltoallvDataPtr->recvType = static_cast<u8>(opParam.All2AllDataDes.recvType);
    u32 rankSize = GetRankSize();
    u64* sendCountsPtr = static_cast<u64*>(alltoallvDataPtr->sendRecvInfos);
    u64* recvCountsPtr = sendCountsPtr + rankSize;
    u64* sdisplsPtr = recvCountsPtr + rankSize;
    u64* rdisplsPtr = sdisplsPtr + rankSize;
    for (u32 i = 0; i < rankSize; i++) {
        CHK_PTR_NULL(static_cast<const u64*>(opParam.All2AllDataDes.sendCounts) + i);
        sendCountsPtr[i] = *(static_cast<const u64*>(opParam.All2AllDataDes.sendCounts) + i);
        CHK_PTR_NULL(static_cast<const u64*>(opParam.All2AllDataDes.recvCounts) + i);
        recvCountsPtr[i] = *(static_cast<const u64*>(opParam.All2AllDataDes.recvCounts) + i);
        CHK_PTR_NULL(static_cast<const u64*>(opParam.All2AllDataDes.sdispls) + i);
        sdisplsPtr[i] = *(static_cast<const u64*>(opParam.All2AllDataDes.sdispls) + i);
        CHK_PTR_NULL(static_cast<const u64*>(opParam.All2AllDataDes.rdispls) + i);
        rdisplsPtr[i] = *(static_cast<const u64*>(opParam.All2AllDataDes.rdispls) + i);
        HCCL_DEBUG(
            "[SetDynamicTilingDataAlltoallv] sendCounts[%llu], recvCounts[%llu], sdispls[%llu], rdispls[%llu]",
            sendCountsPtr[i], recvCountsPtr[i], sdisplsPtr[i], rdisplsPtr[i]);
    }

    if (algName == "RunAlltoAllVTwoLevelPipeline") {
        u64* sendRecvInfoPtr = rdisplsPtr + rankSize;
        CHK_SAFETY_FUNC_RET(
            memcpy_s(sendRecvInfoPtr, hostCollectBuffer_.size(), hostCollectBuffer_.ptr(), hostCollectBuffer_.size()));
    }

    HCCL_DEBUG(
        "[SetDynamicTilingDataAlltoallv] set dynamic tiling data for AllToAllV success, alltoallvDataPtr[%p]",
        alltoallvDataPtr);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetDynamicTilingDataAlltoallvc(const OpParam& opParam, HostMem& dynamicDataMem)
{
    struct OpTilingAlltoallvcDataDes* a2ADataPtr
        = reinterpret_cast<struct OpTilingAlltoallvcDataDes*>(dynamicDataMem.ptr());
    a2ADataPtr->sendType = static_cast<u8>(opParam.All2AllDataDes.sendType);
    a2ADataPtr->recvType = static_cast<u8>(opParam.All2AllDataDes.recvType);
    u32 rankSize = GetRankSize();
    for (u64 i = 0; i < rankSize * rankSize; i++) {
        a2ADataPtr->sendCountMatrix[i] = *(static_cast<const u64*>(opParam.All2AllDataDes.sendCountMatrix) + i);
    }
    return HCCL_SUCCESS;
}

u64 HcclCommunicator::CalcOpTilingVDataDesVDataLen(const u32 rankSize) const
{
    const u32 vFactor = 2; // counts和displs 2个变长数组
    return vFactor * rankSize * sizeof(u64);
}

HcclResult HcclCommunicator::SetDynamicTilingDataV(const OpParam& opParam, HostMem& dynamicDataMem)
{
    CHK_PTR_NULL(static_cast<const u64*>(opParam.VDataDes.counts));
    CHK_PTR_NULL(static_cast<const u64*>(opParam.VDataDes.displs));

    const u32 rankSize = GetRankSize();
    struct OpTilingVDataDes* vDataPtr = reinterpret_cast<struct OpTilingVDataDes*>(dynamicDataMem.ptr());
    vDataPtr->dataType = static_cast<u8>(opParam.VDataDes.dataType);
    vDataPtr->vDataLen = CalcOpTilingVDataDesVDataLen(rankSize);

    u64* countsPtr = static_cast<u64*>(vDataPtr->vData);
    u64* displsPtr = countsPtr + rankSize;
    for (u32 i = 0; i < rankSize; ++i) {
        countsPtr[i] = *(static_cast<const u64*>(opParam.VDataDes.counts) + i);
        displsPtr[i] = *(static_cast<const u64*>(opParam.VDataDes.displs) + i);
        HCCL_DEBUG("[SetDynamicTilingDataV][%u] counts[%llu], displs[%llu]", i, countsPtr[i], displsPtr[i]);
    }

    HCCL_DEBUG("[SetDynamicTilingDataV] set dynamic tiling data success, vDataPtr[%p]", vDataPtr);
    return HCCL_SUCCESS;
}

u64 HcclCommunicator::CalcOpTilingDynamicDataSize(
    const OpParam& opParam, const HcclCMDType& opType, const u32& rankSize, const std::string& algName)
{
    u64 dynamicDataSize = 0ULL;
    if (opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
        dynamicDataSize = sizeof(struct OpTilingBatchSendRecvDataDes)
                          + opParam.BatchSendRecvDataDes.itemNum * sizeof(HcclSendRecvItem)
                          + userRankSize_ * sizeof(u8);
    } else if (opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        dynamicDataSize = sizeof(struct OpTilingAllToAllDataDes);
    } else if (opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        dynamicDataSize = sizeof(struct OpTilingAlltoallvDataDes) + rankSize * ALLTOALL_INFO_MATRIX_SIZE * sizeof(u64);
        if (algName == "RunAlltoAllVTwoLevelPipeline") {
            dynamicDataSize += hostCollectBuffer_.size();
        }
    } else if (opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        dynamicDataSize = sizeof(struct OpTilingAlltoallvcDataDes) + rankSize * rankSize * sizeof(u64);
    } else if (opType == HcclCMDType::HCCL_CMD_ALLGATHER_V || opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
        dynamicDataSize = sizeof(struct OpTilingVDataDes) + CalcOpTilingVDataDesVDataLen(rankSize);
    } else {
        dynamicDataSize = sizeof(struct OpTilingDataDes);
    }
    return dynamicDataSize;
}

HcclResult HcclCommunicator::AicpuInitOpTilingDataFromOpParam(
    const OpParam& opParam, const HcclCMDType& opType, struct OpTilingData* opTilingData)
{
    opTilingData->workflowMode = (IsForceAicpuOpBaseMode(opParam, opType) && !opParam.isZeroCopy) ?
                                     static_cast<u8>(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) :
                                     static_cast<u8>(GetWorkflowMode());
    opTilingData->inputPtr = reinterpret_cast<u64>(opParam.inputPtr);
    opTilingData->outputPtr = reinterpret_cast<u64>(opParam.outputPtr);
    opTilingData->reduceType = static_cast<u8>(opParam.reduceType);
    opTilingData->syncMode = static_cast<u8>(opParam.syncMode);
    opTilingData->root = opParam.root;
    opTilingData->dstRank = opParam.dstRank;
    opTilingData->srcRank = opParam.srcRank;
    opTilingData->opType = static_cast<u8>(opType);
    opTilingData->inplaceSupportRetry = static_cast<u8>(inplaceSupportRetry_);
    opTilingData->retryEnable = static_cast<u8>(retryEnable_);
    opTilingData->inPlaceSupportRetryStatus = static_cast<u8>(inPlaceSupportRetryStatus_);
    opTilingData->isInplacePreSync = static_cast<u8>(isInplacePreSync_);
    opTilingData->isPostSync = static_cast<u8>(isPostSync_);
    opTilingData->userStreamId = opParam.stream.id();
    opTilingData->inputSymWindow = reinterpret_cast<u64>(opParam.inputSymWindow);
    opTilingData->inputOffset = opParam.inputOffset;
    opTilingData->outputSymWindow = reinterpret_cast<u64>(opParam.outputSymWindow);
    opTilingData->outputOffset = opParam.outputOffset;
    opTilingData->aicpuUnfoldMode = static_cast<u8>(opParam.aicpuUnfoldMode);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::KernelLaunchChooseAicpuOrCustom(
    void* inputPtr, void* outputPtr, const rtStream_t stm, u64 addr, void* tilingDataPtr, u32 tilingDataSize,
    const std::string& kernelName, HcclWorkflowMode mode, const std::string& tag, bool isCustom)
{
    return AicpuUnfoldKernelLaunchV2(
        inputPtr, outputPtr, stm, addr, tilingDataPtr, tilingDataSize, kernelName, mode, tag, isCustom);
}

HcclResult HcclCommunicator::SaveTraceInfo(std::string& logInfo)
{
    opBaseAtraceInfo_->SaveTraceInfo(logInfo, AtraceOption::Opbasekey);
    return HCCL_SUCCESS;
}

std::unordered_map<std::string, std::map<u32, HcclIpAddress>> HcclCommunicator::GetPhyIdNicInfo()
{
    return rankDevicePhyIdNicInfoMap_;
}

vector<u32> HcclCommunicator::GetRanksPort() { return nicRanksPort_; }

vector<RankInfo> HcclCommunicator::GetRanksList() { return rankInfoList_; }

std::string HcclCommunicator::GetUniqueId(void)
{
    static std::atomic<u32> idCounter(0);

    std::string uniqueId("");
    uniqueId += std::to_string(SalGetPid());
    uniqueId += '-';
    uniqueId += std::to_string(idCounter.fetch_add(1));
    uniqueId += '-';
    uniqueId += std::to_string(SalGetSysTime());

    return uniqueId;
}

u8 HcclCommunicator::GetDeterministicConfig() const
{
    CHK_SMART_PTR_NULL(implAlg_);
    return implAlg_->GetDeterministicConfig();
}

HcclResult HcclCommunicator::SetDeterministicConfig(const u8 deterministic)
{
    CHK_SMART_PTR_NULL(implAlg_);
    CHK_RET(implAlg_->SetDeterministicConfig(deterministic));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::MigrateLinkToStopOrResume(LINK& link, bool isStop)
{
    if (isStop) {
        return link->Stop();
    }
    return link->Resume();
}

HcclResult HcclCommunicator::MigrateLinkVectorToStopOrResume(const std::vector<LINK>& links, bool isStop)
{
    for (auto it : links) {
        if (it) {
            CHK_RET(MigrateLinkToStopOrResume(it, isStop));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::TraverseLinkVector(std::vector<std::unique_ptr<CommBase>>& commBaseVector, bool isStop)
{
    for (unsigned int i = 0; i < commBaseVector.size(); i++) {
        auto commBase = commBaseVector[i].get();
        if (commBase == nullptr) {
            continue;
        }
        const std::vector<LINK>& ret = commBase->TransportInfo();
        CHK_RET(MigrateLinkVectorToStopOrResume(ret, isStop));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::TraverseSingleSubCommTransport(SingleSubCommTransport& commTransport, bool isStop)
{
    for (unsigned int i = 0; i < commTransport.transportRequests.size(); i++) {
        if (!commTransport.transportRequests[i].isValid) {
            continue;
        }
        if (commTransport.links[i] == nullptr) {
            continue;
        }

        if (isStop) {
            CHK_RET(commTransport.links[i]->Stop());
        } else {
            CHK_RET(commTransport.links[i]->Resume());
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::TraverseLevelNSubCommTransport(LevelNSubCommTransport& levelNSubCommTransport, bool isStop)
{
    for (unsigned int jj = 0; jj < levelNSubCommTransport.size(); jj++) {
        CHK_RET(TraverseSingleSubCommTransport(levelNSubCommTransport[jj], isStop));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::TraverseOpCommTransport(OpCommTransport& opCommTransport, bool isStop)
{
    for (unsigned int ii = 0; ii < opCommTransport.size(); ii++) {
        CHK_RET(TraverseLevelNSubCommTransport(opCommTransport[ii], isStop));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::TraverseAlgResourceResponse(bool isStop)
{
    for (auto& it : resMap_) {
        CHK_RET(TraverseOpCommTransport(it.second.opTransportResponse, isStop));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ResetNotify()
{
    CHK_SMART_PTR_NULL(notifyPool_);
    CHK_SMART_PTR_NULL(queueNotifyManagerRefac_);
    notifyPool_->ResetNotify();
    queueNotifyManagerRefac_->ResetNotify();
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ResetNotifyForDestRank(s64 destRank)
{
    CHK_SMART_PTR_NULL(notifyPool_);
    CHK_SMART_PTR_NULL(queueNotifyManagerRefac_);
    notifyPool_->ResetNotifyForDestRank(destRank);
    return HCCL_SUCCESS;
}

void HcclCommunicator::InsertNewTagToTagMap(std::string& newTag, std::string& tag)
{
    const auto& mapIt = newTagToTagMap_.find(newTag);
    if (mapIt == newTagToTagMap_.end()) {
        newTagToTagMap_.insert({newTag, tag});
    } else {
        mapIt->second = tag;
    }
    return;
}

HcclResult HcclCommunicator::GetTagFromNewTag(const std::string& newTag, std::string& tag)
{
    const auto& mapIt = newTagToTagMap_.find(newTag);
    if (mapIt == newTagToTagMap_.end()) {
        HCCL_ERROR("[OpRetry]newTag[%s] is not in newTagToTagMap_", newTag.c_str());
        return HCCL_E_INTERNAL;
    } else {
        tag = mapIt->second;
    }
    return HCCL_SUCCESS;
}

HcclResult
HcclCommunicator::SetSignalTransport(SingleSubCommTransport& singleSubCommTransport, u32 linkIdx, bool statusStop)
{
    RankId loc = singleSubCommTransport.transportRequests[linkIdx].localUserRank;
    RankId rmt = singleSubCommTransport.transportRequests[linkIdx].remoteUserRank;
    if (statusStop) {
        if (singleSubCommTransport.links[linkIdx]
            && singleSubCommTransport.links[linkIdx]->GetLinkType() == LinkType::LINK_ROCE) {
            CHK_RET(singleSubCommTransport.links[linkIdx]->Stop());
            singleSubCommTransport.status[linkIdx] = TransportStatus::STOP;
            HCCL_INFO("[SetTransportStatus]set transport status to stop, loc[%u], rmt[%u]", loc, rmt);
        }
    } else {
        if (singleSubCommTransport.links[linkIdx] && singleSubCommTransport.status[linkIdx] == TransportStatus::STOP) {
            HCCL_INFO("[SetTransportStatus]set transport status to resume, loc[%u], rmt[%u]", loc, rmt);
            CHK_RET(singleSubCommTransport.links[linkIdx]->DeInit());
            singleSubCommTransport.links[linkIdx] = nullptr; // 赋值为nullptr, 供后面重新建链
            singleSubCommTransport.status[linkIdx] = TransportStatus::INIT;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetBsrTransportStatusImpl(
    OpCommTransport& opCommTransport, bool statusStop, const HcclOpIdentifier& opId, u32 remoteRank)
{
    u32 commIndex = 0;
    if ((userRank_ == opId.detRank && remoteRank > userRank_)
        || (userRank_ == opId.srcRank && remoteRank < userRank_)) {
        commIndex = COMM_INDEX_0;
    } else {
        commIndex = COMM_INDEX_1;
    }
    CHK_PRT_RET(
        commIndex >= opCommTransport[COMM_COMBINE_ORDER].size(),
        HCCL_ERROR(
            "[SetBsrTransportStatusImpl] batchsendrecv op commIndex[%u] is larger than "
            "opTransportResponse size[%zu]",
            remoteRank, opCommTransport[COMM_COMBINE_ORDER].size()),
        HCCL_E_PARA);
    SingleSubCommTransport& commCombined
        = const_cast<SingleSubCommTransport&>(opCommTransport[COMM_COMBINE_ORDER][commIndex]);
    u32 Rank = commCombined.userRank2subCommRank[remoteRank];
    CHK_PRT_RET(
        Rank >= commCombined.links.size(),
        HCCL_ERROR(
            "[SetBsrTransportStatusImpl] batchsendrecv op remoteRank[%u], get Rank[%u],"
            "the size of combinedComm links is [%zu]",
            remoteRank, Rank, commCombined.links.size()),
        HCCL_E_PARA);
    CHK_SMART_PTR_NULL(commCombined.links[Rank]);

    RankId loc = commCombined.transportRequests[Rank].localUserRank;
    RankId rmt = commCombined.transportRequests[Rank].remoteUserRank;
    if (!commCombined.transportRequests[Rank].isValid) {
        return HCCL_SUCCESS;
    }
    if (statusStop) {
        if (commCombined.links[Rank]->GetLinkType() == LinkType::LINK_ROCE) {
            CHK_RET(commCombined.links[Rank]->Stop());
            commCombined.status[Rank] = TransportStatus::STOP;
            HCCL_INFO(
                "[SetBsrTransportStatusImpl]set bsr transport status to stop, comindex[%u] loc[%u], rmt[%u]", commIndex,
                loc, rmt);
        }
    } else {
        if (commCombined.status[Rank] == TransportStatus::STOP) {
            HCCL_INFO(
                "[SetBsrTransportStatusImpl]set bsr transport status to resume, comindex[%u] loc[%u], rmt[%u]",
                commIndex, loc, rmt);
            CHK_RET(commCombined.links[Rank]->DeInit());
            commCombined.links[Rank] = nullptr; // 赋值为nullptr, 供后面重新建链
            commCombined.status[Rank] = TransportStatus::INIT;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetBsrTransportStatusImplforchange(
    OpCommTransport& opCommTransport, const HcclOpIdentifier& opId, u32 remoteRank,
    const std::map<u32, bool>& remoteRankPortMap, bool isUseDefault, const std::map<u32, bool>& isChangeLinkMap,
    bool isCurTag)
{
    bool isPortSatisfy
        = (remoteRankPortMap.find(remoteRank) != remoteRankPortMap.end()
           && remoteRankPortMap.find(remoteRank)->second == isUseDefault);
    bool isChangeLink
        = (isChangeLinkMap.find(remoteRank) != isChangeLinkMap.end() && isChangeLinkMap.find(remoteRank)->second);
    HCCL_INFO(
        "[SetBsrTransportStatusImplforchange]remoteRank[%u], isUseDefault[%d], "
        "isPortSatisfy[%d], isChangeLink[%d], isCurTag[%d]",
        remoteRank, isUseDefault, isPortSatisfy, isChangeLink, isCurTag);
    if (!isPortSatisfy || !(isChangeLink || isCurTag)) {
        return HCCL_SUCCESS;
    }

    u32 commIndex = 0;
    if ((userRank_ == opId.detRank && remoteRank > userRank_)
        || (userRank_ == opId.srcRank && remoteRank < userRank_)) {
        commIndex = COMM_INDEX_0;
    } else {
        commIndex = COMM_INDEX_1;
    }
    CHK_PRT_RET(
        commIndex >= opCommTransport[COMM_COMBINE_ORDER].size(),
        HCCL_ERROR(
            "[SetBsrTransportStatusImplforchange] batchsendrecv op commIndex[%u] is larger than "
            "opTransportResponse size[%zu]",
            commIndex, opCommTransport[COMM_COMBINE_ORDER].size()),
        HCCL_E_PARA);
    SingleSubCommTransport& commCombined
        = static_cast<SingleSubCommTransport&>(opCommTransport[COMM_COMBINE_ORDER][commIndex]);
    u32 rank = commCombined.userRank2subCommRank[remoteRank];
    CHK_PRT_RET(
        rank >= commCombined.links.size(),
        HCCL_ERROR(
            "[SetBsrTransportStatusImplforchange] batchsendrecv op remoteRank[%u], get Rank[%u],"
            "the size of combinedComm links is [%zu]",
            remoteRank, rank, commCombined.links.size()),
        HCCL_E_PARA);
    CHK_SMART_PTR_NULL(commCombined.links[rank]);

    RankId loc = commCombined.transportRequests[rank].localUserRank;
    RankId rmt = commCombined.transportRequests[rank].remoteUserRank;
    if (!commCombined.transportRequests[rank].isValid) {
        return HCCL_SUCCESS;
    }

    if (commCombined.status[rank] == TransportStatus::STOP) {
        HCCL_INFO(
            "[SetBsrTransportStatusImplforchange]set bsr transport status to resume, comindex[%u] loc[%u], rmt[%u]",
            commIndex, loc, rmt);
        CHK_RET(commCombined.links[rank]->DeInit());
        commCombined.links[rank] = nullptr; // 赋值为nullptr, 供后面重新建链
        commCombined.status[rank] = TransportStatus::INIT;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetTransportStatusImpl(
    OpCommTransport& opCommTransport, bool statusStop, const HcclOpIdentifier& opId, u32 remoteRank,
    const std::map<u32, bool>& remoteRankPortMap, bool isUseDefault)
{
    bool isSendRecv = opId.isSendRecv;

    // stop阶段及原地重执行的resume阶段
    // bsr判断当前故障的send、recv是否remoterank是否相同的情况，如果是相同只操作故障op，如果不同都操作
    u32 sendRemoteRank = userRank_ == opId.bsrInfo[HCCL_SEND].detRank ? opId.bsrInfo[HCCL_SEND].srcRank :
                                                                        opId.bsrInfo[HCCL_SEND].detRank;
    u32 recvRemoteRank = userRank_ == opId.bsrInfo[HCCL_RECV].detRank ? opId.bsrInfo[HCCL_RECV].srcRank :
                                                                        opId.bsrInfo[HCCL_RECV].detRank;
    bool isBsrPortSatisfy
        = (remoteRankPortMap.find(remoteRank) != remoteRankPortMap.end()
           && remoteRankPortMap.find(remoteRank)->second == isUseDefault);
    bool isQpnSatify = (opId.bsrInfo[HCCL_RECV].tpQpn != 0) && (opId.bsrInfo[HCCL_SEND].tpQpn != 0);
    HCCL_INFO(
        "[SetBsrTransportStatusImpl]SendremoteRank[%u], RecvremoteRank[%u], isUseDefault[%d], isQpnSatisfy[%d], "
        "isBsrPortSatisfy[%d]",
        sendRemoteRank, recvRemoteRank, isUseDefault, isQpnSatify, isBsrPortSatisfy);
    if (opId.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV && sendRemoteRank == recvRemoteRank && isBsrPortSatisfy
        && isQpnSatify) {
        CHK_RET(SetBsrTransportStatusImpl(opCommTransport, statusStop, opId, remoteRank));
        return HCCL_SUCCESS;
    }

    for (auto& levelNSubCommTransport : opCommTransport) {
        for (auto& singleSubCommTransport : levelNSubCommTransport) {
            for (u32 i = 0; i < singleSubCommTransport.transportRequests.size(); i++) {
                u32 transportRemoteRank = singleSubCommTransport.transportRequests[i].remoteUserRank;
                bool isValid = singleSubCommTransport.transportRequests[i].isValid;
                bool isRankSatisfy = ((!isSendRecv) || (isSendRecv && remoteRank == transportRemoteRank));
                // isPortSatisfy表示当前对端使用的主备网口是否和changeLinkInfo一致
                bool isPortSatisfy
                    = (remoteRankPortMap.find(transportRemoteRank) != remoteRankPortMap.end()
                       && remoteRankPortMap.find(transportRemoteRank)->second == isUseDefault);
                HCCL_INFO(
                    "[SetTransportStatus]remoteRank[%u], isUseDefault[%d], isValid[%d], isRankSatisfy[%d], "
                    "isPortSatisfy[%d]",
                    transportRemoteRank, isUseDefault, isValid, isRankSatisfy, isPortSatisfy);
                if (isValid && isRankSatisfy && isPortSatisfy) {
                    CHK_RET(SetSignalTransport(singleSubCommTransport, i, statusStop));
                }
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetTransportStatusImplForChange(
    OpCommTransport& opCommTransport, const HcclOpIdentifier& opId, u32 remoteRank,
    const std::map<u32, bool>& remoteRankPortMap, bool isUseDefault, const std::map<u32, bool>& isChangeLinkMap,
    bool isCurTag)
{
    bool isSendRecv = opId.isSendRecv;

    // bsr判断当前故障的send、recv是否remoterank是否相同的情况，如果是相同只操作故障op，如果不同都操作
    u32 sendRemoteRank = userRank_ == opId.bsrInfo[HCCL_SEND].detRank ? opId.bsrInfo[HCCL_SEND].srcRank :
                                                                        opId.bsrInfo[HCCL_SEND].detRank;
    u32 recvRemoteRank = userRank_ == opId.bsrInfo[HCCL_RECV].detRank ? opId.bsrInfo[HCCL_RECV].srcRank :
                                                                        opId.bsrInfo[HCCL_RECV].detRank;
    bool isQpnSatify = (opId.bsrInfo[HCCL_RECV].tpQpn != 0) && (opId.bsrInfo[HCCL_SEND].tpQpn != 0);
    HCCL_INFO(
        "[SetBsrTransportStatusImpl]SendremoteRank[%u], RecvremoteRank[%u], isUseDefault[%d], isQpnSatisfy[%d]",
        sendRemoteRank, recvRemoteRank, isUseDefault, isQpnSatify);
    if (opId.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV && sendRemoteRank == recvRemoteRank && isQpnSatify) {
        CHK_RET(SetBsrTransportStatusImplforchange(
            opCommTransport, opId, remoteRank, remoteRankPortMap, isUseDefault, isChangeLinkMap, isCurTag));
        return HCCL_SUCCESS;
    }

    // 借轨的resume阶段
    for (auto& levelNSubCommTransport : opCommTransport) {
        for (auto& singleSubCommTransport : levelNSubCommTransport) {
            for (u32 i = 0; i < singleSubCommTransport.transportRequests.size(); i++) {
                u32 transportRemoteRank = singleSubCommTransport.transportRequests[i].remoteUserRank;
                bool isValid = singleSubCommTransport.transportRequests[i].isValid;
                bool isRankSatisfy = (!isSendRecv || (isSendRecv && remoteRank == transportRemoteRank));
                // isPortSatisfy表示当前对端使用的主备网口是否和changeLinkInfo一致
                bool isPortSatisfy
                    = (remoteRankPortMap.find(transportRemoteRank) != remoteRankPortMap.end()
                       && remoteRankPortMap.find(transportRemoteRank)->second == isUseDefault);
                bool isChangeLink
                    = (isChangeLinkMap.find(transportRemoteRank) != isChangeLinkMap.end()
                       && isChangeLinkMap.find(transportRemoteRank)->second);
                HCCL_INFO(
                    "[SetTransportStatus]remoteRank[%u], isUseDefault[%d], isValid[%d], isRankSatisfy[%d], "
                    "isPortSatisfy[%d], isChangeLink[%d], isCurTag[%d]",
                    transportRemoteRank, isUseDefault, isValid, isRankSatisfy, isPortSatisfy, isChangeLink, isCurTag);
                if (isValid && isRankSatisfy && isPortSatisfy && (isChangeLink || isCurTag)) {
                    CHK_RET(SetSignalTransport(singleSubCommTransport, i, false));
                }
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetTransportStatus(
    const HcclOpIdentifier& opId, bool statusStop, const std::map<u32, bool>& remoteRankPortMap,
    const std::map<u32, bool>& isChangeLinkMap, bool isChangeLinkFlag)
{
    std::string newTag(reinterpret_cast<const char*>(opId.newTag));
    u32 remoteRank = userRank_ == opId.detRank ? opId.srcRank : opId.detRank;

    if (resMap_.find(newTag) == resMap_.end()) {
        HCCL_ERROR("HcclCommunicator SetTransportStatus failed: newTag[%s] is not in resMap", newTag.c_str());
        return HCCL_E_INTERNAL;
    }

    if (statusStop) {
        CHK_RET(SetTransportStatusImpl(
            resMap_[newTag].opTransportResponse, statusStop, opId, remoteRank, remoteRankPortMap, true));
        CHK_RET(SetTransportStatusImpl(
            resMap_[newTag].opTransportResponseBackUp, statusStop, opId, remoteRank, remoteRankPortMap, false));
    } else {
        if (isChangeLinkFlag) {
            // 借轨场景
            for (auto& resMapIt : resMap_) {
                bool isCurTag = false;
                if (resMapIt.first == newTag) {
                    isCurTag = true;
                }
                if (hostResMap_.find(resMapIt.first) != hostResMap_.end()) {
                    // 若当前tag未进行aicpu展开，则不重新build资源
                    continue;
                }

                if ((HcclCMDType::HCCL_CMD_BATCH_SEND_RECV == opId.opType && !isCurTag)
                    || (HcclCMDType::HCCL_CMD_BATCH_SEND_RECV != opId.opType
                        && resMapIt.first.find("BatchSendRecv") != std::string::npos)) {
                    continue;
                }
                CHK_RET(SetTransportStatusImplForChange(
                    resMapIt.second.opTransportResponse, opId, remoteRank, remoteRankPortMap, true, isChangeLinkMap,
                    isCurTag));
                CHK_RET(SetTransportStatusImplForChange(
                    resMapIt.second.opTransportResponseBackUp, opId, remoteRank, remoteRankPortMap, false,
                    isChangeLinkMap, isCurTag));

                std::string tag;
                CHK_RET(GetTagFromNewTag(resMapIt.first, tag));
                CHK_RET(ReAllocTransports(tag, resMapIt.first));
                CHK_RET(BuildOpRemoteResParam(resMapIt.second, resMapIt.first, opId.opType, true));
                HCCL_RUN_INFO("[%s]success to set status of [%s] resume", __func__, resMapIt.first.c_str());
            }
            CHK_RET(CopyHostOpResToDeviceParam(newTag));
        } else {
            // 原地重执行
            CHK_RET(SetTransportStatusImpl(
                resMap_[newTag].opTransportResponse, statusStop, opId, remoteRank, remoteRankPortMap, true));
            CHK_RET(SetTransportStatusImpl(
                resMap_[newTag].opTransportResponseBackUp, statusStop, opId, remoteRank, remoteRankPortMap, false));
            std::string tag(reinterpret_cast<const char*>(opId.tag));
            CHK_RET(ReAllocTransports(tag, newTag));
            CHK_RET(BuildOpRemoteResParam(resMap_[newTag], newTag, opId.opType, true));
            CHK_RET(CopyHostOpResToDeviceParam(newTag));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetTransportResumeStatus(
    const std::map<u32, bool>& remoteRankPortMap, const std::map<u32, bool>& isChangeLinkMap, bool isChangeLinkFlag,
    bool statusStop)
{
    HCCL_INFO(
        "[SetTransportResumeStatus]isChangeLinkFlag[%d], rank[%u], group[%s]", isChangeLinkFlag, userRank_,
        identifier_.c_str());

    if (statusStop) {
        for (auto& resMapIt : resMap_) {
            CHK_RET(ResumeTransportsImpl(resMapIt.second.opTransportResponse, remoteRankPortMap, true, statusStop));
            CHK_RET(
                ResumeTransportsImpl(resMapIt.second.opTransportResponseBackUp, remoteRankPortMap, false, statusStop));
        }
    } else {
        if (isChangeLinkFlag) {
            for (auto& resMapIt : resMap_) {
                if (hostResMap_.find(resMapIt.first) != hostResMap_.end()) {
                    continue;
                }
                CHK_RET(ResumeTransportsImplForChange(
                    resMapIt.second.opTransportResponse, remoteRankPortMap, isChangeLinkMap, true));
                CHK_RET(ResumeTransportsImplForChange(
                    resMapIt.second.opTransportResponseBackUp, remoteRankPortMap, isChangeLinkMap, false));

                std::string tag;
                CHK_RET(GetTagFromNewTag(resMapIt.first, tag));
                CHK_RET(ReAllocTransports(tag, resMapIt.first));
                CHK_RET(BuildOpRemoteResParam(resMapIt.second, resMapIt.first, HcclCMDType::HCCL_CMD_ALL, true));
                CHK_RET(CopyHostOpResToDeviceParam(resMapIt.first));
            }
        } else {
            for (auto& resMapIt : resMap_) {
                CHK_RET(ResumeTransportsImpl(resMapIt.second.opTransportResponse, remoteRankPortMap, true, statusStop));
                CHK_RET(ResumeTransportsImpl(
                    resMapIt.second.opTransportResponseBackUp, remoteRankPortMap, false, statusStop));
                std::string tag;
                CHK_RET(GetTagFromNewTag(resMapIt.first, tag));
                CHK_RET(ReAllocTransports(tag, resMapIt.first));
                CHK_RET(BuildOpRemoteResParam(resMapIt.second, resMapIt.first, HcclCMDType::HCCL_CMD_ALL, true));
                CHK_RET(CopyHostOpResToDeviceParam(resMapIt.first));
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ResumeTransportsImplForChange(
    OpCommTransport& opCommTransport, const std::map<u32, bool>& remoteRankPortMap,
    const std::map<u32, bool>& isChangeLinkMap, bool isUseDefault)
{
    // 借轨的resume阶段
    for (auto& levelNSubCommTransport : opCommTransport) {
        for (auto& singleSubCommTransport : levelNSubCommTransport) {
            for (u32 i = 0; i < singleSubCommTransport.transportRequests.size(); i++) {
                u32 transportRemoteRank = singleSubCommTransport.transportRequests[i].remoteUserRank;
                bool isValid = singleSubCommTransport.transportRequests[i].isValid;
                // isPortSatisfy表示当前对端使用的主备网口是否和changeLinkInfo一致
                bool isPortSatisfy
                    = (remoteRankPortMap.find(transportRemoteRank) != remoteRankPortMap.end()
                       && remoteRankPortMap.find(transportRemoteRank)->second == isUseDefault);
                bool isChangeLink = (isChangeLinkMap.find(transportRemoteRank) != isChangeLinkMap.end());
                HCCL_INFO(
                    "[SetTransportStatus]remoteRank[%u], isUseDefault[%d], isValid[%d], "
                    "isPortSatisfy[%d], isChangeLink[%d]",
                    transportRemoteRank, isUseDefault, isValid, isPortSatisfy, isChangeLink);
                if (isValid && isPortSatisfy && isChangeLink) {
                    CHK_RET(SetSignalTransport(singleSubCommTransport, i, false));
                }
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ResumeTransportsImpl(
    OpCommTransport& opCommTransport, const std::map<u32, bool>& remoteRankPortMap, bool isUseDefault, bool statusStop)
{
    for (auto& levelNSubCommTransport : opCommTransport) {
        for (auto& singleSubCommTransport : levelNSubCommTransport) {
            for (u32 i = 0; i < singleSubCommTransport.transportRequests.size(); i++) {
                u32 transportRemoteRank = singleSubCommTransport.transportRequests[i].remoteUserRank;
                bool isValid = singleSubCommTransport.transportRequests[i].isValid;
                // isPortSatisfy表示当前对端使用的主备网口是否和changeLinkInfo一致
                bool isPortSatisfy
                    = (remoteRankPortMap.find(transportRemoteRank) != remoteRankPortMap.end()
                       && remoteRankPortMap.find(transportRemoteRank)->second == isUseDefault);
                HCCL_INFO(
                    "[SetTransportStatus]remoteRank[%u], isUseDefault[%d], isValid[%d], isPortSatisfy[%d]",
                    transportRemoteRank, isUseDefault, isValid, isPortSatisfy);
                if (isValid && isPortSatisfy) {
                    CHK_RET(SetSignalTransport(singleSubCommTransport, i, statusStop));
                }
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ReAllocTransports(const std::string& tag, const std::string& newTag)
{
    HcclResult ret = HCCL_SUCCESS;
    HCCL_INFO("[%s] alloc tag[%s] transports", __func__, newTag.c_str());

    AlgResourceResponse& algResResponse = resMap_[newTag];
    DeviceMem expMem = cclBufferManager_.GetCommCCLBuffer();

    TransportIOMem transMem{
        algResResponse.cclInputMem,
        algResResponse.cclOutputMem,
        algResResponse.paramInputMem,
        algResResponse.paramOutputMem,
        algResResponse.scratchMem,
        algResResponse.aivInputMem,
        algResResponse.aivOutputMem,
        expMem,
        DeviceMem(),
        {}};

    {
        // Transport资源 重建链, 一定是AICPU展开，所以 isAicpuModeEn=true
        StateGuard<HcclCommunicator, HcclCommState> guard(this, HcclCommState::BUILDING);
        ret = transportManager_->Alloc(tag, transMem, algResResponse.opTransportResponse, true);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s]Realloc transports failed, tag[%s]", __func__, newTag.c_str()), ret);
    }

    if (IsEnableBackupLink()) {
        HCCL_INFO("[%s] alloc tag[%s] backup transports", __func__, newTag.c_str());
        // 超节点 && level2支持重执行 && Aicpu：备用Transport资源 重建链
        StateGuard<HcclCommunicator, HcclCommState> guard(this, HcclCommState::BUILDING);
        ret = transportManager_->Alloc(tag, transMem, algResResponse.opTransportResponseBackUp, true, true);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s]Alloc backup transports failed, tag[%s]", __func__, newTag.c_str()),
            ret);
    }
    SaveLinkRes(algResResponse.opTransportResponse);
    SaveLinkRes(algResResponse.opTransportResponseBackUp);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Stop()
{
    HcclUs startut = TIME_NOW();
    isSuspending = true;
    HCCL_DEBUG("HcclCommunicator Stop begin.");
    for (auto& it : tagCommInfo_) {
        CHK_RET(TraverseLinkVector(it.second.commLevel1, true));
        CHK_RET(TraverseLinkVector(it.second.commLevel0, true));
        CHK_RET(TraverseLinkVector(it.second.commLevel2, true));
        CHK_RET(TraverseLinkVector(it.second.commP2P, true));
        if (it.second.commIntraServer) {
            const std::vector<LINK>& ret = it.second.commIntraServer->TransportInfo();
            CHK_RET(MigrateLinkVectorToStopOrResume(ret, true));
        }
    }
    CHK_RET(TraverseAlgResourceResponse(true));
    HcclUs endut = TIME_NOW();
    HCCL_RUN_INFO("HcclCommunicator::Stop, Stop take time:[%lld]us", DURATION_US(endut - startut).count());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::HostMC2EnvResume()
{
    if (GetAicpuUnfoldFlag() || GetAicpuCommEngine()) {
        HCCL_DEBUG("[NsRecovery]reset the suspending flag");
        KfcExecControl controlCmd;
        controlCmd.kfcCmd = KfcCommand::kNone;
        controlCmd.bgCmd = BackgroundCommand::kNone;
        controlCmd.suspendingStatus = HcclComSuspendingFlag::isResume;
        CHK_RET(kfcControlTransferH2D_->Put(0, sizeof(KfcExecControl), reinterpret_cast<uint8_t*>(&controlCmd)));
        if (IsEnableCustom()) {
            CHK_RET(customControlTransferH2D_->Put(0, sizeof(KfcExecControl), reinterpret_cast<uint8_t*>(&controlCmd)));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ClearWinBuffer()
{
    DeviceMem winBuffer = cclBufferManager_.GetCommExpBuffer();
    if (winBuffer.ptr() != nullptr) {
        HCCL_INFO("HcclCommunicator::Resume, start to clear win buffer");
        CHK_RET(hrtMemSet(static_cast<u8*>(winBuffer.ptr()), EXP_BUFFER_SIZE, EXP_BUFFER_SIZE));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AivResume()
{
    if (GetExternalInputHcclAivMode()) {
        HCCL_DEBUG("AivResume begin.");

        CHK_RET(cclBufferManager_.ClearCommAIVbuffer());
        HCCL_INFO("[AIV][AivResumeClearSyncBuf] clear aiv buffer done");

        aivOpbaseTag_ = TAG_INIT_VALUE;
        aivOffloadTag_ = TAG_INIT_VALUE;
        HCCL_INFO("[AIV][AivResume] clear aiv tag done");
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Resume()
{
    HcclUs startut = TIME_NOW();
    bool isChangedLink = false;
    HCCL_RUN_INFO("HcclCommunicator Resume begin, group[%s].", identifier_.c_str());
    // 发生N秒快恢, 头尾计数可能不对，需要将头尾计数清零
    CHK_RET(ClearOpCounterMem());
    for (auto& it : tagCommInfo_) {
        CHK_RET(TraverseLinkVector(it.second.commLevel1, false));
        CHK_RET(TraverseLinkVector(it.second.commLevel0, false));
        CHK_RET(TraverseLinkVector(it.second.commLevel2, false));
        CHK_RET(TraverseLinkVector(it.second.commP2P, false));
        if (it.second.commIntraServer) {
            const std::vector<LINK>& ret = it.second.commIntraServer->TransportInfo();
            CHK_RET(MigrateLinkVectorToStopOrResume(ret, false));
        }
    }

    if (GetAicpuUnfoldFlag() || GetAicpuCommEngine()) {
        CHK_RET(CheckExitWaitResumeState(isChangedLink));
    }

    if (!isChangedLink) {
        CHK_RET(TraverseAlgResourceResponse(false));
    }
    HcclUs cleanNotifyStart = TIME_NOW();
    CHK_RET(hrtResourceClean());
    HcclUs cleanNotifyEnd = TIME_NOW();
    HCCL_RUN_INFO(
        "HcclCommunicator::Resume, hrtResourceClean notify take time:[%lld]us",
        DURATION_US(cleanNotifyEnd - cleanNotifyStart).count());
    CHK_RET(HostMC2EnvResume());
    CHK_RET(ClearWinBuffer());
    CHK_RET(AivResume());
    isSuspending = false;

    HcclUs endut = TIME_NOW();
    HCCL_RUN_INFO(
        "HcclCommunicator::Resume, Resume take time:[%lld]us, group[%s]", DURATION_US(endut - startut).count(),
        identifier_.c_str());

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CheckSuspendingStatus()
{
    if (isSuspending) {
        return HCCL_E_SUSPENDING;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SwitchNic(uint32_t nRanks, uint32_t* ranks, bool* useBackup)
{
    CHK_RET(SwitchNic(nRanks, ranks, useBackup, kfcControlTransferH2D_, kfcStatusTransferD2H_));
    if (IsEnableCustom()) {
        CHK_RET(SwitchNic(nRanks, ranks, useBackup, customControlTransferH2D_, customStatusTransferD2H_));
    }
    return HCCL_SUCCESS;
}

void HcclCommunicator::SaveLinkRes(const OpCommTransport& opTransportResponse)
{
    std::lock_guard<std::mutex> commLock(linkResMapMutex_);
    for (auto& opCommTransport : opTransportResponse) {
        for (auto& transports : opCommTransport) {
            for (u32 i = 0; i < transports.transportRequests.size(); i++) {
                if (transports.links[i] != nullptr
                    && transports.links[i]->GetTransportType() == TransportType::TRANS_TYPE_IBV_EXP) {
                    auto remoteRank = transports.transportRequests[i].remoteUserRank;
                    std::string localServerId
                        = rankInfoList_.size() > userRank_ ? rankInfoList_[userRank_].serverId : "";
                    s32 localDevicePhyId = rankInfoList_.size() > userRank_ ? rankInfoList_[userRank_].devicePhyId : -1;
                    std::string remoteServerId
                        = rankInfoList_.size() > remoteRank ? rankInfoList_[remoteRank].serverId : "";
                    s32 remoteDevicePhyId
                        = rankInfoList_.size() > remoteRank ? rankInfoList_[remoteRank].devicePhyId : -1;
                    LinkInfo linkInfo(
                        identifier_, userRank_, localServerId, localDevicePhyId, remoteRank, remoteServerId,
                        remoteDevicePhyId);
                    linkResMap_.emplace(transports.links[i].get(), linkInfo);
                }
            }
        }
    }
    return;
}

HcclResult
HcclCommunicator::GetTransportCqeErrors(const HcclNetDevCtx netDevCtx, std::vector<ErrCqeInfo>& infos, u32& num)
{
    if (netDevCtx == nullptr) {
        return HCCL_SUCCESS;
    }
    HcclIpAddress localIp;
    CHK_RET(HcclNetDevGetLocalIp(netDevCtx, localIp));

    u32 qpn = 0;
    std::vector<std::pair<Transport*, CqeInfo>> infolist;
    Transport::GetTransportErrorCqe(netDevCtx, infolist, num);
    std::lock_guard<std::mutex> commLock(linkResMapMutex_);
    for (auto& info : infolist) {
        auto iter = linkResMap_.find(info.first);
        if (iter != linkResMap_.end()) {
            CHK_RET((info.first)->GetTransportId(qpn));
            infos.push_back(ErrCqeInfo(info.second, iter->second, qpn));
        } else {
            LinkInfo linkInfo;
            linkInfo.localDevicePhyId = -1;
            linkInfo.remoteDevicePhyId = -1;
            infos.push_back(ErrCqeInfo(info.second, linkInfo, qpn));
            HCCL_RUN_WARNING("[GetTransportCqeErrors]Transport linkInfo was not saved, only print CqeInfo");
        }
    }
    num = infos.size();
    return HCCL_SUCCESS;
}

void HcclCommunicator::ClearOpTransportResponseLinks(OpCommTransport& opTransportResponse)
{
    for (auto& levelNSubCommTransport : opTransportResponse) {
        for (auto& singleSubCommTransport : levelNSubCommTransport) {
            u32 size = singleSubCommTransport.transportRequests.size();
            singleSubCommTransport.links.resize(size, nullptr);
            singleSubCommTransport.status.resize(size, TransportStatus::INIT);
            HCCL_INFO("[%s] size[%u], linksSize[%d]", __func__, size, singleSubCommTransport.links.size());
        }
    }
}

HcclResult HcclCommunicator::SetDevIbverbsData(
    CommBase* comm, bool isSupportNormalQP, u64 commBufferSize, void* commInPtr, void* commOutPtr)
{
    const u32 curRankId = comm->Rank();
    const u32 rankSize = comm->RankSize();

    CHK_RET(AllocAndClearHostMem(sizeof(TransportDeviceNormalData) * rankSize, transDevIbverbsDataMem_));
    TransportDeviceNormalData* transDevIbverbsData
        = static_cast<TransportDeviceNormalData*>(transDevIbverbsDataMem_->ptr());

    for (u32 i = 0; i < rankSize; i++) {
        auto& data = transDevIbverbsData[i];
        if (i != curRankId) {
            // 对端link的信息
            const auto transport = comm->GetTransportByRank(i);
            CHK_PTR_NULL(transport);
            void* bufferIn = nullptr;
            void* bufferOut = nullptr;
            u32 remoteInMemKey = 0;
            u32 remoteOutMemKey = 0;
            CHK_RET(transport->GetRemoteMem(UserMemType::INPUT_MEM, &bufferIn));
            CHK_RET(transport->GetRemoteMem(UserMemType::OUTPUT_MEM, &bufferOut));
            data.remoteInputMem.addr = reinterpret_cast<uint64_t>(bufferIn);
            data.remoteOutputMem.addr = reinterpret_cast<uint64_t>(bufferOut);
            CHK_RET(transport->GetRemoteMemSize(UserMemType::INPUT_MEM, data.remoteInputMem.size));
            CHK_RET(transport->GetRemoteMemSize(UserMemType::OUTPUT_MEM, data.remoteOutputMem.size));
            // IBV链路需要的资源
            if (transport->GetTransportType() == TransportType::TRANS_TYPE_IBV_EXP) {
                CHK_RET(transport->GetRemoteMemKey(UserMemType::INPUT_MEM, &remoteInMemKey));
                CHK_RET(transport->GetRemoteMemKey(UserMemType::OUTPUT_MEM, &remoteOutMemKey));
                data.remoteInputMem.key = remoteInMemKey;
                data.remoteOutputMem.key = remoteOutMemKey;
                CHK_RET(transport->GetLocalMemDetails(UserMemType::INPUT_MEM, data.localInputMem));
                CHK_RET(transport->GetLocalMemDetails(UserMemType::OUTPUT_MEM, data.localOutputMem));
                std::vector<HcclQpInfoV2> qpInfos;
                CHK_RET(transport->GetAiQpInfo(qpInfos));
                data.qpInfo = qpInfos[0];
            }
        } else {
            // 本rank的信息
            data.localInputMem.addr = reinterpret_cast<uint64_t>(commInPtr);
            data.localInputMem.size = commBufferSize;
            data.localOutputMem.addr = reinterpret_cast<uint64_t>(commOutPtr);
            data.localOutputMem.size = commBufferSize;
        }

        if (isSupportNormalQP) {
            data.qpMode = QPMode::NORMAL;
        }
        // Debugging info
        data.Print();
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetTransportLocalMem(
    const std::shared_ptr<Transport>& transport, UserMemType memType, MemDetails& detail) const
{
    if (transport->GetTransportType() == TransportType::TRANS_TYPE_IBV_EXP) {
        CHK_RET(transport->GetLocalMemDetails(memType, detail));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetTransportRemoteMem(
    const std::shared_ptr<Transport>& transport, UserMemType memType, MemDetails& detail) const
{
    void* addr = nullptr;
    CHK_RET(transport->GetRemoteMem(memType, &addr));

    detail.addr = reinterpret_cast<uint64_t>(addr);
    CHK_RET(transport->GetRemoteMemSize(memType, detail.size));

    if (transport->GetTransportType() == TransportType::TRANS_TYPE_IBV_EXP) {
        CHK_RET(transport->GetRemoteMemKey(memType, &detail.key));
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GenAiRMAInfo(CommBase* comm)
{
    CHK_PTR_NULL(aiRMAInfoMem_);
    HcclAiRMAInfo* aiRMAInfoPtr = reinterpret_cast<HcclAiRMAInfo*>(aiRMAInfoMem_->ptr());
    CHK_PTR_NULL(aiRMAInfoPtr);

    const std::string& tag = comm->Tag();
    aiRMAInfoPtr->curRankId = comm->Rank();
    aiRMAInfoPtr->rankNum = comm->RankSize();

    CHK_RET(GetAIVNormalQPInfo(comm, tag));

    u32 tmpQueueSize = aiRMAInfoPtr->rankNum * aiRMAInfoPtr->qpNum;
    u32 tmpMemSize = aiRMAInfoPtr->rankNum;
    u32 tmpMemDetailSize = aiRMAInfoPtr->rankNum * AiMemMaxNum;

    CHK_RET(AllocAndClearHostMem(sizeof(HcclAiRMAWQ) * tmpQueueSize, aiSqMem_));
    CHK_RET(AllocAndClearHostMem(sizeof(HcclAiRMACQ) * tmpQueueSize, aiScqMem_));
    CHK_RET(AllocAndClearHostMem(sizeof(HcclAiRMAWQ) * tmpQueueSize, aiRqMem_));
    CHK_RET(AllocAndClearHostMem(sizeof(HcclAiRMACQ) * tmpQueueSize, aiRcqMem_));
    CHK_RET(AllocAndClearHostMem(sizeof(HcclAiRMAMemInfo) * tmpMemSize, aiMemMem_));
    HcclAiRMAMemInfo* aiMemHost = reinterpret_cast<HcclAiRMAMemInfo*>(aiMemMem_->ptr());

    CHK_RET(AllocAndClearHostMem(sizeof(MemDetails) * tmpMemDetailSize, aiMemDetailsMem_));
    MemDetails* aiMemDetailsHost = reinterpret_cast<MemDetails*>(aiMemDetailsMem_->ptr());
    CHK_RET(DeviceMem::alloc(aiMemDetailsDev_, aiMemDetailsMem_->size()));
    u64 memBase = reinterpret_cast<uint64_t>(aiMemDetailsDev_.ptr());

    for (u32 i = 0; i < aiRMAInfoPtr->rankNum; i++) {
        MemDetails& remoteIn = aiMemDetailsHost[i * AiMemMaxNum + GetAiMemTypeVal(HcclAiRMAMemType::REMOTE_INPUT)];
        MemDetails& remoteOut = aiMemDetailsHost[i * AiMemMaxNum + GetAiMemTypeVal(HcclAiRMAMemType::REMOTE_OUTPUT)];
        MemDetails& localIn = aiMemDetailsHost[i * AiMemMaxNum + GetAiMemTypeVal(HcclAiRMAMemType::LOCAL_INPUT)];
        MemDetails& localOut = aiMemDetailsHost[i * AiMemMaxNum + GetAiMemTypeVal(HcclAiRMAMemType::LOCAL_OUTPUT)];

        if (i != aiRMAInfoPtr->curRankId) {
            // link rank info
            const auto transport = comm->GetTransportByRank(i);
            CHK_PTR_NULL(transport);
            CHK_RET(GetTransportRemoteMem(transport, UserMemType::INPUT_MEM, remoteIn));
            CHK_RET(GetTransportRemoteMem(transport, UserMemType::OUTPUT_MEM, remoteOut));
            CHK_RET(GetTransportLocalMem(transport, UserMemType::INPUT_MEM, localIn));
            CHK_RET(GetTransportLocalMem(transport, UserMemType::OUTPUT_MEM, localOut));

            if (transport->GetTransportType() == TransportType::TRANS_TYPE_IBV_EXP) {
                CHK_RET(GenIbvAiRMAInfo(i, transport, tag, aiRMAInfoPtr));
            }
        } else {
            void* commInPtr = nullptr;
            void* commOutPtr = nullptr;
            u64 commInSize;
            u64 commOutSize;
            CHK_RET(cclBufferManager_.GetInCCLbuffer(commInPtr, commInSize));
            CHK_RET(cclBufferManager_.GetOutCCLbuffer(commOutPtr, commOutSize));
            localIn.addr = reinterpret_cast<uint64_t>(commInPtr);
            localIn.size = commInSize;
            localOut.addr = reinterpret_cast<uint64_t>(commOutPtr);
            localOut.size = commOutSize;
        }

        aiMemHost[i].memMaxNum = AiMemMaxNum;
        aiMemHost[i].sizeOfMemDetails = static_cast<u32>(sizeof(MemDetails));
        aiMemHost[i].memDetailPtr = memBase + i * AiMemMaxNum * aiMemHost[i].sizeOfMemDetails;

        HCCL_DEBUG(
            "[%s] tag[%s] curRankId[%u] dstRankId[%u] rankNum[%u] qpNum[%u] memMaxNum[%u] sizeOfMemDetails[%u] "
            "memDetailPtr[%p] remoteInAddr[%p] remoteInSize[%llu] remoteInKey[%u] remoteOutAddr[%p] "
            "remoteOutSize[%llu] remoteOutKey[%u] localInAddr[%p] localInSize[%llu] localInKey[%u] "
            "localOutAddr[%p] localOutSize[%llu] localOutKey[%u]",
            __func__, tag.c_str(), aiRMAInfoPtr->curRankId, i, aiRMAInfoPtr->rankNum, aiRMAInfoPtr->qpNum,
            aiMemHost[i].memMaxNum, aiMemHost[i].sizeOfMemDetails, aiMemHost[i].memDetailPtr, remoteIn.addr,
            remoteIn.size, remoteIn.key, remoteOut.addr, remoteOut.size, remoteOut.key, localIn.addr, localIn.size,
            localIn.key, localOut.addr, localOut.size, localOut.key);
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GenAiRMAInfoV2(const std::string& tag)
{
    CHK_PTR_NULL(rmaInfoMem_);
    HcclRMAInfo* rmaInfoPtr = reinterpret_cast<HcclRMAInfo*>(rmaInfoMem_->ptr());
    CHK_PTR_NULL(rmaInfoPtr);
    rmaInfoPtr->curRankId = userRank_;
    ;
    rmaInfoPtr->rankNum = userRankSize_;
    LevelNSubCommTransport& commTransport = resMap_[tag].opTransportResponse[COMM_COMBINE_ORDER];
    CHK_PRT_RET(
        commTransport.size() <= 0,
        HCCL_ERROR(
            "[%s] no LevelComm resource, please create comm first. "
            "tag[%s], curRankId[%u] rankNum[%u]",
            __func__, tag.c_str(), rmaInfoPtr->curRankId, rmaInfoPtr->rankNum),
        HCCL_E_INTERNAL);
    std::vector<LINK>& links = commTransport[0].links;
    CHK_PRT_RET(
        links.size() <= 0,
        HCCL_ERROR(
            "[%s] no transport resource, please create links first. "
            "tag[%s], curRankId[%u] rankNum[%u]",
            __func__, tag.c_str(), rmaInfoPtr->curRankId, rmaInfoPtr->rankNum),
        HCCL_E_INTERNAL);
    CHK_RET(GetAIVNormalQPInfoV2(links, tag));

    u32 tmpQueueSize = rmaInfoPtr->rankNum * rmaInfoPtr->qpNum;
    u32 tmpMemSize = rmaInfoPtr->rankNum;
    u32 tmpMemDetailSize = rmaInfoPtr->rankNum * AiMemMaxNum;

    CHK_RET(AllocAndClearHostMem(sizeof(HcclAiRMAWQ) * tmpQueueSize, aiSqMem_));
    CHK_RET(AllocAndClearHostMem(sizeof(HcclAiRMACQ) * tmpQueueSize, aiScqMem_));
    CHK_RET(AllocAndClearHostMem(sizeof(HcclAiRMAWQ) * tmpQueueSize, aiRqMem_));
    CHK_RET(AllocAndClearHostMem(sizeof(HcclAiRMACQ) * tmpQueueSize, aiRcqMem_));
    CHK_RET(AllocAndClearHostMem(sizeof(HcclAiRMAMemInfo) * tmpMemSize, aiMemMem_));
    HcclAiRMAMemInfo* aiMemHost = reinterpret_cast<HcclAiRMAMemInfo*>(aiMemMem_->ptr());

    CHK_RET(AllocAndClearHostMem(sizeof(MemDetails) * tmpMemDetailSize, aiMemDetailsMem_));
    MemDetails* aiMemDetailsHost = reinterpret_cast<MemDetails*>(aiMemDetailsMem_->ptr());

    aiMemDetailsDev_ = DeviceMem::alloc(aiMemDetailsMem_->size());
    u64 memBase = reinterpret_cast<uint64_t>(aiMemDetailsDev_.ptr());

    for (u32 i = 0; i < rmaInfoPtr->rankNum; i++) {
        MemDetails& remoteIn = aiMemDetailsHost[i * AiMemMaxNum + GetAiMemTypeVal(HcclAiRMAMemType::REMOTE_INPUT)];
        MemDetails& remoteOut = aiMemDetailsHost[i * AiMemMaxNum + GetAiMemTypeVal(HcclAiRMAMemType::REMOTE_OUTPUT)];
        MemDetails& localIn = aiMemDetailsHost[i * AiMemMaxNum + GetAiMemTypeVal(HcclAiRMAMemType::LOCAL_INPUT)];
        MemDetails& localOut = aiMemDetailsHost[i * AiMemMaxNum + GetAiMemTypeVal(HcclAiRMAMemType::LOCAL_OUTPUT)];

        if (i != rmaInfoPtr->curRankId) {
            // link rank info
            const auto transport = links[i];
            CHK_RET(GetTransportRemoteMem(transport, UserMemType::INPUT_MEM, remoteIn));
            CHK_RET(GetTransportRemoteMem(transport, UserMemType::OUTPUT_MEM, remoteOut));
            CHK_RET(GetTransportLocalMem(transport, UserMemType::INPUT_MEM, localIn));
            CHK_RET(GetTransportLocalMem(transport, UserMemType::OUTPUT_MEM, localOut));

            if (transport->GetTransportType() == TransportType::TRANS_TYPE_IBV_EXP) {
                CHK_RET(GenIbvAiRMAInfo(i, transport, tag, rmaInfoPtr));
            }
        } else {
            void* commInPtr = nullptr;
            void* commOutPtr = nullptr;
            u64 commInSize;
            u64 commOutSize;
            CHK_RET(cclBufferManager_.GetInCCLbuffer(commInPtr, commInSize));
            CHK_RET(cclBufferManager_.GetOutCCLbuffer(commOutPtr, commOutSize));
            localIn.addr = reinterpret_cast<uint64_t>(commInPtr);
            localIn.size = commInSize;
            localOut.addr = reinterpret_cast<uint64_t>(commOutPtr);
            localOut.size = commOutSize;
        }

        aiMemHost[i].memMaxNum = AiMemMaxNum;
        aiMemHost[i].sizeOfMemDetails = static_cast<u32>(sizeof(MemDetails));
        aiMemHost[i].memDetailPtr = memBase + i * AiMemMaxNum * aiMemHost[i].sizeOfMemDetails;

        HCCL_DEBUG(
            "[%s] tag[%s] curRankId[%u] dstRankId[%u] rankNum[%u] qpNum[%u] memMaxNum[%u] sizeOfMemDetails[%u] "
            "memDetailPtr[%p] remoteInAddr[%p] remoteInSize[%llu] remoteInKey[%u] remoteOutAddr[%p] "
            "remoteOutSize[%llu] remoteOutKey[%u] localInAddr[%p] localInSize[%llu] localInKey[%u] "
            "localOutAddr[%p] localOutSize[%llu] localOutKey[%u]",
            __func__, tag.c_str(), rmaInfoPtr->curRankId, i, rmaInfoPtr->rankNum, rmaInfoPtr->qpNum,
            aiMemHost[i].memMaxNum, aiMemHost[i].sizeOfMemDetails, aiMemHost[i].memDetailPtr, remoteIn.addr,
            remoteIn.size, remoteIn.key, remoteOut.addr, remoteOut.size, remoteOut.key, localIn.addr, localIn.size,
            localIn.key, localOut.addr, localOut.size, localOut.key);
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::H2DAiRMAInfo(const std::string& tag, rtStream_t aiCpuStream)
{
    CHK_PTR_NULL(aiRMAInfoMem_);
    HcclAiRMAInfo* aiRMAInfoPtr = reinterpret_cast<HcclAiRMAInfo*>(aiRMAInfoMem_->ptr());
    CHK_PTR_NULL(aiRMAInfoPtr);

    CHK_PTR_NULL(combinOparaMem_);
    HcclCombinOpParam* combinOparaPtr = reinterpret_cast<HcclCombinOpParam*>(combinOparaMem_->ptr());
    CHK_PTR_NULL(combinOparaPtr);

    aiRMAInfoPtr->sizeOfAiRMAWQ = static_cast<u32>(sizeof(HcclAiRMAWQ));
    aiRMAInfoPtr->sizeOfAiRMACQ = static_cast<u32>(sizeof(HcclAiRMACQ));
    aiRMAInfoPtr->sizeOfAiRMAMem = static_cast<u32>(sizeof(HcclAiRMAMemInfo));

    CHK_RET(DeviceMem::alloc(aiSqDev_, aiSqMem_->size()));
    aiRMAInfoPtr->sqPtr = aiSqDev_.ptr();
    CHK_RET(hrtMemAsyncCopy(
        aiSqDev_.ptr(), aiSqDev_.size(), aiSqMem_->ptr(), aiSqDev_.size(),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE, aiCpuStream));

    CHK_RET(DeviceMem::alloc(aiScqDev_, aiScqMem_->size()));
    aiRMAInfoPtr->scqPtr = aiScqDev_.ptr();
    CHK_RET(hrtMemAsyncCopy(
        aiScqDev_.ptr(), aiScqDev_.size(), aiScqMem_->ptr(), aiScqDev_.size(),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE, aiCpuStream));

    CHK_RET(DeviceMem::alloc(aiRqDev_, aiRqMem_->size()));
    aiRMAInfoPtr->rqPtr = aiRqDev_.ptr();
    CHK_RET(hrtMemAsyncCopy(
        aiRqDev_.ptr(), aiRqDev_.size(), aiRqMem_->ptr(), aiRqDev_.size(),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE, aiCpuStream));

    CHK_RET(DeviceMem::alloc(aiRcqDev_, aiRcqMem_->size()));
    aiRMAInfoPtr->rcqPtr = aiRcqDev_.ptr();
    CHK_RET(hrtMemAsyncCopy(
        aiRcqDev_.ptr(), aiRcqDev_.size(), aiRcqMem_->ptr(), aiRcqDev_.size(),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE, aiCpuStream));

    CHK_RET(hrtMemAsyncCopy(
        aiMemDetailsDev_.ptr(), aiMemDetailsDev_.size(), aiMemDetailsMem_->ptr(), aiMemDetailsDev_.size(),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE, aiCpuStream));

    CHK_RET(DeviceMem::alloc(aiMemDev_, aiMemMem_->size()));
    aiRMAInfoPtr->memPtr = aiMemDev_.ptr();
    CHK_RET(hrtMemAsyncCopy(
        aiMemDev_.ptr(), aiMemDev_.size(), aiMemMem_->ptr(), aiMemDev_.size(),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE, aiCpuStream));

    combinOparaPtr->sizeOfAiRMAInfo = static_cast<u64>(sizeof(HcclAiRMAInfo));
    CHK_RET(DeviceMem::alloc(aiRMAInfoDev_, combinOparaPtr->sizeOfAiRMAInfo));
    combinOparaPtr->aiRMAInfo = aiRMAInfoDev_.ptr();
    CHK_RET(hrtMemAsyncCopy(
        aiRMAInfoDev_.ptr(), aiRMAInfoDev_.size(), aiRMAInfoMem_->ptr(), aiRMAInfoDev_.size(),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE, aiCpuStream));

    HCCL_INFO(
        "[%s] tag[%s] curRankId[%u] rankNum[%u] qpNum[%u] aiRMAInfo[%p] sizeOfAiRMAInfo[%llu] "
        "sizeOfAiRMAWQ[%u] sizeOfAiRMACQ[%u] sizeOfAiRMAMem[%u] sqPtr[%p] sqSize[%llu] sqCount[%zu] "
        "scqPtr[%p] scqSize[%llu] scqCount[%zu] rqPtr[%p] rqSize[%llu] rqCount[%zu] rcqPtr[%p] "
        "rcqSize[%llu] rcqCount[%zu] memPtr[%p] memSize[%llu] memCount[%zu] memDetailCount[%zu]",
        __func__, tag.c_str(), aiRMAInfoPtr->curRankId, aiRMAInfoPtr->rankNum, aiRMAInfoPtr->qpNum,
        combinOparaPtr->aiRMAInfo, combinOparaPtr->sizeOfAiRMAInfo, aiRMAInfoPtr->sizeOfAiRMAWQ,
        aiRMAInfoPtr->sizeOfAiRMACQ, aiRMAInfoPtr->sizeOfAiRMAMem, aiRMAInfoPtr->sqPtr, aiSqDev_.size(),
        aiSqMem_->size(), aiRMAInfoPtr->scqPtr, aiScqDev_.size(), aiScqMem_->size(), aiRMAInfoPtr->rqPtr,
        aiRqDev_.size(), aiRqMem_->size(), aiRMAInfoPtr->rcqPtr, aiRcqDev_.size(), aiRcqMem_->size(),
        aiRMAInfoPtr->memPtr, aiMemDev_.size(), aiMemMem_->size(), aiMemDetailsMem_->size());

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::H2DAiRMAInfoV2(const std::string& tag, rtStream_t aiCpuStream)
{
    CHK_PTR_NULL(rmaInfoMem_);
    HcclRMAInfo* rmaInfoPtr = reinterpret_cast<HcclRMAInfo*>(rmaInfoMem_->ptr());
    CHK_PTR_NULL(rmaInfoPtr);

    CHK_PTR_NULL(combinOparaMem_);
    HcclCombinOpParam* combinOparaPtr = reinterpret_cast<HcclCombinOpParam*>(combinOparaMem_->ptr());
    CHK_PTR_NULL(combinOparaPtr);

    rmaInfoPtr->sizeOfRMAWQ = static_cast<u32>(sizeof(HcclAiRMAWQ));
    rmaInfoPtr->sizeOfRMACQ = static_cast<u32>(sizeof(HcclAiRMACQ));
    rmaInfoPtr->sizeOfRMAMem = static_cast<u32>(sizeof(HcclAiRMAMemInfo));

    CHK_RET(DeviceMem::alloc(aiSqDev_, aiSqMem_->size()));
    rmaInfoPtr->sqPtr = reinterpret_cast<uintptr_t>(aiSqDev_.ptr());
    CHK_RET(hrtMemAsyncCopy(
        aiSqDev_.ptr(), aiSqDev_.size(), aiSqMem_->ptr(), aiSqDev_.size(),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE, aiCpuStream));

    CHK_RET(DeviceMem::alloc(aiScqDev_, aiScqMem_->size()));
    rmaInfoPtr->scqPtr = reinterpret_cast<uintptr_t>(aiScqDev_.ptr());
    CHK_RET(hrtMemAsyncCopy(
        aiScqDev_.ptr(), aiScqDev_.size(), aiScqMem_->ptr(), aiScqDev_.size(),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE, aiCpuStream));

    CHK_RET(DeviceMem::alloc(aiRqDev_, aiRqMem_->size()));
    rmaInfoPtr->rqPtr = reinterpret_cast<uintptr_t>(aiRqDev_.ptr());
    CHK_RET(hrtMemAsyncCopy(
        aiRqDev_.ptr(), aiRqDev_.size(), aiRqMem_->ptr(), aiRqDev_.size(),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE, aiCpuStream));

    CHK_RET(DeviceMem::alloc(aiRcqDev_, aiRcqMem_->size()));
    rmaInfoPtr->rcqPtr = reinterpret_cast<uintptr_t>(aiRcqDev_.ptr());
    CHK_RET(hrtMemAsyncCopy(
        aiRcqDev_.ptr(), aiRcqDev_.size(), aiRcqMem_->ptr(), aiRcqDev_.size(),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE, aiCpuStream));

    CHK_RET(hrtMemAsyncCopy(
        aiMemDetailsDev_.ptr(), aiMemDetailsDev_.size(), aiMemDetailsMem_->ptr(), aiMemDetailsDev_.size(),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE, aiCpuStream));

    CHK_RET(DeviceMem::alloc(aiMemDev_, aiMemMem_->size()));
    rmaInfoPtr->memPtr = reinterpret_cast<uintptr_t>(aiMemDev_.ptr());
    CHK_RET(hrtMemAsyncCopy(
        aiMemDev_.ptr(), aiMemDev_.size(), aiMemMem_->ptr(), aiMemDev_.size(),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE, aiCpuStream));

    combinOparaPtr->sizeOfAiRMAInfo = static_cast<u64>(sizeof(HcclAiRMAInfo));
    CHK_RET(DeviceMem::alloc(aiRMAInfoDev_, combinOparaPtr->sizeOfAiRMAInfo));
    combinOparaPtr->aiRMAInfo = aiRMAInfoDev_.ptr();
    CHK_RET(hrtMemAsyncCopy(
        aiRMAInfoDev_.ptr(), aiRMAInfoDev_.size(), rmaInfoMem_->ptr(), aiRMAInfoDev_.size(),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE, aiCpuStream));

    HCCL_INFO(
        "[%s] tag[%s] curRankId[%u] rankNum[%u] qpNum[%u] aiRMAInfo[%p] sizeOfAiRMAInfo[%llu] "
        "sizeOfAiRMAWQ[%u] sizeOfAiRMACQ[%u] sizeOfAiRMAMem[%u] sqPtr[%p] sqSize[%llu] sqCount[%zu] "
        "scqPtr[%p] scqSize[%llu] scqCount[%zu] rqPtr[%p] rqSize[%llu] rqCount[%zu] rcqPtr[%p] "
        "rcqSize[%llu] rcqCount[%zu] memPtr[%p] memSize[%llu] memCount[%zu] memDetailCount[%zu]",
        __func__, tag.c_str(), rmaInfoPtr->curRankId, rmaInfoPtr->rankNum, rmaInfoPtr->qpNum, combinOparaPtr->aiRMAInfo,
        combinOparaPtr->sizeOfAiRMAInfo, rmaInfoPtr->sizeOfRMAWQ, rmaInfoPtr->sizeOfRMACQ, rmaInfoPtr->sizeOfRMAMem,
        rmaInfoPtr->sqPtr, aiSqDev_.size(), aiSqMem_->size(), rmaInfoPtr->scqPtr, aiScqDev_.size(), aiScqMem_->size(),
        rmaInfoPtr->rqPtr, aiRqDev_.size(), aiRqMem_->size(), rmaInfoPtr->rcqPtr, aiRcqDev_.size(), aiRcqMem_->size(),
        rmaInfoPtr->memPtr, aiMemDev_.size(), aiMemMem_->size(), aiMemDetailsMem_->size());

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetAIVNormalQPInfo(CommBase* comm, const std::string& tag)
{
    // 获取 Transport QP 数量
    CHK_PTR_NULL(aiRMAInfoMem_);
    HcclAiRMAInfo* aiRMAInfoPtr = reinterpret_cast<HcclAiRMAInfo*>(aiRMAInfoMem_->ptr());
    CHK_PTR_NULL(aiRMAInfoPtr);

    aiRMAInfoPtr->qpNum = HCCL_QPS_PER_CONNECTION_DEFAULT;
    for (u32 i = 0; i < aiRMAInfoPtr->rankNum; i++) {
        if (i != aiRMAInfoPtr->curRankId) {
            const auto transport = comm->GetTransportByRank(i);
            CHK_PTR_NULL(transport);
            if (transport->GetTransportType() == TransportType::TRANS_TYPE_IBV_EXP) {
                std::vector<HcclAiRMAQueueInfo> aiQpVec;
                CHK_RET(transport->GetAiRMAQueueInfo(aiQpVec));
                aiRMAInfoPtr->qpNum = static_cast<u32>(aiQpVec.size());
            }
        }
    }

    CHK_PRT_RET(
        aiRMAInfoPtr->qpNum <= 0,
        HCCL_ERROR(
            "[%s] invalid qpNum. tag[%s] curRankId[%u] rankNum[%u] qpNum[%u]", __func__, tag.c_str(),
            aiRMAInfoPtr->curRankId, aiRMAInfoPtr->rankNum, aiRMAInfoPtr->qpNum),
        HCCL_E_INTERNAL);

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetAIVNormalQPInfoV2(std::vector<LINK>& links, const std::string& tag)
{
    // 获取 Transport QP 数量
    CHK_PTR_NULL(rmaInfoMem_);
    HcclRMAInfo* rmaInfoPtr = reinterpret_cast<HcclRMAInfo*>(rmaInfoMem_->ptr());
    CHK_PTR_NULL(rmaInfoPtr);
    // 获取 Transport QP 数量（暂时只支持单QP）
    rmaInfoPtr->qpNum = HCCL_QPS_PER_CONNECTION_DEFAULT;
    for (u32 i = 0; i < links.size(); i++) {
        if (i != rmaInfoPtr->curRankId) {
            if (links[i]->GetTransportType() == TransportType::TRANS_TYPE_IBV_EXP) {
                std::vector<HcclAiRMAQueueInfo> aiQpVec;
                CHK_RET(links[i]->GetAiRMAQueueInfo(aiQpVec));
                rmaInfoPtr->qpNum = static_cast<u32>(aiQpVec.size());
                break;
            }
        }
    }

    CHK_PRT_RET(
        rmaInfoPtr->qpNum <= 0,
        HCCL_ERROR(
            "[%s] invalid qpNum. tag[%s] curRankId[%u] rankNum[%u] qpNum[%u]", __func__, tag.c_str(),
            rmaInfoPtr->curRankId, rmaInfoPtr->rankNum, rmaInfoPtr->qpNum),
        HCCL_E_INTERNAL);

    return HCCL_SUCCESS;
}

template <typename T>
HcclResult HcclCommunicator::GenIbvAiRMAInfo(
    u32 rankid, const std::shared_ptr<Transport>& transport, const std::string& tag, T* aiRMAInfoPtr)
{
    HCCL_INFO("[HcclCommunicator][%s] Start prepare.", __func__);
    std::vector<HcclAiRMAQueueInfo> aiQpVec;
    CHK_RET(transport->GetAiRMAQueueInfo(aiQpVec));

    CHK_PTR_NULL(aiRMAInfoPtr);
    CHK_PRT_RET(
        aiQpVec.size() != aiRMAInfoPtr->qpNum,
        HCCL_ERROR(
            "[%s] different qpNum. tag[%s] curRankId[%u] rankNum[%u] qpNum[%u] qpVecNum[%u]", __func__, tag.c_str(),
            aiRMAInfoPtr->curRankId, aiRMAInfoPtr->rankNum, aiRMAInfoPtr->qpNum, aiQpVec.size()),
        HCCL_E_INTERNAL);

    HcclAiRMAWQ* aiSqHost = reinterpret_cast<HcclAiRMAWQ*>(aiSqMem_->ptr());
    HcclAiRMACQ* aiScqHost = reinterpret_cast<HcclAiRMACQ*>(aiScqMem_->ptr());
    HcclAiRMAWQ* aiRqHost = reinterpret_cast<HcclAiRMAWQ*>(aiRqMem_->ptr());
    HcclAiRMACQ* aiRcqHost = reinterpret_cast<HcclAiRMACQ*>(aiRcqMem_->ptr());

    for (u32 j = 0; j < aiRMAInfoPtr->qpNum; j++) {
        const auto& aiQpInfo = aiQpVec[j];
        u32 qpIndex = rankid * aiRMAInfoPtr->qpNum + j;
        aiSqHost[qpIndex] = aiQpInfo.sq;
        aiScqHost[qpIndex] = aiQpInfo.scq;
        aiRqHost[qpIndex] = aiQpInfo.rq;
        aiRcqHost[qpIndex] = aiQpInfo.rcq;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetAivCoreLimit(u32 aivCoreLimit)
{
    numBlocks_ = aivCoreLimit;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetAlgInfo(const std::string& algConfig, const std::string& tag, std::string& algName)
{
    CHK_PRT_RET(
        (ALGCFG_TO_NAME.find(algConfig) == ALGCFG_TO_NAME.end()),
        HCCL_ERROR("[%s] invalid algConfig=[%s]", __func__, algConfig.c_str()), HCCL_E_PARA);

    algName = ALGCFG_TO_NAME[algConfig];
    HCCL_INFO("[%s] tag=[%s], algName=[%s]", __func__, tag.c_str(), algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetGroupMode(bool isGroup)
{
    isGroupMode_ = isGroup;
    CHK_RET(transportManager_->SetGroupMode(isGroup));
    return HCCL_SUCCESS;
}

bool HcclCommunicator::GetGroupMode() const { return isGroupMode_; }

HcclResult HcclCommunicator::GetCommUserMemSize(uint64_t& size)
{
    if (!isUserMemRegisted_ || userMemMap_.empty()) {
        HCCL_INFO("[HcclCommunicator][%s] get comm user mem size failed", __func__);
        return HCCL_E_NOT_FOUND;
    }
    size = userMemMap_.begin()->second->size();
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetAivQPInfoV2(std::vector<LINK>& links, const std::string& tag)
{
    HCCL_DEBUG("[HcclCommunicator][%s] Start prepare.", __func__);
    // 获取 Transport QP 数量
    CHK_PTR_NULL(aiRMAInfoMem_);
    HcclAiRMAInfo* aiRMAInfoPtr = reinterpret_cast<HcclAiRMAInfo*>(aiRMAInfoMem_->ptr());
    CHK_PTR_NULL(aiRMAInfoPtr);
    // 获取 Transport QP 数量（暂时只支持单QP）
    aiRMAInfoPtr->qpNum = HCCL_QPS_PER_CONNECTION_DEFAULT;
    for (u32 i = 0; i < links.size(); i++) {                             // server num
        if (i != (aiRMAInfoPtr->curRankId / meshAggregationRankSize_)) { // 判断server
            if (links[i]->GetTransportType() == TransportType::TRANS_TYPE_IBV_EXP) {
                std::vector<HcclAiRMAQueueInfo> aiQpVec;
                CHK_RET(links[i]->GetAiRMAQueueInfo(aiQpVec));
                aiRMAInfoPtr->qpNum = static_cast<u32>(aiQpVec.size());
                break;
            }
        }
    }
    CHK_PRT_RET(
        aiRMAInfoPtr->qpNum <= 0,
        HCCL_ERROR(
            "[%s] invalid qpNum. tag[%s] curRankId[%u] rankNum[%u] qpNum[%u]", __func__, tag.c_str(),
            aiRMAInfoPtr->curRankId, aiRMAInfoPtr->rankNum, aiRMAInfoPtr->qpNum),
        HCCL_E_INTERNAL);

    return HCCL_SUCCESS;
}

template HcclResult HcclCommunicator::GenIbvAiRMAInfo<HcclAiRMAInfo>(
    u32 rankid, const std::shared_ptr<Transport>& transport, const std::string& tag, HcclAiRMAInfo* aiRMAInfoPtr);
} // namespace hccl

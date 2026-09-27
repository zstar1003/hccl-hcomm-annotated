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

#include "externalinput_pub.h"
#include "env_config.h"
#include "p2p_mgmt_pub.h"
#include "opexecounter_pub.h"
#include "config.h"
#include "stream_active_manager.h"
#include "device_capacity.h"
#include "profiling_manager_pub.h"
#include "task_exception_handler_pub.h"
#include "rank_consistentcy_checker.h"
#include "hccl_aiv.h"
#include "adapter_rts_common.h"
#include "coll_alg_utils.h"
#include "../common/src/state_guard.h"
#include "detect_connect_anomalies.h"
#include "alg_profiling.h"
#include "mmpa_api.h"
#include "stream_utils.h"
#include "config_log.h"
#include "../nslbdp/hccl_nslbdp.h"
#include "../common/src/h2d_tlv/hccl_h2dtlv.h"
#include "hccl_one_sided_service.h"
#include "launch_device.h"
#include "hccl_communicator.h"
#include "launch_aicpu.h"
#include "order_launch/order_launch.h"
#include "comm_configer.h"
#include "snapshot_control.h"
#include "comm_topo_desc.h"
#include "hccl_net_dev_defs.h"
#include "aclgraph_callback.h"

using namespace std;
constexpr u32 MODULE_NUM_FOUR = 4;
constexpr u16 MAX_VALUE_U16 = 0xFFFF;

namespace hccl {
static std::mutex g_hcomInitMutex;
static std::atomic<u32> g_enableBackupLinkCommCount{0}; // 开启借轨的通信域计数
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
constexpr u32 TYPE_USER_MEM = 1;
constexpr u32 NON_BATCH_WRITE_MAX_STREAM_NUM = 19U;
constexpr u64 GIGABYTE_TO_BYTE = 1024ULL * 1024ULL * 1024ULL;
constexpr u8 AICPU_ORDERLAUNCH_INVALID_HCOM_MODE = 255; // 图模式下无附属从流，不进行按序下发
enum class TransferMemInfoIdx : u32 {
    TRANSFER_MEM_INFO_KEY_IDX = 0,
    TRANSFER_MEM_INFO_VALUE_IDX = 1,
    TRANSFER_MEM_INFO_RDMA_ENVELOPE_IDX = 2,
    TRANSFER_MEM_INFO_IDX_NUM = 3
};

enum class AicpuLocalNotifyIdx : u32 {
    // host-aicpu同步
    HOST_TO_AICPU_0 = 0,
    HOST_TO_AICPU_1 = 1,

    // 用于控制单算子模式各通信域kernel按序占核的notify
    ORDER_INDEX_OPBASE_0 = 2, // host_order流 record, kernel流 wait
    ORDER_INDEX_OPBASE_1 = 3, // aicpu_order流 record, host_order流 wait

    // 用于控制Aclgraph模式各通信域kernel按序占核的notify
    ORDER_INDEX_ACLGRAPH_0 = 4, // host_order流 record, kernel流 wait
    ORDER_INDEX_ACLGRAPH_1 = 5, // aicpu_order流 record, host_order流 wait

    // 用于控制图模式各通信域kernel按序占核的notify
    ORDER_INDEX_HCOM_0 = 6, // host_order流 record, kernel流 wait
    ORDER_INDEX_HCOM_1 = 7  // aicpu_order流 record, host_order流 wait
};

enum class AicpuLocalEventIdx : u32 {
    /**
     *@brief 用于控制Aclgraph模式按序下发控制流入图的event
     *@note
     *通信域绑定Context，而Stream是Context管理的资源，因此对应下在Stream上的event与通信域强相关，需要communicator管理
     **/
    ORDER_INDEX_ACLGRAPH_EVENT_0 = 0, // kernel流 record, host_order流 wait
    ORDER_INDEX_ACLGRAPH_EVENT_1 = 1, // host_order流 record, kernel流 wait
};

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
      multiSuperPodDiffDeviceNumMode_(false),
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
    zeroCopyAclGraph_.reset(new (std::nothrow) ZeroCopyAclGraph());
    if (zeroCopyAclGraph_ == nullptr) {
        HCCL_ERROR("new ZeroCopyAclGraph failed!");
    }
    commConfig_ = CommConfig();
    dpuManager_.reset(new (std::nothrow) DpuManager());
    if (dpuManager_ == nullptr) {
        HCCL_ERROR("new DpuManager failed!");
    }
}

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
    zeroCopyAclGraph_.reset(new (std::nothrow) ZeroCopyAclGraph());
    if (zeroCopyAclGraph_ == nullptr) {
        HCCL_ERROR("new ZeroCopyAclGraph failed!");
    }
    commConfig_ = commConfig;
    dpuManager_.reset(new (std::nothrow) DpuManager());
    if (dpuManager_ == nullptr) {
        HCCL_ERROR("new DpuManager failed!");
    }
}

HcclCommunicator::~HcclCommunicator()
{
    HCCL_DEBUG("Enter ~HcclCommunicator.");

    DeinitZeroCopyMemoryAgent(true);
    if (!isInvalidComm_) {
        (void)DestroyAicpuComm();
        (void)UnRegisterBackGroundThread();
    } else {
        HCCL_WARNING(
            "The comm[%s] is invalid in snapshot, rank[%u]. deviceLogicId[%u]. "
            "There is no aicpu comm in device, skip aicpu comm destroy in destructor.",
            identifier_.c_str(), userRank_, deviceLogicId_);
    }

    if (dpuManager_ != nullptr) {
        (void)dpuManager_->DeInitDpuKernel();
        dpuManager_ = nullptr;
    }

    UnRegisterToHeartBeat();
    DeleteOpInfoToHeartBeat();
    AlgWrap::GetInstance().UnregisterAlgCallBack(identifier_);
    DetectConnectionAnomalies::GetInstance(deviceLogicId_).Deinit();
    UnRegisterToCommConfiger();
    AclgraphCallback::GetInstance().CleanCaptureRes(this);

    if (zeroCopyAclGraph_ != nullptr) {
        zeroCopyAclGraph_ = nullptr;
    }

    if (implAlg_ != nullptr) {
        implAlg_ = nullptr;
    }

    for (auto& res : resMap_) {
        DestroyAlgResource(res.second);
    }

    if (releaseChannel_ != nullptr) {
        releaseChannel_();
    }

    if (opRetryManager_ != nullptr) {
        OpRetryManager::DeleteLinkInfoByIdentifier(deviceLogicId_, identifier_);
        opRetryManager_->UnRegisterOpRetryManager(identifier_);
        opRetryManager_ = nullptr;
    }

    if (IsEnableBackupLink()) {
        if (g_enableBackupLinkCommCount.load() == 0) {
            HCCL_ERROR("[Destroy] g_enableBackupLinkCommCount is 0");
        } else {
            g_enableBackupLinkCommCount--;
        }
    }

    resMap_.clear();
    deviceResOrigMem_.clear();
    hostResMap_.clear();
    tagCommInfo_.clear();
    tagWorkSpaceMem_.clear();
    tagStreamInfo_.clear();

    if (opRetryStreamPtr_ != nullptr) {
        opRetryStreamPtr_->clear();
        opRetryStreamPtr_ = nullptr;
    }

    OrderLaunch::GetInstance(deviceLogicId_).UnRegisterOrderLaunch(identifier_);
    for (u32 i = 0; i < AICPU_LOCAL_EVENT_SIZE; ++i) {
        if (localAicpuOpEvent_[i] != nullptr) {
            (void)hrtEventDestroy(localAicpuOpEvent_[i]);
            localAicpuOpEvent_[i] = nullptr;
        }
    }

    (void)UnRegistTaskExceptionHandler();
    for (auto streamId : aicpuStreamIds_) {
        UnregisterGetAicpuTaskExceptionCallBack(streamId, deviceLogicId_);
    }
    aicpuStreamIds_.clear();
    kfcControlTransferH2D_ = nullptr;
    kfcStatusTransferD2H_ = nullptr;
    customControlTransferH2D_ = nullptr;
    customStatusTransferD2H_ = nullptr;

    oneSideService_ = nullptr;
    if (isOneSidedServiceNetDevCtxInited) {
        DeInitOneSidedServiceNetDevCtx();
    }

    /* 网络资源销毁 */
    DestroyNetworkResources();
    notifyPool_ = nullptr;
    queueNotifyManager_ = nullptr;
    /* driver关联资源释放 */
    if (drvInit_) {
        if (DisablePreResource() != HCCL_SUCCESS) {
            HCCL_WARNING("driver resource is not released successfully");
        }
    }

    if (isExecuteProfilingInit_) {
        (void)DeinitProfiling();
    }

    if (OpExeCounter::GetInstance(deviceLogicId_).DeInitCounter() != HCCL_SUCCESS) {
        HCCL_WARNING("op exec counter resource free failed");
    }

    /* 销毁当前trace句柄 */
    if (opBaseAtraceInfo_ != nullptr) {
        opBaseAtraceInfo_->DeInit();
        opBaseAtraceInfo_ = nullptr;
    }

    ReleaseWorkSpacebuffer();
    ReleaseCommContextbuffer();

    for (u32 i = 0; i < AICPU_LOCAL_NOTIFY_SIZE; i++) {
        if (localAiCpuOpNotify_[i]) {
            HcclResult ret = localAiCpuOpNotify_[i]->Destroy();
            localAiCpuOpNotify_[i] = nullptr;
            if (ret != RT_ERROR_NONE) {
                HCCL_ERROR(
                    "[Destroy][AicpuNotify]errNo[0x%016llx] rt notify destroy fail, "
                    "aicpuOpNotify[%u] return[%d].",
                    HCCL_ERROR_CODE(HCCL_E_RUNTIME), i, ret);
            }
        }
    }

    while (!aiCpuNoIpcEvnet_.empty()) {
        rtEvent_t eventInfo = aiCpuNoIpcEvnet_.back();
        HcclResult ret = hrtEventDestroy(eventInfo);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[Destroy][AicpuNoIpcEvent]errNo[0x%016llx] rt event destroy fail, "
                "return[%d].",
                HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret);
        }
        aiCpuNoIpcEvnet_.pop_back();
    }

    UnloadAICPUKernel();
    UnloadCustomKernel();
    if (dispatcher_ != nullptr) {
        HcclDispatcherDestroy(dispatcher_);
        dispatcher_ = nullptr;
    }
    if (dispatcherCtx_ != nullptr) {
        DestroyDispatcherCtx(dispatcherCtx_, identifier_.c_str());
        dispatcherCtx_ = nullptr;
    }
    if (vDispatcher_ != nullptr) {
        HcclDispatcherDestroy(vDispatcher_);
        vDispatcher_ = nullptr;
    }
    if (deviceType_ == DevType::DEV_TYPE_910B || deviceType_ == DevType::DEV_TYPE_910_93) {
        UnRegisterFromSnapshot();
    }
    HCCL_DEBUG("~HcclCommunicator success.");
}

HcclResult HcclCommunicator::SaveTopoDesc(std::string& identifier)
{
    CommTopo topoType = CommTopo::COMM_TOPO_RESERVED;
    CHK_RET(GetInstTopoTypeByNetLayer(0, &topoType)); // layer 0

    CommTopoDesc::GetInstance().SaveRankSize(identifier, userRankSize_);
    CommTopoDesc::GetInstance().SaveL0TopoType(identifier, topoType);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Init(HcclCommParams& params, const RankTable_t& rankTable)
{
    CHK_RET(InitCommParams(params));
    CHK_RET(attrCollector_.Init(params, rankTable, commConfig_.GetConfigHcclAlgoMap()));
    CHK_RET(InitRankInfo(rankTable));
    CHK_RET(InitNetResource(rankTable));
    CHK_RET(InitDebug());
    CHK_RET(InitNotifyManager());
    CHK_RET(InitStreamManager());
    CHK_RET(InitProfiler());
    CHK_RET(InitDispatcher());
    CHK_RET(InitTransportManager());
    CHK_RET(InitCombinOpara());
    CHK_RET(RegisterRanksToDca());
    /*--------------加锁区--------------*/
    std::unique_lock<std::mutex> lock(g_hcomInitMutex);
    CHK_RET(RegistTaskExceptionHandler());

    attrCollector_.GenCollectiveId(params, rankTable);
    collectiveId_ = attrCollector_.GetCollectiveId();

    // 初始化参数(需要放置在ranktable解析之后)
    HcclResult ret = InitPara();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[HcclCommunicator][Init]errNo[0x%016llx] collectiveid[%s] parameter initialization failed",
            HCCL_ERROR_CODE(ret), params.id.internal),
        ret);
    lock.unlock();
    /*--------------加锁区--------------*/
    if (deviceType_ == DevType::DEV_TYPE_910B || deviceType_ == DevType::DEV_TYPE_910_93) {
        CHK_RET(RegisterKernel(deviceType_));
    }
    CHK_RET(LoadCustomKernel());
    CHK_RET(LoadAICPUKernel());
    CHK_RET(InitHDCommunicate());
    CHK_RET(InitOpRetry());
    CHK_RET(InitOpResPara());

    CHK_RET(InitOneSidedService(rankTable));
    CHK_RET(OrderLaunch::GetInstance(deviceLogicId_).RegisterOrderLaunch(identifier_));
    HcclTopoAttr topoAttr;
    attrCollector_.GetTopoAttr(topoAttr);
    CHK_RET(rankGraph_.Init(rankTable, topoAttr));
    CHK_RET(SaveTopoDesc(params.identifier));
    CHK_RET(RegisterToSnapshot());
    CHK_RET(InitSymmetricMemory());

    CHK_RET(InitMyRankConnectMode(params, rankTable));
    if (dpuManager_ != nullptr && myRankConnectMode_ != 0) { /* 当前只有host nic--device nic使用 */
        CHK_RET(dpuManager_->Init(identifier_, deviceLogicId_));
    }

    return HCCL_SUCCESS;
}

HcclResult
HcclCommunicator::Init(HcclCommParams& params, const std::vector<RankInfo>& rankList, WorldGroupInfo& groupCommonData)
{
    CHK_RET(InitCommParams(params));
    CHK_RET(attrCollector_.Init(params, rankList, groupCommonData, commConfig_.GetConfigHcclAlgoMap()));
    CHK_RET(InitRankInfoSubGroup(groupCommonData));
    CHK_RET(InitDebugSubGroup());
    CHK_RET(InitNotifyManager());
    CHK_RET(InitDispatcher());
    CHK_RET(InitStreamManager());
    CHK_RET(InitRaResource());
    CHK_RET(InitTransportManager());
    CHK_RET(InitHcclAlg());
    CHK_RET(LoadCustomKernel());
    CHK_RET(LoadAICPUKernel());
    CHK_RET(InitHDCommunicate());
    CHK_RET(InitOpRetry());
    CHK_RET(InitOpResPara());
    CHK_RET(RegisterRanksToDca());
    CHK_RET(OrderLaunch::GetInstance(deviceLogicId_).RegisterOrderLaunch(identifier_));
    HcclTopoAttr topoAttr;
    attrCollector_.GetTopoAttr(topoAttr);
    CHK_RET(rankGraph_.Init(topoAttr));
    CHK_RET(SaveTopoDesc(params.identifier));
    CHK_RET(RegisterToSnapshot());
    CHK_RET(InitSymmetricMemory());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::LoadAICPUKernel(void)
{
    if (binHandle_ == nullptr) {
        std::string jsonPath;
        CHK_RET(GetKernelFilePath(jsonPath));
        jsonPath += "ccl_kernel.json";
        HcclResult ret = LoadBinaryFromFile(jsonPath.c_str(), ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE, 0, binHandle_);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[LoadAICPUKernel]errNo[0x%016llx]load aicpu file fail, path[%s] optionType[%u] "
                "cpuKernelMode[%u].",
                ret, jsonPath.c_str(), ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE, 0),
            ret);
    }
    return HCCL_SUCCESS;
}

void HcclCommunicator::UnloadAICPUKernel(void)
{
    if (binHandle_ != nullptr) {
        aclError aclRet = aclrtBinaryUnLoad(binHandle_);
        if (aclRet != ACL_SUCCESS) {
            HCCL_ERROR(
                "[UnloadAICPUKernel]errNo[0x%016llx] unload binary from binHandle[%p] error.", aclRet, binHandle_);
        }
        binHandle_ = nullptr;
    }
    return;
}

HcclResult HcclCommunicator::LoadCustomKernel(void)
{
    // 加载自定义算子
    // 请勿删除，该函数为用户自定义算子时使用，应加载句柄
    // 读取customEnable环境变量，开启了就执行
    std::string jsonPath;
    CHK_RET(GetCustomKernelFilePath(jsonPath));
    jsonPath += "libaicpu_custom.json";
    CHK_RET(LoadCustomFile(jsonPath.c_str(), ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE, 1, binHandle_));
    return HCCL_SUCCESS;
}

void HcclCommunicator::UnloadCustomKernel(void)
{
    // 卸载自定义算子
    // 请勿删除，该函数为用户自定义算子时使用，应释放句柄：UnloadBinary(binCustomHandle_);
    return;
}

HcclResult HcclCommunicator::InitOneSidedService(const RankTable_t& rankTable)
{
    EXCEPTION_CATCH(
        (oneSideService_ = std::make_unique<HcclOneSidedService>(socketManager_, notifyPool_, commConfig_)),
        return HCCL_E_INTERNAL);
    hcclRankLinkInfo_.userRank = userRank_;
    hcclRankLinkInfo_.devicePhyId = devicePhyId_;

    if (devIpAddr_.empty()) {
        HCCL_ERROR("[%s] device ip is invalid, please set device ip first.", __func__);
        return HCCL_E_NOT_FOUND;
    }
    hcclRankLinkInfo_.ip = devIpAddr_[0];
    if (nicRanksPort_.size() <= userRank_) {
        HCCL_ERROR("[%s] userRank_[%u] port is invalid, please set port first", __func__, userRank_);
        return HCCL_E_NOT_FOUND;
    }
    hcclRankLinkInfo_.port = nicRanksPort_[userRank_];
    hcclRankLinkInfo_.socketsPerLink = 1;
    HCCL_DEBUG(
        "[%s]hcclRankLinkInfo_ userRank[%u], devicePhyId[%u], ip[%s], port[%u]", __func__, hcclRankLinkInfo_.userRank,
        hcclRankLinkInfo_.devicePhyId, hcclRankLinkInfo_.ip.GetReadableIP(), hcclRankLinkInfo_.port);
    CHK_RET(oneSideService_->Config(
        dispatcher_, hcclRankLinkInfo_, &rankTable, identifier_, isStandardCard_, enableP2PRankIds_));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitOneSidedServiceNetDevCtx(u32 remoteRankId)
{
    if (nicDeployment_ != NICDeployment::NIC_DEPLOYMENT_DEVICE) {
        // 单边操作当前只支持Device网卡，不支持host
        HCCL_ERROR(
            "[%s]nicDeployment_[%d], userRankSize_[%u], do not support oneSidedService.", __func__, nicDeployment_,
            userRankSize_);
        return HCCL_E_INTERNAL;
    }

    std::string localServerId = serverId_;
    std::string localSuperPodId = superPodId_;
    std::string remoteServerId = rankInfoList_.at(remoteRankId).serverId;
    std::string remoteSuperPodId = rankInfoList_.at(remoteRankId).superPodId;
    u32 intraRoceSwitch = GetExternalInputIntraRoceSwitch();
    bool useRdma = false;
    if (intraRoceSwitch || (!useSuperPodMode_ && localServerId != remoteServerId)
        || (localSuperPodId != remoteSuperPodId)) {
        // 1. 初始化网口
        CHK_RET(InitNic());
        isOneSidedServiceNicInited = true;

        // 2. 单边操作SetNetDevCtx, RDMA
        if (netDevCtxMap_.find(devIpAddr_[0]) == netDevCtxMap_.end()) {
            HCCL_ERROR("[%s] nicDeployment_[%d], device nic init fail, please check", __func__, nicDeployment_);
            return HCCL_E_NOT_FOUND;
        }
        useRdma = true;
        oneSideService_->SetNetDevCtx(netDevCtxMap_[devIpAddr_[0]], useRdma);
        HCCL_INFO("[%s]init device Nic for oneSidedService success.", __func__);
    } else {
        // 单边操作SetNetDevCtx, IPC
        oneSideService_->SetNetDevCtx(netDevCtxMap_[localVnicIp_], useRdma);
        HCCL_INFO("[%s]init vNic for oneSidedService success.", __func__);
    }
    isOneSidedServiceNetDevCtxInited = true;
    HCCL_DEBUG("[%s]nicDeployment_[%d], intraRoceSwitch[%u]", __func__, nicDeployment_, intraRoceSwitch);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::DeInitOneSidedServiceNetDevCtx()
{
    if (nicDeployment_ != NICDeployment::NIC_DEPLOYMENT_DEVICE) {
        // 单边操作当前只支持Device网卡，不支持host
        HCCL_ERROR(
            "[%s]nicDeployment_[%d], userRankSize_[%u], do not support oneSidedService.", __func__, nicDeployment_,
            userRankSize_);
        return HCCL_E_INTERNAL;
    }
    if (isOneSidedServiceNicStartListen_) {
        socketManager_->DestroySockets();
        u32 port = GetLocalNicPort(NicType::DEVICE_NIC_TYPE);
        CHK_RET(socketManager_->ServerDeInit(onesidedServiceNicIpAddr_, port));
        isOneSidedServiceNicStartListen_ = false;
        HCCL_INFO("[HcclCommunicator][%s] DeInit socket server success.tag[%s].", __func__, identifier_.c_str());
    }
    u32 intraRoceSwitch = GetExternalInputIntraRoceSwitch();
    if (isOneSidedServiceNicInited) {
        // 1. close sockets
        if (raResourceInit_) {
            socketManager_->DestroySockets();
        }
        // 2. 去初始化网口
        CHK_RET(DeinitNic());
        isOneSidedServiceNicInited = false;
        HCCL_INFO("[%s]Deinit device Nic for oneSidedService success.", __func__);
    }
    isOneSidedServiceNetDevCtxInited = false;
    HCCL_DEBUG("[%s]nicDeployment_[%d], intraRoceSwitch[%u]", __func__, nicDeployment_, intraRoceSwitch);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetOneSidedService(IHcclOneSidedService** service)
{
    *service = oneSideService_.get();
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::OneSidedServiceStartListen(NicType nicType, HcclNetDevCtx netDevCtx)
{
    HCCL_INFO("[HcclCommunicator][%s] Start prepare netDevCtx.", __func__);
    u32 port = GetLocalNicPort(nicType);
    CHK_RET(socketManager_->ServerInit(netDevCtx, port));
    if (nicType == NicType::DEVICE_NIC_TYPE) {
        CHK_RET(HcclNetDevGetLocalIp(netDevCtx, onesidedServiceNicIpAddr_));
        isOneSidedServiceNicStartListen_ = true;
    }
    isOneSidedServiceNetDevCtxInited = true;
    HCCL_INFO("[HcclCommunicator][%s] netDevCtx[%p] port[%u] server init success.", __func__, netDevCtx, port);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetOneSidedServiceDevIpAndPort(NicType nicType, HcclIpAddress& ipAddress, u32& port)
{
    if (nicDeployment_ != NICDeployment::NIC_DEPLOYMENT_DEVICE) {
        // 单边操作当前只支持Device网卡，不支持host
        HCCL_ERROR(
            "[%s]nicDeployment_[%d], userRankSize_[%u], do not support oneSidedService.", __func__, nicDeployment_,
            userRankSize_);
        return HCCL_E_INTERNAL;
    }
    port = GetLocalNicPort(nicType);
    if (nicType == NicType::VNIC_TYPE) {
        ipAddress = localVnicIp_;
        HCCL_INFO("[GetOneSidedServiceDevIpAddr] vnic ipAddress[%s] get success.", ipAddress.GetReadableAddress());
        return HCCL_SUCCESS;
    } else if (nicType == NicType::DEVICE_NIC_TYPE) {
        u32 nicNum = devIpAddr_.size();
        for (u32 i = 0; i < nicNum; i++) {
            if (devIpAddr_[i].IsInvalid()) {
                HCCL_INFO("[GetOneSidedServiceDevIpAddr]nic num[%u] deviceip is invalid, total nicNum[%u]", i, nicNum);
                continue;
            }
            ipAddress = devIpAddr_[i];
            HCCL_INFO("[GetOneSidedServiceDevIpAddr] nic ipAddress[%s] get success.", ipAddress.GetReadableAddress());
            return HCCL_SUCCESS;
        }
    }
    HCCL_ERROR("[HcclCommunicator][%s] ipAddress get fail. tag[%s]", __func__, identifier_.c_str());
    return HCCL_E_NOT_FOUND;
}

HcclResult HcclCommunicator::DeinitOneSidedService()
{
    if (oneSideService_ != nullptr) {
        CHK_RET(oneSideService_->DeInit());
    }
    return HCCL_SUCCESS;
}

bool HcclCommunicator::IsSupportSymmetricMemory(HcclCMDType opType, OpParam& opParam)
{
    CHK_PRT_RET(symmetricMemory_ == nullptr, HCCL_DEBUG("symmetricMemory_ is a nullptr"), false);
    HCCL_INFO(
        "[%s] aicpuUnfold[%d], workflowMode[%d], deviceType[%d], "
        "deviceNumPerAggregation_[%d], multiModuleDiffDeviceNumMode_[%d], tag[%s].",
        __func__, opParam.aicpuUnfoldMode, GetWorkflowMode(), deviceType_, deviceNumPerAggregation_,
        multiModuleDiffDeviceNumMode_, opParam.tag.c_str());

    // 目前只支持allgather, allreduce, reducescatter
    CHK_PRT_RET(
        opType != HcclCMDType::HCCL_CMD_ALLGATHER && opType != HcclCMDType::HCCL_CMD_ALLREDUCE
            && opType != HcclCMDType::HCCL_CMD_ALLTOALL && opType != HcclCMDType::HCCL_CMD_REDUCE_SCATTER,
        HCCL_INFO("[%s] opType[%d] not support symmetric memory", __func__, opType), false);

    // 只支持aicpu展开、单算子模式、910_93芯片
    CHK_PRT_RET(
        !opParam.aicpuUnfoldMode,
        HCCL_INFO("[%s] aicpuUnfold:%d not support symmetric memory", __func__, opParam.aicpuUnfoldMode), false);
    CHK_PRT_RET(
        GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE,
        HCCL_INFO("[%s] workflowMode:%d not support symmetric memory", __func__, GetWorkflowMode()), false);
    CHK_PRT_RET(
        deviceType_ != DevType::DEV_TYPE_910_93,
        HCCL_INFO("[%s] deviceType:%d not support symmetric memory", __func__, deviceType_), false);
    CHK_PRT_RET(
        superPodNum_ == 1 && serverNum_ > 1 && GetExternalInputInterHccsDisable(),
        HCCL_INFO("[%s] multi-server uses roce, not support symmetric memory", __func__), false);

    // 判断拓扑逻辑是否支持symmetric memory
    // 每个节点只有一张卡或节点间非对称场景不支持对称内存
    CHK_PRT_RET(
        deviceNumPerAggregation_ == 1 || multiModuleDiffDeviceNumMode_,
        HCCL_INFO(
            "[%s] deviceNumPerAggregation[%u], multiModuleDiffDeviceNumMode_[%d] not support symmetric memory",
            __func__, deviceNumPerAggregation_, multiModuleDiffDeviceNumMode_),
        false);

    // 判断输入输出地址是否都注册为对称内存
    HcclResult ret = symmetricMemory_->FindSymmetricWindow(
        opParam.inputPtr, opParam.inputSize, &opParam.inputSymWindow, &opParam.inputOffset);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS || opParam.inputSymWindow == nullptr,
        HCCL_INFO(
            "[%s] input[%p] size[%llu] is not support symmetric memory", __func__, opParam.inputPtr, opParam.inputSize),
        false);
    ret = symmetricMemory_->FindSymmetricWindow(
        opParam.outputPtr, opParam.outputSize, &opParam.outputSymWindow, &opParam.outputOffset);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS || opParam.outputSymWindow == nullptr,
        HCCL_INFO(
            "[%s] output[%p] size[%llu] is not support symmetric memory", __func__, opParam.outputPtr,
            opParam.outputSize),
        false);

    HCCL_INFO(
        "[HcclCommunicator][IsSupportSymmetricMemory] opParam.inputPtr[%p], inputOffset[%llu], inputSymWindow[%p]",
        opParam.inputPtr, opParam.inputOffset, opParam.inputSymWindow);
    HCCL_INFO(
        "[HcclCommunicator][IsSupportSymmetricMemory] opParam.outputPtr[%p], outputOffset[%llu], outputSymWindow[%p]",
        opParam.outputPtr, opParam.outputOffset, opParam.outputSymWindow);

    return true;
}

bool HcclCommunicator::IsSupportZeroCopy(const OpParam& opParam) const
{
    HCCL_INFO(
        "[%s] aicpuUnfold[%d], workflowMode[%d], deviceType[%d], "
        "deviceNumPerAggregation_[%d], multiModuleDiffDeviceNumMode_[%d], tag[%s].",
        __func__, opParam.aicpuUnfoldMode, GetWorkflowMode(), deviceType_, deviceNumPerAggregation_,
        multiModuleDiffDeviceNumMode_, opParam.tag.c_str());

    // 只支持aicpu展开、非重执行、单算子模式、910_93芯片
    CHK_PRT_RET(
        !opParam.aicpuUnfoldMode,
        HCCL_INFO("[%s] aicpuUnfold:%d not support zero copy", __func__, opParam.aicpuUnfoldMode), false);
    CHK_PRT_RET(
        GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE,
        HCCL_INFO("[%s] workflowMode:%d not support zero copy", __func__, GetWorkflowMode()), false);
    CHK_PRT_RET(
        deviceType_ != DevType::DEV_TYPE_910_93,
        HCCL_INFO("[%s] deviceType:%d not support zero copy", __func__, deviceType_), false);

    // 判断拓扑逻辑是否支持zero copy
    // 每个节点只有一张卡或节点间非对称场景不支持零拷贝
    CHK_PRT_RET(
        deviceNumPerAggregation_ == 1 || multiModuleDiffDeviceNumMode_,
        HCCL_INFO(
            "[%s] deviceNumPerAggregation[%u], multiModuleDiffDeviceNumMode_[%d] not support zero copy", __func__,
            deviceNumPerAggregation_, multiModuleDiffDeviceNumMode_),
        false);

    // 判断输入输出地址是否都是支持零Copy特性的
    CHK_PRT_RET(
        !ZeroCopyMemoryAgent::IsActivateCommMemoryAddr(opParam.inputPtr, opParam.inputSize),
        HCCL_INFO("[%s] input[%p] size[%llu] is not support zero copy", __func__, opParam.inputPtr, opParam.inputSize),
        false);
    CHK_PRT_RET(
        !ZeroCopyMemoryAgent::IsActivateCommMemoryAddr(opParam.outputPtr, opParam.outputSize),
        HCCL_INFO(
            "[%s] output[%p] size[%llu] is not support zero copy", __func__, opParam.outputPtr, opParam.outputSize),
        false);

    return true;
}

HcclResult HcclCommunicator::PrepareZeroCopy(const std::string& algName, const AlgDesc& algDesc, OpParam& opParam)
{
    if (!algDesc.isZeroCopy) {
        opParam.supportSymmetricMemory
            = false; //  当前对称内存与零拷贝算法绑定，对称内存使能关闭，确保aicpu侧不走对称内存分支
        HCCL_INFO("[HcclCommunicator][PrepareZeroCopy] algName[%s] not support zerocopy.", algName.c_str());
        return HCCL_SUCCESS;
    }

    if (opParam.supportSymmetricMemory) {
        HCCL_INFO(
            "[HcclCommunicator][PrepareZeroCopy] algName[%s] symmetric memory is enabled, not use zerocopy.",
            algName.c_str());
        return HCCL_SUCCESS;
    }
    // ARS特性不支持零拷贝
    if ((opParam.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER || opParam.opType == HcclCMDType::HCCL_CMD_ALLGATHER
         || opParam.opType == HcclCMDType::HCCL_CMD_ALLREDUCE)
        && deviceType_ == DevType::DEV_TYPE_910_93 && multiModuleDiffDeviceNumMode_
        && !multiSuperPodDiffDeviceNumMode_) {
        return HCCL_SUCCESS;
    }

    // 如果自己侧的共享内存没有申请，那么进行申请，并设置给transportManager，后续p2p建链时进行交换
    if (zeroCopyLocalBuffer_.ptr() == nullptr) {
        CHK_RET(DeviceMem::alloc(zeroCopyLocalBuffer_, ZERO_COPY_IPC_BUFFER_LENGTH));
        CHK_RET(hrtMemSet(zeroCopyLocalBuffer_.ptr(), zeroCopyLocalBuffer_.size(), zeroCopyLocalBuffer_.size()));
        zeroCopyIpcPtrs_[userRank_ % deviceNumPerAggregation_] = zeroCopyLocalBuffer_.ptr();

        HCCL_RUN_INFO(
            "[HCCL_TRACE][PrepareZeroCopy]Create ZeroCopy buffer success. buffer ptr[%p] size[%llu]",
            zeroCopyLocalBuffer_.ptr(), zeroCopyLocalBuffer_.size());
    }
    opParam.isZeroCopy = true;
    HCCL_INFO("[HcclCommunicator][PrepareZeroCopy] success to use zero copy feature");
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::UpdateZeroCopy(const OpParam& opParam, const AlgResourceResponse& algResource)
{
    if (!opParam.isZeroCopy) {
        return HCCL_SUCCESS;
    }

    // 遍历所有transport，找出里面的p2p链路对应的对端地址
    for (auto& singleSubCommTransport : algResource.opTransportResponse[COMM_LEVEL0]) {
        for (u64 i = 0; i < singleSubCommTransport.links.size(); ++i) {
            LINK link = singleSubCommTransport.links[i];
            if (link == nullptr || !singleSubCommTransport.transportRequests[i].isValid) {
                // 无效或者不支持的链路
                continue;
            }

            // 在使能零拷贝场景，我们使用控制面内存做OpenIpc交换，因此这里取出input即可
            u32 remoteRank = link->GetRemoteRank();

            void* remotePtr = nullptr;
            CHK_RET(link->GetRemoteMem(UserMemType::INPUT_MEM, &remotePtr));
            CHK_PRT_RET(
                remotePtr == nullptr, HCCL_ERROR("[BuildZeroCopyParam] invalid remotePtr[%p]", remotePtr), HCCL_E_PARA);
            CHK_PRT_RET(
                zeroCopyIpcPtrs_[remoteRank % deviceNumPerAggregation_] != nullptr
                    && zeroCopyIpcPtrs_[remoteRank % deviceNumPerAggregation_] != remotePtr,
                HCCL_ERROR(
                    "[BuildZeroCopyParam] zeroCopyIpcPtrs_[%u] is [%p] not equal to %p", remoteRank,
                    zeroCopyIpcPtrs_[remoteRank % deviceNumPerAggregation_], remotePtr),
                HCCL_E_PARA);

            zeroCopyIpcPtrs_[remoteRank % deviceNumPerAggregation_] = remotePtr;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildZeroCopyParam()
{
    // 不支持ZeroCopy
    if (zeroCopyLocalBuffer_.ptr() == nullptr) {
        return HCCL_SUCCESS;
    }

    for (u32 i = 0; i < AICPU_ZERO_COPY_MAX_DEVICE_NUM_A3; ++i) {
        opResPara_.zeroCopyIpcPtrs[i] = reinterpret_cast<u64>(zeroCopyIpcPtrs_[i]);
    }

    for (u32 i = 0; i < rankInfoList_.size(); ++i) {
        opResPara_.zeroCopyDevicePhyId[i % deviceNumPerAggregation_] = rankInfoList_[i].devicePhyId;
    }

    CHK_RET(ZeroCopyMemoryAgent::GetRingBufferAddr(
        opResPara_.zeroCopyRingBuffer, opResPara_.zeroCopyHeadPtr, opResPara_.zeroCopyTailPtr));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitCommParams(HcclCommParams& params)
{
    commHandle_ = params.commHandle;
    userRank_ = params.rank;
    realUserRank_ = params.userRank;
    userRankSize_ = params.totalRanks;
    deviceLogicId_ = params.logicDevId;
    profilingOption_ = params.profilingOption;
    profilingInitiated_ = params.profilingInitiated;
    deviceType_ = params.deviceType;
    commWorkMode_ = params.commWorkMode;
    hcomGroupNicInit_ = params.hcomGroupNicInit;
    identifier_ = params.identifier;
    collectiveId_ = params.id.internal;
    ranktableCrc_ = params.ranktableCrc;
    commConnections_ = params.commConnections;
    commPortConfig_ = params.commPortConfig;
    cclBuffName_ = params.cclBuffName;
    isShareComm_ = !cclBuffName_.empty();

    HCCL_DEBUG(
        " userRank_: %u realUserRank_: %u userRankSize_: %u deviceLogicId_: %u deviceType_: %u commWorkMode_: %u.",
        userRank_, realUserRank_, userRankSize_, deviceLogicId_, deviceType_, commWorkMode_);

    return HCCL_SUCCESS;
}

bool HcclCommunicator::Is310PDuoCard()
{
    return (
        Is310P3Common(isHaveCpuRank_, deviceType_)
        && (pairLinkInfo_[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)].size() == userRankSize_));
}

// 910B A+X 在RDMA未启用情况下，两模块间的device数目需要一致且两模块中使用的卡都在同一平面上
HcclResult HcclCommunicator::CheckSingleServerComm(const std::vector<RankInfo_t>& rankList) const
{
    if (serverNum_ == 1 && moduleNum_ == HCCL_MODULE_NUM_TWO && GetExternalInputIntraRoceSwitch() == 0
        && !isStandardCard_) {
        std::vector<u32> devIdList0;
        std::vector<u32> devIdList1;
        for (RankInfo_t rankInfo : rankList) {
            if (rankInfo.deviceInfo.devicePhyId == HOST_DEVICE_ID) {
                HCCL_ERROR("[Check][SingleServerComm]not support cpu rank");
                return HCCL_E_NOT_SUPPORT;
            }
            if (rankInfo.deviceInfo.devicePhyId < DEVICE_PER_MODULE) {
                devIdList0.push_back(rankInfo.deviceInfo.devicePhyId);
            } else {
                devIdList1.push_back(rankInfo.deviceInfo.devicePhyId);
            }
        }
        std::sort(devIdList0.begin(), devIdList0.end());
        std::sort(devIdList1.begin(), devIdList1.end());

        auto buildDeviceListStr = [](const std::vector<u32>& list) -> std::string {
            std::string result;
            for (const auto& id : list) {
                if (!result.empty()) {
                    result += " ";
                }
                result += std::to_string(id);
            }
            return result;
        };

        std::string devList0Str = buildDeviceListStr(devIdList0);
        std::string devList1Str = buildDeviceListStr(devIdList1);

        if (devIdList0.size() != devIdList1.size()) {
            std::string errormessage = "Device ID " + devList0Str + " in module 0 and device ID " + devList1Str
                                       + " in module 1 are not on the same plane.";
            RPT_INPUT_ERR(
                true, "EI0010", std::vector<std::string>({"reason"}), std::vector<std::string>({errormessage}));
            HCCL_ERROR(
                "[%s][%s]%s", LOG_KEYWORDS_INIT_CHANNEL.c_str(), LOG_KEYWORDS_TIMEOUT.c_str(), errormessage.c_str());
            return HCCL_E_NOT_SUPPORT;
        }
        for (size_t i = 0; i < devIdList0.size(); i++) {
            if (devIdList0[i] % DEVICE_PER_MODULE != devIdList1[i] % DEVICE_PER_MODULE) {
                std::string errormessage = "Device ID " + std::to_string(devIdList0[i]) + " in module 0 and device ID "
                                           + std::to_string(devIdList1[i]) + " in module 1 are not on the same plane.";
                RPT_INPUT_ERR(
                    true, "EI0010", std::vector<std::string>({"reason"}), std::vector<std::string>({errormessage}));
                HCCL_ERROR(
                    "[%s][%s]%s", LOG_KEYWORDS_INIT_CHANNEL.c_str(), LOG_KEYWORDS_TIMEOUT.c_str(),
                    errormessage.c_str());
                return HCCL_E_NOT_SUPPORT;
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CheckDataType(const HcclDataType dataType, bool needReduce)
{
    const vector<string> infoTitle({"ccl_op", "value", "parameter", "expect"});
    if (needReduce) {
        if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
            if ((dataType == HCCL_DATA_TYPE_INT64) || (dataType == HCCL_DATA_TYPE_BFP16)) {
                RPT_INPUT_ERR(
                    true, "EI0003", infoTitle,
                    vector<string>(
                        {"CheckDataType", GetDataTypeEnumStr(dataType), "dataType",
                         "HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_INT16, HCCL_DATA_TYPE_INT32, "
                         "HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_FP32"}));
                HCCL_ERROR(
                    "[%s][%s]errNo[0x%016llx] data type[%s] not supported, support range=[%s]",
                    LOG_KEYWORDS_TASK_EXEC.c_str(), LOG_KEYWORDS_INVALID_ARGUMENT.c_str(),
                    HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetDataTypeEnumStr(dataType).c_str(),
                    GetSupportDataType(needReduce).c_str());
                return HCCL_E_NOT_SUPPORT;
            }
        }
        if ((dataType == HCCL_DATA_TYPE_UINT64) || (dataType == HCCL_DATA_TYPE_UINT8)
            || (dataType == HCCL_DATA_TYPE_UINT16) || (dataType == HCCL_DATA_TYPE_UINT32)
            || (dataType == HCCL_DATA_TYPE_FP64) || (dataType == HCCL_DATA_TYPE_RESERVED)) {
            RPT_INPUT_ERR(
                true, "EI0003", infoTitle,
                vector<string>(
                    {"CheckDataType", GetDataTypeEnumStr(dataType), "dataType",
                     "HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_INT16, HCCL_DATA_TYPE_INT32, "
                     "HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_FP32"}));
            HCCL_ERROR(
                "[%s][%s]errNo[0x%016llx] data type[%s] not supported, support range=[%s]",
                LOG_KEYWORDS_TASK_EXEC.c_str(), LOG_KEYWORDS_INVALID_ARGUMENT.c_str(),
                HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetDataTypeEnumStr(dataType).c_str(),
                GetSupportDataType(needReduce).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
    } else {
        if ((dataType >= HCCL_DATA_TYPE_RESERVED) || (dataType < HCCL_DATA_TYPE_INT8)
            || (Is310P3Common(isHaveCpuRank_, deviceType_) && dataType == HCCL_DATA_TYPE_BFP16)) {
            RPT_INPUT_ERR(
                true, "EI0003", infoTitle,
                vector<string>(
                    {"CheckDataType", GetDataTypeEnumStr(dataType), "dataType",
                     "HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_INT16, HCCL_DATA_TYPE_INT32, "
                     "HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_FP32, HCCL_DATA_TYPE_UINT8, HCCL_DATA_TYPE_UINT16, "
                     "HCCL_DATA_TYPE_UINT32"}));
            HCCL_ERROR(
                "[%s][%s]errNo[0x%016llx] data type[%s] not supported, support range=[%s]",
                LOG_KEYWORDS_TASK_EXEC.c_str(), LOG_KEYWORDS_INVALID_ARGUMENT.c_str(),
                HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetDataTypeEnumStr(dataType).c_str(),
                GetSupportDataType(needReduce).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitZeroCopyMemoryAgent()
{
    CHK_PRT_RET(
        zeroCopyMemoryAgent_ != nullptr,
        HCCL_ERROR("[HcclCommunicator][InitZeroCopyMemoryAgent] ipc memory agent has init"), HCCL_E_INTERNAL);

    // 获取节点内的ranktable
    std::vector<std::vector<std::vector<RankInfo>>> commPlaneVector;
    CHK_SMART_PTR_NULL(implAlg_);
    implAlg_->GetCommPlaneVector(commPlaneVector);
    rankInfoListIntraServer_ = commPlaneVector[COMM_LEVEL0][COMM_INDEX_0];
    zeroCopyMemoryAgent_.reset(static_cast<ZeroCopyMemoryAgent*>(new (std::nothrow) ZeroCopyMemoryAgent(
        socketManager_, devicePhyId_, deviceLogicId_, localVnicIp_, rankInfoListIntraServer_, userRank_,
        useSuperPodMode_, identifier_)));
    CHK_PTR_NULL(zeroCopyMemoryAgent_);
    CHK_RET(zeroCopyMemoryAgent_->Init());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::DeinitZeroCopyMemoryAgent(bool inDestructor)
{
    if (zeroCopyMemoryAgent_ != nullptr) {
        if (!inDestructor && zeroCopyMemoryAgent_->IsResumed()) {
            // 析构函数释放场景不做barrier close
            CHK_RET(zeroCopyMemoryAgent_->BarrierClose());
        }
        CHK_RET(zeroCopyMemoryAgent_->DeInit());
        zeroCopyMemoryAgent_ = nullptr;
    }
    return HCCL_SUCCESS;
}

u8 HcclCommunicator::GetConfigAclGraphZeroCopyEnable() { return commConfig_.GetConfigAclGraphZeroCopyEnable(); }

HcclResult HcclCommunicator::ClearResMap(const std::string& tag, bool& findTag, bool aclGraphDestroyCbk)
{
    auto resIter = resMap_.find(tag);
    if (resIter != resMap_.end()) {
        findTag = true;
        DestroyAlgResource(resIter->second, aclGraphDestroyCbk);
        CHK_RET(StreamActiveManager::GetInstance(deviceLogicId_).StreamsUnactive(resIter->second.slaveStreams));
        resMap_.erase(resIter);
        HCCL_INFO("[%s] clear resMap[%s]", __func__, tag.c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ClearAclgraphHostLinks(const std::unordered_set<std::string>& tags)
{
    std::lock_guard<std::mutex> lock(commResMutex_);
    for (const auto& tag : tags) {
        auto it = tagsRequiringHostCleanup_.find(tag);
        if (it == tagsRequiringHostCleanup_.end()) {
            continue;
        }
        for (auto& rankIt : rankTagRemoteRes_) {
            auto tagIt = rankIt.second.find(tag);
            if (tagIt == rankIt.second.end()) {
                continue;
            }
            HccltagRemoteResV2* hostPtr = tagIt->second.tagRemoteResPtr;
            if (hostPtr != nullptr) {
                // 顺序敏感: ListCommonRemove 必须先于 erase shared_ptr,
                // 否则 hostPtr 成野指针, ListCommonRemove 访问 segfault。
                ListCommonRemove(&hostPtr->nextTagRes);
                for (auto vIt = hostMemVec_.begin(); vIt != hostMemVec_.end(); ++vIt) {
                    if (*vIt && (*vIt)->ptr() == hostPtr) {
                        size_t idx = static_cast<size_t>(vIt - hostMemVec_.begin());
                        if (idx < deviceMemVec_.size()) {
                            deviceMemVec_.erase(deviceMemVec_.begin() + idx);
                        }
                        hostMemVec_.erase(vIt);
                        break;
                    }
                }
            }
            rankIt.second.erase(tagIt);
        }
        ibverbsLocalNotify_.erase(tag);
        ibverbsRemoteNotify_.erase(tag);
        tagsRequiringHostCleanup_.erase(it);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ClearOpResource(const std::string& tag, bool aclGraphDestroyCbk)
{
    std::lock_guard<std::mutex> lock(commResMutex_);
    bool findTag = false;
    CHK_RET(ClearResMap(tag, findTag, aclGraphDestroyCbk));
    CHK_RET(ClearResMap(tag + "_host", findTag, aclGraphDestroyCbk));
    CHK_RET(ClearResMap(tag + "_device", findTag, aclGraphDestroyCbk));
    if (!findTag) {
        HCCL_WARNING("[%s] not find tag[%s] in resMap", __func__, tag.c_str());
    }

    tagCommInfo_.erase(tag);
    // stream解绑定
    auto iterStream = tagStreamInfo_.find(tag);
    if (iterStream != tagStreamInfo_.end()) {
        CHK_RET(StreamActiveManager::GetInstance(deviceLogicId_).StreamsUnactive(iterStream->second.ringStreams));
    }
    tagStreamInfo_.erase(tag);
    if (opRetryStreamPtr_ != nullptr) {
        opRetryStreamPtr_->erase(tag);
    }
    if (implAlg_ != nullptr) {
        CHK_RET(implAlg_->ClearOpResource(tag));
    }
    DestroyWorkspaceResource(tag);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CreateOpBasedResources(
    const HcclCMDType& opType, const std::string& tag, const HcomCollOpInfo& opInfo)
{
    return workSpaceRes_->CreateOpBasedResources(opType, tag, opInfo);
}

HcclResult HcclCommunicator::CreateRemoteOpBasedResources(u64 memSize, const std::string& tag)
{
    return workSpaceRes_->CreateRemoteOpBasedResources(memSize, tag);
}

HcclResult HcclCommunicator::DestroyRemoteOpBasedMem(const std::string& tag)
{
    return workSpaceRes_->DestroyRemoteOpBasedMem(tag);
}

bool HcclCommunicator::IsAtomicInit()
{
    if (!initializedFlag_.test_and_set()) {
        initializedFlag_.clear();
        return false;
    }
    return true;
}

bool HcclCommunicator::IsNeedNicInit()
{
    return (
        (nicInitialized_ == 0) && (!hcomGroupNicInit_) && (userRankSize_ > 1) && !isSingleMeshAggregation_
        && (superPodNum_ > 1 || !isUsedInterHccsMode_));
}

HcclResult HcclCommunicator::GetBandWidthPerNPU(u32 level, float& bandWidth)
{
    return hccl::GetBandWidthPerNPU(level, userRankSize_, deviceNumPerAggregation_, bandWidth);
}

HcclResult HcclCommunicator::GetDeviceNumPerAggregation(u32& deviceNumPerAggregation)
{
    deviceNumPerAggregation = deviceNumPerAggregation_;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitHccpChannel() { return hcclH2dTlv::GetInstance().InitHccpChannel(devicePhyId_); }

std::vector<RankInfo> HcclCommunicator::GetRankLists() { return rankInfoList_; }

HcclResult HcclCommunicator::CheckReduceDataType(const HcclDataType dataType, const HcclReduceOp op)
{
    if ((deviceType_ == DevType::DEV_TYPE_910B) || (deviceType_ == DevType::DEV_TYPE_910_93)) {
        if ((op == HCCL_REDUCE_PROD) && ((dataType == HCCL_DATA_TYPE_INT16) || (dataType == HCCL_DATA_TYPE_BFP16))) {
            RPT_INPUT_ERR(
                true, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
                std::vector<std::string>(
                    {"CheckReduceDataType", GetDataTypeEnumStr(dataType), "dataType",
                     "HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_INT32, HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_FP32"}));
            HCCL_ERROR(
                "[%s][%s]errNo[0x%016llx] device type[%d] does not support the data type[%s] and data "
                "type[%s] for Op[%s]",
                LOG_KEYWORDS_TASK_EXEC.c_str(), LOG_KEYWORDS_INVALID_ARGUMENT.c_str(),
                HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), deviceType_, GetDataTypeEnumStr(HCCL_DATA_TYPE_BFP16).c_str(),
                GetDataTypeEnumStr(HCCL_DATA_TYPE_INT16).c_str(), GetReduceOpEnumStr(op).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
    } else if (deviceType_ == DevType::DEV_TYPE_910) {
        if (dataType == HCCL_DATA_TYPE_INT16) {
            RPT_INPUT_ERR(
                true, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
                std::vector<std::string>(
                    {"CheckReduceDataType", GetDataTypeEnumStr(dataType), "dataType",
                     "HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_INT32, HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_FP32"}));
            HCCL_ERROR(
                "[%s][%s]errNo[0x%016llx] device type[%d] does not support the data type[%s]",
                LOG_KEYWORDS_TASK_EXEC.c_str(), LOG_KEYWORDS_INVALID_ARGUMENT.c_str(),
                HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), deviceType_, GetDataTypeEnumStr(dataType).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
    } else if (deviceType_ == DevType::DEV_TYPE_310P3) {
        if (dataType == HcclDataType::HCCL_DATA_TYPE_INT16 && op != HcclReduceOp::HCCL_REDUCE_SUM) {
            RPT_INPUT_ERR(
                true, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
                std::vector<std::string>({"CheckReduceDataType", GetReduceOpEnumStr(op), "op", "sum"}));
            HCCL_ERROR(
                "[%s][%s]errNo[0x%016llx] device type[%d] does not support the data type[%s] for Op[%s]",
                LOG_KEYWORDS_TASK_EXEC.c_str(), LOG_KEYWORDS_INVALID_ARGUMENT.c_str(),
                HCCL_ERROR_CODE(HCCL_E_NOT_SUPPORT), deviceType_,
                GetDataTypeEnumStr(HcclDataType::HCCL_DATA_TYPE_INT16).c_str(), GetReduceOpEnumStr(op).c_str());
            return HCCL_E_NOT_SUPPORT;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetAlgType(AlgType& algType, HcclCMDType opType)
{
    CHK_SMART_PTR_NULL(implAlg_);
    return implAlg_->GetAlgType(algType, opType);
}

HcclResult HcclCommunicator::GetCommParams(HcclCommParams& params)
{
    params.commHandle = commHandle_;
    params.rank = userRank_;
    params.userRank = realUserRank_;
    params.totalRanks = userRankSize_;
    params.logicDevId = deviceLogicId_;
    params.deviceType = deviceType_;
    params.hcomGroupNicInit = hcomGroupNicInit_;
    params.identifier = identifier_;
    params.ranktableCrc = ranktableCrc_;
    params.commConnections = commConnections_;
    params.commPortConfig.devPortSwitchOn = commPortConfig_.devPortSwitchOn;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetCommRankTable(RankTable_t& rankTable)
{
    for (auto& server : servRankInfo_) {
        for (auto& rank : server.second) {
            rankTable.rankList.emplace_back(rank);
        }
    }
    rankTable.serverNum = serverNum_;
    rankTable.superPodNum = superPodNum_;
    rankTable.nicDeploy = nicDeployment_;
    rankTable.version = attrCollector_.GetRankTableVersion();
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitPara()
{
    // 检查当前user_rank 对应的devid和rt查到的一致
    CHK_RET(attrCollector_.CheckLocalRankInfo());
    CHK_RET(attrCollector_.CalAndSetMeshAggRankSize());
    meshAggregationRankSize_ = attrCollector_.GetMeshAggregationRankSize();

    // 初始化计数任务
    CHK_RET(OpExeCounter::GetInstance(deviceLogicId_).InitCounter());

    notifyPool_.reset(new (std::nothrow) NotifyPool());
    CHK_SMART_PTR_NULL(notifyPool_);
    CHK_RET(notifyPool_->Init(devicePhyId_));

    callbackTask_.reset(new (std::nothrow) HcclCallbackTask(devicePhyId_, deviceLogicId_, dispatcher_, nicDeployment_));
    CHK_SMART_PTR_NULL(callbackTask_);

    workSpaceRes_.reset(new (std::nothrow) WorkspaceResource(devicePhyId_, deviceLogicId_, &cclBufferManager_));
    CHK_SMART_PTR_NULL(workSpaceRes_);

    CHK_RET(InitAlgResource());
    return HCCL_SUCCESS;
}

bool HcclCommunicator::IsStandardCard()
{
    if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
        HCCL_INFO("The current device only supports the StandardCard case.");
        return true;
    }

    return (
        (pairLinkInfo_[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)].size() == 0)
        && (pairLinkInfo_[static_cast<u32>(LinkTypeInServer::HCCS_SW_TYPE)].size() == 0)
        && (pairLinkInfo_[static_cast<u32>(LinkTypeInServer::SIO_TYPE)].size() == 0));
}

HcclResult HcclCommunicator::InitOpRetry()
{
    EXCEPTION_CATCH((opRetryStreamPtr_ = std::make_shared<HcclOpStreamRes>()), return HCCL_E_PTR);
    if (retryEnable_) {
        opRetryManager_.reset(new (std::nothrow) OpRetryManager());
        CHK_SMART_PTR_NULL(opRetryManager_);
        HcclIpAddress hostIp = !rankInfoList_.empty() ? rankInfoList_[0].hostIp : HcclIpAddress();
        u32 hostPort = !rankInfoList_.empty() ? rankInfoList_[0].hostPort : HCCL_INVALID_PORT;
        s32 hostDevId = !rankInfoList_.empty() ? rankInfoList_[0].devicePhyId : 0;
        HcclIpAddress localIp = rankInfoList_.size() > userRank_ ? rankInfoList_[userRank_].hostIp : HcclIpAddress();
        auto notifyResetCallback = [this](bool isSendRecv, s64 destRank) {
            return isSendRecv ? this->ResetNotifyForDestRank(destRank) : this->ResetNotify();
        };

        auto setTransportStatusCallback = [this](
                                              const HcclOpIdentifier& opId, bool statusStop,
                                              const std::map<u32, bool>& remoteRankPortMap,
                                              const std::map<u32, bool>& isChangeLinkMap, bool isChangeLinkFlag) {
            return this->SetTransportStatus(opId, statusStop, remoteRankPortMap, isChangeLinkMap, isChangeLinkFlag);
        };
        auto getSwitchRanksCallback
            = [this](
                  u32* distSwitchRankList, bool* distSwitchUseBackup, u32& distSwitchRankNum,
                  u8* distRemoteRankNicStatus, u32& distRankSize, bool& needCheckDefaultNic, bool& needCheckBackupNic) {
                  return this->GetSwitchRanks(
                      distSwitchRankList, distSwitchUseBackup, distSwitchRankNum, distRemoteRankNicStatus, distRankSize,
                      needCheckDefaultNic, needCheckBackupNic);
              };
        auto setTransportResumeStatusCallback = [this](
                                                    const std::map<u32, bool>& remoteRankPortMap,
                                                    const std::map<u32, bool>& isChangeLinkMap, bool isChangeLinkFlag,
                                                    bool statusStop) {
            return this->SetTransportResumeStatus(remoteRankPortMap, isChangeLinkMap, isChangeLinkFlag, statusStop);
        };
        HcclNetDevCtx netDevCtx = netDevCtxMap_[devIpAddr_[0]];
        HcclNetDevCtx backUpNetDevCtx = {};
        if (IsEnableBackupLink()) {
            g_enableBackupLinkCommCount++;
        }
        if (IsEnableBackupLink() && netDevCtxMap_.find(devBackupIpAddr_[0]) != netDevCtxMap_.end()) {
            backUpNetDevCtx = netDevCtxMap_[devBackupIpAddr_[0]];
        }
        OpRetryServerInfo serverInfo = {hostIp, hostPort, hostDevId};
        OpRetryAgentInfo agentInfo = {userRank_, deviceLogicId_, localIp, devIpAddr_[0], netDevCtx, backUpNetDevCtx};

        OpRetryAgentParam agentParam;
        agentParam.group = identifier_;
        agentParam.agentConnection = commConnections_.agentConnection;
        agentParam.h2dPtr = kfcControlTransferH2D_;
        agentParam.d2hPtr = kfcStatusTransferD2H_;
        agentParam.opStreamPtr = opRetryStreamPtr_;
        agentParam.notifyResetCallback = notifyResetCallback;
        agentParam.setTransportStatusCallback = setTransportStatusCallback;
        agentParam.setTransportResumeStatusCallback = setTransportResumeStatusCallback;
        agentParam.getSwitchRanksCallback = getSwitchRanksCallback;
        agentParam.isEnableBackupLink = IsEnableBackupLink();
        agentParam.isEnableSdmaRetry = commConfig_.GetConfigInterServerRetryEnable();
        agentParam.agentInfo = agentInfo;

        CHK_RET(opRetryManager_->RegisterOpRetryMachine(
            agentParam, userRankSize_, commConnections_.isRoot, commConnections_.serverConnections, serverInfo));
        HCCL_RUN_INFO(
            "[InitOpRetry] group[%s], isEnableBackupLink[%d], g_enableBackupLinkCommCount[%u]", identifier_.c_str(),
            IsEnableBackupLink(), g_enableBackupLinkCommCount.load());
    }
    return HCCL_SUCCESS;
}

bool HcclCommunicator::CompareWithServerId(const ServerInfo_t& left, const ServerInfo_t& right)
{
    return (strcmp(left.serverId.c_str(), right.serverId.c_str()) < 0);
}

bool HcclCommunicator::CompareWithNicName(const NetworkInfo_t& left, const NetworkInfo_t& right)
{
    return (strcmp(left.ethName.c_str(), right.ethName.c_str()) < 0);
}

bool HcclCommunicator::CompareWithUserRank(const RankInfo& left, const RankInfo& right)
{
    return left.userRank < right.userRank;
}

HcclResult HcclCommunicator::InitPreResource(const RankTable_t& rankTable)
{
    if (static_cast<s32>(devicePhyId_) == HOST_DEVICE_ID) {
        HCCL_ERROR("[Init][PreResource]not support cpu rank");
        return HCCL_E_NOT_SUPPORT;
    }
    (void)rankTable;
    // 判断是否为A3多docker场景，该场景需要使用sdid获取到的serverId判断是否属于同一server，若属于同一server则需要enablep2p
    if (deviceType_ == DevType::DEV_TYPE_910_93) {
        uint32_t localRankServerId = 0;
        uint32_t remoteRankServerId = 0;
        rtError_t ret = rtGetServerIDBySDID(rankInfoList_[userRank_].superDeviceId, &localRankServerId);
        CHK_PRT_RET(
            ret != RT_ERROR_NONE,
            HCCL_ERROR(
                "[Init][PreResource]rtGetServerIDBySDID failed sdid[0x%08x], serverID[%u], ret[%u]",
                rankInfoList_[userRank_].superDeviceId, localRankServerId, ret),
            HCCL_E_RUNTIME);
        for (size_t index = 0; index < rankInfoList_.size(); ++index) {
            const RankInfo& rankInfo = rankInfoList_[index];
            ret = rtGetServerIDBySDID(rankInfo.superDeviceId, &remoteRankServerId);
            CHK_PRT_RET(
                ret != RT_ERROR_NONE,
                HCCL_ERROR(
                    "[Init][PreResource]rtGetServerIDBySDID failed sdid[0x%08x], serverID[%u], ret[%u]",
                    rankInfo.superDeviceId, remoteRankServerId, ret),
                HCCL_E_RUNTIME);
            if (serverId_ != rankInfo.serverId && localRankServerId == remoteRankServerId) {
                enableP2PDevices_.push_back(rankInfo.devicePhyId);
                enableP2PRankIds_.insert(rankInfo.userRank);
                HCCL_INFO(
                    "[Init][PreResource]localRankID[%u]-localDevicePhyId[%u] needs to enablep2p with "
                    "remoteRankId[%u]-remoteDevicePhyId[%u], "
                    "and localServerId[%s], localServerIdBySDID[%u], remoteServerId[%s], remoteServerIdBySDID[%u]",
                    userRank_, rankInfoList_[userRank_].devicePhyId, rankInfo.userRank, rankInfo.devicePhyId,
                    serverId_.c_str(), localRankServerId, rankInfo.serverId.c_str(), remoteRankServerId);
            }
        }
    }
    // 查询本rank所在服务器
    auto iterServ = servRankInfo_.find(serverId_);

    bool check = (iterServ == servRankInfo_.end());
    CHK_PRT_RET(
        check, HCCL_ERROR("[Init][PreResource]can't find serverId[%s] in server map", serverId_.c_str()),
        HCCL_E_NOT_FOUND);

    for (u32 i = 0; i < iterServ->second.size(); i++) {
        if (iterServ->second[i].deviceInfo.devicePhyId != HOST_DEVICE_ID) {
            enableP2PDevices_.push_back(iterServ->second[i].deviceInfo.devicePhyId);
            enableP2PRankIds_.insert(iterServ->second[i].rankId);
            HCCL_INFO(
                "[Init][PreResource]In the current server[%s], localRank[%u]-localDevicePhyId[%u] needs to enableP2P "
                "with remoteRankId[%u]-remoteDevicePhyId[%u]",
                serverId_.c_str(), userRank_, rankInfoList_[userRank_].devicePhyId, iterServ->second[i].rankId,
                iterServ->second[i].deviceInfo.devicePhyId);
        }
    }
    HCCL_INFO(
        "[Init][PreResource]Current deviceType[%d], isStandardCard[%s]", deviceType_,
        isStandardCard_ ? "true" : "false");
    if (deviceType_ != DevType::DEV_TYPE_310P3 && !isStandardCard_) {
        HcclResult ret = P2PMgmtPub::EnableP2P(enableP2PDevices_);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[Init][PreResource]Enable P2P Failed, deviceLogicId[%d], ret[%u]", deviceLogicId_, ret), ret);
    }

    drvInit_ = true;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitTcpMode(const RankTable_t& rankTable) const
{
    bool isTcpMode = false;
    HCCL_INFO("[TcpMode][%u] [1:TCP, 2:RDMA, 3:RESERVED]", GetExternalInputProtocolType());
    if (GetExternalInputProtocolType() == ProtocolType::TCP) {
        isTcpMode = true;
    } else if (GetExternalInputProtocolType() == ProtocolType::RDMA) {
        // 通信协议选择RDMA
    } else {
        isTcpMode = (rankTable.nicDeploy == NICDeployment::NIC_DEPLOYMENT_HOST);
        HCCL_INFO("[Init][TcpMode]isTcpMode[%d] nicDeploy[%d]", isTcpMode, rankTable.nicDeploy);
    }
    SetTcpMode(isTcpMode);

    // 异构场景解析外部输入,放在SetTcpMode前防止Tcp用例走错分支，放在RecordProtocolType确保hdc模式下建链通信协议校验正确
    CHK_RET(InitExternalInputHeterog());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitMyRankConnectMode(const HcclCommParams& params, const RankTable_t& rankTable)
{
    if (deviceType_ != DevType::DEV_TYPE_910B) { /* 910B才支持host网卡特性 */
        myRankConnectMode_ = 0;
        return HCCL_SUCCESS;
    }
    uint32_t localRank = params.rank;
    if (rankTable.nicDeploy != NICDeployment::NIC_DEPLOYMENT_HOST) {
        myRankConnectMode_ = 0;
        return HCCL_SUCCESS;
    }
    for (auto it : rankTable.rankList) {
        if (it.rankId == localRank) {
            continue;
        }
        if (it.deviceInfo.nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE) {
            myRankConnectMode_ = 1;
            return HCCL_SUCCESS;
        }
    }
    myRankConnectMode_ = 0;
    return HCCL_SUCCESS;
}

uint32_t HcclCommunicator::GetConnectMode() const { return myRankConnectMode_; }

bool HcclCommunicator::IsEnableBackupLink()
{
    return deviceType_ == DevType::DEV_TYPE_910_93 && IsEnableRoce() && GetAicpuUnfoldConfig() && retryEnable_
           && commConfig_.GetConfigInterSuperPodRetryEnable() && !devBackupIpAddr_[0].IsInvalid()
           && rtsSupportChangeLink_ && !isDiffDeviceType_;
}

HcclResult HcclCommunicator::InitRaNetResource()
{
    CHK_RET(IsHostUseDevNic(isHostUseDevNic_));

    if (static_cast<s32>(devicePhyId_) != HOST_DEVICE_ID || nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_DEVICE) {
        CHK_RET(HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, devicePhyId_, deviceLogicId_, false));
        if (IsEnableBackupLink()) {
            // 超节点 && level2支持重执行 && Aicpu -> 初始化主备hccp资源(Pid粒度)
            CHK_RET(hrtGetPairDevicePhyId(devicePhyId_, deviceBackUpPhyId_));
            if (hrtGetDeviceIndexByPhyId(deviceBackUpPhyId_, deviceBackUpLogicId_) != HCCL_SUCCESS) {
                rtsSupportChangeLink_ = false;
                HCCL_ERROR(
                    "[%s]Runtime does not support changelink, deviceLogicId_[%d], devicePhyId_[%u], "
                    "deviceBackUpPhyId_[%u], deviceBackUpLogicId_[%u], nicDeployment_[%d], IsEnableBackupLink[%d]"
                    "rtsSupportChangeLink_[%d]",
                    __func__, deviceLogicId_, devicePhyId_, deviceBackUpPhyId_, deviceBackUpLogicId_, nicDeployment_,
                    IsEnableBackupLink(), rtsSupportChangeLink_);
                return HCCL_E_NOT_SUPPORT;
            } else {
                CHK_RET(HcclNetInit(
                    NICDeployment::NIC_DEPLOYMENT_DEVICE, deviceBackUpPhyId_, deviceBackUpLogicId_, false, true));
                HCCL_DEBUG(
                    "[%s]Default & backup NetworkManager Init, deviceLogicId[%d], devicePhyId[%u], "
                    "deviceBackUpPhyId_[%u], deviceBackUpLogicId_[%u], nicDeployment_[%d], IsEnableBackupLink[%d]",
                    __func__, deviceLogicId_, devicePhyId_, deviceBackUpPhyId_, deviceBackUpLogicId_, nicDeployment_,
                    IsEnableBackupLink());
            }
        }
    }

    if ((static_cast<s32>(devicePhyId_) != HOST_DEVICE_ID && isHaveCpuRank_)
        || (IsEnableRoce() && nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST)
        || (Is310PDevice() && nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST)) {
        u32 devicePhyID = (static_cast<s32>(devicePhyId_) == HOST_DEVICE_ID) ? 0 : devicePhyId_;
        u32 whiteListEn = (GetExternalInputHcclEnableWhitelist() == HCCL_WHITELIST_ON ? 1 : 0);
        CHK_RET(HcclNetInit(NICDeployment::NIC_DEPLOYMENT_HOST, devicePhyID, deviceLogicId_, whiteListEn));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitRaNic()
{
    CHK_RET(InitSocketManager());

    if (Is310PDevice()) {
        CHK_RET(InitNic());
    } else if (static_cast<s32>(devicePhyId_) != HOST_DEVICE_ID) {
        std::shared_ptr<HcclSocket>& devVnicSocket = commPortConfig_.devVnicListen.first;
        if (devVnicSocket) {
            localVnicIp_ = devVnicSocket->GetLocalIp();
            localVnicListenPort_ = devVnicSocket->GetLocalPort();
            HcclNetDevCtx& devVnicCtx = commPortConfig_.devVnicListen.second;
            CHK_PTR_NULL(devVnicCtx);
            netDevCtxMap_.insert(std::make_pair(localVnicIp_, devVnicCtx));
            CHK_RET(socketManager_->ServerInit(devVnicCtx, localVnicListenPort_));
            commPortConfig_.devVnicListen.second = nullptr;
            HCCL_INFO(
                "[HcclCommunicator][InitRaResource] init vnic with listened socket success, "
                "listened ip[%s] port[%u]",
                localVnicIp_.GetReadableAddress(), localVnicListenPort_);
        } else {
            localVnicListenPort_ = GetLocalNicPort(NicType::VNIC_TYPE);
            localVnicIp_ = HcclIpAddress(devicePhyId_);
            if (useSuperPodMode_) {
                CHK_RET(hrtRaGetSingleSocketVnicIpInfo(
                    devicePhyId_, DeviceIdType::DEVICE_ID_TYPE_SDID, superDeviceId_, localVnicIp_));
            } else {
                CHK_RET(hrtRaGetSingleSocketVnicIpInfo(
                    devicePhyId_, DeviceIdType::DEVICE_ID_TYPE_PHY_ID, devicePhyId_, localVnicIp_));
            }

            HcclNetDevCtx vnicPortCtx;
            CHK_RET(HcclNetOpenDev(&vnicPortCtx, NicType::VNIC_TYPE, devicePhyId_, deviceLogicId_, localVnicIp_));
            CHK_PTR_NULL(vnicPortCtx);
            netDevCtxMap_.insert(std::make_pair(localVnicIp_, vnicPortCtx));
            CHK_RET(socketManager_->ServerInit(vnicPortCtx, localVnicListenPort_));
            HCCL_INFO(
                "[HcclCommunicator][InitRaResource] init vnic with ip[%s] port[%u] success",
                localVnicIp_.GetReadableAddress(), localVnicListenPort_);
        }

        if (IsEnableRoce()) {
            CHK_RET(InitNic());
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitRaResource()
{
    /* 本通信域内只有1个device时，不需要初始化ra资源 */
    if (userRankSize_ <= 1) {
        HCCL_INFO("user rank size <= 1, ra is not needed for single device.");
        return HCCL_SUCCESS;
    }

    CHK_RET(InitRaNetResource());
    CHK_RET(InitRaNic());

    HCCL_INFO(
        "isUsedRdmaLevel0_[%u] nicNum[%u] hostIP[%s], nicDeployment[%d].", isUsedRdmaLevel0_, devIpAddr_.size(),
        hostIp_.GetReadableAddress(), nicDeployment_);

    raResourceInit_ = true; // 全局通信域会初始化，子通信域不会初始化，但是析构均会进入此逻辑，需要标记
    attrCollector_.GenSupportRdmaLite();
    CHK_RET(attrCollector_.GenSupportHccsAndSio());
    isSupportRdmaLite_ = attrCollector_.GetSupportRdmaLite();     // 是否支持Rdma Lite
    isSupportHccsAndSio_ = attrCollector_.GetSupportHccsAndSio(); // 是否支持Hccs Sio并发
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::DisablePreResource()
{
    // 查询本rank所在服务器
    auto iterServ = servRankInfo_.find(serverId_);
    bool check = (iterServ == servRankInfo_.end());
    CHK_PRT_RET(
        check, HCCL_ERROR("[Disable][PreResource]can't find serverId[%s] in server map", serverId_.c_str()),
        HCCL_E_NOT_FOUND);
    HcclResult ret = P2PMgmtPub::DisableP2P(enableP2PDevices_);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[Disable][PreResource]Disable all P2P Failed, deviceLogicId[%d], ret[%u]", deviceLogicId_, ret),
        ret);
    enableP2PDevices_.clear();
    enableP2PRankIds_.clear();
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetWorkspaceSubStreamNum(
    u64 count, HcclDataType dataType, HcclReduceOp op, const std::string& algName, u64& streamNum, u64 dataSize,
    bool ifAiv, HcclCMDType opType)
{
    AlgType algType;

    CHK_RET(GetAlgType(algType, opType));

    std::map<HcclCMDType, u64> gapMap
        = {{HcclCMDType::HCCL_CMD_REDUCE_SCATTER, HCCL_SMALL_COUNT_512_KB + HCCL_SMALL_COUNT_512_KB},
           {HcclCMDType::HCCL_CMD_ALLGATHER, HCCL_SMALL_COUNT_512_KB + HCCL_SMALL_COUNT_512_KB},
           {HcclCMDType::HCCL_CMD_ALLREDUCE, (HCCL_SMALL_COUNT_512_KB + HCCL_SMALL_COUNT_512_KB) * userRankSize_}};

    // 图模式下AIV展开，需要重新计算streamNum
    bool ifHcomWithAiv = ifAiv && (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
    HCCL_INFO(
        "[GetWorkspaceSubStreamNum] ifAiv[%d], workflowMode[%d], ifHcomWithAiv[%d]", ifAiv, GetWorkflowMode(),
        ifHcomWithAiv);
    if (ifHcomWithAiv && (deviceType_ == DevType::DEV_TYPE_910_93 || deviceType_ == DevType::DEV_TYPE_910B)) {
        HCCL_INFO("[GetWorkspaceSubStreamNum] Hcom AIV enabled, calculating the streamNum.");
        // A3 和 A2 公用以下的参数
        std::string newTag;
        std::unique_ptr<CollAlgOperator> algOperator = implAlg_->GetAlgOperator(opType);
        CHK_SMART_PTR_NULL(algOperator);
        OpParam param;
        param.reduceType = op;
        param.opType = opType;

        if (opType == HcclCMDType::HCCL_CMD_ALLTOALL || opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
            param.All2AllDataDes.sendType = dataType;
            param.All2AllDataDes.recvType = dataType;
            param.All2AllDataDes.sendCount = count;
        } else { // 不论 A2 还是 A3，AIV场景下的AllReduce/ReduceScatter还是A2上单独支持AIV的算子都用以下参数
            param.DataDes.count = count;
            param.DataDes.dataType = dataType;
        }
        AlgResourceRequest resRequest;
        CHK_RET(algOperator->CalcResRequest(algName, param, resRequest)); // 计算资源请求
        streamNum = resRequest.streamNum;
        HCCL_INFO(
            "[GetWorkspaceSubStreamNum] Hcom AIV enabled on DeviceType[%d], the streamNum is [%llu]", deviceType_,
            streamNum);
        return HCCL_SUCCESS;
    }

    if (serverNum_ == 1 && deviceType_ == DevType::DEV_TYPE_910_93 && opType == HcclCMDType::HCCL_CMD_ALLGATHER
        && dataSize <= gapMap[opType] && deviceNumPerAggregation_ > HCCL_DEVICE_NUM_TWO) {
        constexpr u64 streamForSmallCount = 3;
        streamNum = streamForSmallCount;
        HCCL_DEBUG("[GetWorkspaceSubStreamNum]DEV_TYPE_910_93 Single Server, the streamNum is %llu", streamNum);
        return HCCL_SUCCESS;
    }

    if (serverNum_ == 1 && deviceType_ == DevType::DEV_TYPE_910_93 && gapMap.find(opType) != gapMap.end()
        && dataSize <= gapMap[opType] && deviceNumPerAggregation_ > HCCL_DEVICE_NUM_TWO) {
        streamNum = deviceNumPerAggregation_ - HCCL_SUB_STREAM_NP_MESH;
        HCCL_DEBUG("[GetWorkspaceSubStreamNum]DEV_TYPE_910_93 Single Server, the streamNum is %llu", streamNum);
        return HCCL_SUCCESS;
    }

    if (deviceType_ == DevType::DEV_TYPE_910_93) {
        streamNum = HCCL_SUB_STREAM_NUM_DOUBLE_RING + RDMA_PLANE_NUM_IN_NPRING_DOUBLE;
        if (algType.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING) {
            streamNum += 1U; // semi_ring算法server内增加一条从流，需要2条从流
        }
        if (opType == HcclCMDType::HCCL_CMD_ALLTOALLV || opType == HcclCMDType::HCCL_CMD_ALLTOALL
            || opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
            streamNum = MAX_RANK_SIZE;
        }
        HCCL_DEBUG("[GetWorkspaceSubStreamNum]DEV_TYPE_910_93, the streamNum is %llu", streamNum);
        return HCCL_SUCCESS;
    }

    // AR RS 在开启Strict && 静态图、RSv 在开启确定性 && 静态图时, 需要重新计算StreamNum
    if (deviceType_ == DevType::DEV_TYPE_910B
        && (((opType == HcclCMDType::HCCL_CMD_ALLREDUCE || opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER)
             && GetExternalInputHcclDeterministicV2() == DETERMINISTIC_STRICT)
            || (opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V
                && GetExternalInputHcclDeterministicV2() != DETERMINISTIC_DISABLE)
            || (opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V && !isSingleMeshAggregation_
                && !multiModuleDiffDeviceNumMode_ // 多机&对称&图模式
                && GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB))) {
        // 图模式 A2规约保序场景，需要重新计算需要的streamNum
        streamNum = CalcStreamNumForReduceOrderPreservation();
        HCCL_DEBUG("[GetWorkspaceSubStreamNum]A2 reduce order preservation, the streamNum is %llu", streamNum);
        return HCCL_SUCCESS;
    }

    if (deviceType_ == DevType::DEV_TYPE_910B && opType == HcclCMDType::HCCL_CMD_ALLREDUCE
        && algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE) {
        streamNum = userRankSize_ / moduleNum_ - 1;
        HCCL_DEBUG("[GetWorkspaceSubStreamNum]A2 pipeline AllReduce, the streamNum is %llu", streamNum);
        return HCCL_SUCCESS;
    }

    // 设置AG和RS的图模式pipeline算法能够申请的streamNum
    if (deviceType_ == DevType::DEV_TYPE_910B &&                   // 910B
        algType.algoLevel0 == AlgTypeLevel0::ALG_LEVEL0_NP_MESH && // fullmesh
        (opType == HcclCMDType::HCCL_CMD_ALLGATHER || opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER
         || opType == HcclCMDType::HCCL_CMD_ALLGATHER_V)
        &&                                                            // AG或RS或AGV
        moduleNum_ > 1 && deviceNumPerAggregation_ > 1 &&             // 多机且每机器出多卡
        (moduleNum_ <= MODULE_NUM_FOUR ||                             // "机器数量小于等于4"
         dataSize > HCCL_SMALL_COUNT_1_MB ||                          // "大数据量"
         algType.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_PIPELINE)) { // "指定level1的算法为pipeline"
        streamNum = userRankSize_ / moduleNum_;
        HCCL_DEBUG("[GetWorkspaceSubStreamNum]DEV_TYPE_910B, the streamNum is %llu", streamNum);
        return HCCL_SUCCESS;
    }

    // 设置310P图模式 alltoall 的streamNum
    if (deviceType_ == DevType::DEV_TYPE_310P3
        && (opType == HcclCMDType::HCCL_CMD_ALLTOALL || opType == HcclCMDType::HCCL_CMD_ALLTOALLV
            || opType == HcclCMDType::HCCL_CMD_ALLTOALLVC)) {
        streamNum = userRankSize_ * RANK_SET_COMPUTE_CONST;
        HCCL_DEBUG("[GetWorkspaceSubStreamNum]DEV_TYPE_310P3, the streamNum is %llu", streamNum);
        return HCCL_SUCCESS;
    }

    // 根据所用算法，选择所需的从stream数目
    switch (algType.algoLevel0) {
        case AlgTypeLevel0::ALG_LEVEL0_NP_MESH:
            streamNum = userRankSize_ / moduleNum_ - HCCL_SUB_STREAM_NP_MESH;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_8P_RING:
            streamNum = HCCL_SUB_STREAM_NUM_8P_RING;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING:
            streamNum = HCCL_SUB_STREAM_NUM_DOUBLE_RING;
            break;
        case AlgTypeLevel0::ALG_LEVEL0_4P_MESH:
            streamNum = HCCL_SUB_STREAM_NUM_4P_MESH;
            break;
        default:
            streamNum = HCCL_SUB_STREAM_NUM_ZERO;
            break;
    }

    if (SatisfyIntraSuperPod(deviceType_, userRankSize_, useSuperPodMode_, superPodNum_)) {
        streamNum = std::max(static_cast<u64>(userRankSize_ - 1u), streamNum);
    } else if (FullmeshPairwiseSatisfyHighPerfAlltoallMeshCondition(
                   deviceType_, meshAggregationRankSize_, useSuperPodMode_,
                   commConfig_.GetConfigHcclAlgo(HcclCMDType::HCCL_CMD_ALLTOALL))) {
        streamNum = std::max(static_cast<u64>(meshAggregationRankSize_ - 1u), streamNum);
    }

    auto iter = HCCL_ALGO_LEVEL0_NAME_MAP.find(algType.algoLevel0);
    CHK_PRT_RET(
        iter == HCCL_ALGO_LEVEL0_NAME_MAP.end(),
        HCCL_ERROR("[GetWorkspaceSubStreamNum]level0: algType[%u] is invalid.", algType.algoLevel0), HCCL_E_INTERNAL);
    HCCL_DEBUG(
        "[GetWorkspaceSubStreamNum]hccl algorithm: In level0, using %s algo, the streamNum is %llu",
        iter->second.c_str(), streamNum);

    u64 sliceNum = CalculatePiplineSliceNum(opType, dataSize, algType, deviceType_, deviceNumPerServer_, serverNum_);
    // 图模式下数据量固定, 按照当前数据量判断是否支持pipeline切分并申请从流
    if (implAlg_ != nullptr && sliceNum >= MIN_PIPELINE_SLICE_NUM) {
        streamNum++;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::DestroyNetworkResources()
{
    transportManager_ = nullptr;
    if (raResourceInit_) {
        socketManager_->DestroySockets();
    }

    /* 本通信域内只有1个device时，不需要卸载ra资源 */
    if (userRankSize_ <= 1) {
        HCCL_INFO("user rank size <= 1, ra is not needed for single device");
        return HCCL_SUCCESS;
    }

    // nic的初始化独立调用，在此单独判断是否需要解初始化
    if (nicInitialized_ > 0) {
        CHK_RET(DeinitNic());
    }

    if (raResourceInit_ && (static_cast<s32>(devicePhyId_) != HOST_DEVICE_ID) && !Is310PDevice()) {
        CHK_RET(socketManager_->ServerDeInit(netDevCtxMap_[localVnicIp_], localVnicListenPort_));
        HcclNetCloseDev(netDevCtxMap_[localVnicIp_]);
        netDevCtxMap_.erase(localVnicIp_);
    }

    CHK_RET(ReleasePreemptSocket());

    if (raResourceInit_) {
        if (static_cast<s32>(devicePhyId_) != HOST_DEVICE_ID
            || nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_DEVICE) {
            if (IsEnableBackupLink()) {
                // 超节点 && level2支持重执行 && Aicpu -> 释放主备hccp资源
                CHK_RET(HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, devicePhyId_, deviceLogicId_));
                CHK_RET(HcclNetDeInit(
                    NICDeployment::NIC_DEPLOYMENT_DEVICE, deviceBackUpPhyId_, deviceBackUpLogicId_, true));
                HCCL_DEBUG(
                    "[%s]Default & backup HcclNetDeInit, deviceLogicId[%d], devicePhyId[%u], "
                    "deviceBackUpPhyId_[%u], deviceBackUpLogicId_[%u], nicDeployment_[%d], IsEnableBackupLink[%d]",
                    __func__, deviceLogicId_, devicePhyId_, deviceBackUpPhyId_, deviceBackUpLogicId_, nicDeployment_,
                    IsEnableBackupLink());
            } else {
                CHK_RET(HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, devicePhyId_, deviceLogicId_));
            }
        }

        if ((static_cast<s32>(devicePhyId_) != HOST_DEVICE_ID && isHaveCpuRank_)
            || (IsEnableRoce() && nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST)
            || (Is310PDevice() && nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST)) {
            u32 devicePhyID = (static_cast<s32>(devicePhyId_) == HOST_DEVICE_ID) ? 0 : devicePhyId_;
            CHK_RET(HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_HOST, devicePhyID, deviceLogicId_));
        }

        socketManager_ = nullptr;
    }

    raResourceInit_ = false;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetWorkspaceResource(
    const std::string& tag, void* memPtr, u64& maxSize, std::vector<rtStream_t>& stream)
{
    return workSpaceRes_->SetWorkspaceResource(tag, memPtr, maxSize, stream);
}

void HcclCommunicator::DestroyWorkspaceResource(const std::string& tag)
{
    if (workSpaceRes_ == nullptr) {
        return;
    }
    workSpaceRes_->DestroyWorkspaceResource(tag);
}

HcclResult HcclCommunicator::AtomicInitSet()
{
    CHK_PRT_RET(
        initializedFlag_.test_and_set(),
        HCCL_ERROR(
            "[HcclCommunicator][AtomicInitSet]errNo[0x%016llx] instance "
            "already been initialized",
            HCCL_ERROR_CODE(HCCL_E_INTERNAL)),
        HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

void HcclCommunicator::AtomicInitClear() { initializedFlag_.clear(); }

u32 HcclCommunicator::GetUserRank() const { return realUserRank_; }

u32 HcclCommunicator::GetGroupRank() const { return userRank_; }

u32 HcclCommunicator::GetRankSize() const { return userRankSize_; }

u32 HcclCommunicator::GetRankInParentComm() const { return rankInParentComm_; }

bool HcclCommunicator::GetNicInitialized() { return nicInitialized_ > 0; }

/*
    1. 选择算法
    2. 计算resource，存到request内
    3. 创建和分配资源
*/
HcclResult HcclCommunicator::HcclSelectAlg(
    HcclCMDType opType, u64 count, void* counts, HcclDataType dataType, HcclReduceOp op, int32_t aivCoreLimit,
    bool& ifAiv, std::string& algName)
{
    HCCL_INFO(
        "[HcclCommunicator][HcclSelectAlg] start to run with opType[%d], count[%llu], dataType[%d], reduceOp[%d], "
        "aivCoreLimit[%d]",
        opType, count, dataType, op, aivCoreLimit);
    ifAiv = false;
    if (opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V || opType == HcclCMDType::HCCL_CMD_ALLGATHER_V
        || opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV || opType == HcclCMDType::HCCL_CMD_BATCH_WRITE) {
        HCCL_INFO("[HcclCommunicator][HcclSelectAlg] opType[%d] no need select AIV algorithm", opType);
        return HCCL_SUCCESS;
    }
    /* 选择算法前，先更新成图模式 */
    auto originWorkflowMode = GetWorkflowMode();
    SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);

    std::unique_ptr<CollAlgOperator> algOperator = implAlg_->GetAlgOperator(opType);
    CHK_SMART_PTR_NULL(algOperator);

    OpParam param;
    param.reduceType = op;
    param.opType = opType;
    if (opType == HcclCMDType::HCCL_CMD_ALLTOALL || opType == HcclCMDType::HCCL_CMD_ALLTOALLV
        || opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        param.All2AllDataDes.sendType = dataType;
        param.All2AllDataDes.recvType = dataType;
        param.All2AllDataDes.sendCount = count;
    } else if (opType == HcclCMDType::HCCL_CMD_ALLGATHER_V || opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
        param.VDataDes.counts = counts;
        param.VDataDes.dataType = dataType;
    } else {
        param.DataDes.count = count;
        param.DataDes.dataType = dataType;
    }

    AlgDesc algDesc;
    std::string newTag;
    ResourceLimit limit{true, true, 0};
    limit.aivCoreLimit = aivCoreLimit;
    CHK_RET(algOperator->SelectAlg("", param, limit, algName, algDesc, newTag));

    /* 非AIV算法直接返回 */
    if (!algDesc.isAivMode) {
        HCCL_INFO("[HcclCommunicator][HcclSelectAlg] select non-Aiv alg, early return");
        return HCCL_SUCCESS;
    }

    /* 完成算法选择和记录后，恢复成原来的模式 */
    SetWorkflowMode(originWorkflowMode);
    ifAiv = true;
    HCCL_INFO("[HcclCommunicator][HcclSelectAlg] compile for aiv, select algName is [%s]", algName.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::HcclCalcNumBlocks(
    HcclCMDType opType, u64 count, void* counts, HcclDataType dataType, int32_t aivCoreLimit, std::string& algName,
    u32& numBlocks)
{
    auto originWorkflowMode = GetWorkflowMode();
    SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
    std::unique_ptr<CollAlgOperator> algOperator = implAlg_->GetAlgOperator(opType);
    CHK_SMART_PTR_NULL(algOperator);
    OpParam param;

    param.opType = opType;
    if (opType == HcclCMDType::HCCL_CMD_ALLTOALL || opType == HcclCMDType::HCCL_CMD_ALLTOALLV
        || opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        param.All2AllDataDes.sendType = dataType;
        param.All2AllDataDes.recvType = dataType;
        param.All2AllDataDes.sendCount = count;
    } else if (opType == HcclCMDType::HCCL_CMD_ALLGATHER_V || opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
        param.VDataDes.counts = counts;
        param.VDataDes.dataType = dataType;
    } else {
        param.DataDes.count = count;
        param.DataDes.dataType = dataType;
    }

    CHK_PRT_RET(
        algOperator->CalNumBlocks(algName, param, numBlocks, aivCoreLimit) != HCCL_SUCCESS,
        HCCL_ERROR("[%s] CalNumBlocks failed", __func__), HCCL_E_PARA);
    SetWorkflowMode(originWorkflowMode);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::HcclGetAlgExecParam(
    const std::string& tag, HcclCMDType opType, u64 count, void* inputPtr, void* outputPtr, bool clearEnable,
    HcclDataType dataType, HcclReduceOp op, void*& commContext, u64& len, u32 aivCoreLimit)
{
    /* 将Host申请和注册好的资源，传给AICPU */
    // 1\ algName 从getstr里某一个名字里获取出来（要防止名字重复） commContext & len 从 response里拿
    // 2\ rtmemcpy 先获取一下algoperator对象，用这个调用getalgxxx
    AivSuperKernelArgs aivSuperKernelArgs;
    SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);

    OpParam param;
    param.DataDes.count = count;
    param.DataDes.dataType = dataType;
    param.reduceType = op;
    param.tag = tag;
    param.inputPtr = inputPtr;
    param.outputPtr = outputPtr;
    param.opType = opType;
    u64 totalSize;
    std::vector<u64> sendCountMatrix(userRankSize_ * userRankSize_, count);
    if (opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        param.All2AllDataDes.sendType = dataType;
        param.All2AllDataDes.recvType = dataType;
        param.All2AllDataDes.sendCount = count;
        param.All2AllDataDes.sendCountMatrix = static_cast<void*>(sendCountMatrix.data());
    }

    if (opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER || opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        totalSize = count * SIZE_TABLE[dataType] * userRankSize_;
    } else {
        totalSize = count * SIZE_TABLE[dataType]; // allreduce就是输入
    }
    param.inputSize = totalSize;
    std::unique_ptr<CollAlgOperator> algOperator = implAlg_->GetAlgOperator(opType);
    CHK_SMART_PTR_NULL(algOperator);
    std::string algName;
    AlgResourceResponse algResResponse;
    std::string newTag;
    ResourceLimit limit;
    limit.ifLimit = true;
    limit.aivCoreLimit = aivCoreLimit;
    AlgDesc algDesc;
    algDesc.isLastSelect = true;
    CHK_RET(algOperator->SelectAlg(param.tag, param, limit, algName, algDesc, newTag));

    // 资源创建
    InsertNewTagToTagMap(newTag, param.tag);
    if (resMap_.find(newTag) == resMap_.end()) {
        HCCL_INFO("[HcclCoommunicator][HcclAllocRes] algName[%s], alloc new res", algName.c_str());
        AlgResourceRequest resRequest;
        CHK_RET(algOperator->CalcResRequest(algName, param, resRequest)); // [重构建议] 计算和alloc可以拆开
        CHK_RET(AllocAlgResource(newTag, opType, param, resRequest, resMap_[newTag]));
        CHK_RET(algOperator->PrepareCommInfoToDevice(algName, resMap_[newTag]));
        // 暂不作心跳注册
    }

    CHK_RET(algOperator->GetAivExecParam(algName, param, resMap_[newTag], aivSuperKernelArgs));

    // gettag
    HCCL_INFO("SPK, rank %llu.", userRank_);
    u32 numBlocks;
    CHK_PRT_RET(
        algOperator->CalNumBlocks(algName, param, numBlocks, aivCoreLimit) != HCCL_SUCCESS,
        HCCL_ERROR("[%s] CalNumBlocks failed", __func__), HCCL_E_PARA);
    if (clearEnable) {
        aivOffloadTag_ = 1;
    }
    GetAivTag(algDesc.aivTagNum, false, aivSuperKernelArgs.tag); // workflowmode为图模式
    aivSuperKernelArgs.numBlocks = numBlocks;

    HCCL_INFO(
        "SPK, Tag %llu aivCoreLimit %u, numBlocks %llu.", aivSuperKernelArgs.tag, aivCoreLimit,
        aivSuperKernelArgs.numBlocks);
    // clearenable
    //  拷贝到Device
    SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    void* sendAlgParamMemPtr = nullptr;
    // alloc device 地址
    CHK_RET(hrtMalloc(&sendAlgParamMemPtr, sizeof(AivSuperKernelArgs)));
    HCCL_INFO("SPK sendalgparam %p.", sendAlgParamMemPtr);

    HcclResult hcclRet = hrtMemSyncCopy(
        sendAlgParamMemPtr, sizeof(AivSuperKernelArgs), &aivSuperKernelArgs, sizeof(AivSuperKernelArgs),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE);
    if (hcclRet != HCCL_SUCCESS) {
        HCCL_ERROR("[HcclCommunicator][%s]hrtMemSyncCopy error, ret[%d]", __func__, hcclRet);
        CHK_RET(hrtFree(sendAlgParamMemPtr));
        return hcclRet;
    }
    commContext = sendAlgParamMemPtr;
    len = sizeof(AivSuperKernelArgs);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetAivTag(s32 tagNum, bool isCapture, s32& aivTag)
{
    bool useOpbaseFlag = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !isCapture);
    if (useOpbaseFlag) {
        aivTag = aivOpbaseTag_;
        aivOpbaseTag_ = GetNextAivTag(aivOpbaseTag_, tagNum);
    } else {
        aivTag = aivOffloadTag_;
        aivOffloadTag_ = GetNextAivTag(aivOffloadTag_, tagNum);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CheckDeviceType(const DevType deviceType) const
{
    if ((deviceType >= DevType::DEV_TYPE_COUNT) || (deviceType < DevType::DEV_TYPE_910)) {
        HCCL_ERROR(
            "[Check][DeviceType]errNo[0x%016llx] device Type[%d] out of range[%d, %d]", HCCL_ERROR_CODE(HCCL_E_PARA),
            deviceType, DevType::DEV_TYPE_910, DevType::DEV_TYPE_NOSOC);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CheckReductionOp(const HcclReduceOp op) const
{
    if ((op >= HCCL_REDUCE_RESERVED) || (op < HCCL_REDUCE_SUM)) {
        HCCL_ERROR("[Check][ReductionOp]errNo[0x%016llx] op:[%d] not supported", HCCL_ERROR_CODE(HCCL_E_PARA), op);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CheckUserRank(const u32 userRank) const
{
    if (userRankSize_ <= userRank) {
        HCCL_ERROR(
            "[Check][UserRank]errNo[0x%016llx] userRank:[%u] is out of range[0 ~ %u]", HCCL_ERROR_CODE(HCCL_E_PARA),
            userRank, userRankSize_);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CheckCount(const u64 count) const
{
    if (count > SYS_MAX_COUNT) {
        HCCL_ERROR(
            "[Check][Count]errNo[0x%016llx] count[%llu] is invalid(bigger than MAX count[%llu])",
            HCCL_ERROR_CODE(HCCL_E_PARA), count, SYS_MAX_COUNT);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetGroupRanksInfo(const std::vector<u32>& groupRanks, std::vector<RankInfo>& ranksInfo)
{
    ranksInfo.clear();
    std::vector<RankInfo> tmpRankInfoList;
    tmpRankInfoList.assign(rankInfoList_.begin(), rankInfoList_.end());

    for (u32 index = 0; index < groupRanks.size(); index++) {
        if (tmpRankInfoList.size() <= groupRanks[index]) {
            HCCL_ERROR(
                "[Get][GroupRanksInfo]errNo[0x%016llx] groupRanks[%u]=[%u], >= rankinfolist size[%zu]",
                HCCL_ERROR_CODE(HCCL_E_PARA), index, groupRanks[index], tmpRankInfoList.size());
            return HCCL_E_PARA;
        }
        tmpRankInfoList[groupRanks[index]].userRank = index;
        ranksInfo.push_back(tmpRankInfoList[groupRanks[index]]);
        HCCL_DEBUG(
            "index: %d userRank: %d host ip: %s host port: %u dev phy id: %d serverIdx:%d", index,
            tmpRankInfoList[groupRanks[index]].userRank, tmpRankInfoList[groupRanks[index]].hostIp.GetReadableAddress(),
            tmpRankInfoList[groupRanks[index]].hostPort, tmpRankInfoList[groupRanks[index]].devicePhyId,
            tmpRankInfoList[groupRanks[index]].serverIdx);
    }

    // 按rank id从小到大的顺序返回
    std::sort(ranksInfo.begin(), ranksInfo.end(), CompareWithUserRank);

    for (u32 index = 0; index < ranksInfo.size(); ++index) {
        if (index != ranksInfo[index].userRank) {
            HCCL_ERROR(
                "[Get][GroupRanksInfo]errNo[0x%016llx] index[%u] != user rank[%u]", HCCL_ERROR_CODE(HCCL_E_PARA), index,
                ranksInfo[index].userRank);
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetGroupCommonData(WorldGroupInfo& groupCommonData) const
{
    groupCommonData.inlineReduceSwitchOn = inlineReduceSwitchOn_;
    groupCommonData.deviceType = deviceType_;
    groupCommonData.deviceLogicId = deviceLogicId_;
    groupCommonData.profilingInitiated = profilingInitiated_;
    groupCommonData.serverId = serverId_;
    groupCommonData.phyIdNicInfoMap = rankDevicePhyIdNicInfoMap_;
    groupCommonData.worldRankInfoList = rankInfoList_;
    groupCommonData.ranksPort = nicRanksPort_;
    groupCommonData.vnicRanksPort = vnicRanksPort_;
    groupCommonData.useSuperPodMode = useSuperPodMode_;
    groupCommonData.devPortSwitchOn = commPortConfig_.devPortSwitchOn;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetWorkspaceMemSize(
    const std::string& opType, u64 count, HcclDataType dataType, u32& rankSize, u64& memSize, DevType& deviceType) const
{
    return workSpaceRes_->GetWorkspaceMemSize(opType, count, dataType, rankSize, memSize, deviceType);
}

DeviceMem HcclCommunicator::GetWorkspaceScracthMem(const std::string& tag, u64 allocMemSize)
{
    return workSpaceRes_->AllocDeviceMem(tag, allocMemSize);
}

std::vector<Stream> HcclCommunicator::GetWorkspaceSubStreams(const std::string& tag, u32 num)
{
    return workSpaceRes_->AllocSlaveStreams(tag, num);
}

HcclResult HcclCommunicator::InitProfiling()
{
    if (static_cast<s32>(devicePhyId_) == HOST_DEVICE_ID) {
        HCCL_ERROR("[Init][Profiling]not support cpu rank");
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(profilingInitiated_, HCCL_DEBUG("Profiling plugin has already been Initiated."), HCCL_SUCCESS);

    if (profilingMode_ != HcomProfilingMode::PROFILING_OPEN && GetExternalInputProfilingMode()) {
        profilingMode_ = HcomProfilingMode::PROFILING_OPEN;
        profilingOption_ = GetExternalInputProfilingOption();
    }
    HCCL_INFO("profiling config information:options[%s], mode[%d]", profilingOption_.c_str(), profilingMode_);

    // profilingInitiated_会广播给所有子通信域，用于避免taskInfoSaver的重复初始化
    profilingInitiated_ = true;
    // isExecuteProfilingInit_用于记录本impl是否执行了taskInfoSaver的初始化，用于进行对应的释放
    isExecuteProfilingInit_ = true;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::DeinitProfiling()
{
    CHK_PRT_RET(!profilingInitiated_, HCCL_DEBUG("Profiling plugin has not been Initiated"), HCCL_SUCCESS);
    profilingInitiated_ = false;
    HCCL_INFO("Profiling is deinitiated.");
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::RegistTaskExceptionHandler() const
{
    CHK_RET(TaskExceptionHandler::Init());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::UnRegistTaskExceptionHandler() const
{
    CHK_RET(TaskExceptionHandler::DeInit());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetInCCLbuffer(void*& buffer, u64& size)
{
    return cclBufferManager_.GetInCCLbuffer(buffer, size);
}

HcclResult HcclCommunicator::GetOutCCLbuffer(void*& buffer, u64& size)
{
    return cclBufferManager_.GetOutCCLbuffer(buffer, size);
}

void HcclCommunicator::ReleaseCommCCLbuffer() { cclBufferManager_.ReleaseCommCCLbuffer(); }

HcclResult HcclCommunicator::ReleaseCommInfos()
{
    if (implAlg_ != nullptr) {
        return implAlg_->ReleaseCommInfos();
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitProfiler()
{
    profilerManager_.reset(new (std::nothrow)
                               ProfilerManager(devicePhyId_, deviceLogicId_, realUserRank_, userRankSize_));
    CHK_SMART_PTR_NULL(profilerManager_);
    HcclResult ret = profilerManager_->InitProfiler();
    CHK_PRT_RET(
        (ret != HCCL_SUCCESS), HCCL_ERROR("[BASE][InitProfiler]profilerManager_ InitProfiler failed."), HCCL_E_PARA);

    HCCL_INFO("[BASE][InitProfiler]Register CtrlCallBack success.");
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CreateCommCCLbuffer()
{
    // user mem和CCL buffer互斥，不支持同时使用
    if (isUserMemRegisted_) {
        HCCL_ERROR(
            "[HcclCommunicator][%s]tag[%s]The user mem has been registered, "
            "does not support create CCL Buffer.",
            __func__, identifier_.c_str());
        return HCCL_E_NOT_SUPPORT;
    }
    return cclBufferManager_.CreateCommCCLbuffer(cclBuffName_);
}

HcclResult HcclCommunicator::InitCCLbuffer(u64 inCCLbufferSize, u64 outCCLbufferSize)
{
    return cclBufferManager_.InitCCLbuffer(inCCLbufferSize, outCCLbufferSize);
}

u32 HcclCommunicator::GetLocalNicPort(NicType nicType)
{
    u32 port = HCCL_INVALID_PORT;
    if (nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST) {
        return GetHostPort(devicePhyId_);
    }
    // isUseRankPort_在ranksPort初始化时一同配置：1. 异构场景 2. 开启device侧端口配置
    // groupRanksPort_为空说明此时处于全局通信域，要从ranksPort_取监听端口；否则取groupRanksPort_
    bool devicePortSwitchOn = commPortConfig_.devPortSwitchOn;
    if (nicType == NicType::HOST_NIC_TYPE) {
        port = GetHostPort(devicePhyId_);
    } else if (devicePortSwitchOn && nicType == NicType::VNIC_TYPE) {
        // vnic ports仅在开启device侧端口配置时单独配置
        std::vector<u32>& ranksPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
        port = GetNicPort(devicePhyId_, ranksPorts, userRank_, isUseRankPort_);
    } else {
        // 1. 开启device侧端口配置时的nic port时使用ranksPorts
        // 2. 异构场景使用ranksPorts
        // 3. 其余场景场景isUseRankPort_应当为false，使用默认port
        std::vector<u32>& ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
        port = GetNicPort(devicePhyId_, ranksPorts, userRank_, isUseRankPort_);
    }
    HCCL_INFO(
        "[HcclCommunicator][GetLocalNicPort] nicType[%u], devicePortSwitchOn[%u], isUseRankPort[%u], "
        "get port[%u], devId[%u]",
        nicType, devicePortSwitchOn, isUseRankPort_, port, devicePhyId_);
    return port;
}

HcclResult HcclCommunicator::CheckOneSidedBackupAndSetDevId(
    u32& backupDevPhyId, u32& backupDevLogicId, std::vector<HcclIpAddress>& localIpList,
    bool& isOneSidedTaskAndBackupInitA3)
{
    if (!IsOneSidedIdentifier(identifier_)) {
        isOneSidedTaskAndBackupInitA3 = false;
        HCCL_INFO("[%s] comm[%s] is not one sided comm.", __func__, identifier_.c_str());
        return HCCL_SUCCESS;
    }
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (deviceType != DevType::DEV_TYPE_910_93) {
        isOneSidedTaskAndBackupInitA3 = false;
        HCCL_INFO(
            "[HcclCommunicator::CheckOneSidedBackupAndSetDevId] DeviceType[%d] is not 910_93, one sided backup not "
            "support",
            static_cast<u32>(deviceType));
        return HCCL_SUCCESS;
    }
    CHK_RET(hrtGetPairDevicePhyId(devicePhyId_, backupDevPhyId));

    std::vector<HcclIpAddress> backupIpList;
    std::vector<std::vector<HcclIpAddress>> chipDeviceIPs;
    CHK_RET(hrtRaGetDeviceAllNicIP(chipDeviceIPs));
    u32 ipIdex = 1U - (devicePhyId_ % 2U);
    std::copy_if(
        chipDeviceIPs[ipIdex].begin(), chipDeviceIPs[ipIdex].end(), std::back_inserter(backupIpList),
        [](const HcclIpAddress& ip) {
            return !ip.IsIPv6();
        });
    HCCL_INFO(
        "devicePhysicID[%u], backupDeviceId[%d], backupDeviceIP[0]:[%s], devIpAddr_[%s], ", devicePhyId_,
        backupDevPhyId, backupIpList[0].GetReadableAddress(), devIpAddr_[0].GetReadableAddress());
    CHK_RET(hrtRaGetDeviceIP(devicePhyId_, localIpList));
    auto equalToLocal = [this](const HcclIpAddress& entry) {
        return entry == devIpAddr_[0];
    };
    isOneSidedTaskAndBackupInitA3 = any_of(backupIpList.begin(), backupIpList.end(), equalToLocal)
                                    && !any_of(localIpList.begin(), localIpList.end(), equalToLocal);
    if (isOneSidedTaskAndBackupInitA3) {
        CHK_RET(hrtGetDeviceIndexByPhyId(backupDevPhyId, backupDevLogicId));
    }

    HCCL_INFO(
        "[HcclCommunicator::CheckOneSidedBackupAndSetDevId] isOneSidedTaskAndBackupInitA3[%s]",
        isOneSidedTaskAndBackupInitA3 ? "true" : "false");
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::OneSidedBackupInitNetResource(
    HcclNetDevCtx& nicPortBackUpCtx, u32& backupDevPhyId, u32& backupDevLogicId,
    std::vector<HcclIpAddress>& localIpList)
{
    devBackupIpAddr_[0] = devIpAddr_[0];
    deviceBackUpPhyId_ = backupDevPhyId;
    deviceBackUpLogicId_ = backupDevLogicId;
    CHK_RET(HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, backupDevPhyId, backupDevLogicId, false, true));
    HCCL_INFO(
        "[HcclCommunicator::OneSidedBackupInitNetResource] OpenDev with backupDevPhyId[%d], backupDevLogicId[%d], "
        "localIpList[%s], backupIp[%s]",
        backupDevPhyId, backupDevLogicId, localIpList[0].GetReadableAddress(), devIpAddr_[0].GetReadableAddress());
    CHK_RET(HcclNetOpenDev(
        &nicPortBackUpCtx, NicType::DEVICE_NIC_TYPE, backupDevPhyId, backupDevLogicId, devIpAddr_[0], localIpList[0]));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::OneSidedBackupServerInit(HcclNetDevCtx& nicPortBackUpCtx)
{
    u32 backupPort = HCCL_INVALID_PORT;
    for (const auto& rankInfo : rankInfoList_) {
        if (rankInfo.userRank == userRank_) {
            backupPort = rankInfo.deviceNicPort;
        }
    }
    CHK_RET(socketManager_->ServerInit(nicPortBackUpCtx, backupPort));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitDevicePrimaryNic([[maybe_unused]] bool isMC2ReInit, bool isOneSidedTaskAndBackupInitA3)
{
    std::shared_ptr<HcclSocket>& devNicSocket = commPortConfig_.devNicListen.first;
    if (devNicSocket && !isOneSidedTaskAndBackupInitA3) {
        HcclNetDevCtx& devNicCtx = commPortConfig_.devNicListen.second;
        CHK_PTR_NULL(devNicCtx);
        netDevCtxMap_.insert(std::make_pair(devNicSocket->GetLocalIp(), devNicCtx));
        CHK_RET(socketManager_->ServerInit(devNicCtx, devNicSocket->GetLocalPort()));
        commPortConfig_.devNicListen.second = nullptr;
        HCCL_INFO(
            "[HcclCommunicator][InitNic] init nic with listened socket success, "
            "listened ip[%s] port[%u]",
            devNicSocket->GetLocalIp().GetReadableAddress(), devNicSocket->GetLocalPort());
    } else if (!isOneSidedTaskAndBackupInitA3) {
        u32 port = GetLocalNicPort(NicType::DEVICE_NIC_TYPE);
        u32 nicNum = devIpAddr_.size();
        for (u32 i = 0; i < nicNum; i++) {
            if (devIpAddr_[i].IsInvalid()) {
                HCCL_INFO("[Init][Nic]nic num[%u] deviceip is invalid, total nicNum[%u]", i, nicNum);
                continue;
            }
            HcclNetDevCtx nicPortCtx;
            CHK_RET(HcclNetOpenDev(&nicPortCtx, NicType::DEVICE_NIC_TYPE, devicePhyId_, deviceLogicId_, devIpAddr_[i]));
            CHK_PTR_NULL(nicPortCtx);
            netDevCtxMap_.insert(std::make_pair(devIpAddr_[i], nicPortCtx));
            CHK_RET(socketManager_->ServerInit(nicPortCtx, port));
            HCCL_INFO(
                "[HcclCommunicator][InitNic] init nic with ip[%s] port[%u] success", devIpAddr_[i].GetReadableAddress(),
                port);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitDeviceBackupNic(
    u32 backupDevPhyId, u32 backupDevLogicId, std::vector<HcclIpAddress>& localIpList,
    bool isOneSidedTaskAndBackupInitA3)
{
    if (IsEnableBackupLink() || isOneSidedTaskAndBackupInitA3) {
        std::shared_ptr<HcclSocket>& backupNicSocket = commPortConfig_.backupDevNicListen.first;
        if (backupNicSocket) {
            HcclNetDevCtx& backupNicCtx = commPortConfig_.backupDevNicListen.second;
            CHK_PTR_NULL(backupNicCtx);
            netDevCtxMap_.insert(std::make_pair(backupNicSocket->GetLocalIp(), backupNicCtx));
            CHK_RET(socketManager_->ServerInit(backupNicCtx, backupNicSocket->GetLocalPort()));
            commPortConfig_.backupDevNicListen.second = nullptr;
            HCCL_INFO(
                "[HcclCommunicator][InitNic] init backup nic with listened socket success, "
                "listened ip[%s] port[%u]",
                backupNicSocket->GetLocalIp().GetReadableAddress(), backupNicSocket->GetLocalPort());
        } else {
            HcclNetDevCtx nicPortBackUpCtx;
            if (isOneSidedTaskAndBackupInitA3) {
                CHK_RET(OneSidedBackupInitNetResource(nicPortBackUpCtx, backupDevPhyId, backupDevLogicId, localIpList));
            } else {
                CHK_RET(HcclNetOpenDev(
                    &nicPortBackUpCtx, NicType::DEVICE_NIC_TYPE, deviceBackUpPhyId_, deviceBackUpLogicId_,
                    devBackupIpAddr_[0], devIpAddr_[0]));
            }
            CHK_PTR_NULL(nicPortBackUpCtx);
            netDevCtxMap_.insert(std::make_pair(devBackupIpAddr_[0], nicPortBackUpCtx));
            if (isOneSidedTaskAndBackupInitA3) {
                CHK_RET(OneSidedBackupServerInit(nicPortBackUpCtx));
            } else {
                CHK_RET(socketManager_->ServerInit(nicPortBackUpCtx, devBackupPort_));
            }
            HCCL_DEBUG(
                "[%s]finish backup ServerInit, deviceBackUpPhyId_[%u], deviceBackUpLogicId_[%u], "
                "devBackupIpAddr_[%s], devBackupPort_[%u], nicDeployment_[%d], IsEnableBackupLink[%d], "
                "netDevCtxMap_.size[%d]",
                __func__, deviceBackUpPhyId_, deviceBackUpLogicId_, devBackupIpAddr_[0].GetReadableAddress(),
                devBackupPort_, nicDeployment_, IsEnableBackupLink(), netDevCtxMap_.size());
            HCCL_INFO(
                "[HcclCommunicator][InitNic] init backup nic with ip[%s] port[%u] success",
                devBackupIpAddr_[0].GetReadableAddress(), devBackupPort_);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitNicDeviceDeploy(
    bool isMC2ReInit, u32 backupDevPhyId, u32 backupDevLogicId, std::vector<HcclIpAddress>& localIpList,
    bool isOneSidedTaskAndBackupInitA3)
{
    CHK_RET(InitDevicePrimaryNic(isMC2ReInit, isOneSidedTaskAndBackupInitA3));
    attrCollector_.GenUsedRdmaLevel0();
    isUsedRdmaLevel0_ = attrCollector_.GetUsedRdmaLevel0();
    CHK_RET(InitDeviceBackupNic(backupDevPhyId, backupDevLogicId, localIpList, isOneSidedTaskAndBackupInitA3));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitNicHostDeploy()
{
    u32 port = GetLocalNicPort(NicType::HOST_NIC_TYPE);
    CHK_PRT_RET(
        (hostIp_.IsInvalid()),
        HCCL_ERROR("[Init][Nic] host ip is invalid when NIC "
                   "deployment is host. "),
        HCCL_E_PARA);
    attrCollector_.GenUsedRdmaLevel0();
    isUsedRdmaLevel0_ = attrCollector_.GetUsedRdmaLevel0();
    u32 devicePhyID = (static_cast<s32>(devicePhyId_) == HOST_DEVICE_ID) ? 0 : devicePhyId_;

    u32 i = 0;
    HcclNetDevCtx nicPortCtx;
    for (i = 0; i < devIpAddr_.size(); i++) {
        if (devIpAddr_[i].IsInvalid()) {
            HCCL_INFO("[Init][Nic]nic num[%u] deviceip is invalid, total nicNum[%u]", i, devIpAddr_.size());
            continue;
        }
        std::vector<u32>& ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
        port = GetNicPort(devicePhyId_, ranksPorts, userRank_, isUseRankPort_);
        CHK_RET(HcclNetOpenDev(&nicPortCtx, NicType::HOST_NIC_TYPE, devicePhyId_, deviceLogicId_, devIpAddr_[i]));
        CHK_PTR_NULL(nicPortCtx);
        netDevCtxMap_.insert(std::make_pair(devIpAddr_[i], nicPortCtx));
        HcclNetDevSetProtoType(nicPortCtx, HCCL_PROTO_TYPE_ROCE);
        break;
    }

    if (i == devIpAddr_.size()) {
        port = GetLocalNicPort(NicType::HOST_NIC_TYPE);
        CHK_RET(HcclNetOpenDev(&nicPortCtx, NicType::HOST_NIC_TYPE, devicePhyId_, deviceLogicId_, hostIp_));
        CHK_PTR_NULL(nicPortCtx);
        netDevCtxMap_.insert(std::make_pair(hostIp_, nicPortCtx));
    }
    HCCL_INFO("[Init][Nic], hostPort[%u], devicePhyID[%u]", port, devicePhyID);
    CHK_RET(socketManager_->ServerInit(nicPortCtx, port));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitNic(bool isMC2ReInit)
{
    if (GetExternalInputIntraRoceSwitch() == 0 && servRankInfo_.size() == 1 && isDiffDeviceModule_ && !isMC2ReInit) {
        return HCCL_SUCCESS;
    }
    u32 backupDevPhyId = INVALID_INT;
    u32 backupDevLogicId = INVALID_INT;
    bool isOneSidedTaskAndBackupInitA3 = false;
    vector<HcclIpAddress> localIpList;
    CHK_RET(
        CheckOneSidedBackupAndSetDevId(backupDevPhyId, backupDevLogicId, localIpList, isOneSidedTaskAndBackupInitA3));

    if (nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_DEVICE) {
        CHK_RET(InitNicDeviceDeploy(
            isMC2ReInit, backupDevPhyId, backupDevLogicId, localIpList, isOneSidedTaskAndBackupInitA3));
    } else if (nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST) {
        CHK_RET(InitNicHostDeploy());
    } else {
        HCCL_ERROR("[Init][Nic]nic deployment[%d] is not supported", nicDeployment_);
        return HCCL_E_PARA;
    }
    isNeedInitNic_ = true;
    attrCollector_.SetNeedInitNicFlag(isNeedInitNic_);
    nicInitialized_++;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::DeinitNicHostDeploy()
{
    u32 port = GetLocalNicPort(NicType::HOST_NIC_TYPE);
    CHK_PRT_RET(
        (hostIp_.IsInvalid()),
        HCCL_ERROR("[DeInit][Nic] host ip is invalid when NIC "
                   "deployment is host. "),
        HCCL_E_PARA);

    u32 i = 0;
    for (i = 0; i < devIpAddr_.size(); i++) {
        if (devIpAddr_[i].IsInvalid()) {
            HCCL_INFO("[Init][Nic]nic num[%u] deviceip is invalid, total nicNum[%u]", i, devIpAddr_.size());
            continue;
        }
        std::vector<u32>& ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
        port = GetNicPort(devicePhyId_, ranksPorts, userRank_, isUseRankPort_);

        CHK_RET(socketManager_->ServerDeInit(netDevCtxMap_[devIpAddr_[i]], port));
        HcclNetCloseDev(netDevCtxMap_[devIpAddr_[i]]);
        netDevCtxMap_.erase(devIpAddr_[i]);

        break;
    }

    if (i == devIpAddr_.size()) {
        CHK_RET(socketManager_->ServerDeInit(netDevCtxMap_[hostIp_], port));
        HcclNetCloseDev(netDevCtxMap_[hostIp_]);
        netDevCtxMap_.erase(hostIp_);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::DeinitNic()
{
    if (nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_DEVICE) {
        u32 port = GetLocalNicPort(NicType::DEVICE_NIC_TYPE);
        u32 nicNum = devIpAddr_.size();
        for (u32 i = 0; i < nicNum; i++) {
            if (devIpAddr_[i].IsInvalid()) {
                HCCL_INFO("continue invalid devIp %s", devIpAddr_[i].GetReadableAddress());
                continue;
            }
            if (netDevCtxMap_.find(devIpAddr_[i]) == netDevCtxMap_.end()) {
                HCCL_INFO("devIp[%s] not found in netDevCtxMap_", devIpAddr_[i].GetReadableAddress());
                continue;
            }
            CHK_RET(socketManager_->ServerDeInit(netDevCtxMap_[devIpAddr_[i]], port));
            // 最后一次调用才删除netCtx
            if (nicInitialized_ - 1 <= 0) {
                HcclNetCloseDev(netDevCtxMap_[devIpAddr_[i]]);
                netDevCtxMap_.erase(devIpAddr_[i]);
            }
        }
        if (IsEnableBackupLink() && netDevCtxMap_.find(devBackupIpAddr_[0]) != netDevCtxMap_.end()) {
            // 超节点 && level2支持重执行 && Aicpu -> 备用网卡 deinit
            CHK_RET(socketManager_->ServerDeInit(netDevCtxMap_[devBackupIpAddr_[0]], devBackupPort_));
            if (nicInitialized_ - 1 <= 0) {
                HcclNetCloseDev(netDevCtxMap_[devBackupIpAddr_[0]]);
                netDevCtxMap_.erase(devBackupIpAddr_[0]);
                HCCL_DEBUG(
                    "[%s]finish backup ServerDeInit devBackupIpAddr_[%s], port[%u], IsEnableBackupLink[%d]", __func__,
                    devBackupIpAddr_[0].GetReadableAddress(), devBackupPort_, IsEnableBackupLink());
            }
        }
    } else if (nicDeployment_ == NICDeployment::NIC_DEPLOYMENT_HOST) {
        CHK_RET(DeinitNicHostDeploy());
    } else {
        HCCL_ERROR("[Deinit][Nic]nic deployment[%d] is not supported", nicDeployment_);
        return HCCL_E_PARA;
    }
    nicInitialized_--;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::RegisterRanksToDca()
{
    if (deviceType_ != DevType::DEV_TYPE_910_93 && deviceType_ != DevType::DEV_TYPE_910B) {
        HCCL_WARNING("[RegisterRanksToDca] not support deviceType[%d]", deviceType_);
        return HCCL_SUCCESS;
    }
    CHK_RET(setVnicIpToRankInfoList());
    DetectConnectionAnomalies::GetInstance(deviceLogicId_).Init(rankInfoList_, isNeedInitNic_);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AddOpInfoToHeartBeat(const OpInfoDesc& opInfo, const std::string& tag)
{
    if (Is310PDevice() || deviceType_ == DevType::DEV_TYPE_310P3
        || GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        return HCCL_SUCCESS;
    }
    return Heartbeat::GetInstance(deviceLogicId_).AddOpInfoToHeartBeat(identifier_, opInfo, tag);
}

void HcclCommunicator::DeleteOpInfoToHeartBeat()
{
    if (Is310PDevice() || deviceType_ == DevType::DEV_TYPE_310P3
        || GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        return;
    }
    for (const auto& tag : hbSendRecvTags_) {
        Heartbeat::GetInstance(deviceLogicId_).DeleteOpInfoToHeartBeat(identifier_, tag);
    }
    Heartbeat::GetInstance(deviceLogicId_).DeleteOpInfoToHeartBeat(identifier_, "");
}

HcclResult HcclCommunicator::RegisterToHeartBeat()
{
    if (Is310PDevice() || deviceType_ == DevType::DEV_TYPE_310P3) {
        return HCCL_SUCCESS;
    }
    u32 localPort = commPortConfig_.devPortSwitchOn ? HCCL_INVALID_PORT : GetLocalNicPort(NicType::DEVICE_NIC_TYPE);
    return Heartbeat::GetInstance(deviceLogicId_)
        .RegisterToHeartBeat(
            userRank_, deviceType_, rankInfoList_, localPort, isNeedInitNic_, identifier_, useSuperPodMode_,
            isUsedRdmaLevel0_, retryEnable_, IsEnableBackupLink());
}

HcclResult HcclCommunicator::RegisterToHeartBeat(u32 peerRankId, string& tag)
{
    u32 localPort = commPortConfig_.devPortSwitchOn ? HCCL_INVALID_PORT : GetLocalNicPort(NicType::DEVICE_NIC_TYPE);
    return Heartbeat::GetInstance(deviceLogicId_)
        .RegisterToHeartBeat(
            userRank_, deviceType_, rankInfoList_, localPort, isNeedInitNic_, peerRankId, identifier_, tag,
            useSuperPodMode_, isUsedRdmaLevel0_, retryEnable_, IsEnableBackupLink());
}

void HcclCommunicator::UnRegisterToHeartBeat()
{
    for (auto tag : hbSendRecvTags_) {
        Heartbeat::GetInstance(deviceLogicId_).UnRegisterToHeartBeat(deviceType_, identifier_, tag);
    }
    Heartbeat::GetInstance(deviceLogicId_).UnRegisterToHeartBeat(deviceType_, identifier_);
}

void HcclCommunicator::UnRegisterToCommConfiger() { CommConfiger::GetInstance().UnRegisterToCommConfiger(identifier_); }

HcclResult HcclCommunicator::SetGlobalWorkSpace(std::vector<void*>& globalWorkSpaceAddr)
{
    CHK_RET(HcclSetGlobalWorkSpace(dispatcher_, globalWorkSpaceAddr));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetandClearOverFlowTasks(std::vector<HcclDumpInfo>& hcclDumpInfo)
{
    if (profilerManager_ != nullptr) {
        CHK_RET(profilerManager_->GetandClearOverFlowTasks(hcclDumpInfo));
    } else {
        HCCL_WARNING("[impl][GetDumpTask] profilerManager_ not set");
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetDeviceId(s32& deviceId) const
{
    deviceId = deviceLogicId_;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetCqeError(HcclResult& result)
{
    CHK_RET(Heartbeat::GetInstance(deviceLogicId_).CheckErrorCqe(identifier_, result));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetOpInconsistentError(HcclResult& result)
{
    CHK_RET(Heartbeat::GetInstance(deviceLogicId_).CheckOpInconsistentError(identifier_, result));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SupportDeterministicOptim(bool& isDeterministicOptim)
{
    CHK_SMART_PTR_NULL(implAlg_);
    CHK_RET(implAlg_->SupportDeterministicOptim(isDeterministicOptim));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetHccsLinkNum(u32& numHccsLink)
{
    auto iter = pairLinkInfo_.find(static_cast<u32>(LinkTypeInServer::HCCS_TYPE));
    if (iter == pairLinkInfo_.end()) {
        HCCL_ERROR("[HcclCommunicator][GetHccsLinkNum]HCCS_TYPE is not found");
        return HCCL_E_PARA;
    }
    numHccsLink = iter->second.size();
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllGather(
    const std::string& tag, void* inputPtr, void* outputPtr, u64 inputCount, HcclDataType dataType, HcclRtStream stream,
    [[maybe_unused]] HcomCollOpInfo* opInfo)
{
    bool isCapture = StreamIsCapture(stream);

    bool aicpuUnfoldMode = false;
    if (EnableAicpuUnfold(isCapture) && (userRankSize_ != 1)) {
        aicpuUnfoldMode = true;
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][AllGather]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32>& ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    std::vector<u32>& vnicRanksPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, vnicRanksPorts, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 totalSize = inputCount * perDataSize;

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = totalSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = totalSize * userRankSize_;
    opParam.DataDes.count = inputCount;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = HcclReduceOp::HCCL_REDUCE_RESERVED;
    opParam.stream = streamObj;
    opParam.syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
    opParam.isCapture = isCapture;
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.aicpuCacheEnable = GetExternalInputAicpuCacheEnable();
    opParam.opType = HcclCMDType::HCCL_CMD_ALLGATHER;

    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_ALLGATHER, opParam));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllGatherV(
    const std::string& tag, const void* sendBuf, u64 sendCount, const void* recvBuf, const void* recvCounts,
    const void* rdispls, HcclDataType dataType, HcclRtStream stream)
{
    bool aicpuUnfoldMode = false;

    if (GetAicpuUnfoldConfig() && (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
        aicpuUnfoldMode = true;
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][AllGatherV]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32>& ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    std::vector<u32>& vnicRanksPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, vnicRanksPorts, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 totalSize = sendCount * perDataSize;

    u64 outputSize = 0;
    const u64* counts = static_cast<const u64*>(recvCounts);
    for (u32 i = 0; i < userRankSize_; i++) {
        outputSize += counts[i] * perDataSize;
    }

    bool isCapture = StreamIsCapture(stream);

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = const_cast<void*>(sendBuf);
    opParam.inputSize = totalSize;
    opParam.outputPtr = const_cast<void*>(recvBuf);
    opParam.outputSize = outputSize;
    opParam.VDataDes.dataType = dataType;
    opParam.VDataDes.counts = const_cast<void*>(recvCounts);
    opParam.VDataDes.displs = const_cast<void*>(rdispls);
    opParam.reduceType = HcclReduceOp::HCCL_REDUCE_RESERVED;
    opParam.stream = streamObj;
    opParam.syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.isCapture = isCapture;
    opParam.aicpuCacheEnable = GetExternalInputAicpuCacheEnable();
    opParam.opType = HcclCMDType::HCCL_CMD_ALLGATHER_V;

    if (UNLIKELY(GetDebugConfig() & HCCL_ALG)) {
        for (u32 i = 0; i < userRankSize_; i++) {
            HCCL_CONFIG_DEBUG(
                HCCL_ALG, "[HcclCommunicator][AllGatherV]userRank_[%u], rankIdx[%u], recvCounts[%llu], rdispls[%llu]",
                userRank_, i, counts[i], static_cast<const u64*>(rdispls)[i]);
        }
    }

    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_ALLGATHER_V, opParam));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AicpuUnfold(
    const std::string& tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op,
    HcclRtStream stream, HcclCMDType cmdType)
{
    Stream streamObj(stream);
    u32 perDataSize = SIZE_TABLE[dataType];
    u64 totalSize = count * perDataSize;
    bool isCapture = StreamIsCapture(stream);
    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = totalSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = totalSize;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = op;
    opParam.stream = streamObj;
    opParam.isCapture = isCapture;
    opParam.syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
    opParam.aicpuCacheEnable = GetExternalInputAicpuCacheEnable();
    AlgType algType;
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_MESH;
    algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;

    // 构造空vector用于入参，无实际意义
    const std::vector<Stream> slaveStreams;
    CHK_RET(RegisterDfxInfo(opParam, algType, slaveStreams));
    HcclResult ret = HCCL_SUCCESS;
    if (!IsExistCommRes(identifier_)) {
        HCCL_INFO(
            "[AicpuUnfold] tag[%s] count[%llu] dataType[%s] op[%s].", identifier_.c_str(), count,
            GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str());
        uint64_t streamMode = 0;
        CHK_RET(hrtStreamGetMode(stream, &streamMode));

        rtStream_t aicpuStream;
        ret = Mc2AiCpuStreamAllocAndGet(streamMode, aicpuStream);
        void* commContext = nullptr;
        ret = CreateCommResource(identifier_, stream, true, &commContext);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[hcclImpl][CreateComm]create aicpu unfold comminfo by tag[%s] failed. return[%d]", identifier_.c_str(),
                ret);
            return ret;
        }
    }

    std::string kernelName = "RunAicpuRpcSrvLaunch";
    AicpuOpTiling opTilingInfo;
    ret = AicpuKfcTilingDataLaunch(opParam, cmdType, commContext_, kernelName, opTilingInfo);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[hcclImpl][TilingData]aicpu unfold tiling data launch failed. return[%d] inputPtr[%p]"
            "outputPtr[%p] count[%llu] dataType[%s] op[%s]",
            ret, inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str());
        return ret;
    }
    CHK_RET(UnRegisterDfxInfo(opParam, slaveStreams));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllGatherOutPlace(
    const std::string& tag, void* inputPtr, void* outputPtr, u64 inputCount, HcclDataType dataType, HcclRtStream stream)
{
    CHK_RET(CheckSuspendingStatus());
    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][AllGatherOutPlace]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    bool isCapture = StreamIsCapture(stream);

    bool aicpuUnfoldMode = false;
    if (EnableAicpuUnfold(isCapture) && (userRankSize_ != 1)) {
        aicpuUnfoldMode = true;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32>& ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    std::vector<u32>& vnicRanksPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, vnicRanksPorts, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 totalSize = inputCount * perDataSize * userRankSize_;

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = inputCount * perDataSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = totalSize;
    opParam.DataDes.count = inputCount;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = HcclReduceOp::HCCL_REDUCE_RESERVED;
    opParam.stream = streamObj;
    opParam.syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
    opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.aicpuCacheEnable = GetExternalInputAicpuCacheEnable();
    opParam.isCapture = isCapture;
    opParam.opType = HcclCMDType::HCCL_CMD_ALLGATHER;

    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_ALLGATHER, opParam));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllGatherVOutPlace(
    const std::string& tag, void* inputPtr, void* outputPtr, u64 inputCount, const void* outputCounts,
    const void* outputDispls, HcclDataType dataType, HcclRtStream stream)
{
    CHK_RET(CheckSuspendingStatus());
    if (userRankSize_ == 1) {
        // rankSize为1时，退化为AllGather
        return AllGatherOutPlace(tag, inputPtr, outputPtr, inputCount, dataType, stream);
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][AllGatherVOutPlace]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    bool aicpuUnfoldMode = false;
    if (GetAicpuUnfoldConfig() && (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
        aicpuUnfoldMode = true;
    }

    bool isCapture = StreamIsCapture(stream);

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32>& ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    std::vector<u32>& vnicRanksPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, vnicRanksPorts, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 outputSize = 0;
    const u64* counts = static_cast<const u64*>(outputCounts);
    for (u32 i = 0; i < userRankSize_; i++) {
        outputSize += counts[i] * perDataSize;
    }

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = inputCount * perDataSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = outputSize;
    opParam.VDataDes.counts = const_cast<void*>(outputCounts);
    opParam.VDataDes.displs = const_cast<void*>(outputDispls);
    opParam.VDataDes.dataType = dataType;
    opParam.reduceType = HcclReduceOp::HCCL_REDUCE_RESERVED;
    opParam.stream = streamObj;
    opParam.syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
    opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.aicpuCacheEnable = GetExternalInputAicpuCacheEnable();
    opParam.isCapture = isCapture;
    opParam.rankSize = userRankSize_;
    opParam.opType = HcclCMDType::HCCL_CMD_ALLGATHER_V;

    if (UNLIKELY(GetDebugConfig() & HCCL_ALG)) {
        for (u32 i = 0; i < userRankSize_; i++) {
            HCCL_CONFIG_DEBUG(
                HCCL_ALG,
                "[HcclCommunicator][AllGatherVOutPlace]userRank_[%u], rankIdx[%u],"
                "outputCounts[%llu], outputDispls[%llu]",
                userRank_, i, counts[i], static_cast<const u64*>(outputDispls)[i]);
        }
    }

    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_ALLGATHER_V, opParam));

    return HCCL_SUCCESS;
}

void HcclCommunicator::GetAndSetSyncMode(SyncMode& preSyncMode, SyncMode newSyncMode)
{
    if (newSyncMode == SyncMode::UNLIMITED_TIMEWAITSYNCMODE) {
        if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
            HCCL_WARNING("310P don't support unlimited notify wait mode");
        } else {
            HcclGetNotifyWaitMode(dispatcher_, &preSyncMode);
            HcclSetNotifyWaitMode(dispatcher_, newSyncMode);
        }
    }
}

void HcclCommunicator::RestorePreSyncMode(SyncMode preSyncMode, SyncMode newSyncMode)
{
    if (newSyncMode == SyncMode::UNLIMITED_TIMEWAITSYNCMODE && !Is310P3Common(isHaveCpuRank_, deviceType_)) {
        HcclSetNotifyWaitMode(dispatcher_, preSyncMode);
    }
}

HcclResult HcclCommunicator::AllReduce(
    const std::string& tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op,
    HcclRtStream stream, SyncMode syncMode, [[maybe_unused]] const HcomCollOpInfo* opInfo)
{
    CHK_RET(CheckSuspendingStatus());
    bool aicpuUnfoldMode = false;
    if (GetAicpuUnfoldConfig() == true && IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op)
        && deviceType_ == DevType::DEV_TYPE_910_93 && (userRankSize_ != 1)) {
        aicpuUnfoldMode = true;
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][AllReduce]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    // 设置notify wait模式
    SyncMode preSyncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
    GetAndSetSyncMode(preSyncMode, syncMode);

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32>& ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    std::vector<u32>& vnicRanksPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, vnicRanksPorts, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 totalSize = count * perDataSize;

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = totalSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = totalSize;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = op;
    opParam.stream = streamObj;
    opParam.syncMode = syncMode;
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.aicpuCacheEnable = GetExternalInputAicpuCacheEnable();
    opParam.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    // 用于inplace支持重执行场景的图模式归一至单算子模式
    retryOrigWorkflowMode_ = GetWorkflowMode();
    bool isHcclOpInplace
        = IsHcclOpInplace(HcclCMDType::HCCL_CMD_ALLREDUCE, opParam, userRank_, userRankSize_, isInplaceStatus_);
    if (aicpuUnfoldMode && retryEnable_ && isHcclOpInplace) {
        HCCL_DEBUG(
            "The retry with inplace case is expected to be supported, "
            "aicpuUnfoldMode[%d], retryEnable_[%d], isHcclOpInplace[%d], "
            "therefore HcclWorkflowMode is converted from [%d] to HCCL_WORKFLOW_MODE_OP_BASE",
            aicpuUnfoldMode, retryEnable_, isHcclOpInplace, static_cast<u8>(retryOrigWorkflowMode_));
        CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
    }

    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_ALLREDUCE, opParam));

    RestorePreSyncMode(preSyncMode, syncMode);
    CHK_RET(SetWorkflowMode(retryOrigWorkflowMode_));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllReduceAicpuUnfold(
    const std::string& tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op,
    HcclRtStream stream)
{
    Stream streamObj(stream);
    u32 perDataSize = SIZE_TABLE[dataType];
    u64 totalSize = count * perDataSize;
    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = totalSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = totalSize;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = op;
    opParam.stream = streamObj;
    opParam.isCapture = StreamIsCapture(stream);
    opParam.syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
    opParam.aicpuCacheEnable = GetExternalInputAicpuCacheEnable();
    AlgType algType;
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING;
    algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    auto slaveStreams = opParam.isCapture ? std::vector<Stream>{opStream_} : std::vector<Stream>{};
    CaptureSlaveStreams(streamObj.ptr(), slaveStreams);
    CHK_RET(RegisterDfxInfo(opParam, algType, slaveStreams));
    HcclResult ret;
    if (!IsExistCommRes(tag)) {
        uint64_t streamMode = 0;
        CHK_RET(hrtStreamGetMode(stream, &streamMode));
        rtStream_t aicpuStream;
        ret = Mc2AiCpuStreamAllocAndGet(streamMode, aicpuStream);
        void* commContext = nullptr;
        ret = CreateCommResource(tag, aicpuStream, true, &commContext);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[hcclImpl][CreateComm]create aicpu unfold comminfo by tag[%s] failed. return[%d]", tag.c_str(), ret);
            return ret;
        }
    }
    AicpuOpTiling opTilingInfo;
    std::string kernelName = "RunAicpuRpcSrvLaunch";
    ret = AicpuKfcTilingDataLaunch(opParam, HcclCMDType::HCCL_CMD_ALLREDUCE, commContext_, kernelName, opTilingInfo);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[hcclImpl][TilingData]aicpu unfold tiling data launch failed. return[%d] inputPtr[%p]"
            "outputPtr[%p] count[%llu] dataType[%s] op[%s]",
            ret, inputPtr, outputPtr, count, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(op).c_str());
        return ret;
    }
    CHK_RET(UnRegisterDfxInfo(opParam, slaveStreams));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllReduceOutPlace(
    const std::string& tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op,
    HcclRtStream stream, SyncMode syncMode)
{
    CHK_RET(CheckSuspendingStatus());
    const u32 RANK_SIZE_TWO = 2;
    bool aicpuUnfoldMode = false;
    if (GetAicpuUnfoldConfig() == true && IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op)) {
        if (userRankSize_ >= RANK_SIZE_TWO && Is310P3Common(isHaveCpuRank_, deviceType_)) {
            HcclResult ret = AllReduceAicpuUnfold(tag, inputPtr, outputPtr, count, dataType, op, stream);
            CHK_PRT_RET(
                (ret != HCCL_SUCCESS),
                HCCL_ERROR(
                    "[HcclCommunicator][AllReduce]errNo[0x%016llx] tag[%s], AllReduce aicpu unfold failed",
                    HCCL_ERROR_CODE(ret), tag.c_str()),
                ret);

            return HCCL_SUCCESS;
        }
        if ((deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
            aicpuUnfoldMode = true;
        }
    }

    bool isCapture = StreamIsCapture(stream);

    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][AllReduceOutPlace]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    // 设置notify wait模式
    SyncMode preSyncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
    GetAndSetSyncMode(preSyncMode, syncMode);

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32>& ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    std::vector<u32>& vnicRanksPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, vnicRanksPorts, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 totalSize = count * perDataSize;

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = totalSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = totalSize;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = op;
    opParam.stream = streamObj;
    opParam.syncMode = syncMode;
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.aicpuCacheEnable = GetExternalInputAicpuCacheEnable();
    opParam.isCapture = isCapture;
    opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
    opParam.opType = HcclCMDType::HCCL_CMD_ALLREDUCE;

    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_ALLREDUCE, opParam));

    RestorePreSyncMode(preSyncMode, syncMode);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AlltoAllV(
    const void* sendBuf, const void* sendCounts, const void* sdispls, HcclDataType sendType, const void* recvBuf,
    const void* recvCounts, const void* rdispls, HcclDataType recvType, rtStream_t stream, const std::string& tag)
{
    CHK_RET(CheckSuspendingStatus());

    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][AlltoAllV]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    if (IsNeedNicInit()) {
        HCCL_INFO("InitNic.");
        CHK_RET(InitNic());
    }

    bool isCapture = StreamIsCapture(stream);

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32>& ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    std::vector<u32>& vnicRanksPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, vnicRanksPorts, isSetHDCModeInfo_, isUseRankPort_);

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = const_cast<void*>(sendBuf);
    opParam.outputPtr = const_cast<void*>(recvBuf);
    opParam.All2AllDataDes.sendType = sendType;
    opParam.All2AllDataDes.recvType = recvType;
    opParam.All2AllDataDes.sendCounts = const_cast<void*>(sendCounts);
    opParam.All2AllDataDes.recvCounts = const_cast<void*>(recvCounts);
    opParam.All2AllDataDes.sdispls = const_cast<void*>(sdispls);
    opParam.All2AllDataDes.rdispls = const_cast<void*>(rdispls);
    opParam.stream = streamObj;
    opParam.opType = HcclCMDType::HCCL_CMD_ALLTOALLV;
    opParam.aicpuUnfoldMode = EnableAicpuUnfold(isCapture);
    opParam.aicpuCacheEnable = GetExternalInputAicpuCacheEnable();
    opParam.isCapture = isCapture;

    if (UNLIKELY(GetDebugConfig() & HCCL_ALG)) {
        for (u32 i = 0; i < userRankSize_; i++) {
            HCCL_CONFIG_INFO(
                HCCL_ALG,
                "[HcclCommunicator][AlltoAllV] rank[%u], sendCounts[%llu], sendDispls[%llu] "
                "recvCounts[%llu], recvDispls[%llu]",
                userRank_, *(static_cast<const u64*>(opParam.All2AllDataDes.sendCounts) + i),
                *(static_cast<const u64*>(opParam.All2AllDataDes.sdispls) + i),
                *(static_cast<const u64*>(opParam.All2AllDataDes.recvCounts) + i),
                *(static_cast<const u64*>(opParam.All2AllDataDes.rdispls) + i));
        }
    }

    CHK_RET(ExecOpAlltoAll(HcclCMDType::HCCL_CMD_ALLTOALLV, opParam));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AlltoAllVOutPlace(
    const void* sendBuf, const void* sendCounts, const void* sdispls, HcclDataType sendType, const void* recvBuf,
    const void* recvCounts, const void* rdispls, HcclDataType recvType, rtStream_t stream, const std::string& tag)
{
    CHK_RET(CheckSuspendingStatus());
    CHK_PRT_RET(
        Is310P3Common(isHaveCpuRank_, deviceType_),
        HCCL_RUN_INFO("[AlltoAllVOutPlace]This method cannot be invoked in the current scenario."), HCCL_SUCCESS);
    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][AlltoAllVOutPlace]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    if (IsNeedNicInit()) {
        HCCL_INFO("InitNic.");
        CHK_RET(InitNic());
    }

    bool isCapture = StreamIsCapture(stream);

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32>& ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    std::vector<u32>& vnicRanksPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, vnicRanksPorts, isSetHDCModeInfo_, isUseRankPort_);

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = const_cast<void*>(sendBuf);
    opParam.outputPtr = const_cast<void*>(recvBuf);
    opParam.All2AllDataDes.sendType = sendType;
    opParam.All2AllDataDes.recvType = recvType;
    opParam.All2AllDataDes.sendCounts = const_cast<void*>(sendCounts);
    opParam.All2AllDataDes.recvCounts = const_cast<void*>(recvCounts);
    opParam.All2AllDataDes.sdispls = const_cast<void*>(sdispls);
    opParam.All2AllDataDes.rdispls = const_cast<void*>(rdispls);
    opParam.stream = streamObj;
    opParam.opType = HcclCMDType::HCCL_CMD_ALLTOALLV;
    opParam.aicpuUnfoldMode = EnableAicpuUnfold(isCapture);
    opParam.aicpuCacheEnable = GetExternalInputAicpuCacheEnable();
    opParam.isCapture = isCapture;

    if (UNLIKELY(GetDebugConfig() & HCCL_ALG)) {
        for (u32 i = 0; i < userRankSize_; i++) {
            HCCL_CONFIG_INFO(
                HCCL_ALG,
                "[HcclCommunicator][AlltoAllVOutPlace] rank[%u], sendCounts[%llu],"
                "sendDispls[%llu], recvCounts[%llu], recvDispls[%llu]",
                userRank_, *(static_cast<const u64*>(opParam.All2AllDataDes.sendCounts) + i),
                *(static_cast<const u64*>(opParam.All2AllDataDes.sdispls) + i),
                *(static_cast<const u64*>(opParam.All2AllDataDes.recvCounts) + i),
                *(static_cast<const u64*>(opParam.All2AllDataDes.rdispls) + i));
        }
    }

    CHK_RET(ExecOpAlltoAll(HcclCMDType::HCCL_CMD_ALLTOALLV, opParam));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AlltoAllVC(
    const void* sendBuf, const void* sendCountMatrix, HcclDataType sendType, const void* recvBuf, HcclDataType recvType,
    rtStream_t stream, const std::string& tag)
{
    CHK_RET(CheckSuspendingStatus());
    if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
        HCCL_ERROR(
            "[%s][%s]AlltoAllVC is not supported", LOG_KEYWORDS_TASK_EXEC.c_str(), LOG_KEYWORDS_NOT_SUPPORTED.c_str());
        return HCCL_E_NOT_SUPPORT;
    }
    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][AlltoAllVC]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    if (IsNeedNicInit()) {
        HCCL_INFO("InitNic.");
        CHK_RET(InitNic());
    }

    bool isCapture = StreamIsCapture(stream);

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32>& ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    std::vector<u32>& vnicRanksPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, vnicRanksPorts, isSetHDCModeInfo_, isUseRankPort_);

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = const_cast<void*>(sendBuf);
    opParam.outputPtr = const_cast<void*>(recvBuf);
    opParam.All2AllDataDes.sendType = sendType;
    opParam.All2AllDataDes.recvType = recvType;
    opParam.All2AllDataDes.sendCountMatrix = const_cast<void*>(sendCountMatrix);
    opParam.stream = streamObj;
    opParam.opType = HcclCMDType::HCCL_CMD_ALLTOALLVC;
    opParam.aicpuUnfoldMode = EnableAicpuUnfold(isCapture);
    opParam.aicpuCacheEnable = GetExternalInputAicpuCacheEnable();
    opParam.isCapture = isCapture;

    if (UNLIKELY(GetDebugConfig() & HCCL_ALG)) {
        for (u32 i = 0; i < userRankSize_; i++) {
            for (u32 j = 0; j < userRankSize_; j++) {
                HCCL_CONFIG_DEBUG(
                    HCCL_ALG,
                    "[HcclCommunicator][AlltoAllVC] usrRank[%u] rank[%u] to remoteRank[%u], "
                    "sendCounts[%llu]",
                    userRank_, i, j,
                    *(static_cast<const u64*>(opParam.All2AllDataDes.sendCountMatrix) + i * userRankSize_ + j));
            }
        }
    }

    CHK_RET(ExecOpAlltoAll(HcclCMDType::HCCL_CMD_ALLTOALLVC, opParam));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AlltoAllVCOutPlace(
    const void* sendBuf, const void* sendCountMatrix, HcclDataType sendType, const void* recvBuf, HcclDataType recvType,
    rtStream_t stream, const std::string& tag)
{
    CHK_RET(CheckSuspendingStatus());
    CHK_PRT_RET(
        Is310P3Common(isHaveCpuRank_, deviceType_),
        HCCL_RUN_INFO("[AlltoAllVCOutPlace]This method cannot be invoked in the current scenario."), HCCL_SUCCESS);

    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][AlltoAllVCOutPlace]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    if (IsNeedNicInit()) {
        HCCL_INFO("InitNic");
        CHK_RET(InitNic());
    }

    bool isCapture = StreamIsCapture(stream);

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32>& ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    std::vector<u32>& vnicRanksPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, vnicRanksPorts, isSetHDCModeInfo_, isUseRankPort_);

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = const_cast<void*>(sendBuf);
    opParam.outputPtr = const_cast<void*>(recvBuf);
    opParam.All2AllDataDes.sendType = sendType;
    opParam.All2AllDataDes.recvType = recvType;
    opParam.All2AllDataDes.sendCountMatrix = const_cast<void*>(sendCountMatrix);
    opParam.stream = streamObj;
    opParam.opType = HcclCMDType::HCCL_CMD_ALLTOALLVC;
    opParam.aicpuUnfoldMode = EnableAicpuUnfold(isCapture);
    opParam.aicpuCacheEnable = GetExternalInputAicpuCacheEnable();
    opParam.isCapture = isCapture;

    if (UNLIKELY(GetDebugConfig() & HCCL_ALG)) {
        for (u32 i = 0; i < userRankSize_; i++) {
            for (u32 j = 0; j < userRankSize_; j++) {
                HCCL_CONFIG_DEBUG(
                    HCCL_ALG,
                    "[HcclCommunicator][AlltoAllVCOutPlace] usrRank[%u] rank[%u]"
                    "to remoteRank[%u], sendCounts[%llu]",
                    userRank_, i, j,
                    *(static_cast<const u64*>(opParam.All2AllDataDes.sendCountMatrix) + i * userRankSize_ + j));
            }
        }
    }

    CHK_RET(ExecOpAlltoAll(HcclCMDType::HCCL_CMD_ALLTOALLVC, opParam));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AlltoAll(
    const void* sendBuf, u64 sendCount, HcclDataType sendType, const void* recvBuf, u64 recvCount,
    HcclDataType recvType, rtStream_t stream, const std::string& tag)
{
    CHK_RET(CheckSuspendingStatus());
    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][AlltoAll]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    if (IsNeedNicInit()) {
        HCCL_INFO("InitNic.");
        CHK_RET(InitNic());
    }

    bool isCapture = StreamIsCapture(stream);

    CHK_RET(callbackTask_->CallbackRegStream(stream));

    // 生成sendCountMatrix矩阵，alltoall的底层实现走alltoallvc
    std::vector<u64> sendCountMatrix(userRankSize_ * userRankSize_, sendCount);

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = const_cast<void*>(sendBuf);
    opParam.outputPtr = const_cast<void*>(recvBuf);
    opParam.All2AllDataDes.sendType = sendType;
    opParam.All2AllDataDes.recvType = recvType;
    opParam.All2AllDataDes.sendCount = sendCount;
    opParam.All2AllDataDes.recvCount = recvCount;
    opParam.All2AllDataDes.sendCountMatrix = static_cast<void*>(sendCountMatrix.data());
    opParam.stream = Stream(stream);
    opParam.opType = HcclCMDType::HCCL_CMD_ALLTOALL;
    opParam.aicpuUnfoldMode = false;
    opParam.aicpuCacheEnable = 0;
    opParam.isCapture = isCapture;
    opParam.inputSize = sendCount * SIZE_TABLE[sendType] * userRankSize_;
    opParam.outputSize = recvCount * SIZE_TABLE[recvType] * userRankSize_;
    if (EnableAicpuUnfold(isCapture)) {
        opParam.aicpuUnfoldMode = true;
        opParam.aicpuCacheEnable = GetExternalInputAicpuCacheEnable();
    }

    std::vector<u32>& ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    std::vector<u32>& vnicRanksPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, vnicRanksPorts, isSetHDCModeInfo_, isUseRankPort_);
    CHK_RET(ExecOpAlltoAll(HcclCMDType::HCCL_CMD_ALLTOALL, opParam));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Broadcast(
    const std::string& tag, void* ptr, u64 count, HcclDataType dataType, u32 root, HcclRtStream stream)
{
    CHK_RET(CheckSuspendingStatus());
    bool aicpuUnfoldMode = false;
    if (GetAicpuUnfoldConfig() == true && deviceType_ == DevType::DEV_TYPE_910_93 && (userRankSize_ != 1)) {
        aicpuUnfoldMode = true;
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][Broadcast]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32>& ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    std::vector<u32>& vnicRanksPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, vnicRanksPorts, isSetHDCModeInfo_, isUseRankPort_);
    u32 perDataSize = SIZE_TABLE[dataType];
    u64 totalSize = count * perDataSize;

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = ptr;
    opParam.outputPtr = ptr;
    opParam.inputSize = totalSize;
    opParam.outputSize = totalSize;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.root = root;
    opParam.stream = streamObj;
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.aicpuCacheEnable = GetExternalInputAicpuCacheEnable();
    opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
    opParam.opType = HcclCMDType::HCCL_CMD_BROADCAST;

    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_BROADCAST, opParam));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BroadcastOutPlace(
    const std::string& tag, void* ptr, u64 count, HcclDataType dataType, u32 root, HcclRtStream stream)
{
    CHK_RET(CheckSuspendingStatus());
    bool aicpuUnfoldMode = false;
    if (GetAicpuUnfoldConfig() == true && deviceType_ == DevType::DEV_TYPE_910_93 && (userRankSize_ != 1)) {
        aicpuUnfoldMode = true;
    }

    CHK_PRT_RET(
        Is310P3Common(isHaveCpuRank_, deviceType_),
        HCCL_RUN_INFO("[BroadcastOutPlace]This method cannot be invoked in the current scenario."), HCCL_SUCCESS);

    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][BroadcastOutPlace]errNo[0x%016llx] hccl init must be called before"
            " call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    bool isCapture = StreamIsCapture(stream);

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32>& ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    std::vector<u32>& vnicRanksPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, vnicRanksPorts, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 totalSize = count * perDataSize;

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = ptr;
    opParam.outputPtr = ptr;
    opParam.inputSize = totalSize;
    opParam.outputSize = totalSize;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.root = root;
    opParam.stream = streamObj;
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.aicpuCacheEnable = GetExternalInputAicpuCacheEnable();
    opParam.isCapture = isCapture;
    opParam.opType = HcclCMDType::HCCL_CMD_BROADCAST;

    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_BROADCAST, opParam));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Scatter(
    const std::string& tag, void* inputPtr, void* outputPtr, u64 recvCount, HcclDataType dataType, u32 root,
    HcclRtStream stream)
{
    CHK_RET(CheckSuspendingStatus());
    if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
        HCCL_ERROR(
            "[%s][%s]Scatter Not Supported Yet", LOG_KEYWORDS_TASK_EXEC.c_str(), LOG_KEYWORDS_NOT_SUPPORTED.c_str());
        return HCCL_E_NOT_SUPPORT;
    }
    bool aicpuUnfoldMode = false;
    if (GetAicpuUnfoldConfig() == true && deviceType_ == DevType::DEV_TYPE_910_93 && (userRankSize_ != 1)) {
        aicpuUnfoldMode = true;
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][Scatter]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32>& ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    std::vector<u32>& vnicRanksPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, vnicRanksPorts, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 outputSize = recvCount * perDataSize;
    u64 totalSize = outputSize * userRankSize_;

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = totalSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = totalSize;
    opParam.DataDes.count = recvCount;
    opParam.DataDes.dataType = dataType;
    opParam.stream = streamObj;
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.aicpuCacheEnable = GetExternalInputAicpuCacheEnable();
    opParam.root = root;
    opParam.opType = HcclCMDType::HCCL_CMD_SCATTER;
    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_SCATTER, opParam));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ScatterOutPlace(
    const std::string& tag, void* inputPtr, void* outputPtr, u64 recvCount, HcclDataType dataType, u32 root,
    HcclRtStream stream)
{
    CHK_RET(CheckSuspendingStatus());
    if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
        HCCL_ERROR(
            "[%s][%s]ScatterOutPlace Not Supported Yet", LOG_KEYWORDS_TASK_EXEC.c_str(),
            LOG_KEYWORDS_NOT_SUPPORTED.c_str());
        return HCCL_E_NOT_SUPPORT;
    }

    bool aicpuUnfoldMode = false;
    if (GetAicpuUnfoldConfig() == true && (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
        aicpuUnfoldMode = true;
    }

    bool isCapture = StreamIsCapture(stream);

    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][ScatterOutPlace]errNo[0x%016llx] hccl init must be called before"
            " call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32>& ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    std::vector<u32>& vnicRanksPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, vnicRanksPorts, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 outputSize = recvCount * perDataSize;
    u64 totalSize = outputSize * userRankSize_;

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = totalSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = totalSize;
    opParam.DataDes.count = recvCount;
    opParam.DataDes.dataType = dataType;
    opParam.stream = streamObj;
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.aicpuCacheEnable = GetExternalInputAicpuCacheEnable();
    opParam.isCapture = isCapture;
    opParam.root = root;
    opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
    opParam.opType = HcclCMDType::HCCL_CMD_SCATTER;

    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_SCATTER, opParam));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Reduce(
    const std::string& tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op,
    u32 root, HcclRtStream stream)
{
    CHK_RET(CheckSuspendingStatus());
    if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
        HCCL_ERROR(
            "[%s][%s]Reduce Not Supported Yet", LOG_KEYWORDS_TASK_EXEC.c_str(), LOG_KEYWORDS_NOT_SUPPORTED.c_str());
        return HCCL_E_NOT_SUPPORT;
    }
    bool aicpuUnfoldMode = false;
    if (GetAicpuUnfoldConfig() == true && IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op)
        && (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
        aicpuUnfoldMode = true;
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][Reduce]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 totalSize = count * perDataSize;
    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = totalSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = totalSize;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = op;
    opParam.root = root;
    opParam.stream = streamObj;
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.aicpuCacheEnable = GetExternalInputAicpuCacheEnable();
    opParam.opType = HcclCMDType::HCCL_CMD_REDUCE;

    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_REDUCE, opParam));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ReduceOutPlace(
    const std::string& tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op,
    u32 root, HcclRtStream stream)
{
    CHK_RET(CheckSuspendingStatus());
    CHK_PRT_RET(
        Is310P3Common(isHaveCpuRank_, deviceType_),
        HCCL_RUN_INFO("[ReduceOutPlace]This method cannot be invoked in the current scenario."), HCCL_SUCCESS);

    bool aicpuUnfoldMode = false;
    if (GetAicpuUnfoldConfig() == true && IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op)
        && (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
        aicpuUnfoldMode = true;
    }

    bool isCapture = StreamIsCapture(stream);

    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][ReduceOutPlace]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32>& ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    std::vector<u32>& vnicRanksPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, vnicRanksPorts, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 totalSize = count * perDataSize;
    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = totalSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = totalSize;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = op;
    opParam.root = root;
    opParam.stream = streamObj;
    opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.aicpuCacheEnable = GetExternalInputAicpuCacheEnable();
    opParam.isCapture = isCapture;
    opParam.opType = HcclCMDType::HCCL_CMD_REDUCE;

    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_REDUCE, opParam));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ReduceScatter(
    const std::string& tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op,
    HcclRtStream stream, [[maybe_unused]] HcomCollOpInfo* opInfo)
{
    CHK_RET(CheckSuspendingStatus());
    bool aicpuUnfoldMode = false;
    if (GetAicpuUnfoldConfig() == true && IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op)
        && (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
        aicpuUnfoldMode = true;
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][ReduceScatter]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32>& ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    std::vector<u32>& vnicRanksPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, vnicRanksPorts, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = userRankSize_ * count * perDataSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = count * perDataSize;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = op;
    opParam.stream = streamObj;
    opParam.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.aicpuCacheEnable = GetExternalInputAicpuCacheEnable();
    // 用于inplace支持重执行场景的图模式归一至单算子模式
    retryOrigWorkflowMode_ = GetWorkflowMode();
    bool isHcclOpInplace
        = IsHcclOpInplace(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, opParam, userRank_, userRankSize_, isInplaceStatus_);
    if (aicpuUnfoldMode && retryEnable_ && isHcclOpInplace) {
        HCCL_DEBUG(
            "The retry with inplace case is expected to be supported, "
            "aicpuUnfoldMode[%d], retryEnable_[%d], isHcclOpInplace[%d], "
            "therefore HcclWorkflowMode is converted from [%d] to HCCL_WORKFLOW_MODE_OP_BASE",
            aicpuUnfoldMode, retryEnable_, isHcclOpInplace, static_cast<u8>(retryOrigWorkflowMode_));
        CHK_RET(SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
    }

    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, opParam));

    CHK_RET(SetWorkflowMode(retryOrigWorkflowMode_));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ReduceScatterOutPlace(
    const std::string& tag, void* inputPtr, void* outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op,
    HcclRtStream stream)
{
    CHK_RET(CheckSuspendingStatus());
    if (userRankSize_ > 1) {
        CHK_RET(CreateCommCCLbuffer());
    }

    bool aicpuUnfoldMode = false;
    if (GetAicpuUnfoldConfig() == true
        && IsSupportSDMAReduce(
            cclBufferManager_.GetInCCLbuffer().ptr(), cclBufferManager_.GetOutCCLbuffer().ptr(), dataType, op)
        && (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
        aicpuUnfoldMode = true;
    }

    bool isCapture = StreamIsCapture(stream);

    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][ReduceScatterOutPlace]errNo[0x%016llx] hccl init must be called before"
            " call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32>& ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    std::vector<u32>& vnicRanksPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, vnicRanksPorts, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = userRankSize_ * count * perDataSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = count * perDataSize;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = op;
    opParam.stream = streamObj;
    opParam.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.aicpuCacheEnable = GetExternalInputAicpuCacheEnable();
    opParam.isCapture = isCapture;
    opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();

    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, opParam));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ReduceScatterV(
    const std::string& tag, void* inputPtr, const void* inputCounts, const void* inputDispls, void* outputPtr,
    u64 outputCount, HcclDataType dataType, HcclReduceOp op, HcclRtStream stream,
    [[maybe_unused]] HcomCollOpInfo* opInfo)
{
    CHK_RET(CheckSuspendingStatus());
    if (userRankSize_ == 1) {
        // rankSize为1时，退化为ReduceScatter
        return ReduceScatter(tag, inputPtr, outputPtr, outputCount, dataType, op, stream);
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][ReduceScatterV]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32>& ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    std::vector<u32>& vnicRanksPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, vnicRanksPorts, isSetHDCModeInfo_, isUseRankPort_);

    const bool aicpuUnfoldMode = GetAicpuUnfoldConfig() && IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op)
                                 && (deviceType_ == DevType::DEV_TYPE_910_93);

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 inputSize = 0;
    const u64* counts = static_cast<const u64*>(inputCounts);
    for (u32 i = 0; i < userRankSize_; i++) {
        inputSize += counts[i] * perDataSize;
    }
    CHK_PRT_RET(inputSize == 0, HCCL_WARNING("inputSize is 0, return ReduceScatterV success"), HCCL_SUCCESS);

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = inputSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = outputCount * perDataSize;
    opParam.srcRank = userRank_; // rankId for access counts
    opParam.VDataDes.counts = const_cast<void*>(inputCounts);
    opParam.VDataDes.displs = const_cast<void*>(inputDispls);
    opParam.VDataDes.dataType = dataType;
    opParam.reduceType = op;
    opParam.stream = streamObj;
    opParam.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V;
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.aicpuCacheEnable = GetExternalInputAicpuCacheEnable();

    if (UNLIKELY(GetDebugConfig() & HCCL_ALG)) {
        for (u32 i = 0; i < userRankSize_; i++) {
            HCCL_CONFIG_DEBUG(
                HCCL_ALG,
                "[HcclCommunicator][ReduceScatterV]userRank_[%u], rankIdx[%u], inputCounts[%llu], inputDispls[%llu]",
                userRank_, i, counts[i], static_cast<const u64*>(inputDispls)[i]);
        }
    }

    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, opParam));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ReduceScatterVOutPlace(
    const std::string& tag, void* inputPtr, void* outputPtr, const void* inputCounts, const void* inputDispls,
    u64 outputCount, HcclDataType dataType, HcclReduceOp op, HcclRtStream stream)
{
    CHK_RET(CheckSuspendingStatus());
    if (userRankSize_ == 1) {
        // rankSize为1时，退化为ReduceScatter
        return ReduceScatterOutPlace(tag, inputPtr, outputPtr, outputCount, dataType, op, stream);
    }

    CHK_RET(CreateCommCCLbuffer());
    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][ReduceScatterVOutPlace]errNo[0x%016llx] hccl init must be called before"
            " call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    bool isCapture = StreamIsCapture(stream);

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32>& ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    std::vector<u32>& vnicRanksPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, vnicRanksPorts, isSetHDCModeInfo_, isUseRankPort_);

    const bool aicpuUnfoldMode = GetAicpuUnfoldConfig() && IsSupportSDMAReduce(inputPtr, outputPtr, dataType, op)
                                 && (deviceType_ == DevType::DEV_TYPE_910_93);

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 inputSize = 0;
    const u64* counts = static_cast<const u64*>(inputCounts);
    for (u32 i = 0; i < userRankSize_; i++) {
        inputSize += counts[i] * perDataSize;
    }

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = inputSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = outputCount * perDataSize;
    opParam.srcRank = userRank_; // rankId for access counts
    opParam.VDataDes.counts = const_cast<void*>(inputCounts);
    opParam.VDataDes.displs = const_cast<void*>(inputDispls);
    opParam.VDataDes.dataType = dataType;
    opParam.reduceType = op;
    opParam.stream = streamObj;
    opParam.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V;
    opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.aicpuCacheEnable = GetExternalInputAicpuCacheEnable();
    opParam.isCapture = isCapture;
    opParam.rankSize = userRankSize_;

    if (UNLIKELY(GetDebugConfig() & HCCL_ALG)) {
        for (u32 i = 0; i < userRankSize_; i++) {
            HCCL_CONFIG_DEBUG(
                HCCL_ALG,
                "[HcclCommunicator][ReduceScatterVOutPlace]userRank_[%u],"
                "rankIdx[%u], inputCounts[%llu], inputDispls[%llu]",
                userRank_, i, counts[i], static_cast<const u64*>(inputDispls)[i]);
        }
    }

    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, opParam));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BatchSendRecv(
    const std::string& tag, HcclSendRecvItem* sendRecvItemsPtr, u32 itemNum, rtStream_t stream)
{
    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][BatchSendRecv]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    bool aicpuUnfoldMode = false;
    if (GetAicpuUnfoldConfig() == true && (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
        aicpuUnfoldMode = true;
    }

    bool isCapture = StreamIsCapture(stream);

    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][BatchSendRecv]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }
    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32>& ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    std::vector<u32>& vnicRanksPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, vnicRanksPorts, isSetHDCModeInfo_, isUseRankPort_);
    OpParam opParam;
    opParam.tag = tag;
    opParam.stream = streamObj;
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.aicpuCacheEnable = GetExternalInputAicpuCacheEnable();
    opParam.isCapture = isCapture;
    opParam.BatchSendRecvDataDes.sendRecvItemsPtr = sendRecvItemsPtr;
    opParam.BatchSendRecvDataDes.itemNum = itemNum;
    opParam.opType = HcclCMDType::HCCL_CMD_BATCH_SEND_RECV;
    opParam.isGroupMode = isGroupMode_;
    if (isGroupMode_) {
        opParam.aicpuUnfoldMode = true; // A2的GroupSendRecv也走aicpu模式
    }

    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, opParam));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Send(
    const std::string& tag, void* inputPtr, u64 count, HcclDataType dataType, u32 destRank, rtStream_t stream,
    u32 srTag, u32 localGroupRank)
{
    CHK_RET(CheckSuspendingStatus());
    bool aicpuUnfoldMode = false;
    if (GetAicpuUnfoldConfig() == true && (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
        aicpuUnfoldMode = true;
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][Send]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 totalSize = count * perDataSize;

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = totalSize;
    opParam.outputPtr = inputPtr;
    opParam.outputSize = totalSize;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.stream = streamObj;
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.aicpuCacheEnable = GetExternalInputAicpuCacheEnable();
    opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
    opParam.dstRank = destRank;
    opParam.opType = HcclCMDType::HCCL_CMD_SEND;
    opParam.srTag = srTag;
    opParam.localGroupRank = localGroupRank;
    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_SEND, opParam));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SendOutPlace(
    const std::string& tag, void* inputPtr, u64 count, HcclDataType dataType, u32 destRank, rtStream_t stream)
{
    CHK_RET(CheckSuspendingStatus());
    bool aicpuUnfoldMode = false;
    if (GetAicpuUnfoldConfig() == true && (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
        aicpuUnfoldMode = true;
    }

    if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
        HCCL_ERROR(
            "[%s][%s]SendOutPlace is not supported", LOG_KEYWORDS_TASK_EXEC.c_str(),
            LOG_KEYWORDS_NOT_SUPPORTED.c_str());
        return HCCL_E_NOT_SUPPORT;
    }
    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][SendOutPlace]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    bool isCapture = StreamIsCapture(stream);

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32>& ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    std::vector<u32>& vnicRanksPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, vnicRanksPorts, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 totalSize = count * perDataSize;

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputPtr;
    opParam.inputSize = totalSize;
    opParam.outputPtr = inputPtr;
    opParam.outputSize = totalSize;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.stream = streamObj;
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.aicpuCacheEnable = GetExternalInputAicpuCacheEnable();
    opParam.isCapture = isCapture;
    opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
    opParam.dstRank = destRank;
    opParam.opType = HcclCMDType::HCCL_CMD_SEND;
    opParam.localGroupRank = userRank_;
    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_SEND, opParam));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Receive(
    const std::string& tag, void* outputPtr, u64 count, HcclDataType dataType, u32 srcRank, rtStream_t stream,
    u32 srTag, u32 localGroupRank)
{
    CHK_RET(CheckSuspendingStatus());
    bool aicpuUnfoldMode = false;
    if (GetAicpuUnfoldConfig() == true && (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
        aicpuUnfoldMode = true;
    }

    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][Receive]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 totalSize = count * perDataSize;

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = outputPtr;
    opParam.inputSize = totalSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = totalSize;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.stream = streamObj;
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.aicpuCacheEnable = GetExternalInputAicpuCacheEnable();
    opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
    opParam.srcRank = srcRank;
    opParam.opType = HcclCMDType::HCCL_CMD_RECEIVE;
    opParam.srTag = srTag;
    opParam.localGroupRank = localGroupRank;
    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_RECEIVE, opParam));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ReceiveOutPlace(
    const std::string& tag, void* outputPtr, u64 count, HcclDataType dataType, u32 srcRank, rtStream_t stream)
{
    CHK_RET(CheckSuspendingStatus());
    bool aicpuUnfoldMode = false;
    if (GetAicpuUnfoldConfig() == true && (deviceType_ == DevType::DEV_TYPE_910_93) && (userRankSize_ != 1)) {
        aicpuUnfoldMode = true;
    }

    if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
        HCCL_ERROR(
            "[%s][%s]ReceiveOutPlace is not supported", LOG_KEYWORDS_TASK_EXEC.c_str(),
            LOG_KEYWORDS_NOT_SUPPORTED.c_str());
        return HCCL_E_NOT_SUPPORT;
    }
    if (!IsAtomicInit()) {
        HCCL_ERROR(
            "[HcclCommunicator][ReceiveOutPlace]errNo[0x%016llx] hccl init must be called before call this function",
            HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
        return HCCL_E_UNAVAIL;
    }

    bool isCapture = StreamIsCapture(stream);

    Stream streamObj(stream);
    CHK_RET(callbackTask_->CallbackRegStream(stream));

    std::vector<u32>& ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
    std::vector<u32>& vnicRanksPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
    implAlg_->SetHDCModeInfo(rankDevicePhyIdNicInfoMap_, ranksPorts, vnicRanksPorts, isSetHDCModeInfo_, isUseRankPort_);

    u32 perDataSize = SIZE_TABLE[dataType];
    u64 totalSize = count * perDataSize;

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = outputPtr;
    opParam.inputSize = totalSize;
    opParam.outputPtr = outputPtr;
    opParam.outputSize = totalSize;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.stream = streamObj;
    opParam.aicpuUnfoldMode = aicpuUnfoldMode;
    opParam.aicpuCacheEnable = GetExternalInputAicpuCacheEnable();
    opParam.isCapture = isCapture;
    opParam.opBaseAtraceInfo = opBaseAtraceInfo_.get();
    opParam.srcRank = srcRank;
    opParam.opType = HcclCMDType::HCCL_CMD_RECEIVE;
    opParam.localGroupRank = userRank_;
    CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_RECEIVE, opParam));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::RegressCalPreOp(
    AlltoAllOperator*& alltoAllOperator, const OpParam& opParam, std::unique_ptr<PreProcessMetaInfo>& preMetaInfo)
{
    HCCL_INFO("Run with Graph, alloc new stream");
    Stream stream(StreamType::STREAM_TYPE_ONLINE);
    return RegressCalPreOp(alltoAllOperator, opParam, preMetaInfo, stream);
}

HcclResult HcclCommunicator::RegressCalPreOp(
    AlltoAllOperator*& alltoAllOperator, [[maybe_unused]] const OpParam& opParam,
    std::unique_ptr<PreProcessMetaInfo>& preMetaInfo, Stream& preProcessStream)
{
    OpParam preProcessOpParam;
    HcclWorkflowMode mode = GetWorkflowMode();
    CHK_PRT_RET(
        mode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_RESERVED, HCCL_ERROR("Invalid Workflow Mode[%d]", mode),
        HCCL_E_INTERNAL);

    // h to d
    CHK_RET(SetInfoToDevice(preMetaInfo, preProcessStream));
    // opParam准备
    CHK_RET(alltoAllOperator->PreparePreOpParam(preProcessOpParam, preMetaInfo, preProcessStream));

    // 回归调用其它算子
    HCCL_INFO(
        "[HcclCommunicator][RegressCalPreOp] Regression calls other operators and opType[%u]", preMetaInfo->opType);
    CHK_RET(ExecOp(preMetaInfo->opType, preProcessOpParam));
    CHK_RET(hcclStreamSynchronize(preProcessStream.ptr(), commConfig_.GetConfigExecTimeOut()));
    HCCL_DEBUG("[HcclCommunicator][RegressCalPreOp] preProcess tag[%s].", preProcessOpParam.tag.c_str());
    SetWorkflowMode(mode);

    // d to h
    HostMem hostCollectBuffer = HostMem::alloc(preMetaInfo->outputSize);
    CHK_PTR_NULL(hostCollectBuffer.ptr());
    CHK_RET(GetInfoFromDevice(preMetaInfo, mode, hostCollectBuffer));

    hostCollectBuffer_ = hostCollectBuffer;
    alltoAllOperator->SetPreProcessResult(std::move(hostCollectBuffer));
    HCCL_INFO("[HcclCommunicator][RegressCalPreOp] run success!");
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SaveRankInfoHasLinked(const AlgResourceRequest& resRequest)
{
    for (auto& levelNSubCommTransport : resRequest.opTransport) {
        for (auto& singleSubCommTransport : levelNSubCommTransport) {
            for (auto& transportRequest : singleSubCommTransport.transportRequests) {
                if (transportRequest.isValid) {
                    ranksLinked_.insert(transportRequest.remoteUserRank);
                    HCCL_INFO(
                        "[HcclCommunicator][SaveRankInfoHasLinked]Insert remote Rank[%u] to ranksLinked Set.",
                        transportRequest.remoteUserRank);
                }
            }
        }
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetCacheMap(
    std::unique_ptr<CollAlgOperator>& algOperator, OpParam& opParam, AlgType& algType, bool selectAivAlg,
    std::string& newTag)
{
    HcclCacheInfo cacheInfo;
    CHK_RET(algOperator->GetCache(cacheInfo));
    if (cacheInfo.isUseCache == false) {
        return HCCL_SUCCESS;
    }
    cacheInfo.algType = algType;
    cacheInfo.selectAivAlg = selectAivAlg;
    cacheInfo.newTag = newTag;

    if (hcclCacheMap_.size() > CACHEMAP_MAXSIZE) {
        size_t clearCount = static_cast<size_t>(CACHEMAP_MAXSIZE * CACHEMAP_CLEARPERCENT);
        for (auto it = hcclCacheMap_.begin(); clearCount > 0 && it != hcclCacheMap_.end(); clearCount--) {
            it = hcclCacheMap_.erase(it);
        }
    }

    hcclCacheMap_.emplace(std::make_pair(opParam, std::move(cacheInfo)));

    HCCL_INFO(
        "[HcclCommunicator][GetCacheMap] algType %s, selectAivAlg %d, newTag %s", AlgTypeToStr(algType).c_str(),
        selectAivAlg, newTag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ExecOpCache(HcclCMDType opType, OpParam& opParam, HcclCacheInfo& cacheInfo)
{
    // 可用核数也需要作为key的一部分，防止cache中拿出来的和计算出来的实际核数不一致
    // cache目前仅支持executor的kernel为1的情况
    cacheInfo.resourceArgs.buffersIn = cacheInfo.buffersIn;
    cacheInfo.resourceArgs.buffersOut = cacheInfo.buffersOut;
    cacheInfo.resourceArgs.stream = opParam.stream.ptr(); // 刷新cache下发的stream
    cacheInfo.opArgs.input = opParam.inputPtr;
    cacheInfo.opArgs.output = opParam.outputPtr;
    AlgType& algType = cacheInfo.algType;
    bool selectAivAlg = cacheInfo.selectAivAlg;
    std::string newTag = cacheInfo.newTag;
    HcclResult ret = HCCL_SUCCESS;
    // 更新aivtag
    GetAivTag(1, opParam.isCapture, cacheInfo.resourceArgs.aivTag);
    HCCL_INFO(
        "[HcclCommunicator][ExecOpCache]buffersIn[%p] buffersOut[%p] tag[%s] opType[%d] "
        "deterministic [%u] count[%llu] op[%d] userRank[%u] aiv tag [%d] stream [%d]",
        cacheInfo.buffersIn, cacheInfo.buffersOut, identifier_.c_str(), opType, opParam.deterministic,
        cacheInfo.opArgs.count, cacheInfo.opArgs.op, userRank_, cacheInfo.resourceArgs.aivTag, opParam.stream.id());
    CHK_RET(HandleAclGraphFirstOpAivBuff(opParam.stream.ptr()));
    // 保留dfx
    CHK_RET(RegisterDfxInfo(opParam, algType, resMap_[newTag].slaveStreams, selectAivAlg));
    // 头计数
    CHK_RET(StarsCounter(dispatcher_, opParam.stream, HEAD, opParam.aicpuUnfoldMode, retryEnable_, selectAivAlg));
    u64 dataSize
        = (opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALL ?
               opParam.All2AllDataDes.sendCount * SIZE_TABLE[opParam.All2AllDataDes.sendType] :
               0);
    if (opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V || opType == HcclCMDType::HCCL_CMD_ALLGATHER_V
        || (opType == HcclCMDType::HCCL_CMD_ALLTOALL && dataSize >= AIV_ALL_TO_ALL_BIG_SIZE)) {
        ret = ExecuteKernelLaunch(
            cacheInfo.opArgs, cacheInfo.topoArgs, cacheInfo.resourceArgs, cacheInfo.algArgs, cacheInfo.extraArgs,
            cacheInfo.profilingInfo);
    } else {
        ret = ExecuteKernelLaunch(
            cacheInfo.opArgs, cacheInfo.topoArgs, cacheInfo.resourceArgs, cacheInfo.algArgs, cacheInfo.profilingInfo);
    }
    // 刷新核数
    numBlocks_ = cacheInfo.resourceArgs.numBlocks;
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ExecOpCache]launch aiv failed, return[%d]", ret), ret);
    CHK_RET(StarsCounter(dispatcher_, opParam.stream, TAIL, opParam.aicpuUnfoldMode, retryEnable_, selectAivAlg));
    CHK_RET(UnRegisterDfxInfo(opParam, resMap_[newTag].slaveStreams));
    if (selectAivAlg) {
        aivClearEnable_ = false;
    }
    return HCCL_SUCCESS;
}

void HcclCommunicator::SplitBsrData(
    OpParam& opParam, std::vector<u8>& isDirectRemoteRank, std::vector<HcclSendRecvItem>& hostSendRecvInfo,
    std::vector<HcclSendRecvItem>& aicpuSendRecvInfo)
{
    u32 itemNum = opParam.BatchSendRecvDataDes.itemNum;
    isDirectRemoteRank.resize(userRankSize_);
    HCCL_INFO("[HcclCommunicator][SplitBsrData] rankSize %u", userRankSize_);
    HcclSendRecvItem* sendRecvInfo = opParam.BatchSendRecvDataDes.sendRecvItemsPtr;
    for (u32 i = 0; i < itemNum; i++) {
        if (sendRecvInfo->buf == nullptr) {
            sendRecvInfo++;
            continue;
        }
        if (remoteTransportMap_[sendRecvInfo->remoteRank] == TransportType::TRANS_TYPE_DEVICE_DIRECT) {
            // host 侧需要下发的数据
            HCCL_INFO(
                "[HcclCommunicator][SplitBsrData]host localRank %u remoteRank %u type %d sendRecvType %d count %llu",
                userRank_, sendRecvInfo->remoteRank, remoteTransportMap_[sendRecvInfo->remoteRank],
                sendRecvInfo->sendRecvType, sendRecvInfo->count);
            isDirectRemoteRank[sendRecvInfo->remoteRank] = true;
            hostSendRecvInfo.push_back(*sendRecvInfo);
        } else {
            // aicpu侧需要下发的数据
            HCCL_INFO(
                "[HcclCommunicator][SplitBsrData]aicpu localRank %u remoteRank %u type %d sendRecvType %d count %llu",
                userRank_, sendRecvInfo->remoteRank, remoteTransportMap_[sendRecvInfo->remoteRank],
                sendRecvInfo->sendRecvType, sendRecvInfo->count);
            isDirectRemoteRank[sendRecvInfo->remoteRank] = false;
            aicpuSendRecvInfo.push_back(*sendRecvInfo);
        }
        sendRecvInfo++;
    }
    HCCL_INFO(
        "[HcclCommunicator][SplitBsrData] itemNum %u hostItemNum %zu aicpuItemNum %zu", itemNum,
        hostSendRecvInfo.size(), aicpuSendRecvInfo.size());
    return;
}

bool HcclCommunicator::IsReduceWithInt64OrProd(HcclCMDType opType, const OpParam& opParam) const
{
    if (opType == HcclCMDType::HCCL_CMD_ALLREDUCE || opType == HcclCMDType::HCCL_CMD_REDUCE
        || opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
        if (opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD
            || opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT64) {
            return true;
        }
    }

    if (opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
        if (opParam.reduceType == HcclReduceOp::HCCL_REDUCE_PROD
            || opParam.VDataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT64) {
            return true;
        }
    }
    return false;
}

HcclResult HcclCommunicator::ExecOp(HcclCMDType opType, OpParam& opParam, bool isCustom)
{
    CHK_PRT_RET(
        isInvalidComm_,
        HCCL_ERROR(
            "[HcclCommunicator][%s] comm[%s], rank[%u], devId[%d], snapshot recovering, "
            "this comm is invalid, no operator is allowed to execute.",
            __func__, identifier_.c_str(), userRank_, deviceLogicId_),
        HCCL_E_UNAVAIL);

    if (retryEnable_ && needWarnAboutReduceProdInt64_ && IsReduceWithInt64OrProd(opType, opParam)) {
        HCCL_RUN_WARNING(
            "[HcclCommunicator][%s]comm[%s], opType[%d], reduceType[%d]. Reduce operators with prod operation or int64 "
            "data type. This operator type is unsupported for AICPU mode, retry disabled",
            __func__, identifier_.c_str(), opType, opParam.reduceType);
        needWarnAboutReduceProdInt64_ = false;
    }
    std::string tag = opParam.tag;
    u32 aivCoreLimit = numBlocks_;
    // 单机AIV场景下cache复用，提升下发性能
    if (implAlg_->GetAivModeConfig() && GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        if (aivCoreLimit == 0) {
            aclError acl_ret = aclrtGetResInCurrentThread(ACL_RT_DEV_RES_VECTOR_CORE, &aivCoreLimit);
            CHK_PRT_RET(
                acl_ret != ACL_SUCCESS,
                HCCL_ERROR("[HcclCommunicator][ExecOp] aclrtGetResInCurrentThread failed, ret=[%d]", acl_ret),
                HCCL_E_PARA);
        }
        opParam.deterministic = implAlg_->GetDeterministicConfig();
        opParam.aivCoreLimit = aivCoreLimit;
        auto it = hcclCacheMap_.find(opParam);
        if (it != hcclCacheMap_.end()) {
            CHK_RET(ExecOpCache(opType, opParam, it->second));
            return HCCL_SUCCESS;
        }
    }

    ForceProf(opParam.isCapture);
    opParam.supportSymmetricMemory = IsSupportSymmetricMemory(opType, opParam);
    opParam.supportZeroCopy = !opParam.supportSymmetricMemory && IsSupportZeroCopy(opParam);
    opParam.aclGraphZeroCopyEnable = GetConfigAclGraphZeroCopyEnable();
    bool isInGraphCaptureZeroCopy = false;
    zeroCopyAclGraph_->SetRetryEnable(retryEnable_);
    isInGraphCaptureZeroCopy = zeroCopyAclGraph_->SetAclGraphZeroCopyMode(
        deviceType_, opType, opParam, implAlg_.get(), cclBufferManager_.GetOutCCLbufferSize());
    if (isInGraphCaptureZeroCopy && userRankSize_ > 1) {
        CHK_RET(CreateCommCCLbuffer());
    }
    if (isShareComm_) {
        CHK_RET(ShareCCLbufferMgr::GetInstance().CheckCCLbuffConflict(cclBuffName_, opParam.stream.id()));
    }
    std::unique_ptr<CollAlgOperator> algOperator = implAlg_->GetAlgOperator(opType);
    CHK_SMART_PTR_NULL(algOperator);
    // 算法选择
    std::string algName;
    std::string newTag;
    if (opParam.aicpuUnfoldMode) {
        // 用于inplace支持重执行判断
        CHK_RET(algOperator->SetRetryEnable(retryEnable_));
    }
    if (GetExternalInputHcclAivMode()) {
        // 用于判断图模式是否清零
        CHK_RET(algOperator->SetAivClearEnable(aivClearEnable_));
    }

    std::unique_lock<std::mutex> lock(commResMutex_);
    ResourceLimit limit;
    limit.ifLimit = true;
    limit.aivCoreLimit = aivCoreLimit;
    AlgDesc algDesc;
    algDesc.isLastSelect = true;
    CHK_RET(algOperator->SelectAlg(opParam.tag, opParam, limit, algName, algDesc, newTag));
    if (isOnlyAiv_ && !algDesc.isAivMode) {
        std::string opTypeName = GetCMDTypeEnumStr(opType);
        HCCL_ERROR(
            "[HcclCommunicator][ExecOp] opType[%s] does not select aiv mode, aiv only mode is not supported.",
            opTypeName.c_str());
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_RET(PrepareZeroCopy(algName, algDesc, opParam));

    if (opParam.isCapture) {
        // aclgraph使用新的Tag，避免影响其他操作
        newTag += "_Capture";
        // aclgraph零拷贝场景下，每个算子都有单独的tag，需要记录，在graph销毁时清理相关资源
        if (isInGraphCaptureZeroCopy) {
            CHK_RET(AclgraphCallback::GetInstance().InsertNewTagToCaptureResMap(this, newTag, opParam));
            tagsRequiringHostCleanup_.insert(newTag);
        }
    }

    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && userRankSize_ > 1) {
        CHK_RET(CreateCommCCLbuffer());
    }
    if (hcclNslbDp::GetInstance().GetGlobalCommTaskId() != 0) {
        NslbDp_CollectOperTable(opType, opParam, algOperator->GetAlgType(), algName);
    }

    // 资源创建
    if ((resMap_.find(newTag) != resMap_.end()) && opParam.isCapture) {
        AlgResourceRequest resRequest;
        CHK_RET(algOperator->CalcResRequest(algName, opParam, resRequest));
        if (HasRoceTransportLinks(resRequest.opTransport)) {
            auto resTmp = resMap_[newTag];
            ++captureCnt_;
            newTag += std::to_string(captureCnt_);
            resMap_[newTag] = resTmp;
            resRequest.isInGraphCaptureZeroCopy = isInGraphCaptureZeroCopy;
            CHK_RET(CleanTransportLinks(resRequest.opTransport, resMap_[newTag].opTransportResponse));
            if (IsEnableBackupLink()) {
                CHK_RET(CleanTransportLinks(resRequest.opTransport, resMap_[newTag].opTransportResponseBackUp));
            }
            // 记录指令信息用于一致性校验
            CHK_RET(RecordOpPara(opType, opParam));
            CHK_RET(IncreAllocLink(newTag, opParam, resRequest, resMap_[newTag]));
            // 移除tag对应的指令信息
            CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(opParam.tag));
            // aclgraph零拷贝场景下，除第一个capture外，需要记录，在graph销毁时清理相关资源
            CHK_RET(AclgraphCallback::GetInstance().InsertNewTagToCaptureResMap(this, newTag, opParam));
            tagsRequiringHostCleanup_.insert(newTag);
        }
    }
    InsertNewTagToTagMap(newTag, opParam.tag);
    bool needIncreLink = false;
    // aiv算法不需要申请host和device侧的从流
    bool selectAivAlg = algDesc.isAivMode;
    if (resMap_.find(newTag) == resMap_.end()) {
        AlgResourceRequest resRequest;
        CHK_RET(algOperator->CalcResRequest(algName, opParam, resRequest));
        if (opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
            CHK_RET(SaveRankInfoHasLinked(resRequest));
        }
        resRequest.isInGraphCaptureZeroCopy = isInGraphCaptureZeroCopy;
        CHK_RET(RecordOpPara(opType, opParam));
        HcclResult ret = AllocAlgResource(newTag, opType, opParam, resRequest, resMap_[newTag], selectAivAlg);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("[HcclCommunicator][ExecOp] AllocAlgResource failed, algName=[%s]", algName.c_str()), ret);
        CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(opParam.tag));

        // 对于91093超节点内aiv跨机通信算子，将不同机的CCLbuffer地址存在约定好的aiv将读取的HBM位置
        CHK_RET(algOperator->PrepareCommInfoToDevice(algName, resMap_[newTag]));

        if (!isHaveCpuRank_) {
            if (isUseRankPort_) {
                std::vector<u32>& nicPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
                std::vector<u32>& vnicPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
                Heartbeat::GetInstance(deviceLogicId_)
                    .SetRankPortInfo(isUseRankPort_, nicPorts, vnicPorts, commPortConfig_.devPortSwitchOn);
            }
            // 开始注册心跳
            if (opType == HcclCMDType::HCCL_CMD_SEND) {
                CHK_RET(RegisterToHeartBeat(opParam.dstRank, tag));
                hbSendRecvTags_.emplace(tag);
            } else if (opType == HcclCMDType::HCCL_CMD_RECEIVE) {
                CHK_RET(RegisterToHeartBeat(opParam.srcRank, tag));
                hbSendRecvTags_.emplace(tag);
            } else {
                CHK_RET(RegisterToHeartBeat());
            }
        }
        CHK_RET(UpdateZeroCopy(opParam, resMap_[newTag]));
    } else if (opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
        // batchsendrecv需要根据任务来确定和哪些卡建链，因此复用tag，并在此基础上实现增量建链
        AlgResourceRequest resRequest;
        CHK_RET(algOperator->CalcIncreLinkRequest(algName, opParam, ranksLinked_, resRequest, needIncreLink));
        if (needIncreLink) {
            CHK_RET(RecordOpPara(opType, opParam));
            CHK_RET(IncreAllocLink(newTag, opParam, resRequest, resMap_[newTag]));
            CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(opParam.tag));
            opParam.needIncreLink = true;
        }
    }

    // 算法执行
    if (selectAivAlg) {
        CHK_RET(HandleAclGraphFirstOpAivBuff(opParam.stream.ptr()));
        if (aivClearEnable_) {
            // 用于判断图模式是否清零
            CHK_RET(algOperator->SetAivClearEnable(aivClearEnable_));
            aivOffloadTag_ = 1;
        }
        GetAivTag(algDesc.aivTagNum, opParam.isCapture, opParam.aivTag);
        HCCL_INFO(
            "[HcclCommunicator][ExecOp] tag[%s] userRank[%u] cur aiv tag [%d]", identifier_.c_str(), userRank_,
            opParam.aivTag);
        opParam.aicpuUnfoldMode = false;
        opParam.aicpuCacheEnable = 0;
        CHK_RET(algOperator->SetNumBlocks(aivCoreLimit));
    }
    std::vector<HcclSendRecvItem> hostSendRecvInfo;
    std::vector<HcclSendRecvItem> aicpuSendRecvInfo;
    std::vector<u8> isDirectRemoteRank;
    if (opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV && deviceType_ == DevType::DEV_TYPE_910_93) {
        SplitBsrData(opParam, isDirectRemoteRank, hostSendRecvInfo, aicpuSendRecvInfo);
        // A3 bsr记录Direct下发方式数据
        opParam.BatchSendRecvDataDes.isDirectRemoteRank = isDirectRemoteRank.data();
        if (!retryEnable_) {
            opParam.BatchSendRecvDataDes.sendRecvItemsPtr = aicpuSendRecvInfo.data();
            opParam.BatchSendRecvDataDes.itemNum = aicpuSendRecvInfo.size();
        }
    }
    // A2 Group SendRecv 将isDirectRemoteRank全部置为false
    if (opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV && deviceType_ == DevType::DEV_TYPE_910B && isGroupMode_) {
        isDirectRemoteRank.resize(userRankSize_, 0);
        opParam.BatchSendRecvDataDes.isDirectRemoteRank = isDirectRemoteRank.data();
    }
    auto algType = algOperator->GetAlgType();
    CHK_RET(RegisterDfxInfo(opParam, algType, resMap_[newTag].slaveStreams, selectAivAlg, tag));
    // 头计数
    CHK_RET(StarsCounter(dispatcher_, opParam.stream, HEAD, opParam.aicpuUnfoldMode, retryEnable_, selectAivAlg));
    if (opParam.aicpuUnfoldMode) {
        isInplaceStatus_ = 0;
        inPlaceSupportRetryStatus_ = InplaceSupportRetryStatus::INPLACE_STATUS_END;
        // algOperator->SupportRetryWithInplaceCheck 依赖 algOperator->SetRetryEnable 才能正确返回是否支持inplace

        inplaceSupportRetry_ = algOperator->SupportRetryWithInplaceCheck(
            opType, opParam, algName, isInplaceStatus_, inPlaceSupportRetryStatus_);
        HCCL_INFO(
            "[HcclCommunicator][ExecOp] aicpu Unfold mode algType[%s], inplaceSupportRetry_[%d], opType[%d], "
            "isInplaceStatus_[%d], inPlaceSupportRetryStatus_[%d].",
            AlgTypeToStr(algType).c_str(), inplaceSupportRetry_, opType, isInplaceStatus_, inPlaceSupportRetryStatus_);
        CHK_RET(OrchestrateAicpu(opType, algName, opParam, resMap_[newTag], newTag, algType, isCustom, needIncreLink));
    } else {
        // HOST展开aclgraph场景，capture从流
        if (!selectAivAlg) {
            CHK_RET(CaptureSlaveStreams(opParam.stream.ptr(), resMap_[newTag].slaveStreams));
        }
        OpCounterInfo opCounter;
        CHK_RET(GetOpCountInfo(opCounter));
        CHK_RET(algOperator->SetOpCounter(opCounter));
        CHK_RET(algOperator->Orchestrate(algName, opParam, resMap_[newTag]));
        if (hostResMap_.find(newTag) == hostResMap_.end()) {
            hostResMap_.insert(newTag);
        }
        CHK_RET(algOperator->GetNumBlocks(numBlocks_));
        if (implAlg_->GetAivModeConfig() && GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
            && !opParam.isCapture) {
            CHK_RET(GetCacheMap(algOperator, opParam, algType, selectAivAlg, newTag));
        }
    }
    // A3 bsr 只有走NPU直驱的时候hostSendRecvInfo才有内容
    if (!hostSendRecvInfo.empty()) {
        // A3 bsr获取到host侧需要下发的数据
        HCCL_INFO("[HcclCommunicator][ExecOp] hostSendRecvInfo size %zu", hostSendRecvInfo.size());
        opParam.BatchSendRecvDataDes.sendRecvItemsPtr = hostSendRecvInfo.data();
        opParam.BatchSendRecvDataDes.itemNum = hostSendRecvInfo.size();
        opParam.aicpuUnfoldMode = false;
        opParam.aicpuCacheEnable = 0;
        std::string tempTag;
        std::unique_ptr<CollAlgOperator> newalgOperator = implAlg_->GetAlgOperator(opType);
        CHK_SMART_PTR_NULL(newalgOperator);
        CHK_RET(newalgOperator->SelectAlg(opParam.tag, opParam, limit, algName, algDesc, tempTag));
        CHK_RET(newalgOperator->Orchestrate(algName, opParam, resMap_[newTag]));
    }
    lock.unlock();
    // 尾计数
    CHK_RET(StarsCounter(dispatcher_, opParam.stream, TAIL, opParam.aicpuUnfoldMode, retryEnable_, selectAivAlg));
    CHK_RET(UnRegisterDfxInfo(opParam, resMap_[newTag].slaveStreams));
    if (selectAivAlg) {
        CHK_RET(algOperator->SetAivClearEnable(false));
        aivClearEnable_ = false;
    }
    if (hcclNslbDp::GetInstance().GetGlobalCommTaskId() != 0 && hcclNslbDp::GetInstance().GetInitNetCoFlag() == true) {
        AdjInfo nslbAdjInfo = {};
        CHK_RET(algOperator->GetAdjInfo(algName, opParam, resMap_[newTag], nslbAdjInfo));
        NslbDp_CollectSendAdjTable(opType, opParam, algOperator->GetAlgType(), nslbAdjInfo);
    }
    if (isInGraphCaptureZeroCopy) {
        SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    }
    return HCCL_SUCCESS;
}

HcclResult
HcclCommunicator::FreeScratchMemOnOpBaseMode(DeviceMem& scratchMem, const OpParam& opParam, const HcclCMDType& opType)
{
    // 当前单算子模式下scratch内存为手动申请，需要手动进行释放
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE || IsForceAicpuOpBaseMode(opParam, opType)) {
        scratchMem.free();
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ReAllocScratchMemForAlltoall(
    HcclCMDType opType, const OpParam& opParam, AlgResourceRequest& resRequest, AlgResourceResponse& algResResponse)
{
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB
        && !IsForceAicpuOpBaseMode(opParam, opType)) {
        if (resRequest.scratchMemSize > 0) {
            algResResponse.scratchMem = GetWorkspaceScracthMem(opParam.tag, resRequest.scratchMemSize);
        }
        HCCL_DEBUG("[%s] WorkflowMode set for workspace opType[%u] tag[%s]", __func__, opType, opParam.tag.c_str());
    } else if (
        GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE || IsForceAicpuOpBaseMode(opParam, opType)) {
        CHK_RET(AllocOpBaseModeScratchMem(opType, opParam, resRequest, algResResponse));
        HCCL_DEBUG("[%s] WorkflowMode set for opType[%u] tag[%s]", __func__, opType, opParam.tag.c_str());
    } else {
        HCCL_ERROR("[%s] WorkflowMode is not set for opType[%u] tag[%s]", __func__, opType, opParam.tag.c_str());
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::HandleExistAlgResource(
    const std::string& newTag, const std::string& algName, HcclCMDType opType, const OpParam& opParam,
    std::unique_ptr<CollAlgOperator>& algOperator, bool selectAivAlg, bool aicpuUnfoldModeFor910B,
    bool needRecreateAlltoallComm)
{
    if (needRecreateAlltoallComm) {
        CHK_RET(hcclStreamSynchronize(opParam.stream.ptr(), commConfig_.GetConfigExecTimeOut()));

        AlgResourceRequest resRequest;
        CHK_RET(algOperator->CalcResRequest(algName, opParam, resRequest));

        // 释放旧内存防止泄漏
        CHK_RET(FreeScratchMemOnOpBaseMode(resMap_[newTag].scratchMem, opParam, opType));

        if (aicpuUnfoldModeFor910B) {
            CHK_RET(ReAllocScratchMemForAlltoall(opType, opParam, resRequest, resMap_[newTag]));
            isContextLaunched_ = true;
        } else {
            CHK_RET(RecordOpPara(opType, opParam));
            CHK_RET(AllocAlgResource(newTag, opType, opParam, resRequest, resMap_[newTag], selectAivAlg));
            CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(opParam.tag));

            if (!isHaveCpuRank_) {
                if (isUseRankPort_) {
                    std::vector<u32>& nicPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
                    std::vector<u32>& vnicPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
                    Heartbeat::GetInstance(deviceLogicId_)
                        .SetRankPortInfo(isUseRankPort_, nicPorts, vnicPorts, commPortConfig_.devPortSwitchOn);
                }
                CHK_RET(RegisterToHeartBeat());
            }
        }
    } else {
        DeviceMem tinySendRecvMem;
        CHK_RET(implAlg_->GetTinyMem(tinySendRecvMem));
        CHK_RET(CalcTinySendRecvMem(opParam, resMap_[newTag], tinySendRecvMem));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ExecOpAlltoAll(HcclCMDType opType, OpParam& opParam, bool isCustom)
{
    CHK_PRT_RET(
        isInvalidComm_,
        HCCL_ERROR(
            "[HcclCommunicator][%s] comm[%s], rank[%u], devId[%d], snapshot recovering, "
            "this comm is invalid, no operator is allowed to execute.",
            __func__, identifier_.c_str(), userRank_, deviceLogicId_),
        HCCL_E_UNAVAIL);

    std::string& tag = opParam.tag;
    u32 aivCoreLimit = numBlocks_;
    // 单机AIV场景下cache复用，提升下发性能
    if (implAlg_->GetAivModeConfig() && GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        if (aivCoreLimit == 0) {
            aclError acl_ret = aclrtGetResInCurrentThread(ACL_RT_DEV_RES_VECTOR_CORE, &aivCoreLimit);
            CHK_PRT_RET(
                acl_ret != ACL_SUCCESS,
                HCCL_ERROR("[HcclCommunicator][ExecOpAlltoAll] aclrtGetResInCurrentThread failed, ret=[%d]", acl_ret),
                HCCL_E_PARA);
        }
        opParam.deterministic = implAlg_->GetDeterministicConfig();
        opParam.aivCoreLimit = aivCoreLimit;
        auto it = hcclCacheMap_.find(opParam);
        if (it != hcclCacheMap_.end()) {
            CHK_RET(ExecOpCache(opType, opParam, it->second));
            return HCCL_SUCCESS;
        }
    }

    ForceProf(opParam.isCapture);
    bool isInGraphCaptureZeroCopy = false;
    zeroCopyAclGraph_->SetRetryEnable(retryEnable_);
    opParam.supportSymmetricMemory = IsSupportSymmetricMemory(opType, opParam);
    opParam.supportZeroCopy = !opParam.supportSymmetricMemory && IsSupportZeroCopy(opParam);
    opParam.aclGraphZeroCopyEnable = GetConfigAclGraphZeroCopyEnable();
    isInGraphCaptureZeroCopy = zeroCopyAclGraph_->SetAclGraphZeroCopyMode(
        deviceType_, opType, opParam, implAlg_.get(), cclBufferManager_.GetOutCCLbufferSize());
    if (isInGraphCaptureZeroCopy && userRankSize_ > 1) {
        CHK_RET(CreateCommCCLbuffer());
    }
    if (isShareComm_) {
        CHK_RET(ShareCCLbufferMgr::GetInstance().CheckCCLbuffConflict(cclBuffName_, opParam.stream.id()));
    }
    std::unique_ptr<CollAlgOperator> algOperator = implAlg_->GetAlgOperator(opType);
    AlltoAllOperator* alltoAllOperator = dynamic_cast<AlltoAllOperator*>(algOperator.get());
    CHK_PTR_NULL(alltoAllOperator);

    bool isSatisfyA2ACPForA3Condition = alltoAllOperator->IsSatisfyA2AContinuousPipelineFor91093Condition(opParam);
    bool IsSatisfyA2ACPForA2Condition = alltoAllOperator->IsSatisfyAlltoallContinuousPipelineCondition(opParam);
    if (IsSatisfyA2ACPForA2Condition || isSatisfyA2ACPForA3Condition) {
        opParam.aicpuUnfoldMode = true;
        opParam.aicpuCacheEnable = GetExternalInputAicpuCacheEnable();
    }

    // 算法选择
    std::string algName;
    std::string newTag;
    if (opParam.aicpuUnfoldMode) {
        // 用于inplace支持重执行判断
        CHK_RET(algOperator->SetRetryEnable(retryEnable_));
    }
    std::unique_ptr<PreProcessMetaInfo> preMetaInfo = std::make_unique<PreProcessMetaInfo>();
    CHK_SMART_PTR_NULL(preMetaInfo);

    bool preProcessFlag = alltoAllOperator->JudgeIfNeedPreProcessAndGetParam(opParam, preMetaInfo);
    if (preProcessFlag) {
        if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            CHK_RET(RegressCalPreOp(alltoAllOperator, opParam, preMetaInfo, const_cast<Stream&>(opParam.stream)));
        } else {
            CHK_RET(RegressCalPreOp(alltoAllOperator, opParam, preMetaInfo));
        }
    }

    if (deviceType_ == DevType::DEV_TYPE_910B && userRankSize_ > 1) {
        // 用于AIV支持Roce直驱判断
        CHK_RET(IsSupportAIVNormalQP(devicePhyId_, opParam.supportRoceDirect));
    }

    std::unique_lock<std::mutex> lock(commResMutex_);
    ResourceLimit limit;
    limit.ifLimit = true;
    limit.aivCoreLimit = aivCoreLimit;
    AlgDesc algDesc;
    algDesc.isLastSelect = true;
    CHK_RET(algOperator->SelectAlg(opParam.tag, opParam, limit, algName, algDesc, newTag));
    // 是否是AIV直驱Roce场景
    opParam.isNpuDirectRoce = algName == "AlltoAllDirectFullmeshAIVExecutor";
    if (isOnlyAiv_ && !algDesc.isAivMode) {
        std::string opTypeName = GetCMDTypeEnumStr(opType);
        HCCL_ERROR(
            "[HcclCommunicator][ExecOp] opType[%s] does not select aiv mode, aiv only mode is not supported.",
            opTypeName.c_str());
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_RET(PrepareZeroCopy(algName, algDesc, opParam));

    if (opParam.isCapture) {
        // aclgraph使用新的Tag，避免影响其他操作
        newTag += "_Capture";
        // aclgraph零拷贝场景下，每个算子都有单独的tag，需要记录，在graph销毁时清理相关资源
        if (isInGraphCaptureZeroCopy) {
            CHK_RET(AclgraphCallback::GetInstance().InsertNewTagToCaptureResMap(this, newTag, opParam));
            tagsRequiringHostCleanup_.insert(newTag);
        }
    }

    auto isSupportAlg = [](const std::string& algName, bool aicpuUnfoldMode) -> bool {
        return ((algName == "RunAlltoAllVFullMesh" || algName == "RunAlltoAllVTwoLevelPipeline") && aicpuUnfoldMode)
               || (algName == "RunAlltoAllDirectFullmesh" || algName == "RunAlltoAllFullMeshSymmetricMemory");
    };
    bool isOpbaseMode = GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
    if ((isOpbaseMode && userRankSize_ > 1) || (isSupportAlg(algName, opParam.aicpuUnfoldMode))) {
        CHK_RET(CreateCommCCLbuffer());
    }
    // 资源创建
    bool selectAivAlg = algDesc.isAivMode;
    if ((resMap_.find(newTag) != resMap_.end()) && opParam.isCapture) {
        AlgResourceRequest resRequest;
        CHK_RET(algOperator->CalcResRequest(algName, opParam, resRequest));
        if (HasRoceTransportLinks(resRequest.opTransport)) {
            auto resTmp = resMap_[newTag];
            ++captureCnt_;
            newTag += std::to_string(captureCnt_);
            resMap_[newTag] = resTmp;
            resRequest.isInGraphCaptureZeroCopy = isInGraphCaptureZeroCopy;
            CHK_RET(CleanTransportLinks(resRequest.opTransport, resMap_[newTag].opTransportResponse));
            if (IsEnableBackupLink()) {
                CHK_RET(CleanTransportLinks(resRequest.opTransport, resMap_[newTag].opTransportResponseBackUp));
            }
            // 记录指令信息用于一致性校验
            CHK_RET(RecordOpPara(opType, opParam));
            CHK_RET(IncreAllocLink(newTag, opParam, resRequest, resMap_[newTag]));
            // 移除tag对应的指令信息
            CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(opParam.tag));
            // aclgraph零拷贝场景下，除第一个capture外，需要记录，在graph销毁时清理相关资源
            CHK_RET(AclgraphCallback::GetInstance().InsertNewTagToCaptureResMap(this, newTag, opParam));
            tagsRequiringHostCleanup_.insert(newTag);
        }
    }
    InsertNewTagToTagMap(newTag, opParam.tag);
    bool aicpuUnfoldModeFor910B = deviceType_ == DevType::DEV_TYPE_910B && opParam.aicpuUnfoldMode
                                  && (algName == "RunAlltoAllVStaged" || algName == "RunAlltoAllVFullMesh");
    bool needRecreateAlltoallComm = false;
    if (resMap_.find(newTag) == resMap_.end()) {
        AlgResourceRequest resRequest;
        CHK_RET(algOperator->CalcResRequest(algName, opParam, resRequest));
        resRequest.isInGraphCaptureZeroCopy = isInGraphCaptureZeroCopy;
        CHK_RET(RecordOpPara(opType, opParam));
        CHK_RET(AllocAlgResource(newTag, opType, opParam, resRequest, resMap_[newTag], selectAivAlg));
        CHK_RET(RankConsistentcyChecker::GetInstance().DelOpPara(opParam.tag));
        if (opParam.isNpuDirectRoce) {
            // AIV直驱roce多机场景，需要生成RMAInfo并拷贝至Device
            CHK_RET(GenAiRMAInfoV2(newTag));
            CHK_RET(H2DAiRMAInfoV2(newTag, opParam.stream.ptr()));
        }
        // 对于91093超节点内aiv跨机通信算子，将不同机的CCLbuffer地址存在约定好的aiv将读取的HBM位置
        CHK_RET(algOperator->PrepareCommInfoToDevice(algName, resMap_[newTag]));

        if (!isHaveCpuRank_) {
            if (isUseRankPort_) {
                std::vector<u32>& nicPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
                std::vector<u32>& vnicPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
                Heartbeat::GetInstance(deviceLogicId_)
                    .SetRankPortInfo(isUseRankPort_, nicPorts, vnicPorts, commPortConfig_.devPortSwitchOn);
            }
            CHK_RET(RegisterToHeartBeat());
        }
        CHK_RET(UpdateZeroCopy(opParam, resMap_[newTag]));
    } else {
        CHK_RET(alltoAllOperator->CheckNeedRecreateComm(
            algName, opParam, resMap_[newTag].scratchMem.size(), needRecreateAlltoallComm));
        HCCL_INFO(
            "resMap_ find this newTag[%s], and need to judge whether recreate comm [%d]", newTag.c_str(),
            needRecreateAlltoallComm);
        CHK_RET(HandleExistAlgResource(
            newTag, algName, opType, opParam, algOperator, selectAivAlg, aicpuUnfoldModeFor910B,
            needRecreateAlltoallComm));
    }
    auto& algRes = resMap_[newTag];

    if (hcclNslbDp::GetInstance().GetGlobalCommTaskId() != 0 && hcclNslbDp::GetInstance().GetInitNetCoFlag() == true) {
        /* NSLB 填充 表  */
        u32 srcLocalRankId = userRank_;
        u32 rootRank = (opParam.root == INVALID_VALUE_RANKID) ? 0 : opParam.root;
        AlgType nslbAlgType = algOperator->GetAlgType();
        AlgTypeLevel1 algValue = nslbAlgType.algoLevel1;
        uint8_t nslbAlg = hcclNslbDp::GetInstance().GetNslbLevel1AlgType(algValue);

        if (algName == "RunAlltoAllVFullMesh" || algName == "RunAlltoAllDirectFullmesh") {
            nslbAlg = NSLBDP_PAIRWISE;
            if (deviceType_ == DevType::DEV_TYPE_910_93) {
                nslbAlg = NSLB_ALGO_TYPE_FULLMESH;
            }
        }

        std::string nslb_identifier = identifier_;
        HCCL_INFO("NSLBDP-SWK NslbDp_CollectOperTable nslb_identifier[%s] .", nslb_identifier.c_str());
        u32 rankSize = userRankSize_;
        u64 count = opParam.All2AllDataDes.sendCount * SIZE_TABLE[opParam.All2AllDataDes.sendType];
        // 填充表2
        hcclNslbDp::GetInstance().GenerateOpAndAdjTable(
            opType, rootRank, srcLocalRankId, nslbAlg, nslb_identifier, count, rankSize);
        AdjInfo nslbAdjInfo = {};
        CHK_RET(algOperator->GetAdjInfo(algName, opParam, algRes, nslbAdjInfo));
        HCCL_INFO(
            "[NSLBDP-WEN]-nslbAdjInfosize[%u]-algName[%s]-rankSize[%u]-commDesc[%s]..", nslbAdjInfo.dstRankNum,
            algName.c_str(), userRankSize_, identifier_.c_str());
        // 填充表3
        hcclNslbDp::GetInstance().GetAlgAdjacencyTable(
            opType, srcLocalRankId, rootRank, nslbAlg, nslb_identifier, nslbAdjInfo);
        /*发送流程*/
        hcclNslbDp::GetInstance().SendAlgorithmInfoTable();
    }
    // 算法执行
    if (opParam.isNpuDirectRoce) {
        // AIV直驱roce多机场景，需要生成RMAInfo并拷贝至Device
        CHK_PTR_NULL(combinOparaMem_);
        HcclCombinOpParam* combinOparaPtr = reinterpret_cast<HcclCombinOpParam*>(combinOparaMem_->ptr());
        CHK_PTR_NULL(combinOparaPtr);
        CHK_RET(algOperator->SetRmaInfo(combinOparaPtr->aiRMAInfo));
    }
    if (selectAivAlg) {
        CHK_RET(HandleAclGraphFirstOpAivBuff(opParam.stream.ptr()));
        if (aivClearEnable_) {
            // 用于判断图模式是否清零
            CHK_RET(algOperator->SetAivClearEnable(aivClearEnable_));
            aivOffloadTag_ = 1;
        }
        GetAivTag(algDesc.aivTagNum, opParam.isCapture, opParam.aivTag);
        HCCL_INFO(
            "[HcclCommunicator][ExecOpAlltoAll] tag[%s] userRank[%u] cur aiv tag [%d].", identifier_.c_str(), userRank_,
            opParam.aivTag);
        opParam.aicpuUnfoldMode = false;
        opParam.aicpuCacheEnable = 0;
        CHK_RET(algOperator->SetNumBlocks(aivCoreLimit));
    }

    auto algType = algOperator->GetAlgType();
    CHK_RET(RegisterDfxInfo(opParam, algType, algRes.slaveStreams, selectAivAlg, tag));
    // 头计数
    CHK_RET(StarsCounter(dispatcher_, opParam.stream, HEAD, opParam.aicpuUnfoldMode, retryEnable_, selectAivAlg));
    // 算法执行
    auto isSupportAicpuAlg = [](const std::string& algName) {
        static const std::set<std::string> aicpuAlgs
            = {"RunAlltoAllVFullMesh",           "RunAlltoAllDirectFullmesh",
               "RunAlltoAllVTwoLevelPipeline",   "RunAlltoAllFullMeshSymmetricMemory",
               "RunAlltoAllVContinuousPipeline", "RunAlltoAllVPipelineFor91093"};
        return aicpuAlgs.count(algName) > 0;
    };
    if (opParam.aicpuUnfoldMode && (isSupportAicpuAlg(algName) || aicpuUnfoldModeFor910B)) {
        isInplaceStatus_ = 0;
        inPlaceSupportRetryStatus_ = InplaceSupportRetryStatus::INPLACE_STATUS_END;
        // algOperator->SupportRetryWithInplaceCheck 依赖 algOperator->SetRetryEnable 才能正确返回是否支持inplace

        inplaceSupportRetry_ = algOperator->SupportRetryWithInplaceCheck(
            opType, opParam, algName, isInplaceStatus_, inPlaceSupportRetryStatus_);
        HCCL_INFO(
            "[HcclCommunicator][ExecOp] aicpu Unfold mode algType[%s], inplaceSupportRetry_[%d], opType[%d], "
            "isInplaceStatus_[%d], inPlaceSupportRetryStatus_[%d].",
            AlgTypeToStr(algType).c_str(), inplaceSupportRetry_, opType, isInplaceStatus_, inPlaceSupportRetryStatus_);
        CHK_RET(OrchestrateAicpu(
            opType, algName, opParam, algRes, newTag, algType, isCustom, false, needRecreateAlltoallComm));
    } else {
        // HOST展开aclgraph场景，capture从流
        if (!selectAivAlg) {
            CHK_RET(CaptureSlaveStreams(opParam.stream.ptr(), algRes.slaveStreams));
        }
        OpCounterInfo opCounter;
        CHK_RET(GetOpCountInfo(opCounter));
        CHK_RET(algOperator->SetOpCounter(opCounter));
        CHK_RET(algOperator->Orchestrate(algName, opParam, algRes));
        // for profiling, numBlocks upload
        CHK_RET(algOperator->GetNumBlocks(numBlocks_));
        if (implAlg_->GetAivModeConfig() && GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
            && !opParam.isCapture) {
            CHK_RET(GetCacheMap(algOperator, opParam, algType, selectAivAlg, newTag));
        }
    }
    lock.unlock();
    // 尾计数
    CHK_RET(StarsCounter(dispatcher_, opParam.stream, TAIL, opParam.aicpuUnfoldMode, retryEnable_, selectAivAlg));
    CHK_RET(UnRegisterDfxInfo(opParam, algRes.slaveStreams));
    if (selectAivAlg) {
        CHK_RET(algOperator->SetAivClearEnable(false));
        aivClearEnable_ = false;
    }

    if (isInGraphCaptureZeroCopy) {
        SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::RecordOpPara(HcclCMDType opType, const OpParam& opParam)
{
    u32 aivCoreLimit = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) ? numBlocks_ : 0;
    u8 deterministic = implAlg_->GetDeterministicConfig();
    switch (opType) {
        case HcclCMDType::HCCL_CMD_ALLGATHER:
        case HcclCMDType::HCCL_CMD_ALLREDUCE:
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER:
        case HcclCMDType::HCCL_CMD_BROADCAST:
            CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(
                opType, opParam.tag, opParam.DataDes.count, opParam.DataDes.dataType, opParam.reduceType, opParam.root,
                cclBufferManager_.GetInCCLbufferSize(), cclBufferManager_.GetOutCCLbufferSize(), identifier_.c_str(),
                ranktableCrc_, deterministic, aivCoreLimit));
            break;
        case HcclCMDType::HCCL_CMD_SCATTER:
        case HcclCMDType::HCCL_CMD_REDUCE:
            CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(
                opType, opParam.tag, opParam.DataDes.count, opParam.DataDes.dataType, opParam.reduceType, opParam.root,
                cclBufferManager_.GetInCCLbufferSize(), cclBufferManager_.GetOutCCLbufferSize(), identifier_.c_str(),
                ranktableCrc_, deterministic));
            break;
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V:
        case HcclCMDType::HCCL_CMD_ALLGATHER_V:
            CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(
                opType, opParam.tag, opParam.VDataDes.counts, opParam.VDataDes.displs, userRankSize_,
                opParam.VDataDes.dataType, opParam.reduceType, cclBufferManager_.GetInCCLbufferSize(),
                cclBufferManager_.GetOutCCLbufferSize(), identifier_.c_str(), ranktableCrc_, deterministic,
                aivCoreLimit));
            break;
        case HcclCMDType::HCCL_CMD_BATCH_SEND_RECV:
            CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(
                opType, opParam.tag, cclBufferManager_.GetInCCLbufferSize(), cclBufferManager_.GetOutCCLbufferSize(),
                identifier_.c_str(), ranktableCrc_));
            break;
        case HcclCMDType::HCCL_CMD_SEND:
            CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(
                opType, opParam.tag, opParam.DataDes.count, opParam.DataDes.dataType, opParam.dstRank, opParam.srTag,
                opParam.localGroupRank, cclBufferManager_.GetInCCLbufferSize(), cclBufferManager_.GetOutCCLbufferSize(),
                identifier_.c_str(), ranktableCrc_));
            break;
        case HcclCMDType::HCCL_CMD_RECEIVE:
            CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(
                opType, opParam.tag, opParam.DataDes.count, opParam.DataDes.dataType, opParam.srcRank, opParam.srTag,
                opParam.localGroupRank, cclBufferManager_.GetInCCLbufferSize(), cclBufferManager_.GetOutCCLbufferSize(),
                identifier_.c_str(), ranktableCrc_));
            break;
        case HcclCMDType::HCCL_CMD_ALLTOALL:
            CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(
                opType, opParam.tag, opParam.All2AllDataDes.sendCount, opParam.All2AllDataDes.sendType,
                opParam.reduceType, opParam.root, cclBufferManager_.GetInCCLbufferSize(),
                cclBufferManager_.GetOutCCLbufferSize(), identifier_.c_str(), ranktableCrc_, aivCoreLimit));
            break;
        case HcclCMDType::HCCL_CMD_ALLTOALLV:
        case HcclCMDType::HCCL_CMD_ALLTOALLVC:
            CHK_RET(RankConsistentcyChecker::GetInstance().RecordOpPara(
                opType, opParam.tag, 0, HCCL_DATA_TYPE_RESERVED, opParam.reduceType, opParam.root,
                cclBufferManager_.GetInCCLbufferSize(), cclBufferManager_.GetOutCCLbufferSize(), identifier_.c_str(),
                ranktableCrc_, aivCoreLimit));
            break;
        default:
            break;
    }
    return HCCL_SUCCESS;
}
HcclResult HcclCommunicator::HandleAclGraphFirstOpAivBuff(rtStream_t mainStream)
{
    aclmdlRI rtModel = nullptr;
    bool isCapture = false;
    u64 modelId = 0;
    CHK_RET(GetStreamCaptureInfo(mainStream, rtModel, isCapture));
    if (isCapture) {
        CHK_PTR_NULL(rtModel);
        // 获取不到modelId会报错
        CHK_RET(GetModelId(rtModel, modelId));
        if (captureModelIds_.find(modelId) == captureModelIds_.end()) {
            CHK_RET(AclgraphCallback::GetInstance().RegisterModelId(this, rtModel, modelId));
            // aclgraph场景，首算子清理AIV buff
            aivClearEnable_ = true;
            captureModelIds_.insert(modelId);
            HCCL_INFO("[HcclCommunicator][%s] modelId[%llu] is inserted to captureModelIds_", __func__, modelId);
        }
    }
    return HCCL_SUCCESS;
}

void HcclCommunicator::EraseCaptureModelId(u64 modelId)
{
    auto it = captureModelIds_.find(modelId);
    if (it != captureModelIds_.end()) {
        captureModelIds_.erase(it);
        HCCL_INFO("[HcclCommunicator][%s] modelId[%llu] is erased from captureModelIds_", __func__, modelId);
    }
    return;
}

bool HcclCommunicator::StreamIsCapture(rtStream_t mainStream) const
{
    bool isCapture = false;
    aclmdlRI rtModel = nullptr;
    CHK_RET(GetStreamCaptureInfo(mainStream, rtModel, isCapture));
    return isCapture;
}

HcclResult HcclCommunicator::CaptureSlaveStreams(rtStream_t mainStream, vector<Stream>& slaveStreams)
{
    if ((deviceType_ != DevType::DEV_TYPE_910_93) && (deviceType_ != DevType::DEV_TYPE_310P3)
        && (deviceType_ != DevType::DEV_TYPE_910B || GetExternalInputHcclEnableFfts())) {
        HCCL_INFO(
            "[HcclCommunicator][%s]Only 310P3 or A2 or A3 device in host expand mode need to capture slave streams.",
            __func__);
        return HCCL_SUCCESS;
    }
    aclmdlRI rtModel = nullptr;
    bool isCapture = false;
    u64 modelId = 0;
    CHK_RET(GetStreamCaptureInfo(mainStream, rtModel, isCapture));
    if (isCapture) {
        CHK_PTR_NULL(rtModel);
        CHK_RET(GetModelId(rtModel, modelId));
        for (auto slaveStream : slaveStreams) {
            CHK_RET(AddStreamToModel(slaveStream.ptr(), rtModel));
            HCCL_DEBUG(
                "[HcclCommunicator][%s]Add stream[%d] to model[%u] success.", __func__, slaveStream.id(), modelId);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildOpLocalScratchMemResParam(
    const AlgResourceResponse& algResource, const std::string& newTag, LocalResInfoV2* localResHostPtr)
{
    if (algResource.scratchMem.size() > 0) {
        hostMemVec_.resize(hostMemVec_.size() + 1);
        CHK_RET(AllocAndClearHostMem(sizeof(HccltagLocalResV2), hostMemVec_.back()));
        HccltagLocalResV2* tagLocalResHostPtr = static_cast<HccltagLocalResV2*>(hostMemVec_.back().get()->ptr());

        deviceMemVec_.resize(deviceMemVec_.size() + 1);
        CHK_RET(AllocAndClearDeviceMem(sizeof(HccltagLocalResV2), deviceMemVec_.back()));
        HccltagLocalResV2* tagLocalResDevicePtr = static_cast<HccltagLocalResV2*>(deviceMemVec_.back().get()->ptr());

        // 初始化HcclRankRelationResV2中的tagRes链表
        ListCommonInit(&tagLocalResDevicePtr->nextTagRes, &tagLocalResHostPtr->nextTagRes);
        // 刷新host空间内容
        CHK_SAFETY_FUNC_RET(
            memcpy_s(tagLocalResHostPtr->tag, sizeof(tagLocalResHostPtr->tag), newTag.c_str(), newTag.length() + 1));
        tagLocalResHostPtr->ScratchmemSize = algResource.scratchMem.size();
        tagLocalResHostPtr->Scratchmem = reinterpret_cast<u64>(algResource.scratchMem.ptr());

        // 3、将节点插入链表头
        ListCommonAddHead(
            &tagLocalResDevicePtr->nextTagRes, &tagLocalResHostPtr->nextTagRes, &localResHostPtr->nextTagRes,
            &opResDeviceParaPtr_->localRes.nextTagRes);
        HCCL_RUN_INFO(
            "[HcclCommunicator][BuildOpLocalScratchMemResParam] LocalResHostPtr head addr[%p], nextHost[%p], "
            "preHost[%p], tag LocalResHostPtr head addr[%p], nextHost[%p],"
            "preHost[%p], tag[%s]",
            &localResHostPtr->nextTagRes, localResHostPtr->nextTagRes.nextHost, localResHostPtr->nextTagRes.preHost,
            &tagLocalResHostPtr->nextTagRes, tagLocalResHostPtr->nextTagRes.nextHost,
            tagLocalResHostPtr->nextTagRes.preHost, tagLocalResHostPtr->tag);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CheckSetRetryStateToWaitResume()
{
    if (retryEnable_ && opRetryManager_ != nullptr) {
        HcclResult ret = opRetryManager_->SetRetryStateToWaitResume(identifier_, commConnections_.isRoot);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[NsRecovery]set opretry state to wait resume timeout."), HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildOpLocalResParam(const AlgResourceResponse& algResource, const std::string& newTag)
{
    LocalResInfoV2* localResHostPtr = &opResPara_.localRes;
    ListCommonInit(&opResDeviceParaPtr_->localRes.nextTagRes, &opResPara_.localRes.nextTagRes);
    if (algResource.slaveDevStreams.size() > LOCAL_STREAM_MAX_NUM) {
        HCCL_ERROR("[HcclCommunicator][BuildOpLocalResParam]Fail to assign stream for tag[%s]", newTag.c_str());
        return HCCL_E_PARA;
    }
    auto signalM2SNum = algResource.notifiesDevMain.size();
    auto signalS2MNum = algResource.notifiesDevAux.size();
    auto signalNum = signalM2SNum + signalS2MNum;
    if (signalNum > LOCAL_NOTIFY_MAX_NUM) {
        HCCL_ERROR("[HcclCommunicator][BuildOpLocalResParam]Fail to assign local notify for tag[%s]", newTag.c_str());
        return HCCL_E_PARA;
    }

    localResHostPtr->streamNum = algResource.slaveDevStreams.size();
    for (u32 i = 0; i < algResource.slaveDevStreams.size(); i++) {
        localResHostPtr->streamParam[i].streamInfo.streamIds = algResource.slaveDevStreams[i].id();
        localResHostPtr->streamParam[i].streamInfo.sqIds = algResource.slaveDevStreams[i].sqId();
        localResHostPtr->streamParam[i].streamInfo.cqIds = algResource.slaveDevStreams[i].cqId();
        localResHostPtr->streamParam[i].streamInfo.logicCqids = algResource.slaveDevStreams[i].logicCqId();
        CHK_RET(AllocAndGetStreamContextBuff(
            algResource.slaveDevStreams[i].id(), localResHostPtr->streamParam[i].sqCqContextAddr,
            localResHostPtr->streamParam[i].sqCqContextSize));
    }

    localResHostPtr->signalNum = signalNum;

    for (u32 i = 0; i < signalM2SNum; i++) {
        algResource.notifiesDevMain[i]->GetNotifyData(localResHostPtr->localSignals[i << 1]);
        algResource.notifiesDevAux[i]->GetNotifyData(localResHostPtr->localSignals[(i << 1) + 1]);
    }
    HcclResult ret = HCCL_SUCCESS;
    ret = CreateAndGetAiCpuNotify(
        localAiCpuOpNotify_[static_cast<u32>(AicpuLocalNotifyIdx::HOST_TO_AICPU_0)],
        localResHostPtr->aicpuOpNotify[static_cast<u32>(AicpuLocalNotifyIdx::HOST_TO_AICPU_0)]);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[HcclCommunicator][BuildOpLocalResParam]get aicpu notify 0 error, "
            "errNo[0x%016llx]",
            HCCL_ERROR_CODE(ret)),
        ret);
    ret = CreateAndGetAiCpuNotify(
        localAiCpuOpNotify_[static_cast<u32>(AicpuLocalNotifyIdx::HOST_TO_AICPU_1)],
        localResHostPtr->aicpuOpNotify[static_cast<u32>(AicpuLocalNotifyIdx::HOST_TO_AICPU_1)]);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[HcclCommunicator][BuildOpLocalResParam]get aicpu notify 1 error, errNo[0x%016llx]", HCCL_ERROR_CODE(ret)),
        ret);

    if (opMainStream_.ptr() == nullptr) {
        opMainStream_ = Stream(StreamType::STREAM_TYPE_DEVICE);
    }
    localResHostPtr->mainStreamParam.streamInfo.streamIds = opMainStream_.id();
    localResHostPtr->mainStreamParam.streamInfo.sqIds = opMainStream_.sqId();
    localResHostPtr->mainStreamParam.streamInfo.cqIds = opMainStream_.cqId();
    localResHostPtr->mainStreamParam.streamInfo.logicCqids = opMainStream_.logicCqId();
    CHK_RET(AllocAndGetStreamContextBuff(
        opMainStream_.id(), localResHostPtr->mainStreamParam.sqCqContextAddr,
        localResHostPtr->mainStreamParam.sqCqContextSize));

    // 按序下发的aicpu控制流
    if (aicpuOrderStream_.ptr() == nullptr) {
        aicpuOrderStream_ = Stream(StreamType::STREAM_TYPE_DEVICE);
    }
    opResPara_.aicpuOrderStreamParam.streamInfo.streamIds = aicpuOrderStream_.id();
    opResPara_.aicpuOrderStreamParam.streamInfo.sqIds = aicpuOrderStream_.sqId();
    opResPara_.aicpuOrderStreamParam.streamInfo.cqIds = aicpuOrderStream_.cqId();
    opResPara_.aicpuOrderStreamParam.streamInfo.logicCqids = aicpuOrderStream_.logicCqId();
    CHK_RET(AllocAndGetStreamContextBuff(
        opResPara_.aicpuOrderStreamParam.streamInfo.streamIds, opResPara_.aicpuOrderStreamParam.sqCqContextAddr,
        opResPara_.aicpuOrderStreamParam.sqCqContextSize));

#ifndef CCL_KERNEL_AICPU
    for (u32 i = 0; i < AICPU_LOCAL_EVENT_SIZE; ++i) {
        aclError ret = aclrtCreateEventExWithFlag(&localAicpuOpEvent_[i], ACL_EVENT_SYNC);
        CHK_PRT_RET(
            ret != ACL_SUCCESS,
            HCCL_ERROR(
                "[%s]aclrtCreateEventExWithFlag failed, ret[%d] event[%p].", __func__, ret, localAicpuOpEvent_[i]),
            HCCL_E_RUNTIME);
    }
#endif

    CHK_RET(BuildOpLocalScratchMemResParam(algResource, newTag, localResHostPtr));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllocAndGetStreamContextBuff(u32 streamId, u64& addr, u64& size)
{
    if (streamIdToStreamContext_.find(streamId) == streamIdToStreamContext_.end()) {
        DeviceMem streamContext;
        CHK_RET(CreateWorkSpace(sizeof(SqCqeContext), streamContext));
        streamIdToStreamContext_.insert({streamId, std::move(streamContext)});
    }
    addr = reinterpret_cast<u64>(streamIdToStreamContext_.at(streamId).ptr());
    size = streamIdToStreamContext_.at(streamId).size();
    HCCL_INFO("%s success, streamId:%u, addr:0x%llx, size:%llu", __func__, streamId, addr, size);
    return HCCL_SUCCESS;
}

u32 HcclCommunicator::UpdateOpIndex(const OpParam& opParam)
{
    u32 opIndex = 0;
    u32 commIndex = 0;
    // 用于重执行和taskException打印的算子计数，bsr/sendrecv/其他算子分别计数
    if (opParam.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
        constexpr s32 batSendRecvIndex = -1; // batchSendRecv使用 key = -1
        commIndex = batSendRecvIndex;
    } else if (opParam.opType == HcclCMDType::HCCL_CMD_SEND) {
        commIndex = opParam.dstRank;
    } else if (opParam.opType == HcclCMDType::HCCL_CMD_RECEIVE) {
        commIndex = opParam.srcRank;
    } else {
        commIndex = userRank_;
    }

    auto it = opIndexMap_.find(commIndex);
    if (it != opIndexMap_.end()) {
        opIndex = ++(it->second);
    } else {
        opIndexMap_.insert({commIndex, 1});
        opIndex = 1;
    }

    HCCL_DEBUG(
        "%s tag:%s opType:%u commIndex:%u opIndex:%u", __func__, opParam.tag.c_str(), opParam.opType, commIndex,
        opIndex);
    return opIndex;
}

HcclResult HcclCommunicator::BuildAicpuCustomParam()
{
    if (aicpuCustomDev_.ptr() == nullptr) {
        CHK_RET(CreateWorkSpace(sizeof(AicpuCustomParam), aicpuCustomDev_));
    }

    opResPara_.aicpuCustomParamAddr = reinterpret_cast<u64>(aicpuCustomDev_.ptr());
    opResPara_.aicpuCustomParamSize = aicpuCustomDev_.size();
    HCCL_INFO(
        "%s success, aicpuCustomParamAddr:0x%llx, aicpuCustomParamSize:%llu", __func__, opResPara_.aicpuCustomParamAddr,
        opResPara_.aicpuCustomParamSize);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildAicpuOrderLaunchNotify()
{
    if (aicpuOrderNotifyAddr_.ptr() == nullptr) {
        CHK_RET(CreateWorkSpace(sizeof(HcclSignalInfo) * AICPU_ORDER_NOTIFY_MAX_NUM, aicpuOrderNotifyAddr_));
    }

    opResPara_.aicpuOrderNotifyAddr = reinterpret_cast<u64>(aicpuOrderNotifyAddr_.ptr());
    opResPara_.aicpuOrderNotifySize = aicpuOrderNotifyAddr_.size();
    HCCL_INFO(
        "%s success, aicpuOrderNotifyAddr:0x%llx, aicpuOrderNotifySize:%llu", __func__, opResPara_.aicpuOrderNotifyAddr,
        opResPara_.aicpuOrderNotifySize);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildAiRmaInfoParam(
    const std::string& newTag, [[maybe_unused]] const std::string& algName, [[maybe_unused]] const HcclCMDType opType)
{
    HCCL_DEBUG("[HcclCommunicator][%s] Start prepare.", __func__);
    CHK_PTR_NULL(aiRMAInfoMem_);
    HcclAiRMAInfo* aiRMAInfoPtr = reinterpret_cast<HcclAiRMAInfo*>(aiRMAInfoMem_->ptr());
    CHK_PTR_NULL(aiRMAInfoPtr);
    aiRMAInfoPtr->curRankId = userRank_;
    aiRMAInfoPtr->rankNum = userRankSize_;
    u32 localRankSize = meshAggregationRankSize_;
    LevelNSubCommTransport& commTransport = resMap_[newTag].opTransportResponse[COMM_LEVEL0];
    CHK_PRT_RET(
        commTransport.size() <= 0,
        HCCL_ERROR(
            "[%s] no LevelComm resource, please create comm first. "
            "tag[%s], curRankId[%u] rankNum[%u]",
            __func__, newTag.c_str(), aiRMAInfoPtr->curRankId, aiRMAInfoPtr->rankNum),
        HCCL_E_INTERNAL);
    std::vector<LINK>& links = commTransport[0].links;
    CHK_PRT_RET(
        links.size() <= 0,
        HCCL_ERROR(
            "[%s] no transport resource, please create links first. "
            "tag[%s], curRankId[%u] rankNum[%u]",
            __func__, newTag.c_str(), aiRMAInfoPtr->curRankId, aiRMAInfoPtr->rankNum),
        HCCL_E_INTERNAL);

    LevelNSubCommTransport& tmpCommTransport = resMap_[newTag].opTransportResponse[COMM_MESH_L1];
    CHK_PRT_RET(
        tmpCommTransport.size() <= 0,
        HCCL_ERROR(
            "[%s] no LevelComm resource, please create comm first. "
            "tag[%s], curRankId[%u] rankNum[%u]",
            __func__, newTag.c_str(), aiRMAInfoPtr->curRankId, aiRMAInfoPtr->rankNum),
        HCCL_E_INTERNAL);
    std::vector<LINK>& tmpLinks = tmpCommTransport[0].links;
    CHK_PRT_RET(
        tmpLinks.size() <= 0,
        HCCL_ERROR(
            "[%s] no transport resource, please create links first. "
            "tag[%s], curRankId[%u] rankNum[%u]",
            __func__, newTag.c_str(), aiRMAInfoPtr->curRankId, aiRMAInfoPtr->rankNum),
        HCCL_E_INTERNAL);

    CHK_RET(GetAivQPInfoV2(tmpLinks, newTag));
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
        if (i != aiRMAInfoPtr->curRankId
            && ((i % localRankSize) == (aiRMAInfoPtr->curRankId % localRankSize)
                || (i / localRankSize) == (aiRMAInfoPtr->curRankId / localRankSize))) {
            auto transport = links[i % localRankSize]; // localranksize个
            if ((i % localRankSize) == (aiRMAInfoPtr->curRankId % localRankSize)) {
                transport = tmpLinks[i / localRankSize]; // servernum个
            }
            // link rank info
            CHK_RET(GetTransportRemoteMem(transport, UserMemType::INPUT_MEM, remoteIn));
            CHK_RET(GetTransportRemoteMem(transport, UserMemType::OUTPUT_MEM, remoteOut));
            CHK_RET(GetTransportLocalMem(transport, UserMemType::INPUT_MEM, localIn));
            CHK_RET(GetTransportLocalMem(transport, UserMemType::OUTPUT_MEM, localOut));

            if (transport->GetTransportType() == TransportType::TRANS_TYPE_IBV_EXP) {
                CHK_RET(GenIbvAiRMAInfo(i, transport, newTag, aiRMAInfoPtr));
            }
        } else if (i == aiRMAInfoPtr->curRankId) {
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
            "memDetailPtr[%p] remoteInAddr[%p] remoteInSize[%llu] remoteOutAddr[%p] "
            "remoteOutSize[%llu] localInAddr[%p] localInSize[%llu] "
            "localOutAddr[%p] localOutSize[%llu] ",
            __func__, newTag.c_str(), aiRMAInfoPtr->curRankId, i, aiRMAInfoPtr->rankNum, aiRMAInfoPtr->qpNum,
            aiMemHost[i].memMaxNum, aiMemHost[i].sizeOfMemDetails, aiMemHost[i].memDetailPtr, remoteIn.addr,
            remoteIn.size, remoteOut.addr, remoteOut.size, localIn.addr, localIn.size, localOut.addr, localOut.size);
    }
    return HCCL_SUCCESS;
}

template <typename T>
HcclResult HcclCommunicator::CopyVectorToDeviceMem(const u64 len, DeviceMem& dstDeviceMem, const std::vector<T>& srcVec)
{
    CHK_PRT_RET(
        len == 0, HCCL_INFO("[HcclCommunicator][CopyVectorToDeviceMem] space size is zero, no need to malloc memory"),
        HCCL_SUCCESS);

    CHK_PRT_RET(
        (len > ULONG_MAX),
        HCCL_ERROR("[HcclCommunicator][CopyVectorToDeviceMem] space size is greater than %llu", ULONG_MAX),
        HCCL_E_PARA);

    CHK_RET(CreateWorkSpace(len, dstDeviceMem));
    std::shared_ptr<HostMem> srcHostMem;
    CHK_RET(AllocAndClearHostMem(len, srcHostMem));
    std::copy(srcVec.begin(), srcVec.end(), static_cast<T*>(srcHostMem.get()->ptr()));
    CHK_RET(hrtMemSyncCopy(
        dstDeviceMem.ptr(), len, srcHostMem.get()->ptr(), len, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildOpTopoResTlvParam(
    [[maybe_unused]] const std::string& algName, const std::vector<std::vector<std::vector<u32>>>& inputVectorInfo,
    DeviceMem& dstTlvDeviceMem, u64& tlvLen)
{
    vector<u32> tlv;
    CommonTlv commonTlv;
    HCCL_DEBUG(
        "[HcclCommunicator][BuildOpTopoResTlvParam] input vector size[%lu], group[%s].", inputVectorInfo.size(),
        identifier_.c_str());
    for (u16 level0Idx = 0; level0Idx < inputVectorInfo.size(); level0Idx++) {
        for (u16 level1Idx = 0; level1Idx < inputVectorInfo[level0Idx].size(); level1Idx++) {
            commonTlv.type = ((level0Idx << TOP_COMM_LEVEL0_SHIFT) | level1Idx);
            commonTlv.length = (sizeof(LENGTH_TYPE) + sizeof(TAG_TYPE))
                               + inputVectorInfo[level0Idx][level1Idx].size() * sizeof(RANK_TYPE);
            tlv.push_back(commonTlv.type);
            tlv.push_back(commonTlv.length);
            tlv.insert(
                tlv.end(), inputVectorInfo[level0Idx][level1Idx].begin(), inputVectorInfo[level0Idx][level1Idx].end());
        }
    }
    for (u64 idx = 0; idx < tlv.size(); idx++) {
        HCCL_DEBUG("[HcclCommunicator][BuildOpTopoResTlvParam] idx[%lu] tlv[%lu].", idx, tlv[idx]);
    }
    tlvLen = tlv.size() * sizeof(u32);
    CHK_RET(CopyVectorToDeviceMem(tlvLen, dstTlvDeviceMem, tlv));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildOpTopoResVectorTlvParam(
    [[maybe_unused]] const std::string& algName,
    const std::vector<std::vector<std::vector<std::vector<u32>>>>& inputVectorInfo, DeviceMem& dstTlvDeviceMem,
    u64& tlvLen)
{
    vector<u32> tlv;
    CommonTlv commonTlv;
    HCCL_DEBUG(
        "[HcclCommunicator][BuildOpTopoResVectorTlvParam] input vector size[%lu], group[%s]", inputVectorInfo.size(),
        identifier_.c_str());
    for (u16 level0Idx = 0; level0Idx < inputVectorInfo.size(); level0Idx++) {
        for (u16 level1Idx = 0; level1Idx < inputVectorInfo[level0Idx].size(); level1Idx++) {
            for (u16 level2Idx = 0; level2Idx < inputVectorInfo[level0Idx][level1Idx].size(); level2Idx++) {
                commonTlv.type = (((level0Idx << TOP_HIERARCHICAL_COMM_LEVEL0_SHIFT) | level1Idx)
                                  << TOP_HIERARCHICAL_COMM_LEVEL1_SHIFT)
                                 | level2Idx;
                commonTlv.length = (sizeof(LENGTH_TYPE) + sizeof(TAG_TYPE))
                                   + inputVectorInfo[level0Idx][level1Idx][level2Idx].size() * sizeof(RANK_TYPE);
                tlv.push_back(commonTlv.type);
                tlv.push_back(commonTlv.length);
                tlv.insert(
                    tlv.end(), inputVectorInfo[level0Idx][level1Idx][level2Idx].begin(),
                    inputVectorInfo[level0Idx][level1Idx][level2Idx].end());
            }
        }
    }
    for (u64 idx = 0; idx < tlv.size(); idx++) {
        HCCL_DEBUG("[HcclCommunicator][BuildOpTopoResVectorTlvParam] idx[%lu] tlv[%lu]", idx, tlv[idx]);
    }
    tlvLen = tlv.size() * sizeof(u32);
    CHK_RET(CopyVectorToDeviceMem(tlvLen, dstTlvDeviceMem, tlv));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildPairLinkCounter([[maybe_unused]] const std::string& algName)
{
    constexpr u32 KEY_VALUE_TO_VECTOR_MODULUS = 2;
    if (pairLinkCounterDevice_.ptr() == nullptr) {
        u64 pairLinkCounterSize = pairLinkCounter_.size();
        HCCL_DEBUG(
            "[HcclCommunicator][BuildPairLinkCounter] pairLinkCounter size[%lu], group[%s]", pairLinkCounterSize,
            identifier_.c_str());
        std::vector<u32> pairLinkCounterVec(pairLinkCounterSize * KEY_VALUE_TO_VECTOR_MODULUS);
        u64 index = 0;
        for (auto& kt : pairLinkCounter_) {
            pairLinkCounterVec[index] = kt.first;
            pairLinkCounterVec[index + 1] = kt.second;
            index += KEY_VALUE_TO_VECTOR_MODULUS; // 每次根据
        }
        u64 len = pairLinkCounterSize * sizeof(u32) * KEY_VALUE_TO_VECTOR_MODULUS; // key-value，都为u32
        CHK_RET(CopyVectorToDeviceMem(len, pairLinkCounterDevice_, pairLinkCounterVec));
        opResPara_.topoInfo.pairLinkCounter = reinterpret_cast<u64>(pairLinkCounterDevice_.ptr());
        opResPara_.topoInfo.pairLinkCounterNum = pairLinkCounterSize * KEY_VALUE_TO_VECTOR_MODULUS;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildIsUsedRdmaRank([[maybe_unused]] const std::string& algName)
{
    constexpr u32 KEY_VALUE_TO_VECTOR_MODULUS = 2;
    if (isUsedRdmaRankPairDevice_.ptr() == nullptr) {
        std::unordered_map<u32, bool> isUsedRdmaMap;
        CHK_RET(implAlg_->GetIsUsedRdmaMap(isUsedRdmaMap));
        u64 isUsedRdmaMapSize = isUsedRdmaMap.size();
        HCCL_DEBUG(
            "[HcclCommunicator][BuildIsUsedRdmaRank] is used Rdma rank size[%lu], group[%s]", isUsedRdmaMapSize,
            identifier_.c_str());
        std::vector<u32> isUsedRdmaPairVec(isUsedRdmaMapSize * KEY_VALUE_TO_VECTOR_MODULUS);
        u64 index = 0;
        for (auto& kt : isUsedRdmaMap) {
            isUsedRdmaPairVec[index] = kt.first;
            isUsedRdmaPairVec[index + 1] = static_cast<u32>(kt.second);
            index += KEY_VALUE_TO_VECTOR_MODULUS;
        }
        u64 len = isUsedRdmaMapSize * sizeof(u32) * KEY_VALUE_TO_VECTOR_MODULUS; // key-value，都为u32
        CHK_RET(CopyVectorToDeviceMem(len, isUsedRdmaRankPairDevice_, isUsedRdmaPairVec));
        opResPara_.topoInfo.isUsedRdmaRankPair = reinterpret_cast<u64>(isUsedRdmaRankPairDevice_.ptr());
        opResPara_.topoInfo.isUsedRdmaRankPairNum = isUsedRdmaMapSize * KEY_VALUE_TO_VECTOR_MODULUS;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildNicList([[maybe_unused]] const std::string& algName)
{
    if (nicListDevice_.ptr() == nullptr) {
        u64 len = nicList_.size() * sizeof(u32);
        HCCL_DEBUG(
            "[HcclCommunicator][BuildNicList] niclist size[%lu], group[%s]", nicList_.size(), identifier_.c_str());
        CHK_RET(CopyVectorToDeviceMem(len, nicListDevice_, nicList_));
        opResPara_.topoInfo.nicList = reinterpret_cast<u64>(nicListDevice_.ptr());
        opResPara_.topoInfo.nicNum = nicList_.size();
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildBridgeRank([[maybe_unused]] const std::string& algName)
{
    if (bridgeRankDevice_.ptr() == nullptr) {
        std::vector<bool> isBridgeVector;
        CHK_RET(implAlg_->GetIsBridgeVector(isBridgeVector));
        u64 len = isBridgeVector.size() * sizeof(bool);
        HCCL_DEBUG(
            "[HcclCommunicator][BuildBridgeRank] Bridge size[%lu], group[%s]", isBridgeVector.size(),
            identifier_.c_str());
        CHK_RET(CopyVectorToDeviceMem(len, bridgeRankDevice_, isBridgeVector));
        opResPara_.topoInfo.bridgeRank = reinterpret_cast<u64>(bridgeRankDevice_.ptr());
        opResPara_.topoInfo.bridgeRankNum = isBridgeVector.size();
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildCommPlanRank(const std::string& algName)
{
    opResPara_.topoInfo.complanRank = 0;
    opResPara_.topoInfo.complanRankLength = 0;
    if (complanRankDevice_.ptr() == nullptr) {
        std::vector<std::vector<std::vector<u32>>> commPlaneRanks;
        CHK_RET(implAlg_->GetCommPlaneRanks(commPlaneRanks));
        u64 tlvLen = 0;
        CHK_RET(BuildOpTopoResTlvParam(algName, commPlaneRanks, complanRankDevice_, tlvLen));
        opResPara_.topoInfo.complanRank = reinterpret_cast<u64>(complanRankDevice_.ptr());
        opResPara_.topoInfo.complanRankLength = tlvLen;
        HCCL_DEBUG(
            "[HcclCommunicator][BuildCommPlanRank] comm plane ranks tlv length[%lu], ptr[%p], group[%s], "
            "local user rankId[%u] ",
            tlvLen, complanRankDevice_.ptr(), identifier_.c_str(), userRank_);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildServerAndsuperPodRank(const std::string& algName)
{
    opResPara_.topoInfo.serverAndsuperPodRank = 0;
    opResPara_.topoInfo.serverAndsuperPodRankLength = 0;
    if (serverAndsuperPodToRankDevice_.ptr() == nullptr) {
        std::vector<std::vector<std::vector<u32>>> serverAndsuperPodToRank;
        CHK_RET(implAlg_->GetRankVecInfo(serverAndsuperPodToRank));
        u64 tlvLen = 0;
        CHK_RET(BuildOpTopoResTlvParam(algName, serverAndsuperPodToRank, serverAndsuperPodToRankDevice_, tlvLen));
        opResPara_.topoInfo.serverAndsuperPodRank = reinterpret_cast<u64>(serverAndsuperPodToRankDevice_.ptr());
        opResPara_.topoInfo.serverAndsuperPodRankLength = tlvLen;
        HCCL_DEBUG(
            "[HcclCommunicator][BuildServerAndsuperPodRank] server and super pod ranks tlv length[%lu], ptr[%p], "
            "group[%s], local user rankId[%u] ",
            tlvLen, serverAndsuperPodToRankDevice_.ptr(), identifier_.c_str(), userRank_);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildOpRetryParam(const AlgResourceResponse& algResource, const std::string& newTag)
{
    opResPara_.config.retryEnable = static_cast<u8>(retryEnable_);
    opResPara_.config.retryHoldTime = commConfig_.GetConfigRetryHoldTime();
    opResPara_.config.retryIntervalTime = commConfig_.GetConfigRetryIntervalTime();
    // aicpu和custom共用同一个opResPara_，aicpu初始化完成后，会修改h2d/d2h的指针，然后重新传给custom
    opResPara_.kfcControlTransferH2DParams = kfcControlTransferH2D_->GetCommunicateParams();
    opResPara_.kfcStatusTransferD2HParams = kfcStatusTransferD2H_->GetCommunicateParams();
    opResPara_.debugConfig = GetDebugConfig();

    CHK_SMART_PTR_NULL(opRetryStreamPtr_);
    if (opRetryStreamPtr_->find(newTag) == opRetryStreamPtr_->end()) {
        std::vector<Stream> retryStreams(algResource.slaveDevStreams.begin(), algResource.slaveDevStreams.end());
        retryStreams.push_back(opMainStream_);
        opRetryStreamPtr_->insert(std::make_pair(newTag, retryStreams));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildCommPlaneSubGroupRank(const std::string& algName)
{
    opResPara_.hierarchicalAlgInfo.commplaneSubGroupRank = 0;
    opResPara_.hierarchicalAlgInfo.commplaneSubGroupRankLength = 0;
    if (commplaneSubGroupRankDevice_.ptr() == nullptr) {
        std::vector<std::vector<std::vector<std::vector<u32>>>> commplaneSubGroupVector;
        CHK_RET(implAlg_->GetCommPlaneSubGroupVector(commplaneSubGroupVector));
        u64 tlvLen = 0;
        CHK_RET(BuildOpTopoResVectorTlvParam(algName, commplaneSubGroupVector, commplaneSubGroupRankDevice_, tlvLen));
        opResPara_.hierarchicalAlgInfo.commplaneSubGroupRank
            = reinterpret_cast<u64>(commplaneSubGroupRankDevice_.ptr());
        opResPara_.hierarchicalAlgInfo.commplaneSubGroupRankLength = tlvLen;
        HCCL_DEBUG(
            "[HcclCommunicator][BuildCommPlaneSubGroupRank] comm plane subGroups ranks tlv length[%lu], ptr[%p], "
            "group[%s], local user rankId[%u] ",
            tlvLen, commplaneSubGroupRankDevice_.ptr(), identifier_.c_str(), userRank_);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildHierarchicalAlgOption(u32* ahcConfInfo)
{
    std::map<AHCConcOpType, TemplateType> hierarchicalAlgOption;
    CHK_RET(implAlg_->GetAHCAlgOption(hierarchicalAlgOption));
    ahcConfInfo[TOP_HIERARCHICAL_CONF_lENGTH_INDEX] = hierarchicalAlgOption.size();

    if (hierarchicalAlgOption.size() >= (TOP_HIERARCHICAL_CONF_SIZE - 1)) {
        HCCL_ERROR(
            "[HcclCommunicator][BuildHierarchicalAlgOption] host hierarchicalAlgOption size[%u] exceed maxsize[%u]",
            hierarchicalAlgOption.size(), (TOP_HIERARCHICAL_CONF_SIZE - 1));
        return HCCL_E_INTERNAL;
    }

    HCCL_DEBUG(
        "[HcclCommunicator][BuildHierarchicalAlgOption] host hierarchicalAlgOption.size() [%u]",
        hierarchicalAlgOption.size());

    // 默认清空内存
    for (u32 i = TOP_HIERARCHICAL_CONF_INFO_INDEX; i < TOP_HIERARCHICAL_CONF_SIZE; i++) {
        ahcConfInfo[i] = 0;
    }

    u32 confDataStartIndex = TOP_HIERARCHICAL_CONF_INFO_INDEX;
    for (auto it = hierarchicalAlgOption.begin(); it != hierarchicalAlgOption.end(); ++it) {
        HCCL_DEBUG(
            "[HcclCommunicator][BuildHierarchicalAlgOption] host Level [%u], ConcType[%u] AHCOpType[%u], TemplateType "
            "[%u]",
            it->first.ahcLevel, it->first.concType, it->first.ahcOpType, it->second);

        u32 confData = (static_cast<u32>(it->first.ahcLevel) << TOP_HIERARCHICAL_CONF_LEVEL_SHIFT)
                       | (static_cast<u32>(it->first.concType) << TOP_HIERARCHICAL_CONF_CONC_TYPE_SHIFT)
                       | (static_cast<u32>(it->first.ahcOpType) << TOP_HIERARCHICAL_CONF_OP_TYPE_SHIFT)
                       | (static_cast<u32>(it->second) << TOP_HIERARCHICAL_CONF_TEMPLATE_TYPE_SHIFT);
        ahcConfInfo[confDataStartIndex] = confData;
        confDataStartIndex = confDataStartIndex + 1;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildOpTopoResParam(
    const std::string& algName, [[maybe_unused]] const AlgResourceResponse& algResource)
{
    opResPara_.topoInfo.userRank = userRank_;
    opResPara_.topoInfo.userRankSize = userRankSize_;
    opResPara_.topoInfo.deviceLogicId = deviceLogicId_;
    opResPara_.topoInfo.isSingleMeshAggregation = isSingleMeshAggregation_;
    opResPara_.topoInfo.deviceNumPerAggregation = deviceNumPerAggregation_;
    opResPara_.topoInfo.superPodNum = superPodNum_;
    opResPara_.topoInfo.devicePhyId = devicePhyId_;
    opResPara_.topoInfo.deviceType = static_cast<u32>(deviceType_);
    TopoType topoType;
    CHK_RET(implAlg_->GetTopoType(topoType));
    opResPara_.topoInfo.topoType = static_cast<u32>(topoType);
    opResPara_.topoInfo.serverNum = serverNum_;
    opResPara_.topoInfo.meshAggregationRankSize = meshAggregationRankSize_;
    opResPara_.topoInfo.multiModuleDiffDeviceNumMode = multiModuleDiffDeviceNumMode_;
    opResPara_.topoInfo.multiSuperPodDiffServerNumMode = multiSuperPodDiffServerNumMode_;
    opResPara_.topoInfo.realUserRank = realUserRank_;
    opResPara_.topoInfo.isDiffDeviceModule = isDiffDeviceModule_;
    opResPara_.topoInfo.isDiffDeviceType = isDiffDeviceType_;
    opResPara_.topoInfo.gcdDeviceNumPerAggregation = gcdDeviceNumPerAggregation_;
    opResPara_.topoInfo.moduleNum = moduleNum_;
    opResPara_.isARSDoubleRing = isARSDoubleRing_;
    opResPara_.multiSuperPodDiffDeviceNumMode = multiSuperPodDiffDeviceNumMode_;
    CHK_RET(BuildPairLinkCounter(algName));
    CHK_RET(BuildIsUsedRdmaRank(algName));
    CHK_RET(BuildNicList(algName));
    CHK_RET(BuildBridgeRank(algName));
    CHK_RET(BuildCommPlanRank(algName));
    CHK_RET(BuildServerAndsuperPodRank(algName));
    CHK_RET(BuildCommPlaneSubGroupRank(algName));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildOpRemoteLinkP2pResParam(
    const LINK& link, HccltagRemoteResV3& tagRemoteRes, TransportLinkType linkType)
{
    // hccs sio并发场景，sio链路（linkType为SIO）打包到linkP2pSio, hccs链路（linkType为HCCS）打包到linkP2p；
    // 其他场景打包到linkP2p
    HcclLinkP2pV2* linkp2p = &(tagRemoteRes.tagRemoteResPtr->linkP2p);
    if (linkType == TransportLinkType::SIO) {
        linkp2p = &(tagRemoteRes.tagRemoteResPtr->linkP2pSio);
    }
    if (linkp2p->localIpcSignal[0].resId != INVALID_U64) {
        HCCL_INFO(
            "[%s]the linkP2p already exists, no need to refresh transport resource, resId[%llu]", __func__,
            linkp2p->localIpcSignal[0].resId);
        return HCCL_SUCCESS;
    }
    // localMem & remoteMem
    void* inbufferPtr = nullptr;
    void* outbufferPtr = nullptr;
    CHK_RET(link->GetRemoteMem(UserMemType::INPUT_MEM, &inbufferPtr));
    CHK_RET(link->GetRemoteMem(UserMemType::OUTPUT_MEM, &outbufferPtr));
    (linkp2p->remoteMem)[INPUT].addr = reinterpret_cast<u64>(inbufferPtr);
    (linkp2p->remoteMem)[OUTPUT].addr = reinterpret_cast<u64>(outbufferPtr);
    CHK_RET(link->GetRemoteMemSize(UserMemType::INPUT_MEM, (linkp2p->remoteMem)[INPUT].size));
    CHK_RET(link->GetRemoteMemSize(UserMemType::OUTPUT_MEM, (linkp2p->remoteMem)[OUTPUT].size));
    MemDetails localMem; // 暂时预留，赋值为空
    (linkp2p->localMem)[0] = localMem;
    (linkp2p->localMem)[1] = localMem;
    HCCL_DEBUG("[%s] finish set localMem & remoteMem info", __func__);
    // localnotify & remotenotify
    u64 notifyNum = 0;
    std::vector<HcclSignalInfo> locIpcSignals;
    std::vector<HcclSignalInfo> rmtIpcSignals;
    CHK_RET(link->GetLocalNotify(locIpcSignals));
    CHK_RET(link->GetRemoteNotify(rmtIpcSignals));

    for (size_t i = 0; i < locIpcSignals.size(); i++) {
        CHK_RET(CheckNotifyOrQPMaxNum(notifyNum, LINK_P2P_MAX_NUM, true));
        linkp2p->localIpcSignal[notifyNum] = locIpcSignals[i];
        linkp2p->remoteIpcSignal[notifyNum] = rmtIpcSignals[i];
        notifyNum++;
    }
    tagRemoteRes.p2pNotifyNum = notifyNum;
    HCCL_DEBUG("[%s] finish set localnotify & remotenotify info, notifyNum[%llu]", __func__, notifyNum);
    // transportAttr
    CHK_RET(link->GetTransportAttr(linkp2p->transportAttr));
    HCCL_DEBUG("[%s] finish set RemoteLinkP2pResParam info", __func__);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildOpRemoteLinkRoceResParam(
    const LINK& link, HccltagRemoteResV3& tagRemoteRes, bool isBackup, bool isRetry, bool isSecondBuild)
{
    u32 iter = isSecondBuild ? 2 : 0;
    HcclLinkRoceV2* linkRoce = isBackup ?
                                   &(tagRemoteRes.tagRemoteResPtr->linkRoce[AICPU_RETRY_LINKROCE_BACKUP + iter]) :
                                   &(tagRemoteRes.tagRemoteResPtr->linkRoce[AICPU_RETRY_LINKROCE_DEFAULT + iter]);
    if (!isRetry && linkRoce->localNotifyList != 0) {
        HCCL_INFO(
            "[%s]the linkRoce already exists, no need to refresh transport resource, localNotifyListPtr[%p], iter[%u]",
            __func__, reinterpret_cast<void*>(linkRoce->localNotifyList), iter);
        return HCCL_SUCCESS;
    }
    // localMem & remoteMem
    CHK_RET(link->GetLocalMemDetails(UserMemType::INPUT_MEM, (linkRoce->localMem)[INPUT]));
    CHK_RET(link->GetLocalMemDetails(UserMemType::OUTPUT_MEM, (linkRoce->localMem)[OUTPUT]));
    void* inbufferPtr = nullptr;
    void* outbufferPtr = nullptr;
    CHK_RET(link->GetRemoteMem(UserMemType::INPUT_MEM, &inbufferPtr));
    CHK_RET(link->GetRemoteMem(UserMemType::OUTPUT_MEM, &outbufferPtr));
    HCCL_DEBUG("[%s]inbufferPtr[%p], outbufferPtr[%p]", __func__, inbufferPtr, outbufferPtr);
    if (inbufferPtr == nullptr || outbufferPtr == nullptr) {
        HCCL_ERROR("[%s]inbufferPtr[%p], outbufferPtr[%p]", __func__, inbufferPtr, outbufferPtr);
        return HCCL_E_INTERNAL;
    }
    (linkRoce->remoteMem)[INPUT].addr = reinterpret_cast<u64>(inbufferPtr);
    (linkRoce->remoteMem)[OUTPUT].addr = reinterpret_cast<u64>(outbufferPtr);
    CHK_RET(link->GetRemoteMemKey(UserMemType::INPUT_MEM, &((linkRoce->remoteMem)[INPUT].key)));
    CHK_RET(link->GetRemoteMemKey(UserMemType::OUTPUT_MEM, &((linkRoce->remoteMem)[OUTPUT].key)));
    CHK_RET(link->GetRemoteMemSize(UserMemType::INPUT_MEM, (linkRoce->remoteMem)[INPUT].size));
    CHK_RET(link->GetRemoteMemSize(UserMemType::OUTPUT_MEM, (linkRoce->remoteMem)[OUTPUT].size));
    HCCL_DEBUG("[%s] finish set localMem & remoteMem info", __func__);
    // notifyValue & Key
    std::vector<AddrKey> notifyValueAddrKey;
    CHK_RET(link->GetLocalNotifyValueAddrKey(notifyValueAddrKey));
    linkRoce->notifyValue = notifyValueAddrKey[0].addr;
    linkRoce->notifyValueKey = notifyValueAddrKey[0].key;
    // QPInfo
    std::vector<HcclQpInfoV2> aiQpInfos;
    CHK_RET(link->GetAiQpInfo(aiQpInfos));
    u32 qpNum = aiQpInfos.size();
    if (qpNum > RDMA_QP_MAX_NUM || qpNum < 1) {
        return HCCL_E_INTERNAL;
    }
    std::copy_n(aiQpInfos.begin(), qpNum, linkRoce->QpInfo);
    linkRoce->qpsPerConnection = qpNum - static_cast<u32>(qpNum > 1); // 多QP数量或单QP模式

    // localnotify & remotenotify
    std::vector<AddrKey> notifyAddrKey;
    std::vector<HcclSignalInfo> signalInfos;
    CHK_RET(link->GetLocalRdmaNotify(signalInfos));
    CHK_RET(link->GetRemoteRdmaNotifyAddrKey(notifyAddrKey));
    if ((signalInfos.size() != notifyAddrKey.size()) || (signalInfos.size() < RDMA_NOTIFY_MIN_NUM)
        || (signalInfos.size() > RDMA_NOTIFY_MAX_NUM) || (notifyAddrKey.size() < RDMA_NOTIFY_MIN_NUM)
        || (notifyAddrKey.size() > RDMA_NOTIFY_MAX_NUM)
        || ((signalInfos.size() - RDMA_NOTIFY_MIN_NUM) % linkRoce->qpsPerConnection != 0)
        || ((notifyAddrKey.size() - RDMA_NOTIFY_MIN_NUM) % linkRoce->qpsPerConnection != 0)) {
        HCCL_ERROR(
            "[HcclCommunicator][BuildOpRemoteLinkRoceResParam] signalInfos %zu notifyAddrKey %zu "
            "qpsPerConnection %u",
            signalInfos.size(), notifyAddrKey.size(), linkRoce->qpsPerConnection);
        return HCCL_E_INTERNAL;
    }
    u64 notifyNum = (notifyAddrKey.size() - RDMA_NOTIFY_MIN_NUM) / linkRoce->qpsPerConnection
                    - static_cast<u32>(linkRoce->qpsPerConnection > 1);
    linkRoce->singleQPNotifyNum = notifyNum;

    u64 len = signalInfos.size() * sizeof(HcclSignalInfo);
    DeviceMem localNotifyListMem;
    CHK_RET(CopyVectorToDeviceMem(len, localNotifyListMem, signalInfos));
    linkRoce->localNotifyList = reinterpret_cast<u64>(localNotifyListMem.ptr());
    ibverbsLocalNotify_[tagRemoteRes.tagRemoteResPtr->tag].emplace_back(std::move(localNotifyListMem));

    len = notifyAddrKey.size() * sizeof(AddrKey);
    DeviceMem remoteNotifyListMem;
    CHK_RET(CopyVectorToDeviceMem(len, remoteNotifyListMem, notifyAddrKey));
    linkRoce->remoteNotifyList = reinterpret_cast<u64>(remoteNotifyListMem.ptr());
    ibverbsRemoteNotify_[tagRemoteRes.tagRemoteResPtr->tag].emplace_back(std::move(remoteNotifyListMem));

    HCCL_DEBUG(
        "[%s] finish set localnotify & remotenotify info, notifyNum[%llu], linkNotifyNum[%llu]", __func__, notifyNum,
        signalInfos.size());

    if (isBackup) {
        tagRemoteRes.roceNotifyNumBackup = linkRoce->singleQPNotifyNum;
        tagRemoteRes.qpNumBackup = linkRoce->qpsPerConnection;
    } else {
        tagRemoteRes.roceNotifyNum = linkRoce->singleQPNotifyNum;
        tagRemoteRes.qpNum = linkRoce->qpsPerConnection;
    }

    linkRoce->useAtomicWrite = link->GetIsUseAtomicWrite();
    HCCL_DEBUG(
        "[%s] finish set Qp info qpNum[%u], linkRoce->localNotifyList[0].resId[%llu], "
        "notifyNum[%u], isBackup[%d], isSecond[%d], qpPtr[%llu], useAtomicWrite[%d]",
        __func__, linkRoce->qpsPerConnection, signalInfos[0].resId, linkRoce->singleQPNotifyNum, isBackup,
        isSecondBuild, linkRoce->QpInfo[0].qpPtr, linkRoce->useAtomicWrite);
    return HCCL_SUCCESS;
}

template <typename T>
HcclResult HcclCommunicator::CreateListNode(T** resHostPtr, T** resDevicePtr)
{
    hostMemVec_.resize(hostMemVec_.size() + 1);
    CHK_RET(AllocAndClearHostMem(sizeof(T), hostMemVec_.back()));
    *resHostPtr = static_cast<T*>(hostMemVec_.back().get()->ptr());

    deviceMemVec_.resize(deviceMemVec_.size() + 1);
    CHK_RET(AllocAndClearDeviceMem(sizeof(T), deviceMemVec_.back()));

    *resDevicePtr = static_cast<T*>(deviceMemVec_.back().get()->ptr());
    // 初始化HcclRankRelationResV2中的tagRes链表
    ListCommonInit(&((*resDevicePtr)->nextTagRes), &((*resHostPtr)->nextTagRes));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildRemoteResByTag(
    const std::string& newTag, const u32& usrRankId, HcclRankRelationResV2*& rankRelationResHostPtr,
    HcclRankRelationResV2*& rankRelationResDevicePtr, [[maybe_unused]] bool isBackup, [[maybe_unused]] bool isRetry)
{
    HCCL_DEBUG(
        "[%s]start to add RemoteRes with newtag[%s] and remoteRankId[%u] to list", __func__, newTag.c_str(), usrRankId);
    if (rankTagRemoteRes_.find(usrRankId) == rankTagRemoteRes_.end()
        || rankTagRemoteRes_[usrRankId].find(newTag) == rankTagRemoteRes_[usrRankId].end()) {
        HccltagRemoteResV2* tagRemoteResHostPtr = nullptr;
        HccltagRemoteResV2* tagRemoteResDevicePtr = nullptr;
        CHK_RET(CreateListNode(&tagRemoteResHostPtr, &tagRemoteResDevicePtr));
        CHK_SAFETY_FUNC_RET(
            memcpy_s(tagRemoteResHostPtr->tag, sizeof(tagRemoteResHostPtr->tag), newTag.c_str(), newTag.length() + 1));
        tagRemoteResHostPtr->linkP2p.localIpcSignal[0].resId = INVALID_U64;
        tagRemoteResHostPtr->linkP2pSio.localIpcSignal[0].resId = INVALID_U64;
        tagRemoteResHostPtr->linkRoce[0].localNotifyList = 0;
        tagRemoteResHostPtr->linkRoce[1].localNotifyList = 0;
        tagRemoteResHostPtr->linkRoce[2].localNotifyList = 0;
        tagRemoteResHostPtr->linkRoce[3].localNotifyList = 0;
        ListCommonAddHead(
            &tagRemoteResDevicePtr->nextTagRes, &tagRemoteResHostPtr->nextTagRes, &rankRelationResHostPtr->nextTagRes,
            &rankRelationResDevicePtr->nextTagRes);
        HccltagRemoteResV3 tempTagRemoteRes;
        tempTagRemoteRes.tagRemoteResPtr = tagRemoteResHostPtr;
        rankTagRemoteRes_[usrRankId][newTag] = tempTagRemoteRes;
        HCCL_RUN_INFO(
            "[%s] successfully add RemoteRes to list with newtag[%s], remoteRankId[%u]"
            "rankRelationResHostPtr head addr[%p], nextHost[%p], preHost[%p], nextDevice[%p], preDevice[%p], "
            "tagRemoteResDevicePtr head addr[%p]",
            __func__, newTag.c_str(), usrRankId, &rankRelationResHostPtr->nextTagRes,
            rankRelationResHostPtr->nextTagRes.nextHost, rankRelationResHostPtr->nextTagRes.preHost,
            rankRelationResHostPtr->nextTagRes.nextDevice, rankRelationResHostPtr->nextTagRes.preDevice,
            &tagRemoteResDevicePtr->nextTagRes);
    } else {
        HCCL_DEBUG(
            "[%s] the RemoteRes with usr rankid[%u] tag[%s] has been added list", __func__, usrRankId, newTag.c_str());
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildRelationResByRemoteRankId(
    const TransportRequest& transportRequest, const LINK& link, HcclRankRelationResV2*& rankRelationResHostPtr,
    HcclRankRelationResV2*& rankRelationResDevicePtr)
{
    const u32 usrRankId = transportRequest.remoteUserRank;
    HCCL_INFO("[%s]start to add RelationRes with remote usr rankid[%u] to list", __func__, usrRankId);
    if (opResPara_.remoteRes[usrRankId].nextHostPtr != 0 && opResPara_.remoteRes[usrRankId].nextDevicePtr != 0) {
        rankRelationResHostPtr = reinterpret_cast<HcclRankRelationResV2*>(opResPara_.remoteRes[usrRankId].nextHostPtr);
        rankRelationResDevicePtr
            = reinterpret_cast<HcclRankRelationResV2*>(opResPara_.remoteRes[usrRankId].nextDevicePtr);
        HCCL_DEBUG(
            "[%s] RelationRes with remote usr rankid[%u] has been added to list, "
            "rankRelationResHostPtr[%p], rankRelationResDevicePtr[%p]",
            __func__, usrRankId, rankRelationResHostPtr, rankRelationResDevicePtr);
    } else {
        CHK_RET(CreateListNode(&rankRelationResHostPtr, &rankRelationResDevicePtr));
        opResPara_.remoteRes[usrRankId].nextHostPtr = reinterpret_cast<u64>(rankRelationResHostPtr);
        opResPara_.remoteRes[usrRankId].nextDevicePtr = reinterpret_cast<u64>(rankRelationResDevicePtr);
        rankRelationResHostPtr->remoteUsrRankId = usrRankId;
        rankRelationResHostPtr->remoteWorldRank = rankInfoList_[usrRankId].worldRank;
        HCCL_DEBUG(
            "[%s]successfully add RelationRes with remote usr rankid[%u] to list, rankRelationResHostPtr[%p],"
            "rankRelationResDevicePtr[%p]",
            __func__, usrRankId, rankRelationResHostPtr, rankRelationResDevicePtr);
    }
    // 刷新远端对应的cclbuffer
    std::vector<void*> extraMemVector;
    if (transportRequest.inputMemType == TransportMemType::CCL_INPUT && rankRelationResHostPtr->windowsIn == 0) {
        void* inbufferPtr = nullptr;
        CHK_RET(link->GetRemoteMem(UserMemType::INPUT_MEM, &inbufferPtr));
        rankRelationResHostPtr->windowsIn = reinterpret_cast<u64>(inbufferPtr);
    }
    if (transportRequest.outputMemType == TransportMemType::CCL_OUTPUT && rankRelationResHostPtr->windowsOut == 0) {
        void* outbufferPtr = nullptr;
        CHK_RET(link->GetRemoteMem(UserMemType::OUTPUT_MEM, &outbufferPtr));
        rankRelationResHostPtr->windowsOut = reinterpret_cast<u64>(outbufferPtr);
    }
    if (rankRelationResHostPtr->windowsExp == 0) {
        std::vector<void*> memPtrVec = {};
        CHK_RET(link->GetRemoteMem(&memPtrVec));
        if (memPtrVec.size() != 0) {
            rankRelationResHostPtr->windowsExp = reinterpret_cast<u64>(memPtrVec[0]);
            if (link->GetTransportType() == TransportType::TRANS_TYPE_P2P) {
                p2pCclBuf_[usrRankId] = memPtrVec[0];
            } else {
                cclBuf_[usrRankId] = memPtrVec[0];
            }
            rankRelationResHostPtr->windowsExp
                += cclBufferManager_.GetInCCLbufferSize() + cclBufferManager_.GetOutCCLbufferSize();
        }
    }
    HCCL_INFO(
        "group[%s] successfully set windowsIn & windowsOut & windowsExp info: userRank[%u], groupRank[%u], "
        "remoteRank[%u], windowsIn[0x%llx], InSize[0x%llx], windowOut[0x%llx], OutSize[0x%llx], "
        "windowExp[0x%llx], ExpSize[0x%llx]",
        identifier_.c_str(), GetUserRank(), GetGroupRank(), transportRequest.remoteUserRank,
        rankRelationResHostPtr->windowsIn, cclBufferManager_.GetInCCLbufferSize(), rankRelationResHostPtr->windowsOut,
        cclBufferManager_.GetOutCCLbufferSize(), rankRelationResHostPtr->windowsExp,
        cclBufferManager_.GetExpBufferSize());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ParseRemoteDataToMem(
    const OpCommTransport& opTransportResponse, const std::string& newTag, const HcclCMDType opType, bool isBackup,
    bool isRetry)
{
    HCCL_INFO("[%s] entry process newtag[%s], isBackup[%d]", __func__, newTag.c_str(), isBackup);
    std::set<u32> bsrTansportRank;
    for (auto& levelNSubCommTransport : opTransportResponse) {
        for (auto& singleSubCommTransport : levelNSubCommTransport) {
            u32 linkIdx = 0;
            for (auto& transportRequest : singleSubCommTransport.transportRequests) {
                if (transportRequest.isValid) {
                    auto tempLink = singleSubCommTransport.links[linkIdx];
                    HCCL_INFO(
                        "[%s]transportRequest.isUsedRdma[%d], isBackup[%d]", __func__, transportRequest.isUsedRdma,
                        isBackup);
                    if ((!transportRequest.isUsedRdma || tempLink->GetLinkType() == LinkType::LINK_SIO)
                        && (isBackup || isRetry)) {
                        HCCL_INFO(
                            "[%s]no need to add p2p backup Link resource, transportRequest.isUsedRdma[%d], "
                            "isBackup[%d]",
                            __func__, transportRequest.isUsedRdma, isBackup);
                        linkIdx++;
                        continue;
                    }
                    HcclRankRelationResV2* rankRelationResHostPtr = nullptr;
                    HcclRankRelationResV2* rankRelationResDevicePtr = nullptr;
                    CHK_RET(BuildRelationResByRemoteRankId(
                        transportRequest, tempLink, rankRelationResHostPtr, rankRelationResDevicePtr));
                    const u32 usrRankId = transportRequest.remoteUserRank;
                    HCCL_INFO(
                        "[%s]successfully BuildRelationResByRemoteRankId with remote usr rankid[%u], "
                        "rankRelationResHostPtr[%p], rankRelationResDevicePtr[%p], newTage[%s]",
                        __func__, usrRankId, rankRelationResHostPtr, rankRelationResDevicePtr, newTag.c_str());
                    CHK_RET(BuildRemoteResByTag(
                        newTag, usrRankId, rankRelationResHostPtr, rankRelationResDevicePtr, isBackup, isRetry));
                    // transport信息保存（notify、qp）
                    if (!transportRequest.isUsedRdma || tempLink->GetLinkType() == LinkType::LINK_SIO) {
                        // sdma -> P2P
                        CHK_RET(BuildOpRemoteLinkP2pResParam(
                            tempLink, rankTagRemoteRes_[usrRankId][newTag], transportRequest.linkType));
                    } else {
                        // rdma -> roce
                        bool isSecondBuild = false;
                        if (opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV
                            && bsrTansportRank.find(transportRequest.remoteUserRank) != bsrTansportRank.end()) {
                            isSecondBuild = true;
                        }
                        bsrTansportRank.insert(transportRequest.remoteUserRank);
                        CHK_RET(BuildOpRemoteLinkRoceResParam(
                            tempLink, rankTagRemoteRes_[usrRankId][newTag], isBackup, isRetry, isSecondBuild));
                    }
                    HCCL_INFO(
                        "[%s] successfully add RemoteRes to list with newtag[%s] rankRelationResHostPtr "
                        "head addr[%p], nextHost[%p], preHost[%p], nextDevice[%p], preDevice[%p], "
                        "rankRelationResDevicePtr head addr[%p]",
                        __func__, newTag.c_str(), &rankRelationResHostPtr->nextTagRes,
                        rankRelationResHostPtr->nextTagRes.nextHost, rankRelationResHostPtr->nextTagRes.preHost,
                        rankRelationResHostPtr->nextTagRes.nextDevice, rankRelationResHostPtr->nextTagRes.preDevice,
                        &rankRelationResDevicePtr->nextTagRes);
                    HCCL_INFO(
                        "[%s] create link success with newtag[%s], linkIdx[%u], isBackup[%d], usrRankId[%u]", __func__,
                        newTag.c_str(), linkIdx, isBackup, usrRankId);
                }
                linkIdx++;
            }
        }
    }
    HCCL_DEBUG("[%s] process success newtag[%s]", __func__, newTag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildOpRemoteResParam(
    const AlgResourceResponse& algResource, const std::string& newTag, const HcclCMDType opType, bool isRetry)
{
    HCCL_DEBUG("[%s]start ParseRemoteDataToMem, IsEnableBackupLink[%d]", __func__, IsEnableBackupLink());
    CHK_RET(ParseRemoteDataToMem(algResource.opTransportResponse, newTag, opType, false, isRetry));
    if (IsEnableBackupLink()) {
        HCCL_DEBUG("[%s]start Parse backupRemoteDataToMem, IsEnableBackupLink[%d]", __func__, IsEnableBackupLink());
        CHK_RET(ParseRemoteDataToMem(algResource.opTransportResponseBackUp, newTag, opType, true, isRetry));
    }
    if (deviceType_ == DevType::DEV_TYPE_910_93 || deviceType_ == DevType::DEV_TYPE_910B) {
        opResPara_.notifysize = 4; // 910B & 910_93 每个notify占4个字节
    } else {
        opResPara_.notifysize = 8; // 其他芯片类型每个notify占8个字节
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CopyHostListResToDeviceParam(
    const std::string& newTag, const ListCommon* headHostList, const u64 size)
{
    ListCommon* nextHostList = reinterpret_cast<ListCommon*>(headHostList->nextHost);
    ListCommon* nextDeviceList = reinterpret_cast<ListCommon*>(headHostList->nextDevice);
    // 该tag已分配过资源，只需刷新单节点（精确匹配）；否则为首次分配，批量拷贝前N个节点
    bool isRefreshSingleNode = (newTagResAlloced_.find(newTag) != newTagResAlloced_.end());
    // ListCommonAddHead把新节点头插，原头节点的preHost/preDevice被改写，两节点都需要刷新到device
    constexpr uint32_t UPDATE_NODE_NUM = 2;
    uint32_t updateNodeCnt = 0;

    while (nextHostList != headHostList && updateNodeCnt < UPDATE_NODE_NUM) {
        HCCL_INFO(
            "[HcclCommunicator][CopyHostListResToDeviceParam] remote resource, tag[%s], head Host List[%p], next "
            "Host List[%p],next Device List[%p]",
            newTag.c_str(), headHostList, nextHostList, nextDeviceList);
        if (isRefreshSingleNode) {
            // 刷新已有资源：遍历链表找到匹配的tag，只拷贝该节点，避免全量刷新
            std::string curTag = (size == sizeof(HccltagLocalResV2)) ?
                                     reinterpret_cast<HccltagLocalResV2*>(nextHostList)->tag :
                                     reinterpret_cast<HccltagRemoteResV2*>(nextHostList)->tag;
            if (curTag == newTag) {
                CHK_RET(hrtMemSyncCopy(
                    static_cast<void*>(nextDeviceList), size, static_cast<void*>(nextHostList), size,
                    HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
                break;
            }
        } else {
            // 首分配置：拷贝前UPDATE_NODE_NUM个节点到device，减少H2D次数
            CHK_RET(hrtMemSyncCopy(
                static_cast<void*>(nextDeviceList), size, static_cast<void*>(nextHostList), size,
                HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
            updateNodeCnt++;
        }
        nextDeviceList = reinterpret_cast<ListCommon*>(nextHostList->nextDevice);
        nextHostList = reinterpret_cast<ListCommon*>(nextHostList->nextHost);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CopyHostOpResToDeviceParam(const std::string& newTag)
{
    // 1、将opResPara_，H2D到device
    CHK_RET(hrtMemSyncCopy(
        opResDevicePara_.ptr(), sizeof(HcclOpResParam), static_cast<void*>(&opResPara_), sizeof(HcclOpResParam),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    HCCL_DEBUG(
        "[HcclCommunicator][CopyHostOpResToDeviceParam] tag[%s] local rankId[%u] workspace[%p] "
        "workspacesize[%lu] ranksize[%u], cclbuffersize[%lu], cclinbuffer[%p], ccloutbuffer[%p], "
        "remote winStart[%u], remote rWinOffset[%u], hostStateInfo[%p], aicpuStateInfo[%p], notifysize[%u], "
        "sizeOfAiRMAInfo[%u],aiRMAInfo[%u]",
        newTag.c_str(), userRank_, opResPara_.mc2WorkSpace.workSpace, opResPara_.mc2WorkSpace.workSpaceSize,
        opResPara_.rankSize, opResPara_.winSize, opResPara_.localWindowsIn, opResPara_.localWindowsOut,
        opResPara_.rWinStart, opResPara_.rWinOffset, opResPara_.hostStateInfo, opResPara_.aicpuStateInfo,
        opResPara_.notifysize, opResPara_.sizeOfAiRMAInfo, opResPara_.aiRMAInfo);
    // 2、将opResPara_中localres的tagRes，H2D到device
    HCCL_DEBUG(
        "[HcclCommunicator][CopyHostOpResToDeviceParam] local resource, tag[%s] streamNum[%u] signalNum[%u]",
        newTag.c_str(), opResPara_.localRes.streamNum, opResPara_.localRes.signalNum);
    CHK_RET(CopyHostListResToDeviceParam(
        newTag, reinterpret_cast<ListCommon*>(&opResPara_.localRes.nextTagRes), sizeof(HccltagLocalResV2)));
    // 3、遍历rank中tag资源，H2D到device
    CHK_RET(CopyHostOpRemoteResToDeviceParam(newTag));
    HCCL_DEBUG("[HcclCommunicator][CopyHostOpResToDeviceParam] copy host resource success!, tag[%s]", newTag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CopyHostAirmaInfoToDeviceParam(
    const std::string& newTag, [[maybe_unused]] const HcclCMDType opType, const rtStream_t aiCpuStream)
{
    HCCL_INFO("[HcclCommunicator][%s] Start prepare.", __func__);
    CHK_PTR_NULL(aiRMAInfoMem_);
    HcclAiRMAInfo* aiRMAInfoPtr = reinterpret_cast<HcclAiRMAInfo*>(aiRMAInfoMem_->ptr());
    CHK_PTR_NULL(aiRMAInfoPtr);

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

    opResPara_.sizeOfAiRMAInfo = static_cast<u64>(sizeof(HcclAiRMAInfo));
    CHK_RET(DeviceMem::alloc(aiRMAInfoDev_, opResPara_.sizeOfAiRMAInfo));
    opResPara_.aiRMAInfo = reinterpret_cast<u64>(aiRMAInfoDev_.ptr());

    CHK_RET(hrtMemAsyncCopy(
        aiRMAInfoDev_.ptr(), aiRMAInfoDev_.size(), aiRMAInfoMem_->ptr(), aiRMAInfoDev_.size(),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE, aiCpuStream));
    HCCL_INFO(
        "[%s] tag[%s] curRankId[%u] rankNum[%u] qpNum[%u] aiRMAInfo[%p] sizeOfAiRMAInfo[%llu] "
        "sizeOfAiRMAWQ[%u] sizeOfAiRMACQ[%u] sizeOfAiRMAMem[%u] sqPtr[%p] sqSize[%llu] sqCount[%zu] "
        "scqPtr[%p] scqSize[%llu] scqCount[%zu] rqPtr[%p] rqSize[%llu] rqCount[%zu] rcqPtr[%p] "
        "rcqSize[%llu] rcqCount[%zu] memPtr[%p] memSize[%llu] memCount[%zu] memDetailCount[%zu],opResPara_.aiRMAInfo",
        __func__, newTag.c_str(), aiRMAInfoPtr->curRankId, aiRMAInfoPtr->rankNum, aiRMAInfoPtr->qpNum,
        opResPara_.aiRMAInfo, opResPara_.sizeOfAiRMAInfo, aiRMAInfoPtr->sizeOfAiRMAWQ, aiRMAInfoPtr->sizeOfAiRMACQ,
        aiRMAInfoPtr->sizeOfAiRMAMem, aiRMAInfoPtr->sqPtr, aiSqDev_.size(), aiSqMem_->size(), aiRMAInfoPtr->scqPtr,
        aiScqDev_.size(), aiScqMem_->size(), aiRMAInfoPtr->rqPtr, aiRqDev_.size(), aiRqMem_->size(),
        aiRMAInfoPtr->rcqPtr, aiRcqDev_.size(), aiRcqMem_->size(), aiRMAInfoPtr->memPtr, aiMemDev_.size(),
        aiMemMem_->size(), aiMemDetailsMem_->size());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildOpResParam(
    const std::string& algName, const AlgResourceResponse& algResource, const std::string& newTag,
    const HcclCMDType opType, const rtStream_t aicpuStream)
{
    opResPara_.localUsrRankId = userRank_;
    opResPara_.rankSize = userRankSize_;

    bool isUseUserMem = isUserMemRegisted_ && !userMemMap_.empty();
    if (!isUseUserMem) {
        opResPara_.winSize = algResource.cclInputMem.size();
        opResPara_.localWindowsIn = reinterpret_cast<u64>(algResource.cclInputMem.ptr());
        opResPara_.localWindowsOut = reinterpret_cast<u64>(algResource.cclOutputMem.ptr());
    } else {
        opResPara_.winSize = userMemMap_.begin()->second->size();
        opResPara_.localWindowsIn = reinterpret_cast<u64>(userMemMap_.begin()->second->ptr());
        opResPara_.localWindowsOut = reinterpret_cast<u64>(userMemMap_.begin()->second->ptr());
    }
    // 填充Exp相关信息 当前该块内存大小恒为1M
    opResPara_.winExpSize = EXP_BUFFER_SIZE;
    opResPara_.localWindowsExp = reinterpret_cast<u64>(cclBufferManager_.GetCommExpBuffer().ptr());
    HCCL_INFO(
        "[HcclCommunicator][%s] isUseUserMem[%d], winSize[%llu], localWindowsIn[%llu],"
        "localWindowsOut[%llu], localWindowsExp[%llu]",
        __func__, isUseUserMem, opResPara_.winSize, opResPara_.localWindowsIn, opResPara_.localWindowsOut,
        opResPara_.localWindowsExp);

    CHK_SAFETY_FUNC_RET(
        memcpy_s(opResPara_.hcomId, sizeof(opResPara_.hcomId), identifier_.c_str(), identifier_.length() + 1));

    opResPara_.config.deterministic = GetDeterministicConfig();
    opResPara_.config.highPerfEnable = 0;
    aclrtFloatOverflowMode floatOverflowMode = ACL_RT_OVERFLOW_MODE_UNDEF;
    CHK_RET(hrtGetDeviceSatMode(&floatOverflowMode));
    opResPara_.config.floatOverflowMode = floatOverflowMode;
    opResPara_.config.taskMonitorInterval = GetExternalInputDfsTaskMonitorInterval();
    bool isSupportAtomicWrite = false; // 涉及到任务编排，当前不能只判断本机驱动版本是否支持
    opResPara_.config.isSupportAtomicWrite = static_cast<u8>(isSupportAtomicWrite);
    opResPara_.config.notifyWaitTime
        = (GetExternalInputHcclExecTimeoutSet() != HcclExecTimeoutSet::HCCL_EXEC_TIMEOUT_NOT_SET
           || commConfig_.GetConfigExecTimeOutSet()) ?
              commConfig_.GetConfigExecTimeOut() :
              NOTIFY_DEFAULT_WAIT_TIME;
    opResPara_.config.linkTimeOut = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());
    opResPara_.config.retryEnable = static_cast<u8>(retryEnable_);
    opResPara_.config.interHccsDisable = GetExternalInputInterHccsDisable();
    opResPara_.config.multiQpThreshold = GetExternalInputMultiQpThreshold();
    opResPara_.rWinStart = offsetof(HcclOpResParam, remoteRes);
    opResPara_.rWinOffset = sizeof(RemoteResPtr);
    opResPara_.notifysize = 0;
    opResPara_.lockAddr = hostDeviceLock_->GetDevMemAddr();
    opResPara_.utraceStatusFlag = GetExternalInputHcclEnableEntryLog();
    DeviceMem tinySendRecvMem;
    CHK_RET(implAlg_->GetTinyMem(tinySendRecvMem));
    opResPara_.tinyMem = reinterpret_cast<u64>(tinySendRecvMem.ptr());
    opResPara_.tinyMemSize = reinterpret_cast<u64>(tinySendRecvMem.size());
    opResPara_.opEntry = GetExternalInputHcclEnableEntryLog();
    opResPara_.hcclSdmaQos = GetHcclQos();

    CHK_RET(BuildOpLocalResParam(algResource, newTag));
    CHK_RET(BuildOpRemoteResParam(algResource, newTag, opType));
    CHK_RET(BuildOpTopoResParam(algName, algResource));
    CHK_RET(BuildOpRetryParam(algResource, newTag));
    CHK_RET(BuildZeroCopyParam());
    CHK_RET(BuildAicpuCustomParam());
    CHK_RET(BuildAicpuOrderLaunchNotify()); // 先申请device侧的关于按序下发的Notify内存
    if (algName == "RunAlltoAllAivDirect") {
        // AIV直驱ROCE
        CHK_RET(BuildAiRmaInfoParam(newTag, algName, opType));
        CHK_RET(CopyHostAirmaInfoToDeviceParam(newTag, opType, aicpuStream));
    }
    CHK_RET(CopyHostOpResToDeviceParam(newTag));
    HCCL_RUN_INFO(
        "[%s]build aicpu unfold resource success, tag[%s] rWinStart[%u] rWinOffset[%u] opEntry[%d]", __func__,
        newTag.c_str(), opResPara_.rWinStart, opResPara_.rWinOffset, opResPara_.opEntry);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::BuildCustomOpResParam()
{
    // custom进程需要刷新h2d/d2h内存
    opResPara_.kfcControlTransferH2DParams = customControlTransferH2D_->GetCommunicateParams();
    opResPara_.kfcStatusTransferD2HParams = customStatusTransferD2H_->GetCommunicateParams();
    CHK_RET(hrtMemSyncCopy(
        opResDevicePara_.ptr(), sizeof(HcclOpResParam), static_cast<void*>(&opResPara_), sizeof(HcclOpResParam),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::RegisterDfxInfo(
    const OpParam& param, AlgType algType, const std::vector<Stream>& slaveStreams, bool isAiv, const std::string& tag)
{
    u64 count = 0;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_RESERVED;
    switch (param.opType) {
        case HcclCMDType::HCCL_CMD_SEND:
        case HcclCMDType::HCCL_CMD_RECEIVE:
        case HcclCMDType::HCCL_CMD_BATCH_SEND_RECV:
            count = param.GetDataCount(userRank_);
            dataType = param.GetDataType();
            HCCL_PROFILER_ADD_TAG_SENDRECV(param.tag, identifier_, GetWorkflowMode());
            HCCL_PROFILER_ADD_GROUPRANK_SENDRECV(identifier_, userRankSize_, userRank_, param.dstRank);
            break;
        case HcclCMDType::HCCL_CMD_ALLTOALL:
        case HcclCMDType::HCCL_CMD_ALLTOALLV:
        case HcclCMDType::HCCL_CMD_ALLTOALLVC:
            CHK_RET(AddGroupTagInfo(param.tag, isAiv));
            count = param.All2AllDataDes.sendCount;
            dataType = param.All2AllDataDes.sendType;
            break;
        default:
            CHK_RET(AddGroupTagInfo(param.tag, isAiv));
            count = param.GetDataCount(userRank_);
            dataType = param.GetDataType();
    }

    if (GetExternalInconsistentCheckSwitch() == InconsistentCheckMode::ON) {
        if (param.opType != HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
            OpInfoDesc opInfo;
            opInfo.opType = param.opType;
            opInfo.dataType = dataType;
            opInfo.reduceOp = param.reduceType;
            opInfo.count = count;
            opInfo.root = param.root;
            opInfo.isValid = true;
            AddOpInfoToHeartBeat(opInfo, tag);
        }
    }

    // task exception使用: 算子计数，算子入参信息(src/dst/datatype/reducetype)
    HCCL_PROFILER_ADD_OPDATA_OP(
        param.tag, count, param.inputPtr, param.outputPtr, dataType, param.root, identifier_, param.reduceType);
    // 记录主流相关信息, 给profiling和task exception使用
    HCCL_PROFILER_ADD_STREAM_BY_STREAMID(param.stream.id(), param.tag, 0, algType);
    if (((GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE)
         && !hccl::ProfilingManagerPub::GetAdditionInfoState() && !hccl::ProfilingManagerPub::GetTaskApiState())
        && !param.isCapture) {
        return HCCL_SUCCESS;
    }
    // 从流信息profiling开关打开的话再注册
    for (u32 streamIndex = 0; streamIndex < slaveStreams.size(); streamIndex++) {
        HCCL_PROFILER_ADD_STREAM_BY_STREAMID(slaveStreams[streamIndex].id(), param.tag, streamIndex + 1, algType);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetReportHcclMC2Info(const Stream& kfcStream, const std::vector<Stream>& aicpuStreams)
{
    hcclMc2Info_.groupName = hrtMsprofGetHashId(identifier_.c_str(), identifier_.length());
    hcclMc2Info_.rankSize = userRankSize_;
    hcclMc2Info_.rankId = userRank_;
    hcclMc2Info_.usrRankId = realUserRank_;
    hcclMc2Info_.aicpuKfcStreamId = static_cast<uint32_t>(kfcStream.id());
    hcclMc2Info_.reserve = 0;
    const uint32_t ONCE_REPORT_STREAM_NUM_MAX = 8;
    for (uint32_t streamIndex = 0, reportId = 0; streamIndex < aicpuStreams.size(); streamIndex++) {
        HCCL_INFO("streamIndex:%u, reportId:%u, streamId:%d, ", streamIndex, reportId, aicpuStreams[streamIndex].id());
        hcclMc2Info_.commStreamIds[reportId++] = aicpuStreams[streamIndex].id();
        if (reportId == ONCE_REPORT_STREAM_NUM_MAX) {
            hcclMc2Info_.commStreamSize = reportId;
            CHK_RET(ProfilingManagerPub::CallMsprofReportMc2CommInfo(
                hrtMsprofSysCycleTime(), &hcclMc2Info_, sizeof(hcclMc2Info_)));
            reportId = 0;
        }
        if (streamIndex == (aicpuStreams.size() - 1)) {
            HCCL_INFO("streamIndex:%u, reportId:%u, streamId:%d", streamIndex, reportId, opMainStream_.id());
            hcclMc2Info_.commStreamIds[reportId++] = opMainStream_.id();
            hcclMc2Info_.commStreamSize = reportId;
            CHK_RET(ProfilingManagerPub::CallMsprofReportMc2CommInfo(
                hrtMsprofSysCycleTime(), &hcclMc2Info_, sizeof(hcclMc2Info_)));
            reportId = 0;
        }
    }
    if (aicpuStreams.empty()) {
        HCCL_INFO("only exist main stream, streamId:%d", opMainStream_.id());
        hcclMc2Info_.commStreamIds[0] = opMainStream_.id();
        hcclMc2Info_.commStreamSize = 1; // 只有主流1条
        CHK_RET(ProfilingManagerPub::CallMsprofReportMc2CommInfo(
            hrtMsprofSysCycleTime(), &hcclMc2Info_, sizeof(hcclMc2Info_)));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::OrchestrateAicpu(
    const HcclCMDType& opType, const std::string& algName, const OpParam& param, const AlgResourceResponse& algResource,
    const std::string& newTag, AlgType algType, bool isCustom, bool needIncreLink, bool needRecreateAlltoallComm)
{
    uint64_t streamMode = 0;
    CHK_RET(hrtStreamGetMode(param.stream.ptr(), &streamMode));
    rtStream_t aicpuStream;
    Mc2AiCpuStreamAllocAndGet(streamMode, aicpuStream); // aicpuStream需要在首次下发时申请
    if (!isContextLaunched_) {
        // 1、通信域内首次下发，从algResource中获取资源，H2D刷新资源，launch init
        rtStream_t aicpuInitStream;
        Mc2AiCpuInitStreamAllocAndGet(streamMode, aicpuInitStream); // 使用aicpuInitStream_下初始化kernel
        Stream tmpStream(aicpuInitStream);
        HCCL_DEBUG("%s ContextLaunched, aicpuInitStream:%p, aicpuStream:%p", __func__, aicpuInitStream, aicpuStream);
        CHK_RET(AicpuResourceInit(algName, algResource, newTag, aicpuInitStream, opType, isCustom));
        CHK_RET(GetReportHcclMC2Info(tmpStream, algResource.slaveDevStreams));
        CHK_RET(SetAicpuUnfoldFlag());
    } else if (
        newTagResAlloced_.find(newTag) == newTagResAlloced_.end()
        || (opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV && needIncreLink) || needRecreateAlltoallComm) {
        // 2、通信域内非首次，但是有新的newTag，查看是否需要补充资源。
        PetersonLockGuard guard(hostDeviceLock_.get());
        CHK_PRT_RET(
            guard.IsLockFailed(), HCCL_ERROR("[HcclCommunicator][OrchestrateAicp] hostDeviceLock lock failed"),
            HCCL_E_INTERNAL);
        CHK_RET(AicpuResourceRefresh(algResource, newTag, opType));
    }
    HCCL_DEBUG(
        "%s isContextLaunched[%u], needRecreateAlltoallComm[%u]", __func__, isContextLaunched_,
        needRecreateAlltoallComm);
    bool isUsedMainStream = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
    // inplace支持重执行的stream资源处理逻辑
    bool isHcclOpInplace = IsHcclOpInplace(opType, param, userRank_, userRankSize_, isInplaceStatus_);
    if ((retryOrigWorkflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) && retryEnable_
        && isHcclOpInplace
        && (opType == HcclCMDType::HCCL_CMD_ALLREDUCE || opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER)) {
        isUsedMainStream = true;
    }
    AicpuOpTiling opTilingInfo;
    opTilingInfo.algName = algName;
    opTilingInfo.newTag = newTag;
    opTilingInfo.algType = algType;
    opTilingInfo.isUsedMainStream = isUsedMainStream;
    opTilingInfo.dumpDebug = GetExternalInputHcclDumpDebug();
    aclrtFloatOverflowMode floatOverflowMode = ACL_RT_OVERFLOW_MODE_UNDEF;
    CHK_RET(hrtGetDeviceSatMode(&floatOverflowMode));
    opTilingInfo.floatOverflowMode = floatOverflowMode;
    HcclResult ret = HCCL_SUCCESS;
    // 根据算子类型，获取 Aicpu Kernel 名称
    auto iter = HCOM_CMD_TYPE_STR_MAP.find(opType);
    CHK_PRT_RET(
        (iter == HCOM_CMD_TYPE_STR_MAP.end()),
        HCCL_ERROR("[%s] RunAicpuRpcSrvLaunchV2 kernel not found, opType=[%d]", __func__, static_cast<int>(opType)),
        HCCL_E_INTERNAL);
    std::string kernelName = std::string("RunAicpuRpcSrvLaunchV2") + "_" + iter->second;
    ret = AicpuKfcTilingDataLaunchExt(param, opType, opResDevicePara_, kernelName, opTilingInfo, isCustom);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcclCommunicator][OrchestrateAicpu]aicpu unfold launch kernel[%s] failed. ret[%d] inputPtr[%p]"
            "outputPtr[%p] count[%llu] dataType[%s] op[%s]",
            kernelName.c_str(), ret, param.inputPtr, param.outputPtr, param.DataDes.count,
            GetDataTypeEnumStr(param.DataDes.dataType).c_str(), GetReduceOpEnumStr(param.reduceType).c_str());
        return ret;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CalcTinySendRecvMem(
    const OpParam& opParam, AlgResourceResponse& algResResponse, DeviceMem& tinySendRecvMem) const
{
    u64 sendCount = 0;
    u64 recvCount = 0;
    if (opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        for (u32 i = 0; i < userRankSize_; i++) {
            u64 curSendCount = *(static_cast<const u64*>(opParam.All2AllDataDes.sendCounts) + i)
                               + *(static_cast<const u64*>(opParam.All2AllDataDes.sdispls) + i);
            sendCount = std::max(sendCount, curSendCount);
            u64 curRecvCount = *(static_cast<const u64*>(opParam.All2AllDataDes.recvCounts) + i)
                               + *(static_cast<const u64*>(opParam.All2AllDataDes.rdispls) + i);
            recvCount = std::max(recvCount, curRecvCount);
        }
    } else {
        for (u32 i = 0; i < userRankSize_; i++) {
            sendCount
                += *(static_cast<const u64*>(opParam.All2AllDataDes.sendCountMatrix) + userRank_ * userRankSize_ + i);
            recvCount
                += *(static_cast<const u64*>(opParam.All2AllDataDes.sendCountMatrix) + userRank_ + userRankSize_ * i);
        }
    }

    u32 sendTypeSize = 0, recvTypeSize = 0;
    CHK_RET(SalGetDataTypeSize(opParam.All2AllDataDes.sendType, sendTypeSize));
    CHK_RET(SalGetDataTypeSize(opParam.All2AllDataDes.recvType, recvTypeSize));

    // 在sendCount/recvCount全0时, 使用tinySendRecvMem, 避免使用空deviceMem
    algResResponse.paramInputMem = sendCount == 0 ? DeviceMem::create(tinySendRecvMem.ptr(), tinySendRecvMem.size()) :
                                                    DeviceMem::create(opParam.inputPtr, sendCount * sendTypeSize);
    algResResponse.paramOutputMem = recvCount == 0 ? DeviceMem::create(tinySendRecvMem.ptr(), tinySendRecvMem.size()) :
                                                     DeviceMem::create(opParam.outputPtr, recvCount * recvTypeSize);

    HCCL_INFO(
        "[HcclCommunicator][CalcTinySendRecvMem] senMem addr[%p], sendSize[%llu], "
        "RecvMem addr[%p], RecvSize[%llu],",
        algResResponse.paramInputMem.ptr(), algResResponse.paramInputMem.size(), algResResponse.paramOutputMem.ptr(),
        algResResponse.paramOutputMem.size());
    return HCCL_SUCCESS;
}

bool HcclCommunicator::HasRoceTransportLinks(OpCommTransport& opTransportReq) const
{
    for (u32 levelIndex = 0; levelIndex < opTransportReq.size(); levelIndex++) {
        for (u32 ringIndex = 0; ringIndex < opTransportReq[levelIndex].size(); ringIndex++) {
            SingleSubCommTransport& reqSingleSubComm = opTransportReq[levelIndex][ringIndex];
            for (u32 rankIndex = 0; rankIndex < reqSingleSubComm.transportRequests.size(); rankIndex++) {
                TransportRequest& transportRequest = reqSingleSubComm.transportRequests[rankIndex];
                if (transportRequest.isUsedRdma) {
                    return true;
                }
            }
        }
    }
    return false;
}

HcclResult HcclCommunicator::CleanTransportLinks(OpCommTransport& opTransportReq, OpCommTransport& opTransportResponse)
{
    for (u32 levelIndex = 0; levelIndex < opTransportReq.size(); levelIndex++) {
        for (u32 ringIndex = 0; ringIndex < opTransportReq[levelIndex].size(); ringIndex++) {
            SingleSubCommTransport& reqSingleSubComm = opTransportReq[levelIndex][ringIndex];
            SingleSubCommTransport& respSingleSubComm = opTransportResponse[levelIndex][ringIndex];
            for (u32 rankIndex = 0; rankIndex < reqSingleSubComm.transportRequests.size(); rankIndex++) {
                TransportRequest& transportRequest = reqSingleSubComm.transportRequests[rankIndex];
                CHK_PRT_RET(
                    rankIndex >= respSingleSubComm.links.size(),
                    HCCL_ERROR(
                        "[CleanTransportLinks] The remote rank_id[%u] is larger than the existent respSingleSubComm "
                        "map "
                        "size[%u]",
                        rankIndex, respSingleSubComm.links.size()),
                    HCCL_E_PARA);
                if (respSingleSubComm.links[rankIndex] != nullptr
                    && respSingleSubComm.links[rankIndex]->GetLinkType() != hccl::LinkType::LINK_RESERVED
                    && !transportRequest.isUsedRdma) {
                    HCCL_INFO(
                        "[CleanTransportLinks] The link to remote userRank[%u] has existed",
                        transportRequest.remoteUserRank);
                    continue;
                }
                respSingleSubComm.links[rankIndex] = nullptr;
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllocAlgNotifys(
    const std::string& tag, const NotifyLoadType notifyLoadType, const u32 notifyNum,
    std::vector<std::shared_ptr<LocalNotify>>& notifiesMain, std::vector<std::shared_ptr<LocalNotify>>& notifiesAux)
{
    std::vector<std::shared_ptr<LocalNotify>> notifys(notifyNum, nullptr);
    CHK_RET(queueNotifyManagerRefac_->Alloc(tag, notifyNum, notifys, notifyLoadType));

    u32 signalNum = notifyNum >> 1;
    notifiesMain.resize(signalNum);
    notifiesAux.resize(signalNum);
    for (u32 i = 0; i < signalNum; i++) {
        notifiesMain[i] = notifys[i << 1];
        notifiesAux[i] = notifys[(i << 1) + 1];
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllocAlgResource(
    const std::string& newTag, HcclCMDType opType, const OpParam& opParam, AlgResourceRequest& resRequest,
    AlgResourceResponse& algResResponse, bool selectAivAlg)
{
    HcclResult ret = HCCL_SUCCESS;
    bool isGraphZeroCopyAlgAlloc = false;
    // 只有aicpu模式下才需要申请从流和相关的notify资源，isNeedSlaveStream为true就代表算子下发是aicpu模式
    bool isNeedSlaveStream = !selectAivAlg && opParam.aicpuUnfoldMode;
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB
        && !IsForceAicpuOpBaseMode(opParam, opType)) {
        isGraphZeroCopyAlgAlloc = resRequest.isInGraphCaptureZeroCopy;
        if (isGraphZeroCopyAlgAlloc) {
            if (resRequest.scratchMemSize > 0) {
                algResResponse.scratchMem
                    = DeviceMem::create(cclBufferManager_.GetOutCCLbuffer().ptr(), resRequest.scratchMemSize);
            }
        } else if (resRequest.scratchMemSize > 0) {
            algResResponse.scratchMem = GetWorkspaceScracthMem(opParam.tag, resRequest.scratchMemSize);
        }

        if (resRequest.streamNum > 0) {
            if (isGraphZeroCopyAlgAlloc) {
                CHK_RET(opStreamManager_->RegisterMaster(opParam.stream));
                algResResponse.slaveStreams
                    = opStreamManager_->AllocSlaves(StreamType::STREAM_TYPE_ONLINE, resRequest.streamNum);
                CHK_PRT_RET(
                    algResResponse.slaveStreams.empty(),
                    HCCL_ERROR(
                        "[AllocAlgResource]tag[%s] get slave stream failed, "
                        "expect to get size [%u], but only alloc 0.",
                        newTag.c_str(), resRequest.streamNum),
                    HCCL_E_INTERNAL);
            } else {
                // aicpu模式下不申请host从流
                if (!opParam.aicpuUnfoldMode) {
                    algResResponse.slaveStreams = GetWorkspaceSubStreams(opParam.tag, resRequest.streamNum);
                }
            }
        }
    } else if (
        GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE || IsForceAicpuOpBaseMode(opParam, opType)) {
        CHK_RET(AllocOpBaseModeScratchMem(opType, opParam, resRequest, algResResponse));
        if ((resRequest.streamNum > 0) && !selectAivAlg) {
            CHK_RET(opStreamManager_->RegisterMaster(opParam.stream));
            algResResponse.slaveStreams
                = opStreamManager_->AllocSlaves(StreamType::STREAM_TYPE_ONLINE, resRequest.streamNum);
            CHK_PRT_RET(
                algResResponse.slaveStreams.empty(),
                HCCL_ERROR(
                    "[AllocAlgResource]tag[%s] get slave stream failed, "
                    "expect to get size [%u], but only alloc 0.",
                    newTag.c_str(), resRequest.streamNum),
                HCCL_E_INTERNAL);
        }
    } else {
        HCCL_ERROR("[AllocAlgResource]WorkflowMode is not set.");
        return HCCL_E_PARA;
    }

    if (isNeedSlaveStream && ((userRankSize_ != 1) || IsForceAicpuOpBaseMode(opParam, opType))) {
        CHK_RET(opStreamManager_->RegisterMaster(opParam.stream));
        algResResponse.slaveDevStreams
            = opStreamManager_->AllocSlaves(StreamType::STREAM_TYPE_DEVICE, LOCAL_STREAM_MAX_NUM);
        CHK_PRT_RET(
            algResResponse.slaveDevStreams.empty(),
            HCCL_ERROR(
                "[AllocAlgResource]tag[%s] get slave device stream failed, "
                "expect to get size [%u], but only alloc 0.",
                newTag.c_str(), LOCAL_STREAM_MAX_NUM),
            HCCL_E_INTERNAL);
        CHK_RET(AllocAlgNotifys(
            opParam.tag, NotifyLoadType::DEVICE_NOTIFY, LOCAL_NOTIFY_MAX_NUM, algResResponse.notifiesDevMain,
            algResResponse.notifiesDevAux));
    }
    uint8_t devNotifyNum = algResResponse.notifiesDevMain.size() + algResResponse.notifiesDevAux.size();
    HCCL_INFO(
        "[AllocAlgResource] tag[%s] alloc host slaveStreamNum[%u],"
        "device slaveStreamNum[%u], devNotifyNum[%u], hostNotifyNum[%u]",
        newTag.c_str(), algResResponse.slaveStreams.size(), algResResponse.slaveDevStreams.size(), devNotifyNum,
        resRequest.notifyNum);
    CHK_RET(AllocAlgNotifys(
        opParam.tag, NotifyLoadType::HOST_NOTIFY, resRequest.notifyNum, algResResponse.notifiesMain,
        algResResponse.notifiesAux));

    algResResponse.cclInputMem = cclBufferManager_.GetInCCLbuffer();
    algResResponse.cclOutputMem = cclBufferManager_.GetOutCCLbuffer();
    DeviceMem expMem = cclBufferManager_.GetCommCCLBuffer(); // 获取拓展内存
    if (opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALLV || opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC
        || opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        DeviceMem tinySendRecvMem;
        CHK_RET(implAlg_->GetTinyMem(tinySendRecvMem));
        CHK_RET(CalcTinySendRecvMem(opParam, algResResponse, tinySendRecvMem));
    } else {
        algResResponse.paramInputMem = DeviceMem::create(opParam.inputPtr, opParam.inputSize);
        algResResponse.paramOutputMem = DeviceMem::create(opParam.outputPtr, opParam.outputSize);
    }

    bool useOpbaseFlag = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !opParam.isCapture);
    if ((AIV_COMM_BUFFER_BITMASK & resRequest.aivBufferRequest) != 0) {
        ret = cclBufferManager_.CreateCommAIVbuffer(useOpbaseFlag);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Alloc][AlgResource]Create CommAIVbuffer failed"), ret);
        if (useOpbaseFlag) { // 单算子非Capture模式，对应aivOpbaseTag_
            algResResponse.aivInputMem = cclBufferManager_.GetInAivOpbaseBuffer();
            algResResponse.aivOutputMem = cclBufferManager_.GetOutAivOpbaseBuffer();
        } else { // 静态图或者Capture模式，对应aivOffloadTag_
            algResResponse.aivInputMem = cclBufferManager_.GetInAivOffloadbuffer();
            algResResponse.aivOutputMem = cclBufferManager_.GetOutAivOffloadbuffer();
        }
        HCCL_INFO("[AllocAlgResource] tag[%s] alloc aiv buffer", newTag.c_str());
    }
    if ((AIV_COMM_INFO_BUFFER_BITMASK & resRequest.aivBufferRequest) != 0 || opParam.isNpuDirectRoce) {
        if (!useOpbaseFlag) {
            DeviceMem aivCommInfoMem; // 图模式每个算子单独一块内存
            CHK_RET(DeviceMem::alloc(aivCommInfoMem, AIV_COMM_INFO_SIZE));
            algResResponse.aivCommInfoMem = aivCommInfoMem;
            aivOffloadCommInfoMem_.emplace_back(std::move(aivCommInfoMem));
        } else {
            ret = cclBufferManager_.CreateCommInfoAIVbuffer();
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Alloc][AlgResource]Create CommInfoAIVbuffer failed"), ret);
            algResResponse.aivCommInfoMem = cclBufferManager_.GetAivCommInfoBuffer(); // 单算子每个通信域只用一块内存
        }
        HCCL_INFO("[AllocAlgResource] tag[%s] alloc aiv comm info buffer", newTag.c_str());
    }

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
    HCCL_DEBUG(
        "algResResponse.cclInputMem[%p], size[%llu]; algResResponse.cclOutputMem[%p], "
        "size[%llu]; algResResponse.paramInputMem[%p], size[%llu]; algResResponse.paramOutputMem[%p], size[%llu].",
        algResResponse.cclInputMem.ptr(), algResResponse.cclInputMem.size(), algResResponse.cclOutputMem.ptr(),
        algResResponse.cclOutputMem.size(), algResResponse.paramInputMem.ptr(), algResResponse.paramInputMem.size(),
        algResResponse.paramOutputMem.ptr(), algResResponse.paramOutputMem.size());
    algResResponse.opTransportResponse = resRequest.opTransport;

    // 零拷贝场景这里只借助P2p的openIpc能力交换控制面zeroCopyLocalBuffer_，不交换实际用户的输出输出
    if (opParam.isZeroCopy) {
        HCCL_INFO(
            "[AllocAlgResource] zero copy change paramInput[%p] paramOutput[%p] scratchMem[%p] to localBuffer[%p]",
            transMem.paramInputMem.ptr(), transMem.paramOutputMem.ptr(), transMem.scratchMem.ptr(),
            zeroCopyLocalBuffer_.ptr());
        transMem.scratchMem = zeroCopyLocalBuffer_;
        transMem.paramInputMem = zeroCopyLocalBuffer_;
        transMem.paramOutputMem = zeroCopyLocalBuffer_;
    } else {
        if (isGraphZeroCopyAlgAlloc) {
            transMem.scratchMem
                = DeviceMem::create(cclBufferManager_.GetOutCCLbuffer().ptr(), resRequest.scratchMemSize);
            HCCL_INFO("[AllocAlgResource] acl graph set transMem.scratchMem =%ul", transMem.scratchMem.size());
        }
    }

    ClearOpTransportResponseLinks(algResResponse.opTransportResponse);
    if (IsEnableBackupLink()) {
        algResResponse.opTransportResponseBackUp = resRequest.opTransport;
        ClearOpTransportResponseLinks(algResResponse.opTransportResponseBackUp);
        HCCL_DEBUG(
            "[%s]IsEnableBackupLink[%d] init backup & default opTransportResponse", __func__, IsEnableBackupLink());
    }

    if (!GetExternalInputHcclEnableFfts() && GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        u32 slaveNum = algResResponse.slaveStreams.size();
        algResResponse.threadManage.resize(slaveNum);
        for (u32 ringIndex = 0; ringIndex < slaveNum; ringIndex++) {
            algResResponse.threadManage[ringIndex].reset(new (std::nothrow)
                                                             ThreadManage(deviceLogicId_, userRank_, dispatcher_));
            CHK_SMART_PTR_NULL(algResResponse.threadManage[ringIndex]);
            HcclResult ret = algResResponse.threadManage[ringIndex]->Init();
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR("[Init][MultiRingResource]ringIndex[%u] ThreadManage failed,return[%d]", ringIndex, ret),
                ret);
            HCCL_INFO("ringThreadsManage Init success[%u]", ringIndex);
        }
    }
    transportManager_->SetOpType(opParam.opType);
    if (isUserMemRegisted_) {
        // user win模式，用exchange接口建链的transport
        algResResponse.opTransportResponse = userMemTransport_;
        CHK_RET(GetRemoteUserMemResource());
    } else {
        StateGuard<HcclCommunicator, HcclCommState> guard(this, HcclCommState::BUILDING);
        ret = transportManager_->Alloc(
            opParam.tag, transMem, algResResponse.opTransportResponse, opParam.aicpuUnfoldMode, false,
            opParam.isZeroCopy, opParam.opType, opParam.isCapture, false, opParam.isNpuDirectRoce, &opParam);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s]Alloc transports failed, tag[%s]", __func__, newTag.c_str()), ret);
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[%s]Alloc transports failed, tag[%s]", __func__, newTag.c_str()), ret);

    if (retryEnable_) {
        // 获取当前rdma相连的所有对端rankList
        std::vector<u32> rankList;
        CHK_RET(transportManager_->GetRemoteRankList(
            algResResponse.opTransportResponse, rankList, TransportType::TRANS_TYPE_IBV_EXP));
        std::string rankListStr = "";
        for (auto remoteRank : rankList) {
            rankListStr += (std::to_string(remoteRank) + ";");
        }
        HCCL_DEBUG("identifier[%s] newTag[%s] rankList[%s]", identifier_.c_str(), newTag.c_str(), rankListStr.c_str());
        CHK_RET(OpRetryManager::AddLinkInfoByIdentifier(deviceLogicId_, identifier_, newTag, rankList));
    }

    if (IsEnableBackupLink()) {
        // 超节点 && level2支持重执行 && Aicpu：创建备用Transport资源
        StateGuard<HcclCommunicator, HcclCommState> guard(this, HcclCommState::BUILDING);
        ret = transportManager_->Alloc(
            opParam.tag, transMem, algResResponse.opTransportResponseBackUp, opParam.aicpuUnfoldMode, true,
            opParam.isCapture);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[%s]Alloc backup transports failed, tag[%s]", __func__, newTag.c_str()),
            ret);
    }
    SaveLinkRes(algResResponse.opTransportResponse);
    SaveLinkRes(algResResponse.opTransportResponseBackUp);
    remoteTransportMap_ = transportManager_->GetRemoteTransportMap();
    HCCL_DEBUG("[%s] process success newtag[%s]", __func__, newTag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetRemoteUserMemResource()
{
    for (auto& levelNSubCommTransport : userMemTransport_) {
        for (auto& singleSubCommTransport : levelNSubCommTransport) {
            u32 linkIdx = 0;
            for (auto& transportRequest : singleSubCommTransport.transportRequests) {
                if (!transportRequest.isValid) {
                    continue;
                }
                auto tempLink = singleSubCommTransport.links[linkIdx];
                MemDetails remoteMem;
                u32 remoteId = tempLink->GetRemoteRank();
                CHK_PRT_RET(
                    (remoteId >= MAX_RANK_NUM_A3),
                    HCCL_ERROR(
                        "[%s]Invalid remoteId, valid range is [0, %u), remoteId[%u]", __func__, MAX_RANK_NUM_A3,
                        remoteId),
                    HCCL_E_PARA);
                void* userMemPtr = nullptr;
                CHK_RET(tempLink->GetRemoteMem(UserMemType::INPUT_MEM, &userMemPtr));
                CHK_PTR_NULL(userMemPtr);
                remoteMem.addr = reinterpret_cast<u64>(userMemPtr);
                CHK_RET(tempLink->GetRemoteMemSize(UserMemType::INPUT_MEM, remoteMem.size));
                opResPara_.userMemRes[remoteId] = remoteMem;
                HCCL_INFO(
                    "[%s]add userMem res success, remoteId[%u], "
                    "remote addr[%llu], linkIdx[%u]",
                    __func__, remoteId, remoteMem.addr, linkIdx);
                linkIdx++;
            }
        }
    }
    opResPara_.userMemType = TYPE_USER_MEM;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::IncreAllocLink(
    const std::string& newTag, const OpParam& opParam, AlgResourceRequest& resRequest,
    AlgResourceResponse& algResResponse)
{
    algResResponse.cclInputMem = cclBufferManager_.GetInCCLbuffer();
    algResResponse.cclOutputMem = cclBufferManager_.GetOutCCLbuffer();
    DeviceMem expMem = cclBufferManager_.GetCommCCLBuffer();
    transportManager_->SetOpType(opParam.opType);

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
        StateGuard<HcclCommunicator, HcclCommState> guard(this, HcclCommState::BUILDING);
        CHK_RET(transportManager_->IncreAlloc(
            opParam.tag, transMem, resRequest.opTransport, algResResponse.opTransportResponse, opParam.aicpuUnfoldMode,
            false, opParam.isCapture, opParam.opType));
    }
    if (retryEnable_) {
        // 获取当前rdma相连的所有对端rankList
        std::vector<u32> rankList;
        CHK_RET(transportManager_->GetIncreRemoteRankList(
            resRequest.opTransport, rankList, TransportType::TRANS_TYPE_IBV_EXP));
        std::string rankListStr = "";
        for (auto remoteRank : rankList) {
            rankListStr += (std::to_string(remoteRank) + ";");
        }
        HCCL_DEBUG("identifier[%s] newTag[%s] rankList[%s]", identifier_.c_str(), newTag.c_str(), rankListStr.c_str());
        CHK_RET(OpRetryManager::AddLinkInfoByIdentifier(deviceLogicId_, identifier_, newTag, rankList, true));
    }
    if (IsEnableBackupLink()) {
        StateGuard<HcclCommunicator, HcclCommState> guard(this, HcclCommState::BUILDING);
        CHK_RET(transportManager_->IncreAlloc(
            opParam.tag, transMem, resRequest.opTransport, algResResponse.opTransportResponseBackUp,
            opParam.aicpuUnfoldMode, true, opParam.isCapture, opParam.opType));
    }
    remoteTransportMap_ = transportManager_->GetRemoteTransportMap();
    SaveLinkRes(algResResponse.opTransportResponse);
    SaveLinkRes(algResResponse.opTransportResponseBackUp);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetDevicePid(s32 devicePid)
{
    devicePid_ = devicePid;
    return HCCL_SUCCESS;
}

void HcclCommunicator::ReleaseWorkSpacebuffer() { workSpace_.free(); }

HcclResult HcclCommunicator::AllocAndClearDeviceMem(u64 size, std::shared_ptr<DeviceMem>& bufferPtr) const
{
    CHK_PRT_RET(
        !size,
        HCCL_INFO("[HcclCommunicator][AllocAndClearDeviceMem]device memory size is zero. not need to malloc memory"),
        HCCL_SUCCESS);

    CHK_PRT_RET(
        (size > ULONG_MAX),
        HCCL_ERROR("[HcclCommunicator][AllocAndClearDeviceMem]device memory size is greater than %llu", ULONG_MAX),
        HCCL_E_PARA);

    DeviceMem tmpBuffer;
    CHK_RET(DeviceMem::alloc(tmpBuffer, size));
    EXCEPTION_CATCH((bufferPtr = std::make_shared<DeviceMem>(std::move(tmpBuffer))), return HCCL_E_PTR);

    CHK_PRT_RET(
        size && !bufferPtr.get()->ptr(),
        HCCL_ERROR(
            "[HcclCommunicator][AllocAndClearDeviceMem]Create DeviceMem size[%llu] fail,"
            "please check workspace size.",
            size),
        HCCL_E_PTR);
    CHK_RET(hrtMemSet(bufferPtr.get()->ptr(), size, size));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllocAndClearHostMem(u64 size, std::shared_ptr<HostMem>& bufferPtr) const
{
    CHK_PRT_RET(
        !size,
        HCCL_INFO("[HcclCommunicator][AllocAndClearHostMem] host memory size is zero. not need to malloc memory"),
        HCCL_SUCCESS);

    CHK_PRT_RET(
        (size > ULONG_MAX),
        HCCL_ERROR("[HcclCommunicator][AllocAndClearHostMem] host memory size is greater than %llu", ULONG_MAX),
        HCCL_E_PARA);

    HostMem tmpBuffer = HostMem::alloc(size);
    EXCEPTION_CATCH((bufferPtr = std::make_shared<HostMem>(std::move(tmpBuffer))), return HCCL_E_PTR);

    CHK_PRT_RET(
        size && !bufferPtr.get()->ptr(),
        HCCL_ERROR(
            "[HcclCommunicator][AllocAndClearHostMem]host memory space size[%llu] fail,"
            "please check workspace size.",
            size),
        HCCL_E_PTR);
    CHK_SAFETY_FUNC_RET(memset_s(bufferPtr.get()->ptr(), size, 0, size));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CreateWorkSpace(u64 size, DeviceMem& buffer) const
{
    CHK_PRT_RET(
        !size, HCCL_INFO("[Create][WorkSpace]work space size is zero. not need to malloc memory"), HCCL_SUCCESS);

    CHK_PRT_RET(
        (size > ULONG_MAX), HCCL_ERROR("[Create][WorkSpace]work space size is greater than %llu", ULONG_MAX),
        HCCL_E_PARA);

    u64 memSize = size;
    CHK_RET(DeviceMem::alloc(buffer, memSize));
    CHK_RET(hrtMemSet(buffer.ptr(), size, size));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetWorkSpace(u64* workSpaceSize, u64* workSpace) const
{
    *workSpaceSize = workSpaceSize_;
    *workSpace = reinterpret_cast<u64>(workSpace_.ptr());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitWorkSpace()
{
    if (workSpace_.ptr() == nullptr) {
        workSpaceSize_ = COMM_MAX_WORK_SPACE_SIZE;
        CHK_RET(CreateWorkSpace(workSpaceSize_, workSpace_));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::FillOpParam(
    const HcclCMDType commType, OpParam& opParam, const uint64_t count, void* pCount, void* pDispls) const
{
    if (commType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER || commType == HcclCMDType::HCCL_CMD_ALLGATHER
        || commType == HcclCMDType::HCCL_CMD_ALLREDUCE) {
        opParam.DataDes.count = count;
        opParam.DataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP16; // 按照fp16配置
    } else if (
        commType == HcclCMDType::HCCL_CMD_ALLTOALLV || commType == HcclCMDType::HCCL_CMD_ALLTOALL
        || commType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        opParam.All2AllDataDes.sendType = HcclDataType::HCCL_DATA_TYPE_FP16;
        opParam.All2AllDataDes.recvType = HcclDataType::HCCL_DATA_TYPE_FP16;
        opParam.All2AllDataDes.sendCounts = pCount;
        opParam.All2AllDataDes.recvCounts = pCount;
        opParam.All2AllDataDes.sdispls = pDispls;
        opParam.All2AllDataDes.rdispls = pDispls;
        opParam.All2AllDataDes.sendCountMatrix = pCount;
    } else if (commType == HcclCMDType::HCCL_CMD_BATCH_WRITE) {
    } else {
        HCCL_ERROR("[%s] invalid commType=[%u]", __func__, static_cast<uint32_t>(commType));
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllocComResource(
    const string& newTag, const string& algName, const HcclCMDType commType, const OpParam& opParam, rtStream_t stream,
    bool isNeedHostSlaveStream)
{
    if (resMap_.find(newTag) == resMap_.end()) { // 计算&申请通信资源
        unique_ptr<CollAlgOperator> algOperator = implAlg_->GetAlgOperator(commType);
        CHK_PRT_RET(algOperator == nullptr, HCCL_ERROR("[%s] algOperator is nullptr", __func__), HCCL_E_INTERNAL);
        AlgResourceRequest resRequest;
        CHK_RET(algOperator->CalcResRequest(algName, opParam, resRequest));
        AlgResourceResponse algResResponse;
        CHK_RET(AllocAlgResource(newTag, commType, opParam, resRequest, algResResponse, isNeedHostSlaveStream));
        resMap_[newTag] = algResResponse;
        CHK_RET(RegisterToHeartBeat());
    }

    CHK_RET(InitWorkSpace());
    HcclResult ret = GetWorkSpace(&(opResPara_.mc2WorkSpace.workSpaceSize), &(opResPara_.mc2WorkSpace.workSpace));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "%s GetWorkSpace fail, size[%llu] space[%llu]", __func__, opResPara_.mc2WorkSpace.workSpaceSize,
            opResPara_.mc2WorkSpace.workSpace),
        ret);

    if (!isContextLaunched_) { // 通信域内首次下发
        uint64_t streamMode = 0;
        CHK_RET(hrtStreamGetMode(opParam.stream.ptr(), &streamMode));
        rtStream_t aicpuStream;
        Mc2AiCpuStreamAllocAndGet(streamMode, aicpuStream); // aicpuStream需要在首次下发时申请

        rtStream_t aicpuInitStream;
        Mc2AiCpuInitStreamAllocAndGet(streamMode, aicpuInitStream);
        Stream tmpStream(aicpuInitStream);
        HCCL_DEBUG("%s ContextLaunched, aicpuInitStream:%p, aicpuStream:%p", __func__, aicpuInitStream, aicpuStream);
        CHK_RET(AicpuResourceInit(algName, resMap_[newTag], newTag, stream, commType));
        CHK_RET(GetReportHcclMC2Info(tmpStream, resMap_[newTag].slaveDevStreams));
    } else if (newTagResAlloced_.find(newTag) == newTagResAlloced_.end()) {
        // 通信域内非首次，但是有新的newTag
        PetersonLockGuard guard(hostDeviceLock_.get());
        CHK_PRT_RET(guard.IsLockFailed(), HCCL_ERROR("[%s] hostDeviceLock lock failed", __func__), HCCL_E_INTERNAL);
        CHK_RET(AicpuResourceRefresh(resMap_[newTag], newTag, commType));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AllocComResourceByTiling(const string& algConfig, void* param)
{
    CHK_PTR_NULL(combinOparaMem_);
    HcclCombinOpParam* combinOparaPtr = reinterpret_cast<HcclCombinOpParam*>(combinOparaMem_->ptr());
    CHK_PTR_NULL(combinOparaPtr);

    string algName, newTag;
    OpParam& opParam = *static_cast<OpParam*>(param);
    CHK_RET(GetAlgInfo(algConfig, opParam.tag, opParam.opType, algName, newTag));
    if (algName == "RunAlltoAllAivDirect") {
        opParam.isNpuDirectRoce = true;
    }
    CHK_RET(CreateAndGetAiCpuNotifyWithNotifyRes(combinOparaPtr->signalInfo.aicpuNotify));
    HCCL_INFO("Create aicpu notify %p.", localAiCpuNotifyRes_[0]->ptr());

    // 只有第一次创建，此处通过CCL Buffer地址有效来防止通信域内非首次重新申请内存
    // 已注册user mem情况下，不创建ccl buffer，使用user mem通信
    if (userMemMap_.empty()) {
        CHK_RET(CreateCommCCLbuffer());
        CHK_RET(cclBufferManager_.GetInCCLbuffer(opParam.inputPtr, opParam.inputSize));
        CHK_RET(cclBufferManager_.GetOutCCLbuffer(opParam.outputPtr, opParam.outputSize));
    } else {
        auto it = userMemMap_.begin();
        opParam.outputSize = it->second->size();
        opParam.inputSize = it->second->size();
    }

    // 按照 ccl buffer size 折算，不同算子折算方式不同, allreduce和cclbuffer size相同
    // allgather、reducescatter、alltoall需除以rank size
    uint64_t count = opParam.outputSize / SIZE_TABLE[HcclDataType::HCCL_DATA_TYPE_FP16];
    if (opParam.opType != HcclCMDType::HCCL_CMD_ALLREDUCE) {
        count = (count + userRankSize_ - 1) / userRankSize_;
    }
    HCCL_INFO("[%s] userRankSize=[%u], count=[%u]", __func__, userRankSize_, count);
    vector<uint64_t> countList(userRankSize_ * userRankSize_, count);
    vector<uint64_t> displsList(userRankSize_, 0);
    void* pCount = static_cast<void*>(&countList[0]);
    void* pDispls = static_cast<void*>(&displsList[0]);
    CHK_RET(FillOpParam(opParam.opType, opParam, count, pCount, pDispls));
    // MC2算子不需要申请host侧的从流
    bool isNeedHostSlaveStream = false;
    CHK_RET(AllocComResource(newTag, algName, opParam.opType, opParam, opParam.stream.ptr(), isNeedHostSlaveStream));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CreateCommResource(
    const std::string& tag, rtStream_t aiCpuStream, bool isOpbaseMode, void** commContext, const std::string& algConfig)
{
    const std::string& suffix = HCCL_MC2_MULTISERVER_SUFFIX;
    string algName = "";
    string newTag = tag;
    if (tag.size() > suffix.size() && tag.compare(tag.size() - suffix.size(), suffix.size(), suffix) == 0) {
        HCCL_INFO("[HcclCommunicator][CreateCommResource] Set isA2MC2MultiServer_ to [true]");
        isA2MC2MultiServer_ = true;
        char* mmSysGetEnvValue = nullptr;
        MM_SYS_GET_ENV(MM_ENV_HCCL_INTRA_PCIE_ENABLE, mmSysGetEnvValue);
        std::string intraPcieEnableEnv = (mmSysGetEnvValue != nullptr) ? mmSysGetEnvValue : "EmptyString";
        bool envA2MC2Hie = (intraPcieEnableEnv == "1") && (GetExternalInputIntraRoceSwitch() == 0);
        if (!algConfig.empty()) {
            CHK_RET(GetAlgInfo(algConfig, tag, algName));
            if (algName == "DispatchCombineHierarchy" || (algName == "BatchWriteBySdma" && envA2MC2Hie)) {
                isA2MC2IntraHie_ = true;
                newTag.insert(newTag.size() - suffix.size(), "_HIE");
            }
        }
    }
    if (isA2MC2MultiServer_ && !isNeedInitNic_) {
        InitNic(true);
    }

    if ((deviceType_ != DevType::DEV_TYPE_910_93 && moduleNum_ > 1 && !isA2MC2MultiServer_)
        || (deviceType_ == DevType::DEV_TYPE_910_93 && superPodNum_ > 1)) {
        HCCL_ERROR(
            "[HcclCommunicator][CommResource]MC2 is not supported in the current scenario, "
            "device type[%d] moduleNum[%d] serverNum[%d] superPodNum[%d], isMC2MultiServer[%d].",
            deviceType_, moduleNum_, serverNum_, superPodNum_, isA2MC2MultiServer_);
        return HCCL_E_NOT_SUPPORT;
    }

    HCCL_INFO(
        "[HcclCommunicator][CommResource]newTag[%s] aicpu stream[%p] isOpbaseMode[%u]", newTag.c_str(), aiCpuStream,
        isOpbaseMode);

    Stream stream(aiCpuStream);
    CHK_RET(CreateCommAndStreamRes(newTag, stream));

    CHK_RET(Mc2CreateAndLaunchContext(aiCpuStream, isOpbaseMode, commContext, newTag));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Mc2CreateAndLaunchContext(
    rtStream_t aiCpuStream, bool isOpbaseMode, void** commContext, const string& tag)
{
    CHK_PTR_NULL(combinOparaMem_);
    HcclCombinOpParam* combinOparaPtr = reinterpret_cast<HcclCombinOpParam*>(combinOparaMem_->ptr());
    CHK_PTR_NULL(combinOparaPtr);
    CHK_RET(InitWorkSpace());

    HcclResult result
        = GetWorkSpace(&(combinOparaPtr->mc2WorkSpace.workSpaceSize), &(combinOparaPtr->mc2WorkSpace.workSpace));
    CHK_PRT_RET(
        result != HCCL_SUCCESS,
        HCCL_ERROR(
            "[HcclCommunicator][CommResource]errNo[0x%016llx] size[%llu] space[%llu]", HCCL_ERROR_CODE(result),
            combinOparaPtr->mc2WorkSpace.workSpaceSize, combinOparaPtr->mc2WorkSpace.workSpace),
        result);

    CHK_SAFETY_FUNC_RET(memcpy_s(
        combinOparaPtr->hcomId, sizeof(combinOparaPtr->hcomId), identifier_.c_str(), identifier_.length() + 1));

    Stream tmpStream(aiCpuStream);
    CHK_RET(CreateAndGetAiCpuNotifyWithNotifyRes(combinOparaPtr->signalInfo.aicpuNotify));
    CHK_RET(CreateAndGetAiCpuNotify(
        localAiCpuOpNotify_[static_cast<u32>(AicpuLocalNotifyIdx::HOST_TO_AICPU_0)],
        combinOparaPtr->signalInfo.aicpuOpNotify[static_cast<u32>(AicpuLocalNotifyIdx::HOST_TO_AICPU_0)]));
    CHK_RET(CreateAndGetAiCpuNotify(
        localAiCpuOpNotify_[static_cast<u32>(AicpuLocalNotifyIdx::HOST_TO_AICPU_1)],
        combinOparaPtr->signalInfo.aicpuOpNotify[static_cast<u32>(AicpuLocalNotifyIdx::HOST_TO_AICPU_1)]));
    // 申请集合通信域存储context的device空间
    CHK_RET(CreateDeviceCommContext(sizeof(HcclCombinOpParam), commContext_));
    combinOparaPtr->config.deterministic = GetDeterministicConfig();
    // retryEnable 写入aicpu_ctx
    combinOparaPtr->config.retryEnable = static_cast<u8>(retryEnable_);
    combinOparaPtr->config.retryHoldTime = commConfig_.GetConfigRetryHoldTime();
    combinOparaPtr->config.retryIntervalTime = commConfig_.GetConfigRetryIntervalTime();
    combinOparaPtr->config.notifyWaitTime
        = (GetExternalInputHcclExecTimeoutSet() != HcclExecTimeoutSet::HCCL_EXEC_TIMEOUT_NOT_SET
           || commConfig_.GetConfigExecTimeOutSet()) ?
              commConfig_.GetConfigExecTimeOut() :
              NOTIFY_DEFAULT_WAIT_TIME;
    combinOparaPtr->config.linkTimeOut = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());

    combinOparaPtr->kfcControlTransferH2DParams = kfcControlTransferH2D_->GetCommunicateParams();
    combinOparaPtr->kfcStatusTransferD2HParams = kfcStatusTransferD2H_->GetCommunicateParams();

    void* overflowAddr = nullptr;
    if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
        CHK_RET(hrtCtxGetOverflowAddr(&overflowAddr));
        combinOparaPtr->overFlowAddr = reinterpret_cast<u64>(overflowAddr);
        HCCL_INFO(
            "[HcclImplBase][Mc2CreateAndLaunchContext]get combinOparaPtr->overFlowAddr %llx",
            combinOparaPtr->overFlowAddr);
        // 非整卡 (2DUO卡各取1芯的场景) 因为受到PCIE限制，不可以使用读操作进行数据拷贝
        if (pairLinkInfo_[static_cast<u32>(LinkTypeInServer::HCCS_TYPE)].size() != userRankSize_) {
            combinOparaPtr->onlyRead = 1;
        }
    }
    HCCL_INFO("read only is set to %u", combinOparaPtr->onlyRead);

    if (isA2MC2MultiServer_) {
        // 拷贝normal transport信息到device侧
        bool isSupportAIVNormalQP = false;
        CHK_RET(IsSupportAIVNormalQP(devicePhyId_, isSupportAIVNormalQP));
        CHK_PTR_NULL(transDevIbverbsDataMem_);
        const u64 ibverbsDataSize = transDevIbverbsDataMem_->size();
        CHK_RET(DeviceMem::alloc(ibverbsDataBuffer_, ibverbsDataSize));
        CHK_RET(hrtMemAsyncCopy(
            ibverbsDataBuffer_.ptr(), ibverbsDataBuffer_.size(), transDevIbverbsDataMem_->ptr(), ibverbsDataSize,
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE, aiCpuStream));

        combinOparaPtr->ibverbsData = reinterpret_cast<u64>(ibverbsDataBuffer_.ptr());
        combinOparaPtr->ibverbsDataSize = ibverbsDataSize;
        combinOparaPtr->multiServerFlag = static_cast<u8>(true);

        CHK_PTR_NULL(combinedCapabilityMem_);
        const u64 capabilitySize = sizeof(CombinedCapability);
        CHK_RET(DeviceMem::alloc(combinedCapabilityBuffer_, capabilitySize));
        CHK_RET(hrtMemAsyncCopy(
            combinedCapabilityBuffer_.ptr(), combinedCapabilityBuffer_.size(), combinedCapabilityMem_->ptr(),
            capabilitySize, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE, aiCpuStream));

        combinOparaPtr->capabilityPtr = reinterpret_cast<u64>(combinedCapabilityBuffer_.ptr());
        combinOparaPtr->capabilitySize = capabilitySize;

        HCCL_INFO(
            "[HcclImplBase][Mc2CreateAndLaunchContext] set ibverbsData to [%llu], "
            "multiServerFlag to [%u]",
            combinOparaPtr->ibverbsData, combinOparaPtr->multiServerFlag);
        if (isSupportAIVNormalQP && isA2MC2IntraHie_) {
            CHK_RET(H2DAiRMAInfo(tag, aiCpuStream));
        }
    }

    // 将通信数据拷贝到device侧，供AICPU算法编排使用
    CHK_RET(hrtMemAsyncCopy(
        commContext_.ptr(), commContext_.size(), combinOparaMem_->ptr(), combinOparaMem_->size(),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE, aiCpuStream));

    std::string kernelName = "RunAicpuKfcResInit";
    CHK_RET(AiCpuKernelLaunch(tmpStream.ptr(), reinterpret_cast<u64>(commContext_.ptr()), kernelName));
    SetMC2EnvFlag();
    if (isOpbaseMode == true) {
        CHK_RET(hcclStreamSynchronize(tmpStream.ptr(), commConfig_.GetConfigExecTimeOut()));
    }

    *commContext = commContext_.ptr();
    return HCCL_SUCCESS;
}

HcclResult
HcclCommunicator::GetAiCpuNotifyData(const std::shared_ptr<LocalNotify>& localNotify, HcclSignalInfo& notifyInfo) const
{
    if (localNotify == nullptr) {
        HCCL_INFO("[HcclCommunicator][GetAiCpuNotifyData]notifyHandle is null");
        notifyInfo.resId = INVALID_U64;
        return HCCL_SUCCESS;
    }

    CHK_RET(localNotify->GetNotifyData(notifyInfo));
    HCCL_INFO(
        "[HcclCommunicator][GetAiCpuNotifyData]resId[%lld], addr[%lld], devId[%u], tsId[%u].", notifyInfo.resId,
        notifyInfo.addr, notifyInfo.devId, notifyInfo.tsId);
    return HCCL_SUCCESS;
}

HcclResult
HcclCommunicator::CreateAndGetAiCpuNotify(std::shared_ptr<LocalNotify>& localNotify, HcclSignalInfo& notifyInfo)
{
    if (localNotify != nullptr) {
        CHK_RET(GetAiCpuNotifyData(localNotify, notifyInfo));
        HCCL_INFO("[HcclCommunicator][CreateAndGetAiCpuNotify]aicpu notify already create ptr[%p]", localNotify->ptr());
        return HCCL_SUCCESS;
    }

    EXCEPTION_CATCH((localNotify = std::make_shared<LocalNotify>()), return HCCL_E_PTR);
    CHK_RET(localNotify->Init(NotifyLoadType::DEVICE_NOTIFY));
    CHK_RET(localNotify->SetIpc());

    CHK_RET(GetAiCpuNotifyData(localNotify, notifyInfo));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Mc2AiCpuStreamAllocAndGet(u32 streamMode, rtStream_t& aiCpuStream)
{
    if (opStream_.ptr() != nullptr) {
        HCCL_INFO("%s already alloc, group:%s, stream id:%u", __func__, identifier_.c_str(), opStream_.id());
        aiCpuStream = opStream_.ptr();
        return HCCL_SUCCESS;
    }

    constexpr u32 aicpuStreamMode = 1; // 单独申请的kernel流，使能遇错即停，避免出错后流卡住不退
    opStream_ = Stream(StreamType::STREAM_TYPE_ONLINE);
    CHK_RET(hrtStreamSetMode(opStream_.ptr(), aicpuStreamMode));
    aiCpuStream = opStream_.ptr();
    HCCL_RUN_INFO(
        "%s alloc success, group:%s, stream id:%u, mainStreamMode:%u, aicpuStreamMode:%u", __func__,
        identifier_.c_str(), opStream_.id(), streamMode, aicpuStreamMode);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::Mc2AiCpuInitStreamAllocAndGet(u32 streamMode, rtStream_t& aiCpuStream)
{
    if (aicpuInitStream_.ptr() != nullptr) {
        HCCL_INFO("%s already alloc, group:%s, stream id:%u", __func__, identifier_.c_str(), aicpuInitStream_.id());
        aiCpuStream = aicpuInitStream_.ptr();
        return HCCL_SUCCESS;
    }

    constexpr u32 aicpuStreamMode = 1; // 单独申请的kernel流，使能遇错即停，避免出错后流卡住不退
    aicpuInitStream_ = Stream(StreamType::STREAM_TYPE_ONLINE);
    CHK_RET(hrtStreamSetMode(aicpuInitStream_.ptr(), aicpuStreamMode));
    aiCpuStream = aicpuInitStream_.ptr();
    HCCL_RUN_INFO(
        "%s alloc success, group:%s, stream id:%u, mainStreamMode:%u, aicpuStreamMode:%u", __func__,
        identifier_.c_str(), aicpuInitStream_.id(), streamMode, aicpuStreamMode);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AicpuResourceInit(
    const std::string& algName, const AlgResourceResponse& algResource, const std::string& newTag,
    const rtStream_t& aicpuStream, const HcclCMDType opType, [[maybe_unused]] bool isCustom)
{
    HCCL_RUN_INFO(
        "[%s] start to init group[%s] aicpu resources newTag[%s] local rankId[%u]", __func__, identifier_.c_str(),
        newTag.c_str(), userRank_);
    isContextLaunched_ = true;
    CHK_RET(BuildOpResParam(algName, algResource, newTag, opType, aicpuStream)); // 构建context结构体
    std::string kernelName = "RunAicpuKfcResInitV2";
    // 在这里构建suspending状态码的HDC通道初始化，并且在host侧进行init
    // （这个主要是针对hcomId；对算子通信域的复用；也就是多个算子复用（tag+Identifier）这个通信域的情况）
    CHK_RET(AiCpuKernelLaunch(aicpuStream, reinterpret_cast<u64>(opResDevicePara_.ptr()), kernelName));
    SetMC2EnvFlag();
    newTagResAlloced_.insert(newTag);
    // 图模多档位场景，需要保证执行序上优先下资源初始化的kernel
    CHK_RET(hcclStreamSynchronize(aicpuStream, commConfig_.GetConfigExecTimeOut()));

    if (IsEnableCustom()) {
        struct InitTask {
            u64 context; // A矩阵地址，通信在前时为sendbuffer
            bool isCustom;
        };
        InitTask customInitTask = {};
        customInitTask.context = reinterpret_cast<u64>(opResDevicePara_.ptr());
        customInitTask.isCustom = true;
        CHK_RET(BuildCustomOpResParam());
        uint64_t customBeginTime = hrtMsprofSysCycleTime();
        const std::string customProfName = "hcomAicpuCustomInit";

        u16 timeOut = 0;
        if (opResPara_.config.notifyWaitTime == 0) {
            timeOut = opResPara_.config.notifyWaitTime;
        } else if (opResPara_.config.notifyWaitTime + AICPU_KERNEL_TIMEOUT_INC >= MAX_VALUE_U16) {
            timeOut = MAX_VALUE_U16;
        } else {
            timeOut = opResPara_.config.notifyWaitTime + AICPU_KERNEL_TIMEOUT_INC;
        }

        CHK_RET(AicpuAclKernelLaunch(
            aicpuStream, static_cast<void*>(&customInitTask), sizeof(customInitTask), binCustomHandle_, kernelName,
            true, timeOut));
        uint64_t customEndTime = hrtMsprofSysCycleTime();
        s32 customthreadId = SalGetTid();
        CHK_RET(ProfilingManagerPub::CallMsprofReportNodeInfo(
            customBeginTime, customEndTime, customProfName, customthreadId));
        CHK_RET(hcclStreamSynchronize(aicpuStream, commConfig_.GetConfigExecTimeOut()));
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AiCpuKernelLaunch(const rtStream_t stm, u64 addr, const std::string& kernelName)
{
    uint64_t beginTime = hrtMsprofSysCycleTime();
    const std::string profName = "hcomAicpuInit";
    struct InitTask {
        u64 context; // A矩阵地址，通信在前时为sendbuffer
        bool isCustom;
    };
    InitTask initTask = {};
    initTask.context = addr;
    initTask.isCustom = false;

    u16 timeOut = 0;
    if (opResPara_.config.notifyWaitTime == 0) {
        timeOut = opResPara_.config.notifyWaitTime;
    } else if (opResPara_.config.notifyWaitTime + AICPU_KERNEL_TIMEOUT_INC >= MAX_VALUE_U16) {
        timeOut = MAX_VALUE_U16;
    } else {
        timeOut = opResPara_.config.notifyWaitTime + AICPU_KERNEL_TIMEOUT_INC;
    }
    CHK_RET(AicpuAclKernelLaunch(
        stm, static_cast<void*>(&initTask), sizeof(initTask), binHandle_, kernelName, true, timeOut));
    uint64_t endTime = hrtMsprofSysCycleTime();
    s32 threadId = SalGetTid();
    CHK_RET(ProfilingManagerPub::CallMsprofReportNodeInfo(beginTime, endTime, profName, threadId));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AicpuKfcTilingDataLaunch(
    const OpParam& opParam, const HcclCMDType& opType, const DeviceMem& deviceContext, const std::string& kernelName,
    const AicpuOpTiling opTilingInfo)
{
    HCCL_DEBUG(
        "AicpuKfcTilingDataLaunch count %llu dataType %s op %s opType %u", opParam.GetDataCount(userRank_),
        GetDataTypeEnumStr(opParam.GetDataType()).c_str(), GetReduceOpEnumStr(opParam.reduceType).c_str(), opType);
    struct HcclKFCTilingData tilingDate = {};
    tilingDate.sendCnt = opParam.DataDes.count;
    tilingDate.dataType = opParam.DataDes.dataType;
    tilingDate.commType = static_cast<uint8_t>(opType);
    tilingDate.reduceOp = opParam.reduceType;
    tilingDate.taskType = HCCL_KFC_TASK_HCCL_ONLY_EXE;
    tilingDate.totalCnt = 1;
    tilingDate.turnNum = 1;
    tilingDate.hasCommOut = 1;
    tilingDate.debugMode = 0;
    CHK_RET(SetNormalMode(dispatcher_));
    HcclWorkflowMode mode = GetWorkflowMode();
    Stream mainStream(opParam.stream.ptr());
    CHK_RET(LocalNotify::Post(
        mainStream, dispatcher_, localAiCpuOpNotify_[static_cast<u32>(AicpuLocalNotifyIdx::HOST_TO_AICPU_0)],
        INVALID_VALUE_STAGE));
    rtStream_t kfcOpStream = opStream_.ptr();
    if (opTilingInfo.isUsedMainStream) {
        kfcOpStream = opParam.stream.ptr();
    }
    CHK_RET(AicpuUnfoldKernelLaunch(
        opParam.inputPtr, opParam.outputPtr, kfcOpStream, reinterpret_cast<u64>(deviceContext.ptr()), &tilingDate,
        sizeof(HcclKFCTilingData), kernelName, mode, opParam.tag));
    CHK_RET(LocalNotify::Wait(
        mainStream, dispatcher_, localAiCpuOpNotify_[static_cast<u32>(AicpuLocalNotifyIdx::HOST_TO_AICPU_1)],
        INVALID_VALUE_STAGE));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AicpuKfcClearOpResLaunch(const std::unordered_set<std::string>& tags)
{
    if (tags.empty()) {
        return HCCL_SUCCESS;
    }
    // 仅 aicpu unfold 模式有 aicpu 端 resMap_/linkRes_ 需要清理；host 模式下没有 binHandle_
    if (binHandle_ == nullptr) {
        HCCL_DEBUG(
            "[AicpuKfcClearOpResLaunch] binHandle_ null (host-mode communicator), skip; tagCount[%zu]", tags.size());
        return HCCL_SUCCESS;
    }
    if (opStream_.ptr() == nullptr) {
        HCCL_WARNING("[AicpuKfcClearOpResLaunch] opStream_ null, skip aicpu cleanup; tagCount[%zu]", tags.size());
        return HCCL_SUCCESS;
    }
    // host args 通道有 size 上限，大 payload 走 args/tiling 会被拒绝。沿用 RunAicpuKfcResInit 模式：HBM buffer 持载
    // payload
    if (!aicpuCleanupBuf_) {
        CHK_RET(DeviceMem::alloc(aicpuCleanupBuf_, sizeof(HcclKfcClearOpResTilingData)));
    }
    if (!aicpuCleanupHostBuf_) {
        aicpuCleanupHostBuf_.reset(new (std::nothrow) HcclKfcClearOpResTilingData());
        CHK_SMART_PTR_NULL(aicpuCleanupHostBuf_);
    }
    HcclKfcClearOpResTilingData& payload = *aicpuCleanupHostBuf_;

    // 必须与 aicpu_kfc_def.h 中 KFCResInitTask 布局一致，aicpu 端按此解包
    struct KFCResInitTask {
        u64 context;
        bool isCustom;
    };
    KFCResInitTask initTask = {reinterpret_cast<u64>(aicpuCleanupBuf_.ptr()), false};
    const u16 timeOut = MAX_VALUE_U16;
    const size_t groupCopyLen = std::min(identifier_.length() + 1, sizeof(payload.group));
    size_t totalBatches = 0;

    // 分批 launch：同 buffer 复用，每批最多 MAX_BATCH 个 tag；launch 后 sync 保证 aicpu 完成才覆盖 buffer 下一批
    auto it = tags.begin();
    while (it != tags.end()) {
        payload.magic = HCCL_KFC_CLEAR_OP_RES_MAGIC;
        CHK_SAFETY_FUNC_RET(memcpy_s(payload.group, sizeof(payload.group), identifier_.c_str(), groupCopyLen));
        payload.group[sizeof(payload.group) - 1] = '\0';

        u32 idx = 0;
        while (it != tags.end() && idx < HCCL_KFC_CLEAR_OP_RES_MAX_BATCH) {
            const std::string& t = *it;
            const size_t tagCopyLen = std::min(t.length() + 1, sizeof(payload.tags[idx]));
            CHK_SAFETY_FUNC_RET(memcpy_s(payload.tags[idx], sizeof(payload.tags[idx]), t.c_str(), tagCopyLen));
            payload.tags[idx][sizeof(payload.tags[idx]) - 1] = '\0';
            ++idx;
            ++it;
        }
        payload.tagCount = idx;

        CHK_RET(hrtMemSyncCopy(
            aicpuCleanupBuf_.ptr(), sizeof(payload), static_cast<void*>(&payload), sizeof(payload),
            HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

        HcclResult ret = AicpuAclKernelLaunchV2(
            opStream_.ptr(), static_cast<void*>(&initTask), sizeof(initTask), binHandle_, "RunAicpuKfcClearOpRes", true,
            timeOut, nullptr, 0, identifier_);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[AicpuKfcClearOpResLaunch] launch fail, group[%s] batch[%zu] tagCount[%u] ret[%d]",
                identifier_.c_str(), totalBatches, idx, ret);
            return ret;
        }
        CHK_RET(hcclStreamSynchronize(opStream_.ptr(), commConfig_.GetConfigExecTimeOut()));
        ++totalBatches;
    }

    HCCL_INFO(
        "[AicpuKfcClearOpResLaunch] dispatched aicpu cleanup, group[%s] totalTags[%zu] batches[%zu]",
        identifier_.c_str(), tags.size(), totalBatches);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AicpuInitOpTilingDataAicpuCache(
    const OpParam& opParam, const HcclCMDType& opType, struct OpTilingData* opTilingData)
{
    opTilingData->aicpuCacheEnable = opParam.aicpuCacheEnable;
    // 开启aicpu cache, 且原来是图模式建链但强制走单算子模式展开
    // 开启aicpu cache，isCapture为true，且是图模式，证明选择了aclgraph零拷贝算法，需要强制刷新cache
    if (opParam.aicpuCacheEnable != 0 && GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB
        && ((IsForceAicpuOpBaseMode(opParam, opType) && !opParam.isZeroCopy) || opParam.isCapture)) {
        // 环境变量传入的aicpuCacheEnable一定 < 10
        constexpr uint8_t FORCE_OP_BASE_DELTA = 10;
        CHK_PRT_RET(
            opParam.aicpuCacheEnable >= FORCE_OP_BASE_DELTA,
            HCCL_ERROR(
                "[HcclCommunicator][AicpuInitOpTilingDataBuf] enforce opbase mode: opParam.aicpuCacheEnable >= %u",
                opParam.aicpuCacheEnable, FORCE_OP_BASE_DELTA),
            HCCL_E_INTERNAL);

        // 1 -> 11: 开启aicpu cache且存在强制单算子模式转换
        opTilingData->aicpuCacheEnable += FORCE_OP_BASE_DELTA;
        HCCL_WARNING(
            "[HcclCommunicator][AicpuInitOpTilingDataBuf] enforce opbase mode: opParam.aicpuCacheEnable[%u]"
            "opTilingData->aicpuCacheEnable[%u]",
            opParam.aicpuCacheEnable, opTilingData->aicpuCacheEnable);

        // 注意: 开启aicpu cache且存在强制单算子模式转换, 传入device的aicpuCacheEnable一定 > 10
        CHK_PRT_RET(
            opTilingData->aicpuCacheEnable <= FORCE_OP_BASE_DELTA,
            HCCL_ERROR(
                "[HcclCommunicator][AicpuInitOpTilingDataBuf] enforce opbase mode: opTilingData->aicpuCacheEnable[%u] "
                "<= %u",
                opTilingData->aicpuCacheEnable, FORCE_OP_BASE_DELTA),
            HCCL_E_INTERNAL);
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AicpuInitOpTilingDataBuf(
    const OpParam& opParam, const HcclCMDType& opType, [[maybe_unused]] const std::string& kernelName,
    const AicpuOpTiling opTilingInfo, u64 dynamicDataSize)
{
    u32 opTilingDataSize = sizeof(struct OpTilingData) + dynamicDataSize;

    if (opTilingDataBuf_.ptr() == nullptr) {
        opTilingDataBuf_ = HostMem::alloc(TILINGDATA_BUF_SIZE);
        CHK_PRT_RET(
            opTilingDataBuf_.ptr() == nullptr,
            HCCL_ERROR("[HcclCommunicator][AicpuInitOpTilingDataBuf] Alloc opTilingDataBuf failed!"), HCCL_E_INTERNAL);
    }

    if (opTilingDataBuf_.ptr() != nullptr && opTilingDataSize > opTilingDataBuf_.size()) {
        opTilingDataBuf_.free();
        opTilingDataBuf_ = HostMem::alloc(opTilingDataSize);
        CHK_PRT_RET(
            opTilingDataBuf_.ptr() == nullptr,
            HCCL_ERROR(
                "[HcclCommunicator][AicpuInitOpTilingDataBuf] in create opTilingDataBuf len[%llu] failed!",
                opTilingDataSize),
            HCCL_E_INTERNAL);
    }

    // 填充固定内容
    HostMem opTilingDataMem = opTilingDataBuf_.range(0, opTilingDataSize);
    struct OpTilingData* opTilingData = static_cast<struct OpTilingData*>(opTilingDataMem.ptr());
    u32 algTypeTranfer
        = (static_cast<u32>(opTilingInfo.algType.algoLevel2) << (HCCL_LEVEL_ALGO_WIDTH + HCCL_LEVEL_ALGO_WIDTH))
          + (static_cast<u32>(opTilingInfo.algType.algoLevel1) << HCCL_LEVEL_ALGO_WIDTH)
          + static_cast<u32>(opTilingInfo.algType.algoLevel0);
    opTilingData->algType = static_cast<u64>(algTypeTranfer);
    opTilingData->floatOverflowMode = opTilingInfo.floatOverflowMode;
    opTilingData->dumpDebug = opTilingInfo.dumpDebug;
    CHK_RET(AicpuInitOpTilingDataFromOpParam(opParam, opType, opTilingData));
    opTilingData->length = dynamicDataSize;
    opTilingData->customDataLength = 0;
    opTilingData->index = UpdateOpIndex(opParam);
    opTilingData->debugMode = 0;
    opTilingData->isZeroCopy = opParam.isZeroCopy;
    opTilingData->isCapture = opParam.isCapture;
    opTilingData->orderLaunchMode = GetOrderLaunchMode(opParam.isCapture);
    opTilingData->isSymmetricMemory = opParam.supportSymmetricMemory;
    opTilingData->needIncreLink = opParam.needIncreLink;
    // 有没有存在对应的Notify
    CHK_RET(InitAndCheckAicpuOrderNotify(opTilingData->orderLaunchMode));
    CHK_RET(BuildHierarchicalAlgOption(opTilingData->ahcConfInfo));
    CHK_RET(AicpuInitOpTilingDataAicpuCache(opParam, opType, opTilingData));

    // 填充动态内容
    HostMem dynamicDataMem = opTilingDataBuf_.range(sizeof(struct OpTilingData), dynamicDataSize);
    CHK_PTR_NULL(dynamicDataMem.ptr());
    if (opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
        struct OpTilingBatchSendRecvDataDes* batchSendRecvDataPtr
            = reinterpret_cast<struct OpTilingBatchSendRecvDataDes*>(dynamicDataMem.ptr());
        batchSendRecvDataPtr->itemNum = opParam.BatchSendRecvDataDes.itemNum;
        for (u32 i = 0; i < opParam.BatchSendRecvDataDes.itemNum; i++) {
            CHK_PTR_NULL(opParam.BatchSendRecvDataDes.sendRecvItemsPtr + i);
            batchSendRecvDataPtr->batchSendRecvItem[i] = *(opParam.BatchSendRecvDataDes.sendRecvItemsPtr + i);
        }

        u8* isDirectRemoteRankPtr
            = reinterpret_cast<u8*>(batchSendRecvDataPtr->batchSendRecvItem + opParam.BatchSendRecvDataDes.itemNum);
        for (u32 i = 0; i < userRankSize_; i++) {
            CHK_PTR_NULL(isDirectRemoteRankPtr + i);
            isDirectRemoteRankPtr[i] = *(opParam.BatchSendRecvDataDes.isDirectRemoteRank + i);
        }
    } else if (opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        CHK_RET(SetDynamicTilingDataAlltoall(opParam, dynamicDataMem));
    } else if (opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        CHK_RET(SetDynamicTilingDataAlltoallv(opParam, dynamicDataMem, opTilingInfo.algName));
    } else if (opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        CHK_RET(SetDynamicTilingDataAlltoallvc(opParam, dynamicDataMem));
    } else if (opType == HcclCMDType::HCCL_CMD_ALLGATHER_V || opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
        CHK_RET(SetDynamicTilingDataV(opParam, dynamicDataMem));
    } else {
        struct OpTilingDataDes* opDataDesPtr = reinterpret_cast<struct OpTilingDataDes*>(dynamicDataMem.ptr());
        opDataDesPtr->count = opParam.DataDes.count;
        opDataDesPtr->dataType = static_cast<u8>(opParam.DataDes.dataType);
    }

    HCCL_INFO("[HcclCommunicator][AicpuInitOpTilingDataBuf]algType[%lu]", opTilingData->algType);
    CHK_SAFETY_FUNC_RET(memcpy_s(
        opTilingData->algName, sizeof(opTilingData->algName), opTilingInfo.algName.c_str(),
        opTilingInfo.algName.length() + 1));
    CHK_SAFETY_FUNC_RET(memcpy_s(
        opTilingData->newTag, sizeof(opTilingData->newTag), opTilingInfo.newTag.c_str(),
        opTilingInfo.newTag.length() + 1));
    CHK_SAFETY_FUNC_RET(
        memcpy_s(opTilingData->tag, sizeof(opTilingData->tag), opParam.tag.c_str(), opParam.tag.length() + 1));
    return HCCL_SUCCESS;
}

u8 HcclCommunicator::GetOrderLaunchMode(bool isCapture)
{
    bool isSupportHcomAttachedStream
        = !(attachedStreams_.empty() || attachedStreams_[0].ptr() == nullptr); // true 表示图模式下成功申请附属从流
    u8 orderLaunchMode = 0;
    HcclWorkflowMode mode = GetWorkflowMode();
    if (isCapture) {
        orderLaunchMode = static_cast<u8>(AicpuNotifyMode::ACLGRAPH_MODE);
    } else if (mode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        orderLaunchMode = static_cast<u8>(AicpuNotifyMode::OPBASE_MODE);
    } else if (mode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB && isSupportHcomAttachedStream) {
        orderLaunchMode = static_cast<u8>(AicpuNotifyMode::HCOM_MODE);
    } else {
        orderLaunchMode = AICPU_ORDERLAUNCH_INVALID_HCOM_MODE;
    }

    return orderLaunchMode;
}

HcclResult HcclCommunicator::InitAndCheckAicpuOrderNotify(u8& orderLaunchMode)
{
    if (orderLaunchMode == AICPU_ORDERLAUNCH_INVALID_HCOM_MODE) {
        HCCL_INFO("[HcclCommunicator][InitAndCheckAicpuOrderNotify] orderLaunchMode is invalid in hcom "
                  "for there is no attached stream included in this operator!");
        return HCCL_SUCCESS;
    }
    u32 idx0;
    u32 idx1;
    if (orderLaunchMode == 0) {
        idx0 = static_cast<u32>(AicpuLocalNotifyIdx::ORDER_INDEX_OPBASE_0);
        idx1 = static_cast<u32>(AicpuLocalNotifyIdx::ORDER_INDEX_OPBASE_1);
    } else if (orderLaunchMode == 1) {
        idx0 = static_cast<u32>(AicpuLocalNotifyIdx::ORDER_INDEX_ACLGRAPH_0);
        idx1 = static_cast<u32>(AicpuLocalNotifyIdx::ORDER_INDEX_ACLGRAPH_1);
    } else {
        idx0 = static_cast<u32>(AicpuLocalNotifyIdx::ORDER_INDEX_HCOM_0);
        idx1 = static_cast<u32>(AicpuLocalNotifyIdx::ORDER_INDEX_HCOM_1);
    }

    if (localAiCpuOpNotify_[idx0] != nullptr) {
        HCCL_INFO("[%s], the orderNotify of orderLaunchMode [%u] is available", __func__, orderLaunchMode);
        return HCCL_SUCCESS;
    }
    HcclSignalInfo orderSignalInfo0;
    HcclResult ret = CreateAndGetAiCpuNotify(localAiCpuOpNotify_[idx0], orderSignalInfo0);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[HcclCommunicator][InitAndCheckAicpuOrderNotify]get aicpu notify [%u] errorCode[%u]", idx0,
            HCCL_ERROR_CODE(ret)),
        ret);

    // 按序下发(aicpu控制流 record host控制流) 使用的notify信息
    HcclSignalInfo orderSignalInfo1;
    ret = CreateAndGetAiCpuNotify(localAiCpuOpNotify_[idx1], orderSignalInfo1);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[HcclCommunicator][InitAndCheckAicpuOrderNotify]get aicpu notify [%u] errorCode[%u]", idx1,
            HCCL_ERROR_CODE(ret)),
        ret);
    HCCL_INFO(
        "[HcclCommunicator][InitAndCheckAicpuOrderNotify] ORDER INDEX 0: resId[%u], ORDER INDEX 1: resId[%u]",
        orderSignalInfo0.resId, orderSignalInfo1.resId);

    CHK_RET(hrtMemSyncCopy(
        static_cast<char*>(aicpuOrderNotifyAddr_.ptr()) + (sizeof(HcclSignalInfo) * orderLaunchMode),
        sizeof(HcclSignalInfo), &orderSignalInfo1, sizeof(HcclSignalInfo),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AicpuKfcTilingDataLaunchIn(
    const OpParam& opParam, const DeviceMem& deviceContext, const std::string& kernelName,
    [[maybe_unused]] const AicpuOpTiling opTilingInfo, u64 opTilingDataSize, bool isCustom)
{
    HostMem opTilingDataMem = opTilingDataBuf_.range(0, opTilingDataSize);
    CHK_RET(SetNormalMode(dispatcher_));
    Stream& mainStream = const_cast<Stream&>(opParam.stream);
    CHK_RET(LocalNotify::Post(
        mainStream, dispatcher_, localAiCpuOpNotify_[static_cast<u32>(AicpuLocalNotifyIdx::HOST_TO_AICPU_0)],
        INVALID_VALUE_STAGE));

    Stream kfcOpStream;
    HcclWorkflowMode mode = GetWorkflowMode();
    bool isSupportHcomAttachedStream
        = !(attachedStreams_.empty() || attachedStreams_[0].ptr() == nullptr); // true 表示图模式下成功申请附属从流
    if (opParam.isCapture || mode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        kfcOpStream = opStream_;
    } else {
        // 如果是图模式，则尝试从附属从流中获取一下stream，如果能拿到则使用，否则退化
        if (isSupportHcomAttachedStream) {
            HCCL_INFO("[HcclCommunicator][AicpuKfcTilingDataLaunchIn] attachedStreams_ is valid in graph mode");
            kfcOpStream = attachedStreams_[0];
        } else {
            HCCL_INFO("[HcclCommunicator][AicpuKfcTilingDataLaunchIn] attachedStreams_ is invalid in graph mode");
            kfcOpStream = opParam.stream;
        }
    }
    uint64_t beginTime = hrtMsprofSysCycleTime();
    std::string profName = GetCMDTypeEnumStr(opParam.opType);
    if (profName == "Invalid HcclCMDType" || profName == "invalid") {
        profName = "HcclOpAicpuKernel";
    } else {
        profName += "AicpuKernel";
    }
    s32 streamId = kfcOpStream.id();
    auto getAicpuTaskExceptionCallBack = [this]() {
        return this->GetAicpuTaskException();
    };
    RegisterGetAicpuTaskExceptionCallBack(streamId, deviceLogicId_, getAicpuTaskExceptionCallBack);
    aicpuStreamIds_.insert(streamId);
    if (streamId != opParam.stream.id()) {
        RegisterGetAicpuTaskExceptionCallBack(opParam.stream.id(), deviceLogicId_, getAicpuTaskExceptionCallBack);
        aicpuStreamIds_.insert(opParam.stream.id());
    }

    HCCL_INFO(
        "%s profName[%s] tag[%s] kfcOpStreamId[%d] mainStreamId[%u] kfcStreamId[%d] isCapture[%d] mode[%d] ", __func__,
        profName.c_str(), opParam.tag.c_str(), streamId, opParam.stream.id(), opStream_.id(), opParam.isCapture, mode);

    if (opParam.isCapture) { // 非主流下发时，acl graph场景，capture从流
        u64 modelId = UINT64_MAX;
        rtModel_t rtModel = nullptr;
        bool isCapture = false;
        CHK_RET(GetStreamCaptureInfo(opParam.stream.ptr(), rtModel, isCapture));
        CHK_PTR_NULL(rtModel);
        CHK_RET(AddStreamToModel(kfcOpStream.ptr(), rtModel));
        CHK_RET(GetModelId(rtModel, modelId));
        HCCL_INFO(
            "[HcclCommunicator][%s]tag[%s], add stream[%d] to modelId[%llu] success.", __func__, opParam.tag.c_str(),
            streamId, modelId);
    }

    u32 timeOut = (opResPara_.config.notifyWaitTime == 0) ? opResPara_.config.notifyWaitTime :
                                                            (opResPara_.config.notifyWaitTime + AICPU_H2D_TIMEOUT_INC);
    OrderLaunch& orderLaunch = OrderLaunch::GetInstance(deviceLogicId_);
    std::shared_ptr<LocalNotify> notify0;
    std::shared_ptr<LocalNotify> notify1;
    if (opParam.isCapture) {
        notify0 = localAiCpuOpNotify_[static_cast<u32>(AicpuLocalNotifyIdx::ORDER_INDEX_ACLGRAPH_0)];
        notify1 = localAiCpuOpNotify_[static_cast<u32>(AicpuLocalNotifyIdx::ORDER_INDEX_ACLGRAPH_1)];
        HcclRtEvent event0 = localAicpuOpEvent_[static_cast<u32>(AicpuLocalEventIdx::ORDER_INDEX_ACLGRAPH_EVENT_0)];
        CHK_RET(orderLaunch.AclgraphLaunchInOrderToOrderStream(
            identifier_, kfcOpStream, opParam.stream, notify0, notify1, timeOut, event0));
    } else if (mode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        notify0 = localAiCpuOpNotify_[static_cast<u32>(AicpuLocalNotifyIdx::ORDER_INDEX_OPBASE_0)];
        notify1 = localAiCpuOpNotify_[static_cast<u32>(AicpuLocalNotifyIdx::ORDER_INDEX_OPBASE_1)];
        CHK_RET(orderLaunch.OpbaseLaunchInOrder(identifier_, kfcOpStream, notify0, notify1, timeOut));
    } else if (mode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB && isSupportHcomAttachedStream) {
        notify0 = localAiCpuOpNotify_[static_cast<u32>(AicpuLocalNotifyIdx::ORDER_INDEX_HCOM_0)];
        notify1 = localAiCpuOpNotify_[static_cast<u32>(AicpuLocalNotifyIdx::ORDER_INDEX_HCOM_1)];
        CHK_RET(orderLaunch.HcomLaunchInOrder(identifier_, kfcOpStream, graphId_, notify0, notify1, timeOut));
    }
    CHK_RET(KernelLaunchChooseAicpuOrCustom(
        opParam.inputPtr, opParam.outputPtr, kfcOpStream.ptr(), reinterpret_cast<u64>(deviceContext.ptr()),
        opTilingDataMem.ptr(), opTilingDataSize, kernelName, mode, opParam.tag, isCustom));
    if (opParam.isCapture) {
        HcclRtEvent event1 = localAicpuOpEvent_[static_cast<u32>(AicpuLocalEventIdx::ORDER_INDEX_ACLGRAPH_EVENT_1)];
        CHK_RET(orderLaunch.AclgraphLaunchInOrderToKernelStream(identifier_, kfcOpStream, event1));
    }

    uint64_t endTime = hrtMsprofSysCycleTime();
    s32 threadId = SalGetTid();
    CHK_RET(ProfilingManagerPub::CallMsprofReportNodeInfo(beginTime, endTime, profName, threadId));
    CHK_RET(LocalNotify::Wait(
        mainStream, dispatcher_, localAiCpuOpNotify_[static_cast<u32>(AicpuLocalNotifyIdx::HOST_TO_AICPU_1)],
        INVALID_VALUE_STAGE, timeOut));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetAttachedStream(u32 graphId, const std::vector<rtStream_t>& streams)
{
    constexpr u32 GRAPH_ATTACHED_STREAM_INDEX = 0; // 图粒度的附属从流
    constexpr u32 GROUP_ATTACHED_STREAM_INDEX = 1; // 通信域粒度的附属从流

    // 在图模式下，通信使用的附属从流可能不同，所以这里直接刷新所有
    attachedStreams_.clear();

    bool isValid = !streams.empty() && (streams.size() > GROUP_ATTACHED_STREAM_INDEX)
                   && streams[GRAPH_ATTACHED_STREAM_INDEX] != nullptr
                   && streams[GROUP_ATTACHED_STREAM_INDEX] != nullptr;
    if (!isValid) {
        HCCL_ERROR("%s Invalid stream configuration, streams vector is null or invalid", __func__);
        return HCCL_E_NOT_FOUND;
    }

    // 向GE申请流的时候，图粒度的流排在第一个，所以在streams列表中，第一条流是图粒度的附属从流
    s32 graphAttachedStreamId = 0;
    OrderLaunch& orderLaunch = OrderLaunch::GetInstance(deviceLogicId_);
    auto& graphStream = streams[GRAPH_ATTACHED_STREAM_INDEX];
    CHK_RET(hrtGetStreamId(graphStream, graphAttachedStreamId));
    orderLaunch.SetHcomStream(graphId, Stream(graphStream, false));
    graphId_ = graphId;

    // 设置通信域粒度流
    auto& groupStream = streams[GROUP_ATTACHED_STREAM_INDEX];
    attachedStreams_.emplace_back(Stream(groupStream, false));

    HCCL_INFO(
        "%s Streams configured graph[%u], graphAttachedStreamId[%d], group[%u],"
        "groupStreamId[%u], graphId[%u], groupId[%s]",
        __func__, GRAPH_ATTACHED_STREAM_INDEX, graphAttachedStreamId, GROUP_ATTACHED_STREAM_INDEX,
        attachedStreams_.back().id(), graphId, identifier_.c_str());

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AicpuKfcTilingDataLaunchExt(
    const OpParam& opParam, const HcclCMDType& opType, const DeviceMem& deviceContext, const std::string& kernelName,
    const AicpuOpTiling opTilingInfo, bool isCustom)
{
    const u64 dataCount = opParam.GetDataCount(userRank_);
    const HcclDataType dataType = opParam.GetDataType();
    HCCL_DEBUG(
        "AicpuKfcTilingDataLaunchExt count %llu dataType %s op %s opType %u retryEnable_ %d, "
        "inPlaceSupportRetryStatus_ %d",
        dataCount, GetDataTypeEnumStr(dataType).c_str(), GetReduceOpEnumStr(opParam.reduceType).c_str(), opType,
        retryEnable_, inPlaceSupportRetryStatus_);

    bool postSyncEnable = false;
    u32 severNum4PostSync = 4;
    bool needPostSync = (superPodNum_ > 1 || serverNum_ >= severNum4PostSync)
                        && postSyncEnable; // reduce/reduce scatter算子是否需要PostSync
    if (opType == HcclCMDType::HCCL_CMD_ALLREDUCE && retryEnable_
        && (inPlaceSupportRetryStatus_ == InplaceSupportRetryStatus::USER_LARGER_THAN_CCL) && (!opParam.isZeroCopy)) {
        u32 itemNum = 2;
        for (u32 i = 0; i < itemNum; i++) {
            if (i == 0) {
                isInplacePreSync_ = true;
            } else {
                isInplacePreSync_ = false;
            }
            HCCL_DEBUG("[AicpuKfcTilingDataLaunchExt][PreSync]The op with isInplacePreSync_[%d].", isInplacePreSync_);
            u64 dynamicDataSize = CalcOpTilingDynamicDataSize(opParam, opType, GetRankSize(), opTilingInfo.algName);
            CHK_RET(AicpuInitOpTilingDataBuf(opParam, opType, kernelName, opTilingInfo, dynamicDataSize));
            CHK_RET(AicpuKfcTilingDataLaunchIn(
                opParam, deviceContext, kernelName, opTilingInfo, sizeof(struct OpTilingData) + dynamicDataSize,
                isCustom));
            isInplacePreSync_ = false;
        }
    } else if (opType == HcclCMDType::HCCL_CMD_REDUCE && retryEnable_ && needPostSync && (!opParam.isZeroCopy)) {
        isPostSync_ = true;
        HCCL_DEBUG("[AicpuKfcTilingDataLaunchExt][PreSync]The op with isPostSync_[%d].", isPostSync_);
        u64 dynamicDataSize = CalcOpTilingDynamicDataSize(opParam, opType, GetRankSize(), opTilingInfo.algName);
        CHK_RET(AicpuInitOpTilingDataBuf(opParam, opType, kernelName, opTilingInfo, dynamicDataSize));
        CHK_RET(AicpuKfcTilingDataLaunchIn(
            opParam, deviceContext, kernelName, opTilingInfo, sizeof(struct OpTilingData) + dynamicDataSize, isCustom));
        isPostSync_ = false;
    } else if (retryEnable_ && opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER && (!opParam.isZeroCopy)) {
        if (inPlaceSupportRetryStatus_ == InplaceSupportRetryStatus::USER_LARGER_THAN_CCL) {
            isInplacePreSync_ = true;
            HCCL_DEBUG("[AicpuKfcTilingDataLaunchExt][PreSync]The op with isInplacePreSync_[%d].", isInplacePreSync_);
            u64 dynamicDataSize = CalcOpTilingDynamicDataSize(opParam, opType, GetRankSize(), opTilingInfo.algName);
            CHK_RET(AicpuInitOpTilingDataBuf(opParam, opType, kernelName, opTilingInfo, dynamicDataSize));
            CHK_RET(AicpuKfcTilingDataLaunchIn(
                opParam, deviceContext, kernelName, opTilingInfo, sizeof(struct OpTilingData) + dynamicDataSize,
                isCustom));
            isInplacePreSync_ = false;
        }
        isInplacePreSync_ = false;
        if (needPostSync) {
            isPostSync_ = true;
        }
        HCCL_DEBUG(
            "[AicpuKfcTilingDataLaunchExt][PreSync]The op with "
            "isInplacePreSync_[%d], isPostSync_[%d].",
            isInplacePreSync_, isPostSync_);
        u64 dynamicDataSize = CalcOpTilingDynamicDataSize(opParam, opType, GetRankSize(), opTilingInfo.algName);
        CHK_RET(AicpuInitOpTilingDataBuf(opParam, opType, kernelName, opTilingInfo, dynamicDataSize));
        CHK_RET(AicpuKfcTilingDataLaunchIn(
            opParam, deviceContext, kernelName, opTilingInfo, sizeof(struct OpTilingData) + dynamicDataSize, isCustom));
        isPostSync_ = false;
    } else if (
        retryEnable_
        && (opType == HcclCMDType::HCCL_CMD_ALLTOALL || opType == HcclCMDType::HCCL_CMD_ALLTOALLV
            || opType == HcclCMDType::HCCL_CMD_ALLTOALLVC)
        && (!opParam.isZeroCopy)) {
        isPostSync_ = postSyncEnable;
        HCCL_DEBUG(
            "[AicpuKfcTilingDataLaunchExt][PreSync]The op with "
            "isInplacePreSync_[%d], isPostSync_[%d].",
            isInplacePreSync_, isPostSync_);
        u64 dynamicDataSize = CalcOpTilingDynamicDataSize(opParam, opType, GetRankSize(), opTilingInfo.algName);
        CHK_RET(AicpuInitOpTilingDataBuf(opParam, opType, kernelName, opTilingInfo, dynamicDataSize));
        CHK_RET(AicpuKfcTilingDataLaunchIn(
            opParam, deviceContext, kernelName, opTilingInfo, sizeof(struct OpTilingData) + dynamicDataSize, isCustom));
        isPostSync_ = false;
    } else {
        u64 dynamicDataSize = CalcOpTilingDynamicDataSize(opParam, opType, GetRankSize(), opTilingInfo.algName);
        HCCL_DEBUG("[AicpuKfcTilingDataLaunchExt]dynamicDataSize[%u]", dynamicDataSize);
        CHK_RET(AicpuInitOpTilingDataBuf(opParam, opType, kernelName, opTilingInfo, dynamicDataSize));
        CHK_RET(AicpuKfcTilingDataLaunchIn(
            opParam, deviceContext, kernelName, opTilingInfo, sizeof(struct OpTilingData) + dynamicDataSize, isCustom));
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AicpuUnfoldKernelLaunch(
    void* inputPtr, void* outputPtr, const rtStream_t stm, u64 addr, void* tilingDataPtr, u32 tilingDataSize,
    const std::string& kernelName, [[maybe_unused]] HcclWorkflowMode mode, [[maybe_unused]] const std::string& tag)
{
    struct ApiParamDef {
        uint64_t x1; // 算子sendbuffer地址
        uint64_t y = 0;
        uint64_t gatherOut; // 算子recvbuffer地址
        uint64_t context;   // 通信资源准备的地址
        uint64_t workspace; // 消息区地址
    };

    struct ApiParamDef apiParam;
    apiParam.x1 = reinterpret_cast<uint64_t>(inputPtr);
    apiParam.gatherOut = reinterpret_cast<uint64_t>(outputPtr);
    apiParam.context = addr;
    apiParam.workspace = reinterpret_cast<uint64_t>(workSpace_.ptr());
    u16 timeOut = 0;
    if (opResPara_.config.notifyWaitTime == 0) {
        timeOut = opResPara_.config.notifyWaitTime;
    } else if (opResPara_.config.notifyWaitTime + AICPU_KERNEL_TIMEOUT_INC >= MAX_VALUE_U16) {
        timeOut = MAX_VALUE_U16;
    } else {
        timeOut = opResPara_.config.notifyWaitTime + AICPU_KERNEL_TIMEOUT_INC;
    }
    CHK_PRT(AicpuAclKernelLaunch(
        stm, static_cast<void*>(&apiParam), sizeof(apiParam), binHandle_, kernelName, false, timeOut, tilingDataPtr,
        tilingDataSize));
    HCCL_INFO("[HcclCommunicator][AicpuUnfoldKernelLaunch] exec succ.");
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::AicpuUnfoldKernelLaunchV2(
    [[maybe_unused]] void* inputPtr, [[maybe_unused]] void* outputPtr, const rtStream_t stm, u64 addr,
    void* tilingDataPtr, u32 tilingDataSize, const std::string& kernelName, [[maybe_unused]] HcclWorkflowMode mode,
    [[maybe_unused]] const std::string& tag, bool isCustom)
{
    u64 context = addr;
    HCCL_INFO("[HcclCommunicator]context[%p] tilingDataPtr[%p] tilingData[%p]", context, tilingDataPtr, tilingDataSize);

    aclrtBinHandle binHandle = isCustom ? binCustomHandle_ : binHandle_;
    if (binHandle == nullptr) {
        HCCL_ERROR("[AicpuUnfoldKernelLaunchV2]isCustom[%d] binHandle is nullptr, please check.", isCustom);
        return HCCL_E_NOT_SUPPORT;
    }
    u16 timeOut = 0;
    if (opResPara_.config.notifyWaitTime == 0) {
        timeOut = opResPara_.config.notifyWaitTime;
    } else if (opResPara_.config.notifyWaitTime + AICPU_KERNEL_TIMEOUT_INC >= MAX_VALUE_U16) {
        timeOut = MAX_VALUE_U16;
    } else {
        timeOut = opResPara_.config.notifyWaitTime + AICPU_KERNEL_TIMEOUT_INC;
    }
    HcclResult ret = AicpuAclKernelLaunchV2(
        stm, static_cast<void*>(&context), sizeof(context), binHandle, kernelName, false, timeOut, tilingDataPtr,
        tilingDataSize, identifier_);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[HcclCommunicator][AicpuUnfoldKernelLaunchV2]isCustom[%d] binHandle[%p]", isCustom, binHandle),
        ret);
    HCCL_INFO("[HcclCommunicator][AicpuUnfoldKernelLaunchV2] exec succ, isCustom[%d].", isCustom);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::InitCombinOpara()
{
    if (combinOparaMem_ == nullptr) {
        CHK_RET(AllocAndClearHostMem(sizeof(HcclCombinOpParam), combinOparaMem_));
    }
    CHK_PTR_NULL(combinOparaMem_);
    HcclCombinOpParam* combinOparaPtr = reinterpret_cast<HcclCombinOpParam*>(combinOparaMem_->ptr());
    CHK_PTR_NULL(combinOparaPtr);

    if (aiRMAInfoMem_ == nullptr) {
        CHK_RET(AllocAndClearHostMem(sizeof(HcclAiRMAInfo), aiRMAInfoMem_));
    }
    if (rmaInfoMem_ == nullptr) {
        CHK_RET(AllocAndClearHostMem(sizeof(HcclRMAInfo), rmaInfoMem_));
    }
    CHK_PTR_NULL(aiRMAInfoMem_);
    CHK_PTR_NULL(aiRMAInfoMem_->ptr());
    CHK_PTR_NULL(rmaInfoMem_);
    CHK_PTR_NULL(rmaInfoMem_->ptr());

    CHK_SAFETY_FUNC_RET(memset_s(combinOparaPtr, sizeof(HcclCombinOpParam), 0, sizeof(HcclCombinOpParam)));

    combinOparaPtr->rankId = INVALID_UINT;
    combinOparaPtr->signalInfo.aicpuNotify.rankId = INVALID_UINT;

    for (u32 i = 0;
         i < sizeof(combinOparaPtr->signalInfo.noIpcNotifys) / sizeof(combinOparaPtr->signalInfo.noIpcNotifys[0]);
         i++) {
        combinOparaPtr->signalInfo.noIpcNotifys[i].rankId = INVALID_UINT;
    }

    for (u32 i = 0;
         i < sizeof(combinOparaPtr->signalInfo.ipcNotifys) / sizeof(combinOparaPtr->signalInfo.ipcNotifys[0]); i++) {
        combinOparaPtr->signalInfo.ipcNotifys[i].rankId = INVALID_UINT;
    }

    for (u32 i = 0;
         i < sizeof(combinOparaPtr->signalInfo.noIpcEvents) / sizeof(combinOparaPtr->signalInfo.noIpcEvents[0]); i++) {
        combinOparaPtr->signalInfo.noIpcEvents[i].rankId = INVALID_UINT;
    }
    return HCCL_SUCCESS;
}

bool HcclCommunicator::GetCommResource(const std::string& tag, void** commContext)
{
    if (LIKELY(IsExistCommRes(tag))) {
        *commContext = commContext_.ptr();
        return true;
    }
    return false;
}

bool HcclCommunicator::GetCommResource(void*& commContext)
{
    commContext = opResDevicePara_.ptr();
    return true;
}

HcclResult HcclCommunicator::GetAicpuOpStreamNotify(HcclRtStream* opStream, u8 aicpuNotifyNum, void** aicpuNotify)
{
    CHK_RET(GetAicpuOpStreamAndNotify(opStream, aicpuNotifyNum, aicpuNotify));
    HCCL_INFO("[HcclCommunicator][GetAicpuOpStreamNotify]opStream %p aicpuNotify %p.", *opStream, *aicpuNotify);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetAicpuOpStreamAndNotify(HcclRtStream* opStream, u8 aicpuNotifyNum, void** aicpuNotify)
{
    *opStream = opStream_.ptr();
    if (localAiCpuNotifyRes_.size() < aicpuNotifyNum) {
        for (u16 i = localAiCpuNotifyRes_.size(); i < aicpuNotifyNum; i++) {
            std::shared_ptr<LocalNotify> localNotify = {nullptr};
            HcclSignalInfo aicpuNotify;
            CHK_RET(CreateAndGetAiCpuNotify(localNotify, aicpuNotify));
            localAiCpuNotifyRes_.push_back(localNotify);
        }
    }

    for (u16 i = 0; i < aicpuNotifyNum; i++) {
        *(aicpuNotify + i) = localAiCpuNotifyRes_[i]->ptr();
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetAicpuNotifyInvalid()
{
    CHK_PTR_NULL(combinOparaMem_);
    HcclCombinOpParam* combinOparaPtr = reinterpret_cast<HcclCombinOpParam*>(combinOparaMem_->ptr());
    CHK_PTR_NULL(combinOparaPtr);
    combinOparaPtr->signalInfo.aicpuNotify.resId = INVALID_U64;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ReplaceCommInfoByTag(const std::string& tag, std::unique_ptr<CommInfo>& commInfo)
{
    std::unique_lock<std::mutex> replLock(commLock_);
    tagCommInfo_.erase(tag);
    tagCommInfo_.insert(std::pair<std::string, CommInfo>(tag, std::move(*commInfo)));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CreateMutiStreamResFor310P(const std::string& tag, level1StreamInfo_t& streamInfo)
{
    u32 rankSize = GetRankSize();
    s32 pid;
    if (SalGetBareTgid(&pid) != HCCL_SUCCESS) {
        HCCL_DEBUG("get pid fail");
    }
    HCCL_INFO(
        "[HcclCommunicator][CreateMutiStreamRes]tag[%s] ranksize[%u] comminfo ranksize[%u] "
        "auxRingCommStreamsDev_ size[%u] ringDeviceSignalAux size[%u] ringDeviceSignal size[%u] "
        "ringDeviceStreams size[%u]",
        tag.c_str(), rankSize, tagCommInfo_[tag].commIntraServer->RankSize(), auxRingCommStreamsDev_.size(),
        streamInfo.ringDeviceSignalAux.size(), streamInfo.ringDeviceSignal.size(), streamInfo.ringDeviceStreams.size());
    if (auxRingCommStreamsDev_.empty() || auxRingCommStreamsDev_.size() < rankSize) {
        auxRingCommStreamsDev_.resize(rankSize);
        u32 resNum = rankSize - 1;
        streamInfo.ringDeviceSignalAux.resize(resNum);
        streamInfo.ringDeviceSignal.resize(resNum);
        for (u32 ringIndex = 0; ringIndex < rankSize; ringIndex++) {
            auxRingCommStreamsDev_[ringIndex] = Stream(StreamType::STREAM_TYPE_DEVICE);
            // 给device侧申请的流不需要setmode，否则rts会捕获流成员Flags为1024的异常
        }
        for (auto& signal : streamInfo.ringDeviceSignal) {
            signal = nullptr;
        }
        for (auto& signal : streamInfo.ringDeviceSignalAux) {
            signal = nullptr;
        }

        u32 notifyNum = resNum * 2; // 2:Signal + SignalAux
        std::vector<std::shared_ptr<LocalNotify>> notifys(notifyNum, nullptr);
        CHK_RET(queueNotifyManager_->Alloc(tag, notifyNum, notifys, NotifyLoadType::DEVICE_NOTIFY));
        for (u32 i = 0; i < resNum; i++) {
            streamInfo.ringDeviceSignal[i] = notifys[2 * i];
            streamInfo.ringDeviceSignalAux[i] = notifys[2 * i + 1];
        }
    }

    if (streamInfo.ringDeviceStreams.empty() || streamInfo.ringDeviceStreams.size() < rankSize) {
        streamInfo.ringDeviceStreams.resize(rankSize);
        for (u32 ringIndex = 0; ringIndex < rankSize; ringIndex++) {
            streamInfo.ringDeviceStreams[ringIndex] = auxRingCommStreamsDev_[ringIndex];
            CHK_SMART_PTR_NULL(streamInfo.ringDeviceStreams[ringIndex]);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CreateCommAndStreamRes(const std::string& tag, Stream& stream)
{
    CHK_SMART_PTR_NULL(implAlg_);
    void* commInputPtr = nullptr;
    void* commOutputPtr = nullptr;
    u64 commInputSize, commOutputSize;

    HcclResult ret = CreateCommCCLbuffer();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[HcclImplBase][CreateCommAndStreamRes]errNo[0x%016llx],create cclbuff failed", HCCL_ERROR_CODE(ret)),
        ret);

    if (isA2MC2MultiServer_) {
        // 该场景下ccl buffer有一块区域在上层会被用作flag区，因此需要先清理一下
        CHK_RET(cclBufferManager_.CleanCCLbuffer());
    }

    CHK_RET(cclBufferManager_.GetInCCLbuffer(commInputPtr, commInputSize));
    CHK_RET(cclBufferManager_.GetOutCCLbuffer(commOutputPtr, commOutputSize));
    DeviceMem expMem = cclBufferManager_.GetCommExpBuffer();
    DeviceMem inputMem = DeviceMem::create(commInputPtr, commInputSize);
    DeviceMem outputMem = DeviceMem::create(commOutputPtr, commOutputSize);
    AlgType algType;
    AlgType algTypeTmp;

    CHK_RET(GetAlgType(algType, HcclCMDType::HCCL_CMD_ALL));
    algTypeTmp = algType;

    CHK_RET(notifyPool_->RegisterOp(tag));

    // 根据tag创建comm和流资源
    if (!(IsExistCommRes(tag))) {
        std::unique_ptr<CommInfo> commInfo = nullptr;
        std::vector<u32>& ranksPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
        std::vector<u32>& vnicRanksPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
        implAlg_->SetHDCModeInfo(
            rankDevicePhyIdNicInfoMap_, ranksPorts, vnicRanksPorts, isSetHDCModeInfo_, isUseRankPort_);
        HcclResult ret
            = implAlg_->CreateComm(tag, inputMem, outputMem, algType, commInfo, INVALID_VALUE_RANKID, false, true);

        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[HcclCommunicator][CreateCommAndStreamRes]errNo[0x%016llx]tag[%s],comm resource create comm failed",
                HCCL_ERROR_CODE(ret), tag.c_str()),
            ret);

        CHK_RET(ReplaceCommInfoByTag(tag, commInfo));
        if (isA2MC2MultiServer_ && isA2MC2IntraHie_) {
            std::string hieSuffix = "_HIE";
            size_t pos = tag.find(hieSuffix);
            std::string oldtag = tag;
            oldtag.erase(pos, hieSuffix.size());
            CHK_RET(ReplaceCommInfoByTag(oldtag, commInfo));
        }
    }

    if (!(IsExistMutiStreamRes(tag))) {
        level1StreamInfo_t streamInfo;
        std::unique_lock<std::mutex> mutiStreamLock(tagStreamInfoLock_);
        // 2p场景下，mc2当前algType为518，streamInfo.ringNum走默认流程值为1导致资源申请不足，910_93
        // mc2固定在节点内默认用mesh
        constexpr u32 RANK_SIZE_TWO = 2;
        if ((GetRankSize() == RANK_SIZE_TWO && !isA2MC2MultiServer_) || (deviceType_ == DevType::DEV_TYPE_910_93)) {
            algTypeTmp.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_MESH;
            algTypeTmp.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
        }
        HcclResult ret = HCCL_SUCCESS;
        if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
            ret = CreateMutiStreamResFor310P(tag, streamInfo);
        } else {
            ret = implAlg_->CreateMutiStreamRes(tag, stream, streamInfo, algTypeTmp, true);
        }
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[HcclCommunicator][CreateCommAndStreamRes]errNo[0x%016llx]tag[%s],comm resource create stream "
                "resource",
                HCCL_ERROR_CODE(ret), tag.c_str()),
            ret);
        tagStreamInfo_.insert(std::pair<std::string, Level1StreamInfo>(tag, std::move(streamInfo)));
        opRetryStreamPtr_->insert(std::make_pair(tag, tagStreamInfo_[tag].ringDeviceStreams));
        mutiStreamLock.unlock();
    }

    HCCL_INFO("resource creation (AllReduce) success, tag[%s]", tag.c_str());
    CHK_RET(notifyPool_->UnregisterOp(tag));
    if (commPortConfig_.devPortSwitchOn) {
        std::vector<u32>& nicPorts = groupNicRanksPort_.empty() ? nicRanksPort_ : groupNicRanksPort_;
        std::vector<u32>& vnicPorts = groupVnicRanksPort_.empty() ? vnicRanksPort_ : groupVnicRanksPort_;
        Heartbeat::GetInstance(deviceLogicId_)
            .SetRankPortInfo(isUseRankPort_, nicPorts, vnicPorts, commPortConfig_.devPortSwitchOn);
    }
    CHK_RET(RegisterToHeartBeat());

    CommBase* comm = nullptr;
    CHK_RET(GetComm(tag, &comm));
    if (comm == nullptr) {
        HCCL_ERROR("comm get err, comm %p", comm);
        return HCCL_E_PTR;
    }
    CHK_RET(
        SetCommResource(commInputSize, commInputPtr, commOutputPtr, expMem.ptr(), comm, tagStreamInfo_[tag], stream));

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetComm(const std::string& tag, CommBase** comm)
{
    if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
        *comm = tagCommInfo_[tag].commIntraServer.get();
    } else if (isA2MC2MultiServer_) {
        // 使用打平RDMA Mesh子通信域
        *comm = tagCommInfo_[tag].commLevel1Rdma[0].get();
    } else {
        *comm = tagCommInfo_[tag].commLevel0[0].get();
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetCommResource(
    u64 commBufferSize, void* commInPtr, void* commOutPtr, void* commExpPtr, CommBase* comm,
    level1StreamInfo_t& streamInfo, Stream& stream)
{
    CHK_PTR_NULL(combinOparaMem_);
    HcclCombinOpParam* combinOparaPtr = reinterpret_cast<HcclCombinOpParam*>(combinOparaMem_->ptr());
    CHK_PTR_NULL(combinOparaPtr);

    u32 rankSize = comm->RankSize();
    u32 curRankId = comm->Rank();
    u32 usrRankId = comm->UserRank();
    combinOparaPtr->rankId = curRankId;
    combinOparaPtr->signalInfo.aicpuNotify.rankId = curRankId;
    combinOparaPtr->rankNum = rankSize;
    combinOparaPtr->winSize = commBufferSize;
    combinOparaPtr->winExpSize = EXP_BUFFER_SIZE;
    combinOparaPtr->config.deterministic = GetDeterministicConfig();
    combinOparaPtr->config.notifyWaitTime
        = (GetExternalInputHcclExecTimeoutSet() != HcclExecTimeoutSet::HCCL_EXEC_TIMEOUT_NOT_SET
           || commConfig_.GetConfigExecTimeOutSet()) ?
              commConfig_.GetConfigExecTimeOut() :
              NOTIFY_DEFAULT_WAIT_TIME;
    hcclMc2Info_.groupName = hrtMsprofGetHashId(identifier_.c_str(), identifier_.length());
    combinOparaPtr->config.linkTimeOut = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());
    hcclMc2Info_.rankSize = rankSize;
    hcclMc2Info_.rankId = curRankId;
    hcclMc2Info_.usrRankId = usrRankId;
    hcclMc2Info_.aicpuKfcStreamId = static_cast<uint32_t>(stream.id());
    hcclMc2Info_.commStreamSize = rankSize;
    hcclMc2Info_.reserve = 0;
    rtEvent_t event = nullptr;
    u32 eventId = 0;
    u32 idx = 0;
    u32 txSigleBase = 2;
    u32 rxSigleBase = 3;

    if (isA2MC2MultiServer_) {
        // MoE融合算子优化，MC2多机场景
        // 判断是否支持NormalQP创建，若不支持，需要额外下发敲Doorbell任务
        bool isSupportNormalQP = false;
        CHK_RET(IsSupportAicpuNormalQP(devicePhyId_, isSupportNormalQP));
        CHK_RET(SetDevIbverbsData(comm, isSupportNormalQP, commBufferSize, commInPtr, commOutPtr));

        bool isSupportAIVNormalQP = false;
        CHK_RET(IsSupportAIVNormalQP(devicePhyId_, isSupportAIVNormalQP));
        if (isSupportAIVNormalQP && isA2MC2IntraHie_) {
            CHK_RET(GenAiRMAInfo(comm));
        } else {
            HCCL_WARNING(
                "[%s] db transfer normal qp not support. tag[%s] curRankId[%u] rankNum[%u] isSupportAIVNormalQP[%u]",
                __func__, comm->Tag().c_str(), curRankId, rankSize, isSupportAIVNormalQP);
        }

        if (combinedCapabilityMem_ == nullptr) {
            CHK_RET(AllocAndClearHostMem(sizeof(CombinedCapability), combinedCapabilityMem_));
        }
        CHK_PTR_NULL(combinedCapabilityMem_);
        CombinedCapability* combinedCapabilityPtr = static_cast<CombinedCapability*>(combinedCapabilityMem_->ptr());
        CHK_PTR_NULL(combinedCapabilityPtr);
        SalSetBitOne(combinedCapabilityPtr->dataplaneModeBitmap, POS_DATA_PLANE_MODE_HOST);
        if (isSupportAIVNormalQP && isA2MC2IntraHie_) {
            SalSetBitOne(combinedCapabilityPtr->dataplaneModeBitmap, POS_DATA_PLANE_MODE_AIV);
        }
        SalSetBitOne(combinedCapabilityPtr->dataplaneModeBitmap, POS_DATA_PLANE_MODE_AICPU);

        HCCL_INFO("[SetCommResource] Set dataplaneModeBitmap to [%llu]", combinedCapabilityPtr->dataplaneModeBitmap);

        // 非NormalQP场景需要传一条流，用于敲Doorbell
        combinOparaPtr->streamInfo[0].streamIds = streamInfo.ringDeviceStreams[0].id();
        combinOparaPtr->streamInfo[0].sqIds = streamInfo.ringDeviceStreams[0].sqId();
        combinOparaPtr->streamInfo[0].cqIds = streamInfo.ringDeviceStreams[0].cqId();
        combinOparaPtr->streamInfo[0].logicCqids = streamInfo.ringDeviceStreams[0].logicCqId();
        HCCL_DEBUG(
            "[SetCommResource] Set streamInfo[0].streamIds[%u].sqIds[%u].cqIds[%u].logicCqids[%u]",
            combinOparaPtr->streamInfo[0].streamIds, combinOparaPtr->streamInfo[0].sqIds,
            combinOparaPtr->streamInfo[0].cqIds, combinOparaPtr->streamInfo[0].logicCqids);
    } else {
        for (u32 i = 0; i < rankSize; i++) {
            if (i != curRankId) {
                void* bufferIn;
                void* bufferOut;
                std::vector<void*> remotePtrVec;
                auto& transport = comm->GetTransportByRank(i);
                CHK_PTR_NULL(transport);
                CHK_RET(transport->GetRemoteMem(UserMemType::INPUT_MEM, &bufferIn));
                combinOparaPtr->windowsIn[i] = reinterpret_cast<u64>(bufferIn);

                CHK_RET(transport->GetRemoteMem(UserMemType::OUTPUT_MEM, &bufferOut));
                combinOparaPtr->windowsOut[i] = reinterpret_cast<u64>(bufferOut);

                CHK_RET(transport->GetRemoteMem(&remotePtrVec));
                if (remotePtrVec.size() != 0) {
                    combinOparaPtr->windowsExp[i] = reinterpret_cast<u64>(remotePtrVec[0]);
                    if (transport->GetTransportType() == TransportType::TRANS_TYPE_P2P) {
                        p2pCclBuf_[i] = remotePtrVec[0];
                    } else {
                        cclBuf_[i] = remotePtrVec[0];
                    }
                    combinOparaPtr->windowsExp[i]
                        += cclBufferManager_.GetInCCLbufferSize() + cclBufferManager_.GetOutCCLbufferSize();
                }
                CHK_RET(transport->GetTxAckDevNotifyInfo(combinOparaPtr->signalInfo.ipcNotifys[i]));
                CHK_RET(transport->GetRxAckDevNotifyInfo(combinOparaPtr->signalInfo.ipcNotifys[i + rankSize]));
                CHK_RET(transport->GetTxDataSigleDevNotifyInfo(
                    combinOparaPtr->signalInfo.ipcNotifys[i + rankSize * txSigleBase]));
                CHK_RET(transport->GetRxDataSigleDevNotifyInfo(
                    combinOparaPtr->signalInfo.ipcNotifys[i + rankSize * rxSigleBase]));
                CHK_RET(GetAiCpuNotifyData(
                    streamInfo.ringDeviceSignalAux[idx], combinOparaPtr->signalInfo.noIpcNotifys[i]));

                CHK_RET(GetAiCpuNotifyData(
                    streamInfo.ringDeviceSignal[idx], combinOparaPtr->signalInfo.noIpcNotifys[i + rankSize]));
                idx++;
            } else {
                combinOparaPtr->windowsIn[i] = reinterpret_cast<u64>(commInPtr);
                combinOparaPtr->windowsOut[i] = reinterpret_cast<u64>(commOutPtr);
                combinOparaPtr->windowsExp[i] = reinterpret_cast<u64>(commExpPtr);
                // 在与aicpu商议后，本卡不再防止无效值。后续代码要删掉
                combinOparaPtr->signalInfo.ipcNotifys[i].resId = INVALID_U64;
                combinOparaPtr->signalInfo.ipcNotifys[i + rankSize].resId = INVALID_U64;
                combinOparaPtr->signalInfo.ipcNotifys[i + rankSize * txSigleBase].resId = INVALID_U64;
                combinOparaPtr->signalInfo.ipcNotifys[i + rankSize * rxSigleBase].resId = INVALID_U64;
            }
            HCCL_INFO(
                "group[%s] successfully set windowsIn & windowsOut & windowsExp info: userRank[%u], groupRank[%u], "
                "windowsIn[0x%llx], InSize[0x%llx], windowOut[0x%llx], OutSize[0x%llx], windowExp[0x%llx], "
                "ExpSize[0x%llu]",
                identifier_.c_str(), GetUserRank(), GetGroupRank(), combinOparaPtr->windowsIn[i],
                cclBufferManager_.GetInCCLbufferSize(), combinOparaPtr->windowsOut[i],
                cclBufferManager_.GetOutCCLbufferSize(), combinOparaPtr->windowsExp[i],
                cclBufferManager_.GetExpBufferSize());

            combinOparaPtr->signalInfo.ipcNotifys[i].rankId = i;
            combinOparaPtr->signalInfo.ipcNotifys[i + rankSize].rankId = i;
            combinOparaPtr->signalInfo.ipcNotifys[i + rankSize * txSigleBase].rankId = i;
            combinOparaPtr->signalInfo.ipcNotifys[i + rankSize * rxSigleBase].rankId = i;
            combinOparaPtr->signalInfo.noIpcNotifys[i].rankId = i;

            hcclMc2Info_.commStreamIds[i] = streamInfo.ringDeviceStreams[i].id();
            combinOparaPtr->streamInfo[i].streamIds = streamInfo.ringDeviceStreams[i].id();
            combinOparaPtr->streamInfo[i].sqIds = streamInfo.ringDeviceStreams[i].sqId();
            combinOparaPtr->streamInfo[i].cqIds = streamInfo.ringDeviceStreams[i].cqId();
            combinOparaPtr->streamInfo[i].logicCqids = streamInfo.ringDeviceStreams[i].logicCqId();
            HCCL_DEBUG("[hccl_Mc2_Info] commStreamIds[%u]:[%u]", i, streamInfo.ringDeviceStreams[i].id());

            CHK_RET(hrtEventCreateWithFlag(&event));

            CHK_RET(hrtGetEventID(event, &eventId));
            aiCpuNoIpcEvnet_.push_back(event);
            combinOparaPtr->signalInfo.noIpcEvents[i].resId = eventId;
            HCCL_DEBUG(
                "SetCommResource ipc notify info pre record local rankid: %u: remote rankid:%u, resId:%llu, "
                "devId:%u, tsId:%u, addr:%llu.",
                curRankId, combinOparaPtr->signalInfo.ipcNotifys[i].rankId,
                combinOparaPtr->signalInfo.ipcNotifys[i].resId, combinOparaPtr->signalInfo.ipcNotifys[i].devId,
                combinOparaPtr->signalInfo.ipcNotifys[i].tsId, combinOparaPtr->signalInfo.ipcNotifys[i].addr);
            HCCL_DEBUG(
                "SetCommResource ipc notify info pre wait local rankid: %u: remote rankid:%u, resId:%llu, "
                "devId:%u, tsId:%u, addr:%llu.",
                curRankId, combinOparaPtr->signalInfo.ipcNotifys[i + rankSize].rankId,
                combinOparaPtr->signalInfo.ipcNotifys[i + rankSize].resId,
                combinOparaPtr->signalInfo.ipcNotifys[i + rankSize].devId,
                combinOparaPtr->signalInfo.ipcNotifys[i + rankSize].tsId,
                combinOparaPtr->signalInfo.ipcNotifys[i + rankSize].addr);
            HCCL_DEBUG(
                "SetCommResource ipc notify info post record local rankid: %u: remote rankid:%u, resId:%llu, "
                "devId:%u, tsId:%u, addr:%llu.",
                curRankId, combinOparaPtr->signalInfo.ipcNotifys[i + rankSize * txSigleBase].rankId,
                combinOparaPtr->signalInfo.ipcNotifys[i + rankSize * txSigleBase].resId,
                combinOparaPtr->signalInfo.ipcNotifys[i + rankSize * txSigleBase].devId,
                combinOparaPtr->signalInfo.ipcNotifys[i + rankSize * txSigleBase].tsId,
                combinOparaPtr->signalInfo.ipcNotifys[i + rankSize * txSigleBase].addr);
            HCCL_DEBUG(
                "SetCommResource ipc notify info post wait local rankid: %u: remote rankid:%u, resId:%llu, "
                "devId:%u, tsId:%u, addr:%llu.",
                curRankId, combinOparaPtr->signalInfo.ipcNotifys[i + rankSize * rxSigleBase].rankId,
                combinOparaPtr->signalInfo.ipcNotifys[i + rankSize * rxSigleBase].resId,
                combinOparaPtr->signalInfo.ipcNotifys[i + rankSize * rxSigleBase].devId,
                combinOparaPtr->signalInfo.ipcNotifys[i + rankSize * rxSigleBase].tsId,
                combinOparaPtr->signalInfo.ipcNotifys[i + rankSize * rxSigleBase].addr);
        }
    }
    HCCL_DEBUG(
        "[hccl_Mc2_Info] groupname:[%s][%llu], rankSize[%u], rankId[%u], usrRankId[%u], aicpuKfcStreamId[%u], "
        "commStreamSize[%u]",
        identifier_.c_str(), hcclMc2Info_.groupName, rankSize, curRankId, usrRankId, static_cast<uint32_t>(stream.id()),
        rankSize);
    CHK_RET(
        ProfilingManagerPub::CallMsprofReportMc2CommInfo(hrtMsprofSysCycleTime(), &hcclMc2Info_, sizeof(hcclMc2Info_)));
    return HCCL_SUCCESS;
}

void HcclCommunicator::ReleaseCommContextbuffer() { commContext_.free(); }

HcclResult HcclCommunicator::CreateDeviceCommContext(u64 size, DeviceMem& buffer) const
{
    CHK_PRT_RET(
        !size,
        HCCL_INFO("[Create][DeviceCommContext]device commContext size is zero. "
                  "not need to malloc memory"),
        HCCL_SUCCESS);

    CHK_PRT_RET(
        (size > ULONG_MAX),
        HCCL_ERROR("[Create][DeviceCommContext]device commContext size %llu is larger than ULONG_MAX", size),
        HCCL_E_PARA);

    if (!buffer.ptr()) {
        u64 memSize = size;
        CHK_RET(DeviceMem::alloc(buffer, memSize));
    }
    return HCCL_SUCCESS;
}

void HcclCommunicator::Break()
{
    if (implAlg_ != nullptr) {
        implAlg_->Break();
    }
    return;
}

HcclResult HcclCommunicator::GetAlltoAllStagedWorkSpaceMemSize(
    u64* sendCounts, u64* sdispls, HcclDataType sendType, u64* recvCounts, u64* rdispls, HcclDataType recvType,
    u64& memSize)
{
    if (Is310P3Common(isHaveCpuRank_, deviceType_)) {
        HCCL_ERROR(
            "[%s][%s]GetAlltoAllStagedWorkSpaceMemSize Not Supported!", LOG_KEYWORDS_TASK_EXEC.c_str(),
            LOG_KEYWORDS_NOT_SUPPORTED.c_str());
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_SMART_PTR_NULL(implAlg_);
    std::unique_ptr<CollAlgOperator> algOperator = implAlg_->GetAlgOperator(HcclCMDType::HCCL_CMD_ALLTOALLV);
    AlltoAllOperator* alltoAllOperator = dynamic_cast<AlltoAllOperator*>(algOperator.get());
    CHK_PTR_NULL(alltoAllOperator);

    OpParam opParam;
    opParam.All2AllDataDes.sendType = sendType;
    opParam.All2AllDataDes.recvType = recvType;
    opParam.All2AllDataDes.sendCounts = static_cast<void*>(sendCounts);
    opParam.All2AllDataDes.recvCounts = static_cast<void*>(recvCounts);
    opParam.All2AllDataDes.sdispls = static_cast<void*>(sdispls);
    opParam.All2AllDataDes.rdispls = static_cast<void*>(rdispls);
    opParam.opType = HcclCMDType::HCCL_CMD_ALLTOALLV;
    opParam.aicpuUnfoldMode = false;
    opParam.aicpuCacheEnable = 0;

    if (alltoAllOperator->IsSatisfyAlltoAllAivCondition(opParam)
        || alltoAllOperator->IsSatisfy91093OffloadCondition()) {
        memSize = 0;
        HCCL_INFO("Calculate workSpace MemSize for aiv AllToAll done, memSize[%llu]", memSize);
        return HCCL_SUCCESS;
    }

    std::unique_ptr<PreProcessMetaInfo> preMetaInfo = std::make_unique<PreProcessMetaInfo>();
    CHK_SMART_PTR_NULL(preMetaInfo);

    CHK_RET(alltoAllOperator->PrepareAlltoAllAddrInfo(
        opParam.All2AllDataDes.sendCounts, opParam.All2AllDataDes.sdispls, opParam.All2AllDataDes.sendType,
        opParam.All2AllDataDes.recvCounts, opParam.All2AllDataDes.rdispls, opParam.All2AllDataDes.recvType,
        preMetaInfo));

    preMetaInfo->opType = HcclCMDType::HCCL_CMD_ALLGATHER;

    CHK_RET(RegressCalPreOp(alltoAllOperator, opParam, preMetaInfo));

    return alltoAllOperator->GetAlltoAllStagedWorkSpaceMemSize(opParam, memSize);
}

HcclResult HcclCommunicator::GetAlltoAllStagedWorkSpaceMemSize(
    std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo, u64& memSize)
{
    CHK_PRT_RET(
        Is310P3Common(isHaveCpuRank_, deviceType_),
        HCCL_ERROR("[HcclCommunicator][GetAlltoAllStagedWorkSpaceMemSize]Not Supported!"), HCCL_E_NOT_SUPPORT);

    CHK_SMART_PTR_NULL(implAlg_);
    return implAlg_->GetAlltoAllStagedWorkSpaceMemSize(allMeshAggregationSendRecvInfo, memSize);
}

HcclResult
HcclCommunicator::GetAllReduceScratchSize(const u64 count, const HcclDataType dataType, u64& scratchSize) const
{
    CHK_SMART_PTR_NULL(implAlg_);
    return implAlg_->GetAllReduceScratchSize(count, dataType, scratchSize);
}

HcclResult HcclCommunicator::SetWorldGroupInfo(
    std::unordered_map<std::string, std::map<u32, HcclIpAddress>> phyIdNicInfoMap, vector<RankInfo> worldRankInfoList,
    vector<u32>& nicRanksPort, vector<u32>& vnicRanksPort)
{
    for (auto& ipInfo : phyIdNicInfoMap) {
        for (auto& devInfo : ipInfo.second) {
            rankDevicePhyIdNicInfoMap_[ipInfo.first][devInfo.first] = devInfo.second;
            HCCL_DEBUG(
                "phyIdNicInfoMap print hostIp[%s] devId[%u] devIp[%s]", ipInfo.first.c_str(), devInfo.first,
                devInfo.second.GetReadableAddress());
        }
    }

    for (auto& rankInfo : worldRankInfoList) {
        worldRankInfoList_.push_back(rankInfo);
    }

    for (auto& port : nicRanksPort) {
        nicRanksPort_.push_back(port);
        HCCL_DEBUG("nicRanksPort port[%u]", port);
    }
    for (auto& port : vnicRanksPort) {
        vnicRanksPort_.push_back(port);
        HCCL_DEBUG("vnicRanksPort port[%u]", port);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetTopoDesc(HcclTopoDescs* topoDescs, uint32_t topoSize) const
{
    if (topoSize < static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_MAX)) {
        HCCL_ERROR("topoDescs size is not enough, please check topoSize[%u]", topoSize);
        return HCCL_E_PARA;
    }

    if (deviceType_ == DevType::DEV_TYPE_910_93) {
        topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L0)].algSets = HCCL_ALG_SWITCH | HCCL_ALG_RING;
        topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L1)].algSets = HCCL_ALG_RING;
    } else if (deviceType_ == DevType::DEV_TYPE_910B) {
        topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L0)].algSets = HCCL_ALG_MESH;
        topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L1)].algSets = 0;
    } else if (deviceType_ == DevType::DEV_TYPE_310P3) {
        topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L0)].algSets = HCCL_ALG_RING;
        topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L1)].algSets = 0;
    }

    topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L0)].rankSize = userRankSize_;
    topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L1)].rankSize = 0;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetAivModeConfig(const bool aivMode)
{
    CHK_SMART_PTR_NULL(implAlg_);
    CHK_RET(implAlg_->SetAivModeConfig(aivMode));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetOnlyAivModeConfig(const bool isOnlyAiv)
{
    CHK_SMART_PTR_NULL(implAlg_);
    CHK_RET(implAlg_->SetOnlyAivModeConfig(isOnlyAiv));
    isOnlyAiv_ = isOnlyAiv;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetAicpuUnfoldConfig(const bool aicpuUnfold)
{
    CHK_SMART_PTR_NULL(implAlg_);
    CHK_RET(implAlg_->SetAicpuUnfoldConfig(aicpuUnfold));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetExecTimeOutConfig(const s32 execTimeOut)
{
    CHK_SMART_PTR_NULL(implAlg_);
    CHK_RET(implAlg_->SetExecTimeOutConfig(execTimeOut));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetAlgoConfig(const std::map<HcclCMDType, std::vector<HcclAlgoType>>& algoMap)
{
    CHK_SMART_PTR_NULL(implAlg_);
    CHK_RET(implAlg_->SetAlgoConfig(algoMap));
    return HCCL_SUCCESS;
}

bool HcclCommunicator::GetAivModeConfig() { return commConfig_.GetConfigAivMode(); }

bool HcclCommunicator::GetConfigIsOnlyAivMode() { return commConfig_.GetConfigIsOnlyAivMode(); }

bool HcclCommunicator::GetAicpuUnfoldConfig() { return commConfig_.GetConfigAicpuUnfold(); }

void HcclCommunicator::SetQpQosAttr(u32 trafficClass, u32 serviceLevel)
{
    if (oneSideService_) {
        oneSideService_->SetTCAndSL(trafficClass, serviceLevel);
        HCCL_INFO("[%s]Set TC[%u] and SL[%u] for oneSidedService success.", __func__, trafficClass, serviceLevel);
    }
    transportManager_->SetQpQosAttr(trafficClass, serviceLevel);
    indptOpTransportManager_->SetQpQosAttr(trafficClass, serviceLevel);
}

HcclResult HcclCommunicator::CheckExitWaitResumeState(bool& isChangedLink)
{
    if (retryEnable_ && opRetryManager_ != nullptr) {
        bool haveCommEnableBackupLink = false;
        if (g_enableBackupLinkCommCount.load() > 0) {
            haveCommEnableBackupLink = true;
        }
        HcclResult ret = opRetryManager_->ExitWaitResumeState(
            identifier_, commConnections_.isRoot, haveCommEnableBackupLink, isChangedLink);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[HcclCommunicator][Resume]opretry exit wait resume state failed."), ret);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetMemoryRange(void* baseVirPtr, size_t size, size_t alignment, uint64_t flags)
{
    CHK_PRT_RET(
        deviceType_ != DevType::DEV_TYPE_910_93,
        HCCL_ERROR("[HcclCommunicator][SetMemoryRange] deviceType[%d] not support zero copy", deviceType_),
        HCCL_E_NOT_SUPPORT);
    if (zeroCopyMemoryAgent_ == nullptr) {
        CHK_RET(InitZeroCopyMemoryAgent());
    }
    CHK_RET(zeroCopyMemoryAgent_->SetMemoryRange(baseVirPtr, size, alignment, flags));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::UnsetMemoryRange(void* baseVirPtr)
{
    CHK_PRT_RET(
        zeroCopyMemoryAgent_ == nullptr,
        HCCL_ERROR("[HcclCommunicator][UnsetMemoryRange] not call HcclCommSetMemoryRange()"), HCCL_E_PARA);
    CHK_RET(zeroCopyMemoryAgent_->UnsetMemoryRange(baseVirPtr));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ActivateCommMemory(void* virPtr, size_t size, size_t offset, void* handle, uint64_t flags)
{
    CHK_PRT_RET(
        zeroCopyMemoryAgent_ == nullptr,
        HCCL_ERROR("[HcclCommunicator][ActivateCommMemory] not call HcclCommSetMemoryRange()"), HCCL_E_PARA);
    CHK_RET(zeroCopyMemoryAgent_->ActivateCommMemory(virPtr, size, offset, handle, flags));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::DeactivateCommMemory(void* virPtr)
{
    CHK_PRT_RET(
        zeroCopyMemoryAgent_ == nullptr,
        HCCL_ERROR("[HcclCommunicator][DeactivateCommMemory] not call HcclCommSetMemoryRange()"), HCCL_E_PARA);
    CHK_RET(zeroCopyMemoryAgent_->DeactivateCommMemory(virPtr));
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetSingleLinkInfo(
    std::unordered_map<u32, bool>& switchRanks, u32 remoteRankId, ChangeLinkInfo& changeLinkInfo)
{
    auto iterLocal = switchRanks.find(userRank_);
    auto iterRemote = switchRanks.find(remoteRankId);

    bool useBackupLink = false;
    if (iterLocal != switchRanks.end() && iterRemote != switchRanks.end()) {
        // 本端卡和对端卡都切，如果两者的目标网卡冲突，则切换失败；否则使用一致的目标网卡的的对应链路
        CHK_PRT_RET(
            iterLocal->second != iterRemote->second,
            HCCL_ERROR(
                "[HcclCommunicator][SetSingleLinkInfo] local rank[%u] plan to switch to nic[%u], "
                "which is conflict with remote rank[%u] planning to switch to nic[%u].",
                userRank_, iterLocal->second, remoteRankId, iterRemote->second),
            HCCL_E_PARA);
        useBackupLink = iterLocal->second;
    } else if (iterLocal != switchRanks.end()) {
        // 仅切换本端卡，根据本端卡的目标网卡，刷新对应链路
        useBackupLink = iterLocal->second;
    } else if (iterRemote != switchRanks.end()) {
        // 仅切换对端卡，根据对端卡的目标网卡，刷新对应链路
        useBackupLink = iterRemote->second;
    } else {
        HCCL_INFO(
            "[HcclCommunicator][SetSingleLinkInfo] comm identifier[%s], local rank[%u], "
            "remote rank[%u], neither the rank need switch, link will not be refreshed.",
            identifier_.c_str(), userRank_, remoteRankId);
        return HCCL_SUCCESS;
    }

    changeLinkInfo.remoteRankList[changeLinkInfo.remoteRankNum] = remoteRankId;
    changeLinkInfo.isUseDefaultPort[changeLinkInfo.remoteRankNum] = !(useBackupLink);
    changeLinkInfo.remoteRankNum++;
    remoteRankNicStatus_[remoteRankId] = useBackupLink ? CONNECT_REMOTE_BACKUP : CONNECT_REMOTE_DEFAULT;
    needCheckBackupNic_ = needCheckBackupNic_ || useBackupLink;
    needCheckDefaultNic_ = needCheckDefaultNic_ || !useBackupLink;

    HCCL_RUN_INFO(
        "[HcclCommunicator][SetSingleLinkInfo] comm identifier[%s], local rank[%u], "
        "remote rank[%u], useBackupLink[%u], link info refreshed.",
        identifier_.c_str(), userRank_, remoteRankId, useBackupLink);
    return HCCL_SUCCESS;
}

HcclResult
HcclCommunicator::SetRemoteRankLinkInfo(std::unordered_map<u32, bool>& switchRanks, ChangeLinkInfo& changeLinkInfo)
{
    // 初始化重置changeLinkInfo
    changeLinkInfo.remoteRankNum = 0;
    needCheckBackupNic_ = false;
    needCheckDefaultNic_ = false;
    // 初始化重置remoteRankNicStatus_
    (void)memset_s(remoteRankNicStatus_, sizeof(remoteRankNicStatus_), 0, sizeof(remoteRankNicStatus_));

    for (auto resIt : resMap_) {
        for (auto& levelNSubCommTransport : resIt.second.opTransportResponse) {
            for (auto& singleSubCommTransport : levelNSubCommTransport) {
                for (auto& transportRequest : singleSubCommTransport.transportRequests) {
                    if (transportRequest.isValid && transportRequest.isUsedRdma) { // 仅RDMA链路需要刷新
                        CHK_RET(SetSingleLinkInfo(switchRanks, transportRequest.remoteUserRank, changeLinkInfo));
                    }
                }
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ActiveStoppedLink(
    std::map<u32, bool>& remoteRankPortMap, OpCommTransport& opTransportResponse, bool isBackup)
{
    for (auto& levelNSubCommTransport : opTransportResponse) {
        for (auto& singleSubCommTransport : levelNSubCommTransport) {
            if (singleSubCommTransport.status.size() == 0) {
                continue;
            }
            if (singleSubCommTransport.status.size() != singleSubCommTransport.transportRequests.size()
                || singleSubCommTransport.links.size() != singleSubCommTransport.transportRequests.size()) {
                HCCL_ERROR(
                    "[HcclCommunicator][ActiveStoppedLink] comm identifier[%s], local rank[%u], "
                    "status num[%u] or links num[%u] is inconsistent with transport request num[%u]. "
                    "Please check whether the resources are allocated correctly.",
                    identifier_.c_str(), userRank_, singleSubCommTransport.status.size(),
                    singleSubCommTransport.links.size(), singleSubCommTransport.transportRequests.size());
                return HCCL_E_INTERNAL;
            }

            for (size_t i = 0; i < singleSubCommTransport.transportRequests.size(); i++) {
                auto& transportRequest = singleSubCommTransport.transportRequests[i];
                auto remoteRankIter = remoteRankPortMap.find(transportRequest.remoteUserRank);
                bool needLink = transportRequest.isValid && transportRequest.isUsedRdma
                                && remoteRankIter != remoteRankPortMap.end() && (remoteRankIter->second != isBackup);
                // STOP状态的Transport需要唤醒，重置位到READY
                if (needLink && singleSubCommTransport.status[i] == TransportStatus::STOP) {
                    HCCL_INFO(
                        "[HcclCommunicator][ActiveStoppedLink] comm identifier[%s], local rank[%u], "
                        "resuming link of remote rank[%u]",
                        identifier_.c_str(), userRank_, transportRequest.remoteUserRank);
                    CHK_RET(singleSubCommTransport.links[i]->Resume());
                    singleSubCommTransport.status[i] = TransportStatus::READY;
                }
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult
HcclCommunicator::PrepareLinkForSwitchNic(std::unordered_map<u32, bool>& switchRanks, ChangeLinkInfo& changeLinkInfo)
{
    CHK_RET(SetRemoteRankLinkInfo(switchRanks, changeLinkInfo));

    std::map<u32, bool> remoteRankPortMap;
    for (u32 i = 0; i < changeLinkInfo.remoteRankNum; i++) {
        remoteRankPortMap.emplace(changeLinkInfo.remoteRankList[i], changeLinkInfo.isUseDefaultPort[i]);
    }
    for (auto resIt : resMap_) {
        CHK_RET(ActiveStoppedLink(remoteRankPortMap, resIt.second.opTransportResponse, false));
        CHK_RET(ActiveStoppedLink(remoteRankPortMap, resIt.second.opTransportResponseBackUp, true));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ParseSwitchRanks(
    uint32_t nRanks, uint32_t* ranks, bool* useBackup, std::unordered_map<u32, bool>& switchRanks)
{
    CHK_PTR_NULL(ranks);
    CHK_PTR_NULL(useBackup);
    switchRanksNum_ = nRanks;
    (void)memset_s(switchRankList_, sizeof(switchRankList_), 0, sizeof(switchRankList_));
    (void)memset_s(switchUseBackup_, sizeof(switchUseBackup_), 0, sizeof(switchUseBackup_));
    s32 ret = memcpy_s(switchRankList_, sizeof(switchRankList_), ranks, sizeof(u32) * nRanks);
    CHK_PRT_RET(
        ret != EOK, HCCL_ERROR("[HcclCommunicator][ParseSwitchRanks] mem copy switch ranks fail."), HCCL_E_INTERNAL);
    ret = memcpy_s(switchUseBackup_, sizeof(switchUseBackup_), useBackup, sizeof(bool) * nRanks);
    CHK_PRT_RET(
        ret != EOK, HCCL_ERROR("[HcclCommunicator][ParseSwitchRanks] mem copy switch use backup fail."),
        HCCL_E_INTERNAL);

    std::string switchRankStr{};
    for (uint32_t i = 0; i < nRanks; i++) {
        CHK_PTR_NULL(ranks + i);
        CHK_PTR_NULL(useBackup + i);
        uint32_t switchRankId = ranks[i];
        bool backup = useBackup[i];
        CHK_PRT_RET(
            switchRankId >= userRankSize_,
            HCCL_ERROR(
                "[HcclCommunicator][ParseSwitchRanks] invalid switchRankId[%u], "
                "which should not be greater than rankSize[%u]",
                switchRankId, userRankSize_),
            HCCL_E_PARA);
        CHK_PRT_RET(
            switchRanks.find(switchRankId) != switchRanks.end(),
            HCCL_ERROR("[HcclCommunicator][ParseSwitchRanks] duplicated switchRankId[%u]", switchRankId), HCCL_E_PARA);
        switchRanks.emplace(switchRankId, backup);
        switchRankStr += std::to_string(switchRankId) + ":" + std::to_string(backup) + ";";
    }
    HCCL_RUN_INFO(
        "[HcclCommunicator][ParseSwitchRanks] comm identifier[%s], userRank[%u], load switchRanks:%s.",
        identifier_.c_str(), userRank_, switchRankStr.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SwitchNic(
    uint32_t nRanks, uint32_t* ranks, bool* useBackup, std::shared_ptr<HDCommunicate>& controlH2D,
    std::shared_ptr<HDCommunicate>& statusD2H)
{
    HcclResult ret = HCCL_SUCCESS;
    CHK_PRT_RET(
        !IsEnableBackupLink(),
        HCCL_RUN_WARNING(
            "[HcclCommunicator][%s]Backup link is not enabled, "
            "switch nic will not be processed, comm identifier[%s], rank[%u], devType[%u], opretry enable[%u], "
            "backup ip valid[%u], roce enable[%u].",
            __func__, identifier_.c_str(), userRank_, deviceType_,
            GetAicpuUnfoldConfig() && commConfig_.GetConfigInterSuperPodRetryEnable(), !devBackupIpAddr_[0].IsInvalid(),
            IsEnableRoce()),
        HCCL_SUCCESS);
    CHK_PRT_RET(
        resMap_.empty(),
        HCCL_ERROR(
            "[HcclCommunicator][%s] "
            "no collective operation has been executed in this communication[%s] on rank[%u], "
            "which does not support to set working device nic.",
            __func__, identifier_.c_str(), userRank_),
        HCCL_E_PARA);
    std::unordered_map<u32, bool> switchRanks;
    ChangeLinkInfo changeLinkInfo;
    ret = ParseSwitchRanks(nRanks, ranks, useBackup, switchRanks);
    if (ret == HCCL_SUCCESS) {
        ret = PrepareLinkForSwitchNic(switchRanks, changeLinkInfo);
    }
    changeLinkInfo.isChangeLinkFlag
        = ret == HCCL_SUCCESS; // 如果入参校验失败，则无需刷新链路；通知aicpu侧，防止其他卡超时等待

    switchNicWaitingResult_ = false;

    u32 changeLinkInfoStart
        = sizeof(KfcCommand) + sizeof(BackgroundCommand) + sizeof(HcclComSuspendingFlag) + sizeof(HcclOpIdentifier);
    CHK_RET(controlH2D->Put(changeLinkInfoStart, sizeof(ChangeLinkInfo), reinterpret_cast<uint8_t*>(&changeLinkInfo)));

    KfcCommand switchNicCommand = KfcCommand::kSwitchNic;
    CHK_RET(controlH2D->Put(0, sizeof(KfcCommand), reinterpret_cast<uint8_t*>(&switchNicCommand)));

    KfcExecStatus switchStatus;
    switchStatus.execStatus.kfcStatus = KfcStatus::kNull;
    u32 waitSwitchExecCmdTimeout = static_cast<u32>(GetExternalInputHcclLinkTimeOut() * 1000 * 2.5f);
    auto waitSwitchExecCmdTimeoutMs
        = std::chrono::milliseconds(waitSwitchExecCmdTimeout); // 等待2.5倍的建链超时时间，给快慢卡场景提供冗余
    auto startTime = std::chrono::steady_clock::now();
    while (true) {
        if (switchNicWaitingResult_) {
            CHK_RET(statusD2H->Get(0, sizeof(KfcExecStatus), reinterpret_cast<uint8_t*>(&switchStatus)));
        }
        if (switchStatus.execStatus.kfcStatus == KfcStatus::kSwitchSuccess) {
            HCCL_INFO(
                "[HcclCommunicator][%s] comm identifier[%s], devicePhyId[%u], userRank[%u] switch nic success.",
                __func__, identifier_.c_str(), devicePhyId_, userRank_);
            ret = HCCL_SUCCESS;
            break;
        } else if (switchStatus.execStatus.kfcStatus == KfcStatus::kSwitchFail) {
            HCCL_ERROR(
                "[HcclCommunicator][%s] comm identifier[%s], devicePhyId[%u], userRank[%u] switch nic fail.", __func__,
                identifier_.c_str(), devicePhyId_, userRank_);
            ret = HCCL_E_INTERNAL;
            break;
        } else if ((std::chrono::steady_clock::now() - startTime) >= waitSwitchExecCmdTimeoutMs) {
            HCCL_ERROR(
                "[HcclCommunicator][%s] comm identifier[%s], devicePhyId[%u], "
                "userRank[%u] switch nic timeout[%u ms], the transport status is undefined. "
                "Please search log with keyword [ErrToWarn] for detail.",
                __func__, identifier_.c_str(), devicePhyId_, userRank_, waitSwitchExecCmdTimeout);
            ret = HCCL_E_TIMEOUT;
            break;
        } else {
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        }
    }
    KfcExecControl clearCommand{};
    CHK_RET(controlH2D->Put(0, sizeof(KfcExecControl), reinterpret_cast<uint8_t*>(&clearCommand)));
    KfcExecStatus clearStatus{};
    CHK_RET(controlH2D->Put(0, sizeof(KfcExecStatus), reinterpret_cast<uint8_t*>(&clearStatus)));
    switchRanksNum_ = 0;
    return ret;
}

HcclResult HcclCommunicator::GetSwitchRanks(
    u32* distSwitchRankList, bool* distSwitchUseBackup, u32& distSwitchRankNum, u8* distRemoteRankNicStatus,
    u32& distNicStatusNum, bool& needCheckDefaultNic, bool& needCheckBackupNic)
{
    s32 ret = memcpy_s(
        distSwitchRankList, sizeof(u32) * AICPU_MAX_RANK_NUM, switchRankList_, sizeof(u32) * switchRanksNum_);
    CHK_PRT_RET(
        ret != EOK, HCCL_ERROR("[HcclCommunicator][GetSwitchRanks] mem copy switch rank list fail, ret[%u].", ret),
        HCCL_E_INTERNAL);
    ret = memcpy_s(
        distSwitchUseBackup, sizeof(bool) * AICPU_MAX_RANK_NUM, switchUseBackup_, sizeof(bool) * switchRanksNum_);
    CHK_PRT_RET(
        ret != EOK, HCCL_ERROR("[HcclCommunicator][GetSwitchRanks] mem copy switch use backup fail, ret[%u].", ret),
        HCCL_E_INTERNAL);
    distSwitchRankNum = switchRanksNum_;
    ret = memcpy_s(
        distRemoteRankNicStatus, sizeof(u8) * AICPU_MAX_RANK_NUM, remoteRankNicStatus_, sizeof(u8) * userRankSize_);
    CHK_PRT_RET(
        ret != EOK,
        HCCL_ERROR(
            "[HcclCommunicator][GetSwitchRanks] mem copy remote rank nic status fail, "
            "ret[%u].",
            ret),
        HCCL_E_INTERNAL);
    distNicStatusNum = userRankSize_;
    needCheckDefaultNic = needCheckDefaultNic_;
    needCheckBackupNic = needCheckBackupNic_;
    switchNicWaitingResult_ = true;
    return HCCL_SUCCESS;
}

HcclResult GetCannPath(const char* binPath, std::string& cannPath)
{
    CHK_PRT_RET(binPath == nullptr, HCCL_ERROR("[HcclCommunicator][GetCannPath]binary path is nullptr"), HCCL_E_PTR);

    std::string tmpPath(binPath); // 存放cann安装路径
    std::string libraryPath;
    HcclResult ret = ParseLibraryPath(libraryPath);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[GetCannPath]errNo[0x%016llx]parse path fail.", ret), ret);

    ret = GetKeyWordPath(libraryPath, "/hccl", tmpPath);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[GetCannPath]cannot find the version file in %s.", libraryPath.c_str()),
        HCCL_E_PARA);
    tmpPath += binPath;
    cannPath = tmpPath;

    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::LoadCustomFile(
    const char* binPath, aclrtBinaryLoadOptionType optionType, uint32_t cpuKernelMode, aclrtBinHandle& binHandle)
{
    binHandle = nullptr;
    HCCL_INFO("[LoadCustomFile]binPath[%s], optionType[%u], cpuKernelMode[%u]", binPath, optionType, cpuKernelMode);
    return HCCL_SUCCESS;
}

void HcclCommunicator::UnloadBinary(aclrtBinHandle& binHandle)
{
    if (binHandle != nullptr) {
        aclError ret = aclrtBinaryUnLoad(binHandle);
        if (ret != ACL_SUCCESS) {
            HCCL_ERROR("[UnloadBinary]errNo[0x%016llx] unload binary from file error.", ret);
        }
        binHandle = nullptr;
    }
    return;
}

HcclResult HcclCommunicator::RegisterCommUserMem(void* addr, u64 size, void** handle)
{
    // user mem和ccl buffer互斥，不支持同时创建
    if (deviceType_ != DevType::DEV_TYPE_910_93 || superPodNum_ > 1 || isUserMemRegisted_
        || cclBufferManager_.GetInCCLbuffer().ptr() != nullptr) {
        HCCL_ERROR(
            "[HcclCommunicator][%s]Registration user mem is not supported with the params. "
            "Device type[%d], superPodNum[%u]; Or user mem/CCL buffer has already registered, addr[%p], "
            "isUserMemRegistered[%d]",
            __func__, deviceType_, superPodNum_, addr, isUserMemRegisted_);
        return HCCL_E_NOT_SUPPORT;
    }
    // DeviceMem::create创建的DeviceMem对象为拷贝构造，析构时不释放内存，内存由上层管理
    DeviceMem userMem = DeviceMem::create(addr, size);
    std::shared_ptr<DeviceMem> userMemPtr = nullptr;
    EXCEPTION_CATCH((userMemPtr = std::make_shared<DeviceMem>(std::move(userMem))), return HCCL_E_PTR);
    *handle = static_cast<void*>(userMemPtr.get());
    userMemMap_.insert(std::make_pair(*handle, userMemPtr));
    HCCL_INFO(
        "[HcclCommunicator][%s]Register user mem success, group[%s], handle[%p], addr[%llu], size[%llu]", __func__,
        identifier_.c_str(), *handle, reinterpret_cast<uint64_t>(addr), size);
    isUserMemRegisted_ = true;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::DeregisterCommUserMem(void* handle)
{
    if (deviceType_ != DevType::DEV_TYPE_910_93 || superPodNum_ > 1) {
        HCCL_ERROR(
            "[HcclCommunicator][%s]Unsupported on the device type[%d] or superPodNum[%u]", __func__, deviceType_,
            superPodNum_);
        return HCCL_E_NOT_SUPPORT;
    }

    CHK_PRT_RET(
        !userMemMap_.erase(handle),
        HCCL_RUN_WARNING("[HcclCommunicator][%s]Mem is not exist, handle[%p]", __func__, handle), HCCL_SUCCESS);

    // 重置user mem和userMemType
    CHK_SAFETY_FUNC_RET(
        memset_s(opResPara_.userMemRes, sizeof(opResPara_.userMemRes), 0, sizeof(opResPara_.userMemRes)));
    opResPara_.userMemType = 0; // CCL Buffer
    isUserMemRegisted_ = false;
    HCCL_INFO(
        "[HcclCommunicator][%s]Deregister mem success, group[%s], handle[%p]", __func__, identifier_.c_str(), handle);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::ExchangeCommUserMem(void* handle, std::vector<u32>& peerRanks)
{
    if (deviceType_ != DevType::DEV_TYPE_910_93 || superPodNum_ > 1 || GetExternalInputInterHccsDisable()) {
        HCCL_ERROR(
            "[HcclCommunicator][%s]Unsupported configuration: device type[%d], superPodNum[%u], "
            "or RDMA usage",
            __func__, deviceType_, superPodNum_);
        return HCCL_E_NOT_SUPPORT;
    }

    if ((peerRanks.size() > rankInfoList_.size())) {
        HCCL_ERROR(
            "[HcclCommunicator][%s]Invalid peerRanksNum[%u], which should be less than communicator "
            "rank nums[%u]",
            __func__, peerRanks.size(), rankInfoList_.size());
        return HCCL_E_PARA;
    }
    // 获取user mem，调exchange接口前需要先调注册接口注册user mem
    if (userMemMap_.find(handle) == userMemMap_.end()) {
        HCCL_ERROR("[HcclCommunicator][%s]Find user mem failed, handle[%p] is not registered", __func__, handle);
        return HCCL_E_NOT_FOUND;
    }
    DeviceMem userMem = *userMemMap_[handle].get();
    CHK_PTR_NULL(userMem.ptr());
    // 构造建链param
    TransportIOMem transMem;
    transMem.userMem = userMem;
    OpCommTransport opCommTransport;
    LevelNSubCommTransport level0Transport;
    SingleSubCommTransport commTransport;

    for (u32 rankIdx = 0; rankIdx < peerRanks.size(); rankIdx++) {
        TransportRequest tmpTransport;
        if (userRank_ != peerRanks[rankIdx]) {
            tmpTransport.isValid = true;
            tmpTransport.localUserRank = userRank_;
            tmpTransport.remoteUserRank = peerRanks[rankIdx];
            tmpTransport.inputMemType = TransportMemType::USER_MEM;
            tmpTransport.outputMemType = TransportMemType::USER_MEM;
        } else {
            // 本rank不需要创建transport
            tmpTransport.isValid = false;
        }
        commTransport.transportRequests.push_back(tmpTransport);
    }
    level0Transport.push_back(commTransport);
    opCommTransport.push_back(level0Transport);
    ClearOpTransportResponseLinks(opCommTransport);
    // 建链
    constexpr char EXCHANGE_USER_MEM_TAG_PREFIX[] = "ExchangeUserMem_";
    string tag = EXCHANGE_USER_MEM_TAG_PREFIX + identifier_;
    HcclResult ret = HCCL_SUCCESS;
    {
        StateGuard<HcclCommunicator, HcclCommState> guard(this, HcclCommState::BUILDING);
        HCCL_RUN_INFO(
            "[%s]Alloc transport, level size[%u], trans request size[%u], mem ptr[%p], mem size[%llu]", __func__,
            opCommTransport.size(), commTransport.transportRequests.size(), userMem.ptr(), userMem.size());
        CHK_PTR_NULL(transportManager_);
        ret = transportManager_->Alloc(tag, transMem, opCommTransport, false);
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[%s]Alloc transports failed, tag[%s]", __func__, tag.c_str()), ret);
    userMemTransport_ = opCommTransport;
    return HCCL_SUCCESS;
}
HcclResult HcclCommunicator::GetLocalCCLBuf(void** addr, uint64_t* size)
{
    uint64_t cclbufSize = cclBufferManager_.GetInCCLbufferSize() + cclBufferManager_.GetOutCCLbufferSize()
                          + cclBufferManager_.GetExpBufferSize();
    *addr = cclBufferManager_.GetCommCCLBuffer().ptr();
    if (nullptr == cclBufferManager_.GetCommCCLBuffer().ptr()) {
        cclbufSize = 0;
    }
    *size = cclbufSize;
    HCCL_INFO(
        "[%s] GetlocalCCLBuf success, addr[%p], size[%u]", identifier_.c_str(),
        cclBufferManager_.GetCommCCLBuffer().ptr(), cclbufSize);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetRemoteCCLBuf(uint32_t remoteRank, void** addr, uint64_t* size)
{
    CHK_PRT_RET(
        (remoteRank >= AICPU_MAX_RANK_NUM), HCCL_ERROR("[%s] invalid remoteRank[%d]", __func__, remoteRank),
        HCCL_E_PARA);
    // 仅sdma场景
    uint64_t cclbufSize = cclBufferManager_.GetInCCLbufferSize() + cclBufferManager_.GetOutCCLbufferSize()
                          + cclBufferManager_.GetExpBufferSize();
    *addr = p2pCclBuf_[remoteRank];

    if (nullptr == p2pCclBuf_[remoteRank]) {
        cclbufSize = 0;
    }
    *size = cclbufSize;
    HCCL_INFO(
        "[%s] GetRemoteCCLBuf success, remoteRank[%u], addr[%p], size[%u]", identifier_.c_str(), remoteRank,
        p2pCclBuf_[remoteRank], cclbufSize);
    return HCCL_SUCCESS;
}
HcclResult HcclCommunicator::GetKFCWorkSpace(void** addr, uint64_t* size)
{
    *addr = workSpace_.ptr();
    *size = workSpaceSize_;
    HCCL_INFO(
        "[%s] GetKFCWorkSpace success, addr[%p], size[%u]", identifier_.c_str(), workSpace_.ptr(), workSpaceSize_);
    return HCCL_SUCCESS;
}
HcclResult HcclCommunicator::IndOpTransportAlloc(
    const std::string& tag, OpCommTransport& opCommTransport, TransportIOMem& transMem, bool isAicpuModeEn)
{
    // Aicpu侧不支持用户注册额外内存
    if (isAicpuModeEn) {
        if (transMem.indOpMem.userDeviceMem.size() > 0 || transMem.indOpMem.userHostMem.size() > 0) {
            HCCL_ERROR("[%s] AICPU engine does not support user-registered memory", __func__);
            return HCCL_E_NOT_SUPPORT;
        }
    }

    StateGuard<HcclCommunicator, HcclCommState> guard(this, HcclCommState::BUILDING);
    CHK_PTR_NULL(indptOpTransportManager_);
    bool isIndOp = true;
    HcclResult ret = indptOpTransportManager_->Alloc(
        tag, transMem, opCommTransport, isAicpuModeEn, false, false, HcclCMDType::HCCL_CMD_INVALID, false, isIndOp);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[%s] Failed to alloc transport, tag[%s], isAicpuModeEn[%d], ret[%d]", __func__, tag.c_str(), isAicpuModeEn,
            ret);
        return ret;
    }

    HCCL_RUN_INFO(
        "[%s] Alloc transport success, tag[%s], isAicpuModeEn[%d], ret[%d]", __func__, tag.c_str(), isAicpuModeEn, ret);
    return HCCL_SUCCESS;
}

HcclTopoAttr HcclCommunicator::GetTopoAttr()
{
    HcclTopoAttr topoAttr;
    attrCollector_.GetTopoAttr(topoAttr);
    return topoAttr;
}

HcclResult HcclCommunicator::GetHDCommunicate(
    HDCommunicateParams& kfcControlTransferH2DParams, HDCommunicateParams& kfcStatusTransferD2HParams)
{
    if (GetSupportHDCommunicate() == false) {
        HCCL_WARNING("%s not support HDCommunicate, skip", __func__);
        return HCCL_SUCCESS;
    }
    CHK_SMART_PTR_NULL(kfcControlTransferH2D_);
    CHK_SMART_PTR_NULL(kfcStatusTransferD2H_);
    kfcControlTransferH2DParams = kfcControlTransferH2D_->GetCommunicateParams();
    kfcStatusTransferD2HParams = kfcStatusTransferD2H_->GetCommunicateParams();
    HCCL_INFO("%s success, group[%s]", __func__, identifier_.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetGetAicpuCommState(std::function<bool()> getAicpuCommState)
{
    getAicpuCommState_ = getAicpuCommState;
    HCCL_DEBUG("%s success, group[%s]", __func__, identifier_.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CommGetNetLayers(uint32_t** netLayers, uint32_t* netLayerNum)
{
    if (deviceType_ == DevType::DEV_TYPE_910_93) {
        netLayer_[0] = static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L0);
        netLayer_[1] = static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L1);
        *netLayerNum = COMM_LAYER_NUM_MAX;
    } else if (deviceType_ == DevType::DEV_TYPE_910B || deviceType_ == DevType::DEV_TYPE_310P3) {
        netLayer_[0] = static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L0);
        *netLayerNum = 1;
    }
    *netLayers = netLayer_;
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CommGetInstSizeByNetLayer(uint32_t netLayer, uint32_t* rankNum) const
{
    if ((netLayer == static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L0))
        || (netLayer == static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L1))) {
        *rankNum = userRankSize_;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::CommGetInstTopoTypeByNetLayer(uint32_t netLayer, u32* topoType) const
{
    if (deviceType_ == DevType::DEV_TYPE_910_93) {
        if (netLayer == static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L0)) {
            *topoType = HCCL_ALG_SWITCH | HCCL_ALG_RING;
        } else if (netLayer == static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L1)) {
            *topoType = HCCL_ALG_RING;
        }
    } else if (deviceType_ == DevType::DEV_TYPE_910B) {
        if (netLayer == static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L0)) {
            *topoType = HCCL_ALG_MESH;
        }
    } else if (deviceType_ == DevType::DEV_TYPE_310P3) {
        if (netLayer == static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L0)) {
            *topoType = HCCL_ALG_RING;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::GetNetLayers(uint32_t** netLayers, uint32_t* netLayerNum)
{
    return rankGraph_.GetNetLayers(netLayers, netLayerNum);
}

HcclResult HcclCommunicator::GetInstSizeByNetLayer(uint32_t netLayer, uint32_t* rankNum)
{
    return rankGraph_.GetInstSizeByNetLayer(netLayer, rankNum);
}

HcclResult HcclCommunicator::GetInstTopoTypeByNetLayer(uint32_t netLayer, CommTopo* topoType)
{
    return rankGraph_.GetInstTopoTypeByNetLayer(netLayer, topoType);
}

HcclResult HcclCommunicator::GetInstRanksByNetLayer(uint32_t netLayer, uint32_t** rankList, uint32_t* rankNum)
{
    return rankGraph_.GetInstRanksByNetLayer(netLayer, rankList, rankNum);
}

HcclResult HcclCommunicator::GetInstSizeListByNetLayer(uint32_t netLayer, uint32_t** instSizeList, uint32_t* listSize)
{
    return rankGraph_.GetInstSizeListByNetLayer(netLayer, instSizeList, listSize);
}

HcclResult HcclCommunicator::GetTopoInstsByLayer(uint32_t netLayer, uint32_t** topoInsts, uint32_t* topoInstNum)
{
    return rankGraph_.GetTopoInstsByLayer(netLayer, topoInsts, topoInstNum);
}

HcclResult HcclCommunicator::GetTopoType(uint32_t netLayer, [[maybe_unused]] uint32_t topoInstId, CommTopo* topoType)
{
    return rankGraph_.GetTopoType(netLayer, topoType);
}

HcclResult
HcclCommunicator::GetRanksByTopoInst(uint32_t netLayer, uint32_t topoInstId, uint32_t** ranks, uint32_t* rankNum)
{
    return rankGraph_.GetRanksByTopoInst(netLayer, topoInstId, ranks, rankNum);
}

HcclResult HcclCommunicator::GetEndpointNum(uint32_t netLayer, uint32_t topoInstId, uint32_t* num)
{
    return rankGraph_.GetEndpointNum(netLayer, topoInstId, num);
}

HcclResult
HcclCommunicator::GetEndpointDesc(uint32_t netLayer, uint32_t topoInstId, uint32_t* descNum, EndpointDesc* endpointDesc)
{
    return rankGraph_.GetEndpointDesc(netLayer, topoInstId, descNum, endpointDesc);
}

HcclResult HcclCommunicator::GetEndpointInfo(
    uint32_t rankId, const EndpointDesc* endPointDesc, EndpointAttr endpointAttr, uint32_t infoLen, void* info)
{
    return rankGraph_.GetEndpointInfo(rankId, endPointDesc, endpointAttr, infoLen, info);
}

HcclResult HcclCommunicator::GetRankGraph(GraphType type, void** graph, uint32_t* len)
{
    return rankGraph_.GetRankGraphInfo(type, graph, len);
}

void* HcclCommunicator::GetRankGraphV1() { return static_cast<void*>(&rankGraph_); }

HcclResult HcclCommunicator::GetLinks(
    uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink** linkList, uint32_t* listSize)
{
    return rankGraph_.GetLinks(netLayer, srcRank, dstRank, linkList, listSize);
}

HcclResult HcclCommunicator::GetHeterogMode(HcclHeterogMode* mode) { return rankGraph_.GetHeterogMode(mode); }

HcclResult HcclCommunicator::RegisterToSnapshot()
{
    if (deviceType_ != DevType::DEV_TYPE_910B && deviceType_ != DevType::DEV_TYPE_910_93) {
        return HCCL_SUCCESS;
    }
    if (userRankSize_ <= 1) {
        HCCL_RUN_INFO(
            "[HcclCommunicator][RegisterToSnapshot]comm identifier[%s], deviceLogicId[%d], "
            "rank size[%u] is no greater than 1, and then will not register to snapshot",
            identifier_.c_str(), deviceLogicId_, userRankSize_);
        return HCCL_SUCCESS;
    }
    auto setInvalidCommCallback = [this](bool isInvalid) {
        return this->SetInvalidComm(isInvalid);
    };
    auto preProcessCallback = [this]() {
        return this->SnapshotCheckPreProcess();
    };
    auto postProcessCallback = [this]() {
        return this->SnapshotCheckPostProcess();
    };
    CHK_RET(SnapshotControl::GetInstance(deviceLogicId_)
                .RegisterComm(identifier_, setInvalidCommCallback, preProcessCallback, postProcessCallback));
    if (IsEnableBackupLink()) {
        CHK_RET(SnapshotControl::GetInstance(deviceLogicId_).RegisterBackup(identifier_, deviceBackUpPhyId_));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::UnRegisterFromSnapshot()
{
    if (deviceType_ != DevType::DEV_TYPE_910B && deviceType_ != DevType::DEV_TYPE_910_93) {
        return HCCL_SUCCESS;
    }
    if (userRankSize_ <= 1) {
        HCCL_RUN_INFO(
            "[HcclCommunicator][UnRegisterFromSnapshot]comm identifier[%s], deviceLogicId[%d], "
            "rank size[%u] is no greater than 1, and then will not unregister from snapshot",
            identifier_.c_str(), deviceLogicId_, userRankSize_);
        return HCCL_SUCCESS;
    }
    CHK_RET(SnapshotControl::GetInstance(deviceLogicId_).UnRegisterComm(identifier_));
    if (IsEnableBackupLink()) {
        CHK_RET(SnapshotControl::GetInstance(deviceLogicId_).UnRegisterBackup(identifier_, deviceBackUpPhyId_));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SetInvalidComm(bool isInvalid)
{
    isInvalidComm_ = isInvalid;
    HCCL_INFO(
        "[HcclCommunicator][SetInvalidComm] comm[%s] is set to invalid, rank[%u], deviceLogicId[%d]",
        identifier_.c_str(), userRank_, deviceLogicId_);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SnapshotCheckPreProcess()
{
    bool errorFlag = false;
    auto pauseTimeout = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());
    auto startTime = std::chrono::steady_clock::now();
    while (true) {
        CHK_PRT_BREAK(
            Heartbeat::GetInstance(deviceLogicId_).IsPaused(),
            HCCL_INFO(
                "[HcclCommunicator][SnapshotCheckPreProcess] comm[%s], rank[%u], deviceLogicId[%d], "
                "heartbeat thread has been paused.",
                identifier_.c_str(), userRank_, deviceLogicId_), );
        CHK_PRT_BREAK(
            (std::chrono::steady_clock::now() - startTime) >= pauseTimeout,
            HCCL_ERROR(
                "[HcclCommunicator][SnapshotCheckPreProcess] comm[%s], rank[%u], deviceLogicId[%d], "
                "pause heartbeat thread timeout[%u s].",
                identifier_.c_str(), userRank_, deviceLogicId_, GetExternalInputHcclLinkTimeOut()),
            errorFlag = true);
    }
    startTime = std::chrono::steady_clock::now();
    while (retryEnable_ && opRetryManager_) {
        CHK_PRT_BREAK(
            opRetryManager_->IsPaused(identifier_),
            HCCL_INFO(
                "[HcclCommunicator][SnapshotCheckPreProcess] comm[%s], rank[%u], deviceLogicId[%d], "
                "opretry threads have been paused.",
                identifier_.c_str(), userRank_, deviceLogicId_), );
        CHK_PRT_BREAK(
            (std::chrono::steady_clock::now() - startTime) >= pauseTimeout,
            HCCL_ERROR(
                "[HcclCommunicator][SnapshotCheckPreProcess] comm[%s], rank[%u], deviceLogicId[%d], "
                "pause opretry threads timeout[%u s].",
                identifier_.c_str(), userRank_, deviceLogicId_, GetExternalInputHcclLinkTimeOut()),
            errorFlag = true);
    }
    startTime = std::chrono::steady_clock::now();
    while (zeroCopyMemoryAgent_) {
        CHK_PRT_BREAK(
            zeroCopyMemoryAgent_->IsPaused(),
            HCCL_INFO(
                "[HcclCommunicator][SnapshotCheckPreProcess] comm[%s], rank[%u], deviceLogicId[%d], "
                "zero-copy memory agent thread has been paused.",
                identifier_.c_str(), userRank_, deviceLogicId_), );
        CHK_PRT_BREAK(
            (std::chrono::steady_clock::now() - startTime) >= pauseTimeout,
            HCCL_ERROR(
                "[HcclCommunicator][SnapshotCheckPreProcess] comm[%s], rank[%u], deviceLogicId[%d], "
                "pause zero-copy memory agent thread timeout[%u s].",
                identifier_.c_str(), userRank_, deviceLogicId_, GetExternalInputHcclLinkTimeOut()),
            errorFlag = true);
    }
    CHK_PRT_RET(
        errorFlag,
        HCCL_ERROR(
            "[HcclCommunicator][SnapshotCheckPreProcess] comm[%s], rank[%u], "
            "deviceLogicId[%d], snapshot pre-process fail due to some background threads pause timeout, please check.",
            identifier_.c_str(), userRank_, deviceLogicId_),
        HCCL_E_INTERNAL);
    HCCL_INFO(
        "[HcclCommunicator][SnapshotCheckPreProcess] comm[%s], rank[%u], deviceLogicId[%d], "
        "snapshot pre-process success.",
        identifier_.c_str(), userRank_, deviceLogicId_);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::SnapshotCheckPostProcess()
{
    bool errorFlag = false;
    auto resumeTimeout = std::chrono::seconds(GetExternalInputHcclLinkTimeOut());
    auto startTime = std::chrono::steady_clock::now();
    while (true) {
        CHK_PRT_BREAK(
            Heartbeat::GetInstance(deviceLogicId_).IsResumed(),
            HCCL_INFO(
                "[HcclCommunicator][SnapshotCheckPostProcess] comm[%s], rank[%u], deviceLogicId[%d], "
                "heartbeat thread has been resumed.",
                identifier_.c_str(), userRank_, deviceLogicId_), );
        CHK_PRT_BREAK(
            (std::chrono::steady_clock::now() - startTime) >= resumeTimeout,
            HCCL_ERROR(
                "[HcclCommunicator][SnapshotCheckPostProcess] comm[%s], rank[%u], deviceLogicId[%d], "
                "resume heartbeat thread timeout[%u s].",
                identifier_.c_str(), userRank_, deviceLogicId_, GetExternalInputHcclLinkTimeOut()),
            errorFlag = true);
    }
    startTime = std::chrono::steady_clock::now();
    while (retryEnable_ && opRetryManager_) {
        CHK_PRT_BREAK(
            opRetryManager_->IsResumed(identifier_),
            HCCL_INFO(
                "[HcclCommunicator][SnapshotCheckPostProcess] comm[%s], rank[%u], deviceLogicId[%d], "
                "opretry threads have been resumed.",
                identifier_.c_str(), userRank_, deviceLogicId_), );
        CHK_PRT_BREAK(
            (std::chrono::steady_clock::now() - startTime) >= resumeTimeout,
            HCCL_ERROR(
                "[HcclCommunicator][SnapshotCheckPostProcess] comm[%s], rank[%u], deviceLogicId[%d], "
                "resume opretry threads timeout[%u s].",
                identifier_.c_str(), userRank_, deviceLogicId_, GetExternalInputHcclLinkTimeOut()),
            errorFlag = true);
    }
    startTime = std::chrono::steady_clock::now();
    while (zeroCopyMemoryAgent_) {
        CHK_PRT_BREAK(
            zeroCopyMemoryAgent_->IsResumed(),
            HCCL_INFO(
                "[HcclCommunicator][SnapshotCheckPostProcess] comm[%s], rank[%u], deviceLogicId[%d], "
                "zero-copy memory agent thread has been resumed.",
                identifier_.c_str(), userRank_, deviceLogicId_), );
        CHK_PRT_BREAK(
            (std::chrono::steady_clock::now() - startTime) >= resumeTimeout,
            HCCL_ERROR(
                "[HcclCommunicator][SnapshotCheckPostProcess] comm[%s], rank[%u], deviceLogicId[%d], "
                "resume zero-copy memory agent thread timeout[%u s].",
                identifier_.c_str(), userRank_, deviceLogicId_, GetExternalInputHcclLinkTimeOut()),
            errorFlag = true);
    }
    CHK_PRT_RET(
        errorFlag,
        HCCL_ERROR(
            "[HcclCommunicator][SnapshotCheckPostProcess] comm[%s], rank[%u], "
            "deviceLogicId[%d], snapshot post-process check fail due to some background threads resume timeout, "
            "please check.",
            identifier_.c_str(), userRank_, deviceLogicId_),
        HCCL_E_INTERNAL);
    HCCL_INFO(
        "[HcclCommunicator][SnapshotCheckPostProcess] comm[%s], rank[%u], deviceLogicId[%d], "
        "snapshot post-process check success.",
        identifier_.c_str(), userRank_, deviceLogicId_);
    return HCCL_SUCCESS;
}

void HcclCommunicator::SetReleaseChannel(std::function<HcclResult()> releaseChannel)
{
    releaseChannel_ = releaseChannel;
    return;
}

CCLBufferManager& HcclCommunicator::GetCCLbufferManager() { return cclBufferManager_; }

void HcclCommunicator::SetHcclQos(u32 hcclQos)
{
    HCCL_INFO("[HcclCommunicator][host][SetHcclQos] hcclQos[%u]", hcclQos);
    hcclQos_ = hcclQos;
}

u32 HcclCommunicator::GetHcclQos() const
{
    HCCL_INFO("[HcclCommunicator][host][GetHcclQos] hcclQos[%u]", hcclQos_);
    return hcclQos_;
}

HcclResult HcclCommunicator::InitSymmetricMemory()
{
    if (superPodNum_ > 1) {
        HCCL_DEBUG("[InitSymmetricMemory] Cross-SuperNode not support symmetric memory");
        return HCCL_SUCCESS;
    }
    if (deviceType_ != DevType::DEV_TYPE_910_93) {
        HCCL_DEBUG("[%s] deviceType:%d not support symmetric memory", __func__, deviceType_);
        return HCCL_SUCCESS;
    }

    u64 stride = commConfig_.GetConfigSymmetricMemoryStride() * GIGABYTE_TO_BYTE;
    HCCL_RUN_INFO(
        "InitSymmetricMemory, comm identifier[%s], userRank[%u], userRankSize[%u], stride[%llu], devicePhyId[%u].",
        identifier_.c_str(), realUserRank_, userRankSize_, stride, devicePhyId_);

    symmetricMemoryAgent_ = std::make_shared<SymmetricMemoryAgent>(
        socketManager_, devicePhyId_, deviceLogicId_, localVnicIp_, rankInfoList_, realUserRank_, useSuperPodMode_,
        identifier_);
    CHK_SMART_PTR_NULL(symmetricMemoryAgent_);

    symmetricMemory_ = std::make_unique<SymmetricMemory>(realUserRank_, userRankSize_, stride, symmetricMemoryAgent_);
    CHK_SMART_PTR_NULL(symmetricMemory_);
    return HCCL_SUCCESS;
}

HcclResult HcclCommunicator::RegisterWindow(void* ptr, size_t size, HcclCommSymWindow* winHandle)
{
    CHK_PRT_RET(
        superPodNum_ > 1, HCCL_ERROR("[RegisterWindow] Cross-SuperNode not support symmetric memory"),
        HCCL_E_NOT_SUPPORT);

    CHK_PRT_RET(
        deviceType_ != DevType::DEV_TYPE_910_93,
        HCCL_ERROR("[%s] deviceType:%d not support symmetric memory", __func__, deviceType_), HCCL_E_NOT_SUPPORT);

    CHK_SMART_PTR_NULL(symmetricMemory_);
    return symmetricMemory_->RegisterSymmetricMem(ptr, size, winHandle);
}

HcclResult HcclCommunicator::DeregisterWindow(HcclCommSymWindow winHandle)
{
    CHK_SMART_PTR_NULL(symmetricMemory_);
    return symmetricMemory_->DeregisterSymmetricMem(winHandle);
}

HcclResult HcclCommunicator::GetCommSymWin(void* ptr, size_t size, HcclCommSymWindow* winHandle, size_t* offset)
{
    CHK_SMART_PTR_NULL(symmetricMemory_);
    return symmetricMemory_->FindSymmetricWindow(ptr, size, winHandle, reinterpret_cast<u64*>(offset));
}

bool HcclCommunicator::EnableAicpuUnfold(bool isCapture)
{
    if (deviceType_ != DevType::DEV_TYPE_910_93 && deviceType_ != DevType::DEV_TYPE_910B) {
        return false;
    }
    // 910B在acl graph场景(isCapture)不启用aicpu展开
    if (deviceType_ == DevType::DEV_TYPE_910B && isCapture) {
        HCCL_INFO(
            "[%s] deviceType[%d] isCapture[1], aicpuUnfoldConfig[%u] 910B does not support aicpuUnfold in acl graph "
            "mode",
            __func__, deviceType_, GetAicpuUnfoldConfig());
        return false;
    }
    HCCL_INFO("[%s] aicpuUnfoldConfig[%u]", __func__, GetAicpuUnfoldConfig());
    return GetAicpuUnfoldConfig();
}

aclrtBinHandle HcclCommunicator::GetBinHandle()
{
    if (binHandle_ == nullptr) {
        HCCL_ERROR("[HcclCommunicator][GetBinHandle] GetBinHandle binHandle failed. binHandle is nullptr");
        return nullptr;
    }
    return binHandle_;
}
HcclResult
HcclCommunicator::GetDevMemWorkSpace(const std::string& memTag, uint64_t* size, void** addr, bool* newCreated)
{
    return dpuManager_->GetDevMemWorkSpace(memTag, size, addr, newCreated);
}
} // namespace hccl

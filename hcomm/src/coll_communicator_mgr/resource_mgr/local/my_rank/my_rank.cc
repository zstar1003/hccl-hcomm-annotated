/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "my_rank.h"
#include "hcomm_c_adpt.h"
#include "endpoint_pair.h"
#include "hccl_res.h"
#include "../common/loggers/channel_logger.h" // 日志记录器
#include "hcclCommDfx.h"
#include "config/env_config.h"
#include "env_config/env_config_v2.h"
#include "channel_process.h"
#include "ccu_dev_mgr_imp.h"
#include "ccu_device_res.h"
#include "ccu_res_desc.h"
#include "ccu_device_pub.h"
#include "ccu_res_desc_mgr.h"
#include "ccu_log.h"
#include "dfx_dlprof_function.h"
#include "config_log.h"
#include "comm_engine_utils.h"
#include "hcom_common.h"
#include "op_base.h"
#include "ccu_res.h"
#include "coll_comm_mgr.h"
#include "new_rank_info.h"
#include "roce_channel_desc_configurator.h"

#include <acl/acl.h>
#include <unordered_map>
#include <unordered_set>
#include "shared_jetty_channel_pool.h"
#include "hccl_log_keywords.h"

using namespace hcomm;

namespace MyRankUtils {

uint32_t ResolveUbCommDomainQos(const hccl::CommConfig& commConfig)
{
    if (commConfig.GetConfigHcclQos() == HCCL_COMM_QOS_CONFIG_NOT_SET) {
        return EnvConfig::UB_QOS_DEFAULT;
    }
    return commConfig.GetConfigHcclQos();
}

HcommChannelDesc ChannelDescHccl2Hcomm(const HcclChannelDesc& hcclDesc, const hccl::CommConfig& commConfig)
{
    HcommChannelDesc hcommDesc{};
    (void)HcommChannelDescInit(&hcommDesc, 1);
    hcommDesc.remoteEndpoint = hcclDesc.remoteEndpoint;
    hcommDesc.notifyNum = hcclDesc.notifyNum;
    hcommDesc.memHandles = reinterpret_cast<HcommMemHandle*>(hcclDesc.memHandles);
    hcommDesc.memHandleNum = hcclDesc.memHandleNum;
    (void)memcpy_s(hcommDesc.raws, sizeof(hcommDesc.raws), hcclDesc.raws, sizeof(hcommDesc.raws));
    // RoCE：透传原始 hcclQos（可为 NOT_SET），由 CheckRoceAttr/ApplyRoceQosCompatToSlTc 决定是否映射 SL/TC
    if (hcclDesc.channelProtocol == COMM_PROTOCOL_ROCE) {
        hcommDesc.qos = commConfig.GetConfigHcclQos();
        hcommDesc.roceAttr.retryCnt = hcclDesc.roceAttr.retryCnt;
        hcommDesc.roceAttr.retryInterval = hcclDesc.roceAttr.retryInterval;
        hcommDesc.roceAttr.sl = hcclDesc.roceAttr.sl;
        hcommDesc.roceAttr.tc = hcclDesc.roceAttr.tc;
        return hcommDesc;
    }
    // UB 等：未配置时落默认 4，供下游 Jetty/TP 使用
    hcommDesc.qos = ResolveUbCommDomainQos(commConfig);
    if (hcclDesc.channelProtocol == COMM_PROTOCOL_UB_MEM) {
        hcommDesc.ubMemAttr.pathMode = hcclDesc.ubMemAttr.pathMode;
    }
    return hcommDesc;
}

/* 公共模块函数返回值定义，跟业务层同步 */
const std::unordered_map<CommProtocol, std::string> HCOM_COMM_PROTOCOL_STR_MAP
    = {{COMM_PROTOCOL_RESERVED, "RESERVED"}, {COMM_PROTOCOL_HCCS, "HCCS"},     {COMM_PROTOCOL_ROCE, "ROCE"},
       {COMM_PROTOCOL_PCIE, "PCIE"},         {COMM_PROTOCOL_SIO, "SIO"},       {COMM_PROTOCOL_UB_CTP, "UB_CTP"},
       {COMM_PROTOCOL_UBC_TP, "UBC_TP"},     {COMM_PROTOCOL_UB_MEM, "UB_MEM"}, {COMM_PROTOCOL_UBOE, "UBOE"},
       {COMM_PROTOCOL_UB_RTP, "UB_RTP"}};

inline std::string GetCommProtocolEnumStr(CommProtocol protocol)
{
    auto iter = HCOM_COMM_PROTOCOL_STR_MAP.find(protocol);
    if (iter == HCOM_COMM_PROTOCOL_STR_MAP.end()) {
        return "CommProtocol(" + std::to_string(protocol) + ")";
    } else {
        return iter->second;
    }
}

inline const char* GetTlsTypeStr(EndpointLocType localType)
{
    return localType == ENDPOINT_LOC_TYPE_HOST ? "HostDpu" : "Device";
}

inline const char* GetTlsStatusStr(Hccl::TlsStatus tlsStatus)
{
    if (tlsStatus == Hccl::TlsStatus::ENABLE) {
        return "ENABLE";
    }
    if (tlsStatus == Hccl::TlsStatus::DISABLE) {
        return "DISABLE";
    }
    return "UNKNOWN";
}

} // namespace MyRankUtils

namespace hccl {

constexpr uint32_t UNREUSE_CHANNEL_IDX = 0xFFFFFFFF;

MyRank::MyRank(
    aclrtBinHandle binHandle, uint32_t rankId, const CommConfig& config, const ManagerCallbacks& callbacks,
    RankGraph* rankGraph, const Hccl::RankIpPortMapPtr& rankIpPortMap)
    : binHandle_(binHandle),
      rankId_(rankId),
      config_(config),
      callbacks_(callbacks),
      rankGraph_(rankGraph),
      rankIpPortMap_(rankIpPortMap)
{}

MyRank::~MyRank()
{
    HCCL_INFO("[MyRank][~MyRank] MyRank deinit, rankId_[%u], devLogicId_[%d]", rankId_, devLogicId_);
    // 共享 Jetty Channel 不归 rankPairMgr_ 管理，需在 rankPairMgr_ 析构前独立清理
    (void)SharedJettyChannelPool::GetInstance().DestroyAllByMyRank(this);
    // 先清空反查索引，避免 rankPairMgr_ 析构 EndpointPair 时仍持有指向其的裸指针；
    // 持锁保证与并发 DestroyChannels 的索引读写一致
    {
        std::lock_guard<std::mutex> lock(channelIndexMtx_);
        handleToEpPair_.clear();
    }
    // 析构有时序要求
    rankPairMgr_ = nullptr; // 内部会销毁channel，可能需要返还endpoint与ccu资源
    endpointMgr_ = nullptr; // 内部会销毁endpoint，可能需要返回ccu资源

    struct ResourceCleanupGuard {
        explicit ResourceCleanupGuard(MyRank& myRank) : myRank_(myRank) {}
        ~ResourceCleanupGuard() noexcept
        {
            myRank_.ccuInsHandle_ = 0;
            myRank_.assignedCcuInsHandle_ = 0;

            if (myRank_.ccuDrvHandle_) {
                myRank_.ccuDrvHandle_ = nullptr; // 先减少引用计数，再尝试关闭
                (void)CcuDeinitFeature(myRank_.devLogicId_);
                // 尝试关闭CCU功能，最后一个调用时会关闭CCU驱动
            }

            myRank_.ReleaseCcuMsCommReservation();

            myRank_.commMems_ = nullptr;
            myRank_.nsRecoveryProcessor_ = nullptr;
        }

        MyRank& myRank_;
    } cleanupGuard(*this);

    if (ccuInsHandle_ != 0 || assignedCcuInsHandle_ != 0) { // 内部清理CCU资源，关闭CCU通道
        // 刷新并获取当前线程的 DeviceId
        int32_t threadDevId = INVALID_INT;
        CHK_RET_NULL(HcclDeviceRefresh(threadDevId));
        HCCL_INFO("[%s] curDeviceLogicId[%d], threadDevId[%d]", __func__, devLogicId_, threadDevId);
        // 先切换为目标 curDeviceLogicId
        bool isDiffDevId = false;
        if (devLogicId_ != threadDevId) {
            CHK_RET_NULL(hrtSetDevice(devLogicId_));
            isDiffDevId = true;
        }
        // 销毁通信域自有的 ccuInstance（QueryCcuIns 创建）
        if (ccuInsHandle_ != 0) {
            CHK_PRT(static_cast<HcclResult>(HcommCcuInsDestroy(ccuInsHandle_)));
        }
        // 销毁通过 Assign 绑定的 ccuInstance（所有权已转移给通信域）
        if (assignedCcuInsHandle_ != 0 && assignedCcuInsHandle_ != ccuInsHandle_) {
            CHK_PRT(static_cast<HcclResult>(HcommCcuInsDestroy(assignedCcuInsHandle_)));
        }
        // 切换回原来的 DeviceId
        if (isDiffDevId) {
            CHK_RET_NULL(hrtSetDevice(threadDevId));
            CHK_PRT(HcclDeviceRefresh(threadDevId));
        }
    }
}

HcclResult MyRank::GetLocalTlsStatus(EndpointLocType localType, Hccl::TlsStatus& tlsStatus) const
{
    tlsStatus = Hccl::TlsStatus::UNKNOWN;
    s32 deviceLogicId = -1;
    u32 devicePhyId = INVALID_UINT;
    CHK_RET(hrtGetDevice(&deviceLogicId));
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(deviceLogicId), devicePhyId));

    RaInfo info{};
    info.mode = localType == ENDPOINT_LOC_TYPE_HOST ? NetworkMode::NETWORK_PEER_ONLINE : NetworkMode::NETWORK_OFFLINE;
    info.phyId = devicePhyId;
    return Hccl::HrtRaGetTlsStatus(&info, tlsStatus);
}

void MyRank::GetAbnormalChannelTlsStatus(
    const HcclChannelDesc* channelDescs, const int32_t* statusList, uint32_t channelNum,
    std::vector<Hccl::TlsStatus>& tlsStatusList) const
{
    tlsStatusList.assign(channelNum, Hccl::TlsStatus::UNKNOWN);
    for (uint32_t i = 0; i < channelNum; ++i) {
        if (statusList[i] == ChannelStatus::READY) {
            continue;
        }
        Hccl::TlsStatus tlsStatus = Hccl::TlsStatus::UNKNOWN;
        HcclResult ret = GetLocalTlsStatus(channelDescs[i].localEndpoint.loc.locType, tlsStatus);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING("[GetLocalTlsStatus] Can not get TlsStatus, channelIndex[%u], ret[%d]", i, ret);
            continue;
        }
        tlsStatusList[i] = tlsStatus;
    }
}

// [中文导读] [AllReduce逐行 S227] MyRank::RegisterCommMemsToEndpoint的接口声明：本地端点句柄；这些参数属于本函数调用边界。
HcclResult MyRank::RegisterCommMemsToEndpoint(EndpointHandle epHandle)
// [中文导读] [AllReduce逐行 S228] 进入MyRank::RegisterCommMemsToEndpoint函数体：取域内全部内存描述/tag/版本并同步到当前 Endpoint。
{
    // [中文导读] [AllReduce逐行 S229] 准备域内注册内存描述数组的局部存储/结构描述，初始化方式以本行声明为准。
    std::vector<HcclMem> memVec;
    // [中文导读] [AllReduce逐行 S230] 准备内存标签数组的局部存储/结构描述，初始化方式以本行声明为准。
    std::vector<std::string> memTag;
    // [中文导读] [AllReduce逐行 S231] 设置内存登记版本为/按`0`。
    uint64_t version = 0;
    // [中文导读] 获取域内所有内存、标签及同一轮版本，再让 EndpointMgr 把该版本同步到当前端点。
    // [中文导读] [AllReduce逐行 S233] 同时取得域内内存描述、tag列表及内存版本；返回非成功时由检查宏立即向上传递。
    CHK_RET(commMems_->GetAllMemory(memVec, memTag, version));
    // [中文导读] [AllReduce逐行 S234] 记录MyRank::RegisterCommMemsToEndpoint的状态/性能诊断，字段包含域内注册内存描述数组的size字段、内存登记版本；日志本身不执行传输。
    HCCL_INFO("[%s] got %zu memory regions to register, version[%llu]", __func__, memVec.size(), version);
    // [中文导读] [AllReduce逐行 S235] 把域内该版本内存注册到当前Endpoint；返回非成功时由检查宏立即向上传递。
    CHK_RET(endpointMgr_->RegisterMemory(epHandle, memTag, memVec, version));
    // [中文导读] [AllReduce逐行 S236] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S237] 结束MyRank::RegisterCommMemsToEndpoint函数体；控制流返回外层。
}

// [中文导读] [AllReduce逐行 S239] MyRank::PrepareMemHandles的接口声明：从域级内存句柄提取 tag，把域内存注册到 Endpoint，再选择当前通道需要的注册句柄；这些参数属于本函数调用边界。
HcclResult MyRank::PrepareMemHandles(
    // [中文导读] [AllReduce逐行 S240] MyRank::PrepareMemHandles的接口声明：本地端点句柄、域级内存句柄数组、用户内存句柄条数、本通道端点注册句柄列表；这些参数属于本函数调用边界。
    EndpointHandle epHandle, void** memHandles, uint32_t memHandleNum, std::vector<MemHandle>& memHandleVec)
// [中文导读] [AllReduce逐行 S241] 进入MyRank::PrepareMemHandles函数体：从域级内存句柄提取 tag，把域内存注册到 Endpoint，再选择当前通道需要的注册句柄。
{
    // 从 CommMems 提取该 channel 需要的 tag 列表
    // GetTagsFromHandles 始终 push cclBuffer；用户 handles 异常时内部跳过，不阻断注册
    // [中文导读] [AllReduce逐行 S244] 准备本通道需要交换的内存标签的局部存储/结构描述，初始化方式以本行声明为准。
    std::vector<std::string> memTags;
    // [中文导读] 先由域句柄解析当前 Channel 需要的标签；包含 CCL 区的规则由 CommMems 统一维护。
    // [中文导读] [AllReduce逐行 S246] 按域内存句柄解析交换tag并加入CCL标签；返回非成功时由检查宏立即向上传递。
    CHK_RET(commMems_->GetTagsFromHandles(memHandles, memHandleNum, memTags));

    // 确保 CommMems 全量内存已注册到该 endpoint（版本一致则跳过）
    // [中文导读] [AllReduce逐行 S249] 调用RegisterCommMemsToEndpoint，使用本地端点句柄；返回非成功时由检查宏立即向上传递。
    CHK_RET(RegisterCommMemsToEndpoint(epHandle));

    // 从 endpoint 查询指定 tag 的 MemHandle
    // [中文导读] [AllReduce逐行 S252] 按tag顺序取得当前Endpoint的底层注册句柄；返回非成功时由检查宏立即向上传递。
    CHK_RET(endpointMgr_->GetMemHandlesByTags(epHandle, memTags, memHandleVec));
    // [中文导读] [AllReduce逐行 S253] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S254] 结束MyRank::PrepareMemHandles函数体；控制流返回外层。
}

HcclResult MyRank::UnregMemByTag(const std::string& tag)
{
    CHK_PTR_NULL(endpointMgr_);
    return endpointMgr_->UnregMemByTag(tag);
}

HcclResult MyRank::ReserveCcuMsCommOrFallback()
{
    if (opExpansionMode_ != CCU_MS_MODE) {
        return HCCL_SUCCESS;
    }

    bool reserved = false;
    CHK_RET(CollCommMgr::GetInstance().TryReserveCcuMsComm(devLogicId_, config_.GetConfigCommName(), reserved));
    if (reserved) {
        ccuMsCommReserved_ = true;
        return HCCL_SUCCESS;
    }

    opExpansionMode_ = CCU_SCHED_MODE;
    HCCL_RUN_WARNING(
        "[MyRank][%s] CCU_MS comm already exists on device[%d], fallback to CCU_SCHED, rankId[%u].", __func__,
        devLogicId_, rankId_);
    return HCCL_SUCCESS;
}

void MyRank::ReleaseCcuMsCommReservation()
{
    if (!ccuMsCommReserved_) {
        return;
    }
    CollCommMgr::GetInstance().ReleaseCcuMsComm(devLogicId_, config_.GetConfigCommName());
    ccuMsCommReserved_ = false;
}

void MyRank::ReconcileCcuMsCommReservation(HcclResult initRet)
{
    if (initRet != HCCL_SUCCESS || opExpansionMode_ != CCU_MS_MODE) {
        ReleaseCcuMsCommReservation();
    }
}

HcclResult MyRank::TryInitCcuInstanceOnDemand()
{
    auto ccuInsType = OpExpansionModeToCcuInstanceType(opExpansionMode_);
    if (ccuInsType == CcuInstanceType::CCU_UNUSED) {
        ccuInsHandle_ = 0;
        return HcclResult::HCCL_SUCCESS;
    }

    if (mainBoardType_ == Hccl::HcclMainboardId::MAINBOARD_OTHERS) {
        CHK_RET(CcuGetMainboardType(devLogicId_, mainBoardType_));
    }

    if (mainBoardType_ == Hccl::HcclMainboardId::MAINBOARD_PCIE_STD
        && ccuInsType == CcuInstanceType::CCU_MS) { // 标卡环境下配置CCU_MS拦截报错
        HCCL_ERROR(
            "[%s] ccuInstanceType[%d] not support in %s", __func__, ccuInsType, mainBoardType_.Describe().c_str());
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    // 拉起ccu驱动
    if (!ccuDrvHandle_) {
        auto ccuInitRet = CcuInitFeature(devLogicId_, ccuDrvHandle_);
        // ccu驱动拉起失败，直接回退至aicpu ts
        if (ccuInitRet == CcuResult::CCU_E_DRV_BUSY) {
            opExpansionMode_ = AICPU_TS_MODE;
            ccuInsHandle_ = 0;
            HCCL_RUN_WARNING(
                "[MyRank][%s] failed to init ccu driver, "
                "fallback to aicpu, rankId[%u].",
                __func__, rankId_);
            return HcclResult::HCCL_SUCCESS;
        }

        // 预期外返回值属于错误
        if (ccuInitRet != CcuResult::CCU_SUCCESS) {
            HCCL_ERROR("[%s] failed, ret[%d] is not expected.", __func__, ccuInitRet);
            ccuInsHandle_ = 0;
            return static_cast<HcclResult>(ccuInitRet);
        }
    }

    // ccu驱动拉起成功
    return HcclResult::HCCL_SUCCESS;
}

HcclResult MyRank::TryInitCcuInstance()
{
    CHK_RET(ReserveCcuMsCommOrFallback());

    // ccu instance 不在 init 时创建，由 QueryCcuIns 创建或由 Assign 显式绑定；
    // 此处仅拉起 ccu 驱动。
    HcclResult ret = TryInitCcuInstanceOnDemand();
    ReconcileCcuMsCommReservation(ret);
    return ret;
}

HcclResult MyRank::GetDevicePortInternal(uint32_t rank, uint32_t* devPort, EndpointLocType locType)
{
    CHK_PTR_NULL(devPort);
    CHK_PTR_NULL(rankGraph_);

    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    // v1 模式 (mode_ == 0): 强制转换为 RankGraphV1 调用 GetDevicePort
    // v2 模式 (mode_ != 0): 使用 rankGraph_->GetDevicePort()
    if (devType == DevType::DEV_TYPE_910B) {
        RankGraphV1* rankGraphV1 = static_cast<RankGraphV1*>(rankGraph_);
        CHK_RET(rankGraphV1->GetDevicePort(rank, devPort));
    } else {
        CHK_RET(rankGraph_->GetListenPort(rank, devPort, locType));
    }
    return HCCL_SUCCESS;
}

HcclResult MyRank::Init(HcclMem cclBuffer, const uint32_t opExpansionMode, uint32_t rankNum)
{
    // EXCEPTION_HANDLE_BEGIN
    CHK_RET(hrtGetDevice(&devLogicId_));

    // ns recovery processor初始化
    EXCEPTION_CATCH(nsRecoveryProcessor_ = std::make_unique<NsRecoveryProcessor>(), return HCCL_E_PTR);

    // 创建通信内存管理器
    EXCEPTION_CATCH(commMems_ = std::make_unique<CommMems>(config_.GetConfigBufferSize()), return HCCL_E_PTR);

    // 初始化通信内存
    CHK_RET(commMems_->Init(cclBuffer));

    EXCEPTION_CATCH(engineCtxs_ = std::make_unique<EngineCtxs>(), return HCCL_E_PTR);

    // 通信域配置config优先级更高，当配置默认展开模式时，读取环境变量配置
    opExpansionMode_ = opExpansionMode;
    if (opExpansionMode_ == DEFAULT_MODE) {
        // 环境变量模块已处理，当用户未配置时，输出ccu sched模式
        auto accelerator = Hccl::EnvConfig::GetInstance().GetAlgoConfig().GetHcclAccelerator();
        HCCL_RUN_INFO("[MyRank][%s] set op expansion mode by env[%s].", __func__, accelerator.Describe().c_str());
        opExpansionMode_ = static_cast<uint32_t>(accelerator);
    }

    // 仅自定义算子ccu流程初始化资源
    if (ccuInsHandle_ == 0 && rankNum != 1 && (opExpansionMode_ == CCU_MS_MODE || opExpansionMode_ == CCU_SCHED_MODE)) {
        const uint32_t originOpExpansionMode = opExpansionMode_; // 记录原始加速模式，避免中间执行修改后丢失
        auto ret = TryInitCcuInstance();
        if (ret != HcclResult::HCCL_SUCCESS) { // 申请成功与回退成功都属于成功，其他均非预期
            HCCL_ERROR(
                "[MyRank][%s] failed to init ccu instance, op expansion mode[%u].", __func__, originOpExpansionMode);
            return ret;
        }
    }

    // 创建端点管理器
    EXCEPTION_CATCH(endpointMgr_ = std::make_unique<hccl::EndpointMgr>(), return HCCL_E_PTR);

    // rankPairMgr_初始化
    EXCEPTION_CATCH(rankPairMgr_ = std::make_unique<RankPairMgr>(rankIpPortMap_), return HCCL_E_PTR);

    Hccl::DfxDlProfFunction::GetInstance().DfxDlProfFunctionInit();
    // EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

HcclResult MyRank::QueryListenPort(
    uint32_t localRank, uint32_t remoteRank, const EndpointDesc& localEndpointDesc,
    const EndpointDesc& remoteEndpointDesc, uint32_t& listenPort, HcommChannelDesc& hcommDesc)
{
    // 查询rmtRankId对应的devPort
    uint32_t rmtPort = 0;
    CHK_RET(GetDevicePortInternal(remoteRank, &rmtPort, remoteEndpointDesc.loc.locType));
    if (rmtPort > Hccl::MAX_VALUE_TCPPORT) {
        HCCL_ERROR(
            "[%s] Invalid port[%u] of Rank[%u], max valid port is %u", __func__, rmtPort, remoteRank,
            Hccl::MAX_VALUE_TCPPORT);
        return HCCL_E_PARA;
    }
    // 查询该socket链接的server端监听的端口（监听方的选择策略需要跟SocketConfig中保持一致）
    Hccl::IpAddress localIpAddr{};
    Hccl::IpAddress remoteIpAddr{};
    CHK_RET(CommAddrToIpAddress(localEndpointDesc.commAddr, localIpAddr));
    CHK_RET(CommAddrToIpAddress(remoteEndpointDesc.commAddr, remoteIpAddr));
    if (localIpAddr < remoteIpAddr) {
        // 查询localRankId对应的devPort
        CHK_RET(GetDevicePortInternal(localRank, &listenPort, localEndpointDesc.loc.locType));
        hcommDesc.role = HcommSocketRole::HCOMM_SOCKET_ROLE_SERVER;
        if (listenPort > Hccl::MAX_VALUE_TCPPORT) {
            HCCL_ERROR("[%s] Invalid port[%u] of Rank[%u]", __func__, listenPort, localRank);
            return HCCL_E_PARA;
        }
        hcommDesc.port = static_cast<uint16_t>(listenPort); // HcommChannelDesc.port中填监听端口号
    } else {
        listenPort = rmtPort;
        hcommDesc.role = HcommSocketRole::HCOMM_SOCKET_ROLE_CLIENT;
        hcommDesc.port
            = static_cast<uint16_t>(rmtPort); // HcommChannelDesc.port中填对端端口号(此场景下对端端口号也就是监听端口号)
    }

    return HCCL_SUCCESS;
}

// [中文导读] [AllReduce逐行 S456] MyRank::GetEndpointPairFromChannel的接口声明：按 Rank 对与 Endpoint 描述对逐级定位 EndpointPair；这些参数属于本函数调用边界。
HcclResult MyRank::GetEndpointPairFromChannel(
    // [中文导读] [AllReduce逐行 S457] MyRank::GetEndpointPairFromChannel的接口声明：当前通道描述、通道请求下标、通道请求条数、对端域内Rank编号；这些参数属于本函数调用边界。
    const HcclChannelDesc& channelDesc, uint32_t channelIndex, uint32_t channelNum, uint32_t& remoteRank,
    // [中文导读] [AllReduce逐行 S458] MyRank::GetEndpointPairFromChannel的接口声明：本端/对端Endpoint资源对、本端/对端Rank资源对；这些参数属于本函数调用边界。
    hcomm::EndpointPair*& endpointPair, RankPair*& rankPair)
// [中文导读] [AllReduce逐行 S459] 进入MyRank::GetEndpointPairFromChannel函数体：按 Rank 对与 Endpoint 描述对逐级定位 EndpointPair。
{
    // [中文导读] [AllReduce逐行 S460] 设置对端域内Rank编号为/按`channelDesc.remoteRank`（当前通道描述的remoteRank字段）。
    remoteRank = channelDesc.remoteRank;
    // [中文导读] [AllReduce逐行 S461] 记录MyRank::GetEndpointPairFromChannel的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S462] 为当前MyRank::GetEndpointPairFromChannel诊断/异常表达式提供格式文本，将报告通道请求下标、通道请求条数；这一物理行没有数据搬运副作用。
        "[%s][%u/%u] remoteRank[%u] localProtocol[%d] remoteProtocol[%d]", __func__, channelIndex + 1, channelNum,
        // [中文导读] [AllReduce逐行 S463] 为前述多行表达式补入`remoteRank, channelDesc.localEndpoint.protocol, channelDesc.remoteEndpoint.protocol)`（对端域内Rank编号、当前通道描述的localEndpoint.protocol字段、当前通道描述的remoteEndpoint.protocol字段）；本行是参数/结构化初始化续行。
        remoteRank, channelDesc.localEndpoint.protocol, channelDesc.remoteEndpoint.protocol);

    // [中文导读] 先按本地/远端 Rank 找 RankPair，再按两端 Endpoint 描述取得 EndpointPair，建立两级资源定位。
    // [中文导读] [AllReduce逐行 S466] 设置Rank对缓存键为/按`std::make_pair(rankId_, remoteRank)`（本端域内Rank编号、对端域内Rank编号）；调用std::make_pair，使用本端域内Rank编号、对端域内Rank编号。
    const RankIdPair rankIdPair = std::make_pair(rankId_, remoteRank);
    // [中文导读] [AllReduce逐行 S467] 设置端点对缓存键为/按`std::make_pair(channelDesc.localEndpoint, channelDesc.remoteEndpoint)`（当前通道描述的localEndpoint字段、当前通道描述的remoteEndpoint字段）；调用std::make_pair，使用当前通道描述的localEndpoint字段、当前通道描述的remoteEndpoint字段。
    const EndpointDescPair endpointDescPair = std::make_pair(channelDesc.localEndpoint, channelDesc.remoteEndpoint);
    // [中文导读] [AllReduce逐行 S468] 按对应缓存键取得/创建资源；返回非成功时由检查宏立即向上传递。
    CHK_RET(rankPairMgr_->Get(rankIdPair, rankPair));
    // [中文导读] [AllReduce逐行 S469] 检查`rankPair`（本端/对端Rank资源对）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(rankPair);
    // [中文导读] [AllReduce逐行 S470] 按两端描述取得或建立EndpointPair；返回非成功时由检查宏立即向上传递。
    CHK_RET(rankPair->GetEndpointPair(endpointDescPair, endpointPair));
    // [中文导读] [AllReduce逐行 S471] 检查`endpointPair`（本端/对端Endpoint资源对）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(endpointPair);
    // [中文导读] [AllReduce逐行 S472] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S473] 结束MyRank::GetEndpointPairFromChannel函数体；控制流返回外层。
}

inline std::string AddProtocolToSocketTag(const std::string& socketTag, const HcclChannelDesc* channelDescs)
{
    std::string newSocketTag = socketTag + "_protocol_" + std::to_string(channelDescs->channelProtocol);
    return newSocketTag;
}

// [中文导读] [AllReduce逐行 S481] MyRank::BatchServerInitForChannels的接口声明：按 RankPair/EndpointPair 准备 Socket 监听，并按协议区分标签；这些参数属于本函数调用边界。
HcclResult MyRank::BatchServerInitForChannels(
    // [中文导读] [AllReduce逐行 S482] MyRank::BatchServerInitForChannels的接口声明：域级通道描述数组、通道请求条数、域和引擎组成的连接标签；这些参数属于本函数调用边界。
    const HcclChannelDesc* channelDescs, uint32_t channelNum, const std::string& socketTag,
    // [中文导读] [AllReduce逐行 S483] MyRank::BatchServerInitForChannels的接口声明：按Rank对/端点对分组的Socket计数表；这些参数属于本函数调用边界。
    ReuseSocketIdxMap& reuseSocketIdxMap)
// [中文导读] [AllReduce逐行 S484] 进入MyRank::BatchServerInitForChannels函数体：按 RankPair/EndpointPair 准备 Socket 监听，并按协议区分标签。
{
    // 批量获取socket，与server监听隔离开
    // [中文导读] [AllReduce逐行 S486] 按`(uint32_t i = 0; i < channelNum; ++i)`（本批条目下标、通道请求条数）遍历本批条目/分片；各次处理保持数组对应关系。
    for (uint32_t i = 0; i < channelNum; ++i) {
        // [中文导读] [AllReduce逐行 S487] 设置本端/对端Endpoint资源对为/按`nullptr`。
        hcomm::EndpointPair* endpointPair = nullptr;
        // [中文导读] [AllReduce逐行 S488] 设置本端/对端Rank资源对为/按`nullptr`。
        RankPair* rankPair = nullptr;
        // [中文导读] [AllReduce逐行 S489] 设置对端域内Rank编号为/按`0`。
        uint32_t remoteRank = 0;

        // [中文导读] [AllReduce逐行 S491] 按Rank对与端点描述对定位资源缓存；返回非成功时由检查宏立即向上传递。
        CHK_RET(GetEndpointPairFromChannel(channelDescs[i], i, channelNum, remoteRank, endpointPair, rankPair));

        // [中文导读] [AllReduce逐行 S493] 仅当`(reuseSocketIdxMap.find(rankPair) == reuseSocketIdxMap.end())`（按Rank对/端点对分组的Socket计数表的find字段、本端/对端Rank资源对、按Rank对/端点对分组的Socket计数表的end字段）成立时进入此分支；调用find, end，使用按Rank对/端点对分组的Socket计数表的find字段、本端/对端Rank资源对、按Rank对/端点对分组的Socket计数表的end字段。
        if (reuseSocketIdxMap.find(rankPair) == reuseSocketIdxMap.end()) {
            // [中文导读] [AllReduce逐行 S494] 准备`std::unordered_map<hcomm::EndpointPair*, u32> endpointPair2Idx{}`的局部存储/结构描述，初始化方式以本行声明为准。
            std::unordered_map<hcomm::EndpointPair*, u32> endpointPair2Idx{};
            // [中文导读] [AllReduce逐行 S495] 调用emplace，使用本端/对端Endpoint资源对；传入/处理本端/对端Endpoint资源对。
            endpointPair2Idx.emplace(endpointPair, 0);
            // [中文导读] [AllReduce逐行 S496] 调用emplace，使用按Rank对/端点对分组的Socket计数表的emplace字段、本端/对端Rank资源对；传入/处理按Rank对/端点对分组的Socket计数表的emplace字段、本端/对端Rank资源对。
            reuseSocketIdxMap.emplace(rankPair, endpointPair2Idx);
        // [中文导读] [AllReduce逐行 S497] 仅当`(reuseSocketIdxMap[rankPair].find(endpointPair) == reuseSocketIdxMap[rankPair].end())`（按Rank对/端点对分组的Socket计数表、本端/对端Rank资源对、本端/对端Endpoint资源对）成立时进入此分支；调用find, end，使用按Rank对/端点对分组的Socket计数表、本端/对端Rank资源对、本端/对端Endpoint资源对。
        } else if (reuseSocketIdxMap[rankPair].find(endpointPair) == reuseSocketIdxMap[rankPair].end()) {
            // [中文导读] [AllReduce逐行 S498] 调用emplace，使用按Rank对/端点对分组的Socket计数表、本端/对端Rank资源对、本端/对端Endpoint资源对；传入/处理按Rank对/端点对分组的Socket计数表、本端/对端Rank资源对、本端/对端Endpoint资源对。
            reuseSocketIdxMap[rankPair].emplace(endpointPair, 0);
        // [中文导读] [AllReduce逐行 S499] 结束`if (reuseSocketIdxMap.find(rankPair) == reuseSocketIdxMap.end())`（按Rank对/端点对分组的Socket计数表的find字段、本端/对端Rank资源对、按Rank对/端点对分组的Socket计数表的end字段）分支/循环；控制流返回外层。
        }
        // [中文导读] [AllReduce逐行 S500] 设置本批通道或Socket复用槽位为/按`reuseSocketIdxMap[rankPair][endpointPair]`（按Rank对/端点对分组的Socket计数表、本端/对端Rank资源对、本端/对端Endpoint资源对）。
        u32& reuseIdx = reuseSocketIdxMap[rankPair][endpointPair];

        // [中文导读] [AllReduce逐行 S502] 准备本端物理设备编号的局部存储/结构描述，初始化方式以本行声明为准。
        uint32_t devicePhyId;
        // [中文导读] [AllReduce逐行 S503] 准备对端物理设备编号的局部存储/结构描述，初始化方式以本行声明为准。
        uint32_t remoteDevicePhyId;
        // [中文导读] [AllReduce逐行 S504] 调用GetDeviceId，使用本端域内Rank编号、本端物理设备编号；传入/处理本端域内Rank编号、本端物理设备编号。
        rankGraph_->GetDeviceId(rankId_, &devicePhyId);
        // [中文导读] [AllReduce逐行 S505] 调用GetDeviceId，使用对端域内Rank编号、对端物理设备编号；传入/处理对端域内Rank编号、对端物理设备编号。
        rankGraph_->GetDeviceId(remoteRank, &remoteDevicePhyId);

        // [中文导读] 把协议加入连接标签后启动监听，避免同一域同一引擎的不同协议连接共用错误 Socket。
        // [中文导读] [AllReduce逐行 S508] 设置附加协议的Socket标签为/按`AddProtocolToSocketTag(socketTag, &channelDescs[i])`（域和引擎组成的连接标签、域级通道描述数组、本批条目下标）；调用AddProtocolToSocketTag，使用域和引擎组成的连接标签、域级通道描述数组、本批条目下标。
        const std::string socketTagAddProto = AddProtocolToSocketTag(socketTag, &channelDescs[i]);
        // [中文导读] [AllReduce逐行 S509] 设置当前调用状态为/按`endpointPair->ServerInit(`（本端/对端Endpoint资源对的ServerInit字段）；为设备端点配置Socket服务器监听，Host分支另行处理。
        auto ret = endpointPair->ServerInit(
            // [中文导读] [AllReduce逐行 S510] 为为设备端点配置Socket服务器监听，Host分支另行处理补入`rankId_, remoteRank, socketTagAddProto, reuseIdx, devicePhyId, remoteDevicePhyId)`（本端域内Rank编号、对端域内Rank编号、附加协议的Socket标签、本批通道或Socket复用槽位、本端物理设备编号、对端物理设备编号）；本行是参数/结构化初始化续行。
            rankId_, remoteRank, socketTagAddProto, reuseIdx, devicePhyId, remoteDevicePhyId);
        // [中文导读] [AllReduce逐行 S511] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
        CHK_PRT_RET(
            // [中文导读] [AllReduce逐行 S512] 向条件错误检查提供`ret != HCCL_SUCCESS,`（当前调用状态），用于确定触发条件或形成对应诊断。
            ret != HCCL_SUCCESS,
            // [中文导读] [AllReduce逐行 S513] 记录MyRank::BatchServerInitForChannels的错误诊断；日志本身不执行传输。
            HCCL_ERROR(
                // [中文导读] [AllReduce逐行 S514] 为当前MyRank::BatchServerInitForChannels诊断/异常表达式提供格式文本，将报告本批条目下标；这一物理行没有数据搬运副作用。
                "[%s] ServerInitFailed, channelIndex[%u], remoteRank[%u], protocol[%d] reuseIdx[%u]", __func__, i,
                // [中文导读] [AllReduce逐行 S515] 为前述多行表达式补入`remoteRank, channelDescs[i].localEndpoint.protocol, reuseIdx),`（对端域内Rank编号、域级通道描述数组、本批条目下标、本批通道或Socket复用槽位）；本行是参数/结构化初始化续行。
                remoteRank, channelDescs[i].localEndpoint.protocol, reuseIdx),
            // [中文导读] [AllReduce逐行 S516] 指定条件命中时要返回的错误状态`ret)`（当前调用状态），未命中则继续原处理路径。
            ret);

        // [中文导读] [AllReduce逐行 S518] 记录MyRank::BatchServerInitForChannels的状态/性能诊断；日志本身不执行传输。
        HCCL_INFO(
            // [中文导读] [AllReduce逐行 S519] 为当前MyRank::BatchServerInitForChannels诊断/异常表达式提供格式文本，将报告本批条目下标、通道请求条数；这一物理行没有数据搬运副作用。
            "[%s][%u/%u] server listen successfully, remoteRank[%u], reuseIdx[%u]", __func__, i + 1, channelNum,
            // [中文导读] [AllReduce逐行 S520] 为前述多行表达式补入`remoteRank, reuseIdx)`（对端域内Rank编号、本批通道或Socket复用槽位）；本行是参数/结构化初始化续行。
            remoteRank, reuseIdx);
    // [中文导读] [AllReduce逐行 S521] 结束`for (uint32_t i = 0; i < channelNum; ++i)`（本批条目下标、通道请求条数）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S522] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S523] 结束MyRank::BatchServerInitForChannels函数体；控制流返回外层。
}

// [中文导读] [AllReduce逐行 S525] MyRank::BatchGetSocketsForChannels的接口声明：取得远端监听端口和已连接 Socket，把连接写回 Hcomm 描述；这些参数属于本函数调用边界。
HcclResult MyRank::BatchGetSocketsForChannels(
    // [中文导读] [AllReduce逐行 S526] MyRank::BatchGetSocketsForChannels的接口声明：域级通道描述数组、通道请求条数、域和引擎组成的连接标签；这些参数属于本函数调用边界。
    const HcclChannelDesc* channelDescs, uint32_t channelNum, const std::string& socketTag,
    // [中文导读] [AllReduce逐行 S527] MyRank::BatchGetSocketsForChannels的接口声明：基础层通道描述数组、按Rank对/端点对分组的Socket计数表；这些参数属于本函数调用边界。
    std::vector<HcommChannelDesc>& hcommDescs, ReuseSocketIdxMap& reuseSocketIdxMap)
// [中文导读] [AllReduce逐行 S528] 进入MyRank::BatchGetSocketsForChannels函数体：取得远端监听端口和已连接 Socket，把连接写回 Hcomm 描述。
{
    // [中文导读] [AllReduce逐行 S529] 按`(uint32_t i = 0; i < channelNum; ++i)`（本批条目下标、通道请求条数）遍历本批条目/分片；各次处理保持数组对应关系。
    for (uint32_t i = 0; i < channelNum; ++i) {
        // [中文导读] [AllReduce逐行 S530] 设置本端/对端Endpoint资源对为/按`nullptr`。
        hcomm::EndpointPair* endpointPair = nullptr;
        // [中文导读] [AllReduce逐行 S531] 设置本端/对端Rank资源对为/按`nullptr`。
        RankPair* rankPair = nullptr;
        // [中文导读] [AllReduce逐行 S532] 设置对端域内Rank编号为/按`0`。
        uint32_t remoteRank = 0;

        // [中文导读] [AllReduce逐行 S534] 按Rank对与端点描述对定位资源缓存；返回非成功时由检查宏立即向上传递。
        CHK_RET(GetEndpointPairFromChannel(channelDescs[i], i, channelNum, remoteRank, endpointPair, rankPair));

        // [中文导读] [AllReduce逐行 S536] 设置连接监听端口为/按`0`。
        uint32_t listenPort = 0;
        // [中文导读] 依据端点类型与拓扑求远端监听端口，并把需要的角色/端口字段补进底层 Channel 描述。
        // [中文导读] [AllReduce逐行 S538] 按本端/对端端点类型与拓扑求监听端口并填底层角色；返回非成功时由检查宏立即向上传递。
        CHK_RET(QueryListenPort(
            // [中文导读] [AllReduce逐行 S539] 为按本端/对端端点类型与拓扑求监听端口并填底层角色补入`rankId_, remoteRank, channelDescs[i].localEndpoint, channelDescs[i].remoteEndpoint, listenPort,`（本端域内Rank编号、对端域内Rank编号、域级通道描述数组、本批条目下标、连接监听端口）；本行是参数/结构化初始化续行。
            rankId_, remoteRank, channelDescs[i].localEndpoint, channelDescs[i].remoteEndpoint, listenPort,
            // [中文导读] [AllReduce逐行 S540] 为按本端/对端端点类型与拓扑求监听端口并填底层角色补入`hcommDescs[i]))`（基础层通道描述数组、本批条目下标）；本行是参数/结构化初始化续行。
            hcommDescs[i]));

        // [中文导读] [AllReduce逐行 S542] 设置本批通道或Socket复用槽位为/按`reuseSocketIdxMap[rankPair][endpointPair]`（按Rank对/端点对分组的Socket计数表、本端/对端Rank资源对、本端/对端Endpoint资源对）。
        u32& reuseIdx = reuseSocketIdxMap[rankPair][endpointPair];
        // [中文导读] [AllReduce逐行 S543] 准备本端物理设备编号的局部存储/结构描述，初始化方式以本行声明为准。
        uint32_t devicePhyId;
        // [中文导读] [AllReduce逐行 S544] 准备对端物理设备编号的局部存储/结构描述，初始化方式以本行声明为准。
        uint32_t remoteDevicePhyId;
        // [中文导读] [AllReduce逐行 S545] 调用GetDeviceId，使用本端域内Rank编号、本端物理设备编号；传入/处理本端域内Rank编号、本端物理设备编号。
        rankGraph_->GetDeviceId(rankId_, &devicePhyId);
        // [中文导读] [AllReduce逐行 S546] 调用GetDeviceId，使用对端域内Rank编号、对端物理设备编号；传入/处理对端域内Rank编号、对端物理设备编号。
        rankGraph_->GetDeviceId(remoteRank, &remoteDevicePhyId);
        // [中文导读] [AllReduce逐行 S547] 记录MyRank::BatchGetSocketsForChannels的状态/性能诊断；日志本身不执行传输。
        HCCL_INFO(
            // [中文导读] [AllReduce逐行 S548] 为当前MyRank::BatchGetSocketsForChannels诊断/异常表达式提供格式文本，将报告本端域内Rank编号；这一物理行没有数据搬运副作用。
            "[MyRank][BatchCreateSockets] rankId_[%u] devicePhyId[%u] remoteRank[%u] remoteDevicePhyId[%u]", rankId_,
            // [中文导读] [AllReduce逐行 S549] 为前述多行表达式补入`devicePhyId, remoteRank, remoteDevicePhyId)`（本端物理设备编号、对端域内Rank编号、对端物理设备编号）；本行是参数/结构化初始化续行。
            devicePhyId, remoteRank, remoteDevicePhyId);
        // [中文导读] [AllReduce逐行 S550] 设置已连接的Socket对象为/按`nullptr`。
        Hccl::Socket* socket = nullptr;
        // [中文导读] [AllReduce逐行 S551] 设置附加协议的Socket标签为/按`AddProtocolToSocketTag(socketTag, &channelDescs[i])`（域和引擎组成的连接标签、域级通道描述数组、本批条目下标）；调用AddProtocolToSocketTag，使用域和引擎组成的连接标签、域级通道描述数组、本批条目下标。
        const std::string socketTagAddProto = AddProtocolToSocketTag(socketTag, &channelDescs[i]);
        // [中文导读] [AllReduce逐行 S552] 设置当前调用状态为/按`endpointPair->GetConnectedSocket(`（本端/对端Endpoint资源对的GetConnectedSocket字段）；取得指定Rank/协议标签/槽位上的已连接Socket。
        auto ret = endpointPair->GetConnectedSocket(
            // [中文导读] [AllReduce逐行 S553] 为取得指定Rank/协议标签/槽位上的已连接Socket补入`rankId_, remoteRank, socketTagAddProto, reuseIdx, listenPort, socket, devicePhyId, remoteDevicePhyId)`（本端域内Rank编号、对端域内Rank编号、附加协议的Socket标签、本批通道或Socket复用槽位、连接监听端口、已连接的Socket对象、本端物理设备编号、对端物理设备编号）；本行是参数/结构化初始化续行。
            rankId_, remoteRank, socketTagAddProto, reuseIdx, listenPort, socket, devicePhyId, remoteDevicePhyId);
        // [中文导读] [AllReduce逐行 S554] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
        CHK_PRT_RET(
            // [中文导读] [AllReduce逐行 S555] 向条件错误检查提供`ret != HCCL_SUCCESS,`（当前调用状态），用于确定触发条件或形成对应诊断。
            ret != HCCL_SUCCESS,
            // [中文导读] [AllReduce逐行 S556] 记录MyRank::BatchGetSocketsForChannels的错误诊断；日志本身不执行传输。
            HCCL_ERROR(
                // [中文导读] [AllReduce逐行 S557] 为当前MyRank::BatchGetSocketsForChannels诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                "[%s] failed to get socket, channelIndex[%u], remoteRank[%u], protocol[%d], reuseIdx[%u], tag[%s]",
                // [中文导读] [AllReduce逐行 S558] 为调用c_str，使用本批条目下标、对端域内Rank编号、域级通道描述数组、本批通道或Socket复用槽位、附加协议的Socket标签的c_str字段补入`__func__, i, remoteRank, channelDescs[i].localEndpoint.protocol, reuseIdx, socketTagAddProto.c_str()),`（本批条目下标、对端域内Rank编号、域级通道描述数组、本批通道或Socket复用槽位、附加协议的Socket标签的c_str字段）；本行是参数/结构化初始化续行。
                __func__, i, remoteRank, channelDescs[i].localEndpoint.protocol, reuseIdx, socketTagAddProto.c_str()),
            // [中文导读] [AllReduce逐行 S559] 指定条件命中时要返回的错误状态`ret)`（当前调用状态），未命中则继续原处理路径。
            ret);
        // [中文导读] [AllReduce逐行 S560] 检查`socket`（已连接的Socket对象）不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(socket);

        // [中文导读] 把已连接 Socket 写入对应描述，之后内存/通道描述交换与一致性交换可使用同一连接。
        // [中文导读] [AllReduce逐行 S563] 设置基础层通道描述数组、本批条目下标为/按`reinterpret_cast<HcommSocket>(socket)`（已连接的Socket对象）。
        hcommDescs[i].socket = reinterpret_cast<HcommSocket>(socket);

        // [中文导读] [AllReduce逐行 S565] 记录MyRank::BatchGetSocketsForChannels的状态/性能诊断；日志本身不执行传输。
        HCCL_INFO(
            // [中文导读] [AllReduce逐行 S566] 为当前MyRank::BatchGetSocketsForChannels诊断/异常表达式提供格式文本，将报告本批条目下标；这一物理行没有数据搬运副作用。
            "[%s][%u/%u] socket created successfully, remoteRank[%u], socket[%p] reuseIdx[%u]", __func__, i + 1,
            // [中文导读] [AllReduce逐行 S567] 为前述多行表达式补入`channelNum, remoteRank, socket, reuseIdx)`（通道请求条数、对端域内Rank编号、已连接的Socket对象、本批通道或Socket复用槽位）；本行是参数/结构化初始化续行。
            channelNum, remoteRank, socket, reuseIdx);
        // [中文导读] [AllReduce逐行 S568] 推进/回退`reuseIdx++`（本批通道或Socket复用槽位），更新当前分片、槽位或状态重试的计数。
        reuseIdx++;
    // [中文导读] [AllReduce逐行 S569] 结束`for (uint32_t i = 0; i < channelNum; ++i)`（本批条目下标、通道请求条数）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S570] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S571] 结束MyRank::BatchGetSocketsForChannels函数体；控制流返回外层。
}

// [中文导读] [AllReduce逐行 S573] MyRank::BatchCreateSockets的接口声明：将全批服务器监听与连接获取分两阶段处理；这些参数属于本函数调用边界。
HcclResult MyRank::BatchCreateSockets(
    // [中文导读] [AllReduce逐行 S574] MyRank::BatchCreateSockets的接口声明：域级通道描述数组、通道请求条数、域和引擎组成的连接标签；这些参数属于本函数调用边界。
    const HcclChannelDesc* channelDescs, uint32_t channelNum, const std::string& socketTag,
    // [中文导读] [AllReduce逐行 S575] MyRank::BatchCreateSockets的接口声明：基础层通道描述数组；这些参数属于本函数调用边界。
    std::vector<HcommChannelDesc>& hcommDescs)
// [中文导读] [AllReduce逐行 S576] 进入MyRank::BatchCreateSockets函数体：将全批服务器监听与连接获取分两阶段处理。
{
    // [中文导读] [AllReduce逐行 S577] 检查`channelDescs`（域级通道描述数组）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(channelDescs);
    // [中文导读] [AllReduce逐行 S578] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
    CHK_PRT_RET(channelNum == 0, HCCL_ERROR("[%s] invalid param: channelNum is zero", __func__), HCCL_E_PARA);

    // [中文导读] [AllReduce逐行 S580] 准备按Rank对/端点对分组的Socket计数表的局部存储/结构描述，初始化方式以本行声明为准。
    ReuseSocketIdxMap reuseSocketIdxMap{};
    // socket服务器首先监听
    // [中文导读] [AllReduce逐行 S582] 为本批所有连接准备Socket监听；返回非成功时由检查宏立即向上传递。
    CHK_RET(BatchServerInitForChannels(channelDescs, channelNum, socketTag, reuseSocketIdxMap));
    // socket添加白名单以及进行连接，获取最后的socket
    // [中文导读] [AllReduce逐行 S584] 为本批连接取得Socket及端口/角色信息；返回非成功时由检查宏立即向上传递。
    CHK_RET(BatchGetSocketsForChannels(channelDescs, channelNum, socketTag, hcommDescs, reuseSocketIdxMap));
    // [中文导读] [AllReduce逐行 S585] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S586] 结束MyRank::BatchCreateSockets函数体；控制流返回外层。
}

// [中文导读] [AllReduce逐行 S588] MyRank::BatchExchangeAndCheckConsistency的接口声明：仅 950 在已建 Socket 上交换域配置及上层登记的一致性信息；这些参数属于本函数调用边界。
HcclResult MyRank::BatchExchangeAndCheckConsistency(
    // [中文导读] [AllReduce逐行 S589] MyRank::BatchExchangeAndCheckConsistency的接口声明：域级通道描述数组、基础层通道描述数组、通道请求条数；这些参数属于本函数调用边界。
    const HcclChannelDesc* channelDescs, const std::vector<HcommChannelDesc>& hcommDescs, uint32_t channelNum,
    // [中文导读] [AllReduce逐行 S590] MyRank::BatchExchangeAndCheckConsistency的接口声明：请求的通信引擎；这些参数属于本函数调用边界。
    const std::vector<std::pair<u32, u32>>& newChannels, CommEngine engine)
// [中文导读] [AllReduce逐行 S591] 进入MyRank::BatchExchangeAndCheckConsistency函数体：仅 950 在已建 Socket 上交换域配置及上层登记的一致性信息。
{
    // [中文导读] [AllReduce逐行 S592] 检查`channelDescs`（域级通道描述数组）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(channelDescs);
    // [中文导读] [AllReduce逐行 S593] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
    CHK_PRT_RET(channelNum == 0, HCCL_ERROR("[%s] invalid param: channelNum is zero", __func__), HCCL_E_PARA);

    // 与非共享路径 MyRank::CreateChannels 一致：仅 DEV_TYPE_950 需要执行通信域一致性校验交换。
    // [中文导读] [AllReduce逐行 S596] 准备设备型号的局部存储/结构描述，初始化方式以本行声明为准。
    DevType devType;
    // [中文导读] [AllReduce逐行 S597] 读取设备型号用于新旧/协议分支选择；返回非成功时由检查宏立即向上传递。
    CHK_RET(hrtGetDeviceType(devType));
    // [中文导读] 本实现仅在 950 执行域一致性交换；其他设备直接成功返回，不能据此推断已跨 Rank 比较。
    // [中文导读] [AllReduce逐行 S599] 仅当`(devType != DevType::DEV_TYPE_950)`（设备型号）成立时进入此分支。
    if (devType != DevType::DEV_TYPE_950) {
        // [中文导读] [AllReduce逐行 S600] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S601] 结束`if (devType != DevType::DEV_TYPE_950)`（设备型号）分支/循环；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S603] 设置auto startConsistency为/按`std::chrono::steady_clock::now()`；调用std::chrono::steady_clock::now。
    auto startConsistency = std::chrono::steady_clock::now();
    // [中文导读] [AllReduce逐行 S604] 仅950经现有Socket交换域配置/上层一致性描述；返回非成功时由检查宏立即向上传递。
    CHK_RET(exchangeInfoMgr_.BatchExchangeAndCheckConsistency(
        // [中文导读] [AllReduce逐行 S605] 为仅950经现有Socket交换域配置/上层一致性描述补入`channelDescs, hcommDescs, channelNum, newChannels, collCommConfigConsistency_, engine))`（域级通道描述数组、基础层通道描述数组、通道请求条数、请求的通信引擎）；本行是参数/结构化初始化续行。
        channelDescs, hcommDescs, channelNum, newChannels, collCommConfigConsistency_, engine));
    // [中文导读] [AllReduce逐行 S606] 设置auto endConsistency为/按`std::chrono::steady_clock::now()`；调用std::chrono::steady_clock::now。
    auto endConsistency = std::chrono::steady_clock::now();
    // [中文导读] [AllReduce逐行 S607] 准备`auto durationConsistency`的局部存储/结构描述，初始化方式以本行声明为准。
    auto durationConsistency
        // [中文导读] [AllReduce逐行 S608] 调用count。
        = std::chrono::duration_cast<std::chrono::microseconds>(endConsistency - startConsistency).count();
    // [中文导读] [AllReduce逐行 S609] 记录MyRank::BatchExchangeAndCheckConsistency的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S610] 为当前MyRank::BatchExchangeAndCheckConsistency诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
        "[MyRank][%s] BatchExchangeAndCheckConsistency Time Elapsed [%lld]us, channelNum [%u]", __func__,
        // [中文导读] [AllReduce逐行 S611] 为前述多行表达式补入`durationConsistency, channelNum)`（通道请求条数）；本行是参数/结构化初始化续行。
        durationConsistency, channelNum);
    // [中文导读] [AllReduce逐行 S612] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S613] 结束MyRank::BatchExchangeAndCheckConsistency函数体；控制流返回外层。
}

constexpr uint32_t MEM_HANDLE_NUM_MAX = 256; // memHandleNum的默认限制最大为256
constexpr uint32_t NOTIFY_NUM_MAX = 64;      // notifynum 的默认限制最大为64

static bool HasDuplicateChannelDesc(const HcclChannelDesc* channelDesc, uint32_t channelNum)
{
    // 通过localEndpoint, remoteEndpoint 和 channelProtocol 判断是否有重复的 channelDesc
    std::unordered_map<EndpointDescPair, std::unordered_set<int32_t>, EndpointDescPairHash, EndpointDescPairEqual>
        descSet;
    descSet.reserve(channelNum);
    for (u32 index = 0; index < channelNum; ++index) {
        EndpointDescPair endpointPair{channelDesc[index].localEndpoint, channelDesc[index].remoteEndpoint};
        auto& protocolSet = descSet[std::move(endpointPair)];
        const int32_t protocol = static_cast<int32_t>(channelDesc[index].channelProtocol);
        if (!protocolSet.insert(protocol).second) {
            return true;
        }
    }
    return false;
}

// [中文导读] [AllReduce逐行 S635] MyRank::CheckChannelParam的接口声明：请求的通信引擎、当前通道描述、通道请求条数；这些参数属于本函数调用边界。
HcclResult MyRank::CheckChannelParam(CommEngine engine, const HcclChannelDesc* channelDesc, uint32_t channelNum) const
// [中文导读] [AllReduce逐行 S636] 进入MyRank::CheckChannelParam函数体：检查通道内存/通知规模、CPU 协议和 CCU 重复描述；以源码实际索引为准。
{
    // [中文导读] [AllReduce逐行 S637] 按`(u32 index = 0; index < channelNum; ++index)`（条目下标、通道请求条数）遍历本批条目/分片；各次处理保持数组对应关系。
    for (u32 index = 0; index < channelNum; ++index) {
        // [中文导读] [AllReduce逐行 S638] 仅当`(engine == COMM_ENGINE_AIV)`（请求的通信引擎）成立时进入此分支。
        if (engine == COMM_ENGINE_AIV) {
            // [中文导读] [AllReduce逐行 S639] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
            CHK_PRT_RET(
                // [中文导读] [AllReduce逐行 S640] 本物理行访问channelDesc首项字段（未使用index），用于对应内存/通知检查或诊断；不能据此称为整批逐项校验。
                (channelDesc->memHandleNum > MEM_HANDLE_NUM_MAX),
                // [中文导读] [AllReduce逐行 S641] 记录MyRank::CheckChannelParam的错误诊断；日志本身不执行传输。
                HCCL_ERROR(
                    // [中文导读] [AllReduce逐行 S642] 为当前MyRank::CheckChannelParam诊断/异常表达式提供格式文本，将报告条目下标；这一物理行没有数据搬运副作用。
                    "[%s]Channeldesc[%u] invalid memHandleNum, memHandleNum[%u], max channel num[%u]", __func__, index,
                    // [中文导读] [AllReduce逐行 S643] 本物理行访问channelDesc首项字段（未使用index），用于对应内存/通知检查或诊断；不能据此称为整批逐项校验。
                    channelDesc->memHandleNum, MEM_HANDLE_NUM_MAX),
                // [中文导读] [AllReduce逐行 S644] 记录MyRank::CheckChannelParam的状态/性能诊断；日志本身不执行传输。
                HCCL_E_PARA);
            // [中文导读] [AllReduce逐行 S645] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
            CHK_PRT_RET(
                // [中文导读] [AllReduce逐行 S646] 本物理行访问channelDesc首项字段（未使用index），用于对应内存/通知检查或诊断；不能据此称为整批逐项校验。
                (channelDesc->memHandleNum != 0 && channelDesc->memHandles == nullptr),
                // [中文导读] [AllReduce逐行 S647] 记录MyRank::CheckChannelParam的错误诊断，字段包含条目下标；日志本身不执行传输。
                HCCL_ERROR("[%s]Channeldesc[%u] invalid memHandles, memHandles is null", __func__, index), HCCL_E_PARA);
        // [中文导读] [AllReduce逐行 S648] 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。
        } else {
            // [中文导读] [AllReduce逐行 S649] 本物理行访问channelDesc首项字段（未使用index），用于对应内存/通知检查或诊断；不能据此称为整批逐项校验。
            if (channelDesc->memHandleNum != 0) {
                // [中文导读] [AllReduce逐行 S650] 记录MyRank::CheckChannelParam的警告诊断；日志本身不执行传输。
                HCCL_WARNING(
                    // [中文导读] [AllReduce逐行 S651] 为当前MyRank::CheckChannelParam诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                    "[%s]Channeldesc[%u] memHandleNum[%u] is non-zero, memHandle exchange is not supported.", __func__,
                    // [中文导读] [AllReduce逐行 S652] 本物理行访问channelDesc首项字段（未使用index），用于对应内存/通知检查或诊断；不能据此称为整批逐项校验。
                    index, channelDesc->memHandleNum);
            // [中文导读] [AllReduce逐行 S653] 结束`if (channelDesc->memHandleNum != 0)`（当前通道描述的memHandleNum字段）分支/循环；控制流返回外层。
            }
        // [中文导读] [AllReduce逐行 S654] 结束`if (engine == COMM_ENGINE_AIV)`（请求的通信引擎）分支/循环；控制流返回外层。
        }

        // HcclChannelAcquire 不支持 CPU+UB
        // [中文导读] [AllReduce逐行 S657] 仅当`(engine == COMM_ENGINE_CPU && channelDesc[index].channelProtocol != COMM_PROTOCOL_ROCE)`（请求的通信引擎、当前通道描述、条目下标）成立时进入此分支。
        if (engine == COMM_ENGINE_CPU && channelDesc[index].channelProtocol != COMM_PROTOCOL_ROCE) {
            // [中文导读] [AllReduce逐行 S658] 记录MyRank::CheckChannelParam的错误诊断；日志本身不执行传输。
            HCCL_ERROR(
                // [中文导读] [AllReduce逐行 S659] 为当前MyRank::CheckChannelParam诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                "[%s] Channeldesc[%u/%u] COMM_ENGINE_CPU only support COMM_PROTOCOL_ROCE, but got protocol[%d]",
                // [中文导读] [AllReduce逐行 S660] 为前述多行表达式补入`__func__, index, channelNum, channelDesc[index].channelProtocol)`（条目下标、通道请求条数、当前通道描述）；本行是参数/结构化初始化续行。
                __func__, index, channelNum, channelDesc[index].channelProtocol);
            // [中文导读] [AllReduce逐行 S661] 返回HCCL_E_PARA，表示参数不满足此分支要求；此路径停止本函数的后续处理。
            return HCCL_E_PARA;
        // [中文导读] [AllReduce逐行 S662] 结束`if (engine == COMM_ENGINE_CPU && channelDesc[index].channelProtocol != COMM_PROTOCOL_ROCE)`（请求的通信引擎、当前通道描述、条目下标）分支/循环；控制流返回外层。
        }
        // [中文导读] [AllReduce逐行 S663] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
        CHK_PRT_RET(
            // [中文导读] [AllReduce逐行 S664] 本物理行访问channelDesc首项字段（未使用index），用于对应内存/通知检查或诊断；不能据此称为整批逐项校验。
            channelDesc->notifyNum > NOTIFY_NUM_MAX,
            // [中文导读] [AllReduce逐行 S665] 记录MyRank::CheckChannelParam的错误诊断；日志本身不执行传输。
            HCCL_ERROR(
                // [中文导读] [AllReduce逐行 S666] 为当前MyRank::CheckChannelParam诊断/异常表达式提供格式文本，将报告条目下标；这一物理行没有数据搬运副作用。
                "[%s]Channeldesc[%u] invalid notifyNum [%u], max notify num[%u]", __func__, index,
                // [中文导读] [AllReduce逐行 S667] 本物理行访问channelDesc首项字段（未使用index），用于对应内存/通知检查或诊断；不能据此称为整批逐项校验。
                channelDesc->notifyNum, NOTIFY_NUM_MAX),
            // [中文导读] [AllReduce逐行 S668] 记录MyRank::CheckChannelParam的状态/性能诊断；日志本身不执行传输。
            HCCL_E_PARA);
    // [中文导读] [AllReduce逐行 S669] 结束`for (u32 index = 0; index < channelNum; ++index)`（条目下标、通道请求条数）分支/循环；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S671] 仅当`(engine == COMM_ENGINE_CCU)`（请求的通信引擎）成立时进入此分支。
    if (engine == COMM_ENGINE_CCU) {
        // [中文导读] [AllReduce逐行 S672] 仅当`(HasDuplicateChannelDesc(channelDesc, channelNum))`（当前通道描述、通道请求条数）成立时进入此分支；调用HasDuplicateChannelDesc，使用当前通道描述、通道请求条数。
        if (HasDuplicateChannelDesc(channelDesc, channelNum)) {
            // [中文导读] [AllReduce逐行 S673] 记录MyRank::CheckChannelParam的错误诊断；日志本身不执行传输。
            HCCL_ERROR("[%s]Duplicate channelDesc found in CCU engine.", __func__);
            // [中文导读] [AllReduce逐行 S674] 返回HCCL_E_NOT_SUPPORT，表示当前引擎/设备/协议不支持此路径；此路径停止本函数的后续处理。
            return HCCL_E_NOT_SUPPORT;
        // [中文导读] [AllReduce逐行 S675] 结束`if (HasDuplicateChannelDesc(channelDesc, channelNum))`（当前通道描述、通道请求条数）分支/循环；控制流返回外层。
        }
    // [中文导读] [AllReduce逐行 S676] 结束`if (engine == COMM_ENGINE_CCU)`（请求的通信引擎）分支/循环；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S678] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S679] 结束MyRank::CheckChannelParam函数体；控制流返回外层。
}

// 批量创建channels，如果CCU资源不足（如Xn, Cke, channel ctx, jetty ctx, wqebb）会失败，返回HCCL_E_UNAVAIL
// [中文导读] 一批通道的本地资源准备：取得/复用Endpoint、准备监听、注册所需内存，再取得EndpointPair槽位。
// [中文导读] EndpointPair按Engine与槽位决定复用还是HcommCollectiveChannelCreate，数量由请求决定。
// [中文导读] 创建句柄与等待连接完成是两个阶段，后者在CreateChannels中的BatchConnectChannels处理。
// [中文导读] [AllReduce逐行 S685] MyRank::BatchCreateChannels的接口声明：逐请求取得本地 Endpoint、注册/选择内存、定位通道槽位并建立 Host 句柄索引；资源不足回滚新建项；这些参数属于本函数调用边界。
HcclResult MyRank::BatchCreateChannels(
    // [中文导读] [AllReduce逐行 S686] MyRank::BatchCreateChannels的接口声明：请求的通信引擎、域级通道描述数组、通道请求条数；这些参数属于本函数调用边界。
    CommEngine engine, const HcclChannelDesc* channelDescs, uint32_t channelNum,
    // [中文导读] [AllReduce逐行 S687] MyRank::BatchCreateChannels的接口声明：基础层通道描述数组、引擎侧通道句柄数组；这些参数属于本函数调用边界。
    std::vector<HcommChannelDesc>& hcommDescs, ChannelHandle* channelHandles,
    // [中文导读] [AllReduce逐行 S688] MyRank::BatchCreateChannels的接口声明：保持各通道注册句柄数组生命周期的容器；这些参数属于本函数调用边界。
    std::vector<std::vector<MemHandle>>& allHandles)
// [中文导读] [AllReduce逐行 S689] 进入MyRank::BatchCreateChannels函数体：逐请求取得本地 Endpoint、注册/选择内存、定位通道槽位并建立 Host 句柄索引；资源不足回滚新建项。
{
    // [中文导读] [AllReduce逐行 S690] 检查`channelDescs`（域级通道描述数组）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(channelDescs);
    // [中文导读] [AllReduce逐行 S691] 检查`channelHandles`（引擎侧通道句柄数组）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(channelHandles);
    // [中文导读] [AllReduce逐行 S692] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
    CHK_PRT_RET(channelNum == 0, HCCL_ERROR("[%s] invalid param: channelNum is zero", __func__), HCCL_E_PARA);

    // 持锁保护 newChannels_/handleToEpPair_；失败路径调用的 DestroyNewChannels 由本函数持锁，内部不再加锁
    // [中文导读] [AllReduce逐行 S695] 调用lock；保持声明的局部对象用于后续处理。
    std::lock_guard<std::mutex> lock(channelIndexMtx_);

    // [中文导读] [AllReduce逐行 S697] 设置本端Rank编号为/按`rankId_`（本端域内Rank编号）。
    uint32_t localRank = rankId_;
    // [中文导读] [AllReduce逐行 S698] 检查`commMems_`（域内存管理器）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_SMART_PTR_NULL(commMems_);
    // [中文导读] [AllReduce逐行 S699] 检查`endpointMgr_`（端点缓存管理器）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(endpointMgr_);
    // [中文导读] [AllReduce逐行 S700] 声明按RankPair→CommEngine→EndpointPair逐级记录本批reuseIdx的临时映射类型，下一行给出变量名。
    std::unordered_map<RankPair*, std::unordered_map<CommEngine, std::unordered_map<hcomm::EndpointPair*, u32>>>
        // [中文导读] 本批复用计数按 RankPair、引擎和 EndpointPair 分组，重复请求依次对应不同 Channel 槽位。
        // [中文导读] [AllReduce逐行 S702] 准备按Rank/引擎/端点对分组的本批槽位计数表的局部存储/结构描述，初始化方式以本行声明为准。
        reuseChannelIdxMap{};

    // 记录本轮新申请的channel
    // [中文导读] [AllReduce逐行 S705] 调用clear，使用本轮新建通道列表的clear字段；传入/处理本轮新建通道列表的clear字段。
    newChannels_.clear();
    // [中文导读] [AllReduce逐行 S706] 设置本批是否未遇资源不足为/按`true`。
    bool isAllSuccess = true;

    // [中文导读] [AllReduce逐行 S708] 按`(uint32_t i = 0; i < channelNum; ++i)`（本批条目下标、通道请求条数）遍历本批条目/分片；各次处理保持数组对应关系。
    for (uint32_t i = 0; i < channelNum; ++i) {
        // [中文导读] [AllReduce逐行 S709] 设置本地端点描述为/按`channelDescs[i].localEndpoint`（域级通道描述数组、本批条目下标）。
        const EndpointDesc& localEndpointDesc = channelDescs[i].localEndpoint;
        // [中文导读] [AllReduce逐行 S710] 设置远端端点描述为/按`channelDescs[i].remoteEndpoint`（域级通道描述数组、本批条目下标）。
        const EndpointDesc& remoteEndpointDesc = channelDescs[i].remoteEndpoint;
        // [中文导读] [AllReduce逐行 S711] 设置对端域内Rank编号为/按`channelDescs[i].remoteRank`（域级通道描述数组、本批条目下标）。
        uint32_t remoteRank = channelDescs[i].remoteRank;

        // [中文导读] [AllReduce逐行 S713] 记录MyRank::BatchCreateChannels的状态/性能诊断；日志本身不执行传输。
        HCCL_INFO(
            // [中文导读] [AllReduce逐行 S714] 为当前MyRank::BatchCreateChannels诊断/异常表达式提供格式文本，将报告本批条目下标、通道请求条数；这一物理行没有数据搬运副作用。
            "[%s][%u/%u] remoteRank[%u] localProtocol[%d] remoteProtocol[%d] engine[%s]", __func__, i + 1, channelNum,
            // [中文导读] [AllReduce逐行 S715] 为前述多行表达式补入`remoteRank, localEndpointDesc.protocol, remoteEndpointDesc.protocol,`（对端域内Rank编号、本地端点描述的protocol字段、远端端点描述的protocol字段）；本行是参数/结构化初始化续行。
            remoteRank, localEndpointDesc.protocol, remoteEndpointDesc.protocol,
            // [中文导读] [AllReduce逐行 S716] 为把枚举转换成诊断名称补入`GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str())`（请求的通信引擎）；本行是参数/结构化初始化续行。
            GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());

        // [中文导读] [AllReduce逐行 S718] 设置本地端点句柄为/按`nullptr`。
        EndpointHandle epHandle = nullptr;
        // [中文导读] 取得当前请求的本地 Endpoint；缓存命中时沿用对象，首次请求才实际创建端点。
        // [中文导读] [AllReduce逐行 S720] 设置当前调用状态为/按`endpointMgr_->Get(localEndpointDesc, epHandle)`（端点缓存管理器的Get字段、本地端点描述、本地端点句柄）；按对应缓存键取得/创建资源。
        auto ret = endpointMgr_->Get(localEndpointDesc, epHandle);
        // [中文导读] [AllReduce逐行 S721] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
        CHK_PRT_RET(
            // [中文导读] [AllReduce逐行 S722] 向条件错误检查提供`ret != HCCL_SUCCESS,`（当前调用状态），用于确定触发条件或形成对应诊断。
            ret != HCCL_SUCCESS,
            // [中文导读] [AllReduce逐行 S723] 记录MyRank::BatchCreateChannels的错误诊断；日志本身不执行传输。
            HCCL_ERROR(
                // [中文导读] [AllReduce逐行 S724] 为当前MyRank::BatchCreateChannels诊断/异常表达式提供格式文本，将报告本批条目下标、对端域内Rank编号；这一物理行没有数据搬运副作用。
                "[%s] failed to get endpoint, channelIndex[%u], remoteRank[%u], protocol[%d]", __func__, i, remoteRank,
                // [中文导读] [AllReduce逐行 S725] 为前述多行表达式补入`localEndpointDesc.protocol),`（本地端点描述的protocol字段）；本行是参数/结构化初始化续行。
                localEndpointDesc.protocol),
            // [中文导读] [AllReduce逐行 S726] 指定条件命中时要返回的错误状态`ret)`（当前调用状态），未命中则继续原处理路径。
            ret);
        // [中文导读] [AllReduce逐行 S727] 检查`epHandle`（本地端点句柄）不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(epHandle);

        // 启动监听
        // [中文导读] [AllReduce逐行 S730] 设置连接监听端口为/按`0`。
        uint32_t listenPort = 0;
        // [中文导读] [AllReduce逐行 S731] 按Rank/位置类型取得本地设备监听端口；返回非成功时由检查宏立即向上传递。
        CHK_RET(GetDevicePortInternal(localRank, &listenPort, localEndpointDesc.loc.locType));
        // [中文导读] 端口仍是默认值时尝试使用环境配置端口范围的首个起点，随后交 Endpoint 启动监听。
        // [中文导读] [AllReduce逐行 S733] 仅当`(listenPort == Hccl::DEFAULT_VALUE_TCPPORT)`（连接监听端口）成立时进入此分支。
        if (listenPort == Hccl::DEFAULT_VALUE_TCPPORT) {
            // [中文导读] [AllReduce逐行 S734] 设置auto portRanges为/按`Hccl::EnvConfig::GetInstance().GetHostNicConfig().GetDeviceSocketPortRange()`；取得该管理器单例。
            auto portRanges = Hccl::EnvConfig::GetInstance().GetHostNicConfig().GetDeviceSocketPortRange();
            // [中文导读] [AllReduce逐行 S735] 仅当`(!portRanges.empty())`成立时进入此分支；检查容器是否没有登记项。
            if (!portRanges.empty()) {
                // [中文导读] [AllReduce逐行 S736] 设置连接监听端口为/按`portRanges[0].min`。
                listenPort = portRanges[0].min;
                // [中文导读] [AllReduce逐行 S737] 记录MyRank::BatchCreateChannels的状态/性能诊断；日志本身不执行传输。
                HCCL_INFO(
                    // [中文导读] [AllReduce逐行 S738] 为当前MyRank::BatchCreateChannels诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                    "[%s] listenPort is default[%u], use port[%u] from HCCL_NPU_SOCKET_PORT_RANGE", __func__,
                    // [中文导读] [AllReduce逐行 S739] 为前述多行表达式补入`Hccl::DEFAULT_VALUE_TCPPORT, listenPort)`（连接监听端口）；本行是参数/结构化初始化续行。
                    Hccl::DEFAULT_VALUE_TCPPORT, listenPort);
            // [中文导读] [AllReduce逐行 S740] 结束`if (!portRanges.empty())`分支/循环；控制流返回外层。
            }
        // [中文导读] [AllReduce逐行 S741] 结束`if (listenPort == Hccl::DEFAULT_VALUE_TCPPORT)`（连接监听端口）分支/循环；控制流返回外层。
        }
        // [中文导读] [AllReduce逐行 S742] 通过基础Endpoint接口启动监听；返回非成功时由检查宏立即向上传递。
        CHK_RET(static_cast<HcclResult>(HcommEndpointStartListen(epHandle, listenPort, nullptr)));

        // [中文导读] [AllReduce逐行 S744] 记录MyRank::BatchCreateChannels的状态/性能诊断；日志本身不执行传输。
        HCCL_INFO(
            // [中文导读] [AllReduce逐行 S745] 为当前MyRank::BatchCreateChannels诊断/异常表达式提供格式文本，将报告本批条目下标、通道请求条数、对端域内Rank编号、本地端点句柄；这一物理行没有数据搬运副作用。
            "[%s][%u/%u] remoteRank[%u] epHandle[%p] protocol[%d]", __func__, i + 1, channelNum, remoteRank, epHandle,
            // [中文导读] [AllReduce逐行 S746] 为前述多行表达式补入`localEndpointDesc.protocol)`（本地端点描述的protocol字段）；本行是参数/结构化初始化续行。
            localEndpointDesc.protocol);

        // [中文导读] allHandles保存本Channel所需注册句柄，hcommDescs引用它参与后续资源描述交换。
        // [中文导读] exchangeAllMems=false表示按显式列表选择，避免把端点上全部内存隐式打包。
        // 注册内存
        // [中文导读] [AllReduce逐行 S751] 将域内存注册到Endpoint再挑选本Channel交换句柄；返回非成功时由检查宏立即向上传递。
        CHK_RET(PrepareMemHandles(epHandle, channelDescs[i].memHandles, channelDescs[i].memHandleNum, allHandles[i]));
        // [中文导读] [AllReduce逐行 S752] 记录MyRank::BatchCreateChannels的状态/性能诊断；日志本身不执行传输。
        HCCL_INFO(
            // [中文导读] [AllReduce逐行 S753] 为当前MyRank::BatchCreateChannels诊断/异常表达式提供格式文本，将报告本批条目下标、通道请求条数、对端域内Rank编号；这一物理行没有数据搬运副作用。
            "[%s][%u/%u] remoteRank[%u] got %zu user memory handles", __func__, i + 1, channelNum, remoteRank,
            // [中文导读] [AllReduce逐行 S754] 为读取容器登记项数补入`allHandles[i].size())`（保持各通道注册句柄数组生命周期的容器、本批条目下标）；本行是参数/结构化初始化续行。
            allHandles[i].size());

        // [中文导读] [AllReduce逐行 S756] 设置基础层通道描述数组、本批条目下标为/按`false`。
        hcommDescs[i].exchangeAllMems = false;
        // [中文导读] [AllReduce逐行 S757] 设置基础层通道描述数组、本批条目下标为/按`allHandles[i].data()`（保持各通道注册句柄数组生命周期的容器、本批条目下标）；调用data，使用保持各通道注册句柄数组生命周期的容器、本批条目下标。
        hcommDescs[i].memHandles = allHandles[i].data();
        // [中文导读] [AllReduce逐行 S758] 设置基础层通道描述数组、本批条目下标为/按`allHandles[i].size()`（保持各通道注册句柄数组生命周期的容器、本批条目下标）；读取容器登记项数。
        hcommDescs[i].memHandleNum = allHandles[i].size();

        // [中文导读] [AllReduce逐行 S760] 设置本端/对端Endpoint资源对为/按`nullptr`。
        hcomm::EndpointPair* endpointPair = nullptr;
        // [中文导读] [AllReduce逐行 S761] 设置Rank对缓存键为/按`std::make_pair(localRank, remoteRank)`（本端Rank编号、对端域内Rank编号）；调用std::make_pair，使用本端Rank编号、对端域内Rank编号。
        RankIdPair rankIdPair = std::make_pair(localRank, remoteRank);
        // [中文导读] [AllReduce逐行 S762] 设置端点对缓存键为/按`std::make_pair(localEndpointDesc, remoteEndpointDesc)`（本地端点描述、远端端点描述）；调用std::make_pair，使用本地端点描述、远端端点描述。
        EndpointDescPair endpointDescPair = std::make_pair(localEndpointDesc, remoteEndpointDesc);
        // [中文导读] [AllReduce逐行 S763] 设置本端/对端Rank资源对为/按`nullptr`。
        RankPair* rankPair = nullptr;
        // [中文导读] [AllReduce逐行 S764] 按对应缓存键取得/创建资源；返回非成功时由检查宏立即向上传递。
        CHK_RET(rankPairMgr_->Get(rankIdPair, rankPair));
        // [中文导读] [AllReduce逐行 S765] 检查`rankPair`（本端/对端Rank资源对）不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(rankPair);
        // [中文导读] [AllReduce逐行 S766] 按两端描述取得或建立EndpointPair；返回非成功时由检查宏立即向上传递。
        CHK_RET(rankPair->GetEndpointPair(endpointDescPair, endpointPair));
        // [中文导读] [AllReduce逐行 S767] 检查`endpointPair`（本端/对端Endpoint资源对）不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(endpointPair);

        // [中文导读] [AllReduce逐行 S769] 仅当`(reuseChannelIdxMap.find(rankPair) == reuseChannelIdxMap.end())`（按Rank/引擎/端点对分组的本批槽位计数表的find字段、本端/对端Rank资源对、按Rank/引擎/端点对分组的本批槽位计数表的end字段）成立时进入此分支；调用find, end，使用按Rank/引擎/端点对分组的本批槽位计数表的find字段、本端/对端Rank资源对、按Rank/引擎/端点对分组的本批槽位计数表的end字段。
        if (reuseChannelIdxMap.find(rankPair) == reuseChannelIdxMap.end()) {
            // [中文导读] [AllReduce逐行 S770] 准备`std::unordered_map<CommEngine, std::unordered_map<hcomm::EndpointPair*, u32>> engine2EndpointPairMap{}`的局部存储/结构描述，初始化方式以本行声明为准。
            std::unordered_map<CommEngine, std::unordered_map<hcomm::EndpointPair*, u32>> engine2EndpointPairMap{};
            // [中文导读] [AllReduce逐行 S771] 准备`std::unordered_map<hcomm::EndpointPair*, u32> endpointPair2Idx{}`的局部存储/结构描述，初始化方式以本行声明为准。
            std::unordered_map<hcomm::EndpointPair*, u32> endpointPair2Idx{};
            // [中文导读] [AllReduce逐行 S772] 调用emplace，使用本端/对端Endpoint资源对；传入/处理本端/对端Endpoint资源对。
            endpointPair2Idx.emplace(endpointPair, 0);
            // [中文导读] [AllReduce逐行 S773] 调用emplace，使用请求的通信引擎；传入/处理请求的通信引擎。
            engine2EndpointPairMap.emplace(engine, endpointPair2Idx);
            // [中文导读] [AllReduce逐行 S774] 调用emplace，使用按Rank/引擎/端点对分组的本批槽位计数表的emplace字段、本端/对端Rank资源对；传入/处理按Rank/引擎/端点对分组的本批槽位计数表的emplace字段、本端/对端Rank资源对。
            reuseChannelIdxMap.emplace(rankPair, engine2EndpointPairMap);
        // [中文导读] [AllReduce逐行 S775] 仅当`(reuseChannelIdxMap[rankPair].find(engine) == reuseChannelIdxMap[rankPair].end())`（按Rank/引擎/端点对分组的本批槽位计数表、本端/对端Rank资源对、请求的通信引擎）成立时进入此分支；调用find, end，使用按Rank/引擎/端点对分组的本批槽位计数表、本端/对端Rank资源对、请求的通信引擎。
        } else if (reuseChannelIdxMap[rankPair].find(engine) == reuseChannelIdxMap[rankPair].end()) {
            // [中文导读] [AllReduce逐行 S776] 准备`std::unordered_map<hcomm::EndpointPair*, u32> endpointPair2Idx{}`的局部存储/结构描述，初始化方式以本行声明为准。
            std::unordered_map<hcomm::EndpointPair*, u32> endpointPair2Idx{};
            // [中文导读] [AllReduce逐行 S777] 调用emplace，使用本端/对端Endpoint资源对；传入/处理本端/对端Endpoint资源对。
            endpointPair2Idx.emplace(endpointPair, 0);
            // [中文导读] [AllReduce逐行 S778] 调用emplace，使用按Rank/引擎/端点对分组的本批槽位计数表、本端/对端Rank资源对、请求的通信引擎；传入/处理按Rank/引擎/端点对分组的本批槽位计数表、本端/对端Rank资源对、请求的通信引擎。
            reuseChannelIdxMap[rankPair].emplace(engine, endpointPair2Idx);
        // [中文导读] [AllReduce逐行 S779] 仅当`(`成立时进入此分支。
        } else if (
            // [中文导读] [AllReduce逐行 S780] 补全本分支/循环判断的`reuseChannelIdxMap[rankPair][engine].find(endpointPair) == reuseChannelIdxMap[rankPair][engine].end())`（按Rank/引擎/端点对分组的本批槽位计数表、本端/对端Rank资源对、请求的通信引擎、本端/对端Endpoint资源对），和前面条件共同决定是否进入后续路径。
            reuseChannelIdxMap[rankPair][engine].find(endpointPair) == reuseChannelIdxMap[rankPair][engine].end()) {
            // [中文导读] [AllReduce逐行 S781] 调用emplace，使用按Rank/引擎/端点对分组的本批槽位计数表、本端/对端Rank资源对、请求的通信引擎、本端/对端Endpoint资源对；传入/处理按Rank/引擎/端点对分组的本批槽位计数表、本端/对端Rank资源对、请求的通信引擎、本端/对端Endpoint资源对。
            reuseChannelIdxMap[rankPair][engine].emplace(endpointPair, 0);
        // [中文导读] [AllReduce逐行 S782] 结束逐分片处理回调；控制流返回外层。
        }

        // [中文导读] 本地/远端端点位置类型不同的 Host–Device 连接禁用槽位复用，以 UNREUSE 请求追加新通道。
        // [中文导读] [AllReduce逐行 S785] 设置本批通道或Socket复用槽位为/按`reuseChannelIdxMap[rankPair][engine][endpointPair]`（按Rank/引擎/端点对分组的本批槽位计数表、本端/对端Rank资源对、请求的通信引擎、本端/对端Endpoint资源对）。
        u32& reuseIdx = reuseChannelIdxMap[rankPair][engine][endpointPair];
        // [中文导读] [AllReduce逐行 S786] 设置本轮槽位/数组索引为/按`reuseIdx`（本批通道或Socket复用槽位）。
        u32 idx = reuseIdx;
        /* hostNIC -- DeviceNic（transport不复用link/Channel），此流程也是新创建channel，需要计入isNewChannel */
        // [中文导读] [AllReduce逐行 S788] 仅当`(localEndpointDesc.loc.locType != remoteEndpointDesc.loc.locType)`（本地端点描述的loc.locType字段、远端端点描述的loc.locType字段）成立时进入此分支。
        if (localEndpointDesc.loc.locType != remoteEndpointDesc.loc.locType) {
            // [中文导读] [AllReduce逐行 S789] 设置本轮槽位/数组索引为/按`UNREUSE_CHANNEL_IDX`。
            idx = UNREUSE_CHANNEL_IDX;
        // [中文导读] [AllReduce逐行 S790] 结束`if (localEndpointDesc.loc.locType != remoteEndpointDesc.loc.locType)`（本地端点描述的loc.locType字段、远端端点描述的loc.locType字段）分支/循环；控制流返回外层。
        }
        // [中文导读] [AllReduce逐行 S791] 设置是否本轮新建通道为/按`(endpointPair->IsChannelNotExist(engine, reuseIdx) || (idx == UNREUSE_CHANNEL_IDX))`（本端/对端Endpoint资源对的IsChannelNotExist字段、请求的通信引擎、本批通道或Socket复用槽位、本轮槽位/数组索引）；检查所选引擎复用槽位是否不存在。
        bool isNewChannel = (endpointPair->IsChannelNotExist(engine, reuseIdx) || (idx == UNREUSE_CHANNEL_IDX));

        // CreateChannel 返回 HCCL_E_UNAVAIL 表示资源不足创建失败
        // [中文导读] [AllReduce逐行 S794] 设置当前调用状态为/按`endpointPair->CreateChannel(epHandle, engine, idx, &hcommDescs[i], channelHandles + i)`（本端/对端Endpoint资源对的CreateChannel字段、本地端点句柄、请求的通信引擎、本轮槽位/数组索引、基础层通道描述数组、本批条目下标、引擎侧通道句柄数组）；进入EndpointPair或具体Channel工厂的创建/复用逻辑。
        ret = endpointPair->CreateChannel(epHandle, engine, idx, &hcommDescs[i], channelHandles + i);
        // [中文导读] [AllReduce逐行 S795] 仅当`(ret == HCCL_E_TIMEOUT || ret == HCCL_E_INTERNAL)`（当前调用状态）成立时进入此分支。
        if (ret == HCCL_E_TIMEOUT || ret == HCCL_E_INTERNAL) {
            // [中文导读] [AllReduce逐行 S796] 设置Hccl::TlsStatus tlsStatus为/按`Hccl::TlsStatus::UNKNOWN`。
            Hccl::TlsStatus tlsStatus = Hccl::TlsStatus::UNKNOWN;
            // [中文导读] [AllReduce逐行 S797] 开始容错诊断检查；当前检查失败记录日志但继续本层处理，不覆盖主建链状态。
            CHK_PRT_CONT(
                // [中文导读] [AllReduce逐行 S798] 为取得本端Host/Device TLS状态用于建链错误诊断补入`GetLocalTlsStatus(localEndpointDesc.loc.locType, tlsStatus) != HCCL_SUCCESS,`（本地端点描述的loc.locType字段）；本行是参数/结构化初始化续行。
                GetLocalTlsStatus(localEndpointDesc.loc.locType, tlsStatus) != HCCL_SUCCESS,
                // [中文导读] [AllReduce逐行 S799] 记录MyRank::BatchCreateChannels的警告诊断；日志本身不执行传输。
                HCCL_WARNING("[GetLocalTlsStatus] Can not get TlsStatus"));
            // [中文导读] [AllReduce逐行 S800] 记录MyRank::BatchCreateChannels的错误诊断；日志本身不执行传输。
            HCCL_ERROR(
                // [中文导读] [AllReduce逐行 S801] 为当前MyRank::BatchCreateChannels诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                "[%s] failed to create channel, channelIndex[%u], localRank[%u], remoteRank[%u], protocol[%s], "
                // [中文导读] [AllReduce逐行 S802] 为当前MyRank::BatchCreateChannels诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                "tlsType[%s], tlsStatus[%s], ret[%d]",
                // [中文导读] [AllReduce逐行 S803] 为前述多行表达式补入`__func__, i, localRank, remoteRank,`（本批条目下标、本端Rank编号、对端域内Rank编号）；本行是参数/结构化初始化续行。
                __func__, i, localRank, remoteRank,
                // [中文导读] [AllReduce逐行 S804] 为调用MyRankUtils::GetCommProtocolEnumStr, c_str，使用本批条目下标、本端Rank编号、对端域内Rank编号、本地端点描述的protocol字段补入`MyRankUtils::GetCommProtocolEnumStr(localEndpointDesc.protocol).c_str(),`（本地端点描述的protocol字段）；本行是参数/结构化初始化续行。
                MyRankUtils::GetCommProtocolEnumStr(localEndpointDesc.protocol).c_str(),
                // [中文导读] [AllReduce逐行 S805] 为调用MyRankUtils::GetCommProtocolEnumStr, c_str, MyRankUtils::GetTlsTypeStr，使用本批条目下标、本端Rank编号、对端域内Rank编号、本地端点描述的protocol字段、本地端点描述的loc.locType字段补入`MyRankUtils::GetTlsTypeStr(localEndpointDesc.loc.locType), MyRankUtils::GetTlsStatusStr(tlsStatus),`（本地端点描述的loc.locType字段）；本行是参数/结构化初始化续行。
                MyRankUtils::GetTlsTypeStr(localEndpointDesc.loc.locType), MyRankUtils::GetTlsStatusStr(tlsStatus),
                // [中文导读] [AllReduce逐行 S806] 为调用MyRankUtils::GetCommProtocolEnumStr, c_str, MyRankUtils::GetTlsTypeStr，使用本批条目下标、本端Rank编号、对端域内Rank编号、本地端点描述的protocol字段、本地端点描述的loc.locType字段、当前调用状态补入`ret)`（当前调用状态）；本行是参数/结构化初始化续行。
                ret);
        // [中文导读] [AllReduce逐行 S807] 结束`if (ret == HCCL_E_TIMEOUT || ret == HCCL_E_INTERNAL)`（当前调用状态）分支/循环；控制流返回外层。
        }
        // [中文导读] 资源不足时停止扩展本批，稍后只清理本轮记录的新通道，保留之前已存在的可复用资源。
        // [中文导读] [AllReduce逐行 S809] 仅当`(ret == HCCL_E_UNAVAIL)`（当前调用状态）成立时进入此分支。
        if (ret == HCCL_E_UNAVAIL) {
            // 申请channel因资源不足失败，清理已申请的channel
            // [中文导读] [AllReduce逐行 S811] 记录MyRank::BatchCreateChannels的警告诊断；日志本身不执行传输。
            HCCL_RUN_WARNING(
                // [中文导读] [AllReduce逐行 S812] 为当前MyRank::BatchCreateChannels诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                "[%s] create channel failed, channelIndex[%u], remoteRank[%u], engine[%s], reuseIdx[%u], need clean "
                // [中文导读] [AllReduce逐行 S813] 为当前MyRank::BatchCreateChannels诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                "new channels",
                // [中文导读] [AllReduce逐行 S814] 为把枚举转换成诊断名称补入`__func__, i + 1, remoteRank, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), reuseIdx)`（本批条目下标、对端域内Rank编号、请求的通信引擎、本批通道或Socket复用槽位）；本行是参数/结构化初始化续行。
                __func__, i + 1, remoteRank, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), reuseIdx);
            // [中文导读] [AllReduce逐行 S815] 设置本批是否未遇资源不足为/按`false`。
            isAllSuccess = false;
            // [中文导读] [AllReduce逐行 S816] 退出当前循环或switch路径，继续其后处理；不代表其他执行流已经完成。
            break;
        // [中文导读] [AllReduce逐行 S817] 结束`if (ret == HCCL_E_UNAVAIL)`（当前调用状态）分支/循环；控制流返回外层。
        }
        // 记录新申请的channel信息，用于清理临时资源
        // [中文导读] [AllReduce逐行 S819] 仅当`(isNewChannel)`（是否本轮新建通道）成立时进入此分支。
        if (isNewChannel) {
            // [中文导读] [AllReduce逐行 S820] 调用emplace_back, std::make_pair，使用本轮新建通道列表的emplace_back字段、本批条目下标、本批通道或Socket复用槽位；传入/处理本轮新建通道列表的emplace_back字段、本批条目下标、本批通道或Socket复用槽位。
            newChannels_.emplace_back(std::make_pair(i, reuseIdx));
        // [中文导读] [AllReduce逐行 S821] 结束`if (isNewChannel)`（是否本轮新建通道）分支/循环；控制流返回外层。
        }

        // [中文导读] [AllReduce逐行 S823] 仅当`(ret != HCCL_SUCCESS)`（当前调用状态）成立时进入此分支。
        if (ret != HCCL_SUCCESS) {
            // [中文导读] [AllReduce逐行 S824] 仅当`(ret != HCCL_E_TIMEOUT && ret != HCCL_E_INTERNAL)`（当前调用状态）成立时进入此分支。
            if (ret != HCCL_E_TIMEOUT && ret != HCCL_E_INTERNAL) {
                // [中文导读] [AllReduce逐行 S825] 记录MyRank::BatchCreateChannels的错误诊断；日志本身不执行传输。
                HCCL_ERROR(
                    // [中文导读] [AllReduce逐行 S826] 为当前MyRank::BatchCreateChannels诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                    "[%s] failed to create channel, channelIndex[%u], remoteRank[%u], engine[%s], reuseIndex[%u]",
                    // [中文导读] [AllReduce逐行 S827] 为把枚举转换成诊断名称补入`__func__, i + 1, remoteRank, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(),`（本批条目下标、对端域内Rank编号、请求的通信引擎）；本行是参数/结构化初始化续行。
                    __func__, i + 1, remoteRank, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(),
                    // [中文导读] [AllReduce逐行 S828] 为把枚举转换成诊断名称补入`reuseIdx)`（本批通道或Socket复用槽位）；本行是参数/结构化初始化续行。
                    reuseIdx);
            // [中文导读] [AllReduce逐行 S829] 结束`if (ret != HCCL_E_TIMEOUT && ret != HCCL_E_INTERNAL)`（当前调用状态）分支/循环；控制流返回外层。
            }
            // [中文导读] [AllReduce逐行 S830] 返回ret，表示保留当前调用的状态（可能成功或失败）；此路径停止本函数的后续处理。
            return ret;
        // [中文导读] [AllReduce逐行 S831] 结束`if (ret != HCCL_SUCCESS)`（当前调用状态）分支/循环；控制流返回外层。
        }
        // [中文导读] [AllReduce逐行 S832] 仅当`(idx != UNREUSE_CHANNEL_IDX)`（本轮槽位/数组索引）成立时进入此分支。
        if (idx != UNREUSE_CHANNEL_IDX) {
            // [中文导读] [AllReduce逐行 S833] 推进/回退`reuseIdx++`（本批通道或Socket复用槽位），更新当前分片、槽位或状态重试的计数。
            reuseIdx++;
        // [中文导读] [AllReduce逐行 S834] 结束`if (idx != UNREUSE_CHANNEL_IDX)`（本轮槽位/数组索引）分支/循环；控制流返回外层。
        }

        // 登记 handle -> EndpointPair 反查索引；真实槽位由 EndpointPair::handleToLoc_ 维护
        // [中文导读] 成功后建立 Host 句柄到 EndpointPair 的反查，供后续按句柄销毁；真实槽位由 EndpointPair 维护。
        // [中文导读] [AllReduce逐行 S838] 设置Host句柄到端点对的反查表、引擎侧通道句柄数组、本批条目下标为/按`endpointPair`（本端/对端Endpoint资源对）。
        handleToEpPair_[channelHandles[i]] = endpointPair;

        // [中文导读] [AllReduce逐行 S840] 记录MyRank::BatchCreateChannels的状态/性能诊断；日志本身不执行传输。
        HCCL_INFO(
            // [中文导读] [AllReduce逐行 S841] 为当前MyRank::BatchCreateChannels诊断/异常表达式提供格式文本，将报告本批条目下标、通道请求条数；这一物理行没有数据搬运副作用。
            "[%s][%u/%u] channel created successfully, remoteRank[%u], channelHandle[%p]", __func__, i + 1, channelNum,
            // [中文导读] [AllReduce逐行 S842] 为前述多行表达式补入`remoteRank, channelHandles[i])`（对端域内Rank编号、引擎侧通道句柄数组、本批条目下标）；本行是参数/结构化初始化续行。
            remoteRank, channelHandles[i]);
    // [中文导读] [AllReduce逐行 S843] 结束`for (uint32_t i = 0; i < channelNum; ++i)`（本批条目下标、通道请求条数）分支/循环；控制流返回外层。
    }

    // 如果申请失败，清理endpoint pair中记录的channel handle
    // [中文导读] [AllReduce逐行 S846] 仅当`(!isAllSuccess)`（本批是否未遇资源不足）成立时进入此分支。
    if (!isAllSuccess) {
        // [中文导读] [AllReduce逐行 S847] 记录MyRank::BatchCreateChannels的警告诊断；日志本身不执行传输。
        HCCL_RUN_WARNING(
            // [中文导读] [AllReduce逐行 S848] 为当前MyRank::BatchCreateChannels诊断/异常表达式提供格式文本，将报告本轮新建通道列表的size字段；这一物理行没有数据搬运副作用。
            "[%s] create channel failed, destroy new channels num[%zu], engine[%s]", __func__, newChannels_.size(),
            // [中文导读] [AllReduce逐行 S849] 为读取容器登记项数；把枚举转换成诊断名称补入`GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str())`（请求的通信引擎）；本行是参数/结构化初始化续行。
            GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
        // [中文导读] [AllReduce逐行 S850] 逆序销毁本轮新建Channel以回滚资源不足路径；返回非成功时由检查宏立即向上传递。
        CHK_RET(DestroyNewChannels(engine, channelDescs, newChannels_));
        // [中文导读] [AllReduce逐行 S851] 返回HCCL_E_UNAVAIL，表示资源不足，保留上层回退语义；此路径停止本函数的后续处理。
        return HCCL_E_UNAVAIL;
    // [中文导读] [AllReduce逐行 S852] 结束`if (!isAllSuccess)`（本批是否未遇资源不足）分支/循环；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S854] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S855] 结束MyRank::BatchCreateChannels函数体；控制流返回外层。
}

HcclResult MyRank::DestroyNewChannels(
    CommEngine engine, const HcclChannelDesc* channelDescs, const std::vector<std::pair<u32, u32>>& newChannels)
{
    HcclResult firstErr = HCCL_SUCCESS;
    uint32_t localRank = rankId_;
    // [中文导读] 按创建顺序逆向回滚，避免删除向量槽位后导致尚待销毁的后续下标前移。
    for (auto idxPairIter = std::rbegin(newChannels); idxPairIter != std::rend(newChannels);
         ++idxPairIter) { // 由于新申请的在申请过的后面，所以要从后往前找reuseIdx销毁
        auto idxPair = *idxPairIter;
        const EndpointDesc& localEndpointDesc = channelDescs[idxPair.first].localEndpoint;
        const EndpointDesc& remoteEndpointDesc = channelDescs[idxPair.first].remoteEndpoint;
        uint32_t remoteRank = channelDescs[idxPair.first].remoteRank;
        hcomm::EndpointPair* endpointPair = nullptr;
        RankIdPair rankIdPair = std::make_pair(localRank, remoteRank);
        EndpointDescPair endpointDescPair = std::make_pair(localEndpointDesc, remoteEndpointDesc);
        RankPair* rankPair = nullptr;
        CHK_RET(rankPairMgr_->Get(rankIdPair, rankPair));
        CHK_PTR_NULL(rankPair);
        CHK_RET(rankPair->GetEndpointPair(endpointDescPair, endpointPair));
        CHK_PTR_NULL(endpointPair);
        // DestroyChannel 会 erase 向量导致下标变化, 需先取出 handle
        ChannelHandle handleToErase = 0;
        endpointPair->GetChannelHandle(engine, idxPair.second, handleToErase);
        // 单个 channel 销毁失败不中断其余清理；记录首个错误，最终统一清空本次新建列表
        HcclResult destroyRet = endpointPair->DestroyChannel(engine, idxPair.second);
        if (destroyRet != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[%s] DestroyChannel failed, engine[%s] reuseIdx[%u] ret[%d], continue.", __func__,
                GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), idxPair.second, destroyRet);
            if (firstErr == HCCL_SUCCESS) {
                firstErr = destroyRet;
            }
        }
        if (handleToErase != 0) {
            handleToEpPair_.erase(handleToErase);
        }
    }
    newChannels_.clear();
    return firstErr;
}

HcclResult
MyRank::QueryOneChannel(CommEngine engine, const HcclChannelDesc& channelDesc, u32 reuseIdx, ChannelHandle& handle)
{
    // [中文导读] 先给出未命中句柄 0；RankPair 或通道槽位不存在时允许查询成功，由句柄值表达未命中。
    handle = 0;
    const RankIdPair rankIdPair = std::make_pair(rankId_, channelDesc.remoteRank);
    const EndpointDescPair endpointDescPair = std::make_pair(channelDesc.localEndpoint, channelDesc.remoteEndpoint);

    RankPair* rankPair = nullptr;
    if (rankPairMgr_->Find(rankIdPair, rankPair) != HCCL_SUCCESS || rankPair == nullptr) {
        return HCCL_SUCCESS;
    }
    hcomm::EndpointPair* epPair = nullptr;
    if (rankPair->GetEndpointPair(endpointDescPair, epPair) != HCCL_SUCCESS || epPair == nullptr) {
        return HCCL_SUCCESS;
    }
    ChannelHandle slotHandle = 0;
    if (epPair->GetChannelHandle(engine, reuseIdx, slotHandle)) {
        handle = slotHandle;
    }
    return HCCL_SUCCESS;
}

HcclResult MyRank::QueryChannels(
    CommEngine engine, const HcclChannelDesc* channelDescs, uint32_t channelNum, ChannelHandle* channels)
{
    CHK_PTR_NULL(channelDescs);
    CHK_PTR_NULL(channels);
    CHK_PRT_RET(channelNum == 0, HCCL_ERROR("[%s] invalid param: channelNum is zero", __func__), HCCL_E_PARA);

    HCCL_INFO(
        "[MyRank][%s] Enter engine[%s] channelNum[%u] rankId[%u]", __func__,
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), channelNum, rankId_);

    // 与 BatchCreateChannels 保持一致的 reuseIdx 累计逻辑
    std::unordered_map<RankIdPair, std::unordered_map<EndpointDescPair, std::unordered_map<CommEngine, u32>>>
        reuseIdxMap{};

    for (uint32_t i = 0; i < channelNum; ++i) {
        channels[i] = 0;
        const auto& channelDesc = channelDescs[i];
        uint32_t remoteRank = channelDesc.remoteRank;
        const RankIdPair rankIdPair = std::make_pair(rankId_, remoteRank);
        const EndpointDescPair endpointDescPair = std::make_pair(channelDesc.localEndpoint, channelDesc.remoteEndpoint);

        u32& reuseIdx = reuseIdxMap[rankIdPair][endpointDescPair][engine];
        u32 idx = reuseIdx;
        if (channelDesc.localEndpoint.loc.locType != channelDesc.remoteEndpoint.loc.locType) {
            idx = UNREUSE_CHANNEL_IDX;
        }

        // 仅当非 UNREUSE 且槽位存在时返回 handle
        if (idx != UNREUSE_CHANNEL_IDX) {
            (void)QueryOneChannel(engine, channelDesc, reuseIdx, channels[i]);
        }

        HCCL_INFO(
            "[MyRank][%s] [%u/%u] remoteRank[%u] exist[%s] handle[0x%llx] reuseIdx[%u] unreuse[%d]", __func__, i + 1,
            channelNum, remoteRank, channels[i] != 0 ? "yes" : "no", channels[i], reuseIdx, idx == UNREUSE_CHANNEL_IDX);

        // 与 BatchCreateChannels 一致: 非 UNREUSE 才递增 reuseIdx(引用, 直接改 map 内值)
        if (idx != UNREUSE_CHANNEL_IDX) {
            reuseIdx++;
        }
    }

    // 对发生句柄转换的引擎，经平台 H2D 反向映射把 host 句柄转换为用户实际使用的句柄
    // （device 句柄），保证 Query 返回值与 HcclChannelAcquire 出参一致
    // [中文导读] AICPU/AIV 查询命中后尝试把 Host 句柄转换成设备表示；未取得有效映射时保留已有句柄。
    if (engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS || engine == COMM_ENGINE_AIV) {
        for (uint32_t i = 0; i < channelNum; ++i) {
            if (channels[i] != 0) {
                ChannelHandle deviceHandle = 0;
                if (hcomm::ChannelProcess::ResolveHostHandleToDevice(channels[i], deviceHandle) == HCCL_SUCCESS
                    && deviceHandle != 0) {
                    channels[i] = deviceHandle;
                }
            }
        }
    }
    return HCCL_SUCCESS;
}

// 记录批量销毁过程中的首个错误与对应计数，供 DestroyOneChannel 复用
static void RecordDestroyError(HcclResult& firstErr, u32& errCnt, HcclResult err)
{
    errCnt++;
    if (firstErr == HCCL_SUCCESS) {
        firstErr = err;
    }
}

HcclResult MyRank::DestroyOneChannel(
    ChannelHandle userHandle, u32 index, HcclResult& firstErr, u32& invalidHandleCnt, u32& failedCnt)
{
    // 反查索引以 host 句柄为键：AIV/AICPU_TS 入参为 device 句柄，先经 D2H 映射解析为 host 句柄
    ChannelHandle hostHandle = userHandle;
    ChannelHandle resolved = 0;
    if (hcomm::ChannelProcess::ResolveUserHandleToHost(userHandle, resolved) == HCCL_SUCCESS && resolved != 0) {
        hostHandle = resolved;
    }
    auto it = handleToEpPair_.find(hostHandle);
    if (it == handleToEpPair_.end()) {
        HCCL_ERROR("[%s] channel handle[0x%llx] not found, channelIndex[%u].", __func__, userHandle, index);
        RecordDestroyError(firstErr, invalidHandleCnt, HCCL_E_NOT_FOUND);
        return HCCL_SUCCESS;
    }
    hcomm::EndpointPair* epPair = it->second;
    if (epPair == nullptr) {
        // 反查索引条目为空指针（异常数据），清理并按无效句柄容错
        HCCL_ERROR("[%s] channel handle[0x%llx] endpoint pair is null, channelIndex[%u].", __func__, userHandle, index);
        handleToEpPair_.erase(it);
        RecordDestroyError(firstErr, invalidHandleCnt, HCCL_E_NOT_FOUND);
        return HCCL_SUCCESS;
    }
    CommEngine engine = COMM_ENGINE_RESERVED;
    u32 reuseIdx = 0;
    if (!epPair->FindChannelLoc(hostHandle, engine, reuseIdx)) {
        HCCL_ERROR("[%s] channel handle[0x%llx] FindChannelLoc failed, channelIndex[%u].", __func__, userHandle, index);
        RecordDestroyError(firstErr, invalidHandleCnt, HCCL_E_NOT_FOUND);
        return HCCL_SUCCESS;
    }
    // 暂只支持 CCU 引擎： 其他场景的 channel 销毁无法保证资源完整释放
    // [中文导读] 当前域级单通道销毁仅接受 CCU，引擎不受支持时累积错误并让外层继续处理其他句柄。
    if (engine != COMM_ENGINE_CCU) {
        HCCL_WARNING(
            "[%s] channel handle[0x%llx] engine[%s] not supported, only CCU engine supported, channelIndex[%u].",
            __func__, userHandle, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), index);
        RecordDestroyError(firstErr, failedCnt, HCCL_E_NOT_SUPPORT);
        return HCCL_SUCCESS;
    }
    HcclResult destroyRet = epPair->DestroyChannel(engine, reuseIdx);
    if (destroyRet != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] DestroyChannel failed, handle[0x%llx] ret[%d], continue.", __func__, hostHandle, destroyRet);
        RecordDestroyError(firstErr, failedCnt, destroyRet);
        return HCCL_SUCCESS;
    }
    handleToEpPair_.erase(it);
    return HCCL_SUCCESS;
}

HcclResult MyRank::DestroyChannels(const ChannelHandle* channels, uint32_t channelNum)
{
    CHK_PTR_NULL(channels);
    CHK_PRT_RET(channelNum == 0, HCCL_ERROR("[%s] invalid param: channelNum is zero", __func__), HCCL_E_PARA);

    std::lock_guard<std::mutex> lock(channelIndexMtx_);

    HCCL_INFO("[MyRank][%s] Enter channelNum[%u] rankId[%u]", __func__, channelNum, rankId_);

    HcclResult firstErr = HCCL_SUCCESS;
    u32 invalidHandleCnt = 0;
    u32 failedCnt = 0;

    // [中文导读] 逐项处理整批并统计无效/失败句柄，最终返回首个错误；某项失败不阻断后面的清理。
    for (uint32_t i = 0; i < channelNum; ++i) {
        (void)DestroyOneChannel(channels[i], i, firstErr, invalidHandleCnt, failedCnt);
    }

    if (firstErr != HCCL_SUCCESS) {
        u32 destroyedCnt = channelNum - invalidHandleCnt - failedCnt;
        HCCL_ERROR(
            "[%s] finished with errors, total[%u] destroyed[%u] failed[%u] invalidHandle[%u] firstErr[%d].", __func__,
            channelNum, destroyedCnt, failedCnt, invalidHandleCnt, static_cast<s32>(firstErr));
        return firstErr;
    }
    return HCCL_SUCCESS;
}

// [中文导读] [AllReduce逐行 S1067] MyRank::BatchConnectChannels的接口声明：轮询全批 Channel 状态，分别处理 AGAIN、UNAVAIL、超时和一般错误；这些参数属于本函数调用边界。
HcclResult
// [中文导读] [AllReduce逐行 S1068] MyRank::BatchConnectChannels的接口声明：域级通道描述数组、引擎侧通道句柄数组、通道请求条数；这些参数属于本函数调用边界。
MyRank::BatchConnectChannels(const HcclChannelDesc* channelDescs, ChannelHandle* channelHandles, uint32_t channelNum)
// [中文导读] [AllReduce逐行 S1069] 进入MyRank::BatchConnectChannels函数体：轮询全批 Channel 状态，分别处理 AGAIN、UNAVAIL、超时和一般错误。
{
    // [中文导读] 从环境取得以秒计的建链超时，使用单调时钟衡量整批连接等待时间。
    // [中文导读] [AllReduce逐行 S1071] 设置超时秒数为/按`std::chrono::seconds(Hccl::EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut())`；取得该管理器单例。
    auto timeout = std::chrono::seconds(Hccl::EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());
    // [中文导读] [AllReduce逐行 S1072] 设置单调时钟开始时刻为/按`std::chrono::steady_clock::now()`；调用std::chrono::steady_clock::now。
    auto startTime = std::chrono::steady_clock::now();

    // [中文导读] [AllReduce逐行 S1074] 记录MyRank::BatchConnectChannels的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S1075] 为当前MyRank::BatchConnectChannels诊断/异常表达式提供格式文本，将报告通道请求条数、超时秒数的count字段；这一物理行没有数据搬运副作用。
        "[%s] start connecting channels, channelNum[%u], timeout[%lld]sec", __func__, channelNum, timeout.count());

    // [中文导读] [AllReduce逐行 S1077] 调用statusVec，使用本批状态数组存储、通道请求条数；对象涉及本批状态数组存储、通道请求条数。
    std::vector<int32_t> statusVec(channelNum, 0);
    // [中文导读] [AllReduce逐行 S1078] 设置本批通道连接状态数组为/按`statusVec.data()`（本批状态数组存储的data字段）；调用data，使用本批状态数组存储的data字段。
    int32_t* statusList = statusVec.data();
    // [中文导读] [AllReduce逐行 S1079] 设置连接状态重试计数为/按`0`。
    uint32_t retryCount = 0;
    // [中文导读] [AllReduce逐行 S1080] 在`(true)`条件下重复执行后续等待或分片处理。
    while (true) {
        // [中文导读] [AllReduce逐行 S1081] 设置当前调用状态为/按`hcomm::ChannelProcess::ChannelGetStatus(channelHandles, channelNum, statusList)`（引擎侧通道句柄数组、通道请求条数、本批通道连接状态数组）；查询各Channel的连接状态并返回全批状态。
        HcclResult ret = hcomm::ChannelProcess::ChannelGetStatus(channelHandles, channelNum, statusList);

        // 卫语句：先处理异常情况

        // 1. 检查超时
        // [中文导读] [AllReduce逐行 S1086] 仅当`((std::chrono::steady_clock::now() - startTime) >= timeout)`（单调时钟开始时刻、超时秒数）成立时进入此分支；调用std::chrono::steady_clock::now，使用单调时钟开始时刻、超时秒数。
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            // [中文导读] [AllReduce逐行 S1087] 准备已经等待的毫秒数的局部存储/结构描述，初始化方式以本行声明为准。
            auto elapsed
                // [中文导读] [AllReduce逐行 S1088] 调用std::chrono::steady_clock::now，使用单调时钟开始时刻；传入/处理单调时钟开始时刻。
                = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                      // [中文导读] [AllReduce逐行 S1089] 调用count。
                      .count();
            // [中文导读] [AllReduce逐行 S1090] 记录MyRank::BatchConnectChannels的错误诊断；日志本身不执行传输。
            HCCL_ERROR(
                // [中文导读] [AllReduce逐行 S1091] 为当前MyRank::BatchConnectChannels诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                "[%s][%s] wait socket establish timeout, channel connect timeout after %lld sec, "
                // [中文导读] [AllReduce逐行 S1092] 为当前MyRank::BatchConnectChannels诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                "channelNum[%u], elapsed[%lld]ms, retryCount[%u]",
                // [中文导读] [AllReduce逐行 S1093] 为调用c_str, c_str, count，使用超时秒数的count字段、通道请求条数、已经等待的毫秒数补入`LOG_KEYWORDS_INIT_CHANNEL.c_str(), LOG_KEYWORDS_TIMEOUT.c_str(), timeout.count(), channelNum, elapsed,`（超时秒数的count字段、通道请求条数、已经等待的毫秒数）；本行是参数/结构化初始化续行。
                LOG_KEYWORDS_INIT_CHANNEL.c_str(), LOG_KEYWORDS_TIMEOUT.c_str(), timeout.count(), channelNum, elapsed,
                // [中文导读] [AllReduce逐行 S1094] 为调用c_str, c_str, count，使用超时秒数的count字段、通道请求条数、已经等待的毫秒数、连接状态重试计数补入`retryCount)`（连接状态重试计数）；本行是参数/结构化初始化续行。
                retryCount);
            // [中文导读] [AllReduce逐行 S1095] 上报本次连接超时对应的输入错误诊断码与原因，随后仍按本函数超时状态返回。
            RPT_INPUT_ERR(
                // [中文导读] [AllReduce逐行 S1096] 为前述多行表达式补入`true, "EI0006", std::vector<std::string>({"reason"}),`；本行是参数/结构化初始化续行。
                true, "EI0006", std::vector<std::string>({"reason"}),
                // [中文导读] [AllReduce逐行 S1097] 为前述多行表达式补入`std::vector<std::string>({GET_SOCKET_TIMEOUT_REASON_CLOSE_DETECT}))`；本行是参数/结构化初始化续行。
                std::vector<std::string>({GET_SOCKET_TIMEOUT_REASON_CLOSE_DETECT}));
            // [中文导读] [AllReduce逐行 S1098] 调用tlsStatusList，使用异常通道TLS状态数组、通道请求条数；对象涉及异常通道TLS状态数组、通道请求条数。
            std::vector<Hccl::TlsStatus> tlsStatusList(channelNum, Hccl::TlsStatus::UNKNOWN);
            // [中文导读] [AllReduce逐行 S1099] 收集未成功连接通道的TLS状态；传入/处理域级通道描述数组、本批通道连接状态数组、通道请求条数、异常通道TLS状态数组。
            GetAbnormalChannelTlsStatus(channelDescs, statusList, channelNum, tlsStatusList);
            // [中文导读] [AllReduce逐行 S1100] 输出失败通道的Rank/句柄/协议/TLS/耗时诊断。
            logger::ChannelLogger::PrintChannelErrorDetails(
                // [中文导读] [AllReduce逐行 S1101] 为输出失败通道的Rank/句柄/协议/TLS/耗时诊断补入`rankId_, channelNum, channelDescs, channelHandles, statusList, static_cast<uint64_t>(elapsed),`（本端域内Rank编号、通道请求条数、域级通道描述数组、引擎侧通道句柄数组、本批通道连接状态数组、已经等待的毫秒数）；本行是参数/结构化初始化续行。
                rankId_, channelNum, channelDescs, channelHandles, statusList, static_cast<uint64_t>(elapsed),
                // [中文导读] [AllReduce逐行 S1102] 为输出失败通道的Rank/句柄/协议/TLS/耗时诊断补入`tlsStatusList.data())`（异常通道TLS状态数组的data字段）；本行是参数/结构化初始化续行。
                tlsStatusList.data());
            // [中文导读] [AllReduce逐行 S1103] 返回HCCL_E_TIMEOUT，表示等待超过本函数期限；此路径停止本函数的后续处理。
            return HCCL_E_TIMEOUT;
        // [中文导读] [AllReduce逐行 S1104] 结束`if ((std::chrono::steady_clock::now() - startTime) >= timeout)`（单调时钟开始时刻、超时秒数）分支/循环；控制流返回外层。
        }

        // 2. 处理重试（去除频繁的重试日志，一秒可能重试上千次）
        // [中文导读] 状态尚未就绪时再次轮询；超时检查位于此前，因此反复 AGAIN 也受本批等待期限约束。
        // [中文导读] [AllReduce逐行 S1108] 仅当`(ret == HCCL_E_AGAIN)`（当前调用状态）成立时进入此分支。
        if (ret == HCCL_E_AGAIN) {
            // [中文导读] [AllReduce逐行 S1109] 推进/回退`retryCount++`（连接状态重试计数），更新当前分片、槽位或状态重试的计数。
            retryCount++;
            // [中文导读] [AllReduce逐行 S1110] 跳过本轮剩余代码并进入下一项/下一次状态查询。
            continue;
        // [中文导读] [AllReduce逐行 S1111] 结束`if (ret == HCCL_E_AGAIN)`（当前调用状态）分支/循环；控制流返回外层。
        }

        // 3. 处理资源不足（属于可预期回退场景）
        // [中文导读] [AllReduce逐行 S1114] 仅当`(ret == HCCL_E_UNAVAIL)`（当前调用状态）成立时进入此分支。
        if (ret == HCCL_E_UNAVAIL) {
            // [中文导读] [AllReduce逐行 S1115] 准备已经等待的毫秒数的局部存储/结构描述，初始化方式以本行声明为准。
            auto elapsed
                // [中文导读] [AllReduce逐行 S1116] 调用std::chrono::steady_clock::now，使用单调时钟开始时刻；传入/处理单调时钟开始时刻。
                = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                      // [中文导读] [AllReduce逐行 S1117] 调用count。
                      .count();
            // [中文导读] [AllReduce逐行 S1118] 记录MyRank::BatchConnectChannels的警告诊断；日志本身不执行传输。
            HCCL_WARNING(
                // [中文导读] [AllReduce逐行 S1119] 为当前MyRank::BatchConnectChannels诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                "[%s] channel connect resource unavailable, channelNum[%u], elapsed[%lld]ms, retryCount[%u]", __func__,
                // [中文导读] [AllReduce逐行 S1120] 为前述多行表达式补入`channelNum, elapsed, retryCount)`（通道请求条数、已经等待的毫秒数、连接状态重试计数）；本行是参数/结构化初始化续行。
                channelNum, elapsed, retryCount);
            // [中文导读] [AllReduce逐行 S1121] 返回ret，表示保留当前调用的状态（可能成功或失败）；此路径停止本函数的后续处理。
            return ret;
        // [中文导读] [AllReduce逐行 S1122] 结束`if (ret == HCCL_E_UNAVAIL)`（当前调用状态）分支/循环；控制流返回外层。
        }

        // 4. 处理失败
        // [中文导读] [AllReduce逐行 S1125] 仅当`(ret != HCCL_SUCCESS)`（当前调用状态）成立时进入此分支。
        if (ret != HCCL_SUCCESS) {
            // [中文导读] [AllReduce逐行 S1126] 准备已经等待的毫秒数的局部存储/结构描述，初始化方式以本行声明为准。
            auto elapsed
                // [中文导读] [AllReduce逐行 S1127] 调用std::chrono::steady_clock::now，使用单调时钟开始时刻；传入/处理单调时钟开始时刻。
                = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                      // [中文导读] [AllReduce逐行 S1128] 调用count。
                      .count();
            // [中文导读] [AllReduce逐行 S1129] 记录MyRank::BatchConnectChannels的错误诊断；日志本身不执行传输。
            HCCL_ERROR(
                // [中文导读] [AllReduce逐行 S1130] 为当前MyRank::BatchConnectChannels诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                "[%s] channel connect failed, channelNum[%u], ret[%d], elapsed[%lld]ms, retryCount[%u]", __func__,
                // [中文导读] [AllReduce逐行 S1131] 为前述多行表达式补入`channelNum, ret, elapsed, retryCount)`（通道请求条数、当前调用状态、已经等待的毫秒数、连接状态重试计数）；本行是参数/结构化初始化续行。
                channelNum, ret, elapsed, retryCount);
            // [中文导读] [AllReduce逐行 S1132] 调用tlsStatusList，使用异常通道TLS状态数组、通道请求条数；对象涉及异常通道TLS状态数组、通道请求条数。
            std::vector<Hccl::TlsStatus> tlsStatusList(channelNum, Hccl::TlsStatus::UNKNOWN);
            // [中文导读] [AllReduce逐行 S1133] 收集未成功连接通道的TLS状态；传入/处理域级通道描述数组、本批通道连接状态数组、通道请求条数、异常通道TLS状态数组。
            GetAbnormalChannelTlsStatus(channelDescs, statusList, channelNum, tlsStatusList);
            // [中文导读] [AllReduce逐行 S1134] 输出失败通道的Rank/句柄/协议/TLS/耗时诊断。
            logger::ChannelLogger::PrintChannelErrorDetails(
                // [中文导读] [AllReduce逐行 S1135] 为输出失败通道的Rank/句柄/协议/TLS/耗时诊断补入`rankId_, channelNum, channelDescs, channelHandles, statusList, static_cast<uint64_t>(elapsed),`（本端域内Rank编号、通道请求条数、域级通道描述数组、引擎侧通道句柄数组、本批通道连接状态数组、已经等待的毫秒数）；本行是参数/结构化初始化续行。
                rankId_, channelNum, channelDescs, channelHandles, statusList, static_cast<uint64_t>(elapsed),
                // [中文导读] [AllReduce逐行 S1136] 为输出失败通道的Rank/句柄/协议/TLS/耗时诊断补入`tlsStatusList.data())`（异常通道TLS状态数组的data字段）；本行是参数/结构化初始化续行。
                tlsStatusList.data());
            // [中文导读] [AllReduce逐行 S1137] 返回ret，表示保留当前调用的状态（可能成功或失败）；此路径停止本函数的后续处理。
            return ret;
        // [中文导读] [AllReduce逐行 S1138] 结束`if (ret != HCCL_SUCCESS)`（当前调用状态）分支/循环；控制流返回外层。
        }

        // 5. 正常情况：所有通道连接成功
        // [中文导读] [AllReduce逐行 S1141] 准备已经等待的毫秒数的局部存储/结构描述，初始化方式以本行声明为准。
        auto elapsed
            // [中文导读] [AllReduce逐行 S1142] 调用std::chrono::steady_clock::now，使用单调时钟开始时刻；传入/处理单调时钟开始时刻。
            = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                  // [中文导读] [AllReduce逐行 S1143] 调用count。
                  .count();
        // [中文导读] [AllReduce逐行 S1144] 记录MyRank::BatchConnectChannels的状态/性能诊断；日志本身不执行传输。
        HCCL_INFO(
            // [中文导读] [AllReduce逐行 S1145] 为当前MyRank::BatchConnectChannels诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
            "[%s] all channels connected successfully, channelNum[%u], elapsed[%lld]ms, retryCount[%u]", __func__,
            // [中文导读] [AllReduce逐行 S1146] 为前述多行表达式补入`channelNum, elapsed, retryCount)`（通道请求条数、已经等待的毫秒数、连接状态重试计数）；本行是参数/结构化初始化续行。
            channelNum, elapsed, retryCount);
        // [中文导读] [AllReduce逐行 S1147] 退出当前循环或switch路径，继续其后处理；不代表其他执行流已经完成。
        break;
    // [中文导读] [AllReduce逐行 S1148] 结束`while (true)`分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S1149] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S1150] 结束MyRank::BatchConnectChannels函数体；控制流返回外层。
}

// [中文导读] [AllReduce逐行 S1152] MyRank::ConfigSqDepthByExpansionMode的接口声明：请求的通信引擎、基础层通道描述；这些参数属于本函数调用边界。
HcclResult MyRank::ConfigSqDepthByExpansionMode(CommEngine engine, HcommChannelDesc& hcommDesc) const
// [中文导读] [AllReduce逐行 S1153] 进入MyRank::ConfigSqDepthByExpansionMode函数体：按域配置/引擎展开模式设置 SQ 深度，保留不支持配置的日志与 CCU 模式错误。
{
    // [中文导读] [AllReduce逐行 S1154] 设置const u32 configuredSqDepth为/按`config_.GetConfigSqDepth()`；调用GetConfigSqDepth。
    const u32 configuredSqDepth = config_.GetConfigSqDepth();
    // [中文导读] [AllReduce逐行 S1155] 仅当`(configuredSqDepth != HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET)`成立时进入此分支。
    if (configuredSqDepth != HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET) {
        // [中文导读] [AllReduce逐行 S1156] 设置const CommProtocol remoteProtocol为/按`hcommDesc.remoteEndpoint.protocol`（基础层通道描述的remoteEndpoint.protocol字段）。
        const CommProtocol remoteProtocol = hcommDesc.remoteEndpoint.protocol;
        // [中文导读] [AllReduce逐行 S1157] 仅当`(engine == COMM_ENGINE_AIV`（请求的通信引擎）成立时进入此分支。
        if (engine == COMM_ENGINE_AIV
            // [中文导读] [AllReduce逐行 S1158] 补全本分支/循环判断的`&& (remoteProtocol == COMM_PROTOCOL_UBC_TP || remoteProtocol == COMM_PROTOCOL_UBC_CTP`，和前面条件共同决定是否进入后续路径。
            && (remoteProtocol == COMM_PROTOCOL_UBC_TP || remoteProtocol == COMM_PROTOCOL_UBC_CTP
                // [中文导读] [AllReduce逐行 S1159] 为前述多行表达式补入`|| remoteProtocol == COMM_PROTOCOL_UBG)) {`；本行是参数/结构化初始化续行。
                || remoteProtocol == COMM_PROTOCOL_UBG)) {
            // [中文导读] [AllReduce逐行 S1160] 设置基础层通道描述的ubAttr.sqDepth字段为/按`configuredSqDepth`。
            hcommDesc.ubAttr.sqDepth = configuredSqDepth;
            // [中文导读] [AllReduce逐行 S1161] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
            return HCCL_SUCCESS;
        // [中文导读] [AllReduce逐行 S1162] 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。
        } else {
            // [中文导读] [AllReduce逐行 S1163] 记录MyRank::ConfigSqDepthByExpansionMode的警告诊断；日志本身不执行传输。
            HCCL_WARNING(
                // [中文导读] [AllReduce逐行 S1164] 为当前MyRank::ConfigSqDepthByExpansionMode诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                "[%s] configured sqDepth[%u] is not supported when engine[%s] protocol[%s].", __func__,
                // [中文导读] [AllReduce逐行 S1165] 为把枚举转换成诊断名称补入`configuredSqDepth, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(),`（请求的通信引擎）；本行是参数/结构化初始化续行。
                configuredSqDepth, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(),
                // [中文导读] [AllReduce逐行 S1166] 为把枚举转换成诊断名称补入`MyRankUtils::GetCommProtocolEnumStr(remoteProtocol).c_str())`；本行是参数/结构化初始化续行。
                MyRankUtils::GetCommProtocolEnumStr(remoteProtocol).c_str());
        // [中文导读] [AllReduce逐行 S1167] 结束当前局部作用域；控制流返回外层。
        }
    // [中文导读] [AllReduce逐行 S1168] 结束`if (configuredSqDepth != HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET)`分支/循环；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S1170] 设置constexpr u32 CCU_MS_MODE_DEPTH为/按`128`。
    constexpr u32 CCU_MS_MODE_DEPTH = 128;
    // [中文导读] [AllReduce逐行 S1171] 设置constexpr u32 CCU_SCHED_MODE_DEPTH为/按`16`。
    constexpr u32 CCU_SCHED_MODE_DEPTH = 16;
    // [中文导读] [AllReduce逐行 S1172] 仅当`(engine == COMM_ENGINE_CCU)`（请求的通信引擎）成立时进入此分支。
    if (engine == COMM_ENGINE_CCU) {
        // [中文导读] [AllReduce逐行 S1173] 仅当`(opExpansionMode_ == CCU_MS_MODE)`（域内算子展开模式）成立时进入此分支。
        if (opExpansionMode_ == CCU_MS_MODE) {
            // [中文导读] [AllReduce逐行 S1174] 设置基础层通道描述的ubAttr.sqDepth字段为/按`CCU_MS_MODE_DEPTH`。
            hcommDesc.ubAttr.sqDepth = CCU_MS_MODE_DEPTH;
        // [中文导读] [AllReduce逐行 S1175] 仅当`(opExpansionMode_ == CCU_SCHED_MODE)`（域内算子展开模式）成立时进入此分支。
        } else if (opExpansionMode_ == CCU_SCHED_MODE) {
            // [中文导读] [AllReduce逐行 S1176] 设置基础层通道描述的ubAttr.sqDepth字段为/按`CCU_SCHED_MODE_DEPTH`。
            hcommDesc.ubAttr.sqDepth = CCU_SCHED_MODE_DEPTH;
        // [中文导读] [AllReduce逐行 S1177] 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。
        } else {
            // [中文导读] [AllReduce逐行 S1178] 记录MyRank::ConfigSqDepthByExpansionMode的错误诊断，字段包含域内算子展开模式；日志本身不执行传输。
            HCCL_ERROR("[%s] unexpected op expansion mode[%u] for ccu,", __func__, opExpansionMode_);
            // [中文导读] [AllReduce逐行 S1179] 返回HCCL_E_INTERNAL，表示内部处理失败；此路径停止本函数的后续处理。
            return HCCL_E_INTERNAL;
        // [中文导读] [AllReduce逐行 S1180] 结束`if (opExpansionMode_ == CCU_MS_MODE)`（域内算子展开模式）分支/循环；控制流返回外层。
        }
    // [中文导读] [AllReduce逐行 S1181] 结束`if (engine == COMM_ENGINE_CCU)`（请求的通信引擎）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S1182] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S1183] 结束MyRank::ConfigSqDepthByExpansionMode函数体；控制流返回外层。
}

void MyRank::LogChannelCreationInfo(
    CommEngine engine, const std::string& commTag, const HcclChannelDesc* channelDescs, uint32_t channelNum,
    const ChannelHandle* hostChannelHandleList) const
{
    for (u32 i = 0; i < channelNum; ++i) {
        u32 remoteRank = channelDescs[i].remoteRank;
        HcclCommDfx::AddChannelRemoteRankId(commTag, hostChannelHandleList[i], remoteRank);
        // 打印UB通道建链信息
        if (channelDescs[i].localEndpoint.loc.locType == ENDPOINT_LOC_TYPE_DEVICE
            && channelDescs[i].remoteEndpoint.loc.locType == ENDPOINT_LOC_TYPE_DEVICE) {
            HCCL_CONFIG_DEBUG(
                HCCL_RES,
                "create channel info:channel handle[%s] comm tag[%s] protocol[%s]"
                " local rank[%u] local dev phyid[%u] remote rank[%u] remote dev phyid[%u] engine[%s]",
                std::to_string(hostChannelHandleList[i]).c_str(), commTag.c_str(),
                MyRankUtils::GetCommProtocolEnumStr(channelDescs[i].localEndpoint.protocol).c_str(), rankId_,
                channelDescs[i].localEndpoint.loc.device.devPhyId, remoteRank,
                channelDescs[i].remoteEndpoint.loc.device.devPhyId,
                GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
        } else {
            HCCL_CONFIG_DEBUG(
                HCCL_RES,
                "create channel info:channel handle[%s] comm tag[%s] protocol[%s]"
                " local rank[%u] remote rank[%u] engine[%s]",
                std::to_string(hostChannelHandleList[i]).c_str(), commTag.c_str(),
                MyRankUtils::GetCommProtocolEnumStr(channelDescs[i].localEndpoint.protocol).c_str(), rankId_,
                remoteRank, GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
        }
    }
}

// [中文导读] [AllReduce逐行 S1216] MyRank::FinalizeChannelsByEngine的接口声明：为 AICPU 初始化设备域并发射通道初始化 Kernel；其他支持引擎复制 Host 句柄；这些参数属于本函数调用边界。
HcclResult MyRank::FinalizeChannelsByEngine(
    // [中文导读] [AllReduce逐行 S1217] MyRank::FinalizeChannelsByEngine的接口声明：请求的通信引擎、通信域标识、通道请求条数、基础层通道描述数组；这些参数属于本函数调用边界。
    CommEngine engine, const std::string& commTag, uint32_t channelNum, std::vector<HcommChannelDesc>& hcommDescs,
    // [中文导读] [AllReduce逐行 S1218] MyRank::FinalizeChannelsByEngine的接口声明：Host控制对象句柄数组、引擎侧通道句柄数组；这些参数属于本函数调用边界。
    ChannelHandle* hostChannelHandleList, ChannelHandle* channelHandles)
// [中文导读] [AllReduce逐行 S1219] 进入MyRank::FinalizeChannelsByEngine函数体：为 AICPU 初始化设备域并发射通道初始化 Kernel；其他支持引擎复制 Host 句柄。
{
    // [中文导读] [AllReduce逐行 S1220] 仅当`(engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS)`（请求的通信引擎）成立时进入此分支。
    if (engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS) {
        // 新增：添加 kernelLaunchAicpuCommInit 调用
        // [中文导读] AICPU 首次使用才下发域初始化 Kernel，成功后标记已初始化，后续建链复用设备侧域状态。
        // [中文导读] [AllReduce逐行 S1223] 仅当`(!callbacks_.getAicpuCommState())`成立时进入此分支；调用getAicpuCommState。
        if (!callbacks_.getAicpuCommState()) {
            // [中文导读] [AllReduce逐行 S1224] 记录MyRank::FinalizeChannelsByEngine的状态/性能诊断；日志本身不执行传输。
            HCCL_INFO("MyRank::%s kernelLaunchAicpuCommInit start.", __func__);
            // [中文导读] [AllReduce逐行 S1225] 设置当前调用状态为/按`callbacks_.kernelLaunchAicpuCommInit()`；首次初始化AICPU设备通信域。
            HcclResult ret = callbacks_.kernelLaunchAicpuCommInit();
            // [中文导读] [AllReduce逐行 S1226] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
            CHK_PRT_RET(
                // [中文导读] [AllReduce逐行 S1227] 向条件错误检查提供`ret != HCCL_SUCCESS, HCCL_ERROR("[%s] kernelLaunchAicpuCommInit failed, return [%d].", __func__, ret),`（当前调用状态），用于确定触发条件或形成对应诊断。
                ret != HCCL_SUCCESS, HCCL_ERROR("[%s] kernelLaunchAicpuCommInit failed, return [%d].", __func__, ret),
                // [中文导读] [AllReduce逐行 S1228] 指定条件命中时要返回的错误状态`ret)`（当前调用状态），未命中则继续原处理路径。
                ret);
            // [中文导读] [AllReduce逐行 S1229] 调用setAicpuCommState。
            callbacks_.setAicpuCommState(true);
        // [中文导读] [AllReduce逐行 S1230] 结束`if (!callbacks_.getAicpuCommState())`分支/循环；控制流返回外层。
        }
        // [中文导读] [AllReduce逐行 S1231] 设置基础层通道描述为/按`hcommDescs.data()`（基础层通道描述数组的data字段）；调用data，使用基础层通道描述数组的data字段。
        HcommChannelDesc* hcommDesc = hcommDescs.data();
        // [中文导读] 把 Host 控制资源带入 AICPU 初始化 Kernel，生成执行侧句柄，并登记故障恢复所需信息。
        // [中文导读] [AllReduce逐行 S1233] 下发集合通信域的通道初始化Kernel，把Host资源恢复为设备对象；返回非成功时由检查宏立即向上传递。
        CHK_RET(ChannelProcess::ChannelKernelLaunchForComm(
            // [中文导读] [AllReduce逐行 S1234] 为下发集合通信域的通道初始化Kernel，把Host资源恢复为设备对象补入`channelHandles, hostChannelHandleList, hcommDesc, channelNum, commTag, binHandle_))`（引擎侧通道句柄数组、Host控制对象句柄数组、基础层通道描述、通道请求条数、通信域标识）；本行是参数/结构化初始化续行。
            channelHandles, hostChannelHandleList, hcommDesc, channelNum, commTag, binHandle_));

        // ns recovery
        // [中文导读] [AllReduce逐行 S1237] 登记通道Host/Device句柄供通信异常恢复；传入/处理请求的通信引擎、引擎侧通道句柄数组、Host控制对象句柄数组、通道请求条数、通信域标识。
        nsRecoveryProcessor_->AddNsRecoveryData(engine, channelHandles, hostChannelHandleList, channelNum, commTag);

        // [中文导读] [AllReduce逐行 S1239] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S1240] 结束`if (engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS)`（请求的通信引擎）分支/循环；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S1242] 仅当`(engine == COMM_ENGINE_CPU || engine == COMM_ENGINE_CCU || engine == COMM_ENGINE_AIV)`（请求的通信引擎）成立时进入此分支。
    if (engine == COMM_ENGINE_CPU || engine == COMM_ENGINE_CCU || engine == COMM_ENGINE_AIV) {
        // [中文导读] [AllReduce逐行 S1243] 按显式目标容量复制描述/句柄/资源包并检查安全函数返回值；检查安全函数返回值后继续。
        CHK_SAFETY_FUNC_RET(memcpy_s(
            // [中文导读] [AllReduce逐行 S1244] 为按显式目标容量复制描述/句柄/资源包并检查安全函数返回值补入`channelHandles, channelNum * sizeof(ChannelHandle), hostChannelHandleList,`（引擎侧通道句柄数组、通道请求条数、Host控制对象句柄数组）；本行是参数/结构化初始化续行。
            channelHandles, channelNum * sizeof(ChannelHandle), hostChannelHandleList,
            // [中文导读] [AllReduce逐行 S1245] 为按显式目标容量复制描述/句柄/资源包并检查安全函数返回值补入`channelNum * sizeof(ChannelHandle)))`（通道请求条数）；本行是参数/结构化初始化续行。
            channelNum * sizeof(ChannelHandle)));
        // [中文导读] [AllReduce逐行 S1246] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S1247] 结束`if (engine == COMM_ENGINE_CPU || engine == COMM_ENGINE_CCU || engine == COMM_ENGINE_AIV)`（请求的通信引擎）分支/循环；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S1249] 记录MyRank::FinalizeChannelsByEngine的错误诊断；日志本身不执行传输。
    HCCL_ERROR(
        // [中文导读] [AllReduce逐行 S1250] 为当前MyRank::FinalizeChannelsByEngine诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
        "[MyRank][%s] unsupported comm engine[%s].", __func__,
        // [中文导读] [AllReduce逐行 S1251] 为把枚举转换成诊断名称补入`GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str())`（请求的通信引擎）；本行是参数/结构化初始化续行。
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
    // [中文导读] [AllReduce逐行 S1252] 返回HCCL_E_NOT_SUPPORT，表示当前引擎/设备/协议不支持此路径；此路径停止本函数的后续处理。
    return HCCL_E_NOT_SUPPORT;
// [中文导读] [AllReduce逐行 S1253] 结束MyRank::FinalizeChannelsByEngine函数体；控制流返回外层。
}

// [中文导读] 建链主干：整理描述 -> Socket准备 -> 本地Channel创建/复用 -> 按需等待连接 -> 一致性交换。
// [中文导读] 末尾FinalizeChannelsByEngine把Host控制对象转换/整理为调用Engine需要的句柄表示。
// [中文导读] 只有本轮存在新建Channel才进入连接等待，不能将复用路径也画成完整新建流程。
// [中文导读] [AllReduce逐行 S1258] MyRank::CreateChannels的接口声明：域建链主干：规范描述、准备 Socket、创建/复用 Channel、按需等连接、配置一致性交换与句柄整理；这些参数属于本函数调用边界。
HcclResult MyRank::CreateChannels(
    // [中文导读] [AllReduce逐行 S1259] MyRank::CreateChannels的接口声明：请求的通信引擎、通信域标识、域级通道描述数组、通道请求条数；这些参数属于本函数调用边界。
    CommEngine engine, const std::string& commTag, const HcclChannelDesc* channelDescs, uint32_t channelNum,
    // [中文导读] [AllReduce逐行 S1260] MyRank::CreateChannels的接口声明：引擎侧通道句柄数组；这些参数属于本函数调用边界。
    ChannelHandle* channelHandles)
// [中文导读] [AllReduce逐行 S1261] 进入MyRank::CreateChannels函数体：域建链主干：规范描述、准备 Socket、创建/复用 Channel、按需等连接、配置一致性交换与句柄整理。
{
    // [中文导读] [AllReduce逐行 S1262] 检查`channelDescs`（域级通道描述数组）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(channelDescs);
    // [中文导读] [AllReduce逐行 S1263] 检查`channelHandles`（引擎侧通道句柄数组）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(channelHandles);
    // [中文导读] [AllReduce逐行 S1264] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
    CHK_PRT_RET(channelNum == 0, HCCL_ERROR("[%s] invalid param: channelNum is zero", __func__), HCCL_E_PARA);

    // [中文导读] [AllReduce逐行 S1266] 记录MyRank::CreateChannels的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S1267] 为当前MyRank::CreateChannels诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
        "[CreateChannels][Enter] engine[%s] commTag[%s] channelNum[%u] rankId[%u]",
        // [中文导读] [AllReduce逐行 S1268] 为把枚举转换成诊断名称补入`GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), commTag.c_str(), channelNum, rankId_)`（请求的通信引擎、通信域标识的c_str字段、通道请求条数、本端域内Rank编号）；本行是参数/结构化初始化续行。
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), commTag.c_str(), channelNum, rankId_);

    // 参数检查
    // [中文导读] [AllReduce逐行 S1271] 调用CheckChannelParam，使用请求的通信引擎、域级通道描述数组、通道请求条数；返回非成功时由检查宏立即向上传递。
    CHK_RET(CheckChannelParam(engine, channelDescs, channelNum));

    // [中文导读] Host 资源句柄单独保存，最终才转换/复制到用户出参；allHandles 保持本批注册句柄数组的有效期。
    // [中文导读] [AllReduce逐行 S1274] 调用hostChannelHandles，使用Host控制对象句柄容器、通道请求条数；对象涉及Host控制对象句柄容器、通道请求条数。
    std::vector<ChannelHandle> hostChannelHandles(channelNum);
    // [中文导读] [AllReduce逐行 S1275] 设置Host控制对象句柄数组为/按`hostChannelHandles.data()`（Host控制对象句柄容器的data字段）；调用data，使用Host控制对象句柄容器的data字段。
    ChannelHandle* hostChannelHandleList = hostChannelHandles.data();

    // [中文导读] [AllReduce逐行 S1277] 设置auto& rdmaConfig为/按`Hccl::EnvConfig::GetInstance().GetRdmaConfig()`；取得该管理器单例。
    auto& rdmaConfig = Hccl::EnvConfig::GetInstance().GetRdmaConfig();
    // [中文导读] [AllReduce逐行 S1278] 调用hcommDescs，使用基础层通道描述数组、通道请求条数；对象涉及基础层通道描述数组、通道请求条数。
    std::vector<HcommChannelDesc> hcommDescs(channelNum);
    // [中文导读] [AllReduce逐行 S1279] 调用allHandles，使用保持各通道注册句柄数组生命周期的容器、通道请求条数；对象涉及保持各通道注册句柄数组生命周期的容器、通道请求条数。
    std::vector<std::vector<MemHandle>> allHandles(channelNum);
    // [中文导读] [AllReduce逐行 S1280] 调用roceDescConfigurator，使用通道请求条数；对象涉及通道请求条数。
    RoceChannelDescConfigurator roceDescConfigurator(channelNum);
    // [中文导读] [AllReduce逐行 S1281] 按`(u32 i = 0; i < channelNum; ++i)`（本批条目下标、通道请求条数）遍历本批条目/分片；各次处理保持数组对应关系。
    for (u32 i = 0; i < channelNum; ++i) {
        // [中文导读] 逐条转换描述并应用多 QP 阈值、展开模式 SQ 深度和 RoCE 源端口列表，形成底层建链参数。
        // [中文导读] [AllReduce逐行 S1283] 设置基础层通道描述数组、本批条目下标为/按`MyRankUtils::ChannelDescHccl2Hcomm(channelDescs[i], config_)`（域级通道描述数组、本批条目下标）；把域级通道描述转换为基础层描述并带入域配置。
        hcommDescs[i] = MyRankUtils::ChannelDescHccl2Hcomm(channelDescs[i], config_);
        // [中文导读] [AllReduce逐行 S1284] 设置基础层通道描述数组、本批条目下标为/按`rdmaConfig.GetRdmaMultiQpThreshold()`；调用GetRdmaMultiQpThreshold。
        hcommDescs[i].roceAttr.qpThreshold = rdmaConfig.GetRdmaMultiQpThreshold();
        // [中文导读] [AllReduce逐行 S1285] 按域配置与展开模式选择通道SQ深度；返回非成功时由检查宏立即向上传递。
        CHK_RET(ConfigSqDepthByExpansionMode(engine, hcommDescs[i]));
        // [中文导读] [AllReduce逐行 S1286] 按RoCE请求填源端口列表，非RoCE路径依配置处理；返回非成功时由检查宏立即向上传递。
        CHK_RET(roceDescConfigurator.FillRoceSrcPortList(channelDescs[i], i, hcommDescs[i]));
    // [中文导读] [AllReduce逐行 S1287] 结束`for (u32 i = 0; i < channelNum; ++i)`（本批条目下标、通道请求条数）分支/循环；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S1289] 设置auto start为/按`std::chrono::steady_clock::now()`；调用std::chrono::steady_clock::now。
    auto start = std::chrono::steady_clock::now();
    // [中文导读] 域标签加引擎区分 Socket 资源，再依次完成连接准备与本地 Channel 创建/复用。
    // [中文导读] [AllReduce逐行 S1291] 设置域和引擎组成的连接标签为/按`commTag + "_engine_" + std::to_string(engine)`（通信域标识、请求的通信引擎）；调用std::to_string，使用通信域标识、请求的通信引擎。
    std::string socketTag = commTag + "_engine_" + std::to_string(engine);
    // [中文导读] [AllReduce逐行 S1292] 先批监听再取得已连接Socket并回填描述；返回非成功时由检查宏立即向上传递。
    CHK_RET(BatchCreateSockets(channelDescs, channelNum, socketTag, hcommDescs));
    // [中文导读] [AllReduce逐行 S1293] 执行本行包裹的资源/任务调用；返回非成功时由检查宏立即向上传递，UNAVAIL资源不足状态保持可识别。
    CHK_RET_UNAVAIL(
        // [中文导读] [AllReduce逐行 S1294] 为逐项创建/复用通道并保存本轮新建索引补入`BatchCreateChannels(engine, channelDescs, channelNum, hcommDescs, hostChannelHandleList, allHandles))`（请求的通信引擎、域级通道描述数组、通道请求条数、基础层通道描述数组、Host控制对象句柄数组、保持各通道注册句柄数组生命周期的容器）；本行是参数/结构化初始化续行。
        BatchCreateChannels(engine, channelDescs, channelNum, hcommDescs, hostChannelHandleList, allHandles));

    // 锁内快照本次新建列表：connect 阶段不再持 channelIndexMtx_，避免长耗时 IO 阻塞
    // Query/Destroy；回滚时基于快照重新取锁清理，保证 newChannels_ 读写均在锁内
    // [中文导读] [AllReduce逐行 S1298] 准备新建通道二元索引列表的局部快照，稍后在锁内复制，用于锁外等待连接和失败回滚。
    std::vector<std::pair<u32, u32>> newChannelsSnapshot;
    // [中文导读] [AllReduce逐行 S1299] 开始MyRank::CreateChannels函数体。
    {
        // [中文导读] [AllReduce逐行 S1300] 调用lock；保持声明的局部对象用于后续处理。
        std::lock_guard<std::mutex> lock(channelIndexMtx_);
        // [中文导读] [AllReduce逐行 S1301] 设置新建通道列表的锁内快照为/按`newChannels_`（本轮新建通道列表）。
        newChannelsSnapshot = newChannels_;
    // [中文导读] [AllReduce逐行 S1302] 结束当前局部作用域；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S1304] 仅当`(!newChannelsSnapshot.empty())`（新建通道列表的锁内快照的empty字段）成立时进入此分支；检查容器是否没有登记项。
    if (!newChannelsSnapshot.empty()) {
        // [中文导读] [AllReduce逐行 S1305] 设置连接等待结果为/按`BatchConnectChannels(channelDescs, hostChannelHandleList, channelNum)`（域级通道描述数组、Host控制对象句柄数组、通道请求条数）；轮询整批Channel连接就绪并按超时/错误分流。
        HcclResult connRet = BatchConnectChannels(channelDescs, hostChannelHandleList, channelNum);
        // [中文导读] CCU 连接阶段遇到资源不足时按新建快照回滚；清理失败只记录，仍保留原连接错误返回。
        // [中文导读] [AllReduce逐行 S1307] 仅当`(connRet == HCCL_E_UNAVAIL && engine == COMM_ENGINE_CCU)`（连接等待结果、请求的通信引擎）成立时进入此分支。
        if (connRet == HCCL_E_UNAVAIL && engine == COMM_ENGINE_CCU) {
            // CCU 场景额外回滚本次新建的 channel，避免资源残留
            // [中文导读] [AllReduce逐行 S1309] 记录MyRank::CreateChannels的警告诊断；日志本身不执行传输。
            HCCL_RUN_WARNING(
                // [中文导读] [AllReduce逐行 S1310] 为当前MyRank::CreateChannels诊断/异常表达式提供格式文本，将报告连接等待结果；这一物理行没有数据搬运副作用。
                "[%s] BatchConnectChannels failed[%d], engine[%s], new channels num[%u]", __func__, connRet,
                // [中文导读] [AllReduce逐行 S1311] 为把枚举转换成诊断名称；读取容器登记项数补入`GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), newChannelsSnapshot.size())`（请求的通信引擎、新建通道列表的锁内快照的size字段）；本行是参数/结构化初始化续行。
                GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), newChannelsSnapshot.size());
            // [中文导读] [AllReduce逐行 S1312] 调用lock；保持声明的局部对象用于后续处理。
            std::lock_guard<std::mutex> lock(channelIndexMtx_);
            // [中文导读] [AllReduce逐行 S1313] 设置回滚清理结果为/按`DestroyNewChannels(engine, channelDescs, newChannelsSnapshot)`（请求的通信引擎、域级通道描述数组、新建通道列表的锁内快照）；逆序销毁本轮新建Channel以回滚资源不足路径。
            HcclResult destroyRet = DestroyNewChannels(engine, channelDescs, newChannelsSnapshot);
            // [中文导读] [AllReduce逐行 S1314] 仅当`(destroyRet != HCCL_SUCCESS)`（回滚清理结果）成立时进入此分支。
            if (destroyRet != HCCL_SUCCESS) {
                // [中文导读] [AllReduce逐行 S1315] 记录MyRank::CreateChannels的错误诊断；日志本身不执行传输。
                HCCL_ERROR(
                    // [中文导读] [AllReduce逐行 S1316] 为当前MyRank::CreateChannels诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                    "[%s] DestroyNewChannels failed[%d] during rollback, connRet[%d], "
                    // [中文导读] [AllReduce逐行 S1317] 为当前MyRank::CreateChannels诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                    "residual newChannels[%zu] may leak.",
                    // [中文导读] [AllReduce逐行 S1318] 为读取容器登记项数补入`__func__, destroyRet, connRet, newChannels_.size())`（回滚清理结果、连接等待结果、本轮新建通道列表的size字段）；本行是参数/结构化初始化续行。
                    __func__, destroyRet, connRet, newChannels_.size());
            // [中文导读] [AllReduce逐行 S1319] 结束`if (destroyRet != HCCL_SUCCESS)`（回滚清理结果）分支/循环；控制流返回外层。
            }
        // [中文导读] [AllReduce逐行 S1320] 结束`if (connRet == HCCL_E_UNAVAIL && engine == COMM_ENGINE_CCU)`（连接等待结果、请求的通信引擎）分支/循环；控制流返回外层。
        }
        // [中文导读] [AllReduce逐行 S1321] 执行本行包裹的资源/任务调用；返回非成功时由检查宏立即向上传递，UNAVAIL资源不足状态保持可识别。
        CHK_RET_UNAVAIL(connRet);
        // [中文导读] [AllReduce逐行 S1322] 设置auto end为/按`std::chrono::steady_clock::now()`；调用std::chrono::steady_clock::now。
        auto end = std::chrono::steady_clock::now();
        // [中文导读] [AllReduce逐行 S1323] 设置auto duration为/按`std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()`；调用count。
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        // [中文导读] [AllReduce逐行 S1324] 记录MyRank::CreateChannels的状态/性能诊断；日志本身不执行传输。
        HCCL_RUN_INFO(
            // [中文导读] [AllReduce逐行 S1325] 为当前MyRank::CreateChannels诊断/异常表达式提供格式文本，将报告通道请求条数；这一物理行没有数据搬运副作用。
            "[MyRank][CreateChannels] CreateChannels Time Elapsed [%lld]us, channelNum [%u]", duration, channelNum);
    // [中文导读] [AllReduce逐行 S1326] 结束`if (!newChannelsSnapshot.empty())`（新建通道列表的锁内快照的empty字段）分支/循环；控制流返回外层。
    }

    // [中文导读] 此处交换的是通信配置和上层登记的算子一致性描述，独立于通道内部的内存访问描述交换。
    // [中文导读] 即使本次Channel复用，也不能仅凭“未创建”推断这一步一定被跳过。
    // 借用hcommDescs.socket，完成一致性校验必要的数据交换
    // [中文导读] [AllReduce逐行 S1331] 仅950经现有Socket交换域配置/上层一致性描述；返回非成功时由检查宏立即向上传递。
    CHK_RET(BatchExchangeAndCheckConsistency(channelDescs, hcommDescs, channelNum, newChannels_, engine));

    // 添加初始化时进行填表
    // [中文导读] [AllReduce逐行 S1334] 登记Host句柄到远端Rank并打印建链信息；传入/处理请求的通信引擎、通信域标识、域级通道描述数组、通道请求条数、Host控制对象句柄数组。
    LogChannelCreationInfo(engine, commTag, channelDescs, channelNum, hostChannelHandleList);

    // [中文导读] [AllReduce逐行 S1336] 直接返回`FinalizeChannelsByEngine(engine, commTag, channelNum, hcommDescs, hostChannelHandleList, channelHandles)`（请求的通信引擎、通信域标识、通道请求条数、基础层通道描述数组、Host控制对象句柄数组、引擎侧通道句柄数组）；根据引擎把Host通道资源转换/整理为执行侧句柄。
    return FinalizeChannelsByEngine(engine, commTag, channelNum, hcommDescs, hostChannelHandleList, channelHandles);
// [中文导读] [AllReduce逐行 S1337] 结束MyRank::CreateChannels函数体；控制流返回外层。
}

// [中文导读] 从远端注册内存列表定位HcclBuffer标签，返回其地址/长度给HCCL算法。
// [中文导读] 无标签的AicpuTsHccsChannel按约定取第0项；不能对所有后端都假定第0项就是任意用户内存。
// [中文导读] [AllReduce逐行 S1341] MyRank::ChannelGetHcclBuffer的接口声明：传输通道句柄、CCL缓冲区地址出参、字节容量或单片字节数；这些参数属于本函数调用边界。
HcclResult MyRank::ChannelGetHcclBuffer(ChannelHandle channel, void** buffer, uint64_t* size)
// [中文导读] [AllReduce逐行 S1342] 进入MyRank::ChannelGetHcclBuffer函数体：根据远端标签或 HCCS 通道约定找到对端 CCL 描述。
{
    // [中文导读] [AllReduce逐行 S1343] 检查`buffer`（CCL缓冲区地址出参）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(buffer);
    // [中文导读] [AllReduce逐行 S1344] 检查`size`（字节容量或单片字节数）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(size);

    // [中文导读] [AllReduce逐行 S1346] 设置u32 memNum为/按`0`。
    u32 memNum = 0;
    // [中文导读] [AllReduce逐行 S1347] 设置CommMem* remoteMem为/按`nullptr`。
    CommMem* remoteMem = nullptr;
    // [中文导读] [AllReduce逐行 S1348] 设置本通道需要交换的内存标签为/按`nullptr`。
    char** memTags = nullptr;
    // [中文导读] [AllReduce逐行 S1349] 调用HcommChannelGetRemoteMems，使用传输通道句柄、本通道需要交换的内存标签；返回非成功时由检查宏立即向上传递。
    CHK_RET(static_cast<HcclResult>(HcommChannelGetRemoteMems(channel, &memNum, &remoteMem, &memTags)));
    // [中文导读] [AllReduce逐行 S1350] 仅当`(memNum > 0)`成立时进入此分支。
    if (memNum > 0) {
        // [中文导读] [AllReduce逐行 S1351] 检查`remoteMem`不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(remoteMem);
        // AicpuTsHccsChannel不使用memTag，返回为空，默认索引0为cclBuffer
        // [中文导读] [AllReduce逐行 S1353] 仅当`(memTags == nullptr)`（本通道需要交换的内存标签）成立时进入此分支。
        if (memTags == nullptr) {
            // [中文导读] [AllReduce逐行 S1354] 设置CCL缓冲区地址出参为/按`remoteMem[0].addr`。
            *buffer = remoteMem[0].addr;
            // [中文导读] [AllReduce逐行 S1355] 设置字节容量或单片字节数为/按`remoteMem[0].size`。
            *size = remoteMem[0].size;
            // [中文导读] [AllReduce逐行 S1356] 记录MyRank::ChannelGetHcclBuffer的状态/性能诊断，字段包含CCL缓冲区地址出参、字节容量或单片字节数；日志本身不执行传输。
            HCCL_INFO("[%s] Found HcclBuffer : addr=%p, size=%llu", __func__, *buffer, *size);
            // [中文导读] [AllReduce逐行 S1357] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
            return HCCL_SUCCESS;
        // [中文导读] [AllReduce逐行 S1358] 结束`if (memTags == nullptr)`（本通道需要交换的内存标签）分支/循环；控制流返回外层。
        }
        // [中文导读] [AllReduce逐行 S1359] 按`(u32 i = 0; i < memNum; ++i)`（本批条目下标）遍历本批条目/分片；各次处理保持数组对应关系。
        for (u32 i = 0; i < memNum; ++i) {
            // [中文导读] [AllReduce逐行 S1360] 设置内存标签为/按`memTags[i]`（本通道需要交换的内存标签、本批条目下标）。
            std::string tag = memTags[i];
            // [中文导读] [AllReduce逐行 S1361] 仅当`(tag == "HcclBuffer")`（内存标签）成立时进入此分支。
            if (tag == "HcclBuffer") {
                // [中文导读] [AllReduce逐行 S1362] 设置CCL缓冲区地址出参为/按`remoteMem[i].addr`（本批条目下标）。
                *buffer = remoteMem[i].addr;
                // [中文导读] [AllReduce逐行 S1363] 设置字节容量或单片字节数为/按`remoteMem[i].size`（本批条目下标）。
                *size = remoteMem[i].size;
                // [中文导读] [AllReduce逐行 S1364] 记录MyRank::ChannelGetHcclBuffer的状态/性能诊断，字段包含CCL缓冲区地址出参、字节容量或单片字节数；日志本身不执行传输。
                HCCL_INFO("[%s] Found HcclBuffer : addr=%p, size=%llu", __func__, *buffer, *size);
                // [中文导读] [AllReduce逐行 S1365] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
                return HCCL_SUCCESS;
            // [中文导读] [AllReduce逐行 S1366] 结束`if (tag == "HcclBuffer")`（内存标签）分支/循环；控制流返回外层。
            }
            // [中文导读] [AllReduce逐行 S1367] 记录MyRank::ChannelGetHcclBuffer的状态/性能诊断，字段包含本通道需要交换的内存标签、本批条目下标；日志本身不执行传输。
            HCCL_INFO("[%s] Found %s : addr=%p, size=%llu", __func__, memTags[i], remoteMem[i].addr, remoteMem[i].size);
        // [中文导读] [AllReduce逐行 S1368] 结束`for (u32 i = 0; i < memNum; ++i)`（本批条目下标）分支/循环；控制流返回外层。
        }
    // [中文导读] [AllReduce逐行 S1369] 结束`if (memNum > 0)`分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S1370] 记录MyRank::ChannelGetHcclBuffer的错误诊断；日志本身不执行传输。
    HCCL_ERROR("[%s] HcclBuffer not found.", __func__);
    // [中文导读] [AllReduce逐行 S1371] 返回HCCL_E_INTERNAL，表示内部处理失败；此路径停止本函数的后续处理。
    return HCCL_E_INTERNAL;
// [中文导读] [AllReduce逐行 S1372] 结束MyRank::ChannelGetHcclBuffer函数体；控制流返回外层。
}

HcclResult
MyRank::ChannelGetRemoteMems(ChannelHandle channel, uint32_t* memNum, CommMem** remoteMem, char*** memTags) const
{
    CHK_PTR_NULL(remoteMem);
    CHK_PTR_NULL(memTags);
    CHK_PTR_NULL(memNum);
    CHK_RET(static_cast<HcclResult>(HcommChannelGetRemoteMems(channel, memNum, remoteMem, memTags)));
    // 添加空指针检查，防止返回的指针为空
    if (*memNum > 0) {
        CHK_PTR_NULL(*remoteMem);
        CHK_PTR_NULL(*memTags);
    }
    HCCL_INFO("[%s] success. memNum[%u]", __func__, *memNum);
    return HCCL_SUCCESS;
}

HcclResult MyRank::ChannelGetRemoteMems(
    ChannelHandle channel, uint32_t* memNum, CommMem** remoteMem, std::vector<std::string>& memTags) const
{
    CHK_PTR_NULL(remoteMem);
    CHK_PTR_NULL(memNum);
    char** rawTags = nullptr;
    CHK_RET(static_cast<HcclResult>(HcommChannelGetRemoteMems(channel, memNum, remoteMem, &rawTags)));
    // 添加空指针检查，防止返回的指针为空
    if (*memNum > 0) {
        CHK_PTR_NULL(*remoteMem);
        CHK_PTR_NULL(rawTags);
        memTags.reserve(*memNum);
        for (uint32_t i = 0; i < *memNum; ++i) {
            memTags.emplace_back(rawTags[i] == nullptr ? "" : rawTags[i]);
        }
    }
    HCCL_INFO("[%s] success. memNum[%u]", __func__, *memNum);
    return HCCL_SUCCESS;
}

std::vector<ChannelHandle> MyRank::GetAllChannelList()
{
    ChannelTable channelTable = rankPairMgr_->GetChannelTable();
    std::vector<ChannelHandle> channelList;
    for (const auto& rankPair : channelTable) {
        for (const auto& endPointPair : rankPair.second) {
            for (const auto& comEngines : endPointPair.second) {
                channelList.insert(channelList.end(), comEngines.second.begin(), comEngines.second.end());
            }
        }
    }

    return channelList;
}

void MyRank::SetKfcControlTransfer(
    std::shared_ptr<HDCommunicate> kfcControlTransferH2D, std::shared_ptr<HDCommunicate> kfcStatusTransferD2H)
{
    if (nsRecoveryProcessor_ == nullptr) {
        HCCL_ERROR("[MyRank][SetKfcControlTransfer] nsRecoveryProcessor_ is null, cannot set KFC control transfer.");
        return;
    }
    nsRecoveryProcessor_->SetKfcControlTransfer(kfcControlTransferH2D, kfcStatusTransferD2H);
}

HcclResult MyRank::StopLaunch()
{
    HCCL_INFO("[NsRecovery][StopLaunch] MyRank::StopLaunch start!");
    auto ret = nsRecoveryProcessor_->StopLaunch();
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[NsRecovery][StopLaunch] MyRank::StopLaunch failed, ret = 0x%016llx", HCCL_ERROR_CODE(ret));
    }
    HCCL_INFO("[NsRecovery][StopLaunch] MyRank::StopLaunch success!");
    return ret;
}

HcclResult MyRank::Clean()
{
    HCCL_INFO("[NsRecovery][Clean] MyRank::Clean start!");
    auto channelList = GetAllChannelList();
    if (channelList.empty()) {
        HCCL_INFO("[NsRecovery][Clean] Channel list empty, No need to clean!");
        return HcclResult::HCCL_SUCCESS;
    }
    auto ret = ChannelProcess::ChannelClean(channelList.data(), channelList.size());
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[NsRecovery][Clean] MyRank::Clean failed, ret = 0x%016llx", HCCL_ERROR_CODE(ret));
        return ret;
    }

    ret = nsRecoveryProcessor_->Clean();
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[NsRecovery][Clean] MyRank::Clean failed, ret = 0x%016llx", HCCL_ERROR_CODE(ret));
        return ret;
    }

    HCCL_INFO("[NsRecovery][Clean] MyRank::Clean success!");
    return HcclResult::HCCL_SUCCESS;
}

HcclResult MyRank::Resume()
{
    HCCL_INFO("[NsRecovery][Resume] MyRank::Resume start!");
    auto channelList = GetAllChannelList();
    if (channelList.empty()) {
        HCCL_INFO("[NsRecovery][Resume] Resume list empty, No need to resume!");
        return HcclResult::HCCL_SUCCESS;
    }

    auto ret = ChannelProcess::ChannelResume(channelList.data(), channelList.size());
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[NsRecovery][Resume] MyRank::Resume failed, ret = 0x%016llx", HCCL_ERROR_CODE(ret));
        return ret;
    }

    ret = nsRecoveryProcessor_->Resume(binHandle_);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[NsRecovery][Resume] MyRank::Resume failed, ret = 0x%016llx", HCCL_ERROR_CODE(ret));
        return ret;
    }

    HCCL_INFO("[NsRecovery][Resume] MyRank::Resume success!");
    return HCCL_SUCCESS;
}

CollCommConfigConsistency& MyRank::GetCollCommConfigConsistency() { return collCommConfigConsistency_; }

} // namespace hccl

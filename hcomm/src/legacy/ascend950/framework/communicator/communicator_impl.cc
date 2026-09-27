/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "communicator_impl.h"
#include <memory>
#include <op_type.h>
#include <adapter_error_manager_pub.h>
#include "orion_adapter_rts.h"
#include "orion_adapter_hal.h"
#include "config_plf_log_v2.h"
#include "hccl_exception.h"
#include "null_ptr_exception.h"
#include "runtime_api_exception.h"
#include "exception_util.h"
#include "hccp_hdc_manager.h"
#include "hccp_peer_manager.h"
#include "rdma_handle_manager.h"
#include "env_config_v2.h"
#include "coll_service_ai_cpu_impl.h"
#include "checkcrc.h"
#include "task_exception_handler.h"
#include "coll_service_device_mode.h"
#include "dlprof_function_v2.h"
#include "kfc.h"
#include "op_params_checker.h"
#include "diff_rank_updater.h"
#include "coll_alg_component_builder.h"
#include "coll_alg_component.h"
#include "hccl_common_v2.h"
#include "tp_manager.h"
#include "hccl_aiv_utils.h"
#include "rts_1ton_cnt_notify.h"
#include "rts_cnt_notify.h"
#include "stream_utils.h"
#include "port.h"
#include "net_instance.h"
#include "ascend_hal_base.h"
#include "acl/acl_rt.h"
#include "types.h"
#include "json_parser.h"
#include "rank_graph_builder_bridge.h"
#include "ccu_jetty_mgr.h"
#include "comm_topo_desc_v2.h"
#include "hostdpu/flush_manager.h"
#include "hostdpu/dpu_kernel_entrance.h"
#include "p2p_enable_manager.h"
#include "adapter_error_manager_pub.h"
#include "ccu_context_all_to_all_v_mesh1d.h"
#include "topo_addr_info.h"

namespace Hccl {
constexpr u64 HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE = (1 * 1024 * 1024); // 指定bufferSize的单位为MB
constexpr u64 HCCL_AIV_OFFLOAD_TAG_BUFFER_SIZE = (4 * 1024 * 1024);     // 指定bufferSize的单位为MB
constexpr u64 HCCL_MC2_ON_AICPU_FIXED_CALC_BUFFER_SIZE
    = 1 * HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE; // MC2适配AICPU，额外需要1M
std::atomic<u32> Hccl::CommunicatorImpl::globalIndex(0);
constexpr u64 HCCL_CCL_AIV_TAG_BUFFER_SIZE = 2;   // 指定存放aiv tag的大小为2M
constexpr u32 HCCL_CCL_AIV_CLEAR_STEP_MAX = 1000; // aiv tag算子下发时++，大于1000置位
constexpr u32 BASE_BIT = 1;                       // 用于左移设置二进制数的特定位
constexpr u64 SHARE_HBM_MEMORY_SIZE = (100 * 1024 * 1024);
constexpr u64 DPU_TASKEXCEPTION_MEMORY_SIZE
    = 10; // DPU TASKEXCEPTION共享内存大小 |stopflag[1]|hcclret[2]|hcclret[2]|,预留5字节
constexpr u64 ALIGN_4K = 4096U;
constexpr const char* DPUTAG = "DPUTAG";
constexpr const char* DPUTASKEXCEPTION = "DPUTASKEXCEPTION";
constexpr u64 INDEPENDENT_OP_BUFFER_SIZE_TIMES = 2; // 自定义算子buffer倍数
constexpr uint8_t DEVICE_SIGNAL_SECOND = 2;
constexpr uint8_t DEVICE_SIGNAL_THIRD = 3;
constexpr uint32_t TEMP_DEV_TYPE_DPU = 0; // 临时适配，后续rts接口上库之后使用rts的定义
static std::atomic<u32> g_commNum(0);     // 一个进程内创建的通信域数量

// 支持零拷贝算子的白名单
std::set<OpType> opWhiteSet
    = {OpType::BROADCAST, OpType::ALLTOALL, OpType::ALLTOALLV, OpType::SEND, OpType::RECV, OpType::ALLGATHER};

static void PrintBackTrace(HcclException& e)
{
    auto backTraces = e.GetBackTraceStrings();
    std::for_each(backTraces.begin(), backTraces.end(), [](string item) {
        HCCL_INFO(item.c_str());
    });
}

static std::shared_ptr<RankGraph> TakeRankGraphOwnership(std::unique_ptr<RankGraph>& rankGraph)
{
    CHK_PRT_THROW(
        rankGraph == nullptr, HCCL_ERROR("[%s] input rankGraph is nullptr.", __func__), NullPtrException,
        "input rankGraph is nullptr.");

    const RankGraphBuilderBridge* bridge = GetRankGraphBuilderBridge();
    CHK_PRT_THROW(
        bridge == nullptr || bridge->adoptRankGraph == nullptr,
        HCCL_ERROR("[%s] RankGraphBuilder ownership bridge is not registered.", __func__), InternalException,
        "RankGraphBuilder ownership bridge is not registered.");

    std::shared_ptr<RankGraph> sharedRankGraph;
    HcclResult ret = bridge->adoptRankGraph(std::move(rankGraph), sharedRankGraph);
    CHK_PRT_THROW(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s] adopt RankGraph failed, errNo[0x%016llx].", __func__, HCCL_ERROR_CODE(ret)), InternalException,
        "adopt RankGraph failed.");
    CHK_PRT_THROW(
        sharedRankGraph == nullptr, HCCL_ERROR("[%s] adopted RankGraph is nullptr.", __func__), InternalException,
        "adopted RankGraph is nullptr.");
    return sharedRankGraph;
}

static void PrepareRankGraphMetadata(
    const RankTableInfo& inputRankTableInfo, const TopoInfo& inputTopoInfo,
    std::unique_ptr<RankTableInfo>& outputRankTableInfo, std::shared_ptr<TopoInfo>& outputTopoInfo)
{
    auto preparedRankTableInfo = std::make_unique<RankTableInfo>(inputRankTableInfo);
    auto preparedTopoInfo = std::make_shared<TopoInfo>(inputTopoInfo);

    outputRankTableInfo = std::move(preparedRankTableInfo);
    outputTopoInfo = std::move(preparedTopoInfo);
}

HcclResult
CommunicatorImpl::Init(const CommParams& commParams, const std::string& ranktableM, const HcclCommConfig& config)
{
    if (!initFlag) {
        initFlag = true;
        try {
            InitCommonData(commParams, config);
            InitHccpHdc(); // tsdOpen + rainit
            InitRankGraph(ranktableM);
            CHK_RET(InitCommResource(commParams));
        } catch (HcclException& e) {
            HCCL_ERROR(e.what());
            PrintBackTrace(e);
            return e.GetErrorCode();
        } catch (exception& e) {
            HCCL_ERROR(e.what());
            return HcclResult::HCCL_E_INTERNAL;
        } catch (...) {
            HCCL_ERROR("Unknown error occurs!");
            return HcclResult::HCCL_E_INTERNAL;
        }
        return HcclResult::HCCL_SUCCESS;
    }
    HCCL_ERROR("Repeated calling init method!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult CommunicatorImpl::InitCommResource(const CommParams& commParams)
{
    HrtSetDevice(devLogicId);
    if (IsNeedDpu()) {
        InitHccpPeer();
    }
    AppendLocalDieIdForLinks();
    InitCcuSuperFastLoad();
    InitNotifyManager();
    InitStreamManager();
    InitPreResource();
    InitSocketManager();
    InitRmaConnManager();
    InitDataBufferManager();
    InitNotifyFixedValue();
    InitMemTransportManager();
    InitHostDeviceSyncNotifyManager();
    InitUbMemoryTransportMgr();
    CollAlgComponentInit(); // 初始化算法组件
    RegisterAicpuKernel();
    InitCollService();
    InitTraceManager();
    DlProfFunction::GetInstance().DlProfFunctionInit();
    InitMirrorTaskManager();
    CHK_RET(InitProfilingReporter());
    InitTaskExceptionHandler();
    InitHDCommunicate();
    notifyTimeoutCfg.Init();
    SetCommStatus(CommStatus::COMM_READY);
    SnapShotParser::GetInstance().SerializeCommonInfo(
        commParams, config, std::move(ranktableInfo), topoInfo, staticBinaryInfo);
    InitOneSidedService();
    RegisterKernel();
    InitDpuKernel();
    return HCCL_SUCCESS;
}

void CommunicatorImpl::InitDpuKernel()
{
    std::unordered_set<IpAddress> hostIps = GetHostIpFromRankGraph();
    if (hostIps.empty()) {
        return;
    }
    for (auto ip : hostIps) {
        FlushManager::GetInstance().initFlushHandle(ip, devPhyId);
    }
    HCCL_INFO("[InitDpuKernel]all FlushHandle init success.");
    /* kernel Launch */
    CHK_RET_THROW(RuntimeApiException, "InitAndLaunchDpuKernel Failed", InitAndLaunchDpuKernel());
    CHK_RET_THROW(
        RuntimeApiException, "InitAndLaunchAicpuKernel Failed",
        InitAndLaunchAicpuKernel()); // 下aicpukernel,将taskexception共享内存保存到aicpu全局map中
}

std::unordered_set<IpAddress> CommunicatorImpl::GetHostIpFromRankGraph()
{
    HCCL_INFO("[GetHostIpFromRankGraph]Start get host ip.");
    std::unordered_set<IpAddress> ips;
    if (rankGraph->GetPeer(myRank) == nullptr) {
        HCCL_ERROR("[GetHostIpFromRankGraph] rankGraph peer is null!");
        return ips;
    }
    std::vector<std::shared_ptr<NetInstance::ConnInterface>> interfaces = rankGraph->GetPeer(myRank)->GetIfaces();
    for (auto interface : interfaces) {
        // 找到所有在host上和LinkProtocol有rdma的ip进行注册
        if (interface->GetPos() == AddrPosition::HOST && interface->GetLinkProtocols().count(LinkProtocol::ROCE) != 0) {
            IpAddress ip = interface->GetAddr();
            ips.insert(ip);
        }
    }
    HCCL_INFO("[GetHostIpFromRankGraph] Successfully completed: GetHostIp finished.");
    return ips;
}

HcclResult
CommunicatorImpl::Init(const CommParams& commParams, const RankTableInfo& ranktable, const HcclCommConfig& config)
{
    if (!initFlag) {
        initFlag = true;
        try {
            InitCommonData(commParams, config);
            InitHccpHdc(); // tsdOpen + rainit
            InitRankGraph(ranktable);
            CHK_RET(InitCommResource(commParams));
        } catch (HcclException& e) {
            HCCL_ERROR(e.what());
            PrintBackTrace(e);
            return e.GetErrorCode();
        } catch (exception& e) {
            HCCL_ERROR(e.what());
            return HcclResult::HCCL_E_INTERNAL;
        } catch (...) {
            HCCL_ERROR("Unknown error occurs!");
            return HcclResult::HCCL_E_INTERNAL;
        }
        return HcclResult::HCCL_SUCCESS;
    }
    HCCL_ERROR("Repeated calling init method!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult
CommunicatorImpl::Init(const CommParams& commParams, std::unique_ptr<RankGraph>& inputRankGraph, DevId inputDevLogicId)
{
    if (!initFlag) {
        initFlag = true;
        try {
            HrtSetDevice(inputDevLogicId);
            InitCommonData(commParams);
            InitRankGraph(inputRankGraph);
            HrtSetDevice(devLogicId);
            if (IsNeedDpu()) {
                InitHccpPeer();
            }
            InitHccpHdc();
            AppendLocalDieIdForLinks();
            InitCcuSuperFastLoad();
            InitNotifyManager();
            InitStreamManager();
            InitSocketManager();
            InitRmaConnManager();
            InitDataBufferManager();
            InitNotifyFixedValue();
            InitMemTransportManager();
            InitHostDeviceSyncNotifyManager();
            InitUbMemoryTransportMgr();
            CollAlgComponentInit();
            RegisterAicpuKernel();
            InitCollService();
            InitTraceManager();
            InitHDCommunicate();
            InitMirrorTaskManager();
            CHK_RET(InitProfilingReporter());
            InitTaskExceptionHandler();
            RegisterKernel();
            InitDpuKernel();
            SetCommStatus(CommStatus::COMM_READY);
        } catch (HcclException& e) {
            HCCL_ERROR(e.what());
            PrintBackTrace(e);
            return e.GetErrorCode();
        } catch (exception& e) {
            HCCL_ERROR(e.what());
            return HcclResult::HCCL_E_INTERNAL;
        } catch (...) {
            HCCL_ERROR("Unknown error occurs!");
            return HcclResult::HCCL_E_INTERNAL;
        }
        return HcclResult::HCCL_SUCCESS;
    }
    HCCL_ERROR("Repeated calling init method!");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult CommunicatorImpl::Init(
    const CommParams& commParams, std::unique_ptr<RankGraph>& inputRankGraph, HcclCommConfig& subConfig,
    DevId inputDevLogicId)
{
    if (!initFlag) {
        initFlag = true;
        TRY_CATCH_RETURN(
            HrtSetDevice(inputDevLogicId); InitCommonData(commParams, subConfig); InitHccpHdc(); InitCcuSuperFastLoad();
            InitNotifyManager(); InitStreamManager(); InitSocketManager(); InitRmaConnManager();
            InitDataBufferManager(); InitNotifyFixedValue(); InitMemTransportManager();
            InitHostDeviceSyncNotifyManager(); InitTraceManager(); InitHDCommunicate(); notifyTimeoutCfg.Init();
            InitRankGraph(inputRankGraph); if (IsNeedDpu()) { InitHccpPeer(); } AppendLocalDieIdForLinks();
            InitUbMemoryTransportMgr(); CollAlgComponentInit(); RegisterAicpuKernel(); InitCollService();
            DlProfFunction::GetInstance().DlProfFunctionInit(); InitMirrorTaskManager();
            CHK_RET(InitProfilingReporter()); InitTaskExceptionHandler(); RegisterKernel(); InitDpuKernel();
            SetCommStatus(CommStatus::COMM_READY);
            SnapShotParser::GetInstance().SerializeSubCommInfo(commParams, subConfig, rankIdsVec, staticBinaryInfo););
        return HcclResult::HCCL_SUCCESS;
    } else {
        HCCL_ERROR("Repeated calling init method!");
        return HcclResult::HCCL_E_INTERNAL;
    }
}

HcclResult CommunicatorImpl::CreateSubComm(
    const CommParams& subCommParams, const std::vector<u32>& rankIds, CommunicatorImpl* subCommImpl)
{
    TRY_CATCH_RETURN(
        if (initFlag) {
            // 创建子虚拟拓扑
            std::unique_ptr<RankGraph> subRankGraph = rankGraph->CreateSubRankGraph(rankIds);
            // 初始化子通信域
            CHK_RET(subCommImpl->Init(subCommParams, subRankGraph, devLogicId));
            RankIpPortMapPtr rankIpPortMapPtr;
            CHK_RET(GetSocketManager().GetSubCommDeviceServerListenPortMap(rankIds, rankIpPortMapPtr));
            CHK_RET(subCommImpl->SetRankIpPortMap(rankIpPortMapPtr));
            CHK_RET(subCommImpl->GetSocketManager().SetDeviceServerListenPortMap(rankIpPortMapPtr));
            return HcclResult::HCCL_SUCCESS;
        } else {
            std::string msg = StringFormat("CreateSubComm fail, communicator has not been initialized, please check.");
            THROW<InternalException>(msg);
        });
    HCCL_ERROR("CreateSubComm fail !");
    return HcclResult::HCCL_E_INTERNAL;
}

HcclResult CommunicatorImpl::CreateSubComm(
    const CommParams& subCommParams, const std::vector<u32>& rankIds, CommunicatorImpl* subCommImpl,
    HcclCommConfig& subConfig)
{
    TRY_CATCH_RETURN(
        if (initFlag) {
            // 创建子虚拟拓扑
            std::unique_ptr<RankGraph> subRankGraph = rankGraph->CreateSubRankGraph(rankIds);
            subCommImpl->rankIdsVec = rankIds;
            HCCL_INFO(
                "[%s]rankIds size[%u], rankIdsVec size[%u]", __func__, rankIds.size(), subCommImpl->rankIdsVec.size());
            // 初始化子通信域
            CHK_RET(subCommImpl->Init(subCommParams, subRankGraph, subConfig, devLogicId));
            RankIpPortMapPtr rankIpPortMapPtr;
            CHK_RET(GetSocketManager().GetSubCommDeviceServerListenPortMap(rankIds, rankIpPortMapPtr));
            CHK_RET(subCommImpl->SetRankIpPortMap(rankIpPortMapPtr));
            CHK_RET(subCommImpl->GetSocketManager().SetDeviceServerListenPortMap(rankIpPortMapPtr));
            return HcclResult::HCCL_SUCCESS;
        } else {
            std::string msg = StringFormat("CreateSubComm fail, communicator has not been initialized, please check.");
            THROW<InternalException>(msg);
        });
    HCCL_ERROR("CreateSubComm fail !");
    return HcclResult::HCCL_E_INTERNAL;
}

void CommunicatorImpl::RefreshSubmittedOpcnt()
{
    if (currentCollOperator->opType == OpType::SEND || currentCollOperator->opType == OpType::RECV) {
        sendRecvIndex++;
        submittedOpCnt = sendRecvIndex;
    } else {
        collOpIndex++;
        submittedOpCnt = collOpIndex;
    }
    HCCL_INFO(
        "[%s] end, opType[%s], submittedOpCnt[%u], sendRecvIndex[%u], collOpIndex[%u].", __func__,
        currentCollOperator->opType.Describe().c_str(), submittedOpCnt, sendRecvIndex, collOpIndex);
}

void CommunicatorImpl::SingleRankProc(const CollOpParams& opParams, void* stream) const
{
    if (opParams.opType == Hccl::OpType::BATCHSENDRECV || opParams.opType == Hccl::OpType::SEND
        || opParams.opType == Hccl::OpType::RECV) {
        HCCL_WARNING("[CommunicatorImpl][%s] ranksize == 1 is not support BATCHSENDRECV SEND RECV", __func__);
        return;
    }
    if (opParams.sendBuf == opParams.recvBuf) {
        HCCL_WARNING("[CommunicatorImpl][%s] sendBuf == recvBuf, return success", __func__);
        return;
    }
    u64 len{0};
    if (opParams.opType == Hccl::OpType::ALLTOALL) {
        len = DataTypeSizeGet(opParams.all2AllDataDes.sendType) * opParams.all2AllDataDes.sendCount;
    } else if (opParams.opType == Hccl::OpType::ALLTOALLV) {
        len = DataTypeSizeGet(opParams.all2AllVDataDes.sendType)
              * *(static_cast<const u64*>(opParams.all2AllVDataDes.sendCounts));
    } else if (opParams.opType == Hccl::OpType::ALLTOALLVC) {
        len = DataTypeSizeGet(opParams.all2AllVCDataDes.sendType)
              * *(static_cast<const u64*>(opParams.all2AllVCDataDes.sendCountMatrix));
    } else {
        len = DataTypeSizeGet(opParams.dataType) * opParams.count;
    }

    HCCL_INFO(
        "[CommunicatorImpl][%s] sendBuf[%p], recvBuf[%p], len[%llu].", __func__, opParams.sendBuf, opParams.recvBuf,
        len);
    if (len > 0) {
        HrtMemAsyncCopy(opParams.recvBuf, len, opParams.sendBuf, len, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    }
}

bool CommunicatorImpl::TryFastCcuLaunch(const CollOpParams& opParams, aclrtStream const stream)
{
    InitCcuSuperFastLoad(); // 存在profiling开关在多次下发算子时动态变化的场景，每次下发流程中都需要更新开关
    superFasterLoad
        = (opParams.opType == OpType::ALLREDUCE || opParams.opType == OpType::ALLGATHER
           || opParams.opType == OpType::REDUCESCATTER || opParams.opType == OpType::BROADCAST
           || opParams.opType == OpType::ALLTOALL || opParams.opType == OpType::REDUCE
           || opParams.opType == OpType::SCATTER || opParams.opType == OpType::ALLTOALLV);
    bool canUpdate = superFasterLoad
                     && (commExecuteConfig.accState == AcceleratorState::CCU_MS
                         || commExecuteConfig.accState == AcceleratorState::CCU_SCHED);
    if (OpType::ALLTOALL == opParams.opType) {
        ccuParamsMappingKey
            = {static_cast<u32>(opParams.reduceOp), static_cast<u32>(opParams.all2AllDataDes.sendType),
               static_cast<u32>(opParams.all2AllDataDes.sendCount)};
    } else if (OpType::ALLTOALLV == opParams.opType) {
        ccuParamsMappingKey
            = {static_cast<u32>(opParams.reduceOp), static_cast<u32>(opParams.all2AllVDataDes.sendType), 0};
    } else if (OpType::BROADCAST == opParams.opType || OpType::SCATTER == opParams.opType) {
        ccuParamsMappingKey
            = {static_cast<u32>(opParams.root), static_cast<u32>(opParams.dataType), static_cast<u32>(opParams.count)};
    } else {
        ccuParamsMappingKey = {
            static_cast<u32>(opParams.reduceOp), static_cast<u32>(opParams.dataType), static_cast<u32>(opParams.count)};
    }
    auto& ccuParamsMapping = colCcuParamMapping[opParams.opType];
    auto ccuParamsMappingKeyIter = ccuParamsMapping.find(ccuParamsMappingKey);
    bool isCCUChangeModel = canUpdate && ccuParamsMappingKeyIter != ccuParamsMapping.end();
    if (!isCCUChangeModel) {
        return false;
    }
    CachedCCUParams& params = ccuParamsMappingKeyIter->second;

    if (opParams.opType == OpType::ALLTOALLV && params.insType != CcuInstType::CCU_ALLTOALLV_MESH_1D_DIRECT) {
        return false;
    }
    if (enableProfilingEnv) {
        uint64_t beginTime = DlProfFunction::GetInstance().dlMsprofSysCycleTime();
        UpdateProfStat();
        auto dfxOpInfo = std::make_shared<DfxOpInfo>();
        CovertToCurrentCollOperator(id, opParams, OpMode::OPBASE);
        dfxOpInfo->op_ = *GetCurrentCollOperator();
        dfxOpInfo->tag_ = dfxOpInfo->op_.opTag;
        dfxOpInfo->algType_ = GetCurAlgName().c_str();
        dfxOpInfo->commIndex_ = GetIdIndex();
        dfxOpInfo->comm_ = this;
        dfxOpInfo->beginTime_ = DlProfFunction::GetInstance().dlMsprofSysCycleTime();
        dfxOpInfo->commId_ = id;
        dfxOpInfo->opIndex_ = opIndex;
        GetMirrorTaskManager().SetCurrDfxOpInfo(dfxOpInfo);
        ExecuteFastCcuLaunch(opParams, stream, params);
        ReportProfInfo(beginTime, opParams.staticShape, true);
    } else {
        ExecuteFastCcuLaunch(opParams, stream, params);
    }
    return true;
}

static void FastCcuLaunchSaveDfxTaskInfo(
    const CommunicatorImpl& comm, const TaskParam& taskParam, bool isMaster,
    const u32 remoteRankId = INVALID_VALUE_RANKID)
{
    u32 taskId;
    u32 streamId;
    HrtGetTaskIdAndStreamID(taskId, streamId);

    std::unique_ptr<TaskInfo> taskInfo = std::make_unique<TaskInfo>(
        streamId, taskId, remoteRankId, taskParam, comm.GetMirrorTaskManager().GetCurrDfxOpInfo(), isMaster);

    HCCL_INFO("Begin to AddTaskInfo: streamId[%lu], taskId[%lu], remoteRankId[%u].", streamId, taskId, remoteRankId);
    comm.GetMirrorTaskManager().AddTaskInfo(std::move(taskInfo));
}

void CommunicatorImpl::FillAllToAllVArgs(const CollOpParams& opParams, rtCcuTaskInfo_t*& ccuParams) const
{
    std::vector<uint64_t> args;
    CcuContextAllToAllVMesh1D::RefreshArgs(opParams, rankSize, args, myRank);
    rtCcuTaskInfo_t* currCcuParam = ccuParams;
    for (u32 i = 0; i < args.size(); i++) {
        // skip token info
        if (i == 2) {
            continue;
        }
        currCcuParam->args[i % RT_CCU_SQE_ARGS_LEN] = args[i];
        if ((i + 1) % RT_CCU_SQE_ARGS_LEN == 0) {
            currCcuParam += 1;
        }
    }
}

void CommunicatorImpl::ExecuteFastCcuLaunch(
    const CollOpParams& opParams, aclrtStream const stream, CachedCCUParams& params)
{
    static thread_local int slaveIndex = 0;
    static thread_local u32 mStreamId = 0;
    static thread_local u32 value = 0;
    static thread_local Rts1ToNCntNotify* cntNotify1ToN = nullptr;
    static thread_local u32 timeout = notifyTimeoutCfg.GetNotifyTimeout();

    rtCcuTaskInfo_t*& ccuParams = params.ccuParams;

    if (params.insType == CcuInstType::CCU_ALLTOALLV_MESH_1D_DIRECT) {
        FillAllToAllVArgs(opParams, ccuParams);
    } else if (
        params.insType == CcuInstType::CCU_ALLTOALL_MESH_1D_2DIE
        || params.insType == CcuInstType::CCU_ALLGATHER_MESH_1D_2DIE
        || params.insType == CcuInstType::CCU_REDUCE_SCATTER_MESH_1D_2DIE) {
        for (std::size_t i = 0; i < params.totalCounts; ++i) {
            (void)memcpy_s(
                &ccuParams[i].args[0], sizeof(ccuParams[i].args[0]), &opParams.sendBuf, sizeof(ccuParams[i].args[0]));
            (void)memcpy_s(
                &ccuParams[i].args[1], sizeof(ccuParams[i].args[1]), &opParams.recvBuf, sizeof(ccuParams[i].args[1]));
        }
    } else {
        (void)memcpy_s(
            &ccuParams[0].args[0], sizeof(ccuParams[0].args[0]), &opParams.sendBuf, sizeof(ccuParams[0].args[0]));
        (void)memcpy_s(
            &ccuParams[0].args[1], sizeof(ccuParams[0].args[1]), &opParams.recvBuf, sizeof(ccuParams[0].args[1]));
    }

    auto vector_zero_count = params.count[0];
    auto& opbaseStream = GetStreamManager().opbase;
    auto mStream = params.isSlave ? opbaseStream->GetSlave(slaveIndex)->GetPtr() : stream;
    u32 streamNum = params.count.size();
    if (streamNum > 1) {
        timeout = notifyTimeoutCfg.GetNotifyTimeout();
        mStreamId = params.isSlave ? opbaseStream->GetSlave(slaveIndex++)->GetId() : HrtGetStreamId(mStream);
        cntNotify1ToN = GetCcuStreamSyncNotifyManager().GetRts1ToNCntNotify(mStreamId);
        // launch LocalPostTo on stream
        value = 0;
        for (u32 i = 0; i < streamNum - 1; ++i) {
            value |= BASE_BIT << i;
        }
        cntNotify1ToN->PostValue(value, mStream);
    }
    if (taskExceptionEnv || enableProfilingEnv) {
        params.taskParams[0].beginTime = DlProfFunction::GetInstance().dlMsprofSysCycleTime();
        SuperFastLoad(ccuParams, mStream, vector_zero_count);
        params.taskParams[0].endTime = DlProfFunction::GetInstance().dlMsprofSysCycleTime();
        FastCcuLaunchSaveDfxTaskInfo(*this, params.taskParams[0], (!params.isSlave));
    } else {
        SuperFastLoad(ccuParams, mStream, vector_zero_count);
    }

    if (streamNum > 1) {
        RtsCntNotify* cntNotifyNTo1 = GetCcuStreamSyncNotifyManager().GetRtsNTo1CntNotify(mStreamId);
        opbaseStream->RegisterMaster(std::make_unique<Stream>(stream));
        //  launch LocalWaitFrom on stream
        cntNotifyNTo1->WaitValue(value, timeout, mStream);
        for (std::size_t i = 0, len = streamNum - 1; i < len; ++i) {
            u32 bitValue = BASE_BIT << i;
            auto slave = opbaseStream->GetSlave(slaveIndex++);
            auto master = opbaseStream->GetMaster();
            GetStreamManager().CaptureSlaveStream(master, slave);
            cntNotify1ToN->WaitBits(bitValue, timeout, *slave);
            if (taskExceptionEnv || enableProfilingEnv) {
                params.taskParams[i + 1].beginTime = DlProfFunction::GetInstance().dlMsprofSysCycleTime();
                SuperFastLoad(ccuParams + params.count[i], slave->GetPtr(), params.count[i + 1]);
                params.taskParams[i + 1].endTime = DlProfFunction::GetInstance().dlMsprofSysCycleTime();
                FastCcuLaunchSaveDfxTaskInfo(*this, params.taskParams[i + 1], slave->IsMaster());
            } else {
                SuperFastLoad(ccuParams + params.count[i], slave->GetPtr(), params.count[i + 1]);
            }
            // launch localPostTo on extra streams
            cntNotifyNTo1->PostBits(bitValue, *slave);
        }
    }
    if (params.insType == CcuInstType::CCU_REDUCE_SCATTER_MESH_1D_2DIE) {
        // 硬编码
        if (taskExceptionEnv || enableProfilingEnv) {
            TaskParam taskParam{};
            taskParam.beginTime = DlProfFunction::GetInstance().dlMsprofSysCycleTime();
            aclrtReduceKind rtReduceOp
                = static_cast<aclrtReduceKind>(static_cast<int>(RtReduceOpGet(opParams.reduceOp)));
            aclDataType rtDataType = static_cast<aclDataType>(static_cast<int>(RtDataTypeGet(opParams.dataType)));
            constexpr std::size_t myScratchPlace = 4;
            const u32 scratchSize = ccuParams[0].args[myScratchPlace];
            auto src = reinterpret_cast<void*>(ccuParams[0].args[3]);
            auto dst = reinterpret_cast<void*>(ccuParams[0].args[1]);
            HrtReduceAsync(dst, scratchSize, src, scratchSize, rtReduceOp, rtDataType, stream);
            taskParam.taskType = TaskParamType::TASK_REDUCE_INLINE;
            taskParam.endTime = DlProfFunction::GetInstance().dlMsprofSysCycleTime();
            taskParam.taskPara.Reduce.src = src;
            taskParam.taskPara.Reduce.dst = dst;
            taskParam.taskPara.Reduce.size = scratchSize;
            taskParam.taskPara.Reduce.notifyID = INVALID_VALUE_NOTIFYID;
            taskParam.taskPara.Reduce.linkType = DfxLinkType::ONCHIP;
            taskParam.taskPara.Reduce.dataType = DataTypeToHcclDataType(opParams.dataType);
            taskParam.taskPara.Reduce.reduceOp = ReduceOpToHcclReduceOp(opParams.reduceOp);
            FastCcuLaunchSaveDfxTaskInfo(*this, taskParam, true, GetMyRank()); // stream为主流
        } else {
            aclrtReduceKind rtReduceOp
                = static_cast<aclrtReduceKind>(static_cast<int>(RtReduceOpGet(opParams.reduceOp)));
            aclDataType rtDataType = static_cast<aclDataType>(static_cast<int>(RtDataTypeGet(opParams.dataType)));
            constexpr std::size_t myScratchPlace = 4;
            const u32 scratchSize = ccuParams[0].args[myScratchPlace];
            auto src = reinterpret_cast<void*>(ccuParams[0].args[3]);
            auto dst = reinterpret_cast<void*>(ccuParams[0].args[1]);
            HrtReduceAsync(dst, scratchSize, src, scratchSize, rtReduceOp, rtDataType, stream);
        }
    }

    slaveIndex = 0;
    collOpIndex++;
    submittedOpCnt = collOpIndex;
    opBaseOpIndex++;
    opIndex++;
    SetCommStatus(CommStatus::COMM_READY);
}

HcclResult CommunicatorImpl::SetAivControlledCoreNum(bool isAiv)
{
    if (isAiv) {
        u32 numBlocksLimit = MAX_NUM_BLOCKS;
        aclError acl_ret = aclrtGetResInCurrentThread(ACL_RT_DEV_RES_VECTOR_CORE, &numBlocksLimit);
        CHK_PRT_RET(
            acl_ret != ACL_SUCCESS,
            HCCL_ERROR(
                "[CommunicatorImpl::SetAivControlledCoreNum] aclrtGetResInCurrentThread failed, ret=[%d]", acl_ret),
            HCCL_E_PARA);
        CHK_PRT_RET(
            numBlocksLimit < 1,
            HCCL_ERROR(
                "[CommunicatorImpl::SetAivControlledCoreNum] block num less than 1, block num[%u]", numBlocksLimit),
            HCCL_E_PARA);
        currentCollOperator->numBlocksLimit = numBlocksLimit;
        HCCL_INFO("[CommunicatorImpl::SetAivControlledCoreNum] Aiv core limit is [%u].", numBlocksLimit);
    }
    return HCCL_SUCCESS;
}

static HcclResult MatchAclgraph(const rtStream_t stream, bool& isCapture)
{
    rtModel_t rtModel = nullptr;
    CHK_RET(GetStreamCaptureInfo(stream, rtModel, isCapture));
    return HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::OffloadResourcePre(std::string& opTag, const CollOpParams& opParams)
{
    CollOffloadOpResReq resReq;
    auto dataSize = opParams.count * DataTypeSizeGet(opParams.dataType);
    auto dataType = DataTypeToHcclDataType(opParams.dataType);
    CHK_RET(CalcCollOffloadOpRes(opParams.opType, dataSize, dataType, resReq));

    // 设定workspace内存资源
    std::vector<rtStream_t> slaveStreams;
    slaveStreams.resize(resReq.requiredSubQueNum);
    for (u64 i = 0; i < resReq.requiredSubQueNum; ++i) {
        slaveStreams[i] = static_cast<rtStream_t>(std::make_unique<Stream>(true, false).get());
    }
    CHK_RET(SetCollOffloadSlaveStreams(opTag, slaveStreams));
    CHK_RET(
        SetCollOffloadScratchBuf(opTag, reinterpret_cast<void*>(GetCclBuffer()->GetAddr()), GetCclBuffer()->GetSize()));
    return HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::LoadOpbasedCollOp(const CollOpParams& opParams, void* stream)
{
    try {
        isLoadOp = true;
        CHK_RET(CheckCommStatus());
        // 等待通信域状态为Ready，执行算子下发
        WaitReady();
        SnapShotParser::GetInstance().SetIsNeedLoadOp(false);
        if (rankSize == 1) {
            HCCL_WARNING("[CommunicatorImpl][%s] ranksize == 1, enter SingleRankProc", __func__);
            SingleRankProc(opParams, stream);
            return HcclResult::HCCL_SUCCESS;
        }
        // 判断是否为aclgraph
        bool isCapture = false; // isCapture为true表示aclgraph
        CHK_RET(MatchAclgraph(stream, isCapture));
        if (!isCapture && TryFastCcuLaunch(opParams, stream)) { // 若是aclgraph则不走快速下发
            return HcclResult::HCCL_SUCCESS;
        }
        curOpParams = opParams;
        CovertToCurrentCollOperator(id, opParams, OpMode::OPBASE);
        opExecuteConfig = commExecuteConfig;
        ExecAlgSelect(opParams, OpMode::OPBASE); // 根据配置选择对应的collService
        if (dynamic_cast<CollServiceDefaultImpl*>(collService) != nullptr) {
            HCCL_ERROR("Opbase mode is not supported in expanding on the host in 950");
            return HcclResult::HCCL_E_NOT_SUPPORT;
        }
        bool isAiv
            = (opExecuteConfig.accState == AcceleratorState::AIV
               || opExecuteConfig.accState == AcceleratorState::AIV_ONLY);
        SetCommStatus(CommStatus::COMM_READY);
        CHK_RET(
            OpParamsChecker::CheckOpDataTypeOpbase(opParams, GetOpCcuFeatureFlag(), GetOpAiCpuTSFeatureFlag(), isAiv));

        // AICPU aclgraph场景传入的stream被capture且算子时支持零拷贝算法的,会切换到图模式
        if (opExecuteConfig.accState == AcceleratorState::AICPU_TS && isCapture
            && (opWhiteSet.find(opParams.opType) != opWhiteSet.end())) {
            std::string tag = opParams.opTag + "_" + std::to_string(tagResourceIndex_++);
            OffloadResourcePre(tag, opParams);
            HCCL_INFO(
                "[CommunicatorImpl][%s]current op support zero copy in aicpu aclgraph, change to offload", __func__);
            return LoadOffloadCollOp(tag, opParams, stream);
        }
        CHK_RET(SetAivControlledCoreNum(isAiv));

        // 避免transport建链前，通讯域被摧毁
        SetCommStatus(CommStatus::COMM_INUSE);
        if (opParams.sendBuf != nullptr) {
            PrintMemoryAttr(opParams.sendBuf);
        }
        if (opParams.recvBuf != nullptr) {
            PrintMemoryAttr(opParams.recvBuf);
        }
        uint64_t beginTime = DlProfFunction::GetInstance().dlMsprofSysCycleTime();

        // 更新开关状态
        UpdateProfStat();
        collService->LoadWithOpBasedMode(*currentCollOperator, std::make_unique<Stream>(stream));
        if (++aivTag > HCCL_CCL_AIV_CLEAR_STEP_MAX) {
            aivTag = 1;
        }
        // ReportProfInfok:opinfo, allTaskInfo
        bool cachedReq = opParams.staticShape || isCapture;
        ReportProfInfo(beginTime, cachedReq, true);
        RefreshSubmittedOpcnt();
        opBaseOpIndex++;
        opIndex++;
        SetCommStatus(CommStatus::COMM_READY);
    } catch (HcclException& e) {
        SetCommStatus(CommStatus::COMM_READY);
        HCCL_ERROR(e.what());
        PrintBackTrace(e);
        u32 idxHcclException = GetSubmittedOpCnt();
        HCCL_ERROR("SubmittedOpCnt: %u, OperatorParams: %s", idxHcclException, opParams.Describe().c_str());
        return e.GetErrorCode();
    } catch (exception& e) {
        SetCommStatus(CommStatus::COMM_READY);
        HCCL_ERROR(e.what());
        u32 idxException = GetSubmittedOpCnt();
        HCCL_ERROR("SubmittedOpCnt: %u, OperatorParams: %s", idxException, opParams.Describe().c_str());
        return HcclResult::HCCL_E_INTERNAL;
    } catch (...) {
        SetCommStatus(CommStatus::COMM_READY);
        u32 idxOthers = GetSubmittedOpCnt();
        HCCL_ERROR("SubmittedOpCnt: %u, OperatorParams: %s", idxOthers, opParams.Describe().c_str());
        HCCL_ERROR("Unknown error occurs!");
        return HcclResult::HCCL_E_INTERNAL;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::CheckCommStatus() const
{
    if (GetCommStatus() == CommStatus::COMM_ERROR) {
        HCCL_ERROR("Comm has been error, can not load opbased operator now!");
        return HcclResult::HCCL_E_INTERNAL;
    }

    if (isSuspended) {
        HCCL_ERROR("Comm has been suspended, can not load opbased operator now!");
        return HcclResult::HCCL_E_SUSPENDING;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::AllocCollOpResource(const CollOpParams& opParams, void** addr)
{
    try {
        if (opParams.commEngine != HcclAccelerator::AICPU_TS) {
            HCCL_ERROR(
                "[CommunicatorImpl][%s] Only AICPU_TS is supported for aicpu unfold on mc2. input is %s", __func__,
                opParams.commEngine.Describe().c_str());
            return HCCL_E_NOT_SUPPORT;
        }
        CHK_RET(CheckCommStatus());

        WaitReady();
        curOpParams = opParams;
        CovertToCurrentCollOperator(id, opParams, OpMode::OPBASE, false);
        opExecuteConfig = commExecuteConfig;
        ExecAlgSelect(opParams, OpMode::OPBASE);
        CHK_PTR_NULL(collService);
        if (dynamic_cast<CollServiceDefaultImpl*>(collService) != nullptr) {
            HCCL_ERROR("The op base is not supported in expanding on the host in 950 with MC2.");
            return HcclResult::HCCL_E_NOT_SUPPORT;
        }

        SetCommStatus(CommStatus::COMM_READY);
        CHK_RET(
            OpParamsChecker::CheckOpDataTypeOpbase(opParams, GetOpCcuFeatureFlag(), GetOpAiCpuTSFeatureFlag(), false));
        SetCommStatus(CommStatus::COMM_INUSE);
        std::string opAlgTag = opParams.opTag + "_" + curAlgName;
        CHK_RET(collService->AllocCollOpResource(*currentCollOperator, opAlgTag, addr));
        SetCommStatus(CommStatus::COMM_READY);
    } catch (HcclException& e) {
        SetCommStatus(CommStatus::COMM_READY);
        HCCL_ERROR(e.what());
        PrintBackTrace(e);
        HCCL_ERROR("AllocCollOpResource OperatorParams: %s", opParams.Describe().c_str());
        return e.GetErrorCode();
    } catch (exception& e) {
        SetCommStatus(CommStatus::COMM_READY);
        HCCL_ERROR(e.what());
        HCCL_ERROR("AllocCollOpResource OperatorParams: %s", opParams.Describe().c_str());
        return HcclResult::HCCL_E_INTERNAL;
    } catch (...) {
        SetCommStatus(CommStatus::COMM_READY);
        HCCL_ERROR("AllocCollOpResource OperatorParams: %s", opParams.Describe().c_str());
        HCCL_ERROR("Unknown error occurs!");
        return HcclResult::HCCL_E_INTERNAL;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::CalcCollOffloadOpRes(
    const OpType opType, u64 dataSize, HcclDataType dataType, CollOffloadOpResReq& resReq)
{
    HCCL_INFO(
        "[CommunicatorImpl][%s] start, opType[%s], dataSize[%llu].", __func__, opType.Describe().c_str(), dataSize);
    try {
        // 资源计算
        HcclResult errCode = collAlgComponent->CalcResOffload(
            opType, dataSize, dataType, GetCommExecuteConfig(), resReq); // 通信域粒度
        if (errCode != HcclResult::HCCL_SUCCESS) {
            std::string msg = StringFormat(
                "[CommunicatorImpl][%s] Error occurs when call collAlgComponent.CalcResOffload, "
                "error code: %d",
                __func__, errCode);
            HCCL_ERROR(msg.c_str());
            return errCode;
        }
    } catch (HcclException& e) {
        HCCL_ERROR(e.what());
        return e.GetErrorCode();
    } catch (exception& e) {
        HCCL_ERROR(e.what());
        return HcclResult::HCCL_E_INTERNAL;
    } catch (...) {
        HCCL_ERROR("Unknown error occurs!");
        return HcclResult::HCCL_E_INTERNAL;
    }
    HCCL_INFO("[CommunicatorImpl][%s] end.", __func__);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::SetCollOffloadSlaveStreams(const std::string& opTag, std::vector<void*> slaveStreams)
{
    try {
        HCCL_INFO("[CommunicatorImpl][%s] start, opTag[%s].", __func__, opTag.c_str());
        // 将slaveStreams注册到streamManager中
        RegisterOffloadSlaveStreams(opTag, slaveStreams);
        HCCL_INFO("[CommunicatorImpl][%s] end.", __func__);
    } catch (HcclException& e) {
        HCCL_ERROR(e.what());
        return e.GetErrorCode();
    } catch (exception& e) {
        HCCL_ERROR(e.what());
        return HcclResult::HCCL_E_INTERNAL;
    } catch (...) {
        HCCL_ERROR("Unknown error occurs!");
        return HcclResult::HCCL_E_INTERNAL;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
CommunicatorImpl::SetCollOffloadScratchBuf(const std::string& opTag, void* scratchMemPtr, u64 requiredScratchMemSize)
{
    try {
        HCCL_INFO(
            "[CommunicatorImpl][%s] start, opTag[%s] requiredScratchMemSize[%llu].", __func__, opTag.c_str(),
            requiredScratchMemSize);
        // 将scratchBuf注册到dataBufManager中
        RegisterOffloadScratchBuffer(opTag, scratchMemPtr, requiredScratchMemSize);
        HCCL_INFO("[CommunicatorImpl][%s] end.", __func__);
    } catch (HcclException& e) {
        HCCL_ERROR(e.what());
        return e.GetErrorCode();
    } catch (exception& e) {
        HCCL_ERROR(e.what());
        return HcclResult::HCCL_E_INTERNAL;
    } catch (...) {
        HCCL_ERROR("Unknown error occurs!");
        return HcclResult::HCCL_E_INTERNAL;
    }
    return HcclResult::HCCL_SUCCESS;
}

void CommunicatorImpl::RegisterOffloadSlaveStreams(const std::string& opTag, std::vector<void*> slaveStreams) const
{
    StreamManager& sm = GetStreamManager();
    sm.offload->RegisterSlaves(opTag, slaveStreams);
}

void CommunicatorImpl::RegisterOffloadScratchBuffer(
    const std::string& opTag, void* scratchMemPtr, u64 requiredScratchMemSize)
{
    auto scratchBuffer = DevBuffer::Create(reinterpret_cast<uintptr_t>(scratchMemPtr), requiredScratchMemSize);
    if (scratchBuffer) {
        offloadScrachBufferMap[opTag] = scratchBuffer;
        HCCL_RUN_INFO(
            "[CommunicatorImpl] offloadScratchBuffer register, opTag[%s], offloadScratchBufferAddr[%llu], "
            "offloadScratchBufferBufSize[%llu]M",
            opTag.c_str(), scratchBuffer->GetAddr(), scratchBuffer->GetSize() / HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE);
    }
}

HcclResult CommunicatorImpl::LoadOffloadCollOp(std::string& opTag, const CollOpParams& opParams, void* stream)
{
    try {
        HCCL_INFO("CommunicatorImpl::LoadOffloadCollOp dataType[%s]", opParams.dataType.Describe().c_str());
        isLoadOp = true;
        curOpParams = opParams;
        if (GetCommStatus() == CommStatus::COMM_ERROR) {
            HCCL_ERROR("Comm has been error, can not offload operator now!");
            return HcclResult::HCCL_E_INTERNAL;
        }

        if (isSuspended) {
            HCCL_ERROR("Comm has been suspended, can not offload operator now!");
            return HcclResult::HCCL_E_SUSPENDING;
        }

        // 等待通信域状态为Ready，执行算子下发
        WaitReady();
        SnapShotParser::GetInstance().SetIsNeedLoadOp(false);
        if (rankSize == 1) {
            HCCL_WARNING("[CommunicatorImpl][%s] ranksize == 1, enter SingleRankProc", __func__);
            SingleRankProc(opParams, stream);
            return HcclResult::HCCL_SUCCESS;
        }
        uint64_t beginTime = DlProfFunction::GetInstance().dlMsprofSysCycleTime();

        // 更新开关状态
        UpdateProfStat();
        // 判断是否为aclgraph(aicpu场景零拷贝会切图模式)
        bool isCapture = false; // isCapture为true表示aclgraph，profiling需要
        CHK_RET(MatchAclgraph(stream, isCapture));
        HCCL_INFO("CommunicatorImpl::LoadOffloadCollOp opParams dataType[%s]", opParams.dataType.Describe().c_str());
        CovertToCurrentCollOperator(opTag, opParams, OpMode::OFFLOAD);
        HCCL_INFO(
            "CommunicatorImpl::LoadOffloadCollOp currentCollOperator dataType[%s]",
            currentCollOperator->dataType.Describe().c_str());
        // 图模式算子加载选择CollService
        opExecuteConfig = commExecuteConfig;
        ExecAlgSelect(opParams, OpMode::OFFLOAD);

        if (opExecuteConfig.accState == AcceleratorState::HOSTCPU_TS) { // 950不支持HOST_TS模式
            HCCL_ERROR("[CommunicatorImpl::LoadOffloadCollOp] HOSTCPU_TS is not support.");
            return HcclResult::HCCL_E_NOT_SUPPORT;
        }

        bool isAiv
            = (opExecuteConfig.accState == AcceleratorState::AIV
               || opExecuteConfig.accState == AcceleratorState::AIV_ONLY);
        CHK_RET(OpParamsChecker::CheckOpDataTypeOffload(
            opParams, GetOpCcuFeatureFlag(), GetOpAiCpuTSFeatureFlag(), isAiv)); // 算子粒度

        if (isAiv) {
            currentCollOperator->numBlocksLimit = aivCoreLimit;
            HCCL_INFO("[CommunicatorImpl::LoadOffloadCollOp] Aiv core limit is [%u].", aivCoreLimit);
        }
        if (isAiv && aivClearEnable) {
            aivOffloadTag = 1;
        } else if (isAiv) {
            aivOffloadTag++;
        }

        // 避免transport建链前，通讯域被摧毁
        SetCommStatus(CommStatus::COMM_INUSE);
        collService->LoadWithOffloadMode(*currentCollOperator, std::make_unique<Stream>(stream));
        SetCommStatus(CommStatus::COMM_READY);
        bool cachedReq = opParams.staticShape || isCapture;
        ReportProfInfo(beginTime, cachedReq, isCapture); // profiling对于aclgraph场景的处理与单算子一致
        opIndex++;
    } catch (HcclException& e) {
        SetCommStatus(CommStatus::COMM_READY);
        HCCL_ERROR(e.what());
        return e.GetErrorCode();
    } catch (exception& e) {
        SetCommStatus(CommStatus::COMM_READY);
        HCCL_ERROR(e.what());
        return HcclResult::HCCL_E_INTERNAL;
    } catch (...) {
        SetCommStatus(CommStatus::COMM_READY);
        HCCL_ERROR("Unknown error occurs!");
        return HcclResult::HCCL_E_INTERNAL;
    }
    return HcclResult::HCCL_SUCCESS;
}

void CommunicatorImpl::CalcA2ASendRecvMem(
    const CollOpParams& opParams, u64& sendSize, u64& recvSize, bool isHcomSelectAlg) const
{
    u64 sendCount = 0;
    u64 recvCount = 0;
    u32 sendTypeSize = 0;
    u32 recvTypeSize = 0;
    if (opParams.opType == OpType::ALLTOALLV && !isHcomSelectAlg) { // alltoallv场景hcomSeletAlg不填写alltoallv的参数
        for (u32 i = 0; i < rankSize; i++) {
            CHECK_NULLPTR(
                (static_cast<const u64*>(opParams.all2AllVDataDes.sendCounts) + i),
                StringFormat("%s failed, opParams.all2AllVDataDes.sendCounts[%u] is nullptr", __func__, i));
            CHECK_NULLPTR(
                (static_cast<const u64*>(opParams.all2AllVDataDes.sdispls) + i),
                StringFormat("%s failed, opParams.all2AllVDataDes.sdispls[%u] is nullptr", __func__, i));
            CHECK_NULLPTR(
                (static_cast<const u64*>(opParams.all2AllVDataDes.recvCounts) + i),
                StringFormat("%s failed, opParams.all2AllVDataDes.recvCounts[%u] is nullptr", __func__, i));
            CHECK_NULLPTR(
                (static_cast<const u64*>(opParams.all2AllVDataDes.rdispls) + i),
                StringFormat("%s failed, opParams.all2AllVDataDes.rdispls[%u] is nullptr", __func__, i));
            u64 curSendCount = *(static_cast<const u64*>(opParams.all2AllVDataDes.sendCounts) + i)
                               + *(static_cast<const u64*>(opParams.all2AllVDataDes.sdispls) + i);
            sendCount = std::max(sendCount, curSendCount);
            u64 curRecvCount = *(static_cast<const u64*>(opParams.all2AllVDataDes.recvCounts) + i)
                               + *(static_cast<const u64*>(opParams.all2AllVDataDes.rdispls) + i);
            recvCount = std::max(recvCount, curRecvCount);
        }
        sendTypeSize = DataTypeSizeGet(opParams.all2AllVDataDes.sendType);
        recvTypeSize = DataTypeSizeGet(opParams.all2AllVDataDes.recvType);
    } else if (
        opParams.opType == OpType::ALLTOALLVC && !isHcomSelectAlg) { // alltoallvc场景hcomSeletAlg不填写alltoallvc的参数
        for (u32 i = 0; i < rankSize; i++) {
            CHECK_NULLPTR(
                (static_cast<const u64*>(opParams.all2AllVCDataDes.sendCountMatrix) + myRank * rankSize + i),
                StringFormat(
                    "%s failed, opParams.all2AllVCDataDes.sendCountMatrix[%u] is nullptr", __func__,
                    (myRank * rankSize + i)));
            sendCount += *(static_cast<const u64*>(opParams.all2AllVCDataDes.sendCountMatrix) + myRank * rankSize + i);
            recvCount += *(static_cast<const u64*>(opParams.all2AllVCDataDes.sendCountMatrix) + myRank + rankSize * i);
        }
        sendTypeSize = DataTypeSizeGet(opParams.all2AllVCDataDes.sendType);
        recvTypeSize = DataTypeSizeGet(opParams.all2AllVCDataDes.recvType);
    } else {
        sendCount = opParams.all2AllDataDes.sendCount * rankSize;
        recvCount = opParams.all2AllDataDes.recvCount * rankSize;
        sendTypeSize = DataTypeSizeGet(opParams.all2AllDataDes.sendType);
        recvTypeSize = DataTypeSizeGet(opParams.all2AllDataDes.recvType);
    }
    sendSize = sendCount * sendTypeSize;
    recvSize = recvCount * recvTypeSize;
}

void CommunicatorImpl::ConvertCollOperatorA2A(const CollOpParams& opParams, bool isLaunch, bool isHcomSelectAlg)
{
    if (currentCollOperator == nullptr) {
        std::string msg = StringFormat("currentCollOperator is nullptr");
        THROW<NullPtrException>(msg);
    }

    if (isLaunch) {
        LaunchConvertCollOperatorA2A(opParams, isHcomSelectAlg);
    } else {
        DefaultConvertCollOperatorA2A(opParams);
    }
}

void CommunicatorImpl::DefaultConvertCollOperatorA2A(const CollOpParams& opParams)
{
    // MC2场景、图模式算法选择场景准备资源场景下只需默认值
    HCCL_INFO("DefaultConvertCollOperatorA2A start.");
    if (opParams.opType == OpType::ALLTOALL) {
        currentCollOperator->all2AllDataDes.sendCount = 0;
        currentCollOperator->all2AllDataDes.recvCount = 0;
        currentCollOperator->all2AllDataDes.sendType = DataType::FP16;
        currentCollOperator->all2AllDataDes.recvType = DataType::FP16;
        currentCollOperator->dataType = DataType::FP16;
    } else if (opParams.opType == OpType::ALLTOALLV) {
        currentCollOperator->all2AllVDataDes.sendType = DataType::FP16;
        currentCollOperator->all2AllVDataDes.recvType = DataType::FP16;
        currentCollOperator->dataType = DataType::FP16;
    } else if (opParams.opType == OpType::ALLTOALLVC) {
        currentCollOperator->all2AllVCDataDes.sendType = DataType::FP16;
        currentCollOperator->all2AllVCDataDes.recvType = DataType::FP16;
        currentCollOperator->dataType = DataType::FP16;
    }
}

void CommunicatorImpl::LaunchConvertCollOperatorA2A(const CollOpParams& opParams, bool isHcomSelectAlg)
{
    // 下发算子场景下需要继承值并准备Mem
    HCCL_INFO("LaunchConvertCollOperatorA2A start.");
    if (opParams.opType == OpType::ALLTOALL) {
        currentCollOperator->all2AllDataDes.sendCount = opParams.all2AllDataDes.sendCount;
        currentCollOperator->all2AllDataDes.recvCount = opParams.all2AllDataDes.recvCount;
        currentCollOperator->all2AllDataDes.sendType = opParams.all2AllDataDes.sendType;
        currentCollOperator->all2AllDataDes.recvType = opParams.all2AllDataDes.recvType;
        currentCollOperator->dataType = opParams.all2AllDataDes.sendType;
        HCCL_INFO(
            "sendCount[%llu], recvCount[%llu]", opParams.all2AllDataDes.sendCount, opParams.all2AllDataDes.recvCount);
    } else if (opParams.opType == OpType::ALLTOALLV) {
        currentCollOperator->all2AllVDataDes.sendCounts = opParams.all2AllVDataDes.sendCounts;
        currentCollOperator->all2AllVDataDes.recvCounts = opParams.all2AllVDataDes.recvCounts;
        currentCollOperator->all2AllVDataDes.sdispls = opParams.all2AllVDataDes.sdispls;
        currentCollOperator->all2AllVDataDes.rdispls = opParams.all2AllVDataDes.rdispls;
        currentCollOperator->all2AllVDataDes.sendType = opParams.all2AllVDataDes.sendType;
        currentCollOperator->all2AllVDataDes.recvType = opParams.all2AllVDataDes.recvType;
        currentCollOperator->dataType = opParams.all2AllVDataDes.sendType;
    } else if (opParams.opType == OpType::ALLTOALLVC) {
        currentCollOperator->all2AllVCDataDes.sendType = opParams.all2AllVCDataDes.sendType;
        currentCollOperator->all2AllVCDataDes.recvType = opParams.all2AllVCDataDes.recvType;
        currentCollOperator->all2AllVCDataDes.sendCountMatrix = opParams.all2AllVCDataDes.sendCountMatrix;
        currentCollOperator->dataType = opParams.all2AllVCDataDes.sendType;
    }

    u64 sendSize = 0;
    u64 recvSize = 0;
    CalcA2ASendRecvMem(opParams, sendSize, recvSize, isHcomSelectAlg);
    HCCL_INFO("sendSize[%llu], recvSize[%llu]", sendSize, recvSize);
    currentCollOperator->inputMem = DevBuffer::Create(reinterpret_cast<uintptr_t>(opParams.sendBuf), sendSize);
    currentCollOperator->outputMem = DevBuffer::Create(reinterpret_cast<uintptr_t>(opParams.recvBuf), recvSize);
}

void CommunicatorImpl::ConvertCollOperatorMem(const CollOpParams& opParams, u64 size)
{
    HCCL_INFO(
        "[CommunicatorImpl][%s] start, opType[%s], size[%llu]", __func__, opParams.opType.Describe().c_str(), size);

    if (opParams.opType == OpType::REDUCESCATTER || opParams.opType == OpType::SCATTER) {
        currentCollOperator->inputMem
            = DevBuffer::Create(reinterpret_cast<uintptr_t>(opParams.sendBuf), size * rankSize);
    } else {
        currentCollOperator->inputMem = DevBuffer::Create(reinterpret_cast<uintptr_t>(opParams.sendBuf), size);
    }

    if (opParams.opType == OpType::ALLGATHER || opParams.opType == OpType::GATHER) {
        currentCollOperator->outputMem
            = DevBuffer::Create(reinterpret_cast<uintptr_t>(opParams.recvBuf), size * rankSize);
    } else {
        currentCollOperator->outputMem = DevBuffer::Create(reinterpret_cast<uintptr_t>(opParams.recvBuf), size);
    }

    HCCL_INFO("[CommunicatorImpl][%s] end.", __func__);
}

void CommunicatorImpl::ConvertCollOperatorMemV(const CollOpParams& opParams, bool isHcomSelectAlg)
{
    if (isHcomSelectAlg) {
        return; // isHcomSeletAlg表示是否为图插件接口进来，若是跳过该步。未来aiv支持reducescatterv/allgatherv算子时，改处需做对应适配。
    }
    HCCL_INFO("[CommunicatorImpl::%s] start, opType[%s]", __func__, opParams.opType.Describe().c_str());
    u64 size = DataTypeSizeGet(opParams.dataType) * opParams.count;

    u64* counts = static_cast<u64*>(opParams.vDataDes.counts);
    u64 totalCount = 0;
    for (size_t index = 0; index < rankSize; index++) {
        totalCount += counts[index];
    }
    u64 totalSize = DataTypeSizeGet(opParams.dataType) * totalCount;

    if (opParams.opType == OpType::REDUCESCATTERV) {
        currentCollOperator->inputMem = DevBuffer::Create(reinterpret_cast<uintptr_t>(opParams.sendBuf), totalSize);
    } else {
        currentCollOperator->inputMem = DevBuffer::Create(reinterpret_cast<uintptr_t>(opParams.sendBuf), size);
    }

    if (opParams.opType == OpType::ALLGATHERV) {
        currentCollOperator->outputMem = DevBuffer::Create(reinterpret_cast<uintptr_t>(opParams.recvBuf), totalSize);
    } else {
        currentCollOperator->outputMem = DevBuffer::Create(reinterpret_cast<uintptr_t>(opParams.recvBuf), size);
    }

    HCCL_INFO("[CommunicatorImpl::%s] end.", __func__);
}

void CommunicatorImpl::CovertToCurrentCollOperator(
    std::string& opTag, const CollOpParams& opParams, OpMode opMode, bool isLaunch, bool isHcomSelectAlg)
{
    std::string errorMsg = "CovertToCurrentCollOperator make_unique<CollOperator> failed";
    TRY_CATCH_THROW(InternalException, errorMsg, currentCollOperator = make_unique<CollOperator>(););
    CHECK_NULLPTR(currentCollOperator, StringFormat("[CommunicatorImpl][%s] currentCollOperator is nullptr", __func__));
    currentCollOperator->opMode = opMode;
    currentCollOperator->opTag = opTag; // 单算子 标签 为通信域id, 图模式 标签 为传入的opTag
    currentCollOperator->staticAddr = opParams.staticAddr;
    currentCollOperator->staticShape = opParams.staticShape;
    currentCollOperator->myRank = GetMyRank();
    if (opMode == OpMode::OPBASE) { // 单算子Scratch buffer为CCL Buffer
        currentCollOperator->scratchMem = DevBuffer::Create(GetCclBuffer()->GetAddr(), GetCclBuffer()->GetSize());
    } else if (opMode == OpMode::OFFLOAD) {
        if (offloadScrachBufferMap.find(opTag) != offloadScrachBufferMap.end()) {
            auto scratchMem = offloadScrachBufferMap[opTag];
            HCCL_INFO(
                "[CommunicatorImpl::CovertToCurrentCollOperator] offloadScratchBufferMap[%s] is [%s]", opTag.c_str(),
                scratchMem->Describe().c_str());
            currentCollOperator->scratchMem = scratchMem;
        }
    }

    currentCollOperator->opType = opParams.opType;
    currentCollOperator->reduceOp = opParams.reduceOp;
    currentCollOperator->root = opParams.root;
    currentCollOperator->outputDataType = opParams.outputDataType;
    currentCollOperator->sendRecvRemoteRank = opParams.dstRank;
    if (opParams.opType == OpType::ALLTOALL || opParams.opType == OpType::ALLTOALLV
        || opParams.opType == OpType::ALLTOALLVC) {
        ConvertCollOperatorA2A(opParams, isLaunch, isHcomSelectAlg);
    } else if (opParams.opType == OpType::BATCHSENDRECV) {
        currentCollOperator->batchSendRecvDataDes.sendRecvItemsPtr = opParams.batchSendRecvDataDes.sendRecvItemsPtr;
        currentCollOperator->batchSendRecvDataDes.itemNum = opParams.batchSendRecvDataDes.itemNum;
        currentCollOperator->dataType = HcclDataTypeToDataType(
            static_cast<HcclSendRecvItem*>(opParams.batchSendRecvDataDes.sendRecvItemsPtr)->dataType);
    } else {
        currentCollOperator->dataType = opParams.dataType;
        currentCollOperator->dataCount = opParams.count;
        if (opParams.opType == OpType::REDUCESCATTERV || opParams.opType == OpType::ALLGATHERV) {
            currentCollOperator->vDataDes.counts = opParams.vDataDes.counts;
            currentCollOperator->vDataDes.displs = opParams.vDataDes.displs;
            currentCollOperator->vDataDes.dataType = opParams.vDataDes.dataType;
            ConvertCollOperatorMemV(opParams, isHcomSelectAlg);
        } else {
            u64 size = DataTypeSizeGet(opParams.dataType) * opParams.count;
            if (size != 0) {
                ConvertCollOperatorMem(opParams, size);
            } else {
                HCCL_WARNING("[CommunicatorImpl::%s] size is 0", __func__);
            }
        }
    }
    HCCL_INFO(
        "CommunicatorImpl::%s op dataType[%s], dataCount[%llu]", __func__,
        currentCollOperator->dataType.Describe().c_str(), currentCollOperator->dataCount);
}

void CommunicatorImpl::InitCommonData(const CommParams& commParams, const HcclCommConfig& commConfig)
{
    InitCommonDataNotInitDevType(commParams, commConfig);
    // 设定devType，初始化能力，算法及其他模块通过Get获取能力
    DevCapability::GetInstance().Init(devType);
}

void CommunicatorImpl::InitCommonDataNotInitDevType(const CommParams& commParams, const HcclCommConfig& commConfig)
{
    InitCommonData(commParams);
    config = commConfig;
    cclBufferSize = config.hcclBufferSize;
}

void CommunicatorImpl::InitCommonData(const CommParams& commParams)
{
    id = commParams.commId;
    idIndex = globalIndex.fetch_add(1);
    establishLinkSocketTag = id + "_establish_link" + "_" + "exchanger";
    myRank = commParams.myRank;
    rankSize = commParams.rankSize;
    rankInParentComm = commParams.rankInParentComm;
    commDepth = commParams.commDepth;
    devType = commParams.devType;
    isWorldGroup = commParams.isWorldGroup;
    devLogicId = HrtGetDevice();
    devPhyId = HrtGetDevicePhyIdByUserDevId(devLogicId);
}

void CommunicatorImpl::CheckRankGraph() const
{
    // 校验虚拟拓扑中的rankSize和通信域的rankSize一致
    u32 virtRankSize = rankGraph->GetRankSize();
    if (virtRankSize != rankSize) {
        std::string msg = StringFormat(
            "Check rankGraph failed, communicator rankSize[%u] does not equal rankTable rankSize[%u]", rankSize,
            virtRankSize);
        THROW<InvalidParamsException>(msg);
    }

    // 校验0值
    u32 num = rankGraph->GetInnerRankSize();
    if (num == 0) {
        std::string msg = StringFormat("Check rankGraph failed, inner rankSize should not be %u", num);
        THROW<InvalidParamsException>(msg);
    }
    CheckRankGraphAddrs();
}

void CommunicatorImpl::CheckRankGraphAddrs() const
{
    if (rankGraph == nullptr || ranktableInfo == nullptr) {
        HCCL_WARNING("[CommunicatorImpl][%s] rankGraph or ranktableInfo is nullptr, skip.", __func__);
        return;
    }

    if (rankGraph->GetRankSize() == 1) {
        HCCL_WARNING("[CommunicatorImpl][%s] single rank no need do this check!", __func__);
        return;
    }

    // 仅能获取到当前进程所在卡的ip，每个卡独立check自己的部分
    std::unordered_set<Eid> localEidSet;
    NewRankInfo localRankInfo;
    for (auto& rank : ranktableInfo->ranks) {
        if (rank.deviceId == devPhyId) { // 获取本卡的ip地址
            // 无UB场景兜底：本rank的level0为pcie fallback（rootinfo探测不到UB时插入的兜底层），
            // EID查询依赖UB设备，此时查询必然失败，且请求会发到设备侧hccp打出ERROR日志；
            // PCIE链路本就不在EID校验范围内，直接跳过EID查询与校验。
            // 依据pcie_fallback显式标记判断（rank_info_detect_client插入时设置），不依赖实例名
            for (auto& levelInfo : rank.rankLevelInfos) {
                if (levelInfo.netLayer == 0 && levelInfo.pcieFallback) {
                    HCCL_WARNING(
                        "[CommunicatorImpl][%s] level0 is pcie fallback (UB unavailable), skip eid check.", __func__);
                    return;
                }
            }
            HRaInfo info(HrtNetworkMode::HDC, rank.deviceId);
            std::vector<HrtDevEidInfo> localEidInfos = HrtRaGetDevEidInfoList(info);
            for (auto& eidInfo : localEidInfos) {
                localEidSet.insert(eidInfo.ipAddress.GetEid());
            }
            localRankInfo = rank;
            break;
        }
    }

    if (localEidSet.empty()) {
        return;
    }

    const std::shared_ptr<NetInstance::Peer>& peer = rankGraph->GetPeer(myRank);
    const std::vector<std::shared_ptr<NetInstance::ConnInterface>>& interfaces = peer->GetIfaces();
    for (auto& interface : interfaces) {
        const std::set<LinkProtocol>& protocols = interface->GetLinkProtocols(); // PCIE和UBOE没有EID
        if (interface->GetPos() == AddrPosition::DEVICE && protocols.count(LinkProtocol::PCIE) == 0
            && protocols.count(LinkProtocol::UBOE) == 0 && localEidSet.count(interface->GetAddr().GetEid()) == 0) {
            RPT_INPUT_ERR(
                true, "EI0016", std::vector<std::string>({"value", "variable", "expect"}),
                std::vector<std::string>({interface->GetAddr().GetIpStr(), "addr", "A right ip address"}));
            HCCL_ERROR(
                "[%s][%s] errNo[0x%016llx] the ip address %s of ranktable in rank %u is error",
                LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RANKTABLE_CHECK.c_str(),
                HCOM_ERROR_CODE(HcclResult::HCCL_E_PARA), interface->GetAddr().Describe().c_str(), devPhyId);
            THROW<InvalidParamsException>(StringFormat(
                "[CommunicatorImpl][%s]"
                "the ip address %s of ranktable in rank %u is error!",
                __func__, interface->GetAddr().Describe().c_str(), devPhyId));
        }
    }
}

u32 GetLocalDieId(PortData&& port, LinkProtocol linkProtocol)
{
    auto userDevId = HrtGetDevice();
    uint32_t devPhyId = HrtGetDevicePhyIdByUserDevId(userDevId);

    auto& rdmaHandleMgr = RdmaHandleManager::GetInstance();
    auto rdmaHandle = rdmaHandleMgr.Get(devPhyId, port, linkProtocol);
    auto dieId = rdmaHandleMgr.GetDieAndFuncId(rdmaHandle).first;
    return dieId;
}

constexpr u32 localPortId = 0;

void CommunicatorImpl::InitRankGraph(const string& ranktableM)
{
    string topoPath = GetTopoFilePath();
    // legacy 通信域保留调用入口，实际构建通过 bridge 转交给 hcomm 中的 RankGraphBuilder。
    const RankGraphBuilderBridge* bridge = GetRankGraphBuilderBridge();
    CHK_PRT_THROW(
        bridge == nullptr || bridge->buildFromString == nullptr,
        HCCL_ERROR("[%s] RankGraphBuilder bridge is not registered.", __func__), InternalException,
        "RankGraphBuilder bridge is not registered.");

    RankGraphBuildResult buildResult{};
    HcclResult ret = bridge->buildFromString(ranktableM, topoPath, myRank, buildResult);
    CHK_PRT_THROW(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s] build rank graph failed, errNo[0x%016llx].", __func__, HCCL_ERROR_CODE(ret)),
        InternalException, "Build rank graph failed.");
    // 三项结果来自同一个 builder，必须一起更新，避免通信域持有不一致的拓扑快照。
    InitRankGraph(std::move(buildResult.rankGraph), buildResult.rankTableInfo, buildResult.topoInfo);
}

std::string CommunicatorImpl::GetTopoFilePath()
{
    HCCL_INFO("[CommunicatorImpl::%s] start.", __func__);

    std::string filePath = "/etc/hccl_rootinfo.json";
    JsonParser jsonParser{};
    nlohmann::json parseJson{};
    std::string topoFilePath{};
    std::ifstream file(filePath);
    if (file.good()) {
        jsonParser.ParseFileToJson(filePath, parseJson);
        std::string msgRankTopoFile = "error occurs when parser object of propName \"topo_file_path\"";
        TRY_CATCH_THROW(InvalidParamsException, msgRankTopoFile,
                        topoFilePath = GetJsonProperty(parseJson, "topo_file_path"););
    } else {
        const size_t bufSize = 1024;
        auto userDevId = HrtGetDevice();
        auto devPhyId = HrtGetDevicePhyIdByUserDevId(userDevId);
        std::vector<char> buffer(bufSize, '\0');
        int result = TopoAddrInfoGetTopoFilePath(devPhyId, buffer.data(), buffer.size());
        CHK_PRT_THROW(
            result != 0, HCCL_ERROR("[%s] Get topo file path failed.", __func__), InvalidParamsException,
            "Get topo file path failed.");
        topoFilePath = std::string(buffer.data());
    }

    // check topo_file_path
    char resolvedPath[PATH_MAX] = {0};
    CHK_PRT_THROW(
        realpath(topoFilePath.c_str(), resolvedPath) == nullptr,
        HCCL_ERROR("[%s] topo_file_path[%s] is not a valid real path", __func__, topoFilePath.c_str()),
        InvalidParamsException, "topo_file_path error");
    return topoFilePath;
}

void CommunicatorImpl::InitRankGraph(const RankTableInfo& ranktable)
{
    string topoPath = GetTopoFilePath();
    // RootInfoDetect 已产出结构化 RankTableInfo 时，避免再次解析字符串，直接调用对应 provider 回调。
    const RankGraphBuilderBridge* bridge = GetRankGraphBuilderBridge();
    CHK_PRT_THROW(
        bridge == nullptr || bridge->buildFromRankTable == nullptr,
        HCCL_ERROR("[%s] RankGraphBuilder bridge is not registered.", __func__), InternalException,
        "RankGraphBuilder bridge is not registered.");

    RankGraphBuildResult buildResult{};
    HcclResult ret = bridge->buildFromRankTable(ranktable, topoPath, myRank, buildResult);
    CHK_PRT_THROW(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s] build rank graph failed, errNo[0x%016llx].", __func__, HCCL_ERROR_CODE(ret)),
        InternalException, "Build rank graph failed.");
    // 同步安装本次构建得到的 RankGraph、RankTableInfo 和 TopoInfo。
    InitRankGraph(std::move(buildResult.rankGraph), buildResult.rankTableInfo, buildResult.topoInfo);
}

void CommunicatorImpl::InitRankGraph(std::unique_ptr<RankGraph>& inputRankGraph)
{
    if (inputRankGraph != nullptr) {
        // 子通信域等旧路径仍传入 unique_ptr，需要补上 hcomm 侧 deleter 后再由通信域共享持有。
        rankGraph = TakeRankGraphOwnership(inputRankGraph);
    } else {
        std::string msg = StringFormat("Init RankGraph failed, inputRankGraph is nullptr");
        THROW<NullPtrException>(msg);
    }
    CheckRankGraph();
    SaveTopoDesc(id);
}

void CommunicatorImpl::InitRankGraph(
    std::shared_ptr<RankGraph> inputRankGraph, const RankTableInfo& inputRankTableInfo, const TopoInfo& inputTopoInfo)
{
    if (inputRankGraph == nullptr) {
        THROW<NullPtrException>("Init RankGraph failed, inputRankGraph is nullptr");
    }

    std::unique_ptr<RankTableInfo> preparedRankTableInfo;
    std::shared_ptr<TopoInfo> preparedTopoInfo;
    PrepareRankGraphMetadata(inputRankTableInfo, inputTopoInfo, preparedRankTableInfo, preparedTopoInfo);

    const u32 preparedRankSize = inputRankGraph->GetRankSize();
    // 以下移动赋值均不分配内存，确保三个拓扑对象只在准备完整后一起发布。
    rankGraph = std::move(inputRankGraph);
    ranktableInfo = std::move(preparedRankTableInfo);
    topoInfo = std::move(preparedTopoInfo);
    rankSize = preparedRankSize;
    HCCL_RUN_INFO("[CommunicatorImpl::%s] rankTableInfo: %s", __func__, ranktableInfo->Describe().c_str());
    HCCL_RUN_INFO("[CommunicatorImpl][%s] topoInfo[%s]", __func__, topoInfo->Describe().c_str());
    CheckRankGraph();
    SaveTopoDesc(id);
    std::vector<LinkData> fullLinks = GetFullMeshLinks();
    for (auto link : fullLinks) {
        HCCL_RUN_INFO("[CommunicatorImpl][%s] link[%s]", __func__, link.Describe().c_str());
    }
}

void CommunicatorImpl::InitDataBufferManager()
{
    // 申请scratchMem
    u64 scratchBufSize = static_cast<u64>(GetBufferSize());
    if (scratchBufSize == 0) {
        scratchBufSize = EnvConfig::GetInstance().GetAlgoConfig().GetBuffSize();
    } else {
        scratchBufSize = scratchBufSize * HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE;
    }
    // 如果是自定义算子流程，cclBufferSize的大小为2倍
    scratchBufSize = scratchBufSize * INDEPENDENT_OP_BUFFER_SIZE_TIMES;
    cclBufferSize = scratchBufSize;

    // aiv mc2预埋1M，并不暴露在内部算子执行逻辑里
    scratchBufSize += HCCL_MC2_ON_AICPU_FIXED_CALC_BUFFER_SIZE;

    if (rankSize > 1) {
        aivOffloadTagBuffer = std::move(DevBuffer::CreateHugePageBuf(HCCL_AIV_OFFLOAD_TAG_BUFFER_SIZE));
        HrtMemset(
            reinterpret_cast<void*>(aivOffloadTagBuffer->GetAddr()), aivOffloadTagBuffer->GetSize(),
            aivOffloadTagBuffer->GetSize());
        cclBuffer = std::move(DevBuffer::CreateHugePageBuf(scratchBufSize));
        HCCL_RUN_INFO(
            "[CommunicatorImpl][InitDataBufferManager] cclBuffer create, commId[%s], addr[%llu], size[%llu]M",
            GetId().c_str(), cclBuffer->GetAddr(), cclBufferSize / HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE);

        u64 aivTagBufSize = HCCL_CCL_AIV_TAG_BUFFER_SIZE * HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE;
        HCCL_INFO(
            "[CommunicatorImpl][InitDataBufferManager] aivTagBufSize[%llu]M",
            aivTagBufSize / HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE);
        aivTagBuffer = std::move(DevBuffer::CreateHugePageBuf(aivTagBufSize));
        HrtMemset(reinterpret_cast<void*>(aivTagBuffer->GetAddr()), aivTagBuffer->GetSize(), aivTagBuffer->GetSize());
    }
    dataBufferManager = std::make_unique<DataBufManager>();

    localRmaBufManager = std::make_unique<LocalRmaBufManager>(*this);

    remoteRmaBufManager = std::make_unique<RemoteRmaBufManager>(*this);
}

void CommunicatorImpl::InitNotifyManager()
{
    aicpuQueueNotifyManager_ = std::make_unique<QueueNotifyManager>(*this);

    ccuQueueNotifyManager_ = std::make_unique<QueueNotifyManager>(*this);

    queueWaitGroupCntNotifyManager = std::make_unique<QueueWaitGroupCntNotifyManager>();

    queueBcastPostCntNotifyManager = std::make_unique<QueueBcastPostCntNotifyManager>();

    connLocalNotifyManager = std::make_unique<ConnLocalNotifyManager>(this);

    connLocalCntNotifyManager = std::make_unique<ConnLocalCntNotifyManager>(this);

    ccuStreamSyncNotifyManager = std::make_unique<CcuStreamSyncNotifyManager>();
}

void CommunicatorImpl::InitStreamManager()
{
    streamManager = std::make_unique<StreamManager>(this);
    aicpuStreamManager = std::make_unique<AicpuStreamManager>();
}

void CommunicatorImpl::InitCollService()
{
    HCCL_INFO("CommunicatorImpl::InitCollServices start");

    auto ccuCollService = std::make_shared<CollServiceDeviceMode>(this);   // host 展开，ccu使用
    auto aiCpuCollService = std::make_shared<CollServiceAiCpuImpl>(this);  // aicpu 展开
    auto hostCollService = std::make_shared<CollServiceDefaultImpl>(this); // host 展开，图模式使用
    ccuCollService->Init();
    aiCpuCollService->Init();
    hostCollService->Init();

    collServices[AcceleratorState::AIV] = ccuCollService;         // host 展开，aiv使用
    collServices[AcceleratorState::AIV_ONLY] = ccuCollService;    // host 展开，aiv使用
    collServices[AcceleratorState::CCU_MS] = ccuCollService;      // host 展开，ccu使用
    collServices[AcceleratorState::CCU_SCHED] = ccuCollService;   // host 展开，ccu使用
    collServices[AcceleratorState::AICPU_TS] = aiCpuCollService;  // aicpu 展开
    collServices[AcceleratorState::HOSTCPU_TS] = hostCollService; // host 展开，图模式使用

    HCCL_INFO("CommunicatorImpl::InitCollServices end");
    return;
}

HcclResult CommunicatorImpl::InitTraceManager()
{
    /* 申请trace资源信息 */
    std::string logInfo = "HCCL_";
    logInfo.append(std::to_string(SalGetTid()));
    logInfo.append("_");
    logInfo.append(std::to_string(GetDeviceLogicId()));
    logInfo.append("_");
    logInfo.append(std::to_string(idIndex));
    trace = std::make_unique<Trace>();
    CHK_PTR_NULL(trace);
    CHK_RET(trace->Init(logInfo));
    return HCCL_SUCCESS;
}

void CommunicatorImpl::InitHDCommunicate()
{
    // 不管是aicpu还是ccu都初始化
    HCCL_INFO("Enter [CommunicatorImpl::InitHDCommunicate]");
    kfcControlTransferH2D = std::make_unique<HDCommunicate>(devLogicId, HCCLV2_HDC_TYPE_H2D, sizeof(KfcCommand));
    kfcControlTransferH2D->Init();
    kfcStatusTransferD2H = std::make_unique<HDCommunicate>(devLogicId, HCCLV2_HDC_TYPE_D2H, sizeof(KfcExecStatus));
    kfcStatusTransferD2H->Init();
}

void CommunicatorImpl::InitHccpHdc() const { HccpHdcManager::GetInstance().Init(devLogicId); }

void CommunicatorImpl::TryInitCcuFeature() const
{
    TpManager::GetInstance(devLogicId).Init();
    HCCL_INFO(
        "[CommunicatorImpl][%s] passed, "
        "will use open source ccu feature.",
        __func__);
    return;
}

void CommunicatorImpl::InitCcuSuperFastLoad()
{
    // ccu 模式 快速下发模式需要的变量初始化
    taskExceptionEnv = EnvConfig::GetInstance().GetLogConfig().GetDfsConfig().taskExceptionEnable;

    bool hostApiState = ProfilingHandler::GetInstance().GetHostApiState();
    bool nodeState = ProfilingHandler::GetInstance().GetHcclNodeState();
    bool l0State = ProfilingHandler::GetInstance().GetHcclL0State();
    bool l1State = ProfilingHandler::GetInstance().GetHcclL1State();

    enableProfilingEnv = hostApiState || nodeState || l0State || l1State;

    HCCL_INFO(
        "taskExceptionEnv[%d], enableProfilingEnv: hostApiState[%d] nodeState[%d] l0State[%d] l1State[%d]",
        taskExceptionEnv, hostApiState, nodeState, l0State, l1State);
}

void CommunicatorImpl::InitPreResource()
{
    // PCIE链路的两端实现enableP2P
    auto links = GetFullMeshLinks();
    for (auto link : links) {
        if (link.GetLinkProtocol() == LinkProtocol::PCIE) {
            DeviceId remotePhyId = link.GetRemoteDeviceId();
            enableP2PDevices_.push_back(remotePhyId);
        }
    }
    CHK_RET_THROW(
        RuntimeApiException, "EnableP2P Failed", P2PEnableManager::GetInstance().EnableP2P(enableP2PDevices_));
}

void CommunicatorImpl::DeInitPreResource()
{
    (void)P2PEnableManager::GetInstance().DisableP2P(devLogicId, enableP2PDevices_);
}

void CommunicatorImpl::InitSocketManager()
{
    socketManager = std::make_unique<SocketManager>(*this, myRank, devPhyId, devLogicId);
    if (ranktableInfo != nullptr) {
        auto rankIpPortMap = ranktableInfo->GetRankDeviceListenPortMap();
        rankIpPortMap_ = std::make_shared<decltype(rankIpPortMap)>(std::move(rankIpPortMap));
        auto ret = socketManager->SetDeviceServerListenPortMap(rankIpPortMap_);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[%s] SetDeviceServerListenPortMap failed, ret[%d]", __func__, ret);
        }
    }
}

void CommunicatorImpl::InitRmaConnManager() { rmaConnectionManager = std::make_unique<RmaConnManager>(*this); }

void CommunicatorImpl::InitNotifyFixedValue()
{
    // 分配 2MB 内存，临时规避性能问题，实际未使用
    // https://gitcode.com/cann/hcomm/pull/4098 中删除了 NotifyFixedValue，虽然该类的成员函数没有场景会调用到，
    // 但是在初始化通信域时会调用其构造函数，malloc 一块 2MB 的内存，用于存放 roce 场景的 notify。
    // 删除后，导致后续 AllGather 在 1GB 数据量下劣化 10us
    // 当前 补上 malloc(2MB)，临时解决性能劣化
    constexpr u64 notifyValueSize = 2 * 1024 * 1024;
    notifyFixedValue = HrtMalloc(notifyValueSize, static_cast<int>(ACL_MEM_TYPE_HIGH_BAND_WIDTH));
}

void CommunicatorImpl::InitMemTransportManager() { memTransportManager = std::make_unique<MemTransportManager>(*this); }

void CommunicatorImpl::InitHostDeviceSyncNotifyManager()
{
    hostDeviceSyncNotifyManager = std::make_unique<HostDeviceSyncNotifyManager>();
}

const string& CommunicatorImpl::GetId() const { return id; }

u32 CommunicatorImpl::GetIdIndex() const { return idIndex; }

const string& CommunicatorImpl::GetEstablishLinkSocketTag() const { return establishLinkSocketTag; }

RankId CommunicatorImpl::GetMyRank() const { return myRank; }

u32 CommunicatorImpl::GetRankSize() const { return rankSize; }

u32 CommunicatorImpl::GetDeviceLogicId() const { return devLogicId; }

u32 CommunicatorImpl::GetDevicePhyId() const { return devPhyId; }

u64 CommunicatorImpl::GetBufferSize() const { return cclBufferSize; }

u32 CommunicatorImpl::GetSubmittedOpCnt() const { return submittedOpCnt; }

u32 CommunicatorImpl::GetOpBaseOpIndex() const { return opBaseOpIndex; }

u32 CommunicatorImpl::GetOpIndex() const { return opIndex; }

bool CommunicatorImpl::GetOpAiCpuTSFeatureFlag() const
{
    return opExecuteConfig.accState == AcceleratorState::AICPU_TS;
}

bool CommunicatorImpl::GetCommAiCpuTSFeatureFlag() const
{
    return commExecuteConfig.accState == AcceleratorState::AICPU_TS;
}

const DevType& CommunicatorImpl::GetDevType() const
{
    HCCL_INFO("CommunicatorImpl::DevType is %s", devType.Describe().c_str());
    return devType;
}

shared_ptr<RankGraph> CommunicatorImpl::GetRankGraph() const
{
    HCCL_INFO("CommunicatorImpl::GetRankGraph ");
    return rankGraph;
}

DataBufManager& CommunicatorImpl::GetDataBufferManager() const
{
    CHECK_NULLPTR(dataBufferManager, "dataBufferManager is nullptr!");
    return *dataBufferManager;
}

LocalRmaBufManager& CommunicatorImpl::GetLocalRmaBufManager() const
{
    CHECK_NULLPTR(localRmaBufManager, "localRmaBufManager is nullptr!");
    return *localRmaBufManager;
}

RemoteRmaBufManager& CommunicatorImpl::GetRemoteRmaBufManager() const
{
    CHECK_NULLPTR(remoteRmaBufManager, "remoteRmaBufManager is nullptr!");
    return *remoteRmaBufManager;
}

QueueNotifyManager& CommunicatorImpl::GetAicpuQueueNotifyManager() const
{
    CHECK_NULLPTR(aicpuQueueNotifyManager_, "aicpuQueueNotifyManager is nullptr!");
    return *aicpuQueueNotifyManager_;
}

QueueNotifyManager& CommunicatorImpl::GetCcuQueueNotifyManager() const
{
    CHECK_NULLPTR(ccuQueueNotifyManager_, "ccuQueueNotifyManager is nullptr!");
    return *ccuQueueNotifyManager_;
}

ConnLocalNotifyManager& CommunicatorImpl::GetConnLocalNotifyManager() const
{
    CHECK_NULLPTR(connLocalNotifyManager, "connLocalNotifyManager is nullptr!");
    return *connLocalNotifyManager;
}

ConnLocalCntNotifyManager& CommunicatorImpl::GetConnLocalCntNotifyManager() const
{
    CHECK_NULLPTR(connLocalCntNotifyManager, "connLocalCntNotifyManager is nullptr!");
    return *connLocalCntNotifyManager;
}

QueueWaitGroupCntNotifyManager& CommunicatorImpl::GetQueueWaitGroupCntNotifyManager() const
{
    CHECK_NULLPTR(queueWaitGroupCntNotifyManager, "queueWaitGroupCntNotifyManager is nullptr!");
    return *queueWaitGroupCntNotifyManager;
}

QueueBcastPostCntNotifyManager& CommunicatorImpl::GetBcastPostCntNotifyManager() const
{
    CHECK_NULLPTR(queueBcastPostCntNotifyManager, "queueBcastPostCntNotifyManager is nullptr!");
    return *queueBcastPostCntNotifyManager;
}

CcuStreamSyncNotifyManager& CommunicatorImpl::GetCcuStreamSyncNotifyManager() const
{
    CHECK_NULLPTR(ccuStreamSyncNotifyManager, "ccuStreamSyncNotifyManager is nullptr!");
    return *ccuStreamSyncNotifyManager;
}

StreamManager& CommunicatorImpl::GetStreamManager() const
{
    CHECK_NULLPTR(streamManager, "streamManager is nullptr!");
    return *streamManager;
}

AicpuStreamManager& CommunicatorImpl::GetAicpuStreamManager() const
{
    CHECK_NULLPTR(aicpuStreamManager, "aicpuStreamManager is nullptr!");
    return *aicpuStreamManager;
}

CollServiceBase* CommunicatorImpl::GetCollService() const { return collService; }

CollServiceBase* CommunicatorImpl::GetCcuCollService() const
{
    // 仅在Task Exception下使用，异常捕获由TaskExceptionHandler::Process管理
    if (collServices.find(AcceleratorState::CCU_SCHED) != collServices.end()) {
        return collServices.at(AcceleratorState::CCU_SCHED).get();
    } else {
        std::string msg{"[CommunicatorImpl] Communicator uninitialized, this should not be arrived"};
        MACRO_THROW(NullPtrException, msg);
    }
}

SocketManager& CommunicatorImpl::GetSocketManager() const
{
    CHECK_NULLPTR(socketManager, "socketManager is nullptr!");
    return *socketManager;
}

RmaConnManager& CommunicatorImpl::GetRmaConnManager() const
{
    CHECK_NULLPTR(rmaConnectionManager, "rmaConnectionManager is nullptr!");
    return *rmaConnectionManager;
}

CollOperator* CommunicatorImpl::GetCurrentCollOperator() const
{
    CHECK_NULLPTR(currentCollOperator, "currentCollOperator is nullptr!");
    return currentCollOperator.get();
}

MemTransportManager* CommunicatorImpl::GetMemTransportManager() const { return memTransportManager.get(); }

bool CommunicatorImpl::GetOpCcuFeatureFlag() const
{
    return IsOpUsingCcuMs() || IsOpUsingCcuSched(); // 算子粒度
}

bool CommunicatorImpl::GetCommCcuFeatureFlag() const
{
    return IsCommUsingCcuMs() || IsCommUsingCcuSched(); // 通信域粒度
}

bool CommunicatorImpl::IsOpUsingAiv() const { return GetOpExecuteConfig().accState == AcceleratorState::AIV; }

bool CommunicatorImpl::IsOpUsingAivOnly() const { return GetOpExecuteConfig().accState == AcceleratorState::AIV_ONLY; }

bool CommunicatorImpl::GetOpAivFeatureFlag() const { return IsOpUsingAiv() || IsOpUsingAivOnly(); }

HcclResult CommunicatorImpl::AllocCommResource(void* mc2Tiling, void** commContext)
{
    try {
        AcceleratorState acceleratorState;
        CHK_RET(GetTilingAccelerator(mc2Tiling, acceleratorState));
        OpExecuteConfig mc2AcceConfig;
        mc2AcceConfig.accState = acceleratorState;
        SetOpExecuteConfig(mc2AcceConfig);
        SelectCollService();
        isLoadOp = true;
        WaitReady();
        collService->AllocCommResource(mc2Tiling, commContext, acceleratorState);
    } catch (HcclException& e) {
        HCCL_ERROR(e.what());
        PrintBackTrace(e);
        return e.GetErrorCode();
    } catch (exception& e) {
        HCCL_ERROR(e.what());
        return HcclResult::HCCL_E_INTERNAL;
    } catch (...) {
        HCCL_ERROR("Unknown error occurs!");
        return HcclResult::HCCL_E_INTERNAL;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::GetCcuTaskInfo(void* tilingData, void* ccuTaskGroup) const
{
    if (!GetCommCcuFeatureFlag()) { // 通信域粒度
        HCCL_ERROR("CommunicatorImpl::GetCcuTaskInfo: ccu is not used, can't GetCcuTaskInfo.");
        return HCCL_E_NOT_SUPPORT;
    }
    try {
        WaitReady();
        collService->GetCcuTaskInfo(tilingData, ccuTaskGroup);
    } catch (HcclException& e) {
        HCCL_ERROR(e.what());
        PrintBackTrace(e);
        return e.GetErrorCode();
    } catch (exception& e) {
        HCCL_ERROR(e.what());
        return HcclResult::HCCL_E_INTERNAL;
    } catch (...) {
        HCCL_ERROR("Unknown error occurs!");
        return HcclResult::HCCL_E_INTERNAL;
    }
    return HcclResult::HCCL_SUCCESS;
}

u32 CommunicatorImpl::GetCcuMc2ServerNum()
{
    if (collServices.find(AcceleratorState::CCU_MS) == collServices.end()
        || collServices.find(AcceleratorState::CCU_SCHED) == collServices.end()) {
        THROW<InternalException>(
            "[CommunicatorImpl][%s] not create collServices type "
            "CCU_MS and CCU_SCHED",
            __func__);
    }

    auto ccuMc2ServerNum = collServices[AcceleratorState::CCU_MS]->GetCcuMc2ServerNum();

    return ccuMc2ServerNum;
}

/* topoDescs 当前只支持l0和l1 */
HcclResult CommunicatorImpl::GetTopoDesc(HcclTopoDescs* topoDescs, uint32_t topoSize) const
{
    if (topoSize < static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_MAX)) {
        HCCL_ERROR("topoDescs size is not enough, please check topoSize[%u]", topoSize);
        return HCCL_E_PARA;
    }

    topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L0)].algSets = HCCL_ALG_MESH;
    topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L1)].algSets = 0;

    topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L0)].rankSize = rankSize;
    topoDescs[static_cast<uint32_t>(HcclTopoLevel::HCCL_TOPO_L1)].rankSize = 0;

    return HcclResult::HCCL_SUCCESS;
}

HostDeviceSyncNotifyManager& CommunicatorImpl::GetHostDeviceSyncNotifyManager() const
{
    return *hostDeviceSyncNotifyManager;
}

Trace& CommunicatorImpl::GetTrace() const { return *trace; }

HDCommunicate& CommunicatorImpl::GetKfcControlTransferH2D() const { return *kfcControlTransferH2D; }

HDCommunicate& CommunicatorImpl::GetKfcStatusTransferD2H() const { return *kfcStatusTransferD2H; }

constexpr u32 WAIT_CMD_TIMEOUT = 10 * 1000; // 最大等待10秒

HcclResult CommunicatorImpl::Suspend()
{
    TRY_CATCH_RETURN(
        if (isSuspended) {
            HCCL_WARNING(
                "[NsRecovery][Suspend] The current communication has been suspended, no need to suspend again.");
            return HcclResult::HCCL_SUCCESS;
        } isSuspended
        = true;
        if (!isAicpuKernelLaunched) {
            HCCL_INFO("[NsRecovery][Suspend] Aicpu kernel is not launched yet. Suspend host only.");
            return HcclResult::HCCL_SUCCESS;
        } KfcCommand opCmd
        = KfcCommand::NS_STOP_LAUNCH;
        CHK_RET(kfcControlTransferH2D->Put(0, sizeof(KfcCommand), reinterpret_cast<uint8_t*>(&opCmd)));
        HCCL_INFO("[NsRecovery][Suspend] send KfcCommand[%d] success, which is NS_STOP_LAUNCH.", opCmd);
        KfcExecStatus opInfo; auto timeout = std::chrono::milliseconds(WAIT_CMD_TIMEOUT);
        auto startTime = std::chrono::steady_clock::now(); while (true) {
            CHK_RET(kfcStatusTransferD2H->Get(0, sizeof(KfcExecStatus), reinterpret_cast<uint8_t*>(&opInfo)));
            if (opInfo.kfcStatus == KfcStatus::STOP_LAUNCH_DONE) {
                HCCL_INFO("[NsRecovery][Suspend] received KfcStatus[%d], which is STOP_LAUNCH_DONE", opInfo.kfcStatus);
                return HcclResult::HCCL_E_SUSPENDING;
            } else if (opInfo.kfcStatus == KfcStatus::ERROR) {
                HCCL_ERROR("[NsRecovery][Suspend] received KfcStatus[%d], which is ERROR", opInfo.kfcStatus);
                return HcclResult::HCCL_E_INTERNAL;
            } else {
                if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
                    HCCL_ERROR(
                        "[NsRecovery][Suspend] Wait suspend response status timeout[%u ms] and get the opExecStatus is "
                        "[%u].",
                        WAIT_CMD_TIMEOUT, opInfo.kfcStatus);
                    return HcclResult::HCCL_E_TIMEOUT;
                }
                continue;
            }
        });
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::Clean()
{
    TRY_CATCH_RETURN(
        if (!isSuspended) {
            HCCL_ERROR("[NsRecovery][Clean] The current communication is not suspended, cannot clean.");
            return HcclResult::HCCL_E_NOT_SUPPORT;
        } isSuspended
        = true;
        if (isCleaned) {
            HCCL_WARNING("[NsRecovery][Clean] The current communication has been cleaned, no need to clean again.");
            return HcclResult::HCCL_SUCCESS;
        } isCleaned
        = true;
        // 清理host侧资源
        if (GetOpCcuFeatureFlag()) {      // 算子粒度加速模式
            if (collService == nullptr) { // 当前通信域没下发过算子
                HCCL_WARNING("[NsRecovery][Clean] The current communication has not loaded op, no need to clean.");
                return HcclResult::HCCL_SUCCESS;
            }
            HCCL_INFO("[NsRecovery][Clean] start to clean host. ccu flag is true");
            auto collServiceCcu = dynamic_cast<CollServiceDeviceMode*>(collService);
            CHECK_NULLPTR(collServiceCcu, "collServiceBase cast to CollServiceDeviceMode failed.");

            CcuInsPreprocessor* ccuInsPreprocessor = collServiceCcu->GetCcuInsPreprocessor();
            CHECK_NULLPTR(ccuInsPreprocessor, "ccuInsPreprocessor is nullptr!");

            CcuCommunicator* ccuComm = ccuInsPreprocessor->GetCcuComm();
            CHECK_NULLPTR(ccuComm, "ccuComm is nullptr!");

            CcuTransportMgr* ccuTransportMgr = ccuComm->GetCcuTransportMgr();
            CHECK_NULLPTR(ccuTransportMgr, "ccuTransportMgr is nullptr!");
            ccuTransportMgr->Clean();
            return HcclResult::HCCL_SUCCESS;
        } else {
            HCCL_INFO("[NsRecovery][Clean] start to clean host. ccu flag is false");
            rmaConnectionManager->Clear();
            memTransportManager->Clear();
        } if (!isAicpuKernelLaunched) {
            HCCL_INFO("[NsRecovery][Clean] Aicpu kernel is not launched yet. Clean host only.");
            return HcclResult::HCCL_SUCCESS;
        } HCCL_INFO("[NsRecovery][Clean] start to clean device, waiting for device STOP_LAUNCH_DONE");
        KfcExecStatus opInfo;
        CHK_RET(kfcStatusTransferD2H->Get(0, sizeof(KfcExecStatus), reinterpret_cast<uint8_t*>(&opInfo)));
        if (opInfo.kfcStatus == KfcStatus::STOP_LAUNCH_DONE) {
            HCCL_INFO("[NsRecovery][Clean] received KfcStatus[%d], which is STOP_LAUNCH_DONE", opInfo.kfcStatus);
            // 通知背景线程清理device侧资源
            KfcCommand opCmd = KfcCommand::NS_CLEAN;
            CHK_RET(kfcControlTransferH2D->Put(0, sizeof(KfcCommand), reinterpret_cast<uint8_t*>(&opCmd)));
            HCCL_INFO("[NsRecovery][Clean] send KfcCommand [%d] success, which is NS_CLEAN", opCmd);
            // 监听背景线程状态
            auto timeout = std::chrono::milliseconds(WAIT_CMD_TIMEOUT);
            auto startTime = std::chrono::steady_clock::now();
            while (true) {
                CHK_RET(kfcStatusTransferD2H->Get(0, sizeof(KfcExecStatus), reinterpret_cast<uint8_t*>(&opInfo)));
                if (opInfo.kfcStatus == KfcStatus::CLEAN_DONE) {
                    HCCL_INFO("[NsRecovery][Clean] received KfcStatus[%d], which is CLEAN_DONE", opInfo.kfcStatus);
                    return HcclResult::HCCL_E_SUSPENDING;
                } else if (opInfo.kfcStatus == KfcStatus::ERROR) {
                    HCCL_ERROR("[NsRecovery][Clean] received KfcStatus[%d], which is ERROR", opInfo.kfcStatus);
                    return HcclResult::HCCL_E_INTERNAL;
                } else {
                    if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
                        HCCL_ERROR(
                            "[NsRecovery][Clean] Wait clean response status timeout[%u ms] and get the opExecStatus is "
                            "[%u].",
                            WAIT_CMD_TIMEOUT, opInfo.kfcStatus);
                        return HcclResult::HCCL_E_TIMEOUT;
                    }
                    continue;
                }
            }
        } else {
            std::string msg = StringFormat("[NsRecovery][Clean] Aicpu kernel is not stopped yet. Cannot clean.");
            THROW<InternalException>(msg);
            return HcclResult::HCCL_E_INTERNAL;
        });
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::Resume()
{
    TRY_CATCH_RETURN(
        if (GetCommStatus() == CommStatus::COMM_ERROR) {
            HCCL_ERROR("[NsRecovery][Resume] Comm has been error, can not resume now!");
            return HcclResult::HCCL_E_INTERNAL;
        } if (!isSuspended) {
            HCCL_WARNING("[NsRecovery][Resume] The current communication is normal, no need to resume.");
            return HcclResult::HCCL_SUCCESS;
        } if (GetOpCcuFeatureFlag() || GetOpAiCpuTSFeatureFlag() || GetOpAivFeatureFlag()) { // CCU和AICPU AIV //
                                                                                             // 算子粒度加速模式
            HCCL_INFO("[NsRecovery][Resume] start to Resume.");
            if (collService != nullptr) {
                collService->Resume();
            } else { // 当前通信域没下发过算子
                HCCL_WARNING("[NsRecovery][Resume] The current communication has not loaded op, no need to resume.");
            }
            isSuspended = false;
            isCleaned = false;
            HCCL_INFO("[NsRecovery][Resume] Resume success.");
        } else { // HOST场景不支持
            HCCL_ERROR("[NsRecovery][Resume] HOST is not supported to resume.");
            return HcclResult::HCCL_E_NOT_SUPPORT;
        });
    return HcclResult::HCCL_SUCCESS;
}

const NotifyTimeoutCfg& CommunicatorImpl::GetNotifyTimeoutCfg() const { return notifyTimeoutCfg; }

/* 当前接口中申请的buffer都是ge图模式下使用 */
HcclResult CommunicatorImpl::CreateCommCclBuf()
{
    HCCL_INFO("[%s] start.", __func__);
    if (inCclBuffer == nullptr) {
        inCclBuffer = std::make_shared<DevBuffer>(cclBufferSize);
        HCCL_INFO("CommunicatorImpl::CreateCommCclBuf, inCclBuffer is %p", inCclBuffer.get());
    }
    if (outCclBuffer == nullptr) {
        outCclBuffer = std::make_shared<DevBuffer>(cclBufferSize);
        HCCL_INFO("CommunicatorImpl::CreateCommCclBuf, outCclBuffer is %p", outCclBuffer.get());
    }
    if (indirectInCclBuffer == nullptr) {
        indirectInCclBuffer = std::make_shared<DevBuffer>(sizeof(uintptr_t));
        HCCL_INFO("Create Indirect In CclBuf success, indirectInCclBuffer = %p", indirectInCclBuffer.get());
    }
    if (indirectOutCclBuffer == nullptr) {
        indirectOutCclBuffer = std::make_shared<DevBuffer>(sizeof(uintptr_t));
        HCCL_INFO("Create Indirect out CclBuf success, indirectOutCclBuffer = %p", indirectOutCclBuffer.get());
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::GetInCclBuf(void*& commInputPtr, u64& commInputSize)
{
    CHK_PTR_NULL(inCclBuffer);
    commInputSize = inCclBuffer->GetSize();
    commInputPtr = reinterpret_cast<void*>(inCclBuffer->GetAddr());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::GetOutCclBuf(void*& commOutputPtr, u64& commOutputSize)
{
    CHK_PTR_NULL(outCclBuffer);
    commOutputSize = outCclBuffer->GetSize();
    commOutputPtr = reinterpret_cast<void*>(outCclBuffer->GetAddr());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::GetIndirectInCclBuf(void*& commIndirectInputPtr, u64& commIndirectInputSize)
{
    HCCL_INFO("[%s] start.", __func__);
    CreateCommCclBuf();
    commIndirectInputPtr = reinterpret_cast<void*>(indirectInCclBuffer->GetAddr());
    commIndirectInputSize = indirectInCclBuffer->GetSize();
    HCCL_INFO(
        "GetIndirectInCclBuf: commIndirectInputPtr[%p], commIndirectInputSize[%lu]", commIndirectInputPtr,
        commIndirectInputSize);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::GetIndirectOutCclBuf(void*& commIndirectOutputPtr, u64& commIndirectOutputSize)
{
    HCCL_INFO("[%s] start.", __func__);
    CreateCommCclBuf();
    commIndirectOutputPtr = reinterpret_cast<void*>(indirectOutCclBuffer->GetAddr());
    commIndirectOutputSize = indirectOutCclBuffer->GetSize();
    HCCL_INFO(
        "GetIndirectOutCclBuf: commIndirectOutputPtr[%p], commIndirectOutputSize[%lu]", commIndirectOutputPtr,
        commIndirectOutputSize);
    return HcclResult::HCCL_SUCCESS;
}

bool CommunicatorImpl::IsWorldGroup() const { return isWorldGroup; }

bool CommunicatorImpl::IsCommReady()
{
    CHECK_NULLPTR(collService, "[CommunicatorImpl::IsCommReady] collService is nullptr!");
    if (collService->IsAllTransportRecoveredReady(GetId())) {
        // 遗留问题：对Comm状态置为ready
        SetCommStatus(CommStatus::COMM_READY);
        return true;
    } else {
        return false;
    }
}

HcclResult CommunicatorImpl::GetSnapShotDynamicBuf(BinaryStream& buf) const
{
    HCCL_INFO(
        "[CommunicatorImpl][%s] opExecuteConfig.accState is [%u], commExecuteConfig.accState "
        "is [%u], isLoadOp is [%d]",
        __func__, static_cast<u32>(opExecuteConfig.accState), static_cast<u32>(commExecuteConfig.accState), isLoadOp);
    buf << static_cast<u32>(opExecuteConfig.accState); // 算子粒度 和 通信域粒度 都保存
    buf << static_cast<u32>(commExecuteConfig.accState);
    buf << isLoadOp;

    buf << submittedOpCnt;
    HCCL_INFO("[CommunicatorImpl][%s], rank[%d], submittedOpCnt[%u]", __func__, myRank, submittedOpCnt);
    if (submittedOpCnt == 0) {
        return HcclResult::HCCL_SUCCESS;
    }

    if (currentCollOperator) {
        HCCL_INFO("[CommunicatorImpl][%s] opMode is %u", __func__, static_cast<u32>(currentCollOperator->opMode));
        buf << static_cast<u32>(currentCollOperator->opMode);

        HCCL_INFO("[CommunicatorImpl][%s] rank[%d], currentCollOperator", __func__, myRank);
        TRY_CATCH_RETURN(collService->GetSnapShotDynamicBuf(*currentCollOperator, buf));
    }
    return HcclResult::HCCL_SUCCESS;
}

u32 CommunicatorImpl::GetRanktableCrc(bool isContainLoaId) const
{
    HCCL_INFO("[CommunicatorImpl][%s], rank[%d], id[%s], idIdex[%u]", __func__, myRank, id.c_str(), idIndex);
    CHK_PTR_NULL(ranktableInfo);
    vector<char> ranktableBuf = ranktableInfo->GetUniqueId(isContainLoaId);
    CheckCrc crc;
    u32 crcValue = 0;
    auto ret = crc.Calc32Crc(reinterpret_cast<const char*>(ranktableBuf.data()), ranktableBuf.size(), &crcValue);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[CommunicatorImpl][GetRanktableCrc] calculate crc failed, ret[%d]", ret), ret);
    return crcValue;
}

// 恢复全局通信域
HcclResult CommunicatorImpl::RecoverComm(SnapShotComm& snapShotComm, u32 stepParam, const char* changeInfo)
{
    if (!initFlag) {
        initFlag = true;
        try {
            HCCL_INFO("[CommunicatorImpl][%s], rank[%d]", __func__, myRank);
            // 将状态设置为resuming
            if (GetCommStatus() == CommStatus::COMM_IDLE) {
                SetCommStatus(CommStatus::COMM_RESUMING);
            } else {
                HCCL_ERROR("Communicator status is not idle, can not resume!");
                return HcclResult::HCCL_E_INTERNAL;
            }
            RecoverOpMode(snapShotComm.opMode);
            InitCommonData(snapShotComm.commParams, snapShotComm.config);
            HrtSetDevice(devLogicId);
            InitHccpHdc(); // 选择ccu加速模式依赖hdc通道打开ccu驱动
            RecoverExeCfgData(
                snapShotComm.opExecuteConfig, snapShotComm.commExecuteConfig,
                snapShotComm.isLoadOp); // 算子粒度 和 通信域粒度都恢复
            HcclResult recoverRet = RecoverRankGraphData(snapShotComm, changeInfo);
            if (recoverRet != HCCL_SUCCESS) {
                SetCommStatus(CommStatus::COMM_IDLE);
                return recoverRet;
            }
            InitNotifyManager();
            InitStreamManager();
            InitSocketManager();
            InitRmaConnManager();
            InitDataBufferManager();
            InitNotifyFixedValue();
            InitMemTransportManager();
            InitHostDeviceSyncNotifyManager();
            InitUbMemoryTransportMgr();
            CollAlgComponentInit();
            RegisterAicpuKernel();
            InitCollService();
            SelectCollService();
            InitTraceManager();
            DlProfFunction::GetInstance().DlProfFunctionInit();
            InitMirrorTaskManager();
            CHK_RET(InitProfilingReporter());
            InitTaskExceptionHandler();
            InitHDCommunicate();
            notifyTimeoutCfg.Init();
            RecoverTransportData(
                snapShotComm.submittedOpCnt, snapShotComm.levelRankPairs, stepParam, snapShotComm.linkGroupPair);
        } catch (HcclException& e) {
            // 异常时状态返回IDLE
            SetCommStatus(CommStatus::COMM_IDLE);
            HCCL_ERROR(e.what());
            PrintBackTrace(e);
            return e.GetErrorCode();
        } catch (exception& e) {
            // 异常时状态返回IDLE
            SetCommStatus(CommStatus::COMM_IDLE);
            HCCL_ERROR(e.what());
            return HcclResult::HCCL_E_INTERNAL;
        } catch (...) {
            // 异常时状态返回IDLE
            SetCommStatus(CommStatus::COMM_IDLE);
            HCCL_ERROR("Unknown error occurs!");
            return HcclResult::HCCL_E_INTERNAL;
        }
        return HcclResult::HCCL_SUCCESS;
    }
    HCCL_ERROR("[CommunicatorImpl][%s] Repeated calling init method!", __func__);
    return HcclResult::HCCL_E_INTERNAL;
}

// 恢复子通信域
HcclResult CommunicatorImpl::RecoverComm(
    const SnapShotSubComm& snapShotSubComm, std::unique_ptr<RankGraph>& inputRankGraph, u32 inputStep)
{
    if (!initFlag) {
        initFlag = true;
        try {
            HCCL_INFO("[CommunicatorImpl][%s], rank[%d]", __func__, myRank);
            // 将状态设置为resuming
            if (GetCommStatus() == CommStatus::COMM_IDLE) {
                SetCommStatus(CommStatus::COMM_RESUMING);
            } else {
                HCCL_ERROR("Communicator status is not idle, can not resume!");
                return HcclResult::HCCL_E_INTERNAL;
            }
            RecoverOpMode(snapShotSubComm.opMode);
            InitCommonDataNotInitDevType(snapShotSubComm.commParams, snapShotSubComm.config);
            HrtSetDevice(devLogicId);
            InitHccpHdc(); // 选择ccu加速模式依赖hdc通道打开ccu驱动
            RecoverExeCfgData(
                snapShotSubComm.opExecuteConfig, snapShotSubComm.commExecuteConfig,
                snapShotSubComm.isLoadOp); // 算子粒度 和 通信域粒度都恢复
            InitRankGraph(inputRankGraph);
            InitNotifyManager();
            InitStreamManager();
            InitSocketManager();
            InitRmaConnManager();
            InitDataBufferManager();
            InitNotifyFixedValue();
            InitMemTransportManager();
            InitHostDeviceSyncNotifyManager();
            InitUbMemoryTransportMgr();
            CollAlgComponentInit();
            RegisterAicpuKernel();
            InitCollService();
            SelectCollService();
            InitTraceManager();
            DlProfFunction::GetInstance().DlProfFunctionInit();
            InitMirrorTaskManager();
            CHK_RET(InitProfilingReporter());
            InitTaskExceptionHandler();
            InitHDCommunicate();
            RecoverTransportData(
                snapShotSubComm.submittedOpCnt, snapShotSubComm.levelRankPairs, inputStep,
                snapShotSubComm.linkGroupPair);
        } catch (HcclException& e) {
            // 异常时状态返回IDLE
            SetCommStatus(CommStatus::COMM_IDLE);
            HCCL_ERROR(e.what());
            PrintBackTrace(e);
            return e.GetErrorCode();
        } catch (exception& e) {
            // 异常时状态返回IDLE
            SetCommStatus(CommStatus::COMM_IDLE);
            HCCL_ERROR(e.what());
            return HcclResult::HCCL_E_INTERNAL;
        } catch (...) {
            // 异常时状态返回IDLE
            SetCommStatus(CommStatus::COMM_IDLE);
            HCCL_ERROR("Unknown error occurs!");
            return HcclResult::HCCL_E_INTERNAL;
        }
        return HcclResult::HCCL_SUCCESS;
    }
    HCCL_ERROR("Repeated calling init method!");
    return HcclResult::HCCL_E_INTERNAL;
}
HcclResult CommunicatorImpl::RecoverOpMode(u32 opMode)
{
    if (currentCollOperator == nullptr) {
        currentCollOperator = make_unique<CollOperator>();
    }
    currentCollOperator->opMode = static_cast<OpMode::Value>(opMode);
    return HcclResult::HCCL_SUCCESS;
}
// 创建子虚拟拓扑并恢复子通信域
HcclResult
CommunicatorImpl::RecoverSubComm(const SnapShotSubComm& snapShotSubComm, CommunicatorImpl* subCommImpl, u32 step)
{
    HCCL_INFO("[CommunicatorImpl][%s] start, myRank is [%d]", __func__, myRank);
    vector<u32> rankIds;
    for (u32 i = 0; i < snapShotSubComm.rankIds.size(); ++i) {
        rankIds.push_back(static_cast<u32>(snapShotSubComm.rankIds[i]));
    }
    try {
        if (initFlag) {
            // 创建子虚拟拓扑
            std::unique_ptr<RankGraph> subRankGraph = rankGraph->CreateSubRankGraph(rankIds);
            // 初始化子通信域
            return subCommImpl->RecoverComm(snapShotSubComm, subRankGraph, step);
        } else {
            // 异常时状态返回IDLE
            SetCommStatus(CommStatus::COMM_IDLE);
            std::string msg = StringFormat("CreateSubComm fail, communicator has not been initialized, please check.");
            THROW<InternalException>(msg);
        }
    } catch (HcclException& e) {
        // 异常时状态返回IDLE
        SetCommStatus(CommStatus::COMM_IDLE);
        HCCL_ERROR(e.what());
        PrintBackTrace(e);
        return e.GetErrorCode();
    } catch (exception& e) {
        // 异常时状态返回IDLE
        SetCommStatus(CommStatus::COMM_IDLE);
        HCCL_ERROR(e.what());
        return HcclResult::HCCL_E_INTERNAL;
    } catch (...) {
        // 异常时状态返回IDLE
        SetCommStatus(CommStatus::COMM_IDLE);
        HCCL_ERROR("Unknown error occurs!");
        return HcclResult::HCCL_E_INTERNAL;
    }
    HCCL_ERROR("CreateSubComm fail !");
    return HcclResult::HCCL_E_INTERNAL;
}
// 恢复全局通信域拓扑信息
HcclResult CommunicatorImpl::RecoverRankGraphData(SnapShotComm& snapShotComm, const char* changeInfo)
{
    HCCL_INFO("[CommunicatorImpl][%s] start, rank[%d]", __func__, myRank);

    // 根据changedInfo更新快照信息
    auto ret = DiffRankUpdater(changeInfo, snapShotComm.rankTableInfo);
    if (ret != HcclResult::HCCL_SUCCESS) {
        THROW<InternalException>("DiffRankUpdater failed");
    }

    const RankGraphBuilderBridge* bridge = GetRankGraphBuilderBridge();
    CHK_PRT_RET(
        bridge == nullptr || bridge->recoverBuild == nullptr,
        HCCL_ERROR("[%s] RankGraphBuilder bridge is not registered.", __func__), HCCL_E_INTERNAL);

    RankGraphBuildResult buildResult{};
    // 恢复构建也由 hcomm 完成，并返回同一版本的 graph、rank table 和 topo 快照。
    ret = bridge->recoverBuild(snapShotComm.rankTableInfo, snapShotComm.topoInfo, myRank, buildResult);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s] recover rank graph failed, errNo[0x%016llx].", __func__, HCCL_ERROR_CODE(ret)), ret);
    CHK_PRT_RET(
        buildResult.rankGraph == nullptr, HCCL_ERROR("[%s] recover rank graph result is invalid.", __func__),
        HCCL_E_INTERNAL);

    std::unique_ptr<RankTableInfo> preparedRankTableInfo;
    std::shared_ptr<TopoInfo> preparedTopoInfo;
    PrepareRankGraphMetadata(buildResult.rankTableInfo, buildResult.topoInfo, preparedRankTableInfo, preparedTopoInfo);

    const u32 preparedRankSize = buildResult.rankGraph->GetRankSize();
    // 仅在恢复结果和两份元数据准备完整后，用不分配内存的移动赋值整体替换拓扑状态。
    rankGraph = std::move(buildResult.rankGraph);
    ranktableInfo = std::move(preparedRankTableInfo);
    topoInfo = std::move(preparedTopoInfo);
    rankSize = preparedRankSize;
    HCCL_INFO(
        "[CommunicatorImpl][%s] Recover topo data from snapshot, rank[%d], id[%s], idIndex[%u], RankTableInfo[%s]",
        __func__, myRank, id.c_str(), idIndex, ranktableInfo->Describe().c_str());

    CheckRankGraph();
    HCCL_INFO("Recover topo data from snapshot success.");
    return HcclResult::HCCL_SUCCESS;
}
// 恢复通信域transport信息
HcclResult CommunicatorImpl::RecoverTransportData(
    u32 savedSubmittedOpCnt, const vector<std::pair<u32, RankId>>& levelRankPairs, u32 savedStep,
    vector<std::pair<LinkGroup, u32>> linkGroupPair)
{
    HCCL_INFO(
        "[CommunicatorImpl][%s] Recover transport data from snapshot.levelRankPairs size is %u", __func__,
        levelRankPairs.size());
    vector<LinkData> links;

    for (uint32_t i = 0; i < levelRankPairs.size(); ++i) {
        CHK_PTR_NULL(rankGraph);
        std::vector<NetInstance::Path> paths
            = rankGraph->GetPaths(levelRankPairs[i].first, myRank, levelRankPairs[i].second);
        for (NetInstance::Path& path : paths) {
            links.emplace_back(LinkData(path));
        }
    }
    // 指令的下标是指令的个数 - 1
    collOpIndex = savedSubmittedOpCnt - 1;
    step = savedStep;
    // 建transport
    collService->RecoverTransport(links, linkGroupPair);
    HCCL_INFO("Recover transport data from snapshot success.");
    return HcclResult::HCCL_SUCCESS;
}

void CommunicatorImpl::WaitReady() const
{
    constexpr u32 loadWaitTimeOut = 10 * 1000; // 待修改，定义最大等待10秒
    auto timeout = std::chrono::milliseconds(loadWaitTimeOut);

    HCCL_INFO("[CommunicatorImpl][%s] start", __func__);
    HcclUs startTime = std::chrono::steady_clock::now();
    while (true) {
        if (GetCommStatus() == CommStatus::COMM_READY) {
            break;
        }
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            THROW<InternalException>("Wait COMM_READY timeout, commId[%s]", id.c_str());
        }
    }
    HCCL_INFO("[CommunicatorImpl][%s] end", __func__);
}

u32 CommunicatorImpl::GetCollOpIndex() const { return collOpIndex; }

u32 CommunicatorImpl::GetStep() const { return step; }

std::set<RankId> CommunicatorImpl::GetNeighboorRanks() const
{
    return rankGraph->GetNetInstanceByRankId(0, myRank)->GetRankIds();
}

void CommunicatorImpl::InitMirrorTaskManager()
{
    mirrorTaskManager
        = std::make_unique<MirrorTaskManager>(devLogicId, &GlobalMirrorTasks::Instance(), false); // host侧写死
}

MirrorTaskManager& CommunicatorImpl::GetMirrorTaskManager() const
{
    CHECK_NULLPTR(mirrorTaskManager, "mirrorTaskManager is nullptr!");
    return *mirrorTaskManager;
}

CommunicatorImpl::~CommunicatorImpl() { DECTOR_TRY_CATCH("CommunicatorImpl", DestroyImpl();) }

void CommunicatorImpl::DestroyImpl()
{
    HCCL_INFO("[~CommunicatorImpl] start CommunicatorImpl destroy, commId[%s]", id.c_str());
    (void)DestroyDpuKernelResource();
    (void)DestroyDpuTaskexpShmemInDevice();
    // 释放
    if (hostShareBuf != nullptr) {
        free(hostShareBuf);
        hostShareBuf = nullptr;
    }
    {
        std::lock_guard<std::mutex> lock(g_serMapMutex);
        auto outerIt = g_taskServiceMap.find(id);
        if (outerIt != g_taskServiceMap.end()) {
            outerIt->second.erase(devLogicId);
            if (outerIt->second.empty()) {
                g_taskServiceMap.erase(id);
            }
        }
        auto expOuterIt = g_taskExpMemMap.find(id);
        if (expOuterIt != g_taskExpMemMap.end()) {
            expOuterIt->second.erase(devLogicId);
            if (expOuterIt->second.empty()) {
                g_taskExpMemMap.erase(id);
            }
        }
    }
    (void)DestroyKFCWorkSpaceVA();
    (void)NotifyAicpuDestroyComm();
    ccuDrvHandle = nullptr;

    // 释放 device 内存
    if (notifyFixedValue != nullptr) {
        HrtFree(notifyFixedValue);
        notifyFixedValue = nullptr;
    }

    DeInitPreResource();
}

HcclResult CommunicatorImpl::DestroyDpuKernelResource()
{
    // 终止Dpu Kernel的TaskRun
    if (!isDpuKernelLaunched) {
        return HCCL_SUCCESS;
    }

    CHK_RET(WaitDpuKernelThreadTerminate());

    // 切换回 dpu ctx
    aclError aclRet = aclrtSetCurrentContext(dpuContext);
    if (ACL_SUCCESS != aclRet) {
        HCCL_ERROR("set dpu Ctx Failed, aclReturn[%d]", aclRet);
        return HCCL_E_RUNTIME;
    }
    // 销毁局部流
    aclRet = aclrtDestroyStreamForce(dpuStream);
    if (ACL_SUCCESS != aclRet) {
        HCCL_ERROR("Destroy Stream Failed, aclReturn[%d]", aclRet);
        aclRet = aclrtSetCurrentContext(npuContext);
        CHK_PRT_RET(aclRet == ACL_SUCCESS, HCCL_ERROR("set npu Ctx Failed, aclReturn[%d]", aclRet), HCCL_E_RUNTIME);
        return HCCL_E_RUNTIME;
    }
    if (g_commNum > 1) {
        g_commNum--;
    } else {
        // reset DPU kernel 线程
        HcclResult ret = HrtResetXpuDevice(TEMP_DEV_TYPE_DPU, 0);
        if (HCCL_SUCCESS != ret) {
            HCCL_ERROR("ResetXpuDevice Failed, return[%d]", ret);
            aclRet = aclrtSetCurrentContext(npuContext);
            CHK_PRT_RET(aclRet == ACL_SUCCESS, HCCL_ERROR("set npu Ctx Failed, aclReturn[%d]", aclRet), HCCL_E_RUNTIME);
            return HCCL_E_RUNTIME;
        }
    }
    // 切回 npu ctx
    aclRet = aclrtSetCurrentContext(npuContext);
    if (ACL_SUCCESS != aclRet) {
        HCCL_ERROR("set npu Ctx Failed, aclReturn[%d]", aclRet);
        return HCCL_E_RUNTIME;
    }

    return HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::WaitDpuKernelThreadTerminate()
{
    if (!isDpuKernelLaunched) {
        return HCCL_SUCCESS;
    }

    if (tagDpuShmemArgsMap_[DPUTAG].va_ == nullptr || tagDpuShmemArgsMap_[DPUTAG].accessVA_ == nullptr) {
        HCCL_ERROR(
            "[CommunicatorImpl::%s] va_[%p] or accessVA_[%p] is nullptr", __func__, tagDpuShmemArgsMap_[DPUTAG].va_,
            tagDpuShmemArgsMap_[DPUTAG].accessVA_);
        return HCCL_E_MEMORY;
    }
    void* hostPtr = tagDpuShmemArgsMap_[DPUTAG].connectType_ == HOST_DEVICE_CONNECT_TYPE_UB ?
                        tagDpuShmemArgsMap_[DPUTAG].va_ :
                        tagDpuShmemArgsMap_[DPUTAG].accessVA_;
    // 先读取是否有dpu线程回写的退出信号
    uint8_t flag = 0;
    errno_t ret = memcpy_s(&flag, sizeof(flag), hostPtr, sizeof(flag));
    if (ret != EOK) {
        HCCL_ERROR("Read Terminate TaskRun Signal Fail, return[%d]", ret);
        return HCCL_E_INTERNAL;
    }
    if (flag == DEVICE_SIGNAL_THIRD) {
        return HCCL_SUCCESS; // dpu线程已退出
    }
    flag = DEVICE_SIGNAL_SECOND;
    ret = memcpy_s(hostPtr, sizeof(flag), &flag, sizeof(flag));
    if (ret != EOK) {
        HCCL_ERROR("Terminate TaskRun Fail, return[%d]", ret);
        return HCCL_E_INTERNAL;
    }
    do {
        ret = memcpy_s(&flag, sizeof(flag), hostPtr, sizeof(flag));
        if (ret != EOK) {
            HCCL_ERROR("Read Terminate TaskRun Signal Fail, return[%d]", ret);
            return HCCL_E_INTERNAL;
        }
    } while (flag != DEVICE_SIGNAL_THIRD);
    return HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::DestroyDpuTaskexpShmemInDevice()
{
    if (!isDpuKernelLaunched) { // 非dpu场景,无共享内存申请
        return HCCL_SUCCESS;
    }

    if (tagDpuShmemArgsMap_[DPUTASKEXCEPTION].va_ == nullptr
        || tagDpuShmemArgsMap_[DPUTASKEXCEPTION].accessVA_ == nullptr) {
        HCCL_ERROR(
            "[CommunicatorImpl::%s] va_[%p] or accessVA_[%p] is nullptr", __func__,
            tagDpuShmemArgsMap_[DPUTASKEXCEPTION].va_, tagDpuShmemArgsMap_[DPUTASKEXCEPTION].accessVA_);
        return HCCL_E_MEMORY;
    }
    void* hostPtr = tagDpuShmemArgsMap_[DPUTASKEXCEPTION].connectType_ == HOST_DEVICE_CONNECT_TYPE_UB ?
                        tagDpuShmemArgsMap_[DPUTASKEXCEPTION].va_ :
                        tagDpuShmemArgsMap_[DPUTASKEXCEPTION].accessVA_;
    uint8_t flag = 1;
    errno_t ret = memcpy_s(hostPtr, sizeof(flag), &flag, sizeof(flag)); // taskexception首字节作为是否停止标志位；
    if (ret != EOK) {
        HCCL_ERROR("DestroyDpuTaskexpShmemInDevice Fail, return[%d]", ret);
        return HCCL_E_INTERNAL;
    }
    auto timeout = std::chrono::milliseconds(WAIT_CMD_TIMEOUT);
    auto startTime = std::chrono::steady_clock::now();
    do { // 等待aicpu背景线程response
        ret = memcpy_s(&flag, sizeof(flag), hostPtr, sizeof(flag));
        if (ret != EOK) {
            HCCL_ERROR("Read Device Response Fail, return[%d]", ret);
            return HCCL_E_INTERNAL;
        }
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            HCCL_WARNING(
                "[CommunicatorImpl][DestroyDpuTaskexpShmemInDevice] Wait device destroy response timeout[%u ms].",
                WAIT_CMD_TIMEOUT);
            return HcclResult::HCCL_E_TIMEOUT;
        }
    } while (flag != 0);
    return HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::NotifyAicpuDestroyComm()
{
    if (!isAicpuKernelLaunched) {
        HCCL_WARNING("[%s] isAicpuKernelLaunched is false", __func__);
        return HcclResult::HCCL_SUCCESS;
    }

    if (kfcControlTransferH2D == nullptr) {
        HCCL_WARNING("[%s] kfcControlTransferH2D is null", __func__);
        return HcclResult::HCCL_SUCCESS;
    }

    KfcCommand opCmd = KfcCommand::DESTROY_AICPU_COMM;
    HCCL_INFO("[%s] send KfcCommand[%d] begin, which is DESTROY_AICPU_COMM.", __func__, opCmd);
    CHK_RET(kfcControlTransferH2D->Put(0, sizeof(KfcCommand), reinterpret_cast<uint8_t*>(&opCmd)));
    HCCL_INFO("[%s] send KfcCommand[%d] success, which is DESTROY_AICPU_COMM.", __func__, opCmd);
    KfcExecStatus opInfo;
    auto timeout = std::chrono::milliseconds(WAIT_CMD_TIMEOUT);
    auto startTime = std::chrono::steady_clock::now();
    while (true) {
        CHK_RET(kfcStatusTransferD2H->Get(0, sizeof(KfcExecStatus), reinterpret_cast<uint8_t*>(&opInfo)));
        if (opInfo.kfcStatus == KfcStatus::DESTROY_AICPU_COMM_DONE) {
            HCCL_INFO("[%s] get KfcStatus[%d], which is DESTROY_AICPU_COMM_DONE", __func__, opInfo.kfcStatus);
            return HcclResult::HCCL_SUCCESS;
        }
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            HCCL_WARNING(
                "[%s] Wait suspend response status timeout[%u ms] and get the "
                "opExecStatus is [%u].",
                __func__, WAIT_CMD_TIMEOUT, opInfo.kfcStatus);
            return HcclResult::HCCL_E_TIMEOUT;
        }
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::InitProfilingReporter()
{
    profilingReporter = std::make_unique<ProfilingReporter>(mirrorTaskManager.get(), &ProfilingHandler::GetInstance());
    CHK_RET(ProfilingHandler::GetInstance().Init());
    CHK_RET(profilingReporter->Init());
    return HCCL_SUCCESS;
}

ProfilingReporter& CommunicatorImpl::GetProfilingReporter() const
{
    CHECK_NULLPTR(profilingReporter, "profilingReporter is nullptr!");
    return *profilingReporter;
}

HcclResult CommunicatorImpl::GetOneSidedService(HcclOneSidedService** service) const
{
    CHECK_NULLPTR(oneSidedService, "oneSidedService is nullptr!");
    *service = oneSidedService.get();
    return HCCL_SUCCESS;
}

void CommunicatorImpl::UpdateProfStat() { profilingReporter->UpdateProfStat(); }

void CommunicatorImpl::ReportProfInfo(uint64_t beginTime, bool cachedReq, bool opbased)
{
    // 上报task信息
    profilingReporter->ReportAllTasks(cachedReq);

    // 上报opInfo信息
    profilingReporter->ReportOp(beginTime, cachedReq, opbased);
}

void CommunicatorImpl::InitTaskExceptionHandler() const
{
    TaskExceptionHandler* handler = TaskExceptionHandlerManager::GetHandler(static_cast<size_t>(devLogicId));
    CHECK_NULLPTR(handler, "handler is nullptr!");
}

void CommunicatorImpl::InitOneSidedService()
{
    HCCL_INFO("[CommunicatorImpl][InitOneSidedService] start!");
    oneSidedService = std::make_unique<HcclOneSidedService>(*this);
    HCCL_INFO("[CommunicatorImpl][InitOneSidedService] end!");
}

u32 CommunicatorImpl::GetUsedChannelCount(u32 dieId)
{
    CHECK_NULLPTR(collService, "collService is nullptr!");
    if (!GetOpCcuFeatureFlag()) {
        return 0;
    } // 防止非ccu模式进入
    CHECK_NULLPTR(dynamic_cast<CollServiceDeviceMode*>(collService), "CollServiceDeviceMode is nullptr!");
    CcuJettyMgr* ccuJettyMgr
        = dynamic_cast<CollServiceDeviceMode*>(collService)->GetCcuInsPreprocessor()->GetCcuComm()->GetCcuJettyMgr();

    if (ccuJettyMgr == nullptr) {
        HCCL_WARNING("[CommunicatorImpl][%s] failed, ccuJettyMgr is nullptr, commId[%s].", __func__, id.c_str());
        return 0;
    }
    return ccuJettyMgr->GetUsedChannelCount(dieId);
}

void CommunicatorImpl::RegisterPrintChannelInfoCallback(std::function<void()> callback)
{
    printChannelInfoCallback = callback;
}

void CommunicatorImpl::PrintChannelInfoCallback() const
{
    if (printChannelInfoCallback == nullptr) {
        HCCL_WARNING(
            "[CommunicatorImpl][PrintChannelInfoCallback] commId[%s], callback function not registered.", id.c_str());
        return;
    }
    // ccu建链时channel资源不足，调用回调函数做维测打印
    printChannelInfoCallback();
}

void CommunicatorImpl::SetCommStatus(CommStatus commStatus) { status_.store(commStatus); }

CommStatus CommunicatorImpl::GetCommStatus() const { return status_.load(); }

std::map<HcclAccelerator, AcceleratorState> accStateMap
    = {{HcclAccelerator::AICPU, AcceleratorState::AICPU_TS},
       {HcclAccelerator::AICPU_TS, AcceleratorState::AICPU_TS},
       {HcclAccelerator::CCU_SCHED, AcceleratorState::CCU_SCHED},
       {HcclAccelerator::DEFAULT, AcceleratorState::CCU_SCHED},
       {HcclAccelerator::CCU_MS, AcceleratorState::CCU_MS}};

// 初始化 算子粒度 = 通信域粒度 选择用 算子粒度 ok
void CommunicatorImpl::ExecAlgSelect(const CollOpParams& opParams, const OpMode& opMode)
{
    HCCL_INFO(
        "[CommunicatorImpl][%s] opType[%s], opMode[%s], primary accelerator[%s]", __func__,
        opParams.opType.Describe().c_str(), opMode.Describe().c_str(), opExecuteConfig.accState.Describe().c_str());
    // 调用算法选择接口，获取algName、展开方式、执行方式
    CollAlgParams params;
    params.opMode = opMode;
    params.maxTmpMemSize = GetBufferSize();
    params.isMc2 = opParams.isMc2;
    if (opParams.isMc2) {
        if (accStateMap.find(opParams.commEngine) == accStateMap.end()) {
            THROW<NotSupportException>(
                "[CommunicatorImpl][ExecAlgSelect] not support commEngine type[%s]!",
                opParams.commEngine.Describe().c_str());
        }
        opExecuteConfig.accState = accStateMap.find(opParams.commEngine)->second;
    }
    OpExecuteConfig inOpExecuteConfig = opExecuteConfig;
    params.opExecuteConfig = inOpExecuteConfig;
    params.algConfig = opParams.algConfig;

    HCCL_DEBUG(
        "CommunicatorImpl::ExecAlgSelect currentCollOperator dataType[%s]",
        currentCollOperator->dataType.Describe().c_str());
    auto ret = collAlgComponent->ExecAlgSelect(*currentCollOperator, params, curAlgName, inOpExecuteConfig);
    if (ret != HcclResult::HCCL_SUCCESS) {
        std::vector<HcclAlgoType> algos
            = std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
        auto configAlgMap = EnvConfig::GetInstance().GetAlgoConfig().GetAlgoConfig();
        auto it = configAlgMap.find(opParams.opType);
        if (it != configAlgMap.end()) {
            algos = it->second;
        }
        auto dataSize = opParams.count * DataTypeSizeGet(opParams.dataType);
        THROW<NotSupportException>(
            "[CommunicatorImpl][ExecAlgSelect] failed. Error code :%u, opType[%s], opMode[%s], accState[%s], "
            "dataType[%s], reduceOp[%s]. Current algName[%s],algos[0]:[%u],algos[1]:[%u],algos[2]:[%u],algos[3]:[%u], "
            "dataSize[%u Bytes] .",
            ret, opParams.opType.Describe().c_str(), opMode.Describe().c_str(),
            opExecuteConfig.accState.Describe().c_str(), opParams.dataType.Describe().c_str(),
            opParams.reduceOp.Describe().c_str(), curAlgName.c_str(), algos[0], algos[1], algos[2], algos[3], dataSize);
    }
    if (params.isMc2
        && (opExecuteConfig.accState == AcceleratorState::CCU_SCHED
            || opExecuteConfig.accState == AcceleratorState::CCU_MS)) {
        algorithmType_ = collAlgComponent->GetAlgorithmTypeForMC2CCU(curAlgName);
    }
    auto opAcceStateCacheIt = opAcceStateCache.find({opParams.opType, curAlgName});
    if (opAcceStateCacheIt != opAcceStateCache.end()) {
        HCCL_INFO(
            "[CommunicatorImpl][%s] opAcceStateCache find, reset accelerator[%s], algName[%s]", __func__,
            opAcceStateCacheIt->second.first.Describe().c_str(), opAcceStateCacheIt->second.second.c_str());
        opExecuteConfig.accState = opAcceStateCacheIt->second.first;
        curAlgName = opAcceStateCacheIt->second.second;
        ExecAlgSelect(opParams, opMode); // 重新走算法选择(数据量、数据类型、reduce类型不一样，算法可能不一样)
        return;
    }
    SetOpExecuteConfig(inOpExecuteConfig); // 算子粒度 ok
    HCCL_INFO(
        "[CommunicatorImpl][%s] current accelerator[%s], algName[%s], algorithmType[%u]", __func__,
        opExecuteConfig.accState.Describe().c_str(), curAlgName.c_str(), algorithmType_);
    SelectCollService();
}

void CommunicatorImpl::SelectCollService()
{
    // 根据执行方式和展开方式，选择对应的CollService
    auto mapIt = collServices.find(GetOpExecuteConfig().accState); // 算子粒度
    if (mapIt == collServices.end()) {
        auto msg = StringFormat(
            "[CommunicatorImpl][%s] not support, accelerator is %s", __func__,
            GetOpExecuteConfig().accState.Describe().c_str());
        THROW<NotSupportException>(msg);
    }
    collService = mapIt->second.get();
}

void CommunicatorImpl::CollAlgComponentInit()
{
    HcclMainboardId hcclMainboardId;
    HrtGetMainboardId(devLogicId, hcclMainboardId);
    CollAlgComponentBuilder collAlgComponentBuilder;
    collAlgComponent = collAlgComponentBuilder.SetRankGraph(GetRankGraph().get())
                           .SetDevType(GetDevType())
                           .SetMyRank(GetMyRank())
                           .SetRankSize(GetRankSize())
                           .SetDmaMode(DmaMode::PUT)
                           .SetMainboardId(static_cast<uint8_t>(hcclMainboardId))
                           .EnableDetour(
                               EnvConfig::GetInstance().GetDetourConfig().GetDetourType()
                               == HcclDetourType::HCCL_DETOUR_ENABLE_2P) // 当前仅支持2P绕路
                           .Build();
    if (collAlgComponent == nullptr) {
        HCCL_ERROR("collAlgComponent is a null pointer!");
        throw NullPtrException("collAlgComponent is a null pointer!");
    }
    HCCL_INFO("[CommunicatorImpl][%s] finished initializing collAlgComponent.", __func__);
}

HcclResult CommunicatorImpl::SetAccelerator(HcclAccelerator hcclAccelerator, bool isCcuMsAvailable)
{
    if (isLoadOp) {
        // 已下发过算子，不允许再设置accelerator
        HCCL_ERROR("[CommunicatorImpl]SetAccelerator is not allowed after load op.");
        return HCCL_E_NOT_SUPPORT;
    }
    AcceleratorState commAccelerator;
    if (hcclAccelerator == HcclAccelerator::DEFAULT) { // 用户没有配，读环境变量
        hcclAccelerator = EnvConfig::GetInstance().GetAlgoConfig().GetHcclAccelerator();
        HCCL_RUN_INFO(
            "[CommunicatorImpl][%s] env OpExpansionMode is [%s]", __func__, hcclAccelerator.Describe().c_str());
    }
    HcclMainboardId hcclMainboardId;
    CHK_RET(HrtGetMainboardId(devLogicId, hcclMainboardId));

    // 开启新流程时，仅mc2场景走回legacy通信域，此时不允许使用ms模式
    HCCL_WARNING("[CommunicatorImpl][%s] legacy communicator not support ccu ms mode for mc2.", __func__);
    isCcuMsAvailable = false;

    switch (hcclAccelerator) {
        case HcclAccelerator::CCU_MS:
            if (hcclMainboardId == HcclMainboardId::MAINBOARD_PCIE_STD) { // 标卡环境下配置CCU_MS加速模式拦截报错
                HCCL_ERROR(
                    "[SetAccelerator] hcclAccelerator[%s] not support in %s", hcclAccelerator.Describe().c_str(),
                    hcclMainboardId.Describe().c_str());
                return HCCL_E_NOT_SUPPORT;
            }
            commAccelerator = isCcuMsAvailable ? AcceleratorState::CCU_MS : AcceleratorState::CCU_SCHED;
            break;
        case HcclAccelerator::CCU_SCHED:
            commAccelerator = AcceleratorState::CCU_SCHED;
            if (IsCommWithPCIEProtocol()) {
                // 若当前通信域存在PCIE链路,不支持ccu展开,默认切换为aicpu展开,在大于8卡不支持aicpu场景由后续算法选择部分切换至aiv展开
                commAccelerator = AcceleratorState::AICPU_TS;
            }
            break;
        case HcclAccelerator::AIV:
            commAccelerator = AcceleratorState::AIV;
            break;
        case HcclAccelerator::AIV_ONLY:
            commAccelerator = AcceleratorState::AIV_ONLY;
            break;
        case HcclAccelerator::AICPU_TS:
            commAccelerator = AcceleratorState::AICPU_TS;
            break;
        case HcclAccelerator::HOSTCPU_TS: // 950不支持HOST展开，进行拦截
            HCCL_ERROR("[SetAccelerator] hcclAccelerator[%s] not support in 950", hcclAccelerator.Describe().c_str());
            return HCCL_E_NOT_SUPPORT;
        case HcclAccelerator::AICPU:
            HCCL_ERROR("[SetAccelerator] hcclAccelerator[%s] not support", hcclAccelerator.Describe().c_str());
            return HCCL_E_NOT_SUPPORT;
        default:
            HCCL_ERROR("[SetAccelerator] hcclAccelerator[%s] internal error", hcclAccelerator.Describe().c_str());
            return HCCL_E_INTERNAL;
    }
    OpExecuteConfig inCommExecuteConfig;
    inCommExecuteConfig.accState = commAccelerator;
    HCCL_DEBUG(
        "[CommunicatorImpl][%s] inCommExecuteConfig[%s]", __func__, inCommExecuteConfig.accState.Describe().c_str());
    TRY_CATCH_RETURN(SetCommExecuteConfig(inCommExecuteConfig)); // 设置通信域粒度加速模式，ccu模式需打开ccu驱动
    SetOpExecuteConfig(inCommExecuteConfig); // 算子粒度加速模式 同步为 通信域粒度加速模式
    HCCL_DEBUG(
        "[CommunicatorImpl][%s] comm accelerator [%s], isCcuMsAvailable is [%d]", __func__,
        GetCommExecuteConfig().accState.Describe().c_str(), isCcuMsAvailable);
    return HCCL_SUCCESS;
}

bool CommunicatorImpl::IsCommWithPCIEProtocol() const
{
    auto links = GetFullMeshLinks();
    for (auto link : links) {
        if (link.GetLinkProtocol() == LinkProtocol::PCIE) {
            HCCL_INFO("[CommunicatorImpl][%s]the current communicator has PCIE link", __func__);
            return true;
        }
    }
    HCCL_INFO("[CommunicatorImpl][%s]the current communicator does not have a PCIE link", __func__);
    return false;
}

HcclResult CommunicatorImpl::GetAccelerator(int32_t* accelerator) const
{
    HcclAccelerator hcclAccelerator{HcclAccelerator::DEFAULT};
    auto commAccelerator = GetCommExecuteConfig().accState;
    std::string acceleraToStr = AcceleratorStateToString.at(commAccelerator);
    HCCL_INFO(
        "[CommunicatorImpl][%s] commId[%s], commAccelerator[%s]", __func__, GetId().c_str(), acceleraToStr.c_str());

    switch (commAccelerator) {
        case AcceleratorState::CCU_MS:
            hcclAccelerator = HcclAccelerator::CCU_MS;
            break;
        case AcceleratorState::CCU_SCHED:
            hcclAccelerator = HcclAccelerator::CCU_SCHED;
            break;
        case AcceleratorState::AIV:
            hcclAccelerator = HcclAccelerator::AIV;
            break;
        case AcceleratorState::AIV_ONLY:
            hcclAccelerator = HcclAccelerator::AIV_ONLY;
            break;
        case AcceleratorState::AICPU_TS:
            hcclAccelerator = HcclAccelerator::AICPU_TS;
            break;
        case AcceleratorState::HOSTCPU_TS:
            hcclAccelerator = HcclAccelerator::HOSTCPU_TS;
            break;
        case AcceleratorState::AICPU:
            hcclAccelerator = HcclAccelerator::AICPU;
            break;
        default:
            HCCL_ERROR("[GetAccelerator] commAccelerator[%s] internal error", acceleraToStr.c_str());
            return HCCL_E_INTERNAL;
    }
    *accelerator = static_cast<int32_t>(hcclAccelerator);
    return HCCL_SUCCESS;
}

bool CommunicatorImpl::IsOpUsingCcuMs() const { return GetOpExecuteConfig().accState == AcceleratorState::CCU_MS; }

bool CommunicatorImpl::IsOpUsingCcuSched() const
{
    return GetOpExecuteConfig().accState == AcceleratorState::CCU_SCHED;
}

bool CommunicatorImpl::IsCommUsingCcuMs() const { return GetCommExecuteConfig().accState == AcceleratorState::CCU_MS; }

bool CommunicatorImpl::IsCommUsingCcuSched() const
{
    return GetCommExecuteConfig().accState == AcceleratorState::CCU_SCHED;
}

HcclResult CommunicatorImpl::RecoverExeCfgData(
    const OpExecuteConfig& inOpExeCfg, const OpExecuteConfig& inCommExeCfg, bool inIsLoadOp)
{
    // mc2目前没有快照恢复，如果增加需要调用该接口
    HCCL_INFO(
        "CommunicatorImpl[%s] Recover ExecuteConfig, opAcceState is %s, commAcceState is %s, isLoadOp is %d", __func__,
        inOpExeCfg.accState.Describe().c_str(), inCommExeCfg.accState.Describe().c_str(), inIsLoadOp);

    // 恢复加速器类型
    SetOpExecuteConfig(inOpExeCfg); // 算子粒度 和 通信域粒度 都考虑
    SetCommExecuteConfig(inCommExeCfg);
    isLoadOp = inIsLoadOp;

    HCCL_INFO("Recover OpExecuteConfig data from snapshot success.");
    return HcclResult::HCCL_SUCCESS;
}

void CommunicatorImpl::RegisterAcceStateCallBack(
    std::function<HcclResult(const std::string& commId, bool isUsingCcuMs, bool isUsingCcuSched)> inCallback)
{
    callback = inCallback;
}

void CommunicatorImpl::SetOpExecuteConfig(const OpExecuteConfig& inConfig)
{
    opExecuteConfig = inConfig;
    HCCL_DEBUG(
        "[CommunicatorImpl][%s] comm id [%s], IsOpUsingCcuMs [%d], IsOpUsingCcuSched [%d]", __func__, GetId().c_str(),
        IsOpUsingCcuMs(), IsOpUsingCcuSched()); // 算子粒度
}

void CommunicatorImpl::SetCommExecuteConfig(const OpExecuteConfig& inConfig)
{
    commExecuteConfig = inConfig;
    HCCL_DEBUG(
        "[CommunicatorImpl][%s] update comm manager ccu status, comm id [%s], IsCommUsingCcuMs [%d], "
        "IsCommUsingCcuSched [%d]",
        __func__, GetId().c_str(), IsCommUsingCcuMs(), IsCommUsingCcuSched()); // 通信域粒度

    TryInitCcuFeature(); // 单例结构整改前临时方案

    callback(GetId(), IsCommUsingCcuMs(), IsCommUsingCcuSched());
}

HcclResult CommunicatorImpl::CalcTaskNum(OpType opType, DataType dataType, u64 count, u32& taskNum) const
{
    HCCL_INFO("[CommunicatorImpl][CalcTaskNum] start!");
    return collAlgComponent->CalcTaskNum(opType, dataType, count, taskNum);
}

void CommunicatorImpl::InitUbMemoryTransportMgr()
{
    ubMemoryTransportMgr = std::make_unique<UbMemoryTransportMgr>(*this);
}

UbMemoryTransportMgr* CommunicatorImpl::GetUbMemoryTransportMgr() const { return ubMemoryTransportMgr.get(); }

HcclResult
CommunicatorImpl::HcomSelectAlg(const CollOpParams& opParams, int32_t aivCoreLimit, bool& ifAiv, std::string& algName)
{
    HCCL_INFO(
        "CommunicatorImpl::HcomSelectAlg opType[%s], count[%llu], dataType[%s], HcclReduceOp[%s], aivCoreLimit[%d]",
        opParams.opType.Describe().c_str(), opParams.count, opParams.dataType.Describe().c_str(),
        opParams.reduceOp.Describe().c_str(), aivCoreLimit);

    if (GetCommStatus() == CommStatus::COMM_ERROR) {
        HCCL_ERROR("Comm has been error, can not select alg now!");
        return HcclResult::HCCL_E_INTERNAL;
    }

    if (isSuspended) {
        HCCL_ERROR("Comm has been suspended, can not select alg now!");
        return HcclResult::HCCL_E_SUSPENDING;
    }
    // 等待通信域状态为Ready，执行算子下发
    WaitReady();

    std::string tag = ""; // 算法选择不需要传入tag，获取kernel arg的时候会用到
    CovertToCurrentCollOperator(tag, opParams, OpMode::OFFLOAD, true, true);
    // 图模式算子加载选择CollService
    opExecuteConfig = commExecuteConfig;
    ExecAlgSelect(opParams, OpMode::OFFLOAD);
    ifAiv
        = (opExecuteConfig.accState == AcceleratorState::AIV || opExecuteConfig.accState == AcceleratorState::AIV_ONLY);
    HcclResult dataTypeChkRes
        = OpParamsChecker::CheckOpDataTypeOffload(opParams, GetOpCcuFeatureFlag(), GetOpAiCpuTSFeatureFlag(), ifAiv);
    if (dataTypeChkRes != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[CommunicatorImpl::HcomSelectAlg] DataType check fail.");
        SetCommStatus(CommStatus::COMM_READY);
        return dataTypeChkRes;
    }
    algName = curAlgName;

    return HcclResult::HCCL_SUCCESS;
}

void CommunicatorImpl::ReportHcclMC2Info(
    const Stream& kfcStream, Stream& stream, const std::vector<Stream*>& aicpuStreams)
{
    HcclResult hcclRet = InitProfilingReporter();
    if (hcclRet != HCCL_SUCCESS) {
        HCCL_ERROR("[ReportHcclMC2Info]InitProfilingReporter failed, ret[%d]", static_cast<s32>(hcclRet));
        return;
    }
    profilingReporter->CallReportMc2CommInfo(kfcStream, stream, aicpuStreams, id, myRank, rankSize, rankInParentComm);
}

void CommunicatorImpl::OpAcceleratorStateFallback()
{
    OpExecuteConfig inOpExecuteConfig;
    // 只要ccu出问题，直接回退到CCU_FALLBACK，走AICPU
    switch (opExecuteConfig.accState) {
        case AcceleratorState::CCU_MS:
            inOpExecuteConfig.accState = AcceleratorState::CCU_FALLBACK;
            break;
        case AcceleratorState::CCU_SCHED:
            inOpExecuteConfig.accState = AcceleratorState::CCU_FALLBACK;
            break;
        default:
            THROW<NotSupportException>(
                StringFormat("[CommunicatorImpl::%s] Only supports CCU accelerator rollback", __func__));
            break;
    }
    SetOpExecuteConfig(inOpExecuteConfig);
}

HcclResult CommunicatorImpl::AcceleratorFallback()
{
    HCCL_RUN_INFO("[CommunicatorImpl][%s] opMode[%s]", __func__, currentCollOperator->opMode.Describe().c_str());
    string needFallBackAlgName = curAlgName;
    OpAcceleratorStateFallback();

    HcclResult ret = HCCL_SUCCESS;
    switch (currentCollOperator->opMode) {
        case OpMode::OPBASE:
            ret = ReLoadOpbasedOp();
            break;
        case OpMode::OFFLOAD:
            ret = ReLoadOffloadOp();
            break;
        default:
            THROW<InternalException>(
                StringFormat("[CommunicatorImpl::%s] OpMode error, accelerator rollback failed", __func__));
            break;
    }

    // 缓存当前算子的加速模式；
    // 下一个算子下发时，做完算法选择后，查找上述加速模式缓存，
    // 若能命中，按照上述已缓存的加速模式下发算子(大概率也是资源不足，走回退)；
    // 否则，按照算法选择的加速模式下发算子。
    opAcceStateCache.insert({{curOpParams.opType, needFallBackAlgName}, {opExecuteConfig.accState, curAlgName}});
    HCCL_INFO(
        "[CommunicatorImpl][%s] opAcceStateCache opType[%s], needFallBackAlgName[%s], accelerator[%s], curAlgName[%s]",
        __func__, curOpParams.opType.Describe().c_str(), needFallBackAlgName.c_str(),
        opExecuteConfig.accState.Describe().c_str(), curAlgName.c_str());

    HCCL_INFO("[CommunicatorImpl][%s] end", __func__);
    return ret;
}

HcclResult CommunicatorImpl::GetCacheMap(AivOpCacheArgs& opCacheParam, std::shared_ptr<InsQueue>& tempInsQue)
{
    if (hcclCacheMap_.size() > CACHEMAP_MAXSIZE) {
        size_t clearCount = static_cast<size_t>(CACHEMAP_MAXSIZE * CACHEMAP_CLEARPERCENT);
        for (auto it = hcclCacheMap_.begin(); clearCount > 0 && it != hcclCacheMap_.end(); clearCount--) {
            it = hcclCacheMap_.erase(it);
        }
    }
    hcclCacheMap_.emplace(std::make_pair(opCacheParam, std::move(tempInsQue)));
    HCCL_INFO("[CommunicatorImpl][GetCacheMap]");
    return HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::ReLoadOpbasedOp()
{
    HCCL_DEBUG(
        "[CommunicatorImpl][%s] status is [%s], isSuspended is [%d]", __func__, GetCommStatus().Describe().c_str(),
        isSuspended);
    ExecAlgSelect(curOpParams, OpMode::OPBASE); // 根据配置选择对应的collService
    if (dynamic_cast<CollServiceDefaultImpl*>(collService) != nullptr) {
        HCCL_ERROR("ReLoadOpbasedOp is not supported in CollServiceDefaultImpl.");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    bool isAiv
        = (opExecuteConfig.accState == AcceleratorState::AIV || opExecuteConfig.accState == AcceleratorState::AIV_ONLY);
    HcclResult dataTypeChkRes = OpParamsChecker::CheckOpDataTypeOpbase(
        curOpParams, GetOpCcuFeatureFlag(), GetOpAiCpuTSFeatureFlag(), isAiv); // 算子粒度
    if (dataTypeChkRes != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[CommunicatorImpl::ReLoadOpbasedOp] DataType check fail.");
        SetCommStatus(CommStatus::COMM_READY);
        return dataTypeChkRes;
    }

    if (currentCollOperator == nullptr) {
        HCCL_ERROR("CurrentCollOperator not initialized.");
        return HcclResult::HCCL_E_PTR;
    }
    collService->ReLoadWithOpBasedMode(*currentCollOperator);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::ReLoadOffloadOp()
{
    HCCL_DEBUG(
        "[CommunicatorImpl][%s] status is [%s], isSuspended is [%d]", __func__, GetCommStatus().Describe().c_str(),
        isSuspended);

    ExecAlgSelect(curOpParams, OpMode::OFFLOAD); // 根据配置选择对应的collService

    if (opExecuteConfig.accState == AcceleratorState::HOSTCPU_TS) { // 950不支持HOST_TS模式
        HCCL_ERROR("[CommunicatorImpl::ReLoadOffloadOp] HOSTCPU_TS is not support.");
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }
    bool isAiv
        = (opExecuteConfig.accState == AcceleratorState::AIV || opExecuteConfig.accState == AcceleratorState::AIV_ONLY);
    HcclResult dataTypeChkRes = OpParamsChecker::CheckOpDataTypeOffload(
        curOpParams, GetOpCcuFeatureFlag(), GetOpAiCpuTSFeatureFlag(), isAiv); // 算子粒度
    if (dataTypeChkRes != HcclResult::HCCL_SUCCESS) {
        HCCL_ERROR("[CommunicatorImpl::ReLoadOffloadCollOp] DataType check fail.");
        SetCommStatus(CommStatus::COMM_READY);
        return dataTypeChkRes;
    }

    if (currentCollOperator == nullptr) {
        HCCL_ERROR("CurrentCollOperator not initialized.");
        return HcclResult::HCCL_E_PTR;
    }
    collService->ReLoadWithOffloadMode(*currentCollOperator);
    return HcclResult::HCCL_SUCCESS;
}

template <typename BufferType>
std::shared_ptr<BufferType> CommunicatorImpl::BarrierAllocBuffer(std::size_t size)
{
    return std::make_shared<BufferType>(size);
}

HcclResult CommunicatorImpl::CreateBarrierMemory(void*& sendBuf, void*& recvBuf, uint64_t count)
{
    HCCL_INFO("CreateBarrierMemory start.");
    if (isFirstBarrier) {
        barrierInMemory = BarrierAllocBuffer<DevBuffer>(count * sizeof(float));
        barrierOutMemory = BarrierAllocBuffer<DevBuffer>(count * sizeof(float));
        // 申请host侧内存，并将初始值设置为0
        std::shared_ptr<HostBuffer> barrierHostMem = BarrierAllocBuffer<HostBuffer>(count * sizeof(float));
        s32 sRet = memset_s(
            reinterpret_cast<void*>(barrierHostMem->GetAddr()), barrierHostMem->GetSize(), 0, count * sizeof(float));
        if (sRet != EOK) {
            barrierInMemory.reset();
            barrierOutMemory.reset();
            barrierHostMem.reset();
            HCCL_ERROR("[CreateBarrierMemory] mem set failed.errorno[%d]", sRet);
            return HCCL_E_MEMORY;
        }
        // H2D拷贝
        HrtMemcpy(
            reinterpret_cast<void*>(barrierInMemory->GetAddr()), barrierInMemory->GetSize(),
            reinterpret_cast<void*>(barrierHostMem->GetAddr()), barrierHostMem->GetSize(), RT_MEMCPY_HOST_TO_DEVICE);
        HrtMemcpy(
            reinterpret_cast<void*>(barrierOutMemory->GetAddr()), barrierOutMemory->GetSize(),
            reinterpret_cast<void*>(barrierHostMem->GetAddr()), barrierHostMem->GetSize(), RT_MEMCPY_HOST_TO_DEVICE);
        isFirstBarrier = false;
    }
    // 将内存指针赋值给传入参数
    sendBuf = reinterpret_cast<void*>(barrierInMemory->GetAddr());
    if (sendBuf == nullptr) {
        HCCL_ERROR("[CreateBarrierMemory] Failed to get barrierInMemory.");
        return HCCL_E_PTR;
    }
    recvBuf = reinterpret_cast<void*>(barrierOutMemory->GetAddr());
    if (recvBuf == nullptr) {
        HCCL_ERROR("[CreateBarrierMemory] Failed to get barrierOutMemory.");
        return HCCL_E_PTR;
    }
    HCCL_INFO("CreateBarrierMemory success.");
    return HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::CreateWorkspaceBuf(const char* memTag, uint64_t* size, bool* newCreated)
{
    std::string tag = memTag != nullptr ? std::string(memTag) : "";
    // empty tag is global workspace
    if (tagWorkspaceMap_.find(tag) == tagWorkspaceMap_.end()) {
        shared_ptr<DevBuffer> workspace = std::make_shared<DevBuffer>(*size);
        tagWorkspaceMap_.insert(make_pair(tag, workspace));
        HCCL_INFO("Create tagMem[%s] WorkspaceBuf success, WorkspaceBuf = %p", tag.c_str(), workspace.get());
        if (newCreated != nullptr) {
            *newCreated = true;
        }
    }
    return HcclResult::HCCL_SUCCESS;
}

// dpu相关
bool CommunicatorImpl::IsNeedDpu()
{
    if (rankGraph == nullptr) {
        return false;
    }
    if (rankGraph->GetPeer(myRank) == nullptr) {
        HCCL_ERROR("[GetHostIpFromRankGraph] rankGraph peer is null!");
        return false;
    }
    // 根据rankgraph直接找peer对应的ConnInterface列表
    std::vector<std::shared_ptr<NetInstance::ConnInterface>> interfaces = rankGraph->GetPeer(myRank)->GetIfaces();
    for (auto interface : interfaces) {
        if (interface->GetPos() == AddrPosition::HOST) {
            HCCL_INFO("[CommunicatorImpl][IsNeedDpu] need host dpu");
            return true;
        }
    }
    return false;
}

void CommunicatorImpl::InitHccpPeer() const
{
    RaSocketSetWhiteListStatus(1); // PEER模式需要手动开启白名单模式
    HccpPeerManager::GetInstance().Init(devLogicId);
}

HcclResult CommunicatorImpl::PrepareDpuKernelResource(aclrtFuncHandle& funcHandle)
{
    // 获取二进制文件路径
    std::string jsonPath;
    std::string getPath = getenv("ASCEND_HOME_PATH");
    if (!getPath.empty()) {
        jsonPath = getPath;
    } else {
        jsonPath = "/usr/local/Ascend/cann/";
        HCCL_WARNING("[CommunicatorImpl::%s] ENV:ASCEND_HOME_PATH is not set", __func__);
    }

    jsonPath += "/opp/built-in/op_impl/dpu/";
    HCCL_DEBUG("[CommunicatorImpl::%s] kernel folder path[%s]", __func__, jsonPath.c_str());

    // cpuKernelMode为1时，json命名需与so命名保持一致， 即libccl_dpu.json与libccl_dpu.so
    jsonPath += "libccl_dpu.json";
    char realPath[PATH_MAX] = {0};
    CHK_PRT_RET(
        realpath(jsonPath.c_str(), realPath) == nullptr,
        HCCL_ERROR("[CommunicatorImpl::%s]: %s is not a valid real path, err[%d]", __func__, jsonPath.c_str(), errno),
        HCCL_E_INTERNAL);
    HCCL_INFO("[CommunicatorImpl::%s] realPath: %s", __func__, realPath);

    aclrtBinHandle binHandle;
    aclrtBinaryLoadOptions options;
    aclrtBinaryLoadOption option;
    option.type = ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE; // AI CPU算子注册模式 ????
    option.value.cpuKernelMode = 1; // 0 ：仅需要加载json，1 ：加载cpu so & json，2: LoadFromData
    options.numOpt = 1;
    options.options = &option;
    if (aclrtBinaryLoadFromFile(realPath, &options, &binHandle) != ACL_SUCCESS) {
        HCCL_ERROR("[CommunicatorImpl::%s] load binary from file error.", __func__);
        return HCCL_E_OPEN_FILE_FAILURE;
    }

    // 创建dpustream
    aclError ret = aclrtCreateStreamWithConfig(&dpuStream, 0, ACL_STREAM_FAST_LAUNCH);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR("[CommunicatorImpl::%s] Create Local Stream Failed", __func__);
        return HCCL_E_INTERNAL;
    }
    PLF_CONFIG_INFO(
        PLF_RES, "Create Stream para: deviceId[%d] streamId[%d]", HrtGetDevice(), HrtGetStreamId(dpuStream));

    // 查找核函数
    if (aclrtBinaryGetFunction(binHandle, "RunDpuRpcSrvLaunch", &funcHandle) != ACL_SUCCESS) {
        HCCL_ERROR("[CommunicatorImpl::%s] Get Function Failed", __func__);
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::LaunchDpuKernel(aclrtFuncHandle& funcHandle)
{
    // 下发
    HCCL_INFO("[CommunicatorImpl::%s] Launch Dpu Kernel", __func__);
    aclrtLaunchKernelCfg cfg;
    aclrtLaunchKernelAttr kernelAttr;
    kernelAttr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    kernelAttr.value.timeout = NOTIFY_DEFAULT_WAIT_TIME > std::numeric_limits<uint16_t>::max() ?
                                   std::numeric_limits<uint16_t>::max() :
                                   NOTIFY_DEFAULT_WAIT_TIME;
    cfg.numAttrs = 1;
    cfg.attrs = &kernelAttr;
    constexpr u32 numBlocks = 1;
    hostArgs.commId = id;
    hostArgs.memorySize = SHARE_HBM_MEMORY_SIZE;
    hostArgs.hostMem = hostShareBuf;
    hostArgs.shareHBM = tagDpuShmemArgsMap_[DPUTAG].connectType_ == HOST_DEVICE_CONNECT_TYPE_UB ?
                            tagDpuShmemArgsMap_[DPUTAG].va_ :
                            tagDpuShmemArgsMap_[DPUTAG].accessVA_;
    hostArgs.deviceId = devLogicId;
    hostArgs.taskexceptionVa = tagDpuShmemArgsMap_[DPUTASKEXCEPTION].connectType_ == HOST_DEVICE_CONNECT_TYPE_UB ?
                                   tagDpuShmemArgsMap_[DPUTASKEXCEPTION].va_ :
                                   tagDpuShmemArgsMap_[DPUTASKEXCEPTION].accessVA_;
    HCCL_INFO(
        "[CommunicatorImpl::%s] DpuKernelLaunchParam{commId:%s; memorySize:%u; shareHBM:%p; taskexceptionVa:%p}",
        __func__, hostArgs.commId.c_str(), hostArgs.memorySize, hostArgs.shareHBM, hostArgs.taskexceptionVa);
    CHK_RET(SaveDpuStreamId());
    size_t argsSize = sizeof(hostArgs);
    aclrtPlaceHolderInfo placeHolderArrays;
    size_t placeHolderNum = 0;
    if (aclrtLaunchKernelWithHostArgs(
            funcHandle, numBlocks, dpuStream, &cfg, &hostArgs, argsSize, &placeHolderArrays, placeHolderNum)
        != ACL_SUCCESS) {
        HCCL_ERROR("[CommunicatorImpl::%s] Launch Dpu Kernel Failed", __func__);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::InitAndLaunchDpuKernel()
{
    HCCL_INFO("[CommunicatorImpl::%s] Start to Launch Dpu Kernel", __func__);
    // 申请共享内存(需要在npu ctx 下进行)
    for (auto& tmpShmem : tagDpuShmemArgsMap_) {
        bool newCreate = false;
        uint64_t memSize = tmpShmem.first == DPUTAG ? static_cast<uint64_t>(SHARE_HBM_MEMORY_SIZE) :
                                                      static_cast<uint64_t>(DPU_TASKEXCEPTION_MEMORY_SIZE);
        HcclResult memRet = GetKFCWorkSpaceVA(tmpShmem.first, &memSize, &tmpShmem.second.accessVA_, &newCreate);
        if (memRet != HCCL_SUCCESS) {
            HCCL_ERROR("[CommunicatorImpl::InitAndLaunchDpuKernel] Alloc Share HBM Failed");
            return memRet;
        }
    }
    // 设置XPU
    HCCL_INFO("[CommunicatorImpl::%s] Switch to Dpu Ctx", __func__);
    if (aclrtGetCurrentContext(&npuContext) != ACL_SUCCESS) {
        HCCL_ERROR("[CommunicatorImpl::%s] Get Npu Ctx Failed", __func__);
        return HCCL_E_INTERNAL;
    }
    if (HrtSetXpuDevice(TEMP_DEV_TYPE_DPU, 0) != HCCL_SUCCESS) {
        HCCL_ERROR("[CommunicatorImpl::%s] Switch to Dpu Ctx Failed", __func__);
        return HCCL_E_INTERNAL;
    }
    if (aclrtGetCurrentContext(&dpuContext) != ACL_SUCCESS) {
        HCCL_ERROR("[CommunicatorImpl::%s] Get Dpu Ctx Failed", __func__);
        return HCCL_E_INTERNAL;
    }

    // 准备资源
    aclrtFuncHandle funcHandle;
    CHK_RET(PrepareDpuKernelResource(funcHandle));

    hostShareBuf = malloc(SHARE_HBM_MEMORY_SIZE);
    CHK_PTR_NULL(hostShareBuf);

    // 下发
    HcclResult ret = LaunchDpuKernel(funcHandle);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[CommunicatorImpl::%s] Launch Dpu Kernel Failed", __func__);
        free(hostShareBuf);
        hostShareBuf = nullptr;
        return ret;
    }

    // 切换回当前Ctx
    HCCL_INFO("[CommunicatorImpl::%s] Switch to Npu Ctx", __func__);
    if (ACL_SUCCESS != aclrtSetCurrentContext(npuContext)) {
        HCCL_ERROR("[CommunicatorImpl::%s] Reset Current Ctx Failed", __func__);
        free(hostShareBuf);
        hostShareBuf = nullptr;
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("[CommunicatorImpl::%s] Launch Dpu Kernel End", __func__);
    isDpuKernelLaunched = true;
    g_commNum++;
    return HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::InitAndLaunchAicpuKernel()
{
    HCCL_INFO("[CommunicatorImpl::%s] Start to Launch Aicpu Kernel", __func__);
    // context切换

    // 准备资源
    std::string kernelName = "HcclDpuTaskexpShmemRestore";
    aclrtFuncHandle funcHandle = GetAicpuKernelFuncHandle(kernelName.c_str());
    constexpr u32 numBlocks = 1;
    aclrtStream tempAicpuStream;                                                                   // 创建局部流
    aclError createRet = aclrtCreateStreamWithConfig(&tempAicpuStream, 0, ACL_STREAM_FAST_LAUNCH); // 后两个入参？
    if (createRet != ACL_SUCCESS) {
        HCCL_ERROR("[CommunicatorImpl::%s] Create Local Stream Failed", __func__);
        return HCCL_E_INTERNAL;
    }
    PLF_CONFIG_INFO(
        PLF_RES, "Create Stream para: deviceId[%d] streamId[%d]", HrtGetDevice(), HrtGetStreamId(tempAicpuStream));
    aclrtLaunchKernelCfg cfg;
    aclrtLaunchKernelAttr kernelAttr;
    kernelAttr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    kernelAttr.value.timeout = NOTIFY_DEFAULT_WAIT_TIME > std::numeric_limits<uint16_t>::max() ?
                                   std::numeric_limits<uint16_t>::max() :
                                   NOTIFY_DEFAULT_WAIT_TIME;
    cfg.numAttrs = 1;
    cfg.attrs = &kernelAttr;
    struct AicpuKernelLaunchParam {
        char commId[COMM_NAME_MAX_LENGTH];
        void* taskexceptionVa;
        u64 memorySize;
        uint32_t deviceId;
    };
    AicpuKernelLaunchParam hostArgsTmp;
    auto sRet = strncpy_s(hostArgsTmp.commId, COMM_NAME_MAX_LENGTH, id.c_str(), COMM_NAME_MAX_LENGTH - 1);
    if (sRet != 0) {
        HCCL_ERROR("[CommunicatorImpl::InitAndLaunchAicpuKernel] commId strncpy_s failed, ret[%d]", sRet);
        return HCCL_E_INTERNAL;
    }
    hostArgsTmp.taskexceptionVa = tagDpuShmemArgsMap_[DPUTASKEXCEPTION].accessVA_;
    hostArgsTmp.memorySize = DPU_TASKEXCEPTION_MEMORY_SIZE;
    hostArgsTmp.deviceId = devLogicId;
    size_t argsSize = sizeof(hostArgsTmp);
    aclrtPlaceHolderInfo placeHolderArrays;
    size_t placeHolderNum = 0;
    // 下发
    rtError_t ret = aclrtLaunchKernelWithHostArgs(
        funcHandle, numBlocks, tempAicpuStream, &cfg, &hostArgsTmp, argsSize, &placeHolderArrays, placeHolderNum);
    if (ret != RT_ERROR_NONE) {
        HCCL_ERROR("[CommunicatorImpl::%s] Launch Aicpu Kernel Failed", __func__);
        return HCCL_E_INTERNAL;
    }
    HcclStreamSynchronize(tempAicpuStream);                     // 同步
    aclError aclRet = aclrtDestroyStreamForce(tempAicpuStream); // 销毁局部流
    if (ACL_SUCCESS != aclRet) {
        HCCL_ERROR("Destroy Stream Failed, aclReturn[%d]", aclRet);
        return HCCL_E_RUNTIME;
    }
    HCCL_INFO("[CommunicatorImpl::%s] Launch Aicpu Kernel End", __func__);
    return HCCL_SUCCESS;
}

void CommunicatorImpl::AppendLocalDieIdForLinks()
{
    if (rankSize == 1) {
        HCCL_INFO("[AppendLocalDieIdForLinks] rankSize = 1, No RankGraph exists");
        return;
    }

    auto srcRankNode = rankGraph->GetPeer(myRank)->GetNodeId();

    auto processLinks = [&](const std::vector<std::shared_ptr<NetInstance::Link>>& links, bool isSource) {
        for (auto link : links) {
            auto iface = isSource ? link->GetSourceIface() : link->GetTargetIface();
            if (iface->GetPos() == AddrPosition::HOST || *(iface->GetLinkProtocols().begin()) == LinkProtocol::PCIE) {
                continue;
            }
            u32 dieId = GetLocalDieId({myRank, *iface}, *(link->GetLinkProtocols().begin()));
            HCCL_INFO("[CommunicatorImpl][AppendLocalDieIdForLinks] get link dieid[%u]", dieId);
            iface->SetLocalDieId(dieId);
        }
    };

    for (auto level : rankGraph->GetLevels(myRank)) {
        auto netInstance = rankGraph->GetNetInstanceByRankId(level, myRank);
        auto& vGraph = netInstance->GetGraph();

        // Process fabric links
        for (auto fabric : netInstance->GetFabrics()) {
            auto dstRankNode = fabric->GetNodeId();
            processLinks(vGraph.GetEdges(srcRankNode, dstRankNode), true);
            processLinks(vGraph.GetEdges(dstRankNode, srcRankNode), false);
        }

        // Process direct peer links
        for (u32 dstRank = 0; dstRank < rankSize; ++dstRank) {
            auto dstRankNode = rankGraph->GetPeer(dstRank)->GetNodeId();
            processLinks(vGraph.GetEdges(srcRankNode, dstRankNode), true);
            processLinks(vGraph.GetEdges(dstRankNode, srcRankNode), false);
        }
    }
}

HcclResult CommunicatorImpl::GetLocalCclBuffer(void** addr, uint64_t* size)
{
    CHK_PTR_NULL(inCclBuffer.get());
    *addr = reinterpret_cast<void*>(inCclBuffer.get()->GetAddr());
    *size = static_cast<uint64_t>(inCclBuffer.get()->GetSize());
    HCCL_INFO("CommunicatorImpl::GetLocalCclBuffer success, addr[%p], size[%llu]", *addr, *size);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
CommunicatorImpl::GetDevMemWorkSpace(const std::string& memTag, uint64_t* size, void** addr, bool* newCreated)
{
    if (memTag == DPUTAG) {
        return GetKFCWorkSpaceVA(memTag, size, addr, newCreated);
    }
    auto iter = tagWorkspaceMap_.find(memTag);
    if (iter != tagWorkspaceMap_.end()) {
        std::shared_ptr<DevBuffer> oldWorkspace = iter->second;
        if (*size != static_cast<uint64_t>(oldWorkspace.get()->GetSize())) {
            HCCL_ERROR(
                "HcclCommunicator::GetDevMemWorkSpace, The size of oldWorkspace %p is non-consistent, target size "
                "compare now size: %llu->%llu",
                *addr, *size, oldWorkspace.get()->GetSize());
            return HCCL_E_PARA;
        }
        *addr = reinterpret_cast<void*>(oldWorkspace.get()->GetAddr());
        if (newCreated != nullptr) {
            *newCreated = false;
        }
        return HcclResult::HCCL_SUCCESS;
    }

    shared_ptr<DevBuffer> newWorkspace = std::make_shared<DevBuffer>(*size);
    tagWorkspaceMap_.insert(make_pair(memTag, newWorkspace));
    HCCL_INFO(
        "Create tagMem[%s] WorkspaceBuf success, WorkspaceBuf: %p -> %p, size[%llu]", memTag.c_str(),
        newWorkspace.get(), newWorkspace.get()->GetAddr(), *size);
    if (newCreated != nullptr) {
        *newCreated = true;
    }
    *addr = reinterpret_cast<void*>(newWorkspace.get()->GetAddr());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult
CommunicatorImpl::AllocAndRegKFCWorkSpace(uint64_t size, const std::string& memTag) // 完成共享内存的申请和注册
{
    auto it = tagDpuShmemArgsMap_.find(memTag);
    if (it == tagDpuShmemArgsMap_.end()) {
        HCCL_ERROR("memTag is invalid, memTag: %s", memTag.c_str());
        return HCCL_E_PARA;
    }

    if (it->second.accessVA_ != nullptr && it->second.va_ != nullptr) {
        HCCL_WARNING(
            "[CommunicatorImpl::%s] memTag[%s] accessVA_ is not nullptr, maybe already register, accessVA_: %p",
            __func__, memTag.c_str(), it->second.accessVA_);
        CHK_RET(DestroyKFCWorkSpaceVA());
    }
    int32_t deviceLogicId = 0;
    aclError aclRet = aclrtGetLogicDevIdByUserDevId(devLogicId, &deviceLogicId); // userDevId 转 logicDevId
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR(
            "[CommunicatorImpl::%s] aclrtGetLogicDevIdByUserDevId failed, devLogicId: %u, ret: %d", __func__,
            devLogicId, aclRet);
        return HCCL_E_RUNTIME;
    }
    CHK_RET(HrtHalGetDeviceInfo(deviceLogicId, MODULE_TYPE_SYSTEM, INFO_TYPE_HD_CONNECT_TYPE, it->second.connectType_));
    HCCL_INFO("[CommunicatorImpl::%s] connectType_: %ld", __func__, it->second.connectType_);

    HcclResult ret = HCCL_SUCCESS;
    if (it->second.connectType_ == HOST_DEVICE_CONNECT_TYPE_PCIE) {
        it->second.va_ = HrtMalloc(size, ACL_MEM_TYPE_HIGH_BAND_WIDTH);
        ret = HrtHalHostRegister(it->second.va_, size, DEV_SVM_MAP_HOST, deviceLogicId, &it->second.accessVA_);
    } else if (it->second.connectType_ == HOST_DEVICE_CONNECT_TYPE_UB) {
        it->second.originVa_ = malloc(size + ALIGN_4K); // 保证足够的对齐余量（最多偏移 ALIGN_4K-1）
        CHK_PTR_NULL(it->second.originVa_);
        void* ptr = it->second.originVa_;
        std::size_t storeSize = 0; // 不预留对象空间，仅做指针向上对齐
        std::size_t space = static_cast<std::size_t>(size + ALIGN_4K);
        // 如果对齐失败，va_为nullptr，在HrtHalHostRegister中校验返回HCCL_E_PTR
        it->second.va_ = std::align(static_cast<std::size_t>(ALIGN_4K), storeSize, ptr, space);
        ret = HrtHalHostRegister(it->second.va_, size, HOST_MEM_MAP_DEV_PCIE_TH, deviceLogicId, &it->second.accessVA_);
    } else {
        return HCCL_E_NOT_SUPPORT;
    }
    if (ret != HCCL_SUCCESS) {
        return CleanupKFCWorkSpaceOnFailure(it->second, ret);
    }
    auto ptr = it->second.connectType_ == HOST_DEVICE_CONNECT_TYPE_UB ? it->second.va_ : it->second.accessVA_;
    errno_t cpyRet = memset_s(ptr, sizeof(uint8_t), 0, sizeof(uint8_t)); // 首字节作为信号标志位
    if (cpyRet != EOK) {
        HCCL_ERROR("[CommunicatorImpl::AllocAndRegKFCWorkSpace] set 0 failed: %d", cpyRet);
        return HCCL_E_MEMORY; // 如果失败，会在~CommunicatorImpl中进行解注册与内存释放
    }
    HCCL_INFO(
        "CommunicatorImpl::AllocAndRegKFCWorkSpace, va_[%p], accessVA_[%p]", it->second.va_, it->second.accessVA_);
    return HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::CleanupKFCWorkSpaceOnFailure(DpuShmem& shmem, HcclResult ret)
{
    HCCL_ERROR(
        "[CommunicatorImpl::%s] HrtHalHostRegister failed, ret: %d, connect type: %ld", __func__, ret,
        shmem.connectType_);
    if (shmem.va_ != nullptr) {
        if (shmem.connectType_ == HOST_DEVICE_CONNECT_TYPE_PCIE) {
            HrtFree(shmem.va_);
        } else if (shmem.connectType_ == HOST_DEVICE_CONNECT_TYPE_UB) {
            free(shmem.originVa_);
            shmem.originVa_ = nullptr;
        }
        shmem.va_ = nullptr;
    }
    shmem.accessVA_ = nullptr;
    return ret;
}

HcclResult
CommunicatorImpl::GetKFCWorkSpaceVA(const std::string& memTag, const uint64_t* size, void** addr, bool* newCreated)
{
    auto it = tagDpuShmemArgsMap_.find(memTag);
    if (it == tagDpuShmemArgsMap_.end()) {
        HCCL_ERROR("memTag is invalid, memTag: %s", memTag.c_str());
        return HCCL_E_PARA;
    }
    auto iter = tagWorkspaceVAMap_.find(memTag);
    if (iter != tagWorkspaceVAMap_.end()) {
        std::shared_ptr<DevBuffer> oldWorkspace = iter->second;
        if (*size != static_cast<uint64_t>(oldWorkspace.get()->GetSize())) {
            HCCL_ERROR(
                "HcclCommunicator::GetKFCWorkSpaceVA, The size of oldWorkspace %p is non-consistent, target size "
                "compare now size: %llu->%llu",
                *addr, *size, oldWorkspace.get()->GetSize());
            return HCCL_E_PARA;
        }
        *addr = reinterpret_cast<void*>(oldWorkspace.get()->GetAddr());
        if (newCreated != nullptr) {
            *newCreated = false;
        }
        return HcclResult::HCCL_SUCCESS;
    }

    CHK_RET(AllocAndRegKFCWorkSpace(*size, memTag));
    shared_ptr<DevBuffer> newWorkspace = DevBuffer::Create(reinterpret_cast<uintptr_t>(it->second.accessVA_), *size);
    tagWorkspaceVAMap_.insert(make_pair(memTag, newWorkspace));
    if (newCreated != nullptr) {
        *newCreated = true;
    }
    *addr = reinterpret_cast<void*>(newWorkspace.get()->GetAddr());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::DestroyKFCWorkSpaceVA()
{
    for (auto& tmpShmem : tagDpuShmemArgsMap_) {
        if (tmpShmem.second.accessVA_ == nullptr && tmpShmem.second.va_ == nullptr) {
            continue;
        }
        int32_t deviceLogicId = 0;
        aclError aclRet = aclrtGetLogicDevIdByUserDevId(devLogicId, &deviceLogicId); // userDevId 转 logicDevId
        if (aclRet != ACL_SUCCESS) {
            HCCL_ERROR(
                "[CommunicatorImpl::%s] aclrtGetLogicDevIdByUserDevId failed, devLogicId: %u, ret: %d", __func__,
                devLogicId, aclRet);
            return HCCL_E_RUNTIME;
        }

        // 必须先halHostUnregister解除映射，再释放设备内存，否则HrtFree会因内存被pin住而异常
        if (tmpShmem.second.va_ != nullptr) {
            HcclResult ret = HrtHalHostUnregister(tmpShmem.second.va_, deviceLogicId);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[CommunicatorImpl::%s] HrtHalHostUnregister failed, ret[%d]", __func__, ret);
            }

            if (tmpShmem.second.connectType_ == HOST_DEVICE_CONNECT_TYPE_PCIE) {
                DECTOR_TRY_CATCH("CommunicatorImpl", HrtFree(tmpShmem.second.va_));
            } else if (tmpShmem.second.connectType_ == HOST_DEVICE_CONNECT_TYPE_UB) {
                DECTOR_TRY_CATCH("CommunicatorImpl", free(tmpShmem.second.originVa_));
                tmpShmem.second.originVa_ = nullptr;
            }
        }

        tmpShmem.second.va_ = nullptr;
        tmpShmem.second.accessVA_ = nullptr;
        tagWorkspaceVAMap_.erase(tmpShmem.first);
    }
    return HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::GetAicpuOpStreamNotify(rtStream_t* opStream, u8 aicpuNotifyNum, void** aicpuNotify) const
{
    GetAicpuStreamManager().AllocFreeStream();
    Stream* stream = GetAicpuStreamManager().GetFreeStream();
    *opStream = stream->GetPtr();
    GetHostDeviceSyncNotifyManager().GetMc2AiCpuNotifys(aicpuNotifyNum, aicpuNotify);
    return HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::GetNetLayers(uint32_t** netLayers, uint32_t* netLayerNum)
{
    try {
        CHK_PTR_NULL(rankGraph);
        u32 rankId = rankGraph->GetMyRank();
        std::set<u32> levels = rankGraph->GetLevels(rankId);
        u32 num = rankGraph->GetLevelNum();
        netLayersVec.clear();
        netLayersVec = std::vector<u32>(levels.begin(), levels.end());
        *netLayers = netLayersVec.data();
        *netLayerNum = num;
        return HCCL_SUCCESS;
    } catch (const InvalidParamsException& e) {
        HCCL_ERROR(e.what());
        return HCCL_E_PARA;
    } catch (const NullPtrException& e) {
        HCCL_ERROR(e.what());
        return HCCL_E_PTR;
    } catch (const std::exception& e) {
        HCCL_ERROR(e.what());
        return HCCL_E_INTERNAL;
    }
}

HcclResult CommunicatorImpl::GetInstRanksByNetLayer(uint32_t netLayer, uint32_t** ranks, uint32_t* rankNum)
{
    CHK_PTR_NULL(rankGraph);
    u32 num = 0;
    rankListVec.clear();
    TRY_CATCH_RETURN(rankGraph->GetLocalInstRanks(netLayer, rankListVec, num));
    *ranks = rankListVec.data();
    *rankNum = num;
    return HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::GetInstTopoTypeByNetLayer(uint32_t netLayer, uint32_t* topoType)
{
    CHK_PTR_NULL(rankGraph);
    TRY_CATCH_RETURN(rankGraph->GetNetType(netLayer));
    auto type = rankGraph->GetNetType(netLayer);
    static const std::unordered_map<NetType, uint32_t> netTypeMap
        = {{NetType::CLOS, static_cast<uint32_t>(CommTopo::COMM_TOPO_CLOS)},
           {NetType::MESH_1D, static_cast<uint32_t>(CommTopo::COMM_TOPO_1DMESH)},
           {NetType::A3_SERVER, static_cast<uint32_t>(CommTopo::COMM_TOPO_910_93)},
           {NetType::A2_AX_SERVER, static_cast<uint32_t>(CommTopo::COMM_TOPO_A2AXSERVER)},
           {NetType::TOPO_FILE_DESC, static_cast<uint32_t>(CommTopo::COMM_TOPO_CUSTOM)}};

    auto it = netTypeMap.find(type);
    if (it != netTypeMap.end()) {
        *topoType = it->second;
        return HCCL_SUCCESS;
    }
    return HCCL_E_PARA;
}

HcclResult CommunicatorImpl::GetInstSizeListByNetLayer(uint32_t netLayer, uint32_t** instSizeList, uint32_t* listSize)
{
    try {
        CHK_PTR_NULL(rankGraph);
        u32 size = 0;
        instSizeVec.clear();
        auto ret = rankGraph->GetNetInstanceList(netLayer, instSizeVec, size);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[CommunicatorImpl::GetInstSizeListByNetLayer] Failed to get instSize[%p] at netLayer[%u]", listSize,
                netLayer);
            return ret;
        }
        *instSizeList = instSizeVec.data();
        *listSize = size;
        return HCCL_SUCCESS;
    } catch (const InvalidParamsException& e) {
        HCCL_ERROR(e.what());
        return HCCL_E_PARA;
    } catch (const NullPtrException& e) {
        HCCL_ERROR(e.what());
        return HCCL_E_PTR;
    } catch (const std::exception& e) {
        HCCL_ERROR(e.what());
        return HCCL_E_INTERNAL;
    }
}

static HcclResult InsertInnerLink(const NetInstance::Path& path, std::vector<CommLink>& linkListVec)
{
    for (const auto& link : path.links) {
        const NetInstance::Link* peer2peer = &link;
        for (LinkProtocol protocol : link.GetLinkProtocols()) {
            CommLink commLink;
            CommLinkInit(&commLink, 1);
            const CommProtocol& commProtocol = LinkProtocolToCommProtocol(protocol);
            commLink.linkAttr.linkProtocol = commProtocol;
            commLink.linkAttr.hop = peer2peer->GetHop();
            commLink.srcEndpointDesc.protocol = commProtocol;
            commLink.dstEndpointDesc.protocol = commProtocol;

            // 设置源端点
            std::shared_ptr<NetInstance::ConnInterface> srcConnInterface = link.GetSourceIface();
            CHK_PTR_NULL(srcConnInterface);
            HcclResult result = GetCommAddr(commLink.srcEndpointDesc.commAddr, srcConnInterface->GetAddr());
            if (result != HCCL_SUCCESS)
                return result;

            // 设置目标端点
            std::shared_ptr<NetInstance::ConnInterface> dstConnInterface = link.GetTargetIface();
            CHK_PTR_NULL(dstConnInterface);
            result = GetCommAddr(commLink.dstEndpointDesc.commAddr, dstConnInterface->GetAddr());
            if (result != HCCL_SUCCESS)
                return result;

            linkListVec.emplace_back(std::move(commLink));
        }
    }

    return HCCL_SUCCESS;
}

static HcclResult InsertClosLinks(const NetInstance::Path& path, std::vector<CommLink>& linkListVec)
{
    const NetInstance::Link* peer2net = nullptr;
    const NetInstance::Link* net2peer = nullptr;
    for (const auto& link : path.links) {
        bool srcNull = (link.GetSourceIface() == nullptr);
        bool dstNull = (link.GetTargetIface() == nullptr);
        if (!srcNull && dstNull) {
            peer2net = &link;
        } else if (srcNull && !dstNull) {
            net2peer = &link;
        }
    }
    auto srcInterface = peer2net->GetSourceIface();
    auto dstInterface = net2peer->GetTargetIface();
    CHK_PTR_NULL(srcInterface);
    CHK_PTR_NULL(dstInterface);
    for (LinkProtocol protocol : peer2net->GetLinkProtocols()) {
        CommLink commLink;
        CommLinkInit(&commLink, 1);
        const CommProtocol& commProtocol = LinkProtocolToCommProtocol(protocol);

        commLink.linkAttr.linkProtocol = commProtocol;
        commLink.linkAttr.hop = peer2net->GetHop();

        commLink.srcEndpointDesc.protocol = commProtocol;
        commLink.dstEndpointDesc.protocol = commProtocol;

        // 设置源端点
        HcclResult result = GetCommAddr(commLink.srcEndpointDesc.commAddr, srcInterface->GetAddr());
        if (result != HCCL_SUCCESS)
            return result;
        // 设置目标端点
        result = GetCommAddr(commLink.dstEndpointDesc.commAddr, dstInterface->GetAddr());
        if (result != HCCL_SUCCESS)
            return result;
        linkListVec.emplace_back(std::move(commLink));
    }
    return HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::GetLinks(
    uint32_t netLayer, uint32_t srcRank, uint32_t dstRank, CommLink** linkList, uint32_t* listSize)
{
    try {
        CHK_PTR_NULL(rankGraph);
        std::vector<NetInstance::Path> paths = rankGraph->GetPaths(netLayer, srcRank, dstRank);
        linkListVec.clear();
        // 遍历每条path
        for (const auto& path : paths) {
            // 检查是否是Clos网络（有nullptr接口）
            bool isClos = false;
            for (const auto& link : path.links) {
                // fabric没有接口
                if (link.GetSourceIface() == nullptr || link.GetTargetIface() == nullptr) {
                    isClos = true;
                    break;
                }
            }
            if (!isClos) {
                // Peer2Peer网络：直接处理每条link
                HcclResult ret = InsertInnerLink(path, linkListVec);
                if (ret != HCCL_SUCCESS)
                    return ret;
            } else {
                // Clos网络：找到peer2net和net2peer，组合成一条链路
                HcclResult ret = InsertClosLinks(path, linkListVec);
                if (ret != HCCL_SUCCESS)
                    return ret;
            }
        }
        *linkList = linkListVec.data();
        *listSize = linkListVec.size();
        return HCCL_SUCCESS;
    } catch (const InvalidParamsException& e) {
        HCCL_ERROR(e.what());
        return HCCL_E_PARA;
    } catch (const NullPtrException& e) {
        HCCL_ERROR(e.what());
        return HCCL_E_PTR;
    } catch (const std::exception& e) {
        HCCL_ERROR(e.what());
        return HCCL_E_INTERNAL;
    }
}

HcclResult CommunicatorImpl::GetTopoInstsByLayer(uint32_t netLayer, uint32_t** topoInsts, uint32_t* topoInstNum)
{
    try {
        CHK_PTR_NULL(rankGraph);
        auto currNetType = rankGraph->GetNetType(netLayer);
        if (currNetType != NetType::TOPO_FILE_DESC) {
            HCCL_ERROR(
                "[CommunicatorImpl::GetTopoInstsByLayer] Only support TOPO_FILE_DESC netType ,current netType is [%d]",
                currNetType);
            return HCCL_E_PARA;
        }

        u32 num = 0;
        rankGraph->GetTopoInstsByLayer(netLayer, topoInstsVec, num);

        *topoInsts = topoInstsVec.data();
        *topoInstNum = topoInstsVec.size();

        return HCCL_SUCCESS;
    } catch (const InvalidParamsException& e) {
        HCCL_ERROR(e.what());
        return HCCL_E_PARA;
    } catch (const NullPtrException& e) {
        HCCL_ERROR(e.what());
        return HCCL_E_PTR;
    } catch (const std::exception& e) {
        HCCL_ERROR(e.what());
        return HCCL_E_INTERNAL;
    }
}

HcclResult CommunicatorImpl::GetTopoType(uint32_t netLayer, uint32_t topoInstId, CommTopo* topoType)
{
    try {
        CHK_PTR_NULL(rankGraph);
        auto currNetType = rankGraph->GetNetType(netLayer);
        if (currNetType != NetType::TOPO_FILE_DESC) {
            HCCL_ERROR(
                "[CommunicatorImpl::GetTopoInstsByLayer] Only support TOPO_FILE_DESC netType, current netType is [%d]",
                currNetType);
            return HCCL_E_PARA;
        }
        Hccl::TopoType type;
        HcclResult ret = rankGraph->GetTopoType(netLayer, topoInstId, type);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[CommunicatorImpl::GetTopoType] Failed to get topo type at netLayer [%u] ret=%d", netLayer, ret);
            return ret;
        }
        static const std::unordered_map<Hccl::TopoType, CommTopo> topoTypeMap
            = {{Hccl::TopoType::CLOS, COMM_TOPO_CLOS},
               {Hccl::TopoType::MESH_1D, COMM_TOPO_1DMESH},
               {Hccl::TopoType::A3_SERVER, COMM_TOPO_910_93},
               {Hccl::TopoType::A2_AX_SERVER, COMM_TOPO_A2AXSERVER}};
        auto it = topoTypeMap.find(type);
        if (it != topoTypeMap.end()) {
            *topoType = it->second;
            return HCCL_SUCCESS;
        }
        return HCCL_E_PARA;
    } catch (const InvalidParamsException& e) {
        HCCL_ERROR(e.what());
        return HCCL_E_PARA;
    } catch (const NullPtrException& e) {
        HCCL_ERROR(e.what());
        return HCCL_E_PTR;
    } catch (const std::exception& e) {
        HCCL_ERROR(e.what());
        return HCCL_E_INTERNAL;
    }
}

HcclResult
CommunicatorImpl::GetRanksByTopoInst(uint32_t netLayer, uint32_t topoInstId, uint32_t** ranks, uint32_t* rankNum)
{
    try {
        CHK_PTR_NULL(rankGraph);
        auto currNetType = rankGraph->GetNetType(netLayer);
        if (currNetType != NetType::TOPO_FILE_DESC) {
            HCCL_ERROR(
                "[CommunicatorImpl::GetTopoInstsByLayer] Only support TOPO_FILE_DESC netType, current netType is [%d]",
                currNetType);
            return HCCL_E_PARA;
        }
        u32 num = 0;
        auto ret = rankGraph->GetRanksByTopoInst(netLayer, topoInstId, ranksVec, num);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[CommunicatorImpl::GetRanksByTopoInst] Failed to get topo type at netLayer [%u] ret=%d", netLayer,
                ret);
            return ret;
        }
        *ranks = ranksVec.data();
        *rankNum = ranksVec.size();
        return HCCL_SUCCESS;
    } catch (const InvalidParamsException& e) {
        HCCL_ERROR(e.what());
        return HCCL_E_PARA;
    } catch (const NullPtrException& e) {
        HCCL_ERROR(e.what());
        return HCCL_E_PTR;
    } catch (const std::exception& e) {
        HCCL_ERROR(e.what());
        return HCCL_E_INTERNAL;
    }
}

HcclResult CommunicatorImpl::GetInstSizeByNetLayer(uint32_t netLayer, uint32_t* rankNum)
{
    try {
        CHK_PTR_NULL(rankGraph);
        u32 num = rankGraph->GetLocalInstSize(netLayer);
        *rankNum = static_cast<uint32_t>(num);
        return HCCL_SUCCESS;
    } catch (const InvalidParamsException& e) {
        HCCL_ERROR(e.what());
        return HCCL_E_PARA;
    } catch (const NullPtrException& e) {
        HCCL_ERROR(e.what());
        return HCCL_E_PTR;
    } catch (const std::exception& e) {
        HCCL_ERROR(e.what());
        return HCCL_E_INTERNAL;
    }
}

HcclResult CommunicatorImpl::GetEndpointNum(uint32_t layer, uint32_t topoInstId, uint32_t* num)
{
    CHK_PTR_NULL(rankGraph);
    HcclResult ret = rankGraph->GetEndpointNum(layer, topoInstId, num);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[CommunicatorImpl::GetEndpointNum] Failed to get endpoint num at netLayer [%u] with topoInstId[%u]", layer,
            topoInstId);
        return ret;
    }
    return HCCL_SUCCESS;
}

HcclResult
CommunicatorImpl::GetEndpointDesc(uint32_t layer, uint32_t topoInstId, uint32_t* descNum, EndpointDesc* endpointDesc)
{
    CHK_PTR_NULL(rankGraph);
    HcclResult ret = rankGraph->GetEndpointDesc(layer, topoInstId, descNum, endpointDesc);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[CommunicatorImpl::GetEndpointDesc] Failed to get endpoint desc at netLayer [%u] with descNum [%p]", layer,
            descNum);
        return ret;
    }
    return HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::GetEndpointInfo(
    uint32_t rankId, const EndpointDesc* endPointDesc, EndpointAttr endpointAttr, uint32_t infoLen, void* info)
{
    CHK_PTR_NULL(rankGraph);
    HcclResult ret = rankGraph->GetEndpointInfo(rankId, endPointDesc, endpointAttr, infoLen, info);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[CommunicatorImpl::GetEndpointInfo] Failed to get endpoint info with rank [%u]", rankId);
        return ret;
    }
    return HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::SaveTopoDesc(std::string& identifier)
{
    uint32_t topoType = 0;
    CHK_RET(GetInstTopoTypeByNetLayer(0, &topoType)); // layer 0

    CommTopoDesc::GetInstance().SaveRankSize(identifier, rankSize);
    CommTopoDesc::GetInstance().SaveL0TopoType(identifier, static_cast<CommTopo>(topoType));
    return HCCL_SUCCESS;
}

void CommunicatorImpl::CheckAcceleratorConsistency(
    AcceleratorState commAccelerator, AcceleratorState tilingAccelerator) const
{
    bool isCommAiv = (commAccelerator == AcceleratorState::AIV || commAccelerator == AcceleratorState::AIV_ONLY);
    bool isTilingCcu
        = (tilingAccelerator == AcceleratorState::CCU_MS || tilingAccelerator == AcceleratorState::CCU_SCHED);

    bool isCommCcu = (commAccelerator == AcceleratorState::CCU_MS || commAccelerator == AcceleratorState::CCU_SCHED);
    bool isTilingAiv = (tilingAccelerator == AcceleratorState::AIV || tilingAccelerator == AcceleratorState::AIV_ONLY);

    if ((isCommAiv && isTilingCcu) || (isCommCcu && isTilingAiv)) {
        HCCL_WARNING(
            "CommunicatorImpl::GetTilingAccelerator comm accelerator is [%s] but tiling accelerator is [%s]",
            commAccelerator.Describe().c_str(), tilingAccelerator.Describe().c_str());
    }
}

HcclResult CommunicatorImpl::GetTilingAccelerator(void* mc2Tiling, AcceleratorState& acceleratorState) const
{
    HCCL_INFO("[CommunicatorImpl::%s] start.", __func__);
    auto tilingVersion = *static_cast<uint32_t*>(mc2Tiling);
    HCCL_INFO("[CommunicatorImpl:%s] Tiling version [%u]", __func__, tilingVersion);
    if (tilingVersion != UNKNOWN_TILING_V1 && tilingVersion != UNKNOWN_TILING_V2) {
        HCCL_ERROR("[CommunicatorImpl::GetTilingAccelerator] Tiling version not support, version[%u]", tilingVersion);
        return HCCL_E_NOT_SUPPORT;
    }
    uint8_t accelerator{0};
    if (tilingVersion == UNKNOWN_TILING_V1) {
        // 从mc2Tiling中获取需要的算法信息,校验所有commConfig的communicationEngine是否一致
        Mc2Tiling* mc2TilingPtr = reinterpret_cast<Mc2Tiling*>(mc2Tiling);
        accelerator = static_cast<Mc2Tiling*>(mc2Tiling)->commConfig.communicationEngine;
        Mc2CommConfig* commConfigPtr = reinterpret_cast<Mc2CommConfig*>(
            reinterpret_cast<uint8_t*>(mc2TilingPtr) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(Mc2ServerCfg));
        for (uint32_t index = 0; index < mc2TilingPtr->commConfigNum; index++) {
            const Mc2CommConfig& commConfig = *(commConfigPtr + index);
            if (commConfig.communicationEngine != accelerator) {
                HCCL_ERROR(
                    "[CommunicatorImpl::GetTilingAccelerator] Input communicationEngine [%u] and [%u] not equal",
                    commConfig.communicationEngine, accelerator);
                return HCCL_E_PARA;
            }
        }
    } else {
        Mc2InitTilingInner* mc2TilingPtr = reinterpret_cast<Mc2InitTilingInner*>(mc2Tiling);
        const auto offset = mc2TilingPtr->offset[0];
        const auto& commConfig
            = *(reinterpret_cast<const Mc2CcTilingInner*>(reinterpret_cast<const uint8_t*>(mc2TilingPtr) + offset));
        accelerator = commConfig.communicationEngine;

        HCCL_INFO("[CommunicatorImpl::%s] tilingAccelerator[%u].", __func__, accelerator);
    }

    HcclAccelerator hcclAccelerator = HcclAccelerator::DEFAULT;
    if (accelerator <= HcclAccelerator::AICPU) {
        hcclAccelerator = static_cast<HcclAccelerator::Value>(accelerator);
    }
    HCCL_INFO("[CommunicatorImpl::%s] hcclAccelerator[%s].", __func__, hcclAccelerator.Describe().c_str());
    switch (hcclAccelerator) {
        case HcclAccelerator::CCU_SCHED:
            acceleratorState = AcceleratorState::CCU_SCHED;
            break;
        case HcclAccelerator::AIV:
            acceleratorState = AcceleratorState::AIV;
            break;
        case HcclAccelerator::AIV_ONLY:
            acceleratorState = AcceleratorState::AIV_ONLY;
            break;
        default:
            HCCL_ERROR(
                "[GetTilingAccelerator] Tiling hcclAccelerator not support, hcclAccelerator[%s]",
                hcclAccelerator.Describe().c_str());
            return HCCL_E_NOT_SUPPORT;
    }

    AcceleratorState commAccelerator = GetCommExecuteConfig().accState;
    CheckAcceleratorConsistency(commAccelerator, acceleratorState);

    return HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::CalcNumBlocks(
    const CollOpParams& opParams, int32_t aivCoreLimit, std::string& algName, u32& numBlocks) const
{
    HCCL_INFO(
        "[CommunicatorImpl::CalcNumBlocks] count[%llu], dataType[%s], opType[%s], aivCoreLimit[%d], algName[%s].",
        opParams.count, opParams.dataType.Describe().c_str(), opParams.opType.Describe().c_str(), aivCoreLimit,
        algName.c_str());
    numBlocks = aivCoreLimit;
    return HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::GetAlgExecParam(
    const CollOpParams& opParams, bool clearEnable, void*& commContext, u64& len, u32 aivCoreLimit)
{
    HCCL_INFO("[CommunicatorImpl::GetAlgExecParam] clearEnable[%d], aivCoreLimit[%u].", clearEnable, aivCoreLimit);
    bool ifAiv = true;
    std::string algName = "";
    CHK_RET(HcomSelectAlg(opParams, aivCoreLimit, ifAiv, algName));
    bool isAiv
        = (opExecuteConfig.accState == AcceleratorState::AIV || opExecuteConfig.accState == AcceleratorState::AIV_ONLY);
    if (!isAiv) {
        HCCL_WARNING("GetAlgExecParam only supported aiv.");
        return HCCL_E_NOT_SUPPORT;
    }

    u32 numBlocks = 0;
    CHK_RET(CalcNumBlocks(opParams, aivCoreLimit, algName, numBlocks));

    return collService->GetAlgExecParam(clearEnable, numBlocks, commContext, len);
}

HcclResult DeregisterOffloadSlaveStreams(const std::string& opTag);

HcclResult CommunicatorImpl::ClearOpResource(const std::string& opTag)
{
    HCCL_INFO("CommunicatorImpl::%s] opTag[%s]", __func__, opTag.c_str());
    // 清空stream资源
    CHK_RET(GetStreamManager().offload->ClearOpStream(opTag));
    // 清空workspaceMem资源
    offloadScrachBufferMap.erase(opTag);
    HCCL_RUN_INFO("[CommunicatorImpl][%s] offloadScratchBuffer free, opTag[%s]", __func__, opTag.c_str());
    // 清空input/output/scratch资源
    CHK_RET(GetDataBufferManager().Deregister(opTag));
    CHK_RET(GetLocalRmaBufManager().Dereg(opTag));
    // 清空transport资源
    CHK_RET(GetMemTransportManager()->ClearOpTransport(opTag));
    // 清空aicpu_ts—host侧打包资源
    CollServiceAiCpuImpl* aiCpuCollService
        = dynamic_cast<CollServiceAiCpuImpl*>(collServices[AcceleratorState::AICPU_TS].get());
    CHK_PTR_NULL(aiCpuCollService);
    CHK_RET(aiCpuCollService->ClearOpLoadedInfo(opTag));
    return HCCL_SUCCESS;
}

std::vector<LinkData> CommunicatorImpl::GetFullMeshLinks() const
{
    HCCL_INFO("[CommunicatorImpl::%s] start.", __func__);

    // 遍历所有rank，两两建链
    std::vector<LinkData> links;
    std::unordered_set<LinkData> linkDataSet;
    int rankSize = GetRankSize();
    int myRank = GetMyRank();
    for (int dRank = 0; dRank < rankSize; dRank++) {
        if (myRank == dRank) {
            continue;
        }
        for (u32 level = 0; level < MAX_NET_LAYER; level++) {
            vector<LinkData> tempLinks;
            std::vector<NetInstance::Path> paths = GetRankGraph()->GetPaths(level, myRank, dRank);
            for (NetInstance::Path& path : paths) {
                tempLinks.emplace_back(LinkData(path));
            }

            if (!tempLinks.empty()) {
                linkDataSet.insert(tempLinks.at(0));
                break;
            }
        }
    }

    links.assign(linkDataSet.begin(), linkDataSet.end());

    HCCL_INFO("[CommunicatorImpl::%s] end, links size[%zu]", __func__, links.size());
    return links;
}

ErrorMessageReport CommunicatorImpl::GetAicpuTaskException()
{
    HcclResult ret = HCCL_SUCCESS;
    ErrorMessageReport errorMessage;
    if (kfcStatusTransferD2H != nullptr) {
        ret = kfcStatusTransferD2H->Get(
            sizeof(KfcStatus) + sizeof(KfcErrType), sizeof(errorMessage), reinterpret_cast<uint8_t*>(&errorMessage));
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("GetAicpuTaskException get aicpu task exception failed.ret[%u]", ret);
        }
    } else {
        HCCL_ERROR("GetAicpuTaskException kfcStatusTransferD2H is nullptr");
    }
    HCCL_INFO("[CommunicatorImpl::GetAicpuTaskException] end");
    return errorMessage;
}

u32 CommunicatorImpl::GetRankInParentComm() const { return static_cast<u32>(rankInParentComm); }

u8 CommunicatorImpl::GetCommDepth() const { return commDepth; }
void CommunicatorImpl::RegisterAicpuKernel() { aicpuKernelHolder_.Load(); }

aclrtFuncHandle CommunicatorImpl::GetAicpuKernelFuncHandle(const char* kernelName) const
{
    return aicpuKernelHolder_.GetAicpuKernelFuncHandle(kernelName);
}

HcclResult CommunicatorImpl::Mc2AiCpuStreamAllocAndGetV2(rtStream_t* aiCpuStream)
{
    if (aicpuStreamManager == nullptr) {
        HCCL_ERROR("[CommunicatorImpl::Mc2AiCpuStreamAllocAndGetV2] aicpuStreamManager is nullPtr!");
        return HCCL_E_PTR;
    }
    aicpuStreamManager->AllocFreeStream();
    Stream* stream = aicpuStreamManager->GetFreeStream();
    *aiCpuStream = stream->GetPtr();
    HCCL_RUN_INFO("[CommunicatorImpl::Mc2AiCpuStreamAllocAndGetV2] success, stream %s", stream->Describe().c_str());
    return HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::SaveDpuStreamId()
{
    dpuStreamId = HrtGetStreamId(dpuStream);
    HCCL_INFO("[CommunicatorImpl::SaveDpuStreamId] dpuStreamId_[%u]", dpuStreamId);
    return HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::GetRankIpPortMap(RankIpPortMapPtr& rankIpPortMap)
{
    CHK_PTR_NULL(rankIpPortMap_);
    rankIpPortMap = rankIpPortMap_;
    return HCCL_SUCCESS;
}

HcclResult CommunicatorImpl::SetRankIpPortMap(const RankIpPortMapPtr& rankIpPortMap)
{
    CHK_PTR_NULL(rankIpPortMap);
    rankIpPortMap_ = rankIpPortMap;
    return HCCL_SUCCESS;
}

} // namespace Hccl

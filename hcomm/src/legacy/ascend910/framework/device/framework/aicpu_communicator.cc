/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iomanip>
#include "aicpu_communicator.h"
#include "dispatcher.h"
#include "aicpu_hccl_process.h"
#include "coll_alg_exec_registry.h"
#include "coll_all_to_all_executor.h"
#include "common/aicpu_hccl_common.h"
#include "executor_tracer.h"
#include "profiling_manager_device.h"
#include "utils/aicpu_hdc_utils.h"
#include "comm_engine_utils.h"
#include "utils/hccl_aicpu_utils.h"
#include "framework/aicpu_hdc.h"
#include "common/aicpu_sqe_context.h"
#include "coll_batch_send_recv_retry_executor.h"
#include "log_control.h"
#include "aicpu_hccl_sqcq.h"
#include "aicpu_hccl_sqcqv1.h"
#include "sal_pub.h"
#include "externalinput_pub.h"
#include "env_config.h"
#include "config_log.h"
#include "aicpu_one_side_service.h"
#include "notify_manager.h"
#include "dispatcher_aicpu.h"
#include "dlprof_function.h"
#include "profiling_command_handle.h"
#include "dispatcher_ctx.h"
#include "aicpu_res_package_helper.h"
#include "aicpu_symmetric_memory.h"

namespace hccl {
constexpr u32 IPC_SIGNAL_MODULUS = 2;
constexpr u32 RDMA_SIGNAL_MODULUS = 3;
constexpr u32 KEY_VALUE_TO_VECTOR_MODULUS = 2;

constexpr u32 BSR_RETRY_SEND_STREAM_INDEX = 0;
constexpr u32 BSR_RETRY_RECV_STREAM_INDEX = 1;
constexpr u32 BSR_RETRY_STREAM_NUM = 2;
constexpr u32 MAX_REPORT_STATUS = 100U; // reportStatus的最大缓存数量
constexpr u32 INPUT = 0;
constexpr u32 OUTPUT = 1;
constexpr u32 AICPU_RETRY_LINKROCE_DEFAULT = 0;
constexpr u32 AICPU_RETRY_LINKROCE_BACKUP = 1;

constexpr u32 BSR_RETRY_SENDRECV_PAIR_NUM_MAX = 2;
constexpr u32 BSR_RETRY_SENDRECV_PAIR_INDEX_0 = 0;
constexpr u32 BSR_RETRY_SENDRECV_PAIR_INDEX_1 = 1;

constexpr u32 NOTIFY_SIZE_FOUR = 4;
constexpr u32 NOTIFY_SIZE_EIGHT = 8;

bool HcclCommAicpu::errMessageReport_ = true;

#define HCCL_RETRY_CHK_RET_AND_TRANS_FSM(result__, exeLog__, error__, state__) \
    do {                                                                       \
        if (UNLIKELY((result__) != HCCL_SUCCESS)) {                            \
            exeLog__;                                                          \
            errorCode = (error__);                                             \
            fsmState = (state__);                                              \
            return (result__);                                                 \
        }                                                                      \
    } while (0)

HcclCommAicpu::HcclCommAicpu() { HCCL_RUN_INFO("Construct HcclCommAicpu complete."); }

HcclCommAicpu::~HcclCommAicpu()
{
    if (UtraceInfo_ != nullptr) {
        UtraceInfo_->DeInit();
        UtraceInfo_ = nullptr;
    }
    if (dispatcher_ != nullptr) {
        HcclDispatcherDestroy(dispatcher_);
        dispatcher_ = nullptr;
    }
    if (dispatcherCtx_ != nullptr) {
        DestroyDispatcherCtx(dispatcherCtx_, identifier_.c_str());
        HCCL_DEBUG("[%s] destroy dispatcherCtx[%p] group[%s] success!", __func__, dispatcherCtx_, identifier_.c_str());
        dispatcherCtx_ = nullptr;
    }

    commPlaneVector_.clear();
    isBridgeVector_.clear();
    indOpCommInitialized_ = false;
    initialized_ = false;

    HCCL_RUN_INFO("Destruct HcclCommAicpu group[%s] success!", identifier_.c_str());
}

HcclResult HcclCommAicpu::Init(const HcclOpResParam* commParam, bool isCustom)
{
    if (initialized_) {
        HCCL_RUN_INFO("[%s][Init]Group[%s] already initialized, skip reinit", __func__, identifier_.c_str());
        return HCCL_SUCCESS;
    }

    CHK_PTR_NULL(commParam);
    identifier_ = commParam->hcomId;
    isCustom_ = isCustom;
    HCCL_RUN_INFO(
        "[HcclCommAicpu][Init]Entry-Init group[%s], rankSize[%u], isCustom[%d].", identifier_.c_str(),
        commParam->rankSize, isCustom_);
    CHK_RET(aicpuShareData_.Init(commParam->aicpuCustomParamAddr, commParam->aicpuCustomParamSize));
    CHK_RET(SetHrtWorkMode(commParam));
    CHK_RET(SetHrtDeviceSatMode(commParam));
    CHK_RET(InitConfigInfo(commParam));
    CHK_RET(InitCclbuffer(commParam));
    CHK_RET(InitTopoInfo(commParam));
    CHK_RET(InitOpNotifyObj(commParam));
    CHK_RET(HcclDispatcherAicpuInit(&dispatcher_, devId_, commParam->hcclSdmaQos, DispatcherType::DISPATCHER_AICPU));
    CHK_RET(RegisterProfilingCallback());
    CHK_RET(InitLocalNotifyObj(commParam));
    CHK_RET(InitMainStreamObj(commParam));
    CHK_RET(InitSlaveStreamObjs(commParam));
    CHK_RET(InitOrderStreamObj(commParam));
    CHK_RET(InitLocalTagRes(commParam->localRes.nextTagRes));
    CHK_RET(InitTimeOutConfig(commParam));
    CHK_RET(InitHostDeviceLock(commParam));
    CHK_RET(InitTopoMatcher());
    CHK_RET(InitOpRetry(commParam));
    CHK_RET(RegisterDispatcherCallback());
    CHK_RET(InitTinyMem(commParam));
    CHK_RET(InitProfResource());
    CHK_RET(InitZeroCopyExchanger(commParam));
    CHK_RET(InitOpCounter(commParam->opCounterInfo));
    CHK_RET(InitUtraceInfo(commParam));
    CHK_RET(aicpuCacheManager_.InitOpUnfoldCache());
    CHK_RET(RegisterProfCallBack());
    InitCommInfoStatus(true);
    SetCommInfoStreamStatus(true);

    initialized_ = true;

    HCCL_RUN_INFO("[HcclCommAicpu][Init] group[%s] success!", identifier_.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::InitUtraceInfo(const HcclOpResParam* commParam)
{
    u32 hostpid = 0;
    u32 cpType = DEVDRV_PROCESS_CPTYPE_MAX;
    CHK_RET(HrtHalDrvQueryProcessHostPid(getpid(), nullptr, nullptr, &hostpid, &cpType));

    HcclTraceInfo::UtraceAttr utraceAttr;
    utraceAttr.utraceStatusFlag = commParam->utraceStatusFlag;
    utraceAttr.deviceid = GetDevId();
    utraceAttr.pid = hostpid;
    UtraceInfo_.reset(new (std::nothrow) HcclTraceInfo(utraceAttr));
    CHK_PTR_NULL(UtraceInfo_);

    /* 申请trace资源信息 */
    std::string logInfo = "HCCL_";
    logInfo.append(std::to_string(SalGetTid()));
    logInfo.append("_");
    logInfo.append(std::to_string(GetDevId()));
    CHK_RET(UtraceInfo_->Init(logInfo));
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::InitProfResource()
{
    groupHashId_ = dfx::ProfilingManager::GetProfHashId(identifier_.c_str(), identifier_.length());
    HCCL_RUN_INFO("[Init][ProfResource]group[%s], groupHashId_[%llu].", identifier_.c_str(), groupHashId_);

    dfx::ProfCommInfo profInfo{groupHashId_, topoInfo_.userRankSize, topoInfo_.userRank};
    CHK_RET(dfx::ProfilingManager::AddProfInfoByStreamId(mainStream_.id(), identifier_, profInfo));
    for (auto& slaveStream : slaveStreams_) {
        CHK_RET(dfx::ProfilingManager::AddProfInfoByStreamId(slaveStream.id(), identifier_, profInfo));
    }
    dfx::ProfilingExtendInfoHelper::InitProfItemId();
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::StreamRestore(u32 streamId)
{
    HcclResult ret = hrtHalResourceIdRestore(devId_, 0, DRV_STREAM_ID, streamId, 0);
    // custom进程需要恢复stream资源, custom进程调用失败直接报错，aicpu进程调用失败做兼容性处理
    if (ret == HCCL_E_NOT_SUPPORT) {
        CHK_PRT_RET(
            isCustom_,
            HCCL_ERROR(
                "%s hrtHalResourceIdRestore fail, drv not support, custom[%d], ret[%d]", __func__, isCustom_, ret),
            HCCL_E_DRV);
    } else if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("%s hrtHalResourceIdRestore fail, ret[%d]", __func__, ret);
        return HCCL_E_DRV;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::SetHrtWorkMode(const HcclOpResParam* commParam)
{
    CHK_RET(hrtSetWorkModeAicpu(true));
    CHK_RET(hrtSetlocalDevice(commParam->topoInfo.deviceLogicId));
    CHK_RET(hrtSetlocalDeviceType(static_cast<DevType>(commParam->topoInfo.deviceType)));
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::SetHrtDeviceSatMode(const HcclOpResParam* commParam)
{
    CHK_RET(hrtSetLocalDeviceSatMode(commParam->config.floatOverflowMode));
    HCCL_RUN_INFO(
        "[HcclCommAicpu][Init]SetHrtDeviceSatMode[%d]", static_cast<u32>(commParam->config.floatOverflowMode));
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::InitTopoMatcher()
{
    externalEnable_.enableFfts = 1; // FFTS+在算法模块不会使用多线程，固定使能
    externalEnable_.deterministic = deterministic_;
    externalEnable_.intraRoceSwitch = 0;
    externalEnable_.dumpDebug = dumpDebug_;
    externalEnable_.interHccsDisable = interHccsDisable_;

    topoMatcher_.reset((new (std::nothrow) TopoMatcher(
        commPlaneVector_, isBridgeVector_, topoInfo_, algoInfo_, externalEnable_, serverAndsuperPodToRank_)));
    CHK_SMART_PTR_NULL(topoMatcher_);
    HCCL_RUN_INFO(
        "[HcclCommAicpu][InitTopoMatcher]topo matcher init success. group[%s] deterministic:%u, "
        "dumpDebug:%u, interHccsDisable:%u",
        identifier_.c_str(), deterministic_, dumpDebug_, interHccsDisable_);
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::InitOpRetry(const HcclOpResParam* commParam)
{
    CHK_PTR_NULL(commParam);
    retryEnable_ = (commParam->config.retryEnable == 1) ? true : false;
    retryHoldTime_ = commParam->config.retryHoldTime;
    retryIntervalTime_ = commParam->config.retryIntervalTime;
    HCCL_RUN_INFO(
        "[InitOpRetry]retryEnable[%d], retryHoldTime[%u ms], retryIntervalTime[%u ms]", retryEnable_, retryHoldTime_,
        retryIntervalTime_);

    if (commParam->kfcControlTransferH2DParams.buffLen != 0 && kfcControlTransferH2D_ == nullptr) {
        EXCEPTION_CATCH((kfcControlTransferH2D_ = std::make_shared<hccl::HDCommunicate>()), return HCCL_E_PTR);
        CHK_SMART_PTR_NULL(kfcControlTransferH2D_);
        CHK_RET(kfcControlTransferH2D_->InitDevice(commParam->kfcControlTransferH2DParams));
    }
    if (commParam->kfcStatusTransferD2HParams.buffLen != 0 && kfcStatusTransferD2H_ == nullptr) {
        EXCEPTION_CATCH((kfcStatusTransferD2H_ = std::make_shared<hccl::HDCommunicate>()), return HCCL_E_PTR);
        CHK_SMART_PTR_NULL(kfcStatusTransferD2H_);
        CHK_RET(kfcStatusTransferD2H_->InitDevice(commParam->kfcStatusTransferD2HParams));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::InitZeroCopyExchanger(const HcclOpResParam* commParam)
{
    CHK_PTR_NULL(commParam);
    auto nSecStopFunc = [this]() -> bool {
        // 检查到OP状态不Ok则认为需要终止
        auto ret = this->CheckOpExecStatus();
        if (ret == HCCL_SUCCESS) {
            return false;
        } else if (ret == HCCL_E_SUSPENDING) {
            // NS快恢场景，需要提前终止
            HcclOpExecFSM fsmState = HcclOpExecFSM::HCCL_OP_EXEC_FSM_INIT;
            KfcError errorCode = KfcError::kNone;
            SetCommRecoveryFlag(true);
            UpdateOpExecStatus(fsmState, KfcStatus::kStoplaunch, errorCode, 0);
            HCCL_RUN_INFO("[HcclCommAicpu][nSecStopFunc] need stop launch");
            return true;
        } else {
            return true;
        }
    };

    u32 timeoutSec = commParam->config.notifyWaitTime;
    HCCL_INFO("[HcclCommAicpu][InitZeroCopyExchanger] set timeout is [%u s]", timeoutSec);

    EXCEPTION_CATCH(
        (ZeroCopyExchanger_ = std::make_shared<hccl::AicpuZeroCopyExchanger>(
             commParam->localUsrRankId, commParam->rankSize, commParam, nSecStopFunc, timeoutSec,
             topoInfo_.deviceNumPerAggregation, taskMonitorInterval_)),
        return HCCL_E_PTR);

    // 通信域第一次初始化时，如果IPC内存有不为空的则认为是使能该特性
    isZeroCopy_ = false;
    for (u32 i = 0; i < AICPU_ZERO_COPY_MAX_DEVICE_NUM_A3; ++i) {
        if (commParam->zeroCopyIpcPtrs[i] != 0) {
            isZeroCopy_ = true;
            break;
        }
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::InitOpCounter(const OpCounterInfo& opCounterInfo)
{
    if (opCounterInfo.isEnableCounter && !retryEnable_
        && (opCounterInfo.headCountMem == 0 || opCounterInfo.tailCountMem == 0 || opCounterInfo.memSize == 0)) {
        HCCL_ERROR("[HcclCommAicpu][InitOpCounter] headCountMem or tailCountMem or memSize is null");
        return HCCL_E_PARA;
    }
    opCounterInfo_ = opCounterInfo;
    return HCCL_SUCCESS;
}

void HcclCommAicpu::SetZeroCopyEnable(bool enable) { isZeroCopy_ = enable; }

void HcclCommAicpu::SetSymmetricMemoryEnable(bool enable)
{
    HCCL_INFO("[HcclCommAicpu::SetSymmetricMemoryEnable] enable[%d]", enable);
    isSymmetricMemory_ = enable;
    aicpuCacheManager_.SetSymmetricMemoryEnable(enable);
}

HcclResult HcclCommAicpu::PrepareZeroCopyExchanger(
    const std::string& newTag, OpParam& opParam, AlgResourceResponse* algResResponse)
{
    return ZeroCopyExchanger_->ExchangeAddress(newTag, opParam.inputPtr, opParam.outputPtr, algResResponse);
}

HcclResult HcclCommAicpu::RegisterProfilingCallback()
{
    (void)RegisterLoadTaskCallBack(dispatcher_, nullptr, dfx::TaskProfilingCallBack);

    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::RegisterDispatcherCallback()
{
    auto checkOpExecStatusCallback = [this]() {
        return this->CheckOpExecStatusCallback();
    };

    return HcclSetOpExecStatusCallback(dispatcher_, checkOpExecStatusCallback);
}

HcclResult HcclCommAicpu::RegisterProfCallBack()
{
    if (MsprofRegisterCallback != nullptr) {
        HCCL_INFO("RegisterProfCallBack not null");
        int32_t ret = MsprofRegisterCallback(AICPU, &DeviceCommandHandle);
        CHK_PRT_RET((ret != 0), HCCL_ERROR("[%s] failed. ret = [%d]", __func__, ret), HCCL_E_PARA);
    } else {
        HCCL_INFO("RegisterProfCallBack is null");
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::GetSuspendingFlag(HcclComSuspendingFlag& flag)
{
    CHK_RET(AicpuHdcUtils::GetSuspendingStatus(kfcControlTransferH2D_, flag));
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::BackGroundGetCmd(KfcCommand& cmd)
{
    CHK_RET(aicpuHdc_.GetOpExecCtrlCmd(kfcControlTransferH2D_, cmd));
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::BackGroundSetStatus(KfcStatus status)
{
    CHK_RET(aicpuHdc_.SetOpExecStatus(kfcStatusTransferD2H_, status, KfcError::kNone, 0));
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::SaveTraceInfo(std::string& logInfo)
{
    CHK_PTR_NULL(UtraceInfo_);
    CHK_RET(UtraceInfo_->SaveTraceInfo(logInfo, AtraceOption::Opbasekey));
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::FlushUtraceInfo()
{
    if (GetCommInfoStatus()) {
        CHK_RET(UtraceInfo_->Flush());
    }
    return HCCL_SUCCESS;
}

std::string HcclCommAicpu::GetExcuteOp()
{
    std::stringstream ss;
    ss << "tag: " << excuteOpId_.tag << ", ";
    ss << "newTag: " << excuteOpId_.newTag << ", ";
    ss << "index: " << excuteOpId_.index;
    return ss.str();
}

void HcclCommAicpu::InitCommInfoStatus(bool commInfo) { commOpenStatus = commInfo; }

HcclResult HcclCommAicpu::InitTinyMem(const HcclOpResParam* commParam)
{
    CHK_PTR_NULL(commParam);
    auto tinyMemPtr = reinterpret_cast<void*>(commParam->tinyMem);
    tinySendRecvMem_ = DeviceMem::create(tinyMemPtr, commParam->tinyMemSize);

    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::InitTimeOutConfig(const HcclOpResParam* commParam)
{
    CHK_PTR_NULL(commParam);
    CHK_RET(HcclSetSqeTimeOut(dispatcher_, commParam->config.notifyWaitTime));
    linkTimeOut_ = commParam->config.linkTimeOut;
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::InitHostDeviceLock(const HcclOpResParam* commParam)
{
    CHK_PTR_NULL(commParam);
    hostDeviceLock_.reset(new (std::nothrow) PetersonLock(
        reinterpret_cast<void*>(commParam->lockAddr), PetersonLock::DEFAULT_LOCK_TIMEOUT_SEC));
    CHK_SMART_PTR_NULL(hostDeviceLock_);
    CHK_RET(hostDeviceLock_->Init());

    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::UpdateNotifyWaitTimeOut(SyncMode syncMode, u64 notifyWaitTime)
{
    sqeWaitTimeOut_ = (notifyWaitTime == 0) ? notifyWaitTime : (notifyWaitTime + AICPU_SQE_TIMEOUT_INC);
    if (syncMode == SyncMode::UNLIMITED_TIMEWAITSYNCMODE) {
        CHK_RET(HcclSetSqeTimeOut(dispatcher_, GetNotifyMaxWaitTime()));
    }
    return HcclSetSqFullWaitTimeOut(dispatcher_, notifyWaitTime);
}

void HcclCommAicpu::PrepareOpRetryHandler(
    u8 inplaceSupportRetry, u8 retryEnable, u8 inPlaceSupportRetryStatus, u8 isInplacePreSync, u8 isPostSync)
{
    algOpContext_.opRetryHandler.inplaceSupportRetry = static_cast<bool>(inplaceSupportRetry);
    algOpContext_.opRetryHandler.retryEnable = static_cast<bool>(retryEnable);
    algOpContext_.opRetryHandler.inPlaceSupportRetryStatus
        = static_cast<InplaceSupportRetryStatus>(inPlaceSupportRetryStatus);
    algOpContext_.opRetryHandler.isInplacePreSync = static_cast<bool>(isInplacePreSync);
    algOpContext_.opRetryHandler.isPostSync = static_cast<bool>(isPostSync);
    HCCL_INFO(
        "[HcclCommAicpu][PrepareOpRetryHandler] inplaceSupportRetry %d, retryEnable %d, "
        "inPlaceSupportRetryStatus %d, isInplacePreSync %d, isPostSync %d.",
        algOpContext_.opRetryHandler.inplaceSupportRetry, algOpContext_.opRetryHandler.retryEnable,
        algOpContext_.opRetryHandler.inPlaceSupportRetryStatus, algOpContext_.opRetryHandler.isInplacePreSync,
        algOpContext_.opRetryHandler.isPostSync);
}

HcclResult HcclCommAicpu::UpdateOpRingBufferIdx()
{
    return HcclSetOpRingBufferIdx(dispatcher_, aicpuShareData_.GetOpRingBufferIdx());
}

HcclResult HcclCommAicpu::InvokeKfcHandler(AicpuKfcHandlerType type, const std::vector<u64> args)
{
    CHK_PRT_RET(
        static_cast<size_t>(type) >= static_cast<size_t>(AicpuKfcHandlerType::kMax),
        HCCL_ERROR("Device mode %u, handler type %u.", static_cast<u32>(isDeviceMode_), static_cast<size_t>(type)),
        HCCL_E_INTERNAL);
    const auto handler = kfcHandlers_[static_cast<size_t>(type)];
    if (handler == nullptr) {
        return HCCL_SUCCESS;
    }
    return handler(args);
}

HcclResult HcclCommAicpu::NotifyPost(void)
{
    if (isDeviceMode_) {
        return InvokeKfcHandler(
            AicpuKfcHandlerType::kNotifyRecord,
            {rpc_, reinterpret_cast<u64>(dispatcher_), reinterpret_cast<u64>(&mainStream_)});
    } else {
        CHK_RET(LocalNotify::Post(mainStream_, dispatcher_, opNotifies_[1]));
        HcclSqeContext* sqeContext = mainStream_.GetSqeContextPtr();
        SqeRingBuffer* sqeContextBuffer = &(sqeContext->buffer);
        return dfx::ProfilingManager::ReportMainStreamTask(mainStream_, sqeContextBuffer->tailSqeTaskId - 1, TAIL_TASK);
    }
}

HcclResult HcclCommAicpu::NotifyWait(void)
{
    if (isDeviceMode_) {
        return InvokeKfcHandler(
            AicpuKfcHandlerType::kNotifyWait,
            {rpc_, reinterpret_cast<u64>(dispatcher_), reinterpret_cast<u64>(&mainStream_)});
    } else {
        HcclSqeContext* sqeContext = mainStream_.GetSqeContextPtr();
        SqeRingBuffer* sqeContextBuffer = &(sqeContext->buffer);
        CHK_RET(dfx::ProfilingManager::ReportMainStreamTask(mainStream_, sqeContextBuffer->tailSqeTaskId, HEAD_TASK));
        return LocalNotify::Wait(mainStream_, dispatcher_, opNotifies_[0]);
    }
}

// 按照算子模式来Post对应的Notify
HcclResult HcclCommAicpu::RecordHostOrder(const HcclOpResParam* commParam, const std::string& tag, u8 orderLaunchMode)
{
    const u8 orderLaunchInvalidInHcom = 255;
    if (orderLaunchMode == orderLaunchInvalidInHcom) {
        HCCL_INFO("[%s] attachedStreams_[%d] is invalid in graph mode", __func__, orderStream_.id());
        return HCCL_SUCCESS;
    }
    if (orderNotifies_[orderLaunchMode] == nullptr) {
        std::shared_ptr<LocalNotify> notify;
        HcclSignalInfo* aicpuOrderNotify = reinterpret_cast<HcclSignalInfo*>(
            static_cast<u64>(commParam->aicpuOrderNotifyAddr) + (sizeof(HcclSignalInfo) * orderLaunchMode));
        HCCL_INFO(
            "[%s] attachedStreams_[%d] aicpuOrderNotify resId[%llu], addr[0x%llx], flag[%u], devId[%u], tsId[%u], "
            "rankId[%u]",
            __func__, orderStream_.id(), aicpuOrderNotify->resId, aicpuOrderNotify->addr, aicpuOrderNotify->flag,
            aicpuOrderNotify->devId, aicpuOrderNotify->tsId, aicpuOrderNotify->rankId);
        HcclResult ret = InitAndVerifySingleSignal(*aicpuOrderNotify, notify);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s] check localRes notify failed, resId[%u], group[%s]", __func__, aicpuOrderNotify->resId,
                identifier_.c_str()),
            ret);
        orderNotifies_[orderLaunchMode] = notify;
        HCCL_INFO("%s success, group[%s], resId[%llu]", __func__, identifier_.c_str(), aicpuOrderNotify->resId);
    }

    HCCL_INFO(
        "%s group[%s] tag[%s] isDeviceMode[%d] orderLaunchMode[%u] mode[%d] streamId[%d] notifyId[%u]", __func__,
        identifier_.c_str(), tag.c_str(), isDeviceMode_, orderLaunchMode, GetWorkflowMode(), orderStream_.id(),
        orderNotifies_[orderLaunchMode]->notifyId_);
    CHK_RET(LocalNotify::Post(orderStream_, dispatcher_, orderNotifies_[orderLaunchMode]));
    CHK_RET(LaunchTask(dispatcher_, const_cast<Stream&>(orderStream_)));
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::GetStreamData(
    const HcclStreamInfo& streamInfo, HcclComStreamInfo& comStreamInfo, u32& sqHead, u32& sqTail)
{
    comStreamInfo.sqId = streamInfo.sqIds;
    comStreamInfo.actualStreamId = streamInfo.streamIds;
    comStreamInfo.logicCqId = streamInfo.logicCqids;
    u64 sq_addr = 0;
    CHK_RET(QuerySqBaseAddr(devId_, streamInfo.sqIds, sq_addr));
    comStreamInfo.sqBaseAddr = reinterpret_cast<void*>(sq_addr);
    if (comStreamInfo.sqBaseAddr == nullptr) {
        HCCL_ERROR("[HcclCommAicpu][GetStreamData]sqe base addr ptr is null.");
        return HCCL_E_PARA;
    }
    CHK_RET(QuerySqStatusByType(devId_, streamInfo.sqIds, DRV_SQCQ_PROP_SQ_DEPTH, comStreamInfo.sqDepth));
    CHK_RET(QuerySqStatusByType(devId_, streamInfo.sqIds, DRV_SQCQ_PROP_SQ_TAIL, sqTail));
    CHK_RET(QuerySqStatusByType(devId_, streamInfo.sqIds, DRV_SQCQ_PROP_SQ_HEAD, sqHead));
    HCCL_DEBUG(
        "[HcclCommAicpu][GetStreamData] get stream data success, group[%s], streamId[%d], sqId[%d], "
        "logicCqId[%u], sqDepth[%u], sqHead[%u], sqTail[%u]",
        identifier_.c_str(), comStreamInfo.actualStreamId, comStreamInfo.sqId, comStreamInfo.logicCqId,
        comStreamInfo.sqDepth, sqHead, sqTail);
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::InitMainStreamObj(const HcclOpResParam* commParam)
{
    CHK_RET(InitStreamObj(commParam->localRes.mainStreamParam, mainStream_));
    HCCL_INFO("%s success, group[%s], streamId[%d]", __func__, identifier_.c_str(), mainStream_.id());
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::InitSlaveStreamObjs(const HcclOpResParam* commParam)
{
    if (commParam->localRes.streamNum > LOCAL_STREAM_MAX_NUM) {
        HCCL_ERROR(
            "[HcclCommAicpu][InitSlaveStreamObjs] local streams exceed max number, current numbers[%u], max "
            "numbers[%u], group[%s]",
            commParam->localRes.streamNum, LOCAL_STREAM_MAX_NUM, identifier_.c_str());
        return HCCL_E_PARA;
    }
    for (u32 i = 0; i < commParam->localRes.streamNum; i++) {
        if (streamToObj_.find(commParam->localRes.streamParam[i].streamInfo.sqIds) == streamToObj_.end()) {
            Stream stream;
            CHK_RET(InitStreamObj(commParam->localRes.streamParam[i], stream));
            slaveStreams_.emplace_back(stream);
            streamToObj_.insert(commParam->localRes.streamParam[i].streamInfo.sqIds);
        }
    }
    HCCL_DEBUG(
        "[HcclCommAicpu][InitSlaveStreamObjs] success, group[%s], slave stream numbers[%u]", identifier_.c_str(),
        commParam->localRes.streamNum);
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::InitOrderStreamObj(const HcclOpResParam* commParam)
{
    CHK_RET(InitStreamObj(commParam->aicpuOrderStreamParam, orderStream_));
    HCCL_INFO("%s success, group[%s], streamId[%d]", __func__, identifier_.c_str(), orderStream_.id());
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::InitStreamObj(const HcclStreamParam& streamParam, Stream& stream)
{
    u32 sqTail;
    u32 sqHead;
    HcclResult ret = HCCL_SUCCESS;
    ret = StreamRestore(streamParam.streamInfo.streamIds);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "%s StreamId[%d] Restore error group[%s]", __func__, streamParam.streamInfo.streamIds, identifier_.c_str()),
        ret);

    HcclComStreamInfo comStreamInfo = {};
    ret = GetStreamData(streamParam.streamInfo, comStreamInfo, sqHead, sqTail);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("%s error group[%s]", __func__, identifier_.c_str()), ret);
    stream = Stream(comStreamInfo);

    // 初始化stream的sqeContext
    SqCqeContext* sqCqeContext = reinterpret_cast<SqCqeContext*>(streamParam.sqCqContextAddr);
    u64 sqCqContextSize = streamParam.sqCqContextSize;
    if (sqCqeContext == nullptr || sqCqContextSize != sizeof(SqCqeContext)) {
        HCCL_ERROR(
            "%s failed, sqCqeContext[%p] is null or size[%llu] does not match expected size[%llu], group[%s]", __func__,
            sqCqeContext, sqCqContextSize, sizeof(SqCqeContext), identifier_.c_str());
        return HCCL_E_PARA;
    }
    ret = stream.InitSqAndCqeContext(sqHead, sqTail, sqCqeContext);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("%s: InitSqAndCqeContext failed, group[%s]", __func__, identifier_.c_str()),
        ret);
    HCCL_INFO("%s success, group[%s], streamId[%d]", __func__, identifier_.c_str(), stream.id());
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::InitOpNotifyObj(const HcclOpResParam* commParam)
{
    HcclResult ret = HCCL_SUCCESS;
    for (u32 i = 0; i < AICPU_OP_NOTIFY_MAX_NUM; i++) {
        ret = InitAndVerifySignal(commParam->localRes.aicpuOpNotify[i], opNotifies_);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[HcclCommAicpu][InitOpNotifyObj] check localRes op notify failed, resId[%u], group[%s]",
                commParam->localRes.localSignals[i].resId, identifier_.c_str()),
            ret);
    }
    u32 hostPid = 0;
    CHK_RET(HrtHalDrvQueryProcessHostPid(getpid(), &devId_, nullptr, &hostPid, nullptr));
    HCCL_INFO("[HcclCommAicpu][InitOpNotifyObj] success, group[%s]", identifier_.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::InitLocalNotifyObj(const HcclOpResParam* commParam)
{
    HcclResult ret = HCCL_SUCCESS;

    if (commParam->localRes.signalNum > LOCAL_NOTIFY_MAX_NUM) {
        HCCL_ERROR(
            "[HcclCommAicpu][InitLocalNotifyObj] local notify count exceeds max number, numbers[%u], max "
            "numbers[%u], group[%s]",
            commParam->localRes.signalNum, LOCAL_NOTIFY_MAX_NUM, identifier_.c_str());
        return HCCL_E_PARA;
    }
    for (u32 i = 0; i < commParam->localRes.signalNum; i++) {
        if (notifysToObj_.find(commParam->localRes.localSignals[i].resId) == notifysToObj_.end()) {
            ret = InitAndVerifySignal(commParam->localRes.localSignals[i], localNotifies_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[HcclCommAicpu][InitLocalNotifyObj] check localRes prenotify failed, resId[%u]",
                    commParam->localRes.localSignals[i].resId),
                ret);
            notifysToObj_.insert(commParam->localRes.localSignals[i].resId);
        }
    }
    HCCL_DEBUG(
        "[HcclCommAicpu][InitLocalNotifyObj] success, group[%s], signal numbers[%u]", identifier_.c_str(),
        commParam->localRes.signalNum);
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::ParseTlvToVector(
    u64 srcTlv, u64 srcTlvTotalLength, std::vector<std::vector<std::vector<u32>>>& vectorInfo)
{
    u64 parseLength = 0;
    u8* srcTlvptr = reinterpret_cast<u8*>(srcTlv);
    if (srcTlvptr == nullptr) {
        HCCL_ERROR("[HcclCommAicpu][ParseTlvToVector]tlv ptr is null.");
        return HCCL_E_PARA;
    }
    u64 unPareseLength = srcTlvTotalLength;
    while (unPareseLength > 0) {
        CommonTlv* commonTlv = reinterpret_cast<CommonTlv*>(srcTlvptr + parseLength);
        if (unPareseLength <= (sizeof(LENGTH_TYPE) + sizeof(TAG_TYPE))) {
            HCCL_ERROR(
                "[HcclCommAicpu][ParseTlvToVector] Tlv length is error, tag[%s], totalLength[%lu], "
                "unParseLength[%lu], already parsed Length[%lu]",
                identifier_.c_str(), srcTlvTotalLength, unPareseLength, parseLength);
            return HCCL_E_PARA;
        }
        if (commonTlv->length > unPareseLength || commonTlv->length % sizeof(RANK_TYPE) != 0) {
            HCCL_ERROR(
                "[HcclCommAicpu][ParseTlvToVector] parse Tlv error, group[%s], total Length[%lu], tlvLength[%lu], "
                "unParsed Length[%lu], already parsed Length[%lu]",
                identifier_.c_str(), srcTlvTotalLength, commonTlv->length, unPareseLength, parseLength);
            return HCCL_E_PARA;
        }
        u16 level0 = (commonTlv->type & TOP_COMM_LEVEL0_LOCATION) >> TOP_COMM_LEVEL0_SHIFT;
        u16 level1 = (commonTlv->type & TOP_COMM_LEVEL1_LOCATION);
        u64 itemNum = (commonTlv->length - sizeof(TAG_TYPE) - sizeof(LENGTH_TYPE)) / sizeof(RANK_TYPE);
        std::vector<RANK_TYPE> values{&commonTlv->value, (&commonTlv->value) + itemNum};
        if (level0 >= vectorInfo.size()) {
            vectorInfo.resize(level0 + 1);
        }
        if (level1 >= vectorInfo[level0].size()) {
            vectorInfo[level0].resize(level1 + 1);
        }
        vectorInfo[level0][level1] = std::move(values);
        parseLength += commonTlv->length;
        unPareseLength -= commonTlv->length;
        HCCL_DEBUG(
            "[HcclCommAicpu][ParseTlvToVector] parse Tlv group[%s], level0[%u], level1[%u], total Length[%lu], "
            "tlvLength[%lu], unParsed Length[%lu], already parsed Length[%lu]",
            identifier_.c_str(), level0, level1, srcTlvTotalLength, commonTlv->length, unPareseLength, parseLength);
    }
    for (u32 idx = 0; idx < vectorInfo.size(); idx++) {
        for (u32 ringidx = 0; ringidx < vectorInfo[idx].size(); ringidx++) {
            HCCL_DEBUG(
                "[HcclCommAicpu][ParseTlvToVector] idx[%u] ringidx[%u] size[%u]", idx, ringidx,
                vectorInfo[idx][ringidx].size());
        }
    }
    HCCL_INFO("[HcclCommAicpu][ParseTlvToVector] success, group[%s]", identifier_.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::ParseTlvToSubGroupVector(
    u64 srcTlv, u64 srcTlvTotalLength, std::vector<std::vector<std::vector<std::vector<u32>>>>& vectorInfo)
{
    u64 parseLength = 0;
    u8* srcTlvptr = reinterpret_cast<u8*>(srcTlv);
    if (srcTlvptr == nullptr) {
        HCCL_ERROR("[HcclCommAicpu][ParseTlvToSubGroupVector]tlv ptr is null.");
        return HCCL_E_PARA;
    }
    u64 unPareseLength = srcTlvTotalLength;
    while (unPareseLength > 0) {
        CommonTlv* commonTlv = reinterpret_cast<CommonTlv*>(srcTlvptr + parseLength);
        if (unPareseLength <= (sizeof(LENGTH_TYPE) + sizeof(TAG_TYPE))) {
            HCCL_ERROR(
                "[HcclCommAicpu][ParseTlvToSubGroupVector] Tlv length is error, tag[%s], totalLength[%lu], "
                "unParseLength[%lu], already parsed Length[%lu]",
                identifier_.c_str(), srcTlvTotalLength, unPareseLength, parseLength);
            return HCCL_E_PARA;
        }
        if (commonTlv->length > unPareseLength || commonTlv->length % sizeof(RANK_TYPE) != 0) {
            HCCL_ERROR(
                "[HcclCommAicpu][ParseTlvToSubGroupVector] parse Tlv error, group[%s], total Length[%lu], "
                "tlvLength[%lu], "
                "unParsed Length[%lu], already parsed Length[%lu]",
                identifier_.c_str(), srcTlvTotalLength, commonTlv->length, unPareseLength, parseLength);
            return HCCL_E_PARA;
        }
        u16 level0 = (commonTlv->type & TOP_HIERARCHICAL_COMM_LEVEL0_LOCATION)
                     >> (TOP_HIERARCHICAL_COMM_LEVEL0_SHIFT + TOP_HIERARCHICAL_COMM_LEVEL1_SHIFT);
        u16 level1 = (commonTlv->type & TOP_HIERARCHICAL_COMM_LEVEL1_LOCATION) >> TOP_HIERARCHICAL_COMM_LEVEL1_SHIFT;
        u16 level2 = (commonTlv->type & TOP_HIERARCHICAL_COMM_LEVEL2_LOCATION);
        u64 itemNum = (commonTlv->length - sizeof(TAG_TYPE) - sizeof(LENGTH_TYPE)) / sizeof(RANK_TYPE);
        std::vector<RANK_TYPE> values{&commonTlv->value, (&commonTlv->value) + itemNum};
        if (level0 >= vectorInfo.size()) {
            vectorInfo.resize(level0 + 1);
        }
        if (level1 >= vectorInfo[level0].size()) {
            vectorInfo[level0].resize(level1 + 1);
        }
        if (level2 >= vectorInfo[level0][level1].size()) {
            vectorInfo[level0][level1].resize(level2 + 1);
        }
        vectorInfo[level0][level1][level2] = std::move(values);
        parseLength += commonTlv->length;
        unPareseLength -= commonTlv->length;
        HCCL_DEBUG(
            "[HcclCommAicpu][ParseTlvToSubGroupVector] parse Tlv group[%s], level0[%u], level1[%u], total Length[%lu], "
            "tlvLength[%lu], unParsed Length[%lu], already parsed Length[%lu]",
            identifier_.c_str(), level0, level1, srcTlvTotalLength, commonTlv->length, unPareseLength, parseLength);
    }
    for (u32 level0Idx = 0; level0Idx < vectorInfo.size(); level0Idx++) {
        for (u32 level1Idx = 0; level1Idx < vectorInfo[level0Idx].size(); level1Idx++) {
            for (u32 level2Idx = 0; level2Idx < vectorInfo[level0Idx][level1Idx].size(); level2Idx++) {
                HCCL_DEBUG(
                    "[HcclCommAicpu][ParseTlvToSubGroupVector] level0Idx[%u] level1Idx[%u] level2Idx[%u] size[%u]",
                    level0Idx, level1Idx, level2Idx, vectorInfo[level0Idx][level1Idx][level2Idx].size());
            }
        }
    }
    HCCL_INFO("[HcclCommAicpu][ParseTlvToSubGroupVector] success, group[%s]", identifier_.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::InitTopoInfo(const HcclOpResParam* commParam)
{
    topoInfo_.userRank = commParam->topoInfo.userRank;
    topoInfo_.userRankSize = commParam->topoInfo.userRankSize;
    topoInfo_.deviceLogicId = commParam->topoInfo.deviceLogicId;
    topoInfo_.isSingleMeshAggregation = commParam->topoInfo.isSingleMeshAggregation;
    topoInfo_.deviceNumPerAggregation = commParam->topoInfo.deviceNumPerAggregation;
    topoInfo_.superPodNum = commParam->topoInfo.superPodNum;
    topoInfo_.devicePhyId = commParam->topoInfo.devicePhyId;
    topoInfo_.deviceType = static_cast<DevType>(commParam->topoInfo.deviceType);
    topoInfo_.topoType = static_cast<TopoType>(commParam->topoInfo.topoType);
    topoInfo_.is310P3Common = (topoInfo_.deviceType == DevType::DEV_TYPE_310P3);
    topoInfo_.serverNum = commParam->topoInfo.serverNum;
    topoInfo_.meshAggregationRankSize = commParam->topoInfo.meshAggregationRankSize;
    topoInfo_.multiModuleDiffDeviceNumMode = commParam->topoInfo.multiModuleDiffDeviceNumMode;
    topoInfo_.multiSuperPodDiffServerNumMode = commParam->topoInfo.multiSuperPodDiffServerNumMode;
    topoInfo_.realUserRank = commParam->topoInfo.realUserRank;
    topoInfo_.isDiffDeviceModule = commParam->topoInfo.isDiffDeviceModule;
    topoInfo_.isDiffDeviceType = commParam->topoInfo.isDiffDeviceType;
    topoInfo_.gcdDeviceNumPerAggregation = commParam->topoInfo.gcdDeviceNumPerAggregation;
    topoInfo_.moduleNum = commParam->topoInfo.moduleNum;
    topoInfo_.useSuperPodMode = true;
    topoInfo_.isARSDoubleRing = commParam->isARSDoubleRing;
    topoInfo_.multiSuperPodDiffDeviceNumMode = commParam->multiSuperPodDiffDeviceNumMode;
    if (commParam->topoInfo.isUsedRdmaRankPairNum % KEY_VALUE_TO_VECTOR_MODULUS != 0) {
        HCCL_ERROR(
            "[HcclCommAicpu][InitTopoInfo]rdma rank pair number[%lu] is error.",
            commParam->topoInfo.isUsedRdmaRankPairNum);
        return HCCL_E_PARA;
    }
    u32* isUsedRdmaRankPairPtr = reinterpret_cast<u32*>(commParam->topoInfo.isUsedRdmaRankPair);
    if (isUsedRdmaRankPairPtr == nullptr) {
        HCCL_ERROR("[HcclCommAicpu][InitTopoInfo]rdma rank pair ptr is null.");
        return HCCL_E_PARA;
    }
    for (u64 i = 0; i < commParam->topoInfo.isUsedRdmaRankPairNum; i += KEY_VALUE_TO_VECTOR_MODULUS) {
        topoInfo_.isUsedRdmaMap.insert({isUsedRdmaRankPairPtr[i], static_cast<bool>(isUsedRdmaRankPairPtr[i + 1])});
    }

    if (commParam->topoInfo.pairLinkCounterNum % KEY_VALUE_TO_VECTOR_MODULUS != 0) {
        HCCL_ERROR(
            "[HcclCommAicpu][InitTopoInfo]pair link count number[%lu] is error.",
            commParam->topoInfo.pairLinkCounterNum);
        return HCCL_E_PARA;
    }
    u32* pairLinkCounterPtr = reinterpret_cast<u32*>(commParam->topoInfo.pairLinkCounter);
    if (pairLinkCounterPtr == nullptr) {
        HCCL_ERROR("[HcclCommAicpu][InitTopoInfo]pair link counter ptr is null.");
        return HCCL_E_PARA;
    }

    for (u64 i = 0; i < commParam->topoInfo.pairLinkCounterNum; i += KEY_VALUE_TO_VECTOR_MODULUS) {
        topoInfo_.pairLinkCounter.insert({pairLinkCounterPtr[i], pairLinkCounterPtr[i + 1]});
    }

    u32* nicListPtr = reinterpret_cast<u32*>(commParam->topoInfo.nicList);
    if (nicListPtr == nullptr) {
        HCCL_ERROR("[HcclCommAicpu][InitTopoInfo]nic list ptr is null.");
        return HCCL_E_PARA;
    }
    std::vector<u32> niclist{nicListPtr, nicListPtr + commParam->topoInfo.nicNum};
    topoInfo_.nicList = std::move(niclist);

    bool* bridgeRankPtr = reinterpret_cast<bool*>(commParam->topoInfo.bridgeRank);
    if (bridgeRankPtr != nullptr) {
        isBridgeVector_.resize(commParam->topoInfo.bridgeRankNum);
        for (u32 i = 0; i < commParam->topoInfo.bridgeRankNum; ++i) {
            isBridgeVector_[i] = bridgeRankPtr[i];
            HCCL_DEBUG("[HcclCommAicpu][InitTopoInfo] bridge rank info idx[%u] value[%u]", i, isBridgeVector_[i]);
        }
    } else {
        HCCL_RUN_INFO("[HcclCommAicpu][InitTopoInfo] bridge rank number is 0, group[%s]", identifier_.c_str());
    }

    HcclResult ret = HCCL_SUCCESS;
    ret = ParseTlvToVector(commParam->topoInfo.complanRank, commParam->topoInfo.complanRankLength, commPlaneVector_);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[HcclCommAicpu][InitTopoInfo]Init CommPlane error group[%s]", identifier_.c_str()), ret);

    ret = ParseTlvToVector(
        commParam->topoInfo.serverAndsuperPodRank, commParam->topoInfo.serverAndsuperPodRankLength,
        serverAndsuperPodToRank_);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[HcclCommAicpu][InitTopoInfo]Init server and superPod rank error group[%s]", identifier_.c_str()),
        ret);

    ret = ParseTlvToSubGroupVector(
        commParam->hierarchicalAlgInfo.commplaneSubGroupRank,
        commParam->hierarchicalAlgInfo.commplaneSubGroupRankLength, topoInfo_.CommPlaneSubGroupVector);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[HcclCommAicpu][InitTopoInfo]Init CommPlaneSubGroup error group[%s]", identifier_.c_str()), ret);

    HCCL_INFO(
        "[HcclCommAicpu][InitTopoInfo] success, group[%s], device Type[%u]", identifier_.c_str(),
        commParam->topoInfo.deviceType);
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::InitCclbuffer(const HcclOpResParam* commParam)
{
    if (commParam->localWindowsIn == 0 || commParam->localWindowsOut == 0 || cclInputBuffer_.ptr() != nullptr
        || cclOutputBuffer_.ptr() != nullptr) {
        HCCL_INFO(
            "[HcclCommAicpu][InitCclBuffer] don't need init cclbuffer "
            "ccl winIn[0x%lx] winout[0x%lx] cclin ptr[%p] cclout ptr[%p]",
            commParam->localWindowsIn, commParam->localWindowsOut, cclInputBuffer_.ptr(), cclOutputBuffer_.ptr());
        return HCCL_SUCCESS;
    }

    auto cclInPtr = reinterpret_cast<void*>(commParam->localWindowsIn);
    auto cclOutPtr = reinterpret_cast<void*>(commParam->localWindowsOut);
    cclInputBuffer_ = DeviceMem::create(cclInPtr, commParam->winSize);
    cclOutputBuffer_ = DeviceMem::create(cclOutPtr, commParam->winSize);
    cclbufferSize_ = commParam->winSize;
    HCCL_INFO(
        "[HcclCommAicpu][InitCclbuffer] success, group[%s], cclin[%llu], cclout[%llu], size[%lu]", identifier_.c_str(),
        commParam->localWindowsIn, commParam->localWindowsOut, commParam->winSize);
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::InitConfigInfo(const HcclOpResParam* commParam)
{
    deterministic_ = commParam->config.deterministic;
    interHccsDisable_ = commParam->config.interHccsDisable;
    multiQpThreshold_ = commParam->config.multiQpThreshold;
    inlineReducEnable_ = true;
    fftsEnable_ = false;
    taskMonitorInterval_ = commParam->config.taskMonitorInterval;
    algoInfo_.inlineReduceSwitchOn = true;
    algoInfo_.identifier = commParam->hcomId;
    algoInfo_.isSupportAtomicWrite = static_cast<bool>(commParam->config.isSupportAtomicWrite);
    notifySize_ = commParam->notifysize;
    slaveStreams_.reserve(LOCAL_STREAM_MAX_NUM);
    localNotifies_.reserve(LOCAL_NOTIFY_MAX_NUM);
    SetExternalInputDebugConfig(commParam->debugConfig);
    InitDebugConfigByValue(commParam->debugConfig);
    HCCL_INFO(
        "[HcclCommAicpu][Init]success, group[%s] reserve noipc notify[%lu], slave streams[%lu].", identifier_.c_str(),
        LOCAL_NOTIFY_MAX_NUM, LOCAL_STREAM_MAX_NUM);
    return HCCL_SUCCESS;
}

void HcclCommAicpu::SetDumpDebug(bool dumpDebug) { dumpDebug_ = dumpDebug; }

template <typename T>
HcclResult
HcclCommAicpu::InitAndVerifySignal(const HcclSignalInfo& signalInfo, std::vector<std::shared_ptr<T>>& notifyVec)
{
    if (signalInfo.resId == INVALID_U64) {
        // 无效值不做校验
        HCCL_INFO("[HcclCommAicpu][InitAndVerifySignal] resId is invalid, need not check");
        return HCCL_SUCCESS;
    }

    std::shared_ptr<T> notify;
    EXCEPTION_CATCH((notify = std::make_shared<T>()), return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(notify);
    HcclSignalInfo notifySignalInfo = signalInfo;
    CHK_RET(notify->Init(notifySignalInfo, NotifyLoadType::DEVICE_NOTIFY));
    notifyVec.push_back(notify);
    HCCL_INFO(
        "[HcclCommAicpu][InitAndVerifySignal] success group[%s], resId[%u], tsId:%d, devId[%u]", identifier_.c_str(),
        signalInfo.resId, signalInfo.tsId, signalInfo.devId);

    return HCCL_SUCCESS;
}

void HcclCommAicpu::HandleExistTagReAlloc(
    HccltagLocalResV2* tagRes, const std::string& tag, bool reAllocFlag, ListCommon*& curList, bool& needSkip)
{
    needSkip = false;
    // 如果 tag 已经存在且需要重新申请内存, 删除旧的资源与记录
    if (localTagResToObj_.find(tag) != localTagResToObj_.end() && reAllocFlag) {
        u64 currSize = tagRes->ScratchmemSize;
        u64 existSize = tagScratchMem_[tag]->size();
        // 新内存更大，删除旧记录，保留新的
        if (currSize > existSize) {
            localTagResToObj_.erase(tag);
            tagScratchMem_.erase(tag);
            HCCL_DEBUG(
                "[HcclCommAicpu][%s] tag exists, replace old small mem, tag[%s], old[%lu] new[%lu]", __func__,
                tag.c_str(), existSize, currSize);
        } else {
            // 旧内存更大，需要跳过
            HCCL_DEBUG(
                "[HcclCommAicpu][%s] tag exists, keep larger old mem, skip new, tag[%s], old[%lu] new[%lu]", __func__,
                tag.c_str(), existSize, currSize);
            curList = reinterpret_cast<ListCommon*>(curList->nextDevice);
            needSkip = true;
        }
    }
}

HcclResult HcclCommAicpu::InitLocalTagRes(const ListCommon& head, bool reAllocFlag)
{
    ListCommon* curList = reinterpret_cast<ListCommon*>(head.nextDevice);
    if (curList == nullptr) {
        HCCL_ERROR("[HcclCommAicpu][InitLocalTagRes]list ptr is null.");
        return HCCL_E_PARA;
    }
    while (curList != &head) {
        HccltagLocalResV2* tagRes = list_entry(curList, HccltagLocalResV2, nextTagRes);
        std::string tag = tagRes->tag;
        bool needSkip = false;
        HandleExistTagReAlloc(tagRes, tag, reAllocFlag, curList, needSkip);
        if (needSkip) {
            continue;
        }
        if (localTagResToObj_.find(tag) == localTagResToObj_.end()
            || localTagResToObj_[tag].find(tagRes->Scratchmem) == localTagResToObj_[tag].end()) {
            auto scratchMemPtr = reinterpret_cast<void*>(tagRes->Scratchmem);
            if (scratchMemPtr == nullptr) {
                HCCL_ERROR("[HcclCommAicpu][InitLocalTagRes]scratch mem ptr is null.");
                return HCCL_E_PARA;
            }
            DeviceMem loalScratchmem = DeviceMem::create(scratchMemPtr, tagRes->ScratchmemSize);
            std::shared_ptr<DeviceMem> loalScratchmemPtr;
            EXCEPTION_CATCH(
                (loalScratchmemPtr = std::make_shared<DeviceMem>(std::move(loalScratchmem))), return HCCL_E_PTR);
            CHK_SMART_PTR_NULL(loalScratchmemPtr);
            if (tagScratchMem_.find(tag) == tagScratchMem_.end()) {
                tagScratchMem_.insert({tag, loalScratchmemPtr});
            }
            std::unordered_set<u64> tmpTagRes;
            tmpTagRes.insert(tagRes->Scratchmem);
            localTagResToObj_[tag] = tmpTagRes;
            HCCL_DEBUG(
                "[HcclCommAicpu][InitLocalTagRes] parse remote resource, tag[%s], Scratchmem[%p], "
                "ScratchmemSize[%lu]",
                tag.c_str(), tagRes->Scratchmem, tagRes->ScratchmemSize);
        }
        curList = reinterpret_cast<ListCommon*>(curList->nextDevice);
        if (curList == nullptr) {
            HCCL_ERROR("[HcclCommAicpu][InitLocalTagRes] next list ptr is null.");
            return HCCL_E_PARA;
        }
    };
    HCCL_INFO("[HcclCommAicpu][InitLocalTagRes] success, group[%s]", identifier_.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::CheckNotifyOrQPMaxNum(u64& existNum, const u64& MaxNum, const bool& isNotifyRes)
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

HcclResult HcclCommAicpu::SetTransportMachinePara(
    MachinePara& machinePara, u32& rankId, const std::string& newTag, TransportLinkType linkType)
{
    machinePara.linkAttribute = 0x03; /* 0x03同时支持目的端和源端发起 */
    if (rankData_.find(rankId) == rankData_.end()) {
        HCCL_ERROR("[%s]there is no link with rankId[%u]", __func__, rankId);
        return HCCL_E_NOT_FOUND;
    }

    machinePara.localUserrank = localUserRank_;
    machinePara.remoteWorldRank = rankData_[rankId].remoteWorldRank;
    machinePara.remoteUserrank = rankData_[rankId].remoteUsrRankId;
    machinePara.deviceLogicId = topoInfo_.deviceLogicId;
    machinePara.localDeviceId = topoInfo_.devicePhyId;
    machinePara.deviceType = topoInfo_.deviceType;
    machinePara.tag = newTag;
    machinePara.isAicpuModeEn = true;
    if (linkType == TransportLinkType::RESERVED) {
        // 非910_93 2die sio与hccs并发场景，specifyLink设置为RESERVED_LINK_TYPE，平台层将按实际链路类型建链
        machinePara.specifyLink = LinkTypeInServer::RESERVED_LINK_TYPE;
    } else {
        // 910_93 2die sio与hccs并发场景，
        // 并发链路中的的hccs链路specifyLink设置为HCCS_SW_TYPE，平台层将使用hccs链路来建链；
        // 并发链路中的的sio链路specifyLink设置为SIO_TYPE，平台层将使用sio链路来建链
        machinePara.specifyLink
            = (linkType == TransportLinkType::SIO) ? LinkTypeInServer::SIO_TYPE : LinkTypeInServer::HCCS_SW_TYPE;
    }

    HCCL_INFO(
        "%s success, group[%s], rankId[%u], linkAttribute[%x], localUserRank[%u], remoteWorldRank[%u], "
        "remoteUserrank[%u], deviceLogicId[%d], localDeviceId[%d], deviceType[%d], newTag[%s], specifyLink[%d], "
        "isAicpuModeEn[%d]",
        __func__, identifier_.c_str(), rankId, machinePara.linkAttribute, machinePara.localUserrank,
        machinePara.remoteWorldRank, machinePara.remoteUserrank, machinePara.deviceLogicId, machinePara.localDeviceId,
        machinePara.deviceType, machinePara.tag.c_str(), machinePara.specifyLink, machinePara.isAicpuModeEn);
    return HCCL_SUCCESS;
}

template <typename T>
HcclResult HcclCommAicpu::InitAndVerifySingleSignal(const HcclSignalInfo& signalInfo, std::shared_ptr<T>& notify)
{
    if (signalInfo.resId == INVALID_U64) {
        // 无效值不做校验
        HCCL_DEBUG("[%s]resId[%llu] is invalid, need not check", __func__, signalInfo.resId);
        return HCCL_SUCCESS;
    }
    HcclSignalInfo tmpSignalInfo = signalInfo;

    EXCEPTION_CATCH((notify = std::make_shared<T>()), return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(notify);
    CHK_RET(notify->Init(tmpSignalInfo, NotifyLoadType::DEVICE_NOTIFY));
    CHK_RET(notify->GetNotifyData(tmpSignalInfo));
    HCCL_DEBUG(
        "[%s] success group[%s], resId[%llu], tsId:%d, devId[%u]", __func__, identifier_.c_str(), tmpSignalInfo.resId,
        tmpSignalInfo.tsId, tmpSignalInfo.devId);

    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::SetTagRemoteRes(u32& rankId, const std::string& tag, HccltagRemoteResV2* tagRes)
{
    if (rankTagRemoteRes_.find(rankId) == rankTagRemoteRes_.end()
        || rankTagRemoteRes_[rankId].find(tag) == rankTagRemoteRes_[rankId].end()) {
        HccltagRemoteResV3 tempTagRemoteRes;
        tempTagRemoteRes.tagRemoteResPtr = tagRes;
        rankTagRemoteRes_[rankId][tag] = tempTagRemoteRes;
    }
    HCCL_DEBUG("[%s]get TagRemoteRes success, rankId[%u], tag[%s]", __func__, rankId, tag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::SetTransportPtpNotify(
    TransportDeviceP2pData& transDevP2pData, u64& p2pNotifyNum, HcclLinkP2pV2& linkP2p, u32 notifyNum)
{
    u64 actualNotifyNum = 0;
    // 获取Ipc notify信息
    CHK_RET(CheckNotifyOrQPMaxNum(actualNotifyNum, LINK_P2P_MAX_NUM, true));
    std::shared_ptr<LocalNotify> ipcPreWaitNotify = std::make_shared<LocalNotify>();
    CHK_RET(InitAndVerifySingleSignal(linkP2p.localIpcSignal[actualNotifyNum], ipcPreWaitNotify));
    transDevP2pData.ipcPreWaitNotify = ipcPreWaitNotify;

    std::shared_ptr<RemoteNotify> ipcPreRecordNotify = std::make_shared<RemoteNotify>();
    CHK_RET(InitAndVerifySingleSignal(linkP2p.remoteIpcSignal[actualNotifyNum], ipcPreRecordNotify));
    transDevP2pData.ipcPreRecordNotify = ipcPreRecordNotify;
    actualNotifyNum++;

    CHK_RET(CheckNotifyOrQPMaxNum(actualNotifyNum, LINK_P2P_MAX_NUM, true));
    std::shared_ptr<LocalNotify> ipcPostWaitNotify = std::make_shared<LocalNotify>();
    CHK_RET(InitAndVerifySingleSignal(linkP2p.localIpcSignal[actualNotifyNum], ipcPostWaitNotify));
    transDevP2pData.ipcPostWaitNotify = ipcPostWaitNotify;

    std::shared_ptr<RemoteNotify> ipcPostRecordNotify = std::make_shared<RemoteNotify>();
    CHK_RET(InitAndVerifySingleSignal(linkP2p.remoteIpcSignal[actualNotifyNum], ipcPostRecordNotify));
    transDevP2pData.ipcPostRecordNotify = ipcPostRecordNotify;
    actualNotifyNum++;

    transDevP2pData.userLocalNotify.resize(notifyNum, nullptr);
    transDevP2pData.userRemoteNotify.resize(notifyNum, nullptr);

    for (u32 idx = 0; idx < notifyNum; idx++) {
        CHK_RET(CheckNotifyOrQPMaxNum(actualNotifyNum, LINK_P2P_MAX_NUM, true));
        std::shared_ptr<LocalNotify> ipcWaitNotify = std::make_shared<LocalNotify>();
        CHK_RET(InitAndVerifySingleSignal(linkP2p.localIpcSignal[actualNotifyNum], ipcWaitNotify));
        transDevP2pData.userLocalNotify[idx] = ipcWaitNotify;

        std::shared_ptr<RemoteNotify> ipcRecordNotify = std::make_shared<RemoteNotify>();
        CHK_RET(InitAndVerifySingleSignal(linkP2p.remoteIpcSignal[actualNotifyNum], ipcRecordNotify));
        transDevP2pData.userRemoteNotify[idx] = ipcRecordNotify;

        actualNotifyNum++;
    }

    HCCL_DEBUG("[%s]get p2pNotify success, actualNotifyNum[%u]", __func__, actualNotifyNum);
    p2pNotifyNum = actualNotifyNum;

    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::SetTransportRoceQP(
    TransportDeviceIbverbsData& transDevIbverbsData, u64& roceQpNum, HcclLinkRoceV2* linkRoce)
{
    roceQpNum = linkRoce->qpsPerConnection;
    u32 roceQpNumSum = linkRoce->qpsPerConnection + static_cast<u32>(linkRoce->qpsPerConnection != 1);
    transDevIbverbsData.qpInfo.resize(roceQpNumSum);
    std::copy_n(linkRoce->QpInfo, roceQpNumSum, transDevIbverbsData.qpInfo.begin());
    transDevIbverbsData.multiQpThreshold = multiQpThreshold_;
    transDevIbverbsData.qpsPerConnection = linkRoce->qpsPerConnection;
    HCCL_INFO(
        "[%s]transDevIbverbsData.qpInfo.qpPtr[%llu], transDevIbverbsData.qpInfo.sqIndex[%u], "
        "transDevIbverbsData.qpInfo.dbIndex[%u], roceQpNum[%llu], roceQpNumSum[%u], multiQpThreshold_[%u]",
        __func__, transDevIbverbsData.qpInfo[0].qpPtr, transDevIbverbsData.qpInfo[0].sqIndex,
        transDevIbverbsData.qpInfo[0].dbIndex, roceQpNum, roceQpNumSum, multiQpThreshold_);
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::SetTransportRoceNotify(
    TransportDeviceIbverbsData& transDevIbverbsData, u64& roceNotifyNum, HcclLinkRoceV2* linkRoce,
    [[maybe_unused]] u32 notifyNum)
{
    u64 actualNotifyNum = 0;
    if (linkRoce->localNotifyList == 0 || linkRoce->remoteNotifyList == 0) {
        HCCL_DEBUG("[%s] Empty local and remote notify lists, skipping notify resource creation.", __func__);
        return HCCL_SUCCESS;
    }
    HcclSignalInfo* localNotifyList = reinterpret_cast<HcclSignalInfo*>(linkRoce->localNotifyList);
    AddrKey* remoteNotifyList = reinterpret_cast<AddrKey*>(linkRoce->remoteNotifyList);
    if (localNotifyList == nullptr || remoteNotifyList == nullptr) {
        HCCL_ERROR("[%s]nullptr found in localNotifyList or remoteNotifyList from device mem, check.", __func__);
        return HCCL_E_INTERNAL;
    }
    // 获取RDMA Notify信息
    std::shared_ptr<LocalNotify> ackNotify = std::make_shared<LocalNotify>();
    CHK_RET(InitAndVerifySingleSignal(localNotifyList[actualNotifyNum], ackNotify));
    transDevIbverbsData.ackNotify = ackNotify;
    transDevIbverbsData.remoteAckNotifyDetails = remoteNotifyList[actualNotifyNum];
    actualNotifyNum++;

    std::shared_ptr<LocalNotify> dataNotify = std::make_shared<LocalNotify>();
    CHK_RET(InitAndVerifySingleSignal(localNotifyList[actualNotifyNum], dataNotify));
    transDevIbverbsData.dataNotify = dataNotify;
    transDevIbverbsData.remoteDataNotifyDetails = remoteNotifyList[actualNotifyNum];
    actualNotifyNum++;

    std::shared_ptr<LocalNotify> dataAckNotify = std::make_shared<LocalNotify>();
    CHK_RET(InitAndVerifySingleSignal(localNotifyList[actualNotifyNum], dataAckNotify));
    transDevIbverbsData.dataAckNotify = dataAckNotify;
    transDevIbverbsData.remoteDataAckNotifyDetails = remoteNotifyList[actualNotifyNum];
    transDevIbverbsData.notifySize = notifySize_;
    actualNotifyNum++;

    transDevIbverbsData.userLocalNotify.resize(linkRoce->qpsPerConnection);
    transDevIbverbsData.userRemoteNotifyDetails.resize(linkRoce->qpsPerConnection);
    // 当前多QP下每个QP会多申请一个DataNotify
    u64 singleQpNotifySize = linkRoce->singleQPNotifyNum + static_cast<u32>(linkRoce->qpsPerConnection > 1);
    for (u32 qpIndex = 0; qpIndex < linkRoce->qpsPerConnection; qpIndex++) {
        transDevIbverbsData.userLocalNotify[qpIndex].resize(singleQpNotifySize, nullptr);
        transDevIbverbsData.userRemoteNotifyDetails[qpIndex].resize(singleQpNotifySize);
        for (u32 i = 0, idx = actualNotifyNum + singleQpNotifySize * qpIndex; i < singleQpNotifySize; ++idx, ++i) {
            std::shared_ptr<LocalNotify> locNotify = std::make_shared<LocalNotify>();
            CHK_RET(InitAndVerifySingleSignal(localNotifyList[idx], locNotify));
            transDevIbverbsData.userLocalNotify[qpIndex][i] = locNotify;
            transDevIbverbsData.userRemoteNotifyDetails[qpIndex][i] = remoteNotifyList[idx];
        }
    }
    roceNotifyNum = linkRoce->singleQPNotifyNum;
    HCCL_DEBUG("[%s]get roceNotify success, roceNotifyNum[%u]", __func__, roceNotifyNum);
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::InitLinkP2p(
    HccltagRemoteResV2* tagRes, u32& rankId, const std::string& newTag, u32 notifyNum, TransportLinkType linkType)
{
    std::unordered_map<u32, std::unordered_map<std::string, std::shared_ptr<Transport>>>& linkRes
        = (linkType == TransportLinkType::SIO) ? linkResSio_ : linkRes_;
    HcclLinkP2pV2& linkP2p = (linkType == TransportLinkType::SIO) ? tagRes->linkP2pSio : tagRes->linkP2p;

    if (linkRes.find(rankId) == linkRes.end() || linkRes[rankId].find(newTag) == linkRes[rankId].end()) {
        // 优先校验notify去判定link是否有效
        if (linkP2p.localIpcSignal[0].resId == INVALID_U64) {
            HCCL_INFO(
                "[%s]the link is invalid, no need to create transport, rankId[%u], newTag[%s]", __func__, rankId,
                newTag.c_str());
            return HCCL_SUCCESS;
        }
        // 创建Transport对象
        MachinePara machinePara;
        CHK_RET(SetTransportMachinePara(machinePara, rankId, newTag, linkType));
        machinePara.notifyNum = notifyNum;
        // 获取localMem & remoteMem
        TransportDeviceP2pData transDevP2pData;
        transDevP2pData.inputBufferPtr = reinterpret_cast<void*>((linkP2p.remoteMem)[INPUT].addr);
        transDevP2pData.outputBufferPtr = reinterpret_cast<void*>((linkP2p.remoteMem)[OUTPUT].addr);
        if (transDevP2pData.inputBufferPtr == nullptr || transDevP2pData.outputBufferPtr == nullptr) {
            HCCL_ERROR(
                "[%s]input ptr[%p] or output ptr[%p] is null.", __func__, transDevP2pData.inputBufferPtr,
                transDevP2pData.outputBufferPtr);
            return HCCL_E_PARA;
        }
        // 获取Notify资源
        CHK_RET(SetTagRemoteRes(rankId, newTag, tagRes));
        HccltagRemoteResV3* tagRemoteRes = &(rankTagRemoteRes_[rankId][newTag]);
        CHK_RET(SetTransportPtpNotify(transDevP2pData, tagRemoteRes->p2pNotifyNum, linkP2p, notifyNum));
        //  获取transportAttr信息
        transDevP2pData.transportAttr = linkP2p.transportAttr;
        //  创建Transport对象
        std::shared_ptr<Transport> link;
        TransportPara para{};
        const std::unique_ptr<NotifyPool> notifyPool;
        link.reset(new (std::nothrow) Transport(
            TransportType::TRANS_TYPE_DEVICE_P2P, para, dispatcher_, notifyPool, machinePara, transDevP2pData));
        CHK_SMART_PTR_NULL(link);
        CHK_RET(link->Init());
        linkRes[rankId][newTag] = link;
        HCCL_INFO("[%s]linkRes_, rankId[%u], newTag[%s]", __func__, rankId, newTag.c_str());
    }
    return HCCL_SUCCESS;
}
HcclResult HcclCommAicpu::InitLinkRoce(
    HccltagRemoteResV2* tagRes, HcclLinkRoceV2* linkRoce, u32& rankId, const std::string& newTag, u32 notifyNum,
    const bool isBackup, const bool isSecond)
{
    // 优先校验notify去判定link是否有效
    if (linkRoce->localNotifyList == 0) {
        HCCL_INFO(
            "[%s]the link is invalid, no need to create transport, rankId[%u], newTag[%s], isBackup[%d]", __func__,
            rankId, newTag.c_str(), isBackup);
        return HCCL_SUCCESS;
    }
    HcclSignalInfo* localNotifyList = reinterpret_cast<HcclSignalInfo*>(linkRoce->localNotifyList);
    if (localNotifyList[0].resId == INVALID_U64) {
        HCCL_INFO(
            "[%s]the link notify resource is invalid, no need to create transport, rankId[%u], newTag[%s], "
            "resId[%llu], "
            "isBackup[%d]",
            __func__, rankId, newTag.c_str(), localNotifyList[0].resId, isBackup);
        return HCCL_SUCCESS;
    }

    // 创建Transport对象
    MachinePara machinePara;
    CHK_RET(SetTransportMachinePara(machinePara, rankId, newTag));
    machinePara.notifyNum = notifyNum;
    // 获取localMem & remoteMem
    TransportDeviceIbverbsData transDevIbverbsData;
    transDevIbverbsData.inputBufferPtr = reinterpret_cast<void*>((linkRoce->remoteMem)[INPUT].addr);
    transDevIbverbsData.outputBufferPtr = reinterpret_cast<void*>((linkRoce->remoteMem)[OUTPUT].addr);
    if (transDevIbverbsData.inputBufferPtr == nullptr || transDevIbverbsData.outputBufferPtr == nullptr) {
        HCCL_ERROR(
            "[%s]input ptr[%p] or output ptr[%p] is null.", __func__, transDevIbverbsData.inputBufferPtr,
            transDevIbverbsData.outputBufferPtr);
        return HCCL_E_PARA;
    }
    transDevIbverbsData.localInputMem = (linkRoce->localMem)[INPUT];
    transDevIbverbsData.localOutputMem = (linkRoce->localMem)[OUTPUT];
    transDevIbverbsData.localNotifyValueAddr = linkRoce->notifyValue;
    transDevIbverbsData.notifyValueKey = linkRoce->notifyValueKey;
    transDevIbverbsData.remoteInputKey = (linkRoce->remoteMem)[INPUT].key;
    transDevIbverbsData.remoteOutputKey = (linkRoce->remoteMem)[OUTPUT].key;
    // 获取QPinfo
    CHK_RET(SetTagRemoteRes(rankId, newTag, tagRes));
    HccltagRemoteResV3* tagRemoteRes = &(rankTagRemoteRes_[rankId][newTag]);
    u64& roceQpNum = isBackup ? tagRemoteRes->qpNumBackup : tagRemoteRes->qpNum;
    CHK_RET(SetTransportRoceQP(transDevIbverbsData, roceQpNum, linkRoce));
    // 获取notify
    u64& roceNotifyNum = isBackup ? tagRemoteRes->roceNotifyNumBackup : tagRemoteRes->roceNotifyNum;
    CHK_RET(SetTransportRoceNotify(transDevIbverbsData, roceNotifyNum, linkRoce, notifyNum));
    HCCL_INFO("[%s]transDevIbverbsData isBackup[%d]", __func__, isBackup);
    // 获取atomic write
    transDevIbverbsData.useAtomicWrite = linkRoce->useAtomicWrite;
    // 创建Transport对象
    std::shared_ptr<Transport> link;
    TransportPara para{};
    para.timeout = linkTimeOut_;
    const std::unique_ptr<NotifyPool> notifyPool;
    link.reset(new (std::nothrow) Transport(
        TransportType::TRANS_TYPE_DEVICE_IBVERBS, para, dispatcher_, notifyPool, machinePara, TransportDeviceP2pData(),
        transDevIbverbsData));
    CHK_SMART_PTR_NULL(link);
    CHK_RET(link->Init());
    if (isBackup) {
        linkRdmaResBackUp_[rankId][newTag].push_back(link);
        HCCL_INFO(
            "[%s]linkRdmaResBackUp_, rankId[%u], newTag[%s], isBackup[%d], isSecond[%d], qpNum[%u], notifyNum[%u]",
            __func__, rankId, newTag.c_str(), isBackup, isSecond, roceQpNum, roceNotifyNum);
    } else {
        linkRdmaRes_[rankId][newTag].push_back(link);
        HCCL_INFO(
            "[%s]linkRdmaRes_, rankId[%u], newTag[%s], isBackup[%d], isSecond[%d], qpNum[%u], notifyNum[%u]", __func__,
            rankId, newTag.c_str(), isBackup, isSecond, roceQpNum, roceNotifyNum);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::InitLinkRoce(
    HccltagRemoteResV2* tagRes, u32& rankId, const std::string& newTag, u32 notifyNum, const bool isBackup)
{
    auto tempLinkRes = isBackup ? linkRdmaResBackUp_ : linkRdmaRes_;
    if (tempLinkRes.find(rankId) == tempLinkRes.end()
        || tempLinkRes[rankId].find(newTag) == tempLinkRes[rankId].end()) {
        bool isBatchSendRecv = newTag.find("BatchSendRecv") != std::string::npos;
        if (isBatchSendRecv) {
            // 如果是batchsendrecv，相同rank，需要刷新两次transport，如果是主的话就刷新0,2 备就刷新1,3
            if (isBackup) {
                CHK_RET(InitLinkRoce(
                    tagRes, &(tagRes->linkRoce[AICPU_RETRY_LINKROCE_BACKUP]), rankId, newTag, notifyNum, isBackup));
                CHK_RET(InitLinkRoce(
                    tagRes, &(tagRes->linkRoce[AICPU_RETRY_LINKROCE_BACKUP + 2]), rankId, newTag, notifyNum, isBackup,
                    true));
            } else {
                CHK_RET(InitLinkRoce(
                    tagRes, &(tagRes->linkRoce[AICPU_RETRY_LINKROCE_DEFAULT]), rankId, newTag, notifyNum, isBackup));
                CHK_RET(InitLinkRoce(
                    tagRes, &(tagRes->linkRoce[AICPU_RETRY_LINKROCE_DEFAULT + 2]), rankId, newTag, notifyNum, isBackup,
                    true));
            }
        } else {
            // 非batchsendrecv只刷新主0，备1
            HcclLinkRoceV2* linkRoce = isBackup ? &(tagRes->linkRoce[AICPU_RETRY_LINKROCE_BACKUP]) :
                                                  &(tagRes->linkRoce[AICPU_RETRY_LINKROCE_DEFAULT]);
            CHK_RET(InitLinkRoce(tagRes, linkRoce, rankId, newTag, notifyNum, isBackup));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::InitRemoteTagRes(
    u32& rankId, const ListCommon& head, const std::string& newTag, u32 notifyNum, TransportLinkType linkType)
{
    HCCL_RUN_INFO(
        "[%s] Entry parse remote resource rankId[%u], group[%s], newTag[%s], head[%p], "
        "linkType[%d]",
        __func__, rankId, identifier_.c_str(), newTag.c_str(), &head, linkType);
    ListCommon* curList = reinterpret_cast<ListCommon*>(head.nextDevice);
    if (curList == nullptr) {
        HCCL_ERROR("[%s]cur list ptr is null.", __func__);
        return HCCL_E_PARA;
    }

    HccltagRemoteResV2* tagRes = nullptr;
    while (curList != &head) {
        HccltagRemoteResV2* tagResTemp = list_entry(curList, HccltagRemoteResV2, nextTagRes);
        if (strcmp(tagResTemp->tag, newTag.c_str()) == 0) {
            tagRes = tagResTemp;
            break;
        }

        curList = reinterpret_cast<ListCommon*>(curList->nextDevice);
        if (curList == nullptr) {
            HCCL_ERROR("[%s]next list ptr is null.", __func__);
            return HCCL_E_PARA;
        }
    }

    if (tagRes == nullptr) {
        HCCL_ERROR(
            "[%s]newTag[%s] not found, rankId[%u], head[%p], curList[%p], nextList[%llu], notifyNum[%u], "
            "linkType[%d]",
            __func__, newTag.c_str(), rankId, &head, curList, curList->nextDevice, notifyNum, linkType);
        return HCCL_E_PARA;
    }

    HCCL_INFO(
        "[%s]newTag[%s], rankId[%u], head[%p], curList[%p], nextList[%llu], notifyNum[%u], linkType[%d]", __func__,
        newTag.c_str(), rankId, &head, curList, curList->nextDevice, notifyNum, linkType);
    if (linkType != TransportLinkType::RDMA) {
        // 创建P2P链路
        CHK_RET(InitLinkP2p(tagRes, rankId, newTag, notifyNum, linkType));
    } else {
        // 创建roce链路
        CHK_RET(InitLinkRoce(tagRes, rankId, newTag, notifyNum));
        // 创建roce链路（备用链路）
        CHK_RET(InitLinkRoce(tagRes, rankId, newTag, notifyNum, true));
    }
    HCCL_INFO(
        "[%s] End parse remote resource rankId[%u] tag[%s], newTag[%s]", __func__, rankId, identifier_.c_str(),
        newTag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::RefreshTransportsResForRank(
    const HcclOpResParam* commParam, u32 rankId, const std::string& newTag, u32 notifyNum, TransportLinkType linkType)
{
    if (rankId >= AICPU_MAX_RANK_NUM) {
        HCCL_ERROR(
            "[%s] rankId[%u] overflow for group[%s], newTag[%s]", __func__, rankId, identifier_.c_str(),
            newTag.c_str());
        return HCCL_E_PARA;
    }
    if (commParam->remoteRes[rankId].nextDevicePtr == 0) {
        return HCCL_SUCCESS;
    }
    HcclRankRelationResV2* rankRelationResPtr
        = reinterpret_cast<HcclRankRelationResV2*>(commParam->remoteRes[rankId].nextDevicePtr);
    if (rankRelationResPtr == nullptr) {
        HCCL_ERROR(
            "[%s]rank relation resource ptr is null, commParam->remoteRes[rankId].nextDevicePtr[%p],"
            " rankId[%u]",
            __func__, reinterpret_cast<HcclRankRelationResV2*>(commParam->remoteRes[rankId].nextDevicePtr), rankId);
        return HCCL_E_PARA;
    }

    // 1. init公共参数（对应remoteWorldRank，remoteUsrRankId暂不处理：windowsIn，windowsOut）
    rankData_[rankId].remoteWorldRank = rankRelationResPtr->remoteWorldRank;
    rankData_[rankId].remoteUsrRankId = rankRelationResPtr->remoteUsrRankId;
    // 2. 遍历链表，获取HccltagRemoteResV2创建Transport对象
    if (reinterpret_cast<ListCommon*>(rankRelationResPtr->nextTagRes.nextDevice) != &(rankRelationResPtr->nextTagRes)) {
        HCCL_DEBUG(
            "[%s] Start to parse rankId[%u] tag resources, head[%p], nextDevice[%p], pre Device[%p], group[%s]",
            __func__, rankId, &rankRelationResPtr->nextTagRes, rankRelationResPtr->nextTagRes.nextDevice,
            rankRelationResPtr->nextTagRes.preDevice, identifier_.c_str());
        CHK_RET(InitRemoteTagRes(rankId, rankRelationResPtr->nextTagRes, newTag, notifyNum, linkType));
    } else {
        HCCL_ERROR(
            "[%s]could not find member in rankRelationRes list, rankId[%u], head[%p], nextDevice[%p]", __func__, rankId,
            &rankRelationResPtr->nextTagRes, rankRelationResPtr->nextTagRes.nextDevice);
        return HCCL_E_PARA;
    }
    HCCL_INFO(
        "[%s] process success rankId[%u], group[%s], newTag[%s]", __func__, rankId, identifier_.c_str(),
        newTag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::GetRdmaLinksByRankAndTag(
    const HcclOpResParam* commParam, CommTransportsType type, u32 rankId, const std::string& newTag, LINK& link,
    bool isBackup, u32 notifyNum, bool isSecond)
{
    HCCL_INFO(
        "[%s] Start to rdma get Link group[%s], rankId[%u], newTag[%s], isBackup[%d], deviceLogicId[%d],"
        "notifyNum[%u], isSecond[%d]",
        __func__, identifier_.c_str(), rankId, newTag.c_str(), isBackup, commParam->topoInfo.deviceLogicId, notifyNum,
        isSecond);

    auto* linkRes = isBackup ? &linkRdmaResBackUp_ : &linkRdmaRes_;
    auto iterRankLinks = linkRes->find(rankId);
    if (iterRankLinks == linkRes->end() || iterRankLinks->second.find(newTag) == iterRankLinks->second.end()) {
        HCCL_INFO(
            "[%s] could not find link resource, rankId[%u], group[%s], newTag[%s]", __func__, rankId,
            identifier_.c_str(), newTag.c_str());
        CHK_RET(RefreshTransportsResForRank(commParam, rankId, newTag, notifyNum, TransportLinkType::RDMA));
        iterRankLinks = linkRes->find(rankId);
        if (iterRankLinks == linkRes->end() || iterRankLinks->second.find(newTag) == iterRankLinks->second.end()) {
            HCCL_ERROR(
                "[%s] refresh transport failed, newTag[%s], remoteUserRankId[%u]", __func__, newTag.c_str(), rankId);
            return HCCL_E_INTERNAL;
        }
    }
    if (isSecond && iterRankLinks->second[newTag].size() <= 1) {
        HCCL_ERROR("[%s] get rdma Link failed, newTag[%s], remoteUserRankId[%u]", __func__, newTag.c_str(), rankId);
        return HCCL_E_INTERNAL;
    }

    link = isSecond ? iterRankLinks->second[newTag][1] : iterRankLinks->second[newTag][0];

    if (receivedAcks_.find(rankId) == receivedAcks_.end()) {
        HCCL_ERROR("[%s]there is no link with rankId[%u]", __func__, rankId);
        return HCCL_E_NOT_FOUND;
    }
    link->SetSupportDataReceivedAck(receivedAcks_[rankId]);
    HCCL_DEBUG(
        "[HcclCommAicpu][GetLinksByRankAndTag]rankid[%d] supportDataReceivedAck is %d", rankId, receivedAcks_[rankId]);
    HCCL_INFO(
        "[%s] group[%s], newTag[%s], rankId[%u] type[%u] success!", __func__, identifier_.c_str(), newTag.c_str(),
        rankId, type);
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::GetSdmaLinksByRankAndTag(
    const HcclOpResParam* commParam, CommTransportsType type, u32 rankId, const std::string& newTag, LINK& link,
    bool isBackup, u32 notifyNum, TransportLinkType linkType)
{
    HCCL_INFO(
        "[%s] Start to get sdma Link group[%s], rankId[%u], newTag[%s], isBackup[%d], deviceLogicId[%d], "
        "notifyNum[%u], linkType[%d]",
        __func__, identifier_.c_str(), rankId, newTag.c_str(), isBackup, commParam->topoInfo.deviceLogicId, notifyNum,
        linkType);

    auto* linkRes = &linkRes_;
    if (linkType == TransportLinkType::SIO) { // HCCS SIO并发场景，HCCS链路与SIO链路分开管理
        linkRes = &linkResSio_;
    }
    auto iterRankLinks = linkRes->find(rankId);
    if (iterRankLinks == linkRes->end() || iterRankLinks->second.find(newTag) == iterRankLinks->second.end()) {
        HCCL_INFO(
            "[%s] could not find link resource, rankId[%u], group[%s], newTag[%s]", __func__, rankId,
            identifier_.c_str(), newTag.c_str());
        CHK_RET(RefreshTransportsResForRank(commParam, rankId, newTag, notifyNum, linkType));
        iterRankLinks = linkRes->find(rankId);
        if (iterRankLinks == linkRes->end() || iterRankLinks->second.find(newTag) == iterRankLinks->second.end()) {
            HCCL_ERROR(
                "[%s] refresh transport failed, newTag[%s], remoteUserRankId[%u], %p", __func__, newTag.c_str(), rankId,
                linkRes);
            return HCCL_E_INTERNAL;
        }
    }

    link = iterRankLinks->second[newTag];
    if (receivedAcks_.find(rankId) == receivedAcks_.end()) {
        HCCL_ERROR("[%s]there is no link with rankId[%u]", __func__, rankId);
        return HCCL_E_NOT_FOUND;
    }
    link->SetSupportDataReceivedAck(receivedAcks_[rankId]);
    HCCL_DEBUG(
        "[HcclCommAicpu][GetLinksByRankAndTag]rankid[%d] supportDataReceivedAck is %d", rankId, receivedAcks_[rankId]);
    HCCL_INFO(
        "[%s] group[%s], newTag[%s], rankId[%u] type[%u] success!", __func__, identifier_.c_str(), newTag.c_str(),
        rankId, type);
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::CleanRoceResource(
    const std::string& newTag, AlgResourceResponse& algResResponse,
    [[maybe_unused]] const std::map<u32, bool>& remoteRankPortMap, [[maybe_unused]] const OpParam& param)
{
    HCCL_INFO("[%s] Entry alloc transport group[%s], tag[%s]", __func__, identifier_.c_str(), newTag.c_str());

    for (auto& levelNSubCommTransport : algResResponse.opTransportResponse) {
        for (auto& singleSubCommTransport : levelNSubCommTransport) {
            for (auto& transportRequest : singleSubCommTransport.transportRequests) {
                if (transportRequest.isValid && transportRequest.isUsedRdma) {
                    u32 remoteUserRank = transportRequest.remoteUserRank;
                    linkRdmaRes_[remoteUserRank].erase(newTag);
                    linkRdmaResBackUp_[remoteUserRank].erase(newTag);
                }
            }
        }
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::CleanAllRoceResource()
{
    HCCL_INFO("Clean all link rdna resources");
    // 清空主链路资源
    linkRdmaRes_.clear();
    // 清空备链路资源
    linkRdmaResBackUp_.clear();
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::ClearOpResource(const std::string& tag)
{
    // 与 GetAlgResponseRes/AllocAlgResource/ReAllocTransportResource 互斥，避免读写竞争 resMap_/linkRes_
    std::lock_guard<std::mutex> lock(preemptMutexForResMap_);

    auto eraseTagFromAll = [this](const std::string& t) -> bool {
        bool hit = false;
        // 主资源map：vector<LINK>析构会让shared_ptr<Transport>引用-1，归零时触发driver资源释放
        if (resMap_.erase(t) > 0) {
            hit = true;
        }
        // SDMA HCCS链路缓存：报错日志中driver halResAddrMap ioctl失败的根本累积点
        for (auto& rankIt : linkRes_) {
            if (rankIt.second.erase(t) > 0) {
                hit = true;
            }
        }
        // SDMA SIO并发链路缓存
        for (auto& rankIt : linkResSio_) {
            if (rankIt.second.erase(t) > 0) {
                hit = true;
            }
        }
        // RDMA主备链路缓存（CleanAllRoceResource按通信域全清，本接口按tag细粒度清）
        for (auto& rankIt : linkRdmaRes_) {
            if (rankIt.second.erase(t) > 0) {
                hit = true;
            }
        }
        for (auto& rankIt : linkRdmaResBackUp_) {
            if (rankIt.second.erase(t) > 0) {
                hit = true;
            }
        }
        // 本地scratchMem和localTagResToObj按tag索引
        if (tagScratchMem_.erase(t) > 0) {
            hit = true;
        }
        if (localTagResToObj_.erase(t) > 0) {
            hit = true;
        }
        for (auto& rankIt : rankTagRemoteRes_) {
            if (rankIt.second.erase(t) > 0) {
                hit = true;
            }
        }
        return hit;
    };

    // 与 host 端 ClearOpResource 对齐：尝试 base/_host/_device 三种后缀
    bool found = false;
    found = eraseTagFromAll(tag) || found;
    found = eraseTagFromAll(tag + "_host") || found;
    found = eraseTagFromAll(tag + "_device") || found;
    if (!found) {
        HCCL_WARNING(
            "[HcclCommAicpu][ClearOpResource] no entries matched tag[%s] "
            "(also tried _host/_device variants), group[%s]",
            tag.c_str(), identifier_.c_str());
    } else {
        HCCL_INFO(
            "[HcclCommAicpu][ClearOpResource] cleared resources for tag[%s], group[%s]", tag.c_str(),
            identifier_.c_str());
    }
    return HCCL_SUCCESS;
}

// 借轨重新刷新资源
HcclResult HcclCommAicpu::ReAllocTransportResource(
    const std::string& newTag, AlgResourceResponse& algResResponse, std::map<u32, bool>& remoteRankPortMap,
    const HcclOpResParam* commParam, const OpParam& param)
{
    HCCL_INFO("[%s] Entry alloc transport group[%s], tag[%s]", __func__, identifier_.c_str(), newTag.c_str());
    std::set<u32> bsrTansportRank;
    for (auto& levelNSubCommTransport : algResResponse.opTransportResponse) {
        for (auto& singleSubCommTransport : levelNSubCommTransport) {
            singleSubCommTransport.links.clear();
            singleSubCommTransport.links.reserve(singleSubCommTransport.transportRequests.size());
            for (auto& transportRequest : singleSubCommTransport.transportRequests) {
                singleSubCommTransport.links.push_back(nullptr);
                if (transportRequest.isValid) {
                    HCCL_INFO(
                        "[%s] alloc transport, newTag[%s], rankId[%u], "
                        "input memory type[%u], output memory type[%u], ",
                        __func__, newTag.c_str(), transportRequest.remoteUserRank, transportRequest.inputMemType,
                        transportRequest.outputMemType);
                    receivedAcks_[transportRequest.remoteUserRank] = singleSubCommTransport.supportDataReceivedAck;
                    bool isBackup = remoteRankPortMap.find(transportRequest.remoteUserRank) != remoteRankPortMap.end()
                                    && !remoteRankPortMap[transportRequest.remoteUserRank];
                    bool isSecondBuild = false;
                    bool isBatchSendRecv = newTag.find("BatchSendRecv") != std::string::npos;
                    if (transportRequest.isUsedRdma && isBatchSendRecv
                        && bsrTansportRank.find(transportRequest.remoteUserRank) != bsrTansportRank.end()) {
                        // 仅在batchsendrecv rdma下发的时候需要第二次刷新，实际第一次下发都刷好了，第二次就是get一下
                        isSecondBuild = true;
                    }
                    // A3 bsr远端是DirectNpu 链路的话则跳过
                    if ((param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV)
                        && (param.BatchSendRecvDataDes.isDirectRemoteRank[transportRequest.remoteUserRank])) {
                        continue;
                    }
                    bsrTansportRank.insert(transportRequest.remoteUserRank);
                    CHK_RET(CreateLink(
                        newTag, transportRequest, commParam, singleSubCommTransport.links.back(),
                        transportRequest.notifyNum, isBackup, isSecondBuild));
                }
            }
        }
    }

    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::AllocTransportResource(
    const std::string& newTag, const OpParam& opParam, const HcclOpResParam* commParam, AlgResourceRequest& resRequest,
    AlgResourceResponse& algResResponse)
{
    HCCL_INFO("[%s] Entry alloc transport group[%s]", __func__, identifier_.c_str());
    algResResponse.opTransportResponse = resRequest.opTransport;

    std::set<u32> bsrTansportRank;
    for (auto& levelNSubCommTransport : algResResponse.opTransportResponse) {
        for (auto& singleSubCommTransport : levelNSubCommTransport) {
            singleSubCommTransport.links.clear();
            singleSubCommTransport.links.reserve(singleSubCommTransport.transportRequests.size());
            for (auto& transportRequest : singleSubCommTransport.transportRequests) {
                singleSubCommTransport.links.push_back(nullptr);
                if (transportRequest.isValid) {
                    localUserRank_ = transportRequest.localUserRank;
                    receivedAcks_[transportRequest.remoteUserRank] = singleSubCommTransport.supportDataReceivedAck;
                    HCCL_DEBUG(
                        "[%s] alloc transport, newTag[%s], rankId[%u], input memory type[%u], "
                        "output memory type[%u], ",
                        __func__, newTag.c_str(), transportRequest.remoteUserRank, transportRequest.inputMemType,
                        transportRequest.outputMemType);

                    bool isSecondBuild = false;
                    if (transportRequest.isUsedRdma && opParam.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV
                        && bsrTansportRank.find(transportRequest.remoteUserRank) != bsrTansportRank.end()) {
                        // 仅仅在batchsendrecv rdma下发的时候需要第二次刷新，实际第一次下发都刷好了，第二次就是get一下
                        isSecondBuild = true;
                    }
                    // A3 bsr远端是DirectNpu 链路的话则跳过
                    if ((opParam.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV)
                        && (opParam.BatchSendRecvDataDes.isDirectRemoteRank[transportRequest.remoteUserRank])) {
                        continue;
                    }
                    bsrTansportRank.insert(transportRequest.remoteUserRank);
                    CHK_RET(CreateLink(
                        newTag, transportRequest, commParam, singleSubCommTransport.links.back(),
                        transportRequest.notifyNum, false, isSecondBuild));
                }
            }
        }
    }

    return HCCL_SUCCESS;
}

// 在resMap_[tag]对应原有通信资源的基础上继续增量建链，目前batchsendrecv会用到本接口
HcclResult HcclCommAicpu::IncreAllocTransportResource(
    const std::string& newTag, const OpParam& opParam, const HcclOpResParam* commParam, AlgResourceRequest& resRequest,
    AlgResourceResponse& algResResponse)
{
    HCCL_INFO("[HcclCommAicpu][IncreAllocTransportResource] Entry alloc transport group[%s]", identifier_.c_str());
    std::set<u32> bsrTansportRank;
    for (u32 levelIndex = 0; levelIndex < resRequest.opTransport.size(); levelIndex++) {
        for (u32 ringIndex = 0; ringIndex < resRequest.opTransport[levelIndex].size(); ringIndex++) {
            SingleSubCommTransport& reqSingleSubComm = resRequest.opTransport[levelIndex][ringIndex];
            SingleSubCommTransport& respSingleSubComm = algResResponse.opTransportResponse[levelIndex][ringIndex];
            for (u32 rankIndex = 0; rankIndex < reqSingleSubComm.transportRequests.size(); rankIndex++) {
                TransportRequest& transportRequest = reqSingleSubComm.transportRequests[rankIndex];
                CHK_PRT_RET(
                    rankIndex >= respSingleSubComm.links.size(),
                    HCCL_ERROR(
                        "[HcclCommAicpu][IncreAllocTransportResource] The remote rank_id[%u] is larger than "
                        "the existent respSingleSubComm map size[%u]",
                        rankIndex, respSingleSubComm.links.size()),
                    HCCL_E_PARA);
                if (respSingleSubComm.links[rankIndex] != nullptr
                    && respSingleSubComm.links[rankIndex]->GetLinkType() != hccl::LinkType::LINK_RESERVED) {
                    HCCL_INFO(
                        "[IncreAlloc] The link to remote userRank[%u] has existed", transportRequest.remoteUserRank);
                    continue;
                }
                if (transportRequest.isValid) {
                    receivedAcks_[transportRequest.remoteUserRank] = reqSingleSubComm.supportDataReceivedAck;
                    respSingleSubComm.transportRequests[rankIndex] = transportRequest;
                    HCCL_DEBUG(
                        "[HcclCommAicpu][IncreAllocTransportResource] alloc transport, newTag[%s], rankId[%u], "
                        "input memory type[%u], output memory type[%u], ",
                        newTag.c_str(), transportRequest.remoteUserRank, transportRequest.inputMemType,
                        transportRequest.outputMemType);
                    bool isSecondBuild = false;
                    if (transportRequest.isUsedRdma && opParam.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV
                        && bsrTansportRank.find(transportRequest.remoteUserRank) != bsrTansportRank.end()) {
                        // 仅仅在batchsendrecv rdma下发的时候需要第二次刷新，实际第一次下发都刷好了，第二次就是get一下
                        isSecondBuild = true;
                    }
                    // A3 bsr远端是DirectNpu 链路的话则跳过
                    if ((opParam.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV)
                        && (opParam.BatchSendRecvDataDes.isDirectRemoteRank[transportRequest.remoteUserRank])) {
                        continue;
                    }
                    bsrTansportRank.insert(transportRequest.remoteUserRank);
                    CHK_RET(CreateLink(
                        newTag, transportRequest, commParam, respSingleSubComm.links[rankIndex],
                        transportRequest.notifyNum, false, isSecondBuild));
                }
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::CreateLink(
    const std::string& newTag, TransportRequest& transportRequest, const HcclOpResParam* commParam, LINK& link,
    u32 notifyNum, bool isBackup, bool isSecond) // 主备的选择
{
    if (transportRequest.isUsedRdma) {
        CHK_RET(GetRdmaLinksByRankAndTag(
            commParam, CommTransportsType::SPECIAL, transportRequest.remoteUserRank, newTag, link, isBackup, notifyNum,
            isSecond));
    } else {
        CHK_RET(GetSdmaLinksByRankAndTag(
            commParam, CommTransportsType::SPECIAL, transportRequest.remoteUserRank, newTag, link, isBackup, notifyNum,
            transportRequest.linkType));
    }

    HCCL_DEBUG("[%s] alloc special transport success!, tag[%s]", __func__, newTag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::AllocLocalNotifysResource(
    const std::string& newTag, const HcclOpResParam* commParam, const u32 notifyNum,
    std::vector<std::shared_ptr<LocalNotify>>& notifiesMain, std::vector<std::shared_ptr<LocalNotify>>& notifiesAux)
{
    HCCL_INFO(
        "[HcclCommAicpu][AllocLocalNotifysResource]requesting for [%u] notifies, tag[%s].", notifyNum, newTag.c_str());
    if (localNotifies_.capacity() < notifyNum) {
        HCCL_ERROR(
            "[HcclCommAicpu][AllocLocalNotifysResource]request number exceeds max notify number, alloc failed. Max "
            "number is [%u],request num[%u], tag[%s]",
            localNotifies_.capacity(), notifyNum, newTag.c_str());
        return HCCL_E_PARA;
    }

    if (localNotifies_.size() < notifyNum) {
        if (InitLocalNotifyObj(commParam) != HCCL_SUCCESS || localNotifies_.size() < notifyNum) {
            HCCL_ERROR(
                "[HcclCommAicpu][AllocLocalNotifysResource] the need of notify is more than the available, group[%s], "
                "need[%u], total[%u]",
                newTag.c_str(), notifyNum, localNotifies_.size());
            return HCCL_E_INTERNAL;
        }
    }

    u32 halfNotifyNum = notifyNum >> 1;
    notifiesMain.resize(halfNotifyNum);
    notifiesAux.resize(halfNotifyNum);
    for (u32 i = 0; i < halfNotifyNum; i++) {
        notifiesMain[i] = localNotifies_[i << 1];
        notifiesAux[i] = localNotifies_[(i << 1) + 1];
    }
    HCCL_INFO(
        "[HcclCommAicpu][AllocLocalNotifysResource]find enough notifies, numbers[%u], tag[%s].", notifyNum,
        newTag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::AllocStreamsResource(
    const std::string& newTag, const HcclOpResParam* commParam, const u32 streamNum, std::vector<Stream>& streams)
{
    HCCL_INFO(
        "[HcclCommAicpu][AllocStreamsResource]requesting for [%u] slave streams, newTag[%s], group[%s].", streamNum,
        newTag.c_str(), identifier_.c_str());
    if (streamNum == 0) {
        return HCCL_SUCCESS;
    }
    if (slaveStreams_.capacity() < streamNum) {
        HCCL_ERROR(
            "[HcclCommAicpu][AllocStreamsResource]request number exceeds max substream number, alloc failed. Max "
            "number is [%u],request num[%u], tag[%s]",
            slaveStreams_.capacity(), streamNum, newTag.c_str());
        return HCCL_E_PARA;
    }
    if (slaveStreams_.size() < streamNum) {
        if (InitSlaveStreamObjs(commParam) != HCCL_SUCCESS || slaveStreams_.size() < streamNum) {
            HCCL_ERROR(
                "[HcclCommAicpu][AllocStreamsResource] the need of streams is more than the "
                "available, tag[%s], need[%u], total[%u]",
                newTag.c_str(), streamNum, slaveStreams_.size());
            return HCCL_E_INTERNAL;
        }
    }
    streams = std::vector<Stream>(slaveStreams_.begin(), slaveStreams_.begin() + streamNum);
    HCCL_INFO(
        "[HcclCommAicpu][AllocStreamsResource]find enough slave streams [%u], tag[%s].", streamNum, newTag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::AllocScratchMemResource(
    const std::string& newTag, const HcclOpResParam* commParam, const u64& scratchMemSize, DeviceMem& scratchMem,
    bool reAllocFlag)
{
    HCCL_INFO(
        "[HcclCommAicpu][AllocScratchMemResource]requesting for [%llu] bytes scratch mem, tag[%s].", scratchMemSize,
        newTag.c_str());
    if (scratchMemSize != 0) {
        if (tagScratchMem_.find(newTag) == tagScratchMem_.end()) {
            HcclResult ret = InitLocalTagRes(commParam->localRes.nextTagRes);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[HcclCommAicpu][AllocScratchMemResource]InitLocalTagRes error group[%s]", identifier_.c_str()),
                ret);
        }

        if (tagScratchMem_.find(newTag) == tagScratchMem_.end()) {
            HCCL_ERROR(
                "[HcclCommAicpu][AllocScratchMemResource]alloc scratch memory failed."
                "requesting for [%llu] bytes, tag[%s].",
                scratchMemSize, newTag.c_str());
            return HCCL_E_NOT_FOUND;
        }

        if (reAllocFlag) {
            HCCL_INFO(
                "[HcclCommAicpu][AllocScratchMemResource]need to reAlloc mem [%llu] bytes scratch mem, tag[%s].",
                scratchMemSize, newTag.c_str());
            HcclResult ret = InitLocalTagRes(commParam->localRes.nextTagRes, reAllocFlag);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[HcclCommAicpu][AllocScratchMemResource]InitLocalTagRes error group[%s]", identifier_.c_str()),
                ret);
        }

        // 因为aicpu_communicator中会对scratchMem做对齐，所以tagScratchMem_中的大小会偏小（被对齐截断一部分）
        // 但是两者的差值应该不能超过2个CCE_REDUCE_ALIGN_SIZE，否则应该是不对的
        if (scratchMemSize - tagScratchMem_[newTag]->size() > (CCE_REDUCE_ALIGN_SIZE + CCE_REDUCE_ALIGN_SIZE)) {
            HCCL_ERROR(
                "[HcclCommAicpu][AllocScratchMemResource]alloc tag[%s] scratch memory failed."
                "requesting [%llu] bytes actual [%llu] bytes",
                newTag.c_str(), scratchMemSize, tagScratchMem_[newTag]->size());
            return HCCL_E_PARA;
        }
        scratchMem = DeviceMem::create(tagScratchMem_[newTag]->ptr(), tagScratchMem_[newTag]->size());
    }
    HCCL_INFO(
        "[HcclCommAicpu][AllocScratchMemResource]find enough [%llu] bytes scratch mem, tag[%s].", scratchMemSize,
        newTag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::AllocAlgResource(
    const std::string& newTag, const OpParam& opParam, const HcclOpResParam* commParam, AlgResourceRequest& resRequest,
    AlgResourceResponse& algResResponse)
{
    algResResponse.cclInputMem = cclInputBuffer_;
    algResResponse.cclOutputMem = cclOutputBuffer_;
    algResResponse.paramInputMem = DeviceMem::create(opParam.inputPtr, opParam.inputSize);
    algResResponse.paramOutputMem = DeviceMem::create(opParam.outputPtr, opParam.outputSize);

    PetersonLockGuard guard(hostDeviceLock_.get());
    CHK_PRT_RET(
        guard.IsLockFailed(), HCCL_ERROR("[HcclCommAicpu][AllocAlgResource] hostDeviceLock lock failed"),
        HCCL_E_INTERNAL);

    CHK_RET(AllocScratchMemResource(newTag, commParam, resRequest.scratchMemSize, algResResponse.scratchMem));
    CHK_RET(AllocStreamsResource(newTag, commParam, resRequest.streamNum, algResResponse.slaveStreams));
    CHK_RET(AllocLocalNotifysResource(
        newTag, commParam, resRequest.notifyNum, algResResponse.notifiesMain, algResResponse.notifiesAux));
    CHK_RET(AllocTransportResource(newTag, opParam, commParam, resRequest, algResResponse));
    HCCL_INFO("[HcclCommAicpu][AllocAlgResource] alloc resource success tag[%s].", newTag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::CalcResRequest(
    const std::string& algName, const OpParam& param, std::unique_ptr<CollExecutorBase>& executor,
    AlgResourceRequest& resourceRequest)
{
    if (executor.get() == nullptr) {
        executor = CollAlgExecRegistry::Instance().GetAlgExec(algName, dispatcher_, topoMatcher_);
        CHK_PRT_RET(
            executor.get() == nullptr,
            HCCL_ERROR("[HcclCommAicpu][CalcResRequest]Fail to find executor for algName[%s]", algName.c_str()),
            HCCL_E_PARA);
        executor->SetAlgType(algType_);
        executor->SetCCLInBuffer(cclbufferSize_);

        if (param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
            bool isSupportSDMAReduce = false;
            if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
                isSupportSDMAReduce
                    = IsSupportSDMAReduce(param.inputPtr, param.outputPtr, param.DataDes.dataType, param.reduceType);
            } else {
                isSupportSDMAReduce = IsSupportSDMAReduce(
                    cclInputBuffer_.ptr(), cclOutputBuffer_.ptr(), param.DataDes.dataType, param.reduceType);
            }
            executor->SetIsSupportSDMAReduce(isSupportSDMAReduce);
        }
    }
    CHK_RET(CalSendRecvInfoFor910B(algName, param, executor));
    return executor->CalcResRequest(param, resourceRequest);
}

u32 HcclCommAicpu::CalculateOpExecIndex(const OpParam& opParam, u32 userRank)
{
    u32 opIndex = 0;
    s32 commIndex = 0;
    // 用于重执行和taskException打印的算子计数，bsr/sendrecv/其他算子分别计数
    if (opParam.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
        commIndex = -1; // batchSendRecv使用 key = -1
    } else if (opParam.opType == HcclCMDType::HCCL_CMD_SEND) {
        commIndex = opParam.dstRank;
    } else if (opParam.opType == HcclCMDType::HCCL_CMD_RECEIVE) {
        commIndex = opParam.srcRank;
    } else {
        commIndex = userRank;
    }

    auto it = opExecIndexMap_.find(commIndex);
    if (it != opExecIndexMap_.end()) {
        opIndex = ++(it->second);
    } else {
        opExecIndexMap_.insert({commIndex, 1});
        opIndex = 1;
    }

    HCCL_DEBUG(
        "%s tag:%s opType:%u commIndex:%d opIndex:%u", __func__, opParam.tag.c_str(), opParam.opType, commIndex,
        opIndex);
    return opIndex;
}

HcclResult HcclCommAicpu::PrepareSymmetricMemory(const OpParam& param, OpCommTransport& opTransportResponse)
{
    CHK_PRT_RET(
        opTransportResponse.size() == 0,
        HCCL_ERROR("[HcclCommAicpu][PrepareSymmetricMemory] opTransportResponse size is 0"), HCCL_E_PARA);

    const std::unordered_set<LinkType> supportedLinkTypes
        = {LinkType::LINK_HCCS, LinkType::LINK_SIO, LinkType::LINK_HCCS_SW};
    for (u32 levelIdx = 0; levelIdx < opTransportResponse.size(); levelIdx++) {
        for (auto& singleSubCommTransport : opTransportResponse[levelIdx]) {
            if (singleSubCommTransport.isZeroCopy == false) {
                continue;
            }
            for (u64 i = 0; i < singleSubCommTransport.links.size(); ++i) {
                LINK& link = singleSubCommTransport.links[i];
                if (link == nullptr || !singleSubCommTransport.transportRequests[i].isValid
                    || supportedLinkTypes.count(link->GetLinkType()) == 0) {
                    continue; // 无效或者不支持的链路
                }
                u32 peerRank = link->GetRemoteRank();
                void* remoteIn = nullptr;
                CHK_RET(HcclSymWinGetPeerPointer(param.inputSymWindow, param.inputOffset, peerRank, &remoteIn));
                void* remoteOut = nullptr;
                CHK_RET(HcclSymWinGetPeerPointer(param.outputSymWindow, param.outputOffset, peerRank, &remoteOut));

                CHK_PRT_RET(
                    remoteIn == nullptr || remoteOut == nullptr,
                    HCCL_ERROR(
                        "[HcclCommAicpu][PrepareSymmetricMemory] remoteRank[%d] in[%p] out[%p] is invalid", peerRank,
                        remoteIn, remoteOut),
                    HCCL_E_INTERNAL);
                HCCL_INFO(
                    "[HcclCommAicpu][PrepareSymmetricMemory] remoteRank[%d] in[%p] out[%p]", peerRank, remoteIn,
                    remoteOut);
                CHK_RET(link->UpdateRemoteAddr(remoteIn, remoteOut));
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::ExecOp(
    const std::string& newTag, const std::string& algName, OpParam& opParam, const HcclOpResParam* commParam)
{
    std::unique_ptr<CollExecutorBase> executor;
    hccl::AlgResourceResponse* algResResponse;
    CHK_RET(GetAlgResponseRes(newTag, algName, opParam, commParam, executor, algResResponse));

    if (isZeroCopy_ || isSymmetricMemory_) {
        if (isSymmetricMemory_) {
            CHK_RET(PrepareSymmetricMemory(opParam, algResResponse->opTransportResponse));
        } else {
            HcclResult ret = PrepareZeroCopyExchanger(newTag, opParam, algResResponse);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[HcclCommAicpu][ExecOp] newTag[%s], localRankId[%u]", newTag.c_str(), commParam->localUsrRankId),
                ret);
        }

        // 零拷贝场景scratchMem的大小会与用户的输入大小不同，会导致后续算法展开模块计算出错
        // 但是该场景下不会直接访问scratchMem，因此直接使用输入作为scratchMem，使得后续计算正确
        if (opParam.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
            algResResponse->scratchMem = DeviceMem::create(opParam.inputPtr, opParam.inputSize);
            HCCL_INFO(
                "[HcclCommAicpu][ExecOp] ZeroCopy reduce-scatter use userInput as scratchMem, inputPtr[%p] "
                "inputSize[%lu]",
                opParam.inputPtr, opParam.inputSize);
        }

        algResResponse->paramInputMem = DeviceMem::create(opParam.inputPtr, opParam.inputSize);
        algResResponse->paramOutputMem = DeviceMem::create(opParam.outputPtr, opParam.outputSize);
        HCCL_INFO(
            "[HcclCommAicpu][ExecOp] zero copy modify paramInput paramOutput to algResResp inputPtr[%p] inputSize[%lu] "
            "outputPtr[%p] outputSize[%lu]",
            algResResponse->paramInputMem.ptr(), algResResponse->paramInputMem.size(),
            algResResponse->paramOutputMem.ptr(), algResResponse->paramOutputMem.size());
    }

    hcclOpExecIndex_ = CalculateOpExecIndex(opParam, localUserRank_);
    HcclResult ret = Orchestrate(newTag, algName, opParam, executor, *algResResponse, commParam);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcclCommAicpu][ExecOp] executor op fail, tag[%s], algName[%s], identifier[%s]", newTag.c_str(),
            algName.c_str(), identifier_.c_str());
        CHK_PRT_CONT(
            retryEnable_, HCCL_ERROR("[HcclCommAicpu][ExecOp] executor op fail, some error logs may be recorded in the "
                                     "log/run/device directory, search keyword [ErrToWarn]"));
        if (printTaskExceptionForErr_) {
            PrintTaskExceptionAllComm();
            PrintAicpuCommExecStatus();
            printTaskExceptionForErr_ = false;
        }
        return ret;
    }

    HCCL_ENTRY_INFO(
        commParam->opEntry,
        "[HcclCommAicpu][ExecOp] executor op success tag[%s], newTag[%s], algName[%s], identifier[%s].",
        opParam.tag.c_str(), newTag.c_str(), algName.c_str(), identifier_.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::RefreshAlgResponseTransportRes(
    const std::string& newTag, AlgResourceResponse& algResResponse, std::map<u32, bool>& remoteRankPortMap,
    bool isChangeLinkFlag, const HcclOpResParam* commParam, const OpParam& param)
{
    CHK_PTR_NULL(commParam);
    auto iter = resMap_.find(newTag);
    CHK_PRT_RET(
        iter == resMap_.end(), HCCL_ERROR("[%s]Fail to find algResResponse for tag[%s]", __func__, newTag.c_str()),
        HCCL_E_PARA);

    PetersonLockGuard guard(hostDeviceLock_.get());
    CHK_PRT_RET(guard.IsLockFailed(), HCCL_ERROR("[%s] hostDeviceLock lock failed", __func__), HCCL_E_INTERNAL);
    if (!isChangeLinkFlag) {
        CleanRoceResource(newTag, algResResponse, remoteRankPortMap, param);
        CHK_RET(ReAllocTransportResource(newTag, algResResponse, remoteRankPortMap, commParam, param));
        HCCL_RUN_INFO("[%s] ChangeLinkFlag[%d], current tag[%s].", __func__, isChangeLinkFlag, newTag.c_str());
    } else {
        // 提前清理所有tag的链路，避免冲突
        for (auto& resMapIt : resMap_) {
            HCCL_RUN_INFO("[%s] clean roce resource of tag[%s].", __func__, resMapIt.first.c_str());
            CleanRoceResource(resMapIt.first, resMapIt.second, remoteRankPortMap, param);
        }
        // 对resMap中所有tag的transport link根据主备进行刷新
        for (auto& resMapIt : resMap_) {
            HCCL_RUN_INFO("[%s] refresh algResResponse of tag[%s].", __func__, resMapIt.first.c_str());
            CHK_RET(ReAllocTransportResource(resMapIt.first, resMapIt.second, remoteRankPortMap, commParam, param));
            if (resMapIt.first == newTag) {
                HCCL_RUN_INFO("[%s] current tag[%s].", __func__, newTag.c_str());
                algResResponse = resMapIt.second;
            }
        }
    }

    HCCL_RUN_INFO("[%s] alloc resource success tag[%s].", __func__, newTag.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::CalSendRecvInfoForAlltoall(const OpParam& param)
{
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        std::vector<u64> sendCountMatrix(
            topoInfo_.userRankSize * topoInfo_.userRankSize, param.All2AllDataDes.sendCount);
        CHK_RET(GetAlltoAllvcSendRecvInfo(
            static_cast<void*>(sendCountMatrix.data()), param.All2AllDataDes.sendType, param.All2AllDataDes.recvType));
    } else if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        CHK_PTR_NULL(sendRecvInfoPtr_);
        CHK_RET(
            GetAlltoAllvSendRecvInfo(sendRecvInfoPtr_, param.All2AllDataDes.sendType, param.All2AllDataDes.recvType));
    } else {
        CHK_RET(GetAlltoAllvcSendRecvInfo(
            param.All2AllDataDes.sendCountMatrix, param.All2AllDataDes.sendType, param.All2AllDataDes.recvType));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::CalSendRecvInfoFor910B(
    const std::string& algName, const OpParam& param, std::unique_ptr<CollExecutorBase>& executor)
{
    // A2 AICPU才有机会走入RunAlltoAllVStaged
    if (algName == "RunAlltoAllVStaged" || algName == "RunAlltoAllVFullMesh") {
        CHK_PRT_RET(
            executor.get() == nullptr,
            HCCL_ERROR("[HcclCommAicpu][%s]Fail to find executor for algName[%s]", __func__, algName.c_str()),
            HCCL_E_PARA);
        CHK_RET(CalSendRecvInfoForAlltoall(param));
        CollAlltoAllExecutor* alltoAllExecutor = dynamic_cast<CollAlltoAllExecutor*>(executor.get());
        CHK_PTR_NULL(alltoAllExecutor);
        CHK_RET(alltoAllExecutor->SetExecutorExtraInfo(allMeshAggregationSendRecvInfo_, cclbufferSize_));
        HCCL_DEBUG("[HcclCommAicpu][%s] running algName[%s], prepare SendRecvInfo.", __func__, algName.c_str());
        return HCCL_SUCCESS;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::GetAlgResponseRes(
    const std::string& newTag, const std::string& algName, const OpParam& opParam, const HcclOpResParam* commParam,
    std::unique_ptr<CollExecutorBase>& executor, AlgResourceResponse*& algResResponse)
{
    HCCL_INFO("[%s] algName[%s]", __func__, algName.c_str());
    CHK_PTR_NULL(commParam);
    // 刷新CCLBuffer
    CHK_RET(InitCclbuffer(commParam));
    auto iter = resMap_.find(newTag);
    if (iter == resMap_.end()) {
        std::lock_guard<std::mutex> lock(preemptMutexForResMap_);
        iter = resMap_.find(newTag);
        if (iter == resMap_.end()) {
            HCCL_RUN_INFO("[%s] Alloc resource for alg[%s], tag[%s]", __func__, algName.c_str(), newTag.c_str());
            AlgResourceRequest resRequest;
            CHK_RET(CalcResRequest(algName, opParam, executor, resRequest));
            CHK_RET(AllocAlgResource(newTag, opParam, commParam, resRequest, resMap_[newTag]));
            iter = resMap_.find(newTag);
        } else {
            HCCL_INFO("[%s] Repeatedly inited for alg [%s] is not allowed.", __func__, algName.c_str());
        }
    } else if (algName == "BatchSendRecv" || algName == "BatchSendRecvRetry" || algName == "BatchSendRecvGroup") {
        // 如果是非aclgraph模式，而且不需要增量建链，则跳过CalcResRequest这个计算，节省时间。在非aclgraph模式下，跳过是安全的。
        bool canSkipCalcResRequest = !opParam.isCapture && !opParam.needIncreLink;
        if (!canSkipCalcResRequest) { // 如果不能跳过计算，则走一遍计算的流程，否则就跳过以下计算
            AlgResourceRequest resRequest;
            HCCL_INFO("[%s]IncreAlloc resource for alg[%s], tag[%s]", __func__, algName.c_str(), newTag.c_str());
            CHK_RET(CalcResRequest(algName, opParam, executor, resRequest));
            CHK_RET(IncreAllocTransportResource(newTag, opParam, commParam, resRequest, resMap_[newTag]));
        }
    } else if (algName == "RunAlltoAllVStaged" || algName == "RunAlltoAllVFullMesh") {
        AlgResourceRequest resRequest;
        CHK_RET(CalcResRequest(algName, opParam, executor, resRequest));
        HCCL_INFO(
            "[%s] check if need refresh resource for alg[%s], tag[%s], old[%lu], new[%lu]", __func__, algName.c_str(),
            newTag.c_str(), resMap_[newTag].scratchMem.size(), resRequest.scratchMemSize);
        bool reAllocFlag = !(resMap_[newTag].scratchMem.size() == resRequest.scratchMemSize);
        if (reAllocFlag) {
            PetersonLockGuard guard(hostDeviceLock_.get());
            CHK_PRT_RET(
                guard.IsLockFailed(), HCCL_ERROR("[HcclCommAicpu][AllocAlgResource] hostDeviceLock lock failed"),
                HCCL_E_INTERNAL);
            CHK_RET(AllocScratchMemResource(
                newTag, commParam, resRequest.scratchMemSize, resMap_[newTag].scratchMem, true));
            HCCL_INFO(
                "[%s] refresh resource success for alg[%s], tag[%s], scratchMemSize[%lu], ptr[%p]", __func__,
                algName.c_str(), newTag.c_str(), resRequest.scratchMemSize, resMap_[newTag].scratchMem.ptr());
        }
    }
    CHK_PRT_RET(
        iter == resMap_.end(), HCCL_ERROR("[%s]Fail to find algResResponse for tag[%s]", __func__, newTag.c_str()),
        HCCL_E_PARA);
    algResResponse = &iter->second;
    HCCL_INFO("[HcclCommAicpu][GetAlgResponseRes] success!");
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::GetAlltoAllVCTotalCount(OpParam& param, u64& sendCount, u64& recvCount)
{
    for (u32 i = 0; i < topoInfo_.userRankSize; i++) {
        sendCount += *(
            static_cast<const u64*>(param.All2AllDataDes.sendCountMatrix) + topoInfo_.userRank * topoInfo_.userRankSize
            + i);
        recvCount += *(
            static_cast<const u64*>(param.All2AllDataDes.sendCountMatrix) + topoInfo_.userRank
            + topoInfo_.userRankSize * i);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::GetAlltoAllTotalCount(OpParam& param, u64& sendCount, u64& recvCount)
{
    sendCount = param.All2AllDataDes.sendCount * topoInfo_.userRankSize;
    recvCount = param.All2AllDataDes.sendCount * topoInfo_.userRankSize;
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::GetAlltoAllVTotalCount(OpParam& param, u64& sendCount, u64& recvCount)
{
    for (u32 i = 0; i < topoInfo_.userRankSize; i++) {
        u64 curSendCount = *(static_cast<const u64*>(param.All2AllDataDes.sendCounts) + i)
                           + *(static_cast<const u64*>(param.All2AllDataDes.sdispls) + i);
        sendCount = std::max(sendCount, curSendCount);
        u64 curRecvCount = *(static_cast<const u64*>(param.All2AllDataDes.recvCounts) + i)
                           + *(static_cast<const u64*>(param.All2AllDataDes.rdispls) + i);
        recvCount = std::max(recvCount, curRecvCount);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::SetAlltoAllInputAndOutPutMem(OpParam& param, AlgResourceResponse& algResource)
{
    u32 sendTypeSize = 0, recvTypeSize = 0;
    CHK_RET(SalGetDataTypeSize(param.All2AllDataDes.sendType, sendTypeSize));
    CHK_RET(SalGetDataTypeSize(param.All2AllDataDes.recvType, recvTypeSize));
    u64 sendCount = 0;
    u64 recvCount = 0;
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        CHK_RET(GetAlltoAllTotalCount(param, sendCount, recvCount));
    } else if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        CHK_RET(GetAlltoAllVTotalCount(param, sendCount, recvCount));
    } else if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        CHK_RET(GetAlltoAllVCTotalCount(param, sendCount, recvCount));
    }
    u64 inputSize = sendCount * sendTypeSize;
    u64 outputSize = recvCount * recvTypeSize;
    algResource.paramInputMem = inputSize == 0 ? tinySendRecvMem_ : DeviceMem::create(param.inputPtr, inputSize);
    algResource.paramOutputMem = outputSize == 0 ? tinySendRecvMem_ : DeviceMem::create(param.outputPtr, outputSize);
    HCCL_DEBUG(
        "[HcclCommAicpu][SetAlltoAllInputAndOutPutMem] Set memory for AllToAll, inputSize[%llu], inputPtr[%p],"
        "outputPtr[%p]!",
        inputSize, param.inputPtr, param.outputPtr);
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::CombineReportOpInfo(OpParam& param, bool isRetry, bool isRelay)
{
    MsprofAicpuHCCLOPInfo hcclOpInfo{};
    hcclOpInfo.relay = (isRelay) ? 1 : 0;
    hcclOpInfo.retry = (isRetry) ? 1 : 0;
    hcclOpInfo.dataType = param.DataDes.dataType;
    hcclOpInfo.count = param.DataDes.count;
    hcclOpInfo.groupName = groupHashId_;
    hcclOpInfo.ranksize = topoInfo_.userRankSize;
    std::string algTypeStr = TransferAlgType(algType_);
    CHK_RET(dfx::ProfilingManager::ReportHcclOpInfo(hcclOpInfo, algTypeStr));
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::UpdateProfReportStartSqeIdx()
{
    if (dfx::ProfilingManager::IsL1fromOffToOn()) {
        std::vector<Stream> streams;
        CHK_RET(GetStreamAll(streams));
        for (auto& tmpStream : streams) {
            HcclSqeContext* sqeContext = tmpStream.GetSqeContextPtr();
            SqeRingBuffer* sqeContextBuffer = &(sqeContext->buffer);
            CHK_RET(dfx::ProfilingManager::UpdateStartReportSqeIdx(tmpStream.id(), sqeContextBuffer->tailSqeIdx));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::Orchestrate(
    const std::string& newTag, const std::string& algName, OpParam& param, std::unique_ptr<CollExecutorBase>& executor,
    AlgResourceResponse& algResource, const HcclOpResParam* commParam)
{
    CHK_PTR_NULL(commParam);
    // 算子下发信息记录在共享内存区
    UpdateOpRingBufferIdx();
    CHK_RET(aicpuShareData_.RecordOpInfo(
        newTag, param, (isDeviceMode_ ? mc2OpIndex_ : hcclOpExecIndex_), localUserRank_, isCustom_));
    CHK_RET(UpdateProfReportStartSqeIdx());

    // 每个算子都刷新一下profiling开关, 支持profiling从中间迭代采集
    bool profL0Open = dfx::ProfilingManager::IsProfL0On();
    bool profL1Open = dfx::ProfilingManager::IsProfL1On();
    HCCL_DEBUG("profL0Open:%d, profL1Open:%d", profL0Open, profL1Open);

    LogControl logControl(false, false); // 重执行ERROR日志保底控制，析构时重置日志设置
    HCCL_ENTRY_INFO(
        commParam->opEntry, "[HcclCommAicpu][Orchestrate]start tag[%s] newTag[%s] algName[%s] identifier[%s]",
        param.tag.c_str(), newTag.c_str(), algName.c_str(), identifier_.c_str());
    HCCL_INFO(
        "opRetryHandler.isInplacePreSync[%d] opRetryHandler.isPostSync[%d]",
        algOpContext_.opRetryHandler.isInplacePreSync, algOpContext_.opRetryHandler.isPostSync);
    if (executor.get() == nullptr) {
        executor = CollAlgExecRegistry::Instance().GetAlgExec(algName, dispatcher_, topoMatcher_);
        CHK_PRT_RET(
            executor.get() == nullptr,
            HCCL_ERROR(
                "[HcclCommAicpu][Orchestrate]Fail to find executor "
                "for algName[%s]",
                algName.c_str()),
            HCCL_E_PARA);
        executor->SetAlgType(algType_);
        executor->SetCCLInBuffer(cclbufferSize_);

        if (param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
            bool isSupportSDMAReduce = IsSupportSDMAReduce(
                cclInputBuffer_.ptr(), cclOutputBuffer_.ptr(), param.DataDes.dataType, param.reduceType);
            executor->SetIsSupportSDMAReduce(isSupportSDMAReduce);
        }
    }
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV
        || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        CHK_RET(CalSendRecvInfoFor910B(algName, param, executor));
        CHK_RET(SetAlltoAllInputAndOutPutMem(param, algResource));
        if (algName == "RunAlltoAllVTwoLevelPipeline") {
            CHK_RET(CalSendRecvInfoForAlltoall(param));
            HCCL_DEBUG("[HcclCommAicpu][Orchestrate] running RunAlltoAllVTwoLevelPipeline, prepare SendRecvInfo.");
            CollAlltoAllExecutor* alltoAllExecutor = dynamic_cast<CollAlltoAllExecutor*>(executor.get());
            CHK_PTR_NULL(alltoAllExecutor);
            CHK_RET(alltoAllExecutor->SetExecutorExtraInfo(allMeshAggregationSendRecvInfo_, cclbufferSize_));
        }
        if (algName == "RunAlltoAllVContinuousPipeline" || algName == "RunAlltoAllVPipelineFor91093") {
            CollAlltoAllExecutor* alltoAllExecutor = dynamic_cast<CollAlltoAllExecutor*>(executor.get());
            CHK_PTR_NULL(alltoAllExecutor);
            alltoAllExecutor->SetWaitFlagTimeoutSec(commParam->config.notifyWaitTime);
        }
    }
    auto waitStopExecCmdTimeoutMs = HcclGetWaitStopExecCmdTimeout();
    auto waitStopExecCmdTimeout = std::chrono::milliseconds(waitStopExecCmdTimeoutMs);

    auto opStartTime = std::chrono::steady_clock::now(); // 记录重执行算子耗时
    auto startTime = std::chrono::steady_clock::now();

    KfcError errorCode = KfcError::kNone;
    uint32_t retryCnt = 0;
    bool retryProcessing = false;
    KfcCommand lastCmd = KfcCommand::kNone;
    uint32_t beginSqePos = INVALID_UINT;
    uint32_t endSqePos = INVALID_UINT;
    HcclOpExecFSM state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_INIT;
    HcclResult ret = HCCL_SUCCESS;
    dfxExtendInfo_.kfcStatus = DfxKfcStatus::kOneStart;
    AicpuComContext* ctx = AicpuGetComContext();
    AicpuHcclProcess::CallMC2MaintenanceThread(ctx);
    u32 loopCnt = 0;
    u32 loopNum = 1;
    commParam_ = commParam;
    CHK_RET(InitExecLoop(param, executor, loopNum));

    while (true) {
        switch (state) {
            case HcclOpExecFSM::HCCL_OP_EXEC_FSM_INIT:
                HCCL_INFO("hccl aicpu execute loop %u", loopCnt);
                ret = HcclOpExecFsmInitProcess(newTag, param, algResource, state, errorCode);
                break;
            case HcclOpExecFSM::HCCL_OP_EXEC_FSM_LAUNCH:
                ret = HcclOpExecFsmLaunchProcess(
                    algName, param, executor, algResource, state, errorCode, beginSqePos, endSqePos, retryCnt);
                if (ret == HCCL_E_SUSPENDING && isDeviceMode_ && retryEnable_) {
                    return HCCL_E_SUSPENDING;
                }
                break;
            case HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_END:
                ret = HcclOpExecFsmWaitEndProcess(
                    param, algResource, state, errorCode, retryCnt, param.tag, beginSqePos);
                if (state == HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPING) {
                    startTime = std::chrono::steady_clock::now();
                }
                break;
            case HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPING:
                retryProcessing = true;
                if ((std::chrono::steady_clock::now() - startTime) >= waitStopExecCmdTimeout) {
                    HCCL_ERROR("[OpRetry][AICPU]hccl aicpu wait stop exec timeout[%u ms].", waitStopExecCmdTimeoutMs);
                    errorCode = KfcError::kTimeout;
                    state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
                } else {
                    ret = HcclOpExecFsmStoppingProcess(param, state, errorCode, retryCnt);
                }
                break;
            case HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPED:
                ret = HcclOpExecFsmStoppedProcess(state, errorCode, retryCnt, algName, param, beginSqePos, endSqePos);
                if (state == HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_RETRY) {
                    startTime = std::chrono::steady_clock::now();
                }
                break;
            case HcclOpExecFSM::HCCL_OP_EXEC_FSM_CHANGE_LINK:
                ret = HcclOpExecChangeLinkProcess(newTag, state, errorCode, retryCnt, algResource, commParam, param);
                HCCL_DEBUG(
                    "[OpRetry][AICPU]retry change link finish, retryCnt:%u, tag:%s, state:%d", retryCnt,
                    param.tag.c_str(), state);
                break;
            case HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_RETRY: {
                auto waitRetryCmdTimeoutMs = HcclGetWaitRetryCmdTimeout(retryCnt);
                auto waitRetryCmdTimeout = std::chrono::milliseconds(waitRetryCmdTimeoutMs);
                if ((std::chrono::steady_clock::now() - startTime) >= waitRetryCmdTimeout) {
                    HCCL_ERROR("[OpRetry][AICPU]hccl aicpu wait retry timeout[%u ms].", waitRetryCmdTimeoutMs);
                    errorCode = KfcError::kTimeout;
                    state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
                } else {
                    ret = HcclOpExecFsmWaitRetryProcess(param, state, errorCode, lastCmd);
                }
            } break;
            case HcclOpExecFSM::HCCL_OP_EXEC_FSM_RETRY:
                // 重执行前清理当前算子展开的SQE缓存 (if any), 防止命中非完整的cache
                CHK_RET(aicpuCacheManager_.ClearOpUnfoldCacheEntry(
                    algName, param, algResource, isDeviceMode_, topoInfo_, topoMatcher_, algOpContext_,
                    GetWorkflowMode()));
                ret = HcclOpExecFsmRetryProcess(
                    algName, param, executor, algResource, state, errorCode, retryCnt, beginSqePos, endSqePos);
                break;
            case HcclOpExecFSM::HCCL_OP_EXEC_FSM_END:
                loopCnt++;
                if ((param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) && retryEnable_) {
                    param.BatchSendRecvDataDes.curIterNum = loopCnt;
                    ResetBSRRetryCnt();
                }
                if (loopCnt < loopNum) {
                    state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_INIT;
                    break;
                }
                if (retryCnt > 0) {
                    RecordReportStatus(dfx::ReportStatus::kRetrySuccess);
                    retryProcessing = false;
                    auto opEndTime = std::chrono::steady_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::seconds>(opEndTime - opStartTime).count();
                    HCCL_RUN_INFO(
                        "[OpRetry][AICPU]retry exec success, retryCnt [%u], tag [%s], take time [%ld]s", retryCnt,
                        param.tag.c_str(), duration);
                }
                CHK_RET(CombineReportOpInfo(param, (retryCnt > 0), false));
                return HcclOpExecFsmEndProcess(retryCnt);
            case HcclOpExecFSM::HCCL_OP_EXEC_STOP_LAUNCH:
                HCCL_DEBUG("[NsRecovery][AICPU] stop the kernel");
                // 停止前清理当前算子展开的SQE缓存 (if any), 防止host侧重新展开该算子并命中非完整的cache (例如step快恢)
                CHK_RET(aicpuCacheManager_.ClearOpUnfoldCacheEntry(
                    algName, param, algResource, isDeviceMode_, topoInfo_, topoMatcher_, algOpContext_,
                    GetWorkflowMode()));
                if (!needsResponseStopLaunch_) {
                    return HCCL_E_SUSPENDING;
                } else {
                    HCCL_RUN_INFO("[NsRecovery][AICPU] stop the kernel for stop cmd");
                    needsResponseStopLaunch_ = false;
                    SetCommRecoveryFlag(true);
                    if (UpdateOpExecStatus(state, KfcStatus::kStoplaunch, errorCode, 0) == HCCL_SUCCESS) {
                        return HCCL_E_SUSPENDING;
                    } else {
                        break;
                    }
                }
            case HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR:
            default: {
                if (retryProcessing) {
                    RecordReportStatus(dfx::ReportStatus::kRetryFail);
                    retryProcessing = false;
                }
                UpdateOpExecStatus(state, excuteOpId_, KfcStatus::kRetryError, errorCode, retryCnt);
                dfxExtendInfo_.kfcStatus = DfxKfcStatus::kOneFinished;
                if (!isDeviceMode_) {
                    isOpLaunch = false;
                }
                HCCL_INFO("hccl aicpu set kfcStatus[%d]", dfxExtendInfo_.kfcStatus);
                if (errorCode == KfcError::kExecConstraint) {
                    return HCCL_E_OPRETRY_FAIL;
                }
                return (ret == HCCL_SUCCESS) ? HCCL_E_INTERNAL : ret;
            }
        }
    }
    return ret;
}

HcclResult HcclCommAicpu::InitBsrSendRecvOpIdAndExcuteOpId(
    OpParam& param, AlgResourceResponse& algResource, HcclOpExecFSM& fsmState, KfcError& errorCode)
{
    auto hcclRet = InitBatchSendRecvOpId(param, algResource);
    HCCL_RETRY_CHK_RET_AND_TRANS_FSM(
        hcclRet, HCCL_ERROR("InitBatchSendRecvOpId failed, ret:%u", hcclRet), KfcError::kInner,
        HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR);
    param.BatchSendRecvDataDes.curMode = BatchSendRecvCurMode::SEND_RECV;
    if (param.BatchSendRecvDataDes.curIterNum == 0) {
        bsrSendStream_ = algResource.slaveStreams[BSR_RETRY_SEND_STREAM_INDEX];
        bsrRecvStream_ = algResource.slaveStreams[BSR_RETRY_RECV_STREAM_INDEX];
    }
    HCCL_INFO(
        "BSR: iter %u, tag:%s index:%u", param.BatchSendRecvDataDes.curIterNum, excuteOpId_.tag, excuteOpId_.index);
    HCCL_INFO(
        "BSR: iter %u, send op Tag:%s index:%u", param.BatchSendRecvDataDes.curIterNum, bsrSendOpId_.tag,
        bsrSendOpId_.index);
    HCCL_INFO(
        "BSR: iter %u, recv op Tag:%s index:%u", param.BatchSendRecvDataDes.curIterNum, bsrRecvOpId_.tag,
        bsrRecvOpId_.index);
    excuteOpId_.bsrInfo[HCCL_SEND].index = bsrSendOpId_.index;
    excuteOpId_.bsrInfo[HCCL_RECV].index = bsrRecvOpId_.index;
    excuteOpId_.bsrInfo[HCCL_SEND].tpQpn = bsrSendOpId_.bsrInfo[HCCL_SEND].tpQpn;
    excuteOpId_.bsrInfo[HCCL_RECV].tpQpn = bsrRecvOpId_.bsrInfo[HCCL_RECV].tpQpn;
    excuteOpId_.bsrInfo[HCCL_SEND].streamId = bsrSendOpId_.streamId;
    excuteOpId_.bsrInfo[HCCL_RECV].streamId = bsrRecvOpId_.streamId;
    excuteOpId_.bsrInfo[HCCL_SEND].srcRank = bsrSendOpId_.srcRank;
    excuteOpId_.bsrInfo[HCCL_SEND].detRank = bsrSendOpId_.detRank;
    excuteOpId_.bsrInfo[HCCL_RECV].srcRank = bsrRecvOpId_.srcRank;
    excuteOpId_.bsrInfo[HCCL_RECV].detRank = bsrRecvOpId_.detRank;
    CHK_SAFETY_FUNC_RET(memcpy_s(
        excuteOpId_.bsrInfo[HCCL_SEND].bsrTag, sizeof(excuteOpId_.bsrInfo[HCCL_SEND].bsrTag), bsrSendOpId_.tag,
        sizeof(bsrSendOpId_.tag)));
    CHK_SAFETY_FUNC_RET(memcpy_s(
        excuteOpId_.bsrInfo[HCCL_RECV].bsrTag, sizeof(excuteOpId_.bsrInfo[HCCL_RECV].bsrTag), bsrRecvOpId_.tag,
        sizeof(bsrRecvOpId_.tag)));
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::HcclOpExecFsmInitProcess(
    const std::string& newTag, OpParam& param, AlgResourceResponse& algResource, HcclOpExecFSM& fsmState,
    KfcError& errorCode)
{
    excuteOpId_.index = isDeviceMode_ ? (++mc2OpIndex_) : hcclOpExecIndex_;
    CHK_SAFETY_FUNC_RET(memset_s(excuteOpId_.tag, sizeof(excuteOpId_.tag), 0, sizeof(excuteOpId_.tag)));
    CHK_SAFETY_FUNC_RET(memcpy_s(excuteOpId_.tag, sizeof(excuteOpId_.tag), param.tag.c_str(), param.tag.size()));
    CHK_SAFETY_FUNC_RET(memset_s(excuteOpId_.newTag, sizeof(excuteOpId_.newTag), 0, sizeof(excuteOpId_.newTag)));
    CHK_SAFETY_FUNC_RET(memcpy_s(excuteOpId_.newTag, sizeof(excuteOpId_.newTag), newTag.c_str(), newTag.size()));
    excuteOpId_.isSendRecv = false;
    excuteOpId_.streamId = ~0u;
    excuteOpId_.opType = param.opType;
    excuteOpId_.isBsrTaskStart = false;
    if (param.opType == HcclCMDType::HCCL_CMD_SEND || param.opType == HcclCMDType::HCCL_CMD_RECEIVE) {
        InitSendRecvOpId(param, excuteOpId_);
    } else if ((param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) && retryEnable_) {
        CHK_RET(InitBsrSendRecvOpIdAndExcuteOpId(param, algResource, fsmState, errorCode));
    }
    if (GetNsStopLaunchStatus()) {
        HCCL_WARNING("the op should not be launched in the suspending status");
        fsmState = HcclOpExecFSM::HCCL_OP_EXEC_STOP_LAUNCH;
        return HCCL_SUCCESS;
    }
    auto ret = aicpuHdc_.InitOpExecStatus(kfcStatusTransferD2H_, excuteOpId_);
    isOpLaunch = true;
    HCCL_INFO("%s tag:%s, isDeviceMode:%d, index:%u", __func__, excuteOpId_.tag, isDeviceMode_, excuteOpId_.index);
    HCCL_RETRY_CHK_RET_AND_TRANS_FSM(
        ret, HCCL_ERROR("InitOpExecStatus failed, ret:%u", ret), KfcError::kInner,
        HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR);
    fsmState = HcclOpExecFSM::HCCL_OP_EXEC_FSM_LAUNCH;
    return ret;
}

bool HcclCommAicpu::HcclOpCheckSupportRetry(HcclCMDType opType)
{
    const std::set<HcclCMDType> HcclSupportRetryOpSet
        = {HcclCMDType::HCCL_CMD_BROADCAST,      HcclCMDType::HCCL_CMD_ALLREDUCE,      HcclCMDType::HCCL_CMD_REDUCE,
           HcclCMDType::HCCL_CMD_ALLGATHER,      HcclCMDType::HCCL_CMD_REDUCE_SCATTER, HcclCMDType::HCCL_CMD_ALLTOALLV,
           HcclCMDType::HCCL_CMD_ALLTOALLVC,     HcclCMDType::HCCL_CMD_ALLTOALL,       HcclCMDType::HCCL_CMD_GATHER,
           HcclCMDType::HCCL_CMD_SCATTER,        HcclCMDType::HCCL_CMD_SEND,           HcclCMDType::HCCL_CMD_RECEIVE,
           HcclCMDType::HCCL_CMD_BATCH_SEND_RECV};
    return (HcclSupportRetryOpSet.find(opType) != HcclSupportRetryOpSet.end());
}

HcclResult HcclCommAicpu::HcclOpExecFsmLaunchProcess(
    const std::string& algName, OpParam& param, std::unique_ptr<CollExecutorBase>& executor,
    AlgResourceResponse& algResource, HcclOpExecFSM& fsmState, KfcError& errorCode, uint32_t& beginSqePos,
    uint32_t& endSqePos, uint32_t retryCnt)
{
    HCCL_DEBUG("hccl aicpu start launch task.");

    HcclResult ret = OrchestrateHcclOp(algName, param, executor, algResource, beginSqePos, endSqePos);
    if (ret == HCCL_SUCCESS) { // 下发成功, 并且没有检测到异常cq或中断命令
        fsmState = HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_END;
    } else if (ret == HCCL_E_SUSPENDING) { // 检测到异常cq或中断命令
        if (isDeviceMode_ && retryEnable_) {
            HCCL_RUN_INFO("Orchestrate hccl op suspending, restart handle by mc2 process.");
            return ret;
        }
        if ((param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) && retryEnable_) {
            // batchsendrecv算子下发过程中出现异常，task下发未完成，send 和 recv 均需要重执行;
            // 第一个g故障op重执行下发task完成后，需要主动上报故障，触发第二个op进行重执行
            SetBSRSendOpExecException();
            SetBSRRecvOpExecException();
            HCCL_RUN_INFO("hccl aicpu abort launch batchsendrecv op, need retry");
        }
        CHK_RET(UpdateSuspendStatus(param, fsmState, errorCode, retryCnt));
    } else {
        HCCL_ERROR("OrchestrateHcclOp failed, ret:%u", ret);
        errorCode = KfcError::kInner;
        fsmState = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
    }
    return ret;
}

HcclResult HcclCommAicpu::HcclOpExecFsmWaitEndProcess(
    OpParam& param, AlgResourceResponse& algResource, HcclOpExecFSM& fsmState, KfcError& errorCode, uint32_t retryCnt,
    std::string& tag, const uint32_t& beginSqePos)
{
    HCCL_DEBUG("hccl aicpu wait task finish.");
    auto ret = WaitFinishWhileLoop(mainStream_, algResource.slaveStreams, tag, beginSqePos, param);
    if (ret == HCCL_SUCCESS) {
        HCCL_DEBUG("hccl aicpu exec complete.");
        fsmState = HcclOpExecFSM::HCCL_OP_EXEC_FSM_END;
    } else if (ret == HCCL_E_SUSPENDING) {
        HCCL_RUN_INFO("hccl aicpu force stop in wait end, retryCnt[%u]", retryCnt);
        CHK_RET(UpdateSuspendStatus(param, fsmState, errorCode, retryCnt));
    } else {
        HCCL_ERROR("WaitTaskFinish failed, ret:%u, identifier[%s]", ret, identifier_.c_str());
        errorCode = KfcError::kExec;
        fsmState = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
    }
    return ret;
}

HcclResult HcclCommAicpu::HcclOpExecFsmStoppingProcess(
    const OpParam& param, HcclOpExecFSM& fsmState, KfcError& errorCode, uint32_t retryCnt)
{
    HCCL_DEBUG("hccl aicpu stopping.");
    KfcCommand cmd = KfcCommand::kNone;
    auto ret = aicpuHdc_.GetOpExecCtrlCmd(kfcControlTransferH2D_, cmd);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[OpRetry][AICPU]GetOpExecCtrlCmd failed, ret:%u", ret);
        errorCode = KfcError::kExec;
        fsmState = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
        return ret;
    }
    if (cmd == KfcCommand::kExit) {
        HCCL_RUN_INFO("[OpRetry][AICPU]hccl aicpu exec fsm stop by exit cmd.");
        errorCode = KfcError::kExit;
        fsmState = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
    } else if (cmd == KfcCommand::kStopExec) {
        HCCL_RUN_INFO("[OpRetry][AICPU]hccl aicpu get stop exec cmd.");
        fsmState = HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPED;
    } else if (cmd == KfcCommand::kStopLaunch) {
        if ((param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) && retryEnable_) {
            HcclOpIdentifier targetOp;
            CHK_RET(aicpuHdc_.GetOpExecCtrlTargetOp(kfcControlTransferH2D_, targetOp));
            std::string targetOpTag = std::string(reinterpret_cast<char*>(&targetOp.tag[0]));
            if (targetOpTag != std::string(reinterpret_cast<char*>(&bsrTargetOpId_.tag[0]))) {
                CHK_RET(UpdateSuspendStatus(param, fsmState, errorCode, retryCnt));
            }
        }
    } else if ((cmd == KfcCommand::kNone) || (cmd == KfcCommand::kRetry)) {
        HCCL_DEBUG("hccl aicpu wait for stop exec cmd.");
        // do nothing
    } else if (cmd == KfcCommand::kReportRetryErr) {
        HCCL_ERROR("[OpRetry][AICPU][HcclOpExecFsmStoppingProcess]hccl aicpu can not retry err cmd[%d]", cmd);
        uint16_t rsErrorCode = TS_ERROR_RETRY_CONSTRAINT;
        CHK_PRT(SendTaskExceptionByMBox(rsErrorCode));
        errorCode = KfcError::kExit;
        fsmState = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
        return HCCL_E_OPRETRY_FAIL;
    } else {
        HCCL_ERROR("[OpRetry][AICPU]GetOpExecCtrlCmd failed, invalid cmd[%u]", cmd);
        errorCode = KfcError::kExec;
        fsmState = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::LoadChangeLinkInfo(ChangeLinkInfo& changeLinkInfo)
{
    HcclResult ret = aicpuHdc_.GetOpExecChangeLink(kfcControlTransferH2D_, changeLinkInfo);

    // DEBUG_INFO aicpu接收的changelinkinfo
    std::string changeLinkInfoStr = "";
    for (u32 i = 0; i < changeLinkInfo.remoteRankNum; i++) {
        changeLinkInfoStr
            += (std::to_string(changeLinkInfo.remoteRankList[i]) + ":"
                + std::to_string(changeLinkInfo.isUseDefaultPort[i]) + "; ");
    }
    HCCL_RUN_INFO(
        "[%s]rank[%u], isChangeLinkFlag[%d], changeLinkInfoStr:%s", __func__, localUserRank_,
        changeLinkInfo.isChangeLinkFlag, changeLinkInfoStr.c_str());

    return ret;
}

HcclResult HcclCommAicpu::HcclOpExecChangeLinkProcess(
    const std::string& newTag, HcclOpExecFSM& state, KfcError& errorCode, uint32_t& retryCnt,
    AlgResourceResponse& algResource, const HcclOpResParam* commParam, const OpParam& param)
{
    ChangeLinkInfo changeLinkInfo;
    HcclResult ret = LoadChangeLinkInfo(changeLinkInfo);
    if (ret != HCCL_SUCCESS) {
        errorCode = KfcError::kExec;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
        return ret;
    }
    bool useBackupLink = false;
    std::map<u32, bool> remoteRankPortMap;
    for (u32 i = 0; i < changeLinkInfo.remoteRankNum; i++) {
        remoteRankPortMap.insert({changeLinkInfo.remoteRankList[i], changeLinkInfo.isUseDefaultPort[i]});
        useBackupLink |= (!changeLinkInfo.isUseDefaultPort[i]);
    }
    ret = RefreshAlgResponseTransportRes(
        newTag, algResource, remoteRankPortMap, changeLinkInfo.isChangeLinkFlag, commParam, param);
    if (ret != HCCL_SUCCESS) {
        errorCode = KfcError::kExec;
        state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
        return ret;
    }
    if (useBackupLink) {
        RecordReportStatus(dfx::ReportStatus::kRetryWithBackupLink);
    }
    if ((param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) && retryEnable_) {
        retryCnt = (bsrRetryOp_ == HCCL_SEND) ? bsrSendRetryCnt_ : bsrRecvRetryCnt_;
    }
    errorCode = KfcError::kNone;
    CHK_RET(UpdateOpExecStatus(state, KfcStatus::kChanged, errorCode, retryCnt));
    state = HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_RETRY;
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::BSRStopedProcess(HcclOpExecFSM& fsmState, KfcError& errorCode)
{
    // 判断batchsendrecv算子的send 和recv 操作停止的位置是否满足重执行条件
    // send / recv 的stream停止位置不能位于该算子的首个sqe 和末尾sqe
    u32 bsrSendSqHead;
    u32 bsrRecvSqHead;
    auto ret = QuerySqStatusByType(devId_, bsrSendStream_.sqId(), DRV_SQCQ_PROP_SQ_HEAD, bsrSendSqHead);
    HCCL_RETRY_CHK_RET_AND_TRANS_FSM(
        ret, HCCL_ERROR("[OpRetry][AICPU]query send stream sq head failed, ret:%u", ret), KfcError::kExec,
        HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR);

    ret = QuerySqStatusByType(devId_, bsrRecvStream_.sqId(), DRV_SQCQ_PROP_SQ_HEAD, bsrRecvSqHead);
    HCCL_RETRY_CHK_RET_AND_TRANS_FSM(
        ret, HCCL_ERROR("[OpRetry][AICPU]query recv stream sq head failed, ret:%u", ret), KfcError::kExec,
        HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR);

    ret = ((bsrSendOpBeginSqePos_ == bsrSendSqHead) || (bsrRecvOpBeginSqePos_ == bsrRecvSqHead)) ? HCCL_E_OPRETRY_FAIL :
                                                                                                   HCCL_SUCCESS;
    if (ret != HCCL_SUCCESS) {
        uint16_t rsErrorCode = TS_ERROR_RETRY_CONSTRAINT;
        CHK_PRT(SendTaskExceptionByMBox(rsErrorCode));
    }
    HCCL_RETRY_CHK_RET_AND_TRANS_FSM(
        ret,
        HCCL_ERROR(
            "[OpRetry][AICPU]hccl aicpu wait start task is not complete, can not retry. params: send sq head "
            "%u, recv sq head %u, send sq begin %u, recv sq begin %u",
            bsrSendSqHead, bsrRecvSqHead, bsrSendOpBeginSqePos_, bsrRecvOpBeginSqePos_),
        KfcError::kExecConstraint, HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR);

    ret = ((bsrRetryOp_ == HCCL_SEND) && (bsrSendSqHead == bsrSendOpEndSqePos_)) ? HCCL_E_OPRETRY_FAIL : HCCL_SUCCESS;
    HCCL_RETRY_CHK_RET_AND_TRANS_FSM(
        ret,
        HCCL_ERROR(
            "[OpRetry][AICPU]hccl aicpu send record complete task is complete, can not retry. params: send sq "
            "head %u, recv sq head %u, send sq begin %u, send sq end %u, recv sq begin %u, recv sq end %u",
            bsrSendSqHead, bsrRecvSqHead, bsrSendOpBeginSqePos_, bsrSendOpEndSqePos_, bsrRecvOpBeginSqePos_,
            bsrRecvOpEndSqePos_),
        KfcError::kExecConstraint, HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR);

    ret = ((bsrRetryOp_ == HCCL_RECV) && (bsrRecvSqHead == bsrRecvOpEndSqePos_)) ? HCCL_E_OPRETRY_FAIL : HCCL_SUCCESS;
    HCCL_RETRY_CHK_RET_AND_TRANS_FSM(
        ret,
        HCCL_ERROR(
            "[OpRetry][AICPU]hccl aicpu recv record complete task is complete, can not retry. params: send sq "
            "head %u, recv sq head %u, send sq begin %u, send sq end %u, recv sq begin %u, recv sq end %u",
            bsrSendSqHead, bsrRecvSqHead, bsrSendOpBeginSqePos_, bsrSendOpEndSqePos_, bsrRecvOpBeginSqePos_,
            bsrRecvOpEndSqePos_),
        KfcError::kExecConstraint, HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR);

    HCCL_RUN_INFO(
        "[OpRetry][AICPU]hccl aicpu op is running, can retry. params: send sq head "
        "%u, recv sq head %u, send sq begin %u, send sq end %u, recv sq begin %u, recv sq end %u",
        bsrSendSqHead, bsrRecvSqHead, bsrSendOpBeginSqePos_, bsrSendOpEndSqePos_, bsrRecvOpBeginSqePos_,
        bsrRecvOpEndSqePos_);
    if (IsTaskExceptionForHccs()) {
        HCCL_RUN_INFO("[OpRetry][AICPU]hccl aicpu stop by sdma/write task exception, can retry.");
    }
    errorCode = KfcError::kNone;
    uint32_t retryCnt = (bsrRetryOp_ == HCCL_SEND) ? bsrSendRetryCnt_ : bsrRecvRetryCnt_;
    CHK_RET(UpdateOpExecStatus(fsmState, KfcStatus::kStopExec, errorCode, retryCnt));
    fsmState = HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_RETRY;
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::HcclOpExecFsmStoppedProcess(
    HcclOpExecFSM& fsmState, KfcError& errorCode, uint32_t retryCnt, const std::string& algName, OpParam& param,
    uint32_t beginSqePos, uint32_t endSqePos)
{
    HCCL_DEBUG("hccl aicpu stop exec.");
    KfcCommand cmd = KfcCommand::kNone;
    auto ret = aicpuHdc_.GetOpExecCtrlCmd(kfcControlTransferH2D_, cmd);
    uint16_t rsErrorCode = 0;
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[OpRetry][AICPU]GetOpExecCtrlCmd failed, ret:%u", ret);
        errorCode = KfcError::kExec;
        fsmState = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
        return ret;
    }

    if (cmd == KfcCommand::kExit) {
        HCCL_ERROR("[OpRetry][AICPU]hccl aicpu exec fsm stop by exit cmd.");
        errorCode = KfcError::kExit;
        fsmState = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
        return HCCL_SUCCESS;
    }

    if (cmd == KfcCommand::kReportRetryErr) {
        HCCL_ERROR("[OpRetry][AICPU][HcclOpExecFsmStoppedProcess]hccl aicpu can not retry err cmd[%d]", cmd);
        rsErrorCode = TS_ERROR_RETRY_CONSTRAINT;
        CHK_PRT(SendTaskExceptionByMBox(rsErrorCode));
        errorCode = KfcError::kExit;
        fsmState = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
        return HCCL_E_OPRETRY_FAIL;
    }

    if (!HcclOpSupportRetry(algName, retryEnable_, param)) {
        fsmState = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
        if (param.isInplaceError) {
            errorCode = KfcError::kExecConstraint;
            rsErrorCode = TS_ERROR_RETRY_CONSTRAINT;
            CHK_PRT(SendTaskExceptionByMBox(rsErrorCode));
            HCCL_RETRY_CHK_RET_AND_TRANS_FSM(
                HCCL_E_OPRETRY_FAIL,
                HCCL_ERROR("[Opretry][AICPU][HcclOpExecFsmStoppedProcess]can not retry for inplace error."),
                KfcError::kExecConstraint, HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR);
        } else if (isPollutedZeroCopyOp(param)) {
            errorCode = KfcError::kExecConstraint;
            rsErrorCode = TS_ERROR_RETRY_CONSTRAINT;
            CHK_PRT(SendTaskExceptionByMBox(rsErrorCode));
            HCCL_RETRY_CHK_RET_AND_TRANS_FSM(
                HCCL_E_OPRETRY_FAIL,
                HCCL_ERROR("[Opretry][AICPU][HcclOpExecFsmStoppedProcess]can not retry for zero copy op."),
                KfcError::kExecConstraint, HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR);
            return HCCL_E_OPRETRY_FAIL;
        } else {
            errorCode = KfcError::kExec;
        }
        return HCCL_SUCCESS;
    }

    uint32_t sqHead = 0xFFFFFFFF;
    CHK_RET(QuerySqStatusByType(devId_, mainStream_.sqId(), DRV_SQCQ_PROP_SQ_HEAD, sqHead));
    if (sqHead == endSqePos) {
        rsErrorCode = TS_ERROR_RETRY_CONSTRAINT;
        CHK_PRT(SendTaskExceptionByMBox(rsErrorCode));
        HCCL_RETRY_CHK_RET_AND_TRANS_FSM(
            HCCL_E_OPRETRY_FAIL,
            HCCL_ERROR(
                "[OpRetry][AICPU]hccl aicpu record complete task is complete, can not retry. params: "
                "sqHead %u, beginSqePos %u endSqePos %u",
                sqHead, beginSqePos, endSqePos),
            KfcError::kExecConstraint, HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR);
    } else if (sqHead == beginSqePos) {
        rsErrorCode = TS_ERROR_RETRY_CONSTRAINT;
        CHK_PRT(SendTaskExceptionByMBox(rsErrorCode));
        HCCL_RETRY_CHK_RET_AND_TRANS_FSM(
            HCCL_E_OPRETRY_FAIL,
            HCCL_ERROR(
                "[OpRetry][AICPU]hccl aicpu wait start task is not complete, can not retry. "
                "params: sqHead %u, beginSqePos %u endSqePos %u",
                sqHead, beginSqePos, endSqePos),
            KfcError::kExecConstraint, HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR);
    } else if ((param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) && retryEnable_) {
        CHK_RET(BSRStopedProcess(fsmState, errorCode));
    } else {
        HCCL_RUN_INFO(
            "[OpRetry][AICPU]hccl aicpu op is running, can retry. params: sqHead %u, beginSqePos %u "
            "endSqePos %u",
            sqHead, beginSqePos, endSqePos);
        if (IsTaskExceptionForHccs()) {
            HCCL_RUN_INFO("[OpRetry][AICPU]hccl aicpu stop by sdma/write task exception, can retry.");
        }
        errorCode = KfcError::kNone;
        CHK_RET(UpdateOpExecStatus(fsmState, KfcStatus::kStopExec, errorCode, retryCnt));
        fsmState = HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_RETRY;
    }
    return HCCL_SUCCESS;
}

void HcclCommAicpu::NsCommStop()
{
    if ((StreamsKill(devId_) != HCCL_SUCCESS) || (DeviceQuery(devId_, ts::APP_ABORT_KILL_FINISH, 0U) != HCCL_SUCCESS)) {
        (void)aicpuHdc_.SetOpExecStatus(kfcStatusTransferD2H_, KfcStatus::kError, KfcError::kExec, 0);
        HCCL_ERROR("[NsRecovery][AICPU]Stop failed");
        return;
    }
    // 停止条件算子
    if (isDeviceMode_) {
        (void)InvokeKfcHandler(AicpuKfcHandlerType::kClearCommitTurn, {rpc_});
    }
    (void)aicpuHdc_.SetOpExecStatus(kfcStatusTransferD2H_, KfcStatus::kStopExec, KfcError::kNone, 0);
    (void)HcclOneSideServiceAicpu::DisableAllStreamFunc();
    HCCL_RUN_INFO("[NsRecovery][AICPU]stopFunc Finished");
}

void HcclCommAicpu::NsCommClean()
{
    // 等待drv任务停止
    if ((DeviceQuery(devId_, ts::APP_ABORT_TERMINATE_FINISH, 0U) != HCCL_SUCCESS) || (CleanStreamFunc() != HCCL_SUCCESS)
        || (HcclOneSideServiceAicpu::CleanAllStreamFunc() != HCCL_SUCCESS) || (ResetSqBuff() != HCCL_SUCCESS)) {
        (void)aicpuHdc_.SetOpExecStatus(kfcStatusTransferD2H_, KfcStatus::kError, KfcError::kExec, 0);
        HCCL_ERROR("[NsRecovery][AICPU]stream terminate failed");
        return;
    } else {
        if (isDeviceMode_) {
            (void)InvokeKfcHandler(AicpuKfcHandlerType::kClearMsgArea, {rpc_});
        }
        HCCL_INFO("ClearFunc, after APP_ABORT_TERMINATE_FINISH");
        dfxExtendInfo_.pollStatus = PollStatus::kDefault;
        dfxExtendInfo_.cqeStatus = dfx::CqeStatus::kDefault;
        (void)aicpuHdc_.SetOpExecStatus(kfcStatusTransferD2H_, KfcStatus::kClear, KfcError::kNone, 0);
        endStopLaunch = false;
        isOpLaunch = false;
        needsResponseStopLaunch_ = false;
        errMessageReport_ = true;
        HCCL_RUN_INFO("[NsRecovery][AICPU] clean Finish");
    }
}

HcclResult HcclCommAicpu::GetBackGroundCommand(BackgroundCommand& bgCmd)
{
    return AicpuHdcUtils::GetBackGroundCommand(kfcControlTransferH2D_, bgCmd);
}

HcclResult HcclCommAicpu::ResponseBackGroundStatus(KfcExecStatus& status)
{
    return AicpuHdcUtils::ResponseBackGroundStatus(kfcStatusTransferD2H_, status);
}

HcclResult HcclCommAicpu::GetKfcCommand(KfcCommand& cmd)
{
    return AicpuHdcUtils::GetKfcCommand(kfcControlTransferH2D_, cmd);
}

HcclResult HcclCommAicpu::SetStreamEnable(Stream& stream)
{
    const HcclComStreamInfo& streamInfo = stream.GetHcclStreamInfo();
    HCCL_INFO("[SetStreamEnable] streamid[%d]", streamInfo.actualStreamId);
    CHK_RET(ConfigSqStatusByType(GetDevId(), streamInfo.sqId, DRV_SQCQ_PROP_SQ_DISABLE_TO_ENABLE, 1));
    HandleCqeException(stream, true);
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::CleanStreamFunc()
{
    CHK_RET(SetStreamEnable(mainStream_));
    for (auto& stream : slaveStreams_) {
        CHK_RET(SetStreamEnable(stream));
    }
    CHK_RET(SetStreamEnable(orderStream_));
    return HCCL_SUCCESS;
}

std::string HcclCommAicpu::PrintInplaceStatus(u8 isInplaceStatus)
{
    const u8 kNoOverlap = 0;
    const u8 kAllToAllOverlap = 1;
    const u8 kInplaceOverlap = 2;
    switch (isInplaceStatus) {
        case kNoOverlap:
            // input和output不重叠
            return "There is no overlap.";
        case kAllToAllOverlap:
            // alltoall类算子的input和output重叠
            return "The param.inputPtr is equal to param.outputPtr, hence they overlap.";
        case kInplaceOverlap:
            // input和output重叠
            return "It's inplace case. hence they overlap.";
        default:
            return "It's an unknown overlap case.";
    }
    return "";
}

std::string HcclCommAicpu::PrintInplaceSupportRetryStatus(InplaceSupportRetryStatus inPlaceSupportRetryStatus)
{
    switch (inPlaceSupportRetryStatus) {
        case InplaceSupportRetryStatus::AG_BD_CASE: // 不需要去变成非DMA削减
            // allgather or broadcast 算子
            return "The Allgather or broadcast op supports inplace retry.";
        case InplaceSupportRetryStatus::RETRY_1_ALLOW_NO_DMA_REDUCE_CASE1: // 需要去变成非DMA削减
            // 使用AllReduceMeshSmallCountExecutor, ReduceScatterDeterExecutor
            // 且环境变量配置RetryEnable:1
            return "Since retryEnable:1, the executor without DMAReduce will be applied.";
        case InplaceSupportRetryStatus::RETRY_0_NOT_ALLOW_NO_DMA_REDUCE_CASE1: // 不需要去变成非DMA削减
            // 使用AllReduceMeshSmallCountExecutor, ReduceScatterDeterExecutor
            // 且环境变量配置RetryEnable:0
            return "Since retryEnable:0, ExecutorOnlySupportDMAReduce is not allowed for inplace case.";
        case InplaceSupportRetryStatus::ALWAYS_NO_DMA_REDUCE: // 不需要去变成非DMA削减，本身就是
            // 使用AllReduceComm/ReduceScatterComm
            return "AllReduceComm or ReduceScatterComm is used for inplace case.";
        case InplaceSupportRetryStatus::RETRY_1_ALLOW_NO_DMA_REDUCE_CASE2: // 需要去变成非DMA削减
            // 使用其余在91093场景下使用的reduce scatter, allreduce executor
            // 且环境变量配置RetryEnable:1
            return "Since retryEnable:1, the executor will be applied without DMAReduce operation.";
        case InplaceSupportRetryStatus::RETRY_0_NOT_ALLOW_NO_DMA_REDUCE_CASE2: // 不需要去变成非DMA削减
            // 使用其余在91093场景下使用的reduce scatter, allreduce executor
            // 且环境变量配置RetryEnable:0
            return "Since retryEnable:0, the executor without DMAReduce operation can not be applied.";
        case InplaceSupportRetryStatus::UNKONWN_EXECUTOR: // 不需要去变成非DMA削减
            // 使用未知的executor
            return "The unknown executor does not support for an inplace case yet.";
        case InplaceSupportRetryStatus::USER_LARGER_THAN_CCL: // 不需要去变成非DMA削减
            // UserInMem > CCLInMem 场景
            return "UserInMem > CCLInMem case";
        case InplaceSupportRetryStatus::NOT_BASIC_OP_CASE: // 不需要去变成非DMA削减
            // 非 RS AR AG BD算子场景
            return "Is not ReduceScatter, AllReduce, AllGather or Broadcast case";
        default:
            return "It's unknown case. They overlap.";
    }
    return "";
}

bool HcclCommAicpu::IsNoNeedMonitor(void)
{
    if (taskMonitorInterval_ == 0)
        return true;

    KfcCommand kfcCmd = KfcCommand::kNone;
    (void)GetKfcCommand(kfcCmd);
    if (kfcCmd != KfcCommand::kNone)
        return true;

    BackgroundCommand bgCmd = BackgroundCommand::kNone;
    (void)GetBackGroundCommand(bgCmd);
    if (bgCmd != BackgroundCommand::kNone)
        return true;

    HcclComSuspendingFlag suspendingFlag = HcclComSuspendingFlag::isNull;
    (void)GetSuspendingFlag(suspendingFlag);
    if (suspendingFlag != HcclComSuspendingFlag::isNull)
        return true;
    return false;
}

void HcclCommAicpu::InsertMonitorData(Stream& stream, HcclUs& curTime, u32 sqHead, uint16_t taskId, uint8_t type)
{
    AicpuStreamMontior tmpTaskMonitor;
    tmpTaskMonitor.historyTime = curTime;
    tmpTaskMonitor.historyHead = sqHead;
    tmpTaskMonitor.historyTaskId = taskId;
    tmpTaskMonitor.historyType = type;
    streamTaskMonitor_.insert(std::make_pair(stream.sqId(), tmpTaskMonitor));
    return;
}

bool HcclCommAicpu::IsNeedRefreshMonitorData(
    AicpuStreamMontior& streamMontior, HcclUs& curTime, uint32_t remoteRank, uint16_t taskId, u32 sqHead, u32 sqTail,
    uint8_t type)
{
    auto& historyTime = streamMontior.historyTime;
    auto& historyHead = streamMontior.historyHead;
    auto& historyTaskId = streamMontior.historyTaskId;
    auto& historyType = streamMontior.historyType;
    if ((historyTaskId != taskId) || (sqHead != historyHead) || (sqHead == sqTail) || (historyType != type)
        || ((type == RT_STARS_SQE_TYPE_NOTIFY_WAIT) && (remoteRank == INVALID_VALUE_RANKID))) {
        historyTime = curTime;
        historyHead = sqHead;
        historyTaskId = taskId;
        historyType = type;
        return true;
    }
    return false;
}

HcclResult HcclCommAicpu::StreamTaskMonitor(void)
{
    // 通信域资源已经释放
    CHK_PRT_RET(
        !commOpenStatus, HCCL_DEBUG("[StreamTaskMonitor]group[%s] has been destroyed", identifier_.c_str()),
        HCCL_SUCCESS);
    if (IsNoNeedMonitor())
        return HCCL_SUCCESS;
    HCCL_DEBUG("StreamTaskMonitor print");
    std::vector<Stream> totalStream = {mainStream_};
    totalStream.insert(totalStream.end(), slaveStreams_.begin(), slaveStreams_.end());
    HcclUs curTime = TIME_NOW();
    for (auto& stream : totalStream) {
        u32 sqHead = 0U, sqTail = 0U;
        (void)QuerySqStatus(devId_, stream.sqId(), sqHead, sqTail);
        HcclSqeContext* sqeContext = stream.GetSqeContextPtr();
        CHK_PTR_NULL(sqeContext);
        SqeRingBuffer* sqeContextBuffer = &(sqeContext->buffer);
        CHK_PTR_NULL(sqeContextBuffer);

        uint8_t type = 0;
        uint16_t taskId = 0;
        uint32_t remoteRank = 0;
        std::string tmp = GetTaskExceptionTaskInfo(sqHead, sqeContextBuffer, type, taskId, remoteRank);
        HCCL_DEBUG("GetTaskExceptionTaskInfo type %u taskId %u", type, taskId);
        auto mapIt = streamTaskMonitor_.find(stream.sqId());
        if (mapIt == streamTaskMonitor_.end()) {
            InsertMonitorData(stream, curTime, sqHead, taskId, type);
            continue;
        }

        auto& streamMontior = mapIt->second;
        if (IsNeedRefreshMonitorData(streamMontior, curTime, remoteRank, taskId, sqHead, sqTail, type)) {
            continue;
        }

        auto timeVal = DURATION_US(curTime - streamMontior.historyTime).count();
        const int TIME_CONVERSION = 1000;
        if (timeVal >= taskMonitorInterval_ * TIME_CONVERSION) {
            HCCL_RUN_INFO(
                "[StreamTaskMonitor]prof monitor streamId:%d, sqid:%d, head:%u, tail:%u, time %s us, %s", stream.id(),
                stream.sqId(), sqHead, sqTail, std::to_string(timeVal).c_str(), tmp.c_str());
            HCCL_RUN_INFO(
                "[StreamTaskMonitor]prof monitor %s", GetTaskExceptionOpInfo(sqHead, sqeContextBuffer).c_str());
            PrintTaskExceptionTaskQue(sqHead, sqeContextBuffer, true);
            streamMontior.historyTime = curTime;
            streamMontior.historyHead = sqHead;
            streamMontior.historyTaskId = taskId;
            streamMontior.historyType = type;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::SupportRetryWithInplaceCheck(const std::string& algName, OpParam& param)
{
    // 不支持inplace的通信算子重执行
    u8 isInplaceStatus = 0;
    InplaceSupportRetryStatus inPlaceSupportRetryStatus = InplaceSupportRetryStatus::INPLACE_STATUS_END;
    if (IsHcclOpInplace(param.opType, param, topoInfo_.userRank, topoInfo_.userRankSize, isInplaceStatus)) {
        if (!FitRetryConditionforInPlaceOp(
                param.opType, param, algName, cclbufferSize_, topoInfo_.userRankSize,
                algOpContext_.opRetryHandler.retryEnable, inPlaceSupportRetryStatus)) {
            HCCL_RUN_INFO(
                "[OpRetry][AICPU]hccl supports inplace status: isInplaceStatus[%s], "
                "opRetryHandler.inplaceSupportRetry[%d], opRetryHandler.inPlaceSupportRetryStatus[%s], "
                "opRetryHandler.isInplacePreSync[%d], opRetryHandler.isPostSync[%d].",
                PrintInplaceStatus(isInplaceStatus).c_str(), 0,
                PrintInplaceSupportRetryStatus(inPlaceSupportRetryStatus).c_str(),
                algOpContext_.opRetryHandler.isInplacePreSync, algOpContext_.opRetryHandler.isPostSync);
        } else {
            HCCL_RUN_INFO(
                "[OpRetry][AICPU]hccl supports inplace status: isInplaceStatus[%s], "
                "opRetryHandler.inplaceSupportRetry[%d], opRetryHandler.inPlaceSupportRetryStatus[%s], "
                "opRetryHandler.isInplacePreSync[%d], opRetryHandler.isPostSync[%d].",
                PrintInplaceStatus(isInplaceStatus).c_str(), 1,
                PrintInplaceSupportRetryStatus(inPlaceSupportRetryStatus).c_str(),
                algOpContext_.opRetryHandler.isInplacePreSync, algOpContext_.opRetryHandler.isPostSync);
        }
    } else {
        HCCL_RUN_INFO(
            "[OpRetry][AICPU]hccl supports inplace status: isInplaceStatus[%s], "
            "opRetryHandler.isInplacePreSync[%d], opRetryHandler.isPostSync[%d].",
            PrintInplaceStatus(isInplaceStatus).c_str(), algOpContext_.opRetryHandler.isInplacePreSync,
            algOpContext_.opRetryHandler.isPostSync);
    }
    return HCCL_SUCCESS;
}

bool HcclCommAicpu::HcclOpSupportRetry(const std::string& algName, bool retryEnable, OpParam& param)
{
    HCCL_RUN_INFO(
        "[OpRetry][AICPU]hccl supports retry status: enable[%u], param.tag[%s].", retryEnable, param.tag.c_str());
    if (!retryEnable) {
        HCCL_ERROR("[OpRetry][AICPU]hccl aicpu can not retry, enable[%u].", retryEnable);
        return false;
    }
    if (isPollutedZeroCopyOp(param)) {
        HCCL_ERROR(
            "[OpRetry][AICPU]hccl aicpu can not retry, isZeroCopy[%d], opType[%s].", param.isZeroCopy,
            GetCMDTypeEnumStr(param.opType).c_str());
        return false;
    }

    CHK_RET(SupportRetryWithInplaceCheck(algName, param));
    // 不支持inplace的通信算子重执行
    if ((!algOpContext_.opRetryHandler.inplaceSupportRetry) && (!algOpContext_.opRetryHandler.isInplacePreSync)
        && (!algOpContext_.opRetryHandler.isPostSync)) {
        HCCL_ERROR(
            "[OpRetry][AICPU]hccl aicpu can not retry, not support inplace case, opType[%s], "
            "inputPtr[0x%016lx], outputPtr[0x%016lx], opRetryHandler.inplaceSupportRetry[%d], "
            "opRetryHandler.isInplacePreSync[%d], opRetryHandler.isPostSync[%d]",
            GetCMDTypeEnumStr(param.opType).c_str(), param.inputPtr, param.outputPtr,
            algOpContext_.opRetryHandler.inplaceSupportRetry, algOpContext_.opRetryHandler.isInplacePreSync,
            algOpContext_.opRetryHandler.isPostSync);
        param.isInplaceError = true;
        return false;
    }

    // 不支持的通信算子重执行
    if (HcclOpCheckSupportRetry(param.opType) == false) {
        HCCL_ERROR(
            "[OpRetry][AICPU]hccl aicpu can not retry, not support opType[%s].",
            GetCMDTypeEnumStr(param.opType).c_str());
        return false;
    }
    return true;
}

bool HcclCommAicpu::isPollutedZeroCopyOp(OpParam& param)
{
    // allreduce\reduce\reducescatter\reducescatterv with zerocopy can not support retry.
    bool isPollutedOp
        = ((param.opType == HcclCMDType::HCCL_CMD_ALLREDUCE) || (param.opType == HcclCMDType::HCCL_CMD_REDUCE)
           || (param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER)
           || (param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V));
    return param.isZeroCopy && isPollutedOp;
}

HcclResult HcclCommAicpu::UpdateOpExecStatus(
    HcclOpExecFSM& fsmState, HcclOpIdentifier& opId, KfcStatus state, KfcError& errorCode, uint32_t retryCnt)
{
    HCCL_INFO(
        "UpdateOpExecStatus fsmState %d, tag %s, index %u, state %d, errorCode %d, retryCnt %u.", fsmState, opId.tag,
        opId.index, state, errorCode, retryCnt);
    auto ret = aicpuHdc_.SetOpExecStatus(kfcStatusTransferD2H_, opId, state, errorCode, retryCnt);

    HCCL_RETRY_CHK_RET_AND_TRANS_FSM(
        ret, HCCL_ERROR("SetOpExecStatus failed, ret:%u", ret), KfcError::kExec, HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR);

    return ret;
}

HcclResult
HcclCommAicpu::UpdateOpExecStatus(HcclOpExecFSM& fsmState, KfcStatus state, KfcError& errorCode, uint32_t retryCnt)
{
    HCCL_INFO(
        "UpdateOpExecStatus fsmState %d, state %d, errorCode %d, retryCnt %u.", fsmState, state, errorCode, retryCnt);
    auto ret = aicpuHdc_.SetOpExecStatus(kfcStatusTransferD2H_, state, errorCode, retryCnt);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("SetOpExecStatus failed, ret:%u", ret);
        errorCode = KfcError::kExec;
        fsmState = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
    }
    return ret;
}

static constexpr u32 HCCL_AICPU_WAIT_HOST_BASE_TIME_MS = 200 * 1000;
static constexpr u32 TIME_S_TO_MS = 1000;
u32 HcclCommAicpu::HcclGetWaitStopExecCmdTimeout()
{
    return std::max(static_cast<u32>(linkTimeOut_.count()), HCCL_AICPU_WAIT_HOST_BASE_TIME_MS);
}

u32 HcclCommAicpu::HcclGetWaitRetryCmdTimeout(uint32_t retryCnt)
{
    if (retryCnt == 0) {
        return HcclGetWaitStopExecCmdTimeout() + retryHoldTime_;
    } else {
        return HcclGetWaitStopExecCmdTimeout() + retryIntervalTime_;
    }
}

HcclResult HcclCommAicpu::HcclOpExecFsmWaitRetryProcess(
    const OpParam& param, HcclOpExecFSM& fsmState, KfcError& errorCode, KfcCommand& lastCmd)
{
    HCCL_DEBUG("hccl aicpu wait for retry cmd.");
    KfcCommand cmd = KfcCommand::kNone;
    auto ret = aicpuHdc_.GetOpExecCtrlCmd(kfcControlTransferH2D_, cmd);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("GetOpExecCtrlCmd failed, ret:%u", ret);
        errorCode = KfcError::kExec;
        fsmState = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
        return ret;
    }
    if (cmd == KfcCommand::kRetry) {
        HCCL_RUN_INFO("[OpRetry][AICPU]hccl aicpu recv retry cmd from host.");
        dfxExtendInfo_.pollStatus = PollStatus::kDefault;
        dfxExtendInfo_.cqeStatus = dfx::CqeStatus::kDefault;
        ret = ResetOpRetryException(param.opType);
        HCCL_RETRY_CHK_RET_AND_TRANS_FSM(
            ret, HCCL_ERROR("reset stream buff failed, ret:%u", ret), KfcError::kInner,
            HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR);
        fsmState = HcclOpExecFSM::HCCL_OP_EXEC_FSM_RETRY;
    } else if (cmd == KfcCommand::kChangeLink && lastCmd != KfcCommand::kChangeLink) { // 防止重复执行
        HCCL_RUN_INFO("[OpRetry][AICPU]hccl aicpu recv change link cmd, identify[%s]", identifier_.c_str());
        fsmState = HcclOpExecFSM::HCCL_OP_EXEC_FSM_CHANGE_LINK;
    } else if (cmd == KfcCommand::kExit) {
        HCCL_RUN_INFO("[OpRetry][AICPU]hccl aicpu recv exit cmd from host.");
        errorCode = KfcError::kExit;
        fsmState = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
    } else if (cmd == KfcCommand::kReportRetryErr) {
        HCCL_RUN_INFO("[OpRetry][AICPU]hccl aicpu get report retry err cmd.");
        CHK_PRT(SendTaskExceptionByMBox(TS_ERROR_RETRY_CONSTRAINT));
        errorCode = KfcError::kExit;
        fsmState = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
        return HCCL_E_OPRETRY_FAIL;
    } else if (cmd == KfcCommand::NsStopLaunch && endStopLaunch == false) {
        HCCL_RUN_INFO("[NsRecovery][AICPU]hccl aicpu force stop in launch loop.");
        endStopLaunch = true;
        needsResponseStopLaunch_ = true;
        fsmState = HcclOpExecFSM::HCCL_OP_EXEC_STOP_LAUNCH;
    } else {
        // do nothing
    }
    lastCmd = cmd;
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::ResetOpRetryException(HcclCMDType opType)
{
    if (opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
        CHK_RET(ResetBSRException());
    } else {
        std::vector<Stream> totalStream = {mainStream_};
        totalStream.insert(totalStream.end(), slaveStreams_.begin(), slaveStreams_.end());
        for (auto& stream : totalStream) {
            CHK_RET(CleanStream(stream));
            CHK_RET(ClearStreamCqeException(stream));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::ResetSqBuff()
{
    CHK_RET(CleanStream(mainStream_));
    for (auto& stream : slaveStreams_) {
        CHK_RET(CleanStream(stream));
    }
    CHK_RET(CleanStream(orderStream_));
    HCCL_INFO("reset stream sq buffer success.");
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::UpdateSqStatus(Stream& stream)
{
    HcclSqeContext* sqeContext = stream.GetSqeContextPtr();
    CHK_PTR_NULL(sqeContext);
    SqeRingBuffer* sqeContextBuffer = &(sqeContext->buffer);
    CHK_PTR_NULL(sqeContextBuffer);
    auto& head = sqeContextBuffer->sqHead;
    auto& tail = sqeContextBuffer->sqTail;

    CHK_RET(QuerySqStatusByType(devId_, stream.sqId(), DRV_SQCQ_PROP_SQ_TAIL, head));
    CHK_RET(QuerySqStatusByType(devId_, stream.sqId(), DRV_SQCQ_PROP_SQ_HEAD, tail));
    HCCL_INFO("UpdateSqStatus, sqid:%u head:%u tail:%u.", stream.sqId(), head, tail);
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::HcclOpExecFsmRetryProcess(
    const std::string& algName, OpParam& param, std::unique_ptr<CollExecutorBase>& executor,
    AlgResourceResponse& algResource, HcclOpExecFSM& fsmState, KfcError& errorCode, uint32_t& retryCnt,
    uint32_t& beginSqePos, uint32_t& endSqePos)
{
    retryCnt++;
    if ((param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) && retryEnable_) {
        UpdateBSRRetryCnt();
    }
    HCCL_RUN_INFO("[OpRetry][AICPU]retry launch start, retryCnt:%u, tag[%s].", retryCnt, param.tag.c_str());

    auto ret = RetryOrchestrateHcclOp(algName, param, executor, algResource, beginSqePos, endSqePos);
    if (ret == HCCL_SUCCESS) {
        errorCode = KfcError::kNone;
        if ((param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) && retryEnable_) {
            // 是否有前一个重执行阶段积累的故障未上报，需要再次触发重执行
            CHK_RET(CommitBSRStoredException(fsmState, errorCode));
        } else {
            CHK_RET(UpdateOpExecStatus(fsmState, KfcStatus::kRuning, errorCode, retryCnt));
            fsmState = HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_END;
        }
    } else if (ret == HCCL_E_SUSPENDING) {
        HCCL_RUN_INFO("hccl aicpu force stop in retry launch process");
        if ((param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) && retryEnable_) {
            // batchsendrecv算子下发过程中出现异常，task下发未完成，send 和 recv 均需要重执行
            if (bsrRetryOp_ == HCCL_SEND) {
                SetBSRSendOpExecException();
            } else {
                SetBSRRecvOpExecException();
            }
            HCCL_RUN_INFO("hccl aicpu abort launch batchsendrecv op, need retry.");
        }
        CHK_RET(UpdateSuspendStatus(param, fsmState, errorCode, retryCnt));
    } else {
        HCCL_ERROR("RetryLaunchHcclOp failed, ret:%u", ret);
        errorCode = KfcError::kInner;
        fsmState = HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR;
    }
    return ret;
}

HcclResult HcclCommAicpu::HcclOpExecFsmEndProcess(uint32_t retryCnt)
{
    auto ret
        = aicpuHdc_.SetOpExecStatus(kfcStatusTransferD2H_, excuteOpId_, KfcStatus::kEnd, KfcError::kNone, retryCnt);
    if (!isDeviceMode_) {
        isOpLaunch = false;
    }
    dfxExtendInfo_.kfcStatus = DfxKfcStatus::kOneFinished;
    HCCL_DEBUG("---------- end AICPU_HcclOpExecFsmEndProcess ----------");
    return ret;
}

HcclResult HcclCommAicpu::PrintTaskExceptionAllThreads()
{
    // 非独立算子场景，跳过
    CHK_PRT_RET(
        !GetIsInitIndOp(), HCCL_RUN_INFO("[%s] IndOp group[%s] not init, skip", __func__, identifier_.c_str()),
        HCCL_SUCCESS);

    for (auto& thread : threads_) {
        CHK_RET(taskExecption_.PrintTaskException(*(thread->GetStream())));
    }
    return HCCL_SUCCESS;
}

void HcclCommAicpu::PrintTaskExceptionAllComm()
{
    std::shared_lock<std::shared_mutex> rwlock(AicpuHcclProcess::AicpuGetCommMutex());

    // 先打印本通信域的taskException
    (void)PrintTaskExceptionAllStreams();
    (void)PrintTaskExceptionAllThreads();

    // 再打印其他通信域的taskException
    std::vector<std::pair<std::string, hccl::HcclCommAicpu*>> aicpuCommInfo;
    (void)AicpuHcclProcess::AicpuGetCommAll(aicpuCommInfo);
    for (auto& commInfo : aicpuCommInfo) {
        hccl::HcclCommAicpu* hcclAicpu = commInfo.second;
        if (hcclAicpu == nullptr || hcclAicpu->identifier_ == identifier_) {
            continue;
        }
        (void)hcclAicpu->PrintTaskExceptionAllThreads();
        (void)hcclAicpu->PrintTaskExceptionAllStreams();
    }
}

void HcclCommAicpu::PrintAicpuCommExecStatus()
{
    std::shared_lock<std::shared_mutex> rwlock(AicpuHcclProcess::AicpuGetCommMutex());

    // 记录通信域占核情况
    int64_t inExecGroupNum = 0;
    int64_t aicpuCoreNum = 0;
    (void)hrtHalGetDeviceInfo(devId_, MODULE_TYPE_AICPU, INFO_TYPE_CORE_NUM, &aicpuCoreNum);

    std::vector<std::pair<std::string, hccl::HcclCommAicpu*>> aicpuCommInfo;
    (void)AicpuHcclProcess::AicpuGetCommAll(aicpuCommInfo);
    for (auto& commInfo : aicpuCommInfo) {
        hccl::HcclCommAicpu* hcclAicpu = commInfo.second;
        if (hcclAicpu == nullptr || !hcclAicpu->GetCommInfoStatus()) {
            continue;
        }

        // 获取并打印通信域是否在执行中，以及最后一次下发的算子
        bool isInExec = AicpuHcclProcess::GetCommExecStatus(hcclAicpu->identifier_);
        inExecGroupNum += isInExec ? 1 : 0;
        std::string execStatus = isInExec ? "inExec" : "unExec";
        HCCL_RUN_INFO(
            "AicpuComm: group[%s], status[%s], op[%s], aicpuCoreNum[%lld]", hcclAicpu->identifier_.c_str(),
            execStatus.c_str(), hcclAicpu->GetExcuteOp().c_str(), aicpuCoreNum);
    }

    // AICPU核被占满，部分通信域得不到调度，可能导致通信阻塞，打印维测信息
    if (inExecGroupNum >= aicpuCoreNum && static_cast<int64_t>(aicpuCommInfo.size()) > aicpuCoreNum) {
        HCCL_RUN_WARNING(
            "In Execution group num[%lld], total group num[%u], bigger than Aicpu cores num[%lld]. "
            "Aicpu core being fully utilized may cause tasks to get stuck, and it is necessary to reduce the num of "
            "comm.",
            inExecGroupNum, aicpuCommInfo.size(), aicpuCoreNum);
    }
}

HcclResult HcclCommAicpu::PrintTaskExceptionAllStreams()
{
    // 通信域资源已经释放
    CHK_PRT_RET(
        !commOpenStatus,
        HCCL_RUN_INFO("[PrintTaskExceptionAllStreams]group[%s] has been destroyed", identifier_.c_str()), HCCL_SUCCESS);
    CHK_RET(UtraceInfo_->Flush());
    std::vector<Stream> totalStream = {mainStream_};
    HcclResult ret;
    totalStream.insert(totalStream.end(), slaveStreams_.begin(), slaveStreams_.end());
    for (auto& stream : totalStream) {
        HCCL_RUN_INFO("[PrintTaskExceptionAllStreams]group[%s] streamid[%d] print", identifier_.c_str(), stream.id());
        u32 sqHead = 0U;
        u32 sqTail = 0U;
        ret = QuerySqStatus(devId_, stream.sqId(), sqHead, sqTail);
        if (ret != HCCL_SUCCESS || sqHead == sqTail) { // 此流为空时，不打印
            HCCL_RUN_INFO(
                "[PrintTaskExceptionAllStreams] group[%s] streamid[%d] is empty"
                "or QuerySqStatus failed, ret[%d]",
                identifier_.c_str(), stream.id(), ret);
            continue;
        }
        HcclSqeContext* sqeContext = stream.GetSqeContextPtr();
        SqeRingBuffer* sqeContextBuffer = &(sqeContext->buffer);
        CHK_PTR_NULL(sqeContextBuffer);
        if (stream.id() == mainStream_.id()) {
            SqeInfo sqeInfo;
            ret = SqeContextUtils::QuerySqeInfo(
                sqeContextBuffer->rtsMirrorBuffer + sqHead * HCCL_SQE_SIZE, sqeContextBuffer->rtsqSqeType[sqHead],
                sqeContextBuffer->addInfo[sqHead], &sqeInfo);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[%s]QuerySqeInfo failed, ret[%d]", __func__, ret);
            } else {
                // 根据主流卡在host notify上，则说明未被执行到不打印
                if (sqeInfo.type == RT_STARS_SQE_TYPE_NOTIFY_WAIT && sqeInfo.notifyId == opNotifies_[0]->notifyId_) {
                    HCCL_RUN_INFO(
                        "[PrintTaskExceptionAllStreams] group[%s] op is not activated, do nothing",
                        identifier_.c_str());
                    return HCCL_SUCCESS;
                }
                // 根据主流当前位置，判断该算子是否已经打印过taskException
                if (IsRepeatedOpTaskException(sqHead, sqeContextBuffer)) {
                    HCCL_INFO(
                        "[PrintTaskExceptionAllStreams] group[%s] op has been printed, do nothing",
                        identifier_.c_str());
                    return HCCL_SUCCESS;
                }
            }
        }

        uint8_t type = 0;
        uint16_t taskId = 0;
        uint32_t remoteRank = 0;
        HCCL_ERROR(
            "[TaskException]base information is streamId:%d, sqid:%d, head:%u, tail:%u, %s", stream.id(), stream.sqId(),
            sqHead, sqTail, GetTaskExceptionTaskInfo(sqHead, sqeContextBuffer, type, taskId, remoteRank).c_str());
        PrintTaskExceptionTaskQue(sqHead, sqeContextBuffer);
    }
    return HCCL_SUCCESS;
}

bool HcclCommAicpu::IsRepeatedOpTaskException(u32 idx, SqeRingBuffer* sqeContextBuffer)
{
    const AicpuOpInfo* opInfo = aicpuShareData_.GetAicpuOpInfo(sqeContextBuffer->rtsDfxInfo[idx].opRingBufferIdx);
    CHK_PRT_RET(opInfo == nullptr, HCCL_ERROR("%s fail, opInfo is nullptr", __func__), false);
    std::string opTag = opInfo->tagBuff;
    u32 opIndex = opInfo->opIndex;
    bool opHasPrinted = opTaskException_.find(opTag) != opTaskException_.end() && opTaskException_[opTag] == opIndex;
    opTaskException_[opTag] = opIndex;
    CHK_PRT_CONT(
        opHasPrinted, HCCL_RUN_INFO(
                          "[IsRepeatedOpTaskException]group[%s], op[%s], opIndex[%u] "
                          "has been printed",
                          identifier_.c_str(), opInfo->tagBuff, opInfo->opIndex));
    return opHasPrinted;
}

void HcclCommAicpu::PrepareMc2Handler()
{
    auto& handler = algOpContext_.mc2Handler;
    handler.stepSize = 0U;
    if (!isDeviceMode_) {
        HCCL_INFO("Unset step size for non-MC2.");
        return;
    }
    (void)InvokeKfcHandler(AicpuKfcHandlerType::kSetStepSize, {rpc_, reinterpret_cast<u64>(&handler), GetRankSize()});
}

HcclResult HcclCommAicpu::OrchestrateHcclOp(
    const std::string& algName, OpParam& param, std::unique_ptr<CollExecutorBase>& executor,
    AlgResourceResponse& algResource, uint32_t& beginSqePos, uint32_t& endSqePos)
{
    LogControl logControl(false, false); // 重执行ERROR日志控制，析构时重置日志设置
    PrepareMc2Handler();
    HcclResult ret = HCCL_SUCCESS;
    // task的尾指针，已便重执行stop时判断是否已执行该task，如果该task已执行完成则可支持通信重执行
    CHK_RET(QuerySqStatusByType(devId_, mainStream_.sqId(), DRV_SQCQ_PROP_SQ_TAIL, beginSqePos));

    const bool retryForBatchSndRcv = (param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV && retryEnable_);
    if (retryForBatchSndRcv) {
        CHK_RET(QueryBatchSendRecvPairBeginPos());
        if (param.BatchSendRecvDataDes.curIterNum == 0) {
            // batchsendrecv算子拆分为多轮执行，只有第一个step和最后一个step需要和主stream交互
            CHK_RET(NotifyWait());
        }
        HCCL_INFO(
            "batch send recv op: step %u, mode:%u", param.BatchSendRecvDataDes.curIterNum,
            param.BatchSendRecvDataDes.curMode);
    } else {
        CHK_RET(NotifyWait());
        // 重执行场景, 算子计数在host侧; MC2场景也不开启卡住检测能力
        if (opCounterInfo_.isEnableCounter && !retryEnable_ && !isDeviceMode_) {
            CHK_RET(HcclReduceAsync(
                dispatcher_, reinterpret_cast<void*>(opCounterInfo_.addOneMem),
                opCounterInfo_.memSize / sizeof(int32_t), HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, mainStream_,
                reinterpret_cast<void*>(opCounterInfo_.headCountMem), INVALID_VALUE_RANKID, LinkType::LINK_ONCHIP,
                INLINE_REDUCE_BIT));
        }
    }

    // 打印当前展开的算子信息
    HCCL_INFO(
        "[HcclCommAicpu][OrchestrateHcclOp] opUnfoldIdx_[%u] opType[%d] curRank[%u] rankSize[%u] algName[%s]",
        opUnfoldIdx_, param.opType, topoInfo_.userRank, GetRankSize(), algName.c_str());
    HCCL_INFO(
        "[HcclCommAicpu][OrchestrateHcclOp] inputPtr[0x%016llx] inputSize[%llu] outputPtr[0x%016llx] outputSize[%llu]",
        param.inputPtr, param.inputSize, param.outputPtr, param.outputSize);
    opUnfoldIdx_ += 1;

    // 检查算子展开的动态缓存, 确认是否可以跳过算子展开
    bool needExecute = true;
    bool isCacheMiss = false;
    auto setProfStartCallback = [this]() {
        return this->InvokeKfcHandler(AicpuKfcHandlerType::kSetProfTimeStart, {});
    };
    CHK_RET(aicpuCacheManager_.LookupOpUnfoldCache(
        algName, param, algResource, needExecute, isCacheMiss, mainStream_, slaveStreams_, dispatcher_, isDeviceMode_,
        topoInfo_, topoMatcher_, algOpContext_, ZeroCopyExchanger_, GetWorkflowMode(), tinySendRecvMem_,
        setProfStartCallback));

    // 根据needExecute有条件的执行算子展开
    // 需要算子执行的场景: (i) expansion mode为AI_CPU_NO_CACHE; (ii) uncacheable算子/场景; (iii) cache miss
    if (needExecute) {
        // Cache miss前执行cache相关的预处理
        if (isCacheMiss) {
            CHK_RET(aicpuCacheManager_.PreProcessForCacheMiss(param, executor));
        }

        // executor设置AlgOpContext
        CHK_RET(executor->SetAlgOpContext(algOpContext_));
        (void)InvokeKfcHandler(AicpuKfcHandlerType::kSetProfTimeStart, {});
        ret = executor->Orchestrate(param, algResource);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[HcclCommAicpu][Orchestrate]executor process failed algName[%s], ret = %u", algName.c_str(), ret);
            printTaskExceptionForErr_ |= (ret == HCCL_E_AGAIN);
            return ret;
        }

        // Cache miss后执行cache相关的后处理
        if (isCacheMiss) {
            CHK_RET(aicpuCacheManager_.PostProcessForCacheMiss(
                param, executor, mainStream_, slaveStreams_, dispatcher_, topoInfo_, algOpContext_, GetWorkflowMode()));
        }
    } // 正常算子展开

    // batchsendrecv算子拆分为多轮执行，只有第一个step和最后一个step需要和主stream交互
    if (!retryForBatchSndRcv || param.BatchSendRecvDataDes.curIterNum + 1 >= bsrSendRecvPairs_.size()) {
        // 重执行场景, 算子计数在host侧 MC2场景也不开
        if (opCounterInfo_.isEnableCounter && !retryEnable_ && !isDeviceMode_) {
            CHK_RET(HcclReduceAsync(
                dispatcher_, reinterpret_cast<void*>(opCounterInfo_.addOneMem),
                opCounterInfo_.memSize / sizeof(int32_t), HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, mainStream_,
                reinterpret_cast<void*>(opCounterInfo_.tailCountMem), INVALID_VALUE_RANKID, LinkType::LINK_ONCHIP,
                INLINE_REDUCE_BIT));
        }
        CHK_RET(NotifyPost());
    }
    (void)InvokeKfcHandler(AicpuKfcHandlerType::kSetProfTimeOrch, {});
    ret = LaunchTask(dispatcher_, const_cast<Stream&>(mainStream_));
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcclCommAicpu][LaunchTask]algName[%s] ret = %u", algName.c_str(), ret);
        printTaskExceptionForErr_ |= (ret == HCCL_E_AGAIN);
        return ret;
    }
    ret = LaunchSlaveStreamTask(algResource);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcclCommAicpu][LaunchSlaveStreamTask]algName[%s] ret = %u", algName.c_str(), ret);
        printTaskExceptionForErr_ |= (ret == HCCL_E_AGAIN);
        return ret;
    }
    (void)InvokeKfcHandler(AicpuKfcHandlerType::kSetProfTimeEnd, {});
    if (retryForBatchSndRcv) {
        CHK_RET(QueryBatchSendRecvPairEndPos());
    }

    CHK_RET(QuerySqStatusByType(devId_, mainStream_.sqId(), DRV_SQCQ_PROP_SQ_TAIL, endSqePos));

    HCCL_INFO(
        "hccl aicpu launch hccl op task success. stream sqid:%u begin:%u end:%u", mainStream_.sqId(), beginSqePos,
        endSqePos);
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::RetryOrchestrateHcclOp(
    const std::string& algName, OpParam& param, std::unique_ptr<CollExecutorBase>& executor,
    AlgResourceResponse& algResource, uint32_t& beginSqePos, uint32_t& endSqePos)
{
    LogControl logControl(false, false); // 重执行ERROR日志控制，析构时重置日志设置
    if ((param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) && retryEnable_) {
        param.BatchSendRecvDataDes.curMode
            = (bsrRetryOp_ == HCCL_SEND) ? BatchSendRecvCurMode::SEND : BatchSendRecvCurMode::RECV;
        if (param.BatchSendRecvDataDes.curMode == BatchSendRecvCurMode::SEND) {
            HCCL_INFO(
                "BSR: iter %u, retry send op tag:%s index:%u", param.BatchSendRecvDataDes.curIterNum, bsrSendOpId_.tag,
                bsrSendOpId_.index);
        } else {
            HCCL_INFO(
                "BSR: iter %u, retry recv op tag:%s index:%u", param.BatchSendRecvDataDes.curIterNum, bsrRecvOpId_.tag,
                bsrRecvOpId_.index);
        }
    }

    CHK_RET(AddRetryExecFlipTask(algResource));
    HcclResult ret = executor->Orchestrate(param, algResource);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcclCommAicpu][Orchestrate]executor process failed algName[%s]", algName.c_str());
        return ret;
    }

    if ((param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) && retryEnable_) {
        // batchsendrecv算子重执行时，aicpu 主stream没有clean，不需要重新下发notify record
        // do nothing
    } else {
        CHK_RET(NotifyPost());
    }
    CHK_RET(LaunchTask(dispatcher_, const_cast<Stream&>(mainStream_)));
    CHK_RET(LaunchSlaveStreamTask(algResource));

    CHK_RET(QuerySqStatusByType(devId_, mainStream_.sqId(), DRV_SQCQ_PROP_SQ_TAIL, endSqePos));
    if ((param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) && retryEnable_) {
        CHK_RET(QueryBatchSendRecvPairEndPos());
    }

    HCCL_RUN_INFO(
        "[OpRetry][AICPU]hccl aicpu retry launch hccl op task success. stream sqid:%u begin:%u end:%u",
        mainStream_.sqId(), beginSqePos, endSqePos);
    return HCCL_SUCCESS;
}

bool HcclCommAicpu::IsTaskExceptionForHccs()
{
    if (dfxExtendInfo_.cqeStatus != dfx::CqeStatus::kCqeException) {
        return false;
    }

    // NOTE: 需要task exception补全dfx能力，定位故障task的remote rank;
    // 目前暂不具备识别是否跨片的能力，默认失败的task均为跨片操作。
    if (dfxExtendInfo_.cqeException.sqeType == RT_STARS_SQE_TYPE_SDMA
        && (dfxExtendInfo_.cqeException.errorCode == RT_SDMA_COMPDATAERR
            || dfxExtendInfo_.cqeException.errorCode == RT_SDMA_COMPERR)) {
        return true;
    }
    return false;
}

void HcclCommAicpu::SetAlgType(u64 algType)
{
    algType_.algoLevel0 = static_cast<AlgTypeLevel0>(static_cast<u32>(algType) & ((1 << HCCL_LEVEL_ALGO_WIDTH) - 1));
    algType_.algoLevel1 = static_cast<AlgTypeLevel1>(
        (static_cast<u32>(algType) >> HCCL_LEVEL_ALGO_WIDTH) & ((1 << HCCL_LEVEL_ALGO_WIDTH) - 1));
    algType_.algoLevel2
        = static_cast<AlgTypeLevel2>(static_cast<u32>(algType) >> (HCCL_LEVEL_ALGO_WIDTH + HCCL_LEVEL_ALGO_WIDTH));
    HCCL_INFO("[HcclCommAicpu][SetAlgType]algType:%u", algType);
}

void HcclCommAicpu::SetDebugMode(u8 debugMode) { debugMode_ = debugMode; }

void HcclCommAicpu::SetSendRecvInfoPtr(void* sendRecvInfoPtr) { sendRecvInfoPtr_ = sendRecvInfoPtr; }

bool HcclCommAicpu::IsNoNeedWait(void)
{
    return isDeviceMode_ || (debugMode_ != MC2_DEBUG_WAIT_COMM && retryEnable_ == false);
}

bool HcclCommAicpu::GetOpRetryEnable() { return retryEnable_; }

HcclResult HcclCommAicpu::ReportHcclTaskInfo(Stream& mainStream, std::vector<Stream>& subStreams)
{
    if (dfx::ProfilingManager::GetProfL1State()) {
        CHK_RET(dfx::ProfilingManager::ReportTaskInfo(mainStream.id(), mainStream.GetSqeContextPtr()));
        for (auto& subStream : subStreams) {
            CHK_RET(dfx::ProfilingManager::ReportTaskInfo(subStream.id(), subStream.GetSqeContextPtr()));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::ClearLocalBuff(Stream& mainStream, std::vector<Stream>& subStreams)
{
    CHK_RET(mainStream.ClearLocalBuff());
    CHK_RET(dfx::ProfilingManager::UpdateStartReportSqeIdx(mainStream.id(), 0));
    for (auto& subStream : subStreams) {
        CHK_RET(subStream.ClearLocalBuff());
        CHK_RET(dfx::ProfilingManager::UpdateStartReportSqeIdx(subStream.id(), 0));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::WaitFinishWhileLoop(
    Stream& mainStream, std::vector<Stream>& subStreams, std::string& tag, const uint32_t& beginSqePos, OpParam& param)
{
    // 上报Profiling HCCL INFO信息
    CHK_RET(ReportHcclTaskInfo(mainStream, subStreams));
    CHK_RET(ClearLocalBuff(mainStream, subStreams));
    if (IsNoNeedWait()) {
        return HCCL_SUCCESS;
    }
    const uint64_t startUsec = GetCurCpuTimestamp();
    uint64_t lastUsec = startUsec;
    int32_t sqId = mainStream.sqId();
    uint32_t sqHead = 0;
    uint32_t sqTail = 0;
    CHK_RET(QuerySqStatusByType(devId_, sqId, DRV_SQCQ_PROP_SQ_TAIL, sqTail));
    CHK_RET(QuerySqStatusByType(devId_, sqId, DRV_SQCQ_PROP_SQ_HEAD, sqHead));
    while (sqHead != sqTail) {
        HcclResult ret = CheckOpExecStatus(); // 检查执行状态，判断是否有异常cq或中断命令
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_RUN_INFO("[HcclCommAicpu][WaitFinishWhileLoop]CheckOpExecStatus exception, ret[%u]", ret), ret);

        CHK_RET(QuerySqStatusByType(devId_, sqId, DRV_SQCQ_PROP_SQ_HEAD, sqHead));
        if ((param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) && (retryEnable_) && (sqHead != beginSqePos)
            && (!excuteOpId_.isBsrTaskStart)) {
            // 更新D2H通道里的信息opid中isBsrTaskStart = true
            excuteOpId_.isBsrTaskStart = true;
            HcclResult ret1
                = aicpuHdc_.SetOpExecStatus(kfcStatusTransferD2H_, excuteOpId_, KfcStatus::kRuning, KfcError::kNone, 0);
            CHK_PRT_RET(ret1 != HCCL_SUCCESS, HCCL_ERROR("update OpExecStatus failed, ret:%u", ret1), ret1);
            HCCL_INFO(
                "[HcclCommAicpu][WaitFinishWhileLoop]bsr start task is completed. devId:%d sqid:%d, head:%u,"
                "beginSqePos[%u] group[%s] tag[%s]",
                devId_, sqId, sqHead, beginSqePos, identifier_.c_str(), tag.c_str());
        }
        uint64_t curUsec = GetCurCpuTimestamp();
        if (curUsec - lastUsec > static_cast<uint64_t>(NSEC_PER_SEC) * dfx::kPrintSqInterval) {
            lastUsec = curUsec;
            HCCL_RUN_INFO(
                "[HcclCommAicpu][WaitFinishWhileLoop]Current state. devId:%d sqid:%d, head:%u, tail:%u, "
                "group[%s] tag[%s]",
                devId_, sqId, sqHead, sqTail, identifier_.c_str(), tag.c_str());
        }
        CHK_RET(CheckTaskTimeout(mainStream, startUsec));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::CheckTaskTimeout(const Stream& mainStream, const uint64_t startUsec)
{
    if (sqeWaitTimeOut_ != 0
        && (GetCurCpuTimestamp() - startUsec > static_cast<uint64_t>(NSEC_PER_SEC) * sqeWaitTimeOut_)) {
        uint32_t status = 0U;
        int32_t sqId = mainStream.sqId();
        auto ret = QuerySqStatusByType(devId_, sqId, DRV_SQCQ_PROP_SQ_CQE_STATUS, status);
        if (ret != 0) {
            HCCL_ERROR("[HcclCommAicpu]QuerySqStatusByType status failed. ret = %u sqid:%d", ret, sqId);
        }

        HCCL_ERROR("[HcclCommAicpu]KFC timeout.. group[%s].", identifier_.c_str());
        printTaskExceptionForErr_ = true;
        return HCCL_E_TIMEOUT;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::AddRetryExecFlipTask(AlgResourceResponse& algResource)
{
    CHK_RET(AddRetryPreamble(dispatcher_, mainStream_));
    for (u32 i = 0; i < algResource.slaveStreams.size(); ++i) {
        HcclResult ret = AddRetryPreamble(dispatcher_, algResource.slaveStreams[i]);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[HcclCommAicpu][RetryOrchestrateHcclOp] launch place holder failed, sqid:%u, ret:%u",
                algResource.slaveStreams[i].sqId(), ret);
            return ret;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::LaunchSlaveStreamTask(AlgResourceResponse& algResource)
{
    // 单算子模式中在算法编排中已经执行过LaunchTask，所以这里不需要再执行
    // 只有图模式需要再额外执行一次对从流中的task下发
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        HCCL_INFO("[HcclCommAicpu][LaunchSlaveStreamTask] op base mode don't need launch slave stream task");
        return HCCL_SUCCESS;
    }

    for (u32 i = 0; i < algResource.slaveStreams.size(); ++i) {
        HcclResult ret = LaunchTask(dispatcher_, algResource.slaveStreams[i]);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[HcclCommAicpu][LaunchSlaveStreamTask] launch task failed, sqid:%u, ret:%u",
                algResource.slaveStreams[i].sqId(), ret);
            return ret;
        }
    }

    return HCCL_SUCCESS;
}

HcclResult
HcclCommAicpu::GetAlltoAllvSendRecvInfo(const void* sendRecvInfoPtr, HcclDataType sendType, HcclDataType recvType)
{
    allMeshAggregationSendRecvInfo_.clear();
    u64 stepSize = sizeof(u64) * topoInfo_.userRankSize;
    const u32 addrItemNum = 4;
    const u32 recvLengthStep = 2;
    const u32 recvOffsetStep = 3;
    for (u32 i = 0; i < topoInfo_.userRankSize; i++) {
        SendRecvInfo sendRecvInfo;
        sendRecvInfo.sendLength.resize(topoInfo_.userRankSize);
        sendRecvInfo.sendOffset.resize(topoInfo_.userRankSize);
        sendRecvInfo.recvLength.resize(topoInfo_.userRankSize);
        sendRecvInfo.recvOffset.resize(topoInfo_.userRankSize);
        CHK_SAFETY_FUNC_RET(memcpy_s(
            sendRecvInfo.sendLength.data(), stepSize,
            static_cast<const u8*>(sendRecvInfoPtr) + i * stepSize * addrItemNum + 0 * stepSize, stepSize));
        CHK_SAFETY_FUNC_RET(memcpy_s(
            sendRecvInfo.sendOffset.data(), stepSize,
            static_cast<const u8*>(sendRecvInfoPtr) + i * stepSize * addrItemNum + stepSize, stepSize));
        CHK_SAFETY_FUNC_RET(memcpy_s(
            sendRecvInfo.recvLength.data(), stepSize,
            static_cast<const u8*>(sendRecvInfoPtr) + i * stepSize * addrItemNum + recvLengthStep * stepSize,
            stepSize));
        CHK_SAFETY_FUNC_RET(memcpy_s(
            sendRecvInfo.recvOffset.data(), stepSize,
            static_cast<const u8*>(sendRecvInfoPtr) + i * stepSize * addrItemNum + recvOffsetStep * stepSize,
            stepSize));
        allMeshAggregationSendRecvInfo_.push_back(std::move(sendRecvInfo));
    }

    for (auto& sendRecvInfo : allMeshAggregationSendRecvInfo_) {
        for (u32 i = 0; i < topoInfo_.userRankSize; i++) {
            sendRecvInfo.sendCounts.push_back(sendRecvInfo.sendLength[i] / SIZE_TABLE[sendType]);
            sendRecvInfo.sendDispls.push_back(sendRecvInfo.sendOffset[i] / SIZE_TABLE[sendType]);
            sendRecvInfo.recvCounts.push_back(sendRecvInfo.recvLength[i] / SIZE_TABLE[recvType]);
            sendRecvInfo.recvDispls.push_back(sendRecvInfo.recvOffset[i] / SIZE_TABLE[recvType]);
            HCCL_INFO(
                "[GetAlltoAllvSendRecvInfo] rank[%u], sendCounts[%llu], sendDispls[%llu], "
                "recvCounts[%llu], recvDispls[%llu]",
                i, sendRecvInfo.sendCounts[i], sendRecvInfo.sendDispls[i], sendRecvInfo.recvCounts[i],
                sendRecvInfo.recvDispls[i]);
            HCCL_INFO(
                "[GetAlltoAllvSendRecvInfo] rank[%u], sendLength[%llu], sendOffset[%llu], "
                "recvLength[%llu], recvOffset[%llu]",
                i, sendRecvInfo.sendLength[i], sendRecvInfo.sendOffset[i], sendRecvInfo.recvLength[i],
                sendRecvInfo.recvOffset[i]);
        }
    }
    CHK_RET(CheckSendRecvParams(allMeshAggregationSendRecvInfo_));
    return HCCL_SUCCESS;
}

HcclResult
HcclCommAicpu::GetAlltoAllvcSendRecvInfo(const void* sendCountMatrix, HcclDataType sendType, HcclDataType recvType)
{
    allMeshAggregationSendRecvInfo_.clear();
    for (u32 i = 0; i < topoInfo_.userRankSize; i++) {
        SendRecvInfo sendRecvInfo;
        sendRecvInfo.sendCounts.resize(topoInfo_.userRankSize);
        sendRecvInfo.sendDispls.resize(topoInfo_.userRankSize);
        sendRecvInfo.sendLength.resize(topoInfo_.userRankSize);
        sendRecvInfo.sendOffset.resize(topoInfo_.userRankSize);
        u64 curSendDispls = 0;
        u64 curSendOffset = 0;
        sendRecvInfo.recvCounts.resize(topoInfo_.userRankSize);
        sendRecvInfo.recvDispls.resize(topoInfo_.userRankSize);
        sendRecvInfo.recvLength.resize(topoInfo_.userRankSize);
        sendRecvInfo.recvOffset.resize(topoInfo_.userRankSize);
        u64 curRecvDispls = 0;
        u64 curRecvOffset = 0;
        for (u32 j = 0; j < topoInfo_.userRankSize; j++) {
            u64 curSendCounts = *(static_cast<const u64*>(sendCountMatrix) + i * topoInfo_.userRankSize + j);
            u64 curSendLength = curSendCounts * SIZE_TABLE[sendType];
            sendRecvInfo.sendCounts[j] = curSendCounts;
            sendRecvInfo.sendDispls[j] = curSendDispls;
            sendRecvInfo.sendLength[j] = curSendLength;
            sendRecvInfo.sendOffset[j] = curSendOffset;
            curSendDispls += curSendCounts;
            curSendOffset += curSendLength;
            u64 curRecvCounts = *(static_cast<const u64*>(sendCountMatrix) + i + topoInfo_.userRankSize * j);
            u64 curRecvLength = curRecvCounts * SIZE_TABLE[recvType];
            sendRecvInfo.recvCounts[j] = curRecvCounts;
            sendRecvInfo.recvDispls[j] = curRecvDispls;
            sendRecvInfo.recvLength[j] = curRecvLength;
            sendRecvInfo.recvOffset[j] = curRecvOffset;
            curRecvDispls += curRecvCounts;
            curRecvOffset += curRecvLength;
            HCCL_DEBUG(
                "GetAlltoAllvcSendRecvInfo rank[%u], sendCounts[%llu], sendDispls[%llu] "
                "recvCounts[%llu], recvDispls[%llu]",
                i, sendRecvInfo.sendCounts[j], sendRecvInfo.sendDispls[j], sendRecvInfo.recvCounts[j],
                sendRecvInfo.recvDispls[j]);
        }
        allMeshAggregationSendRecvInfo_.push_back(sendRecvInfo);
    }
    CHK_RET(CheckSendRecvParams(allMeshAggregationSendRecvInfo_));
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::CheckSendRecvParams(const std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo)
{
    u32 rankSize = allMeshAggregationSendRecvInfo.size();
    for (u32 i = 0; i < rankSize; i++) {
        u32 sendsSize = allMeshAggregationSendRecvInfo[i].sendLength.size();
        u32 recvsSize = allMeshAggregationSendRecvInfo[i].recvLength.size();
        if (rankSize != sendsSize || rankSize != recvsSize) {
            HCCL_ERROR(
                "[AlltoAllV][CheckSendRecvParam] rankSize[%u], sendsSize[%u], recvsSize[%u] are not match Index[%u]",
                rankSize, sendsSize, recvsSize, i);
            return HCCL_E_PARA;
        }
        for (u32 j = 0; j < sendsSize; j++) {
            if (allMeshAggregationSendRecvInfo[i].sendLength[j] != allMeshAggregationSendRecvInfo[j].recvLength[i]) {
                HCCL_ERROR(
                    "SendLength[%u][%u]: %llu and recvLength[%u][%u]: %llu are not match", i, j,
                    allMeshAggregationSendRecvInfo[i].sendLength[j], j, i,
                    allMeshAggregationSendRecvInfo[j].recvLength[i]);
                return HCCL_E_PARA;
            }
        }
    }
    return HCCL_SUCCESS;
}
HcclResult HcclCommAicpu::GetStreamAll(std::vector<Stream>& streams)
{
    streams.assign(slaveStreams_.begin(), slaveStreams_.end());
    streams.push_back(mainStream_);
    return HCCL_SUCCESS;
}

// 校验是否有ERROR CQE和停止/退出命令，注册到dispatcher层调用
HcclResult HcclCommAicpu::CheckOpExecStatusCallback()
{
    HcclResult ret = CheckOpExecStatus();
    bool logLevel = (ret == HCCL_E_SUSPENDING);
    // 返回HCCL_E_SUSPENDING时，需要跨作用域修改ERROR日志->RUN_WARNING
    LogControl(logLevel, logLevel);
    return ret;
}

HcclResult HcclCommAicpu::CheckOpExecStatus()
{
    // 检测是否有ERROR CQE
    if (dfxExtendInfo_.pollStatus == PollStatus::kStopAsException) {
        if (IsTaskExceptionForHccs() && retryEnable_) {
            HCCL_RUN_INFO(
                "[OpRetry][AICPU]hccl aicpu stop wait task exec finish, for task exception, identify[%s]",
                identifier_.c_str());
            return HCCL_E_SUSPENDING;
        } else {
            if (!printTaskExceptionForErr_) {
                printTaskExceptionForErr_ = true;
                HCCL_ERROR(
                    "hccl aicpu exec failed, for task exception, identify[%s], cqeStatus[%d], sqeType[%u], "
                    "errorCode[%u]",
                    identifier_.c_str(), dfxExtendInfo_.cqeStatus, dfxExtendInfo_.cqeException.sqeType,
                    dfxExtendInfo_.cqeException.errorCode);
            }
            return HCCL_E_INTERNAL;
        }
    }

    // 检测是否有停止/退出命令
    KfcCommand cmd = KfcCommand::kNone;
    CHK_RET(aicpuHdc_.GetOpExecCtrlCmd(kfcControlTransferH2D_, cmd));
    if (cmd == KfcCommand::kStopLaunch && retryEnable_) {
        HCCL_RUN_INFO(
            "[OpRetry][AICPU]hccl aicpu stop wait finish, for recv stop launch cmd, identify[%s]", identifier_.c_str());
        return HCCL_E_SUSPENDING;
    } else if ((cmd == KfcCommand::NsStopLaunch) && (endStopLaunch == false)) {
        needsResponseStopLaunch_ = true;
        endStopLaunch = true;
        HCCL_RUN_INFO("hccl aicpu stop wait finish, for recv stop launch cmd");
        return HCCL_E_SUSPENDING;
    } else if (cmd == KfcCommand::kDestroyComm) {
        HCCL_ERROR("hccl aicpu stop wait finish, for recv destroy comm cmd");
        return HCCL_E_INTERNAL;
    } else if (cmd == KfcCommand::kExit) {
        HCCL_ERROR("hccl aicpu stop wait finish, for recv exit cmd, identify[%s]", identifier_.c_str());
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::UpdateSuspendStatus(
    const OpParam& param, HcclOpExecFSM& fsmState, KfcError& errorCode, uint32_t retryCnt)
{
    if (needsResponseStopLaunch_ == true) {
        HCCL_RUN_INFO("[NsRecovery][AICPU]hccl aicpu force stop in launch loop");
        fsmState = HcclOpExecFSM::HCCL_OP_EXEC_STOP_LAUNCH;
    } else if (retryEnable_) {
        HCCL_RUN_INFO(
            "[OpRetry][AICPU]hccl aicpu force stop for stop cmd or recoverable task exception, identify[%s]",
            identifier_.c_str());
        errorCode = IsTaskExceptionForHccs() ? KfcError::kSdma : errorCode;

        if ((param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV)) {
            HcclResult ret = GetBSRRetryOpId(param, bsrTargetOpId_);
            HCCL_RETRY_CHK_RET_AND_TRANS_FSM(
                ret, HCCL_ERROR("get batchsendrecv target op failed, ret:%u", ret), KfcError::kExec,
                HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR);
            uint32_t bsrRetryCnt = (bsrRetryOp_ == HCCL_SEND) ? bsrSendRetryCnt_ : bsrRecvRetryCnt_;
            bsrTargetOpId_.isBsrTaskStart = true;
            ret = aicpuHdc_.SetOpExecStatus(
                kfcStatusTransferD2H_, bsrTargetOpId_, KfcStatus::kStoplaunch, errorCode, bsrRetryCnt);
            HCCL_RETRY_CHK_RET_AND_TRANS_FSM(
                ret, HCCL_ERROR("SetOpExecStatus failed, ret:%u", ret), KfcError::kExec,
                HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR);
        } else {
            CHK_RET(UpdateOpExecStatus(fsmState, KfcStatus::kStoplaunch, errorCode, retryCnt));
        }
        fsmState = HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPING;
    } else {
        HCCL_RUN_INFO(
            "[HcclCommAicpu][UpdateSuspendStatus] aicpu force stop in launch loop; needsResponseStopLaunch_[%u] "
            "retryEnable_[%u]",
            needsResponseStopLaunch_, retryEnable_);
        fsmState = HcclOpExecFSM::HCCL_OP_EXEC_STOP_LAUNCH;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::TasktypeTransferD2H(const uint8_t sqeType, TaskType& taskType)
{
    switch (sqeType) {
        case RT_STARS_SQE_TYPE_PLACE_HOLDER:
        case RT_STARS_SQE_TYPE_NOTIFY_WAIT:
            taskType = TaskType::TASK_NOTIFY_WAIT;
            break;
        case RT_STARS_SQE_TYPE_SDMA:
            taskType = TaskType::TASK_SDMA;
            break;
        case RT_STARS_SQE_TYPE_NOTIFY_RECORD:
            taskType = TaskType::TASK_NOTIFY_RECORD;
            break;
        case RT_STARS_SQE_TYPE_WRITE_VALUE:
            taskType = TaskType::TASK_NOTIFY_WAIT;
            break;
        default:
            HCCL_ERROR("TasktypeTransferD2H sqeType[%d] error.", sqeType);
            return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::GenTaskExceptionInfo(u8 sqeType, hccl::Stream& stream, u32 head)
{
    HcclSqeContext* sqeContext = stream.GetSqeContextPtr();
    SqeRingBuffer* sqeContextBuffer = &(sqeContext->buffer);
    CHK_PTR_NULL(sqeContextBuffer);

    const AicpuOpInfo* opInfo = aicpuShareData_.GetAicpuOpInfo(sqeContextBuffer->rtsDfxInfo[head].opRingBufferIdx);
    std::string opTag = opInfo == nullptr ? "unKnown" : opInfo->tagBuff;

    // 获取需要上报的关键信息
    ErrorMessageReport emrInfo{};
    SqeInfo sqeInfo;
    SqeContextUtils::QuerySqeInfo(
        sqeContextBuffer->rtsMirrorBuffer + head * HCCL_SQE_SIZE, sqeContextBuffer->rtsqSqeType[head],
        sqeContextBuffer->addInfo[head], &sqeInfo);
    emrInfo.remoteUserRank = sqeContextBuffer->rtsDfxInfo[head].remoteRank;
    emrInfo.streamId = stream.id();
    emrInfo.taskId = sqeInfo.taskId;
    emrInfo.notifyId = sqeInfo.notifyId;
    emrInfo.rankId = localUserRank_;
    emrInfo.rankSize = topoInfo_.userRankSize;
    emrInfo.algType = algType_;
    emrInfo.opIndex = opInfo == nullptr ? 0 : opInfo->opIndex;
    emrInfo.count = opInfo == nullptr ? 0 : opInfo->count;
    emrInfo.dataType = opInfo == nullptr ? 0 : opInfo->dataType;
    emrInfo.dstAddr = opInfo == nullptr ? 0 : opInfo->dstAddr;
    emrInfo.srcAddr = opInfo == nullptr ? 0 : opInfo->srcAddr;
    emrInfo.reduceType = opInfo == nullptr ? 255 : opInfo->reduceType; // 255 为 HcclReduceOp::HCCL_REDUCE_RESERVED
    CHK_RET(TasktypeTransferD2H(sqeType, emrInfo.taskType));

    CHK_SAFETY_FUNC_RET(memcpy_s(emrInfo.tag, sizeof(emrInfo.tag), opTag.c_str(), opTag.size()));
    CHK_SAFETY_FUNC_RET(memcpy_s(emrInfo.group, sizeof(emrInfo.group), identifier_.c_str(), identifier_.size()));
    CHK_RET(aicpuHdc_.SetErrorMessage(kfcStatusTransferD2H_, emrInfo));
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::PrintTaskExceptionByTaskId(u8 sqeType, u16 taskId, hccl::Stream& stream, u32 tail)
{
    HcclSqeContext* sqeContext = stream.GetSqeContextPtr();
    CHK_PTR_NULL(sqeContext);
    HCCL_ERROR("[HcclCommAicpu][PrintTaskExceptionByTaskId]streamId:%d tail:%u cqeType:%u", stream.id(), tail, sqeType);
    SqeRingBuffer* sqeContextBuffer = &(sqeContext->buffer);
    CHK_PTR_NULL(sqeContextBuffer);
    uint8_t* sqeMirrorBufferAddr = sqeContextBuffer->rtsMirrorBuffer + (tail - 1) * HCCL_SQE_SIZE;
    rtStarsSqeHeader_t* const sqeHeader = (rtStarsSqeHeader_t*)sqeMirrorBufferAddr;

    s32 taskNum = sqeHeader->taskId - taskId;
    HCCL_DEBUG("[HcclCommAicpu]tail sqe taskId[%u] cqe taskId[%u] cqe type[%u]", sqeHeader->taskId, taskId, sqeType);
    s32 sqeIdx = tail - taskNum - 1;
    u32 sqHead = (sqeIdx + HCCL_SQE_MAX_CNT) % HCCL_SQE_MAX_CNT;
    uint8_t type = 0;
    uint16_t taskIdTmp = 0;
    uint32_t remoteRank = 0;
    HCCL_ERROR(
        "[TaskException][AICPU]base information is streamId:%d, sqid:%d, head:%u, tail:%u, %s", stream.id(),
        stream.sqId(), sqHead, tail,
        GetTaskExceptionTaskInfo(sqHead, sqeContextBuffer, type, taskIdTmp, remoteRank).c_str());
    PrintTaskExceptionTaskQue(sqHead, sqeContextBuffer);
    return HCCL_SUCCESS;
}

std::string HcclCommAicpu::GetTaskExceptionOpInfo(u32 idx, SqeRingBuffer* sqeContextBuffer)
{
    const AicpuOpInfo* opInfo = aicpuShareData_.GetAicpuOpInfo(sqeContextBuffer->rtsDfxInfo[idx].opRingBufferIdx);
    CHK_PRT_RET(opInfo == nullptr, HCCL_ERROR("%s fail, opInfo is nullptr", __func__), "unKnown");

    std::stringstream ss;
    ss << "tag:" << opInfo->tagBuff << ", ";
    ss << "group:" << identifier_ << ", ";
    ss << "isCustom:" << opInfo->isCustom << ", ";
    ss << "opLaunchIdx:" << opInfo->opIndex << ", ";
    ss << "opExecIdx:" << opInfo->opExecIndex << ", ";
    ss << "count:" << opInfo->count << ", ";
    ss << "dataType:" << static_cast<u16>(opInfo->dataType) << ", ";
    ss << "opType:" << static_cast<u16>(opInfo->opType) << ", ";
    ss << "rootId:" << opInfo->rootId << ", ";
    ss << "dstAddr:0x" << std::hex << opInfo->dstAddr << ", ";
    ss << "srcAddr:0x" << std::hex << opInfo->srcAddr << ".";
    return ss.str();
}

std::string HcclCommAicpu::GetTaskExceptionTaskInfo(
    u32 sqHead, SqeRingBuffer* sqeContextBuffer, uint8_t& type, uint16_t& taskId, uint32_t& remoteRank)
{
    SqeInfo sqeInfo;
    SqeContextUtils::QuerySqeInfo(
        sqeContextBuffer->rtsMirrorBuffer + sqHead * HCCL_SQE_SIZE, sqeContextBuffer->rtsqSqeType[sqHead],
        sqeContextBuffer->addInfo[sqHead], &sqeInfo);
    type = sqeInfo.type;
    taskId = sqeInfo.taskId;
    remoteRank = sqeContextBuffer->rtsDfxInfo[sqHead].remoteRank;
    std::stringstream ss;
    ss << "type:" << SqeContextUtils::RtsqTaskTypeToStr(sqeInfo.type) << ", ";
    ss << "localRank:" << localUserRank_ << ", ";
    ss << "remoteRank:" << remoteRank << ", ";
    ss << "taskId:" << sqeInfo.taskId << ", ";
    ss << "notifyId:" << sqeInfo.notifyId << ", ";
    ss << "length:" << sqeInfo.length << ", ";
    ss << "addr1High:0x" << std::hex << sqeInfo.addr1High << ", ";
    ss << "addr1Low:0x" << std::hex << sqeInfo.addr1Low << ", ";
    ss << "addr2High:0x" << std::hex << sqeInfo.addr2High << ", ";
    ss << "addr2Low:0x" << std::hex << sqeInfo.addr2Low << ".";
    return ss.str();
}

// 获取指定索引的算子信息
const AicpuOpInfo* HcclCommAicpu::GetOpInfoFromSqIdx(u32 sqIdx, SqeRingBuffer* sqeContextBuffer)
{
    return aicpuShareData_.GetAicpuOpInfo(sqeContextBuffer->rtsDfxInfo[sqIdx].opRingBufferIdx);
}

// 打印算子数据信息
void HcclCommAicpu::PrintOpDataInfo(u32 sqIdx, SqeRingBuffer* sqeContextBuffer, bool isMonitor)
{
    std::string opInfo = GetTaskExceptionOpInfo(sqIdx, sqeContextBuffer);
    if (isMonitor) {
        HCCL_RUN_INFO("[StreamTaskMonitor]opData information is %s", opInfo.c_str());
    } else {
        HCCL_ERROR("[TaskException]opData information is %s", opInfo.c_str());
    }
}

// 打印task序列行
void HcclCommAicpu::PrintTaskLine(bool isMonitor, u32 lineNum, u32 totalPrinted, const std::string& taskLine) const
{
    if (isMonitor) {
        HCCL_RUN_INFO("[StreamTaskMonitor]task sequence[%u/%u] is %s", lineNum, totalPrinted, taskLine.c_str());
    } else {
        HCCL_ERROR("[TaskException]task sequence[%u/%u] is %s", lineNum, totalPrinted, taskLine.c_str());
    }
}

// 更新算子上下文
void HcclCommAicpu::UpdateOpContext(
    u32& opIndex, std::string& opTag, u32& lineCount, std::vector<std::string>& currentOpTasks, u32 newOpIndex,
    const std::string& newOpTag) const
{
    opIndex = newOpIndex;
    opTag = newOpTag;
    lineCount = 0;
    currentOpTasks.clear();
    currentOpTasks.push_back("OP(" + std::to_string(opIndex) + ")");
}

// 准备下一行数据
void HcclCommAicpu::PrepareNextLine(u32 opIndex, u32& lineCount, std::vector<std::string>& currentOpTasks) const
{
    lineCount++;
    currentOpTasks.clear();
    currentOpTasks.push_back("OP(" + std::to_string(opIndex) + ") continued");
}

// 拼接task列表为字符串
std::string HcclCommAicpu::ConcatTaskLine(const std::vector<std::string>& tasks) const
{
    std::string taskLine;
    for (const auto& task : tasks) {
        taskLine += task;
    }
    return taskLine;
}

// 打印剩余未满行的tasks
void HcclCommAicpu::PrintRemainingTasks(
    bool isMonitor, u32 lineCount, u32 printedCount, const std::vector<std::string>& currentOpTasks) const
{
    // size > 1 表示除了OP头还有task
    if (currentOpTasks.size() > 1) {
        std::string taskLine = ConcatTaskLine(currentOpTasks);
        PrintTaskLine(isMonitor, lineCount + 1, printedCount, taskLine);
    }
}

void HcclCommAicpu::PrintTaskExceptionTaskQue(u32 sqIdx, SqeRingBuffer* sqeContextBuffer, bool isMonitor)
{
    const u32 sqeNum = 200;
    const u32 maxTasksPerLine = 50;

    // 获取初始算子信息
    const AicpuOpInfo* lastOpInfo = GetOpInfoFromSqIdx(sqIdx, sqeContextBuffer);
    CHK_PRT_RET(lastOpInfo == nullptr, HCCL_ERROR("%s fail, opInfo is nullptr", __func__), );

    // 初始化状态
    u32 opIndex = lastOpInfo->opIndex;
    std::string opTag = lastOpInfo->tagBuff;
    u32 lineCount = 0;
    u32 printedCount = 0;
    u32 extraSlotPerLine = 2;
    std::vector<std::string> currentOpTasks;
    currentOpTasks.push_back("OP(" + std::to_string(opIndex) + ")");

    // 主循环：遍历并打印
    for (u32 i = 0; i < sqeNum; i++) {
        // 计算新索引并获取算子信息
        u32 newSqIdx = (sqIdx - i + HCCL_SQE_MAX_CNT) % HCCL_SQE_MAX_CNT;
        const AicpuOpInfo* newOpInfo = GetOpInfoFromSqIdx(newSqIdx, sqeContextBuffer);
        CHK_PRT_RET(newOpInfo == nullptr, HCCL_ERROR("%s fail, opInfo is nullptr", __func__), );

        u32 sizeAfterAdd = currentOpTasks.size() + 1;
        if (sizeAfterAdd >= (maxTasksPerLine + extraSlotPerLine)
            || (i > 0 && (newOpInfo->opIndex != opIndex || newOpInfo->tagBuff != opTag))) {
            // 打印算子信息（仅第一次）
            if (lineCount == 0) {
                PrintOpDataInfo(sqIdx, sqeContextBuffer, isMonitor);
            }

            // 拼接并打印task行
            std::string taskLine = ConcatTaskLine(currentOpTasks);
            PrintTaskLine(isMonitor, lineCount + 1, printedCount, taskLine);

            // 更新状态
            if (newOpInfo->opIndex != opIndex || newOpInfo->tagBuff != opTag) {
                UpdateOpContext(opIndex, opTag, lineCount, currentOpTasks, newOpInfo->opIndex, newOpInfo->tagBuff);
            } else {
                PrepareNextLine(opIndex, lineCount, currentOpTasks);
            }
        }
        // 添加task到序列
        currentOpTasks.push_back("," + GetTaskBriefsInfo(newSqIdx, sqeContextBuffer));
        printedCount++;
    }
    // 打印剩余tasks
    PrintRemainingTasks(isMonitor, lineCount, printedCount, currentOpTasks);
}

std::string HcclCommAicpu::GetTaskBriefsInfo(u32 idx, SqeRingBuffer* sqeContextBuffer)
{
    uint8_t* sqeMirrorBufferAddr = sqeContextBuffer->rtsMirrorBuffer + idx * HCCL_SQE_SIZE;
    rtStarsSqeHeader_t* const sqeHeader = (rtStarsSqeHeader_t*)sqeMirrorBufferAddr;
    uint8_t sqeType = sqeHeader->type;

    SqeInfo sqeInfo;
    SqeContextUtils::QuerySqeInfo(
        sqeContextBuffer->rtsMirrorBuffer + idx * HCCL_SQE_SIZE, sqeContextBuffer->rtsqSqeType[idx],
        sqeContextBuffer->addInfo[idx], &sqeInfo);
    uint8_t subType = sqeInfo.subType;

    std::stringstream ss;
    std::string taskName = "UN";
    switch (sqeType) {
        case RT_STARS_SQE_TYPE_NOTIFY_RECORD:
            taskName = "NR"; // Notify Record
            break;
        case RT_STARS_SQE_TYPE_WRITE_VALUE:
            if (subType == RT_STARS_WRITE_VALUE_SUB_TYPE_NOTIFY_RECORD_IPC_NO_PCIE) {
                taskName = "NR";
            } else if (subType == RT_STARS_WRITE_VALUE_SUB_TYPE_EVENT_RESET) {
                taskName = "NW"; // Notify Wait
            } else if (subType == RT_STARS_WRITE_VALUE_SUB_TYPE_RDMA_DB_SEND) {
                taskName = "RS"; // Rdma Send
            }
            break;
        case RT_STARS_SQE_TYPE_NOTIFY_WAIT:
            taskName = "NW";
            break;
        case RT_STARS_SQE_TYPE_EVENT_WAIT:
            taskName = "NW";
            break;
        case RT_STARS_SQE_TYPE_SDMA:
            taskName = "SD"; // SDMA
            break;
        case RT_STARS_SQE_TYPE_COND:
            taskName = "CO";
            break;
        case RT_STARS_SQE_TYPE_PLACE_HOLDER:
            taskName = "PH";
            break;
        default:
            taskName = std::to_string(sqeType);
            break;
    }

    ss << taskName << "(";
    if (sqeContextBuffer->rtsDfxInfo[idx].remoteRank != INVALID_VALUE_RANKID) {
        ss << sqeContextBuffer->rtsDfxInfo[idx].remoteRank;
    } else {
        ss << "/";
    }
    ss << ",";
    if (sqeContextBuffer->rtsDfxInfo[idx].notifyId != INVALID_VALUE_RANKID) {
        ss << sqeContextBuffer->rtsDfxInfo[idx].notifyId;
    } else {
        ss << "/";
    }
    ss << ")";
    return ss.str();
}

void HcclCommAicpu::RecordReportStatus(dfx::ReportStatus status)
{
    std::unique_lock<std::mutex> lock(reportQueueMutex_);
    while (reportStatusQueue_.size() >= MAX_REPORT_STATUS) {
        HCCL_WARNING(
            "[HcclCommAicpu][RecordReportStatus] retry status queue reach the limit[%u], "
            "the front status[%u] is dropped.",
            MAX_REPORT_STATUS, reportStatusQueue_.front());
        reportStatusQueue_.pop();
    }
    reportStatusQueue_.push(status);
    HCCL_INFO("[HcclCommAicpu][RecordReportStatus]push[%u], retry queue size()[%u]", status, reportStatusQueue_.size());
}

void HcclCommAicpu::GetReportStatusQueue(std::queue<dfx::ReportStatus>& reportStatusQue)
{
    std::unique_lock<std::mutex> lock(reportQueueMutex_);
    std::swap(reportStatusQueue_, reportStatusQue);
}

void HcclCommAicpu::SetStreamCqeExceptionStatus(const Stream& stream, CqeExceptionStatus cqeStatus)
{
    HCCL_RUN_INFO("SetStreamCqeExceptionStatus: stream sq id %u, cqe exception %u", stream.sqId(), cqeStatus);
    auto iter = streamCqeExceptionStatus_.find(stream.sqId());
    if (iter != streamCqeExceptionStatus_.end()) {
        iter->second = cqeStatus;
    } else {
        streamCqeExceptionStatus_.insert({stream.sqId(), cqeStatus});
    }
    return;
}

CqeExceptionStatus HcclCommAicpu::GetStreamCqeExceptionStatus(const Stream& stream)
{
    auto iter = streamCqeExceptionStatus_.find(stream.sqId());
    if (iter != streamCqeExceptionStatus_.end()) {
        return iter->second;
    } else {
        return CqeExceptionStatus::kNone;
    }
}

void HcclCommAicpu::ResetStreamCqeExceptionStatus(const Stream& stream)
{
    auto iter = streamCqeExceptionStatus_.find(stream.sqId());
    if (iter != streamCqeExceptionStatus_.end()) {
        iter->second = CqeExceptionStatus::kNone;
    }
    HCCL_INFO("ResetStreamCqeExceptionStatus: stream sq id %u", stream.sqId());
    return;
}

void HcclCommAicpu::SetBSRSendOpExecException()
{
    HCCL_INFO("set send stream exec exception");
    bsrSendOpExecException_ = true;
    return;
}

void HcclCommAicpu::SetBSRRecvOpExecException()
{
    HCCL_INFO("set recv stream exec exception");
    bsrRecvOpExecException_ = true;
    return;
}

bool HcclCommAicpu::GetBSRSendOpExecException()
{
    bool ret = (GetStreamCqeExceptionStatus(bsrSendStream_) == CqeExceptionStatus::kSdmaErr) || bsrSendOpExecException_;
    HCCL_INFO(
        "GetBSRSendOpExecException: stream %u cqe status %u, send exec status %u", bsrSendStream_.sqId(),
        GetStreamCqeExceptionStatus(bsrSendStream_), bsrSendOpExecException_);
    return ret;
}

bool HcclCommAicpu::GetBSRRecvOpExecException()
{
    HCCL_INFO(
        "GetBSRRecvOpExecException: stream %u cqe status %u, recv exec status %u", bsrRecvStream_.sqId(),
        GetStreamCqeExceptionStatus(bsrRecvStream_), bsrRecvOpExecException_);
    return (GetStreamCqeExceptionStatus(bsrRecvStream_) == CqeExceptionStatus::kSdmaErr) || bsrRecvOpExecException_;
}

HcclResult HcclCommAicpu::CleanStream(Stream& stream)
{
    CHK_RET(stream.ClearLocalBuff());
    CHK_RET(UpdateSqStatus(stream));
    ResetStreamCqeExceptionStatus(stream);
    HCCL_INFO("CleanStream %u success.", stream.sqId());
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::ClearStreamCqeException(Stream& stream)
{
    HandleCqeException(stream, true);
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::ResetBSRSendOpExecException()
{
    bsrSendOpExecException_ = false;
    CHK_RET(CleanStream(bsrSendStream_));
    CHK_RET(ClearStreamCqeException(bsrSendStream_));
    HCCL_INFO("ResetBSRSendOpExecException success.");
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::ResetBSRRecvOpExecException()
{
    bsrRecvOpExecException_ = false;
    CHK_RET(CleanStream(bsrRecvStream_));
    CHK_RET(ClearStreamCqeException(bsrRecvStream_));
    HCCL_INFO("ResetBSRRecvOpExecException success.");
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::ResetBSRException()
{
    if (bsrRetryOp_ == HCCL_SEND) {
        CHK_RET(ResetBSRSendOpExecException());
        HCCL_INFO("reset batchsendrecv exception success, tag:%s, index:%u", bsrSendOpId_.tag, bsrSendOpId_.index);
    } else if (bsrRetryOp_ == HCCL_RECV) {
        CHK_RET(ResetBSRRecvOpExecException());
        HCCL_INFO("reset batchsendrecv exception success, tag:%s, index:%u", bsrRecvOpId_.tag, bsrRecvOpId_.index);
    } else {
        HCCL_INFO("reset batchsendrecv exception success, tag:%s", bsrTargetOpId_.tag);
    }
    return HCCL_SUCCESS;
}

void HcclCommAicpu::UpdateBSRRetryCnt()
{
    if (bsrRetryOp_ == HCCL_SEND) {
        bsrSendRetryCnt_++;
    } else {
        bsrRecvRetryCnt_++;
    }
    HCCL_INFO("UpdateBSRRetryCnt, SendCnt[%u], RecvCnt[%u]", bsrSendRetryCnt_, bsrRecvRetryCnt_);
    return;
}

void HcclCommAicpu::ResetBSRRetryCnt()
{
    bsrSendRetryCnt_ = 0;
    bsrRecvRetryCnt_ = 0;
    return;
}

void HcclCommAicpu::InitSendRecvOpId(const OpParam& param, HcclOpIdentifier& opId)
{
    // send算子入参中只有dst对端rank号，而recv算子入参中只有src源端rank号
    if (param.opType == HcclCMDType::HCCL_CMD_SEND) {
        opId.detRank = param.dstRank;
        opId.srcRank = topoInfo_.userRank;
    } else {
        opId.srcRank = param.srcRank;
        opId.detRank = topoInfo_.userRank;
    }
    opId.isSendRecv = true;
    HCCL_DEBUG(
        "[HcclCommAicpu][InitSendRecvOpId]src=[%u] dst=[%u] isSendRecv=[%u]", opId.srcRank, opId.detRank,
        opId.isSendRecv);
    return;
}

u32 HcclCommAicpu::HcclUpdateBatchSendRecvOpIndex(std::map<u32, u32>& bsrIndexMap, u32 peerRank)
{
    u32 ret = 0;
    auto opIndexMapIter = bsrIndexMap.find(peerRank);
    if (opIndexMapIter != bsrIndexMap.end()) {
        (opIndexMapIter->second)++;
        ret = opIndexMapIter->second;
    } else {
        bsrIndexMap.insert({peerRank, 1});
        ret = 1;
    }
    return ret;
}

u32 HcclCommAicpu::HcclUpdateBatchSendRecvOpIndex(HcclSendRecvType opType, u32 srcRank, u32 dstRank)
{
    u32 peerRank = (opType == HcclSendRecvType::HCCL_SEND) ? dstRank : srcRank;
    auto& opIndexMap = (opType == HcclSendRecvType::HCCL_SEND) ? bsrSendIndexMap_ : bsrRecvIndexMap_;

    return HcclUpdateBatchSendRecvOpIndex(opIndexMap, peerRank);
}
HcclResult
HcclCommAicpu::GetBsrTransportQpn(const HcclSendRecvItem* sendrecvPair, AlgResourceResponse& algResource, u32& qpn)
{
    CHK_PTR_NULL(sendrecvPair);
    LINK targetLink;
    u32 commIndex = 0;
    u32 remoteRank = sendrecvPair->remoteRank;
    u32 localRank = topoInfo_.userRank;
    HcclSendRecvType sendRecvType = sendrecvPair->sendRecvType;
    HCCL_DEBUG(
        "[GetBsrTransportQpn] bsrOptype =[%d], localRank=[%u] remoteRank=[%u]", sendRecvType, localRank, remoteRank);

    if ((sendRecvType == HcclSendRecvType::HCCL_SEND && remoteRank < localRank)
        || (sendRecvType == HcclSendRecvType::HCCL_RECV && remoteRank > localRank)) {
        commIndex = COMM_INDEX_0;
    } else {
        commIndex = COMM_INDEX_1;
    }
    CHK_PRT_RET(
        commIndex >= algResource.opTransportResponse[COMM_COMBINE_ORDER].size(),
        HCCL_ERROR(
            "[GetBsrTransportQpn] batchsendrecv op commIndex[%u] is larger than "
            "opTransportResponse size[%zu]",
            commIndex, algResource.opTransportResponse[COMM_COMBINE_ORDER].size()),
        HCCL_E_PARA);
    SingleSubCommTransport& commCombined
        = static_cast<SingleSubCommTransport&>(algResource.opTransportResponse[COMM_COMBINE_ORDER][commIndex]);

    CHK_PRT_RET(
        sendrecvPair->remoteRank >= commCombined.userRank2subCommRank.size(),
        HCCL_ERROR(
            "[GetBsrTransportQpn]batchsendrecv op remoteUserRank[%u] is larger than "
            "userRank2subCommRank map size[%zu]",
            sendrecvPair->remoteRank, commCombined.userRank2subCommRank.size()),
        HCCL_E_PARA);

    u32 rank = commCombined.userRank2subCommRank[sendrecvPair->remoteRank];
    CHK_PRT_RET(
        rank >= commCombined.links.size(),
        HCCL_ERROR(
            "[GetBsrTransportQpn] batchsendrecv op remoteUserRank[%u], get rank[%u],"
            "the size of combinedComm links is [%zu]",
            sendrecvPair->remoteRank, rank, commCombined.links.size()),
        HCCL_E_PARA);
    targetLink = commCombined.links[rank];

    CHK_SMART_PTR_NULL(targetLink);
    if (targetLink->GetLinkType() == LinkType::LINK_ROCE) {
        CHK_RET(targetLink->GetTransportId(qpn));
    }
    HCCL_DEBUG(
        "[HcclCommAicpu][GetBsrTransportQpn] localrank=[%u] remoteuserRank=[%u] remoteRank=[%u],sendrecvType=[%d] qpn "
        "=[%u], comindex [%u]",
        topoInfo_.userRank, sendrecvPair->remoteRank, remoteRank, sendrecvPair->sendRecvType, qpn, commIndex);
    return HCCL_SUCCESS;
}
HcclResult HcclCommAicpu::InitBatchSendRecvOpId(
    const OpParam& param, const HcclSendRecvItem* sendrecvPair, HcclOpIdentifier& opId, u32 streamId,
    AlgResourceResponse& algResource)
{
    CHK_PTR_NULL(sendrecvPair);
    if (sendrecvPair->sendRecvType == HcclSendRecvType::HCCL_RECV) {
        opId.srcRank = sendrecvPair->remoteRank;
        opId.detRank = topoInfo_.userRank;
    } else {
        opId.srcRank = topoInfo_.userRank;
        opId.detRank = sendrecvPair->remoteRank;
    }
    opId.index = HcclUpdateBatchSendRecvOpIndex(sendrecvPair->sendRecvType, opId.srcRank, opId.detRank);
    opId.isSendRecv = true;
    opId.opType = HcclCMDType::HCCL_CMD_BATCH_SEND_RECV;
    std::string sendrecvTag = param.tag + "_BSR_" + std::to_string(opId.srcRank) + "_" + std::to_string(opId.detRank);
    CHK_SAFETY_FUNC_RET(memcpy_s(opId.tag, sizeof(opId.tag), sendrecvTag.c_str(), sendrecvTag.size()));
    std::string sendrecvNewTag = param.tag + "_device";
    CHK_SAFETY_FUNC_RET(memcpy_s(opId.newTag, sizeof(opId.newTag), sendrecvNewTag.c_str(), sendrecvNewTag.size()));

    u32 qpn = 0;
    if ((opId.srcRank != opId.detRank) && (!param.BatchSendRecvDataDes.isDirectRemoteRank[sendrecvPair->remoteRank])) {
        CHK_RET(GetBsrTransportQpn(sendrecvPair, algResource, qpn));
    }

    auto& bsrinfo = opId.bsrInfo[sendrecvPair->sendRecvType];
    bsrinfo.detRank = opId.detRank;
    bsrinfo.srcRank = opId.srcRank;
    bsrinfo.index = opId.index;
    bsrinfo.streamId = streamId;
    bsrinfo.tpQpn = qpn;
    CHK_SAFETY_FUNC_RET(memcpy_s(bsrinfo.bsrTag, sizeof(bsrinfo.bsrTag), sendrecvTag.c_str(), sendrecvTag.size()));

    HCCL_INFO(
        "[HcclCommAicpu][InitBatchSendRecvOpId] tag=[%s] index=[%u] src=[%u] det=[%u] qpn =[%u]", opId.tag, opId.index,
        opId.srcRank, opId.detRank, qpn);
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::InitBatchSendRecvOpId(const OpParam& param, AlgResourceResponse& algResource)
{
    CHK_SAFETY_FUNC_RET(
        memset_s(reinterpret_cast<void*>(&bsrSendOpId_), sizeof(bsrSendOpId_), 0, sizeof(bsrSendOpId_)));
    CHK_SAFETY_FUNC_RET(
        memset_s(reinterpret_cast<void*>(&bsrRecvOpId_), sizeof(bsrRecvOpId_), 0, sizeof(bsrRecvOpId_)));

    CHK_PRT_RET(
        (algResource.slaveStreams.size() < BSR_RETRY_STREAM_NUM),
        HCCL_ERROR("in batchsendrecv op, slave stream is not enough."), HCCL_E_INTERNAL);
    bsrSendOpId_.streamId = algResource.slaveStreams[BSR_RETRY_SEND_STREAM_INDEX].id();
    bsrRecvOpId_.streamId = algResource.slaveStreams[BSR_RETRY_RECV_STREAM_INDEX].id();

    u32 iter = param.BatchSendRecvDataDes.curIterNum;
    std::vector<std::vector<HcclSendRecvItem*>>& pairs = bsrSendRecvPairs_;
    CHK_PRT_RET(
        (pairs.size() <= iter),
        HCCL_ERROR("batchsendrecv sendrecv pairs size[%u] less than or equal to curiter[%u]", pairs.size(), iter),
        HCCL_E_INTERNAL);

    for (auto& pair : pairs[iter]) {
        HcclOpIdentifier& opId = (pair->sendRecvType == HcclSendRecvType::HCCL_SEND) ? bsrSendOpId_ : bsrRecvOpId_;
        auto streamId
            = (pair->sendRecvType == HcclSendRecvType::HCCL_SEND) ? bsrSendOpId_.streamId : bsrRecvOpId_.streamId;
        CHK_RET(InitBatchSendRecvOpId(param, pair, opId, streamId, algResource));
    }

    // 补全batchsendrecv中sendrecv的bsrInfo, 两边都发生故障的时候要用
    bsrSendOpId_.bsrInfo[HCCL_RECV].index = bsrRecvOpId_.index;
    bsrSendOpId_.bsrInfo[HCCL_RECV].streamId = bsrRecvOpId_.streamId;
    bsrSendOpId_.bsrInfo[HCCL_RECV].srcRank = bsrRecvOpId_.srcRank;
    bsrSendOpId_.bsrInfo[HCCL_RECV].detRank = bsrRecvOpId_.detRank;
    bsrSendOpId_.bsrInfo[HCCL_RECV].tpQpn = bsrRecvOpId_.bsrInfo[HCCL_RECV].tpQpn;
    CHK_SAFETY_FUNC_RET(memcpy_s(
        bsrSendOpId_.bsrInfo[HCCL_RECV].bsrTag, sizeof(bsrSendOpId_.bsrInfo[HCCL_RECV].bsrTag), bsrRecvOpId_.tag,
        sizeof(bsrRecvOpId_.tag)));

    bsrRecvOpId_.bsrInfo[HCCL_SEND].index = bsrSendOpId_.index;
    bsrRecvOpId_.bsrInfo[HCCL_SEND].streamId = bsrSendOpId_.streamId;
    bsrRecvOpId_.bsrInfo[HCCL_SEND].srcRank = bsrSendOpId_.srcRank;
    bsrRecvOpId_.bsrInfo[HCCL_SEND].detRank = bsrSendOpId_.detRank;
    bsrRecvOpId_.bsrInfo[HCCL_SEND].tpQpn = bsrSendOpId_.bsrInfo[HCCL_SEND].tpQpn;
    CHK_SAFETY_FUNC_RET(memcpy_s(
        bsrRecvOpId_.bsrInfo[HCCL_SEND].bsrTag, sizeof(bsrRecvOpId_.bsrInfo[HCCL_SEND].bsrTag), bsrSendOpId_.tag,
        sizeof(bsrSendOpId_.tag)));
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::QueryBatchSendRecvPairBeginPos()
{
    // 前面已经生成 batchsendrecv 的 send & recv opid 时已经校验过slave stream num数，此处不再重复校验
    CHK_RET(QuerySqStatusByType(devId_, bsrSendStream_.sqId(), DRV_SQCQ_PROP_SQ_TAIL, bsrSendOpBeginSqePos_));
    CHK_RET(QuerySqStatusByType(devId_, bsrRecvStream_.sqId(), DRV_SQCQ_PROP_SQ_TAIL, bsrRecvOpBeginSqePos_));

    HCCL_INFO(
        "QueryBatchSendRecvPairBeginPos send sqePos[%u] recv sqePos[%u]", bsrSendOpBeginSqePos_, bsrRecvOpBeginSqePos_);
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::QueryBatchSendRecvPairEndPos()
{
    // 前面已经生成 batchsendrecv 的 send & recv opid 时已经校验过slave stream num数，此处不再重复校验
    CHK_RET(QuerySqStatusByType(devId_, bsrSendStream_.sqId(), DRV_SQCQ_PROP_SQ_TAIL, bsrSendOpEndSqePos_));
    CHK_RET(QuerySqStatusByType(devId_, bsrRecvStream_.sqId(), DRV_SQCQ_PROP_SQ_TAIL, bsrRecvOpEndSqePos_));

    HCCL_INFO("QueryBatchSendRecvPairEndPos send sqePos[%u] recv sqePos[%u]", bsrSendOpEndSqePos_, bsrRecvOpEndSqePos_);
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::CommitBSRStoredException(HcclOpExecFSM& fsmState, KfcError& errorCode)
{
    if (GetBSRSendOpExecException()) {
        bsrRetryOp_ = HCCL_SEND;
        errorCode = KfcError::kSdma;
        HCCL_INFO("CommitBSRStoredException: send stream remain retry error.");
    } else if (GetBSRRecvOpExecException()) {
        bsrRetryOp_ = HCCL_RECV;
        errorCode = KfcError::kSdma;
        HCCL_INFO("CommitBSRStoredException: recv stream remain retry error.");
    }

    u32 retryCnt = (bsrRetryOp_ == HCCL_SEND) ? bsrSendRetryCnt_ : bsrRecvRetryCnt_;
    if (errorCode != KfcError::kNone) {
        bsrTargetOpId_ = (bsrRetryOp_ == HCCL_SEND) ? bsrSendOpId_ : bsrRecvOpId_;
        HCCL_RUN_INFO(
            "CommitBSRStoredException: stored op tag %s index %u , report retry error. curSendRetryCnt[%u],"
            "curRecvRetryCnt[%u]",
            bsrTargetOpId_.tag, bsrTargetOpId_.index, bsrSendRetryCnt_, bsrRecvRetryCnt_);
        auto ret = aicpuHdc_.SetOpExecStatus(
            kfcStatusTransferD2H_, bsrTargetOpId_, KfcStatus::kStoplaunch, errorCode, retryCnt);
        HCCL_RETRY_CHK_RET_AND_TRANS_FSM(
            ret, HCCL_ERROR("SetOpExecStatus failed, ret:%u", ret), KfcError::kExec,
            HcclOpExecFSM::HCCL_OP_EXEC_FSM_ERROR);
        fsmState = HcclOpExecFSM::HCCL_OP_EXEC_FSM_STOPPING;
    } else {
        CHK_RET(UpdateOpExecStatus(fsmState, KfcStatus::kRuning, errorCode, retryCnt));
        fsmState = HcclOpExecFSM::HCCL_OP_EXEC_FSM_WAIT_END;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::GetBSRRetryOpId(const OpParam& param, HcclOpIdentifier& targetOpId)
{
    KfcCommand cmd = KfcCommand::kNone;
    CHK_RET(aicpuHdc_.GetOpExecCtrlCmd(kfcControlTransferH2D_, cmd));
    if (cmd == KfcCommand::kStopLaunch) {
        HcclOpIdentifier targetOp;
        CHK_RET(aicpuHdc_.GetOpExecCtrlTargetOp(kfcControlTransferH2D_, targetOp));
        std::string targetOpTag = std::string(reinterpret_cast<char*>(&targetOp.tag[0]));
        if (targetOpTag == std::string(reinterpret_cast<char*>(&bsrSendOpId_.tag[0]))) {
            bsrRetryOp_ = HCCL_SEND;
        } else if (targetOpTag == std::string(reinterpret_cast<char*>(&bsrRecvOpId_.tag[0]))) {
            bsrRetryOp_ = HCCL_RECV;
        } else if (targetOpTag == param.tag) {
            if (targetOp.detRank == bsrSendOpId_.detRank) {
                bsrRetryOp_ = HCCL_SEND;
            } else if (targetOp.srcRank == bsrRecvOpId_.srcRank) {
                bsrRetryOp_ = HCCL_RECV;
            } else {
                HCCL_ERROR(
                    "hccl aicpu can not retry, got stop launch command, but target op srcRank[%u] and"
                    "dstRank[%u] is not match with send (dst:%u) or recv (src:%u) op",
                    targetOp.srcRank, targetOp.detRank, bsrSendOpId_.detRank, bsrRecvOpId_.srcRank);
                return HCCL_E_INTERNAL;
            }
        } else {
            // tag 不匹配，报错退出
            HCCL_ERROR(
                "hccl aicpu can not retry, got stop launch command, but target op tag[%s] is not match with"
                "send (tag:%s) or recv (tag:%s) or batchsendrecv (tag:%s)",
                targetOpTag.c_str(), bsrSendOpId_.tag, bsrRecvOpId_.tag, param.tag.c_str());
            return HCCL_E_INTERNAL;
        }
        HCCL_RUN_INFO("hccl aicpu got command %u at op[tag: %s, index: %u].", cmd, targetOpTag.c_str(), targetOp.index);
    } else {
        if (GetBSRSendOpExecException()) {
            bsrRetryOp_ = HCCL_SEND;
        } else if (GetBSRRecvOpExecException()) {
            bsrRetryOp_ = HCCL_RECV;
        } else {
            // 其他场景，报错退出
            HCCL_ERROR("hccl aicpu find task exception, but send and recv op has no exception");
            return HCCL_E_INTERNAL;
        }
    }
    if (bsrRetryOp_ == HCCL_SEND) {
        targetOpId = bsrSendOpId_;
    } else {
        targetOpId = bsrRecvOpId_;
    }
    HCCL_RUN_INFO(
        "GetBSRRetryOpId: bsrRetryOpType %u, targetOpId: tag %s, index %u", bsrRetryOp_, targetOpId.tag,
        targetOpId.index);
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::InitExecLoop(OpParam& param, std::unique_ptr<CollExecutorBase>& executor, u32& loopNum)
{
    if ((param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) && retryEnable_) {
        CHK_RET(executor->CreatePairWiseList(
            param.BatchSendRecvDataDes.sendRecvItemsPtr, param.BatchSendRecvDataDes.itemNum));
        CHK_RET(executor->GetPairWiseList(bsrSendRecvPairs_));
        CHK_PRT_RET(
            bsrSendRecvPairs_.empty(), HCCL_ERROR("[HcclCommAicpu][InitExecLoop]batchsendrecv pairs is empty"),
            HCCL_E_INTERNAL);

        for (size_t i = 0; i < bsrSendRecvPairs_.size(); i++) {
            CHK_PRT_RET(
                (bsrSendRecvPairs_[i].size() > BSR_RETRY_SENDRECV_PAIR_NUM_MAX) || bsrSendRecvPairs_[i].empty(),
                HCCL_ERROR("batchsendrecv pairs[%u] size[%u] is out of range [1,2]", i, bsrSendRecvPairs_[i].size()),
                HCCL_E_INTERNAL);

            for (size_t j = 0; j < bsrSendRecvPairs_[i].size(); j++) {
                CHK_PTR_NULL(bsrSendRecvPairs_[i][j]);
            }

            CHK_PRT_RET(
                ((bsrSendRecvPairs_[i].size() == BSR_RETRY_SENDRECV_PAIR_NUM_MAX)
                 && (bsrSendRecvPairs_[i][BSR_RETRY_SENDRECV_PAIR_INDEX_0]->sendRecvType
                     == bsrSendRecvPairs_[i][BSR_RETRY_SENDRECV_PAIR_INDEX_1]->sendRecvType)),
                HCCL_ERROR(
                    "batchsendrecv pairs[%u] sendRecvType[%u] is same", i,
                    bsrSendRecvPairs_[i][BSR_RETRY_SENDRECV_PAIR_INDEX_0]->sendRecvType),
                HCCL_E_INTERNAL);
        }

        param.BatchSendRecvDataDes.curIterNum = 0;
        loopNum = bsrSendRecvPairs_.size();
    } else {
        loopNum = 1;
    }
    HCCL_INFO("InitExecLoop: execute loop num %u", loopNum);
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::ParseHierarchicalAlgOption(u32* ahcConfInfo)
{
    u32 algOptionSize = ahcConfInfo[TOP_HIERARCHICAL_CONF_lENGTH_INDEX];
    if (algOptionSize >= (TOP_HIERARCHICAL_CONF_SIZE - 1)) {
        HCCL_ERROR(
            "[HcclCommAicpu][ParseHierarchicalAlgOption] hierarchicalAlgOption size[%lu] exceed maxsize[%lu]",
            algOptionSize, (TOP_HIERARCHICAL_CONF_SIZE - 1));
        return HCCL_E_INTERNAL;
    }

    std::map<AHCConcOpType, TemplateType> hierarchicalAlgOption;
    for (u32 i = TOP_HIERARCHICAL_CONF_INFO_INDEX; i < (TOP_HIERARCHICAL_CONF_INFO_INDEX + algOptionSize); i++) {
        AHCConcOpType ahcConcOpType;
        ahcConcOpType.ahcLevel = static_cast<AHCLevel>(
            (ahcConfInfo[i] & TOP_HIERARCHICAL_CONF_LEVEL_LOCATION) >> TOP_HIERARCHICAL_CONF_LEVEL_SHIFT);
        ahcConcOpType.concType = static_cast<ConcType>(
            (ahcConfInfo[i] & TOP_HIERARCHICAL_CONF_CONC_TYPE_LOCATION) >> TOP_HIERARCHICAL_CONF_CONC_TYPE_SHIFT);
        ahcConcOpType.ahcOpType = static_cast<AHCOpType>(
            (ahcConfInfo[i] & TOP_HIERARCHICAL_CONF_OP_TYPE_LOCATION) >> TOP_HIERARCHICAL_CONF_OP_TYPE_SHIFT);
        TemplateType templateType = static_cast<TemplateType>(
            (ahcConfInfo[i] & TOP_HIERARCHICAL_CONF_TEMPLATE_TYPE_LOCATION)
            >> TOP_HIERARCHICAL_CONF_TEMPLATE_TYPE_SHIFT);

        hierarchicalAlgOption[ahcConcOpType] = templateType;
        HCCL_DEBUG(
            "[HcclCommAicpu][ParseHierarchicalAlgOption]: index[%u] ahcLevel[%u] concType[%u] ahcOpType[%u], "
            "templateType[%u]",
            i, ahcConcOpType.ahcLevel, ahcConcOpType.concType, ahcConcOpType.ahcOpType, templateType);
    }
    topoMatcher_->SetAHCAlgOption(hierarchicalAlgOption);
    return HCCL_SUCCESS;
}

void HcclCommAicpu::PollCqeException(
    hccl::Stream& stream, bool isReadClear, rtLogicCqReport_t& cqeException, CqeStatus& cqeStatus)
{
    const HcclComStreamInfo& streamInfo = stream.GetHcclStreamInfo();
    // 以下条件满足其中之一，进行Poll CQE：1、AICPU/Custom读清CQ   2、AICPU进程读取CQ信息
    bool isPollCqe = true;
    while (isPollCqe) {
        LogControl logControl(false, retryEnable_); // 使能重执行场景，修改ERROR->RUN_WARNING，析构时自动恢复
        CqeQueryInput cqeQueryInput;
        dfx_tracer::ExecutorTracer::SetCqeQueryInput(GetDevId(), streamInfo, cqeQueryInput);
        constexpr u32 reportSize = 256;
        rtLogicCqReport_t streamReport[reportSize];
        cqeQueryInput.cqeAddr = reinterpret_cast<uint8_t*>(streamReport); // 用于存放接收到的cq
        cqeStatus = CqReportRecv(cqeQueryInput, cqeException);
        isPollCqe = (cqeStatus == dfx::CqeStatus::kCqeException && isReadClear); // 读清CQ场景，继续查询
    }
}

void HcclCommAicpu::ExchangeCqeContext(
    hccl::Stream& stream, rtLogicCqReport_t& cqeException, CqeStatus& cqeStatus, ErrCqeContext& cqeCtx)
{
    HcclResult ret = HCCL_SUCCESS;
    // aicpu进程把读取到的异常cq，写入共享内存
    if (cqeStatus == dfx::CqeStatus::kCqeException) {
        ret = stream.SetCqeContext(ErrCqeContext(
            static_cast<u32>(cqeStatus), cqeException.taskId, cqeException.errorCode, cqeException.sqeType));
    }

    // aicpu进程和custom进程，都从共享内存中读取cqe信息
    if (ret != HCCL_SUCCESS || stream.GetCqeContext(cqeCtx) != HCCL_SUCCESS) {
        // 写入/读取共享内存失败，标记流状态，避免背景线程继续轮询这条流，导致刷屏
        SetStreamCqeExceptionStatus(stream, CqeExceptionStatus::kOther);
        HCCL_ERROR(
            "%s fail, set streamCqeExceptionStatus streamId[%u] exception status[%d]", __func__, stream.id(),
            CqeExceptionStatus::kOther);
        return;
    }
}

void HcclCommAicpu::HandleCqeException(hccl::Stream& stream, bool isReadClear)
{
    std::unique_lock<std::mutex> lock(queryCqeMutex_);

    // poll cqe信息
    rtLogicCqReport_t cqeException;
    CqeStatus cqeStatus = CqeStatus::kDefault;
    PollCqeException(stream, isReadClear, cqeException, cqeStatus);

    // 以下两种情况直接返回，不处理cq信息：1、读清CQ场景只读取不处理 2、本地记录流上已经有异常信息(避免刷屏)
    if (isReadClear || GetStreamCqeExceptionStatus(stream) != CqeExceptionStatus::kNone) {
        return;
    }

    // aicpu和custom同步异常cqe信息，并记录在本地
    ErrCqeContext cqeCtx;
    ExchangeCqeContext(stream, cqeException, cqeStatus, cqeCtx);
    cqeStatus = static_cast<dfx::CqeStatus>(cqeCtx.cqeStatus);

    // 处理流上的异常cq
    if (cqeStatus != dfx::CqeStatus::kDefault) {
        bool isSdmaTypeErr = cqeStatus == dfx::CqeStatus::kCqeException && cqeCtx.sqeType == RT_STARS_SQE_TYPE_SDMA;
        bool isCompDataErr = cqeCtx.errorCode == RT_SDMA_COMPDATAERR || cqeCtx.errorCode == RT_SDMA_COMPERR
                             || cqeCtx.errorCode == RT_SDMA_DATAERR;
        bool isSdmaCompDataErr = isSdmaTypeErr && isCompDataErr;

        ReportErrCqe(stream, cqeCtx);

        // 标记发生ErrCqe的流
        hccl::CqeExceptionStatus cqeExceptionStatus
            = isSdmaCompDataErr ? hccl::CqeExceptionStatus::kSdmaErr : hccl::CqeExceptionStatus::kOther;
        SetStreamCqeExceptionStatus(stream, cqeExceptionStatus);

        // 通知kfc线程
        dfxExtendInfo_.cqeException.sqeType = cqeCtx.sqeType;
        dfxExtendInfo_.cqeException.errorCode = cqeCtx.errorCode;
        dfxExtendInfo_.cqeStatus = cqeStatus;
        dfxExtendInfo_.pollStatus = PollStatus::kStopAsException;
        HCCL_INFO(
            "update dfxExtendInfo, group %s, streamId %u, cqeStatus %d, sqetype %u, errorCode %u", identifier_.c_str(),
            stream.id(), cqeStatus, cqeCtx.sqeType, cqeCtx.errorCode);
    }
}

void HcclCommAicpu::ReportErrCqe(hccl::Stream& stream, ErrCqeContext& cqeCtx)
{
    dfx::CqeStatus cqeStatus = static_cast<dfx::CqeStatus>(cqeCtx.cqeStatus);
    const HcclComStreamInfo& streamInfo = stream.GetHcclStreamInfo();
    u32 head = 0;
    u32 tail = 0;
    QuerySqStatusByType(devId_, streamInfo.sqId, DRV_SQCQ_PROP_SQ_HEAD, head);
    QuerySqStatusByType(devId_, streamInfo.sqId, DRV_SQCQ_PROP_SQ_TAIL, tail);

    bool isSdmaTypeErr = cqeStatus == dfx::CqeStatus::kCqeException && cqeCtx.sqeType == RT_STARS_SQE_TYPE_SDMA;
    bool isCompDataErr = cqeCtx.errorCode == RT_SDMA_COMPDATAERR || cqeCtx.errorCode == RT_SDMA_COMPERR
                         || cqeCtx.errorCode == RT_SDMA_DATAERR;
    bool isSdmaCompDataErr = isSdmaTypeErr && isCompDataErr;

    // 发生sdma、notify_wait、place_holder、write_value错误，上报error message
    bool isComReportErrMesg
        = cqeStatus == dfx::CqeStatus::kCqeException
          && (cqeCtx.sqeType == RT_STARS_SQE_TYPE_SDMA || cqeCtx.sqeType == RT_STARS_SQE_TYPE_NOTIFY_WAIT
              || cqeCtx.sqeType == RT_STARS_SQE_TYPE_PLACE_HOLDER || cqeCtx.sqeType == RT_STARS_SQE_TYPE_WRITE_VALUE);

    // 使能重执行且触发SDMA ERROR的场景，修改ERROR->RUN_WARNING
    LogControl retryLog(false, retryEnable_ && isSdmaCompDataErr);

    // 只在AICPU进程上报taskException，Custom进程不上报
    if (isComReportErrMesg && errMessageReport_) {
        // 记录关键信息，并通过D2H通信通道交给host内存
        GenTaskExceptionInfo(cqeCtx.sqeType, stream, head);
        // 通知ts错误信息，触发非SDMA错误、不使能重执行触发SDMA错误、使能重执行触发无法重执行的错误时，进行故障上报
        if (!isSdmaTypeErr || (isSdmaTypeErr && (!retryEnable_ || !isCompDataErr))) {
            HcclResult ret = SendTaskExceptionByMBox(cqeCtx.errorCode);
            HCCL_RUN_INFO(
                "[HcclCommAicpu][SendTaskExceptionByMBox]group[%s]:"
                "Try to send task exception by mailbox, errType[%u], errCode[%u], streamId[%d]",
                identifier_.c_str(), cqeCtx.sqeType, cqeCtx.errorCode, stream.id());
            CHK_PRT_CONT(
                ret != HCCL_SUCCESS,
                HCCL_ERROR(
                    "[OpRetry][AICPU]group[%s]:Send task exception by mailBox failed, streamId[%d]",
                    identifier_.c_str(), stream.id()));
        }
        // 当前阶段， 每个通信域在plog打印一次，error message作为host上报，只打印首次
        errMessageReport_ = false;
    }

    // 发生ErrCqe之后，统一在AICPU进程上报taskException
    if (cqeStatus == dfx::CqeStatus::kCqeException) {
        if (IsNoNeedWait() && cqeCtx.sqeType == RT_STARS_SQE_TYPE_PLACE_HOLDER) {
            PrintTaskExceptionAllComm(); // 超时场景打印所有通信域的taskException
            PrintAicpuCommExecStatus();
        } else if (IsNoNeedWait()) {
            ErrCqeContext cqeCtx;
            stream.GetCqeContext(cqeCtx);
            PrintTaskExceptionByTaskId(cqeCtx.sqeType, cqeCtx.taskId, stream, tail); // 仅打印本条流的taskException
        }
    }

    CHK_PRT_CONT(
        !retryEnable_ && isSdmaCompDataErr,
        HCCL_RUN_INFO(
            "[OpRetry][AICPU]group[%s] hccl aicpu can not retry, retryEnable is false.", identifier_.c_str()));

    CHK_PRT_CONT(
        retryEnable_ && isSdmaTypeErr && !isCompDataErr,
        HCCL_RUN_INFO(
            "[OpRetry][AICPU]group[%s] hccl aicpu can not retry, errCode is [%u].", identifier_.c_str(),
            cqeCtx.errorCode));

    HCCL_ERROR(
        "Exception happened, group %s, sqid %d, cqeStatus %d, sqetype %u, errorCode %u, head %u, tail %u",
        identifier_.c_str(), streamInfo.sqId, cqeStatus, cqeCtx.sqeType, cqeCtx.errorCode, head, tail);
}

HcclResult HcclCommAicpu::SendTaskExceptionByMBox(const uint16_t& rsErrorCode)
{
    HcclSignalInfo notifyInfo;
    opNotifies_[1]->GetNotifyData(notifyInfo);

    HCCL_INFO("[HcclCommAicpu][SendTaskExceptionByMBox] HostToDeviceLogicId[%u]", devId_);
    CHK_RET(hccl_plf::SendTaskExceptionByMBox(
        devId_, opNotifies_[1]->notifyId_, notifyInfo.tsId, userStreamId_, rsErrorCode));
    return HCCL_SUCCESS;
}

void HcclCommAicpu::HandleIndOpCqe()
{
    std::unique_lock<std::mutex> lock(queryCqeMutex_);
    std::shared_lock<std::shared_mutex> rwLock(threadAicpuMutex_);
    for (auto& thread : threads_) {
        if (thread == nullptr) {
            continue;
        }
        Stream stream = *thread->GetStream();
        // 流上已有异常信息，不再重复读取
        if (GetStreamCqeExceptionStatus(stream) != CqeExceptionStatus::kNone) {
            continue;
        }

        // poll cqe信息
        rtLogicCqReport_t cqeException;
        CqeQueryInput cqeQueryInput;
        dfx_tracer::ExecutorTracer::SetCqeQueryInput(GetDevId(), stream.GetHcclStreamInfo(), cqeQueryInput);
        constexpr u32 reportSize = 256;
        rtLogicCqReport_t streamReport[reportSize];
        cqeQueryInput.cqeAddr = reinterpret_cast<uint8_t*>(streamReport); // 用于存放接收到的cq
        CqeStatus cqeStatus = CqReportRecv(cqeQueryInput, cqeException);
        // 未读取到异常信息，返回
        if (cqeStatus == dfx::CqeStatus::kDefault) {
            continue;
        }
        ReportIndOpCqe(stream, cqeException, cqeStatus);
    }
}

void HcclCommAicpu::ReportIndOpCqe(hccl::Stream& stream, const rtLogicCqReport_t& cqeException, CqeStatus cqeStatus)
{
    const HcclComStreamInfo& streamInfo = stream.GetHcclStreamInfo();
    u32 head = 0;
    u32 tail = 0;
    QuerySqStatusByType(devId_, streamInfo.sqId, DRV_SQCQ_PROP_SQ_HEAD, head);
    QuerySqStatusByType(devId_, streamInfo.sqId, DRV_SQCQ_PROP_SQ_TAIL, tail);

    // 打印taskException信息
    if (cqeStatus == dfx::CqeStatus::kCqeException) {
        if (cqeException.sqeType == RT_STARS_SQE_TYPE_PLACE_HOLDER) {
            PrintTaskExceptionAllComm(); // 超时场景打印所有通信域的taskException
            PrintAicpuCommExecStatus();
        } else {
            taskExecption_.PrintTaskExceptionByTaskId(cqeException.sqeType, cqeException.taskId, stream, tail);
        }
    }

    // 打印重执行提示信息
    bool isSdmaTypeErr = cqeStatus == dfx::CqeStatus::kCqeException && cqeException.sqeType == RT_STARS_SQE_TYPE_SDMA;
    bool isCompDataErr = cqeException.errorCode == RT_SDMA_COMPDATAERR || cqeException.errorCode == RT_SDMA_COMPERR;
    bool isSdmaCompDataErr = isSdmaTypeErr && isCompDataErr;
    if (retryEnable_ && isSdmaCompDataErr) {
        uint16_t rsErrorCode = TS_ERROR_RETRY_CONSTRAINT;
        CHK_PRT(SendTaskExceptionByMBox(rsErrorCode));
        HCCL_RUN_INFO("[OpRetry][AICPU]group[%s] can not retry, IndOp does not support opRetry", identifier_.c_str());
    }

    // 标记发生ErrCqe的流
    hccl::CqeExceptionStatus cqeExceptionStatus
        = isSdmaCompDataErr ? hccl::CqeExceptionStatus::kSdmaErr : hccl::CqeExceptionStatus::kOther;
    SetStreamCqeExceptionStatus(stream, cqeExceptionStatus);

    HCCL_ERROR(
        "Exception happened, group %s, streamId %u, sqid %d, cqeStatus %d, sqetype %u, errorCode %u, "
        "head %u, tail %u",
        identifier_.c_str(), stream.id(), streamInfo.sqId, cqeStatus, cqeException.sqeType, cqeException.errorCode,
        head, tail);
}

HcclResult HcclCommAicpu::RefreshLinkForSwitchNic(
    const std::string& newTag, const TransportRequest& transportRequest, const std::map<u32, bool>& remoteRankPortMap,
    bool isSecondBuild, LINK& switchLink)
{
    u32 remoteRankId = transportRequest.remoteUserRank;
    auto iterRemoteRank = remoteRankPortMap.find(remoteRankId);
    bool needSwitch = iterRemoteRank != remoteRankPortMap.end();
    bool isBackup = needSwitch && !(iterRemoteRank->second);
    HCCL_INFO(
        "[HcclCommAicpu][%s] newTag[%s], localRank[%u], remoteRank[%u], input memory type[%u], "
        "output memory type[%u], isRdma[%u], isSecondBuild[%u], needSwitch[%u], isBackup[%u].",
        __func__, newTag.c_str(), localUserRank_, remoteRankId, transportRequest.inputMemType,
        transportRequest.outputMemType, transportRequest.isUsedRdma, isSecondBuild, needSwitch, isBackup);

    if (transportRequest.isUsedRdma && needSwitch) {
        auto* linkRes = isBackup ? &linkRdmaResBackUp_ : &linkRdmaRes_;
        auto iterRankLinks = linkRes->find(remoteRankId);
        CHK_PRT_RET(
            iterRankLinks == linkRes->end(),
            HCCL_ERROR(
                "[HcclCommAicpu][%s] comm[%s], local rank[%u], fail to find relative link for remote rank[%u], "
                "isBackup[%u]",
                __func__, identifier_.c_str(), localUserRank_, remoteRankId, isBackup),
            HCCL_E_INTERNAL);
        auto iterRankTagLinks = iterRankLinks->second.find(newTag);
        CHK_PRT_RET(
            iterRankTagLinks == iterRankLinks->second.end(),
            HCCL_ERROR(
                "[HcclCommAicpu][%s] comm[%s], local rank[%u], "
                "fail to find relative tag[%s] of links for remote rank[%u], isBackup[%u]",
                __func__, identifier_.c_str(), localUserRank_, newTag.c_str(), remoteRankId, isBackup),
            HCCL_E_INTERNAL);
        CHK_PRT_RET(
            isSecondBuild && iterRankLinks->second[newTag].size() < 2U,
            HCCL_ERROR(
                "[HcclCommAicpu][%s] comm[%s], local rank[%u], isSecondBuild[%u], "
                "fail to find second link for remote rank[%u]",
                __func__, identifier_.c_str(), localUserRank_, isSecondBuild, remoteRankId),
            HCCL_E_INTERNAL);

        switchLink = isSecondBuild ? iterRankLinks->second[newTag][1] : iterRankLinks->second[newTag][0];
        CHK_SMART_PTR_NULL(switchLink);

        auto iterAck = receivedAcks_.find(remoteRankId);
        CHK_PRT_RET(
            iterAck == receivedAcks_.end(), HCCL_ERROR("[%s]there is no link with rankId[%u]", __func__, remoteRankId),
            HCCL_E_NOT_FOUND);
        switchLink->SetSupportDataReceivedAck(iterAck->second);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::ReAllocTransportForSwitchNic(
    const std::string& newTag, AlgResourceResponse& algResResponse, std::map<u32, bool>& remoteRankPortMap)
{
    HCCL_INFO(
        "[HcclCommAicpu][%s] Entry realloc transport for switch nic, comm identifier[%s], localRank[%u], tag[%s]",
        __func__, identifier_.c_str(), localUserRank_, newTag.c_str());
    std::set<u32> bsrTansportRank;
    for (auto& levelNSubCommTransport : algResResponse.opTransportResponse) {
        for (auto& singleSubCommTransport : levelNSubCommTransport) {
            for (size_t i = 0; i < singleSubCommTransport.transportRequests.size(); i++) {
                auto& transportRequest = singleSubCommTransport.transportRequests[i];
                if (transportRequest.isValid) {
                    receivedAcks_[transportRequest.remoteUserRank] = singleSubCommTransport.supportDataReceivedAck;
                    bool isSecondBuild = false;
                    if (transportRequest.isUsedRdma && newTag.find("BatchSendRecv") != std::string::npos
                        && bsrTansportRank.find(transportRequest.remoteUserRank) != bsrTansportRank.end()) {
                        // 仅在batchsendrecv rdma下发的时候需要第二次刷新，实际第一次下发都刷好了，第二次就是get一下
                        isSecondBuild = true;
                    }
                    bsrTansportRank.insert(transportRequest.remoteUserRank);
                    CHK_RET(RefreshLinkForSwitchNic(
                        newTag, transportRequest, remoteRankPortMap, isSecondBuild, singleSubCommTransport.links[i]));
                }
            }
        }
    }

    return HCCL_SUCCESS;
}

HcclResult
HcclCommAicpu::RefreshRoceTransportsForSwitchNic(std::unordered_map<std::string, OpCommTransport>& reservedLinks)
{
    ChangeLinkInfo changeLinkInfo;
    HcclResult ret = LoadChangeLinkInfo(changeLinkInfo);

    CHK_PRT_RET(
        changeLinkInfo.isChangeLinkFlag == false,
        HCCL_ERROR(
            "[HcclCommAicpu][%s] some error happened on host, switch nic has failed on rank[%u]. "
            "The error message will be broadcast to other ranks in comm[%s].",
            __func__, localUserRank_, identifier_.c_str()),
        HCCL_E_PARA);

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[HcclCommAicpu][%s] comm identifier[%s], rank[%u], "
            "load change link info failed.",
            __func__, identifier_.c_str(), localUserRank_),
        ret);

    std::map<u32, bool> remoteRankPortMap;
    for (u32 i = 0; i < changeLinkInfo.remoteRankNum; i++) {
        remoteRankPortMap.emplace(changeLinkInfo.remoteRankList[i], changeLinkInfo.isUseDefaultPort[i]);
        HCCL_INFO(
            "[HcclCommAicpu][%s] comm identifier[%s], rank[%u], remote rank[%u], isUseDefaultPort[%u].", __func__,
            identifier_.c_str(), localUserRank_, changeLinkInfo.remoteRankList[i], changeLinkInfo.isUseDefaultPort[i]);
    }

    // 对resMap中所有tag的transport link根据主备进行刷新
    for (auto& resMapIt : resMap_) {
        HCCL_RUN_INFO(
            "[HcclCommAicpu][%s] rank[%u] refresh algResResponse of tag[%s].", __func__, localUserRank_,
            resMapIt.first.c_str());
        reservedLinks.emplace(resMapIt.first, resMapIt.second.opTransportResponse);
        CHK_RET(ReAllocTransportForSwitchNic(resMapIt.first, resMapIt.second, remoteRankPortMap));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::RevertTransportsForSwitchNic(std::unordered_map<std::string, OpCommTransport>& reservedLinks)
{
    if (reservedLinks.size() == 0) {
        HCCL_INFO(
            "[HcclCommAicpu][%s] comm identifier[%s], rank[%u], reserved link is empty, "
            "no need to revert transport.",
            __func__, identifier_.c_str(), localUserRank_);
        return HCCL_SUCCESS;
    }
    for (auto& resMapIt : resMap_) {
        HCCL_RUN_INFO(
            "[HcclCommAicpu][%s] comm identifier[%s], rank[%u], revert transport in algResResponse of tag[%s].",
            __func__, identifier_.c_str(), localUserRank_, resMapIt.first.c_str());
        auto linkIt = reservedLinks.find(resMapIt.first);
        if (linkIt == reservedLinks.end()) {
            HCCL_RUN_WARNING(
                "[HcclCommAicpu][%s] comm identifier[%s], rank[%u], fail to find transport to revert.", __func__,
                identifier_.c_str(), localUserRank_);
            continue;
        }
        resMapIt.second.opTransportResponse = linkIt->second;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::SwitchNicWaitHandleCommand(std::unordered_map<std::string, OpCommTransport>& reservedLinks)
{
    auto waitSwitchCmdTimeoutMs = HcclGetWaitRetryCmdTimeout(0);
    auto waitSwitchCmdTimeout = std::chrono::milliseconds(waitSwitchCmdTimeoutMs);

    HcclResult ret = HCCL_E_INTERNAL;
    auto startTime = std::chrono::steady_clock::now();
    while (true) {
        KfcCommand switchHandleCmd = KfcCommand::kNone;
        CHK_RET(aicpuHdc_.GetOpExecCtrlCmd(kfcControlTransferH2D_, switchHandleCmd));
        if (switchHandleCmd == KfcCommand::kWaitSwitchNic) {
            HCCL_INFO(
                "[HcclCommAicpu][%s] comm identifier[%s], rank[%u], hccl aicpu recv switch nic handle command.",
                __func__, identifier_.c_str(), localUserRank_);
            ret = aicpuHdc_.SetOpExecStatus(kfcStatusTransferD2H_, KfcStatus::kWaitSwitchRes, KfcError::kNone, 0);
            if (ret != HCCL_SUCCESS) {
                (void)RevertTransportsForSwitchNic(reservedLinks);
                return ret;
            }
            return HCCL_SUCCESS;
        } else if ((std::chrono::steady_clock::now() - startTime) >= waitSwitchCmdTimeout) {
            HCCL_ERROR(
                "[HcclCommAicpu][%s] comm identifier[%s], rank[%u], "
                "hccl aicpu wait switch nic handle command timeout[%u ms], local nic switch is reverted.",
                __func__, identifier_.c_str(), localUserRank_, waitSwitchCmdTimeoutMs);
            (void)RevertTransportsForSwitchNic(reservedLinks);
            return HCCL_E_TIMEOUT;
        }
    }

    return ret;
}

HcclResult HcclCommAicpu::SwitchNicWaitResult(std::unordered_map<std::string, OpCommTransport>& reservedLinks)
{
    auto waitSwitchResultTimeoutSecond = GetExternalInputHcclLinkTimeOut() * 2U;
    const auto waitSwitchResultTimeout = std::chrono::seconds(waitSwitchResultTimeoutSecond);

    auto startTime = std::chrono::steady_clock::now();
    while (true) {
        KfcCommand switchResultCmd = KfcCommand::kNone;
        CHK_RET(aicpuHdc_.GetOpExecCtrlCmd(kfcControlTransferH2D_, switchResultCmd));
        if (switchResultCmd == KfcCommand::kAllSwitched) {
            HCCL_INFO(
                "[HcclCommAicpu][%s] comm identifier[%s], rank[%u], "
                "hccl aicpu switch nic success, all nic switch transport.",
                __func__, identifier_.c_str(), localUserRank_);
            return HCCL_SUCCESS;
        } else if (switchResultCmd == KfcCommand::kSwitchFail) {
            HCCL_ERROR(
                "[HcclCommAicpu][%s] comm identifier[%s], rank[%u], "
                "hccl aicpu comm switch nic error, local nic switch is reverted.",
                __func__, identifier_.c_str(), localUserRank_);
            (void)RevertTransportsForSwitchNic(reservedLinks);
            return HCCL_E_INTERNAL;
        } else if ((std::chrono::steady_clock::now() - startTime) >= waitSwitchResultTimeout) {
            HCCL_ERROR(
                "[HcclCommAicpu][%s] comm identifier[%s], rank[%u], "
                "hccl aicpu wait switch nic result timeout[%u s].",
                __func__, identifier_.c_str(), localUserRank_, waitSwitchResultTimeoutSecond);
            (void)RevertTransportsForSwitchNic(reservedLinks);
            return HCCL_E_TIMEOUT;
        }
    }

    return HCCL_E_INTERNAL;
}

HcclResult HcclCommAicpu::SwitchNic()
{
    KfcStatus state = KfcStatus::kSwitchError;
    std::unordered_map<std::string, OpCommTransport> reservedLinks;
    HcclResult ret = RefreshRoceTransportsForSwitchNic(reservedLinks);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcclCommAicpu][%s] comm identifier[%s], rank[%u], refresh roce transports failed.", __func__,
            identifier_.c_str(), localUserRank_);
        state = KfcStatus::kSwitchError;
        ret = aicpuHdc_.SetOpExecStatus(kfcStatusTransferD2H_, state, KfcError::kNone, 0);
    } else {
        HCCL_INFO(
            "[HcclCommAicpu][%s] comm identifier[%s], rank[%u], refresh roce transports success.", __func__,
            identifier_.c_str(), localUserRank_);
        state = KfcStatus::kPlanSwitch;
        ret = aicpuHdc_.SetOpExecStatus(kfcStatusTransferD2H_, state, KfcError::kNone, 0);
    }

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcclCommAicpu][SwitchNic] comm identifier[%s], rank[%u], "
            "send switch status[%u] to host fail.",
            __func__, identifier_.c_str(), localUserRank_, state);
    }

    CHK_RET(SwitchNicWaitHandleCommand(reservedLinks));

    return SwitchNicWaitResult(reservedLinks);
}

HcclResult HcclCommAicpu::ResumeChangeLink()
{
    ChangeLinkInfo changeLinkInfo;
    HcclResult ret = LoadChangeLinkInfo(changeLinkInfo);
    bool useBackupLink = false;
    std::map<u32, bool> remoteRankPortMap;
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcclCommAicpu][%s] comm identifier[%s], rank[%u], load change link info failed.", __func__,
            identifier_.c_str(), localUserRank_);
        return ret;
    }
    for (u32 i = 0; i < changeLinkInfo.remoteRankNum; i++) {
        remoteRankPortMap.insert({changeLinkInfo.remoteRankList[i], changeLinkInfo.isUseDefaultPort[i]});
        useBackupLink |= (!changeLinkInfo.isUseDefaultPort[i]);
    }
    ret = RefreshCommResponseTransportRes(remoteRankPortMap);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcclCommAicpu][%s] comm identifier[%s], rank[%u], refresh roce transports failed.", __func__,
            identifier_.c_str(), localUserRank_);
        return ret;
    } else {
        HCCL_INFO(
            "[HcclCommAicpu][%s] comm identifier[%s], rank[%u], refresh roce transports success.", __func__,
            identifier_.c_str(), localUserRank_);
        return HCCL_SUCCESS;
    }
}

HcclResult HcclCommAicpu::RefreshCommResponseTransportRes(std::map<u32, bool>& remoteRankPortMap)
{
    PetersonLockGuard guard(hostDeviceLock_.get());
    CHK_PRT_RET(guard.IsLockFailed(), HCCL_ERROR("[%s] hostDeviceLock lock failed", __func__), HCCL_E_INTERNAL);
    OpParam param;
    // 提前清理所有tag的链路，避免冲突
    for (auto& resMapIt : resMap_) {
        HCCL_RUN_INFO("[%s] clean roce resource of tag[%s].", __func__, resMapIt.first.c_str());
        CleanRoceResource(resMapIt.first, resMapIt.second, remoteRankPortMap, param);
    }
    for (auto& resMapIt : resMap_) {
        HCCL_RUN_INFO("[%s] refresh algResResponse of tag[%s].", __func__, resMapIt.first.c_str());
        CHK_RET(ReAllocTransportResource(resMapIt.first, resMapIt.second, remoteRankPortMap, commParam_, param));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::InitAicpuIndOp(CommAicpuParam* commAicpuParam)
{
    if (indOpCommInitialized_) {
        HCCL_RUN_INFO("[%s][InitAicpuIndOp]Group[%s] already initialized, skip reinit", __func__, identifier_.c_str());
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(commAicpuParam);
    topoInfo_.deviceLogicId = commAicpuParam->deviceLogicId;
    topoInfo_.devicePhyId = commAicpuParam->devicePhyId;
    topoInfo_.deviceType = static_cast<DevType>(commAicpuParam->deviceType);
    identifier_ = std::string(commAicpuParam->hcomId);
    topoInfo_.userRankSize = commAicpuParam->userRankSize;
    topoInfo_.userRank = commAicpuParam->userRank;
    notifys_.reserve(hccl::HCCL_THREAD_NOTIFY_MAX_NUM);
    if (topoInfo_.deviceType == DevType::DEV_TYPE_910_93 || topoInfo_.deviceType == DevType::DEV_TYPE_910B) {
        notifySize_ = NOTIFY_SIZE_FOUR;
    } else {
        notifySize_ = NOTIFY_SIZE_EIGHT;
    }

    CHK_RET(hrtSetWorkModeAicpu(true));
    CHK_RET(hrtSetlocalDevice(topoInfo_.deviceLogicId));
    CHK_RET(hrtSetlocalDeviceType(topoInfo_.deviceType));
    CHK_RET(hrtDrvGetLocalDevIDByHostDevID(topoInfo_.devicePhyId, &devId_));
    CHK_RET(taskExecption_.Init(devId_, topoInfo_.userRank, identifier_));
    CHK_RET(RegisterProfCallBack());

    if (topoInfo_.deviceType == DevType::DEV_TYPE_950 || topoInfo_.deviceType == DevType::DEV_TYPE_960) {
        HCCL_INFO("[HcclCommAicpu][InitAicpuIndOp] InitAicpuIndOpV2 start");
        indOpCommInitialized_ = true;
        return HCCL_SUCCESS;
    }

    HCCL_INFO("[HcclCommAicpu][InitAicpuIndOp] InitAicpuIndOp start");
    if (!FindDispatcherByCommId(&dispatcherCtx_, identifier_.c_str())) {
        CHK_RET(CreateDispatcherCtx(&dispatcherCtx_, devId_, identifier_.c_str()));
    }
    CHK_PTR_NULL(dispatcherCtx_);
    hccl::DispatcherCtx* Ctx_temp = static_cast<DispatcherCtx*>(dispatcherCtx_);
    HCCL_INFO("[%s] Ctx_temp[%p]", __func__, (void*)Ctx_temp);
    (void)RegisterLoadTaskCallBack(Ctx_temp->GetDispatcher(), nullptr, dfx::TaskProfilingCallBack); // 注册dispatcher
    if (commAicpuParam->kfcControlTransferH2DParams.buffLen != 0 && kfcControlTransferH2D_ == nullptr) {
        EXCEPTION_CATCH((kfcControlTransferH2D_ = std::make_shared<hccl::HDCommunicate>()), return HCCL_E_PTR);
        CHK_SMART_PTR_NULL(kfcControlTransferH2D_);
        CHK_RET(kfcControlTransferH2D_->InitDevice(commAicpuParam->kfcControlTransferH2DParams));
    }
    if (commAicpuParam->kfcStatusTransferD2HParams.buffLen != 0 && kfcStatusTransferD2H_ == nullptr) {
        EXCEPTION_CATCH((kfcStatusTransferD2H_ = std::make_shared<hccl::HDCommunicate>()), return HCCL_E_PTR);
        CHK_SMART_PTR_NULL(kfcStatusTransferD2H_);
        CHK_RET(kfcStatusTransferD2H_->InitDevice(commAicpuParam->kfcStatusTransferD2HParams));
    }

    indOpCommInitialized_ = true;

    // 在indOpCommInitialized_变为true后拉起背景线程
    AicpuComContext* ctx = AicpuGetComContext();
    AicpuHcclProcess::CallMC2MaintenanceThread(ctx);

    HCCL_RUN_INFO(
        "%s group[%s] success!, deviceLogicId[%u], devicePhyId[%u], deviceType[%u], notifySize[%u], "
        "dispatcherCtx[%p]",
        __func__, identifier_.c_str(), topoInfo_.deviceLogicId, topoInfo_.devicePhyId, topoInfo_.deviceType,
        notifySize_, dispatcherCtx_);

    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::InitThreads(ThreadMgrAicpuParam* param)
{
    CHK_PTR_NULL(param);
    u32 threadNum = param->threadNum;
    std::vector<std::shared_ptr<Thread>> outThreads;
    outThreads.reserve(threadNum);
    std::string hcomId(param->hcomId);
    for (u32 i = 0; i < threadNum; ++i) {
        std::string thdUniqueId(param->threadParam[i], THREAD_UNIQUE_ID_MAX_SIZE);
        if (UNLIKELY(HcclCheckLogLevel(HCCL_LOG_INFO))) {
            std::ostringstream oss;
            oss << "threadParam[" << i << "] raw bytes: ";
            for (u32 j = 0; j < THREAD_UNIQUE_ID_MAX_SIZE; ++j) {
                oss << std::hex << std::setw(2) << std::setfill('0')
                    << static_cast<unsigned int>(static_cast<unsigned char>(param->threadParam[i][j])) << " ";
            }
            HCCL_INFO("[HcclCommAicpu][%s] %s", __func__, oss.str().c_str());
        }
        std::shared_ptr<AicpuTsThread> thread;
        EXCEPTION_CATCH((thread = std::make_shared<AicpuTsThread>(thdUniqueId)), return HCCL_E_PTR);
        thread->SetCommEngine(param->engine);
        HcclResult ret = thread->Init();
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[HcclCommAicpu][%s] comm identifier[%s], init threads num[%u] failed at index %u", __func__,
                hcomId.c_str(), param->threadNum, i);
            return ret;
        }
        outThreads.emplace_back(thread);
    }

    ThreadHandle* threadArray = static_cast<ThreadHandle*>(param->deviceHandle);
    // 空指针校验
    CHK_PTR_NULL(threadArray);
    for (size_t i = 0; i < threadNum; ++i) {
        threadArray[i] = reinterpret_cast<ThreadHandle>(outThreads[i].get()); // 拷贝裸指针
        HCCL_INFO("[HcclCommAicpu][%s] threadArray[%u] = [%lu]", __func__, i, threadArray[i]);
    }
    std::unique_lock<std::shared_mutex> rwLock(threadAicpuMutex_);
    threads_.insert(
        threads_.end(), std::make_move_iterator(outThreads.begin()), std::make_move_iterator(outThreads.end()));
    HCCL_INFO(
        "[HcclCommAicpu][%s] comm identifier[%s], init threads num[%u] success", __func__, hcomId.c_str(), threadNum);
    // 为上报翻转初始化资源
    if (topoInfo_.deviceType != DevType::DEV_TYPE_950 && topoInfo_.deviceType != DevType::DEV_TYPE_960) {
        CHK_RET(InitProfthreadResource(threadNum));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::InitProfthreadResource(u32 threadNum)
{
    groupHashId_ = dfx::ProfilingManager::GetProfHashId(identifier_.c_str(), identifier_.length());
    HCCL_INFO(
        "[%s], group[%s], groupHash[%llu], threadNum[%u] ", __func__, identifier_.c_str(), groupHashId_, threadNum);
    dfx::ProfCommInfo profInfo{groupHashId_, topoInfo_.userRankSize, topoInfo_.userRank};
    // 添加检查确保 threadNum 不超过线程总数
    if (threadNum > threads_.size()) {
        HCCL_ERROR("[%s] threadNum Err", __func__);
        return HCCL_E_PARA;
    }
    // 从后往前迭代指定数量
    auto begin = threads_.rbegin();
    auto end = begin + threadNum;
    for (auto it = begin; it != end; ++it) {
        CHK_RET(dfx::ProfilingManager::AddProfInfoByStreamId((*it)->GetStream()->id(), identifier_, profInfo));
    }
    dfx::ProfilingExtendInfoHelper::InitProfItemId();
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::AllocChannelResource(HcclIndOpChannelRemoteResV3* commParam)
{
    CHK_PTR_NULL(commParam);
    if (commParam->engine != COMM_ENGINE_AICPU && commParam->engine != COMM_ENGINE_AICPU_TS) {
        HCCL_ERROR(
            "[HcclCommAicpu][%s] engine type[%s] is not supported", __func__,
            GetEnumToString(GetCommEngineStatusStrMap(), commParam->engine).c_str());
        return HCCL_E_PARA;
    }

    if (commParam->listNum != 0 && commParam->remoteResV2 == nullptr) {
        HCCL_ERROR("[HcclCommAicpu][%s] listNum[%u], remoteResV2 is nullptr", __func__, commParam->listNum);
        return HCCL_E_PARA;
    }

    multiQpThreshold_ = commParam->multiQpThreshold;
    localUserRank_ = commParam->localUserRank;
    HCCL_INFO(
        "%s multiQpThreshold[%u], localUserRank[%u], deviceLogicId[%d], devicePhyId[%u], deviceType[%d], "
        "listNum[%u]",
        __func__, multiQpThreshold_, localUserRank_, topoInfo_.deviceLogicId, topoInfo_.devicePhyId,
        topoInfo_.deviceType, commParam->listNum);
    for (u32 idx = 0; idx < commParam->listNum; idx++) {
        HCCL_INFO(
            "%s listNum[%u], listIdx[%u], remoteWorldRank[%u], remoteRank[%u], isUsedRdma[%d]", __func__,
            commParam->listNum, idx, commParam->remoteResV2[idx].remoteWorldRank,
            commParam->remoteResV2[idx].remoteRank, commParam->remoteResV2[idx].isUsedRdma);

        rankData_[commParam->remoteResV2[idx].remoteRank].remoteWorldRank = commParam->remoteResV2[idx].remoteWorldRank;
        rankData_[commParam->remoteResV2[idx].remoteRank].remoteUsrRankId = commParam->remoteResV2[idx].remoteRank;
        if (commParam->remoteResV2[idx].isUsedRdma) {
            CHK_RET(InitRoceChannel(commParam, idx));
        } else {
            CHK_RET(InitP2pChannel(commParam, idx));
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::InitP2pChannel(HcclIndOpChannelRemoteResV3* commParam, uint32_t channelIndex)
{
    CHK_PTR_NULL(commParam);
    CHK_PTR_NULL(commParam->channelList);
    HcclIndOpChannelRemoteResV2& remoteResV2 = commParam->remoteResV2[channelIndex];
    u32 linkType = static_cast<u32>(remoteResV2.channelP2p.transportAttr.linkType);
    std::string channelKey = std::string(commParam->channelTag) + ":" + std::to_string(commParam->engine) + ":"
                             + std::to_string(remoteResV2.remoteRank) + ":" + std::to_string(linkType);
    HCCL_INFO("%s channelKey[%s]", __func__, channelKey.c_str());
    if (channelHandleMap_.find(channelKey) != channelHandleMap_.end()) {
        HCCL_ERROR("[%s]the channel has existed.", __func__);
        return HCCL_E_INTERNAL;
    }

    HcclChannelP2p& channelP2p = remoteResV2.channelP2p;
    if (channelP2p.localIpcSignal[0].resId == INVALID_U64) {
        HCCL_ERROR("[%s]the Channel is invalid", __func__);
        return HCCL_E_INTERNAL;
    }

    // 创建Transport对象
    MachinePara machinePara;
    CHK_RET(SetTransportMachinePara(machinePara, remoteResV2.remoteRank, commParam->channelTag));
    machinePara.notifyNum = remoteResV2.p2pNotifyNum;
    // 获取localMem & remoteMem
    TransportDeviceP2pData transDevP2pData;
    transDevP2pData.inputBufferPtr = reinterpret_cast<void*>(channelP2p.remoteHcclbuffer.addr);
    transDevP2pData.outputBufferPtr = reinterpret_cast<void*>(channelP2p.remoteHcclbuffer.addr);
    if (transDevP2pData.inputBufferPtr == nullptr || transDevP2pData.outputBufferPtr == nullptr) {
        HCCL_ERROR(
            "[%s]input ptr[%p] or output ptr[%p] is null.", __func__, transDevP2pData.inputBufferPtr,
            transDevP2pData.outputBufferPtr);
        return HCCL_E_PARA;
    }
    // 获取Notify资源
    CHK_RET(SetChannelP2pNotify(transDevP2pData, remoteResV2.p2pNotifyNum, channelP2p)); // 待确认notify是否
    //  获取transportAttr信息
    transDevP2pData.transportAttr = channelP2p.transportAttr;
    //  创建Transport对象
    std::shared_ptr<Transport> link;
    TransportPara para{};
    const std::unique_ptr<NotifyPool> notifyPool;
    DispatcherCtx* ctx = static_cast<DispatcherCtx*>(dispatcherCtx_);
    CHK_PRT(ctx->SetDispatcherHcclQos(remoteResV2.channelP2p.qos)); // 调度器添加hcclQos
    CHK_PTR_NULL(ctx);
    link.reset(new (std::nothrow) Transport(
        TransportType::TRANS_TYPE_DEVICE_P2P, para, ctx->GetDispatcher(), notifyPool, machinePara, transDevP2pData));
    CHK_SMART_PTR_NULL(link);
    CHK_RET(link->Init()); // 初始化需要增加远端用户注册内存

    ChannelHandle channelHandle = reinterpret_cast<ChannelHandle>(link.get());
    channelHandleMap_[channelKey] = channelHandle;
    linkMap_[channelHandle] = link;

    // 恢复出的channelHandle回填到commParam中
    CHK_PTR_NULL(commParam->channelList);
    ChannelHandle* channelList = reinterpret_cast<ChannelHandle*>(commParam->channelList);
    channelList[channelIndex] = channelHandle;

    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::SetChannelP2pNotify(
    TransportDeviceP2pData& transDevP2pData, u64& p2pNotifyNum, HcclChannelP2p& channelP2p)
{
    u64 actualNotifyNum = 0;
    // 获取Ipc notify信息
    CHK_RET(CheckNotifyOrQPMaxNum(actualNotifyNum, LINK_P2P_MAX_NUM, true));
    std::shared_ptr<LocalNotify> ipcPreWaitNotify = std::make_shared<LocalNotify>();
    CHK_RET(InitAndVerifySingleSignal(channelP2p.localIpcSignal[actualNotifyNum], ipcPreWaitNotify));
    transDevP2pData.ipcPreWaitNotify = ipcPreWaitNotify;

    std::shared_ptr<RemoteNotify> ipcPreRecordNotify = std::make_shared<RemoteNotify>();
    CHK_RET(InitAndVerifySingleSignal(channelP2p.remoteIpcSignal[actualNotifyNum], ipcPreRecordNotify));
    transDevP2pData.ipcPreRecordNotify = ipcPreRecordNotify;
    actualNotifyNum++;

    CHK_RET(CheckNotifyOrQPMaxNum(actualNotifyNum, LINK_P2P_MAX_NUM, true));
    std::shared_ptr<LocalNotify> ipcPostWaitNotify = std::make_shared<LocalNotify>();
    CHK_RET(InitAndVerifySingleSignal(channelP2p.localIpcSignal[actualNotifyNum], ipcPostWaitNotify));
    transDevP2pData.ipcPostWaitNotify = ipcPostWaitNotify;

    std::shared_ptr<RemoteNotify> ipcPostRecordNotify = std::make_shared<RemoteNotify>();
    CHK_RET(InitAndVerifySingleSignal(channelP2p.remoteIpcSignal[actualNotifyNum], ipcPostRecordNotify));
    transDevP2pData.ipcPostRecordNotify = ipcPostRecordNotify;
    actualNotifyNum++;

    transDevP2pData.userLocalNotify.resize(p2pNotifyNum, nullptr);
    transDevP2pData.userRemoteNotify.resize(p2pNotifyNum, nullptr);

    for (u32 idx = 0; idx < p2pNotifyNum; idx++) {
        std::shared_ptr<LocalNotify> ipcWaitNotify = std::make_shared<LocalNotify>();
        CHK_RET(InitAndVerifySingleSignal(channelP2p.localIpcSignal[actualNotifyNum], ipcWaitNotify));
        transDevP2pData.userLocalNotify[idx] = ipcWaitNotify;

        std::shared_ptr<RemoteNotify> ipcRecordNotify = std::make_shared<RemoteNotify>();
        CHK_RET(InitAndVerifySingleSignal(channelP2p.remoteIpcSignal[actualNotifyNum], ipcRecordNotify));
        transDevP2pData.userRemoteNotify[idx] = ipcRecordNotify;

        actualNotifyNum++;
    }

    HCCL_DEBUG(
        "%s get p2pNotify success, p2pNotifyNum[%llu], actualNotifyNum[%llu]", __func__, p2pNotifyNum, actualNotifyNum);
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::InitRoceChannel(HcclIndOpChannelRemoteResV3* commParam, uint32_t channelIndex)
{
    HcclIndOpChannelRemoteResV2& remoteResV2 = commParam->remoteResV2[channelIndex];
    std::string channelKey = std::string(commParam->channelTag) + ":" + std::to_string(commParam->engine) + ":"
                             + std::to_string(remoteResV2.remoteRank) + ":"
                             + std::to_string(CommProtocol::COMM_PROTOCOL_ROCE);
    if (channelHandleMap_.find(channelKey) != channelHandleMap_.end()) {
        HCCL_ERROR("[%s]the channel has existed.", __func__);
        return HCCL_E_INTERNAL;
    }

    HcclChannelRoce& channelRoce = remoteResV2.channelRoce;
    if (channelRoce.localNotifyList == 0) {
        HCCL_ERROR("[%s]the Channel is invalid", __func__);
        return HCCL_E_INTERNAL;
    }
    HcclSignalInfo* localNotifyList = reinterpret_cast<HcclSignalInfo*>(channelRoce.localNotifyList);
    if (localNotifyList[0].resId == INVALID_U64) {
        HCCL_INFO("[%s]the channel notify resource is invalid", __func__);
        return HCCL_E_INTERNAL;
    }

    // 创建Transport对象
    MachinePara machinePara;
    CHK_RET(SetTransportMachinePara(machinePara, remoteResV2.remoteRank, commParam->channelTag)); // 待确认是否填充完毕
    machinePara.notifyNum = remoteResV2.roceNotifyNum;
    // 获取localMem & remoteMem
    TransportDeviceIbverbsData transDevIbverbsData;
    transDevIbverbsData.inputBufferPtr = reinterpret_cast<void*>(channelRoce.remoteHcclbuffer.addr);
    transDevIbverbsData.outputBufferPtr = reinterpret_cast<void*>(channelRoce.remoteHcclbuffer.addr);
    if (transDevIbverbsData.inputBufferPtr == nullptr || transDevIbverbsData.outputBufferPtr == nullptr) {
        HCCL_ERROR(
            "[%s]input ptr[%p] or output ptr[%p] is null.", __func__, transDevIbverbsData.inputBufferPtr,
            transDevIbverbsData.outputBufferPtr);
        return HCCL_E_PARA;
    }
    transDevIbverbsData.localInputMem = channelRoce.localHcclbuffer;
    transDevIbverbsData.localOutputMem = channelRoce.localHcclbuffer;
    transDevIbverbsData.localNotifyValueAddr = channelRoce.notifyValue;
    transDevIbverbsData.notifyValueKey = channelRoce.notifyValueKey;
    transDevIbverbsData.remoteInputKey = channelRoce.remoteHcclbuffer.key;
    transDevIbverbsData.remoteOutputKey = channelRoce.remoteHcclbuffer.key;
    // 需要添加远端用户注册内存

    // 获取QPinfo
    u32 roceQpNumSum = channelRoce.qpsPerConnection + static_cast<u32>(channelRoce.qpsPerConnection != 1);
    transDevIbverbsData.qpInfo.resize(roceQpNumSum);
    std::copy_n(channelRoce.QpInfo, roceQpNumSum, transDevIbverbsData.qpInfo.begin());
    transDevIbverbsData.multiQpThreshold = multiQpThreshold_;
    transDevIbverbsData.qpsPerConnection = channelRoce.qpsPerConnection;

    // 获取notify
    u64& roceNotifyNum = remoteResV2.roceNotifyNum; // 是否需要待确认
    CHK_RET(SetChannelRoceNotify(transDevIbverbsData, roceNotifyNum, channelRoce));

    // 创建Transport对象
    std::shared_ptr<Transport> link;
    TransportPara para{};
    para.timeout = linkTimeOut_; // 暂无法设置
    const std::unique_ptr<NotifyPool> notifyPool;
    DispatcherCtx* ctx = static_cast<DispatcherCtx*>(dispatcherCtx_);
    CHK_PTR_NULL(ctx);
    link.reset(new (std::nothrow) Transport(
        TransportType::TRANS_TYPE_DEVICE_IBVERBS, para, ctx->GetDispatcher(), notifyPool, machinePara,
        TransportDeviceP2pData(), transDevIbverbsData));
    CHK_SMART_PTR_NULL(link);
    CHK_RET(link->Init()); // 初始化需要增加远端用户注册内存

    ChannelHandle channelHandle = reinterpret_cast<ChannelHandle>(link.get());
    channelHandleMap_[channelKey] = channelHandle;
    linkMap_[channelHandle] = link;

    // 恢复出的channelHandle回填到commParam中
    ChannelHandle* channelList = reinterpret_cast<ChannelHandle*>(commParam->channelList);
    channelList[channelIndex] = channelHandle;

    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::SetChannelRoceNotify(
    TransportDeviceIbverbsData& transDevIbverbsData, u64& roceNotifyNum, HcclChannelRoce& channelRoce)
{
    u64 actualNotifyNum = 0;
    if (channelRoce.localNotifyList == 0 || channelRoce.remoteNotifyList == 0) {
        HCCL_DEBUG("[%s] Empty local and remote notify lists, skipping notify resource creation.", __func__);
        return HCCL_SUCCESS;
    }
    HcclSignalInfo* localNotifyList = reinterpret_cast<HcclSignalInfo*>(channelRoce.localNotifyList);
    AddrKey* remoteNotifyList = reinterpret_cast<AddrKey*>(channelRoce.remoteNotifyList);
    if (localNotifyList == nullptr || remoteNotifyList == nullptr) {
        HCCL_ERROR("[%s]nullptr found in localNotifyList or remoteNotifyList from device mem, check.", __func__);
        return HCCL_E_INTERNAL;
    }
    // 获取RDMA Notify信息
    std::shared_ptr<LocalNotify> ackNotify = std::make_shared<LocalNotify>();
    CHK_RET(InitAndVerifySingleSignal(localNotifyList[actualNotifyNum], ackNotify));
    transDevIbverbsData.ackNotify = ackNotify;
    transDevIbverbsData.remoteAckNotifyDetails = remoteNotifyList[actualNotifyNum];
    actualNotifyNum++;

    std::shared_ptr<LocalNotify> dataNotify = std::make_shared<LocalNotify>();
    CHK_RET(InitAndVerifySingleSignal(localNotifyList[actualNotifyNum], dataNotify));
    transDevIbverbsData.dataNotify = dataNotify;
    transDevIbverbsData.remoteDataNotifyDetails = remoteNotifyList[actualNotifyNum];
    actualNotifyNum++;

    std::shared_ptr<LocalNotify> dataAckNotify = std::make_shared<LocalNotify>();
    CHK_RET(InitAndVerifySingleSignal(localNotifyList[actualNotifyNum], dataAckNotify));
    transDevIbverbsData.dataAckNotify = dataAckNotify;
    transDevIbverbsData.remoteDataAckNotifyDetails = remoteNotifyList[actualNotifyNum];
    transDevIbverbsData.notifySize = notifySize_;
    actualNotifyNum++;

    transDevIbverbsData.userLocalNotify.resize(channelRoce.qpsPerConnection);
    transDevIbverbsData.userRemoteNotifyDetails.resize(channelRoce.qpsPerConnection);
    // 当前多QP下每个QP会多申请一个DataNotify
    u64 singleQpNotifySize = channelRoce.singleQPNotifyNum + static_cast<u32>(channelRoce.qpsPerConnection > 1);
    for (u32 qpIndex = 0; qpIndex < channelRoce.qpsPerConnection; qpIndex++) {
        transDevIbverbsData.userLocalNotify[qpIndex].resize(singleQpNotifySize, nullptr);
        transDevIbverbsData.userRemoteNotifyDetails[qpIndex].resize(singleQpNotifySize);
        for (u32 i = 0, idx = actualNotifyNum + singleQpNotifySize * qpIndex; i < singleQpNotifySize; ++idx, ++i) {
            std::shared_ptr<LocalNotify> locNotify = std::make_shared<LocalNotify>();
            CHK_RET(InitAndVerifySingleSignal(localNotifyList[idx], locNotify));
            transDevIbverbsData.userLocalNotify[qpIndex][i] = locNotify;
            transDevIbverbsData.userRemoteNotifyDetails[qpIndex][i] = remoteNotifyList[idx];
        }
    }
    roceNotifyNum = channelRoce.singleQPNotifyNum;
    HCCL_DEBUG("[%s]get roceNotify success, roceNotifyNum[%u]", __func__, roceNotifyNum);
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::NotifyAlloc(NotifyMgrAicpuParam* param)
{
    u32 notifyNum = param->notifyNum;
    std::string notifysStr = std::string(param->notifyParam, NOTIFY_UNIQUE_ID_MAX_SIZE);
    std::string hcomId(param->hcomId);
    size_t notifySize = notifys_.size();
    HCCL_INFO(
        "[HcclCommAicpu][%s] comm identifier[%s], alloc notify num[%u] begin, before notifySize[%u]", __func__,
        hcomId.c_str(), notifyNum, notifySize);
    if (UNLIKELY(HcclCheckLogLevel(HCCL_LOG_INFO))) {
        std::ostringstream oss;
        oss << "notifyParam" << " raw bytes: ";
        for (u32 i = 0; i < NOTIFY_UNIQUE_ID_MAX_SIZE; ++i) {
            oss << std::hex << std::setw(2) << std::setfill('0')
                << static_cast<unsigned int>(static_cast<unsigned char>(param->notifyParam[i])) << " ";
        }
        HCCL_INFO("[HcclCommAicpu][%s] %s", __func__, oss.str().c_str());
    }
    HcclResult ret = NotifyManager::ParseBinNotifys(notifysStr, notifys_);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcclCommAicpu][%s] comm identifier[%s], alloc notify num[%u] failed %u", __func__, hcomId.c_str(),
            notifyNum, ret);
        return ret;
    }
    HCCL_INFO(
        "[HcclCommAicpu][%s] comm identifier[%s], alloc notify num[%u] end, after notifySize[%u]", __func__,
        hcomId.c_str(), notifyNum, notifys_.size());
    NotifyHandle* notifyArray = static_cast<NotifyHandle*>(param->deviceHandle);
    CHK_PTR_NULL(notifyArray);
    // 空指针校验
    for (size_t i = 0; i < notifyNum; ++i) {
        notifyArray[i] = reinterpret_cast<NotifyHandle>(notifys_[i + notifySize].get()); // 拷贝裸指针
        HCCL_INFO("[HcclCommAicpu][%s] notifyArray[%u] = [%lu]", __func__, i + notifySize, notifyArray[i]);
    }

    HCCL_INFO(
        "[HcclCommAicpu][%s] comm identifier[%s], alloc notify num[%u] success", __func__, hcomId.c_str(), notifyNum);
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::NotifyFree(NotifyMgrAicpuParam* param)
{
    u32 notifyNum = param->notifyNum;
    NotifyHandle* notifyArray = static_cast<NotifyHandle*>(param->deviceHandle);
    std::string hcomId(param->hcomId);
    // 空指针校验
    CHK_PTR_NULL(notifyArray);
    for (size_t i = 0; i < notifyNum; ++i) {
        LocalNotify* notify = reinterpret_cast<LocalNotify*>(notifyArray[i]);
        HCCL_INFO("[HcclCommAicpu][%s] notifyArray[%u]=[%lu]", __func__, i, notifyArray[i]);
        auto it = std::find_if(notifys_.begin(), notifys_.end(), [notify](const std::unique_ptr<LocalNotify>& ptr) {
            return ptr.get() == notify;
        });
        if (it != notifys_.end()) {
            HCCL_INFO(
                "[HcclCommAicpu][%s] comm identifier[%s], free notify[%u] success", __func__, hcomId.c_str(),
                notifyArray[i]);
            notifys_.erase(it);
        } else {
            HCCL_RUN_WARNING("[HcclCommAicpu][%s] localNotify[%u] not found in notifys_", __func__, i);
        }
    }

    HCCL_INFO(
        "[HcclCommAicpu][%s] comm identifier[%s], free notify num[%u] success", __func__, hcomId.c_str(), notifyNum);
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::RegisterOpInfo(void* opInfo, u32 size)
{
    CHK_RET(taskExecption_.RegisterOpInfo(opInfo, size));
    u32 opIdx = taskExecption_.GetOpRingBufferIdx();
    CHK_RET(SetDispatcherCtxOpIdx(opIdx));
    HCCL_INFO("%s success, group[%s], opRingBufferId[%u]", __func__, identifier_.c_str(), opIdx);
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::RegOpTaskException(HcommGetOpInfoCallback callback)
{
    CHK_RET(taskExecption_.RegisterOpInfoCallback(callback));
    return HCCL_SUCCESS;
}

HcclResult HcclCommAicpu::SetDispatcherCtxOnThread()
{
    // 设置 DispatcherCtx 到线程变量
    CHK_RET(SetDispatcherCtx(dispatcherCtx_));
    return HCCL_SUCCESS;
}
} // namespace hccl

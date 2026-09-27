/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_comm_aicpu.h"
#include "coll_comm_aicpu_mgr.h"
#include "aicpu_communicator.h"
#include "adapter_hal_pub.h"
#include "ns_recovery/aicpu/ns_recovery_func_lite.h"
#include "dlhal_function_v2.h"
#include "dfx_profiling_command_handle_lite.h"
#include "hcclCommTaskExceptionLite.h"
#include "hcclCommOp.h"
#include "hcclCommDfxLite.h"
#include "env_config/env_config_v2.h"
#include "log.h"
#include <shared_mutex>

CollCommAicpu::~CollCommAicpu()
{
    HCCL_RUN_INFO("[CollCommAicpu][%s]Group[%s] destroy success", __func__, identifier_.c_str());
}

HcclResult CollCommAicpu::InitAicpuIndOp(CommAicpuParam* commAicpuParam)
{
    if (commStatus_ == HcclCommStatus::HCCL_COMM_STATUS_READY) {
        HCCL_RUN_INFO("[CollCommAicpu][%s]Group[%s] already initialized, skip reinit", __func__, identifier_.c_str());
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(commAicpuParam);
    topoInfo_.deviceLogicId = commAicpuParam->deviceLogicId;
    topoInfo_.devicePhyId = commAicpuParam->devicePhyId;
    topoInfo_.deviceType = static_cast<DevType>(commAicpuParam->deviceType);
    identifier_ = std::string(commAicpuParam->hcomId);
    topoInfo_.userRankSize = commAicpuParam->userRankSize;
    topoInfo_.userRank = commAicpuParam->userRank;

    CHK_RET(hrtSetWorkModeAicpu(true));
    CHK_RET(hrtSetlocalDevice(topoInfo_.deviceLogicId));
    CHK_RET(hrtSetlocalDeviceType(topoInfo_.deviceType));
    CHK_RET(hrtDrvGetLocalDevIDByHostDevID(topoInfo_.devicePhyId, &devId_));
    CHK_RET(dfx_.Init(devId_, identifier_, topoInfo_.userRankSize, topoInfo_.userRank));
    CHK_RET(RegisterProfCallBack());
    CHK_RET(InitHDCommunicate(commAicpuParam));

    EXCEPTION_CATCH(nsRecoveryLitePtr_ = std::make_shared<hccl::NsRecoveryLite>(), return HCCL_E_PTR);
    nsRecoveryLitePtr_->Init(kfcControlTransferH2D_, kfcStatusTransferD2H_);

    CHK_RET(Hccl::DlHalFunctionV2::GetInstance().DlHalFunctionInit());

    // commEngineResMgr_/channelMgr_ 为 CollCommAicpu 成员（unique_ptr），生命周期被 this 严格包含，
    // 因此 lambda 捕获 this 安全，不会产生悬垂指针（析构顺序见 coll_comm_aicpu.h 成员声明）
    EXCEPTION_CATCH(
        commEngineResMgr_ = std::make_unique<CommEngineResAicpuMgr>(
            dfx_,
            [this](bool isTimeout) {
                return this->CheckIndOpExecStatus(isTimeout);
            }),
        return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(commEngineResMgr_);

    EXCEPTION_CATCH(channelMgr_ = std::make_unique<ChannelAicpuMgr>(dfx_, topoInfo_), return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(channelMgr_);

    commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_READY;

    HCCL_RUN_INFO(
        "[%s]success, group[%s], deviceLogicId[%u], devicePhyId[%u], deviceType[%u], rankSize[%u] "
        "userRank[%u], devId[%u]",
        __func__, identifier_.c_str(), topoInfo_.deviceLogicId, topoInfo_.devicePhyId, topoInfo_.deviceType,
        topoInfo_.userRankSize, topoInfo_.userRank, devId_);
    return HCCL_SUCCESS;
}

HcclResult CollCommAicpu::InitHDCommunicate(CommAicpuParam* commAicpuParam)
{
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
    return HCCL_SUCCESS;
}

void CollCommAicpu::SetCommmStatus(HcclCommStatus status)
{
    HCCL_INFO("[%s]group[%s], commStatus[%d]", __func__, identifier_.c_str(), static_cast<int>(status));
    commStatus_ = status;
}

HcclResult CollCommAicpu::Clean()
{
    CHK_SMART_PTR_NULL(channelMgr_);
    return channelMgr_->Clean();
}

HcclResult CollCommAicpu::Resume(HcclChannelUrmaRes* commParam)
{
    CHK_PTR_NULL(commParam);
    CHK_SMART_PTR_NULL(channelMgr_);
    CHK_RET(channelMgr_->Resume(commParam));
    nsRecoveryLitePtr_->SetNeedClean(false);

    // 重置Rtsq维护的 sq tail/head
    std::shared_lock<std::shared_mutex> lock(GetCommEngineResMgr()->GetThreadMutex());
    const std::vector<std::shared_ptr<hccl::Thread>>& threads = GetCommEngineResMgr()->GetAllThread();
    for (auto& thread : threads) {
        Hccl::StreamLite* streamLitePtr = reinterpret_cast<Hccl::StreamLite*>(thread->GetStreamLitePtr());
        streamLitePtr->GetRtsq()->Reset(true);
    }

    SetErrorReported(false);
    commStatus_ = HcclCommStatus::HCCL_COMM_STATUS_READY;
    HCCL_RUN_INFO("[CollCommAicpu][Resume] commId[%s] resume success.", identifier_.c_str());

    return HCCL_SUCCESS;
}

hccl::NsRecoveryLitePtr CollCommAicpu::GetNsRecoveryLitePtr() { return nsRecoveryLitePtr_; }

HcclResult CollCommAicpu::CheckIndOpExecStatus(bool timeout)
{
    if (timeout) {
        HCCL_ERROR("[%s]comm[%s] op launch timeout, print taskException", __func__, identifier_.c_str());
        hcomm::HcclCommTaskExceptionLite::GetInstance().PrintCommTaskException(this);
        hcomm::HcclCommTaskExceptionLite::GetInstance().PrintAllCommTaskException();
        return HCCL_E_INTERNAL;
    } else if (commStatus_ == HCCL_COMM_STATUS_SUSPENDING) {
        HCCL_WARNING("[%s]comm[%s] commStatus[%d] is suspending", __func__, identifier_.c_str(), commStatus_);
        return HCCL_E_SUSPENDING;
    } else if (commStatus_ != HCCL_COMM_STATUS_READY) {
        HCCL_ERROR("[%s]comm[%s] commStatus[%d] is not ready, return fail", __func__, identifier_.c_str(), commStatus_);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult CollCommAicpu::BackGroundGetCmd(Hccl::KfcCommand& cmd)
{
    CHK_SMART_PTR_NULL(kfcControlTransferH2D_);
    HcclResult ret = kfcControlTransferH2D_->Get(0, sizeof(Hccl::KfcCommand), reinterpret_cast<uint8_t*>(&cmd));
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[%s]fail, group[%s]", __func__, identifier_.c_str()), ret);
    return HCCL_SUCCESS;
}

HcclResult CollCommAicpu::BackGroundSetStatus(Hccl::KfcStatus state)
{
    Hccl::KfcExecStatus status;
    status.kfcStatus = state;
    HCCL_INFO("[%s]group[%s], state[%d]", __func__, identifier_.c_str(), static_cast<int>(state));
    HcclResult ret = kfcStatusTransferD2H_->Put(0, sizeof(status.kfcStatus), reinterpret_cast<uint8_t*>(&status));
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[%s]fail, group[%s]", __func__, identifier_.c_str()), ret);
    return HCCL_SUCCESS;
}

HcclResult CollCommAicpu::SendErrorMessageReportToHost(Hccl::ErrorMessageReport& errMsgInfo)
{
    CHK_SMART_PTR_NULL(kfcStatusTransferD2H_);
    CHK_RET(kfcStatusTransferD2H_->Put(
        sizeof(Hccl::KfcStatus) + sizeof(Hccl::KfcErrType), sizeof(errMsgInfo),
        reinterpret_cast<uint8_t*>(&errMsgInfo)));
    return HCCL_SUCCESS;
}

HcclResult CollCommAicpu::RegisterProfCallBack() { return Hccl::DfxRegisterProfCallBack(); }

u32 CollCommAicpu::UpdateIndex() { return index_ += 1; }

HcclResult CollCommAicpu::InitDfxOpInfo(HcclDfxOpInfo* aicpuDfxInfo)
{
    HCCL_INFO(
        "[%s]group[%s], algTag[%s], profiling L0[%d], L1[%d]", __func__, identifier_.c_str(), aicpuDfxInfo->algTag,
        Hccl::DfxProfilingHandlerLite::GetInstance().GetProfL0State(),
        Hccl::DfxProfilingHandlerLite::GetInstance().GetProfL1State());

    Hccl::DfxDfxOpInfo newDfxOpInfo{};
    newDfxOpInfo.opType = static_cast<u8>(aicpuDfxInfo->opType);
    newDfxOpInfo.dataType = static_cast<u8>(aicpuDfxInfo->dataType);

    newDfxOpInfo.commHandle = this;
    newDfxOpInfo.count = aicpuDfxInfo->dataCount;
    newDfxOpInfo.srcAddr = aicpuDfxInfo->inputMemAddr;
    newDfxOpInfo.dstAddr = aicpuDfxInfo->outputMemAddr;
    newDfxOpInfo.srcSize = aicpuDfxInfo->inputMemSize;
    newDfxOpInfo.dstSize = aicpuDfxInfo->outputMemSize;
    newDfxOpInfo.opIndex = UpdateIndex();
    newDfxOpInfo.cpuWaitAicpuNotifyId = aicpuDfxInfo->cpuWaitAicpuNotifyId;
    newDfxOpInfo.algType = static_cast<u8>(Hccl::AlgTypeVal::ALG_TYPE_NOT_SPECIFIED);
    auto algTagLen = strnlen(aicpuDfxInfo->algTag, sizeof(newDfxOpInfo.algTag) - 1);
    CHK_SAFETY_FUNC_RET(
        memcpy_s(newDfxOpInfo.algTag, sizeof(newDfxOpInfo.algTag) - 1, aicpuDfxInfo->algTag, algTagLen));

    CHK_RET(dfx_.SetCurrDfxOpInfo(&newDfxOpInfo));
    return HCCL_SUCCESS;
}

HcclResult CollCommAicpu::ProfilingReportDeviceOp()
{
    hccl::HcclCommDfxLite* hcclCommDfxLite = GetHcclCommDfxLite();
    CHK_PTR_NULL(hcclCommDfxLite);
    auto* currDfxOpInfo = static_cast<const Hccl::DfxDfxOpInfo*>(hcclCommDfxLite->GetLatestDfxOpInfo());
    if (currDfxOpInfo == nullptr) {
        HCCL_WARNING("[%s] no op info registered, skip ProfilingReportDeviceOp.", __func__);
        return HCCL_SUCCESS;
    }

    const auto& sharedThreads = commEngineResMgr_->GetAllThread();
    std::vector<hccl::Thread*> threads;
    threads.reserve(sharedThreads.size());
    for (const auto& t : sharedThreads) {
        threads.push_back(t.get());
    }
    hcclCommDfxLite->ReportAllTasks(threads);
    EXCEPTION_CATCH(
        Hccl::DfxProfilingHandlerLite::GetInstance().ReportHcclOpInfo(
            *currDfxOpInfo, hcclCommDfxLite->GetDfxCommContext()),
        return HCCL_E_INTERNAL);
    return HCCL_SUCCESS;
}

HcclResult CollCommAicpu::UpdateTask()
{
    CHK_RET(dfx_.UpdateProfStat());
    return HCCL_SUCCESS;
}

hccl::HcclCommAicpu* CollCommAicpu::GetLegacy910CollComm() { return legacy910CollComm_.first.get(); }

void CollCommAicpu::SetLegacy910CollComm(std::shared_ptr<hccl::HcclCommAicpu> comm)
{
    legacy910CollComm_.first = std::move(comm);
}

bool CollCommAicpu::IsLegacy910CollCommBusy() { return legacy910CollComm_.second.load(); }

void CollCommAicpu::SetLegacy910CollCommBusy(bool busy) { legacy910CollComm_.second.store(busy); }

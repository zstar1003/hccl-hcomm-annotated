/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <shared_mutex>
#include "inc/aicpu_mc2_handler.h"
#include "inc/aicpu_utils.h"
#include "communicator_impl_lite_manager.h"
#include "log.h"

namespace Hccl {
AicpuMc2Handler::AicpuMc2Handler() {}

AicpuMc2Handler& AicpuMc2Handler::GetInstance()
{
    static AicpuMc2Handler instance_;
    return instance_;
}

HcclResult AicpuMc2Handler::HcclGetCommHandleByCtx(void* ctx, void** opHandle) const
{
    HCCL_RUN_INFO("[%s]HcclGetCommHandleByCtx begin, ctx:%p, *ctx:%llu", __func__, ctx, *((uint64_t*)ctx));
    // 存储kernel参数
    unique_lock<std::shared_timed_mutex> handlerLock(AicpuUtils::GetInstance().handlerMutex_);
    AicpuUtils::GetInstance().kernelParam_ = reinterpret_cast<HcclKernelParamLite*>(ctx);
    uint32_t commIdIndex = AicpuUtils::GetInstance().kernelParam_->comm.idIndex;
    if (AicpuUtils::GetInstance().kernelParamMap_.find(commIdIndex)
        == AicpuUtils::GetInstance().kernelParamMap_.end()) {
        AicpuUtils::GetInstance().kernelParamMap_[commIdIndex] = AicpuUtils::GetInstance().kernelParam_;
    }
    handlerLock.unlock();

    // 创建单例对象
    std::shared_lock<std::shared_timed_mutex> sharedLock(AicpuUtils::GetInstance().handlerMutex_);
    AicpuUtils::GetInstance().CreateSingleInstance(ctx);
    CHK_RET(AicpuUtils::GetInstance().Init());

    // 初始化硬件参数
    DevCapability::GetInstance().Init(AicpuUtils::GetInstance().kernelParam_->comm.devType);

    HCCL_INFO(
        "[%s]DevCapability %s, kernelParam_.algName[%s], commIdIndex[%u]", __func__,
        AicpuUtils::GetInstance().kernelParam_->comm.devType.Describe().c_str(),
        AicpuUtils::GetInstance().kernelParam_->algName, commIdIndex);

    CommunicatorImplLite* communicatorImplLite = CommunicatorImplLiteMgr::GetInstance().Get(commIdIndex);
    CHK_PTR_NULL(communicatorImplLite);
    return AicpuUtils::GetInstance().GetCommHandle(communicatorImplLite, opHandle);
}

// HcclReleaseComm 设置isUsed标记未使用，不会释放opHandle
HcclResult AicpuMc2Handler::HcclReleaseComm(void* opHandle) const
{
    HCCL_RUN_INFO("[%s]HcclReleaseComm begin", __func__);
    CommunicatorImplLite* communicatorImplLite = reinterpret_cast<CommunicatorImplLite*>(opHandle);
    // isUsed状态置false
    unique_lock<std::mutex> aicpuLock(communicatorImplLite->GetAicpuMc2Mutex());
    communicatorImplLite->SetIsUsed(false);
    aicpuLock.unlock();

    unique_lock<std::shared_timed_mutex> handlerLock(AicpuUtils::GetInstance().handlerMutex_);
    uint32_t commIdIndex = communicatorImplLite->GetCommIdIndex();
    auto it = AicpuUtils::GetInstance().kernelParamMap_.find(commIdIndex);
    if (it != AicpuUtils::GetInstance().kernelParamMap_.end()) {
        AicpuUtils::GetInstance().kernelParamMap_.erase(it);
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuMc2Handler::HcclGetTaskStatus(void* opHandle, HcclTaskStatus* status) const
{
    CommunicatorImplLite* communicatorImplLite = reinterpret_cast<CommunicatorImplLite*>(opHandle);

    auto* streamLiteMgr = communicatorImplLite->GetStreamLiteMgr();
    CHK_PTR_NULL(streamLiteMgr);

    StreamLite* curStream = streamLiteMgr->GetMaster();
    CHK_PTR_NULL_WITH_MSG(curStream, "commId[%u].", communicatorImplLite->GetCommIdIndex());
    HCCL_INFO("[%s]commId[%u], stream[%u].", __func__, communicatorImplLite->GetCommIdIndex(), curStream->GetId());
    if (AicpuUtils::GetInstance().GetException(curStream, GET_TASK_STATUS, communicatorImplLite) == 1) {
        *status = HcclTaskStatus::HCCL_CQE_ERROR;
        return HCCL_SUCCESS;
    }

    for (uint32_t id = 0; id < streamLiteMgr->SizeOfSlaves(); id++) {
        curStream = streamLiteMgr->GetSlave(id);
        CHK_PTR_NULL_WITH_MSG(curStream, "commId[%u]", communicatorImplLite->GetCommIdIndex());
        HCCL_INFO("[%s]commId[%u], stream[%u].", __func__, communicatorImplLite->GetCommIdIndex(), curStream->GetId());
        if (AicpuUtils::GetInstance().GetException(curStream, GET_TASK_STATUS, communicatorImplLite) == 1) {
            *status = HcclTaskStatus::HCCL_CQE_ERROR;
            return HCCL_SUCCESS;
        }
    }

    *status = HcclTaskStatus::HCCL_NORMAL_STATUS;
    return HCCL_SUCCESS;
}

HcclResult AicpuMc2Handler::HcclCheckFinishByStream(void* opHandle) const
{
    CommunicatorImplLite* communicatorImplLite = reinterpret_cast<CommunicatorImplLite*>(opHandle);

    auto* streamLiteMgr = communicatorImplLite->GetStreamLiteMgr();
    CHK_PTR_NULL(streamLiteMgr);

    StreamLite* stream = streamLiteMgr->GetMaster();
    CHK_PTR_NULL(stream);

    // 比较主流首尾指针
    RtsqBase* rtsq = stream->GetRtsq();
    CHK_PTR_NULL_WITH_MSG(rtsq, "commId[%u], stream[%u].", communicatorImplLite->GetCommIdIndex(), stream->GetId());

    auto sqHead = rtsq->QuerySqHead();
    auto sqTail = rtsq->QuerySqTail();
    if (sqTail == sqHead) {
        HCCL_INFO(
            "[%s]Stream %u finished, sq id %u, head&tail %u.", __func__, stream->GetId(), stream->GetSqId(), sqHead);
        return HCCL_SUCCESS;
    }
    return HCCL_E_UNAVAIL;
}

HcclResult AicpuMc2Handler::HcclPrintTaskExceptionAllComm(void* opHandle) const
{
    // 打印全部通信域状态信息
    CommunicatorImplLite* curCommunicatorImplLite = reinterpret_cast<CommunicatorImplLite*>(opHandle);
    string additionInfo;
    auto communicatorImplLiteVec = CommunicatorImplLiteMgr::GetInstance().GetAll();
    for (CommunicatorImplLite* communicatorImplLite : communicatorImplLiteVec) {
        // 打印主流信息
        if (communicatorImplLite == curCommunicatorImplLite) {
            additionInfo = "[HcclPrintTaskExceptionAllComm]Current communicatorImplLite exists exception,commId "
                           + to_string(communicatorImplLite->GetCommIdIndex());
        } else {
            additionInfo = "";
        }
        auto* streamLiteMgr = communicatorImplLite->GetStreamLiteMgr();
        if (streamLiteMgr == nullptr) {
            HCCL_WARNING("[%s]CommunicatorImplLite streamLiteMgr is nullptr", __func__);
            continue;
        }

        StreamLite* curStream = streamLiteMgr->GetMaster();
        string nullInfo = "streamLiteMgr->GetMaster is nullptr";
        AicpuUtils::GetInstance().GetStreamException(curStream, nullInfo, communicatorImplLite, additionInfo);

        for (uint32_t id = 0; id < streamLiteMgr->SizeOfSlaves(); id++) {
            curStream = streamLiteMgr->GetSlave(id);
            nullInfo = "streamLiteMgr->GetSlave(" + to_string(id) + ") is nullptr";
            AicpuUtils::GetInstance().GetStreamException(curStream, nullInfo, communicatorImplLite, additionInfo);
        }
    }
    return HCCL_SUCCESS;
}

// ccore sqe wait拼写并下发流
HcclResult AicpuMc2Handler::HcclLaunchCcoreWait(
    void* opHandle, uint64_t waitAddr, uint32_t turnNum, uint64_t turnNumAddr, bool isLast) const
{
    HCCL_INFO(
        "[%s]opHandle %p, waitAddr %llu, turnNum %u, turnNumAddr %llu, isLast %u.", __func__, opHandle, waitAddr,
        turnNum, turnNumAddr, isLast);
    return AicpuUtils::GetInstance().HcclLaunchCcore(opHandle, waitAddr, turnNum, turnNumAddr, isLast, CCORE_WAIT_TYPE);
}

// ccore sqe record拼写并下发流
HcclResult
AicpuMc2Handler::HcclLaunchCcorePost(void* opHandle, uint64_t recordAddr, uint32_t turnNum, uint64_t turnNumAddr) const
{
    HCCL_INFO(
        "[%s]opHandle %p, recordAddr %llu, turnNum %u, turnNumAddr %llu.", __func__, opHandle, recordAddr, turnNum,
        turnNumAddr);
    return AicpuUtils::GetInstance().HcclLaunchCcore(
        opHandle, recordAddr, turnNum, turnNumAddr, false, CCORE_NOTIFY_TYPE);
}

HcclResult AicpuMc2Handler::HcclLaunchOp(void* opHandle, HcclOpData* data) const
{
    CommunicatorImplLite* communicatorImplLite = reinterpret_cast<CommunicatorImplLite*>(opHandle);
    CHK_RET(AicpuUtils::GetInstance().RecoverKernelParam(communicatorImplLite, data));
    CHK_RET(AicpuUtils::GetInstance().RestoreOpRes(communicatorImplLite));
    CHK_RET(AicpuUtils::GetInstance().ExecuteOp(communicatorImplLite));

    // 用于算法编排内存释放
    unique_lock<std::shared_timed_mutex> handlerLock(AicpuUtils::GetInstance().handlerMutex_);
    AicpuUtils::GetInstance().kernelParam_->op.algOperator.scratchMem = nullptr;
    handlerLock.unlock();
    return HCCL_SUCCESS;
}
} // namespace Hccl

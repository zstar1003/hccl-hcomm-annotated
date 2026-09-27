/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "profiling_manager_pub.h"
#include "hccl_comm_pub.h"
#include "common.h"
#include "hccl/hcom.h"
#include "hccl/hccl_types.h"
#include "hcom_device_profiling.h"
#include "stream_pub.h"
#ifdef CCL_KERNEL_AICPU
#include "device/inc/profiling_manager_device.h"
#endif
using namespace hccl;

extern HcclResult HcommProfilingInit(ThreadHandle* threads, u32 threadNum)
{
#ifdef CCL_KERNEL_AICPU
    bool profL0Open = dfx::ProfilingManager::IsProfL0On();
    bool profL1Open = dfx::ProfilingManager::IsProfL1On();
    HCCL_DEBUG("[%s] profL0Open:%d, profL1Open:%d", __func__, profL0Open, profL1Open);
    if (!profL1Open) {
        HCCL_INFO("[%s] L1 is off", __func__);
        return HCCL_SUCCESS;
    }
    for (u32 i = 0; i < threadNum; i++) {
        const SqeRingBuffer& sqeBuffer = GetStream(threads[i])->GetSqeContextPtr()->buffer;
        u16 taskId = sqeBuffer.tailSqeIdx;
        HCCL_DEBUG("[%s] thread id = [%u] task id = [%u]", __func__, GetStream(threads[i])->id(), taskId);
        CHK_RET(dfx::ProfilingManager::UpdateStartReportSqeIdx(GetStream(threads[i])->id(), taskId));
    }
#else
    HCCL_INFO("[%s] not support, do nothing", __func__);
#endif
    return HCCL_SUCCESS;
}

extern HcclResult HcommProfilingReportMainStreamAndFirstTask(ThreadHandle thread)
{
#ifdef CCL_KERNEL_AICPU
    bool profL0Open = dfx::ProfilingManager::IsProfL0On();
    bool profL1Open = dfx::ProfilingManager::IsProfL1On();
    HCCL_DEBUG("[%s] profL0Open:%d, profL1Open:%d", __func__, profL0Open, profL1Open);
    Stream* stream = GetStream(thread);
    if (stream == nullptr) {
        HCCL_ERROR("[%s] stream is nullptr", __func__);
        return HCCL_E_PTR;
    }
    HcclSqeContext* sqeContext = stream->GetSqeContextPtr();
    if (sqeContext == nullptr) {
        HCCL_ERROR("[%s] sqeContext is nullptr", __func__);
        return HCCL_E_PTR;
    }
    const SqeRingBuffer& sqeBuffer = sqeContext->buffer;
    uint16_t HEAD_TASK = 0;
    u16 taskId = sqeBuffer.tailSqeTaskId;
    HCCL_DEBUG("[%s] thread id = [%u] task id = [%u]", __func__, stream->id(), taskId);
    return dfx::ProfilingManager::ReportMainStreamTask(*stream, taskId, HEAD_TASK);
#else
    HCCL_INFO("[%s] not support, do nothing", __func__);
#endif
    return HCCL_SUCCESS;
}

extern HcclResult HcommProfilingReportMainStreamAndLastTask(ThreadHandle thread)
{
#ifdef CCL_KERNEL_AICPU
    const SqeRingBuffer& sqeBuffer = GetStream(thread)->GetSqeContextPtr()->buffer;
    uint16_t TAIL_TASK = 1;
    u16 taskId = sqeBuffer.tailSqeTaskId - 1;
    HCCL_DEBUG("[%s] thread id = [%u] task id = [%u]", __func__, GetStream(thread)->id(), taskId);
    return dfx::ProfilingManager::ReportMainStreamTask(*GetStream(thread), taskId, TAIL_TASK);
#else
    HCCL_INFO("[%s] not support, do nothing", __func__);
#endif
    return HCCL_SUCCESS;
}

// device 侧的op
extern HcclResult HcommProfilingReportDeviceHcclOpInfo(HcomProInfo profInfo)
{
#ifdef CCL_KERNEL_AICPU
    MsprofAicpuHCCLOPInfo hcclOpInfo{};
    hcclOpInfo.relay = 0; // 目前全是false
    hcclOpInfo.retry = 0; // 目前全是false
    hcclOpInfo.dataType = static_cast<HcclDataType>(profInfo.dataType);
    hcclOpInfo.count = profInfo.dataCount;
    uint64_t groupHashId = dfx::ProfilingManager::GetProfHashId(profInfo.commName, profInfo.commNameLen);
    hcclOpInfo.groupName = groupHashId;
    hcclOpInfo.ranksize = profInfo.rankSize;
    std::string algTypeStr(profInfo.algType);
    hcclOpInfo.streamId = 0; // kfc的流， 目前没有默认为0 AicpuGetStreamId()
    AlgType algType;
    CHK_PRT_RET(
        TransferStrToAlgType(algTypeStr, algType) == false,
        HCCL_ERROR("[%s] Fail to transfer [%s] to AlgType", __func__, algTypeStr.c_str()), HCCL_E_PARA);
    algTypeStr = TransferAlgType(algType);
    HCCL_INFO(
        "[%s] groupName = [%u], commName = [%s], ranksize = [%u], taskId = [%u], streamId = [%u], dataType = [%u], "
        "algTypeStr = [%s]",
        __func__, hcclOpInfo.groupName, profInfo.commName, hcclOpInfo.ranksize, hcclOpInfo.taskId, hcclOpInfo.streamId,
        hcclOpInfo.dataType, algTypeStr.c_str());
    CHK_RET(dfx::ProfilingManager::ReportHcclOpInfo(hcclOpInfo, algTypeStr));
    return HCCL_SUCCESS;
#else
    HCCL_INFO("[%s] not support, do nothing", __func__);
#endif
    return HCCL_SUCCESS;
}

extern HcclResult HcommProfilingEnd(ThreadHandle* threads, u32 threadNum)
{
#ifdef CCL_KERNEL_AICPU
    // 上报task
    if (dfx::ProfilingManager::GetProfL1State()) {
        for (u32 i = 0; i < threadNum; i++) {
            HCCL_DEBUG("[%s] thread id = [%u]", __func__, GetStream(threads[i])->id());
            CHK_RET(dfx::ProfilingManager::ReportTaskInfo(
                GetStream(threads[i])->id(), GetStream(threads[i])->GetSqeContextPtr()));
        }
    }

    for (u32 i = 0; i < threadNum; i++) {
        HCCL_DEBUG("[%s] thread id = [%u]", __func__, GetStream(threads[i])->id());
        CHK_RET(GetStream(threads[i])->ClearLocalBuff());
        CHK_RET(dfx::ProfilingManager::UpdateStartReportSqeIdx(GetStream(threads[i])->id(), 0));
    }
#else
    HCCL_INFO("[%s] not support, do nothing", __func__);
#endif
    return HCCL_SUCCESS;
}

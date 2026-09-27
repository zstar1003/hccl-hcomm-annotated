/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opexecounter.h"
#include "adapter_rts_common.h"
#include "env_config.h"
#include "dispatcher.h"
namespace hccl {
OpExeCounter& OpExeCounter::GetInstance(s32 deviceLogicID)
{
    static OpExeCounter opCounter[MAX_MODULE_DEVICE_NUM];
    if (static_cast<u32>(deviceLogicID) >= MAX_MODULE_DEVICE_NUM - 1) {
        HCCL_WARNING("[OpExeCounter][GetInstance] deviceLogicID[%d] is invalid", deviceLogicID);
        return opCounter[MAX_MODULE_DEVICE_NUM - 1];
    }
    return opCounter[deviceLogicID];
}

HcclResult OpExeCounter::InitCounter()
{
    DevType devType = DevType::DEV_TYPE_910;
    CHK_RET(hrtGetDeviceType(devType));
    if (!GetExternalInputStuckDetect() || devType == DevType::DEV_TYPE_310P3 || devType == DevType::DEV_TYPE_910
        || devType == DevType::DEV_TYPE_310P1) {
        isNeedOpCounter_ = false;
        HCCL_RUN_INFO("do not need add counter");
        return HCCL_SUCCESS;
    }
    if (refCount_ <= 0) {
        refCount_ = 0;
        int32_t defCount = 0;
        ReleaseMemHandles();
        memSize_ = sizeof(int32_t);
        CHK_RET(hrtMalloc(&headCountMem_, memSize_));
        CHK_PTR_NULL(headCountMem_);
        CHK_RET(hrtMemSyncCopy(
            headCountMem_, memSize_, &defCount, memSize_, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

        CHK_RET(hrtMalloc(&tailCountMem_, memSize_));
        CHK_PTR_NULL(tailCountMem_);
        CHK_RET(hrtMemSyncCopy(
            tailCountMem_, memSize_, &defCount, memSize_, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

        int32_t addOneVal = 1;
        CHK_RET(hrtMalloc(&addOneMem_, memSize_));
        CHK_PTR_NULL(addOneMem_);
        CHK_RET(hrtMemSyncCopy(
            addOneMem_, memSize_, &addOneVal, memSize_, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

        HCCL_RUN_INFO("alloc counter mem resource.");
    }
    refCount_++;
    isNeedOpCounter_ = true;
    return HCCL_SUCCESS;
}

OpExeCounter::~OpExeCounter() {}

HcclResult OpExeCounter::DeInitCounter()
{
    if (!isNeedOpCounter_) {
        HCCL_DEBUG("do not need add counter");
        ReleaseMemHandles();
        return HCCL_SUCCESS;
    }
    refCount_--;
    if (refCount_ <= 0) {
        std::pair<int32_t, int32_t> counter;
        CHK_RET(GetCounter(counter));
        HCCL_RUN_INFO(
            "[OpExeCounter][DeInitCounter] head counter[%d], tail counter[%d]", counter.first, counter.second);
        ReleaseMemHandles();
        isNeedOpCounter_ = false;
        HCCL_RUN_INFO("free counter mem resource");
    }
    return HCCL_SUCCESS;
}

HcclResult OpExeCounter::AddCounter(
    const HcclDispatcher& dispatcher, Stream& stream, int flag) // flag 0为下发前计数，1为下发后计数
{
    if (!isNeedOpCounter_) {
        HCCL_DEBUG("do not need add counter");
        return HCCL_SUCCESS;
    }
    if ((stream.ptr() == nullptr)) {
        HCCL_WARNING("stream is nullptr");
        return HCCL_SUCCESS;
    }
    CHK_RET(HcclReduceAsync(
        dispatcher, static_cast<void*>(addOneMem_), 1, HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, stream,
        (flag == HEAD) ? static_cast<void*>(headCountMem_) : static_cast<void*>(tailCountMem_), INVALID_VALUE_RANKID,
        LinkType::LINK_ONCHIP, INLINE_REDUCE_BIT));

    HCCL_DEBUG("add %s count.", (flag == HEAD) ? "head" : "tail");

    return HCCL_SUCCESS;
}

HcclResult OpExeCounter::GetCounter(std::pair<int32_t, int32_t>& counter)
{
    if (!isNeedOpCounter_) {
        HCCL_DEBUG("do not need add counter");
        return HCCL_SUCCESS;
    }
    CHK_RET(hrtMemSyncCopy(
        &counter.first, memSize_, headCountMem_, memSize_, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST));

    CHK_RET(hrtMemSyncCopy(
        &counter.second, memSize_, tailCountMem_, memSize_, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST));

    HCCL_DEBUG("head:%d, tail:%d", counter.first, counter.second);

    return HCCL_SUCCESS;
}

HcclResult OpExeCounter::GetOpCountInfo(OpCounterInfo& opCounterInfo)
{
    opCounterInfo.isEnableCounter = GetExternalInputStuckDetect();
    if (!isNeedOpCounter_) {
        HCCL_DEBUG("do not need add counter");
        return HCCL_SUCCESS;
    }

    if (headCountMem_ == nullptr || tailCountMem_ == nullptr || addOneMem_ == nullptr) {
        HCCL_ERROR("[OpExeCounter][GetOpCountInfo] aicpu headCountMem or tailCountMem or addOneMem is nullptr");
        return HCCL_E_PTR;
    }
    opCounterInfo.headCountMem = reinterpret_cast<u64>(headCountMem_);
    opCounterInfo.tailCountMem = reinterpret_cast<u64>(tailCountMem_);
    opCounterInfo.addOneMem = reinterpret_cast<u64>(addOneMem_);
    opCounterInfo.memSize = memSize_;
    return HCCL_SUCCESS;
}

HcclResult OpExeCounter::ClearOpCounterMem()
{
    if (!isNeedOpCounter_) {
        HCCL_DEBUG("do not need add counter");
        return HCCL_SUCCESS;
    }
    if (headCountMem_ == nullptr || tailCountMem_ == nullptr) {
        HCCL_ERROR("[OpExeCounter][ClearOpCounterMem] headCountMem or tailCountMem is nullptr");
        return HCCL_E_PTR;
    }
    CHK_RET(hrtMemSet(headCountMem_, memSize_, memSize_));
    CHK_RET(hrtMemSet(tailCountMem_, memSize_, memSize_));
    HCCL_DEBUG("[OpExeCounter][ClearOpCounterMem] headCountMem or tailCountMem is successfully set to 0");
    return HCCL_SUCCESS;
}

void OpExeCounter::ReleaseMemHandles()
{
    if (headCountMem_ != nullptr) {
        CHK_PRT(hrtFree(headCountMem_));
        headCountMem_ = nullptr;
    }
    if (tailCountMem_ != nullptr) {
        CHK_PRT(hrtFree(tailCountMem_));
        tailCountMem_ = nullptr;
    }
    if (addOneMem_ != nullptr) {
        CHK_PRT(hrtFree(addOneMem_));
        addOneMem_ = nullptr;
    }
}

HcclResult FftsHeadCounter(const HcclDispatcher& dispatcher, Stream& stream)
{
    if (!GetExternalInputHcclEnableFfts()
        || GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        HCCL_DEBUG("do not need add ffts mode counter");
        return HCCL_SUCCESS;
    }
    s32 devLogicID = 0;
    CHK_RET(hrtGetDevice(&devLogicID));
    return OpExeCounter::GetInstance(devLogicID).AddCounter(dispatcher, stream, HEAD);
}

HcclResult FftsTailCounter(const HcclDispatcher& dispatcher, Stream& stream)
{
    if (!GetExternalInputHcclEnableFfts()
        || GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB) {
        HCCL_DEBUG("do not need add ffts mode counter");
        return HCCL_SUCCESS;
    }
    s32 devLogicID = 0;
    CHK_RET(hrtGetDevice(&devLogicID));
    return OpExeCounter::GetInstance(devLogicID).AddCounter(dispatcher, stream, TAIL);
}

HcclResult
StarsCounter(const HcclDispatcher& dispatcher, Stream& stream, int flag, bool isAicpuMode, bool isRetry, bool isAivMode)
{
    // 不需要STARS头尾计数的场景: AICPU展开不开重执行 或者 AIV 或者 HOST展开FFTS+模式
    if ((isAicpuMode && !isRetry) || isAivMode
        || (!isAicpuMode && GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
            && GetExternalInputHcclEnableFfts())) {
        HCCL_DEBUG("do not need add stars mode counter");
        return HCCL_SUCCESS;
    }
    // 需要添加STARS头尾计数的场景: AICPU开启重执行 或者 非AICPU展开 单算子STARS、图模式
    s32 devLogicID = 0;
    CHK_RET(hrtGetDevice(&devLogicID));
    return OpExeCounter::GetInstance(devLogicID).AddCounter(dispatcher, stream, flag);
}

HcclResult GetOpCountInfo(OpCounterInfo& opCounterInfo)
{
    s32 devLogicID = 0;
    CHK_RET(hrtGetDevice(&devLogicID));
    return OpExeCounter::GetInstance(devLogicID).GetOpCountInfo(opCounterInfo);
}

HcclResult ClearOpCounterMem()
{
    s32 devLogicID = 0;
    CHK_RET(hrtGetDevice(&devLogicID));
    return OpExeCounter::GetInstance(devLogicID).ClearOpCounterMem();
}

__attribute__((constructor)) void CallBackInit()
{
    RegisterInitTaskCallBack(FftsHeadCounter);
    RegisterLaunchTaskCallBack(FftsTailCounter);
}

} // namespace hccl

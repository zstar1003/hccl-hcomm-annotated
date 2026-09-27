/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dispatcher_ctx.h"
#include "adapter_hal.h"
#include "dispatcher_graph_pub.h"
#include "dispatcher_pub.h"
#include "dispatcher_aicpu_pub.h"
#include "dispatcher_virtural_pub.h"
#include "dlhal_function.h"

namespace hccl {
FftsCounterCallBack DispatcherCtx::GetInitTaskCallback() const { return g_InitTaskCallback; }

FftsCounterCallBack DispatcherCtx::GetLaunchTaskCallback() const { return g_LaunchTaskCallback; }

HcclResult DispatcherCtx::Init()
{
    CHK_RET(DlHalFunction::GetInstance().DlHalFunctionInit());
    // 获取host侧还是device侧
    u32 info = 0;
    CHK_RET(hrtDrvGetPlatformInfo(&info));
    bool isDeviceSide = (info == 0);
    HCCL_INFO("[DispatcherCtx][Init] isDeviceSide[%d]", isDeviceSide);
    CtxDispatcherType type = CtxDispatcherType::DISPATCHER_NORMAL;
    // 如果是host侧
    if (!isDeviceSide) {
        CHK_RET(hrtGetDeviceType(deviceType_));
        if ((deviceType_ == DevType::DEV_TYPE_910B) && GetExternalInputHcclEnableFfts()) {
            CHK_PRT_CONT(
                GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && !GetExternalInputHcclAicpuUnfold(),
                HCCL_RUN_INFO("[DispatcherCtx][Init] Will use FFTS mode."));
            type = CtxDispatcherType::DISPATCHER_FFTS;
        } else {
            HCCL_RUN_INFO("[DispatcherCtx][Init] Will use NORMAL mode.");
            type = CtxDispatcherType::DISPATCHER_NORMAL;
        }
    } else { // 如果是device侧 那么默认aicpu
        HCCL_RUN_INFO("[DispatcherCtx][Init] Will use AICPU mode.");
        type = CtxDispatcherType::DISPATCHER_AICPU;
    }
    CHK_RET(DispatcherInit(type, devicePhyId_, &dispatcher_));
    CHK_SMART_PTR_NULL(dispatcher_);

    return HCCL_SUCCESS;
}
HcclResult DispatcherCtx::Destroy()
{
    const std::lock_guard<std::mutex> lock(destroyMutex_);
    if (dispatcher_ != nullptr) {
        DispatcherPub* dispatcher = reinterpret_cast<DispatcherPub*>(dispatcher_);
        delete dispatcher;
        dispatcher_ = nullptr;
    }
    return HCCL_SUCCESS;
}
HcclDispatcher DispatcherCtx::GetDispatcher() const { return dispatcher_; }

u32 DispatcherCtx::GetWaitTimeOut() const { return waitTimeOut_; }

HcclResult DispatcherCtx::SetWaitTimeOut(u32 waitTimeOut)
{
    waitTimeOut_ = waitTimeOut;
    return HCCL_SUCCESS;
}

HcclResult DispatcherCtx::DispatcherInit(CtxDispatcherType type, const s32 devicePhyId, HcclDispatcher* dispatcher)
{
    CHK_RET(DlHalFunction::GetInstance().DlHalFunctionInit());
    CHK_PTR_NULL(dispatcher);
    dispatcherType_ = type;
    DispatcherPub* pDispatcher = nullptr;
    switch (type) {
        case CtxDispatcherType::DISPATCHER_FFTS: {
            u32 deviceLogicId = INVALID_UINT;
            CHK_RET(hrtGetDeviceIndexByPhyId(devicePhyId, deviceLogicId));
#ifndef HCCD
            pDispatcher = new (std::nothrow) DispatcherGraph(deviceLogicId);
#endif
            break;
        }
        case CtxDispatcherType::DISPATCHER_NORMAL: {
            u32 deviceLogicId = INVALID_UINT;
            CHK_RET(hrtGetDeviceIndexByPhyId(devicePhyId, deviceLogicId));
            pDispatcher = new (std::nothrow) DispatcherPub(deviceLogicId);
            break;
        }
        case CtxDispatcherType::DISPATCHER_VIRTURAL: {
            u32 deviceLogicId = INVALID_UINT;
            CHK_RET(hrtGetDeviceIndexByPhyId(devicePhyId, deviceLogicId));
            pDispatcher = new (std::nothrow) DispatcherVirtural(deviceLogicId);
            break;
        }
        case CtxDispatcherType::DISPATCHER_AICPU: {
#ifdef CCL_KERNEL
            pDispatcher = new (std::nothrow) DispatcherAiCpu(devicePhyId);
#endif
            break;
        }
        default: {
            HCCL_ERROR("Not support the dispatcher type[%d]", type);
            return HCCL_E_NOT_SUPPORT;
        }
    }

    CHK_PTR_NULL(pDispatcher);
    HcclResult ret = pDispatcher->Init();
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("Dispatcher init failed, type[%d]", type);
        delete pDispatcher;
        pDispatcher = nullptr;
        return ret;
    }
    *dispatcher = pDispatcher;
    return HCCL_SUCCESS;
}

HcclResult DispatcherCtx::SetDispatcherHcclQos(u32 hcclQos)
{
    HCCL_INFO("SetDispatcherHcclQos hcclQos = %u", hcclQos);
    CHK_PTR_NULL(dispatcher_);
    auto aiCpuDispatcher = static_cast<DispatcherAiCpu*>(dispatcher_);
    aiCpuDispatcher->SetHcclQos(hcclQos);
    return HCCL_SUCCESS;
}
} // namespace hccl

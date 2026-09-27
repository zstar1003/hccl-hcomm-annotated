/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include "hcomm_c_adpt.h"
#include "hcomm_c_adpt_common.h"
#include "log.h"
#include "hcomm_res_defs.h"
#include "hcom_common.h"
#include "endpoint.h"
#include "thread.h"
#include "aicpu_ts_thread.h"
#include "cpu_ts_thread.h"
#include "aicpu_ts_urma_channel.h"
#include "mem_device_pub.h"
#include "channel_param.h"
#include "env_config/env_config_v2.h"
#include "hcomm_res_mgr.h"
#include "param_check_pub.h"
#include "comm_engine_utils.h"
#include "exception_handler.h"
#include "hcclCommDfx.h"
#include "launch_device.h"
#include "shared_jetty_mgr.h"
#include "launch_aicpu.h"
#include "comm_configer.h"
#include "hcomm_adapter_runtime.h"

using namespace hcomm;

namespace {
HcclResult RefreshCurrentDeviceContext()
{
    s32 deviceLogicId = 0;
    CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
    u32 devicePhyId = 0;
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(deviceLogicId), devicePhyId, true));
    HCCL_INFO("[RefreshCurrentDeviceContext] deviceLogicId[%d], devicePhyId[%u].", deviceLogicId, devicePhyId);
    return HCCL_SUCCESS;
}
} // namespace

HcclResult RefreshEndpointContext(const EndpointDesc& endpointDesc)
{
    if (endpointDesc.loc.locType != ENDPOINT_LOC_TYPE_DEVICE) {
        return HCCL_SUCCESS;
    }
    return RefreshCurrentDeviceContext();
}

HcclResult RefreshCommEngineContext(CommEngine engine)
{
    if (engine != COMM_ENGINE_AICPU && engine != COMM_ENGINE_AICPU_TS) {
        return HCCL_SUCCESS;
    }
    return RefreshCurrentDeviceContext();
}

HcommResult HcommResMgrInit(uint32_t devPhyId)
{
    // 预热 SharedJettyMgr：使其先于 HcommResMgr 构造（从而先于 CollCommMgr 构造），
    // 后于两者析构。否则 SharedJettyMgr 在运行期由 HcommChannelCreate/Destroy 首次触发，
    // 晚于 CollCommMgr 构造，进程退出时先于 CollCommMgr 析构；
    // ~CollCommMgr → ~hccl::EndpointMgr → HcommEndpointDestroy → CheckEndpointDestroy
    // 会命中已析构的 SharedJettyMgr 静态对象（mtx_/contexts_ 已销毁）。
    (void)SharedJettyMgr::GetInstance();

    bool noDevice = false;
    if (devPhyId == UINT32_MAX) {
        CHK_RET(ResolveRuntimeDevicePhyId(devPhyId, noDevice));
    }

    // 临时方案：触发统一平台层单例触发静态对象声明
    // 内部流程触发各种单例声明，保证时序
    EXCEPTION_HANDLE_BEGIN
    if (noDevice) {
        (void)HcommResMgr::GetInstance().GetDeviceResMgr(devPhyId);
        return HCCL_SUCCESS;
    }

    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        (void)HcommResMgr::GetInstance().GetDeviceResMgr(devPhyId);
        return HcclResult::HCCL_SUCCESS;
    }());
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

HcommResult HcommDfxKernelLaunch(const std::string& commTag, aclrtBinHandle binHandle, HcclDfxOpInfo dfxOpInfo)
{
    // 申请device侧内存
    hccl::DeviceMem devicePackBuf = hccl::DeviceMem::alloc(sizeof(dfxOpInfo));
    CHK_PTR_NULL(devicePackBuf.ptr());

    // 将dfxOpInfo信息传递给device侧
    CHK_RET(hrtMemSyncCopy(
        devicePackBuf.ptr(), sizeof(dfxOpInfo), &dfxOpInfo, sizeof(dfxOpInfo),
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    // 创建局部流
    hccl::Stream localStream(hccl::StreamType::STREAM_TYPE_ONLINE);
    constexpr u32 aicpuStreamMode = 1;
    CHK_RET(hrtStreamSetMode(localStream.ptr(), aicpuStreamMode));

    // 下kernel
    std::string kernelName = "RunAicpuDfxInitV2";

    struct InitTask {
        u64 context;
        char commTag[256];
    };

    InitTask customInitTask = {0, ""};
    customInitTask.context = ReinterpretAs<u64>(devicePackBuf.ptr());
    s32 sRet = strncpy_s(customInitTask.commTag, sizeof(customInitTask.commTag), commTag.c_str(), TAG_MAX_LENGTH - 1);
    CHK_PRT_RET(sRet != EOK, HCCL_ERROR("[%s] str copy fail. return[%d]", __func__, sRet), HCCL_E_INTERNAL);

    CHK_RET(hccl::AicpuAclKernelLaunch(
        localStream.ptr(), ReinterpretAs<void*>(&customInitTask), sizeof(customInitTask), binHandle, kernelName, true,
        NOTIFY_DEFAULT_WAIT_TIME));

    CHK_RET(
        hcclStreamSynchronize(localStream.ptr(), hccl::CommConfiger::GetInstance().GetCommConfigExecTimeOut(commTag)));

    HCCL_INFO("[%s] channel kernel launch success.", __func__);

    return HCCL_SUCCESS;
}

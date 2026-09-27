/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ub_rtp_endpoint.h"
#include "log.h"
#include "hccl/hccl_res.h"
#include "ub_reged_mem_mgr.h"
#include "adapter_rts_common.h"
#include "rdma_handle_manager.h"
#include "mgr/endpoint_ctx_mgr.h"
#include "hcomm_res_mgr.h"

namespace hcomm {

UbRtpEndpoint::UbRtpEndpoint(const EndpointDesc& endpointDesc) : Endpoint(endpointDesc) {}

UbRtpEndpoint::~UbRtpEndpoint() noexcept { (void)ReleaseEndpointCtx(); }

bool UbRtpEndpoint::IsCtxHandleValid() const
{
    if (ctxHandle_ == nullptr) {
        return false;
    }
    return Hccl::RdmaHandleManager::GetInstance().IsHandleValid(static_cast<Hccl::RdmaHandle>(ctxHandle_));
}

CommQueueContext* UbRtpEndpoint::GetCommQueueContext()
{
    std::call_once(jettyContextOnce_, [this] {
        jettyContext_ = std::make_unique<JettyContext>();
    });
    return jettyContext_.get();
}

HcclResult UbRtpEndpoint::ReleaseEndpointCtx()
{
    if (endpointCtx_ == nullptr) {
        HCCL_WARNING("[UbRtpEndpoint][%s] endpointCtx_ is null, nothing to release", __func__);
        return HCCL_E_PTR;
    }
    // 先放掉自身引用，再触发 EndpointCtxMgr 移除
    EndpointCtxKey key = endpointCtx_->key;
    endpointCtx_.reset();
    HcommResMgr::GetInstance().GetDeviceResMgr(key.devPhyId).GetEndpointCtxMgr().Release(key);
    return HCCL_SUCCESS;
}

HcclResult UbRtpEndpoint::Init()
{
    HCCL_INFO("[%s] localEndpoint protocol[%d]", __func__, endpointDesc_.protocol);

    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (deviceType != DevType::DEV_TYPE_950 && deviceType != DevType::DEV_TYPE_960) {
        HCCL_ERROR(
            "[%s] UB_RTP protocol only supports DEV_TYPE_950/960, current deviceType=%d", __func__,
            static_cast<int>(deviceType));
        return HCCL_E_NOT_SUPPORT;
    }

    // UB_RTP 直接从 EID type CommAddr 获取地址，不做 IP→EID 转换
    Hccl::IpAddress eidAddr{};
    CHK_RET(CommAddrToIpAddress(endpointDesc_.commAddr, eidAddr));

    s32 deviceLogicId;
    u32 devPhyId;
    CHK_RET(hrtGetDevice(&deviceLogicId));
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(deviceLogicId), devPhyId));
    endpointDesc_.loc.device.devPhyId = devPhyId;

    Hccl::HccpHdcManager::GetInstance().Init(deviceLogicId);
    EndpointCtxKey ctxKey{devPhyId, COMM_PROTOCOL_UB_CTP, endpointDesc_.loc.locType, eidAddr};
    CHK_RET(HcommResMgr::GetInstance()
                .GetDeviceResMgr(ctxKey.devPhyId)
                .GetEndpointCtxMgr()
                .Acquire(ctxKey, false, endpointCtx_));
    CHK_PTR_NULL(endpointCtx_);
    ctxHandle_ = endpointCtx_->ctxHandle;
    HCCL_INFO(
        "%s success, devPhyId[%u], eidAddr[%s], ctxHandle[%p]", __func__, devPhyId, eidAddr.Describe().c_str(),
        ctxHandle_);

    EXCEPTION_CATCH(regedMemMgr_ = std::make_shared<UbRegedMemMgr>(ctxHandle_), {
        CHK_RET(ReleaseEndpointCtx());
        return HCCL_E_INTERNAL;
    });

    return HcclResult::HCCL_SUCCESS;
}

} // namespace hcomm

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_jetty.h"

#include "hccp_ctx.h"
#include "rdma_handle_manager.h"
#include "local_ub_rma_buffer.h"

namespace Hccl {

HcclResult CcuCreateJetty(const IpAddress& ipAddr, const CcuJettyInfo& jettyInfo, std::unique_ptr<CcuJetty>& ccuJetty)
{
    TRY_CATCH_RETURN(ccuJetty = std::make_unique<CcuJetty>(ipAddr, jettyInfo););
    return HcclResult::HCCL_SUCCESS;
}

CcuJetty::CcuJetty(const IpAddress& ipAddr, const CcuJettyInfo& jettyInfo) : ipAddr_(ipAddr), jettyInfo_(jettyInfo)
{
    devLogicId_ = HrtGetDevice();
    uint32_t devPhyId = HrtGetDevicePhyIdByUserDevId(devLogicId_);
    auto& rdmaHandleMgr = RdmaHandleManager::GetInstance();
    rdmaHandle_ = rdmaHandleMgr.GetByIp(devPhyId, ipAddr);
    const auto jfcHandle = 1;
    const auto tokenValue = GetUbToken();
    const auto jettyMode = HrtJettyMode::CCU_CCUM_CACHE; // 当前仅支持该模式

    inParam_ = HrtRaUbCreateJettyParam{
        jfcHandle,
        jfcHandle,
        tokenValue,
        0,
        jettyMode,
        jettyInfo.taJettyId,
        jettyInfo.sqBufVa,
        jettyInfo.sqBufSize,
        jettyInfo.wqeBBStartId,
        jettyInfo.sqDepth};
}

CcuJetty::~CcuJetty() {}

HcclResult CcuJetty::CreateJetty()
{
    return HcclResult::HCCL_SUCCESS; // 打桩保证创建一定成功
}

HrtRaUbCreateJettyParam CcuJetty::GetCreateJettyParam() const { return inParam_; }

HrtRaUbJettyCreatedOutParam CcuJetty::GetJettyedOutParam() const { return outParam_; }

} // namespace Hccl

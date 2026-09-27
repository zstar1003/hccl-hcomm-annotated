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
    Hccl::CqCreateInfo cqInfo{};
    uint32_t devPhyId = HrtGetDevicePhyIdByUserDevId(devLogicId_);
    auto& rdmaHandleMgr = RdmaHandleManager::GetInstance();
    rdmaHandle_ = rdmaHandleMgr.GetByIp(devPhyId, ipAddr);
    const auto jfcHandle = rdmaHandleMgr.GetJfcHandle(rdmaHandle_, cqInfo, HrtUbJfcMode::CCU_POLL);
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

CcuJetty::~CcuJetty()
{
    DECTOR_TRY_CATCH("CcuJetty", {
        if (isCreated_ && outParam_.handle != 0) {
            HrtRaUbDestroyJetty(outParam_.handle);
        }
    });
}

bool CheckRequestResult(RequestHandle& reqHandle)
{
    if (reqHandle == 0) {
        return true;
    }

    ReqHandleResult result = HrtRaGetAsyncReqResult(reqHandle);
    if (result == ReqHandleResult::NOT_COMPLETED) {
        return false;
    }

    if (result != ReqHandleResult::COMPLETED) {
        THROW<InternalException>(
            "[CcuJetty][%s] failed, result[%s] is unexpected.", __func__, result.Describe().c_str());
    }

    return true;
}

static HcclResult ParseCreateInfo(
    const struct QpCreateInfo* infoPtr, const JettyHandle jettyHandle, HrtRaUbJettyCreatedOutParam& outParam)
{
    outParam.handle = jettyHandle;
    auto ret = memcpy_s(outParam.key, HRT_UB_QP_KEY_MAX_LEN, infoPtr->key.value, infoPtr->key.size);
    if (ret != 0) {
        HCCL_ERROR("[CcuJetty][%s] create info key memcpy_s failed, ret[%d].", __func__, ret);
        return HcclResult::HCCL_E_MEMORY;
    }
    outParam.jettyVa = infoPtr->va;
    outParam.uasid = infoPtr->ub.uasid;
    outParam.id = infoPtr->ub.id;
    outParam.keySize = infoPtr->key.size;
    outParam.dbVa = infoPtr->ub.dbAddr;
    outParam.dbTokenId = infoPtr->ub.dbTokenId >> URMA_TOKEN_ID_RIGHT_SHIFT;
    // 不提供 tokenValue，不得打印token相关信息
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuJetty::HandleAsyncRequest()
{
    TRY_CATCH_RETURN(
        if (reqHandle_ == 0) {
            reqHandle_ = RaUbCreateJettyAsync(rdmaHandle_, inParam_, reqDataBuffer_, jettyHandlePtr_);
            return HcclResult::HCCL_E_AGAIN;
        }

        if (!CheckRequestResult(reqHandle_)) { return HcclResult::HCCL_E_AGAIN; };);

    const struct QpCreateInfo* info = reinterpret_cast<const QpCreateInfo*>(reqDataBuffer_.data());
    const JettyHandle jettyHandle = reinterpret_cast<JettyHandle>(jettyHandlePtr_);
    return ParseCreateInfo(info, jettyHandle, outParam_);
}

HcclResult CcuJetty::CreateJetty()
{
    if (isError_) {
        HCCL_ERROR(
            "[CcuJetty][%s] failed, jetty[%u] is error, "
            "refused to create.",
            __func__, inParam_.jettyId);
        return HcclResult::HCCL_E_INTERNAL;
    }

    if (isCreated_) {
        HCCL_INFO("[CcuJetty][%s] passed, jetty[%u] has been created.", __func__, inParam_.jettyId);
        return HcclResult::HCCL_SUCCESS;
    }

    auto ret = HandleAsyncRequest();
    if (ret == HcclResult::HCCL_SUCCESS) {
        isCreated_ = true;
    } else if (ret != HcclResult::HCCL_E_AGAIN) {
        isError_ = true;
    }

    return ret;
}

HrtRaUbCreateJettyParam CcuJetty::GetCreateJettyParam() const { return inParam_; }

HrtRaUbJettyCreatedOutParam CcuJetty::GetJettyedOutParam() const { return outParam_; }

void CcuJetty::GetJettyInfo(ConnJettyInfo& connJettyInfo)
{
    if (isCreated_ && outParam_.handle != 0) {
        connJettyInfo.localJetty = outParam_.handle;
    }
}

HcclResult CcuJetty::Clean()
{
    TRY_CATCH_RETURN(if (isCreated_ && outParam_.handle != 0) {
        isCreated_ = false;
        reqHandle_ = 0;
        jettyHandlePtr_ = nullptr;
        reqDataBuffer_.clear();
    } isError_ = false;);
    return HcclResult::HCCL_SUCCESS;
}
} // namespace Hccl

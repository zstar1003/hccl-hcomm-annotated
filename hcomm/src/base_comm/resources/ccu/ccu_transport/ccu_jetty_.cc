/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_jetty_.h"

#include "hcom_common.h"

#include "hccp_ctx.h"

#include "exception_handler.h"

// 当前复用orion数据结构
#include "rdma_handle_manager.h"
#include "local_ub_rma_buffer.h"
#include "orion_adapter_hccp.h"

namespace hcomm {

HcclResult
CcuCreateJetty(const Hccl::IpAddress& ipAddr, const CcuJettyInfo& jettyInfo, std::unique_ptr<CcuJetty>& ccuJetty)
{
    EXCEPTION_HANDLE_BEGIN

    ccuJetty = std::make_unique<CcuJetty>(ipAddr, jettyInfo);
    CHK_RET(ccuJetty->Init());

    EXCEPTION_HANDLE_END
    return HcclResult::HCCL_SUCCESS;
}

CcuJetty::CcuJetty(const Hccl::IpAddress& ipAddr, const CcuJettyInfo& jettyInfo)
    : ipAddr_(ipAddr),
      jettyInfo_(jettyInfo)
{}

HcclResult CcuJetty::Init()
{
    EXCEPTION_HANDLE_BEGIN
    userDevId_ = HcclGetThreadDeviceId();
    uint32_t devPhyId{0};
    Hccl::CqCreateInfo cqInfo{};
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<uint32_t>(userDevId_), devPhyId));
    auto& rdmaHandleMgr = Hccl::RdmaHandleManager::GetInstance();
    ctxHandle_ = rdmaHandleMgr.GetByIp(devPhyId, ipAddr_);
    CHK_PRT_RET(
        !rdmaHandleMgr.IsHandleValid(ctxHandle_),
        HCCL_ERROR(
            "[CcuJetty][%s] ctxHandle_[%p] is not valid, "
            "RdmaHandleManager may have DeInit this device",
            __func__, ctxHandle_),
        HcclResult::HCCL_E_INTERNAL);
    const auto _jfcHandle = rdmaHandleMgr.GetJfcHandle(ctxHandle_, cqInfo, Hccl::HrtUbJfcMode::CCU_POLL);
    const JfcHandle jfcHandle = reinterpret_cast<JfcHandle>(_jfcHandle);
    const auto tokenValue = Hccl::GetUbToken();
    const auto jettyMode = jettyInfo_.jettyType == CcuJettyType::CCUM_CACHED_JETTY ? HrtJettyMode::CCU_CCUM_CACHE :
                                                                                     HrtJettyMode::CCU_TA_CACHE;

    inParam_ = HrtRaUbCreateJettyParam{
        jfcHandle,
        jfcHandle,
        tokenValue,
        nullptr,
        jettyMode,
        jettyInfo_.taJettyId,
        jettyInfo_.sqBufVa,
        jettyInfo_.sqBufSize,
        jettyInfo_.wqeBBStartId,
        jettyInfo_.sqDepth}; // CTP默认为8s
    EXCEPTION_HANDLE_END

    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuJetty::SetMappedJettyPriority(uint32_t priority)
{
    const uint8_t mapped = static_cast<uint8_t>(priority & 0xFU);

    if (mappedJettyPrioritySet_ && mappedJettyPriority_ != mapped) {
        HCCL_ERROR(
            "[CcuJetty][%s] mappedJettyPriority conflict on shared jetty: existing[%u] new[%u] "
            "jettyId[%u] isCreated[%d].",
            __func__, static_cast<unsigned>(mappedJettyPriority_), static_cast<unsigned>(mapped), jettyInfo_.taJettyId,
            static_cast<int>(isCreated_));
        return HcclResult::HCCL_E_INTERNAL;
    }

    // 多 channel 复用：jetty 已 create，qos 已写入 URMA，不可再改 inParam_
    if (isCreated_) {
        HCCL_INFO(
            "[CcuJetty][%s] jetty[%u] already created, skip mappedJettyPriority[%u].", __func__, jettyInfo_.taJettyId,
            static_cast<unsigned>(mapped));
        return HcclResult::HCCL_SUCCESS;
    }

    if (mappedJettyPrioritySet_) {
        return HcclResult::HCCL_SUCCESS;
    }

    mappedJettyPriority_ = mapped;
    mappedJettyPrioritySet_ = true;
    inParam_.qos = mapped;
    return HcclResult::HCCL_SUCCESS;
}

CcuJetty::~CcuJetty() { (void)Clean(); }

static HcclResult CheckRequestResult(RequestHandle& reqHandle)
{
    if (reqHandle == 0) {
        return HcclResult::HCCL_SUCCESS;
    }

    RequestResult result = HccpGetAsyncReqResult(reqHandle);
    if (result == RequestResult::NOT_COMPLETED) {
        return HcclResult::HCCL_E_AGAIN;
    }

    if (result != RequestResult::COMPLETED) {
        HCCL_ERROR("[TpMgr][%s] failed, result[%s] is unexpected.", __func__, result.Describe().c_str());
        return HcclResult::HCCL_E_NETWORK;
    }

    return HcclResult::HCCL_SUCCESS;
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

    constexpr uint32_t URMA_TOKEN_ID_RIGHT_SHIFT = 8;

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
    if (reqHandle_ == 0) {
        CHK_RET(HccpUbCreateJettyAsync(ctxHandle_, inParam_, reqDataBuffer_, jettyHandlePtr_, reqHandle_));
        return HcclResult::HCCL_E_AGAIN; // 首次触发异步接口调用，动作一定未完成
    }

    auto ret = CheckRequestResult(reqHandle_);
    if (ret == HcclResult::HCCL_E_AGAIN) {
        return ret;
    }
    CHK_RET(ret);

    const struct QpCreateInfo* info = reinterpret_cast<const QpCreateInfo*>(reqDataBuffer_.data());
    const JettyHandle jettyHandle = reinterpret_cast<JettyHandle>(jettyHandlePtr_);
    return ParseCreateInfo(info, jettyHandle, outParam_);
}

HcclResult CcuJetty::CreateJetty(u8 errTimeout)
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

    inParam_.errTimeout = errTimeout;
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

HcclResult CcuJetty::Clean()
{
    if (isCreated_ && outParam_.handle != 0) {
        auto jettyHandle = outParam_.handle;
        outParam_ = {}; // 移动handle并置空，防止二次释放
        isCreated_ = false;
        reqHandle_ = 0;
        jettyHandlePtr_ = nullptr;
        reqDataBuffer_.clear();

        auto ret = RaCtxQpDestroy(jettyHandle);
        if (ret != 0) {
            HCCL_ERROR("[CcuJetty][%s] failed, jettyHandle[0x%llx].", __func__, jettyHandle);
            return HcclResult::HCCL_E_NETWORK;
        }
    }
    isError_ = false;
    return HcclResult::HCCL_SUCCESS;
}
} // namespace hcomm

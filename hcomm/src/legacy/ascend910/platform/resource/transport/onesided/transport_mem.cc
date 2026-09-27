/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "transport_mem.h"
#include "log.h"
#include "transport_ipc_mem.h"
#include "transport_roce_mem.h"
#ifdef CCL_KERNEL
#include "transport_device_roce_mem.h"
#endif

namespace hccl {
constexpr u32 INVALID_REMOTE_RANK_ID = 0xFFFFFFFF;
TransportMem::TransportMem(
    const std::unique_ptr<NotifyPool>& notifyPool, const HcclNetDevCtx& netDevCtx, const HcclDispatcher& dispatcher,
    AttrInfo& attrInfo)
    : TransportMem(notifyPool, netDevCtx, dispatcher, attrInfo, false)
{}

TransportMem::TransportMem(
    const std::unique_ptr<NotifyPool>& notifyPool, const HcclNetDevCtx& netDevCtx, const HcclDispatcher& dispatcher,
    AttrInfo& attrInfo, bool aicpuUnfoldMode)
    : notifyPool_(notifyPool),
      netDevCtx_(netDevCtx),
      dispatcher_(dispatcher),
      localRankId_(attrInfo.localRankId),
      remoteRankId_(attrInfo.remoteRankId),
      aicpuUnfoldMode_(aicpuUnfoldMode)
{}

TransportMem::~TransportMem() {}

// static
std::shared_ptr<TransportMem> TransportMem::Create(
    TpType tpType, const std::unique_ptr<NotifyPool>& notifyPool, const HcclNetDevCtx& netDevCtx,
    const HcclDispatcher& dispatcher, AttrInfo& attrInfo)
{
    return Create(tpType, notifyPool, netDevCtx, dispatcher, attrInfo, false);
}

std::shared_ptr<TransportMem> TransportMem::Create(
    TpType tpType, [[maybe_unused]] const std::unique_ptr<NotifyPool>& notifyPool,
    [[maybe_unused]] const HcclNetDevCtx& netDevCtx, [[maybe_unused]] const HcclDispatcher& dispatcher,
    [[maybe_unused]] AttrInfo& attrInfo, [[maybe_unused]] bool aicpuUnfoldMode)
{
    std::shared_ptr<TransportMem> transportMemPtr;
#if !defined(CCL_KERNEL) || defined(CCL_LLT)
    CHK_PRT_RET((netDevCtx == nullptr), HCCL_ERROR("[TransportMem][Create]netDevCtx is null"), nullptr);
    HCCL_DEBUG(
        "transportMem create tpType:%u netDevCtx:%p dispatcher:%p localRankId:%u remoteRankId:%u sdid:%u "
        "serverId:%u trafficClass:%u serviceLevel:%u",
        tpType, netDevCtx, dispatcher, attrInfo.localRankId, attrInfo.remoteRankId, attrInfo.sdid, attrInfo.serverId,
        attrInfo.trafficClass, attrInfo.serviceLevel);
    switch (tpType) {
        case TpType::ROCE:
            transportMemPtr
                = std::make_unique<TransportRoceMem>(notifyPool, netDevCtx, dispatcher, attrInfo, aicpuUnfoldMode);
            break;
        case TpType::IPC:
            transportMemPtr
                = std::make_unique<TransportIpcMem>(notifyPool, netDevCtx, dispatcher, attrInfo, aicpuUnfoldMode);
            break;
        default:
            break;
    }
#else
    HCCL_ERROR("[TransportMem] The Create interface with qpInfo should be used on the AICPU, tpType[%u]", tpType);
#endif
    return transportMemPtr;
}

std::shared_ptr<TransportMem> TransportMem::Create(
    TpType tpType, [[maybe_unused]] const HcclQpInfoV2& qpInfo, const HcclDispatcher& dispatcher, AttrInfo& attrInfo)
{
    const std::unique_ptr<NotifyPool> notifyPool = nullptr; // dummy for device ibv transport
    const HcclNetDevCtx netDevCtx = nullptr;                // dummy
    HCCL_DEBUG(
        "[TransportMem] create tpType:%u netDevCtx:%p dispatcher:%p localRankId:%u remoteRankId:%u", tpType, netDevCtx,
        dispatcher, attrInfo.localRankId, attrInfo.remoteRankId);
    std::shared_ptr<TransportMem> transportMemPtr;
    switch (tpType) {
        case TpType::ROCE_DEVICE:
#ifdef CCL_KERNEL
            transportMemPtr = std::make_unique<TransportDeviceRoceMem>(
                notifyPool, netDevCtx, dispatcher, attrInfo, false, qpInfo); // aicpuUnfoldMode is set by host
#else
            HCCL_ERROR("[TransportMem] ROCE_DEVICE Only running on the AICPU");
#endif
            break;
        default:
            HCCL_ERROR("[TransportMem] unsupported TpType[%u] on the AICPU", tpType);
            break;
    }
    return transportMemPtr;
}

HcclResult TransportMem::SetDataSocket(const std::shared_ptr<HcclSocket>& socket)
{
    dataSocket_ = socket;
    return HCCL_SUCCESS;
}

HcclResult
TransportMem::DoExchangeMemDesc(const RmaMemDescs& localMemDescs, RmaMemDescs& remoteMemDescs, u32& actualNumOfRemote)
{
    HCCL_INFO(
        "[HcclOneSidedConn][ExchangeMemDesc]localRank[%u] exchange memDesc begin, role[%u]", localRankId_,
        dataSocket_->GetLocalRole());

    if (dataSocket_->GetLocalRole() == HcclSocketRole::SOCKET_ROLE_CLIENT) {
        // 先收后发
        CHK_RET(ReceiveRemoteMemDesc(remoteMemDescs, actualNumOfRemote));
        CHK_RET(SendLocalMemDesc(localMemDescs));
    } else {
        // 先发后收
        CHK_RET(SendLocalMemDesc(localMemDescs));
        CHK_RET(ReceiveRemoteMemDesc(remoteMemDescs, actualNumOfRemote));
    }
    HCCL_INFO("[HcclOneSidedConn][ExchangeMemDesc]get actualNumOfRemote[%u]", actualNumOfRemote);
    // 校验remoteDescs中的remoteRankId和conn对象中保存的localRankId是否一样
    for (u32 i = 0; i < actualNumOfRemote; i++) {
        CHK_PTR_NULL((remoteMemDescs.array) + i);
        u32 tempRankId = remoteMemDescs.array[i].remoteRankId;
        HCCL_DEBUG("[TransportMem][ExchangeMemDesc]tempRankId:%u, userRank:%u", tempRankId, localRankId_);
        if (tempRankId == INVALID_REMOTE_RANK_ID) {
            HCCL_INFO("[DoExchangeMemDesc] It's unnecessary to check remoteID.");
            continue;
        }
        if (tempRankId != localRankId_) {
            HCCL_ERROR(
                "[TransportMem][ExchangeMemDesc]localRank[%u] receive remoteMemDesc from wrong localRank[%u], "
                "connection is for localRank[%u]",
                localRankId_, tempRankId, localRankId_);
            return HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TransportMem::SendLocalMemDesc(const RmaMemDescs& localMemDescs)
{
    HcclResult ret = dataSocket_->Send(&localMemDescs.arrayLength, sizeof(u32));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "errNo[0x%016llx] localRank[%u] send localMemDesc.arrayLength to remote "
            "failed, ret[%u]",
            HCCL_ERROR_CODE(ret), localRankId_, ret),
        ret);
    HCCL_DEBUG("send localMemDescs.arrayLength:%u", localMemDescs.arrayLength);

    if (localMemDescs.arrayLength == 0) {
        HCCL_INFO("localMemDescs.arrayLength[%u], no need to send data", localMemDescs.arrayLength);
    } else {
        HCCL_DEBUG("send descSize:%u", localMemDescs.arrayLength * sizeof(RmaMemDesc));
        ret = dataSocket_->Send(localMemDescs.array, localMemDescs.arrayLength * sizeof(RmaMemDesc));
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "errNo[0x%016llx] localRank[%u] send localMemDesc to remote "
                "failed, ret[%u]",
                HCCL_ERROR_CODE(ret), localRankId_, ret),
            ret);
    }
    return HCCL_SUCCESS;
}

HcclResult TransportMem::ReceiveRemoteMemDesc(RmaMemDescs& remoteMemDescs, u32& actualNumOfRemote)
{
    HcclResult ret = dataSocket_->Recv(&actualNumOfRemote, sizeof(u32));
    remoteMemDescs.arrayLength = actualNumOfRemote;
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "errNo[0x%016llx] localRank[%u] receive actualNumOfRemote to remote "
            "failed, ret[%u]",
            HCCL_ERROR_CODE(ret), localRankId_, ret),
        ret);
    HCCL_DEBUG("receive actualNumOfRemote:%u", actualNumOfRemote);
    if (actualNumOfRemote == 0) {
        HCCL_INFO("actualNumOfRemote[%u], no need to receive data", actualNumOfRemote);
    } else {
        HCCL_DEBUG("receive descSize:%u", actualNumOfRemote * sizeof(RmaMemDesc));
        ret = dataSocket_->Recv(remoteMemDescs.array, actualNumOfRemote * sizeof(RmaMemDesc));
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "errNo[0x%016llx] localRank[%u] receive remoteMemDesc from remote "
                "failed, ret[%u]",
                HCCL_ERROR_CODE(ret), localRankId_, ret),
            ret);
    }
    return HCCL_SUCCESS;
}
} // namespace hccl

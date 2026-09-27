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
#include "dev_aicpu_ts_roce_channel.h"
#include <securec.h>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <vector>
#include "channel_param.h"
#include "adapter_hal_pub.h"
#include "adapter_rts_common.h"
#include "dispatcher_ctx.h"
#include "hccl_dispatcher_ctx.h"
#include "log.h"
#include "transport_pub.h"

using namespace hccl;

u64 DevAicpuTsRoceChannel::commSeq_{0};

DevAicpuTsRoceChannel::~DevAicpuTsRoceChannel()
{
    for (auto& pair : slots_) {
        if (pair.second.link != nullptr) {
            (void)pair.second.link->DeInit();
            pair.second.link.reset();
        }
        if (pair.second.ctx != nullptr) {
            (void)DestroyDispatcherCtx(pair.second.ctx, pair.second.commId);
        }
    }
    slots_.clear();
}

namespace {

constexpr u32 RDMA_QP_MAX_NUM = 32U;
constexpr u32 HCCL_MULTI_QP_THRESHOLD_DEFAULT = 0U;

constexpr u32 DEFAULT_TIMEOUT_MS = 10000;

HcclResult FillIbverbsDataFromRes(const HcommRoceChannelRes* res, TransportDeviceIbverbsData& ibd)
{
    std::vector<RoceMemDetails> localMd;
    auto localBase = static_cast<const RoceMemDetails*>(res->localMem);
    if (localBase != nullptr) {
        for (u32 i = 0; i < res->localMemCount; ++i) {
            localMd.push_back(localBase[i]);
        }
    }
    std::vector<RoceMemDetails> remoteMd;
    auto remoteBase = static_cast<const RoceMemDetails*>(res->remoteMem);
    if (remoteBase != nullptr) {
        for (u32 i = 0; i < res->remoteMemCount; ++i) {
            remoteMd.push_back(remoteBase[i]);
        }
    }
    HCCL_INFO(
        "[DevAicpuTsRoceChannel][Create] Roce mem from channel res: localMemCount[%u] remoteMemCount[%u] "
        "parsed local[%zu] remote[%zu]",
        res->localMemCount, res->remoteMemCount, localMd.size(), remoteMd.size());

    const u32 qpInfoSize = res->qpsPerConnection + static_cast<u32>(res->qpsPerConnection != 1U);
    if (qpInfoSize < 1U || qpInfoSize > RDMA_QP_MAX_NUM) {
        HCCL_ERROR("[DevAicpuTsRoceChannel][Create] bad qp layout qpsPerConn[%u]", res->qpsPerConnection);
        return HCCL_E_PARA;
    }
    std::vector<HcclQpInfoV2> qpVec(qpInfoSize);
    for (u32 i = 0; i < qpInfoSize; ++i) {
        qpVec[i] = res->QpInfo[i];
    }

    ibd.qpInfo = std::move(qpVec);
    ibd.qpsPerConnection = res->qpsPerConnection;
    ibd.multiQpThreshold = HCCL_MULTI_QP_THRESHOLD_DEFAULT;
    ibd.localRoceMemDetailsList = std::move(localMd);
    ibd.remoteRoceMemDetailsList = std::move(remoteMd);
    ibd.useMemDetailsMgr = true;
    ibd.remoteNotifyValueAddr = ReinterpretAs<uint64_t>(res->remoteNotifyAddr);
    ibd.remoteNotifyValueKey = res->remoteNotifyKey;
    ibd.localDataNotifyAddr = ReinterpretAs<uint64_t>(res->localDataNotifyAddr);
    ibd.localDataNotifyKey = res->localDataNotifyKey;
    ibd.notifySize = res->notifySize;
    HCCL_DEBUG(
        "[%s]remoteNotifyAddr[%llu], remoteNotifyKey[%u], localDataNotifyAddr[%llu], localDataNotifyKey[%u],"
        "notifySize[%u]",
        __func__, ibd.remoteNotifyValueAddr, ibd.remoteNotifyValueKey, ibd.localDataNotifyAddr, ibd.localDataNotifyKey,
        ibd.notifySize);

    EXCEPTION_CATCH((ibd.dataNotify = std::make_shared<LocalNotify>()), return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(ibd.dataNotify);
    CHK_RET(ibd.dataNotify->Init(res->localDataSignal, NotifyLoadType::DEVICE_NOTIFY));
    return HCCL_SUCCESS;
}

HcclResult OpenDispatcherForTsRoce(
    const HcommDeviceInfo& deviceInfo, const std::string& commId, [[maybe_unused]] size_t commIdLen, u32& outDevId,
    DispatcherCtxPtr& outDctx, HcclDispatcher& outDispatcher)
{
    outDevId = INVALID_UINT;
    CHK_RET(hrtDrvGetLocalDevIDByHostDevID(deviceInfo.devicePhyId, &outDevId));
    CHK_PRT_RET(
        outDevId == INVALID_UINT,
        HCCL_ERROR("[DevAicpuTsRoceChannel][Create] invalid devId for logicId[%d]", deviceInfo.deviceLogicId),
        HCCL_E_PARA);

    DispatcherCtxPtr dctxPtr = nullptr;
    CHK_RET(CreateDispatcherCtx(&dctxPtr, outDevId, commId.c_str()));
    CHK_PTR_NULL(dctxPtr);
    auto* dctx = static_cast<DispatcherCtx*>(dctxPtr);
    const HcclDispatcher dispatcher = dctx->GetDispatcher();
    if (dispatcher == nullptr) {
        (void)DestroyDispatcherCtx(dctxPtr, commId.c_str());
        HCCL_ERROR("[DevAicpuTsRoceChannel][Create] null dispatcher");
        return HCCL_E_PTR;
    }
    outDctx = dctxPtr;
    outDispatcher = dispatcher;
    return HCCL_SUCCESS;
}

HcclResult CreateAndInitTsRoceTransport(
    const HcommDeviceInfo& deviceInfo, DispatcherCtxPtr dctxPtr, const char* commId, HcclDispatcher dispatcher,
    TransportDeviceIbverbsData&& ibd, std::shared_ptr<Transport>& outLink)
{
    MachinePara machinePara{};
    machinePara.deviceLogicId = deviceInfo.deviceLogicId;
    machinePara.localDeviceId = deviceInfo.devicePhyId;
    DevType devType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(devType));
    machinePara.deviceType = devType;
    machinePara.isAicpuModeEn = true;
    machinePara.isIndOp = true;
    machinePara.notifyNum = 0;
    machinePara.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    machinePara.tag = "hcomm_aicpu_ts_roce" + std::string(commId);
    machinePara.userMemEnable = false;
    machinePara.drainEnable = true;
    machinePara.dctxPtr = dctxPtr;

    TransportPara transportPara{};
    transportPara.timeout = std::chrono::milliseconds(DEFAULT_TIMEOUT_MS);
    transportPara.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    static const std::unique_ptr<NotifyPool> kEmptyNotifyPool;
    std::shared_ptr<Transport> link;
    link.reset(new (std::nothrow) Transport(
        TransportType::TRANS_TYPE_DEVICE_IBVERBS, transportPara, dispatcher, kEmptyNotifyPool, machinePara,
        TransportDeviceP2pData(), ibd));
    if (link == nullptr) {
        (void)DestroyDispatcherCtx(dctxPtr, commId);
        HCCL_ERROR("[DevAicpuTsRoceChannel][Create] Transport alloc failed");
        return HCCL_E_PTR;
    }
    HcclResult tr = link->Init();
    if (tr != HCCL_SUCCESS) {
        link.reset();
        (void)DestroyDispatcherCtx(dctxPtr, commId);
        return tr;
    }
    tr = link->InitDrainNotifyInfo();
    if (tr != HCCL_SUCCESS) {
        link.reset();
        (void)DestroyDispatcherCtx(dctxPtr, commId);
        return tr;
    }
    outLink = std::move(link);
    return HCCL_SUCCESS;
}

} // namespace

HcclResult DevAicpuTsRoceChannel::Create(
    const void* blob, u64 blobBytes, const HcommDeviceInfo& deviceInfo, ChannelHandle& outHandle)
{
    CHK_PTR_NULL(blob);
    if (blobBytes < sizeof(HcommRoceChannelRes)) {
        HCCL_ERROR("[DevAicpuTsRoceChannel][Create] blob too small[%llu]", static_cast<unsigned long long>(blobBytes));
        return HCCL_E_PARA;
    }
    const auto* res = static_cast<const HcommRoceChannelRes*>(blob);

    TransportDeviceIbverbsData ibd{};
    CHK_RET(FillIbverbsDataFromRes(res, ibd));
    const u32 qpInfoSize = res->qpsPerConnection + static_cast<u32>(res->qpsPerConnection != 1U);

    char commId[sizeof(RoceSlot::commId)];
    u32 devId = INVALID_UINT;
    DispatcherCtxPtr dctxPtr = nullptr;
    HcclDispatcher dispatcher = nullptr;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++commSeq_;
        int nc = snprintf_s(
            commId, sizeof(commId), sizeof(commId) - 1U, "hcomm_ts_roce_%d_%llu", deviceInfo.deviceLogicId,
            static_cast<unsigned long long>(commSeq_));
        CHK_PRT_RET(nc < 0, HCCL_ERROR("[DevAicpuTsRoceChannel][Create] snprintf_s failed"), HCCL_E_INTERNAL);
    }

    CHK_RET(OpenDispatcherForTsRoce(deviceInfo, commId, sizeof(commId), devId, dctxPtr, dispatcher));

    std::shared_ptr<Transport> link;
    CHK_RET(CreateAndInitTsRoceTransport(deviceInfo, dctxPtr, commId, dispatcher, std::move(ibd), link));

    outHandle = ReinterpretAs<ChannelHandle>(link.get());
    RoceSlot slot;
    slot.ctx = dctxPtr;
    slot.link = std::move(link);
    CHK_SAFETY_FUNC_RET(memcpy_s(slot.commId, sizeof(slot.commId), commId, sizeof(commId)));

    {
        std::lock_guard<std::mutex> lock(mutex_);
        slots_.emplace(outHandle, std::move(slot));
    }

    HCCL_INFO(
        "[DevAicpuTsRoceChannel][Create] success logicId[%d] phyId[%u] devId[%u] blobBytes[%llu] "
        "localMem[%u] remoteMem[%u] qpsPerConn[%u] qpNum[%u] chipId[%lld] commId[%s] handle[0x%llx]",
        deviceInfo.deviceLogicId, deviceInfo.devicePhyId, devId, static_cast<unsigned long long>(blobBytes),
        res->localMemCount, res->remoteMemCount, res->qpsPerConnection, qpInfoSize, static_cast<long long>(res->chipId),
        commId, static_cast<unsigned long long>(ReinterpretAs<uintptr_t>(outHandle)));
    return HCCL_SUCCESS;
}

bool DevAicpuTsRoceChannel::Destroy(ChannelHandle handle)
{
    RoceSlot slot;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = slots_.find(handle);
        if (it == slots_.end()) {
            return false;
        }
        slot = std::move(it->second);
        slots_.erase(it);
    }
    if (slot.link != nullptr) {
        (void)slot.link->DeInit();
        slot.link.reset();
    }
    if (slot.ctx != nullptr) {
        (void)DestroyDispatcherCtx(slot.ctx, slot.commId);
    }
    HCCL_DEBUG("[DevAicpuTsRoceChannel][Destroy] destroyed handle[0x%llx]", handle);
    return true;
}

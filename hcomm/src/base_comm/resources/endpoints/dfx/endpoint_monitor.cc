/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dfx/endpoint_monitor.h"
#include "urma_endpoint.h"
#include "hcom_common.h"

#include <array>
#include <memory>

namespace hcomm {

constexpr u32 EndpointMonitor::MONITOR_INTERVAL;

namespace {
    uint32_t MonitorSlot(s32 deviceLogicId)
    {
        if ((deviceLogicId < 0) || (static_cast<u32>(deviceLogicId) > MAX_MODULE_DEVICE_NUM)) {
            HCCL_ERROR("[EndpointMonitor] deviceLogicId[%d] not in range [0,%u]", deviceLogicId, MAX_MODULE_DEVICE_NUM);
            return MAX_MODULE_DEVICE_NUM;
        }
        return static_cast<u32>(deviceLogicId);
    }

    std::array<std::shared_ptr<EndpointMonitor>, MAX_MODULE_DEVICE_NUM + 1> MakeMonitorHolders()
    {
        std::array<std::shared_ptr<EndpointMonitor>, MAX_MODULE_DEVICE_NUM + 1> holders;
        for (u32 i = 0; i <= MAX_MODULE_DEVICE_NUM; ++i) {
            holders[i] = std::make_shared<EndpointMonitor>();
        }
        return holders;
    }
} // namespace

EndpointMonitor::~EndpointMonitor() { DeInit(deviceLogicId_); }

std::shared_ptr<EndpointMonitor> EndpointMonitor::GetHolder(s32 deviceId)
{
    static auto holders = MakeMonitorHolders();
    return holders[MonitorSlot(deviceId)];
}

HcclResult EndpointMonitor::RegisterToEndpointMonitor(s32 deviceLogicId, EndpointHandle epHandle)
{
    if ((deviceLogicId < 0) || (static_cast<u32>(deviceLogicId) >= MAX_MODULE_DEVICE_NUM)) {
        HCCL_ERROR(
            "[EndpointMonitor][%s] deviceLogicId[%d] not in range [0,%u)", __func__, deviceLogicId,
            MAX_MODULE_DEVICE_NUM);
        return HCCL_E_PARA;
    }
    CHK_PRT_RET(epHandle == nullptr, HCCL_ERROR("[EndpointMonitor][%s] epHandle is null", __func__), HCCL_E_PTR);

    HCCL_INFO(
        "[EndpointMonitor] deviceLogicId[%d] epHandle[%p] RegisterToEndpointMonitor begin.", deviceLogicId, epHandle);
    u32 devPhyId{0};
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(deviceLogicId), devPhyId));

    {
        std::lock_guard<std::mutex> lock(threadLock_);
        epHandleSet_.emplace(reinterpret_cast<u64>(epHandle));
        if (!initialized_) {
            deviceLogicId_ = deviceLogicId;
            devPhyId_ = devPhyId;
            CHK_RET(RunMonitorThread());
        }
    }

    HCCL_INFO(
        "[EndpointMonitor] deviceLogicId[%d] epHandle[%p] RegisterToEndpointMonitor Completed.", deviceLogicId,
        epHandle);
    return HCCL_SUCCESS;
}

HcclResult EndpointMonitor::RunMonitorThread()
{
    HCCL_INFO("[EndpointMonitor][%s] deviceLogicId[%d] Start Thread.", __func__, deviceLogicId_);
    endpointMonitorThreadFlag_ = true;
    EXCEPTION_CATCH(
        endpointMonitorThread_ = std::make_unique<std::thread>(&EndpointMonitor::MonitorThread, this),
        return HCCL_E_INTERNAL);
    CHK_SMART_PTR_NULL(endpointMonitorThread_);
    initialized_ = true;
    return HCCL_SUCCESS;
}

void EndpointMonitor::MonitorThread()
{
    SetThreadName("Hccl_Ub_Event_Monitor");

    HcclResult ret = hrtSetDevice(deviceLogicId_);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[EndpointMonitor][%s] hrtSetDevice failed, deviceLogicId[%d], ret[%d]", __func__, deviceLogicId_, ret);
        return;
    }

    while (endpointMonitorThreadFlag_) {
        ProcessUbAsyncEvents();
        std::this_thread::sleep_for(std::chrono::milliseconds(MONITOR_INTERVAL));
    }

    ret = hrtResetDevice(deviceLogicId_);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[EndpointMonitor][%s] hrtResetDevice failed, deviceLogicId[%d], ret[%d]", __func__, deviceLogicId_, ret);
    }
}

HcclResult EndpointMonitor::UnRegisterToEndpointMonitor()
{
    s32 deviceLogicId = deviceLogicId_;
    HCCL_INFO("[EndpointMonitor] deviceId[%d] UnRegisterToEndpointMonitor begin.", deviceLogicId);
    {
        std::lock_guard<std::mutex> lock(threadLock_);
        CHK_PRT_RET(
            !initialized_,
            HCCL_WARNING(
                "[EndpointMonitor] deviceId[%d] hcclUbEventMonitor has been destroyed, or not initialized",
                deviceLogicId),
            HCCL_SUCCESS);
        epHandleSet_.clear();
    }

    CHK_RET(DeInit(deviceLogicId_));
    deviceLogicId_ = 0;
    devPhyId_ = 0;
    initialized_ = false;

    HCCL_INFO("[EndpointMonitor] deviceId[%d] UnRegisterToEndpointMonitor completed.", deviceLogicId);
    return HCCL_SUCCESS;
}

void EndpointMonitor::RemoveEpHandleFromEndpointMonitor(EndpointHandle epHandle)
{
    if (epHandle == nullptr) {
        HCCL_ERROR("[EndpointMonitor][%s] epHandle is null", __func__);
        return;
    }

    {
        std::lock_guard<std::mutex> lock(threadLock_);
        auto it = epHandleSet_.find(reinterpret_cast<u64>(epHandle));
        if (it != epHandleSet_.end()) {
            epHandleSet_.erase(it);
            HCCL_INFO(
                "[EndpointMonitor][%s] epHandle[%p] is remove from deviceId[%d]", __func__, epHandle, deviceLogicId_);
        }
    }
}

HcclResult EndpointMonitor::DeInit(s32 deviceLogicId)
{
    endpointMonitorThreadFlag_ = false;
    if (endpointMonitorThread_) {
        if (endpointMonitorThread_->joinable()) {
            try {
                HCCL_INFO("[EndpointMonitor][%s] deviceId[%d] thread join", __func__, deviceLogicId);
                endpointMonitorThread_->join();
                endpointMonitorThread_.reset();
            } catch (const std::exception& e) {
                HCCL_ERROR("[EndpointMonitor][%s] deviceId[%d] join failed: %s", __func__, deviceLogicId, e.what());
                return HCCL_E_INTERNAL;
            }
        }
    }
    return HCCL_SUCCESS;
}

void EndpointMonitor::ProcessUbAsyncEvents()
{
    std::lock_guard<std::mutex> lock(threadLock_);
    for (auto it = epHandleSet_.begin(); it != epHandleSet_.end();) {
        u32 num = ASYNC_EVENT_MAX_NUM;
        Endpoint* localEpPtr = reinterpret_cast<Endpoint*>(*it);
        // GetAsyncEvents 已下移为 UrmaEndpoint 自有方法（基类不感知）：monitor 仅注册 DEVICE 侧 UB endpoint，
        // 按具体类型 downcast 后访问；非 Urma 类转失败属防御分支，跳过且不摘除句柄
        auto* urmaEp = dynamic_cast<UrmaEndpoint*>(localEpPtr);
        if (urmaEp == nullptr) {
            HCCL_WARNING(
                "[EndpointMonitor][%s] deviceId[%d] endpoint[%p] is not UrmaEndpoint, skip async events", __func__,
                deviceLogicId_, localEpPtr);
            ++it;
            continue;
        }
        HcclResult ret = urmaEp->GetAsyncEvents(devPhyId_, events_, num);
        if (ret != HCCL_SUCCESS) {
            it = epHandleSet_.erase(it);
            HCCL_ERROR(
                "[EndpointMonitor][%s] deviceId[%d] HcommGetAsyncEvents failed ret[%d], "
                "epHandle[%p] removed from monitor",
                __func__, deviceLogicId_, ret, localEpPtr);
            continue;
        }

        for (u32 i = 0; i < num; ++i) {
            PrintUbAsyncEventsContext(static_cast<void*>(localEpPtr), i, events_[i]);
        }

        ++it;
    }
}

constexpr u32 SECOND_LAST_OFFSET = 2; // 倒数第二个字节偏移
constexpr u32 LAST_OFFSET = 3;        // 倒数第一个字节偏移
void EndpointMonitor::PrintUbAsyncEventsContext(const void* epHandle, u32 seq, const struct AsyncEvent& event) const
{
    u32 contextLen = event.len;
    if (contextLen > CONTEXT_MAX_LEN) {
        HCCL_ERROR(
            "[EndpointMonitor][%s] deviceId[%d] epHandle[%p] seq[%u] context len[%u] exceed max[%u]", __func__,
            deviceLogicId_, epHandle, seq, contextLen, CONTEXT_MAX_LEN);
        return;
    }

    HCCL_ERROR("************************************** ub async event **************************************");
    HCCL_ERROR(
        "deviceId[%d] epHandle[%p] seq[%u] resId[%u] eventType[%u] contextLen[%u]", deviceLogicId_, epHandle, seq,
        event.resId, event.eventType, event.len);
    if (contextLen != 0) {
        HCCL_ERROR("bytes order: high -> low");
    }
    constexpr u32 bytesPerLine = 4;
    for (u32 i = 0; i < contextLen; i += bytesPerLine) {
        u32 endIndex = std::min(i + bytesPerLine, contextLen);
        HCCL_ERROR(
            "context[byte %3u]: %02x%02x%02x%02x", endIndex,
            (i + LAST_OFFSET < contextLen) ? event.context[i + LAST_OFFSET] : 0,
            (i + SECOND_LAST_OFFSET < contextLen) ? event.context[i + SECOND_LAST_OFFSET] : 0,
            (i + 1 < contextLen) ? event.context[i + 1] : 0, event.context[i]);
    }
    HCCL_ERROR("********************************************************************************************");
}

} // namespace hcomm

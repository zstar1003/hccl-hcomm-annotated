/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "notify_aicpu_mgr.h"
#include "notify_manager.h"
#include "log.h"
#include <sstream>
#include <iomanip>
#include <algorithm>

constexpr u32 NOTIFY_SIZE_EIGHT = 8;

NotifyAicpuMgr::NotifyAicpuMgr() { notifys_.reserve(hccl::HCCL_THREAD_NOTIFY_MAX_NUM); }

void NotifyAicpuMgr::ReserveNotifyCapacity(size_t n) { notifys_.reserve(n); }

HcclResult NotifyAicpuMgr::NotifyFree(NotifyMgrAicpuParam* param)
{
    CHK_PTR_NULL(param);
    u32 notifyNum = param->notifyNum;
    NotifyHandle* notifyArray = static_cast<NotifyHandle*>(param->deviceHandle);
    std::string hcomId(param->hcomId);
    CHK_PTR_NULL(notifyArray);
    for (size_t i = 0; i < notifyNum; ++i) {
        hccl::LocalNotify* notify = reinterpret_cast<hccl::LocalNotify*>(notifyArray[i]);
        HCCL_INFO(
            "[NotifyAicpuMgr][%s] notifyArray[%zu]=[%llu]", __func__, i,
            static_cast<unsigned long long>(notifyArray[i]));
        auto it
            = std::find_if(notifys_.begin(), notifys_.end(), [notify](const std::unique_ptr<hccl::LocalNotify>& ptr) {
                  return ptr.get() == notify;
              });
        if (it != notifys_.end()) {
            HCCL_INFO(
                "[NotifyAicpuMgr][%s] comm identifier[%s], free notify[%llu] success", __func__, hcomId.c_str(),
                static_cast<unsigned long long>(notifyArray[i]));
            notifys_.erase(it);
        } else {
            HCCL_RUN_WARNING("[NotifyAicpuMgr][%s] localNotify[%zu] not found in notifys_", __func__, i);
        }
    }

    HCCL_INFO(
        "[NotifyAicpuMgr][%s] comm identifier[%s], free notify num[%u] success", __func__, hcomId.c_str(), notifyNum);
    return HCCL_SUCCESS;
}

HcclResult NotifyAicpuMgr::NotifyAlloc(NotifyMgrAicpuParam* param)
{
    CHK_PTR_NULL(param);
    u32 notifyNum = param->notifyNum;
    std::string notifysStr = std::string(param->notifyParam, NOTIFY_UNIQUE_ID_MAX_SIZE);
    std::string hcomId(param->hcomId);
    size_t notifySize = notifys_.size();
    HCCL_INFO(
        "[NotifyAicpuMgr][%s] comm identifier[%s], alloc notify num[%u] begin, before notifySize[%zu]", __func__,
        hcomId.c_str(), notifyNum, notifySize);
    if (UNLIKELY(HcclCheckLogLevel(HCCL_LOG_INFO))) {
        std::ostringstream oss;
        oss << "notifyParam" << " raw bytes: ";
        constexpr u32 HEX_WIDTH = 2;
        for (u32 i = 0; i < NOTIFY_UNIQUE_ID_MAX_SIZE; ++i) {
            oss << std::hex << std::setw(HEX_WIDTH) << std::setfill('0')
                << static_cast<unsigned int>(static_cast<unsigned char>(param->notifyParam[i])) << " ";
        }
        HCCL_INFO("[NotifyAicpuMgr][%s] %s", __func__, oss.str().c_str());
    }
    HcclResult ret = hccl::NotifyManager::ParseBinNotifys(notifysStr, notifys_);
    if (ret != HCCL_SUCCESS || notifys_.size() < notifyNum + notifySize) {
        HCCL_ERROR(
            "[NotifyAicpuMgr][%s] alloc notify failed, comm identifier[%s], ret[%d], "
            "alloc notify num[%u], expect num[%u], actual[%u]",
            __func__, hcomId.c_str(), static_cast<int>(ret), notifyNum, notifyNum + notifySize, notifys_.size());
        return HCCL_E_INTERNAL;
    }
    HCCL_INFO(
        "[NotifyAicpuMgr][%s] comm identifier[%s], alloc notify num[%u] end, after notifySize[%zu]", __func__,
        hcomId.c_str(), notifyNum, notifys_.size());
    NotifyHandle* notifyArray = static_cast<NotifyHandle*>(param->deviceHandle);
    CHK_PTR_NULL(notifyArray);
    for (size_t i = 0; i < notifyNum; ++i) {
        notifyArray[i] = reinterpret_cast<NotifyHandle>(notifys_[i + notifySize].get());
        HCCL_INFO(
            "[NotifyAicpuMgr][%s] notifyArray[%zu] = [%llu]", __func__, i + notifySize,
            static_cast<unsigned long long>(notifyArray[i]));
    }

    HCCL_INFO(
        "[NotifyAicpuMgr][%s] comm identifier[%s], alloc notify num[%u] success", __func__, hcomId.c_str(), notifyNum);
    return HCCL_SUCCESS;
}

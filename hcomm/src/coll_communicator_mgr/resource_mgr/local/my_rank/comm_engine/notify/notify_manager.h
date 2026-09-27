/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NOTIFY_MANAGER_H
#define NOTIFY_MANAGER_H

#include "local_notify.h"
#include "hccl_common.h"
#include "hccl/hccl_res.h"
#include "hccl_independent_common.h"
#include "manager_common.h"

namespace hccl {

class NotifyManager {
    struct NotifyInfo {
        CommEngine commEngine;
        ::NotifyType notifyType;
        bool isAicpu;
        NotifyHandle notifyHandle;
    };

public:
#ifndef CCL_KERNEL_AICPU
    NotifyManager(std::string commId, aclrtBinHandle binHandle, const ManagerCallbacks& callbacks);
    ~NotifyManager() = default;
#endif

    static HcclResult
    ParseBinNotifys(const std::string& uniqueIdStr, std::vector<std::unique_ptr<LocalNotify>>& newNotifys);
#ifndef CCL_KERNEL_AICPU
    static std::string
    GetBinNotifys(std::vector<std::unique_ptr<LocalNotify>>& newNotifys, const NotifyLoadType notifyType);

    HcclResult HcclAllocNotify(
        CommEngine commEngine, ::NotifyType notifyType, uint32_t notifyNum, NotifyHandle** notifyHandleList);
    HcclResult HcommFreeNotify(uint32_t notifyNum, NotifyHandle* notifyHandleList);
    HcclResult ResetAllocLocalNotifies();

    inline LocalNotify* GetNotify(u32 index) const
    {
        if (index > notifyNum_) {
            HCCL_ERROR(
                "[NotifyManager][GetNotify]notifyNum[%u], notifyIdx[%u] out of range, maxIndex[%u]", notifyNum_, index,
                (notifyNum_ == 0 ? 0 : notifyNum_ - 1));
            return nullptr;
        }
        return notifys_[index].get();
    }

    inline u32 GetNotifyNum() { return notifyNum_; }
#endif
private:
    static HcclResult
    InitNotifys(std::istringstream& iss, size_t notifyNum, std::vector<std::unique_ptr<LocalNotify>>& newNotifys);
#ifndef CCL_KERNEL_AICPU
    HcclResult NotifyTypeToNotifyLoadType(::NotifyType notifyType, NotifyLoadType& notifyLoadType);
    std::string commId_;
    aclrtBinHandle binHandle_;
    std::mutex notifyMutex_;
    bool isDeviceSide_ = false;
    u32 notifyNum_ = 0;
    std::vector<std::unique_ptr<LocalNotify>> notifys_;
    std::vector<std::unique_ptr<NotifyHandle[]>> handleBlocks_; // 托管所有 handle 数组
    std::unordered_map<LocalNotify*, NotifyInfo> notifysInfo_;
    ManagerCallbacks callbacks_;
#endif
};
} // namespace hccl

#endif

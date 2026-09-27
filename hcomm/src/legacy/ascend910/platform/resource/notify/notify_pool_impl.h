/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NOTIFY_POOL_IMPL_H
#define NOTIFY_POOL_IMPL_H

#include <mutex>
#include <map>
#include <array>
#include "hccl/base.h"
#include "mem_name_repository_pub.h"
#include "local_ipc_notify.h"
#include "remote_notify.h"
#include "dispatcher.h"

namespace hccl {

using NotifyPoolIPCSub = std::vector<std::shared_ptr<LocalIpcNotify>>;
using NotifyPoolNoIPCSub = std::vector<std::shared_ptr<LocalIpcNotify>>;

using NotifyPoolIndicator = struct NotifyPoolIndicatorDef {
    std::map<s64, u32> notifyPoolIPC;   // key: remote, value: index
    std::map<s64, u32> notifyPoolNoIPC; // key: remote, value: index
};

constexpr u32 NOTIFY_RES_MGR_NUM = 3; // notifyResMgr_ 长度
using NotifyResMgr = struct NotifyResMgrDef {
    std::mutex registeredOpMapMutex;
    std::map<std::string, NotifyPoolIndicator> registeredOpMap;

    std::mutex notifyPoolIPCAsignedMutex;
    std::map<s64, NotifyPoolIPCSub> notifyPoolIPCAsignedMap;
    std::map<s64, NotifyPoolIPCSub> notifyPoolDevIPCAsignedMap;

    std::mutex notifyPoolNoIPCAsignedMutex;
    std::map<s64, NotifyPoolNoIPCSub> notifyPoolNoIPCAsignedMap;
    std::map<s64, NotifyPoolNoIPCSub> notifyPoolDevNoIPCAsignedMap;
};

class NotifyPoolImpl {
public:
    explicit NotifyPoolImpl(const s32 devicePhyId);
    ~NotifyPoolImpl();
    HcclResult Init();
    HcclResult Destroy();
    HcclResult RegisterOp(const std::string& tag);
    HcclResult UnregisterOp(const std::string& tag);
    // local notify申请
    HcclResult Alloc(
        const std::string& tag, const RemoteRankInfo& info, const NotifyLoadType type,
        std::shared_ptr<LocalIpcNotify>& localNotify, u32 offsetAlignSize);

    HcclResult ResetNotify();
    HcclResult ResetNotifyForDestRank(s64 destRank);

private:
    HcclResult CreateNotify(
        std::shared_ptr<LocalIpcNotify>& localNotify, const s32 localDeviceId, const s32 remoteDeviceId,
        const NotifyLoadType type, bool withIpc = false, s64 recvId = -1, u32 offsetAlignSize = INVALID_UINT);

    HcclResult AllocIpc(
        const std::string& tag, s64 remote, s64 recvId, const s32 localDeviceId, const s32 remoteDeviceId,
        const NotifyLoadType type, std::shared_ptr<LocalIpcNotify>& localNotify, std::mutex& registeredOpMapMutex,
        std::map<std::string, NotifyPoolIndicator>& registeredOpMap, std::mutex& notifyPoolIPCAsignedMapMutex,
        std::map<s64, NotifyPoolIPCSub>& notifyPoolIPCAsignedMap, u32 offsetAlignSize);
    HcclResult Alloc(
        const std::string& tag, s64 remote, s64 recvId, const s32 localDeviceId, const s32 remoteDeviceId,
        const NotifyLoadType type, std::shared_ptr<LocalIpcNotify>& localNotify, u32 offsetAlignSize);

    HcclResult AllocNoIpc(
        const std::string& tag, s64 remote, const s32 deviceId, const NotifyLoadType type,
        std::shared_ptr<LocalIpcNotify>& localNotify, std::mutex& registeredOpMapMutex,
        std::map<std::string, NotifyPoolIndicator>& registeredOpMap, std::mutex& notifyPoolNoIPCAsignedMapMutex,
        std::map<s64, NotifyPoolNoIPCSub>& notifyPoolNoIPCAsignedMap, u32 offsetAlignSize);
    HcclResult Alloc(
        const std::string& tag, s64 remote, const s32 deviceId, const NotifyLoadType type,
        std::shared_ptr<LocalIpcNotify>& localNotify, u32 offsetAlignSize);
    HcclResult
    IsNotifyOffsetAligned(std::shared_ptr<LocalIpcNotify>& localNotify, u32 offsetAlignSize, bool& isAligned);

    HcclResult DestroyRegisteredOpMap(u32 index);
    HcclResult DestroyNotifyPoolIPCAsignedMap(u32 index);
    HcclResult DestroyNotifyPoolNoIPCAsignedMap(u32 index);
    HcclResult DestroyNotifyPoolDevIPCAsignedMap(u32 index);
    HcclResult DestroyNotifyPoolDevNoIPCAsignedMap(u32 index);
    HcclResult RegisterOpMap(const std::string& tag, std::map<std::string, NotifyPoolIndicator>& registeredOpMap);
    HcclResult UnregisterOpMap(const std::string& tag, std::map<std::string, NotifyPoolIndicator>& registeredOpMap);
    HcclResult DestroyNotify(std::shared_ptr<LocalIpcNotify>& localNotify);
    u32 GetNotifyResIdx(const std::string& tag, u32 offsetAlignSize);

    std::array<NotifyResMgr, NOTIFY_RES_MGR_NUM> notifyResMgr_;
    s32 devicePhyId_;
    s32 pid_{0};
};
} // namespace hccl
#endif /* NOTIFY_POOL_IMPL_H */

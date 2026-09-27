/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SOCKET_MGR_H
#define SOCKET_MGR_H

#include <atomic>
#include <mutex>
#include <condition_variable>
#include "../../../../../legacy/ascend950/unified_platform/resource/socket/socket.h"

namespace hcomm {

class SocketMgr {
public:
    SocketMgr() {};
    ~SocketMgr() {};
    HcclResult GetNewSocket(const Hccl::SocketConfig& socketConfig, Hccl::Socket*& socket);
    HcclResult GetSocket(const Hccl::SocketConfig& socketConfig, Hccl::Socket*& socket);
    HcclResult GetHostSocket(const Hccl::SocketConfig& socketConfig, Hccl::Socket*& socket);
    HcclResult CreateSocketWithSocketHandle(const Hccl::SocketConfig& socketConfig);
    HcclResult MakeSocketInUse(Hccl::Socket*& socket);
    HcclResult PutSocket(const Hccl::SocketConfig*& socketConfig, Hccl::Socket*& socket);
    HcclResult UpdateSocketConfig(const Hccl::SocketConfig*& socketConfig, Hccl::Socket*& socket);
    HcclResult DeleteWhiteList(const Hccl::Socket* socket);
    HcclResult DestroySocket(Hccl::Socket* socket);
    static void DeInit(u32 devPhyId);
    static SocketMgr& GetInstance(s32 phyId);

private:
    HcclResult Init();
    HcclResult GetSocketHandle(const Hccl::SocketConfig& socketConfig, Hccl::SocketHandle& socketHandle) const;
    HcclResult AddWhiteList(const Hccl::SocketConfig& socketConfig, const Hccl::SocketHandle& socketHandle);
    HcclResult CreateSocket(const Hccl::SocketConfig& socketConfig, const Hccl::SocketHandle& socketHandle);

private:
    SocketMgr(const SocketMgr&) = delete;
    SocketMgr& operator=(const SocketMgr&) = delete;

    bool isLoaded_{false};
    bool isHostOnlyInit_{false};
    uint32_t devicePhyId_{};
    uint32_t serverListenPort_{};
    std::unordered_map<Hccl::SocketConfig, std::unique_ptr<Hccl::Socket>> socketMap_{};
    std::unordered_map<Hccl::SocketHandle, std::vector<Hccl::RaSocketWhitelist>> handle2WhiteListMap_{};
    std::unordered_map<Hccl::Socket*, std::atomic<bool>> socketInUseMap_{};
    std::mutex mutex_{};
    std::condition_variable socketAvailableCv_;
};

} // namespace hcomm

#endif // SOCKET_MGR_H

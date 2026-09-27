/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SERVER_SOCKET_MANAGER_H
#define SERVER_SOCKET_MANAGER_H
#include <mutex>
#include <unordered_map>
#include "hccl/hccl_res.h"
// Orion
#include "socket/socket.h"
#include "virtual_topo.h"
#include "socket_config.h"
#include "socket_manager.h"
#include "orion_adapter_rts.h"
#include "socket_handle_manager.h"
#include "host_socket_handle_manager.h"

namespace hcomm {

class ServerSocketManager {
public:
    static ServerSocketManager& GetInstance()
    {
        static ServerSocketManager instance;
        return instance;
    }
    ServerSocketManager(const ServerSocketManager&) = delete;
    ServerSocketManager& operator=(const ServerSocketManager&) = delete;
    ServerSocketManager(ServerSocketManager&&) = delete;
    ServerSocketManager& operator=(ServerSocketManager&&) = delete;

    ~ServerSocketManager() {};

    HcclResult ServerSocketStartListen(
        const Hccl::PortData& localPort, const Hccl::NicType nicType, const uint32_t devPhyId, uint32_t* port);
    HcclResult
    ServerSocketStopListen(const Hccl::PortData& localPort, const Hccl::NicType nicType, const uint32_t port);
    void DeInit(u32 devPhyId);

private:
    ServerSocketManager()
    {
        // Ensure that (Host)SocketHandleManager destruct after the destruction of device/host ServerSocketMap_.
        (void)Hccl::SocketHandleManager::GetInstance();
        (void)Hccl::HostSocketHandleManager::GetInstance();
    };
    HcclResult DeviceSocketListen(const Hccl::PortData& localPort, const uint32_t devPhyId, uint32_t* port);
    HcclResult HostSocketListen(const Hccl::PortData& localPort, const uint32_t devPhyId, uint32_t* port);
    HcclResult DeviceSocketStopListen(const Hccl::PortData& localPort, const uint32_t port);
    HcclResult HostSocketStopListen(const Hccl::PortData& localPort, const uint32_t port);
    HcclResult
    ListenSocketIfNeeded(Hccl::Socket* serverSocket, bool isListen, uint32_t& actualPort, uint32_t* outPort) const;
    HcclResult DeInitCompatServerIfListening(const Hccl::PortData& localPort, const uint32_t port);
    void DeInitDeviceSockets(u32 devPhyId);
    void DeInitHostSockets(u32 devPhyId);

    // PortData : {ListenPort : {Socket, Count}}
    std::unordered_map<Hccl::PortData, std::unordered_map<uint32_t, std::pair<std::unique_ptr<Hccl::Socket>, uint32_t>>>
        deviceServerSocketMap_{};
    std::unordered_map<Hccl::PortData, std::unordered_map<uint32_t, std::pair<std::unique_ptr<Hccl::Socket>, uint32_t>>>
        hostServerSocketMap_{};
    std::unique_ptr<Hccl::SocketManager> socketMgrCompat_;
    std::mutex hostMutex_{};
    std::mutex deviceMutex_{};
};

} // namespace hcomm

#endif // SERVER_SOCKET_MGR_H

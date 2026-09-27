/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SOCKET_PROCESS_H
#define SOCKET_PROCESS_H

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <atomic>
#include <unordered_map>
#include "hccl_comm_socket_c_adpt.h"
#include "socket_mgr.h"

namespace hcomm {
class SocketProcess {
public:
    static SocketProcess& GetInstance(s32 deviceLogicId);
    ~SocketProcess();

    HcclResult DestroySocketHandle(SocketHandle socketHandle);

    HcclResult GetSocket(SocketDesc* socketDesc, SocketHandle& socketHandle);

    HcclResult PutSocket(SocketHandle& socketHandle);

    // 异步推动建链
    HcclResult GetStatus(SocketHandle socketHandle, SocketStates& socketStatus);

    // 发送数据接口
    HcclResult SendNoBlock(SocketHandle socketHandle, void* sendbuffer, u64 sendSize, u64*& sentSize);

    // 接收数据接口
    HcclResult RecvNoBlock(SocketHandle socketHandle, void* recvBuffer, u64 recvSize, u64*& recvedSize);

private:
    HcclResult Init();

    Hccl::SocketRole ConvertToHcclSocketRole(HcommSocketRole& hcommRole) const;

    // 创建socket句柄，建立通信通道
    HcclResult BuildSocket(SocketDesc* socketDesc, const std::string& socketTag);

    std::atomic<bool> isInit_{false};
    std::map<std::pair<Hccl::PortData, u32>, std::unique_ptr<Hccl::Socket>> serverSocketMap_{};
    std::unordered_map<std::string, std::pair<Hccl::Socket*, u32>> tag2socketMap_{};
    std::unordered_map<Hccl::Socket*, std::string> socket2TagMap_{};
    std::mutex mutex_;
    const Hccl::SocketConfig* socketConfig_{nullptr};
    u32 devicePhyId_{0};
};

} // namespace hcomm
#endif // SOCKET_PROCESS_H

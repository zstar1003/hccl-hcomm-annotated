/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SERVER_SOCKET_MGR_H
#define SERVER_SOCKET_MGR_H

#include <mutex>
#include <memory>

#include "hccl/hccl_res.h"
#include "ip_address.h"
#include "../../../../../../legacy/ascend950/unified_platform/resource/socket/socket.h"

namespace hcomm {

class ServerSocketMgr {
public:
    static HcclResult ListenStart(const uint32_t devPhyId, const CommAddr& commAddr, const Hccl::NicType nicType);
    static void DeInit(u32 devPhyId);

private:
    static ServerSocketMgr& GetInstance(const uint32_t devicePhyId);
    HcclResult ListenStart_(const CommAddr& commAddr, const Hccl::NicType nicType);

private:
    ServerSocketMgr() = default;
    ~ServerSocketMgr() = default;
    ServerSocketMgr(const ServerSocketMgr& that) = delete;
    ServerSocketMgr& operator=(const ServerSocketMgr& that) = delete;

    uint32_t devPhyId_{0};
    std::mutex innerMutex_{};
    std::unordered_map<Hccl::IpAddress, std::unique_ptr<Hccl::Socket>> deviceServerSocketMap_{};
    std::unordered_map<Hccl::IpAddress, std::unique_ptr<Hccl::Socket>> hostServerSocketMap_{};
};

} // namespace hcomm

#endif // SERVER_SOCKET_MGR_H

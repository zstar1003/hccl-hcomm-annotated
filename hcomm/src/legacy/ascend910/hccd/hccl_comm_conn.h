/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_COMM_CONN_H
#define HCCL_COMM_CONN_H

#include <memory>
#include <queue>
#include "comm.h"
#include "adapter_hccp.h"
#include "network/hccp_common.h"
#include "transport_heterog_event_roce_pub.h"
#include "memory_alloc_ring.h"
#include "heterog_mem_blocks_manager_pub.h"
#include "dlra_function.h"
#include "mr_manager.h"
#include "hccl_ip_address.h"
#include "hccl_types.h"
#include "network_manager_pub.h"

namespace hccl {
constexpr u32 MAX_CONCURRENCY_LINK_NUM = 16;
constexpr u32 MAX_CONN_LINK_NUM = 512;

class HcclCommConn {
public:
    HcclCommConn();

    ~HcclCommConn();

    HcclResult Connect(HcclAddr& connectAddr);

    HcclResult Bind(HcclAddr& bindAddr);

    HcclResult Listen(int backLog);

    HcclResult Accept(HcclAddr& acceptAddr, HcclCommConn*& acceptConn);

    HcclResult Isend(const void* buf, int count, HcclDataType dataType, HcclRequest& request);

    HcclResult Improbe(int& flag, HcclMessage& msg, HcclStatus& status);

    HcclResult Imrecv(void* buf, int count, HcclDataType datatype, HcclMessage msg, HcclRequest& request);

    HcclResult
    ImrecvScatter(void* buf[], int count[], int bufCount, HcclDataType datatype, HcclMessage msg, HcclRequest& request);

    HcclResult Test(HcclRequest requestHandle, s32& flag, HcclStatus& compState);

    HcclResult InitTransport(u32 role, HcclAddr& localAddr, SocketInfoT& tmpInfo);

    HcclResult ResetCurrentErrorConnection(HcclCommConn*& newCommConn);

    void SetForceClose();

private:
    enum class OpStatus { START, CONNECT, GETSOCKET, BUILDTRANSPORT, END };

    struct AcceptCommConn {
        HcclCommConn* newCommConn{nullptr}; // comm句柄
        SocketInfoT socketInfo{};
    };

    static constexpr u32 INIT_LOCAL_IP = 0;
    static constexpr u32 INIT_REMOTE_IP = 1;
    static constexpr u32 MEMORY_CAPACITY = 256 * 1024;
    static constexpr u32 DEFAULT_LOCAL_RANK = 0;
    static constexpr u32 DEFAULT_REMOTE_RANK = 1;
    static constexpr s32 DEFAULT_TAG = 0;
    static constexpr u32 RESOURCE_MEMORY_CAPACITY = 2048;
    static constexpr u32 DELAY_TIME = 10;
    static constexpr u32 MEM_BLOCK_CAPACITY = 8191;
    static constexpr u32 ACCEPT_MAX_TIME = 10000;

    HcclResult SetAddr(HcclAddr& bindAddr, u32 opType);

    HcclResult InitMsgAndRequestBuffer();

    HcclResult InitMemBlocksAndRecvWrMem();

    HcclResult CheckDataType(const HcclDataType dataType);

    HcclResult PrepareSocketInfoForServer(struct SocketInfoT& socketInfo);

    HcclResult GetSocket(struct SocketInfoT& socketInfo);

    HcclResult PrepareConnectSocketInfoForClient(HcclAddr& bindAddr);

    HcclResult MrManagerInit();

    HcclResult StopListen();

    const HcclAddr& GetRemoteAddr() const;

    HcclResult SocketForceClose(SocketInfoT& socketInfo);
    void SetStartTime();
    void GetStartTime(std::chrono::time_point<std::chrono::steady_clock>& startTime);

    u32 devId_{0};
    u32 role_{SERVER_ROLE_SOCKET};
    std::unique_ptr<TransportHeterog> transport_{};
    HcclAddr remoteAddr_{};
    HcclAddr localAddr_{};
    SocketHandle socketHandle_{nullptr};
    RdmaHandle rdmaHandle_{nullptr};
    SocketConnectInfoT connectInfo_{};
    SocketInfoT socketInfo_{};
    std::unique_ptr<HeterogMemBlocksManager> memBlocksManager_;
    bool isListen_{false};
    OpStatus connectState_{OpStatus::START};

    std::mutex recvWrInfosMutex_{};
    std::mutex msgInfosMutex_{};
    std::mutex reqInfosMutex_{};
    std::unique_ptr<LocklessRingMemoryAllocate<RecvWrInfo>> recvWrInfosMem_{};
    std::unique_ptr<LocklessRingMemoryAllocate<HcclMessageInfo>> msgInfosMem_{};
    std::unique_ptr<LocklessRingMemoryAllocate<HcclRequestInfo>> reqInfosMem_{};

    TransportResourceInfo transportResourceInfo_{
        nullptr, msgInfosMem_, reqInfosMem_, memBlocksManager_, recvWrInfosMem_};

    std::chrono::time_point<std::chrono::steady_clock> startTime_;
    std::mutex bindMutex_{}; // 在bind的时候加锁，防止同一个comm并发bind
    std::mutex connHandleQueueMutex_{};
    std::queue<AcceptCommConn> connHandleQueue_;
};
} // namespace hccl

#endif

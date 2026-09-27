/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_V2_STEST_FAKE_SOCKET_H
#define HCCL_V2_STEST_FAKE_SOCKET_H

#include <map>
#include <tuple>
#include <mutex>
#include <atomic>
#include <vector>
#include "hccp.h"
#include "hccp_ctx.h"
class fake_socket {
public:
    bool Connect(struct SocketConnectInfoT& conn);
    bool Get(int role, struct SocketInfoT& conn);
    bool Send(int* fdHanlde, const void* data, unsigned long long size);
    bool Recv(int* fdHandle, void* data, unsigned long long size);
    int* GetSocketHandle();
    virtual ~fake_socket();

private:
    // (rank1, rank2) 对应fdHandle1, （rank2, rank1）对应fdHandle2, 它们的sendbuffer和recvbuffer刚好相反

    // 存储 fdHandle 使用的 sendbuffer 和 recvbuffer
    std::map<int, std::pair<std::vector<char>*, std::vector<char>*>> fdBuffer;

    // 存储 （localRank, dstRank）对应的 fdHandle
    std::map<std::pair<int, int>, int> ranksAndFdMap;

    std::mutex mutex;
    std::atomic_int fdHandleGenerator;
    void internalConnect(std::pair<int, int> pair);

    std::vector<int*> socketHandleStore{std::vector<int*>(16)};
};

#endif // HCCL_V2_STEST_FAKE_SOCKET_H

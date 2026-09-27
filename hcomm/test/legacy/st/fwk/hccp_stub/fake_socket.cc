/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <set>
#include "fake_socket.h"
#include "../context/st_ctx.h"

using namespace std;

bool fake_socket::Send(int* fdHanlde, const void* data, unsigned long long int size)
{
    std::lock_guard<std::mutex> lock(mutex);

    auto iter = fdBuffer.find(*fdHanlde);
    if (iter == fdBuffer.end()) {
        return false;
    }

    auto dataBegin = static_cast<const char*>(data);
    iter->second.first->insert(iter->second.first->end(), dataBegin, dataBegin + size);
    return true;
}

bool fake_socket::Recv(int* fdHandle, void* data, unsigned long long int size)
{
    std::lock_guard<std::mutex> lock(mutex);

    auto iter = fdBuffer.find(*fdHandle);

    if (iter == fdBuffer.end()) {
        return false;
    }

    if (iter->second.second->size() < size) {
        return false;
    }
    memcpy(data, iter->second.second->data(), size);
    iter->second.second->erase(iter->second.second->begin(), iter->second.second->begin() + size);
    return true;
}

void fake_socket::internalConnect(std::pair<int, int> myDirection)
{
    int fdHandle = fdHandleGenerator++;
    ranksAndFdMap.emplace(myDirection, fdHandle);

    auto oppositeDirection = make_pair(myDirection.second, myDirection.first);

    if (ranksAndFdMap.find(oppositeDirection) != ranksAndFdMap.end()) {
        auto it = ranksAndFdMap.find(oppositeDirection);
        auto buffers = fdBuffer.find(it->second);
        fdBuffer[fdHandle] = make_pair((buffers->second.second), (buffers->second.first));
    } else {
        auto sendBuf = new vector<char>();
        auto recvBuf = new vector<char>();
        fdBuffer[fdHandle] = make_pair(sendBuf, recvBuf);
    }
}

bool fake_socket::Connect(struct SocketConnectInfoT& conn)
{
    std::lock_guard<std::mutex> lock(mutex);
    auto remoteRank = conn.remoteIp.addr.s_addr;
    auto ctx = GetCurrentThreadContext();
    std::pair<int, int> myDirection = std::make_pair(ctx->myRank, remoteRank);

    if (ranksAndFdMap.find(myDirection) != ranksAndFdMap.end()) {
        return true;
    }

    internalConnect(myDirection);

    return true;
}

bool fake_socket::Get(int role, struct SocketInfoT& conn)
{
    std::lock_guard<std::mutex> lock(mutex);

    auto remoteRank = conn.remoteIp.addr.s_addr;
    auto ctx = GetCurrentThreadContext();
    std::pair<int, int> myDirection = std::make_pair(ctx->myRank, remoteRank);

    if (ranksAndFdMap.find(myDirection) == ranksAndFdMap.end()) {
        if (role == 1) {
            return false;
        } else {
            // 对于server role, 并没有调用connect。 需要在此处单独处理
            auto dst = conn.remoteIp.addr.s_addr;
            auto local = GetCurrentThreadContext()->myRank;
            internalConnect(std::make_pair(local, dst));
        }
    }

    conn.status = 1;

    auto it = ranksAndFdMap.find(myDirection);

    if (it != ranksAndFdMap.end()) {
        conn.fdHandle = &(it->second);
    }

    return true;
}

fake_socket::~fake_socket()
{
    std::set<vector<char>*> waitingDel;
    for (auto item : fdBuffer) {
        waitingDel.insert(item.second.first);
        waitingDel.insert(item.second.second);
    }

    for (auto item : waitingDel) {
        delete item;
    }

    for (auto item : socketHandleStore) {
        if (item != nullptr) {
            delete item;
        }
    }
}

int* fake_socket::GetSocketHandle()
{
    int dev = GetCurrentThreadContext()->myRank;
    if (socketHandleStore[dev] != nullptr) {
        return socketHandleStore[dev];
    }
    socketHandleStore[dev] = new int(dev);
    return socketHandleStore[dev];
}

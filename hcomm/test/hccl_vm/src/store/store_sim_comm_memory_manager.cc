/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "store_sim_comm_memory_manager.h"

#include <cstdint>
#include <cstring>
#include <algorithm>

#include "sim_log.h"

namespace sim {
constexpr uint32_t RA_SOCKET_BUF_SIZE = (64 * 1024);

struct CommMemHead {
    uint64_t bufferSize;
    uint32_t writeTotalBytes;
    uint32_t readTotalBytes;
    char data[RA_SOCKET_BUF_SIZE];
};

static CommunicationMemoryManager* s_instance = nullptr;
static std::mutex s_instanceLock;

CommunicationMemoryManager& CommunicationMemoryManager::GetInstance()
{
    if (s_instance == nullptr) {
        std::lock_guard<std::mutex> lock(s_instanceLock);
        if (s_instance == nullptr) {
            s_instance = new CommunicationMemoryManager();
        }
    }
    return *s_instance;
}

CommunicationMemoryManager::CommunicationMemoryManager() {}

CommunicationMemoryManager::~CommunicationMemoryManager() {}

void* CommunicationMemoryManager::AllocCommMem(const char* name)
{
    if (!name) {
        HCCL_VM_ERROR("acquire name is null");
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(m_mutex);

    void* ptr = MemoryManager::GetInstance().AllocMemByName(name, sizeof(CommMemHead));
    if (ptr == nullptr) {
        HCCL_VM_ERROR("alloc failed, name: {}", name);
        return nullptr;
    }

    CommMemHead* head = (CommMemHead*)ptr;
    head->bufferSize = 0;
    head->writeTotalBytes = 0;
    head->readTotalBytes = 0;
    HCCL_VM_INFO("acquire name: {}, ptr: {:p}", name, ptr);
    m_commMemMap[name] = ptr;
    return head->data;
}

void* CommunicationMemoryManager::AcquireCommMem(const char* name)
{
    if (!name) {
        HCCL_VM_ERROR("acquire name is null");
        return nullptr;
    }
    void* ptr = nullptr;
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_commMemMap.find(name);
    if (it != m_commMemMap.end()) {
        ptr = it->second;
    } else {
        ptr = MemoryManager::GetInstance().AcquireMemByName(name);
        if (ptr == nullptr) {
            HCCL_VM_ERROR("acquire failed, name: {}", name);
            return nullptr;
        }
        m_commMemMap[name] = ptr;
    }

    CommMemHead* head = (CommMemHead*)ptr;
    return head->data;
}

int CommunicationMemoryManager::ReleaseCommMem(const char* name)
{
    if (!name) {
        HCCL_VM_ERROR("release name is nullptr");
        return -1;
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_commMemMap.find(name) == m_commMemMap.end()) {
        HCCL_VM_ERROR("release comm mem not found, name: {}", name);
        return -1;
    }
    m_commMemMap.erase(name);
    HCCL_VM_INFO("release name: {}", name);
    MemoryManager::GetInstance().ReleaseMemByName(name);
    return 0;
}

int64_t CommunicationMemoryManager::WriteCommMem(const char* name, const void* dataPtr, size_t size)
{
    if (!name || !dataPtr || size == 0) {
        HCCL_VM_ERROR("write invalid params, name: {}, dataPtr: {}, size: {}", name, dataPtr, size);
        return -1;
    }

    void* ptr = nullptr;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto iter = m_commMemMap.find(name);
        if (iter == m_commMemMap.end()) {
            HCCL_VM_ERROR("write comm mem not found, name: {}", name);
            return -1;
        }
        ptr = iter->second;
    }
    MemoryManager::GetInstance().LockMemByName(name);

    CommMemHead* head = (CommMemHead*)ptr;

    int64_t realSend = RA_SOCKET_BUF_SIZE - head->bufferSize;
    realSend = std::min((int64_t)size, realSend);

    memcpy(head->data + head->bufferSize, dataPtr, realSend);
    head->bufferSize += realSend;
    head->writeTotalBytes += realSend;
    uint64_t totalWrite = head->writeTotalBytes;

    MemoryManager::GetInstance().UnlockMemByName(name);
    HCCL_VM_INFO("write name: {}, size:{} real:{}, totalWriteBytes: {}", name, size, realSend, totalWrite);
    return realSend;
}

int CommunicationMemoryManager::ReadCommMem(const char* name, void* dataPtr, size_t size)
{
    if (!name || !dataPtr || size == 0) {
        HCCL_VM_ERROR("read invalid params, name: {}, dataPtr: {}, size: {}", name, dataPtr, size);
        return -1;
    }

    void* ptr = nullptr;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto iter = m_commMemMap.find(name);
        if (iter == m_commMemMap.end()) {
            HCCL_VM_ERROR("read comm mem not found, name: {}", name);
            return -1;
        }
        ptr = iter->second;
    }
    MemoryManager::GetInstance().LockMemByName(name);

    CommMemHead* head = (CommMemHead*)ptr;
    if (head->bufferSize == 0) {
        MemoryManager::GetInstance().UnlockMemByName(name);
        return 0;
    }

    size_t recvSize = (size < head->bufferSize) ? size : head->bufferSize;
    memcpy(dataPtr, head->data, recvSize);

    head->bufferSize -= recvSize;
    head->readTotalBytes += recvSize;
    uint64_t totalRead = head->readTotalBytes;

    if (head->bufferSize > 0) {
        memmove(head->data, head->data + recvSize, head->bufferSize);
    }

    MemoryManager::GetInstance().UnlockMemByName(name);

    HCCL_VM_INFO("read name: {}, size: {}, totalReadBytes: {}", name, recvSize, totalRead);
    return recvSize;
}

} // namespace sim

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "store_sim_memory_manager.h"

#include "sim_log.h"

namespace sim {
static MemoryManager* s_instance = nullptr;
static std::mutex s_instanceLock;

MemoryManager& MemoryManager::GetInstance()
{
    if (s_instance == nullptr) {
        std::lock_guard<std::mutex> lock(s_instanceLock);
        if (s_instance == nullptr) {
            s_instance = new MemoryManager();
        }
    }
    return *s_instance;
}

MemoryManager::MemoryManager() {}

MemoryManager::~MemoryManager()
{
    std::lock_guard<std::mutex> lock(m_memLock);
    // 遍历所有内存，关闭共享内存
    for (auto& pair : m_memMap) {
        ShmClose(pair.second.ptr);
    }
    // 清空映射表
    m_memMap.clear();
}

void* MemoryManager::AllocMemByName(const char* name, size_t size)
{
    if (!name || size == 0) {
        return NULL;
    }

    std::lock_guard<std::mutex> lock(m_memLock);
    std::string nameStr(name);

    // 检查内存是否已经存在
    auto it = m_memMap.find(nameStr);
    if (it != m_memMap.end()) {
        return NULL;
    }

    // 创建新的共享内存
    void* ptr = ShmCreate(name, size);
    if (ptr == NULL) {
        return NULL;
    }

    // 添加到内存映射表
    MemInfo memInfo;
    memInfo.ptr = ptr;
    memInfo.size = size;
    memInfo.type = 0; // 0: 共享内存
    memInfo.refCount = 1;
    HCCL_VM_INFO("alloc name: {}, size: {:d}, ptr: {:p}", name, size, ptr);
    m_memMap[nameStr] = memInfo;
    return ptr;
}

void MemoryManager::FreeMemByName(const char* name)
{
    if (!name) {
        return;
    }

    std::lock_guard<std::mutex> lock(m_memLock);
    std::string nameStr(name);

    auto it = m_memMap.find(nameStr);
    if (it != m_memMap.end()) {
        // 关闭共享内存
        HCCL_VM_INFO("free name: {}, size: {:d}, ptr: {:p}", name, it->second.size, it->second.ptr);
        ShmClose(it->second.ptr);
        // 从映射表中移除
        m_memMap.erase(it);
    }
}

void MemoryManager::LockMemByName(const char* name)
{
    if (!name) {
        return;
    }

    std::lock_guard<std::mutex> lock(m_memLock);
    std::string nameStr(name);

    auto it = m_memMap.find(nameStr);
    if (it != m_memMap.end()) {
        ShmLock(it->second.ptr);
    }
}

void MemoryManager::UnlockMemByName(const char* name)
{
    if (!name) {
        return;
    }

    std::lock_guard<std::mutex> lock(m_memLock);
    std::string nameStr(name);

    auto it = m_memMap.find(nameStr);
    if (it != m_memMap.end()) {
        ShmUnlock(it->second.ptr);
    }
}

void* MemoryManager::AcquireMemByName(const char* name)
{
    if (!name) {
        return NULL;
    }

    std::lock_guard<std::mutex> lock(m_memLock);
    std::string nameStr(name);

    // 检查内存是否已经存在
    auto it = m_memMap.find(nameStr);
    if (it != m_memMap.end()) {
        // 内存已存在，增加引用计数
        __sync_fetch_and_add(&it->second.refCount, 1);
        return it->second.ptr;
    }

    // 内存不存在，尝试打开
    size_t size = 0;
    void* ptr = ShmOpen(name, &size);
    if (ptr == NULL) {
        return NULL;
    }

    // 添加到内存映射表
    MemInfo memInfo;
    memInfo.ptr = ptr;
    memInfo.size = size; // 使用ShmOpen返回的大小
    memInfo.type = 0;    // 0: 共享内存
    memInfo.refCount = 1;
    HCCL_VM_INFO("acquire name: {}, size: {:d}, ptr: {:p}", name, size, ptr);
    m_memMap[nameStr] = memInfo;
    return ptr;
}

void MemoryManager::ReleaseMemByName(const char* name)
{
    if (!name) {
        return;
    }

    std::lock_guard<std::mutex> lock(m_memLock);
    std::string nameStr(name);

    auto it = m_memMap.find(nameStr);
    if (it != m_memMap.end()) {
        // 减少引用计数
        int refCount = __sync_fetch_and_sub(&it->second.refCount, 1);
        if (refCount == 1) {
            // 引用计数归0，关闭共享内存并从映射表中移除
            HCCL_VM_INFO("release name: {}, ptr: {:p}", name, it->second.ptr);
            ShmClose(it->second.ptr);
            m_memMap.erase(it);
        }
    }
}
} // namespace sim

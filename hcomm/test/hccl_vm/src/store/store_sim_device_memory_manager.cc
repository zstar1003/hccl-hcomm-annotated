/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "store_sim_device_memory_manager.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <thread>
#include <unistd.h>

#include "sim_log.h"
#include "store_sim_memory_manager.h"
#include "store_sim_comm_pool_policy.h"
#include "sim_models.h"
#include "store_sim_run_mode.h"

namespace sim {
// 静态成员变量初始化
DeviceMemoryManager* DeviceMemoryManager::s_instance = nullptr;
std::mutex DeviceMemoryManager::s_instanceLock;

// 获取单例实例
DeviceMemoryManager& DeviceMemoryManager::GetInstance()
{
    if (s_instance == nullptr) {
        std::lock_guard<std::mutex> lock(s_instanceLock);
        if (s_instance == nullptr) {
            s_instance = new DeviceMemoryManager();
        }
    }
    return *s_instance;
}

DeviceMemoryManager::DeviceMemoryManager() {}

DeviceMemoryManager::~DeviceMemoryManager() {}

void* DeviceMemoryManager::AllocVirMem(uint64_t devPhyId, size_t size)
{
    // 向上8字节对齐
    size_t alignedSize = (size + 7) & ~7;
    uint64_t virAddr = 0;
    std::lock_guard<std::mutex> lock(m_virSizeLock);
    // 分配虚拟地址 - 每个设备使用独立的地址空间
    uint64_t deviceBaseAddr = static_cast<uint64_t>(devPhyId) << 48;

    // 从当前使用量+1开始分配虚拟地址
    virAddr = deviceBaseAddr + (m_deviceMemSizes[devPhyId] + 1);

    // 更新设备内存大小
    m_deviceMemSizes[devPhyId] += alignedSize;
    HCCL_VM_INFO("dev:{:d} alloc vir mem:{:p}", devPhyId, reinterpret_cast<void*>(virAddr));
    return reinterpret_cast<void*>(virAddr);
}

// 释放虚拟内存
void DeviceMemoryManager::FreeVirMem(uint64_t devPhyId, void* virAddr)
{
    HCCL_VM_INFO("dev:{:d} free vir mem:{:p}", devPhyId, virAddr);
}

// 分配物理内存：大块（仅校验模式开）引流到复用区，否则按名独立分配
void* DeviceMemoryManager::AllocPhyMem(const char* name, uint64_t deviceId, size_t size)
{
    if (name == nullptr) {
        return nullptr;
    }
    HCCL_VM_INFO("dev:{:d} alloc phy mem:{}, size:{}", deviceId, name, size);
    bool checkOnlyMode = IsCheckOnlyMode();
    // 超过复用区上界直接报错，不回退真实分配（避免巨块吃满 /dev/shm）。
    if (CommPoolPolicy::ExceedsCeiling(size, checkOnlyMode)) {
        HCCL_VM_ERROR(
            "dev:{:d} alloc phy mem:{} size:{} exceeds pool ceiling:{}, reject", deviceId, name, size,
            CommPoolPolicy::kPoolSize);
        return nullptr;
    }
    if (CommPoolPolicy::ShouldRedirect(size, checkOnlyMode)) {
        return MemoryManager::GetInstance().AcquireMemByName(CommPoolPolicy::kPoolName);
    }
    return MemoryManager::GetInstance().AllocMemByName(name, size);
}

// 释放物理内存（非复用区）：是否在复用区由调用方按 size 判断后分流，这里只做真正的释放。
void DeviceMemoryManager::FreePhyMem(const char* name, uint64_t deviceId)
{
    if (name == nullptr) {
        return;
    }
    HCCL_VM_INFO("dev:{:d} free phy mem:{}", deviceId, name);
    MemoryManager::GetInstance().FreeMemByName(name);
}

// 获取物理内存：大块（仅校验模式开）引流到复用区，否则按名独立分配
void* DeviceMemoryManager::AcquirePhyMem(const char* name, uint64_t deviceId, size_t size)
{
    if (name == nullptr) {
        return nullptr;
    }
    HCCL_VM_INFO("dev:{:d} acquire phy mem:{}, size:{}", deviceId, name, size);
    bool checkOnlyMode = IsCheckOnlyMode();
    // 超过复用区上界直接报错，不回退真实分配。
    if (CommPoolPolicy::ExceedsCeiling(size, checkOnlyMode)) {
        HCCL_VM_ERROR(
            "dev:{:d} acquire phy mem:{} size:{} exceeds pool ceiling:{}, reject", deviceId, name, size,
            CommPoolPolicy::kPoolSize);
        return nullptr;
    }
    if (CommPoolPolicy::ShouldRedirect(size, checkOnlyMode)) {
        return MemoryManager::GetInstance().AcquireMemByName(CommPoolPolicy::kPoolName);
    }
    return MemoryManager::GetInstance().AcquireMemByName(name);
}

// 释放物理内存（非复用区）：是否在复用区由调用方按 size 判断后分流，这里只做真正的释放。
int DeviceMemoryManager::ReleasePhyMem(const char* name, uint64_t deviceId)
{
    if (name == nullptr) {
        return 0;
    }
    HCCL_VM_INFO("dev:{:d} release phy mem:{}", deviceId, name);
    MemoryManager::GetInstance().ReleaseMemByName(name);
    return 0;
}

// 缓存设备地址到host地址
void DeviceMemoryManager::MapDevPtrHostPtr(void* devPtr, void* hostPtr)
{
    std::lock_guard<std::mutex> lock(m_devPtrHostPtrLock);
    m_devPtrHostPtrMap[devPtr] = hostPtr;
}

// 去缓存设备地址到host地址
void DeviceMemoryManager::UnmapDevPtrHostPtr(void* devPtr)
{
    std::lock_guard<std::mutex> lock(m_devPtrHostPtrLock);
    m_devPtrHostPtrMap.erase(devPtr);
}

// 获取host地址
void* DeviceMemoryManager::GetHostPtrByDevPtr(void* devPtr)
{
    std::lock_guard<std::mutex> lock(m_devPtrHostPtrLock);
    auto it = m_devPtrHostPtrMap.find(devPtr);
    if (it != m_devPtrHostPtrMap.end()) {
        return it->second;
    }
    return nullptr;
}
} // namespace sim

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_DEVICE_MEMORY_MANAGER_H
#define SIM_DEVICE_MEMORY_MANAGER_H

#include <map>
#include <mutex>
#include <string>

namespace sim {
class DeviceMemoryManager {
public:
    // 获取单例实例
    static DeviceMemoryManager& GetInstance();

    // 禁止拷贝和赋值
    DeviceMemoryManager(const DeviceMemoryManager&) = delete;
    DeviceMemoryManager& operator=(const DeviceMemoryManager&) = delete;

    // 分配虚拟内存
    void* AllocVirMem(uint64_t devPhyId, size_t size);

    // 释放虚拟内存
    void FreeVirMem(uint64_t devPhyId, void* virAddr);

    // 分配物理内存
    void* AllocPhyMem(const char* name, uint64_t deviceId, size_t size);

    // 释放物理内存
    void FreePhyMem(const char* name, uint64_t deviceId);

    // 获取物理内存
    void* AcquirePhyMem(const char* name, uint64_t deviceId, size_t size);

    // 释放物理内存
    int ReleasePhyMem(const char* name, uint64_t deviceId);

    // 缓存设备地址到host地址
    void MapDevPtrHostPtr(void* devPtr, void* hostPtr);

    // 去缓存设备地址到host地址
    void UnmapDevPtrHostPtr(void* devPtr);

    // 获取host地址
    void* GetHostPtrByDevPtr(void* devPtr);

private:
    // 私有构造函数
    DeviceMemoryManager();
    ~DeviceMemoryManager();

private:
    // 静态单例实例
    static DeviceMemoryManager* s_instance;
    static std::mutex s_instanceLock;

    std::mutex m_virSizeLock;
    std::map<int, uint64_t> m_deviceMemSizes; // 每个设备的内存大小

    std::mutex m_devPtrHostPtrLock;
    std::map<void*, void*> m_devPtrHostPtrMap; // 设备地址到host地址映射表
};
} // namespace sim

#endif // SIM_DEVICE_MEMORY_MANAGER_H

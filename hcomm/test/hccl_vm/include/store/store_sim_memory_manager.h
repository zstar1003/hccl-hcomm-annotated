/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_MEMORY_MANAGER_H
#define SIM_MEMORY_MANAGER_H

#include <atomic>
#include <map>
#include <mutex>
#include <string>

#include "store_sim_shm_ops.h"

namespace sim {
struct MemInfo {
    void* ptr;
    size_t size;
    uint32_t type;              // 0: 共享内存 1：文件映射
    volatile uint32_t refCount; // 记录当前进程对共享内存的引用
};

class MemoryManager {
public:
    // 获取单例实例
    static MemoryManager& GetInstance();

    // 禁止拷贝和赋值
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;

    /*创建并获取新内存*/
    void* AllocMemByName(const char* name, size_t size);

    /*释放内存区域，会关闭共享内存*/
    void FreeMemByName(const char* name);

    /*锁定名称对应的整个内存区域*/
    void LockMemByName(const char* name);

    /*解锁名称对应的整个内存区域*/
    void UnlockMemByName(const char* name);

    /*获取已经存在的内存(或打开)，引用计数加1*/
    void* AcquireMemByName(const char* name);

    /*根据名称释放共享内存区域,与AcquireMemByName对应，MemInfo引用计数减1,引用计数归0则关闭共享内存区域*/
    void ReleaseMemByName(const char* name);

private:
    // 私有构造函数
    MemoryManager();

    // 析构函数，释放所有资源
    ~MemoryManager();

    std::mutex m_memLock;
    std::map<std::string, MemInfo> m_memMap;
};
} // namespace sim

#endif // SIM_MEMORY_MANAGER_H

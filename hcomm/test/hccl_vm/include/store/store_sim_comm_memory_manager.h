/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_COMM_MEMORY_MANAGER_H
#define SIM_COMM_MEMORY_MANAGER_H

#include <mutex>
#include <string>
#include <unordered_map>

#include "store_sim_memory_manager.h"

namespace sim {
class CommunicationMemoryManager {
public:
    static CommunicationMemoryManager& GetInstance();

    CommunicationMemoryManager(const CommunicationMemoryManager&) = delete;
    CommunicationMemoryManager& operator=(const CommunicationMemoryManager&) = delete;

    void* AllocCommMem(const char* name);
    void* AcquireCommMem(const char* name);

    int ReleaseCommMem(const char* name);

    int64_t WriteCommMem(const char* name, const void* dataPtr, size_t size);

    int ReadCommMem(const char* name, void* dataPtr, size_t size);

private:
    CommunicationMemoryManager();
    ~CommunicationMemoryManager();

private:
    std::mutex m_mutex;
    std::unordered_map<std::string, void*> m_commMemMap;
};
} // namespace sim

#endif // SIM_COMM_MEMORY_MANAGER_H

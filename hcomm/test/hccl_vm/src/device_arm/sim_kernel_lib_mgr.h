/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_KERNEL_LIB_MGR_H
#define SIM_KERNEL_LIB_MGR_H

#include <cstdint>
#include <map>
#include <string>
#include <mutex>

// 函数原型：入参void*，返回uint32_t
using KernelFn = uint32_t (*)(void*);

namespace sim {

class KernelLibManager {
public:
    static KernelLibManager& GetInstance();

    KernelFn GetOrLoadFunc(const std::string& libName, const std::string& symbolName);

    void Cleanup();

private:
    KernelLibManager();
    ~KernelLibManager();

    void* LoadKernelSo(const std::string& libName);

    KernelLibManager(const KernelLibManager&) = delete;
    KernelLibManager& operator=(const KernelLibManager&) = delete;

    void LoadBaseLibs();

    std::mutex m_mutex;
    bool m_baseLoaded{false};
    std::map<std::string, void*> m_soHandles;      // libName -> dlopen handle
    std::map<std::string, KernelFn> m_symbolCache; // "funcName" -> func ptr
};

} // namespace sim

#endif

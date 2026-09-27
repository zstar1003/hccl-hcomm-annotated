/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sim_kernel_lib_mgr.h"
#include <dlfcn.h>
#include "sim_log.h"
#include "sim_common_defs.h"
#include "sim_common_api.h"

namespace sim {

KernelLibManager::KernelLibManager() { LoadBaseLibs(); }

KernelLibManager& KernelLibManager::GetInstance()
{
    static KernelLibManager instance;
    return instance;
}

void* KernelLibManager::LoadKernelSo(const std::string& libName)
{
    auto it = m_soHandles.find(libName);
    if (it != m_soHandles.end()) {
        return it->second;
    }

    void* handle = dlopen(libName.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        HCCL_VM_ERROR("dlopen({}) failed:{}", libName, dlerror());
        return nullptr;
    }

    m_soHandles[libName] = handle;
    HCCL_VM_INFO("loaded so:{} finished", libName);
    return handle;
}

KernelFn KernelLibManager::GetOrLoadFunc(const std::string& libName, const std::string& symbolName)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    void* handle = LoadKernelSo(libName);
    if (!handle) {
        return nullptr;
    }

    std::string key = libName + "::" + symbolName;
    auto it = m_symbolCache.find(key);
    if (it != m_symbolCache.end()) {
        return it->second;
    }

    void* fn = dlsym(handle, symbolName.c_str());
    if (!fn) {
        HCCL_VM_ERROR("dlsym({}) in {} failed: {}", symbolName, libName, dlerror());
        return nullptr;
    }

    KernelFn func = reinterpret_cast<KernelFn>(fn);
    m_symbolCache[key] = func;
    HCCL_VM_INFO("dlsym({}) in {}", symbolName, libName);
    return func;
}

void KernelLibManager::Cleanup()
{
    HCCL_VM_INFO("cleanup: {} SOs, {} symbols", m_soHandles.size(), m_symbolCache.size());

    m_symbolCache.clear();

    for (auto& [name, handle] : m_soHandles) {
        if (handle) {
            dlclose(handle);
            HCCL_VM_INFO("dlclose({})", name);
        }
    }
    m_soHandles.clear();
}

void KernelLibManager::LoadBaseLibs()
{
    if (m_baseLoaded) {
        return;
    }

    std::string libName = "libslog.so";
    std::string archStr = GetArchStr();
    std::string libPath = InstallPath::ResolveToInstallRoot("lib/" + archStr + "/" + libName);

    void* handle = LoadKernelSo(libPath);
    if (handle == nullptr) {
        HCCL_VM_ERROR("Load base library {} failed, path:{}", libName, libPath);
        return;
    }

    HCCL_VM_INFO("Base library {} loaded successfully", libName);
    m_baseLoaded = true;
    return;
}

KernelLibManager::~KernelLibManager() { Cleanup(); }

} // namespace sim

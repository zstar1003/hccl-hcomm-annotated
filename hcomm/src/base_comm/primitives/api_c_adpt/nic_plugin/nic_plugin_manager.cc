/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "nic_plugin_manager.h"

#include <acl/acl_rt.h>
#include <dirent.h>
#include <dlfcn.h>
#include <climits>
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

#include "hcomm_result_defs.h"
#include "log.h"
#include "param_check_pub.h"
#include "shared_jetty_mgr.h"

namespace hcomm {
namespace {
    constexpr const char* HCOMM_NIC_PLUGIN_DIR = "hcomm_plugin";
    constexpr const char* HCOMM_NIC_PLUGIN_SO_ENV = "HCOMM_NIC_PLUGIN_SO";

    std::once_flag& LoadOnce()
    {
        static std::once_flag loadOnce;
        return loadOnce;
    }

    std::vector<std::unique_ptr<NicPluginEntry>>& LoadedPlugins()
    {
        static std::vector<std::unique_ptr<NicPluginEntry>> loadedPlugins;
        return loadedPlugins;
    }

    std::unordered_map<CommProtocol, const NicPluginEntry*>& ProtocolPlugins()
    {
        static std::unordered_map<CommProtocol, const NicPluginEntry*> protocolPlugins;
        return protocolPlugins;
    }

    bool EndsWithSo(const std::string& path)
    {
        constexpr const char* suffix = ".so";
        constexpr size_t suffixLen = 3U;
        return path.size() >= suffixLen && path.compare(path.size() - suffixLen, suffixLen, suffix) == 0;
    }

    bool IsOpsHeaderValid(const CommAbiHeader& header, uint32_t magicWord, uint32_t version, const char* opsName)
    {
        if (header.magicWord != magicWord) {
            HCCL_RUN_WARNING(
                "[NicPlugin] %s magicWord[0x%08x] mismatch, expected[0x%08x].", opsName, header.magicWord, magicWord);
            return false;
        }
        if (header.version != version) {
            HCCL_RUN_WARNING("[NicPlugin] %s version[%u] mismatch, expected[%u].", opsName, header.version, version);
            return false;
        }
        if (header.size < sizeof(CommAbiHeader)) {
            HCCL_RUN_WARNING(
                "[NicPlugin] %s size[%u] is smaller than ABI header[%zu].", opsName, header.size,
                sizeof(CommAbiHeader));
            return false;
        }
        return true;
    }

    void RegisterPluginProtocols(const NicPluginEntry* plugin)
    {
        auto& protocolPlugins = ProtocolPlugins();
        for (uint32_t idx = 0; idx < plugin->info->protocolCount; ++idx) {
            const CommProtocol protocol = plugin->info->protocols[idx];
            auto iter = protocolPlugins.find(protocol);
            if (iter != protocolPlugins.end()) {
                HCCL_RUN_WARNING(
                    "[NicPlugin] protocol[%d] handler[%s] is overwritten by plugin[%s].", protocol,
                    iter->second->info->name == nullptr ? "unknown" : iter->second->info->name,
                    plugin->info->name == nullptr ? "unknown" : plugin->info->name);
            }
            protocolPlugins[protocol] = plugin;
            HCCL_RUN_INFO(
                "[NicPlugin] protocol[%d] is handled by plugin[%s].", protocol,
                plugin->info->name == nullptr ? "unknown" : plugin->info->name);
        }
    }

    void* LoadSymbol(void* soHandle, const char* soPath, const char* symbol)
    {
        dlerror();
        void* addr = dlsym(soHandle, symbol);
        const char* dlsymErr = dlerror();
        if (dlsymErr != nullptr || addr == nullptr) {
            HCCL_RUN_WARNING(
                "[NicPlugin] dlsym %s from %s failed: %s.", symbol, soPath, dlsymErr == nullptr ? "unknown" : dlsymErr);
            return nullptr;
        }
        return addr;
    }

    void LoadOnePlugin(const std::string& path)
    {
        if (path.empty()) {
            return;
        }
        // 规范化文件路径后再使用，防范路径遍历风险
        char canonicalPath[PATH_MAX] = {0};
        if (realpath(path.c_str(), canonicalPath) == nullptr) {
            HCCL_RUN_WARNING("[LoadOnePlugin] %s is not a valid real path.", path.c_str());
            return;
        }
        void* soHandle = dlopen(canonicalPath, RTLD_NOW | RTLD_LOCAL);
        if (soHandle == nullptr) {
            HCCL_RUN_WARNING("[NicPlugin] dlopen %s failed: %s.", path.c_str(), dlerror());
            return;
        }

        auto getInfo
            = reinterpret_cast<HcommNicPluginGetInfoFunc>(LoadSymbol(soHandle, path.c_str(), "HcommNicPluginGetInfo"));
        auto createEndpoint = reinterpret_cast<HcommNicPluginCreateEndpointFunc>(
            LoadSymbol(soHandle, path.c_str(), "HcommNicPluginCreateEndpoint"));
        auto createChannel = reinterpret_cast<HcommNicPluginCreateChannelFunc>(
            LoadSymbol(soHandle, path.c_str(), "HcommNicPluginCreateChannel"));
        if (getInfo == nullptr || createEndpoint == nullptr || createChannel == nullptr) {
            dlclose(soHandle);
            return;
        }

        const HcommNicPluginInfo* info = getInfo();
        if (!ValidatePluginInfo(path.c_str(), info, createEndpoint, createChannel)) {
            dlclose(soHandle);
            return;
        }

        std::unique_ptr<NicPluginEntry> plugin;
        try {
            plugin = std::make_unique<NicPluginEntry>(NicPluginEntry{soHandle, info, createEndpoint, createChannel});
        } catch (const std::bad_alloc&) {
            HCCL_RUN_WARNING("[NicPlugin] allocate plugin entry for %s failed.", path.c_str());
            dlclose(soHandle);
            return;
        }
        RegisterPluginProtocols(plugin.get());
        LoadedPlugins().emplace_back(std::move(plugin));
    }

    void LoadDefaultDirectory(const std::string& pluginDir)
    {
        DIR* dir = opendir(pluginDir.c_str());
        if (dir == nullptr) {
            HCCL_RUN_INFO("[NicPlugin] plugin directory %s is unavailable.", pluginDir.c_str());
            return;
        }
        std::vector<std::string> soPaths;
        for (dirent* entry = readdir(dir); entry != nullptr; entry = readdir(dir)) {
            const std::string name(entry->d_name);
            if (name == "." || name == ".." || !EndsWithSo(name)) {
                continue;
            }
            soPaths.emplace_back(pluginDir + "/" + name);
        }
        closedir(dir);
        std::sort(soPaths.begin(), soPaths.end());
        for (const auto& path : soPaths) {
            LoadOnePlugin(path);
        }
    }

    void LoadExplicitPlugins(const char* envValue)
    {
        if (envValue == nullptr || envValue[0] == '\0') {
            return;
        }
        const std::string paths(envValue);
        size_t start = 0;
        while (start <= paths.size()) {
            const size_t end = paths.find(':', start);
            const std::string path = paths.substr(start, end == std::string::npos ? std::string::npos : end - start);
            LoadOnePlugin(path);
            if (end == std::string::npos) {
                break;
            }
            start = end + 1;
        }
    }

    void LoadPluginsOnce()
    {
        const char* ascendHomePath = getenv("ASCEND_HOME_PATH");
        if (ascendHomePath != nullptr && ascendHomePath[0] != '\0') {
            LoadDefaultDirectory(std::string(ascendHomePath) + "/" + HCOMM_NIC_PLUGIN_DIR);
        } else {
            HCCL_RUN_INFO("[NicPlugin] ASCEND_HOME_PATH is empty, skip default plugin directory.");
            LoadExplicitPlugins(getenv(HCOMM_NIC_PLUGIN_SO_ENV));
        }
    }

} // namespace

bool ValidateEndpointOps(const HcommNicEndpointOps* ops)
{
    if (ops == nullptr
        || !IsOpsHeaderValid(
            ops->header, HCOMM_NIC_ENDPOINT_OPS_MAGIC_WORD, HCOMM_NIC_ENDPOINT_OPS_VERSION, "endpoint ops")) {
        return false;
    }
    if (!IsPluginOpAvailable(ops, offsetof(HcommNicEndpointOps, destroy), sizeof(decltype(ops->destroy)))
        || ops->destroy == nullptr) {
        HCCL_ERROR("[NicPlugin] plugin endpoint destroy is not implemented.");
        return false;
    }
    return true;
}

bool ValidateChannelOps(const HcommNicChannelOps* ops)
{
    if (ops == nullptr
        || !IsOpsHeaderValid(
            ops->header, HCOMM_NIC_CHANNEL_OPS_MAGIC_WORD, HCOMM_NIC_CHANNEL_OPS_VERSION, "channel ops")) {
        return false;
    }
    if (!IsPluginOpAvailable(ops, offsetof(HcommNicChannelOps, destroy), sizeof(decltype(ops->destroy)))
        || ops->destroy == nullptr) {
        HCCL_ERROR("[NicPlugin] channel destroy is not implemented.");
        return false;
    }
    return true;
}

bool ValidatePluginInfo(
    const char* soPath, const HcommNicPluginInfo* info, HcommNicPluginCreateEndpointFunc createEndpoint,
    HcommNicPluginCreateChannelFunc createChannel)
{
    if (info == nullptr) {
        HCCL_RUN_WARNING("[NicPlugin] %s exports null plugin info.", soPath);
        return false;
    }
    if (!IsOpsHeaderValid(
            info->header, HCOMM_NIC_PLUGIN_INFO_MAGIC_WORD, HCOMM_NIC_PLUGIN_INFO_VERSION, "plugin info")) {
        return false;
    }
    constexpr size_t requiredSize
        = offsetof(HcommNicPluginInfo, protocols) + sizeof(static_cast<HcommNicPluginInfo*>(nullptr)->protocols);
    if (info->header.size < requiredSize) {
        HCCL_RUN_WARNING(
            "[NicPlugin] %s plugin info size[%u] is smaller than required[%zu].", soPath, info->header.size,
            requiredSize);
        return false;
    }
    if (info->protocolCount == 0 || info->protocolCount > HCOMM_NIC_PLUGIN_MAX_PROTOCOLS) {
        HCCL_RUN_WARNING("[NicPlugin] %s invalid protocolCount[%u].", soPath, info->protocolCount);
        return false;
    }
    if (createEndpoint == nullptr || createChannel == nullptr) {
        HCCL_RUN_WARNING("[NicPlugin] %s missing create endpoint/channel symbol.", soPath);
        return false;
    }
    for (uint32_t idx = 0; idx < info->protocolCount; ++idx) {
        const CommProtocol protocol = info->protocols[idx];
        if ((protocol < COMM_PROTOCOL_HCCS || protocol > COMM_PROTOCOL_UBG) && protocol < COMM_PROTOCOL_CUSTOM_BASE) {
            HCCL_RUN_WARNING("[NicPlugin] %s invalid protocol[%d].", soPath, info->protocols[idx]);
            return false;
        }
    }
    return true;
}

void LoadAllNicPlugins() { std::call_once(LoadOnce(), LoadPluginsOnce); }

const NicPluginEntry* FindHostNicPlugin(CommProtocol protocol)
{
    LoadAllNicPlugins();
    const auto& protocolPlugins = ProtocolPlugins();
    auto iter = protocolPlugins.find(protocol);
    const NicPluginEntry* entry = iter == protocolPlugins.end() ? nullptr : iter->second;
    return entry;
}

int32_t DefaultEndpointInit(void* ctx)
{
    (void)ctx;
    HCCL_RUN_WARNING("[NicPlugin] plugin endpoint init is not supported.");
    return HCCL_SUCCESS;
}

int32_t DefaultEndpointRegisterMemory(void* ctx, const CommMem* mem, const char* tag, void** handle)
{
    (void)ctx;
    (void)mem;
    (void)tag;
    (void)handle;
    HCCL_RUN_WARNING("[NicPlugin] plugin endpoint registerMemory is not supported.");
    return HCCL_E_NOT_SUPPORT;
}

int32_t DefaultEndpointUnregisterMemory(void* ctx, void* handle)
{
    (void)ctx;
    (void)handle;
    HCCL_RUN_WARNING("[NicPlugin] plugin endpoint unregisterMemory is not supported.");
    return HCCL_E_NOT_SUPPORT;
}

int32_t DefaultEndpointMemoryExport(void* ctx, void* handle, void** desc, uint32_t* descLen)
{
    (void)ctx;
    (void)handle;
    (void)desc;
    (void)descLen;
    HCCL_RUN_WARNING("[NicPlugin] plugin endpoint memoryExport is not supported.");
    return HCCL_E_NOT_SUPPORT;
}

int32_t DefaultEndpointMemoryImport(void* ctx, const void* desc, uint32_t descLen, CommMem* outMem)
{
    (void)ctx;
    (void)desc;
    (void)descLen;
    (void)outMem;
    HCCL_RUN_WARNING("[NicPlugin] plugin endpoint memoryImport is not supported.");
    return HCCL_E_NOT_SUPPORT;
}

int32_t DefaultEndpointMemoryUnimport(void* ctx, const void* desc, uint32_t descLen)
{
    (void)ctx;
    (void)desc;
    (void)descLen;
    HCCL_RUN_WARNING("[NicPlugin] plugin endpoint memoryUnimport is not supported.");
    return HCCL_E_NOT_SUPPORT;
}

int32_t DefaultEndpointGetListenPort(void* ctx, uint32_t* port)
{
    (void)ctx;
    (void)port;
    HCCL_RUN_WARNING("[NicPlugin] plugin endpoint getListenPort is not supported.");
    return HCCL_E_NOT_SUPPORT;
}

HcommResult FillDefaultEndpointOps(const HcommNicEndpointOps* src, HcommNicEndpointOps** outOps)
{
    if (src == nullptr || outOps == nullptr) {
        return HCCL_E_PARA;
    }
    HcommNicEndpointOps* dst = new (std::nothrow) HcommNicEndpointOps();
    if (dst == nullptr) {
        return HCCL_E_MEMORY;
    }

    size_t copySize = (src->header.size < sizeof(HcommNicEndpointOps)) ? src->header.size : sizeof(HcommNicEndpointOps);
    (void)memcpy_s(dst, sizeof(HcommNicEndpointOps), src, copySize);

    FOR_EACH_ENDPOINT_OP_DEFAULT(FILL_ENDPOINT_OP_DEFAULT)
    *outOps = dst;
    return HCCL_SUCCESS;
}

// ---- Channel ops 默认实现 ----

int32_t DefaultChannelInit(void* ctx)
{
    (void)ctx;
    HCCL_RUN_WARNING("[NicPlugin] channel init is not supported.");
    return HCCL_SUCCESS;
}

int32_t DefaultChannelGetStatus(void* ctx, int32_t* status)
{
    (void)ctx;
    (void)status;
    HCCL_RUN_WARNING("[NicPlugin] channel getStatus is not supported.");
    return HCCL_E_NOT_SUPPORT;
}

int32_t DefaultChannelWriteNbi(void* ctx, void* dst, const void* src, uint64_t len)
{
    (void)ctx;
    (void)dst;
    (void)src;
    (void)len;
    HCCL_RUN_WARNING("[NicPlugin] channel writeNbi is not supported.");
    return HCCL_E_NOT_SUPPORT;
}

int32_t DefaultChannelWriteNbiOnThread(void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t len)
{
    (void)ctx;
    (void)thread;
    (void)dst;
    (void)src;
    (void)len;
    HCCL_RUN_WARNING("[NicPlugin] channel writeNbiOnThread is not supported.");
    return HCCL_E_NOT_SUPPORT;
}

int32_t DefaultChannelWriteOnThread(void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t len)
{
    (void)ctx;
    (void)thread;
    (void)dst;
    (void)src;
    (void)len;
    HCCL_RUN_WARNING("[NicPlugin] channel writeOnThread is not supported.");
    return HCCL_E_NOT_SUPPORT;
}

int32_t DefaultChannelWriteWithNotifyNbi(void* ctx, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx)
{
    (void)ctx;
    (void)dst;
    (void)src;
    (void)len;
    (void)remoteNotifyIdx;
    HCCL_RUN_WARNING("[NicPlugin] channel writeWithNotifyNbi is not supported.");
    return HCCL_E_NOT_SUPPORT;
}

int32_t DefaultChannelWriteWithNotifyNbiOnThread(
    void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx)
{
    (void)ctx;
    (void)thread;
    (void)dst;
    (void)src;
    (void)len;
    (void)remoteNotifyIdx;
    HCCL_RUN_WARNING("[NicPlugin] channel writeWithNotifyNbiOnThread is not supported.");
    return HCCL_E_NOT_SUPPORT;
}

int32_t DefaultChannelWriteWithNotifyOnThread(
    void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx)
{
    (void)ctx;
    (void)thread;
    (void)dst;
    (void)src;
    (void)len;
    (void)remoteNotifyIdx;
    HCCL_RUN_WARNING("[NicPlugin] channel writeWithNotifyOnThread is not supported.");
    return HCCL_E_NOT_SUPPORT;
}

int32_t DefaultChannelWriteReduceOnThread(
    void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t count, HcommDataType dataType,
    HcommReduceOp reduceOp)
{
    (void)ctx;
    (void)thread;
    (void)dst;
    (void)src;
    (void)count;
    (void)dataType;
    (void)reduceOp;
    HCCL_RUN_WARNING("[NicPlugin] channel writeReduceOnThread is not supported.");
    return HCCL_E_NOT_SUPPORT;
}

int32_t DefaultChannelReadReduceOnThread(
    void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t count, HcommDataType dataType,
    HcommReduceOp reduceOp)
{
    (void)reduceOp;
    (void)dataType;
    (void)count;
    (void)src;
    (void)dst;
    (void)thread;
    (void)ctx;
    HCCL_RUN_WARNING("[NicPlugin] channel readReduceOnThread is not supported.");
    return HCCL_E_NOT_SUPPORT;
}

int32_t DefaultChannelWriteReduceWithNotifyOnThread(
    void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t count, HcommDataType dataType,
    HcommReduceOp reduceOp, uint32_t remoteNotifyIdx)
{
    (void)ctx;
    (void)thread;
    (void)dst;
    (void)src;
    (void)count;
    (void)dataType;
    (void)reduceOp;
    (void)remoteNotifyIdx;
    HCCL_RUN_WARNING("[NicPlugin] channel writeReduceWithNotifyOnThread is not supported.");
    return HCCL_E_NOT_SUPPORT;
}

int32_t DefaultChannelReadNbi(void* ctx, void* dst, const void* src, uint64_t len)
{
    (void)ctx;
    (void)dst;
    (void)src;
    (void)len;
    HCCL_RUN_WARNING("[NicPlugin] channel readNbi is not supported.");
    return HCCL_E_NOT_SUPPORT;
}

int32_t DefaultChannelReadNbiOnThread(void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t len)
{
    (void)ctx;
    (void)thread;
    (void)dst;
    (void)src;
    (void)len;
    HCCL_RUN_WARNING("[NicPlugin] channel readNbiOnThread is not supported.");
    return HCCL_E_NOT_SUPPORT;
}

int32_t DefaultChannelReadOnThread(void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t len)
{
    (void)ctx;
    (void)thread;
    (void)dst;
    (void)src;
    (void)len;
    HCCL_RUN_WARNING("[NicPlugin] channel readOnThread is not supported.");
    return HCCL_E_NOT_SUPPORT;
}

int32_t DefaultChannelNotifyRecord(void* ctx, uint32_t remoteNotifyIdx)
{
    (void)ctx;
    (void)remoteNotifyIdx;
    HCCL_RUN_WARNING("[NicPlugin] channel notifyRecord is not supported.");
    return HCCL_E_NOT_SUPPORT;
}

int32_t DefaultChannelNotifyRecordOnThread(void* ctx, ThreadHandle thread, uint32_t remoteNotifyIdx)
{
    (void)ctx;
    (void)thread;
    (void)remoteNotifyIdx;
    HCCL_RUN_WARNING("[NicPlugin] channel notifyRecordOnThread is not supported.");
    return HCCL_E_NOT_SUPPORT;
}

int32_t DefaultChannelNotifyWait(void* ctx, uint32_t localNotifyIdx, uint32_t timeOut)
{
    (void)ctx;
    (void)localNotifyIdx;
    (void)timeOut;
    HCCL_RUN_WARNING("[NicPlugin] channel notifyWait is not supported.");
    return HCCL_E_NOT_SUPPORT;
}

int32_t DefaultChannelNotifyWaitOnThread(void* ctx, ThreadHandle thread, uint32_t localNotifyIdx, uint32_t timeOut)
{
    (void)ctx;
    (void)thread;
    (void)localNotifyIdx;
    (void)timeOut;
    HCCL_RUN_WARNING("[NicPlugin] channel notifyWaitOnThread is not supported.");
    return HCCL_E_NOT_SUPPORT;
}

int32_t DefaultChannelNotifyWaitOnThreadWithDefaultTimeout(void* ctx, ThreadHandle thread, uint32_t localNotifyIdx)
{
    (void)ctx;
    (void)thread;
    (void)localNotifyIdx;
    HCCL_RUN_WARNING("[NicPlugin] channel notifyWaitOnThreadWithDefaultTimeout is not supported.");
    return HCCL_E_NOT_SUPPORT;
}

int32_t DefaultChannelBatchTransferOnThread(
    void* ctx, ThreadHandle thread, const HcommBatchTransferDesc* transferDescs, uint32_t transferDescNum)
{
    (void)ctx;
    (void)thread;
    (void)transferDescs;
    (void)transferDescNum;
    HCCL_RUN_WARNING("[NicPlugin] channel batchTransferOnThread is not supported.");
    return HCCL_E_NOT_SUPPORT;
}

int32_t DefaultChannelFence(void* ctx)
{
    (void)ctx;
    HCCL_RUN_WARNING("[NicPlugin] channel fence is not supported.");
    return HCCL_E_NOT_SUPPORT;
}

int32_t DefaultChannelFenceOnThread(void* ctx, ThreadHandle thread)
{
    (void)ctx;
    (void)thread;
    HCCL_RUN_WARNING("[NicPlugin] channel fenceOnThread is not supported.");
    return HCCL_E_NOT_SUPPORT;
}

int32_t DefaultChannelDrainOnThread(void* ctx, ThreadHandle thread)
{
    (void)ctx;
    (void)thread;
    HCCL_RUN_WARNING("[NicPlugin] channel drainOnThread is not supported.");
    return HCCL_E_NOT_SUPPORT;
}

HcommResult FillDefaultChannelOps(const HcommNicChannelOps* src, HcommNicChannelOps** outOps)
{
    if (src == nullptr || outOps == nullptr) {
        return HCCL_E_PARA;
    }
    HcommNicChannelOps* dst = new (std::nothrow) HcommNicChannelOps();
    if (dst == nullptr) {
        return HCCL_E_MEMORY;
    }

    size_t copySize = (src->header.size < sizeof(HcommNicChannelOps)) ? src->header.size : sizeof(HcommNicChannelOps);
    (void)memcpy_s(dst, sizeof(HcommNicChannelOps), src, copySize);

    FOR_EACH_CHANNEL_OP_DEFAULT(FILL_CHANNEL_OP_DEFAULT)
    *outOps = dst;
    return HCCL_SUCCESS;
}

} // namespace hcomm

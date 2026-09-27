/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_NIC_PLUGIN_HOLDER_H
#define HCOMM_NIC_PLUGIN_HOLDER_H

#include <optional>
#include "endpoint.h"
#include "server_socket_context/server_socket_context.h"
#include "channel.h"
#include "nic_plugin_manager.h"

namespace hcomm {

/**
 * @note 职责：NIC 插件 endpoint 的 RegedMemMgr，内存方法转发到 nicOps_。
 *       GetAllMemHandles 插件 ops 表无此字段，返回 NOT_SUPPORT。
 *       nicOps_/nicCtx_ 构造期注入，运行期只读。
 */
class PluginRegedMemMgr : public RegedMemMgr {
public:
    PluginRegedMemMgr(HcommNicEndpointOps* nicOps, void* nicCtx) : nicOps_(nicOps), nicCtx_(nicCtx) {}
    ~PluginRegedMemMgr() override = default;

    HcclResult RegisterMemory(const HcommMem* mem, const char* memTag, void** memHandle) override
    {
        return static_cast<HcclResult>(nicOps_->registerMemory(nicCtx_, mem, memTag, memHandle));
    }
    HcclResult UnregisterMemory(void* memHandle) override
    {
        return static_cast<HcclResult>(nicOps_->unregisterMemory(nicCtx_, memHandle));
    }
    HcclResult
    MemoryExport(const EndpointDesc& endpointDesc, void* memHandle, void** memDesc, uint32_t* memDescLen) override
    {
        (void)endpointDesc;
        return static_cast<HcclResult>(nicOps_->memoryExport(nicCtx_, memHandle, memDesc, memDescLen));
    }
    HcclResult MemoryImport(const void* memDesc, uint32_t descLen, HcommMem* outMem) override
    {
        return static_cast<HcclResult>(nicOps_->memoryImport(nicCtx_, memDesc, descLen, outMem));
    }
    HcclResult MemoryUnimport(const void* memDesc, uint32_t descLen) override
    {
        return static_cast<HcclResult>(nicOps_->memoryUnimport(nicCtx_, memDesc, descLen));
    }
    HcclResult GetAllMemHandles([[maybe_unused]] void** memHandles, [[maybe_unused]] uint32_t* memHandleNum) override
    {
        return HCCL_E_NOT_SUPPORT;
    }

private:
    HcommNicEndpointOps* nicOps_{nullptr};
    void* nicCtx_{nullptr};
};

template <typename Ops>
void DestroyNicPluginOpsAndCtx(Ops*& nicOps, void* nicCtx)
{
    if (nicOps != nullptr) {
        if (nicOps->destroy != nullptr) {
            int32_t ret = nicOps->destroy(nicCtx);
            if (ret != HCCL_SUCCESS) {
                HCCL_WARNING("[%s] plugin destroy failed, ret[%d].", __func__, ret);
            }
        }
        delete nicOps;
        nicOps = nullptr;
    }
}

/**
 * @note 职责：NIC 插件 endpoint 的 ServerSocketContext。
 *       GetListenPort 转发 nicOps_->getListenPort
 */
class PluginServerSocketContext : public ServerSocketContext {
public:
    PluginServerSocketContext(HcommNicEndpointOps* nicOps, void* nicCtx) : nicOps_(nicOps), nicCtx_(nicCtx) {}
    ~PluginServerSocketContext() override = default;

    HcclResult ServerSocketGetListenPort(uint32_t* port) override
    {
        return static_cast<HcclResult>(nicOps_->getListenPort(nicCtx_, port));
    }
    HcclResult ServerSocketListen([[maybe_unused]] uint32_t port) override
    {
        return HCCL_E_NOT_SUPPORT; // ops 表无 listen 字段
    }

private:
    HcommNicEndpointOps* nicOps_{nullptr};
    void* nicCtx_{nullptr};
};

/**
 * @note 职责：NIC插件Endpoint占位子类，nicOps_/nicCtx_ 从 Endpoint 基类移入此 private 成员。
 *         内存操作经 GetRegedMemMgr() 返回组合持有的 PluginRegedMemMgr（转发到 nicOps_）。
 */
class PluginEndpointHolder : public Endpoint {
public:
    explicit PluginEndpointHolder(const EndpointDesc& endpointDesc, const NicPluginEntry* pluginEntry)
        : Endpoint(endpointDesc),
          pluginEntry_(pluginEntry)
    {}
    ~PluginEndpointHolder() override { DestroyNicPluginOpsAndCtx(nicOps_, nicCtx_); }

    const NicPluginEntry* GetPluginEntry() const { return pluginEntry_; }

    void SetNicEndpointCtx(HcommNicEndpointOps* nicOps, void* nicCtx)
    {
        nicOps_ = nicOps;
        nicCtx_ = nicCtx;
        regedMemMgr_ = std::make_shared<PluginRegedMemMgr>(nicOps, nicCtx);
        serverSocketContext_.emplace(nicOps, nicCtx);
    }
    HcommNicEndpointOps* GetNicOps() const { return nicOps_; }
    void* GetNicCtx() const { return nicCtx_; }

    // 返回组合持有的 PluginRegedMemMgr
    RegedMemMgr* GetRegedMemMgr() override { return regedMemMgr_.get(); }

    // 返回组合持有的 PluginServerSocketContext（SetNicEndpointCtx 前返回 nullptr）
    ServerSocketContext* GetServerSocketContext() override
    {
        return serverSocketContext_.has_value() ? &serverSocketContext_.value() : nullptr;
    }

    HcclResult Init() override { return HCCL_E_NOT_SUPPORT; }
    void* GetRdmaHandle() override { return nullptr; }
    bool IsCtxHandleValid() const override { return false; }

private:
    const NicPluginEntry* pluginEntry_;
    HcommNicEndpointOps* nicOps_{nullptr};
    void* nicCtx_{nullptr};
    std::shared_ptr<PluginRegedMemMgr> regedMemMgr_{};               // 组合持有 PluginRegedMemMgr
    std::optional<PluginServerSocketContext> serverSocketContext_{}; // 组合持有 PluginServerSocketContext
};

class PluginChannelHolder : public Channel {
public:
    explicit PluginChannelHolder(const NicPluginEntry* pluginEntry) : pluginEntry_(pluginEntry) {}
    ~PluginChannelHolder() override { DestroyNicPluginOpsAndCtx(nicOps_, nicCtx_); }

    const NicPluginEntry* GetPluginEntry() const { return pluginEntry_; }

    HcclResult Init() override { return HCCL_E_NOT_SUPPORT; }
    HcclResult GetNotifyNum(uint32_t* notifyNum) const override
    {
        (void)notifyNum;
        return HCCL_E_NOT_SUPPORT;
    }
    HcclResult GetRemoteMems(uint32_t* memNum, CommMem** remoteMem, char*** memInfos) override
    {
        (void)memNum;
        (void)remoteMem;
        (void)memInfos;
        return HCCL_E_NOT_SUPPORT;
    }
    ChannelStatus GetStatus() override { return ChannelStatus::FAILED; }
    HcclResult Clean() override { return HCCL_E_NOT_SUPPORT; }
    HcclResult Resume() override { return HCCL_E_NOT_SUPPORT; }
    HcclResult NotifyRecord(const uint32_t remoteNotifyIdx) override
    {
        (void)remoteNotifyIdx;
        return HCCL_E_NOT_SUPPORT;
    }
    HcclResult NotifyWait(const uint32_t localNotifyIdx, const uint32_t timeout) override
    {
        (void)localNotifyIdx;
        (void)timeout;
        return HCCL_E_NOT_SUPPORT;
    }
    HcclResult WriteWithNotify(void* dst, const void* src, const uint64_t len, uint32_t remoteNotifyIdx) override
    {
        (void)dst;
        (void)src;
        (void)len;
        (void)remoteNotifyIdx;
        return HCCL_E_NOT_SUPPORT;
    }
    HcclResult Write(void* dst, const void* src, uint64_t len) override
    {
        (void)dst;
        (void)src;
        (void)len;
        return HCCL_E_NOT_SUPPORT;
    }
    HcclResult Read(void* dst, const void* src, uint64_t len) override
    {
        (void)dst;
        (void)src;
        (void)len;
        return HCCL_E_NOT_SUPPORT;
    }
    HcclResult ChannelFence() override { return HCCL_E_NOT_SUPPORT; }
    const HcommChannelDesc& GetChannelDesc() const override
    {
        static const HcommChannelDesc kEmptyDesc{};
        return kEmptyDesc;
    }

private:
    const NicPluginEntry* pluginEntry_;
};

} // namespace hcomm

#endif // HCOMM_NIC_PLUGIN_HOLDER_H

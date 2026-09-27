/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_EXPERIMENTAL_PLUGIN_OPS_H
#define HCOMM_EXPERIMENTAL_PLUGIN_OPS_H

#include "hcomm_nic_plugin.h"

#include <memory>
#include <new>
#include "channel.h"
#include "hccl_mem_defs.h"
#include "log.h"
#include "param_check_pub.h"

namespace hcomm_experimental {

template <typename Traits>
class NicPluginOps {
public:
    using EndpointT = typename Traits::EndpointT;
    using ChannelT = typename Traits::ChannelT;

    // ==================== Endpoint Ops ====================

    static int32_t InitEndpoint(void* ctx)
    {
        CHK_PTR_NULL(ctx);
        return static_cast<EndpointT*>(ctx)->Init();
    }

    static int32_t DestroyEndpoint(void* ctx)
    {
        CHK_PTR_NULL(ctx);
        delete static_cast<EndpointT*>(ctx);
        return HCCL_SUCCESS;
    }

    static int32_t RegisterMemory(void* ctx, const CommMem* mem, const char* tag, void** handle)
    {
        CHK_PTR_NULL(ctx);
        CHK_PTR_NULL(mem);
        CHK_PTR_NULL(handle);
        return static_cast<HcclResult>(static_cast<EndpointT*>(ctx)->RegisterMemory(*mem, tag, handle));
    }

    static int32_t UnregisterMemory(void* ctx, void* handle)
    {
        CHK_PTR_NULL(ctx);
        return static_cast<HcclResult>(static_cast<EndpointT*>(ctx)->UnregisterMemory(handle));
    }

    static int32_t MemoryExport(void* ctx, void* handle, void** desc, uint32_t* descLen)
    {
        CHK_PTR_NULL(ctx);
        return static_cast<HcclResult>(static_cast<EndpointT*>(ctx)->MemoryExport(handle, desc, descLen));
    }

    static int32_t MemoryImport(void* ctx, const void* desc, uint32_t descLen, CommMem* outMem)
    {
        CHK_PTR_NULL(ctx);
        return static_cast<HcclResult>(static_cast<EndpointT*>(ctx)->MemoryImport(desc, descLen, outMem));
    }

    static int32_t MemoryUnimport(void* ctx, const void* desc, uint32_t descLen)
    {
        CHK_PTR_NULL(ctx);
        return static_cast<HcclResult>(static_cast<EndpointT*>(ctx)->MemoryUnimport(desc, descLen));
    }

    static int32_t GetListenPort(void* ctx, uint32_t* port)
    {
        CHK_PTR_NULL(ctx);
        CHK_PTR_NULL(port);
        return static_cast<HcclResult>(static_cast<EndpointT*>(ctx)->ServerSocketGetListenPort(port));
    }

    static HcommResult CreateEndpoint(const EndpointDesc* endpoint, void** endpointCtx)
    {
        CHK_PTR_NULL(endpoint);
        CHK_PTR_NULL(endpointCtx);
        auto ep = std::unique_ptr<EndpointT>(new (std::nothrow) EndpointT(*endpoint));
        CHK_SMART_PTR_NULL(ep);
        *endpointCtx = ep.release();
        return HCCL_SUCCESS;
    }

    // ==================== Channel Ops ====================

    static int32_t InitChannel(void* ctx)
    {
        CHK_PTR_NULL(ctx);
        return static_cast<ChannelT*>(ctx)->Init();
    }

    static int32_t DestroyChannel(void* ctx)
    {
        CHK_PTR_NULL(ctx);
        delete static_cast<ChannelT*>(ctx);
        return HCCL_SUCCESS;
    }

    static int32_t GetStatus(void* ctx, int32_t* status)
    {
        CHK_PTR_NULL(ctx);
        CHK_PTR_NULL(status);
        auto channelStatus = static_cast<ChannelT*>(ctx)->GetStatus();

        switch (channelStatus) {
            case ChannelStatus::FAILED:
                *status = HCOMM_CHANNEL_STATUS_FAILED;
                break;
            case ChannelStatus::SOCKET_TIMEOUT:
                *status = HCOMM_CHANNEL_STATUS_TIMEOUT;
                break;
            case ChannelStatus::READY:
                *status = HCOMM_CHANNEL_STATUS_READY;
                break;
            default:
                *status = HCOMM_CHANNEL_STATUS_CONNECTING;
                break;
        }

        return HCCL_SUCCESS;
    }

    // write
    static int32_t WriteNbi(void* ctx, void* dst, const void* src, uint64_t len)
    {
        CHK_PTR_NULL(ctx);
        return static_cast<ChannelT*>(ctx)->Write(dst, src, len);
    }

    static int32_t WriteNbiOnThread(void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t len)
    {
        (void)thread;
        CHK_PTR_NULL(ctx);
        return static_cast<ChannelT*>(ctx)->Write(dst, src, len);
    }

    // writeWithNotify
    static int32_t WriteWithNotifyNbi(void* ctx, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx)
    {
        CHK_PTR_NULL(ctx);
        return static_cast<ChannelT*>(ctx)->WriteWithNotify(dst, src, len, remoteNotifyIdx);
    }

    static int32_t WriteWithNotifyNbiOnThread(
        void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx)
    {
        (void)thread;
        CHK_PTR_NULL(ctx);
        return static_cast<ChannelT*>(ctx)->WriteWithNotify(dst, src, len, remoteNotifyIdx);
    }

    // writeReduce

    // read
    static int32_t ReadNbi(void* ctx, void* dst, const void* src, uint64_t len)
    {
        CHK_PTR_NULL(ctx);
        return static_cast<ChannelT*>(ctx)->Read(dst, src, len);
    }

    static int32_t ReadNbiOnThread(void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t len)
    {
        (void)thread;
        CHK_PTR_NULL(ctx);
        return static_cast<ChannelT*>(ctx)->Read(dst, src, len);
    }

    // notify
    static int32_t NotifyRecord(void* ctx, uint32_t remoteNotifyIdx)
    {
        CHK_PTR_NULL(ctx);
        return static_cast<ChannelT*>(ctx)->NotifyRecord(remoteNotifyIdx);
    }

    static int32_t NotifyRecordOnThread(void* ctx, ThreadHandle thread, uint32_t remoteNotifyIdx)
    {
        (void)thread;
        CHK_PTR_NULL(ctx);
        return static_cast<ChannelT*>(ctx)->NotifyRecord(remoteNotifyIdx);
    }

    static int32_t NotifyWait(void* ctx, uint32_t localNotifyIdx, uint32_t timeout)
    {
        CHK_PTR_NULL(ctx);
        return static_cast<ChannelT*>(ctx)->NotifyWait(localNotifyIdx, timeout);
    }

    static int32_t NotifyWaitOnThread(void* ctx, ThreadHandle thread, uint32_t localNotifyIdx, uint32_t timeout)
    {
        (void)thread;
        CHK_PTR_NULL(ctx);
        return static_cast<ChannelT*>(ctx)->NotifyWait(localNotifyIdx, timeout);
    }

    // batch / fence / drain

    static int32_t Fence(void* ctx)
    {
        CHK_PTR_NULL(ctx);
        return static_cast<ChannelT*>(ctx)->ChannelFence();
    }

    static int32_t FenceOnThread(void* ctx, ThreadHandle thread)
    {
        (void)thread;
        CHK_PTR_NULL(ctx);
        return static_cast<ChannelT*>(ctx)->ChannelFence();
    }

    // ==================== Ops Tables ====================

    static HcommNicEndpointOps EndpointOps()
    {
        return {
            {HCOMM_NIC_ENDPOINT_OPS_VERSION, HCOMM_NIC_ENDPOINT_OPS_MAGIC_WORD, sizeof(HcommNicEndpointOps), 0},
            InitEndpoint,     // init
            DestroyEndpoint,  // destroy
            RegisterMemory,   // registerMemory
            UnregisterMemory, // unregisterMemory
            MemoryExport,     // memoryExport
            MemoryImport,     // memoryImport
            MemoryUnimport,   // memoryUnimport
            GetListenPort,    // getListenPort
        };
    }

    static HcommNicChannelOps ChannelOps()
    {
        return {
            {HCOMM_NIC_CHANNEL_OPS_VERSION, HCOMM_NIC_CHANNEL_OPS_MAGIC_WORD, sizeof(HcommNicChannelOps), 0},
            InitChannel,                // init
            DestroyChannel,             // destroy
            GetStatus,                  // getStatus
            WriteNbi,                   // writeNbi
            WriteNbiOnThread,           // writeNbiOnThread
            nullptr,                    // writeOnThread
            WriteWithNotifyNbi,         // writeWithNotifyNbi
            WriteWithNotifyNbiOnThread, // writeWithNotifyNbiOnThread
            nullptr,                    // writeWithNotifyOnThread
            nullptr,                    // writeReduceOnThread
            nullptr,                    // writeReduceWithNotifyOnThread
            ReadNbi,                    // readNbi
            ReadNbiOnThread,            // readNbiOnThread
            nullptr,                    // readOnThread
            nullptr,                    // readReduceOnThread
            NotifyRecord,               // notifyRecord
            NotifyRecordOnThread,       // notifyRecordOnThread
            NotifyWait,                 // notifyWait
            NotifyWaitOnThread,         // notifyWaitOnThread
            nullptr,                    // notifyWaitOnThreadWithDefaultTimeout
            nullptr,                    // batchTransferOnThread
            Fence,                      // fence
            FenceOnThread,              // fenceOnThread
            nullptr,                    // drainOnThread
        };
    }

    static HcommResult CreateChannel(void* endpointCtx, const HcommChannelDesc* channelDesc, void** channelCtx)
    {
        CHK_PTR_NULL(endpointCtx);
        CHK_PTR_NULL(channelDesc);
        CHK_PTR_NULL(channelCtx);
        auto ch = std::unique_ptr<ChannelT>(new (std::nothrow)
                                                ChannelT(reinterpret_cast<EndpointHandle>(endpointCtx), *channelDesc));
        CHK_SMART_PTR_NULL(ch);
        *channelCtx = ch.release();
        return HCCL_SUCCESS;
    }

    // ==================== Export Helpers ====================

    static int32_t CreateEndpointExport(
        const EndpointDesc* endpointDesc, void** outCtx, HcommNicEndpointOps** outOps, HcommNicEndpointOps* endpointOps)
    {
        CHK_PTR_NULL(endpointDesc);
        CHK_PTR_NULL(outCtx);
        CHK_PTR_NULL(outOps);
        CHK_RET(static_cast<HcclResult>(CreateEndpoint(endpointDesc, outCtx)));
        *outOps = endpointOps;
        return HCCL_SUCCESS;
    }

    static int32_t CreateChannelExport(
        void* epCtx, const HcommChannelDesc* channelDesc, void** outCtx, HcommNicChannelOps** outOps,
        HcommNicChannelOps* channelOps)
    {
        CHK_PTR_NULL(channelDesc);
        CHK_PTR_NULL(outCtx);
        CHK_PTR_NULL(outOps);
        CHK_RET(static_cast<HcclResult>(CreateChannel(epCtx, channelDesc, outCtx)));
        *outOps = channelOps;
        return HCCL_SUCCESS;
    }
};

} // namespace hcomm_experimental

#define HCOMM_EXPERIMENTAL_NIC_PLUGIN_EXPORTS(PluginOpsType)                                          \
    extern "C" const HcommNicPluginInfo* HcommNicPluginGetInfo(void) { return &kPluginInfo; }         \
                                                                                                      \
    extern "C" int32_t HcommNicPluginCreateEndpoint(                                                  \
        const EndpointDesc* endpointDesc, void** outCtx, HcommNicEndpointOps** outOps)                \
    {                                                                                                 \
        return PluginOpsType::CreateEndpointExport(endpointDesc, outCtx, outOps, &kEndpointOps);      \
    }                                                                                                 \
                                                                                                      \
    extern "C" int32_t HcommNicPluginCreateChannel(                                                   \
        void* epCtx, const HcommChannelDesc* channelDesc, void** outCtx, HcommNicChannelOps** outOps) \
    {                                                                                                 \
        return PluginOpsType::CreateChannelExport(epCtx, channelDesc, outCtx, outOps, &kChannelOps);  \
    }

#endif // HCOMM_EXPERIMENTAL_PLUGIN_OPS_H

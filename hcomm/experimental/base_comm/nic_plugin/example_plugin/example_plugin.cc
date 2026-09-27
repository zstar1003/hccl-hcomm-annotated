/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcomm_nic_plugin.h"

namespace {
/* ---- 插件元信息：声明一个自定义协议号（拓展模式） ---- */
// 拓展模式要求协议号 ≥ COMM_PROTOCOL_CUSTOM_BASE(1000)。该常量定义于 HCOMM 内部
// nic_plugin_manager.h，未随 SDK 头文件导出，插件侧按约定使用 ≥1000 的自定义协议号。
constexpr CommProtocol kMyProtocol = static_cast<CommProtocol>(1000);

const HcommNicPluginInfo kPluginInfo = {
    {HCOMM_NIC_PLUGIN_INFO_VERSION, HCOMM_NIC_PLUGIN_INFO_MAGIC_WORD, sizeof(HcommNicPluginInfo), 0},
    "example_plugin", // 插件名称
    1U,               // protocolCount
    {kMyProtocol, COMM_PROTOCOL_RESERVED, COMM_PROTOCOL_RESERVED, COMM_PROTOCOL_RESERVED},
    {0, 0, 0, 0, 0, 0, 0, 0}, // reserved[8]
};

/* ==================== Endpoint ops（实现体留空） ==================== */
static int32_t EndpointInit(void* ctx)
{
    (void)ctx;
    return 0;
}
static int32_t EndpointDestroy(void* ctx)
{
    (void)ctx;
    return 0;
}
static int32_t RegisterMemory(void* ctx, const CommMem* mem, const char* tag, void** handle)
{
    (void)ctx;
    (void)mem;
    (void)tag;
    (void)handle;
    return 0;
}
static int32_t UnregisterMemory(void* ctx, void* handle)
{
    (void)ctx;
    (void)handle;
    return 0;
}
static int32_t MemoryExport(void* ctx, void* handle, void** desc, uint32_t* descLen)
{
    (void)ctx;
    (void)handle;
    (void)desc;
    (void)descLen;
    return 0;
}
static int32_t MemoryImport(void* ctx, const void* desc, uint32_t descLen, CommMem* outMem)
{
    (void)ctx;
    (void)desc;
    (void)descLen;
    (void)outMem;
    return 0;
}
static int32_t MemoryUnimport(void* ctx, const void* desc, uint32_t descLen)
{
    (void)ctx;
    (void)desc;
    (void)descLen;
    return 0;
}
static int32_t GetListenPort(void* ctx, uint32_t* port)
{
    (void)ctx;
    (void)port;
    return 0;
}

HcommNicEndpointOps kEndpointOps = {
    {HCOMM_NIC_ENDPOINT_OPS_VERSION, HCOMM_NIC_ENDPOINT_OPS_MAGIC_WORD, sizeof(HcommNicEndpointOps), 0},
    EndpointInit,     // init
    EndpointDestroy,  // destroy
    RegisterMemory,   // registerMemory
    UnregisterMemory, // unregisterMemory
    MemoryExport,     // memoryExport
    MemoryImport,     // memoryImport
    MemoryUnimport,   // memoryUnimport
    GetListenPort,    // getListenPort
};

/* ==================== Channel ops（实现体留空） ==================== */
static int32_t ChannelInit(void* ctx)
{
    (void)ctx;
    return 0;
}
static int32_t ChannelDestroy(void* ctx)
{
    (void)ctx;
    return 0;
}
static int32_t GetStatus(void* ctx, int32_t* status)
{
    (void)ctx;
    (void)status;
    return 0;
}

static int32_t WriteNbi(void* ctx, void* dst, const void* src, uint64_t len)
{
    (void)ctx;
    (void)dst;
    (void)src;
    (void)len;
    return 0;
}
static int32_t WriteNbiOnThread(void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t len)
{
    (void)ctx;
    (void)thread;
    (void)dst;
    (void)src;
    (void)len;
    return 0;
}
static int32_t WriteOnThread(void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t len)
{
    (void)ctx;
    (void)thread;
    (void)dst;
    (void)src;
    (void)len;
    return 0;
}
static int32_t WriteWithNotifyNbi(void* ctx, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx)
{
    (void)ctx;
    (void)dst;
    (void)src;
    (void)len;
    (void)remoteNotifyIdx;
    return 0;
}
static int32_t WriteWithNotifyNbiOnThread(
    void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx)
{
    (void)ctx;
    (void)thread;
    (void)dst;
    (void)src;
    (void)len;
    (void)remoteNotifyIdx;
    return 0;
}
static int32_t WriteWithNotifyOnThread(
    void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx)
{
    (void)ctx;
    (void)thread;
    (void)dst;
    (void)src;
    (void)len;
    (void)remoteNotifyIdx;
    return 0;
}
static int32_t WriteReduceOnThread(
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
    return 0;
}
static int32_t WriteReduceWithNotifyOnThread(
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
    return 0;
}

static int32_t ReadNbi(void* ctx, void* dst, const void* src, uint64_t len)
{
    (void)ctx;
    (void)dst;
    (void)src;
    (void)len;
    return 0;
}
static int32_t ReadNbiOnThread(void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t len)
{
    (void)ctx;
    (void)thread;
    (void)dst;
    (void)src;
    (void)len;
    return 0;
}
static int32_t ReadOnThread(void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t len)
{
    (void)ctx;
    (void)thread;
    (void)dst;
    (void)src;
    (void)len;
    return 0;
}
static int32_t ReadReduceOnThread(
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
    return 0;
}

static int32_t NotifyRecord(void* ctx, uint32_t remoteNotifyIdx)
{
    (void)ctx;
    (void)remoteNotifyIdx;
    return 0;
}
static int32_t NotifyRecordOnThread(void* ctx, ThreadHandle thread, uint32_t remoteNotifyIdx)
{
    (void)ctx;
    (void)thread;
    (void)remoteNotifyIdx;
    return 0;
}
static int32_t NotifyWait(void* ctx, uint32_t localNotifyIdx, uint32_t timeOut)
{
    (void)ctx;
    (void)localNotifyIdx;
    (void)timeOut;
    return 0;
}
static int32_t NotifyWaitOnThread(void* ctx, ThreadHandle thread, uint32_t localNotifyIdx, uint32_t timeOut)
{
    (void)ctx;
    (void)thread;
    (void)localNotifyIdx;
    (void)timeOut;
    return 0;
}
static int32_t NotifyWaitOnThreadWithDefaultTimeout(void* ctx, ThreadHandle thread, uint32_t localNotifyIdx)
{
    (void)ctx;
    (void)thread;
    (void)localNotifyIdx;
    return 0;
}
static int32_t BatchTransferOnThread(
    void* ctx, ThreadHandle thread, const HcommBatchTransferDesc* transferDescs, uint32_t transferDescNum)
{
    (void)ctx;
    (void)thread;
    (void)transferDescs;
    (void)transferDescNum;
    return 0;
}

static int32_t Fence(void* ctx)
{
    (void)ctx;
    return 0;
}
static int32_t FenceOnThread(void* ctx, ThreadHandle thread)
{
    (void)ctx;
    (void)thread;
    return 0;
}
static int32_t DrainOnThread(void* ctx, ThreadHandle thread)
{
    (void)ctx;
    (void)thread;
    return 0;
}

HcommNicChannelOps kChannelOps = {
    {HCOMM_NIC_CHANNEL_OPS_VERSION, HCOMM_NIC_CHANNEL_OPS_MAGIC_WORD, sizeof(HcommNicChannelOps), 0},
    ChannelInit,                          // init
    ChannelDestroy,                       // destroy
    GetStatus,                            // getStatus
    WriteNbi,                             // writeNbi
    WriteNbiOnThread,                     // writeNbiOnThread
    WriteOnThread,                        // writeOnThread
    WriteWithNotifyNbi,                   // writeWithNotifyNbi
    WriteWithNotifyNbiOnThread,           // writeWithNotifyNbiOnThread
    WriteWithNotifyOnThread,              // writeWithNotifyOnThread
    WriteReduceOnThread,                  // writeReduceOnThread
    WriteReduceWithNotifyOnThread,        // writeReduceWithNotifyOnThread
    ReadNbi,                              // readNbi
    ReadNbiOnThread,                      // readNbiOnThread
    ReadOnThread,                         // readOnThread
    ReadReduceOnThread,                   // readReduceOnThread
    NotifyRecord,                         // notifyRecord
    NotifyRecordOnThread,                 // notifyRecordOnThread
    NotifyWait,                           // notifyWait
    NotifyWaitOnThread,                   // notifyWaitOnThread
    NotifyWaitOnThreadWithDefaultTimeout, // notifyWaitOnThreadWithDefaultTimeout
    BatchTransferOnThread,                // batchTransferOnThread
    Fence,                                // fence
    FenceOnThread,                        // fenceOnThread
    DrainOnThread,                        // drainOnThread
};

} // namespace

/* ==================== 3 个导出符号 ==================== */

extern "C" const HcommNicPluginInfo* HcommNicPluginGetInfo(void) { return &kPluginInfo; }

extern "C" int32_t
HcommNicPluginCreateEndpoint(const EndpointDesc* endpointDesc, void** outCtx, HcommNicEndpointOps** outOps)
{
    (void)endpointDesc;
    if (outCtx != nullptr) {
        *outCtx = nullptr; // 框架示例：仅作为独立编译示例，不创建真实 endpoint 上下文，实际使用时需要创建对应的上下文
    }
    if (outOps != nullptr) {
        *outOps = &kEndpointOps;
    }
    return 0;
}

extern "C" int32_t HcommNicPluginCreateChannel(
    void* epCtx, const HcommChannelDesc* channelDesc, void** outCtx, HcommNicChannelOps** outOps)
{
    (void)epCtx;
    (void)channelDesc;
    if (outCtx != nullptr) {
        *outCtx = nullptr; // 框架示例：仅作为独立编译示例，不创建真实 channel 上下文，实际使用时需要创建对应的上下文
    }
    if (outOps != nullptr) {
        *outOps = &kChannelOps;
    }
    return 0;
}

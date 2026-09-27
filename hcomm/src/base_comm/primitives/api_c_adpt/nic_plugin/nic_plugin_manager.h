/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_NIC_PLUGIN_MANAGER_H
#define HCOMM_NIC_PLUGIN_MANAGER_H

#include "cast_utils.h"
#include <stddef.h>
#include <stdint.h>

#include "hcomm_c_adpt.h"
#include "hcomm_nic_plugin.h"

namespace hcomm {

struct NicPluginEntry {
    void* soHandle;
    const HcommNicPluginInfo* info;
    HcommNicPluginCreateEndpointFunc createEndpoint;
    HcommNicPluginCreateChannelFunc createChannel;
};

constexpr uintptr_t HCOMM_PLUGIN_HANDLE_FLAG = (static_cast<uintptr_t>(1) << 63);
constexpr int32_t COMM_PROTOCOL_CUSTOM_BASE = 1000;

class Channel;

inline ChannelHandle MakePluginChHandle(ChannelHandle h)
{
    return static_cast<ChannelHandle>(static_cast<uintptr_t>(h) | HCOMM_PLUGIN_HANDLE_FLAG);
}

inline ::hcomm::Channel* ChannelFromHandle(ChannelHandle h)
{
    return ReinterpretAs<::hcomm::Channel*>(
        static_cast<uintptr_t>(h) & ~static_cast<uintptr_t>(HCOMM_PLUGIN_HANDLE_FLAG));
}

#define IS_PLUGIN_HANDLE(h) ((((static_cast<uintptr_t>(h))) & ::hcomm::HCOMM_PLUGIN_HANDLE_FLAG) != 0)
#define MAKE_PLUGIN_CH_HANDLE(p) ::hcomm::MakePluginChHandle(p)
#define CHANNEL_FROM_HANDLE(h) ::hcomm::ChannelFromHandle(h)

void LoadAllNicPlugins();
const NicPluginEntry* FindHostNicPlugin(CommProtocol protocol);
bool ValidatePluginInfo(
    const char* soPath, const HcommNicPluginInfo* info, HcommNicPluginCreateEndpointFunc createEndpoint,
    HcommNicPluginCreateChannelFunc createChannel);

HcommResult FillDefaultEndpointOps(const HcommNicEndpointOps* src, HcommNicEndpointOps** outOps);

int32_t DefaultEndpointInit(void* ctx);
int32_t DefaultEndpointRegisterMemory(void* ctx, const CommMem* mem, const char* tag, void** handle);
int32_t DefaultEndpointUnregisterMemory(void* ctx, void* handle);
int32_t DefaultEndpointMemoryExport(void* ctx, void* handle, void** desc, uint32_t* descLen);
int32_t DefaultEndpointMemoryImport(void* ctx, const void* desc, uint32_t descLen, CommMem* outMem);
int32_t DefaultEndpointMemoryUnimport(void* ctx, const void* desc, uint32_t descLen);
int32_t DefaultEndpointGetListenPort(void* ctx, uint32_t* port);

// HcommNicEndpointOps 新增成员时，在此追加一行 F(op字段名, 默认实现函数) 即可，destroy要求插件必须实现
#define FOR_EACH_ENDPOINT_OP_DEFAULT(F)                  \
    F(init, DefaultEndpointInit)                         \
    F(registerMemory, DefaultEndpointRegisterMemory)     \
    F(unregisterMemory, DefaultEndpointUnregisterMemory) \
    F(memoryExport, DefaultEndpointMemoryExport)         \
    F(memoryImport, DefaultEndpointMemoryImport)         \
    F(memoryUnimport, DefaultEndpointMemoryUnimport)     \
    F(getListenPort, DefaultEndpointGetListenPort)

// 供 FOR_EACH_ENDPOINT_OP_DEFAULT 使用的填充器
#define FILL_ENDPOINT_OP_DEFAULT(op, defaultFunc)                                               \
    if (!IsPluginOpAvailable(dst, offsetof(HcommNicEndpointOps, op), sizeof(decltype(dst->op))) \
        || dst->op == nullptr) {                                                                \
        dst->op = defaultFunc;                                                                  \
    }

HcommResult FillDefaultChannelOps(const HcommNicChannelOps* src, HcommNicChannelOps** outOps);

int32_t DefaultChannelInit(void* ctx);
int32_t DefaultChannelGetStatus(void* ctx, int32_t* status);
int32_t DefaultChannelWriteNbi(void* ctx, void* dst, const void* src, uint64_t len);
int32_t DefaultChannelWriteNbiOnThread(void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t len);
int32_t DefaultChannelWriteOnThread(void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t len);
int32_t DefaultChannelWriteWithNotifyNbi(void* ctx, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx);
int32_t DefaultChannelWriteWithNotifyNbiOnThread(
    void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx);
int32_t DefaultChannelWriteWithNotifyOnThread(
    void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx);
int32_t DefaultChannelWriteReduceOnThread(
    void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t count, HcommDataType dataType,
    HcommReduceOp reduceOp);
int32_t DefaultChannelWriteReduceWithNotifyOnThread(
    void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t count, HcommDataType dataType,
    HcommReduceOp reduceOp, uint32_t remoteNotifyIdx);
int32_t DefaultChannelReadNbi(void* ctx, void* dst, const void* src, uint64_t len);
int32_t DefaultChannelReadNbiOnThread(void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t len);
int32_t DefaultChannelReadOnThread(void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t len);
int32_t DefaultChannelReadReduceOnThread(
    void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t count, HcommDataType dataType,
    HcommReduceOp reduceOp);
int32_t DefaultChannelNotifyRecord(void* ctx, uint32_t remoteNotifyIdx);
int32_t DefaultChannelNotifyRecordOnThread(void* ctx, ThreadHandle thread, uint32_t remoteNotifyIdx);
int32_t DefaultChannelNotifyWait(void* ctx, uint32_t localNotifyIdx, uint32_t timeOut);
int32_t DefaultChannelNotifyWaitOnThread(void* ctx, ThreadHandle thread, uint32_t localNotifyIdx, uint32_t timeOut);
int32_t DefaultChannelNotifyWaitOnThreadWithDefaultTimeout(void* ctx, ThreadHandle thread, uint32_t localNotifyIdx);
int32_t DefaultChannelBatchTransferOnThread(
    void* ctx, ThreadHandle thread, const HcommBatchTransferDesc* transferDescs, uint32_t transferDescNum);
int32_t DefaultChannelFence(void* ctx);
int32_t DefaultChannelFenceOnThread(void* ctx, ThreadHandle thread);
int32_t DefaultChannelDrainOnThread(void* ctx, ThreadHandle thread);

// HcommNicChannelOps 新增成员时，在此追加一行 F(op字段名, 默认实现函数) 即可，destroy要求插件必须实现
#define FOR_EACH_CHANNEL_OP_DEFAULT(F)                                                          \
    F(init, DefaultChannelInit)                                                                 \
    F(getStatus, DefaultChannelGetStatus)                                                       \
    F(writeNbi, DefaultChannelWriteNbi)                                                         \
    F(writeNbiOnThread, DefaultChannelWriteNbiOnThread)                                         \
    F(writeOnThread, DefaultChannelWriteOnThread)                                               \
    F(writeWithNotifyNbi, DefaultChannelWriteWithNotifyNbi)                                     \
    F(writeWithNotifyNbiOnThread, DefaultChannelWriteWithNotifyNbiOnThread)                     \
    F(writeWithNotifyOnThread, DefaultChannelWriteWithNotifyOnThread)                           \
    F(writeReduceOnThread, DefaultChannelWriteReduceOnThread)                                   \
    F(writeReduceWithNotifyOnThread, DefaultChannelWriteReduceWithNotifyOnThread)               \
    F(readNbi, DefaultChannelReadNbi)                                                           \
    F(readNbiOnThread, DefaultChannelReadNbiOnThread)                                           \
    F(readOnThread, DefaultChannelReadOnThread)                                                 \
    F(readReduceOnThread, DefaultChannelReadReduceOnThread)                                     \
    F(notifyRecord, DefaultChannelNotifyRecord)                                                 \
    F(notifyRecordOnThread, DefaultChannelNotifyRecordOnThread)                                 \
    F(notifyWait, DefaultChannelNotifyWait)                                                     \
    F(notifyWaitOnThread, DefaultChannelNotifyWaitOnThread)                                     \
    F(notifyWaitOnThreadWithDefaultTimeout, DefaultChannelNotifyWaitOnThreadWithDefaultTimeout) \
    F(batchTransferOnThread, DefaultChannelBatchTransferOnThread)                               \
    F(fence, DefaultChannelFence)                                                               \
    F(fenceOnThread, DefaultChannelFenceOnThread)                                               \
    F(drainOnThread, DefaultChannelDrainOnThread)

// 供 FOR_EACH_CHANNEL_OP_DEFAULT 使用的填充器
#define FILL_CHANNEL_OP_DEFAULT(op, defaultFunc)                                               \
    if (!IsPluginOpAvailable(dst, offsetof(HcommNicChannelOps, op), sizeof(decltype(dst->op))) \
        || dst->op == nullptr) {                                                               \
        dst->op = defaultFunc;                                                                 \
    }

bool ValidateEndpointOps(const HcommNicEndpointOps* ops);
bool ValidateChannelOps(const HcommNicChannelOps* ops);

template <typename PluginOps>
bool IsPluginOpAvailable(const PluginOps* ops, size_t opOffset, size_t opSize)
{
    return ops != nullptr && ops->header.size >= opOffset + opSize;
}

template <typename PluginOps>
void DestroyPluginCtx(PluginOps* ops, void* pluginCtx)
{
    if (ops != nullptr && IsPluginOpAvailable(ops, offsetof(PluginOps, destroy), sizeof(ops->destroy))
        && ops->destroy != nullptr) {
        int32_t ret = ops->destroy(pluginCtx);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING("[%s] plugin destroy failed, ret[%d].", __func__, ret);
        }
    }
}

} // namespace hcomm

#endif // HCOMM_NIC_PLUGIN_MANAGER_H

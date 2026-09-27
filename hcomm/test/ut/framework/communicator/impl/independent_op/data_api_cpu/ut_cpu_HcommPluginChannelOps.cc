/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include <memory>
#include <mockcpp/mockcpp.hpp>

#define private public
#define protected public

#include "adapter_rts_common.h"
#include "hccl_api_data.h"
#include "hcomm_nic_plugin.h"
#include "hcomm_primitives.h"
#include "hcomm_res.h"
#include "new/hccl_primitive_remote.h"
#include "builtin_channel_ops.h"
#include "channel_process.h"
#include "cpu_ts_thread.h"
#include "host_cpu_roce_channel.h"
#include "nic_plugin_holder.h"
#include "nic_plugin_manager.h"

using namespace hcomm;

namespace {

constexpr int FAKE_PROTOCOL = 1001;

// ---------- fake channel state & ops ----------
struct FakePluginChannelState {
    uint32_t getStatusCalls = 0;
    uint32_t writeNbiCalls = 0;
    uint32_t writeNbiOnThreadCalls = 0;
    uint32_t writeOnThreadCalls = 0;
    uint32_t writeWithNotifyNbiCalls = 0;
    uint32_t writeWithNotifyNbiOnThreadCalls = 0;
    uint32_t writeWithNotifyOnThreadCalls = 0;
    uint32_t writeReduceOnThreadCalls = 0;
    uint32_t writeReduceWithNotifyOnThreadCalls = 0;
    uint32_t readNbiCalls = 0;
    uint32_t readNbiOnThreadCalls = 0;
    uint32_t readOnThreadCalls = 0;
    uint32_t readReduceOnThreadCalls = 0;
    uint32_t notifyRecordCalls = 0;
    uint32_t notifyRecordOnThreadCalls = 0;
    uint32_t notifyWaitCalls = 0;
    uint32_t notifyWaitOnThreadCalls = 0;
    uint32_t notifyWaitOnThreadWithDefaultTimeoutCalls = 0;
    uint32_t batchTransferOnThreadCalls = 0;
    uint32_t fenceCalls = 0;
    uint32_t fenceOnThreadCalls = 0;
    uint32_t drainOnThreadCalls = 0;
};

FakePluginChannelState g_fakeState{};
HcommNicChannelOps g_fakeChannelOps{};
HcommNicChannelOps g_unsupportedChannelOps{};

constexpr ThreadHandle FAKE_THREAD = 0x1357U;
constexpr uint32_t FAKE_NOTIFY_IDX = 3U;
constexpr uint32_t FAKE_TIMEOUT = 100U;

// init / destroy
int32_t FakeChannelInit(void* ctx)
{
    (void)ctx;
    return HCCL_SUCCESS;
}
int32_t FakeChannelDestroy(void* ctx)
{
    (void)ctx;
    return HCCL_SUCCESS;
}

// getStatus
int32_t FakeGetStatus(void* ctx, int32_t* status)
{
    (void)ctx;
    (void)status;
    static_cast<FakePluginChannelState*>(ctx)->getStatusCalls++;
    return HCCL_SUCCESS;
}

// write
int32_t FakeWriteNbi(void* ctx, void* dst, const void* src, uint64_t len)
{
    (void)dst;
    (void)src;
    (void)len;
    static_cast<FakePluginChannelState*>(ctx)->writeNbiCalls++;
    return HCCL_SUCCESS;
}
int32_t FakeWriteNbiOnThread(void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t len)
{
    (void)thread;
    (void)dst;
    (void)src;
    (void)len;
    static_cast<FakePluginChannelState*>(ctx)->writeNbiOnThreadCalls++;
    return HCCL_SUCCESS;
}
int32_t FakeWriteOnThread(void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t len)
{
    (void)thread;
    (void)dst;
    (void)src;
    (void)len;
    static_cast<FakePluginChannelState*>(ctx)->writeOnThreadCalls++;
    return HCCL_SUCCESS;
}

// writeWithNotify
int32_t FakeWriteWithNotifyNbi(void* ctx, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx)
{
    (void)dst;
    (void)src;
    (void)len;
    (void)remoteNotifyIdx;
    static_cast<FakePluginChannelState*>(ctx)->writeWithNotifyNbiCalls++;
    return HCCL_SUCCESS;
}
int32_t FakeWriteWithNotifyNbiOnThread(
    void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx)
{
    (void)thread;
    (void)dst;
    (void)src;
    (void)len;
    (void)remoteNotifyIdx;
    static_cast<FakePluginChannelState*>(ctx)->writeWithNotifyNbiOnThreadCalls++;
    return HCCL_SUCCESS;
}
int32_t FakeWriteWithNotifyOnThread(
    void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx)
{
    (void)thread;
    (void)dst;
    (void)src;
    (void)len;
    (void)remoteNotifyIdx;
    static_cast<FakePluginChannelState*>(ctx)->writeWithNotifyOnThreadCalls++;
    return HCCL_SUCCESS;
}

// writeReduce
int32_t FakeWriteReduceOnThread(
    void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t count, HcommDataType dataType,
    HcommReduceOp reduceOp)
{
    (void)thread;
    (void)dst;
    (void)src;
    (void)count;
    (void)dataType;
    (void)reduceOp;
    static_cast<FakePluginChannelState*>(ctx)->writeReduceOnThreadCalls++;
    return HCCL_SUCCESS;
}
int32_t FakeWriteReduceWithNotifyOnThread(
    void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t count, HcommDataType dataType,
    HcommReduceOp reduceOp, uint32_t remoteNotifyIdx)
{
    (void)thread;
    (void)dst;
    (void)src;
    (void)count;
    (void)dataType;
    (void)reduceOp;
    (void)remoteNotifyIdx;
    static_cast<FakePluginChannelState*>(ctx)->writeReduceWithNotifyOnThreadCalls++;
    return HCCL_SUCCESS;
}

// read
int32_t FakeReadNbi(void* ctx, void* dst, const void* src, uint64_t len)
{
    (void)dst;
    (void)src;
    (void)len;
    static_cast<FakePluginChannelState*>(ctx)->readNbiCalls++;
    return HCCL_SUCCESS;
}
int32_t FakeReadNbiOnThread(void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t len)
{
    (void)thread;
    (void)dst;
    (void)src;
    (void)len;
    static_cast<FakePluginChannelState*>(ctx)->readNbiOnThreadCalls++;
    return HCCL_SUCCESS;
}
int32_t FakeReadOnThread(void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t len)
{
    (void)thread;
    (void)dst;
    (void)src;
    (void)len;
    static_cast<FakePluginChannelState*>(ctx)->readOnThreadCalls++;
    return HCCL_SUCCESS;
}
int32_t FakeReadReduceOnThread(
    void* ctx, ThreadHandle thread, void* dst, const void* src, uint64_t count, HcommDataType dataType,
    HcommReduceOp reduceOp)
{
    (void)thread;
    (void)dst;
    (void)src;
    (void)count;
    (void)dataType;
    (void)reduceOp;
    static_cast<FakePluginChannelState*>(ctx)->readReduceOnThreadCalls++;
    return HCCL_SUCCESS;
}

// notify
int32_t FakeNotifyRecord(void* ctx, uint32_t remoteNotifyIdx)
{
    (void)remoteNotifyIdx;
    static_cast<FakePluginChannelState*>(ctx)->notifyRecordCalls++;
    return HCCL_SUCCESS;
}
int32_t FakeNotifyRecordOnThread(void* ctx, ThreadHandle thread, uint32_t remoteNotifyIdx)
{
    (void)thread;
    (void)remoteNotifyIdx;
    static_cast<FakePluginChannelState*>(ctx)->notifyRecordOnThreadCalls++;
    return HCCL_SUCCESS;
}
int32_t FakeNotifyWait(void* ctx, uint32_t localNotifyIdx, uint32_t timeout)
{
    (void)localNotifyIdx;
    (void)timeout;
    static_cast<FakePluginChannelState*>(ctx)->notifyWaitCalls++;
    return HCCL_SUCCESS;
}
int32_t FakeNotifyWaitOnThread(void* ctx, ThreadHandle thread, uint32_t localNotifyIdx, uint32_t timeout)
{
    (void)thread;
    (void)localNotifyIdx;
    (void)timeout;
    static_cast<FakePluginChannelState*>(ctx)->notifyWaitOnThreadCalls++;
    return HCCL_SUCCESS;
}
int32_t FakeNotifyWaitOnThreadWithDefaultTimeout(void* ctx, ThreadHandle thread, uint32_t localNotifyIdx)
{
    (void)thread;
    (void)localNotifyIdx;
    static_cast<FakePluginChannelState*>(ctx)->notifyWaitOnThreadWithDefaultTimeoutCalls++;
    return HCCL_SUCCESS;
}

// batch / fence / drain
int32_t FakeBatchTransferOnThread(
    void* ctx, ThreadHandle thread, const HcommBatchTransferDesc* transferDescs, uint32_t transferDescNum)
{
    (void)thread;
    (void)transferDescs;
    (void)transferDescNum;
    static_cast<FakePluginChannelState*>(ctx)->batchTransferOnThreadCalls++;
    return HCCL_SUCCESS;
}
int32_t FakeFence(void* ctx)
{
    static_cast<FakePluginChannelState*>(ctx)->fenceCalls++;
    return HCCL_SUCCESS;
}
int32_t FakeFenceOnThread(void* ctx, ThreadHandle thread)
{
    (void)thread;
    static_cast<FakePluginChannelState*>(ctx)->fenceOnThreadCalls++;
    return HCCL_SUCCESS;
}
int32_t FakeDrainOnThread(void* ctx, ThreadHandle thread)
{
    (void)thread;
    static_cast<FakePluginChannelState*>(ctx)->drainOnThreadCalls++;
    return HCCL_SUCCESS;
}

void InitFakeChannelOps(HcommNicChannelOps* ops)
{
    *ops = {};
    ops->header = {HCOMM_NIC_CHANNEL_OPS_VERSION, HCOMM_NIC_CHANNEL_OPS_MAGIC_WORD, sizeof(HcommNicChannelOps), 0};
    ops->init = FakeChannelInit;
    ops->destroy = FakeChannelDestroy;
    ops->getStatus = FakeGetStatus;
    ops->writeNbi = FakeWriteNbi;
    ops->writeNbiOnThread = FakeWriteNbiOnThread;
    ops->writeOnThread = FakeWriteOnThread;
    ops->writeWithNotifyNbi = FakeWriteWithNotifyNbi;
    ops->writeWithNotifyNbiOnThread = FakeWriteWithNotifyNbiOnThread;
    ops->writeWithNotifyOnThread = FakeWriteWithNotifyOnThread;
    ops->writeReduceOnThread = FakeWriteReduceOnThread;
    ops->writeReduceWithNotifyOnThread = FakeWriteReduceWithNotifyOnThread;
    ops->readNbi = FakeReadNbi;
    ops->readNbiOnThread = FakeReadNbiOnThread;
    ops->readOnThread = FakeReadOnThread;
    ops->readReduceOnThread = FakeReadReduceOnThread;
    ops->notifyRecord = FakeNotifyRecord;
    ops->notifyRecordOnThread = FakeNotifyRecordOnThread;
    ops->notifyWait = FakeNotifyWait;
    ops->notifyWaitOnThread = FakeNotifyWaitOnThread;
    ops->notifyWaitOnThreadWithDefaultTimeout = FakeNotifyWaitOnThreadWithDefaultTimeout;
    ops->batchTransferOnThread = FakeBatchTransferOnThread;
    ops->fence = FakeFence;
    ops->fenceOnThread = FakeFenceOnThread;
    ops->drainOnThread = FakeDrainOnThread;
}

void InitFakeUnsupportedChannelOps(HcommNicChannelOps* ops)
{
    *ops = {};
    ops->header = {HCOMM_NIC_CHANNEL_OPS_VERSION, HCOMM_NIC_CHANNEL_OPS_MAGIC_WORD, sizeof(HcommNicChannelOps), 0};
    ops->init = FakeChannelInit;
    ops->destroy = FakeChannelDestroy;
}

// ---------- fake endpoint ops ----------
int32_t FakeEndpointInit(void* ctx)
{
    (void)ctx;
    return HCCL_SUCCESS;
}
int32_t FakeEndpointDestroy(void* ctx)
{
    (void)ctx;
    return HCCL_SUCCESS;
}

HcommNicEndpointOps g_fakeEndpointOps{};

void InitFakeEndpointOps()
{
    g_fakeEndpointOps = {};
    g_fakeEndpointOps.header
        = {HCOMM_NIC_ENDPOINT_OPS_VERSION, HCOMM_NIC_ENDPOINT_OPS_MAGIC_WORD, sizeof(HcommNicEndpointOps), 0};
    g_fakeEndpointOps.init = FakeEndpointInit;
    g_fakeEndpointOps.destroy = FakeEndpointDestroy;
}

// ---------- fake plugin entry + createEndpoint / createChannel ----------
int32_t FakeCreateEndpoint(const EndpointDesc* endpointDesc, void** outCtx, HcommNicEndpointOps** outOps)
{
    (void)endpointDesc;
    *outCtx = nullptr;
    *outOps = &g_fakeEndpointOps;
    return HCCL_SUCCESS;
}

int32_t FakeCreateChannel(void* epCtx, const HcommChannelDesc* channelDesc, void** outCtx, HcommNicChannelOps** outOps)
{
    (void)epCtx;
    (void)channelDesc;
    *outCtx = &g_fakeState;
    *outOps = &g_fakeChannelOps;
    return HCCL_SUCCESS;
}

int32_t FakeCreateUnsupportedChannel(
    void* epCtx, const HcommChannelDesc* channelDesc, void** outCtx, HcommNicChannelOps** outOps)
{
    (void)epCtx;
    (void)channelDesc;
    *outCtx = &g_fakeState;
    *outOps = &g_unsupportedChannelOps;
    return HCCL_SUCCESS;
}

NicPluginEntry g_fakePluginEntry{};
NicPluginEntry g_unsupportedPluginEntry{};

// MOCK: 替换 FindHostNicPlugin 返回 fake plugin entry
const NicPluginEntry* FakeFindHostNicPlugin(CommProtocol protocol)
{
    (void)protocol;
    return &g_fakePluginEntry;
}

const NicPluginEntry* FakeFindHostUnsupportedPlugin(CommProtocol protocol)
{
    (void)protocol;
    return &g_unsupportedPluginEntry;
}

EndpointHandle g_epHandle = nullptr;
ChannelHandle g_chHandle = 0;

// 统一创建 endpoint + channel
void CreateEpAndCh(EndpointHandle* epHandle, ChannelHandle* chHandle)
{
    EndpointDesc endpointDesc{};
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    endpointDesc.protocol = static_cast<CommProtocol>(FAKE_PROTOCOL);
    HcommEndpointCreate(&endpointDesc, epHandle);

    HcommChannelDesc channelDesc{};
    HcommChannelDescInit(&channelDesc, 1);
    channelDesc.remoteEndpoint.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    channelDesc.remoteEndpoint.protocol = static_cast<CommProtocol>(FAKE_PROTOCOL);
    HcommChannelCreate(*epHandle, COMM_ENGINE_CPU, &channelDesc, 1, chHandle);
}

void DestroyEpAndCh(EndpointHandle* epHandle, ChannelHandle* chHandle)
{
    if (*chHandle != 0) {
        HcommChannelDestroy(chHandle, 1);
        *chHandle = 0;
    }
    if (*epHandle != nullptr) {
        HcommEndpointDestroy(*epHandle);
        *epHandle = nullptr;
    }
}
} // namespace

class UtCpuHcommPluginChannelOps : public testing::Test {
protected:
    void SetUp() override
    {
        g_fakeState = {};
        InitFakeChannelOps(&g_fakeChannelOps);
        InitFakeUnsupportedChannelOps(&g_unsupportedChannelOps);
        InitFakeEndpointOps();
    }

    void TearDown() override
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }

    char srcBuf[8]{};
    char dstBuf[8]{};
    void* dst = dstBuf;
    const void* src = srcBuf;
    uint64_t len = sizeof(srcBuf);
    uint64_t count = 1;
    HcommBatchTransferDesc desc{};
};

// ---------- 测试用例 ----------

TEST_F(UtCpuHcommPluginChannelOps, Ut_PluginChannel_Expect_DispatchToPlugin)
{
    g_fakePluginEntry.createEndpoint = FakeCreateEndpoint;
    g_fakePluginEntry.createChannel = FakeCreateChannel;
    MOCKER((hcomm::FindHostNicPlugin)).stubs().will(invoke(FakeFindHostNicPlugin));
    CreateEpAndCh(&g_epHandle, &g_chHandle);

    int32_t status = 0;
    HcommBatchTransferDesc transferDesc{};
    transferDesc.transType = HCOMM_TRANSFER_TYPE_WRITE;
    transferDesc.transferInfo.write.dst = dst;
    transferDesc.transferInfo.write.src = const_cast<void*>(src);
    transferDesc.transferInfo.write.len = len;

    // getStatus
    HcommChannelGetStatus(&g_chHandle, 1, &status);
    // write
    HcommWriteNbi(g_chHandle, dst, src, len);
    HcommWriteNbiOnThread(FAKE_THREAD, g_chHandle, dst, src, len);
    HcommWriteOnThread(FAKE_THREAD, g_chHandle, dst, src, len);
    // writeWithNotify
    HcommWriteWithNotifyNbi(g_chHandle, dst, src, len, FAKE_NOTIFY_IDX);
    HcommWriteWithNotifyNbiOnThread(FAKE_THREAD, g_chHandle, dst, src, len, FAKE_NOTIFY_IDX);
    HcommWriteWithNotifyOnThread(FAKE_THREAD, g_chHandle, dst, src, len, FAKE_NOTIFY_IDX);
    // writeReduce
    HcommWriteReduceOnThread(FAKE_THREAD, g_chHandle, dst, src, count, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM);
    HcommWriteReduceWithNotifyOnThread(
        FAKE_THREAD, g_chHandle, dst, src, count, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM, FAKE_NOTIFY_IDX);
    // read
    HcommReadNbi(g_chHandle, dst, src, len);
    HcommReadNbiOnThread(FAKE_THREAD, g_chHandle, dst, src, len);
    HcommReadOnThread(FAKE_THREAD, g_chHandle, dst, src, len);
    HcommReadReduceOnThread(FAKE_THREAD, g_chHandle, dst, src, count, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM);
    // notify
    HcommChannelNotifyRecord(g_chHandle, FAKE_NOTIFY_IDX);
    HcommChannelNotifyRecordOnThread(FAKE_THREAD, g_chHandle, FAKE_NOTIFY_IDX);
    HcommChannelNotifyWait(g_chHandle, FAKE_NOTIFY_IDX, FAKE_TIMEOUT);
    HcommChannelNotifyWaitOnThread(FAKE_THREAD, g_chHandle, FAKE_NOTIFY_IDX, FAKE_TIMEOUT);
    HcommChannelNotifyWaitOnThreadWithDefaultTimeout(FAKE_THREAD, g_chHandle, FAKE_NOTIFY_IDX);
    // batch / fence / drain
    HcommBatchTransferOnThread(FAKE_THREAD, g_chHandle, &transferDesc, 1);
    HcommChannelFence(g_chHandle);
    HcommChannelFenceOnThread(FAKE_THREAD, g_chHandle);
    HcommChannelDrainOnThread(FAKE_THREAD, g_chHandle);

    EXPECT_EQ(g_fakeState.getStatusCalls, 1U);
    EXPECT_EQ(g_fakeState.writeNbiCalls, 1U);
    EXPECT_EQ(g_fakeState.writeNbiOnThreadCalls, 1U);
    EXPECT_EQ(g_fakeState.writeOnThreadCalls, 1U);
    EXPECT_EQ(g_fakeState.writeWithNotifyNbiCalls, 1U);
    EXPECT_EQ(g_fakeState.writeWithNotifyNbiOnThreadCalls, 1U);
    EXPECT_EQ(g_fakeState.writeWithNotifyOnThreadCalls, 1U);
    EXPECT_EQ(g_fakeState.writeReduceOnThreadCalls, 1U);
    EXPECT_EQ(g_fakeState.writeReduceWithNotifyOnThreadCalls, 1U);
    EXPECT_EQ(g_fakeState.readNbiCalls, 1U);
    EXPECT_EQ(g_fakeState.readNbiOnThreadCalls, 1U);
    EXPECT_EQ(g_fakeState.readOnThreadCalls, 1U);
    EXPECT_EQ(g_fakeState.readReduceOnThreadCalls, 1U);
    EXPECT_EQ(g_fakeState.notifyRecordCalls, 1U);
    EXPECT_EQ(g_fakeState.notifyRecordOnThreadCalls, 1U);
    EXPECT_EQ(g_fakeState.notifyWaitCalls, 1U);
    EXPECT_EQ(g_fakeState.notifyWaitOnThreadCalls, 1U);
    EXPECT_EQ(g_fakeState.notifyWaitOnThreadWithDefaultTimeoutCalls, 1U);
    EXPECT_EQ(g_fakeState.batchTransferOnThreadCalls, 1U);
    EXPECT_EQ(g_fakeState.fenceCalls, 1U);
    EXPECT_EQ(g_fakeState.fenceOnThreadCalls, 1U);
    EXPECT_EQ(g_fakeState.drainOnThreadCalls, 1U);

    DestroyEpAndCh(&g_epHandle, &g_chHandle);
}

TEST_F(UtCpuHcommPluginChannelOps, Ut_PluginChannel_Expect_DispatchToPlugin_NotSupport)
{
    // 重新 mock FindHostNicPlugin 返回 unsupported plugin entry
    GlobalMockObject::reset();
    g_unsupportedPluginEntry.createEndpoint = FakeCreateEndpoint;
    g_unsupportedPluginEntry.createChannel = FakeCreateUnsupportedChannel;
    MOCKER((hcomm::FindHostNicPlugin)).stubs().will(invoke(FakeFindHostUnsupportedPlugin));

    EndpointHandle epHandle2 = nullptr;
    ChannelHandle channel = 0;
    CreateEpAndCh(&epHandle2, &channel);

    int32_t status = 0;
    HcommBatchTransferDesc transferDesc{};
    transferDesc.transType = HCOMM_TRANSFER_TYPE_WRITE;
    transferDesc.transferInfo.write.dst = dst;
    transferDesc.transferInfo.write.src = const_cast<void*>(src);
    transferDesc.transferInfo.write.len = len;

    // getStatus
    EXPECT_EQ(HcommChannelGetStatus(&channel, 1, &status), HCCL_E_NOT_SUPPORT);
    // write
    EXPECT_EQ(HcommWriteNbi(channel, dst, src, len), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(HcommWriteNbiOnThread(FAKE_THREAD, channel, dst, src, len), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(HcommWriteOnThread(FAKE_THREAD, channel, dst, src, len), HCCL_E_NOT_SUPPORT);
    // writeWithNotify
    EXPECT_EQ(HcommWriteWithNotifyNbi(channel, dst, src, len, FAKE_NOTIFY_IDX), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(
        HcommWriteWithNotifyNbiOnThread(FAKE_THREAD, channel, dst, src, len, FAKE_NOTIFY_IDX), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(HcommWriteWithNotifyOnThread(FAKE_THREAD, channel, dst, src, len, FAKE_NOTIFY_IDX), HCCL_E_NOT_SUPPORT);
    // writeReduce
    EXPECT_EQ(
        HcommWriteReduceOnThread(FAKE_THREAD, channel, dst, src, count, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM),
        HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(
        HcommWriteReduceWithNotifyOnThread(
            FAKE_THREAD, channel, dst, src, count, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM, FAKE_NOTIFY_IDX),
        HCCL_E_NOT_SUPPORT);
    // read
    EXPECT_EQ(HcommReadNbi(channel, dst, src, len), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(HcommReadNbiOnThread(FAKE_THREAD, channel, dst, src, len), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(HcommReadOnThread(FAKE_THREAD, channel, dst, src, len), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(
        HcommReadReduceOnThread(FAKE_THREAD, channel, dst, src, count, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM),
        HCCL_E_NOT_SUPPORT);
    // notify
    EXPECT_EQ(HcommChannelNotifyRecord(channel, FAKE_NOTIFY_IDX), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(HcommChannelNotifyRecordOnThread(FAKE_THREAD, channel, FAKE_NOTIFY_IDX), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(HcommChannelNotifyWait(channel, FAKE_NOTIFY_IDX, FAKE_TIMEOUT), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(HcommChannelNotifyWaitOnThread(FAKE_THREAD, channel, FAKE_NOTIFY_IDX, FAKE_TIMEOUT), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(
        HcommChannelNotifyWaitOnThreadWithDefaultTimeout(FAKE_THREAD, channel, FAKE_NOTIFY_IDX), HCCL_E_NOT_SUPPORT);
    // batch / fence / drain
    EXPECT_EQ(HcommBatchTransferOnThread(FAKE_THREAD, channel, &transferDesc, 1), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(HcommChannelFence(channel), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(HcommChannelFenceOnThread(FAKE_THREAD, channel), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(HcommChannelDrainOnThread(FAKE_THREAD, channel), HCCL_E_NOT_SUPPORT);

    DestroyEpAndCh(&epHandle2, &channel);
}

TEST_F(UtCpuHcommPluginChannelOps, Ut_BuiltinChannel_Expect_DispatchToBuiltinOps)
{
    EndpointDesc epDesc{};
    epDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    epDesc.protocol = COMM_PROTOCOL_ROCE;
    EndpointHandle epHandle = nullptr;
    EXPECT_EQ(HcommEndpointCreate(&epDesc, &epHandle), HCCL_SUCCESS);

    // mock hrtGetDevice，CreateChannelsLoop / ChannelGet / ChannelDestroy 链路中依赖该函数
    MOCKER(hrtGetDevice).stubs().will(invoke(+[](s32* d) -> HcclResult {
        *d = 0;
        return HCCL_SUCCESS;
    }));
    // mock hrtGetDeviceType，Builtin* 系列函数需要 devType == DEV_TYPE_950 才走 Channel 虚函数路径
    MOCKER(hrtGetDeviceType).stubs().will(invoke(+[](DevType& t) -> HcclResult {
        t = DevType::DEV_TYPE_950;
        return HCCL_SUCCESS;
    }));

    // mock Channel::CreateChannel：跳过 HostCpuRoceChannel::Init() 避免真实硬件初始化
    MOCKER((hcomm::Channel::CreateChannel))
        .stubs()
        .will(invoke(
            +[](EndpointHandle endpointHandle, CommEngine engine, HcommChannelDesc channelDesc,
                std::shared_ptr<hcomm::Channel>& out) -> HcclResult {
                auto ch = std::make_shared<hcomm::HostCpuRoceChannel>(endpointHandle, channelDesc);
                ch->engine_ = engine;
                out = std::move(ch);
                return HCCL_SUCCESS;
            }));

    HcommChannelDesc chDesc{};
    HcommChannelDescInit(&chDesc, 1);
    chDesc.remoteEndpoint.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    chDesc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
    ChannelHandle chHandle = 0;
    EXPECT_EQ(HcommChannelCreate(epHandle, COMM_ENGINE_CPU, &chDesc, 1, &chHandle), HCCL_SUCCESS);

    // HOST+ROCE → HostCpuRoceChannel，MOCKER_CPP_VIRTUAL mock 子类虚函数
    hcomm::HostCpuRoceChannel* ch = nullptr;
    HcommChannelGet(chHandle, reinterpret_cast<void**>(&ch));
    MOCKER_CPP_VIRTUAL(ch, &hcomm::HostCpuRoceChannel::Write).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(ch, &hcomm::HostCpuRoceChannel::WriteWithNotify).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(ch, &hcomm::HostCpuRoceChannel::Read).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(ch, &hcomm::HostCpuRoceChannel::ChannelFence).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(ch, &hcomm::HostCpuRoceChannel::NotifyRecord).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(ch, &hcomm::HostCpuRoceChannel::NotifyWait)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(
        ch, static_cast<hcomm::ChannelStatus (hcomm::HostCpuRoceChannel::*)()>(&hcomm::HostCpuRoceChannel::GetStatus))
        .stubs()
        .will(returnValue(hcomm::ChannelStatus(hcomm::ChannelStatus::READY)));

    char srcBuf[8]{};
    char dstBuf[8]{};
    int32_t status = 0;
    HcommBatchTransferDesc transferDesc{};
    transferDesc.transType = HCOMM_TRANSFER_TYPE_WRITE;
    transferDesc.transferInfo.write.dst = dstBuf;
    transferDesc.transferInfo.write.src = srcBuf;
    transferDesc.transferInfo.write.len = sizeof(srcBuf);

    // getStatus
    EXPECT_EQ(HcommChannelGetStatus(&chHandle, 1, &status), HCCL_SUCCESS);
    // write
    EXPECT_EQ(HcommWriteNbi(chHandle, dstBuf, srcBuf, sizeof(srcBuf)), HCCL_SUCCESS);
    EXPECT_EQ(HcommWriteNbiOnThread(FAKE_THREAD, chHandle, dstBuf, srcBuf, sizeof(srcBuf)), HCCL_SUCCESS);
    // writeWithNotify
    EXPECT_EQ(HcommWriteWithNotifyNbi(chHandle, dstBuf, srcBuf, sizeof(srcBuf), FAKE_NOTIFY_IDX), HCCL_SUCCESS);
    EXPECT_EQ(
        HcommWriteWithNotifyNbiOnThread(FAKE_THREAD, chHandle, dstBuf, srcBuf, sizeof(srcBuf), FAKE_NOTIFY_IDX),
        HCCL_SUCCESS);
    // read
    EXPECT_EQ(HcommReadNbi(chHandle, dstBuf, srcBuf, sizeof(srcBuf)), HCCL_SUCCESS);
    EXPECT_EQ(HcommReadNbiOnThread(FAKE_THREAD, chHandle, dstBuf, srcBuf, sizeof(srcBuf)), HCCL_SUCCESS);
    // notify
    EXPECT_EQ(HcommChannelNotifyRecord(chHandle, FAKE_NOTIFY_IDX), HCCL_SUCCESS);
    EXPECT_EQ(HcommChannelNotifyWait(chHandle, FAKE_NOTIFY_IDX, FAKE_TIMEOUT), HCCL_SUCCESS);
    EXPECT_EQ(
        HcommChannelNotifyWaitOnThreadWithDefaultTimeout(FAKE_THREAD, chHandle, FAKE_NOTIFY_IDX), HCCL_E_NOT_SUPPORT);
    // batch / fence
    EXPECT_EQ(HcommBatchTransferOnThread(FAKE_THREAD, chHandle, &transferDesc, 1), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(HcommChannelFence(chHandle), HCCL_SUCCESS);
    EXPECT_EQ(HcommChannelFenceOnThread(FAKE_THREAD, chHandle), HCCL_SUCCESS);

    HcommChannelDestroy(&chHandle, 1);
    HcommEndpointDestroy(epHandle);
}

TEST_F(UtCpuHcommPluginChannelOps, Ut_BuiltinDrainOn950WithThread_Expect_DispatchToChannelDrain)
{
    hccl::CpuTsThread threadOnHost{hccl::StreamType::STREAM_TYPE_ONLINE, 1, hccl::NotifyLoadType::DEVICE_NOTIFY};
    threadOnHost.stream_.reset(new (std::nothrow) hccl::Stream());
    const ThreadHandle thread = reinterpret_cast<ThreadHandle>(&threadOnHost);
    EndpointHandle epHandle = reinterpret_cast<EndpointHandle>(0x01);
    HcommChannelDesc channelDesc{};
    hcomm::HostCpuRoceChannel channelOnHost{epHandle, channelDesc};
    channelOnHost.SetNicChannelCtx(const_cast<HcommNicChannelOps*>(&g_BuiltinChannelOps), &channelOnHost);
    const ChannelHandle channel = reinterpret_cast<ChannelHandle>(&channelOnHost);
    DevType devType = DevType::DEV_TYPE_950;

    MOCKER(&hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    MOCKER(&HcclRemoteDrain).expects(never());
    MOCKER_CPP_VIRTUAL(channelOnHost, &hcomm::HostCpuRoceChannel::ChannelDrain)
        .expects(once())
        .will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(HcommChannelDrainOnThread(thread, channel), HCCL_SUCCESS);
}

TEST_F(UtCpuHcommPluginChannelOps, Ut_BuiltinDrainOn950WithoutThread_Expect_DispatchToChannelDrain)
{
    EndpointHandle epHandle = reinterpret_cast<EndpointHandle>(0x01);
    HcommChannelDesc channelDesc{};
    hcomm::HostCpuRoceChannel channelOnHost{epHandle, channelDesc};
    channelOnHost.SetNicChannelCtx(const_cast<HcommNicChannelOps*>(&g_BuiltinChannelOps), &channelOnHost);
    const ChannelHandle channel = reinterpret_cast<ChannelHandle>(&channelOnHost);
    DevType devType = DevType::DEV_TYPE_950;

    MOCKER(&hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(channelOnHost, &hcomm::HostCpuRoceChannel::ChannelDrain)
        .expects(once())
        .will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(HcommChannelDrainOnThread(0, channel), HCCL_SUCCESS);
}

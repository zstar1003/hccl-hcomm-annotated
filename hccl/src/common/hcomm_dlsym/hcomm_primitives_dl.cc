/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcomm_primitives_dl.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>

DEFINE_WEAK_FUNC(
    int32_t, HcommWriteWithNotifyOnThread, ThreadHandle thread, ChannelHandle channel, void* dst, const void* src,
    uint64_t len, uint32_t remoteNotifyIdx);
DEFINE_WEAK_FUNC(
    int32_t, HcommWriteReduceWithNotifyOnThread, ThreadHandle thread, ChannelHandle channel, void* dst, const void* src,
    uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp, uint32_t remoteNotifyIdx);
DEFINE_WEAK_FUNC(
    int32_t, HcommWriteNbiOnThread, ThreadHandle thread, ChannelHandle channel, void* dst, const void* src,
    uint64_t len);
DEFINE_WEAK_FUNC(int32_t, HcommWriteNbi, ChannelHandle channel, void* dst, const void* src, uint64_t len);
DEFINE_WEAK_FUNC(
    int32_t, HcommWriteWithNotifyNbiOnThread, ThreadHandle thread, ChannelHandle channel, void* dst, const void* src,
    uint64_t len, uint32_t remoteNotifyIdx);
DEFINE_WEAK_FUNC(
    int32_t, HcommWriteWithNotifyNbi, ChannelHandle channel, void* dst, const void* src, uint64_t len,
    uint32_t remoteNotifyIdx);
DEFINE_WEAK_FUNC(
    int32_t, HcommReadNbiOnThread, ThreadHandle thread, ChannelHandle channel, void* dst, const void* src,
    uint64_t len);
DEFINE_WEAK_FUNC(int32_t, HcommReadNbi, ChannelHandle channel, void* dst, const void* src, uint64_t len);
DEFINE_WEAK_FUNC(
    int32_t, HcommReadReduceOnThread, ThreadHandle thread, ChannelHandle channel, void* dst, const void* src,
    uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp);
DEFINE_WEAK_FUNC(int32_t, HcommChannelNotifyRecord, ChannelHandle channel, uint32_t remoteNotifyIdx);
DEFINE_WEAK_FUNC(int32_t, HcommChannelNotifyWait, ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeout);
DEFINE_WEAK_FUNC(
    int32_t, HcommThreadNotifyRecordOnThread, ThreadHandle thread, ThreadHandle dstThread, uint32_t dstNotifyIdx);
DEFINE_WEAK_FUNC(int32_t, HcommThreadNotifyWaitOnThread, ThreadHandle thread, uint32_t notifyIdx, uint32_t timeOut);
DEFINE_WEAK_FUNC(
    int32_t, HcommChannelNotifyRecordOnThread, ThreadHandle thread, ChannelHandle channel, uint32_t remoteNotifyIdx);
DEFINE_WEAK_FUNC(
    int32_t, HcommChannelNotifyWaitOnThread, ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx,
    uint32_t timeOut);
DEFINE_WEAK_FUNC(
    HcclResult, HcommSymWinGetPeerPointer, HcclCommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr);
DEFINE_WEAK_FUNC(int32_t, HcommThreadSynchronize, ThreadHandle thread);
DEFINE_WEAK_FUNC(
    int32_t, HcommSendRequest, uint64_t handle, const char* msgTag, const void* src, size_t sizeByte, uint32_t* msgId);
DEFINE_WEAK_FUNC(int32_t, HcommWaitResponse, uint64_t handle, void* dst, size_t sizeByte, uint32_t* msgId);
DEFINE_WEAK_FUNC(int32_t, HcommFlush);
DEFINE_WEAK_FUNC(int32_t, HcommChannelFence, ChannelHandle channel);
DEFINE_WEAK_FUNC(int32_t, HcommFenceOnThread, ThreadHandle thread);
DEFINE_WEAK_FUNC(int32_t, HcommChannelFenceOnThread, ThreadHandle thread, ChannelHandle channel);
DEFINE_WEAK_FUNC(int32_t, HcommChannelDrainOnThread, ThreadHandle thread, ChannelHandle channel);
DEFINE_WEAK_FUNC(HcclResult, HcommThreadJoin, ThreadHandle thread, uint32_t timeout);
#ifdef HCOMM_TIMEOUT_FLOAT_TYPE
DEFINE_WEAK_FUNC(int32_t, HcommThreadResAcquireTimeOut, float timeOut);
DEFINE_WEAK_FUNC(int32_t, HcommSetNotifyWaitTimeOut, float timeOut);
#else
DEFINE_WEAK_FUNC(int32_t, HcommThreadResAcquireTimeOut, uint32_t timeOut);
DEFINE_WEAK_FUNC(int32_t, HcommSetNotifyWaitTimeOut, uint32_t timeOut);
#endif
DEFINE_WEAK_FUNC(int32_t, HcommThreadNotifyWaitOnThreadWithDefaultTimeout, ThreadHandle thread, uint32_t notifyIdx);
DEFINE_WEAK_FUNC(
    int32_t, HcommChannelNotifyWaitOnThreadWithDefaultTimeout, ThreadHandle thread, ChannelHandle channel,
    uint32_t localNotifyIdx);
DEFINE_WEAK_FUNC(int32_t, HcommChannelNotifyWaitWithDefaultTimeout, ChannelHandle channel, uint32_t localNotifyIdx);
DEFINE_WEAK_FUNC(int32_t, HcommAicpuTsTaskCacheLookup, const char* tag, bool* isHit);
DEFINE_WEAK_FUNC(int32_t, HcommAicpuTsTaskCacheStart, const char* tag, void** addrs, uint64_t* sizes, uint64_t count);
DEFINE_WEAK_FUNC(int32_t, HcommAicpuTsTaskCacheEnd, const char* tag);
DEFINE_WEAK_FUNC(int32_t, HcommAicpuTsTaskCacheExecute, const char* tag, void** addrs, uint64_t* sizes, uint64_t count);
DEFINE_WEAK_FUNC(int32_t, HcommAicpuTsTaskCacheClear, const char* tag);

using HcclHcommBatchTransferOnThreadFunc
    = int32_t (*)(ThreadHandle, ChannelHandle, const HcclHcommBatchTransferDesc*, uint32_t);

static bool g_HcommBatchTransferOnThreadSupported = false;
static HcclHcommBatchTransferOnThreadFunc g_HcommBatchTransferOnThread = nullptr;

extern "C" bool HcommIsSupportHcommBatchTransferOnThread(void) { return g_HcommBatchTransferOnThreadSupported; }

// [中文导读] [AllReduce逐行 S87] 声明HCCL批传输动态适配入口，保持C链接符号。
extern "C" int32_t HcclHcommBatchTransferOnThread(
    // [中文导读] [AllReduce逐行 S88] 接收执行Thread、Channel和只读批描述数组。
    ThreadHandle thread, ChannelHandle channel, const HcclHcommBatchTransferDesc* transferDescs,
    // [中文导读] [AllReduce逐行 S89] 接收描述数，不是数据字节数。
    uint32_t transferDescNum)
// [中文导读] [AllReduce逐行 S90] 进入动态桥函数体。
{
    // [中文导读] [AllReduce逐行 S91] 先检查HCOMM批传输符号是否加载成功。
    if (g_HcommBatchTransferOnThread == nullptr) {
        // [中文导读] [AllReduce逐行 S92] 缺少符号时记录兼容层错误。
        HCCL_COMPAT_ERROR("[HcclWrapper] HcommBatchTransferOnThread not supported");
        // [中文导读] [AllReduce逐行 S93] 缺少批接口返回-1，未执行传输。
        return -1;
    // [中文导读] [AllReduce逐行 S94] 结束符号缺失分支。
    }
    // [中文导读] [AllReduce逐行 S95] 调用加载的HCOMM函数，转发Thread、Channel、描述数组和数量，并返回其结果。
    return g_HcommBatchTransferOnThread(thread, channel, transferDescs, transferDescNum);
// [中文导读] [AllReduce逐行 S96] 结束批传输桥。
}

// ---------- 初始化函数 ----------
void HcommPrimitivesDlInit(void* libHcommHandle)
{
    INIT_SUPPORT_FLAG(libHcommHandle, HcommWriteWithNotifyOnThread);
    INIT_SUPPORT_FLAG(libHcommHandle, HcommWriteReduceWithNotifyOnThread);
    INIT_SUPPORT_FLAG(libHcommHandle, HcommWriteNbiOnThread);
    INIT_SUPPORT_FLAG(libHcommHandle, HcommWriteNbi);
    INIT_SUPPORT_FLAG(libHcommHandle, HcommWriteWithNotifyNbiOnThread);
    INIT_SUPPORT_FLAG(libHcommHandle, HcommWriteWithNotifyNbi);
    INIT_SUPPORT_FLAG(libHcommHandle, HcommReadNbiOnThread);
    INIT_SUPPORT_FLAG(libHcommHandle, HcommReadNbi);
    INIT_SUPPORT_FLAG(libHcommHandle, HcommReadReduceOnThread);
    INIT_SUPPORT_FLAG(libHcommHandle, HcommChannelNotifyRecord);
    INIT_SUPPORT_FLAG(libHcommHandle, HcommChannelNotifyWait);
    INIT_SUPPORT_FLAG(libHcommHandle, HcommThreadNotifyRecordOnThread);
    INIT_SUPPORT_FLAG(libHcommHandle, HcommThreadNotifyWaitOnThread);
    INIT_SUPPORT_FLAG(libHcommHandle, HcommChannelNotifyRecordOnThread);
    INIT_SUPPORT_FLAG(libHcommHandle, HcommChannelNotifyWaitOnThread);
    INIT_SUPPORT_FLAG(libHcommHandle, HcommSymWinGetPeerPointer);
    INIT_SUPPORT_FLAG(libHcommHandle, HcommThreadSynchronize);
    INIT_SUPPORT_FLAG(libHcommHandle, HcommSendRequest);
    INIT_SUPPORT_FLAG(libHcommHandle, HcommWaitResponse);
    INIT_SUPPORT_FLAG(libHcommHandle, HcommFlush);
    INIT_SUPPORT_FLAG(libHcommHandle, HcommChannelFence);
    INIT_SUPPORT_FLAG(libHcommHandle, HcommFenceOnThread);
    INIT_SUPPORT_FLAG(libHcommHandle, HcommChannelFenceOnThread);
    INIT_SUPPORT_FLAG(libHcommHandle, HcommChannelDrainOnThread);
    INIT_SUPPORT_FLAG(libHcommHandle, HcommThreadJoin);

    INIT_SUPPORT_FLAG(libHcommHandle, HcommThreadResAcquireTimeOut);
    INIT_SUPPORT_FLAG(libHcommHandle, HcommSetNotifyWaitTimeOut);
    INIT_SUPPORT_FLAG(libHcommHandle, HcommThreadNotifyWaitOnThreadWithDefaultTimeout);
    INIT_SUPPORT_FLAG(libHcommHandle, HcommChannelNotifyWaitOnThreadWithDefaultTimeout);
    INIT_SUPPORT_FLAG(libHcommHandle, HcommChannelNotifyWaitWithDefaultTimeout);
    INIT_SUPPORT_FLAG(libHcommHandle, HcommAicpuTsTaskCacheLookup);
    INIT_SUPPORT_FLAG(libHcommHandle, HcommAicpuTsTaskCacheStart);
    INIT_SUPPORT_FLAG(libHcommHandle, HcommAicpuTsTaskCacheEnd);
    INIT_SUPPORT_FLAG(libHcommHandle, HcommAicpuTsTaskCacheExecute);
    INIT_SUPPORT_FLAG(libHcommHandle, HcommAicpuTsTaskCacheClear);
    g_HcommBatchTransferOnThread
        = reinterpret_cast<HcclHcommBatchTransferOnThreadFunc>(HcclDlsym(libHcommHandle, "HcommBatchTransferOnThread"));
    if (g_HcommBatchTransferOnThread == nullptr) {
        g_HcommBatchTransferOnThreadSupported = false;
        HCCL_COMPAT_DEBUG("[HcclWrapper] %s not supported", "HcommBatchTransferOnThread");
    } else {
        g_HcommBatchTransferOnThreadSupported = true;
    }
}

// [中文导读] [AllReduce逐行 S147] 声明默认超时能力联合探测函数。
bool IsHcommDefaultTimeoutSupported()
// [中文导读] [AllReduce逐行 S148] 进入能力检查函数体。
{
    // [中文导读] [AllReduce逐行 S149] 只有设置默认等待时间和默认Thread Wait两种符号均支持才返回true。
    return HcommIsSupportHcommSetNotifyWaitTimeOut() && HcommIsSupportHcommThreadNotifyWaitOnThreadWithDefaultTimeout();
// [中文导读] [AllReduce逐行 S150] 结束能力联合检查。
}

// [中文导读] [AllReduce逐行 S152] 声明通知等待时间的HCCL适配接口，timeout沿用下层单位。
HcclResult HcclSetNotifyWaitTimeOut(uint32_t timeout)
// [中文导读] [AllReduce逐行 S153] 进入通知等待时间设置函数体。
{
    // [中文导读] [AllReduce逐行 S154] 先探测动态加载的HcommSetNotifyWaitTimeOut是否支持。
    if (!HcommIsSupportHcommSetNotifyWaitTimeOut()) {
        // [中文导读] [AllReduce逐行 S155] 符号缺失时返回NOT_SUPPORT，不调用空入口。
        return HCCL_E_NOT_SUPPORT;
    // [中文导读] [AllReduce逐行 S156] 结束接口缺失分支。
    }
// [中文导读] [AllReduce逐行 S157] 编译期选择HCOMM使用float超时参数的ABI。
#ifdef HCOMM_TIMEOUT_FLOAT_TYPE
    // [中文导读] [AllReduce逐行 S158] 将timeout转为float调用HcommSetNotifyWaitTimeOut，并转换返回码类型。
    return static_cast<HcclResult>(HcommSetNotifyWaitTimeOut(static_cast<float>(timeout)));
// [中文导读] [AllReduce逐行 S159] 编译期切换到整数参数ABI。
#else
    // [中文导读] [AllReduce逐行 S160] 保持timeout整数类型调用HcommSetNotifyWaitTimeOut，并返回其状态。
    return static_cast<HcclResult>(HcommSetNotifyWaitTimeOut(timeout));
// [中文导读] [AllReduce逐行 S161] 结束超时ABI条件编译。
#endif
// [中文导读] [AllReduce逐行 S162] 结束HcclSetNotifyWaitTimeOut。
}

// [中文导读] [AllReduce逐行 S164] 声明执行流资源申请等待时间的HCCL适配接口，timeout沿用下层单位。
HcclResult HcclThreadResAcquireTimeOut(uint32_t timeout)
// [中文导读] [AllReduce逐行 S165] 进入执行流资源申请等待时间设置函数体。
{
    // [中文导读] [AllReduce逐行 S166] 先探测动态加载的HcommThreadResAcquireTimeOut是否支持。
    if (!HcommIsSupportHcommThreadResAcquireTimeOut()) {
        // [中文导读] [AllReduce逐行 S167] 符号缺失时返回NOT_SUPPORT，不调用空入口。
        return HCCL_E_NOT_SUPPORT;
    // [中文导读] [AllReduce逐行 S168] 结束接口缺失分支。
    }
// [中文导读] [AllReduce逐行 S169] 编译期选择HCOMM使用float超时参数的ABI。
#ifdef HCOMM_TIMEOUT_FLOAT_TYPE
    // [中文导读] [AllReduce逐行 S170] 将timeout转为float调用HcommThreadResAcquireTimeOut，并转换返回码类型。
    return static_cast<HcclResult>(HcommThreadResAcquireTimeOut(static_cast<float>(timeout)));
// [中文导读] [AllReduce逐行 S171] 编译期切换到整数参数ABI。
#else
    // [中文导读] [AllReduce逐行 S172] 保持timeout整数类型调用HcommThreadResAcquireTimeOut，并返回其状态。
    return static_cast<HcclResult>(HcommThreadResAcquireTimeOut(timeout));
// [中文导读] [AllReduce逐行 S173] 结束超时ABI条件编译。
#endif
// [中文导读] [AllReduce逐行 S174] 结束HcclThreadResAcquireTimeOut。
}

// [中文导读] [AllReduce逐行 S176] 声明Thread等待适配入口，接收Thread、通知槽和兼容超时。
HcclResult HcclThreadNotifyWaitOnThreadDefault(ThreadHandle thread, uint32_t notifyIdx, uint32_t fallbackTimeout)
// [中文导读] [AllReduce逐行 S177] 进入Thread等待适配函数体。
{
    // [中文导读] [AllReduce逐行 S178] 要求设置默认超时和默认Thread Wait两项动态能力都支持。
    if (HcommIsSupportHcommSetNotifyWaitTimeOut() && HcommIsSupportHcommThreadNotifyWaitOnThreadWithDefaultTimeout()) {
        // [中文导读] [AllReduce逐行 S179] 走默认超时Thread Wait；通知槽与Record配对，返回排队结果。
        return static_cast<HcclResult>(HcommThreadNotifyWaitOnThreadWithDefaultTimeout(thread, notifyIdx));
    // [中文导读] [AllReduce逐行 S180] 结束默认超时路径。
    }
    // [中文导读] [AllReduce逐行 S181] 能力不足时使用显式fallbackTimeout等待，并返回下层状态。
    return static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(thread, notifyIdx, fallbackTimeout));
// [中文导读] [AllReduce逐行 S182] 结束Thread默认超时适配。
}

// [中文导读] [AllReduce逐行 S184] 声明在Thread上等待Channel通知的HCCL兼容接口。
HcclResult HcclChannelNotifyWaitOnThreadDefault(
    // [中文导读] [AllReduce逐行 S185] 接收Thread、Channel、本地通知槽以及兼容超时。
    ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx, uint32_t fallbackTimeout)
// [中文导读] [AllReduce逐行 S186] 进入Channel等待适配函数体。
{
    // [中文导读] [AllReduce逐行 S187] 同时探测默认超时设置与Channel默认Wait能力。
    if (HcommIsSupportHcommSetNotifyWaitTimeOut() && HcommIsSupportHcommChannelNotifyWaitOnThreadWithDefaultTimeout()) {
        // [中文导读] [AllReduce逐行 S188] 把默认Wait返回码转换为HCCL返回码。
        return static_cast<HcclResult>(
            // [中文导读] [AllReduce逐行 S189] 在给定Thread等待Channel的localNotifyIdx，超时使用先前默认配置。
            HcommChannelNotifyWaitOnThreadWithDefaultTimeout(thread, channel, localNotifyIdx));
    // [中文导读] [AllReduce逐行 S190] 结束默认Channel等待路径。
    }
    // [中文导读] [AllReduce逐行 S191] 能力不齐时将fallbackTimeout显式传给普通Channel Wait并返回状态。
    return static_cast<HcclResult>(HcommChannelNotifyWaitOnThread(thread, channel, localNotifyIdx, fallbackTimeout));
// [中文导读] [AllReduce逐行 S192] 结束Channel默认超时适配。
}

HcclResult HcclChannelNotifyWaitDefault(ChannelHandle channel, uint32_t localNotifyIdx, uint32_t fallbackTimeout)
{
    if (HcommIsSupportHcommSetNotifyWaitTimeOut() && HcommIsSupportHcommChannelNotifyWaitWithDefaultTimeout()) {
        return static_cast<HcclResult>(HcommChannelNotifyWaitWithDefaultTimeout(channel, localNotifyIdx));
    }
    return static_cast<HcclResult>(HcommChannelNotifyWait(channel, localNotifyIdx, fallbackTimeout));
}

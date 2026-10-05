/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include "hccl_api_data.h"
#include "new/hccl_primitive_local.h"
#include "new/hccl_primitive_remote.h"
#include "thread.h"
#include "aicpu_ts_thread.h"
#include "launch_context.h"

#include "ub_transport_lite_impl.h"
#include "device/framework/aicpu_hccl_process.h"
#include "coll_comm_aicpu_mgr.h"
#include "aicpu_indop_env.h"
#include "hcclCommDfxLite.h"
#include "hcclCommProfilingLite.h"
#include "dfx_profiling_handler_lite.h"
#include "hcclCommOp.h"
#include "hcomm_diag.h"
#include "aicpu_ts_primitives_c_adpt.h"
#include "hccl_diag.h"
#include "channel.h"
#include "aicpu_ts_channel_helper.h"
#include "sqe_build_a5.h"
#include "config_plf_log_v2.h"

using Hccl::PLF_DATA_OP;

using namespace hccl;
// [中文导读] 每个调用线程各有提交上下文，记录参与批提交的通信 Thread 和等待配置，避免不同调用线程混用状态。
thread_local LaunchContext g_threadLaunchCtx;

bool IsBatchLaunchMode() { return g_threadLaunchCtx.IsBatchLaunchMode(); }

uint32_t GetSqFullTimeOut() { return g_threadLaunchCtx.GetSqFullTimeOut(); }

inline bool GetProfilingEnable()
{
    return Hccl::DfxProfilingHandlerLite::GetInstance().GetProfL0State()
           || Hccl::DfxProfilingHandlerLite::GetInstance().GetProfL1State();
}

void AddThread(ThreadHandle thread) { g_threadLaunchCtx.AddThread(thread); }

HcclResult HandleDispatchAllStreams() { return g_threadLaunchCtx.HandleDispatchAllStreams(); }

bool IsSupportReduce(HcommDataType dataType, HcommReduceOp op)
{
    bool checkDataType
        = (dataType == HCOMM_DATA_TYPE_FP32 || dataType == HCOMM_DATA_TYPE_FP16 || dataType == HCOMM_DATA_TYPE_INT8
           || dataType == HCOMM_DATA_TYPE_INT16 || dataType == HCOMM_DATA_TYPE_INT32
           || dataType == HCOMM_DATA_TYPE_BFP16);
    bool checkReduceType = (op == HCOMM_REDUCE_SUM || op == HCOMM_REDUCE_MAX || op == HCOMM_REDUCE_MIN);
    return checkDataType && checkReduceType;
}

HcclResult HcommThreadGetNotifyId(ThreadHandle thread, uint32_t notifyIdx, uint32_t* notifyId)
{
    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);
    LocalNotify* const notifyPtr = threadPtr->GetNotify(notifyIdx);
    CHK_PTR_NULL(notifyPtr);
    *notifyId = notifyPtr->notifyId_;

    return HCCL_SUCCESS;
}

namespace {
// 刷新SQE profiling置位开关：L1开启且设备为960(A6)时推送1，否则推送0。
// 须先于HcclDfxRegOpInfoByCommId全部提前return执行，否则开关从开到关后无法刷回0，
// SQE将永久错误置位，故收敛在唯一入口先行调用。
void RefreshSqeProfilingState()
{
    bool isSqeProfEnabled = false;
    if (Hccl::DfxProfilingHandlerLite::GetInstance().GetProfL1State()) {
        DevType devType = DevType::DEV_TYPE_COUNT;
        (void)hrtGetDeviceType(devType);
        isSqeProfEnabled = (devType == DevType::DEV_TYPE_960);
    }
    Hccl::SetSqeProfilingEnabled(isSqeProfEnabled);
}

HcclResult HcclDfxRegOpInfoByCommIdImpl(char* commId, void* hcclDfxOpInfo);
} // namespace

HcclResult HcclDfxRegOpInfoByCommId(char* commId, void* hcclDfxOpInfo)
{
    RefreshSqeProfilingState();

    return HcclDfxRegOpInfoByCommIdImpl(commId, hcclDfxOpInfo);
}

namespace {
HcclResult HcclDfxRegOpInfoByCommIdImpl(char* commId, void* hcclDfxOpInfo)
{
    if (!GetProfilingEnable() && !hcomm::GetTaskExceptionEnable()) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(commId);
    CHK_PTR_NULL(hcclDfxOpInfo);

    DevType deviceType;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (deviceType == DevType::DEV_TYPE_910B) {
        HCCL_INFO("[%s] is not supported, commId[%s], devType[%d]", __func__, commId, deviceType);
        return HCCL_SUCCESS;
    }

    HcclDfxOpInfo* aicpuDfxInfo = ReinterpretAs<HcclDfxOpInfo*>(hcclDfxOpInfo);
    CHK_RET(HcommThreadGetNotifyId(
        aicpuDfxInfo->cpuTsThread, aicpuDfxInfo->cpuWaitAicpuNotifyIdx, &aicpuDfxInfo->cpuWaitAicpuNotifyId));
    CollCommAicpu* currentComm = CollCommAicpuMgr::GetInstance().GetCurrentComm();
    CHK_PTR_NULL(currentComm);
    CHK_RET(currentComm->InitDfxOpInfo(aicpuDfxInfo));

    return HCCL_SUCCESS;
}
} // namespace

// [中文导读] 数据搬运原语：在给定Thread排入本地src→dst的len字节拷贝，不需要跨Rank通道。
// [中文导读] AddThread登记本次提交涉及的Thread；A5分支交Thread::LocalCopy，旧分支使用Stream适配接口。
int32_t HcommLocalCopyOnThread(ThreadHandle thread, void* dst, const void* src, uint64_t len)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP, "[%s] thread[0x%llx], dst[0x%llx], src[0x%llx], len[%llu].", __func__, thread, dst, src, len);

    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);
    // [中文导读] 批量模式下登记本次执行 Thread，EAGER 下登记函数直接返回；后续批提交汇总已登记的执行序列。
    AddThread(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    HcclResult ret = HCCL_SUCCESS;
    if (threadPtr->IsDeviceA5()) {
        EXCEPTION_CATCH(ret = threadPtr->LocalCopy(dst, src, len), ret = HCCL_E_INTERNAL);
    } else {
        // [中文导读] 兼容实现用 len 字节构造源/目标描述并取得对应 Stream，把本地拷贝交给旧原语适配层。
        HcclBuf srcBuf{const_cast<void*>(src), len, nullptr};
        HcclBuf dstBuf{dst, len, nullptr};
        Stream* stream = GetStream(thread);
        CHK_PTR_NULL(stream);
        ret = HcclLocalCopy(stream, &dstBuf, &srcBuf);
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s] FAIL. thread[0x%llx], dst[0x%llx], src[0x%llx], len[%llu].", __func__, thread, dst, src, len),
        ret);
    return HCCL_SUCCESS;
}

// [中文导读] 数据计算原语：按dataType/reduceOp将源数据归约到目标；count是元素数，内部换算为字节len。
// [中文导读] 这是供算法组合的局部能力，不是一次完整AllReduce；调用者仍负责Rank间通信及依赖关系。
int32_t HcommLocalReduceOnThread(
    ThreadHandle thread, void* dst, const void* src, uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP, "[%s] thread[0x%llx], dst[0x%llx], src[0x%llx], count[%llu], dataType[%d], reduceOp[%d].",
        __func__, thread, dst, src, count, dataType, reduceOp);

    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);
    AddThread(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    // [中文导读] 按元素类型大小把 count 换算为字节范围，底层本地归约接收该范围与类型/操作。
    uint64_t len = count * SIZE_TABLE[dataType];

    HcclResult ret = HCCL_SUCCESS;
    if (threadPtr->IsDeviceA5()) {
        EXCEPTION_CATCH(ret = threadPtr->LocalReduce(dst, src, len, dataType, reduceOp), ret = HCCL_E_INTERNAL);
    } else {
        // [中文导读] 旧设备分支限制其实际支持的类型和归约操作，再转换为 HcclReduceInfo 调用兼容原语。
        CHK_PRT_RET(
            (IsSupportReduce(dataType, reduceOp) == false),
            HCCL_ERROR(
                "[%s] Not support reduce, "
                "dst[0x%llx], src[0x%llx], count[%llu], dataType[%d], reduceOp[%d]",
                __func__, dst, src, count, dataType, reduceOp),
            HCCL_E_PARA);
        HcclBuf srcBuf{const_cast<void*>(src), len, nullptr};
        HcclBuf dstBuf{dst, len, nullptr};
        HcclReduceInfo reduceInfo{static_cast<HcclDataType>(dataType), static_cast<HcclReduceOp>(reduceOp)};
        Stream* stream = GetStream(thread);
        CHK_PTR_NULL(stream);
        ret = HcclLocalCopyReduce(stream, &dstBuf, &srcBuf, reduceInfo);
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] FAIL. thread[0x%llx], dst[0x%llx], src[0x%llx], count[%llu], dataType[%d], reduceOp[%d].", __func__,
            thread, dst, src, count, dataType, reduceOp),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

// [中文导读] 本地Thread间同步：在thread的执行序列上，向dstThread拥有的dstNotifyIdx通知槽发信号。
// [中文导读] A5分支先将通知槽索引解析为notifyId，再交LocalNotifyRecord；索引和硬件ID不是同一参数。
int32_t HcommThreadNotifyRecordOnThread(ThreadHandle thread, ThreadHandle dstThread, uint32_t dstNotifyIdx)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP, "[%s] thread[0x%llx], dstThread[0x%llx], dstNotifyIdx[%u].", __func__, thread, dstThread,
        dstNotifyIdx);

    AddThread(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);
    Thread* const dstThreadPtr = ReinterpretAs<Thread*>(dstThread);
    CHK_PTR_NULL(dstThreadPtr);

    HcclResult ret = HCCL_SUCCESS;
    if (threadPtr->IsDeviceA5()) {
        // [中文导读] 通知资源属于目标 dstThread；读取它的槽位 ID，信号任务却排入发送者 thread 的执行序列。
        LocalNotify* const notifyPtr = dstThreadPtr->GetNotify(dstNotifyIdx);
        CHK_PTR_NULL(notifyPtr);
        const uint32_t notifyId = notifyPtr->notifyId_;
        EXCEPTION_CATCH(ret = threadPtr->LocalNotifyRecord(notifyId), ret = HCCL_E_INTERNAL);
    } else {
        Stream* stream = GetStream(thread);
        CHK_PTR_NULL(stream);
        LocalNotify* notify = GetNotify(dstThread, dstNotifyIdx);
        CHK_PTR_NULL(notify);
        ret = HcclLocalNotifyRecord(stream, notify);
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] FAIL. thread[0x%llx], dstThread[0x%llx], dstNotifyIdx[%u].", __func__, thread, dstThread,
            dstNotifyIdx),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

// [中文导读] 在本Thread排入等待自己notifyIdx槽位的任务，timeOut以秒表示；与另一Thread的Record配对。
int32_t HcommThreadNotifyWaitOnThread(ThreadHandle thread, uint32_t notifyIdx, uint32_t timeOut)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP, "[%s] thread[0x%llx], notifyIdx[%u], timeOut[%u s].", __func__, thread, notifyIdx, timeOut);

    AddThread(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    HcclResult ret = HCCL_SUCCESS;
    if (threadPtr->IsDeviceA5()) {
        // [中文导读] 等待资源来自当前 Thread 自己的槽位，将它转换为硬件通知 ID 后按 timeOut 秒建立等待。
        LocalNotify* const notifyPtr = threadPtr->GetNotify(notifyIdx);
        CHK_PTR_NULL(notifyPtr);
        const uint32_t notifyId = notifyPtr->notifyId_;
        EXCEPTION_CATCH(ret = threadPtr->LocalNotifyWait(notifyId, timeOut), ret = HCCL_E_INTERNAL);
    } else {
        Stream* stream = GetStream(thread);
        CHK_PTR_NULL(stream);
        LocalNotify* notify = GetNotify(thread, notifyIdx);
        CHK_PTR_NULL(notify);
        ret = HcclLocalNotifyWait(stream, notify, timeOut);
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s] FAIL. thread[0x%llx], notifyIdx[%u], timeOut[%u s].", __func__, thread, notifyIdx, timeOut),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommAclrtNotifyRecordOnThread(ThreadHandle thread, uint64_t dstNotifyId)
{
    PLF_CONFIG_INFO(PLF_DATA_OP, "[%s] thread[0x%llx], dstNotifyId[%llu].", __func__, thread, dstNotifyId);

    // [中文导读] 这个接口直接接收运行时 notifyId；不像 ThreadNotifyRecord 那样先从目标 Thread 的索引查槽位。
    AddThread(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    HcclResult ret = HCCL_SUCCESS;
    if (threadPtr->IsDeviceA5()) {
        EXCEPTION_CATCH(ret = threadPtr->LocalNotifyRecord(dstNotifyId), ret = HCCL_E_INTERNAL);
    } else {
        Stream* stream = GetStream(thread);
        CHK_PTR_NULL(stream);
        ret = HcclLocalBareNotifyRecord(stream, dstNotifyId);
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[%s] FAIL. thread[0x%llx], dstNotifyId[%llu].", __func__, thread, dstNotifyId),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommAclrtNotifyWaitOnThread(ThreadHandle thread, uint64_t notifyId, uint32_t timeOut)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP, "[%s] thread[0x%llx], notifyId[%llu], timeOut[%u s].", __func__, thread, notifyId, timeOut);

    AddThread(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    HcclResult ret = HCCL_SUCCESS;
    if (threadPtr->IsDeviceA5()) {
        EXCEPTION_CATCH(ret = threadPtr->LocalNotifyWait(notifyId, timeOut), ret = HCCL_E_INTERNAL);
    } else {
        Stream* stream = GetStream(thread);
        CHK_PTR_NULL(stream);
        ret = HcclLocalBareNotifyWait(stream, notifyId, timeOut);
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s] FAIL. thread[0x%llx], notifyId[%llu], timeOut[%u s].", __func__, thread, notifyId, timeOut),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

HcclResult CommTaskPrepare(char* key, uint32_t keyLen) // host ffts+使用
{
    // [中文导读] 有效 key 按 keyLen 字节构造缓存标识；未提供有效 key 时使用临时标识进入任务准备。
    std::string keyStr = "temp_key";
    if (key != nullptr && keyLen != 0) {
        keyStr = std::string(key, keyLen);
        HCCL_DEBUG("[CommTaskPrepare]key[%s], keyLen[%u]", key, keyLen);
    } else {
        HCCL_DEBUG("[CommTaskPrepare]disable cache, key[0x%llx], keyLen[%u]", key, keyLen);
    }

    return HcclTaskPrepare(const_cast<char_t*>(keyStr.c_str()), keyStr.length());
}

HcclResult CommTaskLaunch(ThreadHandle* threads, uint32_t threadNum) // host ffts+或aicpu stars使用"
{
    CHK_PTR_NULL(threads);
    CHK_PRT_RET(threadNum < 1, HCCL_ERROR("[CommTaskLaunch]threadNum is less than 1"), HCCL_E_PARA);

    Thread* threadPtr = ReinterpretAs<Thread*>(threads[0]);
    CHK_PTR_NULL(threadPtr);

    if (threadPtr->IsDeviceA5()) {
        HCCL_INFO("[%s] Running on A5.", __func__);
        // [中文导读] A5 逐执行 Thread 调用 LaunchTask，任一异常转成 INTERNAL；成功表示任务提交接口完成。
        for (uint32_t i = 0; i < threadNum; i++) {
            Thread* threadPtrLoop = ReinterpretAs<Thread*>(threads[i]);
            CHK_PTR_NULL(threadPtrLoop);
            HCCL_INFO("[%s] Launching task in thread[0x%llx].", __func__, threads[i]);
            EXCEPTION_CATCH(threadPtrLoop->LaunchTask(), return HCCL_E_INTERNAL);
        }
        return HCCL_SUCCESS;
    }

    // [中文导读] 兼容实现把 Thread 列表转换为 Stream 数组，再以相同 threadNum 交给统一任务发射接口。
    std::vector<hccl::Stream> streams;
    for (uint32_t i = 0; i < threadNum; i++) {
        hccl::Stream* stream = GetStream(threads[i]);
        CHK_PTR_NULL(stream);
        streams.push_back(*stream);
    }

    return HcclTaskLaunch(streams.data(), threadNum);
}

HcclResult DispatchAllStreams(const ThreadHandle* threads, uint32_t threadNum)
{
    CHK_PTR_NULL(threads);
    CHK_PRT_RET(threadNum < 1, HCCL_ERROR("[DispatchAllStreams]threadNum is less than 1"), HCCL_E_PARA);

    Thread* threadPtr = ReinterpretAs<Thread*>(threads[0]);
    CHK_PTR_NULL(threadPtr);

    if (!threadPtr->IsDeviceA5()) {
        HCCL_ERROR("[%s] DispatchAllStreams is only supported on A5 device.", __func__);
        return HCCL_E_NOT_SUPPORT;
    }

    // [中文导读] 仅 A5 接受此批发射接口，逐 Thread 尝试提交已组织的任务，不在这里等待全部任务执行结束。
    for (uint32_t i = 0; i < threadNum; i++) {
        Thread* threadPtrLoop = ReinterpretAs<Thread*>(threads[i]);
        CHK_PTR_NULL(threadPtrLoop);
        EXCEPTION_CATCH(threadPtrLoop->TryLaunchTask(), return HCCL_E_INTERNAL);
    }
    return HCCL_SUCCESS;
}

namespace {
// Convert hccl::HcommDataType => Hccl::DataType, hccl::HcommReduceOp => Hccl::ReduceOp

std::unordered_map<HcommDataType, Hccl::DataType> mapHcommDataTypeToA5
    = {{HcommDataType::HCOMM_DATA_TYPE_INT8, Hccl::DataType::INT8},
       {HcommDataType::HCOMM_DATA_TYPE_INT16, Hccl::DataType::INT16},
       {HcommDataType::HCOMM_DATA_TYPE_INT32, Hccl::DataType::INT32},
       {HcommDataType::HCOMM_DATA_TYPE_FP16, Hccl::DataType::FP16},
       {HcommDataType::HCOMM_DATA_TYPE_FP32, Hccl::DataType::FP32},
       {HcommDataType::HCOMM_DATA_TYPE_INT64, Hccl::DataType::INT64},
       {HcommDataType::HCOMM_DATA_TYPE_UINT64, Hccl::DataType::UINT64},
       {HcommDataType::HCOMM_DATA_TYPE_UINT8, Hccl::DataType::UINT8},
       {HcommDataType::HCOMM_DATA_TYPE_UINT16, Hccl::DataType::UINT16},
       {HcommDataType::HCOMM_DATA_TYPE_UINT32, Hccl::DataType::UINT32},
       {HcommDataType::HCOMM_DATA_TYPE_FP64, Hccl::DataType::FP64},
       {HcommDataType::HCOMM_DATA_TYPE_BFP16, Hccl::DataType::BFP16},
       {HcommDataType::HCOMM_DATA_TYPE_INT128, Hccl::DataType::INT128},
#ifndef OPEN_BUILD_PROJECT
       {HcommDataType::HCOMM_DATA_TYPE_HIF8, Hccl::DataType::HIF8},
       {HcommDataType::HCOMM_DATA_TYPE_FP8E4M3, Hccl::DataType::FP8E4M3},
       {HcommDataType::HCOMM_DATA_TYPE_FP8E5M2, Hccl::DataType::FP8E5M2},
       {HcommDataType::HCOMM_DATA_TYPE_FP8E8M0, Hccl::DataType::FP8E8M0}
#endif
};

std::unordered_map<HcommReduceOp, Hccl::ReduceOp> mapHcommReduceOpToA5
    = {{HcommReduceOp::HCOMM_REDUCE_SUM, Hccl::ReduceOp::SUM},
       {HcommReduceOp::HCOMM_REDUCE_PROD, Hccl::ReduceOp::PROD},
       {HcommReduceOp::HCOMM_REDUCE_MAX, Hccl::ReduceOp::MAX},
       {HcommReduceOp::HCOMM_REDUCE_MIN, Hccl::ReduceOp::MIN}};

inline HcclResult CheckDataTypeAndReduceOp(HcommDataType dataType, HcommReduceOp reduceOp)
{
    if (mapHcommDataTypeToA5.find(dataType) == mapHcommDataTypeToA5.end()) {
        HCCL_ERROR("[%s] type[%u] is not supported.", __func__, dataType);
        return HCCL_E_PARA;
    }

    if (mapHcommReduceOpToA5.find(reduceOp) == mapHcommReduceOpToA5.end()) {
        HCCL_ERROR("[%s] op[%u] is not supported.", __func__, reduceOp);
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}

} // namespace

// 设置notify wait的等待超时时间，默认单位为秒
int32_t HcommSetNotifyWaitTimeOut(float timeOut)
{
    if (std::isnan(timeOut) || timeOut < 0.0f || timeOut > static_cast<float>(UINT32_MAX)) {
        HCCL_ERROR("[%s] in aicpu_ts timeOut[%f s] is invalid.", __func__, timeOut);
        return HCCL_E_PARA;
    }
    // [中文导读] 浮点秒数校验后转换为整数秒，小数部分被截去；新默认值存入当前调用线程的提交上下文。
    uint32_t timeOutInt = static_cast<uint32_t>(timeOut);
    HCCL_INFO("[%s] START in aicpu_ts. timeOut[%u s].", __func__, timeOutInt);
    return g_threadLaunchCtx.SetNotifyWaitTimeOut(timeOutInt);
}

int32_t HcommThreadResAcquireTimeOut(float timeOut)
{
    if (std::isnan(timeOut) || timeOut < 0.0f || timeOut > static_cast<float>(UINT32_MAX)) {
        HCCL_ERROR("[%s] in aicpu_ts timeOut[%f s] is invalid.", __func__, timeOut);
        return HCCL_E_PARA;
    }
    uint32_t timeOutInt = static_cast<uint32_t>(timeOut);
    HCCL_INFO("[%s] START in aicpu_ts. timeOut[%u s].", __func__, timeOutInt);
    // [中文导读] 这里设置 SQ 满时资源获取的等待期限，与 NotifyWait 的默认超时使用不同上下文字段。
    return g_threadLaunchCtx.SetSqFullTimeOut(timeOutInt);
}

int32_t
HcommChannelNotifyWaitOnThreadWithDefaultTimeout(ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx)
{
    HCCL_INFO(
        "[%s] START. thread[0x%llx], channel[0x%llx], localNotifyIdx[%u].", __func__, thread, channel, localNotifyIdx);

    uint32_t notifyWaitTimeOut;
    g_threadLaunchCtx.GetNotifyWaitTimeOut(notifyWaitTimeOut);

    HCCL_DEBUG("[%s] Using default timeout: %u s", __func__, notifyWaitTimeOut);

    int32_t ret = HcommChannelNotifyWaitOnThread(thread, channel, localNotifyIdx, notifyWaitTimeOut);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[%s] HcommChannelNotifyWaitOnThread FAILED. thread[0x%llx], channel[0x%llx], localNotifyIdx[%u], ret[%d]",
            __func__, thread, channel, localNotifyIdx, ret);
        return ret;
    }

    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommThreadNotifyWaitOnThreadWithDefaultTimeout(ThreadHandle thread, uint32_t notifyIdx)
{
    HCCL_INFO("[%s] START. thread[0x%llx], notifyIdx[%u].", __func__, thread, notifyIdx);

    uint32_t notifyWaitTimeOut;
    g_threadLaunchCtx.GetNotifyWaitTimeOut(notifyWaitTimeOut);

    HCCL_DEBUG("[%s] Using default timeout: %u s", __func__, notifyWaitTimeOut);

    int32_t ret = HcommThreadNotifyWaitOnThread(thread, notifyIdx, notifyWaitTimeOut);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[%s] HcommThreadNotifyWaitOnThread FAILED. thread[0x%llx], notifyIdx[%u], ret[%d]", __func__, thread,
            notifyIdx, ret);
        return ret;
    }

    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

// [中文导读] 跨Rank写：src为本地地址，dst为远端地址，len是字节数；Channel必须已准备好访问所需资源。
// [中文导读] A5路径解包通道、构造本地RMA描述，再经BaseTransportLiteImpl::Write在StreamLite上组织传输。
// [中文导读] 本函数不包含上层ACK/DATA_SIGNAL协议；不能以一次Write返回代替接收端可消费数据的同步。
int32_t HcommWriteOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP, "[%s] thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu].", __func__, thread,
        channel, dst, src, len);

    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);
    // [中文导读] 先解包用户通道句柄，再按 Thread 所属设备选择底层传输对象，避免直接把包装句柄当对象地址。
    CHK_RET(UnwrapChannelHandle(channel));
    AddThread(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    HcclResult ret = HCCL_SUCCESS;
    if (threadPtr->IsDeviceA5()) {
        auto* const transportLitePtr = ReinterpretAs<Hccl::BaseTransportLiteImpl*>(channel);
        CHK_PTR_NULL(transportLitePtr);
        auto* const streamLitePtr = static_cast<Hccl::StreamLite*>(threadPtr->GetStreamLitePtr());
        CHK_PTR_NULL(streamLitePtr);

        Hccl::RmaBufferLite locRmaBuf;
        // [中文导读] 为本地源范围构造 RMA 描述；构造失败立即返回，不提交后面的远端写任务。
        ret = transportLitePtr->BuildLocRmaBufferLite(ReinterpretAs<uintptr_t>(src), len, locRmaBuf);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s] FAIL at BuildLocRmaBufferLite. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], "
                "len[%llu].",
                __func__, thread, channel, dst, src, len),
            ret);
        // [中文导读] 远端目标只用地址和字节长度描述，和本地已解析的 RMA 源一起交给当前 StreamLite。
        const Hccl::Buffer rmtBuf{ReinterpretAs<uintptr_t>(dst), len};

        EXCEPTION_CATCH(transportLitePtr->Write(locRmaBuf, rmtBuf, *streamLitePtr), ret = HCCL_E_INTERNAL);
    } else {
        HcclBuf locBuf{const_cast<void*>(src), len, nullptr};
        HcclBuf rmtBuf{dst, len, nullptr};

        Stream* stream = GetStream(thread);
        CHK_PTR_NULL(stream);

        ret = HcclRemoteWrite(stream, ReinterpretAs<void*>(channel), &rmtBuf, &locBuf);
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] FAIL. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu].", __func__, thread,
            channel, dst, src, len),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommWriteReduceOnThread(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t count, HcommDataType dataType,
    HcommReduceOp reduceOp)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP,
        "[%s] thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], count[%llu], dataType[%d], reduceOp[%d].",
        __func__, thread, channel, dst, src, count, dataType, reduceOp);

    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);
    CHK_RET(UnwrapChannelHandle(channel));
    AddThread(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    HcclResult ret = HCCL_SUCCESS;
    if (threadPtr->IsDeviceA5()) {
        // [中文导读] A5 检查类型/操作映射，兼容分支检查旧归约能力；验证通过后才能计算范围并构造归约参数。
        ret = CheckDataTypeAndReduceOp(dataType, reduceOp);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s] FAIL at CheckDataTypeAndReduceOp. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], "
                "count[%llu], dataType[%d], reduceOp[%d].",
                __func__, thread, channel, dst, src, count, dataType, reduceOp),
            ret);
    } else {
        CHK_PRT_RET(
            (IsSupportReduce(dataType, reduceOp) == false),
            HCCL_ERROR(
                "[%s] Not support reduce, "
                "dst[0x%llx], src[0x%llx], count[%llu], dataType[%d], reduceOp[%d]",
                __func__, dst, src, count, dataType, reduceOp),
            HCCL_E_PARA);
    }
    uint64_t len = count * SIZE_TABLE[dataType];

    if (threadPtr->IsDeviceA5()) {
        auto* const transportLitePtr = ReinterpretAs<Hccl::BaseTransportLiteImpl*>(channel);
        CHK_PTR_NULL(transportLitePtr);
        auto* const streamLitePtr = static_cast<Hccl::StreamLite*>(threadPtr->GetStreamLitePtr());
        CHK_PTR_NULL(streamLitePtr);

        Hccl::RmaBufferLite locRmaBuf;
        ret = transportLitePtr->BuildLocRmaBufferLite(ReinterpretAs<uintptr_t>(src), len, locRmaBuf);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s] FAIL at BuildLocRmaBufferLite. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], "
                "count[%llu], dataType[%d], reduceOp[%d].",
                __func__, thread, channel, dst, src, count, dataType, reduceOp),
            ret);
        const Hccl::Buffer rmtBuf{ReinterpretAs<uintptr_t>(dst), len};

        // [中文导读] 把公开枚举转换为 A5 底层枚举，描述向远端目标执行的归约操作。
        Hccl::ReduceIn reduceIn{mapHcommDataTypeToA5.at(dataType), mapHcommReduceOpToA5.at(reduceOp)};

        EXCEPTION_CATCH(
            transportLitePtr->WriteReduce(locRmaBuf, rmtBuf, reduceIn, *streamLitePtr), ret = HCCL_E_INTERNAL);
    } else {
        HcclBuf locBuf{const_cast<void*>(src), len, nullptr};
        HcclBuf rmtBuf{dst, len, nullptr};
        HcclReduceInfo reduceInfo{static_cast<HcclDataType>(dataType), static_cast<HcclReduceOp>(reduceOp)};

        Stream* stream = GetStream(thread);
        CHK_PTR_NULL(stream);

        ret = HcclRemoteWriteReduce(stream, ReinterpretAs<void*>(channel), &rmtBuf, &locBuf, reduceInfo);
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] FAIL. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], count[%llu], dataType[%d], "
            "reduceOp[%d].",
            __func__, thread, channel, dst, src, count, dataType, reduceOp),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

HcclResult CommWriteReduceWithNotify(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t count, HcommDataType dataType,
    HcommReduceOp reduceOp, uint32_t remoteNotifyIdx)
{
    CHK_PTR_NULL(src);
    CHK_PTR_NULL(dst);
    CHK_RET(UnwrapChannelHandle(channel));
    AddThread(thread);
    CHK_PRT_RET(
        (IsSupportReduce(dataType, reduceOp) == false),
        HCCL_ERROR(
            "[%s] Not support reduce, "
            "dst[0x%llx], src[0x%llx], count[%llu], dataType[%d], reduceOp[%d]",
            __func__, dst, src, count, dataType, reduceOp),
        HCCL_E_PARA);
    HcclBuf locBuf{const_cast<void*>(src), count * SIZE_TABLE[dataType], nullptr};
    HcclBuf rmtBuf{dst, count * SIZE_TABLE[dataType], nullptr};
    HcclReduceInfo reduceInfo{static_cast<HcclDataType>(dataType), static_cast<HcclReduceOp>(reduceOp)};

    Stream* stream = GetStream(thread);
    CHK_PTR_NULL(stream);

    return HcclRemoteWriteReduceWithNotify(
        stream, ReinterpretAs<void*>(channel), &rmtBuf, &locBuf, reduceInfo, remoteNotifyIdx);
}

int32_t HcommWriteWithNotifyOnThread(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx)
{
    PLF_CONFIG_INFO(
        PLF_DATA_OP, "[%s] thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu], remoteNotifyIdx[%u].",
        __func__, thread, channel, dst, src, len, remoteNotifyIdx);

    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);
    CHK_RET(UnwrapChannelHandle(channel));
    AddThread(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    HcclResult ret = HCCL_SUCCESS;
    if (threadPtr->IsDeviceA5()) {
        HCCL_DEBUG("[%s] Running on A5.", __func__);
        auto* const transportLitePtr = ReinterpretAs<Hccl::BaseTransportLiteImpl*>(channel);
        CHK_PTR_NULL(transportLitePtr);
        auto* const streamLitePtr = static_cast<Hccl::StreamLite*>(threadPtr->GetStreamLitePtr());
        CHK_PTR_NULL(streamLitePtr);

        Hccl::RmaBufferLite locRmaBuf;
        ret = transportLitePtr->BuildLocRmaBufferLite(ReinterpretAs<uintptr_t>(src), len, locRmaBuf);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s] FAIL at BuildLocRmaBufferLite. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], "
                "len[%llu], remoteNotifyIdx[%u].",
                __func__, thread, channel, dst, src, len, remoteNotifyIdx),
            ret);
        const Hccl::Buffer rmtBuf{ReinterpretAs<uintptr_t>(dst), len};

        // [中文导读] remoteNotifyIdx 选择远端普通通知槽，把写入与通知描述交给同一个 WriteWithNotify 操作。
        Hccl::WithNotifyIn withNotify{Hccl::TransportNotifyType::NORMAL, remoteNotifyIdx};

        EXCEPTION_CATCH(
            transportLitePtr->WriteWithNotify(locRmaBuf, rmtBuf, withNotify, *streamLitePtr), ret = HCCL_E_INTERNAL);
    } else {
        HcclBuf locBuf{const_cast<void*>(src), len, nullptr};
        HcclBuf rmtBuf{dst, len, nullptr};

        Stream* stream = GetStream(thread);
        CHK_PTR_NULL(stream);

        ret = HcclRemoteWriteWithNotify(stream, ReinterpretAs<void*>(channel), &rmtBuf, &locBuf, remoteNotifyIdx);
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] FAIL. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu], remoteNotifyIdx[%u].",
            __func__, thread, channel, dst, src, len, remoteNotifyIdx),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommWriteReduceWithNotifyOnThread(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t count, HcommDataType dataType,
    HcommReduceOp reduceOp, uint32_t remoteNotifyIdx)
{
    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);
    CHK_RET(UnwrapChannelHandle(channel));

    PLF_CONFIG_INFO(
        PLF_DATA_OP,
        "[%s] thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], count[%llu], dataType[%d], reduceOp[%d], "
        "remoteNotifyIdx[%u].",
        __func__, thread, channel, dst, src, count, dataType, reduceOp, remoteNotifyIdx);

    AddThread(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    uint64_t len = count * SIZE_TABLE[dataType];

    HcclResult ret = HCCL_SUCCESS;
    if (threadPtr->IsDeviceA5()) {
        HCCL_DEBUG("[%s] Running on A5.", __func__);
        auto* const transportLitePtr = ReinterpretAs<Hccl::BaseTransportLiteImpl*>(channel);
        CHK_PTR_NULL(transportLitePtr);
        auto* const streamLitePtr = static_cast<Hccl::StreamLite*>(threadPtr->GetStreamLitePtr());
        CHK_PTR_NULL(streamLitePtr);

        Hccl::RmaBufferLite locRmaBuf;
        ret = transportLitePtr->BuildLocRmaBufferLite(ReinterpretAs<uintptr_t>(src), len, locRmaBuf);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s] FAIL at BuildLocRmaBufferLite. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], "
                "count[%llu], dataType[%d], reduceOp[%d], remoteNotifyIdx[%u].",
                __func__, thread, channel, dst, src, count, dataType, reduceOp, remoteNotifyIdx),
            ret);
        const Hccl::Buffer rmtBuf{ReinterpretAs<uintptr_t>(dst), len};

        ret = CheckDataTypeAndReduceOp(dataType, reduceOp);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s] FAIL at CheckDataTypeAndReduceOp. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], "
                "count[%llu], dataType[%d], reduceOp[%d], remoteNotifyIdx[%u].",
                __func__, thread, channel, dst, src, count, dataType, reduceOp, remoteNotifyIdx),
            ret);
        Hccl::ReduceIn reduceIn{mapHcommDataTypeToA5.at(dataType), mapHcommReduceOpToA5.at(reduceOp)};

        // [中文导读] 向底层一次传入数据范围、归约方式和远端通知槽；非 A5 分支明确返回 NOT_SUPPORT。
        Hccl::WithNotifyIn withNotify{Hccl::TransportNotifyType::NORMAL, remoteNotifyIdx};

        EXCEPTION_CATCH(
            transportLitePtr->WriteReduceWithNotify(locRmaBuf, rmtBuf, reduceIn, withNotify, *streamLitePtr),
            ret = HCCL_E_INTERNAL);
    } else {
        ret = HCCL_E_NOT_SUPPORT;
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] FAIL. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], count[%llu], dataType[%d], "
            "reduceOp[%d], remoteNotifyIdx[%u].",
            __func__, thread, channel, dst, src, count, dataType, reduceOp, remoteNotifyIdx),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

// [中文导读] 跨Rank读：src为远端地址，dst为本地地址，len为字节数；方向与Write的本地/远端角色相反。
// [中文导读] A5分支给本地dst构造RMA描述，远端src包装为Buffer，再交BaseTransportLiteImpl::Read组织读取。
int32_t HcommReadOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len)
{
    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);
    CHK_RET(UnwrapChannelHandle(channel));

    PLF_CONFIG_INFO(
        PLF_DATA_OP, "[%s] thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu].", __func__, thread,
        channel, dst, src, len);

    AddThread(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    HcclResult ret = HCCL_SUCCESS;
    if (threadPtr->IsDeviceA5()) {
        auto* const transportLitePtr = ReinterpretAs<Hccl::BaseTransportLiteImpl*>(channel);
        CHK_PTR_NULL(transportLitePtr);
        auto* const streamLitePtr = static_cast<Hccl::StreamLite*>(threadPtr->GetStreamLitePtr());
        CHK_PTR_NULL(streamLitePtr);

        Hccl::RmaBufferLite locRmaBuf;
        // [中文导读] 读操作注册描述取本地目标 dst；远端源 src 只提供地址/长度，与 Write 的本地源角色不同。
        ret = transportLitePtr->BuildLocRmaBufferLite(ReinterpretAs<uintptr_t>(dst), len, locRmaBuf);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s] FAIL at BuildLocRmaBufferLite. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], "
                "len[%llu].",
                __func__, thread, channel, dst, src, len),
            ret);
        const Hccl::Buffer rmtBuf{ReinterpretAs<uintptr_t>(src), len};

        EXCEPTION_CATCH(transportLitePtr->Read(locRmaBuf, rmtBuf, *streamLitePtr), ret = HCCL_E_INTERNAL);
    } else {
        HcclBuf locBuf{dst, len, nullptr};
        HcclBuf rmtBuf{const_cast<void*>(src), len, nullptr};

        Stream* stream = GetStream(thread);
        CHK_PTR_NULL(stream);

        ret = HcclRemoteRead(stream, ReinterpretAs<void*>(channel), &locBuf, &rmtBuf);
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] FAIL. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu].", __func__, thread,
            channel, dst, src, len),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommReadReduceOnThread(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t count, HcommDataType dataType,
    HcommReduceOp reduceOp)
{
    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);
    CHK_RET(UnwrapChannelHandle(channel));

    PLF_CONFIG_INFO(
        PLF_DATA_OP,
        "[%s] thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], count[%llu], dataType[%d], reduceOp[%d].",
        __func__, thread, channel, dst, src, count, dataType, reduceOp);

    AddThread(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    uint64_t len = count * SIZE_TABLE[dataType];

    HcclResult ret = HCCL_SUCCESS;
    if (threadPtr->IsDeviceA5()) {
        auto* const transportLitePtr = ReinterpretAs<Hccl::BaseTransportLiteImpl*>(channel);
        CHK_PTR_NULL(transportLitePtr);
        auto* const streamLitePtr = static_cast<Hccl::StreamLite*>(threadPtr->GetStreamLitePtr());
        CHK_PTR_NULL(streamLitePtr);

        Hccl::RmaBufferLite locRmaBuf;
        ret = transportLitePtr->BuildLocRmaBufferLite(ReinterpretAs<uintptr_t>(dst), len, locRmaBuf);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s] FAIL at BuildLocRmaBufferLite. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], "
                "count[%llu], dataType[%d], reduceOp[%d].",
                __func__, thread, channel, dst, src, count, dataType, reduceOp),
            ret);
        const Hccl::Buffer rmtBuf{ReinterpretAs<uintptr_t>(src), len};

        ret = CheckDataTypeAndReduceOp(dataType, reduceOp);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s] FAIL at CheckDataTypeAndReduceOp. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], "
                "count[%llu], dataType[%d], reduceOp[%d].",
                __func__, thread, channel, dst, src, count, dataType, reduceOp),
            ret);
        // [中文导读] 将远端读取数据归约到本地 dst，归约类型和操作转为 A5 底层描述后排入当前执行流。
        Hccl::ReduceIn reduceIn{mapHcommDataTypeToA5.at(dataType), mapHcommReduceOpToA5.at(reduceOp)};

        EXCEPTION_CATCH(
            transportLitePtr->ReadReduce(locRmaBuf, rmtBuf, reduceIn, *streamLitePtr), ret = HCCL_E_INTERNAL);
    } else {
        CHK_PRT_RET(
            (IsSupportReduce(dataType, reduceOp) == false),
            HCCL_ERROR(
                "[%s] Not support reduce, "
                "dst[0x%llx], src[0x%llx], count[%llu], dataType[%d], reduceOp[%d]",
                __func__, dst, src, count, dataType, reduceOp),
            HCCL_E_PARA);
        HcclBuf locBuf{dst, len, nullptr};
        HcclBuf rmtBuf{const_cast<void*>(src), len, nullptr};
        HcclReduceInfo reduceInfo{static_cast<HcclDataType>(dataType), static_cast<HcclReduceOp>(reduceOp)};

        Stream* stream = GetStream(thread);
        CHK_PTR_NULL(stream);

        ret = HcclRemoteReadReduce(stream, ReinterpretAs<void*>(channel), &locBuf, &rmtBuf, reduceInfo);
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] FAIL. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], count[%llu], dataType[%d], "
            "reduceOp[%d].",
            __func__, thread, channel, dst, src, count, dataType, reduceOp),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommBatchTransferOnThread(
    ThreadHandle thread, ChannelHandle channel, const HcommBatchTransferDesc* transferDescs, uint32_t transferDescNum)
{
    CHK_PTR_NULL(transferDescs);
    CHK_RET(UnwrapChannelHandle(channel));

    PLF_CONFIG_INFO(
        PLF_DATA_OP, "[%s] thread[0x%llx], channel[0x%llx], transferDescNum[%u].", __func__, thread, channel,
        transferDescNum);

    // [中文导读] transferDescNum 是批传输描述条数；这里拒绝空批，再取得执行 Thread 并登记提交上下文。
    CHK_PRT_RET(transferDescNum == 0, HCCL_ERROR("[%s] transferDescNum is 0.", __func__), HCCL_E_PARA);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);
    AddThread(thread);
    HcclResult ret = HCCL_SUCCESS;
    if (threadPtr->IsDeviceA5()) {
        auto* const ubTransportLitePtr = ReinterpretAs<Hccl::UbTransportLiteImpl*>(channel);
        CHK_PTR_NULL(ubTransportLitePtr);
        auto* const streamLitePtr = static_cast<Hccl::StreamLite*>(threadPtr->GetStreamLitePtr());
        CHK_PTR_NULL(streamLitePtr);
        // [中文导读] A5 批传输使用 UB 传输对象执行描述数组；兼容路径交 Transport::BatchTransferAsync 并保留不支持状态。
        ret = ubTransportLitePtr->ExecuteBatchTransfer(streamLitePtr, transferDescs, transferDescNum);
    } else {
        Stream* stream = GetStream(thread);
        CHK_PTR_NULL(stream);
        hccl::Transport* transport = ReinterpretAs<hccl::Transport*>(channel);
        CHK_PTR_NULL(transport);
        ret = transport->BatchTransferAsync(transferDescs, transferDescNum, *stream);
        if (ret == HCCL_E_NOT_SUPPORT) {
            return HCCL_E_NOT_SUPPORT;
        }
    }
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[%s] BatchTransferAsync failed.", __func__), ret);

    HCCL_INFO("[%s] SUCCESS. transferDescNum[%u].", __func__, transferDescNum);
    return ret;
}

int32_t HcommWriteNbiOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len)
{
    HCCL_DEBUG(
        "[%s] thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu].", __func__, thread, channel, dst,
        src, len);
    CHK_PTR_NULL(src);
    CHK_PTR_NULL(dst);
    // [中文导读] 此文件的 NBI 写接口尚未实现；非空地址校验通过也只返回 NOT_SUPPORT，不会发出传输任务。
    return HCCL_E_NOT_SUPPORT;
}

int32_t HcommWriteNbi(ChannelHandle channel, void* dst, const void* src, uint64_t len)
{
    HCCL_DEBUG("[%s] channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu].", __func__, channel, dst, src, len);
    CHK_PTR_NULL(src);
    CHK_PTR_NULL(dst);
    return HCCL_E_NOT_SUPPORT;
}

int32_t HcommWriteWithNotifyNbiOnThread(
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx)
{
    HCCL_DEBUG(
        "[%s] thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu], remoteNotifyIdx[%u].", __func__,
        thread, channel, dst, src, len, remoteNotifyIdx);
    CHK_PTR_NULL(src);
    CHK_PTR_NULL(dst);
    return HCCL_E_NOT_SUPPORT;
}

int32_t
HcommWriteWithNotifyNbi(ChannelHandle channel, void* dst, const void* src, uint64_t len, uint32_t remoteNotifyIdx)
{
    HCCL_DEBUG(
        "[%s] channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu], remoteNotifyIdx[%u].", __func__, channel, dst, src,
        len, remoteNotifyIdx);
    CHK_PTR_NULL(src);
    CHK_PTR_NULL(dst);
    return HCCL_E_NOT_SUPPORT;
}

int32_t HcommReadNbiOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len)
{
    HCCL_DEBUG(
        "[%s] thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu].", __func__, thread, channel, dst,
        src, len);
    CHK_PTR_NULL(src);
    CHK_PTR_NULL(dst);
    return HCCL_E_NOT_SUPPORT;
}

int32_t HcommReadNbi(ChannelHandle channel, void* dst, const void* src, uint64_t len)
{
    HCCL_DEBUG("[%s] channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu].", __func__, channel, dst, src, len);
    CHK_PTR_NULL(src);
    CHK_PTR_NULL(dst);
    return HCCL_E_NOT_SUPPORT;
}

// [中文导读] 跨Rank通知：经channel向远端remoteNotifyIdx槽位发信号，A5落到BaseTransportLiteImpl::Post。
// [中文导读] 它和ThreadNotifyRecord的目标不同：前者使用远端通道，后者使用本地目标Thread。
int32_t HcommChannelNotifyRecordOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t remoteNotifyIdx)
{
    CHK_RET(UnwrapChannelHandle(channel));

    PLF_CONFIG_INFO(
        PLF_DATA_OP, "[%s] thread[0x%llx], channel[0x%llx], remoteNotifyIdx[%u].", __func__, thread, channel,
        remoteNotifyIdx);

    AddThread(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    HcclResult ret = HCCL_SUCCESS;
    if (threadPtr->IsDeviceA5()) {
        HCCL_DEBUG("[%s] Running on A5.", __func__);
        auto* const transportLitePtr = ReinterpretAs<Hccl::BaseTransportLiteImpl*>(channel);
        CHK_PTR_NULL(transportLitePtr);
        auto* const streamLitePtr = static_cast<Hccl::StreamLite*>(threadPtr->GetStreamLitePtr());
        CHK_PTR_NULL(streamLitePtr);
        HCCL_INFO("channel streamlite ptr %p.", streamLitePtr);

        // [中文导读] 在当前执行流向对端槽位发通知；是否表示 ACK 或数据完成由上层的槽位分配协议决定。
        EXCEPTION_CATCH(transportLitePtr->Post(remoteNotifyIdx, *streamLitePtr), ret = HCCL_E_INTERNAL);
    } else {
        Stream* stream = GetStream(thread);
        CHK_PTR_NULL(stream);

        ret = HcclRemoteNotifyRecord(stream, ReinterpretAs<void*>(channel), remoteNotifyIdx);
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] FAIL. thread[0x%llx], channel[0x%llx], remoteNotifyIdx[%u].", __func__, thread, channel,
            remoteNotifyIdx),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommChannelNotifyRecord(ChannelHandle channel, uint32_t remoteNotifyIdx)
{
    HCCL_DEBUG("[%s] channel[0x%llx], remoteNotifyIdx[%u].", __func__, channel, remoteNotifyIdx);
    return HCCL_E_NOT_SUPPORT;
}

// [中文导读] 等待此通道本地localNotifyIdx槽位被远端通知；A5交WaitWithTimeout，timeOut以秒表示。
// [中文导读] 发送端的remoteNotifyIdx应与接收端localNotifyIdx按同一协议配对，ACK与数据完成槽不能混用。
int32_t
HcommChannelNotifyWaitOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeOut)
{
    CHK_RET(UnwrapChannelHandle(channel));

    PLF_CONFIG_INFO(
        PLF_DATA_OP, "[%s] thread[0x%llx], channel[0x%llx], localNotifyIdx[%u], timeOut[%u s].", __func__, thread,
        channel, localNotifyIdx, timeOut);

    AddThread(thread);

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    HcclResult ret = HCCL_SUCCESS;
    if (threadPtr->IsDeviceA5()) {
        HCCL_DEBUG("[%s] Running on A5.", __func__);
        auto* const transportLitePtr = ReinterpretAs<Hccl::BaseTransportLiteImpl*>(channel);
        CHK_PTR_NULL(transportLitePtr);
        auto* const streamLitePtr = static_cast<Hccl::StreamLite*>(threadPtr->GetStreamLitePtr());
        CHK_PTR_NULL(streamLitePtr);

        // [中文导读] 在当前执行流等待通道本地槽位通知，超时由底层 WaitWithTimeout 处理，异常统一转换为 INTERNAL。
        EXCEPTION_CATCH(
            transportLitePtr->WaitWithTimeout(localNotifyIdx, *streamLitePtr, timeOut), ret = HCCL_E_INTERNAL);
    } else {
        Stream* stream = GetStream(thread);
        CHK_PTR_NULL(stream);

        ret = HcclRemoteNotifyWait(stream, ReinterpretAs<void*>(channel), localNotifyIdx, timeOut);
    }
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] FAIL. thread[0x%llx], channel[0x%llx], localNotifyIdx[%u], timeOut[%u s].", __func__, thread, channel,
            localNotifyIdx, timeOut),
        ret);
    HCCL_INFO("[%s] SUCCESS.", __func__);
    return HCCL_SUCCESS;
}

int32_t HcommChannelNotifyWait(ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeOut)
{
    HCCL_DEBUG("[%s] channel[0x%llx], localNotifyIdx[%u], timeOut[%u s].", __func__, channel, localNotifyIdx, timeOut);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult CommFence(ThreadHandle thread, ChannelHandle channel) // 控制前后的任务保序
{
    CHK_RET(UnwrapChannelHandle(channel));

    HCCL_DEBUG("[CommFence] thread[0x%llx], channel[0x%llx].", thread, channel);
    Stream* stream = GetStream(thread);
    CHK_PTR_NULL(stream);

    return HcclRemoteFence(stream, ReinterpretAs<void*>(channel), false);
}

int32_t HcommSetLaunchMode(const char* launchTag, HcommLaunchMode mode)
{
    HCCL_DEBUG("HcommSetLaunchMode launchTag[%s]", launchTag);
    // [中文导读] 把标签和提交模式交给线程私有上下文，批量起止接口都通过这里改变同一提交状态。
    return g_threadLaunchCtx.SetLaunchMode(launchTag, mode);
}

// [中文导读] 执行控制：将当前launch上下文切换为以batchTag标识的批量模式，不负责创建通信域或通道。
int32_t HcommBatchModeStart(const char* batchTag) { return HcommSetLaunchMode(batchTag, HCOMM_LAUNCH_MODE_BATCH); }

// [中文导读] 恢复EAGER提交模式；提交细节由LaunchContext处理，不应将返回值解释为全设备同步完成。
int32_t HcommBatchModeEnd(const char* batchTag) { return HcommSetLaunchMode(batchTag, HCOMM_LAUNCH_MODE_EAGER); }

// [中文导读] 取得已存在的设备侧通信域供本次展开使用；950/960走AcquireCommForUse，不是从零建域。
// [中文导读] 正常路径应与ReleaseComm配对，防止域在任务展开期间被管理操作提前回收。
int32_t HcommAcquireComm(const char* commId)
{
    CHK_PTR_NULL(commId);
    DevType deviceType;
    CHK_RET(hrtGetDeviceType(deviceType));
    HCCL_INFO("[%s]comId[%s], devType[%d]", __func__, commId, deviceType);
    if (deviceType != DevType::DEV_TYPE_950 && deviceType != DevType::DEV_TYPE_960) {
        HcclCommAicpu* hcclComm = AicpuHcclProcess::AicpuGetCommbyGroup(commId);
        CHK_PRT_RET(!hcclComm, HCCL_ERROR("%s AicpuGetCommbyGroup is null, commId[%s]", __func__, commId), HCCL_E_PTR);
        // [中文导读] 兼容设备除了查域，还要把该域的 Dispatcher 上下文绑定到当前调用线程。
        CHK_RET(hcclComm->SetDispatcherCtxOnThread());
    } else {
        // [中文导读] 950/960 通过域管理器等待并设置独占使用标记，再取得当前域；该实现用 isUsed 状态而非引用计数。
        CollCommAicpu* hcclComm = CollCommAicpuMgr::GetInstance().AcquireCommForUse(commId);
        CHK_PRT_RET(!hcclComm, HCCL_ERROR("%s AcquireCommForUse is null, commId[%s]", __func__, commId), HCCL_E_PTR);
    }
    return HCCL_SUCCESS;
}

int32_t HcommChannelRegisterDfx(
    ChannelHandle channel, [[maybe_unused]] std::function<HcclResult(u32, u32, const Hccl::TaskParam&, u64)> callback)
{
    CHK_RET(UnwrapChannelHandle(channel));
    HCCL_INFO("[HcommChannelRegisterDfx] ChannelHandle[0x%llx] Init success", channel);
    return HCCL_SUCCESS;
}

int32_t
HcommThreadRegisterDfx(ThreadHandle thread, std::function<HcclResult(u32, u32, const Hccl::TaskParam&, u64)> callback)
{
    Thread* threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);
    CHK_RET(threadPtr->SetAddTaskInfoCallback(callback));
    HCCL_INFO("[HcommThreadRegisterDfx] ThreadHandle[0x%llx] Init success", thread);
    return HCCL_SUCCESS;
}

int32_t HcommThreadRegisterCheckExecStatus(ThreadHandle thread, std::function<HcclResult(bool)> callback)
{
    Thread* threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);
    CHK_RET(threadPtr->SetCheckExecStatusCallback(callback));
    return HCCL_SUCCESS;
}

int32_t HcommNewThreadRegisterDfx(ThreadHandle thread, std::function<void(Hccl::TaskInfoCircularQueue*)> callback)
{
    hccl::AicpuTsThread* tsThread = ReinterpretAs<hccl::AicpuTsThread*>(thread);
    CHK_PTR_NULL(tsThread);
    tsThread->SetReportStreamTaskCallback(std::move(callback));
    HCCL_INFO("[HcommNewThreadRegisterDfx] ThreadHandle[0x%llx] Init success", thread);
    return HCCL_SUCCESS;
}

int32_t HcommNewThreadRegisterGetLatestDfxOpInfo(ThreadHandle thread, std::function<const void*()> callback)
{
    hccl::AicpuTsThread* tsThread = ReinterpretAs<hccl::AicpuTsThread*>(thread);
    CHK_PTR_NULL(tsThread);
    tsThread->SetGetLatestDfxOpInfoCallback(std::move(callback));
    HCCL_INFO("[HcommNewThreadRegisterGetLatestDfxOpInfo] ThreadHandle[0x%llx] Init success", thread);
    return HCCL_SUCCESS;
}

// [中文导读] 结束本次对设备侧通信域的使用，与AcquireComm配对；不是要求销毁可供后续算子复用的整个域。
int32_t HcommReleaseComm(const char* commId)
{
    CHK_PTR_NULL(commId);
    DevType deviceType;
    CHK_RET(hrtGetDeviceType(deviceType));
    HCCL_INFO("[%s]comId[%s], devType[%d]", __func__, commId, deviceType);
    if (deviceType != DevType::DEV_TYPE_950 && deviceType != DevType::DEV_TYPE_960) {
        AicpuHcclProcess::AicpuReleaseCommbyGroup(commId);
    } else {
        // [中文导读] 950/960 清除此前 AcquireCommForUse 设置的域占用标记；域本身仍由管理器管理生命周期。
        CollCommAicpuMgr::GetInstance().ReleaseComm(commId);
    }
    return HCCL_SUCCESS;
}

int32_t HcommFenceOnThread(ThreadHandle thread)
{
    HCCL_DEBUG("[%s] thread[0x%llx].", __func__, thread);
    return HCCL_E_NOT_SUPPORT;
}

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
int32_t HcommFlush() { return HCCL_E_NOT_SUPPORT; }

int32_t HcommChannelFenceOnThread(ThreadHandle thread, ChannelHandle channel)
{
    CHK_RET(UnwrapChannelHandle(channel));

    HCCL_DEBUG("[%s] thread[0x%llx], channel[0x%llx].", __func__, thread, channel);
    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);
    if (threadPtr->IsDeviceA5()) {
        auto* const transportLitePtr = ReinterpretAs<Hccl::BaseTransportLiteImpl*>(channel);
        CHK_PTR_NULL(transportLitePtr);
        // [中文导读] A5 解包通道后调用传输层 Fence；此实现没有 AddThread，也没有发起 ThreadJoin 的 SQ 完成轮询。
        CHK_RET(transportLitePtr->Fence());
    }

    return HCCL_SUCCESS;
}

int32_t HcommChannelFence(ChannelHandle channel)
{
    HCCL_DEBUG("[%s] channel[0x%llx].", __func__, channel);
    return HCCL_E_NOT_SUPPORT;
}

int32_t HcommThreadJoin(ThreadHandle thread, uint32_t timeout)
{
    hccl::Thread* threadPtr = ReinterpretAs<hccl::Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    HCCL_INFO("[%s] START. thread[0x%llx].", __func__, thread);

    if (threadPtr->IsDeviceA5()) {
        HCCL_INFO("[%s] Running on A5.", __func__);
        auto* const streamLitePtr = static_cast<Hccl::StreamLite*>(threadPtr->GetStreamLitePtr());
        CHK_PTR_NULL(streamLitePtr);
        auto* const rtsqPtr = streamLitePtr->GetRtsq();
        CHK_PTR_NULL(rtsqPtr);

        uint32_t head = 0;
        uint32_t tail = 0;
        uint32_t sqId = streamLitePtr->GetSqId();
        // [中文导读] 先固定当前 SQ 尾位置，再轮询头位置是否追上该目标；等待范围由读取尾位置时的队列状态决定。
        EXCEPTION_CATCH(tail = rtsqPtr->QuerySqTail(), return HCCL_E_INTERNAL);
        HCCL_INFO("[%s] aicpu stream sqid[%u] tail[%u]", __func__, sqId, tail);

        u64 startUsec = GetCurAicpuTimestamp();
        u64 lastUsec = startUsec;
        constexpr uint64_t NANOSECOND_TO_SECOND = 1000000000U;
        const uint64_t kPrintSqInterval = 30U;
        // [中文导读] 以头尾相等判断当前目标完成，timeout 使用秒并换算为时间戳尺度，等待超时返回 TIMEOUT。
        do {
            EXCEPTION_CATCH(head = rtsqPtr->QuerySqHead(), return HCCL_E_INTERNAL);
            u64 curUsec = GetCurAicpuTimestamp();
            if (curUsec - startUsec > NANOSECOND_TO_SECOND * timeout) {
                HCCL_ERROR("[%s] timeout %us. curhead:%u, curtail:%u, sqId:%u", __func__, timeout, head, tail, sqId);
                return HCCL_E_TIMEOUT;
            }

            // 等待下发阶段，每隔30s打印一次状态
            if (curUsec - lastUsec > NANOSECOND_TO_SECOND * kPrintSqInterval) {
                lastUsec = curUsec;
                HCCL_RUN_INFO("[%s]Current state. sqid:%d, head:%u, tail:%u", __func__, sqId, head, tail);
            }
        } while (head != tail);
        HCCL_INFO("[%s] SUCCESS. RTSQ's head[%u] == tail[%u].", __func__, head, tail);
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("[%s]Does not support this interface.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

int32_t HcommChannelDrainOnThread(ThreadHandle thread, ChannelHandle channel)
{
    CHK_RET(UnwrapChannelHandle(channel));

    AddThread(thread);
    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);

    HcclResult ret = HCCL_SUCCESS;
    if (threadPtr->IsDeviceA5()) {
        HCCL_DEBUG("[%s] Running on A5.", __func__);
        auto* const transportLitePtr = ReinterpretAs<Hccl::BaseTransportLiteImpl*>(channel);
        CHK_PTR_NULL(transportLitePtr);
        auto* const streamLitePtr = static_cast<Hccl::StreamLite*>(threadPtr->GetStreamLitePtr());
        CHK_PTR_NULL(streamLitePtr);

        // [中文导读] 把通道 Drain 操作交给当前 StreamLite；它属于执行序列中的传输控制，接口不做 Host 完成轮询。
        EXCEPTION_CATCH(transportLitePtr->Drain(*streamLitePtr), ret = HCCL_E_INTERNAL);
    } else {
        Stream* stream = GetStream(thread);
        CHK_PTR_NULL(stream);
        ret = HcclRemoteDrain(stream, ReinterpretAs<void*>(channel));
    }

    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[%s] Run FAIL. thread[0x%llx], channel[0x%llx].", __func__, thread, channel),
        ret);

    return HCCL_SUCCESS;
}
#ifdef __cplusplus
}
#endif // __cplusplus

HcclResult HcommProfilingReportDeviceOp(const char* groupname)
{
    if (!GetProfilingEnable()) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(groupname);

    DevType deviceType;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (deviceType != DevType::DEV_TYPE_950 && deviceType != DevType::DEV_TYPE_960) {
        return HCCL_SUCCESS;
    }

    CollCommAicpu* currentComm = CollCommAicpuMgr::GetInstance().GetCurrentComm();
    CHK_PTR_NULL(currentComm);
    CHK_RET(currentComm->ProfilingReportDeviceOp());
    return HCCL_SUCCESS;
}

HcclResult HcommProfilingReportKernelStartTask(uint64_t thread, const char* groupname)
{
    if (!GetProfilingEnable()) {
        return HCCL_SUCCESS;
    }

    DevType deviceType;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (deviceType != DevType::DEV_TYPE_950 && deviceType != DevType::DEV_TYPE_960) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(groupname);
    CollCommAicpu* currentComm = CollCommAicpuMgr::GetInstance().GetCurrentComm();
    CHK_PTR_NULL(currentComm);
    CHK_RET(currentComm->UpdateTask());
    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PTR_NULL(threadPtr);
    auto* const streamLitePtr = static_cast<Hccl::StreamLite*>(threadPtr->GetStreamLitePtr());
    CHK_PTR_NULL(streamLitePtr);
    Hccl::DfxFlagTaskInfo flagTaskInfo;
    flagTaskInfo.taskId = streamLitePtr->GetRtsq()->GetTaskId();
    flagTaskInfo.type = Hccl::DfxMainStreamTaskType::HEAD;
    Hccl::DfxProfilingHandlerLite::GetInstance().ReportMainStreamTask(flagTaskInfo);
    HCCL_INFO("[%s] END, thread [%llu], groupname[%s], taskId[%u].", __func__, thread, groupname, flagTaskInfo.taskId);
    return HCCL_SUCCESS;
}

HcclResult HcommProfilingReportKernelEndTask(uint64_t thread, const char* groupname)
{
    if (!GetProfilingEnable()) {
        return HCCL_SUCCESS;
    }
    CHK_PTR_NULL(groupname);
    HCCL_INFO("[%s] START. thread [%llu], groupname[%s].", __func__, thread, groupname);

    DevType deviceType;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (deviceType != DevType::DEV_TYPE_950 && deviceType != DevType::DEV_TYPE_960) {
        return HCCL_SUCCESS;
    }

    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    CHK_PRT_RET(threadPtr == nullptr, HCCL_ERROR("[%s] threadPtr is null", __func__), HCCL_E_PTR);
    auto* const streamLitePtr = static_cast<Hccl::StreamLite*>(threadPtr->GetStreamLitePtr());
    CHK_PRT_RET(streamLitePtr == nullptr, HCCL_ERROR("[%s] streamLitePtr is null", __func__), HCCL_E_PTR);
    // FlagTaskInfo Report
    Hccl::DfxFlagTaskInfo flagTaskInfo;
    flagTaskInfo.type = Hccl::DfxMainStreamTaskType::TAIL;
    auto* rtsq = streamLitePtr->GetRtsq();
    CHK_PRT_RET(rtsq == nullptr, HCCL_ERROR("[%s] rtsq is null", __func__), HCCL_E_PTR);
    uint16_t streamId = 0;
    uint16_t taskId = 0;
    HcclResult ret = rtsq->GetLastStreamIdAndTaskId(streamId, taskId);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[%s] GetLastStreamIdAndTaskId fail, ret[%d], sqId[%u].", __func__, ret, streamLitePtr->GetSqId()),
        ret);
    constexpr uint32_t UINT16_BIT_WIDTH = 16U;
    flagTaskInfo.taskId = (static_cast<uint32_t>(taskId) << UINT16_BIT_WIDTH) | static_cast<uint32_t>(streamId);

    Hccl::DfxProfilingHandlerLite::GetInstance().ReportMainStreamTask(flagTaskInfo);
    return HCCL_SUCCESS;
}

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
// [中文导读] [AllReduce逐行 S130] HcommLocalCopyOnThread的接口声明：当前执行Thread句柄、操作目标地址、操作源地址、本次字节长度；这些参数属于本函数调用边界。
int32_t HcommLocalCopyOnThread(ThreadHandle thread, void* dst, const void* src, uint64_t len)
// [中文导读] [AllReduce逐行 S131] 进入HcommLocalCopyOnThread函数体：在通信 Thread 排入本地 src 到 dst 的 len 字节拷贝。
{
    // [中文导读] [AllReduce逐行 S132] 记录HcommLocalCopyOnThread的状态/性能诊断；日志本身不执行传输。
    PLF_CONFIG_INFO(
        // [中文导读] [AllReduce逐行 S133] 为前述多行表达式补入`PLF_DATA_OP, "[%s] thread[0x%llx], dst[0x%llx], src[0x%llx], len[%llu].", __func__, thread, dst, src, len)`（当前执行Thread句柄、操作目标地址、操作源地址、本次字节长度）；本行是参数/结构化初始化续行。
        PLF_DATA_OP, "[%s] thread[0x%llx], dst[0x%llx], src[0x%llx], len[%llu].", __func__, thread, dst, src, len);

    // [中文导读] [AllReduce逐行 S135] 检查`dst`（操作目标地址）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(dst);
    // [中文导读] [AllReduce逐行 S136] 检查`src`（操作源地址）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(src);
    // [中文导读] 批量模式下登记本次执行 Thread，EAGER 下登记函数直接返回；后续批提交汇总已登记的执行序列。
    // [中文导读] [AllReduce逐行 S138] BATCH模式将Thread加入参与列表，EAGER模式不登记；传入/处理当前执行Thread句柄。
    AddThread(thread);

    // [中文导读] [AllReduce逐行 S140] 设置当前Thread对象为/按`ReinterpretAs<Thread*>(thread)`（当前执行Thread句柄）。
    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    // [中文导读] [AllReduce逐行 S141] 检查`threadPtr`（当前Thread对象）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(threadPtr);

    // [中文导读] [AllReduce逐行 S143] 设置当前调用状态为/按`HCCL_SUCCESS`。
    HcclResult ret = HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S144] 仅当`(threadPtr->IsDeviceA5())`（当前Thread对象的IsDeviceA5字段）成立时进入此分支；检查Thread是否采用A5轻量设备实现。
    if (threadPtr->IsDeviceA5()) {
        // [中文导读] [AllReduce逐行 S145] 在异常捕获边界执行A5分支交AicpuTsThread生成SDMA本地复制；异常按后续处理语句转换成HCCL状态或提前返回。
        EXCEPTION_CATCH(ret = threadPtr->LocalCopy(dst, src, len), ret = HCCL_E_INTERNAL);
    // [中文导读] [AllReduce逐行 S146] 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。
    } else {
        // [中文导读] 兼容实现用 len 字节构造源/目标描述并取得对应 Stream，把本地拷贝交给旧原语适配层。
        // [中文导读] [AllReduce逐行 S148] 准备操作源地址、本次字节长度的局部存储/结构描述，初始化方式以本行声明为准。
        HcclBuf srcBuf{const_cast<void*>(src), len, nullptr};
        // [中文导读] [AllReduce逐行 S149] 准备操作目标地址、本次字节长度的局部存储/结构描述，初始化方式以本行声明为准。
        HcclBuf dstBuf{dst, len, nullptr};
        // [中文导读] [AllReduce逐行 S150] 设置承载任务的执行流为/按`GetStream(thread)`（当前执行Thread句柄）；取得兼容执行Stream对象。
        Stream* stream = GetStream(thread);
        // [中文导读] [AllReduce逐行 S151] 检查`stream`（承载任务的执行流）不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(stream);
        // [中文导读] [AllReduce逐行 S152] 设置当前调用状态为/按`HcclLocalCopy(stream, &dstBuf, &srcBuf)`（承载任务的执行流）；兼容原语在Stream上排入本地字节拷贝。
        ret = HcclLocalCopy(stream, &dstBuf, &srcBuf);
    // [中文导读] [AllReduce逐行 S153] 结束`if (threadPtr->IsDeviceA5())`（当前Thread对象的IsDeviceA5字段）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S154] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S155] 向条件错误检查提供`ret != HCCL_SUCCESS,`（当前调用状态），用于确定触发条件或形成对应诊断。
        ret != HCCL_SUCCESS,
        // [中文导读] [AllReduce逐行 S156] 记录HcommLocalCopyOnThread的错误诊断，字段包含当前执行Thread句柄、操作目标地址、操作源地址、本次字节长度；日志本身不执行传输。
        HCCL_ERROR("[%s] FAIL. thread[0x%llx], dst[0x%llx], src[0x%llx], len[%llu].", __func__, thread, dst, src, len),
        // [中文导读] [AllReduce逐行 S157] 指定条件命中时要返回的错误状态`ret)`（当前调用状态），未命中则继续原处理路径。
        ret);
    // [中文导读] [AllReduce逐行 S158] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S159] 结束HcommLocalCopyOnThread函数体；控制流返回外层。
}

// [中文导读] 数据计算原语：按dataType/reduceOp将源数据归约到目标；count是元素数，内部换算为字节len。
// [中文导读] 这是供算法组合的局部能力，不是一次完整AllReduce；调用者仍负责Rank间通信及依赖关系。
// [中文导读] [AllReduce逐行 S163] HcommLocalReduceOnThread的接口声明：将本地 src 按数据类型/操作归约到本地 dst，count 为元素数；这些参数属于本函数调用边界。
int32_t HcommLocalReduceOnThread(
    // [中文导读] [AllReduce逐行 S164] HcommLocalReduceOnThread的接口声明：当前执行Thread句柄、操作目标地址、操作源地址、归约元素数、元素数据类型、归约操作；这些参数属于本函数调用边界。
    ThreadHandle thread, void* dst, const void* src, uint64_t count, HcommDataType dataType, HcommReduceOp reduceOp)
// [中文导读] [AllReduce逐行 S165] 进入HcommLocalReduceOnThread函数体：将本地 src 按数据类型/操作归约到本地 dst，count 为元素数。
{
    // [中文导读] [AllReduce逐行 S166] 记录HcommLocalReduceOnThread的状态/性能诊断；日志本身不执行传输。
    PLF_CONFIG_INFO(
        // [中文导读] [AllReduce逐行 S167] 为前述多行表达式补入`PLF_DATA_OP, "[%s] thread[0x%llx], dst[0x%llx], src[0x%llx], count[%llu], dataType[%d], reduceOp[%d].",`；本行是参数/结构化初始化续行。
        PLF_DATA_OP, "[%s] thread[0x%llx], dst[0x%llx], src[0x%llx], count[%llu], dataType[%d], reduceOp[%d].",
        // [中文导读] [AllReduce逐行 S168] 为前述多行表达式补入`__func__, thread, dst, src, count, dataType, reduceOp)`（当前执行Thread句柄、操作目标地址、操作源地址、归约元素数、元素数据类型、归约操作）；本行是参数/结构化初始化续行。
        __func__, thread, dst, src, count, dataType, reduceOp);

    // [中文导读] [AllReduce逐行 S170] 检查`dst`（操作目标地址）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(dst);
    // [中文导读] [AllReduce逐行 S171] 检查`src`（操作源地址）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(src);
    // [中文导读] [AllReduce逐行 S172] BATCH模式将Thread加入参与列表，EAGER模式不登记；传入/处理当前执行Thread句柄。
    AddThread(thread);

    // [中文导读] [AllReduce逐行 S174] 设置当前Thread对象为/按`ReinterpretAs<Thread*>(thread)`（当前执行Thread句柄）。
    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    // [中文导读] [AllReduce逐行 S175] 检查`threadPtr`（当前Thread对象）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(threadPtr);

    // [中文导读] 按元素类型大小把 count 换算为字节范围，底层本地归约接收该范围与类型/操作。
    // [中文导读] [AllReduce逐行 S178] 设置本次字节长度为/按`count * SIZE_TABLE[dataType]`（归约元素数、元素数据类型）。
    uint64_t len = count * SIZE_TABLE[dataType];

    // [中文导读] [AllReduce逐行 S180] 设置当前调用状态为/按`HCCL_SUCCESS`。
    HcclResult ret = HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S181] 仅当`(threadPtr->IsDeviceA5())`（当前Thread对象的IsDeviceA5字段）成立时进入此分支；检查Thread是否采用A5轻量设备实现。
    if (threadPtr->IsDeviceA5()) {
        // [中文导读] [AllReduce逐行 S182] 在异常捕获边界执行A5分支交AicpuTsThread生成SDMA本地归约；异常按后续处理语句转换成HCCL状态或提前返回。
        EXCEPTION_CATCH(ret = threadPtr->LocalReduce(dst, src, len, dataType, reduceOp), ret = HCCL_E_INTERNAL);
    // [中文导读] [AllReduce逐行 S183] 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。
    } else {
        // [中文导读] 旧设备分支限制其实际支持的类型和归约操作，再转换为 HcclReduceInfo 调用兼容原语。
        // [中文导读] [AllReduce逐行 S185] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
        CHK_PRT_RET(
            // [中文导读] [AllReduce逐行 S186] 向条件错误检查提供`(IsSupportReduce(dataType, reduceOp) == false),`（元素数据类型、归约操作），用于确定触发条件或形成对应诊断。
            (IsSupportReduce(dataType, reduceOp) == false),
            // [中文导读] [AllReduce逐行 S187] 记录HcommLocalReduceOnThread的错误诊断；日志本身不执行传输。
            HCCL_ERROR(
                // [中文导读] [AllReduce逐行 S188] 为当前HcommLocalReduceOnThread诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                "[%s] Not support reduce, "
                // [中文导读] [AllReduce逐行 S189] 为当前HcommLocalReduceOnThread诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                "dst[0x%llx], src[0x%llx], count[%llu], dataType[%d], reduceOp[%d]",
                // [中文导读] [AllReduce逐行 S190] 为前述多行表达式补入`__func__, dst, src, count, dataType, reduceOp),`（操作目标地址、操作源地址、归约元素数、元素数据类型、归约操作）；本行是参数/结构化初始化续行。
                __func__, dst, src, count, dataType, reduceOp),
            // [中文导读] [AllReduce逐行 S191] 记录HcommLocalReduceOnThread的状态/性能诊断；日志本身不执行传输。
            HCCL_E_PARA);
        // [中文导读] [AllReduce逐行 S192] 准备操作源地址、本次字节长度的局部存储/结构描述，初始化方式以本行声明为准。
        HcclBuf srcBuf{const_cast<void*>(src), len, nullptr};
        // [中文导读] [AllReduce逐行 S193] 准备操作目标地址、本次字节长度的局部存储/结构描述，初始化方式以本行声明为准。
        HcclBuf dstBuf{dst, len, nullptr};
        // [中文导读] [AllReduce逐行 S194] 准备元素数据类型、归约操作的局部存储/结构描述，初始化方式以本行声明为准。
        HcclReduceInfo reduceInfo{static_cast<HcclDataType>(dataType), static_cast<HcclReduceOp>(reduceOp)};
        // [中文导读] [AllReduce逐行 S195] 设置承载任务的执行流为/按`GetStream(thread)`（当前执行Thread句柄）；取得兼容执行Stream对象。
        Stream* stream = GetStream(thread);
        // [中文导读] [AllReduce逐行 S196] 检查`stream`（承载任务的执行流）不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(stream);
        // [中文导读] [AllReduce逐行 S197] 设置当前调用状态为/按`HcclLocalCopyReduce(stream, &dstBuf, &srcBuf, reduceInfo)`（承载任务的执行流）；兼容原语在Stream上排入本地归约。
        ret = HcclLocalCopyReduce(stream, &dstBuf, &srcBuf, reduceInfo);
    // [中文导读] [AllReduce逐行 S198] 结束`if (threadPtr->IsDeviceA5())`（当前Thread对象的IsDeviceA5字段）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S199] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S200] 向条件错误检查提供`ret != HCCL_SUCCESS,`（当前调用状态），用于确定触发条件或形成对应诊断。
        ret != HCCL_SUCCESS,
        // [中文导读] [AllReduce逐行 S201] 记录HcommLocalReduceOnThread的错误诊断；日志本身不执行传输。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S202] 为当前HcommLocalReduceOnThread诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
            "[%s] FAIL. thread[0x%llx], dst[0x%llx], src[0x%llx], count[%llu], dataType[%d], reduceOp[%d].", __func__,
            // [中文导读] [AllReduce逐行 S203] 为前述多行表达式补入`thread, dst, src, count, dataType, reduceOp),`（当前执行Thread句柄、操作目标地址、操作源地址、归约元素数、元素数据类型、归约操作）；本行是参数/结构化初始化续行。
            thread, dst, src, count, dataType, reduceOp),
        // [中文导读] [AllReduce逐行 S204] 指定条件命中时要返回的错误状态`ret)`（当前调用状态），未命中则继续原处理路径。
        ret);
    // [中文导读] [AllReduce逐行 S205] 记录HcommLocalReduceOnThread的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO("[%s] SUCCESS.", __func__);
    // [中文导读] [AllReduce逐行 S206] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S207] 结束HcommLocalReduceOnThread函数体；控制流返回外层。
}

// [中文导读] 本地Thread间同步：在thread的执行序列上，向dstThread拥有的dstNotifyIdx通知槽发信号。
// [中文导读] A5分支先将通知槽索引解析为notifyId，再交LocalNotifyRecord；索引和硬件ID不是同一参数。
// [中文导读] [AllReduce逐行 S211] HcommThreadNotifyRecordOnThread的接口声明：当前执行Thread句柄、本地目标Thread句柄、目标Thread的通知槽索引；这些参数属于本函数调用边界。
int32_t HcommThreadNotifyRecordOnThread(ThreadHandle thread, ThreadHandle dstThread, uint32_t dstNotifyIdx)
// [中文导读] [AllReduce逐行 S212] 进入HcommThreadNotifyRecordOnThread函数体：在发送 Thread 向本地目标 Thread 的通知槽记录信号。
{
    // [中文导读] [AllReduce逐行 S213] 记录HcommThreadNotifyRecordOnThread的状态/性能诊断；日志本身不执行传输。
    PLF_CONFIG_INFO(
        // [中文导读] [AllReduce逐行 S214] 为前述多行表达式补入`PLF_DATA_OP, "[%s] thread[0x%llx], dstThread[0x%llx], dstNotifyIdx[%u].", __func__, thread, dstThread,`（当前执行Thread句柄、本地目标Thread句柄）；本行是参数/结构化初始化续行。
        PLF_DATA_OP, "[%s] thread[0x%llx], dstThread[0x%llx], dstNotifyIdx[%u].", __func__, thread, dstThread,
        // [中文导读] [AllReduce逐行 S215] 为前述多行表达式补入`dstNotifyIdx)`（目标Thread的通知槽索引）；本行是参数/结构化初始化续行。
        dstNotifyIdx);

    // [中文导读] [AllReduce逐行 S217] BATCH模式将Thread加入参与列表，EAGER模式不登记；传入/处理当前执行Thread句柄。
    AddThread(thread);

    // [中文导读] [AllReduce逐行 S219] 设置当前Thread对象为/按`ReinterpretAs<Thread*>(thread)`（当前执行Thread句柄）。
    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    // [中文导读] [AllReduce逐行 S220] 检查`threadPtr`（当前Thread对象）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(threadPtr);
    // [中文导读] [AllReduce逐行 S221] 设置目标Thread对象为/按`ReinterpretAs<Thread*>(dstThread)`（本地目标Thread句柄）。
    Thread* const dstThreadPtr = ReinterpretAs<Thread*>(dstThread);
    // [中文导读] [AllReduce逐行 S222] 检查`dstThreadPtr`（目标Thread对象）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(dstThreadPtr);

    // [中文导读] [AllReduce逐行 S224] 设置当前调用状态为/按`HCCL_SUCCESS`。
    HcclResult ret = HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S225] 仅当`(threadPtr->IsDeviceA5())`（当前Thread对象的IsDeviceA5字段）成立时进入此分支；检查Thread是否采用A5轻量设备实现。
    if (threadPtr->IsDeviceA5()) {
        // [中文导读] 通知资源属于目标 dstThread；读取它的槽位 ID，信号任务却排入发送者 thread 的执行序列。
        // [中文导读] [AllReduce逐行 S227] 设置通知资源对象为/按`dstThreadPtr->GetNotify(dstNotifyIdx)`（目标Thread对象的GetNotify字段、目标Thread的通知槽索引）；按所属Thread的通知槽索引取得资源对象。
        LocalNotify* const notifyPtr = dstThreadPtr->GetNotify(dstNotifyIdx);
        // [中文导读] [AllReduce逐行 S228] 检查`notifyPtr`（通知资源对象）不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(notifyPtr);
        // [中文导读] [AllReduce逐行 S229] 设置硬件通知ID为/按`notifyPtr->notifyId_`（通知资源对象的notifyId_字段）。
        const uint32_t notifyId = notifyPtr->notifyId_;
        // [中文导读] [AllReduce逐行 S230] 在异常捕获边界执行A5分支向目标硬件通知ID生成记录任务；异常按后续处理语句转换成HCCL状态或提前返回。
        EXCEPTION_CATCH(ret = threadPtr->LocalNotifyRecord(notifyId), ret = HCCL_E_INTERNAL);
    // [中文导读] [AllReduce逐行 S231] 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。
    } else {
        // [中文导读] [AllReduce逐行 S232] 设置承载任务的执行流为/按`GetStream(thread)`（当前执行Thread句柄）；取得兼容执行Stream对象。
        Stream* stream = GetStream(thread);
        // [中文导读] [AllReduce逐行 S233] 检查`stream`（承载任务的执行流）不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(stream);
        // [中文导读] [AllReduce逐行 S234] 设置LocalNotify* notify为/按`GetNotify(dstThread, dstNotifyIdx)`（本地目标Thread句柄、目标Thread的通知槽索引）；按所属Thread的通知槽索引取得资源对象。
        LocalNotify* notify = GetNotify(dstThread, dstNotifyIdx);
        // [中文导读] [AllReduce逐行 S235] 检查`notify`不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(notify);
        // [中文导读] [AllReduce逐行 S236] 设置当前调用状态为/按`HcclLocalNotifyRecord(stream, notify)`（承载任务的执行流）；兼容Stream向本地目标通知对象记录信号。
        ret = HcclLocalNotifyRecord(stream, notify);
    // [中文导读] [AllReduce逐行 S237] 结束`if (threadPtr->IsDeviceA5())`（当前Thread对象的IsDeviceA5字段）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S238] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S239] 向条件错误检查提供`ret != HCCL_SUCCESS,`（当前调用状态），用于确定触发条件或形成对应诊断。
        ret != HCCL_SUCCESS,
        // [中文导读] [AllReduce逐行 S240] 记录HcommThreadNotifyRecordOnThread的错误诊断；日志本身不执行传输。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S241] 为当前HcommThreadNotifyRecordOnThread诊断/异常表达式提供格式文本，将报告当前执行Thread句柄、本地目标Thread句柄；这一物理行没有数据搬运副作用。
            "[%s] FAIL. thread[0x%llx], dstThread[0x%llx], dstNotifyIdx[%u].", __func__, thread, dstThread,
            // [中文导读] [AllReduce逐行 S242] 为前述多行表达式补入`dstNotifyIdx),`（目标Thread的通知槽索引）；本行是参数/结构化初始化续行。
            dstNotifyIdx),
        // [中文导读] [AllReduce逐行 S243] 指定条件命中时要返回的错误状态`ret)`（当前调用状态），未命中则继续原处理路径。
        ret);
    // [中文导读] [AllReduce逐行 S244] 记录HcommThreadNotifyRecordOnThread的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO("[%s] SUCCESS.", __func__);
    // [中文导读] [AllReduce逐行 S245] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S246] 结束HcommThreadNotifyRecordOnThread函数体；控制流返回外层。
}

// [中文导读] 在本Thread排入等待自己notifyIdx槽位的任务，timeOut以秒表示；与另一Thread的Record配对。
// [中文导读] [AllReduce逐行 S249] HcommThreadNotifyWaitOnThread的接口声明：当前执行Thread句柄、本地通知槽索引、通知等待超时秒数；这些参数属于本函数调用边界。
int32_t HcommThreadNotifyWaitOnThread(ThreadHandle thread, uint32_t notifyIdx, uint32_t timeOut)
// [中文导读] [AllReduce逐行 S250] 进入HcommThreadNotifyWaitOnThread函数体：在当前 Thread 等待自己的通知槽，timeOut 为秒。
{
    // [中文导读] [AllReduce逐行 S251] 记录HcommThreadNotifyWaitOnThread的状态/性能诊断；日志本身不执行传输。
    PLF_CONFIG_INFO(
        // [中文导读] [AllReduce逐行 S252] 为前述多行表达式补入`PLF_DATA_OP, "[%s] thread[0x%llx], notifyIdx[%u], timeOut[%u s].", __func__, thread, notifyIdx, timeOut)`（当前执行Thread句柄、本地通知槽索引、通知等待超时秒数）；本行是参数/结构化初始化续行。
        PLF_DATA_OP, "[%s] thread[0x%llx], notifyIdx[%u], timeOut[%u s].", __func__, thread, notifyIdx, timeOut);

    // [中文导读] [AllReduce逐行 S254] BATCH模式将Thread加入参与列表，EAGER模式不登记；传入/处理当前执行Thread句柄。
    AddThread(thread);

    // [中文导读] [AllReduce逐行 S256] 设置当前Thread对象为/按`ReinterpretAs<Thread*>(thread)`（当前执行Thread句柄）。
    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    // [中文导读] [AllReduce逐行 S257] 检查`threadPtr`（当前Thread对象）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(threadPtr);

    // [中文导读] [AllReduce逐行 S259] 设置当前调用状态为/按`HCCL_SUCCESS`。
    HcclResult ret = HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S260] 仅当`(threadPtr->IsDeviceA5())`（当前Thread对象的IsDeviceA5字段）成立时进入此分支；检查Thread是否采用A5轻量设备实现。
    if (threadPtr->IsDeviceA5()) {
        // [中文导读] 等待资源来自当前 Thread 自己的槽位，将它转换为硬件通知 ID 后按 timeOut 秒建立等待。
        // [中文导读] [AllReduce逐行 S262] 设置通知资源对象为/按`threadPtr->GetNotify(notifyIdx)`（当前Thread对象的GetNotify字段、本地通知槽索引）；按所属Thread的通知槽索引取得资源对象。
        LocalNotify* const notifyPtr = threadPtr->GetNotify(notifyIdx);
        // [中文导读] [AllReduce逐行 S263] 检查`notifyPtr`（通知资源对象）不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(notifyPtr);
        // [中文导读] [AllReduce逐行 S264] 设置硬件通知ID为/按`notifyPtr->notifyId_`（通知资源对象的notifyId_字段）。
        const uint32_t notifyId = notifyPtr->notifyId_;
        // [中文导读] [AllReduce逐行 S265] 在异常捕获边界执行A5分支向本地通知ID生成秒级等待任务；异常按后续处理语句转换成HCCL状态或提前返回。
        EXCEPTION_CATCH(ret = threadPtr->LocalNotifyWait(notifyId, timeOut), ret = HCCL_E_INTERNAL);
    // [中文导读] [AllReduce逐行 S266] 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。
    } else {
        // [中文导读] [AllReduce逐行 S267] 设置承载任务的执行流为/按`GetStream(thread)`（当前执行Thread句柄）；取得兼容执行Stream对象。
        Stream* stream = GetStream(thread);
        // [中文导读] [AllReduce逐行 S268] 检查`stream`（承载任务的执行流）不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(stream);
        // [中文导读] [AllReduce逐行 S269] 设置LocalNotify* notify为/按`GetNotify(thread, notifyIdx)`（当前执行Thread句柄、本地通知槽索引）；按所属Thread的通知槽索引取得资源对象。
        LocalNotify* notify = GetNotify(thread, notifyIdx);
        // [中文导读] [AllReduce逐行 S270] 检查`notify`不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(notify);
        // [中文导读] [AllReduce逐行 S271] 设置当前调用状态为/按`HcclLocalNotifyWait(stream, notify, timeOut)`（承载任务的执行流、通知等待超时秒数）；兼容Stream等待本地通知对象。
        ret = HcclLocalNotifyWait(stream, notify, timeOut);
    // [中文导读] [AllReduce逐行 S272] 结束`if (threadPtr->IsDeviceA5())`（当前Thread对象的IsDeviceA5字段）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S273] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S274] 向条件错误检查提供`ret != HCCL_SUCCESS,`（当前调用状态），用于确定触发条件或形成对应诊断。
        ret != HCCL_SUCCESS,
        // [中文导读] [AllReduce逐行 S275] 记录HcommThreadNotifyWaitOnThread的错误诊断，字段包含当前执行Thread句柄、本地通知槽索引、通知等待超时秒数；日志本身不执行传输。
        HCCL_ERROR("[%s] FAIL. thread[0x%llx], notifyIdx[%u], timeOut[%u s].", __func__, thread, notifyIdx, timeOut),
        // [中文导读] [AllReduce逐行 S276] 指定条件命中时要返回的错误状态`ret)`（当前调用状态），未命中则继续原处理路径。
        ret);
    // [中文导读] [AllReduce逐行 S277] 记录HcommThreadNotifyWaitOnThread的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO("[%s] SUCCESS.", __func__);
    // [中文导读] [AllReduce逐行 S278] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S279] 结束HcommThreadNotifyWaitOnThread函数体；控制流返回外层。
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

// [中文导读] [AllReduce逐行 S346] CommTaskLaunch的接口声明：执行Thread句柄数组、执行Thread条数；这些参数属于本函数调用边界。
HcclResult CommTaskLaunch(ThreadHandle* threads, uint32_t threadNum) // host ffts+或aicpu stars使用"
// [中文导读] [AllReduce逐行 S347] 进入CommTaskLaunch函数体：逐 A5 Thread 提交组织的 SQE，兼容路径组装 Stream 数组统一发射。
{
    // [中文导读] [AllReduce逐行 S348] 检查`threads`（执行Thread句柄数组）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(threads);
    // [中文导读] [AllReduce逐行 S349] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
    CHK_PRT_RET(threadNum < 1, HCCL_ERROR("[CommTaskLaunch]threadNum is less than 1"), HCCL_E_PARA);

    // [中文导读] [AllReduce逐行 S351] 设置当前Thread对象为/按`ReinterpretAs<Thread*>(threads[0])`（执行Thread句柄数组）。
    Thread* threadPtr = ReinterpretAs<Thread*>(threads[0]);
    // [中文导读] [AllReduce逐行 S352] 检查`threadPtr`（当前Thread对象）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(threadPtr);

    // [中文导读] [AllReduce逐行 S354] 仅当`(threadPtr->IsDeviceA5())`（当前Thread对象的IsDeviceA5字段）成立时进入此分支；检查Thread是否采用A5轻量设备实现。
    if (threadPtr->IsDeviceA5()) {
        // [中文导读] [AllReduce逐行 S355] 记录CommTaskLaunch的状态/性能诊断；日志本身不执行传输。
        HCCL_INFO("[%s] Running on A5.", __func__);
        // [中文导读] A5 逐执行 Thread 调用 LaunchTask，任一异常转成 INTERNAL；成功表示任务提交接口完成。
        // [中文导读] [AllReduce逐行 S357] 按`(uint32_t i = 0; i < threadNum; i++)`（本批条目下标、执行Thread条数）遍历本批条目/分片；各次处理保持数组对应关系。
        for (uint32_t i = 0; i < threadNum; i++) {
            // [中文导读] [AllReduce逐行 S358] 设置本轮Thread对象为/按`ReinterpretAs<Thread*>(threads[i])`（执行Thread句柄数组、本批条目下标）。
            Thread* threadPtrLoop = ReinterpretAs<Thread*>(threads[i]);
            // [中文导读] [AllReduce逐行 S359] 检查`threadPtrLoop`（本轮Thread对象）不是空对象；宏命中失败条件时立即返回对应指针错误。
            CHK_PTR_NULL(threadPtrLoop);
            // [中文导读] [AllReduce逐行 S360] 记录CommTaskLaunch的状态/性能诊断，字段包含执行Thread句柄数组、本批条目下标；日志本身不执行传输。
            HCCL_INFO("[%s] Launching task in thread[0x%llx].", __func__, threads[i]);
            // [中文导读] [AllReduce逐行 S361] 在异常捕获边界执行将已生成的任务提交到具体RTSQ队列，不等同于全部任务完成；异常按后续处理语句转换成HCCL状态或提前返回。
            EXCEPTION_CATCH(threadPtrLoop->LaunchTask(), return HCCL_E_INTERNAL);
        // [中文导读] [AllReduce逐行 S362] 结束`for (uint32_t i = 0; i < threadNum; i++)`（本批条目下标、执行Thread条数）分支/循环；控制流返回外层。
        }
        // [中文导读] [AllReduce逐行 S363] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S364] 结束`if (threadPtr->IsDeviceA5())`（当前Thread对象的IsDeviceA5字段）分支/循环；控制流返回外层。
    }

    // [中文导读] 兼容实现把 Thread 列表转换为 Stream 数组，再以相同 threadNum 交给统一任务发射接口。
    // [中文导读] [AllReduce逐行 S367] 准备`std::vector<hccl::Stream> streams`的局部存储/结构描述，初始化方式以本行声明为准。
    std::vector<hccl::Stream> streams;
    // [中文导读] [AllReduce逐行 S368] 按`(uint32_t i = 0; i < threadNum; i++)`（本批条目下标、执行Thread条数）遍历本批条目/分片；各次处理保持数组对应关系。
    for (uint32_t i = 0; i < threadNum; i++) {
        // [中文导读] [AllReduce逐行 S369] 设置承载任务的执行流为/按`GetStream(threads[i])`（执行Thread句柄数组、本批条目下标）；取得兼容执行Stream对象。
        hccl::Stream* stream = GetStream(threads[i]);
        // [中文导读] [AllReduce逐行 S370] 检查`stream`（承载任务的执行流）不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(stream);
        // [中文导读] [AllReduce逐行 S371] 将当前条目追加到对应数组/列表；传入/处理承载任务的执行流。
        streams.push_back(*stream);
    // [中文导读] [AllReduce逐行 S372] 结束`for (uint32_t i = 0; i < threadNum; i++)`（本批条目下标、执行Thread条数）分支/循环；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S374] 直接返回`HcclTaskLaunch(streams.data(), threadNum)`（执行Thread条数）；兼容路径统一发射Stream数组对应任务。
    return HcclTaskLaunch(streams.data(), threadNum);
// [中文导读] [AllReduce逐行 S375] 结束CommTaskLaunch函数体；控制流返回外层。
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
// [中文导读] [AllReduce逐行 S519] HcommWriteOnThread的接口声明：当前执行Thread句柄、传输通道句柄、操作目标地址、操作源地址、本次字节长度；这些参数属于本函数调用边界。
int32_t HcommWriteOnThread(ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t len)
// [中文导读] [AllReduce逐行 S520] 进入HcommWriteOnThread函数体：把本地 src 的 len 字节写到通道远端 dst，A5 调用 UB/P2P 虚派发。
{
    // [中文导读] [AllReduce逐行 S521] 记录HcommWriteOnThread的状态/性能诊断；日志本身不执行传输。
    PLF_CONFIG_INFO(
        // [中文导读] [AllReduce逐行 S522] 为前述多行表达式补入`PLF_DATA_OP, "[%s] thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu].", __func__, thread,`（当前执行Thread句柄）；本行是参数/结构化初始化续行。
        PLF_DATA_OP, "[%s] thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu].", __func__, thread,
        // [中文导读] [AllReduce逐行 S523] 为前述多行表达式补入`channel, dst, src, len)`（传输通道句柄、操作目标地址、操作源地址、本次字节长度）；本行是参数/结构化初始化续行。
        channel, dst, src, len);

    // [中文导读] [AllReduce逐行 S525] 检查`dst`（操作目标地址）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(dst);
    // [中文导读] [AllReduce逐行 S526] 检查`src`（操作源地址）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(src);
    // [中文导读] 先解包用户通道句柄，再按 Thread 所属设备选择底层传输对象，避免直接把包装句柄当对象地址。
    // [中文导读] [AllReduce逐行 S528] 解包用户Channel表示以取得原始底层句柄；返回非成功时由检查宏立即向上传递。
    CHK_RET(UnwrapChannelHandle(channel));
    // [中文导读] [AllReduce逐行 S529] BATCH模式将Thread加入参与列表，EAGER模式不登记；传入/处理当前执行Thread句柄。
    AddThread(thread);

    // [中文导读] [AllReduce逐行 S531] 设置当前Thread对象为/按`ReinterpretAs<Thread*>(thread)`（当前执行Thread句柄）。
    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    // [中文导读] [AllReduce逐行 S532] 检查`threadPtr`（当前Thread对象）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(threadPtr);

    // [中文导读] [AllReduce逐行 S534] 设置当前调用状态为/按`HCCL_SUCCESS`。
    HcclResult ret = HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S535] 仅当`(threadPtr->IsDeviceA5())`（当前Thread对象的IsDeviceA5字段）成立时进入此分支；检查Thread是否采用A5轻量设备实现。
    if (threadPtr->IsDeviceA5()) {
        // [中文导读] [AllReduce逐行 S536] 设置auto* const transportLitePtr为/按`ReinterpretAs<Hccl::BaseTransportLiteImpl*>(channel)`（传输通道句柄）。
        auto* const transportLitePtr = ReinterpretAs<Hccl::BaseTransportLiteImpl*>(channel);
        // [中文导读] [AllReduce逐行 S537] 检查`transportLitePtr`不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(transportLitePtr);
        // [中文导读] [AllReduce逐行 S538] 设置设备轻量执行流为/按`static_cast<Hccl::StreamLite*>(threadPtr->GetStreamLitePtr())`（当前Thread对象的GetStreamLitePtr字段）；取得A5设备轻量执行流对象地址。
        auto* const streamLitePtr = static_cast<Hccl::StreamLite*>(threadPtr->GetStreamLitePtr());
        // [中文导读] [AllReduce逐行 S539] 检查`streamLitePtr`（设备轻量执行流）不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(streamLitePtr);

        // [中文导读] [AllReduce逐行 S541] 准备本端RMA描述的局部存储/结构描述，初始化方式以本行声明为准。
        Hccl::RmaBufferLite locRmaBuf;
        // [中文导读] 为本地源范围构造 RMA 描述；构造失败立即返回，不提交后面的远端写任务。
        // [中文导读] [AllReduce逐行 S543] 设置当前调用状态为/按`transportLitePtr->BuildLocRmaBufferLite(ReinterpretAs<uintptr_t>(src), len, locRmaBuf)`（操作源地址、本次字节长度、本端RMA描述）；为本端地址与字节范围解析已注册RMA token。
        ret = transportLitePtr->BuildLocRmaBufferLite(ReinterpretAs<uintptr_t>(src), len, locRmaBuf);
        // [中文导读] [AllReduce逐行 S544] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
        CHK_PRT_RET(
            // [中文导读] [AllReduce逐行 S545] 向条件错误检查提供`ret != HCCL_SUCCESS,`（当前调用状态），用于确定触发条件或形成对应诊断。
            ret != HCCL_SUCCESS,
            // [中文导读] [AllReduce逐行 S546] 记录HcommWriteOnThread的错误诊断；日志本身不执行传输。
            HCCL_ERROR(
                // [中文导读] [AllReduce逐行 S547] 为当前HcommWriteOnThread诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                "[%s] FAIL at BuildLocRmaBufferLite. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], "
                // [中文导读] [AllReduce逐行 S548] 为当前HcommWriteOnThread诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                "len[%llu].",
                // [中文导读] [AllReduce逐行 S549] 为前述多行表达式补入`__func__, thread, channel, dst, src, len),`（当前执行Thread句柄、传输通道句柄、操作目标地址、操作源地址、本次字节长度）；本行是参数/结构化初始化续行。
                __func__, thread, channel, dst, src, len),
            // [中文导读] [AllReduce逐行 S550] 指定条件命中时要返回的错误状态`ret)`（当前调用状态），未命中则继续原处理路径。
            ret);
        // [中文导读] 远端目标只用地址和字节长度描述，和本地已解析的 RMA 源一起交给当前 StreamLite。
        // [中文导读] [AllReduce逐行 S552] 准备操作目标地址、本次字节长度的局部存储/结构描述，初始化方式以本行声明为准。
        const Hccl::Buffer rmtBuf{ReinterpretAs<uintptr_t>(dst), len};

        // [中文导读] [AllReduce逐行 S554] 在异常捕获边界执行向连接远端目标生成WRITE WQE，本端为源；异常按后续处理语句转换成HCCL状态或提前返回。
        EXCEPTION_CATCH(transportLitePtr->Write(locRmaBuf, rmtBuf, *streamLitePtr), ret = HCCL_E_INTERNAL);
    // [中文导读] [AllReduce逐行 S555] 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。
    } else {
        // [中文导读] [AllReduce逐行 S556] 准备兼容路径的本端数据描述、操作源地址、本次字节长度的局部存储/结构描述，初始化方式以本行声明为准。
        HcclBuf locBuf{const_cast<void*>(src), len, nullptr};
        // [中文导读] [AllReduce逐行 S557] 准备操作目标地址、本次字节长度的局部存储/结构描述，初始化方式以本行声明为准。
        HcclBuf rmtBuf{dst, len, nullptr};

        // [中文导读] [AllReduce逐行 S559] 设置承载任务的执行流为/按`GetStream(thread)`（当前执行Thread句柄）；取得兼容执行Stream对象。
        Stream* stream = GetStream(thread);
        // [中文导读] [AllReduce逐行 S560] 检查`stream`（承载任务的执行流）不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(stream);

        // [中文导读] [AllReduce逐行 S562] 设置当前调用状态为/按`HcclRemoteWrite(stream, ReinterpretAs<void*>(channel), &rmtBuf, &locBuf)`（承载任务的执行流、传输通道句柄、兼容路径的本端数据描述）；兼容设备Stream通过旧Transport排入远端写。
        ret = HcclRemoteWrite(stream, ReinterpretAs<void*>(channel), &rmtBuf, &locBuf);
    // [中文导读] [AllReduce逐行 S563] 结束`if (threadPtr->IsDeviceA5())`（当前Thread对象的IsDeviceA5字段）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S564] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S565] 向条件错误检查提供`ret != HCCL_SUCCESS,`（当前调用状态），用于确定触发条件或形成对应诊断。
        ret != HCCL_SUCCESS,
        // [中文导读] [AllReduce逐行 S566] 记录HcommWriteOnThread的错误诊断；日志本身不执行传输。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S567] 为当前HcommWriteOnThread诊断/异常表达式提供格式文本，将报告当前执行Thread句柄；这一物理行没有数据搬运副作用。
            "[%s] FAIL. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], len[%llu].", __func__, thread,
            // [中文导读] [AllReduce逐行 S568] 为前述多行表达式补入`channel, dst, src, len),`（传输通道句柄、操作目标地址、操作源地址、本次字节长度）；本行是参数/结构化初始化续行。
            channel, dst, src, len),
        // [中文导读] [AllReduce逐行 S569] 指定条件命中时要返回的错误状态`ret)`（当前调用状态），未命中则继续原处理路径。
        ret);
    // [中文导读] [AllReduce逐行 S570] 记录HcommWriteOnThread的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO("[%s] SUCCESS.", __func__);
    // [中文导读] [AllReduce逐行 S571] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S572] 结束HcommWriteOnThread函数体；控制流返回外层。
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

// [中文导读] [AllReduce逐行 S866] HcommReadReduceOnThread的接口声明：读取通道远端 src 并归约到本地 dst；count 换算为字节长度；这些参数属于本函数调用边界。
int32_t HcommReadReduceOnThread(
    // [中文导读] [AllReduce逐行 S867] HcommReadReduceOnThread的接口声明：当前执行Thread句柄、传输通道句柄、操作目标地址、操作源地址、归约元素数、元素数据类型；这些参数属于本函数调用边界。
    ThreadHandle thread, ChannelHandle channel, void* dst, const void* src, uint64_t count, HcommDataType dataType,
    // [中文导读] [AllReduce逐行 S868] HcommReadReduceOnThread的接口声明：归约操作；这些参数属于本函数调用边界。
    HcommReduceOp reduceOp)
// [中文导读] [AllReduce逐行 S869] 进入HcommReadReduceOnThread函数体：读取通道远端 src 并归约到本地 dst；count 换算为字节长度。
{
    // [中文导读] [AllReduce逐行 S870] 检查`dst`（操作目标地址）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(dst);
    // [中文导读] [AllReduce逐行 S871] 检查`src`（操作源地址）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(src);
    // [中文导读] [AllReduce逐行 S872] 解包用户Channel表示以取得原始底层句柄；返回非成功时由检查宏立即向上传递。
    CHK_RET(UnwrapChannelHandle(channel));

    // [中文导读] [AllReduce逐行 S874] 记录HcommReadReduceOnThread的状态/性能诊断；日志本身不执行传输。
    PLF_CONFIG_INFO(
        // [中文导读] [AllReduce逐行 S875] 为前述多行表达式补入`PLF_DATA_OP,`；本行是参数/结构化初始化续行。
        PLF_DATA_OP,
        // [中文导读] [AllReduce逐行 S876] 为当前HcommReadReduceOnThread诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
        "[%s] thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], count[%llu], dataType[%d], reduceOp[%d].",
        // [中文导读] [AllReduce逐行 S877] 为前述多行表达式补入`__func__, thread, channel, dst, src, count, dataType, reduceOp)`（当前执行Thread句柄、传输通道句柄、操作目标地址、操作源地址、归约元素数、元素数据类型、归约操作）；本行是参数/结构化初始化续行。
        __func__, thread, channel, dst, src, count, dataType, reduceOp);

    // [中文导读] [AllReduce逐行 S879] BATCH模式将Thread加入参与列表，EAGER模式不登记；传入/处理当前执行Thread句柄。
    AddThread(thread);

    // [中文导读] [AllReduce逐行 S881] 设置当前Thread对象为/按`ReinterpretAs<Thread*>(thread)`（当前执行Thread句柄）。
    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    // [中文导读] [AllReduce逐行 S882] 检查`threadPtr`（当前Thread对象）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(threadPtr);

    // [中文导读] [AllReduce逐行 S884] 设置本次字节长度为/按`count * SIZE_TABLE[dataType]`（归约元素数、元素数据类型）。
    uint64_t len = count * SIZE_TABLE[dataType];

    // [中文导读] [AllReduce逐行 S886] 设置当前调用状态为/按`HCCL_SUCCESS`。
    HcclResult ret = HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S887] 仅当`(threadPtr->IsDeviceA5())`（当前Thread对象的IsDeviceA5字段）成立时进入此分支；检查Thread是否采用A5轻量设备实现。
    if (threadPtr->IsDeviceA5()) {
        // [中文导读] [AllReduce逐行 S888] 设置auto* const transportLitePtr为/按`ReinterpretAs<Hccl::BaseTransportLiteImpl*>(channel)`（传输通道句柄）。
        auto* const transportLitePtr = ReinterpretAs<Hccl::BaseTransportLiteImpl*>(channel);
        // [中文导读] [AllReduce逐行 S889] 检查`transportLitePtr`不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(transportLitePtr);
        // [中文导读] [AllReduce逐行 S890] 设置设备轻量执行流为/按`static_cast<Hccl::StreamLite*>(threadPtr->GetStreamLitePtr())`（当前Thread对象的GetStreamLitePtr字段）；取得A5设备轻量执行流对象地址。
        auto* const streamLitePtr = static_cast<Hccl::StreamLite*>(threadPtr->GetStreamLitePtr());
        // [中文导读] [AllReduce逐行 S891] 检查`streamLitePtr`（设备轻量执行流）不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(streamLitePtr);

        // [中文导读] [AllReduce逐行 S893] 准备本端RMA描述的局部存储/结构描述，初始化方式以本行声明为准。
        Hccl::RmaBufferLite locRmaBuf;
        // [中文导读] [AllReduce逐行 S894] 设置当前调用状态为/按`transportLitePtr->BuildLocRmaBufferLite(ReinterpretAs<uintptr_t>(dst), len, locRmaBuf)`（操作目标地址、本次字节长度、本端RMA描述）；为本端地址与字节范围解析已注册RMA token。
        ret = transportLitePtr->BuildLocRmaBufferLite(ReinterpretAs<uintptr_t>(dst), len, locRmaBuf);
        // [中文导读] [AllReduce逐行 S895] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
        CHK_PRT_RET(
            // [中文导读] [AllReduce逐行 S896] 向条件错误检查提供`ret != HCCL_SUCCESS,`（当前调用状态），用于确定触发条件或形成对应诊断。
            ret != HCCL_SUCCESS,
            // [中文导读] [AllReduce逐行 S897] 记录HcommReadReduceOnThread的错误诊断；日志本身不执行传输。
            HCCL_ERROR(
                // [中文导读] [AllReduce逐行 S898] 为当前HcommReadReduceOnThread诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                "[%s] FAIL at BuildLocRmaBufferLite. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], "
                // [中文导读] [AllReduce逐行 S899] 为当前HcommReadReduceOnThread诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                "count[%llu], dataType[%d], reduceOp[%d].",
                // [中文导读] [AllReduce逐行 S900] 为前述多行表达式补入`__func__, thread, channel, dst, src, count, dataType, reduceOp),`（当前执行Thread句柄、传输通道句柄、操作目标地址、操作源地址、归约元素数、元素数据类型、归约操作）；本行是参数/结构化初始化续行。
                __func__, thread, channel, dst, src, count, dataType, reduceOp),
            // [中文导读] [AllReduce逐行 S901] 指定条件命中时要返回的错误状态`ret)`（当前调用状态），未命中则继续原处理路径。
            ret);
        // [中文导读] [AllReduce逐行 S902] 准备操作源地址、本次字节长度的局部存储/结构描述，初始化方式以本行声明为准。
        const Hccl::Buffer rmtBuf{ReinterpretAs<uintptr_t>(src), len};

        // [中文导读] [AllReduce逐行 S904] 设置当前调用状态为/按`CheckDataTypeAndReduceOp(dataType, reduceOp)`（元素数据类型、归约操作）；检查公开归约类型/操作是否存在底层支持映射。
        ret = CheckDataTypeAndReduceOp(dataType, reduceOp);
        // [中文导读] [AllReduce逐行 S905] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
        CHK_PRT_RET(
            // [中文导读] [AllReduce逐行 S906] 向条件错误检查提供`ret != HCCL_SUCCESS,`（当前调用状态），用于确定触发条件或形成对应诊断。
            ret != HCCL_SUCCESS,
            // [中文导读] [AllReduce逐行 S907] 记录HcommReadReduceOnThread的错误诊断；日志本身不执行传输。
            HCCL_ERROR(
                // [中文导读] [AllReduce逐行 S908] 为当前HcommReadReduceOnThread诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                "[%s] FAIL at CheckDataTypeAndReduceOp. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], "
                // [中文导读] [AllReduce逐行 S909] 为当前HcommReadReduceOnThread诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                "count[%llu], dataType[%d], reduceOp[%d].",
                // [中文导读] [AllReduce逐行 S910] 为前述多行表达式补入`__func__, thread, channel, dst, src, count, dataType, reduceOp),`（当前执行Thread句柄、传输通道句柄、操作目标地址、操作源地址、归约元素数、元素数据类型、归约操作）；本行是参数/结构化初始化续行。
                __func__, thread, channel, dst, src, count, dataType, reduceOp),
            // [中文导读] [AllReduce逐行 S911] 指定条件命中时要返回的错误状态`ret)`（当前调用状态），未命中则继续原处理路径。
            ret);
        // [中文导读] 将远端读取数据归约到本地 dst，归约类型和操作转为 A5 底层描述后排入当前执行流。
        // [中文导读] [AllReduce逐行 S913] 调用at, at，使用底层归约类型/操作描述、元素数据类型、归约操作；对象涉及底层归约类型/操作描述、元素数据类型、归约操作。
        Hccl::ReduceIn reduceIn{mapHcommDataTypeToA5.at(dataType), mapHcommReduceOpToA5.at(reduceOp)};

        // [中文导读] [AllReduce逐行 S915] 在异常捕获边界执行后续表达式；异常按后续处理语句转换成HCCL状态或提前返回。
        EXCEPTION_CATCH(
            // [中文导读] [AllReduce逐行 S916] 为生成读取远端源并归约到本地目标的READ WQE补入`transportLitePtr->ReadReduce(locRmaBuf, rmtBuf, reduceIn, *streamLitePtr), ret = HCCL_E_INTERNAL)`（本端RMA描述、底层归约类型/操作描述、设备轻量执行流、当前调用状态）；本行是参数/结构化初始化续行。
            transportLitePtr->ReadReduce(locRmaBuf, rmtBuf, reduceIn, *streamLitePtr), ret = HCCL_E_INTERNAL);
    // [中文导读] [AllReduce逐行 S917] 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。
    } else {
        // [中文导读] [AllReduce逐行 S918] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
        CHK_PRT_RET(
            // [中文导读] [AllReduce逐行 S919] 向条件错误检查提供`(IsSupportReduce(dataType, reduceOp) == false),`（元素数据类型、归约操作），用于确定触发条件或形成对应诊断。
            (IsSupportReduce(dataType, reduceOp) == false),
            // [中文导读] [AllReduce逐行 S920] 记录HcommReadReduceOnThread的错误诊断；日志本身不执行传输。
            HCCL_ERROR(
                // [中文导读] [AllReduce逐行 S921] 为当前HcommReadReduceOnThread诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                "[%s] Not support reduce, "
                // [中文导读] [AllReduce逐行 S922] 为当前HcommReadReduceOnThread诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
                "dst[0x%llx], src[0x%llx], count[%llu], dataType[%d], reduceOp[%d]",
                // [中文导读] [AllReduce逐行 S923] 为前述多行表达式补入`__func__, dst, src, count, dataType, reduceOp),`（操作目标地址、操作源地址、归约元素数、元素数据类型、归约操作）；本行是参数/结构化初始化续行。
                __func__, dst, src, count, dataType, reduceOp),
            // [中文导读] [AllReduce逐行 S924] 记录HcommReadReduceOnThread的状态/性能诊断；日志本身不执行传输。
            HCCL_E_PARA);
        // [中文导读] [AllReduce逐行 S925] 准备兼容路径的本端数据描述、操作目标地址、本次字节长度的局部存储/结构描述，初始化方式以本行声明为准。
        HcclBuf locBuf{dst, len, nullptr};
        // [中文导读] [AllReduce逐行 S926] 准备操作源地址、本次字节长度的局部存储/结构描述，初始化方式以本行声明为准。
        HcclBuf rmtBuf{const_cast<void*>(src), len, nullptr};
        // [中文导读] [AllReduce逐行 S927] 准备元素数据类型、归约操作的局部存储/结构描述，初始化方式以本行声明为准。
        HcclReduceInfo reduceInfo{static_cast<HcclDataType>(dataType), static_cast<HcclReduceOp>(reduceOp)};

        // [中文导读] [AllReduce逐行 S929] 设置承载任务的执行流为/按`GetStream(thread)`（当前执行Thread句柄）；取得兼容执行Stream对象。
        Stream* stream = GetStream(thread);
        // [中文导读] [AllReduce逐行 S930] 检查`stream`（承载任务的执行流）不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(stream);

        // [中文导读] [AllReduce逐行 S932] 设置当前调用状态为/按`HcclRemoteReadReduce(stream, ReinterpretAs<void*>(channel), &locBuf, &rmtBuf, reduceInfo)`（承载任务的执行流、传输通道句柄、兼容路径的本端数据描述）；兼容设备Stream通过旧Transport排入远端读取归约。
        ret = HcclRemoteReadReduce(stream, ReinterpretAs<void*>(channel), &locBuf, &rmtBuf, reduceInfo);
    // [中文导读] [AllReduce逐行 S933] 结束`if (threadPtr->IsDeviceA5())`（当前Thread对象的IsDeviceA5字段）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S934] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S935] 向条件错误检查提供`ret != HCCL_SUCCESS,`（当前调用状态），用于确定触发条件或形成对应诊断。
        ret != HCCL_SUCCESS,
        // [中文导读] [AllReduce逐行 S936] 记录HcommReadReduceOnThread的错误诊断；日志本身不执行传输。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S937] 为当前HcommReadReduceOnThread诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
            "[%s] FAIL. thread[0x%llx], channel[0x%llx], dst[0x%llx], src[0x%llx], count[%llu], dataType[%d], "
            // [中文导读] [AllReduce逐行 S938] 为当前HcommReadReduceOnThread诊断/异常表达式提供格式文本；这一物理行没有数据搬运副作用。
            "reduceOp[%d].",
            // [中文导读] [AllReduce逐行 S939] 为前述多行表达式补入`__func__, thread, channel, dst, src, count, dataType, reduceOp),`（当前执行Thread句柄、传输通道句柄、操作目标地址、操作源地址、归约元素数、元素数据类型、归约操作）；本行是参数/结构化初始化续行。
            __func__, thread, channel, dst, src, count, dataType, reduceOp),
        // [中文导读] [AllReduce逐行 S940] 指定条件命中时要返回的错误状态`ret)`（当前调用状态），未命中则继续原处理路径。
        ret);
    // [中文导读] [AllReduce逐行 S941] 记录HcommReadReduceOnThread的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO("[%s] SUCCESS.", __func__);
    // [中文导读] [AllReduce逐行 S942] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S943] 结束HcommReadReduceOnThread函数体；控制流返回外层。
}

// [中文导读] [AllReduce逐行 S945] HcommBatchTransferOnThread的接口声明：解析当前 Thread/Channel 后提交一批数据/通知描述到选中传输实现；这些参数属于本函数调用边界。
int32_t HcommBatchTransferOnThread(
    // [中文导读] [AllReduce逐行 S946] HcommBatchTransferOnThread的接口声明：当前执行Thread句柄、传输通道句柄、公开批传输描述数组、批操作描述条数；这些参数属于本函数调用边界。
    ThreadHandle thread, ChannelHandle channel, const HcommBatchTransferDesc* transferDescs, uint32_t transferDescNum)
// [中文导读] [AllReduce逐行 S947] 进入HcommBatchTransferOnThread函数体：解析当前 Thread/Channel 后提交一批数据/通知描述到选中传输实现。
{
    // [中文导读] [AllReduce逐行 S948] 检查`transferDescs`（公开批传输描述数组）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(transferDescs);
    // [中文导读] [AllReduce逐行 S949] 解包用户Channel表示以取得原始底层句柄；返回非成功时由检查宏立即向上传递。
    CHK_RET(UnwrapChannelHandle(channel));

    // [中文导读] [AllReduce逐行 S951] 记录HcommBatchTransferOnThread的状态/性能诊断；日志本身不执行传输。
    PLF_CONFIG_INFO(
        // [中文导读] [AllReduce逐行 S952] 为前述多行表达式补入`PLF_DATA_OP, "[%s] thread[0x%llx], channel[0x%llx], transferDescNum[%u].", __func__, thread, channel,`（当前执行Thread句柄、传输通道句柄）；本行是参数/结构化初始化续行。
        PLF_DATA_OP, "[%s] thread[0x%llx], channel[0x%llx], transferDescNum[%u].", __func__, thread, channel,
        // [中文导读] [AllReduce逐行 S953] 为前述多行表达式补入`transferDescNum)`（批操作描述条数）；本行是参数/结构化初始化续行。
        transferDescNum);

    // [中文导读] transferDescNum 是批传输描述条数；这里拒绝空批，再取得执行 Thread 并登记提交上下文。
    // [中文导读] [AllReduce逐行 S956] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
    CHK_PRT_RET(transferDescNum == 0, HCCL_ERROR("[%s] transferDescNum is 0.", __func__), HCCL_E_PARA);

    // [中文导读] [AllReduce逐行 S958] 设置当前Thread对象为/按`ReinterpretAs<Thread*>(thread)`（当前执行Thread句柄）。
    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    // [中文导读] [AllReduce逐行 S959] 检查`threadPtr`（当前Thread对象）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(threadPtr);
    // [中文导读] [AllReduce逐行 S960] BATCH模式将Thread加入参与列表，EAGER模式不登记；传入/处理当前执行Thread句柄。
    AddThread(thread);
    // [中文导读] [AllReduce逐行 S961] 设置当前调用状态为/按`HCCL_SUCCESS`。
    HcclResult ret = HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S962] 仅当`(threadPtr->IsDeviceA5())`（当前Thread对象的IsDeviceA5字段）成立时进入此分支；检查Thread是否采用A5轻量设备实现。
    if (threadPtr->IsDeviceA5()) {
        // [中文导读] [AllReduce逐行 S963] 设置auto* const ubTransportLitePtr为/按`ReinterpretAs<Hccl::UbTransportLiteImpl*>(channel)`（传输通道句柄）。
        auto* const ubTransportLitePtr = ReinterpretAs<Hccl::UbTransportLiteImpl*>(channel);
        // [中文导读] [AllReduce逐行 S964] 检查`ubTransportLitePtr`不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(ubTransportLitePtr);
        // [中文导读] [AllReduce逐行 S965] 设置设备轻量执行流为/按`static_cast<Hccl::StreamLite*>(threadPtr->GetStreamLitePtr())`（当前Thread对象的GetStreamLitePtr字段）；取得A5设备轻量执行流对象地址。
        auto* const streamLitePtr = static_cast<Hccl::StreamLite*>(threadPtr->GetStreamLitePtr());
        // [中文导读] [AllReduce逐行 S966] 检查`streamLitePtr`（设备轻量执行流）不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(streamLitePtr);
        // [中文导读] A5 批传输使用 UB 传输对象执行描述数组；兼容路径交 Transport::BatchTransferAsync 并保留不支持状态。
        // [中文导读] [AllReduce逐行 S968] 设置当前调用状态为/按`ubTransportLitePtr->ExecuteBatchTransfer(streamLitePtr, transferDescs, transferDescNum)`（设备轻量执行流、公开批传输描述数组、批操作描述条数）；把公开批描述解析成UB slice/操作/通知数组并一次组织WQE。
        ret = ubTransportLitePtr->ExecuteBatchTransfer(streamLitePtr, transferDescs, transferDescNum);
    // [中文导读] [AllReduce逐行 S969] 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。
    } else {
        // [中文导读] [AllReduce逐行 S970] 设置承载任务的执行流为/按`GetStream(thread)`（当前执行Thread句柄）；取得兼容执行Stream对象。
        Stream* stream = GetStream(thread);
        // [中文导读] [AllReduce逐行 S971] 检查`stream`（承载任务的执行流）不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(stream);
        // [中文导读] [AllReduce逐行 S972] 设置hccl::Transport* transport为/按`ReinterpretAs<hccl::Transport*>(channel)`（传输通道句柄）。
        hccl::Transport* transport = ReinterpretAs<hccl::Transport*>(channel);
        // [中文导读] [AllReduce逐行 S973] 检查`transport`不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(transport);
        // [中文导读] [AllReduce逐行 S974] 设置当前调用状态为/按`transport->BatchTransferAsync(transferDescs, transferDescNum, *stream)`（公开批传输描述数组、批操作描述条数、承载任务的执行流）；兼容Transport按Stream组织批传输，保留NOT_SUPPORT状态。
        ret = transport->BatchTransferAsync(transferDescs, transferDescNum, *stream);
        // [中文导读] [AllReduce逐行 S975] 仅当`(ret == HCCL_E_NOT_SUPPORT)`（当前调用状态）成立时进入此分支。
        if (ret == HCCL_E_NOT_SUPPORT) {
            // [中文导读] [AllReduce逐行 S976] 返回HCCL_E_NOT_SUPPORT，表示当前引擎/设备/协议不支持此路径；此路径停止本函数的后续处理。
            return HCCL_E_NOT_SUPPORT;
        // [中文导读] [AllReduce逐行 S977] 结束`if (ret == HCCL_E_NOT_SUPPORT)`（当前调用状态）分支/循环；控制流返回外层。
        }
    // [中文导读] [AllReduce逐行 S978] 结束`if (threadPtr->IsDeviceA5())`（当前Thread对象的IsDeviceA5字段）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S979] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[%s] BatchTransferAsync failed.", __func__), ret);

    // [中文导读] [AllReduce逐行 S981] 记录HcommBatchTransferOnThread的状态/性能诊断，字段包含批操作描述条数；日志本身不执行传输。
    HCCL_INFO("[%s] SUCCESS. transferDescNum[%u].", __func__, transferDescNum);
    // [中文导读] [AllReduce逐行 S982] 返回ret，表示保留当前调用的状态（可能成功或失败）；此路径停止本函数的后续处理。
    return ret;
// [中文导读] [AllReduce逐行 S983] 结束HcommBatchTransferOnThread函数体；控制流返回外层。
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
// [中文导读] [AllReduce逐行 S1046] HcommChannelNotifyRecordOnThread的接口声明：当前执行Thread句柄、传输通道句柄、对端通知槽索引；这些参数属于本函数调用边界。
int32_t HcommChannelNotifyRecordOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t remoteNotifyIdx)
// [中文导读] [AllReduce逐行 S1047] 进入HcommChannelNotifyRecordOnThread函数体：在当前 Thread 通过通道向远端通知槽发信号。
{
    // [中文导读] [AllReduce逐行 S1048] 解包用户Channel表示以取得原始底层句柄；返回非成功时由检查宏立即向上传递。
    CHK_RET(UnwrapChannelHandle(channel));

    // [中文导读] [AllReduce逐行 S1050] 记录HcommChannelNotifyRecordOnThread的状态/性能诊断；日志本身不执行传输。
    PLF_CONFIG_INFO(
        // [中文导读] [AllReduce逐行 S1051] 为前述多行表达式补入`PLF_DATA_OP, "[%s] thread[0x%llx], channel[0x%llx], remoteNotifyIdx[%u].", __func__, thread, channel,`（当前执行Thread句柄、传输通道句柄）；本行是参数/结构化初始化续行。
        PLF_DATA_OP, "[%s] thread[0x%llx], channel[0x%llx], remoteNotifyIdx[%u].", __func__, thread, channel,
        // [中文导读] [AllReduce逐行 S1052] 为前述多行表达式补入`remoteNotifyIdx)`（对端通知槽索引）；本行是参数/结构化初始化续行。
        remoteNotifyIdx);

    // [中文导读] [AllReduce逐行 S1054] BATCH模式将Thread加入参与列表，EAGER模式不登记；传入/处理当前执行Thread句柄。
    AddThread(thread);

    // [中文导读] [AllReduce逐行 S1056] 设置当前Thread对象为/按`ReinterpretAs<Thread*>(thread)`（当前执行Thread句柄）。
    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    // [中文导读] [AllReduce逐行 S1057] 检查`threadPtr`（当前Thread对象）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(threadPtr);

    // [中文导读] [AllReduce逐行 S1059] 设置当前调用状态为/按`HCCL_SUCCESS`。
    HcclResult ret = HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S1060] 仅当`(threadPtr->IsDeviceA5())`（当前Thread对象的IsDeviceA5字段）成立时进入此分支；检查Thread是否采用A5轻量设备实现。
    if (threadPtr->IsDeviceA5()) {
        // [中文导读] [AllReduce逐行 S1061] 记录HcommChannelNotifyRecordOnThread的调试诊断；日志本身不执行传输。
        HCCL_DEBUG("[%s] Running on A5.", __func__);
        // [中文导读] [AllReduce逐行 S1062] 设置auto* const transportLitePtr为/按`ReinterpretAs<Hccl::BaseTransportLiteImpl*>(channel)`（传输通道句柄）。
        auto* const transportLitePtr = ReinterpretAs<Hccl::BaseTransportLiteImpl*>(channel);
        // [中文导读] [AllReduce逐行 S1063] 检查`transportLitePtr`不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(transportLitePtr);
        // [中文导读] [AllReduce逐行 S1064] 设置设备轻量执行流为/按`static_cast<Hccl::StreamLite*>(threadPtr->GetStreamLitePtr())`（当前Thread对象的GetStreamLitePtr字段）；取得A5设备轻量执行流对象地址。
        auto* const streamLitePtr = static_cast<Hccl::StreamLite*>(threadPtr->GetStreamLitePtr());
        // [中文导读] [AllReduce逐行 S1065] 检查`streamLitePtr`（设备轻量执行流）不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(streamLitePtr);
        // [中文导读] [AllReduce逐行 S1066] 记录HcommChannelNotifyRecordOnThread的状态/性能诊断，字段包含设备轻量执行流；日志本身不执行传输。
        HCCL_INFO("channel streamlite ptr %p.", streamLitePtr);

        // [中文导读] 在当前执行流向对端槽位发通知；是否表示 ACK 或数据完成由上层的槽位分配协议决定。
        // [中文导读] [AllReduce逐行 S1069] 在异常捕获边界执行通过远端通知slice写入内联值1并组织Doorbell；异常按后续处理语句转换成HCCL状态或提前返回。
        EXCEPTION_CATCH(transportLitePtr->Post(remoteNotifyIdx, *streamLitePtr), ret = HCCL_E_INTERNAL);
    // [中文导读] [AllReduce逐行 S1070] 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。
    } else {
        // [中文导读] [AllReduce逐行 S1071] 设置承载任务的执行流为/按`GetStream(thread)`（当前执行Thread句柄）；取得兼容执行Stream对象。
        Stream* stream = GetStream(thread);
        // [中文导读] [AllReduce逐行 S1072] 检查`stream`（承载任务的执行流）不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(stream);

        // [中文导读] [AllReduce逐行 S1074] 设置当前调用状态为/按`HcclRemoteNotifyRecord(stream, ReinterpretAs<void*>(channel), remoteNotifyIdx)`（承载任务的执行流、传输通道句柄、对端通知槽索引）；兼容Transport在Stream排入远端通知记录。
        ret = HcclRemoteNotifyRecord(stream, ReinterpretAs<void*>(channel), remoteNotifyIdx);
    // [中文导读] [AllReduce逐行 S1075] 结束`if (threadPtr->IsDeviceA5())`（当前Thread对象的IsDeviceA5字段）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S1076] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S1077] 向条件错误检查提供`ret != HCCL_SUCCESS,`（当前调用状态），用于确定触发条件或形成对应诊断。
        ret != HCCL_SUCCESS,
        // [中文导读] [AllReduce逐行 S1078] 记录HcommChannelNotifyRecordOnThread的错误诊断；日志本身不执行传输。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S1079] 为当前HcommChannelNotifyRecordOnThread诊断/异常表达式提供格式文本，将报告当前执行Thread句柄、传输通道句柄；这一物理行没有数据搬运副作用。
            "[%s] FAIL. thread[0x%llx], channel[0x%llx], remoteNotifyIdx[%u].", __func__, thread, channel,
            // [中文导读] [AllReduce逐行 S1080] 为前述多行表达式补入`remoteNotifyIdx),`（对端通知槽索引）；本行是参数/结构化初始化续行。
            remoteNotifyIdx),
        // [中文导读] [AllReduce逐行 S1081] 指定条件命中时要返回的错误状态`ret)`（当前调用状态），未命中则继续原处理路径。
        ret);
    // [中文导读] [AllReduce逐行 S1082] 记录HcommChannelNotifyRecordOnThread的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO("[%s] SUCCESS.", __func__);
    // [中文导读] [AllReduce逐行 S1083] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S1084] 结束HcommChannelNotifyRecordOnThread函数体；控制流返回外层。
}

int32_t HcommChannelNotifyRecord(ChannelHandle channel, uint32_t remoteNotifyIdx)
{
    HCCL_DEBUG("[%s] channel[0x%llx], remoteNotifyIdx[%u].", __func__, channel, remoteNotifyIdx);
    return HCCL_E_NOT_SUPPORT;
}

// [中文导读] 等待此通道本地localNotifyIdx槽位被远端通知；A5交WaitWithTimeout，timeOut以秒表示。
// [中文导读] 发送端的remoteNotifyIdx应与接收端localNotifyIdx按同一协议配对，ACK与数据完成槽不能混用。
// [中文导读] [AllReduce逐行 S1094] HcommChannelNotifyWaitOnThread的接口声明：在当前 Thread 等待通道本地通知槽，timeout 为秒；这些参数属于本函数调用边界。
int32_t
// [中文导读] [AllReduce逐行 S1095] HcommChannelNotifyWaitOnThread的接口声明：当前执行Thread句柄、传输通道句柄、通道本地通知槽索引、通知等待超时秒数；这些参数属于本函数调用边界。
HcommChannelNotifyWaitOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeOut)
// [中文导读] [AllReduce逐行 S1096] 进入HcommChannelNotifyWaitOnThread函数体：在当前 Thread 等待通道本地通知槽，timeout 为秒。
{
    // [中文导读] [AllReduce逐行 S1097] 解包用户Channel表示以取得原始底层句柄；返回非成功时由检查宏立即向上传递。
    CHK_RET(UnwrapChannelHandle(channel));

    // [中文导读] [AllReduce逐行 S1099] 记录HcommChannelNotifyWaitOnThread的状态/性能诊断；日志本身不执行传输。
    PLF_CONFIG_INFO(
        // [中文导读] [AllReduce逐行 S1100] 为前述多行表达式补入`PLF_DATA_OP, "[%s] thread[0x%llx], channel[0x%llx], localNotifyIdx[%u], timeOut[%u s].", __func__, thread,`（当前执行Thread句柄）；本行是参数/结构化初始化续行。
        PLF_DATA_OP, "[%s] thread[0x%llx], channel[0x%llx], localNotifyIdx[%u], timeOut[%u s].", __func__, thread,
        // [中文导读] [AllReduce逐行 S1101] 为前述多行表达式补入`channel, localNotifyIdx, timeOut)`（传输通道句柄、通道本地通知槽索引、通知等待超时秒数）；本行是参数/结构化初始化续行。
        channel, localNotifyIdx, timeOut);

    // [中文导读] [AllReduce逐行 S1103] BATCH模式将Thread加入参与列表，EAGER模式不登记；传入/处理当前执行Thread句柄。
    AddThread(thread);

    // [中文导读] [AllReduce逐行 S1105] 设置当前Thread对象为/按`ReinterpretAs<Thread*>(thread)`（当前执行Thread句柄）。
    Thread* const threadPtr = ReinterpretAs<Thread*>(thread);
    // [中文导读] [AllReduce逐行 S1106] 检查`threadPtr`（当前Thread对象）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(threadPtr);

    // [中文导读] [AllReduce逐行 S1108] 设置当前调用状态为/按`HCCL_SUCCESS`。
    HcclResult ret = HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S1109] 仅当`(threadPtr->IsDeviceA5())`（当前Thread对象的IsDeviceA5字段）成立时进入此分支；检查Thread是否采用A5轻量设备实现。
    if (threadPtr->IsDeviceA5()) {
        // [中文导读] [AllReduce逐行 S1110] 记录HcommChannelNotifyWaitOnThread的调试诊断；日志本身不执行传输。
        HCCL_DEBUG("[%s] Running on A5.", __func__);
        // [中文导读] [AllReduce逐行 S1111] 设置auto* const transportLitePtr为/按`ReinterpretAs<Hccl::BaseTransportLiteImpl*>(channel)`（传输通道句柄）。
        auto* const transportLitePtr = ReinterpretAs<Hccl::BaseTransportLiteImpl*>(channel);
        // [中文导读] [AllReduce逐行 S1112] 检查`transportLitePtr`不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(transportLitePtr);
        // [中文导读] [AllReduce逐行 S1113] 设置设备轻量执行流为/按`static_cast<Hccl::StreamLite*>(threadPtr->GetStreamLitePtr())`（当前Thread对象的GetStreamLitePtr字段）；取得A5设备轻量执行流对象地址。
        auto* const streamLitePtr = static_cast<Hccl::StreamLite*>(threadPtr->GetStreamLitePtr());
        // [中文导读] [AllReduce逐行 S1114] 检查`streamLitePtr`（设备轻量执行流）不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(streamLitePtr);

        // [中文导读] 在当前执行流等待通道本地槽位通知，超时由底层 WaitWithTimeout 处理，异常统一转换为 INTERNAL。
        // [中文导读] [AllReduce逐行 S1117] 在异常捕获边界执行后续表达式；异常按后续处理语句转换成HCCL状态或提前返回。
        EXCEPTION_CATCH(
            // [中文导读] [AllReduce逐行 S1118] 为本地通知索引解析ID并生成有秒级超时等待补入`transportLitePtr->WaitWithTimeout(localNotifyIdx, *streamLitePtr, timeOut), ret = HCCL_E_INTERNAL)`（通道本地通知槽索引、设备轻量执行流、通知等待超时秒数、当前调用状态）；本行是参数/结构化初始化续行。
            transportLitePtr->WaitWithTimeout(localNotifyIdx, *streamLitePtr, timeOut), ret = HCCL_E_INTERNAL);
    // [中文导读] [AllReduce逐行 S1119] 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。
    } else {
        // [中文导读] [AllReduce逐行 S1120] 设置承载任务的执行流为/按`GetStream(thread)`（当前执行Thread句柄）；取得兼容执行Stream对象。
        Stream* stream = GetStream(thread);
        // [中文导读] [AllReduce逐行 S1121] 检查`stream`（承载任务的执行流）不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(stream);

        // [中文导读] [AllReduce逐行 S1123] 设置当前调用状态为/按`HcclRemoteNotifyWait(stream, ReinterpretAs<void*>(channel), localNotifyIdx, timeOut)`（承载任务的执行流、传输通道句柄、通道本地通知槽索引、通知等待超时秒数）；兼容Transport在Stream排入通道本地通知等待。
        ret = HcclRemoteNotifyWait(stream, ReinterpretAs<void*>(channel), localNotifyIdx, timeOut);
    // [中文导读] [AllReduce逐行 S1124] 结束`if (threadPtr->IsDeviceA5())`（当前Thread对象的IsDeviceA5字段）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S1125] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S1126] 向条件错误检查提供`ret != HCCL_SUCCESS,`（当前调用状态），用于确定触发条件或形成对应诊断。
        ret != HCCL_SUCCESS,
        // [中文导读] [AllReduce逐行 S1127] 记录HcommChannelNotifyWaitOnThread的错误诊断；日志本身不执行传输。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S1128] 为当前HcommChannelNotifyWaitOnThread诊断/异常表达式提供格式文本，将报告当前执行Thread句柄、传输通道句柄；这一物理行没有数据搬运副作用。
            "[%s] FAIL. thread[0x%llx], channel[0x%llx], localNotifyIdx[%u], timeOut[%u s].", __func__, thread, channel,
            // [中文导读] [AllReduce逐行 S1129] 为前述多行表达式补入`localNotifyIdx, timeOut),`（通道本地通知槽索引、通知等待超时秒数）；本行是参数/结构化初始化续行。
            localNotifyIdx, timeOut),
        // [中文导读] [AllReduce逐行 S1130] 指定条件命中时要返回的错误状态`ret)`（当前调用状态），未命中则继续原处理路径。
        ret);
    // [中文导读] [AllReduce逐行 S1131] 记录HcommChannelNotifyWaitOnThread的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO("[%s] SUCCESS.", __func__);
    // [中文导读] [AllReduce逐行 S1132] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S1133] 结束HcommChannelNotifyWaitOnThread函数体；控制流返回外层。
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

// [中文导读] [AllReduce逐行 S1152] HcommSetLaunchMode的接口声明：任务提交标签、请求的提交模式；这些参数属于本函数调用边界。
int32_t HcommSetLaunchMode(const char* launchTag, HcommLaunchMode mode)
// [中文导读] [AllReduce逐行 S1153] 进入HcommSetLaunchMode函数体：提交模式与标签的 C 接口包装。
{
    // [中文导读] [AllReduce逐行 S1154] 记录HcommSetLaunchMode的调试诊断，字段包含任务提交标签；日志本身不执行传输。
    HCCL_DEBUG("HcommSetLaunchMode launchTag[%s]", launchTag);
    // [中文导读] 把标签和提交模式交给线程私有上下文，批量起止接口都通过这里改变同一提交状态。
    // [中文导读] [AllReduce逐行 S1156] 直接返回`g_threadLaunchCtx.SetLaunchMode(launchTag, mode)`（任务提交标签、请求的提交模式）；更新线程私有提交模式并按模式准备/发射/清理。
    return g_threadLaunchCtx.SetLaunchMode(launchTag, mode);
// [中文导读] [AllReduce逐行 S1157] 结束HcommSetLaunchMode函数体；控制流返回外层。
}

// [中文导读] 执行控制：将当前launch上下文切换为以batchTag标识的批量模式，不负责创建通信域或通道。
// [中文导读] [AllReduce逐行 S1160] 本行定义并直接执行HcommBatchModeStart：切换当前调用线程的 LaunchContext 到 BATCH，延迟显式任务提交；调用HcommSetLaunchMode。
int32_t HcommBatchModeStart(const char* batchTag) { return HcommSetLaunchMode(batchTag, HCOMM_LAUNCH_MODE_BATCH); }

// [中文导读] 恢复EAGER提交模式；提交细节由LaunchContext处理，不应将返回值解释为全设备同步完成。
// [中文导读] [AllReduce逐行 S1163] 本行定义并直接执行HcommBatchModeEnd：恢复 EAGER 并由 LaunchContext 提交参与 Thread 的任务后清理批状态；调用HcommSetLaunchMode。
int32_t HcommBatchModeEnd(const char* batchTag) { return HcommSetLaunchMode(batchTag, HCOMM_LAUNCH_MODE_EAGER); }

// [中文导读] 取得已存在的设备侧通信域供本次展开使用；950/960走AcquireCommForUse，不是从零建域。
// [中文导读] 正常路径应与ReleaseComm配对，防止域在任务展开期间被管理操作提前回收。
// [中文导读] [AllReduce逐行 S1167] HcommAcquireComm的接口声明：通信域名称；这些参数属于本函数调用边界。
int32_t HcommAcquireComm(const char* commId)
// [中文导读] [AllReduce逐行 S1168] 进入HcommAcquireComm函数体：按设备版本获得域并设置当前线程 Dispatcher 或设备域占用状态。
{
    // [中文导读] [AllReduce逐行 S1169] 检查`commId`（通信域名称）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(commId);
    // [中文导读] [AllReduce逐行 S1170] 准备设备型号的局部存储/结构描述，初始化方式以本行声明为准。
    DevType deviceType;
    // [中文导读] [AllReduce逐行 S1171] 读取设备型号用于新旧/协议分支选择；返回非成功时由检查宏立即向上传递。
    CHK_RET(hrtGetDeviceType(deviceType));
    // [中文导读] [AllReduce逐行 S1172] 记录HcommAcquireComm的状态/性能诊断，字段包含通信域名称、设备型号；日志本身不执行传输。
    HCCL_INFO("[%s]comId[%s], devType[%d]", __func__, commId, deviceType);
    // [中文导读] [AllReduce逐行 S1173] 仅当`(deviceType != DevType::DEV_TYPE_950 && deviceType != DevType::DEV_TYPE_960)`（设备型号）成立时进入此分支。
    if (deviceType != DevType::DEV_TYPE_950 && deviceType != DevType::DEV_TYPE_960) {
        // [中文导读] [AllReduce逐行 S1174] 设置该分支的设备通信域对象为/按`AicpuHcclProcess::AicpuGetCommbyGroup(commId)`（通信域名称）；从兼容设备域注册表按名称查通信域。
        HcclCommAicpu* hcclComm = AicpuHcclProcess::AicpuGetCommbyGroup(commId);
        // [中文导读] [AllReduce逐行 S1175] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
        CHK_PRT_RET(!hcclComm, HCCL_ERROR("%s AicpuGetCommbyGroup is null, commId[%s]", __func__, commId), HCCL_E_PTR);
        // [中文导读] 兼容设备除了查域，还要把该域的 Dispatcher 上下文绑定到当前调用线程。
        // [中文导读] [AllReduce逐行 S1177] 为兼容设备把当前域Dispatcher绑定到调用线程；返回非成功时由检查宏立即向上传递。
        CHK_RET(hcclComm->SetDispatcherCtxOnThread());
    // [中文导读] [AllReduce逐行 S1178] 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。
    } else {
        // [中文导读] 950/960 通过域管理器等待并设置独占使用标记，再取得当前域；该实现用 isUsed 状态而非引用计数。
        // [中文导读] [AllReduce逐行 S1180] 设置该分支的设备通信域对象为/按`CollCommAicpuMgr::GetInstance().AcquireCommForUse(commId)`（通信域名称）；取得该管理器单例；等待域占用状态可用后设置isUsed并返回设备域对象。
        CollCommAicpu* hcclComm = CollCommAicpuMgr::GetInstance().AcquireCommForUse(commId);
        // [中文导读] [AllReduce逐行 S1181] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
        CHK_PRT_RET(!hcclComm, HCCL_ERROR("%s AcquireCommForUse is null, commId[%s]", __func__, commId), HCCL_E_PTR);
    // [中文导读] [AllReduce逐行 S1182] 结束`if (deviceType != DevType::DEV_TYPE_950 && deviceType != DevType::DEV_TYPE_960)`（设备型号）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S1183] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S1184] 结束HcommAcquireComm函数体；控制流返回外层。
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
// [中文导读] [AllReduce逐行 S1231] HcommReleaseComm的接口声明：通信域名称；这些参数属于本函数调用边界。
int32_t HcommReleaseComm(const char* commId)
// [中文导读] [AllReduce逐行 S1232] 进入HcommReleaseComm函数体：结束当前设备域使用，按设备版本清理上下文/占用标记。
{
    // [中文导读] [AllReduce逐行 S1233] 检查`commId`（通信域名称）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(commId);
    // [中文导读] [AllReduce逐行 S1234] 准备设备型号的局部存储/结构描述，初始化方式以本行声明为准。
    DevType deviceType;
    // [中文导读] [AllReduce逐行 S1235] 读取设备型号用于新旧/协议分支选择；返回非成功时由检查宏立即向上传递。
    CHK_RET(hrtGetDeviceType(deviceType));
    // [中文导读] [AllReduce逐行 S1236] 记录HcommReleaseComm的状态/性能诊断，字段包含通信域名称、设备型号；日志本身不执行传输。
    HCCL_INFO("[%s]comId[%s], devType[%d]", __func__, commId, deviceType);
    // [中文导读] [AllReduce逐行 S1237] 仅当`(deviceType != DevType::DEV_TYPE_950 && deviceType != DevType::DEV_TYPE_960)`（设备型号）成立时进入此分支。
    if (deviceType != DevType::DEV_TYPE_950 && deviceType != DevType::DEV_TYPE_960) {
        // [中文导读] [AllReduce逐行 S1238] 结束兼容设备域使用；传入/处理通信域名称。
        AicpuHcclProcess::AicpuReleaseCommbyGroup(commId);
    // [中文导读] [AllReduce逐行 S1239] 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。
    } else {
        // [中文导读] 950/960 清除此前 AcquireCommForUse 设置的域占用标记；域本身仍由管理器管理生命周期。
        // [中文导读] [AllReduce逐行 S1241] 取得该管理器单例；清除设备域占用标记与当前域指针；传入/处理通信域名称。
        CollCommAicpuMgr::GetInstance().ReleaseComm(commId);
    // [中文导读] [AllReduce逐行 S1242] 结束`if (deviceType != DevType::DEV_TYPE_950 && deviceType != DevType::DEV_TYPE_960)`（设备型号）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S1243] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S1244] 结束HcommReleaseComm函数体；控制流返回外层。
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

// [中文导读] [AllReduce逐行 S1280] HcommThreadJoin的接口声明：当前执行Thread句柄、超时秒数；这些参数属于本函数调用边界。
int32_t HcommThreadJoin(ThreadHandle thread, uint32_t timeout)
// [中文导读] [AllReduce逐行 S1281] 进入HcommThreadJoin函数体：A5 读取 SQ 当前尾位置后轮询头位置追上目标，timeout 为秒。
{
    // [中文导读] [AllReduce逐行 S1282] 设置当前Thread对象为/按`ReinterpretAs<hccl::Thread*>(thread)`（当前执行Thread句柄）。
    hccl::Thread* threadPtr = ReinterpretAs<hccl::Thread*>(thread);
    // [中文导读] [AllReduce逐行 S1283] 检查`threadPtr`（当前Thread对象）不是空对象；宏命中失败条件时立即返回对应指针错误。
    CHK_PTR_NULL(threadPtr);

    // [中文导读] [AllReduce逐行 S1285] 记录HcommThreadJoin的状态/性能诊断，字段包含当前执行Thread句柄；日志本身不执行传输。
    HCCL_INFO("[%s] START. thread[0x%llx].", __func__, thread);

    // [中文导读] [AllReduce逐行 S1287] 仅当`(threadPtr->IsDeviceA5())`（当前Thread对象的IsDeviceA5字段）成立时进入此分支；检查Thread是否采用A5轻量设备实现。
    if (threadPtr->IsDeviceA5()) {
        // [中文导读] [AllReduce逐行 S1288] 记录HcommThreadJoin的状态/性能诊断；日志本身不执行传输。
        HCCL_INFO("[%s] Running on A5.", __func__);
        // [中文导读] [AllReduce逐行 S1289] 设置设备轻量执行流为/按`static_cast<Hccl::StreamLite*>(threadPtr->GetStreamLitePtr())`（当前Thread对象的GetStreamLitePtr字段）；取得A5设备轻量执行流对象地址。
        auto* const streamLitePtr = static_cast<Hccl::StreamLite*>(threadPtr->GetStreamLitePtr());
        // [中文导读] [AllReduce逐行 S1290] 检查`streamLitePtr`（设备轻量执行流）不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(streamLitePtr);
        // [中文导读] [AllReduce逐行 S1291] 设置执行队列对象为/按`streamLitePtr->GetRtsq()`（设备轻量执行流的GetRtsq字段）；返回当前StreamLite持有的具体执行队列。
        auto* const rtsqPtr = streamLitePtr->GetRtsq();
        // [中文导读] [AllReduce逐行 S1292] 检查`rtsqPtr`（执行队列对象）不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(rtsqPtr);

        // [中文导读] [AllReduce逐行 S1294] 设置本次轮询到的SQ头位置为/按`0`。
        uint32_t head = 0;
        // [中文导读] [AllReduce逐行 S1295] 设置读取时固定的SQ尾目标为/按`0`。
        uint32_t tail = 0;
        // [中文导读] [AllReduce逐行 S1296] 设置硬件SQ编号为/按`streamLitePtr->GetSqId()`（设备轻量执行流的GetSqId字段）；读取硬件SQ编号。
        uint32_t sqId = streamLitePtr->GetSqId();
        // [中文导读] 先固定当前 SQ 尾位置，再轮询头位置是否追上该目标；等待范围由读取尾位置时的队列状态决定。
        // [中文导读] [AllReduce逐行 S1298] 在异常捕获边界执行通过驱动查询SQ尾，作为此次Join等待的固定目标；异常按后续处理语句转换成HCCL状态或提前返回。
        EXCEPTION_CATCH(tail = rtsqPtr->QuerySqTail(), return HCCL_E_INTERNAL);
        // [中文导读] [AllReduce逐行 S1299] 记录HcommThreadJoin的状态/性能诊断，字段包含硬件SQ编号、读取时固定的SQ尾目标；日志本身不执行传输。
        HCCL_INFO("[%s] aicpu stream sqid[%u] tail[%u]", __func__, sqId, tail);

        // [中文导读] [AllReduce逐行 S1301] 设置u64 startUsec为/按`GetCurAicpuTimestamp()`；读取AICPU时间戳用于SQ完成等待计时。
        u64 startUsec = GetCurAicpuTimestamp();
        // [中文导读] [AllReduce逐行 S1302] 设置u64 lastUsec为/按`startUsec`。
        u64 lastUsec = startUsec;
        // [中文导读] [AllReduce逐行 S1303] 设置constexpr uint64_t NANOSECOND_TO_SECOND为/按`1000000000U`。
        constexpr uint64_t NANOSECOND_TO_SECOND = 1000000000U;
        // [中文导读] [AllReduce逐行 S1304] 设置const uint64_t kPrintSqInterval为/按`30U`。
        const uint64_t kPrintSqInterval = 30U;
        // [中文导读] 以头尾相等判断当前目标完成，timeout 使用秒并换算为时间戳尺度，等待超时返回 TIMEOUT。
        // [中文导读] [AllReduce逐行 S1306] 开始至少执行一次的SQ完成轮询，循环末尾比较实际头位置与固定尾目标。
        do {
            // [中文导读] [AllReduce逐行 S1307] 在异常捕获边界执行通过驱动查询SQ头，判断是否追上固定尾位置；异常按后续处理语句转换成HCCL状态或提前返回。
            EXCEPTION_CATCH(head = rtsqPtr->QuerySqHead(), return HCCL_E_INTERNAL);
            // [中文导读] [AllReduce逐行 S1308] 设置u64 curUsec为/按`GetCurAicpuTimestamp()`；读取AICPU时间戳用于SQ完成等待计时。
            u64 curUsec = GetCurAicpuTimestamp();
            // [中文导读] [AllReduce逐行 S1309] 仅当`(curUsec - startUsec > NANOSECOND_TO_SECOND * timeout)`（超时秒数）成立时进入此分支。
            if (curUsec - startUsec > NANOSECOND_TO_SECOND * timeout) {
                // [中文导读] [AllReduce逐行 S1310] 记录HcommThreadJoin的错误诊断，字段包含超时秒数、本次轮询到的SQ头位置、读取时固定的SQ尾目标、硬件SQ编号；日志本身不执行传输。
                HCCL_ERROR("[%s] timeout %us. curhead:%u, curtail:%u, sqId:%u", __func__, timeout, head, tail, sqId);
                // [中文导读] [AllReduce逐行 S1311] 返回HCCL_E_TIMEOUT，表示等待超过本函数期限；此路径停止本函数的后续处理。
                return HCCL_E_TIMEOUT;
            // [中文导读] [AllReduce逐行 S1312] 结束`if (curUsec - startUsec > NANOSECOND_TO_SECOND * timeout)`（超时秒数）分支/循环；控制流返回外层。
            }

            // 等待下发阶段，每隔30s打印一次状态
            // [中文导读] [AllReduce逐行 S1315] 仅当`(curUsec - lastUsec > NANOSECOND_TO_SECOND * kPrintSqInterval)`成立时进入此分支。
            if (curUsec - lastUsec > NANOSECOND_TO_SECOND * kPrintSqInterval) {
                // [中文导读] [AllReduce逐行 S1316] 设置lastUsec为/按`curUsec`。
                lastUsec = curUsec;
                // [中文导读] [AllReduce逐行 S1317] 记录HcommThreadJoin的状态/性能诊断，字段包含硬件SQ编号、本次轮询到的SQ头位置、读取时固定的SQ尾目标；日志本身不执行传输。
                HCCL_RUN_INFO("[%s]Current state. sqid:%d, head:%u, tail:%u", __func__, sqId, head, tail);
            // [中文导读] [AllReduce逐行 S1318] 结束`if (curUsec - lastUsec > NANOSECOND_TO_SECOND * kPrintSqInterval)`分支/循环；控制流返回外层。
            }
        // [中文导读] [AllReduce逐行 S1319] 当前轮处理结束后按`(head != tail)`（本次轮询到的SQ头位置、读取时固定的SQ尾目标）决定是否继续轮询。
        } while (head != tail);
        // [中文导读] [AllReduce逐行 S1320] 记录HcommThreadJoin的状态/性能诊断，字段包含本次轮询到的SQ头位置、读取时固定的SQ尾目标；日志本身不执行传输。
        HCCL_INFO("[%s] SUCCESS. RTSQ's head[%u] == tail[%u].", __func__, head, tail);
        // [中文导读] [AllReduce逐行 S1321] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S1322] 结束`if (threadPtr->IsDeviceA5())`（当前Thread对象的IsDeviceA5字段）分支/循环；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S1324] 记录HcommThreadJoin的错误诊断；日志本身不执行传输。
    HCCL_ERROR("[%s]Does not support this interface.", __func__);
    // [中文导读] [AllReduce逐行 S1325] 返回HCCL_E_NOT_SUPPORT，表示当前引擎/设备/协议不支持此路径；此路径停止本函数的后续处理。
    return HCCL_E_NOT_SUPPORT;
// [中文导读] [AllReduce逐行 S1326] 结束HcommThreadJoin函数体；控制流返回外层。
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

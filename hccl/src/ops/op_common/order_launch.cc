/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "order_launch.h"
#include "hcomm_primitives_dl.h"
#include "hccl_res_dl.h"
#include "dlhcomm_function.h"

namespace ops_hccl {

// [中文导读] [AllReduce逐行 S18] 声明OpLaunchGetUnfoldStream接口：以可选接口查询Host展开线程关联ACL stream。
static HcclResult OpLaunchGetUnfoldStream(HcclComm comm, ThreadHandle unfoldThread, aclrtStream& resolvedStream)
// [中文导读] [AllReduce逐行 S19] 开始OpLaunchGetUnfoldStream的函数体。
{
    // [中文导读] [AllReduce逐行 S20] 初始化待查询的Host展开流地址。
    void* unfoldStream = nullptr;
    // [中文导读] [AllReduce逐行 S21] 取得可选HCOMM动态函数表。
    auto& HcclThreadResGetInfoFunc = ops_hccl::DlHcommFunction::GetInstance();
    // [中文导读] [AllReduce逐行 S22] 缺少HcclThreadResGetInfo函数时无法查询展开流。
    if (!HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo) {
        // [中文导读] [AllReduce逐行 S23] 没有查询能力时将流输出置空。
        resolvedStream = nullptr;
        // [中文导读] [AllReduce逐行 S24] 输出警告日志，记录OpLaunchGetUnfoldStream当前阶段和相关参数。
        HCCL_WARNING("HcclThreadResGetInfoFunc dlHcclThreadResGetInfo is invalid.");
        // [中文导读] [AllReduce逐行 S25] 此辅助函数以成功和空流表示查询不可用，后续调用者再检查。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S26] 结束条件if (!HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo)。
    }
    // [中文导读] [AllReduce逐行 S27] 声明展开流查询返回码。
    HcclResult ret
        // [中文导读] [AllReduce逐行 S28] 查询unfoldThread的ACL stream关联资源。
        = HcclThreadResGetInfoFunc.dlHcclThreadResGetInfo(comm, unfoldThread, 0, sizeof(void*), &unfoldStream);
    // [中文导读] [AllReduce逐行 S29] 明确NOT_SUPPORT时使用空流返回。
    if (ret == HCCL_E_NOT_SUPPORT) {
        // [中文导读] [AllReduce逐行 S30] 将不支持查询的展开流结果设为空。
        resolvedStream = nullptr;
        // [中文导读] [AllReduce逐行 S31] 输出警告日志，记录OpLaunchGetUnfoldStream当前阶段和相关参数。
        HCCL_WARNING("HcclThreadResGetInfoFunc dlHcclThreadResGetInfo not support.");
        // [中文导读] [AllReduce逐行 S32] 返回成功，由上层OpLaunchGetOrderStreams检查空流。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S33] 其它查询错误也返回空流，不向上传递原ret。
    } else if (ret != HCCL_SUCCESS) {
        // [中文导读] [AllReduce逐行 S34] 将查询失败输出流设为空。
        resolvedStream = nullptr;
        // [中文导读] [AllReduce逐行 S35] 输出警告日志，记录OpLaunchGetUnfoldStream当前阶段和相关参数。
        HCCL_WARNING("HcclThreadResGetInfoFunc dlHcclThreadResGetInfo not success.");
        // [中文导读] [AllReduce逐行 S36] 返回成功，由使用者处理空流。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S37] 结束条件} else if (ret != HCCL_SUCCESS)。
    }
    // [中文导读] [AllReduce逐行 S38] 查询成功时输出实际展开流。
    resolvedStream = unfoldStream;
    // [中文导读] [AllReduce逐行 S39] 以可选接口查询Host展开线程关联ACL stream处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S40] 结束OpLaunchGetUnfoldStream函数体。
}

// [中文导读] [AllReduce逐行 S42] 声明OpLaunchGetHostOrderStream接口：以可选接口查询Host保序线程关联ACL stream。
static HcclResult OpLaunchGetHostOrderStream(ThreadHandle hostOrderThread, aclrtStream& resolvedStream)
// [中文导读] [AllReduce逐行 S43] 开始OpLaunchGetHostOrderStream的函数体。
{
    // [中文导读] [AllReduce逐行 S44] 初始化Host保序流输出地址。
    void* hostOrderStream = nullptr;
    // [中文导读] [AllReduce逐行 S45] 取得可选HCOMM动态函数表。
    auto& HcclThreadResGetInfoFunc = ops_hccl::DlHcommFunction::GetInstance();
    // [中文导读] [AllReduce逐行 S46] 缺少无通信域版本HcommThreadResGetInfo函数时无法查询保序流。
    if (!HcclThreadResGetInfoFunc.dlHcommThreadResGetInfo) {
        // [中文导读] [AllReduce逐行 S47] 没有查询能力时流输出置空。
        resolvedStream = nullptr;
        // [中文导读] [AllReduce逐行 S48] 输出警告日志，记录OpLaunchGetHostOrderStream当前阶段和相关参数。
        HCCL_WARNING("HcclThreadResGetInfoFunc dlHcommThreadResGetInfo is invalid.");
        // [中文导读] [AllReduce逐行 S49] 返回成功和空流供上层检查。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S50] 结束条件if (!HcclThreadResGetInfoFunc.dlHcommThreadResGetInfo)。
    }
    // [中文导读] [AllReduce逐行 S51] 声明保序流查询返回码。
    HcclResult ret
        // [中文导读] [AllReduce逐行 S52] 查询Host专用保序线程对应ACL stream。
        = HcclThreadResGetInfoFunc.dlHcommThreadResGetInfo(hostOrderThread, 0, sizeof(void*), &hostOrderStream);
    // [中文导读] [AllReduce逐行 S53] 明确NOT_SUPPORT时使用空流结果。
    if (ret == HCCL_E_NOT_SUPPORT) {
        // [中文导读] [AllReduce逐行 S54] 将不支持查询的保序流输出置空。
        resolvedStream = nullptr;
        // [中文导读] [AllReduce逐行 S55] 输出警告日志，记录OpLaunchGetHostOrderStream当前阶段和相关参数。
        HCCL_WARNING("HcclThreadResGetInfoFunc dlHcommThreadResGetInfo not support.");
        // [中文导读] [AllReduce逐行 S56] 返回成功，由上层统一处理空流。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S57] 其它查询错误同样转换为空流。
    } else if (ret != HCCL_SUCCESS) {
        // [中文导读] [AllReduce逐行 S58] 将保序流查询失败输出置空。
        resolvedStream = nullptr;
        // [中文导读] [AllReduce逐行 S59] 输出警告日志，记录OpLaunchGetHostOrderStream当前阶段和相关参数。
        HCCL_WARNING("HcclThreadResGetInfoFunc dlHcommThreadResGetInfo not success.");
        // [中文导读] [AllReduce逐行 S60] 返回成功，未传播原始ret。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S61] 结束条件} else if (ret != HCCL_SUCCESS)。
    }
    // [中文导读] [AllReduce逐行 S62] 查询成功时输出真实Host保序流。
    resolvedStream = hostOrderStream;
    // [中文导读] [AllReduce逐行 S63] 以可选接口查询Host保序线程关联ACL stream处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S64] 结束OpLaunchGetHostOrderStream函数体。
}

// [中文导读] [AllReduce逐行 S66] 声明GetOrderLaunchModeName接口：将保序模式转换为日志名字。
static const char* GetOrderLaunchModeName(OrderLaunchMode mode)
// [中文导读] [AllReduce逐行 S67] 开始GetOrderLaunchModeName的函数体。
{
    // [中文导读] [AllReduce逐行 S68] ACL图捕获使用Aclgraph模式日志名字。
    if (mode == OrderLaunchMode::ORDER_LAUNCH_ACLGRAPH) {
        // [中文导读] [AllReduce逐行 S69] 返回Aclgraph字符串。
        return "Aclgraph";
    // [中文导读] [AllReduce逐行 S70] GE图执行使用GE模式日志名字。
    } else if (mode == OrderLaunchMode::ORDER_LAUNCH_GE) {
        // [中文导读] [AllReduce逐行 S71] 返回GE字符串。
        return "GE";
    // [中文导读] [AllReduce逐行 S72] 结束条件} else if (mode == OrderLaunchMode::ORDER_LAUNCH_GE)。
    }
    // [中文导读] [AllReduce逐行 S73] 其余模式使用Opbase名字，本例在此返回。
    return "Opbase";
// [中文导读] [AllReduce逐行 S74] 结束GetOrderLaunchModeName函数体。
}

// [中文导读] [AllReduce逐行 S76] 声明GetOrderLaunchHostThreadType接口：按OPBASE/GE/ACLGRAPH选择Host专用保序线程类型。
static HcclDedicatedThreadType GetOrderLaunchHostThreadType(OrderLaunchMode mode)
// [中文导读] [AllReduce逐行 S77] 开始GetOrderLaunchHostThreadType的函数体。
{
    // [中文导读] [AllReduce逐行 S78] GE模式选择GE专用Host保序线程。
    if (mode == OrderLaunchMode::ORDER_LAUNCH_GE) {
        // [中文导读] [AllReduce逐行 S79] 返回AICPU_ORDER_LAUNCH_GE线程类型。
        return HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_GE;
    // [中文导读] [AllReduce逐行 S80] ACLGRAPH模式选择ACL图专用Host保序线程。
    } else if (mode == OrderLaunchMode::ORDER_LAUNCH_ACLGRAPH) {
        // [中文导读] [AllReduce逐行 S81] 返回AICPU_ORDER_LAUNCH_ACLGRAPH线程类型。
        return HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_ACLGRAPH;
    // [中文导读] [AllReduce逐行 S82] 结束条件} else if (mode == OrderLaunchMode::ORDER_LAUNCH_ACLGRAPH)。
    }
    // [中文导读] [AllReduce逐行 S83] 其余模式选择OPBASE专用Host保序线程。
    return HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_OPBASE;
// [中文导读] [AllReduce逐行 S84] 结束GetOrderLaunchHostThreadType函数体。
}

// [中文导读] [AllReduce逐行 S86] 声明OpLaunchGetOrderStreams接口：取得保序和展开两个ACL stream并检查非空。
static HcclResult OpLaunchGetOrderStreams(
    // [中文导读] [AllReduce逐行 S87] 函数参数包含通信域句柄、Host展开线程句柄，本行延续接口声明。
    HcclComm comm, ThreadHandle hostOrderThread, ThreadHandle unfoldThread, aclrtStream& hostOrderStream,
    // [中文导读] [AllReduce逐行 S88] 函数参数包含aclrtStream& unfoldStream，本行延续接口声明。
    aclrtStream& unfoldStream)
// [中文导读] [AllReduce逐行 S89] 开始OpLaunchGetOrderStreams的函数体。
{
    // [中文导读] [AllReduce逐行 S90] 查询Host保序线程对应ACL stream。
    CHK_RET(OpLaunchGetHostOrderStream(hostOrderThread, hostOrderStream));
    // [中文导读] [AllReduce逐行 S91] 查询Host展开线程对应ACL stream。
    CHK_RET(OpLaunchGetUnfoldStream(comm, unfoldThread, unfoldStream));
    // [中文导读] [AllReduce逐行 S92] 开始条件检查宏，条件成立时记录错误并返回下方指定错误码。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S93] 任一查询流为空时返回运行时错误。
        hostOrderStream == nullptr || unfoldStream == nullptr,
        // [中文导读] [AllReduce逐行 S94] 输出错误日志，记录OpLaunchGetOrderStreams当前阶段和相关参数。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S95] 日志输出查询到的两条流地址，方便定位空流错误。
            "[%s] failed to get hostOrderStream[%p] or unfoldStream[%p]", __func__, hostOrderStream, unfoldStream),
        // [中文导读] [AllReduce逐行 S96] 提供上述日志的实参：HCCL_E_RUNTIME。
        HCCL_E_RUNTIME);
    // [中文导读] [AllReduce逐行 S97] 取得保序和展开两个ACL stream并检查非空处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S98] 结束OpLaunchGetOrderStreams函数体。
}

// [中文导读] [AllReduce逐行 S100] 声明AclgraphOrderLaunchEventToOrderStream接口：ACL图第一阶段：展开流RecordEvent，Host保序流等待。
static HcclResult AclgraphOrderLaunchEventToOrderStream(
    // [中文导读] [AllReduce逐行 S101] 函数参数包含通信域句柄、Host展开线程句柄，本行延续接口声明。
    HcclComm comm, ThreadHandle hostOrderThread, ThreadHandle unfoldThread, HcclRtEvent event)
// [中文导读] [AllReduce逐行 S102] 开始AclgraphOrderLaunchEventToOrderStream的函数体。
{
    // [中文导读] [AllReduce逐行 S103] 初始化ACL图第一阶段Host保序流地址。
    aclrtStream hostOrderStream = nullptr;
    // [中文导读] [AllReduce逐行 S104] 初始化ACL图第一阶段展开流地址。
    aclrtStream unfoldStream = nullptr;
    // [中文导读] [AllReduce逐行 S105] 同时获取Host保序流和展开流，任一为空则停止。
    CHK_RET(OpLaunchGetOrderStreams(comm, hostOrderThread, unfoldThread, hostOrderStream, unfoldStream));

    // [中文导读] [AllReduce逐行 S107] 在展开流记录第一阶段event。
    aclError retEvent = aclrtRecordEvent(event, unfoldStream);
    // [中文导读] [AllReduce逐行 S108] 开始条件检查宏，条件成立时记录错误并返回下方指定错误码。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S109] RecordEvent失败时打印ACL返回码并触发错误返回。
        retEvent != ACL_SUCCESS, HCCL_ERROR("[%s]aclrtRecordEvent failed, ret[%d]", __func__, retEvent),
        // [中文导读] [AllReduce逐行 S110] 补足错误检查宏返回值：HCCL_E_RUNTIME。
        HCCL_E_RUNTIME);
    // [中文导读] [AllReduce逐行 S111] 在Host保序流排入event等待，确保展开流前序步骤先完成。
    retEvent = aclrtStreamWaitEvent(hostOrderStream, event);
    // [中文导读] [AllReduce逐行 S112] 开始条件检查宏，条件成立时记录错误并返回下方指定错误码。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S113] WaitEvent失败时打印ACL返回码并触发错误返回。
        retEvent != ACL_SUCCESS, HCCL_ERROR("[%s]aclrtStreamWaitEvent failed, ret[%d]", __func__, retEvent),
        // [中文导读] [AllReduce逐行 S114] 补足错误检查宏返回值：HCCL_E_RUNTIME。
        HCCL_E_RUNTIME);
    // [中文导读] [AllReduce逐行 S115] ACL图第一阶段：展开流RecordEvent，Host保序流等待处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S116] 结束AclgraphOrderLaunchEventToOrderStream函数体。
}

// [中文导读] [AllReduce逐行 S118] 声明AclgraphOrderLaunchEventToKernelStream接口：ACL图第二阶段：Host保序流RecordEvent，展开流等待。
static HcclResult AclgraphOrderLaunchEventToKernelStream(
    // [中文导读] [AllReduce逐行 S119] 函数参数包含通信域句柄、Host展开线程句柄，本行延续接口声明。
    HcclComm comm, ThreadHandle hostOrderThread, ThreadHandle unfoldThread, HcclRtEvent event)
// [中文导读] [AllReduce逐行 S120] 开始AclgraphOrderLaunchEventToKernelStream的函数体。
{
    // [中文导读] [AllReduce逐行 S121] 初始化ACL图第二阶段Host保序流地址。
    aclrtStream hostOrderStream = nullptr;
    // [中文导读] [AllReduce逐行 S122] 初始化ACL图第二阶段展开流地址。
    aclrtStream unfoldStream = nullptr;
    // [中文导读] [AllReduce逐行 S123] 同时获取Host保序流和展开流，任一为空则停止。
    CHK_RET(OpLaunchGetOrderStreams(comm, hostOrderThread, unfoldThread, hostOrderStream, unfoldStream));

    // [中文导读] [AllReduce逐行 S125] 在Host保序流记录第二阶段event。
    aclError retEvent = aclrtRecordEvent(event, hostOrderStream);
    // [中文导读] [AllReduce逐行 S126] 开始条件检查宏，条件成立时记录错误并返回下方指定错误码。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S127] RecordEvent失败时打印ACL返回码并触发错误返回。
        retEvent != ACL_SUCCESS, HCCL_ERROR("[%s]aclrtRecordEvent failed, ret[%d]", __func__, retEvent),
        // [中文导读] [AllReduce逐行 S128] 补足错误检查宏返回值：HCCL_E_RUNTIME。
        HCCL_E_RUNTIME);
    // [中文导读] [AllReduce逐行 S129] 在展开流排入event等待，连接Host保序流顺序。
    retEvent = aclrtStreamWaitEvent(unfoldStream, event);
    // [中文导读] [AllReduce逐行 S130] 开始条件检查宏，条件成立时记录错误并返回下方指定错误码。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S131] WaitEvent失败时打印ACL返回码并触发错误返回。
        retEvent != ACL_SUCCESS, HCCL_ERROR("[%s]aclrtStreamWaitEvent failed, ret[%d]", __func__, retEvent),
        // [中文导读] [AllReduce逐行 S132] 补足错误检查宏返回值：HCCL_E_RUNTIME。
        HCCL_E_RUNTIME);
    // [中文导读] [AllReduce逐行 S133] ACL图第二阶段：Host保序流RecordEvent，展开流等待处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S134] 结束AclgraphOrderLaunchEventToKernelStream函数体。
}

/**
 * @brief 按序launch第一阶段：将通信算子按序launch到OrderStream。
 * OPBASE / ACLGRAPH / HCOMM 三模式统一入口，流程如下：
 * 1. 获取并导出Host侧保序流（hostOrderThread），写入param.exportHostOrderThread；
 * 2. 仅ACLGRAPH模式：在unfoldStream上record event，hostOrderStream等待该event，建立流间依赖；
 * 3. 获取Device侧保序流（deviceOrderThread），写入param.deviceOrderThread；
 * 4. hostOrderThread在unfoldThread上record notify(idx)，unfoldThread等待该notify，完成第一阶段同步。
 * @param comm          通信域
 * @param param         算子参数，函数会写入exportHostOrderThread和deviceOrderThread
 * @param unfoldThread  展开线程句柄
 * @param notifyIdx     notify索引
 * @param timeout       超时时间
 * @param mode          启动模式（OPBASE/ACLGRAPH/HCOMM）
 * @param event         ACLGRAPH模式使用的event，其他模式传nullptr
 * @return HcclResult
 */
// [中文导读] [AllReduce逐行 S152] 声明HcclOrderLaunchToOrderStream接口：第一阶段：取得Host/Device保序线程，Host保序流通知展开流。
HcclResult HcclOrderLaunchToOrderStream(
    // [中文导读] [AllReduce逐行 S153] 函数参数包含通信域句柄、算子参数、Host展开线程句柄，本行延续接口声明。
    HcclComm comm, OpParam& param, ThreadHandle unfoldThread, u32 notifyIdx, u32 timeout, OrderLaunchMode mode,
    // [中文导读] [AllReduce逐行 S154] 函数参数包含HcclRtEvent event，本行延续接口声明。
    HcclRtEvent event)
// [中文导读] [AllReduce逐行 S155] 开始HcclOrderLaunchToOrderStream的函数体。
{
    // [中文导读] [AllReduce逐行 S156] 取得当前OPBASE/GE/ACLGRAPH的日志名字。
    const char* modeName = GetOrderLaunchModeName(mode);
    // [中文导读] [AllReduce逐行 S157] 选择该模式的专用Host保序线程类型。
    HcclDedicatedThreadType hostThreadType = GetOrderLaunchHostThreadType(mode);

    // 1.1、获取Host侧保序流
    // [中文导读] [AllReduce逐行 S160] 声明Host保序线程句柄。
    ThreadHandle hostOrderThread;
    // [中文导读] [AllReduce逐行 S161] 声明Host保序线程导出到Device的句柄。
    ThreadHandle exportHostOrderThread;
    // [中文导读] [AllReduce逐行 S162] 运行时没有专用线程申请能力时跳过保序。
    if (!HcommIsSupportHcclDedicatedThreadAcquire()) {
        // [中文导读] [AllReduce逐行 S163] 把参数中的Host保序导出句柄清零，Device不发保序通知。
        param.exportHostOrderThread = 0;
        // [中文导读] [AllReduce逐行 S164] 把Device保序线程句柄清零，Device入口跳过保序通知。
        param.deviceOrderThread = 0;
        // [中文导读] [AllReduce逐行 S165] 输出警告日志，记录HcclOrderLaunchToOrderStream当前阶段和相关参数。
        HCCL_WARNING("[%s]. HcclDedicatedThreadAcquire not supported, %s OrderLaunch is skipped.", __func__, modeName);
        // [中文导读] [AllReduce逐行 S166] 专用线程能力缺失时当前保序阶段成功跳过。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S167] 结束条件if (!HcommIsSupportHcclDedicatedThreadAcquire())。
    }
    // [中文导读] [AllReduce逐行 S168] 获取当前模式的专用Host保序线程及其通知资源。
    CHK_RET(HcclDedicatedThreadAcquire(comm, hostThreadType, HOST_ORDER_THREAD_NOTIFY_NUM, &hostOrderThread));
    // [中文导读] [AllReduce逐行 S169] 输出运行日志，记录HcclOrderLaunchToOrderStream当前阶段和相关参数。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S170] 补充日志格式：[%s]. %s After HcclDedicatedThreadAcquire hostOrderThread [0x%llx]", __func__, modeName, hostOrderThread)。
        "[%s]. %s After HcclDedicatedThreadAcquire hostOrderThread [0x%llx]", __func__, modeName, hostOrderThread);
    // [中文导读] [AllReduce逐行 S171] 专用线程返回0表示无需该保序链，按源码规则成功跳过。
    if (hostOrderThread == 0) {
        // [中文导读] [AllReduce逐行 S172] 清除Host保序线程导出句柄。
        param.exportHostOrderThread = 0;
        // [中文导读] [AllReduce逐行 S173] 清除Device保序线程句柄。
        param.deviceOrderThread = 0;
        // [中文导读] [AllReduce逐行 S174] 输出运行日志，记录HcclOrderLaunchToOrderStream当前阶段和相关参数。
        HCCL_INFO(
            // [中文导读] [AllReduce逐行 S175] 补充日志格式：[%s]. Communication domains Number is less than cores Number, %s OrderLaunch is not Required.", __func__。
            "[%s]. Communication domains Number is less than cores Number, %s OrderLaunch is not Required.", __func__,
            // [中文导读] [AllReduce逐行 S176] 提供上述日志的实参：modeName。
            modeName);
        // [中文导读] [AllReduce逐行 S177] 无需保序时成功返回，后面不建立通知链。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S178] 结束条件if (hostOrderThread == 0)。
    }

    // 1.2、导出Host侧保序流
    // [中文导读] [AllReduce逐行 S181] 把Host保序线程导出到AICPU_TS，使Device入口可发通知。
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &hostOrderThread, COMM_ENGINE_AICPU_TS, &exportHostOrderThread));
    // [中文导读] [AllReduce逐行 S182] 把导出后的句柄放入OpParam供Device kernel读取。
    param.exportHostOrderThread = exportHostOrderThread;
    // [中文导读] [AllReduce逐行 S183] 输出运行日志，记录HcclOrderLaunchToOrderStream当前阶段和相关参数。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S184] 补充日志格式：[%s]. %s After HcclThreadExportToCommEngine hostOrderThread [0x%llx], exportHostOrderThread[0x%llx]", __func__。
        "[%s]. %s After HcclThreadExportToCommEngine hostOrderThread [0x%llx], exportHostOrderThread[0x%llx]", __func__,
        // [中文导读] [AllReduce逐行 S185] 提供上述日志的实参：modeName, hostOrderThread, exportHostOrderThread。
        modeName, hostOrderThread, exportHostOrderThread);

    // 2、仅 ACLGRAPH 模式需要在 unfoldStream 上 record event，hostOrderStream 等待 event
    // [中文导读] [AllReduce逐行 S188] 仅ACLGRAPH模式用第一阶段event建立额外流依赖。
    if (mode == OrderLaunchMode::ORDER_LAUNCH_ACLGRAPH) {
        // [中文导读] [AllReduce逐行 S189] 展开流RecordEvent，Host保序流等待同一event。
        CHK_RET(AclgraphOrderLaunchEventToOrderStream(comm, hostOrderThread, unfoldThread, event));
    // [中文导读] [AllReduce逐行 S190] 结束条件if (mode == OrderLaunchMode::ORDER_LAUNCH_ACLGRAPH)。
    }

    // 3、获取Device侧保序流
    // [中文导读] [AllReduce逐行 S193] 声明Device专用保序线程句柄。
    ThreadHandle deviceOrderThread;
    // [中文导读] [AllReduce逐行 S194] 开始取得Device专用保序线程。
    CHK_RET(HcclDedicatedThreadAcquire(
        // [中文导读] [AllReduce逐行 S195] 请求AICPU_ORDER_LAUNCH_DEVICE线程及其通知容量。
        comm, HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE, DEVICE_ORDER_THREAD_NOTIFY_NUM, &deviceOrderThread));
    // [中文导读] [AllReduce逐行 S196] Device专用保序线程句柄为0时跳过后续保序。
    if (deviceOrderThread == 0) {
        // [中文导读] [AllReduce逐行 S197] 清除已经写入的Host导出保序句柄。
        param.exportHostOrderThread = 0;
        // [中文导读] [AllReduce逐行 S198] 清除Device保序句柄。
        param.deviceOrderThread = 0;
        // [中文导读] [AllReduce逐行 S199] 输出运行日志，记录HcclOrderLaunchToOrderStream当前阶段和相关参数。
        HCCL_INFO(
            // [中文导读] [AllReduce逐行 S200] 补充日志格式：[%s]. HcclDedicatedThreadAcquire unable to obtain deviceOrderThread, %s OrderLaunch is not Required.。
            "[%s]. HcclDedicatedThreadAcquire unable to obtain deviceOrderThread, %s OrderLaunch is not Required.",
            // [中文导读] [AllReduce逐行 S201] 提供上述日志的实参：__func__, modeName。
            __func__, modeName);
        // [中文导读] [AllReduce逐行 S202] 无Device保序线程时成功跳过当前保序阶段。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S203] 结束条件if (deviceOrderThread == 0)。
    }
    // [中文导读] [AllReduce逐行 S204] 保存Device专用保序线程句柄到OpParam。
    param.deviceOrderThread = deviceOrderThread;
    // [中文导读] [AllReduce逐行 S205] 输出运行日志，记录HcclOrderLaunchToOrderStream当前阶段和相关参数。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S206] 补充日志格式：[%s]. %s After HcclDedicatedThreadAcquire deviceOrderThread [0x%llx]", __func__, modeName, deviceOrderThread)。
        "[%s]. %s After HcclDedicatedThreadAcquire deviceOrderThread [0x%llx]", __func__, modeName, deviceOrderThread);

    // 4、notify0
    // [中文导读] [AllReduce逐行 S209] 在Host保序线程向Host展开线程记录第一阶段通知。
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyRecordOnThread(hostOrderThread, unfoldThread, notifyIdx)));
    // [中文导读] [AllReduce逐行 S210] Host展开线程等待第一阶段通知，保护后续kernel发射顺序。
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(unfoldThread, notifyIdx, timeout)));
    // [中文导读] [AllReduce逐行 S211] 输出运行日志，记录HcclOrderLaunchToOrderStream当前阶段和相关参数。
    HCCL_INFO("[%s]. %s OrderLaunch Phase1 Success, timeout[%u].", __func__, modeName, timeout);
    // [中文导读] [AllReduce逐行 S212] 第一阶段：取得Host/Device保序线程，Host保序流通知展开流处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S213] 结束HcclOrderLaunchToOrderStream函数体。
}

/**
 * @brief 按序launch第二阶段：将通信算子按序launch到KernelStream。
 * OPBASE / ACLGRAPH / HCOMM 三模式统一入口，流程如下：
 * 1. 获取Host侧保序流（hostOrderThread）；
 * 2. hostOrderThread等待第一阶段record的notify(idx)，完成第二阶段同步；
 * 3. 仅ACLGRAPH模式：在hostOrderStream上record event，unfoldStream等待该event，建立流间依赖。
 * @param comm          通信域
 * @param unfoldThread  展开线程句柄（仅ACLGRAPH模式使用）
 * @param notifyIdx     notify索引
 * @param timeout       超时时间
 * @param mode          启动模式（OPBASE/ACLGRAPH/HCOMM）
 * @param event         ACLGRAPH模式使用的event，其他模式传nullptr
 * @return HcclResult
 */
// [中文导读] [AllReduce逐行 S229] 声明HcclOrderLaunchToKernelStream接口：第二阶段：Host保序流等待Device入口已进入展开的通知。
HcclResult HcclOrderLaunchToKernelStream(
    // [中文导读] [AllReduce逐行 S230] 函数参数包含通信域句柄、Host展开线程句柄，本行延续接口声明。
    HcclComm comm, ThreadHandle unfoldThread, u32 notifyIdx, u32 timeout, OrderLaunchMode mode, HcclRtEvent event)
// [中文导读] [AllReduce逐行 S231] 开始HcclOrderLaunchToKernelStream的函数体。
{
    // [中文导读] [AllReduce逐行 S232] 取得本次模式日志名字。
    const char* modeName = GetOrderLaunchModeName(mode);
    // [中文导读] [AllReduce逐行 S233] 选择OPBASE/GE/ACLGRAPH对应Host保序线程类型。
    HcclDedicatedThreadType hostThreadType = GetOrderLaunchHostThreadType(mode);
    // [中文导读] [AllReduce逐行 S234] 输出运行日志，记录HcclOrderLaunchToKernelStream当前阶段和相关参数。
    HCCL_INFO("%s OrderLaunch Phase2 Start, Comm[%p], timeout[%u].", modeName, comm, timeout);

    // 1、获取Host侧保序流
    // [中文导读] [AllReduce逐行 S237] 声明第二阶段Host保序线程句柄。
    ThreadHandle hostOrderThread;
    // [中文导读] [AllReduce逐行 S238] 没有专用线程申请能力时第二阶段成功跳过。
    if (!HcommIsSupportHcclDedicatedThreadAcquire()) {
        // [中文导读] [AllReduce逐行 S239] 输出警告日志，记录HcclOrderLaunchToKernelStream当前阶段和相关参数。
        HCCL_WARNING("[%s]. HcclDedicatedThreadAcquire not supported, %s OrderLaunch is skipped.", __func__, modeName);
        // [中文导读] [AllReduce逐行 S240] 缺少能力时返回成功，不执行后续保序通知等待。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S241] 结束条件if (!HcommIsSupportHcclDedicatedThreadAcquire())。
    }
    // [中文导读] [AllReduce逐行 S242] 取得第一阶段使用的同类型专用Host保序线程。
    CHK_RET(HcclDedicatedThreadAcquire(comm, hostThreadType, HOST_ORDER_THREAD_NOTIFY_NUM, &hostOrderThread));
    // [中文导读] [AllReduce逐行 S243] 输出运行日志，记录HcclOrderLaunchToKernelStream当前阶段和相关参数。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S244] 补充日志格式：[%s]. %s After HcclDedicatedThreadAcquire hostOrderThread [0x%llx]", __func__, modeName, hostOrderThread)。
        "[%s]. %s After HcclDedicatedThreadAcquire hostOrderThread [0x%llx]", __func__, modeName, hostOrderThread);
    // [中文导读] [AllReduce逐行 S245] 取得的Host保序线程句柄为0表示无需保序。
    if (hostOrderThread == 0) {
        // [中文导读] [AllReduce逐行 S246] 输出运行日志，记录HcclOrderLaunchToKernelStream当前阶段和相关参数。
        HCCL_INFO(
            // [中文导读] [AllReduce逐行 S247] 补充日志格式：[%s]. Communication domains Number is less than cores Number, %s OrderLaunch is not Required.", __func__。
            "[%s]. Communication domains Number is less than cores Number, %s OrderLaunch is not Required.", __func__,
            // [中文导读] [AllReduce逐行 S248] 提供上述日志的实参：modeName。
            modeName);
        // [中文导读] [AllReduce逐行 S249] 无需保序时成功返回。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S250] 结束条件if (hostOrderThread == 0)。
    }

    // 2、wait notify1
    // [中文导读] [AllReduce逐行 S253] Host保序线程等待Device入口发来的第二阶段通知。
    CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(hostOrderThread, notifyIdx, timeout)));

    // 3、仅 ACLGRAPH 模式需要在 hostOrderStream 上 record event，unfoldStream 等待 event
    // [中文导读] [AllReduce逐行 S256] 仅ACLGRAPH模式额外建立Host保序流到展开流的event依赖。
    if (mode == OrderLaunchMode::ORDER_LAUNCH_ACLGRAPH) {
        // [中文导读] [AllReduce逐行 S257] Host保序流RecordEvent，展开流等待该event。
        CHK_RET(AclgraphOrderLaunchEventToKernelStream(comm, hostOrderThread, unfoldThread, event));
    // [中文导读] [AllReduce逐行 S258] 结束条件if (mode == OrderLaunchMode::ORDER_LAUNCH_ACLGRAPH)。
    }

    // [中文导读] [AllReduce逐行 S260] 输出运行日志，记录HcclOrderLaunchToKernelStream当前阶段和相关参数。
    HCCL_INFO("[%s]. %s OrderLaunch Phase2 Success.timeout[%u].", __func__, modeName, timeout);
    // [中文导读] [AllReduce逐行 S261] 第二阶段：Host保序流等待Device入口已进入展开的通知处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S262] 结束HcclOrderLaunchToKernelStream函数体。
}
} // namespace ops_hccl

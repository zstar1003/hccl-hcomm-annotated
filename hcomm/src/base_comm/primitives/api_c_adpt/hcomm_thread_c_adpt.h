/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_THREAD_C_ADPT_H
#define HCOMM_THREAD_C_ADPT_H

#include "hcomm_res_defs.h"
#include "hccl/hccl_res.h"
#include "thread.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * @brief 基于已有 rtStream 分配单个通信线程，写入统一映射表并返回句柄
 * @param[in] engine 通信引擎类型（如 COMM_ENGINE_CPU_TS）
 * @param[in] stream 已有的 rtStream 句柄
 * @param[in] notifyNum 该线程所需的 notify 资源数量
 * @param[out] thread 输出的线程句柄，失败时不会被修改
 * @return HcommResult 成功返回 HCCL_SUCCESS，失败返回对应错误码
 */
HcommResult HcommThreadAllocWithStream(CommEngine engine, rtStream_t stream, uint32_t notifyNum, ThreadHandle* thread);

/**
 * @brief 批量补充线程 notify 资源；AICPU 引擎额外触发 device 侧批量 kernel launch
 * @param[in] engine 通信引擎类型，仅 COMM_ENGINE_AICPU 触发 device kernel launch
 * @param[in] handles 线程句柄数组
 * @param[in] threadNum 线程句柄数量
 * @param[in] supplementNotifyNums 每个线程需补充的 notify 增量数量数组，长度须为 threadNum
 * @return HcommResult 成功返回 HCCL_SUCCESS，失败返回对应错误码
 */
HcommResult HcommThreadSupplementNotify(
    CommEngine engine, ThreadHandle* handles, uint32_t threadNum, uint32_t* supplementNotifyNums);

/**
 * @brief 查询单个线程当前已分配的 notify 数量
 * @param[in] thread 线程句柄
 * @param[out] notifyNum 输出的 notify 数量
 * @return HcommResult 成功返回 HCCL_SUCCESS，失败返回对应错误码
 */
HcommResult HcommThreadGetNotifyNum(ThreadHandle thread, uint32_t* notifyNum);

/**
 * @brief 跨引擎导出线程句柄。按目标引擎分派：
 *        - CPU/CCU 方向：查 device→host 映射表，返回对应的 host 句柄
 *        - AICPU 方向：先查 FindThreadByCommEngine，未命中则批量 kernel launch 创建并写入映射表
 * @param[in] hostHandles 源线程句柄数组
 * @param[in] commId 通信域标识字符串，AICPU 路径用于 kernel launch；可为空
 * @param[in] threadNum 线程句柄数量
 * @param[in] dstEngine 目标引擎类型
 * @param[out] outDeviceHandles 导出的句柄数组，长度须为 threadNum
 * @return HcommResult 成功返回 HCCL_SUCCESS，失败返回对应错误码
 */
HcommResult HcommThreadExportToCommEngine(
    ThreadHandle* hostHandles, const char* commId, uint32_t threadNum, CommEngine dstEngine,
    ThreadHandle* outDeviceHandles);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // HCOMM_THREAD_C_ADPT_H

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_RES_H
#define HCCL_RES_H

#include <stdint.h>
#include "acl/acl_rt.h"
#include "hccl_types.h"
#include "hcomm_res_defs.h"
#include "hccl_channel.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/// HCCL资源标识最大长度（字节）
const uint32_t HCCL_RES_TAG_MAX_LEN = 255;

/**
 * @name 通信内存获取
 * @{
 */
/**
 * @brief 获取通信域中的Hccl缓存(HCCL Buffer)
 * @param[in] comm 通信域句柄
 * @param[out] buffer Hccl缓存地址
 * @param[out] size Hccl缓存大小
 * @return HcclResult 执行结果状态码
 * @warning 重要约束：返回的buffer内存由库内管理，调用者严禁释放
 */
extern HcclResult HcclGetHcclBuffer(HcclComm comm, void** buffer, uint64_t* size);

/**
 * @name 清零通信内存获取
 * @{
 */
/**
 * @brief 获取通信域中的已清零Hccl缓存(HCCL Buffer)
 * @param[in] comm 通信域句柄
 * @param[out] buffer Hccl缓存地址
 * @param[out] size Hccl缓存大小
 * @return HcclResult 执行结果状态码
 * @warning 重要约束：返回的buffer内存由库内管理，调用者严禁释放
 */
extern HcclResult HcclGetHcclBufferCleared(HcclComm comm, void** buffer, uint64_t* size);

/**
 * @defgroup 通信引擎资源管理
 * @{
 */

/**
 * @brief 获取通信线程资源
 *
 * @param[in] comm 通信域句柄
 * @param[in] engine 通信引擎类型
 * @param[in] threadNum 线程数量
 * @param[in] notifyNumPerThread 每线程的通知数量
 * @param[out] threads 返回的线程句柄
 * @return HcclResult 执行结果状态码
 */
extern HcclResult HcclThreadAcquire(
    HcclComm comm, CommEngine engine, uint32_t threadNum, uint32_t notifyNumPerThread, ThreadHandle* threads);

/**
 * @brief 获取通信线程资源
 *
 * @param[in] comm 通信域句柄
 * @param[in] engine 通信引擎类型
 * @param[in] threadNum 线程数量
 * @param[in] type 线程类型
 * @param[in] config 每线程的config
 * @param[out] threads 返回的线程句柄
 * @return HcclResult 执行结果状态码
 * @note config 的大小需要保证与 threadNum 长度一致
 */
extern HcclResult HcclThreadAcquireWithConfig(
    HcclComm comm, CommEngine engine, uint32_t threadNum, ThreadType type, const ThreadConfig* config,
    ThreadHandle* threads);

/**
 * @brief 基于已有rts stream获取指定notifyNum的通信线程资源
 * @param[in] comm 通信域句柄
 * @param[in] engine 通信引擎类型
 * @param[in] stream stream句柄
 * @param[in] notifyNum 通知数量
 * @param[out] thread 返回的线程句柄
 * @note 当前适用于CPU_TS场景
 * @return HcclResult 执行结果状态码
 */
extern HcclResult HcclThreadAcquireWithStream(
    HcclComm comm, CommEngine engine, aclrtStream stream, uint32_t notifyNum, ThreadHandle* thread);

typedef enum {
    HCCL_DED_THREAD_TYPE_INVALID = -1,
    HCCL_DED_THREAD_TYPE_AICPU_LAUNCH = 0,
    HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE = 1,
    HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_OPBASE = 2,
    HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_ACLGRAPH = 3,
    HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_GE = 4,
    HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE = 5
} HcclDedicatedThreadType;

/**
 * @brief 申请专用thread
 * @param[in] comm 通信域句柄
 * @param[in] useType 专用线程使用类型(HCCL_DED_THREAD_TYPE_AICPU_LAUNCH等)
 * @param[in] notifyNumPerThread 每线程的通知数量
 * @param[out] thread 返回的线程句柄
 * @return HcclResult 执行结果状态码
 * @note 内部逻辑：先判断通信域中是否已存在该useType对应的专用线程，
 *       若存在则直接返回，不存在则调用HcclThreadAcquire创建并缓存(图模式下不创建直接返回0)
 */
extern HcclResult HcclDedicatedThreadAcquire(
    HcclComm comm, HcclDedicatedThreadType useType, uint32_t notifyNumPerThread, ThreadHandle* thread);

/** @} */ // 通信引擎资源管理

/**
 * @defgroup 通信引擎上下文管理接口（编程控制面可选接口）
 * @{
 */

/**
 * @brief 创建算子通信引擎上下文
 * @param[in] comm 通信域句柄
 * @param[in] ctxTag 引擎标签（最大字符长度为HCCL_RES_TAG_MAX_LEN）
 * @param[in] engine 通信引擎类型
 * @param[in] size ctx内存大小
 * @param[out] ctx 通信引擎上下文
 * @return HcclResult 执行结果状态码
 */
extern HcclResult HcclEngineCtxCreate(HcclComm comm, const char* ctxTag, CommEngine engine, uint64_t size, void** ctx);

/**
 * @brief 获取算子通信引擎上下文
 * @param[in] comm 通信域句柄
 * @param[in] ctxTag 引擎标签（最大字符长度为HCCL_RES_TAG_MAX_LEN）
 * @param[in] engine 通信引擎类型
 * @param[out] ctx 通信引擎上下文
 * @param[out] size ctx内存大小
 * @return HcclResult 执行结果状态码
 * @note 使用者可先查询ctx是否已存在，再决定是否重新申请ctx地址
 */
extern HcclResult HcclEngineCtxGet(HcclComm comm, const char* ctxTag, CommEngine engine, void** ctx, uint64_t* size);

/**
 * @brief 拷贝算子通信引擎上下文
 * @param[in] comm 通信域句柄
 * @param[in] engine 通信引擎类型
 * @param[in] ctxTag 引擎标签（最大字符长度为HCCL_RES_TAG_MAX_LEN）
 * @param[in] srcCtx 拷贝的源引擎
 * @param[in] size 拷贝的ctx内存大小
 * @param[in] dstCtxOffset 拷贝的ctx地址偏移
 * @return HcclResult 执行结果状态码
 * @note 1、目标ctx通过ctxTag获取
 */
extern HcclResult HcclEngineCtxCopy(
    HcclComm comm, CommEngine engine, const char* ctxTag, const void* srcCtx, uint64_t size, uint64_t dstCtxOffset);

/**
 * @brief 销毁通信引擎资源上下文
 * @param[in] comm 通信域句柄
 * @param[in] ctxTag 引擎标签（最大字符长度为HCCL_RES_TAG_MAX_LEN）
 * @param[in] engine 通信引擎类型
 * @return HcclResult 执行结果状态码
 */
extern HcclResult HcclEngineCtxDestroy(HcclComm comm, const char* ctxTag, CommEngine engine);

/**
 * @brief 向通信域注册内存
 * @param[in] comm 通信域句柄
 * @param[in] memTag 内存字符串标签，以"\0"结尾，最大字符长度为HCCL_RES_TAG_MAX_LEN
 * @param[in] mem 内存信息
 * @param[out] memHandle 注册内存句柄
 * @return HcclResult 执行结果状态码
 * @note 通信域内以memTag作为key存储该内存。
 * @warning
 */
extern HcclResult HcclCommMemReg(HcclComm comm, const char* memTag, const CommMem* mem, HcclMemHandle* memHandle);

typedef aclrtStream ThreadResTypeStream;

/**
 * @brief 获取Thread底层资源信息
 * @param[in] comm 通信域句柄
 * @param[in] thread 线程句柄(软件抽象)
 * @param[in] resType 底层资源类型(如Stream)
 * @param[in] infoLen 目标资源信息长度
 * @param[out] info 资源信息输出缓冲区
 * @warning 调用者必须确保
 *   1. infoLen参数必须等于目标资源类型的大小
 *   2. info缓冲区必须按资源类型对齐且可写
 * @code {.c}
 * ThreadResTypeStream stream;
 * uint32_t size = sizeof(ThreadResTypeStream); // 必须等于目标类型大小
 * HcclThreadResGetInfo(comm, thread, ThreadResType::THREAD_RES_TYPE_STREAM, size, &stream);
 * @endcode
 * @return HcclResult 执行结果状态码
 */
extern HcclResult
HcclThreadResGetInfo(HcclComm comm, ThreadHandle thread, ThreadResType resType, uint32_t infoLen, void** info);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif

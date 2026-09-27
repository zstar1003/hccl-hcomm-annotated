/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_HCCL_PRIMITIVE_LOCAL_H
#define HCOMM_HCCL_PRIMITIVE_LOCAL_H
#include <stdint.h>
#include <acl/acl.h>
#include "hccl_primitive.h"
#include "hccl_mem.h"
#include "stream_pub.h"
#include <string>
#include <vector>
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// 本地通信原语
/**
 * @brief 本地数据拷贝操作
 * @param streamHandle 异步流句柄
 * @param dst 目标缓冲区
 * @param src 源缓冲区
 * @return 执行状态码 HcclResult
 */
extern HcclResult HcclLocalCopy(StreamHandle streamHandle, HcclBuf* dst, HcclBuf* src);

/**
 * @brief 本地拷贝并规约操作
 * @param[in] streamHandle 异步流句柄
 * @param[in] dst 目标缓冲区
 * @param[in] src 源缓冲区
 * @param[in] reduceInfo 规约操作信息
 * @return 执行状态码 HcclResult
 */
extern HcclResult HcclLocalCopyReduce(StreamHandle streamHandle, HcclBuf* dst, HcclBuf* src, HcclReduceInfo reduceInfo);

/**
 * @brief 进行task下发
 * @param stream 指定的stream
 * @param subStreams 指定的子stream
 * @return HcclResult 返回HCCL_SUCCESS
 */

extern HcclResult HcclLocalLaunchTaskExtend(aclrtStream& stream, std::vector<aclrtStream>& subStreams);

/**
 * @brief 初始化task
 * @param stream 指定的stream
 * @param enableCache 是否使能cache
 * @param key 指定的key
 * @param useGraphConstructorV2 是否使用GraphConstructorV2
 */
extern HcclResult HcclLocalInitTask(
    aclrtStream stream, const bool enableCache, const std::string& key, bool useGraphConstructorV2 = false);

/**
 * @brief 记录通知事件（生产者）
 * @param[in] streamHandle 异步流句柄
 * @param[in] notify 通知句柄
 * @return 执行状态码 HcclResult
 */
extern HcclResult HcclLocalNotifyRecord(StreamHandle streamHandle, aclrtNotify notify);

/**
 * @brief 等待通知事件（消费者）
 * @param[in] streamHandle 异步流句柄
 * @param[in] notify 通知句柄
 * @param[in] timeOut 超时时间
 * @return 执行状态码 HcclResult
 */
extern HcclResult HcclLocalNotifyWait(StreamHandle streamHandle, aclrtNotify notify, const uint32_t timeOut);

extern HcclResult HcclTaskPrepare(char* key, uint32_t keyLen);

extern HcclResult HcclTaskLaunch(hccl::Stream* streams, uint32_t streamNum);

/**
 * @brief 记录通知事件（生产者）
 * @param[in] streamHandle 异步流句柄
 * @param[in] dstNotifyId 通知id
 * @return 执行状态码 HcclResult
 */
extern HcclResult HcclLocalBareNotifyRecord(StreamHandle streamHandle, uint64_t dstNotifyId);

/**
 * @brief 等待通知事件（消费者）
 * @param[in] streamHandle 异步流句柄
 * @param[in] notifyId 通知id
 * @param[in] timeOut 超时时间
 * @return 执行状态码 HcclResult
 */
extern HcclResult HcclLocalBareNotifyWait(StreamHandle streamHandle, uint64_t notifyId, uint32_t timeOut);

extern HcclResult HcclTaskClear(std::string key); // host ffts+使用
#ifdef __cplusplus
}
#endif // __cplusplus
#endif

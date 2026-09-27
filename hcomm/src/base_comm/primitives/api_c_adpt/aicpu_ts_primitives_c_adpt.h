/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_API_DATA_AICPU_TS_H
#define HCCL_API_DATA_AICPU_TS_H

bool IsBatchLaunchMode();
uint32_t GetSqFullTimeOut();
HcclResult HandleDispatchAllStreams();

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
/**
 * @brief NPU上查询 rtsq任务执行完成的接口（阻塞）
 * @param[in] thread NPU上执行的线程句柄
 * @param[in] timeout 超时时间(秒)
 * @return int32_t 执行结果状态码
 *
 * WARNING: experimental API, No compatibility is currently guaranteed for this API
 */
extern int32_t HcommThreadJoin(ThreadHandle thread, uint32_t timeout);
#ifdef __cplusplus
}
#endif // __cplusplus

#endif // HCCL_API_DATA_AICPU_TS_H

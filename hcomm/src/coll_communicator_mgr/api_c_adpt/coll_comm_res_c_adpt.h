/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_COMM_RES_C_ADPT_H
#define COLL_COMM_RES_C_ADPT_H

#include <cstdint>
#include "hccl/hccl_res.h"

namespace hccl {
class hcclComm;
} // namespace hccl

#ifdef __cplusplus
extern "C" {
#endif

HcclResult
ProcessHcclResPackReq(const HcclChannelDesc& channelDesc, HcclChannelDesc& channelDescFinal, hccl::hcclComm* hcclComm);

HcclResult
ProcessHcclChannelDesc(const HcclChannelDesc& channelDesc, HcclChannelDesc& channelDescFinal, hccl::hcclComm* hcclComm);

HcclResult HcclGroupStart();
HcclResult HcclGroupEnd();

/**
 * @note 职责：集合通信的通信域资源管理的C接口声明（暂未对外的接口）
 */

/**
 * @note 非对外接口声明示例
 * @code {.c}
 * extern HcclResult HcclThreadAcquire(HcclComm comm, CommEngine engine, uint32_t threadNum,
 *     uint32_t notifyNumPerThread, ThreadHandle *threads);
 * @endcode
 */

#ifdef __cplusplus
}
#endif

#endif // COLL_COMM_RES_C_ADPT_H

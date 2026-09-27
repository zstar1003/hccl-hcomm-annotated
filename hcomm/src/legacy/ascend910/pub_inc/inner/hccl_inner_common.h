/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_INNER_COMMON_H
#define HCCL_INNER_COMMON_H

namespace hccl {
constexpr u64 DEVICE_MEM_MAX_COUNT = 0x1000000000; // device申请内存最大值 64GB
constexpr u64 HOST_MEM_MAX_COUNT = 0x10000000000;  // host申请内存最大值 1TB

constexpr u32 MAX_DEV_NUM = 32;
constexpr u32 DEFAULT_DEVICE_LOGIC_ID = MAX_DEV_NUM - 1;

enum class RmaType { IPC_RMA = 0, RDMA_RMA, RMA_TYPE_RESERVED };
} // namespace hccl
#endif // HCCL_INNER_COMMON_H

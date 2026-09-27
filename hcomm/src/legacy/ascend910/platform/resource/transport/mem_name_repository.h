/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEM_NAME_REPOSITORY_H
#define MEM_NAME_REPOSITORY_H

#include "mem_name_repository_pub.h"

namespace hccl {
constexpr u32 HCCL_IPC_PID_ARRAY_SIZE = 1; // 固定每次只传一个PID数据
constexpr u32 MAX_DEV_NUM_IPC_MEM = 16;
constexpr int32_t INVALID_PAGESIZE = -1;
} // namespace hccl
#endif /* * __MEM_NAME_REPOSITRY_H__ */

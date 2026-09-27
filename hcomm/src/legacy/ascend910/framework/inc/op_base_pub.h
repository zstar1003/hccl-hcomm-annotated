/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_BASE_PUB_H
#define OP_BASE_PUB_H

#include <vector>
#include <hccl/hccl_types.h>
#include <hccl/hccl_comm.h>
#include <hccl/hccl_inner.h>

#include "hccl/base.h"

struct OpBaseMemPara {
    u64 beginIndex;
    u64 count;
    u64 tmpMemSize;
};

struct GatherPara {
    std::vector<u64> addrInfo;
    std::vector<u64> addrInfoCountPerRank;
    u32 rankSize;
    s32 addrLength;
};
#endif // OP_BASE_PUB_H

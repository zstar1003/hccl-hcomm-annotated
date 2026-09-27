/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_ONE_SIDED_DATA_H
#define HCCL_ONE_SIDED_DATA_H
#include "hccl/base.h"
#include "hccl_mem_defs.h"

const u32 HCCL_MEM_DESC_LENGTH = 511;

struct HcclMemDesc {
    char desc[HCCL_MEM_DESC_LENGTH + 1]; // 具体内容对调用者不可见
};

struct HcclMemDescs {
    HcclMemDesc* array;
    u32 arrayLength;
};

struct HcclOneSideOpDesc {
    void* localAddr;  // 本端VA
    void* remoteAddr; // 远端VA
    u64 count;
    HcclDataType dataType;
};

constexpr size_t TRANSPORT_EMD_ESC_SIZE = 512U - (sizeof(u32) * 2);
struct RmaMemDesc {
    s32 localRankId;
    s32 remoteRankId;
    char memDesc[TRANSPORT_EMD_ESC_SIZE];
};

#endif // HCCL_ONE_SIDED_DATA_H

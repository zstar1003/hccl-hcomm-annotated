/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_COMMUNICATOR_DPU_TYPES_H
#define HCCL_COMMUNICATOR_DPU_TYPES_H

#include <string>
#include <cstdint>
#include "types.h"

struct DpuKernelLaunchParam {
    u64 memorySize{0};
    void* shareHBM{nullptr};
    void* hostMem{nullptr};
    uint32_t deviceId{UINT32_MAX};
    std::string commId;
    void* taskexceptionVa{nullptr};
};

// Dpu Kernel Launch 申请的共享内存
struct DpuShmem {
    void* originVa_{nullptr}; // ub场景时申请的原始地址
    void* va_{nullptr};       // 申请的地址,pcie场景为原始地址,ub场景为4k对齐后的地址
    void* accessVA_{nullptr}; // 注册后映射的地址
    int64_t connectType_{0};  // 连接类型
};

#endif // HCCL_COMMUNICATOR_DPU_TYPES_H

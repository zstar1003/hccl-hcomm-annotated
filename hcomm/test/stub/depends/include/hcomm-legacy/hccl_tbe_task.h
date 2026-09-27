/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_TBE_TASK_H
#define HCCL_TBE_TASK_H

#include <hccl/hccl_types.h>
#include <hccl/base.h>
#include "acl/acl_base.h"

constexpr u32 TBE_MAX_MODULE_DEVICE_NUM = 32; // 单server双模组时支持最大的设备数量
struct TbeReduceParam {
    void* src1{nullptr};
    void* src2{nullptr};
    void* dst{nullptr};
    uint64_t count{0};
    HcclDataType dataType{HCCL_DATA_TYPE_RESERVED};
    HcclReduceOp redOp{HCCL_REDUCE_RESERVED};
};

struct TbeReduceArg {
    uint32_t blockDim{0};
    void* funcAddr{nullptr};
    void* addrListDevMem{nullptr};
    void* argsHandle{nullptr};
};

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

extern HcclResult HcclTbeTaskInit(int32_t deviceLogicId);
extern HcclResult HcclTbeTaskDeInit(int32_t deviceLogicId);
extern HcclResult
HcclTbeMemClean(int64_t addrList[], int64_t sizeList[], uint32_t count, aclrtStream stream, int32_t deviceLogicId);
extern HcclResult HcclGetVectorBlockSize(uint32_t* blockSize, int32_t deviceLogicId);
extern HcclResult HcclTbeReduce(
    const TbeReduceParam* param, aclrtStream stream, void* overflowAddrs[], uint32_t overflowCount,
    int32_t deviceLogicId);
extern HcclResult HcclTbeReduceGenArgs(
    const TbeReduceParam* param, aclrtStream stream, void* overflowAddrs[], uint32_t overflowCount, TbeReduceArg* args,
    int32_t deviceLogicId);
#ifdef __cplusplus
}
#endif // __cplusplus
#endif // HCCL_TBE_TASK_H

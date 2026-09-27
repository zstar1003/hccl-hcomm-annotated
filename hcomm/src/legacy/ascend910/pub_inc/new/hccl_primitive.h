/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_PRIMITIVE_H
#define HCCL_PRIMITIVE_H

#include <stdint.h>
#include <acl/acl.h>
#include "hccl_mem.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef void* StreamHandle;

/**
 * @struct HcclReduceInfo
 * @brief 规约操作元数据描述结构体
 * @var dataType - 规约操作数据类型
 * @var reduceOp - 规约操作类型
 */
typedef struct {
    HcclDataType dataType;
    HcclReduceOp reduceOp;
} HcclReduceInfo;

/**
 * @struct HcclBufPair
 * @brief 缓冲区对结构体
 * @var loc - 本地缓冲区描述
 * @var rmt - 对应的远端缓冲区描述
 */
typedef struct {
    HcclBuf loc;
    HcclBuf rmt;
} HcclBufPair;

/**
 * @struct HcclBatchTransferInfo
 * @brief 批量传输操作元数据描述结构体
 * @var bufPairs   - 缓冲区对信息
 * @var transType  - 传输操作类型
 * @var reduceInfo - 规约操作信息
 */
typedef struct {
    HcclBufPair bufPairs;
    // TransferType transType;
    HcclReduceInfo reduceInfo;
} HcclBatchTransferInfo;
#ifdef __cplusplus
}
#endif // __cplusplus
#endif

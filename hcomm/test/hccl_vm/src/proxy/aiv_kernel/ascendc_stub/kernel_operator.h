/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_KERNEL_OPERATOR_H
#define AIV_KERNEL_OPERATOR_H

#include "ascendc_base_stub.h"
#include "ascendc_copy_stub.h"
#include "ascendc_math_stub.h"
#include "ascendc_memory_stub.h"
#include "ascendc_sync_stub.h"
#include "ascendc_utils_stub.h"
#include "hccl_types.h"

enum KernelType {
    K_TYPE_AICORE = 1,       // c100/m200
    K_TYPE_AIC = 2,          // v220-cube
    K_TYPE_AIV = 3,          // v220-vec
    K_TYPE_MIX_AIC_MAIN = 4, // v220 mix cube/vector 1:2
    K_TYPE_MIX_AIV_MAIN = 5, // v220 mix vector/cube 1:2
    K_TYPE_AIC_ROLLBACK = 6, // v220-cube，aic rollback
    K_TYPE_AIV_ROLLBACK = 7, // v220-vec，aiv rollback
    K_TYPE_MAX
};

struct BaseTlv { // TLV头部定义
    unsigned short type;
    unsigned short len;
};

enum FuncMetaType {                       // 函数级TLV类型
    F_TYPE_KTYPE = 1,                     // kernel type tlv
    F_TYPE_CROSS_CORE_SYNC = 2,           // cross core sync
    F_TYPE_MIX_TASK_RATION = 3,           // MIX CORE TYPE
    F_TYPE_L0_EXCEPTION_DFX = 4,          // DFX tlv for header
    F_TYPE_L0_EXCEPTION_DFX_ARGSINFO = 5, // DFX tlv for args info
    F_TYPE_L0_EXCEPTION_DFX_IS_TIK = 6,   // DFX tlv mark for TIK
    F_TYPE_MAX
};

struct FunMetaKType {
    BaseTlv head;
    unsigned int ktype;
};

struct FunLevelKType {
    struct FunMetaKType ktypeMeta;
};

#endif // AIV_KERNEL_OPERATOR_H

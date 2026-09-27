/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIMPLE_PARAM_H
#define SIMPLE_PARAM_H

#include "checker_def.h"

namespace checker {

struct SimpleParam {
    CheckerOpType opType{CheckerOpType::ALLREDUCE};
    std::string algName;
    CheckerOpMode opMode{CheckerOpMode::OPBASE};
    CheckerReduceOp reduceType{CheckerReduceOp::REDUCE_RESERVED};
    CheckerDevType devtype{CheckerDevType::DEV_TYPE_910B};
    bool is310P3V = false; // 仅当310PV卡的时候，设置为1
    u32 root{0};
    u32 dstRank{1};
    u32 srcRank{0};
    u64 count{160};
    CheckerDataType dataType{CheckerDataType::DATA_TYPE_FP32};
};

HcclResult GenTestOpParams(u32 rankSize, const SimpleParam& uiParams, CheckerOpParam& checkerParams);

} // namespace checker

#endif

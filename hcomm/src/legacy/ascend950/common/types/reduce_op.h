/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_REDUCE_OP_H
#define HCCLV2_REDUCE_OP_H

#include <map>
#include <string>
#include <hccl/hccl_types.h>

#include "enum_factory.h"
#include "string_util.h"
#include "../utils/exception_util.h"
#include "../exception/invalid_params_exception.h"
namespace Hccl {

MAKE_ENUM(ReduceOp, SUM, PROD, MAX, MIN, EQUAL)

const std::map<ReduceOp, HcclReduceOp> HCCL_REDUCE_OP_MAP
    = {{ReduceOp::SUM, HCCL_REDUCE_SUM},
       {ReduceOp::PROD, HCCL_REDUCE_PROD},
       {ReduceOp::MAX, HCCL_REDUCE_MAX},
       {ReduceOp::MIN, HCCL_REDUCE_MIN},
       {ReduceOp::INVALID, HCCL_REDUCE_RESERVED}};

const std::map<HcclReduceOp, ReduceOp> REDUCE_OP_MAP
    = {{HCCL_REDUCE_SUM, ReduceOp::SUM},
       {HCCL_REDUCE_PROD, ReduceOp::PROD},
       {HCCL_REDUCE_MAX, ReduceOp::MAX},
       {HCCL_REDUCE_MIN, ReduceOp::MIN},
       {HCCL_REDUCE_RESERVED, ReduceOp::INVALID}};

inline std::string ReduceOpToString(ReduceOp reduceOp) { return reduceOp.Describe(); }

inline HcclReduceOp ReduceOpToHcclReduceOp(const ReduceOp reduceOp)
{
    if (HCCL_REDUCE_OP_MAP.find(reduceOp) == HCCL_REDUCE_OP_MAP.end()) {
        THROW<InvalidParamsException>(
            StringFormat("%s reduceOp[%s] is not supported.", __func__, reduceOp.Describe().c_str()));
    }
    return HCCL_REDUCE_OP_MAP.at(reduceOp);
}

inline ReduceOp HcclReduceOpToReduceOp(const HcclReduceOp hcclReduceOp)
{
    if (REDUCE_OP_MAP.find(hcclReduceOp) == REDUCE_OP_MAP.end()) {
        THROW<InvalidParamsException>(StringFormat("%s hcclReduceOp[%d] is not supported.", __func__, hcclReduceOp));
    }
    return REDUCE_OP_MAP.at(hcclReduceOp);
}

} // namespace Hccl
#endif

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_REDUCE_IN_H
#define HCCLV2_REDUCE_IN_H
#include <string>
#include "data_type.h"
#include "reduce_op.h"
namespace Hccl {

struct ReduceIn {
    DataType dataType;
    ReduceOp reduceOp;
    ReduceIn(DataType dataType, ReduceOp reduceOp) : dataType(dataType), reduceOp(reduceOp) {}
    std::string Describe() const
    {
        return StringFormat(
            "ReduceIn[dataType=%s, reduceOp=%s]", dataType.Describe().c_str(), reduceOp.Describe().c_str());
    }
};

} // namespace Hccl
#endif

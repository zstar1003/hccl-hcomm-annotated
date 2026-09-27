/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_OPERATOR_H
#define HCCL_OPERATOR_H

#include <queue>
#include <map>
#include <memory>
#include <functional>
#include <vector>
#include "op_task.h"
#include "operator_param_type.h"

namespace hccl {
using HcclOpPtr = void*;
using TaskInfoPtr = void*;

template <typename ParamType>
class HcclOperator {
public:
    friend class OpExecutor;

public:
    explicit HcclOperator() = default;
    explicit HcclOperator(ParamType& param) : param_(param) {}
    virtual ~HcclOperator() = default;
    void PushTask(OpTaskPtr&& opTask) { taskQue_.push(std::forward<OpTaskPtr&&>(opTask)); }
    void Clear()
    {
        while (!taskQue_.empty()) {
            taskQue_.pop();
        }
    }

    // 需要默认初始化
    ParamType param_{};
    std::vector<u32> ranks_{};

private:
    std::queue<OpTaskPtr> taskQue_;
};

} // namespace hccl
#endif

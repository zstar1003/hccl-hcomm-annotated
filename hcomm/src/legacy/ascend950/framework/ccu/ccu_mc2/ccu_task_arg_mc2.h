/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_TASK_ARG_MC2_H
#define HCCL_CCU_TASK_ARG_MC2_H

#include <cstdint>
#include "ccu_task_arg.h"
namespace Hccl {
class CcuTaskArgMc2 : public CcuTaskArg {
public:
    explicit CcuTaskArgMc2(uint64_t token) : token_(token) {}
    uint64_t token_;
};
} // namespace Hccl

#endif // HCCL_CCU_TASK_ARG_MC2_H

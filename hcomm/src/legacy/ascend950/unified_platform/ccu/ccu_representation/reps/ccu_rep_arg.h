/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_REPRESENTATION_ARG_H
#define HCCL_CCU_REPRESENTATION_ARG_H

#include <vector>
#include <memory>

#include "ccu_datatype.h"

namespace Hccl {
namespace CcuRep {

    enum class CcuArgType {
        VARIABLE,
        MEMORY,
        VARIABLE_LIST,
        MEMORY_LIST,
    };

    struct CcuRepArg {
        explicit CcuRepArg(const Variable& var) : type(CcuArgType::VARIABLE), var(var) {}
        explicit CcuRepArg(const Memory& mem) : type(CcuArgType::MEMORY), mem(mem) {}
        explicit CcuRepArg(const std::vector<Variable>& varList) : type(CcuArgType::VARIABLE_LIST), varList(varList) {}
        explicit CcuRepArg(const std::vector<Memory>& memList) : type(CcuArgType::MEMORY_LIST), memList(memList) {}

        CcuArgType type;
        Variable var;
        Memory mem;
        std::vector<Variable> varList;
        std::vector<Memory> memList;
    };

}; // namespace CcuRep
}; // namespace Hccl
#endif // HCCL_CCU_REPRESENTATION_ARG_H

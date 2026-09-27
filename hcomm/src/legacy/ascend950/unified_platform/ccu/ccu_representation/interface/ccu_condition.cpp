/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_rep.h"
#include "ccu_interface_assist.h"

#include "string_util.h"
#include "exception_util.h"
#include "ccu_api_exception.h"

namespace Hccl {
namespace CcuRep {

    Condition::Condition(CcuRepContext* context, CcuRelationalOperator<Variable, uint64_t> rel) : context(context)
    {
        std::string label = "Condition";
        endLabel = std::make_shared<CcuRepJumpLabel>(label);
        Variable tmp;
        auto ret = CreateVariable(context, tmp);
        if (ret != HcclResult::HCCL_SUCCESS) {
            THROW<CcuApiException>("CreateVariable failed!");
        }
        // 当传入条件为真时, 执行Block, 对应不执行Jump
        if (rel.type == CcuRelationalOperatorType::NOT_EQUAL) {
            jump = std::make_shared<CcuRepJumpEQ>(label, tmp, rel.lhs, rel.rhs);
        } else if (rel.type == CcuRelationalOperatorType::EQUAL) {
            jump = std::make_shared<CcuRepJumpNE>(label, tmp, rel.lhs, rel.rhs);
        } else {
            THROW<CcuApiException>("Unsupported relational operation");
        }
        jump->Reference(endLabel);

        AppendToContext(context, jump);
    }

    Condition::~Condition() { DECTOR_TRY_CATCH("Condition", AppendToContext(context, endLabel)); }

    bool Condition::Check() const { return !isExecuted; }

    void Condition::Run() { isExecuted = true; }

}; // namespace CcuRep
}; // namespace Hccl

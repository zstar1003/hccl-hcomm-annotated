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

    Repeat::Repeat(CcuRepContext* context, CcuRelationalOperator<Variable, uint64_t> rel) : context(context)
    {
        std::string label = "Repeat";
        beginLabel = std::make_shared<CcuRepJumpLabel>(label);
        endLabel = std::make_shared<CcuRepJumpLabel>("Break");
        Variable tmp;
        auto ret = CreateVariable(context, tmp);
        if (ret != HcclResult::HCCL_SUCCESS) {
            THROW<CcuApiException>("CreateVariable is failed. ret[%d]", ret);
        }
        if (rel.type == CcuRelationalOperatorType::NOT_EQUAL) {
            jump = std::make_shared<CcuRepJumpNE>(label, tmp, rel.lhs, rel.rhs);
        } else if (rel.type == CcuRelationalOperatorType::EQUAL) {
            jump = std::make_shared<CcuRepJumpEQ>(label, tmp, rel.lhs, rel.rhs);
        } else {
            THROW<CcuApiException>("Unsupported relational operation");
        }
        jump->Reference(beginLabel);

        AppendToContext(context, beginLabel);
    }

    Repeat::~Repeat()
    {
        DECTOR_TRY_CATCH("Repeat", AppendToContext(context, jump));
        DECTOR_TRY_CATCH("Repeat", AppendToContext(context, endLabel));
    }

    void Repeat::Break()
    {
        Variable tmp;
        auto ret = CreateVariable(context, tmp);
        if (ret != HcclResult::HCCL_SUCCESS) {
            THROW<CcuApiException>("CreateVariable is failed. ret[%d]", ret);
        }
        auto jumpToEnd = std::make_shared<CcuRepJump>("Break", tmp);
        jumpToEnd->Reference(endLabel);
        AppendToContext(context, jumpToEnd);
    }

    bool Repeat::Check() const { return !isExecuted; }

    void Repeat::Run() { isExecuted = true; }

}; // namespace CcuRep
}; // namespace Hccl

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_CONDITION_H
#define CCU_CONDITION_H

#include "ccu_rep_jump_v1.h"
#include "ccu_rep_jumplabel_v1.h"
#include "ccu_rep_context_v1.h"

namespace hcomm {
namespace CcuRep {

#define CCU_IF(x) CCU_IF_HELPER1(__COUNTER__, x)

#define CCU_IF_HELPER1(ctr, x) CCU_IF_HELPER2(ctr, x)

#define CCU_IF_HELPER2(ctr, x)                                                                           \
    for (auto __ccuConditionHidden##ctr = CcuRep::Condition(this, x); __ccuConditionHidden##ctr.Check(); \
         __ccuConditionHidden##ctr.Run())

    class Condition {
    public:
        Condition(CcuRepContext* context, CcuRelationalOperator<Variable, uint64_t> rel);
        ~Condition();
        bool Check() const;
        void Run();

    private:
        CcuRepContext* context{nullptr};
        bool isExecuted{false};

        std::shared_ptr<CcuRepJumpBase> jump{nullptr};
        std::shared_ptr<CcuRepJumpLabel> endLabel{nullptr};
    };

}; // namespace CcuRep
}; // namespace hcomm
#endif // _CCU_CONDITION_H

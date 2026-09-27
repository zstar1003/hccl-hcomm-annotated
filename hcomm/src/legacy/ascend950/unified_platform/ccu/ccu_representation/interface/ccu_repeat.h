/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_REPEAT_H
#define HCCL_CCU_REPEAT_H

#include "ccu_rep_jump.h"
#include "ccu_rep_jumplabel.h"
#include "ccu_rep_context.h"

namespace Hccl {
namespace CcuRep {

#define CCU_WHILE(x) \
    for (auto __ccuRepeatHidden = CcuRep::Repeat(this, x); __ccuRepeatHidden.Check(); __ccuRepeatHidden.Run())
#define CCU_BREAK __ccuRepeatHidden.Break()

    class Repeat {
    public:
        Repeat(CcuRepContext* context, CcuRelationalOperator<Variable, uint64_t> rel);
        ~Repeat();
        void Break();
        bool Check() const;
        void Run();

    private:
        CcuRepContext* context{nullptr};
        bool isExecuted{false};

        std::shared_ptr<CcuRepJumpBase> jump{nullptr};
        std::shared_ptr<CcuRepJumpLabel> beginLabel{nullptr};
        std::shared_ptr<CcuRepJumpLabel> endLabel{nullptr};
    };

}; // namespace CcuRep
}; // namespace Hccl
#endif // HCCL_CCU_REPEAT_H

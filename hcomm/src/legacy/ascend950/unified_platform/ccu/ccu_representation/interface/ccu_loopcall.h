/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_LOOP_CALL_H
#define HCCL_CCU_LOOP_CALL_H

#include "ccu_rep_loopcall.h"
#include "ccu_rep_context.h"

namespace Hccl {
namespace CcuRep {

    class LoopCall {
    public:
        LoopCall(CcuRepContext* context, const std::string& label);
        const std::string& GetLabel() const { return label; }

        template <typename... Arguments>
        LoopCall& operator()(const Arguments&... args)
        {
            SetArgHelper(args...);
            AppendToContext();
            return *this;
        }

        LoopCall& operator()()
        {
            AppendToContext();
            return *this;
        }

        void AppendToContext();

    private:
        template <typename First>
        void SetArgHelper(const First& first)
        {
            repLoopCall->SetInArg(first);
        }

        template <typename First, typename... Rest>
        void SetArgHelper(const First& first, const Rest&... rest)
        {
            repLoopCall->SetInArg(first);
            SetArgHelper(rest...);
        }

        CcuRepContext* context{nullptr};
        std::string label;

        std::shared_ptr<CcuRepLoopCall> repLoopCall{nullptr};
    };

}; // namespace CcuRep
}; // namespace Hccl
#endif // HCCL_CCU_LOOP_CALL_H

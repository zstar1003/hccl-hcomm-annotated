/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_FUNC_CALL_H
#define CCU_FUNC_CALL_H

#include "ccu_rep_funccall_v1.h"
#include "ccu_rep_context_v1.h"

namespace hcomm {

namespace CcuRep {

    class FuncCall {
    public:
        FuncCall(CcuRepContext* context, std::string label);
        FuncCall(CcuRepContext* context, const Variable& funcAddr);

        template <typename... Arguments>
        FuncCall& operator()(const Arguments&... args)
        {
            SetArgHelper(args...);
            AppendToContext();
            return *this;
        }

        FuncCall& operator()()
        {
            AppendToContext();
            return *this;
        }

        void AppendToContext();

        template <typename T>
        void SetInArg(T&& arg)
        {
            repFuncCall->SetInArg(std::forward<T>(arg));
        }

        template <typename T>
        void SetOutArg(T&& arg)
        {
            repFuncCall->SetOutArg(std::forward<T>(arg));
        }

    private:
        template <typename First>
        void SetArgHelper(const First& first)
        {
            repFuncCall->SetInArg(first);
        }

        template <typename First, typename... Rest>
        void SetArgHelper(const First& first, const Rest&... rest)
        {
            repFuncCall->SetInArg(first);
            SetArgHelper(rest...);
        }

        CcuRepContext* context{nullptr};
        std::string label;

        std::shared_ptr<CcuRepFuncCall> repFuncCall{nullptr};
    };

}; // namespace CcuRep
}; // namespace hcomm
#endif // _CCU_FUNC_CALL_H

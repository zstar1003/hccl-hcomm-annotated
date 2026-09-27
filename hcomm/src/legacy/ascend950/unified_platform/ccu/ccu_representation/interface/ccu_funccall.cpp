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
#include "ccu_ctx.h"
#include "ccu_interface_assist.h"

#include "string_util.h"
#include "exception_util.h"
#include "ccu_api_exception.h"

namespace Hccl {
namespace CcuRep {

    FuncCall::FuncCall(CcuRepContext* context, std::string label) : context(context), label(label)
    {
        repFuncCall = std::make_shared<CcuRepFuncCall>(label);
    }

    FuncCall::FuncCall(CcuRepContext* context, const Variable& funcAddr) : context(context)
    {
        repFuncCall = std::make_shared<CcuRepFuncCall>(funcAddr);
    }

    void FuncCall::AppendToContext()
    {
        if (context == nullptr) {
            THROW<CcuApiException>("context is nullptr");
        }
        return context->Append(repFuncCall);
    }

}; // namespace CcuRep
}; // namespace Hccl

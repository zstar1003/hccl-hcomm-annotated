/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_rep_v1.h"
#include "ccu_kernel.h"
#include "ccu_interface_assist_v1.h"

#include "string_util.h"
#include "exception_util.h"
#include "ccu_api_exception.h"

#include "ccu_ins_generator_base.h"

namespace hcomm {
namespace CcuRep {

    FuncCall::FuncCall(CcuRepContext* context, std::string label) : context(context), label(label)
    {
        CcuInsGeneratorBase* insGenPtr = context->GetInsGenerator();
        Hccl::CHECK_NULLPTR(insGenPtr, "[FuncCall::FuncCall](label) insGenPtr is nullptr!");
        repFuncCall = std::make_shared<CcuRepFuncCall>(insGenPtr, label);
    }

    FuncCall::FuncCall(CcuRepContext* context, const Variable& funcAddr) : context(context)
    {
        CcuInsGeneratorBase* insGenPtr = context->GetInsGenerator();
        Hccl::CHECK_NULLPTR(insGenPtr, "[FuncCall::FuncCall](funcAddr) insGenPtr is nullptr!");
        repFuncCall = std::make_shared<CcuRepFuncCall>(insGenPtr, funcAddr);
    }

    void FuncCall::AppendToContext()
    {
        if (context == nullptr) {
            Hccl::THROW<Hccl::CcuApiException>("context is nullptr, func call, append to context");
        }
        return context->Append(repFuncCall);
    }

}; // namespace CcuRep
}; // namespace hcomm

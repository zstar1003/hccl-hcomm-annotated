/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_loopcall_v1.h"

#include "exception_util.h"
#include "ccu_api_exception.h"

namespace hcomm {
namespace CcuRep {

    LoopCall::LoopCall(CcuRepContext* context, const std::string& label) : context(context), label(label)
    {
        // repLoopCall = std::make_shared<CcuRepLoopCall>(label);
    }

    void LoopCall::AppendToContext()
    {
        if (context == nullptr) {
            Hccl::THROW<Hccl::CcuApiException>("context is nullptr, loopCall");
        }
        return context->Append(repLoopCall);
    }

}; // namespace CcuRep
}; // namespace hcomm

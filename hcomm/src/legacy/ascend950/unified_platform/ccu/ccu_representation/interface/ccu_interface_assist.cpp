/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_interface_assist.h"
#include "ccu_ctx.h"

#include "exception_util.h"
#include "ccu_api_exception.h"

namespace Hccl {
namespace CcuRep {

    void AppendToContext(CcuRepContext* context, std::shared_ptr<CcuRep::CcuRepBase> rep)
    {
        if (context == nullptr) {
            THROW<CcuApiException>("context is nullptr");
        } else {
            return context->Append(rep);
        }
    }

    std::shared_ptr<CcuRep::CcuRepBlock> CurrentBlock(CcuRepContext* context)
    {
        if (context == nullptr) {
            THROW<CcuApiException>("context is nullptr");
        }
        return context->CurrentBlock();
    }

    void SetCurrentBlock(CcuRepContext* context, std::shared_ptr<CcuRep::CcuRepBlock> repBlock)
    {
        if (context == nullptr) {
            THROW<CcuApiException>("context is nullptr");
        }
        context->SetCurrentBlock(repBlock);
    }

    HcclResult CreateVariable(CcuRepContext* context, Variable& variable)
    {
        HCCL_INFO("[CreateVariable] Input params: context[%p]", context);
        if (context == nullptr) {
            HCCL_ERROR("context is nullptr");
            return HCCL_E_PTR;
        }
        auto ctx = dynamic_cast<CcuContext*>(context);
        if (ctx == nullptr) {
            HCCL_ERROR("Invalid context");
            return HCCL_E_PTR;
        }
        variable = ctx->CreateVariable();
        return HCCL_SUCCESS;
    }

}; // namespace CcuRep
}; // namespace Hccl

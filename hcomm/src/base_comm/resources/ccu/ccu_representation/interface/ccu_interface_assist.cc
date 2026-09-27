/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_interface_assist_v1.h"
#include "ccu_kernel.h"

#include "exception_util.h"
#include "ccu_api_exception.h"

namespace hcomm {
namespace CcuRep {

    void AppendToContext(CcuRepContext* context, std::shared_ptr<CcuRep::CcuRepBase> rep)
    {
        if (context == nullptr) {
            Hccl::THROW<Hccl::CcuApiException>("context is nullptr, AppendToContext assist[%d]", rep->Type());
        } else {
            return context->Append(rep);
        }
    }

    std::shared_ptr<CcuRep::CcuRepBlock> CurrentBlock(CcuRepContext* context)
    {
        if (context == nullptr) {
            Hccl::THROW<Hccl::CcuApiException>("context is nullptr, currentBlock");
        }
        return context->CurrentBlock();
    }

    void SetCurrentBlock(CcuRepContext* context, std::shared_ptr<CcuRep::CcuRepBlock> repBlock)
    {
        if (context == nullptr) {
            Hccl::THROW<Hccl::CcuApiException>("context is nullptr, set currentBlock");
        }
        context->SetCurrentBlock(repBlock);
    }

    Variable CreateVariable(CcuRepContext* context)
    {
        if (context == nullptr) {
            Hccl::THROW<Hccl::CcuApiException>("context is nullptr, CreateVar");
        }
        auto ctx = dynamic_cast<CcuKernel*>(context);
        if (ctx == nullptr) {
            Hccl::THROW<Hccl::CcuApiException>("Invalid context");
        }
        return ctx->CreateVariable();
    }

}; // namespace CcuRep
}; // namespace hcomm

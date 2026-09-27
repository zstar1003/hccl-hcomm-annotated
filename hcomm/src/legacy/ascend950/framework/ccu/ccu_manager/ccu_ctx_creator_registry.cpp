/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_ctx_creator_registry.h"

namespace Hccl {

CcuCtxCreatorRegistry& CcuCtxCreatorRegistry::GetInstance()
{
    static CcuCtxCreatorRegistry ccuCtxRegister;
    return ccuCtxRegister;
}

CcuCtxCreatorRegistry::CreateCtxFunc CcuCtxCreatorRegistry::GetCreateFunc(CcuInstType ccuInstType) const
{
    auto it = creators.find(ccuInstType);
    if (it == creators.end()) {
        THROW<NotSupportException>(StringFormat(
            "[CcuCtxCreatorRegistry::%s] ccuInstType[%s] is not supported", __func__, ccuInstType.Describe().c_str()));
    }
    return it->second;
}

} // namespace Hccl

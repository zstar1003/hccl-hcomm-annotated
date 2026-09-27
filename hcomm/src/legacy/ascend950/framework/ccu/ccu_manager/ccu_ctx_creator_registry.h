/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_CTX_CREATOR_REGISTRY_H
#define CCU_CTX_CREATOR_REGISTRY_H

#include <memory>
#include "ccu_ctx.h"
#include "ccu_ins.h"

namespace Hccl {

class CcuCtxCreatorRegistry {
public:
    using CreateCtxFunc = std::function<std::unique_ptr<CcuContext>(
        const CcuCtxArg& ccuCtxArg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)>;

    static CcuCtxCreatorRegistry& GetInstance();

    template <typename T>
    void Register(CcuInstType ccuInstType)
    {
        creators.emplace(
            ccuInstType,
            [](const CcuCtxArg& ccuCtxArg, const std::vector<CcuTransport*>& transports,
               const CcuTransportGroup& group) -> std::unique_ptr<CcuContext> {
                return std::make_unique<T>(ccuCtxArg, transports, group);
            });
    }

    CreateCtxFunc GetCreateFunc(CcuInstType ccuInstType) const;

private:
    std::unordered_map<CcuInstType, CreateCtxFunc, std::EnumClassHash> creators;
};

template <typename T>
class CcuInstRegister {
public:
    explicit CcuInstRegister(CcuInstType ccuInstType) { CcuCtxCreatorRegistry::GetInstance().Register<T>(ccuInstType); }
};

} // namespace Hccl

#endif // CCU_CTX_CREATOR_REGISTRY_H

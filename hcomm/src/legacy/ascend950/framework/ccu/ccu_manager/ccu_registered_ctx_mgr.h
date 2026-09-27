/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_REGISTERED_CTX_MGR_H
#define CCU_REGISTERED_CTX_MGR_H

#include <memory>
#include <unordered_map>
#include "ccu_ctx_mgr.h"

namespace Hccl {

class RegisteredCcuCtxMgr {
public:
    explicit RegisteredCcuCtxMgr(DevId devLogicId) : devLogicId(devLogicId) {}

    bool HasRegistered(const CcuCtxSignature& ctxSignature, const uintptr_t& resPackId, u64& execId);

    u64 Register(
        std::unique_ptr<CcuCtxGroup> ccuCtxGroupPtr, const CcuCtxSignature& ctxSignature, const uintptr_t& resPackId,
        bool isFuncBlock);

    ~RegisteredCcuCtxMgr();

private:
    DevId devLogicId;
    unordered_map<CcuCtxSignature, unordered_map<uintptr_t, u64>> registeredIds;
};

} // namespace Hccl

#endif // CCU_REGISTERED_CTX_MGR_H

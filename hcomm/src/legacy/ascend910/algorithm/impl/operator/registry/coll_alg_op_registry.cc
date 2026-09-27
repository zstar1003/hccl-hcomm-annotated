/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_alg_op_registry.h"

namespace hccl {

CollAlgOpRegistry& CollAlgOpRegistry::Instance()
{
    static CollAlgOpRegistry globalOpRegistry;
    return globalOpRegistry;
}

HcclResult CollAlgOpRegistry::Register(const HcclCMDType& opType, const CollAlgOpCreator& collAlgOpCreator)
{
    const std::lock_guard<std::mutex> lock(mu_);
    if (opCreators_.find(opType) != opCreators_.end()) {
        HCCL_ERROR("[CollAlgOpRegistry]Op Type[%d] already registered.", opType);
        return HcclResult::HCCL_E_INTERNAL;
    }
    opCreators_.emplace(opType, collAlgOpCreator);
    return HcclResult::HCCL_SUCCESS;
}

std::unique_ptr<CollAlgOperator> CollAlgOpRegistry::GetAlgOp(
    const HcclCMDType& opType, AlgConfigurator* algConfigurator, CCLBufferManager& cclBufferManager,
    HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
{
    if (opCreators_.find(opType) == opCreators_.end()) {
        HCCL_ERROR("[CollAlgOpRegistry]Creator for op type[%d] has not registered.", opType);
        return nullptr;
    }
    return std::unique_ptr<CollAlgOperator>(
        opCreators_[opType](algConfigurator, cclBufferManager, dispatcher, topoMatcher));
}

} // namespace hccl

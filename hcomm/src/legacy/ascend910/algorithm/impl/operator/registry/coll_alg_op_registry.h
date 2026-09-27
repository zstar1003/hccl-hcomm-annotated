/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALG_OP_REGISTRY_H
#define COLL_ALG_OP_REGISTRY_H

#include <unordered_map>
#include <functional>
#include <memory>

#include "coll_alg_operator.h"

namespace hccl {

struct EnumClassHash {
    template <typename T>
    std::size_t operator()(T t) const
    {
        return static_cast<std::size_t>(t);
    }
};

using CollAlgOpCreator = std::function<CollAlgOperator*(
    AlgConfigurator* algConfigurator, CCLBufferManager&, HcclDispatcher, std::unique_ptr<TopoMatcher>&)>;

template <typename P>
static CollAlgOperator* DefaultOpCreator(
    AlgConfigurator* algConfigurator, CCLBufferManager& cclBufferManager, HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher>& topoMatcher)
{
    static_assert(std::is_base_of<CollAlgOperator, P>::value, "CollAlgOp type must derived from Hccl::CollAlgOperator");
    return new (std::nothrow) P(algConfigurator, cclBufferManager, dispatcher, topoMatcher);
}

class CollAlgOpRegistry {
public:
    static CollAlgOpRegistry& Instance();
    HcclResult Register(const HcclCMDType& opType, const CollAlgOpCreator& collAlgOpCreator);
    std::unique_ptr<CollAlgOperator> GetAlgOp(
        const HcclCMDType& opType, AlgConfigurator* algConfigurator, CCLBufferManager& cclBufferManager,
        HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);

private:
    std::unordered_map<HcclCMDType, const CollAlgOpCreator, EnumClassHash> opCreators_;
    mutable std::mutex mu_;
};

#define REGISTER_OP_HELPER(ctr, type, name, collOpBase) \
    static HcclResult g_func_##name##_##ctr = CollAlgOpRegistry::Instance().Register(type, DefaultOpCreator<collOpBase>)

#define REGISTER_OP_HELPER_1(ctr, type, name, collOpBase) REGISTER_OP_HELPER(ctr, type, name, collOpBase)

#define REGISTER_OP(type, name, collOpBase) REGISTER_OP_HELPER_1(__COUNTER__, type, name, collOpBase)

} // namespace hccl

#endif

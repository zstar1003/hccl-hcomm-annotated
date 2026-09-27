/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_COLL_ALG_REGISTRY
#define HCCLV2_COLL_ALG_REGISTRY

#include <mutex>
#include <functional>
#include "coll_alg_base.h"
#include "log.h"

namespace Hccl {

using CollAlgCreator = std::function<CollAlgBase*()>;
template <typename P>
static CollAlgBase* DefaultCreator()
{
    static_assert(std::is_base_of<CollAlgBase, P>::value, "CollAlg type must derived from Hccl::CollAlgBase");
    return new (std::nothrow) P;
}

class CollAlgRegistry {
public:
    static CollAlgRegistry* Global();
    HcclResult Register(const OpType type, const std::string& funcName, const CollAlgCreator& collAlgCreator);
    void PrintAllImpls();
    std::shared_ptr<CollAlgBase> GetAlgImpl(const OpType type, const std::string& funcName);
    std::map<OpType, std::vector<std::string>> GetAvailAlgs();

private:
    std::map<OpType, std::map<std::string, const CollAlgCreator>> impls_;
    mutable std::mutex mu_;
};

#define REGISTER_IMPL_HELPER(ctr, type, name, collAlgBase) \
    static HcclResult g_func_##name##_##ctr                \
        = CollAlgRegistry::Global()->Register(type, std::string(#name), DefaultCreator<collAlgBase>)

#define REGISTER_IMPL_HELPER_1(ctr, type, name, collAlgBase) REGISTER_IMPL_HELPER(ctr, type, name, collAlgBase)

#define REGISTER_IMPL(type, name, collAlgBase) REGISTER_IMPL_HELPER_1(__COUNTER__, type, name, collAlgBase)

#define REGISTER_IMPL_BY_TEMP_HELPER(ctr, type, name, collAlgBase, AlgTopoMatch, AlgTemplate) \
    static HcclResult g_func_##name##_##ctr = CollAlgRegistry::Global()->Register(            \
        type, std::string(#name), DefaultCreator<collAlgBase<AlgTopoMatch, AlgTemplate>>)

#define REGISTER_IMPL_BY_TEMP_HELPER_1(ctr, type, name, collAlgBase, AlgTopoMatch, AlgTemplate) \
    REGISTER_IMPL_BY_TEMP_HELPER(ctr, type, name, collAlgBase, AlgTopoMatch, AlgTemplate)

#define REGISTER_IMPL_BY_TEMP(type, name, collAlgBase, AlgTopoMatch, AlgTemplate) \
    REGISTER_IMPL_BY_TEMP_HELPER_1(__COUNTER__, type, name, collAlgBase, AlgTopoMatch, AlgTemplate)

#define REGISTER_IMPL_BY_TWO_TEMPS_HELPER(ctr, type, name, collAlgBase, AlgTopoMatch, AlgTemplateRS, AlgTemplateAG) \
    static HcclResult g_func_##name##_##ctr = CollAlgRegistry::Global()->Register(                                  \
        type, std::string(#name), DefaultCreator<collAlgBase<AlgTopoMatch, AlgTemplateRS, AlgTemplateAG>>)

#define REGISTER_IMPL_BY_TWO_TEMPS_HELPER_1(ctr, type, name, collAlgBase, AlgTopoMatch, AlgTemplateRS, AlgTemplateAG) \
    REGISTER_IMPL_BY_TWO_TEMPS_HELPER(ctr, type, name, collAlgBase, AlgTopoMatch, AlgTemplateRS, AlgTemplateAG)

#define REGISTER_IMPL_BY_TWO_TEMPS(type, name, collAlgBase, AlgTopoMatch, AlgTemplateRS, AlgTemplateAG) \
    REGISTER_IMPL_BY_TWO_TEMPS_HELPER_1(                                                                \
        __COUNTER__, type, name, collAlgBase, AlgTopoMatch, AlgTemplateRS, AlgTemplateAG)

} // namespace Hccl

#endif

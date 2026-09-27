/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_coll_alg_registry.h"

#include <string>
#include <iostream>
#include <map>
#include <mutex>
#include <new>

namespace Hccl {

InsCollAlgRegistry* InsCollAlgRegistry::Global()
{
    static InsCollAlgRegistry* globalAlgImplRegistry = new (std::nothrow) InsCollAlgRegistry;
    if (globalAlgImplRegistry == nullptr) {
        HCCL_ERROR("[Alg][Registry] allocate global InsCollAlgRegistry failed, out of memory.");
        return nullptr;
    }
    return globalAlgImplRegistry;
}

HcclResult
InsCollAlgRegistry::Register(const OpType type, const std::string& funcName, const InsCollAlgCreator& collAlgCreator)
{
    const std::lock_guard<std::mutex> lock(mu_);
    if (impls_[type].count(funcName) != 0) {
        HCCL_ERROR("[%d]: [%s] already registered.", type, funcName.c_str());
        return HcclResult::HCCL_E_INTERNAL;
    }
    impls_[type].emplace(funcName, collAlgCreator);
    return HcclResult::HCCL_SUCCESS;
}

void InsCollAlgRegistry::PrintAllImpls()
{
    for (auto& iter : impls_) {
        HCCL_DEBUG("-------------------------------------");
        HCCL_DEBUG("Optype [%s]", iter.first.Describe().c_str());
        for (auto& alg : iter.second) {
            HCCL_DEBUG("    with alg [%s]", alg.first.c_str());
            if (alg.second == nullptr) {
                HCCL_DEBUG("    alg func is nullptr");
            }
        }
    }
}

std::map<OpType, std::vector<std::string>> InsCollAlgRegistry::GetAvailAlgs()
{
    std::map<OpType, std::vector<std::string>> algs;
    for (auto& iter : impls_) {
        HCCL_DEBUG("OpType [%s]", iter.first.Describe().c_str());
        std::vector<std::string> tmpAvailAlgs;
        for (auto& alg : iter.second) {
            HCCL_DEBUG("AlgName [%s]", alg.first.c_str());
            tmpAvailAlgs.push_back(alg.first);
            if (alg.second == nullptr) {
                HCCL_DEBUG("Alg function is nullptr.");
            }
        }
        algs.insert(std::make_pair(iter.first, tmpAvailAlgs));
    }
    return algs;
}

std::shared_ptr<InsCollAlgBase> InsCollAlgRegistry::GetAlgImpl(const OpType type, const std::string& funcName)
{
    if (impls_.count(type) == 0 || impls_[type].count(funcName) == 0) {
        HCCL_WARNING("%s:%s is not registered.", type.Describe().c_str(), funcName.c_str());
        return nullptr;
    }
    return std::shared_ptr<InsCollAlgBase>(impls_[type][funcName]());
}
} // namespace Hccl

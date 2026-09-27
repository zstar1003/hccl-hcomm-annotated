/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_rep_reference_manager.h"
#include "exception_util.h"
#include "ccu_api_exception.h"

namespace Hccl {
namespace CcuRep {

    CcuRepReferenceManager::CcuRepReferenceManager(uint8_t deiId) : dieId(deiId)
    {
        funcInVar.resize(FUNC_IN_MAX);
        funcOutVar.resize(FUNC_OUT_MAX);
        funcCallVar.resize(
            1 + FUNC_NEST_MAX
            + 1); // FUNC_NEST_MAX个xn存放返回地址，1个xn存放block的起始地址和1个xn存放函数地址调用时返回地址
    }

    CcuResReq CcuRepReferenceManager::GetResReq(uint8_t reqDieId)
    {
        CcuResReq resReq;
        resReq.xnReq[reqDieId] = FUNC_IN_MAX + FUNC_OUT_MAX + 1 + FUNC_NEST_MAX + 1;
        return resReq;
    }

    void CcuRepReferenceManager::GetRes(CcuRepResource& res)
    {
        res.variable[dieId].insert(res.variable[dieId].end(), funcInVar.begin(), funcInVar.end());
        res.variable[dieId].insert(res.variable[dieId].end(), funcOutVar.begin(), funcOutVar.end());
        res.variable[dieId].insert(res.variable[dieId].end(), funcCallVar.begin(), funcCallVar.end());
    }

    bool CcuRepReferenceManager::CheckValid(const std::string& label)
    {
        return (referenceMap.find(label) != referenceMap.end());
    }

    bool CcuRepReferenceManager::CheckUnique(const std::string& label)
    {
        return (referenceMap.find(label) == referenceMap.end());
    }

    std::shared_ptr<CcuRepBlock> CcuRepReferenceManager::GetRefBlock(const std::string& label)
    {
        if (!CheckValid(label)) {
            THROW<CcuApiException>("Invalid Reference: %s", label.c_str());
        }
        return referenceMap[label];
    }

    void CcuRepReferenceManager::SetRefBlock(const std::string& label, std::shared_ptr<CcuRepBlock> refBlock)
    {
        if (!CheckUnique(label)) {
            THROW<CcuApiException>("Duplicate Definition: %s", label.c_str());
        }
        referenceMap[label] = refBlock;
    }

    uint16_t CcuRepReferenceManager::GetFuncAddr(const std::string& label)
    {
        if (!CheckValid(label)) {
            THROW<CcuApiException>("Invalid Reference: %s", label.c_str());
        }

        if (referenceMap[label]->Type() != CcuRepType::FUNC_BLOCK) {
            THROW<CcuApiException>("Invalid Type, %s Must be FuncBlock", label.c_str());
        }

        return referenceMap[label]->StartInstrId();
    }

    const Variable& CcuRepReferenceManager::GetFuncCall() { return funcCallVar[0]; }

    const Variable& CcuRepReferenceManager::GetFuncRet(uint16_t callLayer)
    {
        if (callLayer > FUNC_NEST_MAX) {
            THROW<CcuApiException>("Max Func Call Nest Num is %u, callLayer = %u", FUNC_NEST_MAX, callLayer);
        }
        return funcCallVar[callLayer + 1];
    }

    const std::vector<Variable>& CcuRepReferenceManager::GetFuncIn() { return funcInVar; }

    const std::vector<Variable>& CcuRepReferenceManager::GetFuncOut() { return funcOutVar; }

    void CcuRepReferenceManager::Dump() const
    {
        for (const auto& kv : referenceMap) {
            HCCL_INFO("refBlock[%s]:", kv.first.c_str());
        }
    }

    void CcuRepReferenceManager::ClearRepReference() { referenceMap.clear(); }

}; // namespace CcuRep
}; // namespace Hccl

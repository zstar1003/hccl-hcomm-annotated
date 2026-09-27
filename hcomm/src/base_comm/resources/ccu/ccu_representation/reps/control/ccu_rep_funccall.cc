/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_rep_v1.h"
#include "ccu_rep_reference_manager_v1.h"

#include "string_util.h"
#include "exception_util.h"
#include "ccu_api_exception.h"

#include "ccu_ins_generator_base.h"
#include "ccu_kernel.h"

namespace hcomm {
namespace CcuRep {

    using namespace Hccl;

    CcuRepFuncCall::CcuRepFuncCall(CcuInsGeneratorBase* insGenPtr, const std::string& label)
        : insGeneratorPtr_(insGenPtr),
          label(label)
    {
        type = CcuRepType::FUNC_CALL;
        instrCount = 0;
    }

    CcuRepFuncCall::CcuRepFuncCall(CcuInsGeneratorBase* insGenPtr, const Variable& funcAddrVar)
        : insGeneratorPtr_(insGenPtr),
          label(""),
          funcAddrVar(funcAddrVar)
    {
        type = CcuRepType::FUNC_CALL;
    }

    const std::string& CcuRepFuncCall::GetLabel() const { return label; }

    void CcuRepFuncCall::Reference(std::shared_ptr<CcuRepFuncBlock> refRep) { funcBlock = refRep; }

    void CcuRepFuncCall::SetFuncManager(CcuRepReferenceManager* funcManager) { this->funcManager = funcManager; }

    void CcuRepFuncCall::SetInArg(const Variable& var)
    {
        inArgCount++;
        inArgs.push_back(CcuRepArg(var));
    }

    void CcuRepFuncCall::SetOutArg(const Variable& var)
    {
        outArgCount++;
        if (outArgCount > FUNC_ARG_MAX) {
            Hccl::THROW<Hccl::CcuApiException>("CcuFunc Max ArgCount = %u", FUNC_ARG_MAX);
        }
        outArgs.push_back(CcuRepArg(var));
    }

    void CcuRepFuncCall::SetInArg(const std::vector<Variable>& varList)
    {
        inArgCount += varList.size();
        inArgs.push_back(CcuRepArg(varList));
    }

    void CcuRepFuncCall::SetOutArg(const std::vector<Variable>& varList)
    {
        outArgCount += varList.size();
        if (outArgCount > FUNC_ARG_MAX) {
            Hccl::THROW<Hccl::CcuApiException>("CcuFunc Max ArgCount = %u", FUNC_ARG_MAX);
        }
        outArgs.push_back(CcuRepArg(varList));
    }

    uint16_t CcuRepFuncCall::InstrCount()
    {
        instrCount = inArgCount + outArgCount
                     + insGeneratorPtr_->GetInstrCount(type); // funcCall除去入参和出参的处理外，需要额外4条指令
        return instrCount;
    }

    bool CcuRepFuncCall::Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep)
    {
        if (funcManager == nullptr) {
            Hccl::THROW<Hccl::CcuApiException>("funcManager is nullptr");
        }
        // 未实现, FuncCall和FuncBlock中的args个数校验

        if (this->instr == nullptr) {
            this->instrId = instrId;
            this->instr = instr;
            instr += InstrCount();
            instrId += InstrCount();
        }

        if (funcBlock != nullptr && !funcBlock->Translated()) {
            return translated;
        }

        translated = true;

        CHK_PRT_THROW(
            insGeneratorPtr_->CcuRepFuncCallTranslate(ccuKernel, instr, instrId, this, dep) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[CcuRepFuncCall][Translate] failed to translate for instrId[%u]", instrId),
            Hccl::CcuApiException, "CcuRepFuncCall translate failed");

        return translated;
    }

    std::string CcuRepFuncCall::Describe() { return Hccl::StringFormat("FuncCall[%s]", label.c_str()); }

    int32_t CcuRepFuncCall::GetCallLayer() { return funcBlock == nullptr ? FUNC_NEST_MAX : funcBlock->GetCallLayer(); }

}; // namespace CcuRep
}; // namespace hcomm

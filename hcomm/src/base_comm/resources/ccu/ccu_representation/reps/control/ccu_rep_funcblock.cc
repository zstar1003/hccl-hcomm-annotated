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
#include "ccu_rep_translator_v1.h"

#include "string_util.h"
#include "exception_util.h"
#include "ccu_api_exception.h"

#include "ccu_ins_generator_base.h"
#include "ccu_kernel.h"

namespace hcomm {
namespace CcuRep {

    using namespace Hccl;

    CcuRepFuncBlock::CcuRepFuncBlock(CcuInsGeneratorBase* insGenPtr, const std::string& label)
        : CcuRepBlock(insGenPtr, label)
    {
        type = CcuRepType::FUNC_BLOCK;
        instrCount = 0;
    }

    std::string CcuRepFuncBlock::Describe() { return Hccl::StringFormat("FuncBlock[%s]", GetLabel().c_str()); }

    void CcuRepFuncBlock::SetFuncManager(CcuRepReferenceManager* funcManager) { this->funcManager = funcManager; }

    void CcuRepFuncBlock::SetCallLayer(uint16_t callLayer)
    {
        if (callLayer != FUNC_CALL_LAYER_INVALID) {
            this->callLayer = callLayer;
            return;
        }

        uint16_t innerCallLayer = 0;
        for (const auto& rep : GetReps()) {
            if (rep->Type() == CcuRepType::FUNC_CALL) {
                innerCallLayer = std::static_pointer_cast<CcuRepFuncCall>(rep)->GetCallLayer() + 1;
                this->callLayer = this->callLayer > innerCallLayer ? this->callLayer : innerCallLayer;
            }
        }
        if (this->callLayer > FUNC_NEST_MAX - 1) {
            Hccl::THROW<Hccl::CcuApiException>("Max Func Call Nest Num is %u", FUNC_NEST_MAX);
        }
    }

    uint16_t CcuRepFuncBlock::GetCallLayer() const { return callLayer; }

    void CcuRepFuncBlock::DefineInArg(const Variable& var)
    {
        inArgCount++;
        inArgs.push_back(CcuRepArg(var));
        HCCL_INFO("Define Input Arg: Index[%u], Type[Variable] Id[%u]", inArgs.size(), var.Id());
    }

    void CcuRepFuncBlock::DefineOutArg(const Variable& var)
    {
        outArgCount++;
        if (outArgCount > FUNC_ARG_MAX) {
            Hccl::THROW<Hccl::CcuApiException>("CcuFunc Max ArgCount = %u", FUNC_ARG_MAX);
        }
        outArgs.push_back(CcuRepArg(var));
        HCCL_INFO("Define Output Arg: Index[%u], Type[Variable] Id[%u]", outArgs.size(), var.Id());
    }

    void CcuRepFuncBlock::DefineInArg(const std::vector<Variable>& varList)
    {
        inArgCount += varList.size();
        inArgs.push_back(CcuRepArg(varList));
        HCCL_INFO("Define Input Arg: Index[%u], Type[Variable List]: ", inArgs.size());
        for (uint32_t index = 0; index < varList.size(); index++) {
            HCCL_INFO("    Index[%u].Id[%u]", index, varList[index].Id());
        }
    }

    void CcuRepFuncBlock::DefineOutArg(const std::vector<Variable>& varList)
    {
        outArgCount += varList.size();
        if (outArgCount > FUNC_ARG_MAX) {
            Hccl::THROW<Hccl::CcuApiException>("CcuFunc Max ArgCount = %u", FUNC_ARG_MAX);
        }
        outArgs.push_back(CcuRepArg(varList));
        HCCL_INFO("Define Output Arg: Index[%u], Type[Variable List]: ", outArgs.size());
        for (uint32_t index = 0; index < varList.size(); index++) {
            HCCL_INFO("    Index[%u].Id[%u]", index, varList[index].Id());
        }
    }

    std::vector<Variable> CcuRepFuncBlock::GetInArgVars() const
    {
        std::vector<Variable> vars;
        for (const auto& arg : inArgs) {
            if (arg.type == CcuArgType::VARIABLE) {
                vars.push_back(arg.var);
            } else if (arg.type == CcuArgType::VARIABLE_LIST) {
                vars.insert(vars.end(), arg.varList.begin(), arg.varList.end());
            }
        }
        return vars;
    }
    uint16_t CcuRepFuncBlock::InstrCount()
    {
        instrCount = CcuRepBlock::InstrCount() + inArgCount + outArgCount
                     + insGeneratorPtr_->GetInstrCount(type); // FuncBlock需要额外指令
        return instrCount;
    }

    bool CcuRepFuncBlock::Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep)
    {
        if (funcManager == nullptr) {
            Hccl::THROW<Hccl::CcuApiException>("funcManager is nullptr");
        }

        this->instrId = instrId;
        translated = true;

        CHK_PRT_THROW(
            insGeneratorPtr_->CcuRepFuncBlockTranslate(ccuKernel, instr, instrId, this, dep, 0)
                != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[CcuRepFuncBlock][Translate] failed to translate inArgs processing for instrId[%u]", instrId),
            Hccl::CcuApiException, "CcuRepFuncBlock translate failed");
        // 使用空实现的自定义删除器，避免智能指针析构时释放对象
        auto translator = CcuRepTranslator(
            std::shared_ptr<CcuRepReferenceManager>(funcManager, []([[maybe_unused]] CcuRepReferenceManager* ptr) {}),
            dep);
        translator.Translate(
            ccuKernel, GetReps(), instr, instrId, []([[maybe_unused]] std::shared_ptr<CcuRepBase> rep) -> bool {
                return true;
            });

        CHK_PRT_THROW(
            insGeneratorPtr_->CcuRepFuncBlockTranslate(ccuKernel, instr, instrId, this, dep, 1)
                != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[CcuRepFuncBlock][Translate] failed to translate outArgs processing for instrId[%u]", instrId),
            Hccl::CcuApiException, "CcuRepFuncBlock translate failed");

        return translated;
    }

}; // namespace CcuRep
}; // namespace hcomm

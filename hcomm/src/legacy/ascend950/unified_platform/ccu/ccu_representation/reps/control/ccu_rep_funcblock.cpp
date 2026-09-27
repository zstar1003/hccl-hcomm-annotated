/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_rep.h"
#include "ccu_rep_reference_manager.h"
#include "ccu_rep_translator.h"

#include "string_util.h"
#include "exception_util.h"
#include "ccu_api_exception.h"

namespace Hccl {
namespace CcuRep {

    CcuRepFuncBlock::CcuRepFuncBlock(const std::string& label) : CcuRepBlock(label) { type = CcuRepType::FUNC_BLOCK; }

    std::string CcuRepFuncBlock::Describe() { return StringFormat("FuncBlock[%s]", GetLabel().c_str()); }

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
            THROW<CcuApiException>("Max Func Call Nest Num is %u", FUNC_NEST_MAX);
        }
    }

    uint16_t CcuRepFuncBlock::GetCallLayer() const { return callLayer; }

    void CcuRepFuncBlock::DefineInArg(const Variable& var)
    {
        inArgCount++;
        if (inArgCount > FUNC_IN_MAX) {
            HCCL_ERROR("[%s]CcuFunc Max ArgCount = %u, Current ArgCount = %u", __func__, FUNC_IN_MAX, inArgCount);
            THROW<CcuApiException>("CcuFunc Max ArgCount = %u", FUNC_IN_MAX);
        }
        inArgs.push_back(CcuRepArg(var));
        HCCL_INFO("[%s]Define Input Arg: Index[%u], Type[Variable] Id[%u]", __func__, inArgs.size(), var.Id());
    }

    void CcuRepFuncBlock::DefineOutArg(const Variable& var)
    {
        outArgCount++;
        if (outArgCount > FUNC_OUT_MAX) {
            THROW<CcuApiException>("CcuFunc Max ArgCount = %u", FUNC_OUT_MAX);
        }
        outArgs.push_back(CcuRepArg(var));
        HCCL_INFO("Define Output Arg: Index[%u], Type[Variable] Id[%u]", outArgs.size(), var.Id());
    }

    void CcuRepFuncBlock::DefineInArg(const std::vector<Variable>& varList)
    {
        inArgCount += varList.size();
        if (inArgCount > FUNC_IN_MAX) {
            HCCL_ERROR("[%s]CcuFunc Max ArgCount = %u, Current ArgCount = %u", __func__, FUNC_IN_MAX, inArgCount);
            THROW<CcuApiException>("CcuFunc Max ArgCount = %u", FUNC_IN_MAX);
        }
        inArgs.push_back(CcuRepArg(varList));
        HCCL_INFO("[%s]Define Input Arg: Index[%u], Type[Variable List]: ", __func__, inArgs.size());
        for (uint32_t index = 0; index < varList.size(); index++) {
            HCCL_INFO("    Index[%u].Id[%u]", index, varList[index].Id());
        }
    }

    void CcuRepFuncBlock::DefineOutArg(const std::vector<Variable>& varList)
    {
        outArgCount += varList.size();
        if (outArgCount > FUNC_OUT_MAX) {
            THROW<CcuApiException>("CcuFunc Max ArgCount = %u", FUNC_OUT_MAX);
        }
        outArgs.push_back(CcuRepArg(varList));
        HCCL_INFO("Define Output Arg: Index[%u], Type[Variable List]: ", outArgs.size());
        for (uint32_t index = 0; index < varList.size(); index++) {
            HCCL_INFO("    Index[%u].Id[%u]", index, varList[index].Id());
        }
    }

    uint16_t CcuRepFuncBlock::InstrCount()
    {
        instrCount = CcuRepBlock::InstrCount() + inArgCount + outArgCount + 2; // FuncBlock需要2外两条指令
        return instrCount;
    }

    bool CcuRepFuncBlock::Translate(CcuInstr*& instr, uint16_t& instrId, const TransDep& dep)
    {
        if (funcManager == nullptr) {
            THROW<CcuApiException>("funcManager is nullptr");
        }

        this->instrId = instrId;
        translated = true;

        // 函数入口为Nop指令
        LoadImdToXnInstr(instr++, dep.reserveXnId, 0);
        instrId++;

        uint32_t iInArg = 0;
        for (uint32_t i = 0; i < inArgs.size(); i++) {
            if (inArgs[i].type == CcuArgType::VARIABLE) {
                LoadXXInstr(instr++, inArgs[i].var.Id(), funcManager->GetFuncIn()[iInArg++].Id(), dep.reserveXnId);
                instrId++;
            } else if (inArgs[i].type == CcuArgType::VARIABLE_LIST) {
                for (uint32_t j = 0; j < inArgs[i].varList.size(); j++) {
                    LoadXXInstr(
                        instr++, inArgs[i].varList[j].Id(), funcManager->GetFuncIn()[iInArg++].Id(), dep.reserveXnId);
                    instrId++;
                }
            }
        }

        // 使用空实现的自定义删除器，避免智能指针析构时释放对象
        auto translator = CcuRepTranslator(
            std::shared_ptr<CcuRepReferenceManager>(funcManager, []([[maybe_unused]] CcuRepReferenceManager* ptr) {}),
            dep);
        translator.Translate(GetReps(), instr, instrId, []([[maybe_unused]] std::shared_ptr<CcuRepBase> rep) -> bool {
            return true;
        });

        uint32_t iOutArg = 0;
        for (uint32_t i = 0; i < outArgs.size(); i++) {
            if (outArgs[i].type == CcuArgType::VARIABLE) {
                LoadXXInstr(instr++, funcManager->GetFuncOut()[iOutArg++].Id(), outArgs[i].var.Id(), dep.reserveXnId);
                instrId++;
            } else if (outArgs[i].type == CcuArgType::VARIABLE_LIST) {
                for (uint32_t j = 0; j < outArgs[i].varList.size(); j++) {
                    LoadXXInstr(
                        instr++, funcManager->GetFuncOut()[iOutArg++].Id(), outArgs[i].varList[j].Id(),
                        dep.reserveXnId);
                    instrId++;
                }
            }
        }

        JumpInstr(instr++, funcManager->GetFuncRet(callLayer).Id(), dep.reserveXnId, 1);
        instrId++;

        return translated;
    }

}; // namespace CcuRep
}; // namespace Hccl

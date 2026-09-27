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

#include "string_util.h"
#include "exception_util.h"
#include "ccu_api_exception.h"

namespace Hccl {
namespace CcuRep {

    CcuRepFuncCall::CcuRepFuncCall(const std::string& label) : label(label) { type = CcuRepType::FUNC_CALL; }

    CcuRepFuncCall::CcuRepFuncCall(const Variable& funcAddrVar) : label(""), funcAddrVar(funcAddrVar)
    {
        type = CcuRepType::FUNC_CALL;
    }

    const std::string& CcuRepFuncCall::GetLabel() const { return label; }

    void CcuRepFuncCall::Reference(std::shared_ptr<CcuRepFuncBlock> refRep) { funcBlock = refRep; }

    void CcuRepFuncCall::SetFuncManager(CcuRepReferenceManager* funcManager) { this->funcManager = funcManager; }

    void CcuRepFuncCall::SetInArg(const Variable& var)
    {
        inArgCount++;
        if (inArgCount > FUNC_IN_MAX) {
            HCCL_ERROR("[%s]CcuFunc Max ArgCount = %u, Current ArgCount = %u", __func__, FUNC_IN_MAX, inArgCount);
            THROW<CcuApiException>("CcuFunc Max ArgCount = %u", FUNC_IN_MAX);
        }
        inArgs.push_back(CcuRepArg(var));
        HCCL_INFO("[%s]Define Input Arg: Index[%u], Type[Variable] Id[%u]", __func__, inArgs.size(), var.Id());
    }

    void CcuRepFuncCall::SetOutArg(const Variable& var)
    {
        outArgCount++;
        if (outArgCount > FUNC_OUT_MAX) {
            THROW<CcuApiException>("CcuFunc Max ArgCount = %u", FUNC_OUT_MAX);
        }
        outArgs.push_back(CcuRepArg(var));
    }

    void CcuRepFuncCall::SetInArg(const std::vector<Variable>& varList)
    {
        inArgCount += varList.size();
        if (inArgCount > FUNC_IN_MAX) {
            HCCL_ERROR("[%s]CcuFunc Max ArgCount = %u, Current ArgCount = %u", __func__, FUNC_IN_MAX, inArgCount);
            THROW<CcuApiException>("CcuFunc Max ArgCount = %u", FUNC_IN_MAX);
        }
        inArgs.push_back(CcuRepArg(varList));
        HCCL_INFO("[%s]Define Input Arg: Index[%u], Type[Variable List]: ", __func__, inArgs.size());
    }

    void CcuRepFuncCall::SetOutArg(const std::vector<Variable>& varList)
    {
        outArgCount += varList.size();
        if (outArgCount > FUNC_OUT_MAX) {
            THROW<CcuApiException>("CcuFunc Max ArgCount = %u", FUNC_OUT_MAX);
        }
        outArgs.push_back(CcuRepArg(varList));
    }

    uint16_t CcuRepFuncCall::InstrCount()
    {
        instrCount = inArgCount + outArgCount + 4; // funcCall除去入参和出参的处理外，需要额外4条指令
        return instrCount;
    }

    bool CcuRepFuncCall::Translate(CcuInstr*& instr, uint16_t& instrId, const TransDep& dep)
    {
        if (funcManager == nullptr) {
            THROW<CcuApiException>("funcManager is nullptr");
        }
        // 未实现, FuncCall和FuncBlock中的args个数校验
        uint32_t extraInstrNum = 4; // funcCall除去入参和出参的处理外，需要额外4条指令
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

        uint32_t iInArg = 0;
        for (uint32_t i = 0; i < inArgs.size(); i++) {
            if (inArgs[i].type == CcuArgType::VARIABLE) {
                LoadXXInstr(
                    this->instr + iInArg, funcManager->GetFuncIn()[iInArg].Id(), inArgs[i].var.Id(), dep.reserveXnId);
                iInArg++;
            } else if (inArgs[i].type == CcuArgType::VARIABLE_LIST) {
                for (uint32_t j = 0; j < inArgs[i].varList.size(); j++) {
                    LoadXXInstr(
                        this->instr + iInArg, funcManager->GetFuncIn()[iInArg].Id(), inArgs[i].varList[j].Id(),
                        dep.reserveXnId);
                    iInArg++;
                }
            }
        }

        uint32_t locId = 0;
        if (funcBlock != nullptr) {
            LoadImdToXnInstr(
                this->instr + inArgCount + locId++, funcManager->GetFuncCall().Id(), funcBlock->StartInstrId());
        } else {
            LoadXXInstr(
                this->instr + inArgCount + locId++, funcManager->GetFuncCall().Id(), funcAddrVar.Id(), dep.reserveXnId);
        }

        LoadImdToXnInstr(
            this->instr + inArgCount + locId++, funcManager->GetFuncRet(GetCallLayer()).Id(),
            this->instrId + inArgCount + 3); // 需要指向函数返回位置，为输入指令Id + 3
        JumpInstr(this->instr + inArgCount + locId++, funcManager->GetFuncCall().Id(), dep.reserveXnId, 1);
        LoadImdToXnInstr(this->instr + inArgCount + locId++, dep.reserveXnId, 0);

        uint32_t iOutArg = 0;
        for (uint32_t i = 0; i < outArgs.size(); i++) {
            if (outArgs[i].type == CcuArgType::VARIABLE) {
                LoadXXInstr(
                    this->instr + inArgCount + extraInstrNum + iOutArg, outArgs[i].var.Id(),
                    funcManager->GetFuncOut()[iOutArg].Id(), dep.reserveXnId);
                iOutArg++;
            } else if (outArgs[i].type == CcuArgType::VARIABLE_LIST) {
                for (uint32_t j = 0; j < outArgs[i].varList.size(); j++) {
                    LoadXXInstr(
                        this->instr + inArgCount + extraInstrNum + iOutArg, outArgs[i].varList[j].Id(),
                        funcManager->GetFuncOut()[iOutArg].Id(), dep.reserveXnId);
                    iOutArg++;
                }
            }
        }

        return translated;
    }

    std::string CcuRepFuncCall::Describe() { return StringFormat("FuncCall[%s]", label.c_str()); }

    int32_t CcuRepFuncCall::GetCallLayer() { return funcBlock == nullptr ? FUNC_NEST_MAX : funcBlock->GetCallLayer(); }

}; // namespace CcuRep
}; // namespace Hccl

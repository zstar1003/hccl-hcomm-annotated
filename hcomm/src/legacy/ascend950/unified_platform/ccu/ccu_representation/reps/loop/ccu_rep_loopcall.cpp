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

#include "string_util.h"
#include "exception_util.h"
#include "ccu_api_exception.h"

namespace Hccl {
namespace CcuRep {

    CcuRepLoopCall::CcuRepLoopCall(const std::string& label) : label(label) { type = CcuRepType::LOOP_CALL; }

    const std::string& CcuRepLoopCall::GetLabel() const { return label; }

    void CcuRepLoopCall::Reference(std::shared_ptr<CcuRepLoopBlock> refRep) { loopBlock = refRep; }

    void CcuRepLoopCall::SetInArg(const Variable& var)
    {
        inArgCount++;
        inArgInstrCount++;
        inArgs.push_back(CcuRepArg(var));
    }

    void CcuRepLoopCall::SetInArg(const std::vector<Variable>& varList)
    {
        inArgCount += varList.size();
        inArgInstrCount += varList.size();
        inArgs.push_back(CcuRepArg(varList));
    }

    void CcuRepLoopCall::SetInArg(const Memory& mem)
    {
        inArgCount++;
        inArgInstrCount += 2; // 传递Memory需要2条指令
        inArgs.push_back(CcuRepArg(mem));
    }

    void CcuRepLoopCall::SetInArg(const std::vector<Memory>& memList)
    {
        inArgCount += memList.size();
        inArgInstrCount += memList.size() * 2; // 传递Memory需要2条指令
        inArgs.push_back(CcuRepArg(memList));
    }

    uint16_t CcuRepLoopCall::InstrCount()
    {
        instrCount = inArgInstrCount;
        return instrCount;
    }

    bool CcuRepLoopCall::Translate(CcuInstr*& instr, uint16_t& instrId, const TransDep& dep)
    {
        this->instrId = instrId;
        translated = true;

        Hccl::CHECK_NULLPTR(loopBlock, "[CcuRepLoopCall::Translate] LoopBlock is nullptr!");

        if (!loopBlock->Translated()) {
            THROW<CcuApiException>("Reference To Invalid LoopBlock");
        }

        for (uint32_t i = 0; i < inArgs.size(); i++) {
            if (inArgs[i].type == CcuArgType::VARIABLE && loopBlock->GetArg(i).type == CcuArgType::VARIABLE) {
                LoadXXInstr(instr++, loopBlock->GetArg(i).var.Id(), inArgs[i].var.Id(), dep.reserveXnId);
            } else if (
                inArgs[i].type == CcuArgType::VARIABLE_LIST && loopBlock->GetArg(i).type == CcuArgType::VARIABLE_LIST) {
                if (inArgs[i].varList.size() != loopBlock->GetArg(i).varList.size()) {
                    THROW<CcuApiException>("Mismatched Arg Size");
                }
                for (uint32_t j = 0; j < inArgs[i].varList.size(); j++) {
                    LoadXXInstr(
                        instr++, loopBlock->GetArg(i).varList[j].Id(), inArgs[i].varList[j].Id(), dep.reserveXnId);
                }
            } else if (inArgs[i].type == CcuArgType::MEMORY && loopBlock->GetArg(i).type == CcuArgType::MEMORY) {
                LoadGSAGSAInstr(instr++, loopBlock->GetArg(i).mem.addr.Id(), inArgs[i].mem.addr.Id(), dep.reserveGsaId);
                LoadXXInstr(instr++, loopBlock->GetArg(i).mem.token.Id(), inArgs[i].mem.token.Id(), dep.reserveXnId);
            } else if (
                inArgs[i].type == CcuArgType::MEMORY_LIST && loopBlock->GetArg(i).type == CcuArgType::MEMORY_LIST) {
                if (inArgs[i].memList.size() != loopBlock->GetArg(i).memList.size()) {
                    THROW<CcuApiException>("Mismatched Arg Size");
                }
                for (uint32_t j = 0; j < inArgs[i].memList.size(); j++) {
                    LoadGSAGSAInstr(
                        instr++, loopBlock->GetArg(i).memList[j].addr.Id(), inArgs[i].memList[j].addr.Id(),
                        dep.reserveGsaId);
                    LoadXXInstr(
                        instr++, loopBlock->GetArg(i).memList[j].token.Id(), inArgs[i].memList[j].token.Id(),
                        dep.reserveXnId);
                }
            } else {
                THROW<CcuApiException>("Mismatched Arg Type");
            }
        }

        instrId += InstrCount();

        return translated;
    }

    std::string CcuRepLoopCall::Describe() { return StringFormat("LoopCall[%s]", label.c_str()); }

}; // namespace CcuRep
}; // namespace Hccl

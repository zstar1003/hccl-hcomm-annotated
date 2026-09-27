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

namespace Hccl {
namespace CcuRep {

    CcuRepJumpBase::CcuRepJumpBase(const std::string& label, const Variable& targetInstrId)
        : label(label),
          targetInstrId(targetInstrId)
    {}

    void CcuRepJumpBase::Reference(std::shared_ptr<CcuRepJumpLabel> refRep) { jumpLabel = refRep; }

    CcuRepJump::CcuRepJump(const std::string& label, const Variable& targetInstrId)
        : CcuRepJumpBase(label, targetInstrId)
    {
        type = CcuRepType::JUMP;
        instrCount = 2; // jump翻译需要2条指令
    }

    bool CcuRepJump::Translate(CcuInstr*& instr, uint16_t& instrId, const TransDep& dep)
    {
        if (this->instr == nullptr) {
            this->instrId = instrId;
            this->instr = instr;
            instr += instrCount;
            instrId += instrCount;
        }

        if (jumpLabel->Translated()) {
            LoadImdToXnInstr(this->instr + 0, targetInstrId.Id(), jumpLabel->StartInstrId());
            JumpInstr(this->instr + 1, targetInstrId.Id(), dep.reserveXnId, 1);

            translated = true;
        }

        return translated;
    }

    std::string CcuRepJump::Describe() { return StringFormat("Jump To Label[%s]", label.c_str()); }

    CcuRepJumpNE::CcuRepJumpNE(
        const std::string& label, const Variable& targetInstrId, const Variable& condition, uint64_t expected)
        : CcuRepJumpBase(label, targetInstrId),
          condition(condition),
          expected(expected)
    {
        type = CcuRepType::JUMP_NE;
        instrCount = 2; // jumpNE翻译需要2条指令
    }

    bool CcuRepJumpNE::Translate(CcuInstr*& instr, uint16_t& instrId, [[maybe_unused]] const TransDep& dep)
    {
        if (this->instr == nullptr) {
            this->instrId = instrId;
            this->instr = instr;
            instr += instrCount;
            instrId += instrCount;
        }

        if (jumpLabel->Translated()) {
            LoadImdToXnInstr(this->instr + 0, targetInstrId.Id(), jumpLabel->StartInstrId());
            JumpInstr(this->instr + 1, targetInstrId.Id(), condition.Id(), expected);

            translated = true;
        }

        return translated;
    }

    std::string CcuRepJumpNE::Describe()
    {
        return StringFormat(
            "Jump To Label[%s], When Condition[%u] Not equal to Expected[%lu]", label.c_str(), condition.Id(),
            expected);
    }

    CcuRepJumpEQ::CcuRepJumpEQ(
        const std::string& label, const Variable& targetInstrId, const Variable& condition, uint64_t expected)
        : CcuRepJumpBase(label, targetInstrId),
          condition(condition),
          expected(expected)
    {
        type = CcuRepType::JUMP_EQ;
        instrCount = 5; // jumpEQ翻译需要5条指令
    }

    bool CcuRepJumpEQ::Translate(CcuInstr*& instr, uint16_t& instrId, const TransDep& dep)
    {
        if (this->instr == nullptr) {
            this->instrId = instrId;
            this->instr = instr;
            instr += instrCount;
            instrId += instrCount;
        }

        if (jumpLabel->Translated()) {
            uint32_t localInstrIndex = 0;
            LoadImdToXnInstr(
                this->instr + localInstrIndex++, targetInstrId.Id(),
                this->instrId + 4); // 需要指向NOP位置，为输入指令Id + 4
            JumpInstr(this->instr + localInstrIndex++, targetInstrId.Id(), condition.Id(), expected);
            LoadImdToXnInstr(this->instr + localInstrIndex++, targetInstrId.Id(), jumpLabel->StartInstrId());
            JumpInstr(this->instr + localInstrIndex++, targetInstrId.Id(), dep.reserveXnId, 1);
            LoadImdToXnInstr(this->instr + localInstrIndex++, dep.reserveXnId, 0);

            translated = true;
        }

        return translated;
    }

    std::string CcuRepJumpEQ::Describe()
    {
        return StringFormat(
            "Jump To Label[%s], When Condition[%u] Be equal to Expected[%lu]", label.c_str(), condition.Id(), expected);
    }

}; // namespace CcuRep
}; // namespace Hccl

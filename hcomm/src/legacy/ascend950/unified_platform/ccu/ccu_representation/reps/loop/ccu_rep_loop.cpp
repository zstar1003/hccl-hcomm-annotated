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
#include <climits>

#include "string_util.h"
#include "exception_util.h"
#include "ccu_api_exception.h"

namespace Hccl {
namespace CcuRep {

    CcuRepLoop::CcuRepLoop(const std::string& label, const Variable& loopParam) : label(label), loopParam(loopParam)
    {
        type = CcuRepType::LOOP;
        instrCount = 1; // loop翻译需要1条指令
    }

    const std::string& CcuRepLoop::GetLabel() const { return label; }

    void CcuRepLoop::Reference(std::shared_ptr<CcuRepLoopBlock> refRep) { loopBlock = refRep; }

    std::shared_ptr<CcuRepBase> CcuRepLoop::SetLoopParam(Executor executor, Variable var)
    {
        return std::make_shared<CcuRepSetLoop>(loopParam, executor, var);
    }

    bool CcuRepLoop::Translate(CcuInstr*& instr, uint16_t& instrId, [[maybe_unused]] const TransDep& dep)
    {
        this->instrId = instrId;
        translated = true;

        Hccl::CHECK_NULLPTR(loopBlock, "[CcuRepLoop::Translate] LoopBlock is nullptr!");

        if (!loopBlock->Translated()) {
            THROW<CcuApiException>("Reference To Invalid LoopBlock");
        }

        uint16_t startInstrId = loopBlock->StartInstrId();
        uint16_t loopInstrCount = loopBlock->InstrCount();
        if (loopInstrCount == 0) {
            HCCL_ERROR(
                "[CcuRepLoop][Translate] loopInstrCount[%u] is 0, which causes underflow in endInstrId calculation.",
                loopInstrCount);
            return false;
        }
        if (startInstrId > USHRT_MAX - loopInstrCount) {
            HCCL_ERROR(
                "[CcuRepLoop][Translate] startInstrId[%u] + loopInstrCount[%u] exceeds the maximum value of unsigned "
                "short int.",
                startInstrId, loopInstrCount);
            return false;
        }
        if (instrId > USHRT_MAX - instrCount) {
            HCCL_ERROR(
                "[CcuRepLoop][Translate] instrId[%u] + instrCount[%u] exceeds the maximum value of unsigned short int.",
                instrId, instrCount);
            return false;
        }
        uint16_t endInstrId = startInstrId + loopInstrCount - 1;

        LoopInstr(instr++, startInstrId, endInstrId, loopParam.Id());

        instrId += instrCount;

        return translated;
    }

    std::string CcuRepLoop::Describe() { return StringFormat("Loop reference to [%s]", label.c_str()); }

}; // namespace CcuRep
}; // namespace Hccl

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
#include "ccu_assist.h"
#include <climits>

#include "string_util.h"

namespace Hccl {
namespace CcuRep {

    CcuRepSetLoop::CcuRepSetLoop(const Variable& loopParam, const Executor& executor, const Variable& var)
        : loopParam(loopParam),
          executor(executor),
          var(var)
    {
        type = CcuRepType::SET_LOOP;
        instrCount = 2; // set loop 指令数量为2
    }

    bool CcuRepSetLoop::Translate(CcuInstr*& instr, uint16_t& instrId, [[maybe_unused]] const TransDep& dep)
    {
        this->instrId = instrId;
        translated = true;

        LoadImdToXnInstr(instr++, loopParam.Id(), GetLoopParam(executor.Id(), 0, 0));
        LoadXXInstr(instr++, loopParam.Id(), loopParam.Id(), var.Id());

        if (instrId > USHRT_MAX - instrCount) {
            THROW<InternalException>(StringFormat(
                "[CcuRepSetLoop][Translate] instrId[%u] + instrCount[%u] exceeds the "
                "maximum value of unsigned short int.",
                instrId, instrCount));
        }
        instrId += instrCount;

        return translated;
    }

    std::string CcuRepSetLoop::Describe()
    {
        return StringFormat(
            "loopParam[%u] = var[%u], execute on LoopEngine[%u]", loopParam.Id(), var.Id(), executor.Id());
    }

}; // namespace CcuRep
}; // namespace Hccl

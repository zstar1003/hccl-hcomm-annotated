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

namespace Hccl {
namespace CcuRep {

    CcuRepLocPostSem::CcuRepLocPostSem(const MaskSignal& sem, uint16_t mask) : sem(sem), mask(mask)
    {
        type = CcuRepType::LOC_POST_SEM;
        instrCount = 1;
    }

    bool CcuRepLocPostSem::Translate(CcuInstr*& instr, uint16_t& instrId, [[maybe_unused]] const TransDep& dep)
    {
        this->instrId = instrId;
        translated = true;

        SetCKEInstr(instr++, sem.Id(), mask, 0, 0, 1);

        if (instrId > USHRT_MAX - instrCount) {
            THROW<InternalException>(StringFormat(
                "[CcuRepLocPostSem][Translate] instrId[%u] + instrCount[%u] exceeds the "
                "maximum value of unsigned short int.",
                instrId, instrCount));
        }
        instrId += instrCount;

        return translated;
    }

    std::string CcuRepLocPostSem::Describe() { return StringFormat("Set Sem[%u], mask[%04x]", sem.Id(), mask); }

}; // namespace CcuRep
}; // namespace Hccl

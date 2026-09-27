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

    CcuRepRemWaitSem::CcuRepRemWaitSem(
        const CcuTransport& transport, uint16_t semIndex, uint16_t mask, bool isProfiling)
        : transport(transport),
          semIndex(semIndex),
          mask(mask),
          isProfiling(isProfiling)
    {
        type = CcuRepType::REM_WAIT_SEM;
        instrCount = 1;
    }

    bool CcuRepRemWaitSem::Translate(CcuInstr*& instr, uint16_t& instrId, [[maybe_unused]] const TransDep& dep)
    {
        this->instrId = instrId;
        translated = true;

        // 需要profiling的使用SetCKEInstr, 否则使用ClearCKEInstr
        if (isProfiling) {
            SetCKEInstr(instr++, 0, 0, transport.GetLocCkeByIndex(semIndex), mask, 1);
        } else {
            ClearCKEInstr(instr++, 0, 0, transport.GetLocCkeByIndex(semIndex), mask, 1);
        }
        CHK_PRT_THROW(
            instrId > UINT16_MAX - instrCount,
            HCCL_ERROR(
                "[CcuRepRemWaitSem::Translate]uint16 integer overflow occurs, instrId = [%hu], instrCount = [%hu]",
                instrId, instrCount),
            InternalException, "integer overflow");
        instrId += instrCount;

        return translated;
    }

    std::string CcuRepRemWaitSem::Describe()
    {
        return StringFormat("Wait, Use semIndex[%u] and mask[%04x]", semIndex, mask);
    }

}; // namespace CcuRep
}; // namespace Hccl

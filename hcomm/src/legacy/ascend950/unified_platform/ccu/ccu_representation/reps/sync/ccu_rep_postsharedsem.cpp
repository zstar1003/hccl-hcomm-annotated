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

    CcuRepPostSharedSem::CcuRepPostSharedSem(const MaskSignal& sem, uint16_t mask) : sem(sem), mask(mask)
    {
        type = CcuRepType::POST_SHARED_SEM;
        instrCount = 1;
    }

    bool CcuRepPostSharedSem::Translate(CcuInstr*& instr, uint16_t& instrId, const TransDep& dep)
    {
        this->instrId = instrId;
        translated = true;

        if (sem.DieId() != dep.dieId) {
            SyncCKEInstr(instr++, sem.Id(), dep.reserveCkeId, mask, dep.reserveChannalId[1], 0, 0, 0, 0, 1);
        } else {
            SetCKEInstr(instr++, sem.Id(), mask, 0, 0, 1);
        }
        CHK_PRT_THROW(
            (instrId > UINT16_MAX - instrCount),
            HCCL_ERROR(
                "[CcuRepPostSharedSem::Translate]uint16 integer overflow occurs, instrId = [%hu], instrCount = [%hu]",
                instrId, instrCount),
            InternalException, "integer overflow");
        instrId += instrCount;

        return translated;
    }

    std::string CcuRepPostSharedSem::Describe()
    {
        return StringFormat("Post, Use semIndex[%u] and mask[%04x]", sem.Id(), mask);
    }

}; // namespace CcuRep
}; // namespace Hccl

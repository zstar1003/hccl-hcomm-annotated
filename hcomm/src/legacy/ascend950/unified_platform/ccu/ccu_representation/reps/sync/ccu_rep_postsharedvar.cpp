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

    CcuRepPostSharedVar::CcuRepPostSharedVar(
        const Variable& srcVar, const Variable& dstVar, const MaskSignal& sem, uint16_t mask)
        : srcVar(srcVar),
          dstVar(dstVar),
          sem(sem),
          mask(mask)
    {
        type = CcuRepType::POST_SHARED_VAR;
        instrCount = 2; // 指令数为2个
    }

    bool CcuRepPostSharedVar::Translate(CcuInstr*& instr, uint16_t& instrId, const TransDep& dep)
    {
        this->instrId = instrId;
        translated = true;

        if (sem.DieId() != dep.dieId) {
            SyncXnInstr(instr++, dstVar.Id(), srcVar.Id(), dep.reserveChannalId[1], sem.Id(), mask, 0, 0, 0, 0, 1);
            LoadImdToXnInstr(instr++, dep.reserveXnId, 0);
        } else {
            LoadXXInstr(instr++, dstVar.Id(), srcVar.Id(), dep.reserveXnId);
            SetCKEInstr(instr++, sem.Id(), mask, 0, 0, 1);
        }
        CHK_PRT_THROW(
            (instrId > UINT16_MAX - instrCount),
            HCCL_ERROR(
                "[CcuRepPostSharedVar::Translate]uint16 integer overflow occurs, instrId = [%hu], instrCount = [%hu]",
                instrId, instrCount),
            InternalException, "integer overflow");
        instrId += instrCount;

        return translated;
    }

    std::string CcuRepPostSharedVar::Describe()
    {
        return StringFormat("Post Shared Variable[%u], from Variable[%u]", dstVar.Id(), srcVar.Id());
    }

}; // namespace CcuRep
}; // namespace Hccl

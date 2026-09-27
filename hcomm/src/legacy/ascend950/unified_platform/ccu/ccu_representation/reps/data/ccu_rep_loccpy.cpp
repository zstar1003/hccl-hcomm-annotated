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

    CcuRepLocCpy::CcuRepLocCpy(Memory dst, Memory src, Variable len, MaskSignal sem, uint16_t mask)
        : dst(dst),
          src(src),
          len(len),
          sem(sem),
          mask(mask)
    {
        type = CcuRepType::LOCAL_CPY;
        instrCount = 1;
    }

    CcuRepLocCpy::CcuRepLocCpy(
        Memory dst, Memory src, Variable len, uint16_t dataType, uint16_t opType, MaskSignal sem, uint16_t mask)
        : dst(dst),
          src(src),
          len(len),
          sem(sem),
          mask(mask),
          dataType(dataType),
          opType(opType),
          reduceFlag(1)
    {
        type = CcuRepType::LOCAL_REDUCE;
        instrCount = 1;
    }

    bool CcuRepLocCpy::Translate(CcuInstr*& instr, uint16_t& instrId, const TransDep& dep)
    {
        this->instrId = instrId;
        translated = true;

        if (reduceFlag == 0) {
            TransLocMemToLocMemInstr(
                instr++, dst.addr.Id(), dst.token.Id(), src.addr.Id(), src.token.Id(), len.Id(),
                dep.reserveChannalId[0], sem.Id(), mask, 0, 0, 1, 1);
        } else {
            // 这个翻译需要验证
            TransLocMemToRmtMemInstr(
                instr++, dst.addr.Id(), dst.token.Id(), src.addr.Id(), src.token.Id(), len.Id(),
                dep.reserveChannalId[0], dataType, opType, sem.Id(), mask, 0, 0, 1, 1, reduceFlag);
        }

        instrId += instrCount;

        return translated;
    }

    std::string CcuRepLocCpy::Describe()
    {
        return StringFormat(
            "Read Memory[%u] to Memory[%u], length[%u], set sem[%u] with mask[%04x], dataType[%u], opType[%u]",
            src.addr.Id(), dst.addr.Id(), len.Id(), sem.Id(), mask, dataType, opType);
    }

}; // namespace CcuRep
}; // namespace Hccl

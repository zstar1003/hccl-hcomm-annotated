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

    CcuRepWrite::CcuRepWrite(
        const CcuTransport& transport, Memory rem, Memory loc, Variable len, MaskSignal sem, uint16_t mask)
        : transport(transport),
          rem(rem),
          loc(loc),
          len(len),
          sem(sem),
          mask(mask)
    {
        type = CcuRepType::WRITE;
        instrCount = 1;
    }

    CcuRepWrite::CcuRepWrite(
        const CcuTransport& transport, Memory rem, Memory loc, Variable len, uint16_t dataType, uint16_t opType,
        MaskSignal sem, uint16_t mask)
        : transport(transport),
          rem(rem),
          loc(loc),
          len(len),
          sem(sem),
          mask(mask),
          dataType(dataType),
          opType(opType),
          reduceFlag(1)
    {
        type = CcuRepType::WRITE;
        instrCount = 1;
    }

    bool CcuRepWrite::Translate(CcuInstr*& instr, uint16_t& instrId, [[maybe_unused]] const TransDep& dep)
    {
        this->instrId = instrId;
        translated = true;

        TransLocMemToRmtMemInstr(
            instr++, rem.addr.Id(), rem.token.Id(), loc.addr.Id(), loc.token.Id(), len.Id(), transport.GetChannelId(),
            dataType, opType, sem.Id(), mask, 0, 0, 1, 1, reduceFlag);

        instrId += instrCount;

        return translated;
    }

    std::string CcuRepWrite::Describe()
    {
        return StringFormat(
            "Write Memory[%u] to Memory[%u], length[%u], set sem[%u] with mask[%04x], dataType[%u], opType[%u]",
            loc.addr.Id(), rem.addr.Id(), len.Id(), sem.Id(), mask, dataType, opType);
    }

}; // namespace CcuRep
}; // namespace Hccl

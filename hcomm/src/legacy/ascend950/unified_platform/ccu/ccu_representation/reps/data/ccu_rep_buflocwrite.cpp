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

    CcuRepBufLocWrite::CcuRepBufLocWrite(CcuBuffer src, Memory dst, Variable len, MaskSignal sem, uint16_t mask)
        : src(src),
          dst(dst),
          len(len),
          sem(sem),
          mask(mask)
    {
        type = CcuRepType::BUF_LOC_WRITE;
        instrCount = 1;
    }

    bool CcuRepBufLocWrite::Translate(CcuInstr*& instr, uint16_t& instrId, const TransDep& dep)
    {
        this->instrId = instrId;
        translated = true;

        TransLocMSToLocMemInstr(
            instr++, dst.addr.Id(), dst.token.Id(), src.Id(), len.Id(), dep.reserveChannalId[0], sem.Id(), mask, 0, 0,
            1, 1);

        instrId += instrCount;

        return translated;
    }

    std::string CcuRepBufLocWrite::Describe()
    {
        return StringFormat(
            "Write CcuBuffer[%u] To Loc Mem[%u], len[%u], sem[%u], mask[%04x]", src.Id(), dst.addr.Id(), len.Id(),
            sem.Id(), mask);
    }

}; // namespace CcuRep
}; // namespace Hccl

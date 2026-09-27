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

    CcuRepLoadArg::CcuRepLoadArg(const Variable& var, uint16_t argId) : var(var), argId(argId)
    {
        type = CcuRepType::LOAD_ARG;
        instrCount = 1;
    }

    uint16_t CcuRepLoadArg::GetVarId() const { return var.Id(); }

    bool CcuRepLoadArg::Translate(CcuInstr*& instr, uint16_t& instrId, const TransDep& dep)
    {
        this->instrId = instrId;
        translated = true;

        if (dep.isFuncBlock) {
            // Xn(var) = Xn(loadXnId) + 0
            LoadXXInstr(instr++, var.Id(), dep.loadXnId, dep.reserveXnId);
        } else {
            LoadSqeArgsToXnInstr(instr++, var.Id(), argId);
        }

        instrId += instrCount;

        return translated;
    }

    std::string CcuRepLoadArg::Describe() { return StringFormat("Variable[%u] = Arg[%u]", var.Id(), argId); }

}; // namespace CcuRep
}; // namespace Hccl

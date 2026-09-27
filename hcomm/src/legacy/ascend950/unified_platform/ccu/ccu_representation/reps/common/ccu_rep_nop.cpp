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

    CcuRepNop::CcuRepNop()
    {
        type = CcuRepType::NOP;
        instrCount = 1;
    }

    bool CcuRepNop::Translate(CcuInstr*& instr, uint16_t& instrId, const TransDep& dep)
    {
        this->instrId = instrId;
        translated = true;

        LoadImdToXnInstr(instr++, dep.reserveXnId, 0);

        instrId += instrCount;

        return translated;
    }

    std::string CcuRepNop::Describe() { return StringFormat("Nop"); }

}; // namespace CcuRep
}; // namespace Hccl

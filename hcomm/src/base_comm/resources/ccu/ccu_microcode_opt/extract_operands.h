/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_MICROCODE_OPT_EXTRACT_OPERANDS_H
#define CCU_MICROCODE_OPT_EXTRACT_OPERANDS_H

#include <cstdint>
#include <vector>

#include "ccu_microcode_v1.h"

namespace hcomm {
namespace CcuOpt {

    enum class RegType : uint8_t {
        XN = 0,  // 64-bit 标量寄存器, 8 路 Bank, 主要优化对象
        MS = 1,  // Memory Scratchpad
        CKE = 2, // Completion / Kick Event
    };

    struct RegOperand {
        RegType type{};
        uint16_t regId = 0;
        bool isDef = false; // true = write, false = read

        bool operator==(const RegOperand& other) const
        {
            return type == other.type && regId == other.regId && isDef == other.isDef;
        }
    };

    std::vector<RegOperand> ExtractOperandsV2(const CcuRep::CcuInstr& instr);

} // namespace CcuOpt
} // namespace hcomm

#endif // CCU_MICROCODE_OPT_EXTRACT_OPERANDS_H

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_INSTR_INFO_H
#define HCCL_CCU_INSTR_INFO_H

#include <cstdint>
#include <vector>
#include "ccu_microcode.h"

namespace Hccl {
namespace CcuRep {

    struct CcuInstrInfo {
        std::vector<CcuInstr> instrVec;
        uint16_t startInstrId{0};
        uint16_t instrCount{0};
        uint16_t missionStartInstrId{0};
        uint16_t missionInstrCount{0};
    };

}; // namespace CcuRep
}; // namespace Hccl

#endif // HCCL_CCU_INSTR_INFO_H

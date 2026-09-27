/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_MICROCODE_OPT_BARRIER_CONFIG_H
#define CCU_MICROCODE_OPT_BARRIER_CONFIG_H

#include <cstdint>

namespace hcomm {
namespace CcuOpt {

    // V2 microcode opcode 常量, 汇总自 ccu_microcode_v2_.cc.
    namespace InstrCodeV2 {
        constexpr uint16_t LOAD_TYPE = 0x0;
        constexpr uint16_t CTRL_TYPE = 0x1;
        constexpr uint16_t TRANS_TYPE = 0x2;
        constexpr uint16_t REDUCE_TYPE = 0x3;

        // LOAD_TYPE
        constexpr uint16_t LOADSQEARGSTOX_CODE = 0x1;
        constexpr uint16_t LOADIMDTOX_CODE = 0x2;
        constexpr uint16_t LOADX_CODE = 0x6;  // half-rtt LoadAddImm 翻译目标 (v2.loadStoreX)
        constexpr uint16_t STOREX_CODE = 0x7; // half-rtt AddImmStore 翻译目标 (v2.loadStoreX)
        constexpr uint16_t CLEARX_CODE = 0x8; // half-rtt CascCntClear 翻译目标 (v2.clearX)
        constexpr uint16_t NOP_CODE = 0x9;
        constexpr uint16_t LOAD_CODE = 0xA;
        constexpr uint16_t STORE_CODE = 0xB;
        constexpr uint16_t ADD_CODE = 0xD;
        constexpr uint16_t SUB_CODE = 0xE;
        constexpr uint16_t MUL_CODE = 0xF;
        constexpr uint16_t AND_CODE = 0x10;
        constexpr uint16_t OR_CODE = 0x11;
        constexpr uint16_t NOT_CODE = 0x12;
        constexpr uint16_t XOR_CODE = 0x13;
        constexpr uint16_t SHL_CODE = 0x14;
        constexpr uint16_t SHR_CODE = 0x15;
        constexpr uint16_t POPCNT_CODE = 0x16;

        // CTRL_TYPE
        constexpr uint16_t LOOP_CODE = 0x0;
        constexpr uint16_t LOOPGROUP_CODE = 0x1;
        constexpr uint16_t SETCKBIT_CODE = 0x2;
        constexpr uint16_t CLEARCKBIT_CODE = 0x4;
        constexpr uint16_t JMP_CODE = 0x5;
        constexpr uint16_t WAIT_CODE = 0x7;
        constexpr uint16_t FENCE_CODE = 0x8;

        // TRANS_TYPE
        constexpr uint16_t TRANSLOCMEMTOLOCMS_CODE = 0x0;
        constexpr uint16_t TRANSLOCMSTOLOCMEM_CODE = 0x2;
        constexpr uint16_t TRANSLOCMSTOLOCMS_CODE = 0x5;
        constexpr uint16_t TRANSLOCMEMTOLOCMEM_CODE = 0x6;
        constexpr uint16_t TRANSMEM_CODE = 0x10;
        constexpr uint16_t SYNCWTX_CODE = 0xD;
        constexpr uint16_t SYNCATX_CODE = 0xE;

        // REDUCE_TYPE
        constexpr uint16_t REDUCE_ADD_CODE = 0x0;
        constexpr uint16_t REDUCE_MAX_CODE = 0x1;
        constexpr uint16_t REDUCE_MIN_CODE = 0x2;
    } // namespace InstrCodeV2

} // namespace CcuOpt
} // namespace hcomm

#endif // CCU_MICROCODE_OPT_BARRIER_CONFIG_H

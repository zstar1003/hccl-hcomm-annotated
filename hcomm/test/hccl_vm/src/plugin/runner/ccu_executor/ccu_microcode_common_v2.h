/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SIM_CCU_MICROCODE_COMMON_V2_H
#define HCCL_SIM_CCU_MICROCODE_COMMON_V2_H

#include <cstdint>

#include "sim_common_defs.h"

namespace SimCcuV2 {
// 微码指令
constexpr uint16_t LOAD_TYPE = 0x0;
constexpr uint16_t CTRL_TYPE = 0x1;
constexpr uint16_t TRANS_TYPE = 0x2;
constexpr uint16_t REDUCE_TYPE = 0x3;

constexpr uint16_t LOADSQEARGSTOXN_CODE = 0x1;
constexpr uint16_t LOADIMDTOX_CODE = 0x2;
constexpr uint16_t LOADX_CODE = 0x6;
constexpr uint16_t STOREX_CODE = 0x7;
constexpr uint16_t CLEARX_CODE = 0x8;
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

// 7
constexpr uint16_t LOOP_CODE = 0x0;
constexpr uint16_t LOOPGROUP_CODE = 0x1;
constexpr uint16_t SETCKBIT_CODE = 0x2;
constexpr uint16_t CLEARCKE_CODE = 0x4;
constexpr uint16_t JMP_CODE = 0x5;
constexpr uint16_t WAIT_CODE = 0x7;
constexpr uint16_t FENCE_CODE = 0x8;

// 7
constexpr uint16_t TRANSLOCMEMTOLOCMS_CODE = 0x0;
constexpr uint16_t TRANSLOCMSTOLOCMEM_CODE = 0x2;
constexpr uint16_t TRANSLOCMSTOLOCMS_CODE = 0x5;
constexpr uint16_t TRANSLOCMEMTOLOCMEM_CODE = 0x6;
constexpr uint16_t TRANSMEM_CODE = 0x10;
constexpr uint16_t SYNCXNWTX_CODE = 0xd;
constexpr uint16_t SYNCATX_CODE = 0xe;

constexpr uint16_t REDUCEADD_CODE = 0x0;
constexpr uint16_t REDUCEMAX_CODE = 0x1;
constexpr uint16_t REDUCEMIN_CODE = 0x2;

// CCU资源规格
const uint16_t INVALID_INSTR_START_INDEX = 0xFFFF;
constexpr uint8_t MAX_CCU_CHANNEL_NUM = 128;
constexpr uint16_t CCU_RESOURCE_MS_NUM = 1536;
constexpr uint32_t CCU_RESOURCE_MS_SIZE = CCU_RESOURCE_MS_NUM * HcclSim::BYTE_NUM_4K;
constexpr uint16_t CCU_RESOURCE_GSA_NUM = 3024;        // max:3072
constexpr uint16_t CCU_RESOURCE_XN_NUM = 3024;         // max:3072
constexpr uint16_t CCU_RESOURCE_CKE_NUM = 1008;        // max:1024
constexpr uint16_t CCU_RESOURCE_LOOP_ENGINE_NUM = 192; // max:200

constexpr uint16_t CCU_RESOURCE_GSA_MAX = 4096;
constexpr uint16_t CCU_RESOURCE_XN_MAX = 4096;
constexpr uint16_t CCU_RESOURCE_CKE_MAX = 1024;
constexpr uint16_t CCU_RESOURCE_LOOP_ENGINE_MAX = 200;
constexpr uint16_t CCU_RESOURCE_MEM_SLICE_4K = 4096;
}; // namespace SimCcuV2

#endif // HCCL_SIM_CCU_MICROCODE_COMMON_V2_H

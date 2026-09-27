/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SIM_CCU_SIMULATOR_BASE_H
#define HCCL_SIM_CCU_SIMULATOR_BASE_H
#include <cstdint>

enum CcuExecState {
    EXEC_NORMAL_INSTR,
    EXEC_LOOPGROUP_INSTR,
    EXEC_LOOP_INSTR,
    EXEC_JUMP_INSTR,
    EXEC_SUCCESS,
    EXEC_FAIL
};
enum ReduceAddDataType {
    ADD_FP32,
    ADD_FP16,
    ADD_BF16,
    ADD_HIF8,
    ADD_FP8_E4M3,
    ADD_FP8_E5M2,
    ADD_INT8,
    ADD_UINT8,
    ADD_INT16,
    ADD_INT32,
    ADD_RESERVED
};
enum TransMemReduceDataTypeV1 {
    INT8_V1,
    INT16_V1,
    INT32_V1,
    UINT8_V1,
    UINT16_V1,
    UINT32_V1,
    FP16_NORMAL_V1,
    FP32_V1,
    BF16_V1,
    FP16_SAT_V1,
    INVALID_V1
};
enum TransMemReduceDataTypeV2 {
    INT8_V2,
    INT16_V2,
    INT32_V2,
    INVALID_1,
    INVALID_2,
    UINT32_V2,
    FP16_NORMAL_V2,
    FP32_V2,
    BF16_V2,
    FP16_SAT_V2,
    INVALID_V2
};
enum ReduceMaxMinDataType {
    MAX_MIN_FP32,
    MAX_MIN_FP16,
    MAX_MIN_BF16,
    MAX_MIN_RESERVED1,
    MAX_MIN_RESERVED2,
    MAX_MIN_RESERVED3,
    MAX_MIN_INT8,
    MAX_MIN_UINT8,
    MAX_MIN_INT16,
    MAX_MIN_INT32,
    MAX_MIN_RESERVED4
};

struct LoopStatusInfo {
    uint16_t loopCurInstrId{0};
    uint16_t loopStartInstrId{0};
    uint16_t loopEndInstrId{0};
    uint16_t loopExecCount{0};
    uint16_t curLoopRound{0};    // 用于记录当前loop循环的执行轮次
    uint32_t loopGsaIterStep{0}; // loop迭代时，gsa地址的偏移步长
    uint16_t loopExtendIndex{0}; // loop的扩展index
    uint16_t loopCtxId{0};       // loop Context ID
};

struct LoopGroupInfo {
    uint16_t startLoopId_{0};
    uint16_t loopNum_{0};
    uint16_t loopOffset_{0};
    uint16_t loopExtendNum_{0};
    uint32_t gsaOffset_{0};     // loop指令执行过程中，gsa地址的偏移量
    uint16_t msOffset_{0};      // loop指令执行过程中，ms的偏移量
    uint16_t ckeOffset_{0};     // loop指令执行过程中，cke的偏移量
    uint16_t xnIdOffset{0};     // loop指令执行过程中，cke的偏移量
    LoopStatusInfo loopStatus_; // loop指令执行状态信息
};

struct ExecuteStatusInfo {
    CcuExecState state{CcuExecState::EXEC_NORMAL_INSTR};
    uint16_t curInstrId{0};
    uint16_t startInstrId{0};
    uint16_t endInstrId{0};
    uint16_t jumpInstrId{0};
    uint16_t instrType{0}; // 当前指令的类型(主要用于记录当前执行是否为Loop)
    LoopGroupInfo loopGroupState;
    bool waitCKE{false};     // 是否需要等待CKE
    bool initialized{false}; // 是否已经初始化
};

#endif // HCCL_SIM_CCU_SIMULATOR_BASE_H

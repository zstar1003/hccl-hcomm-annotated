/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_REPRESENTATION_TYPE_H
#define CCU_REPRESENTATION_TYPE_H

namespace hcomm {
namespace CcuRep {

    enum class CcuRepType {
        BASE,
        BLOCK,
        NOP,

        LOAD,
        STORE,
        LOAD_ARG,
        LOAD_VAR,
        STORE_VAR,

        ASSIGN,
        ADD,
        MUL,
        SUB,
        SET_LOOP,

        AND,
        OR,
        XOR,
        NOT,

        JUMP,
        JUMP_NE,
        JUMP_EQ,
        JUMP_LT,
        JUMP_LE,
        JUMP_GT,
        JUMP_GE,
        JUMP_LABEL,

        FUNC_CALL,
        FUNC_BLOCK,

        LOOP_CALL,
        LOOP,
        LOOPGROUP,
        LOOP_BLOCK,
        LOOPGROUP_BLOCK,

        LOC_RECORD_EVENT,
        LOC_WAIT_EVENT,
        LOC_WAIT_NOTIFY,
        REM_POST_SEM,
        REM_WAIT_SEM,
        REM_POST_VAR,
        REM_WAIT_GROUP,

        READ,
        WRITE,
        LOCAL_CPY,
        LOCAL_REDUCE,
        REM_MEM,

        BUF_READ,
        BUF_WRITE,
        BUF_LOC_READ,
        BUF_LOC_WRITE,
        BUF_REDUCE,

        SHR,
        SHL,

        RECORD_SHARED_NOTIFY,

        // 0.5rtt专用
        WRITE_VAR_ATOMIC,
        WRITE_WITH_CNT_INC,
        CASC_CNT_WAIT,
        CASC_CNT_CLEAR,
        LOAD_ADD_IMM,
        STORE_ADD_IMM,
    };

    enum class AssignSubType { INVALID, IMD_TO_VARIABLE, IMD_TO_ADDR, VAR_TO_ADDR, ADDR_TO_ADDR, VAR_TO_VAR };

    enum class AddSubType {
        INVALID,
        ADDR_PLUS_VAR_TO_ADDR,
        ADDR_PLUS_ADDR_TO_ADDR,
        VAR_PLUS_VAR_TO_VAR,
        SELF_ADD_ADDRESS,
        SELF_ADD_VARIABLE,
        VAR_PLUS_IMMED_TO_VAR,
        ADDR_PLUS_IMMED_TO_ADDR,
        VAR_PLUS_VAR_TO_ADDR,
        SELF_ADD_IMMED_ADDRESS,
        SELF_ADD_IMMED_VARIABLE,
        VAR_PLUS_IMMED_TO_ADDR,
        ADDR_PLUS_IMMED_TO_VAR,
        ADDR_PLUS_ADDR_TO_VAR
    };

    enum class MulSubType {
        INVALID,
        VAR_MUL_VAR_TO_VAR,
        VAR_MUL_IMMED_TO_VAR,
        SELF_MUL_VAR_VARIABLE,
        SELF_MUL_IMMED_VARIABLE,
        VAR_MUL_VAR_TO_ADDR,
        VAR_MUL_ADDR_TO_ADDR,
        VAR_MUL_IMMED_TO_ADDR,
        ADDR_MUL_IMMED_TO_ADDR,
        SELF_MUL_VAR_ADDRESS,
        SELF_MUL_IMMED_ADDRESS,
        ADDR_MUL_IMMED_TO_VAR
    };

    enum class MinusSubType {
        INVALID,
        VAR_MINUS_VAR_TO_VAR,
        VAR_MINUS_IMMED_TO_VAR,
        SELF_SUB_VAR_VARIABLE,
        SELF_SUB_IMMED_VARIABLE,
        ADDR_MINUS_VAR_TO_ADDR,
        ADDR_MINUS_IMMED_TO_ADDR,
        SELF_SUB_VAR_ADDRESS,
        SELF_SUB_IMMED_ADDRESS,
        VAR_MINUS_IMMED_TO_ADDR,
        ADDR_MINUS_IMMED_TO_VAR
    };

    enum class AndSubType {
        INVALID,
        VAR_AND_VAR_TO_VAR,
        SELF_AND_VAR_VARIABLE,
    };

    enum class OrSubType { INVALID, VAR_OR_VAR_TO_VAR, SELF_OR_VAR_VARIABLE };

    enum class XorSubType { INVALID, VAR_XOR_VAR_TO_VAR, SELF_XOR_VAR_VARIABLE };

    enum class NotSubType {
        INVALID,
        VAR_EQUALS_NOT_VAR,
    };

    enum class ShiftType { LOGICAL_SHIFT, ARITHMETIC_SHIFT, CIRCULAR_SHIFT, INVALID };

    enum class ShiftSubType {
        INVALID,
        VAR_EQUALS_VAR_SHIFT_VAR,
        VAR_SHIFT_ASSIGN_VAR,
        ADDR_EQUALS_VAR_SHIFT_VAR,
        ADDR_SHIFT_ASSIGN_VAR
    };

}; // namespace CcuRep
}; // namespace hcomm

#endif // _CCU_REPRESENTATION_TYPE_H

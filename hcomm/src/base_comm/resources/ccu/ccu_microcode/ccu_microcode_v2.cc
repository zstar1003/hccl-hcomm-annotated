/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_microcode_v1.h"

#include <unordered_map>

#include "log.h"
#include <string_util.h>

namespace {
constexpr uint16_t LOAD_TYPE = 0x0;

constexpr uint16_t LOADSQEARGSTOX_CODE = 0x1;
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

constexpr uint16_t CTRL_TYPE = 0x1;

constexpr uint16_t LOOP_CODE = 0x0;
constexpr uint16_t LOOPGROUP_CODE = 0x1;
constexpr uint16_t SETCKBIT_CODE = 0x2;
constexpr uint16_t CLEARCKBIT_CODE = 0x4;
constexpr uint16_t JMP_CODE = 0x5;
constexpr uint16_t WAIT_CODE = 0x7;
constexpr uint16_t FENCE_CODE = 0x8;

constexpr uint16_t TRANS_TYPE = 0x2;

constexpr uint16_t TRANSLOCMEMTOLOCMS_CODE = 0x0;
constexpr uint16_t TRANSLOCMSTOLOCMEM_CODE = 0x2;
constexpr uint16_t TRANSLOCMSTOLOCMS_CODE = 0x5;
constexpr uint16_t TRANSLOCMEMTOLOCMEM_CODE = 0x6;
constexpr uint16_t TRANSMEM_CODE = 0x10;
constexpr uint16_t SYNCWTX_CODE = 0xD;
constexpr uint16_t SYNCATX_CODE = 0xE;

constexpr uint16_t REDUCE_TYPE = 0x3;

constexpr uint16_t REDUCE_ADD_CODE = 0x0;
constexpr uint16_t REDUCE_MAX_CODE = 0x1;
constexpr uint16_t REDUCE_MIN_CODE = 0x2;

constexpr uint16_t JMP_TARGET_PC_IF_BRANCH = 5;  // Jmp跳转到IF分支的目标PC
constexpr uint16_t JMP_TARGET_PC_NOP_BRANCH = 3; // Jmp跳转到Nop的目标PC
constexpr uint16_t JMP_COND_TYPE_IF = 2;         // Jmp条件类型: IF分支
constexpr uint16_t JMP_COND_TYPE_UNCOND = 6;     // Jmp条件类型: 无条件
} // namespace

namespace hcomm {
namespace CcuRep {
    namespace CcuV2 {
        // *XnId = *sqeArgsId
        void LoadSqeArgsToX(CcuInstr* instr, uint16_t xnId, uint16_t sqeArgsId, uint16_t setCKEId, uint16_t setCKEMask)
        {
            instr->header = InstrHeader(LOAD_TYPE, LOADSQEARGSTOX_CODE);
            instr->v2.loadSqeArgsToX.xnId = xnId;
            instr->v2.loadSqeArgsToX.sqeArgsId = sqeArgsId;
            instr->v2.loadSqeArgsToX.setCKEId = setCKEId;
            instr->v2.loadSqeArgsToX.setCKEMask = setCKEMask;
        }

        // *XnId = immediate
        void LoadImdToXn(CcuInstr* instr, uint16_t xnId, uint64_t immediate, uint16_t setCKEId, uint16_t setCKEMask)
        {
            instr->header = InstrHeader(LOAD_TYPE, LOADIMDTOX_CODE);
            instr->v2.loadImdToX.xnId = xnId;
            instr->v2.loadImdToX.immediate = immediate;
            instr->v2.loadImdToX.setCKEId = setCKEId;
            instr->v2.loadImdToX.setCKEMask = setCKEMask;
        }

        void ClearX(
            CcuInstr* instr, uint16_t xnId, uint16_t xmId, uint16_t xnIdMode, uint16_t xmIdMode, uint16_t setCKEId,
            uint16_t setCKEMask)
        {
            instr->header = InstrHeader(LOAD_TYPE, CLEARX_CODE);
            instr->v2.clearX.xnId = xnId & 0x7FFF;
            instr->v2.clearX.xnIdMode = xnIdMode & 0x1;
            instr->v2.clearX.xmId = xmId & 0x7FFF;
            instr->v2.clearX.xmIdMode = xmIdMode & 0x1;
            instr->v2.clearX.setCKEId = setCKEId;
            instr->v2.clearX.setCKEMask = setCKEMask;
        }

        void Nop(CcuInstr* instr) { instr->header = InstrHeader(LOAD_TYPE, NOP_CODE); }

        inline void
        Operator(CcuInstr* instr, uint16_t xdId, uint16_t xnId, uint16_t xmId, uint16_t setCKEId, uint16_t setCKEMask)
        {
            instr->v2.operate.xdId = xdId;
            instr->v2.operate.xnId = xnId;
            instr->v2.operate.xmId = xmId;
            instr->v2.operate.setCKEId = setCKEId;
            instr->v2.operate.setCKEMask = setCKEMask;
        }

        void Assign(CcuInstr* instr, uint16_t result, uint16_t operand, uint16_t setCKEId, uint16_t setCKEMask)
        {
            AddI(instr, result, operand, 0, setCKEId, setCKEMask);
        }

        void AssignI(CcuInstr* instr, uint16_t result, uint64_t operand, uint16_t setCKEId, uint16_t setCKEMask)
        {
            LoadImdToXn(instr, result, operand, setCKEId, setCKEMask);
        }

        void
        Add(CcuInstr* instr, uint16_t result, uint16_t operand1, uint16_t operand2, uint16_t setCKEId,
            uint16_t setCKEMask)
        {
            instr->header = InstrHeader(LOAD_TYPE, ADD_CODE);
            instr->v2.operate.parMode = 1;
            Operator(instr, result, operand1, operand2, setCKEId, setCKEMask);
        }

        void
        AddI(CcuInstr* instr, uint16_t result, uint16_t operand, uint16_t imm, uint16_t setCKEId, uint16_t setCKEMask)
        {
            instr->header = InstrHeader(LOAD_TYPE, ADD_CODE);
            instr->v2.operate.parMode = 0;
            Operator(instr, result, operand, imm, setCKEId, setCKEMask);
        }

        void
        Sub(CcuInstr* instr, uint16_t result, uint16_t operand1, uint16_t operand2, uint16_t setCKEId,
            uint16_t setCKEMask)
        {
            instr->header = InstrHeader(LOAD_TYPE, SUB_CODE);

            // 设置指令模式，Par_Mode = 1’b1时，*Xd = *Xn – *Xm
            instr->v2.operate.parMode = 1;
            Operator(instr, result, operand1, operand2, setCKEId, setCKEMask);
        }

        void
        SubI(CcuInstr* instr, uint16_t result, uint16_t operand, uint16_t imm, uint16_t setCKEId, uint16_t setCKEMask)
        {
            instr->header = InstrHeader(LOAD_TYPE, SUB_CODE);

            // 设置指令模式，Par_Mode = 1’b0时，*Xd = *Xn - {48’b0，Immediate[15:0]}
            instr->v2.operate.parMode = 0;
            Operator(instr, result, operand, imm, setCKEId, setCKEMask);
        }

        void
        Mul(CcuInstr* instr, uint16_t result, uint16_t operand1, uint16_t operand2, uint16_t setCKEId,
            uint16_t setCKEMask)
        {
            instr->header = InstrHeader(LOAD_TYPE, MUL_CODE);
            instr->v2.operate.parMode = 1;
            Operator(instr, result, operand1, operand2, setCKEId, setCKEMask);
        }

        void
        MulI(CcuInstr* instr, uint16_t result, uint16_t operand, uint16_t imm, uint16_t setCKEId, uint16_t setCKEMask)
        {
            instr->header = InstrHeader(LOAD_TYPE, MUL_CODE);
            instr->v2.operate.parMode = 0;
            Operator(instr, result, operand, imm, setCKEId, setCKEMask);
        }

        void
        And(CcuInstr* instr, uint16_t result, uint16_t operand1, uint16_t operand2, uint16_t setCKEId,
            uint16_t setCKEMask)
        {
            instr->header = InstrHeader(LOAD_TYPE, AND_CODE);
            instr->v2.operate.parMode = 1;
            Operator(instr, result, operand1, operand2, setCKEId, setCKEMask);
        }

        void
        Or(CcuInstr* instr, uint16_t result, uint16_t operand1, uint16_t operand2, uint16_t setCKEId,
           uint16_t setCKEMask)
        {
            instr->header = InstrHeader(LOAD_TYPE, OR_CODE);
            instr->v2.operate.parMode = 1;
            Operator(instr, result, operand1, operand2, setCKEId, setCKEMask);
        }

        void
        Xor(CcuInstr* instr, uint16_t result, uint16_t operand1, uint16_t operand2, uint16_t setCKEId,
            uint16_t setCKEMask)
        {
            instr->header = InstrHeader(LOAD_TYPE, XOR_CODE);
            instr->v2.operate.parMode = 1;
            Operator(instr, result, operand1, operand2, setCKEId, setCKEMask);
        }

        void Not(CcuInstr* instr, uint16_t result, uint16_t operand, uint16_t setCKEId, uint16_t setCKEMask)
        {
            instr->header = InstrHeader(LOAD_TYPE, NOT_CODE);
            instr->v2.operate.parMode = 0;
            Operator(instr, result, operand, 0, setCKEId, setCKEMask);
        }

        void
        SLL(CcuInstr* instr, uint16_t result, uint16_t operand1, uint16_t operand2, uint16_t setCKEId,
            uint16_t setCKEMask)
        {
            instr->header = InstrHeader(LOAD_TYPE, SHL_CODE);
            instr->v2.operate.shiftType = 0;
            Operator(instr, result, operand1, operand2, setCKEId, setCKEMask);
        }

        void
        SRL(CcuInstr* instr, uint16_t result, uint16_t operand1, uint16_t operand2, uint16_t setCKEId,
            uint16_t setCKEMask)
        {
            instr->header = InstrHeader(LOAD_TYPE, SHR_CODE);
            instr->v2.operate.shiftType = 0;
            Operator(instr, result, operand1, operand2, setCKEId, setCKEMask);
        }

        void
        SLA(CcuInstr* instr, uint16_t result, uint16_t operand1, uint16_t operand2, uint16_t setCKEId,
            uint16_t setCKEMask)
        {
            instr->header = InstrHeader(LOAD_TYPE, SHL_CODE);
            instr->v2.operate.shiftType = 1;
            Operator(instr, result, operand1, operand2, setCKEId, setCKEMask);
        }

        void
        SRA(CcuInstr* instr, uint16_t result, uint16_t operand1, uint16_t operand2, uint16_t setCKEId,
            uint16_t setCKEMask)
        {
            instr->header = InstrHeader(LOAD_TYPE, SHR_CODE);
            instr->v2.operate.shiftType = 1;
            Operator(instr, result, operand1, operand2, setCKEId, setCKEMask);
        }

        // LoadChannel、LoadInstruction暂不实现

        void LoadFromMem(
            CcuInstr* instr, uint16_t dst, uint16_t src, uint16_t srcToken, uint16_t len,
            const CacheConfig& cacheConfig, uint16_t setCKEId, uint16_t setCKEMask)
        {
            instr->header = InstrHeader(LOAD_TYPE, LOAD_CODE);
            instr->v2.load.xdId = dst;
            instr->v2.load.xsId = src;
            instr->v2.load.xstId = srcToken;
            instr->v2.load.xlId = len;

            instr->v2.load.allocHint = cacheConfig.allocHint & 0x3;
            instr->v2.load.victimHint = cacheConfig.victimHint & 0x3;

            instr->v2.load.setCKEId = setCKEId;
            instr->v2.load.setCKEMask = setCKEMask;
        }

        void LoadXFromMem(
            CcuInstr* instr, uint16_t dst, uint16_t src, uint16_t srcToken, uint16_t len,
            const CacheConfig& cacheConfig, uint16_t setCKEId, uint16_t setCKEMask)
        {
            if (instr == nullptr) {
                HCCL_ERROR("[CcuV2::LoadXFromMem] instr is nullptr!");
                return;
            }
            LoadFromMem(instr, dst, src, srcToken, len, cacheConfig, setCKEId, setCKEMask);
            instr->v2.load.dstType = 0x0;
        }

        // HSCB Store暂不实现

        void StoreXToMem(
            CcuInstr* instr, uint16_t dst, uint16_t dstToken, uint16_t src, uint16_t len,
            const CacheConfig& cacheConfig, uint16_t setCKEId, uint16_t setCKEMask)
        {
            instr->header = InstrHeader(LOAD_TYPE, STORE_CODE);
            instr->v2.store.xdId = dst;
            instr->v2.store.xdtId = dstToken;
            instr->v2.store.xsId = src;
            instr->v2.store.xlId = len;
            instr->v2.store.xhId = 0x0;

            instr->v2.store.srcType = 0x0;
            instr->v2.store.storeType = 0b0;
            instr->v2.store.hscbBroadCastDstType = 0x0;

            instr->v2.store.allocHint = cacheConfig.allocHint & 0x3;
            instr->v2.store.victimHint = cacheConfig.victimHint & 0x3;

            instr->v2.store.setCKEId = setCKEId;
            instr->v2.store.setCKEMask = setCKEMask;
        }

        // HSCB Broadcast 暂不实现
        void HSCBStoreXToMem(
            CcuInstr* instr, uint16_t dst, uint16_t src, uint16_t len, const CacheConfig& cacheConfig,
            uint16_t setCKEId, uint16_t setCKEMask)
        {
            instr->header = InstrHeader(LOAD_TYPE, STORE_CODE);
            instr->v2.store.xdId = dst;
            instr->v2.store.xdtId = 0x0;
            instr->v2.store.xsId = src;
            instr->v2.store.xlId = len;
            instr->v2.store.xhId = 0x0;

            instr->v2.store.srcType = 0x0;
            instr->v2.store.storeType = 0b1;
            instr->v2.store.hscbType = 0b0;
            instr->v2.store.hscbBroadCastDstType = 0x0;

            instr->v2.store.allocHint = cacheConfig.allocHint & 0x3;
            instr->v2.store.victimHint = cacheConfig.victimHint & 0x3;

            instr->v2.store.setCKEId = setCKEId;
            instr->v2.store.setCKEMask = setCKEMask;
        }

        void LoadX(
            CcuInstr* instr, uint16_t dst, uint16_t src, uint16_t srcOffset, uint16_t dstOffset, uint16_t oMode,
            uint16_t setCKEId, uint16_t setCKEMask)
        {
            instr->header = InstrHeader(LOAD_TYPE, LOADX_CODE);
            instr->v2.loadStoreX.xdId = dst;
            instr->v2.loadStoreX.xsId = srcOffset;
            instr->v2.loadStoreX.xso = src;
            instr->v2.loadStoreX.xdo = dstOffset;
            instr->v2.loadStoreX.oMode = oMode & 0x1;
            instr->v2.loadStoreX.setCKEId = setCKEId;
            instr->v2.loadStoreX.setCKEMask = setCKEMask;
        }

        void StoreX(
            CcuInstr* instr, uint16_t dst, uint16_t src, uint16_t srcOffset, uint16_t dstOffset, uint16_t oMode,
            uint16_t setCKEId, uint16_t setCKEMask)
        {
            instr->header = InstrHeader(LOAD_TYPE, STOREX_CODE);
            instr->v2.loadStoreX.xdId = dstOffset;
            instr->v2.loadStoreX.xsId = src;
            instr->v2.loadStoreX.xso = srcOffset;
            instr->v2.loadStoreX.xdo = dst;
            instr->v2.loadStoreX.oMode = oMode & 0x1;
            instr->v2.loadStoreX.setCKEId = setCKEId;
            instr->v2.loadStoreX.setCKEMask = setCKEMask;
        }

        // startInstrId ~ endInstrId之间的指令构成loop
        // Xm寄存器中的内容：LoopCtxId[52:45], Offset[44:13], IterNum[12:0]
        // IterNum[12:0] loop执行IterNum次 loop每次执行, 地址偏移为Offset loop在第LoopCtxId个LoopEngine上执行
        void Loop(
            CcuInstr* instr, uint16_t startInstrId, uint16_t endInstrId, uint16_t iterNum, uint16_t offset,
            uint16_t contextId)
        {
            instr->header = InstrHeader(CTRL_TYPE, LOOP_CODE);
            instr->v2.loop.startInstrId = startInstrId;
            instr->v2.loop.endInstrId = endInstrId;
            instr->v2.loop.xmId = iterNum;
            instr->v2.loop.xnId = offset;
            instr->v2.loop.xpId = contextId;
            instr->v2.loop.mode = 0;
            instr->v2.loop.wishCKEBit = 0;
        }

        // startLoopInstrId为LoopGroup所包含的Loop的起始地址
        // Xn寄存器中的内容：ExtendNum[22:16], RepeatLoopIndex[15:9], LoopNum[8:0]
        // Xm寄存器中的内容：gsaOffset[52:21], MSOffset[20:10], ckeOffset[9:0]
        // xnOffset[52:21], xnOffset[31:0]
        // 从startLoopInstrId开始，共LoopNum个Loop，并且从RepeatLoopIndex个开始，展开ExtendNum次
        // 每个展开的Loop，使用的MSId偏移为msOffset，使用的CKEId偏移为ckeOffset，使用的地址偏移为gsaOffset,
        // 使用的XnId偏移为xnOffset
        void LoopGroup(
            CcuInstr* instr, uint16_t startLoopInstrId, uint16_t loopGroupConfig, uint16_t resOffset, uint16_t xnOffset)
        {
            instr->header = InstrHeader(CTRL_TYPE, LOOPGROUP_CODE);
            instr->v2.loopGroup.startLoopInstrId = startLoopInstrId;
            instr->v2.loopGroup.xnId = loopGroupConfig;
            instr->v2.loopGroup.xmId = resOffset;
            instr->v2.loopGroup.xpId = xnOffset;
        }

        // 后续函数中, 均需要wait到<waitCKEId, waitCKEMask>后, 再执行相关操作, 执行完之后再set<setCKEId, setCKEMask>
        // clearType = 1时, wait到之后需要对<waitCKEId, waitCKEMask>清零, 否则不清零

        void SetCKE(
            CcuInstr* instr, uint16_t setCKEId, uint16_t setCKEMask, uint16_t waitCKEId, uint16_t waitCKEMask,
            uint16_t clearType)
        {
            instr->header = InstrHeader(CTRL_TYPE, SETCKBIT_CODE);
            instr->v2.setCKE.clearType = clearType & 0x1;
            instr->v2.setCKE.setCKEId = setCKEId;
            instr->v2.setCKE.setCKEMask = setCKEMask;
            instr->v2.setCKE.waitCKEId = waitCKEId;
            instr->v2.setCKE.waitCKEMask = waitCKEMask;
            instr->v2.setCKE.userData = 0;
        }

        void ClearCKE(
            CcuInstr* instr, uint16_t clearCKEId, uint16_t clearMask, uint16_t waitCKEId, uint16_t waitCKEMask,
            uint16_t clearType)
        {
            instr->header = InstrHeader(CTRL_TYPE, CLEARCKBIT_CODE);
            instr->v2.clearCKE.clearType = clearType & 0x1;
            instr->v2.clearCKE.clearCKEId = clearCKEId;
            instr->v2.clearCKE.clearMask = clearMask;
            instr->v2.clearCKE.waitCKEId = waitCKEId;
            instr->v2.clearCKE.waitCKEMask = waitCKEMask;
        }

        void Jump(
            CcuInstr* instr, uint16_t relTarInstrXnId, uint16_t conditionXnId, uint16_t expectedXnId,
            uint16_t conditionType)
        {
            instr->header = InstrHeader(CTRL_TYPE, JMP_CODE);
            instr->v2.jmp.expectedXnId = expectedXnId;
            instr->v2.jmp.conditionXnId = conditionXnId;
            instr->v2.jmp.relTarInstrXnId = relTarInstrXnId;
            instr->v2.jmp.conditionType = conditionType & 0xF;
        }

        void Wait(CcuInstr* instr, uint16_t conditionXnId, uint16_t expectedXnId, uint16_t conditionType)
        {
            instr->header = InstrHeader(CTRL_TYPE, WAIT_CODE);
            instr->v2.wait.expectedXnId = expectedXnId;
            instr->v2.wait.conditionXnId = conditionXnId;
            instr->v2.wait.conditionType = conditionType & 0xF;
        }

        void Fence(CcuInstr* instr) { instr->header = InstrHeader(CTRL_TYPE, FENCE_CODE); }

        // 本端Memory传输到本端MS
        void TransLocMemToLocMS(
            CcuInstr* instr, uint16_t ms, uint16_t src, uint16_t srcToken, uint16_t len, uint16_t offset,
            uint16_t setCKEId, uint16_t setCKEMask, const CacheConfig& cacheConfig)
        {
            instr->header = InstrHeader(TRANS_TYPE, TRANSLOCMEMTOLOCMS_CODE);
            instr->v2.transLocMemToLocMS.msId = ms;
            instr->v2.transLocMemToLocMS.xsId = src;
            instr->v2.transLocMemToLocMS.xstId = srcToken;
            instr->v2.transLocMemToLocMS.xlId = len;
            instr->v2.transLocMemToLocMS.xoId = offset;
            instr->v2.transLocMemToLocMS.allocHint = cacheConfig.allocHint & 0x3;
            instr->v2.transLocMemToLocMS.victimHint = cacheConfig.victimHint & 0x3;
            instr->v2.transLocMemToLocMS.setCKEId = setCKEId;
            instr->v2.transLocMemToLocMS.setCKEMask = setCKEMask;
        }

        // 本端MS传输到本端Memory
        void TransLocMSToLocMem(
            CcuInstr* instr, uint16_t dst, uint16_t dstToken, uint16_t ms, uint16_t len, uint16_t offset,
            uint16_t setCKEId, uint16_t setCKEMask, const CacheConfig& cacheConfig)
        {
            instr->header = InstrHeader(TRANS_TYPE, TRANSLOCMSTOLOCMEM_CODE);
            instr->v2.transLocMSToLocMem.xdId = dst;
            instr->v2.transLocMSToLocMem.xdtId = dstToken;
            instr->v2.transLocMSToLocMem.msId = ms;
            instr->v2.transLocMSToLocMem.xlId = len;
            instr->v2.transLocMSToLocMem.xoId = offset;
            instr->v2.transLocMSToLocMem.allocHint = cacheConfig.allocHint & 0x3;
            instr->v2.transLocMSToLocMem.victimHint = cacheConfig.victimHint & 0x3;
            instr->v2.transLocMSToLocMem.setCKEId = setCKEId;
            instr->v2.transLocMSToLocMem.setCKEMask = setCKEMask;
        }

        void TransLocMemToLocMem(
            CcuInstr* instr, uint16_t dst, uint16_t dstToken, uint16_t src, uint16_t srcToken, uint16_t len,
            uint16_t usedMSId, uint16_t usedMSNum, uint16_t setCKEId, uint16_t setCKEMask,
            const CacheConfig& srcCacheConfig, const CacheConfig& dstcacheConfig)
        {
            instr->header = InstrHeader(TRANS_TYPE, TRANSLOCMEMTOLOCMEM_CODE);
            instr->v2.transLocMemToLocMem.xdId = dst;
            instr->v2.transLocMemToLocMem.xdtId = dstToken;
            instr->v2.transLocMemToLocMem.xsId = src;
            instr->v2.transLocMemToLocMem.xstId = srcToken;
            instr->v2.transLocMemToLocMem.xlId = len;
            instr->v2.transLocMemToLocMem.usedMSId = usedMSId;
            instr->v2.transLocMemToLocMem.msNum = usedMSNum;

            instr->v2.transLocMemToLocMem.srcAllocHint = srcCacheConfig.allocHint & 0x3;
            instr->v2.transLocMemToLocMem.srcVictimHint = srcCacheConfig.victimHint & 0x3;
            instr->v2.transLocMemToLocMem.dstAllocHint = dstcacheConfig.allocHint & 0x3;
            instr->v2.transLocMemToLocMem.dstVictimHint = dstcacheConfig.victimHint & 0x3;

            instr->v2.transLocMemToLocMem.setCKEId = setCKEId;
            instr->v2.transLocMemToLocMem.setCKEMask = setCKEMask;
        }

        void TransMem(
            CcuInstr* instr, uint16_t dst, uint16_t dstToken, uint16_t src, uint16_t srcToken, uint16_t len,
            uint16_t channel, const TransMemNotifyInfo& notify, const TransMemReduceInfo& reduce,
            const TransMemConfig& config, uint16_t setCKEId, uint16_t setCKEMask)
        {
            instr->header = InstrHeader(TRANS_TYPE, TRANSMEM_CODE);
            instr->v2.transMem.xdId = dst;
            instr->v2.transMem.xdtId = dstToken;
            instr->v2.transMem.xsId = src;
            instr->v2.transMem.xstId = srcToken;
            instr->v2.transMem.xlId = len;
            instr->v2.transMem.xcId = channel;
            instr->v2.transMem.xnId = notify.xnId;
            instr->v2.transMem.xntId = notify.xntId;
            instr->v2.transMem.value = notify.value;
            instr->v2.transMem.udfType = reduce.udfType & 0xFF;
            instr->v2.transMem.reduceDataType = reduce.reduceDataType & 0xF;
            instr->v2.transMem.reduceOpCode = reduce.reduceOpCode & 0xF;
            instr->v2.transMem.dmaOpCode = config.dmaOpCode & 0xFF;

            instr->v2.transMem.order = config.order & 0x7;
            instr->v2.transMem.fence = config.fence & 0x1;
            instr->v2.transMem.cqe = config.cqe & 0x1;
            instr->v2.transMem.nf = config.nf & 0x1;
            instr->v2.transMem.udfEnable = config.udfEnable & 0x1;
            instr->v2.transMem.splitMode = config.splitMode & 0x1;
            instr->v2.transMem.se = config.se & 0x1;
            instr->v2.transMem.rmtJettyType = config.rmtJettyType & 0x3;

            instr->v2.transMem.src_mode = config.src_mode & 0x1;
            instr->v2.transMem.dst_mode = config.dst_mode & 0x1;
            instr->v2.transMem.msIdMode = config.msIdMode & 0x1;

            instr->v2.transMem.targetHint = 0x0;

            instr->v2.transMem.setCKEId = setCKEId;
            instr->v2.transMem.setCKEMask = setCKEMask;
        }

        // 将本端Xn的值写入远端8B地址
        inline void SyncWtX(
            CcuInstr* instr, uint16_t dst, uint16_t dstToken, uint16_t xn, uint16_t channelId, uint16_t setCKEId,
            uint16_t setCKEMask)
        {
            instr->header = InstrHeader(TRANS_TYPE, SYNCWTX_CODE);
            instr->v2.syncWtX.xdId = dst;
            instr->v2.syncWtX.xdtId = dstToken;
            instr->v2.syncWtX.xsId = xn;
            instr->v2.syncWtX.xcId = channelId;

            instr->v2.syncWtX.setCKEId = setCKEId;
            instr->v2.syncWtX.setCKEMask = setCKEMask;
        }

        // 将本端Xn的值写入远端8B地址并置位远端CKE
        void SyncWtX(
            CcuInstr* instr, uint16_t dst, uint16_t dstToken, uint16_t xn, uint16_t channelId,
            const TransMemNotifyInfo& notify, uint16_t setCKEId, uint16_t setCKEMask)
        {
            SyncWtX(instr, dst, dstToken, xn, channelId, setCKEId, setCKEMask);
            instr->v2.syncWtX.xnId = notify.xnId;
            instr->v2.syncWtX.xntId = notify.xntId;
            instr->v2.syncWtX.value = notify.value;

            instr->v2.syncWtX.notifyValid = 1;
            instr->v2.syncWtX.parMode = 1;
        }

        // 置位远端CKE
        void SyncWtX(
            CcuInstr* instr, const TransMemNotifyInfo& notify, uint16_t channelId, uint16_t setCKEId,
            uint16_t setCKEMask)
        {
            SyncWtX(instr, notify.xnId, notify.xntId, notify.value, channelId, setCKEId, setCKEMask);

            instr->v2.syncWtX.notifyValid = 0;
            instr->v2.syncWtX.parMode = 0;
        }

        // 将本端Xn的值以atomic store add的方式写入远端8B地址
        void SyncAtX(
            CcuInstr* instr, uint16_t dstAddr, uint16_t dstToken, uint16_t srcId, uint16_t channelId, uint16_t setCKEId,
            uint16_t setCKEMask)
        {
            instr->header = InstrHeader(TRANS_TYPE, SYNCATX_CODE);
            instr->v2.syncAtX.xdId = dstAddr;
            instr->v2.syncAtX.xdtId = dstToken;
            instr->v2.syncAtX.xsId = srcId;
            instr->v2.syncAtX.xcId = channelId;

            instr->v2.syncAtX.parMode = 1;

            instr->v2.syncAtX.setCKEId = setCKEId;
            instr->v2.syncAtX.setCKEMask = setCKEMask;
        }

        // MSA~MSH Reduce到 MSA
        inline void Reduce(
            CcuInstr* instr, const uint16_t* ms, uint16_t count, uint16_t castEn, uint16_t dataType, uint16_t setCKEId,
            uint16_t setCKEMask, uint16_t xnIdLength)
        {
            // 由调用者保证传入的count >= 2(reduce的数据源)
            if (count < CCU_REDUCE_MIN_MS) {
                HCCL_ERROR("[CcuV2::Reduce] count %u < min %u", count, CCU_REDUCE_MIN_MS);
                return;
            }
            count -= 2; // CCU指令中指定count数为实际参与运算的MS数减2

            for (uint16_t index = 0; index < CCU_REDUCE_MAX_MS; index++) {
                instr->v2.reduce.msId[index] = ms[index];
            }
            instr->v2.reduce.XnIdLength = xnIdLength;
            instr->v2.reduce.count = count & 0x7;
            instr->v2.reduce.castEn = castEn & 0x3;
            instr->v2.reduce.dataType = dataType & 0x1f;
            instr->v2.reduce.setCKEId = setCKEId;
            instr->v2.reduce.setCKEMask = setCKEMask;
        }

        void ReduceAdd(
            CcuInstr* instr, const uint16_t* ms, uint16_t count, uint16_t castEn, uint16_t dataType, uint16_t setCKEId,
            uint16_t setCKEMask, uint16_t XnIdLength)
        {
            if (instr == nullptr) {
                HCCL_ERROR("[CcuV2::ReduceAdd] instr is nullptr!");
                return;
            }
            instr->header = InstrHeader(REDUCE_TYPE, REDUCE_ADD_CODE);
            Reduce(instr, ms, count, castEn, dataType, setCKEId, setCKEMask, XnIdLength);
        }

        void ReduceMax(
            CcuInstr* instr, const uint16_t* ms, uint16_t count, uint16_t dataType, uint16_t setCKEId,
            uint16_t setCKEMask, uint16_t XnIdLength)
        {
            if (instr == nullptr) {
                HCCL_ERROR("[CcuV2::ReduceMax] instr is nullptr!");
                return;
            }
            instr->header = InstrHeader(REDUCE_TYPE, REDUCE_MAX_CODE);
            Reduce(instr, ms, count, 0, dataType, setCKEId, setCKEMask, XnIdLength);
        }

        void ReduceMin(
            CcuInstr* instr, const uint16_t* ms, uint16_t count, uint16_t dataType, uint16_t setCKEId,
            uint16_t setCKEMask, uint16_t XnIdLength)
        {
            if (instr == nullptr) {
                HCCL_ERROR("[CcuV2::ReduceMin] instr is nullptr!");
                return;
            }
            instr->header = InstrHeader(REDUCE_TYPE, REDUCE_MIN_CODE);
            Reduce(instr, ms, count, 0, dataType, setCKEId, setCKEMask, XnIdLength);
        }

        std::string ParseLoadSqeArgsToX(const CcuInstr* instr)
        {
            uint16_t xnId = instr->v2.loadSqeArgsToX.xnId;
            uint16_t sqeArgsId = instr->v2.loadSqeArgsToX.sqeArgsId;
            uint16_t setCKEId = instr->v2.loadSqeArgsToX.setCKEId;
            uint16_t setCKEMask = instr->v2.loadSqeArgsToX.setCKEMask;

            // 使用 StringFormat 格式化字符串
            return Hccl::StringFormat(
                "Load SqeArg[%u] to Xn[%u], Set CKE[%u:%04x]", sqeArgsId, xnId, setCKEId, setCKEMask);
        }

        std::string ParseLoadImdToXn(const CcuInstr* instr)
        {
            uint16_t xnId = instr->v2.loadImdToX.xnId;
            uint64_t immediate = instr->v2.loadImdToX.immediate;
            uint16_t setCKEId = instr->v2.loadImdToX.setCKEId;
            uint16_t setCKEMask = instr->v2.loadImdToX.setCKEMask;

            // 使用 StringFormat 格式化字符串
            return Hccl::StringFormat(
                "Load immediate[%llu] to Xn[%u], Set CKE[%u:%04x]", immediate, xnId, setCKEId, setCKEMask);
        }

        std::string ParseClearX(const CcuInstr* instr)
        {
            uint16_t xnId = instr->v2.clearX.xnId;
            uint16_t xmId = instr->v2.clearX.xmId;
            uint16_t setCKEId = instr->v2.clearX.setCKEId;
            uint16_t setCKEMask = instr->v2.clearX.setCKEMask;

            // 使用 StringFormat 格式化字符串
            return Hccl::StringFormat(
                "ClearX: xnId[%u] to xmId[%u], Set CKE[%u:%04x]", xnId, xmId, setCKEId, setCKEMask);
        }

        std::string ParseNop(const CcuInstr* instr)
        {
            (void)instr;
            return Hccl::StringFormat("Nop");
        }

        std::string ParseAdd(const CcuInstr* instr)
        {
            uint16_t xdId = instr->v2.operate.xdId;
            uint16_t xnId = instr->v2.operate.xnId;
            uint16_t xmId = instr->v2.operate.xmId;
            uint16_t setCKEId = instr->v2.operate.setCKEId;
            uint16_t setCKEMask = instr->v2.operate.setCKEMask;
            uint16_t parMode = instr->v2.operate.parMode;

            // 使用 StringFormat 格式化字符串
            if (instr->v2.operate.parMode == 0) {
                return Hccl::StringFormat(
                    "AddI:xdId[%u] = xnId[%u] + immediate[%u], Set CKE[%u:%04x], parMode[%u]", xdId, xnId, xmId,
                    setCKEId, setCKEMask, parMode);
            } else {
                return Hccl::StringFormat(
                    "Add:xdId[%u] = xnId[%u] + xmId[%u], Set CKE[%u:%04x], parMode[%u]", xdId, xnId, xmId, setCKEId,
                    setCKEMask, parMode);
            }
        }

        std::string ParseSub(const CcuInstr* instr)
        {
            uint16_t xdId = instr->v2.operate.xdId;
            uint16_t xnId = instr->v2.operate.xnId;
            uint16_t xmId = instr->v2.operate.xmId;
            uint16_t setCKEId = instr->v2.operate.setCKEId;
            uint16_t setCKEMask = instr->v2.operate.setCKEMask;
            uint16_t parMode = instr->v2.operate.parMode;

            // 使用 StringFormat 格式化字符串
            if (instr->v2.operate.parMode == 0) {
                return Hccl::StringFormat(
                    "SubI:xdId[%u] = xnId[%u] - immediate[%u], Set CKE[%u:%04x], parMode[%u]", xdId, xnId, xmId,
                    setCKEId, setCKEMask, parMode);
            } else {
                return Hccl::StringFormat(
                    "Sub:xdId[%u] = xnId[%u] - xmId[%u], Set CKE[%u:%04x], parMode[%u]", xdId, xnId, xmId, setCKEId,
                    setCKEMask, parMode);
            }
        }

        std::string ParseMul(const CcuInstr* instr)
        {
            uint16_t xdId = instr->v2.operate.xdId;
            uint16_t xnId = instr->v2.operate.xnId;
            uint16_t xmId = instr->v2.operate.xmId;
            uint16_t setCKEId = instr->v2.operate.setCKEId;
            uint16_t setCKEMask = instr->v2.operate.setCKEMask;
            uint16_t parMode = instr->v2.operate.parMode;

            // 使用 StringFormat 格式化字符串
            if (instr->v2.operate.parMode == 0) {
                return Hccl::StringFormat(
                    "MulI:xdId[%u] = xnId[%u] * immediate[%u], Set CKE[%u:%04x], parMode[%u]", xdId, xnId, xmId,
                    setCKEId, setCKEMask, parMode);
            } else {
                return Hccl::StringFormat(
                    "Mul:xdId[%u] = xnId[%u] * xmId[%u], Set CKE[%u:%04x], parMode[%u]", xdId, xnId, xmId, setCKEId,
                    setCKEMask, parMode);
            }
        }

        std::string ParseAnd(const CcuInstr* instr)
        {
            uint16_t xdId = instr->v2.operate.xdId;
            uint16_t xnId = instr->v2.operate.xnId;
            uint16_t xmId = instr->v2.operate.xmId;
            uint16_t setCKEId = instr->v2.operate.setCKEId;
            uint16_t setCKEMask = instr->v2.operate.setCKEMask;

            // 使用 StringFormat 格式化字符串
            return Hccl::StringFormat(
                "And:xdId[%u] = xnId[%u] & xmId[%u], Set CKE[%u:%04x]", xdId, xnId, xmId, setCKEId, setCKEMask);
        }

        std::string ParseOr(const CcuInstr* instr)
        {
            uint16_t xdId = instr->v2.operate.xdId;
            uint16_t xnId = instr->v2.operate.xnId;
            uint16_t xmId = instr->v2.operate.xmId;
            uint16_t setCKEId = instr->v2.operate.setCKEId;
            uint16_t setCKEMask = instr->v2.operate.setCKEMask;

            // 使用 StringFormat 格式化字符串
            return Hccl::StringFormat(
                "Or:xdId[%u] = xnId[%u] | xmId[%u], Set CKE[%u:%04x]", xdId, xnId, xmId, setCKEId, setCKEMask);
        }

        std::string ParseXor(const CcuInstr* instr)
        {
            uint16_t xdId = instr->v2.operate.xdId;
            uint16_t xnId = instr->v2.operate.xnId;
            uint16_t xmId = instr->v2.operate.xmId;
            uint16_t setCKEId = instr->v2.operate.setCKEId;
            uint16_t setCKEMask = instr->v2.operate.setCKEMask;

            // 使用 StringFormat 格式化字符串
            return Hccl::StringFormat(
                "Xor:xdId[%u] = xnId[%u] ^ xmId[%u], Set CKE[%u:%04x]", xdId, xnId, xmId, setCKEId, setCKEMask);
        }

        std::string ParseNot(const CcuInstr* instr)
        {
            uint16_t xdId = instr->v2.operate.xdId;
            uint16_t xnId = instr->v2.operate.xnId;
            uint16_t setCKEId = instr->v2.operate.setCKEId;
            uint16_t setCKEMask = instr->v2.operate.setCKEMask;

            // 使用 StringFormat 格式化字符串
            return Hccl::StringFormat("Not:xdId[%u] = ~xnId[%u], Set CKE[%u:%04x]", xdId, xnId, setCKEId, setCKEMask);
        }

        std::string ParseShl(const CcuInstr* instr)
        {
            uint16_t xdId = instr->v2.operate.xdId;
            uint16_t xnId = instr->v2.operate.xnId;
            uint16_t xmId = instr->v2.operate.xmId;
            uint16_t shiftType = instr->v2.operate.shiftType;
            uint16_t setCKEId = instr->v2.operate.setCKEId;
            uint16_t setCKEMask = instr->v2.operate.setCKEMask;

            // 使用 StringFormat 格式化字符串
            if (shiftType == 0) {
                return Hccl::StringFormat(
                    "logic left shift:xdId[%u] = xnId[%u] << xmId[%u], Set CKE[%u:%04x]", xdId, xnId, xmId, setCKEId,
                    setCKEMask);
            } else {
                return Hccl::StringFormat(
                    "arithmetic left shift:xdId[%u] = xnId[%u] << xmId[%u], Set CKE[%u:%04x]", xdId, xnId, xmId,
                    setCKEId, setCKEMask);
            }
        }

        std::string ParseShr(const CcuInstr* instr)
        {
            uint16_t xdId = instr->v2.operate.xdId;
            uint16_t xnId = instr->v2.operate.xnId;
            uint16_t xmId = instr->v2.operate.xmId;
            uint16_t shiftType = instr->v2.operate.shiftType;
            uint16_t setCKEId = instr->v2.operate.setCKEId;
            uint16_t setCKEMask = instr->v2.operate.setCKEMask;

            // 使用 StringFormat 格式化字符串
            if (shiftType == 0) {
                return Hccl::StringFormat(
                    "logic right shift:xdId[%u] = xnId[%u] >> xmId[%u], Set CKE[%u:%04x]", xdId, xnId, xmId, setCKEId,
                    setCKEMask);
            } else {
                return Hccl::StringFormat(
                    "arithmetic right shift:xdId[%u] = xnId[%u] >> xmId[%u], Set CKE[%u:%04x]", xdId, xnId, xmId,
                    setCKEId, setCKEMask);
            }
        }

        std::string ParseLoadFromMem(const CcuInstr* instr)
        {
            uint16_t xdId = instr->v2.load.xdId;
            uint16_t xsId = instr->v2.load.xsId;
            uint16_t xstId = instr->v2.load.xstId;
            uint16_t xlId = instr->v2.load.xlId;
            uint16_t allocHint = instr->v2.load.allocHint;
            uint16_t victimHint = instr->v2.load.victimHint;
            uint16_t setCKEId = instr->v2.load.setCKEId;
            uint16_t setCKEMask = instr->v2.load.setCKEMask;
            uint16_t dstType = instr->v2.load.dstType;

            return Hccl::StringFormat(
                "LoadFromMem: xdId[%u] xsId[%u] xstId[%u] xlId[%u], allocHint[%u], victimHint[%u], setCKEId[%u], "
                "setCKEMask[%u] dstType[%u]",
                xdId, xsId, xstId, xlId, allocHint, victimHint, setCKEId, setCKEMask, dstType);
        }

        std::string ParseStoreXToMem(const CcuInstr* instr)
        {
            uint16_t xdId = instr->v2.store.xdId;
            uint16_t xdtId = instr->v2.store.xdtId;
            uint16_t xsId = instr->v2.store.xsId;
            uint16_t xlId = instr->v2.store.xlId;
            uint16_t xhId = instr->v2.store.xhId;
            uint16_t srcType = instr->v2.store.srcType;
            uint16_t allocHint = instr->v2.store.allocHint;
            uint16_t victimHint = instr->v2.store.victimHint;
            uint16_t storeType = instr->v2.store.storeType;
            uint16_t hscbType = instr->v2.store.hscbType;
            uint16_t hscbBroadCastDstType = instr->v2.store.hscbBroadCastDstType;
            uint16_t setCKEId = instr->v2.store.setCKEId;
            uint16_t setCKEMask = instr->v2.store.setCKEMask;

            // 使用 StringFormat 格式化字符串
            return Hccl::StringFormat(
                "StoreXToMem: xdId[%u] xdtId[%u], xsId[%u], xlId[%u], xhId[%u], srcType[%u], allocHint[%u], "
                "victimHint[%u], storeType[%u], hscbType[%u], hscbBroadCastDstType[%u], set CKE[%u:%04x]",
                xdId, xdtId, xsId, xlId, xhId, srcType, allocHint, victimHint, storeType, hscbType,
                hscbBroadCastDstType, setCKEId, setCKEMask);
        }

        std::string ParseLoadX(const CcuInstr* instr)
        {
            uint16_t xdId = instr->v2.loadStoreX.xdId;
            uint16_t xsId = instr->v2.loadStoreX.xsId;
            uint16_t xso = instr->v2.loadStoreX.xso;
            uint16_t xdo = instr->v2.loadStoreX.xdo;
            uint16_t oMode = instr->v2.loadStoreX.oMode;
            uint16_t setCKEId = instr->v2.loadStoreX.setCKEId;
            uint16_t setCKEMask = instr->v2.loadStoreX.setCKEMask;

            return Hccl::StringFormat(
                "LoadX: xdId[%u] xsId[%u] xso[%u] xdo[%u] oMode[%u], set CKE[%u:%04x]", xdId, xsId, xso, xdo, oMode,
                setCKEId, setCKEMask);
        }

        std::string ParseStoreX(const CcuInstr* instr)
        {
            uint16_t xdId = instr->v2.loadStoreX.xdId;
            uint16_t xsId = instr->v2.loadStoreX.xsId;
            uint16_t xso = instr->v2.loadStoreX.xso;
            uint16_t xdo = instr->v2.loadStoreX.xdo;
            uint16_t oMode = instr->v2.loadStoreX.oMode;
            uint16_t setCKEId = instr->v2.loadStoreX.setCKEId;
            uint16_t setCKEMask = instr->v2.loadStoreX.setCKEMask;

            return Hccl::StringFormat(
                "StoreX: xdId[%u] xsId[%u] xso[%u] xdo[%u] oMode[%u], set CKE[%u:%04x]", xdId, xsId, xso, xdo, oMode,
                setCKEId, setCKEMask);
        }

        std::string ParseLoop(const CcuInstr* instr)
        {
            uint16_t startInstrId = instr->v2.loop.startInstrId;
            uint16_t endInstrId = instr->v2.loop.endInstrId;
            uint16_t xmId = instr->v2.loop.xmId;
            uint16_t xnId = instr->v2.loop.xnId;
            uint16_t xpId = instr->v2.loop.xpId;
            uint16_t wishCKEBit = instr->v2.loop.wishCKEBit;
            uint16_t mode = instr->v2.loop.mode;

            // 使用 StringFormat 格式化字符串
            return Hccl::StringFormat(
                "Loop From startInstrId[%u] to endInstrId[%u] with iterNumXn[%u], OffsetXn[%u], ContextIdXn[%u],"
                "wishCKEBit[%u], mode[%u]",
                startInstrId, endInstrId, xmId, xnId, xpId, wishCKEBit, mode);
        }

        std::string ParseLoopGroup(const CcuInstr* instr)
        {
            uint16_t startLoopInstrId = instr->v2.loopGroup.startLoopInstrId;
            uint16_t xnId = instr->v2.loopGroup.xnId;
            uint16_t xmId = instr->v2.loopGroup.xmId;
            uint16_t xpId = instr->v2.loopGroup.xpId;

            // 使用 StringFormat 格式化字符串
            return Hccl::StringFormat(
                "LoopGroup From startLoopInstrId[%u] with loopGroupXn[%u], resOffsetXn[%u], xnOffset[%u]",
                startLoopInstrId, xnId, xmId, xpId);
        }

        std::string ParseJump(const CcuInstr* instr)
        {
            uint16_t relTarInstrXnId = instr->v2.jmp.relTarInstrXnId;
            uint16_t conditionXnId = instr->v2.jmp.conditionXnId;
            uint16_t expectedXnId = instr->v2.jmp.expectedXnId;
            uint16_t conditionType = instr->v2.jmp.conditionType;

            // 使用 StringFormat 格式化字符串
            return Hccl::StringFormat(
                "When conditionXn[%u] condition with[%u] expectedXn[%u], Jump To RelInstrIdXn[%u]", conditionXnId,
                conditionType, expectedXnId, relTarInstrXnId);
        }

        std::string ParseWait(const CcuInstr* instr)
        {
            uint16_t conditionXnId = instr->v2.wait.conditionXnId;
            uint16_t expectedXnId = instr->v2.wait.expectedXnId;
            uint16_t conditionType = instr->v2.wait.conditionType;

            // 使用 StringFormat 格式化字符串
            return Hccl::StringFormat(
                "Wait conditionXn[%u] condition with[%u] expectedXn[%u]", conditionXnId, conditionType, expectedXnId);
        }

        std::string ParseFence(const CcuInstr* instr)
        {
            (void)instr;
            return Hccl::StringFormat("Fence");
        }

        std::string ParseSetCKE(const CcuInstr* instr)
        {
            uint16_t setCKEId = instr->v2.setCKE.setCKEId;
            uint16_t setCKEMask = instr->v2.setCKE.setCKEMask;
            uint16_t waitCKEId = instr->v2.setCKE.waitCKEId;
            uint16_t waitCKEMask = instr->v2.setCKE.waitCKEMask;
            uint16_t clearType = instr->v2.setCKE.clearType;
            uint64_t userData = instr->v2.setCKE.userData;

            // 使用 StringFormat 格式化字符串
            return Hccl::StringFormat(
                "Wait CKE[%u:%04x], Set CKE[%u:%04x], clearType[%u], userData[%llu]", waitCKEId, waitCKEMask, setCKEId,
                setCKEMask, clearType, userData);
        }

        std::string ParseClearCKE(const CcuInstr* instr)
        {
            uint16_t clearCKEId = instr->v2.clearCKE.clearCKEId;
            uint16_t clearMask = instr->v2.clearCKE.clearMask;
            uint16_t waitCKEId = instr->v2.clearCKE.waitCKEId;
            uint16_t waitCKEMask = instr->v2.clearCKE.waitCKEMask;
            uint16_t clearType = instr->v2.clearCKE.clearType & 0x1;

            // 使用 StringFormat 格式化字符串
            return Hccl::StringFormat(
                "Wait CKE[%u:%04x], Clear CKE[%u:%04x], clearType[%u]", waitCKEId, waitCKEMask, clearCKEId, clearMask,
                clearType);
        }

        std::string ParseTransLocMemToLocMS(const CcuInstr* instr)
        {
            uint16_t ms = instr->v2.transLocMemToLocMS.msId;
            uint16_t src = instr->v2.transLocMemToLocMS.xsId;
            uint16_t srcToken = instr->v2.transLocMemToLocMS.xstId;
            uint16_t len = instr->v2.transLocMemToLocMS.xlId;
            uint16_t offset = instr->v2.transLocMemToLocMS.xoId;
            uint16_t allocHint = instr->v2.transLocMemToLocMS.allocHint;
            uint16_t victimHint = instr->v2.transLocMemToLocMS.victimHint;
            uint16_t setCKEId = instr->v2.transLocMemToLocMS.setCKEId;
            uint16_t setCKEMask = instr->v2.transLocMemToLocMS.setCKEMask;

            // 使用 StringFormat 格式化字符串
            return Hccl::StringFormat(
                "Trans LocMem[%u:%u] To LocMS[%u] With LengthXn[%u], Offset[%u], Set CKE[%u:%04x], "
                "allocHint[%u], victimHint[%u]",
                src, srcToken, ms, len, offset, setCKEId, setCKEMask, allocHint, victimHint);
        }

        std::string ParseTransLocMSToLocMem(const CcuInstr* instr)
        {
            uint16_t dst = instr->v2.transLocMSToLocMem.xdId;
            uint16_t dstToken = instr->v2.transLocMSToLocMem.xdtId;
            uint16_t ms = instr->v2.transLocMSToLocMem.msId;
            uint16_t len = instr->v2.transLocMSToLocMem.xlId;
            uint16_t offset = instr->v2.transLocMSToLocMem.xoId;
            uint16_t allocHint = instr->v2.transLocMSToLocMem.allocHint;
            uint16_t victimHint = instr->v2.transLocMSToLocMem.victimHint;
            uint16_t setCKEId = instr->v2.transLocMSToLocMem.setCKEId;
            uint16_t setCKEMask = instr->v2.transLocMSToLocMem.setCKEMask;

            // 使用 StringFormat 格式化字符串
            return Hccl::StringFormat(
                "Trans LocMS[%u] To LocMem[%u:%u] With LengthXn[%u], Offset[%u], Set CKE[%u:%04x], "
                "allocHint[%u], victimHint[%u]",
                ms, dst, dstToken, len, offset, setCKEId, setCKEMask, allocHint, victimHint);
        }

        std::string ParseTransLocMemToLocMem(const CcuInstr* instr)
        {
            uint16_t dst = instr->v2.transLocMemToLocMem.xdId;
            uint16_t dstToken = instr->v2.transLocMemToLocMem.xdtId;
            uint16_t src = instr->v2.transLocMemToLocMem.xsId;
            uint16_t srcToken = instr->v2.transLocMemToLocMem.xstId;
            uint16_t len = instr->v2.transLocMemToLocMem.xlId;
            uint16_t usedMSId = instr->v2.transLocMemToLocMem.usedMSId;
            uint16_t msNum = instr->v2.transLocMemToLocMem.msNum;

            uint16_t srcAllocHint = instr->v2.transLocMemToLocMem.srcAllocHint;
            uint16_t srcVictimHint = instr->v2.transLocMemToLocMem.srcVictimHint;
            uint16_t dstAllocHint = instr->v2.transLocMemToLocMem.dstAllocHint;
            uint16_t dstVictimHint = instr->v2.transLocMemToLocMem.dstVictimHint;

            uint16_t setCKEId = instr->v2.transLocMemToLocMem.setCKEId;
            uint16_t setCKEMask = instr->v2.transLocMemToLocMem.setCKEMask;

            // 使用 StringFormat 格式化字符串
            return Hccl::StringFormat(
                "Trans LocMem[%u:%u] To LocMem[%u:%u] With Length[%u], Set CKE[%u:%04x], usedMSId[%u], msNum[%u],"
                "srcAllocHint[%u], srcVictimHint[%u], dstAllocHint[%u], dstVictimHint[%u]",
                src, srcToken, dst, dstToken, len, setCKEId, setCKEMask, usedMSId, msNum, srcAllocHint, srcVictimHint,
                dstAllocHint, dstVictimHint);
        }

        std::string ParseTransMem(const CcuInstr* instr)
        {
            uint16_t dst = instr->v2.transMem.xdId;
            uint16_t dstToken = instr->v2.transMem.xdtId;
            uint16_t src = instr->v2.transMem.xsId;
            uint16_t srcToken = instr->v2.transMem.xstId;
            uint16_t len = instr->v2.transMem.xlId;
            uint16_t channel = instr->v2.transMem.xcId;

            uint16_t notifyXnId = instr->v2.transMem.xnId;
            uint16_t notifyXntId = instr->v2.transMem.xntId;
            uint16_t notifyValue = instr->v2.transMem.value;

            uint16_t reduceUdfType = instr->v2.transMem.udfType;
            uint16_t reduceDataType = instr->v2.transMem.reduceDataType;
            uint16_t reduceOpCode = instr->v2.transMem.reduceOpCode;

            uint16_t configOpCode = instr->v2.transMem.dmaOpCode;
            uint16_t configOrder = instr->v2.transMem.order;
            uint16_t configFence = instr->v2.transMem.fence;
            uint16_t configCqe = instr->v2.transMem.cqe;
            uint16_t configNf = instr->v2.transMem.nf;
            uint16_t configUdfEnable = instr->v2.transMem.udfEnable;
            uint16_t configSplitMode = instr->v2.transMem.splitMode;
            uint16_t configSe = instr->v2.transMem.se;
            uint16_t configRmtJettyType = instr->v2.transMem.rmtJettyType;
            uint16_t configTargetHint = instr->v2.transMem.targetHint;

            uint16_t setCKEId = instr->v2.transMem.setCKEId;
            uint16_t setCKEMask = instr->v2.transMem.setCKEMask;

            uint16_t srcMode = instr->v2.transMem.src_mode;
            uint16_t dstMode = instr->v2.transMem.dst_mode;
            uint16_t msIdMode = instr->v2.transMem.msIdMode;

            // 使用 StringFormat 格式化字符串
            return Hccl::StringFormat(
                "Trans Mem[%u:%u] To Mem[%u:%u] With Len[%u] Use Channel[%u] Set CKE[%u:%04x], notifyXnId[%u], "
                "notifyXntId[%u], notifyValue[%u], reduceUdfType[%u], reduceDataType[%u], reduceOpCode[%u], "
                "configOpCode[%u], Order[%u], Fence[%u], Cqe[%u], Nf[%u], UdfEnable[%u], SplitMode[%u], Se[%u], "
                "RmtJettyType[%u], configTargetHint[%u], srcMode[%u], dstMode[%u], msIdMode[%u]",
                src, srcToken, dst, dstToken, len, channel, setCKEId, setCKEMask, notifyXnId, notifyXntId, notifyValue,
                reduceUdfType, reduceDataType, reduceOpCode, configOpCode, configOrder, configFence, configCqe,
                configNf, configUdfEnable, configSplitMode, configSe, configRmtJettyType, configTargetHint, srcMode,
                dstMode, msIdMode);
        }

        std::string ParseSyncWtX(const CcuInstr* instr)
        {
            uint16_t dst = instr->v2.syncWtX.xdId;
            uint16_t dstToken = instr->v2.syncWtX.xdtId;
            uint16_t xn = instr->v2.syncWtX.xsId;
            uint16_t channelId = instr->v2.syncWtX.xcId;

            uint16_t setCKEId = instr->v2.syncWtX.setCKEId;
            uint16_t setCKEMask = instr->v2.syncWtX.setCKEMask;

            // 检查是否包含通知信息
            bool hasNotify = (instr->v2.syncWtX.notifyValid != 0);
            uint16_t parMode = instr->v2.syncWtX.parMode;
            uint16_t notifyXnId = hasNotify ? instr->v2.syncWtX.xnId : 0;
            uint16_t notifyXntId = hasNotify ? instr->v2.syncWtX.xntId : 0;
            uint16_t notifyValue = hasNotify ? instr->v2.syncWtX.value : 0;

            // 使用 StringFormat 格式化字符串
            if (hasNotify) {
                return Hccl::StringFormat(
                    "Sync Xn[%u] To Mem[%u:%u] Use ChannelXn[%u], with notify[%u:%u:%04x], and Set CKE[%u:%04x], "
                    "parMode[%u]",
                    xn, dst, dstToken, channelId, notifyXnId, notifyXntId, notifyValue, setCKEId, setCKEMask, parMode);
            } else {
                return Hccl::StringFormat(
                    "Sync Immd[%u] To Mem[%u:%u] Use ChannelXn[%u], and Set CKE[%u:%04x], parMode[%u]", xn, dst,
                    dstToken, channelId, setCKEId, setCKEMask, parMode);
            }
        }

        std::string ParseSyncAtX(const CcuInstr* instr)
        {
            uint16_t xdId = instr->v2.syncAtX.xdId;
            uint16_t xdtId = instr->v2.syncAtX.xdtId;
            uint16_t xsId = instr->v2.syncAtX.xsId;
            uint16_t xcId = instr->v2.syncAtX.xcId;
            uint16_t parMode = instr->v2.syncAtX.parMode;
            uint16_t setCKEId = instr->v2.syncAtX.setCKEId;
            uint16_t setCKEMask = instr->v2.syncAtX.setCKEMask;

            return Hccl::StringFormat(
                "Atomic Sync xsId[%u] To Mem[%u:%u] Use ChannelXn[%u], parMode[%u], setCKEId[%u:%04x]", xsId, xdId,
                xdtId, xcId, parMode, setCKEId, setCKEMask);
        }

        std::string ParseMSListV2(const uint16_t* msId, uint16_t msNum)
        {
            std::string res = "MS[";
            for (uint16_t i = 0; i < msNum + 2 && i < CCU_REDUCE_MAX_MS; i++) { // 循环范围 0 ~ count+ 2，防止越界
                if (i == msNum + 1) {
                    res += std::to_string(msId[i] / 0x8000) + ":" + std::to_string(msId[i] % 0x8000) + "]";
                } else {
                    res += std::to_string(msId[i] / 0x8000) + ":" + std::to_string(msId[i] % 0x8000) + ", ";
                }
            }

            return res;
        }

        std::string ParseReduce(const CcuInstr* instr)
        {
            uint16_t reduceType = instr->header.code;
            uint16_t count = instr->v2.reduce.count;
            uint16_t castEn = instr->v2.reduce.castEn;
            uint16_t dataType = instr->v2.reduce.dataType;
            uint16_t setCKEId = instr->v2.reduce.setCKEId;
            uint16_t setCKEMask = instr->v2.reduce.setCKEMask;

            return Hccl::StringFormat(
                "Reduce %s, count[%u], reduceType[%u], castEn[%u] and dataType[%u], Set CKE[%u:%04x]",
                ParseMSListV2(instr->v2.reduce.msId, count).c_str(), count, reduceType, castEn, dataType, setCKEId,
                setCKEMask);
        }

        using ParseInstrFunc = std::string (*)(const CcuInstr*);
        static const std::unordered_map<uint16_t, ParseInstrFunc> g_parseInstrSqeMapV2
            = {{InstrHeader(LOAD_TYPE, LOADSQEARGSTOX_CODE).header, &ParseLoadSqeArgsToX},
               {InstrHeader(LOAD_TYPE, LOADIMDTOX_CODE).header, &ParseLoadImdToXn},
               {InstrHeader(LOAD_TYPE, CLEARX_CODE).header, &ParseClearX},
               {InstrHeader(LOAD_TYPE, NOP_CODE).header, &ParseNop},
               {InstrHeader(LOAD_TYPE, ADD_CODE).header, &ParseAdd},
               {InstrHeader(LOAD_TYPE, SUB_CODE).header, &ParseSub},
               {InstrHeader(LOAD_TYPE, MUL_CODE).header, &ParseMul},
               {InstrHeader(LOAD_TYPE, AND_CODE).header, &ParseAnd},
               {InstrHeader(LOAD_TYPE, OR_CODE).header, &ParseOr},
               {InstrHeader(LOAD_TYPE, XOR_CODE).header, &ParseXor},
               {InstrHeader(LOAD_TYPE, NOT_CODE).header, &ParseNot},
               {InstrHeader(LOAD_TYPE, SHL_CODE).header, &ParseShl},
               {InstrHeader(LOAD_TYPE, SHR_CODE).header, &ParseShr},
               {InstrHeader(LOAD_TYPE, LOAD_CODE).header, &ParseLoadFromMem},
               {InstrHeader(LOAD_TYPE, STORE_CODE).header, &ParseStoreXToMem},
               {InstrHeader(LOAD_TYPE, LOADX_CODE).header, &ParseLoadX},
               {InstrHeader(LOAD_TYPE, STOREX_CODE).header, &ParseStoreX},
               {InstrHeader(CTRL_TYPE, LOOP_CODE).header, &ParseLoop},
               {InstrHeader(CTRL_TYPE, LOOPGROUP_CODE).header, &ParseLoopGroup},
               {InstrHeader(CTRL_TYPE, JMP_CODE).header, &ParseJump},
               {InstrHeader(CTRL_TYPE, WAIT_CODE).header, &ParseWait},
               {InstrHeader(CTRL_TYPE, FENCE_CODE).header, &ParseFence},
               {InstrHeader(CTRL_TYPE, SETCKBIT_CODE).header, &ParseSetCKE},
               {InstrHeader(CTRL_TYPE, CLEARCKBIT_CODE).header, &ParseClearCKE},
               {InstrHeader(TRANS_TYPE, TRANSLOCMEMTOLOCMS_CODE).header, &ParseTransLocMemToLocMS},
               {InstrHeader(TRANS_TYPE, TRANSLOCMSTOLOCMEM_CODE).header, &ParseTransLocMSToLocMem},
               {InstrHeader(TRANS_TYPE, TRANSLOCMEMTOLOCMEM_CODE).header, &ParseTransLocMemToLocMem},
               {InstrHeader(TRANS_TYPE, TRANSMEM_CODE).header, &ParseTransMem},
               {InstrHeader(TRANS_TYPE, SYNCWTX_CODE).header, &ParseSyncWtX},
               {InstrHeader(TRANS_TYPE, SYNCATX_CODE).header, &ParseSyncAtX},
               {InstrHeader(REDUCE_TYPE, REDUCE_ADD_CODE).header, &ParseReduce},
               {InstrHeader(REDUCE_TYPE, REDUCE_MAX_CODE).header, &ParseReduce},
               {InstrHeader(REDUCE_TYPE, REDUCE_MIN_CODE).header, &ParseReduce}};

        std::string ParseInstrV2(const CcuInstr* instr)
        {
            if (instr == nullptr) {
                HCCL_ERROR("ParseInstr input nullptr!");
                return std::string();
            }

            auto it = g_parseInstrSqeMapV2.find(instr->header.header);
            if (it == g_parseInstrSqeMapV2.end()) {
                HCCL_ERROR("Not support instr type %u code %u", instr->header.type, instr->header.code);
                return std::string();
            }

            return it->second(instr);
        }

        void RelJmp(CcuInstr* instr, uint16_t targetXnId, uint32_t jmpInstrId, uint16_t xn0, uint16_t xn1)
        {
            CcuV2::LoadImdToXn(instr++, xn0, jmpInstrId);
            CcuV2::LoadImdToXn(instr++, xn1, JMP_TARGET_PC_IF_BRANCH);
            CcuV2::Jump(instr++, xn1, targetXnId, xn0, JMP_COND_TYPE_IF);
            CcuV2::LoadImdToXn(instr++, xn0, 0x10000 - jmpInstrId);
            CcuV2::Add(instr++, targetXnId, targetXnId, xn0);
            CcuV2::LoadImdToXn(instr++, xn1, JMP_TARGET_PC_NOP_BRANCH);
            CcuV2::Jump(instr++, xn1, 0, 0, JMP_COND_TYPE_UNCOND);
            CcuV2::Sub(instr++, targetXnId, targetXnId, xn0);
            CcuV2::Nop(instr++);
        }
    }; // namespace CcuV2

}; // namespace CcuRep
}; // namespace hcomm

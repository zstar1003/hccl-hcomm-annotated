/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <unordered_map>

#include "string_util.h"
#include "ccu_microcode.h"

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
} // namespace

namespace Hccl {
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

            instr->v2.store.srcType = 0x0;

            instr->v2.store.allocHint = cacheConfig.allocHint & 0x3;
            instr->v2.store.victimHint = cacheConfig.victimHint & 0x3;

            instr->v2.store.setCKEId = setCKEId;
            instr->v2.store.setCKEMask = setCKEMask;
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

        // Wait暂不实现
        // Fence暂不实现

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
            uint16_t usedMSId, uint16_t setCKEId, uint16_t setCKEMask, const CacheConfig& srcCacheConfig,
            const CacheConfig& dstcacheConfig)
        {
            instr->header = InstrHeader(TRANS_TYPE, TRANSLOCMEMTOLOCMEM_CODE);
            instr->v2.transLocMemToLocMem.xdId = dst;
            instr->v2.transLocMemToLocMem.xdtId = dstToken;
            instr->v2.transLocMemToLocMem.xsId = src;
            instr->v2.transLocMemToLocMem.xstId = srcToken;
            instr->v2.transLocMemToLocMem.xlId = len;
            instr->v2.transLocMemToLocMem.usedMSId = usedMSId;
            instr->v2.transLocMemToLocMem.msNum = CCU_MS_INTERLEAVE;

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

            instr->v2.transMem.order = config.order & 0x7;
            instr->v2.transMem.fence = config.fence & 0x1;
            instr->v2.transMem.cqe = config.cqe & 0x1;
            instr->v2.transMem.nf = config.nf & 0x1;
            instr->v2.transMem.udfEnable = config.udfEnable & 0x1;
            instr->v2.transMem.splitMode = config.splitMode & 0x1;
            instr->v2.transMem.se = config.se & 0x1;
            instr->v2.transMem.rmtJettyType = config.rmtJettyType & 0x3;

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
            CcuInstr* instr, uint16_t dst, uint16_t dstToken, uint16_t mask, uint16_t channelId, uint16_t setCKEId,
            uint16_t setCKEMask)
        {
            instr->header = InstrHeader(TRANS_TYPE, SYNCATX_CODE);
            instr->v2.syncAtX.xdId = dst;
            instr->v2.syncAtX.xdtId = dstToken;
            instr->v2.syncAtX.xsId = mask;
            instr->v2.syncAtX.xcId = channelId;

            instr->v2.syncAtX.parMode = 1;

            instr->v2.syncAtX.setCKEId = setCKEId;
            instr->v2.syncAtX.setCKEMask = setCKEMask;
        }

        // MSA~MSH Reduce到 MSA
        inline void Reduce(
            CcuInstr* instr, uint16_t* ms, uint16_t count, uint16_t castEn, uint16_t dataType, uint16_t setCKEId,
            uint16_t setCKEMask)
        {
            // 由调用者保证传入的count >= 2(reduce的数据源)
            count -= 2; // CCU指令中指定count数为实际参与运算的MS数减2

            for (uint16_t index = 0; index < CCU_REDUCE_MAX_MS; index++) {
                instr->v2.reduce.msId[index] = ms[index];
            }
            instr->v2.reduce.count = count & 0x7;
            instr->v2.reduce.castEn = castEn & 0x3;
            instr->v2.reduce.dataType = dataType & 0x1f;
            instr->v2.reduce.setCKEId = setCKEId;
            instr->v2.reduce.setCKEMask = setCKEMask;
        }

        void ReduceAdd(
            CcuInstr* instr, uint16_t* ms, uint16_t count, uint16_t castEn, uint16_t dataType, uint16_t setCKEId,
            uint16_t setCKEMask)
        {
            if (instr == nullptr) {
                HCCL_ERROR("[CcuV2::ReduceAdd] instr is nullptr!");
                return;
            }
            instr->header = InstrHeader(REDUCE_TYPE, REDUCE_ADD_CODE);
            Reduce(instr, ms, count, castEn, dataType, setCKEId, setCKEMask);
        }

        void ReduceMax(
            CcuInstr* instr, uint16_t* ms, uint16_t count, uint16_t dataType, uint16_t setCKEId, uint16_t setCKEMask)
        {
            instr->header = InstrHeader(REDUCE_TYPE, REDUCE_MAX_CODE);
            Reduce(instr, ms, count, 0, dataType, setCKEId, setCKEMask);
        }

        void ReduceMin(
            CcuInstr* instr, uint16_t* ms, uint16_t count, uint16_t dataType, uint16_t setCKEId, uint16_t setCKEMask)
        {
            instr->header = InstrHeader(REDUCE_TYPE, REDUCE_MIN_CODE);
            Reduce(instr, ms, count, 0, dataType, setCKEId, setCKEMask);
        }
    }; // namespace CcuV2

}; // namespace CcuRep
}; // namespace Hccl

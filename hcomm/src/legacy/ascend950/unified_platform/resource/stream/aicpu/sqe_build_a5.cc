/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sqe_build_a5.h"
#include "communicator_impl_lite_manager.h"
#include "adapter_rts_common.h"

namespace Hccl {

thread_local uint8_t g_sqeProfBit = 0;

void SetSqeProfilingEnabled(bool isEnabled) { g_sqeProfBit = isEnabled ? 1U : 0U; }

u32 GetKernelExecTimeoutFromEnvConfig()
{
    const u32 envTimeout = CommunicatorImplLiteMgr::GetInstance().GetEnvConfig().hcclExecTimeout;
    return envTimeout;
}

namespace {
    void ConstructLHWI(const Rt91095StarsCondIsaRegister_t dstReg, const u64 immd, Rt91095StarsCondOpLHWI_t& opLHWI)
    {
        opLHWI.opCode = static_cast<uint32_t>(Hccl::Rt91095StarsCondIsaOpCode_t::RT_91095_STARS_COND_ISA_OP_CODE_LWI);
        opLHWI.func3 = static_cast<uint32_t>(Rt91095StarsCondIsaLwiFunc3_t::RT_91095_STARS_COND_ISA_LWI_FUNC3_LHWI);
        opLHWI.rd = static_cast<uint32_t>(dstReg);
        opLHWI.immd = static_cast<uint32_t>((immd >> 49U) & 0x7FFFU); // High15-immd[63:49]
    }

    void ConstructLLWI(const Rt91095StarsCondIsaRegister_t dstReg, const u64 immd, Rt91095StarsCondOpLLWI_t& opLLWI)
    {
        opLLWI.opCode = static_cast<uint32_t>(Rt91095StarsCondIsaOpCode_t::RT_91095_STARS_COND_ISA_OP_CODE_LWI);
        opLLWI.func3 = static_cast<uint32_t>(Rt91095StarsCondIsaLwiFunc3_t::RT_91095_STARS_COND_ISA_LWI_FUNC3_LLWI);
        opLLWI.rd = static_cast<uint32_t>(dstReg);
        opLLWI.immdHigh = static_cast<uint32_t>((immd >> 32U) & 0x1FFFFU); // Low49-immd[48:32]
        opLLWI.immdLow = static_cast<uint32_t>(immd & 0xFFFFFFFFU);        // Low49-immd[31:0]
    }

    void ConstructLoadImm(
        const Rt91095StarsCondIsaRegister_t dstReg, const u64 addr, const Rt91095StarsCondIsaLoadImmFunc3_t func3,
        Rt91095StarsCondOpLoadImm_t& loadImm)
    {
        loadImm.opCode = static_cast<uint32_t>(Rt91095StarsCondIsaOpCode_t::RT_91095_STARS_COND_ISA_OP_CODE_LOAD_IMM);
        loadImm.rd = static_cast<uint32_t>(dstReg);
        loadImm.func3 = static_cast<uint32_t>(func3);
        loadImm.immdAddrHigh = static_cast<uint32_t>((addr >> 32U) & 0X1FFFFU); // bit[48:32]
        loadImm.immdAddrLow = static_cast<uint32_t>(addr & 0xFFFFFFFFU);        // bit[31:0]
    }

    void ConstructBranch(
        const Rt91095StarsCondIsaRegister_t rs1Reg, const Rt91095StarsCondIsaRegister_t rs2Reg,
        const Rt91095StarsCondIsaBranchFunc3_t func3, const uint8_t instrOffset, Rt91095StarsCondOpBranch_t& opBranch)
    {
        opBranch.opCode = static_cast<uint32_t>(Rt91095StarsCondIsaOpCode_t::RT_91095_STARS_COND_ISA_OP_CODE_BRANCH);
        opBranch.func3 = static_cast<uint32_t>(func3);
        opBranch.rs1 = static_cast<uint32_t>(rs1Reg);
        opBranch.rs2 = static_cast<uint32_t>(rs2Reg);
        opBranch.jumpInstrOffset = instrOffset & 0xFU; // Jump-immd[3:0]
    }

    void ConstructStore(
        const Rt91095StarsCondIsaRegister_t addrReg, const Rt91095StarsCondIsaRegister_t valReg,
        const uint16_t immdOffset, const Rt91095StarsCondIsaStoreFunc3_t func3, Rt91095StarsCondOpStore_t& opStore)
    {
        opStore.opCode = static_cast<uint32_t>(Rt91095StarsCondIsaOpCode_t::RT_91095_STARS_COND_ISA_OP_CODE_STORE);
        opStore.immdLow = static_cast<uint8_t>(immdOffset & 0x1FU); // S-immd[4:0]
        opStore.func3 = static_cast<uint32_t>(func3);
        opStore.rs1 = static_cast<uint32_t>(addrReg);
        opStore.rs2 = static_cast<uint32_t>(valReg);
        opStore.immdHigh = static_cast<uint8_t>((immdOffset & 0xFE0U) >> 5U); // S-immd[11:5]
    }

    void ConstructNop(Rt91095StarsCondOpNop_t& nop)
    {
        nop.opCode = static_cast<uint32_t>(Rt91095StarsCondIsaOpCode_t::RT_91095_STARS_COND_ISA_OP_CODE_NOP);
        nop.rd = static_cast<uint32_t>(Rt91095StarsCondIsaRegister_t::RT_91095_STARS_COND_ISA_REGISTER_R0);
        nop.func3 = static_cast<uint32_t>(Rt91095StarsCondIsaOpImmFunc3_t::RT_91095_STARS_COND_ISA_OP_IMM_FUNC3_NOP);
        nop.rs1 = static_cast<uint32_t>(Rt91095StarsCondIsaRegister_t::RT_91095_STARS_COND_ISA_REGISTER_R0);
        nop.immd = 0U;
    }
} // namespace

void BuildA5SqeCCoreNotifyWait(u32 streamId, u32 taskId, u64 waitAddr, u64 actAddr, bool last, uint8_t* const sqeIn)
{
    Rt91095StarsCCoreSqeNotifyWait* sqe = (Rt91095StarsCCoreSqeNotifyWait*)sqeIn;
    sqe->header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_COND);
    sqe->header.rtStreamId = static_cast<uint16_t>(taskId);
    sqe->header.taskId = static_cast<uint16_t>(taskId >> LOW_BITS);

    sqe->kernelCredit = RT_STARS_DEFAULT_KERNEL_CREDIT;
    sqe->csc = 1U;

    constexpr Rt91095StarsCondIsaRegister_t r0 = Rt91095StarsCondIsaRegister_t::RT_91095_STARS_COND_ISA_REGISTER_R0;
    constexpr Rt91095StarsCondIsaRegister_t r1 = Rt91095StarsCondIsaRegister_t::RT_91095_STARS_COND_ISA_REGISTER_R1;
    constexpr Rt91095StarsCondIsaRegister_t r2 = Rt91095StarsCondIsaRegister_t::RT_91095_STARS_COND_ISA_REGISTER_R2;
    constexpr Rt91095StarsCondIsaRegister_t r3 = Rt91095StarsCondIsaRegister_t::RT_91095_STARS_COND_ISA_REGISTER_R3;

    // load current Turn to r3
    ConstructLoadImm(
        r3, actAddr, Rt91095StarsCondIsaLoadImmFunc3_t::RT_91095_STARS_COND_ISA_LOAD_IMM_FUNC3_LHU, sqe->ldrImm1);

    // load sendcnt to r2
    ConstructLoadImm(
        r2, waitAddr, Rt91095StarsCondIsaLoadImmFunc3_t::RT_91095_STARS_COND_ISA_LOAD_IMM_FUNC3_LHU, sqe->ldrImm2);
    uint8_t loadInstrOff
        = (offsetof(Rt91095StarsCCoreSqeNotifyWait, ldrImm2) - offsetof(Rt91095StarsCCoreSqeNotifyWait, ldrImm1));
    loadInstrOff = loadInstrOff / sizeof(uint32_t);

    // r2(sendCnt) < r3(curTurn)，goto reload r2
    ConstructBranch(
        r2, r3, Rt91095StarsCondIsaBranchFunc3_t::RT_91095_STARS_COND_ISA_BRANCH_FUNC3_BLTU, loadInstrOff, sqe->beq);

    if (last) {
        // load sendcount addr to r1
        ConstructLLWI(r1, waitAddr, sqe->clear.llwi1);
        ConstructLHWI(r1, waitAddr, sqe->clear.lhwi1);
        // the last turn clear sendCnt, r0(0) value store to r1(sendCnt),
        ConstructStore(
            r1, r0, 0U, Rt91095StarsCondIsaStoreFunc3_t::RT_91095_STARS_COND_ISA_STORE_FUNC3_SH, sqe->clear.sw);
        for (Rt91095StarsCondOpNop_t& nop : sqe->clear.nop) {
            ConstructNop(nop);
        }
    } else {
        for (Rt91095StarsCondOpNop_t& nop : sqe->nop) {
            ConstructNop(nop);
        }
    }

    sqe->header.sqeProf = g_sqeProfBit;
    HCCL_INFO(
        "[SQE]CCoreWait: waitAddr=%llu, actAddr=%llu, last=%u, streamId=%u, taskId=%u, sqeProf=%u, "
        "ISA=%08x %08x %08x %08x %08x %08x %08x",
        waitAddr, actAddr, last, streamId, taskId, static_cast<uint32_t>(sqe->header.sqeProf), sqe->ldrImm1,
        sqe->ldrImm2, sqe->beq, sqe->clear.llwi1, sqe->clear.lhwi1, sqe->clear.sw, sqe->clear.nop[0]);
}

void BuildA5SqeCCoreNotifyRecord(u32 streamId, u32 taskId, u64 writeAddr, u64 valueAddr, uint8_t* const sqeIn)
{
    Rt91095StarsCCoreSqeNotifyRecord* sqe = (Rt91095StarsCCoreSqeNotifyRecord*)sqeIn;
    sqe->header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_COND);
    sqe->header.rtStreamId = static_cast<uint16_t>(taskId);
    sqe->header.taskId = static_cast<uint16_t>(taskId >> LOW_BITS);

    sqe->kernelCredit = RT_STARS_DEFAULT_KERNEL_CREDIT;
    sqe->csc = 1U;

    constexpr Rt91095StarsCondIsaRegister_t r1 = Rt91095StarsCondIsaRegister_t::RT_91095_STARS_COND_ISA_REGISTER_R1;
    constexpr Rt91095StarsCondIsaRegister_t r2 = Rt91095StarsCondIsaRegister_t::RT_91095_STARS_COND_ISA_REGISTER_R2;

    ConstructLoadImm(
        r1, valueAddr, Rt91095StarsCondIsaLoadImmFunc3_t::RT_91095_STARS_COND_ISA_LOAD_IMM_FUNC3_LHU, sqe->ldrImm);
    ConstructLLWI(r2, writeAddr, sqe->llwi1);
    ConstructLHWI(r2, writeAddr, sqe->lhwi1);

    ConstructStore(r2, r1, 0U, Rt91095StarsCondIsaStoreFunc3_t::RT_91095_STARS_COND_ISA_STORE_FUNC3_SH, sqe->sw);
    for (Rt91095StarsCondOpNop_t& nop : sqe->nop) {
        ConstructNop(nop);
    }

    sqe->header.sqeProf = g_sqeProfBit;
    HCCL_INFO(
        "[SQE]CCoreWrite: writeAddr=%p, valueAddr=%p, streamId=%u, taskId=%u, sqeProf=%u, "
        "ISA=%08x %08x %08x %08x %08x",
        writeAddr, valueAddr, streamId, taskId, static_cast<uint32_t>(sqe->header.sqeProf), sqe->ldrImm, sqe->llwi1,
        sqe->lhwi1, sqe->sw, sqe->nop[0]);
}

// 写64bit值的形式来完成敲DB
void BuildA5SqeRdmaDbSend(u32 streamId, u32 taskId, u64 dbAddr, u64 dbValue, uint8_t* const sqeIn)
{
    (void)streamId;
    Rt91095StarsWriteValueSqe* sqe = reinterpret_cast<Rt91095StarsWriteValueSqe*>(sqeIn);
    SetSqeHeaderTaskFields(sqe, taskId);
    sqe->header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_WRITE_VALUE);

    sqe->kernelCredit = RT_STARS_DEFAULT_KERNEL_CREDIT;
    sqe->header.rtStreamId = streamId;
    sqe->header.taskId = taskId;

    sqe->writeAddrLow = dbAddr & MASK_32_BIT;
    sqe->writeAddrHigh = (dbAddr >> UINT32_BIT_NUM) & MASK_17_BIT;

    sqe->awsize = RtStarsWriteValueSizeType::RT_STARS_WRITE_VALUE_SIZE_TYPE_64BIT;             // writeValue 为 8 byte
    sqe->writeValuePart[0] = static_cast<uint32_t>(dbValue & MASK_32_BIT);                     // low  32 bit
    sqe->writeValuePart[1] = static_cast<uint32_t>((dbValue >> UINT32_BIT_NUM) & MASK_32_BIT); // high 32 bit

    sqe->va = 1U;
    sqe->header.sqeProf = g_sqeProfBit;
}

} // namespace Hccl

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_INS_GENERATOR_V2
#define CCU_INS_GENERATOR_V2

#include "ccu_ins_generator_base.h"

namespace hcomm {

namespace CcuRep {

    class CcuInsGeneratorV2 : public CcuInsGeneratorBase {
    public:
        CcuInsGeneratorV2() {}

        // 虚析构函数，确保派生类对象正确析构
        ~CcuInsGeneratorV2() override = default;

        // data
        HcclResult CcuRepBufLocReadTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepBufLocRead* repBufLocRead, const TransDep& dep) override;
        HcclResult CcuRepBufLocWriteTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepBufLocWrite* repBufLocWrite, const TransDep& dep) override;
        HcclResult CcuRepBufReadTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepBufRead* repBufRead, const TransDep& dep) override;
        HcclResult
        CcuRepBufReduceTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepBufReduce* ccuRepBufReduce) override;
        HcclResult CcuRepBufWriteTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepBufWrite* ccuRepBufWrite, const TransDep& dep) override;
        HcclResult CcuRepLocCpyTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepLocCpy* ccuRepLocCpy, const TransDep& dep) override;
        HcclResult CcuRepReadTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepRead* repRemMem) override;
        HcclResult CcuRepRemMemTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepRemMem* repRemMem) override;
        HcclResult CcuRepWriteTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepWrite* repWrite) override;

        // sync
        HcclResult CcuRepLocRecordEventTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepLocRecordEvent* ccuRepLocRecordEvent) override;
        HcclResult CcuRepLocWaitEventTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepLocWaitEvent* ccuRepLocWaitEvent) override;
        HcclResult CcuRepLocWaitNotifyTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepLocWaitNotify* ccuRepLocWaitNotify) override;
        HcclResult CcuRepRecordSharedNotifyTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepRecordSharedNotify* ccuRepRecordSharedNotify,
            const TransDep& dep) override;
        HcclResult
        CcuRepRemWaitSemTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepRemWaitSem* ccuRepRemWaitSem) override;
        HcclResult
        CcuRepRemPostVarTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepRemPostVar* ccuRepRemPostVar) override;
        HcclResult CcuRepRemPostSemTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepRemPostSem* ccuRepRemPostSem, const TransDep& dep) override;

        // logical
        HcclResult
        CcuRepAndTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepAnd* ccuRepAnd, const TransDep& dep) override;
        HcclResult
        CcuRepNotTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepNot* ccuRepNot, const TransDep& dep) override;
        HcclResult
        CcuRepOrTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepOr* ccuRepOr, const TransDep& dep) override;
        HcclResult
        CcuRepXorTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepXor* ccuRepXor, const TransDep& dep) override;

        // shift
        HcclResult
        CcuRepShLTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepShL* ccuRepShL, const TransDep& dep) override;
        HcclResult
        CcuRepShRTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepShR* ccuRepShR, const TransDep& dep) override;

        // arithmetic
        HcclResult
        CcuRepAddTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepAdd* ccuRepAdd, const TransDep& dep) override;
        HcclResult CcuRepAssignTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepAssign* ccuRepAssign, const TransDep& dep) override;
        HcclResult CcuRepMulTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepMul* ccuRepMul) override;
        HcclResult CcuRepSubTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepSub* ccuRepSub) override;

        // control
        HcclResult CcuRepFuncBlockTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepFuncBlock* funcBlockPtr,
            const TransDep& dep, uint32_t step) override;
        HcclResult CcuRepFuncCallTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepFuncCall* funcCallPtr,
            const TransDep& dep) override;
        HcclResult CcuRepJumpTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepJump* jumpPtr,
            const TransDep& dep) override;
        HcclResult CcuRepJumpNETranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepJumpNE* jumpNEPtr,
            const TransDep& dep) override;
        HcclResult CcuRepJumpEQTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepJumpEQ* jumpEQPtr,
            const TransDep& dep) override;
        HcclResult CcuRepJumpLETranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepJumpLE* jumpLEPtr,
            const TransDep& dep) override;
        HcclResult CcuRepJumpGETranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepJumpGE* jumpGEPtr,
            const TransDep& dep) override;
        HcclResult CcuRepJumpGTTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepJumpGT* jumpGTPtr,
            const TransDep& dep) override;
        HcclResult CcuRepJumpLTTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepJumpLT* jumpLTPtr,
            const TransDep& dep) override;

        // loop
        HcclResult
        CcuRepLoopTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepLoop* loopPtr) override;
        HcclResult CcuRepLoopCallTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepLoopCall* loopCallPtr,
            const TransDep& dep) override;
        HcclResult CcuRepSetLoopTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepSetLoop* setLoopPtr) override;
        HcclResult CcuRepLoopGroupBundleTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepLoopGroupBundle* bundlePtr,
            const TransDep& dep) override;
        uint16_t CcuRepLoopGroupBundleInstrCount(const CcuRepLoopGroupBundle* bundlePtr) const override;

        // common
        HcclResult CcuRepLoadTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepLoad* loadPtr,
            const TransDep& dep) override;
        HcclResult CcuRepLoadVarTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepLoadVar* loadVarPtr,
            const TransDep& dep) override;
        HcclResult CcuRepLoadArgTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepLoadArg* loadArgPtr,
            const TransDep& dep) override;
        HcclResult CcuRepNopTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepNop* nopPtr,
            const TransDep& dep) override;
        HcclResult CcuRepStoreTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepStore* storePtr,
            const TransDep& dep) override;
        HcclResult CcuRepStoreVarTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepStoreVar* storeVarPtr,
            const TransDep& dep) override;

        // halfRtt
        HcclResult CcuRepWriteVarAtomicTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepWriteVarAtomic* writeVarAtomPtr,
            const TransDep& dep) override;
        HcclResult CcuRepWriteWithCntIncTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepWriteWithCntInc* writeCntIncPtr,
            const TransDep& dep) override;
        HcclResult CcuRepCascCntWaitTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepCascCntWait* cascCntWaitPtr,
            const TransDep& dep) override;
        HcclResult CcuRepCascCntClearTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepCascCntClear* cascCntClearPtr,
            const TransDep& dep) override;
        HcclResult CcuRepLoadAddImmTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepLoadAddImm* loadAddImmPtr,
            const TransDep& dep) override;
        HcclResult CcuRepStoreAddImmTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepStoreAddImm* storeAddImmPtr,
            const TransDep& dep) override;

        uint32_t GetInstrCount(CcuRepType repType) override;

        HcclResult PrepareConstValue(CcuRepBase* repPtr, const TransDep& dep, CcuKernel* ccuKernel) override;

    private:
        HcclResult CcuRepJumpTranslateV2Base(
            CcuInstr* const& curInstr, uint16_t& curInstrId, CcuRepJumpBase* jumpBasePtr, uint64_t expected,
            const Variable& condition, const Variable& expectedVar, ConditionType condType) const;

        void LoadFuncCallInArgs(
            CcuInstr* instr, std::vector<CcuRepArg>& inArgs, std::vector<Variable>& formalIns,
            uint16_t reserveXnId) const;
        void LoadFuncCallOutArgs(
            CcuInstr* instr, uint32_t offset, std::vector<CcuRepArg>& outArgs, CcuRepReferenceManager* funcManager,
            uint16_t reserveXnId) const;
        HcclResult LoadLoopCallArg(CcuInstr*& instr, const CcuRepArg& inArg, const CcuRepArg& blkArg) const;
        HcclResult PrepareLoadConstValue(CcuRepBase* repPtr, const TransDep& dep, CcuKernel* ccuKernel) const;
        HcclResult PrepareLoadVarConstValue(CcuRepBase* repPtr, const TransDep& dep, CcuKernel* ccuKernel) const;
        HcclResult PrepareStoreConstValue(CcuRepBase* repPtr, const TransDep& dep, CcuKernel* ccuKernel) const;
        HcclResult PrepareStoreVarConstValue(CcuRepBase* repPtr, const TransDep& dep, CcuKernel* ccuKernel) const;
        HcclResult PrepareRemPostSemConstValue(CcuRepBase* repPtr, const TransDep& dep, CcuKernel* ccuKernel) const;
        HcclResult PrepareRemPostVarConstValue(CcuRepBase* repPtr, const TransDep& dep, CcuKernel* ccuKernel) const;
        HcclResult PrepareWriteConstValue(CcuRepBase* repPtr, const TransDep& dep, CcuKernel* ccuKernel) const;
        HcclResult PrepareReadConstValue(CcuRepBase* repPtr, const TransDep& dep, CcuKernel* ccuKernel) const;
        HcclResult PrepareBufWriteConstValue(CcuRepBase* repPtr, const TransDep& dep, CcuKernel* ccuKernel) const;
        HcclResult PrepareBufReadConstValue(CcuRepBase* repPtr, const TransDep& dep, CcuKernel* ccuKernel) const;
        HcclResult PrepareLocCpyConstValue(CcuRepBase* repPtr, const TransDep& dep, CcuKernel* ccuKernel) const;
        HcclResult
        PrepareRecordSharedNotifyConstValue(CcuRepBase* repPtr, const TransDep& dep, CcuKernel* ccuKernel) const;
        HcclResult LoopConfigTranslate(
            CcuInstr*& instr, uint16_t& curInstrId, const CcuRepLoopGroupBundle* bundlePtr, const TransDep& dep) const;
        HcclResult LoopGroupConfigTranslate(
            CcuInstr*& instr, uint16_t& curInstrId, const CcuRepLoopGroupBundle* bundlePtr, const TransDep& dep,
            bool isConfig, bool isCompat, uint16_t& loopGroupConfigId) const;

        static constexpr uint32_t V2_FUNC_BLOCK_INSTR_NUM = 11; // FuncBlock: RelJmp(9)+Jump(1)+Nop(1)
        static constexpr uint32_t V2_FUNC_CALL_INSTR_NUM = 13;  // FuncCall: FuncBlock(11)+Call(2)

        std::unordered_map<CcuRepType, uint32_t> repTypeInstrCount
            = {{CcuRepType::READ, 1},
               {CcuRepType::WRITE, 1},
               {CcuRepType::REM_MEM, 2},
               {CcuRepType::BUF_READ, 1},
               {CcuRepType::LOCAL_CPY, 1},
               {CcuRepType::LOCAL_REDUCE, 1},
               {CcuRepType::BUF_WRITE, 1},
               {CcuRepType::BUF_REDUCE, 1},
               {CcuRepType::BUF_LOC_READ, 1},
               {CcuRepType::BUF_LOC_WRITE, 1},

               {CcuRepType::ASSIGN, 1},
               {CcuRepType::ADD, 1},
               {CcuRepType::MUL, 1},
               {CcuRepType::SUB, 1},

               {CcuRepType::LOC_RECORD_EVENT, 1},
               {CcuRepType::LOC_WAIT_EVENT, 1},
               {CcuRepType::LOC_WAIT_NOTIFY, 1},
               {CcuRepType::RECORD_SHARED_NOTIFY, 1},
               {CcuRepType::REM_POST_SEM, 1},
               {CcuRepType::REM_POST_VAR, 1},
               {CcuRepType::REM_WAIT_SEM, 1},

               {CcuRepType::FUNC_BLOCK, V2_FUNC_BLOCK_INSTR_NUM},
               {CcuRepType::FUNC_CALL, V2_FUNC_CALL_INSTR_NUM},
               {CcuRepType::JUMP, 2},
               {CcuRepType::JUMP_NE, 3},
               {CcuRepType::JUMP_EQ, 3},
               {CcuRepType::LOOP, 1},
               {CcuRepType::LOOPGROUP, 1},
               {CcuRepType::SET_LOOP, 1},

               {CcuRepType::LOAD, 3},
               {CcuRepType::LOAD_VAR, 3},
               {CcuRepType::LOAD_ARG, 1},
               {CcuRepType::STORE, 3},
               {CcuRepType::STORE_VAR, 3},

               {CcuRepType::WRITE_VAR_ATOMIC, 1},
               {CcuRepType::WRITE_WITH_CNT_INC, 1},
               {CcuRepType::CASC_CNT_WAIT, 2},
               {CcuRepType::CASC_CNT_CLEAR, 1},
               {CcuRepType::LOAD_ADD_IMM, 1},
               {CcuRepType::STORE_ADD_IMM, 1},

               {CcuRepType::AND, 1},
               {CcuRepType::NOT, 1},
               {CcuRepType::OR, 1},
               {CcuRepType::XOR, 1},
               {CcuRepType::SHL, 1},
               {CcuRepType::SHR, 1},

               {CcuRepType::NOP, 1}};
    };

    uint32_t GetRelativeInstrId(uint32_t currentInstrId, uint32_t targetInstrId);

} // namespace CcuRep
} // namespace hcomm

#endif

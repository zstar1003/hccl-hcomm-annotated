/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_INS_GENERATOR_V1
#define CCU_INS_GENERATOR_V1

#include "ccu_ins_generator_base.h"
namespace hcomm {
namespace CcuRep {

    class CcuInsGeneratorV1 : public CcuInsGeneratorBase {
    public:
        CcuInsGeneratorV1() {}

        ~CcuInsGeneratorV1() override = default;

        // data
        HcclResult CcuRepLocCpyTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepLocCpy* ccuRepLocCpy, const TransDep& dep) override;
        HcclResult CcuRepReadTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepRead* repRemMem) override;
        HcclResult CcuRepRemMemTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepRemMem* repRemMem) override;
        HcclResult CcuRepWriteTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepWrite* repWrite) override;
        HcclResult CcuRepBufLocReadTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepBufLocRead* repBufLocRead, const TransDep& dep) override;
        HcclResult CcuRepBufLocWriteTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepBufLocWrite* repBufLocWrite, const TransDep& dep) override;
        HcclResult CcuRepBufWriteTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepBufWrite* ccuRepBufWrite, const TransDep& dep) override;
        HcclResult CcuRepBufReadTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepBufRead* repBufRead, const TransDep& dep) override;
        HcclResult
        CcuRepBufReduceTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepBufReduce* ccuRepBufReduce) override;

        // sync
        HcclResult CcuRepRecordSharedNotifyTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepRecordSharedNotify* ccuRepRecordSharedNotify,
            const TransDep& dep) override;
        HcclResult
        CcuRepRemPostVarTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepRemPostVar* ccuRepRemPostVar) override;
        HcclResult CcuRepRemPostSemTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepRemPostSem* ccuRepRemPostSem, const TransDep& dep) override;
        HcclResult
        CcuRepRemWaitSemTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepRemWaitSem* ccuRepRemWaitSem) override;
        HcclResult CcuRepLocRecordEventTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepLocRecordEvent* ccuRepLocRecordEvent) override;
        HcclResult CcuRepLocWaitEventTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepLocWaitEvent* ccuRepLocWaitEvent) override;
        HcclResult CcuRepLocWaitNotifyTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepLocWaitNotify* ccuRepLocWaitNotify) override;

        // logical
        HcclResult
        CcuRepOrTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepOr* ccuRepOr, const TransDep& dep) override;
        HcclResult
        CcuRepXorTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepXor* ccuRepXor, const TransDep& dep) override;
        HcclResult
        CcuRepAndTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepAnd* ccuRepAnd, const TransDep& dep) override;
        HcclResult
        CcuRepNotTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepNot* ccuRepNot, const TransDep& dep) override;

        // arithmetic
        HcclResult
        CcuRepAddTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepAdd* ccuRepAdd, const TransDep& dep) override;
        HcclResult CcuRepAssignTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepAssign* ccuRepAssign, const TransDep& dep) override;
        HcclResult CcuRepMulTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepMul* ccuRepMul) override;
        HcclResult CcuRepSubTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepSub* ccuRepSub) override;

        // shift
        HcclResult
        CcuRepShLTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepShL* ccuRepShL, const TransDep& dep) override;
        HcclResult
        CcuRepShRTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepShR* ccuRepShR, const TransDep& dep) override;

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
        HcclResult CcuRepJumpEQTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepJumpEQ* jumpEQPtr,
            const TransDep& dep) override;
        HcclResult CcuRepJumpNETranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepJumpNE* jumpNEPtr,
            const TransDep& dep) override;
        HcclResult CcuRepJumpGETranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepJumpGE* jumpGEPtr,
            const TransDep& dep) override;
        HcclResult CcuRepJumpLETranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepJumpLE* jumpLEPtr,
            const TransDep& dep) override;
        HcclResult CcuRepJumpLTTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepJumpLT* jumpLTPtr,
            const TransDep& dep) override;
        HcclResult CcuRepJumpGTTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepJumpGT* jumpGTPtr,
            const TransDep& dep) override;

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
        HcclResult CcuRepStoreTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepStore* storePtr,
            const TransDep& dep) override;
        HcclResult CcuRepStoreVarTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepStoreVar* storeVarPtr,
            const TransDep& dep) override;
        HcclResult CcuRepNopTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepNop* nopPtr,
            const TransDep& dep) override;

        uint32_t GetInstrCount(CcuRepType repType) override;

    private:
        void LoadFuncCallInArgs(
            CcuInstr* instr, std::vector<CcuRepArg>& inArgs, std::vector<Variable>& formalIns,
            uint16_t reserveXnId) const;
        void LoadFuncCallOutArgs(
            CcuInstr* instr, uint32_t offset, std::vector<CcuRepArg>& outArgs, CcuRepReferenceManager* funcManager,
            uint16_t reserveXnId) const;
        HcclResult
        LoadLoopCallArg(CcuInstr*& instr, const CcuRepArg& inArg, const CcuRepArg& blkArg, const TransDep& dep) const;
        void LoadLoopGroupParams(
            CcuInstr*& instr, uint16_t& curInstrId, const CcuRepLoopGroupBundle* bundlePtr, const TransDep& dep) const;
        void
        LoadLoopGroupBundleConfig(CcuInstr*& instr, uint16_t& curInstrId, const CcuRepLoopGroupBundle* bundlePtr) const;

        std::unordered_map<CcuRepType, uint32_t> repTypeInstrCount
            = {{CcuRepType::READ, 1},
               {CcuRepType::WRITE, 1},
               {CcuRepType::REM_MEM, 2},
               {CcuRepType::BUF_READ, 1},
               {CcuRepType::BUF_WRITE, 1},
               {CcuRepType::BUF_REDUCE, 1},
               {CcuRepType::BUF_LOC_READ, 1},
               {CcuRepType::BUF_LOC_WRITE, 1},
               {CcuRepType::LOCAL_CPY, 1},
               {CcuRepType::LOCAL_REDUCE, 1},

               {CcuRepType::LOC_RECORD_EVENT, 1},
               {CcuRepType::LOC_WAIT_EVENT, 1},
               {CcuRepType::LOC_WAIT_NOTIFY, 1},
               {CcuRepType::RECORD_SHARED_NOTIFY, 1},
               {CcuRepType::REM_POST_SEM, 1},
               {CcuRepType::REM_POST_VAR, 1},
               {CcuRepType::REM_WAIT_SEM, 1},

               {CcuRepType::ASSIGN, 1},
               {CcuRepType::ADD, 1},

               {CcuRepType::FUNC_BLOCK, 2},
               {CcuRepType::FUNC_CALL, 4},
               {CcuRepType::JUMP, 2},
               {CcuRepType::JUMP_NE, 2},
               {CcuRepType::JUMP_EQ, 5},
               {CcuRepType::LOOP, 1},
               {CcuRepType::LOOPGROUP, 1},
               {CcuRepType::SET_LOOP, 2},
               {CcuRepType::LOAD, 7},
               {CcuRepType::LOAD_VAR, 7},
               {CcuRepType::LOAD_ARG, 1},
               {CcuRepType::STORE, 7},
               {CcuRepType::STORE_VAR, 7},

               {CcuRepType::NOP, 1}};
    };

} // namespace CcuRep
} // namespace hcomm

#endif

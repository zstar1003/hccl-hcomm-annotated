/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_INS_GENERATOR_BASE
#define CCU_INS_GENERATOR_BASE

#include "ccu_rep_base_v1.h"
#include "ccu_datatype_v1.h"
#include "ccu_microcode_v1.h"
#include "ccu_rep_v1.h"
#include "ccu_kernel.h"
#include "ccu_log.h"

namespace hcomm {
namespace CcuRep {

    class CcuInsGeneratorBase {
    public:
        CcuInsGeneratorBase() {}

        // 虚析构函数，确保派生类对象正确析构
        virtual ~CcuInsGeneratorBase() = default;
        // data
        virtual HcclResult CcuRepBufLocReadTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepBufLocRead* repBufLocRead, const TransDep& dep)
            = 0;
        virtual HcclResult CcuRepBufLocWriteTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepBufLocWrite* repBufLocWrite, const TransDep& dep)
            = 0;
        virtual HcclResult
        CcuRepBufReadTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepBufRead* repBufRead, const TransDep& dep)
            = 0;
        virtual HcclResult
        CcuRepBufReduceTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepBufReduce* ccuRepBufReduce)
            = 0;
        virtual HcclResult CcuRepBufWriteTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepBufWrite* ccuRepBufWrite, const TransDep& dep)
            = 0;
        virtual HcclResult
        CcuRepLocCpyTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepLocCpy* ccuRepLocCpy, const TransDep& dep)
            = 0;
        virtual HcclResult CcuRepReadTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepRead* repRemMem) = 0;
        virtual HcclResult CcuRepRemMemTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepRemMem* repRemMem) = 0;
        virtual HcclResult CcuRepWriteTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepWrite* repWrite) = 0;

        // sync
        virtual HcclResult CcuRepLocRecordEventTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepLocRecordEvent* ccuRepLocRecordEvent)
            = 0;
        virtual HcclResult
        CcuRepLocWaitEventTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepLocWaitEvent* ccuRepLocWaitEvent)
            = 0;
        virtual HcclResult
        CcuRepLocWaitNotifyTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepLocWaitNotify* ccuRepLocWaitNotify)
            = 0;
        virtual HcclResult CcuRepRecordSharedNotifyTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepRecordSharedNotify* ccuRepRecordSharedNotify,
            const TransDep& dep)
            = 0;
        virtual HcclResult
        CcuRepRemWaitSemTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepRemWaitSem* ccuRepRemWaitSem)
            = 0;
        virtual HcclResult
        CcuRepRemPostVarTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepRemPostVar* ccuRepRemPostVar)
            = 0;
        virtual HcclResult CcuRepRemPostSemTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepRemPostSem* ccuRepRemPostSem, const TransDep& dep)
            = 0;
        // logical
        virtual HcclResult
        CcuRepAndTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepAnd* ccuRepAnd, const TransDep& dep)
            = 0;
        virtual HcclResult
        CcuRepNotTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepNot* ccuRepNot, const TransDep& dep)
            = 0;
        virtual HcclResult
        CcuRepOrTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepOr* ccuRepOr, const TransDep& dep)
            = 0;
        virtual HcclResult
        CcuRepXorTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepXor* ccuRepXor, const TransDep& dep)
            = 0;

        // shift
        virtual HcclResult
        CcuRepShLTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepShL* ccuRepShL, const TransDep& dep)
            = 0;
        virtual HcclResult
        CcuRepShRTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepShR* ccuRepShR, const TransDep& dep)
            = 0;

        // arithmetic
        virtual HcclResult
        CcuRepAddTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepAdd* ccuRepAdd, const TransDep& dep)
            = 0;
        ;
        virtual HcclResult
        CcuRepAssignTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepAssign* ccuRepAssign, const TransDep& dep)
            = 0;
        virtual HcclResult CcuRepMulTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepMul* ccuRepMul) = 0;
        virtual HcclResult CcuRepSubTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, CcuRepSub* ccuRepSub) = 0;

        // control
        virtual HcclResult CcuRepFuncBlockTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepFuncBlock* funcBlockPtr,
            const TransDep& dep, uint32_t step)
            = 0;
        virtual HcclResult CcuRepFuncCallTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepFuncCall* funcCallPtr,
            const TransDep& dep)
            = 0;
        virtual HcclResult CcuRepJumpTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepJump* jumpPtr, const TransDep& dep)
            = 0;
        virtual HcclResult CcuRepJumpNETranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepJumpNE* jumpNEPtr, const TransDep& dep)
            = 0;
        virtual HcclResult CcuRepJumpEQTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepJumpEQ* jumpEQPtr, const TransDep& dep)
            = 0;
        virtual HcclResult CcuRepJumpLETranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepJumpLE* jumpLEPtr, const TransDep& dep)
            = 0;
        virtual HcclResult CcuRepJumpGETranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepJumpGE* jumpGEPtr, const TransDep& dep)
            = 0;
        virtual HcclResult CcuRepJumpGTTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepJumpGT* jumpGTPtr, const TransDep& dep)
            = 0;
        virtual HcclResult CcuRepJumpLTTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepJumpLT* jumpLTPtr, const TransDep& dep)
            = 0;

        // loop
        virtual HcclResult
        CcuRepLoopTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepLoop* loopPtr)
            = 0;
        virtual HcclResult CcuRepLoopCallTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepLoopCall* loopCallPtr,
            const TransDep& dep)
            = 0;
        virtual HcclResult
        CcuRepSetLoopTranslate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepSetLoop* setLoopPtr)
            = 0;
        virtual HcclResult CcuRepLoopGroupBundleTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepLoopGroupBundle* bundlePtr,
            const TransDep& dep)
            = 0;
        virtual uint16_t CcuRepLoopGroupBundleInstrCount(const CcuRepLoopGroupBundle* bundlePtr) const = 0;

        // common
        virtual HcclResult CcuRepLoadTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepLoad* loadPtr, const TransDep& dep)
            = 0;
        virtual HcclResult CcuRepLoadVarTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepLoadVar* loadVarPtr,
            const TransDep& dep)
            = 0;
        virtual HcclResult CcuRepLoadArgTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepLoadArg* loadArgPtr,
            const TransDep& dep)
            = 0;
        virtual HcclResult CcuRepNopTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepNop* nopPtr, const TransDep& dep)
            = 0;
        virtual HcclResult CcuRepStoreTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepStore* storePtr, const TransDep& dep)
            = 0;
        virtual HcclResult CcuRepStoreVarTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepStoreVar* storeVarPtr,
            const TransDep& dep)
            = 0;

        // halfRtt
        virtual HcclResult CcuRepWriteVarAtomicTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepWriteVarAtomic* writeVarAtomPtr,
            const TransDep& dep);
        virtual HcclResult CcuRepWriteWithCntIncTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepWriteWithCntInc* writeCntIncPtr,
            const TransDep& dep);
        virtual HcclResult CcuRepCascCntWaitTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepCascCntWait* cascCntWaitPtr,
            const TransDep& dep);
        virtual HcclResult CcuRepCascCntClearTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepCascCntClear* cascCntClearPtr,
            const TransDep& dep);
        virtual HcclResult CcuRepLoadAddImmTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepLoadAddImm* loadAddImmPtr,
            const TransDep& dep);
        virtual HcclResult CcuRepStoreAddImmTranslate(
            CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepStoreAddImm* storeAddImmPtr,
            const TransDep& dep);

        virtual uint32_t GetInstrCount(CcuRepType repType) = 0;

        virtual HcclResult PrepareConstValue(CcuRepBase* repPtr, const TransDep& dep, CcuKernel* ccuKernel)
        {
            // A5使用基类空实现；A6需要根据repType做不同处理
            (void)repPtr;
            (void)dep;
            (void)ccuKernel;
            return HcclResult::HCCL_SUCCESS;
        }

    protected:
        struct FuncCallContext {
            uint32_t inArgCount{0};
            CcuRepReferenceManager* funcManager{nullptr};
            std::shared_ptr<CcuRepFuncBlock> funcBlock;
            CcuInstr* instr{nullptr};
            std::vector<Variable> formalIns;
        };

        HcclResult PrepareFuncCallContext(CcuRepFuncCall* funcCallPtr, FuncCallContext& ctx) const
        {
            CHK_PTR_NULL(funcCallPtr);
            ctx.inArgCount = funcCallPtr->GetInArgCount();
            ctx.funcManager = funcCallPtr->GetFuncManager();
            CHK_PTR_NULL(ctx.funcManager);
            ctx.funcBlock = funcCallPtr->GetFuncBlock();
            CHK_PTR_NULL(ctx.funcBlock);
            ctx.instr = funcCallPtr->GetInstr();
            CHK_PTR_NULL(ctx.instr);
            ctx.formalIns = ctx.funcBlock->GetInArgVars();
            if (static_cast<uint32_t>(ctx.formalIns.size()) != ctx.inArgCount) {
                HCCL_ERROR(
                    "FuncCall arg count mismatch: caller = %u, callee formal = %u", ctx.inArgCount,
                    static_cast<uint32_t>(ctx.formalIns.size()));
                return HCCL_E_PARA;
            }
            return HcclResult::HCCL_SUCCESS;
        }
    };
} // namespace CcuRep
} // namespace hcomm

#endif

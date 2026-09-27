/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_ins_generator_base.h"

namespace hcomm {
namespace CcuRep {

    // halfRtt
    HcclResult CcuInsGeneratorBase::CcuRepWriteVarAtomicTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepWriteVarAtomic* writeVarAtomPtr,
        const TransDep& dep)
    {
        (void)ccuKernel;
        (void)instr;
        (void)curInstrId;
        (void)dep;
        CHK_PTR_NULL(writeVarAtomPtr);
        HCCL_ERROR(
            "[CcuInsGeneratorBase][%s] unsupported rep type for this generator: %s", __func__,
            writeVarAtomPtr->Describe().c_str());
        return HCCL_E_NOT_SUPPORT;
    }

    HcclResult CcuInsGeneratorBase::CcuRepWriteWithCntIncTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepWriteWithCntInc* writeCntIncPtr,
        const TransDep& dep)
    {
        (void)ccuKernel;
        (void)instr;
        (void)curInstrId;
        (void)dep;
        CHK_PTR_NULL(writeCntIncPtr);
        HCCL_ERROR(
            "[CcuInsGeneratorBase][%s] unsupported rep type for this generator: %s", __func__,
            writeCntIncPtr->Describe().c_str());
        return HCCL_E_NOT_SUPPORT;
    }

    HcclResult CcuInsGeneratorBase::CcuRepCascCntWaitTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepCascCntWait* cascCntWaitPtr,
        const TransDep& dep)
    {
        (void)ccuKernel;
        (void)instr;
        (void)curInstrId;
        (void)dep;
        CHK_PTR_NULL(cascCntWaitPtr);
        HCCL_ERROR(
            "[CcuInsGeneratorBase][%s] unsupported rep type for this generator: %s", __func__,
            cascCntWaitPtr->Describe().c_str());
        return HCCL_E_NOT_SUPPORT;
    }

    HcclResult CcuInsGeneratorBase::CcuRepCascCntClearTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepCascCntClear* cascCntClearPtr,
        const TransDep& dep)
    {
        (void)ccuKernel;
        (void)instr;
        (void)curInstrId;
        (void)dep;
        CHK_PTR_NULL(cascCntClearPtr);
        HCCL_ERROR(
            "[CcuInsGeneratorBase][%s] unsupported rep type for this generator: %s", __func__,
            cascCntClearPtr->Describe().c_str());
        return HCCL_E_NOT_SUPPORT;
    }

    HcclResult CcuInsGeneratorBase::CcuRepLoadAddImmTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepLoadAddImm* loadAddImmPtr,
        const TransDep& dep)
    {
        (void)ccuKernel;
        (void)instr;
        (void)curInstrId;
        (void)dep;
        CHK_PTR_NULL(loadAddImmPtr);
        HCCL_ERROR(
            "[CcuInsGeneratorBase][%s] unsupported rep type for this generator: %s", __func__,
            loadAddImmPtr->Describe().c_str());
        return HCCL_E_NOT_SUPPORT;
    }

    HcclResult CcuInsGeneratorBase::CcuRepStoreAddImmTranslate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, CcuRepStoreAddImm* storeAddImmPtr,
        const TransDep& dep)
    {
        (void)ccuKernel;
        (void)instr;
        (void)curInstrId;
        (void)dep;
        CHK_PTR_NULL(storeAddImmPtr);
        HCCL_ERROR(
            "[CcuInsGeneratorBase][%s] unsupported rep type for this generator: %s", __func__,
            storeAddImmPtr->Describe().c_str());
        return HCCL_E_NOT_SUPPORT;
    }
} // namespace CcuRep
} // namespace hcomm

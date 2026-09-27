/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_rep_v1.h"
#include "ccu_ins_generator_v1.h"
#include "string_util.h"
#include "ccu_api_exception.h"
#include "ccu_ins_generator_base.h"
#include "ccu_kernel.h"

namespace hcomm {
namespace CcuRep {

    CcuRepBufLocWrite::CcuRepBufLocWrite(
        CcuInsGeneratorBase* insGenPtr, CcuBuf src, LocalAddr dst, Variable len, CompletedEvent sem, uint32_t mask)
        : insGenPtr(insGenPtr),
          src(src),
          dst(dst),
          len(len),
          sem(sem),
          mask(mask)
    {
        type = CcuRepType::BUF_LOC_WRITE;
        instrCount = insGenPtr->GetInstrCount(type);
    }

    bool CcuRepBufLocWrite::Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep)
    {
        this->instrId = instrId;
        translated = true;

        CHK_PRT_THROW(
            insGenPtr->CcuRepBufLocWriteTranslate(ccuKernel, instr, this, dep) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[CcuRepBufLocWrite][Translate] failed to translate for instrId[%u]", instrId),
            Hccl::CcuApiException, "CcuRepBufLocWrite translate failed");
        instrId += instrCount;

        return translated;
    }

    std::string CcuRepBufLocWrite::Describe()
    {
        return Hccl::StringFormat(
            "Write CcuBuf[%u] To Loc Mem[%u], len[%u], sem[%u], mask[%04x]", src.Id(), dst.addr.Id(), len.Id(),
            sem.Id(), mask);
    }

}; // namespace CcuRep
}; // namespace hcomm

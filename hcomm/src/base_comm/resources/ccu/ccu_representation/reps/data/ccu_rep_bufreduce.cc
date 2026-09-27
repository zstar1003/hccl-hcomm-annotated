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
#include "string_util.h"
#include "ccu_ins_generator_v1.h"
#include "ccu_api_exception.h"
#include "ccu_ins_generator_base.h"
#include "ccu_kernel.h"

namespace hcomm {
namespace CcuRep {

    CcuRepBufReduce::CcuRepBufReduce(
        CcuInsGeneratorBase* insGenPtr, const std::vector<CcuBuf>& mem, uint16_t count, uint16_t dataType,
        uint16_t outputDataType, uint16_t opType, CompletedEvent sem, const CcuRep::Variable& len, uint16_t mask)
        : insGenPtr(insGenPtr),
          mem(mem),
          count(count),
          dataType(dataType),
          outputDataType(outputDataType),
          opType(opType),
          sem(sem),
          xnIdLength_(len),
          mask(mask)
    {
        type = CcuRepType::BUF_REDUCE;
        instrCount = insGenPtr->GetInstrCount(type);
    }

    bool CcuRepBufReduce::Translate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, [[maybe_unused]] const TransDep& dep)
    {
        this->instrId = instrId;
        translated = true;

        instrCount = insGenPtr->GetInstrCount(type);
        CHK_PRT_THROW(
            insGenPtr->CcuRepBufReduceTranslate(ccuKernel, instr, this) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[CcuRepBufReduce][Translate] failed to translate for instrId[%u]", instrId),
            Hccl::CcuApiException, "CcuRepBufReduce translate failed");

        instrId += instrCount;

        return translated;
    }

    std::string CcuRepBufReduce::Describe() { return Hccl::StringFormat("Reduce"); }

}; // namespace CcuRep
}; // namespace hcomm

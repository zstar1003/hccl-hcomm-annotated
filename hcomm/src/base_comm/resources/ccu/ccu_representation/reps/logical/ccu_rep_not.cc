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
#include "exception_util.h"
#include "ccu_api_exception.h"
#include "ccu_ins_generator_v1.h"
#include "ccu_kernel.h"
namespace hcomm {
namespace CcuRep {

    CcuRepNot::CcuRepNot(CcuInsGeneratorBase* insGenPtr, const Variable& varC, const Variable& varB)
        : subType(NotSubType::VAR_EQUALS_NOT_VAR),
          varB(varB),
          varC(varC),
          insGenPtr(insGenPtr)
    {
        type = CcuRepType::NOT;
        instrCount = insGenPtr->GetInstrCount(type);
    }

    bool CcuRepNot::Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, const TransDep& dep)
    {
        Hccl::CHECK_NULLPTR(instr, "[CcuRepNot::Translate] instr is nullptr!");
        this->instrId = curInstrId;
        translated = true;
        instrCount = insGenPtr->GetInstrCount(type);
        CHK_PRT_THROW(
            insGenPtr->CcuRepNotTranslate(ccuKernel, instr, this, dep) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[CcuRepNot][Translate] failed to translate for instrId[%u]", instrId), Hccl::CcuApiException,
            "CcuRepNot translate failed");
        CHK_PRT_THROW(
            (curInstrId > UINT16_MAX - instrCount),
            HCCL_ERROR(
                "[CcuRepNot::Translate]uint16 integer overflow occurs, curInstrId = [%hu], instrCount = [%hu]",
                curInstrId, instrCount),
            Hccl::InternalException, "integer overflow");
        curInstrId += instrCount;
        return translated;
    }

    std::string CcuRepNot::Describe()
    {
        switch (subType) {
            case NotSubType::VAR_EQUALS_NOT_VAR: {
                return Hccl::StringFormat("Variable[%u] = ~Variable[%u]", varC.Id(), varB.Id());
            }
            default: {
                return Hccl::StringFormat("Invalid Not");
            }
        }
    }

}; // namespace CcuRep
}; // namespace hcomm

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

    CcuRepOr::CcuRepOr(CcuInsGeneratorBase* insGenPtr, const Variable& varC, const Variable& varA, const Variable& varB)
        : subType(OrSubType::VAR_OR_VAR_TO_VAR),
          varA(varA),
          varB(varB),
          varC(varC),
          insGenPtr(insGenPtr)
    {
        type = CcuRepType::OR;
        instrCount = insGenPtr->GetInstrCount(type);
    }

    CcuRepOr::CcuRepOr(CcuInsGeneratorBase* insGenPtr, const Variable& varC, const Variable& varB)
        : subType(OrSubType::SELF_OR_VAR_VARIABLE),
          varB(varB),
          varC(varC),
          insGenPtr(insGenPtr)
    {
        type = CcuRepType::OR;
        instrCount = insGenPtr->GetInstrCount(type);
    }

    bool CcuRepOr::Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, const TransDep& dep)
    {
        Hccl::CHECK_NULLPTR(instr, "[CcuRepOr::Translate] instr is nullptr!");
        this->instrId = curInstrId;
        translated = true;
        instrCount = insGenPtr->GetInstrCount(type);
        CHK_PRT_THROW(
            insGenPtr->CcuRepOrTranslate(ccuKernel, instr, this, dep) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[CcuRepOr][Translate] failed to translate for instrId[%u]", instrId), Hccl::CcuApiException,
            "CcuRepOr translate failed");
        CHK_PRT_THROW(
            (curInstrId > UINT16_MAX - instrCount),
            HCCL_ERROR(
                "[CcuRepOr::Translate]uint16 integer overflow occurs, curInstrId = [%hu], instrCount = [%hu]",
                curInstrId, instrCount),
            Hccl::InternalException, "integer overflow");
        curInstrId += instrCount;
        return translated;
    }

    std::string CcuRepOr::Describe()
    {
        switch (subType) {
            case OrSubType::VAR_OR_VAR_TO_VAR: {
                return Hccl::StringFormat(
                    "Variable[%u] = Variable[%u] | Variable[%u]", varC.Id(), varA.Id(), varB.Id());
            }
            case OrSubType::SELF_OR_VAR_VARIABLE: {
                return Hccl::StringFormat("Variable[%u] |= Variable[%u]", varC.Id(), varB.Id());
            }
            default: {
                return Hccl::StringFormat("Invalid Or");
            }
        }
    }

}; // namespace CcuRep
}; // namespace hcomm

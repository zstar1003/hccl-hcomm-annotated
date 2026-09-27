/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "string_util.h"
#include "exception_util.h"
#include "ccu_api_exception.h"
#include "ccu_rep_v1.h"
#include "ccu_ins_generator_v1.h"
#include "ccu_kernel.h"
namespace hcomm {
namespace CcuRep {

    CcuRepShR::CcuRepShR(
        CcuInsGeneratorBase* insGenPtr, const Variable& varD, const Variable& varN, const Variable& varM)
        : subType(ShiftSubType::VAR_EQUALS_VAR_SHIFT_VAR),
          shiftType(ShiftType::LOGICAL_SHIFT),
          varN(varN),
          varM(varM),
          varD(varD),
          insGenPtr(insGenPtr)
    {
        type = CcuRepType::SHR;
        instrCount = insGenPtr->GetInstrCount(type);
    }

    CcuRepShR::CcuRepShR(CcuInsGeneratorBase* insGenPtr, const Variable& varD, const Variable& varM)
        : subType(ShiftSubType::VAR_SHIFT_ASSIGN_VAR),
          shiftType(ShiftType::LOGICAL_SHIFT),
          varM(varM),
          varD(varD),
          insGenPtr(insGenPtr)
    {
        type = CcuRepType::SHR;
        instrCount = insGenPtr->GetInstrCount(type);
    }

    CcuRepShR::CcuRepShR(
        CcuInsGeneratorBase* insGenPtr, const Address& addrD, const Variable& varN, const Variable& varM)
        : subType(ShiftSubType::ADDR_EQUALS_VAR_SHIFT_VAR),
          shiftType(ShiftType::LOGICAL_SHIFT),
          varN(varN),
          varM(varM),
          addrD(addrD),
          insGenPtr(insGenPtr)
    {
        type = CcuRepType::SHR;
        instrCount = insGenPtr->GetInstrCount(type);
    }

    CcuRepShR::CcuRepShR(CcuInsGeneratorBase* insGenPtr, const Address& addrD, const Variable& varM)
        : subType(ShiftSubType::ADDR_SHIFT_ASSIGN_VAR),
          shiftType(ShiftType::LOGICAL_SHIFT),
          varM(varM),
          addrD(addrD),
          insGenPtr(insGenPtr)
    {
        type = CcuRepType::SHR;
        instrCount = insGenPtr->GetInstrCount(type);
    }

    bool CcuRepShR::Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, const TransDep& dep)
    {
        Hccl::CHECK_NULLPTR(instr, "[CcuRepShR::Translate] instr is nullptr!");
        this->instrId = curInstrId;
        translated = true;
        instrCount = insGenPtr->GetInstrCount(type);
        CHK_PRT_THROW(
            insGenPtr->CcuRepShRTranslate(ccuKernel, instr, this, dep) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[CcuRepShR][Translate] failed to translate for instrId[%u]", instrId), Hccl::CcuApiException,
            "CcuRepShR translate failed");
        CHK_PRT_THROW(
            (curInstrId > UINT16_MAX - instrCount),
            HCCL_ERROR(
                "[CcuRepShR::Translate]uint16 integer overflow occurs, curInstrId = [%hu], instrCount = [%hu]",
                curInstrId, instrCount),
            Hccl::InternalException, "integer overflow");
        curInstrId += instrCount;
        return translated;
    }

    std::string CcuRepShR::Describe()
    {
        if (shiftType == ShiftType::LOGICAL_SHIFT) {
            switch (subType) {
                case ShiftSubType::VAR_EQUALS_VAR_SHIFT_VAR: {
                    return Hccl::StringFormat(
                        "Variable[%u] = Variable[%u] >> Variable[%u]", varD.Id(), varN.Id(), varM.Id());
                }
                case ShiftSubType::VAR_SHIFT_ASSIGN_VAR: {
                    return Hccl::StringFormat("Variable[%u] >>= Variable[%u]", varD.Id(), varM.Id());
                }
                case ShiftSubType::ADDR_EQUALS_VAR_SHIFT_VAR: {
                    return Hccl::StringFormat(
                        "Address[%u] = Variable[%u] >> Variable[%u]", addrD.Id(), varN.Id(), varM.Id());
                }
                case ShiftSubType::ADDR_SHIFT_ASSIGN_VAR: {
                    return Hccl::StringFormat("Address[%u] >>= Variable[%u]", addrD.Id(), varM.Id());
                }
                default: {
                    return Hccl::StringFormat("Invalid Shift");
                }
            }
        }
        return Hccl::StringFormat("Invalid Shift");
    }

}; // namespace CcuRep
}; // namespace hcomm

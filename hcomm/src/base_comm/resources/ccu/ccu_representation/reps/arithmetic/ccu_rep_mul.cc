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
#include "exception_util.h"
#include "ccu_api_exception.h"
#include "ccu_kernel.h"

namespace hcomm {
namespace CcuRep {
    void CcuRepMul::SetCommonInfo()
    {
        type = CcuRepType::MUL;
        instrCount = insGenPtr->GetInstrCount(type);
    }

    CcuRepMul::CcuRepMul(
        CcuInsGeneratorBase* insGenPtr, const Variable& varC, const Variable& varA, const Variable& varB)
        : insGenPtr(insGenPtr),
          subType(MulSubType::VAR_MUL_VAR_TO_VAR),
          varA(varA),
          varB(varB),
          varC(varC)
    {
        SetCommonInfo();
    }

    CcuRepMul::CcuRepMul(CcuInsGeneratorBase* insGenPtr, const Variable& varC, const Variable& varA, uint16_t immedB)
        : insGenPtr(insGenPtr),
          subType(MulSubType::VAR_MUL_IMMED_TO_VAR),
          varA(varA),
          varC(varC),
          immedB(immedB)
    {
        SetCommonInfo();
    }

    CcuRepMul::CcuRepMul(CcuInsGeneratorBase* insGenPtr, const Variable& varA, const Variable& varB)
        : insGenPtr(insGenPtr),
          subType(MulSubType::SELF_MUL_VAR_VARIABLE),
          varA(varA),
          varB(varB)
    {
        SetCommonInfo();
    }

    CcuRepMul::CcuRepMul(CcuInsGeneratorBase* insGenPtr, const Variable& varA, const uint16_t immedB)
        : insGenPtr(insGenPtr),
          subType(MulSubType::SELF_MUL_IMMED_VARIABLE),
          varA(varA),
          immedB(immedB)
    {
        SetCommonInfo();
    }

    CcuRepMul::CcuRepMul(
        CcuInsGeneratorBase* insGenPtr, const Address& addrC, const Variable& varA, const Variable& varB)
        : insGenPtr(insGenPtr),
          subType(MulSubType::VAR_MUL_VAR_TO_ADDR),
          varA(varA),
          varB(varB),
          addrC(addrC)
    {
        SetCommonInfo();
    }

    CcuRepMul::CcuRepMul(
        CcuInsGeneratorBase* insGenPtr, const Address& addrC, const Variable& varA, const Address& addrB)
        : insGenPtr(insGenPtr),
          subType(MulSubType::VAR_MUL_ADDR_TO_ADDR),
          varA(varA),
          addrB(addrB),
          addrC(addrC)
    {
        SetCommonInfo();
    }

    CcuRepMul::CcuRepMul(CcuInsGeneratorBase* insGenPtr, const Address& addrA, const Variable& varB)
        : insGenPtr(insGenPtr),
          subType(MulSubType::SELF_MUL_VAR_ADDRESS),
          varB(varB),
          addrA(addrA)
    {
        SetCommonInfo();
    }

    CcuRepMul::CcuRepMul(
        CcuInsGeneratorBase* insGenPtr, const Address& addrC, const Address& addrA, const uint16_t immedB)
        : insGenPtr(insGenPtr),
          subType(MulSubType::ADDR_MUL_IMMED_TO_ADDR),
          addrA(addrA),
          addrC(addrC),
          immedB(immedB)
    {
        SetCommonInfo();
    }

    CcuRepMul::CcuRepMul(
        CcuInsGeneratorBase* insGenPtr, const Address& addrC, const Variable& varA, const uint16_t immedB)
        : insGenPtr(insGenPtr),
          subType(MulSubType::VAR_MUL_IMMED_TO_ADDR),
          varA(varA),
          addrC(addrC),
          immedB(immedB)
    {
        SetCommonInfo();
    }

    CcuRepMul::CcuRepMul(CcuInsGeneratorBase* insGenPtr, const Address& addrA, const uint16_t immedB)
        : insGenPtr(insGenPtr),
          subType(MulSubType::SELF_MUL_IMMED_ADDRESS),
          addrA(addrA),
          immedB(immedB)
    {
        SetCommonInfo();
    }

    CcuRepMul::CcuRepMul(
        CcuInsGeneratorBase* insGenPtr, const Variable& varC, const Address& addrA, const uint16_t immedB)
        : insGenPtr(insGenPtr),
          subType(MulSubType::ADDR_MUL_IMMED_TO_VAR),
          varC(varC),
          addrA(addrA),
          immedB(immedB)
    {
        SetCommonInfo();
    }

    void CcuRepMul::ValidateInsGenPtrForMul() const
    {
        CcuInsGeneratorV1* tmpPtrV1 = dynamic_cast<CcuInsGeneratorV1*>(insGenPtr);
        CHK_PRT_THROW(
            (tmpPtrV1 && !supportCcuV1),
            HCCL_ERROR("[CcuRepMul][%s]Cannot translate CcuRepMul for A5 when supportCcuV1 is false", __func__),
            Hccl::CcuApiException, "tmpPtrV1 does not match supportCcuV1");
    }

    bool CcuRepMul::Translate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, [[maybe_unused]] const TransDep& dep)
    {
        ValidateInsGenPtrForMul();
        Hccl::CHECK_NULLPTR(instr, "[CcuRepMul::Translate] instr is nullptr!");
        this->instrId = instrId;
        translated = true;

        CHK_PRT_THROW(
            insGenPtr->CcuRepMulTranslate(ccuKernel, instr, this) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[CcuRepMul][Translate] failed to translate for instrId[%u]", instrId), Hccl::CcuApiException,
            "CcuRepMul translate failed");

        CHK_PRT_THROW(
            (instrId > UINT16_MAX - instrCount),
            HCCL_ERROR(
                "[CcuRepMul::Translate]uint16 integer overflow occurs, instrId = [%hu], instrCount = [%hu]", instrId,
                instrCount),
            Hccl::InternalException, "integer overflow");
        instrId += instrCount;

        return translated;
    }

    std::string CcuRepMul::Describe()
    {
        switch (subType) {
            case MulSubType::VAR_MUL_VAR_TO_VAR: {
                return Hccl::StringFormat(
                    "Variable[%u] = Variable[%u] * Variable[%u]", varC.Id(), varA.Id(), varB.Id());
            }
            case MulSubType::VAR_MUL_IMMED_TO_VAR: {
                return Hccl::StringFormat("Variable[%u] = Variable[%u] * Immed[%u]", varC.Id(), varA.Id(), immedB);
            }
            case MulSubType::SELF_MUL_VAR_VARIABLE: {
                return Hccl::StringFormat("Variable[%u] *= Variable[%u]", varA.Id(), varB.Id());
            }
            case MulSubType::SELF_MUL_IMMED_VARIABLE: {
                return Hccl::StringFormat("Variable[%u] *= Immed[%u]", varA.Id(), immedB);
            }
            case MulSubType::VAR_MUL_VAR_TO_ADDR: {
                return Hccl::StringFormat(
                    "Address[%u] = Variable[%u] * Variable[%u]", addrC.Id(), varA.Id(), varB.Id());
            }
            case MulSubType::VAR_MUL_ADDR_TO_ADDR: {
                return Hccl::StringFormat(
                    "Address[%u] = Variable[%u] * Address[%u]", addrC.Id(), varA.Id(), addrB.Id());
            }
            case MulSubType::VAR_MUL_IMMED_TO_ADDR: {
                return Hccl::StringFormat("address[%u] = Variable[%u] * Immed[%u]", addrC.Id(), varA.Id(), immedB);
            }
            case MulSubType::ADDR_MUL_IMMED_TO_ADDR: {
                return Hccl::StringFormat("address[%u] = address[%u] * Immed[%u]", addrC.Id(), addrA.Id(), immedB);
            }
            case MulSubType::SELF_MUL_VAR_ADDRESS: {
                return Hccl::StringFormat("address[%u] *= Variable[%u]", addrA.Id(), varB.Id());
            }
            case MulSubType::SELF_MUL_IMMED_ADDRESS: {
                return Hccl::StringFormat("address[%u] *= Immed[%u]", addrA.Id(), immedB);
            }
            case MulSubType::ADDR_MUL_IMMED_TO_VAR: {
                return Hccl::StringFormat("Variable[%u] = address[%u] * Immed[%u]", varC.Id(), addrA.Id(), immedB);
            }
            default: {
                return Hccl::StringFormat("Invalid Mul");
            }
        }
    }

    Address CcuRepMul::GetAddrA() { return addrA; }

    Address CcuRepMul::GetAddrB() { return addrB; }

    Address CcuRepMul::GetAddrC() { return addrC; }

    Variable CcuRepMul::GetVarA() { return varA; }

    Variable CcuRepMul::GetVarB() { return varB; }

    Variable CcuRepMul::GetVarC() { return varC; }

    uint16_t CcuRepMul::GetImmedB() const { return immedB; }

    MulSubType CcuRepMul::GetSubType() const { return subType; }
}; // namespace CcuRep
}; // namespace hcomm

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_rep.h"

#include "string_util.h"
#include "exception_util.h"
#include "ccu_api_exception.h"

namespace Hccl {
namespace CcuRep {

    CcuRepAssign::CcuRepAssign(const Variable& varA, uint64_t immediate)
        : subType(AssignSubType::IMD_TO_VARIABLE),
          immediate(immediate),
          varA(varA)
    {
        type = CcuRepType::ASSIGN;
        instrCount = 1;
    }

    CcuRepAssign::CcuRepAssign(const Address& addrA, uint64_t immediate)
        : subType(AssignSubType::IMD_TO_ADDR),
          immediate(immediate),
          addrA(addrA)
    {
        type = CcuRepType::ASSIGN;
        instrCount = 1;
    }

    CcuRepAssign::CcuRepAssign(const Address& addrA, const Variable& varA)
        : subType(AssignSubType::VAR_TO_ADDR),
          immediate(0),
          varA(varA),
          addrA(addrA)
    {
        type = CcuRepType::ASSIGN;
        instrCount = 1;
    }

    CcuRepAssign::CcuRepAssign(const Address& addrB, const Address& addrA)
        : subType(AssignSubType::ADDR_TO_ADDR),
          immediate(0),
          addrA(addrA),
          addrB(addrB)
    {
        type = CcuRepType::ASSIGN;
        instrCount = 1;
    }

    CcuRepAssign::CcuRepAssign(const Variable& varB, const Variable& varA)
        : subType(AssignSubType::VAR_TO_VAR),
          immediate(0),
          varA(varA),
          varB(varB)
    {
        type = CcuRepType::ASSIGN;
        instrCount = 1;
    }

    bool CcuRepAssign::Translate(CcuInstr*& instr, uint16_t& instrId, const TransDep& dep)
    {
        CHECK_NULLPTR(instr, "[CcuRepAssign::Translate] instr is nullptr!");
        this->instrId = instrId;
        translated = true;

        switch (subType) {
            case AssignSubType::IMD_TO_VARIABLE: {
                LoadImdToXnInstr(instr++, varA.Id(), immediate);
                break;
            }
            case AssignSubType::IMD_TO_ADDR: {
                LoadImdToGSAInstr(instr++, addrA.Id(), immediate);
                break;
            }
            case AssignSubType::VAR_TO_ADDR: {
                LoadGSAXnInstr(instr++, addrA.Id(), dep.reserveGsaId, varA.Id());
                break;
            }
            case AssignSubType::ADDR_TO_ADDR: {
                LoadGSAGSAInstr(instr++, addrB.Id(), addrA.Id(), dep.reserveGsaId);
                break;
            }
            case AssignSubType::VAR_TO_VAR: {
                LoadXXInstr(instr++, varB.Id(), varA.Id(), dep.reserveXnId);
                break;
            }
            default: {
                THROW<CcuApiException>("Invalid Assign");
            }
        }

        CHK_PRT_THROW(
            (instrId > UINT16_MAX - instrCount),
            HCCL_ERROR(
                "[CcuRepAssign::Translate]uint16 integer overflow occurs, instrId = [%hu], instrCount = [%hu]", instrId,
                instrCount),
            InternalException, "integer overflow");
        instrId += instrCount;

        return translated;
    }

    std::string CcuRepAssign::Describe()
    {
        switch (subType) {
            case AssignSubType::IMD_TO_VARIABLE: {
                return StringFormat("Variable[%u] = Value[%lu]", varA.Id(), immediate);
            }
            case AssignSubType::IMD_TO_ADDR: {
                return StringFormat("Address[%u] = Value[%lu]", addrA.Id(), immediate);
            }
            case AssignSubType::VAR_TO_ADDR: {
                return StringFormat("Address[%u] = Variable[%u]", addrA.Id(), varA.Id());
            }
            case AssignSubType::ADDR_TO_ADDR: {
                return StringFormat("Address[%u] = Address[%u]", addrB.Id(), addrA.Id());
            }
            case AssignSubType::VAR_TO_VAR: {
                return StringFormat("Var[%u] = Var[%u]", varB.Id(), varA.Id());
            }
            default: {
                return StringFormat("Invalid Assign");
            }
        }
    }

}; // namespace CcuRep
}; // namespace Hccl

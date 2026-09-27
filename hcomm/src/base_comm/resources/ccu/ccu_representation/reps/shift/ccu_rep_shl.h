/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_REPRESENTATION_SHL_H
#define HCCL_CCU_REPRESENTATION_SHL_H

#include "ccu_rep_base_v1.h"
#include "ccu_datatype_v1.h"

namespace hcomm {
namespace CcuRep {

    class CcuRepShL : public CcuRepBase {
    public:
        explicit CcuRepShL(
            CcuInsGeneratorBase* insGenPtr, const Variable& varD, const Variable& varN, const Variable& varM);
        explicit CcuRepShL(CcuInsGeneratorBase* insGenPtr, const Variable& varD, const Variable& varM);
        explicit CcuRepShL(
            CcuInsGeneratorBase* insGenPtr, const Address& addrD, const Variable& varN, const Variable& varM);
        explicit CcuRepShL(CcuInsGeneratorBase* insGenPtr, const Address& addrD, const Variable& varM);
        bool Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, const TransDep& dep) override;

        std::string Describe() override;

    private:
        ShiftSubType subType{ShiftSubType::INVALID};
        ShiftType shiftType{ShiftType::INVALID};
        Variable varN;
        Variable varM;
        Variable varD;
        Address addrD;

        CcuInsGeneratorBase* insGenPtr{nullptr};

    public:
        ShiftSubType GetShiftSubType() const { return subType; }
        ShiftType GetShiftType() const { return shiftType; }
        Variable GetVarN() { return varN; }
        Variable GetVarM() { return varM; }
        Variable GetVarD() { return varD; }
        Address GetAddressD() { return addrD; }
    };

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCCL_CCU_REPRESENTATION_SHL_H

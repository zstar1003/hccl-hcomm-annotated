/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_REPRESENTATION_OR_H
#define HCCL_CCU_REPRESENTATION_OR_H

#include "ccu_rep_base_v1.h"
#include "ccu_datatype_v1.h"

namespace hcomm {
namespace CcuRep {

    class CcuRepOr : public CcuRepBase {
    public:
        explicit CcuRepOr(
            CcuInsGeneratorBase* insGenPtr, const Variable& varC, const Variable& varA, const Variable& varB);
        explicit CcuRepOr(CcuInsGeneratorBase* insGenPtr, const Variable& varC, const Variable& varB);
        bool Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, const TransDep& dep) override;

        std::string Describe() override;

        Variable GetVarA() { return varA; }
        Variable GetVarB() { return varB; }
        Variable GetVarC() { return varC; }
        OrSubType GetSubType() const { return subType; }

    private:
        OrSubType subType{OrSubType::INVALID};

        Variable varA;
        Variable varB;
        Variable varC;

        CcuInsGeneratorBase* insGenPtr{nullptr};
    };

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCCL_CCU_REPRESENTATION_OR_H

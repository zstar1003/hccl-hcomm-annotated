/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_REPRESENTATION_ASSIGN_H
#define HCOMM_CCU_REPRESENTATION_ASSIGN_H

#include "ccu_rep_base_v1.h"
#include "ccu_datatype_v1.h"

namespace hcomm {
namespace CcuRep {

    class CcuRepAssign : public CcuRepBase {
    public:
        explicit CcuRepAssign(CcuInsGeneratorBase* insGenPtr, const Variable& varA, uint64_t immediate);
        explicit CcuRepAssign(CcuInsGeneratorBase* insGenPtr, const Address& addrA, uint64_t immediate);
        explicit CcuRepAssign(CcuInsGeneratorBase* insGenPtr, const Address& addrA, const Variable& varA);
        explicit CcuRepAssign(CcuInsGeneratorBase* insGenPtr, const Address& addrB, const Address& addrA);
        explicit CcuRepAssign(CcuInsGeneratorBase* insGenPtr, const Variable& varB, const Variable& varA);

        bool Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;
        Address GetAddrA();
        Address GetAddrB();
        Variable GetVarA();
        Variable GetVarB();
        uint64_t GetImmed() const;
        AssignSubType GetSubType() const;

    private:
        void SetCommonInfo();
        CcuInsGeneratorBase* insGenPtr{nullptr};
        AssignSubType subType{AssignSubType::INVALID};
        uint64_t immediate{0};

        Variable varA;
        Variable varB;

        Address addrA;
        Address addrB;
    };

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCCL_CCU_REPRESENTATION_ASSIGN_H

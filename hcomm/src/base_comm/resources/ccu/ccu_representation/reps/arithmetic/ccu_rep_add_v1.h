/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_REPRESENTATION_ADD_H
#define HCOMM_CCU_REPRESENTATION_ADD_H

#include "ccu_rep_base_v1.h"
#include "ccu_datatype_v1.h"

namespace hcomm {
namespace CcuRep {

    class CcuRepAdd : public CcuRepBase {
    public:
        // support ccuV1 & ccuV2
        explicit CcuRepAdd(
            CcuInsGeneratorBase* insGenPtr, const Address& addrC, const Address& addrA, const Variable& varB);
        explicit CcuRepAdd(
            CcuInsGeneratorBase* insGenPtr, const Address& addrC, const Address& addrA, const Address& addrB);
        explicit CcuRepAdd(
            CcuInsGeneratorBase* insGenPtr, const Variable& varC, const Variable& varA, const Variable& varB);
        explicit CcuRepAdd(CcuInsGeneratorBase* insGenPtr, const Address& addrA, const Variable& offset);
        explicit CcuRepAdd(CcuInsGeneratorBase* insGenPtr, const Variable& varA, const Variable& offset);

        // only support ccuV2
        explicit CcuRepAdd(
            CcuInsGeneratorBase* insGenPtr, const Variable& varC, const Variable& varA, const uint16_t immedB);
        explicit CcuRepAdd(CcuInsGeneratorBase* insGenPtr, const Variable& varA, const uint16_t immedB);
        explicit CcuRepAdd(
            CcuInsGeneratorBase* insGenPtr, const Address& addrC, const Address& addrA, const uint16_t immedB);
        explicit CcuRepAdd(CcuInsGeneratorBase* insGenPtr, const Address& addrA, const uint16_t immedB);
        explicit CcuRepAdd(
            CcuInsGeneratorBase* insGenPtr, const Address& addrC, const Variable& varA, const uint16_t immedB);
        explicit CcuRepAdd(
            CcuInsGeneratorBase* insGenPtr, const Variable& varC, const Address& addrA, const uint16_t immedB);
        explicit CcuRepAdd(
            CcuInsGeneratorBase* insGenPtr, const Variable& varC, const Address& addrA, const Address& addrB);
        explicit CcuRepAdd(
            CcuInsGeneratorBase* insGenPtr, const Address& addrC, const Variable& varA, const Variable& varB);

        bool Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;

        Address GetAddrA();
        Address GetAddrB();
        Address GetAddrC();
        Variable GetVarA();
        Variable GetVarB();
        Variable GetVarC();
        uint16_t GetImmedB() const;
        AddSubType GetSubType() const;

    private:
        void ValidateInsGenPtrForAdd() const;
        void SetCommonInfo();

        CcuInsGeneratorBase* insGenPtr{nullptr};
        AddSubType subType{AddSubType::INVALID};

        Address addrA;
        Address addrB;
        Address addrC;

        Variable varA;
        Variable varB;
        Variable varC;

        uint16_t immedB{0};

        bool supportCcuV1{false};
    };

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCCL_CCU_REPRESENTATION_ADD_H

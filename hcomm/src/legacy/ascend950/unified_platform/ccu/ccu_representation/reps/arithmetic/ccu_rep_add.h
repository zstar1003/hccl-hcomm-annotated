/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_REPRESENTATION_ADD_H
#define HCCL_CCU_REPRESENTATION_ADD_H

#include "ccu_rep_base.h"
#include "ccu_datatype.h"

namespace Hccl {
namespace CcuRep {

    class CcuRepAdd : public CcuRepBase {
    public:
        explicit CcuRepAdd(const Address& addrC, const Address& addrA, const Variable& varB);
        explicit CcuRepAdd(const Address& addrC, const Address& addrA, const Address& addrB);
        explicit CcuRepAdd(const Variable& varC, const Variable& varA, const Variable& varB);
        explicit CcuRepAdd(const Address& addrA, const Variable& offset);
        explicit CcuRepAdd(const Variable& varA, const Variable& offset);
        bool Translate(CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;

    private:
        void SetCommonInfo();
        AddSubType subType{AddSubType::INVALID};

        Address addrA;
        Address addrB;
        Address addrC;

        Variable varA;
        Variable varB;
        Variable varC;
    };

}; // namespace CcuRep
}; // namespace Hccl
#endif // HCCL_CCU_REPRESENTATION_ADD_H

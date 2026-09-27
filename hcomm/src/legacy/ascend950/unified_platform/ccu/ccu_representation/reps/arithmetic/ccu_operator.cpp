/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_operator.h"
#include "ccu_datatype.h"

namespace Hccl {
namespace CcuRep {

    template <>
    void CcuArithmeticOperator<Variable, Variable>::Check() const
    {
        // nothing
    }
    template <>
    void CcuArithmeticOperator<Variable, Address>::Check() const
    {
        // nothing
    }
    template <>
    void CcuArithmeticOperator<Address, Address>::Check() const
    {
        // nothing
    }

    template <>
    void CcuRelationalOperator<Variable, uint64_t>::Check() const
    {
        // nothing
    }

    CcuArithmeticOperator<Variable, Variable> Variable::operator+(const Variable& varB) const
    {
        return CcuArithmeticOperator<Variable, Variable>(*this, varB, CcuArithmeticOperatorType::ADDITION);
    }
    CcuArithmeticOperator<Variable, Address> Variable::operator+(const Address& addrB) const
    {
        return CcuArithmeticOperator<Variable, Address>(*this, addrB, CcuArithmeticOperatorType::ADDITION);
    }
    CcuArithmeticOperator<Variable, Address> Address::operator+(const Variable& varB) const
    {
        return CcuArithmeticOperator<Variable, Address>(varB, *this, CcuArithmeticOperatorType::ADDITION);
    }
    CcuArithmeticOperator<Address, Address> Address::operator+(const Address& addrB) const
    {
        return CcuArithmeticOperator<Address, Address>(*this, addrB, CcuArithmeticOperatorType::ADDITION);
    }

    CcuRelationalOperator<Variable, uint64_t> Variable::operator!=(uint64_t immediate) const
    {
        return CcuRelationalOperator<Variable, uint64_t>(*this, immediate, CcuRelationalOperatorType::NOT_EQUAL);
    }

    CcuRelationalOperator<Variable, uint64_t> Variable::operator==(uint64_t immediate) const
    {
        return CcuRelationalOperator<Variable, uint64_t>(*this, immediate, CcuRelationalOperatorType::EQUAL);
    }

}; // namespace CcuRep
}; // namespace Hccl

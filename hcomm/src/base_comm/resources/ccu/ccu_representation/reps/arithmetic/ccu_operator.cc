/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_operator_v1.h"
#include "ccu_datatype_v1.h"

namespace hcomm {
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
    void CcuArithmeticOperator<Variable, uint16_t>::Check() const
    {
        // nothing
    }
    template <>
    void CcuArithmeticOperator<Address, Address>::Check() const
    {
        // nothing
    }
    template <>
    void CcuArithmeticOperator<Address, uint16_t>::Check() const
    {
        // nothing
    }

    template <>
    void CcuRelationalOperator<Variable, uint64_t>::Check() const
    {
        // nothing
    }
    template <>
    void CcuRelationalOperator<Variable, Variable>::Check() const
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

    CcuArithmeticOperator<Variable, uint16_t> Variable::operator+(const uint16_t offset) const
    {
        return CcuArithmeticOperator<Variable, uint16_t>(*this, offset, CcuArithmeticOperatorType::ADDITION);
    }

    CcuArithmeticOperator<Variable, Variable> Variable::operator*(const Variable& varB) const
    {
        return CcuArithmeticOperator<Variable, Variable>(*this, varB, CcuArithmeticOperatorType::MULTIPLICATION);
    }
    CcuArithmeticOperator<Variable, Address> Variable::operator*(const Address& addrB) const
    {
        return CcuArithmeticOperator<Variable, Address>(*this, addrB, CcuArithmeticOperatorType::MULTIPLICATION);
    }
    CcuArithmeticOperator<Variable, uint16_t> Variable::operator*(const uint16_t offset) const
    {
        return CcuArithmeticOperator<Variable, uint16_t>(*this, offset, CcuArithmeticOperatorType::MULTIPLICATION);
    }

    CcuArithmeticOperator<Variable, Variable> Variable::operator-(const Variable& varB) const
    {
        return CcuArithmeticOperator<Variable, Variable>(*this, varB, CcuArithmeticOperatorType::SUBTRACTION);
    }
    CcuArithmeticOperator<Variable, Address> Variable::operator-(const Address& addrB) const
    {
        return CcuArithmeticOperator<Variable, Address>(*this, addrB, CcuArithmeticOperatorType::SUBTRACTION);
    }
    CcuArithmeticOperator<Variable, uint16_t> Variable::operator-(const uint16_t offset) const
    {
        return CcuArithmeticOperator<Variable, uint16_t>(*this, offset, CcuArithmeticOperatorType::SUBTRACTION);
    }

    CcuArithmeticOperator<Variable, Address> Address::operator+(const Variable& varB) const
    {
        return CcuArithmeticOperator<Variable, Address>(varB, *this, CcuArithmeticOperatorType::ADDITION);
    }
    CcuArithmeticOperator<Address, Address> Address::operator+(const Address& addrB) const
    {
        return CcuArithmeticOperator<Address, Address>(*this, addrB, CcuArithmeticOperatorType::ADDITION);
    }

    CcuArithmeticOperator<Address, uint16_t> Address::operator+(const uint16_t offset) const
    {
        return CcuArithmeticOperator<Address, uint16_t>(*this, offset, CcuArithmeticOperatorType::ADDITION);
    }

    CcuArithmeticOperator<Variable, Address> Address::operator*(const Variable& varB) const
    {
        return CcuArithmeticOperator<Variable, Address>(varB, *this, CcuArithmeticOperatorType::MULTIPLICATION);
    }
    CcuArithmeticOperator<Address, Address> Address::operator*(const Address& addrB) const
    {
        return CcuArithmeticOperator<Address, Address>(*this, addrB, CcuArithmeticOperatorType::MULTIPLICATION);
    }
    CcuArithmeticOperator<Address, uint16_t> Address::operator*(const uint16_t offset) const
    {
        return CcuArithmeticOperator<Address, uint16_t>(*this, offset, CcuArithmeticOperatorType::MULTIPLICATION);
    }

    CcuArithmeticOperator<Variable, Address> Address::operator-(const Variable& varB) const
    {
        return CcuArithmeticOperator<Variable, Address>(varB, *this, CcuArithmeticOperatorType::SUBTRACTION);
    }
    CcuArithmeticOperator<Address, Address> Address::operator-(const Address& addrB) const
    {
        return CcuArithmeticOperator<Address, Address>(*this, addrB, CcuArithmeticOperatorType::SUBTRACTION);
    }
    CcuArithmeticOperator<Address, uint16_t> Address::operator-(const uint16_t offset) const
    {
        return CcuArithmeticOperator<Address, uint16_t>(*this, offset, CcuArithmeticOperatorType::SUBTRACTION);
    }

    CcuRelationalOperator<Variable, uint64_t> Variable::operator!=(uint64_t immediate) const
    {
        return CcuRelationalOperator<Variable, uint64_t>(*this, immediate, CcuRelationalOperatorType::NOT_EQUAL);
    }

    CcuRelationalOperator<Variable, uint64_t> Variable::operator==(uint64_t immediate) const
    {
        return CcuRelationalOperator<Variable, uint64_t>(*this, immediate, CcuRelationalOperatorType::EQUAL);
    }

    CcuRelationalOperator<Variable, uint64_t> Variable::operator<=(uint64_t immediate) const
    {
        return CcuRelationalOperator<Variable, uint64_t>(*this, immediate, CcuRelationalOperatorType::LESS_EQUAL);
    }

    CcuRelationalOperator<Variable, uint64_t> Variable::operator>(uint64_t immediate) const
    {
        return CcuRelationalOperator<Variable, uint64_t>(*this, immediate, CcuRelationalOperatorType::GREATER_THAN);
    }

    CcuRelationalOperator<Variable, Variable> Variable::operator<=(const Variable& varB) const
    {
        return CcuRelationalOperator<Variable, Variable>(*this, varB, CcuRelationalOperatorType::LESS_EQUAL);
    }

    CcuRelationalOperator<Variable, Variable> Variable::operator>(const Variable& varB) const
    {
        return CcuRelationalOperator<Variable, Variable>(*this, varB, CcuRelationalOperatorType::GREATER_THAN);
    }

    template <>
    void CcuLogicOperator<Variable, Variable>::Check() const
    {
        // nothing
    }

    template <>
    void CcuLogicOperator<Variable, Address>::Check() const
    {
        // nothing
    }

    template <>
    void CcuLogicOperator<Variable, uint64_t>::Check() const
    {
        // nothing
    }

    template <>
    void CcuLogicOperator<Address, Address>::Check() const
    {
        // nothing
    }

    template <>
    void CcuLogicOperator<Address, uint16_t>::Check() const
    {
        // nothing
    }

    template <>
    void CcuLogicOperator<Variable>::Check() const
    {
        // nothing
    }

    template <>
    void CcuLogicOperator<Address>::Check() const
    {
        // nothing
    }

    CcuLogicOperator<Variable, Variable> Variable::operator&(const Variable& varB) const
    {
        return CcuLogicOperator<Variable, Variable>(*this, varB, CcuLogicOperatorType::AND);
    }

    CcuLogicOperator<Variable, Address> Variable::operator&(const Address& addrB) const
    {
        return CcuLogicOperator<Variable, Address>(*this, addrB, CcuLogicOperatorType::AND);
    }

    CcuLogicOperator<Variable, Variable> Variable::operator|(const Variable& varB) const
    {
        return CcuLogicOperator<Variable, Variable>(*this, varB, CcuLogicOperatorType::OR);
    }

    CcuLogicOperator<Variable, Address> Variable::operator|(const Address& addrB) const
    {
        return CcuLogicOperator<Variable, Address>(*this, addrB, CcuLogicOperatorType::OR);
    }

    CcuLogicOperator<Variable, Variable> Variable::operator^(const Variable& varB) const
    {
        return CcuLogicOperator<Variable, Variable>(*this, varB, CcuLogicOperatorType::XOR);
    }

    CcuLogicOperator<Variable, Address> Variable::operator^(const Address& addrB) const
    {
        return CcuLogicOperator<Variable, Address>(*this, addrB, CcuLogicOperatorType::XOR);
    }

    CcuLogicOperator<Variable> Variable::operator~() const
    {
        return CcuLogicOperator<Variable>(*this, CcuLogicOperatorType::NOT);
    }

    CcuLogicOperator<Address, Address> Address::operator^(const Address& addrB) const
    {
        return CcuLogicOperator<Address, Address>(*this, addrB, CcuLogicOperatorType::XOR);
    }

    CcuLogicOperator<Variable, Address> Address::operator^(const Variable& varB) const
    {
        return CcuLogicOperator<Variable, Address>(varB, *this, CcuLogicOperatorType::XOR);
    }

    CcuLogicOperator<Address, Address> Address::operator&(const Address& addrB) const
    {
        return CcuLogicOperator<Address, Address>(*this, addrB, CcuLogicOperatorType::AND);
    }

    CcuLogicOperator<Variable, Address> Address::operator&(const Variable& varB) const
    {
        return CcuLogicOperator<Variable, Address>(varB, *this, CcuLogicOperatorType::AND);
    }

    CcuLogicOperator<Address, Address> Address::operator|(const Address& addrB) const
    {
        return CcuLogicOperator<Address, Address>(*this, addrB, CcuLogicOperatorType::OR);
    }

    CcuLogicOperator<Variable, Address> Address::operator|(const Variable& varB) const
    {
        return CcuLogicOperator<Variable, Address>(varB, *this, CcuLogicOperatorType::OR);
    }

    template <>
    void CcuShiftOperator<Variable, Variable>::Check() const
    {
        // nothing
    }

    CcuShiftOperator<Variable, Variable> Variable::operator>>(const Variable& other) const
    {
        return CcuShiftOperator<Variable, Variable>(*this, other, CcuShiftType::RIGHT);
    }

    CcuShiftOperator<Variable, Variable> Variable::operator<<(const Variable& other) const
    {
        return CcuShiftOperator<Variable, Variable>(*this, other, CcuShiftType::LEFT);
    }

}; // namespace CcuRep
}; // namespace hcomm

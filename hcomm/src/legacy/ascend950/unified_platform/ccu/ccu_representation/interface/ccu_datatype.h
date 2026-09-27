/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_DATATYPE_H
#define HCCL_CCU_DATATYPE_H

#include <memory>

#include "ccu_rep_base.h"
#include "../reps/arithmetic/ccu_operator.h"
#include "ccu_rep_context.h"

namespace Hccl {
namespace CcuRep {

    class Variable;
    class Address;

    class CcuPhyRes {
    public:
        CcuPhyRes() = default;
        ~CcuPhyRes() = default;
        void Reset(uint16_t id);
        void SetDieId(uint16_t dieId);
        uint16_t Id() const;
        uint16_t DieId() const;

    private:
        uint16_t dieId{0};
        uint16_t id{0};
    };

    class CcuVirRes {
    public:
        CcuVirRes(CcuRepContext* context);
        virtual ~CcuVirRes() = default;
        void Reset(uint16_t id);
        void Reset(uint16_t id, uint16_t dieId);
        void SetDieId(uint16_t dieId);
        virtual uint16_t Id() const;
        uint16_t DieId() const;

    protected:
        std::shared_ptr<CcuPhyRes> phyRes{nullptr};
        CcuRepContext* context{nullptr};
    };

    class Variable : public CcuVirRes {
    public:
        explicit Variable(CcuRepContext* context = nullptr);
        Variable(const Variable& other);
        void operator=(Variable&& other);

        void operator=(uint64_t immediate);
        void operator=(const Variable& other);
        void operator=(CcuArithmeticOperator<Variable, Variable> op);

        CcuArithmeticOperator<Variable, Variable> operator+(const Variable& varB) const;
        CcuArithmeticOperator<Variable, Address> operator+(const Address& addrB) const;

        void operator+=(const Variable& other);

        CcuRelationalOperator<Variable, uint64_t> operator!=(uint64_t immediate) const;
        CcuRelationalOperator<Variable, uint64_t> operator==(uint64_t immediate) const;
    };

    class Address : public CcuVirRes {
    public:
        explicit Address(CcuRepContext* context = nullptr);
        Address(const Address& other);
        void operator=(Address&& other);

        void operator=(uint64_t immediate);
        void operator=(const Address& other);
        void operator=(const Variable& other);

        void operator=(CcuArithmeticOperator<Variable, Address> op);
        void operator=(CcuArithmeticOperator<Address, Variable> op);
        void operator=(CcuArithmeticOperator<Address, Address> op);

        CcuArithmeticOperator<Variable, Address> operator+(const Variable& varB) const;
        CcuArithmeticOperator<Address, Address> operator+(const Address& addrB) const;

        void operator+=(const Variable& other);
    };

    class MaskSignal : public CcuVirRes {
    public:
        explicit MaskSignal(CcuRepContext* context = nullptr);
    };

    class CcuBuffer : public CcuVirRes {
    public:
        explicit CcuBuffer(CcuRepContext* context = nullptr);
        uint16_t Id() const override;
        static constexpr uint16_t CCUBUFFER_DIE_ID_BIT = 0x8000; // bit15 表示MS所在的IO Die id
    };

    class Executor : public CcuVirRes {
    public:
        explicit Executor(CcuRepContext* context = nullptr);
    };

    class Memory {
    public:
        Memory() = default;
        Memory(Address addr, Variable token) : addr(addr), token(token) {}
        Address addr;
        Variable token;
    };

}; // namespace CcuRep
}; // namespace Hccl
#endif // HCCL_CCU_DATATYPE_H

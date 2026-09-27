/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_datatype.h"
#include "ccu_rep.h"
#include "ccu_context_resource.h"
#include "ccu_interface_assist.h"

#include "exception_util.h"
#include "ccu_api_exception.h"

namespace Hccl {
namespace CcuRep {

    uint16_t CcuPhyRes::Id() const { return id; }
    uint16_t CcuPhyRes::DieId() const { return dieId; }
    void CcuPhyRes::Reset(uint16_t id) { this->id = id; }
    void CcuPhyRes::SetDieId(uint16_t dieId) { this->dieId = dieId; }

    CcuVirRes::CcuVirRes(CcuRepContext* context) : context(context) { phyRes = std::make_shared<CcuPhyRes>(); }

    void CcuVirRes::Reset(uint16_t id) { phyRes->Reset(id); }

    void CcuVirRes::Reset(uint16_t id, uint16_t dieId)
    {
        phyRes->Reset(id);
        phyRes->SetDieId(dieId);
    }

    void CcuVirRes::SetDieId(uint16_t dieId) { phyRes->SetDieId(dieId); }

    uint16_t CcuVirRes::Id() const { return phyRes->Id(); }

    uint16_t CcuVirRes::DieId() const { return phyRes->DieId(); }

    Variable::Variable(CcuRepContext* context) : CcuVirRes(context) {}

    Variable::Variable(const Variable& other) : CcuVirRes(other.context) { phyRes = other.phyRes; }

    void Variable::operator=(Variable&& other)
    {
        phyRes = other.phyRes;
        context = other.context;
    }

    void Variable::operator=(const Variable& other)
    {
        AppendToContext(context, std::make_shared<CcuRepAssign>(*this, other));
    }

    void Variable::operator=(uint64_t immediate)
    {
        AppendToContext(context, std::make_shared<CcuRepAssign>(*this, immediate));
    }

    void Variable::operator=(CcuArithmeticOperator<Variable, Variable> op)
    {
        AppendToContext(context, std::make_shared<CcuRepAdd>(*this, op.lhs, op.rhs));
    }

    void Variable::operator+=(const Variable& other)
    {
        AppendToContext(context, std::make_shared<CcuRepAdd>(*this, other));
    }

    Address::Address(CcuRepContext* context) : CcuVirRes(context) {}

    Address::Address(const Address& other) : CcuVirRes(other.context) { phyRes = other.phyRes; }

    void Address::operator=(Address&& other)
    {
        phyRes = other.phyRes;
        context = other.context;
    }

    void Address::operator=(const Address& other)
    {
        AppendToContext(context, std::make_shared<CcuRepAssign>(*this, other));
    }

    void Address::operator=(const Variable& other)
    {
        AppendToContext(context, std::make_shared<CcuRepAssign>(*this, other));
    }

    void Address::operator=(uint64_t immediate)
    {
        AppendToContext(context, std::make_shared<CcuRepAssign>(*this, immediate));
    }

    void Address::operator=(CcuArithmeticOperator<Variable, Address> op)
    {
        AppendToContext(context, std::make_shared<CcuRepAdd>(*this, op.rhs, op.lhs));
    }

    void Address::operator=(CcuArithmeticOperator<Address, Address> op)
    {
        AppendToContext(context, std::make_shared<CcuRepAdd>(*this, op.lhs, op.rhs));
    }

    void Address::operator+=(const Variable& other)
    {
        AppendToContext(context, std::make_shared<CcuRepAdd>(*this, other));
    }

    MaskSignal::MaskSignal(CcuRepContext* context) : CcuVirRes(context) {}

    CcuBuffer::CcuBuffer(CcuRepContext* context) : CcuVirRes(context) {}

    uint16_t CcuBuffer::Id() const { return phyRes->Id() + CCUBUFFER_DIE_ID_BIT * phyRes->DieId(); }

    Executor::Executor(CcuRepContext* context) : CcuVirRes(context) {}

}; // namespace CcuRep
}; // namespace Hccl

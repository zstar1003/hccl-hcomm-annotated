/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_REPRESENTATION_WRITE_VAR_ATOMIC_H
#define HCOMM_CCU_REPRESENTATION_WRITE_VAR_ATOMIC_H

#include "ccu_rep_base_v1.h"
#include "ccu_datatype_v1.h"

namespace hcomm {
namespace CcuRep {

    class CcuRepWriteVarAtomic : public CcuRepBase {
    public:
        CcuRepWriteVarAtomic(
            CcuInsGeneratorBase* insGenPtr, Variable& channel, RemoteAddr& varAddr, Variable& targetValue,
            CompletedEvent sem, uint16_t mask = 1);

        bool Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;

        uint16_t GetChannelVarId() { return channel.Id(); }
        uint16_t GetVarAddrId() { return varAddr.addr.Id(); }
        uint16_t GetVarTokenId() { return varAddr.token.Id(); }
        uint16_t GetTargetId() { return targetValue.Id(); }
        uint16_t GetSemId() { return sem.Id(); }

        const Variable& GetChannelVar() const { return channel; }
        const RemoteAddr& GetVarAddr() const { return varAddr; }
        const Variable& GetTarget() const { return targetValue; }
        CompletedEvent GetSem() { return sem; }
        uint16_t GetMask() { return mask; }

    private:
        CcuInsGeneratorBase* insGenPtr{nullptr};
        Variable channel;
        RemoteAddr varAddr;
        Variable targetValue;
        CompletedEvent sem;
        uint16_t mask{0};
    };

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCOMM_CCU_REPRESENTATION_WRITE_VAR_ATOMIC_H

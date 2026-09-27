/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_REPRESENTATION_LOADARG_H
#define HCOMM_CCU_REPRESENTATION_LOADARG_H

#include "ccu_rep_base_v1.h"
#include "ccu_datatype_v1.h"

namespace hcomm {
namespace CcuRep {

    class CcuRepLoadArg : public CcuRepBase {
    public:
        explicit CcuRepLoadArg(CcuInsGeneratorBase* insGenPtr, const Variable& var, uint16_t argId, uint16_t fullArgId);
        bool Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;

        CcuRep::Variable GetVar() { return var; }
        uint16_t GetArgId() const { return argId; }
        uint16_t GetVarId() const { return var.Id(); }
        uint16_t GetFullArgId() const { return fullArgId; }

    private:
        CcuInsGeneratorBase* insGeneratorPtr_{nullptr};
        Variable var;
        uint16_t argId{0};
        uint16_t fullArgId{0};
    };

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCCL_CCU_REPRESENTATION_LOADARG_H

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_REPRESENTATION_LOAD_H
#define HCOMM_CCU_REPRESENTATION_LOAD_H

#include "ccu_rep_base_v1.h"
#include "ccu_datatype_v1.h"

namespace hcomm {
namespace CcuRep {

    class CcuRepLoad : public CcuRepBase {
    public:
        explicit CcuRepLoad(CcuInsGeneratorBase* insGenPtr, uint64_t addr, const Variable& var, uint32_t num = 1);
        bool Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;

        CcuRep::Variable GetVar() { return var; }
        uint64_t GetAddr() const { return addr; }
        uint32_t GetNum() const { return num; }
        uint16_t GetMask() const { return mask; }

    private:
        CcuInsGeneratorBase* insGeneratorPtr_{nullptr};
        Variable var;
        uint64_t addr;
        uint32_t num;
        uint16_t mask{1};
    };

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCCL_CCU_REPRESENTATION_LOAD_H

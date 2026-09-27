/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_REPRESENTATION_STORE_VAR_H_V1
#define HCOMM_CCU_REPRESENTATION_STORE_VAR_H_V1

#include "ccu_rep_base_v1.h"
#include "ccu_datatype_v1.h"

namespace hcomm {
namespace CcuRep {

    class CcuRepStoreVar : public CcuRepBase {
    public:
        CcuRepStoreVar(
            CcuInsGeneratorBase* insGeneratorPtr, const Variable& var, const Variable& dst, uint32_t num = 1,
            bool hscbFlag = false);
        bool Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;

        Variable GetVar() { return var; }
        Variable GetDst() { return dst; }
        uint32_t GetNum() const { return num; }
        uint16_t GetMask() const { return mask; }
        bool GetHscbFlag() const { return hscbFlag; }

    private:
        CcuInsGeneratorBase* insGeneratorPtr_{nullptr};
        Variable var;
        Variable dst;
        uint32_t num;
        uint16_t mask{1};
        bool hscbFlag{false};
    };

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCCL_CCU_REPRESENTATION_STORE_VAR_H

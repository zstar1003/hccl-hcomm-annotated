/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_REP_LOAD_VAR_H
#define HCCL_CCU_REP_LOAD_VAR_H

#include "ccu_rep_base.h"
#include "ccu_datatype.h"

namespace Hccl {
namespace CcuRep {

    class CcuRepLoadVar : public CcuRepBase {
    public:
        CcuRepLoadVar(const Variable& src, const Variable& var, uint32_t num = 1);
        bool Translate(CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;

    private:
        Variable src;
        Variable var;
        uint32_t num;
        uint16_t mask{1};
    };

}; // namespace CcuRep
}; // namespace Hccl
#endif // HCCL_CCU_REP_LOAD_VAR_H

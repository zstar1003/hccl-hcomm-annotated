/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_REPRESENTATION_LOOPGROUP_H
#define HCCL_CCU_REPRESENTATION_LOOPGROUP_H

#include "ccu_datatype.h"
#include "ccu_rep_base.h"
#include "ccu_error_handler.h"

namespace Hccl {
namespace CcuRep {

    class CcuRepLoopGroup : public CcuRepBase {
    public:
        explicit CcuRepLoopGroup(const Variable& parallelParam, const Variable& offsetParam);
        bool Translate(CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;

        std::shared_ptr<CcuRepBase> SetParallelParam(Variable var);
        std::shared_ptr<CcuRepBase> SetOffsetParam(Variable var);
        uint16_t GetStartLoopInstrId() const;

    private:
        Variable parallelParam;
        Variable offsetParam;

        friend class Hccl::CcuErrorHandler;
    };

}; // namespace CcuRep
}; // namespace Hccl
#endif // HCCL_CCU_REPRESENTATION_LOOPGROUP_H

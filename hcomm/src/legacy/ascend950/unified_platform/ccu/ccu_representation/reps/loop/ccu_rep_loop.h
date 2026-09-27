/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_REPRESENTATION_LOOP_H
#define HCCL_CCU_REPRESENTATION_LOOP_H

#include <memory>

#include "ccu_datatype.h"
#include "ccu_rep_base.h"
#include "ccu_rep_loopblock.h"
#include "ccu_error_handler.h"

namespace Hccl {
namespace CcuRep {

    class CcuRepLoop : public CcuRepBase {
    public:
        explicit CcuRepLoop(const std::string& label, const Variable& loopParam);
        bool Translate(CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;
        const std::string& GetLabel() const;

        void Reference(std::shared_ptr<CcuRepLoopBlock> refRep);
        std::shared_ptr<CcuRepBase> SetLoopParam(Executor executor, Variable var);

    private:
        std::string label;
        std::shared_ptr<CcuRepLoopBlock> loopBlock{nullptr};

        Variable loopParam;
        CcuInstr* instr{nullptr};

        friend class Hccl::CcuErrorHandler;
    };

}; // namespace CcuRep
}; // namespace Hccl
#endif // HCCL_CCU_REPRESENTATION_LOOP_H

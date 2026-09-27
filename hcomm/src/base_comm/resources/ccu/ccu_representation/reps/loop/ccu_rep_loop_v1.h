/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_REPRESENTATION_LOOP_H
#define HCOMM_CCU_REPRESENTATION_LOOP_H

#include <memory>

#include "ccu_datatype_v1.h"
#include "ccu_rep_base_v1.h"
#include "ccu_rep_loopblock_v1.h"

namespace hcomm {
namespace CcuRep {

    class CcuRepLoop : public CcuRepBase {
    public:
        explicit CcuRepLoop(CcuInsGeneratorBase* insGeneratorPtr, const std::string& label, const Variable& loopParam);
        explicit CcuRepLoop(
            CcuInsGeneratorBase* insGeneratorPtr, const std::string& label, const Variable& loopParam,
            const Variable& loopIterNum, const Variable& loopGsaOffset);
        bool Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;
        const std::string& GetLabel() const;

        void Reference(std::shared_ptr<CcuRepLoopBlock> refRep);
        std::shared_ptr<CcuRepBase> SetLoopParam(Executor executor, Variable var);
        CcuRepLoopBlock* GetLoopBlock() { return loopBlock.get(); }

        Variable* GetLoopParam() { return &loopParam; }
        Variable GetLoopIterNum() { return loopIterNum; }
        Variable GetLoopGsaOffset() { return loopGsaOffset; }

    private:
        void ValidateInsGeneratorForLoop() const;

        CcuInsGeneratorBase* insGeneratorPtr_;
        std::string label;
        std::shared_ptr<CcuRepLoopBlock> loopBlock{nullptr};

        Variable loopParam;
        Variable loopIterNum;
        Variable loopGsaOffset;
        CcuInstr* instr{nullptr};

        bool supportCcuV1{false};
        bool supportCcuV2{false};
    };

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCCL_CCU_REPRESENTATION_LOOP_H

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_REPRESENTATION_LOOP_CALL_H
#define HCCL_CCU_REPRESENTATION_LOOP_CALL_H

#include "ccu_rep_base.h"
#include "ccu_rep_loopblock.h"

namespace Hccl {
namespace CcuRep {

    class CcuRepLoopCall : public CcuRepBase {
    public:
        explicit CcuRepLoopCall(const std::string& label);
        bool Translate(CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;
        uint16_t InstrCount() override;
        const std::string& GetLabel() const;

        void Reference(std::shared_ptr<CcuRepLoopBlock> refRep);

        void SetInArg(const Variable& var);
        void SetInArg(const std::vector<Variable>& varList);
        void SetInArg(const Memory& mem);
        void SetInArg(const std::vector<Memory>& memList);

    private:
        std::string label;
        std::shared_ptr<CcuRepLoopBlock> loopBlock{nullptr};

        std::vector<CcuRepArg> inArgs;
        uint32_t inArgCount{0};
        uint32_t inArgInstrCount{0}; // 处理LoopCall的入参需要的指令数

        CcuInstr* instr{nullptr};
    };

}; // namespace CcuRep
}; // namespace Hccl
#endif // HCCL_CCU_REPRESENTATION_LOOP_CALL_H

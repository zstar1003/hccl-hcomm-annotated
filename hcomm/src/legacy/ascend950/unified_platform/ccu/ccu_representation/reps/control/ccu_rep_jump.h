/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_REPRESENTATION_JUMP_H
#define HCCL_CCU_REPRESENTATION_JUMP_H

#include <memory>

#include "ccu_datatype.h"
#include "ccu_rep_base.h"
#include "ccu_rep_jumplabel.h"

namespace Hccl {
namespace CcuRep {

    class CcuRepJumpBase : public CcuRepBase {
    public:
        explicit CcuRepJumpBase(const std::string& label, const Variable& targetInstrId);
        void Reference(std::shared_ptr<CcuRepJumpLabel> refRep);

    protected:
        std::string label;
        std::shared_ptr<CcuRepJumpLabel> jumpLabel{nullptr};
        Variable targetInstrId;
        CcuInstr* instr{nullptr};
    };

    class CcuRepJump : public CcuRepJumpBase {
    public:
        explicit CcuRepJump(const std::string& label, const Variable& targetInstrId);
        bool Translate(CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;
    };

    class CcuRepJumpNE : public CcuRepJumpBase {
    public:
        CcuRepJumpNE(
            const std::string& label, const Variable& targetInstrId, const Variable& condition, uint64_t expected);
        bool Translate(CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;

    private:
        Variable condition;
        uint64_t expected{0};
    };

    class CcuRepJumpEQ : public CcuRepJumpBase {
    public:
        CcuRepJumpEQ(
            const std::string& label, const Variable& targetInstrId, const Variable& condition, uint64_t expected);
        bool Translate(CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;

    private:
        Variable condition;
        uint64_t expected{0};
    };

}; // namespace CcuRep
}; // namespace Hccl
#endif // HCCL_CCU_REPRESENTATION_JUMP_H

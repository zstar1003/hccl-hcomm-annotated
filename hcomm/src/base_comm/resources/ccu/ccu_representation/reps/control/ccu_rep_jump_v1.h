/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_REPRESENTATION_JUMP_H
#define CCU_REPRESENTATION_JUMP_H

#include <memory>

#include "ccu_datatype_v1.h"
#include "ccu_rep_base_v1.h"
#include "ccu_rep_jumplabel_v1.h"
#include "ccu_types.h"

namespace hcomm {
namespace CcuRep {

    enum class ConditionType { EQUAL, NOT_EQUAL, GREATER_THAN, GREATER_EQUAL, LESS_THAN, LESS_EQUAL, DEFAULT, INVALID };

    class CcuRepJumpBase : public CcuRepBase {
    public:
        explicit CcuRepJumpBase(
            CcuInsGeneratorBase* insGenPtr, const std::string& label, const Variable& targetInstrId);
        explicit CcuRepJumpBase(
            CcuInsGeneratorBase* insGenPtr, const std::string& label, const Variable& targetInstrId,
            const Variable& expectedVar, const Variable& condition);
        void Reference(std::shared_ptr<CcuRepJumpLabel> refRep);
        void ValidateInsGeneratorForJump();

        std::shared_ptr<CcuRepJumpLabel> GetJumpLabel() { return jumpLabel; }

        Variable& GetTargetInstrId() { return targetInstrId; }

        Variable& GetCondition() { return condition; }

        Variable& GetExpectedVar() { return expectedVar; }

        uint64_t GetExpectedNum() const { return expected; }

        CcuInstr* GetInstr() { return instr; }

        bool IsComparedWithImmd() const { return comp2Immed; }

    protected:
        CcuResult InitInstr(CcuInstr*& instr, uint16_t& curInstrId);

        CcuInsGeneratorBase* insGeneratorPtr_;
        std::string label;
        std::shared_ptr<CcuRepJumpLabel> jumpLabel{nullptr};
        Variable targetInstrId;
        CcuInstr* instr{nullptr};

        bool comp2Immed{false};
        bool supportCcuV1{true}; // 暂定用于识别 使用特定的构造方法时是否支持A5的翻译流程

        Variable expectedVar;
        Variable condition;
        uint64_t expected{0};
    };

    class CcuRepJump : public CcuRepJumpBase {
    public:
        explicit CcuRepJump(CcuInsGeneratorBase* insGenPtr, const std::string& label, const Variable& targetInstrId);
        bool Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;
    };

    class CcuRepJumpNE : public CcuRepJumpBase {
    public:
        CcuRepJumpNE(
            CcuInsGeneratorBase* insGenPtr, const std::string& label, const Variable& targetInstrId,
            const Variable& expectedVar, const Variable& condition, uint64_t expected);
        CcuRepJumpNE(
            CcuInsGeneratorBase* insGenPtr, const std::string& label, const Variable& targetInstrId,
            const Variable& condition, const Variable& expectedVar); // 仅用于A6翻译
        bool Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;
    };

    class CcuRepJumpEQ : public CcuRepJumpBase {
    public:
        CcuRepJumpEQ(
            CcuInsGeneratorBase* insGenPtr, const std::string& label, const Variable& targetInstrId,
            const Variable& expectedVar, const Variable& condition, uint64_t expected);
        CcuRepJumpEQ(
            CcuInsGeneratorBase* insGenPtr, const std::string& label, const Variable& targetInstrId,
            const Variable& condition, const Variable& expectedVar); // 仅用于A6翻译
        bool Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        std::string Describe() override;
    };

    class CcuRepJumpLE : public CcuRepJumpBase {
    public:
        CcuRepJumpLE(
            CcuInsGeneratorBase* insGenPtr, const std::string& label, const Variable& targetInstrId,
            const Variable& condition, const Variable& expectedVar);
        CcuRepJumpLE(
            CcuInsGeneratorBase* insGenPtr, const std::string& label, const Variable& targetInstrId,
            const Variable& expectedVar, const Variable& condition, uint64_t expected);
        bool Translate(CcuKernel* ccuKernel, CcuInstr*& curInstr, uint16_t& curInstrId, const TransDep& dep) override;
        std::string Describe() override;
    };

    class CcuRepJumpGE : public CcuRepJumpBase {
    public:
        CcuRepJumpGE(
            CcuInsGeneratorBase* insGenPtr, const std::string& label, const Variable& targetInstrId,
            const Variable& condition, const Variable& expectedVar);
        CcuRepJumpGE(
            CcuInsGeneratorBase* insGenPtr, const std::string& label, const Variable& targetInstrId,
            const Variable& expectedVar, const Variable& condition, uint64_t expected);
        bool Translate(CcuKernel* ccuKernel, CcuInstr*& curInstr, uint16_t& curInstrId, const TransDep& dep) override;
        std::string Describe() override;
    };

    class CcuRepJumpGT : public CcuRepJumpBase {
    public:
        CcuRepJumpGT(
            CcuInsGeneratorBase* insGenPtr, const std::string& label, const Variable& targetInstrId,
            const Variable& condition, const Variable& expectedVar);
        CcuRepJumpGT(
            CcuInsGeneratorBase* insGenPtr, const std::string& label, const Variable& targetInstrId,
            const Variable& expectedVar, const Variable& condition, uint64_t expected);
        bool Translate(CcuKernel* ccuKernel, CcuInstr*& curInstr, uint16_t& curInstrId, const TransDep& dep) override;
        std::string Describe() override;
    };

    class CcuRepJumpLT : public CcuRepJumpBase {
    public:
        CcuRepJumpLT(
            CcuInsGeneratorBase* insGenPtr, const std::string& label, const Variable& targetInstrId,
            const Variable& condition, const Variable& expectedVar);
        CcuRepJumpLT(
            CcuInsGeneratorBase* insGenPtr, const std::string& label, const Variable& targetInstrId,
            const Variable& expectedVar, const Variable& condition, uint64_t expected);
        bool Translate(CcuKernel* ccuKernel, CcuInstr*& curInstr, uint16_t& curInstrId, const TransDep& dep) override;
        std::string Describe() override;
    };
}; // namespace CcuRep
}; // namespace hcomm
#endif // _CCU_REPRESENTATION_JUMP_H

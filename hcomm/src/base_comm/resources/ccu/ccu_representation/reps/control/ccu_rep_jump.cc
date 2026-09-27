/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_rep_v1.h"
#include "exception_util.h"
#include "ccu_api_exception.h"
#include "ccu_ins_generator_base.h"
#include "ccu_ins_generator_v1.h"

namespace hcomm {
namespace CcuRep {

    using namespace Hccl;

    // jump基类
    CcuRepJumpBase::CcuRepJumpBase(
        CcuInsGeneratorBase* insGenPtr, const std::string& label, const Variable& targetInstrId)
        : insGeneratorPtr_(insGenPtr),
          label(label),
          targetInstrId(targetInstrId)
    {}

    CcuRepJumpBase::CcuRepJumpBase(
        CcuInsGeneratorBase* insGenPtr, const std::string& label, const Variable& targetInstrId,
        const Variable& expectedVar, const Variable& condition)
        : insGeneratorPtr_(insGenPtr),
          label(label),
          targetInstrId(targetInstrId),
          expectedVar(expectedVar),
          condition(condition)
    {}

    void CcuRepJumpBase::Reference(std::shared_ptr<CcuRepJumpLabel> refRep) { jumpLabel = refRep; }

    void CcuRepJumpBase::ValidateInsGeneratorForJump()
    {
        CcuInsGeneratorV1* tmpPtrV1 = dynamic_cast<CcuInsGeneratorV1*>(insGeneratorPtr_);
        if (tmpPtrV1 && !supportCcuV1) {
            // 当右值只传入var没有立即数时，无法在A5上翻译
            Hccl::THROW<Hccl::CcuApiException>(
                "Cannot translate %s for A5 when supportCcuV1 is false!", this->Describe().c_str());
        }
    }

    CcuResult CcuRepJumpBase::InitInstr(CcuInstr*& instr, uint16_t& curInstrId)
    {
        CCU_CHK_PTR_NULL(instr);
        if (this->instr == nullptr) {
            this->instrId = curInstrId;
            this->instr = instr;
            instr += instrCount;
            curInstrId += instrCount;
        }
        return CcuResult::CCU_SUCCESS;
    }

    // direct jump
    CcuRepJump::CcuRepJump(CcuInsGeneratorBase* insGenPtr, const std::string& label, const Variable& targetInstrId)
        : CcuRepJumpBase(insGenPtr, label, targetInstrId)
    {
        type = CcuRepType::JUMP;
        instrCount = insGeneratorPtr_->GetInstrCount(type);
    }

    bool CcuRepJump::Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, const TransDep& dep)
    {
        if (InitInstr(instr, curInstrId) != CcuResult::CCU_SUCCESS) {
            Hccl::THROW<Hccl::CcuApiException>("instr is empty!");
        }

        if (jumpLabel->Translated()) {
            CHK_PRT_THROW(
                insGeneratorPtr_->CcuRepJumpTranslate(ccuKernel, instr, curInstrId, this, dep)
                    != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR("[CcuRepJump][Translate] failed to translate for instrId[%u]", curInstrId),
                Hccl::CcuApiException, "CcuRepJump translate failed");
            translated = true;
        }

        return translated;
    }

    std::string CcuRepJump::Describe() { return Hccl::StringFormat("Jump To Label[%s]", label.c_str()); }

    // jumpNE
    CcuRepJumpNE::CcuRepJumpNE(
        CcuInsGeneratorBase* insGenPtr, const std::string& label, const Variable& targetInstrId,
        const Variable& expectedVar, const Variable& condition, uint64_t expected)
        : CcuRepJumpBase(insGenPtr, label, targetInstrId, expectedVar, condition)
    {
        this->expected = expected;
        type = CcuRepType::JUMP_NE;
        instrCount = insGeneratorPtr_->GetInstrCount(type);
        comp2Immed = true; // A6翻译时插入一条加载立即数指令，A5无关
        supportCcuV1 = true;
    }

    CcuRepJumpNE::CcuRepJumpNE(
        CcuInsGeneratorBase* insGenPtr, const std::string& label, const Variable& targetInstrId,
        const Variable& condition, const Variable& expectedVar)
        : CcuRepJumpBase(insGenPtr, label, targetInstrId, expectedVar, condition)
    {
        // 仅用于A6翻译
        type = CcuRepType::JUMP_NE;
        instrCount = 2; // 2条指令，暂直接填充指令数，insGenerator中未记录这种使用方式对应的指令数
        comp2Immed = false;
        supportCcuV1 = false;
    }

    bool CcuRepJumpNE::Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, const TransDep& dep)
    {
        ValidateInsGeneratorForJump();

        if (InitInstr(instr, curInstrId) != CcuResult::CCU_SUCCESS) {
            Hccl::THROW<Hccl::CcuApiException>("instr is empty!");
        }

        if (jumpLabel->Translated()) {
            CHK_PRT_THROW(
                insGeneratorPtr_->CcuRepJumpNETranslate(ccuKernel, instr, curInstrId, this, dep)
                    != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR("[CcuRepJumpNE][Translate] failed to translate for instrId[%u]", curInstrId),
                Hccl::CcuApiException, "CcuRepJumpNE translate failed");
            translated = true;
        }

        return translated;
    }

    std::string CcuRepJumpNE::Describe()
    {
        return Hccl::StringFormat(
            "Jump To Label[%s], When Condition[%u] Not equal to Expected[%lu]", label.c_str(), condition.Id(),
            expected);
    }

    // jumpEQ
    CcuRepJumpEQ::CcuRepJumpEQ(
        CcuInsGeneratorBase* insGenPtr, const std::string& label, const Variable& targetInstrId,
        const Variable& expectedVar, const Variable& condition, uint64_t expected)
        : CcuRepJumpBase(insGenPtr, label, targetInstrId, expectedVar, condition)
    {
        this->expected = expected;
        type = CcuRepType::JUMP_EQ;
        instrCount = insGeneratorPtr_->GetInstrCount(type);
        comp2Immed = true; // A6翻译时插入一条加载立即数指令，A5无关
        supportCcuV1 = true;
    }

    CcuRepJumpEQ::CcuRepJumpEQ(
        CcuInsGeneratorBase* insGenPtr, const std::string& label, const Variable& targetInstrId,
        const Variable& condition, const Variable& expectedVar)
        : CcuRepJumpBase(insGenPtr, label, targetInstrId, expectedVar, condition)
    {
        type = CcuRepType::JUMP_EQ;
        instrCount = 2; // 2条指令，暂直接填充指令数，insGenerator中未记录这种使用方式对应的指令数
        comp2Immed = false;
        supportCcuV1 = false;
    }

    bool CcuRepJumpEQ::Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& curInstrId, const TransDep& dep)
    {
        ValidateInsGeneratorForJump();
        if (InitInstr(instr, curInstrId) != CcuResult::CCU_SUCCESS) {
            Hccl::THROW<Hccl::CcuApiException>("instr is empty!");
        }

        if (jumpLabel->Translated()) {
            CHK_PRT_THROW(
                insGeneratorPtr_->CcuRepJumpEQTranslate(ccuKernel, instr, curInstrId, this, dep)
                    != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR("[CcuRepJumpEQ][Translate] failed to translate for instrId[%u]", curInstrId),
                Hccl::CcuApiException, "CcuRepJumpEQ translate failed");
            translated = true;
        }

        return translated;
    }

    std::string CcuRepJumpEQ::Describe()
    {
        return Hccl::StringFormat(
            "Jump To Label[%s], When Condition[%u] Be equal to Expected[%lu]", label.c_str(), condition.Id(), expected);
    }

    // jumpLE
    CcuRepJumpLE::CcuRepJumpLE(
        CcuInsGeneratorBase* insGenPtr, const std::string& label, const Variable& targetInstrId,
        const Variable& condition, const Variable& expectedVar)
        : CcuRepJumpBase(insGenPtr, label, targetInstrId, expectedVar, condition)
    {
        type = CcuRepType::JUMP_LE;
        instrCount = 2; // 2条指令
        comp2Immed = false;
        supportCcuV1 = false;
    }

    CcuRepJumpLE::CcuRepJumpLE(
        CcuInsGeneratorBase* insGenPtr, const std::string& label, const Variable& targetInstrId,
        const Variable& expectedVar, const Variable& condition, uint64_t expected)
        : CcuRepJumpBase(insGenPtr, label, targetInstrId, expectedVar, condition)
    {
        this->expected = expected;
        type = CcuRepType::JUMP_LE;
        instrCount = 3; // 3条指令
        comp2Immed = true;
        supportCcuV1 = false;
    }

    bool CcuRepJumpLE::Translate(CcuKernel* ccuKernel, CcuInstr*& curInstr, uint16_t& curInstrId, const TransDep& dep)
    {
        ValidateInsGeneratorForJump();

        if (InitInstr(curInstr, curInstrId) != CcuResult::CCU_SUCCESS) {
            Hccl::THROW<Hccl::CcuApiException>("instr is empty!");
        }

        if (jumpLabel->Translated()) {
            CHK_PRT_THROW(
                insGeneratorPtr_->CcuRepJumpLETranslate(ccuKernel, curInstr, curInstrId, this, dep)
                    != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR("[CcuRepJumpLE][Translate] failed to translate for instrId[%u]", curInstrId),
                Hccl::CcuApiException, "CcuRepJumpLE translate failed");
            translated = true;
        }

        return translated;
    }

    std::string CcuRepJumpLE::Describe()
    {
        return Hccl::StringFormat(
            "Jump To Label[%s], When Condition[%u] <= Expected[%lu]", label.c_str(), condition.Id(), expected);
    }

    // jumpGE
    CcuRepJumpGE::CcuRepJumpGE(
        CcuInsGeneratorBase* insGenPtr, const std::string& label, const Variable& targetInstrId,
        const Variable& condition, const Variable& expectedVar)
        : CcuRepJumpBase(insGenPtr, label, targetInstrId, expectedVar, condition)
    {
        type = CcuRepType::JUMP_GE;
        instrCount = 2; // 2条指令
        comp2Immed = false;
        supportCcuV1 = false;
    }

    CcuRepJumpGE::CcuRepJumpGE(
        CcuInsGeneratorBase* insGenPtr, const std::string& label, const Variable& targetInstrId,
        const Variable& expectedVar, const Variable& condition, uint64_t expected)
        : CcuRepJumpBase(insGenPtr, label, targetInstrId, expectedVar, condition)
    {
        this->expected = expected;
        type = CcuRepType::JUMP_GE;
        instrCount = 3; // 3条指令
        comp2Immed = true;
        supportCcuV1 = false;
    }

    bool CcuRepJumpGE::Translate(CcuKernel* ccuKernel, CcuInstr*& curInstr, uint16_t& curInstrId, const TransDep& dep)
    {
        ValidateInsGeneratorForJump();

        if (InitInstr(curInstr, curInstrId) != CcuResult::CCU_SUCCESS) {
            Hccl::THROW<Hccl::CcuApiException>("instr is empty!");
        }

        if (jumpLabel->Translated()) {
            CHK_PRT_THROW(
                insGeneratorPtr_->CcuRepJumpGETranslate(ccuKernel, curInstr, curInstrId, this, dep)
                    != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR("[CcuRepJumpGE][Translate] failed to translate for instrId[%u]", curInstrId),
                Hccl::CcuApiException, "CcuRepJumpGE translate failed");

            translated = true;
        }

        return translated;
    }

    std::string CcuRepJumpGE::Describe()
    {
        return Hccl::StringFormat(
            "Jump To Label[%s], When Condition[%u] >= Expected[%lu]", label.c_str(), condition.Id(), expected);
    }

    // jumpGT
    CcuRepJumpGT::CcuRepJumpGT(
        CcuInsGeneratorBase* insGenPtr, const std::string& label, const Variable& targetInstrId,
        const Variable& condition, const Variable& expectedVar)
        : CcuRepJumpBase(insGenPtr, label, targetInstrId, expectedVar, condition)
    {
        type = CcuRepType::JUMP_GT;
        instrCount = 2; // 2条指令
        comp2Immed = false;
        supportCcuV1 = false;
    }

    CcuRepJumpGT::CcuRepJumpGT(
        CcuInsGeneratorBase* insGenPtr, const std::string& label, const Variable& targetInstrId,
        const Variable& expectedVar, const Variable& condition, uint64_t expected)
        : CcuRepJumpBase(insGenPtr, label, targetInstrId, expectedVar, condition)
    {
        this->expected = expected;
        type = CcuRepType::JUMP_GT;
        instrCount = 3; // 3条指令
        comp2Immed = true;
        supportCcuV1 = false;
    }

    bool CcuRepJumpGT::Translate(CcuKernel* ccuKernel, CcuInstr*& curInstr, uint16_t& curInstrId, const TransDep& dep)
    {
        ValidateInsGeneratorForJump();
        if (InitInstr(curInstr, curInstrId) != CcuResult::CCU_SUCCESS) {
            Hccl::THROW<Hccl::CcuApiException>("instr is empty!");
        }
        if (jumpLabel->Translated()) {
            CHK_PRT_THROW(
                insGeneratorPtr_->CcuRepJumpGTTranslate(ccuKernel, curInstr, curInstrId, this, dep)
                    != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR("[CcuRepJumpGT][Translate] failed to translate for instrId[%u]", curInstrId),
                Hccl::CcuApiException, "CcuRepJumpGT translate failed");

            translated = true;
        }

        return translated;
    }

    std::string CcuRepJumpGT::Describe()
    {
        return Hccl::StringFormat(
            "Jump To Label[%s], When Condition[%u] > Expected[%lu]", label.c_str(), condition.Id(), expected);
    }

    // jumpLT
    CcuRepJumpLT::CcuRepJumpLT(
        CcuInsGeneratorBase* insGenPtr, const std::string& label, const Variable& targetInstrId,
        const Variable& condition, const Variable& expectedVar)
        : CcuRepJumpBase(insGenPtr, label, targetInstrId, expectedVar, condition)
    {
        type = CcuRepType::JUMP_LT;
        instrCount = 2; // 2条指令
        comp2Immed = false;
        supportCcuV1 = false;
    }

    CcuRepJumpLT::CcuRepJumpLT(
        CcuInsGeneratorBase* insGenPtr, const std::string& label, const Variable& targetInstrId,
        const Variable& expectedVar, const Variable& condition, uint64_t expected)
        : CcuRepJumpBase(insGenPtr, label, targetInstrId, expectedVar, condition)
    {
        this->expected = expected;
        type = CcuRepType::JUMP_LT;
        instrCount = 3; // 3条指令
        comp2Immed = true;
        supportCcuV1 = false;
    }

    bool CcuRepJumpLT::Translate(CcuKernel* ccuKernel, CcuInstr*& curInstr, uint16_t& curInstrId, const TransDep& dep)
    {
        ValidateInsGeneratorForJump();
        if (InitInstr(curInstr, curInstrId) != CcuResult::CCU_SUCCESS) {
            Hccl::THROW<Hccl::CcuApiException>("instr is empty!");
        }

        if (jumpLabel->Translated()) {
            CHK_PRT_THROW(
                insGeneratorPtr_->CcuRepJumpLTTranslate(ccuKernel, curInstr, curInstrId, this, dep)
                    != HcclResult::HCCL_SUCCESS,
                HCCL_ERROR("[CcuRepJumpLT][Translate] failed to translate for instrId[%u]", curInstrId),
                Hccl::CcuApiException, "CcuRepJumpLT translate failed");

            translated = true;
        }

        return translated;
    }

    std::string CcuRepJumpLT::Describe()
    {
        return Hccl::StringFormat(
            "Jump To Label[%s], When Condition[%u] < Expected[%lu]", label.c_str(), condition.Id(), expected);
    }
}; // namespace CcuRep
}; // namespace hcomm

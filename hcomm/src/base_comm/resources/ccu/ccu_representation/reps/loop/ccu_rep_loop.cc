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
#include <climits>

#include "exception_util.h"
#include "ccu_api_exception.h"
#include "ccu_ins_generator_base.h"
#include "ccu_ins_generator_v1.h"
#include "ccu_ins_generator_v2.h"

namespace hcomm {
namespace CcuRep {

    using namespace Hccl;

    CcuRepLoop::CcuRepLoop(CcuInsGeneratorBase* insGeneratorPtr, const std::string& label, const Variable& loopParam)
        : insGeneratorPtr_(insGeneratorPtr),
          label(label),
          loopParam(loopParam)
    {
        type = CcuRepType::LOOP;
        instrCount = insGeneratorPtr_->GetInstrCount(type);
        supportCcuV1 = true;
        supportCcuV2 = false; // 缺少A6格式下必需的loop参数
    }

    CcuRepLoop::CcuRepLoop(
        CcuInsGeneratorBase* insGeneratorPtr, const std::string& label, const Variable& loopParam,
        const Variable& loopIterNum, const Variable& loopGsaOffset)
        : insGeneratorPtr_(insGeneratorPtr),
          label(label),
          loopParam(loopParam),
          loopIterNum(loopIterNum),
          loopGsaOffset(loopGsaOffset)
    {
        type = CcuRepType::LOOP;
        instrCount = insGeneratorPtr_->GetInstrCount(type);
        supportCcuV1 = false; // loopParam按照A6格式填写，不适用A5
        supportCcuV2 = true;
    }

    void CcuRepLoop::ValidateInsGeneratorForLoop() const
    {
        CcuInsGeneratorV1* tmpPtrV1 = dynamic_cast<CcuInsGeneratorV1*>(insGeneratorPtr_);
        CcuInsGeneratorV2* tmpPtrV2 = dynamic_cast<CcuInsGeneratorV2*>(insGeneratorPtr_);
        if (tmpPtrV1 && !supportCcuV1) {
            // 在A5场景下没有使用A5的loop调用方式
            Hccl::THROW<Hccl::CcuApiException>("Cannot translate CcuRepLoop when supportCcuV1 is false!");
        }
        if (tmpPtrV2 && !supportCcuV2) {
            // 在A5场景下没有使用A6的loop调用方式
            Hccl::THROW<Hccl::CcuApiException>("Cannot translate CcuRepLoop when supportCcuV2 is false!");
        }
    }

    const std::string& CcuRepLoop::GetLabel() const { return label; }

    void CcuRepLoop::Reference(std::shared_ptr<CcuRepLoopBlock> refRep) { loopBlock = refRep; }

    std::shared_ptr<CcuRepBase> CcuRepLoop::SetLoopParam(Executor executor, Variable var)
    {
        return std::make_shared<CcuRepSetLoop>(insGeneratorPtr_, loopParam, executor, var);
    }

    bool CcuRepLoop::Translate(
        [[maybe_unused]] CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep)
    {
        (void)dep;
        ValidateInsGeneratorForLoop();

        this->instrId = instrId;
        translated = true;

        Hccl::CHECK_NULLPTR(loopBlock, "[CcuRepLoop::Translate] LoopBlock is nullptr!");

        if (!loopBlock->Translated()) {
            Hccl::THROW<Hccl::CcuApiException>("Reference To Invalid LoopBlock");
        }

        uint16_t startInstrId = loopBlock->StartInstrId();
        uint16_t loopBlockInstrCount = loopBlock->InstrCount();
        if (loopBlockInstrCount == 0) {
            HCCL_ERROR(
                "[CcuRepLoop][Translate] loopBlockInstrCount[%u] is 0, which causes underflow in endInstrId "
                "calculation.",
                loopBlockInstrCount);
            return false;
        }
        if (startInstrId > USHRT_MAX - loopBlockInstrCount) {
            HCCL_ERROR(
                "[CcuRepLoop][Translate] startInstrId[%u] + loopBlockInstrCount[%u] exceeds the maximum value of "
                "unsigned short int.",
                startInstrId, loopBlockInstrCount);
            return false;
        }

        if (instrId > USHRT_MAX - instrCount) {
            HCCL_ERROR("[CcuRepLoop][Translate] instrId[%u] exceeds the maximum value of unsigned short int.", instrId);
            return false;
        }

        uint16_t endInstrId = startInstrId + loopBlockInstrCount - 1;

        LoopInstr(instr++, startInstrId, endInstrId, loopParam.Id());

        instrId += instrCount;

        return translated;
    }

    std::string CcuRepLoop::Describe() { return Hccl::StringFormat("Loop reference to [%s]", label.c_str()); }

}; // namespace CcuRep
}; // namespace hcomm

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_rep_loopgroup_bundle_v1.h"
#include "ccu_ins_generator_base.h"
#include "ccu_api_exception.h"
#include "string_util.h"

namespace hcomm {
namespace CcuRep {

    using namespace Hccl;

    CcuRepLoopGroupBundle::CcuRepLoopGroupBundle(
        CcuInsGeneratorBase* insGenPtr, const CcuLoopGroupCfg& config, const Variable& parallelVar,
        const Variable& offsetVar)
        : insGenPtr_(insGenPtr),
          config_(config),
          parallelVar_(parallelVar),
          offsetVar_(offsetVar)
    {
        type = CcuRepType::LOOPGROUP;
    }

    CcuRepLoopGroupBundle::CcuRepLoopGroupBundle(
        CcuInsGeneratorBase* insGenPtr, const Variable& parallelVar, const Variable& offsetVar)
        : insGenPtr_(insGenPtr),
          config_{},
          parallelVar_(parallelVar),
          offsetVar_(offsetVar),
          layout_(Layout::PackedVar)
    {
        type = CcuRepType::LOOPGROUP;
    }

    void CcuRepLoopGroupBundle::AddLoop(const LoopEntry& entry) { loops_.push_back(entry); }

    uint16_t CcuRepLoopGroupBundle::LoopGroupInstrOffsetInBundle() const
    {
        uint16_t loopCount = static_cast<uint16_t>(loops_.size());
        uint16_t varBasedLoopCount = 0;
        for (const auto& loop : loops_) {
            if (loop.layout != Layout::Config) {
                varBasedLoopCount++;
            }
        }
        constexpr uint16_t kVarBasedLoopInstrCount = 2;
        constexpr uint16_t kConfigLayoutExtraInstrCount = 2;
        return (loopCount - varBasedLoopCount) + (varBasedLoopCount * kVarBasedLoopInstrCount)
               + (layout_ != Layout::Config ? 0 : kConfigLayoutExtraInstrCount);
    }

    uint16_t CcuRepLoopGroupBundle::GetStartLoopInstrId() const { return instrId + LoopGroupInstrOffsetInBundle() + 3; }

    uint16_t CcuRepLoopGroupBundle::InstrCount()
    {
        instrCount = insGenPtr_->CcuRepLoopGroupBundleInstrCount(this);
        return instrCount;
    }

    bool
    CcuRepLoopGroupBundle::Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep)
    {
        for (const auto& loop : loops_) {
            if (!loop.repLoopBlock->Translated()) {
                return false;
            }
        }

        this->instrId = instrId;
        translated = true;

        CHK_PRT_THROW(
            insGenPtr_->CcuRepLoopGroupBundleTranslate(ccuKernel, instr, instrId, this, dep)
                != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[CcuRepLoopGroupBundle][Translate] failed to translate for instrId[%u]", instrId),
            Hccl::CcuApiException, "CcuRepLoopGroupBundle translate failed");

        return translated;
    }

    std::string CcuRepLoopGroupBundle::Describe()
    {
        return Hccl::StringFormat("LoopGroupBundle[loops=%zu, totalLoopNum=%lu]", loops_.size(), totalLoopNum_);
    }

}; // namespace CcuRep
}; // namespace hcomm

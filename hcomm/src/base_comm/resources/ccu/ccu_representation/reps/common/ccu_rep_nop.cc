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

#include "string_util.h"
#include "ccu_api_exception.h"
#include "ccu_ins_generator_base.h"
#include "ccu_kernel.h"

namespace hcomm {
namespace CcuRep {

    using namespace Hccl;

    CcuRepNop::CcuRepNop(CcuInsGeneratorBase* insGeneratorPtr) : insGeneratorPtr_(insGeneratorPtr)
    {
        type = CcuRepType::NOP;
        instrCount = insGeneratorPtr_->GetInstrCount(type);
    }

    bool CcuRepNop::Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep)
    {
        this->instrId = instrId;
        translated = true;

        CHK_PRT_THROW(
            insGeneratorPtr_->CcuRepNopTranslate(ccuKernel, instr, instrId, this, dep) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[CcuRepNop][Translate] failed to translate for instrId[%u]", instrId), Hccl::CcuApiException,
            "CcuRepNop translate failed");

        instrId += instrCount;

        return translated;
    }

    std::string CcuRepNop::Describe() { return Hccl::StringFormat("Nop"); }

}; // namespace CcuRep
}; // namespace hcomm

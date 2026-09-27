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
#include "exception_util.h"
#include "ccu_api_exception.h"

#include "ccu_ins_generator_base.h"
#include "ccu_kernel.h"

namespace hcomm {
namespace CcuRep {

    using namespace Hccl;

    CcuRepLoad::CcuRepLoad(CcuInsGeneratorBase* insGenPtr, uint64_t addr, const Variable& var, uint32_t num)
        : insGeneratorPtr_(insGenPtr),
          var(var),
          addr(addr),
          num(num)
    {
        type = CcuRepType::LOAD;
        instrCount = insGeneratorPtr_->GetInstrCount(type); // 7: Load包含7条指令
    }

    bool CcuRepLoad::Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep)
    {
        Hccl::CHECK_NULLPTR(instr, "[HCcuRepLoad::Translate] instr is nullptr!");
        this->instrId = instrId;
        translated = true;
        CHK_PRT_THROW(
            insGeneratorPtr_->CcuRepLoadTranslate(ccuKernel, instr, instrId, this, dep) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[CcuRepLoad][Translate] failed to translate for instrId[%u]", instrId), Hccl::CcuApiException,
            "CcuRepLoad translate failed");
        CHK_PRT_THROW(
            (instrId > UINT16_MAX - instrCount),
            HCCL_ERROR(
                "[CcuRepLoad::Translate]uint16 integer overflow occurs, instrId = [%hu], instrCount = [%hu]", instrId,
                instrCount),
            Hccl::InternalException, "integer overflow");
        instrId += instrCount;

        return translated;
    }

    std::string CcuRepLoad::Describe() { return Hccl::StringFormat("Load([%llu], [%u], [%u])", addr, var.Id(), num); }

}; // namespace CcuRep
}; // namespace hcomm

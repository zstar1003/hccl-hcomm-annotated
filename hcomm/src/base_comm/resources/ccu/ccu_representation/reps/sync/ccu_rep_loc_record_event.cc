/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_rep_loc_record_event.h"
#include <climits>
#include "ccu_api_exception.h"
#include "ccu_ins_generator_base.h"

namespace hcomm {
namespace CcuRep {

    CcuRepLocRecordEvent::CcuRepLocRecordEvent(
        CcuInsGeneratorBase* insGenPtr, const CompletedEvent& event, uint32_t mask)
        : insGenPtr(insGenPtr),
          event_(event),
          mask_(mask)
    {
        type = CcuRepType::LOC_RECORD_EVENT;
        instrCount = insGenPtr->GetInstrCount(type);
    }

    bool CcuRepLocRecordEvent::Translate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, [[maybe_unused]] const TransDep& dep)
    {
        this->instrId = instrId;
        translated = true;

        CHK_PRT_THROW(
            insGenPtr->CcuRepLocRecordEventTranslate(ccuKernel, instr, this) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[CcuRepLocRecordEvent][Translate] failed to translate for instrId[%u]", instrId),
            Hccl::CcuApiException, "CcuRepLocRecordEvent translate failed");

        CHK_PRT_THROW(
            instrId > USHRT_MAX - instrCount,
            HCCL_ERROR(
                "[CcuRepLocRecordEvent][Translate] instrId[%u] + instrCount[%u] "
                "exceeds the maximum value of unsigned short int.",
                instrId, instrCount),
            Hccl::CcuApiException, "integer overflow");

        instrId += instrCount;
        return translated;
    }

    std::string CcuRepLocRecordEvent::Describe()
    {
        return Hccl::StringFormat("CcuRepLocRecordEvent=id[%u], mask[%04x]", event_.Id(), mask_);
    }

}; // namespace CcuRep
}; // namespace hcomm

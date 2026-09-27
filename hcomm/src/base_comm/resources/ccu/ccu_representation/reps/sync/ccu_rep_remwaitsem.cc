/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "string_util.h"
#include "hcomm_c_adpt.h"
#include "ccu_rep_v1.h"
#include "ccu_api_exception.h"
#include "ccu_ins_generator_v1.h"
#include "ccu_kernel.h"
#include "../../../../endpoint_pairs/channels/ccu/ccu_urma_channel.h"

namespace hcomm {
namespace CcuRep {

    CcuRepRemWaitSem::CcuRepRemWaitSem(
        CcuInsGeneratorBase* insGenPtr, const ChannelHandle channel, uint16_t semIndex, uint16_t mask, bool isProfiling)
        : insGenPtr(insGenPtr),
          channel(channel),
          semIndex(semIndex),
          mask(mask),
          isProfiling(isProfiling)
    {
        type = CcuRepType::REM_WAIT_SEM;
        instrCount = insGenPtr->GetInstrCount(type);
    }

    bool CcuRepRemWaitSem::Translate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, [[maybe_unused]] const TransDep& dep)
    {
        this->instrId = instrId;
        translated = true;

        CHK_PRT_THROW(
            insGenPtr->CcuRepRemWaitSemTranslate(ccuKernel, instr, this) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[CcuRepRemWaitSem][Translate] failed to translate for instrId[%u]", instrId),
            Hccl::CcuApiException, "CcuRepRemWaitSem translate failed");

        CHK_PRT_THROW(
            instrId > UINT16_MAX - instrCount,
            HCCL_ERROR(
                "[CcuRepRemWaitSem::Translate]uint16 integer overflow occurs, instrId = [%hu], instrCount = [%hu]",
                instrId, instrCount),
            Hccl::InternalException, "integer overflow");

        instrId += instrCount;

        return translated;
    }

    std::string CcuRepRemWaitSem::Describe()
    {
        return Hccl::StringFormat("Wait, Use semIndex[%u] and mask[%04x]", semIndex, mask);
    }

}; // namespace CcuRep
}; // namespace hcomm

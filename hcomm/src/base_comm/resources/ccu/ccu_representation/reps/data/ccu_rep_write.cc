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
#include "hcomm_c_adpt.h"
#include "ccu_api_exception.h"
#include "ccu_ins_generator_v1.h"
#include "../../../../endpoint_pairs/channels/ccu/ccu_urma_channel.h"
#include "ccu_kernel.h"
#include "ccu_ins_generator_base.h"

namespace hcomm {
namespace CcuRep {

    CcuRepWrite::CcuRepWrite(
        CcuInsGeneratorBase* insGenPtr, const ChannelHandle channel, RemoteAddr rem, LocalAddr loc, Variable len,
        CompletedEvent sem, uint16_t mask)
        : insGenPtr(insGenPtr),
          channel(channel),
          rem(rem),
          loc(loc),
          len(len),
          sem(sem),
          mask(mask)
    {
        type = CcuRepType::WRITE;
        instrCount = insGenPtr->GetInstrCount(type);
    }

    CcuRepWrite::CcuRepWrite(
        CcuInsGeneratorBase* insGenPtr, const ChannelHandle channel, RemoteAddr rem, LocalAddr loc, Variable len,
        uint16_t dataType, uint16_t opType, CompletedEvent sem, uint16_t mask)
        : insGenPtr(insGenPtr),
          channel(channel),
          rem(rem),
          loc(loc),
          len(len),
          sem(sem),
          mask(mask),
          dataType(dataType),
          opType(opType),
          reduceFlag(1)
    {
        type = CcuRepType::WRITE;
        instrCount = insGenPtr->GetInstrCount(type);
    }

    bool CcuRepWrite::Translate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, [[maybe_unused]] const TransDep& dep)
    {
        this->instrId = instrId;
        translated = true;

        CHK_PRT_THROW(
            insGenPtr->CcuRepWriteTranslate(ccuKernel, instr, this) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[CcuRepWrite][Translate] failed to translate for instrId[%u]", instrId), Hccl::CcuApiException,
            "CcuRepWrite translate failed");
        instrId += instrCount;

        return translated;
    }

    std::string CcuRepWrite::Describe()
    {
        return Hccl::StringFormat(
            "Write RemoteAddr[%u] to LocalAddr[%u], length[%u], set sem[%u] with mask[%04x], dataType[%u], opType[%u]",
            loc.addr.Id(), rem.addr.Id(), len.Id(), sem.Id(), mask, dataType, opType);
    }

}; // namespace CcuRep
}; // namespace hcomm

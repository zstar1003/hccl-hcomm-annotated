/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_rep_read_v1.h"
#include "ccu_rep_v1.h"

#include "string_util.h"
#include "ccu_api_exception.h"
#include "hcomm_c_adpt.h"

#include "../../../../endpoint_pairs/channels/ccu/ccu_urma_channel.h"
#include "ccu_ins_generator_base.h"
#include "ccu_kernel.h"

namespace hcomm {
namespace CcuRep {

    CcuRepRead::CcuRepRead(
        CcuInsGeneratorBase* insGenPtr, const ChannelHandle channel, LocalAddr loc, RemoteAddr rem, Variable len,
        CompletedEvent sem, uint16_t mask)
        : insGenPtr(insGenPtr),
          channel(channel),
          loc(loc),
          rem(rem),
          len(len),
          sem(sem),
          mask(mask)
    {
        type = CcuRepType::READ;
        instrCount = insGenPtr->GetInstrCount(type);
    }

    CcuRepRead::CcuRepRead(
        CcuInsGeneratorBase* insGenPtr, const ChannelHandle channel, LocalAddr loc, RemoteAddr rem, Variable len,
        uint16_t dataType, uint16_t opType, CompletedEvent sem, uint16_t mask)
        : insGenPtr(insGenPtr),
          channel(channel),
          loc(loc),
          rem(rem),
          len(len),
          sem(sem),
          mask(mask),
          dataType(dataType),
          opType(opType),
          reduceFlag(1)
    {
        type = CcuRepType::READ;
        instrCount = insGenPtr->GetInstrCount(type);
    }

    bool CcuRepRead::Translate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, [[maybe_unused]] const TransDep& dep)
    {
        this->instrId = instrId;
        translated = true;

        CHK_PRT_THROW(
            insGenPtr->CcuRepReadTranslate(ccuKernel, instr, this) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[CcuRepRead][Translate] failed to translate for instrId[%u]", instrId), Hccl::CcuApiException,
            "CcuRepRead translate failed");
        instrId += instrCount;

        return translated;
    }

    std::string CcuRepRead::Describe()
    {
        return Hccl::StringFormat(
            "Read LocalAddr[%u] To RemoteAddr[%u], length[%u], set sem[%u] with mask[%04x], dataType[%u], opType[%u]",
            rem.addr.Id(), loc.addr.Id(), len.Id(), sem.Id(), mask, dataType, opType);
    }

}; // namespace CcuRep
}; // namespace hcomm

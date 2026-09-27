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
#include "ccu_assist_v1.h"

#include "string_util.h"

#include "ccu_api_exception.h"
#include "hcomm_c_adpt.h"

#include "../../../../endpoint_pairs/channels/ccu/ccu_urma_channel.h"
#include "ccu_ins_generator_base.h"
#include "ccu_kernel.h"

namespace hcomm {
namespace CcuRep {

    CcuRepRemMem::CcuRepRemMem(CcuInsGeneratorBase* insGenPtr, const ChannelHandle channel, RemoteAddr rem)
        : insGenPtr(insGenPtr),
          channel(channel),
          rem(rem)
    {
        type = CcuRepType::REM_MEM;
        instrCount = insGenPtr->GetInstrCount(type);
    }

    bool CcuRepRemMem::Translate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, [[maybe_unused]] const TransDep& dep)
    {
        this->instrId = instrId;
        translated = true;

        instrCount = insGenPtr->GetInstrCount(type);
        CHK_PRT_THROW(
            insGenPtr->CcuRepRemMemTranslate(ccuKernel, instr, this) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[CcuRepRemMem][Translate] failed to translate for instrId[%u]", instrId), Hccl::CcuApiException,
            "CcuRepRemMem translate failed");
        instrId += instrCount;

        return translated;
    }

    std::string CcuRepRemMem::Describe()
    {
        return Hccl::StringFormat("Get Remote Buffer Addr and TokenInfo By Transport");
    }

}; // namespace CcuRep
}; // namespace hcomm

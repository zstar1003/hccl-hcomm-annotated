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
#include "ccu_rep_loc_wait_notify.h"
#include <climits>
#include "string_util.h"
#include "ccu_api_exception.h"
#include "ccu_ins_generator_v1.h"
#include "ccu_kernel.h"
namespace hcomm {
namespace CcuRep {

    CcuRepLocWaitNotify::CcuRepLocWaitNotify(
        CcuInsGeneratorBase* insGenPtr, const LocalNotify& notify, const uint32_t mask, bool isProfiling)
        : insGenPtr(insGenPtr),
          notify_(notify),
          mask_(mask),
          isProfiling_(isProfiling)
    {
        type = CcuRepType::LOC_WAIT_NOTIFY;
        instrCount = insGenPtr->GetInstrCount(type);
    }

    bool CcuRepLocWaitNotify::Translate(
        CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, [[maybe_unused]] const TransDep& dep)
    {
        this->instrId = instrId;
        translated = true;

        CHK_PRT_THROW(
            insGenPtr->CcuRepLocWaitNotifyTranslate(ccuKernel, instr, this) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[CcuRepLocWaitNotify][Translate] failed to translate for instrId[%u]", instrId),
            Hccl::CcuApiException, "CcuRepLocWaitNotify translate failed");
        CHK_PRT_THROW(
            (instrId > UINT16_MAX - instrCount),
            HCCL_ERROR(
                "[CcuRepLocWaitNotify::Translate]uint16 integer overflow occurs, "
                "instrId = [%hu], instrCount = [%hu]",
                instrId, instrCount),
            Hccl::CcuApiException, "integer overflow");

        instrId += instrCount;
        return translated;
    }

    std::string CcuRepLocWaitNotify::Describe()
    {
        return Hccl::StringFormat("CcuRepLocWaitNotify=id[%u], mask[%04x]", notify_.Id(), mask_);
    }

}; // namespace CcuRep
}; // namespace hcomm

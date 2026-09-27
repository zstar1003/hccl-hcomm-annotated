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
#include "ccu_ins_generator_v1.h"
#include "ccu_api_exception.h"
#include "ccu_kernel.h"
namespace hcomm {
namespace CcuRep {

    CcuRepRecordSharedNotify::CcuRepRecordSharedNotify(
        CcuInsGeneratorBase* insGenPtr, const LocalNotify& notify, uint16_t mask)
        : insGenPtr(insGenPtr),
          notify_(notify),
          mask_(mask)
    {
        type = CcuRepType::RECORD_SHARED_NOTIFY;
        instrCount = insGenPtr->GetInstrCount(type);
    }

    bool
    CcuRepRecordSharedNotify::Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep)
    {
        this->instrId = instrId;
        translated = true;

        CHK_PRT_THROW(
            insGenPtr->CcuRepRecordSharedNotifyTranslate(ccuKernel, instr, this, dep) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[CcuRepRecordSharedNotify][Translate] failed to translate for instrId[%u]", instrId),
            Hccl::CcuApiException, "CcuRepRecordSharedNotify translate failed");
        CHK_PRT_THROW(
            (instrId > UINT16_MAX - instrCount),
            HCCL_ERROR(
                "[CcuRepRecordSharedNotify::Translate]uint16 integer overflow occurs, "
                "instrId = [%hu], instrCount = [%hu]",
                instrId, instrCount),
            Hccl::InternalException, "integer overflow");
        instrId += instrCount;

        return translated;
    }

    std::string CcuRepRecordSharedNotify::Describe()
    {
        return Hccl::StringFormat("Post, Use semIndex[%u] and mask[%04x]", notify_.Id(), mask_);
    }

}; // namespace CcuRep
}; // namespace hcomm

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
#include "exception_util.h"
#include "ccu_api_exception.h"
#include "ccu_ins_generator_base.h"
#include "internal_exception.h"

namespace hcomm {
namespace CcuRep {

    CcuRepWriteWithCntInc::CcuRepWriteWithCntInc(
        CcuInsGeneratorBase* insGenPtr, Variable& channel, RemoteAddr& rem, LocalAddr& loc, Variable& len,
        RemoteAddr& incCnt)
        : insGenPtr(insGenPtr),
          channel(channel),
          rem(rem),
          loc(loc),
          len(len),
          incCnt(incCnt)
    {
        type = CcuRepType::WRITE_WITH_CNT_INC;
        instrCount = insGenPtr->GetInstrCount(type);
    }

    bool
    CcuRepWriteWithCntInc::Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep)
    {
        this->instrId = instrId;
        translated = true;

        CHK_PRT_THROW(
            insGenPtr->CcuRepWriteWithCntIncTranslate(ccuKernel, instr, instrId, this, dep) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[%s] failed to translate for instrId[%u]", __func__, instrId), Hccl::CcuApiException,
            "CcuRepWriteWithCntInc translate failed");
        CHK_PRT_THROW(
            (instrId > UINT16_MAX - instrCount),
            HCCL_ERROR(
                "[%s]uint16 integer overflow occurs, instrId = [%hu], instrCount = [%hu]", __func__, instrId,
                instrCount),
            Hccl::InternalException, "integer overflow");
        instrId += instrCount;

        return translated;
    }

    std::string CcuRepWriteWithCntInc::Describe()
    {
        return Hccl::StringFormat(
            "Write With Cnt Inc, rem[%u] to loc[%u], len[%u], incCntAddr[%u]", rem.addr.Id(), loc.addr.Id(), len.Id(),
            incCnt.addr.Id());
    }

}; // namespace CcuRep
}; // namespace hcomm

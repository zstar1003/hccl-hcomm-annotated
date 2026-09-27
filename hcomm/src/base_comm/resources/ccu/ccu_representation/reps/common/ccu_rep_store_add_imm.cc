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

    CcuRepStoreAddImm::CcuRepStoreAddImm(
        CcuInsGeneratorBase* insGenPtr, Variable& dst, uint16_t dstNum, Variable& dstOffset, uint16_t immAddValue,
        Variable& src)
        : insGenPtr(insGenPtr),
          dst(dst),
          dstOffset(dstOffset),
          src(src),
          immAddValue(immAddValue),
          dstNum(dstNum)
    {
        type = CcuRepType::STORE_ADD_IMM;
        instrCount = insGenPtr->GetInstrCount(type);
    }

    bool CcuRepStoreAddImm::Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep)
    {
        this->instrId = instrId;
        translated = true;

        CHK_PRT_THROW(
            insGenPtr->CcuRepStoreAddImmTranslate(ccuKernel, instr, instrId, this, dep) != HcclResult::HCCL_SUCCESS,
            HCCL_ERROR("[%s] failed to translate for instrId[%u]", __func__, instrId), Hccl::CcuApiException,
            "CcuRepStoreAddImm translate failed");
        CHK_PRT_THROW(
            (instrId > UINT16_MAX - instrCount),
            HCCL_ERROR(
                "[%s]uint16 integer overflow occurs, instrId = [%hu], instrCount = [%hu]", __func__, instrId,
                instrCount),
            Hccl::InternalException, "integer overflow");
        instrId += instrCount;

        return translated;
    }

    std::string CcuRepStoreAddImm::Describe()
    {
        return Hccl::StringFormat(
            "Store Add Imm, dst[%u] dstOffset[%u] src[%u] immAddValue[%u]", dst.Id(), dstOffset.Id(), src.Id(),
            immAddValue);
    }

}; // namespace CcuRep
}; // namespace hcomm

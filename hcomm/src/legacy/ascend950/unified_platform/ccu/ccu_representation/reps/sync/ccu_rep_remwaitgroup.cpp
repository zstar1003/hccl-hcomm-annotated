/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_rep.h"

#include "string_util.h"

namespace Hccl {
namespace CcuRep {

    CcuRepWaitGroup::CcuRepWaitGroup(
        const CcuTransportGroup& transportGroup, uint16_t semIndex, uint16_t mask, bool isProfiling)
        : transportGroup(transportGroup),
          semIndex(semIndex),
          mask(mask),
          isProfiling(isProfiling)
    {
        type = CcuRepType::REM_WAIT_GROUP;
        instrCount = 1;
    }

    bool CcuRepWaitGroup::Translate(CcuInstr*& instr, uint16_t& instrId, [[maybe_unused]] const TransDep& dep)
    {
        this->instrId = instrId;
        translated = true;

        u32 cntCkeId = 0;
        HcclResult ret = transportGroup.GetCntCkeId(semIndex, cntCkeId);
        if (ret != HcclResult::HCCL_SUCCESS) {
            string msg = StringFormat(
                "[Translate]rt get CntCkeId failed. "
                "semIndex[%u], cntCkeId[%u] return[%d].",
                semIndex, cntCkeId, ret);
            MACRO_THROW(CcuApiException, msg);
        }
        // 需要profiling的使用SetCKEInstr, 否则使用ClearCKEInstr
        if (isProfiling) {
            SetCKEInstr(instr++, 0, 0, cntCkeId, mask, 1);
        } else {
            ClearCKEInstr(instr++, 0, 0, cntCkeId, mask, 1);
        }
        CHK_PRT_THROW(
            (instrId > UINT16_MAX - instrCount),
            HCCL_ERROR(
                "[CcuRepWaitGroup::Translate]uint16 integer overflow occurs, instrId = [%hu], instrCount = [%hu]",
                instrId, instrCount),
            InternalException, "integer overflow");
        instrId += instrCount;

        return translated;
    }

    std::string CcuRepWaitGroup::Describe()
    {
        return StringFormat("Wait, Use semIndex[%u] and mask[%04x]", semIndex, mask);
    }

}; // namespace CcuRep
}; // namespace Hccl

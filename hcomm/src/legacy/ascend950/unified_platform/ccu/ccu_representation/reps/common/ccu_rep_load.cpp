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
#include "ccu_res_specs_legacy.h"
#include "string_util.h"

namespace Hccl {
namespace CcuRep {

    CcuRepLoad::CcuRepLoad(uint64_t addr, const Variable& var, uint32_t num) : var(var), addr(addr), num(num)
    {
        type = CcuRepType::LOAD;
        instrCount = 7; // 7: Load包含7条指令
    }

    bool CcuRepLoad::Translate(CcuInstr*& instr, uint16_t& instrId, const TransDep& dep)
    {
        CHECK_NULLPTR(instr, "[HCcuRepLoad::Translate] instr is nullptr!");
        this->instrId = instrId;
        translated = true;
        uint64_t varAddr = dep.xnBaseAddr + CCU_RESOURCE_XN_PER_SIZE * var.Id();

        LoadImdToGSAInstr(instr++, dep.commGsa[0], varAddr);
        LoadImdToGSAInstr(instr++, dep.commGsa[1], addr);
        LoadImdToXnInstr(instr++, dep.commXn[0], dep.ccuResSpaceTokenInfo, CCU_LOAD_TO_XN_SEC_INFO);
        LoadImdToXnInstr(instr++, dep.commXn[1], dep.memTokenInfo, CCU_LOAD_TO_XN_SEC_INFO);
        LoadImdToXnInstr(instr++, dep.commXn[2], CCU_RESOURCE_XN_PER_SIZE * num);
        TransLocMemToLocMemInstr(
            instr++, dep.commGsa[0], dep.commXn[0], dep.commGsa[1], dep.commXn[1], dep.commXn[2],
            dep.reserveChannalId[0], dep.commSignal, mask, 0, 0, 1, 1);
        SetCKEInstr(instr++, 0, 0, dep.commSignal, mask, 1);
        CHK_PRT_THROW(
            (instrId > UINT16_MAX - instrCount),
            HCCL_ERROR(
                "[CcuRepLoad::Translate]uint16 integer overflow occurs, instrId = [%hu], instrCount = [%hu]", instrId,
                instrCount),
            InternalException, "integer overflow");
        instrId += instrCount;

        return translated;
    }

    std::string CcuRepLoad::Describe() { return StringFormat("Load([%llu], [%u], [%u])", addr, var.Id(), num); }

}; // namespace CcuRep
}; // namespace Hccl

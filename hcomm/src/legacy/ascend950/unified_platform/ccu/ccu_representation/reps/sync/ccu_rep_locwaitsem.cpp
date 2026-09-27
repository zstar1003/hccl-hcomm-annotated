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
#include <climits>

#include "string_util.h"

namespace Hccl {
namespace CcuRep {

    CcuRepLocWaitSem::CcuRepLocWaitSem(const MaskSignal& sem, uint16_t mask, bool isProfiling)
        : sem(sem),
          mask(mask),
          isProfiling(isProfiling)
    {
        type = CcuRepType::LOC_WAIT_SEM;
        instrCount = 1;
    }

    uint16_t CcuRepLocWaitSem::GetSemId() const { return sem.Id(); }

    void CcuRepLocWaitSem::SetDependencyInfo(
        const std::unordered_map<uint32_t, std::vector<std::shared_ptr<CcuRepBase>>>& depInfo)
    {
        depInfo_ = depInfo;
    }

    std::vector<std::shared_ptr<CcuRepBase>> CcuRepLocWaitSem::GetDependencyInfo(uint32_t bit)
    {
        // 查找给定 bit 是否存在于 depInfo_ 中
        auto it = depInfo_.find(bit);
        // 如果找到 bit，返回与之关联的 vector
        if (it != depInfo_.end()) {
            return it->second;
        }
        // 如果未找到 bit，返回一个空的 vector
        return std::vector<std::shared_ptr<CcuRepBase>>();
    }

    bool CcuRepLocWaitSem::Translate(CcuInstr*& instr, uint16_t& instrId, [[maybe_unused]] const TransDep& dep)
    {
        this->instrId = instrId;
        translated = true;

        // 需要profiling的使用SetCKEInstr, 否则使用ClearCKEInstr
        if (isProfiling) {
            SetCKEInstr(instr++, 0, 0, sem.Id(), mask, 1);
        } else {
            ClearCKEInstr(instr++, 0, 0, sem.Id(), mask, 1);
        }

        if (instrId > USHRT_MAX - instrCount) {
            THROW<InternalException>(StringFormat(
                "[CcuRepLocWaitSem][Translate] instrId[%u] + instrCount[%u] exceeds the "
                "maximum value of unsigned short int.",
                instrId, instrCount));
        }
        CHK_PRT_THROW(
            (instrId > UINT16_MAX - instrCount),
            HCCL_ERROR(
                "[CcuRepLocWaitSem::Translate]uint16 integer overflow occurs, instrId = [%hu], instrCount = [%hu]",
                instrId, instrCount),
            InternalException, "integer overflow");
        instrId += instrCount;

        return translated;
    }

    std::string CcuRepLocWaitSem::Describe() { return StringFormat("Wait Sem[%u], mask[%04x]", sem.Id(), mask); }

}; // namespace CcuRep
}; // namespace Hccl

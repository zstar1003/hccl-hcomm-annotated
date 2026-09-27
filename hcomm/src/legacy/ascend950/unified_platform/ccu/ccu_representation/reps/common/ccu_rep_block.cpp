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

    CcuRepBlock::CcuRepBlock(const std::string& label) : label(label) { type = CcuRepType::BLOCK; }

    std::vector<std::shared_ptr<CcuRepBase>>& CcuRepBlock::GetReps() { return repVec; }

    void CcuRepBlock::Append(std::shared_ptr<CcuRepBase> rep) { repVec.push_back(rep); }

    const std::string& CcuRepBlock::GetLabel() const { return label; }

    uint16_t CcuRepBlock::InstrCount()
    {
        instrCount = 0;
        for (const auto& repInBlock : repVec) {
            instrCount += repInBlock->InstrCount();
        }
        return instrCount;
    }

    bool CcuRepBlock::Translate(CcuInstr*& instr, uint16_t& instrId, const TransDep& dep)
    {
        this->instrId = instrId;
        translated = true;

        for (const auto& repInBlock : GetReps()) {
            repInBlock->Translate(instr, instrId, dep);
        }

        return translated;
    }

    std::string CcuRepBlock::Describe() { return StringFormat("RepBlock"); }

    std::shared_ptr<CcuRepBase> CcuRepBlock::GetRepByInstrId(uint16_t instrId)
    {
        for (const auto& rep : GetReps()) {
            const uint16_t repInstrCount = rep->InstrCount();
            if (repInstrCount == 0) {
                continue;
            }
            const uint16_t startId = rep->StartInstrId();
            const uint16_t endId = startId + repInstrCount - 1;
            if (instrId >= startId && instrId <= endId) {
                return rep;
            }
        }
        return nullptr;
    }

}; // namespace CcuRep
}; // namespace Hccl

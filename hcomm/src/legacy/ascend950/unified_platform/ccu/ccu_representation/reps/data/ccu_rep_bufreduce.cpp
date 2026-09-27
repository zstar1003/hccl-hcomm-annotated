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

    CcuRepBufReduce::CcuRepBufReduce(
        const std::vector<CcuBuffer>& mem, uint16_t count, uint16_t dataType, uint16_t outputDataType, uint16_t opType,
        MaskSignal sem, const CcuRep::Variable& len, uint16_t mask)
        : mem(mem),
          count(count),
          dataType(dataType),
          outputDataType(outputDataType),
          opType(opType),
          sem(sem),
          xnIdLength_(len),
          mask(mask)
    {
        type = CcuRepType::BUF_REDUCE;
        instrCount = 1;
    }

    bool CcuRepBufReduce::Translate(CcuInstr*& instr, uint16_t& instrId, [[maybe_unused]] const TransDep& dep)
    {
        this->instrId = instrId;
        translated = true;

        if (count > CCU_REDUCE_MAX_MS || mem.size() > CCU_REDUCE_MAX_MS) {
            THROW<CcuApiException>("count and mem size must be less than %u", CCU_REDUCE_MAX_MS);
        }
        if (count < CCU_REDUCE_MIN_MS) {
            THROW<Hccl::CcuApiException>("count must be at least %u", CCU_REDUCE_MIN_MS);
        }

        // 这里需要注意，在数据格式膨胀的情况下，需要传入用来存放输出的MSId
        // 特别是2P场景，输入MS的数目为2，但是在8bit进，32bit出的场景，输出MS的数目为4
        // 传入的MS中已经包含了需要使用的输入输出的最大量，因此，这里应该直接去MS的size
        uint16_t msId[CCU_REDUCE_MAX_MS] = {0};
        for (uint16_t i = 0; i < mem.size(); i++) {
            msId[i] = mem[i].Id();
        }

        if (opType == CCU_REDUCE_SUM) {
            if (outputDataType == 1) { // 1是fp16
                AddInstr(instr++, msId, count, outputDataType, dataType, sem.Id(), mask, 0, 0, 1, xnIdLength_.Id());
            } else if (outputDataType == 2) { // 2是bf16
                AddInstr(instr++, msId, count, outputDataType, dataType, sem.Id(), mask, 0, 0, 1, xnIdLength_.Id());
            } else {
                AddInstr(instr++, msId, count, 0, dataType, sem.Id(), mask, 0, 0, 1, xnIdLength_.Id());
            }
        } else if (opType == CCU_REDUCE_MAX) {
            MaxInstr(instr++, msId, count, dataType, sem.Id(), mask, 0, 0, 1, xnIdLength_.Id());
        } else if (opType == CCU_REDUCE_MIN) {
            MinInstr(instr++, msId, count, dataType, sem.Id(), mask, 0, 0, 1, xnIdLength_.Id());
        }
        instrId += instrCount;

        return translated;
    }

    std::string CcuRepBufReduce::Describe() { return StringFormat("Reduce"); }

}; // namespace CcuRep
}; // namespace Hccl

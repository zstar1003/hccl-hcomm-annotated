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

    CcuRepLoopGroup::CcuRepLoopGroup(const Variable& parallelParam, const Variable& offsetParam)
        : parallelParam(parallelParam),
          offsetParam(offsetParam)
    {
        type = CcuRepType::LOOPGROUP;
        instrCount = 1;
    }

    bool CcuRepLoopGroup::Translate(CcuInstr*& instr, uint16_t& instrId, [[maybe_unused]] const TransDep& dep)
    {
        this->instrId = instrId;
        translated = true;

        constexpr uint16_t INSTR_ID_RESERVE_SIZE = 3;
        if (instrId > USHRT_MAX - INSTR_ID_RESERVE_SIZE) {
            HCCL_ERROR(
                "[CcuRepLoopGroup][Translate] instrId[%u] + 3 exceeds the maximum value of unsigned short int.",
                instrId);
            return false;
        }
        // 这是个非常危险的操作，需要谨慎使用
        // 依赖于LoopGroupCall的实现
        // LoopGroup所指向的Loop的位置为当前指令Id + 3
        LoopGroupInstr(instr++, instrId + 3, parallelParam.Id(), offsetParam.Id(), 0);

        instrId += instrCount;

        return translated;
    }

    std::string CcuRepLoopGroup::Describe() { return StringFormat("LoopGroup"); }

    std::shared_ptr<CcuRepBase> CcuRepLoopGroup::SetParallelParam(Variable var)
    {
        return std::make_shared<CcuRepAssign>(parallelParam, var);
    }

    std::shared_ptr<CcuRepBase> CcuRepLoopGroup::SetOffsetParam(Variable var)
    {
        return std::make_shared<CcuRepAssign>(offsetParam, var);
    }

    uint16_t CcuRepLoopGroup::GetStartLoopInstrId() const
    {
        // 这是个非常危险的操作，需要谨慎使用
        // 依赖于LoopGroupCall的实现
        // LoopGroup所指向的Loop的位置为当前指令Id + 3
        return this->instrId + 3;
    }

}; // namespace CcuRep
}; // namespace Hccl

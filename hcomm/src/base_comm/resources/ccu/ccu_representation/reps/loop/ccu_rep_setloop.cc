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
#include "ccu_assist_v1.h"
#include <climits>

#include "string_util.h"
#include "exception_util.h"
#include "ccu_api_exception.h"
#include "ccu_ins_generator_base.h"
#include "ccu_ins_generator_v1.h"
#include "ccu_kernel.h"

namespace hcomm {
namespace CcuRep {

    using namespace Hccl;

    CcuRepSetLoop::CcuRepSetLoop(
        CcuInsGeneratorBase* insGeneratorPtr, const Variable& loopParam, const Executor& executor, const Variable& var)
        : insGeneratorPtr_(insGeneratorPtr),
          loopParam(loopParam),
          executor(executor),
          var(var)
    {
        type = CcuRepType::SET_LOOP;
        instrCount = insGeneratorPtr_->GetInstrCount(type); // set loop 指令数量为2
    }

    bool CcuRepSetLoop::Translate(
        [[maybe_unused]] CcuKernel* ccuKernel, [[maybe_unused]] CcuInstr*& instr, uint16_t& instrId,
        [[maybe_unused]] const TransDep& dep)
    {
        this->instrId = instrId;
        translated = true;

        if (instrId > USHRT_MAX - instrCount) {
            Hccl::THROW<Hccl::InternalException>(Hccl::StringFormat(
                "[CcuRepSetLoop][Translate] instrId[%u] + instrCount[%u] exceeds the "
                "maximum value of unsigned short int.",
                instrId, instrCount));
        }
        instrId += instrCount;

        return translated;
    }

    std::string CcuRepSetLoop::Describe()
    {
        return Hccl::StringFormat(
            "loopParam[%u] = var[%u], execute on LoopEngine[%u]", loopParam.Id(), var.Id(), executor.Id());
    }

}; // namespace CcuRep
}; // namespace hcomm

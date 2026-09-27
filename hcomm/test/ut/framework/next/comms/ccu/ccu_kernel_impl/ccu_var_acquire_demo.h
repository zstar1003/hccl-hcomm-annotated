/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_primitives.hpp"
#include "ccu_types.h"

namespace ccu = ::AscendC::ccu;

struct CcuVarAcquireDemoKernelArg {
    CcuVariableHandle acqHandle{0};
    uint32_t varNum{0};
    CcuEventHandle acqEventHandle{0};
    uint32_t eventNum{0};
};

CcuResult CcuVarAcquireDemoKernel(CcuKernelArg arg)
{
    auto* args = static_cast<CcuVarAcquireDemoKernelArg*>(arg);

    ccu::Variable sum;
    sum = 0;
    ccu::Array<ccu::Variable> vars(args->acqHandle, args->varNum);
    for (uint32_t i = 0; i < vars.size(); i++) {
        vars[i] = i;
        sum += vars[i];
    }

    ccu::Array<ccu::Event> events(args->acqEventHandle, args->eventNum);
    for (uint32_t i = 0; i < events.size(); i++) {
        ccu::EventRecord(events[i]);
        ccu::EventWait(events[i]);
    }
    return CcuResult::CCU_SUCCESS;
}

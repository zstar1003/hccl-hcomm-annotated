/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_IMM_CASC_DEMO_H
#define CCU_IMM_CASC_DEMO_H

#include "ccu_primitives.hpp"
#include "ccu_types.h"
#include <cstdint>

namespace ccu = ::AscendC::ccu;
struct CcuCascCntDemoArg {
    HcommCcuCascCntHandle cntHandle{0};
    uint64_t tgtValue{0};
};

CcuResult CcuLoadAddImmDemoKernel(CcuKernelArg arg)
{
    (void)arg;
    ccu::Array<ccu::Variable> base(20);
    ccu::Variable offset;
    ccu::Variable dst;
    base[0] = 64;
    offset = 8;
    dst = 0;

    return ccu::LoadAddImm(base, offset, 1, dst);
}

CcuResult CcuAddImmStoreDemoKernel(CcuKernelArg arg)
{
    (void)arg;
    ccu::Array<ccu::Variable> base(20);
    ccu::Variable offset;
    ccu::Variable src;
    base[0] = 64;
    offset = 8;
    src = 32;
    return ccu::AddImmStore(base, offset, 16, src);
}

CcuResult CcuWriteVarAtomicAddDemoKernel(CcuKernelArg arg)
{
    (void)arg;
    ccu::Variable channelId;
    ccu::RemoteAddr varAddr;
    ccu::Variable addValue;
    ccu::Event event;
    channelId = 0;
    varAddr.addr = 0x30000000;
    varAddr.token = 0;
    addValue = 1;
    return ccu::WriteVarAtomicAdd(channelId, varAddr, addValue, event);
}

CcuResult CcuWriteWithCascCntIncDemoKernel(CcuKernelArg arg)
{
    (void)arg;
    ccu::Variable channelId;
    ccu::RemoteAddr remote;
    ccu::LocalAddr local;
    ccu::Variable len;
    ccu::RemoteAddr inCntAddr;
    channelId = 0;
    remote.addr = 0x30000000;
    remote.token = 0;
    local.addr = 0x50000000;
    local.token = 0;
    len = 1024;
    inCntAddr.addr = 0x70000000;
    inCntAddr.token = 0;
    return ccu::WriteWithCascCntInc(channelId, remote, local, len, inCntAddr);
}

CcuResult CcuCascCntWaitDemoKernel(CcuKernelArg arg)
{
    auto* p = static_cast<CcuCascCntDemoArg*>(arg);
    return ccu::CascCntWait(p->cntHandle, p->tgtValue);
}

CcuResult CcuCascCntClearDemoKernel(CcuKernelArg arg)
{
    auto* p = static_cast<CcuCascCntDemoArg*>(arg);
    ccu::Event event;
    return ccu::CascCntClear(p->cntHandle, event);
}

#endif // CCU_IMM_CASC_DEMO_H

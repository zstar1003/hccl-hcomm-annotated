/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adapter_hccp.h"
#include "ccu_primitives.hpp"
#include "ccu_types.h"
#include "ccu_log.h"
#include <vector>

namespace ccu = ::AscendC::ccu;

struct CcuVarAddKernelArg {
    uint64_t numA{0xffffffff};
    uint32_t numB{0};
    ChannelHandle channelHandle{0};
};

void test_method(std::vector<ccu::RemoteAddr>& loopsrc)
{
    for (int i = 0; i < 1; i++) {
        ccu::LocalAddr localAddr1;
        loopsrc.emplace_back(*reinterpret_cast<ccu::RemoteAddr*>(&localAddr1));
    }
}
CcuResult CcuAssignDemoKernel(CcuKernelArg arg)
{
    auto* args = static_cast<CcuVarAddKernelArg*>(arg);
    ccu::RemoteAddr remoteAddr;
    remoteAddr.addr = 0x10000000;
    remoteAddr.token = 0x20000000;
    std::vector<ccu::RemoteAddr> loopsrc;
    test_method(loopsrc);
    loopsrc[0].addr = remoteAddr.addr;
    loopsrc[0].token = remoteAddr.token;
    return CcuResult::CCU_SUCCESS;
}
CcuResult CcuAllocDemoKernel(CcuKernelArg arg)
{
    auto* args = static_cast<CcuVarAddKernelArg*>(arg);
    ccu::Variable varA, varB, result;
    varA = 1024;
    varB = 2048;
    result = varA + varB;
    ccu::LoadArg(varA, 0);
    ccu::LoadArg(varB, 1);

    ccu::Variable varC = ccu::GetResByChannel<ccu::Variable>(args->channelHandle, 0);
    varA = varC;

    ccu::Address addrA, addrB, addrResult;
    addrA = 0x80000000;
    addrB = 0x90000000;
    addrResult = addrA + addrB;

    ccu::Event evt;
    ccu::EventRecord(evt);
    ccu::EventWait(evt);

    ccu::Array<ccu::Event> evt2(2);
    ccu::EventRecord(evt2[0]);
    ccu::EventWait(evt2[0]);
    ccu::EventRecord(evt2[1]);
    ccu::EventWait(evt2[1]);

    return CcuResult::CCU_SUCCESS;
}
CcuResult CcuAddrArithV2DemoKernel(CcuKernelArg arg)
{
    (void)arg;
    ccu::Address addrA, addrB, addrC;
    ccu::Variable varA, varB, varC;
    addrA = 0x1000;
    addrB = 0x2000;
    varA = 3;
    varB = 5;
    const uint16_t imm = 0x40;

    addrC = addrA + imm;
    return CcuResult::CCU_SUCCESS;
}
CcuResult CcuLocalAddrDemoKernel(CcuKernelArg arg)
{
    auto* args = static_cast<CcuVarAddKernelArg*>(arg);
    ccu::LocalAddr localAddr;
    localAddr.addr = 0x10000000;
    localAddr.token = 0x20000000;

    ccu::LocalAddr localAddr2;
    localAddr2 = localAddr;
    return CcuResult::CCU_SUCCESS;
}
CcuResult CcuRemoteAddrDemoKernel(CcuKernelArg arg)
{
    auto* args = static_cast<CcuVarAddKernelArg*>(arg);
    ccu::RemoteAddr remoteAddr;
    remoteAddr.addr = 0x10000000;
    remoteAddr.token = 0x20000000;

    ccu::RemoteAddr remoteAddr2;
    remoteAddr2 = remoteAddr;
    return CcuResult::CCU_SUCCESS;
}
CcuResult CcuLoadStoreDemoKernel(CcuKernelArg arg)
{
    auto* args = static_cast<CcuVarAddKernelArg*>(arg);
    ccu::Variable varA, varB, result;
    ccu::Variable srcAddr, dstAddr;
    srcAddr = 0x30000000;
    dstAddr = 0x40000000;
    ccu::Load(0x10000000, varA);
    ccu::Store(0x20000000, varB);
    ccu::Load(srcAddr, varA);
    ccu::Store(dstAddr, varB);
    ccu::Array<ccu::Variable> varArr(2);
    ccu::Array<ccu::Variable> varArr2(2);
    ccu::Load(0x10000000, varArr, 2);
    ccu::Store(0x20000000, varArr2, 2);
    ccu::Load(srcAddr, varArr, 2);
    ccu::Store(dstAddr, varArr2, 2);
    return CcuResult::CCU_SUCCESS;
}
CcuResult CcuNotifyDemoKernel(CcuKernelArg arg)
{
    auto* args = static_cast<CcuVarAddKernelArg*>(arg);
    // mask 已与 Event 解耦，统一作为 API 参数显式传入。
    ccu::Array<ccu::Event> evt(3);
    ccu::EventRecord(evt[0], 0x12);
    ccu::EventWait(evt[0], 0x12);
    ccu::EventRecord(evt[1], 0x13);
    ccu::EventWait(evt[1], 0x13);
    ccu::EventRecord(evt[2], 0x14);
    ccu::EventWait(evt[2], 0x14);

    ccu::NotifyRecord(args->channelHandle, 0, 0x12);
    ccu::NotifyWait(args->channelHandle, 0, 0x12);
    ccu::Variable varA;
    varA = 1024;
    ccu::WriteVariableWithNotify(args->channelHandle, varA, 0, 0, 0x12);
    ccu::NotifyWait(args->channelHandle, 0, 0x12);
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuLocalNotifyDemoKernel(CcuKernelArg arg)
{
    (void)arg;
    ccu::EventRecord("local_notify_tag_default");
    ccu::EventWait("local_notify_tag_default");

    ccu::EventRecord("local_notify_tag_a", 0x12);
    ccu::EventWait("local_notify_tag_a", 0x12);
    CcuLocalNotifyRecord("local_notify_tag_b", 0x34);
    CcuLocalNotifyWait("local_notify_tag_b", 0x34);
    return CcuResult::CCU_SUCCESS;
}
CcuResult CcuLocalCopyKernel(CcuKernelArg arg)
{
    auto* args = static_cast<CcuVarAddKernelArg*>(arg);
    ccu::LocalAddr src, dst;
    src.addr = 0x10000000;
    src.token = 0x20000000;
    dst.addr = 0x30000000;
    dst.token = 0x40000000;
    ccu::Event evt;
    ccu::CcuBuffer buf;
    ccu::Variable len;
    len = 1024;
    ccu::LocalCopy(buf, src, len, evt);
    ccu::EventWait(evt);
    ccu::LocalCopy(dst, buf, len, evt);
    ccu::EventWait(evt);
    ccu::Array<ccu::CcuBuffer> buffer(1);
    ccu::LocalCopy(buffer[0], src, len, evt);
    ccu::EventWait(evt);
    ccu::LocalCopy(dst, buffer[0], len, evt);
    ccu::EventWait(evt);
    ccu::LocalCopy(dst, src, len, evt);
    ccu::EventWait(evt);
    return CcuResult::CCU_SUCCESS;
}
CcuResult CcuLocalReduceKernel(CcuKernelArg arg)
{
    auto* args = static_cast<CcuVarAddKernelArg*>(arg);
    ccu::LocalAddr src, dst;
    src.addr = 0x10000000;
    src.token = 0x20000000;
    dst.addr = 0x30000000;
    dst.token = 0x40000000;
    ccu::Variable len;
    len = 1024;
    ccu::Event evt;
    ccu::LocalReduce(dst, src, len, HCCL_DATA_TYPE_FP16, HCCL_REDUCE_SUM, evt);
    ccu::EventWait(evt);
    ccu::Array<ccu::CcuBuffer> buf(2);
    ccu::LocalReduce(buf.data(), 2, HCCL_DATA_TYPE_FP16, HCCL_DATA_TYPE_FP16, HCCL_REDUCE_SUM, len, evt);
    ccu::EventWait(evt);

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuRemoteReadKernel(CcuKernelArg arg)
{
    auto* args = static_cast<CcuVarAddKernelArg*>(arg);
    ccu::RemoteAddr dst;
    ccu::LocalAddr src;
    src.addr = 0x10000000;
    src.token = 0x20000000;
    dst.addr = 0x30000000;
    dst.token = 0x40000000;
    ccu::Variable len;
    len = 1024;
    ccu::Event evt;
    ccu::Read(args->channelHandle, src, dst, len, evt);
    ccu::EventWait(evt);
    ccu::Array<ccu::CcuBuffer> buf(1);
    ccu::Read(args->channelHandle, buf[0], dst, len, evt);
    ccu::EventWait(evt);
    ccu::ReadReduce(args->channelHandle, src, dst, len, HCCL_DATA_TYPE_FP16, HCCL_REDUCE_SUM, evt);
    ccu::EventWait(evt);
    return CcuResult::CCU_SUCCESS;
}
CcuResult CcuRemoteWriteKernel(CcuKernelArg arg)
{
    auto* args = static_cast<CcuVarAddKernelArg*>(arg);
    ccu::RemoteAddr dst;
    ccu::LocalAddr src;
    ccu::Array<ccu::CcuBuffer> buf(1);
    ccu::Variable len;
    len = 1024;
    ccu::Event evt;
    src.addr = 0x10000000;
    src.token = 0x20000000;
    dst.addr = 0x30000000;
    dst.token = 0x40000000;

    ccu::Write(args->channelHandle, dst, src, len, evt);
    ccu::EventWait(evt);
    ccu::Write(args->channelHandle, dst, buf[0], len, evt);
    ccu::EventWait(evt);
    ccu::WriteReduce(args->channelHandle, dst, src, len, HCCL_DATA_TYPE_FP16, HCCL_REDUCE_SUM, evt);
    ccu::EventWait(evt);
    return CcuResult::CCU_SUCCESS;
}
struct CcuVariableComputingKernelArg {
    uint64_t varA;
    uint64_t varB;
};
CcuResult CcuVariableComputingKernel(CcuKernelArg arg)
{
    auto* args = static_cast<CcuVarAddKernelArg*>(arg);
    ccu::Variable varA, varB, result;
    varA = 1024;
    varB = 2048;
    result = varA + varB;
    result = varA - varB;
    result = varA * varB;
    result = varA + 100;
    result = varA - 50;
    result = varA * 3;
    result -= varB;
    result *= varB;
    result -= 7;
    result *= 2;
    result = varA & varB;
    result = varA | varB;
    result = varA ^ varB;
    result = ~varA;
    result &= varB;
    result |= varB;
    result ^= varB;
    result = varA << varB;
    result = varA >> varB;
    result <<= varB;
    result >>= varB;
    return CcuResult::CCU_SUCCESS;
}

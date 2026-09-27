/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include "hcomm_primitives.h"

#include "builtin_channel_ops.h"
#include "host_cpu_roce_channel.h"

using namespace hccl;

class UtCpuHcommReadNbiOnThread : public testing::Test {
protected:
    virtual void SetUp() override
    {
        channelOnHost.SetNicChannelCtx(const_cast<HcommNicChannelOps*>(&g_BuiltinChannelOps), &channelOnHost);
    }

    virtual void TearDown() override { GlobalMockObject::verify(); }

    // thread is unused by HcommReadNbiOnThread, any value is acceptable.
    ThreadHandle thread = static_cast<ThreadHandle>(0x01);
    EndpointHandle epHandle = reinterpret_cast<void*>(0x01);
    HcommChannelDesc channelDesc{};
    hcomm::HostCpuRoceChannel channelOnHost{epHandle, channelDesc};
    ChannelHandle channel = reinterpret_cast<ChannelHandle>(&channelOnHost);
    char srcBuf[8]{};
    char dstBuf[8]{};
    void* src = srcBuf;
    void* dst = dstBuf;
    uint64_t len = sizeof(srcBuf);
    int32_t res{HCCL_E_RESERVED};
    DevType t950 = DevType::DEV_TYPE_950;
    DevType t910C = DevType::DEV_TYPE_910_93;
};

// 950

TEST_F(UtCpuHcommReadNbiOnThread, Ut_HcommReadNbiOnThread_When_950_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    MOCKER(&hrtGetDeviceType).stubs().with(outBound(t950)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(channelOnHost, &hcomm::HostCpuRoceChannel::Read)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    res = HcommReadNbiOnThread(thread, channel, dst, src, len);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

TEST_F(UtCpuHcommReadNbiOnThread, Ut_HcommReadNbiOnThread_When_950_Thread_IsNull_Expect_ReturnIsHCCL_SUCCESS)
{
    MOCKER(&hrtGetDeviceType).stubs().with(outBound(t950)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(channelOnHost, &hcomm::HostCpuRoceChannel::Read)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    // thread is cast to void — nullptr is acceptable.
    res = HcommReadNbiOnThread(0, channel, dst, src, len);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

TEST_F(UtCpuHcommReadNbiOnThread, Ut_HcommReadNbiOnThread_When_950_Channel_IsNull_Expect_ReturnIsHCCL_E_PTR)
{
    MOCKER(&hrtGetDeviceType).stubs().with(outBound(t950)).will(returnValue(HCCL_SUCCESS));
    res = HcommReadNbiOnThread(thread, 0, dst, src, len);
    EXPECT_EQ(res, HCCL_E_PTR);
}

TEST_F(UtCpuHcommReadNbiOnThread, Ut_HcommReadNbiOnThread_When_Src_IsNull_Expect_ReturnIsHCCL_E_PTR)
{
    // src is checked before hrtGetDeviceType, so no mock needed.
    res = HcommReadNbiOnThread(thread, channel, dst, nullptr, len);
    EXPECT_EQ(res, HCCL_E_PTR);
}

TEST_F(UtCpuHcommReadNbiOnThread, Ut_HcommReadNbiOnThread_When_Dst_IsNull_Expect_ReturnIsHCCL_E_PTR)
{
    // dst is checked before hrtGetDeviceType, so no mock needed.
    res = HcommReadNbiOnThread(thread, channel, nullptr, src, len);
    EXPECT_EQ(res, HCCL_E_PTR);
}

TEST_F(UtCpuHcommReadNbiOnThread, Ut_HcommReadNbiOnThread_When_950_Read_Fails_Expect_ErrorCodePropagated)
{
    MOCKER(&hrtGetDeviceType).stubs().with(outBound(t950)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(channelOnHost, &hcomm::HostCpuRoceChannel::Read)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_INTERNAL));
    res = HcommReadNbiOnThread(thread, channel, dst, src, len);
    EXPECT_EQ(res, HCCL_E_INTERNAL);
}

// 910C

TEST_F(UtCpuHcommReadNbiOnThread, Ut_HcommReadNbiOnThread_When_910C_Expect_ReturnIsHCCL_E_NOT_SUPPORT)
{
    MOCKER(&hrtGetDeviceType).stubs().with(outBound(t910C)).will(returnValue(HCCL_SUCCESS));
    // Non-950 devices are not supported by the NBI read path.
    res = HcommReadNbiOnThread(thread, channel, dst, src, len);
    EXPECT_EQ(res, HCCL_E_NOT_SUPPORT);
}

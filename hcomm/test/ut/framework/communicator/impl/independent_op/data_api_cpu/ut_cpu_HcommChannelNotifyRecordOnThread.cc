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
#include "new/hccl_primitive_remote.h"

#define private public
#include "cpu_ts_thread.h"
#include "host_cpu_roce_channel.h"
#undef private

using namespace hccl;

class UtCpuHcommChannelNotifyRecordOnThread : public testing::Test {
protected:
    virtual void SetUp() override { threadOnHost.stream_.reset(new (std::nothrow) Stream()); }

    virtual void TearDown() override { GlobalMockObject::verify(); }

    CpuTsThread threadOnHost{StreamType::STREAM_TYPE_ONLINE, 1, NotifyLoadType::DEVICE_NOTIFY};
    ThreadHandle thread = reinterpret_cast<ThreadHandle>(&threadOnHost);
    ChannelHandle channel910C = 0x03;
    EndpointHandle epHandle = reinterpret_cast<void*>(0x01);
    HcommChannelDesc channelDesc{};
    hcomm::HostCpuRoceChannel channelOnHost{epHandle, channelDesc};
    ChannelHandle channel = reinterpret_cast<ChannelHandle>(&channelOnHost);
    uint32_t notifyIdx = 0;
    int32_t res{HCCL_E_RESERVED};
    DevType t950 = DevType::DEV_TYPE_950;
    DevType t910C = DevType::DEV_TYPE_910_93;
};

// 950

TEST_F(
    UtCpuHcommChannelNotifyRecordOnThread,
    Ut_HcommChannelNotifyRecordOnThread_When_950_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    MOCKER(&hrtGetDeviceType).stubs().with(outBound(t950)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(channelOnHost, &hcomm::HostCpuRoceChannel::NotifyRecord)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    res = HcommChannelNotifyRecordOnThread(thread, channel, notifyIdx);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

TEST_F(
    UtCpuHcommChannelNotifyRecordOnThread,
    Ut_HcommChannelNotifyRecordOnThread_When_950_Thread_IsNull_Expect_ReturnIsHCCL_SUCCESS)
{
    MOCKER(&hrtGetDeviceType).stubs().with(outBound(t950)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(channelOnHost, &hcomm::HostCpuRoceChannel::NotifyRecord)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    // On 950, thread is not used, so it could be nullptr.
    res = HcommChannelNotifyRecordOnThread(0, channel, notifyIdx);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

TEST_F(
    UtCpuHcommChannelNotifyRecordOnThread,
    Ut_HcommChannelNotifyRecordOnThread_When_950_Channel_IsNull_Expect_ReturnIsHCCL_E_PTR)
{
    MOCKER(&hrtGetDeviceType).stubs().with(outBound(t950)).will(returnValue(HCCL_SUCCESS));
    res = HcommChannelNotifyRecordOnThread(thread, 0, notifyIdx);
    EXPECT_EQ(res, HCCL_E_PTR);
}

TEST_F(
    UtCpuHcommChannelNotifyRecordOnThread,
    Ut_HcommChannelNotifyRecordOnThread_When_950_NotifyRecord_Fails_Expect_ErrorCodePropagated)
{
    MOCKER(&hrtGetDeviceType).stubs().with(outBound(t950)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(channelOnHost, &hcomm::HostCpuRoceChannel::NotifyRecord)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_INTERNAL));
    res = HcommChannelNotifyRecordOnThread(thread, channel, notifyIdx);
    EXPECT_EQ(res, HCCL_E_INTERNAL);
}

// 910C

TEST_F(
    UtCpuHcommChannelNotifyRecordOnThread,
    Ut_HcommChannelNotifyRecordOnThread_When_910C_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    MOCKER(&hrtGetDeviceType).stubs().with(outBound(t910C)).will(returnValue(HCCL_SUCCESS));
    MOCKER(&HcclRemoteNotifyRecord).stubs().will(returnValue(HCCL_SUCCESS));
    res = HcommChannelNotifyRecordOnThread(thread, channel910C, notifyIdx);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

TEST_F(
    UtCpuHcommChannelNotifyRecordOnThread,
    Ut_HcommChannelNotifyRecordOnThread_When_910C_Thread_IsNull_Expect_ReturnIsHCCL_E_PTR)
{
    MOCKER(&hrtGetDeviceType).stubs().with(outBound(t910C)).will(returnValue(HCCL_SUCCESS));
    res = HcommChannelNotifyRecordOnThread(0, channel910C, notifyIdx);
    EXPECT_EQ(res, HCCL_E_PTR);
}

TEST_F(
    UtCpuHcommChannelNotifyRecordOnThread,
    Ut_HcommChannelNotifyRecordOnThread_When_910C_GetStream_IsNull_Expect_ReturnIsHCCL_E_PTR)
{
    MOCKER(&hrtGetDeviceType).stubs().with(outBound(t910C)).will(returnValue(HCCL_SUCCESS));
    threadOnHost.stream_.reset();
    res = HcommChannelNotifyRecordOnThread(thread, channel910C, notifyIdx);
    EXPECT_EQ(res, HCCL_E_PTR);
}

TEST_F(
    UtCpuHcommChannelNotifyRecordOnThread,
    Ut_HcommChannelNotifyRecordOnThread_When_910C_NotifyRecord_Fails_Expect_ErrorCodePropagated)
{
    MOCKER(&hrtGetDeviceType).stubs().with(outBound(t910C)).will(returnValue(HCCL_SUCCESS));
    MOCKER(&HcclRemoteNotifyRecord).stubs().will(returnValue(HCCL_E_INTERNAL));
    res = HcommChannelNotifyRecordOnThread(thread, channel910C, notifyIdx);
    EXPECT_EQ(res, HCCL_E_INTERNAL);
}

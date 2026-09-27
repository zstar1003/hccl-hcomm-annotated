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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>

#define private public
#include "my_rank.h"
#undef private

#include "channel.h"

using namespace hccl;

namespace {
Hccl::TlsStatus g_expectedTlsStatus = Hccl::TlsStatus::UNKNOWN;
int g_expectedNetworkMode = NetworkMode::NETWORK_OFFLINE;

HcclResult StubHrtGetDeviceSuccess(s32* deviceLogicId)
{
    *deviceLogicId = 1;
    return HCCL_SUCCESS;
}

HcclResult StubHrtGetDevicePhyIdByUserDevIdSuccess(u32, u32& devicePhyId, bool)
{
    devicePhyId = 8;
    return HCCL_SUCCESS;
}

HcclResult StubHrtGetDevicePhyIdByIndexSuccess(u32, u32& devicePhyId, bool)
{
    devicePhyId = 8;
    return HCCL_SUCCESS;
}

HcclResult StubMyRankHrtRaGetTlsStatus(RaInfo* info, Hccl::TlsStatus& tlsStatus)
{
    EXPECT_NE(info, nullptr);
    EXPECT_EQ(info->phyId, 8U);
    EXPECT_EQ(info->mode, g_expectedNetworkMode);
    tlsStatus = g_expectedTlsStatus;
    return HCCL_SUCCESS;
}

HcclChannelDesc CreateChannelDesc(CommProtocol protocol, EndpointLocType locType)
{
    HcclChannelDesc channelDesc{};
    (void)HcclChannelDescInit(&channelDesc, 1);
    channelDesc.localEndpoint.protocol = protocol;
    channelDesc.localEndpoint.loc.locType = locType;
    return channelDesc;
}
} // namespace

class MyRankTlsTest : public testing::Test {
protected:
    void SetUp() override
    {
        callbacks_.getAicpuCommState = []() {
            return false;
        };
        callbacks_.setAicpuCommState = [](bool) {};
        callbacks_.kernelLaunchAicpuCommInit = []() {
            return HCCL_SUCCESS;
        };
        myRank_.reset(new MyRank(nullptr, 0, config_, callbacks_, nullptr, nullptr));
        g_expectedTlsStatus = Hccl::TlsStatus::UNKNOWN;
        g_expectedNetworkMode = NetworkMode::NETWORK_OFFLINE;
    }

    void TearDown() override
    {
        GlobalMockObject::verify();
        myRank_.reset();
    }

    CommConfig config_{};
    ManagerCallbacks callbacks_{};
    std::unique_ptr<MyRank> myRank_{};
};

TEST_F(MyRankTlsTest, Ut_GetLocalTlsStatus_When_HrtGetDeviceFails_Expect_ReturnSameError)
{
    Hccl::TlsStatus tlsStatus = Hccl::TlsStatus::UNKNOWN;

    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_E_RUNTIME));

    HcclResult ret = myRank_->GetLocalTlsStatus(ENDPOINT_LOC_TYPE_DEVICE, tlsStatus);

    EXPECT_EQ(ret, HCCL_E_RUNTIME);
}

TEST_F(MyRankTlsTest, Ut_GetLocalTlsStatus_When_HrtGetDevicePhyIdByUserDevIdFails_Expect_ReturnSameError)
{
    Hccl::TlsStatus tlsStatus = Hccl::TlsStatus::UNKNOWN;

    MOCKER(hrtGetDevice).stubs().will(invoke(StubHrtGetDeviceSuccess));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(returnValue(HCCL_E_RUNTIME));

    HcclResult ret = myRank_->GetLocalTlsStatus(ENDPOINT_LOC_TYPE_DEVICE, tlsStatus);

    EXPECT_EQ(ret, HCCL_E_RUNTIME);
}

TEST_F(MyRankTlsTest, Ut_GetLocalTlsStatus_When_AllDependenciesSucceed_Expect_ReturnSuccessAndTlsStatusUpdated)
{
    Hccl::TlsStatus tlsStatus = Hccl::TlsStatus::UNKNOWN;
    g_expectedTlsStatus = Hccl::TlsStatus::ENABLE;

    MOCKER(hrtGetDevice).stubs().will(invoke(StubHrtGetDeviceSuccess));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(invoke(StubHrtGetDevicePhyIdByUserDevIdSuccess));
    MOCKER(Hccl::HrtRaGetTlsStatus).stubs().will(invoke(StubMyRankHrtRaGetTlsStatus));

    HcclResult ret = myRank_->GetLocalTlsStatus(ENDPOINT_LOC_TYPE_DEVICE, tlsStatus);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(tlsStatus, Hccl::TlsStatus::ENABLE);
}

TEST_F(MyRankTlsTest, Ut_GetAbnormalChannelTlsStatus_When_LaterChannelFails_Expect_OnlyAbnormalChannelQueried)
{
    g_expectedTlsStatus = Hccl::TlsStatus::ENABLE;
    g_expectedNetworkMode = NetworkMode::NETWORK_PEER_ONLINE;
    HcclChannelDesc channelDescs[2] = {
        CreateChannelDesc(COMM_PROTOCOL_HCCS, ENDPOINT_LOC_TYPE_DEVICE),
        CreateChannelDesc(COMM_PROTOCOL_ROCE, ENDPOINT_LOC_TYPE_HOST),
    };
    int32_t statusList[2] = {hcomm::ChannelStatus::READY, hcomm::ChannelStatus::FAILED};
    MOCKER(hrtGetDevice).stubs().will(invoke(StubHrtGetDeviceSuccess));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(invoke(StubHrtGetDevicePhyIdByIndexSuccess));
    MOCKER(Hccl::HrtRaGetTlsStatus).stubs().will(invoke(StubMyRankHrtRaGetTlsStatus));

    std::vector<Hccl::TlsStatus> tlsStatusList;
    myRank_->GetAbnormalChannelTlsStatus(channelDescs, statusList, 2, tlsStatusList);

    ASSERT_EQ(tlsStatusList.size(), 2U);
    EXPECT_EQ(tlsStatusList[0], Hccl::TlsStatus::UNKNOWN);
    EXPECT_EQ(tlsStatusList[1], Hccl::TlsStatus::ENABLE);
}

TEST_F(MyRankTlsTest, Ut_GetAbnormalChannelTlsStatus_When_AllReady_Expect_NoChannelQueried)
{
    HcclChannelDesc channelDescs[2] = {
        CreateChannelDesc(COMM_PROTOCOL_HCCS, ENDPOINT_LOC_TYPE_DEVICE),
        CreateChannelDesc(COMM_PROTOCOL_ROCE, ENDPOINT_LOC_TYPE_HOST),
    };
    int32_t statusList[2] = {hcomm::ChannelStatus::READY, hcomm::ChannelStatus::READY};
    MOCKER(hrtGetDevice).stubs().will(invoke(StubHrtGetDeviceSuccess));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(invoke(StubHrtGetDevicePhyIdByIndexSuccess));
    MOCKER(Hccl::HrtRaGetTlsStatus).stubs().will(invoke(StubMyRankHrtRaGetTlsStatus));

    std::vector<Hccl::TlsStatus> tlsStatusList;
    myRank_->GetAbnormalChannelTlsStatus(channelDescs, statusList, 2, tlsStatusList);

    ASSERT_EQ(tlsStatusList.size(), 2U);
    EXPECT_EQ(tlsStatusList[0], Hccl::TlsStatus::UNKNOWN);
    EXPECT_EQ(tlsStatusList[1], Hccl::TlsStatus::UNKNOWN);
}

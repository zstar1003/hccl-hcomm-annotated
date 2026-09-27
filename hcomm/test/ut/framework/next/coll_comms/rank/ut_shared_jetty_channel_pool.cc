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

#define private public
#define protected public
#include "shared_jetty_channel_pool.h"
#undef private
#undef protected

using namespace hccl;

namespace {
EndpointDesc MakeEpDesc(uint32_t ipInt, uint32_t devPhyId)
{
    EndpointDesc desc{};
    desc.protocol = COMM_PROTOCOL_UB_CTP;
    desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    desc.commAddr.addr.s_addr = ipInt;
    desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    desc.loc.device.devPhyId = devPhyId;
    return desc;
}

EndpointDescPair MakeEpPair(uint32_t localIp, uint32_t remoteIp)
{
    return {MakeEpDesc(localIp, 0), MakeEpDesc(remoteIp, 1)};
}

ChannelHandle g_nextHandle = 0x1000;

HcclResult CreateChannelsStub(uint32_t num, ChannelHandle* out)
{
    for (uint32_t i = 0; i < num; ++i) {
        out[i] = g_nextHandle++;
    }
    return HCCL_SUCCESS;
}

HcclResult CreateChannelsFailStub(uint32_t num, ChannelHandle* out)
{
    (void)num;
    (void)out;
    return HCCL_E_INTERNAL;
}
} // namespace

class SharedJettyChannelPoolTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }
    static void TearDownTestCase()
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }
    void SetUp() override
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        SharedJettyChannelPool::GetInstance().rankPools_.clear();
        g_nextHandle = 0x1000;
    }
    void TearDown() override
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        SharedJettyChannelPool::GetInstance().rankPools_.clear();
    }
};

TEST_F(SharedJettyChannelPoolTest, Ut_AcquireChannels_When_NullMyRank_Expect_HCCL_E_PARA)
{
    EndpointDescPair epPair = MakeEpPair(0x01000001, 0x02000001);
    ChannelHandle out[2];
    EXPECT_EQ(
        SharedJettyChannelPool::GetInstance().AcquireChannels(nullptr, "tag1", epPair, 2, CreateChannelsStub, out),
        HCCL_E_PARA);
}

TEST_F(SharedJettyChannelPoolTest, Ut_AcquireChannels_When_ZeroRequestedNum_Expect_HCCL_E_PARA)
{
    MyRank* myRank = reinterpret_cast<MyRank*>(0x1);
    EndpointDescPair epPair = MakeEpPair(0x01000001, 0x02000001);
    ChannelHandle out[2];
    EXPECT_EQ(
        SharedJettyChannelPool::GetInstance().AcquireChannels(myRank, "tag1", epPair, 0, CreateChannelsStub, out),
        HCCL_E_PARA);
}

TEST_F(SharedJettyChannelPoolTest, Ut_AcquireChannels_When_NullOutChannels_Expect_HCCL_E_PARA)
{
    MyRank* myRank = reinterpret_cast<MyRank*>(0x1);
    EndpointDescPair epPair = MakeEpPair(0x01000001, 0x02000001);
    EXPECT_EQ(
        SharedJettyChannelPool::GetInstance().AcquireChannels(myRank, "tag1", epPair, 2, CreateChannelsStub, nullptr),
        HCCL_E_PARA);
}

TEST_F(SharedJettyChannelPoolTest, Ut_AcquireChannels_When_FirstCreate_Expect_Success)
{
    MyRank* myRank = reinterpret_cast<MyRank*>(0x1);
    EndpointDescPair epPair = MakeEpPair(0x01000001, 0x02000001);
    ChannelHandle out[2];
    uint32_t reusedCount = 0;
    EXPECT_EQ(
        SharedJettyChannelPool::GetInstance().AcquireChannels(
            myRank, "tag1", epPair, 2, CreateChannelsStub, out, &reusedCount),
        HCCL_SUCCESS);
    EXPECT_EQ(reusedCount, 0U);
    EXPECT_EQ(out[0], 0x1000);
    EXPECT_EQ(out[1], 0x1001);
}

TEST_F(SharedJettyChannelPoolTest, Ut_AcquireChannels_When_AllReuse_Expect_Success)
{
    MyRank* myRank = reinterpret_cast<MyRank*>(0x1);
    EndpointDescPair epPair = MakeEpPair(0x01000001, 0x02000001);
    ChannelHandle out1[2];
    ASSERT_EQ(
        SharedJettyChannelPool::GetInstance().AcquireChannels(myRank, "tag1", epPair, 2, CreateChannelsStub, out1),
        HCCL_SUCCESS);

    ChannelHandle out2[2];
    uint32_t reusedCount = 0;
    EXPECT_EQ(
        SharedJettyChannelPool::GetInstance().AcquireChannels(
            myRank, "tag1", epPair, 2, CreateChannelsStub, out2, &reusedCount),
        HCCL_SUCCESS);
    EXPECT_EQ(reusedCount, 2U);
}

TEST_F(SharedJettyChannelPoolTest, Ut_AcquireChannels_When_PartialReuse_Expect_Success)
{
    MyRank* myRank = reinterpret_cast<MyRank*>(0x1);
    EndpointDescPair epPair = MakeEpPair(0x01000001, 0x02000001);
    ChannelHandle out1[2];
    ASSERT_EQ(
        SharedJettyChannelPool::GetInstance().AcquireChannels(myRank, "tag1", epPair, 2, CreateChannelsStub, out1),
        HCCL_SUCCESS);

    ChannelHandle out2[3];
    uint32_t reusedCount = 0;
    EXPECT_EQ(
        SharedJettyChannelPool::GetInstance().AcquireChannels(
            myRank, "tag1", epPair, 3, CreateChannelsStub, out2, &reusedCount),
        HCCL_SUCCESS);
    EXPECT_EQ(reusedCount, 2U);
}

TEST_F(SharedJettyChannelPoolTest, Ut_AcquireChannels_When_CreateFuncFails_Expect_ReturnError)
{
    MyRank* myRank = reinterpret_cast<MyRank*>(0x1);
    EndpointDescPair epPair = MakeEpPair(0x01000001, 0x02000001);
    ChannelHandle out[2];
    EXPECT_EQ(
        SharedJettyChannelPool::GetInstance().AcquireChannels(myRank, "tag1", epPair, 2, CreateChannelsFailStub, out),
        HCCL_E_INTERNAL);
}

TEST_F(SharedJettyChannelPoolTest, Ut_AcquireChannels_When_DifferentTags_Expect_IndependentPools)
{
    MyRank* myRank = reinterpret_cast<MyRank*>(0x1);
    EndpointDescPair epPair = MakeEpPair(0x01000001, 0x02000001);
    ChannelHandle out1[2];
    ASSERT_EQ(
        SharedJettyChannelPool::GetInstance().AcquireChannels(myRank, "tag1", epPair, 2, CreateChannelsStub, out1),
        HCCL_SUCCESS);

    ChannelHandle out2[2];
    uint32_t reusedCount = 1;
    EXPECT_EQ(
        SharedJettyChannelPool::GetInstance().AcquireChannels(
            myRank, "tag2", epPair, 2, CreateChannelsStub, out2, &reusedCount),
        HCCL_SUCCESS);
    EXPECT_EQ(reusedCount, 0U);
}

TEST_F(SharedJettyChannelPoolTest, Ut_AcquireChannels_When_OutReusedCountNull_Expect_Success)
{
    MyRank* myRank = reinterpret_cast<MyRank*>(0x1);
    EndpointDescPair epPair = MakeEpPair(0x01000001, 0x02000001);
    ChannelHandle out[1];
    EXPECT_EQ(
        SharedJettyChannelPool::GetInstance().AcquireChannels(
            myRank, "tag1", epPair, 1, CreateChannelsStub, out, nullptr),
        HCCL_SUCCESS);
}

TEST_F(SharedJettyChannelPoolTest, Ut_DestroyAllByMyRank_When_NullMyRank_Expect_Success)
{
    EXPECT_EQ(SharedJettyChannelPool::GetInstance().DestroyAllByMyRank(nullptr), HCCL_SUCCESS);
}

TEST_F(SharedJettyChannelPoolTest, Ut_DestroyAllByMyRank_When_NoPool_Expect_Success)
{
    MyRank* myRank = reinterpret_cast<MyRank*>(0x1);
    EXPECT_EQ(SharedJettyChannelPool::GetInstance().DestroyAllByMyRank(myRank), HCCL_SUCCESS);
}

TEST_F(SharedJettyChannelPoolTest, Ut_DestroyAllByMyRank_When_HasChannels_Expect_Success)
{
    MyRank* myRank = reinterpret_cast<MyRank*>(0x1);
    EndpointDescPair epPair = MakeEpPair(0x01000001, 0x02000001);
    ChannelHandle out[2];
    ASSERT_EQ(
        SharedJettyChannelPool::GetInstance().AcquireChannels(myRank, "tag1", epPair, 2, CreateChannelsStub, out),
        HCCL_SUCCESS);

    SharedJettyChannelPool::GetInstance().RemoveChannels(myRank, "tag1", epPair, out, 2);
    EXPECT_EQ(SharedJettyChannelPool::GetInstance().DestroyAllByMyRank(myRank), HCCL_SUCCESS);
    EXPECT_EQ(SharedJettyChannelPool::GetInstance().rankPools_.count(myRank), 0U);
}

TEST_F(SharedJettyChannelPoolTest, Ut_DestroyAllByMyRank_When_MultipleTags_Expect_Success)
{
    MyRank* myRank = reinterpret_cast<MyRank*>(0x1);
    EndpointDescPair epPair = MakeEpPair(0x01000001, 0x02000001);
    ChannelHandle out1[1];
    ChannelHandle out2[1];
    ASSERT_EQ(
        SharedJettyChannelPool::GetInstance().AcquireChannels(myRank, "tag1", epPair, 1, CreateChannelsStub, out1),
        HCCL_SUCCESS);
    ASSERT_EQ(
        SharedJettyChannelPool::GetInstance().AcquireChannels(myRank, "tag2", epPair, 1, CreateChannelsStub, out2),
        HCCL_SUCCESS);

    SharedJettyChannelPool::GetInstance().RemoveChannels(myRank, "tag1", epPair, out1, 1);
    SharedJettyChannelPool::GetInstance().RemoveChannels(myRank, "tag2", epPair, out2, 1);
    EXPECT_EQ(SharedJettyChannelPool::GetInstance().DestroyAllByMyRank(myRank), HCCL_SUCCESS);
    EXPECT_EQ(SharedJettyChannelPool::GetInstance().rankPools_.count(myRank), 0U);
}

TEST_F(SharedJettyChannelPoolTest, Ut_CheckMyRankDestroy_When_NoPool_Expect_Success)
{
    MyRank* myRank = reinterpret_cast<MyRank*>(0x1);
    EXPECT_EQ(SharedJettyChannelPool::GetInstance().CheckMyRankDestroy(myRank), HCCL_SUCCESS);
}

TEST_F(SharedJettyChannelPoolTest, Ut_CheckMyRankDestroy_When_HasChannels_Expect_HCCL_E_UNAVAIL)
{
    MyRank* myRank = reinterpret_cast<MyRank*>(0x1);
    EndpointDescPair epPair = MakeEpPair(0x01000001, 0x02000001);
    ChannelHandle out[2];
    ASSERT_EQ(
        SharedJettyChannelPool::GetInstance().AcquireChannels(myRank, "tag1", epPair, 2, CreateChannelsStub, out),
        HCCL_SUCCESS);

    EXPECT_EQ(SharedJettyChannelPool::GetInstance().CheckMyRankDestroy(myRank), HCCL_E_UNAVAIL);
}

TEST_F(SharedJettyChannelPoolTest, Ut_CheckMyRankDestroy_When_AfterDestroy_Expect_Success)
{
    MyRank* myRank = reinterpret_cast<MyRank*>(0x1);
    EndpointDescPair epPair = MakeEpPair(0x01000001, 0x02000001);
    ChannelHandle out[2];
    ASSERT_EQ(
        SharedJettyChannelPool::GetInstance().AcquireChannels(myRank, "tag1", epPair, 2, CreateChannelsStub, out),
        HCCL_SUCCESS);

    SharedJettyChannelPool::GetInstance().RemoveChannels(myRank, "tag1", epPair, out, 2);
    ASSERT_EQ(SharedJettyChannelPool::GetInstance().DestroyAllByMyRank(myRank), HCCL_SUCCESS);
    EXPECT_EQ(SharedJettyChannelPool::GetInstance().CheckMyRankDestroy(myRank), HCCL_SUCCESS);
}

TEST_F(SharedJettyChannelPoolTest, Ut_RemoveChannels_When_Normal_Expect_Removed)
{
    MyRank* myRank = reinterpret_cast<MyRank*>(0x1);
    EndpointDescPair epPair = MakeEpPair(0x01000001, 0x02000001);
    ChannelHandle out[3];
    ASSERT_EQ(
        SharedJettyChannelPool::GetInstance().AcquireChannels(myRank, "tag1", epPair, 3, CreateChannelsStub, out),
        HCCL_SUCCESS);

    ChannelHandle toRemove[] = {out[0]};
    SharedJettyChannelPool::GetInstance().RemoveChannels(myRank, "tag1", epPair, toRemove, 1);

    auto& rankPool = SharedJettyChannelPool::GetInstance().rankPools_[myRank];
    auto& epChannels = rankPool["tag1"][epPair];
    EXPECT_EQ(epChannels.channels.size(), 2U);
}

TEST_F(SharedJettyChannelPoolTest, Ut_RemoveChannels_When_AllRemoved_Expect_EmptyAndResetIdx)
{
    MyRank* myRank = reinterpret_cast<MyRank*>(0x1);
    EndpointDescPair epPair = MakeEpPair(0x01000001, 0x02000001);
    ChannelHandle out[2];
    ASSERT_EQ(
        SharedJettyChannelPool::GetInstance().AcquireChannels(myRank, "tag1", epPair, 2, CreateChannelsStub, out),
        HCCL_SUCCESS);

    ChannelHandle toRemove[] = {out[0], out[1]};
    SharedJettyChannelPool::GetInstance().RemoveChannels(myRank, "tag1", epPair, toRemove, 2);

    auto& rankPool = SharedJettyChannelPool::GetInstance().rankPools_[myRank];
    auto& epChannels = rankPool["tag1"][epPair];
    EXPECT_EQ(epChannels.channels.size(), 0U);
    EXPECT_EQ(epChannels.nextReturnIdx, 0U);
}

TEST_F(SharedJettyChannelPoolTest, Ut_RemoveChannels_When_NotFound_Expect_NoChange)
{
    MyRank* myRank = reinterpret_cast<MyRank*>(0x1);
    EndpointDescPair epPair = MakeEpPair(0x01000001, 0x02000001);
    ChannelHandle out[2];
    ASSERT_EQ(
        SharedJettyChannelPool::GetInstance().AcquireChannels(myRank, "tag1", epPair, 2, CreateChannelsStub, out),
        HCCL_SUCCESS);

    ChannelHandle toRemove[] = {0xDEAD};
    SharedJettyChannelPool::GetInstance().RemoveChannels(myRank, "tag1", epPair, toRemove, 1);

    auto& rankPool = SharedJettyChannelPool::GetInstance().rankPools_[myRank];
    auto& epChannels = rankPool["tag1"][epPair];
    EXPECT_EQ(epChannels.channels.size(), 2U);
}

TEST_F(SharedJettyChannelPoolTest, Ut_RemoveChannels_When_NullMyRank_Expect_Noop)
{
    EndpointDescPair epPair = MakeEpPair(0x01000001, 0x02000001);
    ChannelHandle toRemove[] = {0x1000};
    SharedJettyChannelPool::GetInstance().RemoveChannels(nullptr, "tag1", epPair, toRemove, 1);
    EXPECT_EQ(SharedJettyChannelPool::GetInstance().rankPools_.size(), 0U);
}

TEST_F(SharedJettyChannelPoolTest, Ut_RemoveChannels_When_NoMatchingTag_Expect_Noop)
{
    MyRank* myRank = reinterpret_cast<MyRank*>(0x1);
    EndpointDescPair epPair = MakeEpPair(0x01000001, 0x02000001);
    ChannelHandle out[2];
    ASSERT_EQ(
        SharedJettyChannelPool::GetInstance().AcquireChannels(myRank, "tag1", epPair, 2, CreateChannelsStub, out),
        HCCL_SUCCESS);

    ChannelHandle toRemove[] = {out[0]};
    SharedJettyChannelPool::GetInstance().RemoveChannels(myRank, "nonexistent", epPair, toRemove, 1);
    auto& rankPool = SharedJettyChannelPool::GetInstance().rankPools_[myRank];
    EXPECT_EQ(rankPool["tag1"][epPair].channels.size(), 2U);
}

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
#define protected public
#include "shared_jetty_mgr.h"
#undef private
#undef protected

using namespace hcomm;

class SharedJettyMgrTest : public testing::Test {
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
        SharedJettyMgr::GetInstance().contexts_.clear();
    }
    void TearDown() override
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        SharedJettyMgr::GetInstance().contexts_.clear();
    }
};

TEST_F(SharedJettyMgrTest, Ut_RegisterChannels_When_NullEndpoint_Expect_HCCL_E_PARA)
{
    ChannelHandle channels[] = {0x1000};
    EXPECT_EQ(SharedJettyMgr::GetInstance().RegisterChannels(nullptr, channels, 1), HCCL_E_PARA);
}

TEST_F(SharedJettyMgrTest, Ut_RegisterChannels_When_NullChannels_Expect_HCCL_E_PARA)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    EXPECT_EQ(SharedJettyMgr::GetInstance().RegisterChannels(ep, nullptr, 1), HCCL_E_PARA);
}

TEST_F(SharedJettyMgrTest, Ut_RegisterChannels_When_ZeroChannelNum_Expect_HCCL_E_PARA)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    ChannelHandle channels[] = {0x1000};
    EXPECT_EQ(SharedJettyMgr::GetInstance().RegisterChannels(ep, channels, 0), HCCL_E_PARA);
}

TEST_F(SharedJettyMgrTest, Ut_RegisterChannels_When_Normal_Expect_Success)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    ChannelHandle channels[] = {0x1000, 0x2000};
    EXPECT_EQ(SharedJettyMgr::GetInstance().RegisterChannels(ep, channels, 2), HCCL_SUCCESS);
    EXPECT_TRUE(SharedJettyMgr::GetInstance().HasContext(ep));
    auto& ctx = SharedJettyMgr::GetInstance().contexts_[ep];
    EXPECT_EQ(ctx.channelCount, 2U);
    EXPECT_EQ(ctx.channelHandles.size(), 2U);
}

TEST_F(SharedJettyMgrTest, Ut_RegisterChannels_When_DuplicateChannels_Expect_CountIsUniqueSize)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    ChannelHandle channels[] = {0x1000, 0x1000, 0x2000};
    EXPECT_EQ(SharedJettyMgr::GetInstance().RegisterChannels(ep, channels, 3), HCCL_SUCCESS);
    auto& ctx = SharedJettyMgr::GetInstance().contexts_[ep];
    EXPECT_EQ(ctx.channelCount, 2U);
    EXPECT_EQ(ctx.channelHandles.size(), 2U);
}

TEST_F(SharedJettyMgrTest, Ut_RegisterChannels_When_MultipleEndpoints_Expect_IndependentContexts)
{
    EndpointHandle ep1 = reinterpret_cast<EndpointHandle>(0x1);
    EndpointHandle ep2 = reinterpret_cast<EndpointHandle>(0x2);
    ChannelHandle ch1[] = {0x1000};
    ChannelHandle ch2[] = {0x2000};
    EXPECT_EQ(SharedJettyMgr::GetInstance().RegisterChannels(ep1, ch1, 1), HCCL_SUCCESS);
    EXPECT_EQ(SharedJettyMgr::GetInstance().RegisterChannels(ep2, ch2, 1), HCCL_SUCCESS);
    EXPECT_TRUE(SharedJettyMgr::GetInstance().HasContext(ep1));
    EXPECT_TRUE(SharedJettyMgr::GetInstance().HasContext(ep2));
    EXPECT_EQ(SharedJettyMgr::GetInstance().contexts_.size(), 2U);
}

TEST_F(SharedJettyMgrTest, Ut_UnregisterChannels_When_NullChannels_Expect_Success)
{
    EXPECT_EQ(SharedJettyMgr::GetInstance().UnregisterChannels(nullptr, 0), HCCL_SUCCESS);
}

TEST_F(SharedJettyMgrTest, Ut_UnregisterChannels_When_Normal_Expect_Success)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    ChannelHandle channels[] = {0x1000, 0x2000};
    ASSERT_EQ(SharedJettyMgr::GetInstance().RegisterChannels(ep, channels, 2), HCCL_SUCCESS);

    ChannelHandle toUnregister[] = {0x1000};
    EXPECT_EQ(SharedJettyMgr::GetInstance().UnregisterChannels(toUnregister, 1), HCCL_SUCCESS);
    auto& ctx = SharedJettyMgr::GetInstance().contexts_[ep];
    EXPECT_EQ(ctx.channelCount, 1U);
    EXPECT_EQ(ctx.channelHandles.count(0x1000), 0U);
    EXPECT_EQ(ctx.channelHandles.count(0x2000), 1U);
}

TEST_F(SharedJettyMgrTest, Ut_UnregisterChannels_When_LastChannel_Expect_ContextRemoved)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    ChannelHandle channels[] = {0x1000};
    ASSERT_EQ(SharedJettyMgr::GetInstance().RegisterChannels(ep, channels, 1), HCCL_SUCCESS);
    EXPECT_TRUE(SharedJettyMgr::GetInstance().HasContext(ep));

    ChannelHandle toUnregister[] = {0x1000};
    EXPECT_EQ(SharedJettyMgr::GetInstance().UnregisterChannels(toUnregister, 1), HCCL_SUCCESS);
    EXPECT_FALSE(SharedJettyMgr::GetInstance().HasContext(ep));
    EXPECT_EQ(SharedJettyMgr::GetInstance().contexts_.count(ep), 0U);
}

TEST_F(SharedJettyMgrTest, Ut_UnregisterChannels_When_ChannelNotFound_Expect_Success)
{
    ChannelHandle toUnregister[] = {0x9999};
    EXPECT_EQ(SharedJettyMgr::GetInstance().UnregisterChannels(toUnregister, 1), HCCL_SUCCESS);
}

TEST_F(SharedJettyMgrTest, Ut_UnregisterChannels_When_MultipleAtOnce_Expect_Success)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    ChannelHandle channels[] = {0x1000, 0x2000, 0x3000};
    ASSERT_EQ(SharedJettyMgr::GetInstance().RegisterChannels(ep, channels, 3), HCCL_SUCCESS);

    ChannelHandle toUnregister[] = {0x1000, 0x3000};
    EXPECT_EQ(SharedJettyMgr::GetInstance().UnregisterChannels(toUnregister, 2), HCCL_SUCCESS);
    auto& ctx = SharedJettyMgr::GetInstance().contexts_[ep];
    EXPECT_EQ(ctx.channelCount, 1U);
    EXPECT_EQ(ctx.channelHandles.count(0x2000), 1U);
}

TEST_F(SharedJettyMgrTest, Ut_CheckEndpointDestroy_When_NoContext_Expect_Success)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    EXPECT_EQ(SharedJettyMgr::GetInstance().CheckEndpointDestroy(ep), HCCL_SUCCESS);
}

TEST_F(SharedJettyMgrTest, Ut_CheckEndpointDestroy_When_HasChannels_Expect_HCCL_E_UNAVAIL)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    ChannelHandle channels[] = {0x1000};
    ASSERT_EQ(SharedJettyMgr::GetInstance().RegisterChannels(ep, channels, 1), HCCL_SUCCESS);
    EXPECT_EQ(SharedJettyMgr::GetInstance().CheckEndpointDestroy(ep), HCCL_E_UNAVAIL);
}

TEST_F(SharedJettyMgrTest, Ut_CheckEndpointDestroy_When_AllUnregistered_Expect_Success)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    ChannelHandle channels[] = {0x1000};
    ASSERT_EQ(SharedJettyMgr::GetInstance().RegisterChannels(ep, channels, 1), HCCL_SUCCESS);

    ChannelHandle toUnregister[] = {0x1000};
    ASSERT_EQ(SharedJettyMgr::GetInstance().UnregisterChannels(toUnregister, 1), HCCL_SUCCESS);
    EXPECT_EQ(SharedJettyMgr::GetInstance().CheckEndpointDestroy(ep), HCCL_SUCCESS);
}

TEST_F(SharedJettyMgrTest, Ut_HasContext_When_NoContext_Expect_False)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    EXPECT_FALSE(SharedJettyMgr::GetInstance().HasContext(ep));
}

TEST_F(SharedJettyMgrTest, Ut_HasContext_When_HasContext_Expect_True)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    ChannelHandle channels[] = {0x1000};
    ASSERT_EQ(SharedJettyMgr::GetInstance().RegisterChannels(ep, channels, 1), HCCL_SUCCESS);
    EXPECT_TRUE(SharedJettyMgr::GetInstance().HasContext(ep));
}

TEST_F(SharedJettyMgrTest, Ut_UnregisterEndpoint_When_NoContext_Expect_Noop)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    SharedJettyMgr::GetInstance().UnregisterEndpoint(ep);
    EXPECT_FALSE(SharedJettyMgr::GetInstance().HasContext(ep));
}

TEST_F(SharedJettyMgrTest, Ut_UnregisterEndpoint_When_HasContext_Expect_Removed)
{
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    ChannelHandle channels[] = {0x1000};
    ASSERT_EQ(SharedJettyMgr::GetInstance().RegisterChannels(ep, channels, 1), HCCL_SUCCESS);
    EXPECT_TRUE(SharedJettyMgr::GetInstance().HasContext(ep));

    SharedJettyMgr::GetInstance().UnregisterEndpoint(ep);
    EXPECT_FALSE(SharedJettyMgr::GetInstance().HasContext(ep));
}

TEST_F(SharedJettyMgrTest, Ut_UnregisterEndpoint_When_OnlyRemovesSpecifiedEndpoint_Expect_OthersUntouched)
{
    EndpointHandle ep1 = reinterpret_cast<EndpointHandle>(0x1);
    EndpointHandle ep2 = reinterpret_cast<EndpointHandle>(0x2);
    ChannelHandle ch1[] = {0x1000};
    ChannelHandle ch2[] = {0x2000};
    ASSERT_EQ(SharedJettyMgr::GetInstance().RegisterChannels(ep1, ch1, 1), HCCL_SUCCESS);
    ASSERT_EQ(SharedJettyMgr::GetInstance().RegisterChannels(ep2, ch2, 1), HCCL_SUCCESS);

    SharedJettyMgr::GetInstance().UnregisterEndpoint(ep1);
    EXPECT_FALSE(SharedJettyMgr::GetInstance().HasContext(ep1));
    EXPECT_TRUE(SharedJettyMgr::GetInstance().HasContext(ep2));
}

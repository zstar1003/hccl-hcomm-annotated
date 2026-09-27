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
#include <functional>
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>

#define private public
#define protected public
#include "shared_jetty_channel_helper.h"
#include "endpoint.h"
#include "urma_endpoint.h"
#include "comm_queue_context/jetty_context.h"
#include "dev_ub_connection.h"
#include "shared_jetty_connection_adapter.h"
#include "ut_shared_jetty_test_helper.h"
#undef private
#undef protected

using namespace hcomm;
using namespace Hccl;

namespace {
class StubEndpointForHelper : public UrmaEndpoint {
public:
    StubEndpointForHelper() : UrmaEndpoint(MakeDesc())
    {
        // GetCommQueueContext 已上移为 Endpoint 基类虚函数（stub 经 UrmaEndpoint override 生效）；
        // 此处预创建供测试断言，与 GetCommQueueContext() 首调的延迟创建效果等价。
        jettyContext_ = std::make_unique<JettyContext>();
    }

    HcclResult Init() override { return HCCL_SUCCESS; }

private:
    static EndpointDesc MakeDesc()
    {
        EndpointDesc desc{};
        desc.protocol = COMM_PROTOCOL_UB_CTP;
        desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        Hccl::IpAddress localIp("127.0.0.1");
        desc.commAddr.addr = localIp.GetBinaryAddress().addr;
        desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
        desc.loc.device.devPhyId = 3;
        return desc;
    }
};

inline std::function<std::unique_ptr<DevUbConnection>()> MakeCountingFactory(uint32_t& count)
{
    return [&count]() -> std::unique_ptr<DevUbConnection> {
        count++;
        return MakeTestJettyConnection();
    };
}

} // namespace

class SharedJettyChannelHelperTest : public testing::Test {
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
    }
    void TearDown() override
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }
};

TEST_F(SharedJettyChannelHelperTest, Ut_AcquireSharedJettyForChannel_When_NullEndpoint_Expect_HCCL_E_PARA)
{
    auto conn = MakeTestJettyConnection();
    auto factory = []() {
        return MakeTestJettyConnection();
    };
    JettyContext::Ctx outCtx;
    EXPECT_EQ(AcquireSharedJettyForChannel(nullptr, conn.get(), factory, outCtx), HCCL_E_PARA);
}

TEST_F(SharedJettyChannelHelperTest, Ut_AcquireSharedJettyForChannel_When_NullConnection_Expect_HCCL_E_PARA)
{
    StubEndpointForHelper endpoint;
    auto factory = []() {
        return MakeTestJettyConnection();
    };
    JettyContext::Ctx outCtx;
    EXPECT_EQ(AcquireSharedJettyForChannel(&endpoint, nullptr, factory, outCtx), HCCL_E_PARA);
}

TEST_F(SharedJettyChannelHelperTest, Ut_AcquireSharedJettyForChannel_When_FirstCreate_Expect_Success)
{
    StubEndpointForHelper endpoint;
    auto conn = MakeTestJettyConnection(DevUbConnection::JettyMode::EXTERNAL_INJECT);
    uint32_t factoryCallCount = 0;
    auto factory = MakeCountingFactory(factoryCallCount);

    JettyContext::Ctx outCtx;
    EXPECT_EQ(AcquireSharedJettyForChannel(&endpoint, conn.get(), factory, outCtx), HCCL_SUCCESS);
    EXPECT_EQ(factoryCallCount, 1U);
    EXPECT_EQ(conn->jettyMode_, DevUbConnection::JettyMode::EXTERNAL_INJECT);
}

TEST_F(SharedJettyChannelHelperTest, Ut_AcquireSharedJettyForChannel_When_CacheHit_Expect_Success)
{
    StubEndpointForHelper endpoint;
    auto conn1 = MakeTestJettyConnection(DevUbConnection::JettyMode::EXTERNAL_INJECT);
    auto conn2 = MakeTestJettyConnection(DevUbConnection::JettyMode::EXTERNAL_INJECT);
    uint32_t factoryCallCount = 0;
    auto factory = MakeCountingFactory(factoryCallCount);

    JettyContext::Ctx outCtx;
    ASSERT_EQ(AcquireSharedJettyForChannel(&endpoint, conn1.get(), factory, outCtx), HCCL_SUCCESS);
    EXPECT_EQ(factoryCallCount, 1U);

    EXPECT_EQ(AcquireSharedJettyForChannel(&endpoint, conn2.get(), factory, outCtx), HCCL_SUCCESS);
    EXPECT_EQ(factoryCallCount, 1U);
    EXPECT_EQ(conn2->jettyMode_, DevUbConnection::JettyMode::EXTERNAL_INJECT);
}

TEST_F(SharedJettyChannelHelperTest, Ut_AcquireSharedJettyForChannel_When_ReleaseDecrementsRef_Expect_Success)
{
    StubEndpointForHelper endpoint;
    auto conn = MakeTestJettyConnection(DevUbConnection::JettyMode::EXTERNAL_INJECT);
    auto factory = []() {
        return MakeTestJettyConnection();
    };

    JettyContext::Ctx outCtx;
    ASSERT_EQ(AcquireSharedJettyForChannel(&endpoint, conn.get(), factory, outCtx), HCCL_SUCCESS);

    EXPECT_EQ(endpoint.jettyContext_->Release(), HCCL_SUCCESS);
    EXPECT_EQ(endpoint.jettyContext_->Release(), HCCL_SUCCESS);
}

TEST_F(SharedJettyChannelHelperTest, Ut_AcquireSharedJettyForChannel_When_ReleaseAndReacquire_Expect_Success)
{
    StubEndpointForHelper endpoint;
    auto conn1 = MakeTestJettyConnection(DevUbConnection::JettyMode::EXTERNAL_INJECT);
    auto conn2 = MakeTestJettyConnection(DevUbConnection::JettyMode::EXTERNAL_INJECT);
    uint32_t factoryCallCount = 0;
    auto factory = MakeCountingFactory(factoryCallCount);

    JettyContext::Ctx outCtx;
    ASSERT_EQ(AcquireSharedJettyForChannel(&endpoint, conn1.get(), factory, outCtx), HCCL_SUCCESS);
    EXPECT_EQ(endpoint.jettyContext_->Release(), HCCL_SUCCESS);

    EXPECT_EQ(AcquireSharedJettyForChannel(&endpoint, conn2.get(), factory, outCtx), HCCL_SUCCESS);
    EXPECT_EQ(factoryCallCount, 2U);
    EXPECT_EQ(conn2->jettyMode_, DevUbConnection::JettyMode::EXTERNAL_INJECT);
}

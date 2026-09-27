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
#include "shared_jetty_connection_adapter.h"
#include "endpoint.h"
#include "urma_endpoint.h"
#include "comm_queue_context/jetty_context.h"
#include "dev_ub_connection.h"
#include "ut_shared_jetty_test_helper.h"
#undef private
#undef protected

using namespace hcomm;
using namespace Hccl;

namespace {
// 与 dev_ub_connection.cc 中 OPBASED_UB_SQ_DEPTH_MAX 对齐（构造函数把 UB_SQ_DEPTH_NOT_SET 转换为此值）
constexpr uint32_t DEFAULT_OPBASE_UB_SQ_DEPTH = 8192U;

class StubEndpointForAdapt : public UrmaEndpoint {
public:
    StubEndpointForAdapt() : UrmaEndpoint(MakeDesc())
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
        Hccl::IpAddress localIp("127.0.0.1");
        desc.protocol = COMM_PROTOCOL_UBC_CTP;
        desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        desc.commAddr.addr = localIp.GetBinaryAddress().addr;
        desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
        desc.loc.device.devPhyId = 3;
        return desc;
    }
};

JettyContext::Ctx MakeTestCtx()
{
    JettyContext::Ctx ctx;
    ctx.handle = 0x1234;
    ctx.handlePtr = reinterpret_cast<void*>(0x5678);
    ctx.jettyId = 100;
    ctx.sqBuffVa = 0xABCDEF00;
    ctx.dbAddr = 0x12345678;
    ctx.keySize = Hccl::HRT_UB_QP_KEY_MAX_LEN;
    for (uint32_t i = 0; i < ctx.keySize; ++i) {
        ctx.localQpKey[i] = static_cast<uint8_t>(i);
    }
    ctx.sqDepth = 32;
    ctx.jfcHandle = 0xFCEULL;
    ctx.cqInfo.va = 0xC0FFEEULL;
    ctx.cqInfo.id = 0x1111U;
    ctx.cqInfo.cqDepth = 64U;
    ctx.localPsn = 846930886U;
    return ctx;
}
} // namespace

class SharedJettyConnAdaptTest : public testing::Test {
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

TEST_F(SharedJettyConnAdaptTest, Ut_Inject_When_NullConnection_Expect_HCCL_E_PARA)
{
    JettyContext::Ctx ctx = MakeTestCtx();
    auto releaseCb = [](void*) {};
    EXPECT_EQ(SetSharedJettyFieldsToConn(nullptr, ctx, reinterpret_cast<void*>(0x1), releaseCb), HCCL_E_PARA);
}

TEST_F(SharedJettyConnAdaptTest, Ut_Inject_When_NullEndpointTag_Expect_HCCL_E_PARA)
{
    auto conn = MakeTestJettyConnection();
    JettyContext::Ctx ctx = MakeTestCtx();
    auto releaseCb = [](void*) {};
    EXPECT_EQ(SetSharedJettyFieldsToConn(conn.get(), ctx, nullptr, releaseCb), HCCL_E_PARA);
}

TEST_F(SharedJettyConnAdaptTest, Ut_Inject_When_Normal_Expect_Success)
{
    auto conn = MakeTestJettyConnection(DevUbConnection::JettyMode::EXTERNAL_INJECT);
    StubEndpointForAdapt endpoint;
    JettyContext::Ctx ctx = MakeTestCtx();
    auto releaseCb = [](void*) {};
    EXPECT_EQ(SetSharedJettyFieldsToConn(conn.get(), ctx, &endpoint, releaseCb), HCCL_SUCCESS);
    EXPECT_EQ(conn->jettyMode_, DevUbConnection::JettyMode::EXTERNAL_INJECT);
    EXPECT_EQ(conn->jettyHandle, static_cast<Hccl::JettyHandle>(0x1234));
    EXPECT_EQ(conn->jettyId, 100U);
    EXPECT_EQ(conn->sqBuffVa, 0xABCDEF00ULL);
    EXPECT_EQ(conn->sqDepth, 32U);
    EXPECT_EQ(conn->endpointTag_, static_cast<void*>(&endpoint));
    EXPECT_EQ(conn->jfcHandle, ctx.jfcHandle);
    EXPECT_EQ(conn->cqInfo_.va, ctx.cqInfo.va);
    EXPECT_EQ(conn->cqInfo_.id, ctx.cqInfo.id);
    EXPECT_EQ(conn->cqInfo_.cqDepth, ctx.cqInfo.cqDepth);
    EXPECT_EQ(conn->jettyImportCfg.localPsn, ctx.localPsn);
}

TEST_F(SharedJettyConnAdaptTest, Ut_Inject_When_ZeroKeySize_Expect_Success)
{
    auto conn = MakeTestJettyConnection(DevUbConnection::JettyMode::EXTERNAL_INJECT);
    StubEndpointForAdapt endpoint;
    JettyContext::Ctx ctx = MakeTestCtx();
    ctx.keySize = 0;
    auto releaseCb = [](void*) {};
    EXPECT_EQ(SetSharedJettyFieldsToConn(conn.get(), ctx, &endpoint, releaseCb), HCCL_SUCCESS);
}

TEST_F(SharedJettyConnAdaptTest, Ut_ExtractInfo_When_NullConnection_Expect_HCCL_E_PARA)
{
    JettyContext::Ctx ctx;
    EXPECT_EQ(ExtractJettyInfoFromConn(nullptr, ctx), HCCL_E_PARA);
}

TEST_F(SharedJettyConnAdaptTest, Ut_ExtractInfo_When_Normal_Expect_Success)
{
    auto conn = MakeTestJettyConnection(DevUbConnection::JettyMode::EXTERNAL_INJECT);
    StubEndpointForAdapt endpoint;
    JettyContext::Ctx injectCtx = MakeTestCtx();
    auto releaseCb = [](void*) {};
    ASSERT_EQ(SetSharedJettyFieldsToConn(conn.get(), injectCtx, &endpoint, releaseCb), HCCL_SUCCESS);

    // 设置 JFC 相关字段，验证 ExtractJettyInfoFromConn 能正确提取
    conn->rdmaHandle = reinterpret_cast<void*>(0xABCD);
    conn->jfcHandle = 12345;

    JettyContext::Ctx extractCtx;
    EXPECT_EQ(ExtractJettyInfoFromConn(conn.get(), extractCtx), HCCL_SUCCESS);
    EXPECT_EQ(extractCtx.handle, injectCtx.handle);
    EXPECT_EQ(extractCtx.jettyId, injectCtx.jettyId);
    EXPECT_EQ(extractCtx.sqBuffVa, injectCtx.sqBuffVa);
    EXPECT_EQ(extractCtx.dbAddr, injectCtx.dbAddr);
    EXPECT_EQ(extractCtx.keySize, injectCtx.keySize);
    EXPECT_EQ(extractCtx.sqDepth, injectCtx.sqDepth);
    EXPECT_EQ(extractCtx.rdmaHandle, reinterpret_cast<void*>(0xABCD));
    EXPECT_EQ(extractCtx.jfcHandle, 12345U);
    EXPECT_EQ(extractCtx.cqInfo.va, conn->cqInfo_.va);
    EXPECT_EQ(extractCtx.cqInfo.id, conn->cqInfo_.id);
    EXPECT_EQ(extractCtx.localPsn, injectCtx.localPsn);
    EXPECT_EQ(memcmp(extractCtx.localQpKey, injectCtx.localQpKey, Hccl::HRT_UB_QP_KEY_MAX_LEN), 0);
}

TEST_F(SharedJettyConnAdaptTest, Ut_ExtractInfo_When_EmptyConnection_Expect_DefaultSqDepth)
{
    auto conn = MakeTestJettyConnection();
    JettyContext::Ctx ctx;
    EXPECT_EQ(ExtractJettyInfoFromConn(conn.get(), ctx), HCCL_SUCCESS);
    EXPECT_EQ(ctx.handle, static_cast<Hccl::JettyHandle>(0));
    EXPECT_EQ(ctx.jettyId, 0U);
    EXPECT_EQ(ctx.sqDepth, DEFAULT_OPBASE_UB_SQ_DEPTH);
    EXPECT_EQ(ctx.rdmaHandle, nullptr);
    EXPECT_EQ(ctx.jfcHandle, 0U);
}

TEST_F(SharedJettyConnAdaptTest, Ut_TransferOwnership_When_NullConnection_Expect_HCCL_E_PARA)
{
    EXPECT_EQ(DetachConnJetty(nullptr), HCCL_E_PARA);
}

TEST_F(SharedJettyConnAdaptTest, Ut_TransferOwnership_When_Normal_Expect_Success)
{
    auto conn = MakeTestJettyConnection();
    EXPECT_FALSE(conn->jettyDetached_);
    EXPECT_EQ(DetachConnJetty(conn.get()), HCCL_SUCCESS);
    EXPECT_TRUE(conn->jettyDetached_);
}

TEST_F(SharedJettyConnAdaptTest, Ut_RoundTrip_InjectExtractTransfer_Expect_Consistent)
{
    auto conn = MakeTestJettyConnection(DevUbConnection::JettyMode::EXTERNAL_INJECT);
    StubEndpointForAdapt endpoint;
    JettyContext::Ctx injectCtx = MakeTestCtx();
    auto releaseCb = [](void*) {};
    ASSERT_EQ(SetSharedJettyFieldsToConn(conn.get(), injectCtx, &endpoint, releaseCb), HCCL_SUCCESS);

    conn->rdmaHandle = reinterpret_cast<void*>(0xBEEF);
    conn->jfcHandle = 67890;

    ASSERT_EQ(DetachConnJetty(conn.get()), HCCL_SUCCESS);

    JettyContext::Ctx extractCtx;
    ASSERT_EQ(ExtractJettyInfoFromConn(conn.get(), extractCtx), HCCL_SUCCESS);
    EXPECT_EQ(extractCtx.handle, injectCtx.handle);
    EXPECT_EQ(extractCtx.handlePtr, injectCtx.handlePtr);
    EXPECT_EQ(extractCtx.jettyId, injectCtx.jettyId);
    EXPECT_EQ(extractCtx.sqBuffVa, injectCtx.sqBuffVa);
    EXPECT_EQ(extractCtx.dbAddr, injectCtx.dbAddr);
    EXPECT_EQ(extractCtx.sqDepth, injectCtx.sqDepth);
    EXPECT_EQ(extractCtx.keySize, injectCtx.keySize);
    EXPECT_EQ(extractCtx.rdmaHandle, reinterpret_cast<void*>(0xBEEF));
    EXPECT_EQ(extractCtx.jfcHandle, 67890U);
    EXPECT_EQ(extractCtx.localPsn, injectCtx.localPsn);
}

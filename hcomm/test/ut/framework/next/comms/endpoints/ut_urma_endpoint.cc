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
#include "urma_endpoint.h"
#include "hcomm_res.h"
#include "hcomm_c_adpt.h"
#include "rdma_handle_manager.h"
#include "ip_address.h"
#include "hccp.h"
#include "buffer.h"
#include "endpoint.h"
#include "ub_reged_mem_mgr.h"
#include "adapter_rts_common.h"
#include "server_socket_manager.h"
#include "hccp_hdc_manager.h"
#include "ccu_channel_ctx_pool.h"
#include "hccp_ctx.h"

#define private public
#define protected public
#include "endpoint.h"
#include "rdma_handle_manager.h"
#undef protected
#undef private
using namespace hcomm;

namespace {
// UT 直连：模拟 Init 内的 DeviceServerSocketContext 构造（commAddr 从 endpointDesc 传入，方法内转换 IpAddress）
void EmplaceUrmaServerSocketContext(UrmaEndpoint& ep, EndpointLocType locType)
{
    ep.serverSocketContext_.emplace(Hccl::ConnectProtoType::UB, 0U, locType, ep.endpointDesc_.commAddr);
}
} // namespace

class UrmaEndpointTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "UrmaEndpointTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "UrmaEndpointTest tests tear down." << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in UrmaEndpointTest SetUP" << std::endl;
        Hccl::IpAddress localIp("1.0.0.0");
        endpointDesc.protocol = COMM_PROTOCOL_UBC_TP;
        endpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        endpointDesc.commAddr.addr = localIp.GetBinaryAddress().addr;
        endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
        endpointDesc.loc.device.devPhyId = 0;

        MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in UrmaEndpointTest TearDown" << std::endl;
    }

    EndpointDesc endpointDesc{};
};

TEST_F(UrmaEndpointTest, Ut_When_ServerSocketListen_Normal_Expect_HCCL_SUCCESS)
{
    auto endpoint = std::make_unique<UrmaEndpoint>(endpointDesc);
    endpoint->ccuChannelCtxPool_.reset(new (std::nothrow) CcuChannelCtxPool(0));
    EmplaceUrmaServerSocketContext(*endpoint, ENDPOINT_LOC_TYPE_DEVICE);

    MOCKER_CPP(&hcomm::ServerSocketManager::ServerSocketStartListen).stubs().will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(endpoint->GetServerSocketContext()->ServerSocketListen(60001), HCCL_SUCCESS);
}

TEST_F(UrmaEndpointTest, Ut_When_ServerSocketListen_LocTypeNotDevice_Expect_HCCL_SUCCESS)
{
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    auto endpoint = std::make_unique<UrmaEndpoint>(endpointDesc);
    EmplaceUrmaServerSocketContext(*endpoint, ENDPOINT_LOC_TYPE_HOST);

    EXPECT_EQ(endpoint->GetServerSocketContext()->ServerSocketListen(60001), HCCL_SUCCESS);
}

TEST_F(UrmaEndpointTest, Ut_When_ServerSocketListen_StartListenFailed_Expect_Error)
{
    auto endpoint = std::make_unique<UrmaEndpoint>(endpointDesc);
    endpoint->ccuChannelCtxPool_.reset(new (std::nothrow) CcuChannelCtxPool(0));
    EmplaceUrmaServerSocketContext(*endpoint, ENDPOINT_LOC_TYPE_DEVICE);

    MOCKER_CPP(&hcomm::ServerSocketManager::ServerSocketStartListen).stubs().will(returnValue(HCCL_E_INTERNAL));
    EXPECT_EQ(endpoint->GetServerSocketContext()->ServerSocketListen(60001), HCCL_E_INTERNAL);
}

TEST_F(UrmaEndpointTest, Ut_When_ServerSocketGetListenPort_Normal_Expect_HCCL_SUCCESS)
{
    auto endpoint = std::make_unique<UrmaEndpoint>(endpointDesc);
    endpoint->ccuChannelCtxPool_.reset(new (std::nothrow) CcuChannelCtxPool(0));
    EmplaceUrmaServerSocketContext(*endpoint, ENDPOINT_LOC_TYPE_DEVICE);
    uint32_t portValue = 60001;

    MOCKER_CPP(&hcomm::ServerSocketManager::ServerSocketStartListen)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBoundP(&portValue, sizeof(portValue)))
        .will(returnValue(HCCL_SUCCESS));
    uint32_t port = 0;
    EXPECT_EQ(endpoint->GetServerSocketContext()->ServerSocketGetListenPort(&port), HCCL_SUCCESS);
    EXPECT_EQ(port, 60001);
}

TEST_F(UrmaEndpointTest, Ut_When_ServerSocketGetListenPort_LocTypeNotDevice_Expect_HCCL_SUCCESS)
{
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    auto endpoint = std::make_unique<UrmaEndpoint>(endpointDesc);
    EmplaceUrmaServerSocketContext(*endpoint, ENDPOINT_LOC_TYPE_HOST);

    uint32_t port = 0;
    EXPECT_EQ(endpoint->GetServerSocketContext()->ServerSocketGetListenPort(&port), HCCL_SUCCESS);
}

TEST_F(UrmaEndpointTest, Ut_When_ServerSocketGetListenPort_AlreadyListening_Expect_ReturnCachedPort)
{
    auto endpoint = std::make_unique<UrmaEndpoint>(endpointDesc);
    endpoint->ccuChannelCtxPool_.reset(new (std::nothrow) CcuChannelCtxPool(0));
    EmplaceUrmaServerSocketContext(*endpoint, ENDPOINT_LOC_TYPE_DEVICE);
    uint32_t cachedPort = 50001;
    // dynamicPort_ 收入 ServerSocketContext 基类
    endpoint->serverSocketContext_->dynamicPort_ = cachedPort;

    uint32_t port = 0;
    EXPECT_EQ(endpoint->GetServerSocketContext()->ServerSocketGetListenPort(&port), HCCL_SUCCESS);
    EXPECT_EQ(port, cachedPort);
}

TEST_F(UrmaEndpointTest, Ut_When_ServerSocketGetListenPort_StartListenReturnsPortZero_Expect_HCCL_E_NETWORK)
{
    auto endpoint = std::make_unique<UrmaEndpoint>(endpointDesc);
    endpoint->ccuChannelCtxPool_.reset(new (std::nothrow) CcuChannelCtxPool(0));
    EmplaceUrmaServerSocketContext(*endpoint, ENDPOINT_LOC_TYPE_DEVICE);
    uint32_t portValue = 0;

    MOCKER_CPP(&hcomm::ServerSocketManager::ServerSocketStartListen)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBoundP(&portValue, sizeof(portValue)))
        .will(returnValue(HCCL_SUCCESS));
    uint32_t port = 0;
    EXPECT_EQ(endpoint->GetServerSocketContext()->ServerSocketGetListenPort(&port), HCCL_E_NETWORK);
}

TEST_F(UrmaEndpointTest, Ut_When_ServerSocketGetListenPort_StartListenReturnsInvalidPort_Expect_HCCL_E_NETWORK)
{
    auto endpoint = std::make_unique<UrmaEndpoint>(endpointDesc);
    endpoint->ccuChannelCtxPool_.reset(new (std::nothrow) CcuChannelCtxPool(0));
    EmplaceUrmaServerSocketContext(*endpoint, ENDPOINT_LOC_TYPE_DEVICE);
    uint32_t portValue = HCCL_INVALID_PORT;

    MOCKER_CPP(&hcomm::ServerSocketManager::ServerSocketStartListen)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBoundP(&portValue, sizeof(portValue)))
        .will(returnValue(HCCL_SUCCESS));
    uint32_t port = 0;
    EXPECT_EQ(endpoint->GetServerSocketContext()->ServerSocketGetListenPort(&port), HCCL_E_NETWORK);
}

TEST_F(UrmaEndpointTest, Ut_When_ServerSocketGetListenPort_StartListenFailed_Expect_Error)
{
    auto endpoint = std::make_unique<UrmaEndpoint>(endpointDesc);
    endpoint->ccuChannelCtxPool_.reset(new (std::nothrow) CcuChannelCtxPool(0));
    EmplaceUrmaServerSocketContext(*endpoint, ENDPOINT_LOC_TYPE_DEVICE);

    MOCKER_CPP(&hcomm::ServerSocketManager::ServerSocketStartListen).stubs().will(returnValue(HCCL_E_INTERNAL));
    uint32_t port = 0;
    EXPECT_EQ(endpoint->GetServerSocketContext()->ServerSocketGetListenPort(&port), HCCL_E_INTERNAL);
}

TEST_F(UrmaEndpointTest, Ut_When_IsCtxHandleValid_NullCtxHandle_Expect_False)
{
    auto endpoint = std::make_unique<UrmaEndpoint>(endpointDesc);
    endpoint->ctxHandle_ = nullptr;
    EXPECT_FALSE(endpoint->IsCtxHandleValid());
}

TEST_F(UrmaEndpointTest, Ut_When_IsCtxHandleValid_ValidHandle_Expect_True)
{
    auto endpoint = std::make_unique<UrmaEndpoint>(endpointDesc);
    Hccl::RdmaHandle fakeHandle = (Hccl::RdmaHandle)0x1234;
    endpoint->ctxHandle_ = fakeHandle;
    Hccl::RdmaHandleManager::GetInstance().activeHandles_.insert(fakeHandle);
    EXPECT_TRUE(endpoint->IsCtxHandleValid());
    Hccl::RdmaHandleManager::GetInstance().activeHandles_.erase(fakeHandle);
}

TEST_F(UrmaEndpointTest, Ut_When_IsCtxHandleValid_InvalidatedHandle_Expect_False)
{
    auto endpoint = std::make_unique<UrmaEndpoint>(endpointDesc);
    Hccl::RdmaHandle fakeHandle = (Hccl::RdmaHandle)0x5678;
    endpoint->ctxHandle_ = fakeHandle;
    EXPECT_FALSE(endpoint->IsCtxHandleValid());
}

TEST_F(UrmaEndpointTest, Ut_When_IsCtxHandleValid_ConstObj_Expect_False)
{
    auto endpoint = std::make_unique<UrmaEndpoint>(endpointDesc);
    endpoint->ctxHandle_ = nullptr;
    const UrmaEndpoint* constEndpoint = endpoint.get();
    EXPECT_FALSE(constEndpoint->IsCtxHandleValid());
}

TEST_F(UrmaEndpointTest, Ut_When_GetAsyncEvents_CtxHandleInvalidated_Expect_HCCL_E_INTERNAL)
{
    auto endpoint = std::make_unique<UrmaEndpoint>(endpointDesc);
    endpoint->ccuChannelCtxPool_.reset(new (std::nothrow) CcuChannelCtxPool(0));
    Hccl::RdmaHandle fakeHandle = (Hccl::RdmaHandle)0xABCD;
    endpoint->ctxHandle_ = fakeHandle;

    u32 devPhyId = 0;
    struct AsyncEvent events[ASYNC_EVENT_MAX_NUM];
    u32 num = ASYNC_EVENT_MAX_NUM;
    u32 interfaceVersion = 2;

    MOCKER(RaGetInterfaceVersion)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&interfaceVersion, sizeof(interfaceVersion)))
        .will(returnValue(0));

    EXPECT_EQ(endpoint->GetAsyncEvents(devPhyId, events, num), HCCL_E_INTERNAL);
    GlobalMockObject::verify();
}

TEST_F(UrmaEndpointTest, Ut_When_GetAsyncEvents_CtxHandleValid_Expect_RaCtxGetAsyncEventsCalled)
{
    auto endpoint = std::make_unique<UrmaEndpoint>(endpointDesc);
    endpoint->ccuChannelCtxPool_.reset(new (std::nothrow) CcuChannelCtxPool(0));
    Hccl::RdmaHandle fakeHandle = (Hccl::RdmaHandle)0xDCBA;
    endpoint->ctxHandle_ = fakeHandle;
    Hccl::RdmaHandleManager::GetInstance().activeHandles_.insert(fakeHandle);

    u32 devPhyId = 0;
    struct AsyncEvent events[ASYNC_EVENT_MAX_NUM];
    u32 num = ASYNC_EVENT_MAX_NUM;
    u32 interfaceVersion = 2;

    MOCKER(RaGetInterfaceVersion)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&interfaceVersion, sizeof(interfaceVersion)))
        .will(returnValue(0));
    MOCKER(RaCtxGetAsyncEvents).stubs().will(returnValue(0));

    EXPECT_EQ(endpoint->GetAsyncEvents(devPhyId, events, num), HCCL_SUCCESS);
    Hccl::RdmaHandleManager::GetInstance().activeHandles_.erase(fakeHandle);
    GlobalMockObject::verify();
}

TEST_F(UrmaEndpointTest, Ut_When_GetAsyncEvents_Expect_HCCL_E_INTERNAL)
{
    auto endpoint = std::make_unique<UrmaEndpoint>(endpointDesc);
    endpoint->ccuChannelCtxPool_.reset(new (std::nothrow) CcuChannelCtxPool(0));
    Hccl::RdmaHandle fakeHandle = (Hccl::RdmaHandle)0xDCBA;
    endpoint->ctxHandle_ = fakeHandle;
    Hccl::RdmaHandleManager::GetInstance().activeHandles_.insert(fakeHandle);

    u32 devPhyId = 0;
    struct AsyncEvent events[ASYNC_EVENT_MAX_NUM];
    u32 num = ASYNC_EVENT_MAX_NUM;
    u32 interfaceVersion = 2;

    MOCKER(RaGetInterfaceVersion)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&interfaceVersion, sizeof(interfaceVersion)))
        .will(returnValue(0));
    MOCKER(RaCtxGetAsyncEvents).stubs().will(returnValue(4));

    EXPECT_EQ(endpoint->GetAsyncEvents(devPhyId, events, num), HCCL_E_INTERNAL);
    Hccl::RdmaHandleManager::GetInstance().activeHandles_.erase(fakeHandle);
    GlobalMockObject::verify();
}

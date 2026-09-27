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
#include "cpu_urma_endpoint.h"
#include "hcomm_res.h"
#include "hcomm_c_adpt.h"
#include "rdma_handle_manager.h"
#include "ip_address.h"
#include "buffer.h"
#include "hccp.h"
#include "endpoint.h"
#include "ub_reged_mem_mgr.h"
#include "adapter_rts_common.h"
#include "server_socket_manager.h"
#include "hccp_peer_manager.h"

#define private public
using namespace hcomm;

namespace {
// CpuUrmaEndpoint::regedMemMgr_ 为 shared_ptr<UbRegedMemMgr>，mock 需派生自 UbRegedMemMgr
class FakeRegedMemMgrForEndpointUt : public UbRegedMemMgr {
public:
    FakeRegedMemMgrForEndpointUt() : UbRegedMemMgr(nullptr) {}
    HcclResult RegisterMemory(const HcommMem*, const char*, void** memHandle) override
    {
        *memHandle = reinterpret_cast<void*>(0x42ULL);
        return HCCL_SUCCESS;
    }
    HcclResult UnregisterMemory(void*) override { return HCCL_SUCCESS; }
    HcclResult MemoryExport(const EndpointDesc&, void*, void**, uint32_t*) override { return HCCL_SUCCESS; }
    HcclResult MemoryImport(const void*, uint32_t, HcommMem*) override { return HCCL_SUCCESS; }
    HcclResult MemoryUnimport(const void*, uint32_t) override { return HCCL_SUCCESS; }
    HcclResult GetAllMemHandles(void**, uint32_t*) override { return HCCL_SUCCESS; }
};
} // namespace

class CpuUrmaEndpointTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CpuUrmaEndpointTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "CpuUrmaEndpointTest tests tear down." << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in CpuUrmaEndpointTest SetUP" << std::endl;
        Hccl::IpAddress localIp("1.0.0.0");
        endpointDesc.protocol = COMM_PROTOCOL_UB_CTP;
        endpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        endpointDesc.commAddr.addr = localIp.GetBinaryAddress().addr;
        endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
        rdmaHandle = (void*)0x1000000;

        MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));
        // 出参 devPhyId 需赋合法值：批次1后它作为 GetDeviceResMgr(devPhyId) 的设备数组下标
        unsigned int devicePhyId = 0U;
        MOCKER(hrtGetDevicePhyIdByIndex)
            .stubs()
            .with(mockcpp::any(), outBound(devicePhyId))
            .will(returnValue(HCCL_SUCCESS));
        MOCKER(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(rdmaHandle));
        MOCKER(RaSocketSetWhiteListStatus).stubs().will(returnValue(0));
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CpuUrmaEndpointTest TearDown" << std::endl;
    }

    EndpointDesc endpointDesc{};
    RdmaHandle rdmaHandle{nullptr};
};

TEST_F(CpuUrmaEndpointTest, Ut_When_Normal_Init_Expect_HCCL_SUCCESS)
{
    auto endpoint = std::make_unique<CpuUrmaEndpoint>(endpointDesc);
    EXPECT_EQ(endpoint->Init(), HCCL_SUCCESS);
}

TEST_F(CpuUrmaEndpointTest, Ut_When_ServerSocketListen_Normal_Expect_HCCL_SUCCESS)
{
    auto endpoint = std::make_unique<CpuUrmaEndpoint>(endpointDesc);
    EXPECT_EQ(endpoint->Init(), HCCL_SUCCESS);
    MOCKER_CPP(&hcomm::ServerSocketManager::ServerSocketStartListen).stubs().will(returnValue(HCCL_SUCCESS));
    // 监听 3 方法由 ServerSocketContext 承载，经 GetServerSocketContext() 访问
    EXPECT_EQ(endpoint->GetServerSocketContext()->ServerSocketListen(60001), HCCL_SUCCESS);
}

TEST_F(CpuUrmaEndpointTest, Ut_When_ServerSocketStopListen_Normal_Expect_HCCL_SUCCESS)
{
    auto endpoint = std::make_unique<CpuUrmaEndpoint>(endpointDesc);
    EXPECT_EQ(endpoint->Init(), HCCL_SUCCESS);
    MOCKER_CPP(&hcomm::ServerSocketManager::ServerSocketStopListen).stubs().will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(endpoint->GetServerSocketContext()->ServerSocketStopListen(60001), HCCL_SUCCESS);
}

TEST_F(CpuUrmaEndpointTest, Ut_When_RegisterMemory_Normal_Expect_HCCL_SUCCESS)
{
    auto endpoint = std::make_unique<CpuUrmaEndpoint>(endpointDesc);
    EXPECT_EQ(endpoint->Init(), HCCL_SUCCESS);

    endpoint->regedMemMgr_ = std::make_shared<FakeRegedMemMgrForEndpointUt>();
    HcommMem mem;
    mem.type = COMM_MEM_TYPE_HOST;
    mem.addr = reinterpret_cast<void*>(0x1000U);
    mem.size = 10;
    void* memHandle = nullptr;
    // RegisterMemory 在 RegedMemMgr 上
    EXPECT_EQ(endpoint->GetRegedMemMgr()->RegisterMemory(&mem, "test", &memHandle), HCCL_SUCCESS);
}

TEST_F(CpuUrmaEndpointTest, Ut_When_UnregisterMemory_Normal_Expect_HCCL_SUCCESS)
{
    auto endpoint = std::make_unique<CpuUrmaEndpoint>(endpointDesc);
    EXPECT_EQ(endpoint->Init(), HCCL_SUCCESS);

    endpoint->regedMemMgr_ = std::make_shared<FakeRegedMemMgrForEndpointUt>();
    void* memHandle = (void*)0x12345678;
    // UnregisterMemory 在 RegedMemMgr 上
    EXPECT_EQ(endpoint->GetRegedMemMgr()->UnregisterMemory(memHandle), HCCL_SUCCESS);
}

TEST_F(CpuUrmaEndpointTest, Ut_When_ServerSocketGetListenPort_Normal_Expect_HCCL_SUCCESS)
{
    auto endpoint = std::make_unique<CpuUrmaEndpoint>(endpointDesc);
    EXPECT_EQ(endpoint->Init(), HCCL_SUCCESS);
    uint32_t port = 0;
    uint32_t portValue = 60001;

    MOCKER_CPP(&hcomm::ServerSocketManager::ServerSocketStartListen)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBoundP(&portValue, sizeof(portValue)))
        .will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(endpoint->GetServerSocketContext()->ServerSocketGetListenPort(&port), HCCL_SUCCESS);
}

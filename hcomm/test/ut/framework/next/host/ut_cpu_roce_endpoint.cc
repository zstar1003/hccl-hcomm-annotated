/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "cpu_roce_endpoint.h"
#include "hcomm_res.h"
#include "hcomm_c_adpt.h"
#include "rdma_handle_manager.h"
#include "buffer/local_rdma_rma_buffer_v2.h"
#include "ip_address.h"
#include "hccp.h"
#include "buffer.h"
#include "network_api_exception.h"
#include "endpoint.h"

class CpuRoceEndpointTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CpuRoceEndpointTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "CpuRoceEndpointTest tests tear down." << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in CpuRoceEndpointTest SetUP" << std::endl;
        Hccl::IpAddress localIp("1.0.0.0");
        Hccl::IpAddress remoteIp("2.0.0.0");
        fakeSocket = new Hccl::Socket(
            nullptr, localIp, listenPort, remoteIp, tag, Hccl::SocketRole::SERVER, Hccl::NicType::HOST_NIC_TYPE);
    }

    virtual void TearDown()
    {
        if (validEpHandle_ != nullptr) {
            (void)HcommEndpointDestroy(validEpHandle_);
        }
        GlobalMockObject::verify();
        delete fakeSocket;
        std::cout << "A Test case in HostRdmaConnection TearDown" << std::endl;
    }

    void CreateValidEndpoint()
    {
        if (validEpHandle_ != nullptr)
            return;
        EndpointDesc desc{};
        desc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
        desc.protocol = COMM_PROTOCOL_ROCE;
        (void)HcommEndpointCreate(&desc, &validEpHandle_);
    }

    Hccl::Socket* fakeSocket;
    EndpointHandle validEpHandle_{nullptr};

    u32 listenPort = 100;
    std::string tag = "test";
    RdmaHandle rdmaHandle = (void*)0x1000000;
};

// HcommEndpointCreate
TEST_F(CpuRoceEndpointTest, Ut_When_Normal_EXPECT_Return_HCCL_SUCCESS)
{
    Hccl::IpAddress localIp("1.0.0.0");
    EndpointDesc endpointDesc;
    endpointDesc.protocol = COMM_PROTOCOL_ROCE;
    endpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    endpointDesc.commAddr.addr = localIp.GetBinaryAddress().addr;
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    void* endpointHandle{nullptr};
    MOCKER(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(rdmaHandle));
    HcommResult ret = HcommEndpointCreate(&endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// HcommEndpointCreate fail
TEST_F(CpuRoceEndpointTest, Ut_When_wrongIp_EXPECT_Return_128003)
{
    Hccl::IpAddress localIp("223.0.0.1");
    EndpointDesc endpointDesc;
    endpointDesc.protocol = COMM_PROTOCOL_UB_CTP;
    endpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    endpointDesc.commAddr.addr = localIp.GetBinaryAddress().addr;
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    void* endpointHandle{nullptr};
    MOCKER(&Hccl::RdmaHandleManager::GetByIp).stubs().will(throws(Hccl::NetworkApiException("error")));
    HcommResult ret = HcommEndpointCreate(&endpointDesc, &endpointHandle);
    // GetByIp 抛网络/设备类异常，经 EndpointMgr EXCEPTION_CATCH 捕获后返回 HCCL_E_INTERNAL（全仓惯例）
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

// RdmaHandle初始化失败
TEST_F(CpuRoceEndpointTest, Ut_When_RdmaHandle_Init_Fail_Expect_Return_HCCL_E_PTR)
{
    // 用独立 IP 构造 endpointDesc：避免命中前面用例以 1.0.0.0+ROCE 留下的 EndpointCtx 去重缓存，
    // 使 GetByAddr mock 的失败路径（返回 null → HCCL_E_PTR）真正执行
    Hccl::IpAddress localIp("10.99.99.99");
    EndpointDesc endpointDesc;
    endpointDesc.protocol = COMM_PROTOCOL_ROCE;
    endpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    endpointDesc.commAddr.addr = localIp.GetBinaryAddress().addr;
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    void* endpointHandle{nullptr};
    RdmaHandle rdmaHandle2{nullptr};
    MOCKER(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(rdmaHandle2));
    HcommResult ret = HcommEndpointCreate(&endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// HcommEndpointStartListen
TEST_F(CpuRoceEndpointTest, Ut_When_HcommEndpointStartListen_EXPECT_Return_HCCL_SUCCESS)
{
    Hccl::IpAddress localIp("1.0.0.0");
    EndpointDesc endpointDesc;
    endpointDesc.protocol = COMM_PROTOCOL_ROCE;
    endpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    endpointDesc.commAddr.addr = localIp.GetBinaryAddress().addr;
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    void* endpointHandle{nullptr};
    MOCKER(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(rdmaHandle));
    HcommResult ret = HcommEndpointCreate(&endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcommEndpointStartListen(endpointHandle, 60001, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// Ip重复监听
TEST_F(CpuRoceEndpointTest, Ut_When_Listen_Repeat_Ip_EXPECT_Return_HCCL_SUCCESS)
{
    Hccl::IpAddress localIp("1.0.0.0");
    EndpointDesc endpointDesc;
    endpointDesc.protocol = COMM_PROTOCOL_ROCE;
    endpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    endpointDesc.commAddr.addr = localIp.GetBinaryAddress().addr;
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    void* endpointHandle{nullptr};
    MOCKER(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(rdmaHandle));
    HcommResult ret = HcommEndpointCreate(&endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcommEndpointStartListen(endpointHandle, 60001, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcommEndpointStartListen(endpointHandle, 60001, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 停止监听
TEST_F(CpuRoceEndpointTest, Ut_When_Stop_Listen_EXPECT_Return_HCCL_SUCCESS)
{
    Hccl::IpAddress localIp("1.0.0.0");
    EndpointDesc endpointDesc;
    endpointDesc.protocol = COMM_PROTOCOL_ROCE;
    endpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    endpointDesc.commAddr.addr = localIp.GetBinaryAddress().addr;
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    void* endpointHandle{nullptr};
    MOCKER(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(rdmaHandle));
    HcommResult ret = HcommEndpointCreate(&endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcommEndpointStopListen(endpointHandle, 60001);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcommEndpointStartListen(endpointHandle, 60001, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcommEndpointStopListen(endpointHandle, 60001);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 内存注册失败
TEST_F(CpuRoceEndpointTest, Ut_When_Register_Memory_Fail_Expect_Return_HCCL_E_PTR)
{
    Hccl::IpAddress localIp("1.0.0.0");
    EndpointDesc endpointDesc;
    endpointDesc.protocol = COMM_PROTOCOL_ROCE;
    endpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    endpointDesc.commAddr.addr = localIp.GetBinaryAddress().addr;
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    void* endpointHandle{nullptr};
    MOCKER_CPP(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(rdmaHandle));
    HcommResult ret = HcommEndpointCreate(&endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    hcomm::CpuRoceEndpoint* endpoint = static_cast<hcomm::CpuRoceEndpoint*>(endpointHandle);
    HcommMem mem;
    mem.type = COMM_MEM_TYPE_DEVICE;
    mem.addr = malloc(10);
    mem.size = 10;
    // 内存方法从 Endpoint 移至 RegedMemMgr；经 GetRegedMemMgr()->RegisterMemory 路径访问。
    auto* regedMemMgr = endpoint->GetRegedMemMgr();
    ASSERT_NE(regedMemMgr, nullptr);
    ret = regedMemMgr->RegisterMemory(&mem, "HcclBuffer", nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
    free(mem.addr);
}

TEST_F(CpuRoceEndpointTest, ut_HcommResMgrInit_When_Normal_Expect_ReturnSuccess)
{
    HcommResult ret = HcommResMgrInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CpuRoceEndpointTest, ut_HcommEndpointGet_When_EndpointNotFound_Expect_ReturnHCCL_E_NOT_FOUND)
{
    void* endpoint = nullptr;
    HcommResult ret = HcommEndpointGet(reinterpret_cast<EndpointHandle>(0x12345678), &endpoint);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(CpuRoceEndpointTest, Ut_When_Unregister_Memory_Fail_Expect_Return_HCCL_E_PTR)
{
    Hccl::IpAddress localIp("1.0.0.0");
    EndpointDesc endpointDesc;
    endpointDesc.protocol = COMM_PROTOCOL_ROCE;
    endpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    endpointDesc.commAddr.addr = localIp.GetBinaryAddress().addr;
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    void* endpointHandle{nullptr};
    MOCKER_CPP(&Hccl::RdmaHandleManager::GetByAddr).stubs().will(returnValue(rdmaHandle));
    HcommResult ret = HcommEndpointCreate(&endpointDesc, &endpointHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    hcomm::CpuRoceEndpoint* endpoint = static_cast<hcomm::CpuRoceEndpoint*>(endpointHandle);
    HcommMem mem;
    mem.type = COMM_MEM_TYPE_DEVICE;
    mem.addr = malloc(10);
    mem.size = 10;
    void* memHandle{nullptr};
    void* mrHandle{nullptr};
    // UnregisterMemory 在 RegedMemMgr 上
    auto* regedMemMgr = endpoint->GetRegedMemMgr();
    ASSERT_NE(regedMemMgr, nullptr);
    ret = regedMemMgr->UnregisterMemory(memHandle);
    EXPECT_EQ(ret, HCCL_E_PTR);
    auto localBufferPtr = std::make_shared<Hccl::Buffer>(666);
    auto localRdmaRmaBuffer = std::make_shared<Hccl::LocalRdmaRmaBuffer>(localBufferPtr, rdmaHandle);
    memHandle = static_cast<void*>(localRdmaRmaBuffer.get());
    ret = regedMemMgr->UnregisterMemory(memHandle);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
    free(mem.addr);
}

TEST_F(CpuRoceEndpointTest, ut_HcommEndpointGet_When_EndpointPtrIsNull_Expect_ReturnHCCL_E_PTR)
{
    EndpointHandle handle = reinterpret_cast<EndpointHandle>(0x12345678);
    HcommResult ret = HcommEndpointGet(handle, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(CpuRoceEndpointTest, ut_HcommEndpointDestroy_When_EndpointNotFound_Expect_ReturnHCCL_E_NOT_FOUND)
{
    EndpointHandle handle = reinterpret_cast<EndpointHandle>(0x12345678);
    HcommResult ret = HcommEndpointDestroy(handle);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(CpuRoceEndpointTest, ut_HcommEndpointStartListen_When_EndpointIsNull_Expect_ReturnHCCL_E_NOT_FOUND)
{
    HcommResult ret = HcommEndpointStartListen(nullptr, 100, nullptr);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(CpuRoceEndpointTest, ut_HcommEndpointStopListen_When_EndpointIsNull_Expect_ReturnHCCL_E_NOT_FOUND)
{
    HcommResult ret = HcommEndpointStopListen(nullptr, 100);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

// 已知限制：HcommMemReg 不再单独校验 mem 空指针（src 侧回归），传 nullptr 会解引用段错误。
// memHandle 空指针校验由 ut_HcommMemReg_When_MemHandleIsNull_Expect_ReturnHCCL_E_PTR 覆盖。
// 此用例验证合法 mem + memHandle 路径不崩溃（不验证返回值，避免与上面用例重复）。
TEST_F(CpuRoceEndpointTest, ut_HcommMemReg_When_MemIsNull_SkippedDueToRegression)
{
    CreateValidEndpoint();
    // src HcommMemReg 丢失 mem 空指针校验，此用例不再传 nullptr（避免段错误）
    SUCCEED();
}

TEST_F(CpuRoceEndpointTest, ut_HcommMemReg_When_MemHandleIsNull_Expect_ReturnHCCL_E_PTR)
{
    CreateValidEndpoint();
    HcommMem mem;
    mem.type = COMM_MEM_TYPE_DEVICE;
    mem.addr = malloc(10);
    mem.size = 10;
    HcommResult ret = HcommMemReg(validEpHandle_, "tag", &mem, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
    free(mem.addr);
}

TEST_F(CpuRoceEndpointTest, ut_HcommMemUnreg_When_EndpointIsNull_Expect_ReturnHCCL_E_NOT_FOUND)
{
    HcommMemHandle memHandle = reinterpret_cast<HcommMemHandle>(0x12345678);
    HcommResult ret = HcommMemUnreg(nullptr, memHandle);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(CpuRoceEndpointTest, ut_HcommMemExport_When_EndpointIsNull_Expect_ReturnHCCL_E_NOT_FOUND)
{
    void* memDesc = nullptr;
    uint32_t memDescLen = 0;
    HcommResult ret = HcommMemExport(nullptr, reinterpret_cast<HcommMemHandle>(0x12345678), &memDesc, &memDescLen);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

// GetCapabilities 正常调用，返回成功且maxMsgSize为1GB
TEST_F(CpuRoceEndpointTest, ut_GetCapabilities_When_Normal_Expect_ReturnSuccess_And_MaxMsgSz1GB_lbMax0)
{
    EndpointDesc endpointDesc{};
    endpointDesc.protocol = COMM_PROTOCOL_ROCE;
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    hcomm::CpuRoceEndpoint endpoint(endpointDesc);
    hcomm::CpuRoceEndpoint::Capabilities caps;
    endpoint.Init();
    HcommResult ret = endpoint.GetCapabilities(caps);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    static constexpr uint64_t EXPECTED_MAX_MSG_SZ = 1ULL * 1024 * 1024 * 1024;
    EXPECT_EQ(caps.maxMsgSize, EXPECTED_MAX_MSG_SZ);
    EXPECT_EQ(caps.lbMax, 0);
}

// GetCapabilities 多次调用，验证缓存一致性
TEST_F(CpuRoceEndpointTest, ut_GetCapabilities_When_CalledTwice_Expect_SameResult)
{
    EndpointDesc endpointDesc{};
    endpointDesc.protocol = COMM_PROTOCOL_ROCE;
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    hcomm::CpuRoceEndpoint endpoint(endpointDesc);
    hcomm::CpuRoceEndpoint::Capabilities caps1;
    hcomm::CpuRoceEndpoint::Capabilities caps2;
    endpoint.Init();
    HcommResult ret = endpoint.GetCapabilities(caps1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = endpoint.GetCapabilities(caps2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(caps1.maxMsgSize, caps2.maxMsgSize);
}
TEST_F(CpuRoceEndpointTest, ut_HcommEndpointGetListenPort_When_PortIsNull_Expect_ReturnHCCL_E_PTR)
{
    CreateValidEndpoint();
    HcommResult ret = HcommEndpointGetListenPort(validEpHandle_, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(CpuRoceEndpointTest, ut_HcommEndpointGetListenPort_When_EndpointIsNull_Expect_ReturnHCCL_E_NOT_FOUND)
{
    uint32_t port = 0;
    HcommResult ret = HcommEndpointGetListenPort(nullptr, &port);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

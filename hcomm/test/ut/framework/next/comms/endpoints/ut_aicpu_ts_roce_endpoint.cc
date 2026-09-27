/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <arpa/inet.h>

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>
#include <string>
#include <vector>

#include "adapter_rts_common.h"
#include "hccl_net_dev.h"
#include "hccl_network.h"
#include "hccl_socket.h"
#include "network_manager_pub.h"
#include "reged_mem_mgr.h"
#include "endpoint.h"

#define private public
#include "aicpu_ts_roce_endpoint.h"
#include "aicpu_ts_roce_reged_mem_mgr.h"
#include "server_socket_context/aicpu_ts_roce_server_socket_context.h"
#undef private

using namespace hcomm;

namespace {
HcclResult StubHrtGetDeviceWriteZeroEp(s32* deviceLogicId)
{
    if (deviceLogicId != nullptr) {
        *deviceLogicId = 0;
    }
    return HCCL_SUCCESS;
}

HcclResult StubCreateRdmaHandleEp(
    hccl::NetworkManager* /*self*/, const hccl::HcclIpAddress& /*ipAddr*/, bool /*isBackup*/, NetworkMode /*netMode*/,
    NotifyTypeT /*notifyType*/, HcclNetDevDeployment /*netDevDeployment*/)
{
    return HCCL_SUCCESS;
}

HcclResult StubGetRdmaHandleByIpAddrEp(
    hccl::NetworkManager* /*self*/, const hccl::HcclIpAddress& /*ipAddr*/, RdmaHandle& rdmaHandle)
{
    static int kFakeRdma = 0;
    rdmaHandle = reinterpret_cast<RdmaHandle>(&kFakeRdma);
    return HCCL_SUCCESS;
}

HcclResult StubHcclSocketAcceptForEp(
    hccl::HcclSocket* /*self*/, const std::string& /*tag*/, std::shared_ptr<hccl::HcclSocket>& socket,
    u32 /*acceptTimeOut*/)
{
    socket = std::make_shared<hccl::HcclSocket>(static_cast<HcclNetDevCtx>(nullptr), 18888U);
    return HCCL_SUCCESS;
}

// AicpuTsRoceEndpoint::regedMemMgr_ 为 shared_ptr<AicpuTsRoceRegedMemMgr>，mock 需派生自 AicpuTsRoceRegedMemMgr
class FakeRegedMemMgrForEndpointUt : public AicpuTsRoceRegedMemMgr {
public:
    FakeRegedMemMgrForEndpointUt() : AicpuTsRoceRegedMemMgr(nullptr, nullptr) {}
    HcclResult RegisterMemory(const HcommMem*, const char*, void** memHandle) override
    {
        *memHandle = reinterpret_cast<void*>(0x42ULL);
        return HCCL_SUCCESS;
    }
    HcclResult UnregisterMemory(void*) override { return HCCL_SUCCESS; }
    HcclResult MemoryExport(const EndpointDesc&, void*, void** memDesc, uint32_t* memDescLen) override
    {
        static char kBlob[] = {'r', 'd', 'm', 'a'};
        *memDesc = static_cast<void*>(kBlob);
        *memDescLen = sizeof(kBlob);
        return HCCL_SUCCESS;
    }
    HcclResult MemoryImport(const void*, uint32_t, HcommMem*) override { return HCCL_SUCCESS; }
    HcclResult MemoryUnimport(const void*, uint32_t) override { return HCCL_SUCCESS; }
    HcclResult GetAllMemHandles(void** memHandles, uint32_t* memHandleNum) override
    {
        *memHandles = nullptr;
        *memHandleNum = 0U;
        return HCCL_SUCCESS;
    }
};
} // namespace

HcclResult StubHcclNetDevOpenForEp(const HcclNetDevInfos* info, HcclNetDev* netDev)
{
    static hccl::NetDevContext kNetDevCtx;
    static bool initialized = false;
    if (!initialized) {
        hccl::HcclIpAddress localIp;
        (void)localIp.SetReadableAddress("10.10.10.50");
        kNetDevCtx.Init(NicType::DEVICE_NIC_TYPE, 0, 0, localIp);
        initialized = true;
    }
    *netDev = reinterpret_cast<HcclNetDev>(&kNetDevCtx);
    return HCCL_SUCCESS;
}

HcclResult StubHcclNetDevCloseForEp(HcclNetDev /*netDev*/) { return HCCL_SUCCESS; }

class AicpuTsRoceEndpointTest : public testing::Test {
protected:
    void TearDown() override
    {
        GlobalMockObject::verify();
        auto& m = AicpuTsRoceServerSocketContext::GetServerSocketMap();
        m.clear();
        auto& nd = AicpuTsRoceEndpoint::GetNetDevMap();
        nd.clear();
    }
};

TEST_F(AicpuTsRoceEndpointTest, Ut_Init_WhenLocIsHost_Returns_NOT_SUPPORT)
{
    EndpointDesc desc{};
    desc.protocol = COMM_PROTOCOL_ROCE;
    desc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;

    AicpuTsRoceEndpoint ep(desc);
    EXPECT_EQ(ep.Init(), HCCL_E_NOT_SUPPORT);
}

TEST_F(AicpuTsRoceEndpointTest, Ut_AddListenSocketWhiteList_WhenEmpty_Returns_PARA)
{
    AicpuTsRoceServerSocketContext ctx(nullptr, UINT32_MAX);
    const std::vector<SocketWlistInfo> empty{};
    EXPECT_EQ(ctx.AddListenSocketWhiteList(16666, empty), HCCL_E_PARA);
}

TEST_F(AicpuTsRoceEndpointTest, Ut_AddListenSocketWhiteList_WhenNoListenSocket_Returns_NOT_FOUND)
{
    AicpuTsRoceServerSocketContext ctx(nullptr, 0U);
    SocketWlistInfo entry{};
    entry.connLimit = 1U;
    const std::vector<SocketWlistInfo> one{entry};

    constexpr uint32_t kUnusedListenPort = 49151U;
    EXPECT_EQ(ctx.AddListenSocketWhiteList(kUnusedListenPort, one), HCCL_E_NOT_FOUND);
}

TEST_F(AicpuTsRoceEndpointTest, Ut_AcceptDataSocket_WhenNoListenSocket_Returns_NOT_FOUND)
{
    AicpuTsRoceServerSocketContext ctx(nullptr, 0U);
    std::shared_ptr<hccl::HcclSocket> out;
    constexpr uint32_t kUnusedListenPort = 49152U;
    EXPECT_EQ(ctx.AcceptDataSocket(kUnusedListenPort, "tag", out, 0), HCCL_E_NOT_FOUND);
    EXPECT_EQ(out.get(), nullptr);
}

TEST_F(AicpuTsRoceEndpointTest, Ut_UnregisterMemory_WhenMemHandleNull_Returns_PTR)
{
    EndpointDesc desc{};
    desc.protocol = COMM_PROTOCOL_ROCE;
    desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.10.10.10", &desc.commAddr.addr), 1);

    AicpuTsRoceEndpoint ep(desc);
    ep.regedMemMgr_ = std::make_shared<AicpuTsRoceRegedMemMgr>(nullptr, nullptr);
    EXPECT_EQ(ep.GetRegedMemMgr()->UnregisterMemory(nullptr), HCCL_E_PTR);
}

TEST_F(AicpuTsRoceEndpointTest, Ut_RegisterMemory_WhenMemHandleOutNull_Returns_PTR)
{
    EndpointDesc desc{};
    desc.protocol = COMM_PROTOCOL_ROCE;
    desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.10.10.11", &desc.commAddr.addr), 1);

    AicpuTsRoceEndpoint ep(desc);
    ep.regedMemMgr_ = std::make_shared<AicpuTsRoceRegedMemMgr>(nullptr, nullptr);
    HcommMem mem{};
    mem.addr = reinterpret_cast<void*>(0x1000U);
    mem.size = 4096U;
    mem.type = COMM_MEM_TYPE_DEVICE;
    EXPECT_EQ(ep.GetRegedMemMgr()->RegisterMemory(&mem, "t", nullptr), HCCL_E_PTR);
}

TEST_F(AicpuTsRoceEndpointTest, Ut_RegisterMemory_Delegates_Returns_SUCCESS)
{
    EndpointDesc desc{};
    desc.protocol = COMM_PROTOCOL_ROCE;
    desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.10.10.20", &desc.commAddr.addr), 1);

    AicpuTsRoceEndpoint ep(desc);
    ep.regedMemMgr_ = std::make_shared<FakeRegedMemMgrForEndpointUt>();
    HcommMem mem{};
    mem.addr = reinterpret_cast<void*>(0x2000U);
    mem.size = 2048U;
    mem.type = COMM_MEM_TYPE_DEVICE;
    void* handle = nullptr;
    ASSERT_EQ(ep.GetRegedMemMgr()->RegisterMemory(&mem, "tag", &handle), HCCL_SUCCESS);
    EXPECT_EQ(handle, reinterpret_cast<void*>(0x42ULL));
}

TEST_F(AicpuTsRoceEndpointTest, Ut_MemoryExport_Delegates_Returns_SUCCESS)
{
    EndpointDesc desc{};
    desc.protocol = COMM_PROTOCOL_ROCE;
    desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.10.10.21", &desc.commAddr.addr), 1);

    AicpuTsRoceEndpoint ep(desc);
    ep.regedMemMgr_ = std::make_shared<FakeRegedMemMgrForEndpointUt>();
    void* memDesc = nullptr;
    uint32_t len = 0U;
    ASSERT_EQ(
        ep.GetRegedMemMgr()->MemoryExport(ep.GetEndpointDesc(), reinterpret_cast<void*>(0x1), &memDesc, &len),
        HCCL_SUCCESS);
    EXPECT_NE(memDesc, nullptr);
    EXPECT_EQ(len, 4U);
}

TEST_F(AicpuTsRoceEndpointTest, Ut_GetAllMemHandles_Delegates_Returns_SUCCESS)
{
    EndpointDesc desc{};
    desc.protocol = COMM_PROTOCOL_ROCE;
    desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.10.10.22", &desc.commAddr.addr), 1);

    AicpuTsRoceEndpoint ep(desc);
    ep.regedMemMgr_ = std::make_shared<FakeRegedMemMgrForEndpointUt>();
    void* handles = reinterpret_cast<void*>(0xdeadbeefULL);
    uint32_t n = 99U;
    ASSERT_EQ(ep.GetRegedMemMgr()->GetAllMemHandles(&handles, &n), HCCL_SUCCESS);
    EXPECT_EQ(n, 0U);
    EXPECT_EQ(handles, nullptr);
}

TEST_F(AicpuTsRoceEndpointTest, Ut_CreateEndpoint_DeviceRoce_Returns_SUCCESS_WithInstance)
{
    EndpointDesc desc{};
    desc.protocol = COMM_PROTOCOL_ROCE;
    desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.10.10.30", &desc.commAddr.addr), 1);
    std::unique_ptr<Endpoint> ep;
    ASSERT_EQ(Endpoint::CreateEndpoint(desc, ep), HCCL_SUCCESS);
    ASSERT_NE(ep.get(), nullptr);
}

TEST_F(AicpuTsRoceEndpointTest, Ut_GetNetDev_WhenNotOpened_Returns_Null)
{
    EndpointDesc desc{};
    desc.protocol = COMM_PROTOCOL_ROCE;
    desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.10.10.31", &desc.commAddr.addr), 1);
    AicpuTsRoceEndpoint ep(desc);
    EXPECT_EQ(ep.GetNetDev(), nullptr);
}

TEST_F(AicpuTsRoceEndpointTest, Ut_MemoryImport_Delegates_Returns_SUCCESS)
{
    EndpointDesc desc{};
    desc.protocol = COMM_PROTOCOL_ROCE;
    desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.10.10.32", &desc.commAddr.addr), 1);
    AicpuTsRoceEndpoint ep(desc);
    ep.regedMemMgr_ = std::make_shared<FakeRegedMemMgrForEndpointUt>();
    HcommMem out{};
    ASSERT_EQ(ep.GetRegedMemMgr()->MemoryImport(nullptr, 0U, &out), HCCL_SUCCESS);
}

TEST_F(AicpuTsRoceEndpointTest, Ut_MemoryUnimport_Delegates_Returns_SUCCESS)
{
    EndpointDesc desc{};
    desc.protocol = COMM_PROTOCOL_ROCE;
    desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.10.10.33", &desc.commAddr.addr), 1);
    AicpuTsRoceEndpoint ep(desc);
    ep.regedMemMgr_ = std::make_shared<FakeRegedMemMgrForEndpointUt>();
    ASSERT_EQ(ep.GetRegedMemMgr()->MemoryUnimport(nullptr, 0U), HCCL_SUCCESS);
}

TEST_F(AicpuTsRoceEndpointTest, Ut_UnregisterMemory_Delegates_Returns_SUCCESS)
{
    EndpointDesc desc{};
    desc.protocol = COMM_PROTOCOL_ROCE;
    desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.10.10.34", &desc.commAddr.addr), 1);
    AicpuTsRoceEndpoint ep(desc);
    ep.regedMemMgr_ = std::make_shared<FakeRegedMemMgrForEndpointUt>();
    ASSERT_EQ(ep.GetRegedMemMgr()->UnregisterMemory(reinterpret_cast<void*>(0x99ULL)), HCCL_SUCCESS);
}

TEST_F(AicpuTsRoceEndpointTest, Ut_TwoInitsSameDevicePhyId_ShareOneNetDevSlot)
{
    MOCKER(hrtGetDevice).stubs().with(mockcpp::any()).will(invoke(StubHrtGetDeviceWriteZeroEp));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(mockcpp::any(), outBound(0U)).will(returnValue(HCCL_SUCCESS));
    MOCKER(HcclNetDevOpen).stubs().will(invoke(StubHcclNetDevOpenForEp));
    MOCKER(HcclNetDevClose).stubs().will(invoke(StubHcclNetDevCloseForEp));
    MOCKER_CPP(&hccl::NetworkManager::CreateRdmaHandle).stubs().will(invoke(StubCreateRdmaHandleEp));
    MOCKER_CPP(&hccl::NetworkManager::GetRdmaHandleByIpAddr).stubs().will(invoke(StubGetRdmaHandleByIpAddrEp));
    const SocketMapKey key{0U, 16666U};
    auto& sockMap = AicpuTsRoceServerSocketContext::GetServerSocketMap();
    sockMap[key]
        = AicpuTsListenSocketSlot{std::make_shared<hccl::HcclSocket>(static_cast<HcclNetDevCtx>(nullptr), 16666U), 1U};

    EndpointDesc desc{};
    desc.protocol = COMM_PROTOCOL_ROCE;
    desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.10.10.50", &desc.commAddr.addr), 1);

    {
        AicpuTsRoceEndpoint ep1(desc);
        ASSERT_EQ(ep1.Init(), HCCL_SUCCESS);
        AicpuTsRoceEndpoint ep2(desc);
        ASSERT_EQ(ep2.Init(), HCCL_SUCCESS);
        auto& nd = AicpuTsRoceEndpoint::GetNetDevMap();
        ASSERT_EQ(nd.size(), 1U);
        ASSERT_EQ(nd.at(0U).refCount, 2U);
        ASSERT_EQ(ep1.GetNetDev(), ep2.GetNetDev());
    }
    ASSERT_TRUE(AicpuTsRoceEndpoint::GetNetDevMap().empty());
}

TEST_F(AicpuTsRoceEndpointTest, Ut_Init_DeviceRoce_WhenDepsMocked_Returns_SUCCESS)
{
    MOCKER(hrtGetDevice).stubs().with(mockcpp::any()).will(invoke(StubHrtGetDeviceWriteZeroEp));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(mockcpp::any(), outBound(0U)).will(returnValue(HCCL_SUCCESS));
    MOCKER(HcclNetDevOpen).stubs().will(invoke(StubHcclNetDevOpenForEp));
    MOCKER(HcclNetDevClose).stubs().will(invoke(StubHcclNetDevCloseForEp));
    MOCKER_CPP(&hccl::NetworkManager::CreateRdmaHandle).stubs().will(invoke(StubCreateRdmaHandleEp));
    MOCKER_CPP(&hccl::NetworkManager::GetRdmaHandleByIpAddr).stubs().will(invoke(StubGetRdmaHandleByIpAddrEp));
    auto& sockMap = AicpuTsRoceServerSocketContext::GetServerSocketMap();
    const SocketMapKey key{0U, 16666U};
    sockMap[key]
        = AicpuTsListenSocketSlot{std::make_shared<hccl::HcclSocket>(static_cast<HcclNetDevCtx>(nullptr), 16666U), 1U};

    EndpointDesc desc{};
    desc.protocol = COMM_PROTOCOL_ROCE;
    desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.10.10.50", &desc.commAddr.addr), 1);

    AicpuTsRoceEndpoint ep(desc);
    ASSERT_EQ(ep.Init(), HCCL_SUCCESS);
}

TEST_F(AicpuTsRoceEndpointTest, Ut_ServerSocketListen_WhenReuse_Returns_SUCCESS)
{
    EndpointDesc desc{};
    desc.protocol = COMM_PROTOCOL_ROCE;
    desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.10.10.41", &desc.commAddr.addr), 1);
    AicpuTsRoceEndpoint ep(desc);
    ep.netDevRefPhyId_ = 0U;
    ep.serverSocketContext_.emplace(ep.netDev_, ep.netDevRefPhyId_);
    auto& sockMap = AicpuTsRoceServerSocketContext::GetServerSocketMap();
    const SocketMapKey key{0U, 16666U};
    sockMap[key]
        = AicpuTsListenSocketSlot{std::make_shared<hccl::HcclSocket>(static_cast<HcclNetDevCtx>(nullptr), 16666U), 1U};
    ASSERT_NE(ep.GetServerSocketContext(), nullptr);
    ASSERT_EQ(ep.GetServerSocketContext()->ServerSocketListen(16666U), HCCL_SUCCESS);
}

TEST_F(AicpuTsRoceEndpointTest, Ut_ServerSocketListen_WhenNoExistingSocket_InsertMapSuccess)
{
    MOCKER(hrtGetDevice).stubs().with(any()).will(invoke(StubHrtGetDeviceWriteZeroEp));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(any(), outBound(0U)).will(returnValue(HCCL_SUCCESS));
    MOCKER(HcclNetDevOpen).stubs().will(invoke(StubHcclNetDevOpenForEp));
    MOCKER(HcclNetDevClose).stubs().will(invoke(StubHcclNetDevCloseForEp));
    MOCKER_CPP(&hccl::NetworkManager::GetRdmaHandleByIpAddr).stubs().will(invoke(StubGetRdmaHandleByIpAddrEp));
    MOCKER_CPP(&hccl::HcclSocket::Listen, HcclResult(hccl::HcclSocket::*)()).stubs().will(returnValue(HCCL_SUCCESS));

    EndpointDesc desc{};
    desc.protocol = COMM_PROTOCOL_ROCE;
    desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    AicpuTsRoceEndpoint ep(desc);
    ASSERT_EQ(ep.Init(), HCCL_SUCCESS);

    ASSERT_NE(ep.GetServerSocketContext(), nullptr);
    ASSERT_EQ(ep.GetServerSocketContext()->ServerSocketListen(16666U), HCCL_SUCCESS);

    auto& sockMap = AicpuTsRoceServerSocketContext::GetServerSocketMap();
    const SocketMapKey key{0U, 16666U};
    ASSERT_EQ(sockMap.size(), 1U);
    ASSERT_NE(sockMap.at(key).socket, nullptr);
    ASSERT_EQ(sockMap.at(key).refCount, 1U);
    ASSERT_TRUE(ep.serverSocketContext_->hasListenSocketRef_);
    ASSERT_EQ(ep.serverSocketContext_->listenRefKeys_.size(), 1U);
}

TEST_F(AicpuTsRoceEndpointTest, Ut_AddListenSocketWhiteList_WhenListenPresent_Returns_SUCCESS)
{
    MOCKER_CPP(&hccl::HcclSocket::AddWhiteList).stubs().will(returnValue(HCCL_SUCCESS));
    AicpuTsRoceServerSocketContext ctx(nullptr, 0U);
    auto& sockMap = AicpuTsRoceServerSocketContext::GetServerSocketMap();
    const SocketMapKey key{0U, 17777U};
    sockMap[key]
        = AicpuTsListenSocketSlot{std::make_shared<hccl::HcclSocket>(static_cast<HcclNetDevCtx>(nullptr), 17777U), 1U};
    SocketWlistInfo entry{};
    entry.connLimit = 1U;
    const std::vector<SocketWlistInfo> one{entry};
    ASSERT_EQ(ctx.AddListenSocketWhiteList(17777U, one), HCCL_SUCCESS);
}

TEST_F(AicpuTsRoceEndpointTest, Ut_AcceptDataSocket_WhenListenPresent_Returns_SUCCESS)
{
    MOCKER_CPP(&hccl::HcclSocket::Accept).stubs().will(invoke(StubHcclSocketAcceptForEp));
    AicpuTsRoceServerSocketContext ctx(nullptr, 0U);
    auto& sockMap = AicpuTsRoceServerSocketContext::GetServerSocketMap();
    const SocketMapKey key{0U, 18888U};
    sockMap[key]
        = AicpuTsListenSocketSlot{std::make_shared<hccl::HcclSocket>(static_cast<HcclNetDevCtx>(nullptr), 18888U), 1U};
    std::shared_ptr<hccl::HcclSocket> out;
    ASSERT_EQ(ctx.AcceptDataSocket(18888U, "ut_tag", out, 0U), HCCL_SUCCESS);
    ASSERT_NE(out.get(), nullptr);
}

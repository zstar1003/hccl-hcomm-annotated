/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdio.h>
#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>

#include "hccl/base.h"
#include "hccl/hccl_types.h"
#include "hccl_network.h"
#include "sal.h"
#include "network_manager_pub.h"
#include "externalinput_pub.h"
#include "tsd/tsd_client.h"
#include "dltdt_function.h"
#include "dlra_function.h"

#define private public
#define protected public
#include "hccl_socket.h"
#include "hccl_socket_manager.h"
#undef private
#undef protected

using namespace std;
using namespace hccl;

s32 stub_SocketManagerTest_hrtRaSocketNonBlockSendHB(
    const FdHandle fdHandle, const void* data, u64 size, u64* sent_size)
{
    *sent_size = size;
    return 0;
}

s32 stub_SocketManagerTest_hrtRaSocketNonBlockRecvHB(const FdHandle fdHandle, void* data, u64 size, u64* recvSize)
{
    static u32 count = 0;
    if (count++ % 5 != 0) {
        *recvSize = size;
        count = 0;
    }
    return 0;
}

s32 stub_SocketManagerTest_hrtRaGetSockets(u32 role, struct SocketInfoT conn[], u32 num, u32* connectedNum)
{
    static std::vector<int> fdHandle;
    for (int i = 0; i < num; i++) {
        fdHandle.push_back(0);
        conn[i].fdHandle = 0;
        conn[i].status = CONNECT_OK;
    }
    *connectedNum = num;
    return 0;
}

HcclResult stub_SocketManagerTest_GetIsSupSockBatchCloseImmed(u32 phyId, bool& isSupportBatchClose)
{
    isSupportBatchClose = true;
    return HCCL_SUCCESS;
}

HcclResult stub_SocketManagerTest_HcclNetDevGetTlsStatus(HcclNetDevCtx netDevCtx, TlsStatus* tlsStatus)
{
    *tlsStatus = TlsStatus::ENABLE;
    return HCCL_SUCCESS;
}

class SocketManagerTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        DlTdtFunction::GetInstance().DlTdtFunctionInit();
        DlRaFunction::GetInstance().DlRaFunctionInit();
        std::cout << "\033[36m--ExchangerSocket SetUP--\033[0m" << std::endl;
    }
    static void TearDownTestCase() { std::cout << "\033[36m--ExchangerSocket TearDown--\033[0m" << std::endl; }
    virtual void SetUp()
    {
        MOCKER(GetIsSupSockBatchCloseImmed).stubs().will(invoke(stub_SocketManagerTest_GetIsSupSockBatchCloseImmed));

        MOCKER(hrtRaSocketWhiteListAdd).stubs().will(returnValue(HCCL_SUCCESS));

        MOCKER(hrtRaSocketWhiteListDel).stubs().will(returnValue(HCCL_SUCCESS));

        MOCKER(hrtRaSocketBatchConnect).stubs().will(returnValue(HCCL_SUCCESS));

        MOCKER(hrtRaGetSockets).stubs().will(invoke(stub_SocketManagerTest_hrtRaGetSockets));

        MOCKER(hrtRaSocketBatchClose).stubs().will(returnValue(HCCL_SUCCESS));

        MOCKER(hrtRaSocketNonBlockSend).stubs().will(invoke(stub_SocketManagerTest_hrtRaSocketNonBlockSendHB));

        MOCKER(hrtRaSocketNonBlockRecv).stubs().will(invoke(stub_SocketManagerTest_hrtRaSocketNonBlockRecvHB));

        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};

TEST_F(SocketManagerTest, ut_SocketManagerTest_LocalServer)
{
    TlsStatus tlsStatus = TlsStatus::ENABLE;
    MOCKER(HcclNetDevGetTlsStatus).stubs().will(invoke(stub_SocketManagerTest_HcclNetDevGetTlsStatus));
    u32 ifnumVersion = 3;
    MOCKER(hrtRaGetInterfaceVersion)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&ifnumVersion))
        .will(returnValue(HCCL_SUCCESS));
    HcclResult ret;
    std::shared_ptr<HcclSocketManager> socketManager = nullptr;
    socketManager.reset(new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));

    std::string commTag = "SocketManagerTest";
    bool isInterLink = false;
    u32 socketsPerLink = 1;
    NicType socketType = NicType::VNIC_TYPE;
    HcclSocketRole localRole = HcclSocketRole::SOCKET_ROLE_SERVER;

    HcclIpAddress localIPs(0x01);

    ret = HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclNetDevCtx portCtx;
    ret = HcclNetOpenDev(&portCtx, NicType::VNIC_TYPE, 0, 0, localIPs);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = socketManager->ServerInit(portCtx, 16666);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::map<u32, HcclRankLinkInfo> remoteInfos;
    HcclRankLinkInfo linkInfo1{};
    u32 dstRank1 = 1;
    u32 devicePhyId1 = 1;
    linkInfo1.userRank = dstRank1;
    linkInfo1.devicePhyId = devicePhyId1;
    HcclIpAddress ipAddress1(devicePhyId1);
    linkInfo1.ip = ipAddress1;
    linkInfo1.socketsPerLink = socketsPerLink;
    remoteInfos.insert(std::make_pair(linkInfo1.userRank, linkInfo1));

    HcclRankLinkInfo linkInfo2{};
    u32 dstRank2 = 2;
    u32 devicePhyId2 = 2;
    linkInfo2.userRank = dstRank2;
    linkInfo2.devicePhyId = devicePhyId2;
    HcclIpAddress ipAddress2(devicePhyId2);
    linkInfo2.ip = ipAddress2;
    linkInfo2.socketsPerLink = socketsPerLink;

    remoteInfos.insert(std::make_pair(linkInfo2.userRank, linkInfo2));

    std::map<u32, std::vector<std::shared_ptr<HcclSocket>>> socketsMap;
    bool isSupportReuse = false;

    std::map<u32, u32> dstRankToUserRank;
    ret = socketManager->CreateSockets(
        commTag, isInterLink, portCtx, socketType, localRole, localIPs, remoteInfos, socketsMap, dstRankToUserRank,
        isSupportReuse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    socketManager->PrintErrorConnection(localRole, socketsMap, dstRankToUserRank, tlsStatus);

    for (auto& item : socketsMap) {
        auto& sockets = item.second;
        std::shared_ptr<HcclSocket> tempSocket = sockets[0];
        u8 buff[8] = {};
        u64 compSize = 0;
        tempSocket->fdHandle_ = (void*)0x01;

        ret = tempSocket->ISend((void*)buff, sizeof(buff), compSize);
        EXPECT_EQ(ret, HCCL_SUCCESS);
        ret = tempSocket->ISend((void*)buff, sizeof(buff), compSize);
        EXPECT_EQ(ret, HCCL_SUCCESS);

        tempSocket->fdHandle_ = (void*)0x00;
    }

    socketManager->DestroySockets(commTag, dstRank1);

    socketManager->DestroySockets(commTag);

    socketManager->ServerDeInit(portCtx, 16666);

    HcclNetCloseDev(portCtx);
    HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0);
}

TEST_F(SocketManagerTest, ut_SocketManagerTest_LocalClient)
{
    TlsStatus tlsStatus = TlsStatus::ENABLE;
    u32 ifnumVersion = 3;
    MOCKER(hrtRaGetInterfaceVersion)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&ifnumVersion))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(HcclNetDevGetTlsStatus).stubs().will(invoke(stub_SocketManagerTest_HcclNetDevGetTlsStatus));
    HcclResult ret;
    std::shared_ptr<HcclSocketManager> socketManager = nullptr;
    socketManager.reset(new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));

    std::string commTag = "SocketManagerTest";
    bool isInterLink = false;
    u32 socketsPerLink = 1;
    NicType socketType = NicType::VNIC_TYPE;
    HcclSocketRole localRole = HcclSocketRole::SOCKET_ROLE_CLIENT;

    HcclIpAddress localIPs(0x01);

    ret = HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclNetDevCtx portCtx;
    ret = HcclNetOpenDev(&portCtx, NicType::VNIC_TYPE, 0, 0, localIPs);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = socketManager->ServerInit(portCtx, 16666);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::map<u32, HcclRankLinkInfo> remoteInfos;
    HcclRankLinkInfo linkInfo1{};
    u32 dstRank1 = 1;
    u32 devicePhyId1 = 1;
    linkInfo1.userRank = dstRank1;
    linkInfo1.devicePhyId = devicePhyId1;
    HcclIpAddress ipAddress1(devicePhyId1);
    linkInfo1.ip = ipAddress1;
    linkInfo1.socketsPerLink = socketsPerLink;
    remoteInfos.insert(std::make_pair(linkInfo1.userRank, linkInfo1));

    HcclRankLinkInfo linkInfo2{};
    u32 dstRank2 = 2;
    u32 devicePhyId2 = 2;
    linkInfo2.userRank = dstRank2;
    linkInfo2.devicePhyId = devicePhyId2;
    HcclIpAddress ipAddress2(devicePhyId2);
    linkInfo2.ip = ipAddress2;
    linkInfo2.socketsPerLink = socketsPerLink;

    remoteInfos.insert(std::make_pair(linkInfo2.userRank, linkInfo2));

    std::map<u32, std::vector<std::shared_ptr<HcclSocket>>> socketsMap;
    bool isSupportReuse = false;

    std::map<u32, u32> dstRankToUserRank;
    ret = socketManager->CreateSockets(
        commTag, isInterLink, portCtx, socketType, localRole, localIPs, remoteInfos, socketsMap, dstRankToUserRank,
        isSupportReuse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    socketManager->PrintErrorConnection(localRole, socketsMap, dstRankToUserRank, tlsStatus);

    for (auto& item : socketsMap) {
        auto& sockets = item.second;
        std::shared_ptr<HcclSocket> tempSocket = sockets[0];
        u8 buff[8] = {};
        u64 compSize = 0;
        tempSocket->fdHandle_ = (void*)0x01;
        ret = tempSocket->ISend((void*)buff, sizeof(buff), compSize);
        EXPECT_EQ(ret, HCCL_SUCCESS);
        ret = tempSocket->ISend((void*)buff, sizeof(buff), compSize);
        EXPECT_EQ(ret, HCCL_SUCCESS);
        tempSocket->fdHandle_ = (void*)0x00;
    }

    socketManager->DestroySockets(commTag, dstRank1);

    socketManager->DestroySockets(commTag);

    socketManager->ServerDeInit(portCtx, 16666);

    HcclNetCloseDev(portCtx);
    HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0);
}

TEST_F(SocketManagerTest, ut_SocketManagerTest_GetStatus_SOCKET_ERROR)
{
    TlsStatus tlsStatus = TlsStatus::ENABLE;
    MOCKER(HcclNetDevGetTlsStatus).stubs().will(invoke(stub_SocketManagerTest_HcclNetDevGetTlsStatus));
    HcclResult ret;
    std::shared_ptr<HcclSocketManager> socketManager = nullptr;
    socketManager.reset(new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));

    std::string commTag = "SocketManagerTest";
    bool isInterLink = false;
    u32 socketsPerLink = 1;
    NicType socketType = NicType::VNIC_TYPE;
    HcclSocketRole localRole = HcclSocketRole::SOCKET_ROLE_CLIENT;

    HcclIpAddress localIPs(0x01);

    ret = HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclNetDevCtx portCtx;
    ret = HcclNetOpenDev(&portCtx, NicType::VNIC_TYPE, 0, 0, localIPs);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = socketManager->ServerInit(portCtx, 16666);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::map<u32, HcclRankLinkInfo> remoteInfos;
    HcclRankLinkInfo linkInfo1{};
    u32 dstRank1 = 1;
    u32 devicePhyId1 = 1;
    linkInfo1.userRank = dstRank1;
    linkInfo1.devicePhyId = devicePhyId1;
    HcclIpAddress ipAddress1(devicePhyId1);
    linkInfo1.ip = ipAddress1;
    linkInfo1.socketsPerLink = socketsPerLink;
    remoteInfos.insert(std::make_pair(linkInfo1.userRank, linkInfo1));

    HcclRankLinkInfo linkInfo2{};
    u32 dstRank2 = 2;
    u32 devicePhyId2 = 2;
    linkInfo2.userRank = dstRank2;
    linkInfo2.devicePhyId = devicePhyId2;
    HcclIpAddress ipAddress2(devicePhyId2);
    linkInfo2.ip = ipAddress2;
    linkInfo2.socketsPerLink = socketsPerLink;
    remoteInfos.insert(std::make_pair(linkInfo2.userRank, linkInfo2));

    std::map<u32, std::vector<std::shared_ptr<HcclSocket>>> socketsMap;
    bool isSupportReuse = false;

    MOCKER_CPP(&HcclSocket::GetStatus).stubs().will(returnValue(HcclSocketStatus::SOCKET_ERROR));

    std::map<u32, u32> dstRankToUserRank;
    ret = socketManager->CreateSockets(
        commTag, isInterLink, portCtx, socketType, localRole, localIPs, remoteInfos, socketsMap, dstRankToUserRank,
        isSupportReuse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = socketManager->WaitLinksEstablishCompleted(localRole, socketsMap);
    EXPECT_EQ(ret, HCCL_E_TCP_CONNECT);
    GlobalMockObject::verify();

    MOCKER(GetExternalInputHcclLinkTimeOut).stubs().will(returnValue(1));

    MOCKER_CPP(&HcclSocket::GetStatus).stubs().will(returnValue(HcclSocketStatus::SOCKET_CONNECTING));

    ret = socketManager->WaitLinksEstablishCompleted(localRole, socketsMap);
    EXPECT_EQ(ret, HCCL_E_TIMEOUT);

    socketManager->PrintErrorConnection(localRole, socketsMap, dstRankToUserRank, tlsStatus);

    socketManager->ServerDeInit(portCtx, 16666);

    HcclNetCloseDev(portCtx);
    HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0);
}

TEST_F(SocketManagerTest, ut_NetDev_error)
{
    HcclResult ret;

    MOCKER_CPP(&NetDevContext::Init).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_INTERNAL));

    HcclNetDevCtx portCtx;
    ret = HcclNetOpenDev(&portCtx, NicType::VNIC_TYPE, 0, 0, HcclIpAddress(0));
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    GlobalMockObject::verify();
}

HcclResult stub_IsSupportRaSocketAbort(bool& isSupportRaSocketAbort)
{
    isSupportRaSocketAbort = true;
    return HCCL_SUCCESS;
}

TEST_F(SocketManagerTest, ut_Socket_Abort)
{
    HcclResult ret;

    MOCKER(IsSupportRaSocketAbort).stubs().with(mockcpp::any()).will(invoke(stub_IsSupportRaSocketAbort));
    MOCKER(hrtRaSocketNonBlockBatchAbort).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    // device Nic
    std::string commTag = "SocketManagerTest";
    HcclIpAddress localIp = HcclIpAddress("192.168.100.110");
    HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, false);
    HcclNetDevCtx nicPortCtx;
    HcclNetOpenDev(&nicPortCtx, NicType::DEVICE_NIC_TYPE, 0, 0, localIp);

    std::shared_ptr<HcclSocket> tempSocket = nullptr;
    tempSocket.reset(new (std::nothrow) HcclSocket(nicPortCtx, 16666));
    tempSocket->localRole_ = HcclSocketRole::SOCKET_ROLE_CLIENT;
    tempSocket->SetStatus(HcclSocketStatus::SOCKET_TIMEOUT);
    u32 fakeScoketHandle = 1;
    tempSocket->nicSocketHandle_ = &fakeScoketHandle;
    tempSocket->remoteIp_ = localIp;

    tempSocket->Close();

    HcclNetCloseDev(nicPortCtx);
    HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0);
}

TEST_F(SocketManagerTest, ut_socket_listen_with_port)
{
    HcclResult ret;

    // device Nic
    std::string commTag = "SocketManagerTest";
    HcclIpAddress localIp = HcclIpAddress("192.168.100.110");
    HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, false);
    HcclNetDevCtx nicPortCtx;
    HcclNetOpenDev(&nicPortCtx, NicType::DEVICE_NIC_TYPE, 0, 0, localIp);

    std::shared_ptr<HcclSocket> tempSocket = nullptr;
    tempSocket.reset(new (std::nothrow) HcclSocket(nicPortCtx, 16666));

    ret = tempSocket->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = tempSocket->Listen(16666);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclNetCloseDev(nicPortCtx);
    HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0);

    // Vnic
    HcclIpAddress vnicIp = HcclIpAddress("192.168.100.111");
    HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, false);
    HcclNetDevCtx vnicPortCtx;
    HcclNetOpenDev(&vnicPortCtx, NicType::VNIC_TYPE, 0, 0, vnicIp);

    std::shared_ptr<HcclSocket> tempSocket2 = nullptr;
    tempSocket2.reset(new (std::nothrow) HcclSocket(vnicPortCtx, 16668));

    ret = tempSocket2->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = tempSocket2->Listen(16668);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclNetCloseDev(vnicPortCtx);
    HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0);
}

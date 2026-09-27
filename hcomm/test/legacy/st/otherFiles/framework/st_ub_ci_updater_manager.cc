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
#include "ub_ci_updater.h"
#include "ub_ci_updater_manager.h"
#include "rma_conn_manager.h"
#include "dev_ub_connection.h"
#include "communicator_impl.h"
#include "local_ub_rma_buffer.h"
#include "local_cnt_notify.h"
#include "ub_local_notify.h"
#include "rdma_handle_manager.h"
#undef private
#include "socket.h"
#include "runtime_api_exception.h"
#include "invalid_params_exception.h"
using namespace Hccl;

class UbCiUpdaterManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AdapterRts tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "AdapterRts tests tear down." << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in AdapterRts SetUP" << std::endl;
        fakeSocket
            = new Socket(nullptr, localIp, listenPort, remoteIp, tag, SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        delete fakeSocket;
        std::cout << "A Test case in AdapterRts TearDown" << std::endl;
    }
    Socket* fakeSocket;
    IpAddress localIp;
    IpAddress remoteIp;
    u32 listenPort = 100;
    std::string tag = "test";
};

TEST(UbCiUpdaterManagerTest, UbCiUpdaterManagerSaveConnsCi_ok)
{
    // Given
    // Given: socket time out
    Socket* fakeSocket;
    IpAddress localIp;
    IpAddress remoteIp;
    u32 listenPort = 100;
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::TIMEOUT));
    MOCKER(GetUbToken).stubs().will(returnValue(1));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    RdmaHandleManager::GetInstance().tokenInfoMap[rdmaHandle] = make_unique<TokenInfoManager>(0, rdmaHandle);
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    string opTag = "test";

    // When
    auto devUbConn
        = make_unique<DevUbConnection>(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);
    devUbConn->piVal = 100;
    CommunicatorImpl comm;
    RmaConnManager rmaConnManager(comm);

    rmaConnManager.rmaConnectionMap[opTag][linkData] = std::move(devUbConn);
    UbCiUpdaterManager ubCiUpdaterManager(&rmaConnManager);
    ubCiUpdaterManager.SaveConnsCi(opTag);
    std::vector<DevUbConnection::UbCiUpdater*> ubCiUpdaters = ubCiUpdaterManager.Get(opTag);

    // then
    EXPECT_EQ(ubCiUpdaters.size(), 1);
    EXPECT_EQ(ubCiUpdaters[0]->ciVal, 100);
}

TEST(UbCiUpdaterManagerTest, UbCiUpdaterManagerUpdateConnsCi_ok)
{
    // Given
    // Given: socket time out
    Socket* fakeSocket;
    IpAddress localIp;
    IpAddress remoteIp;
    u32 listenPort = 100;
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::TIMEOUT));
    MOCKER(GetUbToken).stubs().will(returnValue(1));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    string opTag = "test";
    MOCKER(RaUbUpdateCi).stubs();

    // When
    auto devUbConn
        = make_unique<DevUbConnection>(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);
    devUbConn->piVal = 100;
    CommunicatorImpl comm;
    RmaConnManager rmaConnManager(comm);
    rmaConnManager.rmaConnectionMap[opTag][linkData] = std::move(devUbConn);
    UbCiUpdaterManager ubCiUpdaterManager(&rmaConnManager);
    ubCiUpdaterManager.SaveConnsCi(opTag);
    ubCiUpdaterManager.UpdateConnsCi(opTag);

    // then
    EXPECT_EQ(ubCiUpdaterManager.ubCiUpdaters[opTag][0]->devUbConnPtr->GetCiVal(), 100);
}

TEST(UbCiUpdaterManagerTest, should_no_throw_when_calling_BatchCreate)
{
    // Given
    RdmaHandle rdmaHandle = (void*)0x1000000;
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;
    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    LinkData linkData2(portType, 0, 2, 0, 1);
    string opTag = "test";

    // When
    MOCKER(GetUbToken).stubs().will(returnValue(1));
    auto devUbConn
        = make_unique<DevUbConnection>(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);
    CommunicatorImpl comm;
    comm.remoteRmaBufManager = std::make_unique<RemoteRmaBufManager>(comm);
    RmaConnManager rmaConnManager(comm);
    rmaConnManager.rmaConnectionMap[opTag][linkData] = std::move(devUbConn);
    auto devUbConn2
        = make_unique<DevUbConnection>(rdmaHandle, linkData2.GetLocalAddr(), linkData2.GetRemoteAddr(), OpMode::OPBASE);
    rmaConnManager.rmaConnectionMap[opTag][linkData2] = std::move(devUbConn2);

    auto newUbConn
        = make_unique<DevUbConnection>(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);
    auto newUbConn2
        = make_unique<DevUbConnection>(rdmaHandle, linkData2.GetLocalAddr(), linkData2.GetRemoteAddr(), OpMode::OPBASE);

    MOCKER_CPP_VIRTUAL(*newUbConn, &DevUbConnection::GetStatus)
        .stubs()
        .will(returnValue((RmaConnStatus)RmaConnStatus::READY));
    MOCKER_CPP_VIRTUAL(*newUbConn2, &DevUbConnection::GetStatus)
        .stubs()
        .will(returnValue((RmaConnStatus)RmaConnStatus::READY));
    MOCKER_CPP(&RmaConnManager::Create)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(static_cast<RmaConnection*>(newUbConn.get())))
        .then(returnValue(static_cast<RmaConnection*>(newUbConn2.get())));

    // Then
    std::vector<LinkData> links = {linkData, linkData2};
    EXPECT_NO_THROW(rmaConnManager.BatchCreate(links));
}

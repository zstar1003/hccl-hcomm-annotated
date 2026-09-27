/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include "host/host_rdma_connection.h"
#include "socket.h"
#include "orion_adapter_rts.h"
#include "orion_adapter_hccp.h"
#include "hccp.h"
#include "hccp_common.h"
#define private public

class HostRdmaConnectionTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HostRdmaConnectionTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "HostRdmaConnectionTest tests tear down." << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in HostRdmaConnectionTest SetUP" << std::endl;
        fakeSocket = new Hccl::Socket(
            nullptr, localIp, listenPort, remoteIp, tag, Hccl::SocketRole::SERVER, Hccl::NicType::HOST_NIC_TYPE);
        MOCKER(Hccl::HrtGetRaQpStatus).stubs().with(mockcpp::any()).will(returnValue(1));
        MOCKER(Hccl::HrtRaDestroyQpWithCq).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(HCCL_SUCCESS));
        MOCKER(RaCreateCompChannel).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(0));
        MOCKER(RaDestroyCompChannel).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(0));
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        delete fakeSocket;
        std::cout << "A Test case in HostRdmaConnectionTest TearDown" << std::endl;
    }
    Hccl::Socket* fakeSocket;
    Hccl::IpAddress localIp;
    Hccl::IpAddress remoteIp;
    u32 listenPort = 100;
    std::string tag = "test";
};

TEST_F(HostRdmaConnectionTest, Ut_When_Normal_Call_Expect_Status_Consisitent)
{
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    MOCKER(Hccl::HrtRaCreateQpWithCq)
        .stubs()
        .with(
            mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(),
            mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    std::cout << "start" << std::endl;
    // socket 打桩
    MOCKER_CPP(&Hccl::Socket::GetStatus).stubs().will(returnValue((Hccl::SocketStatus)Hccl::SocketStatus::OK));
    char targetChipVer[Hccl::CHIP_VERSION_MAX_LEN] = "Ascend910_9591";
    MOCKER(Hccl::HrtGetSocVer)
        .stubs()
        .with(outBoundP(&targetChipVer[0], sizeof(targetChipVer)), mockcpp::any())
        .will(returnValue(RT_ERROR_NONE));
    QpHandle fakeQpHandle = (void*)0x1000000;
    MOCKER(Hccl::HrtRaQpCreate, QpHandle(*)(RdmaHandle, int, int))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(fakeQpHandle));

    RdmaHandle rdmaHandle = (void*)0x1000000;
    std::string tag = "test";

    // construct HostRdmaConnection
    std::cout << "construct" << std::endl;
    hcomm::HostRdmaConnection hostRdmaConnection(fakeSocket, rdmaHandle);
    std::cout << hostRdmaConnection.rdmaConnStatus_.Describe() << std::endl;
    EXPECT_EQ(hcomm::HostRdmaConnection::RdmaConnStatus::CLOSED, hostRdmaConnection.rdmaConnStatus_);
    // init
    std::cout << "Init" << std::endl;
    HcclResult ret = hostRdmaConnection.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::cout << hostRdmaConnection.rdmaConnStatus_.Describe() << std::endl;
    EXPECT_EQ(hcomm::HostRdmaConnection::RdmaConnStatus::INIT, hostRdmaConnection.rdmaConnStatus_);
    // repeat init
    std::cout << "Repeat Init" << std::endl;
    ret = hostRdmaConnection.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::cout << hostRdmaConnection.rdmaConnStatus_.Describe() << std::endl;
    EXPECT_EQ(hcomm::HostRdmaConnection::RdmaConnStatus::INIT, hostRdmaConnection.rdmaConnStatus_);
    // create qp
    std::cout << "create qp" << std::endl;
    ret = hostRdmaConnection.CreateQp();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::cout << hostRdmaConnection.rdmaConnStatus_.Describe() << std::endl;
    EXPECT_EQ(hcomm::HostRdmaConnection::RdmaConnStatus::QP_CREATED, hostRdmaConnection.rdmaConnStatus_);
    // exchange & modify
    std::cout << "exchange" << std::endl;
    std::unique_ptr<Hccl::Serializable> locQpAttrserial;
    ret = hostRdmaConnection.GetExchangeDto(locQpAttrserial);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = hostRdmaConnection.ParseRmtExchangeDto(*locQpAttrserial);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::cout << "modify" << std::endl;
    hostRdmaConnection.ModifyQp();
    std::cout << hostRdmaConnection.rdmaConnStatus_.Describe() << std::endl;
    EXPECT_EQ(hcomm::HostRdmaConnection::RdmaConnStatus::QP_MODIFIED, hostRdmaConnection.rdmaConnStatus_);
    // destroy
    std::cout << "destroy" << std::endl;
    hostRdmaConnection.DestroyQp();
    std::cout << hostRdmaConnection.rdmaConnStatus_.Describe() << std::endl;
    EXPECT_EQ(hcomm::HostRdmaConnection::RdmaConnStatus::CLOSED, hostRdmaConnection.rdmaConnStatus_);
}

TEST_F(HostRdmaConnectionTest, Ut_When_DevType_NotExpected_Expect_ERROR)
{
    DevType devType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    MOCKER(Hccl::HrtRaCreateQpWithCq)
        .stubs()
        .with(
            mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(),
            mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    std::cout << "start" << std::endl;

    RdmaHandle rdmaHandle = (void*)0x1000000;
    std::string tag = "test";

    // construct HostRdmaConnection
    std::cout << "construct" << std::endl;
    hcomm::HostRdmaConnection hostRdmaConnection(fakeSocket, rdmaHandle);
    std::cout << hostRdmaConnection.rdmaConnStatus_.Describe() << std::endl;
    EXPECT_EQ(hcomm::HostRdmaConnection::RdmaConnStatus::CLOSED, hostRdmaConnection.rdmaConnStatus_);
    // init
    std::cout << "Init" << std::endl;
    HcclResult ret = hostRdmaConnection.Init();
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    std::cout << hostRdmaConnection.rdmaConnStatus_.Describe() << std::endl;
    EXPECT_EQ(hcomm::HostRdmaConnection::RdmaConnStatus::CLOSED, hostRdmaConnection.rdmaConnStatus_);
}

// SOCKET_TIME_OUT
TEST_F(HostRdmaConnectionTest, Ut_When_Socket_TIMEOUT_Expect_ERROR)
{
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    std::cout << "start" << std::endl;
    // socket 打桩
    MOCKER_CPP(&Hccl::Socket::GetStatus).stubs().will(returnValue((Hccl::SocketStatus)Hccl::SocketStatus::TIMEOUT));
    char targetChipVer[Hccl::CHIP_VERSION_MAX_LEN] = "Ascend910_9591";
    MOCKER(Hccl::HrtGetSocVer)
        .stubs()
        .with(outBoundP(&targetChipVer[0], sizeof(targetChipVer)), mockcpp::any())
        .will(returnValue(RT_ERROR_NONE));

    RdmaHandle rdmaHandle = (void*)0x1000000;
    std::string tag = "test";

    // construct HostRdmaConnection
    std::cout << "construct" << std::endl;
    hcomm::HostRdmaConnection hostRdmaConnection(fakeSocket, rdmaHandle);
    std::cout << hostRdmaConnection.rdmaConnStatus_.Describe() << std::endl;
    EXPECT_EQ(hcomm::HostRdmaConnection::RdmaConnStatus::CLOSED, hostRdmaConnection.rdmaConnStatus_);
    // init
    std::cout << "Init" << std::endl;
    HcclResult ret = hostRdmaConnection.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::cout << hostRdmaConnection.rdmaConnStatus_.Describe() << std::endl;
    EXPECT_EQ(hcomm::HostRdmaConnection::RdmaConnStatus::INIT, hostRdmaConnection.rdmaConnStatus_);
    // create qp
    std::cout << "create qp" << std::endl;
    ret = hostRdmaConnection.CreateQp();
    EXPECT_EQ(ret, HCCL_E_AGAIN);
    std::cout << hostRdmaConnection.rdmaConnStatus_.Describe() << std::endl;
    EXPECT_EQ(hcomm::HostRdmaConnection::RdmaConnStatus::INIT, hostRdmaConnection.rdmaConnStatus_);
}

// // Qp Create 失败
TEST_F(HostRdmaConnectionTest, Ut_When_Call_GetStatus_Expect_Return_Ready)
{
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    std::cout << "start" << std::endl;
    // socket 打桩
    MOCKER_CPP(&Hccl::Socket::GetStatus).stubs().will(returnValue((Hccl::SocketStatus)Hccl::SocketStatus::OK));
    char targetChipVer[Hccl::CHIP_VERSION_MAX_LEN] = "Ascend910_9591";
    MOCKER(Hccl::HrtGetSocVer)
        .stubs()
        .with(outBoundP(&targetChipVer[0], sizeof(targetChipVer)), mockcpp::any())
        .will(returnValue(RT_ERROR_NONE));
    MOCKER(Hccl::HrtRaCreateQpWithCq)
        .stubs()
        .with(
            mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(),
            mockcpp::any())
        .will(returnValue(HCCL_E_INTERNAL));

    RdmaHandle rdmaHandle = (void*)0x1000000;
    std::string tag = "test";

    // construct HostRdmaConnection
    std::cout << "construct" << std::endl;
    hcomm::HostRdmaConnection hostRdmaConnection(fakeSocket, rdmaHandle);
    std::cout << hostRdmaConnection.rdmaConnStatus_.Describe() << std::endl;
    EXPECT_EQ(hcomm::HostRdmaConnection::RdmaConnStatus::CLOSED, hostRdmaConnection.rdmaConnStatus_);
    // init
    std::cout << "Init" << std::endl;
    HcclResult ret = hostRdmaConnection.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::cout << hostRdmaConnection.rdmaConnStatus_.Describe() << std::endl;
    EXPECT_EQ(hcomm::HostRdmaConnection::RdmaConnStatus::INIT, hostRdmaConnection.rdmaConnStatus_);
    // create qp
    std::cout << "create qp" << std::endl;
    ret = hostRdmaConnection.CreateQp();
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    std::cout << hostRdmaConnection.rdmaConnStatus_.Describe() << std::endl;
    EXPECT_EQ(hcomm::HostRdmaConnection::RdmaConnStatus::INIT, hostRdmaConnection.rdmaConnStatus_);
}

TEST_F(HostRdmaConnectionTest, Ut_RaSetQpAttr_Qos_TimeOut_RetryCnt_Success)
{
    MOCKER(RaSetQpAttrQos).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(0));
    MOCKER(RaSetQpAttrTimeout).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(0));
    MOCKER(RaSetQpAttrRetryCnt).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(0));
    MOCKER(Hccl::HrtRaCreateQpWithCq)
        .stubs()
        .with(
            mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(),
            mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::Socket::GetStatus).stubs().will(returnValue((Hccl::SocketStatus)Hccl::SocketStatus::OK));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    hcomm::HostRdmaConnection conn(fakeSocket, rdmaHandle);
    conn.Init();
    conn.qpInfo_.trafficClass = 1;
    conn.qpInfo_.serviceLevel = 2;
    conn.qpInfo_.retryInterval = 10;
    conn.qpInfo_.retryCnt = 3;
    HcclResult ret = conn.CreateQp();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HostRdmaConnectionTest, Ut_RaSetQpAttrQos_Fail)
{
    MOCKER(RaSetQpAttrQos).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(-1));
    MOCKER(RaSetQpAttrTimeout).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(0));
    MOCKER(RaSetQpAttrRetryCnt).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(0));
    MOCKER(Hccl::HrtRaCreateQpWithCq)
        .stubs()
        .with(
            mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(),
            mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::Socket::GetStatus).stubs().will(returnValue((Hccl::SocketStatus)Hccl::SocketStatus::OK));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    hcomm::HostRdmaConnection conn(fakeSocket, rdmaHandle);
    conn.Init();
    conn.qpInfo_.trafficClass = 1;
    conn.qpInfo_.serviceLevel = 2;
    conn.qpInfo_.retryInterval = 10;
    conn.qpInfo_.retryCnt = 3;
    HcclResult ret = conn.CreateQp();
    EXPECT_EQ(ret, HCCL_E_NETWORK);
}

TEST_F(HostRdmaConnectionTest, Ut_RaSetQpAttrTimeout_Fail)
{
    MOCKER(RaSetQpAttrQos).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(0));
    MOCKER(RaSetQpAttrTimeout).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(-1));
    MOCKER(RaSetQpAttrRetryCnt).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(0));
    MOCKER(Hccl::HrtRaCreateQpWithCq)
        .stubs()
        .with(
            mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(),
            mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::Socket::GetStatus).stubs().will(returnValue((Hccl::SocketStatus)Hccl::SocketStatus::OK));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    hcomm::HostRdmaConnection conn(fakeSocket, rdmaHandle);
    conn.Init();
    conn.qpInfo_.trafficClass = 1;
    conn.qpInfo_.serviceLevel = 2;
    conn.qpInfo_.retryInterval = 10;
    conn.qpInfo_.retryCnt = 3;
    HcclResult ret = conn.CreateQp();
    EXPECT_EQ(ret, HCCL_E_NETWORK);
}

TEST_F(HostRdmaConnectionTest, Ut_RaSetQpAttrRetryCnt_Fail)
{
    MOCKER(RaSetQpAttrQos).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(0));
    MOCKER(RaSetQpAttrTimeout).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(0));
    MOCKER(RaSetQpAttrRetryCnt).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(-1));
    MOCKER(Hccl::HrtRaCreateQpWithCq)
        .stubs()
        .with(
            mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(),
            mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::Socket::GetStatus).stubs().will(returnValue((Hccl::SocketStatus)Hccl::SocketStatus::OK));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    hcomm::HostRdmaConnection conn(fakeSocket, rdmaHandle);
    conn.Init();
    conn.qpInfo_.trafficClass = 1;
    conn.qpInfo_.serviceLevel = 2;
    conn.qpInfo_.retryInterval = 10;
    conn.qpInfo_.retryCnt = 3;
    HcclResult ret = conn.CreateQp();
    EXPECT_EQ(ret, HCCL_E_NETWORK);
}

// 覆盖 host_rdma_connection.cc ModifyQp 中 RaSetQpLbValue 分支
// 前置依赖：rdmaConnStatus_==QP_CREATED、rmtQpAttr_ 有效、RaGetQpAttr/RaTypicalQpModify 成功
// 用例1: lbValue >= 0 且 RaSetQpLbValue 成功 -> ModifyQp 返回 HCCL_SUCCESS
TEST_F(HostRdmaConnectionTest, Ut_RaSetQpLbValue_Success_When_LbValue_NonNegative)
{
    MOCKER(RaGetQpAttr).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(0));
    MOCKER(RaTypicalQpModify).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(0));
    MOCKER(RaSetQpLbValue).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(0));

    RdmaHandle rdmaHandle = (void*)0x1000000;
    hcomm::HostRdmaConnection conn(fakeSocket, rdmaHandle);
    // 直接构造 QP_CREATED 状态，跳过 CreateQp
    conn.rdmaConnStatus_ = hcomm::HostRdmaConnection::RdmaConnStatus::QP_CREATED;
    // rmtQpAttr_ 必须有效：qpn/psn != UINT32_MAX
    conn.rmtQpAttr_.qpn = 100;
    conn.rmtQpAttr_.psn = 200;
    // qpInfo_.qp 必须非空，避免 L247 qp_num 解引用段错误
    struct ibv_qp fakeQp {};
    fakeQp.qp_num = 12345;
    conn.qpInfo_.qp = &fakeQp;
    conn.qpInfo_.qpHandle = (void*)0x2000000;
    conn.qpInfo_.lbValue = 5;

    HcclResult ret = conn.ModifyQp();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(conn.rdmaConnStatus_, hcomm::HostRdmaConnection::RdmaConnStatus::QP_MODIFIED);
}

// 用例2: lbValue >= 0 且 RaSetQpLbValue 失败 -> ModifyQp 返回 HCCL_E_NETWORK
TEST_F(HostRdmaConnectionTest, Ut_RaSetQpLbValue_Fail_When_LbValue_NonNegative)
{
    MOCKER(RaGetQpAttr).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(0));
    MOCKER(RaTypicalQpModify).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(0));
    MOCKER(RaSetQpLbValue).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(-1));

    RdmaHandle rdmaHandle = (void*)0x1000000;
    hcomm::HostRdmaConnection conn(fakeSocket, rdmaHandle);
    conn.rdmaConnStatus_ = hcomm::HostRdmaConnection::RdmaConnStatus::QP_CREATED;
    conn.rmtQpAttr_.qpn = 100;
    conn.rmtQpAttr_.psn = 200;
    struct ibv_qp fakeQp {};
    fakeQp.qp_num = 12345;
    conn.qpInfo_.qp = &fakeQp;
    conn.qpInfo_.qpHandle = (void*)0x2000000;
    conn.qpInfo_.lbValue = 5;

    HcclResult ret = conn.ModifyQp();
    EXPECT_EQ(ret, HCCL_E_NETWORK);
    // 失败路径不更新状态
    EXPECT_EQ(conn.rdmaConnStatus_, hcomm::HostRdmaConnection::RdmaConnStatus::QP_CREATED);
}

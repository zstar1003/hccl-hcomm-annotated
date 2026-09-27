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
#include "dev_ub_connection.h"
#include "rma_conn_manager.h"
#include "socket.h"
#include "orion_adapter_rts.h"
#include "not_support_exception.h"
#include "rma_conn_exception.h"
#include "rdma_handle_manager.h"
#include "tp_manager.h"
#include "hccp_async.h"
#include "env_config/env_config_v2.h"
#undef protected
#undef private
#include "hccp_async_ctx.h"

using namespace Hccl;

class DevUbConnectionTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DevUbConnection tests set up." << std::endl; }

    static void TearDownTestCase()
    {
        GlobalMockObject::verify();
        std::cout << "DevUbConnection tests tear down." << std::endl;
    }

    virtual void SetUp()
    {
        GlobalMockObject::verify();
        MOCKER_CPP(&TpManager::GetTpInfo)
            .stubs()
            .will(returnValue(HcclResult::HCCL_E_AGAIN))
            .then(returnValue(HcclResult::HCCL_SUCCESS));
        std::cout << "A Test case in DevUbConnection SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in DevUbConnection TearDown" << std::endl;
    }
    IpAddress localIp;
    IpAddress remoteIp;
    u32 listenPort = 100;
    std::string tag = "test";
};

TEST_F(DevUbConnectionTest, set_wq_and_sq_context_info_success)
{
    RdmaHandle rdmaHandle = reinterpret_cast<void*>(0x1000000);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);
    devUbConnection.jettyId = 100U;
    devUbConnection.dbAddr = 0x8000000U;
    devUbConnection.sqBuffVa = 0x9000000U;
    devUbConnection.sqDepth = 32U;
    devUbConnection.tpn = 200U;
    for (size_t idx = 0; idx < sizeof(devUbConnection.rmtReverseEid.raw); ++idx) {
        devUbConnection.rmtReverseEid.raw[idx] = static_cast<uint8_t>(idx + 1U);
    }

    HcclAiRMAWQ wq{};
    SqContext sq{};
    devUbConnection.SetWqInfo(wq);
    devUbConnection.SetSqContextInfo(sq);

    EXPECT_EQ(wq.jettyId, devUbConnection.jettyId);
    EXPECT_EQ(wq.dbAddr, devUbConnection.dbAddr);
    EXPECT_EQ(wq.sqVA, devUbConnection.sqBuffVa);
    EXPECT_EQ(wq.sqDepth, devUbConnection.sqDepth * 4U);
    EXPECT_EQ(wq.tp_id, devUbConnection.tpn);
    EXPECT_EQ(sq.contextInfo.ubJfs.jfsID, devUbConnection.jettyId);
    EXPECT_EQ(sq.contextInfo.ubJfs.dbVa, devUbConnection.dbAddr);
    EXPECT_EQ(sq.contextInfo.ubJfs.sqVa, devUbConnection.sqBuffVa);
    EXPECT_EQ(sq.contextInfo.ubJfs.sqDepth, devUbConnection.sqDepth * 4U);
    EXPECT_EQ(sq.contextInfo.ubJfs.tpID, devUbConnection.tpn);
    for (size_t idx = 0; idx < sizeof(devUbConnection.rmtReverseEid.raw); ++idx) {
        EXPECT_EQ(wq.rmtEid[idx], devUbConnection.rmtReverseEid.raw[idx]);
        EXPECT_EQ(sq.contextInfo.ubJfs.remoteEID[idx], devUbConnection.rmtReverseEid.raw[idx]);
    }
}

TEST_F(DevUbConnectionTest, set_wq_and_sq_context_info_memcpy_fail_throw)
{
    RdmaHandle rdmaHandle = reinterpret_cast<void*>(0x1000000);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);
    HcclAiRMAWQ wq{};
    SqContext sq{};
    MOCKER(memcpy_s).stubs().will(returnValue(-1));

    EXPECT_THROW(devUbConnection.SetWqInfo(wq), RmaConnException);
    EXPECT_THROW(devUbConnection.SetSqContextInfo(sq), RmaConnException);
    EXPECT_EQ(devUbConnection.status, RmaConnStatus::CONN_INVALID);
    EXPECT_EQ(devUbConnection.ubConnStatus, DevUbConnection::UbConnStatus::CONN_INVALID);
}

TEST_F(DevUbConnectionTest, rma_ub_connection_get_status_return_exchanging_and_ok)
{
    // Given
    RdmaHandle rdmaHandle = (void*)0x1000000;

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    std::string tag = "test";

    // construct DevUbConnection
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);
    devUbConnection.tpProtocol = TpProtocol::CTP;
    EXPECT_EQ(RmaConnStatus::INIT, devUbConnection.status);

    //  When:
    u32 tokenValue = 1;
    // Then
    RmaConnStatus status = devUbConnection.GetStatus();
    EXPECT_EQ(RmaConnStatus::INIT, status);
    status = devUbConnection.GetStatus();
    EXPECT_EQ(RmaConnStatus::EXCHANGEABLE, status);
    EXPECT_EQ(DevUbConnection::UbConnStatus::JETTY_CREATED, devUbConnection.ubConnStatus);

    auto rmtDto = devUbConnection.GetExchangeDto();
    devUbConnection.ParseRmtExchangeDto(*rmtDto);
    devUbConnection.ImportRmtDto();
    EXPECT_EQ(RmaConnStatus::EXCHANGEABLE, status);
    EXPECT_EQ(DevUbConnection::UbConnStatus::JETTY_IMPORTING, devUbConnection.ubConnStatus);

    status = devUbConnection.GetStatus();
    EXPECT_EQ(RmaConnStatus::READY, status);
    EXPECT_EQ(DevUbConnection::UbConnStatus::READY, devUbConnection.ubConnStatus);

    status = devUbConnection.GetStatus();
    EXPECT_EQ(RmaConnStatus::READY, status);

    string msg = devUbConnection.Describe();
    EXPECT_NE(0, msg.length());
    GlobalMockObject::verify();
}

TEST_F(DevUbConnectionTest, rma_ub_connection_get_rma_conn_lite)
{
    // Given: socket time out
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::TIMEOUT));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    const char* fakeSocName = "Ascend910B1";
    MOCKER(aclrtGetSocName).stubs().will(returnValue(fakeSocName));

    // When
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);
    devUbConnection.tpProtocol = TpProtocol::CTP;
    EXPECT_EQ(RmaConnStatus::INIT, devUbConnection.status);

    //  When:
    u32 tokenValue = 1;
    // Then
    RmaConnStatus status = devUbConnection.GetStatus();
    EXPECT_EQ(RmaConnStatus::INIT, status);
    status = devUbConnection.GetStatus();
    EXPECT_EQ(RmaConnStatus::EXCHANGEABLE, status);
    EXPECT_EQ(DevUbConnection::UbConnStatus::JETTY_CREATED, devUbConnection.ubConnStatus);
}

TEST_F(DevUbConnectionTest, rma_ub_connection_getstatus_change_status_ready)
{
    // construct DevUbConnection
    RdmaHandle rdmaHandle = (void*)0x1000000;

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    std::string tag = "test";
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);
    devUbConnection.tpProtocol = TpProtocol::CTP;
    EXPECT_EQ(RmaConnStatus::INIT, devUbConnection.status);
    //  When:

    // Then
    RmaConnStatus status = devUbConnection.GetStatus();
    EXPECT_EQ(RmaConnStatus::INIT, status);
    status = devUbConnection.GetStatus();
    EXPECT_EQ(RmaConnStatus::EXCHANGEABLE, status);
    EXPECT_EQ(DevUbConnection::UbConnStatus::JETTY_CREATED, devUbConnection.ubConnStatus);

    auto rmtDto = devUbConnection.GetExchangeDto();
    devUbConnection.ParseRmtExchangeDto(*rmtDto);
    devUbConnection.ImportRmtDto();
    EXPECT_EQ(RmaConnStatus::EXCHANGEABLE, status);
    EXPECT_EQ(DevUbConnection::UbConnStatus::JETTY_IMPORTING, devUbConnection.ubConnStatus);

    status = devUbConnection.GetStatus();
    EXPECT_EQ(RmaConnStatus::READY, status);
    EXPECT_EQ(DevUbConnection::UbConnStatus::READY, devUbConnection.ubConnStatus);

    status = devUbConnection.GetStatus();
    EXPECT_EQ(RmaConnStatus::READY, status);
    EXPECT_EQ(DevUbConnection::UbConnStatus::READY, devUbConnection.ubConnStatus);

    string msg = devUbConnection.Describe();
    EXPECT_NE(0, msg.length());
    GlobalMockObject::verify();
}

TEST_F(DevUbConnectionTest, rma_ub_connection_getstatus_change_status_invalid)
{
    // construct DevUbConnection
    RdmaHandle rdmaHandle = (void*)0x1000000;

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    std::string tag = "test";
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);
    devUbConnection.tpProtocol = TpProtocol::CTP;

    //  When:
    devUbConnection.ubConnStatus = DevUbConnection::UbConnStatus::INVALID;

    // Then
    EXPECT_THROW(devUbConnection.GetStatus(), RmaConnException);
}

// 该接口主流程已不使用
TEST_F(DevUbConnectionTest, rma_ub_connection_suspend_change_status_suspend)
{
    // construct DevUbConnection
    RdmaHandle rdmaHandle = (void*)0x1000000;

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    std::string tag = "test";
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);

    //  When:
    MOCKER(HrtFree).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    devUbConnection.jettyHandle = 1;
    devUbConnection.sqBuffVa = 0x1000000;
    devUbConnection.status = RmaConnStatus::READY;
    devUbConnection.ubConnStatus = DevUbConnection::UbConnStatus::READY;

    // Then
    EXPECT_EQ(true, devUbConnection.Suspend());
    EXPECT_EQ(true, devUbConnection.Suspend());
}

TEST_F(DevUbConnectionTest, rma_ub_connection_suspend_change_status_invalid)
{
    // construct DevUbConnection
    RdmaHandle rdmaHandle = (void*)0x1000000;

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    std::string tag = "test";
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);

    //  When:
    MOCKER(HrtFree).stubs().with(mockcpp::any()).will(ignoreReturnValue());

    // Then
    devUbConnection.status = RmaConnStatus::CLOSE;
    EXPECT_THROW(devUbConnection.Suspend(), RmaConnException);
    EXPECT_EQ(RmaConnStatus::CONN_INVALID, devUbConnection.status);

    string msg = devUbConnection.Describe();
    EXPECT_NE(0, msg.length());
}

TEST_F(DevUbConnectionTest, rma_net_connection_prepare_write_task_with_db_send)
{
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    MOCKER(HrtRaQpCreate).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(fakeQpHandle));

    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);

    // Given
    MemoryBuffer localMemBuffer1(0, 0, 0);
    MemoryBuffer remoteMemBuffer1(2000, 0, 0);
    SqeConfig config{};
    config.wqeMode = WqeMode::DB_SEND;
    // When
    auto result1 = devUbConnection.PrepareWrite(remoteMemBuffer1, localMemBuffer1, config);
    // Then
    EXPECT_EQ(nullptr, result1);

    MemoryBuffer localMemBuffer2(0, 100, 0);
    MemoryBuffer remoteMemBuffer2(2000, 100, 0);
    auto result2 = devUbConnection.PrepareWrite(remoteMemBuffer2, localMemBuffer2, config);
    EXPECT_NE(nullptr, result2);
    EXPECT_EQ(TaskType::UB_SEND, result2->GetType());

    MemoryBuffer localMemBuffer10(0, 0, 0);
    MemoryBuffer remoteMemBuffer10(2000, 10, 0);
    EXPECT_THROW(devUbConnection.PrepareWrite(remoteMemBuffer10, localMemBuffer10, config), InvalidParamsException);
}

TEST_F(DevUbConnectionTest, rma_net_connection_prepare_write_task_with_dwqe)
{
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    MOCKER(HrtRaQpCreate).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(fakeQpHandle));
    HrtRaUbSendWrRespParam postSendRes;
    postSendRes.dwqeSize = 128;
    MOCKER(HrtRaUbPostSend).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(postSendRes));

    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);

    // Given
    MemoryBuffer localMemBuffer1(0, 0, 0);
    MemoryBuffer remoteMemBuffer1(2000, 0, 0);
    SqeConfig config{};
    config.wqeMode = WqeMode::DWQE;
    // When
    auto result1 = devUbConnection.PrepareWrite(remoteMemBuffer1, localMemBuffer1, config);
    // Then

    MemoryBuffer localMemBuffer2(0, 100, 0);
    MemoryBuffer remoteMemBuffer2(2000, 100, 0);
    auto result2 = devUbConnection.PrepareWrite(remoteMemBuffer2, localMemBuffer2, config);
    EXPECT_NE(nullptr, result2);
    EXPECT_EQ(TaskType::UB_DIRECT_SEND, result2->GetType());

    MemoryBuffer localMemBuffer10(0, 0, 0);
    MemoryBuffer remoteMemBuffer10(2000, 10, 0);
    EXPECT_THROW(devUbConnection.PrepareWrite(remoteMemBuffer10, localMemBuffer10, config), InvalidParamsException);
}

TEST_F(DevUbConnectionTest, rma_net_connection_prepare_read_task_with_db_send)
{
    // Given
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    // When
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);

    MemoryBuffer localMemBuffer(0, 1000, 0);
    MemoryBuffer remoteMemBuffer(2000, 1000, 0);
    SqeConfig config{};
    config.wqeMode = WqeMode::DB_SEND;
    auto task = devUbConnection.PrepareRead(remoteMemBuffer, localMemBuffer, config);
    EXPECT_NE(nullptr, task);
}

TEST_F(DevUbConnectionTest, rma_net_connection_prepare_read_task_with_dwqe)
{
    // Given
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    HrtRaUbSendWrRespParam postSendRes;
    postSendRes.dwqeSize = 128;
    MOCKER(HrtRaUbPostSend).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(postSendRes));

    // When
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);

    MemoryBuffer localMemBuffer(0, 1000, 0);
    MemoryBuffer remoteMemBuffer(2000, 1000, 0);
    SqeConfig config{};
    config.wqeMode = WqeMode::DWQE;
    auto task = devUbConnection.PrepareRead(remoteMemBuffer, localMemBuffer, config);
    EXPECT_NE(nullptr, task);
}

TEST_F(DevUbConnectionTest, rma_net_connection_prepare_read_reduce_task_with_db_send)
{
    // Given
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    // When
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);

    MemoryBuffer localMemBuffer(0, 1000, 0);
    MemoryBuffer remoteMemBuffer(2000, 1000, 0);
    SqeConfig config{};
    config.wqeMode = WqeMode::DB_SEND;
    auto task
        = devUbConnection.PrepareReadReduce(remoteMemBuffer, localMemBuffer, DataType::INT8, ReduceOp::SUM, config);
    EXPECT_NE(nullptr, task);
}

TEST_F(DevUbConnectionTest, rma_net_connection_prepare_read_reduce_task_with_dwqe)
{
    // Given
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    HrtRaUbSendWrRespParam postSendRes;
    postSendRes.dwqeSize = 128;
    MOCKER(HrtRaUbPostSend).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(postSendRes));

    // When
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);

    MemoryBuffer localMemBuffer(0, 1000, 0);
    MemoryBuffer remoteMemBuffer(2000, 1000, 0);
    SqeConfig config{};
    config.wqeMode = WqeMode::DWQE;
    auto task
        = devUbConnection.PrepareReadReduce(remoteMemBuffer, localMemBuffer, DataType::INT8, ReduceOp::SUM, config);
    EXPECT_NE(nullptr, task);
}

TEST_F(DevUbConnectionTest, rma_ub_connection_prepare_write_reduce_task_with_db_send)
{
    // Given
    RdmaHandle rdmaHandle = (void*)0x1000000;
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    // When
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);

    MemoryBuffer localMemBuffer(0, 1000, 0);
    MemoryBuffer remoteMemBuffer(2000, 1000, 0);
    SqeConfig config{};
    config.wqeMode = WqeMode::DB_SEND;

    auto task
        = devUbConnection.PrepareWriteReduce(remoteMemBuffer, localMemBuffer, DataType::INT8, ReduceOp::SUM, config);
    EXPECT_NE(nullptr, task);
    EXPECT_EQ(TaskType::UB_SEND, task->GetType());
}

TEST_F(DevUbConnectionTest, rma_ub_connection_prepare_write_reduce_task_with_dwqe)
{
    // Given
    RdmaHandle rdmaHandle = (void*)0x1000000;
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    HrtRaUbSendWrRespParam postSendRes;
    postSendRes.dwqeSize = 128;
    MOCKER(HrtRaUbPostSend).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(postSendRes));

    // When
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);

    MemoryBuffer localMemBuffer(0, 1000, 0);
    MemoryBuffer remoteMemBuffer(2000, 1000, 0);
    SqeConfig config{};
    config.wqeMode = WqeMode::DWQE;

    auto task
        = devUbConnection.PrepareWriteReduce(remoteMemBuffer, localMemBuffer, DataType::INT8, ReduceOp::SUM, config);
    EXPECT_NE(nullptr, task);
    EXPECT_EQ(TaskType::UB_DIRECT_SEND, task->GetType());
}

TEST_F(DevUbConnectionTest, rma_net_connection_prepare_write_with_notify_task_with_db_send)
{
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    MOCKER(HrtRaQpCreate).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(fakeQpHandle));

    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);

    // Given
    MemoryBuffer localMemBuffer(0, 100, 0);
    MemoryBuffer remoteMemBuffer(2000, 100, 0);
    MemoryBuffer remoteNotifyMemBuffer(8000, 100, 0);
    SqeConfig config{};
    config.wqeMode = WqeMode::DB_SEND;

    auto task
        = devUbConnection.PrepareWriteWithNotify(remoteMemBuffer, localMemBuffer, 1, remoteNotifyMemBuffer, config);
    EXPECT_NE(nullptr, task);
    EXPECT_EQ(TaskType::UB_SEND, task->GetType());
}

TEST_F(DevUbConnectionTest, rma_net_connection_prepare_write_with_notify_task_with_dwqe)
{
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    MOCKER(HrtRaQpCreate).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(fakeQpHandle));
    HrtRaUbSendWrRespParam postSendRes;
    postSendRes.dwqeSize = 128;
    MOCKER(HrtRaUbPostSend).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(postSendRes));

    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);

    // Given
    MemoryBuffer localMemBuffer(0, 100, 0);
    MemoryBuffer remoteMemBuffer(2000, 100, 0);
    MemoryBuffer remoteNotifyMemBuffer(8000, 100, 0);
    SqeConfig config{};
    config.wqeMode = WqeMode::DWQE;

    auto task
        = devUbConnection.PrepareWriteWithNotify(remoteMemBuffer, localMemBuffer, 1, remoteNotifyMemBuffer, config);
    EXPECT_NE(nullptr, task);
    EXPECT_EQ(TaskType::UB_DIRECT_SEND, task->GetType());
}

TEST_F(DevUbConnectionTest, rma_ub_connection_prepare_write_reduce_with_notify_task_with_db_send)
{
    // Given
    RdmaHandle rdmaHandle = (void*)0x1000000;
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    // When
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);

    MemoryBuffer localMemBuffer(0, 1000, 0);
    MemoryBuffer remoteMemBuffer(2000, 1000, 0);
    MemoryBuffer remoteNotifyMemBuffer(8000, 100, 0);
    SqeConfig config{};
    config.wqeMode = WqeMode::DB_SEND;

    auto task = devUbConnection.PrepareWriteReduceWithNotify(
        remoteMemBuffer, localMemBuffer, DataType::INT8, ReduceOp::SUM, 1, remoteNotifyMemBuffer, config);
    EXPECT_NE(nullptr, task);
    EXPECT_EQ(TaskType::UB_SEND, task->GetType());
}

TEST_F(DevUbConnectionTest, rma_ub_connection_prepare_write_reduce_with_notify_task_with_dwqe)
{
    // Given
    RdmaHandle rdmaHandle = (void*)0x1000000;
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    HrtRaUbSendWrRespParam postSendRes;
    postSendRes.dwqeSize = 128;
    MOCKER(HrtRaUbPostSend).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(postSendRes));

    // When
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);

    MemoryBuffer localMemBuffer(0, 1000, 0);
    MemoryBuffer remoteMemBuffer(2000, 1000, 0);
    MemoryBuffer remoteNotifyMemBuffer(8000, 100, 0);
    SqeConfig config{};
    config.wqeMode = WqeMode::DWQE;

    auto task = devUbConnection.PrepareWriteReduceWithNotify(
        remoteMemBuffer, localMemBuffer, DataType::INT8, ReduceOp::SUM, 1, remoteNotifyMemBuffer, config);
    EXPECT_NE(nullptr, task);
    EXPECT_EQ(TaskType::UB_DIRECT_SEND, task->GetType());
}

TEST_F(DevUbConnectionTest, rma_ub_connection_prepare_inline_write_task_with_db_send)
{
    // Given
    RdmaHandle rdmaHandle = (void*)0x1000000;
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    // When
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);

    MemoryBuffer remoteMemBuffer(2000, 1000, 0);
    SqeConfig config{};
    config.wqeMode = WqeMode::DB_SEND;

    auto task = devUbConnection.PrepareInlineWrite(remoteMemBuffer, 1, config);
    EXPECT_NE(nullptr, task);
    EXPECT_EQ(TaskType::UB_SEND, task->GetType());
}
// 1011
TEST_F(DevUbConnectionTest, rma_ub_connection_prepare_inline_write_tasks_with_writeval_send)
{
    // Given
    RdmaHandle rdmaHandle = (void*)0x1000000;
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    // When
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);

    MemoryBuffer remoteMemBuffer(2000, 1000, 0);
    SqeConfig config{};
    config.wqeMode = WqeMode::WRITE_VALUE;

    auto task = devUbConnection.PrepareInlineWrite(remoteMemBuffer, 1, config);
    EXPECT_NE(task, nullptr);
    EXPECT_EQ(TaskType::WRITE_VALUE, task->GetType());
}

TEST_F(DevUbConnectionTest, rma_ub_connection_prepare_inline_write_task_with_dwqe)
{
    // Given
    RdmaHandle rdmaHandle = (void*)0x1000000;
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    HrtRaUbSendWrRespParam postSendRes;
    postSendRes.dwqeSize = 128;
    MOCKER(HrtRaUbPostSend).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(postSendRes));

    // When
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);

    MemoryBuffer remoteMemBuffer(2000, 1000, 0);
    SqeConfig config{};
    config.wqeMode = WqeMode::DWQE;

    auto task = devUbConnection.PrepareInlineWrite(remoteMemBuffer, 1, config);
    EXPECT_NE(nullptr, task);
}

TEST_F(DevUbConnectionTest, rma_net_connection_prepare_write_task_in_offload_mode_with_db_send)
{
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::OK));
    MOCKER_CPP(&RdmaHandleManager::GetTokenIdInfo).stubs().will(returnValue(pair<u64, u32>{}));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    MOCKER(HrtRaQpCreate).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(fakeQpHandle));
    HrtRaUbSendWrRespParam postSendRes1;
    postSendRes1.dwqeSize = 64;
    HrtRaUbSendWrRespParam postSendRes2;
    postSendRes2.dwqeSize = 128;
    postSendRes2.piVal = 3;
    HrtRaUbSendWrRespParam postSendRes3;
    postSendRes3.dwqeSize = 100;
    MOCKER(HrtRaUbPostSend)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(postSendRes1))
        .then(returnValue(postSendRes2))
        .then(returnValue(postSendRes3));

    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OFFLOAD);
    EXPECT_EQ(devUbConnection.GetSqDepth(), 128U);

    // Given
    MemoryBuffer localMemBuffer1(0, 0, 0);
    MemoryBuffer remoteMemBuffer1(2000, 0, 0);
    SqeConfig config{};
    config.wqeMode = WqeMode::DB_SEND;
    // When
    auto result1
        = devUbConnection.PrepareWriteReduce(remoteMemBuffer1, localMemBuffer1, DataType::INT8, ReduceOp::SUM, config);
    // Then

    MemoryBuffer localMemBuffer2(0, 100, 0);
    MemoryBuffer remoteMemBuffer2(2000, 100, 0);
    auto result2
        = devUbConnection.PrepareWriteReduce(remoteMemBuffer2, localMemBuffer2, DataType::INT8, ReduceOp::SUM, config);
    EXPECT_NE(nullptr, result2);
    EXPECT_EQ(TaskType::UB_SEND, result2->GetType());

    auto result3
        = devUbConnection.PrepareWriteReduce(remoteMemBuffer2, localMemBuffer2, DataType::INT8, ReduceOp::SUM, config);
    EXPECT_NE(nullptr, result3);
    EXPECT_EQ(TaskType::UB_SEND, result3->GetType());

    EXPECT_THROW(
        devUbConnection.PrepareWriteReduce(remoteMemBuffer2, localMemBuffer2, DataType::INT8, ReduceOp::SUM, config),
        InvalidParamsException);

    MemoryBuffer localMemBuffer10(0, 0, 0);
    MemoryBuffer remoteMemBuffer10(2000, 10, 0);
    EXPECT_THROW(
        devUbConnection.PrepareWriteReduce(remoteMemBuffer10, localMemBuffer10, DataType::INT8, ReduceOp::SUM, config),
        InvalidParamsException);

    MOCKER(HrtRaUbPostNops).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any());
    MOCKER(HrtUbDbSend).stubs().with(mockcpp::any(), mockcpp::any());
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().will(returnValue(static_cast<DevId>(1)));
    Stream stream;
    devUbConnection.AddNop(stream);

    devUbConnection.piVal = 9999;
    EXPECT_THROW(devUbConnection.AddNop(stream), InvalidParamsException);

    DevUbConnection devUbConnection1(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);
    ;
    EXPECT_NO_THROW(devUbConnection1.AddNop(stream));
}

TEST_F(DevUbConnectionTest, rma_ub_connection_get_pi_ci_sqDepth_ok)
{
    // Given: socket time out
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::TIMEOUT));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    // When
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);

    // Then
    EXPECT_EQ(devUbConnection.GetPiVal(), 0);
    EXPECT_EQ(devUbConnection.GetCiVal(), 0);
    EXPECT_EQ(devUbConnection.GetSqDepth(), 8192);
}

TEST_F(DevUbConnectionTest, rma_ub_connection_get_jfcMode_and_jettyHandle_ok)
{
    // Given: socket time out
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::TIMEOUT));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    // When
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);
    JettyHandle jettyHandle = 100;
    devUbConnection.jettyHandle = jettyHandle;

    // Then
    EXPECT_EQ(devUbConnection.GetUbJfcMode(), HrtUbJfcMode::STARS_POLL);
    EXPECT_EQ(devUbConnection.GetJettyHandle(), jettyHandle);
}

TEST_F(DevUbConnectionTest, GetStarsPollUbConns_ok)
{
    // Given: socket time out
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::TIMEOUT));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    // When
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);
    std::vector<RmaConnection*> conns;
    conns.push_back(&devUbConnection);
    std::vector<DevUbConnection*> devUbConns = GetStarsPollUbConns(conns);

    // Then
    EXPECT_EQ(devUbConns.size(), 1);
}

TEST_F(DevUbConnectionTest, IfNeedUpdatingUbCi_false)
{
    // Given: socket time out
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::TIMEOUT));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    // When
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);
    devUbConnection.piVal = 10;
    devUbConnection.ciVal = 0;
    std::vector<DevUbConnection*> conns;
    conns.push_back(&devUbConnection);
    bool ret = IfNeedUpdatingUbCi(conns);

    // Then
    EXPECT_EQ(ret, false);
}

TEST_F(DevUbConnectionTest, IfNeedUpdatingUbCi_true)
{
    // Given: socket time out
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue((SocketStatus)SocketStatus::TIMEOUT));
    RdmaHandle rdmaHandle = (void*)0x1000000;
    string tag = "SENDRECV";
    QpHandle fakeQpHandle = (void*)0x1000000;

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    // When
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);
    devUbConnection.piVal = 4000;
    devUbConnection.ciVal = 0;
    std::vector<DevUbConnection*> conns;
    conns.push_back(&devUbConnection);
    bool ret = IfNeedUpdatingUbCi(conns);

    // Then
    EXPECT_EQ(ret, false);
}

TEST_F(DevUbConnectionTest, getExchangeDto_test)
{
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    RdmaHandle rdmaHandle = (void*)0x1000000;

    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);
    devUbConnection.status = RmaConnStatus::READY;
    devUbConnection.GetExchangeDto();
    devUbConnection.GetUniqueId();
};

TEST_F(DevUbConnectionTest, rma_ub_connection_ready_import_jetty)
{
    // construct DevUbConnection
    RdmaHandle rdmaHandle = (void*)0x1000000;
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    std::string tag = "test";
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);
    EXPECT_EQ(RmaConnStatus::INIT, devUbConnection.status);

    //  When:
    devUbConnection.ubConnStatus = DevUbConnection::UbConnStatus::READY;

    // Then
    EXPECT_NO_THROW(devUbConnection.ImportRmtDto());
}

TEST_F(DevUbConnectionTest, tp_import_test)
{
    MOCKER(HrtGetDevicePhyIdByUserDevId).defaults().will(returnValue(static_cast<s32>(0)));
    // construct DevUbConnection
    RdmaHandle rdmaHandle = (void*)0x1000000;

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 10, 11, 10, 11);
    linkData.localAddr_ = IpAddress("10.0.0.1");
    linkData.remoteAddr_ = IpAddress("10.0.0.2");
    linkData.linkProtocol_ = LinkProtocol::UB_TP;
    std::string tag = "test";
    DevUbTpConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);
    EXPECT_EQ(devUbConnection.GetStatus(), RmaConnStatus::INIT); // TP_INFO_GETTING
    EXPECT_EQ(devUbConnection.GetStatus(), RmaConnStatus::EXCHANGEABLE);

    // Then
    auto rmtDto = devUbConnection.GetExchangeDto();
    devUbConnection.ParseRmtExchangeDto(*rmtDto);
    devUbConnection.ImportRmtDto();
    EXPECT_EQ(devUbConnection.status, RmaConnStatus::EXCHANGEABLE);
    EXPECT_EQ(DevUbConnection::UbConnStatus::JETTY_IMPORTING, devUbConnection.ubConnStatus);

    EXPECT_EQ(devUbConnection.GetStatus(), RmaConnStatus::READY);
    EXPECT_EQ(DevUbConnection::UbConnStatus::READY, devUbConnection.ubConnStatus);
    GlobalMockObject::verify();
}

TEST_F(DevUbConnectionTest, ctp_import_test)
{
    // construct DevUbConnection
    RdmaHandle rdmaHandle = (void*)0x1000000;

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 10, 11, 10, 11);
    linkData.localAddr_ = IpAddress("11.0.0.1");
    linkData.remoteAddr_ = IpAddress("11.0.0.2");
    linkData.linkProtocol_ = LinkProtocol::UB_CTP;
    std::string tag = "test";
    DevUbCtpConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);
    EXPECT_EQ(devUbConnection.GetStatus(), RmaConnStatus::INIT); // TP_INFO_GETTING
    EXPECT_EQ(devUbConnection.GetStatus(), RmaConnStatus::EXCHANGEABLE);

    // Then
    auto rmtDto = devUbConnection.GetExchangeDto();
    devUbConnection.ParseRmtExchangeDto(*rmtDto);
    devUbConnection.ImportRmtDto();
    EXPECT_EQ(devUbConnection.status, RmaConnStatus::EXCHANGEABLE);
    EXPECT_EQ(DevUbConnection::UbConnStatus::JETTY_IMPORTING, devUbConnection.ubConnStatus);

    EXPECT_EQ(devUbConnection.GetStatus(), RmaConnStatus::READY);
    EXPECT_EQ(DevUbConnection::UbConnStatus::READY, devUbConnection.ubConnStatus);

    devUbConnection.tpInfo.tpHandle = 1;                // 控制非0
    EXPECT_NO_THROW(devUbConnection.ReleaseResource()); // 释放tp报错但是不影响流程
    GlobalMockObject::verify();
}

constexpr uint64_t expectSqBuffVa = 10;
constexpr uint32_t expectJettySqDepth = 512U;
uint32_t capturedJettySqDepth = 0U;
uint32_t capturedJettySqBufSize = 0U;
RequestHandle RaUbCreateJettyAsync_stub(
    const RdmaHandle handle, const HrtRaUbCreateJettyParam& in, vector<char_t>& out, void*& jettyHandle)
{
    capturedJettySqDepth = in.sqDepth;
    capturedJettySqBufSize = in.sqBufSize;
    struct QpCreateInfo info;
    info.ub.sqBuffVa = expectSqBuffVa;
    info.ub.id = 1;
    info.key.size = 0;
    info.key.value[0] = 0;
    info.ub.dbAddr = 0;
    out.resize(sizeof(info));
    memcpy(out.data(), &info, sizeof(info));
    jettyHandle = reinterpret_cast<void*>(0x12345678ULL);
    return static_cast<RequestHandle>(0x12345678ULL);
}
TEST_F(DevUbConnectionTest, Ut_CreateJetty_When_CorrectParams_ReturnIsOk)
{
    // construct DevUbConnection
    RdmaHandle rdmaHandle = (void*)0x1000000;
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 10, 11, 10, 11);
    linkData.remoteAddr_ = IpAddress("11.0.0.2");
    linkData.linkProtocol_ = LinkProtocol::UB_CTP;
    std::string tag = "test";

    // When
    capturedJettySqDepth = 0U;
    capturedJettySqBufSize = 0U;
    MOCKER(RaUbCreateJettyAsync).stubs().will(invoke(RaUbCreateJettyAsync_stub));
    DevUbCtpConnection devUbCtpConn(
        rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE, false, HrtUbJfcMode::STARS_POLL,
        IpAddress(), IpAddress(), static_cast<u8>(UB_QOS_DEFAULT), TpManager::TA_TIMEOUT_NOT_SET, COMM_ENGINE_RESERVED,
        expectJettySqDepth);

    // Then
    devUbCtpConn.CreateJetty(false);
    devUbCtpConn.SetJettyInfo();
    EXPECT_EQ(devUbCtpConn.sqBuffVa, expectSqBuffVa);
    EXPECT_EQ(capturedJettySqDepth, expectJettySqDepth);
    EXPECT_EQ(capturedJettySqBufSize, expectJettySqDepth * 64U * 4U);
}

TEST_F(DevUbConnectionTest, Ut_Describe_Tp_Mode)
{
    // construct DevUbConnection
    RdmaHandle rdmaHandle = (void*)0x1000000;

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 10, 11, 10, 11);
    linkData.localAddr_ = IpAddress("10.0.0.1");
    linkData.remoteAddr_ = IpAddress("10.0.0.2");
    linkData.linkProtocol_ = LinkProtocol::UB_TP;
    std::string tag = "test";

    DevUbTpConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);

    MOCKER(HrtRaGetTpAttrAsync).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    std::string testDfx = "";
    HcclResult ret = devUbConnection.Describe(testDfx);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(DevUbConnectionTest, Ut_ImportRmtDto)
{
    MOCKER_CPP(&DevUbConnection::ThrowAbnormalStatus).stubs();
    // Given
    RdmaHandle rdmaHandle = (void*)0x1000000;

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    std::string tag = "test";

    // construct DevUbConnection
    DevUbConnection devUbConnection(rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);
    devUbConnection.ubConnStatus = DevUbConnection::UbConnStatus::JETTY_CREATED;
    devUbConnection.tpProtocol = TpProtocol::UBOE;
    devUbConnection.ImportRmtDto();
    EXPECT_EQ(DevUbConnection::UbConnStatus::JETTY_IMPORTING, devUbConnection.ubConnStatus);
    GlobalMockObject::verify();
}

TEST_F(DevUbConnectionTest, Ut_UbRtpConnection_Constructor_SetsTpProtocolAndQos)
{
    constexpr u32 expectedSqDepth = 512U;
    RdmaHandle rdmaHandle = (void*)0x1000000;
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    Hccl::IpAddress locIpv4Addr("10.0.0.1");
    Hccl::IpAddress rmtIpv4Addr("10.0.0.2");
    constexpr u8 qos = 6U;

    DevUbRtpConnection ubRtpConn(
        rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE, true, HrtUbJfcMode::STARS_POLL,
        locIpv4Addr, rmtIpv4Addr, qos, TpManager::TA_TIMEOUT_NOT_SET, COMM_ENGINE_RESERVED, expectedSqDepth);

    EXPECT_EQ(ubRtpConn.tpProtocol, TpProtocol::UB_RTP);
    EXPECT_EQ(ubRtpConn.jettyTimeOut, 16);
    EXPECT_EQ(ubRtpConn.qos_, qos);
    EXPECT_EQ(ubRtpConn.GetSqDepth(), expectedSqDepth);
}

static HcclResult StubGetTpInfoWithMappedQos(TpManager*, const RaUbGetTpInfoParam&, TpInfo& tpInfo, bool)
{
    tpInfo.tpHandle = 0x555ULL;
    tpInfo.hasMappedJettyPriority = true;
    tpInfo.mappedJettyPriority = 6U;
    return HCCL_SUCCESS;
}

static u8 gCapturedJettyCreateQos = 0U;
RequestHandle CaptureQosRaUbCreateJettyAsync(
    const RdmaHandle handle, const HrtRaUbCreateJettyParam& in, vector<char_t>& out, void*& jettyHandle)
{
    gCapturedJettyCreateQos = in.qos;
    struct QpCreateInfo info {};
    info.ub.sqBuffVa = expectSqBuffVa;
    info.ub.id = 1;
    info.key.size = 0;
    out.resize(sizeof(info));
    memcpy(out.data(), &info, sizeof(info));
    jettyHandle = reinterpret_cast<void*>(0x12345678ULL);
    return static_cast<RequestHandle>(0x12345678ULL);
}

TEST_F(DevUbConnectionTest, Ut_DevUsed_DeferJettyUntilTpReady_And_MapQos)
{
    GlobalMockObject::verify();
    gCapturedJettyCreateQos = 0U;
    MOCKER(RaUbCreateJettyAsync).stubs().will(invoke(CaptureQosRaUbCreateJettyAsync));
    MOCKER_CPP(&TpManager::GetTpInfo).stubs().will(returnValue(HCCL_E_AGAIN)).then(invoke(StubGetTpInfoWithMappedQos));

    RdmaHandle rdmaHandle = reinterpret_cast<void*>(0x1000000);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 20, 21, 20, 21);
    linkData.localAddr_ = IpAddress("12.0.0.1");
    linkData.remoteAddr_ = IpAddress("12.0.0.2");
    linkData.linkProtocol_ = LinkProtocol::UB_CTP;

    const u8 requestQos = 3U;
    DevUbCtpConnection devUbConnection(
        rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE, true, HrtUbJfcMode::STARS_POLL,
        IpAddress(), IpAddress(), requestQos);
    devUbConnection.tpProtocol = TpProtocol::CTP;

    EXPECT_EQ(devUbConnection.GetStatus(), RmaConnStatus::INIT);
    EXPECT_EQ(devUbConnection.ubConnStatus, DevUbConnection::UbConnStatus::TP_INFO_GETTING);
    EXPECT_EQ(devUbConnection.GetStatus(), RmaConnStatus::INIT);
    EXPECT_EQ(devUbConnection.ubConnStatus, DevUbConnection::UbConnStatus::JETTY_CREATING);
    EXPECT_EQ(devUbConnection.qos_, requestQos);
    EXPECT_EQ(gCapturedJettyCreateQos, static_cast<u8>(6U));
    EXPECT_EQ(devUbConnection.GetStatus(), RmaConnStatus::EXCHANGEABLE);
    GlobalMockObject::verify();
}

TEST_F(DevUbConnectionTest, Ut_CreateJetty_PassesConfiguredQos)
{
    gCapturedJettyCreateQos = 0U;
    MOCKER(RaUbCreateJettyAsync).stubs().will(invoke(CaptureQosRaUbCreateJettyAsync));

    RdmaHandle rdmaHandle = reinterpret_cast<void*>(0x1000000);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 10, 11, 10, 11);
    linkData.linkProtocol_ = LinkProtocol::UB_CTP;

    DevUbCtpConnection devUbCtpConn(
        rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE, false, HrtUbJfcMode::STARS_POLL,
        IpAddress(), IpAddress(), 5U);
    devUbCtpConn.tpInfo.tpHandle = 0x100ULL;
    devUbCtpConn.tpInfo.hasMappedJettyPriority = true;
    devUbCtpConn.tpInfo.mappedJettyPriority = 5U;
    devUbCtpConn.CreateJetty(false);
    EXPECT_EQ(gCapturedJettyCreateQos, static_cast<u8>(5U));
    GlobalMockObject::verify();
}

TEST_F(DevUbConnectionTest, Ut_AivEngine_Constructor_CreatesAivUrmaJfc)
{
    RdmaHandle rdmaHandle = (void*)0x1000000;
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);

    DevUbConnection devUbConnection(
        rdmaHandle, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE, false, HrtUbJfcMode::STARS_POLL,
        IpAddress(), IpAddress(), static_cast<u8>(UB_QOS_DEFAULT), TpManager::TA_TIMEOUT_NOT_SET, COMM_ENGINE_AIV);

    EXPECT_EQ(devUbConnection.engine_, COMM_ENGINE_AIV);
    EXPECT_EQ(devUbConnection.jfcHandle, static_cast<JfcHandle>(0x0));
    GlobalMockObject::verify();
}

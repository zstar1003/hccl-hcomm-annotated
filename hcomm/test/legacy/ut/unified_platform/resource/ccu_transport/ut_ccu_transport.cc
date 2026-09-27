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
#include "ccu_transport.h"

#include "hccp_async.h"

#include "port.h"
#include "tp_manager.h"
#include "hccl_common_v2.h"
#include "ccu_connection.h"
#include "socket_exception.h"
#include "invalid_params_exception.h"
#include "internal_exception.h"
#include "orion_adapter_rts.h"
#include "orion_adapter_hccp.h"
#include "rdma_handle_manager.h"
#include "log.h"

#undef private
#undef protected

using namespace Hccl;

class CcuTransportTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CcuTransportTest tests set up." << std::endl; }

    static void TearDownTestCase()
    {
        GlobalMockObject::verify();
        std::cout << "CcuTransportTest tests tear down." << std::endl;
    }

    virtual void SetUp()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CcuTransportTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CcuTransportTest TearDown" << std::endl;
    }
};

HcclResult
AllocCcuResStub(const int32_t deviceLogicId, const uint8_t dieId, const uint32_t num, std::vector<ResInfo>& resInfos)
{
    ResInfo resInfo(0, num);
    resInfos.emplace_back(resInfo);
    return HcclResult::HCCL_SUCCESS;
}

HcclResult AllocCcuResStubUnavail(
    const int32_t deviceLogicId, const uint8_t dieId, const uint32_t num, std::vector<ResInfo>& resInfos)
{
    ResInfo resInfo(0, num);
    resInfos.emplace_back(resInfo);
    return HcclResult::HCCL_E_UNAVAIL;
}

constexpr uint32_t DEFAULT_CCU_RESOURCE_NUM = 16; // 根据业务代码调整

using TransportTuple = tuple<unique_ptr<CcuTransport>, unique_ptr<Socket>, vector<unique_ptr<CcuJetty>>>;
TransportTuple MockMakeCcuTransport(bool allocCkeUnavailFlag, bool allocXnUnavailFlag)
{
    MOCKER(HrtGetDevicePhyIdByUserDevId)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(static_cast<s32>(MAX_MODULE_DEVICE_NUM - 1)));

    HcclResult OkResult = HcclResult::HCCL_SUCCESS;
    HcclResult AgainResult = HcclResult::HCCL_E_AGAIN;
    if (allocCkeUnavailFlag) {
        MOCKER(CcuDeviceManager::AllocCke).stubs().will(invoke(AllocCcuResStub));
    } else {
        MOCKER(CcuDeviceManager::AllocCke).stubs().will(invoke(AllocCcuResStubUnavail));
    }
    if (allocXnUnavailFlag) {
        MOCKER(CcuDeviceManager::AllocXn).stubs().will(invoke(AllocCcuResStub));
    } else {
        MOCKER(CcuDeviceManager::AllocXn).stubs().will(invoke(AllocCcuResStubUnavail));
    }
    MOCKER(CcuDeviceManager::ReleaseCke).stubs().will(returnValue(OkResult));
    MOCKER(CcuDeviceManager::ReleaseXn).stubs().will(returnValue(OkResult));
    MOCKER(CcuDeviceManager::GetCcuResourceSpaceBufInfo).stubs().will(returnValue(OkResult));
    MOCKER(CcuDeviceManager::GetCcuResourceSpaceTokenInfo).stubs().will(returnValue(OkResult));
    MOCKER(CcuDeviceManager::ConfigChannel).stubs().will(returnValue(OkResult));
    MOCKER_CPP(&CcuJetty::CreateJetty).stubs().will(returnValue(AgainResult)).then(returnValue(OkResult));
    MOCKER_CPP(&RdmaHandleManager::GetByIp).stubs().will(returnValue((void*)0x12345678));
    pair<uint32_t, uint32_t> fakeDieFuncPair = make_pair(1, 4);
    MOCKER_CPP(&RdmaHandleManager::GetDieAndFuncId).stubs().will(returnValue(fakeDieFuncPair));
    pair<TokenIdHandle, uint32_t> fakeTokenInfo = make_pair(0x12345678, 1);
    MOCKER_CPP(&RdmaHandleManager::GetTokenIdInfo).stubs().will(returnValue(fakeTokenInfo));
    MOCKER_CPP(&TpManager::GetTpInfo)
        .stubs()
        .will(returnValue(HcclResult::HCCL_E_AGAIN))
        .then(returnValue(HcclResult::HCCL_SUCCESS));

    constexpr uint64_t fakeMemAddr = 0x12345678;

    const uint32_t fakeTaJettyId = 1025;
    const uint64_t fakeSqBufVa = fakeMemAddr;
    const uint32_t fakeSqBufSize = 1024;
    const uint32_t fakeSqDepth = 4;
    const IpAddress locAddr{"1.1.1.1"};
    const IpAddress rmtAddr{"2.2.2.2"};

    CcuChannelInfo channelInfo;
    channelInfo.channelId = 1;
    channelInfo.dieId = 1;

    vector<unique_ptr<CcuJetty>> ccuJettys;
    vector<CcuJetty*> ccuJettyPtrs;
    for (uint32_t i = 0; i < 2; i++) {
        CcuJettyInfo jettyInfo;
        jettyInfo.jettyCtxId = 1 + i;
        jettyInfo.taJettyId = fakeTaJettyId + i;
        jettyInfo.sqDepth = fakeSqDepth;
        jettyInfo.wqeBBStartId = 16;
        jettyInfo.sqBufVa = fakeSqBufVa + i;
        jettyInfo.sqBufSize = fakeSqBufSize + i;
        channelInfo.jettyInfos.push_back(jettyInfo);
        auto ccuJetty = make_unique<CcuJetty>(locAddr, jettyInfo);
        ccuJettyPtrs.emplace_back(ccuJetty.get());
        ccuJettys.emplace_back(std::move(ccuJetty));
    }
    SocketHandle socketHandle = reinterpret_cast<SocketHandle>(0x123);
    unique_ptr<Socket> socket = make_unique<Socket>(
        socketHandle, locAddr, 65001, rmtAddr, string(), SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    socket->fdHandle = socketHandle;
    // 模拟CTP即可
    unique_ptr<CcuConnection> connection = make_unique<CcuCtpConnection>(locAddr, rmtAddr, channelInfo, ccuJettyPtrs);
    EXPECT_EQ(connection->Init(), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(connection->status, CcuConnStatus::INIT);
    CcuTransport::CclBufferInfo cclBuffer;
    cclBuffer.addr = fakeMemAddr;
    cclBuffer.size = fakeSqBufSize;

    unique_ptr<CcuTransport> transport = make_unique<CcuTransport>(socket.get(), std::move(connection), cclBuffer);
    return TransportTuple{std::move(transport), std::move(socket), std::move(ccuJettys)};
}

TEST_F(CcuTransportTest, Ut_GetStatus_When_InterfaceOk_Expect_Return_Ok)
{
    auto transportRes = MockMakeCcuTransport(true, true);
    auto transport = get<0>(transportRes).get();
    EXPECT_EQ(transport->Init(), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(transport->transStatus, CcuTransport::TransStatus::INIT);
    EXPECT_EQ(transport->locRes.ckes.size(), DEFAULT_CCU_RESOURCE_NUM);
    EXPECT_EQ(transport->locRes.xns.size(), DEFAULT_CCU_RESOURCE_NUM);

    const vector<uint32_t> cntCkes(DEFAULT_CCU_RESOURCE_NUM);
    transport->SetCntCke(cntCkes);

    const vector<char> handshakeMsg(128);
    transport->SetHandshakeMsg(handshakeMsg);

    for (uint32_t i = 0; i < 2; i++) { // Connection切换2步
        EXPECT_EQ(transport->GetStatus(), CcuTransport::TransStatus::INIT);
    }

    EXPECT_EQ(transport->GetStatus(), CcuTransport::TransStatus::SEND_ALL_INFO);

    transport->recvData = transport->sendData; // 模拟资源信息交换成功

    EXPECT_EQ(transport->GetStatus(), CcuTransport::TransStatus::RECV_ALL_INFO);

    EXPECT_EQ(transport->GetStatus(), CcuTransport::TransStatus::SEND_FIN);

    EXPECT_EQ(transport->GetStatus(), CcuTransport::TransStatus::RECVING_FIN);

    transport->recvFinishMsg = transport->sendFinishMsg; // 模拟握手信息交换成功

    EXPECT_EQ(transport->GetStatus(), CcuTransport::TransStatus::RECV_FIN);

    EXPECT_EQ(transport->GetStatus(), CcuTransport::TransStatus::READY);
    EXPECT_EQ(transport->GetStatus(), CcuTransport::TransStatus::READY);

    std::cout << transport->Describe();
    EXPECT_EQ(transport->GetDieId(), 1);
    EXPECT_EQ(transport->GetChannelId(), 1);
    EXPECT_EQ(transport->locRes.ckes.size(), DEFAULT_CCU_RESOURCE_NUM);
    EXPECT_EQ(transport->locRes.xns.size(), DEFAULT_CCU_RESOURCE_NUM);
    EXPECT_EQ(transport->locRes.cntCkes.size(), DEFAULT_CCU_RESOURCE_NUM);

    EXPECT_EQ(transport->GetLocCkeByIndex(0), 0);
    EXPECT_EQ(transport->GetLocCkeByIndex(1), 1);
    EXPECT_EQ(transport->GetLocCkeByIndex(2), 2);
    EXPECT_EQ(transport->GetLocCntCkeByIndex(0), 0);
    EXPECT_EQ(transport->GetLocXnByIndex(0), 0);
    EXPECT_EQ(transport->GetLocXnByIndex(1), 1);
    EXPECT_EQ(transport->GetLocXnByIndex(2), 2);
}

TEST_F(CcuTransportTest, Ut_InitFailed_When_InterfaceError_Expect_Return_Error)
{
    MOCKER(CcuDeviceManager::AllocCke).stubs().will(returnValue(HcclResult::HCCL_E_PARA));
    auto transportRes = MockMakeCcuTransport(true, true);
    auto transport = get<0>(transportRes).get();
    EXPECT_EQ(transport->Init(), HcclResult::HCCL_E_PARA);
}

TEST_F(CcuTransportTest, Ut_GetStatusFailed_When_ConnectionError_Expect_Return_Error)
{
    CcuTransport::TransStatus status = CcuTransport::TransStatus::INIT;
    try {
        CcuConnStatus fakeConnStatus = CcuConnStatus::CONN_INVALID;
        MOCKER_CPP(&CcuConnection::GetStatus).stubs().will(returnValue(fakeConnStatus));
        auto transportRes = MockMakeCcuTransport(true, true);
        auto transport = get<0>(transportRes).get();

        EXPECT_EQ(transport->Init(), HcclResult::HCCL_SUCCESS);
        EXPECT_EQ(transport->transStatus, CcuTransport::TransStatus::INIT);
        status = transport->GetStatus();
    } catch (InternalException& e) {
        EXPECT_EQ(CcuTransport::TransStatus::CONNECT_FAILED, status);
    } catch (exception& e) {
        HCCL_ERROR(e.what());
    } catch (...) {
        HCCL_ERROR("Unknown error occurs!");
    }
}

TEST_F(CcuTransportTest, Ut_GetStatusFailed_When_SocketError_Expect_Return_Error)
{
    CcuTransport::TransStatus status = CcuTransport::TransStatus::INIT;
    try {
        SocketStatus fakeSocketStatus = SocketStatus::INIT; // 当前用该状态表示socket初始化失败
        MOCKER_CPP(&Socket::GetAsyncStatus).stubs().will(returnValue(fakeSocketStatus));
        auto transportRes = MockMakeCcuTransport(true, true);
        auto transport = get<0>(transportRes).get();

        EXPECT_EQ(transport->Init(), HcclResult::HCCL_SUCCESS);
        EXPECT_EQ(transport->transStatus, CcuTransport::TransStatus::INIT);
        status = transport->GetStatus();
    } catch (InternalException& e) {
        EXPECT_EQ(CcuTransport::TransStatus::CONNECT_FAILED, status);
    } catch (exception& e) {
        HCCL_ERROR(e.what());
    } catch (...) {
        HCCL_ERROR("Unknown error occurs!");
    }
}

TEST_F(CcuTransportTest, Ut_GetStatusTimeOut_When_SocketTimeOut_Expect_Return_TimeOut)
{
    SocketStatus fakeSocketStatus = SocketStatus::TIMEOUT;
    MOCKER_CPP(&Socket::GetAsyncStatus).stubs().will(returnValue(fakeSocketStatus));
    auto transportRes = MockMakeCcuTransport(true, true);
    auto transport = get<0>(transportRes).get();
    EXPECT_EQ(transport->Init(), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(transport->transStatus, CcuTransport::TransStatus::INIT);
    EXPECT_EQ(transport->GetStatus(), CcuTransport::TransStatus::SOCKET_TIMEOUT);
}

TEST_F(CcuTransportTest, Ut_GetStatusError_When_SocketSendError_Expect_Return_Error)
{
    CcuTransport::TransStatus status = CcuTransport::TransStatus::INIT;
    try {
        MOCKER_CPP(&Socket::SendAsync).stubs().will(throws(SocketException("")));
        auto transportRes = MockMakeCcuTransport(true, true);
        auto transport = get<0>(transportRes).get();

        EXPECT_EQ(transport->Init(), HcclResult::HCCL_SUCCESS);
        for (uint32_t i = 0; i < 2; i++) { // Connection切换2步
            status = transport->GetStatus();
        }
    } catch (SocketException& e) {
        EXPECT_EQ(CcuTransport::TransStatus::CONNECT_FAILED, status);
    } catch (exception& e) {
        HCCL_ERROR(e.what());
    } catch (...) {
        HCCL_ERROR("Unknown error occurs!");
    }
}

TEST_F(CcuTransportTest, Ut_GetStatusError_When_HandshakeMsgInvalid_Expect_Return_Error)
{
    CcuTransport::TransStatus status = CcuTransport::TransStatus::INIT;
    try {
        auto transportRes = MockMakeCcuTransport(true, true);
        auto transport = get<0>(transportRes).get();

        EXPECT_EQ(transport->Init(), HcclResult::HCCL_SUCCESS);
        EXPECT_EQ(transport->transStatus, CcuTransport::TransStatus::INIT);
        EXPECT_EQ(transport->locRes.ckes.size(), DEFAULT_CCU_RESOURCE_NUM);
        EXPECT_EQ(transport->locRes.xns.size(), DEFAULT_CCU_RESOURCE_NUM);

        const vector<uint32_t> cntCkes(DEFAULT_CCU_RESOURCE_NUM);
        transport->SetCntCke(cntCkes);

        const vector<char> handshakeMsg(128);
        transport->SetHandshakeMsg(handshakeMsg);

        EXPECT_EQ(transport->GetStatus(), CcuTransport::TransStatus::INIT);
        EXPECT_EQ(transport->GetStatus(), CcuTransport::TransStatus::INIT);

        EXPECT_EQ(transport->GetStatus(), CcuTransport::TransStatus::SEND_ALL_INFO);

        transport->recvData = transport->sendData;

        EXPECT_EQ(transport->GetStatus(), CcuTransport::TransStatus::RECV_ALL_INFO);

        transport->attr.handshakeMsg.push_back('a');
        status = transport->GetStatus();
    } catch (InvalidParamsException& e) {
        EXPECT_EQ(CcuTransport::TransStatus::CONNECT_FAILED, status);
    } catch (exception& e) {
        HCCL_ERROR(e.what());
    } catch (...) {
        HCCL_ERROR("Unknown error occurs!");
    }
}

TEST_F(CcuTransportTest, Ut_Init_UNAVAIL_When_AppendCkes_Return_UNAVAIL)
{
    CcuConnStatus fakeConnStatus = CcuConnStatus::CONN_INVALID;
    MOCKER_CPP(&CcuConnection::GetStatus).stubs().will(returnValue(fakeConnStatus));
    auto transportRes = MockMakeCcuTransport(false, true);
    auto transport = get<0>(transportRes).get();

    HcclResult unavailResult = HcclResult::HCCL_E_UNAVAIL;

    EXPECT_EQ(transport->Init(), unavailResult);
}

TEST_F(CcuTransportTest, Ut_Init_UNAVAIL_When_AppendXn_Return_UNAVAIL)
{
    CcuConnStatus fakeConnStatus = CcuConnStatus::CONN_INVALID;
    MOCKER_CPP(&CcuConnection::GetStatus).stubs().will(returnValue(fakeConnStatus));
    auto transportRes = MockMakeCcuTransport(true, false);
    auto transport = get<0>(transportRes).get();

    HcclResult unavailResult = HcclResult::HCCL_E_UNAVAIL;

    EXPECT_EQ(transport->Init(), unavailResult);
}

TEST_F(CcuTransportTest, Ut_AppendRes_UNAVAIL_When_AppendCkes_Return_UNAVAIL)
{
    CcuConnStatus fakeConnStatus = CcuConnStatus::CONN_INVALID;
    MOCKER_CPP(&CcuConnection::GetStatus).stubs().will(returnValue(fakeConnStatus));
    auto transportRes = MockMakeCcuTransport(false, true);
    auto transport = get<0>(transportRes).get();

    HcclResult unavailResult = HcclResult::HCCL_E_UNAVAIL;

    EXPECT_EQ(transport->AppendRes(1, 1), unavailResult);
}

TEST_F(CcuTransportTest, Ut_AppendRes_UNAVAIL_When_AppendXn_Return_UNAVAIL)
{
    CcuConnStatus fakeConnStatus = CcuConnStatus::CONN_INVALID;
    MOCKER_CPP(&CcuConnection::GetStatus).stubs().will(returnValue(fakeConnStatus));
    auto transportRes = MockMakeCcuTransport(true, false);
    auto transport = get<0>(transportRes).get();

    HcclResult unavailResult = HcclResult::HCCL_E_UNAVAIL;

    EXPECT_EQ(transport->AppendRes(1, 1), unavailResult);
}

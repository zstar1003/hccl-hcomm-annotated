/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define private public
#define protected public

#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>

#include "ccu_connection.h"

#include "hccp_async.h"

#include "port.h"
#include "socket.h"
#include "ccu_jetty.h"
#include "orion_adapter_hccp.h"
#include "orion_adapter_rts.h"
#include "rdma_handle_manager.h"
#include "hccl_common_v2.h"

#undef private
#undef protected

using namespace Hccl;

class CcuConnectionTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CcuConnectionTest tests set up." << std::endl; }

    static void TearDownTestCase()
    {
        GlobalMockObject::verify();
        std::cout << "CcuConnectionTest tests tear down." << std::endl;
    }

    virtual void SetUp()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CcuConnectionTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CcuConnectionTest TearDown" << std::endl;
    }
};

pair<unique_ptr<CcuConnection>, vector<unique_ptr<CcuJetty>>> MockMakeCcuConnection(TpProtocol tpProtocol)
{
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().with(mockcpp::any()).will(returnValue(MAX_MODULE_DEVICE_NUM - 1));

    HcclResult OkResult = HcclResult::HCCL_SUCCESS;
    HcclResult AgainResult = HcclResult::HCCL_E_AGAIN;
    MOCKER(CcuDeviceManager::GetCcuResourceSpaceBufInfo).stubs().will(returnValue(OkResult));
    MOCKER(CcuDeviceManager::GetCcuResourceSpaceTokenInfo).stubs().will(returnValue(OkResult));
    MOCKER(CcuDeviceManager::ConfigChannel).stubs().will(returnValue(OkResult));
    MOCKER_CPP(&CcuJetty::CreateJetty).stubs().will(returnValue(AgainResult)).then(returnValue(OkResult));
    MOCKER_CPP(&TpManager::GetTpInfo).stubs().will(returnValue(AgainResult)).then(returnValue(OkResult));
    MOCKER_CPP(&RdmaHandleManager::GetByIp).stubs().will(returnValue((void*)0x12345678));
    pair<uint32_t, uint32_t> fakeDieFuncPair = make_pair(1, 4);
    MOCKER_CPP(&RdmaHandleManager::GetDieAndFuncId).stubs().will(returnValue(fakeDieFuncPair));
    pair<TokenIdHandle, uint32_t> fakeTokenInfo = make_pair(0x12345678, 1);
    MOCKER_CPP(&RdmaHandleManager::GetTokenIdInfo).stubs().will(returnValue(fakeTokenInfo));

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

    unique_ptr<CcuConnection> connection;
    if (tpProtocol == TpProtocol::CTP) {
        connection = make_unique<CcuCtpConnection>(locAddr, rmtAddr, channelInfo, ccuJettyPtrs);
    } else {
        connection = make_unique<CcuTpConnection>(locAddr, rmtAddr, channelInfo, ccuJettyPtrs);
    }

    EXPECT_EQ(connection->Init(), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(connection->status, CcuConnStatus::INIT);
    return {std::move(connection), std::move(ccuJettys)};
}

TEST_F(CcuConnectionTest, St_GetStatus_When_InterfaceOk_Expect_Return_Ok)
{
    auto resPair = MockMakeCcuConnection(TpProtocol::CTP);
    auto connection = resPair.first.get();

    EXPECT_EQ(connection->GetStatus(), CcuConnStatus::INIT); // 创建jetty还未成功，状态不变
    EXPECT_EQ(connection->GetStatus(), CcuConnStatus::INIT); // 查询tp信息还未成功，状态不变
    EXPECT_EQ(connection->GetStatus(), CcuConnStatus::EXCHANGEABLE);

    std::vector<char> dtoData;
    connection->Serialize(dtoData);
    connection->Deserialize(dtoData);

    EXPECT_EQ(connection->ccuBufAddr, connection->rmtCcuBufAddr);
    EXPECT_EQ(connection->ccuBufTokenId, connection->rmtCcuBufTokenId);
    EXPECT_EQ(connection->ccuBufTokenValue, connection->rmtCcuBufTokenValue);

    EXPECT_EQ(connection->jettyNum, connection->importJettyCtxs.size());
    for (uint32_t i = 0; i < connection->jettyNum; i++) {
        const auto& jetty = connection->ccuJettys_[i];
        const auto& rmtJettyInfo = connection->importJettyCtxs[i].inParam;
        EXPECT_EQ(jetty->GetCreateJettyParam().tokenValue, rmtJettyInfo.tokenValue);

        EXPECT_EQ(jetty->GetJettyedOutParam().keySize, rmtJettyInfo.keyLen);

        EXPECT_EQ(strcmp((const char*)jetty->GetJettyedOutParam().key, (const char*)rmtJettyInfo.key), 0);
    }

    connection->ImportJetty();
    EXPECT_EQ(connection->GetStatus(), CcuConnStatus::CONNECTED);
}

TEST_F(CcuConnectionTest, St_GetStatus_When_TpInterfaceOk_Expect_Return_Ok)
{
    auto resPair = MockMakeCcuConnection(TpProtocol::TP);
    auto connection = resPair.first.get();

    EXPECT_EQ(connection->GetStatus(), CcuConnStatus::INIT); // 创建jetty还未成功，状态不变
    EXPECT_EQ(connection->GetStatus(), CcuConnStatus::INIT); // 查询tp信息还未成功，状态不变
    EXPECT_EQ(connection->GetStatus(), CcuConnStatus::EXCHANGEABLE);

    std::vector<char> dtoData;
    connection->Serialize(dtoData);
    connection->Deserialize(dtoData);

    EXPECT_EQ(connection->ccuBufAddr, connection->rmtCcuBufAddr);
    EXPECT_EQ(connection->ccuBufTokenId, connection->rmtCcuBufTokenId);
    EXPECT_EQ(connection->ccuBufTokenValue, connection->rmtCcuBufTokenValue);

    EXPECT_EQ(connection->jettyNum, connection->importJettyCtxs.size());
    for (uint32_t i = 0; i < connection->jettyNum; i++) {
        const auto& jetty = connection->ccuJettys_[i];
        const auto& rmtJettyInfo = connection->importJettyCtxs[i].inParam;
        EXPECT_EQ(jetty->GetCreateJettyParam().tokenValue, rmtJettyInfo.tokenValue);

        EXPECT_EQ(jetty->GetJettyedOutParam().keySize, rmtJettyInfo.keyLen);

        EXPECT_EQ(strcmp((const char*)jetty->GetJettyedOutParam().key, (const char*)rmtJettyInfo.key), 0);
    }

    connection->ImportJetty();
    EXPECT_EQ(connection->GetStatus(), CcuConnStatus::CONNECTED);
}

TEST_F(CcuConnectionTest, St_GetStatus_When_TpInfoAlreadyHaveAndInterfaceOk_Expect_Return_Ok)
{
    MOCKER_CPP(&TpManager::GetTpInfo).stubs().will(returnValue(HcclResult::HCCL_SUCCESS)); // 模拟首次查询未完成
    auto resPair = MockMakeCcuConnection(TpProtocol::CTP);
    auto connection = resPair.first.get();

    EXPECT_EQ(connection->GetStatus(), CcuConnStatus::INIT); // 创建jetty还未成功，状态不变
    EXPECT_EQ(connection->GetStatus(), CcuConnStatus::EXCHANGEABLE);

    std::vector<char> dtoData;
    connection->Serialize(dtoData);
    connection->Deserialize(dtoData);

    EXPECT_EQ(connection->ccuBufAddr, connection->rmtCcuBufAddr);
    EXPECT_EQ(connection->ccuBufTokenId, connection->rmtCcuBufTokenId);
    EXPECT_EQ(connection->ccuBufTokenValue, connection->rmtCcuBufTokenValue);

    EXPECT_EQ(connection->jettyNum, connection->importJettyCtxs.size());
    for (uint32_t i = 0; i < connection->jettyNum; i++) {
        const auto& jetty = connection->ccuJettys_[i];
        const auto& rmtJettyInfo = connection->importJettyCtxs[i].inParam;
        EXPECT_EQ(jetty->GetCreateJettyParam().tokenValue, rmtJettyInfo.tokenValue);

        EXPECT_EQ(jetty->GetJettyedOutParam().keySize, rmtJettyInfo.keyLen);

        EXPECT_EQ(strcmp((const char*)jetty->GetJettyedOutParam().key, (const char*)rmtJettyInfo.key), 0);
    }

    connection->ImportJetty();
    EXPECT_EQ(connection->GetStatus(), CcuConnStatus::CONNECTED);

    EXPECT_EQ(connection->CreateJetty(), true); // 重复create，状态不应改变
    EXPECT_EQ(connection->GetStatus(), CcuConnStatus::CONNECTED);

    connection->ImportJetty(); // 重复import，状态不应改变
    EXPECT_EQ(connection->GetStatus(), CcuConnStatus::CONNECTED);
}

TEST_F(CcuConnectionTest, St_CallInterface_When_StatusUnexpected_Expect_Error)
{
    auto resPair = MockMakeCcuConnection(TpProtocol::CTP);
    auto connection = resPair.first.get();
    connection->tpProtocol = TpProtocol::INVALID;

    EXPECT_THROW(connection->GetTpInfo(), InternalException);
    std::vector<char> dtoData;
    EXPECT_THROW(connection->Serialize(dtoData), InternalException);
    EXPECT_THROW(connection->Deserialize(dtoData), InternalException);
    EXPECT_THROW(connection->ImportJetty(), InternalException);

    connection->innerStatus = CcuConnection::InnerStatus::EXCHANGEABLE;
    EXPECT_THROW(connection->ImportJetty(), InternalException);
    EXPECT_THROW(connection->ConfigChannel(), InternalException);

    connection->importJettyCtxs.resize(connection->jettyNum);
    EXPECT_THROW(connection->ImportJetty(), InternalException);
}

TEST_F(CcuConnectionTest, St_GetStatus_When_CreateJettyFailed_Expect_Return_StatusInvalid)
{
    // 打桩放在其他mock之前防止被覆盖
    MOCKER_CPP(&CcuJetty::CreateJetty).stubs().will(returnValue(HcclResult::HCCL_E_INTERNAL));
    auto resPair = MockMakeCcuConnection(TpProtocol::CTP);
    auto connection = resPair.first.get();

    EXPECT_EQ(connection->GetStatus(), CcuConnStatus::CONN_INVALID); // 创建jetty还未成功，状态不变
}

TEST_F(CcuConnectionTest, St_GetStatus_When_ImportJettyFailed_Expect_Return_StatusInvalid)
{
    MOCKER_CPP(&CcuConnection::StartImportJettyRequest).stubs().will(returnValue(HcclResult::HCCL_E_INTERNAL));
    auto resPair = MockMakeCcuConnection(TpProtocol::CTP);
    auto connection = resPair.first.get();

    EXPECT_EQ(connection->GetStatus(), CcuConnStatus::INIT); // 创建jetty还未成功，状态不变
    EXPECT_EQ(connection->GetStatus(), CcuConnStatus::INIT); // 查询tp信息还未成功，状态不变
    EXPECT_EQ(connection->GetStatus(), CcuConnStatus::EXCHANGEABLE);

    std::vector<char> dtoData;
    connection->Serialize(dtoData);
    connection->Deserialize(dtoData);

    EXPECT_EQ(connection->ccuBufAddr, connection->rmtCcuBufAddr);
    EXPECT_EQ(connection->ccuBufTokenId, connection->rmtCcuBufTokenId);
    EXPECT_EQ(connection->ccuBufTokenValue, connection->rmtCcuBufTokenValue);

    EXPECT_EQ(connection->jettyNum, connection->importJettyCtxs.size());
    for (uint32_t i = 0; i < connection->jettyNum; i++) {
        const auto& jetty = connection->ccuJettys_[i];
        const auto& rmtJettyInfo = connection->importJettyCtxs[i].inParam;
        EXPECT_EQ(jetty->GetCreateJettyParam().tokenValue, rmtJettyInfo.tokenValue);

        EXPECT_EQ(jetty->GetJettyedOutParam().keySize, rmtJettyInfo.keyLen);

        EXPECT_EQ(strcmp((const char*)jetty->GetJettyedOutParam().key, (const char*)rmtJettyInfo.key), 0);
    }

    EXPECT_THROW(connection->ImportJetty(), InternalException);
    EXPECT_EQ(connection->GetStatus(), CcuConnStatus::CONN_INVALID);
}

TEST_F(CcuConnectionTest, St_GetStatus_When_ConfigChannelFailed_Expect_Return_StatusInvalid)
{
    MOCKER(CcuDeviceManager::ConfigChannel).stubs().will(returnValue(HcclResult::HCCL_E_INTERNAL));
    auto resPair = MockMakeCcuConnection(TpProtocol::CTP);
    auto connection = resPair.first.get();

    EXPECT_EQ(connection->GetStatus(), CcuConnStatus::INIT); // 创建jetty还未成功，状态不变
    EXPECT_EQ(connection->GetStatus(), CcuConnStatus::INIT); // 查询tp信息还未成功，状态不变
    EXPECT_EQ(connection->GetStatus(), CcuConnStatus::EXCHANGEABLE);

    std::vector<char> dtoData;
    connection->Serialize(dtoData);
    connection->Deserialize(dtoData);

    EXPECT_EQ(connection->ccuBufAddr, connection->rmtCcuBufAddr);
    EXPECT_EQ(connection->ccuBufTokenId, connection->rmtCcuBufTokenId);
    EXPECT_EQ(connection->ccuBufTokenValue, connection->rmtCcuBufTokenValue);

    EXPECT_EQ(connection->jettyNum, connection->importJettyCtxs.size());
    for (uint32_t i = 0; i < connection->jettyNum; i++) {
        const auto& jetty = connection->ccuJettys_[i];
        const auto& rmtJettyInfo = connection->importJettyCtxs[i].inParam;
        EXPECT_EQ(jetty->GetCreateJettyParam().tokenValue, rmtJettyInfo.tokenValue);

        EXPECT_EQ(jetty->GetJettyedOutParam().keySize, rmtJettyInfo.keyLen);

        EXPECT_EQ(strcmp((const char*)jetty->GetJettyedOutParam().key, (const char*)rmtJettyInfo.key), 0);
    }

    connection->ImportJetty();
    EXPECT_EQ(connection->GetStatus(), CcuConnStatus::CONN_INVALID);
}

TEST_F(CcuConnectionTest, Ut_Clean_When_InterfaceOk_Expect_Return_Ok)
{
    auto resPair = MockMakeCcuConnection(TpProtocol::CTP);
    auto connection = resPair.first.get();

    EXPECT_EQ(connection->GetStatus(), CcuConnStatus::INIT); // 创建jetty还未成功，状态不变
    EXPECT_EQ(connection->GetStatus(), CcuConnStatus::INIT); // 查询tp信息还未成功，状态不变
    EXPECT_EQ(connection->GetStatus(), CcuConnStatus::EXCHANGEABLE);

    std::vector<char> dtoData;
    connection->Serialize(dtoData);
    connection->Deserialize(dtoData);

    EXPECT_EQ(connection->ccuBufAddr, connection->rmtCcuBufAddr);
    EXPECT_EQ(connection->ccuBufTokenId, connection->rmtCcuBufTokenId);
    EXPECT_EQ(connection->ccuBufTokenValue, connection->rmtCcuBufTokenValue);

    EXPECT_EQ(connection->jettyNum, connection->importJettyCtxs.size());
    for (uint32_t i = 0; i < connection->jettyNum; i++) {
        const auto& jetty = connection->ccuJettys_[i];
        const auto& rmtJettyInfo = connection->importJettyCtxs[i].inParam;
        EXPECT_EQ(jetty->GetCreateJettyParam().tokenValue, rmtJettyInfo.tokenValue);

        EXPECT_EQ(jetty->GetJettyedOutParam().keySize, rmtJettyInfo.keyLen);

        EXPECT_EQ(strcmp((const char*)jetty->GetJettyedOutParam().key, (const char*)rmtJettyInfo.key), 0);
    }

    connection->ImportJetty();
    EXPECT_EQ(connection->GetStatus(), CcuConnStatus::CONNECTED);

    EXPECT_NO_THROW(connection->Clean());
}

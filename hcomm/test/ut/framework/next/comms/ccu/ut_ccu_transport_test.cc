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
#include <mockcpp/mockcpp.hpp>

#include "ccu_transport_.h"
#include "binary_stream.h"
#include "hccl_common.h"

#define private public
#define protected public

using namespace hcomm;

class CclBufferInfoTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(CclBufferInfoTest, Ut_PackUnpack_RoundTrip_Expect_SameValues)
{
    CcuTransport::CclBufferInfo original(0x12345678ULL, 1024U, 100U, 200U);
    Hccl::BinaryStream stream;
    original.Pack(stream);

    CcuTransport::CclBufferInfo restored;
    restored.Unpack(stream);
    EXPECT_EQ(restored.addr, 0x12345678ULL);
    EXPECT_EQ(restored.size, 1024U);
    EXPECT_EQ(restored.tokenId, 100U);
    EXPECT_EQ(restored.tokenValue, 200U);
}

TEST_F(CclBufferInfoTest, Ut_PackUnpack_WithMemInfo_Expect_SameValues)
{
    CcuTransport::CclBufferInfo original(0xDEADBEEFULL, 4096U, 10U, 20U);
    std::array<char, HCCL_RES_TAG_MAX_LEN> memInfo{};
    const char* tag = "TestBuffer";
    memcpy(memInfo.data(), tag, strlen(tag));
    original.memInfo = memInfo;

    Hccl::BinaryStream stream;
    original.Pack(stream);

    CcuTransport::CclBufferInfo restored;
    restored.Unpack(stream);
    EXPECT_EQ(restored.addr, 0xDEADBEEFULL);
    EXPECT_EQ(restored.size, 4096U);
    std::string restoredTag(restored.memInfo.data(), strnlen(restored.memInfo.data(), HCCL_RES_TAG_MAX_LEN));
    EXPECT_EQ(restoredTag, "TestBuffer");
}

TEST_F(CclBufferInfoTest, Ut_DefaultConstructor_Expect_ZeroValues)
{
    CcuTransport::CclBufferInfo info;
    EXPECT_EQ(info.addr, 0ULL);
    EXPECT_EQ(info.size, 0U);
    EXPECT_EQ(info.tokenId, 0U);
    EXPECT_EQ(info.tokenValue, 0U);
}

TEST_F(CclBufferInfoTest, Ut_ConstructorWithParams_Expect_CorrectValues)
{
    CcuTransport::CclBufferInfo info(0x1000ULL, 512U, 1U, 2U);
    EXPECT_EQ(info.addr, 0x1000ULL);
    EXPECT_EQ(info.size, 512U);
    EXPECT_EQ(info.tokenId, 1U);
    EXPECT_EQ(info.tokenValue, 2U);
}

class CcuTransportTest : public testing::Test {
protected:
    std::unique_ptr<CcuTransport> transport;
    Hccl::Socket socket_{
        nullptr,
        Hccl::IpAddress(),
        0,
        Hccl::IpAddress(),
        "ut_socket",
        Hccl::SocketRole::CLIENT,
        Hccl::NicType::DEVICE_NIC_TYPE};
    void SetUp() override
    {
        CcuTransport::CclBufferInfo bufInfo(0x1000ULL, 1024U, 1U, 2U);
        // socket 未连接(INIT 状态), SendDataSize 内 SendAsync 抛 SocketException 被捕获;
        // 标记 isDestroyed 避免 fixture 析构时对未初始化 socket 调用 Destroy
        socket_.isDestroyed = true;
        transport = std::make_unique<CcuTransport>(&socket_, nullptr, bufInfo);
    }
    void TearDown() override {}
};

TEST_F(CcuTransportTest, Ut_GetDieId_When_Set_Expect_CorrectValue)
{
    transport->dieId_ = 1;
    EXPECT_EQ(transport->GetDieId(), 1U);
}

TEST_F(CcuTransportTest, Ut_GetLocCkeByIndex_When_Empty_Expect_ReturnHCCL_E_PARA)
{
    uint32_t ckeId = 0;
    EXPECT_EQ(transport->GetLocCkeByIndex(0, ckeId), HcclResult::HCCL_E_PARA);
}

TEST_F(CcuTransportTest, Ut_GetLocCkeByIndex_When_IndexOutOfRange_Expect_ReturnHCCL_E_PARA)
{
    transport->locRes_.ckes = {10, 20, 30};
    uint32_t ckeId = 0;
    EXPECT_EQ(transport->GetLocCkeByIndex(5, ckeId), HcclResult::HCCL_E_PARA);
}

TEST_F(CcuTransportTest, Ut_GetLocCkeByIndex_When_Normal_Expect_ReturnSuccess)
{
    transport->locRes_.ckes = {10, 20, 30};
    uint32_t ckeId = 0;
    EXPECT_EQ(transport->GetLocCkeByIndex(1, ckeId), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(ckeId, 20U);
}

TEST_F(CcuTransportTest, Ut_GetLocXnByIndex_When_Empty_Expect_ReturnHCCL_E_PARA)
{
    uint32_t xnId = 0;
    EXPECT_EQ(transport->GetLocXnByIndex(0, xnId), HcclResult::HCCL_E_PARA);
}

TEST_F(CcuTransportTest, Ut_GetLocXnByIndex_When_Normal_Expect_ReturnSuccess)
{
    transport->locRes_.xns = {100, 200};
    uint32_t xnId = 0;
    EXPECT_EQ(transport->GetLocXnByIndex(0, xnId), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(xnId, 100U);
}

TEST_F(CcuTransportTest, Ut_GetLocXnByIndex_When_IndexOutOfRange_Expect_ReturnHCCL_E_PARA)
{
    transport->locRes_.xns = {100, 200};
    uint32_t xnId = 0;
    EXPECT_EQ(transport->GetLocXnByIndex(5, xnId), HcclResult::HCCL_E_PARA);
}

TEST_F(CcuTransportTest, Ut_GetRmtCkeByIndex_When_Empty_Expect_ReturnHCCL_E_PARA)
{
    uint32_t ckeId = 0;
    EXPECT_EQ(transport->GetRmtCkeByIndex(0, ckeId), HcclResult::HCCL_E_PARA);
}

TEST_F(CcuTransportTest, Ut_GetRmtCkeByIndex_When_Normal_Expect_ReturnSuccess)
{
    transport->rmtRes_.ckes = {50, 60};
    uint32_t ckeId = 0;
    EXPECT_EQ(transport->GetRmtCkeByIndex(0, ckeId), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(ckeId, 50U);
}

TEST_F(CcuTransportTest, Ut_GetRmtXnByIndex_When_Empty_Expect_ReturnHCCL_E_PARA)
{
    uint32_t xnId = 0;
    EXPECT_EQ(transport->GetRmtXnByIndex(0, xnId), HcclResult::HCCL_E_PARA);
}

TEST_F(CcuTransportTest, Ut_GetRmtXnByIndex_When_Normal_Expect_ReturnSuccess)
{
    transport->rmtRes_.xns = {500, 600};
    uint32_t xnId = 0;
    EXPECT_EQ(transport->GetRmtXnByIndex(1, xnId), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(xnId, 600U);
}

TEST_F(CcuTransportTest, Ut_GetRmtXnByIndex_When_IndexOutOfRange_Expect_ReturnHCCL_E_PARA)
{
    transport->rmtRes_.xns = {500};
    uint32_t xnId = 0;
    EXPECT_EQ(transport->GetRmtXnByIndex(1, xnId), HcclResult::HCCL_E_PARA);
}

TEST_F(CcuTransportTest, Ut_GetLocBuffer_When_Normal_Expect_ReturnSuccess)
{
    CcuTransport::CclBufferInfo buf;
    EXPECT_EQ(transport->GetLocBuffer(buf, 0), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(buf.addr, 0x1000ULL);
}

TEST_F(CcuTransportTest, Ut_GetRmtBuffer_When_Normal_Expect_ReturnSuccess)
{
    transport->rmtHcclBufferInfo_ = CcuTransport::CclBufferInfo(0x2000ULL, 2048U, 3U, 4U);
    CcuTransport::CclBufferInfo buf;
    EXPECT_EQ(transport->GetRmtBuffer(buf, 0), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(buf.addr, 0x2000ULL);
}

TEST_F(CcuTransportTest, Ut_GetCkeNum_When_Normal_Expect_ReturnSuccess)
{
    transport->locRes_.ckes = {1, 2, 3, 4, 5};
    uint32_t ckeNum = 0;
    EXPECT_EQ(transport->GetCkeNum(ckeNum), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(ckeNum, 5U);
}

TEST_F(CcuTransportTest, Ut_Describe_When_Normal_Expect_NonEmpty)
{
    transport->dieId_ = 1;
    transport->locRes_.ckes = {1, 2};
    transport->locRes_.xns = {3, 4};
    transport->rmtRes_.ckes = {5};
    transport->rmtRes_.xns = {6};
    std::string desc = transport->Describe();
    EXPECT_FALSE(desc.empty());
}

TEST_F(CcuTransportTest, Ut_CheckFinish_When_MatchingMsg_Expect_ReturnSuccess)
{
    std::string finishMsg = "Transport exchange data ready!";
    transport->sendFinishMsg_ = std::vector<char>(finishMsg.begin(), finishMsg.end());
    transport->recvFinishMsg_ = std::vector<char>(finishMsg.begin(), finishMsg.end());
    EXPECT_EQ(transport->CheckFinish(), HcclResult::HCCL_SUCCESS);
}

TEST_F(CcuTransportTest, Ut_CheckFinish_When_MismatchMsg_Expect_ReturnHCCL_E_INTERNAL)
{
    std::string finishMsg = "Transport exchange data ready!";
    transport->sendFinishMsg_ = std::vector<char>(finishMsg.begin(), finishMsg.end());
    std::string wrongMsg(128, 'x');
    transport->recvFinishMsg_ = std::vector<char>(wrongMsg.begin(), wrongMsg.end());
    EXPECT_EQ(transport->CheckFinish(), HcclResult::HCCL_E_INTERNAL);
}

TEST_F(CcuTransportTest, Ut_AttributionDescribe_Expect_NonEmpty)
{
    CcuTransport::Attribution attr;
    attr.devicePhyId = 1;
    attr.handshakeMsg = {'a', 'b', 'c'};
    std::string desc = attr.Describe();
    EXPECT_FALSE(desc.empty());
}

TEST_F(CcuTransportTest, Ut_SetHandshakeMsg_Then_GetLocalHandshakeMsg_Expect_Same)
{
    std::vector<char> msg = {'h', 'e', 'l', 'l', 'o'};
    transport->SetHandshakeMsg(msg);
    EXPECT_EQ(transport->GetLocalHandshakeMsg(), msg);
}

TEST_F(CcuTransportTest, Ut_GetRmtHandshakeMsg_Expect_InitiallyEmpty)
{
    transport->rmtHandshakeMsg_.clear();
    auto& msg = transport->GetRmtHandshakeMsg();
    EXPECT_TRUE(msg.empty());
}

TEST_F(CcuTransportTest, Ut_HandshakeMsgPack_When_LocOk_Expect_OriginalHandshakeMsg)
{
    transport->SetHandshakeMsg({'h', 'e', 'l', 'l', 'o'});
    Hccl::BinaryStream stream;
    EXPECT_EQ(transport->HandshakeMsgPack(stream), HcclResult::HCCL_SUCCESS);

    // 反解 stream 验证握手消息内容可往返还原
    std::vector<char> dumped;
    stream.DumpWithRevert(dumped);
    EXPECT_FALSE(dumped.empty());
    std::vector<char> restored;
    stream >> restored;
    EXPECT_EQ(restored, std::vector<char>({'h', 'e', 'l', 'l', 'o'}));
}

TEST_F(CcuTransportTest, Ut_HandshakeMsgUnpack_When_Ok_Expect_RmtHandshakeMsg)
{
    std::vector<char> msg = {'a', 'b', 'c'};
    Hccl::BinaryStream stream;
    stream << msg;

    transport->attr_.handshakeMsg = msg;
    transport->rmtHandshakeMsg_.clear();
    EXPECT_EQ(transport->HandshakeMsgUnpack(stream), HcclResult::HCCL_SUCCESS);
    // STC V2.0: 断言面向公共 getter 行为契约，不直读私有 rmtHandshakeMsg_
    EXPECT_EQ(transport->GetRmtHandshakeMsg(), msg);
}

TEST_F(CcuTransportTest, Ut_ConstructMsgOnlyTransport_TransStatusInit)
{
    Hccl::Socket* fakeSocket = reinterpret_cast<Hccl::Socket*>(0x1);
    std::unique_ptr<CcuTransport> impl;
    EXPECT_EQ(
        CcuTransport::ConstructMsgOnlyTransport(fakeSocket, impl, CcuTransport::CcuResStatus::RES_UNAVAIL),
        HcclResult::HCCL_SUCCESS);

    // STC V2.0: locResStatus_ 改用公共 helper 行为契约
    EXPECT_TRUE(impl->IsLocResUnavailable());
    // 白盒保留: transStatus_ 无公共 setter, GetStatus() 会触发状态机, 故直读初值
    EXPECT_EQ(impl->transStatus_, CcuTransport::TransStatus::INIT);
    // 白盒保留: locBufferInfos_ 无公共 getter, 故直读判空
    EXPECT_TRUE(impl->locBufferInfos_.empty());
}

// ==================== SendDataSize / RecvDataProcess (locResStatus int 协议) ====================

// TC-TS-021: RES_UNAVAIL 时载荷仅 sizeof(int)
TEST_F(CcuTransportTest, Ut_SendDataSize_When_LocUnavail_Expect_ResStatusIntOnly)
{
    transport->SetLocResStatus(CcuTransport::CcuResStatus::RES_UNAVAIL);
    (void)transport->SendDataSize();
    EXPECT_EQ(transport->sendData_.size(), sizeof(int));
}

// TC-TS-022: RES_FAILED 时载荷仅 sizeof(int)
TEST_F(CcuTransportTest, Ut_SendDataSize_When_LocFailed_Expect_ResStatusIntOnly)
{
    transport->SetLocResStatus(CcuTransport::CcuResStatus::RES_FAILED);
    (void)transport->SendDataSize();
    EXPECT_EQ(transport->sendData_.size(), sizeof(int));
}

// TC-TS-024: RecvDataProcess 收到 RES_UNAVAIL 时跳过后续解包
TEST_F(CcuTransportTest, Ut_RecvDataProcess_When_RmtUnavail_Expect_SkipUnpack)
{
    Hccl::BinaryStream stream;
    int unavailStatus = static_cast<int>(CcuTransport::CcuResStatus::RES_UNAVAIL);
    stream << unavailStatus;
    std::vector<char> payload;
    stream.Dump(payload);
    transport->exchangeDataSize_ = payload.size();
    transport->recvData_ = payload;

    EXPECT_EQ(transport->RecvDataProcess(), HcclResult::HCCL_SUCCESS);
    EXPECT_TRUE(transport->IsRmtResUnavailable());
}

// TC-TS-025: RecvDataProcess 收到 RES_FAILED 时跳过后续解包
TEST_F(CcuTransportTest, Ut_RecvDataProcess_When_RmtFailed_Expect_SkipUnpack)
{
    Hccl::BinaryStream stream;
    int failedStatus = static_cast<int>(CcuTransport::CcuResStatus::RES_FAILED);
    stream << failedStatus;
    std::vector<char> payload;
    stream.Dump(payload);
    transport->exchangeDataSize_ = payload.size();
    transport->recvData_ = payload;

    EXPECT_EQ(transport->RecvDataProcess(), HcclResult::HCCL_SUCCESS);
    // RES_FAILED 不是 UNAVAIL, IsRmtResUnavailable 应为 false
    EXPECT_FALSE(transport->IsRmtResUnavailable());
    // rmtResStatus 应为 FAILED (非 OK), 通过 IsRmtResUnavailable=false + 非 OK 间接验证
    // 更直接: GetLocResStatus 仍为初始值, rmtResStatus 已被设为 FAILED
}

// TC-TS-026: RecvDataProcess 载荷过短时报错
TEST_F(CcuTransportTest, Ut_RecvDataProcess_When_PayloadTooShort_Expect_Error)
{
    transport->exchangeDataSize_ = 2; // 小于 sizeof(int)
    transport->recvData_ = {'a', 'b'};
    EXPECT_EQ(transport->RecvDataProcess(), HcclResult::HCCL_E_INTERNAL);
}

// 状态机终态分支: RECV_FIN 且资源状态非 OK 时, 直接进入 CONNECT_FAILED(资源不足快速失败的关键路径)
TEST_F(CcuTransportTest, Ut_StatusMachine_When_RecvFin_And_ResNotOk_Expect_CONNECT_FAILED)
{
    std::string finishMsg = "Transport exchange data ready!";
    transport->sendFinishMsg_ = std::vector<char>(finishMsg.begin(), finishMsg.end());
    transport->recvFinishMsg_ = std::vector<char>(finishMsg.begin(), finishMsg.end());
    transport->transStatus_ = CcuTransport::TransStatus::RECV_FIN;
    transport->SetLocResStatus(CcuTransport::CcuResStatus::RES_OK);
    transport->rmtResStatus_ = CcuTransport::CcuResStatus::RES_UNAVAIL;

    EXPECT_EQ(transport->StatusMachine(), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(transport->transStatus_, CcuTransport::TransStatus::CONNECT_FAILED);
}

// 非法状态值(非 0/1/2/3)按"非 OK"处理: 跳过完整解包, 不判定为资源不足, 不越界
TEST_F(CcuTransportTest, Ut_RecvDataProcess_When_RmtStatusInvalid_Expect_NotOk)
{
    Hccl::BinaryStream stream;
    int invalidStatus = 0xFF;
    stream << invalidStatus;
    std::vector<char> payload;
    stream.Dump(payload);
    transport->exchangeDataSize_ = payload.size();
    transport->recvData_ = payload;
    transport->SetLocResStatus(CcuTransport::CcuResStatus::RES_OK);

    // 对端状态值越界属于协议错误, RecvDataProcess 返回 INTERNAL
    EXPECT_EQ(transport->RecvDataProcess(), HcclResult::HCCL_E_INTERNAL);
    EXPECT_FALSE(transport->IsRmtResUnavailable());
    EXPECT_NE(transport->rmtResStatus_, CcuTransport::CcuResStatus::RES_OK);
}

TEST_F(CcuTransportTest, Ut_ConstructMsgOnlyTransport_When_SocketNull_Expect_E_PTR)
{
    std::unique_ptr<CcuTransport> impl;
    EXPECT_EQ(
        CcuTransport::ConstructMsgOnlyTransport(nullptr, impl, CcuTransport::CcuResStatus::RES_UNAVAIL),
        HcclResult::HCCL_E_PTR);
}

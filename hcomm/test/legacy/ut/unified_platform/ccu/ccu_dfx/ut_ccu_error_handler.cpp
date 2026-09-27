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
#include <mockcpp/mockcpp.hpp>
#include <mockcpp/mokc.h>
#define private public
#define protected public
#include "orion_adapter_hccp.h"
#include "orion_adapter_rts.h"
#include "rdma_handle_manager.h"
#include "ip_address.h"
#include "ccu_api_exception.h"
#include "ccu_ctx.h"

#include "ccu_datatype.h"
#include "ccu_rep_type.h"
#include "ccu_device_manager.h"
#include "ccu_rep_locpostsem.h"
#include "ccu_rep_locwaitsem.h"
#include "ccu_rep_rempostsem.h"
#include "ccu_rep_remwaitsem.h"
#include "ccu_rep_rempostvar.h"
#include "ccu_rep_remwaitgroup.h"
#include "ccu_rep_postsharedvar.h"
#include "ccu_rep_postsharedsem.h"
#include "ccu_rep_read.h"
#include "ccu_rep_write.h"
#include "ccu_rep_loccpy.h"
#include "ccu_rep_bufread.h"
#include "ccu_rep_bufwrite.h"
#include "ccu_rep_buflocread.h"
#include "ccu_rep_buflocwrite.h"
#include "ccu_rep_bufreduce.h"
#include "ccu_rep_jump.h"
#include "ccu_rep_loop.h"
#include "ccu_rep_loopblock.h"
#include "ccu_rep_loopgroup.h"
#undef protected
#undef private

#define private public
#define protected public
#include "ccu_rep_base.h"
#include "ccu_error_handler.h"
#include "ccu_connection.h"
#include "ccu_transport.h"
#include "ccu_transport_group.h"
#undef private
#undef protected

using namespace std;
using namespace Hccl;
using namespace CcuRep;

class CcuErrorHandlerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CcuErrorHandlerTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "CcuErrorHandlerTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in CcuErrorHandlerTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CcuErrorHandlerTest TearDown" << std::endl;
    }

    unique_ptr<CcuConnection> MockCcuConnection(uint32_t channelId)
    {
        // Mock CcuConnection: 只需能够返回channelId即可

        // 打桩CcuConnection构造函数中所需的调用
        JfcHandle jfcHandle = 0;
        MOCKER_CPP(&RdmaHandleManager::GetJfcHandle)
            .stubs()
            .with(mockcpp::any(), mockcpp::any())
            .will(returnValue(jfcHandle));
        MOCKER(HrtGetDeviceType).stubs().will(returnValue(Hccl::DevType::DEV_TYPE_910A));
        MOCKER_CPP(&RdmaHandleManager::GetDieAndFuncId)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(std::pair<uint32_t, uint32_t>(0, 0)));
        BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
        LinkData linkData(portType, 0, 1, 0, 1);
        ChannelInfo channelInfo{};
        channelInfo.channelId = channelId;
        vector<CcuJetty*> ccuJettys;
        auto utConnection
            = make_unique<CcuConnection>(linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);

        utConnection->channelInfo_ = channelInfo;
        return utConnection;
    }
};

TEST_F(CcuErrorHandlerTest, test_mock_ccu_connection)
{
    auto utCcuConnection = MockCcuConnection(7);
    CcuTransport::CclBufferInfo locCclBufInfo;
    shared_ptr<CcuTransport> utCcuTransport
        = make_shared<CcuTransport>(nullptr, std::move(utCcuConnection), locCclBufInfo);
    EXPECT_EQ(utCcuTransport->GetChannelId(), 7);
}

TEST_F(CcuErrorHandlerTest, test_gen_status_info)
{
    ErrorInfoBase baseInfo{0, 1, 2, 10, 0};
    vector<CcuErrorInfo> errorInfo{};

    baseInfo.status = 0x0100;
    CcuErrorHandler::GenStatusInfo(baseInfo, "", errorInfo);
    EXPECT_EQ(errorInfo.size(), 1);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::MISSION);
    EXPECT_EQ(errorInfo[0].dieId, 1);
    EXPECT_EQ(errorInfo[0].missionId, 2);
    EXPECT_EQ(errorInfo[0].instrId, 10);
    EXPECT_EQ(string(errorInfo[0].msg.mission.missionError), "Unsupported Opcode(0x01)");

    errorInfo.clear();
    baseInfo.status = 0x0400;
    CcuErrorHandler::GenStatusInfo(baseInfo, "", errorInfo);
    EXPECT_EQ(string(errorInfo[0].msg.mission.missionError), "Transaction Retry Counter Exceeded(0x04)");

    errorInfo.clear();
    baseInfo.status = 0x0203;
    CcuErrorHandler::GenStatusInfo(baseInfo, "", errorInfo);
    EXPECT_EQ(
        string(errorInfo[0].msg.mission.missionError),
        "Local Operation Error(0x02), Remote Response Length Error(0x03)");

    errorInfo.clear();
    baseInfo.status = 0x0301;
    CcuErrorHandler::GenStatusInfo(baseInfo, "", errorInfo);
    EXPECT_EQ(
        string(errorInfo[0].msg.mission.missionError),
        "Remote Operation Error(0x03), Remote Unsupported Request(0x01)");

    errorInfo.clear();
    baseInfo.status = 0x0901;
    CcuErrorHandler::GenStatusInfo(baseInfo, "", errorInfo);
    EXPECT_EQ(
        string(errorInfo[0].msg.mission.missionError), "CCUM Execute Error(0x09), SQE instr and key not match(0x01)");

    errorInfo.clear();
    baseInfo.status = 0x0902;
    CcuErrorHandler::GenStatusInfo(baseInfo, "", errorInfo);
    EXPECT_EQ(string(errorInfo[0].msg.mission.missionError), "CCUM Execute Error(0x09), CCU Mission Task Killed(0x02)");

    errorInfo.clear();
    baseInfo.status = 0x0A07;
    CcuErrorHandler::GenStatusInfo(baseInfo, "", errorInfo);
    EXPECT_EQ(string(errorInfo[0].msg.mission.missionError), "CCUA Execute Error(0x0A), Atomic Permission Err(0x07)");

    errorInfo.clear();
    baseInfo.status = 0x0000;
    CcuErrorHandler::GenStatusInfo(baseInfo, "", errorInfo);
    EXPECT_EQ(string(errorInfo[0].msg.mission.missionError), "Unknown Status");

    errorInfo.clear();
    baseInfo.status = 0x02ff;
    CcuErrorHandler::GenStatusInfo(baseInfo, "", errorInfo);
    EXPECT_EQ(string(errorInfo[0].msg.mission.missionError), "Local Operation Error(0x02), Unknown Status");
}

TEST_F(CcuErrorHandlerTest, test_error_info_when_rep_type_is_loc_post_sem)
{
    MaskSignal sem;
    sem.Reset(1);           // sem id
    uint16_t mask = 0x0010; // mask
    MOCKER(CcuErrorHandler::GetCcuCKEValue)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(static_cast<u16>(0xabcd)));

    shared_ptr<CcuRepBase> rep = make_shared<CcuRepLocPostSem>(sem, mask);
    ErrorInfoBase baseInfo{0, 0, 1, 10, 0};
    vector<CcuErrorInfo> errorInfo{};
    CcuErrorHandler::GenErrorInfoByRepType(baseInfo, rep, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::WAIT_SIGNAL);
    EXPECT_EQ(errorInfo[0].repType, CcuRepType::LOC_POST_SEM);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.signalId, 1);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.signalValue, 0xabcd);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.signalMask, mask);
}

TEST_F(CcuErrorHandlerTest, test_error_info_when_rep_type_is_loc_wait_sem)
{
    MaskSignal sem;
    sem.Reset(1);           // sem id
    uint16_t mask = 0x0010; // mask
    MOCKER(CcuErrorHandler::GetCcuCKEValue)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(static_cast<u16>(0xabcd)));

    shared_ptr<CcuRepBase> rep = make_shared<CcuRepLocWaitSem>(sem, mask);
    ErrorInfoBase baseInfo{0, 0, 1, 10, 0};
    vector<CcuErrorInfo> errorInfo{};
    CcuErrorHandler::GenErrorInfoByRepType(baseInfo, rep, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::WAIT_SIGNAL);
    EXPECT_EQ(errorInfo[0].repType, CcuRepType::LOC_WAIT_SEM);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.signalId, 1);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.signalValue, 0xabcd);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.signalMask, mask);
}

TEST_F(CcuErrorHandlerTest, test_error_info_when_rep_type_is_rem_post_sem)
{
    uint16_t semIndex = 1;  // semIndex
    uint16_t mask = 0x0010; // mask

    auto utCcuConnection = MockCcuConnection(7); // channelId
    CcuTransport::CclBufferInfo locCclBufInfo;
    shared_ptr<CcuTransport> utCcuTransport
        = make_shared<CcuTransport>(nullptr, std::move(utCcuConnection), locCclBufInfo);
    utCcuTransport->rmtRes.cntCkes.push_back(100);
    utCcuTransport->rmtRes.cntCkes.push_back(101);

    shared_ptr<CcuRepBase> rep = make_shared<CcuRepRemPostSem>(*utCcuTransport, semIndex, mask, false);
    ErrorInfoBase baseInfo{0, 0, 1, 10, 0};
    vector<CcuErrorInfo> errorInfo{};
    CcuErrorHandler::GenErrorInfoByRepType(baseInfo, rep, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::WAIT_SIGNAL);
    EXPECT_EQ(errorInfo[0].repType, CcuRepType::REM_POST_SEM);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.signalId, 101);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.signalMask, mask);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.channelId[0], 7);
    for (uint32_t i = 1; i < WAIT_SIGNAL_CHANNEL_SIZE; ++i) {
        EXPECT_EQ(errorInfo[0].msg.waitSignal.channelId[i], 0xffff);
    }
}

TEST_F(CcuErrorHandlerTest, test_error_info_when_rep_type_is_rem_wait_sem)
{
    uint16_t semIndex = 1;  // semIndex
    uint16_t mask = 0x0010; // mask

    auto utCcuConnection = MockCcuConnection(7); // channelId
    CcuTransport::CclBufferInfo locCclBufInfo;
    shared_ptr<CcuTransport> utCcuTransport
        = make_shared<CcuTransport>(nullptr, std::move(utCcuConnection), locCclBufInfo);
    utCcuTransport->locRes.cntCkes.push_back(100);
    utCcuTransport->locRes.cntCkes.push_back(101);

    MOCKER(CcuErrorHandler::GetCcuCKEValue)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), eq(101))
        .will(returnValue(static_cast<u16>(0xabcd)));

    shared_ptr<CcuRepBase> rep = make_shared<CcuRepRemWaitSem>(*utCcuTransport, semIndex, mask);
    ErrorInfoBase baseInfo{0, 0, 1, 10, 0};
    vector<CcuErrorInfo> errorInfo{};
    CcuErrorHandler::GenErrorInfoByRepType(baseInfo, rep, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::WAIT_SIGNAL);
    EXPECT_EQ(errorInfo[0].repType, CcuRepType::REM_WAIT_SEM);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.signalId, 101);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.signalValue, 0xabcd);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.signalMask, mask);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.channelId[0], 7);
    for (uint32_t i = 1; i < WAIT_SIGNAL_CHANNEL_SIZE; ++i) {
        EXPECT_EQ(errorInfo[0].msg.waitSignal.channelId[i], 0xffff);
    }
}

TEST_F(CcuErrorHandlerTest, test_error_info_when_rep_type_is_rem_post_var)
{
    uint16_t semIndex = 0;  // semIndex
    uint16_t mask = 0x0010; // mask

    uint16_t paramIndex = 0;
    Variable param;
    param.Reset(1); // param id
    MOCKER(CcuErrorHandler::GetCcuXnValue).stubs().with(mockcpp::any(), mockcpp::any(), eq(1)).will(returnValue(0xa));

    auto utCcuConnection = MockCcuConnection(7); // channelId
    CcuTransport::CclBufferInfo locCclBufInfo;
    shared_ptr<CcuTransport> utCcuTransport
        = make_shared<CcuTransport>(nullptr, std::move(utCcuConnection), locCclBufInfo);
    utCcuTransport->rmtRes.cntCkes.push_back(100);
    utCcuTransport->rmtRes.xns.push_back(101);

    shared_ptr<CcuRepBase> rep = make_shared<CcuRepRemPostVar>(param, *utCcuTransport, paramIndex, semIndex, mask);
    ErrorInfoBase baseInfo{0, 0, 1, 10, 0};
    vector<CcuErrorInfo> errorInfo{};
    CcuErrorHandler::GenErrorInfoByRepType(baseInfo, rep, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::WAIT_SIGNAL);
    EXPECT_EQ(errorInfo[0].repType, CcuRepType::REM_POST_VAR);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.signalId, 100);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.signalMask, mask);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.paramId, 101);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.paramValue, 0xa);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.channelId[0], 7);
    for (uint32_t i = 1; i < WAIT_SIGNAL_CHANNEL_SIZE; ++i) {
        EXPECT_EQ(errorInfo[0].msg.waitSignal.channelId[i], 0xffff);
    }
}

TEST_F(CcuErrorHandlerTest, test_error_info_when_rep_type_is_rem_wait_group)
{
    uint16_t semIndex = 0;  // semIndex
    uint16_t mask = 0x0010; // mask

    CcuTransport::CclBufferInfo locCclBufInfo;
    auto utCcuConnection1 = MockCcuConnection(1); // channelId
    shared_ptr<CcuTransport> utCcuTransport1
        = make_shared<CcuTransport>(nullptr, std::move(utCcuConnection1), locCclBufInfo);
    auto utCcuConnection2 = MockCcuConnection(2); // channelId
    shared_ptr<CcuTransport> utCcuTransport2
        = make_shared<CcuTransport>(nullptr, std::move(utCcuConnection2), locCclBufInfo);
    auto utCcuConnection3 = MockCcuConnection(3); // channelId
    shared_ptr<CcuTransport> utCcuTransport3
        = make_shared<CcuTransport>(nullptr, std::move(utCcuConnection3), locCclBufInfo);

    vector<CcuTransport*> transports{utCcuTransport1.get(), utCcuTransport2.get(), utCcuTransport3.get()};
    // 打桩CcuTransportGroup构造函数与析构函数的调用
    MOCKER_CPP(&CcuTransportGroup::CheckTransports).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER_CPP(&CcuTransportGroup::CheckTransportCntCke)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(CcuDeviceManager::ReleaseCke).defaults().will(returnValue(HcclResult::HCCL_SUCCESS));
    CcuTransportGroup transportGroup{transports, 0}; // 创建CcuTransportGroup
    transportGroup.cntCkesGroup.push_back(100);

    MOCKER(CcuErrorHandler::GetCcuCKEValue)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), eq(100))
        .will(returnValue(static_cast<u16>(0xabcd)));

    shared_ptr<CcuRepBase> rep = make_shared<CcuRepWaitGroup>(transportGroup, semIndex, mask);
    ErrorInfoBase baseInfo{0, 0, 1, 10, 0};
    vector<CcuErrorInfo> errorInfo{};
    CcuErrorHandler::GenErrorInfoByRepType(baseInfo, rep, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::WAIT_SIGNAL);
    EXPECT_EQ(errorInfo[0].repType, CcuRepType::REM_WAIT_GROUP);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.signalId, 100);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.signalValue, 0xabcd);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.signalMask, mask);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.channelId[0], 1);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.channelId[1], 2);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.channelId[2], 3);
    for (uint32_t i = 3; i < WAIT_SIGNAL_CHANNEL_SIZE; ++i) {
        EXPECT_EQ(errorInfo[0].msg.waitSignal.channelId[i], 0xffff);
    }
}

TEST_F(CcuErrorHandlerTest, test_error_info_when_rep_type_is_post_shared_var)
{
    MaskSignal sem;
    sem.Reset(1);           // sem id
    uint16_t mask = 0x0010; // mask

    Variable srcVar;
    srcVar.Reset(2); // srcVar id
    MOCKER(CcuErrorHandler::GetCcuXnValue).stubs().with(mockcpp::any(), mockcpp::any(), eq(2)).will(returnValue(0xb));

    Variable dstVar;
    dstVar.Reset(3); // dstVar id

    shared_ptr<CcuRepBase> rep = make_shared<CcuRepPostSharedVar>(srcVar, dstVar, sem, mask);
    ErrorInfoBase baseInfo{0, 0, 1, 10, 0};
    vector<CcuErrorInfo> errorInfo{};
    CcuErrorHandler::GenErrorInfoByRepType(baseInfo, rep, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::WAIT_SIGNAL);
    EXPECT_EQ(errorInfo[0].repType, CcuRepType::POST_SHARED_VAR);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.signalId, 1);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.signalMask, mask);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.paramId, 3);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.paramValue, 0xb);
}

TEST_F(CcuErrorHandlerTest, test_error_info_when_rep_type_is_post_shared_sem)
{
    MaskSignal sem;
    sem.Reset(1);           // sem id
    uint16_t mask = 0x0010; // mask

    shared_ptr<CcuRepBase> rep = make_shared<CcuRepPostSharedSem>(sem, mask);
    ErrorInfoBase baseInfo{0, 0, 1, 10, 0};
    vector<CcuErrorInfo> errorInfo{};
    CcuErrorHandler::GenErrorInfoByRepType(baseInfo, rep, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::WAIT_SIGNAL);
    EXPECT_EQ(errorInfo[0].repType, CcuRepType::POST_SHARED_SEM);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.signalId, 1);
    EXPECT_EQ(errorInfo[0].msg.waitSignal.signalMask, mask);
}

static uint64_t MockGetCcuGSAValueForRead(int32_t, uint32_t, uint32_t gsaId)
{
    if (gsaId == 1) {
        return 0xa;
    }
    if (gsaId == 3) {
        return 0xc;
    }
    return 0;
}

static uint64_t MockGetCcuXnValueForRead(int32_t, uint32_t, uint32_t xnId)
{
    if (xnId == 2) {
        return 0xb;
    }
    if (xnId == 4) {
        return 0xd;
    }
    if (xnId == 5) {
        return 0xe;
    }
    return 0;
}

TEST_F(CcuErrorHandlerTest, test_error_info_when_rep_type_is_read)
{
    auto utCcuConnection = MockCcuConnection(7); // channelId
    CcuTransport::CclBufferInfo locCclBufInfo;
    shared_ptr<CcuTransport> utCcuTransport
        = make_shared<CcuTransport>(nullptr, std::move(utCcuConnection), locCclBufInfo);

    Address locAddr;
    locAddr.Reset(1); // locAddr id
    Variable locToken;
    locToken.Reset(2); // locToken id
    Memory loc{locAddr, locToken};

    Address remAddr;
    remAddr.Reset(3); // remAddr id
    Variable remToken;
    remToken.Reset(4); // remToken id
    Memory rem{remAddr, remToken};

    Variable len;
    len.Reset(5);

    MaskSignal sem;
    sem.Reset(5); // sem id
    uint16_t mask = 0x0010;

    MOCKER(CcuErrorHandler::GetCcuGSAValue)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MockGetCcuGSAValueForRead));
    MOCKER(CcuErrorHandler::GetCcuXnValue)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MockGetCcuXnValueForRead));

    shared_ptr<CcuRepBase> rep = make_shared<CcuRepRead>(*utCcuTransport, loc, rem, len, sem, mask);
    ErrorInfoBase baseInfo{0, 0, 1, 10, 0};
    vector<CcuErrorInfo> errorInfo{};
    CcuErrorHandler::GenErrorInfoByRepType(baseInfo, rep, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::TRANS_MEM);
    EXPECT_EQ(errorInfo[0].repType, CcuRepType::READ);
    EXPECT_EQ(errorInfo[0].msg.transMem.locAddr, 0xa);
    EXPECT_EQ(errorInfo[0].msg.transMem.locToken, 0xb);
    EXPECT_EQ(errorInfo[0].msg.transMem.rmtAddr, 0xc);
    EXPECT_EQ(errorInfo[0].msg.transMem.rmtToken, 0xd);
    EXPECT_EQ(errorInfo[0].msg.transMem.len, 0xe);
    EXPECT_EQ(errorInfo[0].msg.transMem.signalId, 5);
    EXPECT_EQ(errorInfo[0].msg.transMem.signalMask, mask);
    EXPECT_EQ(errorInfo[0].msg.transMem.channelId, 7);
}

static uint64_t MockGetCcuGSAValueForWrite(int32_t, uint32_t, uint32_t gsaId)
{
    if (gsaId == 1) {
        return 0xa;
    }
    if (gsaId == 3) {
        return 0xc;
    }
    return 0;
}

static uint64_t MockGetCcuXnValueForWrite(int32_t, uint32_t, uint32_t xnId)
{
    if (xnId == 2) {
        return 0xb;
    }
    if (xnId == 4) {
        return 0xd;
    }
    if (xnId == 5) {
        return 0xe;
    }
    return 0;
}

TEST_F(CcuErrorHandlerTest, test_error_info_when_rep_type_is_write)
{
    auto utCcuConnection = MockCcuConnection(7); // channelId
    CcuTransport::CclBufferInfo locCclBufInfo;
    shared_ptr<CcuTransport> utCcuTransport
        = make_shared<CcuTransport>(nullptr, std::move(utCcuConnection), locCclBufInfo);

    Address locAddr;
    locAddr.Reset(1); // locAddr id
    Variable locToken;
    locToken.Reset(2); // locToken id
    Memory loc{locAddr, locToken};

    Address remAddr;
    remAddr.Reset(3); // remAddr id
    Variable remToken;
    remToken.Reset(4); // remToken id
    Memory rem{remAddr, remToken};

    Variable len;
    len.Reset(5);

    MaskSignal sem;
    sem.Reset(5); // sem id
    uint16_t mask = 0x0010;

    MOCKER(CcuErrorHandler::GetCcuGSAValue)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MockGetCcuGSAValueForWrite));
    MOCKER(CcuErrorHandler::GetCcuXnValue)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MockGetCcuXnValueForWrite));

    shared_ptr<CcuRepBase> rep = make_shared<CcuRepWrite>(*utCcuTransport, rem, loc, len, sem, mask);
    ErrorInfoBase baseInfo{0, 0, 1, 10, 0};
    vector<CcuErrorInfo> errorInfo{};
    CcuErrorHandler::GenErrorInfoByRepType(baseInfo, rep, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::TRANS_MEM);
    EXPECT_EQ(errorInfo[0].repType, CcuRepType::WRITE);
    EXPECT_EQ(errorInfo[0].msg.transMem.locAddr, 0xa);
    EXPECT_EQ(errorInfo[0].msg.transMem.locToken, 0xb);
    EXPECT_EQ(errorInfo[0].msg.transMem.rmtAddr, 0xc);
    EXPECT_EQ(errorInfo[0].msg.transMem.rmtToken, 0xd);
    EXPECT_EQ(errorInfo[0].msg.transMem.len, 0xe);
    EXPECT_EQ(errorInfo[0].msg.transMem.signalId, 5);
    EXPECT_EQ(errorInfo[0].msg.transMem.signalMask, mask);
    EXPECT_EQ(errorInfo[0].msg.transMem.channelId, 7);
}

TEST_F(CcuErrorHandlerTest, test_error_info_when_rep_type_is_local_cpy)
{
    Address srcAddr;
    srcAddr.Reset(1); // srcAddr id
    Variable srcToken;
    srcToken.Reset(2); // srcToken id
    Memory src{srcAddr, srcToken};
    MOCKER(CcuErrorHandler::GetCcuGSAValue).stubs().with(mockcpp::any(), mockcpp::any(), eq(1)).will(returnValue(0xa));
    MOCKER(CcuErrorHandler::GetCcuXnValue).stubs().with(mockcpp::any(), mockcpp::any(), eq(2)).will(returnValue(0xb));

    Address dstAddr;
    dstAddr.Reset(3); // dstAddr id
    Variable dstToken;
    dstToken.Reset(4); // dstToken id
    Memory dst{dstAddr, dstToken};
    MOCKER(CcuErrorHandler::GetCcuGSAValue).stubs().with(mockcpp::any(), mockcpp::any(), eq(3)).will(returnValue(0xc));
    MOCKER(CcuErrorHandler::GetCcuXnValue).stubs().with(mockcpp::any(), mockcpp::any(), eq(4)).will(returnValue(0xd));

    Variable len;
    len.Reset(5);
    MOCKER(CcuErrorHandler::GetCcuXnValue).stubs().with(mockcpp::any(), mockcpp::any(), eq(5)).will(returnValue(0xe));

    MaskSignal sem;
    sem.Reset(5); // sem id
    uint16_t mask = 0x0010;

    shared_ptr<CcuRepBase> rep = make_shared<CcuRepLocCpy>(dst, src, len, sem, mask);
    ErrorInfoBase baseInfo{0, 0, 1, 10, 0};
    vector<CcuErrorInfo> errorInfo{};
    CcuErrorHandler::GenErrorInfoByRepType(baseInfo, rep, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::TRANS_MEM);
    EXPECT_EQ(errorInfo[0].repType, CcuRepType::LOCAL_CPY);
    EXPECT_EQ(errorInfo[0].msg.transMem.locAddr, 0xa);
    EXPECT_EQ(errorInfo[0].msg.transMem.locToken, 0xb);
    EXPECT_EQ(errorInfo[0].msg.transMem.rmtAddr, 0xc);
    EXPECT_EQ(errorInfo[0].msg.transMem.rmtToken, 0xd);
    EXPECT_EQ(errorInfo[0].msg.transMem.len, 0xe);
    EXPECT_EQ(errorInfo[0].msg.transMem.signalId, 5);
    EXPECT_EQ(errorInfo[0].msg.transMem.signalMask, mask);
}

TEST_F(CcuErrorHandlerTest, test_error_info_when_rep_type_is_local_reduce)
{
    Address srcAddr;
    srcAddr.Reset(1); // srcAddr id
    Variable srcToken;
    srcToken.Reset(2); // srcToken id
    Memory src{srcAddr, srcToken};
    MOCKER(CcuErrorHandler::GetCcuGSAValue).stubs().with(mockcpp::any(), mockcpp::any(), eq(1)).will(returnValue(0xa));
    MOCKER(CcuErrorHandler::GetCcuXnValue).stubs().with(mockcpp::any(), mockcpp::any(), eq(2)).will(returnValue(0xb));

    Address dstAddr;
    dstAddr.Reset(3); // dstAddr id
    Variable dstToken;
    dstToken.Reset(4); // dstToken id
    Memory dst{dstAddr, dstToken};
    MOCKER(CcuErrorHandler::GetCcuGSAValue).stubs().with(mockcpp::any(), mockcpp::any(), eq(3)).will(returnValue(0xc));
    MOCKER(CcuErrorHandler::GetCcuXnValue).stubs().with(mockcpp::any(), mockcpp::any(), eq(4)).will(returnValue(0xd));

    Variable len;
    len.Reset(5);
    MOCKER(CcuErrorHandler::GetCcuXnValue).stubs().with(mockcpp::any(), mockcpp::any(), eq(5)).will(returnValue(0xe));

    MaskSignal sem;
    sem.Reset(5); // sem id
    uint16_t mask = 0x0010;

    shared_ptr<CcuRepBase> rep = make_shared<CcuRepLocCpy>(dst, src, len, 6, 7, sem, mask);
    ErrorInfoBase baseInfo{0, 0, 1, 10, 0};
    vector<CcuErrorInfo> errorInfo{};
    CcuErrorHandler::GenErrorInfoByRepType(baseInfo, rep, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::TRANS_MEM);
    EXPECT_EQ(errorInfo[0].repType, CcuRepType::LOCAL_REDUCE);
    EXPECT_EQ(errorInfo[0].msg.transMem.locAddr, 0xa);
    EXPECT_EQ(errorInfo[0].msg.transMem.locToken, 0xb);
    EXPECT_EQ(errorInfo[0].msg.transMem.rmtAddr, 0xc);
    EXPECT_EQ(errorInfo[0].msg.transMem.rmtToken, 0xd);
    EXPECT_EQ(errorInfo[0].msg.transMem.len, 0xe);
    EXPECT_EQ(errorInfo[0].msg.transMem.signalId, 5);
    EXPECT_EQ(errorInfo[0].msg.transMem.signalMask, mask);
    EXPECT_EQ(errorInfo[0].msg.transMem.opType, 7);
    EXPECT_EQ(errorInfo[0].msg.transMem.dataType, 6);
}

TEST_F(CcuErrorHandlerTest, test_error_info_when_rep_type_is_buf_read)
{
    auto utCcuConnection = MockCcuConnection(7); // channelId
    CcuTransport::CclBufferInfo locCclBufInfo;
    shared_ptr<CcuTransport> utCcuTransport
        = make_shared<CcuTransport>(nullptr, std::move(utCcuConnection), locCclBufInfo);

    Address addr;
    addr.Reset(1); // addr id
    Variable token;
    token.Reset(2); // token id
    Memory src{addr, token};
    MOCKER(CcuErrorHandler::GetCcuGSAValue).stubs().with(mockcpp::any(), mockcpp::any(), eq(1)).will(returnValue(0xa));
    MOCKER(CcuErrorHandler::GetCcuXnValue).stubs().with(mockcpp::any(), mockcpp::any(), eq(2)).will(returnValue(0xb));

    CcuRep::CcuBuffer dst;
    dst.Reset(3); // dst id

    Variable len;
    len.Reset(5);
    MOCKER(CcuErrorHandler::GetCcuXnValue).stubs().with(mockcpp::any(), mockcpp::any(), eq(5)).will(returnValue(0xe));

    MaskSignal sem;
    sem.Reset(5); // sem id
    uint16_t mask = 0x0010;

    shared_ptr<CcuRepBase> rep = make_shared<CcuRepBufRead>(*utCcuTransport, src, dst, len, sem, mask);
    ErrorInfoBase baseInfo{0, 0, 1, 10, 0};
    vector<CcuErrorInfo> errorInfo{};
    CcuErrorHandler::GenErrorInfoByRepType(baseInfo, rep, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::BUF_TRANS_MEM);
    EXPECT_EQ(errorInfo[0].repType, CcuRepType::BUF_READ);
    EXPECT_EQ(errorInfo[0].msg.bufTransMem.bufId, 3);
    EXPECT_EQ(errorInfo[0].msg.bufTransMem.addr, 0xa);
    EXPECT_EQ(errorInfo[0].msg.bufTransMem.token, 0xb);
    EXPECT_EQ(errorInfo[0].msg.bufTransMem.len, 0xe);
    EXPECT_EQ(errorInfo[0].msg.bufTransMem.signalId, 5);
    EXPECT_EQ(errorInfo[0].msg.bufTransMem.signalMask, mask);
    EXPECT_EQ(errorInfo[0].msg.bufTransMem.channelId, 7);
}

TEST_F(CcuErrorHandlerTest, test_error_info_when_rep_type_is_buf_write)
{
    auto utCcuConnection = MockCcuConnection(7); // channelId
    CcuTransport::CclBufferInfo locCclBufInfo;
    shared_ptr<CcuTransport> utCcuTransport
        = make_shared<CcuTransport>(nullptr, std::move(utCcuConnection), locCclBufInfo);

    Address addr;
    addr.Reset(1); // addr id
    Variable token;
    token.Reset(2); // token id
    Memory dst{addr, token};
    MOCKER(CcuErrorHandler::GetCcuGSAValue).stubs().with(mockcpp::any(), mockcpp::any(), eq(1)).will(returnValue(0xa));
    MOCKER(CcuErrorHandler::GetCcuXnValue).stubs().with(mockcpp::any(), mockcpp::any(), eq(2)).will(returnValue(0xb));

    CcuRep::CcuBuffer src;
    src.Reset(3); // src id

    Variable len;
    len.Reset(5);
    MOCKER(CcuErrorHandler::GetCcuXnValue).stubs().with(mockcpp::any(), mockcpp::any(), eq(5)).will(returnValue(0xe));

    MaskSignal sem;
    sem.Reset(5); // sem id
    uint16_t mask = 0x0010;

    shared_ptr<CcuRepBase> rep = make_shared<CcuRepBufWrite>(*utCcuTransport, src, dst, len, sem, mask);
    ErrorInfoBase baseInfo{0, 0, 1, 10, 0};
    vector<CcuErrorInfo> errorInfo{};
    CcuErrorHandler::GenErrorInfoByRepType(baseInfo, rep, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::BUF_TRANS_MEM);
    EXPECT_EQ(errorInfo[0].repType, CcuRepType::BUF_WRITE);
    EXPECT_EQ(errorInfo[0].msg.bufTransMem.bufId, 3);
    EXPECT_EQ(errorInfo[0].msg.bufTransMem.addr, 0xa);
    EXPECT_EQ(errorInfo[0].msg.bufTransMem.token, 0xb);
    EXPECT_EQ(errorInfo[0].msg.bufTransMem.len, 0xe);
    EXPECT_EQ(errorInfo[0].msg.bufTransMem.signalId, 5);
    EXPECT_EQ(errorInfo[0].msg.bufTransMem.signalMask, mask);
    EXPECT_EQ(errorInfo[0].msg.bufTransMem.channelId, 7);
}

TEST_F(CcuErrorHandlerTest, test_error_info_when_rep_type_is_buf_loc_read)
{
    Address addr;
    addr.Reset(1); // addr id
    Variable token;
    token.Reset(2); // token id
    Memory src{addr, token};
    MOCKER(CcuErrorHandler::GetCcuGSAValue).stubs().with(mockcpp::any(), mockcpp::any(), eq(1)).will(returnValue(0xa));
    MOCKER(CcuErrorHandler::GetCcuXnValue).stubs().with(mockcpp::any(), mockcpp::any(), eq(2)).will(returnValue(0xb));

    CcuRep::CcuBuffer dst;
    dst.Reset(3); // dst id

    Variable len;
    len.Reset(5);
    MOCKER(CcuErrorHandler::GetCcuXnValue).stubs().with(mockcpp::any(), mockcpp::any(), eq(5)).will(returnValue(0xe));

    MaskSignal sem;
    sem.Reset(5); // sem id
    uint16_t mask = 0x0010;

    shared_ptr<CcuRepBase> rep = make_shared<CcuRepBufLocRead>(src, dst, len, sem, mask);
    ErrorInfoBase baseInfo{0, 0, 1, 10, 0};
    vector<CcuErrorInfo> errorInfo{};
    CcuErrorHandler::GenErrorInfoByRepType(baseInfo, rep, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::BUF_TRANS_MEM);
    EXPECT_EQ(errorInfo[0].repType, CcuRepType::BUF_LOC_READ);
    EXPECT_EQ(errorInfo[0].msg.bufTransMem.bufId, 3);
    EXPECT_EQ(errorInfo[0].msg.bufTransMem.addr, 0xa);
    EXPECT_EQ(errorInfo[0].msg.bufTransMem.token, 0xb);
    EXPECT_EQ(errorInfo[0].msg.bufTransMem.len, 0xe);
    EXPECT_EQ(errorInfo[0].msg.bufTransMem.signalId, 5);
    EXPECT_EQ(errorInfo[0].msg.bufTransMem.signalMask, mask);
}

TEST_F(CcuErrorHandlerTest, test_error_info_when_rep_type_is_buf_loc_write)
{
    Address addr;
    addr.Reset(1); // addr id
    Variable token;
    token.Reset(2); // token id
    Memory dst{addr, token};
    MOCKER(CcuErrorHandler::GetCcuGSAValue).stubs().with(mockcpp::any(), mockcpp::any(), eq(1)).will(returnValue(0xa));
    MOCKER(CcuErrorHandler::GetCcuXnValue).stubs().with(mockcpp::any(), mockcpp::any(), eq(2)).will(returnValue(0xb));

    CcuRep::CcuBuffer src;
    src.Reset(3); // src id

    Variable len;
    len.Reset(5);
    MOCKER(CcuErrorHandler::GetCcuXnValue).stubs().with(mockcpp::any(), mockcpp::any(), eq(5)).will(returnValue(0xe));

    MaskSignal sem;
    sem.Reset(5); // sem id
    uint16_t mask = 0x0010;

    shared_ptr<CcuRepBase> rep = make_shared<CcuRepBufLocWrite>(src, dst, len, sem, mask);
    ErrorInfoBase baseInfo{0, 0, 1, 10, 0};
    vector<CcuErrorInfo> errorInfo{};
    CcuErrorHandler::GenErrorInfoByRepType(baseInfo, rep, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::BUF_TRANS_MEM);
    EXPECT_EQ(errorInfo[0].repType, CcuRepType::BUF_LOC_WRITE);
    EXPECT_EQ(errorInfo[0].msg.bufTransMem.bufId, 3);
    EXPECT_EQ(errorInfo[0].msg.bufTransMem.addr, 0xa);
    EXPECT_EQ(errorInfo[0].msg.bufTransMem.token, 0xb);
    EXPECT_EQ(errorInfo[0].msg.bufTransMem.len, 0xe);
    EXPECT_EQ(errorInfo[0].msg.bufTransMem.signalId, 5);
    EXPECT_EQ(errorInfo[0].msg.bufTransMem.signalMask, mask);
}

TEST_F(CcuErrorHandlerTest, test_error_info_when_rep_type_is_buf_reduce)
{
    CcuRep::CcuBuffer buf1;
    buf1.Reset(1); // buf1 id
    CcuRep::CcuBuffer buf2;
    buf2.Reset(2); // buf2 id
    CcuRep::CcuBuffer buf3;
    buf3.Reset(3); // buf3 id
    CcuRep::CcuBuffer buf4;
    buf4.Reset(4); // buf4 id
    vector<CcuRep::CcuBuffer> men{buf1, buf2, buf3, buf4};

    MaskSignal sem;
    sem.Reset(5); // sem id
    uint16_t mask = 0x0010;
    Variable len;

    shared_ptr<CcuRepBase> rep = make_shared<CcuRepBufReduce>(men, 4, 6, 7, 8, sem, len, mask);
    ErrorInfoBase baseInfo{0, 0, 1, 10, 0};
    vector<CcuErrorInfo> errorInfo{};
    CcuErrorHandler::GenErrorInfoByRepType(baseInfo, rep, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::BUF_REDUCE);
    EXPECT_EQ(errorInfo[0].repType, CcuRepType::BUF_REDUCE);
    EXPECT_EQ(errorInfo[0].msg.bufReduce.count, 4);
    EXPECT_EQ(errorInfo[0].msg.bufReduce.dataType, 6);
    EXPECT_EQ(errorInfo[0].msg.bufReduce.outputDataType, 7);
    EXPECT_EQ(errorInfo[0].msg.bufReduce.opType, 8);
    EXPECT_EQ(errorInfo[0].msg.bufReduce.signalId, 5);
    EXPECT_EQ(errorInfo[0].msg.bufReduce.signalMask, mask);
    EXPECT_EQ(errorInfo[0].msg.bufReduce.bufIds[0], 1);
    EXPECT_EQ(errorInfo[0].msg.bufReduce.bufIds[1], 2);
    EXPECT_EQ(errorInfo[0].msg.bufReduce.bufIds[2], 3);
    EXPECT_EQ(errorInfo[0].msg.bufReduce.bufIds[3], 4);
    for (uint32_t i = 4; i < BUF_REDUCE_ID_SIZE; ++i) {
        EXPECT_EQ(errorInfo[0].msg.bufReduce.bufIds[i], 0xffff);
    }
}

TEST_F(CcuErrorHandlerTest, test_error_info_when_default)
{
    Variable targetInstrId;
    shared_ptr<CcuRepBase> rep = make_shared<CcuRepJump>("label", targetInstrId);
    ErrorInfoBase baseInfo{0, 0, 1, 10, 0};
    vector<CcuErrorInfo> errorInfo{};
    CcuErrorHandler::GenErrorInfoByRepType(baseInfo, rep, errorInfo);
    EXPECT_EQ(errorInfo.size(), 1);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::DEFAULT);
    EXPECT_EQ(errorInfo[0].repType, CcuRepType::JUMP);
}

class MockCcuContext : public CcuContext {
public:
    MockCcuContext() : CcuContext() {}
    ~MockCcuContext() {}

    void Algorithm() override {}
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) override { return {}; }
};

TEST_F(CcuErrorHandlerTest, test_loop_group_error_info)
{
    MOCKER(CcuErrorHandler::GenErrorInfoLoop).stubs();

    Variable parallelParam;
    Variable offsetParam;

    LoopGroupXn loopGroupXn{};
    loopGroupXn.loopInsCnt = 5;
    loopGroupXn.expandOffset = 3;
    loopGroupXn.expandCnt = 2;
    MOCKER(CcuErrorHandler::GetCcuXnValue)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(loopGroupXn.value));

    shared_ptr<CcuRepBase> rep = make_shared<CcuRepLoopGroup>(parallelParam, offsetParam);
    ErrorInfoBase baseInfo{0, 0, 1, 10, 0};
    vector<CcuErrorInfo> errorInfo{};
    MockCcuContext mockCcuCtx{};
    CcuErrorHandler::GenErrorInfoLoopGroup(baseInfo, rep, mockCcuCtx, errorInfo);

    EXPECT_EQ(errorInfo.size(), 1);
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::LOOP_GROUP);
    EXPECT_EQ(errorInfo[0].repType, CcuRepType::LOOPGROUP);
    EXPECT_EQ(errorInfo[0].msg.loopGroup.startLoopInsId, 3);
    EXPECT_EQ(errorInfo[0].msg.loopGroup.loopInsCnt, 5);
    EXPECT_EQ(errorInfo[0].msg.loopGroup.expandOffset, 3);
    EXPECT_EQ(errorInfo[0].msg.loopGroup.expandCnt, 2);
}

TEST_F(CcuErrorHandlerTest, test_gen_error_info_loop_should_throw_exception)
{
    ErrorInfoBase baseInfo{0, 0, 1, 10, 0};
    vector<CcuErrorInfo> errorInfo{};

    shared_ptr<CcuRepBase> nullRep = shared_ptr<CcuRepLoop>(nullptr);

    Variable loopParam;
    shared_ptr<CcuRepLoop> loop = make_shared<CcuRepLoop>("loop_label", loopParam);
    shared_ptr<CcuRepLoopBlock> loopBlock = make_shared<CcuRepLoopBlock>("loop_block_label");
    loop->Reference(loopBlock);

    MockCcuContext mockCcuCtx{};
    MOCKER_CPP(&MockCcuContext::GetRepByInstrId)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(nullRep))
        .then(returnValue(static_pointer_cast<CcuRepBase>(loop)));

    // Failed to find Loop REP from CcuContext
    EXPECT_THROW(CcuErrorHandler::GenErrorInfoLoop(baseInfo, mockCcuCtx, errorInfo), CcuApiException);

    MOCKER(CcuErrorHandler::GetCcuXnValue)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(0x5));
    CcuLoopContext loopCtx{};
    MOCKER(CcuErrorHandler::GetCcuLoopContext)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(loopCtx));
    MOCKER_CPP(&CcuRepBlock::GetRepByInstrId).stubs().with(mockcpp::any()).will(returnValue(nullRep));
    // Failed to find REP from Loop
    EXPECT_THROW(CcuErrorHandler::GenErrorInfoLoop(baseInfo, mockCcuCtx, errorInfo), CcuApiException);
}

TEST_F(CcuErrorHandlerTest, test_gen_error_info_loop)
{
    // Mock Loop内的Rep
    MaskSignal sem;
    sem.Reset(0xa);         // sem id
    uint16_t mask = 0x0010; // mask
    MOCKER(CcuErrorHandler::GetCcuCKEValue)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), eq(0xa))
        .will(returnValue(static_cast<u16>(0xabcd)));

    // Mock LoopBlock
    shared_ptr<CcuRepLoopBlock> loopBlock = make_shared<CcuRepLoopBlock>("loop_block_label");
    shared_ptr<CcuRepLocPostSem> locPostSem = make_shared<CcuRepLocPostSem>(sem, mask);
    locPostSem->instrId = 0;
    locPostSem->instrCount = 1;
    loopBlock->Append(locPostSem); // instr 0

    // Mock Loop
    Variable loopParam;
    shared_ptr<CcuRepLoop> loop = make_shared<CcuRepLoop>("loop_label", loopParam); // instr 3
    loop->Reference(loopBlock);
    loop->instrId = 1;
    loop->instrCount = 1;

    MockCcuContext mockCcuCtx{};
    mockCcuCtx.Append(loop);

    LoopXm loopXm{};
    loopXm.loopCnt = 5; // loopCnt
    MOCKER(CcuErrorHandler::GetCcuXnValue)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(loopXm.value));

    CcuLoopContext loopCtx{}; // currentIns = 2, currentCnt = 3, addrStride = 0xaabbccdd
    loopCtx.part10.currentIns = 0;
    loopCtx.part9.currentIns = 0;
    loopCtx.part14.currentCnt = 0;
    loopCtx.part13.currentCnt = 3;
    loopCtx.part10.addrStride = 0b011101;
    loopCtx.part11.addrStride = 0b1110111100110011;
    loopCtx.part12.addrStride = 0b1010101010;
    MOCKER(CcuErrorHandler::GetCcuLoopContext)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(loopCtx));

    ErrorInfoBase baseInfo{0, 0, 0, 1, 0}; // currentInsId = 2
    vector<CcuErrorInfo> errorInfo{};
    CcuErrorHandler::GenErrorInfoLoop(baseInfo, mockCcuCtx, errorInfo);

    EXPECT_EQ(errorInfo.size(), 2);
    // check Loop
    EXPECT_EQ(errorInfo[0].type, CcuErrorType::LOOP);
    EXPECT_EQ(errorInfo[0].repType, CcuRepType::LOOP);
    EXPECT_EQ(errorInfo[0].instrId, 1);
    EXPECT_EQ(errorInfo[0].msg.loop.startInstrId, 0);
    EXPECT_EQ(errorInfo[0].msg.loop.endInstrId, 0);
    EXPECT_EQ(errorInfo[0].msg.loop.loopCnt, 5);
    EXPECT_EQ(errorInfo[0].msg.loop.loopCurrentCnt, 3);
    EXPECT_EQ(errorInfo[0].msg.loop.addrStride, 0xaabbccdd);
    // check Rep in loop
    EXPECT_EQ(errorInfo[1].type, CcuErrorType::WAIT_SIGNAL);
    EXPECT_EQ(errorInfo[1].repType, CcuRepType::LOC_POST_SEM);
    EXPECT_EQ(errorInfo[1].instrId, 0);
    EXPECT_EQ(errorInfo[1].msg.waitSignal.signalId, 0xa);
    EXPECT_EQ(errorInfo[1].msg.waitSignal.signalValue, 0xabcd);
    EXPECT_EQ(errorInfo[1].msg.waitSignal.signalMask, mask);
}

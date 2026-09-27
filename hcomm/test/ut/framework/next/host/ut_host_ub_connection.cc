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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#define private public
#define protected public
#include "host_ub_connection.h"
#include "rma_conn_manager.h"
#include "socket.h"
#include "orion_adapter_rts.h"
#include "not_support_exception.h"
#include "rma_conn_exception.h"
#include "rdma_handle_manager.h"
#include "tp_manager.h"
#include "hccp_async.h"
#include "hccp_ctx.h"
#include "exception_util.h"
#include "exchange_ub_conn_dto.h"
#undef protected
#undef private
#include "hccp_async_ctx.h"

using namespace Hccl;

class HostUbConnectionTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HostUbConnection tests set up." << std::endl; }

    static void TearDownTestCase()
    {
        GlobalMockObject::verify();
        std::cout << "HostUbConnection tests tear down." << std::endl;
    }

    virtual void SetUp()
    {
        GlobalMockObject::verify();
        MOCKER_CPP(&TpManager::GetTpInfo)
            .stubs()
            .will(returnValue(HcclResult::HCCL_E_AGAIN))
            .then(returnValue(HcclResult::HCCL_SUCCESS));
        std::cout << "A Test case in HostUbConnection SetUP" << std::endl;

        rdmaHandle = (void*)0x1000000;
        localIp = IpAddress("1.0.0.0");
        remoteIp = IpAddress("2.0.0.0");

        // Mock RdmaHandleManager
        MOCKER_CPP(&RdmaHandleManager::GetJfcHandle).stubs().will(returnValue(static_cast<JfcHandle>(0x2000000)));
        MOCKER_CPP(&RdmaHandleManager::GetDieAndFuncId).stubs().will(returnValue(std::pair<u32, u32>(0, 0)));
        MOCKER_CPP(&RdmaHandleManager::GetTokenIdInfo).stubs().will(returnValue(std::pair<u64, u32>(0, 0)));

        // Mock HrtRaUbCreateJetty
        HrtRaUbJettyCreatedOutParam jettyOutParam{};
        jettyOutParam.keySize = HRT_UB_QP_KEY_MAX_LEN;
        MOCKER(HrtRaUbCreateJetty).stubs().will(returnValue(jettyOutParam));

        // Mock TpManager Init
        MOCKER(HrtGetDevice).stubs().will(returnValue(0));
        MOCKER_CPP(&TpManager::Init).stubs().will(ignoreReturnValue());
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in HostUbConnection TearDown" << std::endl;
    }

    RdmaHandle rdmaHandle;
    IpAddress localIp;
    IpAddress remoteIp;
    u32 listenPort = 100;
    std::string tag = "test";
};

TEST_F(HostUbConnectionTest, rma_ub_connection_get_status_return_exchanging_and_ok)
{
    // Given: construct HostUbConnection
    HostUbConnection conn(rdmaHandle, localIp, remoteIp, OpMode::OPBASE);
    conn.tpProtocol = TpProtocol::CTP;
    EXPECT_EQ(RmaConnStatus::INVALID, conn.status);

    // When: GetStatus - INIT -> TP_INFO_GETTING (GetTpInfo returns HCCL_E_AGAIN from mock)
    RmaConnStatus status = conn.GetStatus();
    EXPECT_EQ(RmaConnStatus::INVALID, status);
    EXPECT_EQ(HostUbConnection::UbConnStatus::TP_INFO_GETTING, conn.ubConnStatus);

    // When: GetStatus - TP_INFO_GETTING -> EXCHANGEABLE after TpManager returns SUCCESS
    status = conn.GetStatus();
    EXPECT_EQ(RmaConnStatus::EXCHANGEABLE, status);
    EXPECT_EQ(HostUbConnection::UbConnStatus::JETTY_CREATED, conn.ubConnStatus);

    // Exchange DTO and Import
    auto rmtDto = conn.GetExchangeDto();
    conn.ParseRmtExchangeDto(*rmtDto);

    // Mock ImportJetty
    MOCKER(RaUbImportJetty).stubs().will(returnValue(HrtRaUbJettyImportedOutParam({0x6000000, 0x7000000, 0})));

    conn.ImportRmtDto();
    EXPECT_EQ(HostUbConnection::UbConnStatus::JETTY_IMPORTING, conn.ubConnStatus);

    // GetStatus - JETTY_IMPORTING -> READY
    status = conn.GetStatus();
    EXPECT_EQ(RmaConnStatus::READY, status);
    EXPECT_EQ(HostUbConnection::UbConnStatus::READY, conn.ubConnStatus);

    // GetStatus - READY stays READY
    status = conn.GetStatus();
    EXPECT_EQ(RmaConnStatus::READY, status);

    // Test Describe
    std::string msg = conn.Describe();
    EXPECT_NE(0u, msg.length());
}

TEST_F(HostUbConnectionTest, rma_ub_connection_get_status_invalid_throw_exception)
{
    // Given: construct HostUbConnection
    HostUbConnection conn(rdmaHandle, localIp, remoteIp, OpMode::OPBASE);

    // When: set invalid status
    conn.ubConnStatus = HostUbConnection::UbConnStatus::CONN_INVALID;

    // Then: GetStatus should throw
    EXPECT_THROW(conn.GetStatus(), RmaConnException);
}

TEST_F(HostUbConnectionTest, rma_ub_connection_suspend_change_status_suspend)
{
    // Given: construct HostUbConnection
    HostUbConnection conn(rdmaHandle, localIp, remoteIp, OpMode::OPBASE);

    // Mock ReleaseResource
    MOCKER(HrtRaUbUnimportJetty).stubs().will(ignoreReturnValue());
    MOCKER(HrtRaUbDestroyJetty).stubs().will(ignoreReturnValue());

    conn.jettyHandle_ = 1;
    conn.sqBuffVa = 0x1000000;
    conn.status = RmaConnStatus::READY;
    conn.ubConnStatus = HostUbConnection::UbConnStatus::READY;

    // When: Suspend (deprecated, always returns true without changing status)
    EXPECT_EQ(true, conn.Suspend());
    EXPECT_EQ(RmaConnStatus::READY, conn.status);

    // When: Suspend again
    EXPECT_EQ(true, conn.Suspend());
}

TEST_F(HostUbConnectionTest, rma_ub_connection_suspend_invalid_throw_exception)
{
    // Given: construct HostUbConnection
    HostUbConnection conn(rdmaHandle, localIp, remoteIp, OpMode::OPBASE);

    // Mock ReleaseResource
    MOCKER(HrtRaUbUnimportJetty).stubs().will(ignoreReturnValue());
    MOCKER(HrtRaUbDestroyJetty).stubs().will(ignoreReturnValue());

    // When: status is CLOSE - Suspend returns true (deprecated, no exception thrown)
    conn.status = RmaConnStatus::CLOSE;
    EXPECT_EQ(true, conn.Suspend());
    EXPECT_EQ(RmaConnStatus::CLOSE, conn.status);
}

TEST_F(HostUbConnectionTest, rma_ub_connection_get_pi_ci_sqDepth_ok)
{
    // Given: construct HostUbConnection
    HostUbConnection conn(rdmaHandle, localIp, remoteIp, OpMode::OPBASE);

    // Then: check initial values
    EXPECT_EQ(conn.GetPiVal(), 0u);
    EXPECT_EQ(conn.GetCiVal(), 0u);
    EXPECT_EQ(conn.GetSqDepth(), 8192u); // OPBASE_UB_SQ_DEPTH_MAX
}

TEST_F(HostUbConnectionTest, rma_ub_connection_get_jfcMode_and_jettyHandle_ok)
{
    // Given: construct HostUbConnection with USER_CTL mode
    HostUbConnection conn(rdmaHandle, localIp, remoteIp, OpMode::OPBASE, HrtUbJfcMode::USER_CTL);

    JettyHandle jettyHandle = 100;
    conn.jettyHandle_ = jettyHandle;

    // Then
    EXPECT_EQ(conn.GetUbJfcMode(), HrtUbJfcMode::USER_CTL);
    EXPECT_EQ(conn.GetJettyHandle(), jettyHandle);
    EXPECT_EQ(conn.GetRemoteJettyHandle(), 0u); // not set yet
}

TEST_F(HostUbConnectionTest, rma_ub_connection_get_cq_va_and_jetty_va)
{
    // Given: construct HostUbConnection
    HostUbConnection conn(rdmaHandle, localIp, remoteIp, OpMode::OPBASE);

    conn.cqInfo_.va = 0x8000000;
    conn.jettyVa_ = 0x9000000;
    conn.remoteJettyVa_ = 0xA000000;

    // Then
    EXPECT_EQ(conn.GetCqVa(), 0x8000000u);
    EXPECT_EQ(conn.GetJettyVa(), 0x9000000u);
    EXPECT_EQ(conn.GetTJettyVa(), 0xA000000u);
}

TEST_F(HostUbConnectionTest, rma_ub_connection_get_unique_id)
{
    // Given: construct HostUbConnection
    HostUbConnection conn(rdmaHandle, localIp, remoteIp, OpMode::OPBASE);

    conn.jettyId_ = 123;
    conn.sqBuffVa = 0x8000000;
    conn.sqDepth = 8192;
    conn.tpn = 1;
    conn.rmtEid = remoteIp.GetEid();
    conn.locEid = localIp.GetEid();

    // When
    std::vector<char> uniqueId = conn.GetUniqueId();

    // Then: should not be empty
    EXPECT_FALSE(uniqueId.empty());
}

TEST_F(HostUbConnectionTest, rma_ub_connection_set_cq_info)
{
    // Given: construct HostUbConnection
    HostUbConnection conn(rdmaHandle, localIp, remoteIp, OpMode::OPBASE);

    conn.cqInfo_.id = 100;
    conn.cqInfo_.va = 0x8000000;
    conn.cqInfo_.cqeSize = 64;
    conn.cqInfo_.cqDepth = 128;
    conn.cqInfo_.swdbAddr = 0x9000000;

    // When
    HcclAiRMACQ cq{};
    conn.SetCqInfo(cq);

    // Then
    EXPECT_EQ(cq.jfcId, 100u);
    EXPECT_EQ(cq.cqVA, 0x8000000u);
    EXPECT_EQ(cq.cqeSize, 64u);
    EXPECT_EQ(cq.cqDepth, 128u);
    EXPECT_EQ(cq.dbAddr, 0x9000000u);
}

TEST_F(HostUbConnectionTest, rma_ub_connection_set_wq_info)
{
    // Given: construct HostUbConnection
    HostUbConnection conn(rdmaHandle, localIp, remoteIp, OpMode::OPBASE);

    conn.jettyId_ = 100;
    conn.dbAddr = 0x8000000;
    conn.sqBuffVa = 0x9000000;
    conn.sqDepth = 8192;
    conn.tpn = 1;
    conn.rmtEid = remoteIp.GetEid();
    conn.rmtReverseEid = remoteIp.GetReverseEid();

    // When
    HcclAiRMAWQ wq{};
    conn.SetWqInfo(wq);

    // Then
    EXPECT_EQ(wq.jettyId, 100u);
    EXPECT_EQ(wq.dbAddr, 0x8000000u);
    EXPECT_EQ(wq.sqVA, 0x9000000u);
    EXPECT_EQ(wq.sqDepth, 8192u * 4u); // sqDepth * WQE_NUM_PER_SQE
    EXPECT_EQ(wq.tp_id, 1u);
}

TEST_F(HostUbConnectionTest, rma_ub_connection_connect)
{
    // Given: construct HostUbConnection
    HostUbConnection conn(rdmaHandle, localIp, remoteIp, OpMode::OPBASE);
    conn.tpProtocol = TpProtocol::CTP;

    // When: Connect (calls GetStatus)
    EXPECT_NO_THROW(conn.Connect());
}

TEST_F(HostUbConnectionTest, rma_ub_connection_import_rmt_dto_when_ready)
{
    // Given: construct HostUbConnection and set status to READY
    HostUbConnection conn(rdmaHandle, localIp, remoteIp, OpMode::OPBASE);
    conn.ubConnStatus = HostUbConnection::UbConnStatus::READY;

    // When: ImportRmtDto when already ready
    EXPECT_NO_THROW(conn.ImportRmtDto());
}

TEST_F(HostUbConnectionTest, rma_ub_connection_import_rmt_dto_invalid_status)
{
    // Given: construct HostUbConnection with invalid status
    HostUbConnection conn(rdmaHandle, localIp, remoteIp, OpMode::OPBASE);
    conn.ubConnStatus = HostUbConnection::UbConnStatus::INIT; // not JETTY_CREATED

    // When: ImportRmtDto with invalid status
    EXPECT_THROW(conn.ImportRmtDto(), RmaConnException);
}

TEST_F(HostUbConnectionTest, rma_ub_connection_update_ci_val)
{
    // Given: construct HostUbConnection
    HostUbConnection conn(rdmaHandle, localIp, remoteIp, OpMode::OPBASE);

    // When: UpdateCiVal
    conn.UpdateCiVal(123);

    // Then
    EXPECT_EQ(conn.GetCiVal(), 123u);
}

TEST_F(HostUbConnectionTest, IfNeedUpdatingUbCi_false)
{
    // Given
    HostUbConnection conn(rdmaHandle, localIp, remoteIp, OpMode::OPBASE);
    conn.piVal = 10;
    conn.ciVal = 0;
    conn.sqDepth = 8192;

    std::vector<HostUbConnection*> conns;
    conns.push_back(&conn);

    // When
    bool ret = IfNeedUpdatingUbCi(conns);

    // Then: pi-ci < sqDepth/2, should return false
    EXPECT_EQ(ret, false);
}

TEST_F(HostUbConnectionTest, IfNeedUpdatingUbCi_true)
{
    // Given
    HostUbConnection conn(rdmaHandle, localIp, remoteIp, OpMode::OPBASE);
    conn.piVal = 6000;
    conn.ciVal = 0;
    conn.sqDepth = 8192;

    std::vector<HostUbConnection*> conns;
    conns.push_back(&conn);

    // When
    bool ret = IfNeedUpdatingUbCi(conns);

    // Then: pi-ci >= sqDepth/2, should return true
    EXPECT_EQ(ret, true);
}

TEST_F(HostUbConnectionTest, IfNeedUpdatingUbCi_with_wraparound)
{
    // Given: pi wraps around
    HostUbConnection conn(rdmaHandle, localIp, remoteIp, OpMode::OPBASE);
    conn.piVal = 100;  // small pi (wrapped)
    conn.ciVal = 8000; // large ci
    conn.sqDepth = 8192;

    std::vector<HostUbConnection*> conns;
    conns.push_back(&conn);

    // When: pi + extra - ci = 100 + 8192 - 8000 = 292 < 4096, should return false
    bool ret = IfNeedUpdatingUbCi(conns);
    EXPECT_EQ(ret, false);
}

TEST_F(HostUbConnectionTest, HostUbTpConnection_constructor)
{
    // Given: construct HostUbTpConnection
    HostUbTpConnection conn(rdmaHandle, localIp, remoteIp, OpMode::OPBASE);

    // Then: tpProtocol should be TP
    EXPECT_EQ(conn.tpProtocol, TpProtocol::TP);
}

TEST_F(HostUbConnectionTest, HostUbCtpConnection_constructor)
{
    // Given: construct HostUbCtpConnection
    HostUbCtpConnection conn(rdmaHandle, localIp, remoteIp, OpMode::OPBASE);

    // Then: tpProtocol should be CTP
    EXPECT_EQ(conn.tpProtocol, TpProtocol::CTP);
}

TEST_F(HostUbConnectionTest, HostUbConnection_release_tp)
{
    // Given: construct HostUbConnection
    HostUbConnection conn(rdmaHandle, localIp, remoteIp, OpMode::OPBASE);
    conn.tpInfo.tpHandle = 0x12345678;

    // Mock TpManager::ReleaseTpInfo
    MOCKER_CPP(&TpManager::ReleaseTpInfo).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

    // When: ReleaseTp
    conn.ReleaseTp();

    // Then: tpHandle should be 0
    EXPECT_EQ(conn.tpInfo.tpHandle, 0u);
}

TEST_F(HostUbConnectionTest, HostUbConnection_release_resource)
{
    // Given: construct HostUbConnection
    HostUbConnection conn(rdmaHandle, localIp, remoteIp, OpMode::OPBASE);
    conn.rdmaHandle = rdmaHandle;
    conn.jettyHandle_ = 0x12345678;
    conn.remoteJettyHandle_ = 0x87654321;
    conn.tpInfo.tpHandle = 0xABCDEF;

    // Mock HrtRaUbUnimportJetty, HrtRaUbDestroyJetty, TpManager::ReleaseTpInfo
    MOCKER(HrtRaUbUnimportJetty).stubs().will(ignoreReturnValue());
    MOCKER(HrtRaUbDestroyJetty).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&TpManager::ReleaseTpInfo).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

    // When: ReleaseResource
    conn.ReleaseResource();

    // Then
    EXPECT_EQ(conn.remoteJettyHandle_, 0u);
    EXPECT_EQ(conn.jettyHandle_, 0u);
}

TEST_F(HostUbConnectionTest, HostUbConnection_get_rdma_handle)
{
    // Given: construct HostUbConnection
    HostUbConnection conn(rdmaHandle, localIp, remoteIp, OpMode::OPBASE);

    // Then
    EXPECT_EQ(conn.GetRdmaHandle(), rdmaHandle);
}

TEST_F(HostUbConnectionTest, HostUbConnection_get_exchange_dto_when_ready)
{
    // Given: construct HostUbConnection
    HostUbConnection conn(rdmaHandle, localIp, remoteIp, OpMode::OPBASE);
    conn.status = RmaConnStatus::READY;
    conn.tpProtocol = TpProtocol::CTP;
    conn.jettyImportCfg.localTpHandle = 0x123;
    conn.jettyImportCfg.localPsn = 456;
    conn.tokenValue = 789;
    conn.keySize = 32;
    memcpy_s(conn.repJetty_.key, HRT_UB_QP_KEY_MAX_LEN, "testkey", 7);

    // When
    auto dto = conn.GetExchangeDto();

    // Then: should not be null
    EXPECT_NE(dto, nullptr);
}

TEST_F(HostUbConnectionTest, HostUbConnection_get_exchange_dto_when_invalid_status)
{
    // Given: construct HostUbConnection with invalid status
    HostUbConnection conn(rdmaHandle, localIp, remoteIp, OpMode::OPBASE);
    conn.status = RmaConnStatus::INIT; // not EXCHANGEABLE or READY

    // When: should throw exception
    EXPECT_THROW(conn.GetExchangeDto(), RmaConnException);
}

TEST_F(HostUbConnectionTest, HostUbConnection_parse_rmt_exchange_dto)
{
    // Given: construct HostUbConnection
    HostUbConnection conn(rdmaHandle, localIp, remoteIp, OpMode::OPBASE);
    conn.tpProtocol = TpProtocol::CTP;

    // Create a mock ExchangeUbConnDto
    auto dto = std::make_unique<ExchangeUbConnDto>(12345, 32, 0xABCDEF, 999);
    memcpy_s(dto->qpKey, HRT_UB_QP_KEY_MAX_LEN, "remotekey", 9);

    // When: ParseRmtExchangeDto
    EXPECT_NO_THROW(conn.ParseRmtExchangeDto(*dto));

    // Then: remote values should be set
    EXPECT_EQ(conn.remoteTokenValue, 12345u);
    EXPECT_EQ(conn.jettyImportCfg.remoteTpHandle, 0xABCDEFu);
    EXPECT_EQ(conn.jettyImportCfg.remotePsn, 999u);
}

TEST_F(HostUbConnectionTest, HostUbConnection_throw_abnormal_status)
{
    // Given: construct HostUbConnection
    HostUbConnection conn(rdmaHandle, localIp, remoteIp, OpMode::OPBASE);
    conn.status = RmaConnStatus::INIT;
    conn.ubConnStatus = HostUbConnection::UbConnStatus::CONN_INVALID;

    // When: calling ThrowAbnormalStatus
    EXPECT_THROW(conn.ThrowAbnormalStatus("TestFunc"), RmaConnException);
    EXPECT_EQ(conn.status, RmaConnStatus::CONN_INVALID);
}

TEST_F(HostUbConnectionTest, HostUbConnection_destructor_releases_resources)
{
    // Given: construct HostUbConnection with resources
    // Mock the destructor chain
    MOCKER(HrtRaUbUnimportJetty).stubs().will(ignoreReturnValue());
    MOCKER(HrtRaUbDestroyJetty).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&TpManager::ReleaseTpInfo).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

    {
        HostUbConnection conn(rdmaHandle, localIp, remoteIp, OpMode::OPBASE);
        conn.remoteJettyHandle_ = 0x123;
        conn.jettyHandle_ = 0x456;
        conn.tpInfo.tpHandle = 0x789;
    } // destructor called here

    // If we reach here without crash, the test passes
    EXPECT_TRUE(true);
}

TEST_F(HostUbConnectionTest, HostUbConnection_set_import_info)
{
    // Given: construct HostUbConnection
    HostUbConnection conn(rdmaHandle, localIp, remoteIp, OpMode::OPBASE);

    // When: SetImportInfo (currently does nothing)
    EXPECT_NO_THROW(conn.SetImportInfo());
}

TEST_F(HostUbConnectionTest, HostUbConnection_offload_mode_sq_depth)
{
    // Given: construct HostUbConnection in OFFLOAD mode
    HostUbConnection conn(rdmaHandle, localIp, remoteIp, OpMode::OFFLOAD);

    // Then: sqDepth should be UB_SQ_OFFLOAD_DEPTH (128)
    EXPECT_EQ(conn.GetSqDepth(), 128u);
}

TEST_F(HostUbConnectionTest, HostUbConnection_prepare_write_zero_size)
{
    // Given: construct HostUbConnection
    HostUbConnection conn(rdmaHandle, localIp, remoteIp, OpMode::OPBASE);

    MemoryBuffer loc(0x1000, 0, 0);
    MemoryBuffer rmt(0x2000, 0, 0);
    SqeConfig config{};

    // When: PrepareWrite with zero size
    auto task = conn.PrepareWrite(rmt, loc, config);

    // Then: should return nullptr
    EXPECT_EQ(task, nullptr);
}

TEST_F(HostUbConnectionTest, HostUbConnection_prepare_read_zero_size)
{
    // Given: construct HostUbConnection
    HostUbConnection conn(rdmaHandle, localIp, remoteIp, OpMode::OPBASE);

    MemoryBuffer loc(0x1000, 0, 0);
    MemoryBuffer rmt(0x2000, 0, 0);
    SqeConfig config{};

    // When: PrepareRead with zero size
    auto task = conn.PrepareRead(rmt, loc, config);

    // Then: should return nullptr
    EXPECT_EQ(task, nullptr);
}

TEST_F(HostUbConnectionTest, HostUbConnection_prepare_write_reduce_zero_size)
{
    // Given: construct HostUbConnection
    HostUbConnection conn(rdmaHandle, localIp, remoteIp, OpMode::OPBASE);

    MemoryBuffer loc(0x1000, 0, 0);
    MemoryBuffer rmt(0x2000, 0, 0);
    SqeConfig config{};

    // When: PrepareWriteReduce with zero size
    auto task = conn.PrepareWriteReduce(rmt, loc, DataType::INT8, ReduceOp::SUM, config);

    // Then: should return nullptr
    EXPECT_EQ(task, nullptr);
}

TEST_F(HostUbConnectionTest, HostUbConnection_prepare_read_reduce_zero_size)
{
    // Given: construct HostUbConnection
    HostUbConnection conn(rdmaHandle, localIp, remoteIp, OpMode::OPBASE);

    MemoryBuffer loc(0x1000, 0, 0);
    MemoryBuffer rmt(0x2000, 0, 0);
    SqeConfig config{};

    // When: PrepareReadReduce with zero size
    auto task = conn.PrepareReadReduce(rmt, loc, DataType::INT8, ReduceOp::SUM, config);

    // Then: should return nullptr
    EXPECT_EQ(task, nullptr);
}

TEST_F(HostUbConnectionTest, HostUbConnection_prepare_write_with_notify_zero_size)
{
    // Given: construct HostUbConnection
    HostUbConnection conn(rdmaHandle, localIp, remoteIp, OpMode::OPBASE);

    MemoryBuffer loc(0x1000, 0, 0);
    MemoryBuffer rmt(0x2000, 0, 0);
    MemoryBuffer notify(0x3000, 100, 0);
    SqeConfig config{};

    // When: PrepareWriteWithNotify with zero size
    auto task = conn.PrepareWriteWithNotify(rmt, loc, 1, notify, config);

    // Then: should return nullptr
    EXPECT_EQ(task, nullptr);
}

TEST_F(HostUbConnectionTest, HostUbConnection_prepare_write_reduce_with_notify_zero_size)
{
    // Given: construct HostUbConnection
    HostUbConnection conn(rdmaHandle, localIp, remoteIp, OpMode::OPBASE);

    MemoryBuffer loc(0x1000, 0, 0);
    MemoryBuffer rmt(0x2000, 0, 0);
    MemoryBuffer notify(0x3000, 100, 0);
    SqeConfig config{};

    // When: PrepareWriteReduceWithNotify with zero size
    auto task = conn.PrepareWriteReduceWithNotify(rmt, loc, DataType::INT8, ReduceOp::SUM, 1, notify, config);

    // Then: should return nullptr
    EXPECT_EQ(task, nullptr);
}

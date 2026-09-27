/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "roce_transport_lite_impl.h"
#include "rma_buffer_lite.h"
#include "rmt_rma_buffer_lite.h"
#include "notify_lite.h"
#include "binary_stream.h"
#include "stream.h"
#include "stream_lite.h"
#include "rtsq_base.h"
#include "sqe.h"
#include "rdma_conn_lite_v2.h"

#define private public

using namespace Hccl;

static const u32 ROCE_TYPE = 1;

static std::vector<char> BuildSingleRmaBufferUniqueId(u64 addr, u64 size, u32 key)
{
    BinaryStream bs;
    bs << addr;
    bs << size;
    bs << key;
    std::vector<char> result;
    bs.Dump(result);
    return result;
}

static std::vector<char> BuildLocNotifyUniqueIds(u32 notifyNum)
{
    std::vector<char> result;
    for (u32 i = 0; i < notifyNum; i++) {
        u32 notifyId = i + 1;
        u32 devPhyId = i + 10;
        result.insert(
            result.end(), reinterpret_cast<char*>(&notifyId), reinterpret_cast<char*>(&notifyId) + sizeof(notifyId));
        result.insert(
            result.end(), reinterpret_cast<char*>(&devPhyId), reinterpret_cast<char*>(&devPhyId) + sizeof(devPhyId));
    }
    return result;
}

static std::vector<char> BuildRmtNotifyUniqueIds(u32 notifyNum)
{
    std::vector<char> result;
    for (u32 i = 0; i < notifyNum; i++) {
        u64 addr = 0x2000 + i * 0x100;
        u64 size = 0x100;
        u32 rkey = 0x200 + i;
        result.insert(result.end(), reinterpret_cast<char*>(&addr), reinterpret_cast<char*>(&addr) + sizeof(addr));
        result.insert(result.end(), reinterpret_cast<char*>(&size), reinterpret_cast<char*>(&size) + sizeof(size));
        result.insert(result.end(), reinterpret_cast<char*>(&rkey), reinterpret_cast<char*>(&rkey) + sizeof(rkey));
    }
    return result;
}

static std::vector<char> BuildNotifyValueBufferUniqueId() { return BuildSingleRmaBufferUniqueId(0x3000, 0x200, 0x300); }

static std::vector<char> BuildLocBufferUniqueIds(u32 bufferNum)
{
    std::vector<char> result;
    for (u32 i = 0; i < bufferNum; i++) {
        u64 addr = 0x4000 + i * 0x200;
        u64 size = 0x200;
        u32 lkey = 0x400 + i;
        result.insert(result.end(), reinterpret_cast<char*>(&addr), reinterpret_cast<char*>(&addr) + sizeof(addr));
        result.insert(result.end(), reinterpret_cast<char*>(&size), reinterpret_cast<char*>(&size) + sizeof(size));
        result.insert(result.end(), reinterpret_cast<char*>(&lkey), reinterpret_cast<char*>(&lkey) + sizeof(lkey));
    }
    return result;
}

static std::vector<char> BuildRmtBufferUniqueIds(u32 bufferNum)
{
    std::vector<char> result;
    for (u32 i = 0; i < bufferNum; i++) {
        u64 addr = 0x5000 + i * 0x200;
        u64 size = 0x200;
        u32 rkey = 0x600 + i;
        result.insert(result.end(), reinterpret_cast<char*>(&addr), reinterpret_cast<char*>(&addr) + sizeof(addr));
        result.insert(result.end(), reinterpret_cast<char*>(&size), reinterpret_cast<char*>(&size) + sizeof(size));
        result.insert(result.end(), reinterpret_cast<char*>(&rkey), reinterpret_cast<char*>(&rkey) + sizeof(rkey));
    }
    return result;
}

static std::vector<char> BuildSqUniqueId(const RdmaSqContextLite& sqCtx)
{
    BinaryStream bs;
    bs << sqCtx.qpn;
    bs << sqCtx.sqVa;
    bs << sqCtx.wqeSize;
    bs << sqCtx.depth;
    bs << sqCtx.headAddr;
    bs << sqCtx.tailAddr;
    bs << sqCtx.dbHwVa;
    bs << sqCtx.dbSwVa;
    bs << sqCtx.sl;
    bs << sqCtx.dbVendorSpecified;
    std::vector<char> result;
    bs.Dump(result);
    return result;
}

static std::vector<char> BuildCqUniqueId(const RdmaCqContextLite& cqCtx)
{
    BinaryStream bs;
    bs << cqCtx.cqn;
    bs << cqCtx.cqVa;
    bs << cqCtx.cqeSize;
    bs << cqCtx.cqDepth;
    bs << cqCtx.headAddr;
    bs << cqCtx.tailAddr;
    bs << cqCtx.dbSwVa;
    bs << cqCtx.dbVendorSpecified;
    std::vector<char> result;
    bs.Dump(result);
    return result;
}

static std::vector<char>
BuildSingleConnUniqueId(u32 dmaMode, const RdmaSqContextLite& sqCtx, const RdmaCqContextLite& cqCtx)
{
    BinaryStream bs;
    bs << dmaMode;

    std::vector<char> sqUniqueId = BuildSqUniqueId(sqCtx);
    bs << sqUniqueId;

    std::vector<char> cqUniqueId = BuildCqUniqueId(cqCtx);
    bs << cqUniqueId;

    std::vector<char> result;
    bs.Dump(result);
    return result;
}

static std::vector<char> BuildConnUniqueIds(u32 connNum)
{
    std::vector<char> result;
    for (u32 i = 0; i < connNum; i++) {
        RdmaSqContextLite sq{};
        sq.qpn = 1 + i;
        sq.sqVa = 0x10000;
        sq.wqeSize = 64;
        sq.depth = 128;
        sq.headAddr = 0x20000;
        sq.tailAddr = 0x20008;
        sq.sl = 0;
        sq.dbHwVa = 0x30000;
        sq.dbSwVa = 0x70000;
        sq.dbVendorSpecified = 0;
        sq.dbVendorSpecified
            |= (static_cast<uint64_t>(static_cast<uint8_t>(3) & UB_DB_VENDOR_FIELD_MASK)
                << UB_DB_VENDOR_MTUSHIFT_SHIFT);
        sq.dbVendorSpecified
            |= (static_cast<uint64_t>(static_cast<uint8_t>(0x7) & UB_DB_VENDOR_FIELD_MASK) << UB_DB_VENDOR_COS_SHIFT);

        RdmaCqContextLite cq{};
        cq.cqn = 2 + i;
        cq.cqVa = 0x40000;
        cq.cqeSize = 32;
        cq.cqDepth = 256;
        cq.headAddr = 0x50000;
        cq.tailAddr = 0x50008;
        cq.dbSwVa = 0x60000;

        std::vector<char> singleConn = BuildSingleConnUniqueId(1, sq, cq);
        result.insert(result.end(), singleConn.begin(), singleConn.end());
    }
    return result;
}

static std::vector<char> BuildTransportUniqueId(u32 notifyNum, u32 bufferNum, u32 connNum)
{
    BinaryStream bs;
    u32 type = ROCE_TYPE;
    bs << type;
    bs << notifyNum;
    bs << bufferNum;
    bs << connNum;

    std::vector<char> locNotifyIds = BuildLocNotifyUniqueIds(notifyNum);
    bs << locNotifyIds;

    std::vector<char> rmtNotifyIds = BuildRmtNotifyUniqueIds(notifyNum);
    bs << rmtNotifyIds;

    std::vector<char> notifyValueId = BuildNotifyValueBufferUniqueId();
    bs << notifyValueId;

    std::vector<char> locBufferIds = BuildLocBufferUniqueIds(bufferNum);
    bs << locBufferIds;

    std::vector<char> rmtBufferIds = BuildRmtBufferUniqueIds(bufferNum);
    bs << rmtBufferIds;

    std::vector<char> connIds = BuildConnUniqueIds(connNum);
    bs << connIds;

    std::vector<char> result;
    bs.Dump(result);
    return result;
}

class RoceTransportLiteImplTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RoceTransportLiteImplTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "RoceTransportLiteImplTest tests tear down." << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in RoceTransportLiteImplTest SetUP" << std::endl;
        MOCKER_CPP(&RtsqBase::QuerySqBaseAddr)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(reinterpret_cast<u64>(&mockSq)));
        MOCKER_CPP(&RtsqBase::QuerySqDepth)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(static_cast<u32>(AC_SQE_MAX_CNT)));
        MOCKER_CPP(&RtsqBase::QuerySqStatusByType).stubs().with(mockcpp::any()).will(returnValue(static_cast<u32>(0)));
        MOCKER_CPP(&RtsqBase::ConfigSqStatusByType).stubs();
        uniqueId_ = BuildTransportUniqueId(NOTIFY_NUM, BUFFER_NUM, CONN_NUM);
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in RoceTransportLiteImplTest TearDown" << std::endl;
    }

    u32 NOTIFY_NUM = 2;
    u32 BUFFER_NUM = 1;
    u32 CONN_NUM = 1;
    std::vector<char> uniqueId_;
    u8 mockSq[AC_SQE_SIZE * AC_SQE_MAX_CNT]{0};
};

TEST_F(RoceTransportLiteImplTest, Ut_When_Construct_Expect_Success)
{
    std::cout << "Start Ut_When_Construct_Expect_Success" << std::endl;

    RoceTransportLiteImpl transport(uniqueId_);

    std::cout << "End Ut_When_Construct_Expect_Success" << std::endl;
}

TEST_F(RoceTransportLiteImplTest, Ut_When_Describe_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_Describe_Expect_NotEmpty" << std::endl;

    RoceTransportLiteImpl transport(uniqueId_);

    std::string desc = transport.Describe();
    EXPECT_FALSE(desc.empty());

    std::cout << "End Ut_When_Describe_Expect_NotEmpty" << std::endl;
}

TEST_F(RoceTransportLiteImplTest, Ut_When_NotifyNum_Expect_Correct)
{
    std::cout << "Start Ut_When_NotifyNum_Expect_Correct" << std::endl;

    RoceTransportLiteImpl transport(uniqueId_);

    EXPECT_EQ(transport.notifyNum_, NOTIFY_NUM);

    std::cout << "End Ut_When_NotifyNum_Expect_Correct" << std::endl;
}

TEST_F(RoceTransportLiteImplTest, Ut_When_BufferNum_Expect_Correct)
{
    std::cout << "Start Ut_When_BufferNum_Expect_Correct" << std::endl;

    RoceTransportLiteImpl transport(uniqueId_);

    EXPECT_EQ(transport.bufferNum_, BUFFER_NUM);

    std::cout << "End Ut_When_BufferNum_Expect_Correct" << std::endl;
}

TEST_F(RoceTransportLiteImplTest, Ut_When_ConnNum_Expect_Correct)
{
    std::cout << "Start Ut_When_ConnNum_Expect_Correct" << std::endl;

    RoceTransportLiteImpl transport(uniqueId_);

    EXPECT_EQ(transport.connNum_, CONN_NUM);

    std::cout << "End Ut_When_ConnNum_Expect_Correct" << std::endl;
}

TEST_F(RoceTransportLiteImplTest, Ut_When_LocalNotifies_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_LocalNotifies_Expect_NotEmpty" << std::endl;

    RoceTransportLiteImpl transport(uniqueId_);

    EXPECT_EQ(transport.localNotifies_.size(), static_cast<size_t>(NOTIFY_NUM));

    std::cout << "End Ut_When_LocalNotifies_Expect_NotEmpty" << std::endl;
}

TEST_F(RoceTransportLiteImplTest, Ut_When_RemoteNotifies_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_RemoteNotifies_Expect_NotEmpty" << std::endl;

    RoceTransportLiteImpl transport(uniqueId_);

    EXPECT_EQ(transport.remoteNotifies_.size(), static_cast<size_t>(NOTIFY_NUM));

    std::cout << "End Ut_When_RemoteNotifies_Expect_NotEmpty" << std::endl;
}

TEST_F(RoceTransportLiteImplTest, Ut_When_LocBufferVec_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_LocBufferVec_Expect_NotEmpty" << std::endl;

    RoceTransportLiteImpl transport(uniqueId_);

    EXPECT_EQ(transport.locBufferVec_.size(), static_cast<size_t>(BUFFER_NUM));

    std::cout << "End Ut_When_LocBufferVec_Expect_NotEmpty" << std::endl;
}

TEST_F(RoceTransportLiteImplTest, Ut_When_RmtBufferVec_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_RmtBufferVec_Expect_NotEmpty" << std::endl;

    RoceTransportLiteImpl transport(uniqueId_);

    EXPECT_EQ(transport.rmtBufferVec_.size(), static_cast<size_t>(BUFFER_NUM));

    std::cout << "End Ut_When_RmtBufferVec_Expect_NotEmpty" << std::endl;
}

TEST_F(RoceTransportLiteImplTest, Ut_When_ConnVec_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_ConnVec_Expect_NotEmpty" << std::endl;

    RoceTransportLiteImpl transport(uniqueId_);

    EXPECT_EQ(transport.connVec_.size(), static_cast<size_t>(CONN_NUM));

    std::cout << "End Ut_When_ConnVec_Expect_NotEmpty" << std::endl;
}

TEST_F(RoceTransportLiteImplTest, Ut_When_ConnUniqueIdVec_Expect_NotEmpty)
{
    std::cout << "Start Ut_When_ConnUniqueIdVec_Expect_NotEmpty" << std::endl;

    RoceTransportLiteImpl transport(uniqueId_);

    EXPECT_EQ(transport.connUniqueIdVec_.size(), static_cast<size_t>(CONN_NUM));

    std::cout << "End Ut_When_ConnUniqueIdVec_Expect_NotEmpty" << std::endl;
}

TEST_F(RoceTransportLiteImplTest, Ut_When_NotifyValueBuffer_Expect_NotNull)
{
    std::cout << "Start Ut_When_NotifyValueBuffer_Expect_NotNull" << std::endl;

    RoceTransportLiteImpl transport(uniqueId_);

    EXPECT_NE(transport.notifyValueBuffer_, nullptr);

    std::cout << "End Ut_When_NotifyValueBuffer_Expect_NotNull" << std::endl;
}

TEST_F(RoceTransportLiteImplTest, Ut_When_LocalNotifyValues_Expect_Correct)
{
    std::cout << "Start Ut_When_LocalNotifyValues_Expect_Correct" << std::endl;

    RoceTransportLiteImpl transport(uniqueId_);

    ASSERT_EQ(transport.localNotifies_.size(), static_cast<size_t>(NOTIFY_NUM));
    for (u32 i = 0; i < NOTIFY_NUM; i++) {
        EXPECT_EQ(transport.localNotifies_[i]->GetId(), i + 1);
        EXPECT_EQ(transport.localNotifies_[i]->GetDevPhyId(), i + 10);
    }

    std::cout << "End Ut_When_LocalNotifyValues_Expect_Correct" << std::endl;
}

TEST_F(RoceTransportLiteImplTest, Ut_When_RemoteNotifyValues_Expect_Correct)
{
    std::cout << "Start Ut_When_RemoteNotifyValues_Expect_Correct" << std::endl;

    RoceTransportLiteImpl transport(uniqueId_);

    ASSERT_EQ(transport.remoteNotifies_.size(), static_cast<size_t>(NOTIFY_NUM));
    for (u32 i = 0; i < NOTIFY_NUM; i++) {
        EXPECT_EQ(transport.remoteNotifies_[i].GetAddr(), 0x2000 + i * 0x100);
        EXPECT_EQ(transport.remoteNotifies_[i].GetSize(), 0x100u);
        EXPECT_EQ(transport.remoteNotifies_[i].GetRkey(), 0x200 + i);
    }

    std::cout << "End Ut_When_RemoteNotifyValues_Expect_Correct" << std::endl;
}

TEST_F(RoceTransportLiteImplTest, Ut_When_NotifyValueBufferValues_Expect_Correct)
{
    std::cout << "Start Ut_When_NotifyValueBufferValues_Expect_Correct" << std::endl;

    RoceTransportLiteImpl transport(uniqueId_);

    ASSERT_NE(transport.notifyValueBuffer_, nullptr);
    EXPECT_EQ(transport.notifyValueBuffer_->GetAddr(), 0x3000u);
    EXPECT_EQ(transport.notifyValueBuffer_->GetSize(), 0x200u);
    EXPECT_EQ(transport.notifyValueBuffer_->GetLkey(), 0x300u);

    std::cout << "End Ut_When_NotifyValueBufferValues_Expect_Correct" << std::endl;
}

TEST_F(RoceTransportLiteImplTest, Ut_When_LocBufferValues_Expect_Correct)
{
    std::cout << "Start Ut_When_LocBufferValues_Expect_Correct" << std::endl;

    RoceTransportLiteImpl transport(uniqueId_);

    ASSERT_EQ(transport.locBufferVec_.size(), static_cast<size_t>(BUFFER_NUM));
    for (u32 i = 0; i < BUFFER_NUM; i++) {
        EXPECT_EQ(transport.locBufferVec_[i].GetAddr(), 0x4000 + i * 0x200);
        EXPECT_EQ(transport.locBufferVec_[i].GetSize(), 0x200u);
        EXPECT_EQ(transport.locBufferVec_[i].GetLkey(), 0x400 + i);
    }

    std::cout << "End Ut_When_LocBufferValues_Expect_Correct" << std::endl;
}

TEST_F(RoceTransportLiteImplTest, Ut_When_RmtBufferValues_Expect_Correct)
{
    std::cout << "Start Ut_When_RmtBufferValues_Expect_Correct" << std::endl;

    RoceTransportLiteImpl transport(uniqueId_);

    ASSERT_EQ(transport.rmtBufferVec_.size(), static_cast<size_t>(BUFFER_NUM));
    for (u32 i = 0; i < BUFFER_NUM; i++) {
        EXPECT_EQ(transport.rmtBufferVec_[i].GetAddr(), 0x5000 + i * 0x200);
        EXPECT_EQ(transport.rmtBufferVec_[i].GetSize(), 0x200u);
        EXPECT_EQ(transport.rmtBufferVec_[i].GetRkey(), 0x600 + i);
    }

    std::cout << "End Ut_When_RmtBufferValues_Expect_Correct" << std::endl;
}

TEST_F(RoceTransportLiteImplTest, Ut_When_ConnVecValues_Expect_Correct)
{
    std::cout << "Start Ut_When_ConnVecValues_Expect_Correct" << std::endl;

    RoceTransportLiteImpl transport(uniqueId_);

    ASSERT_EQ(transport.connVec_.size(), static_cast<size_t>(CONN_NUM));
    for (u32 i = 0; i < CONN_NUM; i++) {
        EXPECT_NE(transport.connVec_[i], nullptr);
        EXPECT_EQ(transport.connVec_[i]->dmaMode_, 1u);
        EXPECT_EQ(transport.connVec_[i]->sqContext.qpn, 1u + i);
        EXPECT_EQ(transport.connVec_[i]->cqContext.cqn, 2u + i);
    }

    std::cout << "End Ut_When_ConnVecValues_Expect_Correct" << std::endl;
}

TEST_F(RoceTransportLiteImplTest, Ut_RoceTransportLite_Write)
{
    std::cout << "Start Ut_RoceTransportLite_Write" << std::endl;

    RoceTransportLiteImpl transport(uniqueId_);

    ASSERT_EQ(transport.connVec_.size(), static_cast<size_t>(CONN_NUM));

    RmaBufferLite locBuf(0x4000, 2048, 0x400, 0);
    Hccl::Buffer rmtBuf{0x5000, 0x200};

    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(static_cast<void (RdmaConnLiteV2::*)(
                   const RmaBufSliceLite&, const RmtRmaBufSliceLite&, const SqeConfigLite&, unsigned long long&,
                   unsigned long long&)>(&RdmaConnLiteV2::Write))
        .expects(once());
    MOCKER_CPP(&RdmaConnLiteV2::PollCq)
        .expects(once())
        .with(eq(1), eq(5), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    EXPECT_NO_THROW(transport.Write(locBuf, rmtBuf, stream));
    TaskInfoCircularQueue* queue = stream.GetTaskInfos();
    ASSERT_NE(queue, nullptr);
    ASSERT_FALSE(queue->IsEmpty());
    DfxTaskInfo* slot = queue->GetSlot(0);
    ASSERT_NE(slot, nullptr);
    EXPECT_EQ(slot->taskType, static_cast<u8>(TaskParamTypeVal::TASK_RDMA));
    EXPECT_EQ(slot->linkType, static_cast<u8>(DfxLinkTypeVal::LINK_ROCE));
    EXPECT_EQ(slot->transportType, static_cast<u8>(DfxTransportType::DFX_TRANSPORT_TYPE_ROCE));

    std::cout << "End Ut_RoceTransportLite_Write" << std::endl;
}

TEST_F(RoceTransportLiteImplTest, Ut_RoceTransportLite_WriteWithNotify)
{
    std::cout << "Start Ut_RoceTransportLite_WriteWithNotify" << std::endl;

    RoceTransportLiteImpl transport(uniqueId_);

    ASSERT_EQ(transport.connVec_.size(), static_cast<size_t>(CONN_NUM));

    RmaBufferLite locBuf(0x4000, 2048, 0x400, 0);
    Hccl::Buffer rmtBuf{0x5000, 0x200};
    uint32_t remoteNotifyIdx = 0;
    Hccl::WithNotifyIn withNotify{Hccl::TransportNotifyType::NORMAL, remoteNotifyIdx};

    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(static_cast<void (RdmaConnLiteV2::*)(
                   const RmaBufSliceLite&, const RmtRmaBufSliceLite&, const RmaBufSliceLite&, const RmtRmaBufSliceLite&,
                   const SqeConfigLite&, unsigned long long&, unsigned long long&)>(&RdmaConnLiteV2::WriteWithNotify))
        .expects(once());
    MOCKER_CPP(&RdmaConnLiteV2::PollCq)
        .expects(once())
        .with(eq(2), eq(5), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    EXPECT_NO_THROW(transport.WriteWithNotify(locBuf, rmtBuf, withNotify, stream));
    TaskInfoCircularQueue* queue = stream.GetTaskInfos();
    ASSERT_NE(queue, nullptr);
    ASSERT_FALSE(queue->IsEmpty());
    DfxTaskInfo* slot = queue->GetSlot(0);
    ASSERT_NE(slot, nullptr);
    EXPECT_EQ(slot->taskType, static_cast<u8>(TaskParamTypeVal::TASK_WRITE_WITH_NOTIFY));
    EXPECT_EQ(slot->linkType, static_cast<u8>(DfxLinkTypeVal::LINK_ROCE));
    EXPECT_EQ(slot->transportType, static_cast<u8>(DfxTransportType::DFX_TRANSPORT_TYPE_ROCE));

    std::cout << "End Ut_RoceTransportLite_WriteWithNotify" << std::endl;
}

TEST_F(RoceTransportLiteImplTest, Ut_RoceTransportLite_Post)
{
    std::cout << "Start Ut_RoceTransportLite_Post" << std::endl;

    RoceTransportLiteImpl transport(uniqueId_);

    ASSERT_EQ(transport.connVec_.size(), static_cast<size_t>(CONN_NUM));

    uint32_t remoteNotifyIdx = 0;

    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(static_cast<void (RdmaConnLiteV2::*)(
                   const RmaBufSliceLite&, const RmtRmaBufSliceLite&, const SqeConfigLite&, unsigned long long&,
                   unsigned long long&)>(&RdmaConnLiteV2::Write))
        .expects(once());
    MOCKER_CPP(&RdmaConnLiteV2::PollCq)
        .expects(once())
        .with(eq(1), eq(5), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    EXPECT_NO_THROW(transport.Post(remoteNotifyIdx, stream));
    TaskInfoCircularQueue* queue = stream.GetTaskInfos();
    ASSERT_NE(queue, nullptr);
    ASSERT_FALSE(queue->IsEmpty());
    DfxTaskInfo* slot = queue->GetSlot(0);
    ASSERT_NE(slot, nullptr);
    EXPECT_EQ(slot->taskType, static_cast<u8>(TaskParamTypeVal::TASK_RDMA));
    EXPECT_EQ(slot->linkType, static_cast<u8>(DfxLinkTypeVal::LINK_ROCE));
    EXPECT_EQ(slot->transportType, static_cast<u8>(DfxTransportType::DFX_TRANSPORT_TYPE_ROCE));

    std::cout << "End Ut_RoceTransportLite_Post" << std::endl;
}

TEST_F(RoceTransportLiteImplTest, Ut_RoceTransportLite_Fence)
{
    std::cout << "Start Ut_RoceTransportLite_Fence" << std::endl;

    RoceTransportLiteImpl transport(uniqueId_);

    EXPECT_FALSE(transport.fence_);
    EXPECT_EQ(transport.Fence(), HCCL_SUCCESS);
    EXPECT_TRUE(transport.fence_);

    std::cout << "End Ut_RoceTransportLite_Fence" << std::endl;
}

TEST_F(RoceTransportLiteImplTest, Ut_RoceTransportLite_Read)
{
    RoceTransportLiteImpl transport(uniqueId_);
    RmaBufferLite locBuf(0x4000, 0x200, 0x400, 0);
    Hccl::Buffer rmtBuf{0x5000, 0x200};
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);

    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(static_cast<void (RdmaConnLiteV2::*)(
                   const RmaBufSliceLite&, const RmtRmaBufSliceLite&, const SqeConfigLite&, unsigned long long&,
                   unsigned long long&)>(&RdmaConnLiteV2::Read))
        .expects(once());
    MOCKER_CPP(&RdmaConnLiteV2::PollCq)
        .expects(once())
        .with(eq(1), eq(5), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    EXPECT_NO_THROW(transport.Read(locBuf, rmtBuf, stream));
    TaskInfoCircularQueue* queue = stream.GetTaskInfos();
    ASSERT_NE(queue, nullptr);
    ASSERT_FALSE(queue->IsEmpty());
    DfxTaskInfo* slot = queue->GetSlot(0);
    ASSERT_NE(slot, nullptr);
    EXPECT_EQ(slot->taskType, static_cast<u8>(TaskParamTypeVal::TASK_RDMA));
    EXPECT_EQ(slot->linkType, static_cast<u8>(DfxLinkTypeVal::LINK_ROCE));
    EXPECT_EQ(slot->transportType, static_cast<u8>(DfxTransportType::DFX_TRANSPORT_TYPE_ROCE));
}

TEST_F(RoceTransportLiteImplTest, Ut_RoceTransportLite_WriteReduce)
{
    RoceTransportLiteImpl transport(uniqueId_);
    RmaBufferLite locBuf(0x4000, 0x200, 0x400, 0);
    Hccl::Buffer rmtBuf{0x5000, 0x200};
    ReduceIn reduceIn(DataType::FP32, ReduceOp::SUM);
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);

    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(static_cast<void (RdmaConnLiteV2::*)(
                   const RmaBufSliceLite&, const RmtRmaBufSliceLite&, const SqeConfigLite&, DataType, ReduceOp,
                   unsigned long long&, unsigned long long&)>(&RdmaConnLiteV2::WriteReduce))
        .expects(once())
        .with(
            mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(),
            mockcpp::any());
    MOCKER_CPP(&RdmaConnLiteV2::PollCq)
        .expects(once())
        .with(eq(1), eq(5), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    EXPECT_NO_THROW(transport.WriteReduce(locBuf, rmtBuf, reduceIn, stream));
    TaskInfoCircularQueue* queue = stream.GetTaskInfos();
    ASSERT_NE(queue, nullptr);
    ASSERT_FALSE(queue->IsEmpty());
    DfxTaskInfo* slot = queue->GetSlot(0);
    ASSERT_NE(slot, nullptr);
    EXPECT_EQ(slot->taskType, static_cast<u8>(TaskParamTypeVal::TASK_REDUCE_INLINE));
    EXPECT_EQ(slot->linkType, static_cast<u8>(DfxLinkTypeVal::LINK_ROCE));
    EXPECT_EQ(slot->transportType, static_cast<u8>(DfxTransportType::DFX_TRANSPORT_TYPE_ROCE));
    EXPECT_EQ(slot->taskPara.Reduce.srcAddr, 0x4000u);
    EXPECT_EQ(slot->taskPara.Reduce.dstAddr, 0x5000u);
    EXPECT_EQ(slot->taskPara.Reduce.size, 0x200u);
}

TEST_F(RoceTransportLiteImplTest, Ut_RoceTransportLite_WriteReduceWithNotify)
{
    RoceTransportLiteImpl transport(uniqueId_);
    RmaBufferLite locBuf(0x4000, 0x200, 0x400, 0);
    Hccl::Buffer rmtBuf{0x5000, 0x200};
    ReduceIn reduceIn(DataType::FP32, ReduceOp::MAX);
    Hccl::WithNotifyIn withNotify{Hccl::TransportNotifyType::NORMAL, 0};
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);

    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(static_cast<void (RdmaConnLiteV2::*)(
                   const RmaBufSliceLite&, const RmtRmaBufSliceLite&, const RmaBufSliceLite&, const RmtRmaBufSliceLite&,
                   const SqeConfigLite&, DataType, ReduceOp, unsigned long long&, unsigned long long&)>(
                   &RdmaConnLiteV2::WriteReduceWithNotify))
        .expects(once())
        .with(
            mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(),
            mockcpp::any(), mockcpp::any(), mockcpp::any());
    MOCKER_CPP(&RdmaConnLiteV2::PollCq)
        .expects(once())
        .with(eq(2), eq(5), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    EXPECT_NO_THROW(transport.WriteReduceWithNotify(locBuf, rmtBuf, reduceIn, withNotify, stream));
    TaskInfoCircularQueue* queue = stream.GetTaskInfos();
    ASSERT_NE(queue, nullptr);
    ASSERT_FALSE(queue->IsEmpty());
    DfxTaskInfo* slot = queue->GetSlot(0);
    ASSERT_NE(slot, nullptr);
    EXPECT_EQ(slot->taskType, static_cast<u8>(TaskParamTypeVal::TASK_WRITE_REDUCE_WITH_NOTIFY));
    EXPECT_EQ(slot->linkType, static_cast<u8>(DfxLinkTypeVal::LINK_ROCE));
    EXPECT_EQ(slot->transportType, static_cast<u8>(DfxTransportType::DFX_TRANSPORT_TYPE_ROCE));
    EXPECT_EQ(slot->taskPara.Reduce.srcAddr, 0x4000u);
    EXPECT_EQ(slot->taskPara.Reduce.dstAddr, 0x5000u);
    EXPECT_EQ(slot->taskPara.Reduce.size, 0x200u);
}

TEST_F(RoceTransportLiteImplTest, Ut_RoceTransportLite_PollCq_Expect_ResultPropagated)
{
    RoceTransportLiteImpl transport(uniqueId_);
    std::vector<int32_t> errList;

    MOCKER_CPP(&RdmaConnLiteV2::PollCq)
        .expects(once())
        .with(eq(1), eq(5), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_REMOTE));
    EXPECT_EQ(transport.PollCq(1, 5, errList), HCCL_E_REMOTE);
}

TEST_F(RoceTransportLiteImplTest, Ut_RoceTransportLite_WaitWithTimeout)
{
    RoceTransportLiteImpl transport(uniqueId_);
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);

    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));

    EXPECT_NO_THROW(transport.WaitWithTimeout(0, stream, 1800));
}

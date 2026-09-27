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
#include <memory>

#define private public

#include "udma_data_struct.h"
#include "ub_conn_lite.h"
#include "rmt_rma_buffer_lite.h"
#include "stream_lite.h"
#include "notify_lite.h"
#include "rma_buffer_lite.h"

#include "ins_to_sqe_rule.h"
#include "rtsq_a5.h"
#include "orion_adapter_rts.h"

#undef private

using namespace Hccl;

class LiteResTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "LiteResTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "LiteResTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_910A2));
        MOCKER_CPP(&RtsqBase::QuerySqBaseAddr)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(reinterpret_cast<u64>(&mockSq)));
        MOCKER_CPP(&RtsqBase::QuerySqStatusByType).stubs().with(mockcpp::any()).will(returnValue(static_cast<u32>(0)));
        MOCKER_CPP(&RtsqBase::ConfigSqStatusByType).stubs();
        std::cout << "A Test case in LiteResTest SetUp" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in LiteResTest TearDown" << std::endl;
    }
    u32 fakeStreamId = 0;
    u32 fakeSqId = 0;
    u32 fakedevPhyId = 0;

    u32 fakeNotifyId = 1;
    u32 fakeNotifyDevPhyId = 1;

    u8 mockSq[AC_SQE_SIZE * AC_SQE_MAX_CNT]{0};
};

TEST_F(LiteResTest, test_stream_lite)
{
    BinaryStream liteBinaryStream;
    liteBinaryStream << fakeStreamId;
    liteBinaryStream << fakeSqId;
    liteBinaryStream << fakedevPhyId;
    std::vector<char> uniqueId{};
    liteBinaryStream.Dump(uniqueId);

    StreamLite stream(uniqueId);
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    stream.rtsq = std::make_unique<RtsqA5>(rtsq);
    MOCKER_CPP_VIRTUAL(rtsq, &RtsqA5::SdmaCopy)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any());

    EXPECT_EQ(fakeStreamId, stream.GetId());
    EXPECT_EQ(fakeSqId, stream.GetSqId());
    EXPECT_EQ(fakedevPhyId, stream.GetDevPhyId());
    stream.Describe();
}

TEST_F(LiteResTest, test_notify_lite)
{
    BinaryStream binaryStream;
    binaryStream << fakeNotifyId;
    binaryStream << fakeNotifyDevPhyId;
    std::vector<char> result;
    binaryStream.Dump(result);

    NotifyLite lite(result);
    EXPECT_EQ(fakeNotifyId, lite.GetId());
    EXPECT_EQ(fakeNotifyDevPhyId, lite.GetDevPhyId());
    lite.Describe();
}

TEST_F(LiteResTest, test_rma_buffer_lite)
{
    RmaBufferLite ipc(1, 1);
    EXPECT_EQ(1, ipc.GetAddr());
    EXPECT_EQ(1, ipc.GetSize());

    RmaBufferLite rdma(1, 1, 1);
    EXPECT_EQ(1, rdma.GetAddr());
    EXPECT_EQ(1, rdma.GetSize());
    EXPECT_EQ(1, rdma.GetLkey());

    RmaBufferLite ub(1, 1, 1, 1);
    EXPECT_EQ(1, ub.GetAddr());
    EXPECT_EQ(1, ub.GetSize());
    EXPECT_EQ(1, ub.GetTokenId());
    EXPECT_EQ(1, ub.GetTokenValue());

    EXPECT_EQ(1, ub.GetRmaBufSliceLite(0, 1).GetTokenId());
}

TEST_F(LiteResTest, test_rmt_rma_buffer_lite)
{
    RmtRmaBufferLite ipc(1, 1);
    EXPECT_EQ(1, ipc.GetAddr());
    EXPECT_EQ(1, ipc.GetSize());
    ipc.Describe();

    RmtRmaBufferLite rdma(1, 1, 1);
    EXPECT_EQ(1, rdma.GetAddr());
    EXPECT_EQ(1, rdma.GetSize());
    EXPECT_EQ(1, rdma.GetRkey());

    RmtRmaBufferLite ub(1, 1, 1, 1);
    EXPECT_EQ(1, ub.GetAddr());
    EXPECT_EQ(1, ub.GetSize());
    EXPECT_EQ(1, ub.GetTokenId());
    EXPECT_EQ(1, ub.GetTokenValue());

    EXPECT_EQ(1, ub.GetRmtRmaBufSliceLite(0, 1).GetTokenId());
}

TEST_F(LiteResTest, test_RmaBufSliceLite)
{
    RmaBufSliceLite lite(1, 1, 1, 1);
    EXPECT_EQ(1, lite.GetAddr());
    EXPECT_EQ(1, lite.GetSize());
    EXPECT_EQ(1, lite.GetLkey());
    EXPECT_EQ(1, lite.GetTokenId());
    lite.Describe();
}

TEST_F(LiteResTest, test_RmtRmaBufSliceLite)
{
    RmtRmaBufSliceLite lite(1, 1, 1, 1, 1, UINT32_MAX);
    EXPECT_EQ(1, lite.GetAddr());
    EXPECT_EQ(1, lite.GetSize());
    EXPECT_EQ(1, lite.GetRkey());
    EXPECT_EQ(1, lite.GetTokenId());
    EXPECT_EQ(1, lite.GetTokenValue());
}

TEST_F(LiteResTest, test_RmaConnLite)
{
    RmaConnLite rdma(1);
    EXPECT_EQ(1, rdma.GetQpVa());

    UbJettyLiteId id(1, 1, 1);
    UbJettyLiteAttr attr(1, 1, 1, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    RmaConnLite ub(id, attr, rmtEid);

    EXPECT_EQ(1, ub.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ub.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ub.GetRmtEid().raw[0]);
}

TEST_F(LiteResTest, test_UBConnLite_Read)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 4096, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);

    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));

    RmaBufSliceLite loc(0x1111, 64, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 64, 1, 1, 1, UINT32_MAX);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    EXPECT_NO_THROW(ubConn.Read(loc, rmt, cfg, stream, out));
}

TEST_F(LiteResTest, test_UBConnLite_ReadReduce)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 4096, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);

    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));

    RmaBufSliceLite loc(0x1111, 64, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 64, 1, 1, 1, UINT32_MAX);
    RmtRmaBufSliceLite notify(0x2222, 64, 1, 1, 1, UINT32_MAX);
    u64 notifyData(1);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    ReduceIn reduceIn(DataType::INT8, ReduceOp::SUM);
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    EXPECT_NO_THROW(ubConn.ReadReduce(reduceIn, loc, rmt, stream, cfg, out));
}

TEST_F(LiteResTest, test_UBConnLite)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 4096, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);

    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));

    RmaBufSliceLite loc(0x1111, 64, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 64, 1, 1, 1, UINT32_MAX);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    EXPECT_NO_THROW(ubConn.Write(loc, rmt, cfg, stream, out));
}

TEST_F(LiteResTest, test_UBConnLite_WriteReduceWithNotify)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 4096, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);

    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);

    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));

    RmaBufSliceLite loc(0x1111, 64, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 64, 1, 1, 1, UINT32_MAX);
    RmtRmaBufSliceLite notify(0x2222, 64, 1, 1, 1, UINT32_MAX);
    u64 notifyData(1);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    EXPECT_NO_THROW(
        ubConn.WriteReduceWithNotify(DataType::INT8, ReduceOp::SUM, loc, rmt, cfg, stream, out, notify, notifyData));
    EXPECT_NO_THROW(
        ubConn.WriteReduceWithNotify(DataType::INT16, ReduceOp::MAX, loc, rmt, cfg, stream, out, notify, notifyData));
    EXPECT_NO_THROW(
        ubConn.WriteReduceWithNotify(DataType::INT32, ReduceOp::MIN, loc, rmt, cfg, stream, out, notify, notifyData));
    EXPECT_NO_THROW(
        ubConn.WriteReduceWithNotify(DataType::FP16, ReduceOp::SUM, loc, rmt, cfg, stream, out, notify, notifyData));
    EXPECT_NO_THROW(
        ubConn.WriteReduceWithNotify(DataType::FP32, ReduceOp::MAX, loc, rmt, cfg, stream, out, notify, notifyData));
    EXPECT_NO_THROW(
        ubConn.WriteReduceWithNotify(DataType::UINT8, ReduceOp::MAX, loc, rmt, cfg, stream, out, notify, notifyData));
    EXPECT_NO_THROW(
        ubConn.WriteReduceWithNotify(DataType::UINT32, ReduceOp::MAX, loc, rmt, cfg, stream, out, notify, notifyData));
}

TEST_F(LiteResTest, test_UBConnLite_WriteWithNotify)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 4096, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);

    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);

    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));

    RmaBufSliceLite loc(0x1111, 64, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 64, 1, 1, 1, UINT32_MAX);
    RmtRmaBufSliceLite notify(0x2222, 64, 1, 1, 1, UINT32_MAX);
    u64 notifyData(1);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    EXPECT_NO_THROW(ubConn.WriteWithNotify(loc, rmt, cfg, out, notify, stream, notifyData));
}

TEST_F(LiteResTest, test_UBConnLite_InlineWrite)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 4096, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);

    EXPECT_EQ(1, ubConn.GetUbJettyLiteId().GetDieId());
    EXPECT_EQ(1, ubConn.GetUbJettyLiteAttr().dbAddr_);
    EXPECT_EQ(1, ubConn.GetRmtEid().raw[0]);
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));

    RmaBufSliceLite loc(0x1111, 64, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, 64, 1, 1, 1, UINT32_MAX);
    RmtRmaBufSliceLite notify(0x2222, 64, 1, 1, 1, UINT32_MAX);
    u64 notifyData(1);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    u8 data[512];
    out.pi = 0;
    out.data = data;
    UdmaSqeWrite* sqe = (UdmaSqeWrite*)out.data;
    out.dataSize = 64;
    rmt.GetTokenValue();
    u8 write_data(1);
    u16 write_dsize(1);
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    EXPECT_NO_THROW(ubConn.InlineWrite(&write_data, write_dsize, rmt, cfg, stream, out));
}

TEST_F(LiteResTest, test_RmaBufSliceLite_Describe)
{
    RmaBufSliceLite loc(0x1111, 1, 1, 1);
    loc.Describe();
}

TEST_F(LiteResTest, test_UBConnLite_BatchOneSidedRead)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 4, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);

    u32 dataSize = 400 * 1024 * 1024;
    RmaBufSliceLite loc(0x1111, dataSize, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, dataSize, 1, 1, 1);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    ubConn.BatchOneSidedRead({loc}, {rmt}, cfg, stream, out);
}

TEST_F(LiteResTest, test_UBConnLite_BatchOneSidedWrite)
{
    UbJettyLiteId id(1, 1, 1);
    u8 sqVa[1024];
    u64 sqVAaddr = reinterpret_cast<u64>(sqVa);
    UbJettyLiteAttr attr(1, sqVAaddr, 4, 1, false);
    Eid rmtEid;
    rmtEid.raw[0] = 1;

    UbConnLite ubConn(id, attr, rmtEid);

    u32 dataSize = 400 * 1024 * 1024;
    RmaBufSliceLite loc(0x1111, dataSize, 1, 1);
    RmtRmaBufSliceLite rmt(0x2222, dataSize, 1, 1, 1);
    SqeConfigLite cfg;
    ConnLiteOperationOut out;
    std::vector<char> uniqueId{};
    StreamLite stream(uniqueId);
    MOCKER_CPP(&RtsqBase::QuerySqHead).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&RtsqBase::QuerySqTail).stubs().with(mockcpp::any()).will(returnValue(1));
    ubConn.BatchOneSidedWrite({loc}, {rmt}, cfg, stream, out);
}

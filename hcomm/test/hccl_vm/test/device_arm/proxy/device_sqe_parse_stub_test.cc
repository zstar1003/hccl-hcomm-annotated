/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <sys/mman.h>

#include "device_sqe_parse_stub.h"
#include "hccl_device_pub.h"
#include "store_sim_memory_manager.h"
#include "sqe_v82_stub.h"
#include "udma_data_struct_stub.h"
#include "db_sim_runner_db.h"
#include "sim_models.h"

class DeviceSqeParseTest : public testing::Test {
protected:
    void SetUp() override
    {
        SetCurRankId(0);
        RunnerDB::DeleteAll<sim::RaJetty>();
    }

    void TearDown() override
    {
        SetCurRankId(0);
        RunnerDB::DeleteAll<sim::RaJetty>();
    }
};

TEST_F(DeviceSqeParseTest, GetFull64BitAddr_LowOnly)
{
    uint32_t lowAddr = 0x12345678;
    uint32_t highAddr = 0;

    uint64_t result = GetFull64BitAddr(lowAddr, highAddr);
    EXPECT_EQ(result, 0x12345678);
}

TEST_F(DeviceSqeParseTest, GetFull64BitAddr_HighOnly)
{
    uint32_t lowAddr = 0;
    uint32_t highAddr = 0x12345678;

    uint64_t result = GetFull64BitAddr(lowAddr, highAddr);
    EXPECT_EQ(result, 0x1234567800000000ULL);
}

TEST_F(DeviceSqeParseTest, GetFull64BitAddr_Both)
{
    uint32_t lowAddr = 0x12345678;
    uint32_t highAddr = 0x9ABCDEF0;

    uint64_t result = GetFull64BitAddr(lowAddr, highAddr);
    EXPECT_EQ(result, 0x9ABCDEF012345678ULL);
}

TEST_F(DeviceSqeParseTest, GetFull64BitAddr_MaxValues)
{
    uint32_t lowAddr = 0xFFFFFFFF;
    uint32_t highAddr = 0xFFFFFFFF;

    uint64_t result = GetFull64BitAddr(lowAddr, highAddr);
    EXPECT_EQ(result, 0xFFFFFFFFFFFFFFFFULL);
}

TEST_F(DeviceSqeParseTest, GetFull64BitAddr_ZeroValues)
{
    uint32_t lowAddr = 0;
    uint32_t highAddr = 0;

    uint64_t result = GetFull64BitAddr(lowAddr, highAddr);
    EXPECT_EQ(result, 0ULL);
}

TEST_F(DeviceSqeParseTest, ParseReduceTypeDavid_Sum)
{
    uint8_t result = 0x01;
    HcclReduceOp op = ParseReduceTypeDavid(result);
    EXPECT_EQ(op, HcclReduceOp::HCCL_REDUCE_SUM);
}

TEST_F(DeviceSqeParseTest, ParseReduceTypeDavid_Max)
{
    uint8_t result = 0x02;
    HcclReduceOp op = ParseReduceTypeDavid(result);
    EXPECT_EQ(op, HcclReduceOp::HCCL_REDUCE_MAX);
}

TEST_F(DeviceSqeParseTest, ParseReduceTypeDavid_Min)
{
    uint8_t result = 0x03;
    HcclReduceOp op = ParseReduceTypeDavid(result);
    EXPECT_EQ(op, HcclReduceOp::HCCL_REDUCE_MIN);
}

TEST_F(DeviceSqeParseTest, ParseReduceTypeDavid_Invalid)
{
    uint8_t result = 0xFF;
    HcclReduceOp op = ParseReduceTypeDavid(result);
    EXPECT_EQ(op, HcclReduceOp::HCCL_REDUCE_RESERVED);
}

TEST_F(DeviceSqeParseTest, ParseDataTypeDavid_Int8)
{
    uint8_t result = 0x00;
    HcclDataType type = ParseDataTypeDavid(result);
    EXPECT_EQ(type, HcclDataType::HCCL_DATA_TYPE_INT8);
}

TEST_F(DeviceSqeParseTest, ParseDataTypeDavid_Int16)
{
    uint8_t result = 0x10;
    HcclDataType type = ParseDataTypeDavid(result);
    EXPECT_EQ(type, HcclDataType::HCCL_DATA_TYPE_INT16);
}

TEST_F(DeviceSqeParseTest, ParseDataTypeDavid_Int32)
{
    uint8_t result = 0x20;
    HcclDataType type = ParseDataTypeDavid(result);
    EXPECT_EQ(type, HcclDataType::HCCL_DATA_TYPE_INT32);
}

TEST_F(DeviceSqeParseTest, ParseDataTypeDavid_Fp32)
{
    uint8_t result = 0x70;
    HcclDataType type = ParseDataTypeDavid(result);
    EXPECT_EQ(type, HcclDataType::HCCL_DATA_TYPE_FP32);
}

TEST_F(DeviceSqeParseTest, ParseDataTypeDavid_Invalid)
{
    uint8_t result = 0xFF;
    HcclDataType type = ParseDataTypeDavid(result);
    EXPECT_EQ(type, HcclDataType::HCCL_DATA_TYPE_RESERVED);
}

TEST_F(DeviceSqeParseTest, ParseUbReduceTypeDavid_Sum)
{
    uint32_t type = 0xA;
    HcclReduceOp op = ParseUbReduceTypeDavid(type);
    EXPECT_EQ(op, HcclReduceOp::HCCL_REDUCE_SUM);
}

TEST_F(DeviceSqeParseTest, ParseUbReduceTypeDavid_Max)
{
    uint32_t type = 0x8;
    HcclReduceOp op = ParseUbReduceTypeDavid(type);
    EXPECT_EQ(op, HcclReduceOp::HCCL_REDUCE_MAX);
}

TEST_F(DeviceSqeParseTest, ParseUbReduceTypeDavid_Min)
{
    uint32_t type = 0x9;
    HcclReduceOp op = ParseUbReduceTypeDavid(type);
    EXPECT_EQ(op, HcclReduceOp::HCCL_REDUCE_MIN);
}

TEST_F(DeviceSqeParseTest, ParseUbReduceTypeDavid_Invalid)
{
    uint32_t type = 0xFF;
    HcclReduceOp op = ParseUbReduceTypeDavid(type);
    EXPECT_EQ(op, HcclReduceOp::HCCL_REDUCE_RESERVED);
}

TEST_F(DeviceSqeParseTest, ParseUbDataTypeDavid_Int8)
{
    uint32_t type = 0x0;
    HcclDataType dataType = ParseUbDataTypeDavid(type);
    EXPECT_EQ(dataType, HcclDataType::HCCL_DATA_TYPE_INT8);
}

TEST_F(DeviceSqeParseTest, ParseUbDataTypeDavid_Fp32)
{
    uint32_t type = 0x7;
    HcclDataType dataType = ParseUbDataTypeDavid(type);
    EXPECT_EQ(dataType, HcclDataType::HCCL_DATA_TYPE_FP32);
}

TEST_F(DeviceSqeParseTest, ParseUbDataTypeDavid_Invalid)
{
    uint32_t type = 0xFF;
    HcclDataType dataType = ParseUbDataTypeDavid(type);
    EXPECT_EQ(dataType, HcclDataType::HCCL_DATA_TYPE_RESERVED);
}

TEST_F(DeviceSqeParseTest, PrintTaskMetaData_MemCpy)
{
    HcclTaskMetaData taskMeta;
    memset(&taskMeta, 0, sizeof(taskMeta));
    taskMeta.taskType = HccLTaskMetaType::MEM_CPY;
    taskMeta.rankId = 0;
    taskMeta.streamId = 0;
    taskMeta.taskData.transMem.srcOffset = 0x1000;
    taskMeta.taskData.transMem.dstOffset = 0x2000;
    taskMeta.taskData.transMem.len = 1024;

    EXPECT_NO_THROW(PrintTaskMetaData(taskMeta));
}

TEST_F(DeviceSqeParseTest, PrintTaskMetaData_Reduce)
{
    HcclTaskMetaData taskMeta;
    memset(&taskMeta, 0, sizeof(taskMeta));
    taskMeta.taskType = HccLTaskMetaType::REDUCE;
    taskMeta.rankId = 0;
    taskMeta.streamId = 0;

    EXPECT_NO_THROW(PrintTaskMetaData(taskMeta));
}

TEST_F(DeviceSqeParseTest, PrintTaskMetaData_NotifyWait)
{
    HcclTaskMetaData taskMeta;
    memset(&taskMeta, 0, sizeof(taskMeta));
    taskMeta.taskType = HccLTaskMetaType::NOTIFY_WAIT;
    taskMeta.rankId = 0;
    taskMeta.streamId = 0;

    EXPECT_NO_THROW(PrintTaskMetaData(taskMeta));
}

TEST_F(DeviceSqeParseTest, PrintTaskMetaData_NotifyRecord)
{
    HcclTaskMetaData taskMeta;
    memset(&taskMeta, 0, sizeof(taskMeta));
    taskMeta.taskType = HccLTaskMetaType::NOTIFY_RECORD;
    taskMeta.rankId = 0;
    taskMeta.streamId = 0;

    EXPECT_NO_THROW(PrintTaskMetaData(taskMeta));
}

// ==================== ParseDavidSDMASqe Tests ====================

TEST_F(DeviceSqeParseTest, ParseDavidSDMASqe_Memcpy)
{
    // Create a mock SDMA SQE with opcode = 0 (memcpy)
    Rt91095StarsMemcpySqe sqe;
    memset(&sqe, 0, sizeof(sqe));
    sqe.opcode = 0; // memcpy
    sqe.u.strideMode0.lengthMove = 1024;
    sqe.u.strideMode0.srcAddrLow = 0x1000;
    sqe.u.strideMode0.srcAddrHigh = 0;
    sqe.u.strideMode0.dstAddrLow = 0x2000;
    sqe.u.strideMode0.dstAddrHigh = 0;

    // This will call GetRankIdByDevAddr which requires database setup
    // Just verify it doesn't crash
    EXPECT_NO_THROW(ParseDavidSDMASqe(0, &sqe));
}

TEST_F(DeviceSqeParseTest, ParseDavidSDMASqe_Reduce)
{
    // Create a mock SDMA SQE with opcode != 0 (reduce)
    Rt91095StarsMemcpySqe sqe;
    memset(&sqe, 0, sizeof(sqe));
    sqe.opcode = 0x01; // reduce (SUM)
    sqe.u.strideMode0.lengthMove = 1024;
    sqe.u.strideMode0.srcAddrLow = 0x1000;
    sqe.u.strideMode0.srcAddrHigh = 0;
    sqe.u.strideMode0.dstAddrLow = 0x2000;
    sqe.u.strideMode0.dstAddrHigh = 0;

    EXPECT_NO_THROW(ParseDavidSDMASqe(0, &sqe));
}

// ==================== ParseDavidNotifySqe Tests ====================

TEST_F(DeviceSqeParseTest, ParseDavidNotifySqe_Wait)
{
    Rt91095StarsNotifySqe sqe;
    memset(&sqe, 0, sizeof(sqe));
    sqe.notifyId = 123;

    EXPECT_NO_THROW(ParseDavidNotifySqe(0, &sqe, false));
}

TEST_F(DeviceSqeParseTest, ParseDavidNotifySqe_Record)
{
    Rt91095StarsNotifySqe sqe;
    memset(&sqe, 0, sizeof(sqe));
    sqe.notifyId = 456;

    EXPECT_NO_THROW(ParseDavidNotifySqe(0, &sqe, true));
}

// ==================== GetRmtRankIdByEid Tests ====================

TEST_F(DeviceSqeParseTest, GetRmtRankIdByEid_Zero)
{
    // eid = 0 should convert to IP "0.0.0.0"
    uint32_t eid = 0;
    EXPECT_NO_THROW(GetRmtRankIdByEid(eid));
}

TEST_F(DeviceSqeParseTest, GetRmtRankIdByEid_Localhost)
{
    // eid = 0x7F000001 should convert to IP "127.0.0.1"
    uint32_t eid = 0x7F000001;
    EXPECT_NO_THROW(GetRmtRankIdByEid(eid));
}

TEST_F(DeviceSqeParseTest, GetRmtRankIdByEid_ClassA)
{
    // eid = 0x0A0A0A0A should convert to IP "10.10.10.10"
    uint32_t eid = 0x0A0A0A0A;
    EXPECT_NO_THROW(GetRmtRankIdByEid(eid));
}

// ==================== ParseA5SqeFromSqBuffer Tests ====================

TEST_F(DeviceSqeParseTest, ParseA5SqeFromSqBuffer_TailLessThanHead)
{
    // This tests the tail < head branch (data wrap-around)
    // We need to set up a scenario where tail < head
    // Since this function uses global state, we need to be careful

    // Create a mock halSqCqConfigInfo
    struct halSqCqConfigInfo info;
    memset(&info, 0, sizeof(info));
    info.sqId = 0;
    info.value[0] = 5; // tail = 5

    // Set head to a value > tail to trigger the wrap-around path
    // We need to call UpdateSqTail first to set the head
    UpdateSqTail(0, 10); // Set head = 10

    // Now tail (5) < head (10), which should trigger the wrap-around path
    EXPECT_NO_THROW(ParseA5SqeFromSqBuffer(0, &info));
}

TEST_F(DeviceSqeParseTest, ParseA5SqeFromSqBuffer_DefaultSqeType)
{
    struct halSqCqConfigInfo info;
    memset(&info, 0, sizeof(info));
    info.sqId = 0;
    info.value[0] = 1;
    UpdateSqTail(0, 0);
    EXPECT_NO_THROW(ParseA5SqeFromSqBuffer(0, &info));
}

TEST_F(DeviceSqeParseTest, ParseA5SqeFromSqBuffer_SDMAType)
{
    struct halSqCqConfigInfo info;
    memset(&info, 0, sizeof(info));
    info.sqId = 0;
    info.value[0] = 1;
    UpdateSqTail(0, 0);

    uint8_t* sqBuf = nullptr;
    GetSqBufferAddr(&sqBuf);
    if (sqBuf) {
        Rt91095StarsMemcpySqe sqe;
        memset(&sqe, 0, sizeof(sqe));
        sqe.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA);
        sqe.opcode = 0;
        sqe.u.strideMode0.lengthMove = 1024;
        sqe.u.strideMode0.srcAddrLow = 0x1000;
        sqe.u.strideMode0.srcAddrHigh = 0;
        sqe.u.strideMode0.dstAddrLow = 0x2000;
        sqe.u.strideMode0.dstAddrHigh = 0;
        memcpy(sqBuf, &sqe, sizeof(sqe));
    }

    EXPECT_NO_THROW(ParseA5SqeFromSqBuffer(0, &info));
}
TEST_F(DeviceSqeParseTest, ParseA5SqeFromSqBuffer_NotifyWaitType)
{
    struct halSqCqConfigInfo info;
    memset(&info, 0, sizeof(info));
    info.sqId = 0;
    info.value[0] = 1;
    UpdateSqTail(0, 0);

    uint8_t* sqBuf = nullptr;
    GetSqBufferAddr(&sqBuf);
    if (sqBuf) {
        Rt91095StarsNotifySqe sqe;
        memset(&sqe, 0, sizeof(sqe));
        sqe.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
        sqe.notifyId = 100;
        memcpy(sqBuf, &sqe, sizeof(sqe));
    }

    EXPECT_NO_THROW(ParseA5SqeFromSqBuffer(0, &info));
}
TEST_F(DeviceSqeParseTest, ParseA5SqeFromSqBuffer_NotifyRecordType)
{
    struct halSqCqConfigInfo info;
    memset(&info, 0, sizeof(info));
    info.sqId = 0;
    info.value[0] = 1;
    UpdateSqTail(0, 0);

    uint8_t* sqBuf = nullptr;
    GetSqBufferAddr(&sqBuf);
    if (sqBuf) {
        Rt91095StarsNotifySqe sqe;
        memset(&sqe, 0, sizeof(sqe));
        sqe.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD);
        sqe.notifyId = 200;
        memcpy(sqBuf, &sqe, sizeof(sqe));
    }

    EXPECT_NO_THROW(ParseA5SqeFromSqBuffer(0, &info));
}
TEST_F(DeviceSqeParseTest, ParseA5SqeFromSqBuffer_UBDMAType)
{
    struct halSqCqConfigInfo info;
    memset(&info, 0, sizeof(info));
    info.sqId = 0;
    info.value[0] = 1;
    UpdateSqTail(0, 0);

    uint8_t* sqBuf = nullptr;
    GetSqBufferAddr(&sqBuf);
    if (sqBuf) {
        Rt91095StarsUbdmaDBmodeSqe sqe;
        memset(&sqe, 0, sizeof(sqe));
        sqe.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_UBDMA);
        sqe.jettyId1 = 1;
        sqe.piValue1 = 1;
        memcpy(sqBuf, &sqe, sizeof(sqe));
    }

    // With null-check added to ParseDavidUDMASqe, this won't crash anymore
    EXPECT_NO_THROW(ParseA5SqeFromSqBuffer(0, &info));
}

// ==================== ParseDavidUBReadWriteSqe Tests ====================

TEST_F(DeviceSqeParseTest, ParseDavidUBReadWriteSqe_InlineWriteNotify)
{
    UdmaSqeWrite ubWqe;
    memset(&ubWqe, 0, sizeof(ubWqe));
    ubWqe.comm.inlineEn = 1;
    ubWqe.comm.rmtAddrLow = 0x1000;
    ubWqe.comm.rmtAddrHigh = 0;
    ubWqe.comm.rmtEid[0] = 0;

    EXPECT_NO_THROW(ParseDavidUBReadWriteSqe(reinterpret_cast<uint64_t>(&ubWqe), 0, 1, false));
}

TEST_F(DeviceSqeParseTest, ParseDavidUBReadWriteSqe_WriteMemCpy)
{
    UdmaSqeWrite ubWqe;
    memset(&ubWqe, 0, sizeof(ubWqe));
    ubWqe.comm.inlineEn = 0;
    ubWqe.comm.udfFlag = 0;
    ubWqe.comm.rmtAddrLow = 0x2000;
    ubWqe.comm.rmtAddrHigh = 0;
    ubWqe.u.sge.dataAddrLow = 0x1000;
    ubWqe.u.sge.dataAddrHigh = 0;
    ubWqe.u.sge.length = 128;

    EXPECT_NO_THROW(ParseDavidUBReadWriteSqe(reinterpret_cast<uint64_t>(&ubWqe), 0, 1, false));
}

TEST_F(DeviceSqeParseTest, ParseDavidUBReadWriteSqe_WriteMemCpy_WithOffset)
{
    UdmaSqeWrite ubWqe;
    memset(&ubWqe, 0, sizeof(ubWqe));
    ubWqe.comm.inlineEn = 0;
    ubWqe.comm.udfFlag = 0;
    ubWqe.comm.rmtAddrLow = 0x3000;
    ubWqe.comm.rmtAddrHigh = 0;
    ubWqe.u.sge.dataAddrLow = 0x5000;
    ubWqe.u.sge.dataAddrHigh = 0;
    ubWqe.u.sge.length = 256;

    EXPECT_NO_THROW(ParseDavidUBReadWriteSqe(reinterpret_cast<uint64_t>(&ubWqe), 2, 3, false));
}

TEST_F(DeviceSqeParseTest, ParseDavidUBReadWriteSqe_ReadMemCpy)
{
    UdmaSqeWrite ubWqe;
    memset(&ubWqe, 0, sizeof(ubWqe));
    ubWqe.comm.inlineEn = 0;
    ubWqe.comm.udfFlag = 0;
    ubWqe.comm.rmtAddrLow = 0x2000;
    ubWqe.comm.rmtAddrHigh = 0;
    ubWqe.u.sge.dataAddrLow = 0x1000;
    ubWqe.u.sge.dataAddrHigh = 0;
    ubWqe.u.sge.length = 64;

    EXPECT_NO_THROW(ParseDavidUBReadWriteSqe(reinterpret_cast<uint64_t>(&ubWqe), 0, 1, true));
}

TEST_F(DeviceSqeParseTest, ParseDavidUBReadWriteSqe_WriteReduce)
{
    UdmaSqeWrite ubWqe;
    memset(&ubWqe, 0, sizeof(ubWqe));
    ubWqe.comm.inlineEn = 0;
    ubWqe.comm.udfFlag = 1;
    ubWqe.comm.rmtAddrLow = 0x2000;
    ubWqe.comm.rmtAddrHigh = 0;
    ubWqe.comm.inlinedata.udfData.reduceOp = 0xA;
    ubWqe.comm.inlinedata.udfData.reduceType = 0x7;
    ubWqe.u.sge.dataAddrLow = 0x1000;
    ubWqe.u.sge.dataAddrHigh = 0;
    ubWqe.u.sge.length = 1024;

    EXPECT_NO_THROW(ParseDavidUBReadWriteSqe(reinterpret_cast<uint64_t>(&ubWqe), 0, 1, false));
}

TEST_F(DeviceSqeParseTest, ParseDavidUBReadWriteSqe_WriteReduce_WithUdfFlag_Read)
{
    UdmaSqeWrite ubWqe;
    memset(&ubWqe, 0, sizeof(ubWqe));
    ubWqe.comm.inlineEn = 0;
    ubWqe.comm.udfFlag = 1;
    ubWqe.comm.rmtAddrLow = 0x2000;
    ubWqe.comm.rmtAddrHigh = 0;
    ubWqe.comm.inlinedata.udfData.reduceOp = 0xA;
    ubWqe.comm.inlinedata.udfData.reduceType = 0x0;
    ubWqe.u.sge.dataAddrLow = 0x1000;
    ubWqe.u.sge.dataAddrHigh = 0;
    ubWqe.u.sge.length = 512;

    EXPECT_NO_THROW(ParseDavidUBReadWriteSqe(reinterpret_cast<uint64_t>(&ubWqe), 0, 1, true));
}

// ==================== ParseDavidUBWriteWithNotifySqe Tests ====================

TEST_F(DeviceSqeParseTest, ParseDavidUBWriteWithNotify_MemCpy)
{
    UdmaSqeWriteWithNotify ubWqe;
    memset(&ubWqe, 0, sizeof(ubWqe));
    ubWqe.comm.udfFlag = 0;
    ubWqe.comm.rmtAddrLow = 0x2000;
    ubWqe.comm.rmtAddrHigh = 0;
    ubWqe.comm.rmtEid[0] = 0;
    ubWqe.localU.sge.dataAddrLow = 0x1000;
    ubWqe.localU.sge.dataAddrHigh = 0;
    ubWqe.localU.sge.length = 256;
    ubWqe.notify.notifyAddrLow = 0x4000;
    ubWqe.notify.notifyAddrHigh = 0;

    EXPECT_NO_THROW(ParseDavidUBWriteWithNotifySqe(reinterpret_cast<uint64_t>(&ubWqe), 0, 1));
}

TEST_F(DeviceSqeParseTest, ParseDavidUBWriteWithNotify_Reduce)
{
    UdmaSqeWriteWithNotify ubWqe;
    memset(&ubWqe, 0, sizeof(ubWqe));
    ubWqe.comm.udfFlag = 1;
    ubWqe.comm.rmtAddrLow = 0x2000;
    ubWqe.comm.rmtAddrHigh = 0;
    ubWqe.comm.rmtEid[0] = 0;
    ubWqe.comm.inlinedata.udfData.reduceOp = 0xA;
    ubWqe.comm.inlinedata.udfData.reduceType = 0x7;
    ubWqe.localU.sge.dataAddrLow = 0x1000;
    ubWqe.localU.sge.dataAddrHigh = 0;
    ubWqe.localU.sge.length = 512;
    ubWqe.notify.notifyAddrLow = 0x4000;
    ubWqe.notify.notifyAddrHigh = 0;

    EXPECT_NO_THROW(ParseDavidUBWriteWithNotifySqe(reinterpret_cast<uint64_t>(&ubWqe), 0, 2));
}

// ==================== ParseDavidUDMASqe Tests ====================
// With DB-based RaJetty lookup in device_sqe_parse_stub.cc, ParseDavidUDMASqe returns early
// when no corresponding RaJetty exists. We test both missing-jetty and valid-jetty paths.

TEST_F(DeviceSqeParseTest, ParseDavidUDMASqe_NoJettyInDb)
{
    // Without RaJetty in DB, GetWqebufferByJettyId fails
    // ParseDavidUDMASqe should return early with error log
    Rt91095StarsUbdmaDBmodeSqe ubSqe;
    memset(&ubSqe, 0, sizeof(ubSqe));
    ubSqe.jettyId1 = 1;
    ubSqe.piValue1 = 2;

    EXPECT_NO_THROW(ParseDavidUDMASqe(0, &ubSqe));
}

TEST_F(DeviceSqeParseTest, ParseDavidUDMASqe_WriteOpcodeWithShm)
{
    alignas(64) uint8_t wqeBuffer[HCCL_WQE_SIZE * 4] = {};
    sim::RaJetty jetty{};
    jetty.id = 1;
    jetty.sqBuffer = reinterpret_cast<uint64_t>(wqeBuffer);
    RunnerDB::Add<sim::RaJetty>(jetty);

    // Setup WQE with WRITE opcode at ciVal=0
    UdmaSqeWrite* ubWqeWrite = reinterpret_cast<UdmaSqeWrite*>(wqeBuffer);
    memset(ubWqeWrite, 0, sizeof(UdmaSqeWrite));
    ubWqeWrite->comm.opcode = static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE);
    ubWqeWrite->comm.inlineEn = 0;
    ubWqeWrite->comm.udfFlag = 0;
    ubWqeWrite->comm.rmtAddrLow = 0x2000;
    ubWqeWrite->u.sge.dataAddrLow = 0x1000;
    ubWqeWrite->u.sge.length = 128;

    Rt91095StarsUbdmaDBmodeSqe ubSqe;
    memset(&ubSqe, 0, sizeof(ubSqe));
    ubSqe.jettyId1 = 1;
    ubSqe.piValue1 = 2;

    EXPECT_NO_THROW(ParseDavidUDMASqe(0, &ubSqe));
}

TEST_F(DeviceSqeParseTest, ParseDavidUDMASqe_WriteWithNotifyOpcodeWithShm)
{
    alignas(64) uint8_t wqeBuffer[HCCL_WQE_SIZE * 4] = {};
    sim::RaJetty jetty{};
    jetty.id = 1;
    jetty.sqBuffer = reinterpret_cast<uint64_t>(wqeBuffer);
    RunnerDB::Add<sim::RaJetty>(jetty);

    // piValue1=2, ciVal = piValue1-2=0, first WQE has WRITE_WITH_NOTIFY
    UdmaSqeCommon* ubCommon = reinterpret_cast<UdmaSqeCommon*>(wqeBuffer);
    memset(ubCommon, 0, sizeof(UdmaSqeCommon));
    ubCommon->opcode = static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE_WITH_NOTIFY);

    // Setup WriteWithNotify WQE content
    UdmaSqeWriteWithNotify* ubWqe = reinterpret_cast<UdmaSqeWriteWithNotify*>(wqeBuffer);
    ubWqe->comm.udfFlag = 0;
    ubWqe->comm.rmtAddrLow = 0x2000;
    ubWqe->comm.rmtEid[0] = 0;
    ubWqe->localU.sge.dataAddrLow = 0x1000;
    ubWqe->localU.sge.length = 256;
    ubWqe->notify.notifyAddrLow = 0x4000;

    Rt91095StarsUbdmaDBmodeSqe ubSqe;
    memset(&ubSqe, 0, sizeof(ubSqe));
    ubSqe.jettyId1 = 1;
    ubSqe.piValue1 = 2;

    EXPECT_NO_THROW(ParseDavidUDMASqe(0, &ubSqe));
}

TEST_F(DeviceSqeParseTest, ParseDavidUDMASqe_ReadOpcodeWithShm)
{
    alignas(64) uint8_t wqeBuffer[HCCL_WQE_SIZE * 4] = {};
    sim::RaJetty jetty{};
    jetty.id = 1;
    jetty.sqBuffer = reinterpret_cast<uint64_t>(wqeBuffer);
    RunnerDB::Add<sim::RaJetty>(jetty);

    UdmaSqeWrite* ubWqeRead = reinterpret_cast<UdmaSqeWrite*>(wqeBuffer);
    memset(ubWqeRead, 0, sizeof(UdmaSqeWrite));
    ubWqeRead->comm.opcode = static_cast<int>(UdmaSqOpcode::UDMA_OPC_READ);
    ubWqeRead->comm.inlineEn = 0;
    ubWqeRead->comm.udfFlag = 0;
    ubWqeRead->comm.rmtAddrLow = 0x2000;
    ubWqeRead->u.sge.dataAddrLow = 0x1000;
    ubWqeRead->u.sge.length = 64;

    Rt91095StarsUbdmaDBmodeSqe ubSqe;
    memset(&ubSqe, 0, sizeof(ubSqe));
    ubSqe.jettyId1 = 1;
    ubSqe.piValue1 = 2;

    EXPECT_NO_THROW(ParseDavidUDMASqe(0, &ubSqe));
}

TEST_F(DeviceSqeParseTest, ParseDavidUDMASqe_UnsupportedOpcodeWithShm)
{
    alignas(64) uint8_t wqeBuffer[HCCL_WQE_SIZE * 4] = {};
    sim::RaJetty jetty{};
    jetty.id = 1;
    jetty.sqBuffer = reinterpret_cast<uint64_t>(wqeBuffer);
    RunnerDB::Add<sim::RaJetty>(jetty);

    // Put unsupported opcode at ciVal position (adjusted: piValue1-1)
    UdmaSqeCommon* ubCommon = reinterpret_cast<UdmaSqeCommon*>(wqeBuffer + 1 * HCCL_WQE_SIZE);
    memset(ubCommon, 0, sizeof(UdmaSqeCommon));
    ubCommon->opcode = 63;

    Rt91095StarsUbdmaDBmodeSqe ubSqe;
    memset(&ubSqe, 0, sizeof(ubSqe));
    ubSqe.jettyId1 = 1;
    ubSqe.piValue1 = 2;

    EXPECT_NO_THROW(ParseDavidUDMASqe(0, &ubSqe));
}

TEST_F(DeviceSqeParseTest, ParseDavidUDMASqe_AdjustCiValWithShm)
{
    alignas(64) uint8_t wqeBuffer[HCCL_WQE_SIZE * 8] = {};
    sim::RaJetty jetty{};
    jetty.id = 1;
    jetty.sqBuffer = reinterpret_cast<uint64_t>(wqeBuffer);
    RunnerDB::Add<sim::RaJetty>(jetty);

    // piValue1=3, first ciVal = 3-2=1, opcode at [1] is not WRITE_WITH_NOTIFY,
    // so adjusted ciVal = 3-1=2, reads WQE at [2]
    UdmaSqeCommon* ubCommon1 = reinterpret_cast<UdmaSqeCommon*>(wqeBuffer + 1 * HCCL_WQE_SIZE);
    memset(ubCommon1, 0, sizeof(UdmaSqeCommon));
    ubCommon1->opcode = static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE);

    UdmaSqeWrite* ubWqeWrite = reinterpret_cast<UdmaSqeWrite*>(wqeBuffer + 2 * HCCL_WQE_SIZE);
    memset(ubWqeWrite, 0, sizeof(UdmaSqeWrite));
    ubWqeWrite->comm.opcode = static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE);
    ubWqeWrite->comm.inlineEn = 0;
    ubWqeWrite->comm.udfFlag = 0;
    ubWqeWrite->comm.rmtAddrLow = 0x2000;
    ubWqeWrite->u.sge.dataAddrLow = 0x1000;
    ubWqeWrite->u.sge.length = 128;

    Rt91095StarsUbdmaDBmodeSqe ubSqe;
    memset(&ubSqe, 0, sizeof(ubSqe));
    ubSqe.jettyId1 = 1;
    ubSqe.piValue1 = 3;

    EXPECT_NO_THROW(ParseDavidUDMASqe(0, &ubSqe));
}

TEST_F(DeviceSqeParseTest, ParseDavidUDMASqe_NullWqeBuffer)
{
    // RaJetty exists but sqBuffer=0, so wqeBuffer=0
    sim::RaJetty jetty{};
    jetty.id = 1;
    jetty.sqBuffer = 0;
    RunnerDB::Add<sim::RaJetty>(jetty);

    Rt91095StarsUbdmaDBmodeSqe ubSqe;
    memset(&ubSqe, 0, sizeof(ubSqe));
    ubSqe.jettyId1 = 1;
    ubSqe.piValue1 = 2;

    EXPECT_NO_THROW(ParseDavidUDMASqe(0, &ubSqe));
}

TEST_F(DeviceSqeParseTest, ParseDavidUDMASqe_WriteWithNotifyWithShm_Reduce)
{
    alignas(64) uint8_t wqeBuffer[HCCL_WQE_SIZE * 4] = {};
    sim::RaJetty jetty{};
    jetty.id = 1;
    jetty.sqBuffer = reinterpret_cast<uint64_t>(wqeBuffer);
    RunnerDB::Add<sim::RaJetty>(jetty);

    UdmaSqeWriteWithNotify* ubWqe = reinterpret_cast<UdmaSqeWriteWithNotify*>(wqeBuffer);
    memset(ubWqe, 0, sizeof(UdmaSqeWriteWithNotify));
    ubWqe->comm.opcode = static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE_WITH_NOTIFY);
    ubWqe->comm.udfFlag = 1;
    ubWqe->comm.rmtAddrLow = 0x2000;
    ubWqe->comm.rmtEid[0] = 0;
    ubWqe->comm.inlinedata.udfData.reduceOp = 0xA;
    ubWqe->comm.inlinedata.udfData.reduceType = 0x7;
    ubWqe->localU.sge.dataAddrLow = 0x1000;
    ubWqe->localU.sge.length = 512;
    ubWqe->notify.notifyAddrLow = 0x4000;

    Rt91095StarsUbdmaDBmodeSqe ubSqe;
    memset(&ubSqe, 0, sizeof(ubSqe));
    ubSqe.jettyId1 = 1;
    ubSqe.piValue1 = 2;

    EXPECT_NO_THROW(ParseDavidUDMASqe(0, &ubSqe));
}

// ==================== Additional Coverage Tests ====================

TEST_F(DeviceSqeParseTest, ParseDataTypeDavid_Uint8)
{
    uint8_t result = 0x30;
    HcclDataType type = ParseDataTypeDavid(result);
    EXPECT_EQ(type, HcclDataType::HCCL_DATA_TYPE_UINT8);
}

TEST_F(DeviceSqeParseTest, ParseDataTypeDavid_Uint16)
{
    uint8_t result = 0x40;
    HcclDataType type = ParseDataTypeDavid(result);
    EXPECT_EQ(type, HcclDataType::HCCL_DATA_TYPE_UINT16);
}

TEST_F(DeviceSqeParseTest, ParseDataTypeDavid_Uint32)
{
    uint8_t result = 0x50;
    HcclDataType type = ParseDataTypeDavid(result);
    EXPECT_EQ(type, HcclDataType::HCCL_DATA_TYPE_UINT32);
}

TEST_F(DeviceSqeParseTest, ParseDataTypeDavid_Fp16)
{
    uint8_t result = 0x60;
    HcclDataType type = ParseDataTypeDavid(result);
    EXPECT_EQ(type, HcclDataType::HCCL_DATA_TYPE_FP16);
}

TEST_F(DeviceSqeParseTest, ParseDataTypeDavid_Bfp16Alt)
{
    uint8_t result = 0x80;
    HcclDataType type = ParseDataTypeDavid(result);
    EXPECT_EQ(type, HcclDataType::HCCL_DATA_TYPE_BFP16);
}

TEST_F(DeviceSqeParseTest, ParseUbDataTypeDavid_Int16)
{
    uint32_t type = 0x1;
    HcclDataType dataType = ParseUbDataTypeDavid(type);
    EXPECT_EQ(dataType, HcclDataType::HCCL_DATA_TYPE_INT16);
}

TEST_F(DeviceSqeParseTest, ParseUbDataTypeDavid_Int32)
{
    uint32_t type = 0x2;
    HcclDataType dataType = ParseUbDataTypeDavid(type);
    EXPECT_EQ(dataType, HcclDataType::HCCL_DATA_TYPE_INT32);
}

TEST_F(DeviceSqeParseTest, ParseUbDataTypeDavid_Uint8)
{
    uint32_t type = 0x3;
    HcclDataType dataType = ParseUbDataTypeDavid(type);
    EXPECT_EQ(dataType, HcclDataType::HCCL_DATA_TYPE_UINT8);
}

TEST_F(DeviceSqeParseTest, ParseUbDataTypeDavid_Uint16)
{
    uint32_t type = 0x4;
    HcclDataType dataType = ParseUbDataTypeDavid(type);
    EXPECT_EQ(dataType, HcclDataType::HCCL_DATA_TYPE_UINT16);
}

TEST_F(DeviceSqeParseTest, ParseUbDataTypeDavid_Uint32)
{
    uint32_t type = 0x5;
    HcclDataType dataType = ParseUbDataTypeDavid(type);
    EXPECT_EQ(dataType, HcclDataType::HCCL_DATA_TYPE_UINT32);
}

TEST_F(DeviceSqeParseTest, ParseUbDataTypeDavid_Fp16)
{
    uint32_t type = 0x6;
    HcclDataType dataType = ParseUbDataTypeDavid(type);
    EXPECT_EQ(dataType, HcclDataType::HCCL_DATA_TYPE_FP16);
}

TEST_F(DeviceSqeParseTest, PrintTaskMetaData_DefaultType)
{
    HcclTaskMetaData taskMeta;
    memset(&taskMeta, 0, sizeof(taskMeta));
    taskMeta.taskType = static_cast<HccLTaskMetaType>(99);
    EXPECT_NO_THROW(PrintTaskMetaData(taskMeta));
}

TEST_F(DeviceSqeParseTest, ParseDavidSDMASqe_MemcpyWithNonZeroRank)
{
    Rt91095StarsMemcpySqe sqe;
    memset(&sqe, 0, sizeof(sqe));
    sqe.opcode = 0;
    sqe.u.strideMode0.lengthMove = 2048;
    sqe.u.strideMode0.srcAddrLow = 0x4000;
    sqe.u.strideMode0.srcAddrHigh = 0x1;
    sqe.u.strideMode0.dstAddrLow = 0x8000;
    sqe.u.strideMode0.dstAddrHigh = 0x2;

    SetCurRankId(3);
    EXPECT_NO_THROW(ParseDavidSDMASqe(1, &sqe));
    SetCurRankId(0);
}

TEST_F(DeviceSqeParseTest, ParseDavidSDMASqe_ReduceWithCombinedOpcode)
{
    Rt91095StarsMemcpySqe sqe;
    memset(&sqe, 0, sizeof(sqe));
    sqe.opcode = 0x21;
    sqe.u.strideMode0.lengthMove = 512;
    sqe.u.strideMode0.srcAddrLow = 0x1000;
    sqe.u.strideMode0.srcAddrHigh = 0;
    sqe.u.strideMode0.dstAddrLow = 0x2000;
    sqe.u.strideMode0.dstAddrHigh = 0;

    EXPECT_NO_THROW(ParseDavidSDMASqe(0, &sqe));
}

TEST_F(DeviceSqeParseTest, ParseDavidNotifySqe_WaitWithNotifyId)
{
    Rt91095StarsNotifySqe sqe;
    memset(&sqe, 0, sizeof(sqe));
    sqe.notifyId = 999;

    SetCurRankId(2);
    EXPECT_NO_THROW(ParseDavidNotifySqe(1, &sqe, false));
    SetCurRankId(0);
}

TEST_F(DeviceSqeParseTest, ParseDavidNotifySqe_RecordWithNotifyId)
{
    Rt91095StarsNotifySqe sqe;
    memset(&sqe, 0, sizeof(sqe));
    sqe.notifyId = 777;

    SetCurRankId(1);
    EXPECT_NO_THROW(ParseDavidNotifySqe(2, &sqe, true));
    SetCurRankId(0);
}

TEST_F(DeviceSqeParseTest, ParseDavidUBReadWriteSqe_InlineWriteNotifyWithHighAddr)
{
    UdmaSqeWrite ubWqe;
    memset(&ubWqe, 0, sizeof(ubWqe));
    ubWqe.comm.inlineEn = 1;
    ubWqe.comm.rmtAddrLow = 0x1000;
    ubWqe.comm.rmtAddrHigh = 0x1;
    ubWqe.comm.rmtEid[0] = 0x7F000001;

    EXPECT_NO_THROW(ParseDavidUBReadWriteSqe(reinterpret_cast<uint64_t>(&ubWqe), 1, 2, false));
}

TEST_F(DeviceSqeParseTest, ParseDavidUBReadWriteSqe_WriteMemCpyWithHighAddr)
{
    UdmaSqeWrite ubWqe;
    memset(&ubWqe, 0, sizeof(ubWqe));
    ubWqe.comm.inlineEn = 0;
    ubWqe.comm.udfFlag = 0;
    ubWqe.comm.rmtAddrLow = 0x2000;
    ubWqe.comm.rmtAddrHigh = 0x1;
    ubWqe.u.sge.dataAddrLow = 0x1000;
    ubWqe.u.sge.dataAddrHigh = 0x2;
    ubWqe.u.sge.length = 256;

    EXPECT_NO_THROW(ParseDavidUBReadWriteSqe(reinterpret_cast<uint64_t>(&ubWqe), 1, 2, false));
}

TEST_F(DeviceSqeParseTest, ParseDavidUBReadWriteSqe_ReadMemCpyWithHighAddr)
{
    UdmaSqeWrite ubWqe;
    memset(&ubWqe, 0, sizeof(ubWqe));
    ubWqe.comm.inlineEn = 0;
    ubWqe.comm.udfFlag = 0;
    ubWqe.comm.rmtAddrLow = 0x3000;
    ubWqe.comm.rmtAddrHigh = 0x1;
    ubWqe.u.sge.dataAddrLow = 0x5000;
    ubWqe.u.sge.dataAddrHigh = 0x2;
    ubWqe.u.sge.length = 512;

    EXPECT_NO_THROW(ParseDavidUBReadWriteSqe(reinterpret_cast<uint64_t>(&ubWqe), 1, 2, true));
}

TEST_F(DeviceSqeParseTest, ParseDavidUBReadWriteSqe_WriteReduceMax)
{
    UdmaSqeWrite ubWqe;
    memset(&ubWqe, 0, sizeof(ubWqe));
    ubWqe.comm.inlineEn = 0;
    ubWqe.comm.udfFlag = 1;
    ubWqe.comm.rmtAddrLow = 0x2000;
    ubWqe.comm.rmtAddrHigh = 0;
    ubWqe.comm.inlinedata.udfData.reduceOp = 0x8;
    ubWqe.comm.inlinedata.udfData.reduceType = 0x7;
    ubWqe.u.sge.dataAddrLow = 0x1000;
    ubWqe.u.sge.dataAddrHigh = 0;
    ubWqe.u.sge.length = 1024;

    EXPECT_NO_THROW(ParseDavidUBReadWriteSqe(reinterpret_cast<uint64_t>(&ubWqe), 0, 1, false));
}

TEST_F(DeviceSqeParseTest, ParseDavidUBReadWriteSqe_WriteReduceMin)
{
    UdmaSqeWrite ubWqe;
    memset(&ubWqe, 0, sizeof(ubWqe));
    ubWqe.comm.inlineEn = 0;
    ubWqe.comm.udfFlag = 1;
    ubWqe.comm.rmtAddrLow = 0x2000;
    ubWqe.comm.rmtAddrHigh = 0;
    ubWqe.comm.inlinedata.udfData.reduceOp = 0x9;
    ubWqe.comm.inlinedata.udfData.reduceType = 0x0;
    ubWqe.u.sge.dataAddrLow = 0x1000;
    ubWqe.u.sge.dataAddrHigh = 0;
    ubWqe.u.sge.length = 512;

    EXPECT_NO_THROW(ParseDavidUBReadWriteSqe(reinterpret_cast<uint64_t>(&ubWqe), 0, 1, false));
}

TEST_F(DeviceSqeParseTest, ParseDavidUBReadWriteSqe_WriteReduceInvalidOp)
{
    UdmaSqeWrite ubWqe;
    memset(&ubWqe, 0, sizeof(ubWqe));
    ubWqe.comm.inlineEn = 0;
    ubWqe.comm.udfFlag = 1;
    ubWqe.comm.rmtAddrLow = 0x2000;
    ubWqe.comm.rmtAddrHigh = 0;
    ubWqe.comm.inlinedata.udfData.reduceOp = 0xA;
    ubWqe.comm.inlinedata.udfData.reduceType = 0x7;
    ubWqe.u.sge.dataAddrLow = 0x1000;
    ubWqe.u.sge.dataAddrHigh = 0;
    ubWqe.u.sge.length = 256;

    EXPECT_NO_THROW(ParseDavidUBReadWriteSqe(reinterpret_cast<uint64_t>(&ubWqe), 0, 1, false));
}

TEST_F(DeviceSqeParseTest, ParseDavidUBWriteWithNotify_MemCpyWithHighAddr)
{
    UdmaSqeWriteWithNotify ubWqe;
    memset(&ubWqe, 0, sizeof(ubWqe));
    ubWqe.comm.udfFlag = 0;
    ubWqe.comm.rmtAddrLow = 0x2000;
    ubWqe.comm.rmtAddrHigh = 0x1;
    ubWqe.comm.rmtEid[0] = 0;
    ubWqe.localU.sge.dataAddrLow = 0x1000;
    ubWqe.localU.sge.dataAddrHigh = 0x2;
    ubWqe.localU.sge.length = 256;
    ubWqe.notify.notifyAddrLow = 0x4000;
    ubWqe.notify.notifyAddrHigh = 0x3;

    EXPECT_NO_THROW(ParseDavidUBWriteWithNotifySqe(reinterpret_cast<uint64_t>(&ubWqe), 1, 2));
}

TEST_F(DeviceSqeParseTest, ParseDavidUBWriteWithNotify_ReduceMax)
{
    UdmaSqeWriteWithNotify ubWqe;
    memset(&ubWqe, 0, sizeof(ubWqe));
    ubWqe.comm.udfFlag = 1;
    ubWqe.comm.rmtAddrLow = 0x2000;
    ubWqe.comm.rmtAddrHigh = 0;
    ubWqe.comm.rmtEid[0] = 0;
    ubWqe.comm.inlinedata.udfData.reduceOp = 0x8;
    ubWqe.comm.inlinedata.udfData.reduceType = 0x7;
    ubWqe.localU.sge.dataAddrLow = 0x1000;
    ubWqe.localU.sge.dataAddrHigh = 0;
    ubWqe.localU.sge.length = 512;
    ubWqe.notify.notifyAddrLow = 0x4000;
    ubWqe.notify.notifyAddrHigh = 0;

    EXPECT_NO_THROW(ParseDavidUBWriteWithNotifySqe(reinterpret_cast<uint64_t>(&ubWqe), 0, 2));
}

TEST_F(DeviceSqeParseTest, ParseDavidUBWriteWithNotify_ReduceInvalidOp)
{
    UdmaSqeWriteWithNotify ubWqe;
    memset(&ubWqe, 0, sizeof(ubWqe));
    ubWqe.comm.udfFlag = 1;
    ubWqe.comm.rmtAddrLow = 0x2000;
    ubWqe.comm.rmtAddrHigh = 0;
    ubWqe.comm.rmtEid[0] = 0;
    ubWqe.comm.inlinedata.udfData.reduceOp = 0xA;
    ubWqe.comm.inlinedata.udfData.reduceType = 0x7;
    ubWqe.localU.sge.dataAddrLow = 0x1000;
    ubWqe.localU.sge.dataAddrHigh = 0;
    ubWqe.localU.sge.length = 512;
    ubWqe.notify.notifyAddrLow = 0x4000;
    ubWqe.notify.notifyAddrHigh = 0;

    EXPECT_NO_THROW(ParseDavidUBWriteWithNotifySqe(reinterpret_cast<uint64_t>(&ubWqe), 0, 2));
}

TEST_F(DeviceSqeParseTest, ParseA5SqeFromSqBuffer_SDMATypeWithSqeCnt)
{
    struct halSqCqConfigInfo info;
    memset(&info, 0, sizeof(info));
    info.sqId = 0;
    info.value[0] = 1;
    UpdateSqTail(0, 0);

    uint8_t* sqBuf = nullptr;
    GetSqBufferAddr(&sqBuf);
    if (sqBuf) {
        Rt91095StarsMemcpySqe sqe;
        memset(&sqe, 0, sizeof(sqe));
        sqe.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA);
        sqe.opcode = 0;
        sqe.u.strideMode0.lengthMove = 1024;
        sqe.u.strideMode0.srcAddrLow = 0x1000;
        sqe.u.strideMode0.srcAddrHigh = 0;
        sqe.u.strideMode0.dstAddrLow = 0x2000;
        sqe.u.strideMode0.dstAddrHigh = 0;
        memcpy(sqBuf, &sqe, sizeof(sqe));
    }

    EXPECT_NO_THROW(ParseA5SqeFromSqBuffer(0, &info));
}

TEST_F(DeviceSqeParseTest, ParseA5SqeFromSqBuffer_NotifyWaitTypeWithSqeCnt)
{
    struct halSqCqConfigInfo info;
    memset(&info, 0, sizeof(info));
    info.sqId = 0;
    info.value[0] = 1;
    UpdateSqTail(0, 0);

    uint8_t* sqBuf = nullptr;
    GetSqBufferAddr(&sqBuf);
    if (sqBuf) {
        Rt91095StarsNotifySqe sqe;
        memset(&sqe, 0, sizeof(sqe));
        sqe.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
        sqe.notifyId = 100;
        memcpy(sqBuf, &sqe, sizeof(sqe));
    }

    EXPECT_NO_THROW(ParseA5SqeFromSqBuffer(0, &info));
}

TEST_F(DeviceSqeParseTest, ParseA5SqeFromSqBuffer_NotifyRecordTypeWithSqeCnt)
{
    struct halSqCqConfigInfo info;
    memset(&info, 0, sizeof(info));
    info.sqId = 0;
    info.value[0] = 1;
    UpdateSqTail(0, 0);

    uint8_t* sqBuf = nullptr;
    GetSqBufferAddr(&sqBuf);
    if (sqBuf) {
        Rt91095StarsNotifySqe sqe;
        memset(&sqe, 0, sizeof(sqe));
        sqe.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD);
        sqe.notifyId = 200;
        memcpy(sqBuf, &sqe, sizeof(sqe));
    }

    EXPECT_NO_THROW(ParseA5SqeFromSqBuffer(0, &info));
}

TEST_F(DeviceSqeParseTest, ParseA5SqeFromSqBuffer_UBDMATypeWithSqeCnt)
{
    struct halSqCqConfigInfo info;
    memset(&info, 0, sizeof(info));
    info.sqId = 0;
    info.value[0] = 1;
    UpdateSqTail(0, 0);

    sim::RaJetty jetty{};
    jetty.id = 1;
    jetty.sqBuffer = 0;
    RunnerDB::Add<sim::RaJetty>(jetty);

    uint8_t* sqBuf = nullptr;
    GetSqBufferAddr(&sqBuf);
    if (sqBuf) {
        Rt91095StarsUbdmaDBmodeSqe sqe;
        memset(&sqe, 0, sizeof(sqe));
        sqe.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_UBDMA);
        sqe.jettyId1 = 1;
        sqe.piValue1 = 1;
        memcpy(sqBuf, &sqe, sizeof(sqe));
    }

    EXPECT_NO_THROW(ParseA5SqeFromSqBuffer(0, &info));
}

TEST_F(DeviceSqeParseTest, ParseA5SqeFromSqBuffer_DefaultTypeWithSqeCnt)
{
    struct halSqCqConfigInfo info;
    memset(&info, 0, sizeof(info));
    info.sqId = 0;
    info.value[0] = 1;
    UpdateSqTail(0, 0);

    uint8_t* sqBuf = nullptr;
    GetSqBufferAddr(&sqBuf);
    if (sqBuf) {
        Rt91095StarsSqeHeader header;
        memset(&header, 0, sizeof(header));
        header.type = 35;
        memcpy(sqBuf, &header, sizeof(header));
    }

    EXPECT_NO_THROW(ParseA5SqeFromSqBuffer(0, &info));
}

TEST_F(DeviceSqeParseTest, ParseA5SqeFromSqBuffer_TailLessThanHeadWithSqeCnt)
{
    struct halSqCqConfigInfo info;
    memset(&info, 0, sizeof(info));
    info.sqId = 0;
    info.value[0] = 5;

    UpdateSqTail(0, 10);

    uint8_t* sqBuf = nullptr;
    GetSqBufferAddr(&sqBuf);
    if (sqBuf) {
        Rt91095StarsSqeHeader header;
        memset(&header, 0, sizeof(header));
        header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA);
        Rt91095StarsMemcpySqe sqe;
        memset(&sqe, 0, sizeof(sqe));
        sqe.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA);
        sqe.opcode = 0;
        sqe.u.strideMode0.lengthMove = 1024;
        sqe.u.strideMode0.srcAddrLow = 0x1000;
        sqe.u.strideMode0.srcAddrHigh = 0;
        sqe.u.strideMode0.dstAddrLow = 0x2000;
        sqe.u.strideMode0.dstAddrHigh = 0;
        memcpy(sqBuf + 10 * HCCL_SQE_SIZE, &sqe, sizeof(sqe));
    }

    EXPECT_NO_THROW(ParseA5SqeFromSqBuffer(0, &info));
}

TEST_F(DeviceSqeParseTest, ParseA5SqeFromSqBuffer_MultipleSqeTypes)
{
    struct halSqCqConfigInfo info;
    memset(&info, 0, sizeof(info));
    info.sqId = 0;
    info.value[0] = 2;
    UpdateSqTail(0, 0);

    uint8_t* sqBuf = nullptr;
    GetSqBufferAddr(&sqBuf);
    if (sqBuf) {
        Rt91095StarsMemcpySqe sqe1;
        memset(&sqe1, 0, sizeof(sqe1));
        sqe1.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA);
        sqe1.opcode = 0;
        sqe1.u.strideMode0.lengthMove = 1024;
        sqe1.u.strideMode0.srcAddrLow = 0x1000;
        sqe1.u.strideMode0.srcAddrHigh = 0;
        sqe1.u.strideMode0.dstAddrLow = 0x2000;
        sqe1.u.strideMode0.dstAddrHigh = 0;
        memcpy(sqBuf, &sqe1, sizeof(sqe1));

        Rt91095StarsNotifySqe sqe2;
        memset(&sqe2, 0, sizeof(sqe2));
        sqe2.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
        sqe2.notifyId = 100;
        memcpy(sqBuf + HCCL_SQE_SIZE, &sqe2, sizeof(sqe2));
    }

    EXPECT_NO_THROW(ParseA5SqeFromSqBuffer(0, &info));
}

TEST_F(DeviceSqeParseTest, ParseA5SqeFromSqBuffer_NonZeroDevId)
{
    struct halSqCqConfigInfo info;
    memset(&info, 0, sizeof(info));
    info.sqId = 1;
    info.value[0] = 1;
    UpdateSqTail(1, 0);

    uint8_t* sqBuf = nullptr;
    GetSqBufferAddr(&sqBuf);
    if (sqBuf) {
        Rt91095StarsSqeHeader header;
        memset(&header, 0, sizeof(header));
        header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA);
        memcpy(sqBuf, &header, sizeof(header));
    }

    EXPECT_NO_THROW(ParseA5SqeFromSqBuffer(3, &info));
}

TEST_F(DeviceSqeParseTest, ParseA5SqeFromSqBuffer_SDMAReduceType)
{
    struct halSqCqConfigInfo info;
    memset(&info, 0, sizeof(info));
    info.sqId = 0;
    info.value[0] = 1;
    UpdateSqTail(0, 0);

    uint8_t* sqBuf = nullptr;
    GetSqBufferAddr(&sqBuf);
    if (sqBuf) {
        Rt91095StarsMemcpySqe sqe;
        memset(&sqe, 0, sizeof(sqe));
        sqe.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA);
        sqe.opcode = 0x01;
        sqe.u.strideMode0.lengthMove = 512;
        sqe.u.strideMode0.srcAddrLow = 0x1000;
        sqe.u.strideMode0.srcAddrHigh = 0;
        sqe.u.strideMode0.dstAddrLow = 0x2000;
        sqe.u.strideMode0.dstAddrHigh = 0;
        memcpy(sqBuf, &sqe, sizeof(sqe));
    }

    EXPECT_NO_THROW(ParseA5SqeFromSqBuffer(0, &info));
}

TEST_F(DeviceSqeParseTest, GetRmtRankIdByEid_MaxEid)
{
    uint32_t eid = 0xFFFFFFFF;
    EXPECT_NO_THROW(GetRmtRankIdByEid(eid));
}

TEST_F(DeviceSqeParseTest, GetRmtRankIdByEid_SpecificIp)
{
    uint32_t eid = 0xC0A80101;
    EXPECT_NO_THROW(GetRmtRankIdByEid(eid));
}

TEST_F(DeviceSqeParseTest, ParseReduceTypeDavid_Zero)
{
    uint8_t result = 0x00;
    HcclReduceOp op = ParseReduceTypeDavid(result);
    EXPECT_EQ(op, HcclReduceOp::HCCL_REDUCE_RESERVED);
}

TEST_F(DeviceSqeParseTest, ParseDataTypeDavid_ZeroInvalid)
{
    uint8_t result = 0x90;
    HcclDataType type = ParseDataTypeDavid(result);
    EXPECT_EQ(type, HcclDataType::HCCL_DATA_TYPE_RESERVED);
}

// ==================== ParseDavidUDMASqe Tests ====================
// ParseDavidUDMASqe uses GetWqebufferByJettyId to lookup RaJetty.sqBuffer from DB.
// These tests cover various opcode paths with valid RaJetty setup.

TEST_F(DeviceSqeParseTest, ParseDavidUDMASqe_WriteOpcodeWithValidWqe)
{
    alignas(64) uint8_t wqeBuffer[HCCL_WQE_SIZE * 4] = {};
    sim::RaJetty jetty{};
    jetty.id = 1;
    jetty.sqBuffer = reinterpret_cast<uint64_t>(wqeBuffer);
    RunnerDB::Add<sim::RaJetty>(jetty);

    UdmaSqeWrite* ubWqeWrite = reinterpret_cast<UdmaSqeWrite*>(wqeBuffer);
    memset(ubWqeWrite, 0, sizeof(UdmaSqeWrite));
    ubWqeWrite->comm.opcode = static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE);
    ubWqeWrite->comm.inlineEn = 1;
    ubWqeWrite->comm.rmtAddrLow = 0x1000;
    ubWqeWrite->comm.rmtEid[0] = 0;

    Rt91095StarsUbdmaDBmodeSqe ubSqe;
    memset(&ubSqe, 0, sizeof(ubSqe));
    ubSqe.jettyId1 = 1;
    ubSqe.piValue1 = 2;

    EXPECT_NO_THROW(ParseDavidUDMASqe(0, &ubSqe));
}

TEST_F(DeviceSqeParseTest, ParseDavidUDMASqe_WriteWithNotifyOpcodeWithValidWqe)
{
    alignas(64) uint8_t wqeBuffer[HCCL_WQE_SIZE * 4] = {};
    sim::RaJetty jetty{};
    jetty.id = 1;
    jetty.sqBuffer = reinterpret_cast<uint64_t>(wqeBuffer);
    RunnerDB::Add<sim::RaJetty>(jetty);

    UdmaSqeWriteWithNotify* ubWqe = reinterpret_cast<UdmaSqeWriteWithNotify*>(wqeBuffer);
    memset(ubWqe, 0, sizeof(UdmaSqeWriteWithNotify));
    ubWqe->comm.opcode = static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE_WITH_NOTIFY);
    ubWqe->comm.udfFlag = 0;
    ubWqe->comm.rmtAddrLow = 0x2000;
    ubWqe->comm.rmtEid[0] = 0;
    ubWqe->localU.sge.dataAddrLow = 0x1000;
    ubWqe->localU.sge.length = 256;
    ubWqe->notify.notifyAddrLow = 0x4000;

    Rt91095StarsUbdmaDBmodeSqe ubSqe;
    memset(&ubSqe, 0, sizeof(ubSqe));
    ubSqe.jettyId1 = 1;
    ubSqe.piValue1 = 2;

    EXPECT_NO_THROW(ParseDavidUDMASqe(0, &ubSqe));
}

TEST_F(DeviceSqeParseTest, ParseDavidUDMASqe_ReadOpcodeWithValidWqe)
{
    alignas(64) uint8_t wqeBuffer[HCCL_WQE_SIZE * 4] = {};
    sim::RaJetty jetty{};
    jetty.id = 1;
    jetty.sqBuffer = reinterpret_cast<uint64_t>(wqeBuffer);
    RunnerDB::Add<sim::RaJetty>(jetty);

    UdmaSqeWrite* ubWqeRead = reinterpret_cast<UdmaSqeWrite*>(wqeBuffer);
    memset(ubWqeRead, 0, sizeof(UdmaSqeWrite));
    ubWqeRead->comm.opcode = static_cast<int>(UdmaSqOpcode::UDMA_OPC_READ);
    ubWqeRead->comm.inlineEn = 0;
    ubWqeRead->comm.udfFlag = 1;
    ubWqeRead->comm.rmtAddrLow = 0x2000;
    ubWqeRead->comm.inlinedata.udfData.reduceOp = 0xA;
    ubWqeRead->comm.inlinedata.udfData.reduceType = 0x7;
    ubWqeRead->u.sge.dataAddrLow = 0x1000;
    ubWqeRead->u.sge.length = 64;

    Rt91095StarsUbdmaDBmodeSqe ubSqe;
    memset(&ubSqe, 0, sizeof(ubSqe));
    ubSqe.jettyId1 = 1;
    ubSqe.piValue1 = 2;

    EXPECT_NO_THROW(ParseDavidUDMASqe(0, &ubSqe));
}

TEST_F(DeviceSqeParseTest, ParseDavidUDMASqe_UnsupportedOpcodeWithValidWqe)
{
    alignas(64) uint8_t wqeBuffer[HCCL_WQE_SIZE * 4] = {};
    sim::RaJetty jetty{};
    jetty.id = 1;
    jetty.sqBuffer = reinterpret_cast<uint64_t>(wqeBuffer);
    RunnerDB::Add<sim::RaJetty>(jetty);

    UdmaSqeCommon* ubCommon = reinterpret_cast<UdmaSqeCommon*>(wqeBuffer);
    memset(ubCommon, 0, sizeof(UdmaSqeCommon));
    ubCommon->opcode = 99;

    Rt91095StarsUbdmaDBmodeSqe ubSqe;
    memset(&ubSqe, 0, sizeof(ubSqe));
    ubSqe.jettyId1 = 1;
    ubSqe.piValue1 = 2;

    EXPECT_NO_THROW(ParseDavidUDMASqe(0, &ubSqe));
}

TEST_F(DeviceSqeParseTest, ParseDavidUDMASqe_AdjustCiValWithValidWqe)
{
    alignas(64) uint8_t wqeBuffer[HCCL_WQE_SIZE * 8] = {};
    sim::RaJetty jetty{};
    jetty.id = 1;
    jetty.sqBuffer = reinterpret_cast<uint64_t>(wqeBuffer);
    RunnerDB::Add<sim::RaJetty>(jetty);

    UdmaSqeCommon* ubCommon1 = reinterpret_cast<UdmaSqeCommon*>(wqeBuffer + 1 * HCCL_WQE_SIZE);
    memset(ubCommon1, 0, sizeof(UdmaSqeCommon));
    ubCommon1->opcode = static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE);

    UdmaSqeWrite* ubWqeWrite = reinterpret_cast<UdmaSqeWrite*>(wqeBuffer + 2 * HCCL_WQE_SIZE);
    memset(ubWqeWrite, 0, sizeof(UdmaSqeWrite));
    ubWqeWrite->comm.opcode = static_cast<int>(UdmaSqOpcode::UDMA_OPC_WRITE);
    ubWqeWrite->comm.inlineEn = 0;
    ubWqeWrite->comm.udfFlag = 0;
    ubWqeWrite->comm.rmtAddrLow = 0x2000;
    ubWqeWrite->u.sge.dataAddrLow = 0x1000;
    ubWqeWrite->u.sge.length = 128;

    Rt91095StarsUbdmaDBmodeSqe ubSqe;
    memset(&ubSqe, 0, sizeof(ubSqe));
    ubSqe.jettyId1 = 1;
    ubSqe.piValue1 = 3;

    EXPECT_NO_THROW(ParseDavidUDMASqe(0, &ubSqe));
}

// ==================== ParseA5SqeFromSqBuffer with multiple SQEs Tests ====================

TEST_F(DeviceSqeParseTest, ParseA5SqeFromSqBuffer_MultipleSqeTypesV2)
{
    struct halSqCqConfigInfo info;
    memset(&info, 0, sizeof(info));
    info.sqId = 0;
    info.value[0] = 4; // 4 SQEs
    UpdateSqTail(0, 0);

    uint8_t* sqBuf = nullptr;
    GetSqBufferAddr(&sqBuf);
    if (sqBuf) {
        // SQE 0: SDMA memcpy
        Rt91095StarsMemcpySqe sqe0;
        memset(&sqe0, 0, sizeof(sqe0));
        sqe0.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA);
        sqe0.opcode = 0;
        sqe0.u.strideMode0.lengthMove = 1024;
        sqe0.u.strideMode0.srcAddrLow = 0x1000;
        sqe0.u.strideMode0.srcAddrHigh = 0;
        sqe0.u.strideMode0.dstAddrLow = 0x2000;
        sqe0.u.strideMode0.dstAddrHigh = 0;
        memcpy(sqBuf, &sqe0, sizeof(sqe0));

        // SQE 1: NOTIFY_WAIT
        Rt91095StarsNotifySqe sqe1;
        memset(&sqe1, 0, sizeof(sqe1));
        sqe1.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
        sqe1.notifyId = 100;
        memcpy(sqBuf + HCCL_SQE_SIZE, &sqe1, sizeof(sqe1));

        // SQE 2: NOTIFY_RECORD
        Rt91095StarsNotifySqe sqe2;
        memset(&sqe2, 0, sizeof(sqe2));
        sqe2.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD);
        sqe2.notifyId = 200;
        memcpy(sqBuf + 2 * HCCL_SQE_SIZE, &sqe2, sizeof(sqe2));

        // SQE 3: UBDMA
        Rt91095StarsUbdmaDBmodeSqe sqe3;
        memset(&sqe3, 0, sizeof(sqe3));
        sqe3.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_UBDMA);
        sqe3.jettyId1 = 1;
        sqe3.piValue1 = 1;
        memcpy(sqBuf + 3 * HCCL_SQE_SIZE, &sqe3, sizeof(sqe3));
    }

    EXPECT_NO_THROW(ParseA5SqeFromSqBuffer(0, &info));
}

TEST_F(DeviceSqeParseTest, ParseA5SqeFromSqBuffer_WrapAroundWithMultipleSqe)
{
    // Test wrap-around with head > tail
    struct halSqCqConfigInfo info;
    memset(&info, 0, sizeof(info));
    info.sqId = 0;
    info.value[0] = 3;   // tail = 3
    UpdateSqTail(0, 10); // head = 10

    uint8_t* sqBuf = nullptr;
    GetSqBufferAddr(&sqBuf);
    if (sqBuf) {
        // Place SDMA SQE at index 10
        Rt91095StarsMemcpySqe sqe0;
        memset(&sqe0, 0, sizeof(sqe0));
        sqe0.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA);
        sqe0.opcode = 0;
        sqe0.u.strideMode0.lengthMove = 1024;
        sqe0.u.strideMode0.srcAddrLow = 0x1000;
        sqe0.u.strideMode0.srcAddrHigh = 0;
        sqe0.u.strideMode0.dstAddrLow = 0x2000;
        sqe0.u.strideMode0.dstAddrHigh = 0;
        memcpy(sqBuf + 10 * HCCL_SQE_SIZE, &sqe0, sizeof(sqe0));

        // Place NOTIFY_WAIT SQE at index 11
        Rt91095StarsNotifySqe sqe1;
        memset(&sqe1, 0, sizeof(sqe1));
        sqe1.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
        sqe1.notifyId = 100;
        memcpy(sqBuf + 11 * HCCL_SQE_SIZE, &sqe1, sizeof(sqe1));

        // Place NOTIFY_RECORD SQE at index 0 (wrapped)
        Rt91095StarsNotifySqe sqe2;
        memset(&sqe2, 0, sizeof(sqe2));
        sqe2.header.type = static_cast<int>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD);
        sqe2.notifyId = 200;
        memcpy(sqBuf, &sqe2, sizeof(sqe2));
    }

    EXPECT_NO_THROW(ParseA5SqeFromSqBuffer(0, &info));
}

// ==================== ParseDavidUBReadWriteSqe edge cases ====================

TEST_F(DeviceSqeParseTest, ParseDavidUBReadWriteSqe_ReadReduce)
{
    UdmaSqeWrite ubWqe;
    memset(&ubWqe, 0, sizeof(ubWqe));
    ubWqe.comm.inlineEn = 0;
    ubWqe.comm.udfFlag = 1;
    ubWqe.comm.rmtAddrLow = 0x2000;
    ubWqe.comm.rmtAddrHigh = 0;
    ubWqe.comm.inlinedata.udfData.reduceOp = 0xA;
    ubWqe.comm.inlinedata.udfData.reduceType = 0x7;
    ubWqe.u.sge.dataAddrLow = 0x1000;
    ubWqe.u.sge.dataAddrHigh = 0;
    ubWqe.u.sge.length = 1024;

    // isRead = true, so srcOffset = rmtAddr, dstOffset = locAddr
    EXPECT_NO_THROW(ParseDavidUBReadWriteSqe(reinterpret_cast<uint64_t>(&ubWqe), 0, 1, true));
}

TEST_F(DeviceSqeParseTest, ParseDavidUBReadWriteSqe_InlineWriteWithEid)
{
    UdmaSqeWrite ubWqe;
    memset(&ubWqe, 0, sizeof(ubWqe));
    ubWqe.comm.inlineEn = 1;
    ubWqe.comm.rmtAddrLow = 0x1000;
    ubWqe.comm.rmtAddrHigh = 0;
    ubWqe.comm.rmtEid[0] = 0x7F000001; // 127.0.0.1

    EXPECT_NO_THROW(ParseDavidUBReadWriteSqe(reinterpret_cast<uint64_t>(&ubWqe), 0, 1, false));
}

// ==================== ParseDavidUBWriteWithNotifySqe edge cases ====================

TEST_F(DeviceSqeParseTest, ParseDavidUBWriteWithNotify_ReduceMin)
{
    UdmaSqeWriteWithNotify ubWqe;
    memset(&ubWqe, 0, sizeof(ubWqe));
    ubWqe.comm.udfFlag = 1;
    ubWqe.comm.rmtAddrLow = 0x2000;
    ubWqe.comm.rmtAddrHigh = 0;
    ubWqe.comm.rmtEid[0] = 0;
    ubWqe.comm.inlinedata.udfData.reduceOp = 0x9; // MIN
    ubWqe.comm.inlinedata.udfData.reduceType = 0x0;
    ubWqe.localU.sge.dataAddrLow = 0x1000;
    ubWqe.localU.sge.dataAddrHigh = 0;
    ubWqe.localU.sge.length = 512;
    ubWqe.notify.notifyAddrLow = 0x4000;
    ubWqe.notify.notifyAddrHigh = 0;

    EXPECT_NO_THROW(ParseDavidUBWriteWithNotifySqe(reinterpret_cast<uint64_t>(&ubWqe), 0, 2));
}

TEST_F(DeviceSqeParseTest, ParseDavidUBWriteWithNotify_ReduceSum)
{
    UdmaSqeWriteWithNotify ubWqe;
    memset(&ubWqe, 0, sizeof(ubWqe));
    ubWqe.comm.udfFlag = 1;
    ubWqe.comm.rmtAddrLow = 0x2000;
    ubWqe.comm.rmtAddrHigh = 0;
    ubWqe.comm.rmtEid[0] = 0;
    ubWqe.comm.inlinedata.udfData.reduceOp = 0xA;   // SUM
    ubWqe.comm.inlinedata.udfData.reduceType = 0x6; // FP16
    ubWqe.localU.sge.dataAddrLow = 0x1000;
    ubWqe.localU.sge.dataAddrHigh = 0;
    ubWqe.localU.sge.length = 256;
    ubWqe.notify.notifyAddrLow = 0x4000;
    ubWqe.notify.notifyAddrHigh = 0;

    EXPECT_NO_THROW(ParseDavidUBWriteWithNotifySqe(reinterpret_cast<uint64_t>(&ubWqe), 0, 2));
}

// ==================== ParseReduceTypeDavid and ParseDataTypeDavid edge cases ====================

TEST_F(DeviceSqeParseTest, ParseReduceTypeDavid_CombinedOpcode)
{
    // opcode with both reduce type (low 4 bits) and data type (high 4 bits)
    uint8_t result = 0x21; // reduce type = 1 (SUM), data type would be 0x20
    HcclReduceOp op = ParseReduceTypeDavid(result);
    EXPECT_EQ(op, HcclReduceOp::HCCL_REDUCE_SUM);
}

TEST_F(DeviceSqeParseTest, ParseDataTypeDavid_CombinedOpcode)
{
    uint8_t result = 0x21; // data type = 0x20 (INT32)
    HcclDataType type = ParseDataTypeDavid(result);
    EXPECT_EQ(type, HcclDataType::HCCL_DATA_TYPE_INT32);
}

// ==================== GetFull64BitAddr edge cases ====================

TEST_F(DeviceSqeParseTest, GetFull64BitAddr_AlternatingBits)
{
    uint32_t lowAddr = 0xAAAAAAAA;
    uint32_t highAddr = 0x55555555;
    uint64_t result = GetFull64BitAddr(lowAddr, highAddr);
    EXPECT_EQ(result, 0x55555555AAAAAAAAULL);
}

TEST_F(DeviceSqeParseTest, GetFull64BitAddr_SingleBit)
{
    uint32_t lowAddr = 0x1;
    uint32_t highAddr = 0x80000000;
    uint64_t result = GetFull64BitAddr(lowAddr, highAddr);
    EXPECT_EQ(result, 0x8000000000000001ULL);
}

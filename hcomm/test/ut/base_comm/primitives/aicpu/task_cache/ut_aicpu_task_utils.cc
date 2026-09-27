/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>

#include "gtest/gtest.h"

#include "aicpu_task_utils.h"
#include "sqe_v82.h"
#include "udma_data_struct.h"
#include "config_plf_log_v2.h"

using namespace hcomm;
using namespace Hccl;

class AicpuTaskUtilsTest : public testing::Test {
protected:
    uint64_t savedPlfDebugConfig_ = 0;

    void SetUp() override
    {
        savedPlfDebugConfig_ = GetPlfDebugConfigValue();
        SetPlfDebugConfigValue(PLF_TASK);
    }

    void TearDown() override { SetPlfDebugConfigValue(savedPlfDebugConfig_); }
};

TEST_F(AicpuTaskUtilsTest, DumpSqeContent_Ubdma_ReturnsSuccess)
{
    Rt91095StarsUbdmaDBmodeSqe sqe{};
    sqe.header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_UBDMA);
    sqe.mode = 1;
    sqe.doorbellNum = 1;
    sqe.jettyId1 = 2;
    sqe.funcId1 = 3;
    sqe.piValue1 = 4;
    sqe.dieId1 = 5;
    EXPECT_EQ(AicpuTaskUtils::DumpSqeContent(reinterpret_cast<const uint8_t*>(&sqe)), HCCL_SUCCESS);
}

TEST_F(AicpuTaskUtilsTest, DumpSqeContent_SingleNotifyWait_ReturnsSuccess)
{
    Rt91095StarsNotifySqe sqe{};
    sqe.header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    sqe.notifyId = 10;
    sqe.subType = 1;
    sqe.clrFlag = true;
    sqe.timeout = 20;
    EXPECT_EQ(AicpuTaskUtils::DumpSqeContent(reinterpret_cast<const uint8_t*>(&sqe)), HCCL_SUCCESS);
}

TEST_F(AicpuTaskUtilsTest, DumpSqeContent_CountNotifyRecord_ReturnsSuccess)
{
    Rt91095StarsNotifySqe sqe{};
    sqe.header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD);
    sqe.notifyId = 10;
    sqe.subType = 1;
    sqe.cntFlag = true;
    sqe.cntValue = 8;
    sqe.recordModeBit = 2;
    EXPECT_EQ(AicpuTaskUtils::DumpSqeContent(reinterpret_cast<const uint8_t*>(&sqe)), HCCL_SUCCESS);
}

TEST_F(AicpuTaskUtilsTest, DumpSqeContent_CountNotifyWait_ReturnsSuccess)
{
    Rt91095StarsNotifySqe sqe{};
    sqe.header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    sqe.notifyId = 10;
    sqe.subType = 1;
    sqe.cntFlag = true;
    sqe.cntValue = 8;
    sqe.waitModeBit = 1;
    EXPECT_EQ(AicpuTaskUtils::DumpSqeContent(reinterpret_cast<const uint8_t*>(&sqe)), HCCL_SUCCESS);
}

TEST_F(AicpuTaskUtilsTest, DumpSqeContent_Sdma_ReturnsSuccess)
{
    Rt91095StarsMemcpySqe sqe{};
    sqe.header.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA);
    sqe.opcode = 1;
    sqe.sssv = 1;
    sqe.dssv = 1;
    sqe.sns = 1;
    sqe.dns = 1;
    sqe.mapamPartId = 2;
    sqe.u.strideMode0.lengthMove = 128;
    sqe.u.strideMode0.srcAddrLow = 0x1000;
    sqe.u.strideMode0.dstAddrLow = 0x2000;
    EXPECT_EQ(AicpuTaskUtils::DumpSqeContent(reinterpret_cast<const uint8_t*>(&sqe)), HCCL_SUCCESS);
}

TEST_F(AicpuTaskUtilsTest, DumpSqeContent_UnsupportedType_ReturnsSuccess)
{
    Rt91095StarsSqeHeader sqe{};
    sqe.type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_WRITE_VALUE);
    EXPECT_EQ(AicpuTaskUtils::DumpSqeContent(reinterpret_cast<const uint8_t*>(&sqe)), HCCL_SUCCESS);
}

TEST_F(AicpuTaskUtilsTest, DumpSqeContent_NullInput_ReturnsPtrError)
{
    EXPECT_EQ(AicpuTaskUtils::DumpSqeContent(nullptr), HCCL_E_PTR);
}

TEST_F(AicpuTaskUtilsTest, DumpWqeContent_ReadAndWrite_ReturnSuccess)
{
    UdmaSqeWrite wqe{};
    wqe.comm.opcode = static_cast<uint8_t>(UdmaSqOpcode::UDMA_OPC_READ);
    wqe.u.sge.length = 128;
    wqe.u.sge.tokenId = 3;
    wqe.u.sge.dataAddrLow = 0x1000;
    EXPECT_EQ(AicpuTaskUtils::DumpWqeContent(reinterpret_cast<const uint8_t*>(&wqe)), HCCL_SUCCESS);

    wqe.comm.opcode = static_cast<uint8_t>(UdmaSqOpcode::UDMA_OPC_WRITE);
    EXPECT_EQ(AicpuTaskUtils::DumpWqeContent(reinterpret_cast<const uint8_t*>(&wqe)), HCCL_SUCCESS);
}

TEST_F(AicpuTaskUtilsTest, DumpWqeContent_InlineWrite_ReturnsSuccess)
{
    UdmaSqeWrite wqe{};
    wqe.comm.opcode = static_cast<uint8_t>(UdmaSqOpcode::UDMA_OPC_WRITE);
    wqe.comm.inlineEn = 1;
    wqe.comm.inlineMsgLen = 8;
    EXPECT_EQ(AicpuTaskUtils::DumpWqeContent(reinterpret_cast<const uint8_t*>(&wqe)), HCCL_SUCCESS);
}

TEST_F(AicpuTaskUtilsTest, DumpWqeContent_WriteWithNotify_ReturnsSuccess)
{
    UdmaSqeWriteWithNotify wqe{};
    wqe.comm.opcode = static_cast<uint8_t>(UdmaSqOpcode::UDMA_OPC_WRITE_WITH_IMM);
    wqe.localU.sge.length = 128;
    wqe.localU.sge.tokenId = 3;
    wqe.localU.sge.dataAddrLow = 0x1000;
    wqe.notify.notifyTokenId = 4;
    wqe.notify.notifyTokenValue = 5;
    wqe.notify.notifyAddrLow = 0x2000;
    wqe.notify.notifyDataLow = 1;
    EXPECT_EQ(AicpuTaskUtils::DumpWqeContent(reinterpret_cast<const uint8_t*>(&wqe)), HCCL_SUCCESS);
}

TEST_F(AicpuTaskUtilsTest, DumpWqeContent_ReduceFields_ReturnsSuccess)
{
    UdmaSqeWrite wqe{};
    wqe.comm.opcode = static_cast<uint8_t>(UdmaSqOpcode::UDMA_OPC_WRITE);
    wqe.comm.udfFlag = 1;
    wqe.comm.inlinedata.udfData.udfType = 1;
    wqe.comm.inlinedata.udfData.reduceType = 2;
    wqe.comm.inlinedata.udfData.reduceOp = 3;
    EXPECT_EQ(AicpuTaskUtils::DumpWqeContent(reinterpret_cast<const uint8_t*>(&wqe)), HCCL_SUCCESS);
}

TEST_F(AicpuTaskUtilsTest, DumpWqeContent_NullInput_ReturnsPtrError)
{
    EXPECT_EQ(AicpuTaskUtils::DumpWqeContent(nullptr), HCCL_E_PTR);
}

TEST_F(AicpuTaskUtilsTest, DumpWqeContent_UnsupportedType_ReturnsSuccess)
{
    UdmaSqeWrite wqe{};
    wqe.comm.opcode = static_cast<uint8_t>(UdmaSqOpcode::UDMA_OPC_INVALID);
    EXPECT_EQ(AicpuTaskUtils::DumpWqeContent(reinterpret_cast<const uint8_t*>(&wqe)), HCCL_SUCCESS);
}

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <memory>

#include "check_utils.h"
#include "data_slice.h"

namespace HcclSim {
bool IsSendRecvType(HcclCMDType opType);
bool DataSliceSizeIsEqual(std::unique_ptr<DataSlice>& a, std::unique_ptr<DataSlice>& b);
bool DataSliceSizeIsEqual(std::unique_ptr<DataSlice>& a, std::unique_ptr<DataSlice>& b, std::unique_ptr<DataSlice>& c);
std::vector<std::string> SplitString(const std::string& str, const char c);

class CheckUtilsTest : public testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(CheckUtilsTest, IsAllToAllSeries_AllToAll) { EXPECT_TRUE(IsAllToAllSeries(HcclCMDType::HCCL_CMD_ALLTOALL)); }

TEST_F(CheckUtilsTest, IsAllToAllSeries_AllToAllV) { EXPECT_TRUE(IsAllToAllSeries(HcclCMDType::HCCL_CMD_ALLTOALLV)); }

TEST_F(CheckUtilsTest, IsAllToAllSeries_AllToAllVC) { EXPECT_TRUE(IsAllToAllSeries(HcclCMDType::HCCL_CMD_ALLTOALLVC)); }

TEST_F(CheckUtilsTest, IsAllToAllSeries_NotAllToAll)
{
    EXPECT_FALSE(IsAllToAllSeries(HcclCMDType::HCCL_CMD_ALLREDUCE));
    EXPECT_FALSE(IsAllToAllSeries(HcclCMDType::HCCL_CMD_ALLGATHER));
    EXPECT_FALSE(IsAllToAllSeries(HcclCMDType::HCCL_CMD_REDUCE_SCATTER));
}

TEST_F(CheckUtilsTest, IsSendRecvType_Send) { EXPECT_TRUE(IsSendRecvType(HcclCMDType::HCCL_CMD_SEND)); }

TEST_F(CheckUtilsTest, IsSendRecvType_Recv) { EXPECT_TRUE(IsSendRecvType(HcclCMDType::HCCL_CMD_RECEIVE)); }

TEST_F(CheckUtilsTest, IsSendRecvType_NotSendRecv)
{
    EXPECT_FALSE(IsSendRecvType(HcclCMDType::HCCL_CMD_ALLREDUCE));
    EXPECT_FALSE(IsSendRecvType(HcclCMDType::HCCL_CMD_BROADCAST));
}

TEST_F(CheckUtilsTest, CalcDataSize_Int8)
{
    u64 dataSize = 0;
    CalcDataSize(HcclCMDType::HCCL_CMD_ALLREDUCE, 100, HcclDataType::HCCL_DATA_TYPE_INT8, dataSize);
    EXPECT_EQ(dataSize, 100);
}

TEST_F(CheckUtilsTest, CalcDataSize_Int32)
{
    u64 dataSize = 0;
    CalcDataSize(HcclCMDType::HCCL_CMD_ALLREDUCE, 100, HcclDataType::HCCL_DATA_TYPE_INT32, dataSize);
    EXPECT_EQ(dataSize, 400);
}

TEST_F(CheckUtilsTest, CalcDataSize_Fp16)
{
    u64 dataSize = 0;
    CalcDataSize(HcclCMDType::HCCL_CMD_ALLREDUCE, 100, HcclDataType::HCCL_DATA_TYPE_FP16, dataSize);
    EXPECT_EQ(dataSize, 200);
}

TEST_F(CheckUtilsTest, CalcDataSize_Fp32)
{
    u64 dataSize = 0;
    CalcDataSize(HcclCMDType::HCCL_CMD_ALLREDUCE, 100, HcclDataType::HCCL_DATA_TYPE_FP32, dataSize);
    EXPECT_EQ(dataSize, 400);
}

TEST_F(CheckUtilsTest, CalcDataSize_ZeroCount)
{
    u64 dataSize = 0;
    CalcDataSize(HcclCMDType::HCCL_CMD_ALLREDUCE, 0, HcclDataType::HCCL_DATA_TYPE_INT32, dataSize);
    EXPECT_EQ(dataSize, 0);
}

TEST_F(CheckUtilsTest, CalcDataSize_LargeCount)
{
    u64 dataSize = 0;
    CalcDataSize(HcclCMDType::HCCL_CMD_ALLREDUCE, 1000000, HcclDataType::HCCL_DATA_TYPE_INT32, dataSize);
    EXPECT_EQ(dataSize, 4000000);
}

TEST_F(CheckUtilsTest, GenTopoMeta_SingleServer)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 1, 8);

    EXPECT_EQ(topoMeta.size(), 1);
    EXPECT_EQ(topoMeta[0].size(), 1);
    EXPECT_EQ(topoMeta[0][0].size(), 8);
}

TEST_F(CheckUtilsTest, GenTopoMeta_MultipleServers)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 4, 8);

    EXPECT_EQ(topoMeta.size(), 1);
    EXPECT_EQ(topoMeta[0].size(), 4);
}

TEST_F(CheckUtilsTest, GenTopoMeta_MultipleSuperPods)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 8);

    EXPECT_EQ(topoMeta.size(), 2);
}

TEST_F(CheckUtilsTest, CalRankSize_SingleServer)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 1, 8);

    u32 rankSize = CalRankSize(topoMeta);
    EXPECT_EQ(rankSize, 8);
}

TEST_F(CheckUtilsTest, CalRankSize_MultipleServers)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 1, 4, 8);

    u32 rankSize = CalRankSize(topoMeta);
    EXPECT_EQ(rankSize, 32);
}

TEST_F(CheckUtilsTest, CalRankSize_MultipleSuperPods)
{
    TopoMeta topoMeta;
    GenTopoMeta(topoMeta, 2, 4, 8);

    u32 rankSize = CalRankSize(topoMeta);
    EXPECT_EQ(rankSize, 64);
}

TEST_F(CheckUtilsTest, SrcBufDes_Constructor)
{
    SrcBufDes srcBuf(1, BufferType::INPUT, 0x1000);

    EXPECT_EQ(srcBuf.rankId, 1);
    EXPECT_EQ(srcBuf.bufType, BufferType::INPUT);
    EXPECT_EQ(srcBuf.srcAddr, 0x1000);
}

TEST_F(CheckUtilsTest, SrcBufDes_Comparison)
{
    SrcBufDes srcBuf1(1, BufferType::INPUT, 0x1000);
    SrcBufDes srcBuf2(2, BufferType::INPUT, 0x2000);

    EXPECT_TRUE(srcBuf1 < srcBuf2);
    EXPECT_FALSE(srcBuf2 < srcBuf1);
}

TEST_F(CheckUtilsTest, BufferSemantic_Constructor)
{
    BufferSemantic bufSem(0x1000, 1024);

    EXPECT_EQ(bufSem.startAddr, 0x1000);
    EXPECT_EQ(bufSem.size, 1024);
    EXPECT_EQ(bufSem.isReduce, false);
}

TEST_F(CheckUtilsTest, BufferSemantic_WithReduce)
{
    BufferSemantic bufSem(0x1000, 1024, true, HcclReduceOp::HCCL_REDUCE_SUM);

    EXPECT_EQ(bufSem.isReduce, true);
    EXPECT_EQ(bufSem.reduceType, HcclReduceOp::HCCL_REDUCE_SUM);
}

TEST_F(CheckUtilsTest, BufferSemantic_Comparison)
{
    BufferSemantic bufSem1(0x1000, 1024);
    BufferSemantic bufSem2(0x2000, 2048);

    EXPECT_TRUE(bufSem1 < bufSem2);
    EXPECT_FALSE(bufSem2 < bufSem1);
}

TEST_F(CheckUtilsTest, SplitString_SingleChar)
{
    std::string str = "a,b,c";
    auto result = SplitString(str, ',');

    EXPECT_EQ(result.size(), 3);
    EXPECT_EQ(result[0], "a");
    EXPECT_EQ(result[1], "b");
    EXPECT_EQ(result[2], "c");
}

TEST_F(CheckUtilsTest, SplitString_EmptyString)
{
    std::string str = "";
    auto result = SplitString(str, ',');

    EXPECT_EQ(result.size(), 0);
}

TEST_F(CheckUtilsTest, SplitString_NoDelimiter)
{
    std::string str = "abc";
    auto result = SplitString(str, ',');

    EXPECT_EQ(result.size(), 1);
    EXPECT_EQ(result[0], "abc");
}

TEST_F(CheckUtilsTest, SplitString_ConsecutiveDelimiters)
{
    std::string str = "a,,b";
    auto result = SplitString(str, ',');

    EXPECT_EQ(result.size(), 3);
    EXPECT_EQ(result[0], "a");
    EXPECT_EQ(result[1], "");
    EXPECT_EQ(result[2], "b");
}

TEST_F(CheckUtilsTest, DataSliceSizeIsEqual_Equal)
{
    auto slice1 = std::make_unique<DataSlice>();
    auto slice2 = std::make_unique<DataSlice>();
    slice1->SetSize(100);
    slice2->SetSize(100);

    EXPECT_TRUE(DataSliceSizeIsEqual(slice1, slice2));
}

TEST_F(CheckUtilsTest, DataSliceSizeIsEqual_NotEqual)
{
    auto slice1 = std::make_unique<DataSlice>();
    auto slice2 = std::make_unique<DataSlice>();
    slice1->SetSize(100);
    slice2->SetSize(200);

    EXPECT_FALSE(DataSliceSizeIsEqual(slice1, slice2));
}

TEST_F(CheckUtilsTest, DataSliceSizeIsEqual_ThreeSlices)
{
    auto slice1 = std::make_unique<DataSlice>();
    auto slice2 = std::make_unique<DataSlice>();
    auto slice3 = std::make_unique<DataSlice>();
    slice1->SetSize(100);
    slice2->SetSize(100);
    slice3->SetSize(100);

    EXPECT_TRUE(DataSliceSizeIsEqual(slice1, slice2, slice3));
}

TEST_F(CheckUtilsTest, CalcInputOutputSize_AllReduce)
{
    u64 inputSize = 0;
    u64 outputSize = 0;
    CalcInputOutputSize(
        HcclCMDType::HCCL_CMD_ALLREDUCE, 4, 100, HcclDataType::HCCL_DATA_TYPE_INT32, inputSize, outputSize, 0, 0, 0, {},
        {});
    EXPECT_EQ(inputSize, 400);
    EXPECT_EQ(outputSize, 400);
}

TEST_F(CheckUtilsTest, CalcInputOutputSize_Broadcast)
{
    u64 inputSize = 0;
    u64 outputSize = 0;
    CalcInputOutputSize(
        HcclCMDType::HCCL_CMD_BROADCAST, 4, 100, HcclDataType::HCCL_DATA_TYPE_INT32, inputSize, outputSize, 0, 0, 0, {},
        {});
    EXPECT_EQ(inputSize, 400);
    EXPECT_EQ(outputSize, 400);
}

TEST_F(CheckUtilsTest, CalcInputOutputSize_Send)
{
    u64 inputSize = 0;
    u64 outputSize = 0;
    CalcInputOutputSize(
        HcclCMDType::HCCL_CMD_SEND, 4, 100, HcclDataType::HCCL_DATA_TYPE_INT32, inputSize, outputSize, 0, 0, 1, {}, {});
    EXPECT_EQ(inputSize, 400);
    EXPECT_EQ(outputSize, 0);
}

TEST_F(CheckUtilsTest, CalcInputOutputSize_Receive)
{
    u64 inputSize = 0;
    u64 outputSize = 0;
    CalcInputOutputSize(
        HcclCMDType::HCCL_CMD_RECEIVE, 4, 100, HcclDataType::HCCL_DATA_TYPE_INT32, inputSize, outputSize, 1, 0, 1, {},
        {});
    EXPECT_EQ(inputSize, 0);
    EXPECT_EQ(outputSize, 400);
}

TEST_F(CheckUtilsTest, CalcInputOutputSize_Reduce)
{
    u64 inputSize = 0;
    u64 outputSize = 0;
    CalcInputOutputSize(
        HcclCMDType::HCCL_CMD_REDUCE, 4, 100, HcclDataType::HCCL_DATA_TYPE_INT32, inputSize, outputSize, 0, 0, 0, {},
        {});
    EXPECT_EQ(inputSize, 400);
    EXPECT_EQ(outputSize, 400);
}

TEST_F(CheckUtilsTest, CalcInputOutputSize_AllGather)
{
    u64 inputSize = 0;
    u64 outputSize = 0;
    CalcInputOutputSize(
        HcclCMDType::HCCL_CMD_ALLGATHER, 4, 100, HcclDataType::HCCL_DATA_TYPE_INT32, inputSize, outputSize, 0, 0, 0, {},
        {});
    EXPECT_EQ(inputSize, 400);
    EXPECT_EQ(outputSize, 1600);
}

TEST_F(CheckUtilsTest, CalcInputOutputSize_ReduceScatter)
{
    u64 inputSize = 0;
    u64 outputSize = 0;
    CalcInputOutputSize(
        HcclCMDType::HCCL_CMD_REDUCE_SCATTER, 4, 100, HcclDataType::HCCL_DATA_TYPE_INT32, inputSize, outputSize, 0, 0,
        0, {}, {});
    EXPECT_EQ(inputSize, 1600);
    EXPECT_EQ(outputSize, 400);
}

TEST_F(CheckUtilsTest, CalcInputOutputSize_Scatter)
{
    u64 inputSize = 0;
    u64 outputSize = 0;
    CalcInputOutputSize(
        HcclCMDType::HCCL_CMD_SCATTER, 4, 100, HcclDataType::HCCL_DATA_TYPE_INT32, inputSize, outputSize, 0, 0, 0, {},
        {});
    EXPECT_EQ(inputSize, 1600);
    EXPECT_EQ(outputSize, 400);
}

TEST_F(CheckUtilsTest, CalcInputOutputSize_BatchSendRecv)
{
    u64 inputSize = 0;
    u64 outputSize = 0;
    CalcInputOutputSize(
        HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, 4, 100, HcclDataType::HCCL_DATA_TYPE_INT32, inputSize, outputSize, 0, 0,
        0, {}, {});
    EXPECT_EQ(inputSize, 400);
    EXPECT_EQ(outputSize, 400);
}

TEST_F(CheckUtilsTest, CalcInputOutputSize_UnsupportedOp)
{
    u64 inputSize = 0;
    u64 outputSize = 0;
    CalcInputOutputSize(
        HcclCMDType::HCCL_CMD_INVALID, 4, 100, HcclDataType::HCCL_DATA_TYPE_INT32, inputSize, outputSize, 0, 0, 0, {},
        {});
    // For unsupported ops, the function just logs a warning
    EXPECT_EQ(inputSize, 0);
    EXPECT_EQ(outputSize, 0);
}
} // namespace HcclSim

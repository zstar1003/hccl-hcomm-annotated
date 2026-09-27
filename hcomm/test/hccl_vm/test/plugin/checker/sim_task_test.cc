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

#include "data_slice.h"
#include "sim_task.h"

namespace HcclSim {
class SimTaskTest : public testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(SimTaskTest, TaskStubLocalCopy_Describe)
{
    DataSlice srcSlice;
    srcSlice.SetSize(1024);
    srcSlice.SetOffset(0x1000);

    DataSlice dstSlice;
    dstSlice.SetSize(1024);
    dstSlice.SetOffset(0x2000);

    TaskStubLocalCopy task(srcSlice, dstSlice);
    std::string desc = task.Describe();

    EXPECT_FALSE(desc.empty());
    EXPECT_TRUE(desc.find("LocalCopy") != std::string::npos);
}

TEST_F(SimTaskTest, TaskStubLocalCopy_GetSrcSlice)
{
    DataSlice srcSlice;
    srcSlice.SetSize(1024);
    srcSlice.SetOffset(0x1000);

    DataSlice dstSlice;
    dstSlice.SetSize(1024);
    dstSlice.SetOffset(0x2000);

    TaskStubLocalCopy task(srcSlice, dstSlice);

    const DataSlice& src = task.GetSrcSlice();
    EXPECT_EQ(src.GetSize(), 1024);
    EXPECT_EQ(src.GetOffset(), 0x1000);
}

TEST_F(SimTaskTest, TaskStubLocalCopy_GetDstSlice)
{
    DataSlice srcSlice;
    srcSlice.SetSize(1024);
    srcSlice.SetOffset(0x1000);

    DataSlice dstSlice;
    dstSlice.SetSize(1024);
    dstSlice.SetOffset(0x2000);

    TaskStubLocalCopy task(srcSlice, dstSlice);

    const DataSlice& dst = task.GetDstSlice();
    EXPECT_EQ(dst.GetSize(), 1024);
    EXPECT_EQ(dst.GetOffset(), 0x2000);
}

TEST_F(SimTaskTest, TaskStubLocalReduce_Describe)
{
    DataSlice srcSlice;
    srcSlice.SetSize(1024);

    DataSlice dstSlice;
    dstSlice.SetSize(1024);

    TaskStubLocalReduce task(srcSlice, dstSlice, HcclDataType::HCCL_DATA_TYPE_INT32, HcclReduceOp::HCCL_REDUCE_SUM);
    std::string desc = task.Describe();

    EXPECT_FALSE(desc.empty());
    EXPECT_TRUE(desc.find("LocalReduce") != std::string::npos);
}

TEST_F(SimTaskTest, TaskStubLocalReduce_GetDataType)
{
    DataSlice srcSlice;
    DataSlice dstSlice;

    TaskStubLocalReduce task(srcSlice, dstSlice, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM);

    EXPECT_EQ(task.GetDataType(), HcclDataType::HCCL_DATA_TYPE_FP32);
}

TEST_F(SimTaskTest, TaskStubLocalReduce_GetReduceOp)
{
    DataSlice srcSlice;
    DataSlice dstSlice;

    TaskStubLocalReduce task(srcSlice, dstSlice, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_MAX);

    EXPECT_EQ(task.GetReduceOp(), HcclReduceOp::HCCL_REDUCE_MAX);
}

TEST_F(SimTaskTest, TaskStubPost_Describe)
{
    LinkInfo link(LinkProtoStub::RDMA);
    TaskStubPost task(1, link, 100);
    std::string desc = task.Describe();

    EXPECT_FALSE(desc.empty());
    EXPECT_TRUE(desc.find("Post") != std::string::npos);
}

TEST_F(SimTaskTest, TaskStubPost_GetRemoteRank)
{
    LinkInfo link(LinkProtoStub::RDMA);
    TaskStubPost task(5, link, 100);

    EXPECT_EQ(task.GetRemoteRank(), 5);
}

TEST_F(SimTaskTest, TaskStubPost_GetNotifyId)
{
    LinkInfo link(LinkProtoStub::RDMA);
    TaskStubPost task(1, link, 200);

    EXPECT_EQ(task.GetNotifyId(), 200);
}

TEST_F(SimTaskTest, TaskStubPost_SetRemoteRank)
{
    LinkInfo link(LinkProtoStub::RDMA);
    TaskStubPost task(1, link, 100);

    task.SetRemoteRank(10);
    EXPECT_EQ(task.GetRemoteRank(), 10);
}

TEST_F(SimTaskTest, TaskStubPost_SetNotifyId)
{
    LinkInfo link(LinkProtoStub::RDMA);
    TaskStubPost task(1, link, 100);

    task.SetNotifyId(500);
    EXPECT_EQ(task.GetNotifyId(), 500);
}

TEST_F(SimTaskTest, TaskStubWait_Describe)
{
    LinkInfo link(LinkProtoStub::RDMA);
    TaskStubWait task(2, link, 150);
    std::string desc = task.Describe();

    EXPECT_FALSE(desc.empty());
    EXPECT_TRUE(desc.find("Wait") != std::string::npos);
}

TEST_F(SimTaskTest, TaskStubWait_GetRemoteRank)
{
    LinkInfo link(LinkProtoStub::RDMA);
    TaskStubWait task(3, link, 150);

    EXPECT_EQ(task.GetRemoteRank(), 3);
}

TEST_F(SimTaskTest, TaskStubWait_GetNotifyId)
{
    LinkInfo link(LinkProtoStub::RDMA);
    TaskStubWait task(2, link, 300);

    EXPECT_EQ(task.GetNotifyId(), 300);
}

TEST_F(SimTaskTest, TaskStubWait_SetNotifyId)
{
    LinkInfo link(LinkProtoStub::RDMA);
    TaskStubWait task(2, link, 150);

    task.SetNotifyId(400);
    EXPECT_EQ(task.GetNotifyId(), 400);
}

TEST_F(SimTaskTest, TaskStubLocalPostTo_Describe)
{
    TaskStubLocalPostTo task(100);
    std::string desc = task.Describe();

    EXPECT_FALSE(desc.empty());
    EXPECT_TRUE(desc.find("LocalPostTo") != std::string::npos);
}

TEST_F(SimTaskTest, TaskStubLocalPostTo_GetNotifyId)
{
    TaskStubLocalPostTo task(100);

    EXPECT_EQ(task.GetNotifyId(), 100);
}

TEST_F(SimTaskTest, TaskStubLocalPostTo_SetNotifyId)
{
    TaskStubLocalPostTo task(100);

    task.SetNotifyId(200);
    EXPECT_EQ(task.GetNotifyId(), 200);
}

TEST_F(SimTaskTest, TaskStubLocalWaitFrom_Describe)
{
    TaskStubLocalWaitFrom task(150);
    std::string desc = task.Describe();

    EXPECT_FALSE(desc.empty());
    EXPECT_TRUE(desc.find("LocalWaitFrom") != std::string::npos);
}

TEST_F(SimTaskTest, TaskStubLocalWaitFrom_GetNotifyId)
{
    TaskStubLocalWaitFrom task(200);

    EXPECT_EQ(task.GetNotifyId(), 200);
}

TEST_F(SimTaskTest, TaskStubLocalWaitFrom_SetNotifyId)
{
    TaskStubLocalWaitFrom task(150);

    task.SetNotifyId(250);
    EXPECT_EQ(task.GetNotifyId(), 250);
}

TEST_F(SimTaskTest, TaskStubRead_Describe)
{
    DataSlice localSlice;
    localSlice.SetSize(1024);

    DataSlice remoteSlice;
    remoteSlice.SetSize(1024);

    LinkInfo link(LinkProtoStub::SDMA);
    TaskStubRead task(2, link, localSlice, remoteSlice);
    std::string desc = task.Describe();

    EXPECT_FALSE(desc.empty());
    EXPECT_TRUE(desc.find("Read") != std::string::npos);
}

TEST_F(SimTaskTest, TaskStubRead_GetRemoteRank)
{
    DataSlice localSlice;
    DataSlice remoteSlice;
    LinkInfo link(LinkProtoStub::SDMA);
    TaskStubRead task(5, link, localSlice, remoteSlice);

    EXPECT_EQ(task.GetRemoteRank(), 5);
}

TEST_F(SimTaskTest, TaskStubRead_GetLinkType)
{
    DataSlice localSlice;
    DataSlice remoteSlice;
    LinkInfo link(LinkProtoStub::SDMA);
    TaskStubRead task(2, link, localSlice, remoteSlice);

    EXPECT_EQ(task.GetLinkType(), LinkProtoStub::SDMA);
}

TEST_F(SimTaskTest, TaskStubWrite_Describe)
{
    DataSlice localSlice;
    localSlice.SetSize(1024);

    DataSlice remoteSlice;
    remoteSlice.SetSize(1024);

    LinkInfo link(LinkProtoStub::RDMA);
    TaskStubWrite task(3, link, localSlice, remoteSlice);
    std::string desc = task.Describe();

    EXPECT_FALSE(desc.empty());
    EXPECT_TRUE(desc.find("Write") != std::string::npos);
}

TEST_F(SimTaskTest, TaskStubWrite_GetRemoteRank)
{
    DataSlice localSlice;
    DataSlice remoteSlice;
    LinkInfo link(LinkProtoStub::RDMA);
    TaskStubWrite task(4, link, localSlice, remoteSlice);

    EXPECT_EQ(task.GetRemoteRank(), 4);
}

TEST_F(SimTaskTest, TaskStubLoopStart_Describe)
{
    TaskStubLoopStart task(1, 2);
    std::string desc = task.Describe();

    EXPECT_FALSE(desc.empty());
    EXPECT_TRUE(desc.find("LoopStart") != std::string::npos);
}

TEST_F(SimTaskTest, TaskStubLoopEnd_Describe)
{
    TaskStubLoopEnd task(1, 2);
    std::string desc = task.Describe();

    EXPECT_FALSE(desc.empty());
    EXPECT_TRUE(desc.find("LoopEnd") != std::string::npos);
}

TEST_F(SimTaskTest, TaskStubGraphSeparate_Describe)
{
    TaskStubGraphSeparate task;
    std::string desc = task.Describe();

    EXPECT_FALSE(desc.empty());
    EXPECT_TRUE(desc.find("SubGraphSeparate") != std::string::npos);
}

TEST_F(SimTaskTest, TaskStubBeingRead_Describe)
{
    DataSlice localSlice;
    localSlice.SetSize(512);
    DataSlice remoteSlice;
    remoteSlice.SetSize(512);
    LinkInfo link(LinkProtoStub::SDMA);
    TaskStubBeingRead task(1, link, localSlice, remoteSlice);

    std::string desc = task.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_TRUE(desc.find("BeingRead") != std::string::npos);
}

TEST_F(SimTaskTest, TaskStubBeingRead_GetRemoteRank)
{
    DataSlice localSlice;
    DataSlice remoteSlice;
    LinkInfo link(LinkProtoStub::SDMA);
    TaskStubBeingRead task(3, link, localSlice, remoteSlice);
    EXPECT_EQ(task.GetRemoteRank(), 3);
}

TEST_F(SimTaskTest, TaskStubBeingRead_GetLinkType)
{
    DataSlice localSlice;
    DataSlice remoteSlice;
    LinkInfo link(LinkProtoStub::SDMA);
    TaskStubBeingRead task(1, link, localSlice, remoteSlice);
    EXPECT_EQ(task.GetLinkType(), LinkProtoStub::SDMA);
}

TEST_F(SimTaskTest, TaskStubBeingRead_GetLocalSlice)
{
    DataSlice localSlice;
    localSlice.SetSize(256);
    DataSlice remoteSlice;
    LinkInfo link(LinkProtoStub::SDMA);
    TaskStubBeingRead task(1, link, localSlice, remoteSlice);
    EXPECT_EQ(task.GetLocalSlice().GetSize(), 256);
}

TEST_F(SimTaskTest, TaskStubBeingRead_GetRemoteSlice)
{
    DataSlice localSlice;
    DataSlice remoteSlice;
    remoteSlice.SetSize(128);
    LinkInfo link(LinkProtoStub::SDMA);
    TaskStubBeingRead task(1, link, localSlice, remoteSlice);
    EXPECT_EQ(task.GetRemoteSlice().GetSize(), 128);
}

TEST_F(SimTaskTest, TaskStubBeingRead_IsGenFromSync)
{
    DataSlice localSlice;
    DataSlice remoteSlice;
    LinkInfo link(LinkProtoStub::SDMA);
    TaskStubBeingRead task(1, link, localSlice, remoteSlice, true);
    EXPECT_TRUE(task.IsGenFromSync());
}

TEST_F(SimTaskTest, TaskStubBeingReadReduce_Describe)
{
    DataSlice localSlice;
    DataSlice remoteSlice;
    LinkInfo link(LinkProtoStub::RDMA);
    TaskStubBeingReadReduce task(
        2, link, localSlice, remoteSlice, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM);

    std::string desc = task.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_TRUE(desc.find("BeingReadReduce") != std::string::npos);
}

TEST_F(SimTaskTest, TaskStubBeingReadReduce_GetDataType)
{
    DataSlice localSlice;
    DataSlice remoteSlice;
    LinkInfo link(LinkProtoStub::RDMA);
    TaskStubBeingReadReduce task(
        2, link, localSlice, remoteSlice, HcclDataType::HCCL_DATA_TYPE_INT32, HcclReduceOp::HCCL_REDUCE_SUM);
    EXPECT_EQ(task.GetDataType(), HcclDataType::HCCL_DATA_TYPE_INT32);
}

TEST_F(SimTaskTest, TaskStubBeingReadReduce_GetLinkType)
{
    DataSlice localSlice;
    DataSlice remoteSlice;
    LinkInfo link(LinkProtoStub::RDMA);
    TaskStubBeingReadReduce task(
        2, link, localSlice, remoteSlice, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM);
    EXPECT_EQ(task.GetLinkType(), LinkProtoStub::RDMA);
}

TEST_F(SimTaskTest, TaskStubBeingReadReduce_GetLocalSlice)
{
    DataSlice localSlice;
    localSlice.SetSize(512);
    DataSlice remoteSlice;
    LinkInfo link(LinkProtoStub::RDMA);
    TaskStubBeingReadReduce task(
        2, link, localSlice, remoteSlice, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM);
    EXPECT_EQ(task.GetLocalSlice().GetSize(), 512);
}

TEST_F(SimTaskTest, TaskStubBeingReadReduce_GetReduceOp)
{
    DataSlice localSlice;
    DataSlice remoteSlice;
    LinkInfo link(LinkProtoStub::RDMA);
    TaskStubBeingReadReduce task(
        2, link, localSlice, remoteSlice, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_MAX);
    EXPECT_EQ(task.GetReduceOp(), HcclReduceOp::HCCL_REDUCE_MAX);
}

TEST_F(SimTaskTest, TaskStubBeingReadReduce_GetRemoteRank)
{
    DataSlice localSlice;
    DataSlice remoteSlice;
    LinkInfo link(LinkProtoStub::RDMA);
    TaskStubBeingReadReduce task(
        5, link, localSlice, remoteSlice, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM);
    EXPECT_EQ(task.GetRemoteRank(), 5);
}

TEST_F(SimTaskTest, TaskStubBeingReadReduce_GetRemoteSlice)
{
    DataSlice localSlice;
    DataSlice remoteSlice;
    remoteSlice.SetSize(256);
    LinkInfo link(LinkProtoStub::RDMA);
    TaskStubBeingReadReduce task(
        2, link, localSlice, remoteSlice, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM);
    EXPECT_EQ(task.GetRemoteSlice().GetSize(), 256);
}

TEST_F(SimTaskTest, TaskStubBeingReadReduce_IsGenFromSync)
{
    DataSlice localSlice;
    DataSlice remoteSlice;
    LinkInfo link(LinkProtoStub::RDMA);
    TaskStubBeingReadReduce task(
        2, link, localSlice, remoteSlice, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM, true);
    EXPECT_TRUE(task.IsGenFromSync());
}

TEST_F(SimTaskTest, TaskStubBeingWritten_Describe)
{
    DataSlice localSlice;
    localSlice.SetSize(512);
    DataSlice remoteSlice;
    remoteSlice.SetSize(512);
    LinkInfo link(LinkProtoStub::SDMA);
    TaskStubBeingWritten task(1, link, localSlice, remoteSlice);

    std::string desc = task.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_TRUE(desc.find("BeingWritten") != std::string::npos);
}

TEST_F(SimTaskTest, TaskStubBeingWritten_GetRemoteRank)
{
    DataSlice localSlice;
    DataSlice remoteSlice;
    LinkInfo link(LinkProtoStub::SDMA);
    TaskStubBeingWritten task(4, link, localSlice, remoteSlice);
    EXPECT_EQ(task.GetRemoteRank(), 4);
}

TEST_F(SimTaskTest, TaskStubBeingWritten_GetLinkType)
{
    DataSlice localSlice;
    DataSlice remoteSlice;
    LinkInfo link(LinkProtoStub::SDMA);
    TaskStubBeingWritten task(1, link, localSlice, remoteSlice);
    EXPECT_EQ(task.GetLinkType(), LinkProtoStub::SDMA);
}

TEST_F(SimTaskTest, TaskStubBeingWritten_GetLocalSlice)
{
    DataSlice localSlice;
    localSlice.SetSize(256);
    DataSlice remoteSlice;
    LinkInfo link(LinkProtoStub::SDMA);
    TaskStubBeingWritten task(1, link, localSlice, remoteSlice);
    EXPECT_EQ(task.GetLocalSlice().GetSize(), 256);
}

TEST_F(SimTaskTest, TaskStubBeingWritten_GetRemoteSlice)
{
    DataSlice localSlice;
    DataSlice remoteSlice;
    remoteSlice.SetSize(128);
    LinkInfo link(LinkProtoStub::SDMA);
    TaskStubBeingWritten task(1, link, localSlice, remoteSlice);
    EXPECT_EQ(task.GetRemoteSlice().GetSize(), 128);
}

TEST_F(SimTaskTest, TaskStubBeingWritten_IsGenFromSync)
{
    DataSlice localSlice;
    DataSlice remoteSlice;
    LinkInfo link(LinkProtoStub::SDMA);
    TaskStubBeingWritten task(1, link, localSlice, remoteSlice, true);
    EXPECT_TRUE(task.IsGenFromSync());
}

TEST_F(SimTaskTest, TaskStubBeingWrittenReduce_Describe)
{
    DataSlice localSlice;
    DataSlice remoteSlice;
    LinkInfo link(LinkProtoStub::RDMA);
    TaskStubBeingWrittenReduce task(
        2, link, localSlice, remoteSlice, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM);

    std::string desc = task.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_TRUE(desc.find("BeingWrittenReduce") != std::string::npos);
}

TEST_F(SimTaskTest, TaskStubBeingWrittenReduce_GetDataType)
{
    DataSlice localSlice;
    DataSlice remoteSlice;
    LinkInfo link(LinkProtoStub::RDMA);
    TaskStubBeingWrittenReduce task(
        2, link, localSlice, remoteSlice, HcclDataType::HCCL_DATA_TYPE_INT32, HcclReduceOp::HCCL_REDUCE_SUM);
    EXPECT_EQ(task.GetDataType(), HcclDataType::HCCL_DATA_TYPE_INT32);
}

TEST_F(SimTaskTest, TaskStubBeingWrittenReduce_GetLinkType)
{
    DataSlice localSlice;
    DataSlice remoteSlice;
    LinkInfo link(LinkProtoStub::RDMA);
    TaskStubBeingWrittenReduce task(
        2, link, localSlice, remoteSlice, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM);
    EXPECT_EQ(task.GetLinkType(), LinkProtoStub::RDMA);
}

TEST_F(SimTaskTest, TaskStubBeingWrittenReduce_GetLocalSlice)
{
    DataSlice localSlice;
    localSlice.SetSize(512);
    DataSlice remoteSlice;
    LinkInfo link(LinkProtoStub::RDMA);
    TaskStubBeingWrittenReduce task(
        2, link, localSlice, remoteSlice, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM);
    EXPECT_EQ(task.GetLocalSlice().GetSize(), 512);
}

TEST_F(SimTaskTest, TaskStubBeingWrittenReduce_GetRemoteRank)
{
    DataSlice localSlice;
    DataSlice remoteSlice;
    LinkInfo link(LinkProtoStub::RDMA);
    TaskStubBeingWrittenReduce task(
        6, link, localSlice, remoteSlice, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM);
    EXPECT_EQ(task.GetRemoteRank(), 6);
}

TEST_F(SimTaskTest, TaskStubBeingWrittenReduce_GetRemoteSlice)
{
    DataSlice localSlice;
    DataSlice remoteSlice;
    remoteSlice.SetSize(128);
    LinkInfo link(LinkProtoStub::RDMA);
    TaskStubBeingWrittenReduce task(
        2, link, localSlice, remoteSlice, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM);
    EXPECT_EQ(task.GetRemoteSlice().GetSize(), 128);
}

TEST_F(SimTaskTest, TaskStubBeingWrittenReduce_GetReduceOp)
{
    DataSlice localSlice;
    DataSlice remoteSlice;
    LinkInfo link(LinkProtoStub::RDMA);
    TaskStubBeingWrittenReduce task(
        2, link, localSlice, remoteSlice, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_MIN);
    EXPECT_EQ(task.GetReduceOp(), HcclReduceOp::HCCL_REDUCE_MIN);
}

TEST_F(SimTaskTest, TaskStubBeingWrittenReduce_IsGenFromSync)
{
    DataSlice localSlice;
    DataSlice remoteSlice;
    LinkInfo link(LinkProtoStub::RDMA);
    TaskStubBeingWrittenReduce task(
        2, link, localSlice, remoteSlice, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM, true);
    EXPECT_TRUE(task.IsGenFromSync());
}

TEST_F(SimTaskTest, TaskStubLocalBatchReduce_Describe)
{
    DataSlice src1;
    src1.SetSize(128);
    DataSlice src2;
    src2.SetSize(256);
    std::vector<DataSlice> srcSlices = {src1, src2};
    DataSlice dstSlice;
    dstSlice.SetSize(384);
    TaskStubLocalBatchReduce task(
        srcSlices, dstSlice, HcclDataType::HCCL_DATA_TYPE_INT32, HcclReduceOp::HCCL_REDUCE_SUM);

    std::string desc = task.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_TRUE(desc.find("LocalBatchReduce") != std::string::npos);
}

TEST_F(SimTaskTest, TaskStubLocalBatchReduce_GetSrcSlices)
{
    DataSlice src1;
    DataSlice src2;
    std::vector<DataSlice> srcSlices = {src1, src2};
    DataSlice dstSlice;
    TaskStubLocalBatchReduce task(
        srcSlices, dstSlice, HcclDataType::HCCL_DATA_TYPE_INT32, HcclReduceOp::HCCL_REDUCE_SUM);

    EXPECT_EQ(task.GetSrcSlices().size(), 2u);
}

TEST_F(SimTaskTest, TaskStubLocalBatchReduce_GetSrcSlice)
{
    DataSlice src1;
    src1.SetSize(100);
    DataSlice src2;
    src2.SetSize(200);
    std::vector<DataSlice> srcSlices = {src1, src2};
    DataSlice dstSlice;
    TaskStubLocalBatchReduce task(
        srcSlices, dstSlice, HcclDataType::HCCL_DATA_TYPE_INT32, HcclReduceOp::HCCL_REDUCE_SUM);

    EXPECT_EQ(task.GetSrcSlice(0).GetSize(), 100);
    EXPECT_EQ(task.GetSrcSlice(1).GetSize(), 200);
}

TEST_F(SimTaskTest, TaskStubLocalBatchReduce_GetDstSlice)
{
    DataSlice src1;
    std::vector<DataSlice> srcSlices = {src1};
    DataSlice dstSlice;
    dstSlice.SetSize(500);
    TaskStubLocalBatchReduce task(
        srcSlices, dstSlice, HcclDataType::HCCL_DATA_TYPE_INT32, HcclReduceOp::HCCL_REDUCE_SUM);

    EXPECT_EQ(task.GetDstSlice().GetSize(), 500);
}

TEST_F(SimTaskTest, TaskStubLocalBatchReduce_GetDataType)
{
    DataSlice src1;
    std::vector<DataSlice> srcSlices = {src1};
    DataSlice dstSlice;
    TaskStubLocalBatchReduce task(
        srcSlices, dstSlice, HcclDataType::HCCL_DATA_TYPE_FP16, HcclReduceOp::HCCL_REDUCE_SUM);

    EXPECT_EQ(task.GetDataType(), HcclDataType::HCCL_DATA_TYPE_FP16);
}

TEST_F(SimTaskTest, TaskStubLocalBatchReduce_GetReduceOp)
{
    DataSlice src1;
    std::vector<DataSlice> srcSlices = {src1};
    DataSlice dstSlice;
    TaskStubLocalBatchReduce task(
        srcSlices, dstSlice, HcclDataType::HCCL_DATA_TYPE_INT32, HcclReduceOp::HCCL_REDUCE_MIN);

    EXPECT_EQ(task.GetReduceOp(), HcclReduceOp::HCCL_REDUCE_MIN);
}

TEST_F(SimTaskTest, TaskStubLocalPostTo_SetPostQid)
{
    TaskStubLocalPostTo task(100);
    task.SetPostQid(5);
    EXPECT_EQ(task.GetPostQid(), 5u);
}

TEST_F(SimTaskTest, TaskStubLocalPostTo_SetWaitQid)
{
    TaskStubLocalPostTo task(100);
    task.SetWaitQid(10);
    EXPECT_EQ(task.GetWaitQid(), 10u);
}

TEST_F(SimTaskTest, TaskStubLocalPostTo_GetTopicId)
{
    TaskStubLocalPostTo task(100);
    EXPECT_EQ(task.GetTopicId(), 100u);
}

TEST_F(SimTaskTest, TaskStubLocalPostTo_SetTopicId)
{
    TaskStubLocalPostTo task(100);
    task.SetTopicId(250);
    EXPECT_EQ(task.GetTopicId(), 250u);
}

TEST_F(SimTaskTest, TaskStubLocalPostTo_GetTopicIdBack)
{
    TaskStubLocalPostTo task(100);
    EXPECT_EQ(task.GetTopicIdBack(), 100u);
}

TEST_F(SimTaskTest, TaskStubLocalPostTo_GetNotifyIdBack)
{
    TaskStubLocalPostTo task(200);
    EXPECT_EQ(task.GetNotifyIdBack(), 200u);
}

TEST_F(SimTaskTest, TaskStubLocalPostTo_IsInvalidPost)
{
    TaskStubLocalPostTo task(100);
    EXPECT_FALSE(task.IsInvalidPost());
}

TEST_F(SimTaskTest, TaskStubLocalReduce_IsGenFromSync)
{
    DataSlice srcSlice;
    DataSlice dstSlice;
    TaskStubLocalReduce task(
        srcSlice, dstSlice, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM, true);
    EXPECT_TRUE(task.IsGenFromSync());
}

TEST_F(SimTaskTest, TaskStubPost_SetTopicId)
{
    LinkInfo link(LinkProtoStub::RDMA);
    TaskStubPost task(1, link, 100);
    task.SetTopicId(50);
    EXPECT_EQ(task.GetTopicId(), 50u);
}

TEST_F(SimTaskTest, TaskStubWait_SetRemoteRank)
{
    LinkInfo link(LinkProtoStub::RDMA);
    TaskStubWait task(1, link, 100);
    task.SetRemoteRank(20);
    EXPECT_EQ(task.GetRemoteRank(), 20u);
}

TEST_F(SimTaskTest, TaskStubReadReduce_Describe)
{
    DataSlice localSlice;
    localSlice.SetSize(256);
    DataSlice remoteSlice;
    remoteSlice.SetSize(256);
    LinkInfo link(LinkProtoStub::SDMA);
    TaskStubReadReduce task(
        3, link, localSlice, remoteSlice, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM);

    std::string desc = task.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_TRUE(desc.find("ReadReduce") != std::string::npos);
}

TEST_F(SimTaskTest, TaskStubWriteReduce_Describe)
{
    DataSlice localSlice;
    localSlice.SetSize(256);
    DataSlice remoteSlice;
    remoteSlice.SetSize(256);
    LinkInfo link(LinkProtoStub::RDMA);
    TaskStubWriteReduce task(
        3, link, localSlice, remoteSlice, HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM);

    std::string desc = task.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_TRUE(desc.find("WriteReduce") != std::string::npos);
}

TEST_F(SimTaskTest, TaskStubLocalPostToShadow_Describe)
{
    TaskStubLocalPostToShadow task(2, 5, 10);
    std::string desc = task.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_TRUE(desc.find("LocalPostToShadow") != std::string::npos);
}

TEST_F(SimTaskTest, TaskStubLocalPostToShadow_GetNeighborRank)
{
    TaskStubLocalPostToShadow task(3, 5, 10);
    EXPECT_EQ(task.GetNeighborRank(), 3u);
}

TEST_F(SimTaskTest, TaskStubLocalWaitFromShadow_Describe)
{
    TaskStubLocalWaitFromShadow task(1, 3, 7);
    std::string desc = task.Describe();
    EXPECT_FALSE(desc.empty());
    EXPECT_TRUE(desc.find("LocalWaitFromShadow") != std::string::npos);
}

TEST_F(SimTaskTest, TaskStubLocalWaitFromShadow_GetNeighborRank)
{
    TaskStubLocalWaitFromShadow task(4, 3, 7);
    EXPECT_EQ(task.GetNeighborRank(), 4u);
}
} // namespace HcclSim

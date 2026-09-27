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
#define private public
#define protected public
#include "instruction.h"
#include "invalid_params_exception.h"
#include "data_buffer.h"
#undef private
#undef protected

using namespace Hccl;
using namespace std;
class InstructionTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "InstructionTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "InstructionTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        localRank = 0;
        remoteRank = 1;
        dataType = DataType::FP32;
        reduceOp = ReduceOp::SUM;
        u64 size = 100;
        DataBuffer srcBuffer(0x1234560, size);
        DataBuffer dstBuffer(0x1321000, size);

        NotifyType notifyType = NotifyType::NORMAL;
        u32 bitValue = 0;
        u32 topicId = 0;

        BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
        linkData = new LinkData(portType, localRank, remoteRank, 0, 1);
        u64 sliceSize = 0x1000;
        localSlice = new DataSlice(BufferType::SCRATCH, 0, sliceSize);
        remoteSlice = new DataSlice(BufferType::SCRATCH, 0, sliceSize);
        srcSlice = new DataSlice(BufferType::SCRATCH, 0, sliceSize);
        dstSlice = new DataSlice(BufferType::SCRATCH, sliceSize, sliceSize);

        localRmaSlice = new RmaBufSliceLite(100, 200, 300, 400);
        remoteRmaSlice = new RmtRmaBufSliceLite(100, 200, 300, 400, 500, UINT32_MAX);

        insAicpuReduce = new InsAicpuReduce(*srcSlice, *dstSlice, dataType, reduceOp);
        insStreamSync = new InsStreamSync();
        insBatchRead = new InsBatchRead(remoteRank, *linkData);
        insBatchWrite = new InsBatchWrite(remoteRank, *linkData);
        insWriteWithFin = new InsWriteWithFin(remoteRank, *linkData, *localSlice, *remoteSlice, NotifyType::NORMAL);
        insWriteReduce = new InsWriteReduce(remoteRank, *linkData, *localSlice, *remoteSlice, dataType, reduceOp);
        insWriteReduceWithFin = new InsWriteReduceWithFin(
            remoteRank, *linkData, *localSlice, *remoteSlice, dataType, reduceOp, NotifyType::NORMAL);
        insReadExtend = new InsReadExtend(remoteRank, *linkData, srcBuffer, dstBuffer);
        insReadReduceExtend = new InsReadReduceExtend(remoteRank, *linkData, srcBuffer, dstBuffer, dataType, reduceOp);
        insWriteWithFinExtend = new InsWriteWithFinExtend(remoteRank, *linkData, srcBuffer, dstBuffer);
        insWriteReduceExtend
            = new InsWriteReduceExtend(remoteRank, *linkData, srcBuffer, dstBuffer, dataType, reduceOp);
        insWriteExtend = new InsWriteExtend(remoteRank, *linkData, srcBuffer, dstBuffer);
        insWriteReduceWithFinExtend = new InsWriteReduceWithFinExtend(
            remoteRank, *linkData, srcBuffer, dstBuffer, dataType, reduceOp, NotifyType::NORMAL);
        std::cout << "A Test case in InstructionTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        delete linkData;

        delete localSlice;
        delete remoteSlice;
        delete srcSlice;
        delete dstSlice;

        delete localRmaSlice;
        delete remoteRmaSlice;

        delete insAicpuReduce;
        delete insStreamSync;
        delete insBatchRead;
        delete insBatchWrite;
        delete insWriteReduce;
        delete insWriteWithFin;
        delete insWriteReduceWithFin;
        delete insReadExtend;
        delete insReadReduceExtend;
        delete insWriteWithFinExtend;
        delete insWriteReduceExtend;
        delete insWriteExtend;
        delete insWriteReduceWithFinExtend;

        delete localRmaSlice;
        delete remoteRmaSlice;
        std::cout << "A Test case in InstructionTest TearDown" << std::endl;
    }
    RankId localRank;
    RankId remoteRank;

    LinkData* linkData;

    DataSlice* localSlice;
    DataSlice* remoteSlice;
    DataSlice* srcSlice;
    DataSlice* dstSlice;

    RmaBufSliceLite* localRmaSlice;
    RmtRmaBufSliceLite* remoteRmaSlice;

    InsAicpuReduce* insAicpuReduce;
    InsStreamSync* insStreamSync;
    InsBatchRead* insBatchRead;
    InsBatchWrite* insBatchWrite;
    InsWriteReduce* insWriteReduce;
    InsWriteWithFin* insWriteWithFin;
    InsWriteReduceWithFin* insWriteReduceWithFin;
    InsReadExtend* insReadExtend;
    InsReadReduceExtend* insReadReduceExtend;
    InsWriteWithFinExtend* insWriteWithFinExtend;
    InsWriteReduceExtend* insWriteReduceExtend;
    InsWriteExtend* insWriteExtend;
    InsWriteReduceWithFinExtend* insWriteReduceWithFinExtend;
    DataType dataType;
    ReduceOp reduceOp;
};

TEST(InstructionTest, test_print_all_ins)
{
    RankId localRank = 0;
    RankId remoteRank = 1;
    DataType dataType = DataType::FP32;
    ReduceOp reduceOp = ReduceOp::SUM;

    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData* linkData = new LinkData(portType, localRank, remoteRank, 0, 1);
    u64 sliceSize = 0x1000;
    DataSlice* localSlice = new DataSlice(BufferType::SCRATCH, 0, sliceSize);
    DataSlice* remoteSlice = new DataSlice(BufferType::SCRATCH, 0, sliceSize);

    InsWriteReduce* insWriteReduce
        = new InsWriteReduce(remoteRank, *linkData, *localSlice, *remoteSlice, dataType, reduceOp);
    InsWriteWithFin* insWriteWithFin
        = new InsWriteWithFin(remoteRank, *linkData, *localSlice, *remoteSlice, NotifyType::NORMAL);
    InsWriteReduceWithFin* insWriteReduceWithFin = new InsWriteReduceWithFin(
        remoteRank, *linkData, *localSlice, *remoteSlice, dataType, reduceOp, NotifyType::NORMAL);
    InsStreamSync* insStreamSync = new InsStreamSync();

    cout << insWriteReduce->Describe() << endl;
    cout << insWriteWithFin->Describe() << endl;
    cout << insWriteReduceWithFin->Describe() << endl;
    cout << insStreamSync->Describe() << endl;

    delete linkData;
    delete insStreamSync;
    delete localSlice;
    delete remoteSlice;

    delete insWriteReduce;
    delete insWriteWithFin;
    delete insWriteReduceWithFin;
}

TEST(InstructionTest, test_ins_aicpu_reduce)
{
    u64 sliceSize = 0x1000;
    DataType dataType = DataType::FP32;
    ReduceOp reduceOp = ReduceOp::SUM;
    DataSlice* srcSlice = new DataSlice(BufferType::SCRATCH, 0, sliceSize);
    DataSlice* dstSlice = new DataSlice(BufferType::SCRATCH, sliceSize, sliceSize);
    InsAicpuReduce* insAicpuReduce = new InsAicpuReduce(*srcSlice, *dstSlice, dataType, reduceOp);
    cout << insAicpuReduce->Describe() << endl;
    DataSlice testDstSlice = insAicpuReduce->GetDstSlice();
    DataSlice testSrcSlice = insAicpuReduce->GetSrcSlice();

    EXPECT_EQ(true, testDstSlice == *dstSlice);
    EXPECT_EQ(true, testSrcSlice == *srcSlice);
    EXPECT_EQ(true, dataType == insAicpuReduce->GetDataType());
    EXPECT_EQ(true, reduceOp == insAicpuReduce->GetReduceOp());
    delete insAicpuReduce;
    delete srcSlice;
    delete dstSlice;
}

TEST(InstructionTest, test_ins_write_reduce)
{
    RankId localRank = 0;
    RankId remoteRank = 1;
    DataType dataType = DataType::FP32;
    ReduceOp reduceOp = ReduceOp::SUM;

    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData* linkData = new LinkData(portType, localRank, remoteRank, 0, 1);
    u64 sliceSize = 0x1000;
    DataSlice* localSlice = new DataSlice(BufferType::SCRATCH, 0, sliceSize);
    DataSlice* remoteSlice = new DataSlice(BufferType::SCRATCH, 0, sliceSize);

    InsWriteReduce* insWriteReduce
        = new InsWriteReduce(remoteRank, *linkData, *localSlice, *remoteSlice, dataType, reduceOp);
    cout << insWriteReduce->Describe() << endl;

    EXPECT_EQ(true, dataType == insWriteReduce->GetDataType());
    EXPECT_EQ(true, reduceOp == insWriteReduce->GetReduceOp());
    EXPECT_EQ(remoteRank, insWriteReduce->GetRemoteRank());
    EXPECT_EQ(true, *(insWriteReduce->GetLink()) == *linkData);

    EXPECT_EQ(true, insWriteReduce->GetLocalSlice() == *localSlice);
    EXPECT_EQ(true, insWriteReduce->GetRemoteSlice() == *remoteSlice);

    delete linkData;

    delete localSlice;
    delete remoteSlice;

    delete insWriteReduce;
}

TEST(InstructionTest, test_ins_write_with_fin)
{
    RankId localRank = 0;
    RankId remoteRank = 1;

    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData* linkData = new LinkData(portType, localRank, remoteRank, 0, 1);
    u64 sliceSize = 0x1000;
    DataSlice* localSlice = new DataSlice(BufferType::SCRATCH, 0, sliceSize);
    DataSlice* remoteSlice = new DataSlice(BufferType::SCRATCH, 0, sliceSize);

    InsWriteWithFin* insWriteWithFin
        = new InsWriteWithFin(remoteRank, *linkData, *localSlice, *remoteSlice, NotifyType::NORMAL);
    cout << insWriteWithFin->Describe() << endl;

    EXPECT_EQ(remoteRank, insWriteWithFin->GetRemoteRank());
    EXPECT_EQ(true, *(insWriteWithFin->GetLink()) == *linkData);

    EXPECT_EQ(true, insWriteWithFin->GetLocalSlice() == *localSlice);
    EXPECT_EQ(true, insWriteWithFin->GetRemoteSlice() == *remoteSlice);
    EXPECT_EQ(0, insWriteWithFin->GetTopicId());
    EXPECT_EQ(0, insWriteWithFin->GetBitValue());

    delete linkData;

    delete localSlice;
    delete remoteSlice;

    delete insWriteWithFin;
}

TEST(InstructionTest, test_ins_write_reduce_with_fin)
{
    RankId localRank = 0;
    RankId remoteRank = 1;
    DataType dataType = DataType::FP32;
    ReduceOp reduceOp = ReduceOp::SUM;

    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData* linkData = new LinkData(portType, localRank, remoteRank, 0, 1);
    u64 sliceSize = 0x1000;
    DataSlice* localSlice = new DataSlice(BufferType::SCRATCH, 0, sliceSize);
    DataSlice* remoteSlice = new DataSlice(BufferType::SCRATCH, 0, sliceSize);

    InsWriteReduceWithFin* insWriteReduceWithFin = new InsWriteReduceWithFin(
        remoteRank, *linkData, *localSlice, *remoteSlice, dataType, reduceOp, NotifyType::NORMAL);
    cout << insWriteReduceWithFin->Describe() << endl;

    EXPECT_EQ(true, dataType == insWriteReduceWithFin->GetDataType());
    EXPECT_EQ(true, reduceOp == insWriteReduceWithFin->GetReduceOp());
    EXPECT_EQ(remoteRank, insWriteReduceWithFin->GetRemoteRank());
    EXPECT_EQ(true, *(insWriteReduceWithFin->GetLink()) == *linkData);

    EXPECT_EQ(true, insWriteReduceWithFin->GetLocalSlice() == *localSlice);
    EXPECT_EQ(true, insWriteReduceWithFin->GetRemoteSlice() == *remoteSlice);
    EXPECT_EQ(0, insWriteReduceWithFin->GetTopicId());
    EXPECT_EQ(0, insWriteReduceWithFin->GetBitValue());

    delete linkData;

    delete localSlice;
    delete remoteSlice;

    delete insWriteReduceWithFin;
}

TEST(InstructionTest, test_ins_wait_group_fin)
{
    u32 topicId = 100;
    u32 value = 200;
    LinkData link(BasePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB), 0, 1, 0, 1);

    InsWaitGroupFin insWaitGroupFin(topicId);
    insWaitGroupFin.SetValue(value);

    insWaitGroupFin.Append(link);
    cout << insWaitGroupFin.Describe() << endl;

    EXPECT_EQ(topicId, insWaitGroupFin.GetTopicId());
    EXPECT_EQ(value, insWaitGroupFin.GetValue());
}

TEST(InstructionTest, test_ins_ReadExtend)
{
    RankId localRank = 0;
    RankId remoteRank = 1;
    DataType dataType = DataType::FP32;
    ReduceOp reduceOp = ReduceOp::SUM;
    u64 size = 100;
    DataBuffer srcBuffer(0x1234560, size);
    DataBuffer dstBuffer(0x1321000, size);

    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData* linkData = new LinkData(portType, localRank, remoteRank, 0, 1);
    InsReadExtend* insReadExtend = new InsReadExtend(remoteRank, *linkData, srcBuffer, dstBuffer);
    cout << insReadExtend->Describe() << endl;
    EXPECT_EQ(remoteRank, insReadExtend->GetRemoteRank());
    EXPECT_EQ(true, *(insReadExtend->GetLink()) == *linkData);
    insReadExtend->GetLocalBuffer();
    insReadExtend->GetRemoteBuffer();

    delete linkData;
    delete insReadExtend;
}

TEST(InstructionTest, test_ins_ReadReduceExtend)
{
    RankId localRank = 0;
    RankId remoteRank = 1;
    DataType dataType = DataType::FP32;
    ReduceOp reduceOp = ReduceOp::SUM;
    u64 size = 100;
    DataBuffer srcBuffer(0x1234560, size);
    DataBuffer dstBuffer(0x1321000, size);
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData* linkData = new LinkData(portType, localRank, remoteRank, 0, 1);

    InsReadReduceExtend* insReadReduceExtend
        = new InsReadReduceExtend(remoteRank, *linkData, srcBuffer, dstBuffer, dataType, reduceOp);
    cout << insReadReduceExtend->Describe() << endl;
    EXPECT_EQ(remoteRank, insReadReduceExtend->GetRemoteRank());
    EXPECT_EQ(true, *(insReadReduceExtend->GetLink()) == *linkData);
    insReadReduceExtend->GetLocalBuffer();
    insReadReduceExtend->GetRemoteBuffer();
    EXPECT_EQ(dataType, insReadReduceExtend->GetDataType());
    EXPECT_EQ(reduceOp, insReadReduceExtend->GetReduceOp());

    delete linkData;
    delete insReadReduceExtend;
}

TEST(InstructionTest, test_ins_WriteExtend)
{
    RankId localRank = 0;
    RankId remoteRank = 1;
    DataType dataType = DataType::FP32;
    ReduceOp reduceOp = ReduceOp::SUM;
    u64 size = 100;
    DataBuffer srcBuffer(0x1234560, size);
    DataBuffer dstBuffer(0x1321000, size);
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData* linkData = new LinkData(portType, localRank, remoteRank, 0, 1);

    InsWriteExtend* insWriteExtend = new InsWriteExtend(remoteRank, *linkData, srcBuffer, dstBuffer);
    InsWriteWithFinExtend* insWriteWithFinExtend
        = new InsWriteWithFinExtend(remoteRank, *linkData, srcBuffer, dstBuffer);
    EXPECT_EQ(remoteRank, insWriteExtend->GetRemoteRank());
    EXPECT_EQ(remoteRank, insWriteWithFinExtend->GetRemoteRank());

    delete linkData;
    delete insWriteExtend;
    delete insWriteWithFinExtend;
}

TEST(InstructionTest, test_ins_WriteReduceExtend)
{
    RankId localRank = 0;
    RankId remoteRank = 1;
    DataType dataType = DataType::FP32;
    ReduceOp reduceOp = ReduceOp::SUM;
    u64 size = 100;
    DataBuffer srcBuffer(0x1234560, size);
    DataBuffer dstBuffer(0x1321000, size);
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData* linkData = new LinkData(portType, localRank, remoteRank, 0, 1);
    InsWriteReduceExtend* insWriteReduceExtend
        = new InsWriteReduceExtend(remoteRank, *linkData, srcBuffer, dstBuffer, dataType, reduceOp);
    cout << insWriteReduceExtend->Describe() << endl;
    EXPECT_EQ(remoteRank, insWriteReduceExtend->GetRemoteRank());
    EXPECT_EQ(true, *(insWriteReduceExtend->GetLink()) == *linkData);
    insWriteReduceExtend->GetLocalBuffer();
    insWriteReduceExtend->GetRemoteBuffer();
    EXPECT_EQ(dataType, insWriteReduceExtend->GetDataType());
    EXPECT_EQ(reduceOp, insWriteReduceExtend->GetReduceOp());

    delete linkData;
    delete insWriteReduceExtend;
}

TEST(InstructionTest, test_ins_WriteReduceWithFinExtend)
{
    RankId localRank = 0;
    RankId remoteRank = 1;
    DataType dataType = DataType::FP32;
    ReduceOp reduceOp = ReduceOp::SUM;
    u64 size = 100;
    DataBuffer srcBuffer(0x1234560, size);
    DataBuffer dstBuffer(0x1321000, size);
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData* linkData = new LinkData(portType, localRank, remoteRank, 0, 1);

    InsWriteReduceWithFinExtend* insWriteReduceWithFinExtend = new InsWriteReduceWithFinExtend(
        remoteRank, *linkData, srcBuffer, dstBuffer, dataType, reduceOp, NotifyType::NORMAL);
    cout << insWriteReduceWithFinExtend->Describe() << endl;
    EXPECT_EQ(remoteRank, insWriteReduceWithFinExtend->GetRemoteRank());
    EXPECT_EQ(true, *(insWriteReduceWithFinExtend->GetLink()) == *linkData);
    insWriteReduceWithFinExtend->GetLocalBuffer();
    insWriteReduceWithFinExtend->GetRemoteBuffer();
    EXPECT_EQ(dataType, insWriteReduceWithFinExtend->GetDataType());
    EXPECT_EQ(reduceOp, insWriteReduceWithFinExtend->GetReduceOp());

    delete linkData;
    delete insWriteReduceWithFinExtend;
}

TEST(InstructionTest, test_ins_BatchOneSidedRead)
{
    RankId localRank = 0;
    RankId remoteRank = 1;
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData* linkData = new LinkData(portType, localRank, remoteRank, 0, 1);
    RmaBufSliceLite* localRmaSlice = new RmaBufSliceLite(100, 200, 300, 400);
    RmtRmaBufSliceLite* remoteRmaSlice = new RmtRmaBufSliceLite(100, 200, 300, 400, 500, UINT32_MAX);
    InsBatchOneSidedRead* insBatchOneSidedRead
        = new InsBatchOneSidedRead(remoteRank, *linkData, {*localRmaSlice}, {*remoteRmaSlice});
    cout << insBatchOneSidedRead->Describe() << endl;
    EXPECT_EQ(remoteRank, insBatchOneSidedRead->GetRemoteRank());
    EXPECT_EQ(true, *(insBatchOneSidedRead->GetLink()) == *linkData);
    EXPECT_EQ(1, insBatchOneSidedRead->GetLocalSlice().size());
    EXPECT_EQ(1, insBatchOneSidedRead->GetRemoteSlice().size());
    delete insBatchOneSidedRead;
    delete linkData;
    delete localRmaSlice;
    delete remoteRmaSlice;
}

TEST(InstructionTest, test_ins_BatchOneSidedWrite)
{
    RankId localRank = 0;
    RankId remoteRank = 1;
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData* linkData = new LinkData(portType, localRank, remoteRank, 0, 1);
    RmaBufSliceLite* localRmaSlice = new RmaBufSliceLite(100, 200, 300, 400);
    RmtRmaBufSliceLite* remoteRmaSlice = new RmtRmaBufSliceLite(100, 200, 300, 400, 500, UINT32_MAX);
    InsBatchOneSidedWrite* insBatchOneSidedWrite
        = new InsBatchOneSidedWrite(remoteRank, *linkData, {*localRmaSlice}, {*remoteRmaSlice});
    cout << insBatchOneSidedWrite->Describe() << endl;
    EXPECT_EQ(remoteRank, insBatchOneSidedWrite->GetRemoteRank());
    EXPECT_EQ(true, *(insBatchOneSidedWrite->GetLink()) == *linkData);
    EXPECT_EQ(1, insBatchOneSidedWrite->GetLocalSlice().size());
    EXPECT_EQ(1, insBatchOneSidedWrite->GetRemoteSlice().size());
    delete insBatchOneSidedWrite;
    delete linkData;
    delete localRmaSlice;
    delete remoteRmaSlice;
}

TEST(InstructionTest, test_ins_BatchRead)
{
    RankId localRank = 0;
    RankId remoteRank = 1;
    DataType dataType = DataType::FP32;
    ReduceOp reduceOp = ReduceOp::SUM;

    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData* linkData = new LinkData(portType, localRank, remoteRank, 0, 1);
    u64 sliceSize = 0x1000;
    DataSlice* localSlice = new DataSlice(BufferType::SCRATCH, 0, sliceSize);
    DataSlice* remoteSlice = new DataSlice(BufferType::SCRATCH, 0, sliceSize);

    InsBatchRead* insBatchRead = new InsBatchRead(remoteRank, *linkData);
    cout << insBatchRead->Describe() << endl;
    unique_ptr<Instruction> readIns = make_unique<InsRead>(remoteRank, *linkData, *localSlice, *remoteSlice);
    unique_ptr<Instruction> readReduceIns
        = make_unique<InsReadReduce>(remoteRank, *linkData, *localSlice, *remoteSlice, dataType, reduceOp);
    insBatchRead->PushReadIns(move(readIns));
    insBatchRead->PushReadIns(move(readReduceIns));

    EXPECT_EQ(remoteRank, insBatchRead->GetRemoteRank());
    EXPECT_EQ(true, *(insBatchRead->GetLink()) == *linkData);
    EXPECT_EQ(readIns, nullptr);
    EXPECT_EQ(readReduceIns, nullptr);
    EXPECT_EQ(insBatchRead->readInsVec.size(), 2);

    delete linkData;
    delete localSlice;
    delete remoteSlice;
    delete insBatchRead;
}

TEST(InstructionTest, test_ins_batch_read_PushReadIns_fail)
{
    RankId localRank = 0;
    RankId remoteRank = 1;
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData* linkData = new LinkData(portType, localRank, remoteRank, 0, 1);
    u64 sliceSize = 0x1000;
    DataSlice* localSlice = new DataSlice(BufferType::SCRATCH, 0, sliceSize);
    DataSlice* remoteSlice = new DataSlice(BufferType::SCRATCH, 0, sliceSize);
    unique_ptr<Instruction> writeIns = make_unique<InsWrite>(remoteRank, *linkData, *localSlice, *remoteSlice);
    InsBatchRead* insBatchRead = new InsBatchRead(remoteRank, *linkData);

    EXPECT_THROW(insBatchRead->PushReadIns(move(writeIns)), NotSupportException);

    delete linkData;
    delete localSlice;
    delete remoteSlice;
    delete insBatchRead;
}

TEST(InstructionTest, test_ins_BatchWrite)
{
    RankId localRank = 0;
    RankId remoteRank = 1;
    DataType dataType = DataType::FP32;
    ReduceOp reduceOp = ReduceOp::SUM;

    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData* linkData = new LinkData(portType, localRank, remoteRank, 0, 1);
    u64 sliceSize = 0x1000;
    DataSlice* localSlice = new DataSlice(BufferType::SCRATCH, 0, sliceSize);
    DataSlice* remoteSlice = new DataSlice(BufferType::SCRATCH, 0, sliceSize);

    InsBatchWrite* insBatchWrite = new InsBatchWrite(remoteRank, *linkData);
    cout << insBatchWrite->Describe() << endl;
    unique_ptr<Instruction> writeIns = make_unique<InsWrite>(remoteRank, *linkData, *localSlice, *remoteSlice);
    unique_ptr<Instruction> writeReduceIns
        = make_unique<InsWriteReduce>(remoteRank, *linkData, *localSlice, *remoteSlice, dataType, reduceOp);
    insBatchWrite->PushWriteIns(move(writeIns));
    insBatchWrite->PushWriteIns(move(writeReduceIns));

    EXPECT_EQ(remoteRank, insBatchWrite->GetRemoteRank());
    EXPECT_EQ(true, *(insBatchWrite->GetLink()) == *linkData);
    EXPECT_EQ(writeIns, nullptr);
    EXPECT_EQ(writeReduceIns, nullptr);
    EXPECT_EQ(insBatchWrite->writeInsVec.size(), 2);

    delete linkData;
    delete localSlice;
    delete remoteSlice;
    delete insBatchWrite;
}

TEST(InstructionTest, test_ins_batch_write_PushWriteIns_fail)
{
    RankId localRank = 0;
    RankId remoteRank = 1;
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData* linkData = new LinkData(portType, localRank, remoteRank, 0, 1);
    u64 sliceSize = 0x1000;
    DataSlice* localSlice = new DataSlice(BufferType::SCRATCH, 0, sliceSize);
    DataSlice* remoteSlice = new DataSlice(BufferType::SCRATCH, 0, sliceSize);
    unique_ptr<Instruction> readIns = make_unique<InsRead>(remoteRank, *linkData, *localSlice, *remoteSlice);
    InsBatchWrite* insBatchWrite = new InsBatchWrite(remoteRank, *linkData);

    EXPECT_THROW(insBatchWrite->PushWriteIns(move(readIns)), NotSupportException);
    delete linkData;
    delete localSlice;
    delete remoteSlice;
    delete insBatchWrite;
}

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

        postQid = 0;
        waitQid = 1;
        topicId = 0;
        waitValue = 100;
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

        insPostTo = new InsLocalPostTo(waitQid, NotifyType::NORMAL, topicId);
        insPostTo->SetPostQid(postQid);
        insWaitFrom = new InsLocalWaitFrom(postQid, NotifyType::NORMAL, topicId);
        insWaitFrom->SetWaitQid(waitQid);

        insWaitGroup = new InsLocalWaitGroup(topicId);

        insLocalCopy = new InsLocalCopy(*srcSlice, *dstSlice);
        insLocalReduce = new InsLocalReduce(*srcSlice, *dstSlice, dataType, reduceOp);
        insAicpuReduce = new InsAicpuReduce(*srcSlice, *dstSlice, dataType, reduceOp);
        insStreamSync = new InsStreamSync();

        insPostReady = new InsPostReady(remoteRank, *linkData);
        insWaitReady = new InsWaitReady(remoteRank, *linkData);

        insWaitGroupFin = new InsWaitGroupFin(topicId);
        insWaitGroupFin->SetValue(waitValue);

        insPostFin = new InsPostFin(remoteRank, *linkData);
        insWaitFin = new InsWaitFin(remoteRank, *linkData);

        insPostFinAck = new InsPostFinAck(remoteRank, *linkData);
        insWaitFinAck = new InsWaitFinAck(remoteRank, *linkData);

        insRead = new InsRead(remoteRank, *linkData, *localSlice, *remoteSlice);
        insWrite = new InsWrite(remoteRank, *linkData, *localSlice, *remoteSlice);
        insWriteWithFin = new InsWriteWithFin(remoteRank, *linkData, *localSlice, *remoteSlice, NotifyType::NORMAL);

        insBatchRead = new InsBatchRead(remoteRank, *linkData);
        insBatchWrite = new InsBatchWrite(remoteRank, *linkData);

        insReadReduce = new InsReadReduce(remoteRank, *linkData, *localSlice, *remoteSlice, dataType, reduceOp);
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

        insBatchOneSidedRead = new InsBatchOneSidedRead(remoteRank, *linkData, {*localRmaSlice}, {*remoteRmaSlice});
        insBatchOneSidedWrite = new InsBatchOneSidedWrite(remoteRank, *linkData, {*localRmaSlice}, {*remoteRmaSlice});
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

        delete insPostTo;
        delete insWaitFrom;
        delete insWaitGroup;
        delete insLocalCopy;
        delete insLocalReduce;
        delete insAicpuReduce;
        delete insStreamSync;

        delete insPostReady;
        delete insWaitReady;

        delete insPostFin;
        delete insWaitFin;
        delete insWaitGroupFin;

        delete insPostFinAck;
        delete insWaitFinAck;
        delete insRead;
        delete insReadReduce;

        delete insBatchRead;
        delete insBatchWrite;

        delete insWrite;
        delete insWriteReduce;
        delete insWriteWithFin;
        delete insWriteReduceWithFin;
        delete insReadExtend;
        delete insReadReduceExtend;
        delete insWriteWithFinExtend;
        delete insWriteReduceExtend;
        delete insWriteExtend;
        delete insWriteReduceWithFinExtend;
        delete insBatchOneSidedRead;
        delete insBatchOneSidedWrite;
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

    InsLocalPostTo* insPostTo;
    InsLocalWaitFrom* insWaitFrom;
    InsLocalWaitGroup* insWaitGroup;

    InsLocalCopy* insLocalCopy;
    InsLocalReduce* insLocalReduce;
    InsAicpuReduce* insAicpuReduce;
    InsStreamSync* insStreamSync;

    InsPostReady* insPostReady;
    InsWaitReady* insWaitReady;

    InsWaitGroupFin* insWaitGroupFin;

    InsPostFin* insPostFin;
    InsWaitFin* insWaitFin;

    InsPostFinAck* insPostFinAck;
    InsWaitFinAck* insWaitFinAck;

    InsRead* insRead;
    InsReadReduce* insReadReduce;

    InsBatchRead* insBatchRead;
    InsBatchWrite* insBatchWrite;

    InsWrite* insWrite;
    InsWriteReduce* insWriteReduce;
    InsWriteWithFin* insWriteWithFin;
    InsWriteReduceWithFin* insWriteReduceWithFin;
    InsReadExtend* insReadExtend;
    InsReadReduceExtend* insReadReduceExtend;
    InsWriteWithFinExtend* insWriteWithFinExtend;
    InsWriteReduceExtend* insWriteReduceExtend;
    InsWriteExtend* insWriteExtend;
    InsWriteReduceWithFinExtend* insWriteReduceWithFinExtend;

    InsBatchOneSidedRead* insBatchOneSidedRead;
    InsBatchOneSidedWrite* insBatchOneSidedWrite;

    DataType dataType;
    ReduceOp reduceOp;
    QId postQid;
    QId waitQid;
    u32 topicId;
    u32 waitValue;
};

TEST_F(InstructionTest, test_print_all_ins)
{
    cout << insPostTo->Describe() << endl;
    cout << insWaitFrom->Describe() << endl;

    cout << insWaitGroup->Describe() << endl;

    cout << insLocalCopy->Describe() << endl;
    cout << insLocalReduce->Describe() << endl;
    cout << insAicpuReduce->Describe() << endl;
    cout << insStreamSync->Describe() << endl;

    cout << insPostReady->Describe() << endl;
    cout << insWaitReady->Describe() << endl;

    cout << insPostFin->Describe() << endl;
    cout << insWaitFin->Describe() << endl;

    cout << insPostFinAck->Describe() << endl;
    cout << insWaitFinAck->Describe() << endl;

    cout << insRead->Describe() << endl;
    cout << insReadReduce->Describe() << endl;

    cout << insBatchRead->Describe() << endl;
    cout << insBatchWrite->Describe() << endl;

    cout << insWrite->Describe() << endl;
    cout << insWriteReduce->Describe() << endl;
    cout << insWriteWithFin->Describe() << endl;
    cout << insWriteReduceWithFin->Describe() << endl;
}

TEST_F(InstructionTest, test_ins_local_copy)
{
    cout << insLocalCopy->Describe() << endl;
    DataSlice testDstSlice = insLocalCopy->GetDstSlice();
    DataSlice testSrcSlice = insLocalCopy->GetSrcSlice();

    EXPECT_EQ(true, testDstSlice == *dstSlice);
    EXPECT_EQ(true, testSrcSlice == *srcSlice);
}

TEST_F(InstructionTest, test_ins_local_reduce)
{
    cout << insLocalReduce->Describe() << endl;
    DataSlice testDstSlice = insLocalReduce->GetDstSlice();
    DataSlice testSrcSlice = insLocalReduce->GetSrcSlice();

    EXPECT_EQ(true, testDstSlice == *dstSlice);
    EXPECT_EQ(true, testSrcSlice == *srcSlice);
    EXPECT_EQ(true, dataType == insLocalReduce->GetDataType());
    EXPECT_EQ(true, reduceOp == insLocalReduce->GetReduceOp());
}

TEST_F(InstructionTest, test_ins_aicpu_reduce)
{
    cout << insAicpuReduce->Describe() << endl;
    DataSlice testDstSlice = insAicpuReduce->GetDstSlice();
    DataSlice testSrcSlice = insAicpuReduce->GetSrcSlice();

    EXPECT_EQ(true, testDstSlice == *dstSlice);
    EXPECT_EQ(true, testSrcSlice == *srcSlice);
    EXPECT_EQ(true, dataType == insAicpuReduce->GetDataType());
    EXPECT_EQ(true, reduceOp == insAicpuReduce->GetReduceOp());
}

TEST_F(InstructionTest, test_ins_post_to)
{
    cout << insPostTo->Describe() << endl;
    EXPECT_EQ(NotifyType::NORMAL, insPostTo->GetNotifyType());
    EXPECT_EQ(postQid, insPostTo->GetPostQid());
    EXPECT_EQ(waitQid, insPostTo->GetWaitQid());
    EXPECT_EQ(topicId, insPostTo->GetTopicId());

    EXPECT_THROW(insPostTo->SetPostQid(insPostTo->GetWaitQid()), InvalidParamsException);
}

TEST_F(InstructionTest, test_ins_wait_from)
{
    cout << insWaitFrom->Describe() << endl;
    EXPECT_EQ(postQid, insWaitFrom->GetPostQid());
    EXPECT_EQ(waitQid, insWaitFrom->GetWaitQid());
    EXPECT_EQ(topicId, insWaitFrom->GetTopicId());

    EXPECT_THROW(insWaitFrom->SetWaitQid(insWaitFrom->GetPostQid()), InvalidParamsException);
}

TEST_F(InstructionTest, test_ins_wait_group)
{
    insWaitGroup->Append(postQid);
    insWaitGroup->SetWaitQid(waitQid);
    cout << insWaitGroup->Describe() << endl;

    EXPECT_EQ(postQid, *(insWaitGroup->Iter()));
    EXPECT_EQ(waitQid, insWaitGroup->GetWaitQid());
    EXPECT_EQ(topicId, insWaitGroup->GetTopicId());

    EXPECT_THROW(insWaitGroup->SetWaitQid(postQid), InvalidParamsException);
}

TEST_F(InstructionTest, test_ins_post_ready)
{
    cout << insPostReady->Describe() << endl;
    EXPECT_EQ(remoteRank, insPostReady->GetRemoteRank());
    EXPECT_EQ(true, *(insPostReady->GetLink()) == *linkData);
}

TEST_F(InstructionTest, test_ins_wait_ready)
{
    cout << insWaitReady->Describe() << endl;
    EXPECT_EQ(remoteRank, insWaitReady->GetRemoteRank());
    EXPECT_EQ(true, *(insWaitReady->GetLink()) == *linkData);
}

TEST_F(InstructionTest, test_ins_post_fin)
{
    cout << insPostFin->Describe() << endl;
    EXPECT_EQ(remoteRank, insPostFin->GetRemoteRank());
    EXPECT_EQ(true, *(insPostFin->GetLink()) == *linkData);
}

TEST_F(InstructionTest, test_ins_wait_fin)
{
    cout << insWaitFin->Describe() << endl;
    EXPECT_EQ(remoteRank, insWaitFin->GetRemoteRank());
    EXPECT_EQ(true, *(insWaitFin->GetLink()) == *linkData);
}

TEST_F(InstructionTest, test_ins_wait_group_fin)
{
    LinkData link(BasePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB), 0, 1, 0, 1);
    insWaitGroupFin->Append(link);
    cout << insWaitGroupFin->Describe() << endl;

    EXPECT_EQ(link, *(insWaitGroupFin->Iter()));
    EXPECT_EQ(topicId, insWaitGroupFin->GetTopicId());
    EXPECT_EQ(waitValue, insWaitGroupFin->GetValue());
}

TEST_F(InstructionTest, test_ins_post_fin_ack)
{
    cout << insPostFinAck->Describe() << endl;
    EXPECT_EQ(remoteRank, insPostFinAck->GetRemoteRank());
    EXPECT_EQ(true, *(insPostFinAck->GetLink()) == *linkData);
}

TEST_F(InstructionTest, test_ins_wait_fin_ack)
{
    cout << insWaitFinAck->Describe() << endl;
    EXPECT_EQ(remoteRank, insWaitFinAck->GetRemoteRank());
    EXPECT_EQ(true, *(insWaitFinAck->GetLink()) == *linkData);
}

TEST_F(InstructionTest, test_ins_read)
{
    cout << insRead->Describe() << endl;

    EXPECT_EQ(remoteRank, insRead->GetRemoteRank());
    EXPECT_EQ(true, *(insRead->GetLink()) == *linkData);

    EXPECT_EQ(true, insRead->GetLocalSlice() == *localSlice);
    EXPECT_EQ(true, insRead->GetRemoteSlice() == *remoteSlice);
}

TEST_F(InstructionTest, test_ins_read_reduce)
{
    cout << insReadReduce->Describe() << endl;

    EXPECT_EQ(true, dataType == insReadReduce->GetDataType());
    EXPECT_EQ(true, reduceOp == insReadReduce->GetReduceOp());
    EXPECT_EQ(remoteRank, insReadReduce->GetRemoteRank());
    EXPECT_EQ(true, *(insReadReduce->GetLink()) == *linkData);

    EXPECT_EQ(true, insReadReduce->GetLocalSlice() == *localSlice);
    EXPECT_EQ(true, insReadReduce->GetRemoteSlice() == *remoteSlice);
}

TEST_F(InstructionTest, test_ins_batch_read)
{
    EXPECT_EQ(remoteRank, insBatchRead->GetRemoteRank());
    EXPECT_EQ(true, *(insBatchRead->GetLink()) == *linkData);

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    DataSlice locSlice(BufferType::SCRATCH, 0, 100);
    DataSlice rmtSlice(BufferType::SCRATCH, 10, 100);
    unique_ptr<Instruction> readIns = make_unique<InsRead>(100, linkData, locSlice, rmtSlice);
    insBatchRead->PushReadIns(move(readIns));

    DataType dataType(DataType::INT8);
    ReduceOp reduceOp(ReduceOp::SUM);
    unique_ptr<Instruction> readReduceIns
        = make_unique<InsReadReduce>(100, linkData, locSlice, rmtSlice, dataType, reduceOp);
    insBatchRead->PushReadIns(move(readReduceIns));
    EXPECT_EQ(readIns, nullptr);
    EXPECT_EQ(readReduceIns, nullptr);
    EXPECT_EQ(insBatchRead->readInsVec.size(), 2);
}

TEST_F(InstructionTest, test_ins_batch_read_PushReadIns_fail)
{
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    DataSlice locSlice(BufferType::SCRATCH, 0, 100);
    DataSlice rmtSlice(BufferType::SCRATCH, 10, 100);
    unique_ptr<Instruction> writeIns = make_unique<InsWrite>(100, linkData, locSlice, rmtSlice);

    EXPECT_THROW(insBatchRead->PushReadIns(move(writeIns)), NotSupportException);
}

TEST_F(InstructionTest, test_ins_batch_write)
{
    EXPECT_EQ(remoteRank, insBatchWrite->GetRemoteRank());
    EXPECT_EQ(true, *(insBatchWrite->GetLink()) == *linkData);

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    DataSlice locSlice(BufferType::SCRATCH, 0, 100);
    DataSlice rmtSlice(BufferType::SCRATCH, 10, 100);
    unique_ptr<Instruction> writeIns = make_unique<InsWrite>(100, linkData, locSlice, rmtSlice);
    insBatchWrite->PushWriteIns(move(writeIns));

    DataType dataType(DataType::INT8);
    ReduceOp reduceOp(ReduceOp::SUM);
    unique_ptr<Instruction> writeReduceIns
        = make_unique<InsWriteReduce>(100, linkData, locSlice, rmtSlice, dataType, reduceOp);
    insBatchWrite->PushWriteIns(move(writeReduceIns));
    EXPECT_EQ(writeIns, nullptr);
    EXPECT_EQ(writeReduceIns, nullptr);
    EXPECT_EQ(insBatchWrite->writeInsVec.size(), 2);
}

TEST_F(InstructionTest, test_ins_batch_write_PushWriteIns_fail)
{
    EXPECT_EQ(remoteRank, insBatchWrite->GetRemoteRank());
    EXPECT_EQ(true, *(insBatchWrite->GetLink()) == *linkData);

    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    DataSlice locSlice(BufferType::SCRATCH, 0, 100);
    DataSlice rmtSlice(BufferType::SCRATCH, 10, 100);
    unique_ptr<Instruction> readIns = make_unique<InsRead>(100, linkData, locSlice, rmtSlice);

    EXPECT_THROW(insBatchWrite->PushWriteIns(move(readIns)), NotSupportException);
}

TEST_F(InstructionTest, test_ins_write)
{
    cout << insWrite->Describe() << endl;

    EXPECT_EQ(remoteRank, insWrite->GetRemoteRank());
    EXPECT_EQ(true, *(insWrite->GetLink()) == *linkData);

    EXPECT_EQ(true, insWrite->GetLocalSlice() == *localSlice);
    EXPECT_EQ(true, insWrite->GetRemoteSlice() == *remoteSlice);
}

TEST_F(InstructionTest, test_ins_write_reduce)
{
    cout << insWriteReduce->Describe() << endl;

    EXPECT_EQ(true, dataType == insWriteReduce->GetDataType());
    EXPECT_EQ(true, reduceOp == insWriteReduce->GetReduceOp());
    EXPECT_EQ(remoteRank, insWriteReduce->GetRemoteRank());
    EXPECT_EQ(true, *(insWriteReduce->GetLink()) == *linkData);

    EXPECT_EQ(true, insWriteReduce->GetLocalSlice() == *localSlice);
    EXPECT_EQ(true, insWriteReduce->GetRemoteSlice() == *remoteSlice);
}

TEST_F(InstructionTest, test_ins_write_with_fin)
{
    cout << insWriteWithFin->Describe() << endl;

    EXPECT_EQ(remoteRank, insWriteWithFin->GetRemoteRank());
    EXPECT_EQ(true, *(insWriteWithFin->GetLink()) == *linkData);

    EXPECT_EQ(true, insWriteWithFin->GetLocalSlice() == *localSlice);
    EXPECT_EQ(true, insWriteWithFin->GetRemoteSlice() == *remoteSlice);
    EXPECT_EQ(0, insWriteWithFin->GetTopicId());
    EXPECT_EQ(0, insWriteWithFin->GetBitValue());
}

TEST_F(InstructionTest, test_ins_write_reduce_with_fin)
{
    cout << insWriteReduceWithFin->Describe() << endl;

    EXPECT_EQ(true, dataType == insWriteReduceWithFin->GetDataType());
    EXPECT_EQ(true, reduceOp == insWriteReduceWithFin->GetReduceOp());
    EXPECT_EQ(remoteRank, insWriteReduceWithFin->GetRemoteRank());
    EXPECT_EQ(true, *(insWriteReduceWithFin->GetLink()) == *linkData);

    EXPECT_EQ(true, insWriteReduceWithFin->GetLocalSlice() == *localSlice);
    EXPECT_EQ(true, insWriteReduceWithFin->GetRemoteSlice() == *remoteSlice);
    EXPECT_EQ(0, insWriteReduceWithFin->GetTopicId());
    EXPECT_EQ(0, insWriteReduceWithFin->GetBitValue());
}

TEST_F(InstructionTest, test_ins_ReadExtend)
{
    cout << insReadExtend->Describe() << endl;
    EXPECT_EQ(remoteRank, insReadExtend->GetRemoteRank());
    EXPECT_EQ(true, *(insReadExtend->GetLink()) == *linkData);
    insReadExtend->GetLocalBuffer();
    insReadExtend->GetRemoteBuffer();
}

TEST_F(InstructionTest, test_ins_ReadReduceExtend)
{
    cout << insReadReduceExtend->Describe() << endl;
    EXPECT_EQ(remoteRank, insReadReduceExtend->GetRemoteRank());
    EXPECT_EQ(true, *(insReadReduceExtend->GetLink()) == *linkData);
    insReadReduceExtend->GetLocalBuffer();
    insReadReduceExtend->GetRemoteBuffer();
    EXPECT_EQ(dataType, insReadReduceExtend->GetDataType());
    EXPECT_EQ(reduceOp, insReadReduceExtend->GetReduceOp());
}

TEST_F(InstructionTest, test_ins_WriteExtend)
{
    EXPECT_EQ(remoteRank, insWriteExtend->GetRemoteRank());
    EXPECT_EQ(remoteRank, insWriteWithFinExtend->GetRemoteRank());
}

TEST_F(InstructionTest, test_ins_WriteReduceExtend)
{
    cout << insWriteReduceExtend->Describe() << endl;
    EXPECT_EQ(remoteRank, insWriteReduceExtend->GetRemoteRank());
    EXPECT_EQ(true, *(insWriteReduceExtend->GetLink()) == *linkData);
    insWriteReduceExtend->GetLocalBuffer();
    insWriteReduceExtend->GetRemoteBuffer();
    EXPECT_EQ(dataType, insWriteReduceExtend->GetDataType());
    EXPECT_EQ(reduceOp, insWriteReduceExtend->GetReduceOp());
}

TEST_F(InstructionTest, test_ins_WriteReduceWithFinExtend)
{
    cout << insWriteReduceWithFinExtend->Describe() << endl;
    EXPECT_EQ(remoteRank, insWriteReduceWithFinExtend->GetRemoteRank());
    EXPECT_EQ(true, *(insWriteReduceWithFinExtend->GetLink()) == *linkData);
    insWriteReduceWithFinExtend->GetLocalBuffer();
    insWriteReduceWithFinExtend->GetRemoteBuffer();
    EXPECT_EQ(dataType, insWriteReduceWithFinExtend->GetDataType());
    EXPECT_EQ(reduceOp, insWriteReduceWithFinExtend->GetReduceOp());
}

TEST_F(InstructionTest, test_ins_BatchOneSidedRead)
{
    cout << insBatchOneSidedRead->Describe() << endl;
    EXPECT_EQ(remoteRank, insBatchOneSidedRead->GetRemoteRank());
    EXPECT_EQ(true, *(insBatchOneSidedRead->GetLink()) == *linkData);
    EXPECT_EQ(1, insBatchOneSidedRead->GetLocalSlice().size());
    EXPECT_EQ(1, insBatchOneSidedRead->GetRemoteSlice().size());
}

TEST_F(InstructionTest, test_ins_BatchOneSidedWrite)
{
    cout << insBatchOneSidedWrite->Describe() << endl;
    EXPECT_EQ(remoteRank, insBatchOneSidedWrite->GetRemoteRank());
    EXPECT_EQ(true, *(insBatchOneSidedWrite->GetLink()) == *linkData);
    EXPECT_EQ(1, insBatchOneSidedWrite->GetLocalSlice().size());
    EXPECT_EQ(1, insBatchOneSidedWrite->GetRemoteSlice().size());
}

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
#include "primitive.h"
#include "prim_queue.h"
#include "null_ptr_exception.h"

using namespace Hccl;

class PrimitiveTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "PrimitiveTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "PrimitiveTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        master = make_shared<PrimQueue>();
        slave = master->Fork();
        RankId localRankID = 0;
        RankId remoteRank = 1;
        BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
        LinkData link(portType, localRankID, remoteRank, 0, 1);
        DataSlice localSlice(BufferType::SCRATCH, 0, 100);
        DataSlice remoteSlice(BufferType::SCRATCH, 0, 100);

        primSend = new PrimSend(remoteRank, link, localSlice, remoteSlice);
        primRecv = new PrimRecv(remoteRank, link, localSlice, remoteSlice);
        primGroup = new PrimGroup();
        std::cout << "A Test case in PrimitiveTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        delete primSend;
        delete primRecv;
        delete primGroup;
        std::cout << "A Test case in PrimitiveTest TearDown" << std::endl;
    }

    shared_ptr<PrimQueue> master;
    shared_ptr<PrimQueue> slave;

    PrimSend* primSend;
    PrimRecv* primRecv;
    PrimGroup* primGroup;
};

TEST_F(PrimitiveTest, test_prim_post_to)
{
    PrimPostTo prim(slave);

    EXPECT_EQ(0, prim.GetTopicId());

    u32 topicId = 1;
    PrimPostTo prim2(master, NotifyType::NORMAL, topicId);

    EXPECT_EQ(topicId, prim2.GetTopicId());
}

TEST_F(PrimitiveTest, test_prim_wait_group)
{
    PrimWaitGroup prim;

    EXPECT_EQ(0, prim.GetTopicId());

    u32 topicId = 1;
    PrimWaitGroup prim2(topicId);

    EXPECT_EQ(topicId, prim2.GetTopicId());

    prim2.Append(slave);
    prim2.SetParent(master);

    auto iter = prim2.Iter();
    QId qId1 = *iter;
    QId parentQId = prim2.GetParentQid();

    EXPECT_EQ(1, qId1);
    EXPECT_EQ(0, parentQId);

    auto des = prim.Describe();

    EXPECT_THROW(prim2.SetParent(slave), InvalidParamsException);
}

TEST_F(PrimitiveTest, test_prim_wait_from)
{
    PrimWaitFrom prim(slave);

    EXPECT_EQ(0, prim.GetTopicId());

    u32 topicId = 1;
    PrimWaitFrom prim2(master, topicId);

    EXPECT_EQ(topicId, prim2.GetTopicId());
}

TEST_F(PrimitiveTest, test_prim_send_append_ok)
{
    DataSlice localSlice2(BufferType::SCRATCH, 0, 100);
    DataSlice remoteSlice2(BufferType::SCRATCH, 0, 100);
    primSend->Append(localSlice2, remoteSlice2);

    EXPECT_EQ(2, primSend->Size());
    cout << primSend->Describe() << endl;
}

TEST_F(PrimitiveTest, test_prim_recv_append_ok)
{
    DataSlice localSlice2(BufferType::SCRATCH, 0, 100);
    DataSlice remoteSlice2(BufferType::SCRATCH, 0, 100);
    primRecv->Append(localSlice2, remoteSlice2);

    EXPECT_EQ(2, primRecv->Size());
    cout << primRecv->Describe() << endl;
}

TEST_F(PrimitiveTest, test_prim_group_append_error_prim_failed)
{
    auto tmpGroup = make_unique<PrimGroup>();
    try {
        primGroup->Append(std::move(tmpGroup));
    } catch (InvalidParamsException& e) {
        EXPECT_EQ(HcclResult::HCCL_E_PARA, e.GetErrorCode());
    }
    EXPECT_EQ(0, primGroup->GetSize());
}

TEST_F(PrimitiveTest, test_prim_group_append_one_send_ok)
{
    LinkData link = LinkData(BasePortType(PortDeploymentType::P2P, ConnectProtoType::PCIE), 0, 1, 0, 0);
    DataSlice localSlice(BufferType::SCRATCH, 0, 100);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, 100);

    auto send = make_unique<PrimSend>(0, link, localSlice, remoteSlice);
    primGroup->Append(std::move(send));
    EXPECT_EQ(1, primGroup->GetSize());
}

TEST_F(PrimitiveTest, test_prim_group_append_one_recv_ok)
{
    LinkData link = LinkData(BasePortType(PortDeploymentType::P2P, ConnectProtoType::PCIE), 0, 1, 0, 0);
    DataSlice localSlice(BufferType::SCRATCH, 0, 100);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, 100);

    auto recv = make_unique<PrimRecv>(0, link, localSlice, remoteSlice);
    primGroup->Append(std::move(recv));
    EXPECT_EQ(1, primGroup->GetSize());
}

TEST_F(PrimitiveTest, test_prim_group_append_one_send_reduce_ok)
{
    LinkData link = LinkData(BasePortType(PortDeploymentType::P2P, ConnectProtoType::PCIE), 0, 1, 0, 0);
    DataSlice localSlice(BufferType::SCRATCH, 0, 100);
    DataSlice remoteSrcSlice(BufferType::SCRATCH, 0, 100);
    DataSlice remoteDstSlice(BufferType::SCRATCH, 100, 100);

    auto sendReduce = make_unique<PrimSendReduce>(
        0, link, localSlice, remoteSrcSlice, remoteDstSlice, DataType::INT8, ReduceOp::SUM);
    primGroup->Append(std::move(sendReduce));
    EXPECT_EQ(1, primGroup->GetSize());
}

TEST_F(PrimitiveTest, test_prim_group_append_one_recv_reduce_ok)
{
    LinkData link = LinkData(BasePortType(PortDeploymentType::P2P, ConnectProtoType::PCIE), 0, 1, 0, 0);
    DataSlice localSrcSlice(BufferType::SCRATCH, 0, 100);
    DataSlice localDstSlice(BufferType::SCRATCH, 0, 100);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, 100);

    auto recvReduce = make_unique<PrimRecvReduce>(
        0, link, remoteSlice, localSrcSlice, localDstSlice, DataType::INT8, ReduceOp::SUM);
    primGroup->Append(std::move(recvReduce));
    EXPECT_EQ(1, primGroup->GetSize());
}

TEST_F(PrimitiveTest, prim_post_to_check_valid)
{
    shared_ptr<PrimQueue> queue;
    try {
        PrimPostTo prim1(queue);
    } catch (NullPtrException& e) {
        EXPECT_EQ(HcclResult::HCCL_E_PTR, e.GetErrorCode());
    }

    PrimPostTo prim2(slave);
    try {
        prim2.SetParent(queue);
    } catch (NullPtrException& e) {
        EXPECT_EQ(HcclResult::HCCL_E_PTR, e.GetErrorCode());
    }

    try {
        prim2.SetParent(slave);
    } catch (InvalidParamsException& e) {
        EXPECT_EQ(HcclResult::HCCL_E_PARA, e.GetErrorCode());
    }
}

TEST_F(PrimitiveTest, prim_wait_from_check_valid)
{
    shared_ptr<PrimQueue> queue;
    try {
        PrimWaitFrom prim1(queue);
    } catch (NullPtrException& e) {
        EXPECT_EQ(HcclResult::HCCL_E_PTR, e.GetErrorCode());
    }

    PrimWaitFrom prim2(slave);
    try {
        prim2.SetParent(queue);
    } catch (NullPtrException& e) {
        EXPECT_EQ(HcclResult::HCCL_E_PTR, e.GetErrorCode());
    }

    try {
        prim2.SetParent(slave);
    } catch (InvalidParamsException& e) {
        EXPECT_EQ(HcclResult::HCCL_E_PARA, e.GetErrorCode());
    }
}

TEST_F(PrimitiveTest, prim_local_copy_check_valid)
{
    DataSlice srcSlice1(BufferType::INPUT, 0, 100);
    DataSlice dstSlice1(BufferType::INPUT, 100, 200);
    try {
        PrimLocalCopy* primLocalCopy = new PrimLocalCopy(srcSlice1, dstSlice1);
    } catch (InvalidParamsException& e) {
        EXPECT_EQ(HcclResult::HCCL_E_PARA, e.GetErrorCode());
    }

    DataSlice srcSlice2(BufferType::INPUT, 0, 100);
    DataSlice dstSlice2(BufferType::INPUT, 1, 100);
    try {
        PrimLocalCopy* primLocalCopy = new PrimLocalCopy(srcSlice2, dstSlice2);
    } catch (InvalidParamsException& e) {
        EXPECT_EQ(HcclResult::HCCL_E_PARA, e.GetErrorCode());
    }
}

TEST_F(PrimitiveTest, prim_recv_check_valid)
{
    RankId myRank_ = 0;
    RankId remoteRankId_ = 1;
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData link(portType, myRank_, remoteRankId_, 0, 0);
    DataSlice localSlice1(BufferType::INPUT, 0, 100);
    DataSlice remoteSlice1(BufferType::SCRATCH, 0, 200);

    try {
        PrimRecv* primRecv1 = new PrimRecv(remoteRankId_, link, localSlice1, remoteSlice1);
    } catch (InvalidParamsException& e) {
        EXPECT_EQ(HcclResult::HCCL_E_PARA, e.GetErrorCode());
    }

    DataSlice localSlice2(BufferType::INPUT, 0, 100);
    DataSlice remoteSlice2(BufferType::SCRATCH, 0, 100);
    PrimRecv* primRecv2 = new PrimRecv(remoteRankId_, link, localSlice2, remoteSlice2);
    try {
        primRecv2->Append(localSlice1, remoteSlice1);
    } catch (InvalidParamsException& e) {
        EXPECT_EQ(HcclResult::HCCL_E_PARA, e.GetErrorCode());
    }
    delete primRecv2;
}

TEST_F(PrimitiveTest, test_prim_group_check_valid)
{
    LinkData link1 = LinkData(BasePortType(PortDeploymentType::P2P, ConnectProtoType::PCIE), 0, 1, 0, 0);

    DataSlice localSlice1(BufferType::SCRATCH, 0, 100);
    DataSlice localSlice2(BufferType::SCRATCH, 100, 100);
    DataSlice localSlice3(BufferType::SCRATCH, 200, 100);
    DataSlice localSlice4(BufferType::SCRATCH, 300, 100);

    DataSlice remoteSlice1(BufferType::SCRATCH, 0, 100);
    DataSlice remoteSlice2(BufferType::SCRATCH, 100, 100);
    DataSlice remoteSlice3(BufferType::SCRATCH, 200, 100);
    DataSlice remoteSlice4(BufferType::SCRATCH, 300, 100);

    auto send1 = make_unique<PrimSend>(0, link1, localSlice1, remoteSlice1);
    auto send2 = make_unique<PrimSend>(0, link1, localSlice2, remoteSlice2);
    auto send3 = make_unique<PrimSend>(0, link1, localSlice3, remoteSlice3);

    auto recv1 = make_unique<PrimRecv>(0, link1, localSlice1, remoteSlice1);
    auto recv2 = make_unique<PrimRecv>(0, link1, localSlice2, remoteSlice2);
    auto recv3 = make_unique<PrimRecv>(0, link1, localSlice3, remoteSlice3);
    auto recv4 = make_unique<PrimRecv>(0, link1, localSlice4, remoteSlice4);

    auto sendReduce1
        = make_unique<PrimSendReduce>(0, link1, localSlice1, remoteSlice1, remoteSlice2, DataType::INT8, ReduceOp::SUM);
    auto sendReduce2
        = make_unique<PrimSendReduce>(0, link1, localSlice2, remoteSlice3, remoteSlice4, DataType::INT8, ReduceOp::SUM);

    auto recvReduce1
        = make_unique<PrimRecvReduce>(0, link1, remoteSlice1, localSlice1, localSlice2, DataType::INT8, ReduceOp::SUM);
    auto recvReduce2
        = make_unique<PrimRecvReduce>(0, link1, remoteSlice2, localSlice3, localSlice4, DataType::INT8, ReduceOp::SUM);

    PrimGroup* primGroup1 = new PrimGroup();
    primGroup1->Append(std::move(send1));
    primGroup1->Append(std::move(send2));
    try {
        primGroup1->CheckValid();
    } catch (InvalidParamsException& e) {
        EXPECT_EQ(HcclResult::HCCL_E_PARA, e.GetErrorCode());
    }
    delete primGroup1;

    PrimGroup* primGroup2 = new PrimGroup();
    primGroup2->Append(std::move(recv1));
    primGroup2->Append(std::move(recv2));
    try {
        primGroup2->CheckValid();
    } catch (InvalidParamsException& e) {
        EXPECT_EQ(HcclResult::HCCL_E_PARA, e.GetErrorCode());
    }
    delete primGroup2;

    PrimGroup* primGroup3 = new PrimGroup();
    primGroup3->Append(std::move(sendReduce1));
    primGroup3->Append(std::move(sendReduce2));
    try {
        primGroup3->CheckValid();
    } catch (InvalidParamsException& e) {
        EXPECT_EQ(HcclResult::HCCL_E_PARA, e.GetErrorCode());
    }
    delete primGroup3;

    PrimGroup* primGroup4 = new PrimGroup();
    primGroup4->Append(std::move(recvReduce1));
    primGroup4->Append(std::move(recvReduce2));
    try {
        primGroup4->CheckValid();
    } catch (InvalidParamsException& e) {
        EXPECT_EQ(HcclResult::HCCL_E_PARA, e.GetErrorCode());
    }
    delete primGroup4;

    PrimGroup* primGroup5 = new PrimGroup();
    primGroup5->Append(std::move(send3));
    primGroup5->Append(std::move(recv3));
    primGroup5->CheckValid();
    primGroup5->Append(std::move(recv4));
    try {
        primGroup5->CheckValid();
    } catch (InvalidParamsException& e) {
        EXPECT_EQ(HcclResult::HCCL_E_PARA, e.GetErrorCode());
    }
    delete primGroup5;
}

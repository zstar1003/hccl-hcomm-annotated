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
#include "prim_rules.h"
#include "data_slice.h"
#include "types.h"
#include "instruction.h"
#include "ins_queue.h"
#include "primitive.h"
#include "prim_queue.h"
#include "prim_translator.h"
#include "not_support_exception.h"
#include "dev_capability.h"
#include "orion_adapter_rts.cc"
using namespace Hccl;

class PrimRulesTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "PrimRulesTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "PrimRulesTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        dataSliceSize = 128;
        MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_910A2));
        std::cout << "A Test case in PrimRulesTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();

        std::cout << "A Test case in PrimRulesTest TearDown" << std::endl;
    }

    void
    Check_PostReady_WaitFin_PostFinAck(vector<unique_ptr<Instruction>>& result, RankId remote, const LinkData& link)
    {
        EXPECT_EQ(3, result.size());
        EXPECT_EQ(InstructionType::POST_READY, result[0]->GetType());
        EXPECT_EQ(InstructionType::WAIT_FIN, result[1]->GetType());
        EXPECT_EQ(InstructionType::POST_FIN_ACK, result[2]->GetType());
    }

    void Check_PostReady_WaitFin(vector<unique_ptr<Instruction>>& result, RankId remote, const LinkData& link)
    {
        EXPECT_EQ(2, result.size());
        EXPECT_EQ(InstructionType::POST_READY, result[0]->GetType());
        EXPECT_EQ(InstructionType::WAIT_FIN, result[1]->GetType());
    }

    // case: sendGet, recvPut, sendReduceGetWithInlineReduce, recvReducePutWithInlineReduce,
    // sendReduceGetWithoutInlineReduce
    void
    WithoutData_Check_SendGet_Or_RecvPut(vector<unique_ptr<Instruction>>& result, RankId remote, const LinkData& link)
    {
        Check_PostReady_WaitFin_PostFinAck(result, remote, link);
    }

    void Check_WaitReady_DMAData_PostFin_WaitFinAck(
        vector<unique_ptr<Instruction>>& result, RankId remote, const LinkData& link, InstructionType insType,
        u32 insTypeNum)
    {
        EXPECT_EQ(3 + insTypeNum, result.size());
        EXPECT_EQ(InstructionType::WAIT_READY, result[0]->GetType());
        for (u32 i = 0; i < insTypeNum; i++) {
            EXPECT_EQ(insType, result[i + 1]->GetType());
        }

        EXPECT_EQ(InstructionType::POST_FIN, result[insTypeNum + 1]->GetType());
        EXPECT_EQ(InstructionType::WAIT_FIN_ACK, result[insTypeNum + 2]->GetType());
    }

    void Check_WaitReady_DMAData_PostFin(
        vector<unique_ptr<Instruction>>& result, RankId remote, const LinkData& link, InstructionType insType,
        u32 insTypeNum)
    {
        EXPECT_EQ(2 + insTypeNum, result.size());
        EXPECT_EQ(InstructionType::WAIT_READY, result[0]->GetType());
        for (u32 i = 0; i < insTypeNum; i++) {
            EXPECT_EQ(insType, result[i + 1]->GetType());
        }

        EXPECT_EQ(InstructionType::POST_FIN, result[insTypeNum + 1]->GetType());
    }

    void Check_WaitReady_DMAData_WaitFinAck(
        vector<unique_ptr<Instruction>>& result, RankId remote, const LinkData& link, InstructionType insType,
        u32 insTypeNum)
    {
        EXPECT_EQ(2 + insTypeNum, result.size());
        EXPECT_EQ(InstructionType::WAIT_READY, result[0]->GetType());
        for (u32 i = 0; i < insTypeNum - 1; i++) {
            EXPECT_EQ(insType, result[i + 1]->GetType());
        }

        EXPECT_EQ(InstructionType::WAIT_FIN_ACK, result[insTypeNum + 1]->GetType());
    }

    void Check_WaitReady_DMAData(
        vector<unique_ptr<Instruction>>& result, RankId remote, const LinkData& link, InstructionType insType,
        u32 insTypeNum)
    {
        EXPECT_EQ(1 + insTypeNum, result.size());
        EXPECT_EQ(InstructionType::WAIT_READY, result[0]->GetType());
        for (u32 i = 0; i < insTypeNum - 1; i++) {
            EXPECT_EQ(insType, result[i + 1]->GetType());
        }
    }

    void Check_SendPut(vector<unique_ptr<Instruction>>& result, PrimSend& primSend)
    {
        Check_WaitReady_DMAData_WaitFinAck(
            result, primSend.GetRemoteRank(), primSend.GetLink(), InstructionType::WRITE, primSend.Size());
    }

    void Check_RecvGet(vector<unique_ptr<Instruction>>& result, PrimRecv& primRecv)
    {
        Check_WaitReady_DMAData_PostFin_WaitFinAck(
            result, primRecv.GetRemoteRank(), primRecv.GetLink(), InstructionType::READ, primRecv.Size());
    }

    void Check_WithInlineReduce_SendReducePut(vector<unique_ptr<Instruction>>& result, PrimSendReduce& sendReduce)
    {
        Check_WaitReady_DMAData_PostFin_WaitFinAck(
            result, sendReduce.GetRemoteRank(), sendReduce.GetLink(), InstructionType::WRITE_REDUCE, sendReduce.Size());
    }
    void Check_WithoutInlineReduce_SendReducePut(vector<unique_ptr<Instruction>>& result, PrimSendReduce& sendReduce)
    {
        Check_WaitReady_DMAData_PostFin_WaitFinAck(
            result, sendReduce.GetRemoteRank(), sendReduce.GetLink(), InstructionType::WRITE, sendReduce.Size());
    }
    void Check_WithInlineReduce_RecvReducGet(vector<unique_ptr<Instruction>>& result, PrimRecvReduce& recvReduce)
    {
        Check_WaitReady_DMAData_PostFin_WaitFinAck(
            result, recvReduce.GetRemoteRank(), recvReduce.GetLink(), InstructionType::READ_REDUCE, recvReduce.Size());
    }

    void Check_WithoutInlineReduce_RecvReduceGet(vector<unique_ptr<Instruction>>& result, PrimRecvReduce& recvReduce)
    {
        u32 num = recvReduce.Size();
        u32 total = 2 + num * 2;
        EXPECT_EQ(total, result.size());
        EXPECT_EQ(InstructionType::WAIT_READY, result[0]->GetType());

        for (u32 idx = 0; idx < num; idx++) {
            EXPECT_EQ(InstructionType::READ, result[1 + idx]->GetType());
        }

        EXPECT_EQ(InstructionType::POST_FIN, result[1 + num]->GetType());

        for (u32 idx = 0; idx < num; idx++) {
            EXPECT_EQ(InstructionType::LOCAL_REDUCE, result[2 + num + idx]->GetType());
        }
    }

    void Check_WithoutInlineReduce_RecvReducePut(vector<unique_ptr<Instruction>>& result, PrimRecvReduce& recvReduce)
    {
        u32 num = recvReduce.Size();
        u32 total = 3 + num;
        EXPECT_EQ(total, result.size());
        EXPECT_EQ(InstructionType::POST_READY, result[0]->GetType());
        EXPECT_EQ(InstructionType::WAIT_FIN, result[1]->GetType());
        EXPECT_EQ(InstructionType::POST_FIN_ACK, result[2]->GetType());
        for (u32 idx = 0; idx < num; idx++) {
            EXPECT_EQ(InstructionType::LOCAL_REDUCE, result[3 + idx]->GetType());
        }
    }

    void Check_WithoutInlineReduce_WithoutPostFinAck_RecvReducePut(
        vector<unique_ptr<Instruction>>& result, PrimRecvReduce& recvReduce)
    {
        u32 num = recvReduce.Size();
        u32 total = 2 + num;
        EXPECT_EQ(total, result.size());
        EXPECT_EQ(InstructionType::POST_READY, result[0]->GetType());
        EXPECT_EQ(InstructionType::WAIT_FIN, result[1]->GetType());
        for (u32 idx = 0; idx < num; idx++) {
            EXPECT_EQ(InstructionType::LOCAL_REDUCE, result[2 + idx]->GetType());
        }
    }

    void CheckGroupSendRecv(
        vector<unique_ptr<Instruction>>& result, RankId remote, const LinkData& link, InstructionType insType,
        u32 insTypeNum)
    {
        EXPECT_EQ(4 + insTypeNum, result.size());

        EXPECT_EQ(InstructionType::POST_READY, result[0]->GetType());
        EXPECT_EQ(InstructionType::WAIT_READY, result[1]->GetType());

        for (u32 idx = 0; idx < insTypeNum; idx++) {
            EXPECT_EQ(insType, result[2 + idx]->GetType());
        }

        EXPECT_EQ(InstructionType::POST_FIN, result[2 + insTypeNum]->GetType());
        EXPECT_EQ(InstructionType::WAIT_FIN, result[3 + insTypeNum]->GetType());
        EXPECT_EQ(InstructionType::POST_FIN_ACK, result[4 + insTypeNum]->GetType());
        EXPECT_EQ(InstructionType::WAIT_FIN_ACK, result[5 + insTypeNum]->GetType());
    }

    void CheckGroupSendRecvRdma(
        vector<unique_ptr<Instruction>>& result, RankId remote, const LinkData& link, InstructionType insType,
        u32 insTypeNum)
    {
        EXPECT_EQ(3 + insTypeNum, result.size());

        EXPECT_EQ(InstructionType::POST_READY, result[0]->GetType());
        EXPECT_EQ(InstructionType::WAIT_READY, result[1]->GetType());

        for (u32 idx = 0; idx < insTypeNum - 1; idx++) {
            EXPECT_EQ(insType, result[2 + idx]->GetType());
        }

        EXPECT_EQ(InstructionType::WAIT_FIN, result[2 + insTypeNum]->GetType());
    }
    u64 dataSliceSize{100};
};

TEST_F(PrimRulesTest, translate_postto_test)
{
    // Given slave post to master
    shared_ptr<PrimQueue> master = make_shared<PrimQueue>();
    shared_ptr<PrimQueue> slave = master->Fork();

    PrimPostTo primPostTo(master);
    primPostTo.SetParent(slave);

    // When
    vector<unique_ptr<Instruction>> result = Translate(primPostTo);

    // Then
    EXPECT_EQ(1, result.size());
    EXPECT_EQ(InstructionType::LOCAL_POST_TO, result[0]->GetType());

    const InsLocalPostTo& insLocalPostTo = static_cast<const InsLocalPostTo&>(*result[0].get());

    EXPECT_EQ(master->GetId(), insLocalPostTo.GetWaitQid());
}

TEST_F(PrimRulesTest, translate_waitfrom_test)
{
    // Given slave wait from master
    shared_ptr<PrimQueue> master = make_shared<PrimQueue>();
    shared_ptr<PrimQueue> slave = master->Fork();

    PrimWaitFrom primWaitFrom(master);
    primWaitFrom.SetParent(slave);

    // When
    vector<unique_ptr<Instruction>> result = Translate(primWaitFrom);

    // Then
    EXPECT_EQ(1, result.size());
    EXPECT_EQ(InstructionType::LOCAL_WAIT_FROM, result[0]->GetType());

    const InsLocalWaitFrom& insLocalWaitFrom = static_cast<const InsLocalWaitFrom&>(*result[0].get());
    EXPECT_EQ(master->GetId(), insLocalWaitFrom.GetPostQid());
}

TEST_F(PrimRulesTest, translate_waitgroup_test)
{
    shared_ptr<PrimQueue> master = make_shared<PrimQueue>();
    shared_ptr<PrimQueue> slave = master->Fork();

    PrimWaitGroup primWaitGroup;
    primWaitGroup.Append(master);
    primWaitGroup.SetParent(slave);

    // When
    vector<unique_ptr<Instruction>> result = Translate(primWaitGroup);

    // Then
    EXPECT_EQ(1, result.size());
    EXPECT_EQ(InstructionType::LOCAL_WAIT_GROUP, result[0]->GetType());

    const InsLocalWaitGroup& insLocalWaitGroup = static_cast<const InsLocalWaitGroup&>(*result[0].get());
    EXPECT_EQ(master->GetId(), *(insLocalWaitGroup.Iter()));
}

TEST_F(PrimRulesTest, translate_localcopy_test)
{
    // Given
    RankId localRankID = 0;
    RankId remoteRank = 1;
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData link(portType, localRankID, remoteRank, 0, 1);
    DataSlice localSlice(BufferType::INPUT, 0, dataSliceSize);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, dataSliceSize);
    const PrimLocalCopy primLocalCopy(localSlice, remoteSlice);

    // When

    vector<unique_ptr<Instruction>> result = Translate(primLocalCopy);

    // Then
    EXPECT_EQ(1, result.size());
    EXPECT_EQ(InstructionType::LOCAL_COPY, result[0]->GetType());
}

TEST_F(PrimRulesTest, translate_send_link_p2p_dma_default_test)
{
    // Given
    RankId localRankID = 0;
    RankId remoteRank = 1;
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData link(portType, localRankID, remoteRank, 0, 1);
    DataSlice localSlice(BufferType::INPUT, 0, dataSliceSize);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, dataSliceSize);

    PrimSend primSendDmaDefault(remoteRank, link, localSlice, remoteSlice);

    DataSlice localSlice1(BufferType::INPUT, dataSliceSize, dataSliceSize);
    DataSlice remoteSlice1(BufferType::SCRATCH, dataSliceSize, dataSliceSize);
    primSendDmaDefault.Append(localSlice1, remoteSlice1);

    // When
    vector<unique_ptr<Instruction>> result = Translate(primSendDmaDefault);
    // then
    Check_PostReady_WaitFin(result, primSendDmaDefault.GetRemoteRank(), primSendDmaDefault.GetLink());
}

TEST_F(PrimRulesTest, translate_send_link_is_invalid)
{
    // Given
    RankId localRankID = 0;
    RankId remoteRank = 1;
    BasePortType portType(PortDeploymentType::HOST_NET, ConnectProtoType::TCP);
    LinkData link(portType, localRankID, remoteRank, 0, 1);
    DataSlice localSlice(BufferType::INPUT, 0, dataSliceSize);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, dataSliceSize);

    const PrimSend primSendDmaDefault(remoteRank, link, localSlice, remoteSlice);

    EXPECT_THROW(Translate(primSendDmaDefault), NotSupportException);
}

TEST_F(PrimRulesTest, translate_send_link_p2p_dma_get_test)
{
    // Given
    RankId localRankID = 0;
    RankId remoteRank = 1;
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData link(portType, localRankID, remoteRank, 0, 1);
    DataSlice localSlice(BufferType::INPUT, 0, dataSliceSize);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, dataSliceSize);

    const PrimSend primSendDmaGet(remoteRank, link, localSlice, remoteSlice, DmaMode::GET);
    // When
    vector<unique_ptr<Instruction>> result = Translate(primSendDmaGet);
    // then
    Check_PostReady_WaitFin(result, primSendDmaGet.GetRemoteRank(), primSendDmaGet.GetLink());
}

TEST_F(PrimRulesTest, translate_send_link_p2p_dma_put_test)
{
    // Given
    RankId localRankID = 0;
    RankId remoteRank = 1;
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData link(portType, localRankID, remoteRank, 0, 1);
    DataSlice localSlice(BufferType::INPUT, 0, dataSliceSize);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, dataSliceSize);

    PrimSend primSendDmaPut(remoteRank, link, localSlice, remoteSlice, DmaMode::PUT);

    DataSlice localSlice1(BufferType::INPUT, dataSliceSize, dataSliceSize);
    DataSlice remoteSlice1(BufferType::SCRATCH, dataSliceSize, dataSliceSize);
    primSendDmaPut.Append(localSlice1, remoteSlice1);

    // When
    vector<unique_ptr<Instruction>> result = Translate(primSendDmaPut);
    // then
    Check_WaitReady_DMAData_PostFin(
        result, primSendDmaPut.GetRemoteRank(), primSendDmaPut.GetLink(), InstructionType::WRITE,
        primSendDmaPut.Size());
}

TEST_F(PrimRulesTest, translate_send_link_dev_net_ub_dma_default)
{
    // Given
    RankId localRankID = 0;
    RankId remoteRank = 1;
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData link(portType, localRankID, remoteRank, 0, 1);
    DataSlice localSlice(BufferType::INPUT, 0, dataSliceSize);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, dataSliceSize);

    PrimSend primSendDmaDefault(remoteRank, link, localSlice, remoteSlice);
    // When
    vector<unique_ptr<Instruction>> result = Translate(primSendDmaDefault);
    // then
    Check_WaitReady_DMAData_PostFin_WaitFinAck(
        result, primSendDmaDefault.GetRemoteRank(), primSendDmaDefault.GetLink(), InstructionType::WRITE,
        primSendDmaDefault.Size());
}

TEST_F(PrimRulesTest, translate_send_link_dev_net_ub_dma_put)
{
    // Given
    RankId localRankID = 0;
    RankId remoteRank = 1;
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData link(portType, localRankID, remoteRank, 0, 1);
    DataSlice localSlice(BufferType::INPUT, 0, dataSliceSize);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, dataSliceSize);

    PrimSend primSendDmaPut(remoteRank, link, localSlice, remoteSlice, DmaMode::PUT);
    // When
    vector<unique_ptr<Instruction>> result = Translate(primSendDmaPut);
    // then
    Check_WaitReady_DMAData_PostFin_WaitFinAck(
        result, primSendDmaPut.GetRemoteRank(), primSendDmaPut.GetLink(), InstructionType::WRITE,
        primSendDmaPut.Size());
}

TEST_F(PrimRulesTest, translate_recv_link_p2p_dma_default)
{
    RankId localRankID = 0;
    RankId remoteRank = 1;
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData link(portType, localRankID, remoteRank, 0, 1);
    DataSlice localSlice(BufferType::INPUT, 0, dataSliceSize);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, dataSliceSize);

    // Given
    PrimRecv primRecv(remoteRank, link, localSlice, remoteSlice);

    DataSlice localSlice1(BufferType::INPUT, dataSliceSize, dataSliceSize);
    DataSlice remoteSlice1(BufferType::SCRATCH, dataSliceSize, dataSliceSize);
    primRecv.Append(localSlice1, remoteSlice1);

    // When
    vector<unique_ptr<Instruction>> result = Translate(primRecv);
    // then
    Check_WaitReady_DMAData_PostFin(
        result, primRecv.GetRemoteRank(), primRecv.GetLink(), InstructionType::READ, primRecv.Size());
}

TEST_F(PrimRulesTest, translate_recv_link_p2p_dma_get)
{
    RankId localRankID = 0;
    RankId remoteRank = 1;
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData link(portType, localRankID, remoteRank, 0, 1);
    DataSlice localSlice(BufferType::INPUT, 0, dataSliceSize);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, dataSliceSize);

    // Given
    PrimRecv primRecvDmaGet(remoteRank, link, localSlice, remoteSlice, DmaMode::GET);
    // When
    vector<unique_ptr<Instruction>> result = Translate(primRecvDmaGet);

    // then
    Check_WaitReady_DMAData_PostFin(
        result, primRecvDmaGet.GetRemoteRank(), primRecvDmaGet.GetLink(), InstructionType::READ, primRecvDmaGet.Size());
}

TEST_F(PrimRulesTest, translate_recv_link_p2p_dma_put)
{
    RankId localRankID = 0;
    RankId remoteRank = 1;
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData link(portType, localRankID, remoteRank, 0, 1);
    DataSlice localSlice(BufferType::INPUT, 0, dataSliceSize);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, dataSliceSize);

    // Given
    PrimRecv primRecvDmaPut(remoteRank, link, localSlice, remoteSlice, DmaMode::PUT);

    DataSlice localSlice1(BufferType::INPUT, dataSliceSize, dataSliceSize);
    DataSlice remoteSlice1(BufferType::SCRATCH, dataSliceSize, dataSliceSize);
    primRecvDmaPut.Append(localSlice1, remoteSlice1);

    // When
    vector<unique_ptr<Instruction>> result = Translate(primRecvDmaPut);
    // then
    Check_PostReady_WaitFin(result, primRecvDmaPut.GetRemoteRank(), primRecvDmaPut.GetLink());
}

TEST_F(PrimRulesTest, translate_recv_link_dev_net_rdma_dma_default)
{
    RankId localRankID = 0;
    RankId remoteRank = 1;
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::RDMA);
    LinkData link(portType, localRankID, remoteRank, 0, 1);
    DataSlice localSlice(BufferType::INPUT, 0, dataSliceSize);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, dataSliceSize);

    // Given
    const PrimRecv primRecvDmaDefault(remoteRank, link, localSlice, remoteSlice);
    // When
    vector<unique_ptr<Instruction>> result = Translate(primRecvDmaDefault);
    // then
    WithoutData_Check_SendGet_Or_RecvPut(result, primRecvDmaDefault.GetRemoteRank(), primRecvDmaDefault.GetLink());
}

TEST_F(PrimRulesTest, translate_recv_link_dev_net_rdma_dma_put)
{
    RankId localRankID = 0;
    RankId remoteRank = 1;
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::RDMA);
    LinkData link(portType, localRankID, remoteRank, 0, 1);
    DataSlice localSlice(BufferType::INPUT, 0, dataSliceSize);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, dataSliceSize);

    // Given
    const PrimRecv primRecvDmaPut(remoteRank, link, localSlice, remoteSlice, DmaMode::PUT);
    // When
    vector<unique_ptr<Instruction>> result = Translate(primRecvDmaPut);
    // then
    WithoutData_Check_SendGet_Or_RecvPut(result, primRecvDmaPut.GetRemoteRank(), primRecvDmaPut.GetLink());
}

TEST_F(PrimRulesTest, translate_send_reduce_link_p2p_fp32_sum_support_inline_reduce_dma_default_and_dma_get)
{
    RankId localRankID = 0;
    RankId remoteRank = 1;
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData link(portType, localRankID, remoteRank, 0, 1);
    DataSlice localSlice(BufferType::INPUT, 0, dataSliceSize);
    DataSlice remoteSrcSlice(BufferType::SCRATCH, 0, dataSliceSize);
    DataSlice remoteDstSlice(BufferType::SCRATCH, 100, dataSliceSize);

    // 910A2 support FP32+ReduceOp::SUM inline reduce
    PrimSendReduce primSendReduce(
        remoteRank, link, localSlice, remoteSrcSlice, remoteDstSlice, DataType::FP32, ReduceOp::SUM);

    DataSlice localSlice1(BufferType::INPUT, 0, dataSliceSize);
    DataSlice remoteSrcSlice1(BufferType::SCRATCH, 0, dataSliceSize);
    DataSlice remoteDstSlice1(BufferType::SCRATCH, 100, dataSliceSize);
    primSendReduce.Append(localSlice1, remoteSrcSlice1, remoteDstSlice1);

    const PrimSendReduce primSendReduceDmaGet(
        remoteRank, link, localSlice, remoteSrcSlice, remoteDstSlice, DataType::FP32, ReduceOp::SUM, DmaMode::GET);
    // When
    vector<unique_ptr<Instruction>> result1 = Translate(primSendReduce);
    vector<unique_ptr<Instruction>> result2 = Translate(primSendReduceDmaGet);
    // then
    Check_PostReady_WaitFin(result1, primSendReduce.GetRemoteRank(), primSendReduce.GetLink());
    Check_PostReady_WaitFin(result2, primSendReduceDmaGet.GetRemoteRank(), primSendReduceDmaGet.GetLink());
}

TEST_F(PrimRulesTest, translate_send_reduce_link_p2p_uint64_prod_not_support_inline_reduce_dma_default_and_dma_get)
{
    RankId localRankID = 0;
    RankId remoteRank = 1;
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData link(portType, localRankID, remoteRank, 0, 1);
    DataSlice localSlice(BufferType::INPUT, 0, dataSliceSize);
    DataSlice remoteSrcSlice(BufferType::SCRATCH, 0, dataSliceSize);
    DataSlice remoteDstSlice(BufferType::SCRATCH, 100, dataSliceSize);

    // 910A2 does not support UINT64+PROD::SUM inline reduce
    PrimSendReduce primSendReduce(
        remoteRank, link, localSlice, remoteSrcSlice, remoteDstSlice, DataType::UINT64, ReduceOp::PROD);
    PrimSendReduce primSendReduceDmaGet(
        remoteRank, link, localSlice, remoteSrcSlice, remoteDstSlice, DataType::UINT64, ReduceOp::PROD, DmaMode::GET);
    // When
    vector<unique_ptr<Instruction>> result1 = Translate(primSendReduce);
    vector<unique_ptr<Instruction>> result2 = Translate(primSendReduceDmaGet);
    // then
    Check_PostReady_WaitFin(result1, primSendReduce.GetRemoteRank(), primSendReduce.GetLink());
    Check_PostReady_WaitFin(result2, primSendReduceDmaGet.GetRemoteRank(), primSendReduceDmaGet.GetLink());
}

TEST_F(PrimRulesTest, translate_send_reduce_link_dev_net_rdma_uint64_prod_not_support_inline_reduce_dma_default)
{
    RankId localRankID = 0;
    RankId remoteRank = 1;
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::RDMA);
    LinkData link(portType, localRankID, remoteRank, 0, 1);
    DataSlice localSlice(BufferType::INPUT, 0, dataSliceSize);
    DataSlice remoteSrcSlice(BufferType::SCRATCH, 0, dataSliceSize);
    DataSlice remoteDstSlice(BufferType::SCRATCH, 100, dataSliceSize);

    // 910A2 does not support UINT64+ReduceOp::SUM inline reduce
    PrimSendReduce primSendReduce(
        remoteRank, link, localSlice, remoteSrcSlice, remoteDstSlice, DataType::UINT64, ReduceOp::PROD);
    // When
    vector<unique_ptr<Instruction>> result1 = Translate(primSendReduce);

    // then
    Check_WithoutInlineReduce_SendReducePut(result1, primSendReduce);
}

TEST_F(PrimRulesTest, translate_send_reduce_link_dev_net_ub_fp32_sum_support_inline_reduce_dma_default)
{
    RankId localRankID = 0;
    RankId remoteRank = 1;
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData link(portType, localRankID, remoteRank, 0, 1);
    DataSlice localSlice(BufferType::INPUT, 0, dataSliceSize);
    DataSlice remoteSrcSlice(BufferType::SCRATCH, 0, dataSliceSize);
    DataSlice remoteDstSlice(BufferType::SCRATCH, 100, dataSliceSize);

    // 910A2 does not support FP32+ReduceOp::SUM inline reduce
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(Hccl::DevType::DEV_TYPE_910A2));
    PrimSendReduce primSendReduce(
        remoteRank, link, localSlice, remoteSrcSlice, remoteDstSlice, DataType::FP32, ReduceOp::SUM);
    DataSlice localSlice1(BufferType::INPUT, 0, dataSliceSize);
    DataSlice remoteSrcSlice1(BufferType::SCRATCH, 0, dataSliceSize);
    DataSlice remoteDstSlice1(BufferType::SCRATCH, 100, dataSliceSize);
    primSendReduce.Append(localSlice1, remoteSrcSlice1, remoteDstSlice1);

    // When
    vector<unique_ptr<Instruction>> result1 = Translate(primSendReduce);

    // then
    Check_WithoutInlineReduce_SendReducePut(result1, primSendReduce);
}

TEST_F(PrimRulesTest, translate_recv_reduce_link_p2p_fp32_sum_support_inline_reduce_dma_default_and_dma_get)
{
    RankId localRankID = 0;
    RankId remoteRank = 1;
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData link(portType, localRankID, remoteRank, 0, 1);
    DataSlice localSlice(BufferType::INPUT, 0, dataSliceSize);
    DataSlice remoteSrcSlice(BufferType::SCRATCH, 0, dataSliceSize);
    DataSlice remoteDstSlice(BufferType::SCRATCH, 100, dataSliceSize);

    // 910A support FP32+ReduceOp::SUM inline reduce
    PrimRecvReduce primRecvReduce(
        remoteRank, link, localSlice, remoteSrcSlice, remoteDstSlice, DataType::FP32, ReduceOp::SUM);

    DataSlice localSlice1(BufferType::INPUT, 0, dataSliceSize);
    DataSlice remoteSrcSlice1(BufferType::SCRATCH, 0, dataSliceSize);
    DataSlice remoteDstSlice1(BufferType::SCRATCH, 100, dataSliceSize);
    primRecvReduce.Append(localSlice1, remoteSrcSlice1, remoteDstSlice1);

    PrimRecvReduce primRecvReduceDmaGet(
        remoteRank, link, localSlice, remoteSrcSlice, remoteDstSlice, DataType::FP16, ReduceOp::SUM, DmaMode::GET);
    // When
    vector<unique_ptr<Instruction>> result1 = Translate(primRecvReduce);
    vector<unique_ptr<Instruction>> result2 = Translate(primRecvReduceDmaGet);
    // then
    Check_WithoutInlineReduce_RecvReduceGet(result2, primRecvReduceDmaGet);
    Check_WaitReady_DMAData_PostFin(
        result1, primRecvReduce.GetRemoteRank(), primRecvReduce.GetLink(), InstructionType::READ_REDUCE,
        primRecvReduce.Size());
}

TEST_F(
    PrimRulesTest,
    translate_recv_reduce_link_p2p_uint64_prod_not_support_inline_reduce_dma_default_and_dma_get_one_slice)
{
    RankId localRankID = 0;
    RankId remoteRank = 1;
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData link(portType, localRankID, remoteRank, 0, 1);
    DataSlice localSlice(BufferType::INPUT, 0, dataSliceSize);
    DataSlice remoteSrcSlice(BufferType::SCRATCH, 0, dataSliceSize);
    DataSlice remoteDstSlice(BufferType::SCRATCH, 100, dataSliceSize);

    // 910A2 does support UINT64+ReduceOp::PROD inline reduce
    PrimRecvReduce primRecvReduce(
        remoteRank, link, localSlice, remoteSrcSlice, remoteDstSlice, DataType::UINT64, ReduceOp::PROD);
    // When
    vector<unique_ptr<Instruction>> result1 = Translate(primRecvReduce);

    // then
    Check_WithoutInlineReduce_RecvReduceGet(result1, primRecvReduce);
}

TEST_F(
    PrimRulesTest,
    translate_recv_reduce_link_p2p_uint64_prod_not_support_inline_reduce_dma_default_and_dma_get_two_slice)
{
    RankId localRankID = 0;
    RankId remoteRank = 1;
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData link(portType, localRankID, remoteRank, 0, 1);
    DataSlice localSlice(BufferType::INPUT, 0, dataSliceSize);
    DataSlice remoteSrcSlice(BufferType::SCRATCH, 0, dataSliceSize);
    DataSlice remoteDstSlice(BufferType::SCRATCH, 100, dataSliceSize);

    // 910A2 does support UINT64+ReduceOp::PROD inline reduce
    PrimRecvReduce primRecvReduce(
        remoteRank, link, localSlice, remoteSrcSlice, remoteDstSlice, DataType::UINT64, ReduceOp::PROD);

    DataSlice localSlice1(BufferType::INPUT, 0, dataSliceSize);
    DataSlice remoteSrcSlice1(BufferType::SCRATCH, 0, dataSliceSize);
    DataSlice remoteDstSlice1(BufferType::SCRATCH, 100, dataSliceSize);
    primRecvReduce.Append(localSlice1, remoteSrcSlice1, remoteDstSlice1);
    // When
    vector<unique_ptr<Instruction>> result1 = Translate(primRecvReduce);
    // then
    Check_WithoutInlineReduce_RecvReduceGet(result1, primRecvReduce); //
}

TEST_F(PrimRulesTest, translate_recv_reduce_link_dev_net_rdma_fp32_sum_support_inline_reduce_dma_default)
{
    RankId localRankID = 0;
    RankId remoteRank = 1;
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::RDMA);
    LinkData link(portType, localRankID, remoteRank, 0, 1);
    DataSlice localSlice(BufferType::INPUT, 0, dataSliceSize);
    DataSlice remoteSrcSlice(BufferType::SCRATCH, 0, dataSliceSize);
    DataSlice remoteDstSlice(BufferType::SCRATCH, 100, dataSliceSize);

    // 910A2 support FP32+ReduceOp::SUM inline reduce
    PrimRecvReduce primRecvReduce(
        remoteRank, link, localSlice, remoteSrcSlice, remoteDstSlice, DataType::FP32, ReduceOp::SUM);
    DataSlice localSlice1(BufferType::INPUT, 0, dataSliceSize);
    DataSlice remoteSrcSlice1(BufferType::SCRATCH, 0, dataSliceSize);
    DataSlice remoteDstSlice1(BufferType::SCRATCH, 100, dataSliceSize);
    primRecvReduce.Append(localSlice1, remoteSrcSlice1, remoteDstSlice1);
    // When
    vector<unique_ptr<Instruction>> result1 = Translate(primRecvReduce);

    // then
    Check_WithoutInlineReduce_RecvReducePut(result1, primRecvReduce);
}

TEST_F(PrimRulesTest, translate_recv_reduce_link_dev_net_rdma_uint64_prod_support_inline_reduce_dma_default)
{
    RankId localRankID = 0;
    RankId remoteRank = 1;
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::RDMA);
    LinkData link(portType, localRankID, remoteRank, 0, 1);
    DataSlice localSlice(BufferType::INPUT, 0, dataSliceSize);
    DataSlice remoteSrcSlice(BufferType::SCRATCH, 0, dataSliceSize);
    DataSlice remoteDstSlice(BufferType::SCRATCH, 100, dataSliceSize);

    // 910A2 does support UINT64+ReduceOp::PROD inline reduce
    PrimRecvReduce primRecvReduce(
        remoteRank, link, localSlice, remoteSrcSlice, remoteDstSlice, DataType::UINT64, ReduceOp::PROD);
    DataSlice localSlice1(BufferType::INPUT, 0, dataSliceSize);
    DataSlice remoteSrcSlice1(BufferType::SCRATCH, 0, dataSliceSize);
    DataSlice remoteDstSlice1(BufferType::SCRATCH, 100, dataSliceSize);
    primRecvReduce.Append(localSlice1, remoteSrcSlice1, remoteDstSlice1);
    // When
    vector<unique_ptr<Instruction>> result1 = Translate(primRecvReduce);

    // then
    Check_WithoutInlineReduce_RecvReducePut(result1, primRecvReduce); //
}

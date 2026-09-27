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
#include "test_mock_setup.h"

#define private public
#define protected public
#include "coll_service_ai_cpu_impl.h"
#include "communicator_impl.h"
#include "virtual_topo.h"
#include "execute_selector.h"
#include "rdma_handle_manager.h"
#include "aicpu_res_package_helper.h"
#include "host_device_sync_notify_manager.h"
#include "queue_bcast_post_cnt_notify_manager.h"
#include "queue_notify_manager_v2.h"
#include "queue_wait_group_cnt_notify_manager.h"
#include "stream_manager.h"
#include "mem_transport_manager.h"
#include "alg_topo_package_helper.h"
#include "rank_gph.h"
#include "mem_transport_manager.h"
#include "ub_mem_transport.h"
#include "p2p_transport.h"
#include "notify_count.h"
#include "base_mem_transport.h"
#include "dev_type.h"
#include "mirror_task_manager.h"
#include "ccu_ins_preprocessor.h"
#include "aicpu_ins_preprocessor.h"
#undef protected
#undef private
using namespace Hccl;

class CollServiceAiCpuImplTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CollServiceAiCpuImplTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "CollServiceAiCpuImplTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in CollServiceAiCpuImplTest SetUp" << std::endl;
        SETUP_COMM_NOTIFY_MOCKS();
    }

    virtual void TearDown()
    {
        std::cout << "A Test case in CollServiceAiCpuImplTest TearDown" << std::endl;
        GlobalMockObject::verify();
    }
    u32 fakeDevPhyId = 1;
    u64 fakeNotifyHandleAddr = 100;
    u32 fakeNotifyId = 1;
    u64 fakeOffset = 200;
    u64 fakeAddress = 300;
    u32 fakePid = 100;
    char fakeName[65] = "testRtsNotify";
    CommunicatorImpl fakeComm;
};

TEST_F(CollServiceAiCpuImplTest, test_SetHcclKernelLaunchParam)
{
    MOCKER_CPP(&CollServiceBase::GetOpCount).stubs().will(returnValue(std::make_pair<u32, u32>(0, 0)));

    HcclKernelLaunchParam param;
    CommunicatorImpl comm;
    comm.opExecuteConfig.accState = AcceleratorState::AICPU_TS;
    comm.InitNotifyManager();
    comm.InitSocketManager();
    comm.InitRmaConnManager();
    comm.InitStreamManager();
    comm.InitHDCommunicate();
    comm.myRank = 0;
    comm.id = "testTag";
    std::shared_ptr<Buffer> buffer = DevBuffer::Create(0x100, 10);
    std::shared_ptr<Buffer> buffer1 = DevBuffer::Create(0x100, 10);
    comm.dataBufferManager = std::make_unique<DataBufManager>();
    comm.dataBufferManager->Register("testTag", BufferType::SCRATCH, buffer);
    comm.rankGraph = std::make_unique<RankGraph>(0);
    comm.connLocalNotifyManager = std::make_unique<ConnLocalNotifyManager>(&comm);
    comm.connLocalCntNotifyManager = std::make_unique<ConnLocalCntNotifyManager>(&comm);
    comm.rmaConnectionManager = std::make_unique<RmaConnManager>(comm);
    comm.currentCollOperator = std::make_unique<CollOperator>();
    comm.currentCollOperator->opMode = OpMode::OPBASE;
    comm.currentCollOperator->opMode = OpMode::OPBASE;
    comm.currentCollOperator->opType = OpType::DEBUGCASE;
    comm.currentCollOperator->inputMem = DevBuffer::Create(0x100, 10);
    comm.currentCollOperator->outputMem = DevBuffer::Create(0x100, 10);
    s32 rankId = 0;
    s32 localId = 0;
    DeviceId deviceId = 0;
    IpAddress inputAddr(0);
    std::set<std::string> ports = {"0/1"};
    std::set<LinkProtocol> protocols = {LinkProtocol::UB_CTP};
    shared_ptr<NetInstance::Peer> peer0 = std::make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
    shared_ptr<NetInstance::ConnInterface> connInterface = std::make_shared<NetInstance::ConnInterface>(
        inputAddr, ports, AddrPosition::HOST, LinkType::PEER2PEER, protocols);
    peer0->AddConnInterface(connInterface);
    comm.rankGraph->AddPeer(peer0);
    comm.localRmaBufManager = std::make_unique<LocalRmaBufManager>(comm);
    comm.cclBuffer = DevBuffer::Create(0x100, 10);

    CollServiceAiCpuImpl service(&comm);
    service.counterBuf = DevBuffer::Create(0x100, 10);
    service.SetHcclKernelLaunchParam(param, &comm);
}

TEST_F(CollServiceAiCpuImplTest, test_SetHcclKernelLaunchParam_op_BATCHSENDRECV)
{
    MOCKER_CPP(&CollServiceBase::GetOpCount).stubs().will(returnValue(std::make_pair<u32, u32>(0, 0)));
    HcclKernelLaunchParam param;
    CommunicatorImpl comm;
    comm.opExecuteConfig.accState = Hccl::AcceleratorState::AICPU_TS;
    comm.InitNotifyManager();
    comm.InitSocketManager();
    comm.InitRmaConnManager();
    comm.InitStreamManager();
    comm.InitHDCommunicate();
    comm.myRank = 0;
    comm.id = "testTag";
    std::shared_ptr<Buffer> buffer = DevBuffer::Create(0x100, 10);
    std::shared_ptr<Buffer> buffer1 = DevBuffer::Create(0x100, 10);
    comm.dataBufferManager = std::make_unique<DataBufManager>();
    comm.dataBufferManager->Register("testTag", BufferType::SCRATCH, buffer);
    comm.rankGraph = std::make_unique<RankGraph>(0);
    comm.connLocalNotifyManager = std::make_unique<ConnLocalNotifyManager>(&comm);
    comm.connLocalCntNotifyManager = std::make_unique<ConnLocalCntNotifyManager>(&comm);
    comm.rmaConnectionManager = std::make_unique<RmaConnManager>(comm);
    comm.currentCollOperator = std::make_unique<CollOperator>();
    comm.currentCollOperator->opMode = OpMode::OPBASE;
    comm.currentCollOperator->opType = OpType::BATCHSENDRECV;
    comm.currentCollOperator->inputMem = DevBuffer::Create(0x100, 10);
    comm.currentCollOperator->outputMem = DevBuffer::Create(0x100, 10);
    s32 rankId = 0;
    s32 localId = 0;
    DeviceId deviceId = 0;
    IpAddress inputAddr(0);
    std::set<std::string> ports = {"0/1"};
    std::set<LinkProtocol> protocols = {LinkProtocol::UB_CTP};
    shared_ptr<NetInstance::Peer> peer0 = std::make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
    shared_ptr<NetInstance::ConnInterface> connInterface = std::make_shared<NetInstance::ConnInterface>(
        inputAddr, ports, AddrPosition::HOST, LinkType::PEER2PEER, protocols);
    comm.rankGraph->AddPeer(peer0);
    comm.localRmaBufManager = std::make_unique<LocalRmaBufManager>(comm);
    comm.cclBuffer = DevBuffer::Create(0x100, 10);

    CollServiceAiCpuImpl service(&comm);
    service.counterBuf = DevBuffer::Create(0x100, 10);
    service.devBatchSendRecvItemBufs = DevBuffer::Create(0x100, 10);
    service.SetHcclKernelLaunchParam(param, &comm);
}

TEST_F(CollServiceAiCpuImplTest, test_AllocBcastPostCntNotify)
{
    CommunicatorImpl comm;
    comm.myRank = 0;
    CollServiceAiCpuImpl service(&comm);

    comm.queueBcastPostCntNotifyManager = std::make_unique<QueueBcastPostCntNotifyManager>();
    std::vector<std::pair<QId, u32>> bcastPostCntNotifyReq;

    service.AllocBcastPostCntNotify(bcastPostCntNotifyReq);
}

TEST_F(CollServiceAiCpuImplTest, test_AllocWaitGroupCntNotify)
{
    CommunicatorImpl comm;
    comm.myRank = 0;
    CollServiceAiCpuImpl service(&comm);

    comm.queueWaitGroupCntNotifyManager = std::make_unique<QueueWaitGroupCntNotifyManager>();
    std::vector<std::pair<QId, u32>> waitGroupCntNotifyReq;

    service.AllocWaitGroupCntNotify(waitGroupCntNotifyReq);
}

TEST_F(CollServiceAiCpuImplTest, test_AllocOpMem)
{
    MOCKER_CPP(&CollServiceBase::GetOpCount).stubs().will(returnValue(std::make_pair<u32, u32>(0, 0)));

    HcclKernelLaunchParam param;
    CommunicatorImpl comm;
    comm.opExecuteConfig.accState = Hccl::AcceleratorState::AICPU_TS;
    comm.InitNotifyManager();
    comm.InitSocketManager();
    comm.InitRmaConnManager();
    comm.InitStreamManager();
    comm.InitHDCommunicate();
    comm.myRank = 0;
    comm.rankSize = 4;
    comm.id = "testTag";
    std::shared_ptr<Buffer> buffer = DevBuffer::Create(0x100, 10);
    std::shared_ptr<Buffer> buffer1 = DevBuffer::Create(0x100, 10);
    comm.dataBufferManager = std::make_unique<DataBufManager>();
    comm.dataBufferManager->Register("testTag", BufferType::SCRATCH, buffer);
    comm.rankGraph = std::make_unique<RankGraph>(0);
    comm.connLocalNotifyManager = std::make_unique<ConnLocalNotifyManager>(&comm);
    comm.connLocalCntNotifyManager = std::make_unique<ConnLocalCntNotifyManager>(&comm);
    comm.rmaConnectionManager = std::make_unique<RmaConnManager>(comm);
    comm.currentCollOperator = std::make_unique<CollOperator>();
    comm.currentCollOperator->opTag = "testTag";
    comm.currentCollOperator->opMode = OpMode::OFFLOAD;
    comm.currentCollOperator->opType = OpType::ALLTOALLV;
    comm.currentCollOperator->inputMem = DevBuffer::Create(0x100, 10);
    comm.currentCollOperator->outputMem = DevBuffer::Create(0x100, 10);
    comm.currentCollOperator->scratchMem = DevBuffer::Create(0x100, 10);
    s32 rankId = 0;
    s32 localId = 0;
    DeviceId deviceId = 0;
    IpAddress inputAddr(0);
    std::set<std::string> ports = {"0/1"};
    std::set<LinkProtocol> protocols = {LinkProtocol::UB_CTP};
    shared_ptr<NetInstance::Peer> peer0 = std::make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
    shared_ptr<NetInstance::ConnInterface> connInterface = std::make_shared<NetInstance::ConnInterface>(
        inputAddr, ports, AddrPosition::HOST, LinkType::PEER2PEER, protocols);
    peer0->AddConnInterface(connInterface);
    comm.rankGraph->AddPeer(peer0);
    comm.localRmaBufManager = std::make_unique<LocalRmaBufManager>(comm);

    u64* sendCounts = (u64*)malloc(comm.rankSize * sizeof(u64));
    u64* recvCounts = (u64*)malloc(comm.rankSize * sizeof(u64));
    u64* sendDispls = (u64*)malloc(comm.rankSize * sizeof(u64));
    u64* recvDispls = (u64*)malloc(comm.rankSize * sizeof(u64));
    u64 count = 2;
    for (u32 i = 0; i < comm.rankSize; i++) {
        sendCounts[i] = count * (i + 1);
        recvCounts[i] = count * (0 + 1);
        sendDispls[i] = count * i * (i + 1) / 2;
        recvDispls[i] = count * (0 + 1) * i;
    }

    CollOperator op;
    op.opTag = "testTag";
    op.opType = OpType::ALLTOALLV;
    op.dataType = DataType::FP32;
    op.dataCount = 3;
    op.all2AllVDataDes.sendType = DataType::FP32;
    op.all2AllVDataDes.recvType = DataType::FP32;
    op.all2AllVDataDes.sendCounts = sendCounts;
    op.all2AllVDataDes.recvCounts = recvCounts;
    op.all2AllVDataDes.sdispls = sendDispls;
    op.all2AllVDataDes.rdispls = recvDispls;
    CollServiceAiCpuImpl service(&comm);
    service.AllocOpMem(op);
    service.counterBuf = DevBuffer::Create(0x100, 10);
    service.SetHcclKernelLaunchParam(param, &comm);
    EXPECT_EQ(service.sendCountsMem.size(), 64);
    free(sendCounts);
    free(recvCounts);
    free(sendDispls);
    free(recvDispls);
}

TEST_F(CollServiceAiCpuImplTest, test_AllocOpMem_op_BATCHSENDRECV)
{
    MOCKER_CPP(&CollServiceBase::GetOpCount).stubs().will(returnValue(std::make_pair<u32, u32>(0, 0)));
    HcclKernelLaunchParam param;
    CommunicatorImpl comm;
    comm.InitNotifyManager();
    comm.InitSocketManager();
    comm.InitRmaConnManager();
    comm.InitStreamManager();
    comm.myRank = 0;
    comm.rankSize = 4;
    comm.id = "testTag";
    std::shared_ptr<Buffer> buffer = DevBuffer::Create(0x100, 10);
    std::shared_ptr<Buffer> buffer1 = DevBuffer::Create(0x100, 10);
    comm.dataBufferManager = std::make_unique<DataBufManager>();
    comm.dataBufferManager->Register("testTag", BufferType::SCRATCH, buffer);
    comm.rankGraph = std::make_unique<RankGraph>(0);
    comm.connLocalNotifyManager = std::make_unique<ConnLocalNotifyManager>(&comm);
    comm.connLocalCntNotifyManager = std::make_unique<ConnLocalCntNotifyManager>(&comm);
    comm.rmaConnectionManager = std::make_unique<RmaConnManager>(comm);
    comm.currentCollOperator = std::make_unique<CollOperator>();
    comm.currentCollOperator->opTag = "testTag";
    comm.currentCollOperator->opMode = OpMode::OFFLOAD;
    comm.currentCollOperator->opType = OpType::ALLTOALLV;
    comm.currentCollOperator->inputMem = DevBuffer::Create(0x100, 10);
    comm.currentCollOperator->outputMem = DevBuffer::Create(0x100, 10);
    comm.currentCollOperator->scratchMem = DevBuffer::Create(0x100, 10);

    s32 rankId = 0;
    s32 localId = 0;
    DeviceId deviceId = 0;
    IpAddress inputAddr(0);
    std::set<std::string> ports = {"0/1"};
    std::set<LinkProtocol> protocols = {LinkProtocol::UB_CTP};
    shared_ptr<NetInstance::Peer> peer0 = std::make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
    shared_ptr<NetInstance::ConnInterface> connInterface = std::make_shared<NetInstance::ConnInterface>(
        inputAddr, ports, AddrPosition::HOST, LinkType::PEER2PEER, protocols);
    peer0->AddConnInterface(connInterface);
    comm.rankGraph->AddPeer(peer0);
    comm.localRmaBufManager = std::make_unique<LocalRmaBufManager>(comm);

    CollOperator op;
    op.opTag = "testTag";
    op.opType = OpType::BATCHSENDRECV;
    op.dataType = DataType::FP32;
    op.dataCount = 3;
    op.batchSendRecvDataDes.itemNum = 2;
    HcclSendRecvItem hcclSendRecvItem[2];
    // 初始化每个 HcclSendRecvItem
    for (u32 i = 0; i < 2; ++i) {
        hcclSendRecvItem[i].sendRecvType = HcclSendRecvType::HCCL_SEND;
        hcclSendRecvItem[i].count = 10;
        hcclSendRecvItem[i].dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
        hcclSendRecvItem[i].remoteRank = i;
    }
    op.batchSendRecvDataDes.sendRecvItemsPtr = &hcclSendRecvItem[0];

    CollServiceAiCpuImpl service(&comm);
    service.AllocOpMem(op);
    service.counterBuf = DevBuffer::Create(0x100, 10);
}

void AlltoallVCTestSetCommOp(CommunicatorImpl& comm)
{
    // set acc mode to AICPU
    comm.opExecuteConfig.accState = AcceleratorState::AICPU_TS;
    // initialize manager in comm
    comm.InitNotifyManager();
    comm.InitSocketManager();
    comm.InitRmaConnManager();
    comm.InitStreamManager();
    comm.InitHDCommunicate();
    comm.myRank = 0;
    comm.rankSize = 4;
    comm.id = "testTag";
    std::shared_ptr<Buffer> buffer = DevBuffer::Create(0x100, 10);
    comm.dataBufferManager = std::make_unique<DataBufManager>();
    comm.dataBufferManager->Register("testTag", BufferType::SCRATCH, buffer);
    comm.rankGraph = std::make_unique<RankGraph>(0);
    comm.connLocalNotifyManager = std::make_unique<ConnLocalNotifyManager>(&comm);
    comm.connLocalCntNotifyManager = std::make_unique<ConnLocalCntNotifyManager>(&comm);
    comm.rmaConnectionManager = std::make_unique<RmaConnManager>(comm);
    // set coll operator in comm
    comm.currentCollOperator = std::make_unique<CollOperator>();
    comm.currentCollOperator->opTag = "testTag";
    comm.currentCollOperator->opMode = OpMode::OFFLOAD;
    comm.currentCollOperator->opType = OpType::ALLTOALLVC;
    comm.currentCollOperator->inputMem = DevBuffer::Create(0x100, 10);
    comm.currentCollOperator->outputMem = DevBuffer::Create(0x100, 10);
    comm.currentCollOperator->scratchMem = DevBuffer::Create(0x100, 10);

    // set virtual topo
    s32 rankId = 0;
    s32 localId = 0;
    DeviceId deviceId = 0;
    IpAddress inputAddr(0);
    std::set<std::string> ports = {"0/1"};
    std::set<LinkProtocol> protocols = {LinkProtocol::UB_CTP};
    shared_ptr<NetInstance::Peer> peer0 = std::make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
    shared_ptr<NetInstance::ConnInterface> connInterface = std::make_shared<NetInstance::ConnInterface>(
        inputAddr, ports, AddrPosition::HOST, LinkType::PEER2PEER, protocols);
    peer0->AddConnInterface(connInterface);
    comm.rankGraph->AddPeer(peer0);
    comm.localRmaBufManager = std::make_unique<LocalRmaBufManager>(comm);
}

TEST_F(CollServiceAiCpuImplTest, St_AllocOpMem_When_Op_ALLTOALLVC_Expect_Success)
{
    MOCKER_CPP(&CollServiceBase::GetOpCount).stubs().will(returnValue(std::make_pair<u32, u32>(0, 0)));
    HcclKernelLaunchParam param;
    CommunicatorImpl comm;
    CollOperator op;
    AlltoallVCTestSetCommOp(comm);

    // initialize sendCountMatrix
    u64* sendMem = (u64*)malloc(comm.rankSize * comm.rankSize * sizeof(u64));
    u64 count = 2;
    for (u32 i = 0; i < comm.rankSize; i++) {
        for (u32 j = 0; j < comm.rankSize; j++) {
            sendMem[i * comm.rankSize + j] = count;
        }
    }

    // initialize op param
    op.opTag = "testTag";
    op.opType = OpType::ALLTOALLVC;
    op.dataType = DataType::FP32;
    op.all2AllVCDataDes.sendType = DataType::FP32;
    op.all2AllVCDataDes.recvType = DataType::FP32;
    op.all2AllVCDataDes.sendCountMatrix = sendMem;

    CollServiceAiCpuImpl service(&comm);
    service.AllocOpMem(op);
    EXPECT_EQ(service.isCountMemInitedAlltoAllVC, true);
    service.counterBuf = DevBuffer::Create(0x100, 10);
    service.SetHcclKernelLaunchParam(param, &comm);
    EXPECT_EQ(service.sendCountMatrixMem.size(), 64);
    free(sendMem);
}

TEST_F(CollServiceAiCpuImplTest, reg_port_ub_first_time_get_then_second_no_throw_aicpu)
{
    MOCKER(HrtIpcSetMemoryName).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any());
    MOCKER(HrtDevMemAlignWithPage)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any());
    MOCKER(HrtIpcDestroyMemoryName).stubs().with(mockcpp::any());
    MOCKER(HrtUbDevQueryInfo).stubs().with(mockcpp::any(), mockcpp::any());
    shared_ptr<DevBuffer> devBuffer = DevBuffer::Create(0x100, 0x100);
    BufferType bufferType = BufferType::SCRATCH;

    CommunicatorImpl comm;
    comm.opExecuteConfig.accState = Hccl::AcceleratorState::AICPU_TS;
    LocalRmaBufManager localRmaBufManager(comm);
    BasePortType basePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    PortData port(0, basePortType, 0, IpAddress());
    string opTag = "optag";

    void* rdmaHandle = (void*)0x200;

    MOCKER_CPP(&RdmaHandleManager::Get).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(rdmaHandle));

    auto res = localRmaBufManager.Reg(opTag, bufferType, devBuffer, port, LinkProtocol::UB_CTP);
    EXPECT_NE(nullptr, res);
    EXPECT_EQ(RmaType::UB, res->GetRmaType());
    auto res2 = localRmaBufManager.Get(opTag, port, bufferType);
    EXPECT_EQ(res, res2);
    std::cout << res->Describe() << std::endl;

    // 重复注册逻辑修改，不再抛异常，而是返回注册好的资源
    EXPECT_NO_THROW(localRmaBufManager.Reg(opTag, bufferType, devBuffer, port, LinkProtocol::UB_CTP));
    auto res3 = localRmaBufManager.Get(opTag, port, bufferType);
    EXPECT_EQ(res, res3);
}

TEST_F(CollServiceAiCpuImplTest, test_IsAllTransportRecoveredReady)
{
    CommunicatorImpl comm;
    comm.InitNotifyManager();
    comm.InitSocketManager();
    comm.InitRmaConnManager();
    comm.InitStreamManager();
    comm.InitMemTransportManager();
    comm.myRank = 0;
    comm.id = "testTag";
    CollServiceAiCpuImpl service(&comm);
    comm.currentCollOperator = make_unique<CollOperator>();
    comm.GetCurrentCollOperator()->opMode = OpMode::OPBASE;

    MOCKER_CPP(&MemTransportManager::IsAllOpbasedTransportRecoveredReady).stubs().will(returnValue(true));

    std::cout << "station 6" << std::endl;
    service.IsAllTransportRecoveredReady(comm.id);
}

extern NetInstance::Link
InitBaseLink2(std::shared_ptr<NetInstance::Node> srcNodePtr, std::shared_ptr<NetInstance::Node> dstNodePtr, u32 hop = 1)
{
    IpAddress srcAddr = IpAddress(0);
    IpAddress dstAddr = IpAddress(0);
    AddrPosition addrPos = AddrPosition::DEVICE;
    LinkType linkType = LinkType::PEER2PEER;
    std::set<LinkProtocol> protocols = {LinkProtocol::UB_CTP};
    LinkDirection direction = LinkDirection::BOTH;
    std::set<std::string> ports = {"0/1"};

    NetInstance::ConnInterface srcIf = NetInstance::ConnInterface(srcAddr, ports, addrPos, linkType, protocols);

    NetInstance::ConnInterface dstIf = NetInstance::ConnInterface(dstAddr, ports, addrPos, linkType, protocols);

    NetInstance::Link link = NetInstance::Link(
        srcNodePtr, dstNodePtr, std::make_shared<NetInstance::ConnInterface>(srcIf),
        std::make_shared<NetInstance::ConnInterface>(dstIf), linkType, protocols, direction, hop);

    return link;
}

TEST_F(CollServiceAiCpuImplTest, test_RecoverTransport)
{
    LocalRmaBuffer* fakeBuffer = nullptr;
    LocalRmaBuffer* rmaBuffer = reinterpret_cast<LocalRmaBuffer*>(0x12345678);
    MOCKER_CPP(
        &LocalRmaBufManager::Get,
        LocalRmaBuffer * (LocalRmaBufManager::*)(const std::string&, const PortData&, BufferType))
        .stubs()
        .will(returnValue(fakeBuffer))
        .then(returnValue(rmaBuffer));

    MOCKER_CPP(
        &LocalRmaBufManager::Reg, LocalRmaBuffer
                                      * (LocalRmaBufManager::*)(const std::string&, BufferType, std::shared_ptr<Buffer>,
                                                                const PortData&, LinkProtocol))
        .stubs()
        .will(returnValue(rmaBuffer));

    void* addr = reinterpret_cast<void*>(0x12345678);
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(addr));
    MOCKER(HrtFree).stubs();

    CommunicatorImpl comm;
    comm.InitNotifyManager();
    comm.InitSocketManager();
    comm.InitRmaConnManager();
    comm.InitStreamManager();
    comm.InitMemTransportManager();
    comm.InitDataBufferManager();
    comm.myRank = 0;
    comm.id = "testTag";
    std::shared_ptr<Buffer> buffer = DevBuffer::Create(0x100, 10);
    std::shared_ptr<Buffer> buffer1 = DevBuffer::Create(0x100, 10);
    comm.dataBufferManager = std::make_unique<DataBufManager>();
    comm.dataBufferManager->Register("testTag", BufferType::SCRATCH, buffer);
    comm.rankGraph = std::make_unique<RankGraph>(0);
    comm.connLocalNotifyManager = std::make_unique<ConnLocalNotifyManager>(&comm);
    comm.connLocalCntNotifyManager = std::make_unique<ConnLocalCntNotifyManager>(&comm);
    comm.rmaConnectionManager = std::make_unique<RmaConnManager>(comm);
    comm.currentCollOperator = std::make_unique<CollOperator>();
    comm.currentCollOperator->opMode = OpMode::OPBASE;
    comm.currentCollOperator->opMode = OpMode::OPBASE;
    comm.currentCollOperator->opType = OpType::DEBUGCASE;
    comm.currentCollOperator->inputMem = DevBuffer::Create(0x100, 10);
    comm.currentCollOperator->outputMem = DevBuffer::Create(0x100, 10);
    s32 rankId = 0;
    s32 localId = 0;
    DeviceId deviceId = 0;
    IpAddress inputAddr(0);
    std::set<std::string> ports = {"0/1"};
    std::set<LinkProtocol> protocols = {LinkProtocol::UB_CTP};
    shared_ptr<NetInstance::Peer> peer0 = std::make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
    shared_ptr<NetInstance::ConnInterface> connInterface = std::make_shared<NetInstance::ConnInterface>(
        inputAddr, ports, AddrPosition::HOST, LinkType::PEER2PEER, protocols);
    peer0->AddConnInterface(connInterface);
    comm.rankGraph->AddPeer(peer0);
    comm.localRmaBufManager = std::make_unique<LocalRmaBufManager>(comm);

    CollServiceAiCpuImpl service(&comm);

    RankId srcRankId = 27;
    LocalId srcLocalId = 27;
    RankId dstRankId = 53;
    LocalId dstLocalId = 53;

    NetInstance::Peer srcPeer = NetInstance::Peer(srcRankId, srcLocalId, srcLocalId, deviceId);
    NetInstance::Peer dstPeer = NetInstance::Peer(dstRankId, dstLocalId, dstLocalId, deviceId);
    std::shared_ptr<NetInstance::Peer> srcPeerPtr = std::make_shared<NetInstance::Peer>(srcPeer);
    std::shared_ptr<NetInstance::Peer> dstPeerPtr = std::make_shared<NetInstance::Peer>(dstPeer);
    NetInstance::Link link = InitBaseLink2(srcPeerPtr, dstPeerPtr); // use from ut_fabric_group.cc

    std::vector<NetInstance::Link> rawLinks;
    rawLinks.push_back(link);

    NetInstance::Path path;
    path.links = rawLinks;

    std::vector<LinkData> links;
    links.push_back(LinkData(path));
    comm.currentCollOperator = make_unique<CollOperator>();
    comm.GetCurrentCollOperator()->opMode = OpMode::OPBASE;

    MOCKER_CPP(&ConnectionsBuilder::BatchBuild).stubs().will(returnValue(0));
    MOCKER_CPP(&SocketManager::BatchCreateSockets).stubs().will(returnValue(0));
    MOCKER_CPP(&CollServiceAiCpuImpl::AllocNotifies).stubs().will(returnValue(0));
    MOCKER_CPP(&MemTransportManager::BatchRecoverOpbasedTransports).stubs().will(returnValue(0));
    MOCKER_CPP(&MemTransportManager::BatchRecoverOffloadTransports).stubs().will(returnValue(0));

    std::cout << "station 6" << std::endl;

    vector<std::pair<LinkGroup, u32>> linkGroupPair;
    LinkGroup linkGroup;
    linkGroup.AddLink({0, 0, IpAddress{"10.0.0.1"}, IpAddress{"10.0.0.2"}});
    linkGroup.AddLink({1, 0, IpAddress{"10.0.0.3"}, IpAddress{"10.0.0.4"}});
    linkGroupPair.push_back(make_pair(linkGroup, 0));

    EXPECT_NO_THROW(service.RecoverTransport(links, linkGroupPair));
}

extern NetInstance::Link
InitBaseLink(std::shared_ptr<NetInstance::Node> srcNodePtr, std::shared_ptr<NetInstance::Node> dstNodePtr, u32 hop = 1);

TEST_F(CollServiceAiCpuImplTest, test_register_ccl_buffer)
{
    LocalRmaBuffer* fakeBuffer = nullptr;
    LocalRmaBuffer* rmaBuffer = reinterpret_cast<LocalRmaBuffer*>(0x12345678);
    MOCKER_CPP(
        &LocalRmaBufManager::Get,
        LocalRmaBuffer * (LocalRmaBufManager::*)(const std::string&, const PortData&, BufferType))
        .stubs()
        .will(returnValue(fakeBuffer))
        .then(returnValue(rmaBuffer));

    MOCKER_CPP(
        &LocalRmaBufManager::Reg, LocalRmaBuffer
                                      * (LocalRmaBufManager::*)(const std::string&, BufferType, std::shared_ptr<Buffer>,
                                                                const PortData&, LinkProtocol))
        .stubs()
        .will(returnValue(rmaBuffer));

    void* addr = reinterpret_cast<void*>(0x12345678);
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(addr));
    MOCKER(HrtFree).stubs();

    CommunicatorImpl comm;
    comm.InitNotifyManager();
    comm.InitSocketManager();
    comm.InitRmaConnManager();
    comm.InitStreamManager();
    comm.InitMemTransportManager();
    comm.InitDataBufferManager();
    comm.myRank = 0;
    comm.id = "testTag";
    std::shared_ptr<Buffer> buffer = DevBuffer::Create(0x100, 10);
    std::shared_ptr<Buffer> buffer1 = DevBuffer::Create(0x100, 10);
    comm.dataBufferManager = std::make_unique<DataBufManager>();
    comm.dataBufferManager->Register("testTag", BufferType::SCRATCH, buffer);
    comm.rankGraph = std::make_unique<RankGraph>(0);
    comm.connLocalNotifyManager = std::make_unique<ConnLocalNotifyManager>(&comm);
    comm.connLocalCntNotifyManager = std::make_unique<ConnLocalCntNotifyManager>(&comm);
    comm.rmaConnectionManager = std::make_unique<RmaConnManager>(comm);
    comm.currentCollOperator = std::make_unique<CollOperator>();
    comm.currentCollOperator->opMode = OpMode::OPBASE;
    comm.currentCollOperator->opMode = OpMode::OPBASE;
    comm.currentCollOperator->opType = OpType::DEBUGCASE;
    comm.currentCollOperator->inputMem = DevBuffer::Create(0x100, 10);
    comm.currentCollOperator->outputMem = DevBuffer::Create(0x100, 10);
    s32 rankId = 0;
    s32 localId = 0;
    DeviceId deviceId = 0;
    IpAddress inputAddr(0);
    std::set<std::string> ports = {"0/1"};
    std::set<LinkProtocol> protocols = {LinkProtocol::UB_CTP};
    shared_ptr<NetInstance::Peer> peer0 = std::make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
    shared_ptr<NetInstance::ConnInterface> connInterface = std::make_shared<NetInstance::ConnInterface>(
        inputAddr, ports, AddrPosition::HOST, LinkType::PEER2PEER, protocols);
    peer0->AddConnInterface(connInterface);
    comm.rankGraph->AddPeer(peer0);
    comm.localRmaBufManager = std::make_unique<LocalRmaBufManager>(comm);

    CollServiceAiCpuImpl service(&comm);

    RankId srcRankId = 27;
    LocalId srcLocalId = 27;
    RankId dstRankId = 53;
    LocalId dstLocalId = 53;

    NetInstance::Peer srcPeer = NetInstance::Peer(srcRankId, srcLocalId, srcLocalId, deviceId);
    NetInstance::Peer dstPeer = NetInstance::Peer(dstRankId, dstLocalId, dstLocalId, deviceId);
    std::shared_ptr<NetInstance::Peer> srcPeerPtr = std::make_shared<NetInstance::Peer>(srcPeer);
    std::shared_ptr<NetInstance::Peer> dstPeerPtr = std::make_shared<NetInstance::Peer>(dstPeer);
    NetInstance::Link link = InitBaseLink2(srcPeerPtr, dstPeerPtr); // use from ut_fabric_group.cc

    std::vector<NetInstance::Link> rawLinks;
    rawLinks.push_back(link);

    NetInstance::Path path;
    path.links = rawLinks;

    std::vector<LinkData> links;
    links.push_back(LinkData(path));

    service.RegisterCclBuffer(links);
    service.RegisterCclBuffer(links); // for duplicated buffer
}

TEST_F(CollServiceAiCpuImplTest, MemTransportManager_getpackeddata)
{
    // mock
    MOCKER_CPP(&MemTransportManager::IsAllOffloadTransportReady, bool(MemTransportManager::*)(const std::string&))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(true));

    // when
    CommunicatorImpl comm;
    MemTransportManager transportManager(comm);
    std::string opTag = "test_tag";
    std::vector<char> res;
    BinaryStream binaryStream;
    binaryStream << 0;
    binaryStream.Dump(res);

    // check
    EXPECT_EQ(transportManager.GetOffloadPackedData(opTag), res);

    // when
    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData linkData(portType, 0, 1, 0, 1);
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    unique_ptr<P2PTransport> transport = make_unique<P2PTransport>(locRes, attr, linkData, fakeSocket);
    P2PTransport* transportPtr = transport.get();
    transportManager.opTagOffloadMap[opTag][linkData] = std::move(transport);
    transportManager.newOffloadTransports[opTag][linkData] = 0;
    RdmaHandle rdmaHandle;
    BaseMemTransport::LocCntNotifyRes locCntNotifyRes;
    std::unique_ptr<BaseMemTransport> trans = make_unique<P2PTransport>(locRes, attr, linkData, fakeSocket);
    MOCKER_CPP_VIRTUAL(*trans, &BaseMemTransport::GetUniqueId).stubs().will(returnValue(std::vector<char>()));

    // check
    EXPECT_NE(transportManager.GetOffloadPackedData(opTag), res);
}

TEST_F(CollServiceAiCpuImplTest, test_coll_service_aicpu_resume)
{
    CommunicatorImpl comm;
    comm.InitNotifyManager();
    comm.InitSocketManager();
    comm.InitRmaConnManager();
    comm.InitStreamManager();
    comm.InitMemTransportManager();
    comm.InitMirrorTaskManager();
    comm.myRank = 0;
    comm.id = "testTag";
    comm.currentCollOperator = make_unique<CollOperator>();
    comm.GetCurrentCollOperator()->opMode = OpMode::OPBASE;

    int a = 10;
    int b = 20;
    void* ptr = &a;
    auto stream = std::make_unique<Stream>(ptr);
    comm.streamManager->opbase->RegisterMaster(std::move(stream));
    void* ptr2 = &b;
    auto freeStream = make_unique<Stream>(ptr2);
    comm.aicpuStreamManager->freeStream = std::move(freeStream);

    CollServiceAiCpuImpl service(&comm);

    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData linkData(portType, 0, 1, 0, 1);
    service.connectionsBuilders.emplace(comm.GetId(), make_unique<ConnectionsBuilder>(comm));
    service.connectionsBuilders[comm.GetId()]->availableLinks.insert(linkData);
    MOCKER_CPP(&RmaConnManager::BatchCreate).stubs();
    MOCKER_CPP(&MemTransportManager::BatchBuildOpbasedTransports).stubs().with(mockcpp::any());
    MOCKER_CPP(&MemTransportManager::IsAllOpbasedTransportReady).stubs().with().will(returnValue(true));
    MOCKER_CPP(&CollServiceAiCpuImpl::SetHcclKernelLaunchParam).stubs().with(mockcpp::any(), mockcpp::any());
    RtsNotify notify(false);
    RtsNotify notify1(false);
    MOCKER_CPP(&HostDeviceSyncNotifyManager::GetHostWaitNotify).stubs().with().will(returnValue(&notify));
    MOCKER_CPP(&HostDeviceSyncNotifyManager::GetDeviceWaitNotify).stubs().with().will(returnValue(&notify1));
    MOCKER_CPP(&Hccl::MirrorTaskManager::AddTaskInfo).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER(HrtNotifyRecord).stubs().with(mockcpp::any());
    MOCKER(HrtNotifyWaitWithTimeOut).stubs();

    service.Resume();

    service.connectionsBuilders.clear();
    service.Resume();
}

TEST_F(CollServiceAiCpuImplTest, test_init_LoadWithOpBasedMode)
{
    u32 fakeDevPhyId = 1;
    u64 fakeNotifyHandleAddr = 100;
    u32 fakeNotifyId = 1;
    u64 fakeOffset = 200;
    char fakeName[65] = "testRtsNotify";
    SETUP_COMM_NOTIFY_MOCKS();

    // 资源初始化
    MOCKER_CPP(&CcuInsPreprocessor::Preprocess).stubs().with().will(ignoreReturnValue());
    MOCKER_CPP(&AicpuInsPreprocessor::Preprocess).stubs().with().will(ignoreReturnValue());

    Buffer* buf = nullptr;
    LocalRmaBuffer* rmaBuf = nullptr;
    MOCKER_CPP(&DataBufManager::Get)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(buf));
    MOCKER_CPP(
        &LocalRmaBufManager::Reg, LocalRmaBuffer
                                      * (LocalRmaBufManager::*)(const string&, BufferType, std::shared_ptr<Buffer>,
                                                                const PortData&, LinkProtocol))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(rmaBuf));
    RtsNotify notify(false);
    RtsNotify notify1(false);
    MOCKER_CPP(&HostDeviceSyncNotifyManager::GetHostWaitNotify).stubs().with().will(returnValue(&notify));
    MOCKER_CPP(&HostDeviceSyncNotifyManager::GetDeviceWaitNotify).stubs().with().will(returnValue(&notify1));
    MOCKER_CPP(&HostDeviceSyncNotifyManager::GetPackedData)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(std::vector<char>{'1', '2'}));
    void* ptr1 = (void*)1;
    MOCKER(HrtStreamCreateWithFlags).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(ptr1));
    MOCKER(HrtGetStreamId).stubs().with(mockcpp::any()).will(returnValue(0));

    fakeComm.cclBuffer = DevBuffer::Create(0x100, 0x100);
    fakeComm.status = CommStatus::COMM_READY;
    fakeComm.InitNotifyManager();
    fakeComm.InitSocketManager();
    fakeComm.InitRmaConnManager();
    fakeComm.InitStreamManager();
    fakeComm.InitMemTransportManager();
    fakeComm.InitMirrorTaskManager();
    fakeComm.InitProfilingReporter();
    fakeComm.myRank = 0;
    fakeComm.id = "testTag";
    fakeComm.devLogicId = 0;
    fakeComm.streamManager->opbase = make_unique<OpbaseStreamManager>(&fakeComm);
    std::shared_ptr<Buffer> buffer = DevBuffer::Create(0x100, 10);
    fakeComm.dataBufferManager = std::make_unique<DataBufManager>();
    fakeComm.dataBufferManager->Register("testTag", BufferType::SCRATCH, buffer);
    fakeComm.rankGraph = std::make_unique<RankGraph>(0);
    fakeComm.connLocalNotifyManager = std::make_unique<ConnLocalNotifyManager>(&fakeComm);
    fakeComm.connLocalCntNotifyManager = std::make_unique<ConnLocalCntNotifyManager>(&fakeComm);
    fakeComm.rmaConnectionManager = std::make_unique<RmaConnManager>(fakeComm);
    fakeComm.currentCollOperator = std::make_unique<CollOperator>();
    fakeComm.currentCollOperator->opTag = "testTag";
    fakeComm.currentCollOperator->scratchMem = buffer;
    fakeComm.currentCollOperator->staticAddr = false;
    fakeComm.currentCollOperator->opMode = OpMode::OPBASE;
    fakeComm.currentCollOperator->opType = OpType::DEBUGCASE;
    fakeComm.currentCollOperator->inputMem = DevBuffer::Create(0x100, 10);
    fakeComm.currentCollOperator->outputMem = DevBuffer::Create(0x100, 10);
    fakeComm.queueWaitGroupCntNotifyManager = std::make_unique<QueueWaitGroupCntNotifyManager>();
    fakeComm.queueBcastPostCntNotifyManager = std::make_unique<QueueBcastPostCntNotifyManager>();
    fakeComm.hostDeviceSyncNotifyManager = std::make_unique<HostDeviceSyncNotifyManager>();
    fakeComm.memTransportManager = make_unique<MemTransportManager>(fakeComm);
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    dfxOpInfo->op_ = *fakeComm.currentCollOperator;
    dfxOpInfo->comm_ = &fakeComm;
    fakeComm.mirrorTaskManager->currDfxOpInfo_ = dfxOpInfo;

    s32 rankId = 0;
    s32 localId = 0;
    DeviceId deviceId = 0;
    IpAddress inputAddr(0);
    std::set<std::string> ports = {"0/1"};
    std::set<LinkProtocol> protocols = {LinkProtocol::UB_CTP};
    shared_ptr<NetInstance::Peer> peer0 = std::make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
    shared_ptr<NetInstance::ConnInterface> connInterface = std::make_shared<NetInstance::ConnInterface>(
        inputAddr, ports, AddrPosition::HOST, LinkType::PEER2PEER, protocols);
    peer0->AddConnInterface(connInterface);
    fakeComm.rankGraph->AddPeer(peer0);
    fakeComm.localRmaBufManager = std::make_unique<LocalRmaBufManager>(fakeComm);
    fakeComm.trace = std::make_unique<Trace>();

    fakeComm.InitCollService();
    fakeComm.CollAlgComponentInit();
    MOCKER_CPP(&CollAlgComponent::ExecAlgSelect)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    OpExecuteConfig opConfig; // aicpu 展开
    opConfig.accState = AcceleratorState::AICPU_TS;
    fakeComm.opExecuteConfig = opConfig;
    fakeComm.SelectCollService();

    // 算法组件初始化
    CollAlgOpReq collAlgOpReq;
    collAlgOpReq.algName = "testAlg";
    collAlgOpReq.resReq.primQueueNum = 1;
    CollAlgComponent collAlgComponent(nullptr, DevType::DEV_TYPE_950, 0, 1);
    MOCKER_CPP_VIRTUAL(
        collAlgComponent, &CollAlgComponent::Orchestrate,
        HcclResult(CollAlgComponent::*)(
            const CollAlgOperator& op, const CollAlgParams& params, const string& algName, InsQuePtr queue))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(
        collAlgComponent, &CollAlgComponent::CalcResOffload,
        HcclResult(CollAlgComponent::*)(
            const OpType& opType, const u64& dataSize, const HcclDataType& dataType, const OpExecuteConfig& opConfig,
            CollOffloadOpResReq& resReq))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(collAlgComponent, &CollAlgComponent::GetCollAlgOpReq)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(collAlgOpReq));
    MOCKER_CPP(&Trace::Save).stubs();
    MOCKER_CPP(&CollServiceAiCpuImpl::AllocOpMem).stubs();
    MOCKER_CPP(&Stream::InitDevPhyId).stubs();
    MOCKER_CPP(&CollServiceBase::SaveMirrorDfxOpInfo).stubs();
    MOCKER_CPP(&CollServiceAiCpuImpl::AddPostToUserStream).stubs().with(mockcpp::any());
    MOCKER_CPP(&CollServiceAiCpuImpl::AddWaitToUserStream).stubs().with(mockcpp::any());
    MOCKER_CPP(&HostDeviceSyncNotifyManager::GetPackedData)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(std::vector<char>{'1', '2'}));
    MOCKER_CPP(&Trace::Save).stubs();
    MOCKER_CPP(&CollServiceAiCpuImpl::SetHcclKernelLaunchParam).stubs().with(mockcpp::any(), mockcpp::any());
    MOCKER_CPP(&CollServiceAiCpuImpl::AicpuKernelEntranceLaunch).stubs().with(mockcpp::any());

    CollOperator op;
    op.inputMem = DevBuffer::Create(0x100, 10);
    op.outputMem = DevBuffer::Create(0x100, 10);
    op.opMode = OpMode::OFFLOAD;
    op.opType = OpType::DEBUGCASE;

    op.opTag = "testTag";
    op.scratchMem = buffer;
    op.staticAddr = false;
    OpType opType = OpType::ALLREDUCE;
    auto stream = std::make_unique<Stream>();
    auto service = dynamic_cast<CollServiceAiCpuImpl*>(fakeComm.collService);
    EXPECT_NO_THROW(service->LoadWithOpBasedMode(op, std::move(stream)));
}

TEST_F(CollServiceAiCpuImplTest, test_GetSnapShotDynamicBuf)
{
    MOCKER_CPP(&ConnectionsBuilder::BatchBuild).stubs().with(mockcpp::any(), mockcpp::any());
    MOCKER_CPP(&MemTransportManager::BatchBuildOpbasedTransports).stubs().with(mockcpp::any());
    MOCKER_CPP(&MemTransportManager::BatchBuildOffloadTransports).stubs().with(mockcpp::any(), mockcpp::any());
    MOCKER_CPP(&MemTransportManager::IsAllOpbasedTransportReady).stubs().with().will(returnValue(true));
    MOCKER_CPP(&MemTransportManager::IsAllOffloadTransportReady).stubs().with(mockcpp::any()).will(returnValue(true));

    CommunicatorImpl comm;
    comm.InitMemTransportManager();
    comm.InitRmaConnManager();
    comm.currentCollOperator = make_unique<CollOperator>();
    comm.currentCollOperator->opTag = "test";
    comm.currentCollOperator->opMode = OpMode::OPBASE;

    CollAlgOpReq collAlgOpReq;
    collAlgOpReq.algName = "testAlg";
    std::vector<std::pair<u32, RankId>> levelRankPairs;
    levelRankPairs.push_back({1, 1});
    collAlgOpReq.resReq.levelRankPairs = levelRankPairs;

    vector<RankId> utRanks = {0};
    RankGroup utRankGroup(utRanks);
    vector<RankGroup> utRankGroups = {utRankGroup};

    u32 utCntCke = 3;

    CollServiceAiCpuImpl service(&comm);
    CollAlgComponent collAlgComponent(nullptr, DevType::DEV_TYPE_950, 0, 1);
    MOCKER_CPP_VIRTUAL(collAlgComponent, &CollAlgComponent::GetCollAlgOpReq)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(collAlgOpReq));
    comm.collAlgComponent = make_shared<CollAlgComponent>(nullptr, DevType::DEV_TYPE_950, 0, 1);
    CollOperator op;
    BinaryStream bs{};

    u32 opAccState{0};
    u32 commAccState{0};
    bool isLoadOp = true;
    u32 submittedOpCnt = 2;
    u32 opMode{0};
    bs << opAccState << commAccState << isLoadOp << submittedOpCnt << opMode;
    EXPECT_EQ(service.GetSnapShotDynamicBuf(op, bs), HcclResult::HCCL_SUCCESS);

    SnapShotParser parse;
    SnapShotDynamic snapShotDynamicBuf;
    EXPECT_EQ(parse.DeSnapShotDynamicBuf(bs, snapShotDynamicBuf), HcclResult::HCCL_SUCCESS);

    EXPECT_EQ(opAccState, snapShotDynamicBuf.opExecuteConfig.accState);
    EXPECT_EQ(commAccState, snapShotDynamicBuf.commExecuteConfig.accState);
    EXPECT_EQ(isLoadOp, snapShotDynamicBuf.isLoadOp);
    EXPECT_EQ(submittedOpCnt, snapShotDynamicBuf.submittedOpCnt);
    EXPECT_EQ(opMode, snapShotDynamicBuf.opMode);
}

TEST_F(CollServiceAiCpuImplTest, st_GetAlgExecParam_When_Aicpu_Expect_ReturnNotSupport)
{
    CommunicatorImpl comm;
    CollServiceAiCpuImpl service(&comm);

    bool clearEnable = true;
    int32_t numBlocks = 2;
    void* commContext = nullptr;
    u64 len = 0;

    EXPECT_EQ(service.GetAlgExecParam(clearEnable, numBlocks, commContext, len), HCCL_E_NOT_SUPPORT);
}

TEST_F(CollServiceAiCpuImplTest, test_LoadWithOffloadMode_Success)
{
    MOCKER(HrtNotifyRecord).stubs();
    MOCKER(HrtNotifyWaitWithTimeOut).stubs();

    Buffer* buf = nullptr;
    LocalRmaBuffer* rmaBuf = nullptr;
    MOCKER_CPP(&DataBufManager::Get)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(buf));
    MOCKER_CPP(
        &LocalRmaBufManager::Reg, LocalRmaBuffer
                                      * (LocalRmaBufManager::*)(const string&, BufferType, std::shared_ptr<Buffer>,
                                                                const PortData&, LinkProtocol))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(rmaBuf));
    MOCKER_CPP(&CollServiceAiCpuImpl::SetHcclKernelLaunchParam).stubs().with(mockcpp::any(), mockcpp::any());
    MOCKER_CPP(&Trace::Save).stubs();

    CollAlgOpReq collAlgOpReq;
    collAlgOpReq.algName = "testAlg";
    collAlgOpReq.resReq.primQueueNum = 1;
    CollAlgComponent collAlgComponent(nullptr, DevType::DEV_TYPE_950, 0, 1);
    MOCKER_CPP_VIRTUAL(collAlgComponent, &CollAlgComponent::GetCollAlgOpReq)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(collAlgOpReq));

    vector<char> fakeBuffer{'0'};
    DevBuffer dataBuffer(8);
    RtsNotify notify(false);
    RtsNotify notify1(false);
    MOCKER_CPP(&DataBufManager::Get)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(&dataBuffer));
    MOCKER_CPP(&HostDeviceSyncNotifyManager::GetHostWaitNotify).stubs().with().will(returnValue(&notify));
    MOCKER_CPP(&HostDeviceSyncNotifyManager::GetDeviceWaitNotify).stubs().with().will(returnValue(&notify1));

    CommunicatorImpl comm;
    comm.InitNotifyManager();
    comm.InitSocketManager();
    comm.InitRmaConnManager();
    comm.InitStreamManager();
    comm.InitMemTransportManager();
    comm.devLogicId = 0;
    comm.InitMirrorTaskManager();
    comm.myRank = 0;
    comm.id = "testTag";
    std::shared_ptr<Buffer> buffer = DevBuffer::Create(0x100, 10);
    std::shared_ptr<Buffer> buffer1 = DevBuffer::Create(0x100, 10);
    comm.dataBufferManager = std::make_unique<DataBufManager>();
    comm.dataBufferManager->Register("testTag", BufferType::SCRATCH, buffer);
    comm.rankGraph = std::make_unique<RankGraph>(0);
    comm.connLocalNotifyManager = std::make_unique<ConnLocalNotifyManager>(&comm);
    comm.connLocalCntNotifyManager = std::make_unique<ConnLocalCntNotifyManager>(&comm);
    comm.rmaConnectionManager = std::make_unique<RmaConnManager>(comm);
    comm.currentCollOperator = std::make_unique<CollOperator>();
    comm.currentCollOperator->opMode = OpMode::OPBASE;
    comm.currentCollOperator->opMode = OpMode::OPBASE;
    comm.currentCollOperator->opType = OpType::DEBUGCASE;
    comm.currentCollOperator->inputMem = DevBuffer::Create(0x100, 10);
    comm.currentCollOperator->outputMem = DevBuffer::Create(0x100, 10);
    s32 rankId = 0;
    s32 localId = 0;
    DeviceId deviceId = 0;
    IpAddress inputAddr(0);
    std::set<std::string> ports = {"0/1"};
    std::set<LinkProtocol> protocols = {LinkProtocol::UB_CTP};
    shared_ptr<NetInstance::Peer> peer0 = std::make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
    shared_ptr<NetInstance::ConnInterface> connInterface = std::make_shared<NetInstance::ConnInterface>(
        inputAddr, ports, AddrPosition::HOST, LinkType::PEER2PEER, protocols);
    peer0->AddConnInterface(connInterface);
    comm.rankGraph->AddPeer(peer0);
    comm.localRmaBufManager = std::make_unique<LocalRmaBufManager>(comm);

    comm.InitCollService();
    comm.CollAlgComponentInit();
    MOCKER_CPP(&CollAlgComponent::ExecAlgSelect)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    OpExecuteConfig opConfig; // aicpu 展开
    opConfig.accState = AcceleratorState::AICPU_TS;
    comm.opExecuteConfig = opConfig;
    comm.SelectCollService();

    CollOperator op;
    op.inputMem = DevBuffer::Create(0x100, 10);
    op.outputMem = DevBuffer::Create(0x100, 10);
    op.opMode = OpMode::OFFLOAD;
    op.opType = OpType::DEBUGCASE;

    op.opTag = "testTag";
    op.scratchMem = buffer;
    op.staticAddr = false;

    OpType opType = OpType::ALLREDUCE;
    CollOffloadOpResReq resReq;
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    dfxOpInfo->op_ = op;
    dfxOpInfo->comm_ = &comm;
    comm.mirrorTaskManager->currDfxOpInfo_ = dfxOpInfo;
    auto stream = std::make_unique<Stream>((void*)1);
    MOCKER_CPP(&HostDeviceSyncNotifyManager::GetPackedData)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(std::vector<char>{'1', '2'}));
    MOCKER_CPP(&Trace::Save).stubs();
    MOCKER_CPP(&CollServiceBase::SaveMirrorDfxOpInfo).stubs();
    CollServiceAiCpuImpl* service = dynamic_cast<CollServiceAiCpuImpl*>(comm.collService);
    EXPECT_NO_THROW(service->LoadWithOffloadMode(op, std::move(stream)));
}

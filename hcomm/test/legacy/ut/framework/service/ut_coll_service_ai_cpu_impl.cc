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
#include "aicpu_res_package_helper.h"
#include "host_device_sync_notify_manager.h"
#include "queue_bcast_post_cnt_notify_manager.h"
#include "queue_notify_manager_v2.h"
#include "queue_wait_group_cnt_notify_manager.h"
#include "stream_manager.h"
#include "mem_transport_manager.h"
#include "alg_topo_package_helper.h"
#include "rank_gph.h"
#include "local_rma_buf_manager.h"
#include "rdma_handle_manager.h"
#include "dev_type.h"
#include "mirror_task_manager.h"
#include "ccu_ins_preprocessor.h"
#include "aicpu_ins_preprocessor.h"
#include "stream.h"
#include "dev_buffer.h"
#include "rma_buffer.h"
#include "internal_exception.h"

#undef protected
#undef private

#include <memory>

using namespace Hccl;

class CollServiceAiCpuImplTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CollServiceAiCpuImplTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "CollServiceAiCpuImplTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in CollServiceAiCpuImplTest SetUp" << std::endl;
        SETUP_COMM_NOTIFY_MOCKS();
        comm.opExecuteConfig.accState = AcceleratorState::AICPU_TS;
        comm.InitNotifyManager();
        comm.InitSocketManager();
        comm.InitRmaConnManager();
        comm.InitStreamManager();
        comm.InitMemTransportManager();
        comm.devLogicId = 0; // InitMirrorTaskManager依赖此字段
        comm.InitMirrorTaskManager();
        comm.RegisterAicpuKernel();
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
        peer0->AddConnInterface(0, connInterface);
        comm.rankGraph->AddPeer(peer0);
        comm.localRmaBufManager = std::make_unique<LocalRmaBufManager>(comm);
        comm.cclBuffer = DevBuffer::Create(0x100, 10);

        op.inputMem = DevBuffer::Create(0x100, 10);
        op.outputMem = DevBuffer::Create(0x100, 10);
        op.opMode = OpMode::OFFLOAD;
        op.opType = OpType::DEBUGCASE;

        op.opTag = "testTag";
        op.scratchMem = buffer;
        op.staticAddr = false;
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
    CommunicatorImpl comm;
    CollOperator op{};
};

TEST_F(CollServiceAiCpuImplTest, Ut_SetHcclKernelLaunchParam_When_Op_DEBUGCASE_Expect_OK)
{
    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));
    comm.InitHDCommunicate();
    HcclKernelLaunchParam param;
    comm.currentCollOperator->opType = OpType::DEBUGCASE;
    CollServiceAiCpuImpl service(&comm);
    service.counterBuf = DevBuffer::Create(0x100, 10);
    EXPECT_NO_THROW(service.SetHcclKernelLaunchParam(param, &comm));
}

TEST_F(CollServiceAiCpuImplTest, Ut_SetHcclKernelLaunchParam_When_Op_BATCHSENDRECV_Expect_OK)
{
    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));
    void* addr = (void*)malloc(32 * 1024);
    MOCKER(HrtMallocHost).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(addr));
    comm.InitHDCommunicate();
    HcclKernelLaunchParam param;
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
    peer0->AddConnInterface(0, connInterface);
    comm.rankGraph->AddPeer(peer0);
    comm.localRmaBufManager = std::make_unique<LocalRmaBufManager>(comm);
    comm.cclBuffer = DevBuffer::Create(0x100, 10);

    CollOperator& op = *comm.currentCollOperator;
    op.opTag = "testTag";
    op.opType = OpType::BATCHSENDRECV;
    op.dataType = DataType::FP32;
    op.dataCount = 3;
    op.batchSendRecvDataDes.itemNum = 2;
    HcclSendRecvItem* hcclSendRecvItem
        = (HcclSendRecvItem*)malloc(op.batchSendRecvDataDes.itemNum * sizeof(HcclSendRecvItem));
    // 初始化每个 HcclSendRecvItem
    for (u32 i = 0; i < 2; ++i) {
        hcclSendRecvItem[i].sendRecvType = HcclSendRecvType::HCCL_SEND;
        hcclSendRecvItem[i].count = 10;
        hcclSendRecvItem[i].dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
        hcclSendRecvItem[i].remoteRank = i;
    }
    op.batchSendRecvDataDes.sendRecvItemsPtr = &hcclSendRecvItem[0];

    CollServiceAiCpuImpl service(&comm);
    service.counterBuf = DevBuffer::Create(0x100, 10);
    service.devBatchSendRecvItemBufs = DevBuffer::Create(0x100, 10);
    EXPECT_NO_THROW(service.AllocOpMem(op));
    EXPECT_NO_THROW(service.SetHcclKernelLaunchParam(param, &comm));
    free(hcclSendRecvItem);
    free(addr);
}

TEST_F(CollServiceAiCpuImplTest, Ut_AllocOpMem_When_Op_ALLTOALLV_Expect_MemSize_Right)
{
    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));
    void* addr = (void*)malloc(32 * 1024);
    MOCKER(HrtMallocHost).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(addr));
    comm.InitHDCommunicate();
    HcclKernelLaunchParam param;
    comm.rankSize = 4;
    comm.currentCollOperator->opMode = OpMode::OFFLOAD;
    comm.currentCollOperator->opType = OpType::ALLTOALLV;
    comm.currentCollOperator->opTag = "testTag";
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
    peer0->AddConnInterface(0, connInterface);
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
    CollOperator& op = *comm.currentCollOperator;
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

    comm.streamManager->offload->RegisterMaster(op.opTag, std::move(make_unique<Stream>()));

    CollServiceAiCpuImpl service(&comm);
    EXPECT_NO_THROW(service.AllocOpMem(op));
    service.counterBuf = DevBuffer::Create(0x100, 10);
    EXPECT_NO_THROW(service.SetHcclKernelLaunchParam(param, &comm));
    free(sendCounts);
    free(recvCounts);
    free(sendDispls);
    free(recvDispls);
    free(addr);
}

TEST_F(CollServiceAiCpuImplTest, Ut_AllocOpMem_When_BATCHSENDRECV_Expect_OK)
{
    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));
    comm.InitHDCommunicate();
    HcclKernelLaunchParam param;
    comm.rankSize = 4;
    comm.currentCollOperator->opMode = OpMode::OFFLOAD;
    comm.currentCollOperator->opType = OpType::BATCHSENDRECV;
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
    peer0->AddConnInterface(0, connInterface);
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
    EXPECT_NO_THROW(service.AllocOpMem(op));
}

TEST_F(CollServiceAiCpuImplTest, Ut_AllocOpMem_When_Op_ALLTOALLVC_Expect_Success)
{
    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));
    void* addr = (void*)malloc(32 * 1024);
    MOCKER(HrtMallocHost).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(addr));
    comm.InitHDCommunicate();
    comm.rankSize = 4;
    comm.currentCollOperator->opMode = OpMode::OFFLOAD;
    comm.currentCollOperator->opType = OpType::ALLTOALLVC;
    comm.currentCollOperator->inputMem = DevBuffer::Create(0x100, 10);
    comm.currentCollOperator->outputMem = DevBuffer::Create(0x100, 10);
    comm.currentCollOperator->scratchMem = DevBuffer::Create(0x100, 10);

    // initialize sendCountMatrix
    u64* sendMem = (u64*)malloc(comm.rankSize * comm.rankSize * sizeof(u64));
    u64 count = 2;
    for (u32 i = 0; i < comm.rankSize; i++) {
        for (u32 j = 0; j < comm.rankSize; j++) {
            sendMem[i * comm.rankSize + j] = count;
        }
    }

    // initialize op param
    CollOperator& op = *comm.currentCollOperator;
    op.opType = OpType::ALLTOALLVC;
    op.dataType = DataType::FP32;
    op.all2AllVCDataDes.sendType = DataType::FP32;
    op.all2AllVCDataDes.recvType = DataType::FP32;
    op.all2AllVCDataDes.sendCountMatrix = sendMem;

    HcclKernelLaunchParam param;
    CollServiceAiCpuImpl service(&comm);
    service.AllocOpMem(op);
    service.counterBuf = DevBuffer::Create(0x100, 10);
    EXPECT_NO_THROW(service.SetHcclKernelLaunchParam(param, &comm));
    free(sendMem);
    free(addr);
}

TEST_F(CollServiceAiCpuImplTest, Ut_InitAicpuLocBufLite_When_Before_SetHcclKernelLaunchParam_Expect_Success)
{
    std::pair<u32, u32> pair(0, 1);
    MOCKER(HrtUbDevQueryToken).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(pair));
    HcclAicpuLocBufLite lite;
    u64 addr = 0x100;
    u64 size = 100;
    EXPECT_NO_THROW(InitAicpuLocBufLite(lite, addr, size, "test"));
    EXPECT_EQ(lite.addr, addr);
    EXPECT_EQ(lite.size, size);
    EXPECT_EQ(lite.tokenId, 0);
    EXPECT_EQ(lite.tokenValue, 1);

    MOCKER(memset_s).stubs().with(mockcpp::any()).will(returnValue(0));
    comm.InitHDCommunicate();
    comm.currentCollOperator->opMode = OpMode::OPBASE;
    comm.currentCollOperator->opType = OpType::DEBUGCASE;
    comm.currentCollOperator->inputMem = DevBuffer::Create(0x100, 10);
    comm.currentCollOperator->outputMem = DevBuffer::Create(0x100, 10);
    comm.currentCollOperator->scratchMem = DevBuffer::Create(0x100, 10);

    HcclKernelLaunchParam param;
    CollServiceAiCpuImpl service(&comm);
    service.counterBuf = DevBuffer::Create(0x100, 10);
    EXPECT_NO_THROW(service.SetHcclKernelLaunchParam(param, &comm));
}

TEST_F(CollServiceAiCpuImplTest, Ut_IsAllTransportRecoveredReady_When_Normal_Expect_Success)
{
    CollServiceAiCpuImpl service(&comm);
    comm.currentCollOperator = make_unique<CollOperator>();
    comm.currentCollOperator->opMode = OpMode::OPBASE;

    MOCKER_CPP(&MemTransportManager::IsAllOpbasedTransportRecoveredReady).stubs().will(returnValue(true));

    std::cout << "station 6" << std::endl;
    EXPECT_NO_THROW(service.IsAllTransportRecoveredReady(comm.id));
}

TEST_F(CollServiceAiCpuImplTest, Ut_AllocBcastPostCntNotify_When_Normal_Expect_Success)
{
    comm.myRank = 0;
    comm.queueBcastPostCntNotifyManager = std::make_unique<QueueBcastPostCntNotifyManager>();

    CollServiceAiCpuImpl service(&comm);
    std::vector<std::pair<QId, u32>> bcastPostCntNotifyReq;
    EXPECT_NO_THROW(service.AllocBcastPostCntNotify(bcastPostCntNotifyReq));
}

TEST_F(CollServiceAiCpuImplTest, Ut_AllocWaitGroupCntNotify_When_Normal_Expect_Success)
{
    comm.myRank = 0;
    comm.queueWaitGroupCntNotifyManager = std::make_unique<QueueWaitGroupCntNotifyManager>();

    CollServiceAiCpuImpl service(&comm);
    std::vector<std::pair<QId, u32>> waitGroupCntNotifyReq;
    EXPECT_NO_THROW(service.AllocWaitGroupCntNotify(waitGroupCntNotifyReq));
}

extern NetInstance::Link
InitBaseLink1(std::shared_ptr<NetInstance::Node> srcNodePtr, std::shared_ptr<NetInstance::Node> dstNodePtr, u32 hop = 1)
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

TEST_F(CollServiceAiCpuImplTest, Ut_RecoverTransport_When_Normal_Expect_Success)
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

    comm.currentCollOperator->opMode = OpMode::OPBASE;
    comm.currentCollOperator->opType = OpType::DEBUGCASE;

    CollServiceAiCpuImpl service(&comm);

    RankId srcRankId = 27;
    LocalId srcLocalId = 27;
    RankId dstRankId = 53;
    LocalId dstLocalId = 53;
    DeviceId deviceId = 0;
    NetInstance::Peer srcPeer = NetInstance::Peer(srcRankId, srcLocalId, srcLocalId, deviceId);
    NetInstance::Peer dstPeer = NetInstance::Peer(dstRankId, dstLocalId, dstLocalId, deviceId);
    std::shared_ptr<NetInstance::Peer> srcPeerPtr = std::make_shared<NetInstance::Peer>(srcPeer);
    std::shared_ptr<NetInstance::Peer> dstPeerPtr = std::make_shared<NetInstance::Peer>(dstPeer);
    NetInstance::Link link = InitBaseLink1(srcPeerPtr, dstPeerPtr); // use from ut_fabric_group.cc

    std::vector<NetInstance::Link> rawLinks;
    rawLinks.push_back(link);

    NetInstance::Path path;
    path.links = rawLinks;

    std::vector<LinkData> links;
    links.push_back(LinkData(path));
    comm.currentCollOperator = make_unique<CollOperator>();
    comm.GetCurrentCollOperator()->opMode = OpMode::OPBASE;
    MOCKER_CPP(&ConnectionsBuilder::BatchBuild).stubs().will(returnValue(0));
    MOCKER_CPP(&SocketManager::BatchCreateSockets, void(SocketManager::*)(const std::vector<LinkData>&))
        .stubs()
        .will(returnValue(0));
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

TEST_F(CollServiceAiCpuImplTest, Ut_RegisterCclBuffer_When_Normal_Expect_Success)
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

    CollServiceAiCpuImpl service(&comm);
    RankId srcRankId = 27;
    LocalId srcLocalId = 27;
    RankId dstRankId = 53;
    LocalId dstLocalId = 53;
    DeviceId deviceId = 0;

    NetInstance::Peer srcPeer = NetInstance::Peer(srcRankId, srcLocalId, srcLocalId, deviceId);
    NetInstance::Peer dstPeer = NetInstance::Peer(dstRankId, dstLocalId, dstLocalId, deviceId);
    std::shared_ptr<NetInstance::Peer> srcPeerPtr = std::make_shared<NetInstance::Peer>(srcPeer);
    std::shared_ptr<NetInstance::Peer> dstPeerPtr = std::make_shared<NetInstance::Peer>(dstPeer);
    NetInstance::Link link = InitBaseLink1(srcPeerPtr, dstPeerPtr); // use from ut_fabric_group.cc

    std::vector<NetInstance::Link> rawLinks;
    rawLinks.push_back(link);

    NetInstance::Path path;
    path.links = rawLinks;

    std::vector<LinkData> links;
    links.push_back(LinkData(path));

    EXPECT_NO_THROW(service.RegisterCclBuffer(links));
    EXPECT_NO_THROW(service.RegisterCclBuffer(links)); // for duplicated buffer
}

TEST_F(CollServiceAiCpuImplTest, Ut_Resume_When_Normal_Expect_Success)
{
    CommunicatorImpl comm;
    void* addr = (void*)malloc(32 * 1024);
    MOCKER(HrtMallocHost).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(addr));
    comm.InitNotifyManager();
    comm.InitSocketManager();
    comm.InitRmaConnManager();
    comm.InitStreamManager();
    comm.InitMemTransportManager();
    comm.InitMirrorTaskManager();
    comm.RegisterAicpuKernel();
    comm.myRank = 0;
    comm.devLogicId = 0;
    comm.id = "testTag";
    comm.currentCollOperator = make_unique<CollOperator>();
    comm.GetCurrentCollOperator()->opMode = OpMode::OPBASE;
    void* ptr;
    auto stream = std::make_unique<Stream>(ptr);
    MOCKER(HrtStreamGetMode).stubs().will(returnValue((unsigned long long)0));
    comm.streamManager->opbase->RegisterMaster(std::move(stream));
    void* ptr2;
    auto freeStream = make_unique<Stream>(ptr2);
    comm.aicpuStreamManager->freeStream = std::move(freeStream);

    CollServiceAiCpuImpl service(&comm);

    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData linkData(portType, 0, 1, 0, 1);
    service.connectionsBuilders.emplace(comm.GetId(), make_unique<ConnectionsBuilder>(comm));
    service.connectionsBuilders[comm.GetId()]->availableLinks.insert(linkData);
    service.kernelParamBuf_ = make_shared<HostBuffer>(KERNEL_PARAM_BUF_SIZE);
    MOCKER_CPP(&RmaConnManager::BatchCreate).stubs();
    MOCKER_CPP(&MemTransportManager::BatchBuildOpbasedTransports).stubs().with(mockcpp::any());
    MOCKER_CPP(&MemTransportManager::IsAllOpbasedTransportReady).stubs().with().will(returnValue(true));
    MOCKER_CPP(&CollServiceAiCpuImpl::SetHcclKernelLaunchParam).stubs().with(mockcpp::any(), mockcpp::any());
    RtsNotify notify(false);
    RtsNotify notify1(false);
    MOCKER_CPP(&HostDeviceSyncNotifyManager::GetHostWaitNotify).stubs().with().will(returnValue(&notify));
    MOCKER_CPP(&HostDeviceSyncNotifyManager::GetDeviceWaitNotify).stubs().with().will(returnValue(&notify1));
    MOCKER_CPP(static_cast<void (Hccl::MirrorTaskManager::*)(std::unique_ptr<Hccl::TaskInfo>&&)>(
                   &Hccl::MirrorTaskManager::AddTaskInfo))
        .stubs()
        .with(mockcpp::any())
        .will(ignoreReturnValue());

    EXPECT_NO_THROW(service.Resume());

    service.connectionsBuilders.clear();
    EXPECT_NO_THROW(service.Resume());
    free(addr);
}

TEST_F(CollServiceAiCpuImplTest, Ut_LoadWithOpBasedMode_When_Normal_Expect_Success)
{
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
    comm.RegisterAicpuKernel();
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
    peer0->AddConnInterface(0, connInterface);
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

    auto service = dynamic_cast<CollServiceAiCpuImpl*>(comm.collService);
    EXPECT_NO_THROW(service->LoadWithOpBasedMode(op, std::move(stream)));
}

TEST_F(CollServiceAiCpuImplTest, Ut_LoadWithOpBasedMode_When_Loop_Expect_Success)
{
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
    MOCKER_CPP(&Trace::Save).stubs();
    CommunicatorImpl comm;
    comm.InitNotifyManager();
    comm.InitSocketManager();
    comm.InitRmaConnManager();
    comm.InitStreamManager();
    comm.InitMemTransportManager();
    comm.devLogicId = 0;
    comm.InitMirrorTaskManager();
    comm.RegisterAicpuKernel();
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
    peer0->AddConnInterface(0, connInterface);
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

    OpType opType = OpType::ALLREDUCE;
    CollOffloadOpResReq resReq;
    GlobalMirrorTasks& globalMirrorTasks = GlobalMirrorTasks::Instance();
    std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    dfxOpInfo->op_ = op;
    dfxOpInfo->comm_ = &comm;
    comm.mirrorTaskManager->currDfxOpInfo_ = dfxOpInfo;
    auto stream = std::make_unique<Stream>((void*)1);

    shared_ptr<DevBuffer> devMem = std::make_shared<DevBuffer>(10);
    auto service = dynamic_cast<CollServiceAiCpuImpl*>(comm.collService);
    service->collOpLoadedMap.insert(make_pair("testTag", devMem));

    MOCKER_CPP(&HostDeviceSyncNotifyManager::GetPackedData)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(std::vector<char>{'1', '2'}));
    MOCKER_CPP(&AicpuStreamManager::GetStreams).stubs().will(returnValue(std::vector<Stream*>{}));
    EXPECT_NO_THROW(service->LoadWithOpBasedMode(op, std::move(stream)));
}

TEST_F(CollServiceAiCpuImplTest, Ut_GetSnapShotDynamicBuf_When_Normal_Expect_Success)
{
    MOCKER_CPP(&ConnectionsBuilder::BatchBuild).stubs().with(mockcpp::any(), mockcpp::any());
    MOCKER_CPP(&MemTransportManager::BatchBuildOpbasedTransports).stubs().with(mockcpp::any());
    MOCKER_CPP(&MemTransportManager::BatchBuildOffloadTransports).stubs().with(mockcpp::any(), mockcpp::any());
    MOCKER_CPP(&MemTransportManager::IsAllOpbasedTransportReady).stubs().with().will(returnValue(true));
    MOCKER_CPP(&MemTransportManager::IsAllOffloadTransportReady).stubs().with(mockcpp::any()).will(returnValue(true));

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

TEST_F(CollServiceAiCpuImplTest, Ut_LoadWithOffloadMode_When_Normal_Loop_Expect_Success)
{
    CommunicatorImpl comm;
    CollServiceAiCpuImpl service(&comm);

    MOCKER_CPP(&CollServiceBase::RegisterOpBufToBufMgr).stubs().will(ignoreReturnValue());

    CollOperator op;
    EXPECT_ANY_THROW(service.LoadWithOffloadMode(op, std::make_unique<Stream>(nullptr)));
}

TEST_F(CollServiceAiCpuImplTest, test_LoadWithOffloadMode_Success)
{
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
    comm.RegisterAicpuKernel();
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
    peer0->AddConnInterface(0, connInterface);
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

TEST_F(CollServiceAiCpuImplTest, Ut_AllocQueueNotify_When_Ccu_Host_Alloc_Queue_Notify_Expect_ReturnIsInternal_Error)
{
    CommunicatorImpl comm;
    CollServiceAiCpuImpl service(&comm);
    const InsQueue insQueue{};
    EXPECT_THROW(service.AllocQueueNotify(insQueue), InternalException);
}

TEST_F(
    CollServiceAiCpuImplTest,
    Ut_AllocQNotifyForSingleQ_When_Ccu_Host_Alloc_Q_Notify_For_Single_Q_Expect_ReturnIsInternal_Error)
{
    CommunicatorImpl comm;
    CollServiceAiCpuImpl service(&comm);
    const InsQueue insQueue{};
    EXPECT_THROW(service.AllocQNotifyForSingleQ(insQueue), InternalException);
}

TEST_F(CollServiceAiCpuImplTest, ut_GetAlgExecParam_When_Aicpu_Expect_ReturnNotSupport)
{
    CommunicatorImpl comm;
    CollServiceAiCpuImpl service(&comm);

    bool clearEnable = true;
    int32_t numBlocks = 2;
    void* commContext = nullptr;
    u64 len = 0;

    EXPECT_EQ(service.GetAlgExecParam(clearEnable, numBlocks, commContext, len), HCCL_E_NOT_SUPPORT);
}

TEST_F(CollServiceAiCpuImplTest, Ut_AllocCollOpResource_When_Normal_Loop_Expect_OK)
{
    comm.InitCollService();
    comm.CollAlgComponentInit();
    MOCKER_CPP(&CollAlgComponent::ExecAlgSelect)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    std::shared_ptr<DevBuffer> buffer = DevBuffer::Create(0x100, 10);
    MOCKER_CPP(&CollServiceAiCpuImpl::OpBasedCollProcess).stubs().will(returnValue(buffer.get()));
    MOCKER_CPP(&AicpuStreamManager::GetStreams).stubs().will(returnValue(std::vector<Stream*>{}));
    MOCKER_CPP(&CollServiceAiCpuImpl::SaveMirrorDfxOpInfo).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::ReportHcclMC2Info).stubs();
    MOCKER_CPP(&Trace::Save).stubs();
    MOCKER_CPP(&CollServiceAiCpuImpl::SetHcclKernelLaunchParam).stubs().with(mockcpp::any(), mockcpp::any());
    void* addr = reinterpret_cast<void*>(0x12345678);
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(addr));

    OpExecuteConfig opConfig; // aicpu 展开
    opConfig.accState = AcceleratorState::AICPU_TS;
    comm.opExecuteConfig = opConfig;
    comm.SelectCollService();
    auto service = dynamic_cast<CollServiceAiCpuImpl*>(comm.collService);
    rtStream_t ptr;
    unique_ptr<Stream> master = make_unique<Stream>((ptr));
    comm.streamManager->opbase->master = std::move(master);

    void* addr0 = nullptr;
    std::string opAlgTag = "opAlgTag";
    EXPECT_NO_THROW(service->AllocCollOpResource(op, opAlgTag, &addr0));
    EXPECT_EQ(1, service->aicpuMc2CommResourceMap_.size());
    std::string opAlgTag2 = "opAlgTag";
    void* addr1 = nullptr;
    EXPECT_NO_THROW(service->AllocCollOpResource(op, opAlgTag2, &addr1));
    EXPECT_EQ(addr, addr1);
    EXPECT_EQ(1, service->aicpuMc2CommResourceMap_.size());
    std::string opAlgTag3 = "opAlgTag3";
    EXPECT_NO_THROW(service->AllocCollOpResource(op, opAlgTag3, &addr0));
    EXPECT_EQ(2, service->aicpuMc2CommResourceMap_.size());
}

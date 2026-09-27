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
#include "coll_service_device_mode.h"
#include "ccu_instruction_all_gather_mesh1d.h"
#include "aicpu_res_package_helper.h"
#include "alg_topo_package_helper.h"
#undef private
#undef protected

using namespace Hccl;
using namespace std;

class AicpuInsPreprocessorTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CommunicatorImplTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "CommunicatorImplTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        SETUP_COMM_NOTIFY_MOCKS();
        std::cout << "A Test case in CommunicatorImplTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CommunicatorImplTest TearDown" << std::endl;
    }

    u32 fakeDevPhyId = 1;
    u64 fakeNotifyHandleAddr = 100;
    u32 fakeNotifyId = 1;
    u64 fakeOffset = 200;
    u64 fakeAddress = 300;
    u32 fakePid = 100;
    char fakeName[65] = "testRtsNotify";
};

TEST(AicpuInsPreprocessorTest, should_no_throw_when_calling_isAicpuResExisted)
{
    GlobalMockObject::verify();
    shared_ptr<CommunicatorImpl> comm = make_shared<CommunicatorImpl>();
    AicpuInsPreprocessor aicpuInsPreprocessor(comm.get());
    EXPECT_THROW(aicpuInsPreprocessor.IsAicpuResExisted("test"), NullPtrException);
    aicpuInsPreprocessor.aicpuResExistedMap["test"] = true;
    shared_ptr<DevBuffer> devbuf = make_shared<DevBuffer>(2);
    aicpuInsPreprocessor.aicpuResMap["test"] = devbuf;
    EXPECT_EQ(aicpuInsPreprocessor.IsAicpuResExisted("test"), true);
}

TEST(AicpuInsPreprocessorTest, should_no_throw_when_calling_getAicpuResBuffer)
{
    shared_ptr<CommunicatorImpl> comm = make_shared<CommunicatorImpl>();
    AicpuInsPreprocessor aicpuInsPreprocessor(comm.get());
    EXPECT_THROW(aicpuInsPreprocessor.GetAicpuResBuffer("test"), NullPtrException);
    shared_ptr<DevBuffer> devbuf = make_shared<DevBuffer>(2);
    aicpuInsPreprocessor.aicpuResMap["test"] = devbuf;
    EXPECT_NE(nullptr, aicpuInsPreprocessor.GetAicpuResBuffer("test"));
}

TEST(AicpuInsPreprocessorTest, should_no_throw_when_calling_allocInterRankNotifies)
{
    MOCKER_CPP(&ConnLocalNotifyManager::ApplyFor).stubs().with(mockcpp::any(), mockcpp::any());
    CommunicatorImpl comm;
    comm.InitNotifyManager();
    AicpuInsPreprocessor aicpuInsPreprocessor(&comm);
    BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
    LinkData linkData(portType, 0, 1, 0, 1);
    std::vector<LinkData> links;
    links.push_back(linkData);
    EXPECT_NO_THROW(aicpuInsPreprocessor.AllocInterRankNotifies(links));
}

TEST(AicpuInsPreprocessorTest, should_no_throw_when_calling_batchBuildTransports)
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
    CollServiceDeviceMode collService{&comm};
    comm.collService = &collService;
    AicpuInsPreprocessor aicpuInsPreprocessor(&comm);
    std::vector<LinkData> links;
    EXPECT_NO_THROW(aicpuInsPreprocessor.BatchBuildTransports(links));

    comm.currentCollOperator->opMode = OpMode::OFFLOAD;
    EXPECT_NO_THROW(aicpuInsPreprocessor.BatchBuildTransports(links));
}

TEST(AicpuInsPreprocessorTest, should_no_throw_when_calling_packResAndCopyToDev)
{
    MOCKER_CPP(&AicpuInsPreprocessor::PackOpData)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(std::vector<char>{'1'}));
    MOCKER(HrtMemcpy).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any());

    CommunicatorImpl comm;
    comm.InitMemTransportManager();
    comm.currentCollOperator = make_unique<CollOperator>();
    comm.currentCollOperator->opTag = "test";
    AicpuInsPreprocessor aicpuInsPreprocessor(&comm);
    CollAlgResReq resReq;
    EXPECT_NO_THROW(aicpuInsPreprocessor.PackResAndCopyToDev("test", resReq));
}

TEST(AicpuInsPreprocessorTest, test_AllocAlltoallVOpMem)
{
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
    IpAddress inputAddr(0);
    std::set<string> ports = {"0/1"};
    std::set<LinkProtocol> protocols = {LinkProtocol::UB_CTP};
    DeviceId deviceId = 0;
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
    comm.currentCollOperator = make_unique<CollOperator>(op);

    AicpuInsPreprocessor aicpuInsPreprocessor(&comm);
    aicpuInsPreprocessor.AllocAlltoallVOpMem();
    aicpuInsPreprocessor.SetAicpuKernelLaunchParam(param);
    EXPECT_EQ(aicpuInsPreprocessor.sendCountsMem.size(), 64);
    free(sendCounts);
    free(recvCounts);
    free(sendDispls);
    free(recvDispls);
}

TEST(AicpuInsPreprocessorTest, should_no_throw_when_calling_setAicpuResExisted)
{
    CommunicatorImpl comm;
    AicpuInsPreprocessor aicpuInsPreprocessor(&comm);
    EXPECT_THROW(aicpuInsPreprocessor.SetAicpuResExisted("test"), NullPtrException);

    aicpuInsPreprocessor.aicpuResExistedMap["test"] = true;
    EXPECT_NO_THROW(aicpuInsPreprocessor.SetAicpuResExisted("test"));
}

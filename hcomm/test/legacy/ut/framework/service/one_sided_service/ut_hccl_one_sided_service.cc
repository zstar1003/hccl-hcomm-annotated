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
#include "stub_communicator_impl_trans_mgr.h"

#define private public
#define protected public

#include "ub_mem_transport.h"
#include "ub_local_notify.h"
#include "local_ub_rma_buffer.h"
#include "socket_exception.h"
#include "hccl_one_sided_service_v2.h"

#include "stub_communicator_impl_trans_mgr.h"
#include "detour_service.h"
#include "phy_topo_builder.h"
#include "rank_table.h"
#include "kfc.h"
#include "transport_urma_mem.h"
#include "dev_buffer.h"
#include "rma_buffer.h"
#undef protected
#undef private

#define HCCL_HDC_TYPE_D2H 0
#define HCCL_HDC_TYPE_H2D 1
#include <memory>
#include <future>
#include <chrono>
#include <vector>
#include <stdexcept>

using namespace Hccl;

class HcclOneSidedServiceTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HcclOneSidedServiceTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "HcclOneSidedServiceTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in HcclOneSidedServiceTest SetUp" << std::endl;
        SETUP_COMM_NOTIFY_MOCKS();
    }

    virtual void TearDown()
    {
        std::cout << "A Test case in HcclOneSidedServiceTest TearDown" << std::endl;
        GlobalMockObject::verify();
    }
    u32 fakeDevPhyId = 1;
    u64 fakeNotifyHandleAddr = 100;
    u32 fakeNotifyId = 1;
    u64 fakeOffset = 200;
    u64 fakeAddress = 300;
    u32 fakePid = 100;
    char fakeName[65] = "testRtsNotify";
};

TEST_F(HcclOneSidedServiceTest, test_RegMemAndDeregMem)
{
    LinkData linkData(BasePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB), 0, 1, 0, 1);
    StubCommunicatorImplTransMgr fakeComm;
    HcclOneSidedService oneSidedService(fakeComm);
    HcclMemDesc localMemDesc;
    RankId remoteRankId = 0;

    oneSidedService.linkDataMap_.emplace(remoteRankId, linkData);

    int* a = new int();
    oneSidedService.RegMem(a, sizeof(int), HcclMemType::HCCL_MEM_TYPE_DEVICE, remoteRankId, localMemDesc);
    oneSidedService.DeregMem(localMemDesc);
    delete a;
}

TEST_F(HcclOneSidedServiceTest, test_RegMem_Fail_1)
{
    StubCommunicatorImplTransMgr fakeComm;
    HcclOneSidedService oneSidedService(fakeComm);
    HcclMemDesc localMemDesc;
    RankId remoteRankId = 0;
    int* a = new int();
    oneSidedService.RegMem(a, sizeof(int), HcclMemType::HCCL_MEM_TYPE_HOST, remoteRankId, localMemDesc);
    delete a;
}

TEST_F(HcclOneSidedServiceTest, test_DeregMem_Fail_1)
{
    StubCommunicatorImplTransMgr fakeComm;
    HcclOneSidedService oneSidedService(fakeComm);
    HcclMemDesc localMemDesc;
    oneSidedService.DeregMem(localMemDesc);
}

TEST_F(HcclOneSidedServiceTest, test_ExchangeMemDesc)
{
    RankId RankIdA = 0;
    RankId RankIdB = 1;
    LinkData linkData1(BasePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB), RankIdA, RankIdB, 0, 1);
    StubCommunicatorImplTransMgr fakeCommA;
    HcclOneSidedService oneSidedServiceA(fakeCommA);
    HcclMemDesc MemDescA;
    HcclMemDesc MemDescB;
    int* a = new int;
    // 打桩--------------------------------------------------------------------------------------------------

    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::DEV_NET), 0, 1, 0, 1);
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    bool isRecvFirst = false;
    std::unique_ptr<UbMemTransport> transport
        = make_unique<UbMemTransport>(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);

    oneSidedServiceA.linkDataMap_.emplace(RankIdB, linkData1);
    MOCKER_CPP(&ConnectionsBuilder::BatchBuild).stubs().will(returnValue(0));
    MOCKER_CPP(&SocketManager::BatchCreateSockets, void(SocketManager::*)(const std::vector<LinkData>&))
        .stubs()
        .will(returnValue(0));
    MOCKER_CPP(&ConnLocalNotifyManager::ApplyFor).stubs().will(returnValue(0));
    MOCKER_CPP(&MemTransportManager::BatchBuildOneSidedTransports).stubs().will(returnValue(0));
    fakeCommA.memTransportManager = std::move(std::make_unique<MemTransportManager>(fakeCommA));
    fakeCommA.GetMemTransportManager()->oneSidedMap[linkData1] = std::move(transport);
    MOCKER_CPP(&MemTransportManager::IsAllOneSidedTransportReady).stubs().will(returnValue(true));

    // // 打桩 SocketManager::GetConnectedSocket
    shared_ptr<Socket> fakeSocketPtr
        = make_shared<Socket>(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    SocketConfig socketConfig(linkData1.GetRemoteRankId(), linkData1, fakeCommA.GetEstablishLinkSocketTag());
    fakeCommA.GetSocketManager().connectedSocketMap[socketConfig] = std::move(fakeSocketPtr);

    SocketStatus fakeSocketStatus = SocketStatus::OK;
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue(fakeSocketStatus));
    MOCKER(HrtRaSocketBlockRecv).stubs().will(returnValue(true));
    MOCKER(HrtRaSocketBlockSend).stubs().will(returnValue(true));
    //-----------------------------------------------------------------------------------------------------
    oneSidedServiceA.RegMem(a, sizeof(int), HcclMemType::HCCL_MEM_TYPE_HOST, RankIdB, MemDescA);
    HcclMemDescs localMemDescs;
    localMemDescs.array = &MemDescA;
    localMemDescs.arrayLength = 1;

    HcclMemDescs remoteMemDescs;
    u32 actualNumOfB = 0;

    oneSidedServiceA.ExchangeMemDesc(RankIdB, localMemDescs, remoteMemDescs, actualNumOfB);
    delete a;
}

TEST_F(HcclOneSidedServiceTest, EnableMemAccess_ConnectionNotFound)
{
    CommunicatorImpl com;
    HcclOneSidedService service(com);
    // 创建一个HcclMemDesc对象，其中localRankId不存在于oneSidedConns_
    RmaMemDesc rmaDesc;
    rmaDesc.localRankId = 999; // 假设999不在oneSidedConns_中
    const size_t bufferSize = 512;
    const size_t copySize = sizeof(RmaMemDesc);
    HcclMemDesc desc;
    memcpy_s(desc.desc, bufferSize, &rmaDesc, copySize);
    HcclMem mem; // 假设HcclMem的构造不会抛出异常
    EXPECT_EQ(service.EnableMemAccess(desc, mem), HCCL_E_NOT_FOUND);
}

TEST_F(HcclOneSidedServiceTest, DisableMemAccess_ConnectionNotFound)
{
    CommunicatorImpl com;
    HcclOneSidedService service(com);

    // 创建一个HcclMemDesc对象，其中localRankId不存在于oneSidedConns_
    HcclMemDesc desc;
    RmaMemDesc* rmaDesc = new RmaMemDesc();
    rmaDesc->localRankId = 999; // 假设999不在oneSidedConns_中
    const size_t bufferSize = 512;
    const size_t copySize = sizeof(RmaMemDesc);
    memcpy_s(desc.desc, bufferSize, rmaDesc, copySize);
    EXPECT_EQ(service.DisableMemAccess(desc), HCCL_E_NOT_FOUND);
    delete rmaDesc; // 释放内存
}

TEST_F(HcclOneSidedServiceTest, test_BatchGet_BatchPut)
{
    RankId RankIdA = 0;
    RankId RankIdB = 1;
    LinkData linkData1(BasePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB), RankIdA, RankIdB, 0, 1);
    StubCommunicatorImplTransMgr fakeCommA;
    fakeCommA.commExecuteConfig.accState = AcceleratorState::AICPU_TS;
    fakeCommA.RegisterAicpuKernel();
    HcclOneSidedService oneSidedServiceA(fakeCommA);
    HcclMemDesc MemDescA;
    HcclMemDesc MemDescB;
    int* a = new int;

    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link(BasePortType(PortDeploymentType::DEV_NET), 0, 1, 0, 1);
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress("1.0.0.0");
    Socket fakeSocket(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    bool isRecvFirst = false;
    std::unique_ptr<UbMemTransport> transport
        = make_unique<UbMemTransport>(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);
    transport->baseStatus = TransportStatus::READY;

    oneSidedServiceA.linkDataMap_.emplace(RankIdB, linkData1);
    MOCKER_CPP(&ConnectionsBuilder::BatchBuild).stubs().will(returnValue(0));
    MOCKER_CPP(&SocketManager::BatchCreateSockets, void(SocketManager::*)(const std::vector<LinkData>&))
        .stubs()
        .will(returnValue(0));
    MOCKER_CPP(&ConnLocalNotifyManager::ApplyFor).stubs().will(returnValue(0));
    MOCKER_CPP(&MemTransportManager::BatchBuildOneSidedTransports).stubs().will(returnValue(0));
    fakeCommA.memTransportManager = std::move(std::make_unique<MemTransportManager>(fakeCommA));
    fakeCommA.GetMemTransportManager()->oneSidedMap[linkData1] = std::move(transport);
    MOCKER_CPP(&MemTransportManager::IsAllOneSidedTransportReady).stubs().will(returnValue(true));
    shared_ptr<Socket> fakeSocketPtr
        = make_shared<Socket>(nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    SocketConfig socketConfig(linkData1.GetRemoteRankId(), linkData1, fakeCommA.GetEstablishLinkSocketTag());
    fakeCommA.GetSocketManager().connectedSocketMap[socketConfig] = std::move(fakeSocketPtr);

    SocketStatus fakeSocketStatus = SocketStatus::OK;
    MOCKER_CPP(&Socket::GetStatus).stubs().will(returnValue(fakeSocketStatus));
    MOCKER(HrtRaSocketBlockRecv).stubs().will(returnValue(true));
    MOCKER(HrtRaSocketBlockSend).stubs().will(returnValue(true));
    oneSidedServiceA.RegMem(a, sizeof(int), HcclMemType::HCCL_MEM_TYPE_DEVICE, RankIdB, MemDescA);
    HcclMemDescs localMemDescs;
    localMemDescs.array = &MemDescA;
    localMemDescs.arrayLength = 1;

    HcclMemDescs remoteMemDescs;
    u32 actualNumOfB = 0;

    oneSidedServiceA.ExchangeMemDesc(RankIdB, localMemDescs, remoteMemDescs, actualNumOfB);

    RmaMemDesc* RmaMemDescA = static_cast<RmaMemDesc*>(static_cast<void*>(MemDescA.desc));
    RmaMemDescA->localRankId = RankIdB;
    HcclMem mem;
    oneSidedServiceA.EnableMemAccess(MemDescA, mem);

    HcclOneSideOpDesc opDesc;
    opDesc.localAddr = a;
    opDesc.remoteAddr = a;
    opDesc.count = 1;
    opDesc.dataType = HCCL_DATA_TYPE_INT32;
    u32 descNum = 1;

    rtStream_t stream = nullptr;
    fakeCommA.hostDeviceSyncNotifyManager = std::make_unique<HostDeviceSyncNotifyManager>();

    fakeCommA.InitStreamManager();
    MOCKER_CPP(&CommunicatorImpl::CovertToCurrentCollOperator).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&AicpuStreamManager::GetPackedData).stubs().will(returnValue(std::vector<char>{'1', '2'}));
    MOCKER_CPP(&QueueNotifyManager::GetPackedData).stubs().will(returnValue(std::vector<char>{'1', '2'}));
    MOCKER_CPP(&QueueWaitGroupCntNotifyManager::GetPackedData).stubs().will(returnValue(std::vector<char>{'1', '2'}));
    MOCKER_CPP(&QueueBcastPostCntNotifyManager::GetPackedData).stubs().will(returnValue(std::vector<char>{'1', '2'}));
    MOCKER_CPP(&HostDeviceSyncNotifyManager::GetPackedData).stubs().will(returnValue(std::vector<char>{'1', '2'}));
    fakeCommA.rankGraph = std::make_unique<RankGraph>(0);

    u32 h2dBufferSize = sizeof(KfcCommand);
    u32 d2hBufferSize = sizeof(KfcExecStatus);
    fakeCommA.kfcControlTransferH2D = std::make_unique<HDCommunicate>(0, HCCL_HDC_TYPE_H2D, h2dBufferSize);
    fakeCommA.kfcStatusTransferD2H = std::make_unique<HDCommunicate>(0, HCCL_HDC_TYPE_D2H, d2hBufferSize);

    uintptr_t fakeAddr = 1000;
    size_t fakeSize = 1000;
    fakeCommA.kfcControlTransferH2D->hostMem = std::make_unique<HostBuffer>(fakeAddr, fakeSize);
    fakeCommA.kfcControlTransferH2D->devMem = std::make_unique<DevBuffer>(fakeAddr, fakeSize);
    fakeCommA.kfcStatusTransferD2H->hostMem = std::make_unique<HostBuffer>(fakeAddr, fakeSize);
    fakeCommA.kfcStatusTransferD2H->devMem = std::make_unique<DevBuffer>(fakeAddr, fakeSize);
    oneSidedServiceA.devBatchPutGetLocalBufs = DevBuffer::Create(fakeAddr, fakeSize);
    oneSidedServiceA.devBatchPutGetRemoteBufs = DevBuffer::Create(fakeAddr, fakeSize);
    EXPECT_EQ(oneSidedServiceA.BatchGet(RankIdB, &opDesc, descNum, stream), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(oneSidedServiceA.BatchPut(RankIdB, &opDesc, descNum, stream), HcclResult::HCCL_SUCCESS);

    oneSidedServiceA.DisableMemAccess(MemDescA);
    oneSidedServiceA.DeregMem(MemDescA);
    delete a;
}

TEST_F(HcclOneSidedServiceTest, Ut_ExchangeMemDesc_When_AccStateNotAicpu_Expect_ReturnNotSupport)
{
    CommunicatorImpl com;
    com.commExecuteConfig.accState = AcceleratorState::CCU_MS;
    HcclOneSidedService service(com);
    HcclMemDesc localDesc{};
    HcclMemDescs localMemDescs{&localDesc, 1};
    HcclMemDescs remoteMemDescs{nullptr, 0};
    u32 actualNum = 0;
    HcclResult ret = service.ExchangeMemDesc(0, localMemDescs, remoteMemDescs, actualNum);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(HcclOneSidedServiceTest, Ut_ExchangeMemDesc_When_CheckLinkFail_Expect_ReturnError)
{
    StubCommunicatorImplTransMgr fakeComm;
    fakeComm.commExecuteConfig.accState = AcceleratorState::AICPU_TS;
    HcclOneSidedService service(fakeComm);
    LinkData badLink(BasePortType(PortDeploymentType::DEV_NET, ConnectProtoType::RDMA), 0, 1, 0, 1);
    service.linkDataMap_.emplace(1, badLink);

    HcclMemDesc localDesc{};
    HcclMemDescs localMemDescs{&localDesc, 1};
    HcclMemDescs remoteMemDescs{nullptr, 0};
    u32 actualNum = 0;
    HcclResult ret = service.ExchangeMemDesc(1, localMemDescs, remoteMemDescs, actualNum);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(service.oneSidedConns_.empty(), true);
}

TEST_F(HcclOneSidedServiceTest, Ut_ExchangeMemDesc_When_CreateConnectionFail_Expect_ReturnError)
{
    StubCommunicatorImplTransMgr fakeComm;
    fakeComm.commExecuteConfig.accState = AcceleratorState::AICPU_TS;
    HcclOneSidedService service(fakeComm);
    LinkData goodLink(BasePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB), 0, 1, 0, 1);
    service.linkDataMap_.emplace(1, goodLink);

    MOCKER_CPP(&HcclOneSidedService::CreateConnection).stubs().will(returnValue(HCCL_E_INTERNAL));

    HcclMemDesc localDesc{};
    HcclMemDescs localMemDescs{&localDesc, 1};
    HcclMemDescs remoteMemDescs{nullptr, 0};
    u32 actualNum = 0;
    HcclResult ret = service.ExchangeMemDesc(1, localMemDescs, remoteMemDescs, actualNum);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    EXPECT_EQ(service.oneSidedConns_.empty(), true);
}

TEST_F(HcclOneSidedServiceTest, Ut_ExchangeMemDesc_When_ConnExists_Expect_Reuse)
{
    StubCommunicatorImplTransMgr fakeComm;
    fakeComm.commExecuteConfig.accState = AcceleratorState::AICPU_TS;
    HcclOneSidedService service(fakeComm);
    LinkData link(BasePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB), 0, 1, 0, 1);
    service.linkDataMap_.emplace(1, link);
    service.oneSidedConns_.emplace(1, std::make_shared<HcclOneSidedConn>(&fakeComm, link));

    MOCKER_CPP(&HcclOneSidedConn::ExchangeMemDesc).stubs().will(returnValue(HCCL_SUCCESS));

    HcclMemDesc localDesc{};
    HcclMemDescs localMemDescs{&localDesc, 1};
    HcclMemDescs remoteMemDescs{nullptr, 0};
    u32 actualNum = 0;
    HcclResult ret = service.ExchangeMemDesc(1, localMemDescs, remoteMemDescs, actualNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(service.oneSidedConns_.size(), 1u);
}

TEST_F(HcclOneSidedServiceTest, Ut_BatchPut_When_ConnNotFound_Expect_ThrowOutOfRange)
{
    CommunicatorImpl com;
    HcclOneSidedService service(com);
    HcclOneSideOpDesc opDesc{};
    rtStream_t stream = nullptr;
    EXPECT_THROW(service.BatchPut(999, &opDesc, 1, stream), std::out_of_range);
}

TEST_F(HcclOneSidedServiceTest, Ut_BatchGet_When_ConnNotFound_Expect_ThrowOutOfRange)
{
    CommunicatorImpl com;
    HcclOneSidedService service(com);
    HcclOneSideOpDesc opDesc{};
    rtStream_t stream = nullptr;
    EXPECT_THROW(service.BatchGet(999, &opDesc, 1, stream), std::out_of_range);
}

TEST_F(HcclOneSidedServiceTest, Ut_ConcurrentReads_When_MultiReadNotFound_Expect_NoDeadlock)
{
    auto com = std::make_shared<CommunicatorImpl>();
    std::shared_ptr<HcclOneSidedService> service = std::make_shared<HcclOneSidedService>(*com);

    std::vector<std::future<void>> readers;
    for (int t = 0; t < 8; ++t) {
        readers.emplace_back(std::async(std::launch::async, [service]() {
            HcclOneSideOpDesc opDesc{};
            rtStream_t stream = nullptr;
            for (int i = 0; i < 50; ++i) {
                try {
                    service->BatchPut(999, &opDesc, 1, stream);
                } catch (const std::out_of_range&) {
                }
            }
        }));
    }
    for (auto& f : readers) {
        EXPECT_EQ(f.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    }
}

TEST_F(HcclOneSidedServiceTest, Ut_ConcurrentReadWhileExchangeMemDesc_When_MixedReadWrite_Expect_NoDeadlock)
{
    auto fakeComm = std::make_shared<StubCommunicatorImplTransMgr>();
    fakeComm->commExecuteConfig.accState = AcceleratorState::AICPU_TS;
    std::shared_ptr<HcclOneSidedService> service = std::make_shared<HcclOneSidedService>(*fakeComm);
    LinkData badLink(BasePortType(PortDeploymentType::DEV_NET, ConnectProtoType::RDMA), 0, 1, 0, 1);
    service->linkDataMap_.emplace(1, badLink);

    auto writer = std::async(std::launch::async, [service, fakeComm]() {
        HcclMemDesc localDesc{};
        HcclMemDescs localMemDescs{&localDesc, 1};
        HcclMemDescs remoteMemDescs{nullptr, 0};
        u32 actualNum = 0;
        for (int i = 0; i < 50; ++i) {
            (void)service->ExchangeMemDesc(1, localMemDescs, remoteMemDescs, actualNum);
        }
    });
    std::vector<std::future<void>> readers;
    for (int t = 0; t < 8; ++t) {
        readers.emplace_back(std::async(std::launch::async, [service]() {
            HcclOneSideOpDesc opDesc{};
            rtStream_t stream = nullptr;
            for (int i = 0; i < 50; ++i) {
                try {
                    service->BatchPut(999, &opDesc, 1, stream);
                } catch (const std::out_of_range&) {
                }
            }
        }));
    }

    EXPECT_EQ(writer.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    for (auto& f : readers) {
        EXPECT_EQ(f.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    }
}

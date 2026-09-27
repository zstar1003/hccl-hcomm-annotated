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
#include "aicpu_kernel_launcher.h"
#undef private
#undef protected

using namespace Hccl;
using namespace std;

class AicpuKernelLauncherTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CommunicatorImplTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "CommunicatorImplTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in CommunicatorImplTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CommunicatorImplTest TearDown" << std::endl;
    }
};

static void SetupAicpuKernelLauncherTestEnv(CommunicatorImpl& comm, bool needMemcpyMock = false)
{
    u32 fakeDevPhyId = 1;
    u64 fakeNotifyHandleAddr = 100;
    u32 fakeNotifyId = 1;
    u64 fakeOffset = 200;
    u64 fakeAddress = 300;
    u32 fakePid = 100;
    char fakeName[65] = "testRtsNotify";
    SETUP_COMM_NOTIFY_MOCKS();
    std::pair<u32, u32> pair(0, 1);
    MOCKER(HrtUbDevQueryToken).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(pair));
    void* temp = nullptr;
    MOCKER(HrtStreamCreateWithFlags).stubs().will(returnValue(temp));
    MOCKER(HrtGetStreamId).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().will(returnValue(static_cast<DevId>(1)));
    if (needMemcpyMock) {
        MOCKER(HrtMemcpy).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any());
    }

    comm.InitNotifyManager();
    comm.InitSocketManager();
    comm.InitRmaConnManager();
    comm.InitStreamManager();
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
    comm.currentCollOperator->opTag = "test";
    comm.currentCollOperator->inputMem = DevBuffer::Create(0x100, 10);
    comm.currentCollOperator->outputMem = DevBuffer::Create(0x100, 10);
    s32 rankId = 0;
    s32 localId = 0;
    DeviceId deviceId = 0;
    IpAddress inputAddr(0);
    std::set<string> ports = {"0/1"};
    shared_ptr<NetInstance::Peer> peer0 = std::make_shared<NetInstance::Peer>(rankId, localId, localId, deviceId);
    std::set<LinkProtocol> protocols = {LinkProtocol::UB_CTP};
    shared_ptr<NetInstance::ConnInterface> connInterface = std::make_shared<NetInstance::ConnInterface>(
        inputAddr, ports, AddrPosition::HOST, LinkType::PEER2PEER, protocols);
    peer0->AddConnInterface(connInterface);
    comm.rankGraph->AddPeer(peer0);
    comm.localRmaBufManager = std::make_unique<LocalRmaBufManager>(comm);
    comm.cclBuffer = DevBuffer::Create(0x100, 10);
    CollServiceDeviceMode collService{&comm};
    comm.collService = &collService;
    comm.collService->counterBuf = DevBuffer::Create(0x100, 10);
    comm.aicpuStreamManager->AllocFreeStream();
}

TEST(AicpuKernelLauncherTest, test_SetHcclKernelLaunchParam_offload)
{
    CommunicatorImpl comm;
    SetupAicpuKernelLauncherTestEnv(comm);

    comm.currentCollOperator->opMode = OpMode::OFFLOAD;
    AicpuKernelLauncher aicpuKernelLauncher(comm);
    HcclKernelLaunchParam param;
    EXPECT_NO_THROW(aicpuKernelLauncher.SetHcclKernelLaunchParam(param));
}

TEST(AicpuKernelLauncherTest, test_SetHcclKernelLaunchParam_opbase)
{
    GlobalMockObject::verify();
    CommunicatorImpl comm;
    SetupAicpuKernelLauncherTestEnv(comm, true);

    comm.currentCollOperator->opMode = OpMode::OPBASE;
    AicpuKernelLauncher aicpuKernelLauncher(comm);
    HcclKernelLaunchParam param;
    EXPECT_NO_THROW(aicpuKernelLauncher.SetHcclKernelLaunchParam(param));
}

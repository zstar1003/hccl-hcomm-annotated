/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include "hcomm_c_adpt.h"
#include "acl/acl_rt.h"
#define private public
#define protected public
#include "server_socket/server_socket_manager.h"
#include "server_socket/server_socket_mgr.h"
#include "socket_mgr.h"
#include "hccp_hdc_manager.h"
#include "rdma_handle_manager.h"
#include "socket_handle_manager.h"
#include "hcomm_res_mgr.h"
#undef protected
#undef private

class DeviceResetCallbackTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DeviceResetCallbackTest set up." << std::endl; }
    static void TearDownTestCase() { std::cout << "DeviceResetCallbackTest tear down." << std::endl; }
    virtual void SetUp() {}
    virtual void TearDown() { GlobalMockObject::verify(); }
};

TEST_F(DeviceResetCallbackTest, Ut_When_RegisterDeviceResetCallback_Expect_AclrtRegDeviceStateCallbackCalled)
{
    aclError ret = aclrtRegDeviceStateCallback("hcomm_res_mgr_test", nullptr, nullptr);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

TEST_F(DeviceResetCallbackTest, Ut_When_RdmaHandleManagerDeInit_Expect_NoCrash)
{
    Hccl::RdmaHandleManager::GetInstance().DeInit(0);
}

TEST_F(DeviceResetCallbackTest, Ut_When_RdmaHandleManagerDeInit_OutOfRange_Expect_NoCrash)
{
    Hccl::RdmaHandleManager::GetInstance().DeInit(999);
}

TEST_F(DeviceResetCallbackTest, Ut_When_SocketHandleManagerDeInit_Expect_NoCrash)
{
    Hccl::SocketHandleManager::GetInstance().DeInit(0);
}

TEST_F(DeviceResetCallbackTest, Ut_When_SocketHandleManagerDeInit_OutOfRange_Expect_NoCrash)
{
    Hccl::SocketHandleManager::GetInstance().DeInit(999);
}

TEST_F(DeviceResetCallbackTest, Ut_When_HccpHdcManagerDeInit_Expect_NoCrash)
{
    Hccl::HccpHdcManager::GetInstance().DeInit(0);
}

TEST_F(DeviceResetCallbackTest, Ut_When_HccpHdcManagerDeInit_NotInited_Expect_NoCrash)
{
    Hccl::HccpHdcManager::GetInstance().DeInit(5);
}

TEST_F(DeviceResetCallbackTest, Ut_When_HccpHdcManagerInitThenDeInit_Expect_InstanceRemoved)
{
    Hccl::HccpHdcManager::GetInstance().Init(0);
    Hccl::HccpHdcManager::GetInstance().DeInit(0);
}

TEST_F(DeviceResetCallbackTest, Ut_When_SocketMgrDeInit_Expect_NoCrash) { hcomm::SocketMgr::DeInit(0); }

TEST_F(DeviceResetCallbackTest, Ut_When_ServerSocketMgrDeInit_Expect_NoCrash) { hcomm::ServerSocketMgr::DeInit(0); }

TEST_F(DeviceResetCallbackTest, Ut_When_ServerSocketManagerDeInit_Expect_NoCrash)
{
    hcomm::ServerSocketManager::GetInstance().DeInit(0);
}

TEST_F(DeviceResetCallbackTest, Ut_When_ServerSocketManagerDeInit_WithDeviceSockets_Expect_Clear)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::UB);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    uint32_t port = 43210;
    HcclResult ret = hcomm::ServerSocketManager::GetInstance().ServerSocketStartListen(
        localPort, Hccl::NicType::DEVICE_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    hcomm::ServerSocketManager::GetInstance().DeInit(0);
    EXPECT_EQ(hcomm::ServerSocketManager::GetInstance().deviceServerSocketMap_.size(), 0);
}

TEST_F(DeviceResetCallbackTest, Ut_When_ServerSocketManagerDeInit_WithHostSockets_Expect_Clear)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType type = Hccl::DevNetPortType(Hccl::ConnectProtoType::RDMA);
    Hccl::PortData localPort = Hccl::PortData(0, type, 0, ipAddr);
    uint32_t port = 0;
    HcclResult ret = hcomm::ServerSocketManager::GetInstance().ServerSocketStartListen(
        localPort, Hccl::NicType::HOST_NIC_TYPE, 0, &port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(port, 0);
    hcomm::ServerSocketManager::GetInstance().DeInit(0);
    EXPECT_EQ(hcomm::ServerSocketManager::GetInstance().hostServerSocketMap_.size(), 0);
}

TEST_F(DeviceResetCallbackTest, Ut_When_ServerSocketManagerDeInit_WithBothSockets_Expect_Clear)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType typeUb = Hccl::DevNetPortType(Hccl::ConnectProtoType::UB);
    Hccl::PortData localPortUb = Hccl::PortData(0, typeUb, 0, ipAddr);
    uint32_t port1 = 43210;
    HcclResult ret1 = hcomm::ServerSocketManager::GetInstance().ServerSocketStartListen(
        localPortUb, Hccl::NicType::DEVICE_NIC_TYPE, 0, &port1);
    EXPECT_EQ(ret1, HCCL_SUCCESS);

    Hccl::DevNetPortType typeRdma = Hccl::DevNetPortType(Hccl::ConnectProtoType::RDMA);
    Hccl::PortData localPortRdma = Hccl::PortData(0, typeRdma, 0, ipAddr);
    uint32_t port2 = 0;
    HcclResult ret2 = hcomm::ServerSocketManager::GetInstance().ServerSocketStartListen(
        localPortRdma, Hccl::NicType::HOST_NIC_TYPE, 0, &port2);
    EXPECT_EQ(ret2, HCCL_SUCCESS);
    EXPECT_NE(port2, 0);

    hcomm::ServerSocketManager::GetInstance().DeInit(0);
    EXPECT_EQ(hcomm::ServerSocketManager::GetInstance().deviceServerSocketMap_.size(), 0);
    EXPECT_EQ(hcomm::ServerSocketManager::GetInstance().hostServerSocketMap_.size(), 0);
}

TEST_F(DeviceResetCallbackTest, Ut_When_AllDeInitCalled_SimulateDeviceResetPre_Expect_NoCrash)
{
    u32 devPhyId = 0;
    s32 deviceId = 0;
    hcomm::SocketMgr::DeInit(devPhyId);
    hcomm::ServerSocketMgr::DeInit(devPhyId);
    hcomm::ServerSocketManager::GetInstance().DeInit(devPhyId);
    Hccl::RdmaHandleManager::GetInstance().DeInit(devPhyId);
    Hccl::SocketHandleManager::GetInstance().DeInit(devPhyId);
    Hccl::HccpHdcManager::GetInstance().DeInit(deviceId);
}

TEST_F(DeviceResetCallbackTest, Ut_When_AllDeInitCalled_WithNonZeroDevPhyId_Expect_NoCrash)
{
    u32 devPhyId = 1;
    s32 deviceId = 1;
    hcomm::SocketMgr::DeInit(devPhyId);
    hcomm::ServerSocketMgr::DeInit(devPhyId);
    hcomm::ServerSocketManager::GetInstance().DeInit(devPhyId);
    Hccl::RdmaHandleManager::GetInstance().DeInit(devPhyId);
    Hccl::SocketHandleManager::GetInstance().DeInit(devPhyId);
    Hccl::HccpHdcManager::GetInstance().DeInit(deviceId);
}

TEST_F(DeviceResetCallbackTest, Ut_When_HcommResMgrInit_Expect_Success)
{
    HcommResult ret = HcommResMgrInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(DeviceResetCallbackTest, Ut_When_RdmaHandleManagerDeInit_WithDestroyed_Expect_NoCrash)
{
    Hccl::RdmaHandleManager::GetInstance().destroyed.store(true);
    Hccl::RdmaHandleManager::GetInstance().DeInit(0);
    Hccl::RdmaHandleManager::GetInstance().destroyed.store(false);
}

TEST_F(DeviceResetCallbackTest, Ut_When_HccpHdcManagerDeInit_WithDestroyed_Expect_NoCrash)
{
    Hccl::HccpHdcManager::GetInstance().destroyed.store(true);
    Hccl::HccpHdcManager::GetInstance().DeInit(0);
    Hccl::HccpHdcManager::GetInstance().destroyed.store(false);
}

TEST_F(DeviceResetCallbackTest, Ut_When_ServerSocketManagerDeInit_SelectiveDevPhyId_Expect_OnlyMatchingCleared)
{
    Hccl::IpAddress ipAddr("1.0.0.0");
    Hccl::DevNetPortType typeUb = Hccl::DevNetPortType(Hccl::ConnectProtoType::UB);
    Hccl::PortData localPort0 = Hccl::PortData(0, typeUb, 0, ipAddr);
    uint32_t port0 = 43210;
    HcclResult ret0 = hcomm::ServerSocketManager::GetInstance().ServerSocketStartListen(
        localPort0, Hccl::NicType::DEVICE_NIC_TYPE, 0, &port0);
    EXPECT_EQ(ret0, HCCL_SUCCESS);

    hcomm::ServerSocketManager::GetInstance().DeInit(999);
    EXPECT_NE(hcomm::ServerSocketManager::GetInstance().deviceServerSocketMap_.size(), 0);

    hcomm::ServerSocketManager::GetInstance().DeInit(0);
    EXPECT_EQ(hcomm::ServerSocketManager::GetInstance().deviceServerSocketMap_.size(), 0);
}

TEST_F(DeviceResetCallbackTest, Ut_When_RdmaHandleManagerDeInit_WithRdmaHandle_Expect_CleanupRdmaEntry)
{
    auto& mgr = Hccl::RdmaHandleManager::GetInstance();
    u32 devPhyId = 0;
    RdmaHandle fakeRdmaHandle = (RdmaHandle)0xAAAA;
    Hccl::IpAddress ipAddr("1.0.0.0");
    mgr.rdmaHandleMap[devPhyId][2][ipAddr] = fakeRdmaHandle;
    mgr.netWorkModeMap[fakeRdmaHandle] = Hccl::HrtNetworkMode::HDC;
    mgr.DieAndFuncIdMap[fakeRdmaHandle] = {0, 0};
    mgr.RtpEnableMap[fakeRdmaHandle] = true;

    mgr.DeInit(devPhyId);
    EXPECT_EQ(mgr.rdmaHandleMap[devPhyId][2].size(), 0);
    EXPECT_EQ(mgr.netWorkModeMap.count(fakeRdmaHandle), 0);
    EXPECT_EQ(mgr.DieAndFuncIdMap.count(fakeRdmaHandle), 0);
    EXPECT_EQ(mgr.RtpEnableMap.count(fakeRdmaHandle), 0);
}

TEST_F(DeviceResetCallbackTest, Ut_When_RdmaHandleManagerDeInit_WithUbHandle_Expect_CleanupUbEntry)
{
    auto& mgr = Hccl::RdmaHandleManager::GetInstance();
    u32 devPhyId = 1;
    RdmaHandle fakeUbHandle = (RdmaHandle)0xBBBB;
    Hccl::IpAddress ipAddr("2.0.0.0");
    mgr.rdmaHandleMap[devPhyId][3][ipAddr] = fakeUbHandle;
    mgr.tokenInfoMap[fakeUbHandle] = nullptr;

    mgr.DeInit(devPhyId);
    EXPECT_EQ(mgr.rdmaHandleMap[devPhyId][3].size(), 0);
    EXPECT_EQ(mgr.tokenInfoMap.count(fakeUbHandle), 0);
}

TEST_F(DeviceResetCallbackTest, Ut_When_RdmaHandleManagerDeInit_WithUbHandleAndTokenInfo_Expect_CleanupTokenAndUb)
{
    auto& mgr = Hccl::RdmaHandleManager::GetInstance();
    u32 devPhyId = 2;
    RdmaHandle fakeUbHandle = (RdmaHandle)0xCCCC;
    Hccl::IpAddress ipAddr("3.0.0.0");
    mgr.rdmaHandleMap[devPhyId][3][ipAddr] = fakeUbHandle;
    mgr.tokenInfoMap[fakeUbHandle] = std::make_unique<Hccl::TokenInfoManager>(0, fakeUbHandle);
    mgr.DieAndFuncIdMap[fakeUbHandle] = {1, 2};
    mgr.RtpEnableMap[fakeUbHandle] = false;

    mgr.DeInit(devPhyId);
    EXPECT_EQ(mgr.rdmaHandleMap[devPhyId][3].size(), 0);
    EXPECT_EQ(mgr.tokenInfoMap.count(fakeUbHandle), 0);
    EXPECT_EQ(mgr.DieAndFuncIdMap.count(fakeUbHandle), 0);
    EXPECT_EQ(mgr.RtpEnableMap.count(fakeUbHandle), 0);
}

TEST_F(DeviceResetCallbackTest, Ut_When_RdmaHandleManagerDeInit_WithJfcHandle_Expect_CleanupJfcEntry)
{
    auto& mgr = Hccl::RdmaHandleManager::GetInstance();
    u32 devPhyId = 0;
    RdmaHandle fakeRdmaHandle = (RdmaHandle)0xDDDD;
    Hccl::IpAddress ipAddr("4.0.0.0");
    mgr.rdmaHandleMap[devPhyId][2][ipAddr] = fakeRdmaHandle;
    mgr.netWorkModeMap[fakeRdmaHandle] = Hccl::HrtNetworkMode::HDC;
    mgr.jfcHandleMap[fakeRdmaHandle][Hccl::HrtUbJfcMode::NORMAL] = (Hccl::JfcHandle)0x1111;

    mgr.DeInit(devPhyId);
    EXPECT_EQ(mgr.rdmaHandleMap[devPhyId][2].size(), 0);
    EXPECT_EQ(mgr.jfcHandleMap.count(fakeRdmaHandle), 0);
    EXPECT_EQ(mgr.netWorkModeMap.count(fakeRdmaHandle), 0);
}

TEST_F(DeviceResetCallbackTest, Ut_When_RdmaHandleManagerDeInit_WithMixedHandles_Expect_AllCleaned)
{
    auto& mgr = Hccl::RdmaHandleManager::GetInstance();
    // 隔离历史残留：前置用例经 mock handle（未登记 rdmaHandleMap，DeInit 收集不到）真实调用
    // GetJfcHandle/GetDieAndFuncId 会在 per-handle map 留下条目；本用例对这三个 map 做 size 全量断言，先清空
    mgr.jfcHandleMap.clear();
    mgr.DieAndFuncIdMap.clear();
    mgr.RtpEnableMap.clear();
    u32 devPhyId = 0;
    RdmaHandle fakeRdmaHandle = (RdmaHandle)0xEEEE;
    RdmaHandle fakeUbHandle = (RdmaHandle)0xFFFF;
    Hccl::IpAddress ipRdma("5.0.0.0");
    Hccl::IpAddress ipUb("6.0.0.0");

    mgr.rdmaHandleMap[devPhyId][2][ipRdma] = fakeRdmaHandle;
    mgr.rdmaHandleMap[devPhyId][3][ipUb] = fakeUbHandle;
    mgr.netWorkModeMap[fakeRdmaHandle] = Hccl::HrtNetworkMode::HDC;
    mgr.jfcHandleMap[fakeRdmaHandle][Hccl::HrtUbJfcMode::NORMAL] = (Hccl::JfcHandle)0x2222;
    mgr.jfcHandleMap[fakeUbHandle][Hccl::HrtUbJfcMode::STARS_POLL] = (Hccl::JfcHandle)0x3333;
    mgr.tokenInfoMap[fakeUbHandle] = nullptr;
    mgr.DieAndFuncIdMap[fakeRdmaHandle] = {3, 4};
    mgr.DieAndFuncIdMap[fakeUbHandle] = {5, 6};
    mgr.RtpEnableMap[fakeRdmaHandle] = true;
    mgr.RtpEnableMap[fakeUbHandle] = false;

    mgr.DeInit(devPhyId);
    EXPECT_EQ(mgr.rdmaHandleMap[devPhyId][2].size(), 0);
    EXPECT_EQ(mgr.rdmaHandleMap[devPhyId][3].size(), 0);
    EXPECT_EQ(mgr.jfcHandleMap.size(), 0);
    EXPECT_EQ(mgr.netWorkModeMap.count(fakeRdmaHandle), 0);
    EXPECT_EQ(mgr.tokenInfoMap.count(fakeUbHandle), 0);
    EXPECT_EQ(mgr.DieAndFuncIdMap.size(), 0);
    EXPECT_EQ(mgr.RtpEnableMap.size(), 0);
}

TEST_F(DeviceResetCallbackTest, Ut_When_RdmaHandleManagerDestroyAll_WithDestroyed_Expect_NoCrash)
{
    auto& mgr = Hccl::RdmaHandleManager::GetInstance();
    mgr.destroyed.store(true);
    mgr.DestroyAll();
    mgr.destroyed.store(false);
}

TEST_F(DeviceResetCallbackTest, Ut_When_IsHandleValid_NullHandle_Expect_False)
{
    EXPECT_FALSE(Hccl::RdmaHandleManager::GetInstance().IsHandleValid(nullptr));
}

TEST_F(DeviceResetCallbackTest, Ut_When_IsHandleValid_HandleNotInActiveHandles_Expect_False)
{
    Hccl::RdmaHandle fakeHandle = (Hccl::RdmaHandle)0x9999;
    EXPECT_FALSE(Hccl::RdmaHandleManager::GetInstance().IsHandleValid(fakeHandle));
}

TEST_F(DeviceResetCallbackTest, Ut_When_IsHandleValid_HandleInActiveHandles_Expect_True)
{
    auto& mgr = Hccl::RdmaHandleManager::GetInstance();
    Hccl::RdmaHandle fakeHandle = (Hccl::RdmaHandle)0x8887;
    mgr.activeHandles_.insert(fakeHandle);
    EXPECT_TRUE(mgr.IsHandleValid(fakeHandle));
    mgr.activeHandles_.erase(fakeHandle);
}

TEST_F(DeviceResetCallbackTest, Ut_When_IsHandleValid_WithDestroyed_Expect_False)
{
    auto& mgr = Hccl::RdmaHandleManager::GetInstance();
    Hccl::RdmaHandle fakeHandle = (Hccl::RdmaHandle)0x8889;
    mgr.activeHandles_.insert(fakeHandle);
    EXPECT_TRUE(mgr.IsHandleValid(fakeHandle));
    mgr.destroyed.store(true);
    EXPECT_FALSE(mgr.IsHandleValid(fakeHandle));
    mgr.destroyed.store(false);
    mgr.activeHandles_.erase(fakeHandle);
}

TEST_F(DeviceResetCallbackTest, Ut_When_RdmaHandleManagerDeInit_Expect_ActiveHandlesRemovedBeforeDestroy)
{
    auto& mgr = Hccl::RdmaHandleManager::GetInstance();
    u32 devPhyId = 0;
    Hccl::RdmaHandle fakeUbHandle = (Hccl::RdmaHandle)0xABCD;
    Hccl::IpAddress ipAddr("7.0.0.0");
    mgr.rdmaHandleMap[devPhyId][3][ipAddr] = fakeUbHandle;
    mgr.activeHandles_.insert(fakeUbHandle);

    EXPECT_TRUE(mgr.IsHandleValid(fakeUbHandle));

    mgr.DeInit(devPhyId);

    EXPECT_FALSE(mgr.IsHandleValid(fakeUbHandle));
    EXPECT_EQ(mgr.activeHandles_.count(fakeUbHandle), 0);
}

TEST_F(DeviceResetCallbackTest, Ut_When_RdmaHandleManagerDeInit_WithMultipleHandles_Expect_AllRemovedFromActiveHandles)
{
    auto& mgr = Hccl::RdmaHandleManager::GetInstance();
    u32 devPhyId = 3;
    Hccl::RdmaHandle handle1 = (Hccl::RdmaHandle)0x1111;
    Hccl::RdmaHandle handle2 = (Hccl::RdmaHandle)0x2222;
    Hccl::IpAddress ip1("8.0.0.0");
    Hccl::IpAddress ip2("9.0.0.0");
    mgr.rdmaHandleMap[devPhyId][3][ip1] = handle1;
    mgr.rdmaHandleMap[devPhyId][3][ip2] = handle2;
    mgr.activeHandles_.insert(handle1);
    mgr.activeHandles_.insert(handle2);

    EXPECT_TRUE(mgr.IsHandleValid(handle1));
    EXPECT_TRUE(mgr.IsHandleValid(handle2));

    mgr.DeInit(devPhyId);

    EXPECT_FALSE(mgr.IsHandleValid(handle1));
    EXPECT_FALSE(mgr.IsHandleValid(handle2));
}

TEST_F(DeviceResetCallbackTest, Ut_When_DeInitDifferentDevPhyId_Expect_OnlyTargetedHandlesRemoved)
{
    auto& mgr = Hccl::RdmaHandleManager::GetInstance();
    Hccl::RdmaHandle handleDev0 = (Hccl::RdmaHandle)0xA0A0;
    Hccl::RdmaHandle handleDev1 = (Hccl::RdmaHandle)0xB1B1;
    Hccl::IpAddress ip0("10.0.0.0");
    Hccl::IpAddress ip1("11.0.0.0");
    mgr.rdmaHandleMap[0][3][ip0] = handleDev0;
    mgr.rdmaHandleMap[1][3][ip1] = handleDev1;
    mgr.activeHandles_.insert(handleDev0);
    mgr.activeHandles_.insert(handleDev1);

    mgr.DeInit(0);

    EXPECT_FALSE(mgr.IsHandleValid(handleDev0));
    EXPECT_TRUE(mgr.IsHandleValid(handleDev1));
    mgr.activeHandles_.erase(handleDev1);
    mgr.rdmaHandleMap[1][3].erase(ip1);
}

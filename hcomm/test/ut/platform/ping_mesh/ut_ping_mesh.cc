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
#include <mockcpp/mockcpp.hpp>
#include <thread>
#include <chrono>
#include "securec.h"
#define private public
#include "hccl_network_pub.h"
#include "network_manager_pub.h"
#include "ping_mesh.h"
#undef private
#include "adapter_hccp.h"
#include "adapter_tdt.h"
#include "adapter_hal.h"
#include "adapter_rts.h"
#include "adapter_rts_common.h"
#include "hdc_pub.h"
#include "profiler_base_pub.h"
#include "tsd/tsd_client.h"
#include "network/hccp_common.h"
#include "network/hccp_ping.h"
#include "externalinput.h"

using namespace hccl;

// 打桩函数
inline HcclResult hrtGetDeviceStub(s32* deviceLogicId)
{
    *deviceLogicId = 1;
    return HCCL_SUCCESS;
}

inline HcclResult hrtGetDevicePhyIdByIndexStub(u32 deviceLogicId, u32& devicePhyId, bool isRefresh = false)
{
    devicePhyId = deviceLogicId;
    return HCCL_SUCCESS;
}

inline HcclResult hrtTsdProcessOpenStub(u32 deviceLogicId, ProcOpenArgs* openArgs)
{
    // *(openArgs->subPid) = 1;
    return HCCL_SUCCESS;
}

inline HcclResult
hrtRaGetInterfaceVersionStub(unsigned int phyId, unsigned int interfaceOpcode, unsigned int* interfaceVersion)
{
    *interfaceVersion = 1;
    return HCCL_SUCCESS;
}

inline HcclResult hrtRaPingInitStub(struct PingInitAttr* initAttr, struct PingInitInfo* initInfo, void** pingHandle)
{
    std::shared_ptr<PingMesh> pingMesh;
    pingMesh.reset(new (std::nothrow) PingMesh());
    *pingHandle = pingMesh.get();
    return HCCL_SUCCESS;
}

inline HcclResult hrtRaPingDeinitStub(void* pingHandle) { return HCCL_SUCCESS; }

inline HcclResult hrtRaPingTargetAddStub(void* pingHandle, struct PingTargetInfo target[], uint32_t num)
{
    return HCCL_SUCCESS;
}

inline HcclResult hrtRaPingTargetDelStub(void* pingHandle, struct PingTargetInfo target[], uint32_t num)
{
    return HCCL_SUCCESS;
}

inline HcclResult hrtRaPingTaskStartStub(void* pingHandle, struct PingTaskAttr* attr) { return HCCL_SUCCESS; }

inline HcclResult hrtRaPingTaskStopStub(void* pingHandle) { return HCCL_SUCCESS; }

inline HcclResult hrtRaPingGetResultsStub(void* pingHandle, struct PingTargetResult output[], uint32_t* num)
{
    char* payload = "pingmesh";

    for (int i = 0; i < *num; i++) {
        HcclIpAddress targetIp = HcclIpAddress(0x7F000002 + i * 3);
        output[i].result.state = static_cast<PingResultState>(i % 3);
        output[i].result.summary.rttAvg = 10;
        output[i].result.summary.rttMax = 35;
        output[i].result.summary.rttMin = 5;
        output[i].result.summary.sendCnt = 2;
        output[i].result.summary.recvCnt = 0;

        output[i].remoteInfo.qpInfo = {0};
        output[i].remoteInfo.ip.addr.s_addr = targetIp.GetBinaryAddress().addr.s_addr;
    }
    return HCCL_SUCCESS;
}

class PingMesh_UT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "PingMesh_UT SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "PingMesh_UT TearDown" << std::endl; }

    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        MOCKER(hrtGetDevice).stubs().with(mockcpp::any()).will(invoke(hrtGetDeviceStub));

        MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

        MOCKER(TsdProcessOpen).stubs().with(mockcpp::any()).will(invoke(hrtTsdProcessOpenStub));

        MOCKER(ProcessCloseSubProcList).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

        MOCKER(hrtRaGetInterfaceVersion).stubs().with(mockcpp::any()).will(invoke(hrtRaGetInterfaceVersionStub));

        void* ptr = (void*)0xabcd;
        MOCKER(hrtRaPingInit)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), outBoundP(&ptr, sizeof(ptr)))
            .will(returnValue(HCCL_SUCCESS));

        MOCKER(hrtRaPingDeinit).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

        MOCKER(hrtRaPingTargetDel).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

        MOCKER(hrtRaPingTaskStart).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

        MOCKER(hrtRaPingTaskStop).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

        MOCKER(hrtRaPingGetResults).stubs().with(mockcpp::any()).will(invoke(hrtRaPingGetResultsStub));

        MOCKER(hrtRaIsLastUsed).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

        MOCKER(HrtRaInit).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

        MOCKER(HrtRaDeInit).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

        MOCKER(hrtRaSocketInit).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

        MOCKER_CPP(&HcclSocket::Listen, HcclResult(HcclSocket::*)())
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(HCCL_SUCCESS));

        MOCKER_CPP(&HcclSocket::Listen, HcclResult(HcclSocket::*)(u32))
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(HCCL_SUCCESS));

        MOCKER_CPP(&HcclSocket::Accept).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

        MOCKER_CPP(&HcclSocket::Connect).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

        MOCKER_CPP(&HcclSocket::Send, HcclResult(HcclSocket::*)(const void*, u64))
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(HCCL_SUCCESS));

        MOCKER_CPP(
            static_cast<HcclResult (hccl::HcclSocket::*)(void*, uint32_t, uint32_t)>(&hccl::HcclSocket::Recv),
            HcclResult(hccl::HcclSocket::*)(void*, uint32_t, uint32_t))
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(HCCL_SUCCESS));
        // MOCKER_CPP(&HcclSocket::Recv, HcclResult(HcclSocket::*)(const void *, u32, u32))
        // .stubs()
        // .with(mockcpp::any())
        // .will(returnValue(HCCL_SUCCESS));

        MOCKER_CPP(&HcclSocket::DeInit).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

        MOCKER_CPP(&std::thread::detach).stubs().with(mockcpp::any()).will(ignoreReturnValue());

        MOCKER(hrtHalMemCtl).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

        std::cout << "PingMesh_ST Test SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "PingMesh_UT Test TearDown" << std::endl;
    }
};

TEST_F(PingMesh_UT, ut_PingMeshInit)
{
    u32 deviceId = 1;
    u32 mode = static_cast<u32>(LinkType::LINK_ROCE);
    HcclIpAddress ipAddr = HcclIpAddress(0x7F000001);
    u32 port = 13866;
    u32 nodeNum = 10;
    u32 bufferSize = 100U;
    u32 sl = 0;
    u32 tc = 0;

    // 初始化成功
    std::shared_ptr<PingMesh> pingMesh;
    pingMesh.reset(new (std::nothrow) PingMesh());
    auto ret = pingMesh->HccnRpingInit(deviceId, mode, ipAddr, port, nodeNum, bufferSize, sl, tc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

HcclResult stub_hrtRaGetDevEidInfoList(RaInfo info, struct HccpDevEidInfo* eid_info, unsigned int* num)
{
    memset(eid_info->eid.raw, 0, sizeof(eid_info->eid.raw));
    *num = 1;
    return HCCL_SUCCESS;
}

TEST_F(PingMesh_UT, ut_PingMeshInit_950)
{
    const char* fakeA5SocName = "Ascend950PR_958b";
    MOCKER(aclrtGetSocName).stubs().will(returnValue(fakeA5SocName));
    u32 num = 1;
    MOCKER(hrtRaGetDevEidInfoNum)
        .stubs()
        .with(mockcpp::any(), outBoundP(&num, sizeof(&num)))
        .will(returnValue(HCCL_E_NOT_SUPPORT))
        .then(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaGetSecRandom)
        .stubs()
        .with(mockcpp::any(), outBoundP(&num, sizeof(&num)))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaGetDevEidInfoList).stubs().will(invoke(stub_hrtRaGetDevEidInfoList));
    u32 deviceId = 1;
    u32 mode = static_cast<u32>(LinkType::LINK_UB);
    HcclIpAddress ipAddr = HcclIpAddress(0x7F000001);
    u32 port = 13866;
    u32 nodeNum = 10;
    u32 bufferSize = 100U;
    u32 sl = 0;
    u32 tc = 0;

    // 初始化成功
    std::shared_ptr<PingMesh> pingMesh;
    pingMesh.reset(new (std::nothrow) PingMesh());
    auto ret = pingMesh->HccnRpingInit(deviceId, mode, ipAddr, port, nodeNum, bufferSize, sl, tc);
    EXPECT_EQ(ret, HCCL_E_NETWORK);

    ret = pingMesh->HccnRpingInit(deviceId, mode, ipAddr, port, nodeNum, bufferSize, sl, tc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(PingMesh_UT, ut_PingMeshAddDelTarget)
{
    MOCKER_CPP(&PingMesh::RpingRecvTargetInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtRaPingTargetAdd).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    u32 deviceId = 1;
    u32 mode = static_cast<u32>(LinkType::LINK_ROCE);
    HcclIpAddress ipAddr = HcclIpAddress(0x7F000001);
    u32 port = 13866;
    u32 nodeNum = 10;
    u32 bufferSize = 100U;
    u32 sl = 0;
    u32 tc = 0;

    // 初始化成功
    std::shared_ptr<PingMesh> pingMesh;
    pingMesh.reset(new (std::nothrow) PingMesh());
    auto ret = pingMesh->HccnRpingInit(deviceId, mode, ipAddr, port, nodeNum, bufferSize, sl, tc);

    // 添加目标
    hccl::RpingInput target[10];
    char* payload = "pingmesh";
    for (u32 i = 0; i < 10; i++) {
        HcclIpAddress targetIp = HcclIpAddress(0x7F000002 + i);
        target[i].sip = targetIp;
        target[i].dip = targetIp;
        target[i].port = 13866;
        target[i].tc = 1;
        target[i].sl = 1;
        target[i].srcPort = 0;
        target[i].len = 9;
        target[i].addrType = 0;
        memcpy_s(target[i].payload, target[i].len, payload, target[i].len);
    }
    int i = 0;
    pingMesh->pingHandle_ = &i;
    pingMesh->rpingState_ = RpingState::INITED;
    HccnRpingAddTargetConfig config;
    config.connectTimeout = 120000;
    ret = pingMesh->HccnRpingAddTarget(deviceId, nodeNum, target, &config);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 删除目标
    void* netCtx;
    ret = HcclNetOpenDev(&netCtx, NicType::DEVICE_NIC_TYPE, deviceId, deviceId, target[i].sip);
    std::shared_ptr<HcclSocket> socket = std::make_shared<HcclSocket>(netCtx, port);
    socket->Init();
    pingMesh->socketMaps_.insert({target[i].sip.GetReadableIP(), socket});
    ret = pingMesh->HccnRpingRemoveTarget(deviceId, nodeNum, target);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
    HcclNetCloseDev(netCtx);
}

TEST_F(PingMesh_UT, ut_PingMeshAddDelTargetPatch)
{
    MOCKER_CPP(&PingMesh::RpingRecvTargetInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtRaPingTargetAdd).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    u32 deviceId = 1;
    u32 mode = static_cast<u32>(LinkType::LINK_ROCE);
    HcclIpAddress ipAddr = HcclIpAddress(0x7F000001);
    u32 port = 13866;
    u32 nodeNum = 1;
    u32 bufferSize = 100U;
    u32 sl = 0;
    u32 tc = 0;

    // 初始化成功
    std::shared_ptr<PingMesh> pingMesh;
    pingMesh.reset(new (std::nothrow) PingMesh());
    auto ret = pingMesh->HccnRpingInit(deviceId, mode, ipAddr, port, nodeNum, bufferSize, sl, tc);

    // 添加目标
    hccl::RpingInput target[1];
    char* payload = "pingmesh";
    for (u32 i = 0; i < 1; i++) {
        HcclIpAddress targetIp = HcclIpAddress(0x7F000002 + i);
        target[i].sip = targetIp;
        target[i].dip = targetIp;
        target[i].port = 13866;
        target[i].tc = 1;
        target[i].sl = 1;
        target[i].srcPort = 0;
        target[i].len = 9;
        target[i].addrType = 0;
        memcpy_s(target[i].payload, target[i].len, payload, target[i].len);
    }
    int i = 0;
    pingMesh->pingHandle_ = &i;
    pingMesh->rpingState_ = RpingState::INITED;
    HccnRpingAddTargetConfig config;
    config.connectTimeout = 120000;
    ret = pingMesh->HccnRpingAddTarget(deviceId, nodeNum, target, &config);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = pingMesh->HccnRpingAddTarget(deviceId, nodeNum, target, &config);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 删除目标
    void* netCtx;
    ret = HcclNetOpenDev(&netCtx, NicType::DEVICE_NIC_TYPE, deviceId, deviceId, target[i].sip);
    std::shared_ptr<HcclSocket> socket = std::make_shared<HcclSocket>(netCtx, port);
    socket->Init();
    pingMesh->socketMaps_.insert({target[i].sip.GetReadableIP(), socket});
    PingQpInfo rdmainfo{0};
    pingMesh->rdmaInfoMaps_.Emplace(target[i].sip.GetReadableIP(), rdmainfo);
    ret = pingMesh->HccnRpingRemoveTarget(deviceId, nodeNum, target);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HcclNetCloseDev(netCtx);
}

TEST_F(PingMesh_UT, ut_PingMeshAddDelTargetPatch1)
{
    MOCKER_CPP(&PingMesh::RpingRecvTargetInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_NOT_FOUND));

    MOCKER(hrtRaPingTargetAdd).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    u32 deviceId = 1;
    u32 mode = static_cast<u32>(LinkType::LINK_ROCE);
    HcclIpAddress ipAddr = HcclIpAddress(0x7F000001);
    u32 port = 13866;
    u32 nodeNum = 1;
    u32 bufferSize = 100U;
    u32 sl = 0;
    u32 tc = 0;

    // 初始化成功
    std::shared_ptr<PingMesh> pingMesh;
    pingMesh.reset(new (std::nothrow) PingMesh());
    auto ret = pingMesh->HccnRpingInit(deviceId, mode, ipAddr, port, nodeNum, bufferSize, sl, tc);

    // 添加目标
    hccl::RpingInput target[1];
    char* payload = "pingmesh";
    for (u32 i = 0; i < 1; i++) {
        HcclIpAddress targetIp = HcclIpAddress(0x7F000002 + i);
        target[i].sip = targetIp;
        target[i].dip = targetIp;
        target[i].port = 13866;
        target[i].tc = 1;
        target[i].sl = 1;
        target[i].srcPort = 0;
        target[i].len = 100;
        memcpy_s(target[i].payload, 9, payload, 9);
    }
    int i = 0;
    pingMesh->pingHandle_ = &i;
    pingMesh->rpingState_ = RpingState::INITED;
    HccnRpingAddTargetConfig config;
    config.connectTimeout = 120000;
    ret = pingMesh->HccnRpingAddTarget(deviceId, nodeNum, target, &config);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(PingMesh_UT, ut_PingMeshAddDelTargetPatch2)
{
    MOCKER_CPP(&PingMesh::RpingRecvTargetInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtRaPingTargetAdd).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_NOT_FOUND));

    u32 deviceId = 1;
    u32 mode = static_cast<u32>(LinkType::LINK_ROCE);
    HcclIpAddress ipAddr = HcclIpAddress(0x7F000001);
    u32 port = 13866;
    u32 nodeNum = 1;
    u32 bufferSize = 100U;
    u32 sl = 0;
    u32 tc = 0;

    // 初始化成功
    std::shared_ptr<PingMesh> pingMesh;
    pingMesh.reset(new (std::nothrow) PingMesh());
    auto ret = pingMesh->HccnRpingInit(deviceId, mode, ipAddr, port, nodeNum, bufferSize, sl, tc);

    // 添加目标
    hccl::RpingInput target[1];
    char* payload = "pingmesh";
    for (u32 i = 0; i < 1; i++) {
        HcclIpAddress targetIp = HcclIpAddress(0x7F000002 + i);
        target[i].sip = targetIp;
        target[i].dip = targetIp;
        target[i].port = 13866;
        target[i].tc = 1;
        target[i].sl = 1;
        target[i].srcPort = 0;
        target[i].len = 100;
        memcpy_s(target[i].payload, 9, payload, 9);
    }
    int i = 0;
    pingMesh->pingHandle_ = &i;
    pingMesh->rpingState_ = RpingState::INITED;
    HccnRpingAddTargetConfig config;
    config.connectTimeout = 120000;
    ret = pingMesh->HccnRpingAddTarget(deviceId, nodeNum, target, &config);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(PingMesh_UT, ut_PingMeshAddDelTargetPatch3)
{
    MOCKER_CPP(&PingMesh::RpingRecvTargetInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtRaPingTargetAdd).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(memcpy_s).stubs().with(mockcpp::any()).will(returnValue(1));

    u32 deviceId = 1;
    u32 mode = static_cast<u32>(LinkType::LINK_ROCE);
    HcclIpAddress ipAddr = HcclIpAddress(0x7F000001);
    u32 port = 13866;
    u32 nodeNum = 1;
    u32 bufferSize = 100U;
    u32 sl = 0;
    u32 tc = 0;

    // 初始化成功
    std::shared_ptr<PingMesh> pingMesh;
    pingMesh.reset(new (std::nothrow) PingMesh());
    auto ret = pingMesh->HccnRpingInit(deviceId, mode, ipAddr, port, nodeNum, bufferSize, sl, tc);

    // 添加目标
    hccl::RpingInput target[1];
    char* payload = "pingmesh";
    for (u32 i = 0; i < 1; i++) {
        HcclIpAddress targetIp = HcclIpAddress(0x7F000002 + i);
        target[i].sip = targetIp;
        target[i].dip = targetIp;
        target[i].port = 13866;
        target[i].tc = 1;
        target[i].sl = 1;
        target[i].srcPort = 0;
        target[i].len = 1;
        memcpy_s(target[i].payload, 9, payload, 9);
    }
    int i = 0;
    pingMesh->pingHandle_ = &i;
    pingMesh->rpingState_ = RpingState::INITED;
    HccnRpingAddTargetConfig config;
    config.connectTimeout = 120000;
    ret = pingMesh->HccnRpingAddTarget(deviceId, nodeNum, target, &config);
    EXPECT_EQ(ret, HCCL_E_MEMORY);
}

TEST_F(PingMesh_UT, ut_PingMeshStartStopPing)
{
    u32 deviceId = 1;
    u32 mode = static_cast<u32>(LinkType::LINK_ROCE);
    HcclIpAddress ipAddr = HcclIpAddress(0x7F000001);
    u32 port = 13866;
    u32 nodeNum = 10;
    u32 bufferSize = 100U;
    u32 sl = 0;
    u32 tc = 0;

    // 初始化成功
    std::shared_ptr<PingMesh> pingMesh;
    pingMesh.reset(new (std::nothrow) PingMesh());
    auto ret = pingMesh->HccnRpingInit(deviceId, mode, ipAddr, port, nodeNum, bufferSize, sl, tc);

    // 启动任务
    int i = 0;
    pingMesh->pingHandle_ = &i;
    pingMesh->rpingState_ = RpingState::READY;
    pingMesh->rpingTargetNum_ = 2;
    PingInitInfo& initInfo = pingMesh->initInfo_;
    initInfo.result.bufferSize = 4096 * 15;
    u32 pktNum = 10;
    u32 interval = 1;
    u32 timeout = 1000;
    ret = pingMesh->HccnRpingBatchPingStart(deviceId, pktNum, interval, timeout);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 中止任务
    ret = pingMesh->HccnRpingBatchPingStop(deviceId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(PingMesh_UT, ut_PingMeshGetResult)
{
    MOCKER(hrtMemcpyEx).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    u32 deviceId = 1;
    u32 mode = static_cast<u32>(LinkType::LINK_ROCE);
    HcclIpAddress ipAddr = HcclIpAddress(0x7F000001);
    u32 port = 13866;
    u32 nodeNum = 10;
    u32 bufferSize = 100U;
    u32 sl = 0;
    u32 tc = 0;

    // 初始化成功
    std::shared_ptr<PingMesh> pingMesh;
    pingMesh.reset(new (std::nothrow) PingMesh());
    auto ret = pingMesh->HccnRpingInit(deviceId, mode, ipAddr, port, nodeNum, bufferSize, sl, tc);

    // 获取结果
    hccl::RpingInput target[10];
    hccl::RpingOutput result[10];
    char* payload = "pingmesh";
    PingQpInfo rdmainfo{0};
    for (u32 i = 0; i < 10; i++) {
        HcclIpAddress targetIp = HcclIpAddress(0x7F000002 + i);
        pingMesh->rdmaInfoMaps_.Emplace(std::string(targetIp.GetReadableIP()), rdmainfo);
        target[i].sip = targetIp;
        target[i].dip = targetIp;
        target[i].port = 13866;
        target[i].tc = 1;
        target[i].sl = 1;
        target[i].srcPort = 0;
        target[i].len = 9;
        target[i].addrType = 0;
        memcpy_s(target[i].payload, target[i].len, payload, target[i].len);
    }
    int i = 0;
    pingMesh->pingHandle_ = &i;
    pingMesh->rpingState_ = RpingState::RUN;
    pingMesh->rpingTargetNum_ = 10;
    PingInitInfo& initInfo = pingMesh->initInfo_;
    initInfo.result.bufferSize = 4096 * 10;
    initInfo.result.payloadOffset = 100;
    initInfo.result.headerSize = 50;
    ret = pingMesh->HccnRpingGetResult(deviceId, nodeNum, target, result);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    pingMesh->isUsePayload_ = true;
    void* payloadout = nullptr;
    u32 pyaloadlenout = 0;
    pingMesh->mode = HCCN_RPING_MODE_ROCE;
    HccnRpingMode Hrtmode = pingMesh->mode;
    ret = pingMesh->HccnRpingGetPayload(deviceId, &payloadout, &pyaloadlenout, Hrtmode);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    pingMesh->isUsePayload_ = false;
    pingMesh->mode = HCCN_RPING_MODE_ROCE;
    Hrtmode = pingMesh->mode;
    ret = pingMesh->HccnRpingGetPayload(deviceId, &payloadout, &pyaloadlenout, Hrtmode);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(PingMesh_UT, ut_PingMeshGetResultFail)
{
    MOCKER(hrtMemcpyEx).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_NOT_SUPPORT));

    u32 deviceId = 1;
    u32 mode = static_cast<u32>(LinkType::LINK_ROCE);
    HcclIpAddress ipAddr = HcclIpAddress(0x7F000001);
    u32 port = 13866;
    u32 nodeNum = 10;
    u32 bufferSize = 100U;
    u32 sl = 0;
    u32 tc = 0;

    // 初始化成功
    std::shared_ptr<PingMesh> pingMesh;
    pingMesh.reset(new (std::nothrow) PingMesh());
    auto ret = pingMesh->HccnRpingInit(deviceId, mode, ipAddr, port, nodeNum, bufferSize, sl, tc);

    // 获取结果
    hccl::RpingInput target[10];
    hccl::RpingOutput result[10];
    char* payload = "pingmesh";
    PingQpInfo rdmainfo{0};
    for (u32 i = 0; i < 10; i++) {
        HcclIpAddress targetIp = HcclIpAddress(0x7F000002 + i);
        pingMesh->rdmaInfoMaps_.Emplace(std::string(targetIp.GetReadableIP()), rdmainfo);
        target[i].sip = targetIp;
        target[i].dip = targetIp;
        target[i].port = 13866;
        target[i].tc = 1;
        target[i].sl = 1;
        target[i].srcPort = 0;
        target[i].len = 9;
        target[i].addrType = 0;
        memcpy_s(target[i].payload, target[i].len, payload, target[i].len);
    }
    int i = 0;
    pingMesh->pingHandle_ = &i;
    pingMesh->rpingState_ = RpingState::RUN;
    pingMesh->rpingTargetNum_ = 10;
    PingInitInfo& initInfo = pingMesh->initInfo_;
    initInfo.result.bufferSize = 4096 * 10;
    initInfo.result.payloadOffset = 100;
    initInfo.result.headerSize = 50;
    ret = pingMesh->HccnRpingGetResult(deviceId, nodeNum, target, result);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(PingMesh_UT, ut_PingMeshStateCheck)
{
    u32 deviceId = 1;
    u32 mode = static_cast<u32>(LinkType::LINK_ROCE);
    HcclIpAddress ipAddr = HcclIpAddress(0x7F000001);
    u32 port = 13866;
    u32 nodeNum = 10;
    u32 bufferSize = 100U;
    u32 sl = 0;
    u32 tc = 0;

    std::shared_ptr<PingMesh> pingMesh;
    pingMesh.reset(new (std::nothrow) PingMesh());
    pingMesh->HccnRpingInit(deviceId, mode, ipAddr, port, nodeNum, bufferSize, sl, tc);

    // 未初始化状态
    pingMesh->rpingState_ = RpingState::UNINIT;
    hccl::RpingInput* target = nullptr;
    HccnRpingAddTargetConfig config;
    config.connectTimeout = 120000;
    auto ret = pingMesh->HccnRpingAddTarget(deviceId, nodeNum, target, &config);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);

    u32 pktNum = 10;
    u32 interval = 1;
    u32 timeout = 1000;
    ret = pingMesh->HccnRpingBatchPingStart(deviceId, pktNum, interval, timeout);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);

    ret = pingMesh->HccnRpingBatchPingStop(deviceId);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);

    // 初始化状态
    pingMesh->rpingState_ = RpingState::INITED;
    ret = pingMesh->HccnRpingBatchPingStart(deviceId, pktNum, interval, timeout);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);

    ret = pingMesh->HccnRpingBatchPingStop(deviceId);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);

    // 预备状态
    pingMesh->rpingState_ = RpingState::READY;
    ret = pingMesh->HccnRpingBatchPingStop(deviceId);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(PingMesh_UT, ut_PingMeshSendRecvInfo)
{
    u32 deviceId = 1;
    u32 mode = static_cast<u32>(LinkType::LINK_ROCE);
    HcclIpAddress ipAddr = HcclIpAddress(0x7F000001);
    u32 port = 13866;
    u32 nodeNum = 10;
    u32 bufferSize = 100U;
    u32 sl = 0;
    u32 tc = 0;

    std::shared_ptr<PingMesh> pingMesh;
    pingMesh.reset(new (std::nothrow) PingMesh());
    pingMesh->HccnRpingInit(deviceId, mode, ipAddr, port, nodeNum, bufferSize, sl, tc);

    // 发送成功
    PingInitInfo initInfo{};
    std::shared_ptr<HcclSocket> socket = pingMesh->socket_;
    pingMesh->connThreadStop_.store(true);
    auto ret = pingMesh->RpingSendInitInfo(deviceId, port, ipAddr, initInfo, socket);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 接收成功
    MOCKER_CPP(&HcclSocket::GetStatus).expects(atMost(1)).will(returnValue(HcclSocketStatus::SOCKET_OK));

    void* clientNetCtx = pingMesh->netCtx_;
    ret = pingMesh->RpingRecvTargetInfo(clientNetCtx, port, ipAddr, initInfo, 120000);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(PingMesh_UT, ut_PingMeshSendRecvInfoPatch)
{
    // 补充覆盖率
    u32 deviceId = 1;
    u32 mode = static_cast<u32>(LinkType::LINK_ROCE);
    HcclIpAddress ipAddr = HcclIpAddress(0x7F000001);
    u32 port = 13866;
    u32 nodeNum = 10;
    u32 bufferSize = 100U;
    u32 sl = 0;
    u32 tc = 0;

    std::shared_ptr<PingMesh> pingMesh;
    pingMesh.reset(new (std::nothrow) PingMesh());
    pingMesh->HccnRpingInit(deviceId, mode, ipAddr, port, nodeNum, bufferSize, sl, tc);

    // 发送成功
    MOCKER_CPP(&HcclSocket::GetStatus).stubs().will(returnValue(HcclSocketStatus::SOCKET_CONNECTING));

    PingInitInfo initInfo{};
    std::shared_ptr<HcclSocket> socket = pingMesh->socket_;
    std::thread backgroundTh(&PingMesh::RpingSendInitInfo, pingMesh, deviceId, port, ipAddr, initInfo, socket);
    usleep(1000);
    pingMesh->connThreadStop_.store(true);
    backgroundTh.join();
}

TEST_F(PingMesh_UT, ut_PingMeshSendRecvInfoPatch1)
{
    u32 deviceId = 1;
    u32 mode = static_cast<u32>(LinkType::LINK_ROCE);
    HcclIpAddress ipAddr = HcclIpAddress(0x7F000001);
    u32 port = 13866;
    u32 nodeNum = 10;
    u32 bufferSize = 100U;
    u32 sl = 0;
    u32 tc = 0;

    std::shared_ptr<PingMesh> pingMesh;
    pingMesh.reset(new (std::nothrow) PingMesh());
    pingMesh->HccnRpingInit(deviceId, mode, ipAddr, port, nodeNum, bufferSize, sl, tc);

    PingInitInfo initInfo{};
    std::shared_ptr<HcclSocket> socket = nullptr;
    pingMesh->socketMaps_.insert({std::string(ipAddr.GetReadableIP()), socket});
    void* clientNetCtx = pingMesh->netCtx_;
    auto ret = pingMesh->RpingRecvTargetInfo(clientNetCtx, port, ipAddr, initInfo, 120000);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(PingMesh_UT, ut_PingMeshGetPayload)
{
    MOCKER(hrtMemcpyEx).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    u32 deviceId = 1;
    u32 mode = static_cast<u32>(LinkType::LINK_ROCE);
    HcclIpAddress ipAddr = HcclIpAddress(std::string("2001:0db8:85a3:08d3:1319:8a2e:0370:7344"));
    u32 port = 13866;
    u32 nodeNum = 10;
    u32 bufferSize = 100U;
    u32 sl = 0;
    u32 tc = 0;

    // 初始化成功
    std::shared_ptr<PingMesh> pingMesh;
    pingMesh.reset(new (std::nothrow) PingMesh());
    auto ret = pingMesh->HccnRpingInit(deviceId, mode, ipAddr, port, nodeNum, bufferSize, sl, tc);

    // 获取结果
    hccl::RpingInput target[10];
    hccl::RpingOutput result[10];
    char* payload = "pingmesh";
    PingQpInfo rdmainfo{0};
    std::string ipAddrStr = "2001:0db8:85a3:08d3:1319:8a2e:0370:7345";
    HcclIpAddress firstTargetIp = HcclIpAddress(ipAddrStr);

    HcclInAddr ipAddrBin = firstTargetIp.GetBinaryAddress();
    for (u32 i = 0; i < 10; i++) {
        ipAddrBin.addr6.s6_addr[15]++;
        HcclIpAddress targetIp = HcclIpAddress(AF_INET6, ipAddrBin);
        pingMesh->rdmaInfoMaps_.Emplace(std::string(targetIp.GetReadableIP()), rdmainfo);
        target[i].sip = targetIp;
        target[i].dip = targetIp;
        target[i].port = 13866;
        target[i].tc = 1;
        target[i].sl = 1;
        target[i].srcPort = 0;
        target[i].len = 9;
        target[i].addrType = 0;
        memcpy_s(target[i].payload, target[i].len, payload, target[i].len);
    }
    int i = 0;
    pingMesh->pingHandle_ = &i;
    pingMesh->rpingState_ = RpingState::RUN;
    pingMesh->rpingTargetNum_ = 10;
    PingInitInfo& initInfo = pingMesh->initInfo_;
    initInfo.result.bufferSize = 4096 * 10;
    initInfo.result.payloadOffset = 100;
    initInfo.result.headerSize = 50;

    pingMesh->isUsePayload_ = true;
    void* payloadout = nullptr;
    u32 pyaloadlenout = 0;
    pingMesh->mode = HCCN_RPING_MODE_ROCE;
    HccnRpingMode Hrtmode = pingMesh->mode;
    ret = pingMesh->HccnRpingGetPayload(deviceId, &payloadout, &pyaloadlenout, Hrtmode);

    EXPECT_EQ(ret, HCCL_SUCCESS);

    pingMesh->isUsePayload_ = false;
    pingMesh->mode = HCCN_RPING_MODE_ROCE;
    Hrtmode = pingMesh->mode;
    ret = pingMesh->HccnRpingGetPayload(deviceId, &payloadout, &pyaloadlenout, Hrtmode);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 打印结果
    if (payloadout != nullptr) {
        std::cout << "Payload: " << static_cast<char*>(payloadout) << std::endl;
        std::cout << "Payload Length: " << pyaloadlenout << std::endl;
    } else {
        std::cout << "Payload is null." << std::endl;
    }
}

// ====== hrtMemcpyEx 单元测试 ======
TEST_F(PingMesh_UT, HrtMemcpyEx_normal)
{
    char srcBuf[64] = {0};
    char dstBuf[64] = {0};
    const char* testData = "hello hrtMemcpyEx";
    memcpy_s(srcBuf, sizeof(srcBuf), testData, strlen(testData) + 1);

    HcclResult ret
        = hrtMemcpyEx(dstBuf, sizeof(dstBuf), srcBuf, strlen(testData) + 1, rtMemcpyKind_t::RT_MEMCPY_HOST_TO_HOST);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_STREQ(dstBuf, testData);
}

TEST_F(PingMesh_UT, HrtMemcpyEx_null_dst)
{
    char srcBuf[64] = {0};
    HcclResult ret
        = hrtMemcpyEx(nullptr, sizeof(srcBuf), srcBuf, sizeof(srcBuf), rtMemcpyKind_t::RT_MEMCPY_HOST_TO_HOST);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(PingMesh_UT, HrtMemcpyEx_null_src)
{
    char dstBuf[64] = {0};
    HcclResult ret
        = hrtMemcpyEx(dstBuf, sizeof(dstBuf), nullptr, sizeof(dstBuf), rtMemcpyKind_t::RT_MEMCPY_HOST_TO_HOST);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(PingMesh_UT, HrtMemcpyEx_count_exceeds_destMax)
{
    char srcBuf[8] = {0};
    char dstBuf[4] = {0};
    HcclResult ret
        = hrtMemcpyEx(dstBuf, sizeof(dstBuf), srcBuf, sizeof(srcBuf), rtMemcpyKind_t::RT_MEMCPY_HOST_TO_HOST);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(PingMesh_UT, HrtMemcpyEx_rt_fail)
{
    MOCKER(rtMemcpyEx).stubs().with(mockcpp::any()).will(returnValue(RT_ERROR_NONE + 1));
    char srcBuf[64] = {0};
    char dstBuf[64] = {0};
    HcclResult ret
        = hrtMemcpyEx(dstBuf, sizeof(dstBuf), srcBuf, sizeof(srcBuf), rtMemcpyKind_t::RT_MEMCPY_DEVICE_TO_HOST);
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
}

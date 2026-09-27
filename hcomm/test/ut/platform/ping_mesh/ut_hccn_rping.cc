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
#include <stdio.h>

#include "ping_mesh.h"
#include "hccn_rping.h"
#include "hccl_ip_address.h"

#include "hccn_rping.h"
#include "ping_mesh.h"
#include "hccl_ip_address.h"
#include "adapter_hccp.h"
#include "network/hccp_common.h"
#include "network/hccp_ping.h"
#include "dlra_function.h"
#include "adapter_rts.h"

using namespace hccl;

HcclResult HccnRpingGetResultStub(PingMesh* obj, u32 deviceId, u32 targetNum, RpingInput* input, RpingOutput* output)
{
    for (int i = 0; i < targetNum; i++) {
        output[i].state = 2;
        output[i].txPkt = 10;
        output[i].rxPkt = 10;
        output[i].minRTT = 5;
        output[i].maxRTT = 15;
        output[i].avgRTT = 10;
    }
    return HCCL_SUCCESS;
}

inline HcclResult hrtGetDeviceRefreshStub(s32* deviceLogicId)
{
    *deviceLogicId = 1;
    return HCCL_SUCCESS;
}

class HccnRping_UT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HccnRping_UT SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "HccnRping_UT TearDown" << std::endl; }

    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&PingMesh::HccnRpingInit).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

        MOCKER_CPP(&PingMesh::HccnRpingDeinit).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

        MOCKER_CPP(&PingMesh::HccnRpingAddTarget).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

        MOCKER_CPP(&PingMesh::HccnRpingRemoveTarget).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

        MOCKER_CPP(&PingMesh::HccnRpingGetTarget).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

        MOCKER_CPP(&PingMesh::HccnRpingBatchPingStart).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

        MOCKER_CPP(&PingMesh::HccnRpingBatchPingStop).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

        MOCKER_CPP(&PingMesh::HccnRpingGetResult).stubs().with(mockcpp::any()).will(invoke(HccnRpingGetResultStub));

        MOCKER_CPP(&PingMesh::HccnRpingGetPayload).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

        MOCKER(hrtGetDeviceRefresh).stubs().with(mockcpp::any()).will(invoke(hrtGetDeviceRefreshStub));

        std::cout << "HccnRping_UT Test SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "HccnRping_UT Test TearDown" << std::endl;
    }

public:
    char deviceIp_[8][11] = {"192.1.1.58", "192.1.2.58", "192.1.3.58", "192.1.4.58",
                             "192.1.1.59", "192.1.2.59", "192.1.3.59", "192.1.4.59"};

    int ipLen_ = 16;
};

TEST_F(HccnRping_UT, ut_HccnRpingInit)
{
    MOCKER_CPP(&PingMesh::GetDeviceLogicId).stubs().with(mockcpp::any()).will(returnValue(1));

    u32 devId = 1;
    u32 devserverId = 2;
    HccnRpingInitAttr* initAttr = new HccnRpingInitAttr();
    initAttr->mode = HCCN_RPING_MODE_ROCE;
    initAttr->port = 13886;
    initAttr->npuNum = 128;
    initAttr->bufferSize = 4096 * 10;
    initAttr->sl = 0;
    initAttr->tc = 0;
    initAttr->ipAddr = new char[ipLen_];
    strcpy(initAttr->ipAddr, deviceIp_[devId]);
    void* pingmesh = nullptr;

    // 初始化
    auto ret = HccnRpingInit(devId, initAttr, &pingmesh);
    EXPECT_EQ(ret, HCCN_SUCCESS);
    EXPECT_TRUE(pingmesh != nullptr);
    delete[] initAttr->ipAddr;
    delete initAttr;

    // 反初始化
    ret = HccnRpingDeinit(pingmesh);
    EXPECT_EQ(ret, HCCN_SUCCESS);
}

TEST_F(HccnRping_UT, ut_HccnRpingInit_error)
{
    MOCKER_CPP(&PingMesh::GetDeviceLogicId).stubs().with(mockcpp::any()).will(returnValue(1));

    u32 devId = 1;
    u32 devserverId = 2;
    HccnRpingInitAttr* initAttr = nullptr;
    void* pingmesh = nullptr;
    // 空initAttr初始化错误检查
    auto ret = HccnRpingInit(devId, initAttr, &pingmesh);
    EXPECT_EQ(ret, HCCN_E_PARA);

    initAttr = new HccnRpingInitAttr();
    initAttr->mode = HCCN_RPING_MODE_ROCE;
    initAttr->port = 13886;
    initAttr->npuNum = 128;
    initAttr->bufferSize = 4096 * 10;
    initAttr->sl = 0;
    initAttr->tc = 0;
    initAttr->ipAddr = new char[ipLen_];
    strcpy(initAttr->ipAddr, deviceIp_[devId]);

    ret = HccnRpingInit(devId, initAttr, &pingmesh);
    EXPECT_EQ(ret, HCCN_SUCCESS);
    EXPECT_TRUE(pingmesh != nullptr);
    delete[] initAttr->ipAddr;
    delete initAttr;
}

TEST_F(HccnRping_UT, ut_HccnRping)
{
    MOCKER_CPP(&PingMesh::GetDeviceLogicId).stubs().with(mockcpp::any()).will(returnValue(1));

    u32 devId = 1;
    u32 devserverId = 2;
    HccnRpingInitAttr* initAttr = new HccnRpingInitAttr();
    initAttr->mode = HCCN_RPING_MODE_ROCE;
    initAttr->port = 13886;
    initAttr->npuNum = 128;
    initAttr->bufferSize = 4096 * 10;
    initAttr->sl = 0;
    initAttr->tc = 0;
    initAttr->ipAddr = new char[ipLen_];
    strcpy(initAttr->ipAddr, deviceIp_[devId]);
    void* pingmesh = nullptr;

    // 初始化
    auto ret = HccnRpingInit(devId, initAttr, &pingmesh);
    EXPECT_EQ(ret, HCCN_SUCCESS);
    EXPECT_TRUE(pingmesh != nullptr);
    delete[] initAttr->ipAddr;
    delete initAttr;

    // 添加节点
    int targetNum = 1;
    HccnRpingTargetInfo* target = new HccnRpingTargetInfo[1];
    target[0].srcPort = 0;
    target[0].sl = 0;
    target[0].tc = 0;
    target[0].port = 13886;
    target[0].payloadLen = 12;
    target[0].addrType = 0;
    target[0].srcIp = new char[ipLen_];
    target[0].dstIp = new char[ipLen_];
    char payload[12] = "hellotarget";
    strcpy(target[0].payload, payload);
    strcpy(target[0].srcIp, deviceIp_[devserverId]);
    strcpy(target[0].dstIp, deviceIp_[devId]);
    ret = HccnRpingAddTarget(pingmesh, targetNum, target);
    EXPECT_EQ(ret, HCCN_SUCCESS);

    // 获取节点
    HccnRpingAddTargetState targetState[1]{HCCN_RPING_ADDTARGET_STATE_DONE};
    ret = HccnRpingGetTarget(pingmesh, targetNum, target, targetState);
    EXPECT_EQ(ret, HCCN_SUCCESS);

    // 开始批量测试
    uint32_t pktNum = 10;
    uint32_t interval = 1;
    uint32_t timeout = 1000;
    ret = HccnRpingBatchPingStart(pingmesh, pktNum, interval, timeout);
    EXPECT_EQ(ret, HCCN_SUCCESS);

    // 停止批量测试
    ret = HccnRpingBatchPingStop(pingmesh);
    EXPECT_EQ(ret, HCCN_SUCCESS);

    // 获取结果
    HccnRpingResultInfo result[1]{0};
    ret = HccnRpingGetResult(pingmesh, targetNum, target, result);
    EXPECT_EQ(ret, HCCN_SUCCESS);

    // 获取payload
    void* payloadOutput = nullptr;
    u32 payloadLenOutput = 0;
    ret = HccnRpingGetPayload(pingmesh, &payloadOutput, &payloadLenOutput);
    EXPECT_EQ(ret, HCCN_SUCCESS);

    // 删除节点
    ret = HccnRpingRemoveTarget(pingmesh, targetNum, target);
    EXPECT_EQ(ret, HCCN_SUCCESS);

    // 反初始化
    ret = HccnRpingDeinit(pingmesh);
    EXPECT_EQ(ret, HCCN_SUCCESS);

    delete[] target[0].srcIp;
    delete[] target[0].dstIp;
    delete[] target;
}

TEST_F(HccnRping_UT, ut_HccnRping_RpingAddTargetV2)
{
    MOCKER_CPP(&PingMesh::GetDeviceLogicId).stubs().with(mockcpp::any()).will(returnValue(1));

    u32 devId = 1;
    u32 devserverId = 2;
    HccnRpingInitAttr* initAttr = new HccnRpingInitAttr();
    initAttr->mode = HCCN_RPING_MODE_ROCE;
    initAttr->port = 13886;
    initAttr->npuNum = 128;
    initAttr->bufferSize = 4096 * 10;
    initAttr->sl = 0;
    initAttr->tc = 0;
    initAttr->ipAddr = new char[ipLen_];
    strcpy(initAttr->ipAddr, deviceIp_[devId]);
    void* pingmesh = nullptr;

    // 初始化
    auto ret = HccnRpingInit(devId, initAttr, &pingmesh);
    EXPECT_EQ(ret, HCCN_SUCCESS);
    EXPECT_TRUE(pingmesh != nullptr);
    delete[] initAttr->ipAddr;
    delete initAttr;

    // 添加节点
    int targetNum = 1;
    HccnRpingTargetInfo* target = new HccnRpingTargetInfo[1];
    target[0].srcPort = 0;
    target[0].sl = 0;
    target[0].tc = 0;
    target[0].port = 13886;
    target[0].payloadLen = 12;
    target[0].addrType = 0;
    target[0].srcIp = new char[ipLen_];
    target[0].dstIp = new char[ipLen_];
    char payload[12] = "hellotarget";
    strcpy(target[0].payload, payload);
    strcpy(target[0].srcIp, deviceIp_[devserverId]);
    strcpy(target[0].dstIp, deviceIp_[devId]);

    HccnRpingAddTargetConfig config;
    config.connectTimeout = 1000;
    ret = HccnRpingAddTargetV2(pingmesh, targetNum, target, &config);
    EXPECT_EQ(ret, HCCN_SUCCESS);

    // 删除节点
    ret = HccnRpingRemoveTarget(pingmesh, targetNum, target);
    EXPECT_EQ(ret, HCCN_SUCCESS);

    // 反初始化
    ret = HccnRpingDeinit(pingmesh);
    EXPECT_EQ(ret, HCCN_SUCCESS);

    delete[] target[0].srcIp;
    delete[] target[0].dstIp;
    delete[] target;
}

TEST_F(HccnRping_UT, ut_HccnRping_RpingAddTargetWithCfg)
{
    MOCKER_CPP(&PingMesh::GetDeviceLogicId).stubs().with(mockcpp::any()).will(returnValue(1));

    u32 devId = 1;
    u32 devserverId = 2;
    HccnRpingInitAttr* initAttr = new HccnRpingInitAttr();
    initAttr->mode = HCCN_RPING_MODE_ROCE;
    initAttr->port = 13886;
    initAttr->npuNum = 128;
    initAttr->bufferSize = 4096 * 10;
    initAttr->sl = 0;
    initAttr->tc = 0;
    initAttr->ipAddr = new char[ipLen_];
    strcpy(initAttr->ipAddr, deviceIp_[devId]);
    void* pingmesh = nullptr;

    // 初始化
    auto ret = HccnRpingInit(devId, initAttr, &pingmesh);
    EXPECT_EQ(ret, HCCN_SUCCESS);
    EXPECT_TRUE(pingmesh != nullptr);
    delete[] initAttr->ipAddr;
    delete initAttr;

    // 添加节点
    int targetNum = 1;
    HccnRpingTargetInfo* target = new HccnRpingTargetInfo[1];
    target[0].srcPort = 0;
    target[0].sl = 0;
    target[0].tc = 0;
    target[0].port = 13886;
    target[0].payloadLen = 12;
    target[0].addrType = 0;
    target[0].srcIp = new char[ipLen_];
    target[0].dstIp = new char[ipLen_];
    char payload[12] = "hellotarget";
    strcpy(target[0].payload, payload);
    strcpy(target[0].srcIp, deviceIp_[devserverId]);
    strcpy(target[0].dstIp, deviceIp_[devId]);

    HccnRpingAddTargetConfig config;
    config.connectTimeout = 1000;
    ret = HccnRpingAddTargetWithCfg(pingmesh, targetNum, target, &config);
    EXPECT_EQ(ret, HCCN_SUCCESS);

    // 删除节点
    ret = HccnRpingRemoveTarget(pingmesh, targetNum, target);
    EXPECT_EQ(ret, HCCN_SUCCESS);

    // 反初始化
    ret = HccnRpingDeinit(pingmesh);
    EXPECT_EQ(ret, HCCN_SUCCESS);

    delete[] target[0].srcIp;
    delete[] target[0].dstIp;
    delete[] target;
}

TEST_F(HccnRping_UT, ut_HccnRping_RpingAddTargetV2_err)
{
    HccnRpingAddTargetConfig config;
    config.connectTimeout = 0;
    auto ret = HccnRpingAddTargetWithCfg(nullptr, 1, nullptr, &config);
    EXPECT_EQ(ret, HCCN_E_PARA);
}

TEST_F(HccnRping_UT, ut_HccnRpingHandleCheck)
{
    u32 devId = 3;
    u32 devserverId = 2;
    HccnRpingInitAttr* initAttr = new HccnRpingInitAttr();
    initAttr->mode = HCCN_RPING_MODE_ROCE;
    initAttr->port = 13886;
    initAttr->npuNum = 128;
    initAttr->bufferSize = 4096 * 10;
    initAttr->sl = 0;
    initAttr->tc = 0;
    initAttr->ipAddr = new char[ipLen_];
    strcpy(initAttr->ipAddr, deviceIp_[devId]);
    void* pingmesh = nullptr;

    // 初始化
    auto ret = HccnRpingInit(1, initAttr, &pingmesh);
    EXPECT_EQ(ret, HCCN_SUCCESS);
    EXPECT_TRUE(pingmesh != nullptr);
    delete[] initAttr->ipAddr;
    delete initAttr;

    // 添加节点
    int targetNum = 1;
    HccnRpingTargetInfo* target = new HccnRpingTargetInfo[1];
    target[0].srcPort = 0;
    target[0].sl = 0;
    target[0].tc = 0;
    target[0].port = 13886;
    target[0].payloadLen = 12;
    target[0].addrType = 0;
    target[0].srcIp = new char[ipLen_];
    target[0].dstIp = new char[ipLen_];
    char payload[12] = "hellotarget";
    strcpy(target[0].payload, payload);
    strcpy(target[0].srcIp, deviceIp_[devserverId]);
    strcpy(target[0].dstIp, deviceIp_[devId]);
    ret = HccnRpingAddTarget(pingmesh, targetNum, target);
    EXPECT_EQ(ret, HCCN_E_PARA);

    // 获取节点
    HccnRpingAddTargetState targetState[1]{HCCN_RPING_ADDTARGET_STATE_DONE};
    ret = HccnRpingGetTarget(pingmesh, targetNum, target, targetState);
    EXPECT_EQ(ret, HCCN_E_PARA);

    // 开始批量测试
    uint32_t pktNum = 10;
    uint32_t interval = 1;
    uint32_t timeout = 1000;
    ret = HccnRpingBatchPingStart(pingmesh, pktNum, interval, timeout);
    EXPECT_EQ(ret, HCCN_E_PARA);

    // 停止批量测试
    ret = HccnRpingBatchPingStop(pingmesh);
    EXPECT_EQ(ret, HCCN_E_PARA);

    // 获取结果
    HccnRpingResultInfo result[1]{0};
    ret = HccnRpingGetResult(pingmesh, targetNum, target, result);
    EXPECT_EQ(ret, HCCN_E_PARA);

    // 删除节点
    ret = HccnRpingRemoveTarget(pingmesh, targetNum, target);
    EXPECT_EQ(ret, HCCN_E_PARA);

    // 反初始化
    ret = HccnRpingDeinit(pingmesh);
    EXPECT_EQ(ret, HCCN_E_PARA);

    delete[] target[0].srcIp;
    delete[] target[0].dstIp;
    delete[] target;
    delete static_cast<PingMesh*>(pingmesh);
}

TEST_F(HccnRping_UT, ut_HccnRpingAdapter)
{
    // 补充覆盖率
    auto ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 未初始化新增接口
    struct PingInitAttr initAttr {
        0
    };
    struct PingInitInfo initInfo {
        0
    };
    void* pingHandle = nullptr;
    ret = hrtRaPingInit(&initAttr, &initInfo, &pingHandle);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    ret = hrtRaPingDeinit(pingHandle);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);

    struct PingTargetInfo target[2];
    struct PingTargetCommInfo targetRemoteInfo[2];
    struct PingTaskAttr attr;
    u32 num;
    struct PingTargetResult result[2];
    ret = hrtRaPingTargetAdd(pingHandle, target, num);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    ret = hrtRaPingTargetDel(pingHandle, targetRemoteInfo, num);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    ret = hrtRaPingTaskStart(pingHandle, &attr);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    ret = hrtRaPingTaskStop(pingHandle);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    ret = hrtRaPingGetResults(pingHandle, result, &num);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);

    // 手动初始化新增接口
    DlRaFunction::GetInstance().dlRaPingInit = [](struct PingInitAttr*, struct PingInitInfo*, void**) -> int {
        return 0;
    };
    DlRaFunction::GetInstance().dlRaPingDeinit = [](void*) -> int {
        return 0;
    };
    DlRaFunction::GetInstance().dlRaPingTargetAdd = [](void*, struct PingTargetInfo*, uint32_t) -> int {
        return 0;
    };
    DlRaFunction::GetInstance().dlRaPingTargetDel = [](void*, struct PingTargetCommInfo*, uint32_t) -> int {
        return 0;
    };
    DlRaFunction::GetInstance().dlRaPingTaskStart = [](void*, struct PingTaskAttr*) -> int {
        return 0;
    };
    DlRaFunction::GetInstance().dlRaPingTaskStop = [](void*) -> int {
        return 0;
    };
    DlRaFunction::GetInstance().dlRaPingGetResults = [](void*, struct PingTargetResult*, uint32_t*) -> int {
        return 0;
    };

    ret = hrtRaPingInit(&initAttr, &initInfo, &pingHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = hrtRaPingDeinit(pingHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = hrtRaPingTargetAdd(pingHandle, target, num);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = hrtRaPingTargetDel(pingHandle, targetRemoteInfo, num);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = hrtRaPingTaskStart(pingHandle, &attr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = hrtRaPingTaskStop(pingHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = hrtRaPingGetResults(pingHandle, result, &num);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // rts新增接口
    char dst[10]{0};
    uint64_t destMax = 10;
    char src[10]{1};
    uint64_t count = 10;
    HcclRtMemcpyKind kind = HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_HOST;
    ret = hrtMemSyncCopyEx(dst, destMax, src, count, kind);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

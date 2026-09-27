/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// HcommCcuControlApi 系列 UT 的公共测试装置(fixture / mock helper / 通用宏)。
// 由 ut_HcommCcuControlApi_test.cc 与 ut_HcommCcuControlApi_ckenop_test.cc 等共享,
// 避免同一巨型源文件承载过多用例(超大源文件规则)。
#ifndef UT_HCOMM_CCU_CONTROL_API_COMMON_H
#define UT_HCOMM_CCU_CONTROL_API_COMMON_H

#include <iostream>
#include <acl/acl.h>
#include <set>
#include <utility>
#include "hccl_api_base_test.h"

#define private public
#define protected public

#include "log.h"
#include "adapter_rts.h"
#include "hcomm_c_adpt.h"
#include "op_base.h"

#include "ccu_device_pub.h"
#include "ccu_kernel_mgr.h"
#include "ccu_ins_generator_v1.h"
#include "ccu_instance_mgr.h"
#include "ccu_device_res.h"
#include "ccu_res_batch_allocator.h"
#include "ccu_res_specs.h"
#include "ccu_comp.h"
#include "ccu_pfe_cfg_mgr.h"
#include "ccu_launch.h"
#include "ccu_res.h"
#include "ccu_var_event_res_mgr.h"

#include "mocks/ccu_device_mock_utils.h"
#include "mocks/ccu_channel_mock_utils.h"

#include "adapter_hal_pub.h"

#include "hcomm_primitives.h"

#undef protected
#undef private

// 注意: ccu_kernel_impl/*.h 里的 demo kernel 函数是非 inline 的普通函数定义,
// 每个用例 .cc 只能各自 include 自己需要的 demo 头, 不能放进本共享头(否则多 TU 重复定义)。

namespace hcomm {
HcclResult GetHcclVersionForCcuKernelMgr(int& hcclVersion);
}

// 说明: 以下匿名 namespace 内的符号具有 internal linkage, 每个包含本头的 TU 各持一份独立副本,
// 不会产生重复定义。各 UT 用例自包含地初始化设备环境, 故各 TU 副本互不干扰。
namespace {
constexpr int32_t TEST_DEVICE_LOGIC_ID = 0;
constexpr int32_t OTHER_TEST_DEVICE_LOGIC_ID = 1;
constexpr int32_t HCCL_VERSION_USING_VALIDATE_AND_APPLY_DIE = 90100001;
int32_t g_runtimeDeviceLogicId = TEST_DEVICE_LOGIC_ID;
uint32_t g_deviceRefreshCalls = 0;
HcclResult g_deviceRefreshResult = HcclResult::HCCL_SUCCESS;

inline HcclResult MockHrtGetDeviceRefresh(int32_t* deviceLogicId)
{
    ++g_deviceRefreshCalls;
    if (g_deviceRefreshResult != HcclResult::HCCL_SUCCESS) {
        return g_deviceRefreshResult;
    }
    if (deviceLogicId == nullptr) {
        return HcclResult::HCCL_E_PTR;
    }
    *deviceLogicId = g_runtimeDeviceLogicId;
    return HcclResult::HCCL_SUCCESS;
}

inline HcclResult MockHrtGetDevice(int32_t* deviceLogicId)
{
    if (deviceLogicId == nullptr) {
        return HcclResult::HCCL_E_PTR;
    }
    *deviceLogicId = g_runtimeDeviceLogicId;
    return HcclResult::HCCL_SUCCESS;
}

inline HcclResult MockHrtGetDevicePhyIdByUserDevId(uint32_t deviceLogicId, uint32_t& devicePhyId, bool)
{
    devicePhyId = deviceLogicId;
    return HcclResult::HCCL_SUCCESS;
}

inline void MockControlDeviceRefresh(int32_t deviceLogicId)
{
    g_runtimeDeviceLogicId = deviceLogicId;
    g_deviceRefreshCalls = 0;
    g_deviceRefreshResult = HcclResult::HCCL_SUCCESS;
    MOCKER(hrtGetDeviceRefresh).stubs().with(mockcpp::any()).will(invoke(MockHrtGetDeviceRefresh));
}

inline CcuResult
MockControlAllocCntXnBlock(const int32_t deviceLogicId, const uint8_t dieId, hcomm::CntXnBlock& cntXnBlock)
{
    (void)deviceLogicId;
    (void)dieId;
    constexpr uint32_t startId = 12288;
    constexpr uint32_t count = 1024;
    cntXnBlock.resInfo = hcomm::ResInfo(startId, count);
    cntXnBlock.wishCntXns = {startId, startId + 1021};
    cntXnBlock.totalCntXn = startId + 1022;
    cntXnBlock.expectedCntXn = startId + 1023;
    return CcuResult::CCU_SUCCESS;
}
} // namespace

class HcommCcuControlApiTest : public BaseInit {
public:
    void SetUp() override
    {
        GlobalMockObject::verify();
        BaseInit::SetUp();
        // 将enableEntryLog默认返回为true
        MOCKER(GetExternalInputHcclEnableEntryLog).stubs().with(mockcpp::any()).will(returnValue(true));
        MOCKER(hcomm::GetHcclVersionForCcuKernelMgr)
            .stubs()
            .with(outBound(HCCL_VERSION_USING_VALIDATE_AND_APPLY_DIE))
            .will(returnValue(HCCL_SUCCESS));
        // PR 新增 CcuAllocCntXnBlock / CcuReleaseCntXnBlock 在 AllocCascCntBlock 路径中被调用
        MOCKER(hcomm::CcuAllocCntXnBlock)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(invoke(MockControlAllocCntXnBlock));
        MOCKER(hcomm::CcuReleaseCntXnBlock)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(returnValue(CcuResult::CCU_SUCCESS));
    }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }

protected:
};

inline std::pair<EndpointHandle, ChannelHandle>
MockCcuChannelConnect(uint32_t srcDevPhyId, uint32_t dstDevPhyId, uint32_t srcIp, uint32_t dstIp, CommEngine commEngine)
{
    HcommResult hcommRet = 0;

    CommAddr srcAddr{}, dstAddr{};
    srcAddr.type = CommAddrType::COMM_ADDR_TYPE_IP_V4;
    srcAddr.addr.s_addr = srcIp;
    dstAddr.type = CommAddrType::COMM_ADDR_TYPE_IP_V4;
    dstAddr.addr.s_addr = dstIp;

    const auto& srcEpDesc = MockEndpointDesc(srcAddr, srcDevPhyId);
    EndpointHandle srcEpHandle{};
    hcommRet = HcommEndpointCreate(&srcEpDesc, &srcEpHandle);
    EXPECT_EQ(hcommRet, static_cast<HcommResult>(HcclResult::HCCL_SUCCESS));

    const auto& dstEpDesc = MockEndpointDesc(dstAddr, dstDevPhyId);

    const auto& socket = MockHcclSocket(srcAddr, dstAddr);
    HcommSocket socketPtr = static_cast<HcommSocket>(socket.get());
    const auto& rmaBuffer = MockUbRmaBuffer();
    void* memHandle = static_cast<void*>(rmaBuffer.get());
    auto channelDesc = MockHcommChannelDesc(dstEpDesc, socketPtr, memHandle);
    constexpr uint32_t channelNum = 1;
    ChannelHandle channelHandle{0};
    hcommRet = HcommChannelCreate(srcEpHandle, commEngine, &channelDesc, channelNum, &channelHandle);
    EXPECT_EQ(hcommRet, static_cast<HcommResult>(HcclResult::HCCL_SUCCESS));

    int32_t statusList[] = {0};

    constexpr uint32_t MAX_LOOP_TIME = 100;
    uint32_t leftTime = MAX_LOOP_TIME;
    while (leftTime--) {
        hcommRet = HcommChannelGetStatus(&channelHandle, channelNum, statusList);
        if (hcommRet == static_cast<HcommResult>(HcclResult::HCCL_SUCCESS)) {
            break;
        }

        if (hcommRet != static_cast<HcommResult>(HcclResult::HCCL_E_AGAIN)) {
            HCCL_ERROR("[%s] invalid ret[%d].", __func__, hcommRet);
            break;
        }
    }

    EXPECT_EQ(hcommRet, static_cast<HcommResult>(HcclResult::HCCL_SUCCESS));
    return {srcEpHandle, channelHandle};
}

// 各用例通用的默认建链: 使用统一的 devPhyId / ip / commEngine 常量发起 CCU 建链。
// 抽出以消除各 TEST_F 中重复的建链常量声明 + MockCcuChannelConnect 调用样板。
inline std::pair<EndpointHandle, ChannelHandle>
MockCcuChannelConnectDefault(uint32_t srcDevPhyId, CommEngine commEngine = CommEngine::COMM_ENGINE_CCU)
{
    constexpr uint32_t dstDevPhyId = 1;
    constexpr uint32_t srcIp = 167772383; // 需要与RaGetDevEidInfoList接口打桩一致
    constexpr uint32_t dstIp = 0x87654321;
    return MockCcuChannelConnect(srcDevPhyId, dstDevPhyId, srcIp, dstIp, commEngine);
}

inline void MockChannelDestroy(const std::pair<EndpointHandle, ChannelHandle>& handles)
{
    HcommResult hcommRet = 0;
    constexpr uint32_t channelNum = 1;
    hcommRet = HcommChannelDestroy(&(handles.second), channelNum);
    EXPECT_EQ(hcommRet, static_cast<HcommResult>(HcclResult::HCCL_SUCCESS));

    hcommRet = HcommEndpointDestroy(handles.first);
    EXPECT_EQ(hcommRet, static_cast<HcommResult>(HcclResult::HCCL_SUCCESS));
}

inline ThreadHandle MockThreadAllocWithStream(CommEngine commEngine)
{
    // 选择调用Hcomm接口，而不是对具体的Thread实现打桩，减少内部依赖
    MOCKER(aclrtStreamGetId).stubs().will(returnValue(0));
    bool devRunning = false;
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(devRunning)).will(returnValue(HcclResult::HCCL_SUCCESS));

    constexpr uint32_t fakeNotifyNum = 0;
    aclrtStream fakeStream{(void*)0x12345678};
    ThreadHandle fakeThreadHandle{};
    EXPECT_EQ(
        HcommThreadAllocWithStream(commEngine, fakeStream, fakeNotifyNum, &fakeThreadHandle), HcclResult::HCCL_SUCCESS);

    return fakeThreadHandle;
}

inline void SetCcuResDescCcuMs(HcommCcuResDescHandle handle, hcomm::CcuVersion ccuVersion)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    ccuRet = HcommCcuInsResDescSetNum(handle, HCOMM_CCU_RES_TYPE_LOOP, 8 * 8 * 2);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    ccuRet = HcommCcuInsResDescSetNum(handle, HCOMM_CCU_RES_TYPE_CCU_BUF, 64 * 8 * 2);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    if (ccuVersion == hcomm::CcuVersion::CCU_V2) {
        ccuRet = HcommCcuInsResDescSetNum(handle, HCOMM_CCU_RES_TYPE_VARIABLE, 800);
        EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
        ccuRet = HcommCcuInsResDescSetNum(handle, HCOMM_CCU_RES_TYPE_ADDRESS, 0);
        EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    } else {
        ccuRet = HcommCcuInsResDescSetNum(handle, HCOMM_CCU_RES_TYPE_VARIABLE, 400);
        EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
        ccuRet = HcommCcuInsResDescSetNum(handle, HCOMM_CCU_RES_TYPE_ADDRESS, 400);
        EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    }
    ccuRet = HcommCcuInsResDescSetNum(handle, HCOMM_CCU_RES_TYPE_EVENT, 32 + 8 * 8 * 2);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    ccuRet = HcommCcuInsResDescSetNum(handle, HCOMM_CCU_RES_TYPE_CCU_THREAD, 2);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
}

// 打桩当前线程 DeviceId 并初始化 ccu 资源环境，返回 fakeDeviceLogicId 供调用方使用
inline int32_t MockCcuDeviceEnv(uint32_t fakeDevId, hcomm::CcuVersion fakeCcuVersion)
{
    MockControlDeviceRefresh(static_cast<int32_t>(fakeDevId));
    MOCKER(HcclGetThreadDeviceId).stubs().will(returnValue(static_cast<int32_t>(fakeDevId)));
    int32_t fakeDeviceLogicId = static_cast<int32_t>(fakeDevId);
    MOCKER(hrtGetDevice).stubs().with(outBoundP(&fakeDeviceLogicId)).will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex)
        .stubs()
        .with(mockcpp::any(), outBound(static_cast<uint32_t>(fakeDeviceLogicId)), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MockCcuNetworkDeviceDefault(fakeDeviceLogicId);
    EXPECT_EQ(MockCcuResourcesDefault(fakeDeviceLogicId, fakeCcuVersion), HcclResult::HCCL_SUCCESS);
    MockCcuChannelGetRes();
    MOCKER(hrtMemcpy).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    return fakeDeviceLogicId;
}

// 创建两个 die（dieId 0/1）的资源描述符并按 CcuMs 默认值填充
inline void
CreateCcuResDescsPair(HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM], hcomm::CcuVersion ccuVersion)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    ccuRet = HcommCcuInsResDescCreate(0, &resDescs[0]);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    ccuRet = HcommCcuInsResDescCreate(1, &resDescs[1]);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    SetCcuResDescCcuMs(resDescs[0], ccuVersion);
    SetCcuResDescCcuMs(resDescs[1], ccuVersion);
}

// 各用例通用的 V1 CCU 实例初始化: mock 设备环境 -> 建两 die 资源描述符 -> 创建 CcuInstance。
// 抽出以消除各 TEST_F 开头重复的样板代码。
inline void SetupV1CcuInstance(
    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM], CcuInsHandle& insHandle,
    uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2)
{
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    EXPECT_EQ(HcommCcuInsCreate(resDescs, descNum, &insHandle), CcuResult::CCU_SUCCESS);
}

// 销毁资源描述符数组
inline void DestroyCcuResDescs(HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM])
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    ccuRet = HcommCcuInsResDescDestroy(resDescs[0]);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    ccuRet = HcommCcuInsResDescDestroy(resDescs[1]);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
}

// die0 上注册单参数 kernel: HcommCcuKernelRegister + HcommCcuKernelRegisterEnd, 返回 kernelHandle。
// 抽出以消除各 TEST_F 尾部重复的注册样板(重复代码规则)。
inline CcuKernelHandle
RegisterSingleArgKernel(CcuInsHandle insHandle, char* kernelFuncName, void* kernelFunc, const void* kernelArgs[])
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    CcuKernelHandle kernelHandle{0};
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(
        insHandle, fakeDieId, kernelFuncName, kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    return kernelHandle;
}

// 通道 + 实例 + 资源描述符统一销毁, 析构有时序要求(先通道, 再实例, 最后描述符)。
inline void DestroyCcuChannelInsAndDescs(
    const std::pair<EndpointHandle, ChannelHandle>& handlePair, CcuInsHandle insHandle,
    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM])
{
    MockChannelDestroy(handlePair);
    CcuResult ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

// 注册单参数 kernel(不下发) -> 统一销毁。抽出以消除各 TEST_F 尾部重复的注册+销毁样板(重复代码规则)。
inline void RegisterAndCleanupSingleArgKernel(
    CcuInsHandle insHandle, char* kernelFuncName, void* kernelFunc, const void* kernelArgs[],
    const std::pair<EndpointHandle, ChannelHandle>& handlePair,
    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM])
{
    (void)RegisterSingleArgKernel(insHandle, kernelFuncName, kernelFunc, kernelArgs);
    DestroyCcuChannelInsAndDescs(handlePair, insHandle, resDescs);
}

// 注册单参数 kernel -> 申请流 -> 下发(argCount 个 0 值 taskArg) -> 统一销毁。
// argCount 为 0 表示空 taskArgs。抽出以消除各 TEST_F 尾部重复的注册+下发+销毁样板(重复代码规则)。
inline void RegisterLaunchAndCleanupSingleArgKernel(
    CcuInsHandle insHandle, char* kernelFuncName, void* kernelFunc, const void* kernelArgs[],
    const std::pair<EndpointHandle, ChannelHandle>& handlePair,
    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM], CommEngine commEngine, uint32_t argCount)
{
    CcuKernelHandle kernelHandle = RegisterSingleArgKernel(insHandle, kernelFuncName, kernelFunc, kernelArgs);

    auto fakeThreadHandle = MockThreadAllocWithStream(commEngine);
    std::vector<uint64_t> taskArgs(argCount, 0);
    void* fakeTaskArgs = static_cast<void*>(taskArgs.data());
    uint32_t fakeArgNum = taskArgs.size();
    EXPECT_EQ(HcommCcuKernelLaunch(fakeThreadHandle, kernelHandle, fakeTaskArgs, fakeArgNum), CcuResult::CCU_SUCCESS);

    DestroyCcuChannelInsAndDescs(handlePair, insHandle, resDescs);
}

#define CCU_FUNC_KERNEL_TEST(testName, demoFunc, expectRegisterSuccess)                                               \
    TEST_F(HcommCcuControlApiTest, testName)                                                                          \
    {                                                                                                                 \
        CcuResult ccuRet = CcuResult::CCU_E_RESERVED;                                                                 \
        constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;                                                     \
        constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;                                       \
        (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);                                                            \
                                                                                                                      \
        HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};                                            \
        CreateCcuResDescsPair(resDescs, fakeCcuVersion);                                                              \
        constexpr uint32_t descNum = 2;                                                                               \
        CcuInsHandle insHandle{0};                                                                                    \
        ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);                                                    \
        EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);                                                                    \
                                                                                                                      \
        ccuRet = HcommCcuKernelRegisterStart(insHandle);                                                              \
        EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);                                                                    \
                                                                                                                      \
        int32_t dummyArg = 0;                                                                                         \
        CcuKernelArg kernelArg = static_cast<CcuKernelArg>(&dummyArg);                                                \
        const void* kernelArgs[] = {kernelArg};                                                                       \
        CcuKernelHandle kernelHandle{0};                                                                              \
        auto kernelFunc = reinterpret_cast<void*>(demoFunc);                                                          \
        constexpr uint32_t fakeDieId = 0;                                                                             \
        constexpr uint32_t kernelArgNum = 1;                                                                          \
        ccuRet = HcommCcuKernelRegister(                                                                              \
            insHandle, fakeDieId, const_cast<char*>(#demoFunc), kernelFunc, kernelArgs, kernelArgNum, &kernelHandle); \
        if (expectRegisterSuccess) {                                                                                  \
            EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);                                                                \
            ccuRet = HcommCcuKernelRegisterEnd(insHandle);                                                            \
            EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);                                                                \
        } else {                                                                                                      \
            EXPECT_NE(ccuRet, CcuResult::CCU_SUCCESS);                                                                \
        }                                                                                                             \
                                                                                                                      \
        ccuRet = HcommCcuInsDestroy(insHandle);                                                                       \
        EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);                                                                    \
        DestroyCcuResDescs(resDescs);                                                                                 \
    }

#define CCU_FUNC_KERNEL_V2_TEST(testName, demoFunc, expectRegisterSuccess)                                            \
    TEST_F(HcommCcuControlApiTest, testName)                                                                          \
    {                                                                                                                 \
        CcuResult ccuRes = CcuResult::CCU_E_RESERVED;                                                                 \
        constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;                                                     \
        constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V2;                                       \
        (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);                                                            \
                                                                                                                      \
        HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};                                            \
        CreateCcuResDescsPair(resDescs, fakeCcuVersion);                                                              \
        CcuInsHandle insHandle{0};                                                                                    \
        constexpr uint32_t descNum = 2;                                                                               \
        ccuRes = HcommCcuInsCreate(resDescs, descNum, &insHandle);                                                    \
        EXPECT_EQ(ccuRes, CcuResult::CCU_SUCCESS);                                                                    \
                                                                                                                      \
        ccuRes = HcommCcuKernelRegisterStart(insHandle);                                                              \
        EXPECT_EQ(ccuRes, CcuResult::CCU_SUCCESS);                                                                    \
                                                                                                                      \
        int32_t dummyArg = 0;                                                                                         \
        CcuKernelArg kernelArg = static_cast<CcuKernelArg>(&dummyArg);                                                \
        const void* kernelArgs[] = {kernelArg};                                                                       \
        CcuKernelHandle kernelHandle{0};                                                                              \
        auto kernelFunc = reinterpret_cast<void*>(demoFunc);                                                          \
        constexpr uint32_t kernelArgNum = 1;                                                                          \
        constexpr uint32_t fakeDieId = 0;                                                                             \
        ccuRes = HcommCcuKernelRegister(                                                                              \
            insHandle, fakeDieId, const_cast<char*>(#demoFunc), kernelFunc, kernelArgs, kernelArgNum, &kernelHandle); \
        if (expectRegisterSuccess) {                                                                                  \
            EXPECT_EQ(ccuRes, CcuResult::CCU_SUCCESS);                                                                \
            ccuRes = HcommCcuKernelRegisterEnd(insHandle);                                                            \
            EXPECT_EQ(ccuRes, CcuResult::CCU_SUCCESS);                                                                \
        } else {                                                                                                      \
            EXPECT_NE(ccuRes, CcuResult::CCU_SUCCESS);                                                                \
        }                                                                                                             \
                                                                                                                      \
        ccuRes = HcommCcuInsDestroy(insHandle);                                                                       \
        EXPECT_EQ(ccuRes, CcuResult::CCU_SUCCESS);                                                                    \
        DestroyCcuResDescs(resDescs);                                                                                 \
    }
#endif // UT_HCOMM_CCU_CONTROL_API_COMMON_H

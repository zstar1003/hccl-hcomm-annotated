/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* UT for br_c00913534_set_device: 设备状态刷新回调（Register/Unregister/OnDeviceStateRefresh）
 * 覆盖 src/base_comm/hcomm_res_mgr.cc 中新增逻辑的全部分支：
 *   1. RegisterDeviceRefreshCallback
 *      - 注册失败分支：aclrtRegDeviceStateCallback 返回失败，仅打告警且不置位标志
 *      - 首次注册成功分支：regName="hcomm_refresh_device"、callback=OnDeviceStateRefresh、args=nullptr
 *      - 幂等分支：标志位已置位时早退，不再调用 aclrtRegDeviceStateCallback
 *   2. UnregisterDeviceRefreshCallback
 *      - 注销成功分支：以 callback=nullptr 调用 aclrtRegDeviceStateCallback
 *      - 注销失败分支：aclrt 返回失败，仅打告警
 *      - 注销不重置标志位：注销后再次 Register 仍幂等早退
 *   3. OnDeviceStateRefresh（通过注册时捕获的回调指针间接触发）
 *      - state 过滤分支：非 ACL_RT_DEVICE_STATE_SET_POST 早退，不调用任何 RTS API
 *      - hrtGetDeviceRefresh / hrtGetDevicePhyIdByIndex / hrtGetDeviceType 失败早退分支
 *      - 全流程成功分支
 *      - catch (const std::exception&) 与 catch (...) 异常捕获分支
 *      - deviceId 边界值（INT32_MIN 侧负值 / INT32_MAX）
 *
 * 说明：
 *   - 注册标志位 g_deviceRefreshCallbackRegistered 为 hcomm_res_mgr.cc 的文件级静态变量，
 *     进程内一旦置位无法复位。因此本文件必须注册为 hccl_utest_test_next_resource 的首个
 *     测试单元（src_to_test_list 中紧跟 main.cc），保证首个用例执行时标志位仍为 false，
 *     从而覆盖"注册失败"与"首次注册成功"分支；若执行顺序被调整，首个用例会自动退化为
 *     幂等路径验证并跳过严格断言。
 *   - 进程退出守护 RefreshCallbackUnregister 在进程退出时触发注销，无法在用例内直接构造，
 *     由每次 UT 进程退出时隐式执行。
 */

#include <iostream>
#include <stdexcept>
#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include "acl/acl_rt.h"
#include "adapter_rts_common.h"
#include "hccl_common.h"
#include "dtype_common.h"
#include "hcomm_res_mgr.h"

namespace {
/* ===== 注册/注销时捕获的状态（跨用例保留，首个成功注册时写入一次） ===== */
aclrtDeviceStateCallback g_capturedCallback = nullptr; // OnDeviceStateRefresh 函数指针
const char* g_capturedRegName = nullptr;               // 注册名
void* g_capturedArgs = reinterpret_cast<void*>(0x1);   // 哨兵值，区分"未捕获"与"args=nullptr"

/* ===== 每个用例执行前重置的计数/哨兵 ===== */
int g_regCallCount = 0;     // stub_regCapture 被调用次数（注册与注销共用）
int g_regFailCallCount = 0; // stub_regFail 被调用次数
int g_getRefreshCalls = 0;  // hrtGetDeviceRefresh mock 被调用次数
int g_getPhyIdCalls = 0;    // hrtGetDevicePhyIdByIndex mock 被调用次数
int g_getTypeCalls = 0;     // hrtGetDeviceType mock 被调用次数
// 注销时捕获的 callback 入参（哨兵初值，用于区分"未调用"与"callback=nullptr"）
aclrtDeviceStateCallback g_unregSeenCallback = reinterpret_cast<aclrtDeviceStateCallback>(0x1);

/* ===== aclrtRegDeviceStateCallback 的 mock stub ===== */
// 成功路径：记录调用次数与入参；注册时捕获回调指针，注销时记录 callback=nullptr
aclError stub_regCapture(const char* regName, aclrtDeviceStateCallback callback, void* args)
{
    g_regCallCount++;
    g_capturedRegName = regName;
    g_capturedArgs = args;
    if (callback != nullptr) {
        g_capturedCallback = callback;
    } else {
        g_unregSeenCallback = callback;
    }
    return ACL_SUCCESS;
}

// 失败路径：计数并返回错误码
aclError stub_regFail(const char* regName, aclrtDeviceStateCallback callback, void* args)
{
    (void)regName;
    (void)callback;
    (void)args;
    g_regFailCallCount++;
    return static_cast<aclError>(-1);
}

/* ===== hrtGetDeviceRefresh 的 mock stub ===== */
HcclResult stub_getDeviceRefreshSuccess(s32* deviceLogicId)
{
    g_getRefreshCalls++;
    if (deviceLogicId != nullptr) {
        *deviceLogicId = 0;
    }
    return HCCL_SUCCESS;
}

HcclResult stub_getDeviceRefreshFail(s32* deviceLogicId)
{
    g_getRefreshCalls++;
    (void)deviceLogicId;
    return HCCL_E_INTERNAL;
}

// 抛 std::exception，覆盖 OnDeviceStateRefresh 的 catch (const std::exception&) 分支
HcclResult stub_getDeviceRefreshThrowStd(s32* deviceLogicId)
{
    (void)deviceLogicId;
    throw std::runtime_error("stub std exception");
}

// 抛非 std::exception 类型，覆盖 OnDeviceStateRefresh 的 catch (...) 分支
HcclResult stub_getDeviceRefreshThrowInt(s32* deviceLogicId)
{
    (void)deviceLogicId;
    throw 42;
}

/* ===== hrtGetDevicePhyIdByIndex 的 mock stub ===== */
HcclResult stub_getPhyIdByIndexSuccess(u32 deviceLogicId, u32& devicePhyId, bool isRefresh)
{
    (void)deviceLogicId;
    (void)isRefresh;
    g_getPhyIdCalls++;
    devicePhyId = 0;
    return HCCL_SUCCESS;
}

HcclResult stub_getPhyIdByIndexFail(u32 deviceLogicId, u32& devicePhyId, bool isRefresh)
{
    (void)deviceLogicId;
    (void)devicePhyId;
    (void)isRefresh;
    g_getPhyIdCalls++;
    return HCCL_E_INTERNAL;
}

/* ===== hrtGetDeviceType 的 mock stub ===== */
HcclResult stub_getDeviceTypeSuccess(DevType& devType)
{
    g_getTypeCalls++;
    devType = DevType::DEV_TYPE_910;
    return HCCL_SUCCESS;
}

HcclResult stub_getDeviceTypeFail(DevType& devType)
{
    (void)devType;
    g_getTypeCalls++;
    return HCCL_E_INTERNAL;
}
} // namespace

class DeviceRefreshCallbackTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DeviceRefreshCallbackTest set up." << std::endl; }
    static void TearDownTestCase() { std::cout << "DeviceRefreshCallbackTest tear down." << std::endl; }

    virtual void SetUp()
    {
        g_regCallCount = 0;
        g_regFailCallCount = 0;
        g_getRefreshCalls = 0;
        g_getPhyIdCalls = 0;
        g_getTypeCalls = 0;
        g_unregSeenCallback = reinterpret_cast<aclrtDeviceStateCallback>(0x1);
    }
    virtual void TearDown() { GlobalMockObject::verify(); }
};

/* ===== 1. RegisterDeviceRefreshCallback 测试 ===== */

// 用例1.1：注册失败分支。必须为本套件首个用例（全进程首个触发注册的用例）。
// 测试目的：aclrtRegDeviceStateCallback 返回失败时仅打告警返回，不置位标志位。
// 输入设置：mock aclrtRegDeviceStateCallback 返回 -1。
// 预期结果：失败 stub 被调用 1 次；随后以成功 mock 再次注册时真正执行注册
//           （失败分支未置位标志，成功分支得以执行并捕获回调指针）。
TEST_F(DeviceRefreshCallbackTest, Ut_When_RegisterDeviceRefreshCallbackFail_Expect_FlagNotSet)
{
    MOCKER(aclrtRegDeviceStateCallback).stubs().will(invoke(stub_regFail));
    hcomm::HcommResMgr::RegisterDeviceRefreshCallback();
    GlobalMockObject::verify();

    if (g_regFailCallCount == 0) {
        // 标志位已被先行用例置位（测试执行顺序被调整），失败分支不可达，退化为幂等路径验证
        std::cout << "register flag already set, fail-branch not reachable, skip strict asserts." << std::endl;
        return;
    }
    EXPECT_EQ(g_regFailCallCount, 1);

    // 注册失败不置位标志：本次注册应真正调用 aclrtRegDeviceStateCallback 并捕获回调
    MOCKER(aclrtRegDeviceStateCallback).stubs().will(invoke(stub_regCapture));
    hcomm::HcommResMgr::RegisterDeviceRefreshCallback();
    GlobalMockObject::verify();
    EXPECT_EQ(g_regCallCount, 1);
    EXPECT_NE(g_capturedCallback, nullptr);
}

// 用例1.2：首次注册成功分支的入参校验（捕获动作由用例1.1的第二次注册完成）。
// 测试目的：注册时传入正确的 regName / callback / args。
// 预期结果：regName == "hcomm_refresh_device"，callback 非空（OnDeviceStateRefresh），args == nullptr。
TEST_F(DeviceRefreshCallbackTest, Ut_When_RegisterDeviceRefreshCallbackSuccess_Expect_CorrectArgs)
{
    ASSERT_NE(g_capturedCallback, nullptr);
    EXPECT_STREQ(g_capturedRegName, "hcomm_refresh_device");
    EXPECT_EQ(g_capturedArgs, nullptr);
}

// 用例1.3：幂等分支。
// 测试目的：标志位已置位时再次注册早退，不调用 aclrtRegDeviceStateCallback。
// 输入设置：mock 返回失败（若被调用则会计数，用于验证未被调用）。
// 预期结果：失败 stub 调用次数为 0，已捕获的回调指针不变。
TEST_F(DeviceRefreshCallbackTest, Ut_When_RegisterTwice_Expect_IdempotentWithoutAclrtCall)
{
    aclrtDeviceStateCallback before = g_capturedCallback;
    MOCKER(aclrtRegDeviceStateCallback).stubs().will(invoke(stub_regFail));
    hcomm::HcommResMgr::RegisterDeviceRefreshCallback();
    GlobalMockObject::verify();
    EXPECT_EQ(g_regFailCallCount, 0);
    EXPECT_EQ(g_capturedCallback, before);
}

/* ===== 2. UnregisterDeviceRefreshCallback 测试 ===== */

// 用例2.1：注销成功分支。
// 测试目的：注销时以 regName="hcomm_refresh_device"、callback=nullptr 调用 aclrtRegDeviceStateCallback。
// 预期结果：成功 stub 被调用 1 次，捕获的 callback 入参为 nullptr。
TEST_F(DeviceRefreshCallbackTest, Ut_When_UnregisterDeviceRefreshCallback_Expect_NullCallbackPassed)
{
    MOCKER(aclrtRegDeviceStateCallback).stubs().will(invoke(stub_regCapture));
    hcomm::HcommResMgr::UnregisterDeviceRefreshCallback();
    GlobalMockObject::verify();
    EXPECT_EQ(g_regCallCount, 1);
    EXPECT_EQ(g_unregSeenCallback, nullptr);
}

// 用例2.2：注销失败分支。
// 测试目的：aclrtRegDeviceStateCallback 注销返回失败时仅打告警，不影响流程。
// 预期结果：失败 stub 被调用 1 次，进程不崩溃。
TEST_F(DeviceRefreshCallbackTest, Ut_When_UnregisterDeviceRefreshCallbackFail_Expect_NoCrash)
{
    MOCKER(aclrtRegDeviceStateCallback).stubs().will(invoke(stub_regFail));
    hcomm::HcommResMgr::UnregisterDeviceRefreshCallback();
    GlobalMockObject::verify();
    EXPECT_EQ(g_regFailCallCount, 1);
}

// 用例2.3：重复注销。
// 测试目的：注销接口无状态保护，重复调用不崩溃。
// 预期结果：aclrtRegDeviceStateCallback 被调用 2 次，进程不崩溃。
TEST_F(DeviceRefreshCallbackTest, Ut_When_UnregisterTwice_Expect_NoCrash)
{
    MOCKER(aclrtRegDeviceStateCallback).stubs().will(invoke(stub_regCapture));
    hcomm::HcommResMgr::UnregisterDeviceRefreshCallback();
    hcomm::HcommResMgr::UnregisterDeviceRefreshCallback();
    GlobalMockObject::verify();
    EXPECT_EQ(g_regCallCount, 2);
}

// 用例2.4：注销不重置标志位。
// 测试目的：UnregisterDeviceRefreshCallback 注销后，标志位仍为 true，再次注册幂等早退。
// 预期结果：失败 stub 调用次数为 0（注册未真正执行）。
TEST_F(DeviceRefreshCallbackTest, Ut_When_RegisterAfterUnregister_Expect_StillIdempotent)
{
    MOCKER(aclrtRegDeviceStateCallback).stubs().will(invoke(stub_regFail));
    hcomm::HcommResMgr::RegisterDeviceRefreshCallback();
    GlobalMockObject::verify();
    EXPECT_EQ(g_regFailCallCount, 0);
}

/* ===== 3. OnDeviceStateRefresh 间接测试（通过注册时捕获的回调指针触发） ===== */

// 用例3.1：state 过滤分支。
// 测试目的：state 非 ACL_RT_DEVICE_STATE_SET_POST 时早退，不调用任何 RTS API。
// 输入设置：三个非 SET_POST 状态（SET_PRE / RESET_PRE / RESET_POST）。
// 预期结果：hrtGetDeviceRefresh mock 调用次数为 0。
TEST_F(DeviceRefreshCallbackTest, Ut_When_OnRefreshStateNotSetPost_Expect_EarlyReturnWithoutRtsCall)
{
    ASSERT_NE(g_capturedCallback, nullptr);
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_getDeviceRefreshSuccess));
    g_capturedCallback(0, ACL_RT_DEVICE_STATE_SET_PRE, nullptr);
    g_capturedCallback(0, ACL_RT_DEVICE_STATE_RESET_PRE, nullptr);
    g_capturedCallback(0, ACL_RT_DEVICE_STATE_RESET_POST, nullptr);
    GlobalMockObject::verify();
    EXPECT_EQ(g_getRefreshCalls, 0);
}

// 用例3.2：hrtGetDeviceRefresh 失败分支。
// 测试目的：获取刷新设备逻辑 ID 失败时打告警并早退，后续 RTS API 不被调用。
// 预期结果：仅 hrtGetDeviceRefresh 被调用 1 次，后两个 API 调用次数为 0。
TEST_F(DeviceRefreshCallbackTest, Ut_When_OnRefreshGetDeviceRefreshFail_Expect_EarlyReturn)
{
    ASSERT_NE(g_capturedCallback, nullptr);
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_getDeviceRefreshFail));
    g_capturedCallback(0, ACL_RT_DEVICE_STATE_SET_POST, nullptr);
    GlobalMockObject::verify();
    EXPECT_EQ(g_getRefreshCalls, 1);
    EXPECT_EQ(g_getPhyIdCalls, 0);
    EXPECT_EQ(g_getTypeCalls, 0);
}

// 用例3.3：hrtGetDevicePhyIdByIndex 失败分支。
// 测试目的：逻辑 ID 转物理 ID 失败时打告警并早返回，hrtGetDeviceType 不被调用。
// 预期结果：前两个 API 各调用 1 次，hrtGetDeviceType 调用次数为 0。
TEST_F(DeviceRefreshCallbackTest, Ut_When_OnRefreshGetPhyIdFail_Expect_EarlyReturn)
{
    ASSERT_NE(g_capturedCallback, nullptr);
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_getDeviceRefreshSuccess));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(invoke(stub_getPhyIdByIndexFail));
    g_capturedCallback(0, ACL_RT_DEVICE_STATE_SET_POST, nullptr);
    GlobalMockObject::verify();
    EXPECT_EQ(g_getRefreshCalls, 1);
    EXPECT_EQ(g_getPhyIdCalls, 1);
    EXPECT_EQ(g_getTypeCalls, 0);
}

// 用例3.4：hrtGetDeviceType 失败分支。
// 测试目的：获取设备类型失败时打告警并早返回，不走成功日志分支。
// 预期结果：三个 API 均被调用 1 次。
TEST_F(DeviceRefreshCallbackTest, Ut_When_OnRefreshGetDeviceTypeFail_Expect_EarlyReturn)
{
    ASSERT_NE(g_capturedCallback, nullptr);
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_getDeviceRefreshSuccess));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(invoke(stub_getPhyIdByIndexSuccess));
    MOCKER(hrtGetDeviceType).stubs().will(invoke(stub_getDeviceTypeFail));
    g_capturedCallback(0, ACL_RT_DEVICE_STATE_SET_POST, nullptr);
    GlobalMockObject::verify();
    EXPECT_EQ(g_getRefreshCalls, 1);
    EXPECT_EQ(g_getPhyIdCalls, 1);
    EXPECT_EQ(g_getTypeCalls, 1);
}

// 用例3.5：全流程成功分支（正常流程）。
// 测试目的：三个 RTS API 均成功时走到成功日志分支，完整执行刷新流程。
// 预期结果：三个 API 各被调用 1 次，进程不崩溃。
TEST_F(DeviceRefreshCallbackTest, Ut_When_OnRefreshAllApiSuccess_Expect_CompleteFlow)
{
    ASSERT_NE(g_capturedCallback, nullptr);
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_getDeviceRefreshSuccess));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(invoke(stub_getPhyIdByIndexSuccess));
    MOCKER(hrtGetDeviceType).stubs().will(invoke(stub_getDeviceTypeSuccess));
    g_capturedCallback(1, ACL_RT_DEVICE_STATE_SET_POST, nullptr);
    GlobalMockObject::verify();
    EXPECT_EQ(g_getRefreshCalls, 1);
    EXPECT_EQ(g_getPhyIdCalls, 1);
    EXPECT_EQ(g_getTypeCalls, 1);
}

// 用例3.6：边界条件——deviceId 极值。
// 测试目的：deviceId 为负值/INT32_MAX 时刷新流程正常执行，不崩溃。
// 预期结果：三个 API 各被调用 2 次（两种边界各触发一次）。
TEST_F(DeviceRefreshCallbackTest, Ut_When_OnRefreshBoundaryDeviceId_Expect_NoCrash)
{
    ASSERT_NE(g_capturedCallback, nullptr);
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_getDeviceRefreshSuccess));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(invoke(stub_getPhyIdByIndexSuccess));
    MOCKER(hrtGetDeviceType).stubs().will(invoke(stub_getDeviceTypeSuccess));
    g_capturedCallback(-1, ACL_RT_DEVICE_STATE_SET_POST, nullptr);
    g_capturedCallback(0x7FFFFFFF, ACL_RT_DEVICE_STATE_SET_POST, nullptr);
    GlobalMockObject::verify();
    EXPECT_EQ(g_getRefreshCalls, 2);
    EXPECT_EQ(g_getPhyIdCalls, 2);
    EXPECT_EQ(g_getTypeCalls, 2);
}

// 用例3.7：异常场景——RTS API 抛 std::exception。
// 测试目的：覆盖 OnDeviceStateRefresh 的 catch (const std::exception&) 分支，异常被捕获仅打告警。
// 预期结果：进程不崩溃（异常未逃逸出回调）。
TEST_F(DeviceRefreshCallbackTest, Ut_When_OnRefreshThrowStdException_Expect_CaughtNoCrash)
{
    ASSERT_NE(g_capturedCallback, nullptr);
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_getDeviceRefreshThrowStd));
    g_capturedCallback(0, ACL_RT_DEVICE_STATE_SET_POST, nullptr);
}

// 用例3.8：异常场景——RTS API 抛非 std::exception 类型。
// 测试目的：覆盖 OnDeviceStateRefresh 的 catch (...) 分支，未知异常被捕获仅打告警。
// 预期结果：进程不崩溃（异常未逃逸出回调）。
TEST_F(DeviceRefreshCallbackTest, Ut_When_OnRefreshThrowUnknownException_Expect_CaughtNoCrash)
{
    ASSERT_NE(g_capturedCallback, nullptr);
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(stub_getDeviceRefreshThrowInt));
    g_capturedCallback(0, ACL_RT_DEVICE_STATE_SET_POST, nullptr);
}

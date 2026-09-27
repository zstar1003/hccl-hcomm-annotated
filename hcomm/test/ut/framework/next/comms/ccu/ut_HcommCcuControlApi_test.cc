/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>

#include "ut_HcommCcuControlApi_common.h"

#define private public
#define protected public
#include "ccu_kernel_impl/ccu_var_add_simple_demo.h"
#include "ccu_kernel_impl/ccu_loop_add_demo.h"
#include "ccu_kernel_impl/ccu_func_call_demo.h"
#include "ccu_kernel_impl/ccu_jump_demo.h"
#include "ccu_kernel_impl/ccu_micro_sim.h"
#include "ccu_kernel_impl/ccu_reduce_scatter_mesh1d_demo.h"
#include "ccu_kernel_impl/ccu_reduce_scatter_mesh1d_a6_demo.h"
#include "ccu_kernel_impl/ccu_groupcopy_demo.h"
#include "ccu_kernel_impl/ccu_var_acquire_demo.h"
#include "ccu_kernel_impl/ccu_imm_casc_demo.h"
#include "ccu_urma_channel.h"
#undef protected
#undef private

CCU_FUNC_KERNEL_TEST(Ut_FuncCallBasic_Expect_Success, CcuFuncCallBasicDemoKernel, true)
CCU_FUNC_KERNEL_TEST(Ut_FuncCallReuse_Expect_Success, CcuFuncCallReuseDemoKernel, true)
CCU_FUNC_KERNEL_TEST(Ut_FuncCallMultiArg_Expect_Success, CcuFuncCallMultiArgDemoKernel, true)
CCU_FUNC_KERNEL_TEST(Ut_FuncCallInLoop_Expect_Fail, CcuFuncCallInLoopInvalidDemoKernel, false)
CCU_FUNC_KERNEL_TEST(Ut_FuncCallNested_Expect_Fail, CcuFuncCallNestedInvalidDemoKernel, false)
CCU_FUNC_KERNEL_TEST(Ut_IfInLoop_Expect_Fail, CcuIfInLoopInvalidDemoKernel, false)
CCU_FUNC_KERNEL_TEST(Ut_NotifyRecordInLoop_Expect_Fail, CcuNotifyRecordInLoopInvalidDemoKernel, false)
CCU_FUNC_KERNEL_TEST(Ut_WriteVarWithNotifyInLoop_Expect_Fail, CcuWriteVarWithNotifyInLoopInvalidDemoKernel, false)
CCU_FUNC_KERNEL_TEST(Ut_EventRecordTagInLoop_Expect_Fail, CcuEventRecordTagInLoopInvalidDemoKernel, false)
CCU_FUNC_KERNEL_TEST(Ut_EventRecordInLoop_Expect_Fail, CcuEventRecordInLoopInvalidDemoKernel, false)
CCU_FUNC_KERNEL_TEST(Ut_A5MixedLoopCount_Expect_Success, CcuA5MixedLoopCountDemoKernel, true)
CCU_FUNC_KERNEL_TEST(Ut_LoopCfgDemo_Expect_Success, CcuLoopCfgDemoKernel, true)
CCU_FUNC_KERNEL_V2_TEST(Ut_LoadAddImm_Expect_Success, CcuLoadAddImmDemoKernel, true)
CCU_FUNC_KERNEL_V2_TEST(Ut_AddImmStore_Expect_Success, CcuAddImmStoreDemoKernel, true)
CCU_FUNC_KERNEL_V2_TEST(Ut_WriteVarAtomicAdd_Expect_Success, CcuWriteVarAtomicAddDemoKernel, true)
CCU_FUNC_KERNEL_V2_TEST(Ut_WriteWithCascCntInc_Expect_Success, CcuWriteWithCascCntIncDemoKernel, true)

namespace {
void SetupCascCntTest(
    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM], CcuInsHandle& insHandle, HcommCcuCascCntHandle& cntHandle)
{
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V2;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    EXPECT_EQ(HcommCcuInsResDescSetNum(resDescs[0], HCOMM_CCU_RES_TYPE_CASC_CNT, 1), CcuResult::CCU_SUCCESS);
    EXPECT_EQ(HcommCcuInsCreate(resDescs, hcomm::CCU_MAX_IODIE_NUM, &insHandle), CcuResult::CCU_SUCCESS);
    EXPECT_EQ(HcommCcuCascCntAlloc(insHandle, 0, &cntHandle), CcuResult::CCU_SUCCESS);
    EXPECT_NE(cntHandle, 0U);
}
} // namespace

TEST_F(HcommCcuControlApiTest, Ut_CascCntWait_Expect_Success)
{
    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CcuInsHandle insHandle{0};
    HcommCcuCascCntHandle cntHandle = 0;
    SetupCascCntTest(resDescs, insHandle, cntHandle);

    CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    CcuCascCntDemoArg demoArg{};
    demoArg.cntHandle = cntHandle;
    demoArg.tgtValue = 1024;
    CcuKernelArg kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void* kernelArgs[] = {kernelArg};
    auto kernelFunc = reinterpret_cast<void*>(CcuCascCntWaitDemoKernel);
    const char* kernelFuncName = "CcuCascCntWaitDemoKernel";
    CcuKernelHandle kernelHandle{0};
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(
        insHandle, fakeDieId, const_cast<char*>(kernelFuncName), kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_CascCntClear_Expect_Success)
{
    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CcuInsHandle insHandle{0};
    HcommCcuCascCntHandle cntHandle = 0;
    SetupCascCntTest(resDescs, insHandle, cntHandle);

    CcuResult ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    CcuCascCntDemoArg demoArg{};
    demoArg.cntHandle = cntHandle;
    CcuKernelArg kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void* kernelArgs[] = {kernelArg};
    CcuKernelHandle kernelHandle{0};
    auto kernelFunc = reinterpret_cast<void*>(CcuCascCntClearDemoKernel);
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(
        insHandle, fakeDieId, const_cast<char*>("CcuCascCntClearDemoKernel"), kernelFunc, kernelArgs, kernelArgNum,
        &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_LoopObjectApi_Expect_Success)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    CcuLoopAddKernelArg demoArg{};
    demoArg.numA = 3;
    demoArg.numB = 4;
    CcuKernelArg kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void* kernelArgs[] = {kernelArg};
    auto kernelFunc = reinterpret_cast<void*>(CcuLoopAddDemoKernel);

    char* kernelFuncName = "CcuLoopAddDemoKernel";
    CcuKernelHandle kernelHandle{0};
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(
        insHandle, fakeDieId, kernelFuncName, kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_LoopV2GroupApi_When_V2_Expect_Success)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V2;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    int32_t dummyArg = 0;
    CcuKernelArg kernelArg = static_cast<CcuKernelArg>(&dummyArg);
    const void* kernelArgs[] = {kernelArg};
    auto kernelFunc = reinterpret_cast<void*>(CcuV2LoopGroupDemoKernel);
    char* kernelFuncName = "CcuV2LoopGroupDemoKernel";
    CcuKernelHandle kernelHandle{0};
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(
        insHandle, fakeDieId, kernelFuncName, kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_A6MixedLoopCount_When_V2_Expect_Success)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V2;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    int32_t dummyArg = 0;
    CcuKernelArg kernelArg = static_cast<CcuKernelArg>(&dummyArg);
    const void* kernelArgs[] = {kernelArg};
    auto kernelFunc = reinterpret_cast<void*>(CcuA6MixedLoopCountDemoKernel);
    char* kernelFuncName = "CcuA6MixedLoopCountDemoKernel";
    CcuKernelHandle kernelHandle{0};
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(
        insHandle, fakeDieId, kernelFuncName, kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_LoopV2GroupApi_When_V1_Expect_Fail)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    int32_t dummyArg = 0;
    CcuKernelArg kernelArg = static_cast<CcuKernelArg>(&dummyArg);
    const void* kernelArgs[] = {kernelArg};
    auto kernelFunc = reinterpret_cast<void*>(CcuV2LoopGroupDemoKernel);
    char* kernelFuncName = "CcuV2LoopGroupDemoKernel";
    CcuKernelHandle kernelHandle{0};
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(
        insHandle, fakeDieId, kernelFuncName, kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_NE(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_LoopCompatGroupApi_When_V2_Expect_Success)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V2;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    int32_t dummyArg = 0;
    CcuKernelArg kernelArg = static_cast<CcuKernelArg>(&dummyArg);
    const void* kernelArgs[] = {kernelArg};
    auto kernelFunc = reinterpret_cast<void*>(CcuV2CompatLoopGroupDemoKernel);
    char* kernelFuncName = "CcuV2CompatLoopGroupDemoKernel";
    CcuKernelHandle kernelHandle{0};
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(
        insHandle, fakeDieId, kernelFuncName, kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_LoopConfigGroupApi_When_V2_Expect_Success)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V2;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    int32_t dummyArg = 0;
    CcuKernelArg kernelArg = static_cast<CcuKernelArg>(&dummyArg);
    const void* kernelArgs[] = {kernelArg};
    auto kernelFunc = reinterpret_cast<void*>(CcuV2ConfigLoopGroupDemoKernel);
    char* kernelFuncName = "CcuV2ConfigLoopGroupDemoKernel";
    CcuKernelHandle kernelHandle{0};
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(
        insHandle, fakeDieId, kernelFuncName, kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_LoopMixedGroupApi_When_V2_Expect_Success)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V2;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    int32_t dummyArg = 0;
    CcuKernelArg kernelArg = static_cast<CcuKernelArg>(&dummyArg);
    const void* kernelArgs[] = {kernelArg};
    auto kernelFunc = reinterpret_cast<void*>(CcuV2MixedLoopGroupDemoKernel);
    char* kernelFuncName = "CcuV2MixedLoopGroupDemoKernel";
    CcuKernelHandle kernelHandle{0};
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(
        insHandle, fakeDieId, kernelFuncName, kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

// 版本化 cfg 全路径(Loop(CcuLoopCfg)+LoopGroup(CcuLoopGroupCfg)) demo,V2 下建组成功
TEST_F(HcommCcuControlApiTest, Ut_LoopGroupCfgApi_When_V2_Expect_Success)
{
    // 整体打桩，处理ccu资源
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V2;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    // ccuInstance构建
    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    int32_t dummyArg = 0;
    CcuKernelArg kernelArg = static_cast<CcuKernelArg>(&dummyArg);
    const void* kernelArgs[] = {kernelArg};
    auto kernelFunc = reinterpret_cast<void*>(CcuV2LoopGroupCfgDemoKernel);
    char* kernelFuncName = "CcuV2LoopGroupCfgDemoKernel";
    CcuKernelHandle kernelHandle{0};
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(
        insHandle, fakeDieId, kernelFuncName, kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

// 版本化 cfg C ABI 的 strict 校验:非法版本头(magic 未初始化)被拒绝,注册失败。
CCU_FUNC_KERNEL_TEST(Ut_LoopGroupCreateCfg_BadHeader_Expect_Fail, CcuLoopGroupCreateCfgBadHeaderDemoKernel, false)
CCU_FUNC_KERNEL_TEST(Ut_LoopGroupAddLoopCfg_BadHeader_Expect_Fail, CcuLoopGroupAddLoopCfgBadHeaderDemoKernel, false)

static void
RegisterLoopGroupWithMaxLoopNum(hcomm::CcuVersion ccuVersion, uint32_t maxLoopNum, bool expectRegisterSuccess)
{
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    (void)MockCcuDeviceEnv(fakeDevId, ccuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, ccuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    EXPECT_EQ(HcommCcuInsCreate(resDescs, descNum, &insHandle), CcuResult::CCU_SUCCESS);
    EXPECT_EQ(HcommCcuKernelRegisterStart(insHandle), CcuResult::CCU_SUCCESS);

    CcuLoopGroupMaxLoopNumKernelArg demoArg{};
    demoArg.maxLoopNum = maxLoopNum;
    CcuKernelArg kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void* kernelArgs[] = {kernelArg};
    CcuKernelHandle kernelHandle{0};
    auto kernelFunc = reinterpret_cast<void*>(CcuLoopGroupMaxLoopNumDemoKernel);
    CcuResult ccuRet = HcommCcuKernelRegister(
        insHandle, 0, const_cast<char*>("CcuLoopGroupMaxLoopNumDemoKernel"), kernelFunc, kernelArgs, 1, &kernelHandle);
    if (expectRegisterSuccess) {
        EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
        EXPECT_EQ(HcommCcuKernelRegisterEnd(insHandle), CcuResult::CCU_SUCCESS);
    } else {
        EXPECT_NE(ccuRet, CcuResult::CCU_SUCCESS);
    }
    EXPECT_EQ(HcommCcuInsDestroy(insHandle), CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_LoopGroupCreate_HugeMaxLoopNum_When_V1_Expect_Fail)
{
    RegisterLoopGroupWithMaxLoopNum(hcomm::CcuVersion::CCU_V1, UINT32_MAX, false);
}

TEST_F(HcommCcuControlApiTest, Ut_LoopGroupCreate_HugeMaxLoopNum_When_V2_Expect_Fail)
{
    RegisterLoopGroupWithMaxLoopNum(hcomm::CcuVersion::CCU_V2, UINT32_MAX, false);
}

TEST_F(HcommCcuControlApiTest, Ut_LoopGroupCreate_MaxLoopNumAtV1Limit_Expect_Success)
{
    RegisterLoopGroupWithMaxLoopNum(hcomm::CcuVersion::CCU_V1, 128, true);
}

TEST_F(HcommCcuControlApiTest, Ut_LoopGroupCreate_MaxLoopNumOverV1Limit_Expect_Fail)
{
    RegisterLoopGroupWithMaxLoopNum(hcomm::CcuVersion::CCU_V1, 129, false);
}

TEST_F(HcommCcuControlApiTest, Ut_LoopGroupCreate_MaxLoopNumOverV2Limit_Expect_Fail)
{
    RegisterLoopGroupWithMaxLoopNum(hcomm::CcuVersion::CCU_V2, 513, false);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelRegister_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    // 整体打桩，处理ccu资源
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;

    // ccuInstance构建
    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CcuInsHandle insHandle{0};
    SetupV1CcuInstance(resDescs, insHandle, fakeDevId);

    // 建链流程打桩
    constexpr auto commEngine = CommEngine::COMM_ENGINE_CCU;
    const auto& handlePair = MockCcuChannelConnectDefault(fakeDevId, commEngine);

    // 构造CcuKernel实现
    auto demoFunc = CcuLoadStoreDemoKernel;
    CcuVarAddKernelArg demoArg{};
    demoArg.numA = 1;
    demoArg.numB = 2;
    demoArg.channelHandle = handlePair.second;

    auto kernelFunc = reinterpret_cast<void*>(demoFunc);
    auto kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void* kernelArgs[] = {kernelArg};

    // 重置CCU资源
    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    // kernel注册 -> 翻译 -> 申请流 -> 下发(空 taskArgs) -> 清理(析构有时序要求)
    char* kernelFuncName = "ccu_var_add_simple_demo";
    RegisterLaunchAndCleanupSingleArgKernel(
        insHandle, kernelFuncName, kernelFunc, kernelArgs, handlePair, resDescs, commEngine, 0);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelRegister_When_ChannelDieDiffersFromInputDie_Expect_ReturnIsCCU_E_PARA)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;

    // ccuInstance构建
    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CcuInsHandle insHandle{0};
    SetupV1CcuInstance(resDescs, insHandle, fakeDevId);

    constexpr auto commEngine = CommEngine::COMM_ENGINE_CCU;
    constexpr uint32_t srcIp = 167772383; // mock EID 对应 die0
    constexpr uint32_t dstIp = 0x87654321;
    const auto& handlePair = MockCcuChannelConnect(fakeDevId, 1, srcIp, dstIp, commEngine);

    CcuVarAddKernelArg demoArg{};
    demoArg.numA = 1;
    demoArg.numB = 2;
    demoArg.channelHandle = handlePair.second;
    const void* kernelArgs[] = {static_cast<CcuKernelArg>(&demoArg)};

    ASSERT_EQ(HcommCcuKernelRegisterStart(insHandle), CcuResult::CCU_SUCCESS);
    constexpr uint32_t inputDieId = 1;
    CcuKernelHandle kernelHandle{0};
    EXPECT_EQ(
        HcommCcuKernelRegister(
            insHandle, inputDieId, "ccu_alloc_demo", reinterpret_cast<void*>(CcuAllocDemoKernel), kernelArgs, 1,
            &kernelHandle),
        CcuResult::CCU_E_PARA);
    EXPECT_EQ(kernelHandle, 0U);
    EXPECT_EQ(HcommCcuKernelRegisterEnd(insHandle), CcuResult::CCU_SUCCESS);

    MockChannelDestroy(handlePair);
    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelAddrArithV2_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 4;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V2;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    // ccuInstance构建
    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    constexpr auto commEngine = CommEngine::COMM_ENGINE_CCU;
    constexpr uint32_t srcDevPhyId = fakeDevId;
    constexpr uint32_t dstDevPhyId = 1;
    constexpr uint32_t srcIp = 167772383;
    constexpr uint32_t dstIp = 0x87654321;
    const auto& handlePair = MockCcuChannelConnect(srcDevPhyId, dstDevPhyId, srcIp, dstIp, commEngine);

    auto demoFunc = CcuAddrArithV2DemoKernel;
    CcuVarAddKernelArg demoArg{};
    demoArg.channelHandle = handlePair.second;
    auto kernelFunc = reinterpret_cast<void*>(demoFunc);
    auto kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void* kernelArgs[] = {kernelArg};

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    char* kernelFuncName = "ccu_addr_arith_v2_demo";
    RegisterLaunchAndCleanupSingleArgKernel(
        insHandle, kernelFuncName, kernelFunc, kernelArgs, handlePair, resDescs, commEngine, 0);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelDoWhile_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 3;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    // ccuInstance构建
    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    constexpr auto commEngine = CommEngine::COMM_ENGINE_CCU;
    constexpr uint32_t srcDevPhyId = fakeDevId;
    constexpr uint32_t dstDevPhyId = 1;
    constexpr uint32_t srcIp = 167772383;
    constexpr uint32_t dstIp = 0x87654321;
    const auto& handlePair = MockCcuChannelConnect(srcDevPhyId, dstDevPhyId, srcIp, dstIp, commEngine);

    auto demoFunc = CcuDoWhileWhileDemoKernel;
    CcuDoWhileWhileDemoKernelArg demoArg{};
    demoArg.loopCount = 5;
    auto kernelFunc = reinterpret_cast<void*>(demoFunc);
    auto kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void* kernelArgs[] = {kernelArg};

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    char* kernelFuncName = "ccu_do_while_while_demo";
    RegisterAndCleanupSingleArgKernel(insHandle, kernelFuncName, kernelFunc, kernelArgs, handlePair, resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelNestedIfOuterElse_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 4;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    constexpr auto commEngine = CommEngine::COMM_ENGINE_CCU;
    constexpr uint32_t srcDevPhyId = fakeDevId;
    constexpr uint32_t dstDevPhyId = 1;
    constexpr uint32_t srcIp = 167772383;
    constexpr uint32_t dstIp = 0x87654321;
    const auto& handlePair = MockCcuChannelConnect(srcDevPhyId, dstDevPhyId, srcIp, dstIp, commEngine);

    auto demoFunc = CcuNestedIfOuterElseDemoKernel;
    CcuNestedIfOuterElseDemoKernelArg demoArg{};
    demoArg.outerVal = 1;
    demoArg.outerExpected = 1;
    demoArg.innerVal = 2;
    demoArg.innerExpected = 2;
    auto kernelFunc = reinterpret_cast<void*>(demoFunc);
    auto kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void* kernelArgs[] = {kernelArg};

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    char* kernelFuncName = "ccu_nested_if_outer_else_demo";
    RegisterAndCleanupSingleArgKernel(insHandle, kernelFuncName, kernelFunc, kernelArgs, handlePair, resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelNestedIfInnerElse_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 5;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    constexpr auto commEngine = CommEngine::COMM_ENGINE_CCU;
    constexpr uint32_t srcDevPhyId = fakeDevId;
    constexpr uint32_t dstDevPhyId = 1;
    constexpr uint32_t srcIp = 167772383;
    constexpr uint32_t dstIp = 0x87654321;
    const auto& handlePair = MockCcuChannelConnect(srcDevPhyId, dstDevPhyId, srcIp, dstIp, commEngine);

    auto demoFunc = CcuNestedIfInnerElseDemoKernel;
    CcuNestedIfInnerElseDemoKernelArg demoArg{};
    demoArg.outerVal = 1;
    demoArg.outerExpected = 1;
    demoArg.innerVal = 2;
    demoArg.innerExpected = 2;
    auto kernelFunc = reinterpret_cast<void*>(demoFunc);
    auto kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void* kernelArgs[] = {kernelArg};

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    char* kernelFuncName = "ccu_nested_if_inner_else_demo";
    RegisterAndCleanupSingleArgKernel(insHandle, kernelFuncName, kernelFunc, kernelArgs, handlePair, resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelDoWhileUnified_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 6;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    constexpr auto commEngine = CommEngine::COMM_ENGINE_CCU;
    constexpr uint32_t srcDevPhyId = fakeDevId;
    constexpr uint32_t dstDevPhyId = 1;
    constexpr uint32_t srcIp = 167772383;
    constexpr uint32_t dstIp = 0x87654321;
    const auto& handlePair = MockCcuChannelConnect(srcDevPhyId, dstDevPhyId, srcIp, dstIp, commEngine);

    auto demoFunc = CcuDoWhileUnifiedDemoKernel;
    CcuDoWhileUnifiedDemoKernelArg demoArg{};
    demoArg.loopCount = 5;
    auto kernelFunc = reinterpret_cast<void*>(demoFunc);
    auto kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void* kernelArgs[] = {kernelArg};

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    char* kernelFuncName = "ccu_do_while_unified_demo";
    RegisterAndCleanupSingleArgKernel(insHandle, kernelFuncName, kernelFunc, kernelArgs, handlePair, resDescs);
}

// A5(CCU_V1): CCU_IF 的 targetVar 共用一个 Xn，验证所有跳转 rep 的 targetInstrId 指向同一个 Xn
TEST_F(HcommCcuControlApiTest, Ut_CcuIfSharedTargetVar_When_V1_Expect_XnReduced)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 10;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    int32_t fakeDeviceLogicId = MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    CcuIfSharedVarDemoKernelArg demoArg{};
    demoArg.value = 0;
    demoArg.expect = 0;
    CcuKernelArg kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void* kernelArgs[] = {kernelArg};
    auto kernelFunc = reinterpret_cast<void*>(CcuIfSharedVarDemoKernel);
    char* kernelFuncName = "ccu_if_shared_var_demo";
    CcuKernelHandle kernelHandle{0};
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(
        insHandle, fakeDieId, kernelFuncName, kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    auto* kernel = hcomm::CcuKernelMgr::GetInstance(fakeDeviceLogicId).GetKernel(kernelHandle);
    ASSERT_NE(kernel, nullptr);

    // sharedJumpTargetVar_ 应已初始化（A5 共用路径生效）
    EXPECT_TRUE(kernel->sharedJumpTargetVar_ != nullptr);

    // 收集所有条件跳转 rep 的 targetInstrId，验证它们指向同一个 Xn
    uint32_t jumpRepCount = 0;
    std::set<uint16_t> jumpTargetIds;
    for (const auto& rep : kernel->GetRepSequence()) {
        auto t = rep->Type();
        if (t == hcomm::CcuRep::CcuRepType::JUMP_NE || t == hcomm::CcuRep::CcuRepType::JUMP_EQ
            || t == hcomm::CcuRep::CcuRepType::JUMP) {
            auto* jump = dynamic_cast<hcomm::CcuRep::CcuRepJumpBase*>(rep.get());
            if (jump != nullptr) {
                jumpRepCount++;
                jumpTargetIds.insert(jump->GetTargetInstrId().Id());
            }
        }
    }

    // A5 共用：所有 targetInstrId 应为同一个 Xn ID
    EXPECT_EQ(jumpRepCount, 10);
    EXPECT_EQ(jumpTargetIds.size(), 1u);

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

// A6(CCU_V2): targetVar 不共用，验证多个 CCU_IF 的 targetInstrId都不相同
TEST_F(HcommCcuControlApiTest, Ut_CcuIfSharedTargetVar_When_V2_Expect_NoSharing)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 11;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V2;
    int32_t fakeDeviceLogicId = MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    CcuIfSharedVarDemoKernelArg demoArg{};
    demoArg.value = 0;
    demoArg.expect = 0;
    CcuKernelArg kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void* kernelArgs[] = {kernelArg};
    auto kernelFunc = reinterpret_cast<void*>(CcuIfSharedVarDemoKernel);
    char* kernelFuncName = "ccu_if_shared_var_demo_v2";
    CcuKernelHandle kernelHandle{0};
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(
        insHandle, fakeDieId, kernelFuncName, kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    auto* kernel = hcomm::CcuKernelMgr::GetInstance(fakeDeviceLogicId).GetKernel(kernelHandle);
    ASSERT_NE(kernel, nullptr);

    // sharedJumpTargetVar_ 应未初始化（A6 不共用）
    EXPECT_TRUE(kernel->sharedJumpTargetVar_ == nullptr);

    // 收集所有条件跳转 rep 的 targetInstrId，验证它们互不相同
    uint32_t jumpRepCount = 0;
    std::set<uint16_t> jumpTargetIds;
    for (const auto& rep : kernel->GetRepSequence()) {
        auto t = rep->Type();
        if (t == hcomm::CcuRep::CcuRepType::JUMP_NE || t == hcomm::CcuRep::CcuRepType::JUMP_EQ
            || t == hcomm::CcuRep::CcuRepType::JUMP) {
            auto* jump = dynamic_cast<hcomm::CcuRep::CcuRepJumpBase*>(rep.get());
            if (jump != nullptr) {
                jumpRepCount++;
                jumpTargetIds.insert(jump->GetTargetInstrId().Id());
            }
        }
    }
    // A6 不共用：每个跳转 rep 的 targetInstrId 互不相同
    ASSERT_EQ(jumpRepCount, 10u);
    EXPECT_EQ(jumpTargetIds.size(), jumpRepCount);

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

// 参考实现:用纯 C++ 复现 CcuIfSharedVarDemoKernel 各场景的控制流逻辑,不依赖 CCU 代码,
// 用于与解释器执行结果逐场景对比。value 为 uint32,在 Xn 中按 64 位比较,故统一提升到 uint64_t。
// 返回四个结果变量的期望值,顺序与 kIfSharedVarScenarios 一致。
struct CcuIfSharedVarResult {
    uint64_t r_if;     // 场景1: 单层 IF
    uint64_t r_ifelse; // 场景2: IF + ELSE
    uint64_t r_nest;   // 场景3: 嵌套 IF + ELSE
    uint64_t r_for;    // 场景4: for 循环内 IF
};
// 计算参考实现期望结果:用与 CcuIfSharedVarDemoKernel 相同的控制流结构推演四个结果变量的终值,
// 不依赖 CCU 代码。命中值 10、未命中值 20、各场景未触发时保留的标记值,均与 kernel 内常量一致。
static CcuIfSharedVarResult CcuIfSharedVarExpectedResult(uint32_t value, uint64_t expect)
{
    // 与 kernel 一致:变量初值(未命中/未触发时保留的标记值)。
    uint64_t r_if = 1001;     // 场景1
    uint64_t r_ifelse = 1002; // 场景2
    uint64_t r_nest = 1003;   // 场景3
    uint64_t r_for = 1004;    // 场景4
    const uint64_t hit = 10;  // 命中取值
    const uint64_t miss = 20; // 未命中取值
    const uint64_t v = static_cast<uint64_t>(value);

    // 场景1: 单层 IF,无 else。命中则 r_if=hit,否则保持标记 1001。
    if (v == expect) {
        r_if = hit;
    }

    // 场景2: IF + ELSE。命中走 then(hit),否则走 else(miss)。
    if (v == expect) {
        r_ifelse = hit;
    } else {
        r_ifelse = miss;
    }

    // 场景3: 嵌套 IF + ELSE。外层 v==expect 才进入;内层条件与外层相同,外层真时内层走 then(hit)。
    //         外层假时整块跳过,r_nest 保持标记 1003。
    if (v == expect) {
        if (v == expect) {
            r_nest = hit;
        } else {
            r_nest = miss;
        }
    }

    // 场景4: for 循环内 IF。循环 4 次,v==i 命中则 r_for=hit(赋值,至多命中一次)。
    for (int i = 0; i < 4; ++i) {
        if (v == static_cast<uint64_t>(i)) {
            r_for = hit;
        }
    }

    return {r_if, r_ifelse, r_nest, r_for};
}

// 各场景结果变量的标记值(即 kernel 中的初值立即数)与名称。
// 测试通过扫描 LoadImdToXn 指令中 immediate 等于标记值的指令,定位各结果变量的 Xn 编号。
struct IfSharedVarScenario {
    uint64_t marker;
    const char* name;
};
static constexpr IfSharedVarScenario kIfSharedVarScenarios[4] = {
    {1001, "r_if(单层IF)"},
    {1002, "r_ifelse(IF+ELSE)"},
    {1003, "r_nest(嵌套IF+ELSE)"},
    {1004, "r_for(for循环内IF)"},
};

// 执行结果验证:在指定设备上以给定输入注册并翻译 CcuIfSharedVarDemoKernel,
// 将翻译生成的指令序列载入 CcuMicroSim 执行,读取四个结果变量的最终值,
// 与参考实现 CcuIfSharedVarExpectedResult 逐场景对比。设备相关的 mock 由调用方(TEST_P)完成。
static void VerifyIfSharedVarExecDiff(int32_t fakeDeviceLogicId, uint32_t value, uint64_t expect)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, hcomm::CcuVersion::CCU_V1);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    ASSERT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    ASSERT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    CcuIfSharedVarDemoKernelArg demoArg{};
    demoArg.value = value;
    demoArg.expect = expect;
    CcuKernelArg kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void* kernelArgs[] = {kernelArg};
    auto kernelFunc = reinterpret_cast<void*>(CcuIfSharedVarDemoKernel);
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    CcuKernelHandle kernelHandle{0};
    ccuRet = HcommCcuKernelRegister(
        insHandle, fakeDieId, "ccu_if_shared_var_exec_diff", kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    ASSERT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    // RegisterEnd 触发翻译并调用 SetCcuInstrInfo,此后 kernel->instrInfo_ 中即为翻译后的指令序列。
    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    ASSERT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    auto* kernel = hcomm::CcuKernelMgr::GetInstance(fakeDeviceLogicId).GetKernel(kernelHandle);
    ASSERT_NE(kernel, nullptr);

    // 确认当前走在 V1 跳转寄存器复用路径上(与结构测试 Ut_CcuIfSharedTargetVar_When_V1_Expect_XnReduced 呼应)。
    EXPECT_TRUE(kernel->sharedJumpTargetVar_ != nullptr);

    // 通过标记值定位各结果变量的 Xn 编号:扫 LoadImdToXn 指令,immediate 等于标记值者即为对应变量的初值指令。
    uint16_t accXnId[4] = {0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF};
    const auto& instrVec = kernel->instrInfo_.instrVec;
    for (const auto& ins : instrVec) {
        if (ins.header.type != hcomm::CcuRep::MicroSimV1::LOAD_TYPE
            || ins.header.code != hcomm::CcuRep::MicroSimV1::LOADIMDTOXN_CODE) {
            continue;
        }
        for (size_t k = 0; k < 4; ++k) {
            if (ins.v1.loadImdToXn.immediate == kIfSharedVarScenarios[k].marker) {
                accXnId[k] = ins.v1.loadImdToXn.xnId;
            }
        }
    }
    for (size_t k = 0; k < 4; ++k) {
        ASSERT_NE(accXnId[k], 0xFFFF) << "结果变量未定位到 Xn 编号: " << kIfSharedVarScenarios[k].name;
    }

    // 执行翻译后的指令序列。
    hcomm::CcuRep::CcuMicroSim sim;
    sim.Load(kernel->instrInfo_);
    ASSERT_TRUE(sim.Run()) << "sim run failed";

    // 逐场景断言,失败信息标注具体场景,便于定位。
    const auto exp = CcuIfSharedVarExpectedResult(value, expect);
    EXPECT_EQ(sim.GetXn(accXnId[0]), exp.r_if) << "场景1 " << kIfSharedVarScenarios[0].name;
    EXPECT_EQ(sim.GetXn(accXnId[1]), exp.r_ifelse) << "场景2 " << kIfSharedVarScenarios[1].name;
    EXPECT_EQ(sim.GetXn(accXnId[2]), exp.r_nest) << "场景3 " << kIfSharedVarScenarios[2].name;
    EXPECT_EQ(sim.GetXn(accXnId[3]), exp.r_for) << "场景4 " << kIfSharedVarScenarios[3].name;

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

// 执行结果验证:在 CCU_V1(跳转寄存器复用)路径下翻译并执行 CcuIfSharedVarDemoKernel
// (覆盖 CCU_IF / 嵌套 CCU_IF / CCU_ELSE / for 循环内 CCU_IF 四类场景),将执行结果与参考实现对比。
// 采用参数化测试,每组输入使用独立的 fakeDevId 以隔离设备资源池,覆盖:
//   全命中 / IF 未命中走 else / 全未命中 / for 命中不同 i / value 超出 0..3 等场景。
struct CcuIfExecDiffParam {
    uint32_t value;
    uint64_t expect;
    uint32_t devIdOffset; // fakeDevId = MAX_MODULE_DEVICE_NUM - devIdOffset,用于隔离设备资源池
    const char* label;    // 参数化测试实例名(仅字母数字下划线)
};

class CcuIfSharedTargetVarExecDiffTest :
    public HcommCcuControlApiTest,
    public testing::WithParamInterface<CcuIfExecDiffParam> {};

TEST_P(CcuIfSharedTargetVarExecDiffTest, V1_MatchesOracle)
{
    const auto& p = GetParam();
    const uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - p.devIdOffset;
    int32_t fakeDeviceLogicId = MockCcuDeviceEnv(fakeDevId, hcomm::CcuVersion::CCU_V1);

    SCOPED_TRACE(
        std::string(p.label) + " (value=" + std::to_string(p.value) + ", expect=" + std::to_string(p.expect) + ")");
    VerifyIfSharedVarExecDiff(fakeDeviceLogicId, p.value, p.expect);
}

INSTANTIATE_TEST_SUITE_P(
    CcuIfSharedTargetVar_ExecDiff_V1, CcuIfSharedTargetVarExecDiffTest,
    testing::Values(
        CcuIfExecDiffParam{2, 2, 20, "AllHit_ForHitI2"},     // 全命中,for 命中 i=2
        CcuIfExecDiffParam{2, 5, 21, "IfMiss_ElseHit"},      // IF 未命中走 else,for 命中 i=2
        CcuIfExecDiffParam{5, 5, 22, "Hit_ForMiss"},         // IF/嵌套命中,for 未命中
        CcuIfExecDiffParam{5, 3, 23, "AllMiss"},             // 全未命中(走 else),for 未命中
        CcuIfExecDiffParam{0, 0, 24, "AllHit_ForHitI0"},     // 全命中,for 命中 i=0
        CcuIfExecDiffParam{3, 3, 25, "AllHit_ForHitI3"},     // 全命中,for 命中 i=3
        CcuIfExecDiffParam{7, 7, 26, "Hit_ForMiss_BigValue"} // 命中但 value 超出 0..3,for 未命中
        ),
    [](const testing::TestParamInfo<CcuIfExecDiffParam>& info) {
        return info.param.label;
    });

TEST_F(HcommCcuControlApiTest, Ut_DoWhileLabelStackPopForWhile_When_Adjacent_Expect_ReturnLabel)
{
    hcomm::CcuKernel kernel;
    hcomm::CcuRep::CcuInsGeneratorV1 insGen;
    kernel.SetInsGenerater(&insGen);
    const char* label = "ut_dw_adjacent";

    EXPECT_EQ(kernel.DoWhileBegin(label), CcuResult::CCU_SUCCESS);
    kernel.DoWhileLabelStackPush(label);

    const char* popped = kernel.DoWhileLabelStackPopForWhile();
    ASSERT_NE(popped, nullptr);
    EXPECT_STREQ(popped, label);
    EXPECT_TRUE(kernel.doWhileLabelStack_.empty());
}

TEST_F(HcommCcuControlApiTest, Ut_DoWhileLabelStackPopForWhile_When_DanglingAppend_Expect_ReturnNullptr)
{
    hcomm::CcuKernel kernel;
    hcomm::CcuRep::CcuInsGeneratorV1 insGen;
    kernel.SetInsGenerater(&insGen);
    const char* labelOuter = "ut_dw_outer";
    const char* labelDangling = "ut_dw_dangling";

    EXPECT_EQ(kernel.DoWhileBegin(labelOuter), CcuResult::CCU_SUCCESS);
    kernel.DoWhileLabelStackPush(labelOuter);

    // 用第二次 DoWhileBegin 模拟 CCU_DO 与 CCU_WHILE 之间夹杂了会 Append 指令的代码。
    EXPECT_EQ(kernel.DoWhileBegin(labelDangling), CcuResult::CCU_SUCCESS);

    // 快照 rep 数与当前不匹配，PopForWhile 返回 nullptr 拒绝配对，但仍需弹出栈条目。
    const char* popped = kernel.DoWhileLabelStackPopForWhile();
    EXPECT_EQ(popped, nullptr);
    EXPECT_TRUE(kernel.doWhileLabelStack_.empty());
}

TEST_F(HcommCcuControlApiTest, Ut_DoWhileLabelStackPopForWhile_When_InLoopBody_Expect_ReturnNullptr)
{
    // loop body 内(CurrentBlock 为 LOOP_BLOCK)PopForWhile 应直接返回 nullptr 且不消费栈条目。
    hcomm::CcuKernel kernel;
    hcomm::CcuRep::CcuInsGeneratorV1 insGen;
    kernel.SetInsGenerater(&insGen);
    CcuLoop loop = 0;
    ASSERT_EQ(kernel.LoopCreate(&loop), CcuResult::CCU_SUCCESS);
    ASSERT_EQ(kernel.LoopBodyEnter(loop), CcuResult::CCU_SUCCESS);

    const char* label = "ut_dw_in_loop";
    kernel.DoWhileLabelStackPush(label);
    ASSERT_FALSE(kernel.doWhileLabelStack_.empty());

    EXPECT_EQ(kernel.DoWhileLabelStackPopForWhile(), nullptr);
    EXPECT_FALSE(kernel.doWhileLabelStack_.empty());

    EXPECT_EQ(kernel.LoopBodyExit(loop), CcuResult::CCU_SUCCESS);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelLoopAdd_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 7;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    constexpr auto commEngine = CommEngine::COMM_ENGINE_CCU;
    constexpr uint32_t srcDevPhyId = fakeDevId;
    constexpr uint32_t dstDevPhyId = 1;
    constexpr uint32_t srcIp = 167772383;
    constexpr uint32_t dstIp = 0x87654321;
    const auto& handlePair = MockCcuChannelConnect(srcDevPhyId, dstDevPhyId, srcIp, dstIp, commEngine);

    auto demoFunc = CcuLoopAddDemoKernel;
    CcuLoopAddKernelArg demoArg{};
    demoArg.numA = 7;
    demoArg.numB = 11;
    auto kernelFunc = reinterpret_cast<void*>(demoFunc);
    auto kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void* kernelArgs[] = {kernelArg};

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    char* kernelFuncName = "ccu_loop_add_demo";
    RegisterAndCleanupSingleArgKernel(insHandle, kernelFuncName, kernelFunc, kernelArgs, handlePair, resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelRemoteRead_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 8;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    constexpr auto commEngine = CommEngine::COMM_ENGINE_CCU;
    constexpr uint32_t srcDevPhyId = fakeDevId;
    constexpr uint32_t dstDevPhyId = 1;
    constexpr uint32_t srcIp = 167772383;
    constexpr uint32_t dstIp = 0x87654321;
    const auto& handlePair = MockCcuChannelConnect(srcDevPhyId, dstDevPhyId, srcIp, dstIp, commEngine);

    auto demoFunc = CcuRemoteReadKernel;
    CcuVarAddKernelArg demoArg{};
    demoArg.channelHandle = handlePair.second;
    auto kernelFunc = reinterpret_cast<void*>(demoFunc);
    auto kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void* kernelArgs[] = {kernelArg};

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    char* kernelFuncName = "ccu_remote_read_demo";
    RegisterAndCleanupSingleArgKernel(insHandle, kernelFuncName, kernelFunc, kernelArgs, handlePair, resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelRemoteWrite_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 9;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    constexpr auto commEngine = CommEngine::COMM_ENGINE_CCU;
    constexpr uint32_t srcDevPhyId = fakeDevId;
    constexpr uint32_t dstDevPhyId = 1;
    constexpr uint32_t srcIp = 167772383;
    constexpr uint32_t dstIp = 0x87654321;
    const auto& handlePair = MockCcuChannelConnect(srcDevPhyId, dstDevPhyId, srcIp, dstIp, commEngine);

    auto demoFunc = CcuRemoteWriteKernel;
    CcuVarAddKernelArg demoArg{};
    demoArg.channelHandle = handlePair.second;
    auto kernelFunc = reinterpret_cast<void*>(demoFunc);
    auto kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void* kernelArgs[] = {kernelArg};

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    char* kernelFuncName = "ccu_remote_write_demo";
    RegisterAndCleanupSingleArgKernel(insHandle, kernelFuncName, kernelFunc, kernelArgs, handlePair, resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelAlloc_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 10;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    constexpr auto commEngine = CommEngine::COMM_ENGINE_CCU;
    constexpr uint32_t srcDevPhyId = fakeDevId;
    constexpr uint32_t dstDevPhyId = 1;
    constexpr uint32_t srcIp = 167772383;
    constexpr uint32_t dstIp = 0x87654321;
    const auto& handlePair = MockCcuChannelConnect(srcDevPhyId, dstDevPhyId, srcIp, dstIp, commEngine);

    auto demoFunc = CcuAllocDemoKernel;
    CcuVarAddKernelArg demoArg{};
    demoArg.channelHandle = handlePair.second;
    auto kernelFunc = reinterpret_cast<void*>(demoFunc);
    auto kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void* kernelArgs[] = {kernelArg};

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    char* kernelFuncName = "ccu_alloc_demo";
    CcuKernelHandle kernelHandle{0};
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(
        insHandle, fakeDieId, kernelFuncName, kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    // 验证 HcommCcuGetTaskArgsNum：CcuAllocDemoKernel 调用 LoadArg(0) 和 LoadArg(1)，max(argId)+1=2
    uint32_t taskArgsNum = 0xFFFFFFFF;
    EXPECT_EQ(HcommCcuGetTaskArgsNum(kernelHandle, &taskArgsNum), CcuResult::CCU_SUCCESS);
    EXPECT_EQ(taskArgsNum, 2U);

    MockChannelDestroy(handlePair);
    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelReduceScatterMesh1d_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    // 整体打桩，处理ccu资源
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;

    // ccuInstance构建（正常在通信域创建中，本用例仅测试hcomm接口）
    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CcuInsHandle insHandle{0};
    SetupV1CcuInstance(resDescs, insHandle, fakeDevId);

    constexpr auto commEngine = CommEngine::COMM_ENGINE_CCU;
    const auto& handlePair = MockCcuChannelConnectDefault(fakeDevId, commEngine);

    // 构造CcuKernel实现
    auto demoFunc = CcuReduceScatterMesh1dKernel;
    ReduceScatterKernelArg demoArg{};
    demoArg.rankSize = 2;
    demoArg.rankId = 0;
    demoArg.channelCount = 1;
    demoArg.channels[0] = handlePair.second;
    demoArg.dataType = HCCL_DATA_TYPE_FP16;
    demoArg.outputDataType = HCCL_DATA_TYPE_FP16;
    demoArg.reduceOp = HCCL_REDUCE_SUM;

    auto kernelFunc = reinterpret_cast<void*>(demoFunc);
    auto kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void* kernelArgs[] = {kernelArg};

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    // kernel注册
    char* kernelFuncName = "ccu_reduce_scatter_mesh1d_demo";
    RegisterLaunchAndCleanupSingleArgKernel(
        insHandle, kernelFuncName, kernelFunc, kernelArgs, handlePair, resDescs, commEngine, 15);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelReduceScatterMesh1d_When_V2_Expect_ReturnCcuSUCCESS)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 15;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V2;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    constexpr auto commEngine = CommEngine::COMM_ENGINE_CCU;
    constexpr uint32_t srcDevPhyId = fakeDevId;
    constexpr uint32_t dstDevPhyId = 1;
    constexpr uint32_t srcIp = 167772383;
    constexpr uint32_t dstIp = 0x87654321;
    const auto& handlePair = MockCcuChannelConnect(srcDevPhyId, dstDevPhyId, srcIp, dstIp, commEngine);

    auto demoFunc = CcuReduceScatterMesh1dV2Kernel;
    ReduceScatterKernelArgV2 demoArg{};
    demoArg.rankSize = 2;
    demoArg.rankId = 0;
    demoArg.channelCount = 1;
    demoArg.channels[0] = handlePair.second;
    demoArg.dataType = HCCL_DATA_TYPE_FP16;
    demoArg.outputDataType = HCCL_DATA_TYPE_FP16;
    demoArg.reduceOp = HCCL_REDUCE_SUM;

    auto kernelFunc = reinterpret_cast<void*>(demoFunc);
    auto kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void* kernelArgs[] = {kernelArg};

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    char* kernelFuncName = "ccu_reduce_scatter_mesh1d_a6_demo";
    RegisterLaunchAndCleanupSingleArgKernel(
        insHandle, kernelFuncName, kernelFunc, kernelArgs, handlePair, resDescs, commEngine, 15);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuVariableAllocConsecutive_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 13;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    const int32_t fakeDeviceLogicId = MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    constexpr auto MS_INS_TPYE = CcuInstanceType::CCU_MS;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreateLegacy(MS_INS_TPYE, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    constexpr uint8_t fakeDieId = 0;
    constexpr uint32_t varNum = 10;
    CcuVariableHandle varHandle{0};
    ccuRet = HcommCcuVariableAlloc(insHandle, fakeDieId, varNum, &varHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    for (int i = 0; i < varNum; i++) {
        uint64_t startVa = 0;
        ccuRet = HcommCcuVariableGetAddr(varHandle, i, &startVa);
        EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
        std::cout << "[Ut_HcommCcuVariableAllocConsecutive] varHandle=0x" << std::hex << varHandle << ", startVa=0x"
                  << startVa << std::dec << ", i=" << i << std::endl;
    }

    auto& resMgr = hcomm::CcuVarEventResMgr::GetInstance(fakeDeviceLogicId);
    auto it = resMgr.resMap_.find(varHandle);
    ASSERT_NE(it, resMgr.resMap_.end());
    const auto& resInfos = it->second.resInfos;
    EXPECT_EQ(resInfos.size(), 1U);

    uint32_t totalNum = 0;
    for (const auto& info : resInfos) {
        std::cout << "  consecutive block: startId=" << info.startId << ", num=" << info.num << ", xn ids=[";
        for (uint32_t i = 0; i < info.num; i++) {
            std::cout << (info.startId + i);
            if (i + 1 < info.num) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
        totalNum += info.num;
    }
    EXPECT_EQ(totalNum, varNum);

    CcuEventHandle eventHandle{0};
    constexpr uint32_t eventNum = 4;
    ccuRet = HcommCcuEventAlloc(insHandle, fakeDieId, eventNum, &eventHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    for (int i = 0; i < eventNum; i++) {
        uint64_t eventVa = 0;
        ccuRet = HcommCcuEventGetAddr(eventHandle, i, &eventVa);
        EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
        std::cout << "[Ut_HcommCcuVariableAllocConsecutive] eventHandle=0x" << std::hex << eventHandle << ", eventVa=0x"
                  << eventVa << std::dec << ", i=" << i << std::endl;
    }

    auto eventIt = resMgr.resMap_.find(eventHandle);
    ASSERT_NE(eventIt, resMgr.resMap_.end());
    const auto& ckeInfos = eventIt->second.resInfos;
    EXPECT_EQ(ckeInfos.size(), 1U);
    for (const auto& info : ckeInfos) {
        std::cout << "  consecutive block: startId=" << info.startId << ", num=" << info.num << ", xn ids=[";
        for (uint32_t i = 0; i < info.num; i++) {
            std::cout << (info.startId + i);
            if (i + 1 < info.num) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    CcuVarAcquireDemoKernelArg demoArg{};
    demoArg.acqHandle = varHandle;
    demoArg.varNum = varNum;
    demoArg.acqEventHandle = eventHandle;
    demoArg.eventNum = eventNum;
    auto kernelFunc = reinterpret_cast<void*>(CcuVarAcquireDemoKernel);
    auto kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void* kernelArgs[] = {kernelArg};
    char* kernelFuncName = "ccu_var_acquire_demo";
    CcuKernelHandle kernelHandle{0};
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(
        insHandle, fakeDieId, kernelFuncName, kernelFunc, kernelArgs, kernelArgNum, &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelNestedInIfIf_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 12;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    constexpr auto commEngine = CommEngine::COMM_ENGINE_CCU;
    constexpr uint32_t srcDevPhyId = fakeDevId;
    constexpr uint32_t dstDevPhyId = 1;
    constexpr uint32_t srcIp = 167772383;
    constexpr uint32_t dstIp = 0x87654321;
    const auto& handlePair = MockCcuChannelConnect(srcDevPhyId, dstDevPhyId, srcIp, dstIp, commEngine);

    auto demoFunc = CcuNestedInIfIfDemoKernel;
    CcuNestedInIfIfDemoKernelArg demoArg{};
    demoArg.outerVal = 1;
    demoArg.outerExpected = 1;
    demoArg.innerVal_1 = 2;
    demoArg.innerExpected_1 = 2;
    demoArg.innerVal_2 = 3;
    demoArg.innerExpected_2 = 3;
    auto kernelFunc = reinterpret_cast<void*>(demoFunc);
    auto kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void* kernelArgs[] = {kernelArg};

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    char* kernelFuncName = "ccu_nested_in_if_if_demo";
    RegisterAndCleanupSingleArgKernel(insHandle, kernelFuncName, kernelFunc, kernelArgs, handlePair, resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuVariableAndEventAllocAddrGet_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 14;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    const int32_t fakeDeviceLogicId = MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    // ccuInstance构建（正常在通信域创建中，本用例仅测试hcomm接口）
    constexpr auto MS_INS_TPYE = CcuInstanceType::CCU_MS;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreateLegacy(MS_INS_TPYE, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    constexpr uint8_t fakeDieId = 0;

    constexpr uint32_t varNum = 8;
    CcuVariableHandle varHandle{0};
    ccuRet = HcommCcuVariableAlloc(insHandle, fakeDieId, varNum, &varHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    for (uint32_t i = 0; i < varNum; i++) {
        uint64_t startVa = 0;
        ccuRet = HcommCcuVariableGetAddr(varHandle, i, &startVa);
        EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
        EXPECT_NE(startVa, 0U);
    }

    // 越界index应返回失败
    uint64_t invalidVa = 0;
    ccuRet = HcommCcuVariableGetAddr(varHandle, varNum, &invalidVa);
    EXPECT_NE(ccuRet, CcuResult::CCU_SUCCESS);

    constexpr uint32_t eventNum = 4;
    CcuEventHandle eventHandle{0};
    ccuRet = HcommCcuEventAlloc(insHandle, fakeDieId, eventNum, &eventHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    for (uint32_t i = 0; i < eventNum; i++) {
        uint64_t eventVa = 0;
        ccuRet = HcommCcuEventGetAddr(eventHandle, i, &eventVa);
        EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
        EXPECT_NE(eventVa, 0U);
    }

    // 越界index应返回失败
    uint64_t invalidEventVa = 0;
    ccuRet = HcommCcuEventGetAddr(eventHandle, eventNum, &invalidEventVa);
    EXPECT_NE(ccuRet, CcuResult::CCU_SUCCESS);

    // 校验资源管理器中记录了对应连续资源块
    auto& resMgr = hcomm::CcuVarEventResMgr::GetInstance(fakeDeviceLogicId);
    auto varIt = resMgr.resMap_.find(varHandle);
    ASSERT_NE(varIt, resMgr.resMap_.end());
    uint32_t varTotal = 0;
    for (const auto& info : varIt->second.resInfos) {
        varTotal += info.num;
    }
    EXPECT_EQ(varTotal, varNum);

    auto eventIt = resMgr.resMap_.find(eventHandle);
    ASSERT_NE(eventIt, resMgr.resMap_.end());
    uint32_t eventTotal = 0;
    for (const auto& info : eventIt->second.resInfos) {
        eventTotal += info.num;
    }
    EXPECT_EQ(eventTotal, eventNum);

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuVarEventAlloc_When_InvalidParam_Expect_ReturnFailed)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 15;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    const int32_t fakeDeviceLogicId = MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    constexpr auto MS_INS_TPYE = CcuInstanceType::CCU_MS;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreateLegacy(MS_INS_TPYE, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    constexpr uint8_t fakeDieId = 0;
    constexpr uint32_t validNum = 4;
    CcuVariableHandle varHandle{0xFFFF};
    CcuEventHandle eventHandle{0xFFFF};

    // num 为 0：参数非法，且出参必须被清 0，避免调用方误用脏值
    ccuRet = HcommCcuVariableAlloc(insHandle, fakeDieId, 0, &varHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_E_PARA);
    EXPECT_EQ(varHandle, 0U);
    varHandle = 0xFFFF;
    ccuRet = HcommCcuEventAlloc(insHandle, fakeDieId, 0, &eventHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_E_PARA);
    EXPECT_EQ(eventHandle, 0U);
    eventHandle = 0xFFFF;

    // dieId 越界：参数非法
    constexpr uint8_t invalidDieId = static_cast<uint8_t>(hcomm::CCU_MAX_IODIE_NUM);
    ccuRet = HcommCcuVariableAlloc(insHandle, invalidDieId, validNum, &varHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_E_PARA);
    EXPECT_EQ(varHandle, 0U);
    varHandle = 0xFFFF;
    ccuRet = HcommCcuEventAlloc(insHandle, invalidDieId, validNum, &eventHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_E_PARA);
    EXPECT_EQ(eventHandle, 0U);

    // 出参指针为空
    EXPECT_EQ(HcommCcuVariableAlloc(insHandle, fakeDieId, validNum, nullptr), CcuResult::CCU_E_PTR);
    EXPECT_EQ(HcommCcuEventAlloc(insHandle, fakeDieId, validNum, nullptr), CcuResult::CCU_E_PTR);

    // 池中无长度足够的连续块：资源不可用
    constexpr uint32_t hugeNum = 100000;
    varHandle = 0xFFFF;
    ccuRet = HcommCcuVariableAlloc(insHandle, fakeDieId, hugeNum, &varHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_E_UNAVAIL);
    EXPECT_EQ(varHandle, 0U);
    eventHandle = 0xFFFF;
    ccuRet = HcommCcuEventAlloc(insHandle, fakeDieId, hugeNum, &eventHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_E_UNAVAIL);
    EXPECT_EQ(eventHandle, 0U);

    // AddrGet 的错误路径：句柄不存在 / 句柄类型不符 / 出参指针为空
    uint64_t va = 0;
    EXPECT_EQ(HcommCcuVariableGetAddr(0xDEADBEEF, 0, &va), CcuResult::CCU_E_NOT_FOUND);
    EXPECT_EQ(HcommCcuEventGetAddr(0xDEADBEEF, 0, &va), CcuResult::CCU_E_NOT_FOUND);

    ccuRet = HcommCcuVariableAlloc(insHandle, fakeDieId, validNum, &varHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    EXPECT_NE(varHandle, 0U);
    EXPECT_EQ(HcommCcuVariableGetAddr(varHandle, 0, nullptr), CcuResult::CCU_E_PTR);
    // 用 variable 的预约句柄去查 event 地址，类型不符应失败
    EXPECT_EQ(HcommCcuEventGetAddr(varHandle, 0, &va), CcuResult::CCU_E_PARA);

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuVariableAlloc_When_MapAddrFailed_Expect_RollbackAll)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 16;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    const int32_t fakeDeviceLogicId = MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    constexpr auto MS_INS_TPYE = CcuInstanceType::CCU_MS;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreateLegacy(MS_INS_TPYE, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    auto* ccuIns = hcomm::CcuInstanceMgr::GetInstance(fakeDeviceLogicId).Get(insHandle);
    ASSERT_NE(ccuIns, nullptr);
    auto* resPack = ccuIns->GetResPack();
    ASSERT_NE(resPack, nullptr);

    constexpr uint8_t fakeDieId = 0;
    const std::vector<hcomm::ResInfo> poolBefore = resPack->GetCcuResRepo().blockXn[fakeDieId];
    auto& resMgr = hcomm::CcuVarEventResMgr::GetInstance(fakeDeviceLogicId);
    const size_t resMapSizeBefore = resMgr.resMap_.size();

    // 第 1 个资源映射成功、第 2 个起失败，触发已映射资源的逐个解映射 + 整笔归还
    MOCKER(rtGetDevResAddress)
        .stubs()
        .will(returnValue(static_cast<rtError_t>(RT_ERROR_NONE)))
        .then(returnValue(static_cast<rtError_t>(0xFFFFFFFF)));
    // 已成功映射的那 1 个资源必须被精确解映射：次数由 TearDown 的 GlobalMockObject::verify 校验
    MOCKER(rtReleaseDevResAddress).expects(exactly(1)).will(returnValue(static_cast<rtError_t>(RT_ERROR_NONE)));

    constexpr uint32_t varNum = 4;
    CcuVariableHandle varHandle{0xFFFF};
    ccuRet = HcommCcuVariableAlloc(insHandle, fakeDieId, varNum, &varHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_E_RUNTIME);

    // 回滚后：出参不得残留句柄，登记表不得残留记录，资源池必须完全恢复
    EXPECT_EQ(varHandle, 0U);
    EXPECT_EQ(resMgr.resMap_.size(), resMapSizeBefore);

    const std::vector<hcomm::ResInfo>& poolAfter = resPack->GetCcuResRepo().blockXn[fakeDieId];
    uint32_t totalBefore = 0;
    for (const auto& info : poolBefore) {
        totalBefore += info.num;
    }
    uint32_t totalAfter = 0;
    for (const auto& info : poolAfter) {
        totalAfter += info.num;
    }
    EXPECT_EQ(totalAfter, totalBefore);

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
}

// 前几条用例都经 HcommCcuInsCreateLegacy 建实例，走 CcuInstanceMgr::CreateByInsType；
// 本例改走 HcommCcuInsCreate（落到 CreateByResDescs）。两条路径都必须把实例句柄回填给
// CcuInstance，否则析构时 ReleaseByInstance 拿到的 insHandle_ 为 0，匹配不到任何预约记录，
// 预约的 XN/CKE 既不解映射也不还池——不报错、只泄漏。用「解映射次数」与「登记表是否清空」
// 两个可观测量把这条约束钉住。
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuVarEventAlloc_When_InsCreateByResDescs_Expect_ReleaseOnInsDestroy)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 17;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    const int32_t fakeDeviceLogicId = MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    ASSERT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    constexpr uint8_t fakeDieId = 0;
    constexpr uint32_t varNum = 4;
    constexpr uint32_t eventNum = 2;
    // 预约期每个资源各注册一次 VA，实例析构时必须一一解映射；次数由 TearDown 的
    // GlobalMockObject::verify 校验，漏调 SetHandle 时这里会是 0 次
    MOCKER(rtReleaseDevResAddress)
        .expects(exactly(varNum + eventNum))
        .will(returnValue(static_cast<rtError_t>(RT_ERROR_NONE)));

    CcuVariableHandle varHandle{0};
    ASSERT_EQ(HcommCcuVariableAlloc(insHandle, fakeDieId, varNum, &varHandle), CcuResult::CCU_SUCCESS);
    CcuEventHandle eventHandle{0};
    ASSERT_EQ(HcommCcuEventAlloc(insHandle, fakeDieId, eventNum, &eventHandle), CcuResult::CCU_SUCCESS);

    auto& resMgr = hcomm::CcuVarEventResMgr::GetInstance(fakeDeviceLogicId);
    ASSERT_NE(resMgr.resMap_.find(varHandle), resMgr.resMap_.end());
    ASSERT_NE(resMgr.resMap_.find(eventHandle), resMgr.resMap_.end());

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    // 实例销毁后两条预约记录都不得残留
    EXPECT_EQ(resMgr.resMap_.find(varHandle), resMgr.resMap_.end());
    EXPECT_EQ(resMgr.resMap_.find(eventHandle), resMgr.resMap_.end());

    DestroyCcuResDescs(resDescs);
}

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelAllGatherMesh1dMem2Mem_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 13;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    constexpr auto commEngine = CommEngine::COMM_ENGINE_CCU;
    constexpr uint32_t srcDevPhyId = fakeDevId;
    constexpr uint32_t dstDevPhyId = 1;
    constexpr uint32_t srcIp = 167772383;
    constexpr uint32_t dstIp = 0x87654321;
    const auto& handlePair = MockCcuChannelConnect(srcDevPhyId, dstDevPhyId, srcIp, dstIp, commEngine);

    // 构造 demo 入参：rankSize=2, rankId=0, 1 个 peer 通道。
    auto demoFunc = CcuAllGatherMesh1dMem2MemKernel;
    AllGatherKernelArg demoArg{};
    demoArg.rankSize = 2;
    demoArg.rankId = 0;
    demoArg.channelCount = 1;
    demoArg.channels[0] = handlePair.second;

    auto kernelFunc = reinterpret_cast<void*>(demoFunc);
    auto kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void* kernelArgs[] = {kernelArg};

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    char* kernelFuncName = "ccu_all_gather_mesh1d_mem2mem_demo";
    RegisterLaunchAndCleanupSingleArgKernel(
        insHandle, kernelFuncName, kernelFunc, kernelArgs, handlePair, resDescs, commEngine, 15);
}

// HcommCcuInsDestroy 接口在线程 DeviceId 与要析构的目标 DeviceId 不同时，
// 内部正确切换 Device 完成销毁并恢复原 Device
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuInsDestroy_CrossDevice_Expect_Success)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr int32_t curThreadDevId = 0; // 当前线程的 DeviceId
    constexpr int32_t otherDevId = 1;     // 要析构的 DeviceId

    // 在 otherDevId 上创建 CcuInstance
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(static_cast<uint32_t>(otherDevId), fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    MOCKER(hrtSetDevice).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));

    // 销毁 CcuInstance
    MOCKER(HcclGetThreadDeviceId).stubs().will(returnValue(otherDevId));
    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

#define CCU_RELATIONAL_V2_KERNEL_TEST(testName, demoFunc, argType, argInit)                                           \
    TEST_F(HcommCcuControlApiTest, testName)                                                                          \
    {                                                                                                                 \
        CcuResult ccuRet = CcuResult::CCU_E_RESERVED;                                                                 \
        constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 13;                                                    \
        constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V2;                                       \
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
        argType demoArg{};                                                                                            \
        argInit;                                                                                                      \
        CcuKernelArg kernelArg = static_cast<CcuKernelArg>(&demoArg);                                                 \
        const void* kernelArgs[] = {kernelArg};                                                                       \
        CcuKernelHandle kernelHandle{0};                                                                              \
        auto kernelFunc = reinterpret_cast<void*>(demoFunc);                                                          \
        constexpr uint32_t fakeDieId = 0;                                                                             \
        constexpr uint32_t kernelArgNum = 1;                                                                          \
        ccuRet = HcommCcuKernelRegister(                                                                              \
            insHandle, fakeDieId, const_cast<char*>(#demoFunc), kernelFunc, kernelArgs, kernelArgNum, &kernelHandle); \
        EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);                                                                    \
        ccuRet = HcommCcuKernelRegisterEnd(insHandle);                                                                \
        EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);                                                                    \
                                                                                                                      \
        ccuRet = HcommCcuInsDestroy(insHandle);                                                                       \
        EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);                                                                    \
        DestroyCcuResDescs(resDescs);                                                                                 \
    }

CCU_RELATIONAL_V2_KERNEL_TEST(
    Ut_HcommCcuKernelIfRelational_When_V2_Expect_ReturnCcuSUCCESS, CcuIfRelationalDemoKernel,
    CcuIfRelationalDemoKernelArg, (demoArg.value = 5, demoArg.bound = 10))

CCU_RELATIONAL_V2_KERNEL_TEST(
    Ut_HcommCcuKernelIfElseRelational_When_V2_Expect_ReturnCcuSUCCESS, CcuIfElseRelationalDemoKernel,
    CcuIfElseRelationalDemoKernelArg, (demoArg.value = 7, demoArg.threshold = 5))

CCU_RELATIONAL_V2_KERNEL_TEST(
    Ut_HcommCcuKernelWhileRelational_When_V2_Expect_ReturnCcuSUCCESS, CcuWhileRelationalDemoKernel,
    CcuWhileRelationalDemoKernelArg, (demoArg.loopCount = 5))

CCU_RELATIONAL_V2_KERNEL_TEST(
    Ut_HcommCcuKernelDoWhileRelational_When_V2_Expect_ReturnCcuSUCCESS, CcuDoWhileRelationalDemoKernel,
    CcuDoWhileRelationalDemoKernelArg, (demoArg.loopCount = 5))

CCU_RELATIONAL_V2_KERNEL_TEST(
    Ut_HcommCcuKernelVariableComputing_When_V2_Expect_ReturnCcuSUCCESS, CcuVariableComputingKernel,
    CcuVariableComputingKernelArg, (demoArg.varA = 1024, demoArg.varB = 2048))

CCU_RELATIONAL_V2_KERNEL_TEST(
    Ut_HcommCcuKernelVarVarControlFlow_When_V2_Expect_ReturnCcuSUCCESS, CcuVarVarControlFlowDemoKernel,
    CcuVarVarControlFlowDemoKernelArg, (demoArg.valueA = 3, demoArg.valueB = 7, demoArg.loopCount = 4))

TEST_F(HcommCcuControlApiTest, Ut_HcommCcuKernelIfRelational_When_V1_Expect_RegisterEndFail)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 14;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuKernelRegisterStart(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    CcuIfRelationalDemoKernelArg demoArg{};
    demoArg.value = 3;
    demoArg.bound = 5;
    CcuKernelArg kernelArg = static_cast<CcuKernelArg>(&demoArg);
    const void* kernelArgs[] = {kernelArg};
    CcuKernelHandle kernelHandle{0};
    auto kernelFunc = reinterpret_cast<void*>(CcuIfRelationalDemoKernel);
    constexpr uint32_t fakeDieId = 0;
    constexpr uint32_t kernelArgNum = 1;
    ccuRet = HcommCcuKernelRegister(
        insHandle, fakeDieId, const_cast<char*>("CcuIfRelationalDemoKernel_V1"), kernelFunc, kernelArgs, kernelArgNum,
        &kernelHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    // 翻译阶段：V1 generator 走 ValidateInsGeneratorForJump 抛 CcuApiException → 失败
    ccuRet = HcommCcuKernelRegisterEnd(insHandle);
    EXPECT_NE(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

// HcommCcuInsCreate 基于资源描述符数组创建实例，正常路径返回 CCU_SUCCESS
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuInsCreate_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);

    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    EXPECT_NE(insHandle, 0);

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

// HcommCcuInsCreate 内部实例创建或登记失败时返回 CCU_E_INTERNAL，且不产生实例句柄
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuInsCreate_When_ManagerInternalError_Expect_ReturnCcuEINTERNAL)
{
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    MockControlDeviceRefresh(static_cast<int32_t>(fakeDevId));
    MOCKER(HcclGetThreadDeviceId).stubs().will(returnValue(static_cast<int32_t>(fakeDevId)));

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);
    MOCKER_CPP(&hcomm::CcuInstanceMgr::CreateByResDescs).stubs().will(returnValue(CcuResult::CCU_E_INTERNAL));

    CcuInsHandle insHandle{0};
    EXPECT_EQ(HcommCcuInsCreate(resDescs, hcomm::CCU_MAX_IODIE_NUM, &insHandle), CcuResult::CCU_E_INTERNAL);
    EXPECT_EQ(insHandle, 0);

    DestroyCcuResDescs(resDescs);
}

// HcommCcuInsCreate 入参 ccuInsHandle 为空指针，返回 CCU_E_PTR
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuInsCreate_When_NullInsHandle_Expect_ReturnCcuEPTR)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);

    constexpr uint32_t descNum = 2;
    ccuRet = HcommCcuInsCreate(resDescs, descNum, nullptr);
    EXPECT_EQ(ccuRet, CcuResult::CCU_E_PTR);

    DestroyCcuResDescs(resDescs);
}

// HcommCcuInsCreate 入参 resDescNum 为 0，返回 CCU_E_PARA
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuInsCreate_When_ResDescNumZero_Expect_ReturnCcuEPARA)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);

    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, 0, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_E_PARA);

    DestroyCcuResDescs(resDescs);
}

// HcommCcuInsCreate 入参两个 resDesc 的 dieId 重复，返回 CCU_E_PARA
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuInsCreate_When_DieIdDuplicated_Expect_ReturnCcuEPARA)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    // 两个 resDesc 都归属 dieId 0
    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    ccuRet = HcommCcuInsResDescCreate(0, &resDescs[0]);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    ccuRet = HcommCcuInsResDescCreate(0, &resDescs[1]);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    SetCcuResDescCcuMs(resDescs[0], fakeCcuVersion);
    SetCcuResDescCcuMs(resDescs[1], fakeCcuVersion);

    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_E_PARA);

    DestroyCcuResDescs(resDescs);
}

// HcommCcuInsCreate 内部 CcuInitFeature 返回 CCU_E_DRV_BUSY（ccu 驱动拉起失败），
// 期望接口返回 CCU_E_DRV_BUSY，且不产生实例句柄
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuInsCreate_When_DrvBusy_Expect_ReturnCcuEDRVBUSY)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    // 打桩 CcuInitFeature 模拟 ccu 驱动拉起失败
    MOCKER(hcomm::CcuInitFeature).stubs().will(returnValue(CcuResult::CCU_E_DRV_BUSY));

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);

    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_E_DRV_BUSY);
    EXPECT_EQ(insHandle, 0);

    DestroyCcuResDescs(resDescs);
}

// HcommCcuInsCreate 内部 CcuResPack::InitByResDescs 返回 CCU_E_UNAVAIL（资源不足），
// 期望接口返回 CCU_E_UNAVAIL，且不产生实例句柄
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuInsCreate_When_ResUnavail_Expect_ReturnCcuEUNAVAIL)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    // 打桩 CcuResPack::InitByResDescs 模拟资源不足
    MOCKER_CPP(&hcomm::CcuResPack::InitByResDescs).stubs().will(returnValue(CcuResult::CCU_E_UNAVAIL));

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);

    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_E_UNAVAIL);
    EXPECT_EQ(insHandle, 0);

    DestroyCcuResDescs(resDescs);
}

// HcommCcuInsCreateDefault 使用当前 Device 上所有已使能 ioDie 的全部资源，返回 CCU_SUCCESS
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuInsCreateDefault_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    // dieIds/dieNum 为保留参数，调用方按契约传入 nullptr/0
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreateDefault(nullptr, 0, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    EXPECT_NE(insHandle, 0);

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
}

// HcommCcuInsCreateDefault 入参 ccuInsHandle 为空指针，返回 CCU_E_PTR
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuInsCreateDefault_When_NullInsHandle_Expect_ReturnCcuEPTR)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    ccuRet = HcommCcuInsCreateDefault(nullptr, 0, nullptr);
    EXPECT_EQ(ccuRet, CcuResult::CCU_E_PTR);
}

// HcommCcuInsCreateDefault 内部实例创建或登记失败时返回 CCU_E_INTERNAL，且不产生实例句柄
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuInsCreateDefault_When_ManagerInternalError_Expect_ReturnCcuEINTERNAL)
{
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    MockControlDeviceRefresh(static_cast<int32_t>(fakeDevId));
    MOCKER(HcclGetThreadDeviceId).stubs().will(returnValue(static_cast<int32_t>(fakeDevId)));
    MOCKER_CPP(&hcomm::CcuInstanceMgr::CreateByAllRes).stubs().will(returnValue(CcuResult::CCU_E_INTERNAL));

    CcuInsHandle insHandle{0};
    EXPECT_EQ(HcommCcuInsCreateDefault(nullptr, 0, &insHandle), CcuResult::CCU_E_INTERNAL);
    EXPECT_EQ(insHandle, 0);
}

// HcommCcuInsQueryResDesc 查询实例在指定 die 上的占用资源，正常路径返回 CCU_SUCCESS，
// 且查询到的占用数量应为非负值
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuInsQueryResDesc_When_AllFine_Expect_ReturnCcuSUCCESS)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);

    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    // 用 resDescs[0]（归属 dieId 0）作为查询载体，查询该实例在 dieId 0 上的占用资源
    ccuRet = HcommCcuInsQueryResDesc(insHandle, resDescs[0]);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    uint32_t resNum = 0;
    ccuRet = HcommCcuInsResDescQueryNum(resDescs[0], HCOMM_CCU_RES_TYPE_LOOP, &resNum);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    EXPECT_EQ(resNum, 8 * 8 * 2);
    ccuRet = HcommCcuInsResDescQueryNum(resDescs[0], HCOMM_CCU_RES_TYPE_CCU_BUF, &resNum);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    EXPECT_EQ(resNum, 64 * 8 * 2);
    ccuRet = HcommCcuInsResDescQueryNum(resDescs[0], HCOMM_CCU_RES_TYPE_VARIABLE, &resNum);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    EXPECT_EQ(resNum, 400);
    ccuRet = HcommCcuInsResDescQueryNum(resDescs[0], HCOMM_CCU_RES_TYPE_ADDRESS, &resNum);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    EXPECT_EQ(resNum, 400);
    ccuRet = HcommCcuInsResDescQueryNum(resDescs[0], HCOMM_CCU_RES_TYPE_EVENT, &resNum);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    EXPECT_EQ(resNum, 32 + 8 * 8 * 2);
    ccuRet = HcommCcuInsResDescQueryNum(resDescs[0], HCOMM_CCU_RES_TYPE_CCU_THREAD, &resNum);
    EXPECT_EQ(resNum, 2);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

// HcommCcuInsQueryResDesc 入参 ccuInsHandle 为 0，返回 CCU_E_PARA
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuInsQueryResDesc_When_InvalidInsHandle_Expect_ReturnCcuEPARA)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);

    ccuRet = HcommCcuInsQueryResDesc(0, resDescs[0]);
    EXPECT_EQ(ccuRet, CcuResult::CCU_E_PARA);

    DestroyCcuResDescs(resDescs);
}

// HcommCcuInsQueryResDesc 入参 resDesc 为 0，返回 CCU_E_PARA
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuInsQueryResDesc_When_NullResDesc_Expect_ReturnCcuEPARA)
{
    CcuResult ccuRet = CcuResult::CCU_E_RESERVED;
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 2;
    constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
    (void)MockCcuDeviceEnv(fakeDevId, fakeCcuVersion);

    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateCcuResDescsPair(resDescs, fakeCcuVersion);

    constexpr uint32_t descNum = 2;
    CcuInsHandle insHandle{0};
    ccuRet = HcommCcuInsCreate(resDescs, descNum, &insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);

    ccuRet = HcommCcuInsQueryResDesc(insHandle, 0);
    EXPECT_EQ(ccuRet, CcuResult::CCU_E_PARA);

    ccuRet = HcommCcuInsDestroy(insHandle);
    EXPECT_EQ(ccuRet, CcuResult::CCU_SUCCESS);
    DestroyCcuResDescs(resDescs);
}

class HcommCcuInstanceDeviceRefreshTest : public BaseInit {
public:
    void SetUp() override
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        BaseInit::SetUp();
        ResetDeviceState(TEST_DEVICE_LOGIC_ID);
        ResetDeviceState(OTHER_TEST_DEVICE_LOGIC_ID);

        MOCKER(GetExternalInputHcclEnableEntryLog).stubs().with(mockcpp::any()).will(returnValue(true));
        MockControlDeviceRefresh(TEST_DEVICE_LOGIC_ID);
        int32_t seededDeviceLogicId = INVALID_INT;
        ASSERT_EQ(HcclDeviceRefresh(seededDeviceLogicId), HcclResult::HCCL_SUCCESS);
        ASSERT_EQ(seededDeviceLogicId, TEST_DEVICE_LOGIC_ID);
        ASSERT_EQ(HcclGetThreadDeviceId(), TEST_DEVICE_LOGIC_ID);
        g_deviceRefreshCalls = 0;

        MOCKER(hrtGetDevice).stubs().with(mockcpp::any()).will(invoke(MockHrtGetDevice));
        MOCKER(hrtGetDevicePhyIdByIndex)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(invoke(MockHrtGetDevicePhyIdByUserDevId));
        constexpr hcomm::CcuVersion fakeCcuVersion = hcomm::CcuVersion::CCU_V1;
        MockCcuNetworkDeviceDefault(TEST_DEVICE_LOGIC_ID);
        ASSERT_EQ(MockCcuResourcesDefault(TEST_DEVICE_LOGIC_ID, fakeCcuVersion), HcclResult::HCCL_SUCCESS);
        ASSERT_EQ(InitMockCcuResourcesForDevice(OTHER_TEST_DEVICE_LOGIC_ID, fakeCcuVersion), HcclResult::HCCL_SUCCESS);
        MockCcuChannelGetRes();
        MOCKER(hrtMemcpy).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
        // PR 新增 CcuAllocCntXnBlock / CcuReleaseCntXnBlock 在 InitByResDescs 路径中被调用
        MOCKER(hcomm::CcuAllocCntXnBlock)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(returnValue(CcuResult::CCU_SUCCESS));
        MOCKER(hcomm::CcuReleaseCntXnBlock)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
            .will(returnValue(CcuResult::CCU_SUCCESS));
        ASSERT_EQ(hcomm::CcuInstanceMgr::GetInstance(TEST_DEVICE_LOGIC_ID).Init(), CcuResult::CCU_SUCCESS);
        ASSERT_EQ(hcomm::CcuInstanceMgr::GetInstance(OTHER_TEST_DEVICE_LOGIC_ID).Init(), CcuResult::CCU_SUCCESS);
    }

    void TearDown() override
    {
        g_runtimeDeviceLogicId = TEST_DEVICE_LOGIC_ID;
        g_deviceRefreshResult = HcclResult::HCCL_SUCCESS;
        int32_t restoredDeviceLogicId = INVALID_INT;
        EXPECT_EQ(HcclDeviceRefresh(restoredDeviceLogicId), HcclResult::HCCL_SUCCESS);
        EXPECT_EQ(restoredDeviceLogicId, TEST_DEVICE_LOGIC_ID);
        ResetDeviceState(TEST_DEVICE_LOGIC_ID);
        ResetDeviceState(OTHER_TEST_DEVICE_LOGIC_ID);
        BaseInit::TearDown();
        GlobalMockObject::verify();
        GlobalMockObject::reset();
    }

protected:
    static void ResetDeviceState(int32_t deviceLogicId)
    {
        (void)hcomm::CcuInstanceMgr::GetInstance(deviceLogicId).Deinit();
        (void)hcomm::CcuKernelMgr::GetInstance(deviceLogicId).Deinit();
        (void)hcomm::CcuResBatchAllocator::GetInstance(deviceLogicId).Deinit();
        (void)hcomm::CcuComponent::GetInstance(deviceLogicId).Deinit();
        (void)hcomm::CcuPfeCfgMgr::GetInstance(deviceLogicId).Deinit();
        (void)hcomm::CcuResSpecifications::GetInstance(deviceLogicId).Deinit();
    }

    static hcomm::CcuInstanceMgr& InsMgr(int32_t deviceLogicId)
    {
        return hcomm::CcuInstanceMgr::GetInstance(deviceLogicId);
    }

    static void CreateResDescsDirect(int32_t deviceLogicId, HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM])
    {
        auto& resDescMgr = InsMgr(deviceLogicId).GetResDescMgr();
        for (uint32_t dieId = 0; dieId < hcomm::CCU_MAX_IODIE_NUM; ++dieId) {
            ASSERT_EQ(resDescMgr.Create(dieId, resDescs[dieId]), CcuResult::CCU_SUCCESS);
            ASSERT_NE(resDescs[dieId], 0U);
            ASSERT_EQ(resDescMgr.SetResNum(resDescs[dieId], hcomm::ResType::LOOP, 8 * 8 * 2), CcuResult::CCU_SUCCESS);
            ASSERT_EQ(resDescMgr.SetResNum(resDescs[dieId], hcomm::ResType::MS, 64 * 8 * 2), CcuResult::CCU_SUCCESS);
            ASSERT_EQ(resDescMgr.SetResNum(resDescs[dieId], hcomm::ResType::XN, 400), CcuResult::CCU_SUCCESS);
            ASSERT_EQ(resDescMgr.SetResNum(resDescs[dieId], hcomm::ResType::GSA, 400), CcuResult::CCU_SUCCESS);
            ASSERT_EQ(
                resDescMgr.SetResNum(resDescs[dieId], hcomm::ResType::CKE, 32 + 8 * 8 * 2), CcuResult::CCU_SUCCESS);
            ASSERT_EQ(resDescMgr.SetResNum(resDescs[dieId], hcomm::ResType::MISSION, 2), CcuResult::CCU_SUCCESS);
        }
    }
};

// 验证实例创建、资源查询和销毁均使用刷新后的运行时 Device。
TEST_F(
    HcommCcuInstanceDeviceRefreshTest,
    Ut_HcommCcuInstanceApis_When_RuntimeDeviceSwitches_Expect_UseRefreshedDeviceForCreateQueryDestroy)
{
    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateResDescsDirect(OTHER_TEST_DEVICE_LOGIC_ID, resDescs);
    g_runtimeDeviceLogicId = OTHER_TEST_DEVICE_LOGIC_ID;

    CcuInsHandle insHandle = 0;
    EXPECT_EQ(HcommCcuInsCreate(resDescs, hcomm::CCU_MAX_IODIE_NUM, &insHandle), CcuResult::CCU_SUCCESS);
    ASSERT_NE(insHandle, 0U);
    EXPECT_EQ(InsMgr(TEST_DEVICE_LOGIC_ID).Get(insHandle), nullptr);
    EXPECT_NE(InsMgr(OTHER_TEST_DEVICE_LOGIC_ID).Get(insHandle), nullptr);
    EXPECT_EQ(HcommCcuInsQueryResDesc(insHandle, resDescs[0]), CcuResult::CCU_SUCCESS);
    EXPECT_EQ(HcommCcuInsDestroy(insHandle), CcuResult::CCU_SUCCESS);
    EXPECT_EQ(InsMgr(OTHER_TEST_DEVICE_LOGIC_ID).Get(insHandle), nullptr);
    EXPECT_EQ(g_deviceRefreshCalls, 3U);
}

// 验证 Legacy 和 Default 创建入口在运行时 Device 切换后均先刷新并路由到新 Device。
TEST_F(
    HcommCcuInstanceDeviceRefreshTest,
    Ut_HcommCcuDefaultAndLegacyCreate_When_RuntimeDeviceSwitches_Expect_UseRefreshedDevice)
{
    g_runtimeDeviceLogicId = OTHER_TEST_DEVICE_LOGIC_ID;

    CcuInsHandle legacyHandle = 0;
    EXPECT_EQ(HcommCcuInsCreateLegacy(CcuInstanceType::CCU_SCHED, &legacyHandle), CcuResult::CCU_SUCCESS);
    ASSERT_NE(legacyHandle, 0U);
    EXPECT_EQ(InsMgr(TEST_DEVICE_LOGIC_ID).Get(legacyHandle), nullptr);
    EXPECT_NE(InsMgr(OTHER_TEST_DEVICE_LOGIC_ID).Get(legacyHandle), nullptr);
    EXPECT_EQ(g_deviceRefreshCalls, 1U);
    EXPECT_EQ(HcommCcuInsDestroy(legacyHandle), CcuResult::CCU_SUCCESS);
    EXPECT_EQ(g_deviceRefreshCalls, 2U);

    CcuInsHandle defaultHandle = 0;
    // Mock 资源池无法满足 Default 接口的全量资源申请，因此预期返回 CCU_E_UNAVAIL；
    // 通过刷新次数和线程 DeviceId 验证接口已路由到切换后的 Device。
    EXPECT_EQ(HcommCcuInsCreateDefault(nullptr, 0, &defaultHandle), CcuResult::CCU_E_UNAVAIL);
    EXPECT_EQ(defaultHandle, 0U);
    EXPECT_EQ(HcclGetThreadDeviceId(), OTHER_TEST_DEVICE_LOGIC_ID);
    EXPECT_EQ(g_deviceRefreshCalls, 3U);
}

// 验证 Device 刷新失败时，所有实例生命周期入口立即返回错误且不改变管理器状态。
TEST_F(
    HcommCcuInstanceDeviceRefreshTest,
    Ut_HcommCcuInstanceApis_When_DeviceRefreshFails_Expect_ReturnErrorWithoutManagerMutation)
{
    HcommCcuResDescHandle resDescs[hcomm::CCU_MAX_IODIE_NUM] = {0, 0};
    CreateResDescsDirect(TEST_DEVICE_LOGIC_ID, resDescs);
    CcuInsHandle existingHandle = 0;
    ASSERT_EQ(HcommCcuInsCreateLegacy(CcuInstanceType::CCU_SCHED, &existingHandle), CcuResult::CCU_SUCCESS);
    ASSERT_NE(existingHandle, 0U);
    ASSERT_NE(InsMgr(TEST_DEVICE_LOGIC_ID).Get(existingHandle), nullptr);

    uint32_t originalLoopNum = 0;
    ASSERT_EQ(
        InsMgr(TEST_DEVICE_LOGIC_ID).GetResDescMgr().QueryResNum(resDescs[0], hcomm::ResType::LOOP, originalLoopNum),
        CcuResult::CCU_SUCCESS);
    g_deviceRefreshResult = HcclResult::HCCL_E_INTERNAL;
    g_deviceRefreshCalls = 0;

    CcuInsHandle createHandle = 0x1111;
    CcuInsHandle defaultHandle = 0x2222;
    CcuInsHandle legacyHandle = 0x3333;
    EXPECT_EQ(HcommCcuInsCreate(resDescs, hcomm::CCU_MAX_IODIE_NUM, &createHandle), CcuResult::CCU_E_INTERNAL);
    EXPECT_EQ(createHandle, 0x1111U);
    EXPECT_EQ(HcommCcuInsCreateDefault(nullptr, 0, &defaultHandle), CcuResult::CCU_E_INTERNAL);
    EXPECT_EQ(defaultHandle, 0x2222U);
    EXPECT_EQ(HcommCcuInsQueryResDesc(existingHandle, resDescs[0]), CcuResult::CCU_E_INTERNAL);
    EXPECT_EQ(HcommCcuInsDestroy(existingHandle), CcuResult::CCU_E_INTERNAL);
    EXPECT_EQ(HcommCcuInsCreateLegacy(CcuInstanceType::CCU_SCHED, &legacyHandle), CcuResult::CCU_E_INTERNAL);
    EXPECT_EQ(legacyHandle, 0x3333U);

    uint32_t loopNum = 0;
    EXPECT_EQ(
        InsMgr(TEST_DEVICE_LOGIC_ID).GetResDescMgr().QueryResNum(resDescs[0], hcomm::ResType::LOOP, loopNum),
        CcuResult::CCU_SUCCESS);
    EXPECT_EQ(loopNum, originalLoopNum);
    EXPECT_NE(InsMgr(TEST_DEVICE_LOGIC_ID).Get(existingHandle), nullptr);
    EXPECT_EQ(g_deviceRefreshCalls, 5U);
}

CCU_FUNC_KERNEL_TEST(Ut_HcommCcuKernelLocalCopy_When_AllFine_Expect_ReturnCcuSUCCESS, CcuLocalCopyKernel, true)

// ============================================================================================
// HcommCcuGetTaskArgsNum 单元测试
// 对应业务代码：src/base_comm/primitives/api_c_adpt/ccu/ccu_launch.cc
// 该接口取已注册 kernel 注册期间所有 LoadArg 中 argId 的最大值加1，写入 *taskArgsNum。
// 调用链：HcommCcuGetTaskArgsNum -> CcuKernelMgr::GetInstance -> GetCcuKernelInfo -> CcuKernel::GetCcuKernelInfo
// 由于 #define private public 已启用，可直接操作单例 kernelMap_ 注入 fake kernel 精确测试
// ============================================================================================

// 功能用例：loadArgUsedSet_ 含多个 argId，验证返回最大值
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuGetTaskArgsNum_When_MultiArgIds_Expect_MaxArgId)
{
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 10;
    MOCKER(HcclGetThreadDeviceId).stubs().will(returnValue(fakeDevId));

    // 通过单例注入 fake kernel，设置 loadArgUsedSet_ = {0, 3, 1}，max=3，+1=4
    constexpr CcuKernelHandle fakeHandle = 0xBEEF;
    auto& kernelMgr = hcomm::CcuKernelMgr::GetInstance(static_cast<int32_t>(fakeDevId));
    auto fakeKernel = std::make_unique<hcomm::CcuKernel>();
    fakeKernel->loadArgUsedSet_ = {0, 3, 1};
    kernelMgr.kernelMap_[fakeHandle] = std::move(fakeKernel);

    uint32_t taskArgsNum = 0xFFFFFFFF;
    CcuResult ret = HcommCcuGetTaskArgsNum(fakeHandle, &taskArgsNum);
    EXPECT_EQ(ret, CcuResult::CCU_SUCCESS);
    EXPECT_EQ(taskArgsNum, 4U) << "应返回 loadArgUsedSet_ 中最大 argId + 1";

    // 清理 fake kernel，避免单例状态污染后续用例
    kernelMgr.kernelMap_.erase(fakeHandle);
}

// 异常用例：taskArgsNum 为空指针，应返回 CCU_E_PTR
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuGetTaskArgsNum_When_NullPtr_Expect_CcuEPtr)
{
    constexpr CcuKernelHandle fakeHandle = 0xBEEF;
    CcuResult ret = HcommCcuGetTaskArgsNum(fakeHandle, nullptr);
    EXPECT_EQ(ret, CcuResult::CCU_E_PTR);
}

// 异常用例：kernelHandle 不存在，GetCcuKernelInfo 在锁内查找失败返回 CCU_E_NOT_FOUND
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuGetTaskArgsNum_When_InvalidHandle_Expect_CcuENotFound)
{
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 10;
    MOCKER(HcclGetThreadDeviceId).stubs().will(returnValue(fakeDevId));

    // 不注册任何 kernel，使用不存在的 handle
    constexpr CcuKernelHandle invalidHandle = 0xDEAD;
    uint32_t taskArgsNum = 0xFFFFFFFF;
    CcuResult ret = HcommCcuGetTaskArgsNum(invalidHandle, &taskArgsNum);
    EXPECT_EQ(ret, CcuResult::CCU_E_NOT_FOUND);
    // 出参不应被修改（GetCcuKernelInfo 在锁内查找失败，未写入 info）
    EXPECT_EQ(taskArgsNum, 0xFFFFFFFFu);
}

// 安全/边界用例：未调用 LoadArg（loadArgUsedSet_ 为空），应返回 0 并覆盖脏出参
TEST_F(HcommCcuControlApiTest, Ut_HcommCcuGetTaskArgsNum_When_NoLoadArg_Expect_Zero)
{
    constexpr uint32_t fakeDevId = MAX_MODULE_DEVICE_NUM - 10;
    MOCKER(HcclGetThreadDeviceId).stubs().will(returnValue(fakeDevId));

    // 注入 fake kernel，loadArgUsedSet_ 保持默认空集，GetCcuKernelInfo 填充 maxTaskArgsNum=0
    constexpr CcuKernelHandle fakeHandle = 0xBEEF;
    auto& kernelMgr = hcomm::CcuKernelMgr::GetInstance(static_cast<int32_t>(fakeDevId));
    auto fakeKernel = std::make_unique<hcomm::CcuKernel>();
    // loadArgUsedSet_ 默认为空，不 insert 任何 argId
    kernelMgr.kernelMap_[fakeHandle] = std::move(fakeKernel);

    // 出参初始为脏值 0xFFFFFFFF，验证被正确覆盖为 0
    uint32_t taskArgsNum = 0xFFFFFFFF;
    CcuResult ret = HcommCcuGetTaskArgsNum(fakeHandle, &taskArgsNum);
    EXPECT_EQ(ret, CcuResult::CCU_SUCCESS);
    EXPECT_EQ(taskArgsNum, 0U) << "未调用 LoadArg 时 maxArgId 应为 0，脏出参应被覆盖";

    // 清理 fake kernel
    kernelMgr.kernelMap_.erase(fakeHandle);
}

TEST_F(HcommCcuControlApiTest, Ut_CcuChannelIdGet_When_Normal_Expect_Success)
{
    HcommChannelDesc desc{};
    hcomm::CcuUrmaChannel channel(nullptr, desc);
    void* channelPtr = static_cast<hcomm::Channel*>(&channel);
    ChannelHandle handle = 0xAAAA;

    MOCKER(HcommChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(&channelPtr))
        .will(returnValue(static_cast<HcommResult>(0)));

    const uint32_t expectedId = 101U;
    MOCKER_CPP(&hcomm::CcuUrmaChannel::GetChannelId).stubs().will(returnValue(expectedId));

    uint32_t channelId = 0;
    CcuResult ret = CcuChannelIdGet(handle, &channelId);
    EXPECT_EQ(ret, CcuResult::CCU_SUCCESS);
    EXPECT_EQ(channelId, expectedId);
}

TEST_F(HcommCcuControlApiTest, Ut_CcuChannelIdGet_When_NullId_Expect_PtrError)
{
    ChannelHandle handle = 0xBBBB;
    CcuResult ret = CcuChannelIdGet(handle, nullptr);
    EXPECT_EQ(ret, CcuResult::CCU_E_PTR);
}

TEST_F(HcommCcuControlApiTest, Ut_CcuChannelIdGet_When_HcommChannelGetFail_Expect_ReturnError)
{
    ChannelHandle handle = 0xCCCC;
    MOCKER(HcommChannelGet)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(static_cast<HcommResult>(HCCL_E_PARA)));

    uint32_t channelId = 0;
    CcuResult ret = CcuChannelIdGet(handle, &channelId);
    EXPECT_EQ(ret, CcuResult::CCU_E_PARA);
}

TEST_F(HcommCcuControlApiTest, Ut_CcuChannelIdGet_When_ChannelPtrNull_Expect_PtrError)
{
    void* nullPtr = nullptr;
    ChannelHandle handle = 0xDDDD;
    MOCKER(HcommChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(&nullPtr))
        .will(returnValue(static_cast<HcommResult>(0)));

    uint32_t channelId = 0;
    CcuResult ret = CcuChannelIdGet(handle, &channelId);
    EXPECT_EQ(ret, CcuResult::CCU_E_PTR);
}

TEST_F(HcommCcuControlApiTest, Ut_CcuChannelIdGet_When_ChannelIdInvalid_Expect_Unavail)
{
    HcommChannelDesc desc{};
    hcomm::CcuUrmaChannel channel(nullptr, desc);
    void* channelPtr = static_cast<hcomm::Channel*>(&channel);
    ChannelHandle handle = 0xEEEE;

    MOCKER(HcommChannelGet)
        .stubs()
        .with(mockcpp::any(), outBoundP(&channelPtr))
        .will(returnValue(static_cast<HcommResult>(0)));
    MOCKER_CPP(&hcomm::CcuUrmaChannel::GetChannelId).stubs().will(returnValue(UINT32_MAX));

    uint32_t channelId = 0;
    CcuResult ret = CcuChannelIdGet(handle, &channelId);
    EXPECT_EQ(ret, CcuResult::CCU_E_UNAVAIL);
}

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../hccl_api_base_test.h"
#include "hccl/hccl_launch.h"
#include "hccl_independent_common.h"
#include "acl/acl_rt.h"
#include "hccl_group.h"
#include "hccl/hccl_diag.h"
#include "hccl/hccl_res_expt.h"
#include "hcomm_primitives.h"
#include <vector>
#include <memory>

#define private public
#include "group_schedule_mgr.h"
#undef private

extern HcclResult groupLaunchA5();

class TestKernelLaunchAicpu : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        SetHcclP2pTaskNums(0);
        GetHcclGroupCommList().clear();
    }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
        SetHcclP2pTaskNums(0);
        GetHcclGroupCommList().clear();
    }
};

static aclrtStream g_usrStream = reinterpret_cast<aclrtStream>(0x4000);
static uint32_t g_notifyNum = 2;
static aclrtFuncHandle g_mockFuncHandle = reinterpret_cast<aclrtFuncHandle>(0x7000);
static aclrtArgsHandle g_mockArgsHandle = reinterpret_cast<aclrtArgsHandle>(0x8000);
static aclrtParamHandle g_mockParamHandle = reinterpret_cast<aclrtParamHandle>(0x9000);

HcclResult MockGetUsrStream(aclrtStream& usrStream)
{
    usrStream = g_usrStream;
    return HCCL_SUCCESS;
}

HcclResult MockHcclGetNotifyNumInThread(HcclComm, ThreadHandle, CommEngine, uint32_t* notifyNum)
{
    *notifyNum = g_notifyNum;
    return HCCL_SUCCESS;
}

aclError MockAclrtBinaryGetFunction(const aclrtBinHandle, const char*, aclrtFuncHandle* funcHandle)
{
    *funcHandle = g_mockFuncHandle;
    return ACL_SUCCESS;
}

aclError MockAclrtKernelArgsInit(aclrtFuncHandle, aclrtArgsHandle* argsHandle)
{
    *argsHandle = g_mockArgsHandle;
    return ACL_SUCCESS;
}

aclError MockAclrtKernelArgsAppend(aclrtArgsHandle, void*, uint64_t, aclrtParamHandle* paramHandle)
{
    *paramHandle = g_mockParamHandle;
    return ACL_SUCCESS;
}

TEST_F(TestKernelLaunchAicpu, Ut_HcclAicpuKernelLaunch_When_CommIsNull_Expect_HCCL_E_PTR)
{
    HcclOpDesc opInfo;
    HcclKernelFuncInfo funcInfo;
    ThreadHandle aicpuThreadHandle = 0;
    aclrtStream userStream = reinterpret_cast<aclrtStream>(0x1000);
    HcclKernelLaunchCfg kernelLaunchCfg;
    kernelLaunchCfg.timeOut = 120U;

    HcclResult ret
        = HcclAicpuKernelLaunch(nullptr, &opInfo, &funcInfo, aicpuThreadHandle, userStream, &kernelLaunchCfg);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestKernelLaunchAicpu, Ut_HcclAicpuKernelLaunch_When_FuncInfoIsNull_Expect_HCCL_E_PTR)
{
    HcclComm comm = reinterpret_cast<HcclComm>(0x1000);
    HcclOpDesc opInfo;
    ThreadHandle aicpuThreadHandle = 0;
    aclrtStream userStream = reinterpret_cast<aclrtStream>(0x1000);
    HcclKernelLaunchCfg kernelLaunchCfg;
    kernelLaunchCfg.timeOut = 120U;

    HcclResult ret = HcclAicpuKernelLaunch(comm, &opInfo, nullptr, aicpuThreadHandle, userStream, &kernelLaunchCfg);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestKernelLaunchAicpu, Ut_HcclAicpuKernelLaunch_When_UserStreamIsNull_Expect_HCCL_E_PTR)
{
    HcclComm comm = reinterpret_cast<HcclComm>(0x1000);
    HcclOpDesc opInfo;
    HcclKernelFuncInfo funcInfo;
    ThreadHandle aicpuThreadHandle = 0;
    HcclKernelLaunchCfg kernelLaunchCfg;
    kernelLaunchCfg.timeOut = 120U;

    HcclResult ret = HcclAicpuKernelLaunch(comm, &opInfo, &funcInfo, aicpuThreadHandle, nullptr, &kernelLaunchCfg);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestKernelLaunchAicpu, Ut_HcclAicpuKernelLaunch_When_ArgSizePositiveButArgsNull_Expect_HCCL_E_PTR)
{
    HcclComm comm = reinterpret_cast<HcclComm>(0x1000);
    HcclOpDesc opInfo;
    HcclKernelFuncInfo funcInfo;
    funcInfo.argSize = 100;
    funcInfo.args = nullptr;
    ThreadHandle aicpuThreadHandle = 0;
    aclrtStream userStream = reinterpret_cast<aclrtStream>(0x1000);
    HcclKernelLaunchCfg kernelLaunchCfg;
    kernelLaunchCfg.timeOut = 120U;

    HcclResult ret = HcclAicpuKernelLaunch(comm, &opInfo, &funcInfo, aicpuThreadHandle, userStream, &kernelLaunchCfg);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestKernelLaunchAicpu, Ut_HcclAicpuKernelLaunch_Expect_AicpuKernelLaunchDirectPassed)
{
    hccl::hcclComm mockHcclComm;
    HcclComm comm = static_cast<HcclComm>(&mockHcclComm);
    GetHcclGroupCommList().push_back(comm);
    hcclGroupDepth = 0;
    uint8_t arg;
    HcclOpDesc opInfo;
    opInfo.p2p.unfoldStream = reinterpret_cast<aclrtStream>(0x2000);
    HcclKernelFuncInfo funcInfo;
    funcInfo.argSize = 1;
    funcInfo.args = static_cast<void*>(&arg);
    ThreadHandle aicpuThreadHandle = 0;
    aclrtStream userStream = reinterpret_cast<aclrtStream>(0x1000);
    HcclKernelLaunchCfg kernelLaunchCfg;
    kernelLaunchCfg.timeOut = 120U;

    static CollComm g_mockCollComm(nullptr, 0, "test", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    MOCKER_CPP(&hcclComm::GetCollComm).stubs().will(returnValue(&g_mockCollComm));
    MOCKER_CPP(&hcclComm::GetBinHandle).stubs().will(returnValue((aclrtBinHandle)0x1000));
    MOCKER(HcclThreadAcquire).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HcclThreadAcquireWithStream).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HcclThreadExportToCommEngine).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HcclGetNotifyNumInThread).stubs().will(invoke(MockHcclGetNotifyNumInThread));
    MOCKER(HcclGetCommName).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HcommThreadNotifyRecordOnThread).stubs().will(returnValue(0));
    MOCKER(HcommThreadNotifyWaitOnThreadWithDefaultTimeout).stubs().will(returnValue(0));
    MOCKER(aclrtBinaryGetFunction).stubs().will(invoke(MockAclrtBinaryGetFunction));
    MOCKER(aclrtKernelArgsInit).stubs().will(invoke(MockAclrtKernelArgsInit));
    MOCKER(aclrtKernelArgsAppend).stubs().will(invoke(MockAclrtKernelArgsAppend));
    MOCKER(aclrtKernelArgsFinalize).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtLaunchKernelWithConfig).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(HcclReportAicpuKernel).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(HcommGetProfilingSysCycleTime).stubs().will(returnValue(0x1000U));

    HcclResult ret = HcclAicpuKernelLaunch(comm, &opInfo, &funcInfo, aicpuThreadHandle, userStream, &kernelLaunchCfg);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(TestKernelLaunchAicpu, Ut_groupLaunchA5_When_OneSendAndOneRecvTask_Expect_HCCL_SUCCESS)
{
    GetHcclGroupCommList().clear();
    SetHcclP2pTaskNums(2);
    hccl::hcclComm mockHcclComm;
    HcclComm comm = static_cast<HcclComm>(&mockHcclComm);
    GetHcclGroupCommList().push_back(comm);
    static CollComm g_mockCollComm(nullptr, 0, "test", ManagerCallbacks{}, CollCommInitMode::simpleMode);
    g_mockCollComm.groupScheduleMgr.reset(new GroupScheduleMgr());

    std::vector<HcclP2pTask> mockSendQue;
    std::vector<HcclP2pTask> mockRecvQue;
    HcclP2pTask sendTask;
    sendTask.stream = reinterpret_cast<aclrtStream>(0x3000);
    sendTask.usrStream = reinterpret_cast<aclrtStream>(0x4000);
    sendTask.argSize = 1;
    sendTask.funcInfo.kernelFuncName[0] = '\0';
    sendTask.funcInfo.kernelSoName[0] = '\0';
    HcclP2pTask recvTask;
    recvTask.stream = reinterpret_cast<aclrtStream>(0x5000);
    recvTask.usrStream = reinterpret_cast<aclrtStream>(0x6000);
    recvTask.argSize = 1;
    recvTask.funcInfo.kernelFuncName[0] = '\0';
    recvTask.funcInfo.kernelSoName[0] = '\0';
    mockSendQue.push_back(sendTask);
    mockRecvQue.push_back(recvTask);

    MOCKER_CPP(&hcclComm::GetCollComm).stubs().will(returnValue(&g_mockCollComm));
    MOCKER_CPP(&hcclComm::GetBinHandle).stubs().will(returnValue((aclrtBinHandle)0x1000));
    MOCKER_CPP(
        &GroupScheduleMgr::GetP2pTaskSchedule,
        HcclResult(GroupScheduleMgr::*)(std::vector<HcclP2pTask>&, std::vector<HcclP2pTask>&))
        .stubs()
        .with(outBound(mockSendQue), outBound(mockRecvQue))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(HcclThreadAcquire).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HcclThreadAcquireWithStream).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HcclThreadExportToCommEngine).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HcclGetNotifyNumInThread).stubs().will(invoke(MockHcclGetNotifyNumInThread));
    MOCKER(HcommThreadNotifyRecordOnThread).stubs().will(returnValue(0));
    MOCKER(HcommThreadNotifyWaitOnThreadWithDefaultTimeout).stubs().will(returnValue(0));
    MOCKER(aclrtBinaryGetFunction).stubs().will(invoke(MockAclrtBinaryGetFunction));
    MOCKER(aclrtKernelArgsInit).stubs().will(invoke(MockAclrtKernelArgsInit));
    MOCKER(aclrtKernelArgsAppend).stubs().will(invoke(MockAclrtKernelArgsAppend));
    MOCKER(aclrtKernelArgsFinalize).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtLaunchKernelWithConfig).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(HcclGetCommName).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HcclReportAicpuKernel).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HcommGetProfilingSysCycleTime).stubs().will(returnValue(0x1000U));
    MOCKER(HcclProfilingReportOp).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = groupLaunchA5();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(TestKernelLaunchAicpu, Ut_HcclAicpuKernelLaunch_When_OpInfoNull_Expect_HCCL_E_PTR)
{
    HcclComm comm = reinterpret_cast<HcclComm>(0x1000);
    HcclKernelFuncInfo funcInfo;
    ThreadHandle aicpuThreadHandle = 0;
    aclrtStream userStream = reinterpret_cast<aclrtStream>(0x1000);
    HcclKernelLaunchCfg kernelLaunchCfg;
    kernelLaunchCfg.timeOut = 120U;

    HcclResult ret = HcclAicpuKernelLaunch(comm, nullptr, &funcInfo, aicpuThreadHandle, userStream, &kernelLaunchCfg);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestKernelLaunchAicpu, Ut_HcclAicpuKernelLaunch_When_KernelLaunchCfgNull_Expect_HCCL_E_PTR)
{
    HcclComm comm = reinterpret_cast<HcclComm>(0x1000);
    HcclOpDesc opInfo;
    HcclKernelFuncInfo funcInfo;
    ThreadHandle aicpuThreadHandle = 0;
    aclrtStream userStream = reinterpret_cast<aclrtStream>(0x1000);

    HcclResult ret = HcclAicpuKernelLaunch(comm, &opInfo, &funcInfo, aicpuThreadHandle, userStream, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestKernelLaunchAicpu, Ut_groupLaunchA5_When_NoGroupComm_Expect_HCCL_SUCCESS)
{
    GetHcclGroupCommList().clear();
    SetHcclP2pTaskNums(0);
    HcclResult ret = groupLaunchA5();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

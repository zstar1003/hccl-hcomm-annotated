/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>

#ifndef private
#define private public
#define protected public
#endif

#include "hccl_aicpu_interface.h"
#include "aicpu_communicator.h"
#include "aicpu_hccl_process.h"
#include "aicpu_sqe_context.h"
#include "aicpu_hdc_utils.h"
#include "hcomm_primitives.h"
#include "hcomm_diag.h"
#include <securec.h>

#undef private
#undef protected

using namespace std;
using namespace hccl;

class Test_Hccl_Aicpu_Interface : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "UT_Hccl_Aicpu_Interface SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "UT_Hccl_Aicpu_Interface TearDown" << std::endl; }
    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};

HcclResult GetSuspendingFlagStub(HcclCommAicpu* This, HcclComSuspendingFlag& flag)
{
    flag = HcclComSuspendingFlag::isSuspending;
    return HCCL_SUCCESS;
}

HcclCommAicpu* CreateHcclCommAicpuWithDefaultDfx()
{
    HcclCommAicpu* hcclCommAicpu = new HcclCommAicpu();
    DfxExtendInfo* dfxInfo = hcclCommAicpu->GetDfxExtendInfo();
    dfxInfo->cqeStatus = dfx::CqeStatus::kDefault;
    dfxInfo->pollStatus = PollStatus::kDefault;
    return hcclCommAicpu;
}

HcclOpResParam* CreateHcclOpResParam(const char* groupName)
{
    HcclOpResParam* commParam = new HcclOpResParam();
    memset(commParam, 0, sizeof(HcclOpResParam));
    strcpy(commParam->hcomId, groupName);
    return commParam;
}

OpTilingData* CreateOpTilingData(HcclCMDType opType, const char* tag)
{
    OpTilingData* tilingData = new OpTilingData();
    memset(tilingData, 0, sizeof(OpTilingData));
    tilingData->opType = static_cast<u8>(opType);
    strcpy(tilingData->tag, tag);
    tilingData->workflowMode = 0;
    tilingData->isZeroCopy = 0;
    tilingData->isSymmetricMemory = 0;
    return tilingData;
}

void* BuildKfcTaskComm(HcclOpResParam* commParam, OpTilingData* tilingData)
{
    u64 taskSize = sizeof(KFCTaskComm);
    u64 totalSize = taskSize + sizeof(OpTilingData);
    u8* buffer = new u8[totalSize];
    memset(buffer, 0, totalSize);

    KFCTaskComm* task = reinterpret_cast<KFCTaskComm*>(buffer);
    task->context = reinterpret_cast<u64>(commParam);

    OpTilingData* tilingInTask = reinterpret_cast<OpTilingData*>(buffer + sizeof(KFCTaskComm));
    memcpy(tilingInTask, tilingData, sizeof(OpTilingData));
    task->tilingData = reinterpret_cast<u64>(tilingInTask);

    return reinterpret_cast<void*>(task);
}

void SetupRpcSrvLaunchMocks(HcclCommAicpu* hcclCommAicpu)
{
    MOCKER(AicpuHcclProcess::AicpuGetCommbyGroup).stubs().will(returnValue(hcclCommAicpu));

    MOCKER_CPP(&HcclCommAicpu::GetSuspendingFlag).stubs().will(invoke(GetSuspendingFlagStub));

    MOCKER(AicpuHcclProcess::AicpuReleaseCommbyGroup).stubs().will(ignoreReturnValue());
}

static u8 g_capturedExecOpUnfoldMode = 0;

HcclResult ExecOpCaptureUnfoldModeStub(
    HcclCommAicpu* /*This*/, const std::string& /*newTag*/, const std::string& /*algName*/, OpParam& opParam,
    const HcclOpResParam* /*commParam*/)
{
    g_capturedExecOpUnfoldMode = opParam.aicpuUnfoldMode ? 1U : 0U;
    return HCCL_SUCCESS;
}

HcclResult RecordHostOrderStub(
    HcclCommAicpu* /*This*/, const HcclOpResParam* /*commParam*/, const std::string& /*tag*/, u8 /*orderLaunchMode*/)
{
    return HCCL_SUCCESS;
}

HcclResult UpdateNotifyWaitTimeOutStub(HcclCommAicpu* /*This*/, SyncMode /*syncMode*/, u64 /*notifyWaitTime*/)
{
    return HCCL_SUCCESS;
}

HcclResult ParseHierarchicalAlgOptionStub(HcclCommAicpu* /*This*/, u32* /*ahcConfInfo*/) { return HCCL_SUCCESS; }

HcclResult SaveTraceInfoStub(HcclCommAicpu* /*This*/, std::string& /*logInfo*/) { return HCCL_SUCCESS; }

// AicpuRunRpcServerV2 会按 opType 读取 OpTilingData 后的动态数据区，此处额外预留 OpTilingDataDes 空间避免越界
u8* CreateAllgatherTilingDataBuffer(const char* tag, u8 aicpuUnfoldMode)
{
    u64 totalSize = sizeof(OpTilingData) + sizeof(OpTilingDataDes);
    u8* buffer = new u8[totalSize];
    memset_s(buffer, totalSize, 0, totalSize);
    OpTilingData* tilingData = reinterpret_cast<OpTilingData*>(buffer);
    tilingData->opType = static_cast<u8>(HcclCMDType::HCCL_CMD_ALLGATHER);
    strcpy_s(tilingData->tag, sizeof(tilingData->tag), tag);
    tilingData->aicpuUnfoldMode = aicpuUnfoldMode;
    return buffer;
}

TEST_F(Test_Hccl_Aicpu_Interface, Ut_RunAicpuRpcSrvLaunchV2_When_SuspendingFlagIsSuspending_Expect_ReturnZero)
{
    HcclCommAicpu* hcclCommAicpu = CreateHcclCommAicpuWithDefaultDfx();
    SetupRpcSrvLaunchMocks(hcclCommAicpu);

    HcclOpResParam* commParam = CreateHcclOpResParam("test_group");
    OpTilingData* tilingData = CreateOpTilingData(HcclCMDType::HCCL_CMD_ALL, "test_tag");
    void* task = BuildKfcTaskComm(commParam, tilingData);

    uint32_t ret = RunAicpuRpcSrvLaunchV2(task);

    EXPECT_EQ(ret, 0);

    u8* buffer = reinterpret_cast<u8*>(task);
    delete[] buffer;
    delete commParam;
    delete tilingData;
    delete hcclCommAicpu;
}

ThreadNotifyWaitParam*
CreateNotifyWaitParam(const char* commName, uint64_t thread, uint32_t notifyIdx, uint32_t dataType)
{
    ThreadNotifyWaitParam* param = new ThreadNotifyWaitParam();
    memset(param, 0, sizeof(ThreadNotifyWaitParam));
    strncpy_s(param->commName, COMM_NAME_MAX_LENGTH, commName, strlen(commName));
    param->thread = thread;
    param->notifyIdx = notifyIdx;
    param->dataType = dataType;
    return param;
}

ThreadNotifyRecordParam* CreateNotifyRecordParam(
    const char* commName, uint64_t thread, uint64_t dstThread, uint32_t dstNotifyIdx, uint32_t dataType)
{
    ThreadNotifyRecordParam* param = new ThreadNotifyRecordParam();
    memset(param, 0, sizeof(ThreadNotifyRecordParam));
    strncpy_s(param->commName, COMM_NAME_MAX_LENGTH, commName, strlen(commName));
    param->thread = thread;
    param->dstThread = dstThread;
    param->dstNotifyIdx = dstNotifyIdx;
    param->dataType = dataType;
    return param;
}

TEST_F(Test_Hccl_Aicpu_Interface, Ut_RunAicpuNotifyWaitAicpuKernel_When_AllSuccess_Expect_ReturnHCCL_SUCCESS)
{
    ThreadNotifyWaitParam* param = CreateNotifyWaitParam("test_comm", 0x1234, 2, 1);

    MOCKER(HcommAcquireComm).stubs().will(returnValue(0));
    MOCKER(HcclDfxRegOpInfoByCommId).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HcommThreadNotifyWaitOnThreadWithDefaultTimeout).stubs().will(returnValue(0));
    MOCKER(HcommReleaseComm).stubs().will(returnValue(0));

    uint32_t ret = RunAicpuNotifyWaitAicpuKernel(param);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete param;
}

TEST_F(Test_Hccl_Aicpu_Interface, Ut_RunAicpuNotifyRecordAicpuKernel_When_AllSuccess_Expect_ReturnHCCL_SUCCESS)
{
    ThreadNotifyRecordParam* param = CreateNotifyRecordParam("test_comm", 0x1234, 0x5678, 0, 1);

    MOCKER(HcommAcquireComm).stubs().will(returnValue(0));
    MOCKER(HcclDfxRegOpInfoByCommId).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HcommThreadNotifyRecordOnThread).stubs().will(returnValue(0));
    MOCKER(HcommReleaseComm).stubs().will(returnValue(0));

    uint32_t ret = RunAicpuNotifyRecordAicpuKernel(param);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete param;
}

TEST_F(Test_Hccl_Aicpu_Interface, Ut_AicpuRunRpcServerV2_When_AicpuUnfoldModeEnabled_Expect_PropagatedToOpParam)
{
    HcclCommAicpu* hcclCommAicpu = new HcclCommAicpu();
    ASSERT_NE(hcclCommAicpu, nullptr);
    HcclOpResParam* commParam = CreateHcclOpResParam("test_group");
    ASSERT_NE(commParam, nullptr);
    u8* buffer = CreateAllgatherTilingDataBuffer("test_tag", 1);
    OpTilingData* tilingData = reinterpret_cast<OpTilingData*>(buffer);
    g_capturedExecOpUnfoldMode = 0xFF;

    MOCKER_CPP(&HcclCommAicpu::RecordHostOrder).stubs().will(invoke(RecordHostOrderStub));
    MOCKER_CPP(&HcclCommAicpu::UpdateNotifyWaitTimeOut).stubs().will(invoke(UpdateNotifyWaitTimeOutStub));
    MOCKER_CPP(&HcclCommAicpu::ParseHierarchicalAlgOption).stubs().will(invoke(ParseHierarchicalAlgOptionStub));
    MOCKER_CPP(&HcclCommAicpu::SaveTraceInfo).stubs().will(invoke(SaveTraceInfoStub));
    MOCKER_CPP(&HcclCommAicpu::ExecOp).stubs().will(invoke(ExecOpCaptureUnfoldModeStub));

    HcclResult ret = AicpuHcclProcess::AicpuRunRpcServerV2(hcclCommAicpu, tilingData, commParam);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(g_capturedExecOpUnfoldMode, 1U);

    delete[] buffer;
    delete commParam;
    delete hcclCommAicpu;
}

TEST_F(Test_Hccl_Aicpu_Interface, Ut_AicpuRunRpcServerV2_When_AicpuUnfoldModeDisabled_Expect_PropagatedToOpParam)
{
    HcclCommAicpu* hcclCommAicpu = new HcclCommAicpu();
    ASSERT_NE(hcclCommAicpu, nullptr);
    HcclOpResParam* commParam = CreateHcclOpResParam("test_group");
    ASSERT_NE(commParam, nullptr);
    u8* buffer = CreateAllgatherTilingDataBuffer("test_tag", 0);
    OpTilingData* tilingData = reinterpret_cast<OpTilingData*>(buffer);
    g_capturedExecOpUnfoldMode = 0xFF;

    MOCKER_CPP(&HcclCommAicpu::RecordHostOrder).stubs().will(invoke(RecordHostOrderStub));
    MOCKER_CPP(&HcclCommAicpu::UpdateNotifyWaitTimeOut).stubs().will(invoke(UpdateNotifyWaitTimeOutStub));
    MOCKER_CPP(&HcclCommAicpu::ParseHierarchicalAlgOption).stubs().will(invoke(ParseHierarchicalAlgOptionStub));
    MOCKER_CPP(&HcclCommAicpu::SaveTraceInfo).stubs().will(invoke(SaveTraceInfoStub));
    MOCKER_CPP(&HcclCommAicpu::ExecOp).stubs().will(invoke(ExecOpCaptureUnfoldModeStub));

    HcclResult ret = AicpuHcclProcess::AicpuRunRpcServerV2(hcclCommAicpu, tilingData, commParam);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(g_capturedExecOpUnfoldMode, 0U);

    delete[] buffer;
    delete commParam;
    delete hcclCommAicpu;
}

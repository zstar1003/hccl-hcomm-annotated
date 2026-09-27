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
#include "hccl/hccl_res.h"
#include "../../hccl_api_base_test.h"
#include "hccl_tbe_task.h"
#include "thread_manager.h"
#include "launch_aicpu.h"
#include "aicpu_launch_manager.h"
#include "adapter_rts_common.h"
#include "aicpu_ts_thread.h"

using namespace hccl;

static HcclResult StubThreadKernelLaunchForCommDevice(
    std::vector<std::shared_ptr<hccl::Thread>>& newThreads, const std::string& commId,
    std::unique_ptr<ThreadHandle[]>& aicpuHandle, aclrtBinHandle binHandle)
{
    for (size_t i = 0; i < newThreads.size(); ++i) {
        aicpuHandle[i] = static_cast<ThreadHandle>(0x5000 + i);
    }
    return HCCL_SUCCESS;
}

class ThreadManagerTest : public BaseInit {
public:
    void SetUp() override
    {
        std::cout << "ThreadManagerTest SetUp" << std::endl;
        BaseInit::SetUp();
        MOCKER(AicpuAclKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
        ManagerCallbacks callbacks;
        callbacks.getAicpuCommState = []() {
            return true;
        };
        callbacks.setAicpuCommState = [](bool) {};
        callbacks.kernelLaunchAicpuCommInit = []() {
            return HCCL_SUCCESS;
        };
        callbacks.reportProfilingKernel = [](uint64_t, std::string) {
            return HCCL_SUCCESS;
        };
        threadManager = std::make_unique<ThreadMgr>(1, 1, "test", nullptr, callbacks);
    }
    void TearDown() override
    {
        std::cout << "ThreadManagerTest TearDown" << std::endl;
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }

private:
    std::unique_ptr<ThreadMgr> threadManager;
    uint32_t threadNum = 1;
    ThreadHandle threads[1] = {0};
    ThreadHandle exportedThreads[1] = {0};
};

void MockGetRunSideIsDevice()
{
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(bool{false})).will(returnValue(HCCL_SUCCESS));
}

void MockThreadKernelLaunchForComm()
{
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(returnValue(HCCL_SUCCESS));
}

TEST_F(ThreadManagerTest, Ut_ThreadExportToCommEngineAicpu_When_InvalidThreadHandle_Expect_HCCL_E_PARA)
{
    CommEngine dstCommEngine = COMM_ENGINE_AICPU_TS;

    HcclResult ret = threadManager->HcclThreadExportToCommEngine(threadNum, threads, dstCommEngine, exportedThreads);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(ThreadManagerTest, Ut_ThreadExportToCommEngineAicpu_When_Normal_Expect_ReturnHCCL_SUCCESS)
{
    MockGetRunSideIsDevice();
    MockThreadKernelLaunchForComm();

    CommEngine dstCommEngine = COMM_ENGINE_AICPU_TS;

    HcclResult ret = threadManager->HcclThreadAcquireWithStream(CommEngine::COMM_ENGINE_CPU, nullptr, 1, threads);
    if (ret == HCCL_SUCCESS) {
        ret = threadManager->HcclThreadExportToCommEngine(threadNum, threads, dstCommEngine, exportedThreads);
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }
}

TEST_F(ThreadManagerTest, Ut_ResetThreadLocalNotifies_When_NoThreads_Expect_Success)
{
    EXPECT_EQ(threadManager->ResetThreadLocalNotifies(), HCCL_SUCCESS);
}

TEST_F(ThreadManagerTest, Ut_ResetThreadLocalNotifies_When_HrtNotifyResetFailed_Expect_ReturnFailed)
{
    MockGetRunSideIsDevice();
    HcclResult ret = threadManager->HcclThreadAcquireWithStream(CommEngine::COMM_ENGINE_CPU, nullptr, 1, threads);
    ASSERT_EQ(ret, HCCL_SUCCESS);

    MOCKER(hrtNotifyReset).stubs().will(returnValue(HCCL_E_INTERNAL));
    ret = threadManager->ResetThreadLocalNotifies();
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(ThreadManagerTest, Ut_ResetThreadLocalNotifies_When_OrderLaunchThreadRegistered_Expect_Success)
{
    MockGetRunSideIsDevice();
    HcclResult ret = threadManager->HcclThreadAcquireWithStream(CommEngine::COMM_ENGINE_CPU, nullptr, 1, threads);
    ASSERT_EQ(ret, HCCL_SUCCESS);

    ret = threadManager->RegisterOrderLaunchThread(threads[0]);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(threadManager->ResetThreadLocalNotifies(), HCCL_SUCCESS);
}

/* ======================== HcclDedicatedThreadAcquire DEVICE ======================== */

static void MockAicpuThreadEnv()
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(HCCL_SUCCESS));
}

TEST_F(ThreadManagerTest, Ut_DedicatedThreadAcquire_When_DeviceTypeInvalid_Expect_HCCL_E_PARA)
{
    ThreadHandle thread = 0;
    HcclResult ret = threadManager->HcclDedicatedThreadAcquire(HCCL_DED_THREAD_TYPE_INVALID, 1, &thread);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(ThreadManagerTest, Ut_DedicatedThreadAcquire_When_DeviceThreadNullptr_Expect_HCCL_E_PTR)
{
    HcclResult ret
        = threadManager->HcclDedicatedThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE, 1, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(ThreadManagerTest, Ut_DedicatedThreadAcquire_When_DeviceCreateSuccess_Expect_NonZeroThread)
{
    MockAicpuThreadEnv();
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(invoke(StubThreadKernelLaunchForCommDevice));

    ThreadHandle thread = 0;
    HcclResult ret
        = threadManager->HcclDedicatedThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE, 1, &thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(thread, static_cast<ThreadHandle>(0));
}

TEST_F(ThreadManagerTest, Ut_DedicatedThreadAcquire_When_DeviceRepeatedAcquire_Expect_SameThread)
{
    MockAicpuThreadEnv();
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(invoke(StubThreadKernelLaunchForCommDevice));

    ThreadHandle thread1 = 0;
    HcclResult ret
        = threadManager->HcclDedicatedThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE, 1, &thread1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(thread1, static_cast<ThreadHandle>(0));

    ThreadHandle thread2 = 0;
    ret = threadManager->HcclDedicatedThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE, 1, &thread2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(thread1, thread2);
}

TEST_F(ThreadManagerTest, Ut_DedicatedThreadAcquire_When_DeviceKernelLaunchFail_Expect_Error)
{
    MockAicpuThreadEnv();
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(returnValue(HCCL_E_INTERNAL));

    ThreadHandle thread = 0;
    HcclResult ret
        = threadManager->HcclDedicatedThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE, 1, &thread);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../hccl_api_base_test.h"
#include "hcomm_c_adpt.h"
#include "hcomm_thread_c_adpt.h"
#include "local_notify_impl.h"
#include "aicpu_launch_manager.h"
#include "llt_hccl_stub_rank_graph.h"
#include "launch_aicpu.h"

// stub for AicpuLaunchMgr::ThreadKernelLaunchForComm，设置 aicpuHandle out 参数为非零值
static HcclResult StubThreadKernelLaunchForComm(
    std::vector<std::shared_ptr<hccl::Thread>>& newThreads, const std::string& commId,
    std::unique_ptr<ThreadHandle[]>& aicpuHandle, aclrtBinHandle binHandle)
{
    for (size_t i = 0; i < newThreads.size(); ++i) {
        aicpuHandle[i] = static_cast<ThreadHandle>(0x1 + i);
    }
    return HCCL_SUCCESS;
}

// Spy 线程：通过构造/析构计数观测 host 侧线程对象的生命周期，
// 用于验证 FreeThreads 先销毁 device 侧线程、再释放 host 侧 SQ/CQ 资源的顺序
class SpyThreadForFree : public hccl::Thread {
public:
    SpyThreadForFree() { ++aliveCount; }
    ~SpyThreadForFree() override { --aliveCount; }
    static int aliveCount;

    HcclResult Init() override { return HCCL_SUCCESS; }
    HcclResult DeInit() override { return HCCL_SUCCESS; }
    std::string& GetUniqueId() override { return uniqueId_; }
    uint32_t GetNotifyNum() const override { return 0; }
    LocalNotify* GetNotify(uint32_t index) const override { return nullptr; }
    HcclResult SupplementNotify(uint32_t notifyNum) override { return HCCL_SUCCESS; }
    bool IsDeviceA5() const override { return false; }
    Stream* GetStream() const override { return nullptr; }
    void* GetStreamLitePtr() const override { return nullptr; }
    void LaunchTask() const override {}
    void TryLaunchTask() const override {}
    HcclResult LocalNotifyRecord(uint32_t notifyId) const override { return HCCL_SUCCESS; }
    HcclResult LocalNotifyWait(uint32_t notifyId) const override { return HCCL_SUCCESS; }
    HcclResult LocalNotifyRecord(ThreadHandle dstThread, uint32_t dstNotifyIdx) const override { return HCCL_SUCCESS; }
    HcclResult LocalNotifyWait(uint32_t notifyIdx, uint32_t timeOut) const override { return HCCL_SUCCESS; }
    HcclResult LocalCopy(void* dst, const void* src, uint64_t sizeByte) const override { return HCCL_SUCCESS; }
    HcclResult LocalReduce(
        void* dst, const void* src, uint64_t sizeByte, HcommDataType dataType, HcommReduceOp reduceOp) const override
    {
        return HCCL_SUCCESS;
    }
    bool GetMaster() const override { return false; }
    void SetIsMaster(bool isMaster) override {}

private:
    std::string uniqueId_{"spy_thread"};
};
int SpyThreadForFree::aliveCount = 0;

static bool g_destroyCalled = false;
static int g_aliveAtDestroy = -1;
static ThreadHandle g_destroyDeviceHandle = 0;
static uint32_t g_destroyListNum = 0;

static HcclResult StubThreadKernelLaunchDestroy(ThreadHandle* threadHandles, uint32_t listNum, aclrtBinHandle binHandle)
{
    (void)binHandle;
    g_destroyCalled = true;
    g_aliveAtDestroy = SpyThreadForFree::aliveCount;
    g_destroyListNum = listNum;
    g_destroyDeviceHandle = (threadHandles != nullptr && listNum > 0) ? threadHandles[0] : 0;
    return HCCL_SUCCESS;
}

class TestHcclThread : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        unsetenv("HCCL_DFS_CONFIG");
    }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

TEST_F(TestHcclThread, Ut_NotifyLoadType_When_AicpuEngine_Expect_HostNotify)
{
    NotifyLoadType notifyLoadType = NotifyLoadType::DEVICE_NOTIFY;
    EXPECT_EQ(CommEngineToNotifyLoadType(COMM_ENGINE_AICPU_TS, notifyLoadType), HCCL_SUCCESS);
    EXPECT_EQ(notifyLoadType, NotifyLoadType::HOST_NOTIFY);

    notifyLoadType = NotifyLoadType::DEVICE_NOTIFY;
    EXPECT_EQ(GetNotifyLoadType(COMM_ENGINE_AICPU, THREAD_TYPE_TS, notifyLoadType), HCCL_SUCCESS);
    EXPECT_EQ(notifyLoadType, NotifyLoadType::HOST_NOTIFY);
}

TEST_F(TestHcclThread, Ut_TestHcclThread_When_CreateHostCpuTsCommEngineThread_Return_HCCL_Success)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    std::shared_ptr<Thread> cpuHandle;
    HcclResult ret
        = CreateThread(COMM_ENGINE_CPU_TS, StreamType::STREAM_TYPE_ONLINE, 3, NotifyLoadType::HOST_NOTIFY, cpuHandle);
    EXPECT_EQ(ret, 0);
    ret = cpuHandle->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    uint64_t cpu = reinterpret_cast<ThreadHandle>(cpuHandle.get());
    Stream* stream = GetStream(cpu);
    EXPECT_NE(stream, nullptr);
    void* notify = GetNotify(cpu, 0);
    EXPECT_NE(nullptr, notify);
}

TEST_F(TestHcclThread, Ut_TestHcclThread_When_CreateAicpuTsCommEngineThread_Return_HCCL_Success)
{
    std::shared_ptr<Thread> aicpuHandle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    HcclResult ret = CreateThread(
        COMM_ENGINE_AICPU_TS, StreamType::STREAM_TYPE_DEVICE, 2, NotifyLoadType::HOST_NOTIFY, aicpuHandle);
    EXPECT_EQ(ret, 0);
    ret = aicpuHandle->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    uint64_t aicpu = reinterpret_cast<ThreadHandle>(aicpuHandle.get());
    Stream* aicpuStream = GetStream(aicpu);
    EXPECT_NE(aicpuStream, nullptr);
    void* aicpuNotify = GetNotify(aicpu, 0);
    EXPECT_NE(nullptr, aicpuNotify);
}

TEST_F(TestHcclThread, Ut_TestHcclThread_When_CreateNotSurportCommEngineThread_Return_HCCL_E_NOT_SUPPORT)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    HcclResult ret
        = CreateThread(COMM_ENGINE_RESERVED, StreamType::STREAM_TYPE_DEVICE, 2, NotifyLoadType::DEVICE_NOTIFY, Handle);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(TestHcclThread, UT_When_DeviceSide_ResourceAllocateFail_expect_return_HcclEInternal)
{
    bool isDeviceSide{true};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread[3];
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_AICPU_TS, 2, 3, thread);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(TestHcclThread, Ut_TestHcommThreadAlloc_When_ThreadIsNullptr_Allocate_expect_Return_HCCL_E_PTR)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    uint64_t* thread = nullptr;
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_AICPU_TS, 2, 3, thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_TestHcommThreadAlloc_When_WithUnsupportedEngine_expect_return_HCCL_E_PARA)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread[3];
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_AIV, 2, 3, thread);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcclThread, Ut_TestHcommThreadAlloc_When_Thread_Allocate_0Num_expect_Return_HCCL_Success)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread[3];
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_AIV, 0, 3, thread);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcclThread, Ut_TestHcommThreadAlloc_When_WithNotifyInitFail_expect_return_HcclERuntime)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtNotifyGetOffset).stubs().will(returnValue(HCCL_E_RUNTIME));
    ThreadHandle thread[3];
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_AICPU_TS, 2, 3, thread);
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
}

TEST_F(TestHcclThread, Ut_TestHcommThreadAlloc_When_CpuTsThread_Allocate_expect_Return_HCCL_Success)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread[3];
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_CPU_TS, 2, 3, thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    Thread* threadptr0 = reinterpret_cast<Thread*>(thread[0]);
    Thread* threadptr1 = reinterpret_cast<Thread*>(thread[1]);

    EXPECT_EQ(threadptr0->GetNotifyNum(), 3);
    EXPECT_EQ(threadptr1->GetNotifyNum(), 3);
    ret = HcommThreadFree(thread, 2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestHcclThread, Ut_TestHcommThreadAlloc_When_Allocate_MAXThreadNum_expect_Return_HCCL_E_PARA)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread;
    HcommResult ret
        = HcommThreadAlloc(COMM_ENGINE_CPU_TS, hccl::HCOMM_THREADNUM_MAX_NUM + 1, static_cast<uint32_t>(0), &thread);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcclThread, Ut_TestHcommThreadAlloc_When_Allocate_MaxNotifyNum_expect_Return_HCCL_E_PARA)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread;
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_CPU_TS, 1, hccl::HCOMM_NOTIFY_MAX_NUM + 1, &thread);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcclThread, Ut_HcommThreadFree_When_expect_Return_HCCL_Success)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread[2];
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_CPU_TS, 2, 3, thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    Thread* threadptr0 = reinterpret_cast<Thread*>(thread[0]);
    Thread* threadptr1 = reinterpret_cast<Thread*>(thread[1]);

    EXPECT_EQ(threadptr0->GetNotifyNum(), 3);
    EXPECT_EQ(threadptr1->GetNotifyNum(), 3);
    ret = HcommThreadFree(thread, 2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestHcclThread, Ut_HcommThreadFree_When_ThreadNum_Is_0_expect_Return_HCCL_E_PARA)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread[2];
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_CPU_TS, 2, 3, thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    Thread* threadptr0 = reinterpret_cast<Thread*>(thread[0]);
    Thread* threadptr1 = reinterpret_cast<Thread*>(thread[1]);

    EXPECT_EQ(threadptr0->GetNotifyNum(), 3);
    EXPECT_EQ(threadptr1->GetNotifyNum(), 3);
    ret = HcommThreadFree(thread, 0);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcclThread, Ut_HcommThreadFree_When_ThreadNullptr_expect_Return_HCCL_E_PTR)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle* thread = nullptr;
    HcommResult ret = HcommThreadFree(thread, 0);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_FreeThreads_When_DeviceThread_Expect_HostReleasedAfterDeviceDestroy)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

    g_destroyCalled = false;
    g_aliveAtDestroy = -1;
    g_destroyDeviceHandle = 0;
    g_destroyListNum = 0;
    SpyThreadForFree::aliveCount = 0;

    auto spyThread = std::make_shared<SpyThreadForFree>();
    EXPECT_EQ(SpyThreadForFree::aliveCount, 1);
    ThreadHandle hostHandle = reinterpret_cast<ThreadHandle>(spyThread.get());
    ThreadHandle deviceHandle = hostHandle + 0x1000;

    ASSERT_EQ(SaveThreads(std::vector<std::shared_ptr<Thread>>{spyThread}), HCCL_SUCCESS);
    ASSERT_EQ(FillThreadD2HMap(&deviceHandle, &hostHandle, 1), HCCL_SUCCESS);
    spyThread.reset(); // 释放测试栈引用，仅剩全局映射表持有 host 线程
    EXPECT_EQ(SpyThreadForFree::aliveCount, 1);

    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchDestroy).stubs().will(invoke(StubThreadKernelLaunchDestroy));

    HcclResult ret = FreeThreads(&deviceHandle, 1, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_TRUE(g_destroyCalled);
    EXPECT_EQ(g_destroyListNum, 1U);
    EXPECT_EQ(g_destroyDeviceHandle, deviceHandle);
    EXPECT_EQ(g_aliveAtDestroy, 1);             // device 侧销毁时 host 线程仍存活
    EXPECT_EQ(SpyThreadForFree::aliveCount, 0); // FreeThreads 返回后 host 线程已释放
}

TEST_F(TestHcclThread, UT_TestHcommThreadAllocWithStream_When_Allocate_WithStream_expect_return_HcclSuccess)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    Stream* stream = nullptr;
    stream = new (std::nothrow) Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    void* rtStream = stream->ptr();
    ThreadHandle thread;
    HcommResult ret = HcommThreadAllocWithStream(COMM_ENGINE_CPU_TS, rtStream, 3, &thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcommResult freeRet = HcommThreadFreeWithStream(&thread, 1);
    EXPECT_EQ(freeRet, HCCL_SUCCESS);
    delete stream;
}

TEST_F(TestHcclThread, UT_TestHcommThreadFreeWithStream_When_ThreadNumZero_expect_return_HCCL_E_PARA)
{
    ThreadHandle thread = 0;
    HcommResult ret = HcommThreadFreeWithStream(&thread, 0);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcclThread, UT_TestHcommThreadFreeWithStream_When_Nullptr_expect_return_HCCL_E_PTR)
{
    HcommResult ret = HcommThreadFreeWithStream(nullptr, 1);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, UT_TestHcommThreadFreeWithStream_When_HandleNotFound_expect_return_HCCL_Success)
{
    ThreadHandle thread = 0x9999;
    HcommResult ret = HcommThreadFreeWithStream(&thread, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestHcclThread, UT_TestHcommThreadAllocWithStream_When_ThreadNullptr_expect_return_HCCL_E_PTR)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    Stream* stream = nullptr;
    stream = new (std::nothrow) Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    void* rtStream = stream->ptr();
    ThreadHandle* thread{nullptr};
    HcommResult ret = HcommThreadAllocWithStream(COMM_ENGINE_CPU_TS, rtStream, 3, thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, UT_TestHcommThreadAllocWithStream_When_WithInvalidEngine_expect_return_HCCL_E_PARA)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    Stream* stream = nullptr;
    stream = new (std::nothrow) Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    void* rtStream = stream->ptr();
    ThreadHandle thread;
    HcommResult ret = HcommThreadAllocWithStream(COMM_ENGINE_AICPU_TS, rtStream, 3, &thread);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcclThread, UT_TestHcommThreadAllocWithStream_When_NotifyInitFailed_expect_return_HCCL_E_RUNTIME)
{
    std::shared_ptr<Thread> Handle;
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtNotifyGetOffset).stubs().will(returnValue(HCCL_E_RUNTIME));
    Stream* stream = nullptr;
    stream = new (std::nothrow) Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    void* rtStream = stream->ptr();
    ThreadHandle thread;
    HcommResult ret = HcommThreadAllocWithStream(COMM_ENGINE_CPU_TS, rtStream, 3, &thread);
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
}

/* ======================== HcommThreadResGetInfo ======================== */

TEST_F(TestHcclThread, Ut_HcommThreadResGetInfo_When_Normal_Expect_Return_HCCL_Success)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread;
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_CPU_TS, 1, 3, &thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* info = nullptr;
    ret = HcommThreadResGetInfo(thread, ThreadResType::THREAD_RES_TYPE_STREAM, sizeof(ThreadResTypeStream), &info);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(info, nullptr);

    HcommThreadFree(&thread, 1);
}

TEST_F(TestHcclThread, Ut_HcommThreadResGetInfo_When_InfoNull_Expect_Return_HCCL_E_PTR)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread;
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_CPU_TS, 1, 3, &thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcommThreadResGetInfo(thread, ThreadResType::THREAD_RES_TYPE_STREAM, sizeof(ThreadResTypeStream), nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);

    HcommThreadFree(&thread, 1);
}

TEST_F(TestHcclThread, Ut_HcommThreadResGetInfo_When_ThreadZero_Expect_Return_HCCL_E_PTR)
{
    void* info = nullptr;
    HcommResult ret
        = HcommThreadResGetInfo(0, ThreadResType::THREAD_RES_TYPE_STREAM, sizeof(ThreadResTypeStream), &info);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcommThreadResGetInfo_When_ResTypeNotSupport_Expect_Return_HCCL_E_PARA)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread;
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_CPU_TS, 1, 3, &thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* info = nullptr;
    ret = HcommThreadResGetInfo(thread, ThreadResType::THREAD_RES_TYPE_INVALID, sizeof(ThreadResTypeStream), &info);
    EXPECT_EQ(ret, HCCL_E_PARA);

    HcommThreadFree(&thread, 1);
}

TEST_F(TestHcclThread, Ut_HcommThreadResGetInfo_When_InfoLenMismatch_Expect_Return_HCCL_E_PARA)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread;
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_CPU_TS, 1, 3, &thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* info = nullptr;
    ret = HcommThreadResGetInfo(thread, ThreadResType::THREAD_RES_TYPE_STREAM, 1, &info);
    EXPECT_EQ(ret, HCCL_E_PARA);

    HcommThreadFree(&thread, 1);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquire_When_Acquire_CpuTsThread_Return_HCCL_Success)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_CPU_TS, 1, 2, &thread);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquire_When_Acquire_AicpuTsThread_Return_HCCL_Success)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hcclComm::GetAicpuCommState).stubs().will(returnValue(true));
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(returnValue(0));
    MOCKER_CPP(&HcclCommProfiling::ReportKernel).stubs().will(returnValue(0));

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 1, 2, &thread);
    EXPECT_EQ(ret, 0);
}
TEST_F(TestHcclThread, Ut_HcclThreadAcquire_When_Acquire_AicpuTsThread_Reuse_Return_HCCL_Success)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hcclComm::GetAicpuCommState).stubs().will(returnValue(true));
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(returnValue(0));
    MOCKER_CPP(&AicpuLaunchMgr::SupplementNotifyKernelLaunch).expects(exactly(2)).will(returnValue(0));
    MOCKER_CPP(&HcclCommProfiling::ReportKernel).stubs().will(returnValue(0));

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config{};
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 2, 3, &thread);
    EXPECT_EQ(ret, 0);

    ret = HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 3, 3, &thread);
    EXPECT_EQ(ret, 0);

    ret = HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 3, 4, &thread);
    EXPECT_EQ(ret, 0);

    ret = HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 4, 5, &thread);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquire_When_CommNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hcclComm::GetAicpuCommState).stubs().will(returnValue(true));
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(returnValue(0));

    void* comm = nullptr;
    ThreadHandle thread;
    HcclResult ret = HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 1, 2, &thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquire_When_ThreadNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hcclComm::GetAicpuCommState).stubs().will(returnValue(true));
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(returnValue(0));

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle* thread{nullptr};
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 1, 2, thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquire_When_CollCommNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hcclComm::GetAicpuCommState).stubs().will(returnValue(true));
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(returnValue(0));
    MOCKER_CPP(&hcclComm::IsCommunicatorV2).stubs().will(returnValue(true));

    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);

    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    HcclResult ret = HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 1, 2, &thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquire_When_engineResMgrNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hcclComm::GetAicpuCommState).stubs().will(returnValue(true));
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(returnValue(0));
    MOCKER_CPP(&CollComm::Init).stubs().will(returnValue(0));
    MOCKER_CPP(&CollComm::GetHDCommunicate).stubs().will(returnValue(0));

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 1, 2, &thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclGetNotifyNumInThread_When_Normal_Return_HCCL_Success)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread[2];
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_CPU_TS, 2, 3, thread);
    EXPECT_EQ(ret, 0);

    Thread* threadptr0 = reinterpret_cast<Thread*>(thread[0]);

    EXPECT_EQ(threadptr0->GetNotifyNum(), 3);

    uint32_t notifyNum;
    ret = HcclGetNotifyNumInThread(comm, thread[0], COMM_ENGINE_CPU_TS, &notifyNum);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(notifyNum, 3);
}

TEST_F(TestHcclThread, Ut_HcclGetNotifyNumInThread_When_Normal_Reuse_Return_HCCL_Success)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuLaunchMgr::SupplementNotifyKernelLaunch).stubs().will(returnValue(0));

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config{};
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1; // 非CCU模式，避免拉起CCU平台层
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread[2];
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_CPU_TS, 2, 3, thread);
    EXPECT_EQ(ret, 0);
    Thread* threadptr0 = reinterpret_cast<Thread*>(thread[0]);
    EXPECT_EQ(threadptr0->GetNotifyNum(), 3);

    ret = HcclThreadAcquire(comm, COMM_ENGINE_CPU_TS, 3, 3, thread);
    EXPECT_EQ(ret, 0);
    threadptr0 = reinterpret_cast<Thread*>(thread[1]);
    EXPECT_EQ(threadptr0->GetNotifyNum(), 3);

    ret = HcclThreadAcquire(comm, COMM_ENGINE_CPU_TS, 3, 4, thread);
    EXPECT_EQ(ret, 0);
    threadptr0 = reinterpret_cast<Thread*>(thread[2]);
    EXPECT_EQ(threadptr0->GetNotifyNum(), 4);

    ret = HcclThreadAcquire(comm, COMM_ENGINE_CPU_TS, 4, 5, thread);
    EXPECT_EQ(ret, 0);
    threadptr0 = reinterpret_cast<Thread*>(thread[3]);
    EXPECT_EQ(threadptr0->GetNotifyNum(), 5);

    uint32_t notifyNum;
    ret = HcclGetNotifyNumInThread(comm, thread[0], COMM_ENGINE_CPU_TS, &notifyNum);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(notifyNum, 5);
}

TEST_F(TestHcclThread, Ut_HcclGetNotifyNumInThread_When_CommNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread[2];
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_CPU_TS, 2, 3, thread);
    EXPECT_EQ(ret, 0);

    Thread* threadptr0 = reinterpret_cast<Thread*>(thread[0]);

    EXPECT_EQ(threadptr0->GetNotifyNum(), 3);

    uint32_t notifyNum;
    comm = nullptr;
    ret = HcclGetNotifyNumInThread(comm, thread[0], COMM_ENGINE_CPU_TS, &notifyNum);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclGetNotifyNumInThread_When_notifyNumNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread[2];
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_CPU_TS, 2, 3, thread);
    EXPECT_EQ(ret, 0);

    Thread* threadptr0 = reinterpret_cast<Thread*>(thread[0]);

    EXPECT_EQ(threadptr0->GetNotifyNum(), 3);

    uint32_t* notifyNum{nullptr};
    ret = HcclGetNotifyNumInThread(comm, thread[0], COMM_ENGINE_CPU_TS, notifyNum);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclGetNotifyNumInThread_When_CollCommNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);

    ThreadHandle thread[2];
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());

    uint32_t notifyNum;
    HcclResult ret = HcclGetNotifyNumInThread(comm, thread[0], COMM_ENGINE_CPU_TS, &notifyNum);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclGetNotifyNumInThread_When_engineResMgrNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollComm::Init).stubs().will(returnValue(0));
    MOCKER_CPP(&CollComm::GetHDCommunicate).stubs().will(returnValue(0));
    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread[2];
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());

    uint32_t notifyNum;
    ret = HcclGetNotifyNumInThread(comm, thread[0], COMM_ENGINE_CPU_TS, &notifyNum);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquireWithStream_When_Acquire_CpuTsThread_Return_HCCL_Success)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    Stream* stream = nullptr;
    stream = new (std::nothrow) Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    void* rtStream = stream->ptr();

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, rtStream, 2, &thread);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquireWithStream_When_CommNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    Stream* stream = nullptr;
    stream = new (std::nothrow) Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    void* rtStream = stream->ptr();
    ThreadHandle thread;
    void* comm = nullptr;
    HcclResult ret = HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, rtStream, 2, &thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquireWithStream_When_threadNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    Stream* stream = nullptr;
    stream = new (std::nothrow) Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    void* rtStream = stream->ptr();
    ThreadHandle* thread{nullptr};
    void* comm = (void*)0x1234;
    HcclResult ret = HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, rtStream, 2, thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquireWithStream_When_rtStreamNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    void* rtStream = nullptr;
    ThreadHandle thread;
    void* comm = (void*)0x1234;
    HcclResult ret = HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, rtStream, 2, &thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquireWithStream_When_CollcommNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    Stream* stream = nullptr;
    stream = new (std::nothrow) Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    void* rtStream = stream->ptr();

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);

    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    HcclResult ret = HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, rtStream, 2, &thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquireWithStream_When_engineResMgrNullptr_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollComm::Init).stubs().will(returnValue(0));
    MOCKER_CPP(&CollComm::GetHDCommunicate).stubs().will(returnValue(0));
    Stream* stream = nullptr;
    stream = new (std::nothrow) Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    void* rtStream = stream->ptr();

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, rtStream, 2, &thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquireWithStream_When_A3_Acquire_CpuTsThread_Return_HCCL_Success)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommEngineResMgr::HcclThreadAcquireWithStream).stubs().will(returnValue(HCCL_SUCCESS));

    Stream* stream = nullptr;
    stream = new (std::nothrow) Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    void* rtStream = stream->ptr();

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    HcclResult ret = HcclThreadAcquireWithStream(comm, COMM_ENGINE_CPU_TS, rtStream, 2, &thread);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquire_When_Acquire_MAX_Thread_AicpuTsThread_Return_HCCL_E_UNAVAIL)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommEngineResMgr::HcclThreadAcquireV2).stubs().will(returnValue(HCCL_E_UNAVAIL));

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;
    config.hcclRdmaTrafficClass = 0xFFFFFFFF;
    config.hcclRdmaServiceLevel = 0xFFFFFFFF;
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 201, 0, &thread);
    EXPECT_EQ(ret, HCCL_E_UNAVAIL);
}

TEST_F(TestHcclThread, Ut_HcclThreadAcquire_When_Acquire_MAX_Notify_AicpuTsThread_Return_HCCL_E_UNAVAIL)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommEngineResMgr::HcclThreadAcquireV2).stubs().will(returnValue(HCCL_E_UNAVAIL));

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;
    config.hcclRdmaTrafficClass = 0xFFFFFFFF;
    config.hcclRdmaServiceLevel = 0xFFFFFFFF;
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_AICPU_TS, 1, 65535, &thread);
    EXPECT_EQ(ret, HCCL_E_UNAVAIL);
}

TEST_F(TestHcclThread, Ut_HcclThreadResGetInfo_When_Normal_Expect_Return_HCCL_Success)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_CPU_TS, 1, 2, &thread);
    EXPECT_EQ(ret, 0);

    ThreadResTypeStream stream;
    ret = HcclThreadResGetInfo(
        comm, thread, ThreadResType::THREAD_RES_TYPE_STREAM, sizeof(ThreadResTypeStream), &stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestHcclThread, Ut_HcclThreadResGetInfo_When_commNull_Return_HCCL_E_PTR)
{
    HcclComm comm = nullptr;
    ThreadHandle thread;
    ThreadResTypeStream stream;
    HcclResult ret = HcclThreadResGetInfo(
        comm, thread, ThreadResType::THREAD_RES_TYPE_STREAM, sizeof(ThreadResTypeStream), &stream);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclThreadResGetInfo_When_infoNull_Return_HCCL_E_PTR)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ThreadResTypeStream stream;

    ret = HcclThreadResGetInfo(
        comm, thread, ThreadResType::THREAD_RES_TYPE_STREAM, sizeof(ThreadResTypeStream), nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclThreadResGetInfo_When_threadNotFound_Return_HCCL_E_NOT_FOUND)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ThreadResTypeStream stream;

    ret = HcclThreadResGetInfo(
        comm, thread, ThreadResType::THREAD_RES_TYPE_STREAM, sizeof(ThreadResTypeStream), &stream);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(TestHcclThread, Ut_HcclThreadResGetInfo_When_restypeNotSupport_Return_HCCL_E_NOT_SUPPORT)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_CPU_TS, 1, 2, &thread);
    EXPECT_EQ(ret, 0);

    ThreadResTypeStream stream;
    ret = HcclThreadResGetInfo(
        comm, thread, ThreadResType::THREAD_RES_TYPE_INVALID, sizeof(ThreadResTypeStream), &stream);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(TestHcclThread, Ut_HcclThreadResGetInfo_When_infoLenMismatch_Return_HCCL_E_PARA)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
    ThreadHandle thread;
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());
    ret = HcclThreadAcquire(comm, COMM_ENGINE_CPU_TS, 1, 2, &thread);
    EXPECT_EQ(ret, 0);
    ThreadResTypeStream stream;

    ret = HcclThreadResGetInfo(comm, thread, ThreadResType::THREAD_RES_TYPE_STREAM, 1, &stream);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcclThread, Ut_InitCollComm_When_ApplyRdmaTrafficAndServiceLevel_Return_HCCL_Success)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;    // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 120; // 配置RDMA Traffic Class为120，验证正常配置RDMA Traffic Class的场景
    config.hcclRdmaServiceLevel = 3;   // 配置RDMA Service Level为3，验证正常配置RDMA Service Level的场景
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 0);
}

TEST_F(TestHcclThread, Ut_InitCollComm_When_RdmaServiceLevel_Abnormal_Return_HCCL_E_PARA)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 8;          // 配置RDMA Service Level超过正常值
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 1);
}

TEST_F(TestHcclThread, Ut_InitCollComm_When_RdmaTrafficClass_Abnormal_Return_HCCL_E_PARA)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    ;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 300;        // 配置RDMA Traffic Class超过约束范围
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, 1);
}

TEST_F(TestHcclThread, Ut_ThreadKernelLaunchDestroy)
{
    MOCKER(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(AicpuAclKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hcclStreamSynchronize).stubs().will(returnValue(HCCL_SUCCESS));

    constexpr u32 threadHandleNum = 1;
    vector<ThreadHandle> deviceHandle(threadHandleNum);
    void* binHandle = nullptr;
    HcclResult ret = AicpuLaunchMgr::ThreadKernelLaunchDestroy(deviceHandle.data(), deviceHandle.size(), binHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// TEST_F(TestHcclThread, Ut_LaunchNotifyKernel)
// {
//     MOCKER(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));
//     MOCKER(AicpuAclKernelLaunch).stubs().will(returnValue(HCCL_SUCCESS));
//     MOCKER(hcclStreamSynchronize).stubs().will(returnValue(HCCL_SUCCESS));

//     NotifyMgrAicpuParam opParam;
//     void* binCustomHandle = nullptr;
//     HcclResult ret = AicpuLaunchMgr::LaunchNotifyKernel(opParam, binCustomHandle);
//     EXPECT_EQ(ret, HCCL_SUCCESS);
// }

TEST_F(TestHcclThread, Ut_HcclDedicatedThreadAcquire_When_CommNullptr_Return_HCCL_E_PTR)
{
    void* comm = nullptr;
    ThreadHandle thread;
    HcclResult ret = HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_LAUNCH, 2, &thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclDedicatedThreadAcquire_When_ThreadNullptr_Return_HCCL_E_PTR)
{
    void* comm = (void*)0x1234;
    ThreadHandle* thread = nullptr;
    HcclResult ret = HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_LAUNCH, 2, thread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclThread, Ut_HcclDedicatedThreadAcquire_When_UseTypeInvalid_Return_HCCL_E_PARA)
{
    void* comm = (void*)0x1234;
    ThreadHandle thread;
    HcclResult ret = HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_INVALID, 2, &thread);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcclThread, Ut_HcclDedicatedThreadAcquire_When_SingleComm_RepeatedAcquire_Return_SameThreadAndNotifyPlusTwo)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());

    ThreadHandle thread1;
    ret = HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_LAUNCH, 3, &thread1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(thread1, 0ULL);
    Thread* threadPtr1 = reinterpret_cast<Thread*>(thread1);
    EXPECT_EQ(threadPtr1->GetNotifyNum(), 3U);

    ThreadHandle thread2;
    ret = HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_LAUNCH, 5, &thread2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(thread2, thread1);
    Thread* threadPtr2 = reinterpret_cast<Thread*>(thread2);
    EXPECT_EQ(threadPtr2->GetNotifyNum(), 5U);
}

TEST_F(TestHcclThread, Ut_HcclDedicatedThreadAcquire_When_MultiComm_Acquire_Return_DifferentThread)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;

    char commName1[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr1 = make_shared<hccl::hcclComm>(1, 1, commName1);
    HcclCommConfig config1;
    UtInitHcclCommConfig(config1);
    config1.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config1.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config1.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr1->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName1, &config1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm1 = static_cast<HcclComm>(hcclCommPtr1.get());

    char commName2[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr2 = make_shared<hccl::hcclComm>(1, 1, commName2);
    HcclCommConfig config2;
    UtInitHcclCommConfig(config2);
    config2.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config2.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config2.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    ret = hcclCommPtr2->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName2, &config2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm2 = static_cast<HcclComm>(hcclCommPtr2.get());

    ThreadHandle thread1;
    ret = HcclDedicatedThreadAcquire(comm1, HCCL_DED_THREAD_TYPE_AICPU_LAUNCH, 3, &thread1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(thread1, 0ULL);

    ThreadHandle thread2;
    ret = HcclDedicatedThreadAcquire(comm2, HCCL_DED_THREAD_TYPE_AICPU_LAUNCH, 3, &thread2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(thread2, 0ULL);
    EXPECT_NE(thread1, thread2);
}

TEST_F(TestHcclThread, Ut_HcclDedicatedThreadAcquire_When_UseTypeAicpuUnfoldGe_NotInMap_Return_HCCL_SUCCESS)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());

    ThreadHandle thread;
    ret = HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE, 3, &thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestHcclThread, Ut_HcclDedicatedThreadAcquire_When_UseTypeAicpuUnfoldGe_InMap_Return_SameThread)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());

    // 先通过 AICPU_UNFOLD 创建专用线程，再将句柄复制到 AICPU_UNFOLD_GE 槽位，模拟已存在场景
    ThreadHandle unfoldThread;
    ret = HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_LAUNCH, 3, &unfoldThread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(unfoldThread, 0ULL);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    auto* threadMgr = hcclComm->GetCollComm()->GetCommEngineResMgr()->threadMgr_.get();
    threadMgr->dedicatedThreadMap_[HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE] = unfoldThread;

    ThreadHandle thread;
    ret = HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE, 3, &thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(thread, unfoldThread);
}

TEST_F(
    TestHcclThread,
    Ut_HcclDedicatedThreadAcquire_When_UseTypeAicpuUnfoldGe_RepeatedAcquire_Return_SameThreadAndNotifyPlus)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    std::shared_ptr<hccl::hcclComm> hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());

    // 先通过 AICPU_UNFOLD 创建专用线程（notifyNum=2），再将句柄复制到 AICPU_UNFOLD_GE 槽位
    ThreadHandle unfoldThread;
    ret = HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_LAUNCH, 2, &unfoldThread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(unfoldThread, 0ULL);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    auto* threadMgr = hcclComm->GetCollComm()->GetCommEngineResMgr()->threadMgr_.get();
    threadMgr->dedicatedThreadMap_[HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE] = unfoldThread;

    // 第一次获取 GE 线程，notifyNum=3，需补充1个notify
    ThreadHandle thread1;
    ret = HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE, 3, &thread1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(thread1, unfoldThread);
    Thread* threadPtr1 = reinterpret_cast<Thread*>(thread1);
    EXPECT_EQ(threadPtr1->GetNotifyNum(), 3U);

    // 第二次获取 GE 线程，notifyNum=5，需补充2个notify
    ThreadHandle thread2;
    ret = HcclDedicatedThreadAcquire(comm, HCCL_DED_THREAD_TYPE_AICPU_LAUNCH_GE, 5, &thread2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(thread2, thread1);
    Thread* threadPtr2 = reinterpret_cast<Thread*>(thread2);
    EXPECT_EQ(threadPtr2->GetNotifyNum(), 5U);
}

// HcommThreadAllocWithStream + HcommThreadFreeWithStream
TEST_F(TestHcclThread, Ut_HcommThreadAllocWithStream_When_AllocThenFree_Expect_Success)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    Stream* stream = new (std::nothrow) Stream(hccl::StreamType::STREAM_TYPE_ONLINE);
    void* rtStream = stream->ptr();
    ThreadHandle thread;
    HcommResult ret = HcommThreadAllocWithStream(COMM_ENGINE_CPU_TS, rtStream, 3, &thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 进程级 g_ThreadMap 已记，FreeWithStream 后应可 erase
    ret = HcommThreadFreeWithStream(&thread, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    delete stream;
}

// HcommThreadSupplementNotify：单线程补充 notify
TEST_F(TestHcclThread, Ut_HcommThreadSupplementNotify_When_Single_Expect_Success)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread;
    uint32_t notifyNum = 2;
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_CPU_TS, 1, &notifyNum, &thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ThreadHandle handle = thread;
    uint32_t addNotifyNum = 3; // 补 3（增量）
    ret = HcommThreadSupplementNotify(CommEngine::COMM_ENGINE_AICPU_TS, &handle, 1, &addNotifyNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    Thread* t = reinterpret_cast<Thread*>(handle);
    EXPECT_EQ(t->GetNotifyNum(), 5U);
    ret = HcommThreadFree(&thread, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// HcommThreadSupplementNotify：空指针
TEST_F(TestHcclThread, Ut_HcommThreadSupplementNotify_When_NullHandle_Expect_E_PTR)
{
    uint32_t addNotifyNum = 3;
    HcommResult ret = HcommThreadSupplementNotify(CommEngine::COMM_ENGINE_AICPU_TS, nullptr, 1, &addNotifyNum);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// HcommThreadGetNotifyNum：查询
TEST_F(TestHcclThread, Ut_HcommThreadGetNotifyNum_When_Normal_Expect_Success)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    ThreadHandle thread;
    uint32_t notifyNum = 3;
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_CPU_TS, 1, &notifyNum, &thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    uint32_t out = 0;
    ret = HcommThreadGetNotifyNum(thread, &out);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(out, 3U);
    ret = HcommThreadFree(&thread, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// HcommThreadExportToCommEngine：CPU->AICPU 导出
TEST_F(TestHcclThread, Ut_HcommThreadExportToCommEngine_When_CpuToAicpu_Expect_Success)
{
    bool isDeviceSide{false};
    MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuLaunchMgr::ThreadKernelLaunchForComm).stubs().will(invoke(StubThreadKernelLaunchForComm));
    ThreadHandle hostThread;
    uint32_t notifyNum = 2;
    HcommResult ret = HcommThreadAlloc(COMM_ENGINE_CPU_TS, 1, &notifyNum, &hostThread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ThreadHandle devThread = 0;
    std::string commId = "";
    ret = HcommThreadExportToCommEngine(&hostThread, commId.c_str(), 1, COMM_ENGINE_AICPU_TS, &devThread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(devThread, 0ULL);
    ret = HcommThreadFree(&hostThread, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// HcommThreadExportToCommEngine：空指针
TEST_F(TestHcclThread, Ut_HcommThreadExportToCommEngine_When_NullHandle_Expect_E_PTR)
{
    ThreadHandle devThread = 0;
    std::string commId = "";
    HcommResult ret = HcommThreadExportToCommEngine(nullptr, commId.c_str(), 1, COMM_ENGINE_AICPU_TS, &devThread);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

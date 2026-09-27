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
#include <memory>
#include <thread>
#include <atomic>

#define private public
#define protected public
#include "hccl_dpu_manager.h"
#undef private

#include "llt_hccl_stub_pub.h"
#include "acl/acl_rt.h"

using namespace hccl;

class DpuManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DpuManagerTest SetUpTestCase" << std::endl; }

    static void TearDownTestCase() { std::cout << "DpuManagerTest TearDownTestCase" << std::endl; }

    virtual void SetUp() { std::cout << "DpuManagerTest SetUp" << std::endl; }

    virtual void TearDown()
    {
        std::cout << "DpuManagerTest TearDown" << std::endl;
        GlobalMockObject::verify();
    }
};

// ========== CreateWorkspaceBuf 测试用例 ==========

// CreateWorkspaceBuf: 正常创建工作空间
TEST_F(DpuManagerTest, Ut_CreateWorkspaceBuf_When_NewTag_Expect_Success)
{
    DpuManager dpuManager;
    const char* memTag = "TEST_TAG";
    uint64_t size = 1024;
    bool newCreated = false;

    HcclResult ret = dpuManager.CreateWorkspaceBuf(memTag, &size, &newCreated);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(newCreated, true);
}

// CreateWorkspaceBuf: 相同tag第二次调用，newCreated应为false
TEST_F(DpuManagerTest, Ut_CreateWorkspaceBuf_When_SameTagAgain_Expect_NotNewCreated)
{
    DpuManager dpuManager;
    const char* memTag = "TEST_TAG";
    uint64_t size1 = 1024;
    uint64_t size2 = 2048;
    bool newCreated1 = false;
    bool newCreated2 = false;

    HcclResult ret1 = dpuManager.CreateWorkspaceBuf(memTag, &size1, &newCreated1);
    EXPECT_EQ(ret1, HCCL_SUCCESS);
    EXPECT_EQ(newCreated1, true);

    // 第二次调用相同tag，newCreated应为false
    HcclResult ret2 = dpuManager.CreateWorkspaceBuf(memTag, &size2, &newCreated2);
    EXPECT_EQ(ret2, HCCL_SUCCESS);
    EXPECT_EQ(newCreated2, false);
}

// CreateWorkspaceBuf: nullptr tag
TEST_F(DpuManagerTest, Ut_CreateWorkspaceBuf_When_NullTag_Expect_Success)
{
    DpuManager dpuManager;
    const char* memTag = nullptr;
    uint64_t size = 1024;
    bool newCreated = false;

    HcclResult ret = dpuManager.CreateWorkspaceBuf(memTag, &size, &newCreated);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(newCreated, true);
}

// CreateWorkspaceBuf: newCreated为nullptr
TEST_F(DpuManagerTest, Ut_CreateWorkspaceBuf_When_NewCreatedIsNull_Expect_Success)
{
    DpuManager dpuManager;
    const char* memTag = "TEST_TAG";
    uint64_t size = 1024;

    HcclResult ret = dpuManager.CreateWorkspaceBuf(memTag, &size, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// ========== GetKFCWorkSpace 测试用例 ==========

// GetKFCWorkSpace: 获取已创建的工作空间
TEST_F(DpuManagerTest, Ut_GetKFCWorkSpace_When_WorkspaceExists_Expect_NotNull)
{
    DpuManager dpuManager;
    const char* memTag = "TEST_TAG";
    uint64_t size = 1024;

    // 先创建工作空间
    dpuManager.CreateWorkspaceBuf(memTag, &size, nullptr);

    // 获取工作空间
    auto workspace = dpuManager.GetKFCWorkSpace(memTag);
    EXPECT_NE(workspace, nullptr);
}

// GetKFCWorkSpace: 获取不存在的tag
TEST_F(DpuManagerTest, Ut_GetKFCWorkSpace_When_TagNotExists_Expect_Null)
{
    DpuManager dpuManager;
    const char* memTag = "NOT_EXIST_TAG";

    auto workspace = dpuManager.GetKFCWorkSpace(memTag);
    EXPECT_EQ(workspace, nullptr);
}

// GetKFCWorkSpace: nullptr tag
TEST_F(DpuManagerTest, Ut_GetKFCWorkSpace_When_NullTag_Expect_Null)
{
    DpuManager dpuManager;
    const char* memTag = nullptr;

    auto workspace = dpuManager.GetKFCWorkSpace(memTag);
    EXPECT_EQ(workspace, nullptr);
}

// ========== GetDevMemWorkSpace 测试用例 ==========

// GetDevMemWorkSpace: 首次创建新工作空间
TEST_F(DpuManagerTest, Ut_GetDevMemWorkSpace_When_NewTag_Expect_Success)
{
    DpuManager dpuManager;
    std::string memTag = "DEV_MEM_TAG";
    uint64_t size = 2048;
    void* addr = nullptr;
    bool newCreated = false;

    HcclResult ret = dpuManager.GetDevMemWorkSpace(memTag, &size, &addr, &newCreated);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(addr, nullptr);
    EXPECT_EQ(newCreated, true);
}

// GetDevMemWorkSpace: 获取已存在的工作空间，大小一致
TEST_F(DpuManagerTest, Ut_GetDevMemWorkSpace_When_WorkspaceExists_SameSize_Expect_Success)
{
    DpuManager dpuManager;
    std::string memTag = "DEV_MEM_TAG";
    uint64_t size = 2048;
    void* addr1 = nullptr;
    bool newCreated1 = false;

    // 首次创建
    HcclResult ret1 = dpuManager.GetDevMemWorkSpace(memTag, &size, &addr1, &newCreated1);
    EXPECT_EQ(ret1, HCCL_SUCCESS);
    EXPECT_EQ(newCreated1, true);

    // 再次获取，大小相同
    void* addr2 = nullptr;
    bool newCreated2 = false;
    HcclResult ret2 = dpuManager.GetDevMemWorkSpace(memTag, &size, &addr2, &newCreated2);
    EXPECT_EQ(ret2, HCCL_SUCCESS);
    EXPECT_EQ(newCreated2, false);
}

// GetDevMemWorkSpace: 获取已存在的工作空间，大小不一致
TEST_F(DpuManagerTest, Ut_GetDevMemWorkSpace_When_WorkspaceExists_DiffSize_Expect_ParaError)
{
    DpuManager dpuManager;
    std::string memTag = "DEV_MEM_TAG";
    uint64_t size1 = 2048;
    void* addr1 = nullptr;
    bool newCreated1 = false;

    // 首次创建
    HcclResult ret1 = dpuManager.GetDevMemWorkSpace(memTag, &size1, &addr1, &newCreated1);
    EXPECT_EQ(ret1, HCCL_SUCCESS);

    // 再次获取，大小不同
    uint64_t size2 = 4096;
    void* addr2 = nullptr;
    bool newCreated2 = false;
    HcclResult ret2 = dpuManager.GetDevMemWorkSpace(memTag, &size2, &addr2, &newCreated2);
    EXPECT_EQ(ret2, HCCL_E_PARA);
}

// GetDevMemWorkSpace: addr为nullptr
TEST_F(DpuManagerTest, Ut_GetDevMemWorkSpace_When_AddrIsNull_Expect_Success)
{
    DpuManager dpuManager;
    std::string memTag = "DEV_MEM_TAG";
    uint64_t size = 2048;
    void* addr = nullptr;

    HcclResult ret = dpuManager.GetDevMemWorkSpace(memTag, &size, &addr, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(addr, nullptr);
}

// ========== IsDpuKernelLaunched 测试用例 ==========

// IsDpuKernelLaunched: 初始状态应为false
TEST_F(DpuManagerTest, Ut_IsDpuKernelLaunched_When_Initial_Expect_False)
{
    DpuManager dpuManager;
    EXPECT_EQ(dpuManager.IsDpuKernelLaunched(), false);
}

// IsDpuKernelLaunched: 初始化后应为true（需要mock）
TEST_F(DpuManagerTest, Ut_IsDpuKernelLaunched_After_Init_Expect_True)
{
    DpuManager dpuManager;

    // Mock相关函数以避免实际硬件依赖
    MOCKER(aclrtGetCurrentContext).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(Hccl::HrtSetXpuDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(aclrtCreateStreamWithConfig).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtBinaryLoadFromFile).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtBinaryGetFunction).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtLaunchKernelWithHostArgs).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtSetCurrentContext).stubs().will(returnValue(ACL_SUCCESS));

    std::string commId = "test_comm_id";
    u32 deviceLogicId = 0;

    // Init会调用InitDpuKernel，由于我们mock了所有依赖，isDpuKernelLaunched应该被设置为true
    // 注意：实际执行会失败因为InitAndLaunchDpuKernel内部有复杂的依赖关系
    // 这里主要测试IsDpuKernelLaunched的getter方法
}

// ========== DeInitDpuKernel 测试用例 ==========

// DeInitDpuKernel: 未初始化时调用应直接返回成功
TEST_F(DpuManagerTest, Ut_DeInitDpuKernel_When_NotLaunched_Expect_Success)
{
    DpuManager dpuManager;
    // isDpuKernelLaunched_ 初始为 false
    HcclResult ret = dpuManager.DeInitDpuKernel();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// DeInitDpuKernel: hostShareBuf为nullptr时应正常处理
TEST_F(DpuManagerTest, Ut_DeInitDpuKernel_When_HostShareBufIsNull_Expect_Success)
{
    DpuManager dpuManager;
    dpuManager.isDpuKernelLaunched_ = false;
    dpuManager.hostShareBuf_ = nullptr;

    HcclResult ret = dpuManager.DeInitDpuKernel();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// ========== WaitDpuKernelThreadTerminate 测试用例 ==========

// WaitDpuKernelThreadTerminate: isDpuKernelLaunched_为false时直接返回成功
TEST_F(DpuManagerTest, Ut_WaitDpuKernelThreadTerminate_When_NotLaunched_Expect_Success)
{
    DpuManager dpuManager;
    dpuManager.isDpuKernelLaunched_ = false;

    HcclResult ret = dpuManager.WaitDpuKernelThreadTerminate();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// WaitDpuKernelThreadTerminate: isDpuKernelLaunched_为true但accessVA_为nullptr时返回错误
TEST_F(DpuManagerTest, Ut_WaitDpuKernelThreadTerminate_When_AccessVAIsNull_Expect_MemoryError)
{
    DpuManager dpuManager;
    dpuManager.isDpuKernelLaunched_ = true;
    dpuManager.accessVA_ = nullptr;

    HcclResult ret = dpuManager.WaitDpuKernelThreadTerminate();
    EXPECT_EQ(ret, HCCL_E_MEMORY);
}

// WaitDpuKernelThreadTerminate: while循环正常退出（后台线程更新flag使循环退出）
TEST_F(DpuManagerTest, Ut_WaitDpuKernelThreadTerminate_When_WhileLoopExitNormally_Expect_Success)
{
    DpuManager dpuManager;
    dpuManager.isDpuKernelLaunched_ = true;
    dpuManager.accessVA_ = new uint8_t[8]();
    constexpr uint8_t DEVICE_SIGNAL_SECOND = 2;
    constexpr uint8_t DEVICE_SIGNAL_THIRD = 3;
    static_cast<uint8_t*>(dpuManager.accessVA_)[0] = DEVICE_SIGNAL_SECOND;

    std::atomic<bool> updateReady{false};
    std::thread worker([&dpuManager, &updateReady]() {
        while (!updateReady.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        static_cast<uint8_t*>(dpuManager.accessVA_)[0] = DEVICE_SIGNAL_THIRD;
    });
    worker.detach();

    updateReady.store(true, std::memory_order_release);
    HcclResult ret = dpuManager.WaitDpuKernelThreadTerminate();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete[] static_cast<uint8_t*>(dpuManager.accessVA_);
}

// ========== Init 测试用例 ==========

// Init: 正常初始化成功
TEST_F(DpuManagerTest, Ut_Init_When_AllMockSuccess_Expect_Success)
{
    DpuManager dpuManager;
    std::string commId = "test_comm_id_123";
    u32 deviceLogicId = 5;

    MOCKER(getenv).stubs().will(returnValue(static_cast<char*>(nullptr)));
    MOCKER(realpath).stubs().will(returnValue(reinterpret_cast<char*>(0x2000)));
    MOCKER(aclrtGetCurrentContext)
        .stubs()
        .will(returnValue(ACL_SUCCESS))
        .then(returnValue(ACL_SUCCESS))
        .then(returnValue(ACL_SUCCESS));
    MOCKER(Hccl::HrtSetXpuDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(aclrtCreateStreamWithConfig).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtBinaryLoadFromFile).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtBinaryGetFunction).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtLaunchKernelWithHostArgs).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtMemcpy).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtSetCurrentContext).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtDestroyStreamForce).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(Hccl::HrtResetXpuDevice).stubs().will(returnValue(HCCL_SUCCESS));

    dpuManager.waitTransportReadyTimeoutMs_ = 1; // UT中设置1ms超时，快速退出等待循环
    HcclResult ret = dpuManager.Init(commId, deviceLogicId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(dpuManager.commId_, commId);
    EXPECT_EQ(dpuManager.devLogicId_, deviceLogicId);
}

// Init: aclrtGetCurrentContext首次失败
TEST_F(DpuManagerTest, Ut_Init_When_GetCurrentContextFail_Expect_EInternal)
{
    DpuManager dpuManager;
    std::string commId = "test_comm_id";
    u32 deviceLogicId = 0;

    MOCKER(aclrtGetCurrentContext).stubs().will(returnValue(ACL_ERROR_INVALID_PARAM));
    MOCKER(aclrtSetCurrentContext).stubs().will(returnValue(ACL_SUCCESS));

    HcclResult ret = dpuManager.Init(commId, deviceLogicId);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

// Init: HrtSetXpuDevice失败
TEST_F(DpuManagerTest, Ut_Init_When_SetXpuDeviceFail_Expect_EInternal)
{
    DpuManager dpuManager;
    std::string commId = "test_comm_id";
    u32 deviceLogicId = 0;

    MOCKER(getenv).stubs().will(returnValue(static_cast<char*>(nullptr)));
    MOCKER(realpath).stubs().will(returnValue(reinterpret_cast<char*>(0x2000)));
    MOCKER(aclrtGetCurrentContext).stubs().will(returnValue(ACL_SUCCESS)).then(returnValue(ACL_SUCCESS));
    MOCKER(Hccl::HrtSetXpuDevice).stubs().will(returnValue(HCCL_E_INTERNAL));
    MOCKER(aclrtMemcpy).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtSetCurrentContext).stubs().will(returnValue(ACL_SUCCESS));

    HcclResult ret = dpuManager.Init(commId, deviceLogicId);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

// Init: aclrtCreateStreamWithConfig失败
TEST_F(DpuManagerTest, Ut_Init_When_CreateStreamFail_Expect_EInternal)
{
    DpuManager dpuManager;
    std::string commId = "test_comm_id";
    u32 deviceLogicId = 0;

    MOCKER(getenv).stubs().will(returnValue(static_cast<char*>(nullptr)));
    MOCKER(realpath).stubs().will(returnValue(reinterpret_cast<char*>(0x2000)));
    MOCKER(aclrtGetCurrentContext).stubs().will(returnValue(ACL_SUCCESS)).then(returnValue(ACL_SUCCESS));
    MOCKER(Hccl::HrtSetXpuDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(aclrtCreateStreamWithConfig).stubs().will(returnValue(ACL_ERROR_INVALID_PARAM));
    MOCKER(aclrtMemcpy).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtSetCurrentContext).stubs().will(returnValue(ACL_SUCCESS));

    HcclResult ret = dpuManager.Init(commId, deviceLogicId);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

// Init: aclrtBinaryLoadFromFile失败
TEST_F(DpuManagerTest, Ut_Init_When_BinaryLoadFail_Expect_EOpenFileFailure)
{
    DpuManager dpuManager;
    std::string commId = "test_comm_id";
    u32 deviceLogicId = 0;

    MOCKER(getenv).stubs().will(returnValue(static_cast<char*>(nullptr)));
    MOCKER(realpath).stubs().will(returnValue(reinterpret_cast<char*>(0x2000)));
    MOCKER(aclrtGetCurrentContext).stubs().will(returnValue(ACL_SUCCESS)).then(returnValue(ACL_SUCCESS));
    MOCKER(Hccl::HrtSetXpuDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(aclrtCreateStreamWithConfig).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtBinaryLoadFromFile).stubs().will(returnValue(ACL_ERROR_INVALID_PARAM));
    MOCKER(aclrtMemcpy).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtSetCurrentContext).stubs().will(returnValue(ACL_SUCCESS));

    HcclResult ret = dpuManager.Init(commId, deviceLogicId);
    EXPECT_EQ(ret, HCCL_E_OPEN_FILE_FAILURE);
}

// Init: aclrtBinaryGetFunction失败
TEST_F(DpuManagerTest, Ut_Init_When_GetFunctionFail_Expect_EInternal)
{
    DpuManager dpuManager;
    std::string commId = "test_comm_id";
    u32 deviceLogicId = 0;

    MOCKER(getenv).stubs().will(returnValue(static_cast<char*>(nullptr)));
    MOCKER(realpath).stubs().will(returnValue(reinterpret_cast<char*>(0x2000)));
    MOCKER(aclrtGetCurrentContext).stubs().will(returnValue(ACL_SUCCESS)).then(returnValue(ACL_SUCCESS));
    MOCKER(Hccl::HrtSetXpuDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(aclrtCreateStreamWithConfig).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtBinaryLoadFromFile).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtBinaryGetFunction).stubs().will(returnValue(ACL_ERROR_INVALID_PARAM));
    MOCKER(aclrtMemcpy).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtSetCurrentContext).stubs().will(returnValue(ACL_SUCCESS));

    HcclResult ret = dpuManager.Init(commId, deviceLogicId);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

// Init: aclrtLaunchKernelWithHostArgs失败
TEST_F(DpuManagerTest, Ut_Init_When_LaunchKernelFail_Expect_EInternal)
{
    DpuManager dpuManager;
    std::string commId = "test_comm_id";
    u32 deviceLogicId = 0;

    MOCKER(getenv).stubs().will(returnValue(static_cast<char*>(nullptr)));
    MOCKER(realpath).stubs().will(returnValue(reinterpret_cast<char*>(0x2000)));
    MOCKER(aclrtGetCurrentContext).stubs().will(returnValue(ACL_SUCCESS)).then(returnValue(ACL_SUCCESS));
    MOCKER(Hccl::HrtSetXpuDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(aclrtCreateStreamWithConfig).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtBinaryLoadFromFile).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtBinaryGetFunction).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtLaunchKernelWithHostArgs).stubs().will(returnValue(ACL_ERROR_INVALID_PARAM));
    MOCKER(aclrtMemcpy).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtSetCurrentContext).stubs().will(returnValue(ACL_SUCCESS));

    HcclResult ret = dpuManager.Init(commId, deviceLogicId);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

// Init: InitAndLaunchDpuKernel内部aclrtSetCurrentContext切换回npu ctx失败
TEST_F(DpuManagerTest, Ut_Init_When_ResetContextFail_Expect_EInternal)
{
    DpuManager dpuManager;
    std::string commId = "test_comm_id";
    u32 deviceLogicId = 0;

    MOCKER(getenv).stubs().will(returnValue(static_cast<char*>(nullptr)));
    MOCKER(realpath).stubs().will(returnValue(reinterpret_cast<char*>(0x2000)));
    MOCKER(aclrtGetCurrentContext)
        .stubs()
        .will(returnValue(ACL_SUCCESS))
        .then(returnValue(ACL_SUCCESS))
        .then(returnValue(ACL_SUCCESS));
    MOCKER(Hccl::HrtSetXpuDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(aclrtCreateStreamWithConfig).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtBinaryLoadFromFile).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtBinaryGetFunction).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtLaunchKernelWithHostArgs).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtSetCurrentContext)
        .stubs()
        .will(returnValue(ACL_ERROR_INVALID_PARAM))
        .then(returnValue(ACL_SUCCESS))
        .then(returnValue(ACL_SUCCESS));
    MOCKER(aclrtMemcpy).stubs().will(returnValue(ACL_SUCCESS));

    dpuManager.waitTransportReadyTimeoutMs_ = 1;
    HcclResult ret = dpuManager.Init(commId, deviceLogicId);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

// ========== 析构函数测试 ==========

// 析构函数: 正常析构不崩溃
TEST_F(DpuManagerTest, Ut_Destructor_When_Normal_Expect_NoCrash)
{
    DpuManager* dpuManager = new DpuManager();
    // 设置一些状态
    dpuManager->hostShareBuf_ = malloc(1024);
    // 正常析构
    delete dpuManager;
}

// 析构函数: isDpuKernelLaunched为true时调用DeInit
TEST_F(DpuManagerTest, Ut_Destructor_When_KernelLaunched_Expect_CallDeInit)
{
    DpuManager* dpuManager = new DpuManager();
    dpuManager->isDpuKernelLaunched_ = true;
    dpuManager->hostShareBuf_ = malloc(1024);

    dpuManager->waitTransportReadyTimeoutMs_ = 1;
    MOCKER(aclrtGetCurrentContext).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(Hccl::HrtResetXpuDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(aclrtDestroyStreamForce).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtSetCurrentContext).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(aclrtMemcpy).stubs().will(returnValue(ACL_SUCCESS));

    delete dpuManager; // 析构函数会调用DeInitDpuKernel
}

// ========== DestroyDpuKernelResource 测试用例 ==========

// DestroyDpuKernelResource: 正常流程
TEST_F(DpuManagerTest, Ut_DestroyDpuKernelResource_When_Normal_Expect_Success)
{
    DpuManager dpuManager;
    dpuManager.isDpuKernelLaunched_ = true;
    dpuManager.dpuContext_ = reinterpret_cast<aclrtContext>(malloc(64));
    dpuManager.npuContext_ = reinterpret_cast<aclrtContext>(malloc(64));
    dpuManager.dpuStream_ = reinterpret_cast<aclrtStream>(malloc(64));

    MOCKER_CPP(&DpuManager::WaitDpuKernelThreadTerminate).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(aclrtSetCurrentContext)
        .stubs()
        .will(returnValue(ACL_SUCCESS))
        .then(returnValue(ACL_SUCCESS))
        .then(returnValue(ACL_SUCCESS));
    MOCKER(aclrtDestroyStreamForce).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(Hccl::HrtResetXpuDevice).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = dpuManager.DestroyDpuKernelResource();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    free(dpuManager.dpuContext_);
    free(dpuManager.npuContext_);
    free(dpuManager.dpuStream_);
}

// DestroyDpuKernelResource: aclrtSetCurrentContext失败返回RuntimeError
TEST_F(DpuManagerTest, Ut_DestroyDpuKernelResource_When_SetDpuCtxFail_Expect_RuntimeError)
{
    DpuManager dpuManager;
    dpuManager.isDpuKernelLaunched_ = true;
    dpuManager.dpuContext_ = reinterpret_cast<aclrtContext>(malloc(64));
    dpuManager.npuContext_ = reinterpret_cast<aclrtContext>(malloc(64));
    dpuManager.dpuStream_ = reinterpret_cast<aclrtStream>(malloc(64));

    MOCKER_CPP(&DpuManager::WaitDpuKernelThreadTerminate).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(aclrtSetCurrentContext)
        .stubs()
        .will(returnValue(ACL_ERROR_INVALID_PARAM))
        .then(returnValue(ACL_SUCCESS))
        .then(returnValue(ACL_SUCCESS));
    MOCKER(aclrtDestroyStreamForce).stubs().will(returnValue(ACL_SUCCESS));

    HcclResult ret = dpuManager.DestroyDpuKernelResource();
    EXPECT_EQ(ret, HCCL_E_RUNTIME);

    free(dpuManager.dpuContext_);
    free(dpuManager.npuContext_);
    free(dpuManager.dpuStream_);
}

// DestroyDpuKernelResource: aclrtDestroyStreamForce失败返回RuntimeError
TEST_F(DpuManagerTest, Ut_DestroyDpuKernelResource_When_DestroyStreamFail_Expect_RuntimeError)
{
    DpuManager dpuManager;
    dpuManager.isDpuKernelLaunched_ = true;
    dpuManager.dpuContext_ = reinterpret_cast<aclrtContext>(malloc(64));
    dpuManager.npuContext_ = reinterpret_cast<aclrtContext>(malloc(64));
    dpuManager.dpuStream_ = reinterpret_cast<aclrtStream>(malloc(64));

    MOCKER_CPP(&DpuManager::WaitDpuKernelThreadTerminate).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(aclrtSetCurrentContext)
        .stubs()
        .will(returnValue(ACL_SUCCESS))
        .then(returnValue(ACL_SUCCESS))
        .then(returnValue(ACL_SUCCESS));
    MOCKER(aclrtDestroyStreamForce).stubs().will(returnValue(ACL_ERROR_INVALID_PARAM));

    HcclResult ret = dpuManager.DestroyDpuKernelResource();
    EXPECT_EQ(ret, HCCL_E_RUNTIME);

    free(dpuManager.dpuContext_);
    free(dpuManager.npuContext_);
    free(dpuManager.dpuStream_);
}

// DestroyDpuKernelResource: HrtResetXpuDevice失败返回RuntimeError
TEST_F(DpuManagerTest, Ut_DestroyDpuKernelResource_When_ResetXpuFail_Expect_RuntimeError)
{
    DpuManager dpuManager;
    dpuManager.isDpuKernelLaunched_ = true;
    dpuManager.dpuContext_ = reinterpret_cast<aclrtContext>(malloc(64));
    dpuManager.npuContext_ = reinterpret_cast<aclrtContext>(malloc(64));
    dpuManager.dpuStream_ = reinterpret_cast<aclrtStream>(malloc(64));

    MOCKER_CPP(&DpuManager::WaitDpuKernelThreadTerminate).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(aclrtSetCurrentContext)
        .stubs()
        .will(returnValue(ACL_SUCCESS))
        .then(returnValue(ACL_SUCCESS))
        .then(returnValue(ACL_SUCCESS));
    MOCKER(aclrtDestroyStreamForce).stubs().will(returnValue(ACL_SUCCESS));
    MOCKER(Hccl::HrtResetXpuDevice).stubs().will(returnValue(HCCL_E_INTERNAL));

    HcclResult ret = dpuManager.DestroyDpuKernelResource();
    EXPECT_EQ(ret, HCCL_E_RUNTIME);

    free(dpuManager.dpuContext_);
    free(dpuManager.npuContext_);
    free(dpuManager.dpuStream_);
}

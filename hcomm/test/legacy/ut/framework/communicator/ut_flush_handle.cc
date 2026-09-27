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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>

#define private public
#define protected public
#include "flush_handle.h"
#include "rdma_handle_manager.h"
#include "orion_adapter_rts.h"
#include "orion_adapter_hccp.h"
#include "hccp.h"
#undef private
#undef protected

#include <cstdlib>

using namespace Hccl;

class FlushHandleTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "FlushHandleTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "FlushHandleTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        fakeRdmaHandle = reinterpret_cast<void*>(0xDEAD0001);
        fakeQpHandle = reinterpret_cast<void*>(0xDEAD0002);
        fakeMrHandle = reinterpret_cast<void*>(0xDEAD0003);
        fakeLocalMem = reinterpret_cast<void*>(0xBEEF0001);
        fakeDeviceMem = reinterpret_cast<void*>(0xBEEF0002);
    }

    void* fakeRdmaHandle;
    void* fakeQpHandle;
    void* fakeMrHandle;
    void* fakeLocalMem;
    void* fakeDeviceMem;
};

TEST_F(FlushHandleTest, DoubleDestroy_Expect_Idempotent)
{
    GlobalMockObject::verify();
    MOCKER(RaDeregisterMr).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(0));
    MOCKER(RaQpDestroy).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER(HrtFree).stubs();

    FlushHandle handle;
    handle.rdmaHandle = fakeRdmaHandle;
    handle.localMem = fakeLocalMem;
    handle.localMrHandle = fakeMrHandle;
    handle.qpHandle = fakeQpHandle;
    handle.SetFlushOpcodeSupport();

    // 第一次 Destroy — 清理所有资源并置空
    auto ret1 = handle.Destroy();
    EXPECT_EQ(ret1, HCCL_SUCCESS);

    // 第二次 Destroy — 所有资源已空，应安全返回
    auto ret2 = handle.Destroy();
    EXPECT_EQ(ret2, HCCL_SUCCESS);
}

TEST_F(FlushHandleTest, Init_WhenFlushOpcodePath_Expect_SuccessAndFlagSet)
{
    GlobalMockObject::verify();
    MOCKER_CPP(&RdmaHandleManager::GetByAddr)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(fakeRdmaHandle));
    // lbMax > 0 → 走 flushOpcode 路径
    int lbMax = 1;
    MOCKER(RaGetLbMax).stubs().with(mockcpp::any(), outBoundP(&lbMax, sizeof(lbMax))).will(returnValue(0));
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(fakeLocalMem));
    MOCKER(RaLoopbackQpCreate).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(0));
    MOCKER(RaRegisterMr)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&fakeMrHandle, sizeof(fakeMrHandle)))
        .will(returnValue(0));
    MOCKER(HrtFree).stubs();

    FlushHandle handle;
    IpAddress ip("192.168.1.1");
    auto ret = handle.Init(ip, 0);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(handle.flushIsInitialized);
    EXPECT_TRUE(handle.GetFlushOpcodeSupport());
    EXPECT_NE(handle.localMem, nullptr);
    EXPECT_NE(handle.deviceMem, nullptr);
}

TEST_F(FlushHandleTest, Init_WhenAllocateLocalMemoryFails_Expect_MemoryError)
{
    GlobalMockObject::verify();
    MOCKER_CPP(&RdmaHandleManager::GetByAddr)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(fakeRdmaHandle));
    MOCKER(RaGetLbMax).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(0));
    // AllocateLocalMemory（HrtMalloc）失败
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue((void*)nullptr));
    MOCKER(RaDeregisterMr).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(0));
    MOCKER(RaQpDestroy).stubs().with(mockcpp::any()).will(returnValue(0));

    FlushHandle handle;
    IpAddress ip("192.168.1.1");
    auto ret = handle.Init(ip, 0);

    EXPECT_EQ(ret, HCCL_E_MEMORY);
    EXPECT_FALSE(handle.flushIsInitialized);
}

TEST_F(FlushHandleTest, Init_WhenCreateLoopbackQpFails_Expect_RoceConnectError)
{
    GlobalMockObject::verify();
    MOCKER_CPP(&RdmaHandleManager::GetByAddr)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(fakeRdmaHandle));
    MOCKER(RaGetLbMax).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(0));
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(fakeLocalMem));
    MOCKER(RaLoopbackQpCreate).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(-1));
    MOCKER(RaDeregisterMr).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(0));
    MOCKER(RaQpDestroy).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER(HrtFree).stubs();

    FlushHandle handle;
    IpAddress ip("192.168.1.1");
    auto ret = handle.Init(ip, 0);

    EXPECT_EQ(ret, HCCL_E_ROCE_CONNECT);
}

TEST_F(FlushHandleTest, Init_WhenRegisterLocalMrFails_Expect_MemoryError)
{
    GlobalMockObject::verify();
    MOCKER_CPP(&RdmaHandleManager::GetByAddr)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(fakeRdmaHandle));
    MOCKER(RaGetLbMax).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(0));
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(fakeLocalMem));
    MOCKER(RaLoopbackQpCreate).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(0));
    // RaRegisterMr 第一次调用（Local MR）失败
    MOCKER(RaRegisterMr).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(-1));
    MOCKER(RaDeregisterMr).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(0));
    MOCKER(RaQpDestroy).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER(HrtFree).stubs();

    FlushHandle handle;
    IpAddress ip("192.168.1.1");
    auto ret = handle.Init(ip, 0);

    EXPECT_EQ(ret, HCCL_E_MEMORY);
}

TEST_F(FlushHandleTest, Destroy_AfterSuccessfulInit_Expect_AllResourcesCleanedUp)
{
    GlobalMockObject::verify();
    MOCKER_CPP(&RdmaHandleManager::GetByAddr)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(fakeRdmaHandle));
    int lbMax = 1;
    MOCKER(RaGetLbMax).stubs().with(mockcpp::any(), outBoundP(&lbMax, sizeof(lbMax))).will(returnValue(0));
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(fakeLocalMem));
    MOCKER(RaLoopbackQpCreate).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any()).will(returnValue(0));
    MOCKER(RaRegisterMr)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&fakeMrHandle, sizeof(fakeMrHandle)))
        .will(returnValue(0));
    MOCKER(HrtFree).stubs();
    MOCKER(RaDeregisterMr).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(0));
    MOCKER(RaQpDestroy).stubs().with(mockcpp::any()).will(returnValue(0));

    FlushHandle handle;
    IpAddress ip("192.168.1.1");
    auto initRet = handle.Init(ip, 0);
    EXPECT_EQ(initRet, HCCL_SUCCESS);

    auto destroyRet = handle.Destroy();
    EXPECT_EQ(destroyRet, HCCL_SUCCESS);
}

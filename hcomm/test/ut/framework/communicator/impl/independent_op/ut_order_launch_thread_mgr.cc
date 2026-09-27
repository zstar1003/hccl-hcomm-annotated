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
#include "../../hccl_api_base_test.h"
#include "order_launch_thread_mgr.h"

using namespace hccl;

static u64 g_mockCtx = 0;
static ThreadHandle g_mockThreadHandle = 0;
static int64_t g_mockCoreNum = 0;
static HcommResult g_mockAllocRet = HCCL_SUCCESS;
static HcommResult g_mockAllocWithStreamRet = HCCL_SUCCESS;
static ThreadHandle g_nextMockHandle = 0x1000;

static aclError StubAclrtGetCurrentContext(aclrtContext* ctx)
{
    *ctx = reinterpret_cast<aclrtContext>(g_mockCtx);
    return ACL_SUCCESS;
}

static HcommResult
StubHcommThreadAlloc(CommEngine engine, uint32_t threadNum, const uint32_t* notifyNum, ThreadHandle* handle)
{
    if (g_mockAllocRet == HCCL_SUCCESS) {
        *handle = g_nextMockHandle++;
    }
    return g_mockAllocRet;
}

static HcommResult
StubHcommThreadAllocWithStream(CommEngine engine, void* stream, uint32_t notifyNum, ThreadHandle* handle)
{
    if (g_mockAllocWithStreamRet == HCCL_SUCCESS) {
        *handle = g_mockThreadHandle;
    }
    return g_mockAllocWithStreamRet;
}

static aclError StubAclrtGetDeviceInfo(uint32_t deviceId, aclrtDevAttr attr, int64_t* val)
{
    *val = g_mockCoreNum;
    return ACL_SUCCESS;
}

class OrderLaunchThreadMgrTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        mgr_ = std::make_unique<OrderLaunchThreadMgr>();
        g_nextMockHandle = 0x1000;
    }
    void TearDown() override
    {
        mgr_.reset();
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }

    void MockGetCurrentContext(u64 ctx)
    {
        g_mockCtx = ctx;
        MOCKER(aclrtGetCurrentContext).stubs().will(invoke(StubAclrtGetCurrentContext));
    }

    void MockHcommThreadAlloc(HcommResult ret)
    {
        g_mockThreadHandle = static_cast<ThreadHandle>(0x1000);
        g_mockAllocRet = ret;
        HcommResult (*allocFunc)(CommEngine, uint32_t, const uint32_t*, ThreadHandle*) = HcommThreadAlloc;
        MOCKER(allocFunc).stubs().will(invoke(StubHcommThreadAlloc));
    }

    void MockHcommThreadAllocWithStream(HcommResult ret)
    {
        g_mockThreadHandle = static_cast<ThreadHandle>(0x2000);
        g_mockAllocWithStreamRet = ret;
        MOCKER(HcommThreadAllocWithStream).stubs().will(invoke(StubHcommThreadAllocWithStream));
    }

    void MockHcommThreadFree(HcommResult ret) { MOCKER(HcommThreadFree).stubs().will(returnValue(ret)); }

    void MockHcommThreadFreeWithStream(HcommResult ret)
    {
        MOCKER(HcommThreadFreeWithStream).stubs().will(returnValue(ret));
    }

    void MockAclrtGetDeviceInfo(u32 blockNum)
    {
        g_mockCoreNum = static_cast<int64_t>(blockNum);
        MOCKER(aclrtGetDeviceInfo).stubs().will(invoke(StubAclrtGetDeviceInfo));
    }

    std::unique_ptr<OrderLaunchThreadMgr> mgr_;
};

/* ======================== RegisterOrderLaunch ======================== */

TEST_F(OrderLaunchThreadMgrTest, Ut_RegisterOrderLaunch_When_NewGroup_Expect_Success)
{
    HcclResult ret = mgr_->RegisterOrderLaunch("group1");
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OrderLaunchThreadMgrTest, Ut_RegisterOrderLaunch_When_DuplicateGroup_Expect_Success)
{
    mgr_->RegisterOrderLaunch("group1");
    HcclResult ret = mgr_->RegisterOrderLaunch("group1");
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OrderLaunchThreadMgrTest, Ut_RegisterOrderLaunch_When_MultiGroup_Expect_Success)
{
    EXPECT_EQ(mgr_->RegisterOrderLaunch("group1"), HCCL_SUCCESS);
    EXPECT_EQ(mgr_->RegisterOrderLaunch("group2"), HCCL_SUCCESS);
    EXPECT_EQ(mgr_->RegisterOrderLaunch("group3"), HCCL_SUCCESS);
}

/* ======================== UnRegisterOrderLaunch ======================== */

TEST_F(OrderLaunchThreadMgrTest, Ut_UnRegisterOrderLaunch_When_RegisteredGroup_Expect_Success)
{
    mgr_->RegisterOrderLaunch("group1");
    HcclResult ret = mgr_->UnRegisterOrderLaunch("group1");
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OrderLaunchThreadMgrTest, Ut_UnRegisterOrderLaunch_When_UnregisteredGroup_Expect_Success)
{
    HcclResult ret = mgr_->UnRegisterOrderLaunch("group_not_exist");
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OrderLaunchThreadMgrTest, Ut_UnRegisterOrderLaunch_After_Destroy_Expect_Success)
{
    mgr_.reset();
    mgr_ = std::make_unique<OrderLaunchThreadMgr>();
    HcclResult ret = mgr_->UnRegisterOrderLaunch("group1");
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

/* ======================== EnsureContextRes (via EnsureOrderThread) ======================== */

TEST_F(OrderLaunchThreadMgrTest, Ut_EnsureContextRes_When_NewContext_Expect_Success)
{
    MockGetCurrentContext(0x100);
    MockHcommThreadAlloc(HCCL_SUCCESS);
    MockAclrtGetDeviceInfo(0);

    ThreadHandle thread = 0;
    HcclResult ret = mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "group1", 1, thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

/* ======================== EnsureOrderThread ======================== */

TEST_F(OrderLaunchThreadMgrTest, Ut_EnsureOrderThread_When_HcommThreadAllocFails_Expect_Error)
{
    MockGetCurrentContext(0x100);
    MockHcommThreadAlloc(static_cast<HcommResult>(HCCL_E_RUNTIME));
    MockAclrtGetDeviceInfo(0);

    ThreadHandle warmup = 0;
    mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "warmup", 1, warmup);

    ThreadHandle thread = 0;
    HcclResult ret = mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "group1", 1, thread);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(OrderLaunchThreadMgrTest, Ut_EnsureOrderThread_When_GetCurrentContextFails_Expect_Error)
{
    MOCKER(aclrtGetCurrentContext).stubs().will(returnValue(static_cast<aclError>(1)));

    ThreadHandle thread = 0;
    HcclResult ret = mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "group1", 1, thread);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(OrderLaunchThreadMgrTest, Ut_EnsureOrderThread_When_GroupCountLeBlockNum_Expect_ThreadZero)
{
    MockGetCurrentContext(0x100);
    MockAclrtGetDeviceInfo(10);
    mgr_->RegisterOrderLaunch("group1");

    ThreadHandle thread = 0;
    HcclResult ret = mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "group1", 1, thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(thread, static_cast<ThreadHandle>(0));
}

TEST_F(OrderLaunchThreadMgrTest, Ut_EnsureOrderThread_When_GroupCountGtBlockNum_Expect_ThreadCreated)
{
    MockGetCurrentContext(0x100);
    MockAclrtGetDeviceInfo(1);
    MockHcommThreadAlloc(HCCL_SUCCESS);
    mgr_->RegisterOrderLaunch("group1");
    mgr_->RegisterOrderLaunch("group2");
    mgr_->RegisterOrderLaunch("group3");

    ThreadHandle warmup = 0;
    mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "group1", 1, warmup);

    ThreadHandle thread = 0;
    HcclResult ret = mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "group2", 1, thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(thread, static_cast<ThreadHandle>(0));
}

TEST_F(OrderLaunchThreadMgrTest, Ut_EnsureOrderThread_When_ThreadExists_Expect_Reuse)
{
    MockGetCurrentContext(0x100);
    MockAclrtGetDeviceInfo(1);
    MockHcommThreadAlloc(HCCL_SUCCESS);
    mgr_->RegisterOrderLaunch("group1");
    mgr_->RegisterOrderLaunch("group2");

    ThreadHandle warmup = 0;
    mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "warmup", 1, warmup);

    ThreadHandle thread1 = 0;
    mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "group1", 1, thread1);

    ThreadHandle thread2 = 0;
    HcclResult ret = mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "group2", 1, thread2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(thread1, thread2);
}

TEST_F(OrderLaunchThreadMgrTest, Ut_EnsureOrderThread_When_AclgraphMode_Expect_Success)
{
    MockGetCurrentContext(0x100);
    MockAclrtGetDeviceInfo(1);
    MockHcommThreadAlloc(HCCL_SUCCESS);
    mgr_->RegisterOrderLaunch("group1");
    mgr_->RegisterOrderLaunch("group2");

    ThreadHandle warmup = 0;
    mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "warmup", 1, warmup);

    ThreadHandle thread = 0;
    HcclResult ret = mgr_->EnsureOrderThread(OrderThreadMode::ACLGRAPH, "group1", 1, thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(thread, static_cast<ThreadHandle>(0));
}

/* ======================== EnsureDeviceOrderThread ======================== */

TEST_F(OrderLaunchThreadMgrTest, Ut_EnsureDeviceOrderThread_When_GroupCountLeBlockNum_Expect_ThreadZero)
{
    MockGetCurrentContext(0x100);
    MockAclrtGetDeviceInfo(10);
    mgr_->RegisterOrderLaunch("group1");

    ThreadHandle thread = 0;
    HcclResult ret = mgr_->EnsureDeviceOrderThread(nullptr, "group1", 1, thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(thread, static_cast<ThreadHandle>(0));
}

TEST_F(OrderLaunchThreadMgrTest, Ut_EnsureDeviceOrderThread_When_CollCommNullAndNotDisabled_Expect_Error)
{
    MockGetCurrentContext(0x100);
    MockHcommThreadAlloc(HCCL_SUCCESS);
    MockAclrtGetDeviceInfo(1);
    mgr_->RegisterOrderLaunch("group1");
    mgr_->RegisterOrderLaunch("group2");

    ThreadHandle warmup1 = 0;
    mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "warmup1", 1, warmup1);
    ThreadHandle warmup2 = 0;
    mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "warmup2", 1, warmup2);

    ThreadHandle thread = 0;
    HcclResult ret = mgr_->EnsureDeviceOrderThread(nullptr, "group1", 1, thread);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

/* ======================== GetHcomAttachedThreadByGroup ======================== */

TEST_F(OrderLaunchThreadMgrTest, Ut_GetHcomAttachedThreadByGroup_When_NotSet_Expect_Zero)
{
    MockGetCurrentContext(0x100);
    MockAclrtGetDeviceInfo(1);
    ThreadHandle ret = mgr_->GetHcomAttachedThreadByGroup("group1");
    EXPECT_EQ(ret, static_cast<ThreadHandle>(0));
}

TEST_F(OrderLaunchThreadMgrTest, Ut_GetHcomAttachedThreadByGroup_When_GroupCountLeBlockNum_Expect_Zero)
{
    MockGetCurrentContext(0x100);
    MockHcommThreadAllocWithStream(HCCL_SUCCESS);
    MockHcommThreadFree(HCCL_SUCCESS);
    MockHcommThreadFreeWithStream(HCCL_SUCCESS);
    MockAclrtGetDeviceInfo(10);

    void* fakeStream = reinterpret_cast<void*>(0x1234);
    HcclResult ret = mgr_->SetAttachedStream("group1", 100, fakeStream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ThreadHandle handle = mgr_->GetHcomAttachedThreadByGroup("group1");
    EXPECT_EQ(handle, static_cast<ThreadHandle>(0));
}

TEST_F(OrderLaunchThreadMgrTest, Ut_GetHcomAttachedThreadByGroup_When_Set_Expect_Handle)
{
    MockGetCurrentContext(0x100);
    MockHcommThreadAllocWithStream(HCCL_SUCCESS);
    MockHcommThreadFree(HCCL_SUCCESS);
    MockHcommThreadFreeWithStream(HCCL_SUCCESS);
    MockAclrtGetDeviceInfo(1);
    mgr_->RegisterOrderLaunch("group1");
    mgr_->RegisterOrderLaunch("group2");

    ThreadHandle warmup = 0;
    mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "warmup", 1, warmup);

    void* fakeStream = reinterpret_cast<void*>(0x1234);
    HcclResult ret = mgr_->SetAttachedStream("group1", 100, fakeStream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ThreadHandle handle = mgr_->GetHcomAttachedThreadByGroup("group1");
    EXPECT_NE(handle, static_cast<ThreadHandle>(0));
}

TEST_F(OrderLaunchThreadMgrTest, Ut_GetHcomAttachedThreadByGroup_When_DifferentGroup_Expect_Zero)
{
    MockGetCurrentContext(0x100);
    MockHcommThreadAllocWithStream(HCCL_SUCCESS);
    MockHcommThreadFree(HCCL_SUCCESS);
    MockHcommThreadFreeWithStream(HCCL_SUCCESS);
    MockAclrtGetDeviceInfo(1);
    mgr_->RegisterOrderLaunch("group1");
    mgr_->RegisterOrderLaunch("group2");

    ThreadHandle warmup = 0;
    mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "warmup", 1, warmup);

    void* fakeStream = reinterpret_cast<void*>(0x1234);
    mgr_->SetAttachedStream("group1", 100, fakeStream);
    mgr_->SetAttachedStream("group2", 200, fakeStream);

    ThreadHandle handle = mgr_->GetHcomAttachedThreadByGroup("group3");
    EXPECT_EQ(handle, static_cast<ThreadHandle>(0));
}

/* ======================== SetAttachedStream ======================== */

TEST_F(OrderLaunchThreadMgrTest, Ut_SetAttachedStream_When_NullStream_Expect_Error)
{
    HcclResult ret = mgr_->SetAttachedStream("group1", 100, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(OrderLaunchThreadMgrTest, Ut_SetAttachedStream_When_AllocFails_Expect_Error)
{
    MockHcommThreadAllocWithStream(static_cast<HcommResult>(HCCL_E_RUNTIME));

    void* fakeStream = reinterpret_cast<void*>(0x1234);
    HcclResult ret = mgr_->SetAttachedStream("group1", 100, fakeStream);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(OrderLaunchThreadMgrTest, Ut_SetAttachedStream_When_Overwrite_Expect_Success)
{
    MockHcommThreadAllocWithStream(HCCL_SUCCESS);
    MockHcommThreadFree(HCCL_SUCCESS);
    MockHcommThreadFreeWithStream(HCCL_SUCCESS);

    void* fakeStream1 = reinterpret_cast<void*>(0x1234);
    mgr_->SetAttachedStream("group1", 100, fakeStream1);

    void* fakeStream2 = reinterpret_cast<void*>(0x5678);
    HcclResult ret = mgr_->SetAttachedStream("group1", 100, fakeStream2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

/* ======================== OrderLaunchThreadAcquire ======================== */

TEST_F(OrderLaunchThreadMgrTest, Ut_OrderLaunchThreadAcquire_When_InvalidUseType_Expect_Error)
{
    ThreadHandle thread = 0;
    HcclResult ret
        = mgr_->OrderLaunchThreadAcquire(static_cast<HcclDedicatedThreadType>(99), nullptr, "group1", 1, thread);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(OrderLaunchThreadMgrTest, Ut_OrderLaunchThreadAcquire_When_Opbase_Expect_Success)
{
    MockGetCurrentContext(0x100);
    MockAclrtGetDeviceInfo(0);
    MockHcommThreadAlloc(HCCL_SUCCESS);

    ThreadHandle warmup = 0;
    mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "warmup", 1, warmup);

    ThreadHandle thread = 0;
    HcclResult ret
        = mgr_->OrderLaunchThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_OPBASE, nullptr, "group1", 1, thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(thread, static_cast<ThreadHandle>(0));
}

TEST_F(OrderLaunchThreadMgrTest, Ut_OrderLaunchThreadAcquire_When_Aclgraph_Expect_Success)
{
    MockGetCurrentContext(0x100);
    MockAclrtGetDeviceInfo(0);
    MockHcommThreadAlloc(HCCL_SUCCESS);

    ThreadHandle warmup = 0;
    mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "warmup", 1, warmup);

    ThreadHandle thread = 0;
    HcclResult ret = mgr_->OrderLaunchThreadAcquire(
        HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_ACLGRAPH, nullptr, "group1", 1, thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(thread, static_cast<ThreadHandle>(0));
}

TEST_F(OrderLaunchThreadMgrTest, Ut_OrderLaunchThreadAcquire_When_Ge_Expect_Success)
{
    MockGetCurrentContext(0x100);
    MockHcommThreadAllocWithStream(HCCL_SUCCESS);
    MockHcommThreadFree(HCCL_SUCCESS);
    MockHcommThreadFreeWithStream(HCCL_SUCCESS);
    MockAclrtGetDeviceInfo(1);
    mgr_->RegisterOrderLaunch("group1");
    mgr_->RegisterOrderLaunch("group2");

    ThreadHandle warmup = 0;
    mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "warmup", 1, warmup);

    void* fakeStream = reinterpret_cast<void*>(0x1234);
    mgr_->SetAttachedStream("group1", 100, fakeStream);

    ThreadHandle thread = 0;
    HcclResult ret
        = mgr_->OrderLaunchThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_GE, nullptr, "group1", 1, thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(thread, static_cast<ThreadHandle>(0));
}

TEST_F(OrderLaunchThreadMgrTest, Ut_OrderLaunchThreadAcquire_When_GeNotSet_Expect_Success_ThreadZero)
{
    MockGetCurrentContext(0x100);
    MockAclrtGetDeviceInfo(1);
    ThreadHandle thread = 0;
    HcclResult ret
        = mgr_->OrderLaunchThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_GE, nullptr, "group1", 1, thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(thread, static_cast<ThreadHandle>(0));
}

TEST_F(OrderLaunchThreadMgrTest, Ut_OrderLaunchThreadAcquire_When_DeviceCollCommNull_Expect_Error)
{
    MockGetCurrentContext(0x100);
    MockHcommThreadAlloc(HCCL_SUCCESS);
    MockAclrtGetDeviceInfo(1);
    mgr_->RegisterOrderLaunch("group1");
    mgr_->RegisterOrderLaunch("group2");

    ThreadHandle warmup1 = 0;
    mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "warmup1", 1, warmup1);
    ThreadHandle warmup2 = 0;
    mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "warmup2", 1, warmup2);

    ThreadHandle thread = 0;
    HcclResult ret
        = mgr_->OrderLaunchThreadAcquire(HCCL_DED_THREAD_TYPE_AICPU_ORDER_LAUNCH_DEVICE, nullptr, "group1", 1, thread);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

/* ======================== Destroy ======================== */

TEST_F(OrderLaunchThreadMgrTest, Ut_Destroy_When_NoResources_Expect_NoCrash)
{
    mgr_.reset();
    SUCCEED();
}

TEST_F(OrderLaunchThreadMgrTest, Ut_Destroy_When_HasResources_Expect_NoCrash)
{
    MockGetCurrentContext(0x100);
    MockAclrtGetDeviceInfo(0);
    MockHcommThreadAlloc(HCCL_SUCCESS);
    MockHcommThreadFree(HCCL_SUCCESS);

    ThreadHandle thread = 0;
    mgr_->EnsureOrderThread(OrderThreadMode::OPBASE, "group1", 1, thread);

    mgr_.reset();
    SUCCEED();
}

TEST_F(OrderLaunchThreadMgrTest, Ut_Destroy_When_HasDeviceOrderThread_Expect_NoCrash)
{
    MockGetCurrentContext(0x100);
    MockAclrtGetDeviceInfo(1);
    mgr_->RegisterOrderLaunch("group1");
    mgr_->RegisterOrderLaunch("group2");

    mgr_.reset();
    SUCCEED();
}

TEST_F(OrderLaunchThreadMgrTest, Ut_Destroy_When_HasAttachedStream_Expect_NoCrash)
{
    MockHcommThreadAllocWithStream(HCCL_SUCCESS);
    MockHcommThreadFree(HCCL_SUCCESS);
    MockHcommThreadFreeWithStream(HCCL_SUCCESS);

    void* fakeStream = reinterpret_cast<void*>(0x1234);
    mgr_->SetAttachedStream("group1", 100, fakeStream);

    mgr_.reset();
    SUCCEED();
}

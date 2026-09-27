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
#include <mockcpp/mockcpp.hpp>
#include "adapter_rts_common.h"
#include "peterson_lock.h"
#include <cstdlib>

using namespace hccl;

static HcclResult stub_hrtMemSyncCopy(void* dst, size_t dstMax, const void* src, size_t count, HcclRtMemcpyKind kind)
{
    memcpy(dst, src, count);
    return HCCL_SUCCESS;
}

static HcclResult stub_hrtMemSet(void* dst, size_t dstMax, size_t count)
{
    memset(dst, 0, count);
    return HCCL_SUCCESS;
}

static HcclResult stub_hrtMalloc(void** devPtr, u64 size, bool level2Address)
{
    *devPtr = malloc(size);

    return *devPtr == nullptr ? HCCL_E_INTERNAL : HCCL_SUCCESS;
}

static HcclResult stub_hrtFree(void* devPtr)
{
    if (devPtr != nullptr) {
        free(devPtr);
    }

    return HCCL_SUCCESS;
}

class PetersonLockTest : public testing::Test {
public:
    static void SetUpTestCase()
    {
        std::cout << "PetersonLockTest SetUp" << std::endl;
        GlobalMockObject::verify();
        MOCKER(hrtMemSyncCopy).stubs().with(mockcpp::any()).will(invoke(stub_hrtMemSyncCopy));
        MOCKER(hrtMemSet).stubs().with(mockcpp::any()).will(invoke(stub_hrtMemSet));
        MOCKER(hrtMalloc).stubs().with(mockcpp::any()).will(invoke(stub_hrtMalloc));
        MOCKER(hrtFree).stubs().with(mockcpp::any()).will(invoke(stub_hrtFree));
    }

    static void TearDownTestCase()
    {
        std::cout << "PetersonLockTest TearDown" << std::endl;
        GlobalMockObject::verify();
    }

    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }

    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }
};

TEST_F(PetersonLockTest, ut_lock)
{
    PetersonLock hostLock(PetersonLock::DEFAULT_LOCK_TIMEOUT_SEC);
    ASSERT_EQ(hostLock.Init(), HCCL_SUCCESS);

    PetersonLock deviceLock(reinterpret_cast<void*>(hostLock.GetDevMemAddr()), PetersonLock::DEFAULT_LOCK_TIMEOUT_SEC);
    ASSERT_EQ(deviceLock.Init(), HCCL_SUCCESS);

    int loopCount = 100;
    int result = 0;
    int expect = loopCount * 2;

    auto testLockFunc = [loopCount, &result](PetersonLock& lock) {
        for (int i = 0; i < loopCount; ++i) {
            lock.Lock();
            result++;
            lock.Unlock();
        }
    };

    std::thread host(testLockFunc, std::ref(hostLock));
    std::thread device(testLockFunc, std::ref(deviceLock));
    host.join();
    device.join();

    EXPECT_EQ(result, expect);
}

TEST_F(PetersonLockTest, ut_lock_guard)
{
    PetersonLock hostLock(PetersonLock::DEFAULT_LOCK_TIMEOUT_SEC);
    ASSERT_EQ(hostLock.Init(), HCCL_SUCCESS);

    PetersonLock deviceLock(reinterpret_cast<void*>(hostLock.GetDevMemAddr()), PetersonLock::DEFAULT_LOCK_TIMEOUT_SEC);
    ASSERT_EQ(deviceLock.Init(), HCCL_SUCCESS);

    int loopCount = 100;
    int result = 0;
    int expect = loopCount * 2;

    auto testLockFunc = [loopCount, &result](PetersonLock& lock) {
        for (int i = 0; i < loopCount; ++i) {
            PetersonLockGuard guard(&lock);
            EXPECT_EQ(guard.IsLockFailed(), false);
            result++;
        }
    };

    std::thread host(testLockFunc, std::ref(hostLock));
    std::thread device(testLockFunc, std::ref(deviceLock));
    host.join();
    device.join();

    EXPECT_EQ(result, expect);
}

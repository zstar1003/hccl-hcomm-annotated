/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>

#include "hccl_mem_defs.h"
#define private public
#define protected public
#include "global_mem_record.h"
#include "global_mem_manager.h"
#undef protected
#undef private

using namespace std;
using namespace hccl;

class GlobalMemMgrMulThreadTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "--GlobalMemMgrMulThreadTest SetUP--" << std::endl; }
    static void TearDownTestCase() { std::cout << "--GlobalMemMgrMulThreadTest TearDown--" << std::endl; }
    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }
};

HcclResult hrtGetDeviceRefreshStubSameId(s32* deviceLogicID)
{
    *deviceLogicID = 0;
    return HCCL_SUCCESS;
}

TEST_F(GlobalMemMgrMulThreadTest, Ut_GlobalMemMgr_GetInstance_When_DeviceSameID_Expect_Success)
{
    // 测试正常情况：多线程获取实例
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(hrtGetDeviceRefreshStubSameId));

    const int threadCount = 16;
    std::vector<std::thread> threads;
    std::vector<GlobalMemRegMgr*> instances(threadCount);

    // 多线程同时获取实例
    for (int i = 0; i < threadCount; i++) {
        threads.emplace_back([&, i]() {
            instances[i] = &GlobalMemRegMgr::GetInstance();
        });
    }

    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }

    // 验证所有线程获取的是同一个实例
    for (int i = 1; i < threadCount; i++) {
        EXPECT_EQ(instances[i], instances[0]);
    }

    GlobalMockObject::verify();
}

// 模拟不同线程获取不同的设备逻辑 ID
HcclResult hrtGetDeviceRefreshDifferentIdForTest(s32* deviceLogicID)
{
    // 为每个线程分配不同的设备逻辑 ID
    static std::atomic<int> threadId(0);
    int id = threadId++ % MAX_MODULE_DEVICE_NUM;
    *deviceLogicID = id;
    return HCCL_SUCCESS;
}

TEST_F(GlobalMemMgrMulThreadTest, Ut_GlobalMemMgr_GetInstance_When_DifferentDeviceID_Expect_Success)
{
    // 测试不同线程获取不同设备实例的场景
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(hrtGetDeviceRefreshDifferentIdForTest));

    const int threadCount = 16;
    std::vector<std::thread> threads;
    std::vector<GlobalMemRegMgr*> instances(threadCount);

    // 多线程同时获取实例
    for (int i = 0; i < threadCount; i++) {
        threads.emplace_back([&, i]() {
            instances[i] = &GlobalMemRegMgr::GetInstance();
        });
    }

    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }

    // 验证线程获取的实例数量（应该只有 2 个不同的实例，对应设备 0 和 1）
    std::unordered_set<GlobalMemRegMgr*> uniqueInstances;
    for (auto instance : instances) {
        uniqueInstances.insert(instance);
    }
    EXPECT_EQ(uniqueInstances.size(), threadCount);

    GlobalMockObject::verify();
}

TEST_F(GlobalMemMgrMulThreadTest, Ut_GlobalMemMgr_InicNic_When_DeviceSameID_Expect_Success)
{
    // 测试正常情况：多线程获取实例
    MOCKER(hrtGetDeviceRefresh).stubs().will(invoke(hrtGetDeviceRefreshStubSameId));
    MOCKER(HcclNetInit).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(HcclNetDeInit).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    const int threadCount = 16;
    std::vector<std::thread> threads;
    std::vector<HcclResult> ret(threadCount);

    // 多线程同时获取实例
    for (int i = 0; i < threadCount; i++) {
        threads.emplace_back([&, i]() {
            ret[i] = GlobalMemRegMgr::GetInstance().InitNic();
            ;
        });
    }

    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }

    // 验证所有线程获取的是同一个实例
    for (int i = 1; i < threadCount; i++) {
        EXPECT_EQ(ret[i], HCCL_SUCCESS);
    }

    GlobalMemRegMgr::GetInstance().DeInitNic();
    ;

    GlobalMockObject::verify();
}

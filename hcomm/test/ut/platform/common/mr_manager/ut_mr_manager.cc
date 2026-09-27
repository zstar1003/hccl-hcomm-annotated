/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>
#include <string>

#include "hccl/base.h"

#ifndef private
#define private public
#define protected public
#endif
#include "mr_manager.h"
#undef private
#undef protected

using namespace std;
using namespace hccl;

namespace {
// 打桩成员函数: 首参为 this 指针, 其余与原函数一致(参考 ut_heartbeat_lost_report.cc 实现方式)
HcclResult GetMrInfoStub(MrManager*, MrInfo& mrInfo, bool& isInfoNotFound)
{
    mrInfo.lkey = 42;
    mrInfo.addr = reinterpret_cast<void*>(0x1000);
    mrInfo.size = 1024;
    isInfoNotFound = false;
    return HCCL_SUCCESS;
}

HcclResult GetMrInfoStub2(MrManager*, MrInfo& mrInfo, bool& isInfoNotFound)
{
    mrInfo.lkey = 7;
    mrInfo.addr = reinterpret_cast<void*>(0x2000);
    mrInfo.size = 2048;
    isInfoNotFound = false;
    return HCCL_SUCCESS;
}

HcclResult RegTmpMrStub(MrManager*, void*, u64, u32& lkey)
{
    lkey = 99;
    return HCCL_SUCCESS;
}

HcclResult GetMrInfoFailStub(MrManager*, MrInfo&, bool& isInfoNotFound)
{
    isInfoNotFound = true;
    return HCCL_E_INTERNAL;
}

HcclResult GetMrInfoStub4(MrManager*, MrInfo& mrInfo, bool& isInfoNotFound)
{
    mrInfo.lkey = 5;
    mrInfo.addr = reinterpret_cast<void*>(0x4000);
    mrInfo.size = 8192;
    isInfoNotFound = false;
    return HCCL_SUCCESS;
}
} // namespace

class MrManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MrManagerTest SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "MrManagerTest TearDown" << std::endl; }
    virtual void SetUp()
    {
        std::cout << "MrManagerTest SetUP per test" << std::endl;
        manager = new MrManager();
    }
    virtual void TearDown()
    {
        std::cout << "MrManagerTest TearDown per test" << std::endl;
        delete manager;
        GlobalMockObject::verify();
    }
    MrManager* manager;
};

// GetKey: map 非空 + GetMrInfo 找到(成功路径, 509-512 行)
TEST_F(MrManagerTest, ut_GetKey_MapHasEntry_Expect_Success)
{
    void* addr = reinterpret_cast<void*>(0x1000);
    u64 size = 1024;
    u32 lkey = 0;
    MrMapKey key(reinterpret_cast<u64>(addr), size);
    MrInfo info(addr, size);
    info.lkey = 42;
    manager->regedMrMap_[key] = info;

    MOCKER_CPP(&MrManager::GetMrInfo).stubs().will(invoke(GetMrInfoStub));

    HcclResult ret = manager->GetKey(addr, size, lkey);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(lkey, 42u);
}

// GetKey: map 为空但 GetMrInfo 被 mock 为找到(TOCTOU 防御分支, 516-522 行)
TEST_F(MrManagerTest, ut_GetKey_MapEmptyAfterGetMrInfo_Expect_RegTmpMrPath)
{
    void* addr = reinterpret_cast<void*>(0x2000);
    u64 size = 2048;
    u32 lkey = 0;

    MOCKER_CPP(&MrManager::GetMrInfo).stubs().will(invoke(GetMrInfoStub2));
    MOCKER_CPP(&MrManager::RegTmpMr).stubs().will(invoke(RegTmpMrStub));

    HcclResult ret = manager->GetKey(addr, size, lkey);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(lkey, 99u);
}

// ReleaseKey: GetMrInfo 返回失败(536/542 行错误分支)
TEST_F(MrManagerTest, ut_ReleaseKey_GetMrInfoFail_Expect_Internal)
{
    void* addr = reinterpret_cast<void*>(0x3000);
    u64 size = 4096;
    MOCKER_CPP(&MrManager::GetMrInfo).stubs().will(invoke(GetMrInfoFailStub));

    HcclResult ret = manager->ReleaseKey(addr, size);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

// ReleaseKey: GetMrInfo 成功且 map 有记录(gloMemRef>0 时引用计数递减提前返回, 不触发真实DeReg)
TEST_F(MrManagerTest, ut_ReleaseKey_MapHasEntry_Expect_Success)
{
    void* addr = reinterpret_cast<void*>(0x4000);
    u64 size = 8192;
    MrMapKey key(reinterpret_cast<u64>(addr), size);
    MrInfo info(addr, size);
    info.lkey = 5;
    info.tmpMemRef = 1;
    info.gloMemRef = 1; // 全局引用未释放, 递减临时引用后应提前返回成功
    manager->regedMrMap_[key] = info;

    MOCKER_CPP(&MrManager::GetMrInfo).stubs().will(invoke(GetMrInfoStub4));

    HcclResult ret = manager->ReleaseKey(addr, size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(manager->regedMrMap_[key].tmpMemRef, 0);
}

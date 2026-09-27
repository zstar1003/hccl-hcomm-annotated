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
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>

#define private public
#include "notify/notify_aicpu_mgr.h"
#undef private

using namespace hccl;

class NotifyAicpuMgrTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override { GlobalMockObject::verify(); }
};

// 构造时预分配容量
TEST_F(NotifyAicpuMgrTest, Constructor_ReservesCapacity)
{
    NotifyAicpuMgr mgr;
    EXPECT_GE(mgr.notifys_.capacity(), hccl::HCCL_THREAD_NOTIFY_MAX_NUM - 1);
}

// ReserveNotifyCapacity 扩容
TEST_F(NotifyAicpuMgrTest, ReserveNotifyCapacity_Works)
{
    NotifyAicpuMgr mgr;
    mgr.ReserveNotifyCapacity(100);
    EXPECT_GE(mgr.notifys_.capacity(), 99u);
}

// NotifyFree 空指针 → 返回错误
TEST_F(NotifyAicpuMgrTest, NotifyFree_NullParam_ReturnsError)
{
    NotifyAicpuMgr mgr;
    HcclResult ret = mgr.NotifyFree(nullptr);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

// NotifyAlloc 空指针 → 返回错误
TEST_F(NotifyAicpuMgrTest, NotifyAlloc_NullParam_ReturnsError)
{
    NotifyAicpuMgr mgr;
    HcclResult ret = mgr.NotifyAlloc(nullptr);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

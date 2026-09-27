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
#include <stdio.h>

#include "hccl/base.h"
#include <hccl/hccl_types.h>

#include "notify_pool.h"
#include "sal.h"

using namespace std;
using namespace hccl;

class NotifyPoolTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--NotifyPoolTest SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--NotifyPoolTest TearDown--\033[0m" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }
};

TEST_F(NotifyPoolTest, ut_alloc_notify_ipc_ok)
{
    s32 ret = HCCL_SUCCESS;

    NotifyPool pool;
    ret = pool.Init(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::string tag = "test_signal_create";
    ret = pool.RegisterOp(tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<LocalIpcNotify> localNotify = nullptr;
    RemoteRankInfo info(0, 0, 0);
    ret = pool.Alloc(tag, info, localNotify, NotifyLoadType::DEVICE_NOTIFY);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = pool.UnregisterOp(tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = pool.Destroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(NotifyPoolTest, ut_alloc_notify_ipc_fail_tag_invalid)
{
    s32 ret = HCCL_SUCCESS;

    NotifyPool pool;
    ret = pool.Init(0);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::string tag = "test_signal_create";
    std::string tag1 = "test_signal_create_1";
    ret = pool.RegisterOp(tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = pool.RegisterOp(tag);
    EXPECT_EQ(ret, HCCL_E_PARA);

    std::shared_ptr<LocalIpcNotify> localNotify = nullptr;
    RemoteRankInfo info(0, 0, 0);
    ret = pool.Alloc(tag1, info, localNotify, NotifyLoadType::DEVICE_NOTIFY);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = pool.UnregisterOp(tag1);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = pool.UnregisterOp(tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = pool.Destroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(NotifyPoolTest, ut_alloc_notify_no_ipc_ok)
{
    s32 ret = HCCL_SUCCESS;

    NotifyPool pool;
    ret = pool.Init(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::string tag = "test_signal_create";
    ret = pool.RegisterOp(tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<LocalIpcNotify> localNotify = nullptr;
    RemoteRankInfo info(0, 0, 0);
    SalGetBareTgid(&info.remotePid);
    ret = pool.Alloc(tag, info, localNotify, NotifyLoadType::DEVICE_NOTIFY);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = pool.UnregisterOp(tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = pool.Destroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(NotifyPoolTest, ut_alloc_notify_no_ipc_fail_tag_invalid)
{
    s32 ret = HCCL_SUCCESS;

    NotifyPool pool;
    ret = pool.Init(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::string tag = "test_signal_create";
    std::string tag1 = "test_signal_create_1";
    ret = pool.RegisterOp(tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = pool.RegisterOp(tag);
    EXPECT_EQ(ret, HCCL_E_PARA);

    std::shared_ptr<LocalIpcNotify> localNotify = nullptr;
    RemoteRankInfo info(0, 0, 0);
    SalGetBareTgid(&info.remotePid);
    ret = pool.Alloc(tag1, info, localNotify, NotifyLoadType::DEVICE_NOTIFY);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = pool.UnregisterOp(tag1);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = pool.UnregisterOp(tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = pool.Destroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(NotifyPoolTest, ut_alloc_notify_aligned)
{
    s32 ret = HCCL_SUCCESS;

    NotifyPool pool;
    ret = pool.Init(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::string tag = "test_signal_create";
    ret = pool.RegisterOp(tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<LocalIpcNotify> localNotify1 = nullptr;
    RemoteRankInfo info1(0, 0, 0);
    ret = pool.Alloc(tag, info1, localNotify1, NotifyLoadType::DEVICE_NOTIFY);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<LocalIpcNotify> localNotify3 = nullptr;
    RemoteRankInfo info3(0, 0, 0);
    ret = pool.Alloc(tag, info3, localNotify3, NotifyLoadType::DEVICE_NOTIFY, 8);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<LocalIpcNotify> localNotify4 = nullptr;
    RemoteRankInfo info4(0, 0, 0);
    ret = pool.Alloc(tag, info4, localNotify4, NotifyLoadType::DEVICE_NOTIFY, 3);
    EXPECT_NE(ret, HCCL_SUCCESS);

    ret = pool.UnregisterOp(tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = pool.Destroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

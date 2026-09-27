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

#include "aiv_graph_executor_mgr.h"

class AivGraphExecutorMgrTest : public testing::Test {};

TEST_F(AivGraphExecutorMgrTest, GetInstance_ReturnsSameSingleton)
{
    AivGraphExecutorMgr& mgr1 = AivGraphExecutorMgr::GetInstance();
    AivGraphExecutorMgr& mgr2 = AivGraphExecutorMgr::GetInstance();
    EXPECT_EQ(&mgr1, &mgr2);
}

TEST_F(AivGraphExecutorMgrTest, GetAivGraphExecutor_NewLaunchIdx_CreatesNew)
{
    auto& mgr = AivGraphExecutorMgr::GetInstance();
    auto executor = mgr.GetAivGraphExecutor(0, 0);
    EXPECT_NE(executor, nullptr);
}

TEST_F(AivGraphExecutorMgrTest, GetAivGraphExecutor_SameRankAndLaunchIdx_ReturnsSame)
{
    auto& mgr = AivGraphExecutorMgr::GetInstance();
    auto exec1 = mgr.GetAivGraphExecutor(1, 1);
    auto exec2 = mgr.GetAivGraphExecutor(1, 1);
    EXPECT_NE(exec1, nullptr);
    EXPECT_EQ(exec1, exec2);
}

TEST_F(AivGraphExecutorMgrTest, GetAivGraphExecutor_SameRankDifferentLaunchIdx_ReturnsDifferent)
{
    auto& mgr = AivGraphExecutorMgr::GetInstance();
    auto exec1 = mgr.GetAivGraphExecutor(10, 10);
    auto exec2 = mgr.GetAivGraphExecutor(10, 20);
    EXPECT_NE(exec1, nullptr);
    EXPECT_NE(exec2, nullptr);
    EXPECT_NE(exec1, exec2);
}

TEST_F(AivGraphExecutorMgrTest, GetAivGraphExecutor_DifferentRankSameLaunchIdx_ReturnsDifferent)
{
    auto& mgr = AivGraphExecutorMgr::GetInstance();
    auto exec1 = mgr.GetAivGraphExecutor(10, 30);
    auto exec2 = mgr.GetAivGraphExecutor(20, 30);
    EXPECT_NE(exec1, nullptr);
    EXPECT_NE(exec2, nullptr);
    EXPECT_NE(exec1, exec2);
}

TEST_F(AivGraphExecutorMgrTest, GetAivGraphExecutor_MultipleLaunches_EachDistinct)
{
    auto& mgr = AivGraphExecutorMgr::GetInstance();
    auto exec0 = mgr.GetAivGraphExecutor(100, 100);
    auto exec1 = mgr.GetAivGraphExecutor(100, 200);
    auto exec2 = mgr.GetAivGraphExecutor(100, 300);
    EXPECT_NE(exec0, nullptr);
    EXPECT_NE(exec1, nullptr);
    EXPECT_NE(exec2, nullptr);
    EXPECT_NE(exec0, exec1);
    EXPECT_NE(exec0, exec2);
    EXPECT_NE(exec1, exec2);
}

TEST_F(AivGraphExecutorMgrTest, GetAivGraphExecutor_RepeatedRetrieval_Consistent)
{
    auto& mgr = AivGraphExecutorMgr::GetInstance();
    auto e1 = mgr.GetAivGraphExecutor(42, 42);
    auto e2 = mgr.GetAivGraphExecutor(42, 42);
    auto e3 = mgr.GetAivGraphExecutor(42, 42);
    EXPECT_EQ(e1, e2);
    EXPECT_EQ(e1, e3);
}

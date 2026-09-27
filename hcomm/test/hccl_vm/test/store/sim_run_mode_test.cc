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
#include "store_sim_run_mode.h"
#include "db_sim_runner_db.h"
#include "sim_models.h"

// 表为空时 ProbeCheckOnlyMode 返回 false。
TEST(RunModeTest, ProbeCheckOnlyMode_EmptyTable_False)
{
    RunnerDB::DeleteAll<sim::RunModeConfig>();
    EXPECT_FALSE(sim::ProbeCheckOnlyMode());
}

// 写入 mode=1 后 ProbeCheckOnlyMode 返回 true。
TEST(RunModeTest, ProbeCheckOnlyMode_CheckOnlyRow_True)
{
    RunnerDB::DeleteAll<sim::RunModeConfig>();
    sim::RunModeConfig cfg{};
    cfg.mode = 1;
    RunnerDB::Add<sim::RunModeConfig>(cfg);
    EXPECT_TRUE(sim::ProbeCheckOnlyMode());
}

// IsCheckOnlyMode 进程内 latch：第二次调用与第一次一致。
TEST(RunModeTest, IsCheckOnlyMode_Latches)
{
    bool first = sim::IsCheckOnlyMode();
    bool second = sim::IsCheckOnlyMode();
    EXPECT_EQ(first, second);
}

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// clean 模式设备侧集成：IsCheckOnlyMode() 每进程只 latch 一次，本二进制全程不 seed check-only(mode=1)，
// 借此验证 clean 模式下大块走真实独立分配（与仅校验模式同址复用对照）。

#include <gtest/gtest.h>

#include <sys/mman.h> // shm_unlink

#include "store_sim_comm_pool_policy.h"
#include "store_sim_device_memory_manager.h"
#include "store_sim_memory_manager.h"
#include "store_sim_run_mode.h"
#include "db_sim_runner_db.h"
#include "sim_models.h"

class CheckOnlyOffTest : public testing::Test {
protected:
    void SetUp() override
    {
        // 清进程内/磁盘上残留的 HcclCommPool。
        sim::MemoryManager::GetInstance().FreeMemByName(sim::CommPoolPolicy::kPoolName);
        shm_unlink(sim::CommPoolPolicy::kPoolName);
        // 清空 RunModeConfig，保证 ProbeCheckOnlyMode 为 false。本文件不写任何 check-only(mode=1) 行。
        RunnerDB::DeleteAll<sim::RunModeConfig>();
    }
};

TEST_F(CheckOnlyOffTest, NormalMode_BigBlocks_RealIndependentAlloc)
{
    EXPECT_FALSE(sim::IsCheckOnlyMode()); // 全程未 seed，首次 latch 须为 false。
    auto& mgr = sim::DeviceMemoryManager::GetInstance();
    const size_t big = 256ULL * 1024 * 1024;
    void* a = mgr.AllocPhyMem("clean_big_a", 0, big);
    void* b = mgr.AllocPhyMem("clean_big_b", 0, big);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    EXPECT_NE(a, b); // clean 模式各自真实独立分配（仅校验模式下两者同为池首址）。
    mgr.FreePhyMem("clean_big_a", 0);
    mgr.FreePhyMem("clean_big_b", 0);
}

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include <cstring>
#include <gtest/gtest.h>
#include <thread>

#include "store_sim_memory_manager.h"
#include "store_sim_shm_ops.h"

class MemoryManagerTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override
    {
        sim::MemoryManager::GetInstance().FreeMemByName("test_mem");
        sim::MemoryManager::GetInstance().FreeMemByName("test_mem_null");
    }
};

TEST_F(MemoryManagerTest, AllocMemByName_Normal_Success)
{
    void* ptr = sim::MemoryManager::GetInstance().AllocMemByName("test_mem", 1024);
    EXPECT_NE(ptr, nullptr);
}

TEST_F(MemoryManagerTest, AllocMemByName_NullName_Fail)
{
    void* ptr = sim::MemoryManager::GetInstance().AllocMemByName(nullptr, 1024);
    EXPECT_EQ(ptr, nullptr);
}

TEST_F(MemoryManagerTest, AllocMemByName_ZeroSize_Fail)
{
    void* ptr = sim::MemoryManager::GetInstance().AllocMemByName("test_mem_zero", 0);
    EXPECT_EQ(ptr, nullptr);
}

TEST_F(MemoryManagerTest, AllocMemByName_DuplicateName_Fail)
{
    void* ptr1 = sim::MemoryManager::GetInstance().AllocMemByName("test_mem_dup", 1024);
    EXPECT_NE(ptr1, nullptr);
    void* ptr2 = sim::MemoryManager::GetInstance().AllocMemByName("test_mem_dup", 1024);
    EXPECT_EQ(ptr2, nullptr);
    sim::MemoryManager::GetInstance().FreeMemByName("test_mem_dup");
}

TEST_F(MemoryManagerTest, FreeMemByName_Normal_Success)
{
    void* ptr = sim::MemoryManager::GetInstance().AllocMemByName("test_mem_free", 1024);
    EXPECT_NE(ptr, nullptr);
    sim::MemoryManager::GetInstance().FreeMemByName("test_mem_free");
}

TEST_F(MemoryManagerTest, FreeMemByName_NullName_DoNothing)
{
    EXPECT_NO_THROW(sim::MemoryManager::GetInstance().FreeMemByName(nullptr));
}

TEST_F(MemoryManagerTest, FreeMemByName_NotExist_DoNothing)
{
    EXPECT_NO_THROW(sim::MemoryManager::GetInstance().FreeMemByName("not_exist_mem"));
}

TEST_F(MemoryManagerTest, LockUnlockMemByName_Normal_Success)
{
    void* ptr = sim::MemoryManager::GetInstance().AllocMemByName("test_mem_lock", 1024);
    EXPECT_NE(ptr, nullptr);
    EXPECT_NO_THROW(sim::MemoryManager::GetInstance().LockMemByName("test_mem_lock"));
    EXPECT_NO_THROW(sim::MemoryManager::GetInstance().UnlockMemByName("test_mem_lock"));
    sim::MemoryManager::GetInstance().FreeMemByName("test_mem_lock");
}

TEST_F(MemoryManagerTest, LockUnlockMemByName_NullName_DoNothing)
{
    EXPECT_NO_THROW(sim::MemoryManager::GetInstance().LockMemByName(nullptr));
    EXPECT_NO_THROW(sim::MemoryManager::GetInstance().UnlockMemByName(nullptr));
}

TEST_F(MemoryManagerTest, LockUnlockMemByName_NotExist_DoNothing)
{
    EXPECT_NO_THROW(sim::MemoryManager::GetInstance().LockMemByName("not_exist"));
    EXPECT_NO_THROW(sim::MemoryManager::GetInstance().UnlockMemByName("not_exist"));
}

TEST_F(MemoryManagerTest, AcquireMemByName_NotExist_Fail)
{
    void* ptr = sim::MemoryManager::GetInstance().AcquireMemByName("not_exist_acquire");
    EXPECT_EQ(ptr, nullptr);
}

TEST_F(MemoryManagerTest, AcquireMemByName_NullName_Fail)
{
    void* ptr = sim::MemoryManager::GetInstance().AcquireMemByName(nullptr);
    EXPECT_EQ(ptr, nullptr);
}

TEST_F(MemoryManagerTest, AcquireMemByName_AfterAlloc_Success)
{
    void* ptr1 = sim::MemoryManager::GetInstance().AllocMemByName("test_acquire", 1024);
    EXPECT_NE(ptr1, nullptr);
    void* ptr2 = sim::MemoryManager::GetInstance().AcquireMemByName("test_acquire");
    EXPECT_NE(ptr2, nullptr);
    EXPECT_EQ(ptr1, ptr2);
    sim::MemoryManager::GetInstance().FreeMemByName("test_acquire");
}

TEST_F(MemoryManagerTest, ReleaseMemByName_Normal_Success)
{
    void* ptr = sim::MemoryManager::GetInstance().AllocMemByName("test_release", 1024);
    EXPECT_NE(ptr, nullptr);
    EXPECT_NO_THROW(sim::MemoryManager::GetInstance().ReleaseMemByName("test_release"));
}

TEST_F(MemoryManagerTest, ReleaseMemByName_NullName_DoNothing)
{
    EXPECT_NO_THROW(sim::MemoryManager::GetInstance().ReleaseMemByName(nullptr));
}

TEST_F(MemoryManagerTest, ReleaseMemByName_NotExist_DoNothing)
{
    EXPECT_NO_THROW(sim::MemoryManager::GetInstance().ReleaseMemByName("not_exist_release"));
}

TEST_F(MemoryManagerTest, GetInstance_Singleton_SameInstance)
{
    sim::MemoryManager& inst1 = sim::MemoryManager::GetInstance();
    sim::MemoryManager& inst2 = sim::MemoryManager::GetInstance();
    EXPECT_EQ(&inst1, &inst2);
}

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>

#include "store_sim_memory_manager.h"
#include "store_sim_shm_memory_common.h"

class ShmMemoryCommonTest : public testing::Test {
protected:
};

TEST_F(ShmMemoryCommonTest, AcquireDevPtrInNoHostProcess_NullPtr_ReturnsNull)
{
    sim::PhyMemBlock phyMem;
    memset(&phyMem, 0, sizeof(phyMem));
    void* result = sim::AcquireDevPtrInNoHostProcess(nullptr, phyMem);
    EXPECT_EQ(result, nullptr);
}

TEST_F(ShmMemoryCommonTest, AcquireDevPtrInNoHostProcess_InvalidPtr_ReturnsNull)
{
    sim::PhyMemBlock phyMem;
    memset(&phyMem, 0, sizeof(phyMem));
    void* badPtr = reinterpret_cast<void*>(static_cast<uintptr_t>(0xDEADBEEF));
    void* result = sim::AcquireDevPtrInNoHostProcess(badPtr, phyMem);
    EXPECT_EQ(result, nullptr);
}

TEST_F(ShmMemoryCommonTest, AcquireDevPtrInNoHostProcess_MaxAddr_ReturnsNull)
{
    sim::PhyMemBlock phyMem;
    memset(&phyMem, 0, sizeof(phyMem));
    void* maxPtr = reinterpret_cast<void*>(static_cast<uintptr_t>(UINTPTR_MAX));
    void* result = sim::AcquireDevPtrInNoHostProcess(maxPtr, phyMem);
    EXPECT_EQ(result, nullptr);
}

TEST_F(ShmMemoryCommonTest, AcquireDevPtrInNoHostProcess_ZeroAddr_ReturnsNull)
{
    sim::PhyMemBlock phyMem;
    memset(&phyMem, 0, sizeof(phyMem));
    void* zeroPtr = reinterpret_cast<void*>(static_cast<uintptr_t>(0));
    void* result = sim::AcquireDevPtrInNoHostProcess(zeroPtr, phyMem);
    EXPECT_EQ(result, nullptr);
}

TEST_F(ShmMemoryCommonTest, ReleaseInNoHostProcess_Normal_NoThrow)
{
    sim::PhyMemBlock phyMem;
    memset(phyMem.name, 0, sizeof(phyMem.name));
    EXPECT_NO_THROW(sim::ReleaseInNoHostProcess(phyMem));
}

TEST_F(ShmMemoryCommonTest, ReleaseInNoHostProcess_NamedMem_NoThrow)
{
    sim::PhyMemBlock phyMem;
    memset(&phyMem, 0, sizeof(phyMem));
    strncpy(phyMem.name, "test_mem_block", sizeof(phyMem.name));
    EXPECT_NO_THROW(sim::ReleaseInNoHostProcess(phyMem));
}

TEST_F(ShmMemoryCommonTest, ReleaseInNoHostProcess_EmptyName_NoThrow)
{
    sim::PhyMemBlock phyMem;
    memset(&phyMem, 0, sizeof(phyMem));
    EXPECT_NO_THROW(sim::ReleaseInNoHostProcess(phyMem));
}

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <gtest/gtest.h>
#include <sys/mman.h>

#include "store_sim_comm_memory_manager.h"
#include "store_sim_memory_manager.h"

class CommMemoryManagerTest : public testing::Test {
protected:
    void SetUp() override
    {
        shm_unlink("comm_test_alloc");
        shm_unlink("comm_test_acquire");
        shm_unlink("comm_test_write");
        shm_unlink("comm_test_write_empty");
        shm_unlink("test_alloc_null");
        shm_unlink("comm_test_read");
        shm_unlink("comm_test_read_null");
        shm_unlink("comm_test_read_zero");
        shm_unlink("comm_test_read_empty");
        shm_unlink("comm_test_multi");
        shm_unlink("comm_test_write_zero");
        shm_unlink("comm_test_write_ptr");
        shm_unlink("comm_test_acq");
        shm_unlink("comm_test_rel");
    }

    void TearDown() override
    {
        sim::CommunicationMemoryManager::GetInstance().ReleaseCommMem("comm_test_alloc");
        sim::CommunicationMemoryManager::GetInstance().ReleaseCommMem("comm_test_acquire");
        sim::CommunicationMemoryManager::GetInstance().ReleaseCommMem("comm_test_write");
        sim::CommunicationMemoryManager::GetInstance().ReleaseCommMem("comm_test_write_empty");
        sim::CommunicationMemoryManager::GetInstance().ReleaseCommMem("test_alloc_null");
    }
};

TEST_F(CommMemoryManagerTest, GetInstance_Singleton_SameInstance)
{
    sim::CommunicationMemoryManager& inst1 = sim::CommunicationMemoryManager::GetInstance();
    sim::CommunicationMemoryManager& inst2 = sim::CommunicationMemoryManager::GetInstance();
    EXPECT_EQ(&inst1, &inst2);
}

TEST_F(CommMemoryManagerTest, AllocCommMem_Normal_Success)
{
    void* ptr = sim::CommunicationMemoryManager::GetInstance().AllocCommMem("comm_test_alloc");
    EXPECT_NE(ptr, nullptr);
}

TEST_F(CommMemoryManagerTest, AllocCommMem_NullName_Fail)
{
    void* ptr = sim::CommunicationMemoryManager::GetInstance().AllocCommMem(nullptr);
    EXPECT_EQ(ptr, nullptr);
}

TEST_F(CommMemoryManagerTest, AcquireCommMem_Normal_Success)
{
    sim::CommunicationMemoryManager::GetInstance().AllocCommMem("comm_test_acq");
    void* ptr = sim::CommunicationMemoryManager::GetInstance().AcquireCommMem("comm_test_acq");
    EXPECT_NE(ptr, nullptr);
    sim::CommunicationMemoryManager::GetInstance().ReleaseCommMem("comm_test_acq");
}

TEST_F(CommMemoryManagerTest, AcquireCommMem_NotExist_Fail)
{
    void* ptr = sim::CommunicationMemoryManager::GetInstance().AcquireCommMem("not_exist_comm");
    EXPECT_EQ(ptr, nullptr);
}

TEST_F(CommMemoryManagerTest, AcquireCommMem_NullName_Fail)
{
    void* ptr = sim::CommunicationMemoryManager::GetInstance().AcquireCommMem(nullptr);
    EXPECT_EQ(ptr, nullptr);
}

TEST_F(CommMemoryManagerTest, ReleaseCommMem_Normal_Success)
{
    sim::CommunicationMemoryManager::GetInstance().AllocCommMem("comm_test_rel");
    int ret = sim::CommunicationMemoryManager::GetInstance().ReleaseCommMem("comm_test_rel");
    EXPECT_EQ(ret, 0);
}

TEST_F(CommMemoryManagerTest, ReleaseCommMem_NullName_Fail)
{
    int ret = sim::CommunicationMemoryManager::GetInstance().ReleaseCommMem(nullptr);
    EXPECT_EQ(ret, -1);
}

TEST_F(CommMemoryManagerTest, ReleaseCommMem_NotExist_Fail)
{
    int ret = sim::CommunicationMemoryManager::GetInstance().ReleaseCommMem("not_exist_release");
    EXPECT_EQ(ret, -1);
}

TEST_F(CommMemoryManagerTest, WriteCommMem_Normal_Success)
{
    sim::CommunicationMemoryManager::GetInstance().AllocCommMem("comm_test_write");
    const char* data = "test data";
    int ret = sim::CommunicationMemoryManager::GetInstance().WriteCommMem("comm_test_write", data, strlen(data));
    EXPECT_EQ(ret, 0);
}

TEST_F(CommMemoryManagerTest, WriteCommMem_NullName_Fail)
{
    const char* data = "test";
    int ret = sim::CommunicationMemoryManager::GetInstance().WriteCommMem(nullptr, data, strlen(data));
    EXPECT_EQ(ret, -1);
}

TEST_F(CommMemoryManagerTest, WriteCommMem_NullDataPtr_Fail)
{
    sim::CommunicationMemoryManager::GetInstance().AllocCommMem("comm_test_write_ptr");
    int ret = sim::CommunicationMemoryManager::GetInstance().WriteCommMem("comm_test_write_ptr", nullptr, 10);
    EXPECT_EQ(ret, -1);
    sim::CommunicationMemoryManager::GetInstance().ReleaseCommMem("comm_test_write_ptr");
}

TEST_F(CommMemoryManagerTest, WriteCommMem_ZeroSize_DoNothing)
{
    sim::CommunicationMemoryManager::GetInstance().AllocCommMem("comm_test_write_zero");
    const char* data = "test";
    int ret = sim::CommunicationMemoryManager::GetInstance().WriteCommMem("comm_test_write_zero", data, 0);
    EXPECT_EQ(ret, -1);
    sim::CommunicationMemoryManager::GetInstance().ReleaseCommMem("comm_test_write_zero");
}

TEST_F(CommMemoryManagerTest, WriteCommMem_NotExist_Fail)
{
    const char* data = "test";
    int ret = sim::CommunicationMemoryManager::GetInstance().WriteCommMem("not_exist_write", data, strlen(data));
    EXPECT_EQ(ret, -1);
}

TEST_F(CommMemoryManagerTest, ReadCommMem_Normal_Success)
{
    sim::CommunicationMemoryManager::GetInstance().AllocCommMem("comm_test_read");
    const char* writeData = "test data";
    sim::CommunicationMemoryManager::GetInstance().WriteCommMem("comm_test_read", writeData, strlen(writeData));

    char readBuf[64] = {0};
    int ret = sim::CommunicationMemoryManager::GetInstance().ReadCommMem("comm_test_read", readBuf, 64);
    EXPECT_GT(ret, 0);
    EXPECT_STREQ(readBuf, writeData);
    sim::CommunicationMemoryManager::GetInstance().ReleaseCommMem("comm_test_read");
}

TEST_F(CommMemoryManagerTest, ReadCommMem_NullName_Fail)
{
    char buf[64];
    int ret = sim::CommunicationMemoryManager::GetInstance().ReadCommMem(nullptr, buf, 64);
    EXPECT_EQ(ret, -1);
}

TEST_F(CommMemoryManagerTest, ReadCommMem_NullDataPtr_Fail)
{
    sim::CommunicationMemoryManager::GetInstance().AllocCommMem("comm_test_read_null");
    int ret = sim::CommunicationMemoryManager::GetInstance().ReadCommMem("comm_test_read_null", nullptr, 64);
    EXPECT_EQ(ret, -1);
    sim::CommunicationMemoryManager::GetInstance().ReleaseCommMem("comm_test_read_null");
}

TEST_F(CommMemoryManagerTest, ReadCommMem_ZeroSize_DoNothing)
{
    sim::CommunicationMemoryManager::GetInstance().AllocCommMem("comm_test_read_zero");
    char buf[64];
    int ret = sim::CommunicationMemoryManager::GetInstance().ReadCommMem("comm_test_read_zero", buf, 0);
    EXPECT_EQ(ret, -1);
    sim::CommunicationMemoryManager::GetInstance().ReleaseCommMem("comm_test_read_zero");
}

TEST_F(CommMemoryManagerTest, ReadCommMem_NotExist_Fail)
{
    char buf[64];
    int ret = sim::CommunicationMemoryManager::GetInstance().ReadCommMem("not_exist_read", buf, 64);
    EXPECT_EQ(ret, -1);
}

TEST_F(CommMemoryManagerTest, ReadCommMem_EmptyBuffer_ReturnsZero)
{
    sim::CommunicationMemoryManager::GetInstance().AllocCommMem("comm_test_read_empty");
    char buf[64];
    int ret = sim::CommunicationMemoryManager::GetInstance().ReadCommMem("comm_test_read_empty", buf, 64);
    EXPECT_EQ(ret, 0);
    sim::CommunicationMemoryManager::GetInstance().ReleaseCommMem("comm_test_read_empty");
}

TEST_F(CommMemoryManagerTest, WriteRead_MultipleTimes_Success)
{
    sim::CommunicationMemoryManager::GetInstance().AllocCommMem("comm_test_multi");

    const char* data1 = "first";
    int ret = sim::CommunicationMemoryManager::GetInstance().WriteCommMem("comm_test_multi", data1, strlen(data1));
    EXPECT_EQ(ret, 0);

    const char* data2 = "second";
    ret = sim::CommunicationMemoryManager::GetInstance().WriteCommMem("comm_test_multi", data2, strlen(data2));
    EXPECT_EQ(ret, 0);

    char buf[64] = {0};
    int readSize = sim::CommunicationMemoryManager::GetInstance().ReadCommMem("comm_test_multi", buf, 64);
    EXPECT_GT(readSize, 0);
    EXPECT_STREQ(buf, "firstsecond");

    sim::CommunicationMemoryManager::GetInstance().ReleaseCommMem("comm_test_multi");
}

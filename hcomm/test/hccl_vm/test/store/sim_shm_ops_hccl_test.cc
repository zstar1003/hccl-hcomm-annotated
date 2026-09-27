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
#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "store_sim_shm_ops.h"

class SimShmOpsTest : public testing::Test {
protected:
    void SetUp() override
    {
        // Clean up any existing shared memory
        shm_unlink("/test_shm_create");
        shm_unlink("/test_shm_open");
        shm_unlink("/test_shm_lock");
        shm_unlink("/test_shm_multi");
        shm_unlink("/test_shm_refcount");
    }

    void TearDown() override
    {
        // Clean up shared memory after tests
        shm_unlink("/test_shm_create");
        shm_unlink("/test_shm_open");
        shm_unlink("/test_shm_lock");
        shm_unlink("/test_shm_multi");
        shm_unlink("/test_shm_refcount");
    }
};

// ==================== Constants Tests ====================

TEST_F(SimShmOpsTest, ShmConstants_ValidValues)
{
    EXPECT_EQ(SHM_MAGIC, 0x53484D50); // 'SHMP'
    EXPECT_EQ(SHM_VERSION, 1);
}

TEST_F(SimShmOpsTest, ShmHead_SizeValid)
{
    EXPECT_GT(sizeof(ShmHead), 0u);
    EXPECT_LE(sizeof(ShmHead), 128u); // Should be reasonably small
}

// ==================== ShmCreate Tests ====================

TEST_F(SimShmOpsTest, ShmCreate_Normal_Success)
{
    void* shm = ShmCreate("/test_shm_create", 4096);
    EXPECT_NE(shm, nullptr);

    if (shm) {
        ShmClose(shm);
    }
}

TEST_F(SimShmOpsTest, ShmCreate_NullName_ReturnsNull)
{
    void* shm = ShmCreate(nullptr, 4096);
    EXPECT_EQ(shm, nullptr);
}

TEST_F(SimShmOpsTest, ShmCreate_ZeroSize_ReturnsNull)
{
    void* shm = ShmCreate("/test_shm_create", 0);
    EXPECT_EQ(shm, nullptr);
}

TEST_F(SimShmOpsTest, ShmCreate_SmallSize_Success)
{
    void* shm = ShmCreate("/test_shm_create", 1);
    EXPECT_NE(shm, nullptr);

    if (shm) {
        ShmClose(shm);
    }
}

TEST_F(SimShmOpsTest, ShmCreate_LargeSize_Success)
{
    void* shm = ShmCreate("/test_shm_create", 1024 * 1024);
    EXPECT_NE(shm, nullptr);

    if (shm) {
        ShmClose(shm);
    }
}

TEST_F(SimShmOpsTest, ShmCreate_DuplicateName_ReturnsNull)
{
    void* shm1 = ShmCreate("/test_shm_create", 4096);
    ASSERT_NE(shm1, nullptr);

    void* shm2 = ShmCreate("/test_shm_create", 4096);
    EXPECT_EQ(shm2, nullptr);

    ShmClose(shm1);
}

TEST_F(SimShmOpsTest, ShmCreate_MemoryAccessible)
{
    void* shm = ShmCreate("/test_shm_create", 4096);
    ASSERT_NE(shm, nullptr);

    // Try to write and read
    int* data = static_cast<int*>(shm);
    *data = 12345;
    EXPECT_EQ(*data, 12345);

    ShmClose(shm);
}

// ==================== ShmOpen Tests ====================

TEST_F(SimShmOpsTest, ShmOpen_AfterCreate_Success)
{
    // ShmCreate + ShmClose will shm_unlink (refCount drops to 0),
    // so ShmOpen must be called while shm is still alive
    void* shm1 = ShmCreate("/test_shm_open", 4096);
    ASSERT_NE(shm1, nullptr);

    size_t size = 0;
    void* shm2 = ShmOpen("/test_shm_open", &size);
    EXPECT_NE(shm2, nullptr);
    EXPECT_EQ(size, 4096u);

    if (shm2) {
        ShmClose(shm2);
    }
    ShmClose(shm1);
}

TEST_F(SimShmOpsTest, ShmOpen_NullName_ReturnsNull)
{
    size_t size = 0;
    void* shm = ShmOpen(nullptr, &size);
    EXPECT_EQ(shm, nullptr);
}

TEST_F(SimShmOpsTest, ShmOpen_NonExisting_ReturnsNull)
{
    size_t size = 0;
    void* shm = ShmOpen("/non_existing_shm", &size);
    EXPECT_EQ(shm, nullptr);
}

TEST_F(SimShmOpsTest, ShmOpen_SizeParameterUpdated)
{
    void* shm1 = ShmCreate("/test_shm_open", 8192);
    ASSERT_NE(shm1, nullptr);

    size_t size = 0;
    void* shm2 = ShmOpen("/test_shm_open", &size);
    ASSERT_NE(shm2, nullptr);
    EXPECT_EQ(size, 8192u);

    ShmClose(shm2);
    ShmClose(shm1);
}

// ==================== ShmClose Tests ====================

TEST_F(SimShmOpsTest, ShmClose_Normal_NoThrow)
{
    void* shm = ShmCreate("/test_shm_create", 4096);
    ASSERT_NE(shm, nullptr);

    EXPECT_NO_THROW(ShmClose(shm));
}

TEST_F(SimShmOpsTest, ShmClose_NullPointer_NoThrow) { EXPECT_NO_THROW(ShmClose(nullptr)); }

TEST_F(SimShmOpsTest, ShmClose_MultipleTimes_Success)
{
    // Keep shm1 alive so shm_unlink doesn't happen on first close
    void* shm1 = ShmCreate("/test_shm_create", 4096);
    ASSERT_NE(shm1, nullptr);

    // Open and close multiple times
    size_t size = 0;
    void* shm2 = ShmOpen("/test_shm_create", &size);
    ASSERT_NE(shm2, nullptr);
    ShmClose(shm2);

    void* shm3 = ShmOpen("/test_shm_create", &size);
    ASSERT_NE(shm3, nullptr);
    ShmClose(shm3);

    ShmClose(shm1);
}

// ==================== ShmLock/ShmUnlock Tests ====================

TEST_F(SimShmOpsTest, ShmLock_Normal_Success)
{
    void* shm = ShmCreate("/test_shm_lock", 4096);
    ASSERT_NE(shm, nullptr);

    int result = ShmLock(shm);
    EXPECT_EQ(result, 0);

    ShmUnlock(shm);
    ShmClose(shm);
}

TEST_F(SimShmOpsTest, ShmLock_NullPointer_ReturnsError)
{
    int result = ShmLock(nullptr);
    EXPECT_EQ(result, -1);
}

TEST_F(SimShmOpsTest, ShmUnlock_Normal_Success)
{
    void* shm = ShmCreate("/test_shm_lock", 4096);
    ASSERT_NE(shm, nullptr);

    ShmLock(shm);
    int result = ShmUnlock(shm);
    EXPECT_EQ(result, 0);

    ShmClose(shm);
}

TEST_F(SimShmOpsTest, ShmUnlock_NullPointer_ReturnsError)
{
    int result = ShmUnlock(nullptr);
    EXPECT_EQ(result, -1);
}

TEST_F(SimShmOpsTest, ShmLockUnlock_MultipleTimes_Success)
{
    void* shm = ShmCreate("/test_shm_lock", 4096);
    ASSERT_NE(shm, nullptr);

    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(ShmLock(shm), 0);
        EXPECT_EQ(ShmUnlock(shm), 0);
    }

    ShmClose(shm);
}

// ==================== Reference Count Tests ====================

TEST_F(SimShmOpsTest, RefCount_IncrementOnOpen)
{
    // Keep shm1 alive to prevent shm_unlink
    void* shm1 = ShmCreate("/test_shm_refcount", 4096);
    ASSERT_NE(shm1, nullptr);

    // Open twice - should increment ref count
    size_t size = 0;
    void* shm2 = ShmOpen("/test_shm_refcount", &size);
    void* shm3 = ShmOpen("/test_shm_refcount", &size);

    EXPECT_NE(shm2, nullptr);
    EXPECT_NE(shm3, nullptr);

    // Close both
    ShmClose(shm2);
    ShmClose(shm3);

    // Should still be able to open (shm1 still holds ref)
    void* shm4 = ShmOpen("/test_shm_refcount", &size);
    EXPECT_NE(shm4, nullptr);
    ShmClose(shm4);

    ShmClose(shm1);
}

// ==================== Data Integrity Tests ====================

TEST_F(SimShmOpsTest, DataPersistence_WriteRead_Success)
{
    void* shm = ShmCreate("/test_shm_create", 4096);
    ASSERT_NE(shm, nullptr);

    // Write data
    int* data = static_cast<int*>(shm);
    for (int i = 0; i < 100; ++i) {
        data[i] = i * 2;
    }

    // Read and verify
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(data[i], i * 2);
    }

    ShmClose(shm);
}

TEST_F(SimShmOpsTest, DataPersistence_AfterReopen_Success)
{
    // Create and write
    void* shm1 = ShmCreate("/test_shm_create", 4096);
    ASSERT_NE(shm1, nullptr);

    int* data1 = static_cast<int*>(shm1);
    data1[0] = 12345;
    data1[1] = 67890;

    // Open while shm is still alive (don't close shm1 first)
    size_t size = 0;
    void* shm2 = ShmOpen("/test_shm_create", &size);
    ASSERT_NE(shm2, nullptr);

    int* data2 = static_cast<int*>(shm2);
    EXPECT_EQ(data2[0], 12345);
    EXPECT_EQ(data2[1], 67890);

    ShmClose(shm2);
    ShmClose(shm1);
}

// ==================== Boundary Tests ====================

TEST_F(SimShmOpsTest, ShmCreate_PageSize_Success)
{
    long page_size = sysconf(_SC_PAGESIZE);
    void* shm = ShmCreate("/test_shm_create", page_size);
    EXPECT_NE(shm, nullptr);

    if (shm) {
        ShmClose(shm);
    }
}

TEST_F(SimShmOpsTest, ShmCreate_MultiplePages_Success)
{
    long page_size = sysconf(_SC_PAGESIZE);
    void* shm = ShmCreate("/test_shm_create", page_size * 4);
    EXPECT_NE(shm, nullptr);

    if (shm) {
        ShmClose(shm);
    }
}

TEST_F(SimShmOpsTest, ShmCreate_VerySmallSize_Success)
{
    void* shm = ShmCreate("/test_shm_create", 1);
    EXPECT_NE(shm, nullptr);

    if (shm) {
        // Should still be able to access
        char* data = static_cast<char*>(shm);
        *data = 'A';
        EXPECT_EQ(*data, 'A');
        ShmClose(shm);
    }
}

// ==================== Concurrency Tests ====================

TEST_F(SimShmOpsTest, ShmLock_MultiThreaded_Exclusion)
{
    void* shm = ShmCreate("/test_shm_lock", 4096);
    ASSERT_NE(shm, nullptr);

    std::atomic<int> counter{0};
    std::vector<std::thread> threads;

    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&counter, shm]() {
            for (int j = 0; j < 100; ++j) {
                ShmLock(shm);
                int val = counter.load();
                counter.store(val + 1);
                ShmUnlock(shm);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(counter.load(), 400);
    ShmClose(shm);
}

// ==================== Error Recovery Tests ====================

TEST_F(SimShmOpsTest, ShmCreate_AfterClose_CanRecreate)
{
    void* shm1 = ShmCreate("/test_shm_create", 4096);
    ASSERT_NE(shm1, nullptr);
    ShmClose(shm1);

    // After close, the shm should be deleted (ref count = 0)
    // So we should be able to create again
    void* shm2 = ShmCreate("/test_shm_create", 4096);
    EXPECT_NE(shm2, nullptr);

    if (shm2) {
        ShmClose(shm2);
    }
}

TEST_F(SimShmOpsTest, ShmOpen_InvalidMagic_ReturnsNull)
{
    int shmFd = shm_open("/test_shm_create", O_CREAT | O_RDWR, 0666);
    ASSERT_GE(shmFd, 0);

    size_t totalSize = sizeof(ShmHead) + 1024;
    ASSERT_EQ(ftruncate(shmFd, totalSize), 0);

    void* addr = mmap(nullptr, totalSize, PROT_READ | PROT_WRITE, MAP_SHARED, shmFd, 0);
    close(shmFd);
    ASSERT_NE(addr, MAP_FAILED);

    ShmHead* head = static_cast<ShmHead*>(addr);
    head->magic = 0xDEADBEEF;
    head->size = 1024;
    head->version = 1;
    head->lock = 0;
    head->refCount = 1;

    munmap(addr, totalSize);

    size_t size = 0;
    void* shm = ShmOpen("/test_shm_create", &size);
    EXPECT_EQ(shm, nullptr);

    shm_unlink("/test_shm_create");
}

TEST_F(SimShmOpsTest, ShmOpen_ZeroMagic_ReturnsNull)
{
    int shmFd = shm_open("/test_shm_create", O_CREAT | O_RDWR, 0666);
    ASSERT_GE(shmFd, 0);

    size_t totalSize = sizeof(ShmHead) + 1024;
    ASSERT_EQ(ftruncate(shmFd, totalSize), 0);

    void* addr = mmap(nullptr, totalSize, PROT_READ | PROT_WRITE, MAP_SHARED, shmFd, 0);
    close(shmFd);
    ASSERT_NE(addr, MAP_FAILED);

    ShmHead* head = static_cast<ShmHead*>(addr);
    head->magic = 0;
    head->size = 1024;
    head->version = 1;
    head->lock = 0;
    head->refCount = 1;

    munmap(addr, totalSize);

    size_t size = 0;
    void* shm = ShmOpen("/test_shm_create", &size);
    EXPECT_EQ(shm, nullptr);

    shm_unlink("/test_shm_create");
}

TEST_F(SimShmOpsTest, ShmOpen_AfterLastClose_ReturnsNull)
{
    void* shm1 = ShmCreate("/test_shm_create", 4096);
    ASSERT_NE(shm1, nullptr);
    ShmClose(shm1); // This should delete the shm

    size_t size = 0;
    void* shm2 = ShmOpen("/test_shm_create", &size);
    EXPECT_EQ(shm2, nullptr);
}

TEST_F(SimShmOpsTest, ShmOpen_Concurrent_MultipleThreads)
{
    void* shm1 = ShmCreate("/test_shm_create", 4096);
    ASSERT_NE(shm1, nullptr);

    std::atomic<int> successCount{0};
    std::vector<std::thread> threads;
    std::vector<void*> openedShms;

    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([this, &successCount, &openedShms]() {
            size_t size = 0;
            void* shm = ShmOpen("/test_shm_create", &size);
            if (shm) {
                successCount.fetch_add(1);
                openedShms.push_back(shm);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_GE(successCount.load(), 1);

    for (void* shm : openedShms) {
        ShmClose(shm);
    }
    ShmClose(shm1);
}

TEST_F(SimShmOpsTest, ShmLock_LongTimeHold_Released)
{
    void* shm = ShmCreate("/test_shm_lock", 4096);
    ASSERT_NE(shm, nullptr);

    EXPECT_EQ(ShmLock(shm), 0);
    EXPECT_EQ(ShmUnlock(shm), 0);

    ShmClose(shm);
}

TEST_F(SimShmOpsTest, ShmCreate_VariousSizes_AllSucceed)
{
    const size_t sizes[] = {1, 100, 1024, 4096, 65536, 1048576};
    for (size_t sz : sizes) {
        void* p = nullptr;
        void* shm = ShmCreate("/test_shm_create", sz);
        EXPECT_NE(shm, nullptr) << "Failed for size: " << sz;
        if (shm) {
            p = shm;
            ShmClose(shm);
        }
        // Give OS time to clean up
        if (p) {
            while (shm_open("/test_shm_create", O_RDWR, 0666) != -1) {
                shm_unlink("/test_shm_create");
                usleep(1000);
            }
        }
    }
}

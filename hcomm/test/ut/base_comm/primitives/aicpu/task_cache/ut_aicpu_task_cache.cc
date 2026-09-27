/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

#ifndef private
#define private public
#define protected public
#endif

#include "aicpu_task_cache.h"

#undef private
#undef protected

using namespace hcomm;

namespace {
constexpr uint64_t TEST_MAX_CACHE_BYTES = 1024;
}

// ===================== Constructor / Destructor Tests =====================

TEST(AicpuTaskCacheTest, Constructor_DefaultCapacity)
{
    AicpuTaskCache cache;
    EXPECT_EQ(cache.maxCacheBytes_, AicpuTaskCache::AICPU_TASK_CACHE_CAPACITY);
    EXPECT_EQ(cache.cacheBytes_, 0U);
    EXPECT_EQ(cache.cacheHashMap_.size(), 0U);
    EXPECT_FALSE(cache.cacheHitRunInfoPrinted_);
    EXPECT_FALSE(cache.cacheFullRunInfoPrinted_);
}

TEST(AicpuTaskCacheTest, Constructor_CustomCapacity)
{
    AicpuTaskCache cache(TEST_MAX_CACHE_BYTES);
    EXPECT_EQ(cache.maxCacheBytes_, TEST_MAX_CACHE_BYTES);
    EXPECT_EQ(cache.cacheBytes_, 0U);
}

TEST(AicpuTaskCacheTest, Destructor_Empty_NoCrash)
{
    {
        AicpuTaskCache cache(TEST_MAX_CACHE_BYTES);
    }
    SUCCEED();
}

TEST(AicpuTaskCacheTest, Destructor_WithEntries_DeletesAll)
{
    AicpuTaskCacheEntry* entry1 = nullptr;
    AicpuTaskCacheEntry* entry2 = nullptr;
    {
        AicpuTaskCache cache(TEST_MAX_CACHE_BYTES);
        ASSERT_EQ(cache.AddEntry("tag1", &entry1), HCCL_SUCCESS);
        ASSERT_EQ(cache.AddEntry("tag2", &entry2), HCCL_SUCCESS);
        EXPECT_EQ(cache.cacheHashMap_.size(), 2U);
    }
    SUCCEED();
}

// ===================== FindEntry Tests =====================

TEST(AicpuTaskCacheTest, FindEntry_NotFound)
{
    AicpuTaskCache cache(TEST_MAX_CACHE_BYTES);
    AicpuTaskCacheEntry* entryPtr = nullptr;
    EXPECT_EQ(cache.FindEntry("nonexistent", &entryPtr), HCCL_SUCCESS);
    EXPECT_EQ(entryPtr, nullptr);
}

TEST(AicpuTaskCacheTest, FindEntry_NullParam)
{
    AicpuTaskCache cache(TEST_MAX_CACHE_BYTES);
    EXPECT_EQ(cache.FindEntry("tag", nullptr), HCCL_E_PTR);
}

TEST(AicpuTaskCacheTest, FindEntry_FoundAfterAdd)
{
    AicpuTaskCache cache(TEST_MAX_CACHE_BYTES);
    AicpuTaskCacheEntry* addedEntry = nullptr;
    ASSERT_EQ(cache.AddEntry("tag1", &addedEntry), HCCL_SUCCESS);
    ASSERT_NE(addedEntry, nullptr);

    AicpuTaskCacheEntry* foundEntry = nullptr;
    EXPECT_EQ(cache.FindEntry("tag1", &foundEntry), HCCL_SUCCESS);
    EXPECT_EQ(foundEntry, addedEntry);
    EXPECT_TRUE(cache.cacheHitRunInfoPrinted_);
}

TEST(AicpuTaskCacheTest, FindEntry_ConcurrentHits_MarksProcessRunInfoOnce)
{
    constexpr uint32_t THREAD_COUNT = 16;
    AicpuTaskCache cache(TEST_MAX_CACHE_BYTES);
    AicpuTaskCacheEntry* addedEntry = nullptr;
    ASSERT_EQ(cache.AddEntry("tag1", &addedEntry), HCCL_SUCCESS);

    std::vector<AicpuTaskCacheEntry*> foundEntries(THREAD_COUNT, nullptr);
    std::vector<HcclResult> results(THREAD_COUNT, HCCL_E_INTERNAL);
    std::vector<std::thread> threads;
    for (uint32_t index = 0; index < THREAD_COUNT; ++index) {
        threads.emplace_back([&cache, &foundEntries, &results, index]() {
            results[index] = cache.FindEntry("tag1", &foundEntries[index]);
        });
    }
    for (std::thread& thread : threads) {
        thread.join();
    }

    for (uint32_t index = 0; index < THREAD_COUNT; ++index) {
        EXPECT_EQ(results[index], HCCL_SUCCESS);
        EXPECT_EQ(foundEntries[index], addedEntry);
    }
    EXPECT_TRUE(cache.cacheHitRunInfoPrinted_);
}

// ===================== AddEntry Tests =====================

TEST(AicpuTaskCacheTest, AddEntry_Success)
{
    AicpuTaskCache cache(TEST_MAX_CACHE_BYTES);
    AicpuTaskCacheEntry* entryPtr = nullptr;
    EXPECT_EQ(cache.AddEntry("tag1", &entryPtr), HCCL_SUCCESS);
    EXPECT_NE(entryPtr, nullptr);
    EXPECT_EQ(cache.cacheHashMap_.size(), 1U);
}

TEST(AicpuTaskCacheTest, AddEntry_NullParam)
{
    AicpuTaskCache cache(TEST_MAX_CACHE_BYTES);
    EXPECT_EQ(cache.AddEntry("tag1", nullptr), HCCL_E_PTR);
}

TEST(AicpuTaskCacheTest, AddEntry_EntryPtrNotNull)
{
    AicpuTaskCache cache(TEST_MAX_CACHE_BYTES);
    AicpuTaskCacheEntry dummy;
    AicpuTaskCacheEntry* entryPtr = &dummy;
    EXPECT_EQ(cache.AddEntry("tag1", &entryPtr), HCCL_E_INTERNAL);
}

TEST(AicpuTaskCacheTest, AddEntry_CacheFull_ReturnsNullptr)
{
    AicpuTaskCache cache(0);
    AicpuTaskCacheEntry* entryPtr = nullptr;
    EXPECT_EQ(cache.AddEntry("tag1", &entryPtr), HCCL_SUCCESS);
    EXPECT_EQ(entryPtr, nullptr);
    EXPECT_EQ(cache.cacheHashMap_.size(), 0U);
    EXPECT_TRUE(cache.cacheFullRunInfoPrinted_);
}

TEST(AicpuTaskCacheTest, AddEntry_DuplicateTag)
{
    AicpuTaskCache cache(TEST_MAX_CACHE_BYTES);
    AicpuTaskCacheEntry* entry1 = nullptr;
    ASSERT_EQ(cache.AddEntry("tag1", &entry1), HCCL_SUCCESS);

    AicpuTaskCacheEntry* entry2 = nullptr;
    EXPECT_EQ(cache.AddEntry("tag1", &entry2), HCCL_E_INTERNAL);
    EXPECT_EQ(cache.cacheHashMap_.size(), 1U);
}

TEST(AicpuTaskCacheTest, AddEntry_MultipleEntries)
{
    AicpuTaskCache cache(TEST_MAX_CACHE_BYTES);
    AicpuTaskCacheEntry* entry1 = nullptr;
    AicpuTaskCacheEntry* entry2 = nullptr;
    AicpuTaskCacheEntry* entry3 = nullptr;
    ASSERT_EQ(cache.AddEntry("tag1", &entry1), HCCL_SUCCESS);
    ASSERT_EQ(cache.AddEntry("tag2", &entry2), HCCL_SUCCESS);
    ASSERT_EQ(cache.AddEntry("tag3", &entry3), HCCL_SUCCESS);
    EXPECT_EQ(cache.cacheHashMap_.size(), 3U);
    EXPECT_NE(entry1, entry2);
    EXPECT_NE(entry2, entry3);
}

// ===================== IncCacheBytes Tests =====================

TEST(AicpuTaskCacheTest, IncCacheBytes_IncrementsCorrectly)
{
    AicpuTaskCache cache(TEST_MAX_CACHE_BYTES);
    std::string tag = "tag1";
    EXPECT_EQ(cache.IncCacheBytes(tag.c_str(), 100), HCCL_SUCCESS);
    uint64_t expected = 100 + tag.size() + sizeof(AicpuTaskCacheEntry*);
    EXPECT_EQ(cache.cacheBytes_, expected);
}

TEST(AicpuTaskCacheTest, IncCacheBytes_MultipleCalls)
{
    AicpuTaskCache cache(TEST_MAX_CACHE_BYTES);
    std::string tag = "tag";
    ASSERT_EQ(cache.IncCacheBytes(tag.c_str(), 50), HCCL_SUCCESS);
    ASSERT_EQ(cache.IncCacheBytes(tag.c_str(), 50), HCCL_SUCCESS);
    uint64_t expected = 2 * (50 + tag.size() + sizeof(AicpuTaskCacheEntry*));
    EXPECT_EQ(cache.cacheBytes_, expected);
}

TEST(AicpuTaskCacheTest, IncCacheBytes_ZeroBytes)
{
    AicpuTaskCache cache(TEST_MAX_CACHE_BYTES);
    std::string tag = "tag1";
    EXPECT_EQ(cache.IncCacheBytes(tag.c_str(), 0), HCCL_SUCCESS);
    uint64_t expected = tag.size() + sizeof(AicpuTaskCacheEntry*);
    EXPECT_EQ(cache.cacheBytes_, expected);
}

// ===================== ClearEntry Tests =====================

TEST(AicpuTaskCacheTest, ClearEntry_ExistingEntry)
{
    AicpuTaskCache cache(TEST_MAX_CACHE_BYTES);
    AicpuTaskCacheEntry* entryPtr = nullptr;
    ASSERT_EQ(cache.AddEntry("tag1", &entryPtr), HCCL_SUCCESS);
    ASSERT_NE(entryPtr, nullptr);

    EXPECT_EQ(cache.ClearEntry("tag1"), HCCL_SUCCESS);
    EXPECT_EQ(cache.cacheHashMap_.size(), 0U);

    AicpuTaskCacheEntry* foundEntry = nullptr;
    EXPECT_EQ(cache.FindEntry("tag1", &foundEntry), HCCL_SUCCESS);
    EXPECT_EQ(foundEntry, nullptr);
}

TEST(AicpuTaskCacheTest, ClearEntry_NonExistentEntry)
{
    AicpuTaskCache cache(TEST_MAX_CACHE_BYTES);
    EXPECT_EQ(cache.ClearEntry("nonexistent"), HCCL_SUCCESS);
    EXPECT_EQ(cache.cacheHashMap_.size(), 0U);
}

TEST(AicpuTaskCacheTest, ClearEntry_UpdatesCacheBytes)
{
    AicpuTaskCache cache(TEST_MAX_CACHE_BYTES);
    std::string tag = "tag1";
    AicpuTaskCacheEntry* entryPtr = nullptr;
    ASSERT_EQ(cache.AddEntry(tag.c_str(), &entryPtr), HCCL_SUCCESS);

    ASSERT_EQ(cache.IncCacheBytes(tag.c_str(), 200), HCCL_SUCCESS);
    uint64_t bytesBefore = cache.cacheBytes_;
    EXPECT_GT(bytesBefore, 0U);

    // Save entryBytes before ClearEntry deletes the entry
    uint64_t entryBytes = entryPtr->GetEntryBytes();
    // ClearEntry subtracts entryBytes + tag.size() + sizeof(ptr) from cacheBytes_
    EXPECT_EQ(cache.ClearEntry(tag.c_str()), HCCL_SUCCESS);
    uint64_t expectedClearBytes = entryBytes + tag.size() + sizeof(AicpuTaskCacheEntry*);
    uint64_t expectedBytes = (bytesBefore > expectedClearBytes) ? (bytesBefore - expectedClearBytes) : 0;
    EXPECT_EQ(cache.cacheBytes_, expectedBytes);
}

TEST(AicpuTaskCacheTest, ClearEntry_OneOfMultiple)
{
    AicpuTaskCache cache(TEST_MAX_CACHE_BYTES);
    AicpuTaskCacheEntry* entry1 = nullptr;
    AicpuTaskCacheEntry* entry2 = nullptr;
    ASSERT_EQ(cache.AddEntry("tag1", &entry1), HCCL_SUCCESS);
    ASSERT_EQ(cache.AddEntry("tag2", &entry2), HCCL_SUCCESS);

    EXPECT_EQ(cache.ClearEntry("tag1"), HCCL_SUCCESS);
    EXPECT_EQ(cache.cacheHashMap_.size(), 1U);

    AicpuTaskCacheEntry* foundEntry = nullptr;
    EXPECT_EQ(cache.FindEntry("tag2", &foundEntry), HCCL_SUCCESS);
    EXPECT_EQ(foundEntry, entry2);
}

// ===================== Combined Workflow Tests =====================

TEST(AicpuTaskCacheTest, Workflow_AddFindIncClearFind)
{
    AicpuTaskCache cache(TEST_MAX_CACHE_BYTES);
    std::string tag = "workflow_tag";

    // Add
    AicpuTaskCacheEntry* addedEntry = nullptr;
    EXPECT_EQ(cache.AddEntry(tag.c_str(), &addedEntry), HCCL_SUCCESS);
    ASSERT_NE(addedEntry, nullptr);

    // Find
    AicpuTaskCacheEntry* foundEntry = nullptr;
    EXPECT_EQ(cache.FindEntry(tag.c_str(), &foundEntry), HCCL_SUCCESS);
    EXPECT_EQ(foundEntry, addedEntry);

    // IncCacheBytes
    EXPECT_EQ(cache.IncCacheBytes(tag.c_str(), 500), HCCL_SUCCESS);
    EXPECT_GT(cache.cacheBytes_, 0U);
    uint64_t bytesBeforeClear = cache.cacheBytes_;

    // Clear (cacheBytes_ reduced by entry's GetEntryBytes + tag.size + sizeof(ptr))
    uint64_t entryBytes = addedEntry->GetEntryBytes();
    EXPECT_EQ(cache.ClearEntry(tag.c_str()), HCCL_SUCCESS);
    EXPECT_EQ(cache.cacheHashMap_.size(), 0U);
    uint64_t expectedClearBytes = entryBytes + tag.size() + sizeof(AicpuTaskCacheEntry*);
    uint64_t expectedBytes = (bytesBeforeClear > expectedClearBytes) ? (bytesBeforeClear - expectedClearBytes) : 0;
    EXPECT_EQ(cache.cacheBytes_, expectedBytes);

    // Find after clear
    AicpuTaskCacheEntry* notFoundEntry = nullptr;
    EXPECT_EQ(cache.FindEntry(tag.c_str(), &notFoundEntry), HCCL_SUCCESS);
    EXPECT_EQ(notFoundEntry, nullptr);
}

TEST(AicpuTaskCacheTest, Workflow_CacheFullAfterInc_PreventsNewAdd)
{
    AicpuTaskCache cache(TEST_MAX_CACHE_BYTES);
    std::string tag1 = "tag1";
    AicpuTaskCacheEntry* entry1 = nullptr;
    ASSERT_EQ(cache.AddEntry(tag1.c_str(), &entry1), HCCL_SUCCESS);

    // Fill cache to capacity
    ASSERT_EQ(cache.IncCacheBytes(tag1.c_str(), TEST_MAX_CACHE_BYTES), HCCL_SUCCESS);
    EXPECT_GE(cache.cacheBytes_, TEST_MAX_CACHE_BYTES);

    // Next AddEntry should return nullptr (cache full)
    AicpuTaskCacheEntry* entry2 = nullptr;
    EXPECT_EQ(cache.AddEntry("tag2", &entry2), HCCL_SUCCESS);
    EXPECT_EQ(entry2, nullptr);
    EXPECT_EQ(cache.cacheHashMap_.size(), 1U);
}

TEST(AicpuTaskCacheTest, Workflow_ClearThenAddAgain)
{
    AicpuTaskCache cache(TEST_MAX_CACHE_BYTES);
    std::string tag = "tag1";

    AicpuTaskCacheEntry* entry1 = nullptr;
    ASSERT_EQ(cache.AddEntry(tag.c_str(), &entry1), HCCL_SUCCESS);

    ASSERT_EQ(cache.ClearEntry(tag.c_str()), HCCL_SUCCESS);

    // Add same tag again after clear
    AicpuTaskCacheEntry* entry2 = nullptr;
    EXPECT_EQ(cache.AddEntry(tag.c_str(), &entry2), HCCL_SUCCESS);
    EXPECT_NE(entry2, nullptr);
    EXPECT_EQ(cache.cacheHashMap_.size(), 1U);

    // Verify the new entry is usable
    AicpuTaskCacheEntry* foundEntry = nullptr;
    EXPECT_EQ(cache.FindEntry(tag.c_str(), &foundEntry), HCCL_SUCCESS);
    EXPECT_EQ(foundEntry, entry2);
}

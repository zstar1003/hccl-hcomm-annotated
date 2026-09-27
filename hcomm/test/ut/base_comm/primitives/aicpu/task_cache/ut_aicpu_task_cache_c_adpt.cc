/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#ifndef private
#define private public
#define protected public
#endif

#include "dtype_common.h"
#include "hcomm_primitives.h"
#include "aicpu_task_cache_manager.h"
#include "aicpu_task_cache_entry.h"
#include "ub_conn_lite.h"
#include "ub_transport_lite_impl.h"
#include "rtsq_a5.h"
#include "aicpu_ts_thread.h"
#include "sqe_v82.h"
#include "udma_data_struct.h"
#include "ub_jetty_lite.h"
#include "ip_address.h"

#undef private
#undef protected

using namespace hcomm;
using Hccl::Rt91095StarsSqeHeader;
using Hccl::Rt91095StarsSqeType;

namespace {
DevType g_testDeviceType = DevType::DEV_TYPE_950;
}

extern "C" HcclResult hrtGetDeviceType(DevType& devType)
{
    devType = g_testDeviceType;
    return HCCL_SUCCESS;
}

namespace {
constexpr uint32_t TEST_SQ_DEPTH = 128;
constexpr uint64_t TEST_BASE_ADDR_0 = 0x10000ULL;
constexpr uint64_t TEST_MEM_SIZE_0 = 0x1000ULL;
constexpr uint64_t TEST_BASE_ADDR_1 = 0x20000ULL;
constexpr uint64_t TEST_MEM_SIZE_1 = 0x2000ULL;

std::vector<uint8_t> MakeSqeArray(Rt91095StarsSqeType sqeType)
{
    std::vector<uint8_t> sqeArray(Hccl::AC_SQE_SIZE, 0);
    Rt91095StarsSqeHeader* header = reinterpret_cast<Rt91095StarsSqeHeader*>(sqeArray.data());
    header->type = static_cast<uint8_t>(sqeType);
    return sqeArray;
}
} // namespace

class HcommAicpuTsTaskCacheCAdptTest : public testing::Test {
protected:
    std::unique_ptr<Hccl::RtsqA5> rtsq_;
    Hccl::RtsqA5* rtsqPtr_ = nullptr;
    std::unique_ptr<hccl::AicpuTsThread> aicpuTsThread_;
    std::unique_ptr<Hccl::UbConnLite> ubConnLite_;
    std::unique_ptr<Hccl::UbTransportLiteImpl> ubTransport_;
    std::vector<char> emptyUniqueId_;
    std::string savedCacheTag_;
    bool savedIsHit_ = false;
    AicpuTaskCacheEntry* savedCacheEntryPtr_ = nullptr;
    DevType savedDeviceType_ = DevType::DEV_TYPE_950;
    uint64_t savedCacheBytes_ = 0;
    bool savedCacheHitRunInfoPrinted_ = false;
    bool savedCacheFullRunInfoPrinted_ = false;

    void SetUp() override
    {
        rtsq_ = std::make_unique<Hccl::RtsqA5>(0, 0, 0);
        rtsqPtr_ = rtsq_.get();
        aicpuTsThread_ = std::make_unique<hccl::AicpuTsThread>(std::string("ut_c_adpt_thread"));
        ubConnLite_ = std::make_unique<Hccl::UbConnLite>(
            Hccl::UbJettyLiteId(0, 0, 0), Hccl::UbJettyLiteAttr(0, 0, TEST_SQ_DEPTH, 0, true), Hccl::Eid{});
        ubTransport_ = std::make_unique<Hccl::UbTransportLiteImpl>(emptyUniqueId_);

        savedCacheTag_ = AicpuTaskCacheManager::cacheTag;
        savedIsHit_ = AicpuTaskCacheManager::isHit;
        savedCacheEntryPtr_ = AicpuTaskCacheManager::cacheEntryPtr;
        savedDeviceType_ = g_testDeviceType;
        savedCacheBytes_ = AicpuTaskCacheManager::aicpuTaskCache.cacheBytes_;
        savedCacheHitRunInfoPrinted_ = AicpuTaskCacheManager::aicpuTaskCache.cacheHitRunInfoPrinted_;
        savedCacheFullRunInfoPrinted_ = AicpuTaskCacheManager::aicpuTaskCache.cacheFullRunInfoPrinted_;
        g_testDeviceType = DevType::DEV_TYPE_950;
    }

    void TearDown() override
    {
        AicpuTaskCacheManager::aicpuTaskCache.ClearEntry("c_adpt_tag_1");
        AicpuTaskCacheManager::aicpuTaskCache.ClearEntry("c_adpt_tag_2");
        AicpuTaskCacheManager::aicpuTaskCache.ClearEntry("c_adpt_full_workflow");

        AicpuTaskCacheManager::cacheTag = savedCacheTag_;
        AicpuTaskCacheManager::isHit = savedIsHit_;
        AicpuTaskCacheManager::cacheEntryPtr = savedCacheEntryPtr_;
        AicpuTaskCacheManager::aicpuTaskCache.cacheBytes_ = savedCacheBytes_;
        AicpuTaskCacheManager::aicpuTaskCache.cacheHitRunInfoPrinted_ = savedCacheHitRunInfoPrinted_;
        AicpuTaskCacheManager::aicpuTaskCache.cacheFullRunInfoPrinted_ = savedCacheFullRunInfoPrinted_;
        g_testDeviceType = savedDeviceType_;
    }

    void SetDeviceTypeNotSupport() { g_testDeviceType = DevType::DEV_TYPE_910B; }

    void SetCacheMissState(AicpuTaskCacheEntry* entryPtr)
    {
        AicpuTaskCacheManager::isHit = false;
        AicpuTaskCacheManager::cacheEntryPtr = entryPtr;
    }

    void SetCacheHitState(AicpuTaskCacheEntry* entryPtr)
    {
        AicpuTaskCacheManager::isHit = true;
        AicpuTaskCacheManager::cacheEntryPtr = entryPtr;
    }

    void SetCacheFullState()
    {
        AicpuTaskCacheManager::isHit = false;
        AicpuTaskCacheManager::cacheEntryPtr = nullptr;
    }
};

// ===================== HcommAicpuTsTaskCacheLookup Tests =====================

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Lookup_DeviceNotSupport_ReturnsNotSupport)
{
    SetDeviceTypeNotSupport();
    bool isHit = false;
    EXPECT_EQ(HcommAicpuTsTaskCacheLookup("tag1", &isHit), HCCL_E_NOT_SUPPORT);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Lookup_NullTag_ReturnsPtr)
{
    bool isHit = false;
    EXPECT_EQ(HcommAicpuTsTaskCacheLookup(nullptr, &isHit), HCCL_E_PTR);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Lookup_NullIsHit_ReturnsPtr)
{
    EXPECT_EQ(HcommAicpuTsTaskCacheLookup("tag1", nullptr), HCCL_E_PTR);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Lookup_CacheMiss_ReturnsSuccess)
{
    AicpuTaskCacheManager::aicpuTaskCache.cacheHitRunInfoPrinted_ = false;
    AicpuTaskCacheManager::aicpuTaskCache.ClearEntry("c_adpt_tag_1");
    bool isHit = true;
    EXPECT_EQ(HcommAicpuTsTaskCacheLookup("c_adpt_tag_1", &isHit), HCCL_SUCCESS);
    EXPECT_FALSE(isHit);
    EXPECT_EQ(AicpuTaskCacheManager::cacheTag, std::string("c_adpt_tag_1"));
    EXPECT_FALSE(AicpuTaskCacheManager::isHit);
    EXPECT_NE(AicpuTaskCacheManager::cacheEntryPtr, nullptr);
    EXPECT_FALSE(AicpuTaskCacheManager::aicpuTaskCache.cacheHitRunInfoPrinted_);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Lookup_CacheHit_ReturnsSuccess)
{
    AicpuTaskCacheManager::aicpuTaskCache.cacheHitRunInfoPrinted_ = false;
    AicpuTaskCacheManager::aicpuTaskCache.ClearEntry("c_adpt_tag_1");
    bool isHit = false;
    ASSERT_EQ(HcommAicpuTsTaskCacheLookup("c_adpt_tag_1", &isHit), HCCL_SUCCESS);
    ASSERT_FALSE(isHit);

    AicpuTaskCacheEntry* entry = AicpuTaskCacheManager::cacheEntryPtr;
    ASSERT_NE(entry, nullptr);

    uint64_t baseAddrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1};
    uint64_t memSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    ASSERT_EQ(entry->InitCacheEntry(baseAddrs, memSizes, 2), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    ASSERT_EQ(entry->AddSqeArray(rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0), HCCL_SUCCESS);
    ASSERT_EQ(entry->SubmitCacheEntry(), HCCL_SUCCESS);
    uint64_t entryBytes = entry->GetEntryBytes();
    ASSERT_EQ(AicpuTaskCacheManager::aicpuTaskCache.IncCacheBytes("c_adpt_tag_1", entryBytes), HCCL_SUCCESS);

    bool isHit2 = false;
    EXPECT_EQ(HcommAicpuTsTaskCacheLookup("c_adpt_tag_1", &isHit2), HCCL_SUCCESS);
    EXPECT_TRUE(isHit2);
    EXPECT_EQ(AicpuTaskCacheManager::cacheEntryPtr, entry);
    EXPECT_TRUE(AicpuTaskCacheManager::aicpuTaskCache.cacheHitRunInfoPrinted_);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Lookup_CacheFull_MarksProcessRunInfoPrinted)
{
    AicpuTaskCacheManager::aicpuTaskCache.cacheFullRunInfoPrinted_ = false;
    AicpuTaskCacheManager::aicpuTaskCache.cacheBytes_ = AicpuTaskCacheManager::aicpuTaskCache.maxCacheBytes_;

    bool isHit = true;
    EXPECT_EQ(HcommAicpuTsTaskCacheLookup("c_adpt_full_tag_1", &isHit), HCCL_SUCCESS);
    EXPECT_FALSE(isHit);
    EXPECT_EQ(AicpuTaskCacheManager::cacheEntryPtr, nullptr);
    EXPECT_TRUE(AicpuTaskCacheManager::aicpuTaskCache.cacheFullRunInfoPrinted_);

    EXPECT_EQ(HcommAicpuTsTaskCacheLookup("c_adpt_full_tag_2", &isHit), HCCL_SUCCESS);
    EXPECT_FALSE(isHit);
    EXPECT_EQ(AicpuTaskCacheManager::cacheEntryPtr, nullptr);
    EXPECT_TRUE(AicpuTaskCacheManager::aicpuTaskCache.cacheFullRunInfoPrinted_);
}

// ===================== HcommAicpuTsTaskCacheStart Tests =====================

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Start_DeviceNotSupport_ReturnsNotSupport)
{
    SetDeviceTypeNotSupport();
    void* addrs[] = {(void*)TEST_BASE_ADDR_0, (void*)TEST_BASE_ADDR_1};
    uint64_t sizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(HcommAicpuTsTaskCacheStart("tag1", addrs, sizes, 2), HCCL_E_NOT_SUPPORT);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Start_NullTag_ReturnsPtr)
{
    void* addrs[] = {(void*)TEST_BASE_ADDR_0, (void*)TEST_BASE_ADDR_1};
    uint64_t sizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(HcommAicpuTsTaskCacheStart(nullptr, addrs, sizes, 2), HCCL_E_PTR);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Start_NullAddrs_ReturnsPtr)
{
    uint64_t sizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(HcommAicpuTsTaskCacheStart("tag1", nullptr, sizes, 2), HCCL_E_PTR);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Start_NullSizes_ReturnsPtr)
{
    void* addrs[] = {(void*)TEST_BASE_ADDR_0, (void*)TEST_BASE_ADDR_1};
    EXPECT_EQ(HcommAicpuTsTaskCacheStart("tag1", addrs, nullptr, 2), HCCL_E_PTR);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Start_TagMismatch_ReturnsPara)
{
    AicpuTaskCacheManager::cacheTag = "lookup_tag";
    SetCacheMissState(nullptr);
    void* addrs[] = {(void*)TEST_BASE_ADDR_0, (void*)TEST_BASE_ADDR_1};
    uint64_t sizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(HcommAicpuTsTaskCacheStart("wrong_tag", addrs, sizes, 2), HCCL_E_PARA);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Start_CacheHit_ReturnsInternal)
{
    AicpuTaskCacheManager::cacheTag = "tag1";
    AicpuTaskCacheEntry entry;
    SetCacheHitState(&entry);
    void* addrs[] = {(void*)TEST_BASE_ADDR_0, (void*)TEST_BASE_ADDR_1};
    uint64_t sizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(HcommAicpuTsTaskCacheStart("tag1", addrs, sizes, 2), HCCL_E_INTERNAL);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Start_CacheFull_ReturnsSuccess)
{
    AicpuTaskCacheManager::cacheTag = "tag1";
    SetCacheFullState();
    void* addrs[] = {(void*)TEST_BASE_ADDR_0, (void*)TEST_BASE_ADDR_1};
    uint64_t sizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(HcommAicpuTsTaskCacheStart("tag1", addrs, sizes, 2), HCCL_SUCCESS);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Start_CacheMissWithEntry_InitSucceeds)
{
    AicpuTaskCacheManager::cacheTag = "tag1";
    AicpuTaskCacheEntry entry;
    SetCacheMissState(&entry);
    void* addrs[] = {(void*)TEST_BASE_ADDR_0, (void*)TEST_BASE_ADDR_1};
    uint64_t sizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(HcommAicpuTsTaskCacheStart("tag1", addrs, sizes, 2), HCCL_SUCCESS);
    EXPECT_EQ(entry.cachedBaseAddrs_.size(), 2U);
    EXPECT_EQ(entry.cachedBaseAddrs_[0], TEST_BASE_ADDR_0);
    EXPECT_EQ(entry.cachedBaseAddrs_[1], TEST_BASE_ADDR_1);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Start_CacheMissWithEntry_InvalidCount)
{
    AicpuTaskCacheManager::cacheTag = "tag1";
    AicpuTaskCacheEntry entry;
    SetCacheMissState(&entry);
    void* addrs[] = {(void*)TEST_BASE_ADDR_0};
    uint64_t sizes[] = {TEST_MEM_SIZE_0};
    EXPECT_EQ(HcommAicpuTsTaskCacheStart("tag1", addrs, sizes, 1), HCCL_E_PARA);
}

// ===================== HcommAicpuTsTaskCacheEnd Tests =====================

TEST_F(HcommAicpuTsTaskCacheCAdptTest, End_DeviceNotSupport_ReturnsNotSupport)
{
    SetDeviceTypeNotSupport();
    EXPECT_EQ(HcommAicpuTsTaskCacheEnd("tag1"), HCCL_E_NOT_SUPPORT);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, End_NullTag_ReturnsPtr)
{
    EXPECT_EQ(HcommAicpuTsTaskCacheEnd(nullptr), HCCL_E_PTR);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, End_TagMismatch_ReturnsPara)
{
    AicpuTaskCacheManager::cacheTag = "lookup_tag";
    SetCacheMissState(nullptr);
    EXPECT_EQ(HcommAicpuTsTaskCacheEnd("wrong_tag"), HCCL_E_PARA);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, End_CacheHit_ReturnsInternal)
{
    AicpuTaskCacheManager::cacheTag = "tag1";
    AicpuTaskCacheEntry entry;
    SetCacheHitState(&entry);
    EXPECT_EQ(HcommAicpuTsTaskCacheEnd("tag1"), HCCL_E_INTERNAL);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, End_CacheFull_ReturnsSuccess)
{
    AicpuTaskCacheManager::cacheTag = "tag1";
    SetCacheFullState();
    EXPECT_EQ(HcommAicpuTsTaskCacheEnd("tag1"), HCCL_SUCCESS);
    EXPECT_EQ(AicpuTaskCacheManager::cacheTag, std::string(""));
    EXPECT_FALSE(AicpuTaskCacheManager::isHit);
    EXPECT_EQ(AicpuTaskCacheManager::cacheEntryPtr, nullptr);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, End_CacheMissWithUnsubmittedEntry_ReturnsInternal)
{
    AicpuTaskCacheManager::cacheTag = "tag1";
    AicpuTaskCacheEntry entry;
    SetCacheMissState(&entry);
    EXPECT_EQ(HcommAicpuTsTaskCacheEnd("tag1"), HCCL_E_INTERNAL);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, End_CacheMissWithSubmittedEntry_ReturnsSuccess)
{
    AicpuTaskCacheManager::aicpuTaskCache.ClearEntry("c_adpt_tag_2");
    AicpuTaskCacheManager::cacheTag = "c_adpt_tag_2";
    AicpuTaskCacheEntry entry;
    SetCacheMissState(&entry);

    uint64_t baseAddrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1};
    uint64_t memSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    ASSERT_EQ(entry.InitCacheEntry(baseAddrs, memSizes, 2), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    ASSERT_EQ(entry.AddSqeArray(rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0), HCCL_SUCCESS);

    EXPECT_EQ(HcommAicpuTsTaskCacheEnd("c_adpt_tag_2"), HCCL_SUCCESS);
    EXPECT_EQ(AicpuTaskCacheManager::cacheTag, std::string(""));
    EXPECT_FALSE(AicpuTaskCacheManager::isHit);
    EXPECT_EQ(AicpuTaskCacheManager::cacheEntryPtr, nullptr);
}

// ===================== HcommAicpuTsTaskCacheExecute Tests =====================

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Execute_DeviceNotSupport_ReturnsNotSupport)
{
    SetDeviceTypeNotSupport();
    void* addrs[] = {(void*)TEST_BASE_ADDR_0, (void*)TEST_BASE_ADDR_1};
    uint64_t sizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(HcommAicpuTsTaskCacheExecute("tag1", addrs, sizes, 2), HCCL_E_NOT_SUPPORT);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Execute_NullTag_ReturnsPtr)
{
    void* addrs[] = {(void*)TEST_BASE_ADDR_0, (void*)TEST_BASE_ADDR_1};
    uint64_t sizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(HcommAicpuTsTaskCacheExecute(nullptr, addrs, sizes, 2), HCCL_E_PTR);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Execute_NullAddrs_ReturnsPtr)
{
    uint64_t sizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(HcommAicpuTsTaskCacheExecute("tag1", nullptr, sizes, 2), HCCL_E_PTR);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Execute_NullSizes_ReturnsPtr)
{
    void* addrs[] = {(void*)TEST_BASE_ADDR_0, (void*)TEST_BASE_ADDR_1};
    EXPECT_EQ(HcommAicpuTsTaskCacheExecute("tag1", addrs, nullptr, 2), HCCL_E_PTR);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Execute_TagMismatch_ReturnsPara)
{
    AicpuTaskCacheManager::cacheTag = "lookup_tag";
    AicpuTaskCacheEntry entry;
    SetCacheHitState(&entry);
    void* addrs[] = {(void*)TEST_BASE_ADDR_0, (void*)TEST_BASE_ADDR_1};
    uint64_t sizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(HcommAicpuTsTaskCacheExecute("wrong_tag", addrs, sizes, 2), HCCL_E_PARA);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Execute_CacheMiss_ReturnsInternal)
{
    AicpuTaskCacheManager::cacheTag = "tag1";
    AicpuTaskCacheEntry entry;
    SetCacheMissState(&entry);
    void* addrs[] = {(void*)TEST_BASE_ADDR_0, (void*)TEST_BASE_ADDR_1};
    uint64_t sizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(HcommAicpuTsTaskCacheExecute("tag1", addrs, sizes, 2), HCCL_E_INTERNAL);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Execute_CacheHitWithNullPtr_ReturnsPtr)
{
    AicpuTaskCacheManager::cacheTag = "tag1";
    AicpuTaskCacheManager::isHit = true;
    AicpuTaskCacheManager::cacheEntryPtr = nullptr;
    void* addrs[] = {(void*)TEST_BASE_ADDR_0, (void*)TEST_BASE_ADDR_1};
    uint64_t sizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(HcommAicpuTsTaskCacheExecute("tag1", addrs, sizes, 2), HCCL_E_PTR);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Execute_CountMismatch_ReturnsInternal)
{
    AicpuTaskCacheManager::cacheTag = "tag1";
    AicpuTaskCacheEntry entry;
    uint64_t baseAddrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1};
    uint64_t memSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    ASSERT_EQ(entry.InitCacheEntry(baseAddrs, memSizes, 2), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    ASSERT_EQ(entry.AddSqeArray(rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0), HCCL_SUCCESS);
    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);
    SetCacheHitState(&entry);

    void* addrs[] = {(void*)TEST_BASE_ADDR_0};
    uint64_t sizes[] = {TEST_MEM_SIZE_0};
    EXPECT_EQ(HcommAicpuTsTaskCacheExecute("tag1", addrs, sizes, 1), HCCL_E_INTERNAL);
}

// ===================== HcommAicpuTsTaskCacheClear Tests =====================

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Clear_DeviceNotSupport_ReturnsNotSupport)
{
    SetDeviceTypeNotSupport();
    EXPECT_EQ(HcommAicpuTsTaskCacheClear("tag1"), HCCL_E_NOT_SUPPORT);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Clear_NullTag_ReturnsPtr)
{
    EXPECT_EQ(HcommAicpuTsTaskCacheClear(nullptr), HCCL_E_PTR);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Clear_NonExistentEntry_ReturnsSuccess)
{
    EXPECT_EQ(HcommAicpuTsTaskCacheClear("non_existent_tag"), HCCL_SUCCESS);
}

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Clear_ExistingEntry_ReturnsSuccess)
{
    AicpuTaskCacheManager::aicpuTaskCache.ClearEntry("c_adpt_tag_1");
    AicpuTaskCacheEntry* entryPtr = nullptr;
    ASSERT_EQ(AicpuTaskCacheManager::aicpuTaskCache.AddEntry("c_adpt_tag_1", &entryPtr), HCCL_SUCCESS);
    ASSERT_NE(entryPtr, nullptr);

    EXPECT_EQ(HcommAicpuTsTaskCacheClear("c_adpt_tag_1"), HCCL_SUCCESS);

    AicpuTaskCacheEntry* foundPtr = nullptr;
    AicpuTaskCacheManager::aicpuTaskCache.FindEntry("c_adpt_tag_1", &foundPtr);
    EXPECT_EQ(foundPtr, nullptr);
}

// ===================== Full Workflow Tests =====================

TEST_F(HcommAicpuTsTaskCacheCAdptTest, Workflow_LookupStartAddEndLookupExecute)
{
    AicpuTaskCacheManager::aicpuTaskCache.ClearEntry("c_adpt_full_workflow");

    // Step 1: Lookup (cache miss)
    bool isHit = true;
    EXPECT_EQ(HcommAicpuTsTaskCacheLookup("c_adpt_full_workflow", &isHit), HCCL_SUCCESS);
    EXPECT_FALSE(isHit);
    AicpuTaskCacheEntry* entry = AicpuTaskCacheManager::cacheEntryPtr;
    ASSERT_NE(entry, nullptr);

    // Step 2: Start (init cache entry)
    void* addrs[] = {(void*)TEST_BASE_ADDR_0, (void*)TEST_BASE_ADDR_1};
    uint64_t sizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(HcommAicpuTsTaskCacheStart("c_adpt_full_workflow", addrs, sizes, 2), HCCL_SUCCESS);

    // Step 3: Add SQE to entry via manager
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    EXPECT_EQ(AicpuTaskCacheManager::AddSqeArray(rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0), HCCL_SUCCESS);

    // Step 4: End (submit cache entry)
    EXPECT_EQ(HcommAicpuTsTaskCacheEnd("c_adpt_full_workflow"), HCCL_SUCCESS);
    EXPECT_EQ(AicpuTaskCacheManager::cacheTag, std::string(""));
    EXPECT_FALSE(AicpuTaskCacheManager::isHit);
    EXPECT_EQ(AicpuTaskCacheManager::cacheEntryPtr, nullptr);

    // Step 5: Lookup again (cache hit)
    bool isHit2 = false;
    EXPECT_EQ(HcommAicpuTsTaskCacheLookup("c_adpt_full_workflow", &isHit2), HCCL_SUCCESS);
    EXPECT_TRUE(isHit2);
    EXPECT_EQ(AicpuTaskCacheManager::cacheEntryPtr, entry);

    // Step 6: Execute (count mismatch returns error, does not reset context)
    void* wrongAddrs[] = {(void*)TEST_BASE_ADDR_0};
    uint64_t wrongSizes[] = {TEST_MEM_SIZE_0};
    EXPECT_EQ(HcommAicpuTsTaskCacheExecute("c_adpt_full_workflow", wrongAddrs, wrongSizes, 1), HCCL_E_INTERNAL);
}

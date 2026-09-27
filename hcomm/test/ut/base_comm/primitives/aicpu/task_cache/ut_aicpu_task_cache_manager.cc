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
using namespace Hccl;

namespace {
constexpr uint32_t TEST_SQ_DEPTH = 128;
constexpr uint64_t TEST_BASE_ADDR_0 = 0x10000ULL;
constexpr uint64_t TEST_MEM_SIZE_0 = 0x1000ULL;
constexpr uint64_t TEST_BASE_ADDR_1 = 0x20000ULL;
constexpr uint64_t TEST_MEM_SIZE_1 = 0x2000ULL;

Hccl::WqeTask MakeManagerWqeTaskRead()
{
    Hccl::WqeTask wqe;
    memset(&wqe, 0, sizeof(wqe));
    UdmaSqeCommon* common = reinterpret_cast<UdmaSqeCommon*>(&wqe);
    common->opcode = static_cast<uint8_t>(Hccl::UdmaSqOpcode::UDMA_OPC_READ);
    return wqe;
}

Hccl::DbSqeProfInfo MakeManagerDbSqeProfInfo(bool isValid)
{
    Hccl::DbSqeProfInfo info{};
    memset(&info, 0, sizeof(info));
    info.isValid = isValid;
    return info;
}
} // namespace

class AicpuTaskCacheManagerTest : public testing::Test {
protected:
    std::unique_ptr<Hccl::RtsqA5> rtsq_;
    Hccl::RtsqA5* rtsqPtr_ = nullptr;
    std::unique_ptr<hccl::AicpuTsThread> aicpuTsThread_;
    std::unique_ptr<Hccl::UbConnLite> ubConnLite_;
    std::unique_ptr<Hccl::UbTransportLiteImpl> ubTransport_;
    std::unique_ptr<AicpuTaskCacheEntry> entry_;
    std::vector<char> emptyUniqueId_;
    bool savedIsHit_ = false;
    AicpuTaskCacheEntry* savedCacheEntryPtr_ = nullptr;

    void SetUp() override
    {
        rtsq_ = std::make_unique<Hccl::RtsqA5>(0, 0, 0);
        rtsqPtr_ = rtsq_.get();
        aicpuTsThread_ = std::make_unique<hccl::AicpuTsThread>(std::string("ut_mgr_thread"));
        ubConnLite_ = std::make_unique<Hccl::UbConnLite>(
            Hccl::UbJettyLiteId(0, 0, 0), Hccl::UbJettyLiteAttr(0, 0, TEST_SQ_DEPTH, 0, true), Hccl::Eid{});
        ubTransport_ = std::make_unique<Hccl::UbTransportLiteImpl>(emptyUniqueId_);
        entry_ = std::make_unique<AicpuTaskCacheEntry>();

        uint64_t baseAddrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1};
        uint64_t memSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
        ASSERT_EQ(entry_->InitCacheEntry(baseAddrs, memSizes, 2), HCCL_SUCCESS);

        savedIsHit_ = AicpuTaskCacheManager::isHit;
        savedCacheEntryPtr_ = AicpuTaskCacheManager::cacheEntryPtr;
    }

    void TearDown() override
    {
        AicpuTaskCacheManager::isHit = savedIsHit_;
        AicpuTaskCacheManager::cacheEntryPtr = savedCacheEntryPtr_;
        AicpuTaskCacheManager::cacheTag = "";
    }

    void SetNeedCacheTaskState(AicpuTaskCacheEntry* entryPtr)
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

// ===================== NeedCacheTask Tests =====================

TEST_F(AicpuTaskCacheManagerTest, NeedCacheTask_CacheMissWithEntry_ReturnsTrue)
{
    SetNeedCacheTaskState(entry_.get());
    EXPECT_TRUE(AicpuTaskCacheManager::NeedCacheTask());
}

TEST_F(AicpuTaskCacheManagerTest, NeedCacheTask_CacheHit_ReturnsFalse)
{
    SetCacheHitState(entry_.get());
    EXPECT_FALSE(AicpuTaskCacheManager::NeedCacheTask());
}

TEST_F(AicpuTaskCacheManagerTest, NeedCacheTask_CacheFull_ReturnsFalse)
{
    SetCacheFullState();
    EXPECT_FALSE(AicpuTaskCacheManager::NeedCacheTask());
}

TEST_F(AicpuTaskCacheManagerTest, NeedCacheTask_CacheHitWithNullPtr_ReturnsFalse)
{
    AicpuTaskCacheManager::isHit = true;
    AicpuTaskCacheManager::cacheEntryPtr = nullptr;
    EXPECT_FALSE(AicpuTaskCacheManager::NeedCacheTask());
}

// ===================== AddWqeArray Tests =====================

TEST_F(AicpuTaskCacheManagerTest, AddWqeArray_NeedCacheTaskFalse_ReturnsInternal)
{
    SetCacheFullState();
    std::vector<Hccl::WqeTask> wqeTasks = {MakeManagerWqeTaskRead()};
    EXPECT_EQ(
        AicpuTaskCacheManager::AddWqeArray(
            ubConnLite_.get(), ubTransport_.get(), wqeTasks, 0, 0, false, MakeManagerDbSqeProfInfo(false)),
        HCCL_E_INTERNAL);
}

TEST_F(AicpuTaskCacheManagerTest, AddWqeArray_CacheHit_ReturnsInternal)
{
    SetCacheHitState(entry_.get());
    std::vector<Hccl::WqeTask> wqeTasks = {MakeManagerWqeTaskRead()};
    EXPECT_EQ(
        AicpuTaskCacheManager::AddWqeArray(
            ubConnLite_.get(), ubTransport_.get(), wqeTasks, 0, 0, false, MakeManagerDbSqeProfInfo(false)),
        HCCL_E_INTERNAL);
}

TEST_F(AicpuTaskCacheManagerTest, AddWqeArray_NullUbConnLite_ReturnsPtrError)
{
    SetNeedCacheTaskState(entry_.get());
    std::vector<Hccl::WqeTask> wqeTasks = {MakeManagerWqeTaskRead()};
    EXPECT_EQ(
        AicpuTaskCacheManager::AddWqeArray(
            nullptr, ubTransport_.get(), wqeTasks, 0, 0, false, MakeManagerDbSqeProfInfo(false)),
        HCCL_E_PTR);
}

TEST_F(AicpuTaskCacheManagerTest, AddWqeArray_Success_DelegatesToEntry)
{
    SetNeedCacheTaskState(entry_.get());
    std::vector<Hccl::WqeTask> wqeTasks = {MakeManagerWqeTaskRead()};
    EXPECT_EQ(
        AicpuTaskCacheManager::AddWqeArray(
            ubConnLite_.get(), ubTransport_.get(), wqeTasks, 0, 0, false, MakeManagerDbSqeProfInfo(false)),
        HCCL_SUCCESS);
    EXPECT_EQ(entry_->wqeTaskArrayInfos_.size(), 1U);
}

// ===================== AddSqeArray Tests =====================

TEST_F(AicpuTaskCacheManagerTest, AddSqeArray_NeedCacheTaskFalse_ReturnsInternal)
{
    SetCacheFullState();
    std::vector<uint8_t> sqeArray(Hccl::AC_SQE_SIZE, 0);
    Rt91095StarsSqeHeader* header = reinterpret_cast<Rt91095StarsSqeHeader*>(sqeArray.data());
    header->type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    EXPECT_EQ(
        AicpuTaskCacheManager::AddSqeArray(rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0), HCCL_E_INTERNAL);
}

TEST_F(AicpuTaskCacheManagerTest, AddSqeArray_CacheHit_ReturnsInternal)
{
    SetCacheHitState(entry_.get());
    std::vector<uint8_t> sqeArray(Hccl::AC_SQE_SIZE, 0);
    Rt91095StarsSqeHeader* header = reinterpret_cast<Rt91095StarsSqeHeader*>(sqeArray.data());
    header->type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    EXPECT_EQ(
        AicpuTaskCacheManager::AddSqeArray(rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0), HCCL_E_INTERNAL);
}

TEST_F(AicpuTaskCacheManagerTest, AddSqeArray_NullRtsqPtr_ReturnsPtrError)
{
    SetNeedCacheTaskState(entry_.get());
    std::vector<uint8_t> sqeArray(Hccl::AC_SQE_SIZE, 0);
    EXPECT_EQ(AicpuTaskCacheManager::AddSqeArray(nullptr, aicpuTsThread_.get(), 1, sqeArray.data(), 0), HCCL_E_PTR);
}

TEST_F(AicpuTaskCacheManagerTest, AddSqeArray_NullAicpuTsThread_ReturnsPtrError)
{
    SetNeedCacheTaskState(entry_.get());
    std::vector<uint8_t> sqeArray(Hccl::AC_SQE_SIZE, 0);
    EXPECT_EQ(AicpuTaskCacheManager::AddSqeArray(rtsqPtr_, nullptr, 1, sqeArray.data(), 0), HCCL_E_PTR);
}

TEST_F(AicpuTaskCacheManagerTest, AddSqeArray_Success_DelegatesToEntry)
{
    SetNeedCacheTaskState(entry_.get());
    std::vector<uint8_t> sqeArray(Hccl::AC_SQE_SIZE, 0);
    Rt91095StarsSqeHeader* header = reinterpret_cast<Rt91095StarsSqeHeader*>(sqeArray.data());
    header->type = static_cast<uint8_t>(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    EXPECT_EQ(AicpuTaskCacheManager::AddSqeArray(rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0), HCCL_SUCCESS);
    EXPECT_EQ(entry_->sqeArrayInfos_.size(), 1U);
}

// ===================== Static Member Variables Tests =====================

TEST_F(AicpuTaskCacheManagerTest, StaticMembers_InitialValues)
{
    EXPECT_EQ(AicpuTaskCacheManager::cacheTag, "");
    EXPECT_FALSE(AicpuTaskCacheManager::isHit);
    EXPECT_EQ(AicpuTaskCacheManager::cacheEntryPtr, nullptr);
}

TEST_F(AicpuTaskCacheManagerTest, StaticMembers_SetAndGet)
{
    AicpuTaskCacheManager::cacheTag = "test_tag";
    AicpuTaskCacheManager::isHit = true;
    AicpuTaskCacheManager::cacheEntryPtr = entry_.get();
    EXPECT_EQ(AicpuTaskCacheManager::cacheTag, "test_tag");
    EXPECT_TRUE(AicpuTaskCacheManager::isHit);
    EXPECT_EQ(AicpuTaskCacheManager::cacheEntryPtr, entry_.get());
}

TEST_F(AicpuTaskCacheManagerTest, AicpuTaskCache_GlobalInstance_Usable)
{
    AicpuTaskCacheEntry* entryPtr = nullptr;
    ASSERT_EQ(AicpuTaskCacheManager::aicpuTaskCache.AddEntry("manager_test_tag", &entryPtr), HCCL_SUCCESS);
    ASSERT_NE(entryPtr, nullptr);
    EXPECT_EQ(AicpuTaskCacheManager::aicpuTaskCache.ClearEntry("manager_test_tag"), HCCL_SUCCESS);
}

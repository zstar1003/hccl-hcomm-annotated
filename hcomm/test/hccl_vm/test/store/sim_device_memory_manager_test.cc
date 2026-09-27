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

#include <atomic>
#include <cstdio>
#include <cstring>
#include <sys/mman.h> // shm_unlink
#include <thread>
#include <vector>

#include "store_sim_comm_pool_policy.h"
#include "store_sim_device_memory_manager.h"
#include "store_sim_memory_manager.h"
#include "store_sim_run_mode.h"
#include "db_sim_runner_db.h"
#include "sim_models.h"

// 阈值和上界取自 CommPoolPolicy，下面纯判定用例用这两个短别名。
static constexpr size_t kThr = sim::CommPoolPolicy::kBigBlockThreshold; // 200MB
static constexpr size_t kPool = sim::CommPoolPolicy::kPoolSize;         // 4GB

// 仅校验模式关：任何 size 都不引流。
TEST(VmemDecisionTest, CheckOnlyOff_AtThreshold_NoPool)
{
    EXPECT_FALSE(sim::CommPoolPolicy::ShouldRedirect(kThr, false, kThr, kPool));
}
TEST(VmemDecisionTest, CheckOnlyOff_AtPoolCeiling_NoPool)
{
    EXPECT_FALSE(sim::CommPoolPolicy::ShouldRedirect(kPool, false, kThr, kPool));
}
// 仅校验模式开：边界判定。
TEST(VmemDecisionTest, ZeroSize_NoPool) { EXPECT_FALSE(sim::CommPoolPolicy::ShouldRedirect(0, true, kThr, kPool)); }
TEST(VmemDecisionTest, BelowThreshold_NoPool)
{
    EXPECT_FALSE(sim::CommPoolPolicy::ShouldRedirect(kThr - 1, true, kThr, kPool));
}
TEST(VmemDecisionTest, AtThreshold_Pool) { EXPECT_TRUE(sim::CommPoolPolicy::ShouldRedirect(kThr, true, kThr, kPool)); }
TEST(VmemDecisionTest, AboveThreshold_Pool)
{
    EXPECT_TRUE(sim::CommPoolPolicy::ShouldRedirect(kThr + 1, true, kThr, kPool));
}
TEST(VmemDecisionTest, JustBelowPoolCeiling_Pool)
{
    EXPECT_TRUE(sim::CommPoolPolicy::ShouldRedirect(kPool - 1, true, kThr, kPool));
}
TEST(VmemDecisionTest, AtPoolCeiling_Pool)
{
    EXPECT_TRUE(sim::CommPoolPolicy::ShouldRedirect(kPool, true, kThr, kPool));
}
TEST(VmemDecisionTest, AbovePoolCeiling_NoPool)
{
    EXPECT_FALSE(sim::CommPoolPolicy::ShouldRedirect(kPool + 1, true, kThr, kPool));
}

// 上界纯判定：仅校验模式开 >4GB 拦截报错，==4GB 和小块不拦，仅校验模式关不拦。
TEST(VmemCeilingTest, WithinCeiling_NotExceed)
{
    EXPECT_FALSE(sim::CommPoolPolicy::ExceedsCeiling(kPool, true, kPool));
    EXPECT_FALSE(sim::CommPoolPolicy::ExceedsCeiling(kThr, true, kPool));
}
TEST(VmemCeilingTest, AbovePoolCeiling_Exceed)
{
    EXPECT_TRUE(sim::CommPoolPolicy::ExceedsCeiling(kPool + 1, true, kPool));
}
TEST(VmemCeilingTest, CheckOnlyOff_NeverExceed)
{
    EXPECT_FALSE(sim::CommPoolPolicy::ExceedsCeiling(kPool + 1, false, kPool));
}

// 建/拆复用区 HcclCommPool。
static void* CreateCommPool()
{
    return sim::MemoryManager::GetInstance().AllocMemByName(
        sim::CommPoolPolicy::kPoolName, sim::CommPoolPolicy::kPoolSize);
}
static void DestroyCommPool() { sim::MemoryManager::GetInstance().FreeMemByName(sim::CommPoolPolicy::kPoolName); }

class DeviceMemoryManagerTest : public testing::Test {
protected:
    void SetUp() override
    {
        // 清进程内和磁盘上残留的 HcclCommPool，保证乱序和重跑自洽。
        sim::MemoryManager::GetInstance().FreeMemByName(sim::CommPoolPolicy::kPoolName);
        shm_unlink(sim::CommPoolPolicy::kPoolName);
        // 写入仅校验模式，让 IsCheckOnlyMode() 缓存为 true，大块引流集成用例才会命中复用区。
        RunnerDB::DeleteAll<sim::RunModeConfig>();
        sim::RunModeConfig cfg{};
        cfg.mode = 1;
        RunnerDB::Add<sim::RunModeConfig>(cfg);
    }
    void TearDown() override { sim::MemoryManager::GetInstance().FreeMemByName("dev_test_phy"); }
};

TEST_F(DeviceMemoryManagerTest, GetInstance_Singleton_SameInstance)
{
    sim::DeviceMemoryManager& inst1 = sim::DeviceMemoryManager::GetInstance();
    sim::DeviceMemoryManager& inst2 = sim::DeviceMemoryManager::GetInstance();
    EXPECT_EQ(&inst1, &inst2);
}

TEST_F(DeviceMemoryManagerTest, AllocVirMem_Normal_Success)
{
    void* ptr = sim::DeviceMemoryManager::GetInstance().AllocVirMem(0, 1024);
    EXPECT_NE(ptr, nullptr);
}

TEST_F(DeviceMemoryManagerTest, AllocVirMem_DifferentDevices_DifferentAddr)
{
    void* ptr0 = sim::DeviceMemoryManager::GetInstance().AllocVirMem(0, 1024);
    void* ptr1 = sim::DeviceMemoryManager::GetInstance().AllocVirMem(1, 1024);
    EXPECT_NE(ptr0, nullptr);
    EXPECT_NE(ptr1, nullptr);
    EXPECT_NE(ptr0, ptr1);
}

TEST_F(DeviceMemoryManagerTest, AllocVirMem_SameDevice_Sequential_Success)
{
    void* ptr1 = sim::DeviceMemoryManager::GetInstance().AllocVirMem(0, 1024);
    void* ptr2 = sim::DeviceMemoryManager::GetInstance().AllocVirMem(0, 2048);
    EXPECT_NE(ptr1, nullptr);
    EXPECT_NE(ptr2, nullptr);
    EXPECT_NE(ptr1, ptr2);
}

TEST_F(DeviceMemoryManagerTest, FreeVirMem_Normal_DoNothing)
{
    void* ptr = sim::DeviceMemoryManager::GetInstance().AllocVirMem(0, 1024);
    EXPECT_NE(ptr, nullptr);
    EXPECT_NO_THROW(sim::DeviceMemoryManager::GetInstance().FreeVirMem(0, ptr));
}

TEST_F(DeviceMemoryManagerTest, AllocPhyMem_Normal_Success)
{
    void* ptr = sim::DeviceMemoryManager::GetInstance().AllocPhyMem("dev_test_phy", 0, 1024);
    EXPECT_NE(ptr, nullptr);
}

TEST_F(DeviceMemoryManagerTest, AllocPhyMem_NullName_Fail)
{
    void* ptr = sim::DeviceMemoryManager::GetInstance().AllocPhyMem(nullptr, 0, 1024);
    EXPECT_EQ(ptr, nullptr);
}

TEST_F(DeviceMemoryManagerTest, FreePhyMem_Normal_Success)
{
    void* ptr = sim::DeviceMemoryManager::GetInstance().AllocPhyMem("dev_test_phy_free", 0, 1024);
    EXPECT_NE(ptr, nullptr);
    EXPECT_NO_THROW(sim::DeviceMemoryManager::GetInstance().FreePhyMem("dev_test_phy_free", 0));
    sim::MemoryManager::GetInstance().FreeMemByName("dev_test_phy_free");
}

TEST_F(DeviceMemoryManagerTest, FreePhyMem_NullName_DoNothing)
{
    EXPECT_NO_THROW(sim::DeviceMemoryManager::GetInstance().FreePhyMem(nullptr, 0));
}

TEST_F(DeviceMemoryManagerTest, AcquirePhyMem_NotExist_Fail)
{
    void* ptr = sim::DeviceMemoryManager::GetInstance().AcquirePhyMem("not_exist_phy", 0, 1024);
    EXPECT_EQ(ptr, nullptr);
}

TEST_F(DeviceMemoryManagerTest, AcquirePhyMem_AfterAlloc_Success)
{
    void* ptr1 = sim::DeviceMemoryManager::GetInstance().AllocPhyMem("dev_test_acq", 0, 1024);
    EXPECT_NE(ptr1, nullptr);
    void* ptr2 = sim::DeviceMemoryManager::GetInstance().AcquirePhyMem("dev_test_acq", 0, 1024);
    EXPECT_NE(ptr2, nullptr);
    sim::DeviceMemoryManager::GetInstance().FreePhyMem("dev_test_acq", 0);
    sim::MemoryManager::GetInstance().FreeMemByName("dev_test_acq");
}

TEST_F(DeviceMemoryManagerTest, ReleasePhyMem_Normal_Success)
{
    void* ptr = sim::DeviceMemoryManager::GetInstance().AllocPhyMem("dev_test_rel", 0, 1024);
    EXPECT_NE(ptr, nullptr);
    int ret = sim::DeviceMemoryManager::GetInstance().ReleasePhyMem("dev_test_rel", 0);
    EXPECT_EQ(ret, 0);
    sim::MemoryManager::GetInstance().FreeMemByName("dev_test_rel");
}

TEST_F(DeviceMemoryManagerTest, ReleasePhyMem_NullName_Fail)
{
    int ret = sim::DeviceMemoryManager::GetInstance().ReleasePhyMem(nullptr, 0);
    EXPECT_EQ(ret, 0);
}

TEST_F(DeviceMemoryManagerTest, MapDevPtrHostPtr_Normal_Success)
{
    void* devPtr = (void*)0x1000;
    void* hostPtr = (void*)0x2000;
    EXPECT_NO_THROW(sim::DeviceMemoryManager::GetInstance().MapDevPtrHostPtr(devPtr, hostPtr));
    void* result = sim::DeviceMemoryManager::GetInstance().GetHostPtrByDevPtr(devPtr);
    EXPECT_EQ(result, hostPtr);
    sim::DeviceMemoryManager::GetInstance().UnmapDevPtrHostPtr(devPtr);
}

TEST_F(DeviceMemoryManagerTest, UnmapDevPtrHostPtr_AfterMap_Success)
{
    void* devPtr = (void*)0x1000;
    void* hostPtr = (void*)0x2000;
    sim::DeviceMemoryManager::GetInstance().MapDevPtrHostPtr(devPtr, hostPtr);
    EXPECT_NO_THROW(sim::DeviceMemoryManager::GetInstance().UnmapDevPtrHostPtr(devPtr));
    void* result = sim::DeviceMemoryManager::GetInstance().GetHostPtrByDevPtr(devPtr);
    EXPECT_EQ(result, nullptr);
}

TEST_F(DeviceMemoryManagerTest, GetHostPtrByDevPtr_NotExist_ReturnsNull)
{
    void* devPtr = (void*)0x9999;
    void* result = sim::DeviceMemoryManager::GetInstance().GetHostPtrByDevPtr(devPtr);
    EXPECT_EQ(result, nullptr);
}

// SetUp 已写入仅校验模式，集成路径走仅校验模式开分支。仅校验模式关时不引流由 VmemDecisionTest 覆盖。
// 池基址用大块 AllocPhyMem 的返回值获取，命中复用即返回池首址，不依赖内部 getter。

TEST_F(DeviceMemoryManagerTest, AllocPhyMem_BigBlocks_ShareSamePool)
{
    EXPECT_TRUE(sim::IsCheckOnlyMode()); // fixture 已写入仅校验模式，首次缓存须为 true。
    auto& mgr = sim::DeviceMemoryManager::GetInstance();
    ASSERT_NE(CreateCommPool(), nullptr);
    const size_t big = sim::CommPoolPolicy::kBigBlockThreshold; // 200MB
    void* a = mgr.AllocPhyMem("big_a", 0, big);
    void* b = mgr.AllocPhyMem("big_b", 0, big);
    void* small = mgr.AllocPhyMem("small_c", 0, 1024);
    EXPECT_NE(a, nullptr);
    EXPECT_EQ(a, b);     // 两个大块归同一复用区
    EXPECT_NE(a, small); // 小块走独立真实分配
    mgr.FreePhyMem("big_a", 0);
    mgr.FreePhyMem("big_b", 0);
    mgr.FreePhyMem("small_c", 0);
    DestroyCommPool();
}

TEST_F(DeviceMemoryManagerTest, AllocPhyMem_ExceedCeiling_Reject)
{
    // 仅校验模式下单块 >4GB 报错，不回退真实分配。
    auto& mgr = sim::DeviceMemoryManager::GetInstance();
    const size_t tooBig = sim::CommPoolPolicy::kPoolSize + 1; // >4GB
    EXPECT_EQ(mgr.AllocPhyMem("too_big_alloc", 0, tooBig), nullptr);
    EXPECT_EQ(mgr.AcquirePhyMem("too_big_acq", 0, tooBig), nullptr);
    // 报错路径不留记账，后续 free/release 当未记录处理，不抛异常。
    EXPECT_NO_THROW(mgr.FreePhyMem("too_big_alloc", 0));
    EXPECT_EQ(mgr.ReleasePhyMem("too_big_acq", 0), 0);
}

TEST_F(DeviceMemoryManagerTest, FreePhyMem_BigBlock_PoolReleaseMovedToCaller)
{
    // 大块引流到复用区，FreePhyMem 走非池释放，对从未单独注册的大块名是空操作。
    // 复用区的释放由 aclrt 调用方完成，不在 FreePhyMem 处理。
    auto& mgr = sim::DeviceMemoryManager::GetInstance();
    void* base = CreateCommPool();
    ASSERT_NE(base, nullptr);
    void* a = mgr.AllocPhyMem("big_free", 0, sim::CommPoolPolicy::kBigBlockThreshold);
    EXPECT_EQ(a, base);                             // 大块拿到的就是复用区基址
    EXPECT_NO_THROW(mgr.FreePhyMem("big_free", 0)); // 非池释放，不触碰复用区
    // 释放由调用方完成，这里手动配平 AllocPhyMem 内部的那次 acquire
    sim::MemoryManager::GetInstance().ReleaseMemByName(sim::CommPoolPolicy::kPoolName);
    DestroyCommPool();
}

TEST_F(DeviceMemoryManagerTest, AllocPhyMem_SmallBlock_ContentCorrect)
{
    auto& mgr = sim::DeviceMemoryManager::GetInstance();
    ASSERT_NE(CreateCommPool(), nullptr);
    char* p = static_cast<char*>(mgr.AllocPhyMem("small_int", 0, 4096));
    ASSERT_NE(p, nullptr);
    const char* msg = "checker-vmem-small-correct";
    memcpy(p, msg, strlen(msg) + 1);
    EXPECT_STREQ(p, msg);           // 小块内容正确
    mgr.FreePhyMem("small_int", 0); // 小块走原逻辑，内部已 FreeMemByName 释放
    DestroyCommPool();
}

TEST_F(DeviceMemoryManagerTest, AllocPhyMem_RepeatedBig_NoGrowth)
{
    auto& mgr = sim::DeviceMemoryManager::GetInstance();
    ASSERT_NE(CreateCommPool(), nullptr);
    void* first = mgr.AllocPhyMem("loop_0", 0, sim::CommPoolPolicy::kBigBlockThreshold);
    mgr.FreePhyMem("loop_0", 0);
    for (int i = 1; i < 100; ++i) {
        char n[32];
        snprintf(n, sizeof(n), "loop_%d", i);
        void* p = mgr.AllocPhyMem(n, 0, sim::CommPoolPolicy::kBigBlockThreshold);
        EXPECT_EQ(p, first); // 反复申请恒归同一池、占用不增长
        mgr.FreePhyMem(n, 0);
    }
    DestroyCommPool();
}

TEST_F(DeviceMemoryManagerTest, AllocPhyMem_ConcurrentBig_ThreadSafe)
{
    auto& mgr = sim::DeviceMemoryManager::GetInstance();
    ASSERT_NE(CreateCommPool(), nullptr);
    void* base = mgr.AllocPhyMem("cc_base", 0, sim::CommPoolPolicy::kBigBlockThreshold);
    mgr.FreePhyMem("cc_base", 0);
    std::atomic<int> mismatch{0};
    std::vector<std::thread> ts;
    for (int t = 0; t < 8; ++t) {
        ts.emplace_back([&, t] {
            for (int i = 0; i < 50; ++i) {
                char n[40];
                snprintf(n, sizeof(n), "cc_%d_%d", t, i);
                void* p = mgr.AllocPhyMem(n, 0, sim::CommPoolPolicy::kBigBlockThreshold);
                if (p != base) {
                    mismatch++;
                }
                mgr.FreePhyMem(n, 0);
            }
        });
    }
    for (auto& x : ts) {
        x.join();
    }
    EXPECT_EQ(mismatch.load(), 0); // 并发下均归同一池、无竞态
    DestroyCommPool();
}

TEST_F(DeviceMemoryManagerTest, PoolCeiling_FullSpanAddressable_ContentCorrect)
{
    // 复用区 4GB。验证首址和紧贴 4GB 上界的末字节都能写读、内容正确，整段 4GB 在规格内可寻址。
    // mmap 惰性提交，只触碰的页才落 /dev/shm，只占几页，不会真占 4GB。
    auto& mgr = sim::DeviceMemoryManager::GetInstance();
    ASSERT_NE(CreateCommPool(), nullptr);
    char* base = static_cast<char*>(mgr.AllocPhyMem("ceiling_probe", 0, sim::CommPoolPolicy::kBigBlockThreshold));
    ASSERT_NE(base, nullptr);
    const size_t pool = sim::CommPoolPolicy::kPoolSize; // 规格 4GB

    // 首址写读
    const char* sHead = "vmem-pool-head";
    memcpy(base, sHead, strlen(sHead) + 1);
    EXPECT_STREQ(base, sHead);

    // 紧贴 4GB 上界的最后 64 字节写读，验证整段 4GB 在规格内可寻址
    const char* sTail = "vmem-4G-ceiling";
    char* last = base + pool - 64;
    memcpy(last, sTail, strlen(sTail) + 1);
    EXPECT_STREQ(last, sTail);

    mgr.FreePhyMem("ceiling_probe", 0);
    DestroyCommPool();
}

TEST_F(DeviceMemoryManagerTest, AllocPhyMem_BigBlockOverwrite_NoContentGuarantee)
{
    // 两个不同名大块（两个 rank）都引流到同一池区，后写者覆盖先写者，内容不保证正确。
    // 覆盖是共享后备存储的性质，与进程边界无关，每个 rank 都 Acquire 同一 HcclCommPool。
    // 与小块各自独立分配对照。
    auto& mgr = sim::DeviceMemoryManager::GetInstance();
    ASSERT_NE(CreateCommPool(), nullptr);
    const size_t big = sim::CommPoolPolicy::kBigBlockThreshold; // 200MB
    const size_t probe = 4096;                                  // 只触碰首页，避免真占 200MB

    char* a = static_cast<char*>(mgr.AllocPhyMem("rankA_big", 0, big));
    char* b = static_cast<char*>(mgr.AllocPhyMem("rankB_big", 0, big));
    ASSERT_NE(a, nullptr);
    ASSERT_EQ(a, b); // 两个大块别名同一池区

    memset(a, 0xAA, probe); // rankA 先写
    ASSERT_EQ(static_cast<unsigned char>(a[0]), 0xAAu);
    memset(b, 0xBB, probe); // rankB 后写
    // 经 A 读回变成 0xBB，rankA 的数据被 rankB 覆盖
    EXPECT_EQ(static_cast<unsigned char>(a[0]), 0xBBu);
    EXPECT_EQ(static_cast<unsigned char>(a[probe - 1]), 0xBBu);
    mgr.FreePhyMem("rankA_big", 0);
    mgr.FreePhyMem("rankB_big", 0);

    // 对照：两个小块各自独立分配，互不覆盖
    char* sa = static_cast<char*>(mgr.AllocPhyMem("rankA_small", 0, 4096));
    char* sb = static_cast<char*>(mgr.AllocPhyMem("rankB_small", 0, 4096));
    ASSERT_NE(sa, nullptr);
    ASSERT_NE(sb, nullptr);
    EXPECT_NE(sa, sb); // 小块异址
    memset(sa, 0xAA, 64);
    memset(sb, 0xBB, 64);
    EXPECT_EQ(static_cast<unsigned char>(sa[0]), 0xAAu); // 小块不被覆盖
    EXPECT_EQ(static_cast<unsigned char>(sb[0]), 0xBBu);
    mgr.FreePhyMem("rankA_small", 0);
    mgr.FreePhyMem("rankB_small", 0);
    DestroyCommPool();
}

TEST_F(DeviceMemoryManagerTest, AcquirePhyMem_BigBlockExternalPool_HitsPool)
{
    // 池由主进程建好，本进程未建池，大块 Acquire 按 size 引流命中同一池。
    auto& mgr = sim::DeviceMemoryManager::GetInstance();
    void* poolByMain = CreateCommPool();
    ASSERT_NE(poolByMain, nullptr);
    char* big = static_cast<char*>(mgr.AcquirePhyMem("proxy_big", 0, sim::CommPoolPolicy::kBigBlockThreshold));
    ASSERT_NE(big, nullptr);
    EXPECT_EQ(static_cast<void*>(big), poolByMain); // 未建池也命中同一池首址
    mgr.ReleasePhyMem("proxy_big", 0);
    DestroyCommPool(); // 强制关闭并 unlink
}

TEST_F(DeviceMemoryManagerTest, FreeReleasePhyMem_UnrecordedAndNull_Tolerated)
{
    auto& mgr = sim::DeviceMemoryManager::GetInstance();
    EXPECT_NO_THROW(mgr.FreePhyMem("never_alloced", 0));
    EXPECT_EQ(mgr.ReleasePhyMem("never_acquired", 0), 0);
    EXPECT_EQ(mgr.AllocPhyMem(nullptr, 0, sim::CommPoolPolicy::kBigBlockThreshold), nullptr);
    EXPECT_EQ(mgr.AcquirePhyMem(nullptr, 0, sim::CommPoolPolicy::kBigBlockThreshold), nullptr);
}

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 主机侧单测：覆盖大块引流、上界报错、aclrtFreeHost 身份判定。
// MallocHost 走真实 MemoryManager 和 HcclCommPool 共享内存，不用 mock。
// memcpy/memset 的设备侧短路判定由 CommPoolPolicy::ShouldRedirect 的纯函数用例覆盖边界。
// 本文件聚焦不依赖 RunnerDB/Runner 上下文的主机侧入口。

#include <cstdint>
#include <cstring>
#include <sys/mman.h> // shm_unlink
#include <gtest/gtest.h>

#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "store_sim_comm_pool_policy.h"
#include "store_sim_device_memory_manager.h"
#include "store_sim_memory_manager.h"
#include "store_sim_run_mode.h"
#include "db_sim_runner_db.h"
#include "sim_models.h"

extern "C" {
aclError aclrtMallocHost(void** hostPtr, size_t size);
aclError aclrtFreeHost(void* hostPtr);
aclError aclrtMallocHostWithCfg(void** ptr, uint64_t size, aclrtMallocConfig* cfg);
}

// 复用区在整个套件期间保持存活，套件开头建池一次，结束再回收。
class AclrtMemStubTest : public testing::Test {
protected:
    static void SetUpTestSuite()
    {
        // 在套件初始化（已过 main、RunnerDB/sqlite 就绪）时把 mode=1 写进 DB，再主动触发一次
        // IsCheckOnlyMode() 的懒加载，使进程内 static 缓存 latch 成仅校验模式；不依赖任何加载期构造器。
        RunnerDB::DeleteAll<sim::RunModeConfig>();
        sim::RunModeConfig cfg{};
        cfg.mode = 1;
        RunnerDB::Add<sim::RunModeConfig>(cfg);
        ASSERT_TRUE(sim::IsCheckOnlyMode());
        // 清掉上次异常退出残留的 /dev/shm/HcclCommPool（ShmCreate 用 O_EXCL，残留会建池失败）。
        shm_unlink(sim::CommPoolPolicy::kPoolName);
        ASSERT_NE(
            sim::MemoryManager::GetInstance().AllocMemByName(
                sim::CommPoolPolicy::kPoolName, sim::CommPoolPolicy::kPoolSize),
            nullptr);
    }
    static void TearDownTestSuite()
    {
        // 关闭并 unlink，保证 /dev/shm 不泄漏，独立重跑不撞 O_EXCL。
        sim::MemoryManager::GetInstance().FreeMemByName(sim::CommPoolPolicy::kPoolName);
        // 清掉本套件写进共享 DB 的仅校验模式行，避免给其它测试二进制留下 check-only(mode=1)。
        RunnerDB::DeleteAll<sim::RunModeConfig>();
    }
};

// 主机大块两次申请归同一池首址，且与设备侧大块同址，主机与设备共用 HcclCommPool。
TEST_F(AclrtMemStubTest, MallocHost_BigBlock_TwiceSameAddr_SharesDevicePool)
{
    EXPECT_TRUE(sim::IsCheckOnlyMode()); // SetUpTestSuite 已写入 mode=1 并预热缓存，须为 true。
    const size_t big = sim::CommPoolPolicy::kBigBlockThreshold; // 200MB
    void* h1 = nullptr;
    void* h2 = nullptr;
    ASSERT_EQ(aclrtMallocHost(&h1, big), ACL_SUCCESS);
    ASSERT_EQ(aclrtMallocHost(&h2, big), ACL_SUCCESS);
    ASSERT_NE(h1, nullptr);
    EXPECT_EQ(h1, h2); // 两次大块同址

    // 设备侧大块也归同一池基址
    auto& mgr = sim::DeviceMemoryManager::GetInstance();
    void* d = mgr.AllocPhyMem("host_share_probe", 0, big);
    EXPECT_EQ(h1, d); // 主机与设备共用同一池
    mgr.FreePhyMem("host_share_probe", 0);

    EXPECT_EQ(aclrtFreeHost(h1), ACL_SUCCESS);
    EXPECT_EQ(aclrtFreeHost(h2), ACL_SUCCESS);
}

// 主机小块走真实 malloc，独立于池，内容正确。
TEST_F(AclrtMemStubTest, MallocHost_SmallBlock_RealAlloc_NotPool)
{
    void* big = nullptr;
    ASSERT_EQ(aclrtMallocHost(&big, sim::CommPoolPolicy::kBigBlockThreshold), ACL_SUCCESS);

    void* s1 = nullptr;
    void* s2 = nullptr;
    ASSERT_EQ(aclrtMallocHost(&s1, 4096), ACL_SUCCESS);
    ASSERT_EQ(aclrtMallocHost(&s2, 4096), ACL_SUCCESS);
    ASSERT_NE(s1, nullptr);
    ASSERT_NE(s2, nullptr);
    EXPECT_NE(s1, s2);  // 两个小块各自独立分配
    EXPECT_NE(s1, big); // 小块不进池

    // 小块内容正确，互不覆盖
    memset(s1, 0xAA, 4096);
    memset(s2, 0xBB, 4096);
    EXPECT_EQ(static_cast<unsigned char*>(s1)[0], 0xAAu);
    EXPECT_EQ(static_cast<unsigned char*>(s2)[0], 0xBBu);

    EXPECT_EQ(aclrtFreeHost(s1), ACL_SUCCESS);
    EXPECT_EQ(aclrtFreeHost(s2), ACL_SUCCESS);
    EXPECT_EQ(aclrtFreeHost(big), ACL_SUCCESS);
}

// 主机大块 >4GB 报错，不引流也不真实分配。
TEST_F(AclrtMemStubTest, MallocHost_ExceedCeiling_Reject)
{
    void* p = reinterpret_cast<void*>(0xDEADBEEF); // 哨兵：失败时不应被改写
    aclError ret = aclrtMallocHost(&p, sim::CommPoolPolicy::kPoolSize + 1);
    EXPECT_NE(ret, ACL_SUCCESS);
    EXPECT_EQ(p, reinterpret_cast<void*>(0xDEADBEEF)); // 报错路径未写出指针
}

// aclrtFreeHost 身份判定：池内地址 noop，池外真实地址正常 free。
TEST_F(AclrtMemStubTest, FreeHost_PoolAddrNoop_RealAddrFree)
{
    const size_t big = sim::CommPoolPolicy::kBigBlockThreshold;
    void* poolPtr = nullptr;
    ASSERT_EQ(aclrtMallocHost(&poolPtr, big), ACL_SUCCESS);
    ASSERT_NE(poolPtr, nullptr);

    // 池内地址 free 为 noop：返回成功且池仍可写读
    EXPECT_EQ(aclrtFreeHost(poolPtr), ACL_SUCCESS);
    const char* sentinel = "pool-alive-after-noop-free";
    memcpy(poolPtr, sentinel, strlen(sentinel) + 1);
    EXPECT_STREQ(static_cast<char*>(poolPtr), sentinel);

    // 池外真实地址：正常 free，不抛异常
    void* real = nullptr;
    ASSERT_EQ(aclrtMallocHost(&real, 4096), ACL_SUCCESS);
    EXPECT_EQ(aclrtFreeHost(real), ACL_SUCCESS);
}

// MallocHostWithCfg 委托 MallocHost：大块同样引流到池。
TEST_F(AclrtMemStubTest, MallocHostWithCfg_BigBlock_DelegatesToPool)
{
    void* viaPlain = nullptr;
    void* viaCfg = nullptr;
    const size_t big = sim::CommPoolPolicy::kBigBlockThreshold;
    ASSERT_EQ(aclrtMallocHost(&viaPlain, big), ACL_SUCCESS);
    ASSERT_EQ(aclrtMallocHostWithCfg(&viaCfg, big, nullptr), ACL_SUCCESS);
    EXPECT_EQ(viaPlain, viaCfg); // 两条入口归同一池
    aclrtFreeHost(viaPlain);
    aclrtFreeHost(viaCfg);
}

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>

#include "hcomm_c_adpt.h"
#include "hcomm_res_defs.h"
#include "comm_engine_utils.h"

#define private public
#define protected public
#include "engine_ctxs.h"
#undef protected
#undef private

#include <atomic>
#include <vector>
#include <utility>

using namespace hccl;

namespace {
// 记录 HcommEngineCtxDestroy 调用次数的原子计数器（mock 桩使用）
std::atomic<int> g_destroyCallCount{0};
// 记录每次调用传入的 (engine, ctx) 对，用于校验分发路径
std::vector<std::pair<CommEngine, void*>> g_destroyCallArgs;
} // namespace

// ============ 辅助 mock 桩 ============

// HcommEngineCtxCreate mock：分配哨兵地址（不真正 malloc，避免析构时内存泄漏）
static HcommResult stub_HcommEngineCtxCreate(CommEngine engine, uint64_t size, void** ctx)
{
    // 分配一个哨兵地址；不同引擎分配不同哨兵以便区分
    static uint64_t fakeAddr = 0x1000;
    *ctx = reinterpret_cast<void*>(++fakeAddr);
    return HCCL_SUCCESS;
}

// HcommEngineCtxDestroy mock：计数 + 记录参数，不真正 free
static HcommResult stub_HcommEngineCtxDestroy(CommEngine engine, void* ctx)
{
    ++g_destroyCallCount;
    g_destroyCallArgs.emplace_back(engine, ctx);
    return HCCL_SUCCESS;
}

// ============ 测试 fixture ============

class EngineCtxsTest : public testing::Test {
protected:
    void SetUp() override
    {
        g_destroyCallCount = 0;
        g_destroyCallArgs.clear();
        // 默认 mock：Create 分配哨兵，Destroy 计数
        MOCKER(HcommEngineCtxCreate).stubs().will(invoke(stub_HcommEngineCtxCreate));
        MOCKER(HcommEngineCtxDestroy).stubs().will(invoke(stub_HcommEngineCtxDestroy));
    }

    void TearDown() override
    {
        GlobalMockObject::verify();
        g_destroyCallCount = 0;
        g_destroyCallArgs.clear();
    }
};

// ============ 析构非空 contextMap_ 时全部 ctx 被释放 ============
// 4 条 ctx（2 tag × 2 engine），HcommEngineCtxDestroy 调用 4 次
TEST_F(EngineCtxsTest, Ut_EngineCtxsDtor_When_ContextMapNotEmpty_Expect_AllCtxDestroyed)
{
    {
        EngineCtxs ctxs;
        // 创建 2 个 tag、每个 tag 2 个 engine，共 4 条 ctx
        void* ctx1 = nullptr;
        ASSERT_EQ(ctxs.CreateCommEngineCtx("tag1", COMM_ENGINE_CPU, 1024, &ctx1), HCCL_SUCCESS);
        void* ctx2 = nullptr;
        ASSERT_EQ(ctxs.CreateCommEngineCtx("tag1", COMM_ENGINE_CCU, 1024, &ctx2), HCCL_SUCCESS);
        void* ctx3 = nullptr;
        ASSERT_EQ(ctxs.CreateCommEngineCtx("tag2", COMM_ENGINE_CPU, 1024, &ctx3), HCCL_SUCCESS);
        void* ctx4 = nullptr;
        ASSERT_EQ(ctxs.CreateCommEngineCtx("tag2", COMM_ENGINE_CCU, 1024, &ctx4), HCCL_SUCCESS);
        ASSERT_EQ(ctxs.contextMap_.size(), 2u); // 2 个 tag

        // 作用域结束触发析构
    }
    // 验证 HcommEngineCtxDestroy 被调用 4 次
    EXPECT_EQ(g_destroyCallCount.load(), 4);
    // 每次调用传入的 engine 与创建时一致
    std::vector<CommEngine> engines;
    for (const auto& arg : g_destroyCallArgs) {
        engines.push_back(arg.first);
    }
    // 应包含 2 个 CPU + 2 个 CCU（顺序由 unordered_map 遍历决定，不校验顺序）
    int cpuCount = 0;
    int ccuCount = 0;
    for (auto e : engines) {
        if (e == COMM_ENGINE_CPU) {
            ++cpuCount;
        } else if (e == COMM_ENGINE_CCU) {
            ++ccuCount;
        }
    }
    EXPECT_EQ(cpuCount, 2);
    EXPECT_EQ(ccuCount, 2);
    GlobalMockObject::verify();
}

// ============ 析构空 contextMap_ 时无操作无崩溃 ============
// HcommEngineCtxDestroy 调用 0 次
TEST_F(EngineCtxsTest, Ut_EngineCtxsDtor_When_ContextMapEmpty_Expect_NoDestroyCall)
{
    {
        EngineCtxs ctxs; // 不创建任何 ctx
        ASSERT_TRUE(ctxs.contextMap_.empty());
        // 作用域结束触发析构
    }
    // HcommEngineCtxDestroy 调用 0 次（成功无日志）
    EXPECT_EQ(g_destroyCallCount.load(), 0);
    GlobalMockObject::verify();
}

// ============ 析构含混合引擎类型时 4 种引擎均被传入 HcommEngineCtxDestroy ============
TEST_F(EngineCtxsTest, Ut_EngineCtxsDtor_When_MultiEngineType_Expect_AllPassedToDestroy)
{
    {
        EngineCtxs ctxs;
        void* ctxCpu = nullptr;
        ASSERT_EQ(ctxs.CreateCommEngineCtx("t_cpu", COMM_ENGINE_CPU, 1024, &ctxCpu), HCCL_SUCCESS);
        void* ctxAicpu = nullptr;
        ASSERT_EQ(ctxs.CreateCommEngineCtx("t_aicpu", COMM_ENGINE_AICPU, 1024, &ctxAicpu), HCCL_SUCCESS);
        void* ctxAiv = nullptr;
        ASSERT_EQ(ctxs.CreateCommEngineCtx("t_aiv", COMM_ENGINE_AIV, 1024, &ctxAiv), HCCL_SUCCESS);
        void* ctxCcu = nullptr;
        ASSERT_EQ(ctxs.CreateCommEngineCtx("t_ccu", COMM_ENGINE_CCU, 1024, &ctxCcu), HCCL_SUCCESS);

        // 作用域结束触发析构
    }
    // 4 种引擎均被释放
    EXPECT_EQ(g_destroyCallCount.load(), 4);
    // 验证 4 种引擎类型都出现
    bool hasCpu = false;
    bool hasAicpu = false;
    bool hasAiv = false;
    bool hasCcu = false;
    for (const auto& arg : g_destroyCallArgs) {
        if (arg.first == COMM_ENGINE_CPU) {
            hasCpu = true;
        } else if (arg.first == COMM_ENGINE_AICPU) {
            hasAicpu = true;
        } else if (arg.first == COMM_ENGINE_AIV) {
            hasAiv = true;
        } else if (arg.first == COMM_ENGINE_CCU) {
            hasCcu = true;
        }
    }
    EXPECT_TRUE(hasCpu);
    EXPECT_TRUE(hasAicpu);
    EXPECT_TRUE(hasAiv);
    EXPECT_TRUE(hasCcu);
    GlobalMockObject::verify();
}

// ============ 显式 DestroyEngineCtx 后析构不重复释放 ============
// 仅剩余条目被释放
TEST_F(EngineCtxsTest, Ut_EngineCtxsDtor_When_AfterExplicitDestroy_Expect_NoDoubleFree)
{
    {
        EngineCtxs ctxs;
        void* ctx1 = nullptr;
        ASSERT_EQ(ctxs.CreateCommEngineCtx("tag", COMM_ENGINE_CPU, 1024, &ctx1), HCCL_SUCCESS);
        void* ctx2 = nullptr;
        ASSERT_EQ(ctxs.CreateCommEngineCtx("tag", COMM_ENGINE_CCU, 1024, &ctx2), HCCL_SUCCESS);
        void* ctx3 = nullptr;
        ASSERT_EQ(ctxs.CreateCommEngineCtx("tag", COMM_ENGINE_AICPU, 1024, &ctx3), HCCL_SUCCESS);
        ASSERT_EQ(ctxs.contextMap_.size(), 1u);

        // 显式销毁 2 条（DestroyEngineCtx 内部会调 HcommEngineCtxDestroy，计数累加）
        ASSERT_EQ(ctxs.DestroyEngineCtx("tag", COMM_ENGINE_CPU), HCCL_SUCCESS);
        ASSERT_EQ(ctxs.DestroyEngineCtx("tag", COMM_ENGINE_CCU), HCCL_SUCCESS);
        int destroyCountAfterExplicit = g_destroyCallCount.load();
        EXPECT_EQ(destroyCountAfterExplicit, 2); // 显式销毁 2 条

        // 重置计数，仅统计析构阶段调用
        g_destroyCallCount = 0;
        g_destroyCallArgs.clear();

        // 作用域结束触发析构：仅剩余 1 条（AICPU）
    }
    // 析构阶段 HcommEngineCtxDestroy 仅调用 1 次（仅剩余条目 AICPU），无重复释放
    EXPECT_EQ(g_destroyCallCount.load(), 1);
    EXPECT_EQ(g_destroyCallArgs.size(), 1u);
    EXPECT_EQ(g_destroyCallArgs[0].first, COMM_ENGINE_AICPU);
    GlobalMockObject::verify();
}

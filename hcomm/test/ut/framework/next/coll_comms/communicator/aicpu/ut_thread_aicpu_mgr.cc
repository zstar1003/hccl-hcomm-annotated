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
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include <memory>
#include <string>

#define private public
#include "threads/thread_aicpu_mgr.h"
#include "aicpu_ts_thread.h"
#include "stream_lite.h"
#include "rtsq_a5.h"
#undef private

using namespace hccl;

namespace aicpu {
void GetSqeId(const uint32_t num, uint32_t& start, uint32_t& end)
{
    start = 1;
    end = start + num;
}
} // namespace aicpu

class ThreadAicpuMgrTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(ThreadAicpuMgrTest, Constructor_InitializesMembers)
{
    HcclCommDfxLite dfx;
    auto checkCb = [](bool) {
        return HCCL_SUCCESS;
    };
    ThreadAicpuMgr mgr(dfx, checkCb);
    EXPECT_EQ(mgr.GetAllThread().size(), 0u);
}

TEST_F(ThreadAicpuMgrTest, GetThreadMutex_ReturnsValidReference)
{
    HcclCommDfxLite dfx;
    auto checkCb = [](bool) {
        return HCCL_SUCCESS;
    };
    ThreadAicpuMgr mgr(dfx, checkCb);
    // GetThreadMutex 返回有效的 shared_mutex 引用
    std::shared_mutex& mtx = mgr.GetThreadMutex();
    std::shared_lock<std::shared_mutex> lock(mtx);
    EXPECT_TRUE(true);
}

TEST_F(ThreadAicpuMgrTest, InitThreads_NullParam_ReturnsError)
{
    HcclCommDfxLite dfx;
    auto checkCb = [](bool) {
        return HCCL_SUCCESS;
    };
    ThreadAicpuMgr mgr(dfx, checkCb);
    HcclResult ret = mgr.InitThreads(nullptr);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(ThreadAicpuMgrTest, Destructor_CleansUpThreads)
{
    HcclCommDfxLite dfx;
    auto checkCb = [](bool) {
        return HCCL_SUCCESS;
    };
    {
        ThreadAicpuMgr mgr(dfx, checkCb);
        // 析构时无 threads 不应崩溃
    }
    SUCCEED();
}

TEST_F(ThreadAicpuMgrTest, RegisterThreadCacheCallback_AllCallbacksRegistered_ReturnsSuccess)
{
    HcclCommDfxLite dfx;
    ThreadAicpuMgr mgr(dfx, [](bool) {
        return HCCL_SUCCESS;
    });
    AicpuTsThread thread(std::string("ut_register_success"));
    thread.pImpl_ = std::make_unique<Hccl::IAicpuTsThread>(0, 0, 0, 0);
    auto* sl = static_cast<Hccl::StreamLite*>(thread.pImpl_->streamLiteVoidPtr_);
    sl->rtsq = std::make_unique<Hccl::RtsqA5>(0, 0, 0);
    HcclResult ret = mgr.RegisterThreadCacheCallback(&thread);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

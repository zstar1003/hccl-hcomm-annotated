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

#define private public
#include "comm_engine_res_aicpu_mgr.h"
#undef private

using namespace hccl;

class CommEngineResAicpuMgrTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(CommEngineResAicpuMgrTest, Constructor_CreatesSubManagers)
{
    HcclCommDfxLite dfx;
    auto checkCb = [](bool) {
        return HCCL_SUCCESS;
    };
    CommEngineResAicpuMgr mgr(dfx, checkCb);
    EXPECT_NE(mgr.threadMgr_, nullptr);
    EXPECT_NE(mgr.notifyMgr_, nullptr);
}

TEST_F(CommEngineResAicpuMgrTest, GetAllThread_ReturnsEmpty)
{
    HcclCommDfxLite dfx;
    auto checkCb = [](bool) {
        return HCCL_SUCCESS;
    };
    CommEngineResAicpuMgr mgr(dfx, checkCb);
    EXPECT_EQ(mgr.GetAllThread().size(), 0u);
}

TEST_F(CommEngineResAicpuMgrTest, GetThreadMutex_Valid)
{
    HcclCommDfxLite dfx;
    auto checkCb = [](bool) {
        return HCCL_SUCCESS;
    };
    CommEngineResAicpuMgr mgr(dfx, checkCb);
    std::shared_mutex& mtx = mgr.GetThreadMutex();
    std::shared_lock<std::shared_mutex> lock(mtx);
    EXPECT_TRUE(true);
}

TEST_F(CommEngineResAicpuMgrTest, InitThreads_NullParam_ReturnsError)
{
    HcclCommDfxLite dfx;
    auto checkCb = [](bool) {
        return HCCL_SUCCESS;
    };
    CommEngineResAicpuMgr mgr(dfx, checkCb);
    HcclResult ret = mgr.InitThreads(nullptr);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(CommEngineResAicpuMgrTest, NotifyFree_NullParam_ReturnsError)
{
    HcclCommDfxLite dfx;
    auto checkCb = [](bool) {
        return HCCL_SUCCESS;
    };
    CommEngineResAicpuMgr mgr(dfx, checkCb);
    HcclResult ret = mgr.NotifyFree(nullptr);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(CommEngineResAicpuMgrTest, NotifyAlloc_NullParam_ReturnsError)
{
    HcclCommDfxLite dfx;
    auto checkCb = [](bool) {
        return HCCL_SUCCESS;
    };
    CommEngineResAicpuMgr mgr(dfx, checkCb);
    HcclResult ret = mgr.NotifyAlloc(nullptr);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(CommEngineResAicpuMgrTest, ReserveNotifyCapacity_Works)
{
    HcclCommDfxLite dfx;
    auto checkCb = [](bool) {
        return HCCL_SUCCESS;
    };
    CommEngineResAicpuMgr mgr(dfx, checkCb);
    // 不应崩溃
    mgr.ReserveNotifyCapacity(50);
    SUCCEED();
}

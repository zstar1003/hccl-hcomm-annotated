/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

#define private public
#define protected public
#include "flush_manager.h"
#include "flush_handle.h"
#undef private
#undef protected

using namespace Hccl;

namespace {
// ibv_post_send 为 static inline，通过 qp->context->ops.post_send 函数指针调用；
// 这里提供桩函数控制其返回值，避免依赖真实 RDMA 驱动。
int StubPostSendOk(struct ibv_qp*, struct ibv_send_wr*, struct ibv_send_wr**) { return 0; }
int StubPostSendFail(struct ibv_qp*, struct ibv_send_wr*, struct ibv_send_wr**) { return -1; }
} // namespace

class FlushManagerTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override { GlobalMockObject::verify(); }
};

// 覆盖 flush_manager.cc:96,97 — Flush() 调用 ExecuteRdmaRead(...GetExecTimeOut())
TEST_F(FlushManagerTest, Ut_Flush_When_PostSendFails_Expect_NetworkError)
{
    FlushManager fm;

    // 构造 fake ibv 结构：qp->context->ops.post_send 返回失败，
    // 使 ExecuteRdmaRead 在 ibv_post_send 失败后立即返回 HCCL_E_NETWORK。
    struct ibv_context_ops ops = {};
    ops.post_send = &StubPostSendFail;
    struct ibv_context ctx = {};
    ctx.ops = ops;
    struct ibv_cq cq = {};
    cq.context = &ctx;
    struct ibv_qp qp = {};
    qp.context = &ctx;
    qp.send_cq = &cq;

    auto handle = std::make_shared<FlushHandle>();
    handle->loopBackQpParam.ibvQp0 = &qp;
    IpAddress ip("192.168.1.1");
    fm.flushHandleMap_.insert({ip, handle});

    // Flush() 遍历 map，FlushParamPrepare 成功后调用 ExecuteRdmaRead（line 96-97），
    // ibv_post_send 失败 -> ExecuteRdmaRead 返回 HCCL_E_NETWORK -> Flush 返回 HCCL_E_NETWORK。
    HcclResult ret = fm.Flush();
    EXPECT_EQ(ret, HCCL_E_NETWORK);

    // 清空 map，避免 FlushManager 析构时 DestroyAll 触发 fake handle 的 Destroy。
    fm.flushHandleMap_.clear();
}

// 覆盖 flush_manager.cc:124,145 — ExecuteRdmaRead 入口 + 超时分支
TEST_F(FlushManagerTest, Ut_ExecuteRdmaRead_When_Timeout_Expect_HcclETimeout)
{
    FlushManager fm;

    // ibv_post_send 成功，进入轮询循环。
    struct ibv_context_ops ops = {};
    ops.post_send = &StubPostSendOk;
    struct ibv_context ctx = {};
    ctx.ops = ops;
    struct ibv_cq cq = {};
    cq.context = &ctx;
    struct ibv_qp qp = {};
    qp.context = &ctx;
    qp.send_cq = &cq;

    ibv_send_wr swr = {};
    ibv_sge sg = {};
    swr.sg_list = &sg;

    // timeoutSec=0：首次循环 elapsedSec(0) >= 0 立即触发超时分支（line 144-145），返回 HCCL_E_TIMEOUT。
    HcclResult ret = fm.ExecuteRdmaRead(&qp, &cq, swr, 0);
    EXPECT_EQ(ret, HCCL_E_TIMEOUT);
}

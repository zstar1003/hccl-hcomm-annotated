/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 被测目标: hcomm::EndpointCtx —— 保存 ctxHandle 的去重缓存条目（聚合初始化 + 直接成员访问）。

#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>

#include "../../../../../../src/base_comm/resources/endpoints/mgr/endpoint_ctx_mgr.h"
#include "hcomm_res_defs.h"

using namespace hcomm;

class EndpointCtxTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override { GlobalMockObject::verify(); }
};

// TC-EndpointCtx_GetCtxHandle-001: 聚合初始化后 ctxHandle 成员保存传入的句柄
TEST_F(EndpointCtxTest, Ut_GetCtxHandle_When_ConstructedWithHandle_Expect_ReturnSameHandle)
{
    void* fakeCtx = reinterpret_cast<void*>(0xABCD1234);
    EndpointCtxKey key{};
    key.devPhyId = 0;
    key.protocol = COMM_PROTOCOL_ROCE;

    EndpointCtx ctx{fakeCtx, key};

    EXPECT_EQ(ctx.ctxHandle, fakeCtx);
}

// TC-EndpointCtx_GetCtxHandle-001 补充: ctxHandle 为空时成员保持为空
TEST_F(EndpointCtxTest, Ut_GetCtxHandle_When_ConstructedWithNull_Expect_ReturnNull)
{
    EndpointCtxKey key{};
    EndpointCtx ctx{nullptr, key};

    EXPECT_EQ(ctx.ctxHandle, nullptr);
}

// TC-EndpointCtx_GetKey-001: 聚合初始化后 key 成员保存传入的 EndpointCtxKey
TEST_F(EndpointCtxTest, Ut_GetKey_When_ConstructedWithKey_Expect_ReturnSameKey)
{
    void* fakeCtx = reinterpret_cast<void*>(0x1000);
    EndpointCtxKey key{};
    key.devPhyId = 7;
    key.protocol = COMM_PROTOCOL_UB_CTP;

    EndpointCtx ctx{fakeCtx, key};

    EXPECT_EQ(ctx.key.devPhyId, 7u);
    EXPECT_EQ(ctx.key.protocol, COMM_PROTOCOL_UB_CTP);
}

// EndpointCtxKey operator== 与 hash: 同 key 相等，不同 key 不等，可作 unordered_map key
TEST_F(EndpointCtxTest, Ut_EndpointCtxKey_When_SameFields_Expect_EqualAndHashable)
{
    EndpointCtxKey a{};
    a.devPhyId = 3;
    a.protocol = COMM_PROTOCOL_ROCE;
    EndpointCtxKey b = a;

    EXPECT_TRUE(a == b);

    b.devPhyId = 4;
    EXPECT_FALSE(a == b);

    b = a;
    b.protocol = COMM_PROTOCOL_UB_CTP;
    EXPECT_FALSE(a == b);

    // hash 可调用（编译期验证可作为 unordered_map key）
    EndpointCtxKeyHash hasher;
    (void)hasher(a);
    (void)hasher(b);
    SUCCEED();
}

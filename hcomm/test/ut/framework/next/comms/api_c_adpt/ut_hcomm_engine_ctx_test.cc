/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../ut_hcomm_base.h"

class TestHcommEngineCtx : public TestHcommCAdptBase {
public:
    void SetUp() override { TestHcommCAdptBase::SetUp(); }
    void TearDown() override { TestHcommCAdptBase::TearDown(); }
};

TEST_F(TestHcommEngineCtx, Ut_TestHcommEngineCtxCreate_When_CPUEngine_Return_HCCL_Success)
{
    void* ctx = nullptr;
    HcommResult ret = HcommEngineCtxCreate(COMM_ENGINE_CPU, 1024, &ctx);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(ctx, nullptr);

    (void)HcommEngineCtxDestroy(COMM_ENGINE_CPU, ctx);
}

TEST_F(TestHcommEngineCtx, Ut_TestHcommEngineCtxCreate_When_SizeZero_Return_HCCL_Success)
{
    void* ctx = nullptr;
    HcommResult ret = HcommEngineCtxCreate(COMM_ENGINE_CPU, 0, &ctx);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    if (ctx != nullptr) {
        (void)HcommEngineCtxDestroy(COMM_ENGINE_CPU, ctx);
    }
}

TEST_F(TestHcommEngineCtx, Ut_TestHcommEngineCtxCreate_When_CtxNullptr_Return_HCCL_E_PTR)
{
    HcommResult ret = HcommEngineCtxCreate(COMM_ENGINE_CPU, 1024, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommEngineCtx, Ut_TestHcommEngineCtxCreate_When_UnsupportedEngine_Return_HCCL_E_PARA)
{
    void* ctx = nullptr;
    HcommResult ret = HcommEngineCtxCreate(COMM_ENGINE_RESERVED, 1024, &ctx);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcommEngineCtx, Ut_TestHcommEngineCtxDestroy_When_CPUEngine_Return_HCCL_Success)
{
    void* ctx = malloc(1024);
    EXPECT_NE(ctx, nullptr);

    HcommResult ret = HcommEngineCtxDestroy(COMM_ENGINE_CPU, ctx);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestHcommEngineCtx, Ut_TestHcommEngineCtxDestroy_When_CtxNullptr_Return_HCCL_E_PTR)
{
    HcommResult ret = HcommEngineCtxDestroy(COMM_ENGINE_CPU, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommEngineCtx, Ut_TestHcommEngineCtxDestroy_When_UnsupportedEngine_Return_HCCL_E_PARA)
{
    void* ctx = malloc(1024);
    HcommResult ret = HcommEngineCtxDestroy(COMM_ENGINE_RESERVED, ctx);
    EXPECT_EQ(ret, HCCL_E_PARA);
    free(ctx);
}

TEST_F(TestHcommEngineCtx, Ut_TestHcommEngineCtxCopy_When_CPUEngine_Return_HCCL_Success)
{
    void* srcCtx = malloc(1024);
    void* dstCtx = malloc(1024);
    EXPECT_NE(srcCtx, nullptr);
    EXPECT_NE(dstCtx, nullptr);

    HcommResult ret = HcommEngineCtxCopy(COMM_ENGINE_CPU, dstCtx, srcCtx, 1024);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    free(srcCtx);
    free(dstCtx);
}

TEST_F(TestHcommEngineCtx, Ut_TestHcommEngineCtxCopy_When_DstNullptr_Return_HCCL_E_PTR)
{
    void* srcCtx = malloc(1024);
    EXPECT_NE(srcCtx, nullptr);

    HcommResult ret = HcommEngineCtxCopy(COMM_ENGINE_CPU, nullptr, srcCtx, 1024);
    EXPECT_EQ(ret, HCCL_E_PTR);

    free(srcCtx);
}

TEST_F(TestHcommEngineCtx, Ut_TestHcommEngineCtxCopy_When_SrcNullptr_Return_HCCL_E_PTR)
{
    void* dstCtx = malloc(1024);
    EXPECT_NE(dstCtx, nullptr);

    HcommResult ret = HcommEngineCtxCopy(COMM_ENGINE_CPU, dstCtx, nullptr, 1024);
    EXPECT_EQ(ret, HCCL_E_PTR);

    free(dstCtx);
}

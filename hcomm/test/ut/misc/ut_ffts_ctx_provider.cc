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
#include <mockcpp/mockcpp.hpp>
#include "ffts_ctx_provider.h"
#include "ffts_common.h"
#include "ffts_common_pub.h"

using namespace hccl;

class FftsCtxProviderTest : public testing::Test {
public:
    static void SetUpTestCase() { std::cout << "FftsCtxProviderTest SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "FftsCtxProviderTest TearDown" << std::endl; }
    virtual void SetUp()
    {
        s32 portNum = -1;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "FftsCtxProviderTest case SetUP" << std::endl;
    }
    virtual void TearDown() { std::cout << "FftsCtxProviderTest case TearDown" << std::endl; }

    std::unique_ptr<FftsCtxProvider> fftsCtxProvider
        = std::unique_ptr<FftsCtxProvider>(new (std::nothrow) FftsCtxProvider());
};

TEST_F(FftsCtxProviderTest, should_get_an_valid_ctx_given_the_op_is_broadcast)
{
    HcclOpMetaInfo meta = HcclOpMetaInfo::GetOneForBroadcast(true, 0);
    auto ctx = fftsCtxProvider->GetFftsCtx(meta.isEnableCache, meta.GetCacheKey());
    EXPECT_NE(ctx, nullptr);
}

TEST_F(FftsCtxProviderTest, should_get_an_valid_ctx_given_the_op_is_allreduce)
{
    HcclOpMetaInfo meta = HcclOpMetaInfo::GetOneForAllReduce();
    auto ctx = fftsCtxProvider->GetFftsCtx(meta.isEnableCache, meta.GetCacheKey());
    EXPECT_NE(ctx, nullptr);
}

TEST_F(FftsCtxProviderTest, should_get_an_valid_ctx_given_the_op_is_allgather)
{
    HcclOpMetaInfo meta = HcclOpMetaInfo::GetOneForAllGather();
    auto ctx = fftsCtxProvider->GetFftsCtx(meta.isEnableCache, meta.GetCacheKey());
    EXPECT_NE(ctx, nullptr);
}

TEST_F(FftsCtxProviderTest, should_get_an_valid_ctx_given_the_op_is_reduce)
{
    HcclOpMetaInfo meta = HcclOpMetaInfo::GetOneForReduce(true, 0);
    auto ctx = fftsCtxProvider->GetFftsCtx(meta.isEnableCache, meta.GetCacheKey());
    EXPECT_NE(ctx, nullptr);
}

TEST_F(FftsCtxProviderTest, should_get_an_valid_ctx_given_the_op_is_reducescatter)
{
    HcclOpMetaInfo meta = HcclOpMetaInfo::GetOneForReduceScatter();
    auto ctx = fftsCtxProvider->GetFftsCtx(meta.isEnableCache, meta.GetCacheKey());
    EXPECT_NE(ctx, nullptr);
}

TEST_F(FftsCtxProviderTest, should_get_an_valid_ctx_given_the_op_is_alltoallv)
{
    HcclOpMetaInfo meta = HcclOpMetaInfo::GetOneForAllToAllV(CopyPattern::BCOPY, 0);
    auto ctx = fftsCtxProvider->GetFftsCtx(meta.isEnableCache, meta.GetCacheKey());
    EXPECT_NE(ctx, nullptr);
}

TEST_F(FftsCtxProviderTest, should_get_an_valid_ctx_given_the_op_is_alltoallvc)
{
    HcclOpMetaInfo meta = HcclOpMetaInfo::GetOneForAllToAllVC(CopyPattern::BCOPY, 0x8FFFFFFF);
    auto ctx = fftsCtxProvider->GetFftsCtx(meta.isEnableCache, meta.GetCacheKey());
    EXPECT_NE(ctx, nullptr);
}

TEST_F(FftsCtxProviderTest, should_get_an_valid_ctx_given_the_op_is_send)
{
    HcclOpMetaInfo meta = HcclOpMetaInfo::GetOneForSend();
    auto ctx = fftsCtxProvider->GetFftsCtx(meta.isEnableCache, meta.GetCacheKey());
    EXPECT_NE(ctx, nullptr);
}

TEST_F(FftsCtxProviderTest, should_get_an_valid_ctx_given_the_op_is_recieve)
{
    HcclOpMetaInfo meta = HcclOpMetaInfo::GetOneForReceive();
    auto ctx = fftsCtxProvider->GetFftsCtx(meta.isEnableCache, meta.GetCacheKey());
    EXPECT_NE(ctx, nullptr);
}

TEST_F(FftsCtxProviderTest, should_get_a_new_ctx_given_the_copy_pattern_is_bcopy)
{
    HcclOpMetaInfo meta = HcclOpMetaInfo::GetOneForAllToAllV(CopyPattern::BCOPY, 0);
    auto ctxPrev = fftsCtxProvider->GetFftsCtx(meta.isEnableCache, meta.GetCacheKey());
    auto ctx = fftsCtxProvider->GetFftsCtx(meta.isEnableCache, meta.GetCacheKey());
    EXPECT_NE(ctx, ctxPrev);
}

TEST_F(FftsCtxProviderTest, should_get_the_same_ctx_given_the_copy_pattern_is_zcopy)
{
    HcclOpMetaInfo meta = HcclOpMetaInfo::GetOneForAllToAllV(CopyPattern::ZCOPY, 0);
    auto ctxPrev = fftsCtxProvider->GetFftsCtx(meta.isEnableCache, meta.GetCacheKey());
    auto ctx = fftsCtxProvider->GetFftsCtx(meta.isEnableCache, meta.GetCacheKey());
    EXPECT_EQ(ctx, ctxPrev);
}

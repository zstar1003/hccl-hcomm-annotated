/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_api_base_test.h"
#include "hccl/hccl_res.h"
#include "independent_op_context_manager.h"
#include "log.h"
#include "hccl_comm_pub.h"
#include "independent_op.h"
#include <string>

#define private public

using namespace hccl;

class HcclEngineCtxGetTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        UT_USE_RANK_TABLE_910_1SERVER_1RANK;
        UT_COMM_CREATE_DEFAULT(comm);
    }
    void TearDown() override
    {
        Ut_Comm_Destroy(comm);
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

TEST_F(HcclEngineCtxGetTest, ut_HcclEngineCtxGet_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    const char* ctxTag = "1";
    CommEngine engine = COMM_ENGINE_CPU;
    void* ctx;
    uint64_t size = 256;

    HcclResult result = HcclEngineCtxCreate(comm, ctxTag, engine, size, &ctx);
    EXPECT_EQ(result, HCCL_SUCCESS);

    result = HcclEngineCtxGet(comm, ctxTag, engine, &ctx, &size);
    EXPECT_EQ(result, HCCL_SUCCESS);

    result = HcclEngineCtxDestroy(comm, ctxTag, engine);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(HcclEngineCtxGetTest, ut_HcclEngineCtxGet_When_commNULL_Expect_ReturnIsHCCL_ERROR)
{
    CommEngine engine = COMM_ENGINE_CPU;
    void* ctx;
    uint64_t size = 256;

    HcclResult result = HcclEngineCtxGet(nullptr, nullptr, engine, &ctx, &size);
    EXPECT_EQ(result, HCCL_E_PTR);
}

TEST_F(HcclEngineCtxGetTest, ut_HcclEngineCtxGet_When_EngineTagIsNull_Expect_ReturnIsHCCL_SUCCESS)
{
    CommEngine engine = COMM_ENGINE_CPU;
    void* ctx;
    uint64_t size = 256;

    HcclResult result = HcclEngineCtxCreate(comm, nullptr, engine, size, &ctx);
    EXPECT_EQ(result, HCCL_SUCCESS);

    result = HcclEngineCtxGet(comm, nullptr, engine, &ctx, &size);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(HcclEngineCtxGetTest, ut_HcclEngineCtxGet_When_ctxIsNull_Expect_ReturnIsHCCL_ERROR)
{
    const char* ctxTag = "1";
    CommEngine engine = COMM_ENGINE_CPU;
    void* ctx = nullptr;
    uint64_t size = 256;

    HcclResult result = HcclEngineCtxGet(comm, ctxTag, engine, nullptr, &size);
    EXPECT_EQ(result, HCCL_E_PTR);
}

TEST_F(HcclEngineCtxGetTest, ut_HcclEngineCtxGet_When_sizeIsNull_Expect_ReturnIsHCCL_ERROR)
{
    const char* ctxTag = "1";
    CommEngine engine = COMM_ENGINE_CPU;
    void* ctx;

    HcclResult result = HcclEngineCtxGet(comm, ctxTag, engine, &ctx, nullptr);
    EXPECT_EQ(result, HCCL_E_PTR);
}

TEST_F(HcclEngineCtxGetTest, ut_HcclEngineCtxGet_When_EngineTagIslong_Expect_ReturnIsHCCL_ERROR)
{
    char ctxTagBuffer[257];
    memset(ctxTagBuffer, '1', 256);
    ctxTagBuffer[256] = '\0';
    const char* ctxTag = ctxTagBuffer;
    CommEngine engine = COMM_ENGINE_CPU;
    void* ctx;
    uint64_t size = 256;

    HcclResult result = HcclEngineCtxGet(comm, ctxTag, engine, &ctx, &size);
    EXPECT_EQ(result, HCCL_E_PARA);
}

TEST_F(HcclEngineCtxGetTest, ut_HcclEngineCtxGet_When_NotExistCtxwithTag_Expect_ReturnIsHCCL_ERROR)
{
    const char* ctxTag = "1";
    CommEngine engine = COMM_ENGINE_CPU;
    void* ctx;
    uint64_t size = 256;

    HcclResult result = HcclEngineCtxCreate(comm, ctxTag, engine, size, &ctx);
    EXPECT_EQ(result, HCCL_SUCCESS);

    result = HcclEngineCtxGet(comm, "2", engine, &ctx, &size);
    EXPECT_EQ(result, HCCL_E_PARA);
    result = HcclEngineCtxDestroy(comm, ctxTag, engine);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(HcclEngineCtxGetTest, ut_HcclEngineCtxGet_When_NotExistCtxwithTagEngine_ReturnIsHCCL_ERROR)
{
    const char* ctxTag = "1";
    CommEngine engine = COMM_ENGINE_CPU;
    void* ctx;
    uint64_t size = 256;

    HcclResult result = HcclEngineCtxCreate(comm, ctxTag, engine, size, &ctx);
    EXPECT_EQ(result, HCCL_SUCCESS);

    result = HcclEngineCtxGet(comm, ctxTag, COMM_ENGINE_AICPU, &ctx, &size);
    EXPECT_EQ(result, HCCL_E_PARA);

    result = HcclEngineCtxDestroy(comm, ctxTag, engine);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

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
#include "mockcpp/mockcpp.hpp"

#define private public

using namespace hccl;

class HcclEngineCtxDestroyTest : public BaseInit {
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

// 测试空comm指针传入
TEST_F(HcclEngineCtxDestroyTest, Ut_HcclEngineCtxDestroy_When_InputParamNull_Expect_Return_ERROR)
{
    const char* ctxTag = "1";
    CommEngine engine = COMM_ENGINE_CPU;

    HcclResult result = HcclEngineCtxDestroy(nullptr, ctxTag, engine);
    EXPECT_EQ(result, HCCL_E_PTR);
}

// 测试空ctxTag指针传入，tag替换为空字符串，预期不报错
TEST_F(HcclEngineCtxDestroyTest, Ut_HcclEngineCtxDestroy_When_CtxTagNull_Expect_Success)
{
    CommEngine engine = COMM_ENGINE_CPU;
    void* ctx;
    uint64_t size = 256;

    HcclResult createResult = HcclEngineCtxCreate(comm, nullptr, engine, size, &ctx);
    EXPECT_EQ(createResult, HCCL_SUCCESS);

    HcclResult result = HcclEngineCtxDestroy(comm, nullptr, engine);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

// 测试销毁不存在的Context
TEST_F(HcclEngineCtxDestroyTest, Ut_HcclEngineCtxDestroy_When_TagNotExist_Expect_Return_EPARA)
{
    const char* ctxTag = "non_existent_tag";
    CommEngine engine = COMM_ENGINE_CPU;

    HcclResult result = HcclEngineCtxDestroy(comm, ctxTag, engine);
    EXPECT_EQ(result, HCCL_E_PARA);
}

// 测试Tag存在但engine不存在
TEST_F(HcclEngineCtxDestroyTest, Ut_HcclEngineCtxDestroy_When_EngineNotExistInTag_Expect_Return_EPARA)
{
    const char* ctxTag = "tag1";
    void* ctx;
    uint64_t size = 256;

    // 先创建tag1的CPU类型Context
    HcclResult createResult = HcclEngineCtxCreate(comm, ctxTag, COMM_ENGINE_CPU, size, &ctx);
    EXPECT_EQ(createResult, HCCL_SUCCESS);

    // 尝试销毁未创建的AICPU类型engine
    HcclResult destroyResult = HcclEngineCtxDestroy(comm, ctxTag, COMM_ENGINE_AICPU);
    EXPECT_EQ(destroyResult, HCCL_E_PARA);

    // 清理已创建的CPU Context
    HcclResult cleanResult = HcclEngineCtxDestroy(comm, ctxTag, COMM_ENGINE_CPU);
    EXPECT_EQ(cleanResult, HCCL_SUCCESS);
}

// 测试销毁Host类型内存
TEST_F(HcclEngineCtxDestroyTest, Ut_HcclEngineCtxDestroy_When_SuccessDestroyHostMem_Expect_Success)
{
    const char* ctxTag = "host_tag";
    void* ctx;
    uint64_t size = 256;

    // 创建COMM_ENGINE_CPU类型的Context
    HcclResult createResult = HcclEngineCtxCreate(comm, ctxTag, COMM_ENGINE_CPU, size, &ctx);
    EXPECT_EQ(createResult, HCCL_SUCCESS);

    // 销毁Host类型内存
    HcclResult destroyResult = HcclEngineCtxDestroy(comm, ctxTag, COMM_ENGINE_CPU);
    EXPECT_EQ(destroyResult, HCCL_SUCCESS);
}

// 测试销毁Device类型内存
TEST_F(HcclEngineCtxDestroyTest, Ut_HcclEngineCtxDestroy_When_SuccessDestroyDeviceMem_Expect_Success)
{
    const char* ctxTag = "device_tag";
    void* ctx;
    uint64_t size = 256;

    // 创建COMM_ENGINE_AICPU类型的Context
    HcclResult createResult = HcclEngineCtxCreate(comm, ctxTag, COMM_ENGINE_AICPU, size, &ctx);
    EXPECT_EQ(createResult, HCCL_SUCCESS);

    // 销毁Device类型内存
    HcclResult destroyResult = HcclEngineCtxDestroy(comm, ctxTag, COMM_ENGINE_AICPU);
    EXPECT_EQ(destroyResult, HCCL_SUCCESS);
}

// 测试同tag下多engine销毁
TEST_F(HcclEngineCtxDestroyTest, Ut_HcclEngineCtxDestroy_When_MultipleEnginesUnderSameTag_Expect_OnlySpecifiedDeleted)
{
    const char* ctxTag = "multi_engine_tag";
    void* cpuCtx;
    void* aicpuCtx;
    uint64_t size = 256;

    // 创建tag1的CPU和AICPU两种Context
    HcclResult createCpuResult = HcclEngineCtxCreate(comm, ctxTag, COMM_ENGINE_CPU, size, &cpuCtx);
    EXPECT_EQ(createCpuResult, HCCL_SUCCESS);

    HcclResult createAicpuResult = HcclEngineCtxCreate(comm, ctxTag, COMM_ENGINE_AICPU, size, &aicpuCtx);
    EXPECT_EQ(createAicpuResult, HCCL_SUCCESS);

    // 只销毁CPU类型的engine
    HcclResult destroyCpuResult = HcclEngineCtxDestroy(comm, ctxTag, COMM_ENGINE_CPU);
    EXPECT_EQ(destroyCpuResult, HCCL_SUCCESS);

    // 验证AICPU内存仍存在（尝试销毁AICPU应该成功）
    HcclResult destroyAicpuResult = HcclEngineCtxDestroy(comm, ctxTag, COMM_ENGINE_AICPU);
    EXPECT_EQ(destroyAicpuResult, HCCL_SUCCESS);
}

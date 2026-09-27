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

using namespace std;
using namespace hccl;

constexpr const char* RANKTABLE_FILE_NAME = "./ut_independent_op_test.json";

class HcclCommEngineCtxTest : public BaseInit {
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

// HcclMem构造
static inline HcclMem MakeMem(void* addr, uint64_t size, HcclMemType type)
{
    HcclMem m{};
    m.addr = addr;
    m.size = size;
    m.type = type;
    return m;
}

// HcclMem相等
static inline bool IsMemEqual(const HcclMem& lhm, const HcclMem& rhm)
{
    return lhm.type == rhm.type && lhm.addr == rhm.addr && lhm.size == rhm.size;
}

// 创建入参异常校验：comm opTag engineCtx null判断 engineCtx->size 0判断 设备类型支持判断 tag最大长度判断
// 获取入参异常校验：comm opTag null判断 tag最大长度判断
// 销毁入参异常校验：comm engineCtx null判断
TEST_F(HcclCommEngineCtxTest, Ut_CommEngineCtx_When_Error_Input_Expect_Return_ERROR)
{
    // create: comm null
    uint64_t size = 16;
    void* ctx = nullptr;
    HcclResult ret = HcclEngineCtxCreate(nullptr, "opTag", CommEngine::COMM_ENGINE_CPU, size, &ctx);
    EXPECT_EQ(ret, HCCL_E_PTR);

    // create: Ctx null
    ret = HcclEngineCtxCreate(comm, "opTag", CommEngine::COMM_ENGINE_CPU, size, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);

    // create: engine unsupported
    ret = HcclEngineCtxCreate(comm, "opTag", CommEngine::COMM_ENGINE_RESERVED, size, &ctx);
    EXPECT_EQ(ret, HCCL_E_PARA);

    // create: engineCtx->size 0
    ret = HcclEngineCtxCreate(comm, "opTag", CommEngine::COMM_ENGINE_CPU, 0, &ctx);
    EXPECT_EQ(ret, HCCL_E_PARA);

    // create: opTag too long
    char opTag[HCCL_RES_TAG_MAX_LEN + 2];
    memset(opTag, 'A', HCCL_RES_TAG_MAX_LEN + 1);
    opTag[HCCL_RES_TAG_MAX_LEN + 1] = '\0';
    ret = HcclEngineCtxCreate(comm, opTag, CommEngine::COMM_ENGINE_CPU, size, &ctx);
    EXPECT_EQ(ret, HCCL_E_PARA);

    // get: comm null
    ret = HcclEngineCtxGet(nullptr, "opTag", CommEngine::COMM_ENGINE_CPU, &ctx, &size);
    EXPECT_EQ(ret, HCCL_E_PTR);

    // get: opTag too long
    ret = HcclEngineCtxGet(comm, opTag, CommEngine::COMM_ENGINE_CPU, &ctx, &size);
    EXPECT_EQ(ret, HCCL_E_PARA);

    // destroy: comm null
    ret = HcclEngineCtxDestroy(nullptr, "opTag", CommEngine::COMM_ENGINE_CPU);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// 重复创建异常校验 HCCL_E_PARA
TEST_F(HcclCommEngineCtxTest, Ut_HcclEngineCtxCreate_When_Duplicate_Creation_Expect_Return_EPARA)
{
    uint64_t size = 16;
    void* ctx = nullptr;
    HcclResult ret = HcclEngineCtxCreate(comm, "opTag", CommEngine::COMM_ENGINE_CPU, size, &ctx);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(ctx, nullptr);

    ret = HcclEngineCtxCreate(comm, "opTag", CommEngine::COMM_ENGINE_CPU, size, &ctx);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = HcclEngineCtxDestroy(comm, "opTag", CommEngine::COMM_ENGINE_CPU);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 未创建时获取失败校验 HCCL_SUCCESS
TEST_F(HcclCommEngineCtxTest, Ut_HcclEngineCtxGet_When_Uncreated_Expect_Return_SUCCESS)
{
    uint64_t size = 0;
    void* ctx = nullptr;
    HcclResult ret = HcclEngineCtxGet(comm, "opTag", CommEngine::COMM_ENGINE_CPU, &ctx, &size);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// 获取时设备不支持或不匹配校验 HCCL_E_PARA
TEST_F(HcclCommEngineCtxTest, Ut_HcclEngineCtxGet_When_Error_Engine_Expect_Return_EPARA)
{
    uint64_t size = 16;
    void* ctx = nullptr;
    HcclResult ret = HcclEngineCtxCreate(comm, "opTag", CommEngine::COMM_ENGINE_CPU, size, &ctx);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(ctx, nullptr);

    HcclMem* engineCtxPtr;
    ret = HcclEngineCtxGet(comm, "opTag", CommEngine::COMM_ENGINE_AICPU, &ctx, &size);
    EXPECT_EQ(ret, HCCL_E_PARA);
    ret = HcclEngineCtxGet(comm, "opTag", CommEngine::COMM_ENGINE_CCU, &ctx, &size);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = HcclEngineCtxDestroy(comm, "opTag", CommEngine::COMM_ENGINE_CPU);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 未创建时销毁失败校验 HCCL_E_PARA
TEST_F(HcclCommEngineCtxTest, Ut_HcclEngineCtxDestroy_When_Uncreated_Expect_Return_EPARA)
{
    HcclResult ret = HcclEngineCtxDestroy(comm, "opTag", CommEngine::COMM_ENGINE_CPU);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// 正常申请、获取和销毁host侧Context HCCL_SUCCESS
TEST_F(HcclCommEngineCtxTest, Ut_CommEngineCtx_When_Host_Expect_Return_SUCCESS)
{
    uint64_t size = 16;
    void* ctx = nullptr;
    HcclResult ret = HcclEngineCtxCreate(comm, "opTag", CommEngine::COMM_ENGINE_CPU, size, &ctx);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(ctx, nullptr);

    void* ctxGet;
    uint64_t sizeGet = 0;
    ret = HcclEngineCtxGet(comm, "opTag", CommEngine::COMM_ENGINE_CPU, &ctxGet, &sizeGet);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclMem createCtx = MakeMem(ctx, 16, HCCL_MEM_TYPE_HOST);
    HcclMem getCtx = MakeMem(ctxGet, sizeGet, HCCL_MEM_TYPE_HOST);
    EXPECT_EQ(IsMemEqual(createCtx, getCtx), true);

    ret = HcclEngineCtxDestroy(comm, "opTag", CommEngine::COMM_ENGINE_CPU);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 正常申请、获取和销毁device侧Context HCCL_SUCCESS
TEST_F(HcclCommEngineCtxTest, Ut_CommEngineCtx_When_Device_Expect_Return_SUCCESS)
{
    uint64_t size = 16;
    void* ctx = nullptr;
    HcclResult ret = HcclEngineCtxCreate(comm, "opTag", CommEngine::COMM_ENGINE_AICPU, size, &ctx);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(ctx, nullptr);

    void* ctxGet;
    uint64_t sizeGet = 0;
    ret = HcclEngineCtxGet(comm, "opTag", CommEngine::COMM_ENGINE_AICPU, &ctxGet, &sizeGet);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclMem createCtx = MakeMem(ctx, 16, HCCL_MEM_TYPE_HOST);
    HcclMem getCtx = MakeMem(ctxGet, sizeGet, HCCL_MEM_TYPE_HOST);
    EXPECT_EQ(IsMemEqual(createCtx, getCtx), true);

    ret = HcclEngineCtxDestroy(comm, "opTag", CommEngine::COMM_ENGINE_AICPU);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

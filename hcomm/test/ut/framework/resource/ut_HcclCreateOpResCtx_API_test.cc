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
#include "log.h"

#define private public

using namespace hccl;

class HcclCreateOpResCtxTest : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        UT_USE_RANK_TABLE_910_1SERVER_1RANK;
    }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

HcclResult hrtGetDeviceTypeStub91093(DevType& devType)
{
    devType = DevType::DEV_TYPE_910_93;
    return HCCL_SUCCESS;
}

HcclResult hrtGetDeviceTypeStub910B(DevType& devType)
{
    devType = DevType::DEV_TYPE_910B;
    return HCCL_SUCCESS;
}
HcclResult hrtGetDeviceTypeStub91095(DevType& devType)
{
    devType = DevType::DEV_TYPE_950;
    return HCCL_SUCCESS;
}

TEST_F(HcclCreateOpResCtxTest, ut_HcclCreateOpResCtx_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    UT_COMM_CREATE_DEFAULT(comm);
    uint8_t opType = 2;
    HcclDataType srcDataType = HCCL_DATA_TYPE_FP16;
    HcclDataType dstDataType = HCCL_DATA_TYPE_FP16;
    HcclReduceOp reduceType = HCCL_REDUCE_SUM;
    uint64_t count = 256;
    char algConfig[128] = "AllReduce=level0:ring";
    CommEngine engine = COMM_ENGINE_AIV;
    void* ctx;

    MOCKER(hrtGetDeviceType).stubs().will(invoke(hrtGetDeviceTypeStub91093));
    MOCKER(hrtStreamSetMode).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::AllocComResourceByTiling).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult result
        = HcclCreateOpResCtxInner(comm, opType, srcDataType, dstDataType, reduceType, count, algConfig, engine, &ctx);
    EXPECT_EQ(result, HCCL_SUCCESS);

    Ut_Comm_Destroy(comm);
    GlobalMockObject::verify();
}

TEST_F(HcclCreateOpResCtxTest, ut_HcclCreateOpResCtx_When_ParamIsNullptr_Expect_ReturnIsHCCL_E_PTR)
{
    UT_COMM_CREATE_DEFAULT(comm);
    uint8_t opType = 2;
    HcclDataType srcDataType = HCCL_DATA_TYPE_FP16;
    HcclDataType dstDataType = HCCL_DATA_TYPE_FP16;
    HcclReduceOp reduceType = HCCL_REDUCE_SUM;
    uint64_t count = 256;
    char algConfig[128] = "AllReduce=level0:ring";
    CommEngine engine = COMM_ENGINE_AIV;
    void* ctx;

    MOCKER(hrtGetDeviceType).stubs().will(invoke(hrtGetDeviceTypeStub91093));

    HcclResult result = HcclCreateOpResCtxInner(
        nullptr, opType, srcDataType, dstDataType, reduceType, count, algConfig, engine, &ctx);
    EXPECT_EQ(result, HCCL_E_PTR);

    result = HcclCreateOpResCtxInner(comm, opType, srcDataType, dstDataType, reduceType, count, nullptr, engine, &ctx);
    EXPECT_EQ(result, HCCL_E_PTR);

    result = HcclCreateOpResCtxInner(
        comm, opType, srcDataType, dstDataType, reduceType, count, algConfig, engine, nullptr);
    EXPECT_EQ(result, HCCL_E_PTR);

    Ut_Comm_Destroy(comm);
    GlobalMockObject::verify();
}

TEST_F(HcclCreateOpResCtxTest, ut_HcclCreateOpResCtx_When_DevTypeIs91095_Expect_ReturnIsHCCL_E_NOT_SUPPORT)
{
    UT_COMM_CREATE_DEFAULT(comm);
    uint8_t opType = 2;
    HcclDataType srcDataType = HCCL_DATA_TYPE_FP16;
    HcclDataType dstDataType = HCCL_DATA_TYPE_FP16;
    HcclReduceOp reduceType = HCCL_REDUCE_SUM;
    uint64_t count = 256;
    char algConfig[128] = "AllReduce=level0:ring";
    CommEngine engine = COMM_ENGINE_AIV;
    void* ctx;

    MOCKER(hrtGetDeviceType).stubs().will(invoke(hrtGetDeviceTypeStub91095));

    HcclResult result
        = HcclCreateOpResCtxInner(comm, opType, srcDataType, dstDataType, reduceType, count, algConfig, engine, &ctx);
    EXPECT_EQ(result, HCCL_E_NOT_SUPPORT);

    Ut_Comm_Destroy(comm);
    GlobalMockObject::verify();
}

TEST_F(HcclCreateOpResCtxTest, ut_HcclCreateOpResCtx_A2When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    UT_COMM_CREATE_DEFAULT(comm);
    uint8_t opType = 2;
    HcclDataType srcDataType = HCCL_DATA_TYPE_FP16;
    HcclDataType dstDataType = HCCL_DATA_TYPE_FP16;
    HcclReduceOp reduceType = HCCL_REDUCE_SUM;
    uint64_t count = 256;
    char algConfig[128] = "AllReduce=level0:ring";
    CommEngine engine = COMM_ENGINE_AICPU;
    void* ctx;

    MOCKER(hrtGetDeviceType).stubs().will(invoke(hrtGetDeviceTypeStub910B));
    MOCKER(hrtStreamSetMode).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::AllocComResourceByTiling).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::Mc2CreateAndLaunchContext).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult result
        = HcclCreateOpResCtxInner(comm, opType, srcDataType, dstDataType, reduceType, count, algConfig, engine, &ctx);
    EXPECT_EQ(result, HCCL_SUCCESS);

    Ut_Comm_Destroy(comm);
    GlobalMockObject::verify();
}

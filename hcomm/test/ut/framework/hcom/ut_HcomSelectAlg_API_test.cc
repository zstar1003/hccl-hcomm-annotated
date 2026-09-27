/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include <memory>
#include <cstring>

#include "hccl/base.h"
#include "hccl/hccl_types.h"
#include "hcom_pub.h"
#include "hcom_common.h"
#include "comm_config_pub.h"
#include "common.h"

using namespace hccl;

/**
 * @brief HcomSelectAlg 函数测试类
 */
class HcomSelectAlgTest : public testing::Test {
protected:
    void SetUp() override
    {
        comm.reset(new (std::nothrow) hccl::hcclComm());
        if (!comm) {
            HCCL_ERROR("Failed to create hccl::hcclComm");
            return;
        }

        // 初始化测试参数
        count = 1024;
        counts = nullptr;
        dataType = HCCL_DATA_TYPE_FP32;
        reduceOp = HCCL_REDUCE_SUM;
        opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
        aivCoreLimit = 0;
        ifAiv = false;
        memset(algName, 0, ALG_NAME_MAX_LEN);
    }

    void TearDown() override { GlobalMockObject::verify(); }

    std::shared_ptr<hccl::hcclComm> comm;
    u64 count;
    void* counts;
    HcclDataType dataType;
    HcclReduceOp reduceOp;
    HcclCMDType opType;
    int32_t aivCoreLimit;
    bool ifAiv;
    char algName[ALG_NAME_MAX_LEN];
};

// 测试：ifAiv 为空指针，期望返回 HCCL_E_PTR
TEST_F(HcomSelectAlgTest, Ut_HcomSelectAlg_When_IfAivIsNull_Expect_PtrError)
{
    s64 testComm = static_cast<s64>(0x12345678); // 非默认 comm 值

    HcclResult result = HcomSelectAlg(
        testComm, HCCL_WORLD_GROUP, count, counts, dataType, reduceOp, opType, aivCoreLimit, nullptr, algName);

    EXPECT_EQ(result, HCCL_E_PTR);
}

// 测试：使用非默认 comm 值，正常选择算法
TEST_F(HcomSelectAlgTest, Ut_HcomSelectAlg_When_CommIsNotDefault_Expect_Success)
{
    s64 testComm = reinterpret_cast<s64>(comm.get());

    // Mock hcclComm::HcclSelectAlg 方法
    MOCKER_CPP(&hcclComm::HcclSelectAlg).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult result = HcomSelectAlg(
        testComm, HCCL_WORLD_GROUP, count, counts, dataType, reduceOp, opType, aivCoreLimit, &ifAiv, algName);

    EXPECT_EQ(result, HCCL_SUCCESS);
}

// 测试：使用默认 comm 值，通过 group 获取通信域
TEST_F(HcomSelectAlgTest, Ut_HcomSelectAlg_When_CommIsDefault_Expect_Success)
{
    s64 testComm = static_cast<s64>(CommNumHcom::COMM_VALUE_DEFAULT);
    const char* testGroup = "hccl_world_group";

    // Mock HcomGetCommByGroup
    MOCKER(HcomGetCommByGroup).stubs().with(mockcpp::any(), outBound(comm)).will(returnValue(HCCL_SUCCESS));

    // Mock hcclComm::HcclSelectAlg 方法
    MOCKER_CPP(&hcclComm::HcclSelectAlg).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult result
        = HcomSelectAlg(testComm, testGroup, count, counts, dataType, reduceOp, opType, aivCoreLimit, &ifAiv, algName);

    EXPECT_EQ(result, HCCL_SUCCESS);
}

// 测试：group 为 nullptr，使用默认 group
TEST_F(HcomSelectAlgTest, Ut_HcomSelectAlg_When_GroupIsNull_Expect_Success)
{
    s64 testComm = static_cast<s64>(CommNumHcom::COMM_VALUE_DEFAULT);

    // Mock HcomGetCommByGroup
    MOCKER(HcomGetCommByGroup).stubs().with(mockcpp::any(), outBound(comm)).will(returnValue(HCCL_SUCCESS));

    // Mock hcclComm::HcclSelectAlg 方法
    MOCKER_CPP(&hcclComm::HcclSelectAlg).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult result
        = HcomSelectAlg(testComm, nullptr, count, counts, dataType, reduceOp, opType, aivCoreLimit, &ifAiv, algName);

    EXPECT_EQ(result, HCCL_SUCCESS);
}

// 测试：HcomGetCommByGroup 失败
TEST_F(HcomSelectAlgTest, Ut_HcomSelectAlg_When_GetCommByGroupFail_Expect_Error)
{
    s64 testComm = static_cast<s64>(CommNumHcom::COMM_VALUE_DEFAULT);
    const char* testGroup = "invalid_group";

    // Mock HcomGetCommByGroup 返回失败
    MOCKER(HcomGetCommByGroup).stubs().with(mockcpp::any(), outBound(comm)).will(returnValue(HCCL_E_NOT_FOUND));

    HcclResult result
        = HcomSelectAlg(testComm, testGroup, count, counts, dataType, reduceOp, opType, aivCoreLimit, &ifAiv, algName);

    EXPECT_EQ(result, HCCL_E_NOT_FOUND);
}

// 测试：hcclComm::HcclSelectAlg 失败
TEST_F(HcomSelectAlgTest, Ut_HcomSelectAlg_When_HcclSelectAlgFail_Expect_Error)
{
    s64 testComm = reinterpret_cast<s64>(comm.get());

    // Mock hcclComm::HcclSelectAlg 返回失败
    MOCKER_CPP(&hcclComm::HcclSelectAlg).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_INTERNAL));

    HcclResult result = HcomSelectAlg(
        testComm, HCCL_WORLD_GROUP, count, counts, dataType, reduceOp, opType, aivCoreLimit, &ifAiv, algName);

    EXPECT_EQ(result, HCCL_E_INTERNAL);
}

// 测试：不同的操作类型 - ALLGATHER
TEST_F(HcomSelectAlgTest, Ut_HcomSelectAlg_When_OpTypeIsAllGather_Expect_Success)
{
    s64 testComm = reinterpret_cast<s64>(comm.get());
    opType = HcclCMDType::HCCL_CMD_ALLGATHER;

    // Mock hcclComm::HcclSelectAlg 方法
    MOCKER_CPP(&hcclComm::HcclSelectAlg).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult result = HcomSelectAlg(
        testComm, HCCL_WORLD_GROUP, count, counts, dataType, reduceOp, opType, aivCoreLimit, &ifAiv, algName);

    EXPECT_EQ(result, HCCL_SUCCESS);
}

// 测试：不同的操作类型 - REDUCE_SCATTER
TEST_F(HcomSelectAlgTest, Ut_HcomSelectAlg_When_OpTypeIsReduceScatter_Expect_Success)
{
    s64 testComm = reinterpret_cast<s64>(comm.get());
    opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;

    // Mock hcclComm::HcclSelectAlg 方法
    MOCKER_CPP(&hcclComm::HcclSelectAlg).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult result = HcomSelectAlg(
        testComm, HCCL_WORLD_GROUP, count, counts, dataType, reduceOp, opType, aivCoreLimit, &ifAiv, algName);

    EXPECT_EQ(result, HCCL_SUCCESS);
}

// 测试：不同的规约操作 - MAX
TEST_F(HcomSelectAlgTest, Ut_HcomSelectAlg_When_ReduceOpIsMax_Expect_Success)
{
    s64 testComm = reinterpret_cast<s64>(comm.get());
    reduceOp = HCCL_REDUCE_MAX;

    // Mock hcclComm::HcclSelectAlg 方法
    MOCKER_CPP(&hcclComm::HcclSelectAlg).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult result = HcomSelectAlg(
        testComm, HCCL_WORLD_GROUP, count, counts, dataType, reduceOp, opType, aivCoreLimit, &ifAiv, algName);

    EXPECT_EQ(result, HCCL_SUCCESS);
}

// 测试：不同的数据类型 - INT32
TEST_F(HcomSelectAlgTest, Ut_HcomSelectAlg_When_DataTypeIsInt32_Expect_Success)
{
    s64 testComm = reinterpret_cast<s64>(comm.get());
    dataType = HCCL_DATA_TYPE_INT32;

    // Mock hcclComm::HcclSelectAlg 方法
    MOCKER_CPP(&hcclComm::HcclSelectAlg).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult result = HcomSelectAlg(
        testComm, HCCL_WORLD_GROUP, count, counts, dataType, reduceOp, opType, aivCoreLimit, &ifAiv, algName);

    EXPECT_EQ(result, HCCL_SUCCESS);
}

// 测试：count 为 0
TEST_F(HcomSelectAlgTest, Ut_HcomSelectAlg_When_CountIsZero_Expect_Success)
{
    s64 testComm = reinterpret_cast<s64>(comm.get());
    count = 0;

    // Mock hcclComm::HcclSelectAlg 方法
    MOCKER_CPP(&hcclComm::HcclSelectAlg).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult result = HcomSelectAlg(
        testComm, HCCL_WORLD_GROUP, count, counts, dataType, reduceOp, opType, aivCoreLimit, &ifAiv, algName);

    EXPECT_EQ(result, HCCL_SUCCESS);
}

// 测试：aivCoreLimit 非 0
TEST_F(HcomSelectAlgTest, Ut_HcomSelectAlg_When_AivCoreLimitIsNonZero_Expect_Success)
{
    s64 testComm = reinterpret_cast<s64>(comm.get());
    aivCoreLimit = 8;

    // Mock hcclComm::HcclSelectAlg 方法
    MOCKER_CPP(&hcclComm::HcclSelectAlg).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult result = HcomSelectAlg(
        testComm, HCCL_WORLD_GROUP, count, counts, dataType, reduceOp, opType, aivCoreLimit, &ifAiv, algName);

    EXPECT_EQ(result, HCCL_SUCCESS);
}

// 测试：counts 为 nullptr
TEST_F(HcomSelectAlgTest, Ut_HcomSelectAlg_When_CountsIsNull_Expect_Success)
{
    s64 testComm = reinterpret_cast<s64>(comm.get());
    counts = nullptr;

    // Mock hcclComm::HcclSelectAlg 方法
    MOCKER_CPP(&hcclComm::HcclSelectAlg).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult result = HcomSelectAlg(
        testComm, HCCL_WORLD_GROUP, count, counts, dataType, reduceOp, opType, aivCoreLimit, &ifAiv, algName);

    EXPECT_EQ(result, HCCL_SUCCESS);
}

// 测试：使用默认 comm 值，hcclComm::HcclSelectAlg 失败
TEST_F(HcomSelectAlgTest, Ut_HcomSelectAlg_When_DefaultCommAndSelectAlgFail_Expect_Error)
{
    s64 testComm = static_cast<s64>(CommNumHcom::COMM_VALUE_DEFAULT);
    const char* testGroup = "hccl_world_group";

    // Mock HcomGetCommByGroup
    MOCKER(HcomGetCommByGroup).stubs().with(mockcpp::any(), outBound(comm)).will(returnValue(HCCL_SUCCESS));

    // Mock hcclComm::HcclSelectAlg 返回失败
    MOCKER_CPP(&hcclComm::HcclSelectAlg).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_PARA));

    HcclResult result
        = HcomSelectAlg(testComm, testGroup, count, counts, dataType, reduceOp, opType, aivCoreLimit, &ifAiv, algName);

    EXPECT_EQ(result, HCCL_E_PARA);
}

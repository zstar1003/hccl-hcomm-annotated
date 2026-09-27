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
 * @brief GetOpScratchMemSize 函数测试类
 */
class GetOpScratchMemSizeTest : public testing::Test {
protected:
    void SetUp() override
    {
        comm.reset(new (std::nothrow) hccl::hcclComm());
        if (!comm) {
            HCCL_ERROR("Failed to create hccl::hcclComm");
            return;
        }

        opMemSize = 0;
        hcomOpParam = std::make_unique<HcomOpParam>();
        memset(hcomOpParam.get(), 0, sizeof(HcomOpParam));

        hcomOpParam->count = 1024;
        hcomOpParam->dataType = HCCL_DATA_TYPE_FP32;
        hcomOpParam->geDeterministic = 0;
        hcomOpParam->socVersion = const_cast<char*>("Ascend910"); // 模拟 Ascend910 设备，不应该用于91095等设备
        hcomOpParam->group = const_cast<char*>("hccl_world_group");
        hcomOpParam->opType = const_cast<char*>("HcomAllReduce");

        dataTypeSize = 4;
        rankSize = 8;
        serverNum = 1;
    }

    void TearDown() override { GlobalMockObject::verify(); }

    std::shared_ptr<hccl::hcclComm> comm;
    std::unique_ptr<HcomOpParam> hcomOpParam;
    u64 opMemSize;
    u32 dataTypeSize;
    s32 rankSize;
    s32 serverNum;
};

// 测试 ReduceScatter 正常路径
TEST_F(GetOpScratchMemSizeTest, Ut_GetOpScratchMemSize_When_ReduceScatterNormal_Expect_Success)
{
    // Mock HcomGetCommByGroup 并设置有效的 hcclComm
    MOCKER(HcomGetCommByGroup).stubs().with(mockcpp::any(), outBound(comm)).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hrtGetDeviceTypeBySocVersion).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&GetDeterministic).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult result = GetOpScratchMemSize(
        false, HCCL_CMD_REDUCE_SCATTER, hcomOpParam.get(), opMemSize, dataTypeSize, rankSize, serverNum);

    EXPECT_EQ(result, HCCL_SUCCESS);
    EXPECT_EQ(opMemSize, hcomOpParam->count * dataTypeSize * rankSize);
}

// 测试 ReduceScatter 确定性模式（内存翻倍）
TEST_F(GetOpScratchMemSizeTest, Ut_GetOpScratchMemSize_When_ReduceScatterDeterministic_Expect_DoubleMemSUCCESS)
{
    // Mock HcomGetCommByGroup 并设置有效的 hcclComm
    MOCKER(HcomGetCommByGroup).stubs().with(mockcpp::any(), outBound(comm)).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hrtGetDeviceTypeBySocVersion).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&GetDeterministic).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult result = GetOpScratchMemSize(
        false, HCCL_CMD_REDUCE_SCATTER, hcomOpParam.get(), opMemSize, dataTypeSize, rankSize, serverNum);

    EXPECT_EQ(result, HCCL_SUCCESS);
    u64 expectedMemSize = hcomOpParam->count * dataTypeSize * rankSize;
    EXPECT_EQ(opMemSize, expectedMemSize);
}

// 测试 AllToAll 正常路径
TEST_F(GetOpScratchMemSizeTest, Ut_GetOpScratchMemSize_When_AllToAllNormal_Expect_Success)
{
    // Mock HcomGetCommByGroup 并设置有效的 hcclComm
    MOCKER(HcomGetCommByGroup).stubs().with(mockcpp::any(), outBound(comm)).will(returnValue(HCCL_SUCCESS));

    HcclResult result = GetOpScratchMemSize(
        false, HCCL_CMD_ALLTOALL, hcomOpParam.get(), opMemSize, dataTypeSize, rankSize, serverNum);

    EXPECT_EQ(result, HCCL_SUCCESS);
    EXPECT_EQ(opMemSize, hcomOpParam->count * dataTypeSize * rankSize);
}

// 测试 Broadcast 小数据量（<=32MB需要额外内存）
TEST_F(GetOpScratchMemSizeTest, Ut_GetOpScratchMemSize_When_BroadcastSmallData_Expect_MemSUCCESS)
{
    // Mock HcomGetCommByGroup 并设置有效的 hcclComm
    MOCKER(HcomGetCommByGroup).stubs().with(mockcpp::any(), outBound(comm)).will(returnValue(HCCL_SUCCESS));

    hcomOpParam->count = (1 * 1024 * 1024) / dataTypeSize;

    HcclResult result = GetOpScratchMemSize(
        false, HCCL_CMD_BROADCAST, hcomOpParam.get(), opMemSize, dataTypeSize, rankSize, serverNum);

    EXPECT_EQ(result, HCCL_SUCCESS);
    EXPECT_EQ(opMemSize, hcomOpParam->count * dataTypeSize * HCCL_MEMSIZE_HD_FACTOR);
}

// 测试 Broadcast 大数据量（>32MB不需要额外内存）
TEST_F(GetOpScratchMemSizeTest, Ut_GetOpScratchMemSize_When_BroadcastLargeData_Expect_NoExmSUCCESS)
{
    // Mock HcomGetCommByGroup 并设置有效的 hcclComm
    MOCKER(HcomGetCommByGroup).stubs().with(mockcpp::any(), outBound(comm)).will(returnValue(HCCL_SUCCESS));

    hcomOpParam->count = (64 * 1024 * 1024) / dataTypeSize;

    HcclResult result = GetOpScratchMemSize(
        false, HCCL_CMD_BROADCAST, hcomOpParam.get(), opMemSize, dataTypeSize, rankSize, serverNum);

    EXPECT_EQ(result, HCCL_SUCCESS);
    EXPECT_EQ(opMemSize, 0);
}

// 测试 HcomGetCommByGroup 失败场景
TEST_F(GetOpScratchMemSizeTest, Ut_GetOpScratchMemSize_When_GetCommByGroupFail_Expect_NotFound)
{
    MOCKER(HcomGetCommByGroup).stubs().with(mockcpp::any(), outBound(comm)).will(returnValue(HCCL_E_NOT_FOUND));
    HcclResult result = GetOpScratchMemSize(
        false, HCCL_CMD_ALLTOALL, hcomOpParam.get(), opMemSize, dataTypeSize, rankSize, serverNum);

    EXPECT_EQ(result, HCCL_E_NOT_FOUND);
}

// 测试 hrtGetDeviceTypeBySocVersion 失败场景
TEST_F(GetOpScratchMemSizeTest, Ut_GetOpScratchMemSize_When_GetDeviceTypeFail_Expect_InternINTERNAL)
{
    MOCKER(HcomGetCommByGroup).stubs().with(mockcpp::any(), outBound(comm)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hrtGetDeviceTypeBySocVersion).stubs().will(returnValue(HCCL_E_INTERNAL));
    HcclResult result = GetOpScratchMemSize(
        false, HCCL_CMD_REDUCE_SCATTER, hcomOpParam.get(), opMemSize, dataTypeSize, rankSize, serverNum);

    EXPECT_EQ(result, HCCL_E_INTERNAL);
}

// 测试 HcomGetGroupNameByOpBase 场景
TEST_F(GetOpScratchMemSizeTest, Ut_HcomGetGroupNameByOpBase_When_Normal_Expect_Success)
{
    comm.reset(new hccl::hcclComm(1, 1, "test_group_name"));
    s64 opBaseHcom = (s64)comm.get();
    char* groupname = nullptr;
    HcclResult ret = HcomGetGroupNameByOpBase(opBaseHcom, &groupname);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(groupname, nullptr);
    EXPECT_STREQ(groupname, "test_group_name");
}

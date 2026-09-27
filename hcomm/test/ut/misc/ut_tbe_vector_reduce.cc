/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>
#include <stdio.h>
#include <algorithm>
#include <unordered_map>
#include <chrono>
#include <stdarg.h>
#include <string>
#include <exception>
#include <sys/socket.h>
#include <vector>
#include <climits>
#include <hccl/base.h>
#include <hccl/hccl_types.h>
#include <semaphore.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <sys/ioctl.h>
#include <list>
#include <vector>
#include <mutex>
#include "mmpa_api.h"
#include "nlohmann/json.hpp"

#include <driver/ascend_hal.h>
#include "exe_graph/runtime/tiling_parse_context.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_data.h"

#define private public
#define protected public
#include "eletwise_v3.h"
#include "op_json_info.h"
#include "op_tiling.h"
#include "vector_tiling.h"
#include "tbe_vector_reduce.h"
#include "tbe_crack_cleared.h"
#include "tbe_gatherv2_aicore.h"
#include "tbe_unsorted_segment_sum_aicore.h"
#include "auto_tiling_rt2.h"
#include "vector_op_info.h"
#include "auto_tiling_register.h"
#include "vector_tiling_handle.h"
#include "vector_tiling_rt2.h"
#include "fusion.h"
#include "auto_tiling_context.h"

#undef protected
#undef private

using namespace std;
using namespace TbeReduce;

class TbeVectorReduceTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--TbeVectorReduceTest SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--TbeVectorReduceTest TearDown--\033[0m" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }
};

#if 1

TEST_F(TbeVectorReduceTest, EletwistV3_test_need_multi_core1)
{
    TbeReduce::TbeVectorReduce tbeTest;
    tbeTest.deviceType_ = LegacyDevType::DEV_TYPE_910;
    MOCKER_CPP(&TbeReduce::TbeVectorReduce::LoadOpBinary).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&TbeReduce::TbeVectorReduce::GetTilingDataDevMem)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(LaunchKernelWithConfig).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(BinaryLoadFromData).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(GetDeviceType).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    tbeTest.Init();
    int test = 100;
    tbeTest.Run(&test, &test, 4096, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_PROD, &test, &test);
    GlobalMockObject::verify();
}

TEST_F(TbeVectorReduceTest, EletwistV3_test_need_multi_core2)
{
    TbeReduce::TbeVectorReduce tbeTest;
    tbeTest.deviceType_ = LegacyDevType::DEV_TYPE_910;
    MOCKER_CPP(&TbeReduce::TbeVectorReduce::LoadOpBinary).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&TbeReduce::TbeVectorReduce::GetTilingDataDevMem)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(LaunchKernelWithConfig).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(BinaryLoadFromData).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(GetDeviceType).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    tbeTest.Init();
    int test = 100;
    tbeTest.Run(&test, &test, 4096 * 4096, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_PROD, &test, &test);
    GlobalMockObject::verify();
}

TEST_F(TbeVectorReduceTest, EletwistV3_test)
{
    TbeReduce::TbeVectorReduce tbeTest;
    tbeTest.deviceType_ = LegacyDevType::DEV_TYPE_910;
    MOCKER_CPP(&TbeReduce::TbeVectorReduce::LoadOpBinary).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&TbeReduce::TbeVectorReduce::GetTilingDataDevMem)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(LaunchKernelWithConfig).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(BinaryLoadFromData).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(GetDeviceType).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    tbeTest.Init();
    int test = 100;
    tbeTest.Run(&test, &test, 128, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_PROD, &test, &test);
    GlobalMockObject::verify();
}
#endif

#if 1
s32 fake_rtGetSocVersionV81TbeVectorReduceSTest(char* chipVer, const u32 maxLen)
{
    memcpy_s(chipVer, sizeof("Ascend910B1"), "Ascend910B1", sizeof("Ascend910B1"));
    return DRV_ERROR_NONE;
}

TEST_F(TbeVectorReduceTest, EletwistV3_test_need_multi_core1_v81)
{
    TbeReduce::TbeVectorReduce tbeTest;
    tbeTest.deviceType_ = LegacyDevType::DEV_TYPE_910B;
    MOCKER_CPP(&TbeReduce::TbeVectorReduce::LoadOpBinary).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&TbeReduce::TbeVectorReduce::GetTilingDataDevMem)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(LaunchKernelWithConfig).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(BinaryLoadFromData).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(GetDeviceType).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(rtGetSocVersion).stubs().will(invoke(fake_rtGetSocVersionV81TbeVectorReduceSTest));
    tbeTest.Init();
    int test = 100;
    tbeTest.Run(&test, &test, 4096, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_PROD, &test, &test);
    GlobalMockObject::verify();
}

TEST_F(TbeVectorReduceTest, EletwistV3_test_need_multi_core2_v81)
{
    TbeReduce::TbeVectorReduce tbeTest;
    tbeTest.deviceType_ = LegacyDevType::DEV_TYPE_910B;
    MOCKER_CPP(&TbeReduce::TbeVectorReduce::LoadOpBinary).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&TbeReduce::TbeVectorReduce::GetTilingDataDevMem)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(LaunchKernelWithConfig).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(BinaryLoadFromData).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(GetDeviceType).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(rtGetSocVersion).stubs().will(invoke(fake_rtGetSocVersionV81TbeVectorReduceSTest));
    tbeTest.Init();
    int test = 100;
    tbeTest.Run(&test, &test, 4096 * 4096, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_PROD, &test, &test);
    GlobalMockObject::verify();
}

TEST_F(TbeVectorReduceTest, EletwistV3_test_v81)
{
    TbeReduce::TbeVectorReduce tbeTest;
    tbeTest.deviceType_ = LegacyDevType::DEV_TYPE_910B;
    MOCKER_CPP(&TbeReduce::TbeVectorReduce::LoadOpBinary).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&TbeReduce::TbeVectorReduce::GetTilingDataDevMem)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(LaunchKernelWithConfig).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(BinaryLoadFromData).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(GetDeviceType).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(rtGetSocVersion).stubs().will(invoke(fake_rtGetSocVersionV81TbeVectorReduceSTest));
    tbeTest.Init();
    nlohmann::json opDescInfo;
    HcclResult ret = tbeTest.GetOpInfo(HCCL_DATA_TYPE_INT16, HCCL_REDUCE_SUM, opDescInfo, opDescInfo);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    int test = 100;
    tbeTest.Run(&test, &test, 128, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_PROD, &test, &test);
    GlobalMockObject::verify();
}

TEST_F(TbeVectorReduceTest, dataType_error_test)
{
    HcclResult ret;
    nlohmann::json opDescInfo;
    nlohmann::json opTilingInfo;
    TbeReduce::TbeVectorReduce tbeTest;
    tbeTest.deviceType_ = LegacyDevType::DEV_TYPE_910B;
    ret = tbeTest.GetOpInfo(HCCL_DATA_TYPE_BFP16, HCCL_REDUCE_SUM, opDescInfo, opTilingInfo);
    EXPECT_EQ(ret, HCCL_E_PARA);
    GlobalMockObject::verify();
}

TEST_F(TbeVectorReduceTest, crack_crack_get_op_info)
{
    HcclResult ret;
    nlohmann::json opDescInfo;
    nlohmann::json opTilingInfo;
    TbeReduce::TbeCrackCleared tbeTest;
    tbeTest.deviceType_ = LegacyDevType::DEV_TYPE_910B;
    ret = tbeTest.GetOpInfo(opDescInfo, opTilingInfo);
    EXPECT_EQ(ret, HCCL_E_UNAVAIL);
    GlobalMockObject::verify();
}
#endif

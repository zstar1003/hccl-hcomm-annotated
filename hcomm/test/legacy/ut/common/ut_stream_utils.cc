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
#include "log.h"
#include "communicator_impl.h"
#include "internal_exception.h"
#include "invalid_params_exception.h"
#include "stream_utils.h"

using namespace Hccl;

class StreamUtilsTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "StreamUtilsTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "StreamUtilsTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in StreamUtilsTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in StreamUtilsTest TearDown" << std::endl;
    }
};

TEST_F(StreamUtilsTest, Ut_AddStreamToModel_When_rtStreamAddToModel_fail_Expect_HCCL_E_RUNTIME)
{
    // 前置条件
    MOCKER(&rtStreamAddToModel).stubs().will(returnValue(1));
    rtStream_t stream;
    rtModel_t rtModel;

    // 后置验证
    EXPECT_EQ(AddStreamToModel(stream, rtModel), HCCL_E_RUNTIME);
}

TEST_F(StreamUtilsTest, Ut_GetModelId_When_rtModelGetId_fail_Expect_HCCL_E_RUNTIME)
{
    // 前置条件
    MOCKER(&rtModelGetId).stubs().will(returnValue(1));
    rtModel_t rtModel;
    u32 model = 0;

    // 后置验证
    EXPECT_EQ(GetModelId(rtModel, model), HCCL_E_RUNTIME);
}

TEST_F(StreamUtilsTest, Ut_GetStreamCaptureInfo_When_aclmdlRICaptureGetInfo_success_Expect_HCCL_SUCCESS)
{
    // 前置条件
    MOCKER(&aclmdlRICaptureGetInfo).stubs().will(returnValue(207000));
    rtStream_t stream;
    rtModel_t rtModel;
    bool isCapture;

    // 后置验证
    EXPECT_EQ(GetStreamCaptureInfo(stream, rtModel, isCapture), HCCL_SUCCESS);
}

TEST_F(StreamUtilsTest, Ut_GetStreamCaptureInfo_When_aclmdlRICaptureGetInfo_fail_Expect_HCCL_E_RUNTIME)
{
    // 前置条件
    MOCKER(&aclmdlRICaptureGetInfo).stubs().will(returnValue(1));
    rtStream_t stream;
    rtModel_t rtModel;
    bool isCapture;

    // 后置验证
    EXPECT_EQ(GetStreamCaptureInfo(stream, rtModel, isCapture), HCCL_E_RUNTIME);
}

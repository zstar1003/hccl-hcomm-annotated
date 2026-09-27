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
#include "invalid_params_exception.h"
#include "drv_api_exception.h"
#include "network_api_exception.h"
#include "runtime_api_exception.h"
#include "resources_not_exist_exception.h"
#include "not_support_exception.h"
#include "null_ptr_exception.h"
#include "timeout_exception.h"

using namespace Hccl;

// InvalidParamsException
TEST(InvalidParamsExceptionTest, should_return_expected_error_msg_when_calling_what_fucntion)
{
    InvalidParamsException paraException("para1", "less than 0");
    EXPECT_STREQ("Parameter [para1] was invalid: [less than 0].", paraException.what());
}

TEST(InvalidParamsExceptionTest, should_return_expected_result_when_calling_get_error_code_function)
{
    InvalidParamsException paraException("para1", "less than 0");
    EXPECT_EQ(HcclResult::HCCL_E_PARA, paraException.GetErrorCode());
}

TEST(InvalidParamsExceptionTest, should_return_backtrace_when_calling_get_backtrace_function_and_in_debug_mode)
{
    InvalidParamsException paraException("para1", "less than 0");
    EXPECT_NE(0, paraException.GetBacktraceStr().size());
}

// TimeoutException
TEST(TimeoutExceptionTest, should_return_expected_error_msg_when_calling_what_fucntion)
{
    TimeoutException timeoutException("eventName", 1000, "us");
    EXPECT_STREQ("The event [eventName] was running exceeded the time limit [1000][us]", timeoutException.what());
}

TEST(TimeoutExceptionTest, should_return_expected_result_when_calling_get_error_code_function)
{
    TimeoutException timeoutException("eventName", 1000, "us");
    EXPECT_EQ(HcclResult::HCCL_E_TIMEOUT, timeoutException.GetErrorCode());
}

// NullPtrException
TEST(NullPtrExceptionTest, should_return_expected_error_msg_when_calling_what_fucntion)
{
    NullPtrException npeException("fieldName");
    EXPECT_STREQ("The field [fieldName] was nullptr.", npeException.what());
}

TEST(NullPtrExceptionTest, should_return_expected_result_when_calling_get_error_code_function)
{
    NullPtrException npeException("fieldName");
    EXPECT_EQ(HcclResult::HCCL_E_PTR, npeException.GetErrorCode());
}

// ResourcesNotExistException
TEST(ResourcesNotExistExceptionTest, should_return_expected_error_msg_when_calling_what_fucntion)
{
    ResourcesNotExistException notFoundException("resourceName", "filter sentence");
    EXPECT_STREQ("The resource [resourceName] was not found by [filter sentence].", notFoundException.what());
}

TEST(ResourcesNotExistExceptionTest, should_return_expected_result_when_calling_get_error_code_function)
{
    ResourcesNotExistException notFoundException("resourceName", "filter sentence");
    EXPECT_EQ(HcclResult::HCCL_E_NOT_FOUND, notFoundException.GetErrorCode());
}

// NotSupportException
TEST(NotSupportExceptionTest, should_return_expected_error_msg_when_calling_what_fucntion)
{
    NotSupportException notSupportException("featureName");
    EXPECT_STREQ("The [featureName] was not supported now.", notSupportException.what());
}

TEST(NotSupportExceptionTest, should_return_expected_result_when_calling_get_error_code_function)
{
    NotSupportException notSupportException("featureName");
    EXPECT_EQ(HcclResult::HCCL_E_NOT_SUPPORT, notSupportException.GetErrorCode());
}

// RuntimeApiException
TEST(RuntimeApiExceptionTest, should_return_expected_error_msg_when_calling_what_fucntion)
{
    RuntimeApiException callRuntimeException("apiName", "apiPara1=1");
    EXPECT_STREQ("Calling runtime api [apiName] failed, paras [apiPara1=1].", callRuntimeException.what());
}

TEST(RuntimeApiExceptionTest, should_return_expected_result_when_calling_get_error_code_function)
{
    RuntimeApiException callRuntimeException("apiName", "apiPara1=1");
    EXPECT_EQ(HcclResult::HCCL_E_RUNTIME, callRuntimeException.GetErrorCode());
}

// DrvApiException
TEST(DrvApiExceptionTest, should_return_expected_error_msg_when_calling_what_fucntion)
{
    DrvApiException callDrvException("apiName", "apiPara1=1");
    EXPECT_STREQ("Calling Drv api [apiName] failed, paras [apiPara1=1].", callDrvException.what());
}

TEST(DrvApiExceptionTest, should_return_expected_result_when_calling_get_error_code_function)
{
    DrvApiException callDrvException("apiName", "apiPara1=1");
    EXPECT_EQ(HcclResult::HCCL_E_DRV, callDrvException.GetErrorCode());
}

// NetworkApiException
TEST(NetworkApiExceptionTest, should_return_expected_error_msg_when_calling_what_fucntion)
{
    NetworkApiException callNetworkException("apiName", "apiPara1=1");
    EXPECT_STREQ("Calling network api [apiName] failed, paras [apiPara1=1].", callNetworkException.what());
}

TEST(NetworkApiExceptionTest, should_return_expected_result_when_calling_get_error_code_function)
{
    NetworkApiException callNetworkException("apiName", "apiPara1=1");
    EXPECT_EQ(HcclResult::HCCL_E_NETWORK, callNetworkException.GetErrorCode());
}

TEST(TraceApiExceptionTest, should_return_expected_result_when_calling_get_error_code_function)
{
    TraceApiException callTraceException("apiName", "apiPara1=1");
    EXPECT_EQ(HcclResult::HCCL_E_INTERNAL, callNetworkException.GetErrorCode());
}

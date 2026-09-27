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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include "orion_adapter_trace.h"
#include "atrace_api.h"

using namespace Hccl;

class AdapterTraceTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AdapterTraceTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "AdapterTraceTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in AdapterTraceTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in AdapterTraceTest TearDown" << std::endl;
    }
};

TEST(AdapterTraceTest, tracecreate_test)
{
    MOCKER(AtraceCreateWithAttr).stubs().will(returnValue(0));
    TraHandle handle;
    TraceCreate("HCCL");
}

TEST(AdapterTraceTest, St_TraceCreate_Fail_When_AtraceCreateWithAttr_Return_TRACE_INVALID_HANDLE)
{
    GlobalMockObject::verify();
    MOCKER(AtraceCreateWithAttr).stubs().will(returnValue(TRACE_INVALID_HANDLE));
    TraHandle handle;
    EXPECT_EQ(TraceCreate("HCCL"), -1);
}

TEST(AdapterTraceTest, traceSubmit)
{
    MOCKER(AtraceSubmit).stubs().will(returnValue(0));
    TraHandle handle = 0;
    std::string buffer = "HCCL_TEST";
    u32 size = sizeof(buffer);
    const unsigned char* startPos = (const unsigned char*)(buffer.c_str());
    TraceSubmit(handle, startPos, size);
}

TEST(AdapterTraceTest, traceDestroy_test) { MOCKER(AtraceDestroy).stubs(); }

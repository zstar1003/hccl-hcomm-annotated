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

#define __HCCL_SAL_GLOBAL_RES_INCLUDE__

#include <sal.h>
#include <hccl/base.h>
#include <hccl/hccl_types.h>

#include "sal.h"
#include "log.h"
#include "dltrace_function.h"
#include "hccl_trace_info.h"

using namespace std;
using namespace hccl;
class SalTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SalTest SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "SalTest TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }
};

TEST_F(SalTest, ut_ip_addr_check)
{
    MOCKER(inet_ntop).stubs().will(returnValue((char const*)NULL));

    HcclInAddr ipv4;
    ipv4.addr.s_addr = 666;

    HcclInAddr ipv6;
    ipv6.addr6.s6_addr32[0] = 6;
    ipv6.addr6.s6_addr32[1] = 6;
    ipv6.addr6.s6_addr32[2] = 6;
    ipv6.addr6.s6_addr32[3] = 6;

    HcclIpAddress ip_addr(AF_INET, ipv4);
    EXPECT_EQ(ip_addr.IsInvalid(), false);
    HcclIpAddress ipv6_addr(AF_INET6, ipv6);
    EXPECT_EQ(ipv6_addr.IsInvalid(), false);

    string ipStr = "::::";
    HcclIpAddress ip;
    HcclResult ret = ip.SetReadableAddress(ipStr);
    EXPECT_EQ(ret, HCCL_E_PARA);
    GlobalMockObject::verify();
}

TEST_F(SalTest, ut_SalStrToLonglong_error)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string str = "";
    s64 val = 0;
    ret = SalStrToLonglong(str, 10, val);
    EXPECT_EQ(ret, HCCL_E_PARA);
    str = "023";
    ret = SalStrToLonglong(str, 10, val);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    str = "9223372036854775808";
    ret = SalStrToLonglong(str, 10, val);
    EXPECT_EQ(ret, HCCL_E_PARA);
    str = "Something went wrong.";
    ret = SalStrToLonglong(str.c_str(), 10, val);
    EXPECT_EQ(ret, HCCL_E_PARA);
    str = "123";
    ret = SalStrToLonglong(str, 10, val);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(SalTest, ut_atrace_error_test)
{
    DlTraceFunction::GetInstance().DlTraceFunctionInit();
    MOCKER(AtraceCreateWithAttr).stubs().will(returnValue(TRACE_INVALID_HANDLE));

    HcclTraceInfo atrace;
    std::string hccl = "HCCL";
    HcclResult ret = atrace.Init(hccl);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::string log = "TEST LOG";
    ret = atrace.SaveTraceInfo(log, AtraceOption::Opbasekey);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(SalTest, ut_DlTraceFunctionInit_ATrace_Failed_Expect_ReturnError)
{
    MOCKER_CPP(&hccl::DlTraceFunction::DlATraceFunctionInterInit).stubs().will(returnValue(HCCL_E_INTERNAL));

    DlTraceFunction& dlTrace = DlTraceFunction::GetInstance();
    dlTrace.handle_ = nullptr;
    HcclResult ret = dlTrace.DlTraceFunctionInit();
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    GlobalMockObject::verify();
}

TEST_F(SalTest, ut_SalGetDataTypeSize_ValidType_Expect_Success)
{
    u32 dataTypeSize = 0;
    HcclResult ret = SalGetDataTypeSize(HCCL_DATA_TYPE_INT8, dataTypeSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(dataTypeSize, 1U);

    ret = SalGetDataTypeSize(HCCL_DATA_TYPE_FP32, dataTypeSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(dataTypeSize, 4U);
}

TEST_F(SalTest, ut_SalGetDataTypeSize_InvalidType_Expect_ReturnParaError)
{
    u32 dataTypeSize = 0;
    HcclResult ret = SalGetDataTypeSize(HCCL_DATA_TYPE_RESERVED, dataTypeSize);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

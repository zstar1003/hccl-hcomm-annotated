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
#include <stdio.h>
#include <fstream>

#include "sal.h"
#include "adapter_rts.h"

#include "hvd_adapter.h"
#include "llt_hccl_stub_sal_pub.h"
using namespace std;
using namespace hccl;
int g_data = 10;
int* g_dataPtr = &g_data;
void* g_voidPtr = (void*)g_dataPtr;

class HvdAdapterTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--HvdAdapterTest SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--HvdAdapterTest TearDown--\033[0m" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }
};

HcclResult fun1(void* data)
{
    unsigned long long* val = (unsigned long long*)data;
    HCCL_INFO("In function fun1, val is %lld", *val);
    return HCCL_SUCCESS;
}

HcclResult fun2(void* data)
{
    unsigned long long* val = (unsigned long long*)data;
    HCCL_INFO("In function fun2, val is %lld", *val * 2);
    return HCCL_SUCCESS;
}

TEST_F(HvdAdapterTest, ut_hvd_adapter_init_release_success)
{
    MOCKER(aclrtProcessReport).stubs().with(mockcpp::any()).will(returnValue(ACL_SUCCESS));
    void* stream1;
    void* stream2;
    g_hvdAdapterGlobal.HvdAdapterInit(stream1, 0);
    g_hvdAdapterGlobal.HvdAdapterInit(stream2, 1);
    HcomRegHvdCallback(fun1);
    SaluSleep(500);
    HcomRegHvdCallback(fun2);
    SaluSleep(500);
    g_hvdAdapterGlobal.HvdAdapterDestroy();
}

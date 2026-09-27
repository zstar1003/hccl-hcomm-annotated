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
#include "aicpu/launch_device_v2.h"
#define private public
#define protected public
#include "hccl_aiv_utils.h"
#undef private
#undef protected

using namespace Hccl;

using namespace std;

class AivUtilsTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AivUtilsTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "AivUtilsTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in AivUtilsTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in AivUtilsTest TearDown" << std::endl;
    }
};

TEST_F(AivUtilsTest, aiv_utils_test)
{
    char resolvePath[] = "./CMakeList.txt";
    char* path = resolvePath;
    char returnValueVec[] = "0";
    char* returnValueChar = returnValueVec;
    MOCKER(realpath).stubs().with(mockcpp::any(), outBound(path)).will(returnValue(returnValueChar));
    RegisterKernel();

    char libPath[] = "./hcomm/test/llt/stub/workspace/fwkacllib/lib64";
    char* libPathPtr = libPath;

    MOCKER(getenv).stubs().with(mockcpp::any()).will(returnValue(libPathPtr));

    MOCKER(LoadBinaryFromFile).stubs().with(mockcpp::any());

    MOCKER(realpath).stubs().with(mockcpp::any(), outBound(path)).will(returnValue(nullptr));
    RegisterKernel();

    char resolvePath2[] = "./libhccl_v2.so";
    path = resolvePath2;
    MOCKER(realpath).stubs().with(mockcpp::any(), outBound(path)).will(returnValue(returnValueChar));
    RegisterKernel();

    aclrtBinHandle g_binHandle;
    Hccl::LoadBinaryFromFile(
        "./hcomm/test/llt/stub/workspace/fwkacllib/lib64/hccl_aiv_op_91095.o", ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD, 1,
        g_binHandle);
}

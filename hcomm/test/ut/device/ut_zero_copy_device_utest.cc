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

#ifndef private
#define private public
#define protected public
#endif
#include "zero_copy_address_mgr.h"
#undef private
#undef protected

using namespace std;
using namespace hccl;

class Zero_Copy_Device_UT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Zero_Copy_Device_UT SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "Zero_Copy_Device_UT TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = -1;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};

TEST_F(Zero_Copy_Device_UT, ZeroCopyDeviceTest)
{
    ZeroCopyAddressMgr addressMrg;
    addressMrg.InitRingBuffer();
    ZeroCopyRingBufferItem item;
    addressMrg.PushOne(item);
}

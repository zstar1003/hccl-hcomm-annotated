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
#include "mem_host_pub.h"
#include <string>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include "llt_hccl_stub_sal_pub.h"
#include "adapter_rts.h"

#define private public
#define protected public
#include "stream_active_manager.h"
#undef protected
#undef private

using namespace std;
using namespace hccl;

class StreamActiveManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--StreamActiveManager SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--StreamActiveManager TearDown--\033[0m" << std::endl; }
    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }
};

#if 1
TEST_F(StreamActiveManagerTest, ut_StreamActive)
{
    MOCKER(hrtStreamActive).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetStreamId).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = StreamActiveManager::GetInstance(0).StreamActive(nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}
#endif

#if 1
TEST_F(StreamActiveManagerTest, ut_StreamsUnactive)
{
    MOCKER(hrtGetStreamId).stubs().will(returnValue(HCCL_SUCCESS));
    HcclResult ret = StreamActiveManager::GetInstance(0).StreamActive(nullptr, nullptr);
    std::vector<Stream> streams;
    Stream tmpStream;
    streams.push_back(tmpStream);
    ret = StreamActiveManager::GetInstance(0).StreamsUnactive(streams);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}
#endif

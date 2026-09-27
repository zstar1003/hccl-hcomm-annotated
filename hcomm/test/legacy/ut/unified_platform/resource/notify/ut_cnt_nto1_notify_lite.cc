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
#include "cnt_nto1_notify_lite.h"
#include "binary_stream.h"
#include "string_util.h"
#include "log.h"

using namespace Hccl;

class CntNto1NotifyLiteTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CntNto1NotifyLiteTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "CntNto1NotifyLiteTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in CntNto1NotifyLiteTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CntNto1NotifyLiteTest TearDown" << std::endl;
    }
    u32 fakeNotifyId = 1;
    u32 fakeDevPhyId = 2;
};

TEST_F(CntNto1NotifyLiteTest, cnt_nto1_notify_lite_give_uniqueId)
{
    BinaryStream binaryStream;
    binaryStream << fakeNotifyId;
    binaryStream << fakeDevPhyId;
    std::vector<char> result;
    binaryStream.Dump(result);

    CntNto1NotifyLite lite(result);
    EXPECT_EQ(fakeNotifyId, lite.GetId());
    EXPECT_EQ(fakeDevPhyId, lite.GetDevPhyId());
}

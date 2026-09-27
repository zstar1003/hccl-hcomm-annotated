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

#include "hccl/base.h"
#include <hccl/hccl_types.h>

#include "reducer_pub.h"
#include "profiler_manager.h"
#include "transport_base_pub.h"
#include "externalinput.h"

using namespace std;
using namespace hccl;

class ReducerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--ReducerTest SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--ReducerTest TearDown--\033[0m" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};

#if 1
TEST_F(ReducerTest, destructor_D0)
{
    s32 ret = HCCL_SUCCESS;

    Reducer* reducer_int8 = new Reducer(HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, 0);
    Reducer* reducer_int = new Reducer(HCCL_DATA_TYPE_INT32, HCCL_REDUCE_SUM, 0);
    Reducer* reducer_half = new Reducer(HCCL_DATA_TYPE_FP16, HCCL_REDUCE_SUM, 0);
    Reducer* reducer_float = new Reducer(HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, 0);
    Reducer* reducer_res = new Reducer(HCCL_DATA_TYPE_RESERVED, HCCL_REDUCE_SUM, 0);
    Reducer* reducer_int16 = new Reducer(HCCL_DATA_TYPE_INT16, HCCL_REDUCE_SUM, 0);
    Reducer* reducer_bfp16 = new Reducer(HCCL_DATA_TYPE_BFP16, HCCL_REDUCE_SUM, 0);

    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete reducer_int8;
    delete reducer_int;
    delete reducer_half;
    delete reducer_float;
    delete reducer_res;
    delete reducer_int16;
    delete reducer_bfp16;
}
#endif

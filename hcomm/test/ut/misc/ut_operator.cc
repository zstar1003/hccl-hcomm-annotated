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

#include "llt_hccl_stub_pub.h"
#include "hccl_operator.h"
#include "stream_pub.h"

using namespace std;
using namespace hccl;
using namespace hcclCustomOp;

class OperatorTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--OperatorTest SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--OperatorTest TearDown--\033[0m" << std::endl; }
    virtual void SetUp()
    {
        s32 portNum = -1;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }
};

// HcclCounterAdd接口st用例
TEST_F(OperatorTest, utOperatorHcclCounterAdd)
{
    u64 counter = 0;
    Stream stream(StreamType::STREAM_TYPE_OFFLINE);
    HcclOpAddCounter(NULL, 1, stream.ptr());
    HcclOpAddCounter(&counter, 1, NULL);
    HcclOpAddCounter(&counter, 0, stream.ptr());
    HcclOpAddCounter(&counter, 100, stream.ptr());
    EXPECT_EQ(counter, 100);
}

// HcclCounterClear接口st用例
TEST_F(OperatorTest, utOperatorHcclCounterClear)
{
    u64 counter = 0;
    Stream stream(StreamType::STREAM_TYPE_OFFLINE);
    HcclOpClearCounter(NULL, stream.ptr());
    HcclOpClearCounter(&counter, NULL);

    HcclOpAddCounter(&counter, 100, stream.ptr());
    EXPECT_EQ(counter, 100);

    HcclOpClearCounter(&counter, stream.ptr());
    EXPECT_EQ(counter, 0);
}

// HcclCmp接口st用例
TEST_F(OperatorTest, utOperatorHcclCmp)
{
    s32 result = 0xff;
    Stream stream(StreamType::STREAM_TYPE_OFFLINE);

    s32 srcInt8 = 0x5a;
    s32 dstRight8 = 0x5a;
    s32 dstError8 = 0xa5;
    double deviation8 = 0.0;

    s32 srcInt32 = 0x5a;
    s32 dstRightInt32 = 0x5a;
    s32 dstErrorInt32 = 0xa5;
    double deviationInt32 = 0.0;

    float srcFp32 = 0.0000001;
    float dstRightFp32 = 0.0000001;
    float dstErrorFp32 = 0.0000021;
    double deviationFp32 = 0.000001;

    float srcFp16 = 0.0000001;
    float dstRightFp16 = 0.0000001;
    float dstErrorFp16 = 0.0000021;
    double deviationFp16 = 0.000001;

    HcclOpCmp(NULL, &dstRight8, HCCL_OPERATOR_INT8, 1, deviation8, &result, stream.ptr());
    HcclOpCmp(&srcInt8, NULL, HCCL_OPERATOR_INT8, 1, deviation8, &result, stream.ptr());
    HcclOpCmp(&srcInt8, &dstRight8, HCCL_OPERATOR_INT8, 1, deviation8, NULL, stream.ptr());
    HcclOpCmp(&srcInt8, &dstRight8, HCCL_OPERATOR_INT8, 1, deviation8, &result, NULL);
    HcclOpCmp(&srcInt8, &dstRight8, HCCL_OPERATOR_INT8, 0, deviation8, &result, stream.ptr());
    HcclOpCmp(&srcInt8, &dstRight8, HCCL_OPERATOR_RESERVED, 1, deviation8, &result, stream.ptr());

    result = 0xff;
    HcclOpCmp(&srcInt8, &dstRight8, HCCL_OPERATOR_INT8, 1, deviation8, &result, stream.ptr());
    EXPECT_EQ(result, 0);
    result = 0;
    HcclOpCmp(&srcInt8, &dstError8, HCCL_OPERATOR_INT8, 1, deviation8, &result, stream.ptr());
    EXPECT_EQ(result, -1);

    result = 0xff;
    HcclOpCmp(&srcInt32, &dstRightInt32, HCCL_OPERATOR_INT32, 1, deviationInt32, &result, stream.ptr());
    EXPECT_EQ(result, 0);
    result = 0;
    HcclOpCmp(&srcInt32, &dstErrorInt32, HCCL_OPERATOR_INT32, 1, deviationInt32, &result, stream.ptr());
    EXPECT_EQ(result, -1);

    result = 0xff;
    HcclOpCmp(&srcFp32, &dstRightFp32, HCCL_OPERATOR_FP32, 1, deviationFp32, &result, stream.ptr());
    EXPECT_EQ(result, 0);
    result = 0;
    HcclOpCmp(&srcFp32, &dstErrorFp32, HCCL_OPERATOR_FP32, 1, deviationFp32, &result, stream.ptr());
    EXPECT_EQ(result, -1);

    result = 0xff;
    HcclOpCmp(&srcFp16, &dstRightFp16, HCCL_OPERATOR_FP16, 1, deviationFp16, &result, stream.ptr());
    EXPECT_EQ(result, 0);
    result = 0;
    HcclOpCmp(&srcFp16, &dstErrorFp16, HCCL_OPERATOR_FP16, 1, deviationFp16, &result, stream.ptr());
    EXPECT_EQ(result, -1);
}

// HcclLogStr接口st用例
TEST_F(OperatorTest, utOperatorHcclLogStr)
{
    char str[] = "This is a Test";
    Stream stream(StreamType::STREAM_TYPE_OFFLINE);
    HcclOpLogStr(DLOG_INFO, NULL, stream.ptr());
    HcclOpLogStr(DLOG_INFO, str, NULL);
    HcclOpLogStr(7, str, stream.ptr());
    HcclOpLogStr(17, str, stream.ptr());
    HcclOpLogStr(DLOG_INFO, str, stream.ptr());
}

// HcclLogMem接口st用例
TEST_F(OperatorTest, stOperatorHcclLogMem)
{
    s8 tmpInt8[6] = {0, 1, 2, 3, 4, 5};
    float tmpfp16[6] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    s32 tmpInt32[6] = {0, 1, 2, 3, 4, 5};
    float tmpfp32[6] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    Stream stream(StreamType::STREAM_TYPE_OFFLINE);

    HcclOpLogMem(DLOG_INFO, NULL, HCCL_OPERATOR_INT8, 6, stream.ptr());
    HcclOpLogMem(DLOG_INFO, tmpInt8, HCCL_OPERATOR_INT8, 6, NULL);
    HcclOpLogMem(DLOG_INFO, tmpInt8, HCCL_OPERATOR_INT8, 0, stream.ptr());
    HcclOpLogMem(7, tmpInt8, HCCL_OPERATOR_INT8, 6, stream.ptr());
    HcclOpLogMem(17, tmpInt8, HCCL_OPERATOR_INT8, 6, stream.ptr());
    HcclOpLogMem(DLOG_INFO, tmpInt8, HCCL_OPERATOR_INT8, 6, stream.ptr());
    HcclOpLogMem(DLOG_INFO, tmpfp16, HCCL_OPERATOR_FP16, 6, stream.ptr());
    HcclOpLogMem(DLOG_INFO, tmpInt32, HCCL_OPERATOR_INT32, 6, stream.ptr());
    HcclOpLogMem(DLOG_INFO, tmpfp32, HCCL_OPERATOR_FP32, 6, stream.ptr());
    HcclOpLogMem(DLOG_INFO, tmpInt8, HCCL_OPERATOR_RESERVED, 6, stream.ptr());
}

// HcclLogVariable接口st用例
TEST_F(OperatorTest, utOperatorHcclLogVariable)
{
    char str[] = "This is a test! ";
    u64 data1 = 1;
    u64 data2 = 2;
    u64 data3 = 3;
    u64 data4 = 4;
    u64 data5 = 5;
    u64 data6 = 6;
    Stream stream(StreamType::STREAM_TYPE_OFFLINE);

    HcclOpLogVariable(DLOG_INFO, NULL, 6, &data1, &data2, &data3, &data4, &data5, &data6, stream.ptr());
    HcclOpLogVariable(DLOG_INFO, str, 6, &data1, &data2, &data3, &data4, &data5, &data6, NULL);
    HcclOpLogVariable(DLOG_INFO, str, 1, NULL, &data2, &data3, &data4, &data5, &data6, stream.ptr());
    HcclOpLogVariable(DLOG_INFO, str, 2, &data1, NULL, &data3, &data4, &data5, &data6, stream.ptr());
    HcclOpLogVariable(DLOG_INFO, str, 3, &data1, &data2, NULL, &data4, &data5, &data6, stream.ptr());
    HcclOpLogVariable(DLOG_INFO, str, 4, &data1, &data2, &data3, NULL, &data5, &data6, stream.ptr());
    HcclOpLogVariable(DLOG_INFO, str, 5, &data1, &data2, &data3, &data4, NULL, &data6, stream.ptr());
    HcclOpLogVariable(DLOG_INFO, str, 6, &data1, &data2, &data3, &data4, &data5, NULL, stream.ptr());
    HcclOpLogVariable(DLOG_INFO, str, 0, &data1, &data2, &data3, &data4, &data5, &data6, stream.ptr());
    HcclOpLogVariable(7, str, 6, &data1, &data2, &data3, &data4, &data5, &data6, stream.ptr());
    HcclOpLogVariable(17, str, 6, &data1, &data2, &data3, &data4, &data5, &data6, stream.ptr());
    HcclOpLogVariable(DLOG_INFO, str, 1, &data1, &data2, &data3, &data4, &data5, &data6, stream.ptr());
    HcclOpLogVariable(DLOG_INFO, str, 2, &data1, &data2, &data3, &data4, &data5, &data6, stream.ptr());
    HcclOpLogVariable(DLOG_INFO, str, 3, &data1, &data2, &data3, &data4, &data5, &data6, stream.ptr());
    HcclOpLogVariable(DLOG_INFO, str, 4, &data1, &data2, &data3, &data4, &data5, &data6, stream.ptr());
    HcclOpLogVariable(DLOG_INFO, str, 5, &data1, &data2, &data3, &data4, &data5, &data6, stream.ptr());
    HcclOpLogVariable(DLOG_INFO, str, 6, &data1, &data2, &data3, &data4, &data5, &data6, stream.ptr());
    HcclOpLogVariable(DLOG_INFO, str, 0, &data1, &data2, &data3, &data4, &data5, &data6, stream.ptr());
}

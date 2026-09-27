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
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>

#include "../../../hccl_api_base_test.h"
#include "aicpu_ts_thread_interface.h"
#include "stream_lite.h"
#include "rtsq_base.h"

using namespace Hccl;

class TestIAicpuTsThread : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();

        streamLitePtr = aicpuThread.GetStreamLitePtr();
        EXPECT_NE(nullptr, streamLitePtr);

        rtsqPtr = static_cast<StreamLite*>(streamLitePtr)->GetRtsq();
        EXPECT_NE(nullptr, rtsqPtr);

        MOCKER_CPP_VIRTUAL(*rtsqPtr, &RtsqBase::LaunchTask).stubs().will(ignoreReturnValue());
        MOCKER_CPP_VIRTUAL(*rtsqPtr, &RtsqBase::TryLaunchTask).stubs().will(ignoreReturnValue());
        MOCKER_CPP_VIRTUAL(*rtsqPtr, &RtsqBase::NotifyWait, void(RtsqBase::*)(uint32_t, uint32_t))
            .stubs()
            .will(ignoreReturnValue());
        MOCKER_CPP_VIRTUAL(*rtsqPtr, &RtsqBase::NotifyRecordLoc).stubs().will(ignoreReturnValue());
        MOCKER_CPP_VIRTUAL(*rtsqPtr, &RtsqBase::SdmaCopy).stubs().will(ignoreReturnValue());
        MOCKER_CPP_VIRTUAL(*rtsqPtr, &RtsqBase::SdmaReduce).stubs().will(ignoreReturnValue());
    }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
    IAicpuTsThread aicpuThread{1, 2, 3, 4};
    void* streamLitePtr{nullptr};
    RtsqBase* rtsqPtr{nullptr};
};

/**
 * 测试 IAicpuTsThread LaunchTask 正常执行
 * 验证：初始化后调用 LaunchTask 不崩溃
 */
TEST_F(TestIAicpuTsThread, Ut_IAicpuTsThread_LaunchTask_When_Inited_Expect_Success)
{
    // 不应该崩溃
    EXPECT_NO_THROW(aicpuThread.LaunchTask());
}

/**
 * 测试 IAicpuTsThread TryLaunchTask 正常执行
 * 验证：初始化后调用 TryLaunchTask 不崩溃
 */
TEST_F(TestIAicpuTsThread, Ut_IAicpuTsThread_TryLaunchTask_When_Inited_Expect_Success)
{
    // 不应该崩溃
    EXPECT_NO_THROW(aicpuThread.TryLaunchTask());
}

/**
 * 测试 IAicpuTsThread NotifyWait 正常执行
 * 验证：初始化后调用 NotifyWait 返回成功
 */
TEST_F(TestIAicpuTsThread, Ut_IAicpuTsThread_NotifyWait_When_Inited_Expect_Success)
{
    uint32_t notifyId = 100;

    HcclResult ret = aicpuThread.NotifyWait(notifyId);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

/**
 * 测试 IAicpuTsThread NotifyRecordLoc 正常执行
 * 验证：初始化后调用 NotifyRecordLoc 返回成功
 */
TEST_F(TestIAicpuTsThread, Ut_IAicpuTsThread_NotifyRecordLoc_When_Inited_Expect_Success)
{
    uint32_t notifyId = 200;

    HcclResult ret = aicpuThread.NotifyRecordLoc(notifyId);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

/**
 * 测试 IAicpuTsThread SdmaCopy 正常执行
 * 验证：初始化后调用 SdmaCopy 返回成功
 */
TEST_F(TestIAicpuTsThread, Ut_IAicpuTsThread_SdmaCopy_When_Inited_Expect_Success)
{
    uint64_t dstAddr = 0x1000;
    uint64_t srcAddr = 0x2000;
    uint64_t sizeByte = 1024;

    HcclResult ret = aicpuThread.SdmaCopy(dstAddr, srcAddr, sizeByte);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

/**
 * 测试 IAicpuTsThread SdmaCopy sizeByte 超出4GB
 * 验证：sizeByte 超出4GB时返回参数错误
 */
TEST_F(TestIAicpuTsThread, Ut_IAicpuTsThread_SdmaCopy_When_SizeExceedMax_Expect_Return_Para_Error)
{
    uint64_t dstAddr = 0x1000;
    uint64_t srcAddr = 0x2000;
    uint64_t sizeByte = 0x100000001ULL;

    HcclResult ret = aicpuThread.SdmaCopy(dstAddr, srcAddr, sizeByte);
    EXPECT_EQ(HCCL_E_PARA, ret);
}

/**
 * 测试 IAicpuTsThread SdmaReduce 正常执行（DataType=INT8, ReduceOp=SUM）
 * 验证：初始化后调用 SdmaReduce 返回成功
 */
TEST_F(TestIAicpuTsThread, Ut_IAicpuTsThread_SdmaReduce_INT8_SUM_When_Inited_Expect_Success)
{
    IAicpuTsThread aicpuThread(1, 2, 3, 4);
    uint64_t dstAddr = 0x1000;
    uint64_t srcAddr = 0x2000;
    uint64_t sizeByte = 1024;
    uint32_t dataTypeRaw = 0; // INT8
    uint32_t reduceOpRaw = 0; // SUM

    HcclResult ret = aicpuThread.SdmaReduce(dstAddr, srcAddr, sizeByte, dataTypeRaw, reduceOpRaw);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

/**
 * 测试 IAicpuTsThread SdmaReduce 正常执行（DataType=FP16, ReduceOp=PROD）
 * 验证：DataType 和 ReduceOp 映射正确
 */
TEST_F(TestIAicpuTsThread, Ut_IAicpuTsThread_SdmaReduce_FP16_PROD_When_Inited_Expect_Success)
{
    IAicpuTsThread aicpuThread(1, 2, 3, 4);
    uint64_t dstAddr = 0x1000;
    uint64_t srcAddr = 0x2000;
    uint64_t sizeByte = 1024;
    uint32_t dataTypeRaw = 3; // FP16
    uint32_t reduceOpRaw = 1; // PROD

    HcclResult ret = aicpuThread.SdmaReduce(dstAddr, srcAddr, sizeByte, dataTypeRaw, reduceOpRaw);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

/**
 * 测试 IAicpuTsThread SdmaReduce 正常执行（DataType=FP32, ReduceOp=MAX）
 * 验证：DataType 和 ReduceOp 映射正确
 */
TEST_F(TestIAicpuTsThread, Ut_IAicpuTsThread_SdmaReduce_FP32_MAX_When_Inited_Expect_Success)
{
    IAicpuTsThread aicpuThread(1, 2, 3, 4);
    uint64_t dstAddr = 0x1000;
    uint64_t srcAddr = 0x2000;
    uint64_t sizeByte = 1024;
    uint32_t dataTypeRaw = 4; // FP32
    uint32_t reduceOpRaw = 2; // MAX

    HcclResult ret = aicpuThread.SdmaReduce(dstAddr, srcAddr, sizeByte, dataTypeRaw, reduceOpRaw);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

/**
 * 测试 IAicpuTsThread SdmaReduce 正常执行（DataType=INT32, ReduceOp=MIN）
 * 验证：DataType 和 ReduceOp 映射正确
 */
TEST_F(TestIAicpuTsThread, Ut_IAicpuTsThread_SdmaReduce_INT32_MIN_When_Inited_Expect_Success)
{
    IAicpuTsThread aicpuThread(1, 2, 3, 4);
    uint64_t dstAddr = 0x1000;
    uint64_t srcAddr = 0x2000;
    uint64_t sizeByte = 1024;
    uint32_t dataTypeRaw = 2; // INT32
    uint32_t reduceOpRaw = 3; // MIN

    HcclResult ret = aicpuThread.SdmaReduce(dstAddr, srcAddr, sizeByte, dataTypeRaw, reduceOpRaw);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

/**
 * 测试 IAicpuTsThread SdmaReduce sizeByte 超出4GB
 * 验证：sizeByte 超出4GB时返回参数错误
 */
TEST_F(TestIAicpuTsThread, Ut_IAicpuTsThread_SdmaReduce_When_SizeExceedMax_Expect_Return_Para_Error)
{
    IAicpuTsThread aicpuThread(1, 2, 3, 4);
    uint64_t dstAddr = 0x1000;
    uint64_t srcAddr = 0x2000;
    uint64_t sizeByte = 0x100000001ULL;
    uint32_t dataTypeRaw = 0; // INT8
    uint32_t reduceOpRaw = 0; // SUM

    HcclResult ret = aicpuThread.SdmaReduce(dstAddr, srcAddr, sizeByte, dataTypeRaw, reduceOpRaw);
    EXPECT_EQ(HCCL_E_PARA, ret);
}

/**
 * 测试 IAicpuTsThread SdmaReduce 不支持的数据类型
 * 验证：dataType 不在映射表中时返回参数错误
 */
TEST_F(TestIAicpuTsThread, Ut_IAicpuTsThread_SdmaReduce_When_InvalidDataType_Expect_Return_Para_Error)
{
    IAicpuTsThread aicpuThread(1, 2, 3, 4);
    uint64_t dstAddr = 0x1000;
    uint64_t srcAddr = 0x2000;
    uint64_t sizeByte = 1024;
    uint32_t dataTypeRaw = 100; // 不支持的数据类型
    uint32_t reduceOpRaw = 0;   // SUM

    HcclResult ret = aicpuThread.SdmaReduce(dstAddr, srcAddr, sizeByte, dataTypeRaw, reduceOpRaw);
    EXPECT_EQ(HCCL_E_PARA, ret);
}

/**
 * 测试 IAicpuTsThread SdmaReduce 不支持的操作类型
 * 验证：reduceOp 不在映射表中时返回参数错误
 */
TEST_F(TestIAicpuTsThread, Ut_IAicpuTsThread_SdmaReduce_When_InvalidReduceOp_Expect_Return_Para_Error)
{
    IAicpuTsThread aicpuThread(1, 2, 3, 4);
    uint64_t dstAddr = 0x1000;
    uint64_t srcAddr = 0x2000;
    uint64_t sizeByte = 1024;
    uint32_t dataTypeRaw = 0;   // INT8
    uint32_t reduceOpRaw = 100; // 不支持的操作类型

    HcclResult ret = aicpuThread.SdmaReduce(dstAddr, srcAddr, sizeByte, dataTypeRaw, reduceOpRaw);
    EXPECT_EQ(HCCL_E_PARA, ret);
}

/**
 * 测试 IAicpuTsThread GetStreamLitePtr 正常获取
 * 验证：初始化后可以正确获取 StreamLite 指针
 */
TEST_F(TestIAicpuTsThread, Ut_IAicpuTsThread_GetStreamLitePtr_When_Inited_Expect_Success)
{
    IAicpuTsThread aicpuThread(1, 2, 3, 4);
    void* streamLitePtr = aicpuThread.GetStreamLitePtr();

    EXPECT_NE(nullptr, streamLitePtr);
}

/**
 * 测试 IAicpuTsThread GetSqId 正常获取
 * 验证：初始化后可以正确获取 SqId
 */
TEST_F(TestIAicpuTsThread, Ut_IAicpuTsThread_GetSqId_When_Inited_Expect_Success)
{
    IAicpuTsThread aicpuThread(1, 2, 3, 4);
    uint32_t sqId = aicpuThread.GetSqId();
    // sqId 应该被正确设置
    EXPECT_EQ(2, sqId);
}

/**
 * 测试 IAicpuTsThread 析构函数在 StreamLite 已初始化时的行为
 * 验证：析构时正确释放 StreamLite 资源
 */
TEST_F(TestIAicpuTsThread, Ut_IAicpuTsThread_Destructor_When_Inited_Expect_NoMemoryLeak)
{
    IAicpuTsThread* aicpuThread = new IAicpuTsThread(1, 2, 3, 4);
    // 不应该发生内存泄漏
    EXPECT_NO_THROW(delete aicpuThread);
}

/**
 * 测试 IAicpuTsThread 支持 FP8E4M3 数据类型
 * 验证：FP8E4M3 (dataTypeRaw=15) 是支持的数据类型
 */
TEST_F(TestIAicpuTsThread, Ut_IAicpuTsThread_SdmaReduce_FP8E4M3_When_Inited_Expect_Success)
{
    IAicpuTsThread aicpuThread(1, 2, 3, 4);
    uint64_t dstAddr = 0x1000;
    uint64_t srcAddr = 0x2000;
    uint64_t sizeByte = 256;
    uint32_t dataTypeRaw = 15; // FP8E4M3
    uint32_t reduceOpRaw = 0;  // SUM

    HcclResult ret = aicpuThread.SdmaReduce(dstAddr, srcAddr, sizeByte, dataTypeRaw, reduceOpRaw);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

/**
 * 测试 IAicpuTsThread 支持 FP8E5M2 数据类型
 * 验证：FP8E5M2 (dataTypeRaw=16) 是支持的数据类型
 */
TEST_F(TestIAicpuTsThread, Ut_IAicpuTsThread_SdmaReduce_FP8E5M2_When_Inited_Expect_Success)
{
    IAicpuTsThread aicpuThread(1, 2, 3, 4);
    uint64_t dstAddr = 0x1000;
    uint64_t srcAddr = 0x2000;
    uint64_t sizeByte = 256;
    uint32_t dataTypeRaw = 16; // FP8E5M2
    uint32_t reduceOpRaw = 0;  // SUM

    HcclResult ret = aicpuThread.SdmaReduce(dstAddr, srcAddr, sizeByte, dataTypeRaw, reduceOpRaw);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

/**
 * 测试 IAicpuTsThread 支持 BFP16 数据类型
 * 验证：BFP16 (dataTypeRaw=11) 是支持的数据类型
 */
TEST_F(TestIAicpuTsThread, Ut_IAicpuTsThread_SdmaReduce_BFP16_When_Inited_Expect_Success)
{
    IAicpuTsThread aicpuThread(1, 2, 3, 4);
    uint64_t dstAddr = 0x1000;
    uint64_t srcAddr = 0x2000;
    uint64_t sizeByte = 512;
    uint32_t dataTypeRaw = 11; // BFP16
    uint32_t reduceOpRaw = 1;  // PROD

    HcclResult ret = aicpuThread.SdmaReduce(dstAddr, srcAddr, sizeByte, dataTypeRaw, reduceOpRaw);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

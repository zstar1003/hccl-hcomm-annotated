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

#ifndef private
#define private public
#define protected public
#endif

#include "stream_pub.h"
#include "mem_host_pub.h"
#include "mem_device_pub.h"
#include "sal.h"

#include "adapter_rts.h"

#undef private
#undef protected

using namespace std;
using namespace hccl;

class StreamTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--StreamTest SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--StreamTest TearDown--\033[0m" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        streamInfo.actualStreamId = 1;
        streamInfo.sqId = 1;
        streamInfo.sqDepth = 100;
        streamInfo.sqBaseAddr = &streamInfo;
        streamInfo.logicCqId = 1;
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        std::cout << "A Test TearDown" << std::endl;
        GlobalMockObject::verify();
    }

    HcclComStreamInfo streamInfo;
};

TEST_F(StreamTest, constructor_00)
{
    s32 ret = HCCL_SUCCESS;
    rtStream_t rtstream;
    aclrtCreateStream(&rtstream);
    Stream stream(rtstream);
    EXPECT_TRUE(stream.ptr() != nullptr);

    ret = aclrtDestroyStream(rtstream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(StreamTest, constructor_01)
{
    Stream stream(StreamType::STREAM_TYPE_OFFLINE);
    EXPECT_TRUE(stream.ptr() != nullptr);
}

TEST_F(StreamTest, constructor_02)
{
    Stream stream1(StreamType::STREAM_TYPE_OFFLINE);
    EXPECT_TRUE(stream1.ptr() != nullptr);

    Stream stream2(stream1);
    EXPECT_TRUE(stream2.ptr() != nullptr);
}

TEST_F(StreamTest, stream_construct_by_type_fail)
{
    MOCKER(aclrtCreateStream).expects(atMost(1)).will(returnValue(1));

    MOCKER(aclrtCreateStream).expects(atMost(1)).will(returnValue(1));

    Stream stream1(StreamType::STREAM_TYPE_OFFLINE);
    EXPECT_TRUE(stream1.ptr() == nullptr);

    GlobalMockObject::verify();

    MOCKER(hrtGetStreamId).stubs().will(returnValue(1));

    Stream stream2(StreamType::STREAM_TYPE_OFFLINE);
    EXPECT_TRUE(stream2.ptr() == nullptr);

    GlobalMockObject::verify();

    MOCKER(hrtStreamGetSqid).stubs().will(returnValue(1));

    Stream stream3(StreamType::STREAM_TYPE_OFFLINE);
    EXPECT_TRUE(stream3.ptr() == nullptr);

    GlobalMockObject::verify();
}

TEST_F(StreamTest, stream_construct_get_stream_id_fail)
{
    s32 ret = HCCL_SUCCESS;
    rtStream_t rtstream;

    MOCKER(hrtGetStreamId).stubs().will(returnValue(1));

    Stream stream(StreamType::STREAM_TYPE_OFFLINE);
    GlobalMockObject::verify();
    EXPECT_TRUE(stream.ptr() == nullptr);
}

TEST_F(StreamTest, stream_construct_get_sqid_and_cqid_fail)
{
    s32 ret = HCCL_SUCCESS;
    rtStream_t rtstream;

    MOCKER(hrtStreamGetSqid).stubs().will(returnValue(1));

    MOCKER(hrtStreamGetCqid).stubs().will(returnValue(1));

    Stream stream(StreamType::STREAM_TYPE_OFFLINE);
    GlobalMockObject::verify();
    EXPECT_TRUE(stream.ptr() == nullptr);
}

TEST_F(StreamTest, set_stream_mode_fail)
{
    s32 ret = HCCL_SUCCESS;

    MOCKER(hrtStreamSetMode).stubs().will(returnValue(1));

    Stream stream(StreamType::STREAM_TYPE_OFFLINE);
    ret = stream.SetMode(8);
    GlobalMockObject::verify();
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(StreamTest, aicpu_stream_streamInfo)
{
    const HcclComStreamInfo* streamInfoGotten;

    Stream stream(streamInfo, false);
    EXPECT_TRUE(stream.ptr() != nullptr);
    EXPECT_EQ(streamInfo.actualStreamId, stream.id());

    stream.GetStreamInfo(streamInfoGotten);
    EXPECT_EQ(streamInfoGotten->sqDepth, streamInfo.sqDepth);
}

TEST_F(StreamTest, aicpu_stream_sqe_context)
{
    uint32_t sqHead = 0;
    uint32_t sqTail = 100;

    Stream stream(streamInfo, false);
    SqCqeContext sqeCqeCtx;
    sqeCqeCtx.sqContext.inited = false;
    stream.InitSqAndCqeContext(sqHead, sqTail, &sqeCqeCtx);

    // 测试初始化是否成功
    HcclSqeContext* sqeContext = stream.GetSqeContextPtr();
    auto& buff = sqeContext->buffer;
    EXPECT_EQ(buff.sqHead, sqHead);
    EXPECT_EQ(buff.sqTail, sqTail);

    // 测试是否可以识别到sqeContext已经初始化
    sqHead = 10;
    sqTail = 110;
    SqCqeContext sqeCqeCtx1;
    sqeCqeCtx1.sqContext.inited = false;
    stream.InitSqAndCqeContext(sqHead, sqTail, &sqeCqeCtx1);

    // 测试GetNextSqeBufferAddr是否可以溢出
    uint8_t* sqeBufferAddr;
    uint8_t* sqeTypeAddr;
    uint8_t* sqeDfxInfoAddr = nullptr;
    uint16_t taskId;
    s32 ret = HCCL_SUCCESS;

    buff.tailSqeIdx = HCCL_SQE_MAX_CNT;
    ret = stream.GetNextSqeBufferAddr(sqeBufferAddr, sqeTypeAddr, sqeDfxInfoAddr, taskId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 测试GetNextSqeBufferAddr是否可以获取正确的addr
    buff.tailSqeIdx = HCCL_SQE_MAX_CNT - 1;
    ret = stream.GetNextSqeBufferAddr(sqeBufferAddr, sqeTypeAddr, sqeDfxInfoAddr, taskId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 测试是否可以正确清理buffer
    ret = stream.ClearLocalBuff();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(StreamTest, aicpu_stream_constructor)
{
    uint32_t sqHead = 0;
    uint32_t sqTail = 100;

    Stream stream(streamInfo, false);
    SqCqeContext sqeCqeCtx;
    sqeCqeCtx.sqContext.inited = false;
    stream.InitSqAndCqeContext(sqHead, sqTail, &sqeCqeCtx);
    HcclSqeContext* sqeContext = stream.GetSqeContextPtr();
    auto& buff = sqeContext->buffer;
    EXPECT_EQ(buff.sqHead, sqHead);
    EXPECT_EQ(buff.sqTail, sqTail);

    sqHead = 10;
    sqTail = 110;

    Stream streamCopy(stream);
    EXPECT_TRUE(streamCopy);
    SqCqeContext sqeCqeCtx1;
    sqeCqeCtx1.sqContext.inited = false;
    streamCopy.InitSqAndCqeContext(sqHead, sqTail, &sqeCqeCtx1);
    HcclSqeContext* sqeContext1 = streamCopy.GetSqeContextPtr();
    auto& buff1 = sqeContext1->buffer;
    EXPECT_EQ(buff1.sqHead, sqHead);
    EXPECT_EQ(buff1.sqTail, sqTail);

    Stream streamMove(std::move(stream));
    EXPECT_TRUE(streamMove);
    EXPECT_FALSE(stream);
    HcclSqeContext* sqeContext2 = streamMove.GetSqeContextPtr();
    auto& buff2 = sqeContext2->buffer;
    EXPECT_EQ(buff2.sqHead, 0);
    EXPECT_EQ(buff2.sqTail, 100);
}

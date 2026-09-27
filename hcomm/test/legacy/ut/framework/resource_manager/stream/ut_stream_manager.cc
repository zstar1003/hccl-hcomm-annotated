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
#include "log.h"
#define private public
#define protected public
#include "communicator_impl.h"
#include "internal_exception.h"
#include "invalid_params_exception.h"
#include "stream_utils.h"
#include "stream_manager.h"
#undef protected
#undef private

using namespace Hccl;

class StreamManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "StreamManager tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "StreamManager tests tear down." << std::endl; }

    virtual void SetUp()
    {
        impl.opExecuteConfig.accState = AcceleratorState::CCU_MS;
        streamManager = new StreamManager(&impl);

        MOCKER(HrtGetStreamId).stubs().will(returnValue(static_cast<s32>(fakeId)));
        MOCKER(HrtStreamGetSqId).stubs().will(returnValue(fakeSqId));
        MOCKER(HrtStreamDestroy).stubs();
        MOCKER(HrtGetDevice).stubs().will(returnValue(fakeDevLogId));
        MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().will(returnValue(static_cast<DevId>(fakeDevPhyId)));

        std::cout << "A Test case in StreamManager SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        delete streamManager;
        std::cout << "A Test case in StreamManager TearDown" << std::endl;
    }

    u32 fakeId = 1;
    s32 fakeDevLogId = 1;
    u32 fakeDevPhyId = 1;
    u32 fakeSqId = 2;
    u32 fakeCqId = 2;
    u32 num = 1;
    u32 sizePerDto = 12;
    CommunicatorImpl impl;
    StreamManager* streamManager;
};

TEST_F(StreamManagerTest, opbase_not_register_and_get)
{
    // Given

    // when
    auto res = streamManager->opbase->GetMaster();

    // then
    EXPECT_EQ(nullptr, res);
}

TEST_F(StreamManagerTest, opbase_register_master_and_get)
{
    // Given
    auto stream = std::make_unique<Stream>((void*)1);
    streamManager->opbase->RegisterMaster(std::move(stream));

    // when
    auto res = streamManager->opbase->GetMaster();

    // then
    EXPECT_NE((void*)1, res);
}

TEST_F(StreamManagerTest, opbase_register_master_two_same_stream_and_get)
{
    // Given
    auto stream = std::make_unique<Stream>((void*)1);
    auto stream1 = std::make_unique<Stream>((void*)1);

    // when
    streamManager->opbase->RegisterMaster(std::move(stream));
    streamManager->opbase->RegisterMaster(std::move(stream1));

    auto res = streamManager->opbase->GetMaster();

    // then
    EXPECT_NE((void*)1, res);
}

TEST_F(StreamManagerTest, opbase_register_master_two_diff_stream_and_get)
{
    // Given
    auto stream = std::make_unique<Stream>((void*)1);
    auto stream1 = std::make_unique<Stream>((void*)1234);
    // when
    streamManager->opbase->RegisterMaster(std::move(stream));
    streamManager->opbase->RegisterMaster(std::move(stream1));
    auto res = streamManager->opbase->GetMaster();

    // then
    EXPECT_NE((void*)1, res);
}

TEST_F(StreamManagerTest, clear_slaves)
{
    streamManager->opbase->GetOrCreateSlave();
    ASSERT_NO_THROW(streamManager->opbase->Clear());
}

TEST_F(StreamManagerTest, offload_register_master_and_get)
{
    // Given
    auto stream = std::make_unique<Stream>((void*)1);
    std::string opTag = "test";
    auto streamData = std::make_unique<Stream>((void*)1);
    streamManager->offload->slaves[opTag] = std::vector<std::unique_ptr<Stream>>();
    streamManager->offload->slaves[opTag].push_back(std::move(streamData));

    streamManager->offload->RegisterMaster(opTag, std::move(stream));

    // when
    auto res = streamManager->offload->GetMaster(opTag);

    // then
    EXPECT_NE((void*)1, res);
}

TEST_F(StreamManagerTest, offload_register_master_two_diff_stream_and_get)
{
    // Given
    auto stream = std::make_unique<Stream>((void*)1);
    auto stream1 = std::make_unique<Stream>((void*)1234);
    std::string opTag = "test";
    // when
    streamManager->offload->RegisterMaster(opTag, std::move(stream));
    auto res = streamManager->offload->GetMaster(opTag);
    // then
    EXPECT_NE((void*)1, res);

    // then
    EXPECT_THROW(streamManager->offload->RegisterMaster(opTag, std::move(stream1)), InvalidParamsException);
}

TEST_F(StreamManagerTest, offload_register_two_diff_slave_stream_and_get)
{
    // Given
    std::string opTag = "test";
    std::vector<void*> slaveStreams = {(void*)1234, (void*)5678};
    // when
    streamManager->offload->RegisterSlaves(opTag, slaveStreams);
    streamManager->offload->currOpTag = opTag;

    auto res1 = streamManager->offload->GetSlave(opTag);
    auto res2 = streamManager->offload->GetSlave(opTag);

    // then
    EXPECT_NE(nullptr, res1);
    EXPECT_NE(nullptr, res2);

    // then
    EXPECT_THROW(streamManager->offload->RegisterSlaves(opTag, slaveStreams), InvalidParamsException);
}

TEST_F(StreamManagerTest, Ut_CaptureSlaveStream_When_GetStreamCaptureInfo_ERROR_Expect_InternalException)
{
    // 前置条件
    MOCKER(&GetStreamCaptureInfo).stubs().will(returnValue(HCCL_E_RUNTIME));

    auto stream = std::make_unique<Stream>((void*)1);
    streamManager->opbase->RegisterMaster(std::move(stream));
    streamManager->comm->currentCollOperator = std::make_unique<CollOperator>();
    streamManager->comm->currentCollOperator->opMode = OpMode::OPBASE;

    auto stream1 = std::make_unique<Stream>();
    auto stream2 = std::make_unique<Stream>();

    // 执行测试步骤
    EXPECT_THROW(streamManager->CaptureSlaveStream(stream1.get(), stream2.get()), InternalException);
}

TEST_F(StreamManagerTest, Ut_CaptureSlaveStream_When_rtModel_Null_Expect_NullPtrException)
{
    // 前置条件
    bool isCapture = true;
    rtModel_t rtModel = nullptr;
    MOCKER(&GetStreamCaptureInfo)
        .stubs()
        .with(mockcpp::any(), outBound(rtModel), outBound(isCapture))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(&GetModelId).stubs().will(returnValue(HCCL_E_RUNTIME));

    auto stream = std::make_unique<Stream>((void*)1);
    streamManager->opbase->RegisterMaster(std::move(stream));
    streamManager->comm->currentCollOperator = std::make_unique<CollOperator>();
    streamManager->comm->currentCollOperator->opMode = OpMode::OPBASE;

    auto stream1 = std::make_unique<Stream>();
    auto stream2 = std::make_unique<Stream>();

    // 执行测试步骤
    EXPECT_THROW(streamManager->CaptureSlaveStream(stream1.get(), stream2.get()), NullPtrException);
}

TEST_F(StreamManagerTest, Ut_CaptureSlaveStream_When_GetModelId_ERROR_Expect_InternalException)
{
    // 前置条件
    bool isCapture = true;
    rtModel_t rtModel = (void*)0x100;
    MOCKER(&GetStreamCaptureInfo)
        .stubs()
        .with(mockcpp::any(), outBound(rtModel), outBound(isCapture))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(&GetModelId).stubs().will(returnValue(HCCL_E_RUNTIME));

    auto stream = std::make_unique<Stream>((void*)1);
    streamManager->opbase->RegisterMaster(std::move(stream));
    streamManager->comm->currentCollOperator = std::make_unique<CollOperator>();
    streamManager->comm->currentCollOperator->opMode = OpMode::OPBASE;

    auto stream1 = std::make_unique<Stream>();
    auto stream2 = std::make_unique<Stream>();

    // 执行测试步骤
    EXPECT_THROW(streamManager->CaptureSlaveStream(stream1.get(), stream2.get()), InternalException);
}

TEST_F(StreamManagerTest, Ut_CaptureSlaveStream_When_AddStreamToModel_ERROR_InternalException)
{
    // 前置条件
    bool isCapture = true;
    rtModel_t rtModel = (void*)0x100;
    MOCKER(&GetStreamCaptureInfo)
        .stubs()
        .with(mockcpp::any(), outBound(rtModel), outBound(isCapture))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(&GetModelId).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(&AddStreamToModel).stubs().will(returnValue(HCCL_E_RUNTIME));
    auto stream = std::make_unique<Stream>((void*)1);
    streamManager->opbase->RegisterMaster(std::move(stream));
    streamManager->comm->currentCollOperator = std::make_unique<CollOperator>();
    streamManager->comm->currentCollOperator->opMode = OpMode::OPBASE;

    auto stream1 = std::make_unique<Stream>();
    auto stream2 = std::make_unique<Stream>();

    // 执行测试步骤
    EXPECT_THROW(streamManager->CaptureSlaveStream(stream1.get(), stream2.get()), InternalException);
}

TEST_F(StreamManagerTest, Ut_CaptureSlaveStream_Expect_no_throw)
{
    // 前置条件
    bool isCapture = true;
    rtModel_t rtModel = (void*)0x100;
    MOCKER(&GetStreamCaptureInfo)
        .stubs()
        .with(mockcpp::any(), outBound(rtModel), outBound(isCapture))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(&GetModelId).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(&AddStreamToModel).stubs().will(returnValue(HCCL_SUCCESS));
    auto stream = std::make_unique<Stream>((void*)1);
    streamManager->opbase->RegisterMaster(std::move(stream));
    streamManager->comm->currentCollOperator = std::make_unique<CollOperator>();
    streamManager->comm->currentCollOperator->opMode = OpMode::OPBASE;

    auto stream1 = std::make_unique<Stream>();
    auto stream2 = std::make_unique<Stream>();

    // 执行测试步骤
    EXPECT_NO_THROW(streamManager->CaptureSlaveStream(stream1.get(), stream2.get()));
}

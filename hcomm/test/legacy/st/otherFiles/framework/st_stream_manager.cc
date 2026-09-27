/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define private public
#define protected public
#include "gtest/gtest.h"
#include "stream_manager.h"
#include <mockcpp/mockcpp.hpp>
#include "log.h"
#include "communicator_impl.h"
#include "invalid_params_exception.h"
#undef private
#undef protected

using namespace Hccl;

class StreamManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "StreamManager tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "StreamManager tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in StreamManager SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in StreamManager TearDown" << std::endl;
    }
};

#define SETUP_STREAM_MANAGER_MOCKS(streamPtr)                                                    \
    \ do                                                                                         \
    {                                                                                            \
        \ MOCKER(HrtStreamCreateWithFlags).stubs().will(returnValue(streamPtr));                 \
        \ MOCKER(HrtGetStreamId).stubs().will(returnValue(0));                                   \
        \ MOCKER(HrtGetDevice).stubs().will(returnValue(0));                                     \
        \ MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().will(returnValue(static_cast<DevId>(1))); \
        \ MOCKER(HrtStreamDestroy).stubs();                                                      \
        \                                                                                        \
    }                                                                                            \
    while (0)

#define SETUP_STREAM_MGR_TEST()         \
    CommunicatorImpl impl;              \
    StreamManager streamManager(&impl); \
    void* temp = (void*)0x1;            \
    SETUP_STREAM_MANAGER_MOCKS(temp)

TEST(StreamManagerTest, opbase_not_register_and_get)
{
    // Given
    CommunicatorImpl impl;
    StreamManager streamManager(&impl);

    // when
    auto res = streamManager.opbase->GetMaster();

    // then
    EXPECT_EQ(nullptr, res);
}

TEST(StreamManagerTest, opbase_register_master_and_get)
{
    // Given
    SETUP_STREAM_MGR_TEST();

    auto stream = std::make_unique<Stream>(temp);
    streamManager.opbase->RegisterMaster(std::move(stream));

    // when
    auto res = streamManager.opbase->GetMaster();

    // then
    EXPECT_NE(nullptr, res);
}

TEST(StreamManagerTest, opbase_register_master_two_same_stream_and_get)
{
    // Given
    SETUP_STREAM_MGR_TEST();

    auto stream = std::make_unique<Stream>(temp);
    auto stream1 = std::make_unique<Stream>(temp);

    // when
    streamManager.opbase->RegisterMaster(std::move(stream));
    streamManager.opbase->RegisterMaster(std::move(stream1));

    auto res = streamManager.opbase->GetMaster();

    // then
    EXPECT_NE(nullptr, res);
}

TEST(StreamManagerTest, opbase_register_master_two_diff_stream_and_get)
{
    // Given
    SETUP_STREAM_MGR_TEST();

    auto stream = std::make_unique<Stream>(temp);
    auto stream1 = std::make_unique<Stream>((void*)1234);
    // when
    streamManager.opbase->RegisterMaster(std::move(stream));
    streamManager.opbase->RegisterMaster(std::move(stream1));
    auto res = streamManager.opbase->GetMaster();

    // then
    EXPECT_NE(nullptr, res);
}

TEST(StreamManagerTest, clear_slaves)
{
    SETUP_STREAM_MGR_TEST();

    MOCKER(HrtStreamDestroy).stubs();

    streamManager.opbase->GetOrCreateSlave();
    ASSERT_NO_THROW(streamManager.opbase->Clear());
}

TEST(StreamManagerTest, offload_register_master_and_get)
{
    // Given
    SETUP_STREAM_MGR_TEST();

    auto stream = std::make_unique<Stream>(nullptr);
    std::string opTag = "test";
    auto streamData = std::make_unique<Stream>(nullptr);
    streamManager.offload->slaves[opTag] = std::vector<std::unique_ptr<Stream>>();
    streamManager.offload->slaves[opTag].push_back(std::move(streamData));
    streamManager.offload->RegisterMaster(opTag, std::move(stream));

    // when
    auto res = streamManager.offload->GetMaster(opTag);

    // then
    EXPECT_NE(nullptr, res);
}

TEST(StreamManagerTest, offload_register_master_two_diff_stream_and_get)
{
    // Given
    SETUP_STREAM_MGR_TEST();

    auto stream = std::make_unique<Stream>(nullptr);
    auto stream1 = std::make_unique<Stream>((void*)1234);
    std::string opTag = "test";
    // when
    streamManager.offload->RegisterMaster(opTag, std::move(stream));
    auto res = streamManager.offload->GetMaster(opTag);
    // then
    EXPECT_NE(nullptr, res);

    // then
    EXPECT_THROW(streamManager.offload->RegisterMaster(opTag, std::move(stream1)), InvalidParamsException);
}

TEST(StreamManagerTest, offload_register_two_diff_slave_stream_and_get)
{
    // Given
    SETUP_STREAM_MGR_TEST();

    std::string opTag = "test";
    std::vector<void*> slaveStreams = {(void*)1234, (void*)5678};
    // when
    streamManager.offload->RegisterSlaves(opTag, slaveStreams);
    streamManager->offload->currOpTag = opTag;

    auto res1 = streamManager.offload->GetSlave(opTag);
    auto res2 = streamManager.offload->GetSlave(opTag);

    // then
    EXPECT_NE(nullptr, res1);
    EXPECT_NE(nullptr, res2);

    // then
    EXPECT_THROW(streamManager.offload->RegisterSlaves(opTag, slaveStreams), InvalidParamsException);
}

TEST(StreamManagerTest, St_ClearOpStream_When_Normal_Expect_Success)
{
    // Given
    SETUP_STREAM_MGR_TEST();
    MOCKER(HrtStreamActive).stubs();

    std::string opTag = "test";
    std::vector<void*> slaveStreams = {(void*)1111, (void*)2222};
    // when
    auto master = std::make_unique<Stream>((void*)3333);
    streamManager.offload->RegisterSlaves(opTag, slaveStreams);
    streamManager.offload->RegisterMaster(opTag, std::move(master));
    auto res = streamManager.offload->ClearOpStream(opTag);
    EXPECT_EQ(res, HCCL_SUCCESS);
}

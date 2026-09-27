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
#include "stream_manager.h"
#include <mockcpp/mockcpp.hpp>
#include "log.h"
#define private public
#define protected public
#include "communicator_impl.h"
#include "internal_exception.h"
#include "invalid_params_exception.h"
#include "stream_utils.h"
#undef protected
#undef private

using namespace Hccl;

class AicpuStreamManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AicpuStreamManager tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "AicpuStreamManager tests tear down." << std::endl; }

    virtual void SetUp()
    {
        streamManager = new AicpuStreamManager();

        MOCKER(HrtGetStreamId).stubs().will(returnValue(fakeId));
        MOCKER(HrtStreamGetSqId).stubs().will(returnValue(fakeSqId));
        MOCKER(HrtStreamDestroy).stubs();
        MOCKER(HrtGetDevice).stubs().will(returnValue(fakeDevLogId));
        MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().will(returnValue(static_cast<DevId>(fakeDevPhyId)));

        streamManager->AllocStreams(1);
        streamManager->AllocFreeStream();
        std::cout << "A Test case in AicpuStreamManager SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        streamManager->Clear();
        delete streamManager;
        std::cout << "A Test case in AicpuStreamManager TearDown" << std::endl;
    }

    s32 fakeId = 1;
    s32 fakeDevLogId = 1;
    u32 fakeDevPhyId = 1;
    u32 fakeSqId = 2;
    u32 fakeCqId = 2;
    u32 num = 1;
    u32 sizePerDto = 12;
    AicpuStreamManager* streamManager;
};

HcclResult GetStreamCaptureInfoStub(rtStream_t stream, rtModel_t& rtModel, bool& isCapture)
{
    isCapture = true;
    rtModel = (void*)0x100;
    return HCCL_SUCCESS;
}

TEST_F(AicpuStreamManagerTest, Ut_AclGraphCaptureFreeStream_When_GetStreamCaptureInfo_ERROR_Expect_InternalException)
{
    // 前置条件
    MOCKER(&GetStreamCaptureInfo).stubs().will(returnValue(HCCL_E_RUNTIME));

    auto stream = std::make_unique<Stream>();
    // 执行测试步骤
    EXPECT_THROW(streamManager->AclGraphCaptureFreeStream(stream.get()), InternalException);
}

TEST_F(AicpuStreamManagerTest, Ut_AclGraphCaptureFreeStream_When_GetStreamCaptureInfo_is_no_capture_Expect_Success)
{
    // 前置条件
    MOCKER(&GetStreamCaptureInfo).stubs().will(returnValue(HCCL_SUCCESS));

    auto stream = std::make_unique<Stream>(nullptr);
    // 执行测试步骤
    streamManager->AclGraphCaptureFreeStream(stream.get());
}

TEST_F(AicpuStreamManagerTest, Ut_AclGraphCaptureFreeStream_When_GetModelId_ERROR_Expect_InternalException)
{
    // 前置条件
    bool isCapture = true;
    MOCKER(&GetStreamCaptureInfo)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(isCapture))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(&GetModelId).stubs().will(returnValue(HCCL_E_RUNTIME));

    auto stream = std::make_unique<Stream>();
    // 执行测试步骤
    EXPECT_THROW(streamManager->AclGraphCaptureFreeStream(stream.get()), InternalException);
}

TEST_F(AicpuStreamManagerTest, Ut_AclGraphCaptureFreeStream_When_AddStreamToModel_ERROR_InternalException)
{
    // 前置条件
    bool isCapture = true;
    MOCKER(&GetStreamCaptureInfo)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(isCapture))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(&GetModelId).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(&AddStreamToModel).stubs().will(returnValue(HCCL_E_RUNTIME));

    auto stream = std::make_unique<Stream>();
    // 执行测试步骤
    EXPECT_THROW(streamManager->AclGraphCaptureFreeStream(stream.get()), InternalException);
}

TEST_F(AicpuStreamManagerTest, Ut_AclGraphCaptureFreeStream_When_AddStreamToModel_Success)
{
    // 前置条件
    bool isCapture = true;
    MOCKER(&GetStreamCaptureInfo)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(isCapture))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(&GetModelId).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(&AddStreamToModel).stubs().will(returnValue(HCCL_SUCCESS));
    auto stream = std::make_unique<Stream>();

    // 执行测试步骤
    EXPECT_THROW(streamManager->AclGraphCaptureFreeStream(stream.get()), InternalException);
}

TEST_F(AicpuStreamManagerTest, Ut_CaptureFreeStream_When_isCapture_true_Expect_SUCCESS)
{
    // 前置条件
    MOCKER(&GetStreamCaptureInfo).stubs().will(invoke(GetStreamCaptureInfoStub));
    MOCKER(&GetModelId).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(&AddStreamToModel).stubs().will(returnValue(HCCL_SUCCESS));
    auto mainStream = std::make_unique<Stream>();
    auto freeStream = std::make_unique<Stream>();

    // 执行步骤
    auto ret = streamManager->CaptureFreeStream(mainStream.get(), freeStream.get());

    // 后置验证
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

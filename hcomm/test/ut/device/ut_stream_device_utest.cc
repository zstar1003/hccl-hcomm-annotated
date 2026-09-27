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
#include "stream_utils.h"
#include "env_config.h"
#include "op_base.h"
#undef private
#undef protected

using namespace std;

class Stream_Device_UT : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Stream_Device_UT SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "Stream_Device_UT TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};

TEST_F(Stream_Device_UT, StreamDeviceTest)
{
    rtStream_t aicpuStream;
    aclmdlRI rtModel = nullptr;
    bool isCapture = false;
    u64 modelId = 0;
    GetStreamCaptureInfo(aicpuStream, rtModel, isCapture);
    AddStreamToModel(aicpuStream, rtModel);
    GetModelId(rtModel, modelId);
}

TEST_F(Stream_Device_UT, EnvDeviceTest)
{
    std::string opName = "llt";
    std::vector<HcclAlgoType> algType;
    ParseAlgoString(opName, opName, algType);
    SetHcclAlgoConfig(opName);
}

TEST_F(Stream_Device_UT, OpBaseDeviceTest)
{
    aclrtStream stream;
    aclmdlRICaptureStatus captureStatus = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    uint64_t modelId = 0;
    bool isCapture = true;
    GetCaptureInfo(stream, captureStatus, modelId, isCapture);
}

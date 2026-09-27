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
#include "hccp.h"
#include "dev_buffer.h"
#include "local_rdma_rma_buffer_v2.h"
#include "remote_rma_buffer.h"

using namespace Hccl;

class LocalRdmaRmaBufferTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "LocalRdmaRmaBuffer tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "LocalRdmaRmaBuffer tests tear down." << std::endl; }

    virtual void SetUp()
    {
        MOCKER(HrtIpcSetMemoryName).stubs().with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any());
        MOCKER(HrtDevMemAlignWithPage)
            .stubs()
            .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any());
        MOCKER(HrtIpcDestroyMemoryName).stubs().with(mockcpp::any());

        std::cout << "A Test case in LocalRdmaRmaBuffer SetUP." << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in LocalRdmaRmaBuffer TearDown." << std::endl;
    }

    std::shared_ptr<DevBuffer> devBuf = DevBuffer::Create(0x100, 0x100);
};

TEST_F(LocalRdmaRmaBufferTest, Ut_When_LocalRdmaRmaBuffer_Construct_Error_Expect_Exception)
{
    EXPECT_THROW(LocalRdmaRmaBuffer localRdmaRmaBuffer(devBuf, nullptr), NullPtrException);
};

TEST_F(LocalRdmaRmaBufferTest, Ut_When_LocalRdmaRmaBuffer_Construct_With_Invalid_Param_Expect_Exception)
{
    RdmaHandle rdmaHandle = (void*)0x10000000;
    std::shared_ptr<DevBuffer> fakeBuf = std::make_shared<DevBuffer>(0x100, 0);
    EXPECT_THROW(LocalRdmaRmaBuffer localRdmaRmaBuffer(fakeBuf, rdmaHandle), InvalidParamsException);
};

TEST_F(LocalRdmaRmaBufferTest, Ut_When_Serialize_Expect_Success)
{
    RdmaHandle rdmaHandle = (RdmaHandle)0x1000000;
    LocalRdmaRmaBuffer localRdmaRmaBuffer(devBuf, rdmaHandle);
    std::string msg = localRdmaRmaBuffer.Describe();
    EXPECT_NE(0, msg.length());
};

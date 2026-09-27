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
#include "communicator_impl.h"
#include "data_buf_manager.h"
#include "internal_exception.h"
#include "dev_buffer.h"
#include "rma_buffer.h"
using namespace Hccl;
class DataBufManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "DataBufManager tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "DataBufManager tests tear down." << std::endl; }

    virtual void SetUp()
    {
        dataBufManager = new DataBufManager();
        devBuffer = DevBuffer::Create(0x100, 0x100);
        std::cout << "A Test case in DataBufManager SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        delete dataBufManager;
        GlobalMockObject::verify();
        std::cout << "A Test case in DataBufManager TearDown" << std::endl;
    }

    DataBufManager* dataBufManager;
    std::string opTag = "opTag";
    BufferType bufferType = BufferType::SCRATCH;

    shared_ptr<DevBuffer> devBuffer;
};

TEST_F(DataBufManagerTest, get_null)
{
    auto nullRes = dataBufManager->Get(opTag, bufferType);
    EXPECT_EQ(nullptr, nullRes);
}

TEST_F(DataBufManagerTest, register_and_get_then_register_err)
{
    auto expect = devBuffer.get();
    auto res = dataBufManager->Register(opTag, bufferType, devBuffer);
    EXPECT_EQ(expect, res);

    auto res2 = dataBufManager->Register(opTag, bufferType, devBuffer);
    EXPECT_EQ(res2, res);

    auto res3 = dataBufManager->Register(opTag, bufferType, nullptr);
    EXPECT_EQ(res3, nullptr);
}

TEST_F(DataBufManagerTest, deregister_data_buffer)
{
    EXPECT_NO_THROW(dataBufManager->Deregister(opTag, BufferType::SCRATCH));
    EXPECT_NO_THROW(dataBufManager->Register(opTag, bufferType, devBuffer));
    EXPECT_NO_THROW(dataBufManager->Deregister(opTag, bufferType));
}

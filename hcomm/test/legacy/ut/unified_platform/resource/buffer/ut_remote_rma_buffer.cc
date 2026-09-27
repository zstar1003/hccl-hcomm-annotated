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

#include "local_ub_rma_buffer.h"
#include "exchange_ub_buffer_dto.h"
#include "exchange_ipc_buffer_dto.h"
#include "exchange_rdma_buffer_dto.h"
#define private public
#define protected public
#include "dev_buffer.h"
#include "rma_buffer.h"
#include "remote_rma_buffer.h"
#include "buffer_type.h"
#undef protected
#undef private

using namespace Hccl;

class RemoteRmaBufferTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RemoteRmaBuffer tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "RemoteRmaBuffer tests tear down." << std::endl; }

    virtual void SetUp()
    {
        devBuf = DevBuffer::Create(0x100, 0x100);
        std::cout << "A Test case in RemoteRmaBuffer SetUP." << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in RemoteRmaBuffer TearDown." << std::endl;
    }
    std::shared_ptr<DevBuffer> devBuf;
};

TEST_F(RemoteRmaBufferTest, remoterdmarmabuffer_construct_with_dto)
{
    ExchangeRdmaBufferDto dto;
    dto.addr = 0x114;
    dto.size = 0x514;
    dto.rkey = 1919;
    dto.memInfo = "testTag";
    RdmaHandle rdmaHandle = (RdmaHandle)0x1000000;
    RemoteRdmaRmaBuffer buffer(rdmaHandle, dto);
    EXPECT_EQ(dto.addr, buffer.GetAddr());
    EXPECT_EQ(dto.size, buffer.GetSize());
    EXPECT_EQ(dto.rkey, buffer.GetRkey());
    EXPECT_STREQ(dto.memInfo.c_str(), buffer.GetMemInfo().c_str());
};

TEST_F(RemoteRmaBufferTest, remoteubrmabuffer_construct_error)
{
    EXPECT_THROW(RemoteUbRmaBuffer remoteUbRmaBuffer(nullptr), NullPtrException);
};

TEST_F(RemoteRmaBufferTest, remoteubrmabuffer_deserialize_success)
{
    // construct buffer
    BufferType type = BufferType::INPUT;
    void* ptr = nullptr;
    u64 size = 0;
    bool remoteAccess = true;

    std::shared_ptr<DevBuffer> devBuf = DevBuffer::Create(0x100, 0x100);
    Buffer* buf = devBuf.get();

    HcclMemType memType = HcclMemType::HCCL_MEM_TYPE_HOST;
    u32 tokenValue = 1;
    u32 tokenId = 0;
    u64 keySize = 4;
    u8 key[HRT_UB_MEM_KEY_MAX_LEN]{};

    BinaryStream binaryStream;
    binaryStream << buf->GetAddr() << buf->GetSize() << memType << tokenValue << tokenId << keySize << key;
    ExchangeUbBufferDto dto;
    dto.Deserialize(binaryStream);

    RdmaHandle rdmaHandle = (void*)0x1000000;
    RemoteUbRmaBuffer remoteUbRmaBuffer(rdmaHandle, dto);

    UbRmaBufferExchangeData exchangeData;
    exchangeData.addr = buf->GetAddr();
    exchangeData.size = buf->GetSize();
    exchangeData.tokenValue = 1;
    exchangeData.tokenId = 0;
    memcpy_s(exchangeData.key, HRT_UB_MEM_KEY_MAX_LEN, key, HRT_UB_MEM_KEY_MAX_LEN);

    EXPECT_EQ(exchangeData.tokenId, remoteUbRmaBuffer.tokenId);
    EXPECT_EQ(memcmp(remoteUbRmaBuffer.key, exchangeData.key, HRT_UB_MEM_KEY_MAX_LEN), 0);
    EXPECT_EQ(buf->GetAddr(), remoteUbRmaBuffer.addr);
    EXPECT_EQ(buf->GetSize(), remoteUbRmaBuffer.size);
};

TEST_F(RemoteRmaBufferTest, remoterdmarmabuffer_describe_size)
{
    RdmaHandle rdmaHandle = (void*)0x1000000;
    RemoteRdmaRmaBuffer remoteRdmaRmaBuffer(rdmaHandle);

    std::string fakeKeyDesc = "fakeKeyDesc";
    MOCKER(HrtRaGetKeyDescribe).stubs().will(returnValue(fakeKeyDesc));
    remoteRdmaRmaBuffer.Describe();
}

TEST_F(RemoteRmaBufferTest, remoteubrmabuffer_construct_with_addr_device)
{
    uintptr_t addr = 0x12345678;
    u64 size = 0x1000;
    u32 tokenId = 42;
    u32 tokenValue = 7;
    HcclMemType memType = HCCL_MEM_TYPE_DEVICE;
    std::string memInfo = "test_device_mem";

    RemoteUbRmaBuffer buffer(addr, size, tokenId, tokenValue, memType, memInfo);

    EXPECT_EQ(buffer.GetAddr(), addr);
    EXPECT_EQ(buffer.GetSize(), size);
    EXPECT_EQ(buffer.tokenId, tokenId);
    EXPECT_EQ(buffer.tokenValue, tokenValue);
    EXPECT_EQ(buffer.GetMemType(), HCCL_MEM_TYPE_DEVICE);
    EXPECT_STREQ(buffer.GetMemInfo().c_str(), memInfo.c_str());
    EXPECT_EQ(buffer.GetRmaType(), RmaType::UB);
};

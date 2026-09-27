/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#define private public
#include "comm_mems/comm_mems.h"
#undef private

using namespace hccl;

class CommRegMemTest : public testing::Test {
protected:
    void SetUp() override { commMems_ = std::make_unique<CommMems>(4096); }
    void TearDown() override { commMems_.reset(); }

    CommMem MakeMem(void* addr, uint64_t size, CommMemType type = COMM_MEM_TYPE_DEVICE)
    {
        CommMem mem;
        mem.addr = addr;
        mem.size = size;
        mem.type = type;
        return mem;
    }

    std::unique_ptr<CommMems> commMems_;
};

// 自验证: same tag + diff addr → 返回 HCCL_E_PARA
TEST_F(CommRegMemTest, SameTagDiffAddrReturnsError)
{
    CommMem mem1 = MakeMem((void*)0x1000, 1024);
    void* handle = nullptr;
    EXPECT_EQ(commMems_->CommRegMem("mytag", mem1, &handle), HCCL_SUCCESS);

    CommMem mem2 = MakeMem((void*)0x2000, 1024);
    void* handle2 = nullptr;
    EXPECT_EQ(commMems_->CommRegMem("mytag", mem2, &handle2), HCCL_E_PARA);
}

// 自验证: same tag + diff size → 返回 HCCL_E_PARA
TEST_F(CommRegMemTest, SameTagDiffSizeReturnsError)
{
    CommMem mem1 = MakeMem((void*)0x1000, 1024);
    void* handle = nullptr;
    EXPECT_EQ(commMems_->CommRegMem("mytag", mem1, &handle), HCCL_SUCCESS);

    CommMem mem2 = MakeMem((void*)0x1000, 2048); // same addr, different size
    void* handle2 = nullptr;
    EXPECT_EQ(commMems_->CommRegMem("mytag", mem2, &handle2), HCCL_E_PARA);
}

// 自验证: diff tag + same addr → 允许重叠注册, 返回 HCCL_SUCCESS
TEST_F(CommRegMemTest, DiffTagSameAddrReturnsSuccess)
{
    CommMem mem = MakeMem((void*)0x1000, 1024);
    void* handle1 = nullptr;
    void* handle2 = nullptr;

    EXPECT_EQ(commMems_->CommRegMem("tag1", mem, &handle1), HCCL_SUCCESS);
    ASSERT_NE(handle1, nullptr);
    EXPECT_EQ(commMems_->CommRegMem("tag2", mem, &handle2), HCCL_SUCCESS);
    ASSERT_NE(handle2, nullptr);
    // 两个不同 tag 应生成各自独立的句柄
    EXPECT_NE(handle1, handle2);
}

// 自验证: GetTagMemoryHandles tag 映射正确
TEST_F(CommRegMemTest, GetTagMemoryHandlesTagMapping)
{
    HcclMem cclBuffer;
    cclBuffer.addr = (void*)0x5000;
    cclBuffer.size = 4096;
    cclBuffer.type = HCCL_MEM_TYPE_DEVICE;
    commMems_->Init(cclBuffer);

    CommMem mem1 = MakeMem((void*)0x1000, 1024);
    void* handle1 = nullptr;
    EXPECT_EQ(commMems_->CommRegMem("userTagA", mem1, &handle1), HCCL_SUCCESS);
    ASSERT_NE(handle1, nullptr);

    CommMem mem2 = MakeMem((void*)0x3000, 2048);
    void* handle2 = nullptr;
    EXPECT_EQ(commMems_->CommRegMem("userTagB", mem2, &handle2), HCCL_SUCCESS);
    ASSERT_NE(handle2, nullptr);

    std::vector<std::string> memTags;
    void* handles[] = {handle1, handle2};
    EXPECT_EQ(commMems_->GetTagsFromHandles(handles, 2, memTags), HCCL_SUCCESS);

    // 预期: 1 个 HcclBuffer tag + 2 个用户注册 tag → 共 3 条
    ASSERT_EQ(memTags.size(), 3u);
    EXPECT_EQ(memTags[0], "HcclBuffer");
    EXPECT_EQ(memTags[1], "userTagA");
    EXPECT_EQ(memTags[2], "userTagB");
}

// 自验证: 非法入参返回错误
TEST_F(CommRegMemTest, InvalidParamsReturnError)
{
    void* handle = nullptr;
    CommMem zeroSizeMem = MakeMem((void*)0x1000, 0);
    EXPECT_EQ(commMems_->CommRegMem("tag", zeroSizeMem, &handle), HCCL_E_PARA);

    CommMem nullAddrMem = MakeMem(nullptr, 1024);
    EXPECT_EQ(commMems_->CommRegMem("tag", nullAddrMem, &handle), HCCL_E_PARA);

    EXPECT_EQ(commMems_->CommRegMem("tag", MakeMem((void*)0x1000, 1024), nullptr), HCCL_E_PARA);
}

TEST_F(CommRegMemTest, InitWithCclBufferAndGetSizeSuccess)
{
    HcclMem cclBuffer;
    cclBuffer.addr = (void*)0x7000;
    cclBuffer.size = 8192;
    cclBuffer.type = HCCL_MEM_TYPE_DEVICE;
    EXPECT_EQ(commMems_->Init(cclBuffer), HCCL_SUCCESS);

    void* addr = nullptr;
    uint64_t len = 0;
    EXPECT_EQ(commMems_->GetHcclBuffer(addr, len), HCCL_SUCCESS);
    EXPECT_EQ(addr, (void*)0x7000);
    EXPECT_EQ(len, 8192u);
}

TEST_F(CommRegMemTest, InitWithHostMemTypeSuccess)
{
    HcclMem cclBuffer;
    cclBuffer.addr = (void*)0x8000;
    cclBuffer.size = 2048;
    cclBuffer.type = HCCL_MEM_TYPE_HOST;
    EXPECT_EQ(commMems_->Init(cclBuffer), HCCL_SUCCESS);

    std::vector<HcclMem> memVec;
    std::vector<std::string> memTags;
    uint64_t ver = 0;
    EXPECT_EQ(commMems_->GetAllMemory(memVec, memTags, ver), HCCL_SUCCESS);
    ASSERT_EQ(memVec.size(), 1u);
    EXPECT_EQ(memVec[0].type, HCCL_MEM_TYPE_HOST);
}

TEST_F(CommRegMemTest, HcclBufferMemsetSkipWhenClearFlagFalse)
{
    void* addr = (void*)0x9000;
    uint64_t len = 1024;
    EXPECT_EQ(commMems_->HcclBufferMemset(addr, len, false), HCCL_SUCCESS);
}

TEST_F(CommRegMemTest, CommUnregMemSuccess)
{
    CommMem mem = MakeMem((void*)0x1000, 1024);
    void* handle = nullptr;
    EXPECT_EQ(commMems_->CommRegMem("mytag", mem, &handle), HCCL_SUCCESS);
    ASSERT_NE(handle, nullptr);

    EXPECT_EQ(commMems_->CommUnregMem("mytag", handle), HCCL_SUCCESS);
}

TEST_F(CommRegMemTest, CommUnregMemNullHandleReturnsError)
{
    EXPECT_EQ(commMems_->CommUnregMem("tag", nullptr), HCCL_E_PARA);
}

TEST_F(CommRegMemTest, CommUnregMemEmptyTagReturnsError)
{
    CommMem mem = MakeMem((void*)0x1000, 1024);
    void* handle = nullptr;
    EXPECT_EQ(commMems_->CommRegMem("tag", mem, &handle), HCCL_SUCCESS);
    EXPECT_EQ(commMems_->CommUnregMem("", handle), HCCL_E_PARA);
}

TEST_F(CommRegMemTest, CommUnregMemNotFoundTagReturnsError)
{
    CommMem mem = MakeMem((void*)0x1000, 1024);
    void* handle = nullptr;
    EXPECT_EQ(commMems_->CommRegMem("tag", mem, &handle), HCCL_SUCCESS);
    EXPECT_EQ(commMems_->CommUnregMem("notexist", handle), HCCL_E_NOT_FOUND);
}

TEST_F(CommRegMemTest, TagTooLongReturnsError)
{
    std::string longTag(256, 'a');
    CommMem mem = MakeMem((void*)0x1000, 1024);
    void* handle = nullptr;
    EXPECT_EQ(commMems_->CommRegMem(longTag, mem, &handle), HCCL_E_PARA);
}

TEST_F(CommRegMemTest, GetTagMemoryHandlesNullHandleReturnsError)
{
    HcclMem cclBuffer;
    cclBuffer.addr = (void*)0x5000;
    cclBuffer.size = 4096;
    cclBuffer.type = HCCL_MEM_TYPE_DEVICE;
    commMems_->Init(cclBuffer);

    void* handles[1] = {nullptr};
    std::vector<std::string> memTags;
    EXPECT_EQ(commMems_->GetTagsFromHandles(handles, 1, memTags), HCCL_E_NOT_FOUND);
}

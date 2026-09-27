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
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>

#include "comm_mem_manager.h"

using namespace hccl;

class CommMemMgrTest : public testing::Test {
protected:
    CommMemMgr mgr;

    void SetUp() override {}
    void TearDown() override { GlobalMockObject::verify(); }

    // 注册一块内存并返回句柄
    HcclMemHandle RegMem(const std::string& tag, void* addr, uint64_t size)
    {
        HcclMem mem;
        mem.addr = addr;
        mem.size = size;
        mem.type = HcclMemType::HCCL_MEM_TYPE_DEVICE;
        void* handle = nullptr;
        HcclResult ret = mgr.CommRegMem(tag, mem, HcclRegMemAttr{}, &handle);
        EXPECT_EQ(ret, HCCL_SUCCESS);
        return static_cast<HcclMemHandle>(handle);
    }
};

// ============ CommGetLocalRegMemByHandles 相关测试 ============

TEST_F(CommMemMgrTest, CommGetLocalRegMemByHandles_Normal)
{
    HcclMemHandle h1 = RegMem("tagA", (void*)0x3000, 1024);
    HcclMemHandle h2 = RegMem("tagB", (void*)0x4000, 2048);

    HcclMemHandle handles[2] = {h1, h2};
    std::vector<HcclMem> memVec;
    HcclResult ret = mgr.CommGetLocalRegMemByHandles(handles, 2, memVec);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(memVec.size(), 2u);
    EXPECT_EQ(memVec[0].addr, (void*)0x3000);
    EXPECT_EQ(memVec[0].size, 1024u);
    EXPECT_EQ(memVec[1].addr, (void*)0x4000);
    EXPECT_EQ(memVec[1].size, 2048u);
}

TEST_F(CommMemMgrTest, CommGetLocalRegMemByHandles_CrossTag)
{
    // 两个 handle 在不同 tag 下注册，也应全部命中
    HcclMemHandle h1 = RegMem("tagA", (void*)0x3000, 1024);
    HcclMemHandle h2 = RegMem("tagB", (void*)0x4000, 2048);

    HcclMemHandle handles[2] = {h2, h1}; // 乱序查询，结果顺序应与查询顺序一致
    std::vector<HcclMem> memVec;
    HcclResult ret = mgr.CommGetLocalRegMemByHandles(handles, 2, memVec);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(memVec.size(), 2u);
    EXPECT_EQ(memVec[0].addr, (void*)0x4000);
    EXPECT_EQ(memVec[1].addr, (void*)0x3000);
}

TEST_F(CommMemMgrTest, CommGetLocalRegMemByHandles_SameTagMultipleMems)
{
    // 同一 tag 下注册多块内存
    HcclMemHandle h1 = RegMem("tagA", (void*)0x3000, 1024);
    HcclMemHandle h2 = RegMem("tagA", (void*)0x4000, 2048);

    HcclMemHandle handles[2] = {h1, h2};
    std::vector<HcclMem> memVec;
    HcclResult ret = mgr.CommGetLocalRegMemByHandles(handles, 2, memVec);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(memVec.size(), 2u);
    EXPECT_EQ(memVec[0].addr, (void*)0x3000);
    EXPECT_EQ(memVec[1].addr, (void*)0x4000);
}

TEST_F(CommMemMgrTest, CommGetLocalRegMemByHandles_NotFound)
{
    HcclMemHandle h1 = RegMem("tagA", (void*)0x3000, 1024);
    HcclMemHandle unknown = reinterpret_cast<HcclMemHandle>(0x9999);

    HcclMemHandle handles[2] = {h1, unknown};
    std::vector<HcclMem> memVec;
    HcclResult ret = mgr.CommGetLocalRegMemByHandles(handles, 2, memVec);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
    // 失败时 memVec 仍包含已命中的项
    ASSERT_EQ(memVec.size(), 1u);
    EXPECT_EQ(memVec[0].addr, (void*)0x3000);
}

TEST_F(CommMemMgrTest, CommGetLocalRegMemByHandles_AllNotFound)
{
    HcclMemHandle unknown1 = reinterpret_cast<HcclMemHandle>(0x9999);
    HcclMemHandle unknown2 = reinterpret_cast<HcclMemHandle>(0x8888);

    HcclMemHandle handles[2] = {unknown1, unknown2};
    std::vector<HcclMem> memVec;
    HcclResult ret = mgr.CommGetLocalRegMemByHandles(handles, 2, memVec);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
    EXPECT_TRUE(memVec.empty());
}

TEST_F(CommMemMgrTest, CommGetLocalRegMemByHandles_NullPtr)
{
    std::vector<HcclMem> memVec;
    // CHK_PTR_NULL 校验，返回 HCCL_E_PTR
    HcclResult ret = mgr.CommGetLocalRegMemByHandles(nullptr, 1, memVec);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(CommMemMgrTest, CommGetLocalRegMemByHandles_ZeroNum)
{
    HcclMemHandle handle = RegMem("tagA", (void*)0x3000, 1024);
    HcclMemHandle handles[1] = {handle};
    std::vector<HcclMem> memVec;
    HcclResult ret = mgr.CommGetLocalRegMemByHandles(handles, 0, memVec);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(memVec.empty());
}

TEST_F(CommMemMgrTest, CommGetLocalRegMemByHandles_MemTypePreserved)
{
    HcclMem mem;
    mem.addr = (void*)0x5000;
    mem.size = 4096;
    mem.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    void* handle = nullptr;
    HcclResult regRet = mgr.CommRegMem("tagHost", mem, HcclRegMemAttr{}, &handle);
    EXPECT_EQ(regRet, HCCL_SUCCESS);

    HcclMemHandle handles[1] = {static_cast<HcclMemHandle>(handle)};
    std::vector<HcclMem> memVec;
    HcclResult ret = mgr.CommGetLocalRegMemByHandles(handles, 1, memVec);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(memVec.size(), 1u);
    EXPECT_EQ(memVec[0].addr, (void*)0x5000);
    EXPECT_EQ(memVec[0].size, 4096u);
    EXPECT_EQ(memVec[0].type, HcclMemType::HCCL_MEM_TYPE_HOST);
}

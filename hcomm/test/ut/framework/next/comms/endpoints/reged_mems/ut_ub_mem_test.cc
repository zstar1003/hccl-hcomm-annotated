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
#include "hcomm_c_adpt.h"
#include "hcomm_res_defs.h"

#define private public
#define protected public

#include "ub_mem_reged_mem_mgr.h"

#undef protected
#undef private

using namespace hcomm;

class UbMemRegedMemMgrTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "UbMemRegedMemMgrTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "UbMemRegedMemMgrTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in UbMemRegedMemMgrTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in UbMemRegedMemMgrTest TearDown" << std::endl;
    }
};

template <typename Mgr, typename Key>
static uint64_t GetRef(Mgr& mgr, const Key& key)
{
    for (auto it = mgr.Begin(); it != mgr.End(); ++it) {
        if (it->first == key) {
            return it->second.ref;
        }
    }
    return 0;
}

// 父+子集注册 → 先解注册子(alias) → 再解注册父
TEST_F(UbMemRegedMemMgrTest, ut_UbMemRegedMemMgr_When_ParentChild_Expect_UnregisterChildFirst)
{
    UbMemRegedMemMgr ubMemRegedMemMgr{};
    HcommMem mem0;
    mem0.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem0.addr = (void*)0x2000;
    mem0.size = 8192;
    std::string memTag0 = "parent";
    void* memHandle0 = nullptr;
    HcclResult ret = ubMemRegedMemMgr.RegisterMemory(&mem0, memTag0.c_str(), &memHandle0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    auto* parentBuf = static_cast<Hccl::LocalIpcRmaBuffer*>(memHandle0);
    hccl::BufferKey<uintptr_t, u64> parentKey(parentBuf->GetAddr(), static_cast<uint64_t>(parentBuf->GetSize()));
    EXPECT_TRUE(ubMemRegedMemMgr.localIpcRmaBufferMgr_->Find(parentKey).first);
    EXPECT_EQ(GetRef(*ubMemRegedMemMgr.localIpcRmaBufferMgr_, parentKey), 1u);

    // 子集注册（应创建alias buffer，共享父IPC资源）
    HcommMem mem1;
    mem1.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem1.addr = (void*)0x2000;
    mem1.size = 512;
    std::string memTag1 = "child";
    void* memHandle1 = nullptr;
    ret = ubMemRegedMemMgr.RegisterMemory(&mem1, memTag1.c_str(), &memHandle1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(memHandle1, memHandle0);
    EXPECT_EQ(GetRef(*ubMemRegedMemMgr.localIpcRmaBufferMgr_, parentKey), 2u);

    // 解注册子（ref 2→1）
    ret = ubMemRegedMemMgr.UnregisterMemory(memHandle1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(ubMemRegedMemMgr.localIpcRmaBufferMgr_->Find(parentKey).first);
    EXPECT_EQ(GetRef(*ubMemRegedMemMgr.localIpcRmaBufferMgr_, parentKey), 1u);

    // 解注册父（ref 1→0）
    ret = ubMemRegedMemMgr.UnregisterMemory(memHandle0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_FALSE(ubMemRegedMemMgr.localIpcRmaBufferMgr_->Find(parentKey).first);
    EXPECT_EQ(GetRef(*ubMemRegedMemMgr.localIpcRmaBufferMgr_, parentKey), 0u);
}

// 多次注册同一范围，各自解注册
TEST_F(UbMemRegedMemMgrTest, ut_UbMemRegedMemMgr_When_MultipleSameRange_Expect_AllSuccess)
{
    UbMemRegedMemMgr ubMemRegedMemMgr{};
    HcommMem mem;
    mem.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem.addr = (void*)0x4000;
    mem.size = 4096;

    void *h1 = nullptr, *h2 = nullptr, *h3 = nullptr;
    EXPECT_EQ(ubMemRegedMemMgr.RegisterMemory(&mem, "t1", &h1), HCCL_SUCCESS);
    auto* parentBuf = static_cast<Hccl::LocalIpcRmaBuffer*>(h1);
    hccl::BufferKey<uintptr_t, u64> parentKey(parentBuf->GetAddr(), static_cast<uint64_t>(parentBuf->GetSize()));
    EXPECT_EQ(GetRef(*ubMemRegedMemMgr.localIpcRmaBufferMgr_, parentKey), 1u);

    EXPECT_EQ(ubMemRegedMemMgr.RegisterMemory(&mem, "t2", &h2), HCCL_SUCCESS);
    EXPECT_NE(h2, h1);
    EXPECT_EQ(GetRef(*ubMemRegedMemMgr.localIpcRmaBufferMgr_, parentKey), 2u);

    EXPECT_EQ(ubMemRegedMemMgr.RegisterMemory(&mem, "t3", &h3), HCCL_SUCCESS);
    EXPECT_NE(h3, h1);
    EXPECT_EQ(GetRef(*ubMemRegedMemMgr.localIpcRmaBufferMgr_, parentKey), 3u);

    // ref 3→2
    EXPECT_EQ(ubMemRegedMemMgr.UnregisterMemory(h3), HCCL_SUCCESS);
    EXPECT_EQ(GetRef(*ubMemRegedMemMgr.localIpcRmaBufferMgr_, parentKey), 2u);
    // ref 2→1
    EXPECT_EQ(ubMemRegedMemMgr.UnregisterMemory(h2), HCCL_SUCCESS);
    EXPECT_EQ(GetRef(*ubMemRegedMemMgr.localIpcRmaBufferMgr_, parentKey), 1u);
    // ref 1→0
    EXPECT_EQ(ubMemRegedMemMgr.UnregisterMemory(h1), HCCL_SUCCESS);
    EXPECT_EQ(GetRef(*ubMemRegedMemMgr.localIpcRmaBufferMgr_, parentKey), 0u);
}

// 注册不重叠范围 → 应创建两个独立父buffer，各自ref=1
TEST_F(UbMemRegedMemMgrTest, ut_UbMemRegedMemMgr_When_NonOverlap_Expect_TwoParents)
{
    UbMemRegedMemMgr ubMemRegedMemMgr{};
    HcommMem mem0;
    mem0.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem0.addr = (void*)0x1000;
    mem0.size = 100;
    void* h0 = nullptr;
    EXPECT_EQ(ubMemRegedMemMgr.RegisterMemory(&mem0, "t0", &h0), HCCL_SUCCESS);
    auto* buf0 = static_cast<Hccl::LocalIpcRmaBuffer*>(h0);
    hccl::BufferKey<uintptr_t, u64> key0(buf0->GetAddr(), static_cast<uint64_t>(buf0->GetSize()));
    EXPECT_EQ(GetRef(*ubMemRegedMemMgr.localIpcRmaBufferMgr_, key0), 1u);

    // 不重叠的新范围
    HcommMem mem1;
    mem1.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem1.addr = (void*)0x2000;
    mem1.size = 100;
    void* h1 = nullptr;
    EXPECT_EQ(ubMemRegedMemMgr.RegisterMemory(&mem1, "t1", &h1), HCCL_SUCCESS);
    auto* buf1 = static_cast<Hccl::LocalIpcRmaBuffer*>(h1);
    hccl::BufferKey<uintptr_t, u64> key1(buf1->GetAddr(), static_cast<uint64_t>(buf1->GetSize()));
    EXPECT_EQ(GetRef(*ubMemRegedMemMgr.localIpcRmaBufferMgr_, key1), 1u);

    EXPECT_NE(h0, h1);
    // 两个独立key都在树中
    EXPECT_EQ(GetRef(*ubMemRegedMemMgr.localIpcRmaBufferMgr_, key0), 1u);
    EXPECT_EQ(GetRef(*ubMemRegedMemMgr.localIpcRmaBufferMgr_, key1), 1u);

    // 解注册h0
    EXPECT_EQ(ubMemRegedMemMgr.UnregisterMemory(h0), HCCL_SUCCESS);
    EXPECT_EQ(GetRef(*ubMemRegedMemMgr.localIpcRmaBufferMgr_, key0), 0u);
    EXPECT_EQ(GetRef(*ubMemRegedMemMgr.localIpcRmaBufferMgr_, key1), 1u);

    // 解注册h1
    EXPECT_EQ(ubMemRegedMemMgr.UnregisterMemory(h1), HCCL_SUCCESS);
    EXPECT_EQ(GetRef(*ubMemRegedMemMgr.localIpcRmaBufferMgr_, key1), 0u);
}

// 父+子集注册 → 先解注册父(soft-delete) → 验证allBuffers_软删除标记 → 再解注册子
TEST_F(UbMemRegedMemMgrTest, ut_UbMemRegedMemMgr_When_UnregisterParentFirst_Expect_ParentSoftDeleted)
{
    UbMemRegedMemMgr ubMemRegedMemMgr{};
    HcommMem mem0;
    mem0.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem0.addr = (void*)0x6000;
    mem0.size = 4096;
    void* parentHandle = nullptr;
    EXPECT_EQ(ubMemRegedMemMgr.RegisterMemory(&mem0, "parent", &parentHandle), HCCL_SUCCESS);
    auto* parentBuf = static_cast<Hccl::LocalIpcRmaBuffer*>(parentHandle);
    hccl::BufferKey<uintptr_t, u64> parentKey(parentBuf->GetAddr(), static_cast<uint64_t>(parentBuf->GetSize()));
    EXPECT_EQ(GetRef(*ubMemRegedMemMgr.localIpcRmaBufferMgr_, parentKey), 1u);

    // 子集注册（alias）
    HcommMem mem1;
    mem1.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem1.addr = (void*)0x6000;
    mem1.size = 512;
    void* childHandle = nullptr;
    EXPECT_EQ(ubMemRegedMemMgr.RegisterMemory(&mem1, "child", &childHandle), HCCL_SUCCESS);
    EXPECT_NE(childHandle, parentHandle);
    EXPECT_EQ(GetRef(*ubMemRegedMemMgr.localIpcRmaBufferMgr_, parentKey), 2u);
    EXPECT_EQ(ubMemRegedMemMgr.allRegisteredBuffers_.size(), 2u);

    // 先解注册父（ref 2→1，IsInTree=true，父标记为soft-deleted）
    EXPECT_EQ(ubMemRegedMemMgr.UnregisterMemory(parentHandle), HCCL_SUCCESS);
    EXPECT_TRUE(ubMemRegedMemMgr.localIpcRmaBufferMgr_->Find(parentKey).first);
    EXPECT_EQ(GetRef(*ubMemRegedMemMgr.localIpcRmaBufferMgr_, parentKey), 1u);

    // 验证：父在allBuffers_中标记为true（soft-deleted），子标记为false
    EXPECT_EQ(ubMemRegedMemMgr.allRegisteredBuffers_.size(), 2u);
    auto itParent = std::find_if(
        ubMemRegedMemMgr.allRegisteredBuffers_.begin(), ubMemRegedMemMgr.allRegisteredBuffers_.end(),
        [parentBuf](const auto& e) {
            return e.first.get() == parentBuf;
        });
    ASSERT_NE(itParent, ubMemRegedMemMgr.allRegisteredBuffers_.end());
    EXPECT_TRUE(itParent->second);

    auto itChild = std::find_if(
        ubMemRegedMemMgr.allRegisteredBuffers_.begin(), ubMemRegedMemMgr.allRegisteredBuffers_.end(),
        [childHandle](const auto& e) {
            return e.first.get() == childHandle;
        });
    ASSERT_NE(itChild, ubMemRegedMemMgr.allRegisteredBuffers_.end());
    EXPECT_FALSE(itChild->second);

    // 再解注册子（ref 1→0，tree entry removed，子从allBuffers_擦除）
    EXPECT_EQ(ubMemRegedMemMgr.UnregisterMemory(childHandle), HCCL_SUCCESS);
    EXPECT_FALSE(ubMemRegedMemMgr.localIpcRmaBufferMgr_->Find(parentKey).first);
    EXPECT_EQ(GetRef(*ubMemRegedMemMgr.localIpcRmaBufferMgr_, parentKey), 0u);

    // 验证：子已从allBuffers_中移除
    itChild = std::find_if(
        ubMemRegedMemMgr.allRegisteredBuffers_.begin(), ubMemRegedMemMgr.allRegisteredBuffers_.end(),
        [childHandle](const auto& e) {
            return e.first.get() == childHandle;
        });
    EXPECT_EQ(itChild, ubMemRegedMemMgr.allRegisteredBuffers_.end());
}

// RegisterMemory: 非法入参（null addr / zero size / invalid type）→ 返回错误
TEST_F(UbMemRegedMemMgrTest, ut_UbMemRegedMemMgr_When_InvalidParams_Expect_Error)
{
    UbMemRegedMemMgr ubMemRegedMemMgr{};
    void* h = nullptr;

    HcommMem mem0{};
    mem0.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem0.addr = nullptr;
    mem0.size = 4096;
    EXPECT_EQ(ubMemRegedMemMgr.RegisterMemory(&mem0, "t", &h), HCCL_E_PTR);

    HcommMem mem1{};
    mem1.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem1.addr = (void*)0x1000;
    mem1.size = 0;
    EXPECT_EQ(ubMemRegedMemMgr.RegisterMemory(&mem1, "t", &h), HCCL_E_PARA);

    HcommMem mem2{};
    mem2.type = COMM_MEM_TYPE_INVALID;
    mem2.addr = (void*)0x1000;
    mem2.size = 4096;
    EXPECT_EQ(ubMemRegedMemMgr.RegisterMemory(&mem2, "t", &h), HCCL_E_PARA);
}

// UnregisterMemory: null memHandle → 返回错误
TEST_F(UbMemRegedMemMgrTest, ut_UbMemRegedMemMgr_When_UnregisterNullHandle_Expect_Error)
{
    UbMemRegedMemMgr ubMemRegedMemMgr{};
    EXPECT_EQ(ubMemRegedMemMgr.UnregisterMemory(nullptr), HCCL_E_PTR);
}

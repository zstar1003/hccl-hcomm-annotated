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
#include "invalid_params_exception.h"
#include "null_ptr_exception.h"
#include "endpoint_pair.h"

#define private public
#define protected public

#include "ub_reged_mem_mgr.h"

#undef protected
#undef private

using namespace hcomm;

class UrmaRegedMemMgrTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "UrmaRegedMemMgrTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "UrmaRegedMemMgrTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in UrmaRegedMemMgrTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in UrmaRegedMemMgrTest TearDown" << std::endl;
    }
};

// Helper: 遍历树查找指定key的引用计数，未找到返回0
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

TEST_F(UrmaRegedMemMgrTest, Ut_LocalUbRmaBufferAlias_When_Constructed_Expect_ReusesParentRegParam)
{
    auto parentBuf = std::make_shared<Hccl::Buffer>(0xD000, 0x100, HCCL_MEM_TYPE_DEVICE, "ub_parent");
    Hccl::LocalUbRmaBuffer parent(parentBuf);
    parent.reqReg.handle = 0x11U;
    parent.reqReg.keySize = 4U;
    parent.reqReg.targetSegVa = 0x2200U;
    parent.reqReg.key[0] = 1U;
    parent.reqReg.key[1] = 2U;
    parent.reqReg.key[2] = 3U;
    parent.reqReg.key[3] = 4U;

    auto childBuf = std::make_shared<Hccl::Buffer>(0xD040, 0x40, HCCL_MEM_TYPE_DEVICE, "ub_child");
    Hccl::LocalUbRmaBuffer child(childBuf, reinterpret_cast<RdmaHandle>(0x1), parent);

    EXPECT_TRUE(child.IsAlias());
    EXPECT_EQ(child.GetTokenId(), parent.GetTokenId());
    EXPECT_EQ(child.GetTokenValue(), parent.GetTokenValue());
    EXPECT_EQ(child.GetTargetSeg(), parent.GetTargetSeg());
    EXPECT_EQ(child.GetBufferInfo().first, 0xD040U);
    EXPECT_EQ(child.GetBufferInfo().second, 0x40U);
    EXPECT_TRUE(Hccl::LocalUbRmaBuffer::IsSameMemRegOutParam(child.GetMemRegOutParam(), parent.GetMemRegOutParam()));
}

// 父+子集注册 → 先解注册子 → 再解注册父
TEST_F(UrmaRegedMemMgrTest, ut_UbRegedMemMgr_URMA_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    UbRegedMemMgr ubRegedMemMgr{nullptr};
    HcommMem mem0;
    mem0.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem0.addr = (void*)0x100;
    mem0.size = 100;
    std::string memTag0 = "buffer0";
    void* memHandle0 = nullptr;
    HcclResult ret = ubRegedMemMgr.RegisterMemory(&mem0, memTag0.c_str(), &memHandle0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 验证父buffer已注册，ref=1
    auto* parentBuf = static_cast<Hccl::LocalUbRmaBuffer*>(memHandle0);
    hccl::BufferKey<uintptr_t, u64> parentKey(parentBuf->GetAddr(), static_cast<uint64_t>(parentBuf->GetSize()));
    EXPECT_TRUE(ubRegedMemMgr.localUbRmaBufferMgr_->Find(parentKey).first);
    EXPECT_EQ(GetRef(*ubRegedMemMgr.localUbRmaBufferMgr_, parentKey), 1u);

    HcommMem mem1;
    mem1.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem1.addr = (void*)0x100;
    mem1.size = 10;
    std::string memTag1 = "buffer1";
    void* memHandle1 = nullptr;
    ret = ubRegedMemMgr.RegisterMemory(&mem1, memTag1.c_str(), &memHandle1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(memHandle1, memHandle0);
    // 子alias注册后，父ref 1→2
    EXPECT_EQ(GetRef(*ubRegedMemMgr.localUbRmaBufferMgr_, parentKey), 2u);

    // 先解注册子（父ref 2→1，不删除）
    ret = ubRegedMemMgr.UnregisterMemory(memHandle1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(ubRegedMemMgr.localUbRmaBufferMgr_->Find(parentKey).first);
    EXPECT_EQ(GetRef(*ubRegedMemMgr.localUbRmaBufferMgr_, parentKey), 1u);

    // 再解注册父（ref 1→0，删除）
    ret = ubRegedMemMgr.UnregisterMemory(memHandle0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_FALSE(ubRegedMemMgr.localUbRmaBufferMgr_->Find(parentKey).first);
    EXPECT_EQ(GetRef(*ubRegedMemMgr.localUbRmaBufferMgr_, parentKey), 0u);
}

// 多次注册同一范围，各自解注册
TEST_F(UrmaRegedMemMgrTest, ut_UbRegedMemMgr_URMA_When_MultipleSameRange_Expect_AllSuccess)
{
    UbRegedMemMgr ubRegedMemMgr{nullptr};
    HcommMem mem;
    mem.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem.addr = (void*)0x3000;
    mem.size = 1024;

    void *h1 = nullptr, *h2 = nullptr, *h3 = nullptr;
    ASSERT_EQ(ubRegedMemMgr.RegisterMemory(&mem, "t1", &h1), HCCL_SUCCESS);
    auto* parentBuf = static_cast<Hccl::LocalUbRmaBuffer*>(h1);
    hccl::BufferKey<uintptr_t, u64> parentKey(parentBuf->GetAddr(), static_cast<uint64_t>(parentBuf->GetSize()));
    EXPECT_EQ(GetRef(*ubRegedMemMgr.localUbRmaBufferMgr_, parentKey), 1u);

    ASSERT_EQ(ubRegedMemMgr.RegisterMemory(&mem, "t2", &h2), HCCL_SUCCESS);
    EXPECT_NE(h2, h1);
    EXPECT_EQ(GetRef(*ubRegedMemMgr.localUbRmaBufferMgr_, parentKey), 2u);

    ASSERT_EQ(ubRegedMemMgr.RegisterMemory(&mem, "t3", &h3), HCCL_SUCCESS);
    EXPECT_NE(h3, h1);
    EXPECT_EQ(GetRef(*ubRegedMemMgr.localUbRmaBufferMgr_, parentKey), 3u);

    // h3解注册，ref 3→2
    EXPECT_EQ(ubRegedMemMgr.UnregisterMemory(h3), HCCL_SUCCESS);
    EXPECT_EQ(GetRef(*ubRegedMemMgr.localUbRmaBufferMgr_, parentKey), 2u);

    // h2解注册，ref 2→1
    EXPECT_EQ(ubRegedMemMgr.UnregisterMemory(h2), HCCL_SUCCESS);
    EXPECT_EQ(GetRef(*ubRegedMemMgr.localUbRmaBufferMgr_, parentKey), 1u);

    // h1解注册，ref 1→0，删除
    EXPECT_EQ(ubRegedMemMgr.UnregisterMemory(h1), HCCL_SUCCESS);
    EXPECT_EQ(GetRef(*ubRegedMemMgr.localUbRmaBufferMgr_, parentKey), 0u);
}

TEST_F(UrmaRegedMemMgrTest, Ut_MemoryExport_When_MemHandleUnregistered_Expect_NotFound)
{
    UbRegedMemMgr ubRegedMemMgr{nullptr};
    EndpointDesc endpointDesc{};

    auto buf = std::make_shared<Hccl::Buffer>(0xD800, 0x100, HCCL_MEM_TYPE_DEVICE, "ub_export");
    auto localUbRmaBuffer = std::make_shared<Hccl::LocalUbRmaBuffer>(buf);
    void* memHandle = localUbRmaBuffer.get();

    ubRegedMemMgr.allRegisteredBuffers_.emplace_back(localUbRmaBuffer, false);
    ubRegedMemMgr.allRegisteredBuffers_.clear();

    void* memDesc = nullptr;
    uint32_t memDescLen = 0;
    EXPECT_EQ(ubRegedMemMgr.MemoryExport(endpointDesc, memHandle, &memDesc, &memDescLen), HCCL_E_NOT_FOUND);
    EXPECT_EQ(memDesc, nullptr);
    EXPECT_EQ(memDescLen, 0U);
}

// 父+子集注册 → 先解注册父(soft-delete) → 验证handlesRecords_和allBuffers_状态 → 再解注册子
TEST_F(UrmaRegedMemMgrTest, ut_UbRegedMemMgr_URMA_When_UnregisterParentFirst_Expect_ParentSoftDeleted)
{
    UbRegedMemMgr ubRegedMemMgr{nullptr};
    HcommMem mem0;
    mem0.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem0.addr = (void*)0x7000;
    mem0.size = 4096;
    void* parentHandle = nullptr;
    ASSERT_EQ(ubRegedMemMgr.RegisterMemory(&mem0, "parent", &parentHandle), HCCL_SUCCESS);
    auto* parentBuf = static_cast<Hccl::LocalUbRmaBuffer*>(parentHandle);
    hccl::BufferKey<uintptr_t, u64> parentKey(parentBuf->GetAddr(), static_cast<uint64_t>(parentBuf->GetSize()));
    EXPECT_EQ(GetRef(*ubRegedMemMgr.localUbRmaBufferMgr_, parentKey), 1u);
    EXPECT_EQ(ubRegedMemMgr.handlesRecords_.size(), 1u);

    // 子集注册（alias）
    HcommMem mem1;
    mem1.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem1.addr = (void*)0x7000;
    mem1.size = 512;
    void* childHandle = nullptr;
    ASSERT_EQ(ubRegedMemMgr.RegisterMemory(&mem1, "child", &childHandle), HCCL_SUCCESS);
    EXPECT_NE(childHandle, parentHandle);
    EXPECT_EQ(GetRef(*ubRegedMemMgr.localUbRmaBufferMgr_, parentKey), 2u);
    EXPECT_EQ(ubRegedMemMgr.handlesRecords_.size(), 2u);
    EXPECT_EQ(ubRegedMemMgr.allRegisteredBuffers_.size(), 2u);

    // 先解注册父（ref 2→1，IsInTree=true，父标记为soft-deleted）
    EXPECT_EQ(ubRegedMemMgr.UnregisterMemory(parentHandle), HCCL_SUCCESS);
    EXPECT_TRUE(ubRegedMemMgr.localUbRmaBufferMgr_->Find(parentKey).first);
    EXPECT_EQ(GetRef(*ubRegedMemMgr.localUbRmaBufferMgr_, parentKey), 1u);

    // 验证：handlesRecords_中已移除父，但allBuffers_中父标记为soft-deleted
    EXPECT_EQ(ubRegedMemMgr.handlesRecords_.size(), 1u);
    EXPECT_EQ(ubRegedMemMgr.allRegisteredBuffers_.size(), 2u);

    auto itParent = std::find_if(
        ubRegedMemMgr.allRegisteredBuffers_.begin(), ubRegedMemMgr.allRegisteredBuffers_.end(),
        [parentBuf](const auto& e) {
            return e.first.get() == parentBuf;
        });
    ASSERT_NE(itParent, ubRegedMemMgr.allRegisteredBuffers_.end());
    EXPECT_TRUE(itParent->second);

    auto itChild = std::find_if(
        ubRegedMemMgr.allRegisteredBuffers_.begin(), ubRegedMemMgr.allRegisteredBuffers_.end(),
        [childHandle](const auto& e) {
            return e.first.get() == childHandle;
        });
    ASSERT_NE(itChild, ubRegedMemMgr.allRegisteredBuffers_.end());
    EXPECT_FALSE(itChild->second);

    // 再解注册子（ref 1→0，tree entry removed，子从allBuffers_擦除）
    EXPECT_EQ(ubRegedMemMgr.UnregisterMemory(childHandle), HCCL_SUCCESS);
    EXPECT_FALSE(ubRegedMemMgr.localUbRmaBufferMgr_->Find(parentKey).first);
    EXPECT_EQ(GetRef(*ubRegedMemMgr.localUbRmaBufferMgr_, parentKey), 0u);
    EXPECT_EQ(ubRegedMemMgr.handlesRecords_.size(), 0u);

    // 验证：子已从allBuffers_中移除
    itChild = std::find_if(
        ubRegedMemMgr.allRegisteredBuffers_.begin(), ubRegedMemMgr.allRegisteredBuffers_.end(),
        [childHandle](const auto& e) {
            return e.first.get() == childHandle;
        });
    EXPECT_EQ(itChild, ubRegedMemMgr.allRegisteredBuffers_.end());
}

// GetAllMemHandles: 空记录 → 注册 → 多次注册 → 解注册 → 验证句柄数
TEST_F(UrmaRegedMemMgrTest, ut_UbRegedMemMgr_URMA_When_GetAllMemHandles_Expect_CorrectCount)
{
    UbRegedMemMgr ubRegedMemMgr{nullptr};

    void* handles = nullptr;
    uint32_t count = 99U;
    EXPECT_EQ(ubRegedMemMgr.GetAllMemHandles(&handles, &count), HCCL_SUCCESS);
    EXPECT_EQ(count, 0U);
    EXPECT_EQ(handles, nullptr);

    HcommMem mem;
    mem.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem.addr = (void*)0x8000;
    mem.size = 4096;
    void* h1 = nullptr;
    ASSERT_EQ(ubRegedMemMgr.RegisterMemory(&mem, "t1", &h1), HCCL_SUCCESS);

    EXPECT_EQ(ubRegedMemMgr.GetAllMemHandles(&handles, &count), HCCL_SUCCESS);
    EXPECT_EQ(count, 1U);
    EXPECT_NE(handles, nullptr);

    void* h2 = nullptr;
    ASSERT_EQ(ubRegedMemMgr.RegisterMemory(&mem, "t2", &h2), HCCL_SUCCESS);

    EXPECT_EQ(ubRegedMemMgr.GetAllMemHandles(&handles, &count), HCCL_SUCCESS);
    EXPECT_EQ(count, 2U);

    EXPECT_EQ(ubRegedMemMgr.UnregisterMemory(h1), HCCL_SUCCESS);
    EXPECT_EQ(ubRegedMemMgr.GetAllMemHandles(&handles, &count), HCCL_SUCCESS);
    EXPECT_EQ(count, 1U);

    EXPECT_EQ(ubRegedMemMgr.UnregisterMemory(h2), HCCL_SUCCESS);
    EXPECT_EQ(ubRegedMemMgr.GetAllMemHandles(&handles, &count), HCCL_SUCCESS);
    EXPECT_EQ(count, 0U);
}

// RegisterMemory: 非法入参（null addr / zero size / invalid type）→ 返回错误
TEST_F(UrmaRegedMemMgrTest, ut_UbRegedMemMgr_URMA_When_InvalidParams_Expect_Error)
{
    UbRegedMemMgr ubRegedMemMgr{nullptr};
    void* h = nullptr;

    // null mem.addr
    HcommMem mem0{};
    mem0.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem0.addr = nullptr;
    mem0.size = 4096;
    EXPECT_EQ(ubRegedMemMgr.RegisterMemory(&mem0, "t", &h), HCCL_E_PTR);

    // zero size
    HcommMem mem1{};
    mem1.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem1.addr = (void*)0x1000;
    mem1.size = 0;
    EXPECT_EQ(ubRegedMemMgr.RegisterMemory(&mem1, "t", &h), HCCL_E_PARA);

    // invalid type
    HcommMem mem2{};
    mem2.type = COMM_MEM_TYPE_INVALID;
    mem2.addr = (void*)0x1000;
    mem2.size = 4096;
    EXPECT_EQ(ubRegedMemMgr.RegisterMemory(&mem2, "t", &h), HCCL_E_PARA);
}

// UnregisterMemory: null memHandle → 返回错误
TEST_F(UrmaRegedMemMgrTest, ut_UbRegedMemMgr_URMA_When_UnregisterNullHandle_Expect_Error)
{
    UbRegedMemMgr ubRegedMemMgr{nullptr};
    EXPECT_EQ(ubRegedMemMgr.UnregisterMemory(nullptr), HCCL_E_PTR);
}

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
#include "endpoint.h"
#include "endpoint_pair.h"
#include "hcomm_res.h"
#include "hcomm_c_adpt.h"
#include "invalid_params_exception.h"
#include "null_ptr_exception.h"
#include "acl/acl_rt.h"
#include "hccp.h"
#include <mockcpp/mockcpp.hpp>

#define private public
#define protected public

#include "roce_reged_mem_mgr.h"

#undef protected
#undef private

using namespace hcomm;

namespace {

void StubRaMrApis(void* fakeMr = reinterpret_cast<void*>(0xABCDEF))
{
    MOCKER(RaRegisterMr)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&fakeMr, sizeof(fakeMr)))
        .will(returnValue(0));
    MOCKER(RaDeregisterMr).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(0));
}

} // namespace

class RoceRegedMemMgrTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RoceRegedMemMgrTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "RoceRegedMemMgrTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in RoceRegedMemMgrTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in RoceRegedMemMgrTest TearDown" << std::endl;
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

static uint64_t GetAllocRef(RoceRegedMemMgr& mgr, const RoceRegedMemMgr::MemKey& key)
{
    auto it = mgr.allocToMrMap_.find(key);
    return it == mgr.allocToMrMap_.end() ? 0u : it->second.ref;
}

TEST_F(RoceRegedMemMgrTest, Ut_GetParamsFromMemDesc_When_DescLenTooSmall_Expect_Return_Error)
{
    std::shared_ptr<RoceRegedMemMgr> roceRegedMemMgrPtr = std::make_shared<RoceRegedMemMgr>(nullptr);
    EndpointDesc endpointDesc;
    Hccl::ExchangeRdmaBufferDto dto;

    char buffer[10];
    uint32_t descLen = 10;

    HcclResult ret = roceRegedMemMgrPtr->GetParamsFromMemDesc(buffer, descLen, endpointDesc, dto);
    EXPECT_EQ(HCCL_E_INTERNAL, ret);
}

// 父+子集注册 → 先解注册子(alias) → 再解注册父（靠 AddressRange 归一到同一 alloc）
TEST_F(RoceRegedMemMgrTest, ut_RoceRegedMemMgr_When_ParentChild_Expect_UnregisterChildFirst)
{
    void* allocBase = reinterpret_cast<void*>(0x1000);
    size_t allocSize = 4096;
    MOCKER_CPP(aclrtMemGetAddressRange)
        .stubs()
        .with(mockcpp::any(), outBoundP(&allocBase, sizeof(allocBase)), outBoundP(&allocSize, sizeof(allocSize)))
        .will(returnValue(ACL_SUCCESS));
    StubRaMrApis();

    RoceRegedMemMgr roceRegedMemMgr{reinterpret_cast<RdmaHandle>(0x1), true};
    HcommMem mem0;
    mem0.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem0.addr = reinterpret_cast<void*>(0x1000);
    mem0.size = 4096;
    std::string memTag0 = "parent";
    void* memHandle0 = nullptr;
    HcclResult ret = roceRegedMemMgr.RegisterMemory(&mem0, memTag0.c_str(), &memHandle0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    auto* parentBuf = static_cast<Hccl::LocalRdmaRmaBuffer*>(memHandle0);
    ASSERT_NE(parentBuf, nullptr);
    // userKey == allocKey 且首次创建 MR：返回本体而非 alias
    EXPECT_FALSE(parentBuf->IsAlias());

    RoceRegedMemMgr::MemKey allocKey(reinterpret_cast<uintptr_t>(allocBase), static_cast<uint64_t>(allocSize));
    EXPECT_EQ(GetAllocRef(roceRegedMemMgr, allocKey), 1u);

    // 子集注册（应创建 alias，复用同一 alloc MR）
    HcommMem mem1;
    mem1.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem1.addr = reinterpret_cast<void*>(0x1000);
    mem1.size = 1024;
    std::string memTag1 = "child";
    void* memHandle1 = nullptr;
    ret = roceRegedMemMgr.RegisterMemory(&mem1, memTag1.c_str(), &memHandle1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(memHandle1, memHandle0);
    EXPECT_TRUE(static_cast<Hccl::LocalRdmaRmaBuffer*>(memHandle1)->IsAlias());
    EXPECT_EQ(GetAllocRef(roceRegedMemMgr, allocKey), 2u);

    // 解注册子（alloc ref 2→1）
    ret = roceRegedMemMgr.UnregisterMemory(memHandle1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(GetAllocRef(roceRegedMemMgr, allocKey), 1u);

    // 解注册父（alloc ref 1→0）
    ret = roceRegedMemMgr.UnregisterMemory(memHandle0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(GetAllocRef(roceRegedMemMgr, allocKey), 0u);
    EXPECT_TRUE(roceRegedMemMgr.allocToMrMap_.empty());
}

// 多次注册同一范围，各自解注册
TEST_F(RoceRegedMemMgrTest, ut_RoceRegedMemMgr_When_MultipleSameRange_Expect_AllSuccess)
{
    RoceRegedMemMgr roceRegedMemMgr{reinterpret_cast<RdmaHandle>(0x1), false};
    HcommMem mem;
    mem.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem.addr = reinterpret_cast<void*>(0x5000);
    mem.size = 2048;

    void* h1 = nullptr;
    void* h2 = nullptr;
    void* h3 = nullptr;
    EXPECT_EQ(roceRegedMemMgr.RegisterMemory(&mem, "t1", &h1), HCCL_SUCCESS);
    auto* parentBuf = static_cast<Hccl::LocalRdmaRmaBuffer*>(h1);
    hccl::BufferKey<uintptr_t, u64> parentKey(parentBuf->GetAddr(), static_cast<uint64_t>(parentBuf->GetSize()));
    EXPECT_EQ(GetRef(*roceRegedMemMgr.localRdmaRmaBufferMgr_, parentKey), 1u);

    EXPECT_EQ(roceRegedMemMgr.RegisterMemory(&mem, "t2", &h2), HCCL_SUCCESS);
    EXPECT_NE(h2, h1);
    EXPECT_EQ(GetRef(*roceRegedMemMgr.localRdmaRmaBufferMgr_, parentKey), 2u);

    EXPECT_EQ(roceRegedMemMgr.RegisterMemory(&mem, "t3", &h3), HCCL_SUCCESS);
    EXPECT_NE(h3, h1);
    EXPECT_EQ(GetRef(*roceRegedMemMgr.localRdmaRmaBufferMgr_, parentKey), 3u);

    EXPECT_EQ(roceRegedMemMgr.UnregisterMemory(h3), HCCL_SUCCESS);
    EXPECT_EQ(GetRef(*roceRegedMemMgr.localRdmaRmaBufferMgr_, parentKey), 2u);
    EXPECT_EQ(roceRegedMemMgr.UnregisterMemory(h2), HCCL_SUCCESS);
    EXPECT_EQ(GetRef(*roceRegedMemMgr.localRdmaRmaBufferMgr_, parentKey), 1u);
    EXPECT_EQ(roceRegedMemMgr.UnregisterMemory(h1), HCCL_SUCCESS);
    EXPECT_EQ(GetRef(*roceRegedMemMgr.localRdmaRmaBufferMgr_, parentKey), 0u);
}

TEST_F(RoceRegedMemMgrTest, Ut_MemoryExport_When_MemHandleUnregistered_Expect_NotFound)
{
    RoceRegedMemMgr roceRegedMemMgr{nullptr};
    EndpointDesc endpointDesc{};

    auto buf = std::make_shared<Hccl::Buffer>(0xD900, 0x100, HCCL_MEM_TYPE_DEVICE, "roce_export");
    auto localRdmaRmaBuffer = std::make_shared<Hccl::LocalRdmaRmaBuffer>(
        buf, reinterpret_cast<RdmaHandle>(0x1), 11U, 22U, reinterpret_cast<MrHandle>(0x2));
    void* memHandle = localRdmaRmaBuffer.get();

    roceRegedMemMgr.allRegisteredBuffers_.emplace_back(localRdmaRmaBuffer, false);
    roceRegedMemMgr.allRegisteredBuffers_.clear();

    void* memDesc = nullptr;
    uint32_t memDescLen = 0;
    EXPECT_EQ(roceRegedMemMgr.MemoryExport(endpointDesc, memHandle, &memDesc, &memDescLen), HCCL_E_NOT_FOUND);
    EXPECT_EQ(memDesc, nullptr);
    EXPECT_EQ(memDescLen, 0U);
}

TEST_F(RoceRegedMemMgrTest, Ut_GetParamsFromMemDesc_When_DescLenEqualSize_Expect_Return_Success)
{
    std::shared_ptr<RoceRegedMemMgr> roceRegedMemMgrPtr = std::make_shared<RoceRegedMemMgr>(nullptr);
    EndpointDesc endpointDesc;
    Hccl::ExchangeRdmaBufferDto dto;

    char buffer[sizeof(EndpointDesc)];
    uint32_t descLen = sizeof(EndpointDesc);

    MOCKER_CPP_VIRTUAL(dto, &Hccl::ExchangeRdmaBufferDto::Deserialize).stubs();

    HcclResult ret = roceRegedMemMgrPtr->GetParamsFromMemDesc(buffer, descLen, endpointDesc, dto);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

// 父+子集注册 → 先解注册父（本 handle 直接删账）→ 再解注册子
TEST_F(RoceRegedMemMgrTest, ut_RoceRegedMemMgr_When_UnregisterParentFirst_Expect_ParentRemoved)
{
    void* allocBase = reinterpret_cast<void*>(0x7000);
    size_t allocSize = 4096;
    MOCKER_CPP(aclrtMemGetAddressRange)
        .stubs()
        .with(mockcpp::any(), outBoundP(&allocBase, sizeof(allocBase)), outBoundP(&allocSize, sizeof(allocSize)))
        .will(returnValue(ACL_SUCCESS));
    StubRaMrApis();

    RoceRegedMemMgr roceRegedMemMgr{reinterpret_cast<RdmaHandle>(0x1), true};
    HcommMem mem0;
    mem0.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem0.addr = reinterpret_cast<void*>(0x7000);
    mem0.size = 4096;
    void* parentHandle = nullptr;
    ASSERT_EQ(roceRegedMemMgr.RegisterMemory(&mem0, "parent", &parentHandle), HCCL_SUCCESS);
    auto* parentBuf = static_cast<Hccl::LocalRdmaRmaBuffer*>(parentHandle);
    ASSERT_NE(parentBuf, nullptr);
    // userKey == allocKey 且首次创建 MR：返回本体而非 alias
    EXPECT_FALSE(parentBuf->IsAlias());
    RoceRegedMemMgr::MemKey allocKey(reinterpret_cast<uintptr_t>(allocBase), static_cast<uint64_t>(allocSize));
    EXPECT_EQ(GetAllocRef(roceRegedMemMgr, allocKey), 1u);
    EXPECT_EQ(roceRegedMemMgr.handlesRecords_.size(), 1u);

    HcommMem mem1;
    mem1.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem1.addr = reinterpret_cast<void*>(0x7000);
    mem1.size = 512;
    void* childHandle = nullptr;
    ASSERT_EQ(roceRegedMemMgr.RegisterMemory(&mem1, "child", &childHandle), HCCL_SUCCESS);
    EXPECT_NE(childHandle, parentHandle);
    EXPECT_EQ(GetAllocRef(roceRegedMemMgr, allocKey), 2u);
    EXPECT_EQ(roceRegedMemMgr.handlesRecords_.size(), 2u);
    EXPECT_EQ(roceRegedMemMgr.allRegisteredBuffers_.size(), 2u);

    // 先解注册父：本 handle 从账本移除，alloc ref 2→1，硬件 MR 仍在
    EXPECT_EQ(roceRegedMemMgr.UnregisterMemory(parentHandle), HCCL_SUCCESS);
    EXPECT_EQ(GetAllocRef(roceRegedMemMgr, allocKey), 1u);
    EXPECT_EQ(roceRegedMemMgr.handlesRecords_.size(), 1u);
    EXPECT_EQ(roceRegedMemMgr.allRegisteredBuffers_.size(), 1u);
    EXPECT_EQ(roceRegedMemMgr.handleToAllocKey_.count(parentBuf), 0u);
    EXPECT_EQ(roceRegedMemMgr.handleToAllocKey_.count(static_cast<Hccl::LocalRdmaRmaBuffer*>(childHandle)), 1u);

    auto itChild = std::find_if(
        roceRegedMemMgr.allRegisteredBuffers_.begin(), roceRegedMemMgr.allRegisteredBuffers_.end(),
        [childHandle](const auto& e) {
            return e.first.get() == childHandle;
        });
    ASSERT_NE(itChild, roceRegedMemMgr.allRegisteredBuffers_.end());

    // 再解注册子：alloc ref 1→0，硬件 MR 释放
    EXPECT_EQ(roceRegedMemMgr.UnregisterMemory(childHandle), HCCL_SUCCESS);
    EXPECT_EQ(GetAllocRef(roceRegedMemMgr, allocKey), 0u);
    EXPECT_EQ(roceRegedMemMgr.handlesRecords_.size(), 0u);
    EXPECT_TRUE(roceRegedMemMgr.allRegisteredBuffers_.empty());
    EXPECT_TRUE(roceRegedMemMgr.handleToAllocKey_.empty());
    EXPECT_TRUE(roceRegedMemMgr.allocToMrMap_.empty());
}

// GetAllMemHandles: 空记录 → 注册 → 多次注册 → 解注册 → 验证句柄数
TEST_F(RoceRegedMemMgrTest, ut_RoceRegedMemMgr_When_GetAllMemHandles_Expect_CorrectCount)
{
    RoceRegedMemMgr roceRegedMemMgr{reinterpret_cast<RdmaHandle>(0x1), false};

    void* handles = nullptr;
    uint32_t count = 99U;
    EXPECT_EQ(roceRegedMemMgr.GetAllMemHandles(&handles, &count), HCCL_SUCCESS);
    EXPECT_EQ(count, 0U);
    EXPECT_EQ(handles, nullptr);

    HcommMem mem;
    mem.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem.addr = reinterpret_cast<void*>(0x8000);
    mem.size = 4096;
    void* h1 = nullptr;
    ASSERT_EQ(roceRegedMemMgr.RegisterMemory(&mem, "t1", &h1), HCCL_SUCCESS);

    EXPECT_EQ(roceRegedMemMgr.GetAllMemHandles(&handles, &count), HCCL_SUCCESS);
    EXPECT_EQ(count, 1U);
    EXPECT_NE(handles, nullptr);

    void* h2 = nullptr;
    ASSERT_EQ(roceRegedMemMgr.RegisterMemory(&mem, "t2", &h2), HCCL_SUCCESS);

    EXPECT_EQ(roceRegedMemMgr.GetAllMemHandles(&handles, &count), HCCL_SUCCESS);
    EXPECT_EQ(count, 2U);

    EXPECT_EQ(roceRegedMemMgr.UnregisterMemory(h1), HCCL_SUCCESS);
    EXPECT_EQ(roceRegedMemMgr.GetAllMemHandles(&handles, &count), HCCL_SUCCESS);
    EXPECT_EQ(count, 1U);

    EXPECT_EQ(roceRegedMemMgr.UnregisterMemory(h2), HCCL_SUCCESS);
    EXPECT_EQ(roceRegedMemMgr.GetAllMemHandles(&handles, &count), HCCL_SUCCESS);
    EXPECT_EQ(count, 0U);
}

// 同一次 Device malloc 的两个互不包含窗口：useAllocMemBase 映射到同一 alloc，只硬件注册一次
TEST_F(RoceRegedMemMgrTest, ut_RoceRegedMemMgr_When_SameAllocTwoWindows_Expect_AliasReuseMr)
{
    void* allocBase = reinterpret_cast<void*>(0x10000);
    size_t allocSize = 0x2000;
    MOCKER_CPP(aclrtMemGetAddressRange)
        .stubs()
        .with(mockcpp::any(), outBoundP(&allocBase, sizeof(allocBase)), outBoundP(&allocSize, sizeof(allocSize)))
        .will(returnValue(ACL_SUCCESS));

    // UT 链接 llt stub 的 LocalRdmaRmaBuffer 不调用 RaRegisterMr，expects(once()) 拦不到；
    // 去重有效性用共享 lkey/mrHandle + allocRef 断言。
    StubRaMrApis();

    RoceRegedMemMgr roceRegedMemMgr{reinterpret_cast<RdmaHandle>(0x1), true};

    HcommMem memA{};
    memA.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    memA.addr = reinterpret_cast<void*>(0x10000);
    memA.size = 0x800;
    void* handleA = nullptr;
    ASSERT_EQ(roceRegedMemMgr.RegisterMemory(&memA, "winA", &handleA), HCCL_SUCCESS);
    auto* bufA = static_cast<Hccl::LocalRdmaRmaBuffer*>(handleA);
    ASSERT_NE(bufA, nullptr);
    EXPECT_TRUE(bufA->IsAlias());
    EXPECT_EQ(bufA->GetAddr(), reinterpret_cast<uintptr_t>(memA.addr));
    EXPECT_EQ(bufA->GetSize(), memA.size);

    RoceRegedMemMgr::MemKey baseKey(reinterpret_cast<uintptr_t>(allocBase), static_cast<uint64_t>(allocSize));
    EXPECT_EQ(GetAllocRef(roceRegedMemMgr, baseKey), 1u);

    HcommMem memB{};
    memB.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    memB.addr = reinterpret_cast<void*>(0x10800);
    memB.size = 0x800;
    void* handleB = nullptr;
    ASSERT_EQ(roceRegedMemMgr.RegisterMemory(&memB, "winB", &handleB), HCCL_SUCCESS);
    auto* bufB = static_cast<Hccl::LocalRdmaRmaBuffer*>(handleB);
    ASSERT_NE(bufB, nullptr);
    EXPECT_TRUE(bufB->IsAlias());
    EXPECT_NE(handleB, handleA);
    EXPECT_EQ(bufB->GetLkey(), bufA->GetLkey());
    EXPECT_EQ(bufB->GetMrHandle(), bufA->GetMrHandle());
    EXPECT_EQ(GetAllocRef(roceRegedMemMgr, baseKey), 2u);

    EXPECT_EQ(roceRegedMemMgr.UnregisterMemory(handleA), HCCL_SUCCESS);
    EXPECT_EQ(GetAllocRef(roceRegedMemMgr, baseKey), 1u);
    EXPECT_EQ(roceRegedMemMgr.UnregisterMemory(handleB), HCCL_SUCCESS);
    EXPECT_EQ(GetAllocRef(roceRegedMemMgr, baseKey), 0u);
    EXPECT_TRUE(roceRegedMemMgr.allocToMrMap_.empty());
}

// MLX(lbMax==0)：即使 AddressRange 能返回更大 base，也不扩展，按用户窗分别注册
TEST_F(RoceRegedMemMgrTest, ut_RoceRegedMemMgr_When_MlxLbMaxZero_Expect_NoAddressRangeExpand)
{
    // 若误走 AddressRange，会返回整段 alloc；本用例期望不调用或即使调用也不采用。
    void* allocBase = reinterpret_cast<void*>(0x20000);
    size_t allocSize = 0x4000;
    MOCKER_CPP(aclrtMemGetAddressRange)
        .stubs()
        .with(mockcpp::any(), outBoundP(&allocBase, sizeof(allocBase)), outBoundP(&allocSize, sizeof(allocSize)))
        .will(returnValue(ACL_SUCCESS));

    RoceRegedMemMgr roceRegedMemMgr{reinterpret_cast<RdmaHandle>(0x1), false};
    HcommMem mem{};
    mem.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem.addr = reinterpret_cast<void*>(0x20000);
    mem.size = 0x800;
    void* handle = nullptr;
    ASSERT_EQ(roceRegedMemMgr.RegisterMemory(&mem, "mlx", &handle), HCCL_SUCCESS);
    auto* buf = static_cast<Hccl::LocalRdmaRmaBuffer*>(handle);
    ASSERT_NE(buf, nullptr);
    EXPECT_FALSE(buf->IsAlias());
    EXPECT_EQ(buf->GetAddr(), reinterpret_cast<uintptr_t>(mem.addr));
    EXPECT_EQ(buf->GetSize(), mem.size);

    hccl::BufferKey<uintptr_t, u64> userKey(reinterpret_cast<uintptr_t>(mem.addr), mem.size);
    EXPECT_EQ(GetRef(*roceRegedMemMgr.localRdmaRmaBufferMgr_, userKey), 1u);
    EXPECT_TRUE(roceRegedMemMgr.allocToMrMap_.empty());
    EXPECT_EQ(roceRegedMemMgr.UnregisterMemory(handle), HCCL_SUCCESS);
}

// HOST + useAllocMemBase=true：type 非 DEVICE，不走 MemAlloc，回退 legacy
TEST_F(RoceRegedMemMgrTest, ut_RoceRegedMemMgr_When_HostMemWithUseAllocMemBase_Expect_LegacyPath)
{
    void* allocBase = reinterpret_cast<void*>(0x30000);
    size_t allocSize = 0x4000;
    // 即使 AddressRange 可用，HOST 也不应进入 MemAlloc 路径
    MOCKER_CPP(aclrtMemGetAddressRange)
        .stubs()
        .with(mockcpp::any(), outBoundP(&allocBase, sizeof(allocBase)), outBoundP(&allocSize, sizeof(allocSize)))
        .will(returnValue(ACL_SUCCESS));

    RoceRegedMemMgr roceRegedMemMgr{reinterpret_cast<RdmaHandle>(0x1), true};
    HcommMem mem{};
    mem.type = CommMemType::COMM_MEM_TYPE_HOST;
    mem.addr = reinterpret_cast<void*>(0x30000);
    mem.size = 0x800;
    void* handle = nullptr;
    ASSERT_EQ(roceRegedMemMgr.RegisterMemory(&mem, "host", &handle), HCCL_SUCCESS);
    auto* buf = static_cast<Hccl::LocalRdmaRmaBuffer*>(handle);
    ASSERT_NE(buf, nullptr);
    EXPECT_FALSE(buf->IsAlias());

    hccl::BufferKey<uintptr_t, u64> userKey(reinterpret_cast<uintptr_t>(mem.addr), mem.size);
    EXPECT_EQ(GetRef(*roceRegedMemMgr.localRdmaRmaBufferMgr_, userKey), 1u);
    EXPECT_TRUE(roceRegedMemMgr.allocToMrMap_.empty());
    EXPECT_TRUE(roceRegedMemMgr.handleToAllocKey_.empty());
    EXPECT_EQ(roceRegedMemMgr.UnregisterMemory(handle), HCCL_SUCCESS);
}

// AddressRange 失败时直接报错（useAllocMemBase 不允许回退用户 VA）
TEST_F(RoceRegedMemMgrTest, ut_RoceRegedMemMgr_When_AddressRangeFail_Expect_ReturnError)
{
    MOCKER_CPP(aclrtMemGetAddressRange).stubs().will(returnValue(ACL_ERROR_INVALID_PARAM));

    RoceRegedMemMgr roceRegedMemMgr{reinterpret_cast<RdmaHandle>(0x1), true};
    HcommMem mem{};
    mem.type = CommMemType::COMM_MEM_TYPE_DEVICE;
    mem.addr = reinterpret_cast<void*>(0x9000);
    mem.size = 2048;
    void* handle = nullptr;
    EXPECT_EQ(roceRegedMemMgr.RegisterMemory(&mem, "fail", &handle), HCCL_E_MEMORY);
    EXPECT_EQ(handle, nullptr);
    EXPECT_TRUE(roceRegedMemMgr.allocToMrMap_.empty());
}

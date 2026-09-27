/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 被测目标: HcommMemReg / HcommMemUnreg / HcommMemExport / HcommMemImport / HcommMemUnimport C API
//           —— 经 GetEndpointMgr().Get(handle)->GetRegedMemMgr()->xxx() 全局句柄表路径。
// 适配重构：内存方法从 Endpoint 移至 RegedMemMgr；EndpointMgr 收编 HcommEndpointMap；
//          RoceRegedMemMgr rdmaHandle_ 经构造注入。

#include <cstring>
#include <string>
#include "../../ut_hcomm_base.h"
#include "endpoint.h"
#include "../../../../../../src/base_comm/resources/endpoints/mgr/endpoint_mgr.h"
#include "hcomm_c_adpt_common.h"
#include "../../../../../../src/base_comm/hcomm_res_mgr.h"
#include "roce_reged_mem_mgr.h"
#include "hccp.h"
#include "hcomm_c_adpt.h"

namespace {

// FakeEndpoint: 持有外部传入的 RegedMemMgr, GetRegedMemMgr() 返回它。
// 用于跨 EP 共享 / 不共享 RegedMemMgr 的 C API 层测试。
// 内存方法已从 Endpoint 移至 RegedMemMgr，故不再在 Endpoint 上 override 内存方法。
class FakeEndpoint : public hcomm::Endpoint {
public:
    FakeEndpoint(const EndpointDesc& desc, std::shared_ptr<hcomm::RegedMemMgr> mgr)
        : hcomm::Endpoint(desc),
          mgr_(std::move(mgr))
    {}

    HcclResult Init() override { return HCCL_SUCCESS; }

    hcomm::RegedMemMgr* GetRegedMemMgr() override { return mgr_.get(); }
    void* GetRdmaHandle() override { return nullptr; }
    bool IsCtxHandleValid() const override { return false; }

private:
    std::shared_ptr<hcomm::RegedMemMgr> mgr_;
};

EndpointDesc MakeHostRoceDesc()
{
    EndpointDesc desc{};
    desc.protocol = COMM_PROTOCOL_ROCE;
    desc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    return desc;
}

EndpointHandle MakeTestHandle()
{
    static uintptr_t counter = 0x10000;
    return reinterpret_cast<EndpointHandle>(counter++);
}

void RegisterRaMrMockSuccess(MrHandle fakeMrHandle)
{
    MOCKER(RaRegisterMr)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&fakeMrHandle, sizeof(fakeMrHandle)))
        .will(returnValue(0));
    MOCKER(RaDeregisterMr).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(0));
}

std::shared_ptr<hcomm::RoceRegedMemMgr> MakeSharedMgr(RdmaHandle fakeRdmaHandle)
{
    // rdmaHandle_ 经构造注入（rdmaHandle_ 移至派生类 private，不再 public 直写）
    return std::make_shared<hcomm::RoceRegedMemMgr>(fakeRdmaHandle);
}
} // namespace

class TestHcommMem : public TestHcommCAdptBase {
public:
    void SetUp() override
    {
        TestHcommCAdptBase::SetUp();
        // 清空单例残留，保证用例隔离
        hcomm::HcommResMgr::GetInstance().GetEndpointMgr().DeInit();
    }
    void TearDown() override
    {
        for (auto h : injectedHandles_) {
            hcomm::HcommResMgr::GetInstance().GetEndpointMgr().Remove(h);
        }
        injectedHandles_.clear();
        if (validEpHandle_ != nullptr) {
            (void)HcommEndpointDestroy(validEpHandle_);
            validEpHandle_ = nullptr;
        }
        TestHcommCAdptBase::TearDown();
    }

    EndpointHandle InjectEndpoint(std::shared_ptr<hcomm::RegedMemMgr> mgr)
    {
        // handle 语义 = Endpoint 指针（Create 期 reinterpret_cast 注入），
        // C API 经 HcommResMgr::GetInstance().GetEndpointMgr().Get(handle) 全局句柄表校验在册，
        // 故注入句柄必须取 endpoint 实际地址
        auto ep = std::make_unique<FakeEndpoint>(MakeHostRoceDesc(), std::move(mgr));
        EndpointHandle h = reinterpret_cast<EndpointHandle>(ep.get());
        hcomm::HcommResMgr::GetInstance().GetEndpointMgr().Add(h, std::move(ep));
        injectedHandles_.push_back(h);
        return h;
    }

private:
    std::vector<EndpointHandle> injectedHandles_;

    // 为非 endpoint 校验类用例构造一个真实 endpoint，使其能进入新的 ops 分发路径
    void CreateValidEndpoint()
    {
        if (validEpHandle_ != nullptr)
            return;
        EndpointDesc desc{};
        desc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
        desc.protocol = COMM_PROTOCOL_ROCE;
        (void)HcommEndpointCreate(&desc, &validEpHandle_);
    }

    EndpointHandle validEpHandle_{nullptr};
};

// ============ 参数校验类用例（不依赖真实内存注册）============

// TC-HcommMemReg-002/003/004: mem / memTag / memHandle 空指针
TEST_F(TestHcommMem, Ut_TestHcommMemReg_When_InvalidHandle_Return_HCCL_E_NOT_FOUND)
{
    CommMem mem;
    mem.addr = malloc(1024);
    mem.size = 1024;
    mem.type = COMM_MEM_TYPE_HOST;
    void* memHandle = nullptr;

    EndpointHandle invalidHandle = reinterpret_cast<EndpointHandle>(0x0FFFFFFFFFFFFFFF);
    HcommResult ret = HcommMemReg(invalidHandle, "test_mem", &mem, &memHandle);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);

    free(mem.addr);
}

// TC-HcommMemReg-005: 已销毁 endpoint 上注册内存 → NOT_FOUND（无效句柄）
TEST_F(TestHcommMem, Ut_TestHcommMemReg_When_EndpointDestroyed_Return_HCCL_E_NOT_FOUND)
{
    CommMem mem;
    mem.addr = malloc(1024);
    mem.size = 1024;
    mem.type = COMM_MEM_TYPE_HOST;
    void* memHandle = nullptr;

    HcommResult ret = HcommMemReg(reinterpret_cast<EndpointHandle>(0xDEADBEEF), "t", &mem, &memHandle);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);

    free(mem.addr);
}

TEST_F(TestHcommMem, Ut_TestHcommMemReg_When_MemHandleNullptr_Return_HCCL_E_PTR)
{
    CreateValidEndpoint();
    CommMem mem;
    mem.addr = malloc(1024);
    mem.size = 1024;
    mem.type = COMM_MEM_TYPE_HOST;

    HcommResult ret = HcommMemReg(validEpHandle_, "test_mem", &mem, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);

    free(mem.addr);
}

TEST_F(TestHcommMem, Ut_TestHcommMemUnreg_When_InvalidHandle_Return_HCCL_E_NOT_FOUND)
{
    EndpointHandle invalidHandle = reinterpret_cast<EndpointHandle>(0xFFFFFFFFFFFFFFFF);
    HcommResult ret = HcommMemUnreg(invalidHandle, nullptr);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(TestHcommMem, Ut_TestHcommMemExport_When_InvalidMemHandle_Return_HCCL_E_PTR)
{
    CreateValidEndpoint();
    void* memDesc = nullptr;
    uint32_t memDescLen = 0;
    HcommResult ret = HcommMemExport(validEpHandle_, nullptr, &memDesc, &memDescLen);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommMem, Ut_TestHcommMemExport_When_OutputNullptr_Return_HCCL_E_PTR)
{
    CreateValidEndpoint();
    void* dummyHandle = this;
    HcommResult ret = HcommMemExport(validEpHandle_, dummyHandle, nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// 判空下沉：HcommMemImport C-API 不校验 desc 空指针，由 RoceRegedMemMgr::MemoryImport 入口
// CHK_PTR_NULL(memDesc) 返回 HCCL_E_PTR。
TEST_F(TestHcommMem, Ut_TestHcommMemImport_When_InvalidDesc_Return_HCCL_E_PTR)
{
    CreateValidEndpoint();
    HcommMem outMem;
    HcommResult ret = HcommMemImport(validEpHandle_, nullptr, 0, &outMem);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// 判空下沉：HcommMemUnimport 同理经 RegedMemMgr::MemoryUnimport 入口判空返回 HCCL_E_PTR
TEST_F(TestHcommMem, Ut_TestHcommMemUnimport_When_InvalidParams_Return_HCCL_E_PTR)
{
    CreateValidEndpoint();
    HcommResult ret = HcommMemUnimport(validEpHandle_, nullptr, 0);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommMem, Ut_TestHcommMemImport_When_DescLenZero_Return_HCCL_E_INTERNAL)
{
    CreateValidEndpoint();
    HcommMem outMem;
    char desc[8] = {0};
    HcommResult ret = HcommMemImport(validEpHandle_, desc, 0, &outMem);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(TestHcommMem, Ut_TestHcommMemGetAllMemHandles_When_Nullptr_Return_HCCL_E_PTR)
{
    HcommResult ret = HcommMemGetAllMemHandles(nullptr, nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

// ============ 内存注册/注销/导入导出 经 RegedMemMgr 路径 ============

// TC-HcommMemReg-001 / TC-HcommMemReg-007: 经 GetRegedMemMgr()->RegisterMemory 路径
// 用例1: 单 EP 同 buffer 多次注册, 第二次走 alias 复用
TEST_F(TestHcommMem, MemReg_When_SingleEP_SameBufferTwice_Expect_BothSuccessAndHandlesDiffer)
{
    RdmaHandle fakeRdmaHandle = reinterpret_cast<RdmaHandle>(0xABCD);
    MrHandle fakeMrHandle = reinterpret_cast<MrHandle>(0x1234);
    RegisterRaMrMockSuccess(fakeMrHandle);

    auto mgr = MakeSharedMgr(fakeRdmaHandle);
    EndpointHandle epHandle = InjectEndpoint(mgr);

    CommMem mem{};
    mem.type = COMM_MEM_TYPE_HOST;
    mem.addr = reinterpret_cast<void*>(0x1000);
    mem.size = 4096;

    HcommMemHandle h1 = nullptr;
    HcommResult ret1 = HcommMemReg(epHandle, "tag1", &mem, &h1);
    EXPECT_EQ(ret1, HCCL_SUCCESS);
    EXPECT_NE(h1, nullptr);

    HcommMemHandle h2 = nullptr;
    HcommResult ret2 = HcommMemReg(epHandle, "tag2", &mem, &h2);
    EXPECT_EQ(ret2, HCCL_SUCCESS);
    EXPECT_NE(h2, nullptr);

    EXPECT_NE(h1, h2);
}

// 用例2: 跨 EP 不共享 (各自独立 mgr), 同 buffer 各自注册, 两 EP 都成功
TEST_F(TestHcommMem, MemReg_When_CrossEP_NotShared_Expect_BothSuccess)
{
    RdmaHandle fakeRdmaHandle = reinterpret_cast<RdmaHandle>(0xABCD);
    MrHandle fakeMrHandle = reinterpret_cast<MrHandle>(0x1234);
    RegisterRaMrMockSuccess(fakeMrHandle);

    auto mgr1 = MakeSharedMgr(fakeRdmaHandle);
    auto mgr2 = MakeSharedMgr(fakeRdmaHandle);
    EndpointHandle ep1 = InjectEndpoint(mgr1);
    EndpointHandle ep2 = InjectEndpoint(mgr2);

    CommMem mem{};
    mem.type = COMM_MEM_TYPE_HOST;
    mem.addr = reinterpret_cast<void*>(0x5000);
    mem.size = 2048;

    HcommMemHandle h1 = nullptr;
    HcommResult ret1 = HcommMemReg(ep1, "t1", &mem, &h1);
    EXPECT_EQ(ret1, HCCL_SUCCESS);
    EXPECT_NE(h1, nullptr);

    HcommMemHandle h2 = nullptr;
    HcommResult ret2 = HcommMemReg(ep2, "t2", &mem, &h2);
    EXPECT_EQ(ret2, HCCL_SUCCESS);
    EXPECT_NE(h2, nullptr);

    EXPECT_NE(h1, h2);
}

// 用例3: 跨 EP 共享 (共享同一 mgr), 同 buffer 各注册, 第二次走 alias 复用
TEST_F(TestHcommMem, MemReg_When_CrossEP_Shared_Expect_BothSuccessAndHandlesDiffer)
{
    RdmaHandle fakeRdmaHandle = reinterpret_cast<RdmaHandle>(0xABCD);
    MrHandle fakeMrHandle = reinterpret_cast<MrHandle>(0x1234);
    RegisterRaMrMockSuccess(fakeMrHandle);

    auto mgr = MakeSharedMgr(fakeRdmaHandle);
    EndpointHandle ep1 = InjectEndpoint(mgr);
    EndpointHandle ep2 = InjectEndpoint(mgr);

    CommMem mem{};
    mem.type = COMM_MEM_TYPE_HOST;
    mem.addr = reinterpret_cast<void*>(0x5000);
    mem.size = 2048;

    HcommMemHandle h1 = nullptr;
    HcommResult ret1 = HcommMemReg(ep1, "t1", &mem, &h1);
    EXPECT_EQ(ret1, HCCL_SUCCESS);
    EXPECT_NE(h1, nullptr);

    HcommMemHandle h2 = nullptr;
    HcommResult ret2 = HcommMemReg(ep2, "t2", &mem, &h2);
    EXPECT_EQ(ret2, HCCL_SUCCESS);
    EXPECT_NE(h2, nullptr);

    EXPECT_NE(h1, h2);
}

// TC-HcommMemUnreg-001: 跨 EP 共享后各注销
TEST_F(TestHcommMem, MemUnreg_When_CrossEP_Shared_Expect_BothSuccess)
{
    RdmaHandle fakeRdmaHandle = reinterpret_cast<RdmaHandle>(0xABCD);
    MrHandle fakeMrHandle = reinterpret_cast<MrHandle>(0x1234);
    RegisterRaMrMockSuccess(fakeMrHandle);

    auto mgr = MakeSharedMgr(fakeRdmaHandle);
    EndpointHandle ep1 = InjectEndpoint(mgr);
    EndpointHandle ep2 = InjectEndpoint(mgr);

    CommMem mem{};
    mem.type = COMM_MEM_TYPE_HOST;
    mem.addr = reinterpret_cast<void*>(0x5000);
    mem.size = 2048;

    HcommMemHandle h1 = nullptr;
    ASSERT_EQ(HcommMemReg(ep1, "t1", &mem, &h1), HCCL_SUCCESS);
    HcommMemHandle h2 = nullptr;
    ASSERT_EQ(HcommMemReg(ep2, "t2", &mem, &h2), HCCL_SUCCESS);

    HcommResult ret1 = HcommMemUnreg(ep1, h1);
    EXPECT_EQ(ret1, HCCL_SUCCESS);

    HcommResult ret2 = HcommMemUnreg(ep2, h2);
    EXPECT_EQ(ret2, HCCL_SUCCESS);
}

// 用例5: 跨 EP 共享父子 buffer (子集走 alias)
TEST_F(TestHcommMem, MemReg_When_CrossEP_Shared_ParentChild_Expect_BothSuccess)
{
    RdmaHandle fakeRdmaHandle = reinterpret_cast<RdmaHandle>(0xABCD);
    MrHandle fakeMrHandle = reinterpret_cast<MrHandle>(0x1234);
    RegisterRaMrMockSuccess(fakeMrHandle);

    auto mgr = MakeSharedMgr(fakeRdmaHandle);
    EndpointHandle ep1 = InjectEndpoint(mgr);
    EndpointHandle ep2 = InjectEndpoint(mgr);

    CommMem memParent{};
    memParent.type = COMM_MEM_TYPE_HOST;
    memParent.addr = reinterpret_cast<void*>(0x1000);
    memParent.size = 4096;

    CommMem memChild{};
    memChild.type = COMM_MEM_TYPE_HOST;
    memChild.addr = reinterpret_cast<void*>(0x1000);
    memChild.size = 1024;

    HcommMemHandle hParent = nullptr;
    HcommResult retP = HcommMemReg(ep1, "parent", &memParent, &hParent);
    EXPECT_EQ(retP, HCCL_SUCCESS);
    EXPECT_NE(hParent, nullptr);

    HcommMemHandle hChild = nullptr;
    HcommResult retC = HcommMemReg(ep2, "child", &memChild, &hChild);
    EXPECT_EQ(retC, HCCL_SUCCESS);
    EXPECT_NE(hChild, nullptr);

    EXPECT_NE(hParent, hChild);
}

// ============ memTag 长度校验用例(HcommMemReg 入口) ============

// TC-HcommMemReg-006: memTag 长度等于上限(HCOMM_RES_TAG_MAX_LEN) → 注册成功
TEST_F(TestHcommMem, MemReg_When_MemTagLenEqualsMax_Expect_Success)
{
    RdmaHandle fakeRdmaHandle = reinterpret_cast<RdmaHandle>(0xABCD);
    MrHandle fakeMrHandle = reinterpret_cast<MrHandle>(0x1234);
    RegisterRaMrMockSuccess(fakeMrHandle);

    auto mgr = MakeSharedMgr(fakeRdmaHandle);
    EndpointHandle epHandle = InjectEndpoint(mgr);

    CommMem mem{};
    mem.type = COMM_MEM_TYPE_HOST;
    mem.addr = reinterpret_cast<void*>(0x1000);
    mem.size = 4096;

    std::string memTag(HCOMM_RES_TAG_MAX_LEN, 'a');
    ASSERT_EQ(memTag.size(), HCOMM_RES_TAG_MAX_LEN);
    HcommMemHandle h = nullptr;
    HcommResult ret = HcommMemReg(epHandle, memTag.c_str(), &mem, &h);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(h, nullptr);
}

// TC-HcommMemReg-006: memTag 长度超过上限 → 返回 HCCL_E_PARA, 不产生注册副作用
TEST_F(TestHcommMem, MemReg_When_MemTagLenExceedsMax_Expect_ReturnParaError)
{
    RdmaHandle fakeRdmaHandle = reinterpret_cast<RdmaHandle>(0xABCD);
    MrHandle fakeMrHandle = reinterpret_cast<MrHandle>(0x1234);
    RegisterRaMrMockSuccess(fakeMrHandle);

    auto mgr = MakeSharedMgr(fakeRdmaHandle);
    EndpointHandle epHandle = InjectEndpoint(mgr);

    CommMem mem{};
    mem.type = COMM_MEM_TYPE_HOST;
    mem.addr = reinterpret_cast<void*>(0x1000);
    mem.size = 4096;

    std::string memTag(HCOMM_RES_TAG_MAX_LEN + 1, 'a');
    HcommMemHandle h = nullptr;
    HcommResult ret = HcommMemReg(epHandle, memTag.c_str(), &mem, &h);
    EXPECT_EQ(ret, HCCL_E_PARA);
    EXPECT_EQ(h, nullptr);
}

// TC-HcommMemReg-006: memTag 未以\0结尾且前255字节内无\0 → 返回 HCCL_E_PARA(非NUL结尾安全)
TEST_F(TestHcommMem, MemReg_When_MemTagNotNullTerminated_Expect_ReturnParaError)
{
    RdmaHandle fakeRdmaHandle = reinterpret_cast<RdmaHandle>(0xABCD);
    MrHandle fakeMrHandle = reinterpret_cast<MrHandle>(0x1234);
    RegisterRaMrMockSuccess(fakeMrHandle);

    auto mgr = MakeSharedMgr(fakeRdmaHandle);
    EndpointHandle epHandle = InjectEndpoint(mgr);

    CommMem mem{};
    mem.type = COMM_MEM_TYPE_HOST;
    mem.addr = reinterpret_cast<void*>(0x1000);
    mem.size = 4096;

    char memTag[HCOMM_RES_TAG_MAX_LEN + 1];
    memset(memTag, 'a', sizeof(memTag)); // 256字节全非零,无\0结尾
    HcommMemHandle h = nullptr;
    HcommResult ret = HcommMemReg(epHandle, memTag, &mem, &h);
    EXPECT_EQ(ret, HCCL_E_PARA);
    EXPECT_EQ(h, nullptr);
}

// TC-HcommMemReg-006: memTag为nullptr → 跳过长度校验, 注册成功
TEST_F(TestHcommMem, MemReg_When_MemTagNullptr_Expect_Success)
{
    RdmaHandle fakeRdmaHandle = reinterpret_cast<RdmaHandle>(0xABCD);
    MrHandle fakeMrHandle = reinterpret_cast<MrHandle>(0x1234);
    RegisterRaMrMockSuccess(fakeMrHandle);

    auto mgr = MakeSharedMgr(fakeRdmaHandle);
    EndpointHandle epHandle = InjectEndpoint(mgr);

    CommMem mem{};
    mem.type = COMM_MEM_TYPE_HOST;
    mem.addr = reinterpret_cast<void*>(0x1000);
    mem.size = 4096;

    HcommMemHandle h = nullptr;
    HcommResult ret = HcommMemReg(epHandle, nullptr, &mem, &h);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(h, nullptr);
}

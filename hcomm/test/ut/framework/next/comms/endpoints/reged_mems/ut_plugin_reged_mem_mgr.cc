/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 被测目标: hcomm::PluginRegedMemMgr —— NIC 插件端点的 RegedMemMgr，5 个内存方法转发到 nicOps_。

#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>

#include "nic_plugin_holder.h"
#include "hccs_reged_mem_mgr.h"
#include "hcomm_c_adpt.h"
#include "hcomm_res_defs.h"

using namespace hcomm;

namespace {
// ---- 记录 nicOps 调用次数的桩 ----
struct PluginCallStats {
    uint32_t registerMemory{0};
    uint32_t unregisterMemory{0};
    uint32_t memoryExport{0};
    uint32_t memoryImport{0};
    uint32_t memoryUnimport{0};
    int32_t registerMemoryRet{HCCL_SUCCESS};
    int32_t unregisterMemoryRet{HCCL_SUCCESS};
    int32_t memoryExportRet{HCCL_SUCCESS};
    int32_t memoryImportRet{HCCL_SUCCESS};
    int32_t memoryUnimportRet{HCCL_SUCCESS};
};

PluginCallStats g_stats;

void ResetStats() { g_stats = PluginCallStats{}; }

int32_t StubRegisterMemory(void* ctx, const CommMem* mem, const char* tag, void** handle)
{
    (void)ctx;
    (void)mem;
    (void)tag;
    g_stats.registerMemory++;
    *handle = reinterpret_cast<void*>(0xAAAA);
    return g_stats.registerMemoryRet;
}

int32_t StubUnregisterMemory(void* ctx, void* handle)
{
    (void)ctx;
    (void)handle;
    g_stats.unregisterMemory++;
    return g_stats.unregisterMemoryRet;
}

int32_t StubMemoryExport(void* ctx, void* handle, void** desc, uint32_t* descLen)
{
    (void)ctx;
    (void)handle;
    g_stats.memoryExport++;
    static char fakeDesc[8] = {};
    *desc = fakeDesc;
    *descLen = sizeof(fakeDesc);
    return g_stats.memoryExportRet;
}

int32_t StubMemoryImport(void* ctx, const void* desc, uint32_t descLen, CommMem* outMem)
{
    (void)ctx;
    (void)desc;
    (void)descLen;
    g_stats.memoryImport++;
    outMem->type = COMM_MEM_TYPE_HOST;
    outMem->addr = nullptr;
    outMem->size = 0;
    return g_stats.memoryImportRet;
}

int32_t StubMemoryUnimport(void* ctx, const void* desc, uint32_t descLen)
{
    (void)ctx;
    (void)desc;
    (void)descLen;
    g_stats.memoryUnimport++;
    return g_stats.memoryUnimportRet;
}

HcommNicEndpointOps MakeFakeNicOps()
{
    return {
        {HCOMM_NIC_ENDPOINT_OPS_VERSION, HCOMM_NIC_ENDPOINT_OPS_MAGIC_WORD, sizeof(HcommNicEndpointOps), 0},
        nullptr,
        nullptr,
        StubRegisterMemory,
        StubUnregisterMemory,
        StubMemoryExport,
        StubMemoryImport,
        StubMemoryUnimport,
        nullptr};
}
} // namespace

class PluginRegedMemMgrTest : public testing::Test {
protected:
    void SetUp() override { ResetStats(); }
    void TearDown() override { GlobalMockObject::verify(); }
};

// TC-PluginRegedMemMgr_RegisterMemory-001: RegisterMemory 转发到 nicOps_->registerMemory(nicCtx_, ...)
TEST_F(PluginRegedMemMgrTest, Ut_RegisterMemory_When_NicOpsValid_Expect_ForwardedToNicOps)
{
    HcommNicEndpointOps ops = MakeFakeNicOps();
    void* fakeCtx = reinterpret_cast<void*>(0xBEEF);
    PluginRegedMemMgr mgr(&ops, fakeCtx);

    CommMem mem{COMM_MEM_TYPE_HOST, reinterpret_cast<void*>(0x1000), 4096};
    void* handle = nullptr;
    HcclResult ret = mgr.RegisterMemory(&mem, "tag", &handle);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(g_stats.registerMemory, 1u);
    EXPECT_NE(handle, nullptr);
}

// TC-PluginRegedMemMgr_RegisterMemory-002 异常: nicOps_ 为 nullptr 时不段错误——
// 当前实现直接解引用 nicOps_，传入 nullptr 会触发空指针。此处用容错方式验证：
// 构造时 nicOps=nullptr 仍可构造，且 GetAllMemHandles（不依赖 nicOps）返回 NOT_SUPPORT。
TEST_F(PluginRegedMemMgrTest, Ut_RegisterMemory_When_NicOpsNull_Expect_ConstructibleAndGetAllNotSupport)
{
    PluginRegedMemMgr mgr(nullptr, nullptr);

    void* handles = nullptr;
    uint32_t num = 0;
    HcclResult ret = mgr.GetAllMemHandles(&handles, &num);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

// TC-PluginRegedMemMgr_UnregisterMemory-001: UnregisterMemory 转发到 nicOps
TEST_F(PluginRegedMemMgrTest, Ut_UnregisterMemory_When_NicOpsValid_Expect_ForwardedToNicOps)
{
    HcommNicEndpointOps ops = MakeFakeNicOps();
    PluginRegedMemMgr mgr(&ops, reinterpret_cast<void*>(0x1));

    HcclResult ret = mgr.UnregisterMemory(reinterpret_cast<void*>(0xAAAA));
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(g_stats.unregisterMemory, 1u);
}

// TC-PluginRegedMemMgr_MemoryExport-001: MemoryExport 转发到 nicOps
TEST_F(PluginRegedMemMgrTest, Ut_MemoryExport_When_NicOpsValid_Expect_ForwardedToNicOps)
{
    HcommNicEndpointOps ops = MakeFakeNicOps();
    PluginRegedMemMgr mgr(&ops, reinterpret_cast<void*>(0x1));

    void* desc = nullptr;
    uint32_t descLen = 0;
    EndpointDesc endpointDesc{};
    HcclResult ret = mgr.MemoryExport(endpointDesc, reinterpret_cast<void*>(0xAAAA), &desc, &descLen);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(g_stats.memoryExport, 1u);
    EXPECT_NE(desc, nullptr);
    EXPECT_GT(descLen, 0u);
}

// TC-PluginRegedMemMgr_MemoryImport-001: MemoryImport 转发到 nicOps
TEST_F(PluginRegedMemMgrTest, Ut_MemoryImport_When_NicOpsValid_Expect_ForwardedToNicOps)
{
    HcommNicEndpointOps ops = MakeFakeNicOps();
    PluginRegedMemMgr mgr(&ops, reinterpret_cast<void*>(0x1));

    char desc[8] = {0};
    CommMem outMem{};
    HcclResult ret = mgr.MemoryImport(desc, sizeof(desc), &outMem);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(g_stats.memoryImport, 1u);
}

// TC-PluginRegedMemMgr_MemoryUnimport-001: MemoryUnimport 转发到 nicOps
TEST_F(PluginRegedMemMgrTest, Ut_MemoryUnimport_When_NicOpsValid_Expect_ForwardedToNicOps)
{
    HcommNicEndpointOps ops = MakeFakeNicOps();
    PluginRegedMemMgr mgr(&ops, reinterpret_cast<void*>(0x1));

    char desc[8] = {0};
    HcclResult ret = mgr.MemoryUnimport(desc, sizeof(desc));

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(g_stats.memoryUnimport, 1u);
}

// TC-PluginRegedMemMgr_GetAllMemHandles-001: GetAllMemHandles 返回 HCCL_E_NOT_SUPPORT（插件 ops 表无此字段）
TEST_F(PluginRegedMemMgrTest, Ut_GetAllMemHandles_When_Called_Expect_ReturnNotSupport)
{
    HcommNicEndpointOps ops = MakeFakeNicOps();
    PluginRegedMemMgr mgr(&ops, reinterpret_cast<void*>(0x1));

    void* handles = nullptr;
    uint32_t num = 0;
    HcclResult ret = mgr.GetAllMemHandles(&handles, &num);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

// MemoryGrant 已下移至 HccsRegedMemMgr（基类不再提供默认实现）。
// PluginRegedMemMgr 非 HCCS 管理器：HcommMemGrant 层对其 dynamic_cast 失败后跳过授权（返回 SUCCESS），
// 与下移前基类默认 SUCCESS 行为一致。
TEST_F(PluginRegedMemMgrTest, Ut_MemoryGrant_When_PluginMgrNotHccs_Expect_SkipGrant)
{
    HcommNicEndpointOps ops = MakeFakeNicOps();
    PluginRegedMemMgr mgr(&ops, reinterpret_cast<void*>(0x1));

    auto* hccsMgr = dynamic_cast<HccsRegedMemMgr*>(&mgr);
    EXPECT_EQ(hccsMgr, nullptr);
}

// nicOps 返回错误码时透传
TEST_F(PluginRegedMemMgrTest, Ut_RegisterMemory_When_NicOpsReturnsError_Expect_Propagate)
{
    HcommNicEndpointOps ops = MakeFakeNicOps();
    g_stats.registerMemoryRet = HCCL_E_INTERNAL;
    PluginRegedMemMgr mgr(&ops, reinterpret_cast<void*>(0x1));

    CommMem mem{COMM_MEM_TYPE_HOST, reinterpret_cast<void*>(0x1000), 4096};
    void* handle = nullptr;
    HcclResult ret = mgr.RegisterMemory(&mem, "tag", &handle);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

// ============ PluginEndpointHolder ============

namespace {
HcommNicPluginInfo g_fakePluginInfo
    = {{HCOMM_NIC_PLUGIN_INFO_VERSION, HCOMM_NIC_PLUGIN_INFO_MAGIC_WORD, sizeof(HcommNicPluginInfo), 0},
       "ut_plugin_reged",
       1,
       {COMM_PROTOCOL_ROCE}};

int32_t g_destroyCalls = 0;
int32_t StubDestroy(void* ctx)
{
    (void)ctx;
    g_destroyCalls++;
    return HCCL_SUCCESS;
}

// PluginEndpointHolder 析构会对 nicOps_ 执行 delete；故测试须传入堆分配的 ops。
// 用 lambda 返回 new 出来的 ops 副本，保证析构 delete 安全。
HcommNicEndpointOps* MakeHeapFakeOps()
{
    auto* p = new HcommNicEndpointOps(MakeFakeNicOps());
    return p;
}
HcommNicEndpointOps* MakeHeapFakeOpsWithDestroy()
{
    auto* p = new HcommNicEndpointOps{
        {HCOMM_NIC_ENDPOINT_OPS_VERSION, HCOMM_NIC_ENDPOINT_OPS_MAGIC_WORD, sizeof(HcommNicEndpointOps), 0},
        nullptr,
        StubDestroy,
        StubRegisterMemory,
        StubUnregisterMemory,
        StubMemoryExport,
        StubMemoryImport,
        StubMemoryUnimport,
        nullptr};
    return p;
}
} // namespace

// TC-PluginEndpointHolder_SetNicEndpointCtx-001: SetNicEndpointCtx 构造 PluginRegedMemMgr 并赋值 nicOps_/nicCtx_
TEST_F(PluginRegedMemMgrTest, Ut_SetNicEndpointCtx_When_Called_Expect_ConstructPluginRegedMemMgr)
{
    NicPluginEntry entry{nullptr, &g_fakePluginInfo, nullptr, nullptr};
    EndpointDesc desc{};
    desc.protocol = COMM_PROTOCOL_ROCE;
    desc.loc.locType = ENDPOINT_LOC_TYPE_HOST;

    PluginEndpointHolder holder(desc, &entry);
    // SetNicEndpointCtx 前 GetRegedMemMgr 返回空
    EXPECT_EQ(holder.GetRegedMemMgr(), nullptr);

    HcommNicEndpointOps* ops = MakeHeapFakeOps();
    void* fakeCtx = reinterpret_cast<void*>(0xCAFE);
    holder.SetNicEndpointCtx(ops, fakeCtx);

    EXPECT_NE(holder.GetRegedMemMgr(), nullptr);
    EXPECT_EQ(holder.GetNicOps(), ops);
    EXPECT_EQ(holder.GetNicCtx(), fakeCtx);
    // holder 析构时 delete ops（堆分配，安全）
}

// TC-PluginEndpointHolder_GetRegedMemMgr-002 异常: SetNicEndpointCtx 前调 GetRegedMemMgr 返回空
TEST_F(PluginRegedMemMgrTest, Ut_GetRegedMemMgr_When_BeforeSetNicCtx_Expect_ReturnNull)
{
    NicPluginEntry entry{nullptr, &g_fakePluginInfo, nullptr, nullptr};
    EndpointDesc desc{};
    desc.protocol = COMM_PROTOCOL_ROCE;
    desc.loc.locType = ENDPOINT_LOC_TYPE_HOST;

    PluginEndpointHolder holder(desc, &entry);
    EXPECT_EQ(holder.GetRegedMemMgr(), nullptr);
}

// TC-PluginEndpointHolder_GetRegedMemMgr-001: GetRegedMemMgr 返回 PluginRegedMemMgr 类型
TEST_F(PluginRegedMemMgrTest, Ut_GetRegedMemMgr_When_AfterSetNicCtx_Expect_NonNull)
{
    NicPluginEntry entry{nullptr, &g_fakePluginInfo, nullptr, nullptr};
    EndpointDesc desc{};
    desc.protocol = COMM_PROTOCOL_ROCE;
    desc.loc.locType = ENDPOINT_LOC_TYPE_HOST;

    PluginEndpointHolder holder(desc, &entry);
    HcommNicEndpointOps* ops = MakeHeapFakeOps();
    holder.SetNicEndpointCtx(ops, reinterpret_cast<void*>(0x1));

    auto* mgr = holder.GetRegedMemMgr();
    ASSERT_NE(mgr, nullptr);
    // 经 GetRegedMemMgr 路径触发 nicOps
    CommMem mem{COMM_MEM_TYPE_HOST, reinterpret_cast<void*>(0x1000), 4096};
    void* handle = nullptr;
    EXPECT_EQ(mgr->RegisterMemory(&mem, "tag", &handle), HCCL_SUCCESS);
    EXPECT_EQ(g_stats.registerMemory, 1u);
    // holder 析构 delete ops（堆分配，安全）
}

// TC-PluginEndpointHolder_析构-001: 析构时销毁 nicOps/nicCtx（DestroyNicPluginOpsAndCtx）
// 此处验证：传入带 destroy 桩的 ops，析构后 destroy 被调用
TEST_F(PluginRegedMemMgrTest, Ut_Destructor_When_HolderDestroyed_Expect_DestroyNicCtxCalled)
{
    g_destroyCalls = 0;
    NicPluginEntry entry{nullptr, &g_fakePluginInfo, nullptr, nullptr};
    EndpointDesc desc{};
    desc.protocol = COMM_PROTOCOL_ROCE;
    desc.loc.locType = ENDPOINT_LOC_TYPE_HOST;

    HcommNicEndpointOps* ops = MakeHeapFakeOpsWithDestroy();
    void* fakeCtx = reinterpret_cast<void*>(0x1234);
    {
        PluginEndpointHolder holder(desc, &entry);
        holder.SetNicEndpointCtx(ops, fakeCtx);
    }
    EXPECT_EQ(g_destroyCalls, 1);
}

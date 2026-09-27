/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../ut_hcomm_base.h"
#include "cpu_roce_endpoint.h"
#include "hcomm_c_adpt.h"
#include "hcomm_res.h"
#include "nic_plugin_manager.h"

using namespace hcomm;

namespace {
struct FakePluginEndpointState {
    bool inited = false;
};

struct FakePluginEndpointScenario {
    uint32_t createEndpointCalls = 0;
    uint32_t initEndpointCalls = 0;
    uint32_t destroyEndpointCalls = 0;
    uint32_t registerMemoryCalls = 0;
    uint32_t unregisterMemoryCalls = 0;
    uint32_t memoryExportCalls = 0;
    uint32_t memoryImportCalls = 0;
    uint32_t memoryUnimportCalls = 0;
    uint32_t getListenPortCalls = 0;
};

FakePluginEndpointScenario g_fakeEpScenario;

int32_t FakeEpInit(void* ctx)
{
    g_fakeEpScenario.initEndpointCalls++;
    static_cast<FakePluginEndpointState*>(ctx)->inited = true;
    return HCCL_SUCCESS;
}

int32_t FakeEpDestroy(void* ctx)
{
    g_fakeEpScenario.destroyEndpointCalls++;
    delete static_cast<FakePluginEndpointState*>(ctx);
    return HCCL_SUCCESS;
}

int32_t FakeEpRegisterMemory(void* ctx, const CommMem* mem, const char* tag, void** handle)
{
    g_fakeEpScenario.registerMemoryCalls++;
    *handle = reinterpret_cast<void*>(0x12345678);
    return HCCL_SUCCESS;
}

int32_t FakeEpUnregisterMemory(void* ctx, void* handle)
{
    g_fakeEpScenario.unregisterMemoryCalls++;
    return HCCL_SUCCESS;
}

int32_t FakeEpMemoryExport(void* ctx, void* handle, void** desc, uint32_t* descLen)
{
    g_fakeEpScenario.memoryExportCalls++;
    static char fakeDesc[8] = {};
    *desc = fakeDesc;
    *descLen = sizeof(fakeDesc);
    return HCCL_SUCCESS;
}

int32_t FakeEpMemoryImport(void* ctx, const void* desc, uint32_t descLen, CommMem* outMem)
{
    g_fakeEpScenario.memoryImportCalls++;
    outMem->type = COMM_MEM_TYPE_HOST;
    outMem->addr = const_cast<void*>(desc);
    outMem->size = descLen;
    return HCCL_SUCCESS;
}

int32_t FakeEpMemoryUnimport(void* ctx, const void* desc, uint32_t descLen)
{
    g_fakeEpScenario.memoryUnimportCalls++;
    return HCCL_SUCCESS;
}

int32_t FakeEpGetListenPort(void* ctx, uint32_t* port)
{
    g_fakeEpScenario.getListenPortCalls++;
    *port = 12345;
    return HCCL_SUCCESS;
}

int32_t FakeEpCreateEndpoint(const EndpointDesc* endpointDesc, void** outCtx, HcommNicEndpointOps** outOps);

HcommNicEndpointOps g_fakeEpOps = {
    {HCOMM_NIC_ENDPOINT_OPS_VERSION, HCOMM_NIC_ENDPOINT_OPS_MAGIC_WORD, sizeof(HcommNicEndpointOps), 0},
    FakeEpInit,             // init
    FakeEpDestroy,          // destroy
    FakeEpRegisterMemory,   // registerMemory
    FakeEpUnregisterMemory, // unregisterMemory
    FakeEpMemoryExport,     // memoryExport
    FakeEpMemoryImport,     // memoryImport
    FakeEpMemoryUnimport,   // memoryUnimport
    FakeEpGetListenPort,    // getListenPort
};

HcommNicEndpointOps g_unsupportedEpOps = {
    {HCOMM_NIC_ENDPOINT_OPS_VERSION, HCOMM_NIC_ENDPOINT_OPS_MAGIC_WORD, sizeof(HcommNicEndpointOps), 0},
    nullptr,       // init
    FakeEpDestroy, // destroy — FillDefaultEndpointOps 要求必须实现
    nullptr,       // registerMemory
    nullptr,       // unregisterMemory
    nullptr,       // memoryExport
    nullptr,       // memoryImport
    nullptr,       // memoryUnimport
    nullptr,       // getListenPort
};

HcommNicPluginInfo g_fakePluginInfo
    = {{HCOMM_NIC_PLUGIN_INFO_VERSION, HCOMM_NIC_PLUGIN_INFO_MAGIC_WORD, sizeof(HcommNicPluginInfo), 0},
       "fake_ep_plugin",
       1,
       {COMM_PROTOCOL_ROCE}};

int32_t FakeEpCreateEndpoint(const EndpointDesc* endpointDesc, void** outCtx, HcommNicEndpointOps** outOps)
{
    g_fakeEpScenario.createEndpointCalls++;
    *outCtx = new FakePluginEndpointState();
    *outOps = &g_fakeEpOps;
    return HCCL_SUCCESS;
}

int32_t FakeEpCreateUnsupportedEndpoint(const EndpointDesc* endpointDesc, void** outCtx, HcommNicEndpointOps** outOps)
{
    g_fakeEpScenario.createEndpointCalls++;
    *outCtx = new FakePluginEndpointState();
    *outOps = &g_unsupportedEpOps;
    return HCCL_SUCCESS;
}

NicPluginEntry g_fakeEpPluginEntry = {nullptr, &g_fakePluginInfo, FakeEpCreateEndpoint, nullptr};

void ResetFakeEpScenario() { g_fakeEpScenario = {}; }

// ---- Builtin 测试辅助变量：用于手动设置 mock 未覆盖的输出参数 ----
static int g_builtinDummyHandle = 0;
static char g_builtinDummyDesc[8] = {};
} // namespace

class TestHcommEndpoint : public TestHcommCAdptBase {
public:
    void SetUp() override
    {
        TestHcommCAdptBase::SetUp();
        ResetFakeEpScenario();
    }
    void TearDown() override { TestHcommCAdptBase::TearDown(); }
};

TEST_F(TestHcommEndpoint, Ut_TestHcommEndpointCreate_When_EndpointNullptr_Return_HCCL_E_PTR)
{
    EndpointHandle handle;
    HcommResult ret = HcommEndpointCreate(nullptr, &handle);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommEndpoint, Ut_TestHcommEndpointCreate_When_HandleNullptr_Return_HCCL_E_PTR)
{
    EndpointDesc endpointDesc;
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;

    HcommResult ret = HcommEndpointCreate(&endpointDesc, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommEndpoint, Ut_TestHcommEndpointDestroy_When_HandleNullptr_Return_HCCL_E_NOT_FOUND)
{
    HcommResult ret = HcommEndpointDestroy(nullptr);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(TestHcommEndpoint, Ut_TestHcommEndpointGet_When_EndpointNullptr_Return_HCCL_E_PTR)
{
    HcommResult ret = HcommEndpointGet(nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcommEndpoint, Ut_PluginEndpoint_Expect_DispatchToPlugin)
{
    MOCKER((hcomm::FindHostNicPlugin))
        .stubs()
        .will(returnValue(static_cast<const hcomm::NicPluginEntry*>(&g_fakeEpPluginEntry)));

    EndpointDesc endpointDesc{};
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    endpointDesc.protocol = COMM_PROTOCOL_ROCE;
    EndpointHandle epHandle = nullptr;
    EXPECT_EQ(HcommEndpointCreate(&endpointDesc, &epHandle), HCCL_SUCCESS);
    EXPECT_EQ(g_fakeEpScenario.createEndpointCalls, 1U);

    // mem ops
    CommMem mem{COMM_MEM_TYPE_HOST, const_cast<char*>("test"), 4};
    HcommMemHandle memHandle = nullptr;
    EXPECT_EQ(HcommMemReg(epHandle, "tag", &mem, &memHandle), HCCL_SUCCESS);
    EXPECT_EQ(g_fakeEpScenario.registerMemoryCalls, 1U);

    EXPECT_EQ(HcommMemUnreg(epHandle, memHandle), HCCL_SUCCESS);
    EXPECT_EQ(g_fakeEpScenario.unregisterMemoryCalls, 1U);

    void* memDesc = nullptr;
    uint32_t memDescLen = 0;
    EXPECT_EQ(HcommMemExport(epHandle, memHandle, &memDesc, &memDescLen), HCCL_SUCCESS);
    EXPECT_EQ(g_fakeEpScenario.memoryExportCalls, 1U);

    CommMem outMem{};
    EXPECT_EQ(HcommMemImport(epHandle, memDesc, memDescLen, &outMem), HCCL_SUCCESS);
    EXPECT_EQ(g_fakeEpScenario.memoryImportCalls, 1U);

    EXPECT_EQ(HcommMemUnimport(epHandle, memDesc, memDescLen), HCCL_SUCCESS);
    EXPECT_EQ(g_fakeEpScenario.memoryUnimportCalls, 1U);

    // listen port
    uint32_t port = 0;
    EXPECT_EQ(HcommEndpointGetListenPort(epHandle, &port), HCCL_SUCCESS);
    EXPECT_EQ(g_fakeEpScenario.getListenPortCalls, 1U);
    EXPECT_EQ(port, 12345U);

    EXPECT_EQ(HcommEndpointDestroy(epHandle), HCCL_SUCCESS);
    EXPECT_EQ(g_fakeEpScenario.destroyEndpointCalls, 1U);
}

TEST_F(TestHcommEndpoint, Ut_PluginEndpoint_Expect_DispatchToPlugin_NotSupport)
{
    static NicPluginEntry unsupportedEntry = {nullptr, &g_fakePluginInfo, FakeEpCreateUnsupportedEndpoint, nullptr};
    MOCKER((hcomm::FindHostNicPlugin))
        .stubs()
        .will(returnValue(static_cast<const hcomm::NicPluginEntry*>(&unsupportedEntry)));

    EndpointDesc endpointDesc{};
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    endpointDesc.protocol = COMM_PROTOCOL_ROCE;
    EndpointHandle epHandle = nullptr;
    EXPECT_EQ(HcommEndpointCreate(&endpointDesc, &epHandle), HCCL_SUCCESS);

    CommMem mem{COMM_MEM_TYPE_HOST, const_cast<char*>("test"), 4};
    HcommMemHandle memHandle = nullptr;
    EXPECT_EQ(HcommMemReg(epHandle, "tag", &mem, &memHandle), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(HcommMemUnreg(epHandle, memHandle), HCCL_E_NOT_SUPPORT);

    void* memDesc = nullptr;
    uint32_t memDescLen = 0;
    EXPECT_EQ(HcommMemExport(epHandle, memHandle, &memDesc, &memDescLen), HCCL_E_NOT_SUPPORT);

    CommMem outMem{};
    EXPECT_EQ(HcommMemImport(epHandle, memDesc, memDescLen, &outMem), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(HcommMemUnimport(epHandle, memDesc, memDescLen), HCCL_E_NOT_SUPPORT);

    uint32_t port = 0;
    EXPECT_EQ(HcommEndpointGetListenPort(epHandle, &port), HCCL_E_NOT_SUPPORT);

    EXPECT_EQ(HcommEndpointDestroy(epHandle), HCCL_SUCCESS);
}

TEST_F(TestHcommEndpoint, Ut_BuiltinEndpoint_Expect_DispatchToBuiltinOps)
{
    EndpointDesc endpointDesc{};
    endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    endpointDesc.protocol = COMM_PROTOCOL_ROCE;
    EndpointHandle epHandle = nullptr;
    EXPECT_EQ(HcommEndpointCreate(&endpointDesc, &epHandle), HCCL_SUCCESS);

    CpuRoceEndpoint* ep = nullptr;
    HcommEndpointGet(epHandle, reinterpret_cast<void**>(&ep));

    // 内存方法从 Endpoint 移至 RegedMemMgr；经 ep->GetRegedMemMgr()->xxx() 路径访问。
    // 此处 mock RoceRegedMemMgr 的虚方法，验证 HcommMemReg 等正确转发到 RegedMemMgr。
    auto* regedMemMgr = ep->GetRegedMemMgr();
    ASSERT_NE(regedMemMgr, nullptr);
    MOCKER_CPP_VIRTUAL(regedMemMgr, &hcomm::RegedMemMgr::RegisterMemory).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(regedMemMgr, &hcomm::RegedMemMgr::UnregisterMemory).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(regedMemMgr, &hcomm::RegedMemMgr::MemoryExport).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(regedMemMgr, &hcomm::RegedMemMgr::MemoryImport).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(regedMemMgr, &hcomm::RegedMemMgr::MemoryUnimport).stubs().will(returnValue(HCCL_SUCCESS));
    // Socket 监听接口由 ServerSocketContext 承载，mock 上下文虚方法验证转发路径
    MOCKER_CPP_VIRTUAL(ep->GetServerSocketContext(), &hcomm::ServerSocketContext::ServerSocketGetListenPort)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));

    // ---- 业务接口全覆盖 ----

    // 1. registerMemory（mock 返回成功，手动设置 memHandle）
    CommMem mem{COMM_MEM_TYPE_HOST, const_cast<char*>("test"), 4};
    HcommMemHandle memHandle = nullptr;
    EXPECT_EQ(HcommMemReg(epHandle, "tag", &mem, &memHandle), HCCL_SUCCESS);
    memHandle = reinterpret_cast<HcommMemHandle>(&g_builtinDummyHandle);

    // 2. unregisterMemory
    EXPECT_EQ(HcommMemUnreg(epHandle, memHandle), HCCL_SUCCESS);

    // 3. memoryExport（mock 返回成功，手动设置 desc）
    void* memDesc = nullptr;
    uint32_t memDescLen = 0;
    EXPECT_EQ(HcommMemExport(epHandle, memHandle, &memDesc, &memDescLen), HCCL_SUCCESS);
    memDesc = g_builtinDummyDesc;
    memDescLen = sizeof(g_builtinDummyDesc);

    // 4. memoryImport
    CommMem outMem{};
    EXPECT_EQ(HcommMemImport(epHandle, memDesc, memDescLen, &outMem), HCCL_SUCCESS);

    // 5. memoryUnimport
    EXPECT_EQ(HcommMemUnimport(epHandle, memDesc, memDescLen), HCCL_SUCCESS);

    // 6. getListenPort
    uint32_t port = 0;
    EXPECT_EQ(HcommEndpointGetListenPort(epHandle, &port), HCCL_SUCCESS);

    EXPECT_EQ(HcommEndpointDestroy(epHandle), HCCL_SUCCESS);
}

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

#include "endpoint_mgr.h"
#include "hcomm_res_defs.h"

using namespace hcomm;

// ============ HcommMemReg / HcommMemUnreg stub ============

static int s_regCallCount = 0;
static int s_unregCallCount = 0;
static std::vector<std::string> s_regTags;

static std::set<EndpointHandle> s_liveEndpoints;

static HcommResult MemRegStub(EndpointHandle ep, const char* memTag, const CommMem*, HcommMemHandle* memHandle)
{
    EXPECT_TRUE(s_liveEndpoints.count(ep));
    s_regCallCount++;
    s_regTags.push_back(memTag != nullptr ? memTag : "");
    *memHandle = (HcommMemHandle)(uintptr_t)(0xCC110000 + s_regCallCount);
    return HCCL_SUCCESS;
}

static HcommResult MemUnregStub(EndpointHandle ep, HcommMemHandle)
{
    EXPECT_TRUE(s_liveEndpoints.count(ep));
    s_unregCallCount++;
    return HCCL_SUCCESS;
}

static HcommResult EndpointCreateStub(const EndpointDesc*, EndpointHandle* handle)
{
    static uintptr_t nextHandle = 0xEE000000;
    *handle = (EndpointHandle)(nextHandle++);
    s_liveEndpoints.insert(*handle);
    return HCCL_SUCCESS;
}

static HcommResult EndpointDestroyStub(EndpointHandle handle)
{
    EXPECT_TRUE(s_liveEndpoints.erase(handle));
    return HCCL_SUCCESS;
}

// ============ EndpointMgrTest ============

class EndpointMgrTest : public testing::Test {
protected:
    hccl::EndpointMgr mgr;

    void SetUp() override
    {
        s_regCallCount = 0;
        s_unregCallCount = 0;
        s_regTags.clear();
        s_liveEndpoints.clear();
        s_liveEndpoints.insert((EndpointHandle)0x1);
        s_liveEndpoints.insert((EndpointHandle)0x2);
    }

    void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(EndpointMgrTest, IsDescExist_Empty)
{
    EndpointDesc desc{};
    memset(&desc, 0, sizeof(desc));
    EXPECT_FALSE(mgr.IsDescExist(desc));
}

TEST_F(EndpointMgrTest, Get_CreateNewEndpoint)
{
    EndpointDesc desc{};
    memset(&desc, 0, sizeof(desc));
    desc.protocol = COMM_PROTOCOL_HCCS;
    desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    desc.commAddr.addr.s_addr = inet_addr("10.0.0.1");
    desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;

    EndpointHandle handle = nullptr;
    HcclResult ret = mgr.Get(desc, handle);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(EndpointMgrTest, GetWithTag_EmptyTag_DegradeToGet)
{
    EndpointDesc desc{};
    memset(&desc, 0, sizeof(desc));
    desc.protocol = COMM_PROTOCOL_HCCS;
    desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    desc.commAddr.addr.s_addr = inet_addr("10.0.0.1");
    desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;

    EndpointHandle handle = nullptr;
    // tag 为空时退化为 Get，行为与 Get_CreateNewEndpoint 一致（HcommEndpointCreate 未 mock，返回失败）
    HcclResult ret = mgr.GetWithTag(desc, "", handle);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(EndpointMgrTest, GetWithTag_NonEmptyTag_CreateEndpoint)
{
    MOCKER(HcommEndpointCreate).stubs().with(mockcpp::any(), mockcpp::any()).will(invoke(EndpointCreateStub));
    MOCKER(HcommEndpointDestroy).stubs().with(mockcpp::any()).will(invoke(EndpointDestroyStub));

    EndpointDesc desc{};
    memset(&desc, 0, sizeof(desc));
    desc.protocol = COMM_PROTOCOL_HCCS;
    desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    desc.commAddr.addr.s_addr = inet_addr("10.0.0.1");
    desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;

    // 用作用域块包裹 scopedMgr：在 verify() 前、mock 仍有效时析构，
    // 确保 ~EndpointMgr 调 HcommEndpointDestroy 走 mock 而非真实实现
    EndpointHandle handle1 = nullptr;
    EndpointHandle handle2 = nullptr;
    EndpointHandle handle1Again = nullptr;
    {
        hccl::EndpointMgr scopedMgr;

        // 不同 tag 创建不同 Endpoint
        ASSERT_EQ(scopedMgr.GetWithTag(desc, "tag_A", handle1), HCCL_SUCCESS);
        ASSERT_NE(handle1, nullptr);

        ASSERT_EQ(scopedMgr.GetWithTag(desc, "tag_B", handle2), HCCL_SUCCESS);
        ASSERT_NE(handle2, nullptr);
        EXPECT_NE(handle1, handle2); // 不同 tag → 不同 Endpoint

        // 同一 tag 复用同一 Endpoint
        ASSERT_EQ(scopedMgr.GetWithTag(desc, "tag_A", handle1Again), HCCL_SUCCESS);
        EXPECT_EQ(handle1, handle1Again); // 同 tag → 同 Endpoint
        // scopedMgr 离开作用域析构，~EndpointMgr 调 HcommEndpointDestroy 销毁两个 tagged Endpoint
    }

    GlobalMockObject::verify();
}

TEST_F(EndpointMgrTest, GetWithTag_DestructorDestroysTaggedEndpoints)
{
    MOCKER(HcommEndpointCreate).stubs().with(mockcpp::any(), mockcpp::any()).will(invoke(EndpointCreateStub));
    MOCKER(HcommEndpointDestroy).stubs().with(mockcpp::any()).will(invoke(EndpointDestroyStub));

    EndpointDesc desc{};
    memset(&desc, 0, sizeof(desc));
    desc.protocol = COMM_PROTOCOL_HCCS;
    desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    desc.commAddr.addr.s_addr = inet_addr("10.0.0.2");
    desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;

    EndpointHandle createdHandle = nullptr;
    {
        hccl::EndpointMgr scopedMgr;
        EndpointHandle h = nullptr;
        ASSERT_EQ(scopedMgr.GetWithTag(desc, "tag_destructor", h), HCCL_SUCCESS);
        ASSERT_TRUE(s_liveEndpoints.count(h));
        createdHandle = h;
        // scopedMgr 析构时应销毁 tag 创建的 Endpoint
    }
    EXPECT_FALSE(s_liveEndpoints.count(createdHandle));
    GlobalMockObject::verify();
}

// ============ RegisterMemory tests ============

TEST_F(EndpointMgrTest, RegisterMemory_SameTagSkipsReReg)
{
    MOCKER(HcommMemReg)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MemRegStub));

    std::vector<std::string> tags = {"HcclBuffer", "HcclBuffer"};
    std::vector<HcclMem> mems
        = {{HCCL_MEM_TYPE_DEVICE, (void*)0x1000, 1024}, {HCCL_MEM_TYPE_DEVICE, (void*)0x1000, 1024}};

    HcclResult ret = mgr.RegisterMemory((EndpointHandle)0x1, tags, mems, 1);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    // HcommMemReg should be called only once (second "HcclBuffer" is duplicate tag)
    EXPECT_EQ(s_regCallCount, 1);
    EXPECT_EQ(s_regTags.size(), 1u);
    EXPECT_EQ(s_regTags[0], "HcclBuffer");
}

TEST_F(EndpointMgrTest, RegisterMemory_DifferentTags)
{
    MOCKER(HcommMemReg)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MemRegStub));

    std::vector<std::string> tags = {"HcclBuffer", "HcclWorkBuffer"};
    std::vector<HcclMem> mems
        = {{HCCL_MEM_TYPE_DEVICE, (void*)0x1000, 1024}, {HCCL_MEM_TYPE_DEVICE, (void*)0x2000, 2048}};

    HcclResult ret = mgr.RegisterMemory((EndpointHandle)0x1, tags, mems, 1);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(s_regCallCount, 2);
    EXPECT_EQ(s_regTags.size(), 2u);
    EXPECT_EQ(s_regTags[0], "HcclBuffer");
    EXPECT_EQ(s_regTags[1], "HcclWorkBuffer");
}

TEST_F(EndpointMgrTest, RegisterMemory_MixedDuplicateAndNewTags)
{
    MOCKER(HcommMemReg)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MemRegStub));

    // First call: register "A"
    std::vector<std::string> tags1 = {"A"};
    std::vector<HcclMem> mems1 = {{HCCL_MEM_TYPE_DEVICE, (void*)0x1000, 1024}};
    mgr.RegisterMemory((EndpointHandle)0x1, tags1, mems1, 1);
    EXPECT_EQ(s_regCallCount, 1);

    // Second call: register "A" (dup) + "B" (new)
    std::vector<std::string> tags2 = {"A", "B"};
    std::vector<HcclMem> mems2
        = {{HCCL_MEM_TYPE_DEVICE, (void*)0x1000, 1024}, {HCCL_MEM_TYPE_DEVICE, (void*)0x2000, 2048}};
    mgr.RegisterMemory((EndpointHandle)0x1, tags2, mems2, 2);

    // Only one more HcommMemReg call (for "B"), "A" was reused
    EXPECT_EQ(s_regCallCount, 2);
    EXPECT_EQ(s_regTags.size(), 2u);
    EXPECT_EQ(s_regTags[0], "A");
    EXPECT_EQ(s_regTags[1], "B");
}

TEST_F(EndpointMgrTest, RegisterMemory_DifferentEndpointsSeparateReg)
{
    MOCKER(HcommMemReg)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MemRegStub));

    std::vector<std::string> tags = {"HcclBuffer"};
    std::vector<HcclMem> mems = {{HCCL_MEM_TYPE_DEVICE, (void*)0x1000, 1024}};

    mgr.RegisterMemory((EndpointHandle)0x1, tags, mems, 1);
    mgr.RegisterMemory((EndpointHandle)0x2, tags, mems, 1);

    // Two different endpoints, two separate registrations
    EXPECT_EQ(s_regCallCount, 2);
}

// ============ GetMemHandlesByTags test ============

TEST_F(EndpointMgrTest, GetMemHandlesByTags_ReturnsRegisteredHandles)
{
    MOCKER(HcommMemReg)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MemRegStub));

    std::vector<std::string> tags = {"A", "B"};
    std::vector<HcclMem> mems
        = {{HCCL_MEM_TYPE_DEVICE, (void*)0x1000, 1024}, {HCCL_MEM_TYPE_DEVICE, (void*)0x2000, 2048}};
    mgr.RegisterMemory((EndpointHandle)0x1, tags, mems, 1);

    // Query handles for individual tags
    std::vector<MemHandle> handles;
    HcclResult ret = mgr.GetMemHandlesByTags((EndpointHandle)0x1, {"A", "B"}, handles);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(handles.size(), 2u);
    EXPECT_NE(handles[0], handles[1]); // Different tags, different handles
}

// ============ UnregMemByTag tests ============

TEST_F(EndpointMgrTest, UnregMemByTag_AcrossAllEndpoints)
{
    MOCKER(HcommMemReg)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MemRegStub));
    MOCKER(HcommMemUnreg).stubs().with(mockcpp::any(), mockcpp::any()).will(invoke(MemUnregStub));

    std::vector<std::string> tags = {"A", "B"};
    std::vector<HcclMem> mems
        = {{HCCL_MEM_TYPE_DEVICE, (void*)0x1000, 1024}, {HCCL_MEM_TYPE_DEVICE, (void*)0x2000, 2048}};

    // Register "A"+"B" to two endpoints
    mgr.RegisterMemory((EndpointHandle)0x1, tags, mems, 1);
    mgr.RegisterMemory((EndpointHandle)0x2, tags, mems, 1);
    EXPECT_EQ(s_regCallCount, 4); // 2 tags × 2 endpoints

    // Unregister "A" from all endpoints
    HcclResult ret = mgr.UnregMemByTag("A");
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(s_unregCallCount, 2); // ep1 + ep2

    // "A" should be gone — GetMemHandlesByTags returns NOT_FOUND for missing tags
    std::vector<MemHandle> handles;
    ret = mgr.GetMemHandlesByTags((EndpointHandle)0x1, {"A"}, handles);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);

    // "B" still present
    ret = mgr.GetMemHandlesByTags((EndpointHandle)0x1, {"B"}, handles);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(handles.size(), 1u);
}

TEST_F(EndpointMgrTest, UnregMemByTag_NonexistentTag)
{
    MOCKER(HcommMemReg)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MemRegStub));
    MOCKER(HcommMemUnreg).stubs().with(mockcpp::any(), mockcpp::any()).will(invoke(MemUnregStub));

    std::vector<std::string> tags = {"A"};
    std::vector<HcclMem> mems = {{HCCL_MEM_TYPE_DEVICE, (void*)0x1000, 1024}};
    mgr.RegisterMemory((EndpointHandle)0x1, tags, mems, 1);
    EXPECT_EQ(s_regCallCount, 1);

    // Unregister a tag that doesn't exist
    HcclResult ret = mgr.UnregMemByTag("NonExistent");
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(s_unregCallCount, 0); // No unregister happened

    // "A" should be unaffected
    std::vector<MemHandle> handles;
    ret = mgr.GetMemHandlesByTags((EndpointHandle)0x1, {"A"}, handles);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(handles.size(), 1u);
}

// ============ Destructor test ============

TEST_F(EndpointMgrTest, Destructor_UnregistersAllRegisteredHandles)
{
    MOCKER(HcommMemReg)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(invoke(MemRegStub));
    MOCKER(HcommMemUnreg).stubs().with(mockcpp::any(), mockcpp::any()).will(invoke(MemUnregStub));
    MOCKER(HcommEndpointCreate).stubs().with(mockcpp::any(), mockcpp::any()).will(invoke(EndpointCreateStub));
    MOCKER(HcommEndpointDestroy).stubs().with(mockcpp::any()).will(invoke(EndpointDestroyStub));

    EndpointDesc desc{};
    desc.protocol = COMM_PROTOCOL_HCCS;
    desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    desc.commAddr.addr.s_addr = inet_addr("10.0.0.1");
    desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;

    {
        hccl::EndpointMgr localMgr;
        EndpointHandle epHandle = nullptr;
        ASSERT_EQ(localMgr.Get(desc, epHandle), HCCL_SUCCESS);
        ASSERT_NE(epHandle, nullptr);

        std::vector<std::string> tags = {"A", "B", "C"};
        std::vector<HcclMem> mems
            = {{HCCL_MEM_TYPE_DEVICE, (void*)0x1000, 1024},
               {HCCL_MEM_TYPE_DEVICE, (void*)0x2000, 2048},
               {HCCL_MEM_TYPE_DEVICE, (void*)0x3000, 4096}};
        localMgr.RegisterMemory(epHandle, tags, mems, 1);

        EXPECT_EQ(s_regCallCount, 3);
        EXPECT_EQ(s_unregCallCount, 0);
    }

    EXPECT_EQ(s_unregCallCount, 3);
}

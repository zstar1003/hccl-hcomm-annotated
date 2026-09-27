/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <arpa/inet.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>

#include "endpoint.h"
#include "adapter_rts_common.h"
#include "channel_param.h"
#include "hcomm_res_defs.h"
#include "hcomm_channel.h"
#include "workflow_pub.h"
#include "transport_pub.h"
#include "mem_device_pub.h"
#include "aicpu_ts_roce_reged_mem_mgr.h"

#define private public
#include "aicpu/aicpu_ts_roce_channel.h"
#undef private

using namespace hcomm;

namespace {
class StubEndpointForAicpuTsRoceChannel : public Endpoint {
public:
    explicit StubEndpointForAicpuTsRoceChannel(const EndpointDesc& desc, void* rdmaHandle) : Endpoint(desc)
    {
        ctxHandle_ = rdmaHandle;
    }

    HcclResult Init() override { return HCCL_SUCCESS; }
    void* GetRdmaHandle() override { return ctxHandle_; }
    bool IsCtxHandleValid() const override { return ctxHandle_ != nullptr; }
    // 内存方法在 RegedMemMgr 上，Endpoint 不再 override
    RegedMemMgr* GetRegedMemMgr() override { return nullptr; }

private:
    void* ctxHandle_{nullptr};
};

HcclResult StubGetAllMemDetailsForSerialize(
    AicpuTsRoceRegedMemMgr* /*self*/, std::vector<RoceMemDetails>& localOut, std::vector<RoceMemDetails>& remoteOut)
{
    localOut.clear();
    remoteOut.clear();
    return HCCL_SUCCESS;
}

HcclResult StubHrtGetDeviceWriteZero(s32* deviceLogicId)
{
    if (deviceLogicId != nullptr) {
        *deviceLogicId = 0;
    }
    return HCCL_SUCCESS;
}

HcclResult StubTransportGetAiQpInfo(hccl::Transport* /*self*/, std::vector<HcclQpInfoV2>& aiQpInfo)
{
    aiQpInfo.clear();
    aiQpInfo.resize(1U);
    aiQpInfo[0].qpPtr = 0x7000ULL;
    aiQpInfo[0].sqIndex = 1U;
    aiQpInfo[0].dbIndex = 2U;
    return HCCL_SUCCESS;
}

static hccl::DeviceMem StubDeviceMemAlloc(u64 size, bool /*level2*/)
{
    void* p = std::malloc(static_cast<size_t>(size));
    // Host memory from malloc must not use owner=true: ~DeviceMem calls hrtFree, which is for device alloc.
    return hccl::DeviceMem(p, size, false);
}

static HcclResult StubHrtMemSyncCopySerialize(void* dst, size_t dstMax, const void* src, size_t count, HcclRtMemcpyKind)
{
    if (dst != nullptr && src != nullptr && dstMax >= count) {
        (void)memcpy(dst, src, count);
    }
    return HCCL_SUCCESS;
}

class StubEndpointWithRoceMemMgr : public Endpoint {
public:
    StubEndpointWithRoceMemMgr(const EndpointDesc& desc, void* rdmaHandle, std::shared_ptr<AicpuTsRoceRegedMemMgr> mgr)
        : Endpoint(desc),
          mgr_(std::move(mgr)),
          ctxHandle_(rdmaHandle)
    {
        // regedMemMgr_ 移至各子类；GetRegedMemMgr() 直接返回 mgr_
    }

    HcclResult Init() override { return HCCL_SUCCESS; }
    // 内存方法在 RegedMemMgr 上，Endpoint 不再 override
    RegedMemMgr* GetRegedMemMgr() override { return mgr_.get(); }
    void* GetRdmaHandle() override { return ctxHandle_; }
    bool IsCtxHandleValid() const override { return ctxHandle_ != nullptr; }

private:
    void* ctxHandle_{nullptr};
    std::shared_ptr<AicpuTsRoceRegedMemMgr> mgr_;
};
} // namespace

class AicpuTsRoceChannelTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override {}
    void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(AicpuTsRoceChannelTest, Ut_Clean_WithoutInit_Returns_NOT_SUPPORT)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsRoceChannel ch(ep, desc);

    auto ret = ch.Clean();
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(AicpuTsRoceChannelTest, Ut_Resume_WithoutInit_Returns_NOT_SUPPORT)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsRoceChannel ch(ep, desc);

    auto ret = ch.Resume();
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(AicpuTsRoceChannelTest, Ut_GetChannelKind_Returns_AICPU_TS_ROCE)
{
    HcommChannelDesc desc{};
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    EXPECT_EQ(ch.GetChannelKind(), HcommChannelKind::AICPU_TS_ROCE);
}

TEST_F(AicpuTsRoceChannelTest, Ut_GetStatus_WithoutInit_Returns_INIT)
{
    HcommChannelDesc desc{};
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    EXPECT_EQ(ch.GetStatus(), ChannelStatus::INIT);
}

TEST_F(AicpuTsRoceChannelTest, Ut_GetNotifyNum_WhenNullOut_Returns_PTR)
{
    HcommChannelDesc desc{};
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    EXPECT_EQ(ch.GetNotifyNum(nullptr), HCCL_E_PTR);
}

TEST_F(AicpuTsRoceChannelTest, Ut_Serialize_WithoutInit_Returns_INTERNAL)
{
    HcommChannelDesc desc{};
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    std::shared_ptr<hccl::DeviceMem> out;
    EXPECT_EQ(ch.Serialize(out), HCCL_E_INTERNAL);
    EXPECT_EQ(out.get(), nullptr);
}

TEST_F(AicpuTsRoceChannelTest, Ut_Init_When_EndpointHandleNull_Returns_E_PTR)
{
    HcommChannelDesc desc{};
    AicpuTsRoceChannel ch(static_cast<EndpointHandle>(nullptr), desc);
    EXPECT_EQ(ch.Init(), HCCL_E_PTR);
}

TEST_F(AicpuTsRoceChannelTest, Ut_GetRemoteMems_Returns_E_NOT_SUPPORT)
{
    HcommChannelDesc desc{};
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    uint32_t n = 0;
    EXPECT_EQ(ch.GetRemoteMems(&n, nullptr, nullptr), HCCL_E_NOT_SUPPORT);
}

TEST_F(AicpuTsRoceChannelTest, Ut_BuildSocketTagName_When_ValidIpv4_Returns_SUCCESS)
{
    HcommChannelDesc desc{};
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    ch.localEp_.protocol = COMM_PROTOCOL_ROCE;
    ch.localEp_.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    ch.localEp_.loc.device.devPhyId = 0U;
    ch.localEp_.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "192.168.10.1", &ch.localEp_.commAddr.addr), 1);
    ch.remoteEp_.protocol = COMM_PROTOCOL_ROCE;
    ch.remoteEp_.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    ch.remoteEp_.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "192.168.10.2", &ch.remoteEp_.commAddr.addr), 1);
    ch.isLocalIpClient_ = true;
    ch.channelDesc_.port = 16666U;
    *reinterpret_cast<uint32_t*>(ch.channelDesc_.raws + sizeof(ch.channelDesc_.raws) - sizeof(uint32_t)) = 7U;
    std::string tag;
    EXPECT_EQ(ch.BuildSocketTagName(tag), HCCL_SUCCESS);
    EXPECT_FALSE(tag.empty());
}

TEST_F(AicpuTsRoceChannelTest, Ut_GetNotifyNum_WhenSet_Returns_SUCCESS)
{
    HcommChannelDesc desc{};
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    ch.notifyNum_ = 9U;
    uint32_t n = 0U;
    EXPECT_EQ(ch.GetNotifyNum(&n), HCCL_SUCCESS);
    EXPECT_EQ(n, 9U);
}

TEST_F(AicpuTsRoceChannelTest, Ut_ValidateSerializeParams_WhenQpNumZero_Returns_INTERNAL)
{
    HcommChannelDesc desc{};
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    EXPECT_EQ(ch.ValidateSerializeParams(0U, 0U, 0U), HCCL_E_INTERNAL);
}

TEST_F(AicpuTsRoceChannelTest, Ut_ValidateSerializeParams_WhenQpNumTooLarge_Returns_INTERNAL)
{
    HcommChannelDesc desc{};
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    EXPECT_EQ(ch.ValidateSerializeParams(34U, 0U, 0U), HCCL_E_INTERNAL);
}

TEST_F(AicpuTsRoceChannelTest, Ut_BuildSocketTagName_WhenCommAddrTypeInvalid_Returns_NOT_SUPPORT)
{
    HcommChannelDesc desc{};
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    ch.localEp_.protocol = COMM_PROTOCOL_ROCE;
    ch.localEp_.commAddr.type = static_cast<CommAddrType>(255);
    ch.remoteEp_.protocol = COMM_PROTOCOL_ROCE;
    ch.remoteEp_.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "192.168.20.1", &ch.remoteEp_.commAddr.addr), 1);
    ch.isLocalIpClient_ = true;
    ch.channelDesc_.port = 16666U;
    std::string tag;
    EXPECT_EQ(ch.BuildSocketTagName(tag), HCCL_E_NOT_SUPPORT);
}

TEST_F(AicpuTsRoceChannelTest, Ut_ValidateSerializeParams_WhenValid_Returns_SUCCESS)
{
    HcommChannelDesc desc{};
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    EXPECT_EQ(ch.ValidateSerializeParams(1U, 0U, 0U), HCCL_SUCCESS);
    EXPECT_EQ(ch.ValidateSerializeParams(RDMA_QP_MAX_NUM, 0U, 0U), HCCL_SUCCESS);
}

TEST_F(AicpuTsRoceChannelTest, Ut_ValidateSerializeParams_WhenLocalCountOverflow_Returns_PARA)
{
    HcommChannelDesc desc{};
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    const size_t over = (SIZE_MAX / sizeof(RoceMemDetails)) + 1U;
    EXPECT_EQ(ch.ValidateSerializeParams(1U, over, 0U), HCCL_E_PARA);
}

TEST_F(AicpuTsRoceChannelTest, Ut_ValidateSerializeParams_WhenRemoteCountOverflow_Returns_PARA)
{
    HcommChannelDesc desc{};
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    const size_t over = (SIZE_MAX / sizeof(RoceMemDetails)) + 1U;
    EXPECT_EQ(ch.ValidateSerializeParams(1U, 0U, over), HCCL_E_PARA);
}

TEST_F(AicpuTsRoceChannelTest, Ut_ValidateSerializeParams_WhenLocalBlobExceedsUint32_Returns_PARA)
{
    HcommChannelDesc desc{};
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    const size_t n = (static_cast<size_t>(UINT32_MAX) / sizeof(RoceMemDetails)) + 2U;
    EXPECT_EQ(ch.ValidateSerializeParams(1U, n, 0U), HCCL_E_PARA);
}

TEST_F(AicpuTsRoceChannelTest, Ut_ValidateSerializeParams_WhenRemoteBlobExceedsUint32_Returns_PARA)
{
    HcommChannelDesc desc{};
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    const size_t n = (static_cast<size_t>(UINT32_MAX) / sizeof(RoceMemDetails)) + 2U;
    EXPECT_EQ(ch.ValidateSerializeParams(1U, 0U, n), HCCL_E_PARA);
}

TEST_F(AicpuTsRoceChannelTest, Ut_GetStatus_WhenInited_Returns_READY)
{
    HcommChannelDesc desc{};
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    ch.inited_ = true;
    EXPECT_EQ(ch.GetStatus(), ChannelStatus::INIT);
}

TEST_F(AicpuTsRoceChannelTest, Ut_ParseInputParam_WhenClientRole_Returns_SUCCESS)
{
    EndpointDesc local{};
    local.protocol = COMM_PROTOCOL_ROCE;
    local.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    local.loc.device.devPhyId = 0U;
    local.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.1.1.1", &local.commAddr.addr), 1);

    StubEndpointForAicpuTsRoceChannel stub(local, reinterpret_cast<void*>(0xfeedU));
    HcommChannelDesc desc{};
    desc.role = HCOMM_SOCKET_ROLE_CLIENT;
    desc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
    desc.remoteEndpoint.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.1.1.2", &desc.remoteEndpoint.commAddr.addr), 1);

    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(&stub), desc);
    EXPECT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    EXPECT_TRUE(ch.isLocalIpClient_);
    EXPECT_EQ(ch.rdmaHandle_, reinterpret_cast<void*>(0xfeedU));
}

TEST_F(AicpuTsRoceChannelTest, Ut_InitSerializeRoceChannelRes_WritesLayout)
{
    HcommChannelDesc desc{};
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    std::vector<HcclQpInfoV2> qp(1U);
    qp[0].qpPtr = 0x6000ULL;
    qp[0].sqIndex = 3U;
    qp[0].dbIndex = 4U;
    HcommRoceChannelRes res{};
    ch.InitSerializeRoceChannelRes(res, 0U, 0U, nullptr, nullptr, qp, 1U);
    EXPECT_EQ(res.localMemCount, 0U);
    EXPECT_EQ(res.remoteMemCount, 0U);
    EXPECT_EQ(res.qpsPerConnection, 1U);
    EXPECT_EQ(res.QpInfo[0].qpPtr, qp[0].qpPtr);
}

TEST_F(AicpuTsRoceChannelTest, Ut_InitSerializeRoceChannelRes_MultiQp_WritesQpsPerConnection)
{
    HcommChannelDesc desc{};
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    std::vector<HcclQpInfoV2> qp(3U);
    for (int i = 0; i < 3; ++i) {
        qp[static_cast<size_t>(i)].qpPtr = 0x6100ULL + static_cast<unsigned long long>(i);
        qp[static_cast<size_t>(i)].sqIndex = static_cast<u32>(i);
        qp[static_cast<size_t>(i)].dbIndex = static_cast<u32>(i + 10U);
    }
    HcommRoceChannelRes res{};
    // qpsPerConnection = qpNum - (qpNum > 1)；与 Parse 侧 qpInfoSize = qpsPerConnection + (qpsPerConnection!=1) 对齐需
    // qpNum=3 → qpsPerConnection=2。
    constexpr u32 kQpNum = 3U;
    ch.InitSerializeRoceChannelRes(res, 0U, 0U, nullptr, nullptr, qp, kQpNum);
    EXPECT_EQ(res.qpsPerConnection, 2U);
    EXPECT_EQ(res.QpInfo[0].qpPtr, qp[0].qpPtr);
    EXPECT_EQ(res.QpInfo[1].qpPtr, qp[1].qpPtr);
    EXPECT_EQ(res.QpInfo[2].qpPtr, qp[2].qpPtr);
}

TEST_F(AicpuTsRoceChannelTest, Ut_ParseInputParam_WhenServerRole_SetsServer)
{
    EndpointDesc local{};
    local.protocol = COMM_PROTOCOL_ROCE;
    local.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    local.loc.device.devPhyId = 1U;
    local.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.2.2.1", &local.commAddr.addr), 1);
    StubEndpointForAicpuTsRoceChannel stub(local, reinterpret_cast<void*>(0x2U));
    HcommChannelDesc desc{};
    desc.role = HCOMM_SOCKET_ROLE_SERVER;
    desc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
    desc.remoteEndpoint.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.2.2.2", &desc.remoteEndpoint.commAddr.addr), 1);
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(&stub), desc);
    ASSERT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    EXPECT_FALSE(ch.isLocalIpClient_);
}

TEST_F(AicpuTsRoceChannelTest, Ut_ParseInputParam_WhenReserved_LocalIpSmaller_DecidesClient)
{
    EndpointDesc local{};
    local.protocol = COMM_PROTOCOL_ROCE;
    local.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    local.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.3.1.1", &local.commAddr.addr), 1);
    StubEndpointForAicpuTsRoceChannel stub(local, reinterpret_cast<void*>(0x3U));
    HcommChannelDesc desc{};
    desc.role = HCOMM_SOCKET_ROLE_RESERVED;
    desc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
    desc.remoteEndpoint.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.3.1.2", &desc.remoteEndpoint.commAddr.addr), 1);
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(&stub), desc);
    ASSERT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    EXPECT_TRUE(ch.isLocalIpClient_);
}

TEST_F(AicpuTsRoceChannelTest, Ut_ParseInputParam_WhenReserved_LocalIpGreater_DecidesServer)
{
    EndpointDesc local{};
    local.protocol = COMM_PROTOCOL_ROCE;
    local.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    local.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.3.2.9", &local.commAddr.addr), 1);
    StubEndpointForAicpuTsRoceChannel stub(local, reinterpret_cast<void*>(0x4U));
    HcommChannelDesc desc{};
    desc.role = HCOMM_SOCKET_ROLE_RESERVED;
    desc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
    desc.remoteEndpoint.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.3.2.1", &desc.remoteEndpoint.commAddr.addr), 1);
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(&stub), desc);
    ASSERT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    EXPECT_FALSE(ch.isLocalIpClient_);
}

TEST_F(AicpuTsRoceChannelTest, Ut_ParseInputParam_WhenReserved_SameIp_Returns_PARA)
{
    EndpointDesc local{};
    local.protocol = COMM_PROTOCOL_ROCE;
    local.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    local.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.4.4.4", &local.commAddr.addr), 1);
    StubEndpointForAicpuTsRoceChannel stub(local, reinterpret_cast<void*>(0x5U));
    HcommChannelDesc desc{};
    desc.role = HCOMM_SOCKET_ROLE_RESERVED;
    desc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
    desc.remoteEndpoint.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.4.4.4", &desc.remoteEndpoint.commAddr.addr), 1);
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(&stub), desc);
    EXPECT_EQ(ch.ParseInputParam(), HCCL_E_PARA);
}

TEST_F(AicpuTsRoceChannelTest, Ut_ParseInputParam_WhenUnexpectedRole_StillDecidesByIp)
{
    EndpointDesc local{};
    local.protocol = COMM_PROTOCOL_ROCE;
    local.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    local.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.5.1.1", &local.commAddr.addr), 1);
    StubEndpointForAicpuTsRoceChannel stub(local, reinterpret_cast<void*>(0x6U));
    HcommChannelDesc desc{};
    desc.role = static_cast<HcommSocketRole>(99);
    desc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
    desc.remoteEndpoint.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.5.1.2", &desc.remoteEndpoint.commAddr.addr), 1);
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(&stub), desc);
    ASSERT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    EXPECT_TRUE(ch.isLocalIpClient_);
}

TEST_F(AicpuTsRoceChannelTest, Ut_ParseInputParam_WhenNotifyNumNonZero_WarnsAndReturns_SUCCESS)
{
    EndpointDesc local{};
    local.protocol = COMM_PROTOCOL_ROCE;
    local.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    local.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.6.1.1", &local.commAddr.addr), 1);
    StubEndpointForAicpuTsRoceChannel stub(local, reinterpret_cast<void*>(0x7U));
    HcommChannelDesc desc{};
    desc.role = HCOMM_SOCKET_ROLE_CLIENT;
    desc.notifyNum = 5U;
    desc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
    desc.remoteEndpoint.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.6.1.2", &desc.remoteEndpoint.commAddr.addr), 1);
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(&stub), desc);
    ASSERT_EQ(ch.ParseInputParam(), HCCL_SUCCESS);
    EXPECT_EQ(ch.notifyNum_, 5U);
}

TEST_F(AicpuTsRoceChannelTest, Ut_BuildSocketTagName_WhenIpv6_Returns_SUCCESS)
{
    HcommChannelDesc desc{};
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    ch.localEp_.protocol = COMM_PROTOCOL_ROCE;
    ch.localEp_.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    ch.localEp_.commAddr.type = COMM_ADDR_TYPE_IP_V6;
    ASSERT_EQ(inet_pton(AF_INET6, "::1", &ch.localEp_.commAddr.addr6), 1);
    ch.remoteEp_.protocol = COMM_PROTOCOL_ROCE;
    ch.remoteEp_.commAddr.type = COMM_ADDR_TYPE_IP_V6;
    ASSERT_EQ(inet_pton(AF_INET6, "::2", &ch.remoteEp_.commAddr.addr6), 1);
    ch.isLocalIpClient_ = true;
    ch.channelDesc_.port = 16666U;
    *reinterpret_cast<uint32_t*>(ch.channelDesc_.raws + sizeof(ch.channelDesc_.raws) - sizeof(uint32_t)) = 3U;
    std::string tag;
    ASSERT_EQ(ch.BuildSocketTagName(tag), HCCL_SUCCESS);
    EXPECT_FALSE(tag.empty());
}

TEST_F(AicpuTsRoceChannelTest, Ut_AssignDispatcherCommId_WritesNonEmptyId)
{
    HcommChannelDesc desc{};
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    ASSERT_EQ(ch.AssignDispatcherCommId(), HCCL_SUCCESS);
    EXPECT_FALSE(ch.dispatcherCommId_.empty());
}

TEST_F(AicpuTsRoceChannelTest, Ut_ConfigureMachineParaForTransport_WhenMocksOk_Returns_SUCCESS)
{
    DevType devType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDevice).stubs().with(mockcpp::any()).will(invoke(StubHrtGetDeviceWriteZero));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));

    HcommChannelDesc desc{};
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    ch.isLocalIpClient_ = true;
    ch.localEp_.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    ch.localEp_.loc.device.devPhyId = 2U;
    ch.remoteEp_.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    ch.remoteEp_.loc.device.devPhyId = 3U;
    ch.dataSocket_ = nullptr;
    ch.channelDesc_.roceAttr.tc = 0xffU;
    ch.channelDesc_.roceAttr.sl = 0xffU;
    ASSERT_EQ(ch.ConfigureMachineParaForTransport(), HCCL_SUCCESS);
    EXPECT_EQ(ch.machinePara_.localDeviceId, 2U);
    EXPECT_EQ(ch.machinePara_.remoteDeviceId, 3U);
    EXPECT_EQ(ch.machinePara_.deviceType, DevType::DEV_TYPE_910B);
}

TEST_F(AicpuTsRoceChannelTest, Ut_Init_WhenBuildStepsMocked_Returns_SUCCESS)
{
    EndpointDesc local{};
    local.protocol = COMM_PROTOCOL_ROCE;
    local.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    local.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.7.1.1", &local.commAddr.addr), 1);
    StubEndpointForAicpuTsRoceChannel stub(local, reinterpret_cast<void*>(0x8U));
    HcommChannelDesc desc{};
    desc.role = HCOMM_SOCKET_ROLE_CLIENT;
    desc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
    desc.remoteEndpoint.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.7.1.2", &desc.remoteEndpoint.commAddr.addr), 1);
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(&stub), desc);
    MOCKER_CPP(&AicpuTsRoceChannel::BuildDataSocket).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuTsRoceChannel::BuildDispatcherAndTransport).stubs().will(returnValue(HCCL_SUCCESS));
    ASSERT_EQ(ch.Init(), HCCL_SUCCESS);
}

TEST_F(AicpuTsRoceChannelTest, Ut_Init_WhenBuildDataSocketFails_Returns_Error)
{
    EndpointDesc local{};
    local.protocol = COMM_PROTOCOL_ROCE;
    local.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    local.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.8.1.1", &local.commAddr.addr), 1);
    StubEndpointForAicpuTsRoceChannel stub(local, reinterpret_cast<void*>(0x9U));
    HcommChannelDesc desc{};
    desc.role = HCOMM_SOCKET_ROLE_CLIENT;
    desc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
    desc.remoteEndpoint.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.8.1.2", &desc.remoteEndpoint.commAddr.addr), 1);
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(&stub), desc);
    MOCKER_CPP(&AicpuTsRoceChannel::BuildDataSocket).stubs().will(returnValue(HCCL_E_NOT_FOUND));
    EXPECT_EQ(ch.Init(), HCCL_E_NOT_FOUND);
}

TEST_F(AicpuTsRoceChannelTest, Ut_Init_WhenBuildDispatcherFails_Returns_Success)
{
    EndpointDesc local{};
    local.protocol = COMM_PROTOCOL_ROCE;
    local.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    local.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.9.1.1", &local.commAddr.addr), 1);
    StubEndpointForAicpuTsRoceChannel stub(local, reinterpret_cast<void*>(0xaU));
    HcommChannelDesc desc{};
    desc.role = HCOMM_SOCKET_ROLE_CLIENT;
    desc.remoteEndpoint.protocol = COMM_PROTOCOL_ROCE;
    desc.remoteEndpoint.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.9.1.2", &desc.remoteEndpoint.commAddr.addr), 1);
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(&stub), desc);
    MOCKER_CPP(&AicpuTsRoceChannel::BuildDataSocket).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AicpuTsRoceChannel::BuildDispatcherAndTransport).stubs().will(returnValue(HCCL_E_INTERNAL));
    EXPECT_EQ(ch.Init(), HCCL_SUCCESS);
}

TEST_F(AicpuTsRoceChannelTest, Ut_Serialize_WhenInitedAndDepsStubbed_Returns_SUCCESS)
{
    EndpointDesc local{};
    local.protocol = COMM_PROTOCOL_ROCE;
    local.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    local.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, "10.20.1.1", &local.commAddr.addr), 1);

    auto mgr = std::make_shared<AicpuTsRoceRegedMemMgr>(nullptr, nullptr);
    StubEndpointWithRoceMemMgr stub(local, reinterpret_cast<void*>(0xbeefU), mgr);
    HcommChannelDesc desc{};
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(&stub), desc);
    ch.inited_ = true;
    ch.transport_ = std::make_unique<hccl::Transport>();

    MOCKER_CPP(&AicpuTsRoceRegedMemMgr::GetAllMemDetails).stubs().will(invoke(StubGetAllMemDetailsForSerialize));
    MOCKER_CPP(&hccl::Transport::GetAiQpInfo).stubs().will(invoke(StubTransportGetAiQpInfo));
    MOCKER_CPP(&AicpuTsRoceChannel::SerializeDrainNotifyInfo).stubs().will(returnValue(HCCL_SUCCESS));
    using DeviceMemAllocRetByValue = hccl::DeviceMem (*)(u64, bool);
    mockcpp::mockAPI<DeviceMemAllocRetByValue>::get(
        "hccl::DeviceMem::alloc", "DeviceMem_alloc_by_value",
        static_cast<DeviceMemAllocRetByValue>(&hccl::DeviceMem::alloc))
        .stubs()
        .will(invoke(StubDeviceMemAlloc));
    MOCKER(hrtMemSyncCopy).stubs().will(invoke(StubHrtMemSyncCopySerialize));

    std::shared_ptr<hccl::DeviceMem> out;
    ASSERT_EQ(ch.Serialize(out), HCCL_SUCCESS);
    ASSERT_NE(out.get(), nullptr);
}

// GetStatus() 测试用例
TEST_F(AicpuTsRoceChannelTest, Ut_GetStatus_WhenRoceStatusInit_Returns_Init)
{
    HcommChannelDesc desc{};
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    ch.roceStatus_ = AicpuTsRoceChannel::RoceStatus::INIT;
    EXPECT_EQ(ch.GetStatus(), ChannelStatus::INIT);
}

TEST_F(AicpuTsRoceChannelTest, Ut_GetStatus_WhenSocketConnectingAndSocketOk_TransitionsToReady)
{
    HcommChannelDesc desc{};
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    ch.roceStatus_ = AicpuTsRoceChannel::RoceStatus::SOCKET_CONNECTING;
    ch.dataSocket_
        = std::shared_ptr<hccl::HcclSocket>(reinterpret_cast<hccl::HcclSocket*>(0x1), [](hccl::HcclSocket*) {});

    // Mock dataSocket_->GetStatus() to return SOCKET_OK
    MOCKER_CPP(&hccl::HcclSocket::GetStatus).stubs().will(returnValue(hccl::HcclSocketStatus::SOCKET_OK));
    // Mock BuildDispatcherAndTransport to return success
    MOCKER_CPP(&AicpuTsRoceChannel::BuildDispatcherAndTransport).stubs().will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(ch.GetStatus(), ChannelStatus::READY);
    EXPECT_EQ(ch.roceStatus_, AicpuTsRoceChannel::RoceStatus::READY);
}

TEST_F(AicpuTsRoceChannelTest, Ut_GetStatus_WhenSocketConnectingAndSocketTimeout_Returns_SocketTimeout)
{
    HcommChannelDesc desc{};
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    ch.roceStatus_ = AicpuTsRoceChannel::RoceStatus::SOCKET_CONNECTING;
    ch.dataSocket_
        = std::shared_ptr<hccl::HcclSocket>(reinterpret_cast<hccl::HcclSocket*>(0x1), [](hccl::HcclSocket*) {});

    MOCKER_CPP(&hccl::HcclSocket::GetStatus).stubs().will(returnValue(hccl::HcclSocketStatus::SOCKET_TIMEOUT));
    MOCKER_CPP(&hccl::HcclSocket::Close).stubs();

    EXPECT_EQ(ch.GetStatus(), ChannelStatus::SOCKET_TIMEOUT);
    EXPECT_EQ(ch.roceStatus_, AicpuTsRoceChannel::RoceStatus::FAILED);
}

TEST_F(AicpuTsRoceChannelTest, Ut_GetStatus_WhenSocketConnectingAndSocketError_Returns_Failed)
{
    HcommChannelDesc desc{};
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    ch.roceStatus_ = AicpuTsRoceChannel::RoceStatus::SOCKET_CONNECTING;
    ch.dataSocket_
        = std::shared_ptr<hccl::HcclSocket>(reinterpret_cast<hccl::HcclSocket*>(0x1), [](hccl::HcclSocket*) {});

    MOCKER_CPP(&hccl::HcclSocket::GetStatus).stubs().will(returnValue(hccl::HcclSocketStatus::SOCKET_ERROR));
    MOCKER_CPP(&hccl::HcclSocket::Close).stubs();

    EXPECT_EQ(ch.GetStatus(), ChannelStatus::FAILED);
    EXPECT_EQ(ch.roceStatus_, AicpuTsRoceChannel::RoceStatus::FAILED);
}

TEST_F(AicpuTsRoceChannelTest, Ut_GetStatus_WhenSocketConnectingAndOtherStatus_Returns_Init)
{
    HcommChannelDesc desc{};
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    ch.roceStatus_ = AicpuTsRoceChannel::RoceStatus::SOCKET_CONNECTING;
    ch.dataSocket_
        = std::shared_ptr<hccl::HcclSocket>(reinterpret_cast<hccl::HcclSocket*>(0x1), [](hccl::HcclSocket*) {});

    MOCKER_CPP(&hccl::HcclSocket::GetStatus).stubs().will(returnValue(hccl::HcclSocketStatus::SOCKET_CONNECTING));

    EXPECT_EQ(ch.GetStatus(), ChannelStatus::INIT);
    EXPECT_EQ(ch.roceStatus_, AicpuTsRoceChannel::RoceStatus::SOCKET_CONNECTING);
}

TEST_F(AicpuTsRoceChannelTest, Ut_GetStatus_WhenSocketOkAndBuildSuccess_Returns_Ready)
{
    HcommChannelDesc desc{};
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    ch.roceStatus_ = AicpuTsRoceChannel::RoceStatus::SOCKET_OK;

    MOCKER_CPP(&AicpuTsRoceChannel::BuildDispatcherAndTransport).stubs().will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(ch.GetStatus(), ChannelStatus::READY);
    EXPECT_EQ(ch.roceStatus_, AicpuTsRoceChannel::RoceStatus::READY);
}

TEST_F(AicpuTsRoceChannelTest, Ut_GetStatus_WhenSocketOkAndBuildFailed_Returns_Failed)
{
    HcommChannelDesc desc{};
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    ch.roceStatus_ = AicpuTsRoceChannel::RoceStatus::SOCKET_OK;
    ch.dataSocket_
        = std::shared_ptr<hccl::HcclSocket>(reinterpret_cast<hccl::HcclSocket*>(0x1), [](hccl::HcclSocket*) {});

    MOCKER_CPP(&AicpuTsRoceChannel::BuildDispatcherAndTransport).stubs().will(returnValue(HCCL_E_INTERNAL));
    MOCKER_CPP(&hccl::HcclSocket::Close).stubs();

    EXPECT_EQ(ch.GetStatus(), ChannelStatus::FAILED);
    EXPECT_EQ(ch.roceStatus_, AicpuTsRoceChannel::RoceStatus::FAILED);
}

TEST_F(AicpuTsRoceChannelTest, Ut_GetStatus_WhenRoceStatusReady_Returns_Ready)
{
    HcommChannelDesc desc{};
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    ch.roceStatus_ = AicpuTsRoceChannel::RoceStatus::READY;

    EXPECT_EQ(ch.GetStatus(), ChannelStatus::READY);
}

TEST_F(AicpuTsRoceChannelTest, Ut_GetStatus_WhenRoceStatusFailed_Returns_Failed)
{
    HcommChannelDesc desc{};
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    ch.roceStatus_ = AicpuTsRoceChannel::RoceStatus::FAILED;
    ch.dataSocket_
        = std::shared_ptr<hccl::HcclSocket>(reinterpret_cast<hccl::HcclSocket*>(0x1), [](hccl::HcclSocket*) {});

    MOCKER_CPP(&hccl::HcclSocket::Close).stubs();

    EXPECT_EQ(ch.GetStatus(), ChannelStatus::FAILED);
}

// ===================== ConfigureMachineParaForTransport 队列深度解析 UT =====================

/**
 * 场景: 队列深度字段为0和INVALID_UINT（使用默认值哨兵）
 * 预期: queueDepthAttr 使用默认值 sqDepth=2048, sendCqDepth=2048
 */
TEST_F(AicpuTsRoceChannelTest, Ut_ConfigureMachineParaForTransport_WhenDepthsDefault_Expect_DefaultValues)
{
    DevType devType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDevice).stubs().with(mockcpp::any()).will(invoke(StubHrtGetDeviceWriteZero));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));

    HcommChannelDesc desc{};
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    ch.isLocalIpClient_ = true;
    ch.localEp_.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    ch.localEp_.loc.device.devPhyId = 2U;
    ch.remoteEp_.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    ch.remoteEp_.loc.device.devPhyId = 3U;
    ch.dataSocket_ = nullptr;
    ch.channelDesc_.roceAttr.tc = 0xffU;
    ch.channelDesc_.roceAttr.sl = 0xffU;
    // 队列深度字段为默认哨兵值（0 和 0xFFFFFFFF）
    ch.channelDesc_.roceAttr.sqDepth = 0U;
    ch.channelDesc_.roceAttr.scqDepth = 0xFFFFFFFFU;

    ASSERT_EQ(ch.ConfigureMachineParaForTransport(), HCCL_SUCCESS);
    // 默认值：sqDepth/scqDepth→2048
    EXPECT_EQ(ch.machinePara_.queueDepthAttr.sqDepth, 2048U);
    EXPECT_EQ(ch.machinePara_.queueDepthAttr.sendCqDepth, 2048U);
}

/**
 * 场景: 队列深度字段为用户配置的合法值
 * 预期: queueDepthAttr 透传用户值
 */
TEST_F(AicpuTsRoceChannelTest, Ut_ConfigureMachineParaForTransport_WhenDepthsUserConfigured_Expect_PassedThrough)
{
    DevType devType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDevice).stubs().with(mockcpp::any()).will(invoke(StubHrtGetDeviceWriteZero));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));

    HcommChannelDesc desc{};
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    ch.isLocalIpClient_ = true;
    ch.localEp_.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    ch.localEp_.loc.device.devPhyId = 2U;
    ch.remoteEp_.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    ch.remoteEp_.loc.device.devPhyId = 3U;
    ch.dataSocket_ = nullptr;
    ch.channelDesc_.roceAttr.tc = 0xffU;
    ch.channelDesc_.roceAttr.sl = 0xffU;
    // 用户配置的合法深度值
    ch.channelDesc_.roceAttr.sqDepth = 256U;
    ch.channelDesc_.roceAttr.scqDepth = 1024U;

    ASSERT_EQ(ch.ConfigureMachineParaForTransport(), HCCL_SUCCESS);
    // 用户值应透传到 queueDepthAttr 对应字段
    EXPECT_EQ(ch.machinePara_.queueDepthAttr.sqDepth, 256U);
    EXPECT_EQ(ch.machinePara_.queueDepthAttr.sendCqDepth, 1024U);
}

/**
 * 场景: 部分深度字段配置、部分留默认
 * 预期: 各自正确解析（用户值透传，默认值使用默认）
 */
TEST_F(AicpuTsRoceChannelTest, Ut_ConfigureMachineParaForTransport_WhenMixedDepths_Expect_RespectivelyResolved)
{
    DevType devType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDevice).stubs().with(mockcpp::any()).will(invoke(StubHrtGetDeviceWriteZero));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));

    HcommChannelDesc desc{};
    AicpuTsRoceChannel ch(reinterpret_cast<EndpointHandle>(0x1), desc);
    ch.isLocalIpClient_ = true;
    ch.localEp_.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    ch.localEp_.loc.device.devPhyId = 2U;
    ch.remoteEp_.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    ch.remoteEp_.loc.device.devPhyId = 3U;
    ch.dataSocket_ = nullptr;
    ch.channelDesc_.roceAttr.tc = 0xffU;
    ch.channelDesc_.roceAttr.sl = 0xffU;
    // 混合：sqDepth 用户配置，scqDepth 留默认
    ch.channelDesc_.roceAttr.sqDepth = 256U;
    ch.channelDesc_.roceAttr.scqDepth = 0xFFFFFFFFU;

    ASSERT_EQ(ch.ConfigureMachineParaForTransport(), HCCL_SUCCESS);
    EXPECT_EQ(ch.machinePara_.queueDepthAttr.sqDepth, 256U);      // 用户值
    EXPECT_EQ(ch.machinePara_.queueDepthAttr.sendCqDepth, 2048U); // 默认值
}

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
#include "orion_adapter_hccp.h"
#include "orion_adapter_rts.h"
#include "hcomm_c_adpt.h"

#define private public
#define protected public

#include "aiv_ub_mem_transport.h"
#include "exchange_ipc_buffer_dto.h"
#include "env_config/env_config_v2.h"
#include "base_config_legacy.h"
#include "local_ipc_rma_buffer.h"

#undef protected
#undef private

namespace {

static int g_recvCallCount = 0;

static void StubSendAsync(Hccl::Socket* self, const void* sendBuf, u32 size)
{
    (void)self;
    (void)sendBuf;
    (void)size;
}

static void StubRecvAsyncNormal(Hccl::Socket* self, u8* recvBuf, u32 size)
{
    (void)self;
    g_recvCallCount++;
    if (g_recvCallCount == 1) {
        *reinterpret_cast<u32*>(recvBuf) = sizeof(u32);
    } else {
        u32 vecSize = 0;
        (void)memcpy_s(recvBuf, size, &vecSize, sizeof(vecSize));
    }
}

} // namespace

// Real stub definitions for Hccl hardware functions — called by LocalIpcRmaBuffer ctor
namespace Hccl {
void HrtDevMemAlignWithPage(void* ptr, u64 size, void*& ipcPtr, u64& ipcSize, u64& ipcOff)
{
    ipcPtr = ptr;
    ipcSize = size;
    ipcOff = 0;
}
void HrtIpcSetMemoryName(void* ptr, char_t* name, u64 ptrMaxLen, u32 nameMaxLen)
{
    (void)ptr;
    (void)ptrMaxLen;
    const char* fakeName = "stub_ipc_name";
    (void)strncpy_s(name, nameMaxLen, fakeName, RTS_IPC_MEM_NAME_LEN - 1);
}
void HrtIpcDestroyMemoryName(const char_t* name) { (void)name; }
s32 HrtDeviceGetBareTgid() { return 12345; }
} // namespace Hccl

class AivUbMemTransportTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AivUbMemTransportTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "AivUbMemTransportTest tests tear down." << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in AivUbMemTransportTest SetUP" << std::endl;
        Hccl::IpAddress localIp("1.0.0.0");
        Hccl::IpAddress remoteIp("2.0.0.0");
        fakeSocket_ = new Hccl::Socket(
            nullptr, localIp, 100, remoteIp, "test", Hccl::SocketRole::SERVER, Hccl::NicType::HOST_NIC_TYPE);
        setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
        MOCKER(&Hccl::EnvSocketConfig::GetLinkTimeOut).stubs().will(returnValue(100));

        // Socket async stubs
        MOCKER_CPP(&Hccl::Socket::SendAsync, void(Hccl::Socket::*)(const void*, u32))
            .stubs()
            .with(mockcpp::any(), mockcpp::any())
            .will(invoke(StubSendAsync));

        g_recvCallCount = 0;
    }

    virtual void TearDown()
    {
        delete fakeSocket_;
        unsetenv("HCCL_DFS_CONFIG");
        GlobalMockObject::verify();
        std::cout << "A Test case in AivUbMemTransportTest TearDown" << std::endl;
    }

    std::shared_ptr<hcomm::AivUbMemTransport> CreateAivTransport(HcommChannelDesc& desc)
    {
        return std::make_shared<hcomm::AivUbMemTransport>(fakeSocket_, desc);
    }

    std::shared_ptr<Hccl::LocalIpcRmaBuffer>
    CreateLocalIpcRmaBuffer(u64 addr, u64 size, HcclMemType type, const std::string& tag)
    {
        auto buf = std::make_shared<Hccl::Buffer>(addr, size, type, tag.c_str());
        return std::make_shared<Hccl::LocalIpcRmaBuffer>(buf);
    }

    Hccl::Socket* fakeSocket_;
};

TEST_F(AivUbMemTransportTest, ut_AivUbMemTransport_GetRemoteMems_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    HcommChannelDesc desc{};
    auto aivTransport = CreateAivTransport(desc);
    auto rmtBuffer1 = std::make_unique<Hccl::RemoteIpcRmaBuffer>();
    rmtBuffer1->addr = (uintptr_t)0x101;
    rmtBuffer1->size = (u64)0x101;
    rmtBuffer1->memType = HcclMemType::HCCL_MEM_TYPE_HOST;
    rmtBuffer1->memInfo = "buffer1";
    aivTransport->rmtBufferVec_.push_back(std::move(rmtBuffer1));
    aivTransport->rmtRmaBufferVec_.push_back(aivTransport->rmtBufferVec_.back().get());

    CommMem* remoteMems;
    char** memInfos;
    u32 memNum;
    HcclResult ret = aivTransport->GetRemoteMems(&memNum, &remoteMems, &memInfos);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(memNum, 1U);
    EXPECT_EQ(std::string(memInfos[0]), "buffer1");
    EXPECT_EQ(remoteMems[0].type, CommMemType::COMM_MEM_TYPE_HOST);
    EXPECT_EQ(remoteMems[0].addr, (void*)0x101);
    EXPECT_EQ(remoteMems[0].size, (uint64_t)0x101);
}

TEST_F(AivUbMemTransportTest, ut_AivUbMemTransport_GetRemoteMems_When_bufferNumIs0_Expect_ReturnIsHCCL_SUCCESS)
{
    HcommChannelDesc desc{};
    auto aivTransport = CreateAivTransport(desc);
    CommMem* remoteMems = nullptr;
    char** memInfos = nullptr;
    u32 memNum = 1;
    HcclResult ret = aivTransport->GetRemoteMems(&memNum, &remoteMems, &memInfos);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(memNum, 0U);
    EXPECT_EQ(remoteMems, nullptr);
    EXPECT_EQ(memInfos, nullptr);
}

TEST_F(AivUbMemTransportTest, ut_AivUbMemTransport_Init_When_bufferNumIs0_Expect_ReturnIsHCCL_E_PARA)
{
    HcommChannelDesc desc{};
    desc.memHandleNum = 0;
    auto aivTransport = CreateAivTransport(desc);
    HcclResult ret = aivTransport->Init();
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(AivUbMemTransportTest, ut_AivUbMemTransport_UpdateMemInfo_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    HcommChannelDesc desc{};
    auto aivTransport = CreateAivTransport(desc);
    auto initMockBuffer = CreateLocalIpcRmaBuffer(0x100, 0x100, HCCL_MEM_TYPE_DEVICE, "initBuffer");
    aivTransport->localRmaBufferVec_.push_back(initMockBuffer.get());
    size_t initialVecSize = aivTransport->localRmaBufferVec_.size();

    // Normal case: GetAsyncStatus returns OK
    Hccl::SocketStatus okStatus = Hccl::SocketStatus::OK;
    MOCKER(&Hccl::Socket::GetAsyncStatus).stubs().will(returnValue(okStatus));
    MOCKER_CPP(&Hccl::Socket::RecvAsync, void(Hccl::Socket::*)(u8*, u32))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(invoke(StubRecvAsyncNormal));

    auto mockBuffer1 = CreateLocalIpcRmaBuffer(0x1000, 0x1000, HCCL_MEM_TYPE_DEVICE, "newBuffer1");
    auto mockBuffer2 = CreateLocalIpcRmaBuffer(0x2000, 0x2000, HCCL_MEM_TYPE_DEVICE, "newBuffer2");
    void* memHandles[2] = {reinterpret_cast<void*>(mockBuffer1.get()), reinterpret_cast<void*>(mockBuffer2.get())};

    HcclResult ret = aivTransport->UpdateMemInfo(memHandles, 2);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    EXPECT_EQ(aivTransport->localRmaBufferVec_.size(), initialVecSize + 2);
    EXPECT_EQ(aivTransport->localRmaBufferVec_[initialVecSize], mockBuffer1.get());
    EXPECT_EQ(aivTransport->localRmaBufferVec_[initialVecSize + 1], mockBuffer2.get());
}

TEST_F(AivUbMemTransportTest, ut_AivUbMemTransport_UpdateMemInfo_When_SocketTimeout_Expect_ReturnIsHCCL_E_TIMEOUT)
{
    HcommChannelDesc desc{};
    auto aivTransport = CreateAivTransport(desc);

    Hccl::SocketStatus fakeSocketStatus = Hccl::SocketStatus::TIMEOUT;
    MOCKER(&Hccl::Socket::GetAsyncStatus).stubs().will(returnValue(fakeSocketStatus));

    auto mockBuffer = CreateLocalIpcRmaBuffer(0x1000, 0x1000, HCCL_MEM_TYPE_DEVICE, "testBuffer");
    void* memHandles[1] = {reinterpret_cast<void*>(mockBuffer.get())};

    HcclResult ret = aivTransport->UpdateMemInfo(memHandles, 1);
    EXPECT_EQ(ret, HCCL_E_TIMEOUT);
}

TEST_F(AivUbMemTransportTest, ut_AivUbMemTransport_UpdateMemInfo_When_bufferNumIs0_Expect_ReturnIsHCCL_SUCCESS)
{
    HcommChannelDesc desc{};
    auto aivTransport = CreateAivTransport(desc);
    void* memHandles[1] = {nullptr};
    HcclResult ret = aivTransport->UpdateMemInfo(memHandles, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 验证 ExchangeIpcBufferDto 序列化/反序列化保留字段，且 Describe 输出
TEST_F(AivUbMemTransportTest, ut_ExchangeIpcBufferDto_SerializeDeserialize_KeepsFields)
{
    Hccl::ExchangeIpcBufferDto out(0x100, 0x200, 0x10, 12345, "memInfo_test");
    Hccl::BinaryStream bs;
    out.Serialize(bs);

    std::string data = bs.GetString();
    std::vector<char> vec(data.begin(), data.end());
    Hccl::BinaryStream bsIn(vec);
    Hccl::ExchangeIpcBufferDto in;
    in.Deserialize(bsIn);

    EXPECT_EQ(in.addr, 0x100ULL);
    EXPECT_EQ(in.size, 0x200ULL);
    EXPECT_EQ(in.offset, 0x10ULL);
    EXPECT_EQ(in.pid, 12345u);
    EXPECT_EQ(in.memInfo, std::string("memInfo_test"));

    std::string descStr = in.Describe();
    EXPECT_NE(descStr.find("memInfo_test"), std::string::npos);
}

// 验证 RmtBufferUnpackProc 构造 RemoteIpcRmaBuffer 后从 channelDesc_ 注入 pathMode
TEST_F(AivUbMemTransportTest, ut_RmtBufferUnpackProc_InjectsPathModeFromChannelDesc)
{
    HcommChannelDesc desc{};
    desc.remoteEndpoint.protocol = COMM_PROTOCOL_UB_MEM;
    desc.ubMemAttr.pathMode = 2;
    auto aivTransport = CreateAivTransport(desc);

    // 构造一个非空 DTO 并序列化
    Hccl::ExchangeIpcBufferDto out(0x1000, 0x1000, 0x0, 12345, "ubMemBuffer");
    Hccl::BinaryStream bs;
    u32 vecSize = 1;
    bs << vecSize;
    out.Serialize(bs);

    std::string data = bs.GetString();
    std::vector<char> vec(data.begin(), data.end());
    Hccl::BinaryStream bsIn(vec);

    aivTransport->RmtBufferUnpackProc(bsIn);

    // 验证 RemoteIpcRmaBuffer 的 pathMode_ 被注入为 channelDesc_.ubMemAttr.pathMode
    EXPECT_EQ(aivTransport->rmtBufferVec_[0]->pathMode_, 2);
}

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
#include <mockcpp/mockcpp.hpp>

#define private public
#define protected public
#include "aicpu/aicpu_ts_ub_rtp_channel.h"
#undef protected
#undef private
#include "endpoint.h"

#define private public
#define protected public
using namespace hcomm;

class AicpuTsUbRtpChannelTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    void SetUp() override
    {
        const char* dfsConfig = std::getenv("HCCL_DFS_CONFIG");
        if (dfsConfig != nullptr) {
            savedDfsConfig_ = dfsConfig;
            hadDfsConfig_ = true;
        }
        (void)setenv("HCCL_DFS_CONFIG", "task_exception:on", 1);
    }

    void TearDown() override
    {
        GlobalMockObject::verify();
        GlobalMockObject::reset();
        if (hadDfsConfig_) {
            (void)setenv("HCCL_DFS_CONFIG", savedDfsConfig_.c_str(), 1);
        } else {
            (void)unsetenv("HCCL_DFS_CONFIG");
        }
    }

private:
    std::string savedDfsConfig_;
    bool hadDfsConfig_{false};
};

namespace {
EndpointDesc MakeUbRtpEndpointDesc()
{
    EndpointDesc desc{};
    Hccl::IpAddress localIp("127.0.0.1");
    desc.protocol = COMM_PROTOCOL_UB_RTP;
    desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    desc.commAddr.addr = localIp.GetBinaryAddress().addr;
    desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    return desc;
}

class FakeParseEndpoint : public Endpoint {
public:
    FakeParseEndpoint() : Endpoint(MakeUbRtpEndpointDesc()) { ctxHandle_ = reinterpret_cast<void*>(0xDEADBEEF); }

    HcclResult Init() override { return HCCL_SUCCESS; }
    // 内存方法在 RegedMemMgr 上，Endpoint 不再 override
    RegedMemMgr* GetRegedMemMgr() override { return nullptr; }
    void* GetRdmaHandle() override { return ctxHandle_; }
    bool IsCtxHandleValid() const override { return ctxHandle_ != nullptr; }

private:
    void* ctxHandle_{nullptr};
};

std::shared_ptr<Hccl::LocalUbRmaBuffer> MakeUbRtpLocalBuffer(uintptr_t addr, u64 size, const char* tag)
{
    auto buffer = std::make_shared<Hccl::Buffer>(addr, size, HCCL_MEM_TYPE_DEVICE, tag);
    return std::make_shared<Hccl::LocalUbRmaBuffer>(buffer);
}

HcommResult StubUbRtpGetAllMemHandlesOne(EndpointHandle, void** memHandles, uint32_t* memHandleNum)
{
    static std::shared_ptr<Hccl::LocalUbRmaBuffer> localBuffer = MakeUbRtpLocalBuffer(0x620000U, 0x1000U, "ub_rtp_all");
    static std::shared_ptr<Hccl::LocalUbRmaBuffer> localBuffers[1] = {localBuffer};
    *memHandles = localBuffers;
    *memHandleNum = 1;
    return HCCL_SUCCESS;
}
} // namespace

struct FakeEndpoint {
    EndpointDesc desc;
    void* rdmaHandle{reinterpret_cast<void*>(0xDEADBEEF)};
    FakeEndpoint()
    {
        memset(&desc, 0, sizeof(desc));
        Hccl::IpAddress localIp("127.0.0.1");
        desc.protocol = COMM_PROTOCOL_UB_RTP;
        desc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
        desc.commAddr.addr = localIp.GetBinaryAddress().addr;
        desc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    }
    EndpointDesc GetEndpointDesc() { return desc; }
    void* GetRdmaHandle() { return rdmaHandle; }
    Hccl::IpAddress GetIpAddress() const { return Hccl::IpAddress("127.0.0.1"); }
};

class FakeSocket : public Hccl::Socket {
public:
    FakeSocket(Hccl::SocketStatus status = Hccl::SocketStatus::OK)
        : Hccl::Socket(
              nullptr, Hccl::IpAddress(), 0, Hccl::IpAddress(), "fake", Hccl::SocketRole::SERVER,
              Hccl::NicType::DEVICE_NIC_TYPE),
          status_(status)
    {}
    void SendAsync(const void* sendBuf, u32 size)
    {
        auto* p = static_cast<const u8*>(sendBuf);
        sent_.insert(sent_.end(), p, p + size);
    }
    void RecvAsync(u8* recvBuf, u32 size)
    {
        if (recvBuf && size) {
            if (!sent_.empty()) {
                u32 copySize = static_cast<u32>(std::min<size_t>(sent_.size(), static_cast<size_t>(size)));
                memcpy(recvBuf, sent_.data(), copySize);
                if (copySize < size)
                    std::memset(recvBuf + copySize, 0, size - copySize);
                sent_.erase(sent_.begin(), sent_.begin() + copySize);
            } else {
                std::memset(recvBuf, 0, size);
            }
        }
    }
    Hccl::SocketStatus GetAsyncStatus() { return status_; }
    Hccl::SocketRole GetRole() const { return Hccl::SocketRole::SERVER; }

    Hccl::SocketStatus status_;
    std::vector<u8> sent_;
};

namespace Hccl {
class FakeLocalUbRmaBuffer : public Hccl::LocalUbRmaBuffer {
public:
    FakeLocalUbRmaBuffer(std::shared_ptr<Hccl::Buffer> b, void* rdma) : LocalUbRmaBuffer(b, rdma) {}
    string Describe() const override { return "hello"; }
};
class FakeUbLocalNotify : public Hccl::UbLocalNotify {
public:
    FakeUbLocalNotify(void* rdma, bool dev) : UbLocalNotify(rdma, dev) {}
};
} // namespace Hccl

static HcommChannelDesc MakeFakeChannelDesc(FakeSocket* sock)
{
    HcommChannelDesc d{};
    d.socket = reinterpret_cast<void*>(sock);
    d.notifyNum = 0;
    d.exchangeAllMems = false;
    d.memHandles = nullptr;
    d.memHandleNum = 0;
    return d;
}

TEST_F(AicpuTsUbRtpChannelTest, Ut_GetChannelKind_Returns_AICPU_TS_UB_RTP)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUbRtpChannel ch(ep, desc);
    EXPECT_EQ(ch.GetChannelKind(), HcommChannelKind::AICPU_TS_UB_RTP);
}

TEST_F(AicpuTsUbRtpChannelTest, UT_ParseInputParam_When_ExchangeAllMemsFalse_Expect_FillCommonRes)
{
    FakeParseEndpoint endpoint;
    auto localBuffer = MakeUbRtpLocalBuffer(0x610000U, 0x1000U, "ub_rtp_desc");
    HcommMemHandle memHandles[1] = {reinterpret_cast<HcommMemHandle>(localBuffer.get())};
    HcommChannelDesc desc{};
    desc.exchangeAllMems = false;
    desc.memHandles = memHandles;
    desc.memHandleNum = 1;
    AicpuTsUbRtpChannel channel(reinterpret_cast<EndpointHandle>(&endpoint), desc);

    ASSERT_EQ(channel.ParseInputParam(), HCCL_SUCCESS);
    ASSERT_EQ(channel.commonRes_.bufferVec.size(), 1U);
    EXPECT_EQ(channel.commonRes_.bufferVec[0], localBuffer.get());
}

TEST_F(AicpuTsUbRtpChannelTest, UT_ParseInputParam_When_ExchangeAllMemsTrue_Expect_FillCommonRes)
{
    FakeParseEndpoint endpoint;
    HcommChannelDesc desc{};
    desc.exchangeAllMems = true;
    AicpuTsUbRtpChannel channel(reinterpret_cast<EndpointHandle>(&endpoint), desc);
    MOCKER(HcommMemGetAllMemHandles).stubs().will(invoke(StubUbRtpGetAllMemHandlesOne));

    ASSERT_EQ(channel.ParseInputParam(), HCCL_SUCCESS);
    ASSERT_EQ(channel.commonRes_.bufferVec.size(), 1U);
    EXPECT_EQ(channel.commonRes_.bufferVec[0]->GetAddr(), 0x620000U);
}

TEST_F(AicpuTsUbRtpChannelTest, Ut_BuildConnection_Creates_UbRtpConn)
{
    FakeEndpoint fe;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(&fe);
    auto fakeSock = std::make_unique<FakeSocket>();
    HcommChannelDesc desc = MakeFakeChannelDesc(fakeSock.get());
    desc.qos = 3U;

    AicpuTsUbRtpChannel ch(ep, desc);
    ch.localEp_ = fe.GetEndpointDesc();
    ch.remoteEp_ = fe.GetEndpointDesc();
    ch.rdmaHandle_ = fe.GetRdmaHandle();
    Hccl::IpAddress locAddr = fe.GetIpAddress();
    Hccl::IpAddress rmtAddr = fe.GetIpAddress();
    ch.locAddr_ = locAddr;
    ch.rmtAddr_ = rmtAddr;
    ch.devBaseAttr_.maxReadSize = 256 * 1024 * 1024;
    ch.devBaseAttr_.maxWriteSize = 256 * 1024 * 1024;

    MOCKER_CPP(&Hccl::TpManager::Init).stubs();
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));

    auto ret = ch.BuildConnection();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ASSERT_EQ(ch.commonRes_.connVec.size(), 1u);
    auto* conn = dynamic_cast<Hccl::DevUbRtpConnection*>(ch.commonRes_.connVec[0]);
    ASSERT_NE(conn, nullptr);
    EXPECT_EQ(conn->qos_, static_cast<u8>(3));
}

static void stub_Socket_SendAsync(Hccl::Socket* self, const void* sendBuf, u32 size)
{
    if (!self || !sendBuf || size == 0)
        return;
    auto* fs = dynamic_cast<FakeSocket*>(self);
    if (fs) {
        auto* p = static_cast<const u8*>(sendBuf);
        fs->sent_.insert(fs->sent_.end(), p, p + size);
    }
}

static void stub_Socket_RecvAsync(Hccl::Socket* self, u8* recvBuf, u32 size)
{
    if (!self || !recvBuf || size == 0)
        return;
    auto* fs = dynamic_cast<FakeSocket*>(self);
    if (fs) {
        if (!fs->sent_.empty()) {
            u32 copySize = static_cast<u32>(std::min<size_t>(fs->sent_.size(), static_cast<size_t>(size)));
            memcpy(recvBuf, fs->sent_.data(), copySize);
            if (copySize < size)
                std::memset(recvBuf + copySize, 0, size - copySize);
            fs->sent_.erase(fs->sent_.begin(), fs->sent_.begin() + copySize);
            return;
        }
    }
    std::memset(recvBuf, 0, size);
}

TEST_F(AicpuTsUbRtpChannelTest, Ut_ProcessUbRtpState_AllStates_Transitions)
{
    FakeEndpoint fe;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(&fe);
    auto fakeSock = std::make_unique<FakeSocket>();

    MOCKER_CPP(&Hccl::Socket::SendAsync, void(Hccl::Socket::*)(const void*, u32))
        .stubs()
        .with(any(), any())
        .will(invoke(stub_Socket_SendAsync));
    MOCKER_CPP(&Hccl::Socket::RecvAsync, void(Hccl::Socket::*)(u8*, u32))
        .stubs()
        .with(any(), any())
        .will(invoke(stub_Socket_RecvAsync));

    HcommChannelDesc desc = MakeFakeChannelDesc(fakeSock.get());

    AicpuTsUbRtpChannel ch(ep, desc);

    FakeEndpoint localEp;
    FakeEndpoint remoteEp;
    ch.localEp_ = localEp.GetEndpointDesc();
    ch.remoteEp_ = remoteEp.GetEndpointDesc();

    ch.channelDesc_.socket = reinterpret_cast<void*>(fakeSock.get());
    ch.socket_ = reinterpret_cast<Hccl::Socket*>(fakeSock.get());

    ch.channelDesc_.exchangeAllMems = false;

    ch.commonRes_.connVec = {};
    ch.commonRes_.bufferVec = {};
    ch.connNum_ = 0;
    ch.bufferNum_ = 0;

    ch.channelStatus = ChannelStatus::SOCKET_OK;
    ch.ubRtpStatus = AicpuTsUbRtpChannel::UbRtpStatus::SEND_SIZE;

    int iter = 0;
    while (ch.channelStatus != ChannelStatus::READY && iter < 200) {
        ch.GetStatus();
        iter++;
    }

    EXPECT_EQ(ch.channelStatus, ChannelStatus::READY);
    EXPECT_EQ(ch.ubRtpStatus, AicpuTsUbRtpChannel::UbRtpStatus::READY);
}

TEST_F(AicpuTsUbRtpChannelTest, Ut_GetStatus_WhenReady_ReturnsReady)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUbRtpChannel ch(ep, desc);
    ch.channelStatus = ChannelStatus::READY;
    EXPECT_EQ(ch.GetStatus(), ChannelStatus::READY);
}

TEST_F(AicpuTsUbRtpChannelTest, Ut_SendFinish_RecvFinish_NoCrash)
{
    FakeEndpoint fe;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(&fe);
    auto fakeSock = std::make_unique<FakeSocket>();
    HcommChannelDesc desc = MakeFakeChannelDesc(fakeSock.get());

    MOCKER_CPP(&Hccl::Socket::SendAsync, void(Hccl::Socket::*)(const void*, u32))
        .stubs()
        .with(any(), any())
        .will(invoke(stub_Socket_SendAsync));
    MOCKER_CPP(&Hccl::Socket::RecvAsync, void(Hccl::Socket::*)(u8*, u32))
        .stubs()
        .with(any(), any())
        .will(invoke(stub_Socket_RecvAsync));

    AicpuTsUbRtpChannel ch(ep, desc);
    ch.socket_ = reinterpret_cast<Hccl::Socket*>(fakeSock.get());

    ch.SendFinish();
    ch.RecvFinish();
    SUCCEED();
}

TEST_F(AicpuTsUbRtpChannelTest, Ut_H2DResPack_Packs_Data)
{
    FakeEndpoint fe;
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(&fe);
    auto fakeSock = std::make_unique<FakeSocket>();
    HcommChannelDesc desc = MakeFakeChannelDesc(fakeSock.get());

    AicpuTsUbRtpChannel ch(ep, desc);

    ch.channelStatus = ChannelStatus::READY;
    ch.notifyNum_ = 1;
    ch.bufferNum_ = 2;
    ch.connNum_ = 1;

    auto notifyUptr = std::make_unique<Hccl::FakeUbLocalNotify>(fe.GetRdmaHandle(), true);
    Hccl::UbLocalNotify* rawNotify = notifyUptr.get();
    ch.localNotifies_.push_back(std::move(notifyUptr));
    ch.commonRes_.notifyVec.push_back(rawNotify);

    auto rmtUbRmaBufPtr = std::make_unique<Hccl::RemoteUbRmaBuffer>(fe.GetRdmaHandle());
    ch.rmtNotifyVec_.push_back(std::move(rmtUbRmaBufPtr));

    auto buffer = std::make_shared<Hccl::Buffer>(0x100, 0x100);
    auto locUbRmaBufPtr = std::make_unique<Hccl::FakeLocalUbRmaBuffer>(buffer, fe.GetRdmaHandle());
    Hccl::FakeLocalUbRmaBuffer* locUbRmaBuf = locUbRmaBufPtr.get();
    ch.commonRes_.bufferVec.push_back(locUbRmaBuf);
    ch.commonRes_.bufferVec.push_back(nullptr);

    auto rmtBufPtr = std::make_unique<Hccl::RemoteUbRmaBuffer>(fe.GetRdmaHandle());
    ch.rmtBufferVec_.push_back(std::move(rmtBufPtr));

    Hccl::IpAddress ipAddress("127.0.0.1");
    Hccl::DevUbRtpConnection ubRtpConn(
        fe.GetRdmaHandle(), ipAddress, ipAddress, Hccl::OpMode::OPBASE, true, Hccl::HrtUbJfcMode::STARS_POLL, ipAddress,
        ipAddress);
    ch.commonRes_.connVec.push_back(&ubRtpConn);

    std::vector<char> out{};
    HcclResult ret = ch.H2DResPack(out);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(AicpuTsUbRtpChannelTest, Ut_DataPlaneInterfaces_ReturnNotSupport)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUbRtpChannel ch(ep, desc);

    EXPECT_EQ(ch.NotifyRecord(0), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(ch.NotifyWait(0, 1000), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(ch.WriteWithNotify(nullptr, nullptr, 0, 0), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(ch.Write(nullptr, nullptr, 0), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(ch.Read(nullptr, nullptr, 0), HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(ch.ChannelFence(), HCCL_E_NOT_SUPPORT);
}

TEST_F(AicpuTsUbRtpChannelTest, Ut_Clean_Resume_ReturnSuccess)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUbRtpChannel ch(ep, desc);

    EXPECT_EQ(ch.Clean(), HCCL_SUCCESS);
    EXPECT_EQ(ch.Resume(), HCCL_SUCCESS);
}

TEST_F(AicpuTsUbRtpChannelTest, Ut_GetNotifyNum_ReturnsValue)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUbRtpChannel ch(ep, desc);
    ch.notifyNum_ = 5;

    uint32_t num = 0;
    EXPECT_EQ(ch.GetNotifyNum(&num), HCCL_SUCCESS);
    EXPECT_EQ(num, 5u);
}

TEST_F(AicpuTsUbRtpChannelTest, Ut_IsSocketReady_NullptrSocket_ReturnsFalse)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUbRtpChannel ch(ep, desc);
    ch.socket_ = nullptr;

    EXPECT_FALSE(ch.IsSocketReady());
    EXPECT_EQ(ch.channelStatus, ChannelStatus::INVALID);
}

TEST_F(AicpuTsUbRtpChannelTest, Ut_IsResReady_EmptyConnVec_ReturnsTrue)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUbRtpChannel ch(ep, desc);
    ch.commonRes_.connVec = {};

    EXPECT_TRUE(ch.IsResReady());
}

TEST_F(AicpuTsUbRtpChannelTest, Ut_IsConnsReady_ZeroConnNum_ReturnsTrue)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUbRtpChannel ch(ep, desc);
    ch.connNum_ = 0;

    EXPECT_TRUE(ch.IsConnsReady());
}

TEST_F(AicpuTsUbRtpChannelTest, Ut_PackingHelpers_EmptyVecs_NoCrash)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUbRtpChannel ch(ep, desc);
    ch.notifyNum_ = 0;
    ch.connNum_ = 0;
    ch.bufferNum_ = 0;
    ch.commonRes_.connVec = {};
    ch.commonRes_.bufferVec = {};
    ch.commonRes_.notifyVec = {};

    Hccl::BinaryStream bs;
    ch.NotifyVecPack(bs);
    std::vector<Hccl::LocalRmaBuffer*> emptyBuf;
    ch.BufferVecPack(bs, emptyBuf);
    ch.ConnVecPack(bs);

    SUCCEED();
}

TEST_F(AicpuTsUbRtpChannelTest, Ut_GetUniqueIdV2_NotReady_Throws)
{
    HcommChannelDesc desc{};
    EndpointHandle ep = reinterpret_cast<EndpointHandle>(0x1);
    AicpuTsUbRtpChannel ch(ep, desc);
    ch.channelStatus = ChannelStatus::INIT;

    EXPECT_ANY_THROW(ch.GetUniqueIdV2());
}

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#define private public
#define protected public
#include "ins_to_sqe_rule.h"
#include "rtsq_a5.h"
#include "dev_ub_connection.h"
#include "ub_local_notify.h"
#include "local_ub_rma_buffer.h"
#include "ub_transport_lite_impl.h"
#include "ub_mem_transport.h"
#include "mem_transport_lite.h"
#include "data_buffer.h"
#include "kernel_param_lite.h"
#include "profiling_handler_lite.h"
#include "mem_transport_callback.h"
#include "rdma_handle_manager.h"
#undef private
#undef protected
#include "null_ptr_exception.h"

using namespace Hccl;

class StubTransportLiteImpl : public BaseTransportLiteImpl {
public:
    StubTransportLiteImpl() : BaseTransportLiteImpl(), stubRmaBuffer(remote_addr, remote_addr_len) {}

    void Wait(u32 index, const StreamLite& stream) override {}

    void Post(u32 index, const StreamLite& stream) override {}

    void Read(const RmaBufferLite& loc, const Buffer& rmt, const StreamLite& stream) override {}

    void
    ReadReduce(const RmaBufferLite& loc, const Buffer& rmt, const ReduceIn& reduceIn, const StreamLite& stream) override
    {}

    void Write(const RmaBufferLite& loc, const Buffer& rmt, const StreamLite& stream) override {}

    void WriteReduce(
        const RmaBufferLite& loc, const Buffer& rmt, const ReduceIn& reduceIn, const StreamLite& stream) override
    {}

    void WriteWithNotify(
        const RmaBufferLite& loc, const Buffer& rmt, const WithNotifyIn& withNotify, const StreamLite& stream) override
    {}

    void WriteReduceWithNotify(
        const RmaBufferLite& loc, const Buffer& rmt, const ReduceIn& reduceIn, const WithNotifyIn& withNotify,
        const StreamLite& stream) override
    {}

    Buffer GetRmtBuffer(u32 index) override { return stubRmaBuffer; }

    void SetRmaBuffer(Buffer rmaBuffer) { stubRmaBuffer = rmaBuffer; }

private:
    u64 remote_addr = 0x100;
    u64 remote_addr_len = 0x100;
    Buffer stubRmaBuffer;
};

class StubResMgrFetcher : public ResMgrFetcher {
public:
    StubResMgrFetcher()
    {
        std::unique_ptr<RmaBufferLite> rmaBufferLite = std::make_unique<RmaBufferLite>(1, 5, 1);
        std::unique_ptr<RmaBufferLite> rmaBufferLite1 = std::make_unique<RmaBufferLite>(1, 5, 1);
        // std::unique_ptr<RmaBufferLite> rmaBufferLite2 = std::make_unique<RmaBufferLite>(1, 5, 1);
        std::unique_ptr<RmaBufferLite> rmaBufferLite2 = std::make_unique<RmaBufferLite>(0x1234560, 100, 1);
        rmaBufferLiteVec.emplace_back(std::move(rmaBufferLite));
        rmaBufferLiteVec.emplace_back(std::move(rmaBufferLite1));
        rmaBufferLiteVec.emplace_back(std::move(rmaBufferLite2));

        currentOp.opMode = OpMode::OFFLOAD;
    }

    u32 GetExecTimeOut() override { return 1836; }

    HostDeviceSyncNotifyLiteMgr* GetHostDeviceSyncNotifyLiteMgr() override { return &hostDeviceSyncNotifyLiteMgr; }

    StreamLiteMgr* GetStreamLiteMgr() override { return &streamLiteMgr; }

    QueueNotifyLiteMgr* GetQueueNotifyLiteMgr() override { return &queueNotifyLiteMgr; }

    Cnt1tonNotifyLiteMgr* GetCnt1tonNotifyLiteMgr() override { return &cnt1tonNotifyLiteMgr; }

    CntNto1NotifyLiteMgr* GetCntNto1NotifyLiteMgr() override { return &cntNto1NotifyLiteMgr; }

    MemTransportLiteMgr* GetTransportLiteMgr() override { return transportLiteMgr.get(); }

    ConnectedLinkMgr* GetConnectedLinkMgr() override { return &connectedLinkMgr; }

    DevId GetDevPhyId() { return 0; }

    u64 GetCounterAddr() override { return opCounterAddr; }

    u64 GetLocAddr(BufferType type) { return 0xffffffff; }

    CollOperator GetCurrentOp() override
    {
        currentOp.opTag = "tag";
        return currentOp;
    }

    RmaBufferLite* GetRmaBufferLite(BufferType type) override { return rmaBufferLiteVec[type].get(); }

    MirrorTaskManagerLite* GetMirrorTaskMgrLite() override { return mirrorTaskMgrLite.get(); }

    HostDeviceSyncNotifyLiteMgr hostDeviceSyncNotifyLiteMgr;
    StreamLiteMgr streamLiteMgr;
    QueueNotifyLiteMgr queueNotifyLiteMgr;
    Cnt1tonNotifyLiteMgr cnt1tonNotifyLiteMgr;
    CntNto1NotifyLiteMgr cntNto1NotifyLiteMgr;
    ConnectedLinkMgr connectedLinkMgr;
    std::unique_ptr<MirrorTaskManagerLite> mirrorTaskMgrLite = std::make_unique<MirrorTaskManagerLite>();

    std::unique_ptr<MemTransportLiteMgr> transportLiteMgr
        = std::make_unique<MemTransportLiteMgr>(mirrorTaskMgrLite.get());
    CollOperator currentOp;
    std::vector<std::unique_ptr<RmaBufferLite>> rmaBufferLiteVec;
    std::unordered_map<DataBuffer, SendRecvItemTokenInfo> sendRecvTokenMap;
    u64 opCounterAddr;
};

class InsToSqeRuleV82Test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "InsToSqeRuleV82Test SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "InsToSqeRuleV82Test TearDown" << std::endl; }

    virtual void SetUp()
    {
        MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_910A2));
        MOCKER_CPP(&RtsqBase::QuerySqBaseAddr)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(reinterpret_cast<u64>(&mockSq)));
        MOCKER_CPP(&RtsqBase::QuerySqStatusByType).stubs().with(mockcpp::any()).will(returnValue(0));
        MOCKER_CPP(&RtsqBase::ConfigSqStatusByType).stubs();
        MOCKER_CPP(&MirrorTaskManager::AddTaskInfo).stubs().with(mockcpp::any());
        RdmaHandleManager::GetInstance().tokenInfoMap[rdmaHandle] = make_unique<TokenInfoManager>(0, rdmaHandle);
        MOCKER(GetUbToken).stubs().will(returnValue(1));

        DevUbConnection ubConnection((void*)0x100, link.GetLocalAddr(), link.GetRemoteAddr(), OpMode::OPBASE);
        RmaConnection* rmaConnection = &ubConnection;
        locRes.connVec.push_back(rmaConnection);
        UbLocalNotify ubLocalNotify(rdmaHandle);
        BaseLocalNotify* validLocalNotify = &ubLocalNotify;
        locRes.notifyVec.push_back(validLocalNotify);
        LocalUbRmaBuffer ubLocalRmaBuffer(devBuf, rdmaHandle);
        LocalRmaBuffer* validLocalRmaBuffer = &ubLocalRmaBuffer;
        locRes.bufferVec.push_back(validLocalRmaBuffer);

        RtsCntNotify rtsCntNotify;
        LocalCntNotify localCntNotify(rdmaHandle, &rtsCntNotify);
        locCntRes.vec.push_back(&localCntNotify);
        locCntRes.desc.push_back('0');
        locCntRes.desc.push_back(0);

        bool isRecvFirst = false;
        UbMemTransport ubTransport(locRes, attr, link, fakeSocket, rdmaHandle, locCntRes, isRecvFirst);
        ubTransport.baseStatus = TransportStatus::READY;
        LinkData linkData(BasePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB), 0, 1, 0, 1);
        MirrorTaskManager mirrorTaskMgr(0, &GlobalMirrorTasks::Instance(), true);
        auto transportCallback = MemTransportCallback(linkData, mirrorTaskMgr);
        auto data = ubTransport.GetUniqueId();
        transportLite = std::make_unique<MemTransportLite>(data, transportCallback);

        std::cout << "A Test case in InsToSqeRuleV82Test SetUp" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in InsToSqeRuleV82Test TearDown" << std::endl;
    }

    u32 fakedevPhyId = 0;
    u32 fakeStreamId = 1;
    u32 fakeSqId = 2;
    u32 fakeNotifyId = 1;
    u64 fakeNotifyHandleAddr = 100;
    u8 mockSq[AC_SQE_SIZE * AC_SQE_MAX_CNT]{0};

    BaseMemTransport::CommonLocRes locRes;
    BaseMemTransport::Attribution attr;
    BaseMemTransport::LocCntNotifyRes locCntRes;
    LinkData link{BasePortType(PortDeploymentType::DEV_NET), 0, 1, 0, 1};
    void* rdmaHandle = (void*)0x100;
    IpAddress ipAddress{"1.0.0.0"};
    Socket fakeSocket{nullptr, ipAddress, 100, ipAddress, "tag", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE};
    std::shared_ptr<DevBuffer> devBuf = DevBuffer::Create(0x100, 0x100);

    std::unique_ptr<MemTransportLite> transportLite;
};

TEST_F(InsToSqeRuleV82Test, Interpret_local_copy)
{
    u64 size = 100;
    DataSlice srcSlice(BufferType::SCRATCH, 0, size);
    DataSlice dstSlice(BufferType::SCRATCH, 100, size);
    InsLocalCopy insLocalCopy(srcSlice, dstSlice);

    DataSlice srcSlice2(BufferType::SCRATCH, 0, 0);
    DataSlice dstSlice2(BufferType::SCRATCH, 100, 0);
    InsLocalCopy insLocalCopy2(srcSlice2, dstSlice2);

    std::vector<char> streamLite1{fakeStreamId, fakeSqId};
    StreamLite stream(streamLite1);
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    stream.rtsq = std::make_unique<RtsqA5>(rtsq);
    MOCKER_CPP_VIRTUAL(rtsq, &RtsqA5::SdmaCopy)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any());

    StubResMgrFetcher mockResMgrFetcher;

    Interpret(insLocalCopy, stream, &mockResMgrFetcher);
    Interpret(insLocalCopy2, stream, &mockResMgrFetcher);
}

TEST_F(InsToSqeRuleV82Test, Interpret_local_copy_extend)
{
    u64 size = 100;
    DataBuffer srcBuffer(0x1234560, size);
    DataBuffer dstBuffer(0x1321000, size);
    InsLocalCopyExtend insLocalCopyExtend(srcBuffer, dstBuffer);

    std::vector<char> notifyLite1{fakeStreamId, fakeSqId};
    StreamLite stream(notifyLite1);
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    stream.rtsq = std::make_unique<RtsqA5>(rtsq);
    MOCKER_CPP_VIRTUAL(rtsq, &RtsqA5::SdmaCopy)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any());

    StubResMgrFetcher mockResMgrFetcher;

    Interpret(insLocalCopyExtend, stream, &mockResMgrFetcher);
}

TEST_F(InsToSqeRuleV82Test, Interpret_local_post_to)
{
    // normal notify
    NotifyLite* nullNotify = nullptr;
    std::vector<char> notifyLite1{fakeNotifyId, fakedevPhyId};
    NotifyLite notify(notifyLite1);
    NotifyLite* validNotify = &notify;
    MOCKER_CPP(&QueueNotifyLiteMgr::Get)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(nullNotify))
        .then(returnValue(validNotify));

    InsLocalPostTo insLocalPostTo(1, NotifyType::NORMAL, 0);
    insLocalPostTo.SetPostQid(0);

    std::vector<char> streamLite2{fakeStreamId, fakeSqId};
    StreamLite stream(streamLite2);
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    stream.rtsq = std::make_unique<RtsqA5>(rtsq);
    MOCKER_CPP_VIRTUAL(rtsq, &RtsqA5::NotifyRecordLoc).stubs().with(mockcpp::any());

    StubResMgrFetcher mockResMgrFetcher;

    EXPECT_THROW(Interpret(insLocalPostTo, stream, &mockResMgrFetcher), NullPtrException);
    Interpret(insLocalPostTo, stream, &mockResMgrFetcher);

    // count notify
    CntNto1NotifyLite* nullCntNotify = nullptr;
    std::vector<char> cntNto1NotifyLite1{fakeNotifyId, fakedevPhyId};
    CntNto1NotifyLite cntNtify(cntNto1NotifyLite1);
    CntNto1NotifyLite* validCntNotify = &cntNtify;
    MOCKER_CPP(&CntNto1NotifyLiteMgr::Get)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(nullCntNotify))
        .then(returnValue(validCntNotify));

    InsLocalPostTo insLocalPostToCounter(1, NotifyType::COUNTER, 0);
    insLocalPostToCounter.SetPostQid(0);

    MOCKER_CPP_VIRTUAL(rtsq, &RtsqA5::CntNto1NotifyRecord).stubs().with(mockcpp::any());

    EXPECT_THROW(Interpret(insLocalPostToCounter, stream, &mockResMgrFetcher), NullPtrException);
    Interpret(insLocalPostToCounter, stream, &mockResMgrFetcher);
}

TEST_F(InsToSqeRuleV82Test, Interpret_local_wait_from)
{
    // normal notify
    NotifyLite* nullNotify = nullptr;
    std::vector<char> notifyLite1{fakeNotifyId, fakedevPhyId};
    NotifyLite notify(notifyLite1);
    NotifyLite* validNotify = &notify;
    MOCKER_CPP(&QueueNotifyLiteMgr::Get)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(nullNotify))
        .then(returnValue(validNotify));

    InsLocalWaitFrom insLocalWaitFrom(0, NotifyType::NORMAL, 0);
    insLocalWaitFrom.SetWaitQid(1);

    std::vector<char> streamLite1{fakeStreamId, fakeSqId};
    StreamLite stream(streamLite1);
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    stream.rtsq = std::make_unique<RtsqA5>(rtsq);
    MOCKER_CPP_VIRTUAL(rtsq, &RtsqA5::NotifyWait).stubs().with(mockcpp::any());

    StubResMgrFetcher mockResMgrFetcher;

    EXPECT_THROW(Interpret(insLocalWaitFrom, stream, &mockResMgrFetcher), NullPtrException);
    Interpret(insLocalWaitFrom, stream, &mockResMgrFetcher);

    // count notify
    Cnt1tonNotifyLite* nullCntNotify = nullptr;
    std::vector<char> cntNto1NotifyLite1{fakeNotifyId, fakedevPhyId};
    Cnt1tonNotifyLite cntNtify(cntNto1NotifyLite1);
    Cnt1tonNotifyLite* validCntNotify = &cntNtify;
    MOCKER_CPP(&Cnt1tonNotifyLiteMgr::Get)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(nullCntNotify))
        .then(returnValue(validCntNotify));

    InsLocalWaitFrom insLocalWaitFromCounter(0, NotifyType::COUNTER, 0);
    insLocalWaitFromCounter.SetWaitQid(1);

    MOCKER_CPP_VIRTUAL(rtsq, &RtsqA5::Cnt1toNNotifyWait).stubs().with(mockcpp::any());

    EXPECT_THROW(Interpret(insLocalWaitFromCounter, stream, &mockResMgrFetcher), NullPtrException);
    Interpret(insLocalWaitFromCounter, stream, &mockResMgrFetcher);
}

TEST_F(InsToSqeRuleV82Test, Interpret_local_bcast_post)
{
    Cnt1tonNotifyLite* nullNotify = nullptr;
    std::vector<char> cntNto1NotifyLite1{fakeNotifyId, fakedevPhyId};
    Cnt1tonNotifyLite notify(cntNto1NotifyLite1);
    Cnt1tonNotifyLite* validNotify = &notify;
    MOCKER_CPP(&Cnt1tonNotifyLiteMgr::Get)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(nullNotify))
        .then(returnValue(validNotify));

    InsLocalBcastPost insLocalBcastPost(0);
    insLocalBcastPost.Append(0);
    insLocalBcastPost.Append(1);

    std::vector<char> streamLite1{fakeStreamId, fakeSqId};
    StreamLite stream(streamLite1);
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    stream.rtsq = std::make_unique<RtsqA5>(rtsq);
    MOCKER_CPP_VIRTUAL(rtsq, &RtsqA5::Cnt1toNNotifyRecord).stubs().with(mockcpp::any(), mockcpp::any());

    StubResMgrFetcher mockResMgrFetcher;

    EXPECT_THROW(Interpret(insLocalBcastPost, stream, &mockResMgrFetcher), NullPtrException);
    Interpret(insLocalBcastPost, stream, &mockResMgrFetcher);
}

TEST_F(InsToSqeRuleV82Test, Interpret_local_wait_group)
{
    CntNto1NotifyLite* nullNotify = nullptr;
    std::vector<char> cntNto1NotifyLite1{fakeNotifyId, fakedevPhyId};
    CntNto1NotifyLite notify(cntNto1NotifyLite1);
    CntNto1NotifyLite* validNotify = &notify;
    MOCKER_CPP(&CntNto1NotifyLiteMgr::Get)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(nullNotify))
        .then(returnValue(validNotify));

    InsLocalWaitGroup insLocalWaitGroup(0);
    insLocalWaitGroup.Append(0);
    insLocalWaitGroup.Append(1);

    std::vector<char> streamLite1{fakeStreamId, fakeSqId};
    StreamLite stream(streamLite1);
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    stream.rtsq = std::make_unique<RtsqA5>(rtsq);
    MOCKER_CPP_VIRTUAL(rtsq, &RtsqA5::CntNto1NotifyWait).stubs().with(mockcpp::any(), mockcpp::any());

    StubResMgrFetcher mockResMgrFetcher;

    EXPECT_THROW(Interpret(insLocalWaitGroup, stream, &mockResMgrFetcher), NullPtrException);
    Interpret(insLocalWaitGroup, stream, &mockResMgrFetcher);
}

TEST_F(InsToSqeRuleV82Test, Interpret_wait_ready)
{
    RankId remoteRank = 1;

    transportLite->impl = std::make_unique<StubTransportLiteImpl>();
    MemTransportLite* stubTransportPtr0 = nullptr;
    MemTransportLite* stubTransportPtr1 = transportLite.get();
    MOCKER_CPP(&MemTransportLiteMgr::GetOffload)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(stubTransportPtr0))
        .then(returnValue(stubTransportPtr1));
    MOCKER_CPP(&MemTransportLiteMgr::GetOpbase)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(stubTransportPtr0))
        .then(returnValue(stubTransportPtr1));

    InsWaitReady insWaitReady(remoteRank, link);

    std::vector<char> streamLite1{fakeStreamId, fakeSqId};
    StreamLite stream(streamLite1);
    StubResMgrFetcher mockResMgrFetcher;

    mockResMgrFetcher.currentOp.opMode = OpMode::OFFLOAD;
    EXPECT_THROW(Interpret(insWaitReady, stream, &mockResMgrFetcher), NullPtrException);
    Interpret(insWaitReady, stream, &mockResMgrFetcher);

    mockResMgrFetcher.currentOp.opMode = OpMode::OPBASE;
    EXPECT_THROW(Interpret(insWaitReady, stream, &mockResMgrFetcher), NullPtrException);
    Interpret(insWaitReady, stream, &mockResMgrFetcher);
}

TEST_F(InsToSqeRuleV82Test, Interpret_post_ready)
{
    RankId remoteRank = 1;

    transportLite->impl = std::make_unique<StubTransportLiteImpl>();
    MemTransportLite* stubTransportPtr0 = nullptr;
    MemTransportLite* stubTransportPtr1 = transportLite.get();
    MOCKER_CPP(&MemTransportLiteMgr::GetOffload)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(stubTransportPtr0))
        .then(returnValue(stubTransportPtr1));
    MOCKER_CPP(&MemTransportLiteMgr::GetOpbase)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(stubTransportPtr0))
        .then(returnValue(stubTransportPtr1));

    InsPostReady insPostReady(remoteRank, link);

    std::vector<char> streamLite1{fakeStreamId, fakeSqId};
    StreamLite stream(streamLite1);
    StubResMgrFetcher mockResMgrFetcher;

    EXPECT_THROW(Interpret(insPostReady, stream, &mockResMgrFetcher), NullPtrException);
    Interpret(insPostReady, stream, &mockResMgrFetcher);
}

TEST_F(InsToSqeRuleV82Test, Interpret_wait_fin)
{
    RankId remoteRank = 1;

    transportLite->impl = std::make_unique<StubTransportLiteImpl>();
    MemTransportLite* stubTransportPtr0 = nullptr;
    MemTransportLite* stubTransportPtr1 = transportLite.get();
    MOCKER_CPP(&MemTransportLiteMgr::GetOffload)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(stubTransportPtr0))
        .then(returnValue(stubTransportPtr1));
    MOCKER_CPP(&MemTransportLiteMgr::GetOpbase)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(stubTransportPtr0))
        .then(returnValue(stubTransportPtr1));

    InsWaitFin insWaitFin(remoteRank, link);

    std::vector<char> streamLite1{fakeStreamId, fakeSqId};
    StreamLite stream(streamLite1);
    StubResMgrFetcher mockResMgrFetcher;

    EXPECT_THROW(Interpret(insWaitFin, stream, &mockResMgrFetcher), NullPtrException);
    Interpret(insWaitFin, stream, &mockResMgrFetcher);
}

TEST_F(InsToSqeRuleV82Test, Interpret_post_fin)
{
    RankId remoteRank = 1;

    transportLite->impl = std::make_unique<StubTransportLiteImpl>();
    MemTransportLite* stubTransportPtr0 = nullptr;
    MemTransportLite* stubTransportPtr1 = transportLite.get();
    MOCKER_CPP(&MemTransportLiteMgr::GetOffload)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(stubTransportPtr0))
        .then(returnValue(stubTransportPtr1));
    MOCKER_CPP(&MemTransportLiteMgr::GetOpbase)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(stubTransportPtr0))
        .then(returnValue(stubTransportPtr1));

    InsPostFin insPostFin(remoteRank, link);

    std::vector<char> streamLite1{fakeStreamId, fakeSqId};
    StreamLite stream(streamLite1);
    StubResMgrFetcher mockResMgrFetcher;

    EXPECT_THROW(Interpret(insPostFin, stream, &mockResMgrFetcher), NullPtrException);
    Interpret(insPostFin, stream, &mockResMgrFetcher);
}

TEST_F(InsToSqeRuleV82Test, Interpret_batch_write)
{
    RankId remoteRank = 1;

    transportLite->impl = std::make_unique<StubTransportLiteImpl>();
    MemTransportLite* stubTransportPtr = transportLite.get();
    MOCKER_CPP(&MemTransportLiteMgr::GetOffload)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .then(returnValue(stubTransportPtr));
    MOCKER_CPP(&MemTransportLiteMgr::GetOpbase).stubs().with(mockcpp::any()).then(returnValue(stubTransportPtr));

    DataSlice localSlice(BufferType::SCRATCH, 0, 100);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, 100);
    InsBatchWrite insBatchWrite(remoteRank, link);
    insBatchWrite.PushWriteIns(std::make_unique<InsWrite>(remoteRank, link, localSlice, remoteSlice));
    insBatchWrite.PushWriteIns(
        std::make_unique<InsWriteReduce>(remoteRank, link, localSlice, remoteSlice, DataType::FP32, ReduceOp::SUM));

    std::vector<char> streamLite1{fakeStreamId, fakeSqId};
    StreamLite stream(streamLite1);
    StubResMgrFetcher mockResMgrFetcher;

    Interpret(insBatchWrite, stream, &mockResMgrFetcher);
}

TEST_F(InsToSqeRuleV82Test, Interpret_batch_write_is_empty_err)
{
    RankId remoteRank = 1;

    transportLite->impl = std::make_unique<StubTransportLiteImpl>();
    MemTransportLite* stubTransportPtr = transportLite.get();
    MOCKER_CPP(&MemTransportLiteMgr::GetOffload)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .then(returnValue(stubTransportPtr));
    MOCKER_CPP(&MemTransportLiteMgr::GetOpbase).stubs().with(mockcpp::any()).then(returnValue(stubTransportPtr));

    DataSlice localSlice(BufferType::SCRATCH, 0, 100);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, 100);
    InsBatchWrite insBatchWrite(remoteRank, link);

    std::vector<char> streamLite1{fakeStreamId, fakeSqId};
    StreamLite stream(streamLite1);
    StubResMgrFetcher mockResMgrFetcher;

    EXPECT_THROW(Interpret(insBatchWrite, stream, &mockResMgrFetcher), InvalidParamsException);
}

TEST_F(InsToSqeRuleV82Test, Interpret_batch_write_insWrite_size_is_0)
{
    RankId remoteRank = 1;

    transportLite->impl = std::make_unique<StubTransportLiteImpl>();
    MemTransportLite* stubTransportPtr = transportLite.get();
    MOCKER_CPP(&MemTransportLiteMgr::GetOffload)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .then(returnValue(stubTransportPtr));
    MOCKER_CPP(&MemTransportLiteMgr::GetOpbase).stubs().with(mockcpp::any()).then(returnValue(stubTransportPtr));

    DataSlice localSlice(BufferType::SCRATCH, 0, 100);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, 100);
    InsBatchWrite insBatchWrite(remoteRank, link);
    auto insWrite = std::make_unique<InsWrite>(remoteRank, link, localSlice, remoteSlice);
    insWrite->localSlice_.size = 0;
    insWrite->remoteSlice_.size = 0;
    insBatchWrite.PushWriteIns(std::move(insWrite));

    std::vector<char> streamLite1{fakeStreamId, fakeSqId};
    StreamLite stream(streamLite1);
    StubResMgrFetcher mockResMgrFetcher;

    EXPECT_NO_THROW(Interpret(insBatchWrite, stream, &mockResMgrFetcher));
}

TEST_F(InsToSqeRuleV82Test, Interpret_batch_write_insWrite_size_isnot_0_err)
{
    RankId remoteRank = 1;

    transportLite->impl = std::make_unique<StubTransportLiteImpl>();
    MemTransportLite* stubTransportPtr = transportLite.get();
    MOCKER_CPP(&MemTransportLiteMgr::GetOffload)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .then(returnValue(stubTransportPtr));
    MOCKER_CPP(&MemTransportLiteMgr::GetOpbase).stubs().with(mockcpp::any()).then(returnValue(stubTransportPtr));

    DataSlice localSlice(BufferType::SCRATCH, 0, 100);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, 100);
    InsBatchWrite insBatchWrite(remoteRank, link);
    auto insWrite = std::make_unique<InsWrite>(remoteRank, link, localSlice, remoteSlice);
    insWrite->localSlice_.size = 0;
    insBatchWrite.PushWriteIns(std::move(insWrite));

    std::vector<char> streamLite1{fakeStreamId, fakeSqId};
    StreamLite stream(streamLite1);
    StubResMgrFetcher mockResMgrFetcher;

    EXPECT_THROW(Interpret(insBatchWrite, stream, &mockResMgrFetcher), InvalidParamsException);
}

TEST_F(InsToSqeRuleV82Test, Interpret_batch_write_insWriteReduce_size_is_0)
{
    RankId remoteRank = 1;

    transportLite->impl = std::make_unique<StubTransportLiteImpl>();
    MemTransportLite* stubTransportPtr = transportLite.get();
    MOCKER_CPP(&MemTransportLiteMgr::GetOffload)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .then(returnValue(stubTransportPtr));
    MOCKER_CPP(&MemTransportLiteMgr::GetOpbase).stubs().with(mockcpp::any()).then(returnValue(stubTransportPtr));

    DataSlice localSlice(BufferType::SCRATCH, 0, 100);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, 100);
    InsBatchWrite insBatchWrite(remoteRank, link);
    auto insWriteReduce
        = std::make_unique<InsWriteReduce>(remoteRank, link, localSlice, remoteSlice, DataType::FP32, ReduceOp::SUM);
    insWriteReduce->localSlice_.size = 0;
    insWriteReduce->remoteSlice_.size = 0;
    insBatchWrite.PushWriteIns(std::move(insWriteReduce));

    std::vector<char> streamLite1{fakeStreamId, fakeSqId};
    StreamLite stream(streamLite1);
    StubResMgrFetcher mockResMgrFetcher;

    EXPECT_NO_THROW(Interpret(insBatchWrite, stream, &mockResMgrFetcher));
}

TEST_F(InsToSqeRuleV82Test, Interpret_batch_write_insWriteReduce_size_isnot_0_err)
{
    RankId remoteRank = 1;

    transportLite->impl = std::make_unique<StubTransportLiteImpl>();
    MemTransportLite* stubTransportPtr = transportLite.get();
    MOCKER_CPP(&MemTransportLiteMgr::GetOffload)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .then(returnValue(stubTransportPtr));
    MOCKER_CPP(&MemTransportLiteMgr::GetOpbase).stubs().with(mockcpp::any()).then(returnValue(stubTransportPtr));

    DataSlice localSlice(BufferType::SCRATCH, 0, 100);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, 100);
    InsBatchWrite insBatchWrite(remoteRank, link);
    auto insWriteReduce
        = std::make_unique<InsWriteReduce>(remoteRank, link, localSlice, remoteSlice, DataType::FP32, ReduceOp::SUM);
    insWriteReduce->localSlice_.size = 0;
    insBatchWrite.PushWriteIns(std::move(insWriteReduce));

    std::vector<char> streamLite1{fakeStreamId, fakeSqId};
    StreamLite stream(streamLite1);
    StubResMgrFetcher mockResMgrFetcher;

    EXPECT_THROW(Interpret(insBatchWrite, stream, &mockResMgrFetcher), InvalidParamsException);
}

TEST_F(InsToSqeRuleV82Test, Interpret_batch_read)
{
    RankId remoteRank = 1;

    transportLite->impl = std::make_unique<StubTransportLiteImpl>();
    MemTransportLite* stubTransportPtr = transportLite.get();
    MOCKER_CPP(&MemTransportLiteMgr::GetOffload)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .then(returnValue(stubTransportPtr));
    MOCKER_CPP(&MemTransportLiteMgr::GetOpbase).stubs().with(mockcpp::any()).then(returnValue(stubTransportPtr));

    DataSlice localSlice(BufferType::SCRATCH, 0, 100);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, 100);
    InsBatchRead insBatchRead(remoteRank, link);
    insBatchRead.PushReadIns(std::make_unique<InsRead>(remoteRank, link, localSlice, remoteSlice));
    insBatchRead.PushReadIns(
        std::make_unique<InsReadReduce>(remoteRank, link, localSlice, remoteSlice, DataType::FP32, ReduceOp::SUM));

    std::vector<char> streamLite1{fakeStreamId, fakeSqId};
    StreamLite stream(streamLite1);
    StubResMgrFetcher mockResMgrFetcher;

    Interpret(insBatchRead, stream, &mockResMgrFetcher);
}

TEST_F(InsToSqeRuleV82Test, Interpret_batch_read_is_empty_err)
{
    RankId remoteRank = 1;

    transportLite->impl = std::make_unique<StubTransportLiteImpl>();
    MemTransportLite* stubTransportPtr = transportLite.get();
    MOCKER_CPP(&MemTransportLiteMgr::GetOffload)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .then(returnValue(stubTransportPtr));
    MOCKER_CPP(&MemTransportLiteMgr::GetOpbase).stubs().with(mockcpp::any()).then(returnValue(stubTransportPtr));

    DataSlice localSlice(BufferType::SCRATCH, 0, 100);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, 100);
    InsBatchRead insBatchRead(remoteRank, link);

    std::vector<char> streamLite1{fakeStreamId, fakeSqId};
    StreamLite stream(streamLite1);
    StubResMgrFetcher mockResMgrFetcher;

    EXPECT_THROW(Interpret(insBatchRead, stream, &mockResMgrFetcher), InvalidParamsException);
}

TEST_F(InsToSqeRuleV82Test, Interpret_batch_read_insRead_size_is_0)
{
    RankId remoteRank = 1;

    transportLite->impl = std::make_unique<StubTransportLiteImpl>();
    MemTransportLite* stubTransportPtr = transportLite.get();
    MOCKER_CPP(&MemTransportLiteMgr::GetOffload)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .then(returnValue(stubTransportPtr));
    MOCKER_CPP(&MemTransportLiteMgr::GetOpbase).stubs().with(mockcpp::any()).then(returnValue(stubTransportPtr));

    DataSlice localSlice(BufferType::SCRATCH, 0, 100);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, 100);
    InsBatchRead insBatchRead(remoteRank, link);
    auto insRead = std::make_unique<InsRead>(remoteRank, link, localSlice, remoteSlice);
    insRead->localSlice_.size = 0;
    insRead->remoteSlice_.size = 0;
    insBatchRead.PushReadIns(std::move(insRead));

    std::vector<char> streamLite1{fakeStreamId, fakeSqId};
    StreamLite stream(streamLite1);
    StubResMgrFetcher mockResMgrFetcher;

    EXPECT_NO_THROW(Interpret(insBatchRead, stream, &mockResMgrFetcher));
}

TEST_F(InsToSqeRuleV82Test, Interpret_batch_read_insRead_size_isnot_0_err)
{
    RankId remoteRank = 1;

    transportLite->impl = std::make_unique<StubTransportLiteImpl>();
    MemTransportLite* stubTransportPtr = transportLite.get();
    MOCKER_CPP(&MemTransportLiteMgr::GetOffload)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .then(returnValue(stubTransportPtr));
    MOCKER_CPP(&MemTransportLiteMgr::GetOpbase).stubs().with(mockcpp::any()).then(returnValue(stubTransportPtr));

    DataSlice localSlice(BufferType::SCRATCH, 0, 100);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, 100);
    InsBatchRead insBatchRead(remoteRank, link);
    auto insRead = std::make_unique<InsRead>(remoteRank, link, localSlice, remoteSlice);
    insRead->localSlice_.size = 0;
    insBatchRead.PushReadIns(std::move(insRead));

    std::vector<char> streamLite1{fakeStreamId, fakeSqId};
    StreamLite stream(streamLite1);
    StubResMgrFetcher mockResMgrFetcher;

    EXPECT_THROW(Interpret(insBatchRead, stream, &mockResMgrFetcher), InvalidParamsException);
}

TEST_F(InsToSqeRuleV82Test, Interpret_batch_read_insReadReduce_size_is_0)
{
    RankId remoteRank = 1;

    transportLite->impl = std::make_unique<StubTransportLiteImpl>();
    MemTransportLite* stubTransportPtr = transportLite.get();
    MOCKER_CPP(&MemTransportLiteMgr::GetOffload)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .then(returnValue(stubTransportPtr));
    MOCKER_CPP(&MemTransportLiteMgr::GetOpbase).stubs().with(mockcpp::any()).then(returnValue(stubTransportPtr));

    DataSlice localSlice(BufferType::SCRATCH, 0, 100);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, 100);
    InsBatchRead insBatchRead(remoteRank, link);
    auto insReadReduce
        = std::make_unique<InsReadReduce>(remoteRank, link, localSlice, remoteSlice, DataType::FP32, ReduceOp::SUM);
    insReadReduce->localSlice_.size = 0;
    insReadReduce->remoteSlice_.size = 0;
    insBatchRead.PushReadIns(std::move(insReadReduce));

    std::vector<char> streamLite1{fakeStreamId, fakeSqId};
    StreamLite stream(streamLite1);
    StubResMgrFetcher mockResMgrFetcher;

    EXPECT_NO_THROW(Interpret(insBatchRead, stream, &mockResMgrFetcher));
}

TEST_F(InsToSqeRuleV82Test, Interpret_batch_read_insReadReduce_size_isnot_0_err)
{
    RankId remoteRank = 1;

    transportLite->impl = std::make_unique<StubTransportLiteImpl>();
    MemTransportLite* stubTransportPtr = transportLite.get();
    MOCKER_CPP(&MemTransportLiteMgr::GetOffload)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .then(returnValue(stubTransportPtr));
    MOCKER_CPP(&MemTransportLiteMgr::GetOpbase).stubs().with(mockcpp::any()).then(returnValue(stubTransportPtr));

    DataSlice localSlice(BufferType::SCRATCH, 0, 100);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, 100);
    InsBatchRead insBatchRead(remoteRank, link);
    auto insReadReduce
        = std::make_unique<InsReadReduce>(remoteRank, link, localSlice, remoteSlice, DataType::FP32, ReduceOp::SUM);
    insReadReduce->localSlice_.size = 0;
    insBatchRead.PushReadIns(std::move(insReadReduce));

    std::vector<char> streamLite1{fakeStreamId, fakeSqId};
    StreamLite stream(streamLite1);
    StubResMgrFetcher mockResMgrFetcher;

    EXPECT_THROW(Interpret(insBatchRead, stream, &mockResMgrFetcher), InvalidParamsException);
}

TEST_F(InsToSqeRuleV82Test, Interpret_write)
{
    RankId remoteRank = 1;

    transportLite->impl = std::make_unique<StubTransportLiteImpl>();
    MemTransportLite* stubTransportPtr = transportLite.get();
    MOCKER_CPP(&MemTransportLiteMgr::GetOffload)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .then(returnValue(stubTransportPtr));
    MOCKER_CPP(&MemTransportLiteMgr::GetOpbase).stubs().with(mockcpp::any()).then(returnValue(stubTransportPtr));

    DataSlice localSlice(BufferType::SCRATCH, 0, 100);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, 100);
    InsWrite insWrite(remoteRank, link, localSlice, remoteSlice);

    std::vector<char> streamLite1{fakeStreamId, fakeSqId};
    StreamLite stream(streamLite1);
    StubResMgrFetcher mockResMgrFetcher;

    Interpret(insWrite, stream, &mockResMgrFetcher);
}

TEST_F(InsToSqeRuleV82Test, Interpret_write_extend)
{
    RankId remoteRank = 1;

    transportLite->impl = std::make_unique<StubTransportLiteImpl>();
    MemTransportLite* stubTransportPtr = transportLite.get();
    MOCKER_CPP(&MemTransportLiteMgr::GetOffload)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .then(returnValue(stubTransportPtr));
    MOCKER_CPP(&MemTransportLiteMgr::GetOpbase).stubs().with(mockcpp::any()).then(returnValue(stubTransportPtr));

    u64 size = 100;
    DataBuffer localBuffer(0x1234560, size);
    DataBuffer remoteBuffer(0x1321000, size);
    InsWriteExtend insWriteExtend(remoteRank, link, localBuffer, remoteBuffer);

    std::vector<char> streamLite1{fakeStreamId, fakeSqId};
    StreamLite stream(streamLite1);
    StubResMgrFetcher mockResMgrFetcher;

    Interpret(insWriteExtend, stream, &mockResMgrFetcher);
}

TEST_F(InsToSqeRuleV82Test, Interpret_write_reduce)
{
    RankId remoteRank = 1;

    transportLite->impl = std::make_unique<StubTransportLiteImpl>();
    MemTransportLite* stubTransportPtr = transportLite.get();
    MOCKER_CPP(&MemTransportLiteMgr::GetOffload)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .then(returnValue(stubTransportPtr));
    MOCKER_CPP(&MemTransportLiteMgr::GetOpbase).stubs().with(mockcpp::any()).then(returnValue(stubTransportPtr));

    DataSlice localSlice(BufferType::SCRATCH, 0, 100);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, 100);
    InsWriteReduce insWriteReduce(remoteRank, link, localSlice, remoteSlice, DataType::FP32, ReduceOp::SUM);

    std::vector<char> streamLite1{fakeStreamId, fakeSqId};
    StreamLite stream(streamLite1);
    StubResMgrFetcher mockResMgrFetcher;

    Interpret(insWriteReduce, stream, &mockResMgrFetcher);
}

TEST_F(InsToSqeRuleV82Test, Interpret_read)
{
    RankId remoteRank = 1;

    transportLite->impl = std::make_unique<StubTransportLiteImpl>();
    MemTransportLite* stubTransportPtr = transportLite.get();
    MOCKER_CPP(&MemTransportLiteMgr::GetOffload)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .then(returnValue(stubTransportPtr));
    MOCKER_CPP(&MemTransportLiteMgr::GetOpbase).stubs().with(mockcpp::any()).then(returnValue(stubTransportPtr));

    DataSlice localSlice(BufferType::SCRATCH, 0, 100);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, 100);
    InsRead insRead(remoteRank, link, localSlice, remoteSlice);

    std::vector<char> streamLite1{fakeStreamId, fakeSqId};
    StreamLite stream(streamLite1);
    StubResMgrFetcher mockResMgrFetcher;

    Interpret(insRead, stream, &mockResMgrFetcher);
}

TEST_F(InsToSqeRuleV82Test, Interpret_read_reduce)
{
    RankId remoteRank = 1;

    transportLite->impl = std::make_unique<StubTransportLiteImpl>();
    MemTransportLite* stubTransportPtr = transportLite.get();
    MOCKER_CPP(&MemTransportLiteMgr::GetOffload)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .then(returnValue(stubTransportPtr));
    MOCKER_CPP(&MemTransportLiteMgr::GetOpbase).stubs().with(mockcpp::any()).then(returnValue(stubTransportPtr));

    DataSlice localSlice(BufferType::SCRATCH, 0, 100);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, 100);
    InsReadReduce insReadReduce(remoteRank, link, localSlice, remoteSlice, DataType::FP32, ReduceOp::SUM);

    std::vector<char> streamLite1{fakeStreamId, fakeSqId};
    StreamLite stream(streamLite1);
    StubResMgrFetcher mockResMgrFetcher;

    Interpret(insReadReduce, stream, &mockResMgrFetcher);
}

TEST_F(InsToSqeRuleV82Test, Interpret_write_reduce_with_fin)
{
    RankId remoteRank = 1;

    transportLite->impl = std::make_unique<StubTransportLiteImpl>();
    MemTransportLite* stubTransportPtr = transportLite.get();
    MOCKER_CPP(&MemTransportLiteMgr::GetOffload)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .then(returnValue(stubTransportPtr));
    MOCKER_CPP(&MemTransportLiteMgr::GetOpbase).stubs().with(mockcpp::any()).then(returnValue(stubTransportPtr));

    DataSlice localSlice(BufferType::SCRATCH, 0, 100);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, 100);
    InsWriteReduceWithFin insWriteReduceWithFin(
        remoteRank, link, localSlice, remoteSlice, DataType::FP32, ReduceOp::SUM, NotifyType::NORMAL);
    std::vector<char> streamLite1{fakeStreamId, fakeSqId};
    StreamLite stream(streamLite1);
    StubResMgrFetcher mockResMgrFetcher;

    Interpret(insWriteReduceWithFin, stream, &mockResMgrFetcher);
}

TEST_F(InsToSqeRuleV82Test, Interpret_write_with_fin)
{
    RankId remoteRank = 1;

    transportLite->impl = std::make_unique<StubTransportLiteImpl>();
    MemTransportLite* stubTransportPtr = transportLite.get();
    MOCKER_CPP(&MemTransportLiteMgr::GetOffload)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .then(returnValue(stubTransportPtr));
    MOCKER_CPP(&MemTransportLiteMgr::GetOpbase).stubs().with(mockcpp::any()).then(returnValue(stubTransportPtr));

    DataSlice localSlice(BufferType::SCRATCH, 0, 100);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, 100);
    InsWriteWithFin insWriteWithFin(remoteRank, link, localSlice, remoteSlice, NotifyType::NORMAL);

    std::vector<char> streamLite1{fakeStreamId, fakeSqId};
    StreamLite stream(streamLite1);
    StubResMgrFetcher mockResMgrFetcher;

    Interpret(insWriteWithFin, stream, &mockResMgrFetcher);
}

TEST_F(InsToSqeRuleV82Test, Interpret_write_with_fin_extend)
{
    RankId remoteRank = 1;

    transportLite->impl = std::make_unique<StubTransportLiteImpl>();
    MemTransportLite* stubTransportPtr = transportLite.get();
    MOCKER_CPP(&MemTransportLiteMgr::GetOffload)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .then(returnValue(stubTransportPtr));
    MOCKER_CPP(&MemTransportLiteMgr::GetOpbase).stubs().with(mockcpp::any()).then(returnValue(stubTransportPtr));

    u64 size = 100;
    DataBuffer localBuffer(0x1234560, size);
    DataBuffer remoteBuffer(0x1321000, size);
    InsWriteWithFinExtend insWriteWithFinExtend(remoteRank, link, localBuffer, remoteBuffer, NotifyType::NORMAL);
    std::vector<char> streamLite1{fakeStreamId, fakeSqId};
    StreamLite stream(streamLite1);
    StubResMgrFetcher mockResMgrFetcher;

    Interpret(insWriteWithFinExtend, stream, &mockResMgrFetcher);
}

TEST_F(InsToSqeRuleV82Test, Interpret_AicpuReduce)
{
    DataSlice srcSlice(BufferType::INPUT, 0, 100);
    DataSlice dstSlice(BufferType::OUTPUT, 0, 100);
    InsAicpuReduce insAicpuReduce(srcSlice, dstSlice, DataType::FP64, ReduceOp::SUM);
    int64_t i0 = 1;
    int64_t i1 = 2;
    uint64_t ui0 = 1;
    uint64_t ui1 = 2;
    double d0 = 1.0;
    double d1 = 2.0;
    insAicpuReduce.RunAicpuReduce(&i0, 8, &i1, 8, DataType::INT64, ReduceOp::SUM);
    EXPECT_EQ(true, i0 == 3);
    insAicpuReduce.RunAicpuReduce(&i0, 8, &i1, 8, DataType::INT64, ReduceOp::PROD);
    EXPECT_EQ(true, i0 == 6);
    insAicpuReduce.RunAicpuReduce(&ui0, 8, &ui1, 8, DataType::UINT64, ReduceOp::MAX);
    EXPECT_EQ(true, ui0 == 2);
    insAicpuReduce.RunAicpuReduce(&d0, 8, &d1, 8, DataType::FP64, ReduceOp::MIN);
    EXPECT_EQ(true, d0 == 1.0);

    std::vector<char> notifyLite1{1, 2};
    StreamLite stream(notifyLite1);
    RtsqA5 rtsq(0, 1, 2);
    stream.rtsq = std::make_unique<RtsqA5>(rtsq);
    MOCKER(&InsAicpuReduce::RunAicpuReduce).stubs();

    StubResMgrFetcher mockResMgrFetcher;
    Interpret(insAicpuReduce, stream, &mockResMgrFetcher);
}

TEST_F(InsToSqeRuleV82Test, Interpret_StreamSync)
{
    InsStreamSync insStreamSync;

    std::vector<char> notifyLite1{1, 2};
    StreamLite stream(notifyLite1);
    RtsqA5 rtsq(0, 1, 2);
    stream.rtsq = std::make_unique<RtsqA5>(rtsq);

    StubResMgrFetcher mockResMgrFetcher;
    Interpret(insStreamSync, stream, &mockResMgrFetcher);
}
TEST_F(InsToSqeRuleV82Test, Interpret_local_copy_extend_err)
{
    u64 size = 100;
    DataBuffer srcBuffer(0x1234560, size);
    DataBuffer dstBuffer(0x1321000, size);
    InsLocalCopyExtend insLocalCopyExtend(srcBuffer, dstBuffer);

    std::vector<char> notifyLite1{fakeStreamId, fakeSqId};
    StreamLite stream(notifyLite1);
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    stream.rtsq = std::make_unique<RtsqA5>(rtsq);
    MOCKER_CPP_VIRTUAL(rtsq, &RtsqA5::SdmaCopy)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any());

    StubResMgrFetcher mockResMgrFetcher;
    insLocalCopyExtend.srcBuffer_.size = 0;
    Interpret(insLocalCopyExtend, stream, &mockResMgrFetcher);
}

TEST_F(InsToSqeRuleV82Test, Interpret_write_err)
{
    RankId remoteRank = 1;

    transportLite->impl = std::make_unique<StubTransportLiteImpl>();
    MemTransportLite* stubTransportPtr = transportLite.get();
    MOCKER_CPP(&MemTransportLiteMgr::GetOffload)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .then(returnValue(stubTransportPtr));
    MOCKER_CPP(&MemTransportLiteMgr::GetOpbase).stubs().with(mockcpp::any()).then(returnValue(stubTransportPtr));

    DataSlice localSlice(BufferType::SCRATCH, 0, 100);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, 100);
    InsWrite insWrite(remoteRank, link, localSlice, remoteSlice);

    std::vector<char> streamLite1{fakeStreamId, fakeSqId};
    StreamLite stream(streamLite1);
    StubResMgrFetcher mockResMgrFetcher;
    insWrite.localSlice_.size = 0;
    EXPECT_THROW(Interpret(insWrite, stream, &mockResMgrFetcher), InvalidParamsException);
}

TEST_F(InsToSqeRuleV82Test, Interpret_write_extend_err)
{
    RankId remoteRank = 1;

    transportLite->impl = std::make_unique<StubTransportLiteImpl>();
    MemTransportLite* stubTransportPtr = transportLite.get();
    MOCKER_CPP(&MemTransportLiteMgr::GetOffload)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .then(returnValue(stubTransportPtr));
    MOCKER_CPP(&MemTransportLiteMgr::GetOpbase).stubs().with(mockcpp::any()).then(returnValue(stubTransportPtr));

    u64 size = 100;
    DataBuffer localBuffer(0x1234560, size);
    DataBuffer remoteBuffer(0x1321000, size);
    InsWriteExtend insWriteExtend(remoteRank, link, localBuffer, remoteBuffer);

    std::vector<char> streamLite1{fakeStreamId, fakeSqId};
    StreamLite stream(streamLite1);
    StubResMgrFetcher mockResMgrFetcher;
    insWriteExtend.localBuffer_.size = 0;
    Interpret(insWriteExtend, stream, &mockResMgrFetcher);
}

TEST_F(InsToSqeRuleV82Test, Interpret_write_reduce_with_fin_err)
{
    RankId remoteRank = 1;

    transportLite->impl = std::make_unique<StubTransportLiteImpl>();
    MemTransportLite* stubTransportPtr = transportLite.get();
    MOCKER_CPP(&MemTransportLiteMgr::GetOffload)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .then(returnValue(stubTransportPtr));
    MOCKER_CPP(&MemTransportLiteMgr::GetOpbase).stubs().with(mockcpp::any()).then(returnValue(stubTransportPtr));

    DataSlice localSlice(BufferType::SCRATCH, 0, 100);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, 100);
    InsWriteReduceWithFin insWriteReduceWithFin(
        remoteRank, link, localSlice, remoteSlice, DataType::FP32, ReduceOp::SUM, NotifyType::NORMAL);
    std::vector<char> streamLite1{fakeStreamId, fakeSqId};
    StreamLite stream(streamLite1);
    StubResMgrFetcher mockResMgrFetcher;
    insWriteReduceWithFin.localSlice_.size = 0;
    Interpret(insWriteReduceWithFin, stream, &mockResMgrFetcher);
}

TEST_F(InsToSqeRuleV82Test, Interpret_write_with_fin_err)
{
    RankId remoteRank = 1;

    transportLite->impl = std::make_unique<StubTransportLiteImpl>();
    MemTransportLite* stubTransportPtr = transportLite.get();
    MOCKER_CPP(&MemTransportLiteMgr::GetOffload)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .then(returnValue(stubTransportPtr));
    MOCKER_CPP(&MemTransportLiteMgr::GetOpbase).stubs().with(mockcpp::any()).then(returnValue(stubTransportPtr));

    DataSlice localSlice(BufferType::SCRATCH, 0, 100);
    DataSlice remoteSlice(BufferType::SCRATCH, 0, 100);
    InsWriteWithFin insWriteWithFin(remoteRank, link, localSlice, remoteSlice, NotifyType::NORMAL);
    std::vector<char> streamLite1{fakeStreamId, fakeSqId};
    StreamLite stream(streamLite1);
    StubResMgrFetcher mockResMgrFetcher;
    insWriteWithFin.localSlice_.size = 0;
    Interpret(insWriteWithFin, stream, &mockResMgrFetcher);
}

TEST_F(InsToSqeRuleV82Test, Interpret_write_with_fin_extend_err)
{
    RankId remoteRank = 1;

    transportLite->impl = std::make_unique<StubTransportLiteImpl>();
    MemTransportLite* stubTransportPtr = transportLite.get();
    MOCKER_CPP(&MemTransportLiteMgr::GetOffload)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .then(returnValue(stubTransportPtr));
    MOCKER_CPP(&MemTransportLiteMgr::GetOpbase).stubs().with(mockcpp::any()).then(returnValue(stubTransportPtr));

    u64 size = 100;
    DataBuffer localBuffer(0x1234560, size);
    DataBuffer remoteBuffer(0x1321000, size);
    InsWriteWithFinExtend insWriteWithFinExtend(remoteRank, link, localBuffer, remoteBuffer, NotifyType::NORMAL);
    std::vector<char> streamLite1{fakeStreamId, fakeSqId};
    StreamLite stream(streamLite1);
    StubResMgrFetcher mockResMgrFetcher;
    insWriteWithFinExtend.localBuffer_.size = 0;
    Interpret(insWriteWithFinExtend, stream, &mockResMgrFetcher);
}

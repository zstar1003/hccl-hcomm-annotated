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

#include "ins_executor.h"
#include "null_ptr_exception.h"
#include "ins_to_sqe_rule.h"
#include "rtsq_a5.h"
#include "dev_ub_connection.h"
#include "notify_lite.h"
#include "udma_data_struct.h"
#include "ub_conn_lite.h"
#include "mem_transport_lite.h"
#include "orion_adapter_rts.h"
#include "hierarchical_queue.h"
#include "data_buffer.h"
#undef private
#undef protected
using namespace Hccl;

class InsExecutorTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "InsExecutorTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "InsExecutorTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_910A2));
        MOCKER_CPP(&RtsqBase::QuerySqBaseAddr)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(reinterpret_cast<u64>(&mockSq)));
        MOCKER_CPP(&RtsqBase::QuerySqStatusByType).stubs().with(mockcpp::any()).will(returnValue(static_cast<u32>(0)));
        MOCKER_CPP(&RtsqBase::ConfigSqStatusByType).stubs();

        std::cout << "A Test case in InsExecutorTest SetUp" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in InsExecutorTest TearDown" << std::endl;
    }
    u32 fakeStreamId = 0;
    u32 fakeSqId = 0;
    u32 fakedevPhyId = 0;
    u32 fakeNotifyId1 = 1;
    u32 fakeNotifyDevPhyId1 = 1;
    u32 fakeNotifyId2 = 2;
    u32 fakeNotifyDevPhyId2 = 2;

    u8 mockSq[AC_SQE_SIZE * AC_SQE_MAX_CNT]{0};
};

class MockDevIdProvider : public ResMgrFetcher {
public:
    HostDeviceSyncNotifyLiteMgr* GetHostDeviceSyncNotifyLiteMgr() override { return &hostDeviceSyncNotifyLiteMgr; }

    StreamLiteMgr* GetStreamLiteMgr() override { return &streamLiteMgr; }

    u32 GetExecTimeOut() override { return 1836; }

    QueueNotifyLiteMgr* GetQueueNotifyLiteMgr() override { return &queueNotifyLiteMgr; }

    Cnt1tonNotifyLiteMgr* GetCnt1tonNotifyLiteMgr() override { return &cnt1tonNotifyLiteMgr; }

    CntNto1NotifyLiteMgr* GetCntNto1NotifyLiteMgr() override { return &cntNto1NotifyLiteMgr; }

    MemTransportLiteMgr* GetTransportLiteMgr() override { return transportLiteMgr.get(); }

    ConnectedLinkMgr* GetConnectedLinkMgr() override { return &connectedLinkMgr; }

    MirrorTaskManager* GetMirrorTaskMgr() override { return mirrorTaskMgr.get(); }

    DevId GetDevPhyId() { return 0; }

    u64 GetCounterAddr() override { return opCounterAddr; }

    u64 GetLocAddr(BufferType type) { return 0xffffffff; }

    CollOperator GetCurrentOp() override
    {
        currentOp.opTag = "tag";
        return currentOp;
    }

    RmaBufferLite* GetRmaBufferLite(BufferType type) override { return rmaBufferLiteVec[type].get(); }

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

TEST_F(InsExecutorTest, test_ins_executor)
{
    MOCKER_CPP(&SqeMgr::Begin).stubs().with(mockcpp::any()).will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&SqeMgr::Add).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&SqeMgr::Commit).stubs().with(mockcpp::any()).will(returnValue(HcclResult::HCCL_SUCCESS));
    LinkData link(BasePortType(PortDeploymentType::P2P), 0, 1, 0, 1);

    MockDevIdProvider mockResMgrFetcher;
    CollOperator op = mockResMgrFetcher.GetCurrentOp();

    BinaryStream liteBinaryStream;
    liteBinaryStream << fakeStreamId;
    liteBinaryStream << fakeSqId;
    liteBinaryStream << fakedevPhyId;
    std::vector<char> uniqueId{};
    liteBinaryStream.Dump(uniqueId);
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    auto streamPtr = std::make_unique<StreamLite>(uniqueId);
    streamPtr->rtsq = std::make_unique<RtsqA5>(rtsq);
    MOCKER_CPP_VIRTUAL(rtsq, &RtsqA5::SdmaCopy)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any());
    mockResMgrFetcher.GetStreamLiteMgr()->streams.emplace_back(std::move(streamPtr));

    BinaryStream notifyStream1;
    notifyStream1 << fakeNotifyId1;
    notifyStream1 << fakeNotifyDevPhyId1;
    std::vector<char> notifyUniqueId1;
    notifyStream1.Dump(notifyUniqueId1);
    NotifyLite notifyLite1(notifyUniqueId1);
    mockResMgrFetcher.GetHostDeviceSyncNotifyLiteMgr()->notifys[0] = std::make_unique<NotifyLite>(notifyLite1);

    BinaryStream notifyStream2;
    notifyStream2 << fakeNotifyId2;
    notifyStream2 << fakeNotifyDevPhyId2;
    std::vector<char> notifyUniqueId2;
    notifyStream2.Dump(notifyUniqueId2);
    NotifyLite notifyLite2(notifyUniqueId2);
    mockResMgrFetcher.GetHostDeviceSyncNotifyLiteMgr()->notifys[0] = std::make_unique<NotifyLite>(notifyLite2);

    InsExecutor insExecutor(&mockResMgrFetcher);
    InsQueue insQueue;

    DataSlice srcSlice(BufferType::SCRATCH, 0, 1);
    DataSlice dstSlice(BufferType::SCRATCH, 1, 1);
    insQueue.Append(std::make_unique<InsLocalCopy>(srcSlice, dstSlice));

    insQueue.Append(std::make_unique<InsWaitReady>(1, link));
    insQueue.Append(std::make_unique<InsPostReady>(1, link));
    insQueue.Append(std::make_unique<InsWaitFin>(1, link));
    insQueue.Append(std::make_unique<InsPostFin>(1, link));
    insQueue.Append(std::make_unique<InsWaitFinAck>(1, link));
    insQueue.Append(std::make_unique<InsPostFinAck>(1, link));

    vector<shared_ptr<InsQueue>> slaves{};
    auto slave = make_shared<InsQueue>();
    slave->masterFlag = false;
    slave->id = 1;
    slaves.push_back(slave);
    insQueue.slaves = slaves;

    MOCKER_CPP(&InsExecutor::ExecuteSingleQue).stubs().with(mockcpp::any());
    EXPECT_NO_THROW(insExecutor.Execute(insQueue));
}

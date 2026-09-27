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
        MOCKER_CPP(&RtsqBase::QuerySqDepth)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(static_cast<u32>(AC_SQE_MAX_CNT)));
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
    CntNto1NotifyLiteMgr cntNto1NotifyLiteMgr;
    QueueNotifyLiteMgr queueNotifyLiteMgr;
    Cnt1tonNotifyLiteMgr cnt1tonNotifyLiteMgr;
    ConnectedLinkMgr connectedLinkMgr;

    std::unique_ptr<MirrorTaskManagerLite> mirrorTaskMgrLite = std::make_unique<MirrorTaskManagerLite>();

    std::unique_ptr<MemTransportLiteMgr> transportLiteMgr
        = std::make_unique<MemTransportLiteMgr>(mirrorTaskMgrLite.get());
    CollOperator currentOp;
    std::vector<std::unique_ptr<RmaBufferLite>> rmaBufferLiteVec;
    std::unordered_map<DataBuffer, SendRecvItemTokenInfo> sendRecvTokenMap;
    u64 opCounterAddr;
};

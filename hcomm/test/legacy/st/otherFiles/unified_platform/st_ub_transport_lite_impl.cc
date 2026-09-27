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
#include <mockcpp/mockcpp.hpp>
#include <mockcpp/MockObject.h>
#define private public
#include "virtual_topo.h"
#include "dev_ub_connection.h"
#include "task.h"
#include "ub_transport_lite_impl.h"
#include "ub_mem_transport.h"
#include "mem_transport_lite.h"
#include "ub_local_notify.h"
#include "local_ub_rma_buffer.h"
#include "socket.h"
#include "rtsq_a5.h"
#include "ascend_hal.h"
#include "drv_api_exception.h"
#include "ub_conn_lite_mgr.h"
#include "ins_to_sqe_rule.h"
#include "stream_lite.h"
#include "mem_transport_callback.h"
#undef private

using namespace Hccl;

static int memcpy_stub(void* dest, int dest_max, const void* src, int count)
{
    memcpy(dest, src, count);
    return 0;
}

class UbTransportLiteImplTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "UbTransportLiteImplTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "UbTransportLiteImplTest tests tear down." << std::endl; }

    virtual void SetUp()
    {
        MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_910A2));
        MOCKER_CPP(&RtsqBase::QuerySqBaseAddr)
            .stubs()
            .with(mockcpp::any())
            .will(returnValue(reinterpret_cast<u64>(&mockSq)));
        MOCKER_CPP(&RtsqBase::QuerySqStatusByType).stubs().with(mockcpp::any()).will(returnValue(0));
        MOCKER_CPP(&RtsqBase::ConfigSqStatusByType).stubs();
        std::cout << "A Test case in UbTransportLiteImplTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        std::cout << "A Test case in UbTransportLiteImplTest TearDown" << std::endl;
        GlobalMockObject::verify();
    }
    u8 mockSq[AC_SQE_SIZE * AC_SQE_MAX_CNT]{0};
};

std::vector<char> GetNotifyUniqueId(u32 notifyId, u32 devPhyId)
{
    BinaryStream binaryStream;
    binaryStream << notifyId;
    binaryStream << devPhyId;
    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

std::vector<char> GetRmtBufferUniqueId(u64 addr, u64 size, u32 tokenId, u32 tokenValue)
{
    BinaryStream binaryStream;
    binaryStream << addr;
    binaryStream << size;
    binaryStream << tokenId;
    binaryStream << tokenValue;
    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

std::vector<char> GetConnUniqueId()
{
    u32 dieId = 0;
    u32 funcId = 0;
    u32 jettyId = 0;
    u32 jfcPollMode = 0;          // 待修改，0代表STARS POLL，1代表software Poll
    bool dwqeCacheLocked = false; // 待修改，该jetty是否支持dwqeCachedLocked，默认不支持
    u64 dbAddr = 0x100;
    u64 sqVa = 0x100;
    u32 sqDepth = 100;
    u32 tpn = 100;
    Eid rmtEid;

    BinaryStream binaryStream;
    binaryStream << dieId;
    binaryStream << funcId;
    binaryStream << jettyId;

    binaryStream << jfcPollMode;
    binaryStream << dwqeCacheLocked;
    binaryStream << dbAddr;
    binaryStream << sqVa;
    binaryStream << sqDepth;
    binaryStream << tpn;
    binaryStream << rmtEid.raw;

    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

std::vector<char> BuildUbTransportLiteUniqueId()
{
    auto locNotify0 = GetNotifyUniqueId(1, 1);
    auto locNotify1 = GetNotifyUniqueId(2, 2);

    auto rmtNotify0 = GetRmtBufferUniqueId(1, 1, 1, 1);
    auto rmtNotify1 = GetRmtBufferUniqueId(2, 2, 2, 2);

    u32 notifyNum = 2;

    auto rmtBuffer0 = GetRmtBufferUniqueId(300, 200, 3, 3);
    auto rmtBuffer1 = GetRmtBufferUniqueId(300, 200, 4, 4);
    u32 bufferBum = 2;

    auto conn0 = GetConnUniqueId();
    u32 connNum = 1;

    BinaryStream binaryStream;
    u32 type = (u32)TransportType::UB;
    binaryStream << type;
    binaryStream << notifyNum;
    binaryStream << bufferBum;
    binaryStream << connNum;

    std::vector<char> data0;
    data0.insert(data0.end(), locNotify0.begin(), locNotify0.end());
    data0.insert(data0.end(), locNotify1.begin(), locNotify1.end());
    std::cout << "size0=" << data0.size() << endl;
    binaryStream << data0;

    std::vector<char> data1;
    data1.insert(data1.end(), rmtNotify0.begin(), rmtNotify0.end());
    data1.insert(data1.end(), rmtNotify1.begin(), rmtNotify1.end());
    std::cout << "size1=" << data1.size() << endl;
    binaryStream << data1;

    std::vector<char> data2;
    data2.insert(data2.end(), rmtBuffer0.begin(), rmtBuffer0.end());
    data2.insert(data2.end(), rmtBuffer1.begin(), rmtBuffer1.end());
    std::cout << "size2=" << data2.size() << endl;
    binaryStream << data2;

    std::vector<char> data3;
    data3.insert(data3.end(), conn0.begin(), conn0.end());
    binaryStream << data3;

    std::vector<char> liteData;
    binaryStream.Dump(liteData);
    return liteData;
}

TEST_F(UbTransportLiteImplTest, construct_test)
{
    std::vector<char> liteData = BuildUbTransportLiteUniqueId();

    RmaConnLite rmaConnLite;
    RmaConnLite* connLite = &rmaConnLite;
    MOCKER_CPP(&UbConnLiteMgr::Get).stubs().will(returnValue(connLite));
    MOCKER_CPP(&MirrorTaskManager::AddTaskInfo).stubs().with(mockcpp::any());
    LinkData linkData(BasePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB), 0, 1, 0, 1);
    MirrorTaskManager mirrorTaskMgr(0, &GlobalMirrorTasks::Instance(), true);
    auto transportCallback = MemTransportCallback(linkData, mirrorTaskMgr);
    MemTransportLite transportLite(liteData, transportCallback);
    auto& ubTransportLite = *(dynamic_cast<UbTransportLiteImpl*>(transportLite.impl.get()));
    transportLite.Describe();

    std::cout << ubTransportLite.Describe() << std::endl;

    std::cout << ubTransportLite.locNotifyVec[0]->Describe() << std::endl;
    std::cout << ubTransportLite.locNotifyVec[1]->Describe() << std::endl;

    std::cout << ubTransportLite.rmtNotifyVec[0].Describe() << std::endl;
    std::cout << ubTransportLite.rmtNotifyVec[1].Describe() << std::endl;

    std::cout << ubTransportLite.rmtBufferVec[0].Describe() << std::endl;
    std::cout << ubTransportLite.rmtBufferVec[1].Describe() << std::endl;

    u32 fakeStreamId = 1;
    u32 fakeSqId = 1;
    u32 fakedevPhyId = 1;
    BinaryStream liteBinaryStream;
    liteBinaryStream << fakeStreamId;
    liteBinaryStream << fakeSqId;
    liteBinaryStream << fakedevPhyId;
    std::vector<char> uniqueId{};
    liteBinaryStream.Dump(uniqueId);

    StreamLite stream(uniqueId);
    RtsqA5 rtsq(fakedevPhyId, fakeStreamId, fakeSqId);
    stream.rtsq = std::make_unique<RtsqA5>(rtsq);
    MOCKER_CPP_VIRTUAL(rtsq, &RtsqA5::SdmaCopy)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any());

    MOCKER_CPP(&UbTransportLiteImpl::BuildNotifyWaitTask).stubs();
    MOCKER_CPP(&UbTransportLiteImpl::BuildUbDbSendTask).stubs();

    transportLite.Post(0, stream);
    transportLite.Wait(0, stream);

    auto rmtBuffer = transportLite.GetRmtBuffer(0);

    RmaBufferLite locBuffer(100, 200, 300, 400);
    WithNotifyIn withNotify{TransportNotifyType::NORMAL, 0};

    transportLite.Read(locBuffer, rmtBuffer, stream);
    transportLite.Write(locBuffer, rmtBuffer, stream);
    ReduceIn reduceIn(DataType::INT8, ReduceOp::MAX);
    transportLite.ReadReduce(locBuffer, rmtBuffer, reduceIn, stream);
    transportLite.WriteReduce(locBuffer, rmtBuffer, reduceIn, stream);
    transportLite.WriteWithNotify(locBuffer, rmtBuffer, withNotify, stream);

    vector<RmaBufferLite> localRmaBufferVec;
    localRmaBufferVec.push_back(locBuffer);
    vector<Buffer> rmtBufferVec;
    rmtBufferVec.push_back(rmtBuffer);
    vector<BaseTransportLiteImpl::TransferOp> transferOpVec;
    transferOpVec.push_back({TransferType::READ, {DataType::INVALID, ReduceOp::INVALID}});
    transferOpVec.push_back({TransferType::READ, {DataType::INT8, ReduceOp::MAX}});
    transferOpVec.push_back({TransferType::WRITE, {DataType::INVALID, ReduceOp::INVALID}});
    transferOpVec.push_back({TransferType::WRITE, {DataType::INT8, ReduceOp::MAX}});
    transportLite.BatchTransfer(localRmaBufferVec, rmtBufferVec, transferOpVec, stream);

    RmaBufSliceLite locRmaBufferLite(100, 200, 300, 400);
    RmtRmaBufSliceLite rmtRmaBufferLite(100, 200, 300, 400, 500, UINT32_MAX);
    transportLite.BatchOneSidedRead({locRmaBufferLite}, {rmtRmaBufferLite}, stream);
    transportLite.BatchOneSidedWrite({locRmaBufferLite}, {rmtRmaBufferLite}, stream);
}

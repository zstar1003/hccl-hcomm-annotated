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
#include "stream_lite_mgr.h"
#include "ins_to_sqe_rule.h"
#include "rtsq_a5.h"
#include "dev_ub_connection.h"
#include "ub_local_notify.h"
#include "local_ub_rma_buffer.h"
#include "ub_transport_lite_impl.h"
#include "ub_mem_transport.h"
#include "mem_transport_lite.h"
#include "mem_transport_callback.h"
#include "rdma_handle_manager.h"
#include "dev_buffer.h"
#undef private
#undef protected
#include "null_ptr_exception.h"

using namespace Hccl;
class StreamLiteManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "StreamLiteManagerTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "StreamLiteManagerTest TearDown" << std::endl; }

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
        RdmaHandleManager::GetInstance().tokenInfoMap[rdmaHandle] = make_unique<TokenInfoManager>(0, rdmaHandle);

        LinkData linkData(BasePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB), 0, 1, 0, 1);
        DevUbConnection ubConnection((void*)0x100, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE);
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
        MirrorTaskManager mirrorTaskMgr(0, &GlobalMirrorTasks::Instance(), true);
        auto transportCallback = MemTransportCallback(linkData, mirrorTaskMgr);
        auto data = ubTransport.GetUniqueId();
        transportLite = std::make_unique<MemTransportLite>(data, transportCallback);

        std::cout << "A Test case in StreamLiteManagerTest SetUp" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in StreamLiteManagerTest TearDown" << std::endl;
    }

    u32 num1 = 1;
    u32 num2 = 2;
    u32 fakedevPhyId1 = 0;
    s32 fakeStreamId1 = 1;
    u32 fakeSqId1 = 2;
    u32 fakeNotifyId1 = 1;
    u32 fakedevPhyId2 = 1;
    s32 fakeStreamId2 = 2;
    u32 fakeSqId2 = 3;
    u32 fakeNotifyId2 = 2;
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

TEST_F(StreamLiteManagerTest, parse_packed_data)
{
    BinaryStream liteBinaryStream;
    liteBinaryStream << fakeStreamId1;
    liteBinaryStream << fakeSqId1;
    liteBinaryStream << fakedevPhyId1;
    liteBinaryStream << fakeStreamId2;
    liteBinaryStream << fakeSqId2;
    liteBinaryStream << fakedevPhyId2;
    std::vector<char> uniqueId{};
    liteBinaryStream.Dump(uniqueId);

    BinaryStream binaryStream;
    binaryStream << num2;
    binaryStream << uniqueId;
    std::vector<char> packedData;
    binaryStream.Dump(packedData);

    StreamLiteMgr mgr;
    mgr.ParsePackedData(packedData);

    EXPECT_EQ(fakeStreamId1, mgr.GetMaster()->GetId());
    EXPECT_EQ(fakeStreamId2, mgr.GetSlave(0)->GetId());

    EXPECT_EQ(1, mgr.SizeOfSlaves());

    mgr.ParsePackedData(packedData);
    mgr.Reset();
    EXPECT_EQ(nullptr, mgr.GetMaster());
}

TEST_F(StreamLiteManagerTest, update_reset)
{
    // 构造lite dto的序列化数据
    BinaryStream liteBinaryStream;
    liteBinaryStream << fakeStreamId1;
    liteBinaryStream << fakeSqId1;
    liteBinaryStream << fakedevPhyId1;
    std::vector<char> uniqueId{};
    liteBinaryStream.Dump(uniqueId);

    StreamLite stream(uniqueId);
    RtsqA5 rtsq(fakedevPhyId1, fakeStreamId1, fakeSqId1);
    stream.rtsq = std::make_unique<RtsqA5>(rtsq);
    MOCKER_CPP_VIRTUAL(rtsq, &RtsqA5::SdmaCopy)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any());

    BinaryStream binaryStream;
    binaryStream << num1;
    binaryStream << uniqueId;

    std::vector<char> packedData;
    binaryStream.Dump(packedData);
    std::cout << "packedData size = " << packedData.size() << std::endl;

    StreamLiteMgr mgr;
    mgr.ParsePackedData(packedData);

    EXPECT_EQ(stream.GetSqId(), mgr.GetMaster()->GetSqId());
    EXPECT_EQ(0, mgr.SizeOfSlaves());
    EXPECT_EQ(nullptr, mgr.GetSlave(0));

    mgr.Reset();
    EXPECT_EQ(nullptr, mgr.GetMaster());
}

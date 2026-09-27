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
#include "stub_communicator_impl_trans_mgr.h"
#include "internal_exception.h"
#include "dev_ub_connection.h"
#define private public
#include "mem_transport_lite_mgr.h"
#include "mem_transport_manager.h"
#include "ub_mem_transport.h"
#include "ub_conn_lite_mgr.h"
#include "p2p_transport.h"
#include "ub_local_notify.h"
#include "binary_stream.h"
#include "local_ub_rma_buffer.h"
#undef private

using namespace Hccl;

class StubDevUbConnection : public DevUbConnection {
public:
    StubDevUbConnection(const LinkData& linkData)
        : link(linkData),
          DevUbConnection((void*)0x100, linkData.GetLocalAddr(), linkData.GetRemoteAddr(), OpMode::OPBASE)
    {}

    RmaConnStatus GetStatus() override { return RmaConnStatus::INIT; }

private:
    LinkData link;
};

class MemTransportLiteMgrTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MemTransportLiteMgrTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "MemTransportLiteMgrTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in MemTransportLiteMgrTest SetUP" << std::endl;
        MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_910A2));
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in MemTransportLiteMgrTest TearDown" << std::endl;
    }

    std::vector<char> BuildMinimalUbTransportLiteUniqueId()
    {
        BinaryStream binaryStream;
        u32 type = (u32)TransportType::UB;
        binaryStream << type;
        binaryStream << (u32)0; // notifyNum=0
        binaryStream << (u32)0; // bufferNum=0
        binaryStream << (u32)0; // rmtbufferNum=0
        binaryStream << (u32)0; // connNum=0, skip ParseConnVec
        std::vector<char> emptyVec;
        binaryStream << emptyVec; // notifyUniqueIds
        binaryStream << emptyVec; // rmtNotifyUniqueIds
        binaryStream << emptyVec; // locBufferUniqueIds
        binaryStream << emptyVec; // rmtBufferUniqueIds
        binaryStream << emptyVec; // connUniqueIds
        std::vector<char> liteData;
        binaryStream.Dump(liteData);
        return liteData;
    }
};

TEST_F(MemTransportLiteMgrTest, test_get_and_reset)
{
    MirrorTaskManagerLite mirrorTaskMgrLite;
    MemTransportLiteMgr liteMgr(&mirrorTaskMgrLite);
    LinkData linkData(BasePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB), 0, 1, 0, 1);

    EXPECT_EQ(liteMgr.GetOpbase(linkData), nullptr);
    liteMgr.Reset();
}

TEST_F(MemTransportLiteMgrTest, test_parse_opbase_packed_data)
{
    RmaConnLite rmaConnLite;
    RmaConnLite* connLite = &rmaConnLite;
    MOCKER_CPP(&UbConnLiteMgr::Get).stubs().will(returnValue(connLite));
    std::vector<char> transportUniqueId = BuildMinimalUbTransportLiteUniqueId();

    LinkData linkData(BasePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB), 0, 1, 0, 1);
    std::vector<char> linkUniqueId = linkData.GetUniqueId();

    u32 mapSize = 1;
    BinaryStream binaryStream;
    binaryStream << mapSize;
    binaryStream << linkUniqueId;
    binaryStream << transportUniqueId;

    std::vector<char> packedData;
    binaryStream.Dump(packedData);

    MirrorTaskManagerLite mirrorTaskMgrLite;
    MemTransportLiteMgr liteMgr(&mirrorTaskMgrLite);

    liteMgr.ParseOpbasePackedData(packedData);
}

TEST_F(MemTransportLiteMgrTest, test_parse_offload_packed_data)
{
    RmaConnLite rmaConnLite;
    RmaConnLite* connLite = &rmaConnLite;
    MOCKER_CPP(&UbConnLiteMgr::Get).stubs().will(returnValue(connLite));
    std::vector<char> transportUniqueId = BuildMinimalUbTransportLiteUniqueId();

    LinkData linkData(BasePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB), 0, 1, 0, 1);
    std::vector<char> linkUniqueId = linkData.GetUniqueId();

    u32 mapSize = 1;
    BinaryStream binaryStream;
    binaryStream << mapSize;
    binaryStream << linkUniqueId;
    binaryStream << transportUniqueId;

    std::vector<char> packedData;
    binaryStream.Dump(packedData);

    MirrorTaskManagerLite mirrorTaskMgrLite;
    MemTransportLiteMgr liteMgr(&mirrorTaskMgrLite);

    const string opTag = "opTag";
    EXPECT_NO_THROW(liteMgr.ParseOffloadPackedData(opTag, packedData));
}

TEST_F(MemTransportLiteMgrTest, test_parse_all_packed_data)
{
    RmaConnLite rmaConnLite;
    RmaConnLite* connLite = &rmaConnLite;
    MOCKER_CPP(&UbConnLiteMgr::Get).stubs().will(returnValue(connLite));
    std::vector<char> transportUniqueId = BuildMinimalUbTransportLiteUniqueId();

    LinkData linkData(BasePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB), 0, 1, 0, 1);
    std::vector<char> linkUniqueId = linkData.GetUniqueId();

    u32 opbaseMapSize = 1;
    BinaryStream binaryStream;
    binaryStream << opbaseMapSize;
    binaryStream << linkUniqueId;
    binaryStream << transportUniqueId;

    u32 opTagSize = 1;
    const string opTag = "opTag";
    u32 offloadMapSize = 1;
    binaryStream << opTagSize;
    binaryStream << std::vector<char>(opTag.begin(), opTag.end());
    binaryStream << offloadMapSize;
    binaryStream << linkUniqueId;
    binaryStream << transportUniqueId;

    std::vector<char> packedData;
    binaryStream.Dump(packedData);

    MirrorTaskManagerLite mirrorTaskMgrLite;
    MemTransportLiteMgr liteMgr(&mirrorTaskMgrLite);
    EXPECT_NO_THROW(liteMgr.ParseAllPackedData(packedData));
}

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
#define protected public
#include "mem_transport_lite_mgr.h"
#include "mem_transport_manager.h"
#include "ub_mem_transport.h"
#include "p2p_transport.h"
#include "local_ub_rma_buffer.h"
#include "ub_local_notify.h"
#undef private
#undef protected

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
        MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in MemTransportLiteMgrTest TearDown" << std::endl;
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

std::vector<char> FwkGetNotifyUniqueId(u32 notifyId, u32 devPhyId)
{
    BinaryStream binaryStream;
    binaryStream << notifyId;
    binaryStream << devPhyId;
    std::vector<char> result;
    binaryStream.Dump(result);
    return result;
}

std::vector<char> FwkGetRmtBufferUniqueId(u64 addr, u64 size, u32 tokenId, u32 tokenValue)
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

std::vector<char> FwkGetConnUniqueId()
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

std::vector<char> FwkBuildUbTransportLiteUniqueId()
{
    auto locNotify0 = FwkGetNotifyUniqueId(1, 1);
    auto locNotify1 = FwkGetNotifyUniqueId(2, 2);

    auto rmtNotify0 = FwkGetRmtBufferUniqueId(1, 1, 1, 1);
    auto rmtNotify1 = FwkGetRmtBufferUniqueId(2, 2, 2, 2);

    u32 notifyNum = 2;

    auto rmtBuffer0 = FwkGetRmtBufferUniqueId(300, 200, 3, 3);
    auto rmtBuffer1 = FwkGetRmtBufferUniqueId(300, 200, 4, 4);
    u32 bufferBum = 2;

    auto conn0 = FwkGetConnUniqueId();
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

    std::vector<char> data4;
    data4.insert(data4.end(), conn0.begin(), conn0.end());
    binaryStream << data4;

    std::vector<char> liteData;
    binaryStream.Dump(liteData);
    return liteData;
}

TEST_F(MemTransportLiteMgrTest, test_parse_opbase_packed_data)
{
    std::vector<char> transportUniqueId = FwkBuildUbTransportLiteUniqueId();

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
    std::vector<char> transportUniqueId = FwkBuildUbTransportLiteUniqueId();

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
    liteMgr.ParseOffloadPackedData(opTag, packedData);
}

TEST_F(MemTransportLiteMgrTest, test_parse_all_packed_data)
{
    std::vector<char> transportUniqueId = FwkBuildUbTransportLiteUniqueId();

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

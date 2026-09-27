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
#include "mem_transport_lite_mgr.h"
#include "port.h"
#undef private

#include "communicator_impl.h"
#include "data_buf_manager.h"
#include "internal_exception.h"
#include "rmt_data_buffer_mgr.h"
#include "mem_transport_lite.h"

using namespace Hccl;
class RmtDataBufferMgrTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RmtDataBufferMgr tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "RmtDataBufferMgr tests tear down." << std::endl; }

    virtual void SetUp()
    {
        mirrorTaskMgrLite = new MirrorTaskManagerLite();
        memTransportLiteMgr = new MemTransportLiteMgr(mirrorTaskMgrLite);
        algInfo = new CollAlgInfo(mode, tag);
        std::cout << "A Test case in RmtDataBufferMgr SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        delete memTransportLiteMgr;
        delete mirrorTaskMgrLite;
        delete algInfo;
        GlobalMockObject::verify();
        std::cout << "A Test case in RmtDataBufferMgr TearDown" << std::endl;
    }

    MemTransportLiteMgr* memTransportLiteMgr;
    MirrorTaskManagerLite* mirrorTaskMgrLite;
    CollAlgInfo* algInfo;
    std::string tag = "tag";
    OpMode mode{OpMode::OPBASE};

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
        u32 buffeNum = 2;

        auto conn0 = GetConnUniqueId();
        u32 connNum = 1;

        BinaryStream binaryStream;

        u32 type = (u32)TransportType::UB;
        binaryStream << type;
        binaryStream << notifyNum;
        binaryStream << buffeNum;
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
};

TEST_F(RmtDataBufferMgrTest, get_GetBuffer_opbase_success)
{
    std::vector<char> liteData = BuildUbTransportLiteUniqueId();
    LinkData linkData(BasePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB), 0, 1, 0, 1);
    MirrorTaskManagerLite mirrorTaskMgrLite;
    auto transportCallbackLite = MemTransportCallbackLite(linkData, mirrorTaskMgrLite);
    std::unique_ptr<MemTransportLite> transportLite
        = std::make_unique<MemTransportLite>(liteData, transportCallbackLite);
    memTransportLiteMgr->opBaseTranspMap[linkData] = std::move(transportLite);
    RmtDataBufferMgr rmtDataBufferMgr(memTransportLiteMgr, algInfo);

    Buffer buffer1(0x100, 10);
    MOCKER_CPP(&MemTransportLite::GetRmtBuffer).stubs().with(mockcpp::any()).will(returnValue(buffer1));
    DataBuffer buffer2 = rmtDataBufferMgr.GetBuffer(linkData, BufferType::SCRATCH);
    std::cout << "buffer2 addr: " << buffer2.GetAddr() << std::endl;
    std::cout << "buffer2 size: " << buffer2.GetSize() << std::endl;
    EXPECT_EQ(buffer2.GetAddr(), buffer1.GetAddr());
    EXPECT_EQ(buffer2.GetSize(), buffer1.GetSize());
}

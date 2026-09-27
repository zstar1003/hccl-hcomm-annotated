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
#include "peer_info.h"
#include "json_parser.h"
#include "orion_adapter_rts.h"

using namespace Hccl;

class PeerParserTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "PeerParserTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "PeerParserTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in PeerParserTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in PeerParserTest TearDown" << std::endl;
    }
};

// 功能用例
TEST_F(PeerParserTest, St_Deserialize_When_Normal_Expect_Success)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string peerString = R"({"local_id" : 1})";
    JsonParser peerParser;
    PeerInfo peerInfo;
    peerParser.ParseString(peerString, peerInfo);

    PeerInfo peer;
    peer.localId = 1;

    EXPECT_EQ(peerInfo.Describe(), peer.Describe());
}

TEST_F(PeerParserTest, St_Deserialize_When_LocalIdMissing_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string peerString = R"({"id" : "1"})";
    JsonParser peerParser;
    PeerInfo peerInfo;
    EXPECT_THROW(peerParser.ParseString(peerString, peerInfo), InvalidParamsException);
}

TEST_F(PeerParserTest, St_Deserialize_When_InvalidLocalId_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string peerString = R"({"local_id" : "-1"})";
    JsonParser peerParser;
    PeerInfo peerInfo;
    EXPECT_THROW(peerParser.ParseString(peerString, peerInfo), InvalidParamsException);
}

TEST_F(PeerParserTest, St_BinaryStream_When_GetBinStreamToReBuild_Expect_Success)
{
    PeerInfo peerInfo;
    peerInfo.localId = 2;
    BinaryStream binStream;
    peerInfo.GetBinStream(binStream);
    PeerInfo reBuildPeer(binStream);

    ASSERT_EQ(peerInfo.Describe(), reBuildPeer.Describe());
}

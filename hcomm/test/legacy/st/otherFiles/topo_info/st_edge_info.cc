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
#include "edge_info.h"
#include "json_parser.h"
#include "orion_adapter_rts.h"
#include "invalid_params_exception.h"
#include "exception_util.h"
using namespace Hccl;

class EdgeParserTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "EdgeParserTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "EdgeParserTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in EdgeParserTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in EdgeParserTest TearDown" << std::endl;
    }
};

// 功能用例
TEST_F(EdgeParserTest, St_Deserialize_When_Normal_Expect_Success)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string edgeString = R"({
            "net_layer": 0,
            "link_type": "PEER2PEER",
			"protocols": ["UB_CTP"],   
            "topo_type": "1DMESH",
            "topo_instance_id": 0,
			"local_a": 0,
			"local_a_ports": ["0/0"],
			"local_b": 1,
			"local_b_ports": ["0/1"],
			"position": "DEVICE"
		})";

    JsonParser topoParser;
    EdgeInfo edgeInfo;
    topoParser.ParseString(edgeString, edgeInfo);

    EdgeInfo edge0;
    edge0.netLayer = 0;
    edge0.protocols.emplace(LinkProtocol::UB_CTP);
    edge0.linkType = LinkType::PEER2PEER;
    edge0.topoType = TopoType::MESH_1D;
    edge0.topoInstId = 0;
    edge0.localA = 0;
    edge0.localAPorts.emplace("0/0");
    edge0.localB = 1;
    edge0.localBPorts.emplace("0/1");
    edge0.position = AddrPosition::DEVICE;

    EXPECT_EQ(edgeInfo.Describe(), edge0.Describe());
}

// 功能用例，缺省
TEST_F(EdgeParserTest, St_Deserialize_When_OptionalFieldsMissing_Expect_Success)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string edgeString = R"({
            "net_layer": 0,
            "link_type": "PEER2PEER",
			"protocols": ["UB_CTP"],   
			"local_a": 0,
			"local_a_ports": ["0/0"],
			"local_b": 1,
			"local_b_ports": ["0/1"],
			"position": "DEVICE"
		})";

    JsonParser topoParser;
    EdgeInfo edgeInfo;
    topoParser.ParseString(edgeString, edgeInfo);

    EdgeInfo edge0;
    edge0.netLayer = 0;
    edge0.protocols.emplace(LinkProtocol::UB_CTP);
    edge0.linkType = LinkType::PEER2PEER;
    edge0.topoType = TopoType::CLOS;
    edge0.topoInstId = 0;
    edge0.localA = 0;
    edge0.localAPorts.emplace("0/0");
    edge0.localB = 1;
    edge0.localBPorts.emplace("0/1");
    edge0.position = AddrPosition::DEVICE;

    EXPECT_EQ(edgeInfo.Describe(), edge0.Describe());
}

// PEER2NET功能用例，存在localB，应告警
TEST_F(EdgeParserTest, St_Deserialize_When_PEER2NET_ExistB_Expect_Warning)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string edgeString = R"({
            "net_layer": 0,
            "link_type": "PEER2NET",
			"protocols": ["UB_CTP"],   
            "topo_type": "1DMESH",
            "topo_instance_id": 0,
			"local_a": 0,
			"local_a_ports": ["0/0"],
			"local_b": 1,
			"local_b_ports": ["0/1"],
			"position": "DEVICE"
		})";

    JsonParser topoParser;
    EdgeInfo edgeInfo;
    topoParser.ParseString(edgeString, edgeInfo);

    EdgeInfo edge0;
    edge0.netLayer = 0;
    edge0.protocols.emplace(LinkProtocol::UB_CTP);
    edge0.linkType = LinkType::PEER2NET;
    edge0.topoType = TopoType::MESH_1D;
    edge0.topoInstId = 0;
    edge0.localA = 0;
    edge0.localAPorts.emplace("0/0");
    edge0.position = AddrPosition::DEVICE;

    EXPECT_EQ(edgeInfo.Describe(), edge0.Describe());
}

// PEER2NET功能用例，不提供B端口数据
TEST_F(EdgeParserTest, St_Deserialize_When_NormalPeer2Net_Expect_Success)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string edgeString = R"({
            "net_layer": 0,
            "link_type": "PEER2NET",
			"protocols": ["UB_CTP"],   
            "topo_type": "1DMESH",
            "topo_instance_id": 0,
			"local_a": 0,
			"local_a_ports": ["0/0"],
			"position": "DEVICE"
		})";

    JsonParser topoParser;
    EdgeInfo edgeInfo;
    topoParser.ParseString(edgeString, edgeInfo);

    EdgeInfo edge0;
    edge0.netLayer = 0;
    edge0.protocols.emplace(LinkProtocol::UB_CTP);
    edge0.linkType = LinkType::PEER2NET;
    edge0.topoType = TopoType::MESH_1D;
    edge0.topoInstId = 0;
    edge0.localA = 0;
    edge0.localAPorts.emplace("0/0");
    edge0.position = AddrPosition::DEVICE;

    EXPECT_EQ(edgeInfo.Describe(), edge0.Describe());
}

// 缺少字段
TEST_F(EdgeParserTest, St_Deserialize_When_NeededFieldMissing_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string edgeString = R"({
			"net_layer": 0,
			"link_type": "PEER2PEER",
			"protocols": ["UB_CTP"]  
		})";

    JsonParser edgeParser;
    EdgeInfo edgeInfo;
    EXPECT_THROW(edgeParser.ParseString(edgeString, edgeInfo), InvalidParamsException);
}

// net_layer = 8
TEST_F(EdgeParserTest, St_Deserialize_When_InvalidNetLayer_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string edgeString = R"({
			"net_layer": 8,
			"link_type": "PEER2PEER",
			"protocols": ["UB_CTP"],
            "topo_type": "1DMESH",
            "topo_instance_id": 0,
			"local_a": 0,
			"local_a_ports": ["0/0"],
			"local_b": 1,
			"local_b_ports": ["0/1"],
			"position": "DEVICE"
		})";

    JsonParser edgeParser;
    EdgeInfo edgeInfo;
    EXPECT_THROW(edgeParser.ParseString(edgeString, edgeInfo), InvalidParamsException);
}

// 无效的LinkProtocol
TEST_F(EdgeParserTest, St_Deserialize_When_InvalidLinkProtocol_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string edgeString = R"({
			"net_layer": 4,
			"link_type": "PEER2PEER",
			"protocols": ["UDP"],
            "topo_type": "1DMESH",
            "topo_instance_id": 0,
			"local_a": 0,
			"local_a_ports": ["0/0"],
			"local_b": 1,
			"local_b_ports": ["0/1"],
			"position": "DEVICE"
		})";

    JsonParser edgeParser;
    EdgeInfo edgeInfo;
    EXPECT_THROW(edgeParser.ParseString(edgeString, edgeInfo), InvalidParamsException);
}

// LinkProtocol过多
TEST_F(EdgeParserTest, St_Deserialize_When_ToManyLinkProtocols_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string edgeString = R"({
			"net_layer": 2,
			"link_type": "PEER2PEER",
			"protocols": ["UB_CTP", "ROCE"],
            "topo_type": "1DMESH",
            "topo_instance_id": 0,
			"local_a": 0,
			"local_a_ports": ["0/0"],
			"local_b": 1,
			"local_b_ports": ["0/1"],
			"position": "DEVICE"
		})";

    JsonParser edgeParser;
    EdgeInfo edgeInfo;
    edgeParser.ParseString(edgeString, edgeInfo);

    EdgeInfo edge0;
    edge0.netLayer = 2;
    edge0.protocols.emplace(LinkProtocol::UB_CTP);
    edge0.protocols.emplace(LinkProtocol::ROCE);
    edge0.linkType = LinkType::PEER2PEER;
    edge0.topoType = TopoType::MESH_1D;
    edge0.topoInstId = 0;
    edge0.localA = 0;
    edge0.localAPorts.emplace("0/0");
    edge0.localB = 1;
    edge0.localBPorts.emplace("0/1");
    edge0.position = AddrPosition::DEVICE;

    EXPECT_EQ(edgeInfo.Describe(), edge0.Describe());
}

// 无效的TopoType
TEST_F(EdgeParserTest, St_Deserialize_When_InvalidTopoType_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string edgeString = R"({
			"net_layer": 2,
			"link_type": "PEER2PEER",
			"protocols": ["UB_CTP"],
            "topo_type": "1DMesh",
            "topo_instance_id": 0,
			"local_a": 0,
			"local_a_ports": ["0/0"],
			"local_b": 1,
			"local_b_ports": ["0/1"],
			"position": "DEVICE"
		})";

    JsonParser edgeParser;
    EdgeInfo edgeInfo;
    EXPECT_THROW(edgeParser.ParseString(edgeString, edgeInfo), InvalidParamsException);
}

// 无效的LinkType
TEST_F(EdgeParserTest, St_Deserialize_When_InvalidLinkType_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string edgeString = R"({
			"net_layer": 2,
			"link_type": "PEER2BACKUP",
			"protocols": ["UB_CTP"],
            "topo_type": "1DMESH",
            "topo_instance_id": 0,
			"local_a": 0,
			"local_a_ports": ["0/0"],
			"local_b": 1,
			"local_b_ports": ["0/1"],
			"position": "DEVICE"
		})";

    JsonParser edgeParser;
    EdgeInfo edgeInfo;
    EXPECT_THROW(edgeParser.ParseString(edgeString, edgeInfo), InvalidParamsException);
}

// PEER2PEER下，localA==localB
TEST_F(EdgeParserTest, St_Deserialize_When_PEER2PEERHasSameEndPoint_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string edgeString = R"({
			"net_layer": 2,
			"link_type": "PEER2PEER",
			"protocols": ["UB_CTP"],
            "topo_type": "1DMESH",
            "topo_instance_id": 0,
			"local_a": 0,
			"local_a_ports": ["0/0"],
			"local_b": 0,
			"local_b_ports": ["0/1"],
			"position": "DEVICE"
		})";

    JsonParser edgeParser;
    EdgeInfo edgeInfo;
    EXPECT_THROW(edgeParser.ParseString(edgeString, edgeInfo), InvalidParamsException);
}

// PEER2PEER ， 缺少localB
TEST_F(EdgeParserTest, St_Deserialize_When_PEER2PEERlocalBMissing_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string edgeString = R"({
			"net_layer": 2,
			"link_type": "PEER2PEER",
			"protocols": ["UB_CTP"],
            "topo_type": "1DMESH",
            "topo_instance_id": 0,
			"local_a": 0,
			"local_a_ports": ["0/0"],
			"local_b_ports": ["0/1"],
			"position": "DEVICE"
		})";

    JsonParser edgeParser;
    EdgeInfo edgeInfo;
    EXPECT_THROW(edgeParser.ParseString(edgeString, edgeInfo), InvalidParamsException);
}

// port长度非法
TEST_F(EdgeParserTest, St_Deserialize_When_InvalidLengthOfPort_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string edgeString = R"({
			"net_layer": 2,
			"link_type": "PEER2PEER",
			"protocols": ["UB_CTP"],
            "topo_type": "1DMESH",
            "topo_instance_id": 0,
			"local_a": 0,
			"local_a_ports": ["9999999999999999/9999999999999999"],
			"local_b_ports": ["0/1"],
			"position": "DEVICE"
		})";

    JsonParser edgeParser;
    EdgeInfo edgeInfo;
    EXPECT_THROW(edgeParser.ParseString(edgeString, edgeInfo), InvalidParamsException);
}

// position非法
TEST_F(EdgeParserTest, St_Deserialize_When_InvalidPosition_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string edgeString = R"({
			"net_layer": 2,
			"link_type": "PEER2PEER",
			"protocols": ["UB_CTP"],
            "topo_type": "1DMESH",
            "topo_instance_id": 0,
			"local_a": 0,
			"local_a_ports": ["0/0"],
            "local_b": 1,
			"local_b_ports": ["0/1"],
			"position": "DECADE"
		})";

    JsonParser edgeParser;
    EdgeInfo edgeInfo;
    EXPECT_THROW(edgeParser.ParseString(edgeString, edgeInfo), InvalidParamsException);
}

TEST_F(EdgeParserTest, St_BinaryStream_When_GetBinStreamToReBuild_Expect_Success)
{
    EdgeInfo edge0;
    edge0.netLayer = 0;
    edge0.protocols.emplace(LinkProtocol::UB_CTP);
    edge0.linkType = LinkType::PEER2PEER;
    edge0.topoType = TopoType::MESH_1D;
    edge0.topoInstId = 0;
    edge0.localA = 0;
    edge0.localAPorts.emplace("0/0");
    edge0.localAPorts.emplace("0/1");
    edge0.localB = 1;
    edge0.localBPorts.emplace("0/2");
    edge0.position = AddrPosition::DEVICE;

    BinaryStream binStream;
    edge0.GetBinStream(binStream);

    EdgeInfo edgeInfoRebuild(binStream);
    EXPECT_EQ(edgeInfoRebuild.Describe(), edge0.Describe());
}

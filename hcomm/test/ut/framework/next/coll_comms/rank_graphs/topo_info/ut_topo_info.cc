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
#include <nlohmann/json.hpp>
#include "topo_info.h"
#include "json_parser.h"
#include "orion_adapter_rts.h"
#include "invalid_params_exception.h"
#include "exception_util.h"
using namespace Hccl;

namespace {
std::string BuildLargeTopoString()
{
    constexpr u32 peerCount = 64;
    nlohmann::json topoJson;
    topoJson["version"] = "2.0";
    topoJson["hardwareType"] = "Atlas 950 SuperPod 2D";
    topoJson["peer_count"] = peerCount;
    topoJson["edge_count"] = peerCount;
    topoJson["peer_list"] = nlohmann::json::array();
    topoJson["edge_list"] = nlohmann::json::array();

    for (u32 localId = 0; localId < peerCount; ++localId) {
        topoJson["peer_list"].push_back({{"local_id", localId}});
        topoJson["edge_list"].push_back({
            {"net_layer", 2},
            {"link_type", "PEER2NET"},
            {"topo_type", "CLOS"},
            {"topo_instance_id", 0},
            {"topo_attr", ""},
            {"local_a", localId},
            {"local_a_ports", {"0/1", "0/2", "1/1", "1/2"}},
            {"protocols", {"UB_TP"}},
            {"position", "DEVICE"},
        });
    }
    return topoJson.dump();
}
} // namespace

class TopoParserTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "TopoParserTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "TopoParserTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in TopoParserTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in TopoParserTest TearDown" << std::endl;
    }
};

// 功能用例，PEER2NET的B端口缺省，topoType和topoInstId缺省，正常填写
TEST_F(TopoParserTest, Ut_Deserialize_When_Normal_Expect_Success)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string topoString = R"({
	    "version": "2.0",
	    "hardware_type" : "950-2D-Fullmsh_64_plus_1",
	    "peer_count" : 3,
        "peer_list" :[
		    { "local_id" : 0 },
		    { "local_id" : 1 },
		    { "local_id" : 2 }
	    ],
	    "edge_count" : 5,
        "edge_list": [
		    {
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
		    },
		    {
			    "net_layer": 0,
                "link_type": "PEER2PEER",
			    "protocols": ["UB_MEM"],
                "topo_type": "1DMESH",
                "topo_instance_id": 0,
			    "local_a": 0,
			    "local_a_ports": ["0/0"],
			    "local_b": 2,
			    "local_b_ports": ["0/1"],
			    "position": "DEVICE"
		    },
		    {
			    "net_layer": 0,
                "link_type": "PEER2NET",
			    "protocols": ["UB_CTP"],
                "topo_type": "1DMESH",
                "topo_instance_id": 0,
			    "local_a": 0,
			    "local_a_ports": ["0/0"],
			    "position": "HOST"
		    },
		    {
                "net_layer": 1,
                "link_type": "PEER2PEER",
			    "protocols": ["UB_TP"],
                "topo_type": "1DMESH",
                "topo_instance_id": 0,
			    "local_a": 1,
			    "local_a_ports": ["0/0"],
			    "local_b": 2,
			    "local_b_ports": ["0/1"],
			    "position": "DEVICE"
		    },
		    {
                "net_layer": 2,
                "link_type": "PEER2NET",
			    "protocols": ["ROCE"],
			    "local_a": 0,
			    "local_a_ports": ["0/0"],
			    "position": "DEVICE"
		    }
	    ]
        })";

    JsonParser topoParser;
    TopoInfo topoInfo;
    topoParser.ParseString(topoString, topoInfo);

    TopoInfo expectTopoInfo;
    expectTopoInfo.version = "2.0";
    expectTopoInfo.peerCount = 3;

    PeerInfo peer0;
    peer0.localId = 0;
    PeerInfo peer1;
    peer1.localId = 1;
    PeerInfo peer2;
    peer2.localId = 2;
    expectTopoInfo.peers.emplace_back(peer0);
    expectTopoInfo.peers.emplace_back(peer1);
    expectTopoInfo.peers.emplace_back(peer2);

    expectTopoInfo.edgeCount = 5;
    expectTopoInfo.edges[0] = std::vector<EdgeInfo>();
    expectTopoInfo.edges[1] = std::vector<EdgeInfo>();
    expectTopoInfo.edges[2] = std::vector<EdgeInfo>();
    EdgeInfo edge0;
    edge0.netLayer = 0;
    edge0.linkType = LinkType::PEER2PEER;
    edge0.protocols.emplace(LinkProtocol::UB_CTP);
    edge0.topoType = TopoType::MESH_1D;
    edge0.topoInstId = 0;
    edge0.localA = 0;
    edge0.localAPorts.emplace("0/0");
    edge0.localB = 1;
    edge0.localBPorts.emplace("0/1");
    edge0.position = AddrPosition::DEVICE;
    expectTopoInfo.edges[0].emplace_back(edge0);

    EdgeInfo edge1;
    edge1.netLayer = 0;
    edge1.linkType = LinkType::PEER2PEER;
    edge1.protocols.emplace(LinkProtocol::UB_MEM);
    edge1.topoType = TopoType::MESH_1D;
    edge1.topoInstId = 0;
    edge1.localA = 0;
    edge1.localAPorts.emplace("0/0");
    edge1.localB = 2;
    edge1.localBPorts.emplace("0/1");
    edge1.position = AddrPosition::DEVICE;
    expectTopoInfo.edges[0].emplace_back(edge1);

    EdgeInfo edge2;
    edge2.netLayer = 0;
    edge2.linkType = LinkType::PEER2NET;
    edge2.protocols.emplace(LinkProtocol::UB_CTP);
    edge2.topoType = TopoType::MESH_1D;
    edge2.topoInstId = 0;
    edge2.localA = 0;
    edge2.localAPorts.emplace("0/0");
    edge2.position = AddrPosition::HOST;
    expectTopoInfo.edges[0].emplace_back(edge2);

    EdgeInfo edge3;
    edge3.netLayer = 1;
    edge3.linkType = LinkType::PEER2PEER;
    edge3.protocols.emplace(LinkProtocol::UB_TP);
    edge3.topoType = TopoType::MESH_1D;
    edge3.topoInstId = 0;
    edge3.localA = 1;
    edge3.localAPorts.emplace("0/0");
    edge3.localB = 2;
    edge3.localBPorts.emplace("0/1");
    edge3.position = AddrPosition::DEVICE;
    expectTopoInfo.edges[1].emplace_back(edge3);

    EdgeInfo edge4;
    edge4.netLayer = 2;
    edge4.linkType = LinkType::PEER2NET;
    edge4.protocols.emplace(LinkProtocol::ROCE);
    edge4.topoType = TopoType::CLOS;
    edge4.topoInstId = 0;
    edge4.localA = 0;
    edge4.localAPorts.emplace("0/0");
    edge4.position = AddrPosition::DEVICE;
    expectTopoInfo.edges[2].emplace_back(edge4);

    EXPECT_EQ(topoInfo.version, expectTopoInfo.version);
    EXPECT_EQ(topoInfo.peerCount, expectTopoInfo.peerCount);
    EXPECT_EQ(topoInfo.peers.size(), expectTopoInfo.peers.size());
    for (u32 i = 0; i < topoInfo.peers.size(); i++) {
        EXPECT_EQ(topoInfo.peers[i].localId, expectTopoInfo.peers[i].localId);
    }
    EXPECT_EQ(topoInfo.edgeCount, expectTopoInfo.edgeCount);
    EXPECT_EQ(topoInfo.edges.size(), expectTopoInfo.edges.size());

    auto it_topo_edges = topoInfo.edges.begin();
    auto it_expect_edges = expectTopoInfo.edges.begin();
    for (; it_topo_edges != topoInfo.edges.end(); it_topo_edges++, it_expect_edges++) {
        EXPECT_EQ(it_topo_edges->first, it_expect_edges->first);
        EXPECT_EQ((it_topo_edges->second).size(), (it_expect_edges->second).size());
        for (u32 i = 0; i < (it_topo_edges->second).size(); i++) {
            EXPECT_EQ((it_topo_edges->second)[i].netLayer, (it_expect_edges->second)[i].netLayer);
            EXPECT_EQ((it_topo_edges->second)[i].protocols, (it_expect_edges->second)[i].protocols);
            EXPECT_EQ((it_topo_edges->second)[i].linkType, (it_expect_edges->second)[i].linkType);
            EXPECT_EQ((it_topo_edges->second)[i].topoType, (it_expect_edges->second)[i].topoType);
            EXPECT_EQ((it_topo_edges->second)[i].topoInstId, (it_expect_edges->second)[i].topoInstId);
            EXPECT_EQ((it_topo_edges->second)[i].localA, (it_expect_edges->second)[i].localA);
            EXPECT_EQ((it_topo_edges->second)[i].localAPorts, (it_expect_edges->second)[i].localAPorts);
            EXPECT_EQ((it_topo_edges->second)[i].localB, (it_expect_edges->second)[i].localB);
            EXPECT_EQ((it_topo_edges->second)[i].localBPorts, (it_expect_edges->second)[i].localBPorts);
            EXPECT_EQ((it_topo_edges->second)[i].position, (it_expect_edges->second)[i].position);
        }
    }

    EXPECT_EQ(topoInfo.Describe(), expectTopoInfo.Describe());
}

// 有效字段缺少
TEST_F(TopoParserTest, Ut_Deserialize_When_NeededFieldMissing_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string topoString = R"({
	    "hardware_type" : "950-2D-Fullmsh_64_plus_1"
        })";

    JsonParser topoParser;
    TopoInfo topoInfo;
    EXPECT_THROW(topoParser.ParseString(topoString, topoInfo), InvalidParamsException);
}

// version = 1.0
TEST_F(TopoParserTest, Ut_Deserialize_When_InvalidVersion_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string topoString = R"({
	    "version": "1.0",
	    "hardware_type" : "950-2D-Fullmsh_64_plus_1",
	    "peer_count" : 3,
        "peer_list" :[
		    { "id" : 0 },
		    { "id" : 1 },
		    { "id" : 2 }
	    ],
	    "edge_count" : 1,
        "edge_list": [
		    {
			    "net_layer": 0,
                "link_type": "PEER2PEER",
			    "protocols": ["UB_CTP"],
                "topo_type": "1DMESH",
                "topo_instance_id": 2,
			    "local_a": 0,
			    "local_a_ports": ["0/0"],
			    "local_b": 1,
			    "local_b_ports": ["0/1"],
			    "position": "DEVICE"
		    }
	    ]
        })";

    JsonParser topoParser;
    TopoInfo topoInfo;
    EXPECT_THROW(topoParser.ParseString(topoString, topoInfo), InvalidParamsException);
}

// warning, edge 为 0
TEST_F(TopoParserTest, Ut_Deserialize_When_ZeroEdge_Expect_Warning)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string topoString = R"({
	    "version": "2.0",
	    "peer_count" : 3,
        "peer_list" :[
		    { "local_id" : 0 },
		    { "local_id" : 1 },
		    { "local_id" : 2 }
	    ],
	    "edge_count" : 0,
        "edge_list": []
        })";
    JsonParser topoParser;
    TopoInfo topoInfo;
    topoParser.ParseString(topoString, topoInfo);

    TopoInfo expectTopoInfo;
    expectTopoInfo.version = "2.0";
    expectTopoInfo.peerCount = 3;

    PeerInfo peer0;
    peer0.localId = 0;
    PeerInfo peer1;
    peer1.localId = 1;
    PeerInfo peer2;
    peer2.localId = 2;
    expectTopoInfo.peers.emplace_back(peer0);
    expectTopoInfo.peers.emplace_back(peer1);
    expectTopoInfo.peers.emplace_back(peer2);

    expectTopoInfo.edgeCount = 0;

    EXPECT_EQ(topoInfo.version, expectTopoInfo.version);
    EXPECT_EQ(topoInfo.peerCount, expectTopoInfo.peerCount);
    EXPECT_EQ(topoInfo.peers.size(), expectTopoInfo.peers.size());
    for (u32 i = 0; i < topoInfo.peers.size(); i++) {
        EXPECT_EQ(topoInfo.peers[i].localId, expectTopoInfo.peers[i].localId);
    }
    EXPECT_EQ(topoInfo.edgeCount, expectTopoInfo.edgeCount);
    EXPECT_EQ(topoInfo.edges.size(), expectTopoInfo.edges.size());
}

// peer_count != peer_list.size()
TEST_F(TopoParserTest, Ut_Deserialize_When_PeersSizeUnequalToPeerCount_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string topoString = R"({
	    "version": "2.0",
	    "peer_count" : 10,
        "edge_count" : 0,
        "peer_list" :[
		    { "local_id" : 0 },
		    { "local_id" : 1 },
		    { "local_id" : 2 }
	    ]
        })";

    JsonParser topoParser;
    TopoInfo topoInfo;
    EXPECT_THROW(topoParser.ParseString(topoString, topoInfo), InvalidParamsException);
}

// peer.loadId >= peer_count
TEST_F(TopoParserTest, Ut_Deserialize_When_PeerIdGreaterThanPeerCount_Expect_Success)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string topoString = R"({
	    "version": "2.0",
	    "peer_count" : 3,
		"edge_count" : 0,
        "peer_list" :[
		    { "local_id" : 4 },
		    { "local_id" : 0 },
		    { "local_id" : 2 }
	    ],
      "edge_list": []
        })";

    JsonParser topoParser;
    TopoInfo topoInfo;
    topoParser.ParseString(topoString, topoInfo);
    EXPECT_EQ(topoInfo.version, "2.0");
    EXPECT_EQ(topoInfo.peerCount, 3);
    EXPECT_EQ(topoInfo.peers.size(), 3);
    EXPECT_EQ(topoInfo.edgeCount, 0);
    EXPECT_EQ(topoInfo.edges.size(), 0);
}

// 重复的peer
TEST_F(TopoParserTest, Ut_Deserialize_When_DuplicatePeer_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string topoString = R"({
	    "version": "2.0",
	    "peer_count" : 3,
		"edge_count" : 0,
        "peer_list" :[
		    { "local_id" : 0 },
		    { "local_id" : 0 },
		    { "local_id" : 2 }
	    ]
        })";

    JsonParser topoParser;
    TopoInfo topoInfo;
    EXPECT_THROW(topoParser.ParseString(topoString, topoInfo), InvalidParamsException);
}

// edge_count != edge_list.size()
TEST_F(TopoParserTest, Ut_Deserialize_When_EdgesSizeUnequalToEdgeCount_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string topoString = R"({
	    "version": "2.0",
	    "hardware_type" : "950-2D-Fullmsh_64_plus_1",
	    "peer_count" : 3,
        "peer_list" :[
		    { "local_id" : 0 },
		    { "local_id" : 1 },
		    { "local_id" : 2 }
	    ],
	    "edge_count" : 5,
        "edge_list": [
		    {
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
		    },
		    {
                "net_layer": 0,
                "link_type": "PEER2PEER",
			    "protocols": ["UB_CTP"],
                "topo_type": "1DMESH",
                "topo_instance_id": 0,
			    "local_a": 0,
			    "local_a_ports": ["0/0"],
			    "local_b": 2,
			    "local_b_ports": ["0/1"],
			    "position": "DEVICE"
		    }
			]
        })";

    JsonParser topoParser;
    TopoInfo topoInfo;
    EXPECT_THROW(topoParser.ParseString(topoString, topoInfo), InvalidParamsException);
}

// 重复的边 PEER2PEER，localA和localB对调
TEST_F(TopoParserTest, Ut_Deserialize_When_DuplicateEdge_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string topoString = R"({
	    "version": "2.0",
	    "hardware_type" : "950-2D-Fullmsh_64_plus_1",
	    "peer_count" : 3,
        "peer_list" :[
		    { "local_id" : 0 },
		    { "local_id" : 1 },
		    { "local_id" : 2 }
	    ],
	    "edge_count" : 2,
        "edge_list": [
		    {
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
		    },
		    {
                "net_layer": 0,
                "link_type": "PEER2PEER",
			    "protocols": ["UB_CTP"],
                "topo_type": "1DMESH",
                "topo_instance_id": 0,
			    "local_a": 1,
			    "local_a_ports": ["0/1"],
			    "local_b": 0,
			    "local_b_ports": ["0/0"],
			    "position": "DEVICE"
		    }
			]
        })";

    JsonParser topoParser;
    TopoInfo topoInfo;
    EXPECT_THROW(topoParser.ParseString(topoString, topoInfo), InvalidParamsException);
}

// Endpoint的localId无效
TEST_F(TopoParserTest, Ut_Deserialize_When_InvalidEndpointLocalId_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string topoString = R"({
	    "version": "2.0",
	    "hardware_type" : "950-2D-Fullmsh_64_plus_1",
	    "peer_count" : 3,
        "peer_list" :[
		    { "local_id" : 0 },
		    { "local_id" : 1 },
		    { "local_id" : 2 }
	    ],
	    "edge_count" : 2,
        "edge_list": [
		    {
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
		    },
		    {
                "net_layer": 0,
                "link_type": "PEER2PEER",
			    "protocols": ["UB_CTP"],
                "topo_type": "1DMESH",
                "topo_instance_id": 0,
			    "local_a": 1,
			    "local_a_ports": ["0/1"],
			    "local_b": 3,
			    "local_b_ports": ["0/0"],
			    "position": "DEVICE"
		    }
			]
        })";

    JsonParser topoParser;
    TopoInfo topoInfo;
    EXPECT_THROW(topoParser.ParseString(topoString, topoInfo), InvalidParamsException);
}

// 可缺省字段填无效值 "MESH"
TEST_F(TopoParserTest, Ut_Deserialize_When_InvalidTopoType_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string topoString = R"({
	    "version": "2.0",
	    "hardware_type" : "950-2D-Fullmsh_64_plus_1",
	    "peer_count" : 3,
        "peer_list" :[
		    { "local_id" : 0 },
		    { "local_id" : 1 },
		    { "local_id" : 2 }
	    ],
	    "edge_count" : 2,
        "edge_list": [
		    {
                "net_layer": 0,
                "link_type": "PEER2PEER",
			    "protocols": ["UB_CTP"],
                "topo_type": "MESH",
                "topo_instance_id": 0,
			    "local_a": 0,
			    "local_a_ports": ["0/0"],
			    "local_b": 1,
			    "local_b_ports": ["0/1"],
			    "position": "DEVICE"
		    },
		    {
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
		    }
			]
        })";

    JsonParser topoParser;
    TopoInfo topoInfo;
    EXPECT_THROW(topoParser.ParseString(topoString, topoInfo), InvalidParamsException);
}

// 无效的JSON文件
TEST_F(TopoParserTest, Ut_Deserialize_When_InvalidJson_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string topoString = R"({
	    "net_layer": 0,
        "link_type": "PEER2PEER",
		"protocols": ["UB_CTP"
        "topo_type": "1DMESH",
        })";

    JsonParser topoParser;
    TopoInfo topoInfo;
    EXPECT_THROW(topoParser.ParseString(topoString, topoInfo), InvalidParamsException); // ?????????
}

// 字符串输入非法值
//  ranktable对应

// 越界输入 -1; 9999999999999999999999999999999999
TEST_F(TopoParserTest, Ut_Deserialize_When_InvalidTopoInstId_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string topoString = R"({
	    "version": "2.0",
	    "hardware_type" : "950-2D-Fullmsh_64_plus_1",
	    "peer_count" : 3,
        "peer_list" :[
		    { "local_id" : 0 },
		    { "local_id" : 1 },
		    { "local_id" : 2 }
	    ],
	    "edge_count" : 1,
        "edge_list": [
		    {
			    "net_layer": 0,
                "link_type": "PEER2PEER",
			    "protocols": ["UB_CTP"],
                "topo_type": "1DMESH",
                "topo_instance_id": -3,
			    "local_a": 0,
			    "local_a_ports": ["0/0"],
			    "local_b": 1,
			    "local_b_ports": ["0/1"],
			    "position": "DEVICE"
		    }
	    ]
        })";

    JsonParser topoParser;
    TopoInfo topoInfo;
    EXPECT_THROW(topoParser.ParseString(topoString, topoInfo), InvalidParamsException);
}

TEST_F(TopoParserTest, Ut_BinaryStream_When_GetBinStreamToReBuild_Expect_Success)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    TopoInfo expectTopoInfo;
    expectTopoInfo.version = "2.0";
    expectTopoInfo.peerCount = 3;

    PeerInfo peer0;
    peer0.localId = 0;
    PeerInfo peer1;
    peer1.localId = 1;
    PeerInfo peer2;
    peer2.localId = 2;
    expectTopoInfo.peers.emplace_back(peer0);
    expectTopoInfo.peers.emplace_back(peer1);
    expectTopoInfo.peers.emplace_back(peer2);

    expectTopoInfo.edgeCount = 5;
    expectTopoInfo.edges[0] = std::vector<EdgeInfo>();
    expectTopoInfo.edges[1] = std::vector<EdgeInfo>();
    expectTopoInfo.edges[2] = std::vector<EdgeInfo>();
    EdgeInfo edge0;
    edge0.netLayer = 0;
    edge0.linkType = LinkType::PEER2PEER;
    edge0.protocols.emplace(LinkProtocol::UB_CTP);
    edge0.topoType = TopoType::MESH_1D;
    edge0.topoInstId = 0;
    edge0.localA = 0;
    edge0.localAPorts.emplace("0/0");
    edge0.localB = 1;
    edge0.localBPorts.emplace("0/1");
    edge0.position = AddrPosition::DEVICE;
    expectTopoInfo.edges[0].emplace_back(edge0);

    EdgeInfo edge1;
    edge1.netLayer = 0;
    edge1.linkType = LinkType::PEER2PEER;
    edge1.protocols.emplace(LinkProtocol::UB_MEM);
    edge1.topoType = TopoType::MESH_1D;
    edge1.topoInstId = 0;
    edge1.localA = 0;
    edge1.localAPorts.emplace("0/0");
    edge1.localB = 2;
    edge1.localBPorts.emplace("0/1");
    edge1.position = AddrPosition::DEVICE;
    expectTopoInfo.edges[0].emplace_back(edge1);

    EdgeInfo edge2;
    edge2.netLayer = 0;
    edge2.linkType = LinkType::PEER2NET;
    edge2.protocols.emplace(LinkProtocol::UB_CTP);
    edge2.topoType = TopoType::MESH_1D;
    edge2.topoInstId = 0;
    edge2.localA = 0;
    edge2.localAPorts.emplace("0/0");
    edge2.position = AddrPosition::HOST;
    expectTopoInfo.edges[0].emplace_back(edge2);

    EdgeInfo edge3;
    edge3.netLayer = 1;
    edge3.linkType = LinkType::PEER2PEER;
    edge3.protocols.emplace(LinkProtocol::UB_TP);
    edge3.topoType = TopoType::MESH_1D;
    edge3.topoInstId = 0;
    edge3.localA = 1;
    edge3.localAPorts.emplace("0/0");
    edge3.localB = 2;
    edge3.localBPorts.emplace("0/1");
    edge3.position = AddrPosition::DEVICE;
    expectTopoInfo.edges[1].emplace_back(edge3);

    EdgeInfo edge4;
    edge4.netLayer = 2;
    edge4.linkType = LinkType::PEER2NET;
    edge4.protocols.emplace(LinkProtocol::ROCE);
    edge4.topoType = TopoType::CLOS;
    edge4.topoInstId = 0;
    edge4.localA = 0;
    edge4.localAPorts.emplace("0/0");
    edge4.position = AddrPosition::DEVICE;
    expectTopoInfo.edges[2].emplace_back(edge4);

    BinaryStream binStream;
    expectTopoInfo.GetBinStream(binStream);
    TopoInfo reBuildTopo(binStream);

    EXPECT_EQ(expectTopoInfo.version, reBuildTopo.version);
    EXPECT_EQ(expectTopoInfo.peerCount, reBuildTopo.peerCount);
    EXPECT_EQ(expectTopoInfo.peers.size(), reBuildTopo.peers.size());
    for (u32 i = 0; i < expectTopoInfo.peers.size(); i++) {
        EXPECT_EQ(expectTopoInfo.peers[i].localId, reBuildTopo.peers[i].localId);
    }
    EXPECT_EQ(expectTopoInfo.edgeCount, reBuildTopo.edgeCount);
    EXPECT_EQ(expectTopoInfo.edges.size(), reBuildTopo.edges.size());

    auto it_topo_edges = expectTopoInfo.edges.begin();
    auto it_expect_edges = reBuildTopo.edges.begin();
    for (; it_topo_edges != expectTopoInfo.edges.end(); it_topo_edges++, it_expect_edges++) {
        EXPECT_EQ(it_topo_edges->first, it_expect_edges->first);
        EXPECT_EQ((it_topo_edges->second).size(), (it_expect_edges->second).size());
        for (u32 i = 0; i < (it_topo_edges->second).size(); i++) {
            EXPECT_EQ((it_topo_edges->second)[i].netLayer, (it_expect_edges->second)[i].netLayer);
            EXPECT_EQ((it_topo_edges->second)[i].protocols, (it_expect_edges->second)[i].protocols);
            EXPECT_EQ((it_topo_edges->second)[i].linkType, (it_expect_edges->second)[i].linkType);
            EXPECT_EQ((it_topo_edges->second)[i].topoType, (it_expect_edges->second)[i].topoType);
            EXPECT_EQ((it_topo_edges->second)[i].topoInstId, (it_expect_edges->second)[i].topoInstId);
            EXPECT_EQ((it_topo_edges->second)[i].localA, (it_expect_edges->second)[i].localA);
            EXPECT_EQ((it_topo_edges->second)[i].localAPorts, (it_expect_edges->second)[i].localAPorts);
            EXPECT_EQ((it_topo_edges->second)[i].localB, (it_expect_edges->second)[i].localB);
            EXPECT_EQ((it_topo_edges->second)[i].localBPorts, (it_expect_edges->second)[i].localBPorts);
            EXPECT_EQ((it_topo_edges->second)[i].position, (it_expect_edges->second)[i].position);
        }
    }

    EXPECT_EQ(expectTopoInfo.Describe(), reBuildTopo.Describe());
}

TEST_F(TopoParserTest, Ut_DeserializeBinaryStream_When_Normal_Expect_Success)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    JsonParser topoParser;
    TopoInfo topoInfo;
    topoParser.ParseString(BuildLargeTopoString(), topoInfo);

    BinaryStream binaryStream;
    topoInfo.GetBinStream(binaryStream);
    TopoInfo reBuildTopoInfo(binaryStream);
    EXPECT_EQ(topoInfo.Describe(), reBuildTopoInfo.Describe());
}

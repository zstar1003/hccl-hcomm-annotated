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
#include "json_parser.h"
#include "orion_adapter_rts.h"
#include "ip_address.h"

#include "rank_level_info.h"

using namespace Hccl;

class RankLevelInfoParserTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RankLevelInfoParserTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "RankLevelInfoParserTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in RankLevelInfoParserTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in RankLevelInfoParserTest TearDown" << std::endl;
    }
};

TEST_F(RankLevelInfoParserTest, St_Deserialize_When_Normal_Expect_Success)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string rankLevelString = R"(
          {
            "net_layer": 0,
            "net_instance_id": "superPod0-rack3",
            "net_type": "TOPO_FILE_DESC",  
            "net_attr": "",
            "rank_addr_list": [
              {
                "addr_type": "IPV4", 
                "addr": "192.168.100.100",
                "ports": [ "0/1", "0/2" ],
                "plane_id": "planeA"
              },
              {
                "addr_type": "IPV4", 
                "addr": "192.168.100.100",
                "ports": [ "1/1", "1/2" ],
                "plane_id": "planeB"
              }
            ]
          }
    )";
    JsonParser rankLevelParser;
    RankLevelInfo rankLevelInfo;
    rankLevelParser.ParseString(rankLevelString, rankLevelInfo);

    RankLevelInfo rankLevelInfo0;
    rankLevelInfo0.netLayer = 0;
    rankLevelInfo0.netInstId = "superPod0-rack3";
    rankLevelInfo0.netType = NetType::TOPO_FILE_DESC;

    AddressInfo addressInfo0;
    addressInfo0.addrType = AddrType::IPV4;
    IpAddress ipAddress0("192.168.100.100", AF_INET);
    addressInfo0.addr = ipAddress0;
    addressInfo0.ports.emplace("0/1");
    addressInfo0.ports.emplace("0/2");
    addressInfo0.planeId = "planeA";

    AddressInfo addressInfo1;
    addressInfo1.addrType = AddrType::IPV4;
    IpAddress ipAddress1("192.168.100.100", AF_INET);
    addressInfo1.addr = ipAddress1;
    addressInfo1.ports.emplace("1/1");
    addressInfo1.ports.emplace("1/2");
    addressInfo1.planeId = "planeB";

    rankLevelInfo0.rankAddrs.push_back(addressInfo0);
    rankLevelInfo0.rankAddrs.push_back(addressInfo1);
    rankLevelInfo.Describe();

    EXPECT_EQ(rankLevelInfo0.netLayer, rankLevelInfo.netLayer);
    EXPECT_EQ(rankLevelInfo0.netInstId, rankLevelInfo.netInstId);
    EXPECT_EQ(rankLevelInfo0.netType, rankLevelInfo.netType);

    ASSERT_EQ(rankLevelInfo0.rankAddrs.size(), rankLevelInfo.rankAddrs.size());
    for (auto k = 0; k < rankLevelInfo.rankAddrs.size(); k++) {
        EXPECT_EQ(rankLevelInfo0.rankAddrs[k].addrType, rankLevelInfo.rankAddrs[k].addrType);
        EXPECT_EQ(rankLevelInfo0.rankAddrs[k].addr, rankLevelInfo.rankAddrs[k].addr);
        EXPECT_EQ(rankLevelInfo0.rankAddrs[k].ports, rankLevelInfo.rankAddrs[k].ports);
        EXPECT_EQ(rankLevelInfo0.rankAddrs[k].planeId, rankLevelInfo.rankAddrs[k].planeId);
    }
}

TEST_F(RankLevelInfoParserTest, St_Deserialize_When_OptionalFieldsMissing_Expect_Success)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string rankLevelString = R"(
          {
            "net_layer": 0,
            "net_instance_id": "superPod0-rack3",
            "net_type": "TOPO_FILE_DESC",  
            "rank_addr_list": [
              {
                "addr_type": "IPV4", 
                "addr": "192.168.100.100",
                "ports": [ "0/1", "0/2" ]
              },
              {
                "addr_type": "IPV4", 
                "addr": "192.168.100.100",
                "ports": [ "1/1", "1/2" ]
              }
            ]
          }
    )";
    JsonParser rankLevelParser;
    RankLevelInfo rankLevelInfo;
    rankLevelParser.ParseString(rankLevelString, rankLevelInfo);

    RankLevelInfo rankLevelInfo0;
    rankLevelInfo0.netLayer = 0;
    rankLevelInfo0.netInstId = "superPod0-rack3";
    rankLevelInfo0.netType = NetType::TOPO_FILE_DESC;

    AddressInfo addressInfo0;
    addressInfo0.addrType = AddrType::IPV4;
    IpAddress ipAddress0("192.168.100.100", AF_INET);
    addressInfo0.addr = ipAddress0;
    addressInfo0.ports.emplace("0/1");
    addressInfo0.ports.emplace("0/2");

    AddressInfo addressInfo1;
    addressInfo1.addrType = AddrType::IPV4;
    IpAddress ipAddress1("192.168.100.100", AF_INET);
    addressInfo1.addr = ipAddress1;
    addressInfo1.ports.emplace("1/1");
    addressInfo1.ports.emplace("1/2");

    rankLevelInfo0.rankAddrs.push_back(addressInfo0);
    rankLevelInfo0.rankAddrs.push_back(addressInfo1);

    EXPECT_EQ(rankLevelInfo0.netLayer, rankLevelInfo.netLayer);
    EXPECT_EQ(rankLevelInfo0.netInstId, rankLevelInfo.netInstId);
    EXPECT_EQ(rankLevelInfo0.netType, rankLevelInfo.netType);

    ASSERT_EQ(rankLevelInfo0.rankAddrs.size(), rankLevelInfo.rankAddrs.size());
    for (auto k = 0; k < rankLevelInfo.rankAddrs.size(); k++) {
        EXPECT_EQ(rankLevelInfo0.rankAddrs[k].addrType, rankLevelInfo.rankAddrs[k].addrType);
        EXPECT_EQ(rankLevelInfo0.rankAddrs[k].addr, rankLevelInfo.rankAddrs[k].addr);
        EXPECT_EQ(rankLevelInfo0.rankAddrs[k].ports, rankLevelInfo.rankAddrs[k].ports);
    }

    BinaryStream binStream;
    rankLevelInfo.GetBinStream(binStream);
    RankLevelInfo rankLevelInfo1(binStream);
    EXPECT_EQ(rankLevelInfo1.netLayer, rankLevelInfo.netLayer);
    EXPECT_EQ(rankLevelInfo1.netInstId, rankLevelInfo.netInstId);
    EXPECT_EQ(rankLevelInfo1.netType, rankLevelInfo.netType);
    EXPECT_EQ(rankLevelInfo1.portAddrMap, rankLevelInfo.portAddrMap);
}

TEST_F(RankLevelInfoParserTest, St_Deserialize_When_InvalidNetLayer_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string rankLevelString = R"(
          {
            "net_layer": 8,
            "net_instance_id": "superPod0-rack3",
            "net_type": "TOPO_FILE_DESC",  
            "net_attr": "",
            "rank_addr_list": [
              {
                "addr_type": "IPV4", 
                "addr": "192.168.100.100",
                "ports": [ "0/1", "0/2" ]
              },
              {
                "addr_type": "IPV4", 
                "addr": "192.168.100.100",
                "ports": [ "1/1", "1/2" ]
              }
            ]
          }
    )";
    JsonParser rankLevelParser;
    RankLevelInfo rankLevelInfo;

    EXPECT_THROW(rankLevelParser.ParseString(rankLevelString, rankLevelInfo), InvalidParamsException);
}

TEST_F(RankLevelInfoParserTest, St_Deserialize_When_InvalidId_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string rankLevelString = R"(
          {
            "net_layer": 0,
            "net_instance_id": "",
            "net_type": "TOPO_FILE_DESC",  
            "net_attr": "",
            "rank_addr_list": [
              {
                "addr_type": "IPV4", 
                "addr": "192.168.100.100",
                "ports": [ "0/1", "0/2" ]
              },
              {
                "addr_type": "IPV4", 
                "addr": "192.168.100.100",
                "ports": [ "1/1", "1/2" ]
              }
            ]
          }
    )";
    JsonParser rankLevelParser;
    RankLevelInfo rankLevelInfo;

    EXPECT_THROW(rankLevelParser.ParseString(rankLevelString, rankLevelInfo), InvalidParamsException);
}

TEST_F(RankLevelInfoParserTest, St_Deserialize_When_InvalidNetType_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string rankLevelString = R"(
          {
            "net_layer": 0,
            "net_instance_id": "superPod0-rack3",
            "net_type": "TOPO",  
            "net_attr": "",
            "rank_addr_list": [
              {
                "addr_type": "IPV4", 
                "addr": "192.168.100.100",
                "ports": [ "0/1", "0/2" ]
              },
              {
                "addr_type": "IPV4", 
                "addr": "192.168.100.100",
                "ports": [ "1/1", "1/2" ]
              }
            ]
          }
    )";
    JsonParser rankLevelParser;
    RankLevelInfo rankLevelInfo;

    EXPECT_THROW(rankLevelParser.ParseString(rankLevelString, rankLevelInfo), InvalidParamsException);
}

TEST_F(RankLevelInfoParserTest, St_Deserialize_When_InvalidPortstoAddr_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string rankLevelString = R"(
          {
            "net_layer": 0,
            "net_instance_id": "superPod0-rack3",
            "net_type": "TOPO_FILE_DESC",  
            "net_attr": "",
            "rank_addr_list": [
              {
                "addr_type": "IPV4", 
                "addr": "192.168.100.100",
                "ports": [ "0/1", "0/2" ]
              },
              {
                "addr_type": "IPV4", 
                "addr": "192.168.100.101",
                "ports": [ "0/1", "0/2" ]
              }
            ]
          }
    )";
    JsonParser rankLevelParser;
    RankLevelInfo rankLevelInfo;

    rankLevelParser.ParseString(rankLevelString, rankLevelInfo);
}

TEST_F(RankLevelInfoParserTest, St_Deserialize_When_InvalidList_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string rankLevelString = R"(
  {
    "net_layer": 0,
    "net_instance_id": "superPod0-rack3",
    "net_type": "TOPO_FILE_DESC",
    "net_attr": "",
    "rank_addr_list": [
      { "addr_type": "IPV4", "addr": "192.168.100.100", "ports": ["0/1", "0/2"] },
      { "addr_type": "IPV4", "addr": "192.168.100.101", "ports": ["0/1", "0/2"] },
      { "addr_type": "IPV4", "addr": "192.168.100.102", "ports": ["0/1", "0/2"] },
      { "addr_type": "IPV4", "addr": "192.168.100.103", "ports": ["0/1", "0/2"] },
      { "addr_type": "IPV4", "addr": "192.168.100.104", "ports": ["0/1", "0/2"] },
      { "addr_type": "IPV4", "addr": "192.168.100.105", "ports": ["0/1", "0/2"] },
      { "addr_type": "IPV4", "addr": "192.168.100.106", "ports": ["0/1", "0/2"] },
      { "addr_type": "IPV4", "addr": "192.168.100.107", "ports": ["0/1", "0/2"] },
      { "addr_type": "IPV4", "addr": "192.168.100.108", "ports": ["0/1", "0/2"] },
      { "addr_type": "IPV4", "addr": "192.168.100.109", "ports": ["0/1", "0/2"] },
      { "addr_type": "IPV4", "addr": "192.168.100.110", "ports": ["0/1", "0/2"] },
      { "addr_type": "IPV4", "addr": "192.168.100.111", "ports": ["0/1", "0/2"] },
      { "addr_type": "IPV4", "addr": "192.168.100.112", "ports": ["0/1", "0/2"] },
      { "addr_type": "IPV4", "addr": "192.168.100.113", "ports": ["0/1", "0/2"] },
      { "addr_type": "IPV4", "addr": "192.168.100.114", "ports": ["0/1", "0/2"] },
      { "addr_type": "IPV4", "addr": "192.168.100.115", "ports": ["0/1", "0/2"] },
      { "addr_type": "IPV4", "addr": "192.168.100.116", "ports": ["0/1", "0/2"] },
      { "addr_type": "IPV4", "addr": "192.168.100.117", "ports": ["0/1", "0/2"] },
      { "addr_type": "IPV4", "addr": "192.168.100.118", "ports": ["0/1", "0/2"] },
      { "addr_type": "IPV4", "addr": "192.168.100.119", "ports": ["0/1", "0/2"] },
      { "addr_type": "IPV4", "addr": "192.168.100.120", "ports": ["0/1", "0/2"] },
      { "addr_type": "IPV4", "addr": "192.168.100.121", "ports": ["0/1", "0/2"] },
      { "addr_type": "IPV4", "addr": "192.168.100.122", "ports": ["0/1", "0/2"] },
      { "addr_type": "IPV4", "addr": "192.168.100.123", "ports": ["0/1", "0/2"] },
      { "addr_type": "IPV4", "addr": "192.168.100.124", "ports": ["0/1", "0/2"] }
    ]
  }
)";
    JsonParser rankLevelParser;
    RankLevelInfo rankLevelInfo;

    EXPECT_THROW(rankLevelParser.ParseString(rankLevelString, rankLevelInfo), InvalidParamsException);
}

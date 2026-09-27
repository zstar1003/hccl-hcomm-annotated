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

#include "new_rank_info.h"

using namespace Hccl;

class NewRankInfoParserTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "NewRankInfoParserTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "NewRankInfoParserTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in NewRankInfoParserTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in NewRankInfoParserTest TearDown" << std::endl;
    }
};

TEST_F(NewRankInfoParserTest, St_Deserialize_When_Normal_Expect_Success)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string rankListString = R"(
      {
        "rank_id": 0,
        "device_id": 0,  
        "local_id": 0,   
        "replaced_loacl_id": 0,    
        "device_port": 6666,
        "host_port": 7777,     
        "level_list": [
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
        ],
        "controle_plane":{  
               "addr_type": "IPV4",
                "addr": "192.168.100.100",
                "listen_port": 8000
           }
      }
    )";

    JsonParser rankListParser;
    NewRankInfo newRankInfo;
    rankListParser.ParseString(rankListString, newRankInfo);
    newRankInfo.Describe();

    NewRankInfo newRankInfo0;
    newRankInfo0.rankId = 0;
    newRankInfo0.deviceId = 0;
    newRankInfo0.localId = 0;
    newRankInfo0.devicePort = 6666;
    newRankInfo0.hostPort = 7777;

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

    newRankInfo0.rankLevelInfos.push_back(rankLevelInfo0);

    EXPECT_EQ(newRankInfo0.rankId, newRankInfo.rankId);
    EXPECT_EQ(newRankInfo0.localId, newRankInfo0.localId);
    EXPECT_EQ(newRankInfo0.deviceId, newRankInfo0.deviceId);
    EXPECT_EQ(newRankInfo0.devicePort, newRankInfo0.devicePort);
    EXPECT_EQ(newRankInfo0.hostPort, newRankInfo.hostPort);

    ASSERT_EQ(newRankInfo0.rankLevelInfos.size(), newRankInfo.rankLevelInfos.size());
    for (auto j = 0; j < newRankInfo.rankLevelInfos.size(); j++) {
        EXPECT_EQ(newRankInfo0.rankLevelInfos[j].netLayer, newRankInfo.rankLevelInfos[j].netLayer);
        EXPECT_EQ(newRankInfo0.rankLevelInfos[j].netInstId, newRankInfo.rankLevelInfos[j].netInstId);
        EXPECT_EQ(newRankInfo0.rankLevelInfos[j].netType, newRankInfo.rankLevelInfos[j].netType);

        ASSERT_EQ(newRankInfo0.rankLevelInfos[j].rankAddrs.size(), newRankInfo.rankLevelInfos[j].rankAddrs.size());
        for (auto k = 0; k < newRankInfo.rankLevelInfos[j].rankAddrs.size(); k++) {
            EXPECT_EQ(
                newRankInfo0.rankLevelInfos[j].rankAddrs[k].addrType,
                newRankInfo.rankLevelInfos[j].rankAddrs[k].addrType);
            EXPECT_EQ(
                newRankInfo0.rankLevelInfos[j].rankAddrs[k].addr, newRankInfo.rankLevelInfos[j].rankAddrs[k].addr);
            EXPECT_EQ(
                newRankInfo0.rankLevelInfos[j].rankAddrs[k].ports, newRankInfo.rankLevelInfos[j].rankAddrs[k].ports);
            EXPECT_EQ(
                newRankInfo0.rankLevelInfos[j].rankAddrs[k].planeId,
                newRankInfo.rankLevelInfos[j].rankAddrs[k].planeId);
        }
    }
}

TEST_F(NewRankInfoParserTest, St_Deserialize_When_OptionalFieldsMissing_Expect_Success)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string rankListString = R"(
      {
        "rank_id": 0,
        "device_id": 0,  
        "local_id": 0,   
        "replaced_loacl_id": 0,          
        "level_list": [
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
                "addr": "192.168.100.100",
                "ports": [ "1/1", "1/2" ]
              }
            ]
          }
        ]
      }
    )";

    JsonParser rankListParser;
    NewRankInfo newRankInfo;
    rankListParser.ParseString(rankListString, newRankInfo);

    NewRankInfo newRankInfo0;
    newRankInfo0.rankId = 0;
    newRankInfo0.deviceId = 0;
    newRankInfo0.localId = 0;

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

    newRankInfo0.rankLevelInfos.push_back(rankLevelInfo0);

    EXPECT_EQ(newRankInfo0.rankId, newRankInfo.rankId);
    EXPECT_EQ(newRankInfo0.localId, newRankInfo.localId);
    EXPECT_EQ(newRankInfo0.deviceId, newRankInfo.deviceId);

    ASSERT_EQ(newRankInfo0.rankLevelInfos.size(), newRankInfo.rankLevelInfos.size());
    for (auto j = 0; j < newRankInfo.rankLevelInfos.size(); j++) {
        EXPECT_EQ(newRankInfo0.rankLevelInfos[j].netLayer, newRankInfo.rankLevelInfos[j].netLayer);
        EXPECT_EQ(newRankInfo0.rankLevelInfos[j].netInstId, newRankInfo.rankLevelInfos[j].netInstId);
        EXPECT_EQ(newRankInfo0.rankLevelInfos[j].netType, newRankInfo.rankLevelInfos[j].netType);

        ASSERT_EQ(newRankInfo0.rankLevelInfos[j].rankAddrs.size(), newRankInfo.rankLevelInfos[j].rankAddrs.size());
        for (auto k = 0; k < newRankInfo.rankLevelInfos[j].rankAddrs.size(); k++) {
            EXPECT_EQ(
                newRankInfo0.rankLevelInfos[j].rankAddrs[k].addrType,
                newRankInfo.rankLevelInfos[j].rankAddrs[k].addrType);
            EXPECT_EQ(
                newRankInfo0.rankLevelInfos[j].rankAddrs[k].addr, newRankInfo.rankLevelInfos[j].rankAddrs[k].addr);
            EXPECT_EQ(
                newRankInfo0.rankLevelInfos[j].rankAddrs[k].ports, newRankInfo.rankLevelInfos[j].rankAddrs[k].ports);
        }
    }

    BinaryStream binStream;
    newRankInfo.GetBinStream(true, binStream);
    NewRankInfo newRankInfo1(binStream);
    EXPECT_EQ(newRankInfo1.rankId, newRankInfo.rankId);
    EXPECT_EQ(newRankInfo1.localId, newRankInfo.localId);
    EXPECT_EQ(newRankInfo1.deviceId, newRankInfo.deviceId);
    EXPECT_EQ(newRankInfo1.hostPort, newRankInfo.hostPort);
}

TEST_F(NewRankInfoParserTest, St_Deserialize_When_InvalidLoaclId_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string rankListString = R"(
      {
        "rank_id": 0,
        "device_id": 0,  
        "local_id": 65,  
        "replaced_loacl_id": 0,    
        "device_port": 6666,
        "level_list": [
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
        ],
        "controle_plane":{  
               "addr_type": "IPV4",
                "addr": "192.168.100.100",
                "listen_port": 8000
           }
      }
    )";

    JsonParser rankListParser;
    NewRankInfo newRankInfo;

    EXPECT_THROW(rankListParser.ParseString(rankListString, newRankInfo), InvalidParamsException);
}

TEST_F(NewRankInfoParserTest, St_Deserialize_When_InvalidReLoaclId_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string rankListString = R"(
      {
        "rank_id": 0,
        "device_id": 0,  
        "local_id": 64,  
        "replaced_loacl_id": 64,    
        "device_port": 6666,
        "level_list": [
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
        ],
        "controle_plane":{  
               "addr_type": "IPV4",
                "addr": "192.168.100.100",
                "listen_port": 8000
           }
      }
    )";

    JsonParser rankListParser;
    NewRankInfo newRankInfo;

    EXPECT_THROW(rankListParser.ParseString(rankListString, newRankInfo), InvalidParamsException);
}

TEST_F(NewRankInfoParserTest, St_Deserialize_When_InvalidDeviceId_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string rankListString = R"(
      {
        "rank_id": 0,
        "device_id": 128,  
        "local_id": 0,     
        "device_port": 6666,
        "level_list": [
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
        ],
        "controle_plane":{  
               "addr_type": "IPV4",
                "addr": "192.168.100.100",
                "listen_port": 8000
           }
      }
    )";

    JsonParser rankListParser;
    NewRankInfo newRankInfo;

    EXPECT_THROW(rankListParser.ParseString(rankListString, newRankInfo), InvalidParamsException);
}

TEST_F(NewRankInfoParserTest, St_Deserialize_When_InvalidDevicePort_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string rankListString = R"(
      {
        "rank_id": 0,
        "device_id": 0,  
        "local_id": 0,   
        "replaced_loacl_id": 0,    
        "device_port": 66666,       
        "level_list": [
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
        ],
        "controle_plane":{  
               "addr_type": "IPV4",
                "addr": "192.168.100.100",
                "listen_port": 8000
           }
      }
    )";

    JsonParser rankListParser;
    NewRankInfo newRankInfo;

    EXPECT_THROW(rankListParser.ParseString(rankListString, newRankInfo), InvalidParamsException);
}

TEST_F(NewRankInfoParserTest, St_Deserialize_When_InvalidList_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string rankListString = R"(
      {
        "rank_id": 0, 
        "local_id": 0,   
        "replaced_loacl_id": 0,    
        "device_port": 6666,
        "level_list": [],
        "controle_plane":{  
               "addr_type": "IPV4",
                "addr": "192.168.100.100",
                "listen_port": 8000
           }
      }
    )";

    JsonParser rankListParser;
    NewRankInfo newRankInfo;

    EXPECT_THROW(rankListParser.ParseString(rankListString, newRankInfo), InvalidParamsException);
}

TEST_F(NewRankInfoParserTest, St_Deserialize_When_InvalidLevelListLength_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string rankListString = R"(
      {
        "rank_id": 0,
        "device_id": 0,  
        "local_id": 0,     
        "device_port": 6666,
        "level_list": [
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
          },
          {
            "net_layer": 0,
            "net_instance_id": "superPod0-rack3",
            "net_type": "TOPO_FILE_DESC",  
            "net_attr": "",
            "rank_addr_list": [
              {
               "addr_type": "IPV4",
                "addr": "192.168.100.100",
                "ports": [ "0/3", "0/4" ],
                "plane_id": "planeA"
              },
              {
               "addr_type": "IPV4",
                "addr": "192.168.100.100",
                "ports": [ "1/3", "1/4" ],
                "plane_id": "planeB"
              }
            ]
          },
          {
            "net_layer": 0,
            "net_instance_id": "superPod0-rack3",
            "net_type": "TOPO_FILE_DESC",  
            "net_attr": "",
            "rank_addr_list": [
              {
               "addr_type": "IPV4",
                "addr": "192.168.100.100",
                "ports": [ "0/5", "0/6" ],
                "plane_id": "planeA"
              },
              {
               "addr_type": "IPV4",
                "addr": "192.168.100.100",
                "ports": [ "1/5", "1/6" ],
                "plane_id": "planeB"
              }
            ]
          },
          {
            "net_layer": 0,
            "net_instance_id": "superPod0-rack3",
            "net_type": "TOPO_FILE_DESC",  
            "net_attr": "",
            "rank_addr_list": [
              {
               "addr_type": "IPV4",
                "addr": "192.168.100.100",
                "ports": [ "0/7", "0/8" ],
                "plane_id": "planeA"
              },
              {
               "addr_type": "IPV4",
                "addr": "192.168.100.100",
                "ports": [ "1/7", "1/8" ],
                "plane_id": "planeB"
              }
            ]
          },
          {
            "net_layer": 0,
            "net_instance_id": "superPod0-rack3",
            "net_type": "TOPO_FILE_DESC",  
            "net_attr": "",
            "rank_addr_list": [
              {
               "addr_type": "IPV4",
                "addr": "192.168.100.100",
                "ports": [ "0/9", "0/10" ],
                "plane_id": "planeA"
              },
              {
               "addr_type": "IPV4",
                "addr": "192.168.100.100",
                "ports": [ "1/9", "1/10" ],
                "plane_id": "planeB"
              }
            ]
          },
          {
            "net_layer": 0,
            "net_instance_id": "superPod0-rack3",
            "net_type": "TOPO_FILE_DESC",  
            "net_attr": "",
            "rank_addr_list": [
              {
               "addr_type": "IPV4",
                "addr": "192.168.100.100",
                "ports": [ "0/11", "0/12" ],
                "plane_id": "planeA"
              },
              {
               "addr_type": "IPV4",
                "addr": "192.168.100.100",
                "ports": [ "1/11", "1/12" ],
                "plane_id": "planeB"
              }
            ]
          },
          {
            "net_layer": 0,
            "net_instance_id": "superPod0-rack3",
            "net_type": "TOPO_FILE_DESC",  
            "net_attr": "",
            "rank_addr_list": [
              {
               "addr_type": "IPV4",
                "addr": "192.168.100.100",
                "ports": [ "0/13", "0/14" ],
                "plane_id": "planeA"
              },
              {
               "addr_type": "IPV4",
                "addr": "192.168.100.100",
                "ports": [ "1/13", "1/14" ],
                "plane_id": "planeB"
              }
            ]
          },
          {
            "net_layer": 0,
            "net_instance_id": "superPod0-rack3",
            "net_type": "TOPO_FILE_DESC",  
            "net_attr": "",
            "rank_addr_list": [
              {
               "addr_type": "IPV4",
                "addr": "192.168.100.100",
                "ports": [ "0/15", "0/16" ],
                "plane_id": "planeA"
              },
              {
               "addr_type": "IPV4",
                "addr": "192.168.100.100",
                "ports": [ "1/15", "1/16" ],
                "plane_id": "planeB"
              }
            ]
          },
          {
            "net_layer": 0,
            "net_instance_id": "superPod0-rack3",
            "net_type": "TOPO_FILE_DESC",  
            "net_attr": "",
            "rank_addr_list": [
              {
               "addr_type": "IPV4",
                "addr": "192.168.100.100",
                "ports": [ "0/17", "0/18" ],
                "plane_id": "planeA"
              },
              {
               "addr_type": "IPV4",
                "addr": "192.168.100.100",
                "ports": [ "1/17", "1/18" ],
                "plane_id": "planeB"
              }
            ]
          }
        ],
        "controle_plane":{  
               "addr_type": "IPV4",
                "addr": "192.168.100.100",
                "listen_port": 8000
           }
      }
    )";

    JsonParser rankListParser;
    NewRankInfo newRankInfo;

    EXPECT_THROW(rankListParser.ParseString(rankListString, newRankInfo), InvalidParamsException);
}
TEST_F(NewRankInfoParserTest, St_Deserialize_When_InvalidHostPort_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string rankListString = R"(
      {
        "rank_id": 0,
        "device_id": 0,
        "local_id": 0,
        "replaced_loacl_id": 0,
        "host_port": 66666,
        "level_list": [
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
        ],
        "controle_plane":{
               "addr_type": "IPV4",
                "addr": "192.168.100.100",
                "listen_port": 8000
           }
      }
    )";

    JsonParser rankListParser;
    NewRankInfo newRankInfo;

    EXPECT_THROW(rankListParser.ParseString(rankListString, newRankInfo), InvalidParamsException);
}

TEST_F(NewRankInfoParserTest, St_Deserialize_When_InvalidHostPortMin_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string rankListString = R"(
      {
        "rank_id": 0,
        "device_id": 0,
        "local_id": 0,
        "replaced_loacl_id": 0,
        "host_port": 0,
        "level_list": [
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
        ],
        "controle_plane":{
               "addr_type": "IPV4",
                "addr": "192.168.100.100",
                "listen_port": 8000
           }
      }
    )";

    JsonParser rankListParser;
    NewRankInfo newRankInfo;

    EXPECT_THROW(rankListParser.ParseString(rankListString, newRankInfo), InvalidParamsException);
}

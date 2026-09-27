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

#include "rank_table_info.h"

using namespace Hccl;

class RankTableInfoParserTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RankTableInfoParserTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "RankTableInfoParserTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in RankTableInfoParserTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in RankTableInfoParserTest TearDown" << std::endl;
    }
};

TEST_F(RankTableInfoParserTest, St_Deserialize_When_Normal_Expect_Success)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string rankTableString = R"(
    {
    "version": "2.0",  
    "status": "completed",         
    "detour": "true",  
    "rank_count": 1,
    "rank_list": [
      {
        "rank_id": 0,
        "device_id": 0,  
        "local_id": 0,   
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
      ]
     }
    )";

    JsonParser rankTableParser;
    RankTableInfo rankTableInfo;
    rankTableParser.ParseString(rankTableString, rankTableInfo);
    rankTableInfo.Describe();

    RankTableInfo expecetedRankTableInfo;
    expecetedRankTableInfo.version = "2.0";
    expecetedRankTableInfo.detour = true;
    expecetedRankTableInfo.rankCount = 1;

    NewRankInfo newRankInfo0;
    newRankInfo0.rankId = 0;
    newRankInfo0.deviceId = 0;
    newRankInfo0.localId = 0;
    newRankInfo0.devicePort = 6666;

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
    expecetedRankTableInfo.ranks.push_back(newRankInfo0);

    EXPECT_EQ(expecetedRankTableInfo.version, rankTableInfo.version);
    EXPECT_EQ(expecetedRankTableInfo.detour, rankTableInfo.detour);
    EXPECT_EQ(expecetedRankTableInfo.rankCount, rankTableInfo.rankCount);

    ASSERT_EQ(expecetedRankTableInfo.ranks.size(), rankTableInfo.ranks.size());
    for (auto i = 0; i < rankTableInfo.ranks.size(); i++) {
        EXPECT_EQ(expecetedRankTableInfo.ranks[i].rankId, rankTableInfo.ranks[i].rankId);
        EXPECT_EQ(expecetedRankTableInfo.ranks[i].localId, rankTableInfo.ranks[i].localId);
        EXPECT_EQ(expecetedRankTableInfo.ranks[i].deviceId, rankTableInfo.ranks[i].deviceId);
        EXPECT_EQ(expecetedRankTableInfo.ranks[i].devicePort, rankTableInfo.ranks[i].devicePort);

        ASSERT_EQ(expecetedRankTableInfo.ranks[i].rankLevelInfos.size(), rankTableInfo.ranks[i].rankLevelInfos.size());
        for (auto j = 0; j < rankTableInfo.ranks[i].rankLevelInfos.size(); j++) {
            EXPECT_EQ(
                expecetedRankTableInfo.ranks[i].rankLevelInfos[j].netLayer,
                rankTableInfo.ranks[i].rankLevelInfos[j].netLayer);
            EXPECT_EQ(
                expecetedRankTableInfo.ranks[i].rankLevelInfos[j].netInstId,
                rankTableInfo.ranks[i].rankLevelInfos[j].netInstId);
            EXPECT_EQ(
                expecetedRankTableInfo.ranks[i].rankLevelInfos[j].netType,
                rankTableInfo.ranks[i].rankLevelInfos[j].netType);

            ASSERT_EQ(
                expecetedRankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs.size(),
                rankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs.size());
            for (auto k = 0; k < rankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs.size(); k++) {
                EXPECT_EQ(
                    expecetedRankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs[k].addrType,
                    rankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs[k].addrType);
                EXPECT_EQ(
                    expecetedRankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs[k].addr,
                    rankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs[k].addr);
                EXPECT_EQ(
                    expecetedRankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs[k].ports,
                    rankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs[k].ports);
                EXPECT_EQ(
                    expecetedRankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs[k].planeId,
                    rankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs[k].planeId);
            }
        }
    }
}

TEST_F(RankTableInfoParserTest, Ut_Deserialize_When_OptionalFieldsMissing_Expect_Success)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string rankTableString = R"(
    {
    "version": "2.0",  
    "status": "completed",         
    "rank_count": 1,
    "rank_list": [
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
      ]
     }
    )";

    JsonParser rankTableParser;
    RankTableInfo rankTableInfo;
    rankTableParser.ParseString(rankTableString, rankTableInfo);

    RankTableInfo expecetedRankTableInfo;
    expecetedRankTableInfo.version = "2.0";
    expecetedRankTableInfo.rankCount = 1;

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
    expecetedRankTableInfo.ranks.push_back(newRankInfo0);

    EXPECT_EQ(expecetedRankTableInfo.version, rankTableInfo.version);
    EXPECT_EQ(expecetedRankTableInfo.rankCount, rankTableInfo.rankCount);

    ASSERT_EQ(expecetedRankTableInfo.ranks.size(), rankTableInfo.ranks.size());
    for (auto i = 0; i < rankTableInfo.ranks.size(); i++) {
        EXPECT_EQ(expecetedRankTableInfo.ranks[i].rankId, rankTableInfo.ranks[i].rankId);
        EXPECT_EQ(expecetedRankTableInfo.ranks[i].localId, rankTableInfo.ranks[i].localId);
        EXPECT_EQ(expecetedRankTableInfo.ranks[i].deviceId, rankTableInfo.ranks[i].deviceId);

        ASSERT_EQ(expecetedRankTableInfo.ranks[i].rankLevelInfos.size(), rankTableInfo.ranks[i].rankLevelInfos.size());
        for (auto j = 0; j < rankTableInfo.ranks[i].rankLevelInfos.size(); j++) {
            EXPECT_EQ(
                expecetedRankTableInfo.ranks[i].rankLevelInfos[j].netLayer,
                rankTableInfo.ranks[i].rankLevelInfos[j].netLayer);
            EXPECT_EQ(
                expecetedRankTableInfo.ranks[i].rankLevelInfos[j].netInstId,
                rankTableInfo.ranks[i].rankLevelInfos[j].netInstId);
            EXPECT_EQ(
                expecetedRankTableInfo.ranks[i].rankLevelInfos[j].netType,
                rankTableInfo.ranks[i].rankLevelInfos[j].netType);

            ASSERT_EQ(
                expecetedRankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs.size(),
                rankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs.size());
            for (auto k = 0; k < rankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs.size(); k++) {
                EXPECT_EQ(
                    expecetedRankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs[k].addrType,
                    rankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs[k].addrType);
                EXPECT_EQ(
                    expecetedRankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs[k].addr,
                    rankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs[k].addr);
                EXPECT_EQ(
                    expecetedRankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs[k].ports,
                    rankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs[k].ports);
            }
        }
    }

    BinaryStream binStream;
    rankTableInfo.GetBinStream(true, binStream);
    RankTableInfo rankTableInfo1(binStream);
    EXPECT_EQ(rankTableInfo1.version, rankTableInfo.version);
    EXPECT_EQ(rankTableInfo1.rankCount, rankTableInfo.rankCount);
}

TEST_F(RankTableInfoParserTest, St_GetBinStream_When_Normal_Expect_Success)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string rankTableString = R"(
    {
    "version": "2.0",  
    "status": "completed",         
    "rank_count": 1,
    "rank_list": [
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
      ]
     }
    )";

    JsonParser rankTableParser;
    RankTableInfo rankTableInfo;
    rankTableParser.ParseString(rankTableString, rankTableInfo);
    BinaryStream binStream;
    rankTableInfo.GetUniqueId(true);
    rankTableInfo.GetBinStream(true, binStream);
    RankTableInfo expecetedRankTableInfo(binStream);

    EXPECT_EQ(expecetedRankTableInfo.version, rankTableInfo.version);
    EXPECT_EQ(expecetedRankTableInfo.rankCount, rankTableInfo.rankCount);

    ASSERT_EQ(expecetedRankTableInfo.ranks.size(), rankTableInfo.ranks.size());
    for (auto i = 0; i < rankTableInfo.ranks.size(); i++) {
        EXPECT_EQ(expecetedRankTableInfo.ranks[i].rankId, rankTableInfo.ranks[i].rankId);
        EXPECT_EQ(expecetedRankTableInfo.ranks[i].localId, rankTableInfo.ranks[i].localId);
        EXPECT_EQ(expecetedRankTableInfo.ranks[i].deviceId, rankTableInfo.ranks[i].deviceId);

        ASSERT_EQ(expecetedRankTableInfo.ranks[i].rankLevelInfos.size(), rankTableInfo.ranks[i].rankLevelInfos.size());
        for (auto j = 0; j < rankTableInfo.ranks[i].rankLevelInfos.size(); j++) {
            EXPECT_EQ(
                expecetedRankTableInfo.ranks[i].rankLevelInfos[j].netLayer,
                rankTableInfo.ranks[i].rankLevelInfos[j].netLayer);
            EXPECT_EQ(
                expecetedRankTableInfo.ranks[i].rankLevelInfos[j].netInstId,
                rankTableInfo.ranks[i].rankLevelInfos[j].netInstId);
            EXPECT_EQ(
                expecetedRankTableInfo.ranks[i].rankLevelInfos[j].netType,
                rankTableInfo.ranks[i].rankLevelInfos[j].netType);

            ASSERT_EQ(
                expecetedRankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs.size(),
                rankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs.size());
            for (auto k = 0; k < rankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs.size(); k++) {
                EXPECT_EQ(
                    expecetedRankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs[k].addrType,
                    rankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs[k].addrType);
                EXPECT_EQ(
                    expecetedRankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs[k].addr,
                    rankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs[k].addr);
                EXPECT_EQ(
                    expecetedRankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs[k].ports,
                    rankTableInfo.ranks[i].rankLevelInfos[j].rankAddrs[k].ports);
            }
        }
    }
}

TEST_F(RankTableInfoParserTest, St_Deserialize_When_InvalidParameter_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string rankTableString = R"(
    {
    "status": "completed",         
    "detour": "true",  
    "rank_count": 1,
    "rank_list": [
      {
        "rank_id": 0,
        "device_id": 0,  
        "local_id": 0,   
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
      ]
     }
    )";
    JsonParser rankTableParser;
    RankTableInfo rankTableInfo;

    EXPECT_THROW(rankTableParser.ParseString(rankTableString, rankTableInfo), InvalidParamsException);
}

TEST_F(RankTableInfoParserTest, St_Deserialize_When_InvalidVersion_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string rankTableString = R"(
    {
    "version": "8.0",  
    "status": "completed",         
    "detour": "true",  
    "rank_count": 1,
    "rank_list": [
      {
        "rank_id": 0,
        "device_id": 0,  
        "local_id": 0,   
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
      ]
     }
    )";

    JsonParser rankTableParser;
    RankTableInfo rankTableInfo;

    EXPECT_THROW(rankTableParser.ParseString(rankTableString, rankTableInfo), InvalidParamsException);
}

TEST_F(RankTableInfoParserTest, St_Deserialize_When_InvalidRank_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string rankTableString = R"(
    {
    "version": "2.0",  
    "status": "completed",         
    "detour": "true",  
    "rank_count": -1,
    "rank_list": [
      {
        "rank_id": 0,
        "device_id": 0,  
        "local_id": 0,   
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
      ]
     }
    )";

    JsonParser rankTableParser;
    RankTableInfo rankTableInfo;

    EXPECT_THROW(rankTableParser.ParseString(rankTableString, rankTableInfo), InvalidParamsException);
}

TEST_F(RankTableInfoParserTest, St_Deserialize_When_InvalidRankCount_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string rankTableString = R"(
    {
    "version": "2.0",  
    "status": "completed",         
    "detour": "true",  
    "rank_count": 128,
    "rank_list": [
      {
        "rank_id": 0,
        "device_id": 0,  
        "local_id": 0,   
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
      ]
     }
    )";

    JsonParser rankTableParser;
    RankTableInfo rankTableInfo;

    EXPECT_THROW(rankTableParser.ParseString(rankTableString, rankTableInfo), InvalidParamsException);
}

TEST_F(RankTableInfoParserTest, St_Deserialize_When_InvalidRankId_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string rankTableString = R"(
    {
    "version": "2.0",  
    "status": "completed",         
    "detour": "true",  
    "rank_count": 1,
    "rank_list": [
      {
        "rank_id": 1,
        "device_id": 0,  
        "local_id": 0,   
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
      ]
     }
    )";

    JsonParser rankTableParser;
    RankTableInfo rankTableInfo;

    EXPECT_THROW(rankTableParser.ParseString(rankTableString, rankTableInfo), InvalidParamsException);
}

TEST_F(RankTableInfoParserTest, St_Deserialize_When_SameRankId_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string rankTableString = R"(
    {
    "version": "2.0",  
    "status": "completed",         
    "detour": "true",  
    "rank_count": 2,
    "rank_list": [
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
          }
        ]
      },
      {
        "rank_id": 0,
        "device_id": 1,  
        "local_id": 1,     
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
        ]
      }
      ]
     }
    )";

    JsonParser rankTableParser;
    RankTableInfo rankTableInfo;

    EXPECT_THROW(rankTableParser.ParseString(rankTableString, rankTableInfo), InvalidParamsException);
}

TEST_F(RankTableInfoParserTest, St_Deserialize_When_InvalidDetour_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string rankTableString = R"(
    {
    "version": "2.0",  
    "status": "completed",         
    "detour": "sad",  
    "rank_count": 1,
    "rank_list": [
      {
        "rank_id": 0,
        "device_id": 0,  
        "local_id": 0,   
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
      ]
     }
    )";

    JsonParser rankTableParser;
    RankTableInfo rankTableInfo;

    EXPECT_THROW(rankTableParser.ParseString(rankTableString, rankTableInfo), InvalidParamsException);
}

TEST_F(RankTableInfoParserTest, St_Deserialize_When_RankCountExceedsMax_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string rankTableString = R"(
    {
    "version": "2.0",  
    "status": "completed",         
    "detour": "true",  
    "rank_count": 65537,
    "rank_list": [
      {
        "rank_id": 0,
        "device_id": 0,  
        "local_id": 0,   
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
      ]
     }
    )";

    JsonParser rankTableParser;
    RankTableInfo rankTableInfo;

    EXPECT_THROW(rankTableParser.ParseString(rankTableString, rankTableInfo), InvalidParamsException);
}

TEST_F(RankTableInfoParserTest, Ut_RankTableInfo_When_RankNEQ_Expect_InvalidParamsException)
{
    RankTableInfo rankTableInfo;
    rankTableInfo.version = "2.0";
    rankTableInfo.rankCount = 1;
    vector<NewRankInfo> ranks;
    NewRankInfo newRankInfo;
    newRankInfo.rankId = 0;
    newRankInfo.replacedLocalId = 1;
    ranks.push_back(newRankInfo);
    rankTableInfo.ranks = ranks;
    EXPECT_THROW(rankTableInfo.Check(), InvalidParamsException);
}

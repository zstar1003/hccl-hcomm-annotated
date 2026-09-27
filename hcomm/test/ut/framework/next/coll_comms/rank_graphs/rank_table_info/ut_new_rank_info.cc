/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstddef>
#include <string>

#include "gtest/gtest.h"
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include "json_parser.h"
#include "orion_adapter_rts.h"
#include "ip_address.h"

#include "new_rank_info.h"

using namespace Hccl;

namespace {
nlohmann::json
BuildRankAddressJson(const std::string& portPrefix, u32 firstPort, u32 secondPort, const std::string& planeId)
{
    nlohmann::json rankAddressJson = {
        {"addr_type", "IPV4"},
        {"addr", "192.168.100.100"},
        {"ports", {portPrefix + std::to_string(firstPort), portPrefix + std::to_string(secondPort)}},
    };
    if (!planeId.empty()) {
        rankAddressJson["plane_id"] = planeId;
    }
    return rankAddressJson;
}

nlohmann::json BuildRankLevelJson(u32 firstPort, u32 secondPort, bool includePlaneIds = true)
{
    nlohmann::json rankLevelJson = {
        {"net_layer", 0},
        {"net_instance_id", "superPod0-rack3"},
        {"net_type", "TOPO_FILE_DESC"},
        {"net_attr", ""},
    };
    rankLevelJson["rank_addr_list"] = {
        BuildRankAddressJson("0/", firstPort, secondPort, includePlaneIds ? "planeA" : ""),
        BuildRankAddressJson("1/", firstPort, secondPort, includePlaneIds ? "planeB" : ""),
    };
    return rankLevelJson;
}

nlohmann::json BuildRankInfoJson(bool includeOptionalFields = true)
{
    nlohmann::json rankInfoJson = {
        {"rank_id", 0},
        {"device_id", 0},
        {"local_id", 0},
        {"replaced_local_id", 0},
        {"level_list", {BuildRankLevelJson(1, 2, includeOptionalFields)}},
    };
    if (includeOptionalFields) {
        rankInfoJson["device_port"] = 6666;
        rankInfoJson["host_port"] = 7777;
        rankInfoJson["controle_plane"] = {{"addr_type", "IPV4"}, {"addr", "192.168.100.100"}, {"listen_port", 8000}};
    }
    return rankInfoJson;
}

AddressInfo
BuildAddressInfo(const std::string& firstPort, const std::string& secondPort, const std::string& planeId = "")
{
    AddressInfo addressInfo;
    addressInfo.addrType = AddrType::IPV4;
    addressInfo.addr = IpAddress("192.168.100.100", AF_INET);
    addressInfo.ports.emplace(firstPort);
    addressInfo.ports.emplace(secondPort);
    if (!planeId.empty()) {
        addressInfo.planeId = planeId;
    }
    return addressInfo;
}

NewRankInfo BuildExpectedRankInfo(bool includeOptionalFields)
{
    NewRankInfo rankInfo;
    rankInfo.rankId = 0;
    rankInfo.deviceId = 0;
    rankInfo.localId = 0;
    if (includeOptionalFields) {
        rankInfo.devicePort = 6666;
        rankInfo.hostPort = 7777;
    }

    RankLevelInfo rankLevelInfo;
    rankLevelInfo.netLayer = 0;
    rankLevelInfo.netInstId = "superPod0-rack3";
    rankLevelInfo.netType = NetType::TOPO_FILE_DESC;
    rankLevelInfo.rankAddrs.emplace_back(BuildAddressInfo("0/1", "0/2", includeOptionalFields ? "planeA" : ""));
    rankLevelInfo.rankAddrs.emplace_back(BuildAddressInfo("1/1", "1/2", includeOptionalFields ? "planeB" : ""));
    rankInfo.rankLevelInfos.emplace_back(rankLevelInfo);
    return rankInfo;
}

void ExpectAddressInfoEqual(const AddressInfo& expected, const AddressInfo& actual)
{
    EXPECT_EQ(expected.addrType, actual.addrType);
    EXPECT_EQ(expected.addr, actual.addr);
    EXPECT_EQ(expected.ports, actual.ports);
    EXPECT_EQ(expected.planeId, actual.planeId);
}

void ExpectRankLevelInfosEqual(
    const std::vector<RankLevelInfo>& expectedRankLevelInfos, const std::vector<RankLevelInfo>& actualRankLevelInfos)
{
    ASSERT_EQ(expectedRankLevelInfos.size(), actualRankLevelInfos.size());
    for (std::size_t levelIndex = 0; levelIndex < expectedRankLevelInfos.size(); ++levelIndex) {
        const RankLevelInfo& expected = expectedRankLevelInfos[levelIndex];
        const RankLevelInfo& actual = actualRankLevelInfos[levelIndex];
        EXPECT_EQ(expected.netLayer, actual.netLayer);
        EXPECT_EQ(expected.netInstId, actual.netInstId);
        EXPECT_EQ(expected.netType, actual.netType);
        ASSERT_EQ(expected.rankAddrs.size(), actual.rankAddrs.size());
        for (std::size_t addressIndex = 0; addressIndex < expected.rankAddrs.size(); ++addressIndex) {
            ExpectAddressInfoEqual(expected.rankAddrs[addressIndex], actual.rankAddrs[addressIndex]);
        }
    }
}

void ExpectRankInfoEqual(const NewRankInfo& expected, const NewRankInfo& actual)
{
    EXPECT_EQ(expected.rankId, actual.rankId);
    EXPECT_EQ(expected.localId, actual.localId);
    EXPECT_EQ(expected.deviceId, actual.deviceId);
    EXPECT_EQ(expected.devicePort, actual.devicePort);
    EXPECT_EQ(expected.hostPort, actual.hostPort);
    ExpectRankLevelInfosEqual(expected.rankLevelInfos, actual.rankLevelInfos);
}

void ExpectTlsStatusBinaryRoundTrip(NewRankInfo& rankInfo)
{
    rankInfo.tlsStatus = TlsStatus::DISABLE;
    rankInfo.hostDpuTlsStatus = TlsStatus::ENABLE;
    BinaryStream binStream;
    rankInfo.GetBinStream(true, binStream);
    NewRankInfo deserializedRankInfo(binStream);
    EXPECT_EQ(deserializedRankInfo.rankId, rankInfo.rankId);
    EXPECT_EQ(deserializedRankInfo.localId, rankInfo.localId);
    EXPECT_EQ(deserializedRankInfo.deviceId, rankInfo.deviceId);
    EXPECT_EQ(deserializedRankInfo.hostPort, rankInfo.hostPort);
    EXPECT_EQ(deserializedRankInfo.tlsStatus, rankInfo.tlsStatus);
    EXPECT_EQ(deserializedRankInfo.hostDpuTlsStatus, rankInfo.hostDpuTlsStatus);
}

void ExpectInvalidRankInfoJson(const nlohmann::json& rankInfoJson)
{
    JsonParser rankListParser;
    NewRankInfo rankInfo;
    EXPECT_THROW(rankListParser.ParseString(rankInfoJson.dump(), rankInfo), InvalidParamsException);
}
} // namespace

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

TEST_F(NewRankInfoParserTest, Ut_Deserialize_When_Normal_Expect_Success)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    JsonParser rankListParser;
    NewRankInfo actualRankInfo;
    rankListParser.ParseString(BuildRankInfoJson().dump(), actualRankInfo);
    actualRankInfo.Describe();

    EXPECT_EQ(actualRankInfo.hostDpuTlsStatus, TlsStatus::UNKNOWN);
    ExpectRankInfoEqual(BuildExpectedRankInfo(true), actualRankInfo);
}

TEST_F(NewRankInfoParserTest, Ut_Deserialize_When_OptionalFieldsMissing_Expect_Success)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    JsonParser rankListParser;
    NewRankInfo actualRankInfo;
    rankListParser.ParseString(BuildRankInfoJson(false).dump(), actualRankInfo);

    ExpectRankInfoEqual(BuildExpectedRankInfo(false), actualRankInfo);
    ExpectTlsStatusBinaryRoundTrip(actualRankInfo);
}

TEST_F(NewRankInfoParserTest, Ut_Deserialize_When_InvalidLoaclId_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    nlohmann::json rankInfoJson = BuildRankInfoJson();
    rankInfoJson.erase("host_port");
    rankInfoJson["local_id"] = BACKUP_LOCAL_ID + 1;
    ExpectInvalidRankInfoJson(rankInfoJson);
}

TEST_F(NewRankInfoParserTest, Ut_Deserialize_When_InvalidReLoaclId_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    nlohmann::json rankInfoJson = BuildRankInfoJson();
    rankInfoJson.erase("host_port");
    rankInfoJson["local_id"] = BACKUP_LOCAL_ID;
    rankInfoJson["replaced_local_id"] = BACKUP_LOCAL_ID;
    ExpectInvalidRankInfoJson(rankInfoJson);
}

TEST_F(NewRankInfoParserTest, Ut_Deserialize_When_InvalidDeviceId_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    nlohmann::json rankInfoJson = BuildRankInfoJson();
    rankInfoJson.erase("host_port");
    rankInfoJson.erase("replaced_local_id");
    rankInfoJson["device_id"] = MAX_VALUE_DEVICEID + 1;
    ExpectInvalidRankInfoJson(rankInfoJson);
}

TEST_F(NewRankInfoParserTest, Ut_Deserialize_When_InvalidDevicePort_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    nlohmann::json rankInfoJson = BuildRankInfoJson();
    rankInfoJson.erase("host_port");
    rankInfoJson["device_port"] = 66666;
    ExpectInvalidRankInfoJson(rankInfoJson);
}

TEST_F(NewRankInfoParserTest, Ut_Deserialize_When_InvalidList_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    nlohmann::json rankInfoJson = BuildRankInfoJson();
    rankInfoJson.erase("device_id");
    rankInfoJson.erase("host_port");
    rankInfoJson["level_list"] = nlohmann::json::array();
    ExpectInvalidRankInfoJson(rankInfoJson);
}

TEST_F(NewRankInfoParserTest, Ut_Deserialize_When_InvalidLevelListLength_Expect_Exception)
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
TEST_F(NewRankInfoParserTest, Ut_Deserialize_When_InvalidHostPort_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    nlohmann::json rankInfoJson = BuildRankInfoJson();
    rankInfoJson.erase("device_port");
    rankInfoJson["host_port"] = 66666;
    ExpectInvalidRankInfoJson(rankInfoJson);
}

TEST_F(NewRankInfoParserTest, Ut_Deserialize_When_InvalidHostPortMin_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    nlohmann::json rankInfoJson = BuildRankInfoJson();
    rankInfoJson.erase("device_port");
    rankInfoJson["host_port"] = 0;
    ExpectInvalidRankInfoJson(rankInfoJson);
}

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

#include "address_info.h"
#include "adapter_error_manager_pub.h"

using namespace Hccl;

class AddressInfoParserTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AddressInfoParserTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "AddressInfoParserTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in AddressInfoParserTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in AddressInfoParserTest TearDown" << std::endl;
    }
};

namespace {
struct InvalidAddressInfoCase {
    const char* name;
    const char* addressInfoString;
};

nlohmann::json BuildIpv4AddressInfoWithBackupAddrCount(std::size_t count)
{
    nlohmann::json backupAddrs = nlohmann::json::array();
    for (std::size_t idx = 0; idx < count; ++idx) {
        backupAddrs.push_back("192.168.2." + std::to_string(idx + 1));
    }
    return {
        {"addr_type", "IPV4"},        {"addr", "192.168.100.100"},
        {"backup_addr", backupAddrs}, {"ports", nlohmann::json::array({"1/1"})},
        {"plane_id", "planeB"},
    };
}
} // namespace

TEST_F(AddressInfoParserTest, Ut_Deserialize_When_Normal_Expect_Success)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string addressInfoString = R"(
            {
                "addr_type": "IPV4",
                "addr": "192.168.100.100",
                "ports": [ "1/1", "1/2" ],
                "plane_id": "planeB"
            }
    )";
    JsonParser addressInfoParser;
    AddressInfo addressInfo;
    addressInfoParser.ParseString(addressInfoString, addressInfo);

    AddressInfo addressInfo0;
    addressInfo0.addrType = AddrType::IPV4;
    IpAddress ipAddress0("192.168.100.100", AF_INET);
    addressInfo0.addr = ipAddress0;
    addressInfo0.planeId = "planeB";
    addressInfo0.ports = {"1/1", "1/2"};
    addressInfo.Describe();

    EXPECT_EQ(addressInfo0.addrType, addressInfo.addrType);
    EXPECT_EQ(addressInfo0.addr, addressInfo.addr);
    EXPECT_EQ(addressInfo0.planeId, addressInfo.planeId);
    EXPECT_EQ(addressInfo0.ports, addressInfo.ports);
    EXPECT_TRUE(addressInfo.backupAddrs.empty());
}

TEST_F(AddressInfoParserTest, Ut_Deserialize_When_Ipv4BackupAddrValid_Expect_ParseBackupAddrs)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string addressInfoString = R"(
            {
                "addr_type": "IPV4",
                "addr": "192.168.100.100",
                "backup_addr": ["192.168.100.101", "192.168.100.101", "192.168.100.100"],
                "ports": [ "1/1", "1/2" ],
                "plane_id": "planeB"
            }
    )";
    JsonParser addressInfoParser;
    AddressInfo addressInfo;
    addressInfoParser.ParseString(addressInfoString, addressInfo);

    ASSERT_EQ(addressInfo.backupAddrs.size(), 3U);
    EXPECT_EQ(addressInfo.backupAddrs[0], IpAddress("192.168.100.101", AF_INET));
    EXPECT_EQ(addressInfo.backupAddrs[1], IpAddress("192.168.100.101", AF_INET));
    EXPECT_EQ(addressInfo.backupAddrs[2], IpAddress("192.168.100.100", AF_INET));

    BinaryStream binStream;
    addressInfo.GetBinStream(binStream);
    AddressInfo addressInfoFromStream(binStream);
    EXPECT_EQ(addressInfoFromStream.addr, addressInfo.addr);
    EXPECT_EQ(addressInfoFromStream.backupAddrs, addressInfo.backupAddrs);
    EXPECT_EQ(addressInfoFromStream.addrType, addressInfo.addrType);
    EXPECT_EQ(addressInfoFromStream.ports, addressInfo.ports);
    EXPECT_EQ(addressInfoFromStream.planeId, addressInfo.planeId);
}

TEST_F(AddressInfoParserTest, Ut_BinaryStream_When_OldLayoutWithoutBackupAddr_Expect_DeserializeSuccess)
{
    BinaryStream binStream;
    IpAddress primaryAddr("192.168.100.100", AF_INET);
    primaryAddr.GetBinStream(binStream);
    binStream << static_cast<u32>(AddrType::IPV4);
    const size_t portsSize = 1;
    binStream << portsSize;
    binStream << std::string("1/1");
    binStream << std::string("planeB");
    binStream << static_cast<u32>(0);

    AddressInfo addressInfo(binStream);

    EXPECT_EQ(addressInfo.addr, primaryAddr);
    EXPECT_EQ(addressInfo.addrType, AddrType::IPV4);
    EXPECT_EQ(addressInfo.ports, std::set<std::string>({"1/1"}));
    EXPECT_EQ(addressInfo.planeId, "planeB");
    EXPECT_TRUE(addressInfo.backupAddrs.empty());
}

TEST_F(AddressInfoParserTest, Ut_BinaryStream_When_BackupAddrCountExceedsLimit_Expect_Throw)
{
    BinaryStream binStream;
    IpAddress primaryAddr("192.168.100.100", AF_INET);
    primaryAddr.GetBinStream(binStream);
    binStream << static_cast<u32>(AddrType::IPV4);
    const size_t portsSize = 1;
    binStream << portsSize;
    binStream << std::string("1/1");
    binStream << std::string("planeB");
    binStream << static_cast<u32>(0);
    binStream << static_cast<size_t>(MAX_VALUE_BACKUP_ADDR_SIZE + 1U);

    EXPECT_THROW(AddressInfo addressInfo(binStream), InvalidParamsException);
}

TEST_F(AddressInfoParserTest, Ut_Deserialize_When_Ipv6BackupAddrValid_Expect_ParseBackupAddrs)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string addressInfoString = R"(
            {
                "addr_type": "IPV6",
                "addr": "2001:db8::1",
                "backup_addr": ["2001:db8::2", "2001:db8::3"],
                "ports": ["d2h"],
                "plane_id": "roce"
            }
    )";
    JsonParser addressInfoParser;
    AddressInfo addressInfo;
    addressInfoParser.ParseString(addressInfoString, addressInfo);

    ASSERT_EQ(addressInfo.backupAddrs.size(), 2U);
    EXPECT_EQ(addressInfo.backupAddrs[0], IpAddress("2001:db8::2", AF_INET6));
    EXPECT_EQ(addressInfo.backupAddrs[1], IpAddress("2001:db8::3", AF_INET6));
}

TEST_F(AddressInfoParserTest, Ut_Deserialize_When_BackupAddrCountAtLimit_Expect_Success)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    JsonParser addressInfoParser;
    AddressInfo addressInfo;
    const std::string addressInfoString = BuildIpv4AddressInfoWithBackupAddrCount(MAX_VALUE_BACKUP_ADDR_SIZE).dump();

    EXPECT_NO_THROW(addressInfoParser.ParseString(addressInfoString, addressInfo));
    EXPECT_EQ(addressInfo.backupAddrs.size(), MAX_VALUE_BACKUP_ADDR_SIZE);
}

TEST_F(AddressInfoParserTest, Ut_Deserialize_When_BackupAddrCountExceedsLimit_Expect_Throw)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    JsonParser addressInfoParser;
    AddressInfo addressInfo;
    const std::string addressInfoString
        = BuildIpv4AddressInfoWithBackupAddrCount(MAX_VALUE_BACKUP_ADDR_SIZE + 1U).dump();

    EXPECT_THROW(addressInfoParser.ParseString(addressInfoString, addressInfo), InvalidParamsException);
}

TEST_F(AddressInfoParserTest, Ut_Deserialize_When_EID_Expect_Success)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string addressInfoString = R"(
            {
              "addr_type": "EID",
              "addr": "000000000000002000100000df001007",
              "ports": ["0/6"],
              "plane_id": "plane0"
            }
    )";
    JsonParser addressInfoParser;
    AddressInfo addressInfo;
    addressInfoParser.ParseString(addressInfoString, addressInfo);

    AddressInfo addressInfo0;
    addressInfo0.addrType = AddrType::EID;
    addressInfo0.planeId = "plane0";
    addressInfo0.ports = {"0/6"};
    Eid eid0 = IpAddress::StrToEID("000000000000002000100000df001007");
    IpAddress ipAddress0(eid0);
    addressInfo0.addr = ipAddress0;

    EXPECT_EQ(addressInfo0.addr, addressInfo.addr);
    EXPECT_EQ(addressInfo0.addrType, addressInfo.addrType);
    EXPECT_EQ(addressInfo0.planeId, addressInfo.planeId);
    EXPECT_EQ(addressInfo0.ports, addressInfo.ports);
    EXPECT_TRUE(addressInfo.backupAddrs.empty());

    BinaryStream binStream;
    addressInfo.GetBinStream(binStream);
    AddressInfo addressInfo1(binStream);
    EXPECT_EQ(addressInfo1.addr, addressInfo.addr);
    EXPECT_EQ(addressInfo1.addrType, addressInfo.addrType);
    EXPECT_EQ(addressInfo1.planeId, addressInfo.planeId);
    EXPECT_EQ(addressInfo1.ports, addressInfo.ports);
    EXPECT_TRUE(addressInfo1.backupAddrs.empty());
}

TEST_F(AddressInfoParserTest, Ut_Deserialize_When_IPV6_Expect_Success)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string addressInfoString = R"(
            {
              "addr_type": "IPV6",
              "addr": "fe80:0000:0001:0000:0440:44ff:1233:5678",
              "ports": ["0/6"],
              "plane_id": "plane0"
            }
    )";
    JsonParser addressInfoParser;
    AddressInfo addressInfo;
    addressInfoParser.ParseString(addressInfoString, addressInfo);

    AddressInfo addressInfo0;
    addressInfo0.addrType = AddrType::IPV6;
    addressInfo0.planeId = "plane0";
    addressInfo0.ports = {"0/6"};
    IpAddress ipAddress0("fe80:0000:0001:0000:0440:44ff:1233:5678", AF_INET6);
    addressInfo0.addr = ipAddress0;

    EXPECT_EQ(addressInfo0.addr, addressInfo.addr);
    EXPECT_EQ(addressInfo0.addrType, addressInfo.addrType);
    EXPECT_EQ(addressInfo0.planeId, addressInfo.planeId);
    EXPECT_EQ(addressInfo0.ports, addressInfo.ports);
}

TEST_F(AddressInfoParserTest, Ut_Deserialize_When_EmptyAddr_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));
    MOCKER(RptInputErr).stubs().will(returnValue(HCCL_SUCCESS));

    std::string addressInfoString = R"(
            {
                "addr_type": "IPV4",
                "addr": "",
                "ports": [ "1/1", "1/2" ],
                "plane_id": "planeB"
            }
    )";
    JsonParser addressInfoParser;
    AddressInfo addressInfo;
    EXPECT_THROW(addressInfoParser.ParseString(addressInfoString, addressInfo), InvalidParamsException);
}

class AddressInfoParserInvalidTest :
    public AddressInfoParserTest,
    public testing::WithParamInterface<InvalidAddressInfoCase> {};

TEST_P(AddressInfoParserInvalidTest, Ut_Deserialize_When_InvalidInput_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    JsonParser addressInfoParser;
    AddressInfo addressInfo;

    EXPECT_THROW(addressInfoParser.ParseString(GetParam().addressInfoString, addressInfo), InvalidParamsException);
}

INSTANTIATE_TEST_SUITE_P(
    AddressInfoInvalidInputs, AddressInfoParserInvalidTest,
    testing::Values(
        InvalidAddressInfoCase{
            "InvalidEidLower",
            R"({"addr_type":"EID","addr":"000000000000002000100000df0010g7","ports":["0/6"],"plane_id":"plane0"})"},
        InvalidAddressInfoCase{
            "InvalidEidTooLong",
            R"({"addr_type":"EID","addr":"000000000000002000100000df0010077","ports":["0/6"],"plane_id":"plane0"})"},
        InvalidAddressInfoCase{
            "InvalidIpv6DoubleColonAtBothEnds",
            R"({"addr_type":"IPV6","addr":"::1234:5678::","ports":["0/6"],"plane_id":"plane0"})"},
        InvalidAddressInfoCase{
            "InvalidIpv6EmbeddedIpv4",
            R"({"addr_type":"IPV6","addr":"fe80::192.168.256.1","ports":["0/6"],"plane_id":"plane0"})"},
        InvalidAddressInfoCase{
            "InvalidIpv6GroupTooLong",
            R"({"addr_type":"IPV6","addr":"fe80:00000:0001:0000:0440:44ff:1233:5678","ports":["0/6"],"plane_id":"plane0"})"},
        InvalidAddressInfoCase{
            "InvalidAddrType",
            R"({"addr_type":"ipv4","addr":"192.168.100.100","ports":["1/1","1/2"],"plane_id":"planeB"})"},
        InvalidAddressInfoCase{
            "InvalidAddr", R"({"addr_type":"IPV4","addr":"192.168.100","ports":["1/1","1/2"],"plane_id":"planeB"})"},
        InvalidAddressInfoCase{
            "EmptyAddr", R"({"addr_type":"IPV4","addr":"","ports":["1/1","1/2"],"plane_id":"planeB"})"},
        InvalidAddressInfoCase{
            "InvalidPort",
            R"({"addr_type":"IPV4","addr":"192.168.100.100","ports":["9999999999999999/9999999999999999"],"plane_id":"planeB"})"},
        InvalidAddressInfoCase{
            "InvalidIpv4TooLong",
            R"({"addr_type":"IPV4","addr":"192.168.100.1000","ports":["1/1","1/2"],"plane_id":"planeB"})"},
        InvalidAddressInfoCase{
            "InvalidIpv4LeadingZero",
            R"({"addr_type":"IPV4","addr":"01.2.3.4","ports":["1/1","1/2"],"plane_id":"planeB"})"},
        InvalidAddressInfoCase{
            "InvalidIpv4HasAlpha",
            R"({"addr_type":"IPV4","addr":"A.168.1.2.3","ports":["1/1","1/2"],"plane_id":"planeB"})"},
        InvalidAddressInfoCase{
            "BackupAddrNotArray", R"({"addr_type":"IPV4","addr":"192.168.100.100",)"
                                  R"("backup_addr":"192.168.100.101","ports":["1/1"],"plane_id":"planeB"})"},
        InvalidAddressInfoCase{
            "BackupAddrElementNotString",
            R"({"addr_type":"IPV4","addr":"192.168.100.100","backup_addr":[101],"ports":["1/1"],"plane_id":"planeB"})"},
        InvalidAddressInfoCase{
            "InvalidBackupAddr", R"({"addr_type":"IPV4","addr":"192.168.100.100",)"
                                 R"("backup_addr":["192.168.100"],"ports":["1/1"],"plane_id":"planeB"})"},
        InvalidAddressInfoCase{
            "EidBackupAddrNotSupported",
            R"({"addr_type":"EID","addr":"000000000000002000100000df001007",)"
            R"("backup_addr":["000000000000002000100000df001008"],"ports":["0/6"],"plane_id":"plane0"})"},
        InvalidAddressInfoCase{
            "InvalidPorts",
            R"({"addr_type":"IPV4","addr":"192.168.100.100","ports":["1/1","1/2","1/3","1/4","1/5","1/6","1/7","1/8","1/9","1/10","1/11","1/12","1/13","1/14","1/15","1/16","1/17","1/18"],"plane_id":"planeB"})"}),
    [](const testing::TestParamInfo<InvalidAddressInfoCase>& info) {
        return info.param.name;
    });

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

TEST_F(AddressInfoParserTest, St_Deserialize_When_Normal_Expect_Success)
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
}

TEST_F(AddressInfoParserTest, St_Deserialize_When_EID_Expect_Success)
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

    BinaryStream binStream;
    addressInfo.GetBinStream(binStream);
    AddressInfo addressInfo1(binStream);
    EXPECT_EQ(addressInfo1.addr, addressInfo.addr);
    EXPECT_EQ(addressInfo1.addrType, addressInfo.addrType);
    EXPECT_EQ(addressInfo1.planeId, addressInfo.planeId);
    EXPECT_EQ(addressInfo1.ports, addressInfo.ports);
}

TEST_F(AddressInfoParserTest, St_Deserialize_When_IPV6_Expect_Success)
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

TEST_F(AddressInfoParserTest, St_Deserialize_When_InvalidAddrType_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string addressInfoString = R"(
            {
                "addr_type": "ipv4",
                "addr": "192.168.100.100",
                "ports": [ "1/1", "1/2" ],
                "plane_id": "planeB"
            }
    )";

    JsonParser addressInfoParser;
    AddressInfo addressInfo;

    EXPECT_THROW(addressInfoParser.ParseString(addressInfoString, addressInfo), InvalidParamsException);
}

TEST_F(AddressInfoParserTest, St_Deserialize_When_InvalidAddr_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string addressInfoString = R"(
            {
                "addr_type": "IPV4",
                "addr": "192.168.100",
                "ports": [ "1/1", "1/2" ],
                "plane_id": "planeB"
            }
    )";

    JsonParser addressInfoParser;
    AddressInfo addressInfo;

    EXPECT_THROW(addressInfoParser.ParseString(addressInfoString, addressInfo), InvalidParamsException);
}

TEST_F(AddressInfoParserTest, St_Deserialize_When_InvalidPort_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string addressInfoString = R"(
            {
                "addr_type": "IPV4",
                "addr": "192.168.100.100",
                "ports": ["9999999999999999/9999999999999999"],
                "plane_id": "planeB"
            }
    )";

    JsonParser addressInfoParser;
    AddressInfo addressInfo;

    EXPECT_THROW(addressInfoParser.ParseString(addressInfoString, addressInfo), InvalidParamsException);
}

TEST_F(AddressInfoParserTest, St_Deserialize_When_InvalidPorts_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string addressInfoString = R"(
            {
                "addr_type": "IPV4",
                "addr": "192.168.100.100",
                "ports": ["1/1", "1/2" ,"1/3", "1/4" ,"1/5", "1/6" ,"1/7", "1/8" ,"1/9", "1/10" ,"1/11", "1/12" ,"1/13", "1/14" ,"1/15", "1/16" ,"1/17", "1/18"],
                "plane_id": "planeB"
            }
    )";

    JsonParser addressInfoParser;
    AddressInfo addressInfo;

    EXPECT_THROW(addressInfoParser.ParseString(addressInfoString, addressInfo), InvalidParamsException);
}

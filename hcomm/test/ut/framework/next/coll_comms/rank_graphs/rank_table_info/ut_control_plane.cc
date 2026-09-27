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

#include "control_plane.h"

using namespace Hccl;

class ControlPlaneParserTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ControlPlaneParserTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "ControlPlaneParserTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in ControlPlaneParserTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in ControlPlaneParserTest TearDown" << std::endl;
    }
};

TEST_F(ControlPlaneParserTest, Ut_Deserialize_When_Normal_Expect_Success)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string controlPlaneString = R"(
        {
                "addr_type": "IPV4",
                "addr": "192.168.100.100",
                "listen_port": 8000
        }
    )";
    JsonParser controlPlaneParser;
    ControlPlane controlPlane;
    controlPlaneParser.ParseString(controlPlaneString, controlPlane);

    ControlPlane controlPlane0;
    controlPlane0.addrType = AddrType::IPV4;
    IpAddress ipAddress0("192.168.100.100", AF_INET);
    controlPlane0.addr = ipAddress0;
    controlPlane0.listenPort = 8000;
    controlPlane.Describe();

    EXPECT_EQ(controlPlane0.addrType, controlPlane.addrType);
    EXPECT_EQ(controlPlane0.addr, controlPlane.addr);
    EXPECT_EQ(controlPlane0.listenPort, controlPlane.listenPort);
}

TEST_F(ControlPlaneParserTest, Ut_Deserialize_When_EID_Expect_Success)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string controlPlaneString = R"(
        {
              "addr_type": "EID",
              "addr": "000000000000002000100000df001007",
              "listen_port": 8000
        }
    )";
    JsonParser controlPlaneParser;
    ControlPlane controlPlane;
    controlPlaneParser.ParseString(controlPlaneString, controlPlane);

    ControlPlane controlPlane0;
    controlPlane0.addrType = AddrType::EID;
    Eid eid0 = IpAddress::StrToEID("000000000000002000100000df001007");
    IpAddress ipAddress0(eid0);
    controlPlane0.addr = ipAddress0;
    controlPlane0.listenPort = 8000;
    controlPlane.Describe();

    EXPECT_EQ(controlPlane0.addrType, controlPlane.addrType);
    EXPECT_EQ(controlPlane0.addr, controlPlane.addr);
    EXPECT_EQ(controlPlane0.listenPort, controlPlane.listenPort);

    BinaryStream binStream;
    controlPlane.GetBinStream(binStream);
    ControlPlane controlPlane1(binStream);
    EXPECT_EQ(controlPlane1.addr, controlPlane.addr);
    EXPECT_EQ(controlPlane1.addrType, controlPlane.addrType);
    EXPECT_EQ(controlPlane1.listenPort, controlPlane.listenPort);
}

TEST_F(ControlPlaneParserTest, Ut_Deserialize_When_IPV6_Expect_Success)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string controlPlaneString = R"(
        {
              "addr_type": "IPV6",
              "addr": "fe80:0000:0001:0000:0440:44ff:1233:5678",
              "listen_port": 8000
        }
    )";
    JsonParser controlPlaneParser;
    ControlPlane controlPlane;
    controlPlaneParser.ParseString(controlPlaneString, controlPlane);

    ControlPlane controlPlane0;
    controlPlane0.addrType = AddrType::IPV6;
    IpAddress ipAddress0("fe80:0000:0001:0000:0440:44ff:1233:5678", AF_INET6);
    controlPlane0.addr = ipAddress0;
    controlPlane0.listenPort = 8000;
    controlPlane.Describe();

    EXPECT_EQ(controlPlane0.addrType, controlPlane.addrType);
    EXPECT_EQ(controlPlane0.addr, controlPlane.addr);
    EXPECT_EQ(controlPlane0.listenPort, controlPlane.listenPort);
}

TEST_F(ControlPlaneParserTest, Ut_Deserialize_When_InvalidAddrType_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string controlPlaneString = R"(
        {
                "addr_type": "IP",
                "addr": "192.168.100.100",
                "listen_port": 8000
        }
    )";

    JsonParser controlPlaneParser;
    ControlPlane controlPlane;

    EXPECT_THROW(controlPlaneParser.ParseString(controlPlaneString, controlPlane), InvalidParamsException);
}

TEST_F(ControlPlaneParserTest, Ut_Deserialize_When_InvalidAddr_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string controlPlaneString = R"(
        {
                "addr_type": "IPV4",
                "addr": "192.168.100",
                "listen_port": 8000
        }
    )";

    JsonParser controlPlaneParser;
    ControlPlane controlPlane;

    EXPECT_THROW(controlPlaneParser.ParseString(controlPlaneString, controlPlane), InvalidParamsException);
}

TEST_F(ControlPlaneParserTest, Ut_Deserialize_When_InvalidListenPort_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string controlPlaneString = R"(
        {
                "addr_type": "IPV4",
                "addr": "192.168.100.100",
                "listen_port": 66666
        }
    )";

    JsonParser controlPlaneParser;
    ControlPlane controlPlane;

    EXPECT_THROW(controlPlaneParser.ParseString(controlPlaneString, controlPlane), InvalidParamsException);
}

TEST_F(ControlPlaneParserTest, Ut_Deserialize_EidToAddr_Length_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string controlPlaneString = "000000000000002000100000df0010g70";
    ControlPlane controlPlane;

    EXPECT_THROW(controlPlane.EidToAddr(controlPlaneString), InvalidParamsException);
}

TEST_F(ControlPlaneParserTest, Ut_Deserialize_EidToAddr_Format_Expect_Exception)
{
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

    std::string controlPlaneString = "000000000000002000100000df0010g7";
    ControlPlane controlPlane;

    EXPECT_THROW(controlPlane.EidToAddr(controlPlaneString), InvalidParamsException);
}

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
#include "ip_address.h"

using namespace Hccl;

class IpAddressTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "IpAddressTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "IpAddressTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in IpAddressTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in IpAddressTest TearDown" << std::endl;
    }
};

TEST_F(IpAddressTest, Ut_Constructor_When_Default_Expect_InitializedWithDefaults)
{
    Hccl::IpAddress ip;
    EXPECT_EQ(ip.GetFamily(), AF_INET);
    EXPECT_EQ(ip.GetScopeID(), 0);
    EXPECT_EQ(ip.GetBinaryAddress().addr.s_addr, 0);
}

TEST_F(IpAddressTest, Ut_Constructor_When_WithIPv4String_Expect_InitializedWithIPv4)
{
    Hccl::IpAddress ip("192.168.1.1");
    EXPECT_EQ(ip.GetFamily(), AF_INET);
    EXPECT_EQ(ip.GetIpStr(), "192.168.1.1");
}

TEST_F(IpAddressTest, Ut_Constructor_When_WithIPv6String_Expect_InitializedWithIPv6)
{
    Hccl::IpAddress ip("2001:0db8:85a3:0000:0000:8a2e:0370:7334");
    EXPECT_EQ(ip.GetFamily(), AF_INET6);
    EXPECT_EQ(ip.GetIpStr(), "2001:db8:85a3::8a2e:370:7334");
}

TEST_F(IpAddressTest, Ut_IsIPv4_When_ValidIPv4_Expect_True)
{
    EXPECT_TRUE(Hccl::IpAddress::IsIPv4("192.168.1.1"));
    EXPECT_TRUE(Hccl::IpAddress::IsIPv4("10.0.0.1"));
}

TEST_F(IpAddressTest, Ut_IsIPv4_When_InvalidIPv4_Expect_False)
{
    EXPECT_FALSE(Hccl::IpAddress::IsIPv4("256.256.256.256"));
    EXPECT_FALSE(Hccl::IpAddress::IsIPv4("192.168.1"));
}

TEST_F(IpAddressTest, Ut_IsIPv6_When_ValidIPv6_Expect_True)
{
    EXPECT_TRUE(Hccl::IpAddress::IsIPv6("2001:0db8:85a3:0000:0000:8a2e:0370:7334"));
    EXPECT_TRUE(Hccl::IpAddress::IsIPv6("::1"));
    EXPECT_TRUE(Hccl::IpAddress::IsIPv6("fe80::204:61ff:254.157.241.86"));
}

TEST_F(IpAddressTest, Ut_IsIPv6_When_InvalidIPv6_Expect_False)
{
    EXPECT_FALSE(Hccl::IpAddress::IsIPv6("192.168.1.1"));
    EXPECT_FALSE(Hccl::IpAddress::IsIPv6("2001:0db8:85a3:0000:0000:8a2e:0370:7334:1234"));
    EXPECT_FALSE(Hccl::IpAddress::IsIPv6("::1 "));
    EXPECT_FALSE(Hccl::IpAddress::IsIPv6(std::string("::1\0invalid", 11)));
}

TEST_F(IpAddressTest, Ut_GetIpStr_When_IPv4_Expect_CorrectString)
{
    Hccl::IpAddress ip("192.168.1.1");
    EXPECT_EQ(ip.GetIpStr(), "192.168.1.1");
}

TEST_F(IpAddressTest, Ut_GetIpStr_When_IPv6_Expect_CorrectString)
{
    Hccl::IpAddress ip("2001:0db8:85a3:0000:0000:8a2e:0370:7334");
    EXPECT_EQ(ip.GetIpStr(), "2001:db8:85a3::8a2e:370:7334");
}

TEST_F(IpAddressTest, Ut_GetEid_When_IPv4_Expect_CorrectEid)
{
    Hccl::IpAddress ip("192.168.1.1");
    Hccl::Eid eid = ip.GetEid();
    EXPECT_EQ(eid.in4.prefix, Hccl::URMA_EID_IPV4_PREFIX);
    EXPECT_EQ(eid.in4.addr, inet_addr("192.168.1.1"));
}

TEST_F(IpAddressTest, Ut_GetEid_When_IPv6_Expect_CorrectEid)
{
    Hccl::IpAddress ip("2001:0db8:85a3:0000:0000:8a2e:0370:7334");
    Hccl::Eid eid = ip.GetEid();
    EXPECT_EQ(be64toh(eid.in6.subnetPrefix), 0x20010db885a30000);
    EXPECT_EQ(be64toh(eid.in6.interfaceId), 0x00008a2e03707334);
}

TEST_F(IpAddressTest, Ut_IsEID_When_ValidEID_Expect_True)
{
    EXPECT_TRUE(Hccl::IpAddress::IsEID("000000000000002000100000dfdf1234"));
    EXPECT_TRUE(Hccl::IpAddress::IsEID("1234567890abcdef1234567890abcdef"));
    EXPECT_TRUE(Hccl::IpAddress::IsEID("1234567890ABCDEF1234567890ABCDEF"));
    EXPECT_TRUE(Hccl::IpAddress::IsEID("aBcDeF0123456789AbCdEf0123456789"));
}

TEST_F(IpAddressTest, Ut_IsEID_When_InvalidEID_Expect_False)
{
    EXPECT_FALSE(Hccl::IpAddress::IsEID("000000000000002000100000dfdf123"));   // 长度不足
    EXPECT_FALSE(Hccl::IpAddress::IsEID("000000000000002000100000dfdf12345")); // 长度过长
    EXPECT_FALSE(Hccl::IpAddress::IsEID("000000000000002000100000dfdf123G"));  // 包含非十六进制字符
    EXPECT_FALSE(Hccl::IpAddress::IsEID("000000000000002000100000dfdf123g"));  // 包含非十六进制字符
    EXPECT_FALSE(Hccl::IpAddress::IsEID("000000000000002000100000dfdf123:"));  // 包含分隔符
    EXPECT_FALSE(Hccl::IpAddress::IsEID("000000000000002000100000dfdf123 "));  // 包含空格
    EXPECT_FALSE(Hccl::IpAddress::IsEID("000000000000002000100000dfdf123\t")); // 包含制表符
    EXPECT_FALSE(Hccl::IpAddress::IsEID("000000000000002000100000dfdf123\n")); // 包含换行符
}

TEST_F(IpAddressTest, Ut_StrToEID_When_ValidEID_Expect_CorrectEid)
{
    std::string eidStr = "000000000000002000100000dfdf1234";
    Hccl::Eid eid = Hccl::IpAddress::StrToEID(eidStr);

    // 验证Eid的raw数组是否正确
    for (size_t i = 0; i < Hccl::URMA_EID_LEN; ++i) {
        EXPECT_EQ(eid.raw[i], static_cast<uint8_t>(std::stoi(eidStr.substr(i * 2, 2), nullptr, 16)));
    }
}

TEST_F(IpAddressTest, Ut_Constructor_When_WithEidInput_Expect_InitializedWithIPv6)
{
    Hccl::Eid eidInput;
    eidInput.in6.subnetPrefix = 0x0000000000000020;
    eidInput.in6.interfaceId = 0x00100000dfdf1234;

    Hccl::IpAddress ip(eidInput);
    EXPECT_EQ(ip.GetFamily(), AF_INET6);
    EXPECT_EQ(ip.GetEid().in6.subnetPrefix, eidInput.in6.subnetPrefix);
    EXPECT_EQ(ip.GetEid().in6.interfaceId, eidInput.in6.interfaceId);
}

TEST_F(IpAddressTest, Ut_Constructor_When_WithEidInput_Expect_CorrectEid)
{
    Hccl::Eid eidInput;
    eidInput.in6.subnetPrefix = 0x0000000000000020;
    eidInput.in6.interfaceId = 0x00100000dfdf1234;

    Hccl::IpAddress ip(eidInput);
    Hccl::Eid eid = ip.GetEid();
    EXPECT_EQ(eid.in6.subnetPrefix, eidInput.in6.subnetPrefix);
    EXPECT_EQ(eid.in6.interfaceId, eidInput.in6.interfaceId);
}

TEST_F(IpAddressTest, Ut_Constructor_When_Input_Eid_Expect_Equal_To_IPV6)
{
    Eid eid0 = IpAddress::StrToEID("000000000000002000100000dfdf1234");
    IpAddress ipAddress0(eid0);
    std::string ipv6 = "0000:0000:0000:0020:0010:0000:dfdf:1234";
    IpAddress ipAddress1(ipv6, AF_INET6);

    EXPECT_EQ(ipAddress0.GetFamily(), AF_INET6);
    for (int i = 0; i < URMA_EID_LEN; i++) {
        EXPECT_EQ(ipAddress0.GetBinaryAddress().addr6.s6_addr[i], ipAddress1.GetBinaryAddress().addr6.s6_addr[i]);
    }
}

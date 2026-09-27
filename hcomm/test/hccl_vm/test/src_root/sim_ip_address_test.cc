/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

#include "sim_common_defs.h"
#include "sim_ip_address.h"

using namespace HcclSim;

class IpAddressTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(IpAddressTest, DefaultConstructor)
{
    IpAddress addr;
    EXPECT_NO_THROW(addr.GetIpStr());
    EXPECT_NO_THROW(addr.EidToHexString());
    Eid eid = addr.GetEid();
    EXPECT_EQ(eid.in4.prefix, 0u);
}

TEST_F(IpAddressTest, U32Constructor)
{
    IpAddress addr(0x0100007F);
    std::string ip = addr.GetIpStr();
    EXPECT_EQ(ip, "127.0.0.1");
}

TEST_F(IpAddressTest, U32Constructor_Localhost)
{
    IpAddress addr(0x0100007F);
    EXPECT_EQ(addr.GetIpStr(), "127.0.0.1");
    Eid eid = addr.GetEid();
    EXPECT_EQ(eid.in4.addr, 0x0100007Fu);
}

TEST_F(IpAddressTest, StringConstructor_Ipv4)
{
    IpAddress addr("192.168.1.1");
    EXPECT_EQ(addr.GetIpStr(), "192.168.1.1");
}

TEST_F(IpAddressTest, StringConstructor_Ipv4_ExplicitFamily)
{
    IpAddress addr("10.0.0.1", AF_INET);
    EXPECT_EQ(addr.GetIpStr(), "10.0.0.1");
}

TEST_F(IpAddressTest, StringConstructor_Ipv6)
{
    IpAddress addr("::1", AF_INET6);
    EXPECT_EQ(addr.GetIpStr(), "::1");
}

TEST_F(IpAddressTest, StringConstructor_Ipv6_Full)
{
    IpAddress addr("2001:db8::1", AF_INET6);
    EXPECT_EQ(addr.GetIpStr(), "2001:db8::1");
}

TEST_F(IpAddressTest, StringConstructor_UnsupportedFamily)
{
    EXPECT_THROW(IpAddress("127.0.0.1", AF_UNIX), std::invalid_argument);
}

TEST_F(IpAddressTest, StringConstructor_InvalidAddress) { EXPECT_THROW(IpAddress("not_an_ip"), std::invalid_argument); }

TEST_F(IpAddressTest, BinaryAddrConstructor_Ipv4)
{
    BinaryAddr ba{};
    inet_pton(AF_INET, "172.16.0.1", &ba.addr);
    IpAddress addr(ba, AF_INET);
    EXPECT_EQ(addr.GetIpStr(), "172.16.0.1");
}

TEST_F(IpAddressTest, BinaryAddrConstructor_Ipv6)
{
    BinaryAddr ba{};
    inet_pton(AF_INET6, "fe80::1", &ba.addr6);
    IpAddress addr(ba, AF_INET6);
    EXPECT_EQ(addr.GetIpStr(), "fe80::1");
}

TEST_F(IpAddressTest, BinaryAddrConstructor_WithScopeId)
{
    BinaryAddr ba{};
    inet_pton(AF_INET6, "ff02::1", &ba.addr6);
    IpAddress addr(ba, AF_INET6, 5);
    EXPECT_EQ(addr.GetIpStr(), "ff02::1");
}

TEST_F(IpAddressTest, EidConstructor)
{
    Eid eid{};
    eid.raw[0] = 0x20;
    eid.raw[1] = 0x01;
    eid.raw[15] = 0x01;
    IpAddress addr(eid);
    Eid result = addr.GetEid();
    EXPECT_EQ(result.raw[0], 0x20);
    EXPECT_EQ(result.raw[1], 0x01);
    EXPECT_EQ(result.raw[15], 0x01);
}

TEST_F(IpAddressTest, GetBinaryAddress)
{
    IpAddress addr("8.8.8.8");
    BinaryAddr ba = addr.GetBinaryAddress();
    BinaryAddr ba2 = addr.GetBinaryAddress();
    EXPECT_EQ(ba.addr.s_addr, ba2.addr.s_addr);
}

TEST_F(IpAddressTest, EidToHexString)
{
    IpAddress addr("1.1.1.1");
    std::string hex = addr.EidToHexString();
    EXPECT_FALSE(hex.empty());
    EXPECT_EQ(hex.size(), 32u);
}

TEST_F(IpAddressTest, EidToHexString_Ipv6)
{
    IpAddress addr("::1", AF_INET6);
    std::string hex = addr.EidToHexString();
    EXPECT_FALSE(hex.empty());
}

TEST_F(IpAddressTest, GetIpStr_DefaultIsZeroAddr)
{
    IpAddress addr;
    EXPECT_EQ(addr.GetIpStr(), "0.0.0.0");
}

TEST_F(IpAddressTest, IpAddress_EmptyStringConstructor) { EXPECT_THROW(IpAddress addr(""), std::invalid_argument); }

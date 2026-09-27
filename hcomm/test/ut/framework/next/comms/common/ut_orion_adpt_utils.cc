/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <arpa/inet.h>
#include <cstring>

#include "gtest/gtest.h"

#include "hccl_net_dev_defs.h"
#include "hcomm_res_defs.h"
#include "orion_adpt_utils.h"

using namespace hcomm;

static void FillIpv4(CommAddr& ca, const char* dotted)
{
    ca.type = COMM_ADDR_TYPE_IP_V4;
    ASSERT_EQ(inet_pton(AF_INET, dotted, &ca.addr), 1);
}

static void FillIpv6(CommAddr& ca, const char* dotted)
{
    ca.type = COMM_ADDR_TYPE_IP_V6;
    ASSERT_EQ(inet_pton(AF_INET6, dotted, &ca.addr6), 1);
}

TEST(UtOrionAdptUtils, CommAddrToIpAddress_Unsupported_Returns_NotSupport)
{
    CommAddr ca{};
    ca.type = COMM_ADDR_TYPE_ID;
    Hccl::IpAddress ip;
    EXPECT_EQ(CommAddrToIpAddress(ca, ip), HCCL_E_NOT_SUPPORT);
}

TEST(UtOrionAdptUtils, CommAddrToIpAddress_V4_V6_Eid_RoundTrip_IpAddressToCommAddr)
{
    CommAddr v4{};
    FillIpv4(v4, "10.11.12.13");
    Hccl::IpAddress ip4;
    ASSERT_EQ(CommAddrToIpAddress(v4, ip4), HCCL_SUCCESS);
    CommAddr back4{};
    ASSERT_EQ(IpAddressToCommAddr(ip4, back4), HCCL_SUCCESS);
    EXPECT_EQ(back4.type, COMM_ADDR_TYPE_IP_V4);
    EXPECT_EQ(0, memcmp(&back4.addr, &v4.addr, sizeof(v4.addr)));

    CommAddr v6{};
    FillIpv6(v6, "2001:db8::1");
    Hccl::IpAddress ip6;
    ASSERT_EQ(CommAddrToIpAddress(v6, ip6), HCCL_SUCCESS);
    CommAddr back6{};
    ASSERT_EQ(IpAddressToCommAddr(ip6, back6), HCCL_SUCCESS);
    EXPECT_EQ(back6.type, COMM_ADDR_TYPE_IP_V6);
    EXPECT_EQ(0, memcmp(&back6.addr6, &v6.addr6, sizeof(v6.addr6)));

    CommAddr eid{};
    eid.type = COMM_ADDR_TYPE_EID;
    (void)memset(eid.eid, 0xab, COMM_ADDR_EID_LEN);
    Hccl::IpAddress ipEid;
    ASSERT_EQ(CommAddrToIpAddress(eid, ipEid), HCCL_SUCCESS);
}

TEST(UtOrionAdptUtils, CommAddrTypeToHcclAddressType_V4_V6_Default)
{
    HcclAddressType t = HCCL_ADDR_TYPE_RESERVED;
    ASSERT_EQ(CommAddrTypeToHcclAddressType(COMM_ADDR_TYPE_IP_V4, t), HCCL_SUCCESS);
    EXPECT_EQ(t, HCCL_ADDR_TYPE_IP_V4);
    ASSERT_EQ(CommAddrTypeToHcclAddressType(COMM_ADDR_TYPE_IP_V6, t), HCCL_SUCCESS);
    EXPECT_EQ(t, HCCL_ADDR_TYPE_IP_V6);
    EXPECT_EQ(CommAddrTypeToHcclAddressType(COMM_ADDR_TYPE_ID, t), HCCL_E_NOT_FOUND);
}

TEST(UtOrionAdptUtils, CommProtocolToLinkProtocol_AllSupported_And_Invalid)
{
    struct Case {
        CommProtocol p;
        Hccl::LinkProtocol expect;
    } cases[] = {
        {COMM_PROTOCOL_UB_CTP, Hccl::LinkProtocol::UB_CTP}, {COMM_PROTOCOL_UBC_TP, Hccl::LinkProtocol::UB_TP},
        {COMM_PROTOCOL_ROCE, Hccl::LinkProtocol::ROCE},     {COMM_PROTOCOL_HCCS, Hccl::LinkProtocol::HCCS},
        {COMM_PROTOCOL_UB_MEM, Hccl::LinkProtocol::UB_MEM}, {COMM_PROTOCOL_UB_RTP, Hccl::LinkProtocol::UB_RTP},
    };
    for (const auto& c : cases) {
        Hccl::LinkProtocol lp = Hccl::LinkProtocol::INVALID;
        ASSERT_EQ(CommProtocolToLinkProtocol(c.p, lp), HCCL_SUCCESS);
        EXPECT_TRUE(lp == c.expect) << static_cast<int>(c.p);
    }
    Hccl::LinkProtocol bad = Hccl::LinkProtocol::ROCE;
    EXPECT_EQ(CommProtocolToLinkProtocol(static_cast<CommProtocol>(99999), bad), HCCL_E_PARA);
}

TEST(UtOrionAdptUtils, CommProtocolLegacyNamesKeepCompatibleValues)
{
    EXPECT_EQ(COMM_PROTOCOL_UBC_CTP, COMM_PROTOCOL_UB_CTP);
    EXPECT_EQ(COMM_PROTOCOL_UBG, COMM_PROTOCOL_UB_RTP);
}

TEST(UtOrionAdptUtils, BuildDefaultLinkData_Returns_RoceHostNet)
{
    Hccl::LinkData ld = BuildDefaultLinkData();
    EXPECT_TRUE(ld.GetLinkProtocol() == Hccl::LinkProtocol::ROCE);
    EXPECT_TRUE(ld.GetType() == Hccl::PortDeploymentType::HOST_NET);
}

static EndpointDesc MakeEpDev(CommProtocol proto, const CommAddr& addr)
{
    EndpointDesc ep{};
    ep.protocol = proto;
    ep.commAddr = addr;
    ep.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    ep.loc.device.devPhyId = 3U;
    return ep;
}

static EndpointDesc MakeEpHost(CommProtocol proto, const CommAddr& addr)
{
    EndpointDesc ep{};
    ep.protocol = proto;
    ep.commAddr = addr;
    ep.loc.locType = ENDPOINT_LOC_TYPE_HOST;
    ep.loc.host.id = 1U;
    return ep;
}

TEST(UtOrionAdptUtils, EndpointDescPairToLinkData_DeviceAndHost_Ipv4_InvalidLoc)
{
    CommAddr a{};
    CommAddr b{};
    FillIpv4(a, "172.16.0.1");
    FillIpv4(b, "172.16.0.2");

    EndpointDesc locDev = MakeEpDev(COMM_PROTOCOL_ROCE, a);
    EndpointDesc rmtDev = MakeEpDev(COMM_PROTOCOL_ROCE, b);
    Hccl::LinkData ld1 = BuildDefaultLinkData();
    ASSERT_EQ(EndpointDescPairToLinkData(locDev, rmtDev, ld1, 1U), HCCL_SUCCESS);
    EXPECT_TRUE(ld1.GetType() == Hccl::PortDeploymentType::DEV_NET);

    EndpointDesc locHost = MakeEpHost(COMM_PROTOCOL_HCCS, a);
    EndpointDesc rmtHost = MakeEpHost(COMM_PROTOCOL_HCCS, b);
    Hccl::LinkData ld2 = BuildDefaultLinkData();
    ASSERT_EQ(EndpointDescPairToLinkData(locHost, rmtHost, ld2, 0U), HCCL_SUCCESS);
    EXPECT_TRUE(ld2.GetType() == Hccl::PortDeploymentType::HOST_NET);

    EndpointDesc badLoc = locDev;
    badLoc.loc.locType = static_cast<EndpointLocType>(42);
    Hccl::LinkData ld3 = BuildDefaultLinkData();
    EXPECT_EQ(EndpointDescPairToLinkData(badLoc, rmtDev, ld3, 0U), HCCL_E_PARA);
}

TEST(UtOrionAdptUtils, EndpointDescPairToLinkDataWithRankIds_Ipv6_Success)
{
    CommAddr a{};
    CommAddr b{};
    FillIpv6(a, "fe80::1");
    FillIpv6(b, "fe80::2");
    EndpointDesc loc = MakeEpDev(COMM_PROTOCOL_ROCE, a);
    EndpointDesc rmt = MakeEpDev(COMM_PROTOCOL_ROCE, b);
    Hccl::LinkData ld = BuildDefaultLinkData();
    ASSERT_EQ(EndpointDescPairToLinkDataWithRankIds(7U, 8U, loc, rmt, ld, 3U, 4U), HCCL_SUCCESS);
    EXPECT_TRUE(ld.GetLinkProtocol() == Hccl::LinkProtocol::ROCE);
}

TEST(UtOrionAdptUtils, CheckUbSqDepth_LowerBound)
{
    DevBaseAttr devBaseAttr{};
    devBaseAttr.sqMaxDepth = 1024U;
    UbConnBuildContext ctx{};

    ctx.sqDepth = 15U;
    EXPECT_EQ(CheckUbSqDepth(ctx, devBaseAttr), HCCL_E_PARA);

    ctx.sqDepth = 16U;
    EXPECT_EQ(CheckUbSqDepth(ctx, devBaseAttr), HCCL_SUCCESS);
}

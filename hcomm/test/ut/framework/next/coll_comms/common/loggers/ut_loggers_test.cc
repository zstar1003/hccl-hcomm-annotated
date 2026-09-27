/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "comm_addr_logger.h"
#include "endpoint_logger.h"
#include "channel_logger.h"
#include "hccl/hccl_res.h"
#include "hcomm_res_defs.h"
#include <gtest/gtest.h>
#include <string>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <cstring>

namespace hcomm {
namespace logger {
    namespace {

        class CommAddrLoggerTest : public ::testing::Test {
        protected:
            CommAddr addr{};

            void SetUp() override { memset(&addr, 0, sizeof(addr)); }
        };

        TEST_F(CommAddrLoggerTest, GetTypeString_IPv4)
        {
            EXPECT_EQ(CommAddrLogger::GetTypeString(COMM_ADDR_TYPE_IP_V4), "IPv4");
        }

        TEST_F(CommAddrLoggerTest, GetTypeString_IPv6)
        {
            EXPECT_EQ(CommAddrLogger::GetTypeString(COMM_ADDR_TYPE_IP_V6), "IPv6");
        }

        TEST_F(CommAddrLoggerTest, GetTypeString_ID)
        {
            EXPECT_EQ(CommAddrLogger::GetTypeString(COMM_ADDR_TYPE_ID), "ID");
        }

        TEST_F(CommAddrLoggerTest, GetTypeString_EID)
        {
            EXPECT_EQ(CommAddrLogger::GetTypeString(COMM_ADDR_TYPE_EID), "EID");
        }

        TEST_F(CommAddrLoggerTest, GetTypeString_Unknown)
        {
            std::string result = CommAddrLogger::GetTypeString(static_cast<CommAddrType>(99));
            EXPECT_NE(result.find("Unknown"), std::string::npos);
        }

        TEST_F(CommAddrLoggerTest, ConvertIPv4_Normal)
        {
            struct in_addr addr;
            addr.s_addr = inet_addr("192.168.1.1");
            std::string result = CommAddrLogger::ConvertIPv4(addr);
            EXPECT_FALSE(result.empty());
        }

        TEST_F(CommAddrLoggerTest, ConvertIPv6_Normal)
        {
            struct in6_addr addr6;
            inet_pton(AF_INET6, "fe80::1", &addr6);
            std::string result = CommAddrLogger::ConvertIPv6(addr6);
            EXPECT_NE(result.find("fe80"), std::string::npos);
        }

        TEST_F(CommAddrLoggerTest, ConvertID_Normal)
        {
            std::string result = CommAddrLogger::ConvertID(0x12345678);
            EXPECT_FALSE(result.empty());
        }

        TEST_F(CommAddrLoggerTest, ConvertEID_Normal)
        {
            uint8_t eid[16] = {0};
            eid[0] = 0x12;
            eid[8] = 0x34;
            std::string result = CommAddrLogger::ConvertEID(eid);
            EXPECT_NE(result.find("eid["), std::string::npos);
        }

        TEST_F(CommAddrLoggerTest, ToString_IPv4)
        {
            addr.type = COMM_ADDR_TYPE_IP_V4;
            addr.addr.s_addr = inet_addr("10.0.0.1");
            std::string result = CommAddrLogger::ToString(addr);
            EXPECT_NE(result.find("IpAddress["), std::string::npos);
            EXPECT_NE(result.find("AF=v4"), std::string::npos);
        }

        TEST_F(CommAddrLoggerTest, ToString_IPv6)
        {
            addr.type = COMM_ADDR_TYPE_IP_V6;
            inet_pton(AF_INET6, "fe80::204:61ff:fe9d:f156", &addr.addr6);
            std::string result = CommAddrLogger::ToString(addr);
            EXPECT_NE(result.find("IpAddress["), std::string::npos);
            EXPECT_NE(result.find("AF=v6"), std::string::npos);
            EXPECT_NE(result.find("fe80"), std::string::npos);
        }

        TEST_F(CommAddrLoggerTest, ToString_ID)
        {
            addr.type = COMM_ADDR_TYPE_ID;
            addr.id = 0x12345678;
            std::string result = CommAddrLogger::ToString(addr);
            EXPECT_NE(result.find("IpAddress["), std::string::npos);
            EXPECT_NE(result.find("id:"), std::string::npos);
        }

        TEST_F(CommAddrLoggerTest, ToString_EID)
        {
            addr.type = COMM_ADDR_TYPE_EID;
            addr.eid[0] = 0xAB;
            addr.eid[8] = 0xCD;
            std::string result = CommAddrLogger::ToString(addr);
            EXPECT_NE(result.find("IpAddress["), std::string::npos);
            EXPECT_NE(result.find("eid["), std::string::npos);
        }

        TEST_F(CommAddrLoggerTest, ToString_Unknown)
        {
            addr.type = static_cast<CommAddrType>(99);
            std::string result = CommAddrLogger::ToString(addr);
            EXPECT_NE(result.find("Unknown"), std::string::npos);
        }

        TEST_F(CommAddrLoggerTest, Print_DoesNotCrash)
        {
            addr.type = COMM_ADDR_TYPE_IP_V4;
            addr.addr.s_addr = inet_addr("192.168.1.1");
            CommAddrLogger::Print(0, "localEndpoint", addr);
            SUCCEED();
        }

        class ChannelStatusUtilsTest : public ::testing::Test {};

        TEST_F(ChannelStatusUtilsTest, ToString_READY)
        {
            std::string result = ChannelStatusUtils::ToString(static_cast<int32_t>(3));
            EXPECT_NE(result.find("READY"), std::string::npos);
        }

        TEST_F(ChannelStatusUtilsTest, ToString_FAILED)
        {
            std::string result = ChannelStatusUtils::ToString(static_cast<int32_t>(4));
            EXPECT_NE(result.find("FAILED"), std::string::npos);
        }

        TEST_F(ChannelStatusUtilsTest, ToString_SOCKET_TIMEOUT)
        {
            std::string result = ChannelStatusUtils::ToString(static_cast<int32_t>(2));
            EXPECT_NE(result.find("TIMEOUT"), std::string::npos);
        }

        TEST_F(ChannelStatusUtilsTest, ToString_INIT)
        {
            std::string result = ChannelStatusUtils::ToString(static_cast<int32_t>(0));
            EXPECT_NE(result.find("INIT"), std::string::npos);
        }

        TEST_F(ChannelStatusUtilsTest, ToString_SOCKET_OK)
        {
            std::string result = ChannelStatusUtils::ToString(static_cast<int32_t>(1));
            EXPECT_NE(result.find("SOCKET_OK"), std::string::npos);
        }

        class ChannelLoggerTest : public ::testing::Test {
        protected:
            HcclChannelDesc desc{};
            HcclChannelDesc descRoce{};
            HcclChannelDesc descUbMem{};

            void SetUp() override
            {
                memset(&desc, 0, sizeof(desc));
                memset(&descRoce, 0, sizeof(descRoce));
                memset(&descUbMem, 0, sizeof(descUbMem));

                desc.remoteRank = 1;
                desc.channelProtocol = COMM_PROTOCOL_HCCS;
                desc.notifyNum = 2;
                desc.memHandleNum = 3;
                desc.localEndpoint.commAddr.type = COMM_ADDR_TYPE_IP_V4;
                desc.localEndpoint.commAddr.addr.s_addr = inet_addr("10.0.0.1");
                desc.localEndpoint.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
                desc.remoteEndpoint.commAddr.type = COMM_ADDR_TYPE_IP_V4;
                desc.remoteEndpoint.commAddr.addr.s_addr = inet_addr("10.0.0.2");
                desc.remoteEndpoint.loc.locType = ENDPOINT_LOC_TYPE_HOST;

                descRoce = desc;
                descRoce.channelProtocol = COMM_PROTOCOL_ROCE;
                descRoce.roceAttr.queueNum = 4;
                descRoce.roceAttr.retryCnt = 3;
                descRoce.roceAttr.retryInterval = 20;
                descRoce.roceAttr.tc = 0;
                descRoce.roceAttr.sl = 0;

                descUbMem = desc;
                descUbMem.channelProtocol = COMM_PROTOCOL_UB_MEM;
                descUbMem.ubMemAttr.pathMode = 2;
            }
        };

        TEST_F(ChannelLoggerTest, IsRoceProtocol_Roce) { EXPECT_TRUE(ChannelLogger::IsRoceProtocol(descRoce)); }

        TEST_F(ChannelLoggerTest, IsRoceProtocol_NonRoce) { EXPECT_FALSE(ChannelLogger::IsRoceProtocol(desc)); }

        TEST_F(ChannelLoggerTest, FormatRoceAttrCompact_Roce)
        {
            std::string result = ChannelLogger::FormatRoceAttrCompact(descRoce);
            EXPECT_NE(result.find("q:4"), std::string::npos);
            EXPECT_NE(result.find("r:3"), std::string::npos);
            EXPECT_NE(result.find("ri:20"), std::string::npos);
        }

        TEST_F(ChannelLoggerTest, FormatRoceAttrCompact_NonRoce)
        {
            std::string result = ChannelLogger::FormatRoceAttrCompact(desc);
            EXPECT_EQ(result, "-");
        }

        TEST_F(ChannelLoggerTest, FormatRoceAttrDetail_Roce)
        {
            std::string result = ChannelLogger::FormatRoceAttrDetail(descRoce);
            EXPECT_NE(result.find("queueNum[4]"), std::string::npos);
            EXPECT_NE(result.find("retryCnt[3]"), std::string::npos);
            EXPECT_NE(result.find("retryInterval[20]"), std::string::npos);
        }

        TEST_F(ChannelLoggerTest, FormatRoceAttrDetail_NonRoce)
        {
            std::string result = ChannelLogger::FormatRoceAttrDetail(desc);
            EXPECT_TRUE(result.empty());
        }

        TEST_F(ChannelLoggerTest, IsAbnormalStatus_Ready)
        {
            EXPECT_FALSE(ChannelLogger::IsAbnormalStatus(static_cast<int32_t>(3)));
        }

        TEST_F(ChannelLoggerTest, IsAbnormalStatus_Failed)
        {
            EXPECT_TRUE(ChannelLogger::IsAbnormalStatus(static_cast<int32_t>(4)));
        }

        TEST_F(ChannelLoggerTest, IsAbnormalStatus_Timeout)
        {
            EXPECT_TRUE(ChannelLogger::IsAbnormalStatus(static_cast<int32_t>(2)));
        }

        TEST_F(ChannelLoggerTest, NeedDetailPrint_Failed)
        {
            EXPECT_TRUE(ChannelLogger::NeedDetailPrint(static_cast<int32_t>(4)));
        }

        TEST_F(ChannelLoggerTest, NeedDetailPrint_Timeout)
        {
            EXPECT_TRUE(ChannelLogger::NeedDetailPrint(static_cast<int32_t>(2)));
        }

        TEST_F(ChannelLoggerTest, NeedDetailPrint_Ready)
        {
            EXPECT_FALSE(ChannelLogger::NeedDetailPrint(static_cast<int32_t>(3)));
        }

        TEST_F(ChannelLoggerTest, FormatEndpointAddresses_Normal)
        {
            std::string localAddr, remoteAddr;
            ChannelLogger::FormatEndpointAddresses(desc, localAddr, remoteAddr);
            EXPECT_FALSE(localAddr.empty());
            EXPECT_FALSE(remoteAddr.empty());
        }

        TEST_F(ChannelLoggerTest, PrintDescInfo_NonRoce)
        {
            ChannelLogger::PrintDescInfo(0, desc);
            SUCCEED();
        }

        TEST_F(ChannelLoggerTest, PrintDescInfo_Roce)
        {
            ChannelLogger::PrintDescInfo(0, descRoce);
            SUCCEED();
        }

        TEST_F(ChannelLoggerTest, IsUbMemProtocol_UbMem) { EXPECT_TRUE(ChannelLogger::IsUbMemProtocol(descUbMem)); }

        TEST_F(ChannelLoggerTest, IsUbMemProtocol_NonUbMem)
        {
            EXPECT_FALSE(ChannelLogger::IsUbMemProtocol(desc));
            EXPECT_FALSE(ChannelLogger::IsUbMemProtocol(descRoce));
        }

        TEST_F(ChannelLoggerTest, PrintUbMemAttributes_UbMem)
        {
            ChannelLogger::PrintUbMemAttributes(0, descUbMem);
            SUCCEED();
        }

        TEST_F(ChannelLoggerTest, PrintUbMemAttributes_NonUbMem)
        {
            // 非UB_MEM协议应直接返回，不打印
            ChannelLogger::PrintUbMemAttributes(0, desc);
            SUCCEED();
        }

        TEST_F(ChannelLoggerTest, PrintDescInfo_UbMem)
        {
            ChannelLogger::PrintDescInfo(0, descUbMem);
            SUCCEED();
        }

        TEST_F(ChannelLoggerTest, PrintDescTableHeader)
        {
            ChannelLogger::PrintDescTableHeader();
            SUCCEED();
        }

        TEST_F(ChannelLoggerTest, PrintDescInfoRow_NonRoce)
        {
            ChannelLogger::PrintDescInfoRow(0, desc);
            SUCCEED();
        }

        TEST_F(ChannelLoggerTest, PrintDescInfoRow_Roce)
        {
            ChannelLogger::PrintDescInfoRow(0, descRoce);
            SUCCEED();
        }

        TEST_F(ChannelLoggerTest, PrintErrorTableHeader)
        {
            ChannelLogger::PrintErrorTableHeader(0);
            SUCCEED();
        }

        TEST_F(ChannelLoggerTest, PrintErrorInfo_TlsEnable)
        {
            ChannelLogger::PrintErrorInfo(
                0, 0, desc, 0x12345678, static_cast<int32_t>(4), 100, Hccl::TlsStatus::ENABLE);
            SUCCEED();
        }

        TEST_F(ChannelLoggerTest, PrintErrorInfo_TlsDisable)
        {
            ChannelLogger::PrintErrorInfo(
                0, 0, desc, 0x12345678, static_cast<int32_t>(3), 50, Hccl::TlsStatus::DISABLE);
            SUCCEED();
        }

        TEST_F(ChannelLoggerTest, PrintErrorInfo_TlsUnknown)
        {
            ChannelLogger::PrintErrorInfo(
                0, 0, desc, 0x12345678, static_cast<int32_t>(2), 200, Hccl::TlsStatus::UNKNOWN);
            SUCCEED();
        }

        TEST_F(ChannelLoggerTest, PrintChannelErrorDetails_AllReady)
        {
            HcclChannelDesc descs[2] = {desc, descRoce};
            ChannelHandle handles[2] = {0x111, 0x222};
            int32_t statusList[2] = {static_cast<int32_t>(3), static_cast<int32_t>(3)};
            Hccl::TlsStatus tlsStatusList[2] = {Hccl::TlsStatus::ENABLE, Hccl::TlsStatus::ENABLE};
            ChannelLogger::PrintChannelErrorDetails(0, 2, descs, handles, statusList, 100, tlsStatusList);
            SUCCEED();
        }

        TEST_F(ChannelLoggerTest, PrintChannelErrorDetails_WithErrors)
        {
            HcclChannelDesc descs[3] = {desc, descRoce, desc};
            ChannelHandle handles[3] = {0x111, 0x222, 0x333};
            int32_t statusList[3] = {static_cast<int32_t>(3), static_cast<int32_t>(4), static_cast<int32_t>(2)};
            Hccl::TlsStatus tlsStatusList[3]
                = {Hccl::TlsStatus::DISABLE, Hccl::TlsStatus::DISABLE, Hccl::TlsStatus::DISABLE};
            ChannelLogger::PrintChannelErrorDetails(0, 3, descs, handles, statusList, 200, tlsStatusList);
            SUCCEED();
        }

        class EndpointLoggerTest : public ::testing::Test {
        protected:
            EndpointDesc endpointDesc{};

            void SetUp() override
            {
                memset(&endpointDesc, 0, sizeof(endpointDesc));
                endpointDesc.commAddr.type = COMM_ADDR_TYPE_IP_V4;
                endpointDesc.commAddr.addr.s_addr = inet_addr("10.0.0.1");
            }
        };

        TEST_F(EndpointLoggerTest, PrintLocation_Device)
        {
            EndpointLoc loc = {};
            loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
            loc.device.devPhyId = 0;
            loc.device.superDevId = 1;
            loc.device.serverIdx = 2;
            loc.device.superPodIdx = 3;
            EndpointLogger::PrintLocation(0, "localEndpoint", loc);
            SUCCEED();
        }

        TEST_F(EndpointLoggerTest, PrintLocation_Host)
        {
            EndpointLoc loc = {};
            loc.locType = ENDPOINT_LOC_TYPE_HOST;
            loc.host.id = 42;
            EndpointLogger::PrintLocation(0, "remoteEndpoint", loc);
            SUCCEED();
        }

        TEST_F(EndpointLoggerTest, PrintLocation_Other)
        {
            EndpointLoc loc = {};
            loc.locType = static_cast<EndpointLocType>(99);
            EndpointLogger::PrintLocation(0, "localEndpoint", loc);
            SUCCEED();
        }

        TEST_F(EndpointLoggerTest, Print_Normal)
        {
            endpointDesc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
            endpointDesc.loc.device.devPhyId = 0;
            endpointDesc.loc.device.superDevId = 1;
            EndpointLogger::Print(0, "localEndpoint", endpointDesc);
            SUCCEED();
        }

    } // namespace
} // namespace logger
} // namespace hcomm

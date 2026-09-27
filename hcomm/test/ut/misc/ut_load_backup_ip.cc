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
#include <mockcpp/mockcpp.hpp>
#include "llt_hccl_stub_sal_pub.h"

#define private public
#define protected public
#include "device_capacity.h"
#include "adapter_hccp.h"
#include "topoinfo_detect.h"
#include "topoinfo_exchange_agent.h"
#include "topoinfo_ranktableParser_pub.h"
#include "topoinfo_ranktableConcise.h"
#include "hccl_communicator.h"
#undef private

using namespace std;
using namespace hccl;

class LoadBackupIpTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "\033[36m--LoadBackupIpTest SetUP--\033[0m" << endl; }
    static void TearDownTestCase() { cout << "\033[36m--LoadBackupIpTest TearDown--\033[0m" << endl; }
    virtual void SetUp()
    {
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        setenv("HCCL_OP_RETRY_ENABLE", "L0:1,L1:1,L2:1", 1);
        DevType deviceType = DevType::DEV_TYPE_910_93;
        MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

        MOCKER(GetExternalInputInterSuperPodRetryEnable).stubs().will(returnValue(true));

        MOCKER(GetExternalInputHcclAicpuUnfold).stubs().will(returnValue(true));
        cout << "A Test SetUP" << endl;
    }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        unsetenv("HCCL_OP_RETRY_ENABLE");
        cout << "A Test TearDown" << endl;
    }
};

#if 1
TEST_F(LoadBackupIpTest, ut_hrtRaGetDeviceAllNicIP)
{
    HcclResult ret = HCCL_SUCCESS;

    DevType deviceType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    s32 deviceLogicID = 0;
    MOCKER(hrtGetDevice).stubs().with(outBoundP(&deviceLogicID)).will(returnValue(HCCL_SUCCESS));

    u32 devicePhyId = 0;
    MOCKER(hrtGetDevicePhyIdByIndex)
        .stubs()
        .with(mockcpp::any(), outBound(devicePhyId))
        .will(returnValue(HCCL_SUCCESS));

    u32 ifAddrNum = 1;
    MOCKER(hrtGetIfNum).stubs().with(mockcpp::any(), outBound(ifAddrNum)).will(returnValue(0));

    u32 ifnumVersion = 3;
    MOCKER(hrtRaGetInterfaceVersion)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&ifnumVersion))
        .will(returnValue(0));

    struct InterfaceInfo ifAddrInfos[1];
    ifAddrInfos[0].ifaddr.ip.addr.s_addr = 0x100007f;
    ifAddrInfos[0].ifaddr.mask.s_addr = 0xffff;
    ifAddrInfos[0].ifname[0] = 'e';
    ifAddrInfos[0].ifname[1] = 't';
    ifAddrInfos[0].ifname[2] = 'h';
    ifAddrInfos[0].ifname[3] = '0';
    ifAddrInfos[0].ifname[4] = '\0';
    ifAddrInfos[0].family = AF_INET;
    MOCKER(hrtGetIfAddress)
        .stubs()
        .with(mockcpp::any(), outBoundP(ifAddrInfos, sizeof(ifAddrInfos)), mockcpp::any())
        .will(returnValue(0));

    vector<vector<HcclIpAddress>> ipAddr;
    ret = hrtRaGetDeviceAllNicIP(ipAddr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}
#endif

#if 1
TEST_F(LoadBackupIpTest, ut_topo_detect_backup_ip)
{
    HcclRootHandle rootHandle;
    shared_ptr<TopoInfoDetect> topoDetectServer = make_shared<TopoInfoDetect>();

    struct in6_addr addr6;
    addr6.s6_addr32[0] = 787324;
    addr6.s6_addr32[1] = 28934;
    addr6.s6_addr32[2] = 98;
    addr6.s6_addr32[3] = 78933899;
    HcclIpAddress localIp(addr6);
    vector<HcclIpAddress> ipAddr;
    ipAddr.emplace_back(localIp);
    MOCKER(hrtRaGetDeviceIP).stubs().with(mockcpp::any(), outBound(ipAddr)).will(returnValue(HCCL_SUCCESS));

    vector<vector<HcclIpAddress>> chipIpAddr;
    chipIpAddr.emplace_back(ipAddr);
    chipIpAddr.emplace_back(ipAddr);
    MOCKER(hrtRaGetDeviceAllNicIP).stubs().with(outBound(chipIpAddr)).will(returnValue(HCCL_SUCCESS));

    MOCKER(GetExternalInputInterSuperPodRetryEnable).stubs().will(returnValue(true));

    MOCKER(HcclNetDevGetTlsStatus).stubs().will(returnValue(HCCL_SUCCESS));

    HcclBasicRankInfo localRankInfo_;
    HcclResult ret = topoDetectServer->GenerateLocalRankInfo(2, INVALID_VALUE_RANKID, localRankInfo_);

    GlobalMockObject::verify();
}
#endif

#if 1
TEST_F(LoadBackupIpTest, ut_topo_detect_backup_ip_fail)
{
    HcclRootHandle rootHandle;
    shared_ptr<TopoInfoDetect> topoDetectServer = make_shared<TopoInfoDetect>();

    struct in6_addr addr6;
    addr6.s6_addr32[0] = 787324;
    addr6.s6_addr32[1] = 28934;
    addr6.s6_addr32[2] = 98;
    addr6.s6_addr32[3] = 78933899;
    HcclIpAddress localIp(addr6);
    vector<HcclIpAddress> ipAddr;
    ipAddr.emplace_back(localIp);
    MOCKER(hrtRaGetDeviceIP).stubs().with(mockcpp::any(), outBound(ipAddr)).will(returnValue(HCCL_SUCCESS));

    vector<vector<HcclIpAddress>> chipIpAddr;
    chipIpAddr.emplace_back(ipAddr);
    MOCKER(hrtRaGetDeviceAllNicIP).stubs().with(outBound(chipIpAddr)).will(returnValue(HCCL_SUCCESS));

    MOCKER(GetExternalInputInterSuperPodRetryEnable).stubs().will(returnValue(true));

    MOCKER(GetExternalInputHcclAicpuUnfold).stubs().will(returnValue(true));

    MOCKER(HcclNetDevGetTlsStatus).stubs().will(returnValue(HCCL_SUCCESS));

    HcclBasicRankInfo localRankInfo_;
    HcclResult ret = topoDetectServer->GenerateLocalRankInfo(2, INVALID_VALUE_RANKID, localRankInfo_);

    GlobalMockObject::verify();
}
#endif

#if 1
TEST_F(LoadBackupIpTest, ut_topo_exchange_verify_backup_ip)
{
    LinkTypeInServer linkType = LinkTypeInServer::SIO_TYPE;
    MOCKER(hrtGetPairDeviceLinkType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(linkType))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER(GetExternalInputInterSuperPodRetryEnable).stubs().will(returnValue(true));

    bool useSuperPodMode = true;
    MOCKER(IsSuperPodMode).stubs().with(outBound(useSuperPodMode)).will(returnValue(HCCL_SUCCESS));

    HcclIpAddress localIp(1694542016);
    HcclNetDevCtx netDevCtx;
    HcclBasicRankInfo localRankInfo;
    localRankInfo.deviceType = DevType::DEV_TYPE_910_93;
    u32 serverPort = 60000;
    string identifier = "test";
    TopoInfoExchangeAgent agent(localIp, serverPort, identifier, netDevCtx, localRankInfo);

    RankTable_t rankTable;
    rankTable.collectiveId = "192.168.0.101-8000-8001";
    vector<RankInfo_t> rankVec(2);

    HcclIpAddress ipAddr1(1694542016);
    HcclIpAddress ipAddr2(1711319232);
    HcclIpAddress ipAddr3(1711319233);
    rankVec[0].rankId = 0;
    rankVec[0].deviceInfo.devicePhyId = 0;
    rankVec[0].deviceInfo.deviceIp.push_back(ipAddr1);
    rankVec[0].deviceInfo.backupDeviceIp.push_back(ipAddr2);
    rankVec[0].serverIdx = 0;
    rankVec[0].serverId = "192.168.0.101";
    rankVec[0].superPodId = "192.168.0.106";

    rankVec[1].rankId = 1;
    rankVec[1].deviceInfo.devicePhyId = 1;
    rankVec[1].deviceInfo.deviceIp.push_back(ipAddr2);
    rankVec[1].deviceInfo.backupDeviceIp.push_back(ipAddr3);
    rankVec[1].serverIdx = 1;
    rankVec[1].serverId = "192.168.1.101";
    rankVec[1].superPodId = "192.168.0.107";

    rankTable.rankList.assign(rankVec.begin(), rankVec.end());
    rankTable.deviceNum = 2;
    rankTable.serverNum = 2;
    rankTable.superPodNum = 2;

    HcclResult ret = agent.VerifyClusterBackupDeviceIP(rankTable);

    GlobalMockObject::verify();
}
#endif

#if 1
TEST_F(LoadBackupIpTest, ut_cluster_info_backup_ip)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"version", "1.2"},
           {"server_count", "2"},
           {"server_list",
            {{
                {"server_id", "101.0.168.192"},
                {"device",
                 {{
                      {"rank_id", "0"},
                      {"device_id", "0"},
                      {"device_ip", "101.0.168.192"},
                      {"backup_device_ip", "101.0.168.193"},
                  },
                  {
                      {"rank_id", "1"},
                      {"device_id", "1"},
                      {"device_ip", "101.0.168.193"},
                      {"backup_device_ip", "101.0.168.194"},
                  }}},
            }}}};

    nlohmann::json deviceList
        = {{
               {"rank_id", "0"},
               {"device_id", "0"},
               {"device_ip", "101.0.168.192"},
               {"backup_device_ip", "101.0.168.193"},
           },
           {
               {"rank_id", "1"},
               {"device_id", "1"},
               {"device_ip", "101.0.168.193"},
               {"backup_device_ip", "101.0.168.194"},
           }};

    LinkTypeInServer linkType = LinkTypeInServer::SIO_TYPE;
    MOCKER(hrtGetPairDeviceLinkType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(linkType))
        .will(returnValue(HCCL_SUCCESS));

    string rankTableM = rank_table.dump();
    string identify = "test";
    TopoinfoRanktableConcise topoRanktable(rankTableM, identify);

    topoRanktable.params_.deviceType = DevType::DEV_TYPE_910_93;

    MOCKER(GetExternalInputInterSuperPodRetryEnable).stubs().will(returnValue(true));

    std::vector<RankInfo_t> rankinfo0(1);
    topoRanktable.GetSingleBackupDeviceIp(deviceList, 0, rankinfo0[0]);

    topoRanktable.VerifyBackupDeviceIpAndPort(rankinfo0, 0);

    std::vector<RankInfo_t> rankinfo1(1);
    topoRanktable.GetSingleBackupDeviceIp(deviceList, 0, rankinfo1[0]);

    topoRanktable.VerifyBackupDeviceIpAndPort(rankinfo1, 0);

    GlobalMockObject::verify();
}
#endif

#if 1
TEST_F(LoadBackupIpTest, ut_cluster_info_backup_ip_fail_e_para)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"version", "1.2"},
           {"server_count", "2"},
           {"server_list",
            {{
                {"server_id", "101.0.168.192"},
                {"device",
                 {{
                      {"rank_id", "0"},
                      {"device_id", "0"},
                      {"device_ip", "101.0.168.192"},
                      {"backup_device_ip", "101.0.168.193"},
                  },
                  {
                      {"rank_id", "1"},
                      {"device_id", "1"},
                      {"device_ip", "101.0.168.193"},
                      {"backup_device_ip", "101.0.168.194"},
                  }}},
            }}}};

    nlohmann::json deviceList
        = {{
               {"rank_id", "0"},
               {"device_id", "0"},
               {"device_ip", "101.0.168.192"},
               {"backup_device_ip", "101.0.168.193"},
           },
           {
               {"rank_id", "1"},
               {"device_id", "1"},
               {"device_ip", "101.0.168.193"},
               {"backup_device_ip", "101.0.168.194"},
           }};

    LinkTypeInServer linkType = LinkTypeInServer::SIO_TYPE;
    MOCKER(hrtGetPairDeviceLinkType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(linkType))
        .will(returnValue(HCCL_SUCCESS));

    string rankTableM = rank_table.dump();
    string identify = "test";
    TopoinfoRanktableConcise topoRanktable(rankTableM, identify);

    topoRanktable.params_.deviceType = DevType::DEV_TYPE_910_93;

    MOCKER(GetExternalInputInterSuperPodRetryEnable).stubs().will(returnValue(true));

    RankInfo_t rankinfo0;
    auto ret = topoRanktable.GetSingleBackupDeviceIp(deviceList, deviceList.size(), rankinfo0);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = topoRanktable.GetSingleBackupDeviceIp(nlohmann::json::object(), 0, rankinfo0);
    EXPECT_EQ(ret, HCCL_E_PARA);
}
#endif

static void TestConstructParam_SurperPod(HcclCommParams& params, RankTable_t& rankTable)
{
    string commId = "comm ";
    memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1);
    params.rank = 0;
    params.totalRanks = 2;
    params.isHeterogComm = false;
    params.logicDevId = 0;
    params.commWorkMode = WorkMode::HCCL_MODE_NORMAL;
    params.deviceType = DevType::DEV_TYPE_910_93;

    rankTable.collectiveId = "192.168.0.101-8000-8001";
    vector<RankInfo_t> rankVec(2);
    rankVec[0].rankId = 0;
    rankVec[0].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr1(1694542016);
    rankVec[0].deviceInfo.deviceIp.push_back(ipAddr1); // 101.0.168.192
    rankVec[0].serverIdx = 0;
    rankVec[0].serverId = "192.168.0.101";
    rankVec[0].superPodId = "192.168.0.103";
    rankVec[1].rankId = 1;
    rankVec[1].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr2(1711319232);
    rankVec[1].deviceInfo.deviceIp.push_back(ipAddr2); // 101.0.168.192
    rankVec[1].serverIdx = 1;
    rankVec[1].serverId = "192.168.0.102";
    rankVec[1].superPodId = "192.168.0.104";
    rankTable.rankList.assign(rankVec.begin(), rankVec.end());
    rankTable.deviceNum = 2;
    rankTable.serverNum = 2;
}

#if 1
TEST_F(LoadBackupIpTest, ut_GetSingleDevicePort)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"version", "1.2"},
           {"server_count", "2"},
           {"server_list",
            {{
                {"server_id", "101.0.168.192"},
                {"device",
                 {{
                      {"rank_id", "0"},
                      {"device_id", "0"},
                      {"device_ip", "101.0.168.192"},
                      {"backup_device_ip", "101.0.168.193"},
                  },
                  {
                      {"rank_id", "1"},
                      {"device_id", "1"},
                      {"device_ip", "101.0.168.193"},
                      {"backup_device_ip", "101.0.168.194"},
                  }}},
            }}}};

    nlohmann::json deviceList
        = {{
               {"rank_id", "0"},
               {"device_id", "0"},
               {"device_ip", "101.0.168.192"},
               {"backup_device_ip", "101.0.168.193"},
               {"device_port", "16666"},
           },
           {
               {"rank_id", "1"},
               {"device_id", "1"},
               {"device_ip", "101.0.168.193"},
               {"backup_device_ip", "101.0.168.194"},
               {"device_port", "16666"},
           }};
    string rankTableM = rank_table.dump();
    TopoinfoRanktableConcise topoRanktable(rankTableM, "test");
    RankInfo_t rankinfo0;

    topoRanktable.GetSingleDevicePort(deviceList, 0, rankinfo0);
    GlobalMockObject::verify();
}
#endif

#if 1
TEST_F(LoadBackupIpTest, ut_GetSingleDevicePort_Vnic)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"version", "1.2"},
           {"server_count", "2"},
           {"server_list",
            {{
                {"server_id", "101.0.168.192"},
                {"device",
                 {{
                      {"rank_id", "0"},
                      {"device_id", "0"},
                      {"device_ip", "101.0.168.192"},
                      {"backup_device_ip", "101.0.168.193"},
                  },
                  {
                      {"rank_id", "1"},
                      {"device_id", "1"},
                      {"device_ip", "101.0.168.193"},
                      {"backup_device_ip", "101.0.168.194"},
                  }}},
            }}}};

    nlohmann::json deviceList
        = {{
               {"rank_id", "0"},
               {"device_id", "0"},
               {"device_ip", "101.0.168.192"},
               {"backup_device_ip", "101.0.168.193"},
               {"device_port", "16666"},
               {"device_vnic_port", "16667"},
           },
           {
               {"rank_id", "1"},
               {"device_id", "1"},
               {"device_ip", "101.0.168.193"},
               {"backup_device_ip", "101.0.168.194"},
               {"device_port", "16666"},
               {"device_vnic_port", "16667"},
           }};
    string rankTableM = rank_table.dump();
    TopoinfoRanktableConcise topoRanktable(rankTableM, "test");
    RankInfo_t rankinfo0;

    topoRanktable.GetSingleDevicePort(deviceList, 0, rankinfo0);
    GlobalMockObject::verify();
}
#endif

TEST_F(LoadBackupIpTest, ut_topo_exchange_verify_superPodId)
{
    HcclIpAddress localIp(1694542016);
    HcclNetDevCtx netDevCtx;
    HcclBasicRankInfo localRankInfo;
    localRankInfo.deviceType = DevType::DEV_TYPE_910_93;
    u32 serverPort = 60000;
    string identifier = "test";
    TopoInfoExchangeAgent agent(localIp, serverPort, identifier, netDevCtx, localRankInfo);

    RankTable_t rankTable;
    rankTable.collectiveId = "192.168.0.101-8000-8001";
    vector<RankInfo_t> rankVec(3);

    HcclIpAddress ipAddr1(1694542016);
    HcclIpAddress ipAddr2(1711319232);
    HcclIpAddress ipAddr3(1711319233);
    rankVec[0].rankId = 0;
    rankVec[0].deviceInfo.devicePhyId = 0;
    rankVec[0].deviceInfo.deviceIp.push_back(ipAddr1);
    rankVec[0].deviceInfo.backupDeviceIp.push_back(ipAddr2);
    rankVec[0].serverIdx = 0;
    rankVec[0].serverId = "192.168.0.101";
    rankVec[0].superPodId = "192.168.0.106";

    rankVec[1].rankId = 1;
    rankVec[1].deviceInfo.devicePhyId = 1;
    rankVec[1].deviceInfo.deviceIp.push_back(ipAddr2);
    rankVec[1].deviceInfo.backupDeviceIp.push_back(ipAddr3);
    rankVec[1].serverIdx = 1;
    rankVec[1].serverId = "192.168.1.101";
    rankVec[1].superPodId = "192.168.0.107";

    rankVec[2].rankId = 2;
    rankVec[2].deviceInfo.devicePhyId = 2;
    rankVec[2].deviceInfo.deviceIp.push_back(ipAddr3);
    rankVec[2].deviceInfo.backupDeviceIp.push_back(ipAddr1);
    rankVec[2].serverIdx = 0;
    rankVec[2].serverId = "192.168.0.101";
    rankVec[2].superPodId = "192.168.0.106";

    rankTable.rankList.assign(rankVec.begin(), rankVec.end());
    rankTable.deviceNum = 3;
    rankTable.serverNum = 3;
    rankTable.superPodNum = 3;

    HcclResult ret = agent.SetSuperPodIdx(rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(LoadBackupIpTest, ut_GetSingleServer_failed)
{
    nlohmann::json rank_table;
    nlohmann::json serverListObj = nlohmann::json::array(
        {{{"instance_id", 0},
          {"device_name", "device0"},
          {"server_id", "12345678910123456789101234567891012345678910123456789101234567891"}}});
    string rankTableM = rank_table.dump();
    TopoinfoRanktableConcise topoRanktable(rankTableM, "test");
    RankTable_t rankinfo0;
    HcclResult ret = topoRanktable.GetSingleServer(serverListObj, 0, rankinfo0);
    EXPECT_EQ(ret, HCCL_E_PARA);
    GlobalMockObject::verify();
}

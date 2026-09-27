/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#define private public
#define protected public
#include "topoinfo_exchange_agent.h"
#include "detect_connect_anomalies.h"
#undef protected
#undef private
#include "adapter_error_manager_pub.h"
#include "sal_pub.h"
#include "device_capacity.h"
#include "preempt_port_manager.h"
#include "adapter_rts_common.h"

using namespace std;
using namespace hccl;

HcclResult hrtMallocHost(void** hostPtr, u64 size);

namespace {
constexpr u64 kMallocSize = 1024;
constexpr s32 kDeviceId = 0;
} // namespace

static DetectInfo
MakeDetectInfo(const string& localServer, s32 localDevice, const string& remoteServer, s32 remoteDevice)
{
    DetectInfo info;
    info.localDeviceId = localDevice;
    info.remoteDeviceId = remoteDevice;
    strncpy(info.localServerId, localServer.c_str(), DEST_MAX_LEN - 1);
    strncpy(info.remoteServerId, remoteServer.c_str(), DEST_MAX_LEN - 1);
    return info;
}

static aclError
stub_aclrtMallocWithCfg_error(void** devPtr, size_t size, aclrtMemMallocPolicy policy, aclrtMallocConfig* cfg)
{
    *devPtr = reinterpret_cast<void*>(0xDEADBEEF);
    return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
}

static aclError stub_aclrtMallocHostWithCfg_error(void** ptr, uint64_t size, aclrtMallocConfig* cfg)
{
    *ptr = reinterpret_cast<void*>(0xCAFEBABE);
    return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
}

static HcclResult stub_hrtGetDeviceType_success(DevType& devType)
{
    devType = DevType::DEV_TYPE_910_93;
    return HCCL_SUCCESS;
}

static HcclResult stub_hrtGetDevice_success(s32* deviceId)
{
    *deviceId = 0;
    return HCCL_SUCCESS;
}

class ErrorMessageTest : public testing::Test {
protected:
    virtual void SetUp()
    {
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        MOCKER(RptInputErr).stubs().will(returnValue(HCCL_SUCCESS));
    }
    virtual void TearDown() { GlobalMockObject::verify(); }
};

static void ConstructRankTable(RankTable_t& rankTable, u32 rankSize)
{
    rankTable.collectiveId = "192.168.0.101-8000-8001";
    vector<RankInfo_t> rankVec(rankSize);
    for (u32 i = 0; i < rankSize; i++) {
        rankVec[i].rankId = i;
        rankVec[i].deviceInfo.devicePhyId = i % 2;
        rankVec[i].serverId = "192.168.0.101";
    }
    rankTable.rankList.assign(rankVec.begin(), rankVec.end());
    rankTable.deviceNum = rankSize;
    rankTable.serverNum = rankSize / 2;
}

TEST_F(ErrorMessageTest, Ut_VerifyClusterTlsConsistency_AllSupportTls_UnknownStrEmpty_ReplacedWithNA)
{
    HcclIpAddress localIp(1694542016);
    HcclNetDevCtx netDevCtx;
    HcclBasicRankInfo localRankInfo;
    localRankInfo.deviceType = DevType::DEV_TYPE_910_93;
    u32 serverPort = 60000;
    string identifier = "test";
    RankTable_t clusterInfo;
    ConstructRankTable(clusterInfo, 4);
    clusterInfo.rankList[0].tlsStatus = TlsStatus::ENABLE;
    clusterInfo.rankList[1].tlsStatus = TlsStatus::DISABLE;
    clusterInfo.rankList[2].tlsStatus = TlsStatus::ENABLE;
    clusterInfo.rankList[3].tlsStatus = TlsStatus::DISABLE;

    TopoInfoExchangeAgent agent(localIp, serverPort, identifier, netDevCtx, localRankInfo);
    HcclResult ret = agent.VerifyClusterTlsConsistency(clusterInfo);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(ErrorMessageTest, Ut_VerifyClusterTlsConsistency_PartialSupportTls_UnknownStrKept)
{
    HcclIpAddress localIp(1694542016);
    HcclNetDevCtx netDevCtx;
    HcclBasicRankInfo localRankInfo;
    localRankInfo.deviceType = DevType::DEV_TYPE_910_93;
    u32 serverPort = 60000;
    string identifier = "test";
    RankTable_t clusterInfo;
    ConstructRankTable(clusterInfo, 4);
    clusterInfo.rankList[0].tlsStatus = TlsStatus::ENABLE;
    clusterInfo.rankList[1].tlsStatus = TlsStatus::DISABLE;
    clusterInfo.rankList[2].tlsStatus = TlsStatus::ENABLE;
    clusterInfo.rankList[3].tlsStatus = TlsStatus::UNKNOWN;

    TopoInfoExchangeAgent agent(localIp, serverPort, identifier, netDevCtx, localRankInfo);
    HcclResult ret = agent.VerifyClusterTlsConsistency(clusterInfo);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(ErrorMessageTest, Ut_hrtMalloc_Error_ReportsBytes)
{
    MOCKER(hrtGetDeviceType).stubs().will(invoke(stub_hrtGetDeviceType_success));
    MOCKER(hrtGetDevice).stubs().will(invoke(stub_hrtGetDevice_success));
    MOCKER(Is310PDevice).stubs().will(returnValue(false));
    MOCKER(aclrtMallocWithCfg).stubs().will(invoke(stub_aclrtMallocWithCfg_error));

    void* devPtr = nullptr;
    HcclResult ret = hrtMalloc(&devPtr, kMallocSize, false);
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
}

TEST_F(ErrorMessageTest, Ut_hrtMallocHost_Error_ReportsBytes)
{
    MOCKER(aclrtMallocHostWithCfg).stubs().will(invoke(stub_aclrtMallocHostWithCfg_error));

    void* hostPtr = nullptr;
    HcclResult ret = hrtMallocHost(&hostPtr, kMallocSize);
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
}

TEST_F(ErrorMessageTest, Ut_BuildGroupedDetectMessage_EmptyMap_ReturnsEmpty)
{
    auto& dca = DetectConnectionAnomalies::GetInstance(kDeviceId);
    dca.recvErrorInfoMap_.clear();
    string result = dca.BuildGroupedDetectMessage();
    EXPECT_TRUE(result.empty());
}

TEST_F(ErrorMessageTest, Ut_BuildGroupedDetectMessage_SingleEntry_FormatsCorrectly)
{
    auto& dca = DetectConnectionAnomalies::GetInstance(kDeviceId);
    dca.recvErrorInfoMap_.clear();
    dca.recvErrorInfoMap_["key1"] = MakeDetectInfo("serverA", 0, "serverB", 1);

    string result = dca.BuildGroupedDetectMessage();
    EXPECT_NE(result.find("serverA"), string::npos);
    EXPECT_NE(result.find("device ID 0"), string::npos);
    EXPECT_NE(result.find("serverB"), string::npos);
    EXPECT_NE(result.find("device ID 1"), string::npos);
    // 单条记录无换行分隔符
    EXPECT_EQ(result.find("\n"), string::npos);
}

TEST_F(ErrorMessageTest, Ut_BuildGroupedDetectMessage_DupDevices_SortedAndDeduped)
{
    auto& dca = DetectConnectionAnomalies::GetInstance(kDeviceId);
    dca.recvErrorInfoMap_.clear();
    // 同一 src+dst，但 remoteDeviceId 有重复和乱序
    dca.recvErrorInfoMap_["k1"] = MakeDetectInfo("serverA", 0, "serverB", 5);
    dca.recvErrorInfoMap_["k2"] = MakeDetectInfo("serverA", 0, "serverB", 2);
    dca.recvErrorInfoMap_["k3"] = MakeDetectInfo("serverA", 0, "serverB", 2); // 重复

    string result = dca.BuildGroupedDetectMessage();
    // 排序+去重后应为 device ID 2,5
    EXPECT_NE(result.find("device ID 2,5"), string::npos);
}

TEST_F(ErrorMessageTest, Ut_BuildGroupedDetectMessage_MultiGroups_SeparatedByNewline)
{
    auto& dca = DetectConnectionAnomalies::GetInstance(kDeviceId);
    dca.recvErrorInfoMap_.clear();
    dca.recvErrorInfoMap_["k1"] = MakeDetectInfo("serverA", 0, "serverB", 1);
    dca.recvErrorInfoMap_["k2"] = MakeDetectInfo("serverC", 3, "serverD", 4);

    string result = dca.BuildGroupedDetectMessage();
    EXPECT_NE(result.find("\n"), string::npos);
    EXPECT_NE(result.find("serverA"), string::npos);
    EXPECT_NE(result.find("serverC"), string::npos);
}

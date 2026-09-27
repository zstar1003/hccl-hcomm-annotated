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

#define private public
#define protected public
#include "opretry_base.h"
#include "opretry_agent.h"
#include "opretry_server.h"
#include "socket.h"
#include "opretry_manager.h"
#include "opretry_link_manage.h"
#include "comm.h"
#include "hdc_pub.h"
#include "framework/aicpu_hccl_process.h"
#include "hccl_communicator.h"
#include "local_ipc_notify.h"
#include "notify_pool_impl.h"
#include "transport_base_pub.h"
#include "adapter_pub.h"
#include "hccl_network_pub.h"
#undef private

using namespace std;
using namespace hccl;
namespace SwitchNicFsmTest {

constexpr u32 TEST_RANK_NUM = 2;

class SwitchNicFsmTestCase : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "\033[36m--SwitchNicFsmTestCase SetUP--\033[0m" << endl; }
    static void TearDownTestCase() { cout << "\033[36m--SwitchNicFsmTestCase TearDown--\033[0m" << endl; }
    virtual void SetUp()
    {
        cout << "A Test SetUP" << endl;
        OpRetryAgentParam agentParam;
        HcclIpAddress localIp = HcclIpAddress("192.168.100.110");
        agentParam.h2dPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl)));
        agentParam.d2hPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_D2H, sizeof(KfcExecStatus)));
        HcclOpStreamRes myMap;
        std::string key = "example_key";
        std::vector<Stream> slaves;
        for (int i = 0; i < TEST_RANK_NUM; i++) {
            slaves.push_back(Stream(StreamType::STREAM_TYPE_ONLINE));
        }
        myMap[key] = slaves;
        agentParam.opStreamPtr = std::make_shared<HcclOpStreamRes>(myMap);
        s32 deviceLogicId = 0;
        HcclIpAddress deviceIP = HcclIpAddress("10.21.78.208");
        agentParam.agentInfo = {0, 0, localIp, deviceIP};
        agentParam.group = "test_group";
        agentParam.agentConnection = std::make_shared<HcclSocket>(
            "SwitchNicFsmTestCase_Agent", nullptr, localIp, 0, HcclSocketRole::SOCKET_ROLE_CLIENT);
        agentParam.isEnableBackupLink = false;
        std::shared_ptr<OpRetryAgentRunning> retryBase = std::make_shared<OpRetryAgentRunning>();

        agentRetryCtx = std::make_shared<RetryContext>(agentParam, retryBase);
        std::shared_ptr<HcclSocket> serverSocket = std::make_shared<HcclSocket>(
            "SwitchNicFsmTestCase_Server", nullptr, localIp, 0, HcclSocketRole::SOCKET_ROLE_SERVER);
        std::map<u32, std::shared_ptr<HcclSocket>> serverSockets;
        for (int i = 0; i < TEST_RANK_NUM; i++) {
            serverSockets[i] = serverSocket;
        }
        serverRetryCtx = std::make_shared<RetryContext>(serverSockets, retryBase, agentParam.agentInfo);
        for (int i = 0; i < TEST_RANK_NUM; i++) {
            HcclAgentRetryInfo info;
            serverRetryCtx->serverSockets_[i] = info;
        }
    }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        cout << "A Test TearDown" << endl;
    }

    std::shared_ptr<RetryContext> agentRetryCtx;
    std::shared_ptr<RetryContext> serverRetryCtx;
};

HcclResult stub_GetOpExecInfo_kPlanSwitch(std::shared_ptr<HDCommunicate> hdcPtr, KfcExecStatus& opInfo)
{
    opInfo.execStatus.kfcStatus = KfcStatus::kPlanSwitch;
    return HCCL_SUCCESS;
}

TEST_F(SwitchNicFsmTestCase, ut_Agent_SwitchNicSuc)
{
    std::shared_ptr<OpRetryAgentRunning> opRetryAgentRunning = std::make_shared<OpRetryAgentRunning>();
    KfcExecStatus opInfo;
    opInfo.execStatus.kfcStatus = KfcStatus::kPlanSwitch;
    MOCKER_CPP(&OpRetryBase::GetOpExecInfo)
        .stubs()
        .with(mockcpp::any(), outBound(opInfo))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&OpRetryBase::SetOpExecCmd).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(HcclNetDevGetPortStatus).stubs().will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::IssueResponse).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&OpRetryBase::Send).stubs().will(returnValue(HCCL_SUCCESS));
    bool needCheckDefaultNic = true;
    bool needCheckBackupNic = true;
    MOCKER_CPP(&OpRetryBase::GetSwitchRanks)
        .stubs()
        .with(mockcpp::any(), outBound(needCheckDefaultNic), outBound(needCheckBackupNic))
        .will(returnValue(HCCL_SUCCESS));
    RetryCommand command = RETRY_CMD_NOTIFY_SWITCH_SUC;
    MOCKER_CPP(&OpRetryBase::WaitCommand)
        .stubs()
        .with(mockcpp::any(), outBound(command))
        .will(returnValue(HCCL_SUCCESS));
    agentRetryCtx->localRetryInfo_.opInfo.execStatus.kfcStatus = KfcStatus::kSwitchError;
    HcclResult ret = opRetryAgentRunning->ProcessEvent(agentRetryCtx.get());
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::shared_ptr<SwitchNicAgentSendSwitchInfo> agentSendInfo = std::make_shared<SwitchNicAgentSendSwitchInfo>();
    ret = agentSendInfo->ProcessEvent(agentRetryCtx.get());
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

HcclResult
WaitActiveSwitchInfo_Stub(OpRetryBase* opRetryBase, std::shared_ptr<HcclSocket> socket, ActiveSwitchInfo& switchInfo)
{
    switchInfo.switchRankNum = 2;
    switchInfo.remoteRankNum = 1;
    switchInfo.backupPortStatus = true;
    switchInfo.defaultPortStatus = true;
    switchInfo.refreshTransportFin = true;
    switchInfo.localPortsCheckRet = true;

    // rankNum = 2 模拟两卡，都使用备网卡
    switchInfo.switchRankList[0] = 0;
    switchInfo.switchRankList[1] = 1;
    switchInfo.switchUseBackup[0] = true;
    switchInfo.switchUseBackup[1] = true;
    switchInfo.remoteRankNicStatus[1] = CONNECT_REMOTE_BACKUP;

    return HCCL_SUCCESS;
}

TEST_F(SwitchNicFsmTestCase, ut_Server_SwitchNicSuc)
{
    std::shared_ptr<OpRetryServerRunning> retryServerRunning = std::make_shared<OpRetryServerRunning>();

    RetryInfo retryInfo;
    retryInfo.retryState = RETRY_STATE_SEND_SWITCH_INFO;
    MOCKER_CPP(&OpRetryBase::WaitResponse)
        .stubs()
        .with(mockcpp::any(), outBound(retryInfo))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&OpRetryBase::WaitActiveSwitchInfo).stubs().with(mockcpp::any()).will(invoke(WaitActiveSwitchInfo_Stub));
    MOCKER_CPP(&OpRetryBase::IssueCommand).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = retryServerRunning->ProcessEvent(serverRetryCtx.get());
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::shared_ptr<SwitchNicServerCheckAllSwitchRanks> serverCheck
        = std::make_shared<SwitchNicServerCheckAllSwitchRanks>();
    ret = serverCheck->ProcessEvent(serverRetryCtx.get());
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(SwitchNicFsmTestCase, ut_agent_WaitCmd)
{
    MOCKER_CPP(&OpRetryBase::SetOpExecCmd).stubs().will(returnValue(HCCL_SUCCESS));
    agentRetryCtx->switchInfo_.switchRankNum = 1;
    agentRetryCtx->switchInfo_.remoteRankNum = TEST_RANK_NUM;
    agentRetryCtx->switchInfo_.switchUseBackup[0] = true;
    agentRetryCtx->switchInfo_.remoteRankNicStatus[0] = CONNECT_REMOTE_DEFAULT;
    agentRetryCtx->switchInfo_.remoteRankNicStatus[TEST_RANK_NUM - 1] = CONNECT_REMOTE_BACKUP;
    std::shared_ptr<SwitchNicAgentWaitCmd> agentWaitCmd = std::make_shared<SwitchNicAgentWaitCmd>();
    {
        RetryCommand command = RETRY_CMD_NOTIFY_SWITCH_SUC;
        MOCKER_CPP(&OpRetryBase::WaitCommand)
            .stubs()
            .with(mockcpp::any(), outBound(command))
            .will(returnValue(HCCL_SUCCESS));
        HcclResult ret = agentWaitCmd->ProcessEvent(agentRetryCtx.get());
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }

    {
        RetryCommand command = RETRY_CMD_NOTIFY_SWITCH_FAIL;
        MOCKER_CPP(&OpRetryBase::WaitCommand)
            .stubs()
            .with(mockcpp::any(), outBound(command))
            .will(returnValue(HCCL_SUCCESS));
        HcclResult ret = agentWaitCmd->ProcessEvent(agentRetryCtx.get());
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }
}

ActiveSwitchInfo& InitSwitchInfoMap(std::map<u32, ActiveSwitchInfo>& switchInfoMap)
{
    for (int i = 0; i < TEST_RANK_NUM; i++) {
        ActiveSwitchInfo newInfo;
        newInfo.switchRankNum = TEST_RANK_NUM;
        newInfo.remoteRankNum = TEST_RANK_NUM;
        newInfo.switchRankList[TEST_RANK_NUM - 1] = 1;
        newInfo.switchUseBackup[TEST_RANK_NUM - 1] = 1;
        newInfo.localPortsCheckRet = true;
        newInfo.refreshTransportFin = true;
        switchInfoMap[i] = newInfo;
    }
    return switchInfoMap[TEST_RANK_NUM - 1];
}

TEST_F(SwitchNicFsmTestCase, ut_server_Check_0)
{
    MOCKER_CPP(&OpRetryBase::IssueCommand).stubs().will(returnValue(HCCL_SUCCESS));
    std::shared_ptr<SwitchNicServerCheckAllSwitchRanks> serverCheck
        = std::make_shared<SwitchNicServerCheckAllSwitchRanks>();
    {
        auto& info = InitSwitchInfoMap(serverRetryCtx->switchInfoMap_);
        info.switchRankNum = TEST_RANK_NUM + 1;
        // switchRankList数量不一致
        HcclResult ret = serverCheck->ProcessEvent(serverRetryCtx.get());
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }
    // switchRankList内容不一致
    {
        auto& info = InitSwitchInfoMap(serverRetryCtx->switchInfoMap_);
        info.switchRankList[TEST_RANK_NUM - 1] = 2;
        HcclResult ret = serverCheck->ProcessEvent(serverRetryCtx.get());
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }
    // switchRankList内含重复元素
    {
        auto& info = InitSwitchInfoMap(serverRetryCtx->switchInfoMap_);
        info.switchRankList[0] = 1;
        HcclResult ret = serverCheck->ProcessEvent(serverRetryCtx.get());
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }
    // remoteRankNum为0
    {
        auto& info = InitSwitchInfoMap(serverRetryCtx->switchInfoMap_);
        info.remoteRankNum = 0;
        HcclResult ret = serverCheck->ProcessEvent(serverRetryCtx.get());
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }

    // switchUseBackup不一致
    {
        auto& info = InitSwitchInfoMap(serverRetryCtx->switchInfoMap_);
        info.switchUseBackup[0] = 1;
        HcclResult ret = serverCheck->ProcessEvent(serverRetryCtx.get());
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }

    // 本端默认网口down，但远端使用默认网口
    {
        auto& info = InitSwitchInfoMap(serverRetryCtx->switchInfoMap_);
        info.remoteRankNicStatus[0] = CONNECT_REMOTE_DEFAULT;
        info.defaultPortStatus = false;
        HcclResult ret = serverCheck->ProcessEvent(serverRetryCtx.get());
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }

    // 本端备用网口down，但远端使用备网口
    {
        auto& info = InitSwitchInfoMap(serverRetryCtx->switchInfoMap_);
        info.remoteRankNicStatus[0] = CONNECT_REMOTE_BACKUP;
        info.backupPortStatus = false;
        HcclResult ret = serverCheck->ProcessEvent(serverRetryCtx.get());
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }
}

TEST_F(SwitchNicFsmTestCase, ut_server_CollectSingle)
{
    RetryInfo retryInfo;
    retryInfo.retryState = RETRY_STATE_SEND_SWITCH_INFO;
    MOCKER_CPP(&OpRetryBase::WaitResponse)
        .stubs()
        .with(mockcpp::any(), outBound(retryInfo))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&OpRetryBase::IssueCommand).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&OpRetryBase::Recv)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_E_AGAIN))
        .then(returnValue(HCCL_SUCCESS));
    std::shared_ptr<SwitchNicServerCheckAllSwitchRanks> serverCheck
        = std::make_shared<SwitchNicServerCheckAllSwitchRanks>();
    HcclResult ret = serverCheck->ProcessEvent(serverRetryCtx.get());
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

} // namespace SwitchNicFsmTest

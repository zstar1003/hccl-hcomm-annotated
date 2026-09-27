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
#include "dlra_function.h"

#define private public
#define protected public
#include "op_base.h"
#include "hccl_communicator.h"
#include "aicpu_communicator.h"
#include "framework/aicpu_hccl_process.h"
#include "executor_tracer.h"
#include "transport_ibverbs_pub.h"
#undef private
#include "dispatcher_pub.h"

using namespace std;
using namespace hccl;

class SwitchNicTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        DlRaFunction::GetInstance().DlRaFunctionInit();
        cout << "\033[36m--SwitchNicTest SetUP--\033[0m" << endl;
    }
    static void TearDownTestCase() { cout << "\033[36m--SwitchNicTest TearDown--\033[0m" << endl; }
    virtual void SetUp() { cout << "A Test SetUP" << endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        cout << "A Test TearDown" << endl;
    }
};

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

TEST_F(SwitchNicTest, ut_hcclSwitchNic_test)
{
    hcclComm comm;
    comm.communicator_ = make_unique<HcclCommunicator>();
    HcclComm commoPtr = &comm;
    MOCKER_CPP(&HcclCommunicator::SwitchNic, HcclResult(HcclCommunicator::*)(uint32_t, uint32_t*, bool*))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));

    uint32_t nRanks = 1;
    uint32_t ranks[1] = {0};
    bool useBackup[1] = {true};
    auto ret = HcclCommWorkingDevNicSet(commoPtr, ranks, useBackup, nRanks);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(SwitchNicTest, ut_HcclCommunicator_SwitchNicTimeout)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam_SurperPod(params, rankTable);
    unique_ptr<HcclCommunicator> communicator(new (nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().will(returnValue(HCCL_SUCCESS));

    communicator->Init(params, rankTable);

    MOCKER_CPP(&HcclCommunicator::IsEnableBackupLink).stubs().will(returnValue(true));

    MOCKER(GetExternalInputHcclLinkTimeOut).stubs().will(returnValue(0));

    MachinePara machinePara;
    std::chrono::milliseconds timeout;
    std::shared_ptr<Transport> link(
        new Transport(new (std::nothrow) TransportIbverbs(nullptr, nullptr, machinePara, timeout)));

    MOCKER_CPP(&Transport::Resume).stubs().will(returnValue(HCCL_SUCCESS));

    AlgResourceResponse resourceResponse;
    resourceResponse.opTransportResponse.resize(COMM_LEVEL_RESERVED);
    resourceResponse.opTransportResponse[COMM_LEVEL0].resize(2);
    resourceResponse.opTransportResponse[COMM_LEVEL0][0].transportRequests.resize(2);
    resourceResponse.opTransportResponse[COMM_LEVEL0][1].transportRequests.resize(2);
    resourceResponse.opTransportResponse[COMM_LEVEL0][0].transportRequests[0].isValid = true;
    resourceResponse.opTransportResponse[COMM_LEVEL0][0].transportRequests[0].isUsedRdma = true;
    resourceResponse.opTransportResponse[COMM_LEVEL0][0].transportRequests[0].remoteUserRank = 1;
    resourceResponse.opTransportResponse[COMM_LEVEL0][1].transportRequests[0].isValid = true;
    resourceResponse.opTransportResponse[COMM_LEVEL0][1].transportRequests[0].isUsedRdma = true;
    resourceResponse.opTransportResponse[COMM_LEVEL0][1].transportRequests[0].remoteUserRank = 1;
    resourceResponse.opTransportResponse[COMM_LEVEL0][0].status.resize(2);
    resourceResponse.opTransportResponse[COMM_LEVEL0][0].status[0] = TransportStatus::STOP;
    resourceResponse.opTransportResponse[COMM_LEVEL0][0].status[1] = TransportStatus::READY;
    resourceResponse.opTransportResponse[COMM_LEVEL0][0].links.emplace_back(link);
    resourceResponse.opTransportResponse[COMM_LEVEL0][0].links.emplace_back(link);
    resourceResponse.opTransportResponse[COMM_LEVEL0][1].status.resize(2);
    resourceResponse.opTransportResponse[COMM_LEVEL0][1].status[0] = TransportStatus::STOP;
    resourceResponse.opTransportResponse[COMM_LEVEL0][1].status[1] = TransportStatus::READY;

    communicator->resMap_.emplace("test1", resourceResponse);

    uint32_t nRanks = 1;
    uint32_t ranks[1] = {0};
    bool useBackup[1] = {true};

    ret = communicator->SwitchNic(nRanks, ranks, useBackup);
    EXPECT_EQ(ret, HCCL_E_TIMEOUT);

    uint32_t nRanks2 = 2;
    uint32_t ranks2[2] = {0, 1};
    bool useBackup2[2] = {true, false};

    ret = communicator->SwitchNic(nRanks2, ranks2, useBackup2);
    EXPECT_EQ(ret, HCCL_E_TIMEOUT);
    GlobalMockObject::verify();
}

TEST_F(SwitchNicTest, ut_HcclCommunicator_GetSwitchRanks)
{
    HcclResult ret = HCCL_SUCCESS;
    unique_ptr<HcclCommunicator> communicator(new (nothrow) HcclCommunicator());
    communicator->userRankSize_ = 1;

    u32 switchRankNum{0};
    u32 switchRankList[AICPU_MAX_RANK_NUM]{};
    bool switchUseBackup[AICPU_MAX_RANK_NUM] = {};
    u32 nicStatusNum{0};
    u8 remoteRankNicStatus[AICPU_MAX_RANK_NUM]{};
    bool needCheckDefaultNic;
    bool needCheckBackupNic;

    ret = communicator->GetSwitchRanks(
        switchRankList, switchUseBackup, switchRankNum, remoteRankNicStatus, nicStatusNum, needCheckDefaultNic,
        needCheckBackupNic);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

HcclResult GetChangeLinkInfo(AicpuHdc* This, shared_ptr<HDCommunicate> h2dTransfer, ChangeLinkInfo& changeLinkInfo)
{
    changeLinkInfo.remoteRankNum = 1;
    changeLinkInfo.remoteRankList[0] = 1;
    changeLinkInfo.isUseDefaultPort[0] = false;
    changeLinkInfo.isChangeLinkFlag = true;
    return HCCL_SUCCESS;
}

TEST_F(SwitchNicTest, ut_hcclCommAicpu_SwitchNic)
{
    hccl::HcclCommAicpu* hcclCommAicpu = new hccl::HcclCommAicpu;

    MOCKER_CPP(&AicpuHdc::GetOpExecChangeLink).stubs().will(invoke(GetChangeLinkInfo));

    MOCKER_CPP(&HcclCommAicpu::SwitchNicWaitHandleCommand).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommAicpu::SwitchNicWaitResult).stubs().will(returnValue(HCCL_SUCCESS));

    AlgResourceResponse resourceResponse;
    resourceResponse.opTransportResponse.resize(COMM_LEVEL_RESERVED);
    resourceResponse.opTransportResponse[COMM_LEVEL0].resize(2);
    resourceResponse.opTransportResponse[COMM_LEVEL0][0].transportRequests.resize(2);
    resourceResponse.opTransportResponse[COMM_LEVEL0][1].transportRequests.resize(2);
    resourceResponse.opTransportResponse[COMM_LEVEL0][1].transportRequests[0].isValid = true;
    resourceResponse.opTransportResponse[COMM_LEVEL0][1].transportRequests[0].isUsedRdma = true;
    resourceResponse.opTransportResponse[COMM_LEVEL0][1].transportRequests[0].remoteUserRank = 1;
    resourceResponse.opTransportResponse[COMM_LEVEL0][1].links.resize(2);
    hcclCommAicpu->resMap_.emplace("test1", resourceResponse);

    LINK backupLink;
    DispatcherPub* dispatcher = new (std::nothrow) DispatcherPub(0);
    TransportPara para;
    std::unique_ptr<NotifyPool> notifyPool = nullptr;
    notifyPool.reset(new (std::nothrow) NotifyPool());
    MachinePara machinePara;
    para.timeout = std::chrono::seconds(300);
    TransportDeviceP2pData transDevP2pData;
    TransportDeviceIbverbsData transDevIbverbsData;
    backupLink.reset(new Transport(
        TransportType::TRANS_TYPE_IBV_EXP, para, dispatcher, notifyPool, machinePara, transDevP2pData,
        transDevIbverbsData));
    vector<LINK> linkRes;
    linkRes.emplace_back(backupLink);
    linkRes.emplace_back(backupLink);

    std::unordered_map<std::string, std::vector<std::shared_ptr<Transport>>> transportRes;
    transportRes.emplace("test1", linkRes);

    hcclCommAicpu->linkRdmaRes_.emplace(1, transportRes);
    hcclCommAicpu->linkRdmaResBackUp_.emplace(1, transportRes);

    HcclResult ret = hcclCommAicpu->SwitchNic();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    delete dispatcher;
    delete hcclCommAicpu;
    GlobalMockObject::verify();
}

HcclResult GetHandleCmdSuccess(AicpuHdc* This, shared_ptr<HDCommunicate> h2dTransfer, KfcCommand& cmd)
{
    cmd = KfcCommand::kWaitSwitchNic;
    return HCCL_SUCCESS;
}
HcclResult GetHandleCmdNone(AicpuHdc* This, shared_ptr<HDCommunicate> h2dTransfer, KfcCommand& cmd)
{
    cmd = KfcCommand::kNone;
    return HCCL_SUCCESS;
}

TEST_F(SwitchNicTest, ut_hcclCommAicpu_SwitchNic_WaitHandleCommand)
{
    hccl::HcclCommAicpu* hcclCommAicpu = new hccl::HcclCommAicpu;
    AlgResourceResponse resourceResponse;
    resourceResponse.opTransportResponse.resize(COMM_LEVEL_RESERVED);
    resourceResponse.opTransportResponse[COMM_LEVEL0].resize(2);
    resourceResponse.opTransportResponse[COMM_LEVEL0][0].transportRequests.resize(2);
    resourceResponse.opTransportResponse[COMM_LEVEL0][1].transportRequests.resize(2);
    resourceResponse.opTransportResponse[COMM_LEVEL0][1].transportRequests[0].isValid = true;
    resourceResponse.opTransportResponse[COMM_LEVEL0][1].transportRequests[0].isUsedRdma = true;
    resourceResponse.opTransportResponse[COMM_LEVEL0][1].transportRequests[0].remoteUserRank = 1;
    resourceResponse.opTransportResponse[COMM_LEVEL0][1].links.resize(2);
    hcclCommAicpu->resMap_.emplace("test1", resourceResponse);

    std::unordered_map<std::string, OpCommTransport> reservedLinks;
    reservedLinks.emplace("test1", resourceResponse.opTransportResponse);

    MOCKER_CPP(&AicpuHdc::GetOpExecCtrlCmd).stubs().will(invoke(GetHandleCmdSuccess));

    HcclResult ret = hcclCommAicpu->SwitchNicWaitHandleCommand(reservedLinks);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    MOCKER_CPP(&AicpuHdc::GetOpExecCtrlCmd).stubs().will(invoke(GetHandleCmdNone));

    MOCKER_CPP(&HcclCommAicpu::HcclGetWaitRetryCmdTimeout).stubs().will(returnValue(0));

    ret = hcclCommAicpu->SwitchNicWaitHandleCommand(reservedLinks);
    EXPECT_EQ(ret, HCCL_E_TIMEOUT);
    GlobalMockObject::verify();

    delete hcclCommAicpu;
}

HcclResult GetResultCmdSuccess(AicpuHdc* This, shared_ptr<HDCommunicate> h2dTransfer, KfcCommand& cmd)
{
    cmd = KfcCommand::kAllSwitched;
    return HCCL_SUCCESS;
}

HcclResult GetResultCmdFail(AicpuHdc* This, shared_ptr<HDCommunicate> h2dTransfer, KfcCommand& cmd)
{
    cmd = KfcCommand::kSwitchFail;
    return HCCL_SUCCESS;
}

HcclResult GetResultCmdNone(AicpuHdc* This, shared_ptr<HDCommunicate> h2dTransfer, KfcCommand& cmd)
{
    cmd = KfcCommand::kNone;
    return HCCL_SUCCESS;
}

TEST_F(SwitchNicTest, ut_hcclCommAicpu_SwitchNic_WaitResult)
{
    hccl::HcclCommAicpu* hcclCommAicpu = new hccl::HcclCommAicpu;

    AlgResourceResponse resourceResponse;
    resourceResponse.opTransportResponse.resize(COMM_LEVEL_RESERVED);
    resourceResponse.opTransportResponse[COMM_LEVEL0].resize(2);
    resourceResponse.opTransportResponse[COMM_LEVEL0][0].transportRequests.resize(2);
    resourceResponse.opTransportResponse[COMM_LEVEL0][1].transportRequests.resize(2);
    resourceResponse.opTransportResponse[COMM_LEVEL0][1].transportRequests[0].isValid = true;
    resourceResponse.opTransportResponse[COMM_LEVEL0][1].transportRequests[0].isUsedRdma = true;
    resourceResponse.opTransportResponse[COMM_LEVEL0][1].transportRequests[0].remoteUserRank = 1;
    resourceResponse.opTransportResponse[COMM_LEVEL0][1].links.resize(2);
    hcclCommAicpu->resMap_.emplace("test1", resourceResponse);

    std::unordered_map<std::string, OpCommTransport> reservedLinks;
    reservedLinks.emplace("test1", resourceResponse.opTransportResponse);

    MOCKER_CPP(&AicpuHdc::GetOpExecCtrlCmd).stubs().will(invoke(GetResultCmdSuccess));

    HcclResult ret = hcclCommAicpu->SwitchNicWaitResult(reservedLinks);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    MOCKER_CPP(&AicpuHdc::GetOpExecCtrlCmd).stubs().will(invoke(GetResultCmdFail));

    ret = hcclCommAicpu->SwitchNicWaitResult(reservedLinks);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    GlobalMockObject::verify();

    MOCKER_CPP(&AicpuHdc::GetOpExecCtrlCmd).stubs().will(invoke(GetResultCmdNone));

    MOCKER(GetExternalInputHcclLinkTimeOut).stubs().will(returnValue(0));

    ret = hcclCommAicpu->SwitchNicWaitResult(reservedLinks);
    EXPECT_EQ(ret, HCCL_E_TIMEOUT);
    GlobalMockObject::verify();

    delete hcclCommAicpu;
}

hccl::HcclCommAicpu g_commForTest;
HcclResult AicpuGetCommAll_stub(std::vector<std::pair<std::string, hccl::HcclCommAicpu*>>& aicpuCommInfo)
{
    g_commForTest.InitCommInfoStatus(true);
    aicpuCommInfo.push_back(std::make_pair("group1", &g_commForTest));
    return HCCL_SUCCESS;
}

HcclResult BackGroundGetCmd_stub(hccl::HcclCommAicpu* This, KfcCommand& cmd)
{
    cmd = KfcCommand::kSwitchNic;
    return HCCL_SUCCESS;
}

TEST_F(SwitchNicTest, ut_ExecutorTracer_HandleSwitchNic)
{
    AicpuComContext* ctx = AicpuGetComContext();
    ctx->isStopLaunch = false;
    KfcCommand kCmd = KfcCommand::kDestroyComm;
    memset_s(&kCmd, sizeof(KfcCommand), 0, sizeof(KfcCommand));
    KfcExecStatus response;
    memset_s(&response, sizeof(KfcExecStatus), 0, sizeof(KfcExecStatus));
    MOCKER(AicpuHcclProcess::AicpuGetCommAll).stubs().will(invoke(AicpuGetCommAll_stub));
    MOCKER_CPP(&HcclCommAicpu::BackGroundGetCmd).stubs().will(invoke(BackGroundGetCmd_stub));
    MOCKER_CPP(&HcclCommAicpu::SwitchNic).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommAicpu::ResponseBackGroundStatus).stubs().will(returnValue(HCCL_SUCCESS));
    dfx_tracer::ExecutorTracer::HandleSwitchNic(ctx);
}

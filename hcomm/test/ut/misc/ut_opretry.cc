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
#include <sys/time.h>
#include <map>
#include <utility>
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
#undef private
#undef protected
#include "adapter_rts.h"
using namespace std;
using namespace hccl;

constexpr u32 TEST_RANK_NUM = 10;
class RetryTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RetryTest SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "RetryTest TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = -1;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};

class RetrySon : public OpRetryBase {
public:
    RetrySon() {};
    ~RetrySon() {};
    HcclResult Handle(RetryContext* retryCtx)
    {
        OpRetryBase::Handle(retryCtx);
        return HCCL_SUCCESS;
    }
    HcclResult ProcessEvent(RetryContext* retryCtx) { return HCCL_SUCCESS; }
    HcclResult ProcessError(RetryContext* retryCtx) { return HCCL_SUCCESS; }
    /* Server-Agent 交互 */
    HcclResult IssueResponse(std::shared_ptr<HcclSocket>& socket, RetryInfo& retryInfo)
    {
        OpRetryBase::IssueResponse(socket, retryInfo);
        return HCCL_SUCCESS;
    }
    HcclResult WaitResponse(std::shared_ptr<HcclSocket>& socket, RetryInfo& retryInfo) // Server等待Agent回复
    {
        OpRetryBase::WaitResponse(socket, retryInfo);
        return HCCL_SUCCESS;
    }
    HcclResult IssueCommand(std::shared_ptr<HcclSocket>& socket, RetryCommand command) // Server向Agent发送命令
    {
        OpRetryBase::IssueCommand(socket, command);
        return HCCL_SUCCESS;
    }
    HcclResult WaitCommand(std::shared_ptr<HcclSocket>& socket, RetryCommand& command) // Agent等待Server的命令, 阻塞
    {
        OpRetryBase::WaitCommand(socket, command);
        return HCCL_SUCCESS;
    }
    // server向agent发送命令携带opid
    HcclResult IssueCommandWithOpId(std::shared_ptr<HcclSocket>& socket, RetryCommandInfo& commandInfo)
    {
        OpRetryBase::IssueCommandWithOpId(socket, commandInfo);
        return HCCL_SUCCESS;
    }
    // agent等待server的命令，接收opid
    HcclResult WaitCommandWithOpId(std::shared_ptr<HcclSocket>& socket, RetryCommandInfo& commandInfo)
    {
        OpRetryBase::WaitCommandWithOpId(socket, commandInfo);
        return HCCL_SUCCESS;
    }
    /* 校验 */
    HcclResult CheckRetryInfo(RetryContext& context) // 校验收到的N个RetryInfo
    {
        OpRetryBase::CheckRetryInfo(context);
        return HCCL_SUCCESS;
    }
    HcclResult GetRetryInfo(RetryContext* retryCtx, RetryInfo& retryInfo)
    {
        OpRetryBase::GetRetryInfo(retryCtx, retryInfo);
        return HCCL_SUCCESS;
    }
    /* Agent-device 交互 */
    HcclResult GetOpExecInfo(std::shared_ptr<HDCommunicate> hdcPtr, KfcExecStatus& opInfo)
    {
        OpRetryBase::GetOpExecInfo(hdcPtr, opInfo);
        return HCCL_SUCCESS;
    }
    HcclResult SetOpExecCmd(std::shared_ptr<HDCommunicate> hdcPtr, KfcCommand opCmd)
    {
        OpRetryBase::SetOpExecCmd(hdcPtr, opCmd);
        return HCCL_SUCCESS;
    }
    HcclResult ClearStream(std::shared_ptr<HcclOpStreamRes> opStreamPtr, HcclRtStreamClearStep clearStep)
    {
        OpRetryBase::ClearStream(opStreamPtr, clearStep);
        return HCCL_SUCCESS;
    }
    HcclResult SetOpExecCmdWithOpId(std::shared_ptr<HDCommunicate> hdcPtr, KfcCommand opCmd, HcclOpIdentifier& opId)
    {
        OpRetryBase::SetOpExecCmdWithOpId(hdcPtr, opCmd, opId);
        return HCCL_SUCCESS;
    }
    HcclResult ClearStreamWithOpId(
        std::shared_ptr<HcclOpStreamRes> opStreamPtr, HcclRtStreamClearStep clearStep, HcclOpIdentifier& opId,
        HcclOpIdentifier& curOpId)
    {
        OpRetryBase::ClearStreamWithOpId(opStreamPtr, clearStep, opId, curOpId);
        return HCCL_SUCCESS;
    }
    HcclResult ResetNotify(RetryContext* retryCtx)
    {
        OpRetryBase::ResetNotify(retryCtx);
        return HCCL_SUCCESS;
    }
};

HcclResult stub_ResetNotifyForDestRank(s64 detRank) { return HCCL_SUCCESS; }

HcclResult stub_ResetNotify() { return HCCL_SUCCESS; }

auto notifyResetCallback = [](bool isSendRecv, s64 detRank) {
    return isSendRecv ? stub_ResetNotifyForDestRank(detRank) : stub_ResetNotify();
};

auto setTransportStatusCallback
    = [](const HcclOpIdentifier& opId, bool statusStop, const std::map<u32, bool>& remoteRankPortMap,
         const std::map<u32, bool>& isChangeLinkMap, bool isChangeLinkFlag) {
          return HCCL_SUCCESS;
      };

auto getSwitchRanksCallback
    = [](u32* distSwitchRankList, bool* distSwitchUseBackup, u32& distSwitchRankNum, u8* distRemoteRankNicStatus,
         u32& distRankSize, bool& needCheckDefaultNic, bool& needCheckBackupNic) {
          return HCCL_SUCCESS;
      };

HcclResult stub_WaitChangeLink(OpRetryBase* that, std::shared_ptr<HcclSocket> socket, ChangeLinkInfo& changeLinkInfo)
{
    changeLinkInfo.remoteRankNum = 1;
    return HCCL_SUCCESS;
}
#if 0
TEST_F(RetryTest, ut_retry_Agent_processEvent)
{
    MOCKER_CPP(&OpRetryBase::Send).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::Recv).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HDCommunicate::Get).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HDCommunicate::Put).stubs().with(mockcpp::any()).will(returnValue(0));

    MOCKER(hrtCtxSetCurrent).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HCCL_SUCCESS;
    u32 rankId = 0;
    //OpRetryAgentRunning Agent状态机初始化
    RetryCommandInfo context1comd;
    context1comd.command = RETRY_CMD_STOP_AICPU;
    MOCKER_CPP(&OpRetryBase::WaitCommandWithOpId).stubs().with(mockcpp::any(), outBound(context1comd)).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::SetOpExecCmdWithOpId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&OpRetryBase::GetOpExecInfo).stubs().will(returnValue(HCCL_SUCCESS));
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
    agentParam.opStreamPtr =  std::make_shared<HcclOpStreamRes>(myMap);
    s32 deviceLogicId = 0;
    HcclIpAddress deviceIP = HcclIpAddress("10.21.78.208");
    agentParam.agentInfo = {rankId, deviceLogicId, localIp, deviceIP};
    agentParam.group = "group";
    agentParam.agentConnection = std::make_shared<HcclSocket>("st_retry_Agent_processEvent",
        nullptr, localIp, 16666, HcclSocketRole::SOCKET_ROLE_SERVER);
    agentParam.isEnableBackupLink = false;
    agentParam.notifyResetCallback = notifyResetCallback;
    agentParam.setTransportStatusCallback = setTransportStatusCallback;
    agentParam.getSwitchRanksCallback = getSwitchRanksCallback;
    std::shared_ptr<OpRetryAgentRunning> opRetryAgentRunning = std::make_shared<OpRetryAgentRunning>();;
    
    RetryContext context(agentParam, opRetryAgentRunning);
    ret = opRetryAgentRunning->ProcessEvent(&context);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    MOCKER_CPP(&OpRetryBase::WaitCommandWithOpId).stubs().will(returnValue(HCCL_E_TIMEOUT));
    MOCKER_CPP(&OpRetryBase::GetOpExecInfo).stubs().will(returnValue(HCCL_SUCCESS));
    ret = opRetryAgentRunning->ProcessEvent(&context);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    std::shared_ptr<OpRetryAgentRunning> agentRunningTemp = std::make_shared<OpRetryAgentRunning>();
    RetryCommandInfo commandinfo;
    commandinfo.command = RETRY_CMD_RUNNING;
    MOCKER_CPP(&OpRetryBase::WaitCommandWithOpId)
    .stubs()
    .with(mockcpp::any(), outBound(commandinfo))
    .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&OpRetryBase::GetRetryInfo)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&OpRetryBase::IssueResponse)
    .stubs()
    .will(returnValue(HCCL_E_INTERNAL));
    MOCKER_CPP(&OpRetryBase::GetOpExecInfo).
    stubs().
    will(returnValue(HCCL_SUCCESS));
    agentRunningTemp->keepTimeout_ = std::chrono::seconds(0);
    ret = agentRunningTemp->ProcessEvent(&context);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    MOCKER_CPP(&OpRetryBase::Send).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::Recv).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HDCommunicate::Get).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HDCommunicate::Put).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::WaitCommandWithOpId).stubs().with(mockcpp::any(), outBound(context1comd)).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::SetOpExecCmdWithOpId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    // OpRetryAgentRunning ParseKfcErr
    std::shared_ptr<OpRetryAgentRunning> agentRunning = std::make_shared<OpRetryAgentRunning>();
    RetryContext agentCtx(agentParam, agentRunning);
    RetryState nextState = RETRY_STATE_RESERVED;

    agentCtx.localRetryInfo_.opInfo.execStatus.kfcError = KfcError::kNone;
    agentRunning->ParseKfcErr(&agentCtx, nextState);
    EXPECT_EQ(nextState, RETRY_STATE_RESERVED);

    agentCtx.localRetryInfo_.opInfo.execStatus.kfcError = KfcError::kSdma;
    agentRunning->ParseKfcErr(&agentCtx, nextState);
    EXPECT_EQ(nextState, RETRY_STATE_RESP_AICPU_ERR);

    agentCtx.localRetryInfo_.opInfo.execStatus.kfcError = KfcError::kRdma;
    agentRunning->ParseKfcErr(&agentCtx, nextState);
    EXPECT_EQ(nextState, RETRY_STATE_RESERVED);

    //RetryAgentResponse Agent状态机初始化
    std::shared_ptr<OpRetryAgentResponse> retryAgentResponse = std::make_shared<OpRetryAgentResponse>();
    RetryContext context1(agentParam, retryAgentResponse);
    context1.localRetryInfo_.retryState = RETRY_STATE_RESP_AICPU_ERR;
    ret = retryAgentResponse->ProcessEvent(&context1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    //RetryAgentWaitCmd Agent状态机初始化
    std::shared_ptr<OpRetryAgentWaitCmd> retryAgentWaitCmd = std::make_shared<OpRetryAgentWaitCmd>();
    RetryContext context2(agentParam, retryAgentWaitCmd);
    context1comd.command = RETRY_CMD_STOP_AICPU;
    context2.isChangeLinkInfoInit_ = true;
    MOCKER_CPP(&OpRetryBase::WaitCommandWithOpId).stubs().with(mockcpp::any(), outBound(context1comd)).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::SetOpExecCmdWithOpId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&OpRetryBase::InitChangeLinkInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    context2.localRetryInfo_.retryState = RETRY_STATE_WAIT_CMD_STOP_AICPU;
    ret = retryAgentWaitCmd->ProcessEvent(&context2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HcclOpIdentifier opId;
    opId.index = 9;
    HcclOpIdentifier curOpId;
    curOpId.index = rankId;
    curOpId.isSendRecv = true;
    curOpId.streamId = slaves[0].id();
    ret = retryAgentWaitCmd->ClearStreamWithOpId(agentParam.opStreamPtr, HcclRtStreamClearStep::HCCL_STREAM_CLEAR,
        opId, curOpId);
    GlobalMockObject::verify();
    context1comd.command = RETRY_CMD_CLEAR_STREAM;
    MOCKER_CPP(&OpRetryBase::WaitCommandWithOpId).stubs().with(mockcpp::any(), outBound(context1comd)).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::SetOpExecCmdWithOpId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&OpRetryBase::InitChangeLinkInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    context2.localRetryInfo_.retryState = RETRY_STATE_WAIT_CMD_CLEAR_STREAM;
    ret = retryAgentWaitCmd->ProcessEvent(&context2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
    context1comd.command = RETRY_CMD_RESET_NOTIFY;
    MOCKER_CPP(&OpRetryBase::WaitCommandWithOpId).stubs().with(mockcpp::any(), outBound(context1comd)).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::SetOpExecCmdWithOpId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&OpRetryBase::InitChangeLinkInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    context2.localRetryInfo_.retryState = RETRY_STATE_WAIT_CMD_RESET_NOTIFY;
    ret = retryAgentWaitCmd->ProcessEvent(&context2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
    context1comd.command = RETRY_CMD_CHECK_OPNAME;
    MOCKER_CPP(&OpRetryBase::WaitCommandWithOpId).stubs().with(mockcpp::any(), outBound(context1comd)).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::SetOpExecCmdWithOpId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&OpRetryBase::GetRetryInfo).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::InitChangeLinkInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    context2.localRetryInfo_.retryState = RETRY_STATE_WAIT_CMD_CHECK;
    ret = retryAgentWaitCmd->ProcessEvent(&context2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
    context1comd.command = RETRY_CMD_CAN_RETRY;
    MOCKER_CPP(&OpRetryBase::WaitCommandWithOpId).stubs().with(mockcpp::any(), outBound(context1comd)).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::SetOpExecCmdWithOpId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&OpRetryBase::SetOpExecCmd).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::InitChangeLinkInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    context2.localRetryInfo_.retryState = RETRY_STATE_WAIT_CMD_CAN_RETRY;
    ret = retryAgentWaitCmd->ProcessEvent(&context2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
    // 删除205s超时用例 影响线上llt运行时长
    context1comd.command = RETRY_CMD_STOP_STREAM;
    MOCKER_CPP(&OpRetryBase::WaitCommandWithOpId).stubs().with(mockcpp::any(), outBound(context1comd)).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::SetOpExecCmdWithOpId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&OpRetryBase::SetOpExecCmd).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::InitChangeLinkInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    context2.localRetryInfo_.retryState = RETRY_STATE_WAIT_CMD_STOP_STREAM;
    ret = retryAgentWaitCmd->ProcessEvent(&context2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
    context1comd.command = RETRY_CMD_CHECK_LINK;
    MOCKER_CPP(&OpRetryBase::WaitCommandWithOpId).stubs().with(mockcpp::any(), outBound(context1comd)).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::SetOpExecCmdWithOpId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&OpRetryBase::SetOpExecCmd).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::GetLinkPortStatus).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::InitChangeLinkInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    context2.localRetryInfo_.retryState = RETRY_STATE_WAIT_CMD_CHECK_LINK;
    ret = retryAgentWaitCmd->ProcessEvent(&context2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
    context1comd.command = RETRY_CMD_STOP_TRANSPORT;
    MOCKER_CPP(&OpRetryBase::WaitCommandWithOpId).stubs().with(mockcpp::any(), outBound(context1comd)).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::SetOpExecCmdWithOpId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&OpRetryBase::SetOpExecCmd).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::SetTransportStatusForStop).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::SetTransportStatusForResume).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::InitChangeLinkInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    context2.localRetryInfo_.retryState = RETRY_STATE_WAIT_CMD_STOP_TRANSPORT;
    ret = retryAgentWaitCmd->ProcessEvent(&context2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
    context1comd.command = RETRY_CMD_RESUME_TRANSPORT;
    MOCKER_CPP(&OpRetryBase::WaitCommandWithOpId).stubs().with(mockcpp::any(), outBound(context1comd)).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::SetOpExecCmdWithOpId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&OpRetryBase::SetOpExecCmd).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::SetTransportStatusForStop).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::SetTransportStatusForResume).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::SetOpChangeLinkInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&OpRetryBase::InitChangeLinkInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    context2.localRetryInfo_.retryState = RETRY_STATE_WAIT_CMD_RESUME_TRANSPORT;
    ret = retryAgentWaitCmd->ProcessEvent(&context2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
    //RetryAgentPollAicpuStop Agent状态机初始化
    std::shared_ptr<OpRetryAgentPollAicpuStop> retryAgentPollAicpuStop = std::make_shared<OpRetryAgentPollAicpuStop>();
    RetryContext context3(agentParam, retryAgentPollAicpuStop);
    KfcExecStatus opInfo;
    context3.state_ = RETRY_STATE_POLL_AICPU_STOPED;
    opInfo.execStatus.kfcStatus = KfcStatus::kRuning;
    context3.SetRetryState(RETRY_STATE_POLL_AICPU_STOPED, retryAgentPollAicpuStop);
    MOCKER_CPP(&OpRetryBase::GetOpExecInfo).stubs().with(mockcpp::any(), outBound(opInfo)).will(returnValue(0));
    ret = retryAgentPollAicpuStop->ProcessEvent(&context3);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
    context3.state_ = RETRY_STATE_POLL_STREAM_STOPED;
    context3.SetRetryState(RETRY_STATE_POLL_STREAM_STOPED, retryAgentPollAicpuStop);
    opInfo.execStatus.kfcStatus = KfcStatus::kStopExec;
    MOCKER_CPP(&OpRetryBase::GetOpExecInfo).stubs().with(mockcpp::any(), outBound(opInfo)).will(returnValue(0));
    ret = retryAgentPollAicpuStop->ProcessEvent(&context3);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
    context3.state_ = RETRY_STATE_POLL_AICPU_RETRYEND;
    context3.SetRetryState(RETRY_STATE_POLL_AICPU_RETRYEND, retryAgentPollAicpuStop);
    opInfo.execStatus.kfcStatus = KfcStatus::kEnd;
    MOCKER_CPP(&OpRetryBase::GetOpExecInfo).stubs().with(mockcpp::any(), outBound(opInfo)).will(returnValue(0));
    ret = retryAgentPollAicpuStop->ProcessEvent(&context3);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
    //RetryAgentRetryFail Agent状态机初始化
    std::shared_ptr<OpRetryAgentRetryFail> retryAgentRetryFail = std::make_shared<OpRetryAgentRetryFail>();
    MOCKER_CPP(&OpRetryBase::SetOpExecCmd).stubs().with(mockcpp::any()).will(returnValue(0));
    RetryContext context4(agentParam, retryAgentRetryFail);
    ret = retryAgentRetryFail->ProcessEvent(&context4);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
    //OpRetryAgentResponseLinkInfo Agent状态机初始化
    std::shared_ptr<OpRetryAgentResponseLinkInfo> retryAgentResponseLinkInfo = std::make_shared<OpRetryAgentResponseLinkInfo>();
    MOCKER_CPP(&OpRetryBase::IssueLinkPortCheckResult).stubs().with(mockcpp::any()).will(returnValue(0));
    RetryContext context5(agentParam, retryAgentResponseLinkInfo);
    ret = retryAgentResponseLinkInfo->ProcessEvent(&context5);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
    //OpRetryAgentWaitChangeLinkInfo Agent状态机初始化
    std::shared_ptr<OpRetryAgentWaitChangeLinkInfo> retryAgentWaitChangeLinkInfo = std::make_shared<OpRetryAgentWaitChangeLinkInfo>();
    MOCKER_CPP(&OpRetryBase::WaitChangeLink).stubs().with(mockcpp::any()).will(invoke(stub_WaitChangeLink));
    RetryContext context6(agentParam, retryAgentWaitChangeLinkInfo);
    context6.localChangeLinkInfo_.remoteRankNum = 1;
    ret = retryAgentWaitChangeLinkInfo->ProcessEvent(&context6);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(RetryTest, ut_retry_Server_processEvent)
{
    MOCKER_CPP(&OpRetryBase::Send).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::Recv).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER(hrtCtxSetCurrent).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HCCL_SUCCESS;
    HcclIpAddress localIp = HcclIpAddress("192.168.100.110");
    u32 rankId = 0;
    HcclOpStreamRes myMap;
    std::string key = "example_key";
    std::vector<Stream> slaves;
    for (int i = 0; i < 10; i++) {
        slaves.push_back(Stream(StreamType::STREAM_TYPE_OFFLINE));
    }
    myMap[key] = slaves;
    std::shared_ptr<HDCommunicate> h2dPtr;
    h2dPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl)));
    std::shared_ptr<HDCommunicate> d2hPtr;
    d2hPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_D2H, sizeof(KfcExecStatus)));
    std::shared_ptr<HcclOpStreamRes> opStream =  std::make_shared<HcclOpStreamRes>(myMap);

    std::shared_ptr<HcclSocket> ServerSocket1(new (std::nothrow)HcclSocket("Retryfunction1",
        nullptr, localIp, 16666, HcclSocketRole::SOCKET_ROLE_CLIENT));
    std::map<u32, std::shared_ptr<HcclSocket> > ServerSockets;
    ServerSockets.insert(std::make_pair(0, ServerSocket1));
    HcclIpAddress deviceIP = HcclIpAddress("10.21.78.208");
    s32 deviceLogicId_1 = 0;
    OpRetryAgentInfo agentInfo = {rankId, deviceLogicId_1, localIp, deviceIP};

    //RetryServerRunning Server状态机初始化
    std::shared_ptr<OpRetryServerRunning> retryServerRunning = std::make_shared<OpRetryServerRunning>();
    RetryContext context(ServerSockets, retryServerRunning, agentInfo);

    HcclOpIdentifier Identify;
    Identify.index = rankId;
    KfcExecStatus status;
    status.opId = Identify;
    status.execStatus.kfcStatus = KfcStatus::kRuning;
    status.execStatus.kfcError = KfcError::kNone;
    RetryInfo retryinfo;
    retryinfo.rankId = 1;
    retryinfo.retryState = RETRY_STATE_SERVER_RUNNING;
    retryinfo.linkState = true;
    retryinfo.opInfo = status;
    retryinfo.isChangeLinkFlag = true;
    HcclAgentRetryInfo info;
    info.socket = ServerSocket1;
    info.retryInfo = retryinfo;
    for(auto & mapinfo: context.serverSockets_)
    {
        mapinfo.second = info;
    }
    HcclOpIdentifier IdentifyError;
    IdentifyError.index = 1;
    IdentifyError.isSendRecv = true;
    context.errorRankList_.emplace(1, IdentifyError);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    //RetryServerIssueCmd Server状态机初始化
    std::shared_ptr<OpRetryServerIssueCmd> retryServerIssueCmd = std::make_shared<OpRetryServerIssueCmd>();
    RetryContext context1(ServerSockets, retryServerIssueCmd, agentInfo);
    context1.SetRetryState(RETRY_STATE_SERVER_RUNNING, retryServerIssueCmd);
    context1.SetRetryState(RETRY_STATE_CMD_STOP_AICPU, retryServerIssueCmd);
    context1.needRetryServerRanks_.push_back(0);
    ret = retryServerIssueCmd->ProcessEvent(&context1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    //RetryServerWaitResp Server状态机初始化
    std::shared_ptr<OpRetryServerWaitResp> retryServerWaitResp = std::make_shared<OpRetryServerWaitResp>();
    RetryContext context2(ServerSockets, retryServerWaitResp, agentInfo);
    context2.SetRetryState(RETRY_STATE_SERVER_RUNNING, retryServerWaitResp);
    context2.SetRetryState(RETRY_STATE_CMD_STOP_AICPU, retryServerWaitResp);
    context2.SetRetryState(RETRY_STATE_WAIT_LINK_CHECKED, retryServerWaitResp);
    context2.SetRetryState(RETRY_STATE_WAIT_AICPU_STOPED, retryServerWaitResp);
    ret = retryServerWaitResp->ProcessEvent(&context2);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    //RetryServerCheckOp Server状态机初始化
    std::shared_ptr<OpRetryServerCheckOp> retryServerCheckOp = std::make_shared<OpRetryServerCheckOp>();
    RetryContext context3(ServerSockets, retryServerCheckOp, agentInfo);
    for(auto & mapinfo: context3.serverSockets_)
    {
       mapinfo.second =  info;
    }
    context3.needRetryServerRanks_.emplace_back(0);
    ret = retryServerCheckOp->ProcessEvent(&context3);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    //RetryServerCheckOp Server状态机初始化 for sendRecv
    std::shared_ptr<OpRetryServerCheckOp> retryServerCheckOpForSendRecv = std::make_shared<OpRetryServerCheckOp>();
    RetryContext context3ForSendRecv(ServerSockets, retryServerCheckOpForSendRecv, agentInfo);
    for(auto & mapinfo: context3ForSendRecv.serverSockets_)
    {
        info.retryInfo.opInfo.opId.isSendRecv = true;
        mapinfo.second = info;
    }
    context3ForSendRecv.needRetryServerRanks_.emplace_back(0);
    ret = retryServerCheckOpForSendRecv->ProcessEvent(&context3ForSendRecv);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    //RetryServerRetryFail Server状态机初始化
    std::shared_ptr<OpRetryServerRetryFail> retryServerRetryFail = std::make_shared<OpRetryServerRetryFail>();
    RetryContext context4(ServerSockets, retryServerRetryFail, agentInfo);
    for(auto & mapinfo: context4.serverSockets_)
    {
       mapinfo.second =  info;
    }
    ret = retryServerRetryFail->ProcessEvent(&context4);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    //OpRetryServerCheckAllLink Server状态机初始化
    std::shared_ptr<OpRetryServerCheckAllLink> retryServerCheckAllLink = std::make_shared<OpRetryServerCheckAllLink>();
    RetryContext context5(ServerSockets, retryServerCheckAllLink, agentInfo);
    for(auto & mapinfo: context5.serverSockets_)
    {
       mapinfo.second =  info;
    }
    context5.needRetryServerRanks_.push_back(0);
    ret = retryServerCheckAllLink->ProcessEvent(&context5);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    context5.serverSockets_[0].linkPortStatus.rankSize = 1;
    context5.serverSockets_[0].linkPortStatus.rankList[0] = 0;
    context5.serverSockets_[0].linkPortStatus.defaultPort = true;
    ret = retryServerCheckAllLink->ProcessEvent(&context5);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    context5.serverSockets_[0].linkPortStatus.defaultPort = false;
    context5.serverSockets_[0].linkPortStatus.defaultPort = true;
    ret = retryServerCheckAllLink->ProcessEvent(&context5);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    //OpRetryServerWaitLinkInfo Server状态机初始化
    std::shared_ptr<OpRetryServerWaitLinkInfo> retryServerWaitLinkInfo = std::make_shared<OpRetryServerWaitLinkInfo>();
    RetryContext context7(ServerSockets, retryServerWaitLinkInfo, agentInfo);
    for(auto & mapinfo: context7.serverSockets_)
    {
       mapinfo.second =  info;
    }
    context7.needRetryServerRanks_.push_back(0);
    ret = retryServerWaitLinkInfo->ProcessEvent(&context7);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // OpRetryServerIssueChangeLinkAndResume Server状态机初始化
    std::shared_ptr<OpRetryServerIssueChangeLinkAndResume> retryServerIssueChangeLink = std::make_shared<OpRetryServerIssueChangeLinkAndResume>();
    RetryContext context8(ServerSockets, retryServerIssueChangeLink, agentInfo);
    for(auto & mapinfo: context8.serverSockets_)
    {
       mapinfo.second =  info;
    }
    context8.needRetryServerRanks_.push_back(0);
    ret = retryServerIssueChangeLink->ProcessEvent(&context8);
    
    //OpRetryServerHandleError Server状态机初始化
    std::shared_ptr<OpRetryServerHandleError> retryServerHandleError = std::make_shared<OpRetryServerHandleError>();
    RetryContext context9(ServerSockets, retryServerHandleError, agentInfo);
    for(auto & mapinfo: context9.serverSockets_)
    {
       mapinfo.second =  info;
    }
        ret = retryServerHandleError->ProcessEvent(&context9);

    MOCKER_CPP(&OpRetryBase::WaitResponse)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&OpRetryServerRunning::ParaseErrorCode)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    //OpRetryServerHandleError Server状态机初始化 for batchSendRecv
    std::shared_ptr<OpRetryServerHandleError> retryServerHandleErrorForBatchSendRecv = std::make_shared<OpRetryServerHandleError>();
    RetryContext context10(ServerSockets, retryServerHandleErrorForBatchSendRecv, agentInfo);
    for(auto & mapinfo: context10.serverSockets_)
    {
        mapinfo.second = info;
    }
    HcclOpIdentifier srIdentify;
    srIdentify.index = 1;
    srIdentify.detRank = 1;
    srIdentify.srcRank = 1;
    strcpy_s((char *)srIdentify.tag, 128, "sendRecv");
    srIdentify.isSendRecv = true;
    strcpy_s((char *)srIdentify.bsrInfo[0].bsrTag, 128, "sendRecv");
    strcpy_s((char *)srIdentify.bsrInfo[1].bsrTag, 128, "sendRecv");
    srIdentify.bsrInfo[0].index = 1;
    srIdentify.bsrInfo[1].index = 1;
    HcclAgentRetryInfo info1;
    info1.socket = ServerSocket1;
    info1.retryInfo = retryinfo;
    info1.retryInfo.opInfo.opId = srIdentify;
    context10.errorRankList_.emplace(1, srIdentify);
    context10.needRetryServerRanks_.push_back(0);
    context10.serverSockets_.emplace(1, info1);
    ret = retryServerHandleErrorForBatchSendRecv->ProcessEvent(&context10);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RetryTest, ut_retry_Server_SetNeedRetryServerRank)
{
    MOCKER_CPP(&OpRetryBase::Send).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::Recv).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER(hrtCtxSetCurrent).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HCCL_SUCCESS;
    HcclIpAddress localIp = HcclIpAddress("192.168.100.110");
    u32 rankId = 0;

    std::shared_ptr<HcclSocket> ServerSocket1(new (std::nothrow)HcclSocket("Retryfunction1",
        nullptr, localIp, 16666, HcclSocketRole::SOCKET_ROLE_CLIENT));
    std::map<u32, std::shared_ptr<HcclSocket> > ServerSockets;
    ServerSockets.insert(std::make_pair(0, ServerSocket1));
    HcclIpAddress deviceIP = HcclIpAddress("10.21.78.208");
    s32 deviceLogicId_1 = 0;
    OpRetryAgentInfo agentInfo = {rankId, deviceLogicId_1, localIp, deviceIP};

    //retryServerHandleError Server状态机初始化
    std::shared_ptr<OpRetryServerHandleError> retryServerHandleError = std::make_shared<OpRetryServerHandleError>();
    RetryContext context(ServerSockets, retryServerHandleError, agentInfo);

    HcclOpIdentifier Identify;
    Identify.index = rankId;
    Identify.srcRank = rankId;
    Identify.detRank = rankId;
    KfcExecStatus status;
    status.opId = Identify;
    status.execStatus.kfcStatus = KfcStatus::kRuning;
    status.execStatus.kfcError = KfcError::kNone;
    RetryInfo retryinfo;
    retryinfo.rankId = 1;
    retryinfo.retryState = RETRY_STATE_SERVER_RUNNING;
    retryinfo.linkState = true;
    retryinfo.opInfo = status;
    HcclAgentRetryInfo info;
    info.socket = ServerSocket1;
    info.retryInfo = retryinfo;
    for(auto & mapinfo: context.serverSockets_)
    {
       mapinfo.second =  info;
    }
    Identify.isSendRecv = true;
    ret = retryServerHandleError->SetNeedRetryServerRank(&context, Identify);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    Identify.detRank = 1;
    ret = retryServerHandleError->SetNeedRetryServerRank(&context, Identify);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    Identify.isSendRecv = false;
    ret = retryServerHandleError->SetNeedRetryServerRank(&context, Identify);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RetryTest, ut_retry_Server_handleErrTimeout)
{
    MOCKER_CPP(&OpRetryBase::Send).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::Recv).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER(hrtCtxSetCurrent).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HCCL_SUCCESS;
    HcclIpAddress localIp = HcclIpAddress("192.168.100.110");
    HcclIpAddress remoteIp = HcclIpAddress("192.168.100.112");
    u32 rankId = 0;

    std::shared_ptr<HcclSocket> ServerSocket1(new (std::nothrow)HcclSocket("Retryfunction1",
        nullptr, localIp, 16666, HcclSocketRole::SOCKET_ROLE_CLIENT));
    std::shared_ptr<HcclSocket> ServerSocket2(new (std::nothrow)HcclSocket("Retryfunction2",
        nullptr, remoteIp, 16667, HcclSocketRole::SOCKET_ROLE_CLIENT));
    std::map<u32, std::shared_ptr<HcclSocket> > ServerSockets;
    ServerSockets.insert(std::make_pair(1, ServerSocket1));
    ServerSockets.insert(std::make_pair(2, ServerSocket2));
    HcclIpAddress deviceIP = HcclIpAddress("10.21.78.208");
	s32 deviceLogicId_1 = 0;
    OpRetryAgentInfo agentInfo = {rankId, deviceLogicId_1, localIp, deviceIP};

    //OpRetryServerHandleError Server状态机初始化 for batchSendRecv
    std::shared_ptr<OpRetryServerHandleError> retryServerHandleErrorForBatchSendRecv = std::make_shared<OpRetryServerHandleError>();
    RetryContext context10(ServerSockets, retryServerHandleErrorForBatchSendRecv, agentInfo);

    HcclOpIdentifier srIdentify;
    srIdentify.index = 2;
    strcpy_s((char *)srIdentify.tag, 128, "allgather");
    srIdentify.isSendRecv = false;
    HcclOpIdentifier dstdentify;
    dstdentify.index = 2;
    strcpy_s((char *)dstdentify.tag, 128, "reducescater");
    dstdentify.isSendRecv = false;
   
    RetryInfo retryinfo;
    HcclAgentRetryInfo info1;
    info1.socket = ServerSocket1;
    info1.retryInfo = retryinfo;
    info1.retryInfo.opInfo.opId = srIdentify;

    HcclAgentRetryInfo info2;
    info2.socket = ServerSocket2;
    info2.retryInfo = retryinfo;
    info2.retryInfo.opInfo.opId = dstdentify;
    context10.errorRankList_.emplace(1, srIdentify);

    context10.serverSockets_.emplace(1, info1);
    context10.serverSockets_.emplace(2, info2);
    // 超时场景模拟 OP_RETRY_WAIT_CAN_RETRY_RANK超时时长60s影响线上执行时长 无法临时只能下掉
}

TEST_F(RetryTest, ut_retry_base_function)
{
    MOCKER_CPP(&HDCommunicate::Get).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HDCommunicate::Put).stubs().with(mockcpp::any()).will(returnValue(0));
    HcclResult ret = HCCL_SUCCESS;
    HcclIpAddress localIp = HcclIpAddress("192.168.100.110");
    u32 rankId = 0;
    HcclOpStreamRes myMap;
    std::string key = "example_key";
    std::vector<Stream> slaves;
    for (int i = 0; i < 10; i++) {
        slaves.push_back(Stream(StreamType::STREAM_TYPE_OFFLINE));
    }
    myMap[key] = slaves;
    std::shared_ptr<HDCommunicate> h2dPtr;
    h2dPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl)));
    std::shared_ptr<HcclOpStreamRes> opStream =  std::make_shared<HcclOpStreamRes>(myMap);
    std::shared_ptr<HcclSocket> ServerSocket1(new (std::nothrow)HcclSocket("Retryfunction1",
        nullptr, localIp, 16666, HcclSocketRole::SOCKET_ROLE_CLIENT));
    std::map<u32, std::shared_ptr<HcclSocket> > ServerSockets;
    ServerSockets.insert(std::make_pair(0, ServerSocket1));
    HcclIpAddress deviceIP = HcclIpAddress("10.21.78.208");
	s32 deviceLogicId_1 = 0;
    OpRetryAgentInfo agentInfo = {rankId, deviceLogicId_1, localIp, deviceIP};
    //RetryServerRunning Server状态机初始化
    std::shared_ptr<RetrySon> retrySon = std::make_shared<RetrySon>();
    RetryContext context(ServerSockets, retrySon, agentInfo);
    context.needRetryServerRanks_.push_back(0);
    HcclOpIdentifier Identify;
    Identify.index = rankId;
    KfcExecStatus status;
    status.opId = Identify;
    status.execStatus.kfcStatus = KfcStatus::kRuning;
    status.execStatus.kfcError = KfcError::kNone;
    RetryInfo retryinfo;
    retryinfo.rankId = 1;
    retryinfo.retryState = RETRY_STATE_SERVER_RUNNING;
    retryinfo.linkState = true;
    retryinfo.opInfo = status;
    retryinfo.opInfo.execStatus.retryInfo.retryCount = 0;
    HcclAgentRetryInfo info;
    info.socket = ServerSocket1;
    info.retryInfo = retryinfo;
    KfcCommand opCmd = KfcCommand::kNone;
  
    for(auto & mapinfo: context.serverSockets_)
    {
       mapinfo.second =  info;
    }
    ret = retrySon->GetRetryInfo(&context, retryinfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = retrySon->GetOpExecInfo(h2dPtr, status);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = retrySon->SetOpExecCmd(h2dPtr, opCmd);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = retrySon->ClearStream(opStream, HcclRtStreamClearStep::HCCL_STREAM_STOP);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = retrySon->ProcessEvent(&context);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = retrySon->IssueResponse(ServerSocket1, retryinfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = retrySon->WaitResponse(ServerSocket1, retryinfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    RetryCommand command = RETRY_CMD_RUNNING;
    ret = retrySon->IssueCommand(ServerSocket1, command);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = retrySon->WaitCommand(ServerSocket1, command);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = retrySon->CheckRetryInfo(context);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = retrySon->Handle(&context);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RetryTest, ut_retry_base_function_withLink)
{
    MOCKER_CPP(&OpRetryBase::Send).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::Recv).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER(OpRetryManager::GetLinkInfoByIdentifier).stubs().with(mockcpp::any()).will(returnValue(0));
    HcclResult ret = HCCL_SUCCESS;
    HcclIpAddress localIp = HcclIpAddress("192.168.100.110");
    std::shared_ptr<HcclSocket> ServerSocket1(new (std::nothrow)HcclSocket("Retryfunction1",
        nullptr, localIp, 16666, HcclSocketRole::SOCKET_ROLE_CLIENT));
    u32 rankId = 0;
    LinkPortStatus linkPortStatus;
    std::shared_ptr<RetrySon> retrySon = std::make_shared<RetrySon>();
    ret = retrySon->IssueLinkPortCheckResult(ServerSocket1, linkPortStatus);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = retrySon->WaitLinkPortCheckResult(ServerSocket1, linkPortStatus);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ChangeLinkInfo changeLinkInfo;
    ret = retrySon->IssueChangeLink(ServerSocket1, changeLinkInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = retrySon->WaitChangeLink(ServerSocket1, changeLinkInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::map<u32, std::shared_ptr<HcclSocket> > ServerSockets;
    ServerSockets.insert(std::make_pair(0, ServerSocket1));
    HcclIpAddress deviceIP = HcclIpAddress("10.21.78.208");
	s32 deviceLogicId_1 = 0;
    OpRetryAgentInfo agentInfo = {rankId, deviceLogicId_1, localIp, deviceIP};
    RetryContext context(ServerSockets, retrySon, agentInfo);
    ret = retrySon->InitChangeLinkInfo(&context);
    context.localRetryInfo_.opInfo.opId.isSendRecv = true;
    ret = retrySon->InitChangeLinkInfo(&context);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    MOCKER(HrtRaRdmaGetHandle).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetPairDevicePhyId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaRdevGetPortStatus).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_INTERNAL));
    HcclNetDevCtx netDevCtx;
    ret = HcclNetOpenDev(&netDevCtx, NicType::DEVICE_NIC_TYPE, 0, 0, HcclIpAddress("0.0.0.0"));
    EXPECT_EQ(ret, HCCL_SUCCESS);
    bool portStatus;
    ret = HcclNetDevGetPortStatus(netDevCtx, portStatus);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
    HcclNetCloseDev(netDevCtx);
    MOCKER(HcclNetDevGetPortStatus).stubs().will(returnValue(0));
    ret = retrySon->GetLinkPortStatus(&context, linkPortStatus);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

HcclResult stub_GetLinkInfoByIdentifier(s32 deviceLogicID, const std::string &identifier, 
        const std::string &newTag, std::vector<u32> &remoteRankList)
{
    remoteRankList.push_back(0);
    return HCCL_SUCCESS;
}

TEST_F(RetryTest, ut_retry_base_function_withIncreLink)
{
    MOCKER_CPP(&OpRetryBase::Send).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::Recv).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER(OpRetryManager::GetLinkInfoByIdentifier).stubs().with(mockcpp::any()).will(invoke(stub_GetLinkInfoByIdentifier));
    HcclResult ret = HCCL_SUCCESS;
    HcclIpAddress localIp = HcclIpAddress("192.168.100.110");
    std::shared_ptr<HcclSocket> ServerSocket1(new (std::nothrow)HcclSocket("Retryfunction1",
        nullptr, localIp, 16666, HcclSocketRole::SOCKET_ROLE_CLIENT));
    u32 rankId = 0;
    std::shared_ptr<RetrySon> retrySon = std::make_shared<RetrySon>();

    std::map<u32, std::shared_ptr<HcclSocket> > ServerSockets;
    ServerSockets.insert(std::make_pair(0, ServerSocket1));
    HcclIpAddress deviceIP = HcclIpAddress("10.21.78.208");
	s32 deviceLogicId_1 = 0;
    OpRetryAgentInfo agentInfo = {rankId, deviceLogicId_1, localIp, deviceIP};
    RetryContext context(ServerSockets, retrySon, agentInfo);
    ret = retrySon->InitChangeLinkInfo(&context, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RetryTest, ut_retry_base_function_LinkManager)
{
    s32 deviceLogicID = 0;
    const std::string &identifier = "test"; 
    const std::string &newTag = "test_tag";
    std::vector<u32> remoteRankList = {0};
    bool incre = false;
    OpRetryManager::AddLinkInfoByIdentifier(deviceLogicID, identifier, newTag, remoteRankList, incre);
    OpRetryManager::AddLinkInfoByIdentifier(deviceLogicID, identifier, newTag, remoteRankList, incre);
    incre = true;
    OpRetryManager::AddLinkInfoByIdentifier(deviceLogicID, identifier, newTag, remoteRankList, incre);
    const std::string &newTag1 = "test_tag1";
    OpRetryManager::AddLinkInfoByIdentifier(deviceLogicID, identifier, newTag1, remoteRankList, incre);
    const std::string &identifier1 = "test1";
    OpRetryManager::AddLinkInfoByIdentifier(deviceLogicID, identifier1, newTag1, remoteRankList, incre);
    OpRetryManager::GetLinkInfoByIdentifier(deviceLogicID, identifier1, newTag1, remoteRankList);
    const std::string &newTag2 = "test_tag2";
    OpRetryManager::GetLinkInfoByIdentifier(deviceLogicID, identifier1, newTag2, remoteRankList);
    const std::string &identifier2 = "test2";
    OpRetryManager::GetLinkInfoByIdentifier(deviceLogicID, identifier2, newTag2, remoteRankList);
}


TEST_F(RetryTest, ut_retry_base_function_withopid)
{
    MOCKER_CPP(&HDCommunicate::Get).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HDCommunicate::Put).stubs().with(mockcpp::any()).will(returnValue(0));
    HcclResult ret = HCCL_SUCCESS;
    HcclIpAddress localIp = HcclIpAddress("192.168.100.110");
    u32 rankId = 0;
    HcclOpStreamRes myMap;
    std::string key = "example_key";
    std::vector<Stream> slaves;
    for (int i = 0; i < 10; i++) {
        slaves.push_back(Stream(StreamType::STREAM_TYPE_OFFLINE));
    }
    myMap[key] = slaves;
    std::shared_ptr<HDCommunicate> h2dPtr;
    h2dPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl)));
    std::shared_ptr<HcclOpStreamRes> opStream =  std::make_shared<HcclOpStreamRes>(myMap);
    std::shared_ptr<HcclSocket> ServerSocket1(new (std::nothrow)HcclSocket("Retryfunction1",
        nullptr, localIp, 16666, HcclSocketRole::SOCKET_ROLE_CLIENT));
    std::map<u32, std::shared_ptr<HcclSocket> > ServerSockets;
    ServerSockets.insert(std::make_pair(0, ServerSocket1));
    //RetryServerRunning Server状态机初始化
    std::shared_ptr<RetrySon> retrySon = std::make_shared<RetrySon>();
    HcclIpAddress deviceIP = HcclIpAddress("10.21.78.208");
	s32 deviceLogicId_1 = 0;
    OpRetryAgentInfo agentInfo = {rankId, deviceLogicId_1, localIp, deviceIP};
    RetryContext context(ServerSockets, retrySon, agentInfo);
    context.needRetryServerRanks_.push_back(0);
    HcclOpIdentifier Identify;
    Identify.index = rankId;
    KfcExecStatus status;
    status.opId = Identify;
    status.execStatus.kfcStatus = KfcStatus::kRuning;
    status.execStatus.kfcError = KfcError::kNone;
    RetryInfo retryinfo;
    retryinfo.rankId = 1;
    retryinfo.retryState = RETRY_STATE_SERVER_RUNNING;
    retryinfo.linkState = true;
    retryinfo.opInfo = status;
    retryinfo.opInfo.execStatus.retryInfo.retryCount = 0;
    HcclAgentRetryInfo info;
    info.socket = ServerSocket1;
    info.retryInfo = retryinfo;
    KfcCommand opCmd = KfcCommand::kNone;
  
    for(auto & mapinfo: context.serverSockets_)
    {
       mapinfo.second =  info;
    }
    ret = retrySon->GetRetryInfo(&context, retryinfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = retrySon->GetOpExecInfo(h2dPtr, status);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = retrySon->SetOpExecCmdWithOpId(h2dPtr, opCmd, Identify);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = retrySon->ClearStreamWithOpId(opStream, HcclRtStreamClearStep::HCCL_STREAM_STOP, Identify, Identify);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = retrySon->ProcessEvent(&context);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = retrySon->IssueResponse(ServerSocket1, retryinfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = retrySon->WaitResponse(ServerSocket1, retryinfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    RetryCommand command = RETRY_CMD_RUNNING;
    RetryCommandInfo CommandInfo;
    CommandInfo.command = command;
    CommandInfo.opId = Identify;
    ret = retrySon->IssueCommandWithOpId(ServerSocket1, CommandInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = retrySon->WaitCommandWithOpId(ServerSocket1, CommandInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = retrySon->CheckRetryInfo(context);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = retrySon->Handle(&context);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ChangeLinkInfo changeLinkInfo;
    ret = retrySon->SetOpChangeLinkInfo(h2dPtr, opCmd, changeLinkInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RetryTest, ut_hrtNotifyReset)
{
    rtNotify_t notify;
    HcclResult ret = hrtNotifyCreate(0, &notify);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = hrtNotifyReset(notify);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RetryTest, ut_Init_Agent)
{
    MOCKER_CPP(&OpRetryBase::Send).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::Recv).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::GetOpExecInfo).stubs().with(mockcpp::any()).will(returnValue(0));
    u32 rankId = 0;
    u32 rankSize = 1;
    HcclCommConnections commConnect;
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
    agentParam.opStreamPtr =  std::make_shared<HcclOpStreamRes>(myMap);
    s32 deviceLogicId = 0;
    HcclIpAddress deviceIP = HcclIpAddress("10.21.78.208");
    agentParam.agentInfo = {rankId, deviceLogicId, localIp, deviceIP};
    agentParam.group = "test_group";
    agentParam.agentConnection = std::make_shared<HcclSocket>("st_Init_Agent",
        nullptr, localIp, 16666, HcclSocketRole::SOCKET_ROLE_SERVER);
    agentParam.isEnableBackupLink = false;
    agentParam.notifyResetCallback = notifyResetCallback;
    agentParam.setTransportStatusCallback = setTransportStatusCallback;
    agentParam.getSwitchRanksCallback = getSwitchRanksCallback;
    std::shared_ptr<OpRetryAgentRunning> retryAgentRunning = std::make_shared<OpRetryAgentRunning>();

    RetryContext context(agentParam, retryAgentRunning);
    std::shared_ptr<RetryContext> retryCtx = std::make_shared<RetryContext>(context);
    HcclRtContext rtCtx_;
    OpRetryServerInfo serverInfo = {localIp, 16666, 0};
    RetryCommandInfo context1comd;
    context1comd.command = RETRY_CMD_RUNNING;
    MOCKER_CPP(&OpRetryBase::WaitCommandWithOpId).stubs().with(mockcpp::any(), outBound(context1comd)).will(returnValue(0));
	OpRetryManager opRetryManager;
    opRetryManager.RegisterOpRetryMachine(agentParam, rankSize, commConnect.isRoot,
        commConnect.serverConnections, serverInfo);
    opRetryManager.UnRegisterOpRetryManager(agentParam.group);

    // 测试空值场景
    std::shared_ptr<HcclSocket> dummyAgent = nullptr;
    std::map<u32, std::shared_ptr<HcclSocket> > dummyServer;

    opRetryManager.RegisterOpRetryMachine(agentParam, rankSize, commConnect.isRoot,
        dummyServer, serverInfo);
}

TEST_F(RetryTest, ut_NotifyResetCallBack)
{
    HcclCommunicator communication;

    communication.notifyPool_.reset(new (std::nothrow) NotifyPool());
    communication.notifyPool_->Init(0);

    communication.queueNotifyManagerRefac_.reset(new (std::nothrow) QueueNotifyManager());
    communication.queueNotifyManagerRefac_->Init();
    
    HcclResult ret = communication.ResetNotify();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RetryTest, ut_NotifyResetPool)
{
    HcclCommunicator communication;

    communication.notifyPool_.reset(new (std::nothrow) NotifyPool());
    communication.notifyPool_->Init(0);

    communication.queueNotifyManagerRefac_.reset(new (std::nothrow) QueueNotifyManager());
    communication.queueNotifyManagerRefac_->Init();
    
    HcclResult ret = communication.ResetNotifyForDestRank(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = communication.ResetNotifyForDestRank(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // std::shared_ptr<LocalIpcNotify> localIpcNotify = std::make_shared<LocalIpcNotify>();
    // NotifyPoolIPCSub notifyPoolIPCSub;
    // notifyPoolIPCSub.push_back(localIpcNotify);
    // communication.notifyPool_->pimpl_->notifyPoolDeivceIPCAsignedMap_.insert({0, notifyPoolIPCSub});
    // ret = communication.ResetNotifyForDestRank(0);
    // EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RetryTest, ut_PrintAgentInfoAfterFail)
{
    HcclAgentRetryInfo agent1;
    HcclAgentRetryInfo agent2;
    agent2.retryInfo.opInfo.execStatus.kfcStatus = KfcStatus::kEnd;
    std::map<u32, HcclAgentRetryInfo> serverSockets;
    serverSockets.insert({0, agent1});
    serverSockets.insert({1, agent2});
    std::set<u32> recvVaild;
    recvVaild.insert(1);

    std::shared_ptr<OpRetryServerWaitResp> retryServerWaitResp = std::make_shared<OpRetryServerWaitResp>();
    retryServerWaitResp->PrintAgentInfoAfterFail(serverSockets, recvVaild, agent1);
}

TEST_F(RetryTest, ut_retry_Server_SetNeedRetryServerRank_RDMA_Err) {
    MOCKER_CPP(&OpRetryBase::Send).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::Recv).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER(hrtCtxSetCurrent).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HCCL_SUCCESS;
    HcclIpAddress localIp = HcclIpAddress("192.168.100.110");
    u32 rankId = 0;
    HcclOpStreamRes myMap;
    std::string key = "example_key";
    std::vector<Stream> slaves;
    for (int i = 0; i < 10; i++) {
        slaves.push_back(Stream(StreamType::STREAM_TYPE_OFFLINE));
    }
    myMap[key] = slaves;
    std::shared_ptr<HDCommunicate> h2dPtr;
    h2dPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl)));
    std::shared_ptr<HDCommunicate> d2hPtr;
    d2hPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_D2H, sizeof(KfcExecStatus)));
    std::shared_ptr<HcclOpStreamRes> opStream =  std::make_shared<HcclOpStreamRes>(myMap);

    std::shared_ptr<HcclSocket> ServerSocket1(new (std::nothrow)HcclSocket("Retryfunction1",
        nullptr, localIp, 16666, HcclSocketRole::SOCKET_ROLE_CLIENT));
    std::map<u32, std::shared_ptr<HcclSocket> > ServerSockets;
    ServerSockets.insert(std::make_pair(0, ServerSocket1));

    std::shared_ptr<OpRetryServerRunning> retryServerRunning = std::make_shared<OpRetryServerRunning>();
    HcclIpAddress deviceIP = HcclIpAddress("10.21.78.208");
	s32 deviceLogicId_1 = 0;
    OpRetryAgentInfo agentInfo = {rankId, deviceLogicId_1, localIp, deviceIP};
    RetryContext context(ServerSockets, retryServerRunning, agentInfo);

    HcclOpIdentifier Identify;
    Identify.index = rankId;
    KfcExecStatus status;
    status.opId = Identify;
    status.execStatus.kfcStatus = KfcStatus::kRuning;
    status.execStatus.kfcError = KfcError::kNone;
    RetryInfo retryinfo;
    retryinfo.rankId = 1;
    retryinfo.retryState = RETRY_STATE_SERVER_RUNNING;
    retryinfo.linkState = true;
    retryinfo.opInfo = status;
    HcclAgentRetryInfo info;
    info.socket = ServerSocket1;
    info.retryInfo = retryinfo;
    for (auto &mapinfo: context.serverSockets_) {
        mapinfo.second =  info;
    }
    ret = retryServerRunning->ProcessEvent(&context);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

HcclResult stub_GetQpnErr(Heartbeat *heartbeat, const std::string &identifier, 
    std::set<std::tuple<u32, u32, u32>> &qpErrSet)
{
    qpErrSet.insert(std::make_tuple(1, 12, 10));
    return HCCL_SUCCESS;
}

TEST_F(RetryTest, ut_retry_Agent_ParseRdmaErr_sendRecv)
{
    MOCKER_CPP(&OpRetryBase::Send).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::Recv).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HDCommunicate::Get).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HDCommunicate::Put).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&Heartbeat::GetQpnErr).stubs().with(mockcpp::any()).will(invoke(stub_GetQpnErr));

    MOCKER(hrtCtxSetCurrent).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HCCL_SUCCESS;
    HcclIpAddress localIp = HcclIpAddress("192.168.100.110");
    u32 rankId = 0;
    //OpRetryAgentRunning Agent状态机初始化
    RetryCommandInfo context1comd;
    context1comd.command = RETRY_CMD_STOP_AICPU;
    MOCKER_CPP(&OpRetryBase::WaitCommandWithOpId).stubs().with(mockcpp::any(), outBound(context1comd)).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::SetOpExecCmdWithOpId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    HcclCommConnections commConnect;
    OpRetryAgentParam agentParam;
    agentParam.h2dPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl)));
    agentParam.d2hPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_D2H, sizeof(KfcExecStatus)));
    HcclOpStreamRes myMap;
    std::string key = "example_key";
    std::vector<Stream> slaves;
    for (int i = 0; i < TEST_RANK_NUM; i++) {
        slaves.push_back(Stream(StreamType::STREAM_TYPE_ONLINE));
    }
    myMap[key] = slaves;
    agentParam.opStreamPtr =  std::make_shared<HcclOpStreamRes>(myMap);
    s32 deviceLogicId = 0;
    HcclIpAddress deviceIP = HcclIpAddress("10.21.78.208");
    agentParam.agentInfo = {rankId, deviceLogicId, localIp, deviceIP};
    agentParam.group = "group";
    agentParam.agentConnection = std::make_shared<HcclSocket>("st_retry_Agent_ParseRdmaErr_sendRecv",
        nullptr, localIp, 16666, HcclSocketRole::SOCKET_ROLE_SERVER);
    agentParam.isEnableBackupLink = false;
    agentParam.notifyResetCallback = notifyResetCallback;
    agentParam.setTransportStatusCallback = setTransportStatusCallback;
    agentParam.getSwitchRanksCallback = getSwitchRanksCallback;
    std::shared_ptr<OpRetryAgentRunning> opRetryAgentRunning = std::make_shared<OpRetryAgentRunning>();

    RetryContext context(agentParam, opRetryAgentRunning);
    opRetryAgentRunning->pollRcTimeout_ = std::chrono::seconds(0);
    ret = opRetryAgentRunning->ProcessEvent(&context);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // OpRetryAgentRunning ParseRdmaErr
    std::shared_ptr<OpRetryAgentRunning> agentRunning = std::make_shared<OpRetryAgentRunning>();
    RetryContext agentCtx(agentParam, agentRunning);
    RetryState nextState = RETRY_STATE_RESERVED;


    // 验证 sendRecv 分支
    agentCtx.localRetryInfo_.opInfo.execStatus.kfcError = KfcError::kRdma;
    agentCtx.localRetryInfo_.opInfo.opId.isSendRecv = true;
    agentCtx.localRetryInfo_.opInfo.opId.srcRank = 0;
    agentCtx.localRetryInfo_.opInfo.opId.detRank = 1;
    agentRunning->ParseRdmaErr(&agentCtx, nextState);
    // 验证 !sendRecv 分支
    agentCtx.localRetryInfo_.opInfo.opId.isSendRecv = false;
    agentRunning->ParseRdmaErr(&agentCtx, nextState);

    GlobalMockObject::verify();
}

HcclResult stub_GetQpnErr1(Heartbeat *heartbeat, const std::string &identifier, 
    std::set<std::tuple<u32, u32, u32>> &qpErrSet)
{
    qpErrSet.insert(std::make_tuple(2, 11, 11));
    return HCCL_SUCCESS;
}

TEST_F(RetryTest, ut_retry_Agent_ParseRdmaErr1)
{
    MOCKER_CPP(&OpRetryBase::Send).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::Recv).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HDCommunicate::Get).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HDCommunicate::Put).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&Heartbeat::GetQpnErr).stubs().with(mockcpp::any()).will(invoke(stub_GetQpnErr1));

    MOCKER(hrtCtxSetCurrent).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HCCL_SUCCESS;
    //OpRetryAgentRunning Agent状态机初始化
    RetryCommandInfo context1comd;
    context1comd.command = RETRY_CMD_STOP_AICPU;
    MOCKER_CPP(&OpRetryBase::WaitCommandWithOpId).stubs().with(mockcpp::any(), outBound(context1comd)).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::SetOpExecCmdWithOpId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    OpRetryAgentParam agentParam;
    u32 rankId = 0;
    s32 deviceLogicId = 0;
    HcclIpAddress localIp = HcclIpAddress("192.168.100.110");
    HcclIpAddress deviceIP = HcclIpAddress("10.21.78.208");
    agentParam.h2dPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl)));
    agentParam.d2hPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_D2H, sizeof(KfcExecStatus)));
    HcclOpStreamRes myMap;
    std::string key = "example_key";
    std::vector<Stream> slaves;
    for (int i = 0; i < TEST_RANK_NUM; i++) {
        slaves.push_back(Stream(StreamType::STREAM_TYPE_ONLINE));
    }
    myMap[key] = slaves;
    agentParam.opStreamPtr =  std::make_shared<HcclOpStreamRes>(myMap);
    agentParam.agentInfo = {rankId, deviceLogicId, localIp, deviceIP};
    agentParam.group = "group";
    agentParam.agentConnection = std::make_shared<HcclSocket>("st_retry_Agent_ParseRdmaErr1",
        nullptr, localIp, 16666, HcclSocketRole::SOCKET_ROLE_SERVER);
    agentParam.isEnableBackupLink = false;
    agentParam.notifyResetCallback = notifyResetCallback;
    agentParam.setTransportStatusCallback = setTransportStatusCallback;
    agentParam.getSwitchRanksCallback = getSwitchRanksCallback;
    std::shared_ptr<OpRetryAgentRunning> opRetryAgentRunning = std::make_shared<OpRetryAgentRunning>();

    RetryContext context(agentParam, opRetryAgentRunning);
    ret = opRetryAgentRunning->ProcessEvent(&context);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<OpRetryAgentRunning> agentRunning = std::make_shared<OpRetryAgentRunning>();
    RetryContext agentCtx(agentParam, agentRunning);
    RetryState nextState = RETRY_STATE_RESERVED;
    // sendRecv 没有找到dstRank
    agentCtx.localRetryInfo_.opInfo.execStatus.kfcError = KfcError::kRdma;
    agentCtx.localRetryInfo_.opInfo.opId.isSendRecv = true;
    agentCtx.localRetryInfo_.opInfo.opId.opType = HcclCMDType::HCCL_CMD_SEND;
    agentCtx.localRetryInfo_.opInfo.opId.srcRank = 0;
    agentCtx.localRetryInfo_.opInfo.opId.detRank = 1;
    agentRunning->ParseRdmaErr(&agentCtx, nextState);
    // !sendRecv 不支持重执行
    agentCtx.localRetryInfo_.opInfo.opId.isSendRecv = false;
    agentRunning->ParseRdmaErr(&agentCtx, nextState);

    GlobalMockObject::verify();
}
HcclResult stub_GetQpnErr2(Heartbeat *heartbeat, const std::string &identifier, 
    std::set<std::tuple<u32, u32, u32>> &qpErrSet)
{
    qpErrSet.insert(std::make_tuple(2, 11, 11));
    return HCCL_SUCCESS;
}

TEST_F(RetryTest, ut_retry_Agent_ParseRdmaErr2)
{
    MOCKER_CPP(&OpRetryBase::Send).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::Recv).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HDCommunicate::Get).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HDCommunicate::Put).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&Heartbeat::GetQpnErr).stubs().with(mockcpp::any()).will(invoke(stub_GetQpnErr2));

    MOCKER(hrtCtxSetCurrent).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtRaRdmaGetHandle).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaRdevGetPortStatus).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = HCCL_SUCCESS;

    //OpRetryAgentRunning Agent状态机初始化
    RetryCommandInfo context1comd;
    context1comd.command = RETRY_CMD_STOP_AICPU;
    MOCKER_CPP(&OpRetryBase::WaitCommandWithOpId).stubs().with(mockcpp::any(), outBound(context1comd)).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::SetOpExecCmdWithOpId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    OpRetryAgentParam agentParam;
    u32 rankId = 0;
    s32 deviceLogicId = 0;
    HcclIpAddress localIp = HcclIpAddress("192.168.100.110");
    HcclIpAddress deviceIP = HcclIpAddress("10.21.78.208");
    agentParam.h2dPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl)));
    agentParam.d2hPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_D2H, sizeof(KfcExecStatus)));
    HcclOpStreamRes myMap;
    std::string key = "example_key";
    std::vector<Stream> slaves;
    for (int i = 0; i < TEST_RANK_NUM; i++) {
        slaves.push_back(Stream(StreamType::STREAM_TYPE_ONLINE));
    }
    myMap[key] = slaves;
    agentParam.opStreamPtr =  std::make_shared<HcclOpStreamRes>(myMap);
    agentParam.agentInfo = {rankId, deviceLogicId, localIp, deviceIP};
    agentParam.group = "group";
    agentParam.agentConnection = std::make_shared<HcclSocket>("st_retry_Agent_ParseRdmaErr2",
        nullptr, localIp, 16666, HcclSocketRole::SOCKET_ROLE_SERVER);
    agentParam.isEnableBackupLink = false;
    agentParam.notifyResetCallback = notifyResetCallback;
    agentParam.setTransportStatusCallback = setTransportStatusCallback;
    agentParam.getSwitchRanksCallback = getSwitchRanksCallback;
    std::shared_ptr<OpRetryAgentRunning> opRetryAgentRunning = std::make_shared<OpRetryAgentRunning>();

    RetryContext context(agentParam, opRetryAgentRunning);
    ret = opRetryAgentRunning->ProcessEvent(&context);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<OpRetryAgentRunning> agentRunning = std::make_shared<OpRetryAgentRunning>();
    RetryContext agentCtx(agentParam, agentRunning);
    RetryState nextState = RETRY_STATE_RESERVED;

    agentCtx.localRetryInfo_.opInfo.execStatus.kfcError = KfcError::kRdma;
    agentCtx.localRetryInfo_.opInfo.opId.isSendRecv = true;
    agentCtx.localRetryInfo_.opInfo.opId.opType = HcclCMDType::HCCL_CMD_SEND;
    agentCtx.localRetryInfo_.opInfo.opId.srcRank = 0;
    agentCtx.localRetryInfo_.opInfo.opId.detRank = 2;
    agentRunning->ParseRdmaErr(&agentCtx, nextState);

    agentRunning->SetTransportStatusForStop(&agentCtx);

    CreateOpRetryServerByState(RETRY_STATE_CHECK_ALL_LINK, &agentCtx);
    CreateOpRetryServerByState(RETRY_STATE_CMD_CHECK_LINK, &agentCtx);

    agentCtx.localChangeLinkInfo_.remoteRankNum = 1;
    agentCtx.localChangeLinkInfo_.remoteRankList[0] = 0;
    agentCtx.localChangeLinkInfo_.isUseDefaultPort[0] = true;
    agentRunning->SetTransportStatusForResume(&agentCtx);

    GlobalMockObject::verify();
}
HcclResult stub_GetQpnErr3(Heartbeat *heartbeat, const std::string &identifier, 
    std::set<std::tuple<u32, u32, u32>> &qpErrSet)
{
    qpErrSet.insert(std::make_tuple(2, 12, 11));
    return HCCL_SUCCESS;
}
 
TEST_F(RetryTest, ut_retry_Agent_ParseRdmaErr_bsr_SEND)
{
    MOCKER_CPP(&OpRetryBase::Send).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::Recv).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HDCommunicate::Get).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HDCommunicate::Put).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&Heartbeat::GetQpnErr).stubs().with(mockcpp::any()).will(invoke(stub_GetQpnErr3));
 
    MOCKER(hrtCtxSetCurrent).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtRaRdmaGetHandle).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaRdevGetPortStatus).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
 
    HcclResult ret = HCCL_SUCCESS;
    HcclIpAddress localIp = HcclIpAddress("192.168.100.110");
    u32 rankId = 0;
    //OpRetryAgentRunning Agent状态机初始化
    RetryCommandInfo context1comd;
    context1comd.command = RETRY_CMD_STOP_AICPU;
    MOCKER_CPP(&OpRetryBase::WaitCommandWithOpId).stubs().with(mockcpp::any(), outBound(context1comd)).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::SetOpExecCmdWithOpId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    OpRetryAgentParam agentParam;
    agentParam.h2dPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl)));
    agentParam.d2hPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_D2H, sizeof(KfcExecStatus)));
    HcclOpStreamRes myMap;
    std::string key = "example_key";
    std::vector<Stream> slaves;
    for (int i = 0; i < TEST_RANK_NUM; i++) {
        slaves.push_back(Stream(StreamType::STREAM_TYPE_ONLINE));
    }
    myMap[key] = slaves;
    agentParam.opStreamPtr =  std::make_shared<HcclOpStreamRes>(myMap);
    s32 deviceLogicId = 0;
    HcclIpAddress deviceIP = HcclIpAddress("10.21.78.208");
    agentParam.agentInfo = {rankId, deviceLogicId, localIp, deviceIP};
    agentParam.group = "group";
    agentParam.agentConnection = std::make_shared<HcclSocket>("ut_retry_Agent_ParseRdmaErr2",
        nullptr, localIp, 0, HcclSocketRole::SOCKET_ROLE_SERVER);
    agentParam.isEnableBackupLink = false;
    agentParam.notifyResetCallback = notifyResetCallback;
    agentParam.setTransportStatusCallback = setTransportStatusCallback;
    agentParam.getSwitchRanksCallback = getSwitchRanksCallback;
    std::shared_ptr<OpRetryAgentRunning> opRetryAgentRunning = std::make_shared<OpRetryAgentRunning>();

    RetryContext context(agentParam, opRetryAgentRunning);
    ret = opRetryAgentRunning->ProcessEvent(&context);
    EXPECT_EQ(ret, HCCL_SUCCESS);
 
    std::shared_ptr<OpRetryAgentRunning> agentRunning = std::make_shared<OpRetryAgentRunning>();
    RetryContext agentCtx(agentParam, agentRunning);
    RetryState nextState = RETRY_STATE_RESERVED;
 
    agentCtx.localRetryInfo_.opInfo.execStatus.kfcError = KfcError::kRdma;
    agentCtx.localRetryInfo_.opInfo.opId.isSendRecv = true;
    agentCtx.localRetryInfo_.opInfo.opId.opType = HcclCMDType::HCCL_CMD_BATCH_SEND_RECV;
    agentCtx.localRetryInfo_.opInfo.opId.bsrInfo[HCCL_SEND].tpQpn = 11;
    agentCtx.localRetryInfo_.opInfo.opId.srcRank = 0;
    agentCtx.localRetryInfo_.opInfo.opId.detRank = 2;
    EXPECT_EQ(HCCL_SUCCESS, agentRunning->ParseRdmaErr(&agentCtx, nextState));
 
    GlobalMockObject::verify();
}
 
HcclResult stub_GetQpnErr5(Heartbeat *heartbeat, const std::string &identifier, 
    std::set<std::tuple<u32, u32, u32>> &qpErrSet)
{
    qpErrSet.insert(std::make_tuple(2, 12, 22));
    return HCCL_SUCCESS;
}
TEST_F(RetryTest, ut_retry_Agent_ParseRdmaErr_bsr_RECV)
{
    MOCKER_CPP(&OpRetryBase::Send).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::Recv).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HDCommunicate::Get).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HDCommunicate::Put).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&Heartbeat::GetQpnErr).stubs().with(mockcpp::any()).will(invoke(stub_GetQpnErr5));
 
    MOCKER(hrtCtxSetCurrent).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtRaRdmaGetHandle).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaRdevGetPortStatus).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
 
    HcclResult ret = HCCL_SUCCESS;
    HcclIpAddress localIp = HcclIpAddress("192.168.100.110");
    u32 rankId = 0;

    //OpRetryAgentRunning Agent状态机初始化
    RetryCommandInfo context1comd;
    context1comd.command = RETRY_CMD_STOP_AICPU;
    MOCKER_CPP(&OpRetryBase::WaitCommandWithOpId).stubs().with(mockcpp::any(), outBound(context1comd)).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::SetOpExecCmdWithOpId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    OpRetryAgentParam agentParam;
    agentParam.h2dPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl)));
    agentParam.d2hPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_D2H, sizeof(KfcExecStatus)));
    HcclOpStreamRes myMap;
    std::string key = "example_key";
    std::vector<Stream> slaves;
    for (int i = 0; i < TEST_RANK_NUM; i++) {
        slaves.push_back(Stream(StreamType::STREAM_TYPE_ONLINE));
    }
    myMap[key] = slaves;
    agentParam.opStreamPtr =  std::make_shared<HcclOpStreamRes>(myMap);
    s32 deviceLogicId = 0;
    HcclIpAddress deviceIP = HcclIpAddress("10.21.78.208");
    agentParam.agentInfo = {rankId, deviceLogicId, localIp, deviceIP};
    agentParam.group = "group";
    agentParam.agentConnection = std::make_shared<HcclSocket>("ut_retry_Agent_ParseRdmaErr2",
        nullptr, localIp, 0, HcclSocketRole::SOCKET_ROLE_SERVER);
    agentParam.isEnableBackupLink = false;
    agentParam.notifyResetCallback = notifyResetCallback;
    agentParam.setTransportStatusCallback = setTransportStatusCallback;
    agentParam.getSwitchRanksCallback = getSwitchRanksCallback;
    std::shared_ptr<OpRetryAgentRunning> opRetryAgentRunning = std::make_shared<OpRetryAgentRunning>();

    RetryContext context(agentParam, opRetryAgentRunning);
    ret = opRetryAgentRunning->ProcessEvent(&context);
    EXPECT_EQ(ret, HCCL_SUCCESS);
 
    std::shared_ptr<OpRetryAgentRunning> agentRunning = std::make_shared<OpRetryAgentRunning>();
    RetryContext agentCtx(agentParam, agentRunning);
    RetryState nextState = RETRY_STATE_RESERVED;
 
    agentCtx.localRetryInfo_.opInfo.execStatus.kfcError = KfcError::kRdma;
    agentCtx.localRetryInfo_.opInfo.opId.isSendRecv = true;
    agentCtx.localRetryInfo_.opInfo.opId.opType = HcclCMDType::HCCL_CMD_BATCH_SEND_RECV;
    agentCtx.localRetryInfo_.opInfo.opId.bsrInfo[HCCL_RECV].tpQpn = 22;
    agentCtx.localRetryInfo_.opInfo.opId.srcRank = 0;
    agentCtx.localRetryInfo_.opInfo.opId.detRank = 2;
    agentRunning->ParseRdmaErr(&agentCtx, nextState);
    EXPECT_EQ(HCCL_SUCCESS, agentRunning->ParseRdmaErr(&agentCtx, nextState));
 
    GlobalMockObject::verify();
}
 
TEST_F(RetryTest, ut_retry_Agent_ParseRdmaErr_bsr_remainSendErr)
{
    MOCKER_CPP(&OpRetryBase::Send).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::Recv).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HDCommunicate::Get).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HDCommunicate::Put).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&Heartbeat::GetQpnErr).stubs().with(mockcpp::any()).will(returnValue(0));
 
    MOCKER(hrtCtxSetCurrent).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtRaRdmaGetHandle).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaRdevGetPortStatus).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
 
    HcclResult ret = HCCL_SUCCESS;
    //OpRetryAgentRunning Agent状态机初始化
    RetryCommandInfo context1comd;
    context1comd.command = RETRY_CMD_STOP_AICPU;
    MOCKER_CPP(&OpRetryBase::WaitCommandWithOpId).stubs().with(mockcpp::any(), outBound(context1comd)).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::SetOpExecCmdWithOpId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    OpRetryAgentParam agentParam;
    u32 rankId = 0;
    s32 deviceLogicId = 0;
    HcclIpAddress localIp = HcclIpAddress("192.168.100.110");
    HcclIpAddress deviceIP = HcclIpAddress("10.21.78.208");
    agentParam.h2dPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl)));
    agentParam.d2hPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_D2H, sizeof(KfcExecStatus)));
    HcclOpStreamRes myMap;
    std::string key = "example_key";
    std::vector<Stream> slaves;
    for (int i = 0; i < TEST_RANK_NUM; i++) {
        slaves.push_back(Stream(StreamType::STREAM_TYPE_ONLINE));
    }
    myMap[key] = slaves;
    agentParam.opStreamPtr =  std::make_shared<HcclOpStreamRes>(myMap);
    agentParam.agentInfo = {rankId, deviceLogicId, localIp, deviceIP};
    agentParam.group = "group";
    agentParam.agentConnection = std::make_shared<HcclSocket>("ut_retry_Agent_ParseRdmaErr2",
        nullptr, localIp, 0, HcclSocketRole::SOCKET_ROLE_SERVER);
    agentParam.isEnableBackupLink = false;
    agentParam.notifyResetCallback = notifyResetCallback;
    agentParam.setTransportStatusCallback = setTransportStatusCallback;
    agentParam.getSwitchRanksCallback = getSwitchRanksCallback;
    std::shared_ptr<OpRetryAgentRunning> opRetryAgentRunning = std::make_shared<OpRetryAgentRunning>();

    RetryContext context(agentParam, opRetryAgentRunning);
    ret = opRetryAgentRunning->ProcessEvent(&context);
    EXPECT_EQ(ret, HCCL_SUCCESS);
 
    std::shared_ptr<OpRetryAgentRunning> agentRunning = std::make_shared<OpRetryAgentRunning>();
    RetryContext agentCtx(agentParam, agentRunning);
    RetryState nextState = RETRY_STATE_RESERVED;
 
    agentCtx.localRetryInfo_.opInfo.execStatus.kfcError = KfcError::kRdma;
    agentCtx.localRetryInfo_.opInfo.opId.isSendRecv = true;
    agentCtx.localRetryInfo_.opInfo.opId.opType = HcclCMDType::HCCL_CMD_BATCH_SEND_RECV;
    agentCtx.isBSRRdmaSendError_ = true;
    agentCtx.localRetryInfo_.opInfo.opId.srcRank = 0;
    agentCtx.localRetryInfo_.opInfo.opId.detRank = 2;
  
    agentCtx.isBSRRdmaSendError_ = true;
    agentCtx.isBSRRdmaRecvError_ = false;
    agentRunning->ParseRdmaErr(&agentCtx, nextState);
    EXPECT_EQ(HCCL_SUCCESS, agentRunning->ParseRdmaErr(&agentCtx, nextState));
 
    GlobalMockObject::verify();
}
 
TEST_F(RetryTest, ut_retry_Agent_ParseRdmaErr_bsr_remainRECVErr)
{
    MOCKER_CPP(&OpRetryBase::Send).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::Recv).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HDCommunicate::Get).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HDCommunicate::Put).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&Heartbeat::GetQpnErr).stubs().with(mockcpp::any()).will(returnValue(0));
 
    MOCKER(hrtCtxSetCurrent).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtRaRdmaGetHandle).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtRaRdevGetPortStatus).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
 
    HcclResult ret = HCCL_SUCCESS;
    HcclIpAddress localIp = HcclIpAddress("192.168.100.110");
    u32 rankId = 0;
    //OpRetryAgentRunning Agent状态机初始化
    RetryCommandInfo context1comd;
    context1comd.command = RETRY_CMD_STOP_AICPU;
    MOCKER_CPP(&OpRetryBase::WaitCommandWithOpId).stubs().with(mockcpp::any(), outBound(context1comd)).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::SetOpExecCmdWithOpId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    OpRetryAgentParam agentParam;
    agentParam.h2dPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl)));
    agentParam.d2hPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_D2H, sizeof(KfcExecStatus)));
    HcclOpStreamRes myMap;
    std::string key = "example_key";
    std::vector<Stream> slaves;
    for (int i = 0; i < TEST_RANK_NUM; i++) {
        slaves.push_back(Stream(StreamType::STREAM_TYPE_ONLINE));
    }
    myMap[key] = slaves;
    agentParam.opStreamPtr =  std::make_shared<HcclOpStreamRes>(myMap);
    s32 deviceLogicId = 0;
    HcclIpAddress deviceIP = HcclIpAddress("10.21.78.208");
    agentParam.agentInfo = {rankId, deviceLogicId, localIp, deviceIP};
    agentParam.group = "group";
    agentParam.agentConnection = std::make_shared<HcclSocket>("ut_retry_Agent_ParseRdmaErr2",
        nullptr, localIp, 0, HcclSocketRole::SOCKET_ROLE_SERVER);
    agentParam.isEnableBackupLink = false;
    agentParam.notifyResetCallback = notifyResetCallback;
    agentParam.setTransportStatusCallback = setTransportStatusCallback;
    agentParam.getSwitchRanksCallback = getSwitchRanksCallback;
    std::shared_ptr<OpRetryAgentRunning> opRetryAgentRunning = std::make_shared<OpRetryAgentRunning>();

    RetryContext context(agentParam, opRetryAgentRunning);
    ret = opRetryAgentRunning->ProcessEvent(&context);
    EXPECT_EQ(ret, HCCL_SUCCESS);
 
    std::shared_ptr<OpRetryAgentRunning> agentRunning = std::make_shared<OpRetryAgentRunning>();
    RetryContext agentCtx(agentParam, agentRunning);
    RetryState nextState = RETRY_STATE_RESERVED;
 
    agentCtx.localRetryInfo_.opInfo.execStatus.kfcError = KfcError::kRdma;
    agentCtx.localRetryInfo_.opInfo.opId.isSendRecv = true;
    agentCtx.localRetryInfo_.opInfo.opId.opType = HcclCMDType::HCCL_CMD_BATCH_SEND_RECV;
    agentCtx.localRetryInfo_.opInfo.opId.srcRank = 0;
    agentCtx.localRetryInfo_.opInfo.opId.detRank = 2;
 
    agentCtx.isBSRRdmaSendError_ = false;
    agentCtx.isBSRRdmaRecvError_ = true;
    agentRunning->ParseRdmaErr(&agentCtx, nextState);
    EXPECT_EQ(HCCL_SUCCESS, agentRunning->ParseRdmaErr(&agentCtx, nextState));
    GlobalMockObject::verify();
}

TEST_F(RetryTest, ut_SetTransportStatus_bsr)
{
    MOCKER_CPP(&Transport::Stop)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Transport::Resume)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Transport::DeInit)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&TransportManager::Alloc)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::IsEnableBackupLink)
    .stubs()
    .will(returnValue(true));
    
    HcclCommunicator communication;

    HcclOpIdentifier opId;
    std::string newTag = "Tag";
    memcpy_s(opId.newTag, sizeof(opId.newTag), newTag.c_str(), newTag.size());
    communication.newTagToTagMap_[newTag] = newTag;
    bool statusStop = true;
    opId.isSendRecv = false;
    communication.userRank_ = 1;
    opId.opType = HcclCMDType::HCCL_CMD_BATCH_SEND_RECV;
    opId.detRank = 0;
    opId.srcRank = 1;
    opId.bsrInfo[HCCL_SEND].srcRank = 1;
    opId.bsrInfo[HCCL_SEND].detRank = 0;
    opId.bsrInfo[HCCL_RECV].srcRank = 0;
    opId.bsrInfo[HCCL_RECV].detRank = 1;
    std::map<u32, bool> remoteRankPortMap;
    std::map<u32, bool> isChangeLinkMap;
    remoteRankPortMap.insert({0, true});
    isChangeLinkMap.insert({0, true});
    bool isChangeLinkFlag = false;

    struct TransportRequest transportRequest;
    transportRequest.isValid = true;
    transportRequest.localUserRank = 1;
    transportRequest.remoteUserRank = 0;
    struct SingleSubCommTransport singleSubCommTransport;
    singleSubCommTransport.transportRequests.push_back(transportRequest);
    std::shared_ptr<Transport> link = nullptr;
    auto type = TransportType::TRANS_TYPE_IBV_EXP;
    HcclDispatcher dispatcher;
    TransportPara para{};
    const std::unique_ptr<NotifyPool> notifyPool_;
    MachinePara machinePara;
    link.reset(new (std::nothrow) Transport(type, para, dispatcher, notifyPool_, machinePara));
    singleSubCommTransport.links.push_back(link);

    struct TransportRequest transportRequest1;
    transportRequest1.isValid = true;
    transportRequest1.localUserRank = 1;
    transportRequest1.remoteUserRank = 0;
    singleSubCommTransport.transportRequests.push_back(transportRequest1);
    std::shared_ptr<Transport> link1 = nullptr;
    type = TransportType::TRANS_TYPE_IBV_EXP;
    link1.reset(new (std::nothrow) Transport(type, para, dispatcher, notifyPool_, machinePara));
    singleSubCommTransport.links.push_back(link1);
    singleSubCommTransport.status = {TransportStatus::STOP, TransportStatus::STOP};
    singleSubCommTransport.userRank2subCommRank.insert({1, 1});

    hccl::AlgResourceResponse algRes;
    algRes.opTransportResponse.resize(CommPlane::COMM_LEVEL_RESERVED);
    algRes.opTransportResponse[CommPlane::COMM_COMBINE_ORDER] = {std::vector<SingleSubCommTransport> {singleSubCommTransport}};
    communication.resMap_.insert({newTag, algRes});

    auto ret = communication.SetTransportStatusImpl(algRes.opTransportResponse, true, opId, 0, remoteRankPortMap, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = communication.SetTransportStatusImplForChange(algRes.opTransportResponse, opId, 0, remoteRankPortMap, true, isChangeLinkMap, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}
TEST_F(RetryTest, ut_SetTransportStatus)
{
    MOCKER_CPP(&Transport::Stop)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Transport::Resume)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&TransportManager::Alloc)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    HcclCommunicator communication;

    MOCKER_CPP(&HcclCommunicator::IsEnableBackupLink)
    .stubs()
    .will(returnValue(true));

    HcclOpIdentifier opId;
    std::string newTag = "Tag";
    memcpy_s(opId.newTag, sizeof(opId.newTag), newTag.c_str(), newTag.size());
    communication.newTagToTagMap_[newTag] = newTag;
    bool statusStop = true;
    communication.userRank_ = 1;
    opId.isSendRecv = false;
    opId.detRank = 0;
    opId.srcRank = 1;
    std::map<u32, bool> remoteRankPortMap;
    std::map<u32, bool> isChangeLinkMap;
    bool isChangeLinkFlag = false;
    communication.SetTransportStatus(opId, statusStop, remoteRankPortMap, isChangeLinkMap, isChangeLinkFlag);

    struct TransportRequest transportRequest;
    transportRequest.localUserRank = 1;
    transportRequest.remoteUserRank = 0;
    struct SingleSubCommTransport singleSubCommTransport;
    singleSubCommTransport.transportRequests.push_back(transportRequest);
    std::shared_ptr<Transport> link = nullptr;
    auto type = TransportType::TRANS_TYPE_IBV_EXP;
    HcclDispatcher dispatcher;
    TransportPara para{};
    const std::unique_ptr<NotifyPool> notifyPool_;
    MachinePara machinePara;
    link.reset(new (std::nothrow) Transport(type, para, dispatcher, notifyPool_, machinePara));
    singleSubCommTransport.links.push_back(link);

    AlgResourceResponse algRes; 
    algRes.opTransportResponse = std::vector<LevelNSubCommTransport> {std::vector<SingleSubCommTransport> {singleSubCommTransport}};
    communication.resMap_.insert({newTag, algRes});
    communication.SetTransportStatus(opId, statusStop, remoteRankPortMap, isChangeLinkMap, isChangeLinkFlag);
    statusStop = false;
    communication.SetTransportStatus(opId, statusStop, remoteRankPortMap, isChangeLinkMap, isChangeLinkFlag);
    isChangeLinkFlag = true;
    communication.SetTransportStatus(opId, statusStop, remoteRankPortMap, isChangeLinkMap, isChangeLinkFlag);
}

TEST_F(RetryTest, ut_SetSignalTransport)
{
    MOCKER_CPP(&Transport::Stop)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Transport::Resume)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Transport::DeInit)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::IsEnableBackupLink)
    .stubs()
    .will(returnValue(true));
    
    HcclCommunicator communication;
    bool statusStop = true;
    struct TransportRequest transportRequest;
    transportRequest.localUserRank = 0;
    transportRequest.remoteUserRank = 1;
    struct SingleSubCommTransport singleSubCommTransport;
    singleSubCommTransport.transportRequests.push_back(transportRequest);
    std::shared_ptr<Transport> link = nullptr;
    auto type = TransportType::TRANS_TYPE_IBV_EXP;
    HcclDispatcher dispatcher;
    TransportPara para{};
    const std::unique_ptr<NotifyPool> notifyPool_;
    MachinePara machinePara;
    link.reset(new (std::nothrow) Transport(type, para, dispatcher, notifyPool_, machinePara));
    link->pimpl_->transportAttr_.linkType = LinkType::LINK_ROCE;
    singleSubCommTransport.links.push_back(link);
    singleSubCommTransport.status.push_back(TransportStatus::STOP);

    communication.SetSignalTransport(singleSubCommTransport, 0, true);
    communication.SetSignalTransport(singleSubCommTransport, 0, false);
}
HcclResult stub_ClearCqeErr(Heartbeat *heartbeat, const std::string &identifier, u32 remoteRank)
{
    return HCCL_SUCCESS;
}
HcclResult stub_BroadcastCqeErr(Heartbeat *heartbeat, const std::string &identifier)
{
    return HCCL_SUCCESS;
}

TEST_F(RetryTest, ut_retry_Agent_WaitCmdCanRetryCase)
{
    MOCKER(hrtCtxSetCurrent).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HCCL_SUCCESS;
    u32 rankId = 0;
    
    HcclIpAddress localIp = HcclIpAddress("192.168.100.110");
    OpRetryAgentParam agentParam;
    agentParam.h2dPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl)));
    agentParam.d2hPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_D2H, sizeof(KfcExecStatus)));
    HcclOpStreamRes myMap;
    std::string key = "example_key";
    std::vector<Stream> slaves;
    for (int i = 0; i < 2; i++) {
        slaves.push_back(Stream(StreamType::STREAM_TYPE_ONLINE));
    }
    myMap[key] = slaves;
    agentParam.opStreamPtr =  std::make_shared<HcclOpStreamRes>(myMap);
    s32 deviceLogicId = 0;
    HcclIpAddress deviceIP = HcclIpAddress("10.21.78.208");
    agentParam.agentInfo = {rankId, deviceLogicId, localIp, deviceIP};
    agentParam.group = "group";
    agentParam.agentConnection = std::make_shared<HcclSocket>("ut_retry_Agent_WaitCmdCanRetryCase",
        nullptr, localIp, 0, HcclSocketRole::SOCKET_ROLE_SERVER);
    agentParam.isEnableBackupLink = false;
    agentParam.notifyResetCallback = notifyResetCallback;
    agentParam.setTransportStatusCallback = setTransportStatusCallback;
    agentParam.getSwitchRanksCallback = getSwitchRanksCallback;
    std::shared_ptr<OpRetryAgentWaitCmd> retryAgentWaitCmd = std::make_shared<OpRetryAgentWaitCmd>();

    RetryContext context2(agentParam, retryAgentWaitCmd);
    RetryCommandInfo context1comd;
    context1comd.command = RETRY_CMD_CAN_RETRY;
    MOCKER_CPP(&OpRetryBase::WaitCommandWithOpId).stubs().with(mockcpp::any(), outBound(context1comd)).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::SetOpExecCmdWithOpId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Heartbeat::ClearCqeErr).stubs().with(mockcpp::any()).will(invoke(stub_ClearCqeErr));
    MOCKER_CPP(&Heartbeat::BroadcastCqeErr).stubs().with(mockcpp::any()).will(invoke(stub_BroadcastCqeErr));
    MOCKER_CPP(&OpRetryBase::InitChangeLinkInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    context2.isChangeLinkInfoInit_ = true;
    context2.localRetryInfo_.retryState = RETRY_STATE_WAIT_CMD_CAN_RETRY;
    context2.localRetryInfo_.rankId = 0;
    context2.localRetryInfo_.opInfo.execStatus.retryInfo.retryCount = 1;
    context2.localRetryInfo_.opInfo.opId.isSendRecv = true;
    context2.localRetryInfo_.opInfo.opId.opType = HcclCMDType::HCCL_CMD_RECEIVE;
    context2.localRetryInfo_.opInfo.opId.detRank = 0;
    context2.localRetryInfo_.opInfo.opId.srcRank = 1;
    context2.deviceLogicId_ = 0;
    ret = retryAgentWaitCmd->ProcessEvent(&context2);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(RetryTest, ut_retry_Agent_WaitCmdCanRetryCase_BSR) 
{
    MOCKER(hrtCtxSetCurrent).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HCCL_SUCCESS;

    //RetryAgentWaitCmd Agent状态机初始化
    OpRetryAgentParam agentParam;
    u32 rankId = 0;
    s32 deviceLogicId = 0;
    HcclIpAddress localIp = HcclIpAddress("192.168.100.110");
    HcclIpAddress deviceIP = HcclIpAddress("10.21.78.208");
    agentParam.h2dPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl)));
    agentParam.d2hPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_D2H, sizeof(KfcExecStatus)));
    HcclOpStreamRes myMap;
    std::string key = "example_key";
    std::vector<Stream> slaves;
    for (int i = 0; i < 2; i++) {
        slaves.push_back(Stream(StreamType::STREAM_TYPE_ONLINE));
    }
    myMap[key] = slaves;
    agentParam.opStreamPtr =  std::make_shared<HcclOpStreamRes>(myMap);
    agentParam.agentInfo = {rankId, deviceLogicId, localIp, deviceIP};
    agentParam.group = "group";
    agentParam.agentConnection = std::make_shared<HcclSocket>("ut_retry_Agent_WaitCmdCanRetryCase",
        nullptr, localIp, 0, HcclSocketRole::SOCKET_ROLE_SERVER);
    agentParam.isEnableBackupLink = false;
    agentParam.notifyResetCallback = notifyResetCallback;
    agentParam.setTransportStatusCallback = setTransportStatusCallback;
    agentParam.getSwitchRanksCallback = getSwitchRanksCallback;
    std::shared_ptr<OpRetryAgentWaitCmd> retryAgentWaitCmd = std::make_shared<OpRetryAgentWaitCmd>();

    RetryContext context2(agentParam, retryAgentWaitCmd);
    RetryCommandInfo context1comd;
    context1comd.command = RETRY_CMD_CAN_RETRY;
    MOCKER_CPP(&OpRetryBase::WaitCommandWithOpId).stubs().with(mockcpp::any(), outBound(context1comd)).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::SetOpExecCmdWithOpId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Heartbeat::ClearCqeErr).stubs().with(mockcpp::any()).will(invoke(stub_ClearCqeErr));
    MOCKER_CPP(&Heartbeat::BroadcastCqeErr).stubs().with(mockcpp::any()).will(invoke(stub_BroadcastCqeErr));
    MOCKER_CPP(&OpRetryBase::InitChangeLinkInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    context2.isChangeLinkInfoInit_ = true;
    context2.localRetryInfo_.retryState = RETRY_STATE_WAIT_CMD_CAN_RETRY;
    context2.localRetryInfo_.rankId = 0;
    context2.localRetryInfo_.opInfo.execStatus.retryInfo.retryCount = 1;
    context2.localRetryInfo_.opInfo.opId.isSendRecv = true;
    context2.localRetryInfo_.opInfo.opId.opType = HcclCMDType::HCCL_CMD_BATCH_SEND_RECV;
    context2.localRetryInfo_.opInfo.opId.detRank = 0;
    context2.localRetryInfo_.opInfo.opId.srcRank = 1;
    context2.deviceLogicId_ = 0;
    ret = retryAgentWaitCmd->ProcessEvent(&context2);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(RetryTest, ut_InitOpRetry)
{
    HcclCommunicator communication;
    communication.retryEnable_ = true;
    HcclIpAddress deviceIP = HcclIpAddress("10.21.78.208");
    communication.devIpAddr_.push_back(deviceIP);
 
    communication.InitOpRetry();
}

TEST_F(RetryTest, ut_retry_Agent_ProcessError)
{
    std::string group = "group";
    u32 rankId = 0;

    HcclIpAddress localIp = HcclIpAddress("192.168.100.110");
    HcclIpAddress deviceIP = HcclIpAddress("10.21.78.208");

    //OpRetryAgentRunning Agent状态机初始化
    OpRetryAgentParam agentParam;
    agentParam.h2dPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl)));
    agentParam.d2hPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_D2H, sizeof(KfcExecStatus)));
    HcclOpStreamRes myMap;
    std::string key = "example_key";
    std::vector<Stream> slaves;
    for (int i = 0; i < TEST_RANK_NUM; i++) {
        slaves.push_back(Stream(StreamType::STREAM_TYPE_ONLINE));
    }
    myMap[key] = slaves;
    agentParam.opStreamPtr =  std::make_shared<HcclOpStreamRes>(myMap);
    s32 deviceLogicId = 0;
    agentParam.agentInfo = {rankId, deviceLogicId, localIp, deviceIP};
    agentParam.group = "group";
    agentParam.agentConnection = std::make_shared<HcclSocket>("st_retry_Agent_ProcessError",
        nullptr, localIp, 16666, HcclSocketRole::SOCKET_ROLE_SERVER);
    agentParam.isEnableBackupLink = false;
    agentParam.notifyResetCallback = notifyResetCallback;
    agentParam.setTransportStatusCallback = setTransportStatusCallback;
    agentParam.getSwitchRanksCallback = getSwitchRanksCallback;
    std::shared_ptr<OpRetryAgentRunning> opRetryAgentRunning = std::make_shared<OpRetryAgentRunning>();

    RetryContext context(agentParam, opRetryAgentRunning);
    // 处理异常，状态迁移RETRY_STATE_RESP_RUNNING_ERR
    HcclResult ret = opRetryAgentRunning->ProcessError(&context);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RetryTest, ut_retry_Server_processError)
{
    u32 rankId = 0;
    s32 deviceLogicId = 0;
    HcclIpAddress localIp = HcclIpAddress("192.168.100.110");
    std::shared_ptr<HcclSocket> ServerSocket(new (std::nothrow)HcclSocket("st_retry_Server_processError",
        nullptr, localIp, 16666, HcclSocketRole::SOCKET_ROLE_CLIENT));
    std::map<u32, std::shared_ptr<HcclSocket> > ServerSockets;
    ServerSockets.insert(std::make_pair(0, ServerSocket));
    RetryInfo localRetryInfo;
    HcclIpAddress deviceIP = HcclIpAddress("10.21.78.208");
    OpRetryAgentInfo agentInfo = {rankId, deviceLogicId, localIp, deviceIP};

    //RetryServerRunning Server状态机初始化
    std::shared_ptr<OpRetryServerRunning> retryServerRunning = std::make_shared<OpRetryServerRunning>();
    RetryContext context(ServerSockets, retryServerRunning, agentInfo);

    // 处理异常，状态迁移RETRY_STATE_SERVER_RETRY_FAIL
    HcclResult ret = retryServerRunning->ProcessError(&context);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RetryTest, ut_retry_Server_OpName_Inconsistent)
{
    MOCKER_CPP(&OpRetryBase::Send).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::Recv).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER(hrtCtxSetCurrent).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HCCL_SUCCESS;
    HcclIpAddress localIp = HcclIpAddress("192.168.100.110");
    u32 rankId = 0;
    HcclOpStreamRes myMap;
    std::string key = "example_key";
    std::vector<Stream> slaves;
    for (int i = 0; i < 10; i++) {
        slaves.push_back(Stream(StreamType::STREAM_TYPE_OFFLINE));
    }
    myMap[key] = slaves;
    std::shared_ptr<HDCommunicate> h2dPtr;
    h2dPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl)));
    std::shared_ptr<HDCommunicate> d2hPtr;
    d2hPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_D2H, sizeof(KfcExecStatus)));
    std::shared_ptr<HcclOpStreamRes> opStream =  std::make_shared<HcclOpStreamRes>(myMap);

    std::shared_ptr<HcclSocket> ServerSocket1(new (std::nothrow)HcclSocket("RetryServerOpNameInconsistent",
        nullptr, localIp, 16666, HcclSocketRole::SOCKET_ROLE_CLIENT));
    std::map<u32, std::shared_ptr<HcclSocket> > ServerSockets;
    ServerSockets.insert(std::make_pair(0, ServerSocket1));
    HcclIpAddress deviceIP = HcclIpAddress("10.21.78.208");
    s32 deviceLogicId_1 = 0;
    OpRetryAgentInfo agentInfo = {rankId, deviceLogicId_1, localIp, deviceIP};
    HcclOpIdentifier Identify;
    Identify.index = rankId;
    KfcExecStatus status;
    status.opId = Identify;
    status.execStatus.kfcStatus = KfcStatus::kRuning;
    status.execStatus.kfcError = KfcError::kNone;
    RetryInfo retryinfo;
    retryinfo.rankId = 1;
    retryinfo.retryState = RETRY_STATE_SERVER_RUNNING;
    retryinfo.linkState = true;
    retryinfo.opInfo = status;
    retryinfo.isChangeLinkFlag = true;
    HcclAgentRetryInfo info;
    info.socket = ServerSocket1;
    info.retryInfo = retryinfo;

    // server 状态机校验算子不一致的场景
    std::shared_ptr<OpRetryServerCheckOp> retryServerCheckOp = std::make_shared<OpRetryServerCheckOp>();
    RetryContext context1(ServerSockets, retryServerCheckOp, agentInfo);
    context1.isNeedReportOpRetryErr = false;
    for(auto & mapinfo: context1.serverSockets_)
    {
       mapinfo.second =  info;
    }
    MOCKER_CPP(&OpRetryBase::CheckOpName).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_OPRETRY_FAIL));
    context1.needRetryServerRanks_.emplace_back(0);
    ret = retryServerCheckOp->ProcessEvent(&context1);
    EXPECT_EQ(context1.isNeedReportOpRetryErr, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    // server 状态机算子不一致进入重执行失败场景
    std::shared_ptr<OpRetryServerRetryFail> retryServerFail = std::make_shared<OpRetryServerRetryFail>();
    RetryContext context2(ServerSockets, retryServerFail, agentInfo);
    context2.isNeedReportOpRetryErr = true;
    for(auto & mapinfo: context2.serverSockets_)
    {
       mapinfo.second =  info;
    }
    ret = retryServerFail->ProcessEvent(&context2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RetryTest, ut_retry_Agent_OpName_Inconsistent)
{
    u32 rankId = 0;
    HcclIpAddress localIp = HcclIpAddress("192.168.100.110");
    // OpRetryAgentWaitCmdCanRetry Agent状态机接收算子不一致导致重执行失败后server下发的指令
    RetryState nextState = RETRY_STATE_WAIT_CMD_CAN_RETRY;
    OpRetryAgentParam agentParam;
    agentParam.h2dPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl)));
    agentParam.d2hPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_D2H, sizeof(KfcExecStatus)));
    HcclOpStreamRes myMap;
    std::string key = "example_key";
    std::vector<Stream> slaves;
    for (int i = 0; i < TEST_RANK_NUM; i++) {
        slaves.push_back(Stream(StreamType::STREAM_TYPE_ONLINE));
    }
    myMap[key] = slaves;
    agentParam.opStreamPtr =  std::make_shared<HcclOpStreamRes>(myMap);
    s32 deviceLogicId = 0;
    HcclIpAddress deviceIP = HcclIpAddress("10.21.78.208");
    agentParam.agentInfo = {rankId, deviceLogicId, localIp, deviceIP};
    agentParam.group = "group";
    agentParam.agentConnection = std::make_shared<HcclSocket>("st_retry_Agent_OpName_Inconsistentr",
        nullptr, localIp, 16666, HcclSocketRole::SOCKET_ROLE_SERVER);
    agentParam.isEnableBackupLink = false;
    agentParam.notifyResetCallback = notifyResetCallback;
    agentParam.setTransportStatusCallback = setTransportStatusCallback;
    agentParam.getSwitchRanksCallback = getSwitchRanksCallback;
    std::shared_ptr<OpRetryAgentWaitCmd> opRetryAgentWaitCmd = std::make_shared<OpRetryAgentWaitCmd>();

    RetryContext context1(agentParam, opRetryAgentWaitCmd);
    context1.localRetryInfo_.isNeedReportOpRetryErr = false;
    RetryCommandInfo context1comd;
    context1comd.command = RETRY_CMD_RETRY_CONSTRAINT_FAIL;
    MOCKER_CPP(&OpRetryBase::WaitCommandWithOpId).stubs().with(mockcpp::any(), outBound(context1comd)).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::SetOpExecCmdWithOpId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Heartbeat::ClearCqeErr).stubs().with(mockcpp::any()).will(invoke(stub_ClearCqeErr));
    MOCKER_CPP(&Heartbeat::BroadcastCqeErr).stubs().with(mockcpp::any()).will(invoke(stub_BroadcastCqeErr));
    MOCKER_CPP(&OpRetryBase::InitChangeLinkInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    context1.isChangeLinkInfoInit_ = true;
    context1.localRetryInfo_.retryState = RETRY_STATE_WAIT_CMD_CAN_RETRY;
    context1.localRetryInfo_.rankId = 0;
    context1.localRetryInfo_.opInfo.execStatus.retryInfo.retryCount = 1;
    context1.localRetryInfo_.opInfo.opId.isSendRecv = true;
    context1.localRetryInfo_.opInfo.opId.opType = HcclCMDType::HCCL_CMD_BATCH_SEND_RECV;
    context1.localRetryInfo_.opInfo.opId.detRank = 0;
    context1.localRetryInfo_.opInfo.opId.srcRank = 1;
    context1.deviceLogicId_ = 0;
    HcclResult ret = opRetryAgentWaitCmd->ProcessEvent(&context1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(context1.localRetryInfo_.isNeedReportOpRetryErr, true);
    GlobalMockObject::verify();

    //OpRetryAgentRetryFail Agent状态机进入重执行失败状态通知aicpu
    ret = HCCL_E_INTERNAL;
    std::shared_ptr<OpRetryAgentRetryFail> opRetryAgentRetryFail = std::make_shared<OpRetryAgentRetryFail>();
    RetryContext context2(agentParam, opRetryAgentRetryFail);
    context2.localRetryInfo_.isNeedReportOpRetryErr = true;
    MOCKER_CPP(&OpRetryBase::SetOpExecCmd).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = opRetryAgentRetryFail->ProcessEvent(&context2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RetryTest, ut_retry_Server_Inplace_Err)
{
    MOCKER_CPP(&OpRetryBase::Send).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::Recv).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER(hrtCtxSetCurrent).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HCCL_SUCCESS;
    HcclIpAddress localIp = HcclIpAddress("192.168.100.110");
    u32 rankId = 0;
    HcclOpStreamRes myMap;
    std::string key = "example_key";
    std::vector<Stream> slaves;
    for (int i = 0; i < 10; i++) {
        slaves.push_back(Stream(StreamType::STREAM_TYPE_OFFLINE));
    }
    myMap[key] = slaves;
    std::shared_ptr<HDCommunicate> h2dPtr;
    h2dPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl)));
    std::shared_ptr<HDCommunicate> d2hPtr;
    d2hPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_D2H, sizeof(KfcExecStatus)));
    std::shared_ptr<HcclOpStreamRes> opStream =  std::make_shared<HcclOpStreamRes>(myMap);

    std::shared_ptr<HcclSocket> ServerSocket1(new (std::nothrow)HcclSocket("RetryServerInplaceErr",
        nullptr, localIp, 16666, HcclSocketRole::SOCKET_ROLE_CLIENT));
    std::map<u32, std::shared_ptr<HcclSocket> > ServerSockets;
    ServerSockets.insert(std::make_pair(0, ServerSocket1));
    HcclIpAddress deviceIP = HcclIpAddress("10.21.78.208");
    s32 deviceLogicId_1 = 0;
    OpRetryAgentInfo agentInfo = {rankId, deviceLogicId_1, localIp, deviceIP};
    HcclOpIdentifier Identify;
    Identify.index = rankId;
    KfcExecStatus status;
    status.opId = Identify;
    status.execStatus.kfcStatus = KfcStatus::kRuning;
    status.execStatus.kfcError = KfcError::kNone;
    RetryInfo retryinfo;
    retryinfo.rankId = 1;
    retryinfo.retryState = RETRY_STATE_SERVER_RUNNING;
    retryinfo.linkState = true;
    retryinfo.opInfo = status;
    retryinfo.isChangeLinkFlag = true;
    HcclAgentRetryInfo info;
    info.socket = ServerSocket1;
    info.retryInfo = retryinfo;

    //RetryServerWaitResp 
    MOCKER_CPP(&OpRetryBase::WaitResponse).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    std::shared_ptr<OpRetryServerWaitResp> retryServerWaitResp = std::make_shared<OpRetryServerWaitResp>();
    RetryContext context1(ServerSockets, retryServerWaitResp, agentInfo);
    context1.SetRetryState(RETRY_STATE_WAIT_AICPU_STOPED, retryServerWaitResp);
    info.retryInfo.isNeedReportOpRetryErr = true;
    info.retryInfo.retryState = RETRY_STATE_RESP_RUNNING_ERR;
    for(auto & mapinfo: context1.serverSockets_)
    {
        mapinfo.second = info;
    }
    context1.needRetryServerRanks_.push_back(0);
    ret = retryServerWaitResp->ProcessEvent(&context1);
    EXPECT_EQ(context1.isNeedReportOpRetryErr, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RetryTest, ut_retry_agent_wait_resume_processEvent)
{
    HcclResult ret = HCCL_SUCCESS;

    OpRetryAgentParam agentParam;
    agentParam.h2dPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl)));
    agentParam.d2hPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_D2H, sizeof(KfcExecStatus)));
    HcclOpStreamRes myMap;
    std::string key = "example_key";
    std::vector<Stream> slaves;
    for (int i = 0; i < TEST_RANK_NUM; i++) {
        slaves.push_back(Stream(StreamType::STREAM_TYPE_ONLINE));
    }
    myMap[key] = slaves;
    agentParam.opStreamPtr =  std::make_shared<HcclOpStreamRes>(myMap);
    u32 rankId = 0;
    s32 deviceLogicId = 0;
    HcclIpAddress deviceIP = HcclIpAddress("10.21.78.208");
    HcclIpAddress localIp = HcclIpAddress("192.168.100.110");
    agentParam.agentInfo = {rankId, deviceLogicId, localIp, deviceIP};
    agentParam.group = "group";
    agentParam.agentConnection = std::make_shared<HcclSocket>("ut_retry_Agent_processEvent", nullptr, localIp,
        0, HcclSocketRole::SOCKET_ROLE_SERVER);
    agentParam.isEnableBackupLink = false;
    agentParam.notifyResetCallback = notifyResetCallback;
    agentParam.setTransportStatusCallback = setTransportStatusCallback;
    agentParam.getSwitchRanksCallback = getSwitchRanksCallback;

    std::shared_ptr<OpRetryAgentWaitResume> opRetryAgentWaitResume = std::make_shared<OpRetryAgentWaitResume>();
    RetryContext context(agentParam, opRetryAgentWaitResume);

    context.isAgentStateWaitResume_ = true;
    context.SetEnableSendRecv(false);

    RetryCommandInfo context1comd;
    context1comd.command = RESUME_CMD_RUNNING;
    MOCKER_CPP(&OpRetryBase::WaitCommandWithOpId)
        .stubs()
        .with(mockcpp::any(), outBound(context1comd))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&OpRetryBase::IssueResponse).stubs().will(returnValue(HCCL_E_TIMEOUT));
    ret = opRetryAgentWaitResume->ProcessEvent(&context);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    context1comd.command = RESUME_CMD_CHECK_LINK;
    MOCKER_CPP(&OpRetryBase::WaitCommandWithOpId)
        .stubs()
        .with(mockcpp::any(), outBound(context1comd))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&OpRetryBase::IssueResponse).stubs().will(returnValue(HCCL_E_TIMEOUT));
    ret = opRetryAgentWaitResume->ProcessEvent(&context);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    opRetryAgentWaitResume->keepTimeout_ = std::chrono::seconds(0);
    MOCKER_CPP(&OpRetryBase::WaitCommandWithOpId)
        .stubs()
        .with(mockcpp::any(), outBound(context1comd))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&OpRetryBase::GetRetryInfo).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&OpRetryBase::IssueResponse).stubs().will(returnValue(HCCL_E_TIMEOUT));
    ret = opRetryAgentWaitResume->ProcessEvent(&context);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    CreateOpRetryAgentByState(RETRY_STATE_AGENT_RUNNING, &context);
    
    std::shared_ptr<OpRetryAgentWaitCmd> opRetryAgentWaitCmd = std::make_shared<OpRetryAgentWaitCmd>();
    opRetryAgentWaitCmd->ProcessEvent(&context);

    std::shared_ptr<OpRetryAgentPollAicpuStop> opRetryAgentPollAicpuStop = std::make_shared<OpRetryAgentPollAicpuStop>();
    opRetryAgentPollAicpuStop->ProcessEvent(&context);

    std::shared_ptr<OpRetryAgentWaitChangeLinkInfo> opRetryAgentWaitChangeLinkInfo = std::make_shared<OpRetryAgentWaitChangeLinkInfo>();
    opRetryAgentWaitChangeLinkInfo->ProcessEvent(&context);

    std::shared_ptr<SwitchNicAgentWaitCmd> switchNicAgentWaitCmd = std::make_shared<SwitchNicAgentWaitCmd>();
    switchNicAgentWaitCmd->ProcessEvent(&context);

    context.isAgentStateWaitResume_ = false;
    ret = opRetryAgentWaitResume->ProcessEvent(&context);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    context1comd.command = RETRY_CMD_RETRY_FAIL;
    MOCKER_CPP(&OpRetryBase::WaitCommandWithOpId)
    .stubs()
    .with(mockcpp::any(), outBound(context1comd))
    .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&OpRetryBase::InitChangeLinkInfo).stubs().will(returnValue(HCCL_SUCCESS));
    ret = opRetryAgentWaitCmd->ProcessEvent(&context);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(RetryTest, ut_retry_server_wait_resume_processEvent)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclIpAddress localIp = HcclIpAddress("192.168.100.110");
    u32 rankId = 0;
    HcclOpStreamRes myMap;
    std::string key = "example_key";
    std::vector<Stream> slaves;
    for (int i = 0; i < 10; i++) {
        slaves.push_back(Stream(StreamType::STREAM_TYPE_OFFLINE));
    }
    myMap[key] = slaves;
    std::shared_ptr<HDCommunicate> h2dPtr;
    h2dPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl)));
    std::shared_ptr<HDCommunicate> d2hPtr;
    d2hPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_D2H, sizeof(KfcExecStatus)));
    std::shared_ptr<HcclOpStreamRes> opStream =  std::make_shared<HcclOpStreamRes>(myMap);

    std::shared_ptr<HcclSocket> ServerSocket1(new (std::nothrow)HcclSocket("Retryfunction1",
        nullptr, localIp, 0, HcclSocketRole::SOCKET_ROLE_CLIENT));
    std::map<u32, std::shared_ptr<HcclSocket> > ServerSockets;
    ServerSockets.insert(std::make_pair(0, ServerSocket1));
    HcclIpAddress deviceIP = HcclIpAddress("10.21.78.208");
    s32 deviceLogicId_1 = 0;
    OpRetryAgentInfo agentInfo = {rankId, deviceLogicId_1, localIp, deviceIP};

    // RetryServerRunning Server状态机初始化
    std::shared_ptr<OpRetryServerWaitResume> retryServerWaitResume = std::make_shared<OpRetryServerWaitResume>();
    RetryContext context(ServerSockets, retryServerWaitResume, agentInfo);

    context.isServerStateWaitResume_ = true;
    MOCKER_CPP(&OpRetryBase::WaitResponse).stubs().will(returnValue(HCCL_SUCCESS));
    ret = retryServerWaitResume->ProcessEvent(&context);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<SwitchNicServerCheckAllSwitchRanks> switchNicServerCheckAllSwitchRanks = std::make_shared<SwitchNicServerCheckAllSwitchRanks>();
    switchNicServerCheckAllSwitchRanks->ProcessEvent(&context);

    std::shared_ptr<OpRetryServerWaitLinkInfo> opRetryServerWaitLinkInfo = std::make_shared<OpRetryServerWaitLinkInfo>();
    opRetryServerWaitLinkInfo->ProcessEvent(&context);

    std::shared_ptr<OpRetryServerHandleError> opRetryServerHandleError = std::make_shared<OpRetryServerHandleError>();
    opRetryServerHandleError->ProcessEvent(&context);
    GlobalMockObject::verify();

    MOCKER_CPP(&OpRetryBase::WaitResponse).stubs().will(returnValue(HCCL_E_AGAIN));
    MOCKER_CPP(&OpRetryBase::IssueCommandWithOpId).stubs().will(returnValue(HCCL_SUCCESS));
    ret = retryServerWaitResume->ProcessEvent(&context);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    MOCKER_CPP(&OpRetryBase::WaitResponse).stubs().will(returnValue(HCCL_E_INTERNAL));
    ret = retryServerWaitResume->ProcessEvent(&context);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    context.isServerStateWaitResume_ = false;
    context.isRdmaError = false;
    MOCKER_CPP(&OpRetryBase::IssueCommandWithOpId).stubs().will(returnValue(HCCL_SUCCESS));
    ret = retryServerWaitResume->ProcessEvent(&context);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    context.isServerStateWaitResume_ = false;
    context.isRdmaError = true;
    MOCKER_CPP(&OpRetryBase::IssueCommandWithOpId).stubs().will(returnValue(HCCL_SUCCESS));
    ret = retryServerWaitResume->ProcessEvent(&context);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    context.isServerStateWaitResume_ = true;
    CreateOpRetryServerByState(RETRY_STATE_SERVER_RUNNING, &context);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    RetryCtrl retryCtcl;
    retryCtcl.retryCtx = std::make_shared<RetryContext>(context);
    
    OpRetryManager opRetryManager;
    bool isChangedLink = false;
    opRetryManager.agentOpRetry_["group1"] = std::move(retryCtcl);
    opRetryManager.agentOpRetry_["group1"].retryCtx->state_ = RETRY_STATE_AGENT_WAIT_RESUME;
    opRetryManager.SetRetryStateToWaitResume("group1", false);
    opRetryManager.agentOpRetry_["group1"].retryCtx->state_ = RETRY_STATE_AGENT_RUNNING;
    opRetryManager.ExitWaitResumeState("group1", false, false, isChangedLink);

    opRetryManager.agentOpRetry_["group1"].retryCtx->isOpRetryQuit = true;
    opRetryManager.SetRetryStateToWaitResume("group1", false);
    
    opRetryManager.serverOpRetry["group2"] = std::move(opRetryManager.agentOpRetry_["group1"]);
    opRetryManager.serverOpRetry["group2"].retryCtx->state_ = RETRY_STATE_SERVER_WAIT_RESUME;
    opRetryManager.SetRetryStateToWaitResume("group2", true);
    opRetryManager.serverOpRetry["group2"].retryCtx->state_ = RETRY_STATE_SERVER_RUNNING;
    opRetryManager.ExitWaitResumeState("group2", true, false, isChangedLink);
    opRetryManager.serverOpRetry["group2"].retryCtx->isOpRetryQuit = true;
    opRetryManager.SetRetryStateToWaitResume("group2", true);
}

TEST_F(RetryTest, ut_retry_opbase_switch_state)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclIpAddress localIp = HcclIpAddress("192.168.100.110");
    u32 rankId = 0;
    HcclOpStreamRes myMap;
    std::string key = "example_key";
    std::vector<Stream> slaves;
    for (int i = 0; i < 10; i++) {
        slaves.push_back(Stream(StreamType::STREAM_TYPE_OFFLINE));
    }
    myMap[key] = slaves;
    std::shared_ptr<HDCommunicate> h2dPtr;
    h2dPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl)));
    std::shared_ptr<HDCommunicate> d2hPtr;
    d2hPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_D2H, sizeof(KfcExecStatus)));
    std::shared_ptr<HcclOpStreamRes> opStream =  std::make_shared<HcclOpStreamRes>(myMap);

    std::shared_ptr<HcclSocket> ServerSocket1(new (std::nothrow)HcclSocket("Retryfunction1",
        nullptr, localIp, 0, HcclSocketRole::SOCKET_ROLE_CLIENT));
    std::map<u32, std::shared_ptr<HcclSocket> > ServerSockets;
    ServerSockets.insert(std::make_pair(0, ServerSocket1));
    HcclIpAddress deviceIP = HcclIpAddress("10.21.78.208");
    s32 deviceLogicId_1 = 0;
    OpRetryAgentInfo agentInfo = {rankId, deviceLogicId_1, localIp, deviceIP};

    // RetryServerRunning Server状态机初始化
    std::shared_ptr<OpRetryServerWaitResume> opRetryBase = std::make_shared<OpRetryServerWaitResume>();
    RetryContext context(ServerSockets, opRetryBase, agentInfo);
    MOCKER_CPP(&OpRetryBase::WaitResponse).stubs().will(returnValue(HCCL_SUCCESS));

    context.isRootRetryCtx_ = false;
    context.isAgentStateWaitResume_ = true;
    context.state_ = RETRY_STATE_SERVER_WAIT_RESUME;
    opRetryBase->Handle(&context);

    context.isRootRetryCtx_ = true;
    context.isServerStateWaitResume_ = true;
    context.state_ = RETRY_STATE_AGENT_WAIT_RESUME;
    opRetryBase->Handle(&context);
    
    context.ResetAgentState();
    context.IsRootRetryCtx();

    std::shared_ptr<OpRetryServerRetryFail> opRetryServerRetryFail = std::make_shared<OpRetryServerRetryFail>();
    RetryContext context1(ServerSockets, opRetryServerRetryFail, agentInfo);
    MOCKER_CPP(&OpRetryBase::IssueCommandWithOpId).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_INTERNAL));
    context1.needRetryServerRanks_.push_back(0);
    context1.isServerStateWaitResume_ = false;
    context1.isRootRetryCtx_ = true;
    context1.state_ = RETRY_STATE_SERVER_RETRY_FAIL;
    opRetryServerRetryFail->Handle(&context1);
    EXPECT_EQ(context1.isOpRetryQuit, true);
    GlobalMockObject::verify();
}

TEST_F(RetryTest, ut_retry_Agent_Resume_Check_Link)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclOpStreamRes myMap;
    OpRetryAgentParam agentParam;
    agentParam.h2dPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl)));
    agentParam.d2hPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_D2H, sizeof(KfcExecStatus)));
    std::string key = "example_key";
    std::vector<Stream> slaves;
    for (int i = 0; i < TEST_RANK_NUM; i++) {
        slaves.push_back(Stream(StreamType::STREAM_TYPE_ONLINE));
    }
    myMap[key] = slaves;
    agentParam.opStreamPtr =  std::make_shared<HcclOpStreamRes>(myMap);
    u32 rankId = 0;
    s32 deviceLogicId = 0;
    HcclIpAddress deviceIP = HcclIpAddress("10.21.78.208");
    HcclIpAddress localIp = HcclIpAddress("192.168.100.110");
    agentParam.agentInfo = {rankId, deviceLogicId, localIp, deviceIP};
    agentParam.group = "group";
    agentParam.agentConnection = std::make_shared<HcclSocket>("ut_retry_Agent_processEvent", nullptr, localIp,
        0, HcclSocketRole::SOCKET_ROLE_SERVER);
    agentParam.isEnableBackupLink = false;
    agentParam.notifyResetCallback = notifyResetCallback;
    agentParam.setTransportStatusCallback = setTransportStatusCallback;
    agentParam.getSwitchRanksCallback = getSwitchRanksCallback;

    MOCKER_CPP(&OpRetryBase::InitChangeLinkInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&OpRetryBase::GetLinkPortStatus).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&OpRetryBase::IssueLinkPortCheckResult).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&OpRetryBase::SetTransportStatusForStop).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    std::shared_ptr<ResumeAgentCheckLink> opRetryAgentResumeCheckLink = std::make_shared<ResumeAgentCheckLink>();
    RetryContext context1(agentParam, opRetryAgentResumeCheckLink);
    ret = opRetryAgentResumeCheckLink->ProcessEvent(&context1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    KfcExecStatus opInfo;
    RetryState nextState = RETRY_STATE_RESERVED;
    std::shared_ptr<ResumeAgentChangeLink> opRetryAgentResumeChangeLink = std::make_shared<ResumeAgentChangeLink>();
    RetryContext context2(agentParam, opRetryAgentResumeChangeLink);
    opInfo.execStatus.kfcStatus = KfcStatus::kResumeChanged;
    MOCKER_CPP(&OpRetryBase::GetOpExecInfo).stubs().with(mockcpp::any(), outBound(opInfo)).will(returnValue(0));
    MOCKER_CPP(&OpRetryBase::IssueResponse).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&OpRetryBase::SetOpExecCmd).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = opRetryAgentResumeChangeLink->WaitAndRespLinkChanged(&context2, nextState);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    RetryCommandInfo context3comd;
    context3comd.command = RETRY_CMD_RESUME_TRANSPORT;
    MOCKER_CPP(&OpRetryBase::WaitChangeLink).stubs().with(mockcpp::any()).will(invoke(stub_WaitChangeLink));
    MOCKER_CPP(&OpRetryBase::WaitCommandWithOpId).stubs().with(mockcpp::any(), outBound(context3comd)).will(returnValue(HCCL_SUCCESS));
    RetryContext context3(agentParam, opRetryAgentResumeChangeLink);
    context3.localChangeLinkInfo_.remoteRankNum = 1;
    ret = opRetryAgentResumeChangeLink->WaitResumeCmdResumeTransport(&context3);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    RetryContext context4(agentParam, opRetryAgentResumeChangeLink);
    context4.localChangeLinkInfo_.isChangeLinkFlag = true;
    MOCKER_CPP(&ResumeAgentChangeLink::WaitResumeCmdResumeTransport)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&OpRetryBase::SetTransportStatusForResume)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&OpRetryBase::SetOpChangeLinkInfo)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&ResumeAgentChangeLink::WaitAndRespLinkChanged)
        .stubs()
        .with(mockcpp::any(), outBound(nextState))
        .will(returnValue(HCCL_SUCCESS));
    ret = opRetryAgentResumeChangeLink->ProcessEvent(&context4);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(RetryTest, ut_retry_Server_Resume_Check_Link)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclIpAddress localIp = HcclIpAddress("192.168.100.110");
    u32 rankId = 0;
    HcclOpStreamRes myMap;
    std::string key = "example_key";
    std::vector<Stream> slaves;
    for (int i = 0; i < 10; i++) {
        slaves.push_back(Stream(StreamType::STREAM_TYPE_OFFLINE));
    }
    myMap[key] = slaves;
    std::shared_ptr<HDCommunicate> h2dPtr;
    h2dPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, sizeof(KfcExecControl)));
    std::shared_ptr<HDCommunicate> d2hPtr;
    d2hPtr.reset(new (std::nothrow) hccl::HDCommunicate(0, HCCL_HDC_TYPE_D2H, sizeof(KfcExecStatus)));
    std::shared_ptr<HcclOpStreamRes> opStream =  std::make_shared<HcclOpStreamRes>(myMap);

    std::shared_ptr<HcclSocket> ServerSocket1(new (std::nothrow)HcclSocket("Retryfunction1",
        nullptr, localIp, 0, HcclSocketRole::SOCKET_ROLE_CLIENT));
    std::map<u32, std::shared_ptr<HcclSocket> > ServerSockets;
    ServerSockets.insert(std::make_pair(0, ServerSocket1));
    HcclIpAddress deviceIP = HcclIpAddress("10.21.78.208");
    s32 deviceLogicId_1 = 0;
    OpRetryAgentInfo agentInfo = {rankId, deviceLogicId_1, localIp, deviceIP};

    // RetryServerRunning Server状态机初始化
    std::shared_ptr<ResumeServerCheckAllLink> retryServerResumeCheckAllLink = std::make_shared<ResumeServerCheckAllLink>();
    RetryContext context(ServerSockets, retryServerResumeCheckAllLink, agentInfo);

    MOCKER_CPP(&OpRetryBase::WaitLinkPortCheckResult).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = retryServerResumeCheckAllLink->WaitAgentCheckLinkResult(&context);

    RetryState nextState = RETRY_RESUME_STATE_SERVER_CHANGE_LINK;
    ret = retryServerResumeCheckAllLink->CheckAllLink(&context, nextState);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    context.serverSockets_[0].linkPortStatus.rankSize = 1;
    context.serverSockets_[0].linkPortStatus.rankList[0] = 0;
    context.serverSockets_[0].linkPortStatus.defaultPort = true;
    ret = retryServerResumeCheckAllLink->CheckAllLink(&context, nextState);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    context.serverSockets_[0].linkPortStatus.defaultPort = false;
    context.serverSockets_[0].linkPortStatus.defaultPort = true;
    ret = retryServerResumeCheckAllLink->CheckAllLink(&context, nextState);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    MOCKER_CPP(&ResumeServerCheckAllLink::WaitAgentCheckLinkResult).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&ResumeServerCheckAllLink::CheckAllLink).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = retryServerResumeCheckAllLink->ProcessEvent(&context);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<ResumeServerChangeLink> retryServerResumeChangeLink = std::make_shared<ResumeServerChangeLink>();
    RetryContext context1(ServerSockets, retryServerResumeChangeLink, agentInfo);
    MOCKER_CPP(&OpRetryBase::IssueChangeLink).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&OpRetryBase::IssueCommandWithOpId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = retryServerResumeChangeLink->CmdAgentChangeLink(&context1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    nextState = RETRY_STATE_SERVER_RUNNING;
    RetryInfo retryInfo;
    retryInfo.retryState = RETRY_STATE_AGENT_RUNNING;
    retryInfo.opInfo.execStatus.kfcStatus = KfcStatus::kResumeChanged;
    MOCKER_CPP(&OpRetryBase::WaitResponse).stubs().with(mockcpp::any(), outBound(retryInfo)).will(returnValue(HCCL_SUCCESS));
    ret = retryServerResumeChangeLink->WaitAllChangeLinkResult(&context1, nextState);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    retryInfo.retryState = RETRY_STATE_RESP_RUNNING_ERR;
    MOCKER_CPP(&OpRetryBase::WaitResponse).stubs().with(mockcpp::any(), outBound(retryInfo)).will(returnValue(HCCL_SUCCESS));
    ret = retryServerResumeChangeLink->WaitAllChangeLinkResult(&context1, nextState);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER_CPP(&ResumeServerChangeLink::CmdAgentChangeLink).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&ResumeServerChangeLink::WaitAllChangeLinkResult).stubs().with(mockcpp::any(), outBound(nextState)).will(returnValue(HCCL_SUCCESS));
    ret = retryServerResumeChangeLink->ProcessEvent(&context1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RetryTest, ut_SetTransportResumeStatus)
{
    MOCKER_CPP(&Transport::Stop)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Transport::Resume)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&TransportManager::Alloc)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    HcclCommunicator communication;

    MOCKER_CPP(&HcclCommunicator::IsEnableBackupLink)
    .stubs()
    .will(returnValue(true));

    HcclOpIdentifier opId;
    std::string newTag = "Tag";
    memcpy_s(opId.newTag, sizeof(opId.newTag), newTag.c_str(), newTag.size());
    communication.newTagToTagMap_[newTag] = newTag;
    bool statusStop = true;
    communication.userRank_ = 1;
    opId.isSendRecv = false;
    opId.detRank = 0;
    opId.srcRank = 1;
    std::map<u32, bool> remoteRankPortMap;
    std::map<u32, bool> isChangeLinkMap;
    bool isChangeLinkFlag = false;
    communication.SetTransportStatus(opId, statusStop, remoteRankPortMap, isChangeLinkMap, isChangeLinkFlag);

    struct TransportRequest transportRequest;
    transportRequest.localUserRank = 1;
    transportRequest.remoteUserRank = 0;
    struct SingleSubCommTransport singleSubCommTransport;
    singleSubCommTransport.transportRequests.push_back(transportRequest);
    std::shared_ptr<Transport> link = nullptr;
    auto type = TransportType::TRANS_TYPE_IBV_EXP;
    HcclDispatcher dispatcher;
    TransportPara para{};
    const std::unique_ptr<NotifyPool> notifyPool_;
    MachinePara machinePara;
    link.reset(new (std::nothrow) Transport(type, para, dispatcher, notifyPool_, machinePara));
    singleSubCommTransport.links.push_back(link);

    AlgResourceResponse algRes; 
    algRes.opTransportResponse = std::vector<LevelNSubCommTransport> {std::vector<SingleSubCommTransport> {singleSubCommTransport}};
    communication.resMap_.insert({newTag, algRes});
    communication.SetTransportResumeStatus(remoteRankPortMap, isChangeLinkMap, isChangeLinkFlag, statusStop);
    statusStop = false;
    communication.SetTransportResumeStatus(remoteRankPortMap, isChangeLinkMap, isChangeLinkFlag, statusStop);
    isChangeLinkFlag = true;
    communication.SetTransportResumeStatus(remoteRankPortMap, isChangeLinkMap, isChangeLinkFlag, statusStop);
}
#endif
TEST_F(RetryTest, ut_retry_Agent_Inplace_Err)
{
    u32 rankId = 0;
    HcclIpAddress localIp = HcclIpAddress("192.168.100.110");

    // retryAgentPollAicpuStop Agent状态机等待aicpu停止，超时后直接退出
    OpRetryAgentParam agentParam;
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
    agentParam.agentInfo = {rankId, deviceLogicId, localIp, deviceIP};
    agentParam.group = "group";
    agentParam.agentConnection = std::make_shared<HcclSocket>(
        "st_retry_Agent_OpName_Inconsistentr", nullptr, localIp, 16666, HcclSocketRole::SOCKET_ROLE_SERVER);
    agentParam.isEnableBackupLink = false;
    agentParam.notifyResetCallback = notifyResetCallback;
    agentParam.setTransportStatusCallback = setTransportStatusCallback;
    agentParam.getSwitchRanksCallback = getSwitchRanksCallback;

    std::shared_ptr<OpRetryAgentPollAicpuStop> retryAgentPollAicpuStop = std::make_shared<OpRetryAgentPollAicpuStop>();
    RetryContext context1(agentParam, retryAgentPollAicpuStop);
    KfcExecStatus opInfo;
    opInfo.execStatus.kfcError = KfcError::kExecConstraint;
    context1.state_ = RETRY_STATE_POLL_AICPU_STOPED;
    opInfo.execStatus.kfcStatus = KfcStatus::kRetryError;
    context1.SetRetryState(RETRY_STATE_POLL_AICPU_STOPED, retryAgentPollAicpuStop);
    context1.localRetryInfo_.isNeedReportOpRetryErr = false;
    MOCKER_CPP(&OpRetryBase::GetOpExecInfo).stubs().with(mockcpp::any(), outBound(opInfo)).will(returnValue(0));
    HcclResult ret = retryAgentPollAicpuStop->ProcessEvent(&context1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(context1.localRetryInfo_.isNeedReportOpRetryErr, true);
    GlobalMockObject::verify();
}

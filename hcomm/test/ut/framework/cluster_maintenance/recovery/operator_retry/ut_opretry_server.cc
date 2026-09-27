/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

TEST_F(RetryTest, ut_retry_ServerHandleError_NotSendRecv_DirectRetry)
{
    HcclIpAddress localIp = HcclIpAddress("192.168.100.110");
    std::shared_ptr<HcclSocket> ServerSocket1(
        new (std::nothrow) HcclSocket("RetryServer1", nullptr, localIp, 16666, HcclSocketRole::SOCKET_ROLE_CLIENT));

    std::map<u32, std::shared_ptr<HcclSocket>> ServerSockets;
    ServerSockets.insert(std::make_pair(1, ServerSocket1));

    HcclOpIdentifier opId1;
    strcpy_s((char*)opId1.tag, 128, "allreduce");
    opId1.index = 1;
    opId1.isSendRecv = false;

    HcclAgentRetryInfo info1;
    info1.retryInfo.opInfo.opId = opId1;
    info1.retryInfo.rankId = 1;
    info1.retryInfo.retryState = RETRY_STATE_SERVER_RUNNING;
    info1.retryInfo.linkState = true;

    HcclIpAddress deviceIP = HcclIpAddress("10.21.78.208");
    s32 deviceLogicId = 0;
    OpRetryAgentInfo agentInfo = {0, deviceLogicId, localIp, deviceIP};

    std::shared_ptr<OpRetryServerHandleError> retryServerHandleError = std::make_shared<OpRetryServerHandleError>();
    RetryContext context(ServerSockets, retryServerHandleError, agentInfo);

    context.errorRankList_.emplace(1, opId1);

    HcclResult ret = retryServerHandleError->ProcessEvent(&context);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(context.errorRankList_.size(), 0);
    EXPECT_GT(context.needRetryServerRanks_.size(), 0);

    GlobalMockObject::verify();
}

TEST_F(RetryTest, ut_retry_ServerCheckOp_When_CheckFail_Expect_RetryErrTrue)
{
    std::map<u32, std::shared_ptr<HcclSocket>> ServerSockets;
    HcclIpAddress ip = HcclIpAddress("10.21.78.208");
    s32 deviceLogicId = 0;
    OpRetryAgentInfo agentInfo = {0, deviceLogicId, ip, ip};
    std::shared_ptr<OpRetryServerCheckOp> retryServerCheckOp = std::make_shared<OpRetryServerCheckOp>();
    RetryContext context(ServerSockets, retryServerCheckOp, agentInfo);
    context.isNeedReportOpRetryErr = false;

    MOCKER_CPP(&OpRetryBase::CheckRetryInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_OPRETRY_FAIL));
    HcclResult ret = retryServerCheckOp->ProcessEvent(&context);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(context.isNeedReportOpRetryErr, true);

    GlobalMockObject::verify();
}

TEST_F(RetryTest, ut_retry_ResumeServer_WaitAgentCheckLinkResult_RecvSuccess_Log)
{
    HcclIpAddress localIp = HcclIpAddress("192.168.100.110");
    std::shared_ptr<HcclSocket> serverSocket1(
        new (std::nothrow) HcclSocket("RetryServer1", nullptr, localIp, 16666, HcclSocketRole::SOCKET_ROLE_CLIENT));

    std::map<u32, std::shared_ptr<HcclSocket>> serverSockets;
    serverSockets.insert(std::make_pair(1, serverSocket1));

    HcclIpAddress deviceIP = HcclIpAddress("10.21.78.208");
    s32 deviceLogicId = 0;
    OpRetryAgentInfo agentInfo = {0, deviceLogicId, localIp, deviceIP};

    std::shared_ptr<ResumeServerCheckAllLink> resumeServer = std::make_shared<ResumeServerCheckAllLink>();
    RetryContext context(serverSockets, resumeServer, agentInfo);
    context.group_ = "ut_test_group";

    MOCKER_CPP(&OpRetryBase::WaitLinkPortCheckResult)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    context.serverSockets_[1].linkPortStatus.rankList[0] = 7;

    HcclResult ret = resumeServer->WaitAgentCheckLinkResult(&context);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

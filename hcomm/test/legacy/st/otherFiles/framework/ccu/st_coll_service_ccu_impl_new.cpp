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
#define private public
#define protected public
#include "ccu_ctx.h"
#include "ccu_ins_group.h"
#include "ccu_ctx_mgr.h"
#include "ccu_ins_group.h"
#include "ccu_ins_preprocessor.h"
#include "ccu_registered_ctx_mgr.h"
#include "hierarchical_queue.h"
#include "ccu_communicator.h"
#include "ccu_device_manager.h"
#include "ccu_instruction_all_gather_mesh1d.h"
#include "ccu_context_all_gather_mesh1d.h"
#include "ccu_ctx_signature.h"
#include "communicator_impl.h"
#include "stream_manager.h"
#include "coll_alg_component.h"
#include "base_config_legacy.h"
#include "env_config_v2.h"
#include "not_support_exception.h"
#undef private
#undef protected

using namespace Hccl;

using namespace std;

class NewCollServiceCcuImplTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CommunicatorImplTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "CommunicatorImplTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        fakeSocket = new Socket(nullptr, localIp, 100, remoteIp, "test", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
        BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
        LinkData linkData(portType, 0, 1, 0, 1);
        CcuChannelInfo channelInfo;
        vector<CcuJetty*> ccuJettys;
        auto c = std::make_unique<CcuConnection>(
            linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);
        ccuTransport = new CcuTransport(fakeSocket, connection);
        std::cout << "A Test case in CommunicatorImplTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        delete fakeSocket;

        delete ccuTransport;
        std::cout << "A Test case in CommunicatorImplTest TearDown" << std::endl;
    }

    Socket* fakeSocket;
    IpAddress localIp;
    IpAddress remoteIp;

    Eid eid;
    RdmaHandle rdmaHandle;
    CcuConnection* connection;
    CcuTransport* ccuTransport;
};

TEST(NewCollServiceCcuImplTest, should_return_success_when_calling_init)
{
    // when
    MOCKER_CPP(&NewCollServiceCcuImpl::CollAlgComponentInit).stubs().will(ignoreReturnValue());

    // then
    CommunicatorImpl* comm;
    NewCollServiceCcuImpl collServiceCcuImpl(comm);

    // check
    EXPECT_NO_THROW(collServiceCcuImpl.Init());
}

TEST(NewCollServiceCcuImplTest, should_return_success_when_calling_LoadWithOpBasedMode)
{
    // when
    MOCKER_CPP(&CollServiceBase::RegisterOpbasedStream).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    shared_ptr<InsQueue> insQueue = make_shared<InsQueue>();
    MOCKER_CPP(&NewCollServiceCcuImpl::Orchestrate).stubs().with(mockcpp::any()).will(returnValue(insQueue));
    vector<LinkData> links;
    MOCKER_CPP(&NewCollServiceCcuImpl::GetCcuLinks).stubs().with(mockcpp::any()).will(returnValue(links));
    MOCKER_CPP(&SocketManager::BatchCreateSockets).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&CcuInsPreprocessor::Preprocess).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&OffloadStreamManager::SizeOfSlaves).stubs();
    MOCKER_CPP(&Interpreter::Submit).stubs();

    // then
    std::unique_ptr<CommunicatorImpl> comm = std::make_unique<CommunicatorImpl>();
    comm->socketManager = std::make_unique<SocketManager>(*comm, 0, 0, 60001);
    comm->streamManager = std::make_unique<StreamManager>(comm.get());
    comm->trace = std::make_unique<Trace>();
    comm->currentCollOperator = make_unique<CollOperator>();
    NewCollServiceCcuImpl collServiceCcuImpl(comm.get());
    CollOperator op;
    std::unique_ptr<Stream> stream;

    // check
    EXPECT_NO_THROW(collServiceCcuImpl.LoadWithOpBasedMode(op, nullptr));
}

class FakeCollAlgComponent : public CollAlgComponent {
public:
    FakeCollAlgComponent() : CollAlgComponent(nullptr, DevType::DEV_TYPE_950, 0, 1) {};
    HcclResult
    Orchestrate(const CollAlgOperator& op, const CollAlgParams& params, const string& algName, InsQuePtr queue)
    {
        return HCCL_SUCCESS;
    }

    HcclResult
    Orchestrate(const CollAlgOperator& op, const CollAlgParams& params, const string& algName, PrimQuePtr queue)
    {
        return HCCL_SUCCESS;
    }
};

TEST(NewCollServiceCcuImplTest, should_return_success_when_calling_Orchestrate)
{
    CommunicatorImpl* comm;
    NewCollServiceCcuImpl collServiceCcuImpl(comm);
    CollAlgOperator op;
    // collServiceCcuImpl.collAlgComponent = std::make_unique<FakeCollAlgComponent>();
    EXPECT_NO_THROW(collServiceCcuImpl.Orchestrate(op));
}

TEST(NewCollServiceCcuImplTest, should_return_success_when_calling_CollAlgComponentInit)
{
    std::unique_ptr<CommunicatorImpl> comm = std::make_unique<CommunicatorImpl>();
    NewCollServiceCcuImpl collServiceCcuImpl(comm.get());
    comm->rankGraph = std::make_unique<RankGraph>(0);

    EXPECT_NO_THROW(collServiceCcuImpl.CollAlgComponentInit());
}

TEST(NewCollServiceCcuImplTest, should_return_fail_when_calling_CollAlgComponentInit)
{
    GlobalMockObject::verify();
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    EnvTestConfig envTestConfig;
    EnvTestConfig& fakeEnvTestConfig = envTestConfig;
    fakeEnvTestConfig.testCase = CfgField<HcclDebugTestCase>(
        "CHIP_VERIFY_HCCL_TEST_CASE", HcclDebugTestCase::HCCL_INTRA_RANK_NOTIFY, CastTestCase);
    fakeEnvTestConfig.testCase.isParsed = true;
    MOCKER_CPP(&EnvConfig::GetTestConfig).stubs().will(returnValue(fakeEnvTestConfig));

    CommunicatorImpl* comm;
    NewCollServiceCcuImpl collServiceCcuImpl(comm);

    EXPECT_THROW(collServiceCcuImpl.CollAlgComponentInit(), NullPtrException);
}

TEST(NewCollServiceCcuImplTest, should_return_success_when_calling_GetCcuLinks)
{
    std::unique_ptr<CommunicatorImpl> comm = std::make_unique<CommunicatorImpl>();
    NewCollServiceCcuImpl collServiceCcuImpl(comm.get());
    auto insQueue = make_shared<InsQueue>();
    insQueue->Append(std::move(std::make_unique<CcuInstructionAllGatherMesh1D>()));
    insQueue->Append(std::move(std::make_unique<CcuInsGroup>()));
    auto subInsQueue = insQueue->Fork();
    subInsQueue->Append(std::move(std::make_unique<CcuInstructionAllGatherMesh1D>()));
    subInsQueue->Append(std::move(std::make_unique<CcuInsGroup>()));

    EXPECT_NO_THROW(collServiceCcuImpl.GetCcuLinks(insQueue));
}

TEST(NewCollServiceCcuImplTest, should_return_success_when_calling_getCcuTaskInfo)
{
    std::vector<CcuTaskParam> ccuTaskParam(1);
    ccuTaskParam[0].dieId = 0;
    ccuTaskParam[0].argSize = 13;
    MOCKER_CPP(&Mc2Compont::GetCcuTaskInfo).stubs().with(mockcpp::any()).will(returnValue(ccuTaskParam));
    std::unique_ptr<CommunicatorImpl> comm = std::make_unique<CommunicatorImpl>();
    NewCollServiceCcuImpl collServiceCcuImpl(comm.get());
    rtCcuTaskGroup_t group;

    EXPECT_NO_THROW(collServiceCcuImpl.GetCcuTaskInfo(nullptr, (void*)&group));
    EXPECT_EQ(1, group.taskNum);
}

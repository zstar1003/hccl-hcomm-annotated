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
#include "ccu_communicator.h"
#include "ccu_ctx.h"
#include "ccu_ins.h"
#include "ccu_ctx_mgr.h"
#include "ccu_ins_group.h"
#include "ccu_ins_preprocessor.h"
#include "ccu_registered_ctx_mgr.h"
#include "hierarchical_queue.h"
#include "ccu_ctx_factory.h"
#include "ccu_device_manager.h"
#include "ccu_instruction_all_gather_mesh1d.h"
#include "ccu_context_all_gather_mesh1d.h"
#include "ccu_ctx_signature.h"
#include "ccu_component.h"
#include "communicator_impl.h"
#include "port.h"
#undef private
#undef protected

using namespace Hccl;
using namespace std;

class CcuCtxFactoryTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CommunicatorImplTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "CommunicatorImplTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        fakeSocket = new Socket(nullptr, localIp, 100, remoteIp, "test", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
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

    RdmaHandle rdmaHandle;
};

class FakeCcuContextAlltoallMesh2D : public CcuContext {
public:
    FakeCcuContextAlltoallMesh2D(
        const CcuCtxArg& arg, const std::vector<CcuTransport*>& transports, const CcuTransportGroup& group)
        : CcuContext(arg, transports, group)
    {}
    ~FakeCcuContextAlltoallMesh2D() {}

    void Algorithm() override {}
    std::vector<uint64_t> GeneArgs(const CcuTaskArg& arg) override { return {}; }

private:
};

TEST_F(CcuCtxFactoryTest, should_return_success_when_calling_ccuinstregister)
{
    // when
    CcuCtxFactory::GetInstance().creators.clear();
    static CcuInstRegister<CcuContextAllGatherMesh1D> registrarAllGather(CcuInstType::CCU_ALLGATHER_MESH_1D_DIRECT);

    // check
    EXPECT_EQ(1, CcuCtxFactory::GetInstance().creators.size());
}

TEST_F(CcuCtxFactoryTest, should_return_success_when_calling_register)
{
    // check
    EXPECT_NO_THROW(
        CcuCtxFactory::GetInstance().Register<FakeCcuContextAlltoallMesh2D>(CcuInstType::CCU_ALLTOALL_MESH_2D_DIRECT));
    EXPECT_EQ(2, CcuCtxFactory::GetInstance().creators.size());
}

TEST_F(CcuCtxFactoryTest, should_return_success_when_calling_setmgr)
{
    // check
    CommunicatorImpl* comm;
    CcuTransportMgr transportMgr(*comm);
    CcuTransportGroupMgr transportGrpMgr(*comm);
    EXPECT_NO_THROW(CcuCtxFactory::GetInstance().SetTransportMgr(transportMgr));
    EXPECT_NO_THROW(CcuCtxFactory::GetInstance().SetTransportGrpMgr(transportGrpMgr));
}

TEST_F(CcuCtxFactoryTest, should_return_success_when_calling_register_create)
{
    // when
    GlobalMockObject::verify();
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(CcuDeviceManager::ReleaseCke)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(GenerateCcuCtxSignature).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    vector<CcuTransport*> transports;
    IpAddress localIp;
    IpAddress remoteIp;
    RdmaHandle rdmaHandle;
    Socket* fakeSocket
        = new Socket(nullptr, localIp, 100, remoteIp, "test", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    CcuConnection* CcuChannelInfo channelInfo;
    vector<CcuJetty*> ccuJettys;
    auto connection
        = std::make_unique<CcuConnection>(linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);
    connection->status = CcuConnStatus::INIT_COMPLETED;

    CcuTransport::CclBufferInfo locCclBufInfo;
    CcuTransport* ccuTransport = new CcuTransport(fakeSocket, std::move(connection), locCclBufInfo);
    SocketStatus fakeSocketStatus = SocketStatus::OK;
    MOCKER_CPP(&Socket::GetAsyncStatus).stubs().will(returnValue(fakeSocketStatus));

    transports.push_back(ccuTransport);
    std::unique_ptr<CcuTransportGroup> ccuTransportGrp = std::make_unique<CcuTransportGroup>(transports, 0);
    ccuTransportGrp->grpStatus = TransportGrpStatus::INIT;
    MOCKER_CPP(&CcuTransportMgr::PrepareCreate).stubs().with(mockcpp::any()).will(returnValue(ccuTransport));
    MOCKER_CPP(&CcuTransportGroupMgr::PrepareCreate)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(ccuTransportGrp.get()));

    // then
    std::unique_ptr<CcuInstructionAllGatherMesh1D> ccuIns = std::make_unique<CcuInstructionAllGatherMesh1D>();
    std::vector<LinkData> links;
    links.emplace_back(LinkData(portType, 0, 1, 0, 1));
    ccuIns->SetLinks(links);
    const CcuInstruction& ins = dynamic_cast<const CcuInstruction&>(*ccuIns);
    bool transportStatus = false;

    // check
    std::unique_ptr<CcuContext> contextPtr = CcuCtxFactory::GetInstance().Create(ins, transportStatus);
    EXPECT_NE(nullptr, contextPtr);

    delete fakeSocket;
    delete ccuTransport;
}

TEST_F(CcuCtxFactoryTest, should_return_nullptr_when_transport_mgr_prepare_create_fail)
{
    // when
    GlobalMockObject::verify();
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(CcuDeviceManager::ReleaseCke)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(GenerateCcuCtxSignature).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    vector<CcuTransport*> transports;
    IpAddress localIp;
    IpAddress remoteIp;
    RdmaHandle rdmaHandle;
    Socket* fakeSocket
        = new Socket(nullptr, localIp, 100, remoteIp, "test", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    CcuConnection* CcuChannelInfo channelInfo;
    vector<CcuJetty*> ccuJettys;
    auto connection
        = std::make_unique<CcuConnection>(linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);
    connection->status = CcuConnStatus::INIT_COMPLETED;

    CcuTransport::CclBufferInfo locCclBufInfo;
    CcuTransport* ccuTransport = new CcuTransport(fakeSocket, std::move(connection), locCclBufInfo);
    SocketStatus fakeSocketStatus = SocketStatus::OK;
    MOCKER_CPP(&Socket::GetAsyncStatus).stubs().will(returnValue(fakeSocketStatus));

    transports.push_back(ccuTransport);
    std::unique_ptr<CcuTransportGroup> ccuTransportGrp = std::make_unique<CcuTransportGroup>(transports, 0);
    ccuTransportGrp->grpStatus = TransportGrpStatus::INIT;
    MOCKER_CPP(&CcuTransportMgr::PrepareCreate).stubs().with(mockcpp::any()).will(returnValue(ccuTransport));
    CcuTransportGroup* fakePtr = nullptr;
    MOCKER_CPP(&CcuTransportGroupMgr::PrepareCreate)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(fakePtr));

    // then
    std::unique_ptr<CcuInstructionAllGatherMesh1D> ccuIns = std::make_unique<CcuInstructionAllGatherMesh1D>();
    std::vector<LinkData> links;
    links.emplace_back(LinkData(portType, 0, 1, 0, 1));
    ccuIns->SetLinks(links);
    const CcuInstruction& ins = dynamic_cast<const CcuInstruction&>(*ccuIns);
    bool transportStatus = false;

    // check
    std::unique_ptr<CcuContext> contextPtr = CcuCtxFactory::GetInstance().Create(ins, transportStatus);
    EXPECT_EQ(nullptr, contextPtr);

    delete fakeSocket;
    delete ccuTransport;
}

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
#include "op_type.h"
#include "virtual_topo.h"
#include "communicator_impl.h"
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
#include "ccu_context_utils.h"
#include "ins_exe_que.h"
#include "rdma_handle_manager.h"
#include "ccu_ctx_creator_registry.h"
#include "timeout_exception.h"
#undef private
#undef protected

using namespace Hccl;

using namespace std;

class CcuInsPreprocessorTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        GlobalMockObject::verify();
        std::cout << "CommunicatorImplTest SetUP" << std::endl;
    }

    static void TearDownTestCase()
    {
        GlobalMockObject::verify();
        std::cout << "CommunicatorImplTest TearDown" << std::endl;
    }

    virtual void SetUp()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CommunicatorImplTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CommunicatorImplTest TearDown" << std::endl;
    }
};

TEST_F(CcuInsPreprocessorTest, St_CreateCcuCtx_When_InterfaceOk_Expect_Return_Ok)
{
    CcuTransportGroup* group = (CcuTransportGroup*)0x12345678;
    MOCKER_CPP(&CcuTransportGroupMgr::PrepareCreate)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(group));
    MOCKER_CPP(&CcuJettyMgr::PrepareCreate).stubs().with(mockcpp::any()).will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(GenerateCcuCtxSignature).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    CommunicatorImpl* communicator;
    CcuInsPreprocessor preprocessor(communicator);
    CcuCtxCreatorRegistry::GetInstance().Register<CcuContextAllGatherMesh1D>(CcuInstType::CCU_ALLGATHER_MESH_1D_DIRECT);

    std::unique_ptr<CcuInstruction> ins1 = std::make_unique<CcuInstructionAllGatherMesh1D>();
    std::unique_ptr<CcuCtxGroup> ccuCtxGroup = std::make_unique<CcuCtxGroup>();
    bool transportStatus = true;

    EXPECT_NO_THROW(preprocessor.CreateCcuCtxGroup(*ins1, ccuCtxGroup, transportStatus));
    EXPECT_EQ(1, ccuCtxGroup->ctxs.size());
    EXPECT_EQ(true, transportStatus);

    CcuCtxCreatorRegistry::GetInstance().creators.clear();
}

TEST_F(CcuInsPreprocessorTest, St_CreateCcuCtx_When_CcuJettyMgrCreateResUnavailiable_Expect_Return_Nullptr)
{
    MOCKER_CPP(&CcuJettyMgr::PrepareCreate).stubs().with(mockcpp::any()).will(returnValue(HcclResult::HCCL_E_UNAVAIL));
    MOCKER(GenerateCcuCtxSignature).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    CommunicatorImpl* communicator;
    CcuInsPreprocessor preprocessor(communicator);

    std::unique_ptr<CcuInstruction> ins = std::make_unique<CcuInstructionAllGatherMesh1D>();
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    vector<LinkData> links;
    links.push_back(LinkData(portType, 0, 1, 0, 1));
    dynamic_cast<CcuInstructionAllGatherMesh1D*>(ins.get())->SetLinks(links); // 携带link，否则会跳过transport创建
    bool createStatus = true;
    EXPECT_EQ(preprocessor.CreateCcuCtx(*ins, createStatus), nullptr);
    EXPECT_EQ(createStatus, false);
}

TEST_F(CcuInsPreprocessorTest, St_CreateCcuCtx_When_CcuJettyMgrCreateResUnexpectedError_Expect_Return_Nullptr)
{
    MOCKER_CPP(&CcuJettyMgr::PrepareCreate).stubs().with(mockcpp::any()).will(returnValue(HcclResult::HCCL_E_PARA));
    MOCKER(GenerateCcuCtxSignature).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    CommunicatorImpl* communicator;
    CcuInsPreprocessor preprocessor(communicator);

    std::unique_ptr<CcuInstruction> ins = std::make_unique<CcuInstructionAllGatherMesh1D>();
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    vector<LinkData> links;
    links.push_back(LinkData(portType, 0, 1, 0, 1));
    dynamic_cast<CcuInstructionAllGatherMesh1D*>(ins.get())->SetLinks(links); // 携带link，否则会跳过transport创建
    bool createStatus = true;
    EXPECT_THROW(preprocessor.CreateCcuCtx(*ins, createStatus), InternalException);
    EXPECT_EQ(createStatus, false); // 未标记资源不足
}

TEST_F(CcuInsPreprocessorTest, St_CreateCcuCtx_When_CcuTransportMgrCreateResUnavailiable_Expect_Return_Nullptr)
{
    MOCKER_CPP(&CcuJettyMgr::PrepareCreate).stubs().with(mockcpp::any()).will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuTransportMgr::PrepareCreate)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HcclResult::HCCL_E_UNAVAIL));
    MOCKER(GenerateCcuCtxSignature).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    CommunicatorImpl* communicator;
    CcuInsPreprocessor preprocessor(communicator);

    std::unique_ptr<CcuInstruction> ins = std::make_unique<CcuInstructionAllGatherMesh1D>();
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    vector<LinkData> links;
    links.push_back(LinkData(portType, 0, 1, 0, 1));
    dynamic_cast<CcuInstructionAllGatherMesh1D*>(ins.get())->SetLinks(links); // 携带link，否则会跳过transport创建
    bool createStatus = true;
    EXPECT_EQ(preprocessor.CreateCcuCtx(*ins, createStatus), nullptr);
    EXPECT_EQ(createStatus, false);
}

TEST_F(CcuInsPreprocessorTest, St_CreateCcuCtx_When_CcuTransportMgrCreateResUnexpectedError_Expect_Return_Nullptr)
{
    MOCKER_CPP(&CcuJettyMgr::PrepareCreate).stubs().with(mockcpp::any()).will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuTransportMgr::PrepareCreate).stubs().with(mockcpp::any()).will(returnValue(HcclResult::HCCL_E_PARA));
    MOCKER(GenerateCcuCtxSignature).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    CommunicatorImpl* communicator;
    CcuInsPreprocessor preprocessor(communicator);

    std::unique_ptr<CcuInstruction> ins = std::make_unique<CcuInstructionAllGatherMesh1D>();
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    vector<LinkData> links;
    links.push_back(LinkData(portType, 0, 1, 0, 1));
    dynamic_cast<CcuInstructionAllGatherMesh1D*>(ins.get())->SetLinks(links); // 携带link，否则会跳过transport创建
    bool createStatus = true;
    EXPECT_THROW(preprocessor.CreateCcuCtx(*ins, createStatus), InternalException);
    EXPECT_EQ(createStatus, false); // 未标记资源不足
}

TEST_F(CcuInsPreprocessorTest, should_return_success_when_calling_preprocess)
{
    // when
    MOCKER_CPP(&CcuInsPreprocessor::PrepareCcuCtx)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(ignoreReturnValue());
    MOCKER_CPP(&CcuResPackMgr::PrepareAlloc).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&CcuInsPreprocessor::Confirm).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CcuInsPreprocessor::RegisterCtx).stubs().will(ignoreReturnValue());

    // then
    CommunicatorImpl* communicator;
    CcuInsPreprocessor preprocessor(communicator);
    preprocessor.needHandShake = true;
    preprocessor.resAllocSuccess = true;
    auto insQueue = make_shared<InsQueue>();

    // check
    EXPECT_NO_THROW(preprocessor.Preprocess(insQueue));
}

TEST_F(CcuInsPreprocessorTest, should_throw_when_calling_preprocess)
{
    // when
    MOCKER_CPP(&CcuInsPreprocessor::PrepareCcuCtx)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(ignoreReturnValue());
    MOCKER_CPP(&CcuResPackMgr::PrepareAlloc).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&CcuInsPreprocessor::Confirm).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CcuInsPreprocessor::RegisterCtx).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::AcceleratorFallback).stubs().will(returnValue(HCCL_SUCCESS));

    // then
    CommunicatorImpl communicator;
    CcuInsPreprocessor preprocessor(&communicator);
    preprocessor.needHandShake = true;
    preprocessor.resAllocSuccess = false;
    auto insQueue = make_shared<InsQueue>();

    // check
    EXPECT_NO_THROW(preprocessor.Preprocess(insQueue));
}

TEST_F(CcuInsPreprocessorTest, should_return_success_when_calling_prepareccuctx)
{
    // when
    MOCKER_CPP(&CcuInsPreprocessor::InsPreprocess)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(ignoreReturnValue());

    // then
    CommunicatorImpl* communicator;
    CcuInsPreprocessor preprocessor(communicator);
    auto insQueue = make_shared<InsQueue>();
    insQueue->Append(std::move(std::make_unique<CcuInstructionAllGatherMesh1D>()));
    insQueue->Append(std::move(std::make_unique<CcuInsGroup>()));
    auto subInsQueue = insQueue->Fork();
    subInsQueue->Append(std::move(std::make_unique<CcuInstructionAllGatherMesh1D>()));
    subInsQueue->Append(std::move(std::make_unique<CcuInsGroup>()));

    // check
    EXPECT_NO_THROW(preprocessor.PrepareCcuCtx(insQueue, false));
    EXPECT_EQ(1, insQueue->SizeOfSlaves());
    EXPECT_EQ(2, insQueue->Size());
    EXPECT_EQ(2, subInsQueue->Size());
}

TEST_F(CcuInsPreprocessorTest, should_bypass_inspreprocess_when_calling_prepareccuctx)
{
    // when
    MOCKER_CPP(&CcuInsPreprocessor::InsPreprocess)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(ignoreReturnValue());

    // then
    CommunicatorImpl* communicator;
    CcuInsPreprocessor preprocessor(communicator);
    auto insQueue = make_shared<InsQueue>();
    const DataSlice srcSlice;
    const DataSlice dstSlice;
    insQueue->Append(std::make_unique<InsLocalCopy>(srcSlice, dstSlice));
    insQueue->Append(std::make_unique<InsLocalCopy>(srcSlice, dstSlice));
    auto subInsQueue = insQueue->Fork();
    subInsQueue->Append(std::make_unique<InsLocalCopy>(srcSlice, dstSlice));
    subInsQueue->Append(std::make_unique<InsLocalCopy>(srcSlice, dstSlice));

    // check
    EXPECT_NO_THROW(preprocessor.PrepareCcuCtx(insQueue, false));
    EXPECT_EQ(1, insQueue->SizeOfSlaves());
    EXPECT_EQ(2, insQueue->Size());
    EXPECT_EQ(2, subInsQueue->Size());
    EXPECT_EQ(false, preprocessor.needHandShake);
    EXPECT_EQ(true, preprocessor.resAllocSuccess);
}

TEST_F(CcuInsPreprocessorTest, should_return_when_calling_prepareccuctx)
{
    // when
    MOCKER_CPP(&CcuInsPreprocessor::InsPreprocess)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(ignoreReturnValue());

    // then
    CommunicatorImpl* communicator;
    CcuInsPreprocessor preprocessor(communicator);
    auto insQueue = make_shared<InsQueue>();
    insQueue->Append(std::move(std::make_unique<CcuInstructionAllGatherMesh1D>()));
    insQueue->Append(std::move(std::make_unique<CcuInsGroup>()));
    auto subInsQueue = insQueue->Fork();
    subInsQueue->Append(std::move(std::make_unique<CcuInstructionAllGatherMesh1D>()));
    subInsQueue->Append(std::move(std::make_unique<CcuInsGroup>()));
    preprocessor.resAllocSuccess = false;

    // check
    EXPECT_NO_THROW(preprocessor.PrepareCcuCtx(insQueue, false));
    EXPECT_EQ(1, insQueue->SizeOfSlaves());
    EXPECT_EQ(2, insQueue->Size());
    EXPECT_EQ(2, subInsQueue->Size());
}

TEST_F(CcuInsPreprocessorTest, should_normal_process_when_calling_inspreprocess)
{
    // when
    GlobalMockObject::verify();
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    CcuCtxSignature ctxSignature;
    ctxSignature.Append("a");
    MOCKER(GenerateCcuCtxSignature)
        .stubs()
        .with(outBound(ctxSignature), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    CcuResPack ccuResPack;
    ccuResPack.handles.push_back(&ctxSignature);
    CcuCtxSignature ctxSignature2;
    ctxSignature2.Append("b");
    CcuResPack ccuResPack2;
    ccuResPack2.handles.push_back(&ctxSignature2);
    MOCKER_CPP(&CcuResPackMgr::GetCcuResPack)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(ccuResPack))
        .then(returnValue(ccuResPack2));
    MOCKER_CPP(&CcuTransportMgr::PrepareCreate)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(CcuCtxMgr::AllocRes)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuInsPreprocessor::CreateCcuCtxGroup)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&RegisteredCcuCtxMgr::HasRegistered)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(false))
        .then(returnValue(false))
        .then(returnValue(true));

    // then
    vector<unique_ptr<Instruction>> elements;
    elements.push_back(std::make_unique<CcuInstructionAllGatherMesh1D>());
    CcuInsPreprocessor::InsIterator iter = CcuInsPreprocessor::InsIterator(elements);
    CommunicatorImpl* comm;
    CcuInsPreprocessor preprocessor(comm);

    // check
    EXPECT_NO_THROW(preprocessor.InsPreprocess(iter, 0, false));
    EXPECT_EQ(1, preprocessor.ctxSignatures.size());
    EXPECT_EQ(1, preprocessor.ccuCtxGroups.size());
    EXPECT_EQ(1, preprocessor.resPackIdxs.size());
    EXPECT_EQ(1, preprocessor.insPtrs.size());
    EXPECT_EQ(ctxSignature, preprocessor.ctxSignatures[0]);
    EXPECT_EQ(true, preprocessor.needHandShake);
    EXPECT_EQ(true, preprocessor.resAllocSuccess);

    // then
    std::unique_ptr<CcuInsGroup> ccuInsGroup = std::make_unique<CcuInsGroup>();
    ccuInsGroup->Append(std::make_unique<CcuInstructionAllGatherMesh1D>());
    elements.push_back(std::move(ccuInsGroup));
    CcuInsPreprocessor::InsIterator iter2 = CcuInsPreprocessor::InsIterator(elements);
    iter2.Next();

    // check
    EXPECT_NO_THROW(preprocessor.InsPreprocess(iter2, 1, false));
    EXPECT_EQ(2, preprocessor.ctxSignatures.size());
    EXPECT_EQ(1, preprocessor.ccuCtxGroups.size());
    EXPECT_EQ(2, preprocessor.resPackIdxs.size());
    EXPECT_EQ(2, preprocessor.insPtrs.size());
    EXPECT_EQ(ctxSignature, preprocessor.ctxSignatures[1]);

    // when has registered
    vector<unique_ptr<Instruction>> elements2;
    elements2.push_back(std::make_unique<CcuInstructionAllGatherMesh1D>());
    CcuInsPreprocessor::InsIterator iter3 = CcuInsPreprocessor::InsIterator(elements);
    EXPECT_NO_THROW(preprocessor.InsPreprocess(iter3, 0, false));
    EXPECT_EQ(2, preprocessor.ctxSignatures.size());
    EXPECT_EQ(1, preprocessor.ccuCtxGroups.size());
    EXPECT_EQ(2, preprocessor.resPackIdxs.size());
    EXPECT_EQ(2, preprocessor.insPtrs.size());
}

TEST_F(CcuInsPreprocessorTest, should_transport_resalloc_fail_when_calling_inspreprocess)
{
    // when
    GlobalMockObject::verify();
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    CcuCtxSignature ctxSignature;
    ctxSignature.Append("a");
    MOCKER(GenerateCcuCtxSignature)
        .stubs()
        .with(outBound(ctxSignature), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    CcuResPack ccuResPack;
    ccuResPack.handles.push_back(&ctxSignature);
    MOCKER_CPP(&CcuResPackMgr::GetCcuResPack).stubs().with(mockcpp::any()).will(returnValue(ccuResPack));
    MOCKER_CPP(&RegisteredCcuCtxMgr::HasRegistered)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(false));
    MOCKER_CPP(&CcuTransportMgr::PrepareCreate)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(CcuCtxMgr::AllocRes)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    bool transportStatus = false;
    MOCKER_CPP(&CcuInsPreprocessor::CreateCcuCtxGroup)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(transportStatus))
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    // then
    vector<unique_ptr<Instruction>> elements;
    elements.push_back(std::make_unique<CcuInstructionAllGatherMesh1D>());
    CcuInsPreprocessor::InsIterator iter = CcuInsPreprocessor::InsIterator(elements);
    CommunicatorImpl* comm;
    CcuInsPreprocessor preprocessor(comm);

    // check
    EXPECT_NO_THROW(preprocessor.InsPreprocess(iter, 0, false));
    EXPECT_EQ(true, preprocessor.needHandShake);
    EXPECT_EQ(false, preprocessor.resAllocSuccess);
    EXPECT_EQ(1, preprocessor.ctxSignatures.size());
    EXPECT_EQ(1, preprocessor.ccuCtxGroups.size());
    EXPECT_EQ(1, preprocessor.insPtrs.size());
    EXPECT_EQ(ctxSignature, preprocessor.ctxSignatures[0]);
}

TEST_F(CcuInsPreprocessorTest, should_resalloc_fail_when_calling_inspreprocess)
{
    // when
    GlobalMockObject::verify();
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    CcuCtxSignature ctxSignature;
    ctxSignature.Append("a");
    MOCKER(GenerateCcuCtxSignature)
        .stubs()
        .with(outBound(ctxSignature), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    CcuResPack ccuResPack;
    ccuResPack.handles.push_back(&ctxSignature);
    MOCKER_CPP(&CcuResPackMgr::GetCcuResPack).stubs().with(mockcpp::any()).will(returnValue(ccuResPack));
    MOCKER_CPP(&RegisteredCcuCtxMgr::HasRegistered)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(false));
    MOCKER_CPP(&CcuTransportMgr::PrepareCreate)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(CcuCtxMgr::AllocRes)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_E_PARA));
    MOCKER_CPP(&CcuInsPreprocessor::CreateCcuCtxGroup)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    // then
    vector<unique_ptr<Instruction>> elements;
    elements.push_back(std::make_unique<CcuInstructionAllGatherMesh1D>());
    CcuInsPreprocessor::InsIterator iter = CcuInsPreprocessor::InsIterator(elements);
    CommunicatorImpl* comm;
    CcuInsPreprocessor preprocessor(comm);

    // check
    EXPECT_THROW(preprocessor.InsPreprocess(iter, 0, false), InternalException);
    EXPECT_EQ(true, preprocessor.needHandShake);
    EXPECT_EQ(true, preprocessor.resAllocSuccess);
    EXPECT_EQ(1, preprocessor.ctxSignatures.size());
    EXPECT_EQ(1, preprocessor.ccuCtxGroups.size());
    EXPECT_EQ(1, preprocessor.resPackIdxs.size());
    EXPECT_EQ(1, preprocessor.insPtrs.size());
    EXPECT_EQ(ctxSignature, preprocessor.ctxSignatures[0]);
}

TEST_F(CcuInsPreprocessorTest, should_no_throw_when_calling_transportsconnect)
{
    // when
    GlobalMockObject::verify();
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));
    MOCKER_CPP(&RdmaHandleManager::GetJfcHandle)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue((JfcHandle)0));
    MOCKER_CPP(&RdmaHandleManager::GetDieAndFuncId)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(std::pair<uint32_t, uint32_t>(0, 0)));
    vector<std::pair<CcuTransport*, LinkData>> transports;
    IpAddress localIp;
    IpAddress remoteIp;
    RdmaHandle rdmaHandle;
    shared_ptr<Socket> fakeSocket
        = make_shared<Socket>(nullptr, localIp, 100, remoteIp, "test", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    CcuChannelInfo channelInfo;
    vector<CcuJetty*> ccuJettys;
    auto c = std::make_unique<CcuConnection>(linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);
    CcuTransport::CclBufferInfo locCclBufInfo;
    CcuTransport* ccuTransport = new CcuTransport(fakeSocket.get(), std::move(c), locCclBufInfo);
    transports.push_back(make_pair(ccuTransport, linkData));
    MOCKER_CPP(&CcuTransportMgr::GetUnConfirmedTrans)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(transports));

    MOCKER_CPP(&CcuTransport::GetStatus)
        .stubs()
        .with()
        .will(returnValue((CcuTransport::TransStatus)CcuTransport::TransStatus::READY));
    MOCKER_CPP(&CcuTransport::SetHandshakeMsg).stubs().with(mockcpp::any()).will(ignoreReturnValue());

    // then
    std::unique_ptr<CommunicatorImpl> comm = std::make_unique<CommunicatorImpl>();
    comm->currentCollOperator = make_unique<CollOperator>();
    comm->currentCollOperator->opType = OpType::ALLREDUCE;
    CcuTransportMgr ccuTransportMgr(*comm, 0);

    // 该用例打桩会影响后续用例

    delete ccuTransport;
}

TEST_F(CcuInsPreprocessorTest, should_throw_when_calling_transportsconnect)
{
    // when
    GlobalMockObject::verify();
    vector<std::pair<CcuTransport*, LinkData>> transports;
    IpAddress localIp;
    IpAddress remoteIp;
    RdmaHandle rdmaHandle;
    Socket* fakeSocket
        = new Socket(nullptr, localIp, 100, remoteIp, "test", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    CcuChannelInfo channelInfo;
    vector<CcuJetty*> ccuJettys;
    auto c = std::make_unique<CcuConnection>(linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);
    CcuTransport::CclBufferInfo locCclBufInfo;
    CcuTransport* ccuTransport = new CcuTransport(fakeSocket, std::move(c), locCclBufInfo);
    transports.push_back(make_pair(ccuTransport, linkData));
    MOCKER_CPP(&CcuTransportMgr::GetUnConfirmedTrans)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(transports));

    MOCKER_CPP(&CcuTransport::GetStatus)
        .stubs()
        .with()
        .will(returnValue((CcuTransport::TransStatus)CcuTransport::TransStatus::CONNECT_FAILED));

    // then
    std::unique_ptr<CommunicatorImpl> comm = std::make_unique<CommunicatorImpl>();
    comm->currentCollOperator = make_unique<CollOperator>();
    comm->currentCollOperator->opType = OpType::ALLREDUCE;
    comm->opExecuteConfig.accState = AcceleratorState::AICPU_TS;
    CcuTransportMgr ccuTransportMgr(*comm, 0);

    // check
    EXPECT_THROW(ccuTransportMgr.TransportsConnect(), InternalException);

    delete fakeSocket;

    delete ccuTransport;
}

TEST_F(CcuInsPreprocessorTest, should_throw_when_calling_getstatus)
{
    // when
    GlobalMockObject::verify();
    vector<std::pair<CcuTransport*, LinkData>> transports;
    IpAddress localIp;
    IpAddress remoteIp;
    RdmaHandle rdmaHandle;
    Socket* fakeSocket
        = new Socket(nullptr, localIp, 100, remoteIp, "test", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    CcuChannelInfo channelInfo;
    vector<CcuJetty*> ccuJettys;
    auto c = std::make_unique<CcuConnection>(linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);
    CcuTransport::CclBufferInfo locCclBufInfo;
    CcuTransport* ccuTransport = new CcuTransport(fakeSocket, std::move(c), locCclBufInfo);
    transports.push_back(make_pair(ccuTransport, linkData));
    MOCKER_CPP(&CcuTransportMgr::GetUnConfirmedTrans)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(transports));
    MOCKER_CPP(&CcuTransport::GetStatus)
        .stubs()
        .with()
        .will(returnValue((CcuTransport::TransStatus)CcuTransport::TransStatus::CONNECT_FAILED));

    // then
    std::unique_ptr<CommunicatorImpl> comm = std::make_unique<CommunicatorImpl>();
    comm->currentCollOperator = make_unique<CollOperator>();
    comm->currentCollOperator->opType = OpType::ALLREDUCE;
    CcuTransportMgr ccuTransportMgr(*comm, 0);

    // check
    EXPECT_THROW(ccuTransportMgr.TransportsConnect(), InternalException);
    delete fakeSocket;

    delete ccuTransport;
}

TEST_F(CcuInsPreprocessorTest, should_no_throw_when_calling_registerctx)
{
    // when
    GlobalMockObject::verify();
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    CcuResPackMgr* ccuResPackMgr = new CcuResPackMgr();
    CcuResPack ccuResPack;
    MOCKER_CPP(&CcuCommunicator::GetCcuResPackMgr).stubs().with().will(returnValue(ccuResPackMgr));
    MOCKER_CPP(&CcuResPackMgr::GetCcuResPack).stubs().with(mockcpp::any()).will(returnValue(ccuResPack));
    MOCKER(InsExeQue::RegisterExtendInstruction)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    // then
    CommunicatorImpl* comm;
    CcuInsPreprocessor preprocessor(comm);
    preprocessor.resPackIdxs.push_back(0);
    CcuCtxSignature signature;
    u32 key = 0;
    preprocessor.ccuCtxGroups.emplace(signature, std::unordered_map<u32, std::unique_ptr<CcuCtxGroup>>());
    preprocessor.ccuCtxGroups[signature][key] = std::make_unique<CcuCtxGroup>();
    preprocessor.ctxSignatures.push_back(CcuCtxSignature());
    vector<unique_ptr<Instruction>> elements;
    elements.push_back(std::make_unique<CcuInstructionAllGatherMesh1D>());
    CcuInsPreprocessor::InsIterator iter = CcuInsPreprocessor::InsIterator(elements);
    preprocessor.insPtrs.push_back(iter);

    // check
    EXPECT_NO_THROW(preprocessor.RegisterCtx(false));

    delete ccuResPackMgr;
}

TEST_F(CcuInsPreprocessorTest, should_no_throw_when_calling_confirm)
{
    // when
    GlobalMockObject::verify();
    MOCKER_CPP(&CcuResPackMgr::Confirm).stubs().with().will(ignoreReturnValue());
    MOCKER_CPP(&CcuTransportMgr::Confirm).stubs().with().will(ignoreReturnValue());
    MOCKER_CPP(&CcuTransportGroupMgr::Confirm).stubs().with().will(ignoreReturnValue());
    MOCKER_CPP(&CcuJettyMgr::Confirm).stubs().with().will(ignoreReturnValue());

    // then
    CommunicatorImpl* communicator;
    CcuInsPreprocessor preprocessor(communicator);

    // check
    EXPECT_NO_THROW(preprocessor.Confirm());
}

TEST_F(CcuInsPreprocessorTest, should_no_throw_when_calling_fallback)
{
    // when
    GlobalMockObject::verify();
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(CcuCtxMgr::ReleaseRes)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuResPackMgr::Fallback).stubs().with().will(ignoreReturnValue());
    MOCKER_CPP(&CcuTransportMgr::Fallback).stubs().with().will(ignoreReturnValue());
    MOCKER_CPP(&CcuTransportGroupMgr::Fallback).stubs().with().will(ignoreReturnValue());
    MOCKER_CPP(&CcuJettyMgr::Fallback).stubs().with().will(ignoreReturnValue());

    // then
    CommunicatorImpl* comm;
    CcuInsPreprocessor preprocessor(comm);
    CcuCtxSignature signature;
    u32 key = 0;
    preprocessor.ccuCtxGroups.emplace(signature, std::unordered_map<u32, std::unique_ptr<CcuCtxGroup>>());
    preprocessor.ccuCtxGroups[signature][key] = std::make_unique<CcuCtxGroup>();

    // check
    EXPECT_NO_THROW(preprocessor.Fallback());
}

TEST_F(CcuInsPreprocessorTest, should_error_log_when_calling_fallback)
{
    // when
    GlobalMockObject::verify();
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER(CcuCtxMgr::ReleaseRes)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_E_PARA));
    MOCKER_CPP(&CcuResPackMgr::Fallback).stubs().with().will(ignoreReturnValue());
    MOCKER_CPP(&CcuTransportMgr::Fallback).stubs().with().will(ignoreReturnValue());
    MOCKER_CPP(&CcuTransportGroupMgr::Fallback).stubs().with().will(ignoreReturnValue());
    MOCKER_CPP(&CcuJettyMgr::Fallback).stubs().with().will(ignoreReturnValue());

    // then
    CommunicatorImpl* comm;
    CcuInsPreprocessor preprocessor(comm);
    CcuCtxSignature signature;
    u32 key = 0;
    preprocessor.ccuCtxGroups.emplace(signature, std::unordered_map<u32, std::unique_ptr<CcuCtxGroup>>());
    preprocessor.ccuCtxGroups[signature][key] = std::make_unique<CcuCtxGroup>();

    // check
    EXPECT_NO_THROW(preprocessor.Fallback());
}

TEST_F(CcuInsPreprocessorTest, should_when_calling_createccuctx)
{
    // when
    GlobalMockObject::verify();

    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));
    MOCKER_CPP(&RdmaHandleManager::GetJfcHandle)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue((JfcHandle)0));
    MOCKER_CPP(&RdmaHandleManager::GetDieAndFuncId)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(std::pair<uint32_t, uint32_t>(0, 0)));
    vector<CcuTransport*> transports;
    IpAddress localIp;
    IpAddress remoteIp;
    RdmaHandle rdmaHandle;
    shared_ptr<Socket> fakeSocket
        = make_shared<Socket>(nullptr, localIp, 100, remoteIp, "test", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    CcuChannelInfo channelInfo;
    vector<CcuJetty*> ccuJettys;
    auto c = std::make_unique<CcuConnection>(linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);
    CcuTransport::CclBufferInfo locCclBufInfo;
    CcuTransport* ccuTransport = new CcuTransport(fakeSocket.get(), std::move(c), locCclBufInfo);
    transports.push_back(ccuTransport);
    CcuTransportGroup* group = new CcuTransportGroup(transports, 1);
    MOCKER_CPP(&CcuTransportMgr::PrepareCreate).stubs().with(mockcpp::any()).will(returnValue(ccuTransport));
    MOCKER_CPP(&CcuTransportGroupMgr::PrepareCreate)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(group));
    MOCKER(GenerateCcuCtxSignature).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    CommunicatorImpl* comm;
    CcuInsPreprocessor preprocessor(comm);
    CcuCtxCreatorRegistry::GetInstance().Register<CcuContextAllGatherMesh1D>(CcuInstType::CCU_ALLGATHER_MESH_1D_DIRECT);

    // then
    std::unique_ptr<CcuInstruction> ins1 = std::make_unique<CcuInstructionAllGatherMesh1D>();
    std::unique_ptr<CcuCtxGroup> ccuCtxGroup = std::make_unique<CcuCtxGroup>();
    bool transportStatus = true;
    // check
    EXPECT_NO_THROW(preprocessor.CreateCcuCtxGroup(*ins1, ccuCtxGroup, transportStatus));
    EXPECT_EQ(1, ccuCtxGroup->ctxs.size());
    EXPECT_EQ(true, transportStatus);

    CcuCtxCreatorRegistry::GetInstance().creators.clear();

    delete ccuTransport;
    delete group;
}

TEST_F(CcuInsPreprocessorTest, should_throw_socket_timeout_transportsconnect)
{
    // when
    GlobalMockObject::verify();
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));
    MOCKER_CPP(&RdmaHandleManager::GetJfcHandle)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue((JfcHandle)0));
    MOCKER_CPP(&RdmaHandleManager::GetDieAndFuncId)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(std::pair<uint32_t, uint32_t>(0, 0)));
    vector<std::pair<CcuTransport*, LinkData>> transports;
    IpAddress localIp;
    IpAddress remoteIp;
    RdmaHandle rdmaHandle;
    shared_ptr<Socket> fakeSocket
        = make_shared<Socket>(nullptr, localIp, 100, remoteIp, "test", SocketRole::SERVER, NicType::DEVICE_NIC_TYPE);
    BasePortType portType(PortDeploymentType::DEV_NET, ConnectProtoType::UB);
    LinkData linkData(portType, 0, 1, 0, 1);
    CcuChannelInfo channelInfo;
    vector<CcuJetty*> ccuJettys;
    auto c = std::make_unique<CcuConnection>(linkData.GetLocalAddr(), linkData.GetRemoteAddr(), channelInfo, ccuJettys);
    CcuTransport::CclBufferInfo locCclBufInfo;
    CcuTransport* ccuTransport = new CcuTransport(fakeSocket.get(), std::move(c), locCclBufInfo);
    transports.push_back(make_pair(ccuTransport, linkData));
    MOCKER_CPP(&CcuTransportMgr::GetUnConfirmedTrans)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(transports));

    MOCKER_CPP(&CcuTransport::GetStatus)
        .stubs()
        .with()
        .will(returnValue((CcuTransport::TransStatus)CcuTransport::TransStatus::SOCKET_TIMEOUT));
    MOCKER_CPP(&CcuTransport::SetHandshakeMsg).stubs().with(mockcpp::any()).will(ignoreReturnValue());

    // then
    std::unique_ptr<CommunicatorImpl> comm = std::make_unique<CommunicatorImpl>();
    comm->currentCollOperator = make_unique<CollOperator>();
    comm->currentCollOperator->opType = OpType::ALLREDUCE;
    CcuTransportMgr ccuTransportMgr(*comm, 0);
    EXPECT_THROW(ccuTransportMgr.TransportsConnect(), TimeoutException);
    // 该用例打桩会影响后续用例

    delete ccuTransport;
}

TEST_F(CcuInsPreprocessorTest, RecoverCcuTransportCtx_test1)
{
    CcuTransportGroup* group;
    CcuTransport* ccuTrans;
    MOCKER_CPP(&CcuTransportGroupMgr::PrepareCreate)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(group));
    MOCKER_CPP(&CcuTransportMgr::PrepareCreate)
        .stubs()
        .with(mockcpp::any(), outBound(ccuTrans))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(&HrtGetDeviceType).stubs().will(returnValue(DevType(DevType::DEV_TYPE_950)));
    CommunicatorImpl* communicator;
    CcuInsPreprocessor preprocessor(communicator);
    bool transportStatus = true;
    // check
    MOCKER_CPP(&RdmaHandleManager::GetDieAndFuncId).stubs().will(returnValue(make_pair<uint32_t, uint32_t>(0, 0)));
    vector<LinkData> links;
    vector<std::pair<LinkGroup, u32>> linkGroupPair;
    LinkData linkData(
        PortDeploymentType::P2P, LinkProtocol::UB_CTP, 0, 1, IpAddress{"10.0.0.1"}, IpAddress{"10.0.0.2"});
    links.push_back(linkData);
    LinkGroup linkGroup{};
    linkGroup.AddLink({linkData});
    LinkData otherLinkData(
        PortDeploymentType::P2P, LinkProtocol::UB_CTP, 1, 1, IpAddress{"10.0.0.3"}, IpAddress{"10.0.0.4"});
    ;
    linkGroup.AddLink({otherLinkData});
    linkGroupPair.push_back(make_pair(linkGroup, 0));

    EXPECT_NO_THROW(preprocessor.RecoverCcuTransportCtx(links, linkGroupPair));
}

TEST_F(CcuInsPreprocessorTest, RecoverCcuTransportConfirm_test1)
{
    MOCKER_CPP(&CcuTransportMgr::RecoverTransportsConnect).stubs().with().will(ignoreReturnValue());

    CommunicatorImpl* communicator;
    CcuInsPreprocessor preprocessor(communicator);
    EXPECT_NO_THROW(preprocessor.RecoverCcuTransportConfirm());
}

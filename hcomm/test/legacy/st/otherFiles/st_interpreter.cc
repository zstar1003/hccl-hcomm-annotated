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
#include "interpreter.h"
#include "communicator_impl.h"
#include "stream_manager.h"
#include "local_notify_v2.h"
#include "coll_service_device_mode.h"
#include "ccu_instruction_all_gather_mesh1d.h"
#include "aicpu_res_package_helper.h"
#include "alg_topo_package_helper.h"
#include "ccu_ctx.h"
#include "ccu_ins_group.h"
#include "ccu_ctx_mgr.h"
#include "ccu_ins_group.h"
#include "ccu_ins_preprocessor.h"
#include "ccu_registered_ctx_mgr.h"
#include "ccu_context_mgr_imp.h"
#include "ccu_context_all_to_all_mesh2d.h"
#include "stream_manager.h"
#include "stream_utils.h"
#undef private
#undef protected

using namespace Hccl;
using namespace std;

class StubCommunicatorImplInterpreter : public CommunicatorImpl {
public:
    StubCommunicatorImplInterpreter()
    {
        dataBufferManager = make_unique<DataBufManager>();

        localRmaBufManager = make_unique<LocalRmaBufManager>(*this);

        remoteRmaBufManager = make_unique<RemoteRmaBufManager>(*this);

        rmaConnectionManager = make_unique<RmaConnManager>(*this);

        queueNotifyManager = make_unique<QueueNotifyManager>(*this);

        queueWaitGroupCntNotifyManager = make_unique<QueueWaitGroupCntNotifyManager>();

        queueBcastPostCntNotifyManager = make_unique<QueueBcastPostCntNotifyManager>();

        currentCollOperator = make_unique<CollOperator>();
        currentCollOperator->opMode = OpMode::OPBASE;
        currentCollOperator->opTag = "op_base";
    }

    void SetOp(OpMode opMode, string tag)
    {
        currentCollOperator->opMode = opMode;
        currentCollOperator->opTag = tag;
    }

    DataBufManager& GetDataBufferManager() const override { return *dataBufferManager.get(); }

    LocalRmaBufManager& GetLocalRmaBufManager() const override { return *localRmaBufManager.get(); }

    RemoteRmaBufManager& GetRemoteRmaBufManager() const override { return *remoteRmaBufManager.get(); }

    QueueNotifyManager& GetAicpuQueueNotifyManager() const override { return *aicpuQueueNotifyManager_.get(); }

    QueueNotifyManager& GetCcuQueueNotifyManager() const override { return *ccuQueueNotifyManager_.get(); }

    QueueWaitGroupCntNotifyManager& GetQueueWaitGroupCntNotifyManager() const override
    {
        return *queueWaitGroupCntNotifyManager.get();
    }

    QueueBcastPostCntNotifyManager& GetBcastPostCntNotifyManager() const override
    {
        return *queueBcastPostCntNotifyManager.get();
    }

    RmaConnManager& GetRmaConnManager() const override { return *rmaConnectionManager.get(); }

    CollOperator* GetCurrentCollOperator() const override { return currentCollOperator.get(); }

private:
    unique_ptr<DataBufManager> dataBufferManager;
    unique_ptr<LocalRmaBufManager> localRmaBufManager;
    unique_ptr<RemoteRmaBufManager> remoteRmaBufManager;
    unique_ptr<QueueNotifyManager> queueNotifyManager;
    unique_ptr<QueueWaitGroupCntNotifyManager> queueWaitGroupCntNotifyManager;
    unique_ptr<QueueBcastPostCntNotifyManager> queueBcastPostCntNotifyManager;
    unique_ptr<ConnLocalNotifyManager> connLocalNotifyManager;
    unique_ptr<StreamManager> streamManager;
    unique_ptr<SocketManager> socketManager;
    unique_ptr<RmaConnManager> rmaConnectionManager;
    unique_ptr<CollServiceBase> collService;
    unique_ptr<CollOperator> currentCollOperator;
};

class InterpreterTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "InterpreterTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "InterpreterTest TearDown" << std::endl; }

    virtual void SetUp()
    {
        masterInsQue = make_shared<InsQueue>();

        BasePortType portType(PortDeploymentType::P2P, ConnectProtoType::PCIE);
        LinkData link(portType, 0, 1, 0, 0);

        DataSlice srcSlice(BufferType::SCRATCH, 0, 100);
        DataSlice dstSlice(BufferType::SCRATCH, 0, 100);

        unique_ptr<InsLocalCopy> insLocalCopy = make_unique<InsLocalCopy>(srcSlice, dstSlice);
        masterInsQue->Append(std::move(insLocalCopy));
        std::cout << "A Test case in InterpreterTest SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in InterpreterTest TearDown" << std::endl;
    }
    shared_ptr<InsQueue> masterInsQue;
};

CollOpParams GetCollOpParamsStub()
{
    CollOpParams collOpParams;
    collOpParams.opType = OpType::SEND;
    collOpParams.dataType = DataType::INT8; // sizeof(int8) = 1
    collOpParams.reduceOp = ReduceOp::SUM;
    collOpParams.dstRank = 1;
    collOpParams.sendBuf = nullptr;
    collOpParams.recvBuf = nullptr;
    collOpParams.count = 10;
    collOpParams.root = 0;
    collOpParams.staticAddr = true;
    collOpParams.staticShape = true;
    collOpParams.outputDataType = DataType::INT8;

    collOpParams.dstRank = 0;
    return collOpParams;
}

TEST_F(InterpreterTest, St_Submit_When_input_Expect_NO_THROW)
{
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType(DevType::DEV_TYPE_950)));
    MOCKER(HrtGetDevice).defaults().will(returnValue(0));
    MOCKER(CcuDeviceManager::ReleaseCke).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuTransportGroup::CheckTransports).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER_CPP(&CcuTransportGroup::CheckTransportCntCke).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CcuTransportGroup::Destroy).stubs();
    MOCKER_CPP(&CcuTransport::ReleaseTransRes).stubs();
    MOCKER_CPP(&CcuConnection::ReleaseConnRes).stubs().will(returnValue((HcclResult)HcclResult::HCCL_SUCCESS));
    MOCKER(HrtMemcpy).stubs();
    MOCKER(&GetStreamCaptureInfo).stubs().will(returnValue(HCCL_SUCCESS));
    void* ptr = nullptr;
    MOCKER(HrtStreamCreateWithFlags).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(ptr));
    MOCKER(HrtGetStreamId).stubs().will(returnValue(0));

    auto insQueue = make_shared<InsQueue>();
    insQueue->Append(std::move(std::make_unique<CcuInstructionAllGatherMesh1D>()));
    auto ccuInsGroup = std::make_unique<CcuInsGroup>();
    ccuInsGroup->Append(std::move(std::make_unique<CcuInstructionAllGatherMesh1D>()));
    ccuInsGroup->Append(std::move(std::make_unique<CcuInstructionAllGatherMesh1D>()));
    insQueue->Append(std::move(ccuInsGroup));
    insQueue->Append(std::move(std::make_unique<CcuInstructionAllGatherMesh1D>()));

    auto subInsQueue = insQueue->Fork();
    subInsQueue->Append(std::move(std::make_unique<CcuInstructionAllGatherMesh1D>()));
    auto ccuInsGroup0 = std::make_unique<CcuInsGroup>();
    ccuInsGroup0->Append(std::move(std::make_unique<CcuInstructionAllGatherMesh1D>()));
    ccuInsGroup0->Append(std::move(std::make_unique<CcuInstructionAllGatherMesh1D>()));
    ccuInsGroup0->Append(std::move(std::make_unique<CcuInstructionAllGatherMesh1D>()));
    subInsQueue->Append(std::move(ccuInsGroup0));
    auto ccuInsGroup2 = std::make_unique<CcuInsGroup>();
    ccuInsGroup2->Append(std::move(std::make_unique<CcuInstructionAllGatherMesh1D>()));
    ccuInsGroup2->Append(std::move(std::make_unique<CcuInstructionAllGatherMesh1D>()));
    subInsQueue->Append(std::move(ccuInsGroup2));

    CommunicatorImpl comm;
    comm.currentCollOperator = std::make_unique<CollOperator>();
    comm.currentCollOperator->opMode = OpMode::OPBASE;
    comm.currentCollOperator->opMode = OpMode::OPBASE;
    comm.currentCollOperator->opType = OpType::DEBUGCASE;
    comm.currentCollOperator->inputMem = DevBuffer::Create(0x100, 10);
    comm.currentCollOperator->outputMem = DevBuffer::Create(0x100, 10);
    comm.InitStreamManager();
    comm.streamManager->opbase = std::make_unique<OpbaseStreamManager>(&comm);
    comm.streamManager->opbase->master = std::make_unique<Stream>();
    comm.superFasterLoad = true;

    CollAlgOperator collAlgOp;
    collAlgOp.opType = OpType::ALLTOALL;
    collAlgOp.dataType = DataType::INT8;
    collAlgOp.dataCount = 4;

    uint32_t rankId = 2;
    uint32_t rankSize = 4;
    std::vector<uint32_t> dimSize = {2, 2};
    std::vector<uint32_t> dimIds = {rankId % dimSize[0], rankId / dimSize[0]};
    std::vector<std::vector<RankId>> tempVTopo = {{0, 1}, {0, 2}};
    uint16_t axisId;
    axisId = 0;
    CcuCtxArgAlltoAllMesh2D ctxArg0(dimSize, rankId, axisId, collAlgOp, tempVTopo);
    std::vector<CcuTransport*> transports0;
    CcuTransport::CclBufferInfo locCclBufInfo;
    shared_ptr<CcuTransport> utCcuTransport = make_shared<CcuTransport>(nullptr, nullptr, locCclBufInfo);
    transports0.push_back(utCcuTransport.get());
    CcuTransportGroup transportGroup0(transports0, 4);
    transportGroup0.cntCkesGroup = {1128, 1129, 1130, 1131};
    CcuContextAlltoAllMesh2D ctx0(ctxArg0, transports0, transportGroup0);
    CcuTaskArgAlltoAllMesh2D taskArg0(0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    ctx0.GeneArgs(taskArg0);

    CtxMgrImp::GetInstance(0).ctxGroupMap_[0].ctxs.push_back(std::make_unique<CcuContextAlltoAllMesh2D>(ctx0));
    CtxMgrImp::GetInstance(0).ctxGroupMap_[0].ctxs.push_back(std::make_unique<CcuContextAlltoAllMesh2D>(ctx0));

    Interpreter interpreter(comm);
    EXPECT_NO_THROW(interpreter.Submit(*insQueue));
}

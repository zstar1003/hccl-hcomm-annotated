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

#include <string>

#define private public
#define protected public
#include "hccl_alg.h"
#include "hccl_impl.h"
#include "hccl_comm_pub.h"
#include "comm_impl.h"
#include "dispatcher_pub.h"
#include "hccl_communicator.h"
#include "coll_all_gather_comm_executor.h"
#include "coll_all_reduce_comm_executor.h"
#include "coll_all_reduce_mesh_executor.h"
#include "coll_all_reduce_ring_executor.h"
#include "coll_all_reduce_reduce_plus_bcast_executor.h"
#include "coll_reduce_scatter_comm_executor.h"
#include "coll_reduce_scatter_ring_executor.h"
#include "coll_reduce_scatter_mesh_executor.h"
#undef private
#undef protected
#include "dlra_function.h"
#include "adapter_prof.h"

using namespace hccl;
using namespace std;

class HcclImplAlgTestNB : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        s32 ret = HcclDispatcherInit(DispatcherType::DISPATCHER_NORMAL, 0, &dispatcherPtr);
        if (ret != HCCL_SUCCESS)
            return;
        if (dispatcherPtr == nullptr)
            return;
        dispatcher = reinterpret_cast<DispatcherPub*>(dispatcherPtr);
        DlRaFunction::GetInstance().DlRaFunctionInit();
        std::cout << "HcclImplAlgTestNB SetUP" << std::endl;
    }
    static void TearDownTestCase()
    {
        if (dispatcherPtr != nullptr) {
            s32 ret = HcclDispatcherDestroy(dispatcherPtr);
            EXPECT_EQ(ret, HCCL_SUCCESS);
            dispatcherPtr = nullptr;
            dispatcher = nullptr;
        }
        std::cout << "HcclImplAlgTestNB TearDown" << std::endl;
    }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = -1;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&HcclCommunicator::InitPreResource).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(hrtProfRegisterCtrlCallback).stubs().will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
    static HcclDispatcher dispatcherPtr;
    static DispatcherPub* dispatcher;
};
HcclDispatcher HcclImplAlgTestNB::dispatcherPtr = nullptr;
DispatcherPub* HcclImplAlgTestNB::dispatcher = nullptr;

static void TestConstructParam(HcclCommParams& params, RankTable_t& rankTable)
{
    string commId = "comm ";
    memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1);
    params.rank = 0;
    params.totalRanks = 2;
    params.isHeterogComm = false;
    params.logicDevId = 0;
    params.commWorkMode = WorkMode::HCCL_MODE_NORMAL;
    params.deviceType = DevType::DEV_TYPE_910;

    rankTable.collectiveId = "192.168.0.101-8000-8001";
    vector<RankInfo_t> rankVec(2);
    rankVec[0].rankId = 0;
    rankVec[0].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr1(1694542016);
    rankVec[0].deviceInfo.deviceIp.push_back(ipAddr1); // 101.0.168.192
    rankVec[0].serverIdx = 0;
    rankVec[0].serverId = "192.168.0.101";
    rankVec[1].rankId = 1;
    rankVec[1].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr2(1711319232);
    rankVec[1].deviceInfo.deviceIp.push_back(ipAddr2); // 101.0.168.192
    rankVec[1].serverIdx = 1;
    rankVec[1].serverId = "192.168.0.102";
    rankTable.rankList.assign(rankVec.begin(), rankVec.end());
    rankTable.deviceNum = 2;
    rankTable.serverNum = 2;
}

TEST_F(HcclImplAlgTestNB, ut_ReduceScatterComm_NB)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NB;
    impl->topoType_ = TopoType::TOPO_TYPE_8P_RING;

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);

    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_8P_RING;

    CollReduceScatterCommExecutor* executor = new CollReduceScatterCommExecutor(impl->dispatcher_, topoMatcher);

    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 4096;
    opParam.DataDes.count = 1024;
    opParam.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.reduceType = HCCL_REDUCE_SUM;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(
        opParam.tag, HcclCMDType::HCCL_CMD_REDUCE_SCATTER, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    DeviceMem scratchMem = DeviceMem::alloc(4096);
    resourceResponse.scratchMem = scratchMem;
    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    implBase = nullptr;

    delete executor;
    GlobalMockObject::verify();
}

TEST_F(HcclImplAlgTestNB, ut_AllGatherComm_NB)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLGATHER].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLGATHER].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NB;
    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    implBase->InitCCLbuffer(4096, 4096);

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910B;
    impl->topoType_ = TopoType::TOPO_TYPE_8P_RING;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_8P_RING;
    CollAllGatherCommExecutor* executor = new CollAllGatherCommExecutor(impl->dispatcher_, topoMatcher);

    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 4096;
    opParam.DataDes.count = 1024;
    opParam.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.reduceType = HCCL_REDUCE_SUM;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(
        opParam.tag, HcclCMDType::HCCL_CMD_ALLGATHER, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete executor;
    GlobalMockObject::verify();
}

TEST_F(HcclImplAlgTestNB, ut_AllReduceComm_NB)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    implBase->InitCCLbuffer(4096, 4096);

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NB;
    impl->topoType_ = TopoType::TOPO_TYPE_8P_RING;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_8P_RING;
    CollAllReduceCommExecutor* executor = new CollAllReduceCommExecutor(impl->dispatcher_, topoMatcher);

    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 4096;
    opParam.DataDes.count = 1024;
    opParam.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.reduceType = HCCL_REDUCE_SUM;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(
        opParam.tag, HcclCMDType::HCCL_CMD_ALLREDUCE, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete executor;
    GlobalMockObject::verify();
}

TEST_F(HcclImplAlgTestNB, ut_ReduceScatterRingExecutor_NB)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NB;
    impl->topoType_ = TopoType::TOPO_TYPE_4P_RING;

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);

    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_4P_RING;
    CollReduceScatterRingExecutor* executor = new CollReduceScatterRingExecutor(impl->dispatcher_, topoMatcher);

    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 4096;
    opParam.DataDes.count = 1024;
    opParam.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.reduceType = HCCL_REDUCE_SUM;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(
        opParam.tag, HcclCMDType::HCCL_CMD_REDUCE_SCATTER, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    DeviceMem scratchMem = DeviceMem::alloc(4096);
    resourceResponse.scratchMem = scratchMem;
    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    implBase = nullptr;

    delete executor;
    GlobalMockObject::verify();
}

TEST_F(HcclImplAlgTestNB, ut_ReduceScatterMeshExecutor_NB)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_8P_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NB;
    impl->topoType_ = TopoType::TOPO_TYPE_8P_RING;

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);

    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_8P_RING;
    CollReduceScatterMeshExecutor* executor = new CollReduceScatterMeshExecutor(impl->dispatcher_, topoMatcher);

    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 4096;
    opParam.DataDes.count = 1024;
    opParam.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.reduceType = HCCL_REDUCE_SUM;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(AlgTemplateBase::PrepareSliceMeshStreams).stubs().will(returnValue(HCCL_SUCCESS));

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(
        opParam.tag, HcclCMDType::HCCL_CMD_REDUCE_SCATTER, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    DeviceMem scratchMem = DeviceMem::alloc(4096);
    resourceResponse.scratchMem = scratchMem;
    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    implBase = nullptr;

    delete executor;
    GlobalMockObject::verify();
}

TEST_F(HcclImplAlgTestNB, ut_AllReduceReducePlusBcast_NB)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    implBase->InitCCLbuffer(200 * 1024 * 1024, 200 * 1024 * 1024);
    impl->deviceLogicId_ = 0;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_8P_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NB;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    CollAllReduceReducePlusBcastExecutor* executor
        = new CollAllReduceReducePlusBcastExecutor(impl->dispatcher_, topoMatcher);

    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 1024;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 1024;
    opParam.DataDes.count = 4096;
    opParam.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.reduceType = HCCL_REDUCE_SUM;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(
        opParam.tag, HcclCMDType::HCCL_CMD_ALLREDUCE, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete executor;
    GlobalMockObject::verify();
}

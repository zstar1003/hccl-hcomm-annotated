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
#include "hccl_alg.h"
#include "hccl_impl.h"
#include "hccl_communicator.h"
#include "hccl_comm_pub.h"
#include "all_reduce_mesh_opbase_pub.h"
#include "dispatcher_pub.h"
#include "coll_all_reduce_mesh_mid_count_executor.h"
#include "coll_all_reduce_mesh_oneshot_executor.h"
#include "coll_all_reduce_mesh_opbase_executor.h"
#undef private
#undef protected
#include "profiler_manager.h"
#include "dlra_function.h"
#include "adapter_prof.h"

using namespace std;
using namespace hccl;

class AllreduceMeshOpbaseTest : public testing::Test {
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
        std::cout << "AllreduceMeshOpbaseTest SetUP" << std::endl;
    }
    static void TearDownTestCase()
    {
        if (dispatcherPtr != nullptr) {
            s32 ret = HcclDispatcherDestroy(dispatcherPtr);
            EXPECT_EQ(ret, HCCL_SUCCESS);
            dispatcherPtr = nullptr;
            dispatcher = nullptr;
        }
        std::cout << "AllreduceMeshOpbaseTest TearDown" << std::endl;
    }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
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
HcclDispatcher AllreduceMeshOpbaseTest::dispatcherPtr = nullptr;
DispatcherPub* AllreduceMeshOpbaseTest::dispatcher = nullptr;

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

TEST_F(AllreduceMeshOpbaseTest, ut_impl_alg)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    implBase->InitCCLbuffer(200 * 1024 * 1024, 200 * 1024 * 1024);
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    CollAllReduceMeshOpbaseExecutor* executor = new CollAllReduceMeshOpbaseExecutor(impl->dispatcher_, topoMatcher);

    DeviceMem inputPtrMem = DeviceMem::alloc(4096);
    DeviceMem outputPtrMem = DeviceMem::alloc(4096);
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    DeviceMem scratchMem = DeviceMem::alloc(4096);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputPtrMem.ptr();
    opParam.inputSize = 1024;
    opParam.outputPtr = outputPtrMem.ptr();
    opParam.outputSize = 1024;
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
    resourceResponse.scratchMem = scratchMem;
    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete executor;
    GlobalMockObject::verify();
}

TEST_F(AllreduceMeshOpbaseTest, ut_impl_alg2)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    implBase->InitCCLbuffer(200 * 1024 * 1024, 200 * 1024 * 1024);
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    CollAllReduceMeshOneshotExecutor* executor = new CollAllReduceMeshOneshotExecutor(impl->dispatcher_, topoMatcher);

    DeviceMem inputPtrMem = DeviceMem::alloc(4096);
    DeviceMem outputPtrMem = DeviceMem::alloc(4096);
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    DeviceMem scratchMem = DeviceMem::alloc(4096);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputPtrMem.ptr();
    opParam.inputSize = 1024;
    opParam.outputPtr = outputPtrMem.ptr();
    opParam.outputSize = 1024;
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
    resourceResponse.scratchMem = scratchMem;
    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete executor;
    GlobalMockObject::verify();
}

TEST_F(AllreduceMeshOpbaseTest, ut_impl_alg3)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    implBase->InitCCLbuffer(200 * 1024 * 1024, 200 * 1024 * 1024);
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    CollAllReduceMeshOneshotExecutor* executor = new CollAllReduceMeshOneshotExecutor(impl->dispatcher_, topoMatcher);

    DeviceMem inputPtrMem = DeviceMem::alloc(4096);
    DeviceMem outputPtrMem = DeviceMem::alloc(4096);
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    DeviceMem scratchMem = DeviceMem::alloc(4096);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputPtrMem.ptr();
    opParam.inputSize = 1024;
    opParam.outputPtr = outputPtrMem.ptr();
    opParam.outputSize = 1024;
    opParam.DataDes.count = 1024;
    opParam.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.reduceType = HCCL_REDUCE_PROD;
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
    resourceResponse.scratchMem = scratchMem;
    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete executor;
    GlobalMockObject::verify();
}

TEST_F(AllreduceMeshOpbaseTest, ut_impl_alg4)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    implBase->InitCCLbuffer(200 * 1024 * 1024, 200 * 1024 * 1024);
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    CollAllReduceMeshMidCountExecutor* executor = new CollAllReduceMeshMidCountExecutor(impl->dispatcher_, topoMatcher);

    DeviceMem inputPtrMem = DeviceMem::alloc(4096);
    DeviceMem outputPtrMem = DeviceMem::alloc(4096);
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    DeviceMem scratchMem = DeviceMem::alloc(4096);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputPtrMem.ptr();
    opParam.inputSize = 1024;
    opParam.outputPtr = outputPtrMem.ptr();
    opParam.outputSize = 1024;
    opParam.DataDes.count = 1024;
    opParam.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.reduceType = HCCL_REDUCE_PROD;
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
    resourceResponse.scratchMem = scratchMem;
    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete executor;
    GlobalMockObject::verify();
}

TEST_F(AllreduceMeshOpbaseTest, ut_impl_alg5)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    implBase->InitCCLbuffer(200 * 1024 * 1024, 200 * 1024 * 1024);
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    CollAllReduceMeshOpbaseExecutor* executor = new CollAllReduceMeshOpbaseExecutor(impl->dispatcher_, topoMatcher);

    DeviceMem inputPtrMem = DeviceMem::alloc(4096);
    DeviceMem outputPtrMem = DeviceMem::alloc(4096);
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    DeviceMem scratchMem = DeviceMem::alloc(4096);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputPtrMem.ptr();
    opParam.inputSize = 1024;
    opParam.outputPtr = outputPtrMem.ptr();
    opParam.outputSize = 1024;
    opParam.DataDes.count = 1024;
    opParam.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.reduceType = HCCL_REDUCE_PROD;
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

TEST_F(AllreduceMeshOpbaseTest, ut_slice)
{
    std::vector<Stream> meshStreams;
    std::vector<std::shared_ptr<LocalNotify>> meshSignal;
    std::vector<std::shared_ptr<LocalNotify>> meshSignalAux;
    HcomCollOpInfo opInfo;

    std::unique_ptr<AllReduceMeshDirect> executor;
    executor.reset(new (std::nothrow) AllReduceMeshDirect(dispatcher));
    executor->Prepare(0, meshStreams, meshSignal, meshSignalAux, 0, 8, 0, &opInfo);

    std::vector<Slice> dataSlice;
    executor->PrepareSlice(1024, 4, 8, dataSlice);
    for (const auto& slice : dataSlice) {
        HCCL_ERROR("offset: %llu, size: %llu", slice.offset, slice.size);
    }

    executor->PrepareSlice(1023, 4, 8, dataSlice);
    for (const auto& slice : dataSlice) {
        HCCL_ERROR("offset: %llu, size: %llu", slice.offset, slice.size);
    }

    executor->PrepareSlice(16, 4, 8, dataSlice);
    for (const auto& slice : dataSlice) {
        HCCL_ERROR("offset: %llu, size: %llu", slice.offset, slice.size);
    }
}

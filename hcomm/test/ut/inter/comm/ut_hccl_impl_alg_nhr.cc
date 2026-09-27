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
#include "hccl_communicator.h"
#include "hccl_comm_pub.h"
#include "comm_impl.h"
#include "dispatcher_pub.h"
#include "coll_all_gather_comm_executor.h"
#include "coll_all_gather_ring_executor.h"
#include "coll_all_reduce_comm_executor.h"
#include "coll_all_reduce_reduce_plus_bcast_executor.h"
#include "coll_reduce_scatter_comm_executor.h"
#include "coll_reduce_scatter_ring_executor.h"
#include "coll_reduce_scatter_mesh_executor.h"
#include "coll_broadcast_comm_executor.h"
#include "coll_broadcast_mesh_executor.h"
#include "coll_reduce_mesh_executor.h"
#undef private
#undef protected
#include "llt_hccl_stub_sal_pub.h"
#include "dlra_function.h"
#include "adapter_prof.h"

using namespace hccl;
using namespace std;

class HcclImplAlgTestNHR : public testing::Test {
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
        std::cout << "HcclImplAlgTestNHR SetUP" << std::endl;
    }
    static void TearDownTestCase()
    {
        if (dispatcherPtr != nullptr) {
            s32 ret = HcclDispatcherDestroy(dispatcherPtr);
            EXPECT_EQ(ret, HCCL_SUCCESS);
            dispatcherPtr = nullptr;
            dispatcher = nullptr;
        }
        std::cout << "HcclImplAlgTestNHR TearDown" << std::endl;
    }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = 7;
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
HcclDispatcher HcclImplAlgTestNHR::dispatcherPtr = nullptr;
DispatcherPub* HcclImplAlgTestNHR::dispatcher = nullptr;

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

TEST_F(HcclImplAlgTestNHR, ut_AllReduceComm_NHR)
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

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    implBase->InitCCLbuffer(200 * 1024 * 1024, 200 * 1024 * 1024);

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
    impl->topoType_ = TopoType::TOPO_TYPE_8P_RING;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_8P_RING;
    CollAllReduceCommExecutor* executor = new CollAllReduceCommExecutor(impl->dispatcher_, topoMatcher);
    AlgType algType;
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
    algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
    executor->SetAlgType(algType);

    u64 sliceNum = 0;
    executor->GetSliceNum(256, true, sliceNum);
    EXPECT_EQ(sliceNum, 1);
    executor->GetSliceNum(1048576, false, sliceNum);
    EXPECT_EQ(sliceNum, 2);

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

TEST_F(HcclImplAlgTestNHR, ut_BroadcastComm_NHR)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_RESERVED;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    implBase->InitCCLbuffer(200 * 1024 * 1024, 200 * 1024 * 1024);

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910B;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_BROADCAST].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_BROADCAST].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
    impl->topoType_ = TopoType::TOPO_TYPE_8P_RING;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_8P_RING;
    CollBroadcastCommExecutor* executor = new CollBroadcastCommExecutor(impl->dispatcher_, topoMatcher);

    AlgType algType;
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
    algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
    executor->SetAlgType(algType);

    u64 sliceNum = 0;
    executor->GetSliceNum(256, true, sliceNum);
    EXPECT_EQ(sliceNum, 1);
    executor->GetSliceNum(1048576, false, sliceNum);
    EXPECT_EQ(sliceNum, 2);

    delete executor;

    topoMatcher->topoInfo_.userRank = 0;
    topoMatcher->topoInfo_.deviceNumPerAggregation = 2;
    topoMatcher->topoInfo_.userRankSize = 4;
    CollBroadcastMeshExecutor* meshExecutor = new CollBroadcastMeshExecutor(impl->dispatcher_, topoMatcher);

    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_MESH;
    algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
    meshExecutor->SetAlgType(algType);

    meshExecutor->GetSliceNum(256, true, sliceNum);
    EXPECT_EQ(sliceNum, 1);
    meshExecutor->GetSliceNum(1048576, false, sliceNum);
    EXPECT_EQ(sliceNum, 2);

    delete meshExecutor;

    GlobalMockObject::verify();
}

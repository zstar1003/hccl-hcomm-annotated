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
#include "coll_alg_operator.h"
#include "all_gather_operator.h"
#include "reduce_operator.h"
#include "reduce_scatter_operator.h"
#include "broadcast_operator.h"
#include "dispatcher_pub.h"
#include "coll_comm_executor.h"
#include "coll_all_gather_ring_for_910_93_executor.h"
#include "coll_reduce_scatter_ring_for_910_93_executor.h"
#include "coll_reduce_scatter_fast_double_ring_for_910_93_executor.h"
#include "externalinput.h"
#include "ffts_common_pub.h"
#include "heartbeat.h"
#undef private
#undef protected
#include "dlra_function.h"
#include "adapter_rts.h"

using namespace hccl;
using namespace std;

class HcclImplAlgTestDoubleRing : public testing::Test {
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
        std::cout << "HcclImplAlgTestDoubleRing SetUP" << std::endl;
    }
    static void TearDownTestCase()
    {
        if (dispatcherPtr != nullptr) {
            s32 ret = HcclDispatcherDestroy(dispatcherPtr);
            EXPECT_EQ(ret, HCCL_SUCCESS);
            dispatcherPtr = nullptr;
            dispatcher = nullptr;
        }
        std::cout << "HcclImplAlgTestDoubleRing TearDown" << std::endl;
    }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        HcclOpMetaInfo meta;
        bool hasMassTasks = true;
        hccl::Stream stream;
        ::InitTask(dispatcherPtr, stream, meta.isEnableCache, meta.GetCacheKey());
        if (hasMassTasks) {
            SetNormalMode(dispatcherPtr);
        }
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&Heartbeat::Init).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&HcclCommunicator::InitPreResource).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&Stream::ptr).stubs().with(mockcpp::any()).will(returnValue((void*)0x12345678));
        MOCKER_CPP(&Stream::IsMainStream).stubs().with(mockcpp::any()).will(returnValue(false));
        MOCKER(hrtNotifyWaitWithTimeOut).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
        setenv("HCCL_OP_RETRY_ENABLE", "L0:0, L1:0, L2:0", 1);
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
        unsetenv("HCCL_ALGO");
    }
    static HcclDispatcher dispatcherPtr;
    static DispatcherPub* dispatcher;
};
HcclDispatcher HcclImplAlgTestDoubleRing::dispatcherPtr = nullptr;
DispatcherPub* HcclImplAlgTestDoubleRing::dispatcher = nullptr;

static void TestConstructParam(HcclCommParams& params, RankTable_t& rankTable)
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

TEST_F(HcclImplAlgTestDoubleRing, ut_CollReduceScatterDoubleRingExecutor_Ring)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    DeviceMem inputMem = DeviceMem::alloc(4096 * 4);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910_93;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910_93;
    impl->devicePhyId_ = 0;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER].algoLevel0
        = AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;

    impl->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    algConfigurator->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    impl->superPodNum_ = 2;
    CommInfo tmpComm;

    const std::vector<std::vector<u32>> tmpRingNics = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}};
    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = count * 4 * 4;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = count * 4;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = op;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollNativeExecutorBase::CheckCommSize).stubs().will(returnValue(HCCL_SUCCESS));

    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910_93;
    topoMatcher->topoInfo_.devicePhyId = 0;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    topoMatcher->topoInfo_.superPodNum = 2;
    AlgType algType;
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING;
    algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    CollReduceScatterRingFor91093Executor* executor
        = new CollReduceScatterRingFor91093Executor(impl->dispatcher_, topoMatcher);
    executor->SetAlgType(algType);

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(
        opParam.tag, HcclCMDType::HCCL_CMD_REDUCE_SCATTER, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    DeviceMem scratchMem = DeviceMem::alloc(count * 4 * 4 + 64);
    resourceResponse.scratchMem = scratchMem;
    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    executor->IsHugeData(100, &opParam);

    implBase = nullptr;

    delete executor;
    GlobalMockObject::verify();
}

TEST_F(HcclImplAlgTestDoubleRing, ut_CollReduceScatterFastDoubleRingFor91093Executor_Ring)
{
    HcclResult ret = HCCL_SUCCESS;
    ret = InitEnvVarParam();
    std::string tag = "test";
    DeviceMem inputMem = DeviceMem::alloc(4096 * 4);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910_93;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910_93;
    impl->devicePhyId_ = 0;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER].algoLevel0
        = AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    impl->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    algConfigurator->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    impl->superPodNum_ = 2;
    CommInfo tmpComm;

    const std::vector<std::vector<u32>> tmpRingNics = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}};

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = count * 4 * 4;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = count * 4;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = op;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollNativeExecutorBase::CheckCommSize).stubs().will(returnValue(HCCL_SUCCESS));

    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910_93;
    topoMatcher->topoInfo_.devicePhyId = 0;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    topoMatcher->topoInfo_.superPodNum = 2;
    AlgType algType;
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING;
    algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    CollReduceScatterFastDoubleRingFor91093Executor* executor
        = new CollReduceScatterFastDoubleRingFor91093Executor(impl->dispatcher_, topoMatcher);
    executor->SetAlgType(algType);
    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(
        opParam.tag, HcclCMDType::HCCL_CMD_REDUCE_SCATTER, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    DeviceMem scratchMem = DeviceMem::alloc(count * 4 * 4 + 64);
    resourceResponse.scratchMem = scratchMem;
    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    implBase = nullptr;

    delete executor;
    GlobalMockObject::verify();
}

TEST_F(HcclImplAlgTestDoubleRing, ut_SuperPod_CollReduceScatterDoubleRingExecutor_Ring)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam_SurperPod(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910_93;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;

    std::vector<std::vector<RankInfo>> commPlaneVectorL2 = impl->commFactory_->CommPlaneVector_[COMM_LEVEL2];
    impl->commFactory_->CommPlaneVector_[COMM_LEVEL2].push_back(commPlaneVectorL2[0]);
    std::vector<RankInfo> rankInfoVectorL1 = impl->commFactory_->CommPlaneVector_[COMM_LEVEL1][COMM_INDEX_0];
    impl->commFactory_->CommPlaneVector_[COMM_LEVEL1][COMM_INDEX_0].push_back(rankInfoVectorL1[COMM_INDEX_0]);

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910_93;
    impl->devicePhyId_ = 0;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER].algoLevel0
        = AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    impl->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    algConfigurator->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    impl->superPodNum_ = 2;
    CommInfo tmpComm;

    const std::vector<std::vector<u32>> tmpRingNics = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}};

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = count * 4 * 2;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = count * 4;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = op;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollNativeExecutorBase::CheckCommSize).stubs().will(returnValue(HCCL_SUCCESS));

    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910_93;
    topoMatcher->topoInfo_.devicePhyId = 0;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    topoMatcher->topoInfo_.superPodNum = 2;
    AlgType algType;
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING;
    algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    CollReduceScatterRingFor91093Executor* executor
        = new CollReduceScatterRingFor91093Executor(impl->dispatcher_, topoMatcher);
    executor->SetAlgType(algType);
    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(
        opParam.tag, HcclCMDType::HCCL_CMD_REDUCE_SCATTER, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    DeviceMem scratchMem = DeviceMem::alloc(4096 * 2);
    resourceResponse.scratchMem = scratchMem;
    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    implBase = nullptr;

    delete executor;
    GlobalMockObject::verify();
}

TEST_F(HcclImplAlgTestDoubleRing, ut_CollReduceDoubleRingExecutor_Ring)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_PROD;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910_93;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollNativeExecutorBase::CheckCommSize).stubs().will(returnValue(HCCL_SUCCESS));

    ret = implBase->AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;

    impl->topoAttr_.deviceLogicId = 0;
    impl->topoAttr_.devicePhyId = 0;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    impl->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    algConfigurator->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;

    const std::vector<std::vector<u32>> tmpRingNics
        = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}};

    MOCKER_CPP_VIRTUAL(
        *dispatcher, &DispatcherPub::SignalRecord,
        HcclResult(DispatcherPub::*)(HcclRtNotify, hccl::Stream&, u32, u64, s32, bool, u64, u32))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(
        *dispatcher, &DispatcherPub::SignalWait,
        HcclResult(DispatcherPub::*)(HcclRtNotify, hccl::Stream&, u32, u32, s32, bool, u32, u32))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));

    std::printf("[ut_CollReduceDoubleRingExecutor_Ring]");
    ret = implBase->Reduce(tag, inputMem.ptr(), outputMem.ptr(), count, dataType, op, 0, stream.ptr());
    implBase = nullptr;

    GlobalMockObject::verify();
}

TEST_F(HcclImplAlgTestDoubleRing, ut_CollReduceDoubleRingExecutor_Ring_level2)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_PROD;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910_93;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollNativeExecutorBase::CheckCommSize).stubs().will(returnValue(HCCL_SUCCESS));

    ret = implBase->AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;

    impl->topoAttr_.deviceLogicId = 0;
    impl->topoAttr_.devicePhyId = 0;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    impl->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    algConfigurator->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;

    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910_93;
    topoMatcher->topoInfo_.devicePhyId = 0;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    topoMatcher->topoInfo_.superPodNum = 2;

    const std::vector<std::vector<u32>> tmpRingNics
        = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}};

    MOCKER_CPP_VIRTUAL(
        *dispatcher, &DispatcherPub::SignalRecord,
        HcclResult(DispatcherPub::*)(HcclRtNotify, hccl::Stream&, u32, u64, s32, bool, u64, u32))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(
        *dispatcher, &DispatcherPub::SignalWait,
        HcclResult(DispatcherPub::*)(HcclRtNotify, hccl::Stream&, u32, u32, s32, bool, u32, u32))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));

    std::printf("[ut_CollReduceDoubleRingExecutor_Ring]");
    ret = implBase->Reduce(tag, inputMem.ptr(), outputMem.ptr(), count, dataType, op, 0, stream.ptr());
    implBase = nullptr;

    GlobalMockObject::verify();
}

TEST_F(HcclImplAlgTestDoubleRing, ut_AllGatherDoubleRingExecutor_Ring_SuperPod)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096 * 8);
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam_SurperPod(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910_93;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;

    impl->deviceLogicId_ = 0;
    impl->devicePhyId_ = 0;
    impl->superPodNum_ = 2;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLGATHER].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLGATHER].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    impl->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    algConfigurator->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    CommInfo tmpComm;

    std::vector<RankInfo> paraVector;
    IntraExchanger exchanger{};
    tmpComm.commLevel1.resize(1);
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;
    tmpComm.commLevel1[0].reset(new (std::nothrow) CommRing(
        tag, 0, 2, 0, 2, TopoType::TOPO_TYPE_NP_DOUBLE_RING, implBase->dispatcher_, nullptr, netDevCtxMap, exchanger,
        paraVector, inputMem, outputMem, false));
    tmpComm.commLevel1[0]->rankMap_ = {0, 1};
    tmpComm.commLevel0.resize(2);
    for (int i = 0; i < 2; i++) {
        tmpComm.commLevel0[i].reset(new (std::nothrow) CommRing(
            tag, i, 2, i, 2, TopoType::TOPO_TYPE_NP_DOUBLE_RING, implBase->dispatcher_, nullptr, netDevCtxMap,
            exchanger, paraVector, inputMem, outputMem, false));
        tmpComm.commLevel0[i]->rankMap_ = {0, 1};
    }
    tmpComm.commLevel2.resize(2);
    for (int i = 0; i < 2; i++) {
        tmpComm.commLevel2[i].reset(new (std::nothrow) CommRing(
            tag, i, 2, i, 2, TopoType::TOPO_TYPE_NP_DOUBLE_RING, implBase->dispatcher_, nullptr, netDevCtxMap,
            exchanger, paraVector, inputMem, outputMem, false));
        tmpComm.commLevel2[i]->rankMap_ = {0, 1};
    }
    impl->tagCommInfo_.insert(std::pair<std::string, CommInfo>(tag, std::move(tmpComm)));
    Level1StreamInfo tmpInnerStreamInfo;
    tmpInnerStreamInfo.ringNum = 4;
    tmpInnerStreamInfo.ringSignal.resize(3);
    tmpInnerStreamInfo.ringSignalAux.resize(3);
    tmpInnerStreamInfo.ringStreams.resize(3);
    for (int i = 0; i < 3; i++) {
        tmpInnerStreamInfo.ringStreams[i] = Stream(StreamType::STREAM_TYPE_ONLINE);
        tmpInnerStreamInfo.ringSignal[i].reset(new (std::nothrow) LocalNotify());
        tmpInnerStreamInfo.ringSignalAux[i].reset(new (std::nothrow) LocalNotify());
    }
    impl->tagStreamInfo_.insert(std::pair<std::string, Level1StreamInfo>(tag, std::move(tmpInnerStreamInfo)));

    const std::vector<std::vector<u32>> tmpRingNics = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}};

    MOCKER_CPP_VIRTUAL(
        *dispatcher, &DispatcherPub::SignalRecord,
        HcclResult(DispatcherPub::*)(HcclRtNotify, hccl::Stream&, u32, u64, s32, bool, u64, u32))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(
        *dispatcher, &DispatcherPub::SignalWait,
        HcclResult(DispatcherPub::*)(HcclRtNotify, hccl::Stream&, u32, u32, s32, bool, u32, u32))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(
        &DispatcherPub::MemcpyAsync, HcclResult(DispatcherPub::*)(
                                         void* dst, uint64_t destMax, const void* src, u64 count, HcclRtMemcpyKind kind,
                                         hccl::Stream& stream, u32 remoteUserRank, hccl::LinkType inLinkType))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));

    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910_93;
    topoMatcher->topoInfo_.devicePhyId = 0;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    topoMatcher->topoInfo_.superPodNum = 2;

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollNativeExecutorBase::CheckCommSize).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));
    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = count * 4;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = count * 4 * 8;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = op;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);
    AlgType algType;
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING;
    algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    CollAllGatherRingFor91093Executor* executor = new CollAllGatherRingFor91093Executor(impl->dispatcher_, topoMatcher);
    executor->SetAlgType(algType);
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

    implBase = nullptr;

    delete executor;
    GlobalMockObject::verify();
}

TEST_F(HcclImplAlgTestDoubleRing, ut_CollAllGatherDoubleRingExecutor_Ring)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_RESERVED;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910_93;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollNativeExecutorBase::CheckCommSize).stubs().will(returnValue(HCCL_SUCCESS));

    ret = implBase->AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->InitCCLbuffer(200 * 1024 * 1024, 200 * 1024 * 1024);

    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;

    impl->topoAttr_.deviceLogicId = 0;
    impl->topoAttr_.devicePhyId = 0;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLGATHER].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLGATHER].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    impl->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    algConfigurator->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;

    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910_93;
    topoMatcher->topoInfo_.devicePhyId = 0;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    topoMatcher->topoInfo_.superPodNum = 2;

    const std::vector<std::vector<u32>> tmpRingNics
        = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}};

    MOCKER_CPP_VIRTUAL(
        *dispatcher, &DispatcherPub::SignalRecord,
        HcclResult(DispatcherPub::*)(HcclRtNotify, hccl::Stream&, u32, u64, s32, bool, u64, u32))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(
        *dispatcher, &DispatcherPub::SignalWait,
        HcclResult(DispatcherPub::*)(HcclRtNotify, hccl::Stream&, u32, u32, s32, bool, u32, u32))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));

    std::printf("[ut_CollAllGatherDoubleRingExecutor_Ring]");
    ret = implBase->AllGather(tag, inputMem.ptr(), outputMem.ptr(), count, dataType, stream.ptr());
    implBase = nullptr;

    GlobalMockObject::verify();
}

TEST_F(HcclImplAlgTestDoubleRing, ut_SelectAlg)
{
    HcclResult ret = HCCL_SUCCESS;

    std::string tag = "test";
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 100;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_RESERVED;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    setenv("HCCL_ALGO", "level0:NA;level1:H-D_R", 1);
    ret = InitEnvVarParam();

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910_93;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    OpParam opParam;
    opParam.tag = "select_alg_test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = inputMem.size();
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = outputMem.size();
    opParam.DataDes.count = 1024;
    opParam.DataDes.dataType = HcclDataType::HCCL_DATA_TYPE_INT16;
    opParam.reduceType = HcclReduceOp::HCCL_REDUCE_RESERVED;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);
    opParam.syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;

    std::string algName;
    std::string newTag;

    std::unique_ptr<CollAlgOperator> allGatherOperator
        = implBase->implAlg_->GetAlgOperator(HcclCMDType::HCCL_CMD_ALLGATHER);
    allGatherOperator->SelectAlg(opParam.tag, opParam, algName, newTag);

    std::unique_ptr<CollAlgOperator> reduceScatterOperator
        = implBase->implAlg_->GetAlgOperator(HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
    reduceScatterOperator->SelectAlg(opParam.tag, opParam, algName, newTag);

    unsetenv("HCCL_ALGO");
}

TEST_F(HcclImplAlgTestDoubleRing, ut_CollReduceScatterDoubleRingExecutor_UpdateOffsetBasedOnStrideCount)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    DeviceMem inputMem = DeviceMem::alloc(4096 * 4);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910_93;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910_93;
    impl->devicePhyId_ = 0;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER].algoLevel0
        = AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    impl->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    algConfigurator->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    impl->superPodNum_ = 2;
    CommInfo tmpComm;

    const std::vector<std::vector<u32>> tmpRingNics = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}};

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = count * 4 * 4;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = count * 4;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = op;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollNativeExecutorBase::CheckCommSize).stubs().will(returnValue(HCCL_SUCCESS));

    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910_93;
    topoMatcher->topoInfo_.devicePhyId = 0;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    topoMatcher->topoInfo_.superPodNum = 2;

    CollReduceScatterRingFor91093Executor* executor
        = new CollReduceScatterRingFor91093Executor(impl->dispatcher_, topoMatcher);
    opParam.DataDes.strideCount = count * 2;
    std::vector<std::vector<Slice>> multRingsUserMemSlice;
    u32 ringNum = 2;
    u32 sliceNum = 4;
    u32 sliceSize = 512 * 4; // Byte
    for (u32 ringIndex = 0; ringIndex < ringNum; ringIndex++) {
        std::vector<Slice> oneRingUserMemSlice;
        for (u32 sliceIndex = 0; sliceIndex < sliceNum; sliceIndex++) {
            Slice slice;
            slice.offset = sliceIndex * (sliceSize * ringNum) + sliceSize * ringIndex;
            slice.size = sliceSize;
            oneRingUserMemSlice.push_back(slice);
        }
        multRingsUserMemSlice.push_back(oneRingUserMemSlice);
    }
    // Original multRingsUserMemSlice
    // selfRank:0, ringIndex:0, sliceIndex:0, slice.offset:0, slice.size:2048,
    // selfRank:1, ringIndex:0, sliceIndex:1, slice.offset:4096, slice.size:2048,
    // selfRank:2, ringIndex:0, sliceIndex:2, slice.offset:8192, slice.size:2048,
    // selfRank:3, ringIndex:0, sliceIndex:3, slice.offset:12288, slice.size:2048,
    // selfRank:0, ringIndex:1, sliceIndex:0, slice.offset:2048, slice.size:2048,
    // selfRank:1, ringIndex:1, sliceIndex:1, slice.offset:6144, slice.size:2048,
    // selfRank:2, ringIndex:1, sliceIndex:2, slice.offset:10240, slice.size:2048,
    // selfRank:3, ringIndex:1, sliceIndex:3, slice.offset:14336, slice.size:2048,

    ret = executor->UpdateOffsetBasedOnStrideCount(opParam, multRingsUserMemSlice);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::vector<std::vector<Slice>> validMultRingsUserMemSlice;
    std::vector<Slice> firstRingsUserMemSlice;
    Slice slice;
    slice.size = 2048;
    slice.offset = 0;
    firstRingsUserMemSlice.push_back(slice);
    slice.offset = 8192;
    firstRingsUserMemSlice.push_back(slice);
    slice.offset = 16384;
    firstRingsUserMemSlice.push_back(slice);
    slice.offset = 24576;
    firstRingsUserMemSlice.push_back(slice);
    validMultRingsUserMemSlice.push_back(firstRingsUserMemSlice);
    std::vector<Slice> secondRingsUserMemSlice;
    slice.offset = 2048;
    secondRingsUserMemSlice.push_back(slice);
    slice.offset = 10240;
    secondRingsUserMemSlice.push_back(slice);
    slice.offset = 18432;
    secondRingsUserMemSlice.push_back(slice);
    slice.offset = 26624;
    secondRingsUserMemSlice.push_back(slice);
    validMultRingsUserMemSlice.push_back(secondRingsUserMemSlice);
    // Updated multRingsUserMemSlice
    // selfRank:0, ringIndex:0, sliceIndex:0, slice.offset:0, slice.size:2048, updated
    // selfRank:1, ringIndex:0, sliceIndex:1, slice.offset:8192, slice.size:2048, updated
    // selfRank:2, ringIndex:0, sliceIndex:2, slice.offset:16384, slice.size:2048, updated
    // selfRank:3, ringIndex:0, sliceIndex:3, slice.offset:24576, slice.size:2048, updated
    // selfRank:0, ringIndex:1, sliceIndex:0, slice.offset:2048, slice.size:2048, updated
    // selfRank:1, ringIndex:1, sliceIndex:1, slice.offset:10240, slice.size:2048, updated
    // selfRank:2, ringIndex:1, sliceIndex:2, slice.offset:18432, slice.size:2048, updated
    // selfRank:3, ringIndex:1, sliceIndex:3, slice.offset:26624, slice.size:2048, updated

    for (u32 ringIndex = 0; ringIndex < multRingsUserMemSlice.size(); ringIndex++) {
        for (u32 sliceIndex = 0; sliceIndex < multRingsUserMemSlice[ringIndex].size(); sliceIndex++) {
            u64 offset = multRingsUserMemSlice[ringIndex][sliceIndex].offset;
            u64 validOffset = validMultRingsUserMemSlice[ringIndex][sliceIndex].offset;
            if (offset == validOffset) {
                std::cout << "ringIndex:" << ringIndex << ", " << "sliceIndex:" << sliceIndex << ", "
                          << "slice.offset:" << multRingsUserMemSlice[ringIndex][sliceIndex].offset << ", "
                          << "slice.size:" << multRingsUserMemSlice[ringIndex][sliceIndex].size << ", checked"
                          << std::endl;
                continue;
            }
            ret = HCCL_E_PARA;
        }
    }
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase = nullptr;

    delete executor;
    GlobalMockObject::verify();
}

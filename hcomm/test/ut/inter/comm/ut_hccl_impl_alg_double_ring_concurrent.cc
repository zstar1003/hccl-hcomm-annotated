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
#include "all_reduce_operator.h"
#include "reduce_scatter_operator.h"
#include "all_gather_operator.h"
#include "dispatcher_pub.h"
#include "coll_comm_executor.h"
#include "externalinput.h"
#include "comm_factory.h"
#include "profiler_manager.h"
#include "topo_info_extractor.h"
#include "coll_reduce_scatter_executor.h"
#include "coll_reduce_scatter_ring_executor.h"
#include "env_config.h"
#undef private
#undef protected
#include "dlra_function.h"
#include "adapter_prof.h"

using namespace hccl;
using namespace std;

class HcclImplAlgTestDoubleRingConcurrent : public testing::Test {
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
        std::cout << "HcclImplAlgTestDoubleRingConcurrent SetUP" << std::endl;
    }
    static void TearDownTestCase()
    {
        if (dispatcherPtr != nullptr) {
            s32 ret = HcclDispatcherDestroy(dispatcherPtr);
            EXPECT_EQ(ret, HCCL_SUCCESS);
            dispatcherPtr = nullptr;
            dispatcher = nullptr;
        }
        std::cout << "HcclImplAlgTestDoubleRingConcurrent TearDown" << std::endl;
    }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = -1;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&HcclCommunicator::InitPreResource).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(hrtProfRegisterCtrlCallback).stubs().will(returnValue(HCCL_SUCCESS));
        InitEnvParam();
        setenv("HCCL_OP_RETRY_ENABLE", "L0:0, L1:0, L2:0", 1);
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

HcclDispatcher HcclImplAlgTestDoubleRingConcurrent::dispatcherPtr = nullptr;
DispatcherPub* HcclImplAlgTestDoubleRingConcurrent::dispatcher = nullptr;

static void TestConstructParam(HcclCommParams& params, RankTable_t& rankTable)
{
    string commId = "comm ";
    memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1);
    params.rank = 0;
    params.totalRanks = 4;
    params.isHeterogComm = false;
    params.logicDevId = 0;
    params.commWorkMode = WorkMode::HCCL_MODE_NORMAL;
    params.deviceType = DevType::DEV_TYPE_910_93;

    rankTable.collectiveId = "192.168.0.101-8000-8001";
    vector<RankInfo_t> rankVec(4);
    rankVec[0].rankId = 0;
    rankVec[0].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr1(1694542016);
    rankVec[0].deviceInfo.deviceIp.push_back(ipAddr1); // 101.0.168.192
    rankVec[0].serverIdx = 0;
    rankVec[0].serverId = "192.168.0.101";
    rankVec[1].rankId = 1;
    rankVec[1].deviceInfo.devicePhyId = 1;
    HcclIpAddress ipAddr2(1694542017);
    rankVec[1].deviceInfo.deviceIp.push_back(ipAddr2); // 101.0.168.192
    rankVec[1].serverIdx = 0;
    rankVec[1].serverId = "192.168.0.101";
    rankVec[2].rankId = 2;
    rankVec[2].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr3(1711319232);
    rankVec[2].deviceInfo.deviceIp.push_back(ipAddr3); // 101.0.168.192
    rankVec[2].serverIdx = 1;
    rankVec[2].serverId = "192.168.0.102";
    rankVec[3].rankId = 3;
    rankVec[3].deviceInfo.devicePhyId = 1;
    HcclIpAddress ipAddr4(1711319233);
    rankVec[3].deviceInfo.deviceIp.push_back(ipAddr4); // 101.0.168.192
    rankVec[3].serverIdx = 1;
    rankVec[3].serverId = "192.168.0.102";
    rankTable.rankList.assign(rankVec.begin(), rankVec.end());
    rankTable.rankNum = 4;
    rankTable.deviceNum = 4;
    rankTable.serverNum = 2;
}

static void TestConstructParam_1(HcclCommParams& params, RankTable_t& rankTable)
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

static void TestConstructParam_server_1_rank(HcclCommParams& params, RankTable_t& rankTable)
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
    rankVec[1].deviceInfo.devicePhyId = 1;
    HcclIpAddress ipAddr2(1694542017);
    rankVec[1].deviceInfo.deviceIp.push_back(ipAddr2); // 101.0.168.192
    rankVec[1].serverIdx = 1;
    rankVec[1].serverId = "192.168.0.102";

    rankTable.rankList.assign(rankVec.begin(), rankVec.end());
    rankTable.rankNum = 2;
    rankTable.deviceNum = 2;
    rankTable.serverNum = 2;
}

static void TestConstructParam_server(HcclCommParams& params, RankTable_t& rankTable)
{
    string commId = "comm ";
    memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1);
    params.rank = 0;
    params.totalRanks = 4;
    params.isHeterogComm = false;
    params.logicDevId = 0;
    params.commWorkMode = WorkMode::HCCL_MODE_NORMAL;
    params.deviceType = DevType::DEV_TYPE_910_93;

    rankTable.collectiveId = "192.168.0.101-8000-8001";
    vector<RankInfo_t> rankVec(4);
    rankVec[0].rankId = 0;
    rankVec[0].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr1(1694542016);
    rankVec[0].deviceInfo.deviceIp.push_back(ipAddr1); // 101.0.168.192
    rankVec[0].serverIdx = 0;
    rankVec[0].serverId = "192.168.0.101";
    rankVec[1].rankId = 1;
    rankVec[1].deviceInfo.devicePhyId = 1;
    HcclIpAddress ipAddr2(1694542017);
    rankVec[1].deviceInfo.deviceIp.push_back(ipAddr2); // 101.0.168.192
    rankVec[1].serverIdx = 0;
    rankVec[1].serverId = "192.168.0.101";
    rankVec[2].rankId = 2;
    rankVec[2].deviceInfo.devicePhyId = 2;
    HcclIpAddress ipAddr3(1694542018);
    rankVec[2].deviceInfo.deviceIp.push_back(ipAddr3); // 101.0.168.192
    rankVec[2].serverIdx = 0;
    rankVec[2].serverId = "192.168.0.101";
    rankVec[3].rankId = 3;
    rankVec[3].deviceInfo.devicePhyId = 3;
    HcclIpAddress ipAddr4(1694542019);
    rankVec[3].deviceInfo.deviceIp.push_back(ipAddr4); // 101.0.168.192
    rankVec[3].serverIdx = 0;
    rankVec[3].serverId = "192.168.0.101";
    rankTable.rankList.assign(rankVec.begin(), rankVec.end());
    rankTable.rankNum = 4;
    rankTable.deviceNum = 4;
    rankTable.serverNum = 1;
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
    rankVec[0].superPodIdx = 0;
    rankVec[1].rankId = 1;
    rankVec[1].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr2(1711319232);
    rankVec[1].deviceInfo.deviceIp.push_back(ipAddr2); // 101.0.168.192
    rankVec[1].serverIdx = 1;
    rankVec[1].serverId = "192.168.0.102";
    rankVec[1].superPodId = "192.168.0.104";
    rankVec[0].superPodIdx = 1;
    rankTable.rankList.assign(rankVec.begin(), rankVec.end());
    rankTable.deviceNum = 2;
    rankTable.serverNum = 2;
}

static void TestConstructParam_2SurperPod(HcclCommParams& params, RankTable_t& rankTable)
{
    string commId = "comm ";
    memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1);
    params.rank = 0;
    params.totalRanks = 8;
    params.isHeterogComm = false;
    params.logicDevId = 0;
    params.commWorkMode = WorkMode::HCCL_MODE_NORMAL;
    params.deviceType = DevType::DEV_TYPE_910_93;

    rankTable.collectiveId = "192.168.0.101-8000-8001";
    vector<RankInfo_t> rankVec(8);
    rankVec[0].rankId = 0;
    rankVec[0].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr1(1694542016);
    rankVec[0].deviceInfo.deviceIp.push_back(ipAddr1);
    rankVec[0].serverIdx = 0;
    rankVec[0].serverId = "192.168.0.101";
    rankVec[0].superPodId = "192.168.0.105";
    rankVec[0].superPodIdx = 0;

    rankVec[1].rankId = 1;
    rankVec[1].deviceInfo.devicePhyId = 1;
    HcclIpAddress ipAddr2(1694542016);
    rankVec[1].deviceInfo.deviceIp.push_back(ipAddr2);
    rankVec[1].serverIdx = 0;
    rankVec[1].serverId = "192.168.0.101";
    rankVec[1].superPodId = "192.168.0.105";
    rankVec[1].superPodIdx = 0;

    rankVec[2].rankId = 2;
    rankVec[2].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr3(1711319232);
    rankVec[2].deviceInfo.deviceIp.push_back(ipAddr3);
    rankVec[2].serverIdx = 1;
    rankVec[2].serverId = "192.168.0.102";
    rankVec[2].superPodId = "192.168.0.105";
    rankVec[2].superPodIdx = 0;

    rankVec[3].rankId = 3;
    rankVec[3].deviceInfo.devicePhyId = 1;
    HcclIpAddress ipAddr4(1711319232);
    rankVec[3].deviceInfo.deviceIp.push_back(ipAddr4);
    rankVec[3].serverIdx = 1;
    rankVec[3].serverId = "192.168.0.102";
    rankVec[3].superPodId = "192.168.0.105";
    rankVec[3].superPodIdx = 0;

    rankVec[4].rankId = 4;
    rankVec[4].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr5(1728096448);
    rankVec[4].deviceInfo.deviceIp.push_back(ipAddr5);
    rankVec[4].serverIdx = 2;
    rankVec[4].serverId = "192.168.0.103";
    rankVec[4].superPodId = "192.168.0.106";
    rankVec[4].superPodIdx = 1;

    rankVec[5].rankId = 5;
    rankVec[5].deviceInfo.devicePhyId = 1;
    HcclIpAddress ipAddr6(1728096448);
    rankVec[5].deviceInfo.deviceIp.push_back(ipAddr6);
    rankVec[5].serverIdx = 2;
    rankVec[5].serverId = "192.168.0.103";
    rankVec[5].superPodId = "192.168.0.106";
    rankVec[5].superPodIdx = 1;

    rankVec[6].rankId = 6;
    rankVec[6].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr7(1744873664);
    rankVec[6].deviceInfo.deviceIp.push_back(ipAddr7);
    rankVec[6].serverIdx = 3;
    rankVec[6].serverId = "192.168.0.104";
    rankVec[6].superPodId = "192.168.0.106";
    rankVec[6].superPodIdx = 1;

    rankVec[7].rankId = 7;
    rankVec[7].deviceInfo.devicePhyId = 1;
    HcclIpAddress ipAddr8(1744873664);
    rankVec[7].deviceInfo.deviceIp.push_back(ipAddr8);
    rankVec[7].serverIdx = 3;
    rankVec[7].serverId = "192.168.0.104";
    rankVec[7].superPodId = "192.168.0.106";
    rankVec[7].superPodIdx = 1;

    rankTable.rankList.assign(rankVec.begin(), rankVec.end());
    rankTable.rankNum = 8;
    rankTable.deviceNum = 8;
    rankTable.serverNum = 4;
}

TEST_F(HcclImplAlgTestDoubleRingConcurrent, ut_CollReduceScatterDoubleRingConcurrentExecutor_Ring)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    unsetenv("HCCL_INTER_HCCS_DISABLE");
    setenv("HCCL_ALGO", "level0:NA;level1:NA;level2:ring", 1);
    setenv("HCCL_DEBUG_CONFIG", "AIV_OPS_EXC", 1);
    ret = InitEnvVarParam();
    ret = InitEnvParam();

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910_93;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollNativeExecutorBase::CheckCommSize).stubs().will(returnValue(HCCL_SUCCESS));

    ret = implBase->AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);
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
    CommInfo tmpComm;

    const std::vector<std::vector<u32>> tmpRingNics
        = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}};

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));
    ret = implBase->ReduceScatter(
        tag, inputMem.ptr(), outputMem.ptr(), count, dataType, HcclReduceOp::HCCL_REDUCE_SUM, stream.ptr());
    implBase = nullptr;

    GlobalMockObject::verify();
}

TEST_F(HcclImplAlgTestDoubleRingConcurrent, ut_CollAllGatherDoubleRingConcurrentExecutor_Ring)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 100;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_RESERVED;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    setenv("HCCL_ALGO", "level0:NA;level1:NA;level2:ring", 1);
    ret = InitEnvVarParam();

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
    topoMatcher->topoInfo_.devicePhyId = 0;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_NP_DOUBLE_RING;

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

TEST_F(HcclImplAlgTestDoubleRingConcurrent, ut_SetInterAlgoToRingIfConcurrentOn)
{
    HcclResult ret = HCCL_SUCCESS;

    std::string tag = "test";
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 100;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_RESERVED;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    setenv("HCCL_ALGO", "level0:NA;level1:NA;level2:ring", 1);
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
    opParam.tag = "AllGather_test";
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
}

TEST_F(HcclImplAlgTestDoubleRingConcurrent, ut_CollReduceScatterDoubleRingConcurrentExecutor_Ring_op)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    DeviceMem inputMem = DeviceMem::alloc(count * dataType);
    DeviceMem outputMem = DeviceMem::alloc(count * dataType);
    HcclReduceOp op = HCCL_REDUCE_SUM;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    setenv("HCCL_INTER_HCCS_DISABLE", "FALSE", 1);
    SetFftsSwitch(false);
    ret = InitEnvVarParam();

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910_93;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->InitCCLbuffer(count, count);

    MOCKER_CPP(&AlgConfigurator::IsHCCSSWNumEqualToTwiceSIONum).stubs().will(returnValue(true));
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
    CommInfo tmpComm;

    const std::vector<std::vector<u32>> tmpRingNics
        = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}};

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));
    ret = implBase->ReduceScatterOutPlace(
        tag, inputMem.ptr(), outputMem.ptr(), count, dataType, HcclReduceOp::HCCL_REDUCE_SUM, stream.ptr());
    implBase = nullptr;

    InitEnvVarParam();
    GlobalMockObject::verify();
}

TEST_F(HcclImplAlgTestDoubleRingConcurrent, ut_CollAllGatherDoubleRingConcurrentExecutor_Ring_Largesize)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    u64 count = 67108872;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    DeviceMem inputMem = DeviceMem::alloc(count * dataType);
    DeviceMem outputMem = DeviceMem::alloc(count * dataType);
    HcclReduceOp op = HCCL_REDUCE_SUM;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    SetFftsSwitch(false);
    ret = InitEnvVarParam();
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910_93;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->InitCCLbuffer(count, count);
    MOCKER_CPP(&AlgConfigurator::IsHCCSSWNumEqualToTwiceSIONum).stubs().will(returnValue(true));
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    impl->topoAttr_.deviceLogicId = 0;
    impl->topoAttr_.devicePhyId = 0;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLGATHER].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLGATHER].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    impl->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    algConfigurator->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;

    const std::vector<std::vector<u32>> tmpRingNics = {{0, 1, 2, 3, 4, 5, 6, 7}};
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

    MOCKER(LocalNotify::Wait).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(HcclD2DMemcpyAsync).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(AlgTemplateBase::ExecEmptyTask).stubs().will(returnValue(HCCL_SUCCESS));

    std::printf("[ut_collAllGatherDoubleRingConcurrentExecutor_Ring_Largesize]");

    ret = implBase->AllGather(tag, inputMem.ptr(), outputMem.ptr(), count, dataType, stream.ptr());

    implBase = nullptr;
    SetFftsSwitch(true);
    InitEnvVarParam();
    GlobalMockObject::verify();
}

TEST_F(HcclImplAlgTestDoubleRingConcurrent, ut_CollReduceScatterSingleRingConcurrentExecutor_Ring_Largesize)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    u64 count = 8388608;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    DeviceMem inputMem = DeviceMem::alloc(count * dataType);
    DeviceMem outputMem = DeviceMem::alloc(count * dataType);

    HcclReduceOp op = HCCL_REDUCE_SUM;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    unsetenv("HCCL_INTER_HCCS_DISABLE");
    SetFftsSwitch(false);
    ret = InitEnvVarParam();
    HcclCommParams params;
    RankTable_t rankTable;

    TestConstructParam_server_1_rank(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910_93;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollNativeExecutorBase::CheckCommSize).stubs().will(returnValue(HCCL_SUCCESS));

    ret = implBase->AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910_93;
    impl->devicePhyId_ = 0;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER].algoLevel0
        = AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    impl->topoType_ = TopoType::TOPO_TYPE_NP_SINGLE_RING;
    algConfigurator->topoType_ = TopoType::TOPO_TYPE_NP_SINGLE_RING;

    const std::vector<std::vector<u32>> tmpRingNics = {{0, 1, 2, 3, 4, 5, 6, 7}};

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

    MOCKER(LocalNotify::Wait).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(HcclD2DMemcpyAsync).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(AlgTemplateBase::ExecEmptyTask).stubs().will(returnValue(HCCL_SUCCESS));

    ret = implBase->ReduceScatter(
        tag, inputMem.ptr(), outputMem.ptr(), count, dataType, HcclReduceOp::HCCL_REDUCE_SUM, stream.ptr());
    implBase = nullptr;

    SetFftsSwitch(true);
    GlobalMockObject::verify();
}
#if 0
TEST_F(HcclImplAlgTestDoubleRingConcurrent, ut_CollMultiRingAllReduceAndMultiRootScatter)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    DeviceMem inputMem = DeviceMem::alloc(4096 * 2);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;


    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam_1(params, rankTable);

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl> &impl = implBase->implAlg_->pimpl_;
    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910;

    std::vector<Slice> dataSegsSlice;   // 数据分成ranksize份，每份的起始偏移和大小

    ret = AlgTemplateBase::PrepareSliceData(inputMem.size() / 4, 4, 8, 0, dataSegsSlice);

    std::unique_ptr<TopoMatcher> &topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910;

    std::unique_ptr<CollReduceScatterExecutor> executor(new CollReduceScatterRingExecutor(impl->dispatcher_, topoMatcher));

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

    MOCKER_CPP(&TransportManager::Alloc)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    (void) SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);

    std::vector<std::vector<u32>> multiRingOrder;
    std::vector<u32> tmpOuter0 = { 0, 1, 2, 6, 5, 4, 7, 3 }; // 环0
    std::vector<u32> tmpOuter1 = { 0, 3, 7, 4, 5, 6, 2, 1 }; // 环1
    std::vector<u32> tmpOuter2 = { 0, 2, 3, 1, 5, 7, 6, 4 }; // 环2
    std::vector<u32> tmpOuter3 = { 0, 4, 6, 7, 5, 1, 3, 2 }; // 环3
    multiRingOrder.push_back(tmpOuter0);
    multiRingOrder.push_back(tmpOuter1);
    multiRingOrder.push_back(tmpOuter2);
    multiRingOrder.push_back(tmpOuter3);

    MOCKER(LocalNotify::Wait)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));
    MOCKER(LocalNotify::Post)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollCommExecutor::GetRingsOrderByTopoType)
    .stubs()
    .will(returnValue(multiRingOrder));
    MOCKER_CPP(&CollCommExecutor::CheckCommSize)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollNativeExecutorBase::AddSubStreamToProfiling)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&StreamActiveManager::StreamActive)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollNativeExecutorBase::GetRankByUserRank)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    resourceRequest.streamNum = 3;
    resourceRequest.notifyNum = 6;

    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(opParam.tag, HcclCMDType::HCCL_CMD_REDUCE_SCATTER, opParam, resourceRequest, resourceResponse);
    std::vector<Stream> streams(3, Stream(StreamType::STREAM_TYPE_ONLINE));
    resourceResponse.slaveStreams = streams;
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    DeviceMem scratchMem = DeviceMem::alloc(count * 4 * 2 + 64);
    resourceResponse.scratchMem = scratchMem;

    auto slice1 = executor->AnyPathReduceScatterRingSlicePrepare(4, 8, true, outputMem, dataSegsSlice, tag);
    auto slice2 = executor->AnyPathReduceScatterRingSlicePrepare(4, 8, false, outputMem, dataSegsSlice, tag);
    auto slice3 = executor->AnyPathReduceScatterRingSlicePrepare(2, 8, true, outputMem, dataSegsSlice, tag);
    auto slice4 = executor->AnyPathReduceScatterRingSlicePrepare(2, 8, false, outputMem, dataSegsSlice, tag);
    auto slice5 = executor->AnyPathReduceScatterRingSlicePrepare(1, 8, true, outputMem, dataSegsSlice, tag);
    implBase = nullptr;
    executor = nullptr;

    GlobalMockObject::verify();
}
#endif
TEST_F(HcclImplAlgTestDoubleRingConcurrent, ut_GetRingsOrderByTopoType)
{
    HcclResult ret = HCCL_SUCCESS;
    ret = InitEnvVarParam();

    std::vector<u32> nicList = {0, 1, 2, 3, 4, 5, 6, 7};
    u32 rankSize = 8;
    TopoType topoType = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    std::vector<std::vector<u32>> res = GetRingsOrderForAnyPath(rankSize, topoType, nicList);
    EXPECT_EQ(res.size(), 2);
    std::vector<u32> target0 = {0, 1, 3, 2, 4, 5, 7, 6};
    bool isSame = (res[0] == target0);
    EXPECT_EQ(isSame, true);
    std::vector<u32> target1 = {0, 6, 7, 5, 4, 2, 3, 1};
    isSame = (res[1] == target1);
    EXPECT_EQ(isSame, true);
}

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
#undef private
#undef protected
#include "dlra_function.h"
#include "adapter_prof.h"

using namespace hccl;
using namespace std;

class HcclImplAlgTestDoubleRingConcurrentallreduce : public testing::Test {
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
        std::cout << "HcclImplAlgTestDoubleRingConcurrentallreduce SetUP" << std::endl;
    }
    static void TearDownTestCase()
    {
        if (dispatcherPtr != nullptr) {
            s32 ret = HcclDispatcherDestroy(dispatcherPtr);
            EXPECT_EQ(ret, HCCL_SUCCESS);
            dispatcherPtr = nullptr;
            dispatcher = nullptr;
        }
        std::cout << "HcclImplAlgTestDoubleRingConcurrentallreduce TearDown" << std::endl;
    }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        setenv("HCCL_INTER_HCCS_DISABLE", "FALSE", 1);
        s32 portNum = -1;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&HcclCommunicator::InitPreResource).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(hrtProfRegisterCtrlCallback).stubs().will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        unsetenv("HCCL_INTER_HCCS_DISABLE");
        std::cout << "A Test TearDown" << std::endl;
    }
    static HcclDispatcher dispatcherPtr;
    static DispatcherPub* dispatcher;
};

HcclDispatcher HcclImplAlgTestDoubleRingConcurrentallreduce::dispatcherPtr = nullptr;
DispatcherPub* HcclImplAlgTestDoubleRingConcurrentallreduce::dispatcher = nullptr;

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

    rankVec[1].rankId = 1;
    rankVec[1].deviceInfo.devicePhyId = 1;
    HcclIpAddress ipAddr2(1694542016);
    rankVec[1].deviceInfo.deviceIp.push_back(ipAddr2);
    rankVec[1].serverIdx = 0;
    rankVec[1].serverId = "192.168.0.101";
    rankVec[1].superPodId = "192.168.0.105";

    rankVec[2].rankId = 2;
    rankVec[2].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr3(1711319232);
    rankVec[2].deviceInfo.deviceIp.push_back(ipAddr3);
    rankVec[2].serverIdx = 1;
    rankVec[2].serverId = "192.168.0.102";
    rankVec[2].superPodId = "192.168.0.105";

    rankVec[3].rankId = 3;
    rankVec[3].deviceInfo.devicePhyId = 1;
    HcclIpAddress ipAddr4(1711319232);
    rankVec[3].deviceInfo.deviceIp.push_back(ipAddr4);
    rankVec[3].serverIdx = 1;
    rankVec[3].serverId = "192.168.0.102";
    rankVec[3].superPodId = "192.168.0.105";

    rankVec[4].rankId = 4;
    rankVec[4].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr5(1728096448);
    rankVec[4].deviceInfo.deviceIp.push_back(ipAddr5);
    rankVec[4].serverIdx = 2;
    rankVec[4].serverId = "192.168.0.103";
    rankVec[4].superPodId = "192.168.0.106";

    rankVec[5].rankId = 5;
    rankVec[5].deviceInfo.devicePhyId = 1;
    HcclIpAddress ipAddr6(1728096448);
    rankVec[5].deviceInfo.deviceIp.push_back(ipAddr6);
    rankVec[5].serverIdx = 2;
    rankVec[5].serverId = "192.168.0.103";
    rankVec[5].superPodId = "192.168.0.106";

    rankVec[6].rankId = 6;
    rankVec[6].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr7(1744873664);
    rankVec[6].deviceInfo.deviceIp.push_back(ipAddr7);
    rankVec[6].serverIdx = 3;
    rankVec[6].serverId = "192.168.0.104";
    rankVec[6].superPodId = "192.168.0.106";

    rankVec[7].rankId = 7;
    rankVec[7].deviceInfo.devicePhyId = 1;
    HcclIpAddress ipAddr8(1744873664);
    rankVec[7].deviceInfo.deviceIp.push_back(ipAddr8);
    rankVec[7].serverIdx = 3;
    rankVec[7].serverId = "192.168.0.104";
    rankVec[7].superPodId = "192.168.0.106";
    rankTable.rankList.assign(rankVec.begin(), rankVec.end());
    rankTable.rankNum = 8;
    rankTable.deviceNum = 8;
    rankTable.serverNum = 4;
}

TEST_F(HcclImplAlgTestDoubleRingConcurrentallreduce, ut_CollAllReduceDoubleRingConcurrentExecutor_Ring)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
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
    implBase->InitCCLbuffer(1024, 1024);

    MOCKER_CPP(&AlgConfigurator::IsHCCSSWNumEqualToTwiceSIONum).stubs().will(returnValue(true));
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    impl->topoAttr_.deviceLogicId = 0;
    impl->topoAttr_.devicePhyId = 0;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;

    impl->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    algConfigurator->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;

    const std::vector<std::vector<u32>> tmpRingNics
        = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}};
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.devicePhyId = 0;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
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

    std::printf("[st_CollAllReduceDoubleRingConcurrentExecutor_Ring]");

    ret = implBase->AllReduceOutPlace(tag, inputMem.ptr(), outputMem.ptr(), count, dataType, op, stream.ptr());
    implBase = nullptr;
    SetFftsSwitch(true);
    InitEnvVarParam();
    GlobalMockObject::verify();
}

TEST_F(HcclImplAlgTestDoubleRingConcurrentallreduce, ut_CollAllReduceDoubleRingConcurrentExecutor_superpod_Ring)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);
    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    SetFftsSwitch(false);
    ret = InitEnvVarParam();
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam_2SurperPod(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910_93;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&AlgConfigurator::IsHCCSSWNumEqualToTwiceSIONum).stubs().will(returnValue(true));

    ret = implBase->AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->InitCCLbuffer(1024, 1024);
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    impl->topoAttr_.deviceLogicId = 0;
    impl->topoAttr_.devicePhyId = 0;

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

    MOCKER(LocalNotify::Wait).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(HcclD2DMemcpyAsync).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(AlgTemplateBase::ExecEmptyTask).stubs().will(returnValue(HCCL_SUCCESS));

    std::printf("[st_CollAllReduceDoubleRingConcurrentExecutor_Ring]");

    ret = implBase->AllReduceOutPlace(tag, inputMem.ptr(), outputMem.ptr(), count, dataType, op, stream.ptr());
    implBase = nullptr;
    SetFftsSwitch(true);
    InitEnvVarParam();
    GlobalMockObject::verify();
}

TEST_F(HcclImplAlgTestDoubleRingConcurrentallreduce, ut_CollAllReduceDoubleRingConcurrentExecutor_Ring_ffts)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    SetFftsSwitch(true);
    ret = InitEnvVarParam();
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910_93;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->InitCCLbuffer(1024, 1024);

    MOCKER_CPP(&AlgConfigurator::IsHCCSSWNumEqualToTwiceSIONum).stubs().will(returnValue(true));
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    impl->topoAttr_.deviceLogicId = 0;
    impl->topoAttr_.devicePhyId = 0;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;

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

    MOCKER(LocalNotify::Wait).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(HcclD2DMemcpyAsync).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(AlgTemplateBase::ExecEmptyTask).stubs().will(returnValue(HCCL_SUCCESS));

    std::printf("[st_CollAllReduceDoubleRingConcurrentExecutor_Ring]");

    ret = implBase->AllReduceOutPlace(tag, inputMem.ptr(), outputMem.ptr(), count, dataType, op, stream.ptr());
    implBase = nullptr;
    SetFftsSwitch(true);
    InitEnvVarParam();
    GlobalMockObject::verify();
}

TEST_F(HcclImplAlgTestDoubleRingConcurrentallreduce, ut_CollAllReduceDoubleRingConcurrentExecutor_Ring_1server)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);
    vector<HcclRtStream> streamList(1);
    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    ret = InitEnvVarParam();
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam_server(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910_93;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->InitCCLbuffer(1024, 1024);

    MOCKER_CPP(&AlgConfigurator::IsHCCSSWNumEqualToTwiceSIONum).stubs().will(returnValue(true));
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u64 memSize = 55230336;
    void* memPtr = nullptr;
    ret = hrtMalloc(&memPtr, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    impl->topoAttr_.deviceLogicId = 0;
    impl->topoAttr_.devicePhyId = 0;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
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

    MOCKER(LocalNotify::Wait).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(HcclD2DMemcpyAsync).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(AlgTemplateBase::ExecEmptyTask).stubs().will(returnValue(HCCL_SUCCESS));

    std::printf("[st_CollAllReduceDoubleRingConcurrentExecutor_Ring]");
    ret = implBase->AllReduceOutPlace(tag, inputMem.ptr(), outputMem.ptr(), count, dataType, op, stream.ptr());
    implBase = nullptr;
    GlobalMockObject::verify();
}

TEST_F(HcclImplAlgTestDoubleRingConcurrentallreduce, ut_CollAllReduceDoubleRingConcurrentExecutor_Ring_Largesize)
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

    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    impl->topoAttr_.deviceLogicId = 0;
    impl->topoAttr_.devicePhyId = 0;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
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

    std::printf("[st_CollAllReduceDoubleRingConcurrentExecutor_Ring_Largesize]");

    ret = implBase->AllReduceOutPlace(tag, inputMem.ptr(), outputMem.ptr(), count, dataType, op, stream.ptr());
    implBase = nullptr;
    SetFftsSwitch(false);
    InitEnvVarParam();
    GlobalMockObject::verify();
}

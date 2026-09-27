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
#include <stdio.h>

#include "hccl/base.h"
#include <hccl/hccl_types.h>
#include "llt_hccl_stub_pub.h"
#include "dlra_function.h"

#define private public
#define protected public
#include "hccl_alg.h"
#include "hccl_impl.h"
#include "hccl_communicator.h"
#include "hccl_comm_pub.h"
#include "comm_impl.h"
#include "alg_template_base_pub.h"
#include "broadcast_operator.h"
#include "coll_broadcast_ring_executor.h"
#include "coll_broadcast_mesh_executor.h"
#include "coll_broadcast_ring_for_910_93_executor.h"
#include "coll_broadcast_comm_executor.h"
#include "coll_broadcast_for_310p_comm_executor.h"
#include "coll_broadcast_plus_broadcast.h"
#include "coll_broadcast_smallcount_executor.h"
#include "coll_comm_executor.h"
#include "dispatcher_pub.h"
#include "comm_factory.h"
#include "externalinput.h"

#undef private
#undef protected
using namespace std;
using namespace hccl;

class CollBroadcastInterTest : public testing::Test {
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
        std::cout << "\033[36m--CollBroadcastInterTest SetUP--\033[0m" << std::endl;
    }
    static void TearDownTestCase()
    {
        if (dispatcherPtr != nullptr) {
            s32 ret = HcclDispatcherDestroy(dispatcherPtr);
            EXPECT_EQ(ret, HCCL_SUCCESS);
            dispatcherPtr = nullptr;
            dispatcher = nullptr;
        }
        std::cout << "\033[36m--CollBroadcastInterTest TearDown--\033[0m" << std::endl;
    }
    virtual void SetUp()
    {
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        HcclOpMetaInfo meta;
        bool hasMassTasks = true;
        hccl::Stream stream;
        ::InitTask(dispatcherPtr, stream, meta.isEnableCache, meta.GetCacheKey(), false);
        if (hasMassTasks) {
            SetNormalMode(dispatcherPtr);
        }
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

#if 1
TEST_F(CollBroadcastInterTest, broadcast_ring)
{
    MOCKER(hrtCtxSetCurrent).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    CollBroadcastRingExecutor* executor = new CollBroadcastRingExecutor(impl->dispatcher_, topoMatcher);
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(2048);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 2048;
    opParam.DataDes.count = 2048 / 4;
    opParam.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);
    opParam.root = 0;

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(
        opParam.tag, HcclCMDType::HCCL_CMD_BROADCAST, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete executor;
}
#endif

#if 1
TEST_F(CollBroadcastInterTest, broadcast_mesh)
{
    MOCKER(hrtCtxSetCurrent).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    CollBroadcastMeshExecutor* executor = new CollBroadcastMeshExecutor(impl->dispatcher_, topoMatcher);

    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(2048);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 2048;
    opParam.DataDes.count = 2048 / 4;
    opParam.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);
    opParam.root = 0;

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    resourceRequest.streamNum = 0;
    resourceRequest.notifyNum = 0;
    implBase->AllocAlgResource(
        opParam.tag, HcclCMDType::HCCL_CMD_BROADCAST, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete executor;
}
#endif

#if 1
TEST_F(CollBroadcastInterTest, broadcast_smallcount)
{
    MOCKER(hrtCtxSetCurrent).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    CollBroadcastSmallCountExecutor* executor = new CollBroadcastSmallCountExecutor(impl->dispatcher_, topoMatcher);

    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(2048);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 2048;
    opParam.DataDes.count = 2048 / 4;
    opParam.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);
    opParam.root = 0;

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    resourceRequest.streamNum = 0;
    resourceRequest.notifyNum = 0;
    implBase->AllocAlgResource(
        opParam.tag, HcclCMDType::HCCL_CMD_BROADCAST, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete executor;
}
#endif

#if 1
TEST_F(CollBroadcastInterTest, broadcast_common)
{
    MOCKER(hrtCtxSetCurrent).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    CollBroadcastCommExecutor* executor = new CollBroadcastCommExecutor(impl->dispatcher_, topoMatcher);
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(2048);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 2048;
    opParam.DataDes.count = 2048 / 4;
    opParam.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);
    opParam.root = 0;

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(
        opParam.tag, HcclCMDType::HCCL_CMD_BROADCAST, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete executor;
}
#endif

HcclDispatcher CollBroadcastInterTest::dispatcherPtr = nullptr;
DispatcherPub* CollBroadcastInterTest::dispatcher = nullptr;

static void TestConstructParam_91093(HcclCommParams& params, RankTable_t& rankTable)
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

TEST_F(CollBroadcastInterTest, broadcast_superpod_Ring)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);
    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    setenv("HCCL_CONCURRENT_ENABLE", "1", 1);
    setenv("HCCL_OP_RETRY_ENABLE", "L1:0,L2:0", 1);
    SetFftsSwitch(true);
    ret = InitEnvVarParam();
    EXPECT_EQ(GetExternalInputEnableRdmaSdmaConcurrent(), true);
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
    u32 root = 0;
    HcclRtStream opStream;
    HcclCommunicator communicator;
    ret = communicator.Init(params, rankTable);

    std::printf("[ut_CollAllReduceDoubleRingConcurrentExecutor_Ring]");
    ret = communicator.BroadcastOutPlace(tag, inputMem.ptr(), count, dataType, root, opStream);
    implBase = nullptr;
    SetFftsSwitch(true);
    InitEnvVarParam();
    GlobalMockObject::verify();
}

TEST_F(CollBroadcastInterTest, broadcast_for_310P3_comm)
{
    MOCKER(hrtCtxSetCurrent).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));

    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_310P3;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    CollBroadcastFor310PCommExecutor* executor = new CollBroadcastFor310PCommExecutor(impl->dispatcher_, topoMatcher);
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(2048);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 2048;
    opParam.DataDes.count = 2048 / 4;
    opParam.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);
    opParam.root = 0;

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(
        opParam.tag, HcclCMDType::HCCL_CMD_BROADCAST, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete executor;
}

TEST_F(CollBroadcastInterTest, ut_bcast_plus_bcast)
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

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_BROADCAST].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_8P_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_BROADCAST].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NB;

    impl->topoType_ = TopoType::TOPO_TYPE_8P_RING;
    algConfigurator->topoType_ = TopoType::TOPO_TYPE_8P_RING;

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);

    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_8P_RING;
    CollBroadcastPlusBroadcast* executor = new CollBroadcastPlusBroadcast(impl->dispatcher_, topoMatcher);

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

    MOCKER_CPP(&CollCommExecutor::CheckCommSize).stubs().will(returnValue(HCCL_SUCCESS));
    SubCommInfo mockCommInfo{0, 8, std::vector<LINK>()};
    MOCKER_CPP(&CollNativeExecutorBase::GetSubCommInfo).stubs().will(returnValue(mockCommInfo));
    MOCKER_CPP(&CollNativeExecutorBase::GetRankByUserRank).stubs().will(returnValue(HCCL_SUCCESS));

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(
        opParam.tag, HcclCMDType::HCCL_CMD_BROADCAST, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    ret = executor->Orchestrate(opParam, resourceResponse);

    implBase = nullptr;

    delete executor;
    GlobalMockObject::verify();
}

TEST_F(CollBroadcastInterTest, ut_bcast_plus_bcast_NB)
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

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_BROADCAST].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_8P_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_BROADCAST].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NB;

    impl->topoType_ = TopoType::TOPO_TYPE_8P_RING;
    algConfigurator->topoType_ = TopoType::TOPO_TYPE_8P_RING;

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);

    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_8P_RING;
    CollBroadcastPlusBroadcast* executor = new CollBroadcastPlusBroadcast(impl->dispatcher_, topoMatcher);

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

    MOCKER_CPP(&CollCommExecutor::CheckCommSize).stubs().will(returnValue(HCCL_SUCCESS));
    SubCommInfo mockCommInfo{0, 8, std::vector<LINK>()};
    MOCKER_CPP(&CollNativeExecutorBase::GetSubCommInfo).stubs().will(returnValue(mockCommInfo));
    MOCKER_CPP(&CollNativeExecutorBase::GetRankByUserRank).stubs().will(returnValue(HCCL_SUCCESS));

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(
        opParam.tag, HcclCMDType::HCCL_CMD_BROADCAST, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    ret = executor->Orchestrate(opParam, resourceResponse);

    implBase = nullptr;

    delete executor;
    GlobalMockObject::verify();
}

TEST_F(CollBroadcastInterTest, broadcast_test_910B)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    impl->deviceLogicId_ = 0;
    impl->devicePhyId_ = 0;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_BROADCAST].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_4P_MESH;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_BROADCAST].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
    impl->topoType_ = TopoType::TOPO_TYPE_4P_MESH;
    algConfigurator->topoType_ = TopoType::TOPO_TYPE_4P_MESH;
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(2048);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 2048;
    opParam.DataDes.count = 2048 / 4;
    opParam.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);
    opParam.root = 0;
    std::string algName;
    std::string newTag;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;
    std::unique_ptr<BroadCastOperator> operation(
        new (std::nothrow) BroadCastOperator(algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher));
    ret = operation->SelectAlg("", opParam, algName, newTag);
    operation = nullptr;
}

TEST_F(CollBroadcastInterTest, broadcast_test_310P)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_310P3;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    impl->deviceLogicId_ = 0;
    impl->devicePhyId_ = 0;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_BROADCAST].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_BROADCAST].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_HD;
    impl->topoType_ = TopoType::TOPO_TYPE_NP_SINGLE_RING;
    algConfigurator->topoType_ = TopoType::TOPO_TYPE_NP_SINGLE_RING;
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(2048);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 2048;
    opParam.DataDes.count = 2048 / 4;
    opParam.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);
    opParam.root = 0;
    std::string algName;
    std::string newTag;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;
    std::unique_ptr<BroadCastOperator> operation(
        new (std::nothrow) BroadCastOperator(algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher));
    ret = operation->SelectAlg("", opParam, algName, newTag);

    operation = nullptr;
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

TEST_F(CollBroadcastInterTest, broadcast_double_ring_1)
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

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    std::vector<std::vector<RankInfo>> commPlaneVectorL2 = impl->commFactory_->CommPlaneVector_[COMM_LEVEL2];
    impl->commFactory_->CommPlaneVector_[COMM_LEVEL2].push_back(commPlaneVectorL2[0]);
    std::vector<RankInfo> rankInfoVectorL1 = impl->commFactory_->CommPlaneVector_[COMM_LEVEL1][COMM_INDEX_0];
    impl->commFactory_->CommPlaneVector_[COMM_LEVEL1][COMM_INDEX_0].push_back(rankInfoVectorL1[COMM_INDEX_0]);

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910_93;
    impl->devicePhyId_ = 0;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_BROADCAST].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_BROADCAST].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
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
    opParam.outputPtr = inputMem.ptr();
    opParam.outputSize = count * 4;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.root = 0;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollNativeExecutorBase::CheckCommSize).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollCommExecutor::MultiRingScatter).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollCommExecutor::MultiRingAllGather).stubs().will(returnValue(HCCL_SUCCESS));

    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    CollBroadCastRingFor91093* executor = new CollBroadCastRingFor91093(impl->dispatcher_, topoMatcher);

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(
        opParam.tag, HcclCMDType::HCCL_CMD_BROADCAST, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    ret = executor->Orchestrate(opParam, resourceResponse);

    implBase = nullptr;

    delete executor;
    GlobalMockObject::verify();
}

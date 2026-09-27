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
#include "mem_device_pub.h"
#define private public
#define protected public
#include "hccl_impl.h"
#include "hccl_communicator.h"
#include "dispatcher_pub.h"
#include "coll_all_to_all_executor.h"
#undef private
#undef protected
#include "adapter_prof.h"
#include "sal.h"
#include "llt_hccl_stub_pub.h"
#include "externalinput.h"
#include "dlra_function.h"

using namespace hccl;
using namespace std;

class HcclImplAlltoAllAIVTest : public testing::Test {
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
        std::cout << "HcclImplAlltoAllAIVTest SetUP" << std::endl;
        TestConstructParam(params, rankTable);
    }
    static void TearDownTestCase()
    {
        if (dispatcherPtr != nullptr) {
            s32 ret = HcclDispatcherDestroy(dispatcherPtr);
            EXPECT_EQ(ret, HCCL_SUCCESS);
            dispatcherPtr = nullptr;
            dispatcher = nullptr;
        }
        std::cout << "HcclImplAlltoAllAIVTest TearDown" << std::endl;
    }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
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
    static void TestConstructParam(HcclCommParams& params, RankTable_t& rankTable)
    {
        string commId = "comm ";
        memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1);
        params.rank = 0;
        params.totalRanks = 2;
        params.isHeterogComm = false;
        params.logicDevId = 0;
        params.commWorkMode = WorkMode::HCCL_MODE_NORMAL;
        params.deviceType = DevType::DEV_TYPE_910B;

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
    static HcclCommParams params;
    static RankTable_t rankTable;

    static HcclDispatcher dispatcherPtr;
    static DispatcherPub* dispatcher;
};
HcclDispatcher HcclImplAlltoAllAIVTest::dispatcherPtr = nullptr;
DispatcherPub* HcclImplAlltoAllAIVTest::dispatcher = nullptr;

HcclCommParams HcclImplAlltoAllAIVTest::params;
RankTable_t HcclImplAlltoAllAIVTest::rankTable;

#define DEV_NUM_4 4
#define DEV_NUM_8 8

// 8p AlltoAllV
TEST_F(HcclImplAlltoAllAIVTest, ut_alltoallv_8p_mesh_aiv)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test_test_test";
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 512;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::InitNic).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::RegisterToHeartBeat, HcclResult(HcclCommunicator::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AlltoAllOperator::IsSatisfyAlltoAllAivCondition).stubs().will(returnValue(true));

    ret = implBase->AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910B;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLTOALLV].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_MESH;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLTOALLV].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_HD;

    impl->topoType_ = TopoType::TOPO_TYPE_NP_MESH;
    algConfigurator->topoType_ = TopoType::TOPO_TYPE_NP_MESH;
    // 相当于插入个桩，强制设置 isSingleMeshAggregation_ 的值
    impl->isSingleMeshAggregation_ = true;
    algConfigurator->topoAttr_.isSingleMeshAggregation = true;

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910B;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_NP_MESH;
    topoMatcher->topoInfo_.isSingleMeshAggregation = true;

    u64* sendCounts = (u64*)sal_malloc(DEV_NUM_8 * sizeof(u64));
    u64* sdispls = (u64*)sal_malloc(DEV_NUM_8 * sizeof(u64));
    u64* recvCounts = (u64*)sal_malloc(DEV_NUM_8 * sizeof(u64));
    u64* rdispls = (u64*)sal_malloc(DEV_NUM_8 * sizeof(u64));

    u64* sendCountMatrix = (u64*)sal_malloc(DEV_NUM_8 * DEV_NUM_8 * sizeof(u64));

    HcclDataType sendDataType = HCCL_DATA_TYPE_INT8;
    HcclDataType recvDataType = HCCL_DATA_TYPE_INT8;

    /** 初始化输入输出缓存 */

    for (u32 i = 0; i < DEV_NUM_8; i++) {
        for (u32 j = 0; j < DEV_NUM_8; j++) {
            sendCountMatrix[i * DEV_NUM_8 + j] = count;
        }
        sendCounts[i] = count;
        sdispls[i] = i * count;
        recvCounts[i] = count;
        rdispls[i] = i * count;
    }

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));

    u64 memSize = 0;
    ret = implBase->GetAlltoAllStagedWorkSpaceMemSize(
        sendCounts, sdispls, sendDataType, recvCounts, rdispls, recvDataType, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(memSize, 0);

    MOCKER_CPP(&HcclSocket::Listen, HcclResult(HcclSocket::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    ret = implBase->AlltoAllV(
        inputMem.ptr(), (void*)sendCounts, (void*)sdispls, sendDataType, outputMem.ptr(), (void*)recvCounts,
        (void*)rdispls, recvDataType, stream.ptr(), tag);

    EXPECT_EQ(ret, HCCL_SUCCESS);

    implBase = nullptr;
    HCCL_RELEASE_PTR_AND_SET_NULL(sendCounts);
    HCCL_RELEASE_PTR_AND_SET_NULL(sdispls);
    HCCL_RELEASE_PTR_AND_SET_NULL(recvCounts);
    HCCL_RELEASE_PTR_AND_SET_NULL(rdispls);
    HCCL_RELEASE_PTR_AND_SET_NULL(sendCountMatrix);

    GlobalMockObject::verify();
}

TEST_F(HcclImplAlltoAllAIVTest, ut_alltoallv_8p_mesh_aiv_capture)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test_test_test";
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 512;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::InitNic).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::RegisterToHeartBeat, HcclResult(HcclCommunicator::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AlltoAllOperator::IsSatisfyAlltoAllAivCondition).stubs().will(returnValue(true));
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
    int mockModel = 0;
    void* pmockModel = &mockModel;
    MOCKER(aclmdlRICaptureGetInfo)
        .stubs()
        .with(
            mockcpp::any(), outBoundP(&captureStatus, sizeof(captureStatus)),
            outBoundP(&pmockModel, sizeof(pmockModel)))
        .will(returnValue(0));

    ret = implBase->AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910B;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLTOALLV].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_MESH;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLTOALLV].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_HD;

    impl->topoType_ = TopoType::TOPO_TYPE_NP_MESH;
    algConfigurator->topoType_ = TopoType::TOPO_TYPE_NP_MESH;
    // 相当于插入个桩，强制设置 isSingleMeshAggregation_ 的值
    impl->isSingleMeshAggregation_ = true;
    algConfigurator->topoAttr_.isSingleMeshAggregation = true;

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910B;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_NP_MESH;
    topoMatcher->topoInfo_.isSingleMeshAggregation = true;

    u64* sendCounts = (u64*)sal_malloc(DEV_NUM_8 * sizeof(u64));
    u64* sdispls = (u64*)sal_malloc(DEV_NUM_8 * sizeof(u64));
    u64* recvCounts = (u64*)sal_malloc(DEV_NUM_8 * sizeof(u64));
    u64* rdispls = (u64*)sal_malloc(DEV_NUM_8 * sizeof(u64));

    u64* sendCountMatrix = (u64*)sal_malloc(DEV_NUM_8 * DEV_NUM_8 * sizeof(u64));

    HcclDataType sendDataType = HCCL_DATA_TYPE_INT8;
    HcclDataType recvDataType = HCCL_DATA_TYPE_INT8;

    /** 初始化输入输出缓存 */

    for (u32 i = 0; i < DEV_NUM_8; i++) {
        for (u32 j = 0; j < DEV_NUM_8; j++) {
            sendCountMatrix[i * DEV_NUM_8 + j] = count;
        }
        sendCounts[i] = count;
        sdispls[i] = i * count;
        recvCounts[i] = count;
        rdispls[i] = i * count;
    }

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));

    u64 memSize = 0;
    ret = implBase->GetAlltoAllStagedWorkSpaceMemSize(
        sendCounts, sdispls, sendDataType, recvCounts, rdispls, recvDataType, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(memSize, 0);

    ret = implBase->AlltoAllV(
        inputMem.ptr(), (void*)sendCounts, (void*)sdispls, sendDataType, outputMem.ptr(), (void*)recvCounts,
        (void*)rdispls, recvDataType, stream.ptr(), tag);

    EXPECT_EQ(ret, HCCL_SUCCESS);

    implBase = nullptr;
    HCCL_RELEASE_PTR_AND_SET_NULL(sendCounts);
    HCCL_RELEASE_PTR_AND_SET_NULL(sdispls);
    HCCL_RELEASE_PTR_AND_SET_NULL(recvCounts);
    HCCL_RELEASE_PTR_AND_SET_NULL(rdispls);
    HCCL_RELEASE_PTR_AND_SET_NULL(sendCountMatrix);

    GlobalMockObject::verify();
}

TEST_F(HcclImplAlltoAllAIVTest, ut_alltoallv_8p_mesh_aiv_capture_multi_op)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test_test_test";
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 512;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::InitNic).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::RegisterToHeartBeat, HcclResult(HcclCommunicator::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AlltoAllOperator::IsSatisfyAlltoAllAivCondition).stubs().will(returnValue(true));
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
    int mockModel = 0;
    void* pmockModel = &mockModel;
    MOCKER(aclmdlRICaptureGetInfo)
        .stubs()
        .with(
            mockcpp::any(), outBoundP(&captureStatus, sizeof(captureStatus)),
            outBoundP(&pmockModel, sizeof(pmockModel)))
        .will(returnValue(0));

    ret = implBase->AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910B;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLTOALLV].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_MESH;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLTOALLV].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_HD;

    impl->topoType_ = TopoType::TOPO_TYPE_NP_MESH;
    algConfigurator->topoType_ = TopoType::TOPO_TYPE_NP_MESH;
    // 相当于插入个桩，强制设置 isSingleMeshAggregation_ 的值
    impl->isSingleMeshAggregation_ = true;
    algConfigurator->topoAttr_.isSingleMeshAggregation = true;

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910B;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_NP_MESH;
    topoMatcher->topoInfo_.isSingleMeshAggregation = true;

    u64* sendCounts = (u64*)sal_malloc(DEV_NUM_8 * sizeof(u64));
    u64* sdispls = (u64*)sal_malloc(DEV_NUM_8 * sizeof(u64));
    u64* recvCounts = (u64*)sal_malloc(DEV_NUM_8 * sizeof(u64));
    u64* rdispls = (u64*)sal_malloc(DEV_NUM_8 * sizeof(u64));

    u64* sendCountMatrix = (u64*)sal_malloc(DEV_NUM_8 * DEV_NUM_8 * sizeof(u64));

    HcclDataType sendDataType = HCCL_DATA_TYPE_INT8;
    HcclDataType recvDataType = HCCL_DATA_TYPE_INT8;

    /** 初始化输入输出缓存 */

    for (u32 i = 0; i < DEV_NUM_8; i++) {
        for (u32 j = 0; j < DEV_NUM_8; j++) {
            sendCountMatrix[i * DEV_NUM_8 + j] = count;
        }
        sendCounts[i] = count;
        sdispls[i] = i * count;
        recvCounts[i] = count;
        rdispls[i] = i * count;
    }

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::IsEnableBackupLink).stubs().will(returnValue(false));

    u64 memSize = 0;
    ret = implBase->GetAlltoAllStagedWorkSpaceMemSize(
        sendCounts, sdispls, sendDataType, recvCounts, rdispls, recvDataType, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(memSize, 0);

    ret = implBase->AlltoAllV(
        inputMem.ptr(), (void*)sendCounts, (void*)sdispls, sendDataType, outputMem.ptr(), (void*)recvCounts,
        (void*)rdispls, recvDataType, stream.ptr(), tag);

    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = implBase->AlltoAllV(
        inputMem.ptr(), (void*)sendCounts, (void*)sdispls, sendDataType, outputMem.ptr(), (void*)recvCounts,
        (void*)rdispls, recvDataType, stream.ptr(), tag);

    EXPECT_EQ(ret, HCCL_SUCCESS);

    implBase = nullptr;
    HCCL_RELEASE_PTR_AND_SET_NULL(sendCounts);
    HCCL_RELEASE_PTR_AND_SET_NULL(sdispls);
    HCCL_RELEASE_PTR_AND_SET_NULL(recvCounts);
    HCCL_RELEASE_PTR_AND_SET_NULL(rdispls);
    HCCL_RELEASE_PTR_AND_SET_NULL(sendCountMatrix);

    GlobalMockObject::verify();
}

// 8p AlltoAllVC
TEST_F(HcclImplAlltoAllAIVTest, ut_alltoallvc_8p_mesh_aiv)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test_test_test_loop";
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 512;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::InitNic).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::RegisterToHeartBeat, HcclResult(HcclCommunicator::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AlltoAllOperator::IsSatisfyAlltoAllAivCondition).stubs().will(returnValue(true));

    ret = implBase->AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910B;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_BROADCAST].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_MESH;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_BROADCAST].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_HD;

    impl->topoType_ = TopoType::TOPO_TYPE_NP_MESH;
    algConfigurator->topoType_ = TopoType::TOPO_TYPE_NP_MESH;
    // 相当于插入个桩，强制设置 isSingleMeshAggregation_ 的值
    impl->isSingleMeshAggregation_ = true;
    algConfigurator->topoAttr_.isSingleMeshAggregation = true;

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910B;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_NP_MESH;
    topoMatcher->topoInfo_.isSingleMeshAggregation = true;

    u64* sendCounts = (u64*)sal_malloc(DEV_NUM_8 * sizeof(u64));
    u64* sdispls = (u64*)sal_malloc(DEV_NUM_8 * sizeof(u64));
    u64* recvCounts = (u64*)sal_malloc(DEV_NUM_8 * sizeof(u64));
    u64* rdispls = (u64*)sal_malloc(DEV_NUM_8 * sizeof(u64));

    u64* sendCountMatrix = (u64*)sal_malloc(DEV_NUM_8 * DEV_NUM_8 * sizeof(u64));

    HcclDataType sendDataType = HCCL_DATA_TYPE_INT8;
    HcclDataType recvDataType = HCCL_DATA_TYPE_INT8;

    /** 初始化输入输出缓存 */

    for (u32 i = 0; i < DEV_NUM_8; i++) {
        for (u32 j = 0; j < DEV_NUM_8; j++) {
            sendCountMatrix[i * DEV_NUM_8 + j] = count;
        }
        sendCounts[i] = count;
        sdispls[i] = i * count;
        recvCounts[i] = count;
        rdispls[i] = i * count;
    }

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));

    ret = implBase->AlltoAllVC(
        inputMem.ptr(), (void*)sendCountMatrix, sendDataType, outputMem.ptr(), recvDataType, stream.ptr(), tag);

    EXPECT_EQ(ret, HCCL_SUCCESS);

    implBase = nullptr;
    HCCL_RELEASE_PTR_AND_SET_NULL(sendCounts);
    HCCL_RELEASE_PTR_AND_SET_NULL(sdispls);
    HCCL_RELEASE_PTR_AND_SET_NULL(recvCounts);
    HCCL_RELEASE_PTR_AND_SET_NULL(rdispls);
    HCCL_RELEASE_PTR_AND_SET_NULL(sendCountMatrix);

    GlobalMockObject::verify();
}

#if 0
// rmda AlltoAll
TEST_F(HcclImplAlltoAllAIVTest, ut_alltoall_rdma_mesh_aiv)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test_test_test";
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 512;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER(GetExternalInputHcclAivMode)
    .stubs()
    .will(returnValue(true));
    MOCKER_CPP(&HcclCommunicator::InitRaResource)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::InitNic)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::RegisterToHeartBeat, HcclResult(HcclCommunicator::*)())
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&AlltoAllOperator::IsSatisfyAlltoAllAivCondition)
    .stubs()
    .will(returnValue(true));
    MOCKER(CollAlltoAllExecutor::RunAlltoAllVTemplateStaged)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    ret = implBase->AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::unique_ptr<hcclImpl> &impl = implBase->implAlg_->pimpl_;

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910B;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLTOALL].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_MESH;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLTOALL].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_HD;
    impl->topoType_ = TopoType::TOPO_TYPE_NP_MESH;
    algConfigurator->topoType_ = TopoType::TOPO_TYPE_NP_MESH;
    // 相当于插入个桩，强制设置 isSingleMeshAggregation_ 的值
    impl->isSingleMeshAggregation_ = false;
    algConfigurator->topoAttr_.isSingleMeshAggregation = false;

    (void) SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    std::unique_ptr<TopoMatcher> &topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910B;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_NP_MESH;
    topoMatcher->topoInfo_.isSingleMeshAggregation = false;

    HcclDataType sendDataType = HCCL_DATA_TYPE_INT8;
    HcclDataType recvDataType = HCCL_DATA_TYPE_INT8;

    MOCKER_CPP(&TransportManager::Alloc)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    ret = implBase->AlltoAll(inputMem.ptr(), count, sendDataType, outputMem.ptr(), count,
        recvDataType, stream.ptr(), tag);

    EXPECT_EQ(ret, HCCL_SUCCESS);

    implBase = nullptr;

    GlobalMockObject::verify();
}
#endif

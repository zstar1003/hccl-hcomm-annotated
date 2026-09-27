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
#include "hccl_communicator.h"
#include "hccl_comm_pub.h"
#include "comm_impl.h"
#include "coll_alg_operator.h"
#undef private
#undef protected
#include "adapter_prof.h"
#include "externalinput.h"

using namespace std;
using namespace hccl;

class CollAlgOperatorTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        DlRaFunction::GetInstance().DlRaFunctionInit();
        std::cout << "\033[36m--CollAlgOperatorTest SetUP--\033[0m" << std::endl;
    }
    static void TearDownTestCase() { std::cout << "\033[36m--CollAlgOperatorTest TearDown--\033[0m" << std::endl; }
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

TEST_F(CollAlgOperatorTest, is_2u2p_infer)
{
    CallBackInitRts();
    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;

    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    std::unique_ptr<CollAlgOperator> algOperator(new (std::nothrow) CollAlgOperator(
        algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_ALLREDUCE));
    algOperator->Is2U2PInfer();
    GlobalMockObject::verify();
}

TEST_F(CollAlgOperatorTest, need_create_single_mesh_plane)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    std::unique_ptr<CollAlgOperator> algOperator(new (std::nothrow) CollAlgOperator(
        algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_ALLREDUCE));
    algOperator->NeedCreateSingleMeshPlane(true);
    GlobalMockObject::verify();
}

TEST_F(CollAlgOperatorTest, ut_SupportRetryWithInplaceCheck_ptr_check)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    std::unique_ptr<CollAlgOperator> algOperator(new (std::nothrow) CollAlgOperator(
        algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_ALLREDUCE));

    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    DeviceMem scratchMem = DeviceMem::alloc(8192);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 1024;
    opParam.outputPtr = outputMem.ptr();
    opParam.DataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP32;

    u8 isInplaceStatus = 0;
    InplaceSupportRetryStatus inPlaceSupportRetryStatus = InplaceSupportRetryStatus::INPLACE_STATUS_END;
    std::string algName = "AllReduceMeshSmallCountExecutor";
    bool inplaceSupportRetry = false;

    opParam.inputPtr = nullptr;
    inplaceSupportRetry = algOperator->SupportRetryWithInplaceCheck(
        HcclCMDType::HCCL_CMD_ALLREDUCE, opParam, algName, isInplaceStatus, inPlaceSupportRetryStatus);
    EXPECT_EQ(inplaceSupportRetry, true);
    opParam.inputPtr = inputMem.ptr();

    opParam.outputPtr = nullptr;
    inplaceSupportRetry = algOperator->SupportRetryWithInplaceCheck(
        HcclCMDType::HCCL_CMD_ALLREDUCE, opParam, algName, isInplaceStatus, inPlaceSupportRetryStatus);
    EXPECT_EQ(inplaceSupportRetry, true);
    opParam.outputPtr = outputMem.ptr();
    GlobalMockObject::verify();
}

TEST_F(CollAlgOperatorTest, ut_SupportRetryWithInplaceCheck_opType_check)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    std::unique_ptr<CollAlgOperator> algOperator(new (std::nothrow) CollAlgOperator(
        algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_ALLREDUCE));

    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    DeviceMem scratchMem = DeviceMem::alloc(8192);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.outputPtr = outputMem.ptr();
    opParam.DataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    opParam.DataDes.count = 1024;
    opParam.root = 0;

    u8 isInplaceStatus = 0;
    InplaceSupportRetryStatus inPlaceSupportRetryStatus = InplaceSupportRetryStatus::INPLACE_STATUS_END;
    std::string algName = "dummyExecutor";
    bool inplaceSupportRetry = false;
    std::cout << "HCCL_CMD_SEND test" << std::endl;
    inplaceSupportRetry = algOperator->SupportRetryWithInplaceCheck(
        HcclCMDType::HCCL_CMD_SEND, opParam, algName, isInplaceStatus, inPlaceSupportRetryStatus);
    EXPECT_EQ(inplaceSupportRetry, true);
    std::cout << "HCCL_CMD_RECEIVE test" << std::endl;
    inplaceSupportRetry = algOperator->SupportRetryWithInplaceCheck(
        HcclCMDType::HCCL_CMD_RECEIVE, opParam, algName, isInplaceStatus, inPlaceSupportRetryStatus);
    EXPECT_EQ(inplaceSupportRetry, true);

    std::cout << "HCCL_CMD_ALLREDUCE test" << std::endl;
    inplaceSupportRetry = algOperator->SupportRetryWithInplaceCheck(
        HcclCMDType::HCCL_CMD_ALLREDUCE, opParam, algName, isInplaceStatus, inPlaceSupportRetryStatus);
    EXPECT_EQ(inplaceSupportRetry, true);

    opParam.inputPtr = inputMem.ptr();
    opParam.outputPtr = inputMem.ptr();
    inplaceSupportRetry = algOperator->SupportRetryWithInplaceCheck(
        HcclCMDType::HCCL_CMD_ALLREDUCE, opParam, algName, isInplaceStatus, inPlaceSupportRetryStatus);
    EXPECT_EQ(inplaceSupportRetry, false);
    opParam.inputPtr = inputMem.ptr();
    opParam.outputPtr = outputMem.ptr();

    std::cout << "HCCL_CMD_REDUCE test" << std::endl;
    std::cout << "algConfigurator->GetTopoAttr().userRank: " << algConfigurator->GetTopoAttr().userRank << std::endl;
    std::cout << "opParam.root: " << opParam.root << std::endl;
    inplaceSupportRetry = algOperator->SupportRetryWithInplaceCheck(
        HcclCMDType::HCCL_CMD_REDUCE, opParam, algName, isInplaceStatus, inPlaceSupportRetryStatus);
    EXPECT_EQ(inplaceSupportRetry, true);

    opParam.root = 1;
    std::cout << "algConfigurator->GetTopoAttr().userRank: " << algConfigurator->GetTopoAttr().userRank << std::endl;
    std::cout << "opParam.root: " << opParam.root << std::endl;
    inplaceSupportRetry = algOperator->SupportRetryWithInplaceCheck(
        HcclCMDType::HCCL_CMD_REDUCE, opParam, algName, isInplaceStatus, inPlaceSupportRetryStatus);
    EXPECT_EQ(inplaceSupportRetry, true);
    opParam.root = 0;

    std::cout << "HCCL_CMD_ALLGATHER test" << std::endl;
    opParam.DataDes.count = 256;
    std::cout << "algConfigurator->GetTopoAttr().userRankSize: " << algConfigurator->GetTopoAttr().userRankSize
              << std::endl;
    inplaceSupportRetry = algOperator->SupportRetryWithInplaceCheck(
        HcclCMDType::HCCL_CMD_ALLGATHER, opParam, algName, isInplaceStatus, inPlaceSupportRetryStatus);
    EXPECT_EQ(inplaceSupportRetry, true);
    opParam.DataDes.count = 1024;

    std::cout << "HCCL_CMD_REDUCE_SCATTER test" << std::endl;
    opParam.DataDes.count = 256;
    std::cout << "algConfigurator->GetTopoAttr().userRankSize: " << algConfigurator->GetTopoAttr().userRankSize
              << std::endl;
    inplaceSupportRetry = algOperator->SupportRetryWithInplaceCheck(
        HcclCMDType::HCCL_CMD_REDUCE_SCATTER, opParam, algName, isInplaceStatus, inPlaceSupportRetryStatus);
    EXPECT_EQ(inplaceSupportRetry, true);
    opParam.DataDes.count = 1024;

    std::cout << "HCCL_CMD_GATHER test" << std::endl;
    std::cout << "algConfigurator->GetTopoAttr().userRank: " << algConfigurator->GetTopoAttr().userRank << std::endl;
    std::cout << "opParam.root: " << opParam.root << std::endl;
    opParam.DataDes.count = 256;
    inplaceSupportRetry = algOperator->SupportRetryWithInplaceCheck(
        HcclCMDType::HCCL_CMD_GATHER, opParam, algName, isInplaceStatus, inPlaceSupportRetryStatus);
    EXPECT_EQ(inplaceSupportRetry, true);
    opParam.DataDes.count = 1024;

    opParam.root = 1;
    std::cout << "algConfigurator->GetTopoAttr().userRank: " << algConfigurator->GetTopoAttr().userRank << std::endl;
    std::cout << "opParam.root: " << opParam.root << std::endl;
    inplaceSupportRetry = algOperator->SupportRetryWithInplaceCheck(
        HcclCMDType::HCCL_CMD_GATHER, opParam, algName, isInplaceStatus, inPlaceSupportRetryStatus);
    EXPECT_EQ(inplaceSupportRetry, true);
    opParam.root = 0;

    std::cout << "HCCL_CMD_SCATTER test" << std::endl;
    std::cout << "algConfigurator->GetTopoAttr().userRank: " << algConfigurator->GetTopoAttr().userRank << std::endl;
    std::cout << "opParam.root: " << opParam.root << std::endl;
    opParam.DataDes.count = 256;
    inplaceSupportRetry = algOperator->SupportRetryWithInplaceCheck(
        HcclCMDType::HCCL_CMD_SCATTER, opParam, algName, isInplaceStatus, inPlaceSupportRetryStatus);
    EXPECT_EQ(inplaceSupportRetry, true);
    opParam.DataDes.count = 1024;

    opParam.root = 1;
    std::cout << "algConfigurator->GetTopoAttr().userRank: " << algConfigurator->GetTopoAttr().userRank << std::endl;
    std::cout << "opParam.root: " << opParam.root << std::endl;
    inplaceSupportRetry = algOperator->SupportRetryWithInplaceCheck(
        HcclCMDType::HCCL_CMD_SCATTER, opParam, algName, isInplaceStatus, inPlaceSupportRetryStatus);
    EXPECT_EQ(inplaceSupportRetry, true);
    opParam.root = 0;

    std::cout << "HCCL_CMD_ALLTOALLV like test" << std::endl;
    inplaceSupportRetry = algOperator->SupportRetryWithInplaceCheck(
        HcclCMDType::HCCL_CMD_ALLTOALLV, opParam, algName, isInplaceStatus, inPlaceSupportRetryStatus);
    EXPECT_EQ(inplaceSupportRetry, true);
    inplaceSupportRetry = algOperator->SupportRetryWithInplaceCheck(
        HcclCMDType::HCCL_CMD_ALLTOALLVC, opParam, algName, isInplaceStatus, inPlaceSupportRetryStatus);
    EXPECT_EQ(inplaceSupportRetry, true);
    inplaceSupportRetry = algOperator->SupportRetryWithInplaceCheck(
        HcclCMDType::HCCL_CMD_ALLTOALL, opParam, algName, isInplaceStatus, inPlaceSupportRetryStatus);
    EXPECT_EQ(inplaceSupportRetry, true);

    opParam.inputPtr = inputMem.ptr();
    opParam.outputPtr = inputMem.ptr();
    inplaceSupportRetry = algOperator->SupportRetryWithInplaceCheck(
        HcclCMDType::HCCL_CMD_ALLTOALLV, opParam, algName, isInplaceStatus, inPlaceSupportRetryStatus);
    EXPECT_EQ(inplaceSupportRetry, false);
    opParam.inputPtr = inputMem.ptr();
    opParam.outputPtr = outputMem.ptr();

    GlobalMockObject::verify();
}

TEST_F(CollAlgOperatorTest, ut_SupportRetryWithInplaceCheck_retry_condition_check)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclWorkflowMode oldMode = GetWorkflowMode();
    ret = SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    setenv("HCCL_BUFFSIZE", "200", 1);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = implBase->CreateCommCCLbuffer();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    std::unique_ptr<CollAlgOperator> algOperator(new (std::nothrow) CollAlgOperator(
        algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_ALLREDUCE));

    void* commInputPtr = nullptr;
    u64 commInputSize = 0;
    cclBufferManager.GetInCCLbuffer(commInputPtr, commInputSize);
    std::cout << "InCCLbuffer: " << commInputSize << std::endl;
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    DeviceMem scratchMem = DeviceMem::alloc(8192);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.outputPtr = inputMem.ptr();
    opParam.DataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    opParam.DataDes.count = 1024;
    opParam.root = 0;

    u8 isInplaceStatus = 0;
    InplaceSupportRetryStatus inPlaceSupportRetryStatus = InplaceSupportRetryStatus::INPLACE_STATUS_END;
    std::string algName = "dummyExecutor";
    bool inplaceSupportRetry = false;

    std::cout << "HCCL_CMD_ALLGATHER test" << std::endl;
    std::cout << "algConfigurator->GetTopoAttr().userRankSize: " << algConfigurator->GetTopoAttr().userRankSize
              << std::endl;
    inplaceSupportRetry = algOperator->SupportRetryWithInplaceCheck(
        HcclCMDType::HCCL_CMD_ALLGATHER, opParam, algName, isInplaceStatus, inPlaceSupportRetryStatus);
    EXPECT_EQ(inplaceSupportRetry, true);
    std::cout << "HCCL_CMD_BROADCAST test" << std::endl;
    std::cout << "algConfigurator->GetTopoAttr().userRankSize: " << algConfigurator->GetTopoAttr().userRankSize
              << std::endl;
    inplaceSupportRetry = algOperator->SupportRetryWithInplaceCheck(
        HcclCMDType::HCCL_CMD_BROADCAST, opParam, algName, isInplaceStatus, inPlaceSupportRetryStatus);
    EXPECT_EQ(inplaceSupportRetry, true);

    std::cout << "HCCL_CMD_ALLREDUCE test" << std::endl;
    inplaceSupportRetry = algOperator->SupportRetryWithInplaceCheck(
        HcclCMDType::HCCL_CMD_ALLREDUCE, opParam, algName, isInplaceStatus, inPlaceSupportRetryStatus);
    EXPECT_EQ(inplaceSupportRetry, false);

    std::cout << "HCCL_CMD_ALLREDUCE test" << std::endl;
    std::cout << "algConfigurator->GetTopoAttr().userRankSize: " << algConfigurator->GetTopoAttr().userRankSize
              << std::endl;
    algName = "AllReduceMeshSmallCountExecutor";
    algOperator->SetRetryEnable(true);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    inplaceSupportRetry = algOperator->SupportRetryWithInplaceCheck(
        HcclCMDType::HCCL_CMD_ALLREDUCE, opParam, algName, isInplaceStatus, inPlaceSupportRetryStatus);
    std::cout << "isInplaceStatus: " << static_cast<u32>(isInplaceStatus) << std::endl;
    std::cout << "inPlaceSupportRetryStatus: " << static_cast<u32>(inPlaceSupportRetryStatus) << std::endl;
    EXPECT_EQ(inplaceSupportRetry, true);
    EXPECT_EQ(static_cast<u32>(inPlaceSupportRetryStatus), 1);
    algOperator->SetRetryEnable(false);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    inplaceSupportRetry = algOperator->SupportRetryWithInplaceCheck(
        HcclCMDType::HCCL_CMD_ALLREDUCE, opParam, algName, isInplaceStatus, inPlaceSupportRetryStatus);
    std::cout << "isInplaceStatus: " << static_cast<u32>(isInplaceStatus) << std::endl;
    std::cout << "inPlaceSupportRetryStatus: " << static_cast<u32>(inPlaceSupportRetryStatus) << std::endl;
    EXPECT_EQ(inplaceSupportRetry, false);
    EXPECT_EQ(static_cast<u32>(inPlaceSupportRetryStatus), 2);

    algName = "AllReduceComm";
    inplaceSupportRetry = algOperator->SupportRetryWithInplaceCheck(
        HcclCMDType::HCCL_CMD_ALLREDUCE, opParam, algName, isInplaceStatus, inPlaceSupportRetryStatus);
    std::cout << "isInplaceStatus: " << static_cast<u32>(isInplaceStatus) << std::endl;
    std::cout << "inPlaceSupportRetryStatus: " << static_cast<u32>(inPlaceSupportRetryStatus) << std::endl;
    EXPECT_EQ(inplaceSupportRetry, true);
    EXPECT_EQ(static_cast<u32>(inPlaceSupportRetryStatus), 3);

    algName = "AllReduceRingFor91093Executor";
    algOperator->SetRetryEnable(true);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    inplaceSupportRetry = algOperator->SupportRetryWithInplaceCheck(
        HcclCMDType::HCCL_CMD_ALLREDUCE, opParam, algName, isInplaceStatus, inPlaceSupportRetryStatus);
    std::cout << "isInplaceStatus: " << static_cast<u32>(isInplaceStatus) << std::endl;
    std::cout << "inPlaceSupportRetryStatus: " << static_cast<u32>(inPlaceSupportRetryStatus) << std::endl;
    EXPECT_EQ(inplaceSupportRetry, true);
    EXPECT_EQ(static_cast<u32>(inPlaceSupportRetryStatus), 4);
    algOperator->SetRetryEnable(false);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    inplaceSupportRetry = algOperator->SupportRetryWithInplaceCheck(
        HcclCMDType::HCCL_CMD_ALLREDUCE, opParam, algName, isInplaceStatus, inPlaceSupportRetryStatus);
    std::cout << "isInplaceStatus: " << static_cast<u32>(isInplaceStatus) << std::endl;
    std::cout << "inPlaceSupportRetryStatus: " << static_cast<u32>(inPlaceSupportRetryStatus) << std::endl;
    EXPECT_EQ(inplaceSupportRetry, false);
    EXPECT_EQ(static_cast<u32>(inPlaceSupportRetryStatus), 5);

    algName = "dummyExecutor";
    inplaceSupportRetry = algOperator->SupportRetryWithInplaceCheck(
        HcclCMDType::HCCL_CMD_ALLREDUCE, opParam, algName, isInplaceStatus, inPlaceSupportRetryStatus);
    std::cout << "isInplaceStatus: " << static_cast<u32>(isInplaceStatus) << std::endl;
    std::cout << "inPlaceSupportRetryStatus: " << static_cast<u32>(inPlaceSupportRetryStatus) << std::endl;
    EXPECT_EQ(inplaceSupportRetry, false);
    EXPECT_EQ(static_cast<u32>(inPlaceSupportRetryStatus), 6);

    opParam.DataDes.count = 209715200; // 200M count
    std::cout << "InCCLbuffer2: " << commInputSize << std::endl;
    inplaceSupportRetry = algOperator->SupportRetryWithInplaceCheck(
        HcclCMDType::HCCL_CMD_ALLREDUCE, opParam, algName, isInplaceStatus, inPlaceSupportRetryStatus);
    std::cout << "isInplaceStatus: " << static_cast<u32>(isInplaceStatus) << std::endl;
    std::cout << "inPlaceSupportRetryStatus: " << static_cast<u32>(inPlaceSupportRetryStatus) << std::endl;
    EXPECT_EQ(inplaceSupportRetry, false);
    EXPECT_EQ(static_cast<u32>(inPlaceSupportRetryStatus), 7);
    opParam.DataDes.count = 1024;
    unsetenv("HCCL_BUFFSIZE");

    inplaceSupportRetry = algOperator->SupportRetryWithInplaceCheck(
        HcclCMDType::HCCL_CMD_REDUCE, opParam, algName, isInplaceStatus, inPlaceSupportRetryStatus);
    std::cout << "isInplaceStatus: " << static_cast<u32>(isInplaceStatus) << std::endl;
    std::cout << "inPlaceSupportRetryStatus: " << static_cast<u32>(inPlaceSupportRetryStatus) << std::endl;
    EXPECT_EQ(inplaceSupportRetry, false);
    EXPECT_EQ(static_cast<u32>(inPlaceSupportRetryStatus), 8);

    ret = SetWorkflowMode(oldMode);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(CollAlgOperatorTest, ut_HcclCommunicator_IsHcclOpInplace)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclWorkflowMode oldMode = GetWorkflowMode();
    ret = SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = implBase->CreateCommCCLbuffer();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    std::unique_ptr<CollAlgOperator> algOperator(new (std::nothrow) CollAlgOperator(
        algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_ALLREDUCE));

    void* commInputPtr = nullptr;
    u64 commInputSize = 0;
    cclBufferManager.GetInCCLbuffer(commInputPtr, commInputSize);
    std::cout << "InCCLbuffer: " << commInputSize << std::endl;
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    DeviceMem scratchMem = DeviceMem::alloc(8192);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.outputPtr = inputMem.ptr();
    opParam.DataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    opParam.DataDes.count = 1024;
    opParam.root = 0;
    u8 isInplaceStatus = 0;

    bool isHcclOpInplace = IsHcclOpInplace(HcclCMDType::HCCL_CMD_ALLREDUCE, opParam, 0, 8, isInplaceStatus);
    EXPECT_EQ(isHcclOpInplace, true);
    // inputDataSize == 0 || outputDataSize == 0
    opParam.DataDes.count = 0;
    isHcclOpInplace = IsHcclOpInplace(HcclCMDType::HCCL_CMD_ALLREDUCE, opParam, 0, 8, isInplaceStatus);
    EXPECT_EQ(isHcclOpInplace, false);
    GlobalMockObject::verify();
}

TEST_F(CollAlgOperatorTest, ut_AHCAlgOptionSelect_When_SubGroupSizeSmall_Expect_SelectRing)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    std::unique_ptr<CollAlgOperator> algOperator(new (std::nothrow) CollAlgOperator(
        algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_ALLREDUCE));

    AlgTypeLevel1 algType{};
    std::vector<std::vector<std::vector<u32>>> globalSubGroups;
    globalSubGroups.resize(1);
    globalSubGroups[0].resize(1);
    std::map<AHCConcOpType, TemplateType> ahcAlgOption;
    AHCAlgSelectParam ahcAlgSelectParam{};
    ret = algOperator->AHCAlgOptionSelect(algType, globalSubGroups, ahcAlgOption, ahcAlgSelectParam);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(CollAlgOperatorTest, ut_AHCAlgOptionSelect_When_SubGroupSizeLarge_Expect_SelectNhr)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    std::unique_ptr<CollAlgOperator> algOperator(new (std::nothrow) CollAlgOperator(
        algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_ALLREDUCE));

    AlgTypeLevel1 algType{};
    std::vector<std::vector<std::vector<u32>>> globalSubGroups;
    globalSubGroups.resize(1);
    globalSubGroups[0].resize(4);
    std::map<AHCConcOpType, TemplateType> ahcAlgOption;
    AHCAlgSelectParam ahcAlgSelectParam{};
    ret = algOperator->AHCAlgOptionSelect(algType, globalSubGroups, ahcAlgOption, ahcAlgSelectParam);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(CollAlgOperatorTest, ut_HcclCommunicator_IsHcclOpInplace_alltoall)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclWorkflowMode oldMode = GetWorkflowMode();
    ret = SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = implBase->CreateCommCCLbuffer();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    std::unique_ptr<CollAlgOperator> algOperator(new (std::nothrow) CollAlgOperator(
        algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_ALLTOALL));

    void* commInputPtr = nullptr;
    u64 commInputSize = 0;
    cclBufferManager.GetInCCLbuffer(commInputPtr, commInputSize);
    std::cout << "InCCLbuffer: " << commInputSize << std::endl;
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    DeviceMem scratchMem = DeviceMem::alloc(8192);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.outputPtr = inputMem.ptr();
    opParam.DataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    opParam.DataDes.count = 1024;
    opParam.root = 0;
    u8 isInplaceStatus = 0;
    // inputPtr == outputPtr
    bool isHcclOpInplace = IsHcclOpInplace(HcclCMDType::HCCL_CMD_ALLTOALL, opParam, 0, 8, isInplaceStatus);
    EXPECT_EQ(isHcclOpInplace, true);
    // inputPtr != outputPtr
    opParam.outputPtr = outputMem.ptr();
    isHcclOpInplace = IsHcclOpInplace(HcclCMDType::HCCL_CMD_ALLTOALL, opParam, 0, 8, isInplaceStatus);
    EXPECT_EQ(isHcclOpInplace, false);
    GlobalMockObject::verify();
}

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
#include <fstream>
#include <iostream>
#include <unistd.h>
#include "nlohmann/json.hpp"
#include "hccl.h"
#define private public
#define protected public
#include "op_type.h"
#include "hccl_communicator.h"
#include "communicator_impl.h"
#include "hccp_hdc_manager.h"
#include "coll_operator.h"
#include "binary_stream.h"
#include "snap_shot_parse.h"
#include "binary_stream.h"
#undef private
#undef protected

using namespace Hccl;

class HcclCommunicatorTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HcclCommunicatorTest tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "HcclCommunicatorTest tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in HcclCommunicatorTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in HcclCommunicatorTest TearDown" << std::endl;
    }
};

TEST_F(HcclCommunicatorTest, should_return_success_when_calling_communicator_suspend)
{
    MOCKER_CPP(&CommunicatorImpl::Suspend).stubs().will(returnValue(HCCL_SUCCESS));
    CommParams params;
    HcclCommunicator comm(params);
    EXPECT_EQ(HCCL_SUCCESS, comm.Suspend());
}

TEST_F(HcclCommunicatorTest, should_return_success_when_calling_communicator_clean)
{
    MOCKER_CPP(&CommunicatorImpl::Clean).stubs().will(returnValue(HCCL_SUCCESS));
    CommParams params;
    HcclCommunicator comm(params);
    EXPECT_EQ(HCCL_SUCCESS, comm.Clean());
}

TEST_F(HcclCommunicatorTest, should_return_success_when_calling_communicator_resume)
{
    MOCKER_CPP(&CommunicatorImpl::Resume).stubs().will(returnValue(HCCL_SUCCESS));
    CommParams params;
    HcclCommunicator comm(params);
    EXPECT_EQ(HCCL_SUCCESS, comm.Resume());
}

TEST_F(HcclCommunicatorTest, should_return_seccess)
{
    CommParams params;
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    HcclCommunicator comm(params, &config);
}

TEST_F(HcclCommunicatorTest, should_return_success_when_get_id)
{
    CommParams params;
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    HcclCommunicator comm(params, &config);
    u32 rankId = 0;
    EXPECT_EQ(HCCL_SUCCESS, comm.GetRankId(rankId));
}

TEST_F(HcclCommunicatorTest, CollOpParams_desc)
{
    CollOpParams opParams;
    CollOpParams opParams1;
    opParams.DescAlltoall(opParams1);
    opParams.DescAlltoallV(opParams1);
    opParams.DescAlltoallVC(opParams1);
    opParams.DescBatchSendRecv(opParams1);
}

TEST_F(HcclCommunicatorTest, should_return_success_when_calling_communicator_create_sub_group)
{
    MOCKER_CPP(&HcclCommunicator::Init, HcclResult(HcclCommunicator::*)(const std::string&))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(static_cast<HcclResult (CommunicatorImpl::*)(
                   const CommParams& subCommParams, const std::vector<u32>& rankIds, CommunicatorImpl* subCommImpl,
                   HcclCommConfig& subConfig)>(&CommunicatorImpl::CreateSubComm))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    CommParams commParams;
    auto comm = new HcclCommunicator(commParams);
    comm->Init("ranktable.json");
    CommParams subCommParams;
    std::shared_ptr<HcclCommunicator> subHcclComm;
    std::vector<u32> rankIds;
    HcclCommConfig subConfig;

    auto res = comm->CreateSubComm(subCommParams, rankIds, subHcclComm, subConfig);
    EXPECT_EQ(HCCL_SUCCESS, res);
    EXPECT_NE(nullptr, subHcclComm);
    delete comm;
}

TEST_F(HcclCommunicatorTest, should_return_success_when_calling_communicator_create_sub_group_1)
{
    MOCKER_CPP(&HcclCommunicator::Init, HcclResult(HcclCommunicator::*)(const std::string&))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(static_cast<HcclResult (CommunicatorImpl::*)(
                   const CommParams& subCommParams, const std::vector<u32>& rankIds, CommunicatorImpl* subCommImpl)>(
                   &CommunicatorImpl::CreateSubComm))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    CommParams commParams;
    auto comm = new HcclCommunicator(commParams);
    comm->Init("ranktable.json");

    CommParams subCommParams;
    std::shared_ptr<HcclCommunicator> subHcclComm;
    std::vector<u32> rankIds;

    auto res = comm->CreateSubComm(subCommParams, rankIds, subHcclComm);
    EXPECT_EQ(HCCL_SUCCESS, res);
    EXPECT_NE(nullptr, subHcclComm);
    delete comm;
}

TEST_F(HcclCommunicatorTest, CollOperator_CollOpToString)
{
    CollOperator collOp;
    vector<OpType> optypes
        = {OpType::REDUCESCATTER, OpType::ALLREDUCE,  OpType::ALLGATHER, OpType::SCATTER, OpType::ALLTOALL,
           OpType::ALLTOALLV,     OpType::ALLTOALLVC, OpType::SEND,      OpType::RECV,    OpType::REDUCE};
    for (auto optype : optypes) {
        collOp.opType = optype;
        std::cout << CollOpToString(collOp);
        EXPECT_NO_THROW(CollOpToString(collOp));
    }
}

TEST_F(HcclCommunicatorTest, is_comm_ready_should_success)
{
    CommParams params;
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    HcclCommunicator comm(params, &config);

    MOCKER_CPP(&CommunicatorImpl::IsCommReady).stubs().will(returnValue(true));
    EXPECT_TRUE(comm.IsCommReady());
}

TEST_F(HcclCommunicatorTest, get_snap_shot_dynamic_buf_should_success)
{
    CommParams params;
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    HcclCommunicator comm(params, &config);

    comm.pimpl->collService = new CollServiceDefaultImpl(comm.pimpl.get());

    BinaryStream bs{};
    void* buf = &bs;
    EXPECT_EQ(comm.GetSnapShotDynamicBuf(buf), HcclResult::HCCL_SUCCESS);
    delete comm.pimpl->collService;
}

TEST_F(HcclCommunicatorTest, recover_comm_should_success)
{
    CommParams params;
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    HcclCommunicator comm(params, &config);

    MOCKER_CPP(
        static_cast<HcclResult (CommunicatorImpl::*)(SnapShotComm&, u32, const char*)>(&CommunicatorImpl::RecoverComm))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    SnapShotComm snapShotComm{};
    void* ptr = &snapShotComm;
    const char* filePath = "test";
    EXPECT_EQ(comm.RecoverComm(ptr, 0, filePath), HcclResult::HCCL_SUCCESS);
}

TEST_F(HcclCommunicatorTest, recover_sub_comm_should_success)
{
    CommParams params;
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    HcclCommunicator comm(params, &config);

    MOCKER_CPP(&CommunicatorImpl::RecoverSubComm)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    SnapShotSubComm snapShotSubComm{};
    void* ptr = &snapShotSubComm;
    std::shared_ptr<HcclCommunicator> subCommPtr;
    EXPECT_EQ(comm.RecoverSubComm(ptr, subCommPtr, 0), HcclResult::HCCL_SUCCESS);
}

TEST_F(HcclCommunicatorTest, get_static_binary_info_should_success)
{
    CommParams params;
    HcclCommConfig config;
    HcclCommConfigInit(&config);
    HcclCommunicator comm(params, &config);

    EXPECT_NO_THROW(comm.GetStaticBinaryInfo());
}

TEST_F(HcclCommunicatorTest, test_allreduce_GetUniqueId)
{
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.reduceOp = ReduceOp::SUM;
    op.dataType = DataType::INT8;
    op.dataCount = 0;
    op.root = 0;
    op.sendRecvRemoteRank = 0;
    op.opTag = "opTag_test";
    op.staticAddr = false;
    op.staticShape = false;
    op.outputDataType = DataType::INT8;
    op.dataDes.dataCount = 1;
    op.dataDes.dataType = DataType::INT8;
    op.dataDes.strideCount = 1;
    op.GetUniqueId();
}

TEST_F(HcclCommunicatorTest, test_alltoall_GetUniqueId)
{
    CollOperator op;
    op.opType = OpType::ALLTOALL;
    op.reduceOp = ReduceOp::SUM;
    op.dataType = DataType::INT8;
    op.dataCount = 0;
    op.root = 0;
    op.sendRecvRemoteRank = 0;
    op.opTag = "opTag_test";
    op.staticAddr = false;
    op.staticShape = false;
    op.outputDataType = DataType::INT8;
    op.all2AllDataDes.sendType = DataType::INT8;
    op.all2AllDataDes.recvType = DataType::INT8;
    op.all2AllDataDes.sendCount = 1;
    op.all2AllDataDes.recvCount = 1;
    op.GetUniqueId();
}

TEST_F(HcclCommunicatorTest, test_alltoallv_GetUniqueId)
{
    CollOperator op;
    op.opType = OpType::ALLTOALLV;
    op.reduceOp = ReduceOp::SUM;
    op.dataType = DataType::INT8;
    op.dataCount = 0;
    op.root = 0;
    op.sendRecvRemoteRank = 0;
    op.opTag = "opTag_test";
    op.staticAddr = false;
    op.staticShape = false;
    op.outputDataType = DataType::INT8;
    op.all2AllVDataDes.sendType = DataType::INT8;
    op.all2AllVDataDes.recvType = DataType::INT8;
    op.GetUniqueId();
}

TEST_F(HcclCommunicatorTest, test_allgatherv_GetUniqueId)
{
    CollOperator op;
    op.opType = OpType::ALLGATHERV;
    op.reduceOp = ReduceOp::SUM;
    op.dataType = DataType::INT8;
    op.dataCount = 0;
    op.root = 0;
    op.sendRecvRemoteRank = 0;
    op.opTag = "opTag_test";
    op.staticAddr = false;
    op.staticShape = false;
    op.outputDataType = DataType::INT8;
    op.vDataDes.dataType = DataType::INT8;
    op.GetUniqueId();
}

TEST_F(HcclCommunicatorTest, test_allreduce_GetPackedData)
{
    CollOperator op;
    op.opType = OpType::ALLREDUCE;
    op.reduceOp = ReduceOp::SUM;
    op.dataType = DataType::INT8;
    op.dataCount = 0;
    op.root = 0;
    op.sendRecvRemoteRank = 0;
    op.opTag = "opTag_test";
    op.staticAddr = false;
    op.staticShape = false;
    op.outputDataType = DataType::INT8;
    op.dataDes.dataCount = 1;
    op.dataDes.dataType = DataType::INT8;
    op.dataDes.strideCount = 1;
    vector<char> dataVec = op.GetUniqueId();
    op.GetPackedData(dataVec);
}

TEST_F(HcclCommunicatorTest, test_alltoall_GetPackedData)
{
    CollOperator op;
    op.opType = OpType::ALLTOALL;
    op.reduceOp = ReduceOp::SUM;
    op.dataType = DataType::INT8;
    op.dataCount = 0;
    op.root = 0;
    op.sendRecvRemoteRank = 0;
    op.opTag = "opTag_test";
    op.staticAddr = false;
    op.staticShape = false;
    op.outputDataType = DataType::INT8;
    op.all2AllDataDes.sendType = DataType::INT8;
    op.all2AllDataDes.recvType = DataType::INT8;
    op.all2AllDataDes.sendCount = 1;
    op.all2AllDataDes.recvCount = 1;
    vector<char> dataVec = op.GetUniqueId();
    op.GetPackedData(dataVec);
}

TEST_F(HcclCommunicatorTest, test_alltoallv_GetPackedData)
{
    CollOperator op;
    op.opType = OpType::ALLTOALLV;
    op.reduceOp = ReduceOp::SUM;
    op.dataType = DataType::INT8;
    op.dataCount = 0;
    op.root = 0;
    op.sendRecvRemoteRank = 0;
    op.opTag = "opTag_test";
    op.staticAddr = false;
    op.staticShape = false;
    op.outputDataType = DataType::INT8;
    op.all2AllVDataDes.sendType = DataType::INT8;
    op.all2AllVDataDes.recvType = DataType::INT8;
    vector<char> dataVec = op.GetUniqueId();
    op.GetPackedData(dataVec);
}

TEST_F(HcclCommunicatorTest, test_allgatherv_GetPackedData)
{
    CollOperator op;
    op.opType = OpType::ALLGATHERV;
    op.reduceOp = ReduceOp::SUM;
    op.dataType = DataType::INT8;
    op.dataCount = 0;
    op.root = 0;
    op.sendRecvRemoteRank = 0;
    op.opTag = "opTag_test";
    op.staticAddr = false;
    op.staticShape = false;
    op.outputDataType = DataType::INT8;
    op.vDataDes.dataType = DataType::INT8;
    vector<char> dataVec = op.GetUniqueId();
    op.GetPackedData(dataVec);
}

TEST_F(HcclCommunicatorTest, St_GetPackedData_When_OpType_Is_ALLTOALLVC_Expect_Success)
{
    CollOperator op;
    op.opType = OpType::ALLTOALLVC;
    op.reduceOp = ReduceOp::SUM;
    op.dataType = DataType::INT8;
    op.dataCount = 0;
    op.root = 0;
    op.sendRecvRemoteRank = 0;
    op.opTag = "opTag_test";
    op.staticAddr = false;
    op.staticShape = false;
    op.outputDataType = DataType::INT8;
    op.all2AllVCDataDes.sendType = DataType::INT8;
    op.all2AllVCDataDes.recvType = DataType::INT8;
    vector<char> dataVec = op.GetUniqueId();
    op.GetPackedData(dataVec);
}

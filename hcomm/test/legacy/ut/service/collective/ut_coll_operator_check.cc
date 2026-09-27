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
#include "coll_operator_check.h"
#include "exception_util.h"
#include "not_support_exception.h"
#include "invalid_params_exception.h"
#undef private

using namespace Hccl;
using namespace std;

class CollOperatorCheckTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CollOperatorCheckTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "CollOperatorCheckTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in CollOperatorCheckTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CollOperatorCheckTest TearDown" << std::endl;
    }
};

TEST_F(CollOperatorCheckTest, test_CheckCollOperator_allreduce_ok)
{
    CollOperator localOpData;
    localOpData.opType = OpType::ALLREDUCE;
    localOpData.reduceOp = ReduceOp::SUM;
    localOpData.dataType = DataType::INT8;
    localOpData.dataCount = 0;
    localOpData.root = 0;
    localOpData.sendRecvRemoteRank = 0;
    localOpData.opTag = "opTag_test";
    localOpData.staticAddr = false;
    localOpData.staticShape = false;
    localOpData.outputDataType = DataType::INT8;
    localOpData.dataDes.dataCount = 0;
    localOpData.dataDes.dataType = DataType::INT8;
    localOpData.dataDes.strideCount = 0;

    CollOperator remoteOpData;
    remoteOpData.opType = OpType::ALLREDUCE;
    remoteOpData.reduceOp = ReduceOp::SUM;
    remoteOpData.dataType = DataType::INT8;
    remoteOpData.dataCount = 0;
    remoteOpData.root = 0;
    remoteOpData.sendRecvRemoteRank = 0;
    remoteOpData.opTag = "opTag_test";
    remoteOpData.staticAddr = false;
    remoteOpData.staticShape = false;
    remoteOpData.outputDataType = DataType::INT8;
    remoteOpData.dataDes.dataCount = 0;
    remoteOpData.dataDes.dataType = DataType::INT8;
    remoteOpData.dataDes.strideCount = 0;

    EXPECT_NO_THROW(CheckCollOperator(localOpData, remoteOpData));
}

TEST_F(CollOperatorCheckTest, test_CheckCollOperator_sendRecv_ok)
{
    CollOperator localOpData;
    localOpData.opType = OpType::SEND;
    localOpData.reduceOp = ReduceOp::SUM;
    localOpData.dataType = DataType::INT8;
    localOpData.dataCount = 0;
    localOpData.root = 0;
    localOpData.myRank = 0;
    localOpData.opTag = "opTag_test";
    localOpData.staticAddr = false;
    localOpData.staticShape = false;
    localOpData.outputDataType = DataType::INT8;
    localOpData.dataDes.dataCount = 0;
    localOpData.dataDes.dataType = DataType::INT8;
    localOpData.dataDes.strideCount = 0;

    CollOperator remoteOpData;
    remoteOpData.opType = OpType::RECV;
    remoteOpData.reduceOp = ReduceOp::SUM;
    remoteOpData.dataType = DataType::INT8;
    remoteOpData.dataCount = 0;
    remoteOpData.root = 0;
    remoteOpData.sendRecvRemoteRank = 0;
    remoteOpData.opTag = "opTag_test";
    remoteOpData.staticAddr = false;
    remoteOpData.staticShape = false;
    remoteOpData.outputDataType = DataType::INT8;
    remoteOpData.dataDes.dataCount = 0;
    remoteOpData.dataDes.dataType = DataType::INT8;
    remoteOpData.dataDes.strideCount = 0;

    EXPECT_NO_THROW(CheckCollOperator(localOpData, remoteOpData));
}

TEST_F(CollOperatorCheckTest, test_CheckCollOperator_alltoall_ok)
{
    CollOperator localOpData;
    localOpData.opType = OpType::ALLTOALL;
    localOpData.all2AllDataDes.sendType = DataType::INT8;
    localOpData.all2AllDataDes.recvType = DataType::INT8;
    localOpData.all2AllDataDes.sendCount = 1;
    localOpData.all2AllDataDes.recvCount = 1;

    CollOperator remoteOpData;
    remoteOpData.opType = OpType::ALLTOALL;
    remoteOpData.all2AllDataDes.sendType = DataType::INT8;
    remoteOpData.all2AllDataDes.recvType = DataType::INT8;
    remoteOpData.all2AllDataDes.sendCount = 1;
    remoteOpData.all2AllDataDes.recvCount = 1;

    EXPECT_NO_THROW(CheckCollOperator(localOpData, remoteOpData));
}

TEST_F(CollOperatorCheckTest, test_CheckCollOperator_alltoallv_ok)
{
    CollOperator localOpData;
    localOpData.opType = OpType::ALLTOALLV;
    localOpData.all2AllVDataDes.sendType = DataType::INT8;
    localOpData.all2AllVDataDes.recvType = DataType::INT8;

    CollOperator remoteOpData;
    remoteOpData.opType = OpType::ALLTOALLV;
    remoteOpData.all2AllVDataDes.sendType = DataType::INT8;
    remoteOpData.all2AllVDataDes.recvType = DataType::INT8;

    EXPECT_NO_THROW(CheckCollOperator(localOpData, remoteOpData));
}

TEST_F(CollOperatorCheckTest, test_CheckCollOperator_allgatherv_ok)
{
    CollOperator localOpData;
    localOpData.opType = OpType::ALLGATHERV;
    localOpData.reduceOp = ReduceOp::SUM;
    localOpData.dataType = DataType::INT8;
    localOpData.dataCount = 0;
    localOpData.root = 0;
    localOpData.myRank = 0;
    localOpData.opTag = "opTag_test";
    localOpData.staticAddr = false;
    localOpData.staticShape = false;
    localOpData.outputDataType = DataType::INT8;
    localOpData.vDataDes.dataType = DataType::INT8;

    CollOperator remoteOpData;
    remoteOpData.opType = OpType::ALLGATHERV;
    remoteOpData.reduceOp = ReduceOp::SUM;
    remoteOpData.dataType = DataType::INT8;
    remoteOpData.dataCount = 0;
    remoteOpData.root = 0;
    remoteOpData.sendRecvRemoteRank = 0;
    remoteOpData.opTag = "opTag_test";
    remoteOpData.staticAddr = false;
    remoteOpData.staticShape = false;
    remoteOpData.outputDataType = DataType::INT8;
    remoteOpData.vDataDes.dataType = DataType::INT8;

    EXPECT_NO_THROW(CheckCollOperator(localOpData, remoteOpData));
}

TEST_F(CollOperatorCheckTest, test_CheckCollOperator_allgather_opType_not_ok)
{
    CollOperator localOpData;
    localOpData.opType = OpType::ALLGATHER;
    localOpData.reduceOp = ReduceOp::SUM;
    localOpData.dataType = DataType::INT8;
    localOpData.dataCount = 0;
    localOpData.root = 0;
    localOpData.sendRecvRemoteRank = 0;
    localOpData.opTag = "opTag_test";
    localOpData.staticAddr = false;
    localOpData.staticShape = false;
    localOpData.outputDataType = DataType::INT8;

    CollOperator remoteOpData;
    remoteOpData.opType = OpType::ALLREDUCE;
    remoteOpData.reduceOp = ReduceOp::SUM;
    remoteOpData.dataType = DataType::INT8;
    remoteOpData.dataCount = 0;
    remoteOpData.root = 0;
    remoteOpData.sendRecvRemoteRank = 0;
    remoteOpData.opTag = "opTag_test";
    remoteOpData.staticAddr = false;
    remoteOpData.staticShape = false;
    remoteOpData.outputDataType = DataType::INT8;

    EXPECT_THROW(CheckCollOperator(localOpData, remoteOpData), InvalidParamsException);
}

TEST_F(CollOperatorCheckTest, test_CheckCollOperator_allgather_reduceOp_not_ok)
{
    CollOperator localOpData;
    localOpData.opType = OpType::ALLGATHER;
    localOpData.reduceOp = ReduceOp::SUM;
    localOpData.dataType = DataType::INT8;
    localOpData.dataCount = 0;
    localOpData.root = 0;
    localOpData.sendRecvRemoteRank = 0;
    localOpData.opTag = "opTag_test";
    localOpData.staticAddr = false;
    localOpData.staticShape = false;
    localOpData.outputDataType = DataType::INT8;

    CollOperator remoteOpData;
    remoteOpData.opType = OpType::ALLGATHER;
    remoteOpData.reduceOp = ReduceOp::MAX;
    remoteOpData.dataType = DataType::INT8;
    remoteOpData.dataCount = 0;
    remoteOpData.root = 0;
    remoteOpData.sendRecvRemoteRank = 0;
    remoteOpData.opTag = "opTag_test";
    remoteOpData.staticAddr = false;
    remoteOpData.staticShape = false;
    remoteOpData.outputDataType = DataType::INT8;

    EXPECT_THROW(CheckCollOperator(localOpData, remoteOpData), InvalidParamsException);
}

TEST_F(CollOperatorCheckTest, test_CheckCollOperator_allgather_dataType_not_ok)
{
    CollOperator localOpData;
    localOpData.opType = OpType::ALLGATHER;
    localOpData.reduceOp = ReduceOp::SUM;
    localOpData.dataType = DataType::INT8;
    localOpData.dataCount = 0;
    localOpData.root = 0;
    localOpData.sendRecvRemoteRank = 0;
    localOpData.opTag = "opTag_test";
    localOpData.staticAddr = false;
    localOpData.staticShape = false;
    localOpData.outputDataType = DataType::INT8;

    CollOperator remoteOpData;
    remoteOpData.opType = OpType::ALLGATHER;
    remoteOpData.reduceOp = ReduceOp::SUM;
    remoteOpData.dataType = DataType::INT16;
    remoteOpData.dataCount = 0;
    remoteOpData.root = 0;
    remoteOpData.sendRecvRemoteRank = 0;
    remoteOpData.opTag = "opTag_test";
    remoteOpData.staticAddr = false;
    remoteOpData.staticShape = false;
    remoteOpData.outputDataType = DataType::INT8;

    EXPECT_THROW(CheckCollOperator(localOpData, remoteOpData), InvalidParamsException);
}

TEST_F(CollOperatorCheckTest, test_CheckCollOperator_allgather_dataCount_not_ok)
{
    CollOperator localOpData;
    localOpData.opType = OpType::ALLGATHER;
    localOpData.reduceOp = ReduceOp::SUM;
    localOpData.dataType = DataType::INT8;
    localOpData.dataCount = 10;
    localOpData.root = 0;
    localOpData.sendRecvRemoteRank = 0;
    localOpData.opTag = "opTag_test";
    localOpData.staticAddr = false;
    localOpData.staticShape = false;
    localOpData.outputDataType = DataType::INT8;

    CollOperator remoteOpData;
    remoteOpData.opType = OpType::ALLGATHER;
    remoteOpData.reduceOp = ReduceOp::SUM;
    remoteOpData.dataType = DataType::INT8;
    remoteOpData.dataCount = 0;
    remoteOpData.root = 0;
    remoteOpData.sendRecvRemoteRank = 0;
    remoteOpData.opTag = "opTag_test";
    remoteOpData.staticAddr = false;
    remoteOpData.staticShape = false;
    remoteOpData.outputDataType = DataType::INT8;

    EXPECT_THROW(CheckCollOperator(localOpData, remoteOpData), InvalidParamsException);
}

TEST_F(CollOperatorCheckTest, test_CheckCollOperator_allgather_root_not_ok)
{
    CollOperator localOpData;
    localOpData.opType = OpType::ALLGATHER;
    localOpData.reduceOp = ReduceOp::SUM;
    localOpData.dataType = DataType::INT8;
    localOpData.dataCount = 0;
    localOpData.root = 10;
    localOpData.sendRecvRemoteRank = 0;
    localOpData.opTag = "opTag_test";
    localOpData.staticAddr = false;
    localOpData.staticShape = false;
    localOpData.outputDataType = DataType::INT8;

    CollOperator remoteOpData;
    remoteOpData.opType = OpType::ALLGATHER;
    remoteOpData.reduceOp = ReduceOp::SUM;
    remoteOpData.dataType = DataType::INT8;
    remoteOpData.dataCount = 0;
    remoteOpData.root = 0;
    remoteOpData.sendRecvRemoteRank = 0;
    remoteOpData.opTag = "opTag_test";
    remoteOpData.staticAddr = false;
    remoteOpData.staticShape = false;
    remoteOpData.outputDataType = DataType::INT8;

    EXPECT_THROW(CheckCollOperator(localOpData, remoteOpData), InvalidParamsException);
}

TEST_F(CollOperatorCheckTest, test_CheckCollOperator_sendRecv_sendRecvRemoteRank_not_ok)
{
    CollOperator localOpData;
    localOpData.opType = OpType::SEND;
    localOpData.reduceOp = ReduceOp::SUM;
    localOpData.dataType = DataType::INT8;
    localOpData.dataCount = 0;
    localOpData.root = 0;
    localOpData.myRank = 10;
    localOpData.opTag = "opTag_test";
    localOpData.staticAddr = false;
    localOpData.staticShape = false;
    localOpData.outputDataType = DataType::INT8;

    CollOperator remoteOpData;
    remoteOpData.opType = OpType::RECV;
    remoteOpData.reduceOp = ReduceOp::SUM;
    remoteOpData.dataType = DataType::INT8;
    remoteOpData.dataCount = 0;
    remoteOpData.root = 0;
    remoteOpData.sendRecvRemoteRank = 0;
    remoteOpData.opTag = "opTag_test";
    remoteOpData.staticAddr = false;
    remoteOpData.staticShape = false;
    remoteOpData.outputDataType = DataType::INT8;

    EXPECT_THROW(CheckCollOperator(localOpData, remoteOpData), InvalidParamsException);
}

TEST_F(CollOperatorCheckTest, test_CheckCollOperator_allgather_opTag_not_ok)
{
    CollOperator localOpData;
    localOpData.opType = OpType::ALLGATHER;
    localOpData.reduceOp = ReduceOp::SUM;
    localOpData.dataType = DataType::INT8;
    localOpData.dataCount = 0;
    localOpData.root = 0;
    localOpData.sendRecvRemoteRank = 0;
    localOpData.opTag = "opTag_test_local";
    localOpData.staticAddr = false;
    localOpData.staticShape = false;
    localOpData.outputDataType = DataType::INT8;

    CollOperator remoteOpData;
    remoteOpData.opType = OpType::ALLGATHER;
    remoteOpData.reduceOp = ReduceOp::SUM;
    remoteOpData.dataType = DataType::INT8;
    remoteOpData.dataCount = 0;
    remoteOpData.root = 0;
    remoteOpData.sendRecvRemoteRank = 0;
    remoteOpData.opTag = "opTag_test";
    remoteOpData.staticAddr = false;
    remoteOpData.staticShape = false;
    remoteOpData.outputDataType = DataType::INT8;

    EXPECT_THROW(CheckCollOperator(localOpData, remoteOpData), InvalidParamsException);
}

TEST_F(CollOperatorCheckTest, test_CheckCollOperator_allgather_staticAddr_not_ok)
{
    CollOperator localOpData;
    localOpData.opType = OpType::ALLGATHER;
    localOpData.reduceOp = ReduceOp::SUM;
    localOpData.dataType = DataType::INT8;
    localOpData.dataCount = 0;
    localOpData.root = 0;
    localOpData.sendRecvRemoteRank = 0;
    localOpData.opTag = "opTag_test";
    localOpData.staticAddr = true;
    localOpData.staticShape = false;
    localOpData.outputDataType = DataType::INT8;

    CollOperator remoteOpData;
    remoteOpData.opType = OpType::ALLGATHER;
    remoteOpData.reduceOp = ReduceOp::SUM;
    remoteOpData.dataType = DataType::INT8;
    remoteOpData.dataCount = 0;
    remoteOpData.root = 0;
    remoteOpData.sendRecvRemoteRank = 0;
    remoteOpData.opTag = "opTag_test";
    remoteOpData.staticAddr = false;
    remoteOpData.staticShape = false;
    remoteOpData.outputDataType = DataType::INT8;

    EXPECT_THROW(CheckCollOperator(localOpData, remoteOpData), InvalidParamsException);
}

TEST_F(CollOperatorCheckTest, test_CheckCollOperator_allgather_staticShape_not_ok)
{
    CollOperator localOpData;
    localOpData.opType = OpType::ALLGATHER;
    localOpData.reduceOp = ReduceOp::SUM;
    localOpData.dataType = DataType::INT8;
    localOpData.dataCount = 0;
    localOpData.root = 0;
    localOpData.sendRecvRemoteRank = 0;
    localOpData.opTag = "opTag_test";
    localOpData.staticAddr = false;
    localOpData.staticShape = true;
    localOpData.outputDataType = DataType::INT8;

    CollOperator remoteOpData;
    remoteOpData.opType = OpType::ALLGATHER;
    remoteOpData.reduceOp = ReduceOp::SUM;
    remoteOpData.dataType = DataType::INT8;
    remoteOpData.dataCount = 0;
    remoteOpData.root = 0;
    remoteOpData.sendRecvRemoteRank = 0;
    remoteOpData.opTag = "opTag_test";
    remoteOpData.staticAddr = false;
    remoteOpData.staticShape = false;
    remoteOpData.outputDataType = DataType::INT8;

    EXPECT_THROW(CheckCollOperator(localOpData, remoteOpData), InvalidParamsException);
}

TEST_F(CollOperatorCheckTest, test_CheckCollOperator_allgather_outputDataType_not_ok)
{
    CollOperator localOpData;
    localOpData.opType = OpType::ALLGATHER;
    localOpData.reduceOp = ReduceOp::SUM;
    localOpData.dataType = DataType::INT8;
    localOpData.dataCount = 0;
    localOpData.root = 0;
    localOpData.sendRecvRemoteRank = 0;
    localOpData.opTag = "opTag_test";
    localOpData.staticAddr = false;
    localOpData.staticShape = false;
    localOpData.outputDataType = DataType::INT8;

    CollOperator remoteOpData;
    remoteOpData.opType = OpType::ALLGATHER;
    remoteOpData.reduceOp = ReduceOp::SUM;
    remoteOpData.dataType = DataType::INT8;
    remoteOpData.dataCount = 0;
    remoteOpData.root = 0;
    remoteOpData.sendRecvRemoteRank = 0;
    remoteOpData.opTag = "opTag_test";
    remoteOpData.staticAddr = false;
    remoteOpData.staticShape = false;
    remoteOpData.outputDataType = DataType::INT16;

    EXPECT_THROW(CheckCollOperator(localOpData, remoteOpData), InvalidParamsException);
}

TEST_F(CollOperatorCheckTest, test_CheckCollOperator_allgather_dataDes_dataCount_not_ok)
{
    CollOperator localOpData;
    localOpData.opType = OpType::ALLGATHER;
    localOpData.reduceOp = ReduceOp::SUM;
    localOpData.dataType = DataType::INT8;
    localOpData.dataCount = 0;
    localOpData.root = 0;
    localOpData.sendRecvRemoteRank = 0;
    localOpData.opTag = "opTag_test";
    localOpData.staticAddr = false;
    localOpData.staticShape = false;
    localOpData.outputDataType = DataType::INT8;
    localOpData.dataDes.dataCount = 1;
    localOpData.dataDes.dataType = DataType::INT8;
    localOpData.dataDes.strideCount = 1;

    CollOperator remoteOpData;
    remoteOpData.opType = OpType::ALLGATHER;
    remoteOpData.reduceOp = ReduceOp::SUM;
    remoteOpData.dataType = DataType::INT8;
    remoteOpData.dataCount = 0;
    remoteOpData.root = 0;
    remoteOpData.sendRecvRemoteRank = 0;
    remoteOpData.opTag = "opTag_test";
    remoteOpData.staticAddr = false;
    remoteOpData.staticShape = false;
    remoteOpData.outputDataType = DataType::INT8;
    remoteOpData.dataDes.dataCount = 2;
    remoteOpData.dataDes.dataType = DataType::INT8;
    remoteOpData.dataDes.strideCount = 1;

    EXPECT_THROW(CheckCollOperator(localOpData, remoteOpData), InvalidParamsException);
}

TEST_F(CollOperatorCheckTest, test_CheckCollOperator_allgather_dataDes_dataType_not_ok)
{
    CollOperator localOpData;
    localOpData.opType = OpType::ALLGATHER;
    localOpData.reduceOp = ReduceOp::SUM;
    localOpData.dataType = DataType::INT8;
    localOpData.dataCount = 0;
    localOpData.root = 0;
    localOpData.sendRecvRemoteRank = 0;
    localOpData.opTag = "opTag_test";
    localOpData.staticAddr = false;
    localOpData.staticShape = false;
    localOpData.outputDataType = DataType::INT8;
    localOpData.dataDes.dataCount = 1;
    localOpData.dataDes.dataType = DataType::INT8;
    localOpData.dataDes.strideCount = 1;

    CollOperator remoteOpData;
    remoteOpData.opType = OpType::ALLGATHER;
    remoteOpData.reduceOp = ReduceOp::SUM;
    remoteOpData.dataType = DataType::INT8;
    remoteOpData.dataCount = 0;
    remoteOpData.root = 0;
    remoteOpData.sendRecvRemoteRank = 0;
    remoteOpData.opTag = "opTag_test";
    remoteOpData.staticAddr = false;
    remoteOpData.staticShape = false;
    remoteOpData.outputDataType = DataType::INT8;
    remoteOpData.dataDes.dataCount = 1;
    remoteOpData.dataDes.dataType = DataType::INT16;
    remoteOpData.dataDes.strideCount = 1;

    EXPECT_THROW(CheckCollOperator(localOpData, remoteOpData), InvalidParamsException);
}

TEST_F(CollOperatorCheckTest, test_CheckCollOperator_allgather_dataDes_strideCount_not_ok)
{
    CollOperator localOpData;
    localOpData.opType = OpType::ALLGATHER;
    localOpData.reduceOp = ReduceOp::SUM;
    localOpData.dataType = DataType::INT8;
    localOpData.dataCount = 0;
    localOpData.root = 0;
    localOpData.sendRecvRemoteRank = 0;
    localOpData.opTag = "opTag_test";
    localOpData.staticAddr = false;
    localOpData.staticShape = false;
    localOpData.outputDataType = DataType::INT8;
    localOpData.dataDes.dataCount = 1;
    localOpData.dataDes.dataType = DataType::INT8;
    localOpData.dataDes.strideCount = 1;

    CollOperator remoteOpData;
    remoteOpData.opType = OpType::ALLGATHER;
    remoteOpData.reduceOp = ReduceOp::SUM;
    remoteOpData.dataType = DataType::INT8;
    remoteOpData.dataCount = 0;
    remoteOpData.root = 0;
    remoteOpData.sendRecvRemoteRank = 0;
    remoteOpData.opTag = "opTag_test";
    remoteOpData.staticAddr = false;
    remoteOpData.staticShape = false;
    remoteOpData.outputDataType = DataType::INT8;
    remoteOpData.dataDes.dataCount = 1;
    remoteOpData.dataDes.dataType = DataType::INT8;
    remoteOpData.dataDes.strideCount = 2;

    EXPECT_THROW(CheckCollOperator(localOpData, remoteOpData), InvalidParamsException);
}

TEST_F(CollOperatorCheckTest, test_CheckCollOperator_allgatherv_dataType_not_ok)
{
    CollOperator localOpData;
    localOpData.opType = OpType::ALLGATHERV;
    localOpData.reduceOp = ReduceOp::SUM;
    localOpData.dataType = DataType::INT8;
    localOpData.dataCount = 0;
    localOpData.root = 0;
    localOpData.sendRecvRemoteRank = 0;
    localOpData.opTag = "opTag_test";
    localOpData.staticAddr = false;
    localOpData.staticShape = false;
    localOpData.outputDataType = DataType::INT8;
    localOpData.vDataDes.dataType = DataType::INT8;

    CollOperator remoteOpData;
    remoteOpData.opType = OpType::ALLGATHERV;
    remoteOpData.reduceOp = ReduceOp::SUM;
    remoteOpData.dataType = DataType::INT8;
    remoteOpData.dataCount = 0;
    remoteOpData.root = 0;
    remoteOpData.sendRecvRemoteRank = 0;
    remoteOpData.opTag = "opTag_test";
    remoteOpData.staticAddr = false;
    remoteOpData.staticShape = false;
    remoteOpData.outputDataType = DataType::INT8;
    remoteOpData.vDataDes.dataType = DataType::INT16;

    EXPECT_THROW(CheckCollOperator(localOpData, remoteOpData), InvalidParamsException);
}

TEST_F(CollOperatorCheckTest, test_CheckCollOperator_alltoall_sendType_not_ok)
{
    CollOperator localOpData;
    localOpData.opType = OpType::ALLTOALL;
    localOpData.all2AllDataDes.sendType = DataType::INT8;
    localOpData.all2AllDataDes.recvType = DataType::INT8;
    localOpData.all2AllDataDes.sendCount = 1;
    localOpData.all2AllDataDes.recvCount = 1;

    CollOperator remoteOpData;
    remoteOpData.opType = OpType::ALLTOALL;
    remoteOpData.all2AllDataDes.sendType = DataType::INT8;
    remoteOpData.all2AllDataDes.recvType = DataType::INT16;
    remoteOpData.all2AllDataDes.sendCount = 1;
    remoteOpData.all2AllDataDes.recvCount = 1;

    EXPECT_THROW(CheckCollOperator(localOpData, remoteOpData), InvalidParamsException);
}

TEST_F(CollOperatorCheckTest, test_CheckCollOperator_alltoall_recvType_not_ok)
{
    CollOperator localOpData;
    localOpData.opType = OpType::ALLTOALL;
    localOpData.all2AllDataDes.sendType = DataType::INT8;
    localOpData.all2AllDataDes.recvType = DataType::INT16;
    localOpData.all2AllDataDes.sendCount = 1;
    localOpData.all2AllDataDes.recvCount = 1;

    CollOperator remoteOpData;
    remoteOpData.opType = OpType::ALLTOALL;
    remoteOpData.all2AllDataDes.sendType = DataType::INT8;
    remoteOpData.all2AllDataDes.recvType = DataType::INT8;
    remoteOpData.all2AllDataDes.sendCount = 1;
    remoteOpData.all2AllDataDes.recvCount = 1;

    EXPECT_THROW(CheckCollOperator(localOpData, remoteOpData), InvalidParamsException);
}

TEST_F(CollOperatorCheckTest, test_CheckCollOperator_alltoall_sendCount_not_ok)
{
    CollOperator localOpData;
    localOpData.opType = OpType::ALLTOALL;
    localOpData.all2AllDataDes.sendType = DataType::INT8;
    localOpData.all2AllDataDes.recvType = DataType::INT8;
    localOpData.all2AllDataDes.sendCount = 1;
    localOpData.all2AllDataDes.recvCount = 1;

    CollOperator remoteOpData;
    remoteOpData.opType = OpType::ALLTOALL;
    remoteOpData.all2AllDataDes.sendType = DataType::INT8;
    remoteOpData.all2AllDataDes.recvType = DataType::INT8;
    remoteOpData.all2AllDataDes.sendCount = 1;
    remoteOpData.all2AllDataDes.recvCount = 10;

    EXPECT_THROW(CheckCollOperator(localOpData, remoteOpData), InvalidParamsException);
}

TEST_F(CollOperatorCheckTest, test_CheckCollOperator_alltoall_recvCount_not_ok)
{
    CollOperator localOpData;
    localOpData.opType = OpType::ALLTOALL;
    localOpData.all2AllDataDes.sendType = DataType::INT8;
    localOpData.all2AllDataDes.recvType = DataType::INT8;
    localOpData.all2AllDataDes.sendCount = 1;
    localOpData.all2AllDataDes.recvCount = 1;

    CollOperator remoteOpData;
    remoteOpData.opType = OpType::ALLTOALL;
    remoteOpData.all2AllDataDes.sendType = DataType::INT8;
    remoteOpData.all2AllDataDes.recvType = DataType::INT8;
    remoteOpData.all2AllDataDes.sendCount = 10;
    remoteOpData.all2AllDataDes.recvCount = 1;

    EXPECT_THROW(CheckCollOperator(localOpData, remoteOpData), InvalidParamsException);
}

TEST_F(CollOperatorCheckTest, test_CheckCollOperator_alltoallv_sendType_not_ok)
{
    CollOperator localOpData;
    localOpData.opType = OpType::ALLTOALLV;
    localOpData.all2AllVDataDes.sendType = DataType::INT8;
    localOpData.all2AllVDataDes.recvType = DataType::INT8;

    CollOperator remoteOpData;
    remoteOpData.opType = OpType::ALLTOALLV;
    remoteOpData.all2AllVDataDes.sendType = DataType::INT8;
    remoteOpData.all2AllVDataDes.recvType = DataType::INT16;

    EXPECT_THROW(CheckCollOperator(localOpData, remoteOpData), InvalidParamsException);
}

TEST_F(CollOperatorCheckTest, test_CheckCollOperator_alltoallv_recvType_not_ok)
{
    CollOperator localOpData;
    localOpData.opType = OpType::ALLTOALLV;
    localOpData.all2AllVDataDes.sendType = DataType::INT8;
    localOpData.all2AllVDataDes.recvType = DataType::INT16;

    CollOperator remoteOpData;
    remoteOpData.opType = OpType::ALLTOALLV;
    remoteOpData.all2AllVDataDes.sendType = DataType::INT8;
    remoteOpData.all2AllVDataDes.recvType = DataType::INT8;

    EXPECT_THROW(CheckCollOperator(localOpData, remoteOpData), InvalidParamsException);
}

void All2AllVCSetSendTypeAndRecvType(CollOperator& opData, DataType sType, DataType rType)
{
    opData.opType = OpType::ALLTOALLVC;
    opData.all2AllVCDataDes.sendType = sType;
    opData.all2AllVCDataDes.recvType = rType;
}

TEST_F(CollOperatorCheckTest, Ut_CheckCollOperator_When_OpType_Is_ALLTOALLVC_Expect_Success)
{
    CollOperator localOpData;
    CollOperator remoteOpData;
    All2AllVCSetSendTypeAndRecvType(localOpData, DataType::INT8, DataType::INT8);
    All2AllVCSetSendTypeAndRecvType(remoteOpData, DataType::INT8, DataType::INT8);

    EXPECT_NO_THROW(CheckCollOperator(localOpData, remoteOpData));
}

TEST_F(CollOperatorCheckTest, Ut_CheckCollOperator_When_ALLTOALLVC_Loacl_Sendtype_Rmt_Recvtype_Not_Equal_Expect_Fail)
{
    CollOperator localOpData;
    CollOperator remoteOpData;
    All2AllVCSetSendTypeAndRecvType(localOpData, DataType::INT8, DataType::INT8);
    All2AllVCSetSendTypeAndRecvType(remoteOpData, DataType::INT8, DataType::INT16);

    EXPECT_THROW(CheckCollOperator(localOpData, remoteOpData), InvalidParamsException);
}

TEST_F(CollOperatorCheckTest, Ut_CheckCollOperator_When_ALLTOALLVC_Loacl_Recvtype_Rmt_Sendtype_Not_Equal_Expect_Fail)
{
    CollOperator localOpData;
    CollOperator remoteOpData;
    All2AllVCSetSendTypeAndRecvType(localOpData, DataType::INT8, DataType::INT16);
    All2AllVCSetSendTypeAndRecvType(remoteOpData, DataType::INT8, DataType::INT8);

    EXPECT_THROW(CheckCollOperator(localOpData, remoteOpData), InvalidParamsException);
}

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
#include "coll_operator.h"
#include <vector>
#include <string>
#include "binary_stream.h"
#undef private

using namespace Hccl;
using namespace std;

class CollOperatorTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CollOperatorTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "CollOperatorTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in CollOperatorTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test case in CollOperatorTest TearDown" << std::endl;
    }
};

TEST_F(CollOperatorTest, test_allreduce_GetUniqueId)
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

TEST_F(CollOperatorTest, test_alltoall_GetUniqueId)
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

TEST_F(CollOperatorTest, test_alltoallv_GetUniqueId)
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

TEST_F(CollOperatorTest, test_allgatherv_GetUniqueId)
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

TEST_F(CollOperatorTest, test_allreduce_GetPackedData)
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

TEST_F(CollOperatorTest, test_alltoall_GetPackedData)
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

TEST_F(CollOperatorTest, test_alltoallv_GetPackedData)
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

TEST_F(CollOperatorTest, test_allgatherv_GetPackedData)
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

TEST_F(CollOperatorTest, Ut_GetPackedData_When_OpType_Is_ALLTOALLVC_Expect_Success)
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

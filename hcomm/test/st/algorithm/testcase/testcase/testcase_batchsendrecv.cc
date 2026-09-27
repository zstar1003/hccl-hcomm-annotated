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
#include <vector>
#include <iostream>

#define private public
#define protected public
#include "topoinfo_struct.h"
#include "log.h"
#include "checker_def.h"
#include "topo_meta.h"
#include "testcase_utils.h"
#include "checker.h"
#include "coll_batch_send_recv_executor.h"
#include "hccl_alg.h"
#include "hccl_impl.h"
#include "alg_template_base.h"
#include "coll_batch_send_recv_retry_executor.h"
#undef private
#undef protected

using namespace checker;
using namespace hccl;

class BatchSendRecvTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "BatchSendRecvTest set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "BatchSendRecvTest tear down." << std::endl; }

    virtual void SetUp()
    {
        const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string caseName
            = "analysis_result_" + std::string(test_info->test_case_name()) + "_" + std::string(test_info->name());
        Checker::SetDumpFileName(caseName);
    }

    virtual void TearDown()
    {
        Checker::SetDumpFileName("analysis_result");
        // GlobalMockObject::verify();
        // 这边每个case执行完成需要清理所有的环境变量，如果有新增的环境变量，需要在这个函数中进行清理
        ClearHcclEnv();
    }
};

TEST_F(BatchSendRecvTest, batch_send_recv_allsendrecv_intrapod_2ranks)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BATCH_SEND_RECV;
    checkerOpParam.tag = "batchsendrecv";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.allRanksSendRecvInfoVec.resize(rankNum);
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BatchSendRecvTest, batch_send_recv_allsendrecv_intrapod_8ranks)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BATCH_SEND_RECV;
    checkerOpParam.tag = "batchsendrecv";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.allRanksSendRecvInfoVec.resize(rankNum);
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BatchSendRecvTest, batch_send_recv_allsendrecv_single_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BATCH_SEND_RECV;
    checkerOpParam.tag = "batchsendrecv";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.allRanksSendRecvInfoVec.resize(rankNum);
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BatchSendRecvTest, batch_send_recv_allsendrecv_rank)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BATCH_SEND_RECV;
    checkerOpParam.tag = "batchsendrecv";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910_93;
    checkerOpParam.DataDes.count = 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.allRanksSendRecvInfoVec.resize(rankNum);
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BatchSendRecvTest, batch_send_recv_allsendrecv_interpod_48ranks)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 4, 2, 8);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BATCH_SEND_RECV;
    checkerOpParam.tag = "batchsendrecv";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 1024 * 1024;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.allRanksSendRecvInfoVec.resize(rankNum);
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BatchSendRecvTest, batch_send_recv_allsendrecv_intrapod_8ranks_datasize_exceed_cclbuffersize)
{
    RankTable_For_LLT gen;
    TopoMeta topoMeta;
    gen.GenTopoMeta(topoMeta, 1, 1, 8);
    setenv("HCCL_BUFFSIZE", "1", 1);

    CheckerOpParam checkerOpParam;
    checkerOpParam.opType = CheckerOpType::BATCH_SEND_RECV;
    checkerOpParam.tag = "batchsendrecv";
    checkerOpParam.opMode = CheckerOpMode::OPBASE;
    checkerOpParam.devtype = CheckerDevType::DEV_TYPE_910B;
    checkerOpParam.DataDes.count = 1024 * 1024 * 10 + 1;
    checkerOpParam.DataDes.dataType = CheckerDataType::DATA_TYPE_INT8;
    u32 rankNum = GetRankNumFormTopoMeta(topoMeta);
    checkerOpParam.allRanksSendRecvInfoVec.resize(rankNum);
    Checker checker;
    HcclResult ret;
    ret = checker.Check(checkerOpParam, topoMeta);

    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
}

TEST_F(BatchSendRecvTest, batch_send_recv_Orchestrate)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclDispatcher dispatcher = nullptr;
    std::vector<std::vector<std::vector<u32>>> commPlaneRanks;
    std::vector<bool> isBridgeVector;
    hccl::HcclTopoInfo topoInfo;
    topoInfo.userRankSize = 1;
    topoInfo.userRank = 1;
    hccl::HcclAlgoInfo algoInfo;
    hccl::HcclExternalEnable externalEnable;
    std::vector<std::vector<std::vector<u32>>> serverAndSuperPodToRank;
    std::unique_ptr<hccl::TopoMatcher> topoMatcher = std::make_unique<hccl::TopoMatcher>(
        commPlaneRanks, isBridgeVector, topoInfo, algoInfo, externalEnable, serverAndSuperPodToRank);
    hccl::CollBatchSendRecvRetryExecutor* collBatchSendRecvExecutor
        = new hccl::CollBatchSendRecvRetryExecutor(dispatcher, topoMatcher);
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);

    std::vector<HcclSendRecvItem> itemVec(2);
    itemVec[0].remoteRank = 1;
    itemVec[0].buf = inputMem.ptr();
    itemVec[0].count = 1024;
    itemVec[0].dataType = HCCL_DATA_TYPE_FP32;
    itemVec[0].sendRecvType = HcclSendRecvType::HCCL_SEND;

    itemVec[1].remoteRank = 1;
    itemVec[1].buf = inputMem.ptr();
    itemVec[1].count = 1024;
    itemVec[1].dataType = HCCL_DATA_TYPE_FP32;
    itemVec[1].sendRecvType = HcclSendRecvType::HCCL_RECV;

    OpParam opParam;
    opParam.tag = "test";
    opParam.BatchSendRecvDataDes.sendRecvItemsPtr = itemVec.data();
    opParam.BatchSendRecvDataDes.itemNum = 2;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);
    opParam.BatchSendRecvDataDes.curIterNum = 0;
    opParam.BatchSendRecvDataDes.curMode = BatchSendRecvCurMode::SEND_RECV;
    u8 isDirectRemoteRank[16] = {0};
    opParam.BatchSendRecvDataDes.isDirectRemoteRank = isDirectRemoteRank;

    std::vector<HcclSendRecvItem*> sendRecvPair(2);
    sendRecvPair[0] = new HcclSendRecvItem();
    sendRecvPair[1] = new HcclSendRecvItem();
    sendRecvPair[0]->remoteRank = 1;
    sendRecvPair[0]->buf = inputMem.ptr();
    sendRecvPair[0]->count = 1024;
    sendRecvPair[0]->dataType = HCCL_DATA_TYPE_FP32;
    sendRecvPair[0]->sendRecvType = HcclSendRecvType::HCCL_SEND;

    sendRecvPair[1]->remoteRank = 1;
    sendRecvPair[1]->buf = inputMem.ptr();
    sendRecvPair[1]->count = 1024;
    sendRecvPair[1]->dataType = HCCL_DATA_TYPE_FP32;
    sendRecvPair[1]->sendRecvType = HcclSendRecvType::HCCL_RECV;

    MOCKER_CPP(&CollBatchSendRecvRetryExecutor::CalcSendSlices).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollBatchSendRecvRetryExecutor::CalcRecvSlices).stubs().will(returnValue(HCCL_SUCCESS));

    MachinePara machinePara;
    std::chrono::milliseconds timeout;

    AlgResourceResponse resourceResponse;
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    resourceResponse.opTransportResponse.resize(COMM_LEVEL_RESERVED);
    resourceResponse.opTransportResponse[COMM_COMBINE].resize(2);

    ret = collBatchSendRecvExecutor->RunLoop(opParam, resourceResponse, sendRecvPair);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete sendRecvPair[0];
    delete sendRecvPair[1];
    delete collBatchSendRecvExecutor;
}

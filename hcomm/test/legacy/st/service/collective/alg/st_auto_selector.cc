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
#include <mockcpp/mokc.h>

#include <iostream>
#include <vector>

#define private public
#include "dev_capability.h"

#include "virtual_topo.h"
#include "virtual_topo_stub.h"
#include "orion_adapter_rts.h"

#include "coll_alg_params.h"
#include "coll_operator.h"
#include "execute_selector.h"
#include "base_selector.h"
#undef private

using namespace Hccl;

class AutoSelectorTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AutoSelectorTest set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "AutoSelectorTest tear down" << std::endl; }
};

TEST_F(AutoSelectorTest, TestAutoSelectorOneTimesFour)
{
    MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_950));

    VirtualTopoStub virtTopo(0);
    string rankTable = "test";
    virtTopo.TopoInit91095OneTimesFour(rankTable);

    OpExecuteConfig opConfig;
    opConfig.accState = AcceleratorState::CCU_MS;

    CollAlgOperator opAllReduce;
    opAllReduce.opType = OpType::ALLREDUCE;
    opAllReduce.dataType = DataType::FP32;
    opAllReduce.dataCount = 1024 * 1024;
    CollAlgParams params;
    params.opExecuteConfig = opConfig;
    std::string allReduceAlgName;
    params.dataSize = opAllReduce.dataCount * DataTypeSizeGet(opAllReduce.dataType);
    HcclResult status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opAllReduce, params, allReduceAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(allReduceAlgName, "CcuAllReduceMesh1DMultiMission");
    std::cout << "The setted allreduce insCollAlgName: " << allReduceAlgName << std::endl;

    CollAlgOperator opAllGather;
    opAllGather.opType = OpType::ALLGATHER;
    opAllGather.dataType = DataType::FP32;
    opAllGather.dataCount = 1024 * 1024;
    std::string allGatherAlgName;
    params.dataSize = opAllGather.dataCount * DataTypeSizeGet(opAllGather.dataType);
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opAllGather, params, allGatherAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(allGatherAlgName, "CcuAllGatherMesh1DMultiMission");
    std::cout << "The setted allgather insCollAlgName: " << allGatherAlgName << std::endl;

    CollAlgOperator opReduceScatter;
    opReduceScatter.opType = OpType::REDUCESCATTER;
    opReduceScatter.dataType = DataType::FP32;
    opReduceScatter.dataCount = 1024 * 1024;
    std::string reduceScatterAlgName;
    params.dataSize = opReduceScatter.dataCount * DataTypeSizeGet(opReduceScatter.dataType);
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opReduceScatter, params, reduceScatterAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(reduceScatterAlgName, "CcuReduceScatterMesh1DMultiMission");
    std::cout << "The setted reducescatter insCollAlgName: " << reduceScatterAlgName << std::endl;

    CollAlgOperator opBroadcast;
    opBroadcast.opType = OpType::BROADCAST;
    opBroadcast.dataType = DataType::FP32;
    opBroadcast.dataCount = 1024 * 1024;
    std::string broadcastAlgName;
    params.dataSize = opBroadcast.dataCount * DataTypeSizeGet(opBroadcast.dataType);
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opBroadcast, params, broadcastAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(broadcastAlgName, "CcuBroadcastMeshMultiMission1D");
    std::cout << "The setted broadcast insCollAlgName: " << broadcastAlgName << std::endl;

    CollAlgOperator opReduce;
    opReduce.opType = OpType::REDUCE;
    opReduce.dataType = DataType::FP32;
    opReduce.dataCount = 1024 * 1024;
    std::string reduceAlgName;
    params.dataSize = opReduce.dataCount * DataTypeSizeGet(opReduce.dataType);
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opReduce, params, reduceAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(reduceAlgName, "CcuReduceMesh1DMultiMission");
    std::cout << "The setted reduce insCollAlgName: " << reduceAlgName << std::endl;

    CollAlgOperator opScatter;
    opScatter.opType = OpType::SCATTER;
    opScatter.dataType = DataType::FP32;
    std::string scatterAlgName;
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opScatter, params, scatterAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(scatterAlgName, "CcuScatterMesh1D");
    std::cout << "The setted scatter insCollAlgName: " << scatterAlgName << std::endl;

    CollAlgOperator opAlltoall;
    opAlltoall.opType = OpType::ALLTOALL;
    opAlltoall.dataType = DataType::FP32;
    std::string alltoallAlgName;
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opAlltoall, params, alltoallAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(alltoallAlgName, "CcuAlltoAllMesh1D");

    CollAlgOperator opAlltoallv;
    opAlltoallv.opType = OpType::ALLTOALLV;
    opAlltoallv.dataType = DataType::FP32;
    std::string alltoallvAlgName;
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opAlltoallv, params, alltoallvAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(alltoallvAlgName, "CcuAlltoAllVMesh1D");
    std::cout << "The setted alltoallv insCollAlgName: " << alltoallvAlgName << std::endl;

    std::cout << "The setted alltoall insCollAlgName: " << alltoallAlgName << std::endl;
}

TEST_F(AutoSelectorTest, TestAutoSelectorOneTimesFourCcuSchedule)
{
    MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_950));

    VirtualTopoStub virtTopo(0);
    string rankTable = "test";
    virtTopo.TopoInit91095OneTimesFour(rankTable);

    OpExecuteConfig opConfig;
    opConfig.accState = AcceleratorState::CCU_SCHED;

    CollAlgOperator opAllReduce;
    opAllReduce.opType = OpType::ALLREDUCE;
    opAllReduce.dataType = DataType::FP32;
    CollAlgParams params;
    params.opExecuteConfig = opConfig;
    std::string allReduceAlgName;
    HcclResult status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opAllReduce, params, allReduceAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(allReduceAlgName, "CcuAllReduceMeshMem2Mem1D");
    std::cout << "The setted allreduce insCollAlgName: " << allReduceAlgName << std::endl;

    CollAlgOperator opAllGather;
    opAllGather.opType = OpType::ALLGATHER;
    opAllGather.dataType = DataType::FP32;
    std::string allGatherAlgName;
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opAllGather, params, allGatherAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(allGatherAlgName, "CcuAllGatherMeshMem2Mem1D");
    std::cout << "The setted allgather insCollAlgName: " << allGatherAlgName << std::endl;

    CollAlgOperator opReduceScatter;
    opReduceScatter.opType = OpType::REDUCESCATTER;
    opReduceScatter.dataType = DataType::FP32;
    std::string reduceScatterAlgName;
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opReduceScatter, params, reduceScatterAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(reduceScatterAlgName, "CcuReduceScatterMeshMem2Mem1D");
    std::cout << "The setted reducescatter insCollAlgName: " << reduceScatterAlgName << std::endl;

    CollAlgOperator opBroadcast;
    opBroadcast.opType = OpType::BROADCAST;
    opBroadcast.dataType = DataType::FP32;
    std::string broadcastAlgName;
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opBroadcast, params, broadcastAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(broadcastAlgName, "CcuBroadcastMeshMem2Mem1D");
    std::cout << "The setted broadcast insCollAlgName: " << broadcastAlgName << std::endl;

    CollAlgOperator opAlltoall;
    opAlltoall.opType = OpType::ALLTOALL;
    opAlltoall.dataType = DataType::FP32;
    std::string alltoallAlgName;
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opAlltoall, params, alltoallAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(alltoallAlgName, "CcuAlltoAllMesh1D");

    CollAlgOperator opAlltoallv;
    opAlltoallv.opType = OpType::ALLTOALLV;
    opAlltoallv.dataType = DataType::FP32;
    std::string alltoallvAlgName;
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opAlltoallv, params, alltoallvAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(alltoallvAlgName, "CcuAlltoAllVMesh1D");
    std::cout << "The setted alltoallv insCollAlgName: " << alltoallvAlgName << std::endl;
    std::cout << "The setted alltoall insCollAlgName: " << alltoallAlgName << std::endl;
}

TEST_F(AutoSelectorTest, TestAutoSelectorTwoTimesTwo)
{
    MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_950));

    VirtualTopoStub virtTopo(0);
    string rankTable = "test";
    virtTopo.TopoInit91095TwoTimesTwo(rankTable);
    OpExecuteConfig opConfig;
    opConfig.accState = AcceleratorState::CCU_MS;

    CollAlgOperator opAllReduce;
    opAllReduce.opType = OpType::ALLREDUCE;
    opAllReduce.dataType = DataType::FP32;
    opAllReduce.dataCount = 255 * 1024;
    CollAlgParams params;
    params.opExecuteConfig = opConfig;
    std::string allReduceAlgName;
    params.dataSize = opAllReduce.dataCount * DataTypeSizeGet(opAllReduce.dataType);
    HcclResult status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opAllReduce, params, allReduceAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(allReduceAlgName, "CcuAllReduceMesh2DTwoShot");
    std::cout << "The setted insCollAlgName: " << allReduceAlgName << std::endl;

    opAllReduce.opType = OpType::ALLREDUCE;
    opAllReduce.dataType = DataType::FP32;
    opAllReduce.dataCount = 256 * 1024;
    params.opExecuteConfig = opConfig;
    params.dataSize = opAllReduce.dataCount * DataTypeSizeGet(opAllReduce.dataType);
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opAllReduce, params, allReduceAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(allReduceAlgName, "CcuAllReduceMesh2DTwoShotMultiMission");
    std::cout << "The setted insCollAlgName: " << allReduceAlgName << std::endl;

    CollAlgOperator opAllGather;
    opAllGather.opType = OpType::ALLGATHER;
    opAllGather.dataType = DataType::FP32;
    std::string allGatherAlgName;
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opAllGather, params, allGatherAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(allGatherAlgName, "CcuAllGatherMesh2DMultiMission");
    std::cout << "The setted allgather insCollAlgName: " << allGatherAlgName << std::endl;

    CollAlgOperator opReduceScatter;
    opReduceScatter.opType = OpType::REDUCESCATTER;
    opReduceScatter.dataType = DataType::FP32;
    std::string reduceScatterAlgName;
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opReduceScatter, params, reduceScatterAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(reduceScatterAlgName, "CcuReduceScatterMesh2DMultiMission");
    std::cout << "The setted reducescatter insCollAlgName: " << reduceScatterAlgName << std::endl;

    CollAlgOperator opBroadcast;
    opBroadcast.opType = OpType::BROADCAST;
    opBroadcast.dataType = DataType::FP32;
    opBroadcast.dataCount = 1024 * 255;
    std::string broadcastAlgName;
    params.dataSize = opBroadcast.dataCount * DataTypeSizeGet(opBroadcast.dataType);
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opBroadcast, params, broadcastAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(broadcastAlgName, "CcuBroadcastMesh2D");
    std::cout << "The setted broadcast insCollAlgName: " << broadcastAlgName << std::endl;

    CollAlgOperator opBroadcast2d;
    opBroadcast2d.opType = OpType::BROADCAST;
    opBroadcast2d.dataType = DataType::FP32;
    opBroadcast2d.dataCount = 1024 * 1024;
    params.dataSize = opBroadcast2d.dataCount * DataTypeSizeGet(opBroadcast2d.dataType);
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opBroadcast2d, params, broadcastAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(broadcastAlgName, "CcuBroadcastMesh2DMultiMission");
    std::cout << "The setted broadcast insCollAlgName: " << broadcastAlgName << std::endl;

    CollAlgOperator opReduce;
    opReduce.opType = OpType::REDUCE;
    opReduce.dataType = DataType::FP32;
    std::string reduceAlgName;
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opReduce, params, reduceAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(reduceAlgName, "CcuReduceMesh2D");
    std::cout << "The setted reduce insCollAlgName: " << reduceAlgName << std::endl;

    CollAlgOperator opAlltoallv;
    opAlltoallv.opType = OpType::ALLTOALLV;
    opAlltoallv.dataType = DataType::FP32;
    std::string alltoallvAlgName;
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opAlltoallv, params, alltoallvAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(alltoallvAlgName, "CcuAlltoAllVMesh2D");
    std::cout << "The setted alltoallv insCollAlgName: " << alltoallvAlgName << std::endl;
}

TEST_F(AutoSelectorTest, TestAutoSelectorTwoTimesTwoAicpu)
{
    MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_950));

    VirtualTopoStub virtTopo(0);
    string rankTable = "test";
    virtTopo.TopoInit91095TwoTimesTwo(rankTable);

    CollAlgOperator opAllReduce;
    opAllReduce.opType = OpType::ALLREDUCE;
    opAllReduce.dataType = DataType::FP32;
    OpExecuteConfig opConfig;
    opConfig.accState = AcceleratorState::AICPU_TS;

    CollAlgParams params;
    params.opExecuteConfig = opConfig;
    std::string allReduceAlgName;
    HcclResult status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opAllReduce, params, allReduceAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(allReduceAlgName, "InsAllReduceMesh2DTwoShot");
    std::cout << "The setted insCollAlgName: " << allReduceAlgName << std::endl;

    CollAlgOperator opAllGather;
    opAllGather.opType = OpType::ALLGATHER;
    opAllGather.dataType = DataType::FP32;
    std::string allGatherAlgName;
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opAllGather, params, allGatherAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(allGatherAlgName, "InsAllGatherMesh2D");
    std::cout << "The setted allgather insCollAlgName: " << allGatherAlgName << std::endl;

    CollAlgOperator opReduceScatter;
    opReduceScatter.opType = OpType::REDUCESCATTER;
    opReduceScatter.dataType = DataType::FP32;
    std::string reduceScatterAlgName;
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opReduceScatter, params, reduceScatterAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(reduceScatterAlgName, "InsReduceScatterMesh2D");
    std::cout << "The setted reducescatter insCollAlgName: " << reduceScatterAlgName << std::endl;

    CollAlgOperator opScatter;
    opScatter.opType = OpType::SCATTER;
    opScatter.dataType = DataType::FP32;
    std::string scatterAlgName;
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opScatter, params, scatterAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(scatterAlgName, "InsScatterMesh2D");
    std::cout << "The setted scatter insCollAlgName: " << scatterAlgName << std::endl;

    CollAlgOperator opBroadcast;
    opBroadcast.opType = OpType::BROADCAST;
    opBroadcast.dataType = DataType::FP32;
    std::string broadcastAlgName;
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opBroadcast, params, broadcastAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(broadcastAlgName, "InsBroadcastMesh2DTwoShot");
    std::cout << "The setted broadcast insCollAlgName: " << broadcastAlgName << std::endl;

    CollAlgOperator opReduce;
    opReduce.opType = OpType::REDUCE;
    opReduce.dataType = DataType::FP32;
    std::string reduceAlgName;
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opReduce, params, reduceAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(reduceAlgName, "InsReduceMesh2D");
    std::cout << "The setted reduce insCollAlgName: " << reduceAlgName << std::endl;

    CollAlgOperator opSend;
    opSend.opType = OpType::SEND;
    opSend.dataType = DataType::FP32;
    std::string sendAlgName;
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opSend, params, sendAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(sendAlgName, "InsSend");
    std::cout << "The setted send insCollAlgName: " << sendAlgName << std::endl;

    CollAlgOperator opRecv;
    opRecv.opType = OpType::RECV;
    opRecv.dataType = DataType::FP32;
    std::string recvAlgName;
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opRecv, params, recvAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(recvAlgName, "InsRecv");
    std::cout << "The setted recv insCollAlgName: " << recvAlgName << std::endl;

    CollAlgOperator opAlltoall;
    opAlltoall.opType = OpType::ALLTOALL;
    opAlltoall.dataType = DataType::FP32;
    std::string alltoallAlgName;
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opAlltoall, params, alltoallAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(alltoallAlgName, "InsAlltoAllMesh2D");
    std::cout << "The setted alltoall insCollAlgName: " << alltoallAlgName << std::endl;

    CollAlgOperator opAlltoallv;
    opAlltoallv.opType = OpType::ALLTOALLV;
    opAlltoallv.dataType = DataType::FP32;
    std::string alltoallvAlgName;
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opAlltoallv, params, alltoallvAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(alltoallvAlgName, "InsAlltoAllvMesh");
    std::cout << "The setted alltoallv insCollAlgName: " << alltoallvAlgName << std::endl;

    CollAlgOperator batchSendRecv;
    batchSendRecv.opType = OpType::BATCHSENDRECV;
    batchSendRecv.dataType = DataType::FP32;
    std::string batchSendRecvAlgName;
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(batchSendRecv, params, batchSendRecvAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(batchSendRecvAlgName, "InsBatchSendRecv");
    std::cout << "The setted batchSendRecv insCollAlgName: " << batchSendRecvAlgName << std::endl;

    std::string hcclAlgo = "level0:fullmesh;level1:NA";
    std::cout << hcclAlgo << std::endl;
    EnvConfig::GetInstance().algoCfg.hcclAlgoConfig.value = SetHcclAlgoConfig(hcclAlgo);
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opAllGather, params, allGatherAlgName);
    EXPECT_EQ(allGatherAlgName, "InsAllGatherMesh2D");
    std::cout << "The setted allgather insCollAlgName: " << allGatherAlgName << std::endl;
}

TEST_F(AutoSelectorTest, TestAutoSelectorTwoServerTimesTwoDefault)
{
    MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_950));
    VirtualTopoStub virtTopo(0);
    string rankTable = "test";
    virtTopo.TopoInit91095TwoServerTimesTwo(rankTable);

    CollAlgOperator opAllReduce;
    opAllReduce.opType = OpType::ALLREDUCE;
    opAllReduce.dataType = DataType::FP32;
    OpExecuteConfig opConfig;
    opConfig.accState = AcceleratorState::AICPU_TS;

    CollAlgParams params;
    params.opExecuteConfig = opConfig;
    std::string allReduceAlgName;
    HcclResult status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opAllReduce, params, allReduceAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(allReduceAlgName, "InsAllReduceParallelMesh1DNHR");
    std::cout << "The setted insCollAlgName: " << allReduceAlgName << std::endl;

    CollAlgOperator opAllGather;
    opAllGather.opType = OpType::ALLGATHER;
    opAllGather.dataType = DataType::FP32;
    std::string allGatherAlgName;
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opAllGather, params, allGatherAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(allGatherAlgName, "InsAllGatherParallelMesh1DNHR");
    std::cout << "The setted allgather insCollAlgName: " << allGatherAlgName << std::endl;

    CollAlgOperator opReduceScatter;
    opReduceScatter.opType = OpType::REDUCESCATTER;
    opReduceScatter.dataType = DataType::FP32;
    std::string reduceScatterAlgName;
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opReduceScatter, params, reduceScatterAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(reduceScatterAlgName, "InsReduceScatterParallelMesh1DNHR");
    std::cout << "The setted reducescatter insCollAlgName: " << reduceScatterAlgName << std::endl;

    CollAlgOperator opScatter;
    opScatter.opType = OpType::SCATTER;
    opScatter.dataType = DataType::FP32;
    std::string scatterAlgName;
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opScatter, params, scatterAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(scatterAlgName, "InsScatterParallelMesh1DNHR");
    std::cout << "The setted scatter insCollAlgName: " << scatterAlgName << std::endl;

    CollAlgOperator opBroadcast;
    opBroadcast.opType = OpType::BROADCAST;
    opBroadcast.dataType = DataType::FP32;
    std::string broadcastAlgName;
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opBroadcast, params, broadcastAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    std::cout << "The setted broadcast insCollAlgName: " << broadcastAlgName << std::endl;

    CollAlgOperator opReduce;
    opReduce.opType = OpType::REDUCE;
    opReduce.dataType = DataType::FP32;
    std::string reduceAlgName;
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opReduce, params, reduceAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(reduceAlgName, "InsReduceParallelMesh1DNHR");
    std::cout << "The setted reduce insCollAlgName: " << reduceAlgName << std::endl;

    CollAlgOperator opAlltoall;
    opAlltoall.opType = OpType::ALLTOALL;
    opAlltoall.dataType = DataType::FP32;
    std::string alltoallAlgName;
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opAlltoall, params, alltoallAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(alltoallAlgName, "InsAlltoAllMesh");
    std::cout << "The setted alltoall insCollAlgName: " << alltoallAlgName << std::endl;

    CollAlgOperator opAlltoallv;
    opAlltoallv.opType = OpType::ALLTOALLV;
    opAlltoallv.dataType = DataType::FP32;
    std::string alltoallvAlgName;
    status = ExecuteSelector().SetVirtualTopo(&virtTopo).Run(opAlltoallv, params, alltoallvAlgName);
    EXPECT_EQ(status, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(alltoallvAlgName, "InsAlltoAllvMesh");
    std::cout << "The setted alltoallv insCollAlgName: " << alltoallvAlgName << std::endl;
}

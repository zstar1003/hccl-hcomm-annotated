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

#define private public
#define protected public
#include "hccl_alg.h"
#include "hccl_impl.h"
#include "hccl_communicator.h"
#include "hccl_comm_pub.h"
#include "comm_impl.h"
#include "alg_template_base_pub.h"
#include "broadcast_operator.h"
#include "coll_all_gather_executor.h"
#include "coll_all_gather_mesh_opbase_pipeline_executor.h"
#include "coll_all_reduce_comm_executor.h"
#include "coll_all_reduce_ring_for_910_93_executor.h"
#include "coll_all_reduce_executor.h"
#include "coll_all_reduce_mesh_executor.h"
#include "coll_all_reduce_mesh_opbase_pipeline_executor.h"
#include "coll_all_reduce_reduce_plus_bcast_executor.h"
#include "coll_all_reduce_ring_executor.h"
#include "coll_reduce_executor.h"
#include "coll_reduce_scatter_comm_executor.h"
#include "coll_reduce_scatter_double_ring_concurrent_executor.h"
#include "coll_reduce_scatter_executor.h"
#include "coll_reduce_scatter_mesh_executor.h"
#include "coll_reduce_scatter_mesh_opbase_pipeline_executor.h"
#include "coll_reduce_scatter_ring_executor.h"
#include "coll_reduce_scatter_ring_for_910_93_executor.h"
#include "coll_scatter_executor.h"

#include "dispatcher_pub.h"
#include "comm_factory.h"
#include "externalinput.h"

#undef private
#undef protected
using namespace std;
using namespace hccl;

class CollExecutorMultiQpTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--CollExecutorMultiQpTest SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--CollExecutorMultiQpTest TearDown--\033[0m" << std::endl; }
    virtual void SetUp()
    {
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
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

#if 0
TEST_F(CollExecutorMultiQpTest, IsHugeData)
{
    HcclResult ret = HCCL_SUCCESS;
    setenv("HCCL_RDMA_QPS_PER_CONNECTION", "8", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    u32 qpsPerConnection = GetExternalInputQpsPerConnection();
    EXPECT_EQ(qpsPerConnection, 8);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl> &impl = implBase->implAlg_->pimpl_;
    std::unique_ptr<TopoMatcher> &topoMatcher = implBase->implAlg_->topoMatcher_;

    CollAllGatherExecutor* executor0 = new CollAllGatherExecutor(impl->dispatcher_, topoMatcher);
    bool bHuge = executor0->IsHugeData(100);
    EXPECT_EQ(bHuge, false);
    delete executor0;

    CollAllGatherMeshOpbasePipelineExecutor* executor1 = new CollAllGatherMeshOpbasePipelineExecutor(impl->dispatcher_, topoMatcher);
    bHuge = executor1->IsHugeData(100);
    EXPECT_EQ(bHuge, false);
    delete executor1;

    CollAllReduceCommExecutor* executor2 = new CollAllReduceCommExecutor(impl->dispatcher_, topoMatcher);
    bHuge = executor2->IsHugeData(100);
    EXPECT_EQ(bHuge, false);
    delete executor2;

    CollAllReduceDoubleRingConcurrentExecutor* executor3 = new CollAllReduceDoubleRingConcurrentExecutor(impl->dispatcher_, topoMatcher);
    bHuge = executor3->IsHugeData(100);
    EXPECT_EQ(bHuge, false);
    delete executor3;

    CollAllReduceRingFor91093Executor* executor4 = new CollAllReduceRingFor91093Executor(impl->dispatcher_, topoMatcher);
    bHuge = executor4->IsHugeData(100);
    EXPECT_EQ(bHuge, false);
    delete executor4;

    CollAllReduceExecutor* executor5 = new CollAllReduceExecutor(impl->dispatcher_, topoMatcher);
    bHuge = executor5->IsHugeData(100);
    EXPECT_EQ(bHuge, false);
    delete executor5;

    CollAllReduceMeshExecutor* executor6 = new CollAllReduceMeshExecutor(impl->dispatcher_, topoMatcher);
    bHuge = executor6->IsHugeData(100);
    EXPECT_EQ(bHuge, false);
    delete executor6;

    CollAllReduceMeshOpbasePipelineExecutor* executor7 = new CollAllReduceMeshOpbasePipelineExecutor(impl->dispatcher_, topoMatcher);
    bHuge = executor7->IsHugeData(100);
    EXPECT_EQ(bHuge, false);
    delete executor7;

    CollAllReduceReducePlusBcastExecutor* executor8 = new CollAllReduceReducePlusBcastExecutor(impl->dispatcher_, topoMatcher);
    bHuge = executor8->IsHugeData(100);
    EXPECT_EQ(bHuge, false);
    delete executor8;

    CollAllReduceRingExecutor* executor9 = new CollAllReduceRingExecutor(impl->dispatcher_, topoMatcher);
    bHuge = executor9->IsHugeData(100);
    EXPECT_EQ(bHuge, false);
    delete executor9;

    CollReduceExecutor* executor10 = new CollReduceExecutor(impl->dispatcher_, topoMatcher);
    bHuge = executor10->IsHugeData(100);
    EXPECT_EQ(bHuge, false);
    delete executor10;

    CollReduceScatterCommExecutor* executor11 = new CollReduceScatterCommExecutor(impl->dispatcher_, topoMatcher);
    bHuge = executor11->IsHugeData(100);
    EXPECT_EQ(bHuge, false);
    delete executor11;

    MOCKER_CPP(&CollNativeExecutorBase::CheckCommSize)
              .stubs()
              .will(returnValue(HCCL_SUCCESS));

    SubCommInfo level2CommInfo;
    level2CommInfo.localRankSize = 2;
    MOCKER_CPP(&CollNativeExecutorBase::GetSubCommInfo)
              .stubs()
              .will(returnValue(level2CommInfo));

    CollReduceScatterDoubleRingConcurrentExecutor* executor12 = new CollReduceScatterDoubleRingConcurrentExecutor(impl->dispatcher_, topoMatcher);
    bHuge = executor12->IsHugeData(100);
    EXPECT_EQ(bHuge, false);
    delete executor12;

    CollReduceScatterExecutor* executor13 = new CollReduceScatterExecutor(impl->dispatcher_, topoMatcher);
    bHuge = executor13->IsHugeData(100);
    EXPECT_EQ(bHuge, false);
    delete executor13;

    CollReduceScatterMeshExecutor* executor14 = new CollReduceScatterMeshExecutor(impl->dispatcher_, topoMatcher);
    bHuge = executor14->IsHugeData(100);
    EXPECT_EQ(bHuge, false);
    delete executor14;

    CollReduceScatterMeshOpbasePipelineExecutor* executor15 = new CollReduceScatterMeshOpbasePipelineExecutor(impl->dispatcher_, topoMatcher);
    bHuge = executor15->IsHugeData(100);
    EXPECT_EQ(bHuge, false);
    delete executor15;

    CollReduceScatterRingExecutor* executor16 = new CollReduceScatterRingExecutor(impl->dispatcher_, topoMatcher);
    bHuge = executor16->IsHugeData(100);
    EXPECT_EQ(bHuge, false);
    delete executor16;

    OpParam param;
    param.inputPtr = (void *)0x100;
    param.outputPtr = (void *)0x100;
    param.DataDes.dataType = HCCL_DATA_TYPE_INT8;
    param.reduceType = HCCL_REDUCE_PROD;
    CollReduceScatterRingFor91093Executor* executor17 = new CollReduceScatterRingFor91093Executor(impl->dispatcher_, topoMatcher);
    bHuge = executor17->IsHugeData(100, &param);
    EXPECT_EQ(bHuge, false);
    delete executor17;

    CollScatterExecutor* executor18 = new CollScatterExecutor(impl->dispatcher_, topoMatcher);
    bHuge = executor18->IsHugeData(100);
    EXPECT_EQ(bHuge, false);
    delete executor18;

    CollAllReduceRingFor91093Executor* executor19 = new CollAllReduceRingFor91093Executor(impl->dispatcher_, topoMatcher);
    bool smallData = executor19->IsSmallData(100, 100);
    EXPECT_EQ(smallData, false);
    delete executor19;

    unsetenv("HCCL_RDMA_QPS_PER_CONNECTION");
    ResetInitState();
}
#endif

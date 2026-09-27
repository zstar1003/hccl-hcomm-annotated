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
#include "reduce_operator.h"
#include "coll_comm_executor.h"
#include "dispatcher_pub.h"
#include "comm_factory.h"
#include "externalinput.h"

#undef private
#undef protected
using namespace std;
using namespace hccl;

class CollReduceInterTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        s32 ret = HcclDispatcherInit(DispatcherType::DISPATCHER_NORMAL, 0, &dispatcherPtr);
        if (ret != HCCL_SUCCESS)
            return;
        if (dispatcherPtr == nullptr)
            return;
        dispatcher = reinterpret_cast<DispatcherPub*>(dispatcherPtr);
        std::cout << "\033[36m--CollReduceInterTest SetUP--\033[0m" << std::endl;
    }
    static void TearDownTestCase()
    {
        if (dispatcherPtr != nullptr) {
            s32 ret = HcclDispatcherDestroy(dispatcherPtr);
            EXPECT_EQ(ret, HCCL_SUCCESS);
            dispatcherPtr = nullptr;
            dispatcher = nullptr;
        }
        std::cout << "\033[36m--CollReduceInterTest TearDown--\033[0m" << std::endl;
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

HcclDispatcher CollReduceInterTest::dispatcherPtr = nullptr;
DispatcherPub* CollReduceInterTest::dispatcher = nullptr;

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
TEST_F(CollReduceInterTest, ut_reduce_test_910B) {
    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    
    MOCKER(GetWorkflowMode)
    .stubs()
    .will(returnValue(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));

    std::unique_ptr<hcclImpl> &impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    impl->deviceLogicId_ = 0;
    impl->devicePhyId_ = 0;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_4P_MESH;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_HD;
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
    opParam.DataDes.count = 2048/4;
    opParam.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);
    opParam.root = 0;
    std::string algName;
    std::string newTag;
    std::unique_ptr<TopoMatcher> &topoMatcher = implBase->implAlg_->topoMatcher_;
    CCLBufferManager &cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;
    std::unique_ptr<ReduceOperator> operation(new (std::nothrow) ReduceOperator(algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher));
    ret = operation->SelectAlg("", opParam, algName, newTag);
    operation = nullptr;
}
#endif

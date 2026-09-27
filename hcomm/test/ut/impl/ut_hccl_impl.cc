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
#include "workflow_pub.h"
#include "topoinfo_struct.h"
#include "hccl_ip_address.h"
#include "dlra_function.h"
#include "dltdt_function.h"
#include "acl/acl.h"
#include "hccl_comm.h"
#include "hccl_inner.h"
#define private public
#define protected public
#include "hccl_impl.h"
#include "config.h"
#include "hccl_communicator.h"
#include "hccl_communicator_attrs.h"
#include "network_manager_pub.h"
#include "transport_base_pub.h"
#include "transport_p2p_pub.h"
#include "comm_impl.h"
#include "comm_mesh_pub.h"
#include "dispatcher_pub.h"
#include "externalinput.h"
#include "coll_alg_operator.h"
#include "all_gather_operator.h"
#include "reduce_operator.h"
#include "reduce_scatter_operator.h"
#include "broadcast_operator.h"
#include "coll_comm_executor.h"
#include "comm_base_pub.h"
#include "task_abort_handler_pub.h"
#include "adapter_rts.h"
#include "coll_reduce_scatter_v_executor.h"
#include "coll_all_gather_v_executor.h"
#include "coll_reduce_scatter_v_mesh_opbase_executor.h"
#include "coll_all_gather_v_mesh_opbase_executor.h"
#include "coll_all_gather_v_executor.h"
#include "coll_all_gatherv_mesh_aiv_executor.h"
#include "coll_all_gatherv_mesh_aiv_smallcount_executor.h"
#include "coll_all_gather_mesh_aiv_for_910_93_executor.h"
#include "coll_all_gather_v_mesh_executor.h"
#include "coll_aligned_all_gather_v_double_ring_for_910_93_executor.h"
#include "coll_reduce_scatter_v_aiv_big_count_executor.h"
#include "coll_reduce_scatter_v_mesh_aiv_smallcount_executor.h"
#include "coll_reduce_scatter_v_for_310p_ring_executor.h"
#include "coll_all_gatherv_for_310p_executor.h"
#undef private
#undef protected
#include "tbe_vector_reduce.h"
#include "tbe_crack_cleared.h"
#include "param_check_pub.h"
#include "base.h"
#include "adapter_rts_common.h"
#include "coll_all_reduce_ring_executor.h"
#include "coll_all_gather_ring_executor.h"
#include "coll_aligned_all_reduce_double_ring_for_910_93_executor.h"
#include "tbe_vector_reduce.h"
#include "all_gather_v_operator.h"
#include "reduce_scatter_v_operator.h"
#include "profiling_manager.h"
#include "heartbeat.h"

using namespace hccl;
using namespace std;
class HcclImplTest : public testing::Test {
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
        std::cout << "HcclImplTest SetUP" << std::endl;
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
        std::cout << "HcclImplTest TearDown" << std::endl;
    }
    // Some expensive resource shared by all tests.
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
        unsetenv("HCCL_ALGO");
    }
    static void TestConstructParam(HcclCommParams& params, RankTable_t& rankTable)
    {
        string commId = "comm ";
        memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1);
        params.rank = 0;
        params.userRank = 0;
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
    static void TestConstructParamForOneServer(HcclCommParams& params, RankTable_t& rankTable)
    {
        string commId = "comm ";
        memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1);
        params.rank = 0;
        params.userRank = 0;
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
        rankVec[1].deviceInfo.devicePhyId = 1;
        HcclIpAddress ipAddr2(1711319232);
        rankVec[1].deviceInfo.deviceIp.push_back(ipAddr2); // 101.0.168.192
        rankVec[1].serverIdx = 0;
        rankVec[1].serverId = "192.168.0.101";
        rankTable.rankList.assign(rankVec.begin(), rankVec.end());
        rankTable.deviceNum = 2;
        rankTable.serverNum = 1;
    }

    static void TestConstructParam_2Server(HcclCommParams& params, RankTable_t& rankTable)
    {
        string commId = "comm ";
        memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1);
        params.rank = 0;
        params.userRank = 0;
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

    static void TestConstructParam_1server_4p(HcclCommParams& params, RankTable_t& rankTable)
    {
        string commId = "comm ";
        memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1);
        params.rank = 0;
        params.userRank = 0;
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
        rankVec[2].deviceInfo.devicePhyId = 2;
        HcclIpAddress ipAddr3(1711319232);
        rankVec[2].deviceInfo.deviceIp.push_back(ipAddr3); // 101.0.168.192
        rankVec[2].serverIdx = 0;
        rankVec[2].serverId = "192.168.0.101";
        rankVec[3].rankId = 3;
        rankVec[3].deviceInfo.devicePhyId = 3;
        HcclIpAddress ipAddr4(1711319233);
        rankVec[3].deviceInfo.deviceIp.push_back(ipAddr4); // 101.0.168.192
        rankVec[3].serverIdx = 0;
        rankVec[3].serverId = "192.168.0.101";
        rankTable.rankList.assign(rankVec.begin(), rankVec.end());
        rankTable.rankNum = 4;
        rankTable.deviceNum = 4;
        rankTable.serverNum = 1;
    }
    static HcclCommParams params;
    static RankTable_t rankTable;

    static HcclDispatcher dispatcherPtr;
    static DispatcherPub* dispatcher;
};
HcclDispatcher HcclImplTest::dispatcherPtr = nullptr;
DispatcherPub* HcclImplTest::dispatcher = nullptr;

HcclCommParams HcclImplTest::params;
RankTable_t HcclImplTest::rankTable;

TEST_F(HcclImplTest, ut_hcclimpl_GetInnerServerAverageDevice)
{
    HcclCommunicator impl;
    HcclResult ret;
    std::vector<RankInfo> ranks;

    RankInfo tmp_para_0;

    tmp_para_0.userRank = 0;
    tmp_para_0.devicePhyId = 0;
    tmp_para_0.deviceType = DevType::DEV_TYPE_910;
    tmp_para_0.serverIdx = 0;
    tmp_para_0.serverId = "10.0.0.10";
    tmp_para_0.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_1;

    tmp_para_1.userRank = 1;
    tmp_para_1.devicePhyId = 1;
    tmp_para_1.deviceType = DevType::DEV_TYPE_910;
    tmp_para_1.serverId = "10.0.1.10";
    tmp_para_1.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    ranks.push_back(tmp_para_0);
    ranks.push_back(tmp_para_1);

    impl.attrCollector_.serverId_ = "10.0.0.10";
    ret = impl.attrCollector_.SetInnerServerAverageDevice(ranks);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_GetCqeError)
{
    std::unique_ptr<HcclCommunicator> impl(new (std::nothrow) HcclCommunicator());

    impl->deviceLogicId_ = 0;

    HcclResult result;

    impl->GetCqeError(result);
}

TEST_F(HcclImplTest, hcclComm_test_algo)
{
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&OpRetryManager::SetRetryStateToWaitResume).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&OpRetryManager::ExitWaitResumeState).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;

    std::unique_ptr<CollAlgOperator> operation(new (std::nothrow) CollAlgOperator(
        algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_ALL));

    NetworkManager::GetInstance(15).Destroy();

    NetworkManager::GetInstance(15).hostNicInitRef_.Ref();
    NetworkManager::GetInstance(15).isRaInitRepeated_ = false;
    NetworkManager::GetInstance(15).Destroy();
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, hcclComm_test_ring_algo)
{
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;

    std::unique_ptr<CollAlgOperator> operation(new (std::nothrow) CollAlgOperator(
        algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_ALL));

    NetworkManager::GetInstance(15).Destroy();
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, hcclComm_test_nhr_algo)
{
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;

    std::unique_ptr<CollAlgOperator> operation(new (std::nothrow) CollAlgOperator(
        algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_ALL));

    NetworkManager::GetInstance(15).Destroy();
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hcclimpl_AutoSelectAlgTypeLevel1_REDUCE_SCATTER)
{
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    u64 curCount = 100; // 单位：字节(B)
    HcclCMDType hcclCMDType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_UINT64;
    u32 unitSize = SIZE_TABLE[dataType];
    u64 curSize = curCount * unitSize;
    std::string algTypeLevel1Tag;
    impl->moduleNum_ = 2;
    impl->deviceNumPerAggregation_ = 8;
    impl->userRankSize_ = impl->moduleNum_ * impl->deviceNumPerAggregation_;
    impl->deviceType_ = DevType::DEV_TYPE_910;
    algConfigurator->isAlgoLevel1Default_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER] = true;

    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;

    topoMatcher->topoInfo_.deviceNumPerAggregation = impl->deviceNumPerAggregation_;
    topoMatcher->topoInfo_.userRankSize = impl->userRankSize_;
    topoMatcher->topoInfo_.deviceType = impl->deviceType_;
    std::unique_ptr<CollAlgOperator> operation(new (std::nothrow) CollAlgOperator(
        algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_REDUCE_SCATTER));
    ret = operation->AutoSelectAlgTypeLevel1(hcclCMDType, curSize, curSize, algTypeLevel1Tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(algTypeLevel1Tag, "ALG_LEVEL1_RING");
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hcclimpl_AutoSelectAlgTypeLevel1_REDUCE_SCATTER_15SEVER)
{
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;

    u64 curCount = 21000; // 单位：字节(B) 0.126MB
    HcclCMDType hcclCMDType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_UINT64;
    u32 unitSize = SIZE_TABLE[dataType];
    u64 curSize = curCount * unitSize;
    std::string algTypeLevel1Tag;
    implBase->implAlg_->topoAttr_.moduleNum = 15;
    implBase->implAlg_->topoAttr_.deviceNumPerAggregation = 8;
    implBase->implAlg_->topoAttr_.userRankSize
        = implBase->implAlg_->topoAttr_.moduleNum * implBase->implAlg_->topoAttr_.deviceNumPerAggregation;
    implBase->implAlg_->topoAttr_.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.moduleNum = implBase->implAlg_->topoAttr_.moduleNum;
    topoMatcher->topoInfo_.deviceNumPerAggregation = implBase->implAlg_->topoAttr_.deviceNumPerAggregation;
    topoMatcher->topoInfo_.userRankSize = implBase->implAlg_->topoAttr_.userRankSize;
    topoMatcher->topoInfo_.deviceType = implBase->implAlg_->topoAttr_.deviceType;
    algConfigurator->isAlgoLevel1Default_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER] = true;
    std::unique_ptr<CollAlgOperator> operation(new (std::nothrow) CollAlgOperator(
        algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_REDUCE_SCATTER));
    ret = operation->AutoSelectAlgTypeLevel1(hcclCMDType, curSize, curSize, algTypeLevel1Tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(algTypeLevel1Tag, "ALG_LEVEL1_NHR");
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hcclimpl_AutoSelectAlgTypeLevel1_REDUCE_SCATTER_11SEVER_1)
{
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;

    u64 curCount = 21000; // 单位：字节(B) 0.126MB
    HcclCMDType hcclCMDType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_UINT64;
    u32 unitSize = SIZE_TABLE[dataType];
    u64 curSize = curCount * unitSize;
    std::string algTypeLevel1Tag;
    implBase->implAlg_->topoAttr_.moduleNum = 11;
    implBase->implAlg_->topoAttr_.deviceNumPerAggregation = 8;
    implBase->implAlg_->topoAttr_.userRankSize
        = implBase->implAlg_->topoAttr_.moduleNum * implBase->implAlg_->topoAttr_.deviceNumPerAggregation;
    implBase->implAlg_->topoAttr_.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.moduleNum = implBase->implAlg_->topoAttr_.moduleNum;
    topoMatcher->topoInfo_.deviceNumPerAggregation = implBase->implAlg_->topoAttr_.deviceNumPerAggregation;
    topoMatcher->topoInfo_.userRankSize = implBase->implAlg_->topoAttr_.userRankSize;
    topoMatcher->topoInfo_.deviceType = implBase->implAlg_->topoAttr_.deviceType;
    algConfigurator->isAlgoLevel1Default_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER] = true;
    std::unique_ptr<CollAlgOperator> operation(new (std::nothrow) CollAlgOperator(
        algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_REDUCE_SCATTER));
    ret = operation->AutoSelectAlgTypeLevel1(hcclCMDType, curSize, curSize, algTypeLevel1Tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(algTypeLevel1Tag, "ALG_LEVEL1_NHR");
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hcclimpl_AutoSelectAlgTypeLevel1_REDUCE_SCATTER_11SEVER_2)
{
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;

    u64 curCount = 3900; // 单位：字节(B) 0.0312MB
    HcclCMDType hcclCMDType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_UINT64;
    u32 unitSize = SIZE_TABLE[dataType];
    u64 curSize = curCount * unitSize;
    std::string algTypeLevel1Tag;
    implBase->implAlg_->topoAttr_.moduleNum = 11;
    implBase->implAlg_->topoAttr_.deviceNumPerAggregation = 8;
    implBase->implAlg_->topoAttr_.userRankSize
        = implBase->implAlg_->topoAttr_.moduleNum * implBase->implAlg_->topoAttr_.deviceNumPerAggregation;
    implBase->implAlg_->topoAttr_.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.moduleNum = implBase->implAlg_->topoAttr_.moduleNum;
    topoMatcher->topoInfo_.deviceNumPerAggregation = implBase->implAlg_->topoAttr_.deviceNumPerAggregation;
    topoMatcher->topoInfo_.userRankSize = implBase->implAlg_->topoAttr_.userRankSize;
    topoMatcher->topoInfo_.deviceType = implBase->implAlg_->topoAttr_.deviceType;
    algConfigurator->isAlgoLevel1Default_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER] = true;
    std::unique_ptr<CollAlgOperator> operation(new (std::nothrow) CollAlgOperator(
        algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_REDUCE_SCATTER));
    ret = operation->AutoSelectAlgTypeLevel1(hcclCMDType, curSize, curSize, algTypeLevel1Tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(algTypeLevel1Tag, "ALG_LEVEL1_NHR");
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hcclimpl_AutoSelectAlgTypeLevel1_ALLGATHER)
{
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;

    u64 curCount = 100; // 单位：字节(B)
    HcclCMDType hcclCMDType = HcclCMDType::HCCL_CMD_ALLGATHER;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_UINT64;
    u32 unitSize = SIZE_TABLE[dataType];
    u64 curSize = curCount * unitSize;
    std::string algTypeLevel1Tag;
    implBase->implAlg_->topoAttr_.moduleNum = 2;
    implBase->implAlg_->topoAttr_.deviceNumPerAggregation = 8;
    implBase->implAlg_->topoAttr_.userRankSize
        = implBase->implAlg_->topoAttr_.moduleNum * implBase->implAlg_->topoAttr_.deviceNumPerAggregation;
    implBase->implAlg_->topoAttr_.deviceType = DevType::DEV_TYPE_910;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceNumPerAggregation = implBase->implAlg_->topoAttr_.deviceNumPerAggregation;
    topoMatcher->topoInfo_.userRankSize = implBase->implAlg_->topoAttr_.userRankSize;
    topoMatcher->topoInfo_.deviceType = implBase->implAlg_->topoAttr_.deviceType;
    algConfigurator->isAlgoLevel1Default_[HcclCMDType::HCCL_CMD_ALLGATHER] = true;
    std::unique_ptr<CollAlgOperator> operation(new (std::nothrow) CollAlgOperator(
        algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_ALLGATHER));
    ret = operation->AutoSelectAlgTypeLevel1(hcclCMDType, curSize, curSize, algTypeLevel1Tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(algTypeLevel1Tag, "ALG_LEVEL1_RING");
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hcclimpl_AutoSelectAlgTypeLevel1_ALLGATHER_15SEVER)
{
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    DevType deviceType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;

    u64 curCount = 21000; // 单位：字节(B) 0.126MB
    HcclCMDType hcclCMDType = HcclCMDType::HCCL_CMD_ALLGATHER;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_UINT64;
    u32 unitSize = SIZE_TABLE[dataType];
    u64 curSize = curCount * unitSize;
    std::string algTypeLevel1Tag;
    implBase->implAlg_->topoAttr_.moduleNum = 15;
    implBase->implAlg_->topoAttr_.deviceNumPerAggregation = 8;
    implBase->implAlg_->topoAttr_.userRankSize
        = implBase->implAlg_->topoAttr_.moduleNum * implBase->implAlg_->topoAttr_.deviceNumPerAggregation;
    implBase->implAlg_->topoAttr_.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.moduleNum = implBase->implAlg_->topoAttr_.moduleNum;
    topoMatcher->topoInfo_.deviceNumPerAggregation = implBase->implAlg_->topoAttr_.deviceNumPerAggregation;
    topoMatcher->topoInfo_.userRankSize = implBase->implAlg_->topoAttr_.userRankSize;
    topoMatcher->topoInfo_.deviceType = implBase->implAlg_->topoAttr_.deviceType;
    algConfigurator->isAlgoLevel1Default_[HcclCMDType::HCCL_CMD_ALLGATHER] = true;
    std::unique_ptr<CollAlgOperator> operation(new (std::nothrow) CollAlgOperator(
        algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_ALLGATHER));
    ret = operation->AutoSelectAlgTypeLevel1(hcclCMDType, curSize, curSize, algTypeLevel1Tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(algTypeLevel1Tag, "ALG_LEVEL1_NHR");
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hcclimpl_AutoSelectAlgTypeLevel1_ALLGATHER_11SEVER_1)
{
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;

    u64 curCount = 21000; // 单位：字节(B) 0.126MB
    HcclCMDType hcclCMDType = HcclCMDType::HCCL_CMD_ALLGATHER;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_UINT64;
    u32 unitSize = SIZE_TABLE[dataType];
    u64 curSize = curCount * unitSize;
    std::string algTypeLevel1Tag;
    implBase->implAlg_->topoAttr_.moduleNum = 11;
    implBase->implAlg_->topoAttr_.deviceNumPerAggregation = 8;
    implBase->implAlg_->topoAttr_.userRankSize
        = implBase->implAlg_->topoAttr_.moduleNum * implBase->implAlg_->topoAttr_.deviceNumPerAggregation;
    implBase->implAlg_->topoAttr_.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.moduleNum = implBase->implAlg_->topoAttr_.moduleNum;
    topoMatcher->topoInfo_.deviceNumPerAggregation = implBase->implAlg_->topoAttr_.deviceNumPerAggregation;
    topoMatcher->topoInfo_.userRankSize = implBase->implAlg_->topoAttr_.userRankSize;
    topoMatcher->topoInfo_.deviceType = implBase->implAlg_->topoAttr_.deviceType;
    algConfigurator->isAlgoLevel1Default_[HcclCMDType::HCCL_CMD_ALLGATHER] = true;
    std::unique_ptr<CollAlgOperator> operation(new (std::nothrow) CollAlgOperator(
        algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_ALLGATHER));
    ret = operation->AutoSelectAlgTypeLevel1(hcclCMDType, curSize, curSize, algTypeLevel1Tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(algTypeLevel1Tag, "ALG_LEVEL1_NHR");
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hcclimpl_AutoSelectAlgTypeLevel1_ALLGATHER_11SEVER_2)
{
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;

    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    std::unique_ptr<CollAlgOperator> operation(new (std::nothrow) CollAlgOperator(
        algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_ALLGATHER));
    u64 curCount = 3900; // 单位：字节(B) 0.0312MB
    HcclCMDType hcclCMDType = HcclCMDType::HCCL_CMD_ALLGATHER;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_UINT64;
    u32 unitSize = SIZE_TABLE[dataType];
    u64 curSize = curCount * unitSize;
    std::string algTypeLevel1Tag;
    operation->moduleNum_ = 11;
    operation->deviceNumPerAggregation_ = 8;
    operation->userRankSize_ = operation->moduleNum_ * operation->deviceNumPerAggregation_;
    operation->deviceType_ = DevType::DEV_TYPE_910B;
    operation->isAlgoLevel1Default_ = true;
    ret = operation->AutoSelectAlgTypeLevel1(hcclCMDType, curSize, curSize, algTypeLevel1Tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(algTypeLevel1Tag, "ALG_LEVEL1_NHR");
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hcclimpl_AutoSelectAlgTypeLevel1_ALLREDUCE)
{
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;

    std::unique_ptr<CollAlgOperator> operation(new (std::nothrow) CollAlgOperator(
        algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_ALLREDUCE));
    u64 curCount = 100; // 单位：字节(B)
    HcclCMDType hcclCMDType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_UINT64;
    u32 unitSize = SIZE_TABLE[dataType];
    u64 curSize = curCount * unitSize;
    std::string algTypeLevel1Tag;
    operation->moduleNum_ = 2;
    operation->deviceNumPerAggregation_ = 8;
    operation->userRankSize_ = operation->moduleNum_ * operation->deviceNumPerAggregation_;
    operation->deviceType_ = DevType::DEV_TYPE_910;
    operation->isAlgoLevel1Default_ = true;
    ret = operation->AutoSelectAlgTypeLevel1(hcclCMDType, curSize, curSize, algTypeLevel1Tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(algTypeLevel1Tag, "ALG_LEVEL1_RING");
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hcclimpl_AutoSelectAlgTypeLevel1_ALLREDUCE_15SEVER)
{
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    DevType deviceType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;

    std::unique_ptr<CollAlgOperator> operation(new (std::nothrow) CollAlgOperator(
        algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_ALLREDUCE));
    u64 curCount = 21000000; // 单位：字节(B) 126MB
    HcclCMDType hcclCMDType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_UINT64;
    u32 unitSize = SIZE_TABLE[dataType];
    u64 curSize = curCount * unitSize;
    std::string algTypeLevel1Tag;
    operation->moduleNum_ = 15;
    operation->deviceNumPerAggregation_ = 8;
    operation->userRankSize_ = operation->moduleNum_ * operation->deviceNumPerAggregation_;
    operation->deviceType_ = DevType::DEV_TYPE_910B;
    operation->isAlgoLevel1Default_ = true;
    ret = operation->AutoSelectAlgTypeLevel1(hcclCMDType, curSize, curSize, algTypeLevel1Tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(algTypeLevel1Tag, "ALG_LEVEL1_NHR");
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hcclimpl_AutoSelectAlgTypeLevel1_ALLREDUCE_11SEVER_1)
{
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;

    std::unique_ptr<CollAlgOperator> operation(new (std::nothrow) CollAlgOperator(
        algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_ALLREDUCE));
    u64 curCount = 21000000; // 单位：字节(B) 126MB
    HcclCMDType hcclCMDType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_UINT64;
    u32 unitSize = SIZE_TABLE[dataType];
    u64 curSize = curCount * unitSize;
    std::string algTypeLevel1Tag;
    operation->moduleNum_ = 11;
    operation->deviceNumPerAggregation_ = 8;
    operation->userRankSize_ = operation->moduleNum_ * operation->deviceNumPerAggregation_;
    operation->deviceType_ = DevType::DEV_TYPE_910B;
    operation->isAlgoLevel1Default_ = true;
    ret = operation->AutoSelectAlgTypeLevel1(hcclCMDType, curSize, curSize, algTypeLevel1Tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(algTypeLevel1Tag, "ALG_LEVEL1_NHR");
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hcclimpl_AutoSelectAlgTypeLevel1_ALLREDUCE_11SEVER_2)
{
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;

    std::unique_ptr<CollAlgOperator> operation(new (std::nothrow) CollAlgOperator(
        algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_ALLREDUCE));
    u64 curCount = 2000000; // 单位：字节(B) 16MB
    HcclCMDType hcclCMDType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_UINT64;
    u32 unitSize = SIZE_TABLE[dataType];
    u64 curSize = curCount * unitSize;
    std::string algTypeLevel1Tag;
    operation->moduleNum_ = 11;
    operation->deviceNumPerAggregation_ = 8;
    operation->userRankSize_ = operation->moduleNum_ * operation->deviceNumPerAggregation_;
    operation->deviceType_ = DevType::DEV_TYPE_910B;
    operation->isAlgoLevel1Default_ = true;
    ret = operation->AutoSelectAlgTypeLevel1(hcclCMDType, curSize, curSize, algTypeLevel1Tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(algTypeLevel1Tag, "ALG_LEVEL1_NHR");
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hcclimpl_SelectAlgoTypeForReduceScatter)
{
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;

    std::unique_ptr<CollAlgOperator> operation(new (std::nothrow) CollAlgOperator(
        algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_REDUCE_SCATTER));
    float delay = 60;
    u64 curSize = 100; // 单位：字节(B)
    u32 deviceNumPerAggregation = 8;
    float bandWidth = 0.005;
    AlgTypeLevel1 algType;

    operation->moduleNum_ = 2;
    operation->deviceNumPerAggregation_ = deviceNumPerAggregation;
    operation->userRankSize_ = operation->moduleNum_ * operation->deviceNumPerAggregation_;
    ret = operation->SelectAlgoTypeForReduceScatter(delay, curSize, bandWidth, algType);
    auto iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType);
    if (iter == HCCL_ALGO_LEVEL1_NAME_MAP.end()) {
        HCCL_ERROR("level1: algType[%u] is invalid.", algType);
        ret = HCCL_E_INTERNAL;
    }
    HCCL_RUN_INFO(
        "hccl algorithm: there are %u server in level1, using %s algo.", operation->moduleNum_, iter->second.c_str());
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(algType, AlgTypeLevel1::ALG_LEVEL1_RING);

    operation->moduleNum_ = 3;
    operation->deviceNumPerAggregation_ = deviceNumPerAggregation;
    operation->userRankSize_ = operation->moduleNum_ * operation->deviceNumPerAggregation_;
    ret = operation->SelectAlgoTypeForAllGather(delay, curSize, bandWidth, algType);
    iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType);
    if (iter == HCCL_ALGO_LEVEL1_NAME_MAP.end()) {
        HCCL_ERROR("level1: algType[%u] is invalid.", algType);
        ret = HCCL_E_INTERNAL;
    }
    HCCL_RUN_INFO(
        "hccl algorithm: there are %u server in level1, using %s algo.", operation->moduleNum_, iter->second.c_str());
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(algType, AlgTypeLevel1::ALG_LEVEL1_RING);

    operation->moduleNum_ = 4;
    operation->deviceNumPerAggregation_ = deviceNumPerAggregation;
    operation->userRankSize_ = operation->moduleNum_ * operation->deviceNumPerAggregation_;
    ret = operation->SelectAlgoTypeForReduceScatter(delay, curSize, bandWidth, algType);
    iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType);
    if (iter == HCCL_ALGO_LEVEL1_NAME_MAP.end()) {
        HCCL_ERROR("level1: algType[%u] is invalid.", algType);
        ret = HCCL_E_INTERNAL;
    }
    HCCL_RUN_INFO(
        "hccl algorithm: there are %u server in level1, using %s algo.", operation->moduleNum_, iter->second.c_str());
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(algType, AlgTypeLevel1::ALG_LEVEL1_NHR);
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hcclimpl_SelectAlgoTypeForAllGather)
{
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;

    std::unique_ptr<CollAlgOperator> operation(new (std::nothrow) CollAlgOperator(
        algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_ALLGATHER));
    float delay = 60;
    u64 curSize = 100; // 单位：字节(B)
    u32 deviceNumPerAggregation = 8;
    float bandWidth = 0.005;
    AlgTypeLevel1 algType;

    operation->moduleNum_ = 2;
    operation->deviceNumPerAggregation_ = deviceNumPerAggregation;
    operation->userRankSize_ = operation->moduleNum_ * operation->deviceNumPerAggregation_;
    ret = operation->SelectAlgoTypeForAllGather(delay, curSize, bandWidth, algType);
    auto iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType);
    if (iter == HCCL_ALGO_LEVEL1_NAME_MAP.end()) {
        HCCL_ERROR("level1: algType[%u] is invalid.", algType);
        ret = HCCL_E_INTERNAL;
    }
    HCCL_RUN_INFO(
        "hccl algorithm: there are %u server in level1, using %s algo.", operation->moduleNum_, iter->second.c_str());
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(algType, AlgTypeLevel1::ALG_LEVEL1_RING);

    operation->moduleNum_ = 3;
    operation->deviceNumPerAggregation_ = deviceNumPerAggregation;
    operation->userRankSize_ = operation->moduleNum_ * operation->deviceNumPerAggregation_;
    ret = operation->SelectAlgoTypeForAllGather(delay, curSize, bandWidth, algType);
    iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType);
    if (iter == HCCL_ALGO_LEVEL1_NAME_MAP.end()) {
        HCCL_ERROR("level1: algType[%u] is invalid.", algType);
        ret = HCCL_E_INTERNAL;
    }
    HCCL_RUN_INFO(
        "hccl algorithm: there are %u server in level1, using %s algo.", operation->moduleNum_, iter->second.c_str());
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(algType, AlgTypeLevel1::ALG_LEVEL1_RING);

    operation->moduleNum_ = 4;
    operation->deviceNumPerAggregation_ = deviceNumPerAggregation;
    operation->userRankSize_ = operation->moduleNum_ * operation->deviceNumPerAggregation_;
    ret = operation->SelectAlgoTypeForAllGather(delay, curSize, bandWidth, algType);
    iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType);
    if (iter == HCCL_ALGO_LEVEL1_NAME_MAP.end()) {
        HCCL_ERROR("level1: algType[%u] is invalid.", algType);
        ret = HCCL_E_INTERNAL;
    }
    HCCL_RUN_INFO(
        "hccl algorithm: there are %u server in level1, using %s algo.", operation->moduleNum_, iter->second.c_str());
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(algType, AlgTypeLevel1::ALG_LEVEL1_NHR);
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hcclimpl_SelectAlgoTypeForGather)
{
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;

    std::unique_ptr<CollAlgOperator> operation(new (std::nothrow) CollAlgOperator(
        algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_GATHER));
    float delay = 60;
    u64 curSize = 100; // 单位：字节(B)
    u32 deviceNumPerAggregation = 8;
    float bandWidth = 0.005;
    AlgTypeLevel1 algType;

    operation->moduleNum_ = 2;
    operation->deviceNumPerAggregation_ = deviceNumPerAggregation;
    operation->userRankSize_ = operation->moduleNum_ * operation->deviceNumPerAggregation_;
    ret = operation->SelectAlgoTypeForGather(delay, curSize, bandWidth, algType);
    auto iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType);
    if (iter == HCCL_ALGO_LEVEL1_NAME_MAP.end()) {
        HCCL_ERROR("level1: algType[%u] is invalid.", algType);
        ret = HCCL_E_INTERNAL;
    }
    HCCL_RUN_INFO(
        "hccl algorithm: there are %u server in level1, using %s algo.", operation->moduleNum_, iter->second.c_str());
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(algType, AlgTypeLevel1::ALG_LEVEL1_RING);

    operation->moduleNum_ = 3;
    operation->deviceNumPerAggregation_ = deviceNumPerAggregation;
    operation->userRankSize_ = operation->moduleNum_ * operation->deviceNumPerAggregation_;
    ret = operation->SelectAlgoTypeForGather(delay, curSize, bandWidth, algType);
    iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType);
    if (iter == HCCL_ALGO_LEVEL1_NAME_MAP.end()) {
        HCCL_ERROR("level1: algType[%u] is invalid.", algType);
        ret = HCCL_E_INTERNAL;
    }
    HCCL_RUN_INFO(
        "hccl algorithm: there are %u server in level1, using %s algo.", operation->moduleNum_, iter->second.c_str());
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(algType, AlgTypeLevel1::ALG_LEVEL1_RING);

    operation->moduleNum_ = 4;
    operation->deviceNumPerAggregation_ = deviceNumPerAggregation;
    operation->userRankSize_ = operation->moduleNum_ * operation->deviceNumPerAggregation_;
    ret = operation->SelectAlgoTypeForGather(delay, curSize, bandWidth, algType);
    iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType);
    if (iter == HCCL_ALGO_LEVEL1_NAME_MAP.end()) {
        HCCL_ERROR("level1: algType[%u] is invalid.", algType);
        ret = HCCL_E_INTERNAL;
    }
    HCCL_RUN_INFO(
        "hccl algorithm: there are %u server in level1, using %s algo.", operation->moduleNum_, iter->second.c_str());
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(algType, AlgTypeLevel1::ALG_LEVEL1_HD);
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hcclimpl_SelectAlgoTypeForAllReduce)
{
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;

    std::unique_ptr<CollAlgOperator> operation(new (std::nothrow) CollAlgOperator(
        algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_ALLREDUCE));
    float delay = 60;
    u64 curSize = 100; // 单位：字节(B)
    u32 deviceNumPerAggregation = 8;
    float bandWidth = 0.005;
    AlgTypeLevel1 algType;

    operation->moduleNum_ = 2;
    operation->deviceNumPerAggregation_ = deviceNumPerAggregation;
    operation->userRankSize_ = operation->moduleNum_ * operation->deviceNumPerAggregation_;
    ret = operation->SelectAlgoTypeForAllReduce(delay, curSize, bandWidth, algType);
    auto iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType);
    if (iter == HCCL_ALGO_LEVEL1_NAME_MAP.end()) {
        HCCL_ERROR("level1: algType[%u] is invalid.", algType);
        ret = HCCL_E_INTERNAL;
    }
    HCCL_RUN_INFO(
        "hccl algorithm: there are %u server in level1, using %s algo.", operation->moduleNum_, iter->second.c_str());
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(algType, AlgTypeLevel1::ALG_LEVEL1_RING);

    operation->moduleNum_ = 3;
    operation->deviceNumPerAggregation_ = deviceNumPerAggregation;
    operation->userRankSize_ = operation->moduleNum_ * operation->deviceNumPerAggregation_;
    ret = operation->SelectAlgoTypeForAllReduce(delay, curSize, bandWidth, algType);
    iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType);
    if (iter == HCCL_ALGO_LEVEL1_NAME_MAP.end()) {
        HCCL_ERROR("level1: algType[%u] is invalid.", algType);
        ret = HCCL_E_INTERNAL;
    }
    HCCL_RUN_INFO(
        "hccl algorithm: there are %u server in level1, using %s algo.", operation->moduleNum_, iter->second.c_str());
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(algType, AlgTypeLevel1::ALG_LEVEL1_RING);

    operation->moduleNum_ = 4;
    operation->deviceNumPerAggregation_ = deviceNumPerAggregation;
    operation->userRankSize_ = operation->moduleNum_ * operation->deviceNumPerAggregation_;
    ret = operation->SelectAlgoTypeForAllReduce(delay, curSize, bandWidth, algType);
    iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType);
    if (iter == HCCL_ALGO_LEVEL1_NAME_MAP.end()) {
        HCCL_ERROR("level1: algType[%u] is invalid.", algType);
        ret = HCCL_E_INTERNAL;
    }
    HCCL_RUN_INFO(
        "hccl algorithm: there are %u server in level1, using %s algo.", operation->moduleNum_, iter->second.c_str());
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(algType, AlgTypeLevel1::ALG_LEVEL1_NHR);
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hcclimpl_SelectAlgoTypeForBroadcast)
{
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;

    std::unique_ptr<CollAlgOperator> operation(new (std::nothrow) CollAlgOperator(
        algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_BROADCAST));
    float delay = 60;
    u64 curSize = 100; // 单位：字节(B)
    u32 deviceNumPerAggregation = 8;
    float bandWidth = 0.005;
    AlgTypeLevel1 algType;

    operation->moduleNum_ = 2;
    operation->deviceNumPerAggregation_ = deviceNumPerAggregation;
    operation->userRankSize_ = operation->moduleNum_ * operation->deviceNumPerAggregation_;
    ret = operation->SelectAlgoTypeForBroadcast(delay, curSize, bandWidth, algType);
    auto iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType);
    if (iter == HCCL_ALGO_LEVEL1_NAME_MAP.end()) {
        HCCL_ERROR("level1: algType[%u] is invalid.", algType);
        ret = HCCL_E_INTERNAL;
    }
    HCCL_RUN_INFO(
        "hccl algorithm: there are %u server in level1, using %s algo.", operation->moduleNum_, iter->second.c_str());
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(algType, AlgTypeLevel1::ALG_LEVEL1_RING);

    operation->moduleNum_ = 3;
    operation->deviceNumPerAggregation_ = deviceNumPerAggregation;
    operation->userRankSize_ = operation->moduleNum_ * operation->deviceNumPerAggregation_;
    ret = operation->SelectAlgoTypeForBroadcast(delay, curSize, bandWidth, algType);
    iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType);
    if (iter == HCCL_ALGO_LEVEL1_NAME_MAP.end()) {
        HCCL_ERROR("level1: algType[%u] is invalid.", algType);
        ret = HCCL_E_INTERNAL;
    }
    HCCL_RUN_INFO(
        "hccl algorithm: there are %u server in level1, using %s algo.", operation->moduleNum_, iter->second.c_str());
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(algType, AlgTypeLevel1::ALG_LEVEL1_RING);

    operation->moduleNum_ = 4;
    operation->deviceNumPerAggregation_ = deviceNumPerAggregation;
    operation->userRankSize_ = operation->moduleNum_ * operation->deviceNumPerAggregation_;
    ret = operation->SelectAlgoTypeForBroadcast(delay, curSize, bandWidth, algType);
    iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType);
    if (iter == HCCL_ALGO_LEVEL1_NAME_MAP.end()) {
        HCCL_ERROR("level1: algType[%u] is invalid.", algType);
        ret = HCCL_E_INTERNAL;
    }
    HCCL_RUN_INFO(
        "hccl algorithm: there are %u server in level1, using %s algo.", operation->moduleNum_, iter->second.c_str());
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(algType, AlgTypeLevel1::ALG_LEVEL1_HD);
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hcclimpl_SelectAlgoTypeForReduce)
{
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;

    std::unique_ptr<CollAlgOperator> operation(new (std::nothrow) CollAlgOperator(
        algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_REDUCE));
    float delay = 60;
    u64 curSize = 100; // 单位：字节(B)
    u32 deviceNumPerAggregation = 8;
    float bandWidth = 0.005;
    AlgTypeLevel1 algType;

    operation->moduleNum_ = 2;
    operation->deviceNumPerAggregation_ = deviceNumPerAggregation;
    operation->userRankSize_ = operation->moduleNum_ * operation->deviceNumPerAggregation_;
    ret = operation->SelectAlgoTypeForReduce(delay, curSize, bandWidth, algType);
    auto iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType);
    if (iter == HCCL_ALGO_LEVEL1_NAME_MAP.end()) {
        HCCL_ERROR("level1: algType[%u] is invalid.", algType);
        ret = HCCL_E_INTERNAL;
    }
    HCCL_RUN_INFO(
        "hccl algorithm: there are %u server in level1, using %s algo.", operation->moduleNum_, iter->second.c_str());
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(algType, AlgTypeLevel1::ALG_LEVEL1_RING);

    operation->moduleNum_ = 3;
    operation->deviceNumPerAggregation_ = deviceNumPerAggregation;
    operation->userRankSize_ = operation->moduleNum_ * operation->deviceNumPerAggregation_;
    ret = operation->SelectAlgoTypeForReduce(delay, curSize, bandWidth, algType);
    iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType);
    if (iter == HCCL_ALGO_LEVEL1_NAME_MAP.end()) {
        HCCL_ERROR("level1: algType[%u] is invalid.", algType);
        ret = HCCL_E_INTERNAL;
    }
    HCCL_RUN_INFO(
        "hccl algorithm: there are %u server in level1, using %s algo.", operation->moduleNum_, iter->second.c_str());
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(algType, AlgTypeLevel1::ALG_LEVEL1_RING);

    operation->moduleNum_ = 4;
    operation->deviceNumPerAggregation_ = deviceNumPerAggregation;
    operation->userRankSize_ = operation->moduleNum_ * operation->deviceNumPerAggregation_;
    ret = operation->SelectAlgoTypeForReduce(delay, curSize, bandWidth, algType);
    iter = HCCL_ALGO_LEVEL1_NAME_MAP.find(algType);
    if (iter == HCCL_ALGO_LEVEL1_NAME_MAP.end()) {
        HCCL_ERROR("level1: algType[%u] is invalid.", algType);
        ret = HCCL_E_INTERNAL;
    }
    HCCL_RUN_INFO(
        "hccl algorithm: there are %u server in level1, using %s algo.", operation->moduleNum_, iter->second.c_str());
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(algType, AlgTypeLevel1::ALG_LEVEL1_HD);
    GlobalMockObject::verify();
}

RankTable_t get_rank_table_rank_2server_3p_2_1()
{
    RankTable_t rankTable;
    rankTable.deviceNum = 3;

    rankTable.serverNum = 2;
    rankTable.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    rankTable.nicNum = 1;
    rankTable.nicNames.push_back("eth0");
    rankTable.rankNum = 3;

    int server1rankNum = 2;
    int server2rankNum = 1;

    for (int i = 0; i < server1rankNum; ++i) {
        RankInfo_t rank;
        rank.rankId = i;
        rank.serverIdx = 0;
        rank.serverId = "192.168.1.1";
        rank.deviceInfo.devicePhyId = i;
        rank.deviceInfo.deviceIp.push_back(HcclIpAddress("172.17.10.1"));
        rankTable.rankList.push_back(rank);
    }

    for (int i = 0; i < server2rankNum; ++i) {
        RankInfo_t rank;
        rank.rankId = i + server1rankNum;
        rank.serverIdx = 1;
        rank.serverId = "192.168.1.2";
        rank.deviceInfo.devicePhyId = i;
        rank.deviceInfo.deviceIp.push_back(HcclIpAddress("172.17.10.1"));
        rankTable.rankList.push_back(rank);
    }
    return rankTable;
}

TEST_F(HcclImplTest, ut_multiModuleDiffDeviceNumMode_SetAlgType)
{
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    std::map<HcclCMDType, AlgType> algType;
    algConfigurator->topoAttr_.multiModuleDiffDeviceNumMode = true;
    ret = algConfigurator->SelectAlgType(0, DevType::DEV_TYPE_COUNT, algType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel0, AlgTypeLevel0::ALG_LEVEL0_WHOLE_RING);
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel1, AlgTypeLevel1::ALG_LEVEL1_WHOLE_RING);
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hcclImpl_get_double_ring_topo_type_1)
{
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    TopoType topoType;
    AlgType algType;
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING;
    algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    DevType chipType = DevType::DEV_TYPE_910_93;
    ret = algConfigurator->GetTopoTypeByAlgType(algType, chipType, topoType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(topoType, TopoType::TOPO_TYPE_NP_DOUBLE_RING);
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hcclImpl_get_double_ring_topo_type_2)
{
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    TopoType topoType;
    AlgType algType;
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING;
    algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_HD;
    DevType chipType = DevType::DEV_TYPE_910_93;
    ret = algConfigurator->GetTopoTypeByAlgType(algType, chipType, topoType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(topoType, TopoType::TOPO_TYPE_NP_DOUBLE_RING);
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hcclImpl_is_double_ring_topo_pattern_1)
{
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult tmpRet = implBase->Init(params, rankTable);
    EXPECT_EQ(tmpRet, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    algConfigurator->topoAttr_.deviceNumPerAggregation = 8;
    algConfigurator->topoAttr_.pairLinkCounter.clear();
    algConfigurator->topoAttr_.pairLinkCounter[static_cast<u32>(LinkTypeInServer::HCCS_SW_TYPE)] = 48;
    algConfigurator->topoAttr_.pairLinkCounter[static_cast<u32>(LinkTypeInServer::SIO_TYPE)] = 8;

    bool ret = algConfigurator->IsHCCSSWNumEqualToTwiceSIONum();
    EXPECT_EQ(ret, true);
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hcclImpl_is_double_ring_topo_pattern_2)
{
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult tmpRet = implBase->Init(params, rankTable);
    EXPECT_EQ(tmpRet, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    algConfigurator->topoAttr_.deviceNumPerAggregation = 8;
    algConfigurator->topoAttr_.pairLinkCounter.clear();
    algConfigurator->topoAttr_.pairLinkCounter[static_cast<u32>(LinkTypeInServer::HCCS_SW_TYPE)] = 0;
    algConfigurator->topoAttr_.pairLinkCounter[static_cast<u32>(LinkTypeInServer::SIO_TYPE)] = 8;

    bool ret = algConfigurator->IsHCCSSWNumEqualToTwiceSIONum();
    EXPECT_EQ(ret, false);
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hcclImpl_is_double_ring_topo_pattern_3)
{
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclResult tmpRet = implBase->Init(params, rankTable);
    EXPECT_EQ(tmpRet, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    algConfigurator->topoAttr_.deviceNumPerAggregation = 8;
    algConfigurator->topoAttr_.pairLinkCounter.clear();
    algConfigurator->topoAttr_.pairLinkCounter[static_cast<u32>(LinkTypeInServer::HCCS_SW_TYPE)] = 48;
    algConfigurator->topoAttr_.pairLinkCounter[static_cast<u32>(LinkTypeInServer::SIO_TYPE)] = 0;

    bool ret = algConfigurator->IsHCCSSWNumEqualToTwiceSIONum();
    EXPECT_EQ(ret, false);
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hcclImpl_run_double_ring_executer_by_all_gather)
{
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclCommParams localParams;
    RankTable_t localRankTable;
    TestConstructParam(localParams, localRankTable);
    localParams.deviceType = DevType::DEV_TYPE_910_93;

    HcclResult ret = implBase->Init(localParams, localRankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;

    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    std::unique_ptr<AllGatherOperator> operation(
        new (std::nothrow) AllGatherOperator(algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher));
    operation->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    std::string tag = "test";
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);
    HcomCollOpInfo opInfo;
    opInfo.inputAddr = inputMem.ptr();
    opInfo.outputAddr = outputMem.ptr();
    opInfo.count = count;
    opInfo.dataType = dataType;
    opInfo.reduceOp = op;

    OpParam opParam;
    opParam.tag = tag;
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = count * 4 * 2;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = count * 4;
    opParam.DataDes.count = count;
    opParam.DataDes.dataType = dataType;
    opParam.reduceType = op;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);

    std::string algName = "";
    std::string newTag = "";
    ret = operation->SelectAlg(tag, opParam, algName, newTag);
    EXPECT_TRUE(algName == "AlignedAllGatherDoubleRingFor91093Executor");
    AlgResourceRequest resourceRequest;
    ret = operation->CalcResRequest(algName, opParam, resourceRequest);

    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hcclImpl_run_double_ring_executer_by_all_reduce)
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
    TestConstructParam_1server_4p(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910_93;

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&AlgConfigurator::IsHCCSSWNumEqualToTwiceSIONum).stubs().will(returnValue(true));

    ret = implBase->AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->InitCCLbuffer(4096, 4096);
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    impl->topoAttr_.deviceLogicId = 0;
    impl->topoAttr_.devicePhyId = 0;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    impl->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    algConfigurator->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;

    const std::vector<std::vector<u32>> tmpRingNics = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}};

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollCommExecutor::MultiRingReduceScatter).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollCommExecutor::MultiRingAllGather).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));

    std::printf("[st_CollAllReduceRingFor91093Executor_Ring]");

    ret = implBase->AllReduce(tag, inputMem.ptr(), outputMem.ptr(), count, dataType, op, stream.ptr());
    implBase = nullptr;

    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hcclImpl_run_double_ring_executer_for_91093_by_broadcast)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    DeviceMem inputMem = DeviceMem::alloc(4096);
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam_2Server(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910_93;

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&AlgConfigurator::IsHCCSSWNumEqualToTwiceSIONum).stubs().will(returnValue(true));

    ret = implBase->AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->InitCCLbuffer(4096, 4096);
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910_93;
    topoMatcher->topoInfo_.devicePhyId = 0;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    topoMatcher->topoInfo_.superPodNum = 2;

    impl->topoAttr_.deviceLogicId = 0;
    impl->topoAttr_.devicePhyId = 0;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    impl->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    algConfigurator->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;

    const std::vector<std::vector<u32>> tmpRingNics
        = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}};

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollCommExecutor::MultiRingScatter).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollCommExecutor::MultiRingAllGather).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));

    std::printf("[st_CollBroadcastFor91093Level2Executor_Ring]");

    ret = implBase->Broadcast(tag, inputMem.ptr(), count, dataType, 0, stream.ptr());
    implBase = nullptr;

    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hcclImpl_run_double_ring_executer_for_91093_by_all_reduce)
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
    TestConstructParam_2Server(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910_93;

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&AlgConfigurator::IsHCCSSWNumEqualToTwiceSIONum).stubs().will(returnValue(true));

    ret = implBase->AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->InitCCLbuffer(4096, 4096);
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910_93;
    topoMatcher->topoInfo_.devicePhyId = 0;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    topoMatcher->topoInfo_.superPodNum = 2;

    impl->topoAttr_.deviceLogicId = 0;
    impl->topoAttr_.devicePhyId = 0;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    impl->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    algConfigurator->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;

    const std::vector<std::vector<u32>> tmpRingNics
        = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}};

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollCommExecutor::MultiRingReduceScatter).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollCommExecutor::MultiRingAllGather).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));

    std::printf("[st_CollAllReduceRingFor91093Level2Executor_Ring]");

    ret = implBase->AllReduce(tag, inputMem.ptr(), outputMem.ptr(), count, dataType, op, stream.ptr());
    implBase = nullptr;

    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hcclImpl_run_fast_double_ring_for_910_93_executer_by_all_reduce)
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
    setenv("HCCL_ALGO", "level0:ring", 1);
    ret = InitEnvVarParam();
    TestConstructParam_1server_4p(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910_93;

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&AlgConfigurator::IsHCCSSWNumEqualToTwiceSIONum).stubs().will(returnValue(true));

    ret = implBase->AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->InitCCLbuffer(4096, 4096);
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    impl->topoAttr_.deviceLogicId = 0;
    impl->topoAttr_.devicePhyId = 0;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    impl->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    algConfigurator->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;

    const std::vector<std::vector<u32>> tmpRingNics = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}};

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollCommExecutor::MultiRingReduceScatter).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollCommExecutor::MultiRingAllGather).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));

    std::printf("[st_CollAllReduceFastDoubleRingFor91093Executor_Ring]");
    ret = implBase->AllReduce(tag, inputMem.ptr(), outputMem.ptr(), count, dataType, op, stream.ptr());
    implBase = nullptr;

    GlobalMockObject::verify();
}

void get_rank_table_rank_1server_2module_7p_4_3(HcclCommParams& params, RankTable_t& rankTable)
{
    string commId = "comm ";
    memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1);
    params.rank = 0;
    params.userRank = 0;
    params.totalRanks = 7;
    params.isHeterogComm = false;
    params.logicDevId = 0;
    params.commWorkMode = WorkMode::HCCL_MODE_NORMAL;
    params.deviceType = DevType::DEV_TYPE_910B;

    rankTable.deviceNum = 7;
    rankTable.serverNum = 1;
    rankTable.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    rankTable.nicNum = 1;
    rankTable.nicNames.push_back("eth0");
    rankTable.rankNum = 7;

    int server1rankNum = 4;
    int server2rankNum = 3;

    for (int i = 0; i < server1rankNum; ++i) {
        RankInfo_t rank;
        rank.rankId = i;
        rank.serverIdx = 0;
        rank.serverId = "192.168.1.1";
        rank.deviceInfo.devicePhyId = i;
        rank.deviceInfo.deviceIp.push_back(HcclIpAddress("172.17.10.1"));
        rankTable.rankList.push_back(rank);
    }

    for (int i = 0; i < server2rankNum; ++i) {
        RankInfo_t rank;
        rank.rankId = i + server1rankNum;
        rank.serverIdx = 0;
        rank.serverId = "192.168.1.1";
        rank.deviceInfo.devicePhyId = i + 8;
        rank.deviceInfo.deviceIp.push_back(HcclIpAddress("172.17.10.1"));
        rankTable.rankList.push_back(rank);
    }
    return;
}

TEST_F(HcclImplTest, ut_multiModuleDiffDeviceNumMode_1s7p_disableRDMA)
{
    // module0:0,1,2,3  module1:8,9,10  without rdma
    HcclCommParams params;
    RankTable_t rankTable;
    get_rank_table_rank_1server_2module_7p_4_3(params, rankTable);
    std::unique_ptr<HcclCommunicator> impl(new (std::nothrow) HcclCommunicator());

    setenv("HCCL_INTRA_ROCE_ENABLE", "0", 1);
    setenv("HCCL_INTRA_PCIE_ENABLE", "1", 1);
    s32 ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = impl->Init(params, rankTable);
    EXPECT_EQ(impl->isDiffDeviceModule_, true);
    EXPECT_EQ(impl->moduleNum_, 2);
    EXPECT_EQ(impl->multiModuleDiffDeviceNumMode_, true);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);

    unsetenv("HCCL_INTRA_ROCE_ENABLE");
    unsetenv("HCCL_INTRA_PCIE_ENABLE");
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_multiModuleDiffDeviceNumMode_1s7p_enableRDMA)
{
    // module0:0,1,2,3  module1:8,9,10   with rdma
    HcclCommParams params;
    RankTable_t rankTable;
    get_rank_table_rank_1server_2module_7p_4_3(params, rankTable);
    std::unique_ptr<HcclCommunicator> impl(new (std::nothrow) HcclCommunicator());

    setenv("HCCL_INTRA_ROCE_ENABLE", "1", 1);
    setenv("HCCL_INTRA_PCIE_ENABLE", "0", 1);
    s32 ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = impl->Init(params, rankTable);

    EXPECT_EQ(impl->isDiffDeviceModule_, true);
    EXPECT_EQ(impl->moduleNum_, 2);
    EXPECT_EQ(impl->multiModuleDiffDeviceNumMode_, true);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    unsetenv("HCCL_INTRA_ROCE_ENABLE");
    unsetenv("HCCL_INTRA_PCIE_ENABLE");
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_multiModuleDiffDeviceNumMode_1s8p_disableRDMA)
{
    // module0:0,1,2,3  module1:8,9,10,12 without rdma
    HcclCommParams params;
    RankTable_t rankTable;
    get_rank_table_rank_1server_2module_7p_4_3(params, rankTable);

    RankInfo_t rank;
    rank.rankId = 7;
    rank.serverIdx = 0;
    rank.serverId = "192.168.1.1";
    rank.deviceInfo.devicePhyId = 12;
    rank.deviceInfo.deviceIp.push_back(HcclIpAddress("172.17.10.1"));
    rankTable.rankList.push_back(rank);
    rankTable.deviceNum = 8;
    rankTable.rankNum = 8;
    params.totalRanks = 8;

    std::unique_ptr<HcclCommunicator> impl(new (std::nothrow) HcclCommunicator());
    setenv("HCCL_INTRA_ROCE_ENABLE", "0", 1);
    setenv("HCCL_INTRA_PCIE_ENABLE", "1", 1);
    s32 ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = impl->Init(params, rankTable);
    EXPECT_EQ(impl->isDiffDeviceModule_, true);
    EXPECT_EQ(impl->moduleNum_, 2);
    EXPECT_EQ(impl->multiModuleDiffDeviceNumMode_, false);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);

    unsetenv("HCCL_INTRA_ROCE_ENABLE");
    unsetenv("HCCL_INTRA_PCIE_ENABLE");
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_multiModuleDiffDeviceNumMode_1s8p_enableRDMA)
{
    // module0:0,1,2,3  module1:8,9,10,12 without rdma
    HcclCommParams params;
    RankTable_t rankTable;
    get_rank_table_rank_1server_2module_7p_4_3(params, rankTable);

    RankInfo_t rank;
    rank.rankId = 7;
    rank.serverIdx = 0;
    rank.serverId = "192.168.1.1";
    rank.deviceInfo.devicePhyId = 12;
    rank.deviceInfo.deviceIp.push_back(HcclIpAddress("172.17.10.1"));
    rankTable.rankList.push_back(rank);
    rankTable.deviceNum = 8;
    rankTable.rankNum = 8;
    params.totalRanks = 8;

    std::unique_ptr<HcclCommunicator> impl(new (std::nothrow) HcclCommunicator());
    setenv("HCCL_INTRA_ROCE_ENABLE", "1", 1);
    setenv("HCCL_INTRA_PCIE_ENABLE", "0", 1);
    s32 ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = impl->Init(params, rankTable);

    EXPECT_EQ(impl->isDiffDeviceModule_, true);
    EXPECT_EQ(impl->moduleNum_, 2);
    EXPECT_EQ(impl->multiModuleDiffDeviceNumMode_, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    unsetenv("HCCL_INTRA_ROCE_ENABLE");
    unsetenv("HCCL_INTRA_PCIE_ENABLE");
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_HcclCommunicator_AicpuUnfold)
{
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclRtStream stream = NULL;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    HcclCMDType cmdType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    s32 device_id = 0;
    // 申请device memory
    DeviceMem mem_dev_input = DeviceMem::alloc(1024);
    EXPECT_NE(mem_dev_input.ptr(), nullptr);
    DeviceMem mem_dev_output = DeviceMem::alloc(1024);
    EXPECT_NE(mem_dev_output.ptr(), nullptr);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::InitPara).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::InitOneSidedService).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER(hrtStreamGetMode).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
    ret = implBase->AicpuUnfold(
        "tag_test", const_cast<void*>(mem_dev_input.ptr()), const_cast<void*>(mem_dev_output.ptr()), count, dataType,
        HCCL_REDUCE_SUM, stream, cmdType);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);

    unsetenv("HCCL_OP_EXPANSION_MODE");
    // 销毁资源

    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_HcclCommunicator_CheckReduceDataType)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclDataType dataType = HCCL_DATA_TYPE_INT16;
    HcclReduceOp op = HCCL_REDUCE_PROD;

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::InitPara).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::InitOneSidedService).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    DevType deviceType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));
    ret = implBase->CheckReduceDataType(dataType, op);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);

    ret = HcomCheckReduceDataType(dataType, op, deviceType);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);

    deviceType = DevType::DEV_TYPE_910;
    ret = HcomCheckReduceDataType(dataType, op, deviceType);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);

    deviceType = DevType::DEV_TYPE_310P3;
    ret = HcomCheckReduceDataType(dataType, op, deviceType);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hrtRaGetSocketVnicIpInfos)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 phy_id = 2;
    vector<u32> ids;
    std::vector<HcclIpAddress> vnicIp;
    vnicIp.push_back(HcclIpAddress("1.0.0.0"));
    vnicIp.push_back(HcclIpAddress("2.0.0.0"));
    DeviceIdType deviceIdType = DeviceIdType::DEVICE_ID_TYPE_PHY_ID;
    id_type idType = static_cast<id_type>(deviceIdType);
    ret = hrtRaGetSocketVnicIpInfos(phy_id, idType, ids, vnicIp);
    EXPECT_EQ(ret, HCCL_E_PARA);
    ids.push_back(1);
    ids.push_back(2);
    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = hrtRaGetSocketVnicIpInfos(phy_id, idType, ids, vnicIp);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclImplTest, ut_H2DTlvInit)
{
    HcclResult ret = HCCL_SUCCESS;
    struct tlv_init_info init_info;
    uint32_t tlv_handle_id = 0;
    uint32_t buffer_size = 0;
    void* tlv_handle = nullptr;

    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = H2DTlvInit(&init_info, tlv_handle_id, &buffer_size, &tlv_handle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    DlRaFunction::GetInstance().dlH2DTlvInit = nullptr;
    MOCKER(hrtRaGetInterfaceVersion).expects(atMost(1)).will(returnValue(HCCL_E_NOT_SUPPORT));
    ret = H2DTlvInit(&init_info, tlv_handle_id, &buffer_size, &tlv_handle);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(HcclImplTest, ut_H2DTlvRequest)
{
    HcclResult ret = HCCL_SUCCESS;
    struct tlv_msg send_msg;
    struct tlv_msg recv_msg;
    void* tlv_handle = nullptr;
    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = H2DTlvRequest(tlv_handle, &send_msg, &send_msg);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    DlRaFunction::GetInstance().dlH2DTlvRequest = nullptr;
    MOCKER(hrtRaGetInterfaceVersion).expects(atMost(1)).will(returnValue(HCCL_E_NOT_SUPPORT));
    ret = H2DTlvRequest(tlv_handle, &send_msg, &send_msg);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(HcclImplTest, ut_H2DTlvDeinit)
{
    HcclResult ret = HCCL_SUCCESS;
    void* tlv_handle = nullptr;

    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    MOCKER(hrtRaGetInterfaceVersion).expects(atMost(10)).will(returnValue(HCCL_E_NOT_SUPPORT));
    ret = H2DTlvDeinit(&tlv_handle);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(HcclImplTest, ut_hccl_Communicator_SetRetryEnable_Inter_Super_Pod_true)
{
    HcclResult ret;
    setenv("HCCL_OP_RETRY_ENABLE", "L0:0, L1:0, L2:1", 1);
    ret = ParseRetryEnable();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HcclCommunicator impl;
    std::vector<RankInfo> ranks;

    RankInfo tmp_para_0;

    tmp_para_0.userRank = 0;
    tmp_para_0.devicePhyId = 0;
    tmp_para_0.deviceType = DevType::DEV_TYPE_910;
    tmp_para_0.serverIdx = 0;
    tmp_para_0.serverId = "10.0.0.10";
    tmp_para_0.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    tmp_para_0.superPodId = "0";

    RankInfo tmp_para_1;

    tmp_para_1.userRank = 1;
    tmp_para_1.devicePhyId = 1;
    tmp_para_1.deviceType = DevType::DEV_TYPE_910;
    tmp_para_1.serverId = "10.0.1.10";
    tmp_para_1.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    tmp_para_1.superPodId = "1";

    ranks.push_back(tmp_para_0);
    ranks.push_back(tmp_para_1);

    impl.serverId_ = "10.0.0.10";
    impl.attrCollector_.serverId_ = "10.0.0.10";
    std::vector<RankInfo_t> rankListNew;
    ret = impl.attrCollector_.TransformRankList(ranks, rankListNew);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = impl.attrCollector_.SetServerNum(rankListNew);
    impl.serverNum_ = impl.attrCollector_.GetServerNum();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = GetSuperPodNum(rankListNew, impl.superPodNum_);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    impl.deviceNumPerAggregation_ = 8;
    HcclIpAddress serverIp("172.17.10.1");
    HcclIpAddress localIp("172.17.10.1");
    SetRetryEnable(
        DevType::DEV_TYPE_910_93, impl.superPodNum_, impl.serverNum_, impl.deviceNumPerAggregation_,
        impl.isDiffDeviceType_, impl.GetAivModeConfig(), serverIp, localIp, impl.retryEnable_);
    std::cout << "superPodNum_ is " << unsigned(impl.superPodNum_) << std::endl;
    std::cout << "serverNum_ is " << unsigned(impl.serverNum_) << std::endl;
    std::cout << "deviceNumPerAggregation_ is " << unsigned(impl.deviceNumPerAggregation_) << std::endl;
    if (impl.retryEnable_) {
        std::cout << "retryEnable_ is true" << std::endl;
    } else {
        std::cout << "retryEnable_ is false" << std::endl;
    }
    EXPECT_EQ(impl.retryEnable_, true);
    unsetenv("HCCL_OP_RETRY_ENABLE");
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hccl_Communicator_SetRetryEnable_Inter_2Sever_true)
{
    HcclResult ret;
    setenv("HCCL_OP_RETRY_ENABLE", "L0:0, L1:1, L2:0", 1);
    ret = ParseRetryEnable();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HcclCommunicator impl;
    std::vector<RankInfo> ranks;

    RankInfo tmp_para_0;

    tmp_para_0.userRank = 0;
    tmp_para_0.devicePhyId = 0;
    tmp_para_0.deviceType = DevType::DEV_TYPE_910;
    tmp_para_0.serverIdx = 0;
    tmp_para_0.serverId = "10.0.0.10";
    tmp_para_0.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    tmp_para_0.superPodId = "0";

    RankInfo tmp_para_1;

    tmp_para_1.userRank = 1;
    tmp_para_1.devicePhyId = 1;
    tmp_para_1.deviceType = DevType::DEV_TYPE_910;
    tmp_para_1.serverId = "10.0.1.10";
    tmp_para_1.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    tmp_para_1.superPodId = "0";

    ranks.push_back(tmp_para_0);
    ranks.push_back(tmp_para_1);

    impl.serverId_ = "10.0.0.10";
    impl.attrCollector_.serverId_ = "10.0.0.10";
    std::vector<RankInfo_t> rankListNew;
    ret = impl.attrCollector_.TransformRankList(ranks, rankListNew);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = impl.attrCollector_.SetServerNum(rankListNew);
    impl.serverNum_ = impl.attrCollector_.GetServerNum();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = GetSuperPodNum(rankListNew, impl.superPodNum_);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    impl.deviceNumPerAggregation_ = 8;
    HcclIpAddress serverIp("172.17.10.1");
    HcclIpAddress localIp("172.17.10.1");
    SetRetryEnable(
        DevType::DEV_TYPE_910_93, impl.superPodNum_, impl.serverNum_, impl.deviceNumPerAggregation_,
        impl.isDiffDeviceType_, impl.GetAivModeConfig(), serverIp, localIp, impl.retryEnable_);
    std::cout << "superPodNum_ is " << unsigned(impl.superPodNum_) << std::endl;
    std::cout << "serverNum_ is " << unsigned(impl.serverNum_) << std::endl;
    std::cout << "deviceNumPerAggregation_ is " << unsigned(impl.deviceNumPerAggregation_) << std::endl;
    if (impl.retryEnable_) {
        std::cout << "retryEnable_ is true" << std::endl;
    } else {
        std::cout << "retryEnable_ is false" << std::endl;
    }
    EXPECT_EQ(impl.retryEnable_, true);
    unsetenv("HCCL_OP_RETRY_ENABLE");
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hccl_Communicator_SetRetryEnable_Inter_1Sever_true)
{
    HcclResult ret;
    setenv("HCCL_OP_RETRY_ENABLE", "L0:0, L1:1, L2:0", 1);
    ret = ParseRetryEnable();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HcclCommunicator impl;
    std::vector<RankInfo> ranks;

    RankInfo tmp_para_0;

    tmp_para_0.userRank = 0;
    tmp_para_0.devicePhyId = 0;
    tmp_para_0.deviceType = DevType::DEV_TYPE_910;
    tmp_para_0.serverIdx = 0;
    tmp_para_0.serverId = "10.0.0.10";
    tmp_para_0.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    tmp_para_0.superPodId = "0";

    ranks.push_back(tmp_para_0);

    impl.serverId_ = "10.0.0.10";
    impl.attrCollector_.serverId_ = "10.0.0.10";
    std::vector<RankInfo_t> rankListNew;
    ret = impl.attrCollector_.TransformRankList(ranks, rankListNew);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = impl.attrCollector_.SetServerNum(rankListNew);
    impl.serverNum_ = impl.attrCollector_.GetServerNum();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = GetSuperPodNum(rankListNew, impl.superPodNum_);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    impl.deviceNumPerAggregation_ = 8;
    HcclIpAddress serverIp("172.17.10.1");
    HcclIpAddress localIp("172.17.10.1");
    SetRetryEnable(
        DevType::DEV_TYPE_910_93, impl.superPodNum_, impl.serverNum_, impl.deviceNumPerAggregation_,
        impl.isDiffDeviceType_, impl.GetAivModeConfig(), serverIp, localIp, impl.retryEnable_);
    std::cout << "superPodNum_ is " << unsigned(impl.superPodNum_) << std::endl;
    std::cout << "serverNum_ is " << unsigned(impl.serverNum_) << std::endl;
    std::cout << "deviceNumPerAggregation_ is " << unsigned(impl.deviceNumPerAggregation_) << std::endl;
    if (impl.retryEnable_) {
        std::cout << "retryEnable_ is true" << std::endl;
    } else {
        std::cout << "retryEnable_ is false" << std::endl;
    }
    EXPECT_EQ(impl.retryEnable_, false);
    unsetenv("HCCL_OP_RETRY_ENABLE");
    GlobalMockObject::verify();
}
#if 0 // 执行失败
TEST_F(HcclImplTest, ut_hccl_Communicator_SetRetryEnable_Intra_Sever_true)
{
    HcclResult ret;
    setenv("HCCL_OP_RETRY_ENABLE", "L0:1, L1:0, L2:0", 1);
    ret = ParseRetryEnable();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HcclCommunicator impl;
    std::vector<RankInfo> ranks;

    RankInfo tmp_para_0;

    tmp_para_0.userRank = 0;
    tmp_para_0.devicePhyId = 0;
    tmp_para_0.deviceType = DevType::DEV_TYPE_910;
    tmp_para_0.serverIdx = 0;
    tmp_para_0.serverId = "10.0.0.10";
    tmp_para_0.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    tmp_para_0.superPodId = "0";

    ranks.push_back(tmp_para_0);

    impl.serverId_ = "10.0.0.10";
    impl.attrCollector_.serverId_ = "10.0.0.10";
    std::vector<RankInfo_t> rankListNew;
    ret = impl.attrCollector_.TransformRankList(ranks, rankListNew);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = impl.attrCollector_.SetServerNum(rankListNew);
    impl.serverNum_ = impl.attrCollector_.GetServerNum();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = GetSuperPodNum(rankListNew, impl.superPodNum_);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    impl.deviceNumPerAggregation_ = 8;
    HcclIpAddress serverIp("172.17.10.1");
    HcclIpAddress localIp("172.17.10.1");
    SetRetryEnable(DevType::DEV_TYPE_910_93, impl.superPodNum_, impl.serverNum_, impl.deviceNumPerAggregation_,
        impl.isDiffDeviceType_, impl.GetAivModeConfig(), serverIp, localIp, impl.retryEnable_);
    std::cout << "superPodNum_ is " << unsigned(impl.superPodNum_) << std::endl;
    std::cout << "serverNum_ is " << unsigned(impl.serverNum_) << std::endl;
    std::cout << "deviceNumPerAggregation_ is " << unsigned(impl.deviceNumPerAggregation_) << std::endl;
    if (impl.retryEnable_) {
        std::cout << "retryEnable_ is true" << std::endl;
    } else {
        std::cout << "retryEnable_ is false" << std::endl;
    }
    EXPECT_EQ(impl.retryEnable_, false);
    unsetenv("HCCL_OP_RETRY_ENABLE");
    GlobalMockObject::verify();
}
#endif
TEST_F(HcclImplTest, ut_hccl_Communicator_SetRetryEnable_All_False)
{
    HcclResult ret;
    setenv("HCCL_OP_RETRY_ENABLE", "L0:0, L1:0, L2:0", 1);
    ret = ParseRetryEnable();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HcclCommunicator impl;
    std::vector<RankInfo> ranks;

    RankInfo tmp_para_0;

    tmp_para_0.userRank = 0;
    tmp_para_0.devicePhyId = 0;
    tmp_para_0.deviceType = DevType::DEV_TYPE_910;
    tmp_para_0.serverIdx = 0;
    tmp_para_0.serverId = "10.0.0.10";
    tmp_para_0.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    tmp_para_0.superPodId = "0";

    RankInfo tmp_para_1;

    tmp_para_1.userRank = 1;
    tmp_para_1.devicePhyId = 1;
    tmp_para_1.deviceType = DevType::DEV_TYPE_910;
    tmp_para_1.serverId = "10.0.1.10";
    tmp_para_1.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    tmp_para_1.superPodId = "1";

    ranks.push_back(tmp_para_0);
    ranks.push_back(tmp_para_1);

    impl.serverId_ = "10.0.0.10";
    impl.attrCollector_.serverId_ = "10.0.0.10";
    std::vector<RankInfo_t> rankListNew;
    ret = impl.attrCollector_.TransformRankList(ranks, rankListNew);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = impl.attrCollector_.SetServerNum(rankListNew);
    impl.serverNum_ = impl.attrCollector_.GetServerNum();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = GetSuperPodNum(rankListNew, impl.superPodNum_);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    impl.deviceNumPerAggregation_ = 8;
    HcclIpAddress serverIp("172.17.10.1");
    HcclIpAddress localIp("172.17.10.1");
    SetRetryEnable(
        DevType::DEV_TYPE_910_93, impl.superPodNum_, impl.serverNum_, impl.deviceNumPerAggregation_,
        impl.isDiffDeviceType_, impl.GetAivModeConfig(), serverIp, localIp, impl.retryEnable_);
    std::cout << "superPodNum_ is " << unsigned(impl.superPodNum_) << std::endl;
    std::cout << "serverNum_ is " << unsigned(impl.serverNum_) << std::endl;
    std::cout << "deviceNumPerAggregation_ is " << unsigned(impl.deviceNumPerAggregation_) << std::endl;
    if (impl.retryEnable_) {
        std::cout << "retryEnable_ is true" << std::endl;
    } else {
        std::cout << "retryEnable_ is false" << std::endl;
    }
    EXPECT_EQ(impl.retryEnable_, false);
    unsetenv("HCCL_OP_RETRY_ENABLE");
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hccl_Communicator_SetRetryEnable_deviceNumPerAggregation_1)
{
    HcclResult ret;
    setenv("HCCL_OP_RETRY_ENABLE", "L0:1, L1:0, L2:0", 1);
    ret = ParseRetryEnable();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HcclCommunicator impl;
    std::vector<RankInfo> ranks;

    RankInfo tmp_para_0;

    tmp_para_0.userRank = 0;
    tmp_para_0.devicePhyId = 0;
    tmp_para_0.deviceType = DevType::DEV_TYPE_910;
    tmp_para_0.serverIdx = 0;
    tmp_para_0.serverId = "10.0.0.10";
    tmp_para_0.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    tmp_para_0.superPodId = "0";

    RankInfo tmp_para_1;

    tmp_para_1.userRank = 1;
    tmp_para_1.devicePhyId = 1;
    tmp_para_1.deviceType = DevType::DEV_TYPE_910;
    tmp_para_1.serverId = "10.0.1.10";
    tmp_para_1.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    tmp_para_1.superPodId = "1";

    ranks.push_back(tmp_para_0);
    ranks.push_back(tmp_para_1);

    impl.serverId_ = "10.0.0.10";
    impl.attrCollector_.serverId_ = "10.0.0.10";
    std::vector<RankInfo_t> rankListNew;
    ret = impl.attrCollector_.TransformRankList(ranks, rankListNew);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = impl.attrCollector_.SetServerNum(rankListNew);
    impl.serverNum_ = impl.attrCollector_.GetServerNum();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = GetSuperPodNum(rankListNew, impl.superPodNum_);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    impl.deviceNumPerAggregation_ = 1;
    HcclIpAddress serverIp("172.17.10.1");
    HcclIpAddress localIp("172.17.10.1");
    SetRetryEnable(
        DevType::DEV_TYPE_910_93, impl.superPodNum_, impl.serverNum_, impl.deviceNumPerAggregation_,
        impl.isDiffDeviceType_, impl.GetAivModeConfig(), serverIp, localIp, impl.retryEnable_);
    std::cout << "superPodNum_ is " << unsigned(impl.superPodNum_) << std::endl;
    std::cout << "serverNum_ is " << unsigned(impl.serverNum_) << std::endl;
    std::cout << "deviceNumPerAggregation_ is " << unsigned(impl.deviceNumPerAggregation_) << std::endl;
    if (impl.retryEnable_) {
        std::cout << "retryEnable_ is true" << std::endl;
    } else {
        std::cout << "retryEnable_ is false" << std::endl;
    }
    EXPECT_EQ(impl.retryEnable_, false);
    unsetenv("HCCL_OP_RETRY_ENABLE");
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hccl_Communicator_SetRetryEnable_illegal_config)
{
    HcclResult ret;
    setenv("HCCL_OP_RETRY_ENABLE", " ", 1);
    ret = ParseRetryEnable();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    setenv("HCCL_OP_RETRY_ENABLE", "L0,1", 1);
    ret = ParseRetryEnable();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_OP_RETRY_ENABLE", "L3:1", 1);
    ret = ParseRetryEnable();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_OP_RETRY_ENABLE", "L2:1, L2:1", 1);
    ret = ParseRetryEnable();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_OP_RETRY_ENABLE", "L2:0, L1:0, L0:0", 1);
    ret = ParseRetryEnable();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    unsetenv("HCCL_OP_RETRY_ENABLE");
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_transport_base_stop_test)
{
    HcclResult ret = HCCL_SUCCESS;
    DispatcherPub* dispatcher;
    const std::unique_ptr<NotifyPool> notifyPool;
    std::chrono::milliseconds timeout;
    MachinePara machinePara;
    hccl::TransportBase transportBase(dispatcher, notifyPool, machinePara, timeout);
    ret = transportBase.Stop();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclImplTest, ut_transport_base_Resume_test)
{
    HcclResult ret = HCCL_SUCCESS;
    DispatcherPub* dispatcher;
    const std::unique_ptr<NotifyPool> notifyPool;
    std::chrono::milliseconds timeout;
    MachinePara machinePara;
    hccl::TransportBase transportBase(dispatcher, notifyPool, machinePara, timeout);
    ret = transportBase.Resume();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclImplTest, ut_migrate_link_to_stop_or_Resume)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclCommunicator hcclCommunicator;

    std::shared_ptr<Transport> link = std::make_shared<Transport>((TransportBase*)nullptr);

    MOCKER_CPP(&Transport::Stop).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&Transport::Resume).stubs().will(returnValue(HCCL_SUCCESS));

    ret = hcclCommunicator.MigrateLinkToStopOrResume(link, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hcclCommunicator.MigrateLinkToStopOrResume(link, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_transport_base_fence_test)
{
    HcclResult ret = HCCL_SUCCESS;
    DispatcherPub* dispatcher;
    const std::unique_ptr<NotifyPool> notifyPool;
    std::chrono::milliseconds timeout;
    MachinePara machinePara;
    std::shared_ptr<Transport> link_base(
        new Transport(new (std::nothrow) TransportBase(dispatcher, notifyPool, machinePara, timeout)));
    ret = link_base->Fence();
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);

    Stream stream(StreamType::STREAM_TYPE_ONLINE);
    ret = StreamSync(dispatcherPtr, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclImplTest, ut_migrate_link_vector_to_stop_or_Resume_test)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclCommunicator hcclCommunicator;
    const std::vector<LINK> links = {nullptr};

    MOCKER_CPP(&HcclCommunicator::MigrateLinkToStopOrResume).stubs().will(returnValue(HCCL_SUCCESS));

    ret = hcclCommunicator.MigrateLinkVectorToStopOrResume(links, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = hcclCommunicator.MigrateLinkVectorToStopOrResume(links, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_traverse_link_vector_test)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclCommunicator hcclCommunicator;
    std::vector<std::shared_ptr<Transport>> links;
    std::vector<std::unique_ptr<CommBase>> commBaseVector;

    MOCKER_CPP(&HcclCommunicator::MigrateLinkVectorToStopOrResume).stubs().will(returnValue(HCCL_SUCCESS));

    ret = hcclCommunicator.TraverseLinkVector(commBaseVector, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = hcclCommunicator.TraverseLinkVector(commBaseVector, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hccl_communicator_stop_test)
{
    HcclResult ret = HCCL_SUCCESS;
    auto notifypool = std::make_unique<NotifyPool>();
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap = {};
    std::vector<RankInfo> paraVector = {};
    std::string str = "test";
    IntraExchanger exchanger;
    DeviceMem inputMem;
    DeviceMem outputMem;
    auto commBase = std::make_unique<CommBase>(
        str, 0, 0, 0, 0, paraVector, TopoType::TOPO_TYPE_COMMON, nullptr, std::move(notifypool), netDevCtxMap,
        exchanger, inputMem, outputMem, true);
    HcclCommunicator hcclCommunicator;
    std::string tag = "test";
    CommInfo tmpComm;
    tmpComm.commLevel1.push_back(std::move(commBase));
    tmpComm.commLevel0.push_back(std::move(commBase));
    tmpComm.commLevel2.push_back(std::move(commBase));
    tmpComm.commP2P.push_back(std::move(commBase));
    tmpComm.commIntraServer = std::move(commBase);
    hcclCommunicator.tagCommInfo_.insert(std::pair<std::string, CommInfo>(tag, std::move(tmpComm)));

    MOCKER_CPP(&HcclCommunicator::MigrateLinkVectorToStopOrResume).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::TraverseLinkVector).stubs().will(returnValue(HCCL_SUCCESS));

    ret = hcclCommunicator.Stop();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hccl_communicator_Resume_test)
{
    HcclResult ret = HCCL_SUCCESS;
    auto notifypool = std::make_unique<NotifyPool>();
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap = {};
    std::vector<RankInfo> paraVector = {};
    std::string str = "test";
    IntraExchanger exchanger;
    DeviceMem inputMem;
    DeviceMem outputMem;
    auto commBase = std::make_unique<CommBase>(
        str, 0, 0, 0, 0, paraVector, TopoType::TOPO_TYPE_COMMON, nullptr, std::move(notifypool), netDevCtxMap,
        exchanger, inputMem, outputMem, true);
    HcclCommunicator hcclCommunicator;
    std::string tag = "test";
    CommInfo tmpComm;
    tmpComm.commLevel1.push_back(std::move(commBase));
    tmpComm.commLevel0.push_back(std::move(commBase));
    tmpComm.commLevel2.push_back(std::move(commBase));
    tmpComm.commP2P.push_back(std::move(commBase));
    tmpComm.commIntraServer = std::move(commBase);
    hcclCommunicator.tagCommInfo_.insert(std::pair<std::string, CommInfo>(tag, std::move(tmpComm)));
    MOCKER_CPP(&HcclCommunicator::MigrateLinkVectorToStopOrResume).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::TraverseLinkVector).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtResourceClean).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&OpRetryManager::SetRetryStateToWaitResume).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&OpRetryManager::ExitWaitResumeState).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = hcclCommunicator.Resume();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_regist_tast_abort_handler_test)
{
    hcclComm hcclcomm;
    hcclcomm.communicator_ = std::make_unique<HcclCommunicator>();
    auto ret = hcclcomm.RegistTaskAbortHandler();
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(HcclImplTest, ut_un_regist_tast_abort_handler_test)
{
    hcclComm hcclcomm;
    hcclcomm.communicator_ = std::make_unique<HcclCommunicator>();
    MOCKER(hrtTaskAbortHandleCallback).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    auto ret = hcclcomm.UnRegistTaskAbortHandler();
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(HcclImplTest, ut_hccl_comm_suspend_test)
{
    hcclComm hcclcomm;
    hcclcomm.communicator_ = std::make_unique<HcclCommunicator>();
    MOCKER(hrtTaskAbortHandleCallback).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::Suspend, HcclResult(HcclCommunicator::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    auto ret = hcclcomm.Suspend();
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(HcclImplTest, ut_TraverseSingleSubCommTransport_first_test)
{
    TransportType type = TransportType::TRANS_TYPE_P2P;
    TransportPara para;
    const std::unique_ptr<NotifyPool> notifyPool;
    MachinePara machinePara;
    const TransportDeviceP2pData& transDevP2pData = TransportDeviceP2pData();
    HcclCommunicator hcclCommunicator;
    SingleSubCommTransport singleSubCommTransport;
    TransportRequest req;
    singleSubCommTransport.transportRequests.push_back(req);
    singleSubCommTransport.transportRequests[0].isValid = false;
    auto ptr = std::make_shared<Transport>(type, para, dispatcherPtr, notifyPool, machinePara, transDevP2pData);
    singleSubCommTransport.links.push_back(ptr);

    MOCKER_CPP(&Transport::Stop).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&Transport::Resume).stubs().will(returnValue(HCCL_SUCCESS));

    auto ret = hcclCommunicator.TraverseSingleSubCommTransport(singleSubCommTransport, true);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(HcclImplTest, ut_TraverseSingleSubCommTransport_second_test)
{
    TransportType type = TransportType::TRANS_TYPE_P2P;
    TransportPara para;
    const std::unique_ptr<NotifyPool> notifyPool;
    MachinePara machinePara;
    const TransportDeviceP2pData& transDevP2pData = TransportDeviceP2pData();
    HcclCommunicator hcclCommunicator;
    SingleSubCommTransport singleSubCommTransport;
    TransportRequest req;
    singleSubCommTransport.transportRequests.push_back(req);
    singleSubCommTransport.transportRequests[0].isValid = false;
    auto ptr = std::make_shared<Transport>(type, para, dispatcherPtr, notifyPool, machinePara, transDevP2pData);
    singleSubCommTransport.links.push_back(ptr);

    MOCKER_CPP(&Transport::Stop).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&Transport::Resume).stubs().will(returnValue(HCCL_SUCCESS));

    singleSubCommTransport.transportRequests[0].isValid = true;
    auto ret = hcclCommunicator.TraverseSingleSubCommTransport(singleSubCommTransport, true);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(HcclImplTest, ut_TraverseSingleSubCommTransport_third_test)
{
    TransportType type = TransportType::TRANS_TYPE_P2P;
    TransportPara para;
    const std::unique_ptr<NotifyPool> notifyPool;
    MachinePara machinePara;
    const TransportDeviceP2pData& transDevP2pData = TransportDeviceP2pData();
    HcclCommunicator hcclCommunicator;
    SingleSubCommTransport singleSubCommTransport;
    TransportRequest req;
    singleSubCommTransport.transportRequests.push_back(req);
    singleSubCommTransport.transportRequests[0].isValid = false;
    auto ptr = std::make_shared<Transport>(type, para, dispatcherPtr, notifyPool, machinePara, transDevP2pData);
    singleSubCommTransport.links.push_back(ptr);

    MOCKER_CPP(&Transport::Stop).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&Transport::Resume).stubs().will(returnValue(HCCL_SUCCESS));

    singleSubCommTransport.transportRequests[0].isValid = true;
    auto ret = hcclCommunicator.TraverseSingleSubCommTransport(singleSubCommTransport, false);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(HcclImplTest, ut_TraverseLevelNSubCommTransport_test)
{
    LevelNSubCommTransport lev;
    lev.resize(1);
    MOCKER_CPP(&HcclCommunicator::TraverseSingleSubCommTransport).stubs().will(returnValue(HCCL_SUCCESS));

    HcclCommunicator hcclCommunicator;

    auto ret = hcclCommunicator.TraverseLevelNSubCommTransport(lev, true);
    EXPECT_EQ(HCCL_SUCCESS, ret);
    ret = hcclCommunicator.TraverseLevelNSubCommTransport(lev, false);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(HcclImplTest, ut_TraverseOpCommTransport_test)
{
    OpCommTransport op;
    op.resize(1);
    HcclCommunicator hcclCommunicator;

    MOCKER_CPP(&HcclCommunicator::TraverseLevelNSubCommTransport).stubs().will(returnValue(HCCL_SUCCESS));

    auto ret = hcclCommunicator.TraverseOpCommTransport(op, true);
    EXPECT_EQ(HCCL_SUCCESS, ret);
    ret = hcclCommunicator.TraverseOpCommTransport(op, false);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(HcclImplTest, ut_TraverseAlgResourceResponse_test)
{
    HcclCommunicator hcclCommunicator;
    AlgResourceResponse alg;
    hcclCommunicator.resMap_.insert(std::pair<std::string, AlgResourceResponse>(std::make_pair("test", alg)));
    MOCKER_CPP(&HcclCommunicator::TraverseOpCommTransport).stubs().will(returnValue(HCCL_SUCCESS));

    auto ret = hcclCommunicator.TraverseAlgResourceResponse(true);
    EXPECT_EQ(HCCL_SUCCESS, ret);
    ret = hcclCommunicator.TraverseAlgResourceResponse(false);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

HcclResult GetCmd2(hccl::HDCommunicate* This, unsigned int offset, unsigned int length, unsigned char* value)
{
    KfcExecStatus* op = reinterpret_cast<KfcExecStatus*>(value);
    op->execStatus.kfcStatus = KfcStatus::kRuning;
    return HCCL_SUCCESS;
}

HcclResult GetCmd3(hccl::HDCommunicate* This, unsigned int offset, unsigned int length, unsigned char* value)
{
    KfcExecStatus* op = reinterpret_cast<KfcExecStatus*>(value);
    op->execStatus.kfcStatus = KfcStatus::kStoplaunch;
    return HCCL_SUCCESS;
}

HcclResult GetCmd4(hccl::HDCommunicate* This, unsigned int offset, unsigned int length, unsigned char* value)
{
    KfcExecStatus* op = reinterpret_cast<KfcExecStatus*>(value);
    op->execStatus.kfcStatus = KfcStatus::kError;
    return HCCL_SUCCESS;
}

HcclResult GetCmd1(hccl::HDCommunicate* This, unsigned int offset, unsigned int length, unsigned char* value)
{
    KfcExecStatus* op = reinterpret_cast<KfcExecStatus*>(value);
    op->execStatus.kfcStatus = KfcStatus::kEnd;
    return HCCL_SUCCESS;
}

TEST_F(HcclImplTest, ut_stopExec_not_in)
{
    HcclCommunicator hcclCommunicator;
    auto ret = hcclCommunicator.StopExec();
    EXPECT_EQ(HCCL_SUCCESS, ret);
    GlobalMockObject::verify();

    hcclCommunicator.SetAicpuCommEngine(true);
    MOCKER_CPP(&HDCommunicate::Get).stubs().will(invoke(GetCmd1));
    ret = hcclCommunicator.StopExec();
    EXPECT_EQ(HCCL_SUCCESS, ret);
    hcclCommunicator.isAicpuCommEngine_ = false; // 保证析构的时候不会调用到相关命令
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_Clean_not_in)
{
    HcclCommunicator hcclCommunicator;
    auto ret = hcclCommunicator.Clean();
    EXPECT_EQ(HCCL_SUCCESS, ret);
    GlobalMockObject::verify();

    hcclCommunicator.SetAicpuCommEngine(true);
    MOCKER_CPP(&HDCommunicate::Get).stubs().will(invoke(GetCmd2));
    ret = hcclCommunicator.Clean();
    EXPECT_EQ(HCCL_SUCCESS, ret);
    hcclCommunicator.isAicpuCommEngine_ = false; // 保证析构的时候不会调用到相关命令
    GlobalMockObject::verify();
}

static void TestConstructParam_1server_8p(HcclCommParams& params, RankTable_t& rankTable)
{
    string commId = "comm ";
    memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1);
    params.rank = 0;
    params.userRank = 0;
    params.totalRanks = 8;
    params.isHeterogComm = false;
    params.logicDevId = 0;
    params.commWorkMode = WorkMode::HCCL_MODE_NORMAL;
    params.deviceType = DevType::DEV_TYPE_910;

    rankTable.collectiveId = "192.168.0.101-8000-8001";
    vector<RankInfo_t> rankVec(8);
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
    rankVec[2].deviceInfo.devicePhyId = 2;
    HcclIpAddress ipAddr3(1711319232);
    rankVec[2].deviceInfo.deviceIp.push_back(ipAddr3); // 101.0.168.192
    rankVec[2].serverIdx = 0;
    rankVec[2].serverId = "192.168.0.101";

    rankVec[3].rankId = 3;
    rankVec[3].deviceInfo.devicePhyId = 3;
    HcclIpAddress ipAddr4(1711319233);
    rankVec[3].deviceInfo.deviceIp.push_back(ipAddr4); // 101.0.168.192
    rankVec[3].serverIdx = 0;
    rankVec[3].serverId = "192.168.0.101";

    rankVec[4].rankId = 4;
    rankVec[4].deviceInfo.devicePhyId = 4;
    HcclIpAddress ipAddr5(1711319234);
    rankVec[4].deviceInfo.deviceIp.push_back(ipAddr5); // 101.0.168.192
    rankVec[4].serverIdx = 0;
    rankVec[4].serverId = "192.168.0.101";

    rankVec[5].rankId = 5;
    rankVec[5].deviceInfo.devicePhyId = 5;
    HcclIpAddress ipAddr6(1711319235);
    rankVec[5].deviceInfo.deviceIp.push_back(ipAddr6); // 101.0.168.192
    rankVec[5].serverIdx = 0;
    rankVec[5].serverId = "192.168.0.101";

    rankVec[6].rankId = 6;
    rankVec[6].deviceInfo.devicePhyId = 6;
    HcclIpAddress ipAddr7(1711319236);
    rankVec[6].deviceInfo.deviceIp.push_back(ipAddr7); // 101.0.168.192
    rankVec[6].serverIdx = 0;
    rankVec[6].serverId = "192.168.0.101";

    rankVec[7].rankId = 7;
    rankVec[7].deviceInfo.devicePhyId = 7;
    HcclIpAddress ipAddr8(1711319237);
    rankVec[7].deviceInfo.deviceIp.push_back(ipAddr8); // 101.0.168.192
    rankVec[7].serverIdx = 0;
    rankVec[7].serverId = "192.168.0.101";

    rankTable.rankList.assign(rankVec.begin(), rankVec.end());
    rankTable.rankNum = 8;
    rankTable.deviceNum = 8;
    rankTable.serverNum = 1;
}

static void TestConstructParam_1server_8p_910_93(HcclCommParams& params, RankTable_t& rankTable)
{
    string commId = "comm ";
    memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1);
    params.rank = 0;
    params.userRank = 0;
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
    rankVec[2].deviceInfo.devicePhyId = 2;
    HcclIpAddress ipAddr3(1711319232);
    rankVec[2].deviceInfo.deviceIp.push_back(ipAddr3); // 101.0.168.192
    rankVec[2].serverIdx = 0;
    rankVec[2].serverId = "192.168.0.101";

    rankVec[3].rankId = 3;
    rankVec[3].deviceInfo.devicePhyId = 3;
    HcclIpAddress ipAddr4(1711319233);
    rankVec[3].deviceInfo.deviceIp.push_back(ipAddr4); // 101.0.168.192
    rankVec[3].serverIdx = 0;
    rankVec[3].serverId = "192.168.0.101";

    rankVec[4].rankId = 4;
    rankVec[4].deviceInfo.devicePhyId = 4;
    HcclIpAddress ipAddr5(1711319234);
    rankVec[4].deviceInfo.deviceIp.push_back(ipAddr5); // 101.0.168.192
    rankVec[4].serverIdx = 0;
    rankVec[4].serverId = "192.168.0.101";

    rankVec[5].rankId = 5;
    rankVec[5].deviceInfo.devicePhyId = 5;
    HcclIpAddress ipAddr6(1711319235);
    rankVec[5].deviceInfo.deviceIp.push_back(ipAddr6); // 101.0.168.192
    rankVec[5].serverIdx = 0;
    rankVec[5].serverId = "192.168.0.101";

    rankVec[6].rankId = 6;
    rankVec[6].deviceInfo.devicePhyId = 6;
    HcclIpAddress ipAddr7(1711319236);
    rankVec[6].deviceInfo.deviceIp.push_back(ipAddr7); // 101.0.168.192
    rankVec[6].serverIdx = 0;
    rankVec[6].serverId = "192.168.0.101";

    rankVec[7].rankId = 7;
    rankVec[7].deviceInfo.devicePhyId = 7;
    HcclIpAddress ipAddr8(1711319237);
    rankVec[7].deviceInfo.deviceIp.push_back(ipAddr8); // 101.0.168.192
    rankVec[7].serverIdx = 0;
    rankVec[7].serverId = "192.168.0.101";

    rankTable.rankList.assign(rankVec.begin(), rankVec.end());
    rankTable.rankNum = 8;
    rankTable.deviceNum = 8;
    rankTable.serverNum = 1;
}

TEST_F(HcclImplTest, ut_hcclImpl_run_ring_executer_by_all_reduce_stars)
{
    HcclResult ret = HCCL_SUCCESS;
    MOCKER(GetWorkflowMode).stubs().will(returnValue(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&StreamActiveManager::StreamActive).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(GetExternalInputHcclEnableFfts).stubs().will(returnValue(false));

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam_1server_8p(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    ret = implBase->AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->InitCCLbuffer(4096, 4096);
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    impl->topoAttr_.deviceLogicId = 0;
    impl->topoAttr_.devicePhyId = 0;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_8P_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    impl->topoType_ = TopoType::TOPO_TYPE_8P_RING;
    algConfigurator->topoType_ = TopoType::TOPO_TYPE_8P_RING;
    CollAllReduceRingExecutor* executor = new CollAllReduceRingExecutor(impl->dispatcher_, topoMatcher);
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
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
    const std::vector<std::vector<u32>> tmpRingNics = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}};

    std::printf("[st_CollAllReduceRingFor91093Executor_Ring]");

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(
        opParam.tag, HcclCMDType::HCCL_CMD_ALLREDUCE, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    resourceResponse.slaveStreams.resize(3);
    resourceResponse.notifiesMain.resize(3);
    resourceResponse.notifiesAux.resize(3);
    resourceResponse.notifiesDevMain.resize(3);
    resourceResponse.notifiesDevAux.resize(3);

    resourceResponse.threadManage.resize(3);
    for (u32 ringIndex = 0; ringIndex < 3; ringIndex++) {
        resourceResponse.threadManage[ringIndex].reset(new (std::nothrow)
                                                           ThreadManage(ringIndex, ringIndex, impl->dispatcher_));
    }
    ret = executor->Orchestrate(opParam, resourceResponse);
    delete executor;

    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hcclImpl_run_ring_executer_by_all_gather_stars)
{
    HcclResult ret = HCCL_SUCCESS;
    MOCKER(GetWorkflowMode).stubs().will(returnValue(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&StreamActiveManager::StreamActive).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(GetExternalInputHcclEnableFfts).stubs().will(returnValue(false));

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam_1server_8p(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    ret = implBase->AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->InitCCLbuffer(4096, 4096);
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    impl->topoAttr_.deviceLogicId = 0;
    impl->topoAttr_.devicePhyId = 0;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_8P_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    impl->topoType_ = TopoType::TOPO_TYPE_8P_RING;
    algConfigurator->topoType_ = TopoType::TOPO_TYPE_8P_RING;
    CollAllGatherRingExecutor* executor = new CollAllGatherRingExecutor(impl->dispatcher_, topoMatcher);
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 4096;
    opParam.DataDes.count = 1024;
    opParam.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);
    const std::vector<std::vector<u32>> tmpRingNics = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}};

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(
        opParam.tag, HcclCMDType::HCCL_CMD_ALLGATHER, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    resourceResponse.slaveStreams.resize(3);
    resourceResponse.notifiesMain.resize(3);
    resourceResponse.notifiesAux.resize(3);
    resourceResponse.notifiesDevMain.resize(3);
    resourceResponse.notifiesDevAux.resize(3);

    resourceResponse.threadManage.resize(3);
    for (u32 ringIndex = 0; ringIndex < 3; ringIndex++) {
        resourceResponse.threadManage[ringIndex].reset(new (std::nothrow)
                                                           ThreadManage(ringIndex, ringIndex, impl->dispatcher_));
    }
    ret = executor->Orchestrate(opParam, resourceResponse);
    delete executor;

    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hcclImpl_run_double_ring_executer_by_all_reduce_stars_2)
{
    HcclResult ret = HCCL_SUCCESS;
    MOCKER(GetWorkflowMode).stubs().will(returnValue(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Heartbeat::Init).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&StreamActiveManager::StreamActive).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(GetExternalInputHcclEnableFfts).stubs().will(returnValue(false));

    MOCKER_CPP(&ThreadManage::WaitDone).stubs().with(mockcpp::any());

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam_1server_8p_910_93(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910_93;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    ret = implBase->AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->InitCCLbuffer(4096, 4096);
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    impl->topoAttr_.deviceLogicId = 0;
    impl->topoAttr_.devicePhyId = 0;
    algConfigurator->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    impl->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    algConfigurator->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910_93;

    CollAllReduceRingFor91093Executor* executor = new CollAllReduceRingFor91093Executor(impl->dispatcher_, topoMatcher);
    executor->algType_.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING;
    executor->algType_.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
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
    const std::vector<std::vector<u32>> tmpRingNics = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}};

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(
        opParam.tag, HcclCMDType::HCCL_CMD_ALLREDUCE, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    resourceResponse.slaveStreams.resize(3);
    resourceResponse.notifiesMain.resize(3);
    resourceResponse.notifiesAux.resize(3);
    resourceResponse.notifiesDevMain.resize(3);
    resourceResponse.notifiesDevAux.resize(3);
    resourceResponse.opTransportResponse.resize(COMM_LEVEL_RESERVED);
    resourceResponse.opTransportResponse[COMM_LEVEL0].resize(2);
    resourceResponse.opTransportResponse[COMM_LEVEL0][0].links.resize(8);
    resourceResponse.opTransportResponse[COMM_LEVEL0][1].links.resize(8);

    resourceResponse.threadManage.resize(3);
    for (u32 ringIndex = 0; ringIndex < 3; ringIndex++) {
        resourceResponse.threadManage[ringIndex].reset(new (std::nothrow)
                                                           ThreadManage(ringIndex, ringIndex, impl->dispatcher_));
    }
    ret = executor->Orchestrate(opParam, resourceResponse);
    delete executor;

    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_ReduceScatterVFor910BMesh)
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
    params.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910B;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_MESH;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    impl->topoType_ = TopoType::TOPO_TYPE_NP_MESH;

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910B;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_NP_MESH;
    std::unique_ptr<ReduceScatterVOperator> operation(
        new (std::nothrow) ReduceScatterVOperator(algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher));

    CollReduceScatterVMeshOpbaseExecutor* executor
        = new CollReduceScatterVMeshOpbaseExecutor(impl->dispatcher_, topoMatcher);

    std::vector<u64> counts{1, 2, 3, 4};
    std::vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 4096;
    opParam.VDataDes.counts = counts.data();
    opParam.VDataDes.displs = displs.data();
    opParam.VDataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.reduceType = HCCL_REDUCE_SUM;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);

    std::string algName = "";
    std::string newTag = opParam.tag;
    ret = operation->SelectAlg(tag, opParam, algName, newTag);
    opParam.tag = newTag;

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));
    SubCommInfo mockCommInfo{0, 1, std::vector<LINK>()};
    MOCKER_CPP(&CollNativeExecutorBase::GetSubCommInfo).stubs().will(returnValue(mockCommInfo));

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(
        newTag, HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    DeviceMem scratchMem = DeviceMem::alloc(4096);
    resourceResponse.scratchMem = scratchMem;
    resourceResponse.slaveStreams.resize(1);
    resourceResponse.notifiesMain.resize(1);
    resourceResponse.notifiesAux.resize(1);
    resourceResponse.notifiesDevMain.resize(1);
    resourceResponse.notifiesDevAux.resize(1);
    executor->inCCLbufferSize_ = inputMem.size();

    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    implBase = nullptr;

    delete executor;
    GlobalMockObject::verify();
}
TEST_F(HcclImplTest, ut_ReduceScatterVAivSmallCount)
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
    params.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910B;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_2P_MESH;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    impl->topoType_ = TopoType::TOPO_TYPE_2P_MESH;

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910B;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_2P_MESH;
    std::unique_ptr<ReduceScatterVOperator> operation(
        new (std::nothrow) ReduceScatterVOperator(algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher));

    CollReduceScatterVMeshAivSmallCountExecutor* executor
        = new CollReduceScatterVMeshAivSmallCountExecutor(dispatcher, topoMatcher);

    std::vector<u64> counts{1, 2, 3, 4};
    std::vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 4096;
    opParam.VDataDes.counts = counts.data();
    opParam.VDataDes.displs = displs.data();
    opParam.VDataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.reduceType = HCCL_REDUCE_SUM;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);

    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));

    MOCKER(IsSupportAIVReduce).stubs().will(returnValue(true));

    std::string algName = "";
    std::string newTag = opParam.tag;
    operation->topoMatcher_->externalEnable_.deterministic = 0;
    operation->isSingleMeshAggregation_ = true;
    ret = operation->SelectAlg(tag, opParam, algName, newTag);
    opParam.tag = newTag;

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));

    SubCommInfo mockCommInfo{0, 1, std::vector<LINK>()};
    MOCKER_CPP(&CollNativeExecutorBase::GetSubCommInfo).stubs().will(returnValue(mockCommInfo));

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(
        newTag, HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    DeviceMem scratchMem = DeviceMem::alloc(4096);
    resourceResponse.scratchMem = scratchMem;
    resourceResponse.slaveStreams.resize(1);
    resourceResponse.notifiesMain.resize(1);
    resourceResponse.notifiesAux.resize(1);
    resourceResponse.notifiesDevMain.resize(1);
    resourceResponse.notifiesDevAux.resize(1);
    executor->inCCLbufferSize_ = inputMem.size();

    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    implBase = nullptr;

    delete executor;
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hcclImpl_run_a3_aiv_coressnode_opbase)
{
    HcclResult ret = HCCL_SUCCESS;
    MOCKER(GetWorkflowMode).stubs().will(returnValue(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&StreamActiveManager::StreamActive).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(GetExternalInputHcclEnableFfts).stubs().will(returnValue(false));
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam_1server_8p(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    ret = implBase->AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->InitCCLbuffer(4096, 4096);
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    impl->topoAttr_.deviceLogicId = 0;
    impl->topoAttr_.devicePhyId = 0;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLGATHER].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_8P_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLGATHER].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;

    impl->topoType_ = TopoType::TOPO_TYPE_8P_RING;
    algConfigurator->topoType_ = TopoType::TOPO_TYPE_8P_RING;
    CollAllGatherMeshAivFor91093Executor* executor
        = new CollAllGatherMeshAivFor91093Executor(impl->dispatcher_, topoMatcher);
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
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
    const std::vector<std::vector<u32>> tmpRingNics = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}};

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    resourceRequest.aivBufferRequest = 3U;
    implBase->AllocAlgResource(
        opParam.tag, HcclCMDType::HCCL_CMD_ALLGATHER, opParam, resourceRequest, resourceResponse);
    delete executor;

    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hcclImpl_run_a3_aiv_coressnode_offload)
{
    HcclResult ret = HCCL_SUCCESS;
    MOCKER(GetWorkflowMode).stubs().will(returnValue(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB));
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&StreamActiveManager::StreamActive).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(GetExternalInputHcclEnableFfts).stubs().will(returnValue(false));
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam_1server_8p(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    ret = implBase->AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->InitCCLbuffer(4096, 4096);
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    impl->topoAttr_.deviceLogicId = 0;
    impl->topoAttr_.devicePhyId = 0;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLGATHER].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_8P_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLGATHER].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;

    impl->topoType_ = TopoType::TOPO_TYPE_8P_RING;
    algConfigurator->topoType_ = TopoType::TOPO_TYPE_8P_RING;
    CollAllGatherMeshAivFor91093Executor* executor
        = new CollAllGatherMeshAivFor91093Executor(impl->dispatcher_, topoMatcher);
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
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
    const std::vector<std::vector<u32>> tmpRingNics = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}};

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    resourceRequest.aivBufferRequest = 3U;
    implBase->AllocAlgResource(
        opParam.tag, HcclCMDType::HCCL_CMD_ALLGATHER, opParam, resourceRequest, resourceResponse);
    delete executor;

    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_ReduceScatterVFor310PRing)
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
    params.deviceType = DevType::DEV_TYPE_310P3;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_310P3;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_8P_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    impl->topoType_ = TopoType::TOPO_TYPE_4P_RING;

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_310P3;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_4P_RING;
    std::unique_ptr<ReduceScatterVOperator> operation(
        new (std::nothrow) ReduceScatterVOperator(algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher));

    CollReduceScatterVFor310PRingExecutor* executor
        = new CollReduceScatterVFor310PRingExecutor(impl->dispatcher_, topoMatcher);

    std::vector<u64> counts{1, 2, 3, 4};
    std::vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 4096;
    opParam.VDataDes.counts = counts.data();
    opParam.VDataDes.displs = displs.data();
    opParam.VDataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.reduceType = HCCL_REDUCE_SUM;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);

    std::string algName = "";
    std::string newTag = opParam.tag;
    ret = operation->SelectAlg(tag, opParam, algName, newTag);
    opParam.tag = newTag;

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));
    SubCommInfo mockCommInfo{0, 1, std::vector<LINK>()};
    MOCKER_CPP(&CollNativeExecutorBase::GetSubCommInfo).stubs().will(returnValue(mockCommInfo));

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(
        newTag, HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    DeviceMem scratchMem = DeviceMem::alloc(4096);
    resourceResponse.scratchMem = scratchMem;
    resourceResponse.slaveStreams.resize(1);
    resourceResponse.notifiesMain.resize(1);
    resourceResponse.notifiesAux.resize(1);
    resourceResponse.notifiesDevMain.resize(1);
    resourceResponse.notifiesDevAux.resize(1);
    executor->inCCLbufferSize_ = inputMem.size();

    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    implBase = nullptr;

    delete executor;
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_ReduceScatterVFor310PRingNosupportIineReduce)
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
    params.deviceType = DevType::DEV_TYPE_310P3;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_310P3;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_8P_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    impl->topoType_ = TopoType::TOPO_TYPE_4P_RING;

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_310P3;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_4P_RING;
    std::unique_ptr<ReduceScatterVOperator> operation(
        new (std::nothrow) ReduceScatterVOperator(algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher));

    CollReduceScatterVFor310PRingExecutor* executor
        = new CollReduceScatterVFor310PRingExecutor(impl->dispatcher_, topoMatcher);

    std::vector<u64> counts{1, 2, 3, 4};
    std::vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 4096;
    opParam.VDataDes.counts = counts.data();
    opParam.VDataDes.displs = displs.data();
    opParam.VDataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.reduceType = HCCL_REDUCE_MAX;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);

    std::string algName = "";
    std::string newTag = opParam.tag;
    ret = operation->SelectAlg(tag, opParam, algName, newTag);
    opParam.tag = newTag;

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));
    SubCommInfo mockCommInfo{0, 1, std::vector<LINK>()};
    MOCKER_CPP(&CollNativeExecutorBase::GetSubCommInfo).stubs().will(returnValue(mockCommInfo));

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(
        newTag, HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    DeviceMem scratchMem = DeviceMem::alloc(4096);
    resourceResponse.scratchMem = scratchMem;
    resourceResponse.slaveStreams.resize(1);
    resourceResponse.notifiesMain.resize(1);
    resourceResponse.notifiesAux.resize(1);
    resourceResponse.notifiesDevMain.resize(1);
    resourceResponse.notifiesDevAux.resize(1);
    executor->inCCLbufferSize_ = inputMem.size();

    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);

    implBase = nullptr;

    delete executor;
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_AllGatherVFor310PExecutor_Ring)
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
    params.deviceType = DevType::DEV_TYPE_310P3;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_310P3;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_8P_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    impl->topoType_ = TopoType::TOPO_TYPE_4P_RING;

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_310P3;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_4P_RING;
    std::unique_ptr<AllGatherVOperator> operation(
        new (std::nothrow) AllGatherVOperator(algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher));

    CollAllGatherVFor310PExecutor* executor = new CollAllGatherVFor310PExecutor(impl->dispatcher_, topoMatcher);

    std::vector<u64> counts{1, 2, 3, 4};
    std::vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 4096;
    opParam.VDataDes.counts = counts.data();
    opParam.VDataDes.displs = displs.data();
    opParam.VDataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);

    std::string algName = "";
    std::string newTag = opParam.tag;
    ret = operation->SelectAlg(tag, opParam, algName, newTag);
    opParam.tag = newTag;

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));
    SubCommInfo mockCommInfo{0, 1, std::vector<LINK>()};
    MOCKER_CPP(&CollNativeExecutorBase::GetSubCommInfo).stubs().will(returnValue(mockCommInfo));

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(newTag, HcclCMDType::HCCL_CMD_ALLGATHER_V, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    DeviceMem scratchMem = DeviceMem::alloc(4096);
    resourceResponse.scratchMem = scratchMem;
    resourceResponse.slaveStreams.resize(1);
    resourceResponse.notifiesMain.resize(1);
    resourceResponse.notifiesAux.resize(1);
    resourceResponse.notifiesDevMain.resize(1);
    resourceResponse.notifiesDevAux.resize(1);

    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    implBase = nullptr;

    delete executor;
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_AllGatherVFor910BExecutor_Mesh)
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
    TestConstructParamForOneServer(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910B;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_MESH;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    impl->topoType_ = TopoType::TOPO_TYPE_8P_MESH;

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910B;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_8P_MESH;
    std::unique_ptr<AllGatherVOperator> operation(
        new (std::nothrow) AllGatherVOperator(algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher));

    CollAllGatherVMeshOpbaseExecutor* executor = new CollAllGatherVMeshOpbaseExecutor(impl->dispatcher_, topoMatcher);

    std::vector<u64> counts{1, 2, 3, 4};
    std::vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 4096;
    opParam.VDataDes.counts = counts.data();
    opParam.VDataDes.displs = displs.data();
    opParam.VDataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);

    std::string algName = "";
    std::string newTag = opParam.tag;
    ret = operation->SelectAlg(tag, opParam, algName, newTag);
    opParam.tag = newTag;

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));
    SubCommInfo mockCommInfo{0, 1, std::vector<LINK>()};
    MOCKER_CPP(&CollNativeExecutorBase::GetSubCommInfo).stubs().will(returnValue(mockCommInfo));

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(newTag, HcclCMDType::HCCL_CMD_ALLGATHER_V, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    DeviceMem scratchMem = DeviceMem::alloc(4096);
    resourceResponse.scratchMem = scratchMem;
    resourceResponse.slaveStreams.resize(1);
    resourceResponse.notifiesMain.resize(1);
    resourceResponse.notifiesAux.resize(1);
    resourceResponse.notifiesDevMain.resize(1);
    resourceResponse.notifiesDevAux.resize(1);

    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    implBase->profilerManager_->TaskAivProfilerHandle(nullptr, 10);

    implBase = nullptr;

    delete executor;
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_AllGatherVFor91093Executor_ZeroCount)
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
    TestConstructParamForOneServer(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910_93;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910_93;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER].algoLevel0
        = AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    impl->topoType_ = TopoType::ALG_LEVEL0_NP_DOUBLE_RING;

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910_93;
    topoMatcher->topoInfo_.topoType = TopoType::ALG_LEVEL0_NP_DOUBLE_RING;
    std::unique_ptr<AllGatherVOperator> operation(
        new (std::nothrow) AllGatherVOperator(algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher));

    CollAlignedAllGatherVDoubleRingFor91093Executor* executor
        = new CollAlignedAllGatherVDoubleRingFor91093Executor(impl->dispatcher_, topoMatcher);

    std::vector<u64> counts{0, 1, 0, 0};
    std::vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 0;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 1;
    opParam.VDataDes.counts = counts.data();
    opParam.VDataDes.displs = displs.data();
    opParam.VDataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);

    std::string algName = "";
    std::string newTag = opParam.tag;
    ret = operation->SelectAlg(tag, opParam, algName, newTag);
    opParam.tag = newTag;

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));
    SubCommInfo mockCommInfo{0, 1, std::vector<LINK>()};
    MOCKER_CPP(&CollNativeExecutorBase::GetSubCommInfo).stubs().will(returnValue(mockCommInfo));

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(newTag, HcclCMDType::HCCL_CMD_ALLGATHER_V, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    DeviceMem scratchMem = DeviceMem::alloc(4096);
    resourceResponse.scratchMem = scratchMem;
    resourceResponse.slaveStreams.resize(1);
    resourceResponse.notifiesMain.resize(1);
    resourceResponse.notifiesAux.resize(1);
    resourceResponse.notifiesDevMain.resize(1);
    resourceResponse.notifiesDevAux.resize(1);

    ret = executor->Orchestrate(opParam, resourceResponse);
    // EXPECT_EQ(ret, HCCL_SUCCESS);

    implBase->profilerManager_->TaskAivProfilerHandle(nullptr, 10);

    implBase = nullptr;

    delete executor;
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_AllGatherVAivBigCount)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParamForOneServer(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910B;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_2P_MESH;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    impl->topoType_ = TopoType::TOPO_TYPE_2P_MESH;

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910B;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_2P_MESH;
    std::unique_ptr<AllGatherVOperator> operation(
        new (std::nothrow) AllGatherVOperator(algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher));

    AllGatherVMeshAivExecutor* executor = new AllGatherVMeshAivExecutor(dispatcher, topoMatcher);

    std::vector<u64> counts{4194304, 2, 3, 4};
    std::vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 4096;
    opParam.VDataDes.counts = counts.data();
    opParam.VDataDes.displs = displs.data();
    opParam.VDataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.reduceType = HCCL_REDUCE_SUM;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);

    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));

    MOCKER(IsSupportAIVCopy).stubs().will(returnValue(true));

    std::string algName = "";
    std::string newTag = opParam.tag;
    operation->topoMatcher_->externalEnable_.deterministic = 0;
    operation->isSingleMeshAggregation_ = true;
    ret = operation->SelectAlg(tag, opParam, algName, newTag);
    opParam.tag = newTag;

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));

    SubCommInfo mockCommInfo{0, 1, std::vector<LINK>()};
    MOCKER_CPP(&CollNativeExecutorBase::GetSubCommInfo).stubs().will(returnValue(mockCommInfo));

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(newTag, HcclCMDType::HCCL_CMD_ALLGATHER_V, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    DeviceMem scratchMem = DeviceMem::alloc(4096);
    resourceResponse.scratchMem = scratchMem;
    resourceResponse.slaveStreams.resize(1);
    resourceResponse.notifiesMain.resize(1);
    resourceResponse.notifiesAux.resize(1);
    resourceResponse.notifiesDevMain.resize(1);
    resourceResponse.notifiesDevAux.resize(1);
    executor->inCCLbufferSize_ = inputMem.size();

    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    implBase = nullptr;

    delete executor;
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_AllGatherVAivSmallCount)
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
    params.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910B;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLGATHER_V].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_2P_MESH;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLGATHER_V].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    impl->topoType_ = TopoType::TOPO_TYPE_2P_MESH;

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910B;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_2P_MESH;
    std::unique_ptr<AllGatherVOperator> operation(
        new (std::nothrow) AllGatherVOperator(algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher));

    CollAllGatherVMeshAivSmallCountExecutor* executor
        = new CollAllGatherVMeshAivSmallCountExecutor(dispatcher, topoMatcher);

    std::vector<u64> counts{1, 2, 3, 4};
    std::vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 4096;
    opParam.VDataDes.counts = counts.data();
    opParam.VDataDes.displs = displs.data();
    opParam.VDataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.reduceType = HCCL_REDUCE_SUM;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);

    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));

    MOCKER(IsSupportAIVCopy).stubs().will(returnValue(true));

    MOCKER(GetExternalInputProfilingMode).stubs().will(returnValue(true));

    std::string algName = "";
    std::string newTag = opParam.tag;
    operation->topoMatcher_->externalEnable_.deterministic = 0;
    operation->isSingleMeshAggregation_ = true;
    ret = operation->SelectAlg(tag, opParam, algName, newTag);
    opParam.tag = newTag;

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));

    SubCommInfo mockCommInfo{0, 1, std::vector<LINK>()};
    MOCKER_CPP(&CollNativeExecutorBase::GetSubCommInfo).stubs().will(returnValue(mockCommInfo));

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(newTag, HcclCMDType::HCCL_CMD_ALLGATHER_V, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    DeviceMem scratchMem = DeviceMem::alloc(4096);
    resourceResponse.scratchMem = scratchMem;
    resourceResponse.slaveStreams.resize(1);
    resourceResponse.notifiesMain.resize(1);
    resourceResponse.notifiesAux.resize(1);
    resourceResponse.notifiesDevMain.resize(1);
    resourceResponse.notifiesDevAux.resize(1);
    executor->inCCLbufferSize_ = inputMem.size();

    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    implBase = nullptr;

    delete executor;
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_AllGatherV_offload_For910B_Executor_Mesh_v1)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParamForOneServer(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910B;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_4P_MESH;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    impl->topoType_ = TopoType::TOPO_TYPE_4P_MESH;

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910B;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_4P_MESH;
    std::unique_ptr<AllGatherVOperator> operation(
        new (std::nothrow) AllGatherVOperator(algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher));

    CollAllGatherVMeshExecutor* executor = new CollAllGatherVMeshExecutor(impl->dispatcher_, topoMatcher);

    std::vector<u64> counts{1, 2, 3, 4};
    std::vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 4096;
    opParam.VDataDes.counts = counts.data();
    opParam.VDataDes.displs = displs.data();
    opParam.VDataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.stream = Stream(StreamType::STREAM_TYPE_OFFLINE);

    std::string algName = "";
    std::string newTag = opParam.tag;
    ret = operation->SelectAlg(tag, opParam, algName, newTag);
    opParam.tag = newTag;

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));
    SubCommInfo mockCommInfo{0, 1, std::vector<LINK>()};
    MOCKER_CPP(&CollNativeExecutorBase::GetSubCommInfo).stubs().will(returnValue(mockCommInfo));

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(newTag, HcclCMDType::HCCL_CMD_ALLGATHER_V, opParam, resourceRequest, resourceResponse);
    resourceResponse.paramInputMem = inputMem;
    resourceResponse.paramOutputMem = outputMem;
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    DeviceMem scratchMem = DeviceMem::alloc(4096);
    resourceResponse.scratchMem = scratchMem;

    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    implBase = nullptr;

    delete executor;
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_AllGatherV_offload_For910B_Executor_Mesh_v2)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910B;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_4P_MESH;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    impl->topoType_ = TopoType::TOPO_TYPE_4P_MESH;

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910B;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_4P_MESH;
    std::unique_ptr<AllGatherVOperator> operation(
        new (std::nothrow) AllGatherVOperator(algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher));

    CollAllGatherVMeshExecutor* executor = new CollAllGatherVMeshExecutor(impl->dispatcher_, topoMatcher);

    std::vector<u64> counts{1, 2, 3, 4};
    std::vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 4096;
    opParam.VDataDes.counts = counts.data();
    opParam.VDataDes.displs = displs.data();
    opParam.VDataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.stream = Stream(StreamType::STREAM_TYPE_OFFLINE);

    std::string algName = "";
    std::string newTag = opParam.tag;
    ret = operation->SelectAlg(tag, opParam, algName, newTag);
    opParam.tag = newTag;

    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    implBase = nullptr;
    delete executor;
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_ReduceScatterVAivBigCount)
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
    params.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910B;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_2P_MESH;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    impl->topoType_ = TopoType::TOPO_TYPE_2P_MESH;

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    CCLBufferManager& cclBufferManager = implBase->implAlg_->cclBufferManager_;
    const HcclDispatcher dispatcher = implBase->implAlg_->dispatcher_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910B;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_2P_MESH;
    std::unique_ptr<ReduceScatterVOperator> operation(
        new (std::nothrow) ReduceScatterVOperator(algConfigurator.get(), cclBufferManager, dispatcher, topoMatcher));

    CollReduceScatterVAIVBigCountExecutor* executor
        = new CollReduceScatterVAIVBigCountExecutor(dispatcher, topoMatcher);

    std::vector<u64> counts{4194304, 8388608, 12582912, 16777216};
    std::vector<u64> displs{0};
    for (auto i = 1; i < counts.size(); ++i) {
        displs.emplace_back(displs[i - 1] + counts[i - 1]);
    }

    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 4096;
    opParam.VDataDes.counts = counts.data();
    opParam.VDataDes.displs = displs.data();
    opParam.VDataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.reduceType = HCCL_REDUCE_SUM;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);

    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));

    MOCKER(IsSupportAIVReduce).stubs().will(returnValue(true));

    std::string algName = "";
    std::string newTag = opParam.tag;
    operation->topoMatcher_->externalEnable_.deterministic = 0;
    operation->isSingleMeshAggregation_ = true;
    ret = operation->SelectAlg(tag, opParam, algName, newTag);
    opParam.tag = newTag;

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));

    SubCommInfo mockCommInfo{0, 1, std::vector<LINK>()};
    MOCKER_CPP(&CollNativeExecutorBase::GetSubCommInfo).stubs().will(returnValue(mockCommInfo));

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(
        newTag, HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    DeviceMem scratchMem = DeviceMem::alloc(4096);
    resourceResponse.scratchMem = scratchMem;
    resourceResponse.slaveStreams.resize(1);
    resourceResponse.notifiesMain.resize(1);
    resourceResponse.notifiesAux.resize(1);
    resourceResponse.notifiesDevMain.resize(1);
    resourceResponse.notifiesDevAux.resize(1);
    executor->inCCLbufferSize_ = inputMem.size();

    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    implBase = nullptr;

    delete executor;
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, invalid_valid__reserve_release_IpcMemory)
{
    MOCKER_CPP(&HcclCommunicator::SetMemoryRange).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::ActivateCommMemory).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::DeactivateCommMemory).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::UnsetMemoryRange).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::InitZeroCopyMemoryAgent).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::DeinitZeroCopyMemoryAgent)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    hcclComm hcclcomm;
    hcclcomm.communicator_ = std::make_unique<HcclCommunicator>();
    char dummyReserveMem[10];
    void* vir_ptr = dummyReserveMem;
    auto devID = 0;
    auto reserve_mem = 10;
    EXPECT_EQ(HCCL_SUCCESS, HcclCommSetMemoryRange(reinterpret_cast<HcclComm>(&hcclcomm), vir_ptr, reserve_mem, 0, 0));
    int dummyHandle = 0;
    aclrtDrvMemHandle mem_handle = &dummyHandle;
    EXPECT_EQ(
        HCCL_SUCCESS,
        HcclCommActivateCommMemory(reinterpret_cast<HcclComm>(&hcclcomm), vir_ptr, reserve_mem, 0, mem_handle, 0));
    EXPECT_EQ(HCCL_SUCCESS, HcclCommDeactivateCommMemory(reinterpret_cast<HcclComm>(&hcclcomm), vir_ptr));
    EXPECT_EQ(HCCL_SUCCESS, HcclCommUnsetMemoryRange(reinterpret_cast<HcclComm>(&hcclcomm), vir_ptr));
}

TEST_F(HcclImplTest, ut_hcclimpl_AiCpuSetCommResource_MultiServer)
{
    u32 ifnumVersion = 3;
    MOCKER(hrtRaGetInterfaceVersion)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&ifnumVersion))
        .will(returnValue(HCCL_SUCCESS));

    void* commInputPtr = nullptr;
    u64 commInputSize;
    void* commOutputPtr = nullptr;
    u64 commOutputSize = 0;
    HcclResult ret;
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    std::string tag = "test";
    CommInfo tmpComm;
    std::vector<RankInfo> paraVector;
    u32 rankSize = 8;
    u32 curRankId = 0;
    u64 commBufferSize = 20;

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclImpl* impl = implBase->implAlg_->pimpl_.get();

    ret = implBase->GetInCCLbuffer(commInputPtr, commInputSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = implBase->GetOutCCLbuffer(commOutputPtr, commOutputSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    DeviceMem expMem = implBase->cclBufferManager_.GetCommExpBuffer();

    implBase->notifyPool_.reset(new (std::nothrow) NotifyPool());
    ret = implBase->notifyPool_->Init(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    IntraExchanger exchanger{};
    tmpComm.commLevel0.resize(4);
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;
    for (int i = 0; i < 4; i++) {
        tmpComm.commLevel0[i].reset(new (std::nothrow) CommRing(
            tag, 0, 8, curRankId, rankSize, TopoType::TOPO_TYPE_8P_RING, implBase->dispatcher_, implBase->notifyPool_,
            netDevCtxMap, exchanger, paraVector, inputMem, outputMem, false));
    }
    EXPECT_EQ(tmpComm.commLevel0[0]->transportInfo_.size(), rankSize);
    tmpComm.commLevel0[0]->transportInfo_.clear();
    std::chrono::milliseconds kdefaultTimeout = std::chrono::seconds(120);
    MachinePara linkPara;
    std::shared_ptr<Transport> link(
        new Transport(new (std::nothrow) TransportBase(nullptr, nullptr, linkPara, kdefaultTimeout)));
    ret = implBase->notifyPool_->RegisterOp(tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::shared_ptr<LocalIpcNotify> localNotify = nullptr;
    std::shared_ptr<RemoteNotify> remoteNotify = nullptr;

    RemoteRankInfo info(0, -1);
    SalGetBareTgid(&info.remotePid);
    ret = implBase->notifyPool_->Alloc(tag, info, localNotify, NotifyLoadType::DEVICE_NOTIFY);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<u8> data(NOTIFY_INFO_LENGTH, 0);
    ret = localNotify->Serialize(data);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remoteNotify.reset(new (std::nothrow) RemoteNotify());

    ret = remoteNotify->Init(data);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = remoteNotify->Open();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    link->pimpl_->remoteSendDoneDeviceNotify_ = remoteNotify;
    link->pimpl_->localSendDoneDeviceNotify_ = localNotify;
    link->pimpl_->remoteSendReadyDeviceNotify_ = remoteNotify;
    link->pimpl_->localSendReadyDeviceNotify_ = localNotify;
    for (u32 ringIndex = 0; ringIndex < rankSize; ringIndex++) {
        tmpComm.commLevel0[0]->transportInfo_.push_back(link);
    }
    implBase->tagCommInfo_.insert(std::pair<std::string, CommInfo>(tag, std::move(tmpComm)));

    Level1StreamInfo tmpInnerStreamInfo;
    tmpInnerStreamInfo.ringNum = rankSize;
    tmpInnerStreamInfo.ringDeviceSignal.resize(rankSize - 1);
    tmpInnerStreamInfo.ringDeviceSignalAux.resize(rankSize - 1);
    tmpInnerStreamInfo.ringDeviceStreams.resize(rankSize);

    for (u32 ringIndex = 0; ringIndex < tmpInnerStreamInfo.ringNum; ringIndex++) {
        tmpInnerStreamInfo.ringDeviceStreams[ringIndex] = Stream(StreamType::STREAM_TYPE_DEVICE);
        if (ringIndex != tmpInnerStreamInfo.ringNum - 1) {
            tmpInnerStreamInfo.ringDeviceSignal[ringIndex] = localNotify;
            tmpInnerStreamInfo.ringDeviceSignalAux[ringIndex] = localNotify;
        }
    }

    MOCKER_CPP(&Transport::GetRemoteMem, HcclResult(Transport::*)(hccl::UserMemType, void**))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&Transport::GetRemoteMem, HcclResult(Transport::*)(std::vector<void*>*))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&Transport::GetRemoteMemKey, HcclResult(Transport::*)(hccl::UserMemType, uint32_t*))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&Transport::GetLocalMemDetails).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    std::vector<HcclQpInfoV2> qpInfos(1);
    MOCKER_CPP(&Transport::GetAiQpInfo).stubs().with(outBound(qpInfos)).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&Transport::GetChipId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    // 配置profiling开关
    auto& profilingManager = hccl::ProfilingManager::Instance();
    profilingManager.StartAdditionInfoSubscribe();

    MachinePara machinePara;
    machinePara.localDeviceId = 0;
    std::chrono::milliseconds timeout;

    HcclIpAddress remoteIp{};
    HcclIpAddress localIp{};
    std::shared_ptr<HcclSocket> newSocket(
        new (std::nothrow) HcclSocket("test", nullptr, remoteIp, 0, HcclSocketRole::SOCKET_ROLE_SERVER));
    machinePara.sockets.push_back(newSocket);

    std::unique_ptr<NotifyPool> notifyPool = nullptr;
    notifyPool.reset(new (std::nothrow) NotifyPool());
    EXPECT_NE(notifyPool, nullptr);
    ret = notifyPool->Init(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    TransportBase transportBase(dispatcher, notifyPool, machinePara, timeout);
    MOCKER_CPP_VIRTUAL(transportBase, &TransportBase::GetRemoteMemSize)
        .stubs()
        .with(mockcpp::any(), outBound(commBufferSize))
        .will(returnValue(HCCL_SUCCESS));

    rtStream_t aiCpuStream;
    Stream stream(aiCpuStream);
    implBase->isA2MC2MultiServer_ = true;
    implBase->tagStreamInfo_.insert(std::pair<std::string, Level1StreamInfo>(tag, std::move(tmpInnerStreamInfo)));

    ret = implBase->SetCommResource(
        commBufferSize, commInputPtr, commOutputPtr, expMem.ptr(), implBase->tagCommInfo_[tag].commLevel0[0].get(),
        implBase->tagStreamInfo_[tag], stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u64 profConfigL0 = 0x84000985;
    profilingManager.StopSubscribe(profConfigL0);

    HCCL_INFO("check if HcclCombinOpParam is match with aicpu struct HccCommResParamTask");
    HcclCombinOpParam* combinOparaPtr = reinterpret_cast<HcclCombinOpParam*>(implBase->combinOparaMem_->ptr());
    HcclCombinOpParam& combinOpara = *combinOparaPtr;
    EXPECT_EQ(implBase->combinOparaMem_->size(), 8928); // 请确认HccCommResParamTask同步修改
    EXPECT_EQ(combinOpara.rankId, curRankId);
    EXPECT_EQ(combinOpara.signalInfo.aicpuNotify.rankId, curRankId);
    EXPECT_EQ(combinOpara.rankNum, rankSize);
    EXPECT_EQ(combinOpara.winSize, commBufferSize);

    EXPECT_EQ(implBase->transDevIbverbsDataMem_->size() / sizeof(TransportDeviceNormalData), rankSize);
    TransportDeviceNormalData* transDevIbverbsData
        = reinterpret_cast<TransportDeviceNormalData*>(implBase->transDevIbverbsDataMem_->ptr());
    for (u32 i = 0; i < rankSize; i++) {
        if (i != curRankId) {
            void* bufferIn = nullptr;
            void* bufferOut = nullptr;
            uint32_t remoteInMemKey = 0;
            uint32_t remoteOutMemKey = 0;
            EXPECT_EQ(transDevIbverbsData[i].remoteInputMem.addr, reinterpret_cast<uint64_t>(commInputPtr));
            EXPECT_EQ(transDevIbverbsData[i].remoteInputMem.size, commBufferSize);
            EXPECT_EQ(transDevIbverbsData[i].remoteOutputMem.addr, reinterpret_cast<uint64_t>(commOutputPtr));
            EXPECT_EQ(transDevIbverbsData[i].remoteOutputMem.size, commBufferSize);
        } else {
            // 本rank的信息
            EXPECT_EQ(transDevIbverbsData[i].localInputMem.addr, reinterpret_cast<uint64_t>(commInputPtr));
            EXPECT_EQ(transDevIbverbsData[i].localInputMem.size, commBufferSize);
            EXPECT_EQ(transDevIbverbsData[i].localOutputMem.addr, reinterpret_cast<uint64_t>(commOutputPtr));
            EXPECT_EQ(transDevIbverbsData[i].localOutputMem.size, commBufferSize);
        }
    }

    implBase = nullptr;
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hcclimpl_AiCpuSetCommResource_AIVHierarchy)
{
    void* commInputPtr = nullptr;
    u64 commInputSize;
    void* commOutputPtr = nullptr;
    u64 commOutputSize = 0;
    HcclResult ret;
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    std::string tag = "test";
    CommInfo tmpComm;
    std::vector<RankInfo> paraVector;
    u32 rankSize = 8;
    u32 curRankId = 0;
    u64 commBufferSize = 20;

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclImpl* impl = implBase->implAlg_->pimpl_.get();

    ret = implBase->GetInCCLbuffer(commInputPtr, commInputSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = implBase->GetOutCCLbuffer(commOutputPtr, commOutputSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    DeviceMem expMem = implBase->cclBufferManager_.GetCommExpBuffer();

    implBase->notifyPool_.reset(new (std::nothrow) NotifyPool());
    ret = implBase->notifyPool_->Init(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    IntraExchanger exchanger{};
    tmpComm.commLevel0.resize(4);
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;
    for (int i = 0; i < 4; i++) {
        tmpComm.commLevel0[i].reset(new (std::nothrow) CommMesh(
            tag, 0, 4, curRankId, rankSize, TopoType::TOPO_TYPE_8P_MESH, implBase->dispatcher_, implBase->notifyPool_,
            netDevCtxMap, exchanger, paraVector, inputMem, outputMem, true, ""));
    }
    EXPECT_EQ(tmpComm.commLevel0[0]->transportInfo_.size(), rankSize);
    tmpComm.commLevel0[0]->transportInfo_.clear();
    std::chrono::milliseconds kdefaultTimeout = std::chrono::seconds(120);
    /*创建link*/
    MachinePara machine_para;

    machine_para.localDeviceId = 0;
    machine_para.deviceLogicId = 0;
    machine_para.nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE;
    machine_para.localIpAddr = HcclIpAddress("192.168.0.23");
    std::shared_ptr<Transport> link = nullptr;
    TransportPara para = {};

    link.reset(new Transport(TransportType::TRANS_TYPE_IBV_EXP, para, dispatcher, implBase->notifyPool_, machine_para));
    link->Init();
    ret = implBase->notifyPool_->RegisterOp(tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::shared_ptr<LocalIpcNotify> localNotify = nullptr;
    std::shared_ptr<RemoteNotify> remoteNotify = nullptr;

    RemoteRankInfo info(0, -1);
    SalGetBareTgid(&info.remotePid);
    ret = implBase->notifyPool_->Alloc(tag, info, localNotify, NotifyLoadType::DEVICE_NOTIFY);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<u8> data(NOTIFY_INFO_LENGTH, 0);
    ret = localNotify->Serialize(data);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remoteNotify.reset(new (std::nothrow) RemoteNotify());

    ret = remoteNotify->Init(data);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = remoteNotify->Open();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    link->pimpl_->remoteSendDoneDeviceNotify_ = remoteNotify;
    link->pimpl_->localSendDoneDeviceNotify_ = localNotify;
    link->pimpl_->remoteSendReadyDeviceNotify_ = remoteNotify;
    link->pimpl_->localSendReadyDeviceNotify_ = localNotify;
    for (u32 ringIndex = 0; ringIndex < rankSize; ringIndex++) {
        tmpComm.commLevel0[0]->transportInfo_.push_back(link);
        tmpComm.commLevel0[0]->transportType_.push_back(TransportType::TRANS_TYPE_IBV_EXP);
    }
    implBase->tagCommInfo_.insert(std::pair<std::string, CommInfo>(tag, std::move(tmpComm)));

    Level1StreamInfo tmpInnerStreamInfo;
    tmpInnerStreamInfo.ringNum = rankSize;
    tmpInnerStreamInfo.ringDeviceSignal.resize(rankSize - 1);
    tmpInnerStreamInfo.ringDeviceSignalAux.resize(rankSize - 1);
    tmpInnerStreamInfo.ringDeviceStreams.resize(rankSize);

    for (u32 ringIndex = 0; ringIndex < tmpInnerStreamInfo.ringNum; ringIndex++) {
        tmpInnerStreamInfo.ringDeviceStreams[ringIndex] = Stream(StreamType::STREAM_TYPE_DEVICE);
        if (ringIndex != tmpInnerStreamInfo.ringNum - 1) {
            tmpInnerStreamInfo.ringDeviceSignal[ringIndex] = localNotify;
            tmpInnerStreamInfo.ringDeviceSignalAux[ringIndex] = localNotify;
        }
    }

    MOCKER_CPP(&Transport::GetRemoteMem, HcclResult(Transport::*)(hccl::UserMemType, void**))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&Transport::GetRemoteMem, HcclResult(Transport::*)(std::vector<void*>*))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&Transport::GetRemoteMemKey, HcclResult(Transport::*)(hccl::UserMemType, uint32_t*))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&Transport::GetLocalMemDetails).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    std::vector<HcclQpInfoV2> qpInfos(1);
    MOCKER_CPP(&Transport::GetAiQpInfo).stubs().with(outBound(qpInfos)).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&Transport::GetChipId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    // 配置profiling开关
    auto& profilingManager = hccl::ProfilingManager::Instance();
    profilingManager.StartAdditionInfoSubscribe();

    machine_para.localDeviceId = 0;
    std::chrono::milliseconds timeout;

    HcclIpAddress remoteIp{};
    HcclIpAddress localIp{};
    std::shared_ptr<HcclSocket> newSocket(
        new (std::nothrow) HcclSocket("test", nullptr, remoteIp, 0, HcclSocketRole::SOCKET_ROLE_SERVER));
    machine_para.sockets.push_back(newSocket);

    std::unique_ptr<NotifyPool> notifyPool = nullptr;
    notifyPool.reset(new (std::nothrow) NotifyPool());
    EXPECT_NE(notifyPool, nullptr);
    ret = notifyPool->Init(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    TransportBase transportBase(dispatcher, notifyPool, machine_para, timeout);
    MOCKER_CPP_VIRTUAL(transportBase, &TransportBase::GetRemoteMemSize)
        .stubs()
        .with(mockcpp::any(), outBound(commBufferSize))
        .will(returnValue(HCCL_SUCCESS));

    rtStream_t aiCpuStream;
    Stream stream(aiCpuStream);
    implBase->isA2MC2MultiServer_ = true;
    implBase->tagStreamInfo_.insert(std::pair<std::string, Level1StreamInfo>(tag, std::move(tmpInnerStreamInfo)));
    ret = implBase->SetDevIbverbsData(
        implBase->tagCommInfo_[tag].commLevel0[0].get(), false, commBufferSize, commInputPtr, commOutputPtr);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u64 profConfigL0 = 0x84000985;
    profilingManager.StopSubscribe(profConfigL0);

    HCCL_INFO("check if HcclCombinOpParam is match with aicpu struct HccCommResParamTask");

    implBase = nullptr;
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_hcclimpl_AiCpuSetCommResource_AIVRoce)
{
    u32 ifnumVersion = 3;
    MOCKER(hrtRaGetInterfaceVersion)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&ifnumVersion))
        .will(returnValue(HCCL_SUCCESS));

    void* commInputPtr = nullptr;
    u64 commInputSize;
    void* commOutputPtr = nullptr;
    u64 commOutputSize = 0;
    HcclResult ret;
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    std::string tag = "test";
    CommInfo tmpComm;
    std::vector<RankInfo> paraVector;
    u32 rankSize = 8;
    u32 curRankId = 0;
    u64 commBufferSize = 20;

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclImpl* impl = implBase->implAlg_->pimpl_.get();

    ret = implBase->GetInCCLbuffer(commInputPtr, commInputSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = implBase->GetOutCCLbuffer(commOutputPtr, commOutputSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    DeviceMem expMem = implBase->cclBufferManager_.GetCommExpBuffer();

    implBase->notifyPool_.reset(new (std::nothrow) NotifyPool());
    ret = implBase->notifyPool_->Init(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    IntraExchanger exchanger{};
    tmpComm.commLevel0.resize(4);
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;
    for (int i = 0; i < 4; i++) {
        tmpComm.commLevel0[i].reset(new (std::nothrow) CommMesh(
            tag, 0, 4, curRankId, rankSize, TopoType::TOPO_TYPE_8P_MESH, implBase->dispatcher_, implBase->notifyPool_,
            netDevCtxMap, exchanger, paraVector, inputMem, outputMem, true, ""));
    }
    EXPECT_EQ(tmpComm.commLevel0[0]->transportInfo_.size(), rankSize);
    tmpComm.commLevel0[0]->transportInfo_.clear();
    std::chrono::milliseconds kdefaultTimeout = std::chrono::seconds(120);
    /*创建link*/
    MachinePara machine_para;

    machine_para.localDeviceId = 0;
    machine_para.deviceLogicId = 0;
    machine_para.nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE;
    machine_para.localIpAddr = HcclIpAddress("192.168.0.23");
    std::shared_ptr<Transport> link = nullptr;
    TransportPara para = {};

    link.reset(new Transport(TransportType::TRANS_TYPE_IBV_EXP, para, dispatcher, implBase->notifyPool_, machine_para));
    link->Init();
    ret = implBase->notifyPool_->RegisterOp(tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::shared_ptr<LocalIpcNotify> localNotify = nullptr;
    std::shared_ptr<RemoteNotify> remoteNotify = nullptr;

    RemoteRankInfo info(0, -1);
    SalGetBareTgid(&info.remotePid);
    ret = implBase->notifyPool_->Alloc(tag, info, localNotify, NotifyLoadType::DEVICE_NOTIFY);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<u8> data(NOTIFY_INFO_LENGTH, 0);
    ret = localNotify->Serialize(data);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remoteNotify.reset(new (std::nothrow) RemoteNotify());

    ret = remoteNotify->Init(data);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = remoteNotify->Open();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    link->pimpl_->remoteSendDoneDeviceNotify_ = remoteNotify;
    link->pimpl_->localSendDoneDeviceNotify_ = localNotify;
    link->pimpl_->remoteSendReadyDeviceNotify_ = remoteNotify;
    link->pimpl_->localSendReadyDeviceNotify_ = localNotify;
    for (u32 ringIndex = 0; ringIndex < rankSize; ringIndex++) {
        tmpComm.commLevel0[0]->transportInfo_.push_back(link);
        tmpComm.commLevel0[0]->transportType_.push_back(TransportType::TRANS_TYPE_IBV_EXP);
    }
    implBase->tagCommInfo_.insert(std::pair<std::string, CommInfo>(tag, std::move(tmpComm)));

    Level1StreamInfo tmpInnerStreamInfo;
    tmpInnerStreamInfo.ringNum = rankSize;
    tmpInnerStreamInfo.ringDeviceSignal.resize(rankSize - 1);
    tmpInnerStreamInfo.ringDeviceSignalAux.resize(rankSize - 1);
    tmpInnerStreamInfo.ringDeviceStreams.resize(rankSize);

    for (u32 ringIndex = 0; ringIndex < tmpInnerStreamInfo.ringNum; ringIndex++) {
        tmpInnerStreamInfo.ringDeviceStreams[ringIndex] = Stream(StreamType::STREAM_TYPE_DEVICE);
        if (ringIndex != tmpInnerStreamInfo.ringNum - 1) {
            tmpInnerStreamInfo.ringDeviceSignal[ringIndex] = localNotify;
            tmpInnerStreamInfo.ringDeviceSignalAux[ringIndex] = localNotify;
        }
    }

    MOCKER_CPP(&Transport::GetRemoteMem, HcclResult(Transport::*)(hccl::UserMemType, void**))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&Transport::GetRemoteMem, HcclResult(Transport::*)(std::vector<void*>*))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&Transport::GetRemoteMemKey, HcclResult(Transport::*)(hccl::UserMemType, uint32_t*))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&Transport::GetLocalMemDetails).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    std::vector<HcclQpInfoV2> qpInfos(1);
    MOCKER_CPP(&Transport::GetAiQpInfo).stubs().with(outBound(qpInfos)).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&Transport::GetChipId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    // 配置profiling开关
    auto& profilingManager = hccl::ProfilingManager::Instance();
    profilingManager.StartAdditionInfoSubscribe();

    machine_para.localDeviceId = 0;
    std::chrono::milliseconds timeout;

    HcclIpAddress remoteIp{};
    HcclIpAddress localIp{};
    std::shared_ptr<HcclSocket> newSocket(
        new (std::nothrow) HcclSocket("test", nullptr, remoteIp, 0, HcclSocketRole::SOCKET_ROLE_SERVER));
    machine_para.sockets.push_back(newSocket);

    std::unique_ptr<NotifyPool> notifyPool = nullptr;
    notifyPool.reset(new (std::nothrow) NotifyPool());
    EXPECT_NE(notifyPool, nullptr);
    ret = notifyPool->Init(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    TransportBase transportBase(dispatcher, notifyPool, machine_para, timeout);
    MOCKER_CPP_VIRTUAL(transportBase, &TransportBase::GetRemoteMemSize)
        .stubs()
        .with(mockcpp::any(), outBound(commBufferSize))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemAsyncCopy).stubs().will(returnValue(HCCL_SUCCESS));

    rtStream_t aiCpuStream;
    Stream stream(aiCpuStream);
    implBase->isA2MC2MultiServer_ = true;
    implBase->tagStreamInfo_.insert(std::pair<std::string, Level1StreamInfo>(tag, std::move(tmpInnerStreamInfo)));
    ret = implBase->GenAiRMAInfo(implBase->tagCommInfo_[tag].commLevel0[0].get());
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = implBase->H2DAiRMAInfo(tag, aiCpuStream);
    HcclCombinOpParam* combinOparaPtr = reinterpret_cast<HcclCombinOpParam*>(implBase->combinOparaMem_->ptr());
    EXPECT_NE(combinOparaPtr->aiRMAInfo, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u64 profConfigL0 = 0x84000985;
    profilingManager.StopSubscribe(profConfigL0);

    HCCL_INFO("check if HcclCombinOpParam is match with aicpu struct HccCommResParamTask");

    implBase = nullptr;
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_clean_ccl_buffer)
{
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    HcclResult ret;
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = implBase->CreateCommCCLbuffer();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = implBase->cclBufferManager_.CleanCCLbuffer();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclImplTest, st_AivResume)
{
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    HcclResult ret;
    std::unique_ptr<HcclCommunicator> hcclCommunicator(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = hcclCommunicator->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = hcclCommunicator->cclBufferManager_.CreateCommAIVbuffer(true);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER(GetExternalInputHcclAivMode).stubs().will(returnValue(true));

    hcclCommunicator->AivResume();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hcclCommunicator->cclBufferManager_.ReleaseCommAIVbuffer();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_AllocTransport910B)
{
    HcclResult ret = HCCL_SUCCESS;

    TransportType transportType = TransportType::TRANS_TYPE_ROCE;
    MOCKER_CPP(&TransportManager::GetTransportType).stubs().will(returnValue(transportType));

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    std::unique_ptr<HcclCommunicator> communicator(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    communicator->Init(params, rankTable);

    TransportRequest transportReq1;
    transportReq1.isValid = true;
    transportReq1.remoteUserRank = 0;
    transportReq1.remoteUserRank = 1;
    transportReq1.linkType = TransportLinkType::HCCS;
    TransportRequest transportReq2;
    transportReq2.isValid = true;
    transportReq2.remoteUserRank = 0;
    transportReq2.remoteUserRank = 1;
    transportReq2.linkType = TransportLinkType::SIO;

    SingleSubCommTransport singleTrans;
    singleTrans.transportRequests.emplace_back(transportReq1);
    singleTrans.transportRequests.emplace_back(transportReq2);
    singleTrans.links.resize(2, nullptr);
    singleTrans.status.resize(2, TransportStatus::INIT);
    LevelNSubCommTransport levelTrans;
    levelTrans.emplace_back(singleTrans);
    OpCommTransport opTrans;
    opTrans.emplace_back(levelTrans);
    TransportIOMem transMem;

    MOCKER_CPP(&NotifyPool::RegisterOp).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&NotifyPool::UnregisterOp).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_910B)).will(returnValue(HCCL_SUCCESS));

    // stubs in CreateDestSockets
    MOCKER_CPP(&TransportManager::UpdateIsInterRdma)
        .stubs()
        .with(mockcpp::any(), outBound(false), mockcpp::any())
        .will(ignoreReturnValue());
    MOCKER_CPP(&TransportManager::MakeRemoteLinkInfo).stubs().will(returnValue(HCCL_SUCCESS));

    // stubs in CreateLink
    MOCKER(hrtErrMSetErrorContextPub).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER(hrtSetDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&TransportManager::TransportInit).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtResetDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclSocketManager::DestroySockets, void(HcclSocketManager::*)(const std::string&))
        .stubs()
        .with(mockcpp::any())
        .will(ignoreReturnValue());

    std::string tag = "test";
    ret = communicator->transportManager_->Alloc(tag, transMem, opTrans, true, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_AllocTransport910C)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    std::unique_ptr<HcclCommunicator> communicator(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    communicator->Init(params, rankTable);

    TransportRequest transportReq1;
    transportReq1.isValid = true;
    transportReq1.remoteUserRank = 0;
    transportReq1.remoteUserRank = 1;
    transportReq1.inputMemType = CCL_INPUT;
    transportReq1.outputMemType = CCL_OUTPUT;
    transportReq1.linkType = TransportLinkType::HCCS;
    TransportRequest transportReq2;
    transportReq2.isValid = true;
    transportReq2.remoteUserRank = 0;
    transportReq2.remoteUserRank = 1;
    transportReq2.inputMemType = CCL_INPUT;
    transportReq2.outputMemType = CCL_OUTPUT;
    transportReq2.linkType = TransportLinkType::SIO;

    SingleSubCommTransport singleTrans;
    singleTrans.transportRequests.emplace_back(transportReq1);
    singleTrans.transportRequests.emplace_back(transportReq2);
    singleTrans.links.resize(2, nullptr);
    singleTrans.status.resize(2, TransportStatus::INIT);
    TransportIOMem transMem;

    MOCKER_CPP(&TransportManager::GetIOMem).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    // stubs in CreateDestSockets
    MOCKER_CPP(&TransportManager::UpdateIsInterRdma)
        .stubs()
        .with(mockcpp::any(), outBound(false), mockcpp::any())
        .will(ignoreReturnValue());
    MOCKER_CPP(&TransportManager::MakeRemoteLinkInfo).stubs().will(returnValue(HCCL_SUCCESS));

    // stubs in IsHccsTransport
    MOCKER(hrtGetPairDeviceLinkType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(LinkTypeInServer::HCCS_SW_TYPE))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER(Is310PDevice).stubs().will(returnValue(false));
    MOCKER_CPP(&HcclSocketManager::CreateSingleLinkSocket).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    // stubs in CreateLink
    MOCKER(hrtErrMSetErrorContextPub).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER(hrtSetDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&TransportManager::TransportInit).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtResetDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&TransportManager::checkSubCommLinkThreadsStatus)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclSocketManager::DestroySockets, void(HcclSocketManager::*)(const std::string&))
        .stubs()
        .with(mockcpp::any())
        .will(ignoreReturnValue());

    std::string tag = "test";
    ret = communicator->transportManager_->AllocSubCommLinks(tag, transMem, singleTrans, false, false, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_checkSubCommLinkThreadsStatus_910C)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    std::unique_ptr<HcclCommunicator> communicator(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    communicator->Init(params, rankTable);

    TransportRequest transportReq1;
    transportReq1.isValid = true;
    transportReq1.remoteUserRank = 0;
    transportReq1.remoteUserRank = 1;
    transportReq1.inputMemType = CCL_INPUT;
    transportReq1.outputMemType = CCL_OUTPUT;
    transportReq1.linkType = TransportLinkType::HCCS;
    TransportRequest transportReq2;
    transportReq2.isValid = true;
    transportReq2.remoteUserRank = 0;
    transportReq2.remoteUserRank = 1;
    transportReq2.inputMemType = CCL_INPUT;
    transportReq2.outputMemType = CCL_OUTPUT;
    transportReq2.linkType = TransportLinkType::SIO;

    SingleSubCommTransport singleTrans;
    singleTrans.transportRequests.emplace_back(transportReq1);
    singleTrans.transportRequests.emplace_back(transportReq2);
    singleTrans.links.resize(2, nullptr);
    singleTrans.status.resize(2, TransportStatus::INIT);

    std::vector<std::pair<u32, u32>> remoteRankMap = {{0, 1}};
    struct SubCommLinkPara subCommLinkPara(singleTrans, remoteRankMap, 0, 1);

    MOCKER_CPP(&NotifyPool::UnregisterOp).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    std::string tag = "test";
    ret = communicator->transportManager_->checkSubCommLinkThreadsStatus(tag, subCommLinkPara, false);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_TransportMgrConstructTransTag)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    std::unique_ptr<HcclCommunicator> communicator(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    communicator->Init(params, rankTable);

    MOCKER(Is310PDevice).stubs().will(returnValue(false));
    communicator->transportManager_->isHaveCpuRank_ = true;

    std::string tag = "testAlg";
    std::string transTag;
    ret = communicator->transportManager_->ConstructTransTag(tag, transTag, true, 0, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_STREQ(transTag.c_str(), "testAlg_Inter_");

    ret = communicator->transportManager_->ConstructTransTag(tag, transTag, false, 0, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_STREQ(transTag.c_str(), "testAlg_SIO_");

    ret = communicator->transportManager_->ConstructTransTag(tag, transTag, false, 0, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_STREQ(transTag.c_str(), "testAlg_Hccs_");

    bool ret1 = communicator->transportManager_->IsHccsTransport(0, TransportLinkType::HCCS);
    EXPECT_EQ(ret1, true);
    ret1 = communicator->transportManager_->IsHccsTransport(0, TransportLinkType::SIO);
    EXPECT_EQ(ret1, false);

    MOCKER(hrtGetPairDeviceLinkType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(LinkTypeInServer::SIO_TYPE))
        .will(returnValue(HCCL_SUCCESS));
    ret1 = communicator->transportManager_->IsHccsTransport(0, TransportLinkType::RESERVED);
    EXPECT_EQ(ret1, false);
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_TransportMgrExceptionHandle)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    std::unique_ptr<HcclCommunicator> communicator(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    communicator->Init(params, rankTable);

    TransportRequest transportReq1;
    transportReq1.isValid = true;
    transportReq1.remoteUserRank = 0;
    transportReq1.remoteUserRank = 1;
    transportReq1.linkType = TransportLinkType::HCCS;

    SingleSubCommTransport singleTrans;
    singleTrans.transportRequests.emplace_back(transportReq1);
    singleTrans.links.resize(2, nullptr);
    singleTrans.status.resize(2, TransportStatus::INIT);
    LevelNSubCommTransport levelTrans;
    levelTrans.emplace_back(singleTrans);
    OpCommTransport opTrans;
    opTrans.emplace_back(levelTrans);

    MOCKER_CPP(&TransportManager::UpdateIsInterRdma)
        .stubs()
        .with(mockcpp::any(), outBound(false), mockcpp::any())
        .will(ignoreReturnValue());
    MOCKER_CPP(&TransportManager::MakeRemoteLinkInfo).stubs().will(returnValue(HCCL_SUCCESS));

    // stubs in IsHccsTransport
    MOCKER(hrtGetPairDeviceLinkType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(LinkTypeInServer::HCCS_SW_TYPE))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER(Is310PDevice).stubs().will(returnValue(false));
    MOCKER_CPP(
        &HcclSocketManager::AddWhiteList,
        HcclResult(HcclSocketManager::*)(const std::string&, const HcclNetDevCtx, HcclRankLinkInfo))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    std::string tag("test");
    ret = communicator->transportManager_->ExceptionHandle(tag, opTrans);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_TransportMgrSetMachinePara)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    std::unique_ptr<HcclCommunicator> communicator(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    communicator->Init(params, rankTable);

    std::string tag("testSetMachinePara");
    MachineType machineType = MachineType::MACHINE_CLIENT_TYPE;
    std::string serverId("testServer_id");
    std::vector<std::shared_ptr<HcclSocket>> socketList;
    DeviceMem inputMem;
    DeviceMem outputMem;
    DeviceMem expMem;
    MachinePara machinePara;
    RankInfo loaclRankInfo;
    RankInfo remoteRankInfo;

    ret = communicator->transportManager_->SetMachinePara(
        tag, machineType, serverId, 1, true, LinkMode::LINK_DUPLEX_MODE, socketList, inputMem, outputMem, expMem, false,
        false, false, 1, 0, 1, machinePara, loaclRankInfo, remoteRankInfo, TransportLinkType::RESERVED);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(machinePara.specifyLink, LinkTypeInServer::RESERVED_LINK_TYPE);

    ret = communicator->transportManager_->SetMachinePara(
        tag, machineType, serverId, 1, true, LinkMode::LINK_DUPLEX_MODE, socketList, inputMem, outputMem, expMem, false,
        false, false, 1, 0, 1, machinePara, loaclRankInfo, remoteRankInfo, TransportLinkType::HCCS);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(machinePara.specifyLink, LinkTypeInServer::HCCS_SW_TYPE);

    ret = communicator->transportManager_->SetMachinePara(
        tag, machineType, serverId, 1, true, LinkMode::LINK_DUPLEX_MODE, socketList, inputMem, outputMem, expMem, false,
        false, false, 1, 0, 1, machinePara, loaclRankInfo, remoteRankInfo, TransportLinkType::SIO);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(machinePara.specifyLink, LinkTypeInServer::SIO_TYPE);
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_update_zerocopy)
{
    HcclResult ret = HCCL_SUCCESS;
    MOCKER(GetWorkflowMode).stubs().will(returnValue(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&StreamActiveManager::StreamActive).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(GetExternalInputHcclEnableFfts).stubs().will(returnValue(false));
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam_1server_8p(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910_93;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    ret = implBase->AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->InitCCLbuffer(4096, 4096);
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    HcclIpAddress serverIp("172.17.10.1");
    HcclIpAddress localIp("172.17.10.1");
    SetRetryEnable(
        DevType::DEV_TYPE_910, 1, 1, 2, true, implBase->GetAivModeConfig(), serverIp, localIp, implBase->retryEnable_);
    SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    impl->topoAttr_.deviceLogicId = 0;
    impl->topoAttr_.devicePhyId = 0;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_8P_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    impl->topoType_ = TopoType::TOPO_TYPE_8P_RING;
    algConfigurator->topoType_ = TopoType::TOPO_TYPE_8P_RING;
    CollAllGatherRingExecutor* executor = new CollAllGatherRingExecutor(impl->dispatcher_, topoMatcher);
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 64 * 1024 * 1024;
    opParam.DataDes.count = 1024;
    opParam.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);
    opParam.aicpuUnfoldMode = true;
    opParam.isZeroCopy = true;
    const std::vector<std::vector<u32>> tmpRingNics = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}};
    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(
        opParam.tag, HcclCMDType::HCCL_CMD_ALLGATHER, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    resourceResponse.slaveStreams.resize(3);
    resourceResponse.notifiesMain.resize(3);
    resourceResponse.notifiesAux.resize(3);
    resourceResponse.notifiesDevMain.resize(3);
    resourceResponse.notifiesDevAux.resize(3);

    resourceResponse.threadManage.resize(3);
    for (u32 ringIndex = 0; ringIndex < 3; ringIndex++) {
        resourceResponse.threadManage[ringIndex].reset(new (std::nothrow)
                                                           ThreadManage(ringIndex, ringIndex, impl->dispatcher_));
    }

    for (auto& singleSubCommTransport : resourceResponse.opTransportResponse[COMM_LEVEL0]) {
        for (u64 i = 0; i < singleSubCommTransport.links.size(); ++i) {
            singleSubCommTransport.transportRequests[i].isValid = true;
            MachinePara linkPara;
            std::chrono::milliseconds kdefaultTimeout = std::chrono::seconds(120);
            std::shared_ptr<Transport> link(
                new Transport(new (std::nothrow) TransportBase(nullptr, nullptr, linkPara, kdefaultTimeout)));
            singleSubCommTransport.links[i] = link;
        }
    }

    implBase->UpdateZeroCopy(opParam, resourceResponse);

    delete executor;

    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_prepare_zerocopy_algname)
{
    HcclResult ret = HCCL_SUCCESS;
    MOCKER(GetWorkflowMode).stubs().will(returnValue(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&StreamActiveManager::StreamActive).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(GetExternalInputHcclEnableFfts).stubs().will(returnValue(false));
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam_1server_8p(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910_93;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    ret = implBase->AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->InitCCLbuffer(4096, 4096);
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    HcclIpAddress serverIp("172.17.10.1");
    HcclIpAddress localIp("172.17.10.1");
    SetRetryEnable(
        DevType::DEV_TYPE_910, 1, 1, 2, true, implBase->GetAivModeConfig(), serverIp, localIp, implBase->retryEnable_);
    SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    impl->topoAttr_.deviceLogicId = 0;
    impl->topoAttr_.devicePhyId = 0;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_8P_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    impl->topoType_ = TopoType::TOPO_TYPE_8P_RING;
    algConfigurator->topoType_ = TopoType::TOPO_TYPE_8P_RING;
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 64 * 1024 * 1024;
    opParam.DataDes.count = 1024;
    opParam.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);
    opParam.aicpuUnfoldMode = true;
    const std::vector<std::vector<u32>> tmpRingNics = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}};

    bool isSupportZeroCopy = implBase->IsSupportZeroCopy(opParam);
    AlgDesc algDesc;
    algDesc.isZeroCopy = true;
    implBase->PrepareZeroCopy("algName", algDesc, opParam);

    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_prepare_zerocopy_op)
{
    HcclResult ret = HCCL_SUCCESS;
    MOCKER(GetWorkflowMode).stubs().will(returnValue(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&StreamActiveManager::StreamActive).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(GetExternalInputHcclEnableFfts).stubs().will(returnValue(false));
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam_1server_8p(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910_93;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    ret = implBase->AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->InitCCLbuffer(4096, 4096);
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    HcclIpAddress serverIp("172.17.10.1");
    HcclIpAddress localIp("172.17.10.1");
    SetRetryEnable(
        DevType::DEV_TYPE_910, 1, 1, 2, true, implBase->GetAivModeConfig(), serverIp, localIp, implBase->retryEnable_);
    SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    impl->topoAttr_.deviceLogicId = 0;
    impl->topoAttr_.devicePhyId = 0;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_8P_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_RING;
    impl->topoType_ = TopoType::TOPO_TYPE_8P_RING;
    algConfigurator->topoType_ = TopoType::TOPO_TYPE_8P_RING;
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 64 * 1024 * 1024;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 64 * 1024 * 1024;
    opParam.DataDes.count = 1024;
    opParam.DataDes.dataType = HCCL_DATA_TYPE_INT64;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);
    opParam.aicpuUnfoldMode = true;
    const std::vector<std::vector<u32>> tmpRingNics = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}};

    bool isSupportZeroCopy = implBase->IsSupportZeroCopy(opParam);
    AlgDesc algDesc;
    algDesc.isZeroCopy = false;
    implBase->PrepareZeroCopy("algName", algDesc, opParam);

    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_AllocTransportZeroCopy)
{
    HcclResult ret = HCCL_SUCCESS;

    TransportType transportType = TransportType::TRANS_TYPE_ROCE;
    MOCKER_CPP(&TransportManager::GetTransportType).stubs().will(returnValue(transportType));

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910_93;
    std::unique_ptr<HcclCommunicator> communicator(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    communicator->Init(params, rankTable);

    TransportRequest transportReq1;
    transportReq1.isValid = true;
    transportReq1.remoteUserRank = 0;
    transportReq1.remoteUserRank = 1;
    transportReq1.linkType = TransportLinkType::HCCS;
    transportReq1.inputMemType = TransportMemType::CCL_INPUT;
    transportReq1.outputMemType = TransportMemType::CCL_OUTPUT;
    TransportRequest transportReq2;
    transportReq2.isValid = true;
    transportReq2.remoteUserRank = 0;
    transportReq2.remoteUserRank = 1;
    transportReq2.linkType = TransportLinkType::SIO;
    transportReq2.inputMemType = TransportMemType::CCL_INPUT;
    transportReq2.outputMemType = TransportMemType::CCL_OUTPUT;

    SingleSubCommTransport singleTrans;
    singleTrans.transportRequests.emplace_back(transportReq1);
    singleTrans.transportRequests.emplace_back(transportReq2);
    singleTrans.links.resize(2, nullptr);
    singleTrans.status.resize(2, TransportStatus::INIT);
    LevelNSubCommTransport levelTrans;
    levelTrans.emplace_back(singleTrans);
    OpCommTransport opTrans;
    opTrans.emplace_back(levelTrans);
    TransportIOMem transMem;

    MOCKER_CPP(&NotifyPool::RegisterOp).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&NotifyPool::UnregisterOp).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_910_93)).will(returnValue(HCCL_SUCCESS));

    // stubs in CreateDestSockets
    MOCKER_CPP(&TransportManager::UpdateIsInterRdma)
        .stubs()
        .with(mockcpp::any(), outBound(false), mockcpp::any())
        .will(ignoreReturnValue());
    MOCKER_CPP(&TransportManager::MakeRemoteLinkInfo).stubs().will(returnValue(HCCL_SUCCESS));

    // stubs in CreateLink
    MOCKER(hrtErrMSetErrorContextPub).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER(hrtSetDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&TransportManager::TransportInit).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtResetDevice).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclSocketManager::DestroySockets, void(HcclSocketManager::*)(const std::string&))
        .stubs()
        .with(mockcpp::any())
        .will(ignoreReturnValue());

    std::string tag = "test";
    ret = communicator->transportManager_->Alloc(
        tag, transMem, opTrans, true, true, true, HcclCMDType::HCCL_CMD_ALLGATHER);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_zerocopy_alloc_slave_empty)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.totalRanks = 1;
    params.deviceType = DevType::DEV_TYPE_910;
    rankTable.rankList.assign(rankTable.rankList.begin(), rankTable.rankList.begin() + 1);
    rankTable.deviceNum = 1;
    rankTable.serverNum = 1;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER(GetWorkflowMode).stubs().will(returnValue(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB));
    MOCKER_CPP(&HcclCommunicator::IsForceAicpuOpBaseMode).stubs().will(returnValue(false));
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&OpBaseStreamManager::AllocSlaves).stubs().will(returnValue(std::vector<Stream>()));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    CollAllGatherRingExecutor* executor = new CollAllGatherRingExecutor(impl->dispatcher_, topoMatcher);
    executor->workflowMode_ = HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB;
    executor->SetAlgType(AlgType());

    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    OpParam opParam;
    opParam.tag = "test_test_test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 4096;
    opParam.DataDes.count = 1024;
    opParam.DataDes.dataType = HCCL_DATA_TYPE_INT64;
    opParam.reduceType = HCCL_REDUCE_SUM;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);
    opParam.aicpuUnfoldMode = true;
    opParam.isZeroCopy = true;

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    resourceRequest.scratchMemSize = 4096;
    resourceRequest.isInGraphCaptureZeroCopy = true;
    resourceRequest.streamNum = 2;
    implBase->AllocAlgResource(
        opParam.tag, HcclCMDType::HCCL_CMD_ALLGATHER, opParam, resourceRequest, resourceResponse);

    delete executor;
    GlobalMockObject::verify();
}
#if 0
TEST_F(HcclImplTest, ut_Mc2CreateAndLaunchContext_multi_server)
{
    HcclResult ret = HCCL_SUCCESS;
    std::unique_ptr<HcclCommunicator> hcclCommunicator(new (std::nothrow) HcclCommunicator());

    ret = InitEnvVarParam();

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.totalRanks = 1;
    params.deviceType = DevType::DEV_TYPE_910;
    rankTable.rankList.assign(rankTable.rankList.begin(), rankTable.rankList.begin() + 1);
    rankTable.deviceNum = 1;
    rankTable.serverNum = 1;

    MOCKER(GetWorkflowMode)
    .stubs()
    .will(returnValue(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB));
    MOCKER_CPP(&HcclCommunicator::IsForceAicpuOpBaseMode)
    .stubs()
    .will(returnValue(false));
    MOCKER_CPP(&HcclCommunicator::InitRaResource)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&OpBaseStreamManager::AllocSlaves)
    .stubs()
    .will(returnValue(std::vector<Stream>()));

    MOCKER(hrtMemAsyncCopy)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    ret = hcclCommunicator->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclCommunicator->identifier_ = "1";
    hcclCommunicator->isA2MC2MultiServer_ = true;
    constexpr u32 h2dBufferSize = sizeof(KfcExecControl);
    constexpr u32 d2hBufferSize = sizeof(KfcExecStatus);
    hcclCommunicator->kfcControlTransferH2D_.reset(new (std::nothrow)
                                                       hccl::HDCommunicate(0, HCCL_HDC_TYPE_H2D, h2dBufferSize));
    hcclCommunicator->kfcStatusTransferD2H_.reset(new (std::nothrow)
                                                      hccl::HDCommunicate(0, HCCL_HDC_TYPE_D2H, d2hBufferSize));

    rtStream_t stream = (rtStream_t)NULL;
    bool isOpbaseMode = false;
    void* commContext = nullptr;
    std::string tag = "tag";

    MOCKER_CPP(&HcclCommunicator::InitWorkSpace)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    u32 ifnumVersion = 3;
    MOCKER(hrtRaGetInterfaceVersion)
    .stubs()
    .with(mockcpp::any(), mockcpp::any(), outBoundP(&ifnumVersion))
    .will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemSyncCopy)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtAicpuKernelLaunchExWithArgs).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = hcclCommunicator->Mc2CreateAndLaunchContext(stream, isOpbaseMode, &commContext, tag);

    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclCommunicator->kfcControlTransferH2D_ = nullptr;
    hcclCommunicator->kfcStatusTransferD2H_ = nullptr;

    GlobalMockObject::verify();
}
#endif
TEST_F(HcclImplTest, ut_hcclimpl_AiCpuSetCommResource_EnvTest)
{
    u32 ifnumVersion = 3;
    MOCKER(hrtRaGetInterfaceVersion)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&ifnumVersion))
        .will(returnValue(HCCL_SUCCESS));

    void* commInputPtr = nullptr;
    u64 commInputSize;
    void* commOutputPtr = nullptr;
    u64 commOutputSize = 0;
    HcclResult ret;
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    std::string tag = "test_MC2MultiServer";
    CommInfo tmpComm;
    std::vector<RankInfo> paraVector;
    u32 rankSize = 8;
    u32 curRankId = 0;
    u64 commBufferSize = 20;

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclImpl* impl = implBase->implAlg_->pimpl_.get();

    ret = implBase->GetInCCLbuffer(commInputPtr, commInputSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = implBase->GetOutCCLbuffer(commOutputPtr, commOutputSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    DeviceMem expMem = implBase->cclBufferManager_.GetCommExpBuffer();

    implBase->notifyPool_.reset(new (std::nothrow) NotifyPool());
    ret = implBase->notifyPool_->Init(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    IntraExchanger exchanger{};
    tmpComm.commLevel0.resize(4);
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;
    for (int i = 0; i < 4; i++) {
        tmpComm.commLevel0[i].reset(new (std::nothrow) CommMesh(
            tag, 0, 4, curRankId, rankSize, TopoType::TOPO_TYPE_8P_MESH, implBase->dispatcher_, implBase->notifyPool_,
            netDevCtxMap, exchanger, paraVector, inputMem, outputMem, true, ""));
    }
    EXPECT_EQ(tmpComm.commLevel0[0]->transportInfo_.size(), rankSize);
    tmpComm.commLevel0[0]->transportInfo_.clear();
    std::chrono::milliseconds kdefaultTimeout = std::chrono::seconds(120);
    /*创建link*/
    MachinePara machine_para;

    machine_para.localDeviceId = 0;
    machine_para.deviceLogicId = 0;
    machine_para.nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE;
    machine_para.localIpAddr = HcclIpAddress("192.168.0.23");
    std::shared_ptr<Transport> link = nullptr;
    TransportPara para = {};

    link.reset(new Transport(TransportType::TRANS_TYPE_IBV_EXP, para, dispatcher, implBase->notifyPool_, machine_para));
    link->Init();
    ret = implBase->notifyPool_->RegisterOp(tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::shared_ptr<LocalIpcNotify> localNotify = nullptr;
    std::shared_ptr<RemoteNotify> remoteNotify = nullptr;

    RemoteRankInfo info(0, -1);
    SalGetBareTgid(&info.remotePid);
    ret = implBase->notifyPool_->Alloc(tag, info, localNotify, NotifyLoadType::DEVICE_NOTIFY);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<u8> data(NOTIFY_INFO_LENGTH, 0);
    ret = localNotify->Serialize(data);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remoteNotify.reset(new (std::nothrow) RemoteNotify());

    ret = remoteNotify->Init(data);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = remoteNotify->Open();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    link->pimpl_->remoteSendDoneDeviceNotify_ = remoteNotify;
    link->pimpl_->localSendDoneDeviceNotify_ = localNotify;
    link->pimpl_->remoteSendReadyDeviceNotify_ = remoteNotify;
    link->pimpl_->localSendReadyDeviceNotify_ = localNotify;
    for (u32 ringIndex = 0; ringIndex < rankSize; ringIndex++) {
        tmpComm.commLevel0[0]->transportInfo_.push_back(link);
        tmpComm.commLevel0[0]->transportType_.push_back(TransportType::TRANS_TYPE_IBV_EXP);
    }
    implBase->tagCommInfo_.insert(std::pair<std::string, CommInfo>(tag, std::move(tmpComm)));

    Level1StreamInfo tmpInnerStreamInfo;
    tmpInnerStreamInfo.ringNum = rankSize;
    tmpInnerStreamInfo.ringDeviceSignal.resize(rankSize - 1);
    tmpInnerStreamInfo.ringDeviceSignalAux.resize(rankSize - 1);
    tmpInnerStreamInfo.ringDeviceStreams.resize(rankSize);

    for (u32 ringIndex = 0; ringIndex < tmpInnerStreamInfo.ringNum; ringIndex++) {
        tmpInnerStreamInfo.ringDeviceStreams[ringIndex] = Stream(StreamType::STREAM_TYPE_DEVICE);
        if (ringIndex != tmpInnerStreamInfo.ringNum - 1) {
            tmpInnerStreamInfo.ringDeviceSignal[ringIndex] = localNotify;
            tmpInnerStreamInfo.ringDeviceSignalAux[ringIndex] = localNotify;
        }
    }

    MOCKER_CPP(&Transport::GetRemoteMem, HcclResult(Transport::*)(hccl::UserMemType, void**))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&Transport::GetRemoteMem, HcclResult(Transport::*)(std::vector<void*>*))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&Transport::GetRemoteMemKey, HcclResult(Transport::*)(hccl::UserMemType, uint32_t*))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&Transport::GetLocalMemDetails).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    std::vector<HcclQpInfoV2> qpInfos(1);
    MOCKER_CPP(&Transport::GetAiQpInfo).stubs().with(outBound(qpInfos)).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&Transport::GetChipId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    // 配置profiling开关
    auto& profilingManager = hccl::ProfilingManager::Instance();
    profilingManager.StartAdditionInfoSubscribe();

    // MachinePara machinePara;
    machine_para.localDeviceId = 0;
    std::chrono::milliseconds timeout;

    HcclIpAddress remoteIp{};
    HcclIpAddress localIp{};
    std::shared_ptr<HcclSocket> newSocket(
        new (std::nothrow) HcclSocket("test", nullptr, remoteIp, 0, HcclSocketRole::SOCKET_ROLE_SERVER));
    machine_para.sockets.push_back(newSocket);

    std::unique_ptr<NotifyPool> notifyPool = nullptr;
    notifyPool.reset(new (std::nothrow) NotifyPool());
    EXPECT_NE(notifyPool, nullptr);
    ret = notifyPool->Init(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    TransportBase transportBase(dispatcher, notifyPool, machine_para, timeout);
    MOCKER_CPP_VIRTUAL(transportBase, &TransportBase::GetRemoteMemSize)
        .stubs()
        .with(mockcpp::any(), outBound(commBufferSize))
        .will(returnValue(HCCL_SUCCESS));

    rtStream_t aiCpuStream;
    Stream stream(aiCpuStream);
    implBase->isA2MC2MultiServer_ = true;
    implBase->tagStreamInfo_.insert(std::pair<std::string, Level1StreamInfo>(tag, std::move(tmpInnerStreamInfo)));

    u32 intraRoceSwitch = 0;
    MOCKER(GetExternalInputIntraRoceSwitch).stubs().will(returnValue(intraRoceSwitch));

    MOCKER_CPP(&HcclCommunicator::InitNic).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::CreateCommAndStreamRes).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::Mc2CreateAndLaunchContext)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    ret = implBase->CreateCommResource(tag, aiCpuStream, true, nullptr, "BatchWrite=level1:hierarchy");
    EXPECT_EQ(ret, HCCL_SUCCESS);

    implBase = nullptr;
    GlobalMockObject::verify();
}

TEST_F(HcclImplTest, ut_GetRemoteUserMemResource_When_Link_Normal_Expect_ReturnSuccess)
{
    HcclResult ret = HCCL_SUCCESS;
    std::unique_ptr<HcclCommunicator> hcclCommunicator(new (std::nothrow) HcclCommunicator());
    // 构建window resource
    OpCommTransport userMemTransport;
    LevelNSubCommTransport levelNTransport;
    SingleSubCommTransport singleTransport;
    // transport
    std::vector<TransportRequest> transportRequests;
    TransportRequest transportRequest;
    transportRequest.isValid = true;
    transportRequests.push_back(transportRequest);
    singleTransport.transportRequests = transportRequests;
    // links
    std::vector<LINK> links;
    MachinePara machinePara;
    machinePara.remoteWorldRank = 0;
    std::chrono::milliseconds kdefaultTimeout = std::chrono::seconds(120);
    DeviceMem window = DeviceMem::alloc(128);
    auto transportP2p = new (std::nothrow) TransportP2p(nullptr, nullptr, machinePara, kdefaultTimeout);
    transportP2p->remoteInputPtr_ = window.ptr();
    transportP2p->remoteOutputPtr_ = window.ptr();
    transportP2p->remoteInputSize_ = 128;
    transportP2p->remoteOutputSize_ = 128;
    std::shared_ptr<Transport> link(new Transport(transportP2p));
    links.push_back(link);
    singleTransport.links = links;
    levelNTransport.push_back(singleTransport);
    userMemTransport.push_back(levelNTransport);
    hcclCommunicator->userMemTransport_ = userMemTransport;
    // exec
    ret = hcclCommunicator->GetRemoteUserMemResource();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    window.free();
    GlobalMockObject::verify();
}
#if 0 // coredump
TEST_F(HcclImplTest, ut_HcclCommunicator_AicpuUnfold_and_AllReduceAicpuUnfold)
{
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclRtStream stream = NULL;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    HcclCMDType cmdType =  HcclCMDType::HCCL_CMD_ALLREDUCE;
    s32 device_id = 0;
    // 申请device memory
    DeviceMem mem_dev_input = DeviceMem::alloc(1024);
    EXPECT_NE(mem_dev_input.ptr(), nullptr);
    DeviceMem mem_dev_output = DeviceMem::alloc(1024);
    EXPECT_NE(mem_dev_output.ptr(), nullptr);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::InitPara)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::InitOneSidedService)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER_CPP(&HcclCommunicator::IsExistCommRes)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(true));

    MOCKER_CPP(&HcclCommunicator::AicpuKfcTilingDataLaunch)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    ret = implBase->AicpuUnfold("tag_test", const_cast<void*>(mem_dev_input.ptr()),
        const_cast<void*>(mem_dev_output.ptr()), count, dataType, HCCL_REDUCE_SUM, stream, cmdType);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = implBase->AllReduceAicpuUnfold("tag_test", const_cast<void*>(mem_dev_input.ptr()),
        const_cast<void*>(mem_dev_output.ptr()), count, dataType, HCCL_REDUCE_SUM, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    aclError rt_ret = ACL_SUCCESS;
    unsetenv("HCCL_OP_EXPANSION_MODE");
    rt_ret = aclrtFree(const_cast<void*>(mem_dev_input.ptr()));
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    rt_ret = aclrtFree(const_cast<void*>(mem_dev_output.ptr()));
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    // 销毁资源

    GlobalMockObject::verify();
}
#endif

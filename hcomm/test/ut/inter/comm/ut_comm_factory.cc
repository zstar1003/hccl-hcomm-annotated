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
#include <cstdio>

#include "hccl/base.h"
#include <hccl/hccl_types.h>
#include "p2p_mgmt_pub.h"
#include "dltdt_function.h"
#include "dlra_function.h"
#include "dlhal_function.h"

#include "sal.h"
#define private public
#define protected public
#include "comm_factory.h"
#include "network_manager_pub.h"
#undef private
#undef protected
#include "llt_hccl_stub_sal_pub.h"
#include "llt_hccl_stub_gdr.h"

#include <iostream>
#include <fstream>
#include "profiler_manager.h"

using namespace std;
using namespace hccl;

constexpr u32 MESH_AGGREGATION_RANK_SIZE_910 = 4; // 4p mesh
class CommFactoryTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        DlRaFunction::GetInstance().DlRaFunctionInit();
        DlTdtFunction::GetInstance().DlTdtFunctionInit();
        DlHalFunction::GetInstance().DlHalFunctionInit();

        s32 ret = HcclDispatcherInit(DispatcherType::DISPATCHER_NORMAL, 0, &dispatcherPtr);
        if (ret != HCCL_SUCCESS)
            return;
        if (dispatcherPtr == nullptr)
            return;
        dispatcher = reinterpret_cast<DispatcherPub*>(dispatcherPtr);
        std::cout << "\033[36m--CommFactoryTest SetUP--\033[0m" << std::endl;
    }
    static void TearDownTestCase()
    {
        if (dispatcherPtr != nullptr) {
            s32 ret = HcclDispatcherDestroy(dispatcherPtr);
            EXPECT_EQ(ret, HCCL_SUCCESS);
            dispatcherPtr = nullptr;
            dispatcher = nullptr;
        }
        std::cout << "\033[36m--CommFactoryTest TearDown--\033[0m" << std::endl;
    }
    virtual void SetUp()
    {
        TsdOpen(1, 2);
        s32 portNum = -1;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        TsdClose(1);
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
    static HcclDispatcher dispatcherPtr;
    static DispatcherPub* dispatcher;
};
HcclDispatcher CommFactoryTest::dispatcherPtr = nullptr;
DispatcherPub* CommFactoryTest::dispatcher = nullptr;

void get_rank_vector(std::vector<RankInfo>& rank_vector, u32 rank_size)
{
    std::string baseIp = "192.168.0.";
    u8 offset = 11;

    for (int i = 0; i < rank_size; i++) {
        RankInfo tmp_para;
        std::string ipStr = baseIp + std::to_string(offset + i);
        tmp_para.userRank = static_cast<u32>(i);
        tmp_para.devicePhyId = static_cast<u32>(i);
        tmp_para.deviceType = DevType::DEV_TYPE_910;
        tmp_para.serverIdx = 0;
        tmp_para.serverId = "10.0.0.10";
        tmp_para.nicIp.push_back(HcclIpAddress(ipStr));
        tmp_para.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

        rank_vector.push_back(tmp_para);
    }

    return;
}

TEST_F(CommFactoryTest, ut_init)
{
    s32 ret = HCCL_SUCCESS;

    u32 userRank = 0;
    u32 user_rank_size = 8;

    char collectiveId[SAL_UNIQUE_ID_BYTES];
    ret = SalGetUniqueId(collectiveId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string collective_id_tmp = collectiveId;
    // set device
    s32 device_id = 0;
    ret = hrtSetDevice(device_id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> rank_vector;
    get_rank_vector(rank_vector, user_rank_size);

    std::shared_ptr<CommFactory> comm_factory = nullptr;
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;

    HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, false);
    HcclNetDevCtx nicPortCtx[2];
    HcclNetOpenDev(&nicPortCtx[0], NicType::DEVICE_NIC_TYPE, 0, 0, rank_vector[userRank].nicIp[0]);
    netDevCtxMap.insert(std::make_pair(rank_vector[userRank].nicIp[0], nicPortCtx[0]));

    std::shared_ptr<TopoInfoExtractor> topoInfoExt;
    topoInfoExt.reset(new TopoInfoExtractor(
        collective_id_tmp, userRank, user_rank_size, TopoType::TOPO_TYPE_4P_MESH, DevType::DEV_TYPE_910, rank_vector));

    comm_factory.reset(new CommFactory(
        collective_id_tmp, userRank, user_rank_size, dispatcher, nullptr, netDevCtxMap, topoInfoExt, true,
        TopoType::TOPO_TYPE_4P_MESH, DevType::DEV_TYPE_910, rank_vector));

    ret = comm_factory->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclNetCloseDev(nicPortCtx[0]);
    netDevCtxMap.clear();
    HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0);
}

TEST_F(CommFactoryTest, ut_create_comm)
{
    s32 ret = HCCL_SUCCESS;

    u32 userRank = 0;
    u32 user_rank_size = 1;

    char collectiveId[SAL_UNIQUE_ID_BYTES];
    ret = SalGetUniqueId(collectiveId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string collective_id_tmp = collectiveId;
    MOCKER(hrtRaGetInterfaceVersion).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));

    std::vector<RankInfo> rank_vector;
    RankInfo tmp_para_0;
    tmp_para_0.userRank = 0;
    tmp_para_0.devicePhyId = 0;
    tmp_para_0.deviceType = DevType::DEV_TYPE_910;
    tmp_para_0.serverIdx = 0;
    tmp_para_0.serverId = "10.0.0.10";
    tmp_para_0.nicIp.push_back(HcclIpAddress("192.168.0.18"));
    tmp_para_0.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    rank_vector.push_back(tmp_para_0);

    HcclIpAddress localIPs = HcclIpAddress("192.168.0.18");
    ret = HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclNetDevCtx portCtx;
    ret = HcclNetOpenDev(&portCtx, NicType::DEVICE_NIC_TYPE, 0, 0, localIPs);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<HcclSocketManager> socketManager = nullptr;
    socketManager.reset(new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));
    ret = socketManager->ServerInit(portCtx, 16666);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;
    netDevCtxMap.insert(make_pair(localIPs, portCtx));

    std::shared_ptr<TopoInfoExtractor> topoInfoExt;
    topoInfoExt.reset(new TopoInfoExtractor(
        collective_id_tmp, userRank, user_rank_size, TopoType::TOPO_TYPE_COMMON, DevType::DEV_TYPE_910, rank_vector));

    CommFactory* comm_factory = new CommFactory(
        collective_id_tmp, userRank, user_rank_size, dispatcher, nullptr, netDevCtxMap, topoInfoExt, true,
        TopoType::TOPO_TYPE_COMMON, DevType::DEV_TYPE_910, rank_vector);

    ret = comm_factory->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    s32 mem_size = 256;
    DeviceMem inputMem = DeviceMem::alloc(mem_size);
    DeviceMem outputMem = DeviceMem::alloc(mem_size);
    DeviceMem expMem = DeviceMem::alloc(mem_size);
    const string strTag = "test_tag";

    CommParaInfo commParaInfo;
    std::vector<std::unique_ptr<CommBase>> commVec;

    commParaInfo = CommParaInfo(COMM_LEVEL1, CommType::COMM_TAG_RING_INNER);
    ret = comm_factory->CreateCommPlane(strTag, inputMem, outputMem, commParaInfo, commVec);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    commParaInfo = CommParaInfo(COMM_COMBINE, CommType::COMM_TAG_RING_COMBINED);
    ret = comm_factory->CreateCommPlane(strTag, inputMem, outputMem, commParaInfo, commVec);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    commParaInfo = CommParaInfo(COMM_COMBINE, CommType::COMM_TAG_WHOLE_NHR);
    ret = comm_factory->CreateCommPlane(strTag, inputMem, outputMem, commParaInfo, commVec);
    EXPECT_EQ(ret, HCCL_E_PARA);

    commParaInfo = CommParaInfo(COMM_COMBINE, CommType::COMM_TAG_WHOLE_NHR_V1);
    ret = comm_factory->CreateCommPlane(strTag, inputMem, outputMem, commParaInfo, commVec);
    EXPECT_EQ(ret, HCCL_E_PARA);

    commParaInfo = CommParaInfo(COMM_LEVEL1, CommType::COMM_TAG_HALVING_DOUBLING);
    ret = comm_factory->CreateCommPlane(strTag, inputMem, outputMem, commParaInfo, commVec);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    commParaInfo = CommParaInfo(COMM_LEVEL1, CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING);
    ret = comm_factory->CreateCommPlane(strTag, inputMem, outputMem, commParaInfo, commVec);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    commParaInfo = CommParaInfo(COMM_LEVEL1, CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING_V1);
    ret = comm_factory->CreateCommPlane(strTag, inputMem, outputMem, commParaInfo, commVec);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    commParaInfo = CommParaInfo(COMM_COMBINE, CommType::COMM_TAG_WHOLE_NB);
    ret = comm_factory->CreateCommPlane(strTag, inputMem, outputMem, commParaInfo, commVec);
    EXPECT_EQ(ret, HCCL_E_PARA);

    commParaInfo = CommParaInfo(COMM_LEVEL1, CommType::COMM_TAG_NONUNIFORM_BRUCK);
    ret = comm_factory->CreateCommPlane(strTag, inputMem, outputMem, commParaInfo, commVec);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    commParaInfo = CommParaInfo(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    ret = comm_factory->CreateCommPlane(strTag, inputMem, outputMem, commParaInfo, commVec);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    commParaInfo = CommParaInfo(COMM_COMBINE_ORDER, CommType::COMM_TAG_MESH_COMBINED);
    ret = comm_factory->CreateCommPlane(strTag, inputMem, outputMem, commParaInfo, commVec);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER_CPP(&CommBase::IsSupportMC2).stubs().with(mockcpp::any()).will(returnValue(2));

    commParaInfo = CommParaInfo(COMM_COMBINE_ORDER, CommType::COMM_TAG_MESH_COMBINED);
    std::vector<std::vector<RankInfo>> commPlaneVec;
    commPlaneVec.push_back(rank_vector);
    ret = comm_factory->CreateCommMesh(strTag, inputMem, outputMem, commParaInfo, commPlaneVec, false, commVec, expMem);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete comm_factory;

    socketManager->ServerDeInit(portCtx, 0);

    HcclNetCloseDev(portCtx);
    HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0);
}

TEST_F(CommFactoryTest, ut_create_comm_ranksize_7)
{
    s32 ret = HCCL_SUCCESS;

    char collectiveId[SAL_UNIQUE_ID_BYTES];
    ret = SalGetUniqueId(collectiveId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string collective_id_tmp = collectiveId;

    u32 user_rank_size = 7;
    std::vector<RankInfo> rank_vector;
    // ranksize 7 rank 2
    u32 userRank = 2;
    get_rank_vector(rank_vector, user_rank_size);
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;

    HcclNetInit(
        NICDeployment::NIC_DEPLOYMENT_DEVICE, rank_vector[userRank].devicePhyId, rank_vector[userRank].devicePhyId,
        false);
    HcclNetDevCtx nicPortCtx[2];
    HcclNetOpenDev(
        &nicPortCtx[0], NicType::VNIC_TYPE, rank_vector[userRank].devicePhyId, rank_vector[userRank].devicePhyId,
        rank_vector[userRank].nicIp[0]);
    netDevCtxMap.insert(std::make_pair(rank_vector[userRank].nicIp[0], nicPortCtx[0]));

    std::shared_ptr<TopoInfoExtractor> topoInfoExt_2;
    topoInfoExt_2.reset(new TopoInfoExtractor(
        collective_id_tmp, userRank, user_rank_size, TopoType::TOPO_TYPE_COMMON, DevType::DEV_TYPE_910, rank_vector, 0,
        true));
    std::map<HcclCMDType, std::vector<HcclAlgoType>> algoConfig;

    for (u32 opType = 0; opType < static_cast<u32>(HcclCMDType::HCCL_CMD_MAX); opType++) {
        std::vector<HcclAlgoType> defaultAlgoTypes;
        defaultAlgoTypes.push_back(HcclAlgoType::HCCL_ALGO_TYPE_NULL);
        defaultAlgoTypes.push_back(HcclAlgoType::HCCL_ALGO_TYPE_NULL);
        algoConfig[static_cast<HcclCMDType>(opType)] = defaultAlgoTypes;
    }
    algoConfig[HCCL_CMD_ALLREDUCE] = {HcclAlgoType::HCCL_ALGO_TYPE_NULL, HcclAlgoType::HCCL_ALGO_TYPE_AHC};

    ret = topoInfoExt_2->Init(algoConfig);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    CommFactory* comm_factory_rank_2 = new CommFactory(
        collective_id_tmp, userRank, user_rank_size, dispatcher, nullptr, netDevCtxMap, topoInfoExt_2, true,
        TopoType::TOPO_TYPE_COMMON, DevType::DEV_TYPE_910, rank_vector);

    ret = comm_factory_rank_2->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    s32 mem_size = 256;
    DeviceMem inputMem = DeviceMem::alloc(mem_size);
    DeviceMem outputMem = DeviceMem::alloc(mem_size);
    const string strTag = "test_tag";

    CommParaInfo commParaInfo;
    std::vector<std::unique_ptr<CommBase>> commVec;

    commParaInfo = CommParaInfo(COMM_COMBINE, CommType::COMM_TAG_WHOLE_NHR);
    ret = comm_factory_rank_2->CreateCommPlane(strTag, inputMem, outputMem, commParaInfo, commVec);
    EXPECT_NE(ret, HCCL_SUCCESS);

    commParaInfo = CommParaInfo(COMM_LEVEL1, CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING);
    ret = comm_factory_rank_2->CreateCommPlane(strTag, inputMem, outputMem, commParaInfo, commVec);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete comm_factory_rank_2;

    // ranksize 7 rank 5
    userRank = 5;

    HcclNetInit(
        NICDeployment::NIC_DEPLOYMENT_DEVICE, rank_vector[userRank].devicePhyId, rank_vector[userRank].devicePhyId,
        false);
    HcclNetOpenDev(
        &nicPortCtx[1], NicType::VNIC_TYPE, rank_vector[userRank].devicePhyId, rank_vector[userRank].devicePhyId,
        rank_vector[userRank].nicIp[0]);
    netDevCtxMap.insert(std::make_pair(rank_vector[userRank].nicIp[0], nicPortCtx[1]));

    std::shared_ptr<TopoInfoExtractor> topoInfoExt_5;
    topoInfoExt_5.reset(new TopoInfoExtractor(
        collective_id_tmp, userRank, user_rank_size, TopoType::TOPO_TYPE_COMMON, DevType::DEV_TYPE_910, rank_vector, 0,
        true));

    ret = topoInfoExt_5->Init(algoConfig);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    CommFactory* comm_factory_rank_5 = new CommFactory(
        collective_id_tmp, userRank, user_rank_size, dispatcher, nullptr, netDevCtxMap, topoInfoExt_5, true,
        TopoType::TOPO_TYPE_COMMON, DevType::DEV_TYPE_910, rank_vector);

    ret = comm_factory_rank_5->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    commParaInfo = CommParaInfo(COMM_COMBINE, CommType::COMM_TAG_WHOLE_NHR);
    ret = comm_factory_rank_5->CreateCommPlane(strTag, inputMem, outputMem, commParaInfo, commVec);
    EXPECT_NE(ret, HCCL_SUCCESS);

    commParaInfo = CommParaInfo(COMM_LEVEL1, CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING);
    ret = comm_factory_rank_5->CreateCommPlane(strTag, inputMem, outputMem, commParaInfo, commVec);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete comm_factory_rank_5;

    HcclNetCloseDev(nicPortCtx[1]);
    HcclNetCloseDev(nicPortCtx[0]);
    netDevCtxMap.clear();
    HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, rank_vector[2].devicePhyId, rank_vector[2].devicePhyId);
    HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, rank_vector[5].devicePhyId, rank_vector[5].devicePhyId);
}

TEST_F(CommFactoryTest, ut_init_with_err_input)
{
    s32 ret = HCCL_SUCCESS;

    u32 userRank = 0;
    u32 user_rank_size = 8;

    char collectiveId[SAL_UNIQUE_ID_BYTES];
    ret = SalGetUniqueId(collectiveId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string collective_id_tmp = collectiveId;
    // set device
    s32 device_id = 0;
    ret = hrtSetDevice(device_id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> rank_vector;
    get_rank_vector(rank_vector, user_rank_size);

    HcclNetInit(
        NICDeployment::NIC_DEPLOYMENT_DEVICE, rank_vector[userRank].devicePhyId, rank_vector[userRank].devicePhyId,
        false);
    HcclNetDevCtx nicPortCtx[3];
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;
    HcclNetOpenDev(
        &nicPortCtx[0], NicType::DEVICE_NIC_TYPE, rank_vector[userRank].devicePhyId, rank_vector[userRank].devicePhyId,
        rank_vector[userRank].nicIp[0]);
    netDevCtxMap.insert(std::make_pair(rank_vector[userRank].nicIp[0], nicPortCtx[0]));
    std::shared_ptr<CommFactory> comm_factory_0 = nullptr;

    std::shared_ptr<TopoInfoExtractor> topoInfoExt_0;
    topoInfoExt_0.reset(new TopoInfoExtractor(
        collective_id_tmp, userRank, user_rank_size, TopoType::TOPO_TYPE_RESERVED, DevType::DEV_TYPE_910, rank_vector));

    comm_factory_0.reset(new CommFactory(
        collective_id_tmp, userRank, user_rank_size, dispatcher, nullptr, netDevCtxMap, topoInfoExt_0, true,
        TopoType::TOPO_TYPE_RESERVED, DevType::DEV_TYPE_910, rank_vector));

    ret = comm_factory_0->Init();
    EXPECT_NE(ret, HCCL_SUCCESS);

    std::shared_ptr<CommFactory> comm_factory_1 = nullptr;

    std::shared_ptr<TopoInfoExtractor> topoInfoExt_1;
    topoInfoExt_1.reset(new TopoInfoExtractor(
        collective_id_tmp, userRank, user_rank_size, TopoType::TOPO_TYPE_8P_RING, DevType::DEV_TYPE_910, rank_vector));
    topoInfoExt_1->meshAggregationRankSize_ = MESH_AGGREGATION_RANK_SIZE_910;
    comm_factory_1.reset(new CommFactory(
        collective_id_tmp, userRank, user_rank_size, dispatcher, nullptr, netDevCtxMap, topoInfoExt_1, true,
        TopoType::TOPO_TYPE_8P_RING, DevType::DEV_TYPE_910, rank_vector));

    ret = comm_factory_1->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<CommFactory> comm_factory_2 = nullptr;

    std::shared_ptr<TopoInfoExtractor> topoInfoExt_2;
    topoInfoExt_2.reset(new TopoInfoExtractor(
        collective_id_tmp, userRank, user_rank_size, TopoType::TOPO_TYPE_4P_RING, DevType::DEV_TYPE_910, rank_vector));
    topoInfoExt_2->meshAggregationRankSize_ = MESH_AGGREGATION_RANK_SIZE_910;
    comm_factory_2.reset(new CommFactory(
        collective_id_tmp, userRank, user_rank_size, dispatcher, nullptr, netDevCtxMap, topoInfoExt_2, true,
        TopoType::TOPO_TYPE_4P_RING, DevType::DEV_TYPE_910, rank_vector));

    ret = comm_factory_2->Init();
    EXPECT_NE(ret, HCCL_SUCCESS);
    HcclNetCloseDev(nicPortCtx[0]);
    netDevCtxMap.clear();
    HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0);
}

TEST_F(CommFactoryTest, ut_init_with_err_topo)
{
    s32 ret = HCCL_SUCCESS;

    u32 userRank = 0;
    u32 user_rank_size = 8;

    char collectiveId[SAL_UNIQUE_ID_BYTES];
    ret = SalGetUniqueId(collectiveId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string collective_id_tmp = collectiveId;
    // set device
    s32 device_id = 0;
    ret = hrtSetDevice(device_id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> rank_vector;
    get_rank_vector(rank_vector, user_rank_size);

    HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, false);
    HcclNetDevCtx nicPortCtx[1];
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;
    HcclNetOpenDev(
        &nicPortCtx[0], NicType::DEVICE_NIC_TYPE, rank_vector[userRank].devicePhyId, rank_vector[userRank].devicePhyId,
        rank_vector[userRank].nicIp[0]);
    netDevCtxMap.insert(std::make_pair(rank_vector[userRank].nicIp[0], nicPortCtx[0]));
    std::shared_ptr<CommFactory> comm_factory = nullptr;

    std::shared_ptr<TopoInfoExtractor> topoInfoExt;
    topoInfoExt.reset(new TopoInfoExtractor(
        collective_id_tmp, userRank, user_rank_size, TopoType::TOPO_TYPE_RESERVED, DevType::DEV_TYPE_910, rank_vector));
    topoInfoExt->meshAggregationRankSize_ = MESH_AGGREGATION_RANK_SIZE_910;
    comm_factory.reset(new CommFactory(
        collective_id_tmp, userRank, user_rank_size, dispatcher, nullptr, netDevCtxMap, topoInfoExt, true,
        TopoType::TOPO_TYPE_RESERVED, DevType::DEV_TYPE_910, rank_vector));

    ret = comm_factory->Init();
    EXPECT_EQ(ret, HCCL_E_PARA);
    HcclNetCloseDev(nicPortCtx[0]);
    netDevCtxMap.clear();
    HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0);
}

TEST_F(CommFactoryTest, ut_init_with_err_rank_size)
{
    s32 ret = HCCL_SUCCESS;

    u32 userRank = 0;
    u32 user_rank_size = 2;

    char collectiveId[SAL_UNIQUE_ID_BYTES];
    ret = SalGetUniqueId(collectiveId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string collective_id_tmp = collectiveId;
    // set device
    s32 device_id = 0;
    ret = hrtSetDevice(device_id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> rank_vector;
    get_rank_vector(rank_vector, user_rank_size);

    HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, false);
    HcclNetDevCtx nicPortCtx[1];
    HcclNetOpenDev(
        &nicPortCtx[0], NicType::DEVICE_NIC_TYPE, rank_vector[userRank].devicePhyId, rank_vector[userRank].devicePhyId,
        rank_vector[userRank].nicIp[0]);
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;
    netDevCtxMap.insert(std::make_pair(rank_vector[userRank].nicIp[0], nicPortCtx[0]));
    std::shared_ptr<CommFactory> comm_factory = nullptr;

    std::shared_ptr<TopoInfoExtractor> topoInfoExt;
    topoInfoExt.reset(new TopoInfoExtractor(
        collective_id_tmp, userRank, user_rank_size, TopoType::TOPO_TYPE_8P_RING, DevType::DEV_TYPE_910, rank_vector, 0,
        true));
    topoInfoExt->meshAggregationRankSize_ = MESH_AGGREGATION_RANK_SIZE_910;
    comm_factory.reset(new CommFactory(
        collective_id_tmp, userRank, user_rank_size, dispatcher, nullptr, netDevCtxMap, topoInfoExt, true,
        TopoType::TOPO_TYPE_8P_RING, DevType::DEV_TYPE_910, rank_vector));

    std::map<HcclCMDType, std::vector<HcclAlgoType>> algoConfig;
    for (u32 opType = 0; opType < static_cast<u32>(HcclCMDType::HCCL_CMD_MAX); opType++) {
        algoConfig[static_cast<HcclCMDType>(opType)]
            = std::vector<HcclAlgoType>(4, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
    }
    ret = topoInfoExt->Init(algoConfig);
    EXPECT_NE(ret, HCCL_SUCCESS);
    HcclNetCloseDev(nicPortCtx[0]);
    netDevCtxMap.clear();
    HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0);
}

s32 stub_CommFactoryTest_hrtRaSocketNonBlockSendHB(const FdHandle fdHandle, const void* data, u64 size, u64* sent_size)
{
    *sent_size = size;
    return 0;
}

s32 stub_CommFactoryTest_hrtRaSocketNonBlockRecvHB(const FdHandle fdHandle, void* data, u64 size, u64* recvSize)
{
    static u32 count = 0;
    if (count++ % 5 != 0) {
        *recvSize = size;
        count = 0;
    }
    return 0;
}

s32 stub_CommFactoryTest_hrtRaGetSockets(u32 role, struct SocketInfoT conn[], u32 num, u32* connectedNum)
{
    static std::vector<int> fdHandle;
    for (int i = 0; i < num; i++) {
        fdHandle.push_back(0);
        conn[i].fdHandle = 0;
        conn[i].status = CONNECT_OK;
    }
    *connectedNum = num;
    return 0;
}

HcclResult stub_CommFactoryTest_GetIsSupSockBatchCloseImmed(u32 phyId, bool& isSupportBatchClose)
{
    isSupportBatchClose = true;
    return HCCL_SUCCESS;
}

HcclResult stub_CommFactoryTest_GetNicHandleInfo(
    std::map<HcclIpAddress, IpSocket>& socketMap, const HcclIpAddress& ip, SocketHandle& nicSocketHandle)
{
    nicSocketHandle = (void*)0x00000001;
    return HCCL_SUCCESS;
}

TEST_F(CommFactoryTest, ut_create_comm_suppod)
{
    s32 ret = HCCL_SUCCESS;

    DlTdtFunction::GetInstance().DlTdtFunctionInit();
    DlRaFunction::GetInstance().DlRaFunctionInit();

    u32 userRank = 0;
    u32 user_rank_size = 4;

    char collectiveId[SAL_UNIQUE_ID_BYTES];
    ret = SalGetUniqueId(collectiveId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string collective_id_tmp = collectiveId;

    MOCKER_CPP(&CommBase::IsSupportInterHccs).stubs().with(mockcpp::any()).will(returnValue(true));

    MOCKER_CPP(&CommBase::CreateDestLink).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CommBase::GetSuperNodeIntraRankIPInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(GetIsSupSockBatchCloseImmed).stubs().will(invoke(stub_CommFactoryTest_GetIsSupSockBatchCloseImmed));

    MOCKER(hrtRaSocketWhiteListAdd).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtRaSocketWhiteListDel).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtRaSocketBatchConnect).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtRaGetSockets).stubs().will(invoke(stub_CommFactoryTest_hrtRaGetSockets));

    MOCKER(hrtRaSocketBatchClose).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtRaSocketNonBlockSend).stubs().will(invoke(stub_CommFactoryTest_hrtRaSocketNonBlockSendHB));

    MOCKER(hrtRaSocketNonBlockRecv).stubs().will(invoke(stub_CommFactoryTest_hrtRaSocketNonBlockRecvHB));

    std::vector<RankInfo> rank_vector;
    RankInfo tmp_para_0;
    tmp_para_0.userRank = 0;
    tmp_para_0.worldRank = 0;
    tmp_para_0.devicePhyId = 0;
    tmp_para_0.deviceType = DevType::DEV_TYPE_910_93;
    tmp_para_0.serverIdx = 0;
    tmp_para_0.serverId = "10.0.0.10";
    tmp_para_0.nicIp.push_back(HcclIpAddress("192.168.0.18"));
    tmp_para_0.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    ret = HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, tmp_para_0.devicePhyId, tmp_para_0.devicePhyId, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclNetDevCtx portCtx;
    ret = HcclNetOpenDev(
        &portCtx, NicType::DEVICE_NIC_TYPE, tmp_para_0.devicePhyId, tmp_para_0.devicePhyId,
        HcclIpAddress(tmp_para_0.devicePhyId));
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;
    netDevCtxMap.insert(make_pair(HcclIpAddress(tmp_para_0.devicePhyId), portCtx));

    RankInfo tmp_para_1;
    tmp_para_1.userRank = 1;
    tmp_para_1.worldRank = 1;
    tmp_para_1.devicePhyId = 1;
    tmp_para_1.deviceType = DevType::DEV_TYPE_910_93;
    tmp_para_1.serverIdx = 0;
    tmp_para_1.serverId = "10.0.0.10";
    tmp_para_1.nicIp.push_back(HcclIpAddress("192.168.0.19"));
    tmp_para_1.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_2;
    tmp_para_2.userRank = 2;
    tmp_para_2.worldRank = 2;
    tmp_para_2.devicePhyId = 0;
    tmp_para_2.deviceType = DevType::DEV_TYPE_910_93;
    tmp_para_2.serverIdx = 1;
    tmp_para_2.serverId = "10.0.0.20";
    tmp_para_2.nicIp.push_back(HcclIpAddress("192.168.0.20"));
    tmp_para_2.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_3;
    tmp_para_3.userRank = 3;
    tmp_para_3.worldRank = 3;
    tmp_para_3.devicePhyId = 1;
    tmp_para_3.deviceType = DevType::DEV_TYPE_910_93;
    tmp_para_3.serverIdx = 1;
    tmp_para_3.serverId = "10.0.0.20";
    tmp_para_3.nicIp.push_back(HcclIpAddress("192.168.0.21"));
    tmp_para_3.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    rank_vector.push_back(tmp_para_0);
    rank_vector.push_back(tmp_para_1);
    rank_vector.push_back(tmp_para_2);
    rank_vector.push_back(tmp_para_3);

    std::shared_ptr<TopoInfoExtractor> topoInfoExt;
    topoInfoExt.reset(new TopoInfoExtractor(
        collective_id_tmp, userRank, user_rank_size, TopoType::TOPO_TYPE_NP_DOUBLE_RING, DevType::DEV_TYPE_910_93,
        rank_vector));

    CommFactory* comm_factory = new CommFactory(
        collective_id_tmp, userRank, user_rank_size, dispatcher, nullptr, netDevCtxMap, topoInfoExt, true,
        TopoType::TOPO_TYPE_NP_DOUBLE_RING, DevType::DEV_TYPE_910_93, rank_vector, NICDeployment::NIC_DEPLOYMENT_DEVICE,
        false, 0, false, true);

    ret = comm_factory->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    s32 mem_size = 256;
    DeviceMem inputMem = DeviceMem::alloc(mem_size);
    DeviceMem outputMem = DeviceMem::alloc(mem_size);
    const string strTag = collective_id_tmp;

    CommParaInfo commParaInfo;
    std::vector<std::unique_ptr<CommBase>> commVec;

    commParaInfo = CommParaInfo(COMM_LEVEL1, CommType::COMM_TAG_RING_INNER);
    ret = comm_factory->CreateCommPlane(strTag, inputMem, outputMem, commParaInfo, commVec);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclNetCloseDev(portCtx);
    HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, tmp_para_0.devicePhyId, tmp_para_0.devicePhyId);

    delete comm_factory;
}

TEST_F(CommFactoryTest, ut_create_commmesh_combined_1server_16p)
{
    s32 ret = HCCL_SUCCESS;

    u32 userRank = 0;
    u32 user_rank_size = 5;

    char collectiveId[SAL_UNIQUE_ID_BYTES];
    ret = SalGetUniqueId(collectiveId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string collective_id_tmp = collectiveId;

    std::vector<RankInfo> rank_vector;
    for (u32 i = 0; i < 5; i++) {
        RankInfo tmp_para_0;
        tmp_para_0.userRank = i;
        tmp_para_0.devicePhyId = i;
        tmp_para_0.deviceType = DevType::DEV_TYPE_910B;
        tmp_para_0.serverIdx = 1;
        tmp_para_0.serverId = "10.0.0.10";
        tmp_para_0.nicIp.push_back(HcclIpAddress("192.168.0.18"));
        tmp_para_0.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
        rank_vector.push_back(tmp_para_0);
    }

    ret = HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclNetDevCtx portCtx;
    ret = HcclNetOpenDev(
        &portCtx, NicType::VNIC_TYPE, rank_vector[0].devicePhyId, rank_vector[0].devicePhyId, rank_vector[0].nicIp[0]);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;
    netDevCtxMap.insert(make_pair(rank_vector[0].nicIp[0], portCtx));

    std::shared_ptr<TopoInfoExtractor> topoInfoExt;
    topoInfoExt.reset(new TopoInfoExtractor(
        collective_id_tmp, userRank, user_rank_size, TopoType::TOPO_TYPE_COMMON, DevType::DEV_TYPE_910B, rank_vector));

    CommFactory* comm_factory = new CommFactory(
        collective_id_tmp, userRank, user_rank_size, dispatcher, nullptr, netDevCtxMap, topoInfoExt, false,
        TopoType::TOPO_TYPE_COMMON, DevType::DEV_TYPE_910B, rank_vector);

    ret = comm_factory->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    s32 mem_size = 256;
    DeviceMem inputMem = DeviceMem::alloc(mem_size);
    DeviceMem outputMem = DeviceMem::alloc(mem_size);
    const string strTag = "test_tag";

    std::set<u32> targetRanks = {1, 2, 3, 4};

    CommParaInfo commParaInfo;
    std::vector<std::unique_ptr<CommBase>> commVec;

    HcclNetCloseDev(portCtx);
    HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0);
    delete comm_factory;
    GlobalMockObject::verify();
}

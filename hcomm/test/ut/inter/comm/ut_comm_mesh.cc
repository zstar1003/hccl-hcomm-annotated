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

#define protected public
#include "comm_mesh_pub.h"
#undef protected
#include "sal.h"
#include <vector>
#include "comm_factory_pub.h"

#include "llt_hccl_stub_pub.h"
#include "llt_hccl_stub.h"
#include "dlra_function.h"
#include "tsd/tsd_client.h"
#include "dltdt_function.h"
#include "profiler_manager.h"
#include "network_manager_pub.h"
#include "p2p_mgmt_pub.h"
#include "rank_consistentcy_checker.h"
using namespace std;
using namespace hccl;

class CommMeshTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--CommMeshTest SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--CommMeshTest TearDown--\033[0m" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        DlTdtFunction::GetInstance().DlTdtFunctionInit();
        TsdOpen(1, 2);
        std::cout << "A Test SetUP" << std::endl;
        MOCKER(hrtRaGetSingleSocketVnicIpInfo)
            .stubs()
            .with(mockcpp::any())
            .will(invoke(stub_hrtRaGetSingleSocketVnicIpInfo));
        s32 portNum = -1;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
    }
    virtual void TearDown()
    {
        TsdClose(1);
        std::cout << "A Test TearDown" << std::endl;
        GlobalMockObject::verify();
    }
};

typedef struct innerpara_struct_mesh {
    std::string collectiveId;
    u32 userRank;
    u32 user_rank_size;
    u32 rank;
    u32 rank_size;
    u32 devicePhyId;
    std::vector<s32> device_ids;
    std::vector<u32> device_ips;
    std::vector<u32> user_ranks;
    std::string tag;
    HcclDispatcher dispatcher;
    std::unique_ptr<NotifyPool> notifyPool;
    IntraExchanger* exchanger;
    std::vector<RankInfo> para_vector;
    DeviceMem inputMem;
    DeviceMem outputMem;
    std::shared_ptr<CommMesh> comm_mesh;
} innerpara_t_mesh;

HcclDispatcher get_mesh_dispatcher(s32 devid, std::shared_ptr<hccl::ProfilerManager>& profilerManager)
{
    HcclResult ret = HCCL_SUCCESS;
    ret = hrtSetDevice(devid);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 创建dispatcher

    void* dispatcher = nullptr;
    ret = HcclDispatcherInit(DispatcherType::DISPATCHER_NORMAL, devid, &dispatcher);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(dispatcher, nullptr);
    return dispatcher;
}

void* comm_mesh_task_handle(void* para)
{
    HcclResult ret = HCCL_SUCCESS;
    innerpara_t_mesh* para_info = (innerpara_t_mesh*)para;
    s32 rt_ret = 0;
    ret = hrtSetDevice(para_info->devicePhyId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    RankConsistentcyChecker::GetInstance().ClearCheckInfo();
    /* 作为socket server端启动监听 */
    ret = HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, para_info->devicePhyId, para_info->devicePhyId, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclNetDevCtx portCtx;
    ret = HcclNetOpenDev(
        &portCtx, NicType::VNIC_TYPE, para_info->devicePhyId, para_info->devicePhyId,
        HcclIpAddress(para_info->devicePhyId));
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;
    netDevCtxMap.insert(make_pair(HcclIpAddress(para_info->devicePhyId), portCtx));

    IntraExchanger exchanger{};
    ret = CreateIntraExchanger(
        para_info->collectiveId, portCtx, para_info->devicePhyId, para_info->devicePhyId, para_info->userRank,
        para_info->user_rank_size, para_info->device_ids, para_info->user_ranks, true, exchanger);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    TopoType topoFlag = TopoType::TOPO_TYPE_8P_RING;
    para_info->comm_mesh.reset(new CommMesh(
        para_info->collectiveId, para_info->userRank, para_info->user_rank_size, para_info->rank, para_info->rank_size,
        topoFlag, para_info->dispatcher, para_info->notifyPool, netDevCtxMap, exchanger, para_info->para_vector,
        para_info->inputMem, para_info->outputMem, false, para_info->tag));
    ret = para_info->notifyPool->RegisterOp(para_info->tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = para_info->comm_mesh->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = para_info->notifyPool->UnregisterOp(para_info->tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    RankConsistentcyChecker::GetInstance().ClearCheckInfo();
    HcclNetCloseDev(portCtx);
    HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, para_info->devicePhyId, para_info->devicePhyId);
    return (NULL);
}

TEST_F(CommMeshTest, destructor_D0)
{
    std::string rootInfo = "test_collective";

    IntraExchanger exchanger{};
    std::vector<RankInfo> para_vector(1);

    TopoType topoFlag = TopoType::TOPO_TYPE_8P_RING;
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;
    CommMesh* comm_mesh = new CommMesh(
        rootInfo, 0, 1, 0, 1, topoFlag, nullptr, nullptr, netDevCtxMap, exchanger, para_vector, DeviceMem(),
        DeviceMem(), false, "");

    delete comm_mesh;
}

TEST_F(CommMeshTest, init)
{
    s32 ret = HCCL_SUCCESS;

    u32 userRank = 1;
    u32 user_rank_size = 5;

    RankInfo tmp_para;

    tmp_para.userRank = userRank;
    DeviceMem inputMem = DeviceMem::alloc(128 * 3);
    DeviceMem outputMem = DeviceMem::alloc(128 * 3);

    s32 device_id = 0;
    ret = hrtGetDevice(&device_id);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    tmp_para.devicePhyId = device_id;

    tmp_para.serverId = "10.21.78.208";
    tmp_para.nicIp.push_back(HcclIpAddress("10.21.78.208"));

    std::vector<RankInfo> para_vector;
    para_vector.push_back(tmp_para);

    char collectiveId[SAL_UNIQUE_ID_BYTES];
    ret = SalGetUniqueId(collectiveId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string collective_id_tmp = collectiveId;

    IntraExchanger exchanger{};

    TopoType topoFlag = TopoType::TOPO_TYPE_8P_RING;
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;
    CommMesh* comm_mesh = new CommMesh(
        collective_id_tmp, userRank, user_rank_size, 0, 1, topoFlag, nullptr, nullptr, netDevCtxMap, exchanger,
        para_vector, inputMem, outputMem, true, "");

    ret = comm_mesh->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete comm_mesh;
}

HcclResult StupIsSupportAicpuNormalQP(const u32& devicePhyId, bool& isSupportNormalQP)
{
    isSupportNormalQP = devicePhyId >= 0 ? true : false;
    return HCCL_SUCCESS;
}

TEST_F(CommMeshTest, ut_set_machinePara)
{
    s32 ret = HCCL_SUCCESS;

    u32 userRank = 1;
    u32 user_rank_size = 5;

    RankInfo tmp_para;

    tmp_para.userRank = userRank;
    DeviceMem inputMem = DeviceMem::alloc(128 * 3);
    DeviceMem outputMem = DeviceMem::alloc(128 * 3);

    s32 device_id = 0;
    ret = hrtGetDevice(&device_id);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    tmp_para.devicePhyId = device_id;

    tmp_para.serverId = "10.21.78.208";
    tmp_para.nicIp.push_back(HcclIpAddress("10.21.78.208"));

    std::vector<RankInfo> para_vector;
    para_vector.push_back(tmp_para);

    char collectiveId[SAL_UNIQUE_ID_BYTES];
    ret = SalGetUniqueId(collectiveId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string collective_id_tmp = collectiveId;

    IntraExchanger exchanger{};

    TopoType topoFlag = TopoType::TOPO_TYPE_8P_RING;
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;
    CommMesh* comm_mesh = new CommMesh(
        collective_id_tmp, userRank, user_rank_size, 0, 1, topoFlag, nullptr, nullptr, netDevCtxMap, exchanger,
        para_vector, inputMem, outputMem, true, "tag_" + HCCL_MC2_MULTISERVER_SUFFIX);

    ret = comm_mesh->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    const MachineType machineType = MachineType::MACHINE_SERVER_TYPE;
    const u32 dstRank = 0;
    std::vector<std::shared_ptr<HcclSocket>> socketList;
    MachinePara machinePara;

    MOCKER(IsSupportAicpuNormalQP).stubs().with(mockcpp::any()).will(invoke(StupIsSupportAicpuNormalQP));

    ret = comm_mesh->SetMachinePara(machineType, tmp_para.serverId, dstRank, socketList, machinePara);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete comm_mesh;
    GlobalMockObject::verify();
}

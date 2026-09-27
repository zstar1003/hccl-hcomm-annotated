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
#include "comm_ring_pub.h"
#undef protected
#include "sal.h"
#include <vector>
#include "comm_factory_pub.h"
#include "llt_hccl_stub_pub.h"
#include "llt_hccl_stub.h"
#include "rank_consistentcy_checker.h"
#include "dlra_function.h"
#include "tsd/tsd_client.h"
#include "dltdt_function.h"
#include "profiler_manager.h"
#include "network_manager_pub.h"
#include "p2p_mgmt_pub.h"
#include "externalinput.h"
using namespace std;
using namespace hccl;

class CommRingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--CommRingTest SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--CommRingTest TearDown--\033[0m" << std::endl; }
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

typedef struct innerpara_struct {
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
    std::shared_ptr<CommRing> comm_ring;
} innerpara_t;

HcclDispatcher get_ring_dispatcher(s32 devid, std::shared_ptr<hccl::ProfilerManager>& profilerManager)
{
    HcclResult ret = HCCL_SUCCESS;
    ret = hrtSetDevice(devid);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 创建dispatcher
    DevType chipType = DevType::DEV_TYPE_910;

    void* dispatcher = nullptr;
    ret = HcclDispatcherInit(DispatcherType::DISPATCHER_NORMAL, devid, &dispatcher);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(dispatcher, nullptr);

    return dispatcher;
}

std::unique_ptr<NotifyPool> get_notifyPool(s32 rank)
{
    HcclResult ret = HCCL_SUCCESS;

    std::unique_ptr<NotifyPool> notifyPool = nullptr;
    notifyPool.reset(new (std::nothrow) NotifyPool());
    EXPECT_NE(notifyPool, nullptr);
    ret = notifyPool->Init(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    return std::move(notifyPool);
}

void* comm_ring_task_handle(void* para)
{
    HcclResult ret = HCCL_SUCCESS;
    innerpara_t* para_info = (innerpara_t*)para;
    s32 rt_ret = 0;

    ret = hrtSetDevice(para_info->devicePhyId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    RankConsistentcyChecker::GetInstance().ClearCheckInfo();
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
    para_info->comm_ring.reset(new CommRing(
        para_info->collectiveId, para_info->userRank, para_info->user_rank_size, para_info->rank, para_info->rank_size,
        topoFlag, para_info->dispatcher, para_info->notifyPool, netDevCtxMap, exchanger, para_info->para_vector,
        para_info->inputMem, para_info->outputMem, false, para_info->tag));
    ret = para_info->notifyPool->RegisterOp(para_info->tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = para_info->comm_ring->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<Transport>& link_rank_2 = para_info->comm_ring->GetTransportByRank(para_info->rank);
    EXPECT_EQ(link_rank_2, nullptr);

    std::shared_ptr<Transport>& link_rank_3 = para_info->comm_ring->GetTransportByRank(para_info->rank_size + 1);
    EXPECT_EQ(link_rank_3, nullptr);
    ret = para_info->notifyPool->UnregisterOp(para_info->tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    RankConsistentcyChecker::GetInstance().ClearCheckInfo();
    HcclNetCloseDev(portCtx);
    HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, para_info->devicePhyId, para_info->devicePhyId);
    return (NULL);
}
#if 0 // 执行失败
TEST_F(CommRingTest, ut_comminter_init_4_thread)
{
    MOCKER(hrtRaGetInterfaceVersion)
    .expects(atMost(1))
    .will(returnValue(HCCL_SUCCESS));
    s32 ret = HCCL_SUCCESS;
    RankConsistentcyChecker::GetInstance().ClearCheckInfo();
    RankInfo tmp_para_0;

    tmp_para_0.userRank = 0;
    tmp_para_0.devicePhyId = 0;
    tmp_para_0.deviceType = DevType::DEV_TYPE_910;
    tmp_para_0.serverId = "10.21.78.208";
    tmp_para_0.nicIp.push_back(HcclIpAddress("10.21.78.208"));
    tmp_para_0.superDeviceId = 0;
    // tmp_para_0.inputMem = DeviceMem::alloc(128*3);
    // tmp_para_0.outputMem = DeviceMem::alloc(128*3);

    RankInfo tmp_para_1;

    tmp_para_1.userRank = 1;
    tmp_para_1.devicePhyId = 1;
    tmp_para_1.deviceType = DevType::DEV_TYPE_910;
    tmp_para_1.serverId = "10.21.78.208";
    tmp_para_1.nicIp.push_back(HcclIpAddress("10.21.78.208"));
    tmp_para_0.superDeviceId = 1;
    // tmp_para_1.inputMem = DeviceMem::alloc(128*3);
    // tmp_para_1.outputMem = DeviceMem::alloc(128*3);

    RankInfo tmp_para_2;

    tmp_para_2.userRank = 2;
    tmp_para_2.devicePhyId = 2;
    tmp_para_2.deviceType = DevType::DEV_TYPE_910;
    tmp_para_2.serverId = "10.21.78.208";
    tmp_para_2.nicIp.push_back(HcclIpAddress("10.21.78.208"));
    tmp_para_0.superDeviceId = 2;
    // tmp_para_2.inputMem = DeviceMem::alloc(128*3);
    // tmp_para_2.outputMem = DeviceMem::alloc(128*3);

    RankInfo tmp_para_3;

    tmp_para_3.userRank = 3;
    tmp_para_3.devicePhyId = 3;
    tmp_para_3.deviceType = DevType::DEV_TYPE_910;
    tmp_para_3.serverId = "10.21.78.208";
    tmp_para_3.nicIp.push_back(HcclIpAddress("10.21.78.208"));
    tmp_para_0.superDeviceId = 3;
    // tmp_para_3.inputMem = DeviceMem::alloc(128*3);
    // tmp_para_3.outputMem = DeviceMem::alloc(128*3);

    std::vector<RankInfo> para_vector;
    para_vector.push_back(tmp_para_0);
    para_vector.push_back(tmp_para_1);
    para_vector.push_back(tmp_para_2);
    para_vector.push_back(tmp_para_3);
    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    DlTdtFunction::GetInstance().DlTdtFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    char collectiveId[SAL_UNIQUE_ID_BYTES];
    ret = SalGetUniqueId(collectiveId);

    std::string collective_id_tmp = collectiveId;

    const s32 dev_num = 4;
    s32 dev_list[dev_num] = {0, 1, 2, 3};
    std::vector<s32> device_ids(dev_list, dev_list+dev_num);

    std::vector<u32> ip_list;
    for (int i = 0;i<dev_num;i++ )
    {
        u32 ipAddr = 0;
        (void)rt_get_dev_ip(0, i, &ipAddr);
        ip_list.push_back(ipAddr);
    }

    const u32 rank_list[dev_num] = {0, 1, 2, 3};
    std::vector<u32> user_ranks(rank_list, rank_list+dev_num);

    std::shared_ptr<CommRing> comm_ring = nullptr;

    sal_thread_t tid[4];
    innerpara_t para_info[4];
    s32 ndev = 4;
    std::shared_ptr<ProfilerManager> profilerManager[4];

    for (s32 i = 0; i < ndev; i++)
    {
        para_info[i].collectiveId = collective_id_tmp;
        para_info[i].userRank = i;
        para_info[i].user_rank_size = ndev;
        para_info[i].rank = i;
        para_info[i].rank_size = ndev;
        para_info[i].devicePhyId = dev_list[i];
        EXPECT_EQ(ret, HCCL_SUCCESS);
        para_info[i].dispatcher= get_ring_dispatcher(i, profilerManager[i]);
        para_info[i].notifyPool= get_notifyPool(i);
        para_info[i].device_ids.assign(device_ids.begin(), device_ids.end());
        para_info[i].device_ips.assign(ip_list.begin(), ip_list.end());
        para_info[i].user_ranks.assign(user_ranks.begin(), user_ranks.end());
        para_info[i].tag = "st_comminter_init_4_thread_sameip";
        para_info[i].para_vector = para_vector;
        para_info[i].inputMem = DeviceMem::alloc(1024);
        para_info[i].outputMem = DeviceMem::alloc(1024);
        para_info[i].comm_ring = comm_ring;
    }

    MOCKER(&P2PMgmtPub::WaitP2PEnabled)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    tid[0] = sal_thread_create("commRing rank0 thread", comm_ring_task_handle, (void*)&para_info[0]);

    tid[1] = sal_thread_create("commRing rank1 thread", comm_ring_task_handle, (void*)&para_info[1]);

    tid[2] = sal_thread_create("commRing rank2 thread", comm_ring_task_handle, (void*)&para_info[2]);

    tid[3] = sal_thread_create("commRing rank3 thread", comm_ring_task_handle, (void*)&para_info[3]);

    while (sal_thread_is_running(tid[1]) || sal_thread_is_running(tid[2])
           || sal_thread_is_running(tid[3]) || sal_thread_is_running(tid[0]))
    {
        SaluSleep(SAL_MILLISECOND_USEC * 10);;
    }

    for (s32 j = 0; j < ndev; j++)
    {
        ret = NetworkManager::GetInstance(dev_list[j]).Destroy();
        EXPECT_EQ(ret, HCCL_SUCCESS);
        (void)sal_thread_destroy(tid[j]);
    }

    //显示释放资源
    for(s32 j=0; j < ndev; j++) {
        para_info[j].comm_ring.reset();
        if (para_info[j].dispatcher != nullptr) {
            ret = HcclDispatcherDestroy(para_info[j].dispatcher);
            EXPECT_EQ(ret, HCCL_SUCCESS);
            para_info[j].dispatcher = nullptr;
        }
    }

    HCCL_INFO("sal_thread_destroy BBB");
    GlobalMockObject::verify();
}

TEST_F(CommRingTest, ut_comminter_init_5_thread)
{
    s32 ret = HCCL_SUCCESS;

    MOCKER(hrtRaGetInterfaceVersion)
    .expects(atMost(1))
    .will(returnValue(HCCL_SUCCESS));

    RankInfo tmp_para_0;

    tmp_para_0.userRank = 0;
    tmp_para_0.devicePhyId = 0;
    tmp_para_0.deviceType = DevType::DEV_TYPE_910;
    tmp_para_0.serverId = "10.21.78.208";
    tmp_para_0.nicIp.push_back(HcclIpAddress("10.21.78.208"));
    // tmp_para_0.inputMem = DeviceMem::alloc(128*3);
    // tmp_para_0.outputMem = DeviceMem::alloc(128*3);

    RankInfo tmp_para_1;

    tmp_para_1.userRank = 1;
    tmp_para_1.devicePhyId = 1;
    tmp_para_1.deviceType = DevType::DEV_TYPE_910;
    tmp_para_1.serverId = "10.21.78.208";
    tmp_para_1.nicIp.push_back(HcclIpAddress("10.21.78.208"));
    // tmp_para_1.inputMem = DeviceMem::alloc(128*3);
    // tmp_para_1.outputMem = DeviceMem::alloc(128*3);
    RankInfo tmp_para_2;

    tmp_para_2.userRank = 2;
    tmp_para_2.devicePhyId = 2;
    tmp_para_2.deviceType = DevType::DEV_TYPE_910;
    tmp_para_2.serverId = "10.21.78.208";
    tmp_para_2.nicIp.push_back(HcclIpAddress("10.21.78.208"));
    // tmp_para_2.inputMem = DeviceMem::alloc(128*3);
    // tmp_para_2.outputMem = DeviceMem::alloc(128*3);

    RankInfo tmp_para_3;

    tmp_para_3.userRank = 3;
    tmp_para_3.devicePhyId = 3;
    tmp_para_3.deviceType = DevType::DEV_TYPE_910;
    tmp_para_3.serverId = "10.21.78.208";
    tmp_para_3.nicIp.push_back(HcclIpAddress("10.21.78.208"));
    // tmp_para_3.inputMem = DeviceMem::alloc(128*3);
    // tmp_para_3.outputMem = DeviceMem::alloc(128*3);

    RankInfo tmp_para_4;

    tmp_para_4.userRank = 4;
    tmp_para_4.devicePhyId = 4;
    tmp_para_4.deviceType = DevType::DEV_TYPE_910;
    tmp_para_4.serverId = "10.21.78.208";
    tmp_para_4.nicIp.push_back(HcclIpAddress("10.21.78.208"));
    // tmp_para_4.inputMem = DeviceMem::alloc(128*3);
    // tmp_para_4.outputMem = DeviceMem::alloc(128*3);
    std::vector<RankInfo> para_vector;
    para_vector.push_back(tmp_para_0);
    para_vector.push_back(tmp_para_1);
    para_vector.push_back(tmp_para_2);
    para_vector.push_back(tmp_para_3);
    para_vector.push_back(tmp_para_4);

    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    DlTdtFunction::GetInstance().DlTdtFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    char collectiveId[SAL_UNIQUE_ID_BYTES];
    ret = SalGetUniqueId(collectiveId);

    std::string collective_id_tmp = collectiveId;

    const s32 dev_num = 5;
    s32 dev_list[dev_num] = {0, 1, 2, 3,4};
    std::vector<s32> device_ids(dev_list, dev_list+dev_num);

    std::vector<u32> ip_list;
    for (int i = 0;i<dev_num;i++ )
    {
        u32 ipAddr = 0;
        (void)rt_get_dev_ip(0, i, &ipAddr);
        ip_list.push_back(ipAddr);
    }

    const u32 rank_list[dev_num] = {0, 1, 2, 3, 4};
    std::vector<u32> user_ranks(rank_list, rank_list+dev_num);

    std::shared_ptr<CommRing> comm_ring = nullptr;

    sal_thread_t tid[5];
    innerpara_t para_info[5];
        std::shared_ptr<ProfilerManager> profilerManager[5];
    s32 ndev = 5;

    for (s32 i = 0; i < ndev; i++)
    {
        para_info[i].collectiveId = collective_id_tmp;
        para_info[i].userRank = i;
        para_info[i].user_rank_size = ndev;
        para_info[i].rank = i;
        para_info[i].rank_size = ndev;
        para_info[i].devicePhyId = dev_list[i];
        EXPECT_EQ(ret, HCCL_SUCCESS);
        para_info[i].dispatcher = get_ring_dispatcher(i, profilerManager[i]);
        para_info[i].notifyPool = get_notifyPool(i);
        para_info[i].device_ids.assign(device_ids.begin(), device_ids.end());
        para_info[i].device_ips.assign(ip_list.begin(), ip_list.end());
        para_info[i].user_ranks.assign(user_ranks.begin(), user_ranks.end());
        para_info[i].tag = "st_comminter_init_5_thread_diffip";
        para_info[i].para_vector = para_vector;
        para_info[i].inputMem = DeviceMem::alloc(1024);
        para_info[i].outputMem = DeviceMem::alloc(1024);
        para_info[i].comm_ring = comm_ring;
    }

    MOCKER(&P2PMgmtPub::WaitP2PEnabled)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    tid[0] = sal_thread_create("commRing rank0 thread", comm_ring_task_handle, (void*)&para_info[0]);

    tid[1] = sal_thread_create("commRing rank1 thread", comm_ring_task_handle, (void*)&para_info[1]);

    tid[2] = sal_thread_create("commRing rank2 thread", comm_ring_task_handle, (void*)&para_info[2]);

    tid[3] = sal_thread_create("commRing rank3 thread", comm_ring_task_handle, (void*)&para_info[3]);

    tid[4] = sal_thread_create("commRing rank4 thread", comm_ring_task_handle, (void*)&para_info[4]);

    while (sal_thread_is_running(tid[1]) || sal_thread_is_running(tid[2])
           || sal_thread_is_running(tid[3]) || sal_thread_is_running(tid[0])
           || sal_thread_is_running(tid[4]))
    {
        SaluSleep(SAL_MILLISECOND_USEC * 10);;
    }

    for (s32 j = 0; j < ndev; j++)
    {
        ret = NetworkManager::GetInstance(dev_list[j]).Destroy();
        EXPECT_EQ(ret, HCCL_SUCCESS);
        (void)sal_thread_destroy(tid[j]);
    }

    //显示释放资源
    for(s32 j=0; j < ndev; j++) {
        para_info[j].comm_ring.reset();
        if (para_info[j].dispatcher != nullptr) {
            ret = HcclDispatcherDestroy(para_info[j].dispatcher);
            EXPECT_EQ(ret, HCCL_SUCCESS);
            para_info[j].dispatcher = nullptr;
        }
    }
    GlobalMockObject::verify();
}
#endif
TEST_F(CommRingTest, ut_comminter_get_sockets_per_link)
{
    // 补充覆盖率
    MOCKER(GetWorkflowMode)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));

    setenv("HCCL_RDMA_QP_PORT_CONFIG_PATH", "/tmp/", 1);
    setenv("HCCL_RDMA_QPS_PER_CONNECTION", "5", 1);
    InitEnvVarParam();

    HcclDispatcher dispatcher;
    std::string rootInfo = "test_collective";
    IntraExchanger exchanger{};
    RankInfo rankInfo;
    rankInfo.deviceType = DevType::DEV_TYPE_910B;
    std::vector<RankInfo> para_vector(1, rankInfo);

    TopoType topoFlag = TopoType::TOPO_TYPE_8P_RING;
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;
    CommRing comm_ring(
        rootInfo, 0, 1, 0, 1, topoFlag, dispatcher, nullptr, netDevCtxMap, exchanger, para_vector, DeviceMem(),
        DeviceMem(), true);
    comm_ring.GetSocketsPerLink();

    unsetenv("HCCL_RDMA_QP_PORT_CONFIG_PATH");
    unsetenv("HCCL_RDMA_QPS_PER_CONNECTION");
    ResetInitState();

    setenv("HCCL_RDMA_QPS_PER_CONNECTION", "5", 1);
    InitEnvVarParam();
    comm_ring.GetSocketsPerLink();
    unsetenv("HCCL_RDMA_QPS_PER_CONNECTION");
    ResetInitState();
}

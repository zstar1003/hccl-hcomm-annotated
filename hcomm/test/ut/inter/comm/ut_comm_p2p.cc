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
#include "comm_p2p_pub.h"
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

using namespace std;
using namespace hccl;

class CommP2PTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--CommP2PTest SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--CommP2PTest TearDown--\033[0m" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        DlTdtFunction::GetInstance().DlTdtFunctionInit();
        TsdOpen(1, 2);
        s32 portNum = -1;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        TsdClose(1);
        std::cout << "A Test TearDown" << std::endl;
    }
};
typedef struct innerpara_struct_p2p {
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
    std::vector<RankInfo> para_vector;
    DeviceMem inputMem;
    DeviceMem outputMem;
    std::shared_ptr<CommP2P> comm_p2p;
} innerpara_t_p2p;

HcclDispatcher get_p2p_dispatcher(s32 devid, std::shared_ptr<hccl::ProfilerManager>& profilerManager)
{
    HcclResult ret = HCCL_SUCCESS;
    hrtSetDevice(devid);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 创建dispatcher
    void* dispatcher = nullptr;
    ret = HcclDispatcherInit(DispatcherType::DISPATCHER_NORMAL, devid, &dispatcher);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(dispatcher, nullptr);

    return dispatcher;
}

void* comm_p2p_task_handle(void* para)
{
    s32 ret = HCCL_SUCCESS;
    innerpara_t_p2p* para_info = (innerpara_t_p2p*)para;
    hrtSetDevice(para_info->devicePhyId);

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

    std::unique_ptr<NotifyPool> notifyPool = nullptr;
    notifyPool.reset(new (std::nothrow) NotifyPool());
    EXPECT_NE(notifyPool, nullptr);
    ret = notifyPool->Init(para_info->devicePhyId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    TopoType topoFlag = TopoType::TOPO_TYPE_8P_RING;
    if (para_info->userRank == 0) {
        para_info->comm_p2p.reset(new CommP2P(
            para_info->collectiveId, para_info->userRank, para_info->user_rank_size, para_info->rank,
            para_info->rank_size, topoFlag, para_info->dispatcher, notifyPool, netDevCtxMap, exchanger,
            para_info->para_vector, para_info->inputMem, para_info->outputMem, false, para_info->tag, 1));
    }

    if (para_info->userRank == 1) {
        para_info->comm_p2p.reset(new CommP2P(
            para_info->collectiveId, para_info->userRank, para_info->user_rank_size, para_info->rank,
            para_info->rank_size, topoFlag, para_info->dispatcher, notifyPool, netDevCtxMap, exchanger,
            para_info->para_vector, para_info->inputMem, para_info->outputMem, false,

            para_info->tag, 0));
    }

    ret = notifyPool->RegisterOp(para_info->tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = para_info->comm_p2p->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = notifyPool->UnregisterOp(para_info->tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclNetCloseDev(portCtx);
    HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, para_info->devicePhyId, para_info->devicePhyId);
    return (NULL);
}

void* comm_p2p_errtask_handle(void* para)
{
    s32 ret = HCCL_SUCCESS;
    innerpara_t_p2p* para_info = (innerpara_t_p2p*)para;

    hrtSetDevice(para_info->devicePhyId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
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
    para_info->comm_p2p.reset(new CommP2P(
        para_info->collectiveId, para_info->userRank, para_info->user_rank_size, para_info->rank, para_info->rank_size,
        topoFlag, para_info->dispatcher, nullptr, netDevCtxMap, exchanger, para_info->para_vector, para_info->inputMem,
        para_info->outputMem, false, para_info->tag, para_info->userRank));

    ret = para_info->comm_p2p->Init();
    EXPECT_NE(ret, HCCL_SUCCESS);

    HcclNetCloseDev(portCtx);
    HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, para_info->devicePhyId, para_info->devicePhyId);

    return (NULL);
}

void* comm_p2p_errtask_handle1(void* para)
{
    s32 ret = HCCL_SUCCESS;
    innerpara_t_p2p* para_info = (innerpara_t_p2p*)para;

    SocketListenInfoT sockListen;
    hrtSetDevice(para_info->devicePhyId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
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

    if (para_info->userRank == 0) {
        para_info->comm_p2p.reset(new CommP2P(
            para_info->collectiveId, para_info->userRank, para_info->user_rank_size, para_info->rank,
            para_info->rank_size, topoFlag, para_info->dispatcher, nullptr, netDevCtxMap, exchanger,
            para_info->para_vector, para_info->inputMem, para_info->outputMem, false,

            para_info->tag, 1));
    }

    if (para_info->userRank == 1) {
        para_info->comm_p2p.reset(new CommP2P(
            para_info->collectiveId, para_info->userRank, para_info->user_rank_size, para_info->rank,
            para_info->rank_size, topoFlag, para_info->dispatcher, nullptr, netDevCtxMap, exchanger,
            para_info->para_vector, para_info->inputMem, para_info->outputMem, false,

            para_info->tag, 0));
    }

    ret = para_info->comm_p2p->Init();
    EXPECT_NE(ret, HCCL_SUCCESS);

    HcclNetCloseDev(portCtx);
    HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, para_info->devicePhyId, para_info->devicePhyId);

    return (NULL);
}

TEST_F(CommP2PTest, ut_CommP2P_init)
{
    s32 ret = HCCL_SUCCESS;

    u32 userRank = 1;
    u32 user_rank_size = 5;

    RankInfo tmp_para;

    tmp_para.userRank = userRank;

    s32 device_id = 0;
    ret = hrtGetDevice(&device_id);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    tmp_para.devicePhyId = device_id;

    tmp_para.serverId = "10.21.78.208";
    tmp_para.nicIp.push_back(HcclIpAddress("10.21.78.208"));
    DeviceMem inputMem = DeviceMem::alloc(1024);
    DeviceMem outputMem = DeviceMem::alloc(1024);

    std::vector<RankInfo> para_vector;
    para_vector.push_back(tmp_para);

    u32 ifnumVersion = 3;
    MOCKER(hrtRaGetInterfaceVersion)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&ifnumVersion))
        .will(returnValue(HCCL_SUCCESS));

    char collectiveId[SAL_UNIQUE_ID_BYTES];
    ret = SalGetUniqueId(collectiveId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* dispatcher = nullptr;
    ret = HcclDispatcherInit(DispatcherType::DISPATCHER_NORMAL, 0, &dispatcher);
    EXPECT_NE(dispatcher, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    RaResourceInfo raResourceInfo;
    std::string collective_id_tmp = collectiveId;

    IntraExchanger exchanger{};

    TopoType topoFlag = TopoType::TOPO_TYPE_8P_RING;
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;
    CommP2P* comm_p2p = new CommP2P(
        collective_id_tmp, userRank, user_rank_size, 0, 1, topoFlag, dispatcher, nullptr, netDevCtxMap, exchanger,
        para_vector, inputMem, outputMem, false);

    ret = comm_p2p->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete comm_p2p;
    if (dispatcher != nullptr) {
        ret = HcclDispatcherDestroy(dispatcher);
        EXPECT_EQ(ret, HCCL_SUCCESS);
        dispatcher = nullptr;
    }
}

TEST_F(CommP2PTest, ut_CommP2P_init_err0)
{
    s32 ret = HCCL_SUCCESS;
    u32 ifnumVersion = 3;
    MOCKER(hrtRaGetInterfaceVersion)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&ifnumVersion))
        .will(returnValue(HCCL_SUCCESS));
    RankInfo tmp_para_0;
    tmp_para_0.userRank = 0;
    tmp_para_0.devicePhyId = 0;
    tmp_para_0.deviceType = DevType::DEV_TYPE_910;
    tmp_para_0.serverId = "10.21.78.208";
    tmp_para_0.nicIp.push_back(HcclIpAddress("10.21.78.208"));
    // tmp_para_0.inputMem = DeviceMem::alloc(1024);
    // tmp_para_0.outputMem = DeviceMem::alloc(1024);

    RankInfo tmp_para_1;
    tmp_para_1.userRank = 1;
    tmp_para_1.devicePhyId = 1;
    tmp_para_1.deviceType = DevType::DEV_TYPE_910;
    tmp_para_1.serverId = "10.21.78.208";
    tmp_para_1.nicIp.push_back(HcclIpAddress("10.21.78.209"));
    // tmp_para_1.inputMem = DeviceMem::alloc(1024);
    // tmp_para_1.outputMem = DeviceMem::alloc(1024);

    std::vector<RankInfo> para_vector;
    para_vector.push_back(tmp_para_0);
    para_vector.push_back(tmp_para_1);

    char collectiveId[SAL_UNIQUE_ID_BYTES];
    ret = SalGetUniqueId(collectiveId);

    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::string collective_id_tmp = collectiveId;

    const s32 dev_num = 2;
    s32 dev_list[dev_num] = {0, 1};
    std::vector<s32> device_ids(dev_list, dev_list + dev_num);

    std::vector<u32> ip_list;
    for (int i = 0; i < dev_num; i++) {
        u32 ipAddr = 0;
        (void)rt_get_dev_ip(0, i, &ipAddr);
        ip_list.push_back(ipAddr);
    }

    const u32 rank_list[dev_num] = {0, 1};
    std::vector<u32> user_ranks(rank_list, rank_list + dev_num);

    std::shared_ptr<CommP2P> comm_p2p = nullptr;

    sal_thread_t tid[dev_num];
    innerpara_t_p2p para_info[dev_num];
    s32 ndev = dev_num;
    std::shared_ptr<ProfilerManager> profilerManager[2] = {nullptr};

    for (s32 i = 0; i < ndev; i++) {
        para_info[i].collectiveId = collective_id_tmp;
        para_info[i].userRank = i;
        para_info[i].user_rank_size = ndev;
        para_info[i].rank = i;
        para_info[i].rank_size = ndev;
        para_info[i].devicePhyId = dev_list[i];
        para_info[i].dispatcher = get_p2p_dispatcher(i, profilerManager[i]);
        para_info[i].device_ids.assign(device_ids.begin(), device_ids.end());
        para_info[i].device_ips.assign(ip_list.begin(), ip_list.end());
        para_info[i].user_ranks.assign(user_ranks.begin(), user_ranks.end());
        para_info[i].tag = "ut_CommP2P_init_err0";
        para_info[i].para_vector = para_vector;
        para_info[i].inputMem = DeviceMem::alloc(1024);
        para_info[i].outputMem = DeviceMem::alloc(1024);
        para_info[i].comm_p2p = comm_p2p;
    }

    tid[0] = sal_thread_create("CommP2P rank0 thread err", comm_p2p_errtask_handle, (void*)&para_info[0]);

    tid[1] = sal_thread_create("CommP2P rank1 thread err", comm_p2p_errtask_handle, (void*)&para_info[1]);

    while (sal_thread_is_running(tid[0]) || sal_thread_is_running(tid[1])) {
        SaluSleep(SAL_MILLISECOND_USEC * 10);
        ;
    }

    for (s32 j = 0; j < ndev; j++) {
        ret = NetworkManager::GetInstance(dev_list[j]).Destroy();
        EXPECT_EQ(ret, HCCL_SUCCESS);
        (void)sal_thread_destroy(tid[j]);
    }

    // 显示释放资源
    for (s32 j = 0; j < ndev; j++) {
        para_info[j].comm_p2p.reset();
        if (para_info[j].dispatcher != nullptr) {
            ret = HcclDispatcherDestroy(para_info[j].dispatcher);
            EXPECT_EQ(ret, HCCL_SUCCESS);
            para_info[j].dispatcher = nullptr;
        }
    }
}

TEST_F(CommP2PTest, ut_CommP2P_init_err1)
{
    s32 ret = HCCL_SUCCESS;

    u32 ifnumVersion = 3;
    MOCKER(hrtRaGetInterfaceVersion)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&ifnumVersion))
        .will(returnValue(HCCL_SUCCESS));

    RankInfo tmp_para_0;
    tmp_para_0.userRank = 2;
    tmp_para_0.devicePhyId = 2;
    tmp_para_0.deviceType = DevType::DEV_TYPE_910;
    tmp_para_0.serverId = "10.21.78.208";
    tmp_para_0.nicIp.push_back(HcclIpAddress("10.21.78.208"));
    // tmp_para_0.inputMem = DeviceMem::alloc(1024);
    // tmp_para_0.outputMem = DeviceMem::alloc(1024);

    RankInfo tmp_para_1;
    tmp_para_1.userRank = 3;
    tmp_para_1.devicePhyId = 3;
    tmp_para_1.deviceType = DevType::DEV_TYPE_910;
    tmp_para_1.serverId = "10.21.78.208";
    tmp_para_1.nicIp.push_back(HcclIpAddress("10.21.78.209"));
    // tmp_para_1.inputMem = DeviceMem::alloc(1024);
    // tmp_para_1.outputMem = DeviceMem::alloc(1024);

    std::vector<RankInfo> para_vector;
    para_vector.push_back(tmp_para_0);
    para_vector.push_back(tmp_para_1);

    char collectiveId[SAL_UNIQUE_ID_BYTES];
    ret = SalGetUniqueId(collectiveId);

    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::string collective_id_tmp = collectiveId;

    const s32 dev_num = 2;
    s32 dev_list[dev_num] = {0, 1};
    std::vector<s32> device_ids(dev_list, dev_list + dev_num);

    std::vector<u32> ip_list;
    for (int i = 0; i < dev_num; i++) {
        u32 ipAddr = 0;
        (void)rt_get_dev_ip(0, i, &ipAddr);
        ip_list.push_back(ipAddr);
    }

    const u32 rank_list[dev_num] = {0, 1};
    std::vector<u32> user_ranks(rank_list, rank_list + dev_num);

    std::shared_ptr<CommP2P> comm_p2p = nullptr;

    sal_thread_t tid[dev_num];
    innerpara_t_p2p para_info[dev_num];
    s32 ndev = dev_num;
    std::shared_ptr<ProfilerManager> profilerManager[2] = {nullptr};
    for (s32 i = 0; i < ndev; i++) {
        para_info[i].collectiveId = collective_id_tmp;
        para_info[i].userRank = i;
        para_info[i].user_rank_size = ndev;
        para_info[i].rank = i;
        para_info[i].rank_size = ndev;
        para_info[i].devicePhyId = dev_list[i];
        para_info[i].dispatcher = get_p2p_dispatcher(i, profilerManager[i]);
        para_info[i].device_ids.assign(device_ids.begin(), device_ids.end());
        para_info[i].device_ips.assign(ip_list.begin(), ip_list.end());
        para_info[i].user_ranks.assign(user_ranks.begin(), user_ranks.end());
        para_info[i].tag = "ut_CommP2P_init_err1";
        para_info[i].para_vector = para_vector;
        para_info[i].inputMem = DeviceMem::alloc(1024);
        para_info[i].outputMem = DeviceMem::alloc(1024);
        para_info[i].comm_p2p = comm_p2p;
    }

    tid[0] = sal_thread_create("CommP2P rank0 thread err1", comm_p2p_errtask_handle, (void*)&para_info[0]);

    tid[1] = sal_thread_create("CommP2P rank1 thread err1", comm_p2p_errtask_handle, (void*)&para_info[1]);

    while (sal_thread_is_running(tid[0]) || sal_thread_is_running(tid[1])) {
        SaluSleep(SAL_MILLISECOND_USEC * 10);
        ;
    }

    for (s32 j = 0; j < ndev; j++) {
        ret = NetworkManager::GetInstance(dev_list[j]).Destroy();
        EXPECT_EQ(ret, HCCL_SUCCESS);
        (void)sal_thread_destroy(tid[j]);
    }

    // 显示释放资源
    for (s32 j = 0; j < ndev; j++) {
        para_info[j].comm_p2p.reset();
        if (para_info[j].dispatcher != nullptr) {
            ret = HcclDispatcherDestroy(para_info[j].dispatcher);
            EXPECT_EQ(ret, HCCL_SUCCESS);
            para_info[j].dispatcher = nullptr;
        }
    }
}

TEST_F(CommP2PTest, ut_CommP2P_init_err2)
{
    s32 ret = HCCL_SUCCESS;

    u32 ifnumVersion = 3;
    MOCKER(hrtRaGetInterfaceVersion)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&ifnumVersion))
        .will(returnValue(HCCL_SUCCESS));

    RankInfo tmp_para_0;
    tmp_para_0.userRank = 0;
    tmp_para_0.devicePhyId = 0;
    tmp_para_0.deviceType = DevType::DEV_TYPE_910;
    tmp_para_0.serverId = "10.21.78.208";
    tmp_para_0.nicIp.push_back(HcclIpAddress("10.21.78.208"));
    // tmp_para_0.inputMem = DeviceMem::alloc(1024);
    // tmp_para_0.outputMem = DeviceMem::alloc(1024);

    RankInfo tmp_para_1;
    tmp_para_1.devicePhyId = 1;
    tmp_para_1.deviceType = DevType::DEV_TYPE_910;
    tmp_para_1.serverId = "10.21.78.208";
    tmp_para_1.nicIp.push_back(HcclIpAddress("10.21.78.209"));
    // tmp_para_1.inputMem = DeviceMem::alloc(1024);
    // tmp_para_1.outputMem = DeviceMem::alloc(1024);

    std::vector<RankInfo> para_vector;
    para_vector.push_back(tmp_para_0);
    para_vector.push_back(tmp_para_1);

    char collectiveId[SAL_UNIQUE_ID_BYTES];
    ret = SalGetUniqueId(collectiveId);

    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::string collective_id_tmp = collectiveId;

    const s32 dev_num = 2;
    s32 dev_list[dev_num] = {0, 1};
    std::vector<s32> device_ids(dev_list, dev_list + dev_num);

    std::vector<u32> ip_list;
    for (int i = 0; i < dev_num; i++) {
        u32 ipAddr = 0;
        (void)rt_get_dev_ip(0, i, &ipAddr);
        ip_list.push_back(ipAddr);
    }

    const u32 rank_list[dev_num] = {0, 1};
    std::vector<u32> user_ranks(rank_list, rank_list + dev_num);

    std::shared_ptr<CommP2P> comm_p2p = nullptr;

    sal_thread_t tid[dev_num];
    innerpara_t_p2p para_info[dev_num];
    s32 ndev = dev_num;
    std::shared_ptr<ProfilerManager> profilerManager[2] = {nullptr};
    for (s32 i = 0; i < ndev; i++) {
        para_info[i].collectiveId = collective_id_tmp;
        para_info[i].userRank = i;
        para_info[i].user_rank_size = ndev;
        para_info[i].rank = 3;
        para_info[i].rank_size = ndev;
        para_info[i].devicePhyId = dev_list[i];
        para_info[i].dispatcher = get_p2p_dispatcher(i, profilerManager[i]);
        para_info[i].device_ids.assign(device_ids.begin(), device_ids.end());
        para_info[i].device_ips.assign(ip_list.begin(), ip_list.end());
        para_info[i].user_ranks.assign(user_ranks.begin(), user_ranks.end());
        para_info[i].tag = "ut_CommP2P_init_err1";
        para_info[i].para_vector = para_vector;
        para_info[i].inputMem = DeviceMem::alloc(1024);
        para_info[i].outputMem = DeviceMem::alloc(1024);
        para_info[i].comm_p2p = comm_p2p;
    }

    tid[0] = sal_thread_create("CommP2P rank0 thread err1", comm_p2p_errtask_handle, (void*)&para_info[0]);

    tid[1] = sal_thread_create("CommP2P rank1 thread err1", comm_p2p_errtask_handle, (void*)&para_info[1]);

    while (sal_thread_is_running(tid[0]) || sal_thread_is_running(tid[1])) {
        SaluSleep(SAL_MILLISECOND_USEC * 10);
        ;
    }

    for (s32 j = 0; j < ndev; j++) {
        ret = NetworkManager::GetInstance(dev_list[j]).Destroy();
        EXPECT_EQ(ret, HCCL_SUCCESS);
        (void)sal_thread_destroy(tid[j]);
    }
    // 显示释放资源
    for (s32 j = 0; j < ndev; j++) {
        para_info[j].comm_p2p.reset();
        if (para_info[j].dispatcher != nullptr) {
            ret = HcclDispatcherDestroy(para_info[j].dispatcher);
            EXPECT_EQ(ret, HCCL_SUCCESS);
            para_info[j].dispatcher = nullptr;
        }
    }
}

TEST_F(CommP2PTest, ut_CommP2P_init_err3)
{
    s32 ret = HCCL_SUCCESS;
    MOCKER(hrtRaGetInterfaceVersion).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    RankInfo tmp_para_0;
    tmp_para_0.userRank = 0;
    tmp_para_0.devicePhyId = 0;
    tmp_para_0.deviceType = DevType::DEV_TYPE_910;
    tmp_para_0.serverId = "10.21.78.208";
    tmp_para_0.nicIp.push_back(HcclIpAddress("10.21.78.208"));
    // tmp_para_0.inputMem = DeviceMem::alloc(1024);
    // tmp_para_0.outputMem = DeviceMem::alloc(1024);

    RankInfo tmp_para_1;
    tmp_para_1.userRank = 1;
    tmp_para_1.devicePhyId = 1;
    tmp_para_1.deviceType = DevType::DEV_TYPE_910;
    tmp_para_1.serverId = "10.21.78.209";
    tmp_para_1.nicIp.push_back(HcclIpAddress("10.21.78.209"));
    // tmp_para_1.inputMem = DeviceMem::alloc(1024);
    // tmp_para_1.outputMem = DeviceMem::alloc(1024);

    std::vector<RankInfo> para_vector;
    para_vector.push_back(tmp_para_0);
    para_vector.push_back(tmp_para_1);

    char collectiveId[SAL_UNIQUE_ID_BYTES];
    ret = SalGetUniqueId(collectiveId);

    std::string collective_id_tmp = collectiveId;

    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    const s32 dev_num = 2;
    s32 dev_list[dev_num] = {0, 1};
    std::vector<s32> device_ids(dev_list, dev_list + dev_num);

    std::vector<u32> ip_list;
    for (int i = 0; i < dev_num; i++) {
        u32 ipAddr = 0;
        (void)rt_get_dev_ip(0, i, &ipAddr);
        ip_list.push_back(ipAddr);
    }

    const u32 rank_list[dev_num] = {0, 1};
    std::vector<u32> user_ranks(rank_list, rank_list + dev_num);

    std::shared_ptr<CommP2P> comm_p2p = nullptr;

    sal_thread_t tid[dev_num];
    innerpara_t_p2p para_info[dev_num];
    s32 ndev = dev_num;
    std::shared_ptr<ProfilerManager> profilerManager[2] = {nullptr};
    for (s32 i = 0; i < ndev; i++) {
        para_info[i].collectiveId = collective_id_tmp;
        para_info[i].userRank = i;
        para_info[i].user_rank_size = ndev;
        para_info[i].rank = i;
        para_info[i].rank_size = ndev;
        para_info[i].devicePhyId = dev_list[i];
        para_info[i].dispatcher = get_p2p_dispatcher(i, profilerManager[i]);
        para_info[i].device_ids.assign(device_ids.begin(), device_ids.end());
        para_info[i].device_ips.assign(ip_list.begin(), ip_list.end());
        para_info[i].user_ranks.assign(user_ranks.begin(), user_ranks.end());
        para_info[i].tag = "ut_CommP2P_init_err1";
        para_info[i].para_vector = para_vector;
        para_info[i].inputMem = DeviceMem::alloc(1024);
        para_info[i].outputMem = DeviceMem::alloc(1024);
        para_info[i].comm_p2p = comm_p2p;
    }

    tid[0] = sal_thread_create("CommP2P rank0 thread err1", comm_p2p_errtask_handle1, (void*)&para_info[0]);

    tid[1] = sal_thread_create("CommP2P rank1 thread err1", comm_p2p_errtask_handle1, (void*)&para_info[1]);

    while (sal_thread_is_running(tid[0]) || sal_thread_is_running(tid[1])) {
        SaluSleep(SAL_MILLISECOND_USEC * 10);
        ;
    }

    for (s32 j = 0; j < ndev; j++) {
        ret = NetworkManager::GetInstance(dev_list[j]).Destroy();
        EXPECT_EQ(ret, HCCL_SUCCESS);
        (void)sal_thread_destroy(tid[j]);
    }

    // 显示释放资源
    for (s32 j = 0; j < ndev; j++) {
        para_info[j].comm_p2p.reset();
        if (para_info[j].dispatcher != nullptr) {
            ret = HcclDispatcherDestroy(para_info[j].dispatcher);
            EXPECT_EQ(ret, HCCL_SUCCESS);
            para_info[j].dispatcher = nullptr;
        }
    }
}

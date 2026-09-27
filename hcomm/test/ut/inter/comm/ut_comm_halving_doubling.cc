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
#include "comm_halving_doubling_pub.h"
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

using namespace std;
using namespace hccl;

class CommBinaryBlocksHDTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--CommBinaryBlocksHDTest SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--CommBinaryBlocksHDTest TearDown--\033[0m" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        TsdOpen(1, 2);
        DlTdtFunction::GetInstance().DlTdtFunctionInit();
        // TsdOpen(1, 2);
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

class CommBinaryBlocksHDTmp : public CommHalvingDoubling {
public:
    explicit CommBinaryBlocksHDTmp(
        const std::string& collectiveId, const u32 userRank, const u32 user_rank_size, const u32 rank,
        const u32 rank_size, const TopoType topoFlag, const HcclDispatcher dispatcher,
        std::map<HcclIpAddress, HcclNetDevCtx>& netDevCtxMap, const IntraExchanger& exchanger,
        const std::vector<RankInfo> para_vector, const DeviceMem& inputMem, const DeviceMem& outputMem,
        const u64 comm_attribute = 0, const std::string& tag = "");
    virtual ~CommBinaryBlocksHDTmp();
};

CommBinaryBlocksHDTmp::CommBinaryBlocksHDTmp(
    const std::string& collectiveId, const u32 userRank, const u32 user_rank_size, const u32 rank, const u32 rank_size,
    const TopoType topoFlag, const HcclDispatcher dispatcher, std::map<HcclIpAddress, HcclNetDevCtx>& netDevCtxMap,
    const IntraExchanger& exchanger, const std::vector<RankInfo> para_vector, const DeviceMem& inputMem,
    const DeviceMem& outputMem, const u64 comm_attribute, const std::string& tag)
    : CommHalvingDoubling(
          collectiveId, userRank, user_rank_size, rank, rank_size, topoFlag, dispatcher, nullptr, netDevCtxMap,
          exchanger, para_vector, inputMem, outputMem, true, tag)
{}

CommBinaryBlocksHDTmp::~CommBinaryBlocksHDTmp() {}

typedef struct innerpara_struct_hd {
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
    std::shared_ptr<CommBinaryBlocksHDTmp> comm_binary_blocks_H_D;
} innerpara_t_hd;

HcclDispatcher get_H_D_dispatcher(s32 devid, std::shared_ptr<hccl::ProfilerManager>& profilerManager)
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

void* comm_binary_blocks_H_D_task_handle(void* para)
{
    HcclResult ret = HCCL_SUCCESS;
    innerpara_t_hd* para_info = (innerpara_t_hd*)para;
    s32 rt_ret = 0;
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
    para_info->comm_binary_blocks_H_D.reset(new CommBinaryBlocksHDTmp(
        para_info->collectiveId, para_info->userRank, para_info->user_rank_size, para_info->rank, para_info->rank_size,
        topoFlag, para_info->dispatcher, netDevCtxMap, exchanger, para_info->para_vector, para_info->inputMem,
        para_info->outputMem));

    // ret = para_info->comm_binary_blocks_H_D->Init();
    // EXPECT_EQ(ret, HCCL_SUCCESS);
    HcclNetCloseDev(portCtx);
    HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, para_info->devicePhyId, para_info->devicePhyId);
    return (NULL);
}

void* comm_H_D_task_handle(void* para)
{
    HcclResult ret = HCCL_SUCCESS;
    innerpara_t_hd* para_info = (innerpara_t_hd*)para;
    s32 rt_ret = 0;
    u32 port = 16666;

    hrtSetDevice(para_info->devicePhyId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = NetworkManager::GetInstance(para_info->devicePhyId).Init(NICDeployment::NIC_DEPLOYMENT_DEVICE);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = NetworkManager::GetInstance(para_info->devicePhyId).StartVnic(HcclIpAddress(para_info->devicePhyId), port);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    IntraExchanger exchanger{};

    std::unique_ptr<NotifyPool> notifyPool = nullptr;
    notifyPool.reset(new (std::nothrow) NotifyPool());
    EXPECT_NE(notifyPool, nullptr);
    ret = notifyPool->Init(para_info->devicePhyId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    TopoType topoFlag = TopoType::TOPO_TYPE_8P_RING;
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;
    para_info->comm_binary_blocks_H_D.reset(new CommBinaryBlocksHDTmp(
        para_info->collectiveId, para_info->userRank, para_info->user_rank_size, para_info->rank, para_info->rank_size,
        topoFlag, para_info->dispatcher, netDevCtxMap, exchanger, para_info->para_vector, para_info->inputMem,
        para_info->outputMem, 6));

    ret = notifyPool->RegisterOp(para_info->tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = para_info->comm_binary_blocks_H_D->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = notifyPool->UnregisterOp(para_info->tag);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    return (NULL);
}

extern std::unique_ptr<NotifyPool> get_notifyPool(s32 rank);
TEST_F(CommBinaryBlocksHDTest, ut_comm_B_B_H_D_init_3_thread_sameip)
{
    s32 ret = HCCL_SUCCESS;

    MOCKER(hrtRaGetInterfaceVersion).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));

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

    std::vector<RankInfo> para_vector;
    para_vector.push_back(tmp_para_0);
    para_vector.push_back(tmp_para_1);
    para_vector.push_back(tmp_para_2);

    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    char collectiveId[SAL_UNIQUE_ID_BYTES];
    ret = SalGetUniqueId(collectiveId);

    std::string collective_id_tmp = collectiveId;

    const s32 dev_num = 3;
    s32 dev_list[dev_num] = {0, 1, 2};

    std::vector<s32> device_ids(dev_list, dev_list + dev_num);

    std::vector<u32> ip_list;
    for (int i = 0; i < dev_num; i++) {
        u32 ipAddr = 0;
        (void)rt_get_dev_ip(0, i, &ipAddr);
        ip_list.push_back(ipAddr);
    }

    const u32 rank_list[dev_num] = {0, 1, 2};
    std::vector<u32> user_ranks(rank_list, rank_list + dev_num);

    std::shared_ptr<CommBinaryBlocksHDTmp> comm_binary_blocks_H_D = nullptr;
    std::shared_ptr<ProfilerManager> profilerManager[3] = {nullptr};
    sal_thread_t tid[3];
    innerpara_t_hd para_info[3];
    s32 ndev = 3;

    for (s32 i = 0; i < ndev; i++) {
        para_info[i].collectiveId = collective_id_tmp;
        para_info[i].userRank = i;
        para_info[i].user_rank_size = ndev;
        para_info[i].rank = i;
        para_info[i].rank_size = ndev;
        para_info[i].devicePhyId = dev_list[i];
        para_info[i].dispatcher = get_H_D_dispatcher(i, profilerManager[i]);
        para_info[i].notifyPool = get_notifyPool(i);
        para_info[i].device_ids.assign(device_ids.begin(), device_ids.end());
        para_info[i].device_ips.assign(ip_list.begin(), ip_list.end());
        para_info[i].user_ranks.assign(user_ranks.begin(), user_ranks.end());
        para_info[i].tag = "st_comm_B_B_H_D_init_3_thread_sameip";
        para_info[i].para_vector = para_vector;
        para_info[i].inputMem = DeviceMem::alloc(1024);
        para_info[i].outputMem = DeviceMem::alloc(1024);
        para_info[i].comm_binary_blocks_H_D = comm_binary_blocks_H_D;
    }

    tid[0]
        = sal_thread_create("HalvingDoubling rank0 thread", comm_binary_blocks_H_D_task_handle, (void*)&para_info[0]);

    tid[1]
        = sal_thread_create("HalvingDoubling rank1 thread", comm_binary_blocks_H_D_task_handle, (void*)&para_info[1]);

    tid[2]
        = sal_thread_create("HalvingDoubling rank2 thread", comm_binary_blocks_H_D_task_handle, (void*)&para_info[2]);

    while (sal_thread_is_running(tid[1]) || sal_thread_is_running(tid[2]) || sal_thread_is_running(tid[0])) {
        SaluSleep(SAL_MILLISECOND_USEC * 10);
        ;
    }

    for (s32 j = 0; j < ndev; j++) {
        ret = NetworkManager::GetInstance(dev_list[j]).Destroy();
        EXPECT_EQ(ret, HCCL_SUCCESS);
        (void)sal_thread_destroy(tid[j]);
        if (para_info[j].dispatcher != nullptr) {
            ret = HcclDispatcherDestroy(para_info[j].dispatcher);
            EXPECT_EQ(ret, HCCL_SUCCESS);
            para_info[j].dispatcher = nullptr;
        }
    }
}

TEST_F(CommBinaryBlocksHDTest, ut_comm_B_B_H_D_init_1_rank)
{
    s32 ret = HCCL_SUCCESS;

    RankInfo tmp_para_0;
    tmp_para_0.userRank = 0;
    tmp_para_0.devicePhyId = 0;
    tmp_para_0.deviceType = DevType::DEV_TYPE_910;
    tmp_para_0.serverId = "10.21.78.208";
    tmp_para_0.nicIp.push_back(HcclIpAddress("10.21.78.208"));
    // tmp_para_0.inputMem = DeviceMem::alloc(128*3);
    // tmp_para_0.outputMem = DeviceMem::alloc(128*3);

    std::vector<RankInfo> para_vector;
    para_vector.push_back(tmp_para_0);

    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    char collectiveId[SAL_UNIQUE_ID_BYTES];
    ret = SalGetUniqueId(collectiveId);

    std::string collective_id_tmp = collectiveId;

    const s32 dev_num = 1;
    s32 dev_list[dev_num] = {0};
    std::vector<s32> device_ids(dev_list, dev_list + dev_num);

    std::vector<u32> ip_list;
    for (int i = 0; i < dev_num; i++) {
        u32 ipAddr = 0;
        (void)rt_get_dev_ip(0, i, &ipAddr);
        ip_list.push_back(ipAddr);
    }

    const u32 rank_list[dev_num] = {0};
    std::shared_ptr<ProfilerManager> profilerManager[1] = {nullptr};
    ;
    std::vector<u32> user_ranks(rank_list, rank_list + dev_num);

    std::shared_ptr<CommBinaryBlocksHDTmp> comm_binary_blocks_H_D = nullptr;

    sal_thread_t tid[1];
    innerpara_t_hd para_info[1];
    s32 ndev = 1;

    for (s32 i = 0; i < ndev; i++) {
        para_info[i].collectiveId = collective_id_tmp;
        para_info[i].userRank = i;
        para_info[i].user_rank_size = ndev;
        para_info[i].rank = i;
        para_info[i].rank_size = ndev;
        para_info[i].devicePhyId = dev_list[i];
        para_info[i].dispatcher = get_H_D_dispatcher(i, profilerManager[i]);
        para_info[i].notifyPool = get_notifyPool(i);
        para_info[i].device_ids.assign(device_ids.begin(), device_ids.end());
        para_info[i].device_ips.assign(ip_list.begin(), ip_list.end());
        para_info[i].user_ranks.assign(user_ranks.begin(), user_ranks.end());
        para_info[i].tag = "st_comm_B_B_H_D_init_1_rank";
        para_info[i].para_vector = para_vector;
        para_info[i].inputMem = DeviceMem::alloc(1024);
        para_info[i].outputMem = DeviceMem::alloc(1024);
        para_info[i].comm_binary_blocks_H_D = comm_binary_blocks_H_D;
    }

    tid[0] = sal_thread_create("HalvingDoubling rank0 thread", comm_H_D_task_handle, (void*)&para_info[0]);

    while (sal_thread_is_running(tid[0])) {
        SaluSleep(SAL_MILLISECOND_USEC * 10);
        ;
    }

    for (s32 j = 0; j < ndev; j++) {
        ret = NetworkManager::GetInstance(dev_list[j]).Destroy();
        EXPECT_EQ(ret, HCCL_SUCCESS);
        (void)sal_thread_destroy(tid[j]);
        if (para_info[j].dispatcher != nullptr) {
            ret = HcclDispatcherDestroy(para_info[j].dispatcher);
            EXPECT_EQ(ret, HCCL_SUCCESS);
            para_info[j].dispatcher = nullptr;
        }
    }
}

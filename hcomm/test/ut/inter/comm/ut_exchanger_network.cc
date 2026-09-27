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
#include <stdio.h>
#include <mockcpp/mockcpp.hpp>
#include "hccl/base.h"
#include "hccl_comm_pub.h"
#include "llt_hccl_stub_pub.h"
#include "llt_hccl_stub.h"
#include "sal.h"
#include "dlra_function.h"
#include "exchanger_network_pub.h"
#include "network_manager_pub.h"
#include "tsd/tsd_client.h"
#include "dltdt_function.h"
using namespace std;
using namespace hccl;

class ExchangerNetworkTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--ExchangerNetworkTest SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--ExchangerNetworkTest TearDown--\033[0m" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        static s32 call_cnt = 0;
        DlTdtFunction::GetInstance().DlTdtFunctionInit();
        TsdOpen(1, 2);
        string name = std::to_string(call_cnt++) + "_" + __PRETTY_FUNCTION__;
        ra_set_shm_name(name.c_str());
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        TsdClose(1);
        std::cout << "A Test TearDown" << std::endl;
    }
};

HcclResult stub_HrtRaRdmaInitWithAttr(struct RdevInitInfo init_info, struct rdev rdevInfo, RdmaHandle& rdmaHandle)
{
    int val = 0;
    rdmaHandle = &val;
    return HCCL_SUCCESS;
}
TEST_F(ExchangerNetworkTest, ut_init_rdma)
{
    MOCKER(HrtRaRdmaInitWithAttr).stubs().with(mockcpp::any()).will(invoke(stub_HrtRaRdmaInitWithAttr));

    MOCKER(HrtRaRdmaInit).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    s32 ret = HCCL_SUCCESS;
    HcclIpAddress ipAddr;

    int val = 0;
    SocketHandle socketHandle = &val;
    IpSocket ipSocketInfo;
    ipSocketInfo.nicSocketHandle = socketHandle;

    NetworkManager::GetInstance(1).raResourceInfo_.nicSocketMap.insert(std::make_pair(ipAddr, ipSocketInfo));
    NetworkManager::GetInstance(1).raResourceInfo_.hostNetSocketMap.insert(std::make_pair(ipAddr, ipSocketInfo));

    ret = NetworkManager::GetInstance(1).InitRdmaHandle(1, ipAddr, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(ExchangerNetworkTest, ut_init_1_dev)
{
    s32 ret = HCCL_SUCCESS;

    s32 userRank = 0;
    const u32 user_rank_size = 1;
    const s32 dev_num = 1;
    u32 port = 16666;

    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::string collectiveId("default");
    s32 dev_list[dev_num] = {0};
    std::vector<s32> device_ids(dev_list, dev_list + 1);

    u32 device_id = 0, localIp = 0, idx = 0;
    hrtSetDevice(device_id);
    ret = NetworkManager::GetInstance(device_id).Init(NICDeployment::NIC_DEPLOYMENT_DEVICE);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = NetworkManager::GetInstance(device_id).StartVnic(HcclIpAddress(device_id), port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    const u32 rank_list[dev_num] = {0};
    std::vector<u32> user_ranks(rank_list, rank_list + 1);
    const std::string tag("default_tag");
    ExchangerNetwork exchangernetwork(collectiveId, userRank, user_rank_size);

    ret = exchangernetwork.AppendSockets(device_ids, user_ranks);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = NetworkManager::GetInstance(device_id).Destroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

typedef struct ExchangerNetworkpara_struct {
    std::string collectiveId;
    u32 userRank;
    u32 user_rank_size;
    u32 devicePhyId;
    std::shared_ptr<ExchangerNetwork> exchanger_network;
    std::vector<s32> device_ids;
    std::vector<u32> device_ips;
    std::vector<u32> user_ranks;
    std::string tag;
    s32* ret_value;
    std::shared_ptr<std::string> send_str;
    std::shared_ptr<std::string> recv_str;
    u32 send_rank;
    u32 send_to;
    u32 recv_rank;
    u32 recv_from;
} ExchangerNetworkpara_t;

void* exchanger_network_task_handle(void* para)
{
    HcclResult ret = HCCL_SUCCESS;
    s32 rt_ret = 0;
    u32 port = 16666;

    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ExchangerNetworkpara_t* para_info = (ExchangerNetworkpara_t*)para;
    hrtSetDevice(para_info->devicePhyId);
    /* 作为socket server端启动监听 */
    ret = NetworkManager::GetInstance(para_info->devicePhyId).Init(NICDeployment::NIC_DEPLOYMENT_DEVICE);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = NetworkManager::GetInstance(para_info->devicePhyId).StartVnic(HcclIpAddress(para_info->devicePhyId), port);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    para_info->exchanger_network.reset(
        new ExchangerNetwork(para_info->collectiveId, para_info->userRank, para_info->user_rank_size));
    *(para_info->ret_value) = para_info->exchanger_network->AppendSockets(para_info->device_ids, para_info->user_ranks);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("exchanger_network init failed");
        *(para_info->ret_value) = HCCL_E_INTERNAL;
        return (NULL);
    }

    /* 延时等待建链成功 */
    SaluSleep(1000 * 1000);

    if (para_info->send_str != nullptr && para_info->send_rank == para_info->userRank) {
        s32 dest_rank = para_info->recv_rank;
        ret = para_info->exchanger_network->Send(para_info->send_to, *(para_info->send_str));
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("send fail");
            *(para_info->ret_value) = HCCL_E_INTERNAL;
            return (NULL);
        }
    }

    if (para_info->recv_str != nullptr && para_info->recv_rank == para_info->userRank) {
        s32 src_rank = para_info->send_rank;
        ret = para_info->exchanger_network->Recv(para_info->recv_from, *(para_info->recv_str));
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("receive fail");
            *(para_info->ret_value) = HCCL_E_INTERNAL;
            return (NULL);
        }
    }
    return (NULL);
}
#if 1
TEST_F(ExchangerNetworkTest, ut_thread_init_2_dev)
{
    s32 ret = HCCL_SUCCESS;
    HcclRootInfo commId;

    ret = hcclComm::GetUniqueId(&commId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

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

    const u32 rank_list[dev_num] = {1, 0};
    std::vector<u32> user_ranks(rank_list, rank_list + dev_num);

    sal_thread_t tid[dev_num];
    ExchangerNetworkpara_t para_info[dev_num];

    for (s32 i = 0; i < dev_num; i++) {
        para_info[i].collectiveId = collective_id_tmp;
        para_info[i].userRank = rank_list[i];
        para_info[i].user_rank_size = dev_num;
        para_info[i].devicePhyId = dev_list[i];
        para_info[i].exchanger_network = nullptr;
        para_info[i].device_ids.assign(device_ids.begin(), device_ids.end());
        para_info[i].device_ips.assign(ip_list.begin(), ip_list.end());
        para_info[i].user_ranks.assign(user_ranks.begin(), user_ranks.end());
        para_info[i].tag = "default_tag";
        para_info[i].send_str = nullptr;
        para_info[i].recv_str = nullptr;
        para_info[i].send_rank = 0;
        para_info[i].send_to = 0;
        para_info[i].recv_rank = 0;
        para_info[i].recv_from = 0;
        para_info[i].ret_value = &ret;
    }

    tid[0] = sal_thread_create("exchanger_network rank0 thread", exchanger_network_task_handle, (void*)&para_info[0]);
    tid[1] = sal_thread_create("exchanger_network rank1 thread", exchanger_network_task_handle, (void*)&para_info[1]);

    while (sal_thread_is_running(tid[0]) || sal_thread_is_running(tid[1])) {
        SaluSleep(SAL_MILLISECOND_USEC * 10);
    }

    for (s32 j = 0; j < dev_num; j++) {
        NetworkManager::GetInstance(dev_list[j]).Destroy();
        EXPECT_EQ(*(para_info[j].ret_value), HCCL_SUCCESS);
        (void)sal_thread_destroy(tid[j]);
    }
}
#endif

#if 1
TEST_F(ExchangerNetworkTest, 2_thread_send_receive)
{
    s32 ret;
    HcclRootInfo commId;

    ret = hcclComm::GetUniqueId(&commId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    char collectiveId[SAL_UNIQUE_ID_BYTES];
    ret = SalGetUniqueId(collectiveId);
    std::string collective_id_tmp = collectiveId;

    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    const s32 ndev = 2;
    s32 dev_list[ndev] = {0, 1};
    std::vector<s32> device_ids(dev_list, dev_list + ndev);

    std::vector<u32> ip_list;
    for (int i = 0; i < ndev; i++) {
        u32 ipAddr = 0;
        (void)rt_get_dev_ip(0, i, &ipAddr);
        ip_list.push_back(ipAddr);
    }

    const u32 rank_list[ndev] = {1, 0};
    std::vector<u32> user_ranks(rank_list, rank_list + ndev);

    sal_thread_t tid[ndev];
    ExchangerNetworkpara_t para_info[ndev];
    const s32 send_rank = 0;
    const s32 recv_rank = 1;
    const s32 buff_size = 100;
    std::shared_ptr<std::string> send_str;
    std::shared_ptr<std::string> recv_str;
    send_str.reset(new string("test_str"));
    recv_str.reset(new string(""));

    for (s32 i = 0; i < ndev; i++) {
        para_info[i].collectiveId = collective_id_tmp;
        para_info[i].userRank = rank_list[i];
        para_info[i].user_rank_size = ndev;
        para_info[i].devicePhyId = dev_list[i];
        para_info[i].exchanger_network = nullptr;
        para_info[i].device_ids.assign(device_ids.begin(), device_ids.end());
        para_info[i].device_ips.assign(ip_list.begin(), ip_list.end());
        para_info[i].user_ranks.assign(user_ranks.begin(), user_ranks.end());
        para_info[i].tag = "default_tag";
        para_info[i].send_str = send_str;
        para_info[i].recv_str = recv_str;
        para_info[i].send_rank = send_rank;
        para_info[i].send_to = recv_rank;
        para_info[i].recv_rank = recv_rank;
        para_info[i].recv_from = send_rank;
        para_info[i].ret_value = &ret;
    }

    tid[0] = sal_thread_create("exchanger_network rank0 thread", exchanger_network_task_handle, (void*)&para_info[0]);
    tid[1] = sal_thread_create("exchanger_network rank1 thread", exchanger_network_task_handle, (void*)&para_info[1]);

    while (sal_thread_is_running(tid[0]) || sal_thread_is_running(tid[1])) {
        SaluSleep(SAL_MILLISECOND_USEC * 10);
    }

    for (s32 j = 0; j < ndev; j++) {
        NetworkManager::GetInstance(dev_list[j]).Destroy();
        EXPECT_EQ(*(para_info[j].ret_value), HCCL_SUCCESS);
        (void)sal_thread_destroy(tid[j]);
    }

    ret = send_str->compare(*recv_str);

    EXPECT_EQ(ret, 0);
}
#endif

#if 1
TEST_F(ExchangerNetworkTest, 8_thread_1send_1receive)
{
    s32 ret = 0;
    HcclRootInfo commId;

    ret = hcclComm::GetUniqueId(&commId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    char collectiveId[SAL_UNIQUE_ID_BYTES];
    ret = SalGetUniqueId(collectiveId);
    std::string collective_id_tmp = collectiveId;

    const s32 ndev = 8;
    s32 dev_list[ndev] = {1, 0, 2, 3, 4, 5, 6, 7};
    std::vector<s32> device_ids(dev_list, dev_list + ndev);

    std::vector<u32> ip_list;
    for (int i = 0; i < ndev; i++) {
        u32 ipAddr = 0;
        (void)rt_get_dev_ip(0, i, &ipAddr);
        ip_list.push_back(ipAddr);
    }

    const u32 rank_list[ndev] = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<u32> user_ranks(rank_list, rank_list + ndev);

    sal_thread_t tid[ndev];
    ExchangerNetworkpara_t para_info[ndev];
    const s32 send_rank = 0;
    const s32 recv_rank = 4;
    std::shared_ptr<std::string> send_str;
    std::shared_ptr<std::string> recv_str;
    send_str.reset(new string("test_str"));
    recv_str.reset(new string(""));

    for (s32 i = 0; i < ndev; i++) {
        para_info[i].collectiveId = collective_id_tmp;
        para_info[i].userRank = i;
        para_info[i].user_rank_size = ndev;
        para_info[i].devicePhyId = dev_list[i];
        para_info[i].exchanger_network = nullptr;
        para_info[i].device_ids.assign(device_ids.begin(), device_ids.end());
        para_info[i].device_ips.assign(ip_list.begin(), ip_list.end());
        para_info[i].user_ranks.assign(user_ranks.begin(), user_ranks.end());
        para_info[i].tag = "default_tag";
        para_info[i].send_str = send_str;
        para_info[i].recv_str = recv_str;
        para_info[i].send_rank = send_rank;
        para_info[i].send_to = recv_rank;
        para_info[i].recv_rank = recv_rank;
        para_info[i].recv_from = send_rank;
        para_info[i].ret_value = &ret;
    }

    tid[0] = sal_thread_create("exchanger_network rank0 thread", exchanger_network_task_handle, (void*)&para_info[0]);
    tid[1] = sal_thread_create("exchanger_network rank1 thread", exchanger_network_task_handle, (void*)&para_info[1]);
    tid[2] = sal_thread_create("exchanger_network rank2 thread", exchanger_network_task_handle, (void*)&para_info[2]);
    tid[3] = sal_thread_create("exchanger_network rank3 thread", exchanger_network_task_handle, (void*)&para_info[3]);
    tid[4] = sal_thread_create("exchanger_network rank4 thread", exchanger_network_task_handle, (void*)&para_info[4]);
    tid[5] = sal_thread_create("exchanger_network rank5 thread", exchanger_network_task_handle, (void*)&para_info[5]);
    tid[6] = sal_thread_create("exchanger_network rank6 thread", exchanger_network_task_handle, (void*)&para_info[6]);
    tid[7] = sal_thread_create("exchanger_network rank7 thread", exchanger_network_task_handle, (void*)&para_info[7]);

    while (sal_thread_is_running(tid[0]) || sal_thread_is_running(tid[1]) || sal_thread_is_running(tid[2])
           || sal_thread_is_running(tid[3]) || sal_thread_is_running(tid[4]) || sal_thread_is_running(tid[5])
           || sal_thread_is_running(tid[6]) || sal_thread_is_running(tid[7])) {
        SaluSleep(SAL_MILLISECOND_USEC * 10);
    }

    for (s32 j = 0; j < ndev; j++) {
        NetworkManager::GetInstance(dev_list[j]).Destroy();
        EXPECT_EQ(ret, HCCL_SUCCESS);
        EXPECT_EQ(*(para_info[j].ret_value), HCCL_SUCCESS);
        (void)sal_thread_destroy(tid[j]);
    }

    ret = send_str->compare(*recv_str);

    EXPECT_EQ(ret, 0);
}
#endif
TEST_F(ExchangerNetworkTest, 8_thread_4send_4receive)
{
    s32 ret = 0;
    HcclRootInfo commId;

    ret = hcclComm::GetUniqueId(&commId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    char collectiveId[SAL_UNIQUE_ID_BYTES];
    ret = SalGetUniqueId(collectiveId);
    std::string collective_id_tmp = collectiveId;

    const s32 ndev = 8;
    s32 dev_list[ndev] = {1, 0, 2, 3, 4, 5, 6, 7};
    std::vector<s32> device_ids(dev_list, dev_list + ndev);

    std::vector<u32> ip_list;
    for (int i = 0; i < ndev; i++) {
        u32 ipAddr = 0;
        (void)rt_get_dev_ip(0, i, &ipAddr);
        ip_list.push_back(ipAddr);
    }

    const u32 rank_list[ndev] = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<u32> user_ranks(rank_list, rank_list + ndev);

    sal_thread_t tid[ndev];
    ExchangerNetworkpara_t para_info[ndev];
    std::vector<std::shared_ptr<std::string>> send_str;
    send_str.resize(8);
    std::vector<std::shared_ptr<std::string>> recv_str;
    recv_str.resize(8);
    std::vector<std::string> send_words{"what", "how", "when", "why", "where", "other", "first", "second"};

    for (s32 i = 0; i < ndev; i++) {
        para_info[i].collectiveId = collective_id_tmp;
        para_info[i].userRank = i;
        para_info[i].user_rank_size = ndev;
        para_info[i].devicePhyId = dev_list[i];
        para_info[i].exchanger_network = nullptr;
        para_info[i].device_ids.assign(device_ids.begin(), device_ids.end());
        para_info[i].device_ips.assign(ip_list.begin(), ip_list.end());
        para_info[i].user_ranks.assign(user_ranks.begin(), user_ranks.end());
        para_info[i].tag = "default_tag";
        send_str[i].reset(new string(send_words[i]));
        recv_str[i].reset(new string(""));
        para_info[i].send_str = send_str[i];
        para_info[i].recv_str = recv_str[i];
        /* 前四个发送，后四个接收 */
        if (i < 4) {
            para_info[i].send_rank = i;
            para_info[i].send_to = i + 4;
            para_info[i].recv_rank = -1;
            para_info[i].recv_from = -1;
        } else {
            para_info[i].send_rank = -1;
            para_info[i].send_to = -1;
            para_info[i].recv_rank = i;
            para_info[i].recv_from = i - 4;
        }
        para_info[i].ret_value = &ret;
    }

    tid[0] = sal_thread_create("exchanger_network rank0 thread", exchanger_network_task_handle, (void*)&para_info[0]);
    tid[1] = sal_thread_create("exchanger_network rank1 thread", exchanger_network_task_handle, (void*)&para_info[1]);
    tid[2] = sal_thread_create("exchanger_network rank2 thread", exchanger_network_task_handle, (void*)&para_info[2]);
    tid[3] = sal_thread_create("exchanger_network rank3 thread", exchanger_network_task_handle, (void*)&para_info[3]);
    tid[4] = sal_thread_create("exchanger_network rank4 thread", exchanger_network_task_handle, (void*)&para_info[4]);
    tid[5] = sal_thread_create("exchanger_network rank5 thread", exchanger_network_task_handle, (void*)&para_info[5]);
    tid[6] = sal_thread_create("exchanger_network rank6 thread", exchanger_network_task_handle, (void*)&para_info[6]);
    tid[7] = sal_thread_create("exchanger_network rank7 thread", exchanger_network_task_handle, (void*)&para_info[7]);

    while (sal_thread_is_running(tid[0]) || sal_thread_is_running(tid[1]) || sal_thread_is_running(tid[2])
           || sal_thread_is_running(tid[3]) || sal_thread_is_running(tid[4]) || sal_thread_is_running(tid[5])
           || sal_thread_is_running(tid[6]) || sal_thread_is_running(tid[7])) {
        SaluSleep(SAL_MILLISECOND_USEC * 10);
    }

    for (s32 j = 0; j < ndev; j++) {
        ret = NetworkManager::GetInstance(dev_list[j]).Destroy();
        EXPECT_EQ(ret, HCCL_SUCCESS);
        EXPECT_EQ(*(para_info[j].ret_value), HCCL_SUCCESS);
        (void)sal_thread_destroy(tid[j]);
    }

    for (s32 k = 0; k < 4; k++) {
        ret = send_str[k]->compare(*(recv_str[k + 4]));
        EXPECT_EQ(ret, 0);
        HCCL_INFO("send[%s] vs recv[%s]", send_str[k]->c_str(), recv_str[k]->c_str());
    }
}

TEST_F(ExchangerNetworkTest, 8_thread_send_receive_by_ring)
{
    s32 ret = 0;
    HcclRootInfo commId;

    ret = hcclComm::GetUniqueId(&commId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    char collectiveId[SAL_UNIQUE_ID_BYTES];
    ret = SalGetUniqueId(collectiveId);
    std::string collective_id_tmp = collectiveId;

    const s32 ndev = 8;
    s32 dev_list[ndev] = {1, 0, 2, 3, 4, 5, 6, 7};
    std::vector<s32> device_ids(dev_list, dev_list + ndev);

    std::vector<u32> ip_list;
    for (int i = 0; i < ndev; i++) {
        u32 ipAddr = 0;
        (void)rt_get_dev_ip(0, i, &ipAddr);
        ip_list.push_back(ipAddr);
    }

    const u32 rank_list[ndev] = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<u32> user_ranks(rank_list, rank_list + ndev);

    sal_thread_t tid[ndev];
    ExchangerNetworkpara_t para_info[ndev];
    std::vector<std::shared_ptr<std::string>> send_str;
    send_str.resize(8);
    std::vector<std::shared_ptr<std::string>> recv_str;
    recv_str.resize(8);
    std::vector<std::string> send_words{"what", "how", "when", "why", "where", "other", "first", "second"};

    for (s32 i = 0; i < ndev; i++) {
        para_info[i].collectiveId = collective_id_tmp;
        para_info[i].userRank = i;
        para_info[i].user_rank_size = ndev;
        para_info[i].devicePhyId = dev_list[i];
        para_info[i].exchanger_network = nullptr;
        para_info[i].device_ids.assign(device_ids.begin(), device_ids.end());
        para_info[i].device_ips.assign(ip_list.begin(), ip_list.end());
        para_info[i].user_ranks.assign(user_ranks.begin(), user_ranks.end());
        para_info[i].tag = "default_tag";
        send_str[i].reset(new string(send_words[i]));
        recv_str[i].reset(new string(""));
        para_info[i].send_str = send_str[i];
        para_info[i].recv_str = recv_str[i];
        para_info[i].send_rank = i;
        para_info[i].send_to = (i == ndev - 1) ? 0 : (i + 1);
        para_info[i].recv_rank = i;
        para_info[i].recv_from = (i == 0) ? (ndev - 1) : (i - 1); // 从前一个节点接收
        para_info[i].ret_value = &ret;
    }

    tid[0] = sal_thread_create("exchanger_network rank0 thread", exchanger_network_task_handle, (void*)&para_info[0]);
    tid[1] = sal_thread_create("exchanger_network rank1 thread", exchanger_network_task_handle, (void*)&para_info[1]);
    tid[2] = sal_thread_create("exchanger_network rank2 thread", exchanger_network_task_handle, (void*)&para_info[2]);
    tid[3] = sal_thread_create("exchanger_network rank3 thread", exchanger_network_task_handle, (void*)&para_info[3]);
    tid[4] = sal_thread_create("exchanger_network rank4 thread", exchanger_network_task_handle, (void*)&para_info[4]);
    tid[5] = sal_thread_create("exchanger_network rank5 thread", exchanger_network_task_handle, (void*)&para_info[5]);
    tid[6] = sal_thread_create("exchanger_network rank6 thread", exchanger_network_task_handle, (void*)&para_info[6]);
    tid[7] = sal_thread_create("exchanger_network rank7 thread", exchanger_network_task_handle, (void*)&para_info[7]);

    while (sal_thread_is_running(tid[0]) || sal_thread_is_running(tid[1]) || sal_thread_is_running(tid[2])
           || sal_thread_is_running(tid[3]) || sal_thread_is_running(tid[4]) || sal_thread_is_running(tid[5])
           || sal_thread_is_running(tid[6]) || sal_thread_is_running(tid[7])) {
        SaluSleep(SAL_MILLISECOND_USEC * 10);
    }

    for (s32 j = 0; j < ndev; j++) {
        ret = NetworkManager::GetInstance(dev_list[j]).Destroy();
        EXPECT_EQ(ret, HCCL_SUCCESS);
        EXPECT_EQ(*(para_info[j].ret_value), HCCL_SUCCESS);
        (void)sal_thread_destroy(tid[j]);
    }

    for (s32 k = 0; k < ndev; k++) {
        send_str[k]->compare(*(send_str[(k == ndev - 1) ? 0 : (k + 1)]));
    }
}

TEST_F(ExchangerNetworkTest, ut_init_input_invalid)
{
    s32 ret = HCCL_SUCCESS;

    u32 userRank = 0;
    const u32 user_rank_size = 2;
    const s32 dev_num = 2;

    std::string collectiveId("default");
    s32 dev_list[dev_num] = {0, 1};
    std::vector<s32> device_ids(dev_list, dev_list + 2);
    const s32 rank_list[dev_num] = {0, 1};
    std::vector<u32> user_ranks(rank_list, rank_list + 2);
    const std::string tag("default_tag");
    const std::string illegal_tag("123456789012345678901234567890123456789012345678901234567890123456789012345678901234"
                                  "5678901234567890123456789012345678901234567890_tag");
    ExchangerNetwork exchangernetwork(illegal_tag, userRank, user_rank_size);

    ret = exchangernetwork.AppendSockets(device_ids, user_ranks);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    std::vector<s32> illegal_device_ids(dev_list, dev_list + 1);

    ExchangerNetwork exchangernetwork1(collectiveId, userRank, user_rank_size);

    ret = exchangernetwork1.AppendSockets(illegal_device_ids, user_ranks);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(ExchangerNetworkTest, ut_thread_invalid_send_recv_para)
{
    s32 ret;
    HcclRootInfo commId;

    ret = hcclComm::GetUniqueId(&commId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    char collectiveId[SAL_UNIQUE_ID_BYTES];
    ret = SalGetUniqueId(collectiveId);
    std::string collective_id_tmp = collectiveId;

    const s32 ndev = 2;
    s32 dev_list[ndev] = {0, 1};
    std::vector<s32> device_ids(dev_list, dev_list + ndev);
    const u32 rank_list[ndev] = {0, 1};
    std::vector<u32> user_ranks(rank_list, rank_list + ndev);
    std::vector<u32> ip_list;
    for (int i = 0; i < ndev; i++) {
        u32 ipAddr = 0;
        (void)rt_get_dev_ip(0, i, &ipAddr);
        ip_list.push_back(ipAddr);
    }

    sal_thread_t tid[ndev];
    ExchangerNetworkpara_t para_info[ndev];
    const s32 send_rank = 0;
    const s32 recv_rank = 1;
    const s32 buff_size = 2049;
    char sendbuff[buff_size];
    sal_memset(sendbuff, buff_size, 1, buff_size);
    sendbuff[buff_size - 1] = '\0';
    std::shared_ptr<std::string> send_str;
    std::shared_ptr<std::string> recv_str;
    send_str.reset(new string(sendbuff));
    recv_str.reset(new string(""));

    for (s32 i = 0; i < ndev; i++) {
        para_info[i].collectiveId = collective_id_tmp;
        para_info[i].userRank = i;
        para_info[i].user_rank_size = ndev;
        para_info[i].devicePhyId = dev_list[i];
        para_info[i].exchanger_network = nullptr;
        para_info[i].device_ids.assign(device_ids.begin(), device_ids.end());
        para_info[i].user_ranks.assign(user_ranks.begin(), user_ranks.end());
        para_info[i].device_ips.assign(ip_list.begin(), ip_list.end());
        para_info[i].tag = "default_tag";
        para_info[i].send_str = send_str;
        para_info[i].recv_str = recv_str;
        para_info[i].send_rank = send_rank;
        para_info[i].send_to = recv_rank;
        para_info[i].recv_rank = -1;
        para_info[i].recv_from = -1;
        para_info[i].ret_value = &ret;
    }

    tid[0] = sal_thread_create("exchanger_network rank0 thread", exchanger_network_task_handle, (void*)&para_info[0]);
    tid[1] = sal_thread_create("exchanger_network rank1 thread", exchanger_network_task_handle, (void*)&para_info[1]);

    while (sal_thread_is_running(tid[0]) || sal_thread_is_running(tid[1])) {
        SaluSleep(SAL_MILLISECOND_USEC * 10);
    }

    EXPECT_EQ(*(para_info[0].ret_value), HCCL_E_INTERNAL);
    EXPECT_EQ(*(para_info[1].ret_value), HCCL_E_INTERNAL);
    for (s32 j = 0; j < ndev; j++) {
        ret = NetworkManager::GetInstance(dev_list[j]).Destroy();
        EXPECT_EQ(ret, HCCL_SUCCESS);
        (void)sal_thread_destroy(tid[j]);
    }
}

TEST_F(ExchangerNetworkTest, ut_deinit_failed)
{
    s32 ret = HCCL_SUCCESS;

    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    u32 userRank = 0;
    const u32 user_rank_size = 1;
    const s32 dev_num = 1;

    std::string collectiveId("default");
    s32 dev_list[dev_num] = {0};
    std::vector<s32> device_ids(dev_list, dev_list + 1);

    u32 device_id = 0, localIp = 0, idx = 0;
    hrtSetDevice(device_id);
    ret = NetworkManager::GetInstance(device_id).Init(NICDeployment::NIC_DEPLOYMENT_HOST);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    const u32 rank_list[dev_num] = {0};
    std::vector<u32> user_ranks(rank_list, rank_list + 1);
    const std::string tag("default_tag");
    ExchangerNetwork exchangernetwork(collectiveId, userRank, user_rank_size);

    ret = exchangernetwork.AppendSockets(device_ids, user_ranks);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER(HrtRaDeInit).stubs().will(returnValue(HCCL_E_NETWORK));
    ret = NetworkManager::GetInstance(device_id).Destroy();

    u32 port = 16666;

    hrtSetDevice(device_id);
    ret = NetworkManager::GetInstance(device_id).Init(NICDeployment::NIC_DEPLOYMENT_DEVICE);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = NetworkManager::GetInstance(device_id).StartVnic(HcclIpAddress(device_id), port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = exchangernetwork.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER(HrtRaDeInit).stubs().will(returnValue(HCCL_E_NETWORK));
    ret = NetworkManager::GetInstance(device_id).Destroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(ExchangerNetworkTest, ut_NetworkManager_GetInstance)
{
    s32 device_id = 64;
    NetworkManager::GetInstance(device_id);
}

TEST_F(ExchangerNetworkTest, ut_MemNameRepository_GetInstance)
{
    s32 device_id = 64;
    MemNameRepository::GetInstance(device_id);
}

TEST_F(ExchangerNetworkTest, ut_init_2_dev_nic_fail)
{
    s32 ret = HCCL_SUCCESS;

    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    u32 userRank = 0;
    const u32 user_rank_size = 2;
    const s32 dev_num = 2;
    u32 port = 16666;

    std::string collectiveId("default");
    std::vector<HcclIpAddress> device_ips;
    device_ips.push_back(HcclIpAddress("10.21.78.208"));
    device_ips.push_back(HcclIpAddress("10.21.78.209"));

    std::vector<u32> user_ranks{0, 1};

    u32 device_id = 0, localIp = 0, idx = 0;
    hrtSetDevice(device_id);
    ret = NetworkManager::GetInstance(device_id).Init(NICDeployment::NIC_DEPLOYMENT_DEVICE);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = NetworkManager::GetInstance(device_id).StartVnic(HcclIpAddress(device_id), port);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    for (HcclIpAddress ip : device_ips) {
        ret = NetworkManager::GetInstance(device_id).StartNic(ip, 22, true);
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }

    const std::string tag("default_tag");
    ExchangerNetwork exchangernetwork(collectiveId, userRank, user_rank_size);

    MOCKER(hrtRaSocketBatchConnect).stubs().will(returnValue(1));

    MOCKER(GetExternalInputHcclLinkTimeOut).stubs().will(returnValue(1));

    ret = exchangernetwork.AppendSockets(device_ips, user_ranks);
    EXPECT_NE(ret, HCCL_SUCCESS);
    ret = NetworkManager::GetInstance(device_id).Destroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

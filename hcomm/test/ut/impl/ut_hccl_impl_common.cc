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
#include <stdlib.h>
#include <pthread.h>
#include <assert.h>
#include <semaphore.h>
#include <signal.h>
#include <syscall.h>
#include <sys/prctl.h>
#include <syslog.h>
#include <unistd.h>
#include <errno.h>

#include <securec.h>

#include <sys/types.h>
#include <stddef.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <driver/ascend_hal.h>

#define private public
#define protected public
#include "hccl_impl.h"
#include "hccl_communicator.h"
#undef protected
#undef private

#include "hccl/base.h"
#include <hccl/hccl_types.h>
#include "llt_hccl_stub_pub.h"
#include <sys/mman.h>
#include <fcntl.h>

#include "sal.h"

#include "config.h"
#include "dlra_function.h"
#include "rank_consistentcy_checker.h"
#include "ut_rank_table.h"
#include "tsd/tsd_client.h"
#include "dltdt_function.h"
#include <iostream>
#include <fstream>
#include "network_manager_pub.h"
#include "opexecounter_pub.h"
#include "param_check_pub.h"
#include "hcom_private.h"
using namespace std;
using namespace hccl;

typedef struct para_struct {
    HcclRootInfo rootInfo;
    std::string identify;
    s32 comm_num;
    s32 device_id;
    s32 ranks_local; // 本服务器内的rank数

    char* file_name;
    void* sendbuff;
    void* recvbuff;
    s32 count;
    HcclDataType datatype;
    HcclReduceOp op;
    s32 root;
    rtStream_t stream;
    int id;
    volatile s32* sync_addr;
} para_t;

void* impl_common_broadcast_task(void* parg)
{
    HcclResult ret = HCCL_SUCCESS;
    para_t* para_info = (para_t*)parg;
    s32 rank_num_tmp;

    HcomInfo hcom_info;
    std::string ranktable_file = para_info->file_name;
    std::string rankTableM;
    std::string realFilePath;

    hrtSetDevice(para_info->device_id);
    RankConsistentcyChecker::GetInstance().ClearCheckInfo();
    u32 devicePhyId;
    ret = hrtGetDevicePhyIdByIndex((u32)para_info->device_id, devicePhyId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // ret = NetworkManager::GetInstance(devicePhyId).Init(NICDeployment::NIC_DEPLOYMENT_DEVICE);
    // EXPECT_EQ(ret, HCCL_SUCCESS);
    // ret = NetworkManager::GetInstance(devicePhyId).StartVnic();
    // EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, para_info->identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    sal_memcpy(hcom_info.params.id.internal, sizeof(HcclRootInfo), &para_info->rootInfo, sizeof(HcclRootInfo));

    hcom_info.pComm.reset(new (std::nothrow) hccl::hcclComm());
    rtModel_t model = (void*)1;

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] task rt_set_device fails", hcom_info.params.rank);
    }

    CommConfig commConfig("hccl_world_group");
    ret = hcom_info.pComm->init(hcom_info.params, commConfig, hcom_info.rankTable);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] task broadcast fails", para_info->device_id);
    }

    bool swapped;

    rank_num_tmp = *(para_info->sync_addr) - 1;

    do {
        rank_num_tmp += 1;

        swapped = __sync_bool_compare_and_swap(para_info->sync_addr, rank_num_tmp, rank_num_tmp + 1);
    } while (!swapped);

    while (*(para_info->sync_addr) < para_info->ranks_local) {
        sched_yield();
    } // linux提供一个系统调用运行进程主动让出执行权

    __sync_synchronize(); // 插入内存屏障，对顺序性有要求，但是有没有使用lock指令的时候
    HCCL_DEBUG("all %d  ranks init ok ,then broadcast", hcom_info.params.totalRanks);
    ret = hcom_info.pComm->Broadcast(
        "tag_impl_common_broadcast_task", para_info->sendbuff, para_info->count, para_info->datatype, para_info->root,
        para_info->stream);

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("rank[%d] task broadcast fails", hcom_info.params.rank);
    }

    rtError_t rt_ret = RT_ERROR_NONE;
    rt_ret = aclrtSynchronizeStream(para_info->stream);

    if (rt_ret != RT_ERROR_NONE) {
        HCCL_ERROR("rank[%d] task allgather fails", hcom_info.params.rank);
    }
    RankConsistentcyChecker::GetInstance().ClearCheckInfo();
    return (NULL);
}

void* impl_common_all_reduce_task(void* parg)
{
    HcclResult ret = HCCL_SUCCESS;
    para_t* para_info = (para_t*)parg;
    s32 rank_num_tmp;

    HcomInfo hcom_info;
    std::string ranktable_file = para_info->file_name;
    std::string rankTableM;
    std::string realFilePath;

    hrtSetDevice(para_info->device_id);
    RankConsistentcyChecker::GetInstance().ClearCheckInfo();

    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    u32 devicePhyId;
    ret = hrtGetDevicePhyIdByIndex((u32)para_info->device_id, devicePhyId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // ret = NetworkManager::GetInstance(devicePhyId).Init(NICDeployment::NIC_DEPLOYMENT_DEVICE);
    // EXPECT_EQ(ret, HCCL_SUCCESS);
    // ret = NetworkManager::GetInstance(devicePhyId).StartVnic();
    // EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, para_info->identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    sal_memcpy(hcom_info.params.id.internal, sizeof(HcclRootInfo), &para_info->rootInfo, sizeof(HcclRootInfo));

    hcom_info.pComm.reset(new (std::nothrow) hccl::hcclComm());
    rtModel_t model = (void*)1;

    CommConfig commConfig("hccl_world_group");
    ret = hcom_info.pComm->init(hcom_info.params, commConfig, hcom_info.rankTable);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] task all_reduce fails", para_info->device_id);
    }
    bool swapped;

    rank_num_tmp = *(para_info->sync_addr) - 1;

    do {
        rank_num_tmp += 1;

        swapped = __sync_bool_compare_and_swap(para_info->sync_addr, rank_num_tmp, rank_num_tmp + 1);
    } while (!swapped);

    while (*(para_info->sync_addr) < para_info->ranks_local) {
        sched_yield();
    } // linux提供一个系统调用运行进程主动让出执行权

    __sync_synchronize(); // 插入内存屏障，对顺序性有要求，但是有没有使用lock指令的时候
    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
    ret = hcom_info.pComm->AllReduce(
        "tag_impl_common_all_reduce_task", para_info->sendbuff, para_info->recvbuff, para_info->count,
        para_info->datatype, para_info->op, para_info->stream);

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] task HcclAllReduce fails", para_info->device_id);
    }

    rtError_t rt_ret = RT_ERROR_NONE;
    rt_ret = aclrtSynchronizeStream(para_info->stream);

    if (rt_ret != RT_ERROR_NONE) {
        HCCL_ERROR("rank[%d] task allgather fails", hcom_info.params.rank);
    }
    RankConsistentcyChecker::GetInstance().ClearCheckInfo();
    return (NULL);
}

class HcclImplCommonTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HcclImplCommonTest SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "HcclImplCommonTest TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        GlobalMockObject::verify();
        static s32 call_cnt = 0;
        DlTdtFunction::GetInstance().DlTdtFunctionInit();
        TsdOpen(1, 2);
        string name = std::to_string(call_cnt++) + "_" + __PRETTY_FUNCTION__;
        ra_set_shm_name(name.c_str());
        ra_set_test_type(0, "UT_TEST");
        set_board_id(0x2000);
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        TsdClose(1);
        set_board_id(0x0000);
        std::cout << "A Test TearDown" << std::endl;
    }
};

#define DEV_NUM_8 8
#define DEV_NUM_4 4
#define DEV_NUM_5 5
#define DEV_NUM_2 2
#define DEV_NUM_3 3

#if 1
TEST_F(HcclImplCommonTest, ut_hccl_impl_610_8rank_1server_allreduce_char)
{
    setenv("PROFILING_MODE", "true", 1);
    char file_name_t[] = "./st_hccl_impl_610_8rank_1server_allreduce_char.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << g_rank_table_610_8rank_1server << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }
    outfile.close();
    RankConsistentcyChecker::GetInstance().ClearCheckInfo();
    set_board_id(0x2000);

    s32 rank, errors = 0;

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;

    s8* result_buff[DEV_NUM_8];
    s8* sendbuf[DEV_NUM_8];
    s8* recvbuf[DEV_NUM_8];
    s8* inputbuf[DEV_NUM_8];
    s8* outputbuf[DEV_NUM_8];

    s32 sync_value = 0;

    rtStream_t stream[DEV_NUM_8];
    sal_thread_t tid[DEV_NUM_8];
    para_t para_info[DEV_NUM_8];

    HcclDataType datatype = HCCL_DATA_TYPE_INT8;

    HcclReduceOp op = HCCL_REDUCE_SUM;
    s32 count = 1024;
    s32 ndev = DEV_NUM_8;
    HcclRootInfo rootInfo;
    ret = hccl::hcclComm::GetUniqueId(&rootInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    /** 初始化输入输出缓存 */
    for (s32 i = 0; i < ndev; i++) {
        ret = hrtMalloc((void**)&sendbuf[i], count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(sendbuf[i], count * sizeof(s8), 0, count * sizeof(s8));
        ret = hrtMalloc((void**)&recvbuf[i], count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(recvbuf[i], count * sizeof(s8), 0, count * sizeof(s8));
        ret = hrtMalloc((void**)&result_buff[i], count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(result_buff[i], count * sizeof(s8), 0, count * sizeof(s8));
        inputbuf[i] = sendbuf[i];
        outputbuf[i] = recvbuf[i];
    }

    // sendbuf 赋值
    for (u32 j = 0; j < ndev; j++) {
        for (u32 i = 0; i < count; i++) {
            inputbuf[j][i] = 1;
        }
    }

    // resultbuf 赋值
    for (s32 i = 0; i < ndev; ++i) {
        for (u32 j = 0; j < count; j++) {
            result_buff[i][j] = ndev;
        }
    }
    for (s32 i = 0; i < ndev; ++i) {
        rt_ret = aclrtCreateStream(&stream[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    for (s32 i = 0; i < ndev; i++) {
        sal_memcpy(&para_info[i].rootInfo, sizeof(HcclRootInfo), &rootInfo, sizeof(HcclRootInfo));
        std::ostringstream identify("");
        identify << i;
        para_info[i].identify = identify.str();
        para_info[i].comm_num = ndev;
        para_info[i].device_id = i;
        para_info[i].ranks_local = ndev;

        para_info[i].count = count;
        para_info[i].datatype = datatype;
        para_info[i].sendbuff = inputbuf[i];
        para_info[i].stream = stream[i];
        para_info[i].recvbuff = outputbuf[i];
        para_info[i].op = op;

        para_info[i].sync_addr = &sync_value;
        para_info[i].file_name = file_name_t;
    }

    // 创建每个Dev的allreduce任务线程
    for (s32 i = 0; i < ndev; i++) {
        tid[i] = sal_thread_create("thread", impl_common_all_reduce_task, (void*)&para_info[i]);
        EXPECT_NE(tid[i], (sal_thread_t)NULL);
    }

    for (s32 i = 0; i < ndev; i++) {
        while (sal_thread_is_running(tid[i])) {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
    }

    // 获取stream的操作的同步信号量
    for (s32 i = 0; i < ndev; i++) {
        for (s32 j = 0; j < count; j++) {
            s8 res = result_buff[i][j];
            s8 recv = outputbuf[i][j];

            if (res != recv) {
                HCCL_ERROR(" rank :%d recvbuf[%d] :%d result_buff[%d]:%d \n", i, j, recv, j, res);
            }
        }
    }
    if (errors) {
        HCCL_ERROR("%d errors. Test FAILED.\n", errors);
    } else {
        HCCL_INFO("Test PASSED.\n");
    }
    for (s32 i = 0; i < ndev; i++) {
        hrtFree(sendbuf[i]);
        hrtFree(recvbuf[i]);
        hrtFree(result_buff[i]);
        rt_ret = aclrtDestroyStream(stream[i]);

        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }
    set_board_id(0);
    remove(file_name_t);
    EXPECT_EQ(errors, 0);
    setenv("PROFILING_MODE", "false", 1);
}
#endif

#if 0
TEST_F(HcclImplCommonTest, ut_hccl_impl_610_8rank_1server_allreduce_float)
{
    char file_name_t[] = "./st_hccl_impl_610_8rank_1server_allreduce_float.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(4) << g_rank_table_610_8rank_1server << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }
    else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }
    outfile.close();

    set_board_id(0x2000);

    s32 rank, errors = 0;

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;

    float* result_buff[DEV_NUM_8];
    float* sendbuf[DEV_NUM_8];
    float* recvbuf[DEV_NUM_8];
    float* inputbuf[DEV_NUM_8];
    float* outputbuf[DEV_NUM_8];

    s32 sync_value = 0;

    rtStream_t stream[DEV_NUM_8];
    sal_thread_t tid[DEV_NUM_8];
    para_t para_info[DEV_NUM_8];

    HcclDataType datatype = HCCL_DATA_TYPE_FP32;

    HcclReduceOp op = HCCL_REDUCE_SUM;
    s32 count = 1024;
    s32 ndev = DEV_NUM_8;
    HcclRootInfo rootInfo;
    ret = hccl::hcclComm::GetUniqueId(&rootInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    /** 初始化输入输出缓存 */
    for (s32 i = 0; i < ndev; i++ )
    {
        ret = hrtMalloc((void**)&sendbuf[i], count * sizeof(float));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(sendbuf[i], count * sizeof(float), 0, count * sizeof(float));
        ret = hrtMalloc((void**)&recvbuf[i], count * sizeof(float));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(recvbuf[i], count * sizeof(float), 0, count * sizeof(float));

        result_buff[i] = (float*)sal_malloc(count * sizeof(float));
        sal_memset(result_buff[i], count * sizeof(float), 0, count * sizeof(float));
        inputbuf[i] = sendbuf[i];
        outputbuf[i] = recvbuf[i];
    }

    //sendbuf 赋值
    for (u32 j = 0; j < ndev; j++)
    {
        for (u32 i = 0; i < count; i++)
        {
            inputbuf[j][i] = 1.0f;
        }
    }

    //resultbuf 赋值
   for (s32 i = 0; i < ndev; ++i)
 {
    for (u32 j = 0; j < count; j++)
     {
            result_buff[i][j] = 1.0f * ndev;
     }
    }
    for (s32 i = 0; i < ndev; ++i)
    {
        rt_ret = aclrtCreateStream(&stream[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    for (s32 i = 0; i < ndev; i++)
    {
        sal_memcpy(&para_info[i].rootInfo, sizeof(HcclRootInfo), &rootInfo, sizeof(HcclRootInfo));
        std::ostringstream identify("");
        identify << i;
        para_info[i].identify = identify.str();
        para_info[i].comm_num = ndev;
        para_info[i].device_id = i ;
        para_info[i].ranks_local = ndev;

        para_info[i].count = count;
        para_info[i].datatype = datatype;
        para_info[i].sendbuff = inputbuf[i];
        para_info[i].stream = stream[i];
        para_info[i].recvbuff = outputbuf[i];
        para_info[i].op = op;

        para_info[i].sync_addr = &sync_value;
        para_info[i].file_name = file_name_t;
    }

    // 创建每个Dev的allreduce任务线程
    for (s32 i = 0; i < ndev; i++)
    {
        tid[i] = sal_thread_create("thread", impl_common_all_reduce_task, (void*)&para_info[i]);
        EXPECT_NE(tid[i], (sal_thread_t )NULL);
    }

    for (s32 i = 0; i < ndev; i++)
    {
        while ( sal_thread_is_running(tid[i]))
        {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
        // u32 devicePhyId;
        // ret = hrtGetDevicePhyIdByIndex((u32)i, devicePhyId);
        // EXPECT_EQ(ret, HCCL_SUCCESS);
        // ret = NetworkManager::GetInstance(devicePhyId).Destroy();
    }

    //获取stream的操作的同步信号量
    for (s32 i = 0; i < ndev; i++)
  {
     for (s32 j = 0; j < count; j++)
    {
            float res = result_buff[i][j];
            float recv = outputbuf[i][j];

            if (abs(res - recv)>1e-6)
            {
                HCCL_ERROR(" rank :%d recvbuf[%d] :%d result_buff[%d]:%d \n", i, j, recv, j, res);
                errors ++;
                break;
            }
    }
        }
      if (errors)
        {
            HCCL_ERROR("%d errors. Test FAILED.\n", errors);
        }
        else
        {
            HCCL_INFO("Test PASSED.\n");
        }
    for (s32 i = 0; i < ndev; i++)
   {
        hrtFree(sendbuf[i]);
        hrtFree(recvbuf[i]);
        sal_free(result_buff[i]);
    rt_ret = aclrtDestroyStream(stream[i]);

    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
   }
    set_board_id(0);
    remove(file_name_t);
    EXPECT_EQ(errors, 0);
}
#endif

#if 0
TEST_F(HcclImplCommonTest, ut_hccl_impl_610_5rank_1server_allreduce_char)
{
    char file_name_t[] = "./st_hccl_impl_610_5rank_1server_allreduce_char.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(4) << g_rank_table_610_5rank_1server << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }
    else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }
    outfile.close();

    set_board_id(0x2000);

    s32 rank, errors = 0;

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;

    s8* result_buff[DEV_NUM_5];
    s8* sendbuf[DEV_NUM_5];
    s8* recvbuf[DEV_NUM_5];
    s8* inputbuf[DEV_NUM_5];
    s8* outputbuf[DEV_NUM_5];

    s32 sync_value = 0;

    rtStream_t stream[DEV_NUM_5];
    sal_thread_t tid[DEV_NUM_5];
    para_t para_info[DEV_NUM_5];

    HcclDataType datatype = HCCL_DATA_TYPE_INT8;

    HcclReduceOp op = HCCL_REDUCE_SUM;
    s32 count = 1024;
    s32 ndev = DEV_NUM_5;
    HcclRootInfo rootInfo;
    ret = hccl::hcclComm::GetUniqueId(&rootInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    /** 初始化输入输出缓存 */
    for (s32 i = 0; i < ndev; i++ )
    {
         ret = hrtMalloc((void**)&sendbuf[i], count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(sendbuf[i], count * sizeof(s8), 0, count * sizeof(s8));
        ret = hrtMalloc((void**)&recvbuf[i], count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(recvbuf[i], count * sizeof(s8), 0, count * sizeof(s8));
        ret = hrtMalloc((void**)&result_buff[i], count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(result_buff[i], count * sizeof(s8), 0, count * sizeof(s8));
        inputbuf[i] = sendbuf[i];
        outputbuf[i] = recvbuf[i];
    }

    //sendbuf 赋值
    for (u32 j = 0; j < ndev; j++)
    {
        for (u32 i = 0; i < count; i++)
        {
            inputbuf[j][i] = 1;
        }
    }

    //resultbuf 赋值
   for (s32 i = 0; i < ndev; ++i)
 {
    for (u32 j = 0; j < count; j++)
     {
            result_buff[i][j] = ndev;
     }
    }
    for (s32 i = 0; i < ndev; ++i)
    {
        rt_ret = aclrtCreateStream(&stream[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    for (s32 i = 0; i < ndev; i++)
    {
        sal_memcpy(&para_info[i].rootInfo, sizeof(HcclRootInfo), &rootInfo, sizeof(HcclRootInfo));
        std::ostringstream identify("");
        identify << i;
        para_info[i].identify = identify.str();
        para_info[i].comm_num = ndev;
        para_info[i].device_id = i ;
        para_info[i].ranks_local = ndev;

        para_info[i].count = count;
        para_info[i].datatype = datatype;
        para_info[i].sendbuff = inputbuf[i];
        para_info[i].stream = stream[i];
        para_info[i].recvbuff = outputbuf[i];
        para_info[i].op = op;

        para_info[i].sync_addr = &sync_value;
        para_info[i].file_name = file_name_t;
    }

    // 创建每个Dev的allreduce任务线程
    for (s32 i = 0; i < ndev; i++)
    {
        tid[i] = sal_thread_create("thread", impl_common_all_reduce_task, (void*)&para_info[i]);
        EXPECT_NE(tid[i], (sal_thread_t )NULL);
    }

    for (s32 i = 0; i < ndev; i++)
    {
        while ( sal_thread_is_running(tid[i]))
        {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
    }

    //获取stream的操作的同步信号量
    for (s32 i = 0; i < ndev; i++)
  {
     for (s32 j = 0; j < count; j++)
    {
            s8 res = result_buff[i][j];
            s8 recv = outputbuf[i][j];

            if (res != recv)
            {
                HCCL_ERROR(" rank :%d recvbuf[%d] :%d result_buff[%d]:%d \n", i, j, recv, j, res);
            }
    }
        }
      if (errors)
        {
            HCCL_ERROR("%d errors. Test FAILED.\n", errors);
        }
        else
        {
            HCCL_INFO("Test PASSED.\n");
        }
    for (s32 i = 0; i < ndev; i++)
   {
        hrtFree(sendbuf[i]);
        hrtFree(recvbuf[i]);
        hrtFree(result_buff[i]);
    rt_ret = aclrtDestroyStream(stream[i]);

    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
   }
    set_board_id(0);
    remove(file_name_t);
    EXPECT_EQ(errors, 0);
}
#endif

#if 1
TEST_F(HcclImplCommonTest, ut_hccl_impl_610_4rank_1server_allreduce_char)
{
    char file_name_t[] = "./st_hccl_impl_610_4rank_1server_allreduce_char.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << g_rank_table_610_4rank_1server << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }
    outfile.close();

    set_board_id(0x2000);

    s32 rank, errors = 0;

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;

    s8* result_buff[DEV_NUM_4];
    s8* sendbuf[DEV_NUM_4];
    s8* recvbuf[DEV_NUM_4];
    s8* inputbuf[DEV_NUM_4];
    s8* outputbuf[DEV_NUM_4];

    s32 sync_value = 0;

    rtStream_t stream[DEV_NUM_4];
    sal_thread_t tid[DEV_NUM_4];
    para_t para_info[DEV_NUM_4];

    HcclDataType datatype = HCCL_DATA_TYPE_INT8;

    HcclReduceOp op = HCCL_REDUCE_SUM;
    s32 count = 1024;
    s32 ndev = DEV_NUM_4;
    HcclRootInfo rootInfo;
    ret = hccl::hcclComm::GetUniqueId(&rootInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    /** 初始化输入输出缓存 */
    for (s32 i = 0; i < ndev; i++) {
        ret = hrtMalloc((void**)&sendbuf[i], count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(sendbuf[i], count * sizeof(s8), 0, count * sizeof(s8));
        ret = hrtMalloc((void**)&recvbuf[i], count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(recvbuf[i], count * sizeof(s8), 0, count * sizeof(s8));
        ret = hrtMalloc((void**)&result_buff[i], count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(result_buff[i], count * sizeof(s8), 0, count * sizeof(s8));
        inputbuf[i] = sendbuf[i];
        outputbuf[i] = recvbuf[i];
    }

    // sendbuf 赋值
    for (u32 j = 0; j < ndev; j++) {
        for (u32 i = 0; i < count; i++) {
            inputbuf[j][i] = 1;
        }
    }

    // resultbuf 赋值
    for (s32 i = 0; i < ndev; ++i) {
        for (u32 j = 0; j < count; j++) {
            result_buff[i][j] = ndev;
        }
    }
    for (s32 i = 0; i < ndev; ++i) {
        rt_ret = aclrtCreateStream(&stream[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    for (s32 i = 0; i < ndev; i++) {
        sal_memcpy(&para_info[i].rootInfo, sizeof(HcclRootInfo), &rootInfo, sizeof(HcclRootInfo));
        std::ostringstream identify("");
        identify << i;
        para_info[i].identify = identify.str();
        para_info[i].comm_num = ndev;
        para_info[i].device_id = i;
        para_info[i].ranks_local = ndev;

        para_info[i].count = count;
        para_info[i].datatype = datatype;
        para_info[i].sendbuff = inputbuf[i];
        para_info[i].stream = stream[i];
        para_info[i].recvbuff = outputbuf[i];
        para_info[i].op = op;

        para_info[i].sync_addr = &sync_value;
        para_info[i].file_name = file_name_t;
    }

    // 创建每个Dev的allreduce任务线程
    for (s32 i = 0; i < ndev; i++) {
        tid[i] = sal_thread_create("thread", impl_common_all_reduce_task, (void*)&para_info[i]);
        EXPECT_NE(tid[i], (sal_thread_t)NULL);
    }

    for (s32 i = 0; i < ndev; i++) {
        while (sal_thread_is_running(tid[i])) {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
    }

    // 获取stream的操作的同步信号量
    for (s32 i = 0; i < ndev; i++) {
        for (s32 j = 0; j < count; j++) {
            s8 res = result_buff[i][j];
            s8 recv = outputbuf[i][j];

            if (res != recv) {
                HCCL_ERROR(" rank :%d recvbuf[%d] :%d result_buff[%d]:%d \n", i, j, recv, j, res);
            }
        }
    }
    if (errors) {
        HCCL_ERROR("%d errors. Test FAILED.\n", errors);
    } else {
        HCCL_INFO("Test PASSED.\n");
    }
    for (s32 i = 0; i < ndev; i++) {
        hrtFree(sendbuf[i]);
        hrtFree(recvbuf[i]);
        hrtFree(result_buff[i]);
        rt_ret = aclrtDestroyStream(stream[i]);

        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }
    set_board_id(0);
    remove(file_name_t);
    EXPECT_EQ(errors, 0);
}
#endif

#if 1
TEST_F(HcclImplCommonTest, ut_hccl_impl_610_2rank_1server_allreduce_char)
{
    char file_name_t[] = "./st_hccl_impl_610_2rank_1server_allreduce_char.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << g_rank_table_610_2rank_1server << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }
    outfile.close();

    set_board_id(0x2000);

    s32 rank, errors = 0;

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;

    s8* result_buff[DEV_NUM_2];
    s8* sendbuf[DEV_NUM_2];
    s8* recvbuf[DEV_NUM_2];
    s8* inputbuf[DEV_NUM_2];
    s8* outputbuf[DEV_NUM_2];

    s32 sync_value = 0;

    rtStream_t stream[DEV_NUM_2];
    sal_thread_t tid[DEV_NUM_2];
    para_t para_info[DEV_NUM_2];

    HcclDataType datatype = HCCL_DATA_TYPE_INT8;

    HcclReduceOp op = HCCL_REDUCE_SUM;
    s32 count = 1024;
    s32 ndev = DEV_NUM_2;
    HcclRootInfo rootInfo;
    ret = hccl::hcclComm::GetUniqueId(&rootInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    /** 初始化输入输出缓存 */
    for (s32 i = 0; i < ndev; i++) {
        ret = hrtMalloc((void**)&sendbuf[i], count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(sendbuf[i], count * sizeof(s8), 0, count * sizeof(s8));
        ret = hrtMalloc((void**)&recvbuf[i], count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(recvbuf[i], count * sizeof(s8), 0, count * sizeof(s8));
        ret = hrtMalloc((void**)&result_buff[i], count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(result_buff[i], count * sizeof(s8), 0, count * sizeof(s8));
        inputbuf[i] = sendbuf[i];
        outputbuf[i] = recvbuf[i];
    }

    // sendbuf 赋值
    for (u32 j = 0; j < ndev; j++) {
        for (u32 i = 0; i < count; i++) {
            inputbuf[j][i] = 1;
        }
    }

    // resultbuf 赋值
    for (s32 i = 0; i < ndev; ++i) {
        for (u32 j = 0; j < count; j++) {
            result_buff[i][j] = ndev;
        }
    }
    for (s32 i = 0; i < ndev; ++i) {
        rt_ret = aclrtCreateStream(&stream[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    for (s32 i = 0; i < ndev; i++) {
        sal_memcpy(&para_info[i].rootInfo, sizeof(HcclRootInfo), &rootInfo, sizeof(HcclRootInfo));
        std::ostringstream identify("");
        identify << i;
        para_info[i].identify = identify.str();
        para_info[i].comm_num = ndev;
        para_info[i].device_id = i;
        para_info[i].ranks_local = ndev;

        para_info[i].count = count;
        para_info[i].datatype = datatype;
        para_info[i].sendbuff = inputbuf[i];
        para_info[i].stream = stream[i];
        para_info[i].recvbuff = outputbuf[i];
        para_info[i].op = op;

        para_info[i].sync_addr = &sync_value;
        para_info[i].file_name = file_name_t;
    }

    // 创建每个Dev的allreduce任务线程
    for (s32 i = 0; i < ndev; i++) {
        tid[i] = sal_thread_create("thread", impl_common_all_reduce_task, (void*)&para_info[i]);
        EXPECT_NE(tid[i], (sal_thread_t)NULL);
    }

    for (s32 i = 0; i < ndev; i++) {
        while (sal_thread_is_running(tid[i])) {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
    }

    // 获取stream的操作的同步信号量
    for (s32 i = 0; i < ndev; i++) {
        for (s32 j = 0; j < count; j++) {
            s8 res = result_buff[i][j];
            s8 recv = outputbuf[i][j];

            if (res != recv) {
                HCCL_ERROR(" rank :%d recvbuf[%d] :%d result_buff[%d]:%d \n", i, j, recv, j, res);
            }
        }
    }
    if (errors) {
        HCCL_ERROR("%d errors. Test FAILED.\n", errors);
    } else {
        HCCL_INFO("Test PASSED.\n");
    }
    for (s32 i = 0; i < ndev; i++) {
        hrtFree(sendbuf[i]);
        hrtFree(recvbuf[i]);
        hrtFree(result_buff[i]);
        rt_ret = aclrtDestroyStream(stream[i]);

        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }
    set_board_id(0);
    remove(file_name_t);
    EXPECT_EQ(errors, 0);
}

#endif

#if 0
TEST_F(HcclImplCommonTest, ut_hccl_impl_610_3rank_1server_allreduce_float)
{
    char file_name_t[] = "./st_hccl_impl_610_3rank_1server_allreduce_float.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(4) << g_rank_table_610_3rank_1server << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }
    else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }
    outfile.close();

    set_board_id(0x2000);

    s32 rank, errors = 0;

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;

    float* result_buff[DEV_NUM_3];
    float* sendbuf[DEV_NUM_3];
    float* recvbuf[DEV_NUM_3];
    float* inputbuf[DEV_NUM_3];
    float* outputbuf[DEV_NUM_3];

    s32 sync_value = 0;

    rtStream_t stream[DEV_NUM_3];
    sal_thread_t tid[DEV_NUM_3];
    para_t para_info[DEV_NUM_3];

    HcclDataType datatype = HCCL_DATA_TYPE_FP32;

    HcclReduceOp op = HCCL_REDUCE_SUM;
    s32 count = 1024;
    s32 ndev = DEV_NUM_3;
    HcclRootInfo rootInfo;
    ret = hccl::hcclComm::GetUniqueId(&rootInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    /** 初始化输入输出缓存 */
    for (s32 i = 0; i < ndev; i++ )
    {
        ret = hrtMalloc((void**)&sendbuf[i], count * sizeof(float));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(sendbuf[i], count * sizeof(float), 0, count * sizeof(float));
        ret = hrtMalloc((void**)&recvbuf[i], count * sizeof(float));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(recvbuf[i], count * sizeof(float), 0, count * sizeof(float));
        ret = hrtMalloc((void**)&result_buff[i], count * sizeof(float));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(result_buff[i], count * sizeof(float), 0, count * sizeof(float));
        inputbuf[i] = sendbuf[i];
        outputbuf[i] = recvbuf[i];
    }

    //sendbuf 赋值
    for (u32 j = 0; j < ndev; j++)
    {
        for (u32 i = 0; i < count; i++)
        {
            inputbuf[j][i] = 1;
        }
    }

    //resultbuf 赋值
   for (s32 i = 0; i < ndev; ++i)
 {
    for (u32 j = 0; j < count; j++)
     {
            result_buff[i][j] = ndev;
     }
    }
    for (s32 i = 0; i < ndev; ++i)
    {
        rt_ret = aclrtCreateStream(&stream[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    for (s32 i = 0; i < ndev; i++)
    {
        sal_memcpy(&para_info[i].rootInfo, sizeof(HcclRootInfo), &rootInfo, sizeof(HcclRootInfo));
        std::ostringstream identify("");
        identify << i;
        para_info[i].identify = identify.str();
        para_info[i].comm_num = ndev;
        para_info[i].device_id = i ;
        para_info[i].ranks_local = ndev;

        para_info[i].count = count;
        para_info[i].datatype = datatype;
        para_info[i].sendbuff = inputbuf[i];
        para_info[i].stream = stream[i];
        para_info[i].recvbuff = outputbuf[i];
        para_info[i].op = op;

        para_info[i].sync_addr = &sync_value;
        para_info[i].file_name = file_name_t;
    }

    // 创建每个Dev的allreduce任务线程
    for (s32 i = 0; i < ndev; i++)
    {
        tid[i] = sal_thread_create("thread", impl_common_all_reduce_task, (void*)&para_info[i]);
        EXPECT_NE(tid[i], (sal_thread_t )NULL);
    }

    for (s32 i = 0; i < ndev; i++)
    {
        while ( sal_thread_is_running(tid[i]))
        {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
        // u32 devicePhyId;
        // ret = hrtGetDevicePhyIdByIndex((u32)i, devicePhyId);
        // EXPECT_EQ(ret, HCCL_SUCCESS);
        // ret = NetworkManager::GetInstance(devicePhyId).Destroy();
    }

    //获取stream的操作的同步信号量
    for (s32 i = 0; i < ndev; i++)
  {
     for (s32 j = 0; j < count; j++)
    {
            float res = result_buff[i][j];
            float recv = outputbuf[i][j];

            if (abs(res - recv)>1e-5)
            {
                HCCL_ERROR(" rank :%d recvbuf[%d] :%d result_buff[%d]:%d \n", i, j, recv, j, res);
                errors ++;
                break;
            }
    }
        }
      if (errors)
        {
            HCCL_ERROR("%d errors. Test FAILED.\n", errors);
        }
        else
        {
            HCCL_INFO("Test PASSED.\n");
        }
    for (s32 i = 0; i < ndev; i++)
   {
        hrtFree(sendbuf[i]);
        hrtFree(recvbuf[i]);
        hrtFree(result_buff[i]);
    rt_ret = aclrtDestroyStream(stream[i]);

    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
   }
    set_board_id(0);
    remove(file_name_t);
    EXPECT_EQ(errors, 0);
}
#endif

#if 0
TEST_F(HcclImplCommonTest, ut_hccl_impl_610_2rank_1server_allreduce_float)
{
    char file_name_t[] = "./st_hccl_impl_610_2rank_1server_allreduce_float.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(4) << g_rank_table_610_2rank_1server << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }
    else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }
    outfile.close();

    set_board_id(0x2000);

    s32 rank, errors = 0;

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;

    float* result_buff[DEV_NUM_2];
    float* sendbuf[DEV_NUM_2];
    float* recvbuf[DEV_NUM_2];
    float* inputbuf[DEV_NUM_2];
    float* outputbuf[DEV_NUM_2];

    s32 sync_value = 0;

    rtStream_t stream[DEV_NUM_2];
    sal_thread_t tid[DEV_NUM_2];
    para_t para_info[DEV_NUM_2];

    HcclDataType datatype = HCCL_DATA_TYPE_FP32;

    HcclReduceOp op = HCCL_REDUCE_SUM;
    s32 count = 1024;
    s32 ndev = DEV_NUM_2;
    HcclRootInfo rootInfo;
    ret = hccl::hcclComm::GetUniqueId(&rootInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    /** 初始化输入输出缓存 */
    for (s32 i = 0; i < ndev; i++ )
    {
        ret = hrtMalloc((void**)&sendbuf[i], count * sizeof(float));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(sendbuf[i], count * sizeof(float), 0, count * sizeof(float));
        ret = hrtMalloc((void**)&recvbuf[i], count * sizeof(float));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(recvbuf[i], count * sizeof(float), 0, count * sizeof(float));
        ret = hrtMalloc((void**)&result_buff[i], count * sizeof(float));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(result_buff[i], count * sizeof(float), 0, count * sizeof(float));
        inputbuf[i] = sendbuf[i];
        outputbuf[i] = recvbuf[i];
    }

    //sendbuf 赋值
    for (u32 j = 0; j < ndev; j++)
    {
        for (u32 i = 0; i < count; i++)
        {
            inputbuf[j][i] = 1;
        }
    }

    //resultbuf 赋值
   for (s32 i = 0; i < ndev; ++i)
 {
    for (u32 j = 0; j < count; j++)
     {
            result_buff[i][j] = ndev;
     }
    }
    for (s32 i = 0; i < ndev; ++i)
    {
        rt_ret = aclrtCreateStream(&stream[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    for (s32 i = 0; i < ndev; i++)
    {
        sal_memcpy(&para_info[i].rootInfo, sizeof(HcclRootInfo), &rootInfo, sizeof(HcclRootInfo));
        std::ostringstream identify("");
        identify << i;
        para_info[i].identify = identify.str();
        para_info[i].comm_num = ndev;
        para_info[i].device_id = i ;
        para_info[i].ranks_local = ndev;

        para_info[i].count = count;
        para_info[i].datatype = datatype;
        para_info[i].sendbuff = inputbuf[i];
        para_info[i].stream = stream[i];
        para_info[i].recvbuff = outputbuf[i];
        para_info[i].op = op;

        para_info[i].sync_addr = &sync_value;
        para_info[i].file_name = file_name_t;
    }

    // 创建每个Dev的allreduce任务线程
    for (s32 i = 0; i < ndev; i++)
    {
        tid[i] = sal_thread_create("thread", impl_common_all_reduce_task, (void*)&para_info[i]);
        EXPECT_NE(tid[i], (sal_thread_t )NULL);
    }

    for (s32 i = 0; i < ndev; i++)
    {
        while ( sal_thread_is_running(tid[i]))
        {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
        // u32 devicePhyId;
        // ret = hrtGetDevicePhyIdByIndex((u32)i, devicePhyId);
        // EXPECT_EQ(ret, HCCL_SUCCESS);
        // ret = NetworkManager::GetInstance(devicePhyId).Destroy();
    }

    //获取stream的操作的同步信号量
    for (s32 i = 0; i < ndev; i++)
  {
     for (s32 j = 0; j < count; j++)
    {
            float res = result_buff[i][j];
            float recv = outputbuf[i][j];

            if (abs(res - recv)>1e-5)
            {
                HCCL_ERROR(" rank :%d recvbuf[%d] :%d result_buff[%d]:%d \n", i, j, recv, j, res);
                errors ++;
                break;
            }
    }
        }
      if (errors)
        {
            HCCL_ERROR("%d errors. Test FAILED.\n", errors);
        }
        else
        {
            HCCL_INFO("Test PASSED.\n");
        }
    for (s32 i = 0; i < ndev; i++)
   {
        hrtFree(sendbuf[i]);
        hrtFree(recvbuf[i]);
        hrtFree(result_buff[i]);
    rt_ret = aclrtDestroyStream(stream[i]);

    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
   }
    set_board_id(0);
    remove(file_name_t);
    EXPECT_EQ(errors, 0);
}

#endif

#if 0
TEST_F(HcclImplCommonTest, ut_hccl_impl_610_8rank_1server_broadcast_char)
{
    char file_name_t[] = "./st_hccl_impl_610_8rank_1server_broadcast_char.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(4) << g_rank_table_610_8rank_1server << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }
    else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }
    outfile.close();

    set_board_id(0x2000);

    s32 rank, errors = 0;

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;

    s8* result_buff[DEV_NUM_8];
    s8* sendbuf[DEV_NUM_8];
    s8* recvbuf[DEV_NUM_8];

    s32 sync_value = 0;

    rtStream_t stream[DEV_NUM_8];
    sal_thread_t tid[DEV_NUM_8];
    para_t para_info[DEV_NUM_8];

    HcclDataType datatype = HCCL_DATA_TYPE_INT8;

    HcclReduceOp op = HCCL_REDUCE_SUM;
    s32 count = 1024;
    s32 ndev = DEV_NUM_8;
    HcclRootInfo rootInfo;
    ret = hccl::hcclComm::GetUniqueId(&rootInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    /** 初始化输入输出缓存 */
    for (s32 i = 0; i < ndev; i++ )
    {
        ret = hrtMalloc((void**)&sendbuf[i], count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(sendbuf[i], count * sizeof(s8), 0, count * sizeof(s8));
        ret = hrtMalloc((void**)&recvbuf[i], count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(recvbuf[i], count * sizeof(s8), 0, count * sizeof(s8));
        ret = hrtMalloc((void**)&result_buff[i], count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(result_buff[i], count * sizeof(s8), 0, count * sizeof(s8));
    }

    //sendbuf 赋值
    for (u32 i = 0; i < count; i++)
    {
        sendbuf[0][i] = 1;
    }


    //resultbuf 赋值
   for (s32 i = 0; i < ndev; ++i)
 {
    for (u32 j = 0; j < count; j++)
     {
            result_buff[i][j] = 1;
     }
    }
    for (s32 i = 0; i < ndev; ++i)
    {
        rt_ret = aclrtCreateStream(&stream[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    for (s32 i = 0; i < ndev; i++)
    {
        sal_memcpy(&para_info[i].rootInfo, sizeof(HcclRootInfo), &rootInfo, sizeof(HcclRootInfo));
        std::ostringstream identify("");
        identify << i;
        para_info[i].identify = identify.str();
        para_info[i].comm_num = ndev;
        para_info[i].device_id = i ;
        para_info[i].ranks_local = ndev;

        para_info[i].count = count;
        para_info[i].datatype = datatype;
        para_info[i].sendbuff = sendbuf[i];
        para_info[i].stream = stream[i];
        para_info[i].recvbuff = recvbuf[i];
        para_info[i].op = op;
        para_info[i].root = 0;

        para_info[i].sync_addr = &sync_value;
        para_info[i].file_name = file_name_t;
    }

    // 创建每个Dev的allreduce任务线程
    for (s32 i = 0; i < ndev; i++)
    {
        tid[i] = sal_thread_create("thread", impl_common_broadcast_task, (void*)&para_info[i]);
        EXPECT_NE(tid[i], (sal_thread_t )NULL);
    }

    for (s32 i = 0; i < ndev; i++)
    {
        while ( sal_thread_is_running(tid[i]))
        {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
        // u32 devicePhyId;
        // ret = hrtGetDevicePhyIdByIndex((u32)i, devicePhyId);
        // EXPECT_EQ(ret, HCCL_SUCCESS);
        // ret = NetworkManager::GetInstance(devicePhyId).Destroy();
    }

    //获取stream的操作的同步信号量
    for (s32 i = 0; i < ndev; i++) {
        for (s32 j = 0; j < count; j++) {
            s8 res = result_buff[i][j];
            s8 recv = sendbuf[i][j];

            if (res != recv)
            {
                HCCL_ERROR(" rank :%d recvbuf[%d] :%d result_buff[%d]:%d \n", i, j, recv, j, res);
                errors ++;
                break;
            }
        }
    }
      if (errors)
        {
            HCCL_ERROR("%d errors. Test FAILED.\n", errors);
        }
        else
        {
            HCCL_INFO("Test PASSED.\n");
        }
    for (s32 i = 0; i < ndev; i++)
   {
        hrtFree(sendbuf[i]);
        hrtFree(recvbuf[i]);
        hrtFree(result_buff[i]);
    rt_ret = aclrtDestroyStream(stream[i]);

    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
   }
    set_board_id(0);
    remove(file_name_t);
    EXPECT_EQ(errors, 0);
}
#endif

#if 0
TEST_F(HcclImplCommonTest, ut_hccl_impl_610_8rank_1server_broadcast_float)
{
    setenv("PROFILING_MODE", "true", 1);
    char file_name_t[] = "./st_hccl_impl_610_8rank_1server_broadcast_float.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(4) << g_rank_table_610_8rank_1server << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }
    else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }
    outfile.close();

    set_board_id(0x2000);

    s32 rank, errors = 0;

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;

    float* result_buff[DEV_NUM_8];
    float* sendbuf[DEV_NUM_8];
    float* recvbuf[DEV_NUM_8];

    s32 sync_value = 0;

    rtStream_t stream[DEV_NUM_8];
    sal_thread_t tid[DEV_NUM_8];
    para_t para_info[DEV_NUM_8];

    HcclDataType datatype = HCCL_DATA_TYPE_FP32;

    HcclReduceOp op = HCCL_REDUCE_SUM;
    s32 count = 1024;
    s32 ndev = DEV_NUM_8;
    HcclRootInfo rootInfo;
    ret = hccl::hcclComm::GetUniqueId(&rootInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    /** 初始化输入输出缓存 */
    for (s32 i = 0; i < ndev; i++ )
    {
        ret = hrtMalloc((void**)&sendbuf[i], count * sizeof(float) );
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(sendbuf[i], count * sizeof(float), 0, count * sizeof(float));
        ret = hrtMalloc((void**)&recvbuf[i], count * sizeof(float));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(recvbuf[i], count * sizeof(float), 0, count * sizeof(float));
        ret = hrtMalloc((void**)&result_buff[i], count * sizeof(float));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(result_buff[i], count * sizeof(float), 0, count * sizeof(float));
    }

    //sendbuf 赋值
    for (u32 i = 0; i < count; i++)
    {
        sendbuf[0][i] = 1;
    }


    //resultbuf 赋值
   for (s32 i = 0; i < ndev; ++i)
 {
    for (u32 j = 0; j < count; j++)
     {
            result_buff[i][j] = 1;
     }
    }
    for (s32 i = 0; i < ndev; ++i)
    {
        rt_ret = aclrtCreateStream(&stream[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    for (s32 i = 0; i < ndev; i++)
    {
        sal_memcpy(&para_info[i].rootInfo, sizeof(HcclRootInfo), &rootInfo, sizeof(HcclRootInfo));
        std::ostringstream identify("");
        identify << i;
        para_info[i].identify = identify.str();
        para_info[i].comm_num = ndev;
        para_info[i].device_id = i ;
        para_info[i].ranks_local = ndev;

        para_info[i].count = count;
        para_info[i].datatype = datatype;
        para_info[i].sendbuff = sendbuf[i];
        para_info[i].stream = stream[i];
        para_info[i].recvbuff = recvbuf[i];
        para_info[i].op = op;
        para_info[i].root = 0;

        para_info[i].sync_addr = &sync_value;
        para_info[i].file_name = file_name_t;
    }

    // 创建每个Dev的allreduce任务线程
    for (s32 i = 0; i < ndev; i++)
    {
        tid[i] = sal_thread_create("thread", impl_common_broadcast_task, (void*)&para_info[i]);
        EXPECT_NE(tid[i], (sal_thread_t )NULL);
    }

    for (s32 i = 0; i < ndev; i++)
    {
        while ( sal_thread_is_running(tid[i]))
        {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
        // u32 devicePhyId;
        // ret = hrtGetDevicePhyIdByIndex((u32)i, devicePhyId);
        // EXPECT_EQ(ret, HCCL_SUCCESS);
        // ret = NetworkManager::GetInstance(devicePhyId).Destroy();
    }

    //获取stream的操作的同步信号量
    for (s32 i = 0; i < ndev; i++)
  {
     for (s32 j = 0; j < count; j++)
    {
            float res = result_buff[i][j];
            float recv = sendbuf[i][j];

            if (abs(res - recv)>1e-5)
            {
                HCCL_ERROR(" rank :%d recvbuf[%d] :%d result_buff[%d]:%d \n", i, j, recv, j, res);
                errors ++;
                break;
            }
    }
        }
      if (errors)
        {
            HCCL_ERROR("%d errors. Test FAILED.\n", errors);
        }
        else
        {
            HCCL_INFO("Test PASSED.\n");
        }
    for (s32 i = 0; i < ndev; i++)
   {
        hrtFree(sendbuf[i]);
        hrtFree(recvbuf[i]);
        hrtFree(result_buff[i]);
    rt_ret = aclrtDestroyStream(stream[i]);

    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
   }
    set_board_id(0);
    remove(file_name_t);
    EXPECT_EQ(errors, 0);
    setenv("PROFILING_MODE", "false", 1);
}
#endif

#if 0
TEST_F(HcclImplCommonTest, ut_hccl_impl_610_2rank_1server_broadcast_char)
{
    char file_name_t[] = "./st_hccl_impl_610_2rank_1server_broadcast_char.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(4) << g_rank_table_610_2rank_1server << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }
    else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }
    outfile.close();

    set_board_id(0x2000);

    s32 rank, errors = 0;

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;

    s8* result_buff[DEV_NUM_2];
    s8* sendbuf[DEV_NUM_2];
    s8* recvbuf[DEV_NUM_2];
    s32 sync_value = 0;

    rtStream_t stream[DEV_NUM_2];
    sal_thread_t tid[DEV_NUM_2];
    para_t para_info[DEV_NUM_2];

    HcclDataType datatype = HCCL_DATA_TYPE_INT8;

    HcclReduceOp op = HCCL_REDUCE_SUM;
    s32 count = 1024;
    s32 ndev = DEV_NUM_2;
    HcclRootInfo rootInfo;
    ret = hccl::hcclComm::GetUniqueId(&rootInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    /** 初始化输入输出缓存 */
    for (s32 i = 0; i < ndev; i++ )
    {
        ret = hrtMalloc((void**)&sendbuf[i], count * sizeof(s8) );
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(sendbuf[i], count * sizeof(s8) , 0, count * sizeof(s8));
        ret = hrtMalloc((void**)&recvbuf[i], count * sizeof(s8) );
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(recvbuf[i], count * sizeof(s8) , 0, count * sizeof(s8) );
        ret = hrtMalloc((void**)&result_buff[i], count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(result_buff[i], count * sizeof(s8), 0, count * sizeof(s8));
    }

    //sendbuf 赋值
    for (u32 i = 0; i < count; i++)
    {
        sendbuf[0][i] = 1;
    }


    //resultbuf 赋值
   for (s32 i = 0; i < ndev; ++i)
 {
    for (u32 j = 0; j < count; j++)
     {
            result_buff[i][j] = 1;
     }
    }
    for (s32 i = 0; i < ndev; ++i)
    {
        rt_ret = aclrtCreateStream(&stream[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    for (s32 i = 0; i < ndev; i++)
    {
        sal_memcpy(&para_info[i].rootInfo, sizeof(HcclRootInfo), &rootInfo, sizeof(HcclRootInfo));
        std::ostringstream identify("");
        identify << i;
        para_info[i].identify = identify.str();
        para_info[i].comm_num = ndev;
        para_info[i].device_id = i ;
        para_info[i].ranks_local = ndev;

        para_info[i].count = count;
        para_info[i].datatype = datatype;
        para_info[i].sendbuff = sendbuf[i];
        para_info[i].stream = stream[i];
        para_info[i].recvbuff = recvbuf[i];
        para_info[i].op = op;
        para_info[i].root = 0;

        para_info[i].sync_addr = &sync_value;
        para_info[i].file_name = file_name_t;
    }

    // 创建每个Dev的allreduce任务线程
    for (s32 i = 0; i < ndev; i++)
    {
        tid[i] = sal_thread_create("thread", impl_common_broadcast_task, (void*)&para_info[i]);
        EXPECT_NE(tid[i], (sal_thread_t )NULL);
    }

    for (s32 i = 0; i < ndev; i++)
    {
        while ( sal_thread_is_running(tid[i]))
        {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
        // u32 devicePhyId;
        // ret = hrtGetDevicePhyIdByIndex((u32)i, devicePhyId);
        // EXPECT_EQ(ret, HCCL_SUCCESS);
        // ret = NetworkManager::GetInstance(devicePhyId).Destroy();
    }

    //获取stream的操作的同步信号量
    for (s32 i = 0; i < ndev; i++)
  {
     for (s32 j = 0; j < count; j++)
    {
            s8 res = result_buff[i][j];
            s8 recv = sendbuf[i][j];

            if (res != recv)
            {
                HCCL_ERROR(" rank :%d recvbuf[%d] :%d result_buff[%d]:%d \n", i, j, recv, j, res);
                errors ++;
                break;
            }
    }
        }
      if (errors)
        {
            HCCL_ERROR("%d errors. Test FAILED.\n", errors);
        }
        else
        {
            HCCL_INFO("Test PASSED.\n");
        }
    for (s32 i = 0; i < ndev; i++)
   {
        hrtFree(sendbuf[i]);
        hrtFree(recvbuf[i]);
        hrtFree(result_buff[i]);
    rt_ret = aclrtDestroyStream(stream[i]);

    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
   }
    set_board_id(0);
    remove(file_name_t);
    EXPECT_EQ(errors, 0);
}

#endif

#if 0
TEST_F(HcclImplCommonTest, ut_hccl_impl_610_2rank_1server_broadcast_float)
{
    char file_name_t[] = "./st_hccl_impl_610_2rank_1server_broadcast_float.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(4) << g_rank_table_610_2rank_1server << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }
    else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }
    outfile.close();

    set_board_id(0x2000);

    s32 rank, errors = 0;

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;

    float* result_buff[DEV_NUM_2];
    float* sendbuf[DEV_NUM_2];
    float* recvbuf[DEV_NUM_2];

    s32 sync_value = 0;

    rtStream_t stream[DEV_NUM_2];
    sal_thread_t tid[DEV_NUM_2];
    para_t para_info[DEV_NUM_2];

    HcclDataType datatype = HCCL_DATA_TYPE_FP32;

    HcclReduceOp op = HCCL_REDUCE_RESERVED;
    s32 count = 1024;
    s32 ndev = DEV_NUM_2;
    HcclRootInfo rootInfo;
    ret = hccl::hcclComm::GetUniqueId(&rootInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    /** 初始化输入输出缓存 */
    for (s32 i = 0; i < ndev; i++ )
    {
        ret = hrtMalloc((void**)&sendbuf[i], count * sizeof(float));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(sendbuf[i], count * sizeof(float), 0, count * sizeof(float));
        ret = hrtMalloc((void**)&recvbuf[i], count * sizeof(float));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(recvbuf[i], count * sizeof(float), 0, count * sizeof(float));
        ret = hrtMalloc((void**)&result_buff[i], count * sizeof(float));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(result_buff[i], count * sizeof(float), 0, count * sizeof(float));
    }

    //sendbuf 赋值
    for (u32 i = 0; i < count; i++)
    {
        sendbuf[0][i] = 1.0f;
    }


    //resultbuf 赋值
   for (s32 i = 0; i < ndev; ++i)
 {
    for (u32 j = 0; j < count; j++)
     {
            result_buff[i][j] = 1.0f;
     }
    }
    for (s32 i = 0; i < ndev; ++i)
    {
        rt_ret = aclrtCreateStream(&stream[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    for (s32 i = 0; i < ndev; i++)
    {
        sal_memcpy(&para_info[i].rootInfo, sizeof(HcclRootInfo), &rootInfo, sizeof(HcclRootInfo));
        std::ostringstream identify("");
        identify << i;
        para_info[i].identify = identify.str();
        para_info[i].comm_num = ndev;
        para_info[i].device_id = i ;
        para_info[i].ranks_local = ndev;

        para_info[i].count = count;
        para_info[i].datatype = datatype;
        para_info[i].sendbuff = sendbuf[i];
        para_info[i].stream = stream[i];
        para_info[i].recvbuff = recvbuf[i];
        para_info[i].op = op;
        para_info[i].root = 0;

        para_info[i].sync_addr = &sync_value;
        para_info[i].file_name = file_name_t;
    }

    // 创建每个Dev的allreduce任务线程
    for (s32 i = 0; i < ndev; i++)
    {
        tid[i] = sal_thread_create("thread", impl_common_broadcast_task, (void*)&para_info[i]);
        EXPECT_NE(tid[i], (sal_thread_t )NULL);
    }

    for (s32 i = 0; i < ndev; i++)
    {
        while ( sal_thread_is_running(tid[i]))
        {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
        // u32 devicePhyId;
        // ret = hrtGetDevicePhyIdByIndex((u32)i, devicePhyId);
        // EXPECT_EQ(ret, HCCL_SUCCESS);
        // ret = NetworkManager::GetInstance(devicePhyId).Destroy();
    }

    //获取stream的操作的同步信号量
    for (s32 i = 0; i < ndev; i++)
  {
     for (s32 j = 0; j < count; j++)
    {
            float res = result_buff[i][j];
            float recv = sendbuf[i][j];

            if (abs(res - recv)>1e-5)
            {
                HCCL_ERROR(" rank :%d recvbuf[%d] :%d result_buff[%d]:%d \n", i, j, recv, j, res);
                errors ++;
                break;
            }
    }
        }
      if (errors)
        {
            HCCL_ERROR("%d errors. Test FAILED.\n", errors);
        }
        else
        {
            HCCL_INFO("Test PASSED.\n");
        }
    for (s32 i = 0; i < ndev; i++)
   {
        hrtFree(sendbuf[i]);
        hrtFree(recvbuf[i]);
        hrtFree(result_buff[i]);
    rt_ret = aclrtDestroyStream(stream[i]);

    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
   }
    set_board_id(0);
    remove(file_name_t);
    EXPECT_EQ(errors, 0);
}

#endif
#if 0
TEST_F(HcclImplCommonTest, ut_hccl_impl_610_3rank_1server_broadcast_float)
{
    char file_name_t[] = "./st_hccl_impl_610_3rank_1server_broadcast_float.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(4) << g_rank_table_610_3rank_1server << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }
    else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }
    outfile.close();

    set_board_id(0x2000);

    s32 rank, errors = 0;

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;

    float* result_buff[DEV_NUM_3];
    float* sendbuf[DEV_NUM_3];
    float* recvbuf[DEV_NUM_3];

    s32 sync_value = 0;

    rtStream_t stream[DEV_NUM_3];
    sal_thread_t tid[DEV_NUM_3];
    para_t para_info[DEV_NUM_3];

    HcclDataType datatype = HCCL_DATA_TYPE_FP32;

    HcclReduceOp op = HCCL_REDUCE_RESERVED;
    s32 count = 1024;
    s32 ndev = DEV_NUM_3;
    HcclRootInfo rootInfo;
    ret = hccl::hcclComm::GetUniqueId(&rootInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    /** 初始化输入输出缓存 */
    for (s32 i = 0; i < ndev; i++ )
    {
        ret = hrtMalloc((void**)&sendbuf[i], count * sizeof(float));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(sendbuf[i], count * sizeof(float), 0, count * sizeof(float));
        ret = hrtMalloc((void**)&recvbuf[i], count * sizeof(float));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(recvbuf[i], count * sizeof(float), 0, count * sizeof(float));
        ret = hrtMalloc((void**)&result_buff[i], count * sizeof(float));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(result_buff[i], count * sizeof(float), 0, count * sizeof(float));
    }

    //sendbuf 赋值
    for (u32 i = 0; i < count; i++)
    {
        sendbuf[0][i] = 1.0f;
    }


    //resultbuf 赋值
   for (s32 i = 0; i < ndev; ++i)
 {
    for (u32 j = 0; j < count; j++)
     {
            result_buff[i][j] = 1.0f;
     }
    }
    for (s32 i = 0; i < ndev; ++i)
    {
        rt_ret = aclrtCreateStream(&stream[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    for (s32 i = 0; i < ndev; i++)
    {
        sal_memcpy(&para_info[i].rootInfo, sizeof(HcclRootInfo), &rootInfo, sizeof(HcclRootInfo));
        std::ostringstream identify("");
        identify << i;
        para_info[i].identify = identify.str();
        para_info[i].comm_num = ndev;
        para_info[i].device_id = i ;
        para_info[i].ranks_local = ndev;

        para_info[i].count = count;
        para_info[i].datatype = datatype;
        para_info[i].sendbuff = sendbuf[i];
        para_info[i].stream = stream[i];
        para_info[i].recvbuff = recvbuf[i];
        para_info[i].op = op;
        para_info[i].root = 0;

        para_info[i].sync_addr = &sync_value;
        para_info[i].file_name = file_name_t;
    }

    // 创建每个Dev的allreduce任务线程
    for (s32 i = 0; i < ndev; i++)
    {
        tid[i] = sal_thread_create("thread", impl_common_broadcast_task, (void*)&para_info[i]);
        EXPECT_NE(tid[i], (sal_thread_t )NULL);
    }

    for (s32 i = 0; i < ndev; i++)
    {
        while ( sal_thread_is_running(tid[i]))
        {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
        // u32 devicePhyId;
        // ret = hrtGetDevicePhyIdByIndex((u32)i, devicePhyId);
        // EXPECT_EQ(ret, HCCL_SUCCESS);
        // ret = NetworkManager::GetInstance(devicePhyId).Destroy();
    }

    //获取stream的操作的同步信号量
    for (s32 i = 0; i < ndev; i++)
  {
     for (s32 j = 0; j < count; j++)
    {
            float res = result_buff[i][j];
            float recv = sendbuf[i][j];

            if (abs(res - recv)>1e-5)
            {
                HCCL_ERROR(" rank :%d recvbuf[%d] :%d result_buff[%d]:%d \n", i, j, recv, j, res);
                errors ++;
                break;
            }
    }
        }
      if (errors)
        {
            HCCL_ERROR("%d errors. Test FAILED.\n", errors);
        }
        else
        {
            HCCL_INFO("Test PASSED.\n");
        }
    for (s32 i = 0; i < ndev; i++)
   {
        hrtFree(sendbuf[i]);
        hrtFree(recvbuf[i]);
        hrtFree(result_buff[i]);
    rt_ret = aclrtDestroyStream(stream[i]);

    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
   }
    set_board_id(0);
    remove(file_name_t);
    EXPECT_EQ(errors, 0);
}

#endif
#if 0
TEST_F(HcclImplCommonTest, ut_hccl_impl_610_3rank_1server_broadcast_char)
{
    char file_name_t[] = "./st_hccl_impl_610_3rank_1server_broadcast_char.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(4) << g_rank_table_610_3rank_1server << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }
    else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }
    outfile.close();

    set_board_id(0x2000);

    s32 rank, errors = 0;

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;

    s8* result_buff[DEV_NUM_3];
    s8* sendbuf[DEV_NUM_3];
    s8* recvbuf[DEV_NUM_3];

    s32 sync_value = 0;

    rtStream_t stream[DEV_NUM_3];
    sal_thread_t tid[DEV_NUM_3];
    para_t para_info[DEV_NUM_3];

    HcclDataType datatype = HCCL_DATA_TYPE_INT8;

    HcclReduceOp op = HCCL_REDUCE_RESERVED;
    s32 count = 1024;
    s32 ndev = DEV_NUM_3;
    HcclRootInfo rootInfo;
    ret = hccl::hcclComm::GetUniqueId(&rootInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    /** 初始化输入输出缓存 */
    for (s32 i = 0; i < ndev; i++ )
    {
        ret = hrtMalloc((void**)&sendbuf[i], count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(sendbuf[i], count * sizeof(s8), 0, count * sizeof(s8));
        ret = hrtMalloc((void**)&recvbuf[i], count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(recvbuf[i], count * sizeof(s8), 0, count * sizeof(s8));
        ret = hrtMalloc((void**)&result_buff[i], count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(result_buff[i], count * sizeof(s8), 0, count * sizeof(s8));
    }

    //sendbuf 赋值
    for (u32 i = 0; i < count; i++)
    {
        sendbuf[0][i] = 1;
    }


    //resultbuf 赋值
   for (s32 i = 0; i < ndev; ++i)
 {
    for (u32 j = 0; j < count; j++)
     {
            result_buff[i][j] = 1;
     }
    }
    for (s32 i = 0; i < ndev; ++i)
    {
        rt_ret = aclrtCreateStream(&stream[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    for (s32 i = 0; i < ndev; i++)
    {
        sal_memcpy(&para_info[i].rootInfo, sizeof(HcclRootInfo), &rootInfo, sizeof(HcclRootInfo));
        std::ostringstream identify("");
        identify << i;
        para_info[i].identify = identify.str();
        para_info[i].comm_num = ndev;
        para_info[i].device_id = i ;
        para_info[i].ranks_local = ndev;

        para_info[i].count = count;
        para_info[i].datatype = datatype;
        para_info[i].sendbuff = sendbuf[i];
        para_info[i].stream = stream[i];
        para_info[i].recvbuff = recvbuf[i];
        para_info[i].op = op;
        para_info[i].root = 0;

        para_info[i].sync_addr = &sync_value;
        para_info[i].file_name = file_name_t;
    }

    // 创建每个Dev的allreduce任务线程
    for (s32 i = 0; i < ndev; i++)
    {
        tid[i] = sal_thread_create("thread", impl_common_broadcast_task, (void*)&para_info[i]);
        EXPECT_NE(tid[i], (sal_thread_t )NULL);
    }

    for (s32 i = 0; i < ndev; i++)
    {
        while ( sal_thread_is_running(tid[i]))
        {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
        // u32 devicePhyId;
        // ret = hrtGetDevicePhyIdByIndex((u32)i, devicePhyId);
        // EXPECT_EQ(ret, HCCL_SUCCESS);
        // ret = NetworkManager::GetInstance(devicePhyId).Destroy();
    }

    //获取stream的操作的同步信号量
    for (s32 i = 0; i < ndev; i++) {
        for (s32 j = 0; j < count; j++) {
            s8 res = result_buff[i][j];
            s8 recv = sendbuf[i][j];

            if (res != recv)
            {
                HCCL_ERROR(" rank :%d recvbuf[%d] :%d result_buff[%d]:%d \n", i, j, recv, j, res);
                errors ++;
                break;
            }
        }
    }
      if (errors)
        {
            HCCL_ERROR("%d errors. Test FAILED.\n", errors);
        }
        else
        {
            HCCL_INFO("Test PASSED.\n");
        }
    for (s32 i = 0; i < ndev; i++)
   {
        hrtFree(sendbuf[i]);
        hrtFree(recvbuf[i]);
        hrtFree(result_buff[i]);
    rt_ret = aclrtDestroyStream(stream[i]);

    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
   }
    set_board_id(0);
    remove(file_name_t);
    EXPECT_EQ(errors, 0);
}
#endif
#if 0
TEST_F(HcclImplCommonTest, ut_hccl_impl_610_5rank_1server_broadcast_float)
{
    char file_name_t[] = "./st_hccl_impl_610_5rank_1server_broadcast_float.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(4) << g_rank_table_610_5rank_1server << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }
    else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }
    outfile.close();

    set_board_id(0x2000);

    s32 rank, errors = 0;

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;

    float* result_buff[DEV_NUM_5];
    float* sendbuf[DEV_NUM_5];
    float* recvbuf[DEV_NUM_5];

    s32 sync_value = 0;

    rtStream_t stream[DEV_NUM_5];
    sal_thread_t tid[DEV_NUM_5];
    para_t para_info[DEV_NUM_5];

    HcclDataType datatype = HCCL_DATA_TYPE_FP32;

    HcclReduceOp op = HCCL_REDUCE_RESERVED;
    s32 count = 1024;
    s32 ndev = DEV_NUM_5;
    HcclRootInfo rootInfo;
    ret = hccl::hcclComm::GetUniqueId(&rootInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    /** 初始化输入输出缓存 */
    for (s32 i = 0; i < ndev; i++ )
    {
        ret = hrtMalloc((void**)&sendbuf[i], count * sizeof(float));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(sendbuf[i], count * sizeof(float), 0, count * sizeof(float));
        ret = hrtMalloc((void**)&recvbuf[i], count * sizeof(float));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(recvbuf[i], count * sizeof(float), 0, count * sizeof(float));
        ret = hrtMalloc((void**)&result_buff[i], count * sizeof(float));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(result_buff[i], count * sizeof(float), 0, count * sizeof(float));
    }

    //sendbuf 赋值
    for (u32 i = 0; i < count; i++)
    {
        sendbuf[0][i] = 1.0f;
    }


    //resultbuf 赋值
   for (s32 i = 0; i < ndev; ++i)
 {
    for (u32 j = 0; j < count; j++)
     {
            result_buff[i][j] = 1.0f;
     }
    }
    for (s32 i = 0; i < ndev; ++i)
    {
        rt_ret = aclrtCreateStream(&stream[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    for (s32 i = 0; i < ndev; i++)
    {
        sal_memcpy(&para_info[i].rootInfo, sizeof(HcclRootInfo), &rootInfo, sizeof(HcclRootInfo));
        std::ostringstream identify("");
        identify << i;
        para_info[i].identify = identify.str();
        para_info[i].comm_num = ndev;
        para_info[i].device_id = i ;
        para_info[i].ranks_local = ndev;

        para_info[i].count = count;
        para_info[i].datatype = datatype;
        para_info[i].sendbuff = sendbuf[i];
        para_info[i].stream = stream[i];
        para_info[i].recvbuff = recvbuf[i];
        para_info[i].op = op;
        para_info[i].root = 0;

        para_info[i].sync_addr = &sync_value;
        para_info[i].file_name = file_name_t;
    }

    // 创建每个Dev的allreduce任务线程
    for (s32 i = 0; i < ndev; i++)
    {
        tid[i] = sal_thread_create("thread", impl_common_broadcast_task, (void*)&para_info[i]);
        EXPECT_NE(tid[i], (sal_thread_t )NULL);
    }

    for (s32 i = 0; i < ndev; i++)
    {
        while ( sal_thread_is_running(tid[i]))
        {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
        // u32 devicePhyId;
        // ret = hrtGetDevicePhyIdByIndex((u32)i, devicePhyId);
        // EXPECT_EQ(ret, HCCL_SUCCESS);
        // ret = NetworkManager::GetInstance(devicePhyId).Destroy();
    }

    //获取stream的操作的同步信号量
    for (s32 i = 0; i < ndev; i++)
  {
     for (s32 j = 0; j < count; j++)
    {
            float res = result_buff[i][j];
            float recv = sendbuf[i][j];

            if (abs(res - recv)>1e-5)
            {
                HCCL_ERROR(" rank :%d recvbuf[%d] :%d result_buff[%d]:%d \n", i, j, recv, j, res);
                errors ++;
                break;
            }
    }
        }
      if (errors)
        {
            HCCL_ERROR("%d errors. Test FAILED.\n", errors);
        }
        else
        {
            HCCL_INFO("Test PASSED.\n");
        }
    for (s32 i = 0; i < ndev; i++)
   {
        hrtFree(sendbuf[i]);
        hrtFree(recvbuf[i]);
        hrtFree(result_buff[i]);
    rt_ret = aclrtDestroyStream(stream[i]);

    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
   }
    set_board_id(0);
    remove(file_name_t);
    EXPECT_EQ(errors, 0);
}

#endif
#if 0
TEST_F(HcclImplCommonTest, ut_hccl_impl_610_5rank_1server_broadcast_char)
{
    char file_name_t[] = "./st_hccl_impl_610_5rank_1server_broadcast_char.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(4) << g_rank_table_610_5rank_1server << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }
    else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }
    outfile.close();

    set_board_id(0x2000);

    s32 rank, errors = 0;

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;

    s8* result_buff[DEV_NUM_5];
    s8* sendbuf[DEV_NUM_5];
    s8* recvbuf[DEV_NUM_5];

    s32 sync_value = 0;

    rtStream_t stream[DEV_NUM_5];
    sal_thread_t tid[DEV_NUM_5];
    para_t para_info[DEV_NUM_5];

    HcclDataType datatype = HCCL_DATA_TYPE_INT8;

    HcclReduceOp op = HCCL_REDUCE_RESERVED;
    s32 count = 1024;
    s32 ndev = DEV_NUM_5;
    HcclRootInfo rootInfo;
    ret = hccl::hcclComm::GetUniqueId(&rootInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    /** 初始化输入输出缓存 */
    for (s32 i = 0; i < ndev; i++ )
    {
        ret = hrtMalloc((void**)&sendbuf[i], count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(sendbuf[i], count * sizeof(s8), 0, count * sizeof(s8));
        ret = hrtMalloc((void**)&recvbuf[i], count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(recvbuf[i], count * sizeof(s8), 0, count * sizeof(s8));
        ret = hrtMalloc((void**)&result_buff[i], count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(result_buff[i], count * sizeof(s8), 0, count * sizeof(s8));
    }

    //sendbuf 赋值
    for (u32 i = 0; i < count; i++)
    {
        sendbuf[0][i] = 1;
    }


    //resultbuf 赋值
   for (s32 i = 0; i < ndev; ++i)
 {
    for (u32 j = 0; j < count; j++)
     {
            result_buff[i][j] = 1;
     }
    }
    for (s32 i = 0; i < ndev; ++i)
    {
        rt_ret = aclrtCreateStream(&stream[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    for (s32 i = 0; i < ndev; i++)
    {
        sal_memcpy(&para_info[i].rootInfo, sizeof(HcclRootInfo), &rootInfo, sizeof(HcclRootInfo));
        std::ostringstream identify("");
        identify << i;
        para_info[i].identify = identify.str();
        para_info[i].comm_num = ndev;
        para_info[i].device_id = i ;
        para_info[i].ranks_local = ndev;

        para_info[i].count = count;
        para_info[i].datatype = datatype;
        para_info[i].sendbuff = sendbuf[i];
        para_info[i].stream = stream[i];
        para_info[i].recvbuff = recvbuf[i];
        para_info[i].op = op;
        para_info[i].root = 0;

        para_info[i].sync_addr = &sync_value;
        para_info[i].file_name = file_name_t;
    }

    // 创建每个Dev的allreduce任务线程
    for (s32 i = 0; i < ndev; i++)
    {
        tid[i] = sal_thread_create("thread", impl_common_broadcast_task, (void*)&para_info[i]);
        EXPECT_NE(tid[i], (sal_thread_t )NULL);
    }

    for (s32 i = 0; i < ndev; i++)
    {
        while ( sal_thread_is_running(tid[i]))
        {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
        // u32 devicePhyId;
        // ret = hrtGetDevicePhyIdByIndex((u32)i, devicePhyId);
        // EXPECT_EQ(ret, HCCL_SUCCESS);
        // ret = NetworkManager::GetInstance(devicePhyId).Destroy();
    }

    //获取stream的操作的同步信号量
    for (s32 i = 0; i < ndev; i++) {
        for (s32 j = 0; j < count; j++) {
            s8 res = result_buff[i][j];
            s8 recv = sendbuf[i][j];

            if (res != recv)
            {
                HCCL_ERROR(" rank :%d recvbuf[%d] :%d result_buff[%d]:%d \n", i, j, recv, j, res);
                errors ++;
                break;
            }
        }
    }
      if (errors)
        {
            HCCL_ERROR("%d errors. Test FAILED.\n", errors);
        }
        else
        {
            HCCL_INFO("Test PASSED.\n");
        }
    for (s32 i = 0; i < ndev; i++)
   {
        hrtFree(sendbuf[i]);
        hrtFree(recvbuf[i]);
        hrtFree(result_buff[i]);
    rt_ret = aclrtDestroyStream(stream[i]);

    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
   }
    set_board_id(0);
    remove(file_name_t);
    EXPECT_EQ(errors, 0);
}
#endif

#if 1
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

#endif

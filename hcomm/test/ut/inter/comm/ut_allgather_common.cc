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

#include "hcom_pub.h"
#include "hccl/base.h"
#include <hccl/hccl_types.h>
#include "llt_hccl_stub_pub.h"
#include <sys/mman.h>
#include <fcntl.h>
#include "hccl_comm_pub.h"
#include "sal.h"
#include "tsd/tsd_client.h"
#include "dltdt_function.h"
#include "dlra_function.h"
#include "externalinput_pub.h"
#include "config.h"
#include "topoinfo_ranktableParser_pub.h"
#include "rank_consistentcy_checker.h"

#include "../misc/st_rank_table.h"
#include <iostream>
#include <fstream>
#include "network_manager_pub.h"
#include "param_check_pub.h"
#include "dltdt_function.h"
#include "heartbeat.h"
#include "externalinput.h"

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

void* impl_common_all_gather_task(void* parg)
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

    char* charModel = new char;
    rtModel_t model = (void*)charModel;

    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, para_info->identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    sal_memcpy(hcom_info.params.id.internal, sizeof(HcclRootInfo), &para_info->rootInfo, sizeof(HcclRootInfo));

    hcom_info.pComm.reset(new (std::nothrow) hccl::hcclComm(209715200, 209715200));

    CommConfig commConfig("hccl_world_group");
    ret = hcom_info.pComm->init(hcom_info.params, commConfig, hcom_info.rankTable);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] task all_reduce fails", para_info->device_id);
    }

    bool swapped;

    //-----------------Get Workspace Resource Start------------------//
    u64 stream_list_size = 0;
    ret = hcom_info.pComm->GetWorkspaceSubStreamNum(stream_list_size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HCCL_INFO("get stream_list_size[%d] success", stream_list_size);
    rtError_t rt_ret;
    vector<HcclRtStream> streamList(stream_list_size);
    u32 rankSize = 0;
    u64 memSize = 0;
    void* memptr = nullptr;

    // 生成从stream
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = aclrtCreateStreamWithConfig(&streamList[i], 0, ACL_STREAM_PERSISTENT);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
        // 从流bind到model
        rt_ret = rtModelBindStream(model, streamList[i], RT_MODEL_WAIT_ACTIVE_STREAM);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    ret = hcom_info.pComm->GetRankSize(rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hcom_info.pComm->GetWorkspaceMemSize(
        HCCL_KERNEL_OP_TYPE_ALLGATHER, para_info->count, para_info->datatype, rankSize, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtMalloc(&memptr, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hcom_info.pComm->SetWorkspaceResource("tag_impl_common_all_gather_task", memptr, memSize, streamList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    //-----------------Get Workspace Resource End------------------//
    rank_num_tmp = *(para_info->sync_addr) - 1;

    do {
        rank_num_tmp += 1;

        swapped = __sync_bool_compare_and_swap(para_info->sync_addr, rank_num_tmp, rank_num_tmp + 1);
    } while (!swapped);

    while (*(para_info->sync_addr) < para_info->ranks_local) {
        sched_yield();
    } // linux提供一个系统调用运行进程主动让出执行权

    __sync_synchronize(); // 插入内存屏障，对顺序性有要求，但是有没有使用lock指令的时候

    ret = hcom_info.pComm->AllGather(
        "tag_impl_common_all_gather_task", para_info->sendbuff, para_info->recvbuff, para_info->count,
        para_info->datatype, para_info->stream);

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] task hcclall_gather fails", para_info->device_id);
    }

    rt_ret = RT_ERROR_NONE;
    rt_ret = aclrtSynchronizeStream(para_info->stream);

    if (rt_ret != RT_ERROR_NONE) {
        HCCL_ERROR("rank[%d] task allgather fails", hcom_info.params.rank);
    }
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = rtModelUnbindStream(model, streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
        rt_ret = aclrtDestroyStream(streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }
    hrtFree(memptr);

    delete charModel;
    charModel = nullptr;
    RankConsistentcyChecker::GetInstance().ClearCheckInfo();
    return (NULL);
}

class AllGatherCommonTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AllGatherCommonTest SetUP" << std::endl;
        DlTdtFunction::GetInstance().DlTdtFunctionInit();
    }
    static void TearDownTestCase() { std::cout << "AllGatherCommonTest TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        static s32 call_cnt = 0;
        DlTdtFunction::GetInstance().DlTdtFunctionInit();
        DlRaFunction::GetInstance().DlRaFunctionInit();
        string name = std::to_string(call_cnt++) + "_" + __PRETTY_FUNCTION__;
        ra_set_shm_name(name.c_str());
        ra_set_test_type(0, "UT_TEST");
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&Heartbeat::Init).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&Heartbeat::DeInit).stubs().will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }
};

#define DEV_NUM_8 8
#define DEV_NUM_4 4
#define DEV_NUM_5 5
#define DEV_NUM_2 2
#define DEV_NUM_3 3

#if 1
TEST_F(AllGatherCommonTest, ut_hccl_impl_610_8rank_1server_all_gather_char)
{
    char file_name_t[] = "./st_hccl_impl_610_8rank_1server_all_gather_char.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << g_rank_table_610_8rank_1server << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
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
        ret = hrtMalloc((void**)&recvbuf[i], ndev * count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(recvbuf[i], ndev * count * sizeof(s8), 0, count * sizeof(s8));
        ret = hrtMalloc((void**)&result_buff[i], ndev * count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(result_buff[i], ndev * count * sizeof(s8), 0, count * sizeof(s8));
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
        for (u32 j = 0; j < ndev * count; j++) {
            result_buff[i][j] = 1;
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

    // 创建每个Dev的all_gather任务线程
    for (s32 i = 0; i < ndev; i++) {
        tid[i] = sal_thread_create("thread", impl_common_all_gather_task, (void*)&para_info[i]);
        EXPECT_NE(tid[i], (sal_thread_t)NULL);
    }

    for (s32 i = 0; i < ndev; i++) {
        while (sal_thread_is_running(tid[i])) {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
    }

    // 获取stream的操作的同步信号量

    for (s32 i = 0; i < ndev; i++) {
        for (s32 j = 0; j < ndev * count; j++) {
            s8 res = result_buff[i][j];
            s8 recv = outputbuf[i][j];

            if (res != recv) {
                HCCL_ERROR(" rank :%d recvbuf[%d] :%d result_buff[%d]:%d \n", i, j, recv, j, res);
                errors++;
                break;
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
TEST_F(AllGatherCommonTest, ut_hccl_impl_610_8rank_1server_all_gather_float)
{
    setenv("PROFILING_MODE", "true", 1);
    char file_name_t[] = "./st_hccl_impl_610_8rank_1server_all_gather_float.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << g_rank_table_610_8rank_1server << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
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
    for (s32 i = 0; i < ndev; i++) {
        ret = hrtMalloc((void**)&sendbuf[i], count * sizeof(float));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(sendbuf[i], count * sizeof(float), 0, count * sizeof(float));
        ret = hrtMalloc((void**)&recvbuf[i], ndev * count * sizeof(float));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(recvbuf[i], ndev * count * sizeof(float), 0, count * sizeof(float));

        result_buff[i] = (float*)sal_malloc(ndev * count * sizeof(float));
        sal_memset(result_buff[i], ndev * count * sizeof(float), 0, count * sizeof(float));
        inputbuf[i] = sendbuf[i];
        outputbuf[i] = recvbuf[i];
    }

    // sendbuf 赋值
    for (u32 j = 0; j < ndev; j++) {
        for (u32 i = 0; i < count; i++) {
            inputbuf[j][i] = 1.0f;
        }
    }

    // resultbuf 赋值
    for (s32 i = 0; i < ndev; ++i) {
        for (u32 j = 0; j < ndev * count; j++) {
            result_buff[i][j] = 1.0f;
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

    // 创建每个Dev的all_gather任务线程
    for (s32 i = 0; i < ndev; i++) {
        tid[i] = sal_thread_create("thread", impl_common_all_gather_task, (void*)&para_info[i]);
        EXPECT_NE(tid[i], (sal_thread_t)NULL);
    }

    for (s32 i = 0; i < ndev; i++) {
        while (sal_thread_is_running(tid[i])) {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
        // u32 devicePhyId;
        // ret = hrtGetDevicePhyIdByIndex((u32)i, devicePhyId);
        // EXPECT_EQ(ret, HCCL_SUCCESS);
        // ret = NetworkManager::GetInstance(devicePhyId).Destroy();
    }

    // 获取stream的操作的同步信号量
    for (s32 i = 0; i < ndev; i++) {
        for (s32 j = 0; j < ndev * count; j++) {
            float res = result_buff[i][j];
            float recv = outputbuf[i][j];

            if (abs(res - recv) > 1e-6) {
                HCCL_ERROR(" rank :%d recvbuf[%d] :%d result_buff[%d]:%d \n", i, j, recv, j, res);
                errors++;
                break;
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
        sal_free(result_buff[i]);
        rt_ret = aclrtDestroyStream(stream[i]);

        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }
    set_board_id(0);
    remove(file_name_t);
    EXPECT_EQ(errors, 0);
    setenv("PROFILING_MODE", "false", 1);
}
#endif

#if 1
TEST_F(AllGatherCommonTest, ut_hccl_impl_610_5rank_1server_all_gather_char)
{
    char file_name_t[] = "./st_hccl_impl_610_5rank_1server_all_gather_char.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << g_rank_table_610_5rank_1server << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
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
    for (s32 i = 0; i < ndev; i++) {
        ret = hrtMalloc((void**)&sendbuf[i], count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(sendbuf[i], count * sizeof(s8), 0, count * sizeof(s8));
        ret = hrtMalloc((void**)&recvbuf[i], ndev * count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(recvbuf[i], ndev * count * sizeof(s8), 0, count * sizeof(s8));
        ret = hrtMalloc((void**)&result_buff[i], ndev * count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(result_buff[i], ndev * count * sizeof(s8), 0, count * sizeof(s8));
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
        for (u32 j = 0; j < ndev * count; j++) {
            result_buff[i][j] = 1;
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

    // 创建每个Dev的all_gather任务线程
    for (s32 i = 0; i < ndev; i++) {
        tid[i] = sal_thread_create("thread", impl_common_all_gather_task, (void*)&para_info[i]);
        EXPECT_NE(tid[i], (sal_thread_t)NULL);
    }

    for (s32 i = 0; i < ndev; i++) {
        while (sal_thread_is_running(tid[i])) {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
        // u32 devicePhyId;
        // ret = hrtGetDevicePhyIdByIndex((u32)i, devicePhyId);
        // EXPECT_EQ(ret, HCCL_SUCCESS);
        // ret = NetworkManager::GetInstance(devicePhyId).Destroy();
    }

    // 获取stream的操作的同步信号量
    for (s32 i = 0; i < ndev; i++) {
        for (s32 j = 0; j < ndev * count; j++) {
            s8 res = result_buff[i][j];
            s8 recv = outputbuf[i][j];

            if (res != recv) {
                HCCL_ERROR(" rank :%d recvbuf[%d] :%d result_buff[%d]:%d \n", i, j, recv, j, res);
                errors++;
                break;
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
TEST_F(AllGatherCommonTest, ut_hccl_impl_610_4rank_1server_all_gather_char)
{
    char file_name_t[] = "./st_hccl_impl_610_4rank_1server_all_gather_char.json";
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
        ret = hrtMalloc((void**)&recvbuf[i], ndev * count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(recvbuf[i], ndev * count * sizeof(s8), 0, count * sizeof(s8));
        ret = hrtMalloc((void**)&result_buff[i], ndev * count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(result_buff[i], ndev * count * sizeof(s8), 0, count * sizeof(s8));
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
        for (u32 j = 0; j < ndev * count; j++) {
            result_buff[i][j] = 1;
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

    // 创建每个Dev的all_gather任务线程
    for (s32 i = 0; i < ndev; i++) {
        tid[i] = sal_thread_create("thread", impl_common_all_gather_task, (void*)&para_info[i]);
        EXPECT_NE(tid[i], (sal_thread_t)NULL);
    }

    for (s32 i = 0; i < ndev; i++) {
        while (sal_thread_is_running(tid[i])) {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
        // u32 devicePhyId;
        // ret = hrtGetDevicePhyIdByIndex((u32)i, devicePhyId);
        // EXPECT_EQ(ret, HCCL_SUCCESS);
        // ret = NetworkManager::GetInstance(devicePhyId).Destroy();
    }

    // 获取stream的操作的同步信号量
    for (s32 i = 0; i < ndev; i++) {
        for (s32 j = 0; j < ndev * count; j++) {
            s8 res = result_buff[i][j];
            s8 recv = outputbuf[i][j];

            if (res != recv) {
                HCCL_ERROR(" rank :%d recvbuf[%d] :%d result_buff[%d]:%d \n", i, j, recv, j, res);
                errors++;
                break;
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
TEST_F(AllGatherCommonTest, ut_hccl_impl_610_2rank_1server_all_gather_char)
{
    char file_name_t[] = "./st_hccl_impl_610_2rank_1server_all_gather_char.json";
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
        ret = hrtMalloc((void**)&recvbuf[i], ndev * count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(recvbuf[i], ndev * count * sizeof(s8), 0, count * sizeof(s8));
        ret = hrtMalloc((void**)&result_buff[i], ndev * count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(result_buff[i], ndev * count * sizeof(s8), 0, count * sizeof(s8));
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
        for (u32 j = 0; j < ndev * count; j++) {
            result_buff[i][j] = 1;
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

    // 创建每个Dev的all_gather任务线程
    for (s32 i = 0; i < ndev; i++) {
        tid[i] = sal_thread_create("thread", impl_common_all_gather_task, (void*)&para_info[i]);
        EXPECT_NE(tid[i], (sal_thread_t)NULL);
    }

    for (s32 i = 0; i < ndev; i++) {
        while (sal_thread_is_running(tid[i])) {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
        // u32 devicePhyId;
        // ret = hrtGetDevicePhyIdByIndex((u32)i, devicePhyId);
        // EXPECT_EQ(ret, HCCL_SUCCESS);
        // ret = NetworkManager::GetInstance(devicePhyId).Destroy();
    }

    // 获取stream的操作的同步信号量
    for (s32 i = 0; i < ndev; i++) {
        for (s32 j = 0; j < ndev * count; j++) {
            s8 res = result_buff[i][j];
            s8 recv = outputbuf[i][j];

            if (res != recv) {
                HCCL_ERROR(" rank :%d recvbuf[%d] :%d result_buff[%d]:%d \n", i, j, recv, j, res);
                errors++;
                break;
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
TEST_F(AllGatherCommonTest, ut_hccl_impl_610_3rank_1server_all_gather_float)
{
    char file_name_t[] = "./st_hccl_impl_610_3rank_1server_all_gather_float.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << g_rank_table_610_3rank_1server << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
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
    for (s32 i = 0; i < ndev; i++) {
        ret = hrtMalloc((void**)&sendbuf[i], count * sizeof(float));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(sendbuf[i], count * sizeof(float), 0, count * sizeof(float));
        ret = hrtMalloc((void**)&recvbuf[i], ndev * count * sizeof(float));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(recvbuf[i], ndev * count * sizeof(float), 0, count * sizeof(float));
        ret = hrtMalloc((void**)&result_buff[i], ndev * count * sizeof(float));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(result_buff[i], ndev * count * sizeof(float), 0, count * sizeof(float));
        inputbuf[i] = sendbuf[i];
        outputbuf[i] = recvbuf[i];
    }

    // sendbuf 赋值
    for (u32 j = 0; j < ndev; j++) {
        for (u32 i = 0; i < count; i++) {
            inputbuf[j][i] = 1;
        }

        // u32 devicePhyId;
        // ret = hrtGetDevicePhyIdByIndex(j, devicePhyId);
        // EXPECT_EQ(ret, HCCL_SUCCESS);
        // ret = NetworkManager::GetInstance(devicePhyId).Destroy();
    }

    // resultbuf 赋值
    for (s32 i = 0; i < ndev; ++i) {
        for (u32 j = 0; j < ndev * count; j++) {
            result_buff[i][j] = 1;
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

    // 创建每个Dev的all_gather任务线程
    for (s32 i = 0; i < ndev; i++) {
        tid[i] = sal_thread_create("thread", impl_common_all_gather_task, (void*)&para_info[i]);
        EXPECT_NE(tid[i], (sal_thread_t)NULL);
    }

    for (s32 i = 0; i < ndev; i++) {
        while (sal_thread_is_running(tid[i])) {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
    }

    // 获取stream的操作的同步信号量
    for (s32 i = 0; i < ndev; i++) {
        for (s32 j = 0; j < ndev * count; j++) {
            float res = result_buff[i][j];
            float recv = outputbuf[i][j];

            if (abs(res - recv) > 1e-5) {
                HCCL_ERROR(" rank :%d recvbuf[%d] :%d result_buff[%d]:%d \n", i, j, recv, j, res);
                errors++;
                break;
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
TEST_F(AllGatherCommonTest, ut_hccl_impl_610_2rank_1server_all_gather_float)
{
    char file_name_t[] = "./st_hccl_impl_610_2rank_1server_all_gather_float.json";
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
    for (s32 i = 0; i < ndev; i++) {
        ret = hrtMalloc((void**)&sendbuf[i], count * sizeof(float));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(sendbuf[i], count * sizeof(float), 0, count * sizeof(float));
        ret = hrtMalloc((void**)&recvbuf[i], ndev * count * sizeof(float));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(recvbuf[i], ndev * count * sizeof(float), 0, count * sizeof(float));
        ret = hrtMalloc((void**)&result_buff[i], ndev * count * sizeof(float));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(result_buff[i], ndev * count * sizeof(float), 0, count * sizeof(float));
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
        for (u32 j = 0; j < ndev * count; j++) {
            result_buff[i][j] = 1;
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

    // 创建每个Dev的all_gather任务线程
    for (s32 i = 0; i < ndev; i++) {
        tid[i] = sal_thread_create("thread", impl_common_all_gather_task, (void*)&para_info[i]);
        EXPECT_NE(tid[i], (sal_thread_t)NULL);
    }

    for (s32 i = 0; i < ndev; i++) {
        while (sal_thread_is_running(tid[i])) {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
        // u32 devicePhyId;
        // ret = hrtGetDevicePhyIdByIndex((u32)i, devicePhyId);
        // EXPECT_EQ(ret, HCCL_SUCCESS);
        // ret = NetworkManager::GetInstance(devicePhyId).Destroy();
    }

    // 获取stream的操作的同步信号量
    for (s32 i = 0; i < ndev; i++) {
        for (s32 j = 0; j < ndev * count; j++) {
            float res = result_buff[i][j];
            float recv = outputbuf[i][j];

            if (abs(res - recv) > 1e-5) {
                HCCL_ERROR(" rank :%d recvbuf[%d] :%d result_buff[%d]:%d \n", i, j, recv, j, res);
                errors++;
                break;
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
TEST_F(AllGatherCommonTest, ut_hccl_impl_610_8rank_1server_all_gather_memsampler_char)
{
    char file_name_t[] = "./st_hccl_impl_610_8rank_1server_all_gather_memsampler_char.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << g_rank_table_610_8rank_1server << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }
    outfile.close();

    ResetInitState();
    setenv("HCCL_MEM_SAMPLER_PARAM", "0x40000000", 1);

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
        ret = hrtMalloc((void**)&recvbuf[i], ndev * count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(recvbuf[i], ndev * count * sizeof(s8), 0, count * sizeof(s8));
        ret = hrtMalloc((void**)&result_buff[i], ndev * count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(result_buff[i], ndev * count * sizeof(s8), 0, count * sizeof(s8));
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
        for (u32 j = 0; j < ndev * count; j++) {
            result_buff[i][j] = 1;
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

    // 创建每个Dev的all_gather任务线程
    for (s32 i = 0; i < ndev; i++) {
        tid[i] = sal_thread_create("thread", impl_common_all_gather_task, (void*)&para_info[i]);
        EXPECT_NE(tid[i], (sal_thread_t)NULL);
    }

    for (s32 i = 0; i < ndev; i++) {
        while (sal_thread_is_running(tid[i])) {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
        // u32 devicePhyId;
        // ret = hrtGetDevicePhyIdByIndex((u32)i, devicePhyId);
        // EXPECT_EQ(ret, HCCL_SUCCESS);
        // ret = NetworkManager::GetInstance(devicePhyId).Destroy();
    }

    // 获取stream的操作的同步信号量
    for (s32 i = 0; i < ndev; i++) {
        for (s32 j = 0; j < ndev * count; j++) {
            s8 res = result_buff[i][j];
            s8 recv = outputbuf[i][j];

            if (res != recv) {
                HCCL_ERROR(" rank :%d recvbuf[%d] :%d result_buff[%d]:%d \n", i, j, recv, j, res);
                errors++;
                break;
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
    unsetenv("HCCL_MEM_SAMPLER_PARAM");
    ResetInitState();
    remove(file_name_t);
    EXPECT_EQ(errors, 0);
}
#endif

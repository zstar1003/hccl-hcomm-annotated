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
// #include <mpi.h>

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
#include "rank_consistentcy_checker.h"
#include "workflow_pub.h"
#include "dltdt_function.h"
#include "param_check_pub.h"

#include "hccl/base.h"
#include <hccl/hccl_types.h>
#include "llt_hccl_stub_pub.h"
#include <sys/mman.h>
#include <fcntl.h>
#include "hcom_private.h"
#include "sal.h"
#include "config.h"
// #include "topoinfo_ranktableParser_pub.h"
#include "ranktable/v80_rank_table.h"
#include "dlra_function.h"
#include "network_manager_pub.h"
#include "hccl_communicator.h"
#include <iostream>
#include <fstream>
#include "llt_hccl_stub.h"
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

void* inter_reduce_task(void* parg)
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
    ret = NetworkManager::GetInstance(para_info->device_id).Destroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, para_info->identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    sal_memcpy(hcom_info.params.id.internal, sizeof(HcclRootInfo), &para_info->rootInfo, sizeof(HcclRootInfo));

    hcom_info.pComm.reset(new (std::nothrow) hccl::hcclComm());
    rtModel_t model = (void*)1;

    hrtSetDevice(para_info->device_id);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] task rt_set_device fails", hcom_info.params.rank);
    }

    CommConfig commConfig("hccl_world_group");
    ret = hcom_info.pComm->init(hcom_info.params, commConfig, hcom_info.rankTable);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] task reduce fails", para_info->device_id);
    }

    bool swapped;
    //-----------------Get Workspace Resource Start------------------//
    u64 stream_list_size = 0;
    ret = hcom_info.pComm->GetWorkspaceSubStreamNum(stream_list_size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HCCL_INFO("get stream_list_size[%d] success", stream_list_size);
    vector<HcclRtStream> streamList(stream_list_size);

    rtError_t rt_ret;
    //???stream
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = aclrtCreateStreamWithConfig(&streamList[i], 0, ACL_STREAM_PERSISTENT);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
        //??bind?model
        rt_ret = rtModelBindStream(model, streamList[i], RT_MODEL_WAIT_ACTIVE_STREAM);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    u32 rankSize = 0;
    ret = hcom_info.pComm->GetRankSize(rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u64 memSize = 0;
    ret = hcom_info.pComm->GetWorkspaceMemSize("HcomReduce", para_info->count, para_info->datatype, rankSize, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* memptr = nullptr;
    ret = hrtMalloc(&memptr, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hcom_info.pComm->SetWorkspaceResource("tag_inter_reduce_task_inter", memptr, memSize, streamList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    //-----------------Get Workspace Resource End------------------//
    rank_num_tmp = *(para_info->sync_addr) - 1;

    do {
        rank_num_tmp += 1;

        swapped = __sync_bool_compare_and_swap(para_info->sync_addr, rank_num_tmp, rank_num_tmp + 1);
    } while (!swapped);

    while (*(para_info->sync_addr) < para_info->ranks_local) {
        sched_yield();
    } // linux鎻愪緵涓€涓郴缁熻皟鐢ㄨ繍琛岃繘绋嬩富鍔ㄨ鍑烘墽琛屾潈

    __sync_synchronize(); // 鎻掑叆鍐呭瓨灞忛殰锛屽椤哄簭鎬ф湁瑕佹眰锛屼絾鏄湁娌℃湁浣跨敤lock鎸囦护鐨勬椂鍊�

    HCCL_DEBUG("all %d  ranks init ok ,then reduce", hcom_info.params.totalRanks);

    ret = hcom_info.pComm->Reduce(
        "tag_inter_reduce_task_inter", para_info->sendbuff, para_info->recvbuff, para_info->count, para_info->datatype,
        para_info->op, para_info->root, para_info->stream);

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("rank[%d] task reduce fails", hcom_info.params.rank);
    }

    rt_ret = RT_ERROR_NONE;
    rt_ret = aclrtSynchronizeStream(para_info->stream);

    if (rt_ret != RT_ERROR_NONE) {
        HCCL_ERROR("rank[%d] task allgather fails", hcom_info.params.rank);
    }
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = rtModelUnbindStream(model, streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }
    for (int i = 0; i < stream_list_size; i++) {
        rt_ret = aclrtDestroyStream(streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }
    hrtFree(memptr);
    RankConsistentcyChecker::GetInstance().ClearCheckInfo();
    return (NULL);
}

void* inter_all_gather_task(void* parg)
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
    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, para_info->identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    sal_memcpy(hcom_info.params.id.internal, sizeof(HcclRootInfo), &para_info->rootInfo, sizeof(HcclRootInfo));

    hcom_info.pComm.reset(new (std::nothrow) hccl::hcclComm(209715200, 209715200));
    rtModel_t model = (void*)1;

    CommConfig commConfig("hccl_world_group");
    ret = hcom_info.pComm->init(hcom_info.params, commConfig, hcom_info.rankTable);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] task all_gather fails", para_info->device_id);
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
    //-----------------Set Workspace Resource Start------------------//
    u64 stream_list_size = 0;
    ret = hcom_info.pComm->GetWorkspaceSubStreamNum(stream_list_size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    u32 rankSize = 0;
    ret = hcom_info.pComm->GetRankSize(rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HCCL_INFO("get stream_list_size[%d] and rank size[%d] success", stream_list_size, rankSize);
    vector<HcclRtStream> streamList(stream_list_size);

    rtError_t rt_ret;
    // 生成从stream
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = aclrtCreateStreamWithConfig(&streamList[i], 0, ACL_STREAM_PERSISTENT);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
        //??bind?model
        rt_ret = rtModelBindStream(model, streamList[i], RT_MODEL_WAIT_ACTIVE_STREAM);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    u64 memSize = 0;
    ret = hcom_info.pComm->GetWorkspaceMemSize(
        HCCL_KERNEL_OP_TYPE_ALLGATHER, para_info->count, para_info->datatype, rankSize, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* memptr = nullptr;
    ret = hrtMalloc(&memptr, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hcom_info.pComm->SetWorkspaceResource("tag_inter_all_gather_task_inter", memptr, memSize, streamList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    //-----------------Set Workspace Resource End------------------//
    HCCL_DEBUG("all %d  ranks init ok ,then allgather", hcom_info.params.totalRanks);
    string groupName = "group";
    ret = RankConsistentcyChecker::GetInstance().RecordOpPara(
        HcclCMDType::HCCL_CMD_ALLGATHER, "tag", para_info->count, para_info->datatype, 0, 0, groupName.c_str(), 0);
    ret = hcom_info.pComm->AllGather(
        "tag_inter_all_gather_task_inter", para_info->sendbuff, para_info->recvbuff, para_info->count,
        para_info->datatype, para_info->stream);

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("rank[%d] task allgather fails", hcom_info.params.rank);
    }

    rt_ret = RT_ERROR_NONE;
    rt_ret = aclrtSynchronizeStream(para_info->stream);

    if (rt_ret != RT_ERROR_NONE) {
        HCCL_ERROR("rank[%d] task allgather fails", hcom_info.params.rank);
    }
    RankConsistentcyChecker::GetInstance().ClearCheckInfo();
    return (NULL);
}

void* inter_reduce_scatter_task(void* parg)
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
        HCCL_ERROR("dev[%d] task reduce_scatter fails", para_info->device_id);
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
    SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
    //-----------------Set Workspace Resource Start------------------//
    u64 stream_list_size = 0;
    ret = hcom_info.pComm->GetWorkspaceSubStreamNum(stream_list_size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    u32 rankSize = 0;
    ret = hcom_info.pComm->GetRankSize(rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HCCL_INFO("get stream_list_size[%d] and rank size[%d] success", stream_list_size, rankSize);
    vector<HcclRtStream> streamList(stream_list_size);

    rtError_t rt_ret;
    // 生成从stream
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = aclrtCreateStreamWithConfig(&streamList[i], 0, ACL_STREAM_PERSISTENT);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
        //??bind?model
        rt_ret = rtModelBindStream(model, streamList[i], RT_MODEL_WAIT_ACTIVE_STREAM);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    u64 memSize = 0;
    ret = hcom_info.pComm->GetWorkspaceMemSize(
        HCCL_KERNEL_OP_TYPE_REDUCESCATTER, para_info->count, para_info->datatype, rankSize, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* memptr = nullptr;
    ret = hrtMalloc(&memptr, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hcom_info.pComm->SetWorkspaceResource("tag_inter_reduce_scatter_task_inter", memptr, memSize, streamList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    //-----------------Set Workspace Resource End------------------//
    ret = hcom_info.pComm->ReduceScatter(
        "tag_inter_reduce_scatter_task_inter", para_info->sendbuff, para_info->recvbuff, para_info->count,
        para_info->datatype, para_info->op, para_info->stream);

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] task reduce_scatter fails", para_info->device_id);
    }

    rt_ret = RT_ERROR_NONE;
    rt_ret = aclrtSynchronizeStream(para_info->stream);

    if (rt_ret != RT_ERROR_NONE) {
        HCCL_ERROR("rank[%d] task allgather fails", hcom_info.params.rank);
    }
    RankConsistentcyChecker::GetInstance().ClearCheckInfo();
    return (NULL);
}

class HcclInterTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HcclInterTest SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "HcclInterTest TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        DlTdtFunction::GetInstance().DlTdtFunctionInit();
        TsdOpen(1, 2);

        static s32 call_cnt = 0;
        string name = std::to_string(call_cnt++) + "_" + __PRETTY_FUNCTION__;
        ra_set_shm_name(name.c_str());

        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};

#define HCC_REDUCE_DATA_SIZE 10
#define HCC_REDUCE_DATA_SIZE_2M (1024 * 1024 * 2 + 2)

#define DEV_NUM_4 4
#define DEV_NUM_2 2

#if 1
#define HCCL_ALLGATHER_DATA_SIZE 10
#define HCC_ALLGATHER_SIZE_2M (1024 * 1024 * 2 + 3)
#if 1
TEST_F(HcclInterTest, ut_allgather_inter_char)
{
    // ranktable 的读取，直接使用进程
    nlohmann::json rank_table = rank_table_910_1server_2rank;

    char file_name_t[] = "./ut_allgather_inter_char.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    s32 errors = 0;

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

    s32 count = HCCL_ALLGATHER_DATA_SIZE;
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
        sal_memset(recvbuf[i], ndev * count * sizeof(s8), 0, ndev * count * sizeof(s8));

        result_buff[i] = (s8*)sal_malloc(ndev * count * sizeof(s8));
        sal_memset(result_buff[i], ndev * count * sizeof(s8), 0, ndev * count * sizeof(s8));
    }

    for (u32 j = 0; j < ndev; j++) {
        for (u32 i = 0; i < count; i++) {
            sendbuf[j][i] = 1;
        }
    }

    // resultbuf 赋值
    for (u32 i = 0; i < ndev; i++) {
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
        para_info[i].sendbuff = sendbuf[i];
        para_info[i].stream = stream[i];
        para_info[i].recvbuff = recvbuf[i];

        para_info[i].sync_addr = &sync_value;
        para_info[i].file_name = file_name_t;
    }

    // 创建每个Dev的allreduce任务线程
    for (s32 i = 0; i < ndev; ++i) {
        tid[i] = sal_thread_create("thread", inter_all_gather_task, (void*)&para_info[i]);

        EXPECT_NE(tid[i], (sal_thread_t)NULL);
    }

    for (s32 i = 0; i < ndev; ++i) {
        while (sal_thread_is_running(tid[i])) {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
    }

    // 获取stream的操作的同步信号量

    /*check result*/
    for (s32 j = 0; j < ndev; j++) {
        for (s32 i = 0; i < count * ndev; i++) {
            s8 res = result_buff[j][i];
            s8 recv = recvbuf[j][i];

            if (res != recv) {
                HCCL_ERROR("recvbuf[%d][%d]:%d \n", j, i, recv);
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

    for (s32 j = 0; j < ndev; j++) {
        hrtFree(sendbuf[j]);
        hrtFree(recvbuf[j]);
        sal_free(result_buff[j]);
        rt_ret = aclrtDestroyStream(stream[j]);

        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }
    remove(file_name_t);
    EXPECT_EQ(errors, 0);
}
#endif

#if 1
TEST_F(HcclInterTest, ut_allgather_inter_char_common_pid)
{
    // ranktable 的读取，直接使用进程
    nlohmann::json rank_table = rank_table_910_1server_2rank;

    rtSetCommonPidMode(true);
    char file_name_t[] = "./ut_allgather_inter_char.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    s32 errors = 0;

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

    s32 count = HCCL_ALLGATHER_DATA_SIZE;
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
        sal_memset(recvbuf[i], ndev * count * sizeof(s8), 0, ndev * count * sizeof(s8));

        result_buff[i] = (s8*)sal_malloc(ndev * count * sizeof(s8));
        sal_memset(result_buff[i], ndev * count * sizeof(s8), 0, ndev * count * sizeof(s8));
    }

    for (u32 j = 0; j < ndev; j++) {
        for (u32 i = 0; i < count; i++) {
            sendbuf[j][i] = 1;
        }
    }

    // resultbuf 赋值
    for (u32 i = 0; i < ndev; i++) {
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
        para_info[i].sendbuff = sendbuf[i];
        para_info[i].stream = stream[i];
        para_info[i].recvbuff = recvbuf[i];

        para_info[i].sync_addr = &sync_value;
        para_info[i].file_name = file_name_t;
    }

    // 创建每个Dev的allreduce任务线程
    for (s32 i = 0; i < ndev; ++i) {
        tid[i] = sal_thread_create("thread", inter_all_gather_task, (void*)&para_info[i]);

        EXPECT_NE(tid[i], (sal_thread_t)NULL);
    }

    for (s32 i = 0; i < ndev; ++i) {
        while (sal_thread_is_running(tid[i])) {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
    }

    // 获取stream的操作的同步信号量

    /*check result*/
    for (s32 j = 0; j < ndev; j++) {
        for (s32 i = 0; i < count * ndev; i++) {
            s8 res = result_buff[j][i];
            s8 recv = recvbuf[j][i];

            if (res != recv) {
                HCCL_ERROR("recvbuf[%d][%d]:%d \n", j, i, recv);
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

    for (s32 j = 0; j < ndev; j++) {
        hrtFree(sendbuf[j]);
        hrtFree(recvbuf[j]);
        sal_free(result_buff[j]);
        rt_ret = aclrtDestroyStream(stream[j]);

        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }
    rtSetCommonPidMode(false);
    remove(file_name_t);
    EXPECT_EQ(errors, 0);
}
#endif
#if 1
TEST_F(HcclInterTest, ut_allgather_inter_float_slice)
{
    // ranktable 的读取，直接使用进程
    nlohmann::json rank_table = rank_table_910_1server_2rank;

    char file_name_t[] = "./ut_allgather_inter_float_slice.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    s32 errors = 0;

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;

    s32* result_buff[DEV_NUM_2];
    s32* sendbuf[DEV_NUM_2];
    s32* recvbuf[DEV_NUM_2];

    s32 sync_value = 0;

    rtStream_t stream[DEV_NUM_2];
    sal_thread_t tid[DEV_NUM_2];
    para_t para_info[DEV_NUM_2];

    HcclDataType datatype = HCCL_DATA_TYPE_FP32;

    s32 count = HCC_ALLGATHER_SIZE_2M;
    s32 ndev = DEV_NUM_2;
    HcclRootInfo rootInfo;
    ret = hccl::hcclComm::GetUniqueId(&rootInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    /** 初始化输入输出缓存 */
    for (s32 i = 0; i < ndev; i++) {
        ret = hrtMalloc((void**)&sendbuf[i], count * sizeof(s32));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(sendbuf[i], count * sizeof(s32), 0, count * sizeof(s32));

        ret = hrtMalloc((void**)&recvbuf[i], ndev * count * sizeof(s32));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(recvbuf[i], ndev * count * sizeof(s32), 0, ndev * count * sizeof(s32));

        result_buff[i] = (s32*)sal_malloc(ndev * count * sizeof(s32));
        sal_memset(result_buff[i], ndev * count * sizeof(s32), 0, ndev * count * sizeof(s32));
    }

    for (u32 j = 0; j < ndev; j++) {
        for (u32 i = 0; i < count; i++) {
            sendbuf[j][i] = 1.0;
        }
    }

    // resultbuf 赋值
    for (u32 i = 0; i < ndev; i++) {
        for (u32 j = 0; j < ndev * count; j++) {
            result_buff[i][j] = 1.0;
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
        para_info[i].sendbuff = sendbuf[i];
        para_info[i].stream = stream[i];
        para_info[i].recvbuff = recvbuf[i];

        para_info[i].sync_addr = &sync_value;
        para_info[i].file_name = file_name_t;
    }

    // 创建每个Dev的allreduce任务线程
    for (s32 i = 0; i < ndev; ++i) {
        tid[i] = sal_thread_create("thread", inter_all_gather_task, (void*)&para_info[i]);

        EXPECT_NE(tid[i], (sal_thread_t)NULL);
    }

    for (s32 i = 0; i < ndev; ++i) {
        while (sal_thread_is_running(tid[i])) {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
    }

    // 获取stream的操作的同步信号量

    /*check result*/
    for (s32 j = 0; j < ndev; j++) {
        for (s32 i = 0; i < count * ndev; i++) {
            s32 res = result_buff[j][i];
            s32 recv = recvbuf[j][i];

            if (abs(res - recv) > 1e-6) {
                HCCL_ERROR("recvbuf[%d][%d]:%d \n", j, i, recv);
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

    for (s32 j = 0; j < ndev; j++) {
        hrtFree(sendbuf[j]);
        hrtFree(recvbuf[j]);
        sal_free(result_buff[j]);
        rt_ret = aclrtDestroyStream(stream[j]);

        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }
    remove(file_name_t);
    EXPECT_EQ(errors, 0);
}
#endif
#endif

#if 1
#define HCCL_REDUCESCATTER_DATA_SIZE 12
TEST_F(HcclInterTest, ut_reducescatter_inter_char)
{
    nlohmann::json rank_table = rank_table_910_1server_2rank;

    char file_name_t[] = "./ut_reducescatter_inter_char.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

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
    s32 count = HCCL_REDUCESCATTER_DATA_SIZE;
    s32 ndev = DEV_NUM_2;
    HcclRootInfo rootInfo;
    ret = hccl::hcclComm::GetUniqueId(&rootInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    /** 初始化输入输出缓存 */
    for (s32 i = 0; i < ndev; i++) {
        ret = hrtMalloc((void**)&sendbuf[i], ndev * count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(sendbuf[i], ndev * count * sizeof(s8), 0, ndev * count * sizeof(s8));

        ret = hrtMalloc((void**)&recvbuf[i], count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(recvbuf[i], count * sizeof(s8), 0, count * sizeof(s8));
        result_buff[i] = (s8*)sal_malloc(count * sizeof(s8));
        sal_memset(result_buff[i], count * sizeof(s8), 0, count * sizeof(s8));
    }

    // sendbuf 赋值
    for (u32 j = 0; j < ndev; j++) {
        for (u32 i = 0; i < ndev * count; i++) {
            sendbuf[j][i] = i % 12;
        }
    }

    // resultbuf 赋值
    for (s32 i = 0; i < ndev; i++) {
        for (u32 j = 0; j < count; j++) {
            result_buff[i][j] = ndev * j;
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
        para_info[i].sendbuff = sendbuf[i];
        para_info[i].stream = stream[i];
        para_info[i].recvbuff = recvbuf[i];
        para_info[i].op = op;

        para_info[i].sync_addr = &sync_value;
        para_info[i].file_name = file_name_t;
    }

    // 创建每个Dev的allreduce任务线程
    for (s32 i = 0; i < ndev; ++i) {
        tid[i] = sal_thread_create("thread", inter_reduce_scatter_task, (void*)&para_info[i]);
        EXPECT_NE(tid[i], (sal_thread_t)NULL);
    }

    for (s32 i = 0; i < ndev; ++i) {
        while (sal_thread_is_running(tid[i])) {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
    }

    // 获取stream的操作的同步信号量

    for (s32 i = 0; i < ndev; i++) {
        for (s32 j = 0; j < count; j++) {
            s8 res = result_buff[i][j];
            s8 recv = recvbuf[i][j];

            if (res != recv) {
                HCCL_ERROR(" recvbuf[%d] result_buff[%d] \n", recv, res);
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
    remove(file_name_t);
    EXPECT_EQ(errors, 0);
}
#endif

#define HCCL_REDUCESCATTER_DATA_SLICE 1024 * 4 + 2

TEST_F(HcclInterTest, ut_reducescatter_inter_float_slice)
{
    nlohmann::json rank_table = rank_table_910_1server_2rank;

    char file_name_t[] = "./ut_reducescatter_inter_float_slice.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    s32 rank, errors = 0;

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;

    s32* result_buff[DEV_NUM_2];
    s32* sendbuf[DEV_NUM_2];
    s32* recvbuf[DEV_NUM_2];

    s32 sync_value = 0;

    rtStream_t stream[DEV_NUM_2];
    sal_thread_t tid[DEV_NUM_2];
    para_t para_info[DEV_NUM_2];

    HcclDataType datatype = HCCL_DATA_TYPE_FP32;

    HcclReduceOp op = HCCL_REDUCE_MAX;
    s32 count = HCCL_REDUCESCATTER_DATA_SLICE;
    s32 ndev = DEV_NUM_2;
    HcclRootInfo rootInfo;
    ret = hccl::hcclComm::GetUniqueId(&rootInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    /** 初始化输入输出缓存 */
    for (s32 i = 0; i < ndev; i++) {
        ret = hrtMalloc((void**)&sendbuf[i], ndev * count * sizeof(s32));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(sendbuf[i], ndev * count * sizeof(s32), 0, ndev * count * sizeof(s32));

        ret = hrtMalloc((void**)&recvbuf[i], count * sizeof(s32));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(recvbuf[i], count * sizeof(s32), 0, count * sizeof(s32));
        result_buff[i] = (s32*)sal_malloc(count * sizeof(s32));
        sal_memset(result_buff[i], count * sizeof(s32), 0, count * sizeof(s32));
    }

    // sendbuf 赋值
    for (u32 j = 0; j < ndev; j++) {
        for (u32 i = 0; i < ndev * count; i++) {
            sendbuf[j][i] = 1.0;
        }
    }

    // resultbuf 赋值
    for (s32 i = 0; i < ndev; i++) {
        for (u32 j = 0; j < count; j++) {
            result_buff[i][j] = 1.0;
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
        para_info[i].sendbuff = sendbuf[i];
        para_info[i].stream = stream[i];
        para_info[i].recvbuff = recvbuf[i];
        para_info[i].op = op;

        para_info[i].sync_addr = &sync_value;
        para_info[i].file_name = file_name_t;
    }

    // 创建每个Dev的allreduce任务线程
    for (s32 i = 0; i < ndev; ++i) {
        tid[i] = sal_thread_create("thread", inter_reduce_scatter_task, (void*)&para_info[i]);
        EXPECT_NE(tid[i], (sal_thread_t)NULL);
    }

    for (s32 i = 0; i < ndev; ++i) {
        while (sal_thread_is_running(tid[i])) {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
    }

    // 获取stream的操作的同步信号量

    for (s32 i = 0; i < ndev; i++) {
        for (s32 j = 0; j < count; j++) {
            s32 res = result_buff[i][j];
            s32 recv = recvbuf[i][j];

            if (abs(res - recv) > 1e-6) {
                HCCL_ERROR(" recvbuf[%f] result_buff[%f] \n", recv, res);
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
    remove(file_name_t);
    EXPECT_EQ(errors, 0);
}

TEST_F(HcclInterTest, ut_reduce_inter_sum_char)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "4"},
           {"para_plane_nic_name", {"eth0", "eth1", "eth2", "eth3"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "4"},
              {"server_num", "1"},
              {"instance_count", "4"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.12"}}}}},

                   {{"rank_id", "1"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.14"}}}}},
                   {{"rank_id", "2"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.16"}}}}},

                   {{"rank_id", "3"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.18"}}}}},

               }},
              {"server_list",
               {
                   {{"server_id", "192.168.10.2"},
                    {"para_plane_info",
                     {{
                          {"eth1", "192.168.210.2"},
                      },
                      {
                          {"eth0", "192.168.200.2"},
                      }}}

                   },
               }}}}}};

    char file_name_t[] = "./ut_reduce_inter_sum_char.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    s32 rank, errors = 0;

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;

    s8* result_buff[DEV_NUM_4];
    s8* sendbuf[DEV_NUM_4];
    s8* recvbuf[DEV_NUM_4];

    s32 sync_value = 0;

    rtStream_t stream[DEV_NUM_4];
    sal_thread_t tid[DEV_NUM_4];
    para_t para_info[DEV_NUM_4];

    HcclDataType datatype = HCCL_DATA_TYPE_INT8;

    HcclReduceOp op = HCCL_REDUCE_SUM;
    s32 count = HCC_REDUCE_DATA_SIZE;
    s32 ndev = DEV_NUM_4;

    HcclRootInfo rootInfo;
    ret = hccl::hcclComm::GetUniqueId(&rootInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    /** 鍒濆鍖栬緭鍏ヨ緭鍑虹紦瀛� */
    for (s32 i = 0; i < ndev; i++) {
        ret = hrtMalloc((void**)&sendbuf[i], count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(sendbuf[i], count * sizeof(s8), 0, count * sizeof(s8));
        ret = hrtMalloc((void**)&recvbuf[i], count * sizeof(s8));
        sal_memset(recvbuf[i], count * sizeof(s8), 0, count * sizeof(s8));
        ret = hrtMalloc((void**)&result_buff[i], count * sizeof(s8));
        sal_memset(result_buff[i], count * sizeof(s8), 0, count * sizeof(s8));
    }

    // sendbuf 璧嬪€�
    for (u32 j = 0; j < ndev; j++) {
        for (u32 i = 0; i < count; i++) {
            sendbuf[j][i] = 1;
        }
    }

    // resultbuf 璧嬪€�

    for (u32 j = 0; j < count; j++) {
        result_buff[0][j] = ndev;
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
        para_info[i].sendbuff = sendbuf[i];
        para_info[i].stream = stream[i];
        para_info[i].recvbuff = recvbuf[i];
        para_info[i].op = op;
        para_info[i].root = 0;
        para_info[i].sync_addr = &sync_value;
        para_info[i].file_name = file_name_t;
    }

    // 鍒涘缓姣忎釜Dev鐨刟llreduce浠诲姟绾跨▼
    for (s32 i = 0; i < ndev; ++i) {
        tid[i] = sal_thread_create("thread", inter_reduce_task, (void*)&para_info[i]);
        EXPECT_NE(tid[i], (sal_thread_t)NULL);
    }

    for (s32 i = 0; i < ndev; ++i) {
        while (sal_thread_is_running(tid[i])) {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
    }

    // 鑾峰彇stream鐨勬搷浣滅殑鍚屾淇″彿閲�
    for (s32 j = 0; j < ndev; j++) {
        rt_ret = aclrtSynchronizeStream(stream[j]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    for (s32 i = 0; i < count; i++) {
        s32 res = result_buff[0][i];
        s32 recv = recvbuf[0][i];

        if (res != recv) {
            HCCL_ERROR(" root recvbuf[%d] result_buff[%d] \n", recv, res);
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
        ret = NetworkManager::GetInstance(i).Destroy();
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }
    remove(file_name_t);
    EXPECT_EQ(errors, 0);
}

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

#define private public
#define protected public
#include "hccl_communicator.h"
#include "hccl_impl.h"
#include "network_manager_pub.h"
#include "coll_alg_operator.h"
#undef protected
#undef private

#include "stream_pub.h"
#include "mem_host_pub.h"
#include "mem_device_pub.h"
#include "hccl_comm_pub.h"
#include "gradient_segment.h"
#include "sal.h"

#include "llt_hccl_stub_pub.h"
#include "externalinput.h"
#include "config.h"
#include "topoinfo_ranktableParser_pub.h"
#include "rank_consistentcy_checker.h"
#include <iostream>
#include <fstream>
#include "v80_rank_table.h"
#include "dlra_function.h"
#include <fcntl.h>
#include <unistd.h>
#include "llt_hccl_stub_profiling_plugin.h"
#include "task_profiling_pub.h"
#include "workflow_pub.h"
#include "dltdt_function.h"
#include "heartbeat.h"
#include "opexecounter_pub.h"
#include "param_check_pub.h"
#include "callback_thread_manager.h"
#include "dispatcher_graph_pub.h"
#include "dispatcher_pub.h"
#include "dispatcher_graph_pub.h"
#include "reduce_scatter_operator.h"
#include "hcom_private.h"
#include "graph_ctx_mgr.h"
using namespace std;
using namespace hccl;

class HcclCommTest910B : public testing::TestWithParam<bool> {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--HcclCommTest910B SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--HcclCommTest910B TearDown--\033[0m" << std::endl; }
    Some expensive resource shared by all tests.virtual void SetUp()
    {
        GlobalMockObject::verify();
        DlTdtFunction::GetInstance().DlTdtFunctionInit();
        DlRaFunction::GetInstance().DlRaFunctionInit();
        TsdOpen(1, 2);
        static s32 call_cnt = 0;
        string name = std::to_string(call_cnt++) + "_" + __PRETTY_FUNCTION__;
        ra_set_shm_name(name.c_str());
        MOCKER_CPP(&Heartbeat::Init).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&Heartbeat::RegisterRanks).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&Heartbeat::UnRegisterRanks).stubs().will(returnValue(HCCL_SUCCESS));
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
};

#define DEV_NUM_4 2
#define HCCL_ALLREDUCE_DATA_SIZE 2048
#define HCCL_ALLREDUCE_DATA_SLICE 1024 * 1024 * 2 + 10

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
    bool offline;
    u32 deviceNumPerServer;
} para_t;

void* inter_all_reduce_task_1(void* parg)
{
    HcclResult ret = HCCL_SUCCESS;
    para_t* para_info = (para_t*)parg;
    s32 rank_num_tmp;

    HcomInfo hcom_info;
    std::string ranktable_file = para_info->file_name;
    std::string rankTableM;
    std::string realFilePath;

    hrtSetDevice(para_info->device_id);
    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, para_info->identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sal_memset(hcom_info.params.id.internal, HCCL_ROOT_INFO_BYTES, 0, sizeof(hcom_info.params.id.internal));
    sal_memcpy(hcom_info.params.id.internal, sizeof(HcclRootInfo), &para_info->rootInfo, sizeof(HcclRootInfo));

    hcom_info.pComm.reset(new (std::nothrow) hccl::hcclComm());
    rtModel_t model = (void*)1;

    CommConfig commConfig("hccl_world_group");
    ret = hcom_info.pComm->init(hcom_info.params, commConfig, hcom_info.rankTable);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] task all_reduce fails", para_info->device_id);
    }
    u64 stream_list_size = 0;
    ret = hcom_info.pComm->GetWorkspaceSubStreamNum(stream_list_size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HCCL_INFO("get stream_list_size[%d] success", stream_list_size);
    vector<HcclRtStream> streamList(stream_list_size);
    void* memptr = nullptr;

    //-----------------Set Workspace Resource Start------------------//
    rtError_t rt_ret;
    // 生成从stream
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = aclrtCreateStreamWithConfig(&streamList[i], 0, ACL_STREAM_PERSISTENT);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
        // 从流bind到model
        rt_ret = rtModelBindStream(model, streamList[i], RT_MODEL_WAIT_ACTIVE_STREAM);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    u32 rankSize = 0;
    ret = hcom_info.pComm->GetRankSize(rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u64 memSize = 0;
    ret = hcom_info.pComm->GetWorkspaceMemSize(
        HCCL_KERNEL_OP_TYPE_ALLREDUCE, para_info->count, para_info->datatype, rankSize, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtMalloc(&memptr, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    string strTag = "allreduce_tag_magic4561637";

    ret = hcom_info.pComm->SetWorkspaceResource(strTag, memptr, memSize, streamList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    //-----------------Set Workspace Resource End------------------//

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

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] comm get map streamModel fail!", para_info->device_id);
    }
    ret = hcom_info.pComm->AllReduce(
        strTag, para_info->sendbuff, para_info->recvbuff, para_info->count, para_info->datatype, para_info->op,
        para_info->stream);

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] task HcclAllReduce fails", hcom_info.params.rank);
    }

    rt_ret = aclrtSynchronizeStream(para_info->stream);
    //--------------Resource destroy----------------//
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = rtModelUnbindStream(model, streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);

        rt_ret = aclrtDestroyStream(streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }
    hrtFree(memptr);

    if (rt_ret != RT_ERROR_NONE) {
        HCCL_ERROR("rank[%d] task allgather fails", hcom_info.params.rank);
    }

    return (NULL);
}

void* inter_reduce_scatter_task_1(void* parg)
{
    HcclResult ret = HCCL_SUCCESS;
    para_t* para_info = (para_t*)parg;
    s32 rank_num_tmp;

    HcomInfo hcom_info;
    std::string ranktable_file = para_info->file_name;
    std::string rankTableM;
    std::string realFilePath;
    hrtSetDevice(para_info->device_id);
    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, para_info->identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sal_memset(hcom_info.params.id.internal, HCCL_ROOT_INFO_BYTES, 0, sizeof(hcom_info.params.id.internal));
    sal_memcpy(hcom_info.params.id.internal, sizeof(HcclRootInfo), &para_info->rootInfo, sizeof(HcclRootInfo));

    hcom_info.pComm.reset(new (std::nothrow) hccl::hcclComm());
    rtModel_t model = (void*)1;

    CommConfig commConfig("hccl_world_group");
    ret = hcom_info.pComm->init(hcom_info.params, commConfig, hcom_info.rankTable);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] task reduce_scatter fails", para_info->device_id);
    }

    u64 stream_list_size = 0;
    ret = hcom_info.pComm->GetWorkspaceSubStreamNum(stream_list_size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HCCL_INFO("get stream_list_size[%d] success", stream_list_size);
    vector<HcclRtStream> streamList(stream_list_size);
    void* memptr = nullptr;

    //-----------------Set Workspace Resource Start------------------//
    rtError_t rt_ret;
    // 生成从stream
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = aclrtCreateStreamWithConfig(&streamList[i], 0, ACL_STREAM_PERSISTENT);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
        // 从流bind到model
        rt_ret = rtModelBindStream(model, streamList[i], RT_MODEL_WAIT_ACTIVE_STREAM);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    u32 rankSize = 0;
    ret = hcom_info.pComm->GetRankSize(rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u64 memSize = 0;
    ret = hcom_info.pComm->GetWorkspaceMemSize(
        HCCL_KERNEL_OP_TYPE_REDUCESCATTER, para_info->count, para_info->datatype, rankSize, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtMalloc(&memptr, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hcom_info.pComm->SetWorkspaceResource("tag_inter_reduce_scatter_task_1", memptr, memSize, streamList);
    EXPECT_EQ(ret, HCCL_SUCCESS);

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

    ret = hcom_info.pComm->ReduceScatter(
        "tag_inter_reduce_scatter_task_1", para_info->sendbuff, para_info->recvbuff, para_info->count,
        para_info->datatype, para_info->op, para_info->stream);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("rank[%d] task reduce_scatter fails", hcom_info.params.rank);
    }

    rt_ret = RT_ERROR_NONE;
    rt_ret = aclrtSynchronizeStream(para_info->stream);
    //--------------Resource destroy----------------//
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = rtModelUnbindStream(model, streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);

        rt_ret = aclrtDestroyStream(streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }
    hrtFree(memptr);
    if (rt_ret != RT_ERROR_NONE) {
        HCCL_ERROR("rank[%d] task allgather fails", hcom_info.params.rank);
    }
    return (NULL);
}

void* inter_reduce_scatter_task_1_ffts(void* parg)
{
    HcclResult ret = HCCL_SUCCESS;
    para_t* para_info = (para_t*)parg;
    s32 rank_num_tmp;

    HcomInfo hcom_info;
    std::string ranktable_file = para_info->file_name;
    std::string rankTableM;
    std::string realFilePath;
    hrtSetDevice(para_info->device_id);
    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, para_info->identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sal_memset(hcom_info.params.id.internal, HCCL_ROOT_INFO_BYTES, 0, sizeof(hcom_info.params.id.internal));
    sal_memcpy(hcom_info.params.id.internal, sizeof(HcclRootInfo), &para_info->rootInfo, sizeof(HcclRootInfo));

    hcom_info.pComm.reset(new (std::nothrow) hccl::hcclComm());
    rtModel_t model = (void*)1;

    hcom_info.params.deviceType = DevType::DEV_TYPE_910B;
    SetFftsSwitch(true);
    InitEnvVarParam();
    CommConfig commConfig("hccl_world_group");
    ret = hcom_info.pComm->init(hcom_info.params, commConfig, hcom_info.rankTable);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] task reduce_scatter fails", para_info->device_id);
    }

    u64 stream_list_size = 0;
    ret = hcom_info.pComm->GetWorkspaceSubStreamNum(stream_list_size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HCCL_INFO("get stream_list_size[%d] success", stream_list_size);
    vector<HcclRtStream> streamList(stream_list_size);
    void* memptr = nullptr;

    //-----------------Set Workspace Resource Start------------------//
    rtError_t rt_ret;
    // 生成从stream
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = aclrtCreateStreamWithConfig(&streamList[i], 0, ACL_STREAM_PERSISTENT);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
        // 从流bind到model
        rt_ret = rtModelBindStream(model, streamList[i], RT_MODEL_WAIT_ACTIVE_STREAM);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    u32 rankSize = 0;
    ret = hcom_info.pComm->GetRankSize(rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u64 memSize = 0;
    ret = hcom_info.pComm->GetWorkspaceMemSize(
        HCCL_KERNEL_OP_TYPE_REDUCESCATTER, para_info->count, para_info->datatype, rankSize, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtMalloc(&memptr, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hcom_info.pComm->SetWorkspaceResource("tag_inter_reduce_scatter_task_1_ffts", memptr, memSize, streamList);
    EXPECT_EQ(ret, HCCL_SUCCESS);

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

    ret = hcom_info.pComm->ReduceScatterOutPlace(
        "tag_inter_reduce_scatter_task_1_ffts", para_info->sendbuff, para_info->recvbuff, para_info->count,
        para_info->datatype, para_info->op, para_info->stream);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("rank[%d] task reduce_scatter fails", hcom_info.params.rank);
    }

    rt_ret = RT_ERROR_NONE;
    rt_ret = aclrtSynchronizeStream(para_info->stream);
    //--------------Resource destroy----------------//
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = rtModelUnbindStream(model, streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);

        rt_ret = aclrtDestroyStream(streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }
    hrtFree(memptr);
    if (rt_ret != RT_ERROR_NONE) {
        HCCL_ERROR("rank[%d] task allgather fails", hcom_info.params.rank);
    }
    SetFftsSwitch(false);
    InitEnvVarParam();
    return (NULL);
}

#if 1
void* inter_reduce_scatter_atomic_opbase_task_1(void* parg)
{
    HcclResult ret = HCCL_SUCCESS;
    para_t* para_info = (para_t*)parg;
    s32 rank_num_tmp;

    HcomInfo hcom_info;
    std::string ranktable_file = para_info->file_name;
    std::string rankTableM;
    std::string realFilePath;
    hrtSetDevice(para_info->device_id);
    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, para_info->identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sal_memset(hcom_info.params.id.internal, HCCL_ROOT_INFO_BYTES, 0, sizeof(hcom_info.params.id.internal));
    sal_memcpy(hcom_info.params.id.internal, sizeof(HcclRootInfo), &para_info->rootInfo, sizeof(HcclRootInfo));

    hcom_info.pComm.reset(new (std::nothrow) hccl::hcclComm(200 * 1024 * 1024, 200 * 1024 * 1024, HCCL_WORLD_GROUP));
    rtModel_t model = (void*)1;
    hcom_info.params.deviceType = DevType::DEV_TYPE_910B;
    CommConfig commConfig("hccl_world_group");
    ret = hcom_info.pComm->init(hcom_info.params, commConfig, hcom_info.rankTable);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] task reduce_scatter fails", para_info->device_id);
    }

    u64 stream_list_size = 0;
    ret = hcom_info.pComm->GetWorkspaceSubStreamNum(stream_list_size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HCCL_INFO("get stream_list_size[%d] success", stream_list_size);
    vector<HcclRtStream> streamList(stream_list_size);
    void* memptr = nullptr;

    //-----------------Set Workspace Resource Start------------------//
    rtError_t rt_ret;
    // 生成从stream
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = aclrtCreateStreamWithConfig(&streamList[i], 0, ACL_STREAM_PERSISTENT);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
        // 从流bind到model
        rt_ret = rtModelBindStream(model, streamList[i], RT_MODEL_WAIT_ACTIVE_STREAM);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    u32 rankSize = 0;
    ret = hcom_info.pComm->GetRankSize(rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u64 memSize = 0;
    ret = hcom_info.pComm->GetWorkspaceMemSize(
        HCCL_KERNEL_OP_TYPE_REDUCESCATTER, para_info->count, para_info->datatype, rankSize, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtMalloc(&memptr, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hcom_info.pComm->SetWorkspaceResource(
        "tag_inter_reduce_scatter_atomic_opbase_task_1", memptr, memSize, streamList);
    EXPECT_EQ(ret, HCCL_SUCCESS);

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

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    setenv("HCCL_DETERMINISTIC", "true", 1);
    ret = hcom_info.pComm->communicator_->ReduceScatterOutPlace(
        "tag_inter_reduce_scatter_atomic_opbase_task_1", para_info->sendbuff, para_info->recvbuff, para_info->count,
        para_info->datatype, para_info->op, para_info->stream);
    unsetenv("HCCL_DETERMINISTIC");
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("rank[%d] task reduce_scatter fails", hcom_info.params.rank);
    }

    rt_ret = RT_ERROR_NONE;
    rt_ret = aclrtSynchronizeStream(para_info->stream);
    //--------------Resource destroy----------------//
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = rtModelUnbindStream(model, streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);

        rt_ret = aclrtDestroyStream(streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }
    hrtFree(memptr);
    if (rt_ret != RT_ERROR_NONE) {
        HCCL_ERROR("rank[%d] task reduce_scatter fails", hcom_info.params.rank);
    }
    HCCL_ERROR("rank[%d] task done", hcom_info.params.rank);
    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
    return (NULL);
}
#endif

void* inter_all_gather_task_1(void* parg)
{
    HcclResult ret = HCCL_SUCCESS;
    para_t* para_info = (para_t*)parg;
    s32 rank_num_tmp;

    HcomInfo hcom_info;
    std::string ranktable_file = para_info->file_name;
    std::string rankTableM;
    std::string realFilePath;

    hrtSetDevice(para_info->device_id);
    ret = NetworkManager::GetInstance(para_info->device_id).Destroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
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
    u64 stream_list_size = 0;
    ret = hcom_info.pComm->GetWorkspaceSubStreamNum(stream_list_size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HCCL_INFO("get stream_list_size[%d] success", stream_list_size);
    vector<HcclRtStream> streamList(stream_list_size);
    void* memptr = nullptr;

    //-----------------Set Workspace Resource Start------------------//
    rtError_t rt_ret;
    // 生成从stream
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = aclrtCreateStreamWithConfig(&streamList[i], 0, ACL_STREAM_PERSISTENT);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
        // 从流bind到model
        rt_ret = rtModelBindStream(model, streamList[i], RT_MODEL_WAIT_ACTIVE_STREAM);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    u32 rankSize = 0;
    ret = hcom_info.pComm->GetRankSize(rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u64 memSize = 0;
    ret = hcom_info.pComm->GetWorkspaceMemSize(
        HCCL_KERNEL_OP_TYPE_ALLREDUCE, para_info->count, para_info->datatype, rankSize, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtMalloc(&memptr, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    string strTag = "allgather_tag_magic45456";

    ret = hcom_info.pComm->SetWorkspaceResource(strTag, memptr, memSize, streamList);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    bool swapped;

    rank_num_tmp = *(para_info->sync_addr) - 1;

    do {
        rank_num_tmp += 1;

        swapped = __sync_bool_compare_and_swap(para_info->sync_addr, rank_num_tmp, rank_num_tmp + 1);
    } while (!swapped);

    while (*(para_info->sync_addr) < para_info->ranks_local) {
        sched_yield();
    } // linux鎻愪緵涓€涓郴缁熻皟鐢ㄨ繍琛岃繘绋嬩富鍔ㄨ鍑烘墽琛屾潈

    __sync_synchronize(); // 鎻掑叆鍐呭瓨灞忛殰锛屽椤哄簭鎬ф湁瑕佹眰锛屼絾鏄湁娌℃湁浣跨敤lock鎸囦护鐨勬椂鍊�

    HCCL_DEBUG("all %d  ranks init ok ,then allgather", hcom_info.params.totalRanks);
    ret = hcom_info.pComm->AllGather(
        strTag, para_info->sendbuff, para_info->recvbuff, para_info->count, para_info->datatype, para_info->stream);

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("rank[%d] task allgather fails", hcom_info.params.rank);
    }

    rt_ret = RT_ERROR_NONE;
    rt_ret = aclrtSynchronizeStream(para_info->stream);
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = rtModelUnbindStream(model, streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);

        rt_ret = aclrtDestroyStream(streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }
    hrtFree(memptr);
    if (rt_ret != RT_ERROR_NONE) {
        HCCL_ERROR("rank[%d] task allgather fails", hcom_info.params.rank);
    }

    return (NULL);
}

void* inter_all_gather_task_1_ffts(void* parg)
{
    HcclResult ret = HCCL_SUCCESS;
    para_t* para_info = (para_t*)parg;
    s32 rank_num_tmp;

    HcomInfo hcom_info;
    std::string ranktable_file = para_info->file_name;
    std::string rankTableM;
    std::string realFilePath;

    hrtSetDevice(para_info->device_id);
    ret = NetworkManager::GetInstance(para_info->device_id).Destroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, para_info->identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    sal_memcpy(hcom_info.params.id.internal, sizeof(HcclRootInfo), &para_info->rootInfo, sizeof(HcclRootInfo));

    hcom_info.pComm.reset(new (std::nothrow) hccl::hcclComm(209715200, 209715200));
    rtModel_t model = (void*)1;

    hcom_info.params.deviceType = DevType::DEV_TYPE_910B;
    SetFftsSwitch(true);
    InitEnvVarParam();

    CommConfig commConfig("hccl_world_group");
    ret = hcom_info.pComm->init(hcom_info.params, commConfig, hcom_info.rankTable);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] task all_gather fails", para_info->device_id);
    }
    u64 stream_list_size = 0;
    ret = hcom_info.pComm->GetWorkspaceSubStreamNum(stream_list_size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HCCL_INFO("get stream_list_size[%d] success", stream_list_size);
    vector<HcclRtStream> streamList(stream_list_size);
    void* memptr = nullptr;

    //-----------------Set Workspace Resource Start------------------//
    rtError_t rt_ret;
    // 生成从stream
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = aclrtCreateStreamWithConfig(&streamList[i], 0, ACL_STREAM_PERSISTENT);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
        // 从流bind到model
        rt_ret = rtModelBindStream(model, streamList[i], RT_MODEL_WAIT_ACTIVE_STREAM);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    u32 rankSize = 0;
    ret = hcom_info.pComm->GetRankSize(rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u64 memSize = 0;
    ret = hcom_info.pComm->GetWorkspaceMemSize(
        HCCL_KERNEL_OP_TYPE_ALLREDUCE, para_info->count, para_info->datatype, rankSize, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtMalloc(&memptr, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    string strTag = "allgather_tag_magic45456_ffts";

    ret = hcom_info.pComm->SetWorkspaceResource(strTag, memptr, memSize, streamList);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    bool swapped;

    rank_num_tmp = *(para_info->sync_addr) - 1;

    do {
        rank_num_tmp += 1;

        swapped = __sync_bool_compare_and_swap(para_info->sync_addr, rank_num_tmp, rank_num_tmp + 1);
    } while (!swapped);

    while (*(para_info->sync_addr) < para_info->ranks_local) {
        sched_yield();
    } // linux鎻愪緵涓€涓郴缁熻皟鐢ㄨ繍琛岃繘绋嬩富鍔ㄨ鍑烘墽琛屾潈

    __sync_synchronize(); // 鎻掑叆鍐呭瓨灞忛殰锛屽椤哄簭鎬ф湁瑕佹眰锛屼絾鏄湁娌℃湁浣跨敤lock鎸囦护鐨勬椂鍊�

    HCCL_DEBUG("all %d  ranks init ok ,then allgather", hcom_info.params.totalRanks);
    ret = hcom_info.pComm->AllGatherOutPlace(
        strTag, para_info->sendbuff, para_info->recvbuff, para_info->count, para_info->datatype, para_info->stream);

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("rank[%d] task allgather fails", hcom_info.params.rank);
    }

    rt_ret = RT_ERROR_NONE;
    rt_ret = aclrtSynchronizeStream(para_info->stream);
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = rtModelUnbindStream(model, streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);

        rt_ret = aclrtDestroyStream(streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }
    hrtFree(memptr);
    if (rt_ret != RT_ERROR_NONE) {
        HCCL_ERROR("rank[%d] task allgather fails", hcom_info.params.rank);
    }

    SetFftsSwitch(false);
    InitEnvVarParam();

    return (NULL);
}

void* inter_all_gather_outplace_ffts(void* parg)
{
    HcclResult ret = HCCL_SUCCESS;
    para_t* para_info = (para_t*)parg;
    s32 rank_num_tmp;

    HcomInfo hcom_info;
    std::string ranktable_file = para_info->file_name;
    std::string rankTableM;
    std::string realFilePath;

    hrtSetDevice(para_info->device_id);
    ret = NetworkManager::GetInstance(para_info->device_id).Destroy();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, para_info->identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    sal_memcpy(hcom_info.params.id.internal, sizeof(HcclRootInfo), &para_info->rootInfo, sizeof(HcclRootInfo));

    hcom_info.pComm.reset(new (std::nothrow) hccl::hcclComm(1024 * 1024 * 10, 1024 * 1024 * 10, HCCL_WORLD_GROUP));
    rtModel_t model = (void*)1;

    hcom_info.params.deviceType = DevType::DEV_TYPE_910B;

    CommConfig commConfig("hccl_world_group");
    ret = hcom_info.pComm->init(hcom_info.params, commConfig, hcom_info.rankTable);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] task all_gather fails", para_info->device_id);
    }
    u64 stream_list_size = 0;
    ret = hcom_info.pComm->GetWorkspaceSubStreamNum(stream_list_size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HCCL_INFO("get stream_list_size[%d] success", stream_list_size);
    vector<HcclRtStream> streamList(stream_list_size);
    void* memptr = nullptr;

    //-----------------Set Workspace Resource Start------------------//
    rtError_t rt_ret;
    // 生成从stream
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = aclrtCreateStreamWithConfig(&streamList[i], 0, ACL_STREAM_PERSISTENT);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
        // 从流bind到model
        rt_ret = rtModelBindStream(model, streamList[i], RT_MODEL_WAIT_ACTIVE_STREAM);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    u32 rankSize = 0;
    ret = hcom_info.pComm->GetRankSize(rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u64 memSize = 0;
    ret = hcom_info.pComm->GetWorkspaceMemSize(
        HCCL_KERNEL_OP_TYPE_ALLGATHER, para_info->count, para_info->datatype, rankSize, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtMalloc(&memptr, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    string strTag = "allgather_tag_magic9999998";

    ret = hcom_info.pComm->SetWorkspaceResource(strTag, memptr, memSize, streamList);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    bool swapped;

    rank_num_tmp = *(para_info->sync_addr) - 1;

    do {
        rank_num_tmp += 1;

        swapped = __sync_bool_compare_and_swap(para_info->sync_addr, rank_num_tmp, rank_num_tmp + 1);
    } while (!swapped);

    while (*(para_info->sync_addr) < para_info->ranks_local) {
        sched_yield();
    }

    __sync_synchronize();

    HCCL_DEBUG("all %d  ranks init ok ,then allgather", hcom_info.params.totalRanks);

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    HcomCollOpInfo opInfo;
    opInfo.inputAddr = para_info->sendbuff;
    opInfo.outputAddr = para_info->recvbuff;
    opInfo.count = para_info->count;
    opInfo.dataType = para_info->datatype;
    ret = hcom_info.pComm->communicator_->AllGatherOutPlace(
        strTag, para_info->sendbuff, para_info->recvbuff, para_info->count, para_info->datatype, para_info->stream);

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("rank[%d] task allgather fails", hcom_info.params.rank);
    }

    rt_ret = RT_ERROR_NONE;
    rt_ret = aclrtSynchronizeStream(para_info->stream);
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = rtModelUnbindStream(model, streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);

        rt_ret = aclrtDestroyStream(streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }
    hrtFree(memptr);
    if (rt_ret != RT_ERROR_NONE) {
        HCCL_ERROR("rank[%d] task allgather fails", hcom_info.params.rank);
    }

    return (NULL);
}

void* inter_all_reduce_outplace_task_1(void* parg)
{
    HcclResult ret = HCCL_SUCCESS;
    para_t* para_info = (para_t*)parg;
    s32 rank_num_tmp;

    HcomInfo hcom_info;
    std::string ranktable_file = para_info->file_name;
    std::string rankTableM;
    std::string realFilePath;

    hrtSetDevice(para_info->device_id);
    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, para_info->identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sal_memset(hcom_info.params.id.internal, HCCL_ROOT_INFO_BYTES, 0, sizeof(hcom_info.params.id.internal));
    sal_memcpy(hcom_info.params.id.internal, sizeof(HcclRootInfo), &para_info->rootInfo, sizeof(HcclRootInfo));

    MOCKER_CPP(&GraphMgr::GraphCtxMgr::ConstructFftsNotifyRecordRemoteCtx)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollAlgOperator::Is2U2PInfer).stubs().with(mockcpp::any()).will(returnValue(true));
    hcom_info.pComm.reset(new (std::nothrow)
                              hccl::hcclComm(HCCL_ALLREDUCE_DATA_SLICE, HCCL_ALLREDUCE_DATA_SLICE, HCCL_WORLD_GROUP));
    rtModel_t model = (void*)1;

    CommConfig commConfig("hccl_world_group");
    ret = hcom_info.pComm->init(hcom_info.params, commConfig, hcom_info.rankTable);
    HcclCommunicator* impl = dynamic_cast<HcclCommunicator*>(hcom_info.pComm->communicator_.get());
    impl->implAlg_->pimpl_->topoType_ = TopoType::TOPO_TYPE_NP_SINGLE_RING;
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] task all_reduce fails", para_info->device_id);
    }
    u64 stream_list_size = 0;
    ret = hcom_info.pComm->GetWorkspaceSubStreamNum(stream_list_size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HCCL_INFO("get stream_list_size[%d] success", stream_list_size);
    vector<HcclRtStream> streamList(stream_list_size);
    void* memptr = nullptr;

    //-----------------Set Workspace Resource Start------------------//
    rtError_t rt_ret;
    // 生成从stream
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = aclrtCreateStreamWithConfig(&streamList[i], 0, ACL_STREAM_PERSISTENT);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
        // 从流bind到model
        rt_ret = rtModelBindStream(model, streamList[i], RT_MODEL_WAIT_ACTIVE_STREAM);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    u32 rankSize = 0;
    ret = hcom_info.pComm->GetRankSize(rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u64 memSize = 0;
    ret = hcom_info.pComm->GetWorkspaceMemSize(
        HCCL_KERNEL_OP_TYPE_ALLREDUCE, para_info->count, para_info->datatype, rankSize, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtMalloc(&memptr, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    string strTag = "allreduce_tag_magic9999999";

    ret = hcom_info.pComm->SetWorkspaceResource(strTag, memptr, memSize, streamList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    //-----------------Set Workspace Resource End------------------//

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

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] comm get map streamModel fail!", para_info->device_id);
    }

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    HcomCollOpInfo opInfo;
    opInfo.inputAddr = para_info->sendbuff;
    opInfo.outputAddr = para_info->recvbuff;
    opInfo.count = para_info->count;
    opInfo.dataType = para_info->datatype;
    ret = impl->AllReduceOutPlace(
        strTag, para_info->sendbuff, para_info->recvbuff, para_info->count, para_info->datatype, para_info->op,
        para_info->stream, SyncMode::DEFAULT_TIMEWAITSYNCMODE);

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] task HcclAllReduce fails", hcom_info.params.rank);
    }

    rt_ret = aclrtSynchronizeStream(para_info->stream);
    //--------------Resource destroy----------------//
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = rtModelUnbindStream(model, streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);

        rt_ret = aclrtDestroyStream(streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }
    hrtFree(memptr);

    if (rt_ret != RT_ERROR_NONE) {
        HCCL_ERROR("rank[%d] task allgather fails", hcom_info.params.rank);
    }

    return (NULL);
}

#define DEV_NUM_4 4

#define HCCL_ALLGATHER_DATA_SIZE 10
#define HCC_ALLGATHER_SIZE_2M (1024 * 1024 * 2 + 3)

INSTANTIATE_TEST_CASE_P(FFTSSwitch, HcclCommTest910B, testing::Bool());
#if 0
TEST_P(HcclCommTest910B, ut_allgather_inter_char)
{
    // ranktable 鐨勮鍙栵紝鐩存帴浣跨敤杩涚▼
    nlohmann::json rank_table = rank_table_910_1server_4rank;
    char file_name_t[] = "./st_allgather_inter_char.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }
    else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    s32 errors = 0;

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

    s32 count = HCCL_ALLGATHER_DATA_SIZE;
    s32 ndev = DEV_NUM_4;
    HcclRootInfo rootInfo;
    ret = hccl::hcclComm::GetUniqueId(&rootInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    /** 鍒濆鍖栬緭鍏ヨ緭鍑虹紦瀛� */
    for (s32 i = 0; i < ndev; i++ )
    {
        ret = hrtMalloc((void **)&sendbuf[i], count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(sendbuf[i], count * sizeof(s8), 0, count * sizeof(s8));

        ret = hrtMalloc((void **)&recvbuf[i], ndev * count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(recvbuf[i], ndev * count * sizeof(s8), 0, ndev  * count * sizeof(s8));

        result_buff[i] = (s8*)sal_malloc(ndev * count * sizeof(s8));
        sal_memset(result_buff[i], ndev  * count * sizeof(s8), 0, ndev * count * sizeof(s8));
    }

    for (u32 j = 0; j < ndev; j++)
    {
        for (u32 i = 0; i < count; i++)
        {
            sendbuf[j][i] = 1;
        }
    }

    //resultbuf 璧嬪€�
    for (u32 i = 0; i < ndev; i++)
    {
        for (u32 j = 0; j < ndev * count; j++)
        {
            result_buff[i][j] = 1 ;
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

        para_info[i].sync_addr = &sync_value;
        para_info[i].file_name = file_name_t;
    }

    bool fftsSwitch = GetParam();

    // 鍒涘缓姣忎釜Dev鐨刟llreduce浠诲姟绾跨▼
    MOCKER_CPP(&HcclCommunicator::ExecOp)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));
    for (s32 i = 0; i < ndev; ++i)
    {
        if (fftsSwitch) {
            tid[i] = sal_thread_create("thread0", inter_all_gather_task_1_ffts, (void*)&para_info[i]);
        } else {
            tid[i] = sal_thread_create("thread", inter_all_gather_task_1, (void*)&para_info[i]);
        }


        EXPECT_NE(tid[i], (sal_thread_t )NULL);
    }

    for (s32 i = 0; i < ndev; ++i)
    {
        while ( sal_thread_is_running(tid[i]))
        {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
    }

    //鑾峰彇stream鐨勬搷浣滅殑鍚屾淇″彿閲�


    /*check result*/
    if (!fftsSwitch) {
        for (s32 j = 0; j < ndev; j++)
        {
            for (s32 i = 0; i < count * ndev; i++)
            {
                s8 res = result_buff[j][i];
                s8 recv = recvbuf[j][i];

                if (res != recv)
                {
                    HCCL_ERROR("recvbuf[%d][%d]:%d \n", j, i, recv);
                    errors++;
                // break;
                }
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

    for (s32 j = 0; j < ndev; j++)
    {
        hrtFree(sendbuf[j]);
        hrtFree(recvbuf[j]);
        sal_free(result_buff[j]);
        rt_ret = aclrtDestroyStream(stream[j]);

        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }
    remove(file_name_t);
}
#endif

void* inter_all_gather_outplace_task_1(void* parg)
{
    HcclResult ret = HCCL_SUCCESS;
    para_t* para_info = (para_t*)parg;
    s32 rank_num_tmp;

    HcomInfo hcom_info;
    std::string ranktable_file = para_info->file_name;
    std::string rankTableM;
    std::string realFilePath;

    hrtSetDevice(para_info->device_id);
    set_chip_type_stub(para_info->device_id, static_cast<s32>(DevType::DEV_TYPE_910B));
    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    InitExternalInput();

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, para_info->identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sal_memset(hcom_info.params.id.internal, HCCL_ROOT_INFO_BYTES, 0, sizeof(hcom_info.params.id.internal));
    sal_memcpy(hcom_info.params.id.internal, sizeof(HcclRootInfo), &para_info->rootInfo, sizeof(HcclRootInfo));

    hcom_info.pComm.reset(new (std::nothrow) hccl::hcclComm(1024 * 1024 * 10, 1024 * 1024 * 10, HCCL_WORLD_GROUP));
    rtModel_t model = (void*)1;

    hcom_info.params.deviceType = DevType::DEV_TYPE_910B;

    CommConfig commConfig("hccl_world_group");
    ret = hcom_info.pComm->init(hcom_info.params, commConfig, hcom_info.rankTable);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] task all_gather fails", para_info->device_id);
    }
    u64 stream_list_size = 0;
    ret = hcom_info.pComm->GetWorkspaceSubStreamNum(stream_list_size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HCCL_INFO("get stream_list_size[%d] success", stream_list_size);
    vector<HcclRtStream> streamList(stream_list_size);
    void* memptr = nullptr;

    //-----------------Set Workspace Resource Start------------------//
    rtError_t rt_ret;
    // 生成从stream
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = aclrtCreateStreamWithConfig(&streamList[i], 0, ACL_STREAM_PERSISTENT);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
        // 从流bind到model
        rt_ret = rtModelBindStream(model, streamList[i], RT_MODEL_WAIT_ACTIVE_STREAM);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    u32 rankSize = 0;
    ret = hcom_info.pComm->GetRankSize(rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u64 memSize = 0;
    ret = hcom_info.pComm->GetWorkspaceMemSize(
        HCCL_KERNEL_OP_TYPE_ALLGATHER, para_info->count, para_info->datatype, rankSize, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtMalloc(&memptr, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    string strTag = "allgather_tag_magic9999998";

    ret = hcom_info.pComm->SetWorkspaceResource(strTag, memptr, memSize, streamList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    //-----------------Set Workspace Resource End------------------//

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

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] comm get map streamModel fail!", para_info->device_id);
    }

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    HcomCollOpInfo opInfo;
    opInfo.inputAddr = para_info->sendbuff;
    opInfo.outputAddr = para_info->recvbuff;
    opInfo.count = para_info->count;
    opInfo.dataType = para_info->datatype;
    ret = hcom_info.pComm->communicator_->AllGatherOutPlace(
        strTag, para_info->sendbuff, para_info->recvbuff, para_info->count, para_info->datatype, para_info->stream);

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] task HcclAllGather fails", hcom_info.params.rank);
    }

    rt_ret = aclrtSynchronizeStream(para_info->stream);
    //--------------Resource destroy----------------//
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = rtModelUnbindStream(model, streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);

        rt_ret = aclrtDestroyStream(streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }
    hrtFree(memptr);

    if (rt_ret != RT_ERROR_NONE) {
        HCCL_ERROR("rank[%d] task allgather fails", hcom_info.params.rank);
    }

    return (nullptr);
}

TEST_P(HcclCommTest910B, ut_allgather_outplace_4p_mesh)
{
    RankConsistentcyChecker::GetInstance().ClearCheckInfo();
    nlohmann::json rank_table = rank_table_910_1server_4rank;
    char file_name_t[] = "./ut_allgather_4p_mesh.json";
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
    s8* inputbuf[DEV_NUM_4];
    s8* outputbuf[DEV_NUM_4];

    s32 sync_value = 0;

    rtStream_t stream[DEV_NUM_4];
    sal_thread_t tid[DEV_NUM_4];
    para_t para_info[DEV_NUM_4];

    HcclDataType datatype = HCCL_DATA_TYPE_INT8;
    s32 count = HCCL_ALLGATHER_DATA_SIZE;
    HCCL_ERROR("count : [%d]", count);
    s32 ndev = DEV_NUM_4;
    HcclRootInfo rootInfo;
    set_board_id(0x0000);
    for (s32 i = 0; i < ndev; i++) {
        set_chip_type_stub(i, static_cast<s32>(DevType::DEV_TYPE_910B));
    }
    ret = hccl::hcclComm::GetUniqueId(&rootInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    /** 初始化输入输出缓存 */
    for (s32 i = 0; i < ndev; i++) {
        ret = hrtMalloc((void**)&(sendbuf[i]), count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(sendbuf[i], count * sizeof(s8), 0, count * sizeof(s8));
        ret = hrtMalloc((void**)&(recvbuf[i]), count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(recvbuf[i], count * sizeof(s8) * ndev, 0, count * sizeof(s8) * ndev);
        ret = hrtMalloc((void**)&(result_buff[i]), count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(result_buff[i], count * sizeof(s8) * ndev, 0, count * sizeof(s8) * ndev);
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
        for (u32 j = 0; j < count * ndev; j++) {
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

        para_info[i].sync_addr = &sync_value;
        para_info[i].file_name = file_name_t;
        para_info[i].offline = false;
    }

    // 创建每个Dev的allgather任务线程
    for (s32 i = 0; i < ndev; i++) {
        tid[i] = sal_thread_create("thread", inter_all_gather_outplace_task_1, (void*)&para_info[i]);
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
            HCCL_ERROR("i, j, result_buff[i][j], %d, %d, %d", i, j, result_buff[i][j]);
            s8 recv = outputbuf[i][j];
            HCCL_ERROR("i, j, outputbuf[i][j], %d, %d, %d", i, j, outputbuf[i][j]);
            if (res != recv) {
                HCCL_ERROR(" recvbuf[%d] result_buff[%d] \n", recv, res);
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
    for (s32 i = 0; i < ndev; i++) {
        set_chip_type_stub(i, static_cast<s32>(DevType::DEV_TYPE_910));
    }
    set_board_id(0);
    remove(file_name_t);
}

TEST_F(HcclCommTest910B, ut_reducescatter_4p_mesh_atomic_opbase)
{
    setenv("HCCL_DETERMINISTIC", "true", 1);
    ResetInitState();
    InitExternalInput();
    RankConsistentcyChecker::GetInstance().ClearCheckInfo();
    nlohmann::json rank_table = rank_table_910_1server_4rank;
    char file_name_t[] = "./ut_allreduce_4p_ring.json";
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

    float* result_buff[DEV_NUM_4];
    float* sendbuf[DEV_NUM_4];
    float* recvbuf[DEV_NUM_4];

    s32 sync_value = 0;

    rtStream_t stream[DEV_NUM_4];
    sal_thread_t tid[DEV_NUM_4];
    para_t para_info[DEV_NUM_4];

    HcclDataType datatype = HCCL_DATA_TYPE_FP32;

    HcclReduceOp op = HCCL_REDUCE_SUM;
    s32 count = 10;
    s32 ndev = DEV_NUM_4;
    set_board_id(0x0000);
    for (s32 i = 0; i < ndev; i++) {
        set_chip_type_stub(i, static_cast<s32>(DevType::DEV_TYPE_910B));
    }

    HcclRootInfo rootInfo;
    ret = hccl::hcclComm::GetUniqueId(&rootInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    /** 初始化输入输出缓存 */
    for (s32 i = 0; i < ndev; i++) {
        ret = hrtMalloc((void**)&sendbuf[i], ndev * count * sizeof(float));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(sendbuf[i], ndev * count * sizeof(float), 0, ndev * count * sizeof(float));

        ret = hrtMalloc((void**)&recvbuf[i], count * sizeof(float));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(recvbuf[i], count * sizeof(float), 0, count * sizeof(float));
        result_buff[i] = (float*)sal_malloc(count * sizeof(float));
        sal_memset(result_buff[i], count * sizeof(float), 0, count * sizeof(float));
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
            result_buff[i][j] = 4.0;
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
        para_info[i].offline = false;
    }

    for (s32 i = 0; i < ndev; ++i) {
        tid[i] = sal_thread_create("thread", inter_reduce_scatter_atomic_opbase_task_1, (void*)&para_info[i]);
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
            float res = result_buff[i][j];
            float recv = recvbuf[i][j];

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
    for (s32 i = 0; i < ndev; i++) {
        set_chip_type_stub(i, static_cast<s32>(DevType::DEV_TYPE_910));
    }

    set_board_id(0);
    unsetenv("HCCL_DETERMINISTIC");
    ResetInitState();
    InitExternalInput();
    remove(file_name_t);
}

#if 1
TEST_F(HcclCommTest910B, ut_reducescatter_4p_mesh)
{
    setenv("HCCL_DETERMINISTIC", "true", 1);
    ResetInitState();
    InitExternalInput();
    RankConsistentcyChecker::GetInstance().ClearCheckInfo();
    nlohmann::json rank_table = rank_table_910_1server_4rank;
    char file_name_t[] = "./ut_allreduce_4p_ring.json";
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

    float* result_buff[DEV_NUM_4];
    float* sendbuf[DEV_NUM_4];
    float* recvbuf[DEV_NUM_4];

    s32 sync_value = 0;

    rtStream_t stream[DEV_NUM_4];
    sal_thread_t tid[DEV_NUM_4];
    para_t para_info[DEV_NUM_4];

    HcclDataType datatype = HCCL_DATA_TYPE_FP32;

    HcclReduceOp op = HCCL_REDUCE_SUM;
    //    s32 count = 512;
    s32 count = 10;
    s32 ndev = DEV_NUM_4;
    set_board_id(0x0000);
    for (s32 i = 0; i < ndev; i++) {
        set_chip_type_stub(i, static_cast<s32>(DevType::DEV_TYPE_910B));
    }

    HcclRootInfo rootInfo;
    ret = hccl::hcclComm::GetUniqueId(&rootInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    /** 初始化输入输出缓存 */
    for (s32 i = 0; i < ndev; i++) {
        ret = hrtMalloc((void**)&sendbuf[i], ndev * count * sizeof(float));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(sendbuf[i], ndev * count * sizeof(float), 0, ndev * count * sizeof(float));

        ret = hrtMalloc((void**)&recvbuf[i], count * sizeof(float));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(recvbuf[i], count * sizeof(float), 0, count * sizeof(float));
        result_buff[i] = (float*)sal_malloc(count * sizeof(float));
        sal_memset(result_buff[i], count * sizeof(float), 0, count * sizeof(float));
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
            result_buff[i][j] = 4.0;
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
        para_info[i].offline = false;
    }

    // bool fftsSwitch = GetParam();
    // 创建每个Dev的allreduce任务线程
    for (s32 i = 0; i < ndev; ++i) {
        // if (fftsSwitch) {
        // tid[i] = sal_thread_create("thread", inter_reduce_scatter_task_1_ffts, (void*)&para_info[i]);
        // } else {
        tid[i] = sal_thread_create("thread", inter_reduce_scatter_task_1, (void*)&para_info[i]);
        // }

        EXPECT_NE(tid[i], (sal_thread_t)NULL);
    }

    for (s32 i = 0; i < ndev; ++i) {
        while (sal_thread_is_running(tid[i])) {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
    }

    // 获取stream的操作的同步信号量

    // if (!fftsSwitch) {
    HCCL_ERROR("ndev, %u", ndev);
    for (s32 i = 0; i < ndev; i++) {
        for (s32 j = 0; j < count; j++) {
            float res = result_buff[i][j];
            float recv = recvbuf[i][j];

            if (abs(res - recv) > 1e-6) {
                HCCL_ERROR(" recvbuf[%f] result_buff[%f] \n", recv, res);
                errors++;
                break;
            }
        }
    }
    // }

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
    for (s32 i = 0; i < ndev; i++) {
        set_chip_type_stub(i, static_cast<s32>(DevType::DEV_TYPE_910));
    }
    set_board_id(0);
    unsetenv("HCCL_DETERMINISTIC");
    ResetInitState();
    InitExternalInput();
    remove(file_name_t);
}
#endif

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
    rankVec[0].deviceInfo.port = 16666;
    rankVec[0].deviceInfo.vnicPort = 16667;
    HcclIpAddress ipAddr1(1694542016);
    rankVec[0].deviceInfo.deviceIp.push_back(ipAddr1); // 101.0.168.192
    rankVec[0].serverIdx = 0;
    rankVec[0].serverId = "192.168.0.101";
    rankVec[1].rankId = 1;
    rankVec[1].deviceInfo.devicePhyId = 0;
    rankVec[1].deviceInfo.port = 16666;
    rankVec[1].deviceInfo.vnicPort = 16667;
    HcclIpAddress ipAddr2(1711319232);
    rankVec[1].deviceInfo.deviceIp.push_back(ipAddr2); // 101.0.168.192
    rankVec[1].serverIdx = 1;
    rankVec[1].serverId = "192.168.0.102";
    rankTable.rankList.assign(rankVec.begin(), rankVec.end());
    rankTable.deviceNum = 2;
    rankTable.serverNum = 2;
}

void get_ranks_8server_1dev(std::vector<RankInfo>& rank_vector)
{
    RankInfo tmp_para_0;
    tmp_para_0.userRank = 0;
    tmp_para_0.devicePhyId = 0;
    tmp_para_0.worldRank = 0;
    tmp_para_0.deviceType = DevType::DEV_TYPE_910;
    tmp_para_0.serverIdx = 0;
    tmp_para_0.serverId = "10.0.0.10";
    tmp_para_0.nicIp.push_back(HcclIpAddress("192.168.0.11"));
    tmp_para_0.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    tmp_para_0.deviceNicPort = 16666;
    tmp_para_0.deviceVnicPort = 16667;

    RankInfo tmp_para_1;
    tmp_para_1.userRank = 1;
    tmp_para_1.devicePhyId = 1;
    tmp_para_1.worldRank = 1;
    tmp_para_1.deviceType = DevType::DEV_TYPE_910;
    tmp_para_1.serverIdx = 1;
    tmp_para_1.serverId = "10.0.1.10";
    tmp_para_1.nicIp.push_back(HcclIpAddress("192.168.0.12"));
    tmp_para_1.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    tmp_para_1.deviceNicPort = 16666;
    tmp_para_1.deviceVnicPort = 16667;

    RankInfo tmp_para_2;
    tmp_para_2.userRank = 2;
    tmp_para_2.devicePhyId = 2;
    tmp_para_2.worldRank = 2;
    tmp_para_2.deviceType = DevType::DEV_TYPE_910;
    tmp_para_2.serverIdx = 2;
    tmp_para_2.serverId = "10.0.2.10";
    tmp_para_2.nicIp.push_back(HcclIpAddress("192.168.0.13"));
    tmp_para_2.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    tmp_para_2.deviceNicPort = 16666;
    tmp_para_2.deviceVnicPort = 16667;

    RankInfo tmp_para_3;
    tmp_para_3.userRank = 3;
    tmp_para_3.devicePhyId = 3;
    tmp_para_3.worldRank = 3;
    tmp_para_3.deviceType = DevType::DEV_TYPE_910;
    tmp_para_3.serverIdx = 3;
    tmp_para_3.serverId = "10.0.3.10";
    tmp_para_3.nicIp.push_back(HcclIpAddress("192.168.0.14"));
    tmp_para_3.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    tmp_para_3.deviceNicPort = 16666;
    tmp_para_3.deviceVnicPort = 16667;

    RankInfo tmp_para_4;
    tmp_para_4.userRank = 4;
    tmp_para_4.devicePhyId = 4;
    tmp_para_4.worldRank = 4;
    tmp_para_4.deviceType = DevType::DEV_TYPE_910;
    tmp_para_4.serverIdx = 4;
    tmp_para_4.serverId = "10.0.4.10";
    tmp_para_4.nicIp.push_back(HcclIpAddress("192.168.0.15"));
    tmp_para_4.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    tmp_para_4.deviceNicPort = 16666;
    tmp_para_4.deviceVnicPort = 16667;

    RankInfo tmp_para_5;
    tmp_para_5.userRank = 5;
    tmp_para_5.devicePhyId = 5;
    tmp_para_5.worldRank = 5;
    tmp_para_5.deviceType = DevType::DEV_TYPE_910;
    tmp_para_5.serverIdx = 5;
    tmp_para_5.serverId = "10.0.5.10";
    tmp_para_5.nicIp.push_back(HcclIpAddress("192.168.0.16"));
    tmp_para_5.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    tmp_para_5.deviceNicPort = 16666;
    tmp_para_5.deviceVnicPort = 16667;

    RankInfo tmp_para_6;
    tmp_para_6.userRank = 6;
    tmp_para_6.devicePhyId = 6;
    tmp_para_6.worldRank = 6;
    tmp_para_6.deviceType = DevType::DEV_TYPE_910;
    tmp_para_6.serverIdx = 6;
    tmp_para_6.serverId = "10.0.6.10";
    tmp_para_6.nicIp.push_back(HcclIpAddress("192.168.0.17"));
    tmp_para_6.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    tmp_para_6.deviceNicPort = 16666;
    tmp_para_6.deviceVnicPort = 16667;

    RankInfo tmp_para_7;
    tmp_para_7.userRank = 7;
    tmp_para_7.devicePhyId = 7;
    tmp_para_7.worldRank = 7;
    tmp_para_7.deviceType = DevType::DEV_TYPE_910;
    tmp_para_7.serverIdx = 7;
    tmp_para_7.serverId = "10.0.7.10";
    tmp_para_7.nicIp.push_back(HcclIpAddress("192.168.0.18"));
    tmp_para_7.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    tmp_para_7.deviceNicPort = 16666;
    tmp_para_7.deviceVnicPort = 16667;

    rank_vector.push_back(tmp_para_0);
    rank_vector.push_back(tmp_para_1);
    rank_vector.push_back(tmp_para_2);
    rank_vector.push_back(tmp_para_3);
    rank_vector.push_back(tmp_para_4);
    rank_vector.push_back(tmp_para_5);
    rank_vector.push_back(tmp_para_6);
    rank_vector.push_back(tmp_para_7);
    return;
}

TEST_F(HcclCommTest910B, ut_SetAlgType_module_8server_1dev_ring_ring)
{
    HcclResult ret;
    std::string algo = "level0:ring;level1:ring";
    ret = SetHcclAlgoConfig(algo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclCommunicator implstub;
    std::vector<RankInfo> ranks;
    get_ranks_8server_1dev(ranks);
    std::map<HcclCMDType, AlgType> algType;
    MOCKER_CPP_VIRTUAL(implstub, &HcclCommunicator::IsStandardCard).stubs().will(returnValue(false));

    u32 ifnumVersion = 3;
    MOCKER(hrtRaGetInterfaceVersion)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&ifnumVersion))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&NetworkManager::CheckAutoListenVersion).stubs().will(returnValue(HCCL_SUCCESS));

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    params.totalRanks = 8;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    WorldGroupInfo groupCommonData;
    groupCommonData.serverId = "10.0.0.10";
    groupCommonData.ranksPort = {16666, 16666, 16666, 16666, 16666, 16666, 16666, 16666};
    groupCommonData.vnicRanksPort = {16666, 16666, 16666, 16666, 16666, 16666, 16666, 16666};
    groupCommonData.deviceLogicId = 0;
    MOCKER_CPP(&HcclSocket::Listen, HcclResult(HcclSocket::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, ranks, groupCommonData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;

    // 当前将serverId设置为impl的成员变量，之后添加新llt用例需要考虑serverid以获取server内dev数
    ret = implBase->attrCollector_.SetInnerServerAverageDevice(ranks);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    impl->deviceNumPerAggregation_ = implBase->deviceNumPerAggregation_;
    impl->deviceNumPerServer_ = implBase->deviceNumPerServer_;
    ret = algConfigurator->SelectAlgType(8, DevType::DEV_TYPE_COUNT, algType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel0, AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING);
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel1, AlgTypeLevel1::ALG_LEVEL1_RING);

    GlobalMockObject::verify();
}

#if 1
void* inter_reduce_scatter_mesh_atomic_single_operator_task(void* parg)
{
    HcclResult ret = HCCL_SUCCESS;
    para_t* para_info = (para_t*)parg;
    s32 rank_num_tmp;

    HcomInfo hcom_info;
    std::string ranktable_file = para_info->file_name;
    std::string rankTableM;
    std::string realFilePath;
    hrtSetDevice(para_info->device_id);
    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, para_info->identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    InitExternalInput();

    sal_memset(hcom_info.params.id.internal, HCCL_ROOT_INFO_BYTES, 0, sizeof(hcom_info.params.id.internal));
    sal_memcpy(hcom_info.params.id.internal, sizeof(HcclRootInfo), &para_info->rootInfo, sizeof(HcclRootInfo));

    hcom_info.pComm.reset(new (std::nothrow) hccl::hcclComm(200 * 1024 * 1024, 200 * 1024 * 1024, HCCL_WORLD_GROUP));
    rtModel_t model = (void*)1;

    CommConfig commConfig("hccl_world_group");
    ret = hcom_info.pComm->init(hcom_info.params, commConfig, hcom_info.rankTable);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] task reduce_scatter fails", para_info->device_id);
    }

    u64 stream_list_size = 0;
    ret = hcom_info.pComm->GetWorkspaceSubStreamNum(stream_list_size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HCCL_INFO("get stream_list_size[%d] success", stream_list_size);
    vector<HcclRtStream> streamList(stream_list_size);
    void* memptr = nullptr;

    //-----------------Set Workspace Resource Start------------------//
    rtError_t rt_ret;
    // 生成从stream
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = aclrtCreateStreamWithConfig(&streamList[i], 0, ACL_STREAM_PERSISTENT);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
        // 从流bind到model
        rt_ret = rtModelBindStream(model, streamList[i], RT_MODEL_WAIT_ACTIVE_STREAM);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    u32 rankSize = 0;
    ret = hcom_info.pComm->GetRankSize(rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u64 memSize = 0;
    ret = hcom_info.pComm->GetWorkspaceMemSize(
        HCCL_KERNEL_OP_TYPE_REDUCESCATTER, para_info->count, para_info->datatype, rankSize, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtMalloc(&memptr, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hcom_info.pComm->SetWorkspaceResource(
        "tag_inter_reduce_scatter_mesh_atomic_single_operator_task", memptr, memSize, streamList);
    EXPECT_EQ(ret, HCCL_SUCCESS);

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
    ret = hcom_info.pComm->communicator_->ReduceScatterOutPlace(
        "tag_inter_reduce_scatter_mesh_atomic_single_operator_task", para_info->sendbuff, para_info->recvbuff,
        para_info->count, para_info->datatype, para_info->op, para_info->stream);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("rank[%d] task reduce_scatter fails", hcom_info.params.rank);
    }

    rt_ret = RT_ERROR_NONE;
    rt_ret = aclrtSynchronizeStream(para_info->stream);
    //--------------Resource destroy----------------//
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = rtModelUnbindStream(model, streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);

        rt_ret = aclrtDestroyStream(streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }
    hrtFree(memptr);
    if (rt_ret != RT_ERROR_NONE) {
        HCCL_ERROR("rank[%d] task allgather fails", hcom_info.params.rank);
    }
    return (nullptr);
}
#endif

#if 1
TEST_F(HcclCommTest910B, ut_reducescatter_4p_mesh_atomic_single_operator)
{
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    RankConsistentcyChecker::GetInstance().ClearCheckInfo();
    nlohmann::json rank_table = rank_table_910_1server_4rank;
    char file_name_t[] = "./ut_allreduce_4p_ring.json";
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

    float* result_buff[4];
    float* sendbuf[4];
    float* recvbuf[4];

    s32 sync_value = 0;

    rtStream_t stream[4];
    sal_thread_t tid[4];
    para_t para_info[4];

    HcclDataType datatype = HCCL_DATA_TYPE_FP32;

    HcclReduceOp op = HCCL_REDUCE_SUM;
    s32 count = 10;
    s32 ndev = 4;
    set_board_id(0x0000);
    for (s32 i = 0; i < ndev; i++) {
        set_chip_type_stub(i, static_cast<s32>(DevType::DEV_TYPE_910B));
    }

    HcclRootInfo rootInfo;
    ret = hccl::hcclComm::GetUniqueId(&rootInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    /** 初始化输入输出缓存 */
    for (s32 i = 0; i < ndev; i++) {
        ret = hrtMalloc((void**)&sendbuf[i], ndev * count * sizeof(float));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(sendbuf[i], ndev * count * sizeof(float), 0, ndev * count * sizeof(float));

        ret = hrtMalloc((void**)&recvbuf[i], count * sizeof(float));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(recvbuf[i], count * sizeof(float), 0, count * sizeof(float));
        result_buff[i] = (float*)sal_malloc(count * sizeof(float));
        sal_memset(result_buff[i], count * sizeof(float), 0, count * sizeof(float));
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
            result_buff[i][j] = 4.0;
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
        para_info[i].offline = false;
    }

    // 创建每个Dev的allreduce任务线程
    for (s32 i = 0; i < ndev; ++i) {
        tid[i]
            = sal_thread_create("thread", inter_reduce_scatter_mesh_atomic_single_operator_task, (void*)&para_info[i]);
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
            float res = result_buff[i][j];
            float recv = recvbuf[i][j];

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
    for (s32 i = 0; i < ndev; i++) {
        set_chip_type_stub(i, static_cast<s32>(DevType::DEV_TYPE_910));
    }
    set_board_id(0);
    unsetenv("HCCL_OP_EXPANSION_MODE");
    remove(file_name_t);
}
#endif

void* inter_all_gather_outplace_task_single_operator(void* parg)
{
    HcclResult ret = HCCL_SUCCESS;
    para_t* para_info = (para_t*)parg;
    s32 rank_num_tmp;

    HcomInfo hcom_info;
    std::string ranktable_file = para_info->file_name;
    std::string rankTableM;
    std::string realFilePath;

    hrtSetDevice(para_info->device_id);
    set_chip_type_stub(para_info->device_id, static_cast<s32>(DevType::DEV_TYPE_910B));
    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    InitExternalInput();

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, para_info->identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sal_memset(hcom_info.params.id.internal, HCCL_ROOT_INFO_BYTES, 0, sizeof(hcom_info.params.id.internal));
    sal_memcpy(hcom_info.params.id.internal, sizeof(HcclRootInfo), &para_info->rootInfo, sizeof(HcclRootInfo));

    hcom_info.pComm.reset(new (std::nothrow) hccl::hcclComm(1024 * 1024 * 10, 1024 * 1024 * 10, HCCL_WORLD_GROUP));
    rtModel_t model = (void*)1;
    hcom_info.params.deviceType = DevType::DEV_TYPE_910B;

    CommConfig commConfig("hccl_world_group");
    ret = hcom_info.pComm->init(hcom_info.params, commConfig, hcom_info.rankTable);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] task all_gather fails", para_info->device_id);
    }
    u64 stream_list_size = 0;
    ret = hcom_info.pComm->GetWorkspaceSubStreamNum(stream_list_size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HCCL_INFO("get stream_list_size[%d] success", stream_list_size);
    vector<HcclRtStream> streamList(stream_list_size);
    void* memptr = nullptr;

    //-----------------Set Workspace Resource Start------------------//
    rtError_t rt_ret;
    // 生成从stream
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = aclrtCreateStreamWithConfig(&streamList[i], 0, ACL_STREAM_PERSISTENT);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
        // 从流bind到model
        rt_ret = rtModelBindStream(model, streamList[i], RT_MODEL_WAIT_ACTIVE_STREAM);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    u32 rankSize = 0;
    ret = hcom_info.pComm->GetRankSize(rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u64 memSize = 0;
    ret = hcom_info.pComm->GetWorkspaceMemSize(
        HCCL_KERNEL_OP_TYPE_ALLGATHER, para_info->count, para_info->datatype, rankSize, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtMalloc(&memptr, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    string strTag = "allgather_tag_magic9999998";

    ret = hcom_info.pComm->SetWorkspaceResource(strTag, memptr, memSize, streamList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    //-----------------Set Workspace Resource End------------------//

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

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] comm get map streamModel fail!", para_info->device_id);
    }

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
    HcomCollOpInfo opInfo;
    opInfo.inputAddr = para_info->sendbuff;
    opInfo.outputAddr = para_info->recvbuff;
    opInfo.count = para_info->count;
    opInfo.dataType = para_info->datatype;
    ret = hcom_info.pComm->communicator_->AllGatherOutPlace(
        strTag, para_info->sendbuff, para_info->recvbuff, para_info->count, para_info->datatype, para_info->stream);

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] task HcclAllGather fails", hcom_info.params.rank);
    }

    rt_ret = aclrtSynchronizeStream(para_info->stream);
    //--------------Resource destroy----------------//
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = rtModelUnbindStream(model, streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);

        rt_ret = aclrtDestroyStream(streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }
    hrtFree(memptr);

    if (rt_ret != RT_ERROR_NONE) {
        HCCL_ERROR("rank[%d] task allgather fails", hcom_info.params.rank);
    }

    return (nullptr);
}

TEST_F(HcclCommTest910B, ut_allgather_outplace_4p_mesh_single_operator)
{
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    RankConsistentcyChecker::GetInstance().ClearCheckInfo();
    nlohmann::json rank_table = rank_table_910_1server_4rank;
    char file_name_t[] = "./ut_allgather_4p_mesh.json";
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

    s8* result_buff[4];
    s8* sendbuf[4];
    s8* recvbuf[4];
    s8* inputbuf[4];
    s8* outputbuf[4];

    s32 sync_value = 0;

    rtStream_t stream[4];
    sal_thread_t tid[4];
    para_t para_info[4];

    HcclDataType datatype = HCCL_DATA_TYPE_INT8;
    s32 count = HCCL_ALLGATHER_DATA_SIZE;
    HCCL_ERROR("count : [%d]", count);
    s32 ndev = 4;
    HcclRootInfo rootInfo;
    set_board_id(0x0000);
    for (s32 i = 0; i < ndev; i++) {
        set_chip_type_stub(i, static_cast<s32>(DevType::DEV_TYPE_910B));
    }
    ret = hccl::hcclComm::GetUniqueId(&rootInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    /** 初始化输入输出缓存 */
    for (s32 i = 0; i < ndev; i++) {
        ret = hrtMalloc((void**)&(sendbuf[i]), count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(sendbuf[i], count * sizeof(s8), 0, count * sizeof(s8));
        ret = hrtMalloc((void**)&(recvbuf[i]), count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(recvbuf[i], count * sizeof(s8) * ndev, 0, count * sizeof(s8) * ndev);
        ret = hrtMalloc((void**)&(result_buff[i]), count * sizeof(s8));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(result_buff[i], count * sizeof(s8) * ndev, 0, count * sizeof(s8) * ndev);
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
        for (u32 j = 0; j < count * ndev; j++) {
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

        para_info[i].sync_addr = &sync_value;
        para_info[i].file_name = file_name_t;
        para_info[i].offline = false;
    }

    // 创建每个Dev的allgather任务线程
    for (s32 i = 0; i < ndev; i++) {
        tid[i] = sal_thread_create("thread", inter_all_gather_outplace_task_single_operator, (void*)&para_info[i]);
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
            HCCL_ERROR("i, j, result_buff[i][j], %d, %d, %d", i, j, result_buff[i][j]);
            s8 recv = outputbuf[i][j];
            HCCL_ERROR("i, j, outputbuf[i][j], %d, %d, %d", i, j, outputbuf[i][j]);
            if (res != recv) {
                HCCL_ERROR(" recvbuf[%d] result_buff[%d] \n", recv, res);
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
    for (s32 i = 0; i < ndev; i++) {
        set_chip_type_stub(i, static_cast<s32>(DevType::DEV_TYPE_910));
    }
    set_board_id(0);
    unsetenv("HCCL_OP_EXPANSION_MODE");
    remove(file_name_t);
}

void* inter_reduce_scatter_task_undeter_opbase(void* parg)
{
    HcclResult ret = HCCL_SUCCESS;
    para_t* para_info = (para_t*)parg;
    s32 rank_num_tmp;

    HcomInfo hcom_info;
    std::string ranktable_file = para_info->file_name;
    std::string rankTableM;
    std::string realFilePath;

    hrtSetDevice(para_info->device_id);
    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, para_info->identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sal_memset(hcom_info.params.id.internal, HCCL_ROOT_INFO_BYTES, 0, sizeof(hcom_info.params.id.internal));
    sal_memcpy(hcom_info.params.id.internal, sizeof(HcclRootInfo), &para_info->rootInfo, sizeof(HcclRootInfo));

    hcom_info.pComm.reset(new (std::nothrow)
                              hccl::hcclComm(HCCL_ALLREDUCE_DATA_SLICE, HCCL_ALLREDUCE_DATA_SLICE, HCCL_WORLD_GROUP));
    rtModel_t model = (void*)1;

    CommConfig commConfig("hccl_world_group");
    ret = hcom_info.pComm->init(hcom_info.params, commConfig, hcom_info.rankTable);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] task all_reduce fails", para_info->device_id);
    }
    u64 stream_list_size = 0;
    ret = hcom_info.pComm->GetWorkspaceSubStreamNum(stream_list_size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HCCL_INFO("get stream_list_size[%d] success", stream_list_size);
    vector<HcclRtStream> streamList(stream_list_size);
    void* memptr = nullptr;

    //-----------------Set Workspace Resource Start------------------//
    rtError_t rt_ret;
    // 生成从stream
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = aclrtCreateStreamWithConfig(&streamList[i], 0, ACL_STREAM_PERSISTENT);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
        // 从流bind到model
        rt_ret = rtModelBindStream(model, streamList[i], RT_MODEL_WAIT_ACTIVE_STREAM);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    u32 rankSize = 0;
    ret = hcom_info.pComm->GetRankSize(rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u64 memSize = 131072;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtMalloc(&memptr, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    string strTag = "reducescatter_tag_deter4561637_" + to_string(para_info->id);

    ret = hcom_info.pComm->SetWorkspaceResource(strTag, memptr, memSize, streamList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    //-----------------Set Workspace Resource End------------------//

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

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] comm get map streamModel fail!", para_info->device_id);
    }

    if (para_info->deviceNumPerServer > 0) {
        hcclImpl* impl = hcom_info.pComm->communicator_->implAlg_->pimpl_.get();
        impl->deviceNumPerAggregation_ = para_info->deviceNumPerServer;
    }
    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    ret = hcom_info.pComm->ReduceScatterOutPlace(
        strTag, para_info->sendbuff, para_info->recvbuff, para_info->count, para_info->datatype, para_info->op,
        para_info->stream);

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] task HcclReduceScatter fails", hcom_info.params.rank);
    }

    rt_ret = aclrtSynchronizeStream(para_info->stream);
    //--------------Resource destroy----------------//
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = rtModelUnbindStream(model, streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);

        rt_ret = aclrtDestroyStream(streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }
    hrtFree(memptr);
    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_RESERVED);
    if (rt_ret != RT_ERROR_NONE) {
        HCCL_ERROR("rank[%d] task allgather fails", hcom_info.params.rank);
    }

    return nullptr;
}

TEST_F(HcclCommTest910B, ut_reducescatter_8p_mesh_undeterministic_small_count_opbase)
{
    ResetInitState();
    InitExternalInput();

    RankConsistentcyChecker::GetInstance().ClearCheckInfo();
    nlohmann::json rank_table = rank_table_910_1server_4rank;
    char file_name_t[] = "./ut_allreduce_4p_ring.json";
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
    s32 count = 10;
    s32 ndev = DEV_NUM_4;
    HcclRootInfo rootInfo;
    set_board_id(0x0000);
    for (s32 i = 0; i < ndev; i++) {
        set_chip_type_stub(i, static_cast<s32>(DevType::DEV_TYPE_910B));
    }
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
        for (u32 i = 0; i < count * ndev; i++) {
            sendbuf[j][i] = 1;
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
        para_info[i].id = 0;

        para_info[i].count = count;
        para_info[i].datatype = datatype;
        para_info[i].sendbuff = sendbuf[i];
        para_info[i].stream = stream[i];
        para_info[i].recvbuff = recvbuf[i];
        para_info[i].op = op;

        para_info[i].sync_addr = &sync_value;
        para_info[i].file_name = file_name_t;
        para_info[i].offline = false;
        para_info[i].deviceNumPerServer = 8;
    }

    // 创建每个Dev的allreduce任务线程
    for (s32 i = 0; i < ndev; i++) {
        tid[i] = sal_thread_create("thread", inter_reduce_scatter_task_undeter_opbase, (void*)&para_info[i]);
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
            s8 recv = recvbuf[i][j];

            if (res != recv) {
                HCCL_ERROR(" recvbuf[%d] result_buff[%d] \n", recv, res);
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
    for (s32 i = 0; i < ndev; i++) {
        set_chip_type_stub(i, static_cast<s32>(DevType::DEV_TYPE_910));
    }
    set_board_id(0);
    remove(file_name_t);
    ResetInitState();
    InitExternalInput();
}

#if 1
void* inter_reduce_scatter_atomic_single_operator_task(void* parg)
{
    HcclResult ret = HCCL_SUCCESS;
    para_t* para_info = (para_t*)parg;
    s32 rank_num_tmp;

    HcomInfo hcom_info;
    std::string ranktable_file = para_info->file_name;
    std::string rankTableM;
    std::string realFilePath;
    hrtSetDevice(para_info->device_id);
    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, para_info->identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    InitExternalInput();

    sal_memset(hcom_info.params.id.internal, HCCL_ROOT_INFO_BYTES, 0, sizeof(hcom_info.params.id.internal));
    sal_memcpy(hcom_info.params.id.internal, sizeof(HcclRootInfo), &para_info->rootInfo, sizeof(HcclRootInfo));

    hcom_info.pComm.reset(new (std::nothrow) hccl::hcclComm(200 * 1024 * 1024, 200 * 1024 * 1024, HCCL_WORLD_GROUP));
    rtModel_t model = (void*)1;

    CommConfig commConfig("hccl_world_group");
    ret = hcom_info.pComm->init(hcom_info.params, commConfig, hcom_info.rankTable);
    HcclCommunicator* impl = dynamic_cast<HcclCommunicator*>(hcom_info.pComm->communicator_.get());
    impl->implAlg_->pimpl_->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] task reduce_scatter fails", para_info->device_id);
    }

    u64 stream_list_size = 0;
    ret = hcom_info.pComm->GetWorkspaceSubStreamNum(stream_list_size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HCCL_INFO("get stream_list_size[%d] success", stream_list_size);
    vector<HcclRtStream> streamList(stream_list_size);
    void* memptr = nullptr;

    //-----------------Set Workspace Resource Start------------------//
    rtError_t rt_ret;
    // 生成从stream
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = aclrtCreateStreamWithConfig(&streamList[i], 0, ACL_STREAM_PERSISTENT);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
        // 从流bind到model
        rt_ret = rtModelBindStream(model, streamList[i], RT_MODEL_WAIT_ACTIVE_STREAM);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    u32 rankSize = 0;
    ret = hcom_info.pComm->GetRankSize(rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u64 memSize = 0;
    ret = hcom_info.pComm->GetWorkspaceMemSize(
        HCCL_KERNEL_OP_TYPE_REDUCESCATTER, para_info->count, para_info->datatype, rankSize, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtMalloc(&memptr, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hcom_info.pComm->SetWorkspaceResource(
        "tag_inter_reduce_scatter_atomic_single_operator_task", memptr, memSize, streamList);
    EXPECT_EQ(ret, HCCL_SUCCESS);

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

    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    ret = hcom_info.pComm->communicator_->ReduceScatterOutPlace(
        "tag_inter_reduce_scatter_atomic_single_operator_task", para_info->sendbuff, para_info->recvbuff,
        para_info->count, para_info->datatype, para_info->op, para_info->stream);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("rank[%d] task reduce_scatter fails", hcom_info.params.rank);
    }

    rt_ret = RT_ERROR_NONE;
    rt_ret = aclrtSynchronizeStream(para_info->stream);
    //--------------Resource destroy----------------//
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = rtModelUnbindStream(model, streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);

        rt_ret = aclrtDestroyStream(streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }
    hrtFree(memptr);
    if (rt_ret != RT_ERROR_NONE) {
        HCCL_ERROR("rank[%d] task allgather fails", hcom_info.params.rank);
    }
    return (nullptr);
}

TEST_F(HcclCommTest910B, ut_reducescatter_4p_atomic_single_operator)
{
    MOCKER(IsSuperPodMode).stubs().with(mockcpp::any()).will(returnValue(false));
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    RankConsistentcyChecker::GetInstance().ClearCheckInfo();
    nlohmann::json rank_table = rank_table_910_1server_4rank;
    char file_name_t[] = "./ut_allreduce_4p_ring.json";
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

    float* result_buff[4];
    float* sendbuf[4];
    float* recvbuf[4];

    s32 sync_value = 0;

    rtStream_t stream[4];
    sal_thread_t tid[4];
    para_t para_info[4];

    HcclDataType datatype = HCCL_DATA_TYPE_FP32;

    HcclReduceOp op = HCCL_REDUCE_SUM;
    s32 count = 10;
    s32 ndev = 4;
    set_board_id(0x0000);
    for (s32 i = 0; i < ndev; i++) {
        set_chip_type_stub(i, static_cast<s32>(DevType::DEV_TYPE_910_93));
    }

    HcclRootInfo rootInfo;
    ret = hccl::hcclComm::GetUniqueId(&rootInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    /** 初始化输入输出缓存 */
    for (s32 i = 0; i < ndev; i++) {
        ret = hrtMalloc((void**)&sendbuf[i], ndev * count * sizeof(float));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(sendbuf[i], ndev * count * sizeof(float), 0, ndev * count * sizeof(float));

        ret = hrtMalloc((void**)&recvbuf[i], count * sizeof(float));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(recvbuf[i], count * sizeof(float), 0, count * sizeof(float));
        result_buff[i] = (float*)sal_malloc(count * sizeof(float));
        sal_memset(result_buff[i], count * sizeof(float), 0, count * sizeof(float));
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
            result_buff[i][j] = 4.0;
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
        para_info[i].offline = false;
    }
    MOCKER_CPP(&HcclCommunicator::ExecOp).stubs().will(returnValue(HCCL_SUCCESS));
    // 创建每个Dev的allreduce任务线程
    for (s32 i = 0; i < ndev; ++i) {
        tid[i] = sal_thread_create("thread", inter_reduce_scatter_atomic_single_operator_task, (void*)&para_info[i]);
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
            float res = result_buff[i][j];
            float recv = recvbuf[i][j];

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
    for (s32 i = 0; i < ndev; i++) {
        set_chip_type_stub(i, static_cast<s32>(DevType::DEV_TYPE_910));
    }
    set_board_id(0);
    unsetenv("HCCL_OP_EXPANSION_MODE");
    remove(file_name_t);
    GlobalMockObject::verify();
}
#endif

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
#include "all_reduce_mesh_opbase_pub.h"
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
#include "ranktable/v80_rank_table.h"
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
#include "dispatcher_pub.h"
#include "dispatcher_pub.h"
using namespace std;
using namespace hccl;

class HcclCommTest910BNEW : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--HcclCommTest910B SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--HcclCommTest910B TearDown--\033[0m" << std::endl; }
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
        MOCKER_CPP(&Heartbeat::RegisterRanks).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&Heartbeat::UnRegisterRanks).stubs().will(returnValue(HCCL_SUCCESS));
        ResetInitState();
        SetFftsSwitch(true);
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        TsdClose(1);
        SetFftsSwitch(true);
        std::cout << "A Test TearDown" << std::endl;
    }
};

#define DEV_NUM_4 4
#define HCCL_ALLREDUCE_DATA_SIZE 1024 * 1024 * 1024
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
    Stream stream;
    int id;
    volatile s32* sync_addr;
    bool offline;
} para_t;

void* inter_all_reduce_outplace_task_1_new(void* parg)
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
    u32* notifyIdOut = new u32(0);
    MOCKER(hrtGetNotifyID).stubs().with(mockcpp::any(), outBound(notifyIdOut)).will(returnValue(HCCL_SUCCESS));
    CommConfig commConfig("hccl_world_group");
    ret = hcom_info.pComm->init(hcom_info.params, commConfig, hcom_info.rankTable);
    hcclImpl* impl = dynamic_cast<hcclImpl*>(hcom_info.pComm->impl_.get());
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
        para_info->stream, SyncMode::CONFIGURABLE_TIMEWAITSYNCMODE);

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] task HcclAllReduce fails", hcom_info.params.rank);
    }

    rt_ret = aclrtSynchronizeStream(para_info->stream);

    if (rt_ret != RT_ERROR_NONE) {
        HCCL_ERROR("rank[%d] task allgather fails", hcom_info.params.rank);
    }
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
    delete notifyIdOut;
    (void)SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
    return (nullptr);
}

TEST_F(HcclCommTest910BNEW, ut_allreduce_outplace_4p_mesh)
{
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
    s8* inputbuf[DEV_NUM_4];
    s8* outputbuf[DEV_NUM_4];

    s32 sync_value = 0;

    rtStream_t stream[DEV_NUM_4];
    sal_thread_t tid[DEV_NUM_4];
    para_t para_info[DEV_NUM_4];

    HcclDataType datatype = HCCL_DATA_TYPE_INT8;

    HcclReduceOp op = HCCL_REDUCE_SUM;
    s32 count = HCCL_ALLREDUCE_DATA_SIZE;
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
        sal_memset(recvbuf[i], count * sizeof(s8), 0, count * sizeof(s8));
        ret = hrtMalloc((void**)&(result_buff[i]), count * sizeof(s8));
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
        para_info[i].offline = false;
    }

    // 创建每个Dev的allreduce任务线程
    for (s32 i = 0; i < ndev; i++) {
        tid[i] = sal_thread_create("thread", inter_all_reduce_outplace_task_1_new, (void*)&para_info[i]);
        EXPECT_NE(tid[i], (sal_thread_t)NULL);
    }

    for (s32 i = 0; i < ndev; i++) {
        while (sal_thread_is_running(tid[i])) {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
    }

    // 获取stream的操作的同步信号量
    for (s32 i = 0; i < ndev; i++) {
        HCCL_ERROR("count %d", count);
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

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

#include <driver/ascend_hal.h>

#include "hccl/hcom.h"
#include "hcom_pub.h"
#include "llt_hccl_stub_pub.h"
#include <sys/mman.h>
#include <fcntl.h>
#include "sal.h"
#include "hccl_comm_pub.h"
#include "hccl_communicator.h"
#include "externalinput.h"

#include "rank_consistentcy_checker.h"
#include <iostream>
#include <fstream>

#include "config.h"
#include "dlra_function.h"
#include "topoinfo_ranktableParser_pub.h"

#include "v80_rank_table.h"
#include "network_manager_pub.h"
#include "tsd/tsd_client.h"
#include "dltdt_function.h"
#include "param_check_pub.h"
#include "env_config.h"
using namespace std;
using namespace hccl;
class ST_Send_Receive_Test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ST_Send_Receive_Test SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "ST_Send_Receive_Test TearDown" << std::endl; }

    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        static s32 call_cnt = 0;
        DlTdtFunction::GetInstance().DlTdtFunctionInit();
        TsdOpen(1, 2);
        std::cout << "tsd open" << std::endl;

        string name = std::to_string(call_cnt++) + "_" + __PRETTY_FUNCTION__;
        ra_set_shm_name(name.c_str());
        setenv("HCCL_DFS_CONFIG", "connection_fault_detection_time:0", 1);
        InitEnvParam();
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A TestCase TearDown" << std::endl;
    }
};

#define P2P_DATA_SIZE_LIGHT 20
#define P2P_DATA_SIZE_HEAVY 1200000
#define P2P_DATA_SIZE_S_HEAVY 3000000 /* 超大数据，约10M */

typedef struct p2p_para_struct {
    HcclRootInfo rootInfo;
    std::string identify;
    s32 device_id;
    char* file_name;
    bool sender_flag;
    s32 peer_rank;
    void* buffer;
    s32 count;
    HcclDataType datatype;
    rtStream_t stream;
    volatile s32* sync_addr;
    const char* tag;
    char* groupName;
    u32 groupRanksNum;
    u32* pGroupRanks;
    s32 ranks_local;
} p2p_para_t;

void* intra_send_receive_task(void* parg)
{
    s32 portNum = 7;
    MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HCCL_SUCCESS;
    p2p_para_t* para_info = (p2p_para_t*)parg;

    hrtSetDevice(para_info->device_id);

    HcomInfo hcom_info;
    std::string ranktable_file = para_info->file_name;
    std::string rankTableM;
    std::string realFilePath;

    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, para_info->identify, hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    char* charModel = new char;
    rtModel_t model = (void*)charModel;

    sal_memcpy(hcom_info.params.id.internal, sizeof(HcclRootInfo), &para_info->rootInfo, sizeof(HcclRootInfo));

    hcom_info.pComm.reset(new (std::nothrow) hccl::hcclComm());

    CommConfig commConfig("hccl_world_group");
    ret = hcom_info.pComm->init(hcom_info.params, commConfig, hcom_info.rankTable);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] task send_receive fails", para_info->device_id);
    }

    bool swapped;

    //-----------------Get Workspace Resource Start------------------//
    u64 stream_list_size = 0;
    ret = hcom_info.pComm->GetWorkspaceSubStreamNum(stream_list_size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HCCL_INFO("get stream_list_size[%d] success", stream_list_size);
    vector<HcclRtStream> streamList(stream_list_size);

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
    ret = hcom_info.pComm->GetWorkspaceMemSize("HcomSend", para_info->count, para_info->datatype, rankSize, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* memptr = nullptr;
    ret = hrtMalloc(&memptr, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hcom_info.pComm->SetWorkspaceResource("default_tag", memptr, memSize, streamList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    //-----------------Get Workspace Resource End------------------//

    s32 rank_num_tmp = *(para_info->sync_addr) - 1;

    do {
        rank_num_tmp += 1;

        swapped = __sync_bool_compare_and_swap(para_info->sync_addr /** &rank_num */, rank_num_tmp, rank_num_tmp + 1);
    } while (!swapped);

    while (*(para_info->sync_addr) < para_info->ranks_local) {
        sched_yield(); // linux提供一个系统调用运行进程主动让出执行权
    }

    if (para_info->sender_flag) {
        ret = hcom_info.pComm->send(
            para_info->tag, para_info->buffer, para_info->count, para_info->datatype, para_info->peer_rank,
            para_info->stream);
        HCCL_INFO(
            "rank[%s] device[%d] send to rank[%d]", para_info->identify.c_str(), para_info->device_id,
            para_info->peer_rank);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("dev[%d] task send fails", para_info->device_id);
        }

    } else {
        ret = hcom_info.pComm->receive(
            para_info->tag, para_info->buffer, para_info->count, para_info->datatype, para_info->peer_rank,
            para_info->stream);
        HCCL_INFO(
            "rank[%s] device[%d] recv from rank[%d]", para_info->identify.c_str(), para_info->device_id,
            para_info->peer_rank);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("dev[%d] task receive fails", para_info->device_id);
        }
    }

    rt_ret = RT_ERROR_NONE;
    rt_ret = aclrtSynchronizeStream(para_info->stream);

    if (rt_ret != RT_ERROR_NONE) {
        HCCL_ERROR("task sync fails");
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
    delete charModel;
    charModel = nullptr;
    return (NULL);
}
#define SEND_DEV_NUM 2
#if 0 // 执行失败 aclrtNotifyImportByKey
TEST_F(ST_Send_Receive_Test, ut_send_receive_8ranks_1server_float)
{
    char file_name_t[] = "./st_send_receive_2ranks_1server_float.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(4) << rank_table_1server_8rank << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }
    else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }
    outfile.close();
    RankConsistentcyChecker::GetInstance().ClearCheckInfo();
    s32 nnode, rank, errors = 0;

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;

    MOCKER(hrtRaGetInterfaceVersion)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    float* buffer[SEND_DEV_NUM];
    set_board_id(0x0000);
    s32 sync_value = 0;
    s32 noderank = 0;
    rtStream_t stream[SEND_DEV_NUM];
    sal_thread_t tid[SEND_DEV_NUM];
    p2p_para_t para_info[SEND_DEV_NUM];

    HcclDataType datatype = HCCL_DATA_TYPE_FP32;
    s32 count = P2P_DATA_SIZE_S_HEAVY;
    s32 ndev = SEND_DEV_NUM;

    HcclRootInfo rootInfo;
    ret = hccl::hcclComm::GetUniqueId(&rootInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    const s32 send_node_id = 0;
    const s32 recv_node_id = 1;

    /** 初始化输入输出缓存 */
    for (s32 i = 0; i < ndev; i++ )
    {
        ret = hrtMalloc((void**)&buffer[i], (count * sizeof(float)));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(buffer[i], (count * sizeof(float)), 0, (count * sizeof(float)));
    }

        //sendbuf 赋值

        for (u32 i = 0; i < count; i++)
        {
            buffer[0][i] = 1.0 * i * 10000000;
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
        identify << (i);
        para_info[i].tag = "default_tag";
        para_info[i].identify = identify.str();
        para_info[i].device_id = i ;
        para_info[i].ranks_local = ndev;
        para_info[i].count = count;
        para_info[i].datatype = datatype;
        para_info[i].stream = stream[i];
        para_info[i].buffer = buffer[i];
        para_info[i].sync_addr = &sync_value;
        para_info[i].file_name = file_name_t;
        if (i == send_node_id) {
            para_info[i].sender_flag = true;
            para_info[i].peer_rank   = recv_node_id;
            noderank = 0;
        } else if (i == recv_node_id){
            para_info[i].sender_flag = false;
            para_info[i].peer_rank   = send_node_id;
            noderank = 1;
            }
    }

    for (s32 i = 0; i < ndev; ++i)
    {
        tid[i] = sal_thread_create("thread", intra_send_receive_task, (void*)&para_info[i]);
        EXPECT_NE(tid[i], (sal_thread_t )NULL);
    }

    for (s32 i = 0; i < ndev; ++i)
    {
        while ( sal_thread_is_running(tid[i]))
        {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
    }

    //获取stream的操作的同步信号量
    for (s32 j = 0; j < ndev; j++)
    {
       rt_ret = aclrtSynchronizeStream(stream[j]);
       EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    /*check result*/
    for (s32 j = 0; j < ndev; j++)
    {
        for (s32 i = 0; i < count; i++)
        {
            float recv = 1.0 * i * 10000000;

            if (abs(buffer[j][i] - recv) > 1e-6)
            {
                HCCL_ERROR("node:%d result[%d][%d]:%f \n", noderank, j, i, buffer[j][i]);
                errors++;
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

    for (s32 j = 0; j < ndev; j++)
    {
        hrtFree(buffer[j]);
        rt_ret = aclrtDestroyStream(stream[j]);

        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }
    remove(file_name_t);

    EXPECT_EQ(errors, 0);
}

TEST_F(ST_Send_Receive_Test, ut_send_receive_8ranks_1server_float_1)
{
    MOCKER(hrtRaGetInterfaceVersion)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Heartbeat::Init)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::RegisterToHeartBeat, HcclResult(HcclCommunicator::*)(u32, string &))
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));
    char file_name_t[] = "./st_send_receive_2ranks_1server_float.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(4) << rank_table_1server_8rank << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }
    else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }
    outfile.close();
    RankConsistentcyChecker::GetInstance().ClearCheckInfo();
    s32 nnode, rank, errors = 0;

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;

    float* buffer[SEND_DEV_NUM];
    set_board_id(0x0000);
    s32 sync_value = 0;
    s32 noderank = 0;
    rtStream_t stream[SEND_DEV_NUM];
    sal_thread_t tid[SEND_DEV_NUM];
    p2p_para_t para_info[SEND_DEV_NUM];

    HcclDataType datatype = HCCL_DATA_TYPE_FP32;
    s32 count = P2P_DATA_SIZE_S_HEAVY;
    s32 ndev = SEND_DEV_NUM;

    HcclRootInfo rootInfo;
    ret = hccl::hcclComm::GetUniqueId(&rootInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    const s32 send_node_id = 0;
    const s32 recv_node_id = 1;

    /** 初始化输入输出缓存 */
    for (s32 i = 0; i < ndev; i++ )
    {
        ret = hrtMalloc((void**)&buffer[i], (count * sizeof(float)));
        EXPECT_EQ(ret, HCCL_SUCCESS);
        sal_memset(buffer[i], (count * sizeof(float)), 0, (count * sizeof(float)));
    }

        //sendbuf 赋值

        for (u32 i = 0; i < count; i++)
        {
            buffer[0][i] = 1.0 * i * 10000000;
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
        s32 devid = (i==0) ? 1: 5;
        identify << (devid);
        para_info[i].tag = "default_tag";
        para_info[i].identify = identify.str();

        para_info[i].device_id = devid ;
        para_info[i].ranks_local = ndev;
        para_info[i].count = count;
        para_info[i].datatype = datatype;
        para_info[i].stream = stream[i];
        para_info[i].buffer = buffer[i];
        para_info[i].sync_addr = &sync_value;
        para_info[i].file_name = file_name_t;
        if (i == send_node_id) {
            para_info[i].sender_flag = true;
            para_info[i].peer_rank   = 5;
        } else if (i == recv_node_id){
            para_info[i].sender_flag = false;
            para_info[i].peer_rank   = 1;
            }
    }

    for (s32 i = 0; i < ndev; ++i)
    {
        tid[i] = sal_thread_create("thread", intra_send_receive_task, (void*)&para_info[i]);
        EXPECT_NE(tid[i], (sal_thread_t )NULL);
    }

    for (s32 i = 0; i < ndev; ++i)
    {
        while ( sal_thread_is_running(tid[i]))
        {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
    }

    //获取stream的操作的同步信号量
    for (s32 j = 0; j < ndev; j++)
    {
       rt_ret = aclrtSynchronizeStream(stream[j]);
       EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    /*check result*/
    for (s32 j = 0; j < ndev; j++)
    {
        for (s32 i = 0; i < count; i++)
        {
            float recv = 1.0 * i * 10000000;

            if (abs(buffer[j][i] - recv) > 1e-6)
            {
                HCCL_ERROR("result[%d][%d]:%f \n", j, i, buffer[j][i]);
                errors++;
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

    for (s32 j = 0; j < ndev; j++)
    {
        hrtFree(buffer[j]);
        rt_ret = aclrtDestroyStream(stream[j]);

        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }
    remove(file_name_t);
    GlobalMockObject::verify();
    EXPECT_EQ(errors, 0);
}
#endif

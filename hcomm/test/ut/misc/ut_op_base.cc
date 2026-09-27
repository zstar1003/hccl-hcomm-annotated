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
#include <assert.h>
#include <securec.h>
#include <ifaddrs.h>
#include <sys/socket.h>
#include <netdb.h>

#include <sys/types.h>
#include <stddef.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <hccl/hccl_comm.h>
#include <hccl/hccl_inner.h>
#include <hccl/hccl_ex.h>
#include "hccl_ctrl_plane.h"

#define private public
#define protected public
#include "topoinfo_detect.h"
#include "hccl_impl.h"
#include "hccl_comm_pub.h"
#include "hccl_communicator.h"
#include "coll_batch_send_recv_executor.h"
#include "coll_reduce_scatter_v_executor.h"
#include "coll_all_gather_v_executor.h"
#include "hccl_network.h"
#include "preempt_port_manager.h"
#undef protected
#undef private
#include "profiling_manager.h"
#include "llt_hccl_stub_pub.h"
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include "hccl/base.h"
#include "hccl/hccl_ex.h"
#include <hccl/hccl_comm.h>
#include <hccl/hccl_inner.h>
#include <hccl/hccl_types.h>
#include "topoinfo_ranktableParser_pub.h"
#include "tsd/tsd_client.h"
#include "dltdt_function.h"
#include <unistd.h>
#include "externalinput_pub.h"
#include "v80_rank_table.h"
#include "externalinput.h"
#include "op_base.h"
#include "coll_comm_mgr.h"
#include "param_check_pub.h"
#include "hcom_pub.h"
#include "comm_config_pub.h"
#include "kernel_tiling/kernel_tiling.h"
#include "exception_handler.h"
#include "plugin_runner_pub.h"
#include "task_exception_handler_pub.h"
#include "hccl_group.h"

using namespace std;
using namespace hccl;

class OpbaseTest : public testing::TestWithParam<bool> {
protected:
    // static void SetUpTestCase()
    // {
    //     std::cout << "OpbaseTest SetUP" << std::endl;
    // }
    // static void TearDownTestCase()
    // {
    //     std::cout << "OpbaseTest TearDown" << std::endl;
    // }
    virtual void SetUp()
    {
        static s32 call_cnt = 0;
        DlTdtFunction::GetInstance().DlTdtFunctionInit();
        TsdOpen(1, 2);
        string name = std::to_string(call_cnt++) + "_" + __PRETTY_FUNCTION__;
        ra_set_shm_name(name.c_str());
        ResetInitState();
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        TsdClose(1);
        std::cout << "A Test TearDown" << std::endl;
    }
};

INSTANTIATE_TEST_CASE_P(FFTSSwitch, OpbaseTest, testing::Bool());
#define HCCL_COM_DATA_SIZE 1024
TEST_P(OpbaseTest, ut_hcclBroadcast)
{
    MOCKER(GetExternalInputHcclEnableEntryLog).stubs().with(mockcpp::any()).will(returnValue(true));
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "1"},
           {"para_plane_nic_name", {"eth0"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "1"},
              {"server_num", "1"},
              {"instance_count", "1"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.12"}}}}},
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

    char file_name_t[] = "./st_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;

    // 走1910 4pring
    const char* rank_table_file = "./st_opbase_test.json";
    u32 rank_ID = 0;

    ret = HcclCommInitClusterInfo(rank_table_file, rank_ID, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    bool fftsSwitch = GetParam();
    if (fftsSwitch) {
        SetFftsSwitch(true);
        InitEnvVarParam();
    }
    ret = HcclBroadcastInner(sendbuf, count, HCCL_DATA_TYPE_INT8, 0, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    for (int j = 0; j < count; j++) {
        if (sendbuf[j] != 2) {
            HCCL_ERROR("\n rank:%d sendbuf[%d]:%f", rank, j, sendbuf[j]);
            errors++;
            break;
        }
    }

    sal_free(sendbuf);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    EXPECT_EQ(errors, 0);

    if (fftsSwitch) {
        SetFftsSwitch(false);
        InitEnvVarParam();
    }
}

TEST_F(OpbaseTest, ut_hcclAllReduce)
{
    MOCKER(GetExternalInputHcclEnableEntryLog).stubs().with(mockcpp::any()).will(returnValue(true));
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "1"},
           {"para_plane_nic_name", {"eth0"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "1"},
              {"server_num", "1"},
              {"instance_count", "1"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.12"}}}}},
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

    char file_name_t[] = "./st_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    u32 rankSize = 0;
    u32 rankId = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;

    // 走1910 4pring
    const char* rank_table_file = "./st_opbase_test.json";
    u32 rank_ID = 0;

    ret = HcclCommInitClusterInfo(rank_table_file, rank_ID, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    ret = HcclGetRankSize(comm, &rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(rankSize, 1);

    ret = HcclGetRankId(comm, &rankId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(rankId, 0);

    ret = HcclAllReduceInner(sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclAllReduceInner(sendbuf, sendbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    for (int j = 0; j < count; j++) {
        if (recvbuf[j] != 2) {
            errors++;
            break;
        }
    }

    sal_free(sendbuf);
    sal_free(recvbuf);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    EXPECT_EQ(errors, 0);
}

TEST_F(OpbaseTest, ut_PluginRunner_not_support)
{
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
    int mockModel = 0;
    void* pmockModel = &mockModel;
    MOCKER(aclmdlRICaptureGetInfo)
        .stubs()
        .with(
            mockcpp::any(), outBoundP(&captureStatus, sizeof(captureStatus)),
            outBoundP(&pmockModel, sizeof(pmockModel)))
        .will(returnValue(207000));

    MOCKER(GetExternalInputHcclEnableFfts).stubs().with(mockcpp::any()).will(returnValue(true));

    MOCKER(GetWorkflowMode)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));

    MOCKER(GetExternalInputTaskExceptionSwitch).stubs().will(returnValue(1));

    u32 deviceLogicId = 0;
    TaskExceptionHandler taskExceptionHandler(deviceLogicId);

    HcclResult ret = taskExceptionHandler.Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    PluginRunner pluginRunner(&taskExceptionHandler);

    aclrtStream steam;
    aclrtCreateStream(&steam);

    TaskParaDMA dma;
    TaskParaReduce reduce;
    TaskParaNotify notify;

    pluginRunner(steam, hccl::TaskType::TASK_SDMA, dma);
    pluginRunner(steam, hccl::TaskType::TASK_REDUCE_INLINE, reduce);
    pluginRunner(steam, hccl::TaskType::TASK_NOTIFY_RECORD, notify);
    aclrtDestroyStream(steam);
    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_hcclAllReduce_capture)
{
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
    int mockModel = 0;
    void* pmockModel = &mockModel;
    MOCKER(aclmdlRICaptureGetInfo)
        .stubs()
        .with(
            mockcpp::any(), outBoundP(&captureStatus, sizeof(captureStatus)),
            outBoundP(&pmockModel, sizeof(pmockModel)))
        .will(returnValue(207000));

    MOCKER(GetExternalInputHcclEnableEntryLog).stubs().with(mockcpp::any()).will(returnValue(true));
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "1"},
           {"para_plane_nic_name", {"eth0"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "1"},
              {"server_num", "1"},
              {"instance_count", "1"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.12"}}}}},
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

    char file_name_t[] = "./st_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    u32 rankSize = 0;
    u32 rankId = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;

    // 走1910 4pring
    const char* rank_table_file = "./st_opbase_test.json";
    u32 rank_ID = 0;

    ret = HcclCommInitClusterInfo(rank_table_file, rank_ID, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    ret = HcclGetRankSize(comm, &rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(rankSize, 1);

    ret = HcclGetRankId(comm, &rankId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(rankId, 0);

    ret = HcclAllReduceInner(sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclAllReduceInner(sendbuf, sendbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    for (int j = 0; j < count; j++) {
        if (recvbuf[j] != 2) {
            errors++;
            break;
        }
    }

    sal_free(sendbuf);
    sal_free(recvbuf);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    EXPECT_EQ(errors, 0);
}

TEST_F(OpbaseTest, ut_hcclAllReduce_capture_rdma)
{
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
    int mockModel = 0;
    void* pmockModel = &mockModel;
    MOCKER(aclmdlRICaptureGetInfo)
        .stubs()
        .with(
            mockcpp::any(), outBoundP(&captureStatus, sizeof(captureStatus)),
            outBoundP(&pmockModel, sizeof(pmockModel)))
        .will(returnValue(207000));

    MOCKER(GetExternalInputHcclEnableEntryLog).stubs().with(mockcpp::any()).will(returnValue(true));
    nlohmann::json rank_table = rank_table_910_2server_4rank;

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::AllocAlgResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::ExecOp).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::StreamIsCapture).stubs().with(mockcpp::any()).will(returnValue(true));

    char file_name_t[] = "./st_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    u32 rankSize = 0;
    u32 rankId = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;

    // 走1910 4pring
    const char* rank_table_file = "./st_opbase_test.json";
    u32 rank_ID = 0;

    ret = HcclCommInitClusterInfo(rank_table_file, rank_ID, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    ret = hcclComm->GetRankSize(rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    ret = HcclGetRankId(comm, &rankId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(rankId, 0);

    ret = HcclAllReduceInner(sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclAllReduceInner(sendbuf, sendbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sal_free(sendbuf);
    sal_free(recvbuf);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_hcclReducescatter)
{
    MOCKER(GetExternalInputHcclEnableEntryLog).stubs().with(mockcpp::any()).will(returnValue(true));
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "1"},
           {"para_plane_nic_name", {"eth0"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "1"},
              {"server_num", "1"},
              {"instance_count", "1"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.12"}}}}},
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

    char file_name_t[] = "./st_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;

    // 走1910 4pring
    const char* rank_table_file = "./st_opbase_test.json";
    u32 rank_ID = 0;

    ret = HcclCommInitClusterInfo(rank_table_file, rank_ID, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    ret = HcclReduceScatterInner(sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclReduceScatterInner(sendbuf, sendbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    for (int j = 0; j < count; j++) {
        if (recvbuf[j] != 2) {
            errors++;
            break;
        }
    }

    sal_free(sendbuf);
    sal_free(recvbuf);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    EXPECT_EQ(errors, 0);
}

TEST_F(OpbaseTest, ut_hcclReducescatter_capture)
{
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
    int mockModel = 0;
    void* pmockModel = &mockModel;
    MOCKER(aclmdlRICaptureGetInfo)
        .stubs()
        .with(
            mockcpp::any(), outBoundP(&captureStatus, sizeof(captureStatus)),
            outBoundP(&pmockModel, sizeof(pmockModel)))
        .will(returnValue(0));

    MOCKER(GetExternalInputHcclEnableEntryLog).stubs().with(mockcpp::any()).will(returnValue(true));
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "1"},
           {"para_plane_nic_name", {"eth0"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "1"},
              {"server_num", "1"},
              {"instance_count", "1"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.12"}}}}},
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

    char file_name_t[] = "./st_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;

    // 走1910 4pring
    const char* rank_table_file = "./st_opbase_test.json";
    u32 rank_ID = 0;

    ret = HcclCommInitClusterInfo(rank_table_file, rank_ID, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    ret = HcclReduceScatterInner(sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclReduceScatterInner(sendbuf, sendbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    for (int j = 0; j < count; j++) {
        if (recvbuf[j] != 2) {
            errors++;
            break;
        }
    }

    sal_free(sendbuf);
    sal_free(recvbuf);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    EXPECT_EQ(errors, 0);
}

TEST_F(OpbaseTest, ut_hcclScatter_Ring_1rank)
{
    MOCKER(GetExternalInputHcclEnableEntryLog).stubs().with(mockcpp::any()).will(returnValue(true));
    nlohmann::json rank_table = rank_table_910_1server_1rank;

    char file_name_t[] = "./st_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;

    // 走1910 4pring
    const char* rank_table_file = "./st_opbase_test.json";
    u32 rank_ID = 0;

    ret = HcclCommInitClusterInfo(rank_table_file, rank_ID, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    ret = HcclScatterInner(sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, 0, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    for (int j = 0; j < count; j++) {
        if (recvbuf[j] != 2) {
            errors++;
            break;
        }
    }

    sal_free(sendbuf);
    sal_free(recvbuf);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    EXPECT_EQ(errors, 0);
    GlobalMockObject::verify();
}

TEST_P(OpbaseTest, ut_hcclScatter_inptr_EQ_outPtr)
{
    bool fftsSwitch = GetParam();
    if (fftsSwitch) {
        SetFftsSwitch(true);
        SetFftsSwitch(true);
    }

    nlohmann::json rank_table = rank_table_910_1server_1rank;

    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;

    const char* rank_table_file = "./ut_opbase_test.json";
    u32 rank_ID = 0;

    ret = HcclCommInitClusterInfo(rank_table_file, rank_ID, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf = sendbuf;

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    ret = HcclScatterInner(sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, 0, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    for (int j = 0; j < count; j++) {
        if (recvbuf[j] != 2) {
            errors++;
            break;
        }
    }

    sal_free(sendbuf);
    recvbuf = nullptr;

    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    EXPECT_EQ(errors, 0);
    if (fftsSwitch) {
        SetFftsSwitch(true);
    }
    GlobalMockObject::verify();
}

TEST_P(OpbaseTest, ut_HcclReduce_inptr_EQ_outPtr)
{
    MOCKER(GetExternalInputHcclEnableEntryLog).stubs().with(mockcpp::any()).will(returnValue(true));
    bool fftsSwitch = GetParam();
    if (fftsSwitch) {
        SetFftsSwitch(true);
    }

    nlohmann::json rank_table = rank_table_910_1server_1rank;

    char file_name_t[] = "./st_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;

    const char* rank_table_file = "./st_opbase_test.json";
    u32 rank_ID = 0;
    string tmpOptions = "";
    HcomSetProfilingMode(HcomProfilingMode::PROFILING_OPEN, tmpOptions.c_str());
    ret = HcclCommInitClusterInfo(rank_table_file, rank_ID, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf = sendbuf;

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }
    unsigned int rankSize = 0;
    ret = HcclGetRankSize(comm, &rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HCCL_INFO("HCCL TEST get rank size[%u] success.", rankSize);

    unsigned int rankId = 0;
    ret = HcclGetRankId(comm, &rankId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HCCL_INFO("HCCL TEST get rank id[%u] success.", rankId);

    ret = HcclBarrier(comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclReduceInner(sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, 0, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    HCCL_ERROR("count %d", count);
    for (int j = 0; j < count; j++) {
        if (recvbuf[j] != 2) {
            HCCL_ERROR("j %d", j);
            errors++;
            break;
        }
    }

    sal_free(sendbuf);

    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    EXPECT_EQ(errors, 0);
    if (fftsSwitch) {
        SetFftsSwitch(true);
    }
}

TEST_F(OpbaseTest, ut_HcclAllGatherOutPlace310P_ranksize_1)
{
    MOCKER(hrtRaGetInterfaceVersion).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));
    MOCKER(GetExternalInputHcclEnableEntryLog).stubs().with(mockcpp::any()).will(returnValue(true));
    s8* sendBuf;
    s8* recvBuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    HcclRootInfo id;
    char group[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {0};
    void* commContext = nullptr;
    void* aicpuNotify = nullptr;
    rtStream_t stream;

    DevType deviceType = DevType::DEV_TYPE_310P3;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = HcclGetRootInfo(&id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclComm newcomm;
    ret = HcclCommInitRootInfo(1, &id, 0, &newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sendBuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendBuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvBuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvBuf, count * sizeof(s8), 0, count * sizeof(s8));

    HcclCommunicator impl;
    impl.AtomicInitSet();
    HcclCommParams params;
    string commId = "AllGather310p";
    memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1);
    params.rank = 0;
    params.totalRanks = 1;
    params.isHeterogComm = false;
    params.logicDevId = 0;
    params.deviceType = DevType::DEV_TYPE_310P3;

    RankTable_t rankTable;
    rankTable.collectiveId = "192.168.10.101-8000-8001";
    vector<RankInfo_t> rankVec(1);
    rankVec[0].rankId = 0;
    rankVec[0].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr1(1695197376);                 // 1,695,197,376
    rankVec[0].deviceInfo.deviceIp.push_back(ipAddr1); // 101.10.168.192
    rankVec[0].serverIdx = 0;
    rankVec[0].serverId = "192.168.0.101";
    rankTable.rankList.assign(rankVec.begin(), rankVec.end());
    rankTable.deviceNum = 1;
    rankTable.serverNum = 1;
    aclrtSetDevice(0);

    ret = impl.Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    impl.userRankSize_ = 1;

    ret = impl.AllGatherOutPlace(commId, sendBuf, recvBuf, count, HCCL_DATA_TYPE_INT8, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    aclrtSynchronizeStream(stream);

    ret = HcclCommDestroy(newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    aclrtDestroyStream(stream);
    sal_free(sendBuf);
    sal_free(recvBuf);
    impl.ReleaseCommCCLbuffer();
    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_HcclAllGatherOutPlace_mix_ranksize_1_capture)
{
    setenv("HCCL_OP_EXPANSION_MODE", "HOST", 1);
    ResetInitState();
    InitExternalInput();
    SetFftsSwitch(false);

    DevType deviceType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCallbackTask::CallbackRegStream).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    s32 portNum = -1;
    MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::RegisterToHeartBeat, HcclResult(HcclCommunicator::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    HcclResult ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendBuf;
    s8* recvBuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;
    s32 ndev = 8;

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sendBuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendBuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvBuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvBuf, count * sizeof(s8), 0, count * sizeof(s8));

    HcclComm newcomm;
    HcclRootInfo id;
    ret = HcclGetRootInfo(&id);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclCommInitRootInfo(1, &id, 0, &newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::IsAtomicInit).stubs().will(returnValue(true));
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
    int mockModel = 0;
    void* pmockModel = &mockModel;
    MOCKER(aclmdlRICaptureGetInfo)
        .stubs()
        .with(
            mockcpp::any(), outBoundP(&captureStatus, sizeof(captureStatus)),
            outBoundP(&pmockModel, sizeof(pmockModel)))
        .will(returnValue(0));

    HcclCommunicator impl;
    HcclCommParams params;
    string commId = "AllGatherMixOpbase";
    memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1);
    params.rank = 0;
    params.totalRanks = 1;
    params.isHeterogComm = false;
    params.logicDevId = 0;
    params.deviceType = DevType::DEV_TYPE_910_93;

    RankTable_t rankTable;
    rankTable.collectiveId = "192.168.10.101-8000-8001";
    vector<RankInfo_t> rankVec(1);
    rankVec[0].rankId = 0;
    rankVec[0].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr1(1695197376);                 // 1,695,197,376
    rankVec[0].deviceInfo.deviceIp.push_back(ipAddr1); // 101.10.168.192
    rankVec[0].serverIdx = 0;
    rankVec[0].serverId = "192.168.0.101";
    rankTable.rankList.assign(rankVec.begin(), rankVec.end());
    rankTable.deviceNum = 1;
    rankTable.serverNum = 1;
    aclrtSetDevice(0);

    ret = impl.Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    impl.userRankSize_ = 1;

    ret = impl.AllGatherOutPlace(commId, sendBuf, recvBuf, count, HCCL_DATA_TYPE_INT8, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    sal_free(sendBuf);
    sal_free(recvBuf);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    GlobalMockObject::verify();

    unsetenv("HCCL_OP_EXPANSION_MODE");
    ResetInitState();
    InitExternalInput();
}

TEST_F(OpbaseTest, ut_hcclAllGather)
{
    MOCKER(GetExternalInputHcclEnableEntryLog).stubs().with(mockcpp::any()).will(returnValue(true));
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "1"},
           {"para_plane_nic_name", {"eth0"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "1"},
              {"server_num", "1"},
              {"instance_count", "1"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.12"}}}}},
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

    char file_name_t[] = "./st_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;

    // 走1910 4pring
    const char* rank_table_file = "./st_opbase_test.json";
    u32 rank_ID = 0;

    ret = HcclCommInitClusterInfo(rank_table_file, rank_ID, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    ret = HcclAllGatherInner(sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclAllGatherInner(sendbuf, sendbuf, count, HCCL_DATA_TYPE_INT8, comm, stream);
    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    for (int j = 0; j < count; j++) {
        if (recvbuf[j] != 2) {
            errors++;
            break;
        }
    }

    sal_free(sendbuf);
    sal_free(recvbuf);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    EXPECT_EQ(errors, 0);
}

TEST_F(OpbaseTest, ut_hcclAllGather_capture)
{
    MOCKER(GetExternalInputHcclEnableEntryLog).stubs().with(mockcpp::any()).will(returnValue(true));
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
    int mockModel = 0;
    void* pmockModel = &mockModel;
    MOCKER(aclmdlRICaptureGetInfo)
        .stubs()
        .with(
            mockcpp::any(), outBoundP(&captureStatus, sizeof(captureStatus)),
            outBoundP(&pmockModel, sizeof(pmockModel)))
        .will(returnValue(207000));

    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "1"},
           {"para_plane_nic_name", {"eth0"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "1"},
              {"server_num", "1"},
              {"instance_count", "1"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.12"}}}}},
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

    char file_name_t[] = "./st_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;

    // 走1910 4pring
    const char* rank_table_file = "./st_opbase_test.json";
    u32 rank_ID = 0;

    ret = HcclCommInitClusterInfo(rank_table_file, rank_ID, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    ret = HcclAllGatherInner(sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclAllGatherInner(sendbuf, sendbuf, count, HCCL_DATA_TYPE_INT8, comm, stream);
    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    for (int j = 0; j < count; j++) {
        if (recvbuf[j] != 2) {
            errors++;
            break;
        }
    }

    sal_free(sendbuf);
    sal_free(recvbuf);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    EXPECT_EQ(errors, 0);
}

TEST_F(OpbaseTest, ut_hcclAllGather_capture_rdma)
{
    MOCKER(GetExternalInputHcclEnableEntryLog).stubs().with(mockcpp::any()).will(returnValue(true));

    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
    int mockModel = 0;
    void* pmockModel = &mockModel;
    MOCKER(aclmdlRICaptureGetInfo)
        .stubs()
        .with(
            mockcpp::any(), outBoundP(&captureStatus, sizeof(captureStatus)),
            outBoundP(&pmockModel, sizeof(pmockModel)))
        .will(returnValue(207000));

    MOCKER_CPP(&HcclCommunicator::StreamIsCapture).stubs().with(mockcpp::any()).will(returnValue(true));

    DevType deviceType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::ExecOp).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    nlohmann::json rank_table = rank_table_910_2server_4rank;

    char file_name_t[] = "./st_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    u32 rankSize = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;

    // 走1910 4pring
    const char* rank_table_file = "./st_opbase_test.json";
    u32 rank_ID = 0;

    ret = HcclCommInitClusterInfo(rank_table_file, rank_ID, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    ret = hcclComm->GetRankSize(rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf = (s8*)sal_malloc(rankSize * count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    ret = HcclAllGatherInner(sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclAllGatherInner(sendbuf, sendbuf, count, HCCL_DATA_TYPE_INT8, comm, stream);
    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sal_free(sendbuf);
    sal_free(recvbuf);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_hcclAllGatherV)
{
    MOCKER(GetExternalInputHcclEnableEntryLog).stubs().with(mockcpp::any()).will(returnValue(true));

    DevType deviceType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::AllocAlgResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::ExecOp).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "2"},
           {"para_plane_nic_name", {"eth0", "eth1"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "2"},
              {"server_num", "1"},
              {"instance_count", "2"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.12"}}}}},

                   {{"rank_id", "1"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.14"}}}}},
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

    char file_name_t[] = "./st_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s8* recvbuf;
    u64* recvCounts;
    u64* recvDispls;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    u32 rankSize = 0;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;

    // 走1910 4pring
    const char* rank_table_file = "./st_opbase_test.json";
    u32 rank_ID = 0;

    ret = HcclCommInitClusterInfo(rank_table_file, rank_ID, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    ret = hcclComm->GetRankSize(rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    recvCounts = (u64*)sal_malloc(rankSize * sizeof(u64));
    sal_memset(recvCounts, rankSize * sizeof(u64), 0, rankSize * sizeof(u64));
    recvDispls = (u64*)sal_malloc(rankSize * sizeof(u64));
    sal_memset(recvDispls, rankSize * sizeof(u64), 0, rankSize * sizeof(u64));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    for (int i = 0; i < rankSize; i++) {
        recvCounts[i] = count;
        if (i > 0) {
            recvDispls[i] = recvDispls[i - 1] + recvCounts[i - 1];
        }
    }

    ret = HcclAllGatherVInner(sendbuf, count, recvbuf, recvCounts, recvDispls, HCCL_DATA_TYPE_INT8, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclAllGatherVInner(sendbuf, count, recvbuf, recvCounts, recvDispls, HCCL_DATA_TYPE_INT8, comm, stream);
    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sal_free(sendbuf);
    sal_free(recvbuf);
    sal_free(recvCounts);
    sal_free(recvDispls);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_hcomAllGatherV)
{
    MOCKER(GetExternalInputHcclEnableEntryLog).stubs().with(mockcpp::any()).will(returnValue(true));

    DevType deviceType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::ExecOp).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    nlohmann::json rank_table = rank_table_910_1server_2rank;

    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    setenv("HCCL_DEBUG_CONFIG", "alg", 1);

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s8* recvbuf;
    u64* recvCounts;
    u64* recvDispls;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    u32 rankSize = 4;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;

    // 走1910 4pring
    const char* rank_table_file = "./ut_opbase_test.json";
    u32 rank_ID = 0;

    ret = HcclCommInitClusterInfo(rank_table_file, rank_ID, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf = (s8*)sal_malloc(rankSize * count * sizeof(s8));
    sal_memset(recvbuf, rankSize * count * sizeof(s8), 0, rankSize * count * sizeof(s8));

    recvCounts = (u64*)sal_malloc(rankSize * sizeof(u64));
    sal_memset(recvCounts, rankSize * sizeof(u64), 0, rankSize * sizeof(u64));
    recvDispls = (u64*)sal_malloc(rankSize * sizeof(u64));
    sal_memset(recvDispls, rankSize * sizeof(u64), 0, rankSize * sizeof(u64));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    for (int i = 0; i < rankSize; i++) {
        recvCounts[i] = count;
        if (i > 0) {
            recvDispls[i] = recvDispls[i - 1] + recvCounts[i - 1];
        }
    }

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    string strTag = "allgatherv";

    ret = hcclComm->AllGatherV(
        strTag, static_cast<void*>(sendbuf), 2, static_cast<void*>(recvbuf), static_cast<void*>(recvCounts),
        static_cast<void*>(recvDispls), HCCL_DATA_TYPE_INT8, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sal_free(sendbuf);
    sal_free(recvbuf);
    sal_free(recvCounts);
    sal_free(recvDispls);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    unsetenv("HCCL_DEBUG_CONFIG");

    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_HcclReduceScatterV)
{
    MOCKER(GetExternalInputHcclEnableEntryLog).stubs().with(mockcpp::any()).will(returnValue(true));

    DevType deviceType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::AllocAlgResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::ExecOp).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "2"},
           {"para_plane_nic_name", {"eth0", "eth1"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "2"},
              {"server_num", "1"},
              {"instance_count", "2"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.12"}}}}},

                   {{"rank_id", "1"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.14"}}}}},
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

    char file_name_t[] = "./st_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s8* recvbuf;
    u64* sendCounts;
    u64* sendDispls;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    u32 rankSize = 0;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;

    // 走1910 4pring
    const char* rank_table_file = "./st_opbase_test.json";
    u32 rank_ID = 0;

    ret = HcclCommInitClusterInfo(rank_table_file, rank_ID, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    ret = hcclComm->GetRankSize(rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sendbuf = (s8*)sal_malloc(rankSize * count * sizeof(s8));
    sal_memset(sendbuf, rankSize * count * sizeof(s8), 0, rankSize * count * sizeof(s8));
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    sendCounts = (u64*)sal_malloc(rankSize * sizeof(u64));
    sal_memset(sendCounts, rankSize * sizeof(u64), 0, rankSize * sizeof(u64));
    sendDispls = (u64*)sal_malloc(rankSize * sizeof(u64));
    sal_memset(sendDispls, rankSize * sizeof(u64), 0, rankSize * sizeof(u64));

    for (int j = 0; j < rankSize * count; j++) {
        sendbuf[j] = 2;
    }

    for (int i = 0; i < rankSize; i++) {
        sendCounts[i] = count;
        if (i > 0) {
            sendDispls[i] = sendDispls[i - 1] + sendCounts[i - 1];
        }
    }

    ret = HcclReduceScatterVInner(
        sendbuf, sendCounts, sendDispls, recvbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclReduceScatterVInner(
        sendbuf, sendCounts, sendDispls, recvbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sal_free(sendbuf);
    sal_free(recvbuf);
    sal_free(sendCounts);
    sal_free(sendDispls);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_hcomReduceScatterV)
{
    MOCKER(GetExternalInputHcclEnableEntryLog).stubs().with(mockcpp::any()).will(returnValue(true));

    DevType deviceType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::ExecOp).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    nlohmann::json rank_table = rank_table_910_1server_2rank;

    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s8* recvbuf;
    u64* sendCounts;
    u64* sendDispls;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    u32 rankSize = 4;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;

    // 走1910 4pring
    const char* rank_table_file = "./ut_opbase_test.json";
    u32 rank_ID = 0;

    ret = HcclCommInitClusterInfo(rank_table_file, rank_ID, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));
    sendbuf = (s8*)sal_malloc(rankSize * count * sizeof(s8));
    sal_memset(sendbuf, rankSize * count * sizeof(s8), 0, rankSize * count * sizeof(s8));

    sendCounts = (u64*)sal_malloc(rankSize * sizeof(u64));
    sal_memset(sendCounts, rankSize * sizeof(u64), 0, rankSize * sizeof(u64));
    sendDispls = (u64*)sal_malloc(rankSize * sizeof(u64));
    sal_memset(sendDispls, rankSize * sizeof(u64), 0, rankSize * sizeof(u64));

    for (int j = 0; j < count; j++) {
        recvbuf[j] = 2;
    }

    for (int i = 0; i < rankSize; i++) {
        sendCounts[i] = count;
        if (i > 0) {
            sendDispls[i] = sendDispls[i - 1] + sendDispls[i - 1];
        }
    }

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    string strTag = "reducescatterv";
    ret = hcclComm->ReduceScatterV(
        strTag, static_cast<void*>(sendbuf), static_cast<void*>(sendCounts), static_cast<void*>(sendDispls),
        static_cast<void*>(recvbuf), 2, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sal_free(sendbuf);
    sal_free(recvbuf);
    sal_free(sendCounts);
    sal_free(sendDispls);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_hcclAllGatherVFor310P3)
{
    MOCKER(GetExternalInputHcclEnableEntryLog).stubs().with(mockcpp::any()).will(returnValue(true));

    DevType deviceType = DevType::DEV_TYPE_310P3;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::AllocAlgResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::ExecOp).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "2"},
           {"para_plane_nic_name", {"eth0", "eth1"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "2"},
              {"server_num", "1"},
              {"instance_count", "2"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.12"}}}}},

                   {{"rank_id", "1"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.14"}}}}},
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

    char file_name_t[] = "./st_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s8* recvbuf;
    u64* recvCounts;
    u64* recvDispls;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    u32 rankSize = 0;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;

    // 走1910 4pring
    const char* rank_table_file = "./st_opbase_test.json";
    u32 rank_ID = 0;

    ret = HcclCommInitClusterInfo(rank_table_file, rank_ID, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    ret = hcclComm->GetRankSize(rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    recvCounts = (u64*)sal_malloc(rankSize * sizeof(u64));
    sal_memset(recvCounts, rankSize * sizeof(u64), 0, rankSize * sizeof(u64));
    recvDispls = (u64*)sal_malloc(rankSize * sizeof(u64));
    sal_memset(recvDispls, rankSize * sizeof(u64), 0, rankSize * sizeof(u64));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    for (int i = 0; i < rankSize; i++) {
        recvCounts[i] = count;
        if (i > 0) {
            recvDispls[i] = recvDispls[i - 1] + recvCounts[i - 1];
        }
    }

    ret = HcclAllGatherVInner(sendbuf, count, recvbuf, recvCounts, recvDispls, HCCL_DATA_TYPE_INT8, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclAllGatherVInner(sendbuf, count, recvbuf, recvCounts, recvDispls, HCCL_DATA_TYPE_INT8, comm, stream);
    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sal_free(sendbuf);
    sal_free(recvbuf);
    sal_free(recvCounts);
    sal_free(recvDispls);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_HcclReduceScatterVFor310P3)
{
    MOCKER(GetExternalInputHcclEnableEntryLog).stubs().with(mockcpp::any()).will(returnValue(true));

    DevType deviceType = DevType::DEV_TYPE_310P3;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::AllocAlgResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::ExecOp).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "2"},
           {"para_plane_nic_name", {"eth0", "eth1"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "2"},
              {"server_num", "1"},
              {"instance_count", "2"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.12"}}}}},

                   {{"rank_id", "1"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.14"}}}}},
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

    char file_name_t[] = "./st_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s8* recvbuf;
    u64* sendCounts;
    u64* sendDispls;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    u32 rankSize = 0;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;

    // 走1910 4pring
    const char* rank_table_file = "./st_opbase_test.json";
    u32 rank_ID = 0;

    ret = HcclCommInitClusterInfo(rank_table_file, rank_ID, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    ret = hcclComm->GetRankSize(rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sendbuf = (s8*)sal_malloc(rankSize * count * sizeof(s8));
    sal_memset(sendbuf, rankSize * count * sizeof(s8), 0, rankSize * count * sizeof(s8));
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    sendCounts = (u64*)sal_malloc(rankSize * sizeof(u64));
    sal_memset(sendCounts, rankSize * sizeof(u64), 0, rankSize * sizeof(u64));
    sendDispls = (u64*)sal_malloc(rankSize * sizeof(u64));
    sal_memset(sendDispls, rankSize * sizeof(u64), 0, rankSize * sizeof(u64));

    for (int j = 0; j < rankSize * count; j++) {
        sendbuf[j] = 2;
    }

    for (int i = 0; i < rankSize; i++) {
        sendCounts[i] = count;
        if (i > 0) {
            sendDispls[i] = sendDispls[i - 1] + sendCounts[i - 1];
        }
    }

    ret = HcclReduceScatterVInner(
        sendbuf, sendCounts, sendDispls, recvbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclReduceScatterVInner(
        sendbuf, sendCounts, sendDispls, recvbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sal_free(sendbuf);
    sal_free(recvbuf);
    sal_free(sendCounts);
    sal_free(sendDispls);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    GlobalMockObject::verify();
}

#if 1
#define HCCL_COM_BIG_DATA_SIZE (300 * 1024 * 1024) // 300M
TEST_F(OpbaseTest, ut_BighcclAllReduce)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "1"},
           {"para_plane_nic_name", {"eth0"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "1"},
              {"server_num", "1"},
              {"instance_count", "1"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.12"}}}}},
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

    char file_name_t[] = "./st_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_BIG_DATA_SIZE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;

    // 走1910 4pring
    const char* rank_table_file = "./st_opbase_test.json";
    u32 rank_ID = 0;

    ret = HcclCommInitClusterInfo(rank_table_file, rank_ID, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    ret = HcclAllReduceInner(sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    for (int j = 0; j < count; j++) {
        if (recvbuf[j] != 2) {
            errors++;
            break;
        }
    }

    sal_free(sendbuf);
    sal_free(recvbuf);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    EXPECT_EQ(errors, 0);
}

TEST_F(OpbaseTest, ut_BighcclReducescatter)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "1"},
           {"para_plane_nic_name", {"eth0"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "1"},
              {"server_num", "1"},
              {"instance_count", "1"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.12"}}}}},
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

    char file_name_t[] = "./st_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_BIG_DATA_SIZE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;

    // 走1910 4pring
    const char* rank_table_file = "./st_opbase_test.json";
    u32 rank_ID = 0;

    ret = HcclCommInitClusterInfo(rank_table_file, rank_ID, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    ret = HcclReduceScatterInner(sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    for (int j = 0; j < count; j++) {
        if (recvbuf[j] != 2) {
            errors++;
            break;
        }
    }

    sal_free(sendbuf);
    sal_free(recvbuf);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    EXPECT_EQ(errors, 0);
}

TEST_F(OpbaseTest, ut_BighcclAllGather)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "1"},
           {"para_plane_nic_name", {"eth0"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "1"},
              {"server_num", "1"},
              {"instance_count", "1"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.12"}}}}},
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

    char file_name_t[] = "./st_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_BIG_DATA_SIZE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;

    // 走1910 4pring
    const char* rank_table_file = "./st_opbase_test.json";
    u32 rank_ID = 0;

    ret = HcclCommInitClusterInfo(rank_table_file, rank_ID, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    ret = HcclAllGatherInner(sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    for (int j = 0; j < count; j++) {
        if (recvbuf[j] != 2) {
            errors++;
            break;
        }
    }

    sal_free(sendbuf);
    sal_free(recvbuf);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    EXPECT_EQ(errors, 0);
}

TEST_F(OpbaseTest, ut_BIGhcclBroadcast)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "1"},
           {"para_plane_nic_name", {"eth0"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "1"},
              {"server_num", "1"},
              {"instance_count", "1"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.12"}}}}},
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

    char file_name_t[] = "./st_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_BIG_DATA_SIZE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;

    // 走1910 4pring
    const char* rank_table_file = "./st_opbase_test.json";
    u32 rank_ID = 0;

    ret = HcclCommInitClusterInfo(rank_table_file, rank_ID, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    ret = HcclBroadcastInner(sendbuf, count, HCCL_DATA_TYPE_INT8, 0, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    for (int j = 0; j < count; j++) {
        if (sendbuf[j] != 2) {
            HCCL_ERROR("\n rank:%d sendbuf[%d]:%f", rank, j, sendbuf[j]);
            errors++;
            break;
        }
    }

    sal_free(sendbuf);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    EXPECT_EQ(errors, 0);
}

#endif

TEST_F(OpbaseTest, ut_multi_hcclAllReduce)
{
    nlohmann::json rank_table
        = {{"status", "completed"},
           {"deploy_mode", "lab"},
           {"group_count", "1"},
           {"chip_info", "910"},
           {"board_id", "0x0000"},
           {"para_plane_nic_location", "device"},
           {"para_plane_nic_num", "1"},
           {"para_plane_nic_name", {"eth0"}},
           {"group_list",
            {{{"group_name", ""},
              {"device_num", "1"},
              {"server_num", "1"},
              {"instance_count", "1"},
              {"instance_list",
               {
                   {{"rank_id", "0"},
                    {"server_id", "10.0.0.10"},
                    {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.12"}}}}},
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

    char file_name_t[] = "./st_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;

    // 走1910 4pring
    const char* rank_table_file = "./st_opbase_test.json";
    u32 rank_ID = 0;

    ret = HcclCommInitClusterInfo(rank_table_file, rank_ID, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }
    auto& profilingManager = hccl::ProfilingManager::Instance();
    profilingManager.StartFftsLaunchSubscribe();
    profilingManager.StartHostApiSubscribe();
    profilingManager.StartTaskApiSubscribe();
    profilingManager.StartHostHcclOpSubscribe();
    ret = HcclBarrier(comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclAllReduceInner(sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    ret = HcclAllReduceInner(sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    ret = HcclAllReduceInner(sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    ret = HcclAllReduceInner(sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    for (int j = 0; j < count; j++) {
        if (recvbuf[j] != 2) {
            errors++;
            break;
        }
    }

    sal_free(sendbuf);
    sal_free(recvbuf);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    EXPECT_EQ(errors, 0);
}
#if 0
TEST_F(OpbaseTest, ut_hcclCommInitRank_single_server_1p_success_security)
{
    nlohmann::json whitelist =
    {
        {
            "host_ip",
            {
                "127.0.0.1",
            }
        }
    };

    HcclResult ret;
    hrtSetDevice(0);
    char *buffer;

    MOCKER(hrtRaGetInterfaceVersion)
    .expects(atMost(1))
    .will(returnValue(HCCL_SUCCESS));
    //也可以将buffer作为输出参数
    if((buffer = getcwd(NULL, 0)) != NULL)
    {
        std::string dirPath = buffer;
        std::string fileName = "/ut_hccl_host_allowlist";
        std::string realPath = dirPath.c_str() + fileName;
        ResetInitState();
        setenv("HCCL_WHITELIST_FILE", realPath.c_str(), 1);
        ofstream outfile(realPath);
        outfile << std::setw(4) << whitelist << std::endl;
        outfile.close();
        NetworkManager::GetInstance(0).hostNicInitRef_.Clear();
        HcclRootInfo id;
        ret = HcclGetRootInfo(&id);
        EXPECT_EQ(ret, HCCL_SUCCESS);

        HcclComm newcomm;
        ret = HcclCommInitRootInfo(1, &id, 0, &newcomm);
        EXPECT_EQ(ret, HCCL_SUCCESS);

        ret = HcclCommDestroy(newcomm);
        EXPECT_EQ(ret, HCCL_SUCCESS);

        unsetenv("HCCL_WHITELIST_FILE");
        remove(realPath.c_str());
    }
    free(buffer);
}

TEST_F(OpbaseTest, ut_HcclGetCommName_1)
{
    char *group;
    HcclRootInfo id;
    HcclResult ret = HcclGetRootInfo(&id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclComm newcomm;
    ret = HcclCommInitRootInfo(1, &id, 0, &newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclGetCommName(newcomm, group);
    EXPECT_EQ(ret, HCCL_E_PTR);
    ret = HcclCommDestroy(newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTest, ut_HcclGetCommName_2)
{
    char *group;
    HcclComm newcomm;
    HcclResult ret = HcclGetCommName(&newcomm, group);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(OpbaseTest, ut_hcclGetRootInfo_single_server_1p_success_normal)
{
    HcclResult ret;
    hrtSetDevice(0);

    setenv("HCCL_WHITELIST_DISABLE", "1", 1);
    setenv("HCCL_IF_IP", "127.0.0.1", 1);

    HcclRootInfo id;
    ret = HcclGetRootInfo(&id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclComm newcomm;
    ret = HcclCommInitRootInfo(1, &id, 0, &newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclCommDestroy(newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    unsetenv("HCCL_WHITELIST_DISABLE");
    unsetenv("HCCL_IF_IP");
}
TEST_F(OpbaseTest, ut_hcclsend_hcclrecv)
{
    MOCKER(GetExternalInputHcclEnableEntryLog)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(true));
    nlohmann::json rank_table = rank_table_1server_8rank;

    char file_name_t[] = "./st_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }
    else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;
    s32 ndev = 8;
    // 走1910 8pring
    const char* rank_table_file = "./st_opbase_test.json";

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sendbuf= (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf= (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));
    ret = HcclCommInitClusterInfo(rank_table_file, 0, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    for (int j = 0; j < count; j++)
    {
        sendbuf[j] = 2;
    }
    MOCKER_CPP(&hcclComm::SendOutPlace)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(0));
    ret = HcclSendInner(sendbuf, count, HCCL_DATA_TYPE_INT8, 1, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER_CPP(&hcclComm::ReceiveOutPlace)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(0));
    ret = HcclRecvInner(recvbuf, count, HCCL_DATA_TYPE_INT8, 0, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sal_free(sendbuf);
    sal_free(recvbuf);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
}

TEST_F(OpbaseTest, ut_HcclCommInitClusterInfo)
{
    setenv("HCCL_WHITELIST_DISABLE", "1", 1);

    MOCKER(GetExternalInputHcclLinkTimeOut)
    .stubs()
    .will(returnValue(0));

    HcclRootInfo id;
    HcclResult ret = HcclGetRootInfo(&id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclComm newcomm;
    MOCKER(InitExternalInput)
    .expects(atMost(10))
    .will(returnValue(1));
    ret = HcclCommInitRootInfo(1, &id, 0, &newcomm);
    EXPECT_EQ(ret, 1);
    unsetenv("HCCL_WHITELIST_DISABLE");
    SalSleep(1);
    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_HcclGroupCommInitClusterInfo)
{
    MOCKER(GetExternalInputHcclEnableEntryLog)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(true));
    nlohmann::json rank_table = rank_table_1server_8rank;

    char file_name_t[] = "./st_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }
    else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;

    // 走1910 8pring
    const char* rank_table_file = "./st_opbase_test.json";

    ret = HcclGroupStart();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclCommInitClusterInfo(rank_table_file, 0, comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclGroupEnd();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTest, ut_check_alltoallv_external_mem)
{
    constexpr s32 rankSize = 2;
    u64 sendCounts[rankSize] = {10, INVALID_U64};
    u64 recvCounts[rankSize] = {10, INVALID_U64};
    void *addr = static_cast<void *>(sendCounts);
    HcclResult ret = HcomCheckAlltoAllVExternalMem(addr, sendCounts, addr, recvCounts, rankSize);
    EXPECT_EQ(ret, HCCL_E_PARA);

    sendCounts[1] = 10;
    ret = HcomCheckAlltoAllVExternalMem(addr, sendCounts, addr, recvCounts, rankSize);
    EXPECT_EQ(ret, HCCL_E_PARA);
}
extern u64 HcclGetLookupUpdateWorkspace(s32 count, s32 valueDim, HcclDataType keyType, HcclDataType valueType, s32 flags);

TEST_F(OpbaseTest, ut_HcclConfig_deterministic)
{
    union HcclConfigValue hcclConfigValue;
    hcclConfigValue.value = 1;
    HcclResult ret = HcclSetConfig(HCCL_DETERMINISTIC, hcclConfigValue);
    union HcclConfigValue hcclConfigValueRet;
    ret = HcclGetConfig(HCCL_DETERMINISTIC, &hcclConfigValueRet);
    EXPECT_EQ(hcclConfigValue.value, hcclConfigValueRet.value);

    hcclConfigValue.value = 0;
    ret = HcclSetConfig(HCCL_DETERMINISTIC, hcclConfigValue);
    ret = HcclGetConfig(HCCL_DETERMINISTIC, &hcclConfigValueRet);
    EXPECT_EQ(hcclConfigValue.value, hcclConfigValueRet.value);

    DevType deviceType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));
    hcclConfigValue.value = 2;
    ret = HcclSetConfig(HCCL_DETERMINISTIC, hcclConfigValue);
    ret = HcclGetConfig(HCCL_DETERMINISTIC, &hcclConfigValueRet);
    EXPECT_EQ(hcclConfigValue.value, hcclConfigValueRet.value);
    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_HcclConfig_deterministic_fail)
{
    union HcclConfigValue hcclConfigValue;
    DevType deviceType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));
    hcclConfigValue.value = 2;
    HcclResult ret = HcclSetConfig(HCCL_DETERMINISTIC, hcclConfigValue);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_HcclGetLookupUpdateWorkspace)
{
    s32 keyMaxNum = 10;
    s32 valueDim = 1;
    HcclDataType keyType = HCCL_DATA_TYPE_UINT64;
    HcclDataType valueType = HCCL_DATA_TYPE_FP32;
    s32 flags = 1;

    HcclGetLookupUpdateWorkspace(keyMaxNum, valueDim, keyType, valueType, flags);
}

TEST_F(OpbaseTest, ut_HcclHostmemFree)
{
    s32 ret = HCCL_SUCCESS;
    HostMem mem1 =  HostMem::alloc(8);
    mem1.free();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTest, ut_gather_alltoallv)
{
    MOCKER(GetExternalInputHcclEnableEntryLog)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(true));
    nlohmann::json rank_table = rank_table_910_1server_1rank;

    char file_name_t[] = "./st_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }
    else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s32 rankSize = 1;
    s32 rank = 0;
    s32 count = 200;
    u32 countsNum = 200;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;

    const char* rank_table_file = "./st_opbase_test.json";

    ret = HcclCommInitClusterInfo(rank_table_file, rank, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    u64 memSize = count * rankSize * sizeof(s32) * countsNum;
    DeviceMem recvMem = DeviceMem::alloc(memSize);
    DeviceMem gatherMem = DeviceMem::alloc(memSize);
    vector<u64> addrInfo;
    vector<u64> addrInfoCountPerRank(rankSize, countsNum);
    HostMem hostSendMem = HostMem::alloc(memSize);
    for (u32 i = 0; i < count * rankSize * countsNum; i++) {
        *((s32 *)hostSendMem.ptr() + i) = rank + 1;
        if (i % count == 0) {
            addrInfo.push_back((uintptr_t)((s32 *)hostSendMem.ptr() + i));
            addrInfo.push_back(count * sizeof(s32));
        }
    }

    DeviceMem devAddrInfo = DeviceMem::alloc(addrInfo.size() * sizeof(u64));
    ret = hrtMemSyncCopy(devAddrInfo.ptr(), addrInfo.size() * sizeof(u64), addrInfo.data(), addrInfo.size() * sizeof(u64), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    DeviceMem devCountPerRank = DeviceMem::alloc(addrInfoCountPerRank.size() * sizeof(u64));
    ret = hrtMemSyncCopy(devCountPerRank.ptr(), addrInfoCountPerRank.size() * sizeof(u64), addrInfoCountPerRank.data(),
                         addrInfoCountPerRank.size() * sizeof(u64), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    vector<u64> recvCounts(rankSize, count);
    vector<u64> rdispls(rankSize, 0);
    for (int i = 0; i < rankSize; i++) {
        rdispls[i] = count * i;
        HCCL_INFO("num[%d] displs[%d]", i, count * i);
    }
    DeviceMem devRecvCounts = DeviceMem::alloc(rankSize * sizeof(u64));
    ret = hrtMemSyncCopy(devRecvCounts.ptr(), rankSize * sizeof(u64), recvCounts.data(), rankSize * sizeof(u64), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    DeviceMem devRdispls = DeviceMem::alloc(rankSize * sizeof(u64));
    ret = hrtMemSyncCopy(devRdispls.ptr(), rankSize * sizeof(u64), rdispls.data(), rankSize * sizeof(u64), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcomGatherAllToAllVParams params;
    params.addrInfo = devAddrInfo.ptr();
    params.addrInfoCountPerRank = devCountPerRank.ptr();
    params.recvbuf = recvMem.ptr();
    params.recvcounts = devRecvCounts.ptr();
    params.rdispls = devRdispls.ptr();
    params.recvtype = HCCL_DATA_TYPE_INT32;
    params.gatheredbuf = gatherMem.ptr();
    params.group = nullptr;
    params.addrLength = -1;

    MOCKER(HcclAlltoAllVInner)
    .expects(atMost(10))
    .will(returnValue(0));

    ret = HcclGatherAlltoAllV(params, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    (void)aclrtResetDevice(0);
}

TEST_F(OpbaseTest, ut_GetLocalHostIP)
{
    MOCKER(FindLocalHostIP)
    .stubs()
    .will(returnValue(HCCL_E_INTERNAL));
    std::string serverid;
    std::string ret = GetLocalServerId(serverid);
    EXPECT_EQ(ret, "0.0.0.0");
    GlobalMockObject::verify();
}

TEST_P(OpbaseTest, ut_HcclAlltoAll)
{
    MOCKER(GetExternalInputHcclEnableEntryLog)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(true));
    bool fftsSwitch = GetParam();
    if (fftsSwitch) {
        SetFftsSwitch(true);
    }
    nlohmann::json rank_table = rank_table_910_1server_1rank;
    DevType deviceType = DevType::DEV_TYPE_910;
    MOCKER(hrtGetDeviceType)
    .stubs()
    .with(outBound(deviceType))
    .will(returnValue(HCCL_SUCCESS));

    char file_name_t[] = "./st_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }
    else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;

    const char* rank_table_file = "./st_opbase_test.json";
    u32 rank_ID = 0;
    string tmpOptions = "";
    HcomSetProfilingMode(HcomProfilingMode::PROFILING_OPEN, tmpOptions.c_str());
    ret = HcclCommInitClusterInfo(rank_table_file, rank_ID, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sendbuf= (s8*)sal_malloc(count * sizeof(s8));
     sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf= (s8*)sal_malloc(count * sizeof(s8));
     sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    for (int j = 0; j < count; j++)
    {
        sendbuf[j] = 2;
    }
    unsigned int rankSize = 0;
    ret = HcclGetRankSize(comm, &rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HCCL_INFO("HCCL TEST get rank size[%u] success.", rankSize);

    unsigned int rankId = 0;
    ret = HcclGetRankId(comm, &rankId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HCCL_INFO("HCCL TEST get rank id[%u] success.", rankId);

    ret = HcclAlltoAllInner(sendbuf, count, HCCL_DATA_TYPE_INT8,
                       recvbuf, count, HCCL_DATA_TYPE_INT8,
                       comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    HCCL_ERROR("count %d",count);
    for (int j = 0; j < count; j++)
    {
        if (recvbuf[j] != 2)
        {
            HCCL_ERROR("j %d",j);
            errors ++;
            break;
        }
    }

    sal_free(sendbuf);
    sal_free(recvbuf);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    EXPECT_EQ(errors, 0);
    if (fftsSwitch) {
        SetFftsSwitch(true);
    }
    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_HcclCommGetAsyncError)
{
    nlohmann::json rank_table =
    {
        {"status", "completed"},
        {"deploy_mode", "lab"},
        {"group_count", "1"},
        {"chip_info", "910"},
        {"board_id", "0x0000"},
        {"para_plane_nic_location", "device"},
        {"para_plane_nic_num", "1"},
        {"para_plane_nic_name", {"eth0"}},
        {
            "group_list",
            {
                {
                    {"group_name", ""},
                    {"device_num", "1"},
                    {"server_num", "1"},
                    {"instance_count", "1"},
                        {
                            "instance_list",
                            {
                                {   {"rank_id", "0"}, {"server_id", "10.0.0.10"},
                                    {
                                        "devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.12"}}}
                                    }
                                },
                            }
                        },
                        {
                            "server_list",
                            {
                                {
                                    {"server_id", "192.168.10.2"},
                                    {
                                        "para_plane_info",
                                        {{
                                                {"eth1", "192.168.210.2"},
                                            },
                                            {
                                                {"eth0", "192.168.200.2"},
                                            }
                                        }
                                    }

                                },
                            }
                        }
                }
            }
        }
    };

    char file_name_t[] = "./st_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }
    else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;

    // 走1910 4pring
    const char* rank_table_file = "./st_opbase_test.json";
    u32 rank_ID = 0;

    ret = HcclCommInitClusterInfo(rank_table_file, rank_ID, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER_CPP(&hcclComm::CommCheckErrorCqe)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    HcclResult result = HCCL_SUCCESS;
    HcclGetCommAsyncError((HcclComm)comm, &result);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);

}

TEST_F(OpbaseTest, ut_HcclGetErrorString)
{
    EXPECT_EQ("no error", std::string(HcclGetErrorString(HCCL_SUCCESS)));
    EXPECT_EQ("parameter error", std::string(HcclGetErrorString(HCCL_E_PARA)));
    EXPECT_EQ("empty pointer", std::string(HcclGetErrorString(HCCL_E_PTR)));
    EXPECT_EQ("memory error", std::string(HcclGetErrorString( HCCL_E_MEMORY)));
    EXPECT_EQ("internal error", std::string(HcclGetErrorString(HCCL_E_INTERNAL)));
    EXPECT_EQ("not support feature", std::string(HcclGetErrorString(HCCL_E_NOT_SUPPORT)));
    EXPECT_EQ("not found specific resource", std::string(HcclGetErrorString(HCCL_E_NOT_FOUND)));
    EXPECT_EQ("resource unavailable", std::string(HcclGetErrorString(HCCL_E_UNAVAIL)));
    EXPECT_EQ("call system interface error", std::string(HcclGetErrorString(HCCL_E_SYSCALL)));
    EXPECT_EQ("timeout", std::string(HcclGetErrorString(HCCL_E_TIMEOUT)));
    EXPECT_EQ("open file fail", std::string(HcclGetErrorString(HCCL_E_OPEN_FILE_FAILURE)));
    EXPECT_EQ("tcp connect fail", std::string(HcclGetErrorString(HCCL_E_TCP_CONNECT)));
    EXPECT_EQ("roce connect fail", std::string(HcclGetErrorString(HCCL_E_ROCE_CONNECT)));
    EXPECT_EQ("tcp transfer fail", std::string(HcclGetErrorString(HCCL_E_TCP_TRANSFER)));
    EXPECT_EQ("roce transfer fail", std::string(HcclGetErrorString(HCCL_E_ROCE_TRANSFER)));
    EXPECT_EQ("call runtime api fail", std::string(HcclGetErrorString(HCCL_E_RUNTIME)));
    EXPECT_EQ("call profiling api fail", std::string(HcclGetErrorString(HCCL_E_PROFILING)));
    EXPECT_EQ("call cce api fail", std::string(HcclGetErrorString(HCCL_E_CCE)));
    EXPECT_EQ("call network api fail", std::string(HcclGetErrorString(HCCL_E_NETWORK)));
    EXPECT_EQ("error cqe", std::string(HcclGetErrorString(HCCL_E_REMOTE)));
    EXPECT_EQ("call network api fail", std::string(HcclGetErrorString(HCCL_E_NETWORK)));
    EXPECT_EQ("unknown error", std::string(HcclGetErrorString(HCCL_E_RESERVED)));
}

TEST_F(OpbaseTest, ut_hcclGetRootInfo_single_server_success_normal_9)
{
    // 初始化环境变量，just for ut，防止用例间影响
    ResetInitState();
    setenv("HCCL_WHITELIST_DISABLE", "1", 1);

    // 网卡信息在ra_get_ifaddrs接口已初始化（eth0,docker,lo）
    // 配置匹配eth1，enp前缀的网卡，无法找到Host网卡
    setenv("HCCL_SOCKET_IFNAME", "=eth1,enp", 1);
    HcclRootInfo id;
    HcclResult ret = HcclGetRootInfo(&id);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);

    // 配置匹配enp，env前缀的网卡，无法找到Host网卡
    ResetInitState();
    setenv("HCCL_SOCKET_IFNAME", "enp,env", 1);
    ret = HcclGetRootInfo(&id);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);

    // 配置不匹配,eth0，lo，docker网卡，无法找到Host网卡
    ResetInitState();
    setenv("HCCL_SOCKET_IFNAME", "^=eth0,lo,docker", 1);
    ret = HcclGetRootInfo(&id);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
    // 配置不匹配,eth，dock，和lo前缀的网卡，无法找到Host网卡
    ResetInitState();
    setenv("HCCL_SOCKET_IFNAME", "^eth,dock,lo", 1);
    ret = HcclGetRootInfo(&id);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);

    unsetenv("HCCL_WHITELIST_DISABLE");
    unsetenv("HCCL_SOCKET_IFNAME");
}

TEST_F(OpbaseTest, ut_HcclAiCpuResource91093)
{
    HcclRootInfo id;
    char group[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {0};
    void *commContext = nullptr;
    void *aicpuNotify = nullptr;
    rtStream_t Opstream;

    DevType deviceType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType)
    .stubs()
    .with(outBound(deviceType))
    .will(returnValue(HCCL_SUCCESS));
	MOCKER(hrtRaGetSingleSocketVnicIpInfo)
	.stubs()
	.with(mockcpp::any())
	.will(invoke(stub_hrtRaGetSingleSocketVnicIpInfo));
    MOCKER_CPP(&HcclCommunicator::HcclGetCmdTimeout)
    .stubs()
    .will(returnValue(50));
    HcclResult ret = HcclGetRootInfo(&id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclComm newcomm;
    ret = HcclCommInitRootInfo(1, &id, 0, &newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclGetCommName(newcomm, group);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    printf("commName:%s", group);

    ret = HcclCreateComResource(group, 1, &commContext);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclGetAicpuOpStreamNotify(group, &Opstream, &aicpuNotify);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(aicpuNotify, nullptr);

    HcclComm commHandle;
    ret = HcomGetCommHandleByGroup(group, &commHandle);

    ret = HcclCreateComResourceByComm(commHandle, 1, true, &commContext);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclGetAicpuOpStreamAndNotify(commHandle, &Opstream, 1, &aicpuNotify);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(aicpuNotify, nullptr);

    ret = HcclCommDestroy(newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_HcclAiCpuResource)
{
    DevType deviceType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDeviceType)
    .stubs()
    .with(outBound(deviceType))
    .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::HcclGetCmdTimeout)
    .stubs()
    .will(returnValue(50));

    HcclRootInfo id;
    char group[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {0};
    void *commContext = nullptr;
    void *aicpuNotify = nullptr;
    rtStream_t Opstream;

    HcclResult ret = HcclGetRootInfo(&id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclComm newcomm;
    ret = HcclCommInitRootInfo(1, &id, 0, &newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclGetCommName(newcomm, group);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    printf("commName:%s", group);

    ret = HcclCreateComResource(group, 1, &commContext);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(commContext, nullptr);

    ret = HcclGetAicpuOpStreamNotify(group, &Opstream, &aicpuNotify);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(aicpuNotify, nullptr);

    HcclComm commHandle;
    ret = HcomGetCommHandleByGroup(group, &commHandle);

    ret = HcclCreateComResourceByComm(commHandle, 1, true, &commContext);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(commContext, nullptr);

    ret = HcclGetAicpuOpStreamAndNotify(commHandle, &Opstream, 1, &aicpuNotify);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(aicpuNotify, nullptr);

    ret = HcclCommDestroy(newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

#pragma pack(4)
struct MC2Tiling {
    uint32_t version = 1;
    uint32_t mc2HcommCnt = 2;
    Mc2ServerCfg serverCfg;
    Mc2HcommCfg cfg1;
    Mc2HcommCfg cfg2;
};
#pragma pack()

TEST_F(OpbaseTest, ut_HcclAiCpuResourceByTiling)
{
    HcclRootInfo id;
    char group[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {0};
    void *commContext = nullptr;
    void *aicpuNotify = nullptr;
    rtStream_t Opstream = nullptr;

    DevType deviceType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDeviceType)
    .stubs()
    .with(outBound(deviceType))
    .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::HcclGetCmdTimeout)
    .stubs()
    .will(returnValue(50));

    HcclResult ret = HcclGetRootInfo(&id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclComm newcomm;
    ret = HcclCommInitRootInfo(1, &id, 0, &newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclGetCommName(newcomm, group);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    printf("commName:%s\n", group);

    HcclComm commHandle;
    ret = HcomGetCommHandleByGroup(group, &commHandle);

    MC2Tiling mc2Tiling;
    mc2Tiling.version = 2;
    mc2Tiling.mc2HcommCnt = 2;

    mc2Tiling.cfg1.opType = 8;
    mc2Tiling.cfg1.reduceType = 0;
    strcpy_s(mc2Tiling.cfg1.groupName, 128, group);
    strcpy_s(mc2Tiling.cfg1.algConfig, 128, "AlltoAll=level0:fullmesh;level1:pairwise");

    mc2Tiling.cfg2.opType = 6;
    mc2Tiling.cfg2.reduceType = 0;
    strcpy_s(mc2Tiling.cfg2.groupName, 128, group);
    strcpy_s(mc2Tiling.cfg2.algConfig, 128, "AllGather=level0:ring");

    rtStream_t stream;
    rtError_t rt_ret = RT_ERROR_NONE;
    rt_ret = aclrtCreateStream(&stream);

    ret = HcclAllocComResourceByTiling(commHandle, stream, &mc2Tiling, &commContext);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(commContext, nullptr);

    ret = HcclGetAicpuOpStreamAndNotify(commHandle, &Opstream, 1, &aicpuNotify);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(Opstream, nullptr);

    mc2Tiling.version = 0;

    ret = HcclAllocComResourceByTiling(commHandle, stream, &mc2Tiling, &commContext);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(commContext, nullptr);

    ret = HcclGetAicpuOpStreamAndNotify(commHandle, &Opstream, 1, &aicpuNotify);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(Opstream, nullptr);

    rt_ret = aclrtDestroyStream(stream);
    ret = HcclCommDestroy(newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

#pragma pack(4)
struct MC2TilingV2 {
    uint32_t version = 1;
    uint32_t mc2HcommCnt = 2;
    uint32_t offset[8];
    Mc2HcommCfg cfg1;
    uint32_t tmp;
    Mc2HcommCfg cfg2;
};
#pragma pack()

TEST_F(OpbaseTest, ut_HcclAiCpuResourceByTiling_A3)
{
    HcclRootInfo id;
    char group[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {0};
    void *commContext = nullptr;
    void *aicpuNotify = nullptr;
    rtStream_t Opstream = nullptr;

    DevType deviceType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType)
    .stubs()
    .with(outBound(deviceType))
    .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::HcclGetCmdTimeout)
    .stubs()
    .will(returnValue(50));

    HcclResult ret = HcclGetRootInfo(&id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclComm newcomm;
    ret = HcclCommInitRootInfo(1, &id, 0, &newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclGetCommName(newcomm, group);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    printf("commName:%s\n", group);

    HcclComm commHandle;
    ret = HcomGetCommHandleByGroup(group, &commHandle);

    MC2TilingV2 mc2Tiling;
    mc2Tiling.version = 2;
    mc2Tiling.mc2HcommCnt = 2;
    mc2Tiling.offset[0] = reinterpret_cast<uint8_t *>(&(mc2Tiling.cfg1)) - reinterpret_cast<uint8_t *>(&mc2Tiling);
    mc2Tiling.offset[1] = reinterpret_cast<uint8_t *>(&(mc2Tiling.cfg2)) - reinterpret_cast<uint8_t *>(&mc2Tiling);

    mc2Tiling.cfg1.opType = 8;
    mc2Tiling.cfg1.reduceType = 0;
    strcpy_s(mc2Tiling.cfg1.groupName, 128, group);
    strcpy_s(mc2Tiling.cfg1.algConfig, 128, "AlltoAll=level0:fullmesh;level1:pairwise");

    mc2Tiling.cfg2.opType = 6;
    mc2Tiling.cfg2.reduceType = 0;
    strcpy_s(mc2Tiling.cfg2.groupName, 128, group);
    strcpy_s(mc2Tiling.cfg2.algConfig, 128, "AllGather=level0:ring");

    rtStream_t stream;
    rtError_t rt_ret = RT_ERROR_NONE;
    rt_ret = aclrtCreateStream(&stream);

    ret = HcclAllocComResourceByTiling(commHandle, stream, &mc2Tiling, &commContext);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(commContext, nullptr);

    ret = HcclGetAicpuOpStreamAndNotify(commHandle, &Opstream, 1, &aicpuNotify);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(Opstream, nullptr);

    mc2Tiling.version = 3;

    ret = HcclAllocComResourceByTiling(commHandle, stream, &mc2Tiling, &commContext);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(commContext, nullptr);

    ret = HcclGetAicpuOpStreamAndNotify(commHandle, &Opstream, 1, &aicpuNotify);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(Opstream, nullptr);

    rt_ret = aclrtDestroyStream(stream);
    ret = HcclCommDestroy(newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

extern thread_local s32 g_hcclDeviceId;
TEST_F(OpbaseTest, ut_LegacyGetHcclOpInfoCtx_cover_bottom_false)
{
    for (int i = 0; i < MAX_MODULE_DEVICE_NUM + 1; i++) {
        CollCommMgr::GetInstance().LegacyGetOpHcomInfo(i).isUsed = false;
    }
    g_hcclDeviceId = INVALID_INT;
    HcclResult ret = HCCL_SUCCESS;
    MOCKER(hrtGetDevice)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_E_INTERNAL));
    const char *group = "group";
    std::shared_ptr<hccl::hcclComm> comm;
    ret = HcclGetCommHandle(group, comm);
    EXPECT_EQ(ret, HCCL_E_PARA);

    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_LegacyGetHcclOpInfoCtx_cover_bottom_true)
{
    for (int i = 0; i < MAX_MODULE_DEVICE_NUM; i++) {
        CollCommMgr::GetInstance().LegacyGetOpHcomInfo(i).isUsed = false;
    }
    CollCommMgr::GetInstance().LegacyGetOpHcomInfo(MAX_MODULE_DEVICE_NUM).isUsed = true;
    g_hcclDeviceId = INVALID_INT;
    HcclResult ret = HCCL_SUCCESS;
    MOCKER(hrtGetDevice)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));
    const char *group = "group";
    std::shared_ptr<hccl::hcclComm> comm;
    g_hcclDeviceId = 0;
    ret = HcclGetCommHandle(group, comm);
    EXPECT_EQ(ret, HCCL_E_PARA);

    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_opbaseinit_hcomstream)
{
    int ret = HCCL_SUCCESS;
    uint32_t ndev = 1;
    int32_t devices[ndev] = {0};
    HcclComm comms[ndev];
    for (int i = 0; i < ndev; i++) {
        ret = hrtSetDevice(devices[i]);
        EXPECT_EQ(ret, 0);
    }
    ret = HcclCommInitAll(ndev, devices, comms);
    EXPECT_EQ(ret, 0);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comms[0]);
    string groupName = hcclComm->GetIdentifier();
    u64 memSize = 0;
    ret = HcomGetWorkspaceMemSize("HcomReduceScatter", 1, HCCL_DATA_TYPE_INT8, groupName.c_str(), memSize);
    EXPECT_EQ(ret, 0);

    for (uint32_t i = 0; i < ndev; i++) {
        ret = hrtResetDevice(devices[i]);
        EXPECT_EQ(ret, 0);
        ret = HcclCommDestroy(comms[i]);
        EXPECT_EQ(ret, 0);
    }
    GlobalMockObject::verify();
}

#if 1
#define HCCL_COM_DATA_SIZE 1024
#define DEV_NUM_8 8

TEST_F(OpbaseTest, ut_BatchSendRecv_self)
{
    MOCKER(GetExternalInputHcclEnableEntryLog)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(true));
    nlohmann::json rank_table = rank_table_910_1server_1rank;

    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }
    else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;
    s32 ndev = DEV_NUM_8;

    const char* rank_table_file = "./ut_opbase_test.json";

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sendbuf= (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf= (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));
    ret = HcclCommInitClusterInfo(rank_table_file, 0, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    for (int j = 0; j < count; j++)
    {
        sendbuf[j] = 2;
    }

    MOCKER_CPP(&TransportManager::Alloc)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollBatchSendRecvExecutor::ProcessSendDataSlice)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollBatchSendRecvExecutor::ProcessRecvDataSlice)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    HcclSendRecvItem data[] = {{HcclSendRecvType::HCCL_SEND, sendbuf, HCCL_COM_DATA_SIZE, HcclDataType::HCCL_DATA_TYPE_INT8, 0},
                               {HcclSendRecvType::HCCL_RECV, recvbuf, HCCL_COM_DATA_SIZE, HcclDataType::HCCL_DATA_TYPE_INT8, 0}};
    HcclSendRecvItem* dataPtr = data;
    ret = HcclBatchSendRecvInner(dataPtr, 2, comm, stream);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sal_free(sendbuf);
    sal_free(recvbuf);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    GlobalMockObject::verify();
}
#endif

#if 1
TEST_F(OpbaseTest, ut_BatchSendRecv_2rank_send)
{

    nlohmann::json rank_table = rank_table_910_1server_2rank;

    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }
    else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;
    s32 ndev = DEV_NUM_8;
    // 走1910 8pring
    const char* rank_table_file = "./ut_opbase_test.json";

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sendbuf= (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf= (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));
    ret = HcclCommInitClusterInfo(rank_table_file, 0, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    for (int j = 0; j < count; j++)
    {
        sendbuf[j] = 2;
    }
    MOCKER_CPP(&TransportManager::Alloc)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollBatchSendRecvExecutor::ProcessSendDataSlice)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollBatchSendRecvExecutor::ProcessRecvDataSlice)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollBatchSendRecvExecutor::GetSendTargetLink)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollBatchSendRecvExecutor::GetRecvTargetLink)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollBatchSendRecvExecutor::RunLoopInHostUnfoldMode)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    HcclSendRecvItem data[] = {{HcclSendRecvType::HCCL_SEND, sendbuf, HCCL_COM_DATA_SIZE, HcclDataType::HCCL_DATA_TYPE_INT8, 1}};
    HcclSendRecvItem* dataPtr = data;
    ret = HcclBatchSendRecvInner(dataPtr, 1, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sal_free(sendbuf);
    sal_free(recvbuf);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_BatchSendRecv_2rank_send_91093_aicpu_bigsize)
{
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    DevType deviceType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType)
    .stubs()
    .with(outBound(deviceType))
    .will(returnValue(HCCL_SUCCESS));

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s32 rank = 0;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;
    s32 ndev = DEV_NUM_8;

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    HcclComm newcomm;
    HcclRootInfo id;
    ret = HcclGetRootInfo(&id);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclCommInitRootInfo(1, &id, 0, &newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER_CPP(&TransportManager::Alloc)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollBatchSendRecvExecutor::ProcessSendDataSlice)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollBatchSendRecvExecutor::ProcessRecvDataSlice)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::IsAtomicInit)
    .stubs()
    .will(returnValue(true));
    MOCKER(&hrtAicpuKernelLaunchExWithArgs)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollBatchSendRecvExecutor::GetSendTargetLink)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollBatchSendRecvExecutor::GetRecvTargetLink)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollBatchSendRecvExecutor::RunLoopInHostUnfoldMode)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    HcclCommunicator impl;
    HcclCommParams params;
    string commId = "BatchSendRecv";
    memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1);
    params.rank = 0;
    params.totalRanks = 1;
    params.isHeterogComm = false;
    params.logicDevId = 0;
    params.deviceType = DevType::DEV_TYPE_910_93;

    RankTable_t rankTable;
    rankTable.collectiveId = "192.168.0.101-8000-8001";
    vector<RankInfo_t> rankVec(1);
    rankVec[0].rankId = 0;
    rankVec[0].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr1(1694542016);
    rankVec[0].deviceInfo.deviceIp.push_back(ipAddr1); // 101.0.168.192
    rankVec[0].serverIdx = 0;
    rankVec[0].serverId = "192.168.0.101";
    rankTable.rankList.assign(rankVec.begin(), rankVec.end());
    rankTable.deviceNum = 1;
    rankTable.serverNum = 1;
    aclrtSetDevice(0);

    ret = impl.Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    impl.userRankSize_ = 2;
    impl.InitCCLbuffer(1024, 1024);

    void* inputPtr = nullptr;
    void* outputPtr = nullptr;
    u64 addr = 0U;
    void* tilingDataPtr = nullptr;
    u32 tilingDataSize = 48 * 1024;
    tilingDataPtr= (void*)sal_malloc(tilingDataSize * sizeof(s8));
    sal_memset(tilingDataPtr, tilingDataSize * sizeof(s8), 0, tilingDataSize * sizeof(s8));
    std::string kernelName = "RunAicpuRpcSrvLaunchV2";
    ret  = impl.AicpuUnfoldKernelLaunchV2(inputPtr, outputPtr, stream, addr, tilingDataPtr, tilingDataSize, kernelName,
        HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE, commId, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sal_free(tilingDataPtr);
    rt_ret = aclrtDestroyStream(stream);
    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    unsetenv("HCCL_OP_EXPANSION_MODE");
    GlobalMockObject::verify();
}

#endif

#if 1
TEST_F(OpbaseTest, ut_BatchSendRecv_2rank_recv)
{

    nlohmann::json rank_table = rank_table_910_1server_2rank;

    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }
    else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;
    s32 ndev = DEV_NUM_8;
    // 走1910 8pring
    const char* rank_table_file = "./ut_opbase_test.json";

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sendbuf= (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf= (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));
    ret = HcclCommInitClusterInfo(rank_table_file, 0, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    for (int j = 0; j < count; j++)
    {
        sendbuf[j] = 2;
    }
    MOCKER_CPP(&TransportManager::Alloc)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollBatchSendRecvExecutor::ProcessSendDataSlice)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollBatchSendRecvExecutor::ProcessRecvDataSlice)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollBatchSendRecvExecutor::GetSendTargetLink)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollBatchSendRecvExecutor::GetRecvTargetLink)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollBatchSendRecvExecutor::RunLoopInHostUnfoldMode)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    HcclSendRecvItem data[] = {{HcclSendRecvType::HCCL_RECV, recvbuf, HCCL_COM_DATA_SIZE, HcclDataType::HCCL_DATA_TYPE_INT8, 1}};
    HcclSendRecvItem* dataPtr = data;
    ret = HcclBatchSendRecvInner(dataPtr, 1, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sal_free(sendbuf);
    sal_free(recvbuf);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    GlobalMockObject::verify();
}
#endif

#if 1
TEST_P(OpbaseTest, ut_BatchSendRecv_4rank_send_recv)
{

    bool fftsSwitch = GetParam();
    if (fftsSwitch) {
        SetFftsSwitch(true);
    }
    nlohmann::json rank_table = rank_table_910_1server_4rank;

    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }
    else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;
    s32 ndev = DEV_NUM_8;
    // 走1910 8pring
    const char* rank_table_file = "./ut_opbase_test.json";

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sendbuf= (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf= (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));
    ret = HcclCommInitClusterInfo(rank_table_file, 0, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    for (int j = 0; j < count; j++)
    {
        sendbuf[j] = 2;
    }

    MOCKER_CPP(&TransportManager::Alloc)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollBatchSendRecvExecutor::ProcessSendDataSlice)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollBatchSendRecvExecutor::ProcessRecvDataSlice)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollBatchSendRecvExecutor::GetSendTargetLink)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollBatchSendRecvExecutor::GetRecvTargetLink)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollBatchSendRecvExecutor::RunLoopInHostUnfoldMode)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    HcclSendRecvItem data[] = {{HcclSendRecvType::HCCL_RECV, recvbuf, HCCL_COM_DATA_SIZE, HcclDataType::HCCL_DATA_TYPE_INT8, 1},
                                {HcclSendRecvType::HCCL_SEND, sendbuf, HCCL_COM_DATA_SIZE, HcclDataType::HCCL_DATA_TYPE_INT8, 1}};
    HcclSendRecvItem* dataPtr = data;
    ret = HcclBatchSendRecvInner(dataPtr, 2, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sal_free(sendbuf);
    sal_free(recvbuf);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);

    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    if (fftsSwitch) {
        SetFftsSwitch(true);
    }
    GlobalMockObject::verify();
}
#endif

#if 1
TEST_F(OpbaseTest, ut_BatchSendRecv_4rank_send_multitimes)
{

    nlohmann::json rank_table = rank_table_910_1server_4rank;

    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }
    else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;
    s32 ndev = DEV_NUM_8;
    // 走1910 8pring
    const char* rank_table_file = "./ut_opbase_test.json";

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sendbuf= (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf= (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));
    ret = HcclCommInitClusterInfo(rank_table_file, 0, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    for (int j = 0; j < count; j++)
    {
        sendbuf[j] = 2;
    }
    MOCKER_CPP(&TransportManager::Alloc)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TransportManager::IncreAlloc)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollBatchSendRecvExecutor::ProcessSendDataSlice)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollBatchSendRecvExecutor::ProcessRecvDataSlice)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollBatchSendRecvExecutor::GetSendTargetLink)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollBatchSendRecvExecutor::GetRecvTargetLink)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollBatchSendRecvExecutor::RunLoopInHostUnfoldMode)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));
    HcclSendRecvItem data[] = {{HcclSendRecvType::HCCL_SEND, sendbuf, HCCL_COM_DATA_SIZE, HcclDataType::HCCL_DATA_TYPE_INT8, 1}};
    HcclSendRecvItem* dataPtr = data;
    ret = HcclBatchSendRecvInner(dataPtr, 1, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclSendRecvItem data1[] = {{HcclSendRecvType::HCCL_SEND, sendbuf, HCCL_COM_DATA_SIZE, HcclDataType::HCCL_DATA_TYPE_INT8, 2}};
    HcclSendRecvItem* dataPtr1 = data1;
    ret = HcclBatchSendRecvInner(dataPtr1, 1, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sal_free(sendbuf);
    sal_free(recvbuf);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    GlobalMockObject::verify();
}
#endif

#if 1
TEST_F(OpbaseTest, ut_hcclGetRootInfo_91093_single_server_success_normal_3)
{
    setenv("HCCL_WHITELIST_DISABLE", "1", 1);
    setenv("HCCL_IF_IP", "127.0.0.2", 1);

    DevType type91093 = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType)
    .stubs()
    .with(outBound(type91093))
    .will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtGetDeviceInfo)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    HcclRootInfo id;
    HcclResult ret = HcclGetRootInfo(&id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclComm newcomm;
    ret = HcclCommInitRootInfo(1, &id, 0, &newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclCommDestroy(newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    unsetenv("HCCL_WHITELIST_DISABLE");
    unsetenv("HCCL_IF_IP");
    GlobalMockObject::verify();
}
#endif

TEST_F(OpbaseTest, ut_HcclAiCpuUnfold310P)
{
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);

    s8* sendBuf;
    s8* recvBuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    HcclRootInfo id;
    char group[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {0};
    void *commContext = nullptr;
    void *aicpuNotify = nullptr;
    rtStream_t stream;

    DevType deviceType = DevType::DEV_TYPE_310P3;
    MOCKER(hrtGetDeviceType)
    .stubs()
    .with(outBound(deviceType))
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::RegisterToHeartBeat, HcclResult(HcclCommunicator::*)())
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    HcclResult ret = HcclGetRootInfo(&id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclComm newcomm;
    ret = HcclCommInitRootInfo(1, &id, 0, &newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sendBuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendBuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvBuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvBuf, count * sizeof(s8), 0, count * sizeof(s8));

    HcclCommunicator impl;
    HcclCommParams params;
    string commId = "AllReduce";
    memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1);
    params.rank = 0;
    params.totalRanks = 2;
    params.isHeterogComm = false;
    params.logicDevId = 0;
    params.deviceType = DevType::DEV_TYPE_310P3;

    RankTable_t rankTable;
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
    aclrtSetDevice(0);

    ret = impl.Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    impl.userRankSize_ = 2;
    impl.InitCCLbuffer(1024, 1024);

    impl.AllReduceOutPlace(commId, sendBuf, recvBuf, count, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, stream, SyncMode::DEFAULT_TIMEWAITSYNCMODE);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclCommDestroy(newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    aclrtDestroyStream(stream);
    sal_free(sendBuf);
    sal_free(recvBuf);
    impl.ReleaseCommCCLbuffer();
    unsetenv("HCCL_OP_EXPANSION_MODE");
    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_hcclGetRootInfo_single_server_fail_empty_allowlist)
{
    char *buffer;
    //也可以将buffer作为输出参数
    if((buffer = getcwd(nullptr, 0)) != nullptr)
    {
        std::string dirPath = buffer;
        std::string fileName = "/ut_hccl_host_allowlist";
        std::string realPath = dirPath.c_str() + fileName;
        setenv("HCCL_WHITELIST_FILE", realPath.c_str(), 1);
        setenv("HCCL_WHITELIST_DISABLE", "0", 1);
        ofstream outfile(realPath);
        outfile << "" << endl;
        outfile.close();

        HcclRootInfo id;
        HcclResult ret = HcclGetRootInfo(&id);
        EXPECT_NE(ret, HCCL_SUCCESS);

        unsetenv("HCCL_WHITELIST_FILE");
        unsetenv("HCCL_WHITELIST_DISABLE");
        remove(realPath.c_str());
    }
    free(buffer);
}

TEST_F(OpbaseTest, ut_HcclAiCpuUnfold910B)
{
    setenv("HCCL_OP_EXPANSION_MODE", "AI_CPU", 1);
    s8* sendBuf;
    s8* recvBuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    HcclRootInfo id;
    char group[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {0};
    void *commContext = nullptr;
    void *aicpuNotify = nullptr;
    rtStream_t stream;

    DevType deviceType = DevType::DEV_TYPE_910B;
    (void) SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
    MOCKER(hrtGetDeviceType)
    .stubs()
    .with(outBound(deviceType))
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::RegisterToHeartBeat, HcclResult(HcclCommunicator::*)())
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    HcclResult ret = HcclGetRootInfo(&id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclComm newcomm;
    ret = HcclCommInitRootInfo(1, &id, 0, &newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sendBuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendBuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvBuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvBuf, count * sizeof(s8), 0, count * sizeof(s8));

    HcclCommunicator impl;
    HcclCommParams params;
    string commId = "AllReduce";
    memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1);
    params.rank = 0;
    params.totalRanks = 2;
    params.isHeterogComm = false;
    params.logicDevId = 0;
    params.deviceType = DevType::DEV_TYPE_910B;

    RankTable_t rankTable;
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
    aclrtSetDevice(0);

    ret = impl.Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_E_PARA);

    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    impl.userRankSize_ = 2;
    impl.InitCCLbuffer(1024, 1024);
    impl.isSingleMeshAggregation_ = true;
    impl.AllReduceOutPlace(commId, sendBuf, recvBuf, count, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, stream, SyncMode::DEFAULT_TIMEWAITSYNCMODE);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = HcclCommDestroy(newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    aclrtDestroyStream(stream);
    sal_free(sendBuf);
    sal_free(recvBuf);
    impl.ReleaseCommCCLbuffer();
    unsetenv("HCCL_OP_EXPANSION_MODE");
    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_HcclCommInitRootInfoConfig_default)
{
    HcclRootInfo id;

    DevType deviceType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDeviceType)
    .stubs()
    .with(outBound(deviceType))
    .will(returnValue(HCCL_SUCCESS));

    HcclResult ret = HcclGetRootInfo(&id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclRootHandle rootHandle;
    s32 sRet = memcpy_s(&rootHandle, sizeof(HcclRootHandle), id.internal, sizeof(HcclRootHandle));
    EXPECT_EQ(ret, 0);

    HcclCommConfig config;
    HcclCommConfigInit(&config);

    HcclComm newcomm;
    ret = HcclCommInitRootInfoConfig(1, &id, 0, &config, &newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclComm *pComm = static_cast<hcclComm *>(newcomm);
    EXPECT_EQ(pComm->GetConfigInCCLbufferSize(), 200 * 1024 * 1024);
    EXPECT_EQ(pComm->GetConfigOutCCLbufferSize(), 200 * 1024 * 1024);
    EXPECT_EQ(pComm->communicator_->GetDeterministicConfig(), 0);
    EXPECT_EQ(pComm->GetIdentifier(), rootHandle.identifier);

    ret = HcclCommDestroy(newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_HcclCommInitRootInfoConfig_user_config)
{
    HcclRootInfo id;

    DevType deviceType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDeviceType)
    .stubs()
    .with(outBound(deviceType))
    .will(returnValue(HCCL_SUCCESS));

    HcclResult ret = HcclGetRootInfo(&id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclCommConfig config;
    HcclCommConfigInit(&config);

    config.hcclBufferSize = 300;
    config.hcclDeterministic = 1;
    strcpy_s(config.hcclCommName, COMM_NAME_MAX_LENGTH, "comm1");

    HcclComm newcomm;
    ret = HcclCommInitRootInfoConfig(1, &id, 0, &config, &newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclComm *pComm = static_cast<hcclComm *>(newcomm);
    EXPECT_EQ(pComm->GetConfigInCCLbufferSize(), 300 * 1024 * 1024);
    EXPECT_EQ(pComm->GetConfigOutCCLbufferSize(), 300 * 1024 * 1024);
    EXPECT_EQ(pComm->communicator_->GetDeterministicConfig(), 1);
    EXPECT_EQ(pComm->GetIdentifier(), "comm1");

    ret = HcclCommDestroy(newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_HcclCommInitRootInfoConfig_user_config_world_group)
{
    HcclRootInfo id;

    DevType deviceType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDeviceType)
    .stubs()
    .with(outBound(deviceType))
    .will(returnValue(HCCL_SUCCESS));

    HcclResult ret = HcclGetRootInfo(&id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclCommConfig config;
    HcclCommConfigInit(&config);

    config.hcclBufferSize = 300;
    config.hcclDeterministic = 1;

    strcpy_s(config.hcclCommName, 128, HCCL_WORLD_GROUP);

    HcclComm newcomm;
    ret = HcclCommInitRootInfoConfig(1, &id, 0, &config, &newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclComm *pComm = static_cast<hcclComm *>(newcomm);
    EXPECT_EQ(pComm->GetConfigInCCLbufferSize(), 300 * 1024 * 1024);
    EXPECT_EQ(pComm->GetConfigOutCCLbufferSize(), 300 * 1024 * 1024);
    EXPECT_EQ(pComm->communicator_->GetDeterministicConfig(), 1);
    EXPECT_EQ(pComm->GetIdentifier(), HCCL_WORLD_GROUP);

    ret = HcclCommDestroy(newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_HcclCommInitRootInfoConfig_duplicate_comm_name)
{
    HcclRootInfo id;

    DevType deviceType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDeviceType)
    .stubs()
    .with(outBound(deviceType))
    .will(returnValue(HCCL_SUCCESS));

    HcclResult ret = HcclGetRootInfo(&id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclCommConfig config;
    HcclCommConfigInit(&config);

    strcpy_s(config.hcclCommName, COMM_NAME_MAX_LENGTH, "comm1");

    HcclComm newcomm;
    ret = HcclCommInitRootInfoConfig(1, &id, 0, &config, &newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclComm newcomm2;
    ret = HcclCommInitRootInfoConfig(1, &id, 0, &config, &newcomm2); // 使用了相同的comm name，预期会报错
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = HcclCommDestroy(newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_HcclGetCommConfigCapability)
{
    uint32_t capability = HcclGetCommConfigCapability();
    EXPECT_EQ(capability, HCCL_COMM_CONFIG_RESERVED);
    EXPECT_EQ(capability > HCCL_COMM_CONFIG_BUFFER_SIZE, 1);
    EXPECT_EQ(capability > HCCL_COMM_CONFIG_DETERMINISTIC, 1);
    EXPECT_EQ(capability > HCCL_COMM_CONFIG_COMM_NAME, 1);
}

TEST_F(OpbaseTest, ut_SetDynamicTilingDataAlltoall)
{
    HcclCommunicator impl;
    OpParam opParam;
    opParam.All2AllDataDes.sendType = HcclDataType::HCCL_DATA_TYPE_FP32;
    opParam.All2AllDataDes.recvType = HcclDataType::HCCL_DATA_TYPE_FP32;
    opParam.All2AllDataDes.sendCount = 16;
    HostMem dynamicDataMem = HostMem::alloc(sizeof(struct OpTilingAllToAllDataDes));
    HcclResult ret = impl.SetDynamicTilingDataAlltoall(opParam, dynamicDataMem);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(OpbaseTest, ut_initWithConfig)
{

    nlohmann::json rank_table = rank_table_910_1server_1rank;

    char file_name_t[] = "./st_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }
    else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s32 count = HCCL_COM_DATA_SIZE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;

    // 走1910 4pring
    const char* rank_table_file = "./st_opbase_test.json";
    u32 rank_ID = 0;

    HcclCommConfig commConfig;
    HcclCommConfigInit(&commConfig);
    commConfig.hcclBufferSize=400;
    commConfig.hcclBufferSize=1;

    ret = HcclCommInitClusterInfoConfig(rank_table_file, rank_ID, &commConfig, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
}

TEST_F(OpbaseTest, ut_initSubComm)
{

    nlohmann::json rank_table = rank_table_910_1server_1rank;

    char file_name_t[] = "./st_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }
    else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s32 count = HCCL_COM_DATA_SIZE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* global_comm;
    void* comm;

    // 走1910 4pring
    const char* rank_table_file = "./st_opbase_test.json";
    u32 rank_ID = 0;

    HcclCommConfig commConfig;
    HcclCommConfigInit(&commConfig);
    commConfig.hcclBufferSize=400;
    commConfig.hcclBufferSize=1;

    ret = HcclCommInitClusterInfoConfig(rank_table_file, rank_ID, &commConfig, &global_comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    uint32_t rankIds[1] = {0};
    ret = HcclCreateSubCommConfig(&global_comm, 1, rankIds, 2, rank_ID, &commConfig, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclCommDestroy(global_comm);
    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
}

TEST_F(OpbaseTest, ut_FreeScratchMemOnOpBaseMode)
{
    HcclCommunicator impl;
    OpParam opParam;
    opParam.aicpuUnfoldMode = true;
    HcclCMDType opType = HcclCMDType::HCCL_CMD_ALLTOALL;
    DeviceMem scratchMem;
    HcclResult ret = impl.FreeScratchMemOnOpBaseMode(scratchMem, opParam, opType);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(OpbaseTest, ut_CheckDataTypeAllBranch)
{
    MOCKER(Is310P3Common)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(true));

    HcclCommunicator impl;
    HcclResult ret = impl.CheckDataType(HcclDataType::HCCL_DATA_TYPE_RESERVED, false);
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, ret);

    ret = impl.CheckDataType(HcclDataType::HCCL_DATA_TYPE_INT64, true);
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, ret);

    ret = impl.CheckDataType(HcclDataType::HCCL_DATA_TYPE_UINT64, true);
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, ret);

    GlobalMockObject::verify();
}
TEST_F(OpbaseTest, ut_310P3CommonFrontLogPrint)
{
    MOCKER(Is310P3Common)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(true));
    MOCKER_CPP(&HcclCommunicator::IsAtomicInit)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(true));

    HcclCommunicator impl;
    void *sendBuf = nullptr;
    void *recvBuf = nullptr;
    void *sendCounts = nullptr;
    void *recvCounts = nullptr;
    void *sdispls = nullptr;
    void *rdispls = nullptr;
    rtStream_t stream;
    aclrtCreateStream(&stream);
    void *comm = nullptr;
    HcclResult ret = HCCL_SUCCESS;
    ret = impl.AlltoAllVC(sendBuf, sendCounts, HcclDataType::HCCL_DATA_TYPE_RESERVED,
        recvBuf, HcclDataType::HCCL_DATA_TYPE_RESERVED, stream, "alltoallvc");
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, ret);
    ret = impl.Scatter("scatter", sendBuf, recvBuf, 0, HcclDataType::HCCL_DATA_TYPE_RESERVED, 0, stream);
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, ret);
    ret = impl.ScatterOutPlace("ScatterOutPlace", sendBuf, recvBuf, 0,
            HcclDataType::HCCL_DATA_TYPE_RESERVED, 0, stream);
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, ret);
    ret = impl.Reduce("Reduce", sendBuf, recvBuf, 0, HcclDataType::HCCL_DATA_TYPE_RESERVED,
            HcclReduceOp::HCCL_REDUCE_RESERVED, 0, stream);
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, ret);
    ret = impl.SendOutPlace("SendOutPlace", sendBuf, 0, HcclDataType::HCCL_DATA_TYPE_RESERVED, 0, stream);
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, ret);
    ret = impl.ReceiveOutPlace("ReceiveOutPlace", recvBuf, 0, HcclDataType::HCCL_DATA_TYPE_RESERVED, 1, stream);
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, ret);
    u64 *sendCounts1 = nullptr;
    u64 *recvCounts1 = nullptr;
    u64 *sdispls1 = nullptr;
    u64 *rdispls1 = nullptr;
    u64 memSize = 0;
    ret = impl.GetAlltoAllStagedWorkSpaceMemSize(sendCounts1, sdispls1, HcclDataType::HCCL_DATA_TYPE_RESERVED,
            recvCounts1, rdispls1, HcclDataType::HCCL_DATA_TYPE_RESERVED, memSize);
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, ret);
    rtError_t rt_ret = RT_ERROR_NONE;
    rt_ret = aclrtDestroyStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_HcclCommInitClusterInfoMemConfig_multiple_comm)
{
    typedef HcclResult (*HcclOneSideServiceCallBack)(std::unique_ptr<hccl::IHcclOneSidedService> &,
    std::unique_ptr<hccl::HcclSocketManager> &, std::unique_ptr<hccl::NotifyPool> &);

    nlohmann::json rank_table_one = rank_table_910_1server_2rank;
    nlohmann::json rank_table_two = rank_table_910_1server_4rank;
    std::string clusterString_one = rank_table_one.dump();
    std::string clusterString_two = rank_table_two.dump();

    int ret = HCCL_SUCCESS;
    u32 rank_ID = 0;
    void* commOne;
    void* commTwo;

    HcclCommConfig commConfigOne;
    HcclCommConfigInit(&commConfigOne);
    commConfigOne.hcclBufferSize=800;
    strcpy_s(commConfigOne.hcclCommName, COMM_NAME_MAX_LENGTH, "comm1");

    HcclCommConfig commConfigTwo;
    HcclCommConfigInit(&commConfigTwo);
    commConfigTwo.hcclBufferSize=800;
    strcpy_s(commConfigTwo.hcclCommName, COMM_NAME_MAX_LENGTH, "comm2");

    unsetenv("HCCL_INTRA_PCIE_ENABLE");
    setenv("HCCL_INTRA_ROCE_ENABLE", "1", 1);
    ret = HcclCommInitClusterInfoMemConfig(const_cast<char*>(clusterString_one.c_str()), rank_ID, &commConfigOne, &commOne);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclCommInitClusterInfoMemConfig(const_cast<char*>(clusterString_two.c_str()), rank_ID, &commConfigTwo, &commTwo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hccl::hcclComm* hcclCommOne = static_cast<hccl::hcclComm *>(commOne);
    hccl::hcclComm* hcclCommTwo = static_cast<hccl::hcclComm *>(commTwo);
    IHcclOneSidedService *iServiceOne = nullptr;
    ret = hcclCommOne->GetOneSidedService(&iServiceOne);
    IHcclOneSidedService *iServiceTwo = nullptr;
    ret = hcclCommTwo->GetOneSidedService(&iServiceTwo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclCommDestroy(commOne);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclCommDestroy(commTwo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    unsetenv("HCCL_INTRA_ROCE_ENABLE");
}

TEST_F(OpbaseTest, ut_HcclCommInitClusterInfoMemConfig_single_comm)
{
    typedef HcclResult (*HcclOneSideServiceCallBack)(std::unique_ptr<hccl::IHcclOneSidedService> &,
    std::unique_ptr<hccl::HcclSocketManager> &, std::unique_ptr<hccl::NotifyPool> &);

    nlohmann::json rank_table = rank_table_910_1server_2rank;
    std::string clusterString = rank_table.dump();

    int ret = HCCL_SUCCESS;
    void* comm;
    u32 rank_ID = 0;

    HcclCommConfig commConfig;
    HcclCommConfigInit(&commConfig);
    commConfig.hcclBufferSize=800;
    strcpy_s(commConfig.hcclCommName, COMM_NAME_MAX_LENGTH, "comm1");

    unsetenv("HCCL_INTRA_PCIE_ENABLE");
    setenv("HCCL_INTRA_ROCE_ENABLE", "1", 1);
    ret = HcclCommInitClusterInfoMemConfig(const_cast<char*>(clusterString.c_str()), rank_ID, &commConfig, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    IHcclOneSidedService *iService = nullptr;
    ret = hcclComm->GetOneSidedService(&iService);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    unsetenv("HCCL_INTRA_ROCE_ENABLE");
}

TEST_F(OpbaseTest, ut_HcclCommInitClusterInfoMemConfig_no_commname)
{
    nlohmann::json rank_table = rank_table_910_1server_2rank;
    std::string clusterString = rank_table.dump();

    int ret = HCCL_SUCCESS;
    void* comm;
    u32 rank_ID = 0;

    HcclCommConfig commConfig;
    HcclCommConfigInit(&commConfig);
    commConfig.hcclBufferSize=800;

    unsetenv("HCCL_INTRA_PCIE_ENABLE");
    setenv("HCCL_INTRA_ROCE_ENABLE", "1", 1);
    ret = HcclCommInitClusterInfoMemConfig(const_cast<char*>(clusterString.c_str()), rank_ID, &commConfig, &comm);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(OpbaseTest, ut_topoInfoExchangeServer_timeout)
{
    setenv("HCCL_CONNECT_TIMEOUT", "5", 1);
    MOCKER(GetExternalInputHcclLinkTimeOut)
    .stubs()
    .will(returnValue(5));
    MOCKER_CPP(&HcclSocket::Accept)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_E_TIMEOUT));
    MOCKER_CPP(&TopoInfoExchangeServer::DisplayConnectedRank)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    HcclIpAddress hostIP("127.0.0.1");
    u32 hostPort = 61111;
    std::vector<HcclIpAddress> whitelist ;
    HcclNetDevCtx netDevCtx;
    std::shared_ptr<HcclSocket> listenSocket;
    const std::string identifier = "TOPPINFOEXCHANGE_TEST";
    TopoInfoExchangeServer topoExServer(hostIP, hostPort, whitelist, netDevCtx, listenSocket, identifier);
    std::map<std::string, std::shared_ptr<HcclSocket>> connectSockets;
    u32 rankSize = 0;
    topoExServer.Connect(connectSockets, rankSize);
    std::shared_ptr<HcclSocket> socket;
    connectSockets.insert({"0001", socket});
    topoExServer.DisplayConnectingStatus(4, 3, connectSockets);

    unsetenv("HCCL_CONNECT_TIMEOUT");
    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_topoInfoExchangeServer_ErrTCP)
{
    setenv("HCCL_CONNECT_TIMEOUT", "5", 1);
     MOCKER(GetExternalInputHcclLinkTimeOut)
    .stubs()
    .will(returnValue(5));
    MOCKER_CPP(&HcclSocket::Accept)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_E_TCP_CONNECT));
    MOCKER_CPP(&TopoInfoExchangeServer::DisplayConnectedRank)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));


    HcclIpAddress hostIP("127.0.0.1");
    u32 hostPort = 61111;
    std::vector<HcclIpAddress> whitelist ;
    HcclNetDevCtx netDevCtx;
    std::shared_ptr<HcclSocket> listenSocket;
    const std::string identifier = "TOPPINFOEXCHANGE_TEST";
    TopoInfoExchangeServer topoExServer(hostIP, hostPort, whitelist, netDevCtx, listenSocket, identifier);
    std::map<std::string, std::shared_ptr<HcclSocket>> connectSockets;
    u32 rankSize = 0;
    topoExServer.Connect(connectSockets, rankSize);

    unsetenv("HCCL_CONNECT_TIMEOUT");
    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_HcclCommInitRootInfoConfig_op_expansion_mode)
{
    HcclRootInfo id;

    DevType deviceType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDeviceType)
    .stubs()
    .with(outBound(deviceType))
    .will(returnValue(HCCL_SUCCESS));

    HcclResult ret = HcclGetRootInfo(&id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclCommConfig config;
    HcclCommConfigInit(&config);

    config.hcclBufferSize = 300;
    config.hcclDeterministic = 1;
    config.hcclOpExpansionMode = 3;
    strcpy_s(config.hcclCommName, COMM_NAME_MAX_LENGTH, "comm1");

    HcclComm newcomm;
    ret = HcclCommInitRootInfoConfig(1, &id, 0, &config, &newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclComm *pComm = static_cast<hcclComm *>(newcomm);
    EXPECT_EQ(pComm->GetConfigInCCLbufferSize(), 300 * 1024 * 1024);
    EXPECT_EQ(pComm->GetConfigOutCCLbufferSize(), 300 * 1024 * 1024);
    EXPECT_EQ(pComm->communicator_->GetDeterministicConfig(), 1);
    EXPECT_EQ(pComm->GetIdentifier(), "comm1");
    EXPECT_EQ(pComm->communicator_->SetDeterministicConfig(config.hcclDeterministic), HCCL_SUCCESS);

    ret = HcclCommDestroy(newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}


TEST_F(OpbaseTest, ut_hcclAllGatherV_capture)
{
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
    int mockModel = 0;
    void *pmockModel = &mockModel;
    MOCKER(aclmdlRICaptureGetInfo)
    .stubs()
    .with(mockcpp::any(), outBoundP(&captureStatus, sizeof(captureStatus)), outBoundP(&pmockModel, sizeof(pmockModel)))
    .will(returnValue(0));

    MOCKER(GetExternalInputHcclEnableEntryLog)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(true));

    DevType deviceType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDeviceType)
    .stubs()
    .with(outBound(deviceType))
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::InitRaResource)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::AllocAlgResource)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::ExecOp)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    nlohmann::json rank_table =
    {
        {"status", "completed"},
        {"deploy_mode", "lab"},
        {"group_count", "1"},
        {"chip_info", "910"},
        {"board_id", "0x0000"},
        {"para_plane_nic_location", "device"},
        {"para_plane_nic_num", "2"},
        {"para_plane_nic_name", {"eth0", "eth1"}},
        {
            "group_list",
            {
                {
                    {"group_name", ""},
                    {"device_num", "2"},
                    {"server_num", "1"},
                    {"instance_count", "2"},
                        {
                            "instance_list",
                            {
                                {   {"rank_id", "0"}, {"server_id", "10.0.0.10"},
                                    {
                                        "devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.12"}}}
                                    }
                                },

                                {   {"rank_id", "1"}, {"server_id", "10.0.0.10"},
                                    {
                                        "devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.14"}}}
                                    }
                                },
                            }
                        },
                        {
                            "server_list",
                            {
                                {
                                    {"server_id", "192.168.10.2"},
                                    {
                                        "para_plane_info",
                                        {{
                                                {"eth1", "192.168.210.2"},
                                            },
                                            {
                                                {"eth0", "192.168.200.2"},
                                            }
                                        }
                                    }

                                },
                            }
                        }
                }
            }
        }
    };

    char file_name_t[] = "./st_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }
    else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s8* recvbuf;
    u64* recvCounts;
    u64* recvDispls;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    u32 rankSize = 0;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;

    // 走1910 4pring
    const char* rank_table_file = "./st_opbase_test.json";
    u32 rank_ID = 0;

    ret = HcclCommInitClusterInfo(rank_table_file, rank_ID, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    ret = hcclComm->GetRankSize(rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sendbuf= (s8*)sal_malloc(count * sizeof(s8));
     sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf= (s8*)sal_malloc(count * sizeof(s8));
     sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    recvCounts= (u64*)sal_malloc(rankSize * sizeof(u64));
     sal_memset(recvCounts, rankSize * sizeof(u64), 0, rankSize * sizeof(u64));
    recvDispls= (u64*)sal_malloc(rankSize * sizeof(u64));
     sal_memset(recvDispls, rankSize * sizeof(u64), 0, rankSize * sizeof(u64));

    for (int j = 0; j < count; j++)
    {
        sendbuf[j] = 2;
    }

    for (int i = 0; i < rankSize; i++)
    {
        recvCounts[i] = count;
        if (i > 0) {
            recvDispls[i] = recvDispls[i-1] + recvCounts[i-1];
        }
    }

    ret = HcclAllGatherVInner(sendbuf, count, recvbuf, recvCounts, recvDispls, HCCL_DATA_TYPE_INT8, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclAllGatherVInner(sendbuf, count, recvbuf, recvCounts, recvDispls, HCCL_DATA_TYPE_INT8, comm, stream);
    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sal_free(sendbuf);
    sal_free(recvbuf);
    sal_free(recvCounts);
    sal_free(recvDispls);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_topoInfoExchangeServer_PreemptPortManager_releaseSocket)
{
    HcclIpAddress hostIP("127.0.0.1");
    u32 hostPort = 61111;
    std::vector<HcclIpAddress> whitelist ;
    HcclNetDevCtx netDevCtx;
    std::shared_ptr<HcclSocket> listenSocket(new (std::nothrow)HcclSocket("test", nullptr,
        hostIP, 0, HcclSocketRole::SOCKET_ROLE_SERVER));

    const std::string identifier = "releaseSocket";
    TopoInfoExchangeServer topoExServer(hostIP, hostPort, whitelist, netDevCtx, listenSocket, identifier);
    MOCKER_CPP(&PreemptPortManager::Release).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    bool portSwitch = 1;
    MOCKER(GetExternalInputHostPortSwitch).stubs().will(returnValue(portSwitch));

    EXPECT_EQ(topoExServer.StopSocketListen(whitelist, hostPort), HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_PreemptDevSocket)
{
    TopoInfoDetect topoDetectAgent;
    MOCKER_CPP(&PreemptPortManager::ListenPreempt)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    HcclIpAddress ipAddr(1694542016);
    u32 port = 0;
    HcclResult ret = topoDetectAgent.PreemptDeviceNicPort(0, 0, ipAddr, port);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hccl::NetDevContext* pNetDevCtx =
        static_cast<hccl::NetDevContext *>(topoDetectAgent.commPortConfig_.devNicListen.second);
    delete pNetDevCtx;
    topoDetectAgent.commPortConfig_.devNicListen.second = nullptr;

    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_PreemptBackupNicSocket)
{
    TopoInfoDetect topoDetectAgent;
    MOCKER_CPP(&PreemptPortManager::ListenPreempt)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    HcclIpAddress ipAddr(1694542016);
    u32 port = 0;
    HcclResult ret = topoDetectAgent.PreemptBackupDeviceNicPort(0, 0, ipAddr, ipAddr, port);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hccl::NetDevContext* pNetDevCtx =
        static_cast<hccl::NetDevContext *>(topoDetectAgent.commPortConfig_.backupDevNicListen.second);
    delete pNetDevCtx;
    topoDetectAgent.commPortConfig_.backupDevNicListen.second = nullptr;

    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_hcomAllGatherV_91093)
{
    MOCKER(GetExternalInputHcclEnableEntryLog)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(true));

    DevType deviceType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType)
    .stubs()
    .with(outBound(deviceType))
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TransportManager::Alloc)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::ExecOp)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));


    nlohmann::json rank_table = rank_table_910_1server_2rank;

    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open())
    {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    }
    else
    {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendbuf;
    s8* recvbuf;
    u64* recvCounts;
    u64* recvDispls;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    u32 rankSize = 4;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;

    // 走1910 4pring
    const char* rank_table_file = "./ut_opbase_test.json";
    u32 rank_ID = 0;

    ret = HcclCommInitClusterInfo(rank_table_file, rank_ID, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sendbuf= (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf= (s8*)sal_malloc(rankSize * count * sizeof(s8));
     sal_memset(recvbuf, rankSize * count * sizeof(s8), 0, rankSize * count * sizeof(s8));

    recvCounts= (u64*)sal_malloc(rankSize * sizeof(u64));
     sal_memset(recvCounts, rankSize * sizeof(u64), 0, rankSize * sizeof(u64));
    recvDispls= (u64*)sal_malloc(rankSize * sizeof(u64));
     sal_memset(recvDispls, rankSize * sizeof(u64), 0, rankSize * sizeof(u64));

    for (int j = 0; j < count; j++)
    {
        sendbuf[j] = 2;
    }

    for (int i = 0; i < rankSize; i++)
    {
        recvCounts[i] = count;
        if (i > 0) {
            recvDispls[i] = recvDispls[i-1] + recvCounts[i-1];
        }
    }

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
    string strTag = "allgatherv";
    ret = hcclComm->AllGatherV(strTag, static_cast<void *>(sendbuf), 2, static_cast<void *>(recvbuf), static_cast<void *>(recvCounts), static_cast<void *>(recvDispls), HCCL_DATA_TYPE_INT8, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sal_free(sendbuf);
    sal_free(recvbuf);
    sal_free(recvCounts);
    sal_free(recvDispls);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    remove(file_name_t);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    GlobalMockObject::verify();
}


TEST_F(OpbaseTest, ut_hcclGetRootInfo_hierarchical_rank_0)
{
    setenv("HCCL_WHITELIST_DISABLE", "1", 1);

     MOCKER_CPP(&TopoInfoExchangeAgent::Connect)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_E_INTERNAL));

    MOCKER(GetExternalInputHcclLinkTimeOut)
    .stubs()
    .will(returnValue(1));

    HcclRootInfo id;
    HcclResult ret = HcclGetRootInfo(&id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclComm newcomm;
    ret = HcclCommInitRootInfo(128*1024, &id, 0, &newcomm);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    GlobalMockObject::verify();
    unsetenv("HCCL_WHITELIST_DISABLE");
}

TEST_F(OpbaseTest, ut_hcclGetRootInfo_hierarchical_rank_1)
{
    setenv("HCCL_WHITELIST_DISABLE", "1", 1);
    GroupLeader_t GrpLeaderInfo;
    nlohmann::json leaderListJson;
    nlohmann::json leaderJson;
    nlohmann::json GroupLeaderJson;

    leaderJson[PROP_NETWORK_IPADDR] = "127.0.0.1";
    leaderJson[PROP_NETWORK_NETWORKPORT] = 60009;
    leaderJson[PROP_NETWORK_IDENTIFIER] = "test";
    leaderJson[PROP_DEPLOY_MODE] = NICDeployment::NIC_DEPLOYMENT_HOST;
    leaderJson[PROP_RANK_ID] = 0;
    leaderListJson.push_back(leaderJson);

    GroupLeaderJson[PROP_RANK_NUM] = 1;
    GroupLeaderJson[PROP_STEP] = 0;
    GroupLeaderJson[PROP_GROUP_LEADER_LIST] = leaderListJson;

    MOCKER_CPP(&TopoInfoExchangeBase::RecvClusterJson)
    .stubs()
    .with(mockcpp::any(), outBound(GroupLeaderJson))
    .will(returnValue(0));

    MOCKER(GetExternalInputHcclLinkTimeOut)
    .stubs()
    .will(returnValue(1));

    HcclRootInfo id;
    HcclResult ret = HcclGetRootInfo(&id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclComm newcomm;
    ret = HcclCommInitRootInfo(128*1024, &id, 0, &newcomm);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    GlobalMockObject::verify();
    unsetenv("HCCL_WHITELIST_DISABLE");
}


TEST_F(OpbaseTest, ut_hcclGetRootInfo_hierarchical_root)
{

    setenv("HCCL_WHITELIST_DISABLE", "1", 1);
    HcclIpAddress hostIP("127.0.0.1");
    u32 hostPort = 61111;
    std::vector<HcclIpAddress> whitelist ;
    HcclNetDevCtx netDevCtx;
    std::shared_ptr<HcclSocket> listenSocket(new (std::nothrow)HcclSocket("test", nullptr,
        hostIP, 0, HcclSocketRole::SOCKET_ROLE_SERVER));

    const std::string identifier = "releaseSocket";
    TopoInfoExchangeServer topoExServer(hostIP, hostPort, whitelist, netDevCtx, listenSocket, identifier);

    MOCKER_CPP(&HcclSocket::Send, HcclResult(HcclSocket::*)(const void *, u64))
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclSocket::Recv, HcclResult(HcclSocket::*)(void *, u32))
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    std::map<std::string, std::shared_ptr<HcclSocket>> connectSockets;
    connectSockets.insert(std::pair<std::string, std::shared_ptr<HcclSocket>>("001", listenSocket));
    GroupLeader_t groupLeader;

    HcclResult ret = topoExServer.RecvGroupLeaderInfo(connectSockets, groupLeader);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
    unsetenv("HCCL_WHITELIST_DISABLE");
}

TEST_F(OpbaseTest, ut_HcclCommInitRootInfo_SetupGroupMember)
{
    TopoInfoDetect topoDetectAgent;
    HcclResult ret;

    // 初始化环境变量，just for st，防止用例间影响
    ResetInitState();
    setenv("HCCL_WHITELIST_DISABLE", "1", 1);

    // 网卡信息在ra_get_ifaddrs接口已初始化（eth0,docker,lo）
    // 不匹配enp，env名称的网卡

    HcclRootHandle rootHandle;
    std::vector<HcclIpAddress> whitelist;
    std::shared_ptr<HcclSocket> socket;
    HcclRootInfo id;
    ret = HcclGetRootInfo(&id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclComm newcomm;
    ret = HcclCommInitRootInfo(1, &id, 0, &newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    s32 sRet = memcpy_s(&rootHandle, HCCL_ROOT_INFO_BYTES, id.internal, sizeof(HcclRootHandle));
    EXPECT_EQ(sRet, 0);

    u32 nRanks = 1;
    u32 myRank = 0;

    MOCKER_CPP(&HcclSocket::Send, HcclResult(HcclSocket::*)(const void *, u64))
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclSocket::Recv, HcclResult(HcclSocket::*)(void *, u32))
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    // 用例本身走超时分支 由于超时时间过长影响线上运行时长且超时时间无法修改只能选择打桩临时
    MOCKER_CPP(&TopoInfoExchangeAgent::GetConnection, HcclResult(TopoInfoExchangeAgent::*)(HcclIpAddress &, u32, std::shared_ptr<HcclSocket> &))
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_E_TIMEOUT));

    ret = topoDetectAgent.SetupGroupMember(nRanks, myRank, rootHandle);
    EXPECT_EQ(ret, HCCL_E_TIMEOUT);

    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_HcclCommInitRootInfo_SetupTopoGroupLeader_error0)
{
    TopoInfoDetect topoDetectAgent;
    HcclResult ret;

    MOCKER_CPP(&TopoInfoExchangeServer::SetupGroupLeader)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_E_INTERNAL));

    MOCKER(hrtResetDevice)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_E_INTERNAL));

    std::vector<HcclIpAddress> whitelist;
    std::shared_ptr<HcclSocket> socket;
    HcclRankHandle groupLeader;
    HcclIpAddress hostIP("127.0.0.1");
    HcclNetDevCtx netDevCtx;

    topoDetectAgent.SetupTopoGroupLeader(0, 0, hostIP, 61111, whitelist, netDevCtx, socket, socket, false);

    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_HcclCommInitRootInfo_SetupTopoGroupLeader_error1)
{
    TopoInfoDetect topoDetectAgent;
    HcclResult ret;

    MOCKER(hrtSetDevice)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_E_INTERNAL));

    std::vector<HcclIpAddress> whitelist;
    std::shared_ptr<HcclSocket> socket;
    HcclRankHandle groupLeader;
    HcclIpAddress hostIP("127.0.0.1");
    HcclNetDevCtx netDevCtx;

    topoDetectAgent.SetupTopoGroupLeader(0, 0, hostIP, 61111, whitelist, netDevCtx, socket, socket, false);

    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_HcclGetTopoDesc)
{
    HcclRootInfo id;
    char group[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {0};

    DevType deviceType = DevType::DEV_TYPE_310P3;
    MOCKER(hrtGetDeviceType)
    .stubs()
    .with(outBound(deviceType))
    .will(returnValue(HCCL_SUCCESS));

    HcclResult ret = HcclGetRootInfo(&id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclComm newcomm;
    ret = HcclCommInitRootInfo(1, &id, 0, &newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclGetCommName(newcomm, group);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclTopoDescs topoDescs[2];
    u32 topoSize = 2;
    ret = HcclGetTopoDesc(newcomm, topoDescs, topoSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_ExchangeCommUserMem_When_Mem_Valid_Expect_ReturnSuccess)
{
    HcclRootInfo id;

    DevType deviceType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType)
    .stubs()
    .with(outBound(deviceType))
    .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&TransportManager::Alloc)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    HcclResult ret = HcclGetRootInfo(&id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclCommConfig config;
    HcclCommConfigInit(&config);

    config.hcclBufferSize = 300;
    config.hcclDeterministic = 1;
    config.hcclOpExpansionMode = 3;
    strcpy_s(config.hcclCommName, COMM_NAME_MAX_LENGTH, "comm1");

    HcclComm comm;
    ret = HcclCommInitRootInfoConfig(1, &id, 0, &config, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u64 size = 1024 * 1024;
    DeviceMem windowMem = DeviceMem::alloc(size);
    void *windowHandle = nullptr;
    uint32_t ranks[] = {0};
    ret = HcclCommRegister(comm, windowMem.ptr(), size, &windowHandle, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclCommExchangeMem(comm, windowHandle, ranks, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclCommDeregister(comm, windowHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    windowMem.free();
    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_ExchangeCommUserMem_When_ranksNum_Valid_Expect_ReturnE_PARA)
{
    HcclRootInfo id;

    DevType deviceType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType)
    .stubs()
    .with(outBound(deviceType))
    .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&TransportManager::Alloc)
    .stubs()
    .will(returnValue(HCCL_SUCCESS));

    HcclResult ret = HcclGetRootInfo(&id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclCommConfig config;
    HcclCommConfigInit(&config);

    config.hcclBufferSize = 300;
    config.hcclDeterministic = 1;
    config.hcclOpExpansionMode = 3;
    strcpy_s(config.hcclCommName, COMM_NAME_MAX_LENGTH, "comm1");

    HcclComm comm;
    ret = HcclCommInitRootInfoConfig(1, &id, 0, &config, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u64 size = 1024 * 1024;
    DeviceMem windowMem = DeviceMem::alloc(size);
    void *windowHandle = nullptr;
    uint32_t ranks[] = {1, 2};
    ret = HcclCommRegister(comm, windowMem.ptr(), size, &windowHandle, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclCommExchangeMem(comm, windowHandle, ranks, 2);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = HcclCommDeregister(comm, windowHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    windowMem.free();
    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_ExchangeCommUserMem_When_usedRDMA_Expect_ReturnE_NOT_SUPPORT)
{
    HcclRootInfo id;

    DevType deviceType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType)
    .stubs()
    .with(outBound(deviceType))
    .will(returnValue(HCCL_SUCCESS));
    MOCKER(GetExternalInputInterHccsDisable)
    .stubs()
    .will(returnValue(true));

    HcclResult ret = HcclGetRootInfo(&id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclCommConfig config;
    HcclCommConfigInit(&config);

    config.hcclBufferSize = 300;
    config.hcclDeterministic = 1;
    config.hcclOpExpansionMode = 3;
    strcpy_s(config.hcclCommName, COMM_NAME_MAX_LENGTH, "comm1");

    HcclComm comm;
    ret = HcclCommInitRootInfoConfig(1, &id, 0, &config, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u64 size = 1024 * 1024;
    DeviceMem windowMem = DeviceMem::alloc(size);
    void *windowHandle = nullptr;
    uint32_t ranks[] = {1, 2};
    ret = HcclCommRegister(comm, windowMem.ptr(), size, &windowHandle, 0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclCommExchangeMem(comm, windowHandle, ranks, 2);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);

    ret = HcclCommDeregister(comm, windowHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    windowMem.free();
    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_cclbuf_op)
{
    hcclComm comm;
    comm.communicator_ = make_unique<HcclCommunicator>();
    HcclComm commoPtr = &comm;

    void *addr;
    uint64_t size = 0;

    auto ret = CommGetLocalCCLBuf(commoPtr, &addr, &size);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    auto ret1 = CommGetRemoteCCLBuf(commoPtr, 0, &addr, &size);
    EXPECT_EQ(ret1, HCCL_SUCCESS);

    auto ret2 = CommGetKFCWorkSpace(commoPtr, &addr, &size);
    EXPECT_EQ(ret2, HCCL_SUCCESS);

    auto ret3 = CommGetCCLBufSizeCfg(commoPtr, &size);
    EXPECT_EQ(ret3, HCCL_SUCCESS);
}

TEST_F(OpbaseTest, ut_cclbuf_op_check_size)
{
    hcclComm comm;
    comm.communicator_ = make_unique<HcclCommunicator>();
    HcclComm commoPtr = &comm;

    setenv("HCCL_BUFFSIZE", "3000", 1);
    InitExternalInput();
    void *addr;
    uint64_t size = 0;
    comm.communicator_->CreateCommCCLbuffer();
    auto ret = CommGetLocalCCLBuf(commoPtr, &addr, &size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(size, 6292504576);
}

TEST_F(OpbaseTest, ut_GroupLeaderAccept_When_Accept_E_Connect_Then_ReturnE_Connect)
{
    HcclResult ret = HCCL_SUCCESS;
    MOCKER_CPP(&HcclSocket::Accept)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_E_TCP_CONNECT));
    MOCKER_CPP(&TopoInfoExchangeServer::DisplayConnectedRank)
    .stubs()
    .with(mockcpp::any())
    .will(returnValue(HCCL_SUCCESS));

    HcclIpAddress hostIP("127.0.0.1");
    u32 hostPort = 61111;
    std::vector<HcclIpAddress> whitelist ;
    HcclNetDevCtx netDevCtx;
    std::shared_ptr<HcclSocket> listenSocket;
    const std::string identifier = "TOPPINFOEXCHANGE_TEST";
    TopoInfoExchangeServer topoExServer(hostIP, hostPort, whitelist, netDevCtx, listenSocket, identifier);
    std::map<std::string, std::shared_ptr<HcclSocket>> connectSockets;
    ret = topoExServer.GroupLeaderConnect(connectSockets);
    EXPECT_EQ(ret, HCCL_E_TCP_CONNECT);

    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_HcclSetCommConfig_NOT_SUPPORT)
{   HcclComm newcomm;
    HcclConfig config{HCCL_DETERMINISTIC};
    HcclConfigValue configValue{0};
    //HCCL_ACCELERATOR
    auto ret = HcclSetCommConfig(newcomm, config, configValue);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_HcclSetCommConfig_HCCL_SUCCESS)
{
    hcclComm test;
    HcclComm newcomm = &test;
    HcclConfig config{HCCL_ACCELERATOR};
    HcclConfigValue configValue{0};

    auto ret = HcclSetCommConfig(newcomm, config, configValue);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_HcclGetCommConfig_NOT_SUPPORT)
{   HcclComm newcomm;
    HcclConfig config{HCCL_DETERMINISTIC};
    HcclConfigValue configValue{0};
    //HCCL_ACCELERATOR
    auto ret = HcclGetCommConfig(newcomm, config, &configValue);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_HcclGetCommConfig_HCCL_SUCCESS)
{
    hcclComm test;
    HcclComm newcomm = &test;
    HcclConfig config{HCCL_ACCELERATOR};
    HcclConfigValue configValue{0};

    auto ret = HcclGetCommConfig(newcomm, config, &configValue);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_HcclCollCommInitAll_When_Normal_Expect_Success)
{
    u32 ndev = 1;
    s32 devices[1] = {0};
    HcclComm comms[1] = {nullptr};

    MOCKER(hrtSetDevice)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtResetDevice)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));

    HcclResult ret = HcclCollCommInitAll(ndev, devices, comms);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_HcclCollCommInitAll_When_NdevZero_Expect_E_PARA)
{
    u32 ndev = 0;
    s32 devices[1] = {0};
    HcclComm comms[1] = {nullptr};

    HcclResult ret = HcclCollCommInitAll(ndev, devices, comms);
    EXPECT_EQ(ret, HCCL_E_PARA);
    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_HcclCollCommInitAll_When_CommsNull_Expect_E_PARA)
{
    u32 ndev = 1;
    s32 devices[1] = {0};

    HcclResult ret = HcclCollCommInitAll(ndev, devices, nullptr);
    EXPECT_EQ(ret, HCCL_E_PARA);
    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_HcclCollCommInitAll_When_DevicesNull_Expect_E_PARA)
{
    u32 ndev = 1;
    HcclComm comms[1] = {nullptr};

    HcclResult ret = HcclCollCommInitAll(ndev, nullptr, comms);
    EXPECT_EQ(ret, HCCL_E_PARA);
    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_HcclGetCollCommAll_When_Normal_Expect_Success)
{
    u32 ndev = 1;
    s32 devices[1] = {0};
    HcclComm comms[1] = {nullptr};

    MOCKER(hrtSetDevice)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtResetDevice)
        .stubs()
        .will(returnValue(HCCL_SUCCESS));

    HcclResult ret = HcclGetCollCommAll(ndev, devices, comms);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_HcclGetCollCommAll_When_NdevZero_Expect_E_PARA)
{
    u32 ndev = 0;
    s32 devices[1] = {0};
    HcclComm comms[1] = {nullptr};

    HcclResult ret = HcclGetCollCommAll(ndev, devices, comms);
    EXPECT_EQ(ret, HCCL_E_PARA);
    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_HcclGetCollCommAll_When_CommsNull_Expect_E_PARA)
{
    u32 ndev = 1;
    s32 devices[1] = {0};

    HcclResult ret = HcclGetCollCommAll(ndev, devices, nullptr);
    EXPECT_EQ(ret, HCCL_E_PARA);
    GlobalMockObject::verify();
}

TEST_F(OpbaseTest, ut_HcclGetCollCommAll_When_DevicesNull_Expect_E_PARA)
{
    u32 ndev = 1;
    HcclComm comms[1] = {nullptr};

    HcclResult ret = HcclGetCollCommAll(ndev, nullptr, comms);
    EXPECT_EQ(ret, HCCL_E_PARA);
    GlobalMockObject::verify();
}
#endif

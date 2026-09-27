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
#include "hccl_communicator_attrs.h"
#include "hccl_impl.h"
#include "coll_alg_operator.h"
#include "all_gather_operator.h"
#include "all_reduce_operator.h"
#include "broadcast_operator.h"
#include "network_manager_pub.h"
#include "topoinfo_struct.h"
#include "preempt_port_manager.h"
#include "network_manager_pub.h"
#undef protected
#undef private

#include "stream_pub.h"
#include "mem_host_pub.h"
#include "mem_device_pub.h"
#include "hccl_comm_pub.h"
#include "gradient_segment.h"
#include "sal.h"

#include "adapter_trace.h"
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
#include "dltrace_function.h"
#include "param_check_pub.h"
#include "callback_thread_manager.h"
#include "dispatcher_pub.h"
#include "dispatcher_pub.h"
#include "hccd_impl_pml.h"
#include "gradient_segment.h"
#include "hcom_private.h"
#include "hcom_pub.h"
#include "hcom_common.h"
#include "rt_external.h"
#include "acl_rt.h"
using namespace std;
using namespace hccl;

class HcclCommTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--HcclCommTest SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--HcclCommTest TearDown--\033[0m" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        MOCKER(GetExternalInputHcclLinkTimeOut).stubs().will(returnValue(1));
        DlTdtFunction::GetInstance().DlTdtFunctionInit();
        DlTraceFunction::GetInstance().DlTraceFunctionInit();
        TsdOpen(1, 2);
        static s32 call_cnt = 0;
        string name = std::to_string(call_cnt++) + "_" + __PRETTY_FUNCTION__;
        ra_set_shm_name(name.c_str());
        MOCKER_CPP(&Heartbeat::RegisterRanks).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&Heartbeat::UnRegisterRanks).stubs().will(returnValue(HCCL_SUCCESS));
    }
    virtual void TearDown()
    {
        TsdClose(1);
        GlobalMockObject::verify();
    }
};

void public_stubs(bool needStubOp)
{
    u32 interfaceVersion = 1;
    MOCKER(hrtRaGetInterfaceVersion)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&interfaceVersion))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtTraceCreateWithAttr).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(hccl::RegisterKernel).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::InitProfiler).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclSocketManager::ServerInit).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    if (needStubOp) {
        MOCKER_CPP(&HcclCommunicator::ExecOp).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    }
}

static void TestConstructParam(HcclCommParams& params, RankTable_t& rankTable)
{
    string commId = "comm ";
    memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1);
    params.rank = 0;
    params.totalRanks = 2;
    params.isHeterogComm = false;
    params.logicDevId = 0;
    params.commWorkMode = WorkMode::HCCL_MODE_NORMAL;
    params.deviceType = DevType::DEV_TYPE_910B;

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

static void
TestConstructParamsByRankInfo(HcclCommParams& params, WorldGroupInfo& groupCommonData, std::vector<RankInfo>& ranks)
{
    string commId = "comm ";
    memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1);
    params.rank = 0;
    params.totalRanks = ranks.size();
    params.isHeterogComm = false;
    params.logicDevId = 0;
    params.commWorkMode = WorkMode::HCCL_MODE_NORMAL;
    params.deviceType = ranks[0].deviceType;

    groupCommonData.deviceType = ranks[0].deviceType;
    groupCommonData.serverId = ranks[0].serverId;
}

/************************************************屏蔽by jiangchen 2019-2-23*******************************/

TEST_F(HcclCommTest, hcclImpl_constructor)
{
    s32 ret = HCCL_SUCCESS;
    HcclCommunicator impl;

    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclCommTest, hcclImpl_check_count_err)
{
    s32 ret = HCCL_SUCCESS;
    HcclCommunicator impl;
    ret = impl.CheckCount(-1);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcclCommTest, hcclImpl_check_count_ok)
{
    s32 ret = HCCL_SUCCESS;
    HcclCommunicator impl;
    ret = impl.CheckCount(1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclCommTest, hccdImpl_check_data_type_ok)
{
    s32 ret = HCCL_SUCCESS;
    HccdImplPml impl;
    ret = impl.CheckDataType(HCCL_DATA_TYPE_INT8, true);
    ret = impl.CheckDataType(HCCL_DATA_TYPE_UINT64, true);
    ret = impl.CheckDataType(HCCL_DATA_TYPE_UINT32, true);
    ret = impl.CheckDataType(HCCL_DATA_TYPE_RESERVED, false);
}

TEST_F(HcclCommTest, hccdImpl_InitTcpMode)
{
    bool isTcp = GetExternalInputHcclIsTcpMode();
    InitExternalInputHeterog();
    s32 ret = HCCL_SUCCESS;
    HccdImplPml impl;
    RankTable_t rankTable;
    MOCKER(GetExternalInputProtocolType).stubs().will(returnValue(ProtocolType::RESERVED));
    ret = impl.InitTcpMode(rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    SetTcpMode(isTcp);
}

TEST_F(HcclCommTest, hcclImpl_check_data_type_ok)
{
    s32 ret = HCCL_SUCCESS;
    HcclCommunicator impl;
    ret = impl.CheckDataType(HCCL_DATA_TYPE_INT8, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclCommTest, hcclImpl_check_data_type_err)
{
    s32 ret = HCCL_SUCCESS;
    HcclCommunicator impl;
    ret = impl.CheckDataType(HCCL_DATA_TYPE_RESERVED, true);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(HcclCommTest, hcclImpl_check_reduce_option_err)
{
    s32 ret = HCCL_SUCCESS;
    HcclCommunicator impl;
    ret = impl.CheckReductionOp(HCCL_REDUCE_RESERVED);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcclCommTest, hcclImpl_check_reduce_option_ok)
{
    s32 ret = HCCL_SUCCESS;
    HcclCommunicator impl;
    ret = impl.CheckReductionOp(HCCL_REDUCE_MIN);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclCommTest, hcclImpl_atomic_init_set)
{
    s32 ret = HCCL_SUCCESS;
    HcclCommunicator impl;

    ret = impl.AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclCommTest, hcclImpl_atomic_init_set_2times)
{
    s32 ret = HCCL_SUCCESS;
    HcclCommunicator impl;

    ret = impl.AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = impl.AtomicInitSet();
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(HcclCommTest, hcclImpl_atomic_init_clear)
{
    s32 ret = HCCL_SUCCESS;
    HcclCommunicator impl;

    ret = impl.AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    impl.AtomicInitClear();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

RankTable_t get_rank_table_rank_nic_host()
{
    RankTable_t rankTable;
    rankTable.deviceNum = 1;

    rankTable.serverNum = 1;
    rankTable.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    rankTable.nicNum = 1;
    rankTable.nicNames.push_back("eth0");
    rankTable.rankNum = 1;

    // rank 信息
    RankInfo_t rank;
    rank.rankId = 0;
    rank.serverIdx = 0;
    rank.serverId = "192.168.1.1";
    rank.deviceInfo.devicePhyId = 0;
    rankTable.rankList.push_back(rank);

    // 服务器信息
    ServerInfo_t server;
    server.serverId = "192.168.1.1";

    NetworkInfo_t net;
    net.ethName = "eth0";
    net.ipAddr = HcclIpAddress("172.17.10.1");
    net.planeID = 0;
    server.networkInfo.push_back(net);
    rankTable.serverList.push_back(server);

    return rankTable;
}

RankTable_t get_rank_table_rank_4p_mesh()
{
    RankTable_t rankTable;
    rankTable.deviceNum = 8;

    rankTable.serverNum = 1;
    rankTable.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    rankTable.nicNum = 1;
    rankTable.nicNames.push_back("eth0");
    rankTable.rankNum = 1;

    for (int i = 0; i < rankTable.deviceNum; ++i) {
        // rank 信息
        RankInfo_t rank;
        rank.rankId = i;
        rank.serverIdx = 0;
        rank.serverId = "192.168.1.1";
        rank.deviceInfo.devicePhyId = i;
        rank.deviceInfo.deviceIp.push_back(HcclIpAddress("172.17.10.1"));
        rankTable.rankList.push_back(rank);
    }

    return rankTable;
}

RankTable_t get_rank_table_rank_nic_device()
{
    RankTable_t rankTable;
    rankTable.deviceNum = 1;

    rankTable.serverNum = 1;
    rankTable.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    rankTable.nicNum = 1;
    rankTable.nicNames.push_back("eth0");
    rankTable.rankNum = 1;

    // rank 信息
    RankInfo_t rank;
    rank.rankId = 0;
    rank.serverIdx = 0;
    rank.serverId = "192.168.1.1";
    rank.deviceInfo.devicePhyId = 0;
    rank.deviceInfo.deviceIp.push_back(HcclIpAddress("172.17.10.1"));
    rankTable.rankList.push_back(rank);

    //    rank.rankId = 1;
    //    rank.deviceInfo.devicePhyId = 1;
    //    rank.deviceInfo.deviceIp = "172.17.10.2";
    //    rankTable.rankList.push_back(rank);
    //
    //    rank.rankId = 2;
    //    rank.deviceInfo.devicePhyId = 2;
    //    rank.deviceInfo.deviceIp = "172.17.10.3";
    //    rankTable.rankList.push_back(rank);
    //
    //    rank.rankId = 3;
    //    rank.deviceInfo.devicePhyId = 3;
    //    rank.deviceInfo.deviceIp = "172.17.10.4";
    //    rankTable.rankList.push_back(rank);

    return rankTable;
}

// hcclComm的API成功用例
TEST_F(HcclCommTest, hcclComm_init_nic_host)
{
    public_stubs(false);
    s32 ret = HCCL_SUCCESS;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclComm comm(0, 0, HCCL_WORLD_GROUP);
    HcclCommParams para;
    ret = hcclComm::GetUniqueId(&para.id);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    para.rank = 0;
    para.totalRanks = 1;
    para.deviceType = DevType::DEV_TYPE_910;
    para.logicDevId = 0;
    para.deviceType = DevType::DEV_TYPE_910;

    RankTable_t rankTable = get_rank_table_rank_nic_device();
    CommConfig commConfig("hccl_world_group");
    ret = comm.init(para, commConfig, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclCommTest, hcclComm_init_inline_reduce_switch)
{
    public_stubs(false);
    s32 ret = HCCL_SUCCESS;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    hcclComm comm;
    HcclCommParams para;
    ret = hcclComm::GetUniqueId(&para.id);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    para.rank = 0;
    para.totalRanks = 1;
    para.deviceType = DevType::DEV_TYPE_910;
    para.logicDevId = 0;
    para.deviceType = DevType::DEV_TYPE_910;
    RankTable_t rankTable = get_rank_table_rank_nic_device();
    CommConfig commConfig("hccl_world_group");
    ret = comm.init(para, commConfig, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclCommTest, hcclComm_get_unique_id)
{
    s32 ret = HCCL_SUCCESS;
    hcclComm comm;
    HcclRootInfo uniqueid;
    ret = comm.GetUniqueId(&uniqueid);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclCommTest, hcclComm_allgather)
{
    public_stubs(true);
    s32 ret = HCCL_SUCCESS;
    s32 rt_ret = RT_ERROR_NONE;

    setenv("HCCL_NET_NAME", "eth0", 1);

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 申请stream
    rtStream_t stream;
    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    // 申请device memory
    aclrtMallocAttrValue moduleIdValue;
    moduleIdValue.moduleId = HCCL;
    aclrtMallocAttribute attrs{.attr = ACL_RT_MEM_ATTR_MODULE_ID, .value = moduleIdValue};
    aclrtMallocConfig cfg{.attrs = &attrs, .numAttrs = 1};
    void* mem_dev_input = NULL;
    aclError aclRet = aclrtMallocWithCfg(&mem_dev_input, 1024, ACL_MEM_TYPE_HIGH_BAND_WIDTH, &cfg);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    void* mem_dev_output = NULL;
    aclRet = aclrtMallocWithCfg(&mem_dev_output, 1024, ACL_MEM_TYPE_HIGH_BAND_WIDTH, &cfg);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // 初始化通信域
    HcclCommParams comm_params;
    comm_params.rank = 0;
    comm_params.totalRanks = 1;
    comm_params.logicDevId = 0;
    comm_params.deviceType = DevType::DEV_TYPE_910;
    ret = hcclComm::GetUniqueId(&comm_params.id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclComm comm;
    RankTable_t rankTable = get_rank_table_rank_nic_device();
    CommConfig commConfig("hccl_world_group");
    ret = comm.init(comm_params, commConfig, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    if (HCCL_SUCCESS == ret) {
        // 执行all-gather
        ret = comm.AllGather("allgather", mem_dev_input, mem_dev_output, 1, HCCL_DATA_TYPE_INT8, stream);
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }

    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, HCCL_SUCCESS);

    rt_ret = aclrtFree(mem_dev_input);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    rt_ret = aclrtFree(mem_dev_output);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    rt_ret = aclrtDestroyStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
}

TEST_F(HcclCommTest, hcclComm_allreduce)
{
    public_stubs(true);
    s32 ret = HCCL_SUCCESS;
    s32 rt_ret = RT_ERROR_NONE;

    setenv("HCCL_LL_THRESHOLD", "2", 1);
    setenv("HCCL_HB_THRESHOLD", "4", 1);
    setenv("HCCL_NET_NAME", "eth0", 1);

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 申请stream
    rtStream_t stream;
    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    // 申请device memory
    aclrtMallocAttrValue moduleIdValue;
    moduleIdValue.moduleId = HCCL;
    aclrtMallocAttribute attrs{.attr = ACL_RT_MEM_ATTR_MODULE_ID, .value = moduleIdValue};
    aclrtMallocConfig cfg{.attrs = &attrs, .numAttrs = 1};
    void* mem_dev_input = NULL;
    aclError aclRet = aclrtMallocWithCfg(&mem_dev_input, 1024, ACL_MEM_TYPE_HIGH_BAND_WIDTH, &cfg);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    void* mem_dev_output = NULL;
    aclRet = aclrtMallocWithCfg(&mem_dev_output, 1024, ACL_MEM_TYPE_HIGH_BAND_WIDTH, &cfg);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // 初始化通信域
    HcclCommParams comm_params;
    comm_params.rank = 0;
    comm_params.totalRanks = 1;
    comm_params.logicDevId = 0;
    comm_params.deviceType = DevType::DEV_TYPE_910;
    ret = hcclComm::GetUniqueId(&comm_params.id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    hcclComm comm;
    RankTable_t rankTable = get_rank_table_rank_nic_device();
    CommConfig commConfig("hccl_world_group");
    ret = comm.init(comm_params, commConfig, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    if (HCCL_SUCCESS == ret) {
        // 执行all-reduce
        ret = comm.AllReduce(
            "allreduce", mem_dev_input, mem_dev_output, 1, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, stream);
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }

    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    rt_ret = aclrtFree(mem_dev_input);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    rt_ret = aclrtFree(mem_dev_output);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    rt_ret = aclrtDestroyStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
}

TEST_F(HcclCommTest, hcclComm_allreduce_mesh)
{
    public_stubs(true);
    s32 ret = HCCL_SUCCESS;
    s32 rt_ret = RT_ERROR_NONE;

    setenv("HCCL_LL_THRESHOLD", "2", 1);
    setenv("HCCL_HB_THRESHOLD", "4", 1);
    setenv("HCCL_NET_NAME", "eth0", 1);

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 申请stream
    rtStream_t stream;
    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    // 申请device memory
    aclrtMallocAttrValue moduleIdValue;
    moduleIdValue.moduleId = HCCL;
    aclrtMallocAttribute attrs{.attr = ACL_RT_MEM_ATTR_MODULE_ID, .value = moduleIdValue};
    aclrtMallocConfig cfg{.attrs = &attrs, .numAttrs = 1};
    void* mem_dev_input = NULL;
    aclError aclRet = aclrtMallocWithCfg(&mem_dev_input, 1024, ACL_MEM_TYPE_HIGH_BAND_WIDTH, &cfg);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    void* mem_dev_output = NULL;
    aclRet = aclrtMallocWithCfg(&mem_dev_output, 1024, ACL_MEM_TYPE_HIGH_BAND_WIDTH, &cfg);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // 初始化通信域
    HcclCommParams comm_params;
    comm_params.rank = 0;
    comm_params.totalRanks = 1;
    comm_params.logicDevId = 0;
    comm_params.deviceType = DevType::DEV_TYPE_910;
    ret = hcclComm::GetUniqueId(&comm_params.id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclComm comm;
    RankTable_t rankTable = get_rank_table_rank_nic_device();
    CommConfig commConfig("hccl_world_group");
    ret = comm.init(comm_params, commConfig, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    if (HCCL_SUCCESS == ret) {
        // 执行all-reduce
        ret = comm.AllReduce(
            "allreduce", mem_dev_input, mem_dev_output, 1, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, stream);
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }

    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    rt_ret = aclrtFree(mem_dev_input);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    rt_ret = aclrtFree(mem_dev_output);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    rt_ret = aclrtDestroyStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
}

#define DEV_NUM_4 4
#define HCCL_ALLREDUCE_DATA_SIZE 10
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
} para_t;

void* inter_all_reduce_task_0(void* parg)
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
    SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
    u64 stream_list_size = 0;
    ret = hcom_info.pComm->GetWorkspaceSubStreamNum(
        para_info->count, para_info->datatype, para_info->op, para_info->identify, stream_list_size);
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
        rt_ret = aclmdlRIBindStream(model, streamList[i], RT_MODEL_WAIT_ACTIVE_STREAM);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    u32 rankSize = 0;
    ret = hcom_info.pComm->GetRankSize(rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u64 memSize = 0;
    ret = hcom_info.pComm->GetWorkspaceMemSize(
        "HcomAllReduce", para_info->count, para_info->datatype, rankSize, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtMalloc(&memptr, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hcom_info.pComm->SetWorkspaceResource("tag_inter_all_reduce_task_0_inter", memptr, memSize, streamList);
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
        "tag_inter_all_reduce_task_0_inter", para_info->sendbuff, para_info->recvbuff, para_info->count,
        para_info->datatype, para_info->op, para_info->stream);

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] task HcclAllReduce fails", hcom_info.params.rank);
    }

    rt_ret = aclrtSynchronizeStream(para_info->stream);
    //--------------Resource destroy----------------//
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = aclmdlRIUnbindStream(model, streamList[i]);
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

TEST_F(HcclCommTest, ut_allreduce_4p_ring)
{
    public_stubs(true);
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
        tid[i] = sal_thread_create("thread", inter_all_reduce_task_0, (void*)&para_info[i]);
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
        hrtFree(result_buff[i]);
        rt_ret = aclrtDestroyStream(stream[i]);

        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }
    set_board_id(0);
    remove(file_name_t);
    EXPECT_EQ(errors, 0);
}

TEST_F(HcclCommTest, hcclComm_broadcast)
{
    public_stubs(true);
    s32 ret = HCCL_SUCCESS;
    s32 rt_ret = RT_ERROR_NONE;

    setenv("HCCL_NET_NAME", "eth0", 1);

    // 申请stream
    rtStream_t stream;
    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    // 申请device memory
    aclrtMallocAttrValue moduleIdValue;
    moduleIdValue.moduleId = HCCL;
    aclrtMallocAttribute attrs{.attr = ACL_RT_MEM_ATTR_MODULE_ID, .value = moduleIdValue};
    aclrtMallocConfig cfg{.attrs = &attrs, .numAttrs = 1};
    void* mem_dev = NULL;
    aclError aclRet = aclrtMallocWithCfg(&mem_dev, 1024, ACL_MEM_TYPE_HIGH_BAND_WIDTH, &cfg);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // 初始化通信域
    HcclCommParams comm_params;
    comm_params.rank = 0;
    comm_params.totalRanks = 1;
    comm_params.logicDevId = 0;
    comm_params.deviceType = DevType::DEV_TYPE_910;
    ret = hcclComm::GetUniqueId(&comm_params.id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclComm comm;
    RankTable_t rankTable = get_rank_table_rank_nic_device();
    CommConfig commConfig("hccl_world_group");
    ret = comm.init(comm_params, commConfig, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    if (HCCL_SUCCESS == ret) {
        // 执行broadcast
        ret = comm.Broadcast("broadcast", mem_dev, 1, HCCL_DATA_TYPE_INT8, 0, stream);
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }

    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    rt_ret = aclrtFree(mem_dev);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    rt_ret = aclrtDestroyStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
}

TEST_F(HcclCommTest, hcclComm_broadcast_mesh)
{
    public_stubs(true);
    s32 ret = HCCL_SUCCESS;
    s32 rt_ret = RT_ERROR_NONE;

    setenv("HCCL_NET_NAME", "eth0", 1);

    // 申请stream
    rtStream_t stream;
    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    // 申请device memory
    aclrtMallocAttrValue moduleIdValue;
    moduleIdValue.moduleId = HCCL;
    aclrtMallocAttribute attrs{.attr = ACL_RT_MEM_ATTR_MODULE_ID, .value = moduleIdValue};
    aclrtMallocConfig cfg{.attrs = &attrs, .numAttrs = 1};
    void* mem_dev = NULL;
    aclError aclRet = aclrtMallocWithCfg(&mem_dev, 1024, ACL_MEM_TYPE_HIGH_BAND_WIDTH, &cfg);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // 初始化通信域
    HcclCommParams comm_params;
    comm_params.rank = 0;
    comm_params.totalRanks = 1;
    comm_params.logicDevId = 0;
    comm_params.deviceType = DevType::DEV_TYPE_910;
    ret = hcclComm::GetUniqueId(&comm_params.id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclComm comm;
    RankTable_t rankTable = get_rank_table_rank_nic_device();
    CommConfig commConfig("hccl_world_group");
    ret = comm.init(comm_params, commConfig, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    if (HCCL_SUCCESS == ret) {
        // 执行broadcast
        ret = comm.Broadcast("broadcast", mem_dev, 1, HCCL_DATA_TYPE_INT8, 0, stream);
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }

    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    rt_ret = aclrtFree(mem_dev);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    rt_ret = aclrtDestroyStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
}

#define HCCL_BROADCAST_DATA_SIZE 10
#define HCCL_BROADCAST_DATA_SLICE 1024

void* inter_broadcast_task_0(void* parg)
{
    s32 portNum = 7;
    MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
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
        HCCL_ERROR("dev[%d] task broadcast fails", para_info->device_id);
    }

    bool swapped;

    SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

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
        "tag_inter_broadcast_task_0_inter", para_info->sendbuff, para_info->count, para_info->datatype, para_info->root,
        para_info->stream);

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("rank[%d] task broadcast fails", hcom_info.params.rank);
    }

    rtError_t rt_ret = RT_ERROR_NONE;
    rt_ret = aclrtSynchronizeStream(para_info->stream);

    if (rt_ret != RT_ERROR_NONE) {
        HCCL_ERROR("rank[%d] task allgather fails", hcom_info.params.rank);
    }

    return (NULL);
}

TEST_F(HcclCommTest, hcclComm_reduce)
{
    public_stubs(true);
    s32 ret = HCCL_SUCCESS;
    s32 rt_ret = RT_ERROR_NONE;

    setenv("HCCL_NET_NAME", "eth0", 1);

    // 申请stream
    rtStream_t stream;
    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    // 申请device memory
    aclrtMallocAttrValue moduleIdValue;
    moduleIdValue.moduleId = HCCL;
    aclrtMallocAttribute attrs{.attr = ACL_RT_MEM_ATTR_MODULE_ID, .value = moduleIdValue};
    aclrtMallocConfig cfg{.attrs = &attrs, .numAttrs = 1};
    void* mem_dev_input = NULL;
    aclError aclRet = aclrtMallocWithCfg(&mem_dev_input, 1024, ACL_MEM_TYPE_HIGH_BAND_WIDTH, &cfg);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    void* mem_dev_output = NULL;
    aclRet = aclrtMallocWithCfg(&mem_dev_output, 1024, ACL_MEM_TYPE_HIGH_BAND_WIDTH, &cfg);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // 初始化通信域
    HcclCommParams comm_params;
    comm_params.rank = 0;
    comm_params.totalRanks = 1;
    comm_params.logicDevId = 0;
    comm_params.deviceType = DevType::DEV_TYPE_910;
    ret = hcclComm::GetUniqueId(&comm_params.id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclComm comm;
    RankTable_t rankTable = get_rank_table_rank_nic_device();
    CommConfig commConfig("hccl_world_group");
    ret = comm.init(comm_params, commConfig, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    if (HCCL_SUCCESS == ret) {
        // 执行reduce
        ret = comm.Reduce(
            "reduce", mem_dev_input + 128, mem_dev_output + 128, 1, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, 0, stream);
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }

    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    rt_ret = aclrtFree(mem_dev_input);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    rt_ret = aclrtFree(mem_dev_output);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    rt_ret = aclrtDestroyStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
}

TEST_F(HcclCommTest, hcclComm_reduce_scatter_ring)
{
    public_stubs(true);
    s32 ret = HCCL_SUCCESS;
    s32 rt_ret = RT_ERROR_NONE;

    setenv("HCCL_NET_NAME", "eth0", 1);

    // 申请stream
    rtStream_t stream;
    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    // 申请device memory
    aclrtMallocAttrValue moduleIdValue;
    moduleIdValue.moduleId = HCCL;
    aclrtMallocAttribute attrs{.attr = ACL_RT_MEM_ATTR_MODULE_ID, .value = moduleIdValue};
    aclrtMallocConfig cfg{.attrs = &attrs, .numAttrs = 1};
    void* mem_dev_input = NULL;
    aclError aclRet = aclrtMallocWithCfg(&mem_dev_input, 1024, ACL_MEM_TYPE_HIGH_BAND_WIDTH, &cfg);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    void* mem_dev_output = NULL;
    aclRet = aclrtMallocWithCfg(&mem_dev_output, 1024, ACL_MEM_TYPE_HIGH_BAND_WIDTH, &cfg);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // 初始化通信域
    HcclCommParams comm_params;
    comm_params.rank = 0;
    comm_params.totalRanks = 1;
    comm_params.logicDevId = 0;
    comm_params.deviceType = DevType::DEV_TYPE_910;
    ret = hcclComm::GetUniqueId(&comm_params.id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclComm comm;
    RankTable_t rankTable = get_rank_table_rank_nic_device();
    CommConfig commConfig("hccl_world_group");
    ret = comm.init(comm_params, commConfig, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
    //-----------------Set Workspace Resource Start------------------//
    u64 stream_list_size = 0;
    u64 streamNum = 0;
    HcclDataType datatype = HCCL_DATA_TYPE_INT32;
    HcclReduceOp opType = HCCL_REDUCE_SUM;
    string identify = "allReduce";
    ret = comm.GetWorkspaceSubStreamNum(streamNum, datatype, opType, identify, stream_list_size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HCCL_INFO("get stream_list_size[%d] success", stream_list_size);
    vector<HcclRtStream> streamList(stream_list_size);
    // 从流bind到model
    rtModel_t model = (void*)1;
    // 生成从stream
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = aclrtCreateStreamWithConfig(&streamList[i], 0, ACL_STREAM_PERSISTENT);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
        rt_ret = aclmdlRIBindStream(model, streamList[i], RT_MODEL_WAIT_ACTIVE_STREAM);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    u32 rankSize = 0;
    ret = comm.GetRankSize(rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u64 memSize = 0;
    ret = comm.GetWorkspaceMemSize(HCCL_KERNEL_OP_TYPE_REDUCESCATTER, 1, HCCL_DATA_TYPE_INT8, rankSize, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HCCL_INFO("HCCL TEST memSize[%llu]", memSize);
    void* memptr = nullptr;
    ret = hrtMalloc(&memptr, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = comm.SetWorkspaceResource("reducescatter", memptr, memSize, streamList);

    EXPECT_EQ(ret, HCCL_SUCCESS);

    //-----------------Set Workspace Resource End------------------//
    if (HCCL_SUCCESS == ret) {
        // 执行reduce-scatter
        ret = comm.ReduceScatter(
            "reducescatter", mem_dev_input + 128, mem_dev_output + 128, 1, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM,
            stream);
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }

    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    rt_ret = aclrtFree(mem_dev_input);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    rt_ret = aclrtFree(mem_dev_output);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    rt_ret = aclrtDestroyStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
}

TEST_F(HcclCommTest, hcclComm_reduce_scatter)
{
    public_stubs(true);
    s32 ret = HCCL_SUCCESS;
    s32 rt_ret = RT_ERROR_NONE;

    setenv("HCCL_NET_NAME", "eth0", 1);

    // 申请stream
    rtStream_t stream;
    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    // 申请device memory
    aclrtMallocAttrValue moduleIdValue;
    moduleIdValue.moduleId = HCCL;
    aclrtMallocAttribute attrs{.attr = ACL_RT_MEM_ATTR_MODULE_ID, .value = moduleIdValue};
    aclrtMallocConfig cfg{.attrs = &attrs, .numAttrs = 1};
    void* mem_dev_input = NULL;
    aclError aclRet = aclrtMallocWithCfg(&mem_dev_input, 1024, ACL_MEM_TYPE_HIGH_BAND_WIDTH, &cfg);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    void* mem_dev_output = NULL;
    aclRet = aclrtMallocWithCfg(&mem_dev_output, 1024, ACL_MEM_TYPE_HIGH_BAND_WIDTH, &cfg);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // 初始化通信域
    HcclCommParams comm_params;
    comm_params.rank = 0;
    comm_params.totalRanks = 1;
    comm_params.logicDevId = 0;
    comm_params.deviceType = DevType::DEV_TYPE_910;
    ret = hcclComm::GetUniqueId(&comm_params.id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclComm comm;
    RankTable_t rankTable = get_rank_table_rank_nic_device();
    CommConfig commConfig("hccl_world_group");
    ret = comm.init(comm_params, commConfig, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
    //-----------------Set Workspace Resource Start------------------//
    u64 stream_list_size = 0;
    u64 streamNum = 0;
    HcclDataType datatype = HCCL_DATA_TYPE_INT32;
    HcclReduceOp opType = HCCL_REDUCE_SUM;
    string identify = "allReduce";
    ret = comm.GetWorkspaceSubStreamNum(streamNum, datatype, opType, identify, stream_list_size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HCCL_INFO("get stream_list_size[%d] success", stream_list_size);
    vector<HcclRtStream> streamList(stream_list_size);
    // 从流bind到model
    rtModel_t model = (void*)1;
    // 生成从stream
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = aclrtCreateStreamWithConfig(&streamList[i], 0, ACL_STREAM_PERSISTENT);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
        rt_ret = aclmdlRIBindStream(model, streamList[i], RT_MODEL_WAIT_ACTIVE_STREAM);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    u32 rankSize = 0;
    ret = comm.GetRankSize(rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u64 memSize = 0;
    ret = comm.GetWorkspaceMemSize(HCCL_KERNEL_OP_TYPE_REDUCESCATTER, 1, HCCL_DATA_TYPE_INT8, rankSize, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HCCL_INFO("HCCL TEST memSize[%llu]", memSize);
    void* memptr = nullptr;
    ret = hrtMalloc(&memptr, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = comm.SetWorkspaceResource("reducescatter", memptr, memSize, streamList);

    EXPECT_EQ(ret, HCCL_SUCCESS);

    //-----------------Set Workspace Resource End------------------//
    if (HCCL_SUCCESS == ret) {
        // 执行reduce-scatter
        ret = comm.ReduceScatter(
            "reducescatter", mem_dev_input + 128, mem_dev_output + 128, 1, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM,
            stream);
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }

    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    rt_ret = aclrtFree(mem_dev_input);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    rt_ret = aclrtFree(mem_dev_output);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    rt_ret = aclrtDestroyStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
}

TEST_F(HcclCommTest, hcclImpl_check_root_err)
{
    public_stubs(true);
    s32 ret = HCCL_SUCCESS;
    HcclCommParams comm_params;
    comm_params.rank = 0;
    comm_params.totalRanks = 1;
    comm_params.deviceType = DevType::DEV_TYPE_910;
    ret = hcclComm::GetUniqueId(&comm_params.id);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    RankTable_t rankTable = get_rank_table_rank_nic_device();
    HcclCommunicator impl;
    ret = impl.Init(comm_params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = impl.CheckUserRank(-1);
    EXPECT_EQ(ret, HCCL_E_PARA);
}
#if 1
TEST_F(HcclCommTest, hcclImpl_check_params_err)
{
    public_stubs(true);

    nlohmann::json rank_table = rank_table_910_1server_4rank;

    char file_name_t[] = "./hcclImpl_check_params_err.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    HcomInfo hcom_info;
    HcclResult ret = HCCL_SUCCESS;
    std::string ranktable_file = file_name_t;
    std::string rankTableM;
    std::string realFilePath;

    rtError_t rt_ret = RT_ERROR_NONE;
    hrtSetDevice(0);

    ret = HcomLoadRanktableFile(ranktable_file.c_str(), rankTableM, realFilePath);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CfgGetClusterInfo(rankTableM, "0", hcom_info.params, hcom_info.rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcom_info.pComm.reset(new (std::nothrow) hccl::hcclComm());
    hcom_info.params.totalRanks = 10;
    hcom_info.params.rank = 12;
    CommConfig commConfig("hccl_world_group");
    ret = hcom_info.pComm->init(hcom_info.params, commConfig, hcom_info.rankTable);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("rank[%d] task all_gather fails", hcom_info.params.rank);
    }

    HcclCommParams params;
    RankTable_t rankTable;
    params.totalRanks = 10;
    params.rank = 12;
    params.logicDevId = 0;
    params.deviceType = DevType::DEV_TYPE_910;
    ret = hcom_info.pComm->init(params, commConfig, rankTable);
    EXPECT_EQ(ret, HCCL_E_PARA);

    std::vector<RankInfo> rankList;
    WorldGroupInfo groupCommonData;

    ret = hcom_info.pComm->init(params, commConfig, rankList, groupCommonData);
    EXPECT_EQ(ret, HCCL_E_PARA);

    params.totalRanks = 10;
    params.rank = 3;
    params.logicDevId = 0;
    params.deviceType = DevType::DEV_TYPE_COUNT;
    ret = hcom_info.pComm->init(params, commConfig, rankList, groupCommonData);
    EXPECT_EQ(ret, HCCL_E_PARA);

    params.totalRanks = 10;
    params.rank = 3;
    params.logicDevId = 0;
    params.deviceType = DevType::DEV_TYPE_COUNT;
    ret = hcom_info.pComm->init(params, commConfig, rankTable);
    remove(file_name_t);
    EXPECT_EQ(ret, HCCL_E_PARA);
}
#endif

TEST_F(HcclCommTest, hcclImpl_get_deviceId)
{
    HcclResult ret;
    HcclCommunicator impl;
    s32 deviceId;
    ret = impl.GetDeviceId(deviceId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclCommTest, hcclImpl_check_board_type_err)
{
    s32 ret = HCCL_SUCCESS;
    HcclCommunicator impl;
    ret = impl.CheckDeviceType(DevType::DEV_TYPE_COUNT);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcclCommTest, hcclImpl_check_2n_err)
{
    bool ret;
    HcclCommunicator impl;
    ret = impl.attrCollector_.Check2N(-2);
    EXPECT_EQ(ret, false);
}

TEST_F(HcclCommTest, hcclImpl_check_dev_count_err)
{
    s32 ret = HCCL_SUCCESS;
    HcclCommunicator impl;
    ret = impl.attrCollector_.CheckDevCount(-2);
    EXPECT_EQ(ret, HCCL_E_PARA);
    ret = impl.attrCollector_.CheckDevCount(5);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcclCommTest, ut_compare_with_serverid)
{
    bool ret = HCCL_SUCCESS;
    HcclCommunicator impl;

    ServerInfo_t left;
    left.serverId = "2";
    ServerInfo_t right;
    right.serverId = "1";
    ret = impl.CompareWithServerId(left, right);
    EXPECT_EQ(ret, 0);
}

TEST_F(HcclCommTest, ut_check_rank_table1)
{
    s32 ret = HCCL_SUCCESS;
    HcclCommunicator impl;
    RankTable_t rankTable = get_rank_table_rank_nic_device();
    rankTable.nicDeploy = NICDeployment::NIC_DEPLOYMENT_RESERVED;
    ServRankInfo_t servRankInfo;
    ret = impl.attrCollector_.CheckRankTable(rankTable, servRankInfo);
    EXPECT_EQ(ret, HCCL_E_PARA);

    rankTable.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    std::vector<RankInfo_t> server1;
    RankInfo_t rank1;
    server1.push_back(rank1);
    server1.push_back(rank1);
    string serverId = "1";
    std::pair<std::string, std::vector<RankInfo_t>> rankInfoPair1(serverId, server1);
    servRankInfo.insert(rankInfoPair1);

    std::vector<RankInfo_t> server2;
    server2.push_back(rank1);
    serverId = "2";
    std::pair<std::string, std::vector<RankInfo_t>> rankInfoPair2(serverId, server2);
    servRankInfo.insert(rankInfoPair2);
    ret = impl.attrCollector_.CheckRankTable(rankTable, servRankInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclCommTest, ut_check_rank_table2)
{
    s32 ret = HCCL_SUCCESS;
    RankTable_t rankTable = get_rank_table_rank_nic_device();
    ServRankInfo_t servRankInfo;
    HcclCommunicator impl;
    std::vector<RankInfo_t> server1;
    RankInfo_t rank1;
    rank1.deviceInfo.devicePhyId = 2;
    server1.push_back(rank1);
    server1.push_back(rank1);
    string serverId = "1";
    std::pair<std::string, std::vector<RankInfo_t>> rankInfoPair1(serverId, server1);
    servRankInfo.insert(rankInfoPair1);

    std::vector<RankInfo_t> server2;
    rank1.deviceInfo.devicePhyId = 1;
    server2.push_back(rank1);
    server2.push_back(rank1);
    serverId = "2";
    std::pair<std::string, std::vector<RankInfo_t>> rankInfoPair2(serverId, server2);
    servRankInfo.insert(rankInfoPair2);
    ret = impl.attrCollector_.CheckRankTable(rankTable, servRankInfo);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcclCommTest, ut_check_rank_table3)
{
    s32 ret = HCCL_SUCCESS;
    RankTable_t rankTable = get_rank_table_rank_nic_device();
    rankTable.deviceNum = 10;
    ServRankInfo_t servRankInfo;
    HcclCommunicator impl;
    std::vector<RankInfo_t> server1;
    RankInfo_t rank1;
    rank1.deviceInfo.devicePhyId = 1;
    server1.push_back(rank1);
    server1.push_back(rank1);
    string serverId = "1";
    std::pair<std::string, std::vector<RankInfo_t>> rankInfoPair1(serverId, server1);
    servRankInfo.insert(rankInfoPair1);

    std::vector<RankInfo_t> server2;
    rank1.deviceInfo.devicePhyId = 1;
    server2.push_back(rank1);
    server2.push_back(rank1);
    serverId = "2";
    std::pair<std::string, std::vector<RankInfo_t>> rankInfoPair2(serverId, server2);
    servRankInfo.insert(rankInfoPair2);
    ret = impl.attrCollector_.CheckRankTable(rankTable, servRankInfo);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcclCommTest, ut_check_rank_table4)
{
    s32 ret = HCCL_SUCCESS;
    RankTable_t rankTable = get_rank_table_rank_nic_device();
    rankTable.deviceNum = 6;
    ServRankInfo_t servRankInfo;
    HcclCommunicator impl;
    std::vector<RankInfo_t> server1;
    RankInfo_t rank1;
    rank1.deviceInfo.devicePhyId = 1;
    server1.push_back(rank1);
    server1.push_back(rank1);
    server1.push_back(rank1);
    string serverId = "1";
    std::pair<std::string, std::vector<RankInfo_t>> rankInfoPair1(serverId, server1);
    servRankInfo.insert(rankInfoPair1);

    std::vector<RankInfo_t> server2;
    rank1.deviceInfo.devicePhyId = 1;
    server2.push_back(rank1);
    server2.push_back(rank1);
    server2.push_back(rank1);
    serverId = "2";
    std::pair<std::string, std::vector<RankInfo_t>> rankInfoPair2(serverId, server2);
    servRankInfo.insert(rankInfoPair2);
    ret = impl.attrCollector_.CheckRankTable(rankTable, servRankInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclCommTest, ut_allreduce_common_char)
{
    public_stubs(true);
    setenv("PROFILING_MODE", "true", 1); // 此用例开启Profiling, 结束后关闭

    nlohmann::json rank_table = rank_table_910_1server_4rank;

    char file_name_t[] = "./ut_allreduce_common_char.json";
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
    s32 count = 256;
    s32 ndev = DEV_NUM_4;
    HcclRootInfo rootInfo;
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
        hrtSetDevice(i);
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
        tid[i] = sal_thread_create("thread", inter_all_reduce_task_0, (void*)&para_info[i]);
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
        hrtFree(result_buff[i]);
        hrtSetDevice(i);
        rt_ret = aclrtDestroyStream(stream[i]);

        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }
    remove(file_name_t);
    EXPECT_EQ(errors, 0);

    setenv("PROFILING_MODE", "false", 1); // 此用例开启Profiling, 结束后关闭
}

TEST_F(HcclCommTest, ut_get_nic_info)
{
    public_stubs(false);
    s32 ret = HCCL_SUCCESS;

    // 初始化通信域
    HcclCommParams comm_params;
    comm_params.rank = 0;
    comm_params.totalRanks = 8;
    comm_params.logicDevId = 0;
    comm_params.deviceType = DevType::DEV_TYPE_910;
    hrtSetDevice(comm_params.logicDevId);
    ret = hcclComm::GetUniqueId(&comm_params.id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclComm comm;
    RankTable_t rankTable = get_rank_table_rank_4p_mesh();
    CommConfig commConfig("hccl_world_group");
    ret = comm.init(comm_params, commConfig, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclCommTest, ut_create_comm_by_alg)
{
    public_stubs(true);
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    u32 ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclImpl* impl = implBase->implAlg_->pimpl_.get();

    DeviceMem output = DeviceMem::alloc(0);
    DeviceMem input = DeviceMem::alloc(0);
    DeviceMem expMem = DeviceMem::alloc(0);
    CommInfo commInfo;
    AlgType algType = AlgType::Reserved();
    ret = impl->CreateCommByAlg("qq", algType, commInfo, input, output, expMem);
    EXPECT_EQ(ret, HCCL_E_PARA);
    implBase = nullptr;
    GlobalMockObject::verify();
}

TEST_F(HcclCommTest, TestCommTypeStar)
{
    public_stubs(true);
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    u32 ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclImpl* impl = implBase->implAlg_->pimpl_.get();

    DeviceMem output = DeviceMem::alloc(0);
    DeviceMem input = DeviceMem::alloc(0);
    DeviceMem expMem = DeviceMem::alloc(0);
    CommInfo commInfo;
    AlgType algType;
    algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_STAR;

    ret = impl->CreateCommByAlg("qq", algType, commInfo, input, output, expMem);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    implBase = nullptr;
    GlobalMockObject::verify();
}

TEST_F(HcclCommTest, TestCommTypeWholeNHR)
{
    public_stubs(true);
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    u32 ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclImpl* impl = implBase->implAlg_->pimpl_.get();

    DeviceMem output = DeviceMem::alloc(0);
    DeviceMem input = DeviceMem::alloc(0);
    DeviceMem expMem = DeviceMem::alloc(0);
    CommInfo commInfo;
    AlgType algType;
    algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR;
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_RESERVED;

    ret = impl->CreateCommByAlg("qq", algType, commInfo, input, output, expMem);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    implBase = nullptr;
    GlobalMockObject::verify();
}

TEST_F(HcclCommTest, TestCommTypeWholeNHRV1)
{
    public_stubs(true);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    u32 ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclImpl* impl = implBase->implAlg_->pimpl_.get();

    DeviceMem output = DeviceMem::alloc(0);
    DeviceMem input = DeviceMem::alloc(0);
    DeviceMem expMem = DeviceMem::alloc(0);
    CommInfo commInfo;
    AlgType algType;
    algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NHR_V1;
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_RESERVED;

    ret = impl->CreateCommByAlg("qq", algType, commInfo, input, output, expMem);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    implBase = nullptr;
    GlobalMockObject::verify();
}

TEST_F(HcclCommTest, TestCommTypeWholeAHC_BROKE)
{
    public_stubs(true);
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    u32 ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclImpl* impl = implBase->implAlg_->pimpl_.get();

    DeviceMem output = DeviceMem::alloc(0);
    DeviceMem input = DeviceMem::alloc(0);
    DeviceMem expMem = DeviceMem::alloc(0);
    CommInfo commInfo;
    AlgType algType;
    algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE;
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_RESERVED;

    ret = impl->CreateCommByAlg("qq", algType, commInfo, input, output, expMem);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    implBase = nullptr;
    GlobalMockObject::verify();
}

TEST_F(HcclCommTest, TestCommTypeWholeNB)
{
    public_stubs(true);
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    u32 ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclImpl* impl = implBase->implAlg_->pimpl_.get();

    DeviceMem output = DeviceMem::alloc(0);
    DeviceMem input = DeviceMem::alloc(0);
    DeviceMem expMem = DeviceMem::alloc(0);
    CommInfo commInfo;
    AlgType algType;
    algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_NB;
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_RESERVED;

    ret = impl->CreateCommByAlg("qq", algType, commInfo, input, output, expMem);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    implBase = nullptr;
    GlobalMockObject::verify();
}

TEST_F(HcclCommTest, ut_create_comm_tag_null)
{
    public_stubs(true);

    s32 ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclImpl* impl = implBase->implAlg_->pimpl_.get();

    DeviceMem output = DeviceMem::alloc(0);
    DeviceMem input = DeviceMem::alloc(0);

    ret = impl->CreateComm("", input, output, AlgType());
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcclCommTest, ut_hccl_create_comm_fail)
{
    public_stubs(false);
    s32 ret = HCCL_SUCCESS;
    s32 rt_ret = RT_ERROR_NONE;

    // 初始化通信域
    HcclCommParams comm_params;
    comm_params.rank = 0;
    comm_params.totalRanks = 1;
    comm_params.logicDevId = 0;
    comm_params.deviceType = DevType::DEV_TYPE_910;
    hrtSetDevice(comm_params.logicDevId);
    ret = hcclComm::GetUniqueId(&comm_params.id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclComm comm;
    RankTable_t rankTable = get_rank_table_rank_nic_device();
    CommConfig commConfig("hccl_world_group");
    ret = comm.init(comm_params, commConfig, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);

    // 申请stream
    rtStream_t stream;
    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    //-----------------Set Workspace Resource Start------------------//
    u64 stream_list_size = 0;
    u64 streamNum = 0;
    HcclDataType datatype = HCCL_DATA_TYPE_INT32;
    HcclReduceOp opType = HCCL_REDUCE_SUM;
    string identify = "allReduce";
    ret = comm.GetWorkspaceSubStreamNum(streamNum, datatype, opType, identify, stream_list_size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HCCL_INFO("get stream_list_size[%d] success", stream_list_size);
    vector<HcclRtStream> streamList(stream_list_size);

    char* charModel = new char;
    rtModel_t model = (void*)charModel;

    // 生成从stream
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = aclrtCreateStreamWithConfig(&streamList[i], 0, ACL_STREAM_PERSISTENT);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
        // 从流bind到model
        rt_ret = aclmdlRIBindStream(model, streamList[i], RT_MODEL_WAIT_ACTIVE_STREAM);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    u32 rankSize = 0;
    ret = comm.GetRankSize(rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u64 memSize = 0;
    ret = comm.GetWorkspaceMemSize("HcomReduceScatter", 1, HCCL_DATA_TYPE_INT8, rankSize, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* memptr = nullptr;
    ret = hrtMalloc(&memptr, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = comm.SetWorkspaceResource("reducescatter", memptr, memSize, streamList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    //-----------------Set Workspace Resource End------------------//

    // 申请device memory
    aclrtMallocAttrValue moduleIdValue;
    moduleIdValue.moduleId = HCCL;
    aclrtMallocAttribute attrs{.attr = ACL_RT_MEM_ATTR_MODULE_ID, .value = moduleIdValue};
    aclrtMallocConfig cfg{.attrs = &attrs, .numAttrs = 1};
    void* mem_dev_input = NULL;
    aclError aclRet = aclrtMallocWithCfg(&mem_dev_input, 1024, ACL_MEM_TYPE_HIGH_BAND_WIDTH, &cfg);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    void* mem_dev_output = NULL;
    aclRet = aclrtMallocWithCfg(&mem_dev_output, 1024, ACL_MEM_TYPE_HIGH_BAND_WIDTH, &cfg);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    MOCKER_CPP(&TransportManager::Alloc).expects(atMost(1)).will(returnValue(HCCL_E_INTERNAL));

    if (HCCL_SUCCESS == ret) {
        // 执行reduce-scatter
        ret = comm.ReduceScatter(
            "reducescatter", mem_dev_input + 1, mem_dev_output + 1, 1, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, stream);
        EXPECT_EQ(ret, HCCL_E_INTERNAL);
    }

    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    GlobalMockObject::verify();

    rt_ret = aclrtFree(mem_dev_input);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    rt_ret = aclrtFree(mem_dev_output);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    rt_ret = aclrtDestroyStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    //--------------Resource destroy----------------//
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = aclmdlRIUnbindStream(model, streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
        rt_ret = aclrtDestroyStream(streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    hrtFree(memptr);

    delete charModel;
    charModel = nullptr;
}

TEST_F(HcclCommTest, ut_comm_inner_create)
{
    public_stubs(true);

    s32 ret = HCCL_SUCCESS;
    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParam(params, rankTable);
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclImpl* impl = implBase->implAlg_->pimpl_.get();

    CommType commType = CommType::COMM_TAG_MAX;
    CommInfo commInfo;
    AlgType algType;
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_8P_RING;
    algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_HD;
    DeviceMem output = DeviceMem::alloc(0);
    DeviceMem input = DeviceMem::alloc(0);
    DeviceMem expMem = DeviceMem::alloc(0);
    HcclResult retOut = HCCL_SUCCESS;
    ErrContextPub err_context;
    err_context.work_stream_id = 0;
    CommParaInfo commParaInfo(COMM_LEVEL1, commType);
    ret = impl->CreateCommThread(err_context, "bb", input, output, expMem, commParaInfo, commInfo.commLevel1, retOut);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcclCommTest, ut_hccl_create_comm_success_ringInner)
{
    public_stubs(true);

    s32 ret = HCCL_SUCCESS;
    s32 rt_ret = RT_ERROR_NONE;

    // 初始化通信域
    HcclCommParams comm_params;
    comm_params.rank = 0;
    comm_params.totalRanks = 1;
    comm_params.logicDevId = 0;
    comm_params.deviceType = DevType::DEV_TYPE_910;
    ret = hcclComm::GetUniqueId(&comm_params.id);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HcclCommunicator impl;
    RankTable_t rankTable = get_rank_table_rank_nic_device();
    ret = impl.Init(comm_params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    CommType commType = CommType::COMM_TAG_RING_INNER;
    CommInfo commInfo;
    AlgType algType;
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_8P_RING;
    algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_HD;
    DeviceMem output = DeviceMem::alloc(0);
    DeviceMem input = DeviceMem::alloc(0);
    DeviceMem expMem = DeviceMem::alloc(0);
    HcclResult retOut = HCCL_SUCCESS;
    hcclImpl* innImpl = impl.implAlg_->pimpl_.get();
    ErrContextPub err_context;
    err_context.work_stream_id = 0;
    CommParaInfo commParaInfo(COMM_LEVEL1, commType);
    ret = innImpl->CreateCommThread(
        err_context, "cc", input, output, expMem, commParaInfo, commInfo.commLevel1, retOut);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclCommTest, ut_get_rank_info_list)
{
    public_stubs(true);

    s32 ret = HCCL_SUCCESS;

    // 初始化通信域
    HcclCommParams comm_params;
    comm_params.rank = 0;
    comm_params.totalRanks = 1;
    comm_params.logicDevId = 0;
    comm_params.deviceType = DevType::DEV_TYPE_910;
    ret = hcclComm::GetUniqueId(&comm_params.id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclComm comm;
    RankTable_t rankTable = get_rank_table_rank_nic_device();
    rankTable.serverList.clear();

    CommConfig commConfig("hccl_world_group");
    ret = comm.init(comm_params, commConfig, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcclCommTest, ut_get_rank_info_list1)
{
    public_stubs(true);

    s32 ret = HCCL_SUCCESS;

    // 初始化通信域
    HcclCommParams comm_params;
    comm_params.rank = 0;
    comm_params.totalRanks = 1;
    comm_params.logicDevId = 0;
    comm_params.deviceType = DevType::DEV_TYPE_910;
    ret = hcclComm::GetUniqueId(&comm_params.id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclComm comm;
    RankTable_t rankTable = get_rank_table_rank_nic_device();
    rankTable.rankList[0].rankId = 2;
    CommConfig commConfig("hccl_world_group");
    ret = comm.init(comm_params, commConfig, rankTable);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

#define DEV_NUM_8 8
TEST_F(HcclCommTest, ut_comm_8pring_1910)
{
    public_stubs(true);

    nlohmann::json rank_table_1910 = rank_table_1server_8rank;
    char file_name_t[] = "./ut_comm_8pring_1910.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(4) << rank_table_1910 << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

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

    HcclReduceOp op = HCCL_REDUCE_MAX;
    s32 count = 128 * 8;
    s32 ndev = DEV_NUM_8;
    HcclRootInfo rootInfo;
    ret = hccl::hcclComm::GetUniqueId(&rootInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HCCL_ERROR("test allreduce");
    set_board_id(0x0000);
    /** 初始化输入输出缓存 */
    for (s32 i = 0; i < ndev; i++) {
        ret = hrtMalloc((void**)&sendbuf[i], (count * sizeof(float)));
        EXPECT_EQ(ret, HCCL_SUCCESS);

        sal_memset(sendbuf[i], count * sizeof(float), 0, count * sizeof(float));

        ret = hrtMalloc((void**)&recvbuf[i], (count * sizeof(float)));
        EXPECT_EQ(ret, HCCL_SUCCESS);

        sal_memset(recvbuf[i], count * sizeof(float), 0, count * sizeof(float));
        result_buff[i] = (float*)sal_malloc(count * sizeof(float));
        sal_memset(result_buff[i], count * sizeof(float), 0, count * sizeof(float));

        inputbuf[i] = sendbuf[i];
        outputbuf[i] = recvbuf[i];
    }

    // sendbuf 赋值
    for (u32 j = 0; j < ndev; j++) {
        for (u32 i = 0; i < count; i++) {
            inputbuf[j][i] = i * 1.0;
        }
    }

    // resultbuf 赋值
    for (s32 i = 0; i < ndev; ++i) {
        for (u32 j = 0; j < count; j++) {
            result_buff[i][j] = j * 1.0;
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
        tid[i] = sal_thread_create("thread", inter_all_reduce_task_0, (void*)&para_info[i]);
        EXPECT_NE(tid[i], (sal_thread_t)NULL);
    }

    for (s32 i = 0; i < ndev; i++) {
        while (sal_thread_is_running(tid[i])) {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
    }

    // 获取stream的操作的同步信号量

    // 获取stream的操作的同步信号量
    for (s32 i = 0; i < ndev; i++) {
        for (s32 j = 0; j < count; j++) {
            float res = result_buff[i][j];
            float recv = outputbuf[i][j];

            if (abs(res - recv) > 1e-6) {
                HCCL_ERROR("rank:%d result[%d]:%f recv[%d]:%f \n", i, j, res, j, recv);
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

    set_board_id(0);
    for (s32 i = 0; i < ndev; i++) {
        hrtFree(sendbuf[i]);
        hrtFree(recvbuf[i]);
        sal_free(result_buff[i]);
        rt_ret = aclrtDestroyStream(stream[i]);

        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }
    remove(file_name_t);
    // EXPECT_EQ(errors, 0);
}

TEST_F(HcclCommTest, hcclComm_ra_init_failed)
{
    public_stubs(false);
    s32 ret = HCCL_SUCCESS;
    s32 rt_ret = RT_ERROR_NONE;

    // 初始化通信域
    HcclCommParams comm_params;
    comm_params.rank = 0;
    comm_params.totalRanks = 8;
    comm_params.logicDevId = 0;
    comm_params.deviceType = DevType::DEV_TYPE_910;
    ret = hcclComm::GetUniqueId(&comm_params.id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclComm comm;
    RankTable_t rankTable = get_rank_table_rank_4p_mesh();

    MOCKER(HrtRaInit).expects(atMost(1)).will(returnValue(HCCL_E_NETWORK));
    CommConfig commConfig("hccl_world_group");
    ret = comm.init(comm_params, commConfig, rankTable);

    EXPECT_EQ(ret, HCCL_E_NETWORK);

    NetworkManager::GetInstance(comm_params.logicDevId).DeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE);
    GlobalMockObject::verify();
}

TEST_F(HcclCommTest, hcclComm_ra_deinit_multi_proccess)
{
    s32 ret = HCCL_SUCCESS;

    u32 devLogicId = 0;

    bool supportMultiProcHCCP = true;
    MOCKER_CPP(&NetworkManager::TsdCapabilityGet)
        .stubs()
        .with(outBound(supportMultiProcHCCP))
        .will(returnValue(HCCL_SUCCESS));

    NetworkManager::GetInstance(devLogicId).deviceNicInitRef_.Ref();

    NetworkManager::GetInstance(devLogicId).DeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE);
    GlobalMockObject::verify();
}

void* inter_reduce_task_0(void* parg)
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

    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sal_memset(hcom_info.params.id.internal, HCCL_ROOT_INFO_BYTES, 0, sizeof(hcom_info.params.id.internal));
    sal_memcpy(hcom_info.params.id.internal, sizeof(HcclRootInfo), &para_info->rootInfo, sizeof(HcclRootInfo));

    hcom_info.pComm.reset(new (std::nothrow) hccl::hcclComm(0, 0, HCCL_WORLD_GROUP));
    rtModel_t model = (void*)1;

    CommConfig commConfig("hccl_world_group");
    ret = hcom_info.pComm->init(hcom_info.params, commConfig, hcom_info.rankTable);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("dev[%d] task reduce fails", para_info->device_id);
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

    HCCL_DEBUG("all %d  ranks init ok ,then reduce", hcom_info.params.totalRanks);

    //-----------------Set Workspace Resource Start------------------//
    u64 stream_list_size = 0;
    ret = hcom_info.pComm->GetWorkspaceSubStreamNum(
        para_info->count, para_info->datatype, para_info->op, para_info->identify, stream_list_size);
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
        HCCL_INFO("HCCL TEST NNNNNN i[%d]", i);

        rt_ret = aclmdlRIBindStream(model, streamList[i], RT_MODEL_WAIT_ACTIVE_STREAM);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }

    u64 memSize = 0;
    ret = hcom_info.pComm->GetWorkspaceMemSize(
        HCCL_KERNEL_OP_TYPE_REDUCE, para_info->count, para_info->datatype, rankSize, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* memptr = nullptr;
    ret = hrtMalloc(&memptr, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hcom_info.pComm->SetWorkspaceResource("tag_inter_reduce_task_0_inter", memptr, memSize, streamList);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    //-----------------Set Workspace Resource End------------------//
    ret = hcom_info.pComm->Reduce(
        "tag_inter_reduce_task_0_inter", para_info->sendbuff, para_info->recvbuff, para_info->count,
        para_info->datatype, para_info->op, para_info->root, para_info->stream);

    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("rank[%d] task reduce fails", hcom_info.params.rank);
    }

    rt_ret = RT_ERROR_NONE;
    rt_ret = aclrtSynchronizeStream(para_info->stream);
    for (s32 i = 0; i < stream_list_size; i++) {
        rt_ret = aclmdlRIUnbindStream(model, streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);

        rt_ret = aclrtDestroyStream(streamList[i]);
        EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    }
    if (rt_ret != RT_ERROR_NONE) {
        HCCL_ERROR("rank[%d] task allgather fails", hcom_info.params.rank);
    }

    return (NULL);
}
void* inter_reduce_scatter_task_0(void* parg)
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
        "tag_inter_reduce_scatter_task_0_inter", para_info->sendbuff, para_info->recvbuff, para_info->count,
        para_info->datatype, para_info->op, para_info->stream);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("rank[%d] task reduce_scatter fails", hcom_info.params.rank);
    }

    rtError_t rt_ret = RT_ERROR_NONE;
    rt_ret = aclrtSynchronizeStream(para_info->stream);

    if (rt_ret != RT_ERROR_NONE) {
        HCCL_ERROR("rank[%d] task allgather fails", hcom_info.params.rank);
    }
    return (nullptr);
}

TEST_F(HcclCommTest, ut_comm_reduce_V80_inline)
{
    public_stubs(true);

    nlohmann::json rank_table = rank_table_910_1server_4rank;

    char file_name_t[] = "./ut_reduce_inter_sum_float_slice.json";
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
    float* inputbuf[DEV_NUM_4];
    float* outputbuf[DEV_NUM_4];

    s32 sync_value = 0;

    rtStream_t stream[DEV_NUM_4];
    sal_thread_t tid[DEV_NUM_4];
    para_t para_info[DEV_NUM_4];

    HcclDataType datatype = HCCL_DATA_TYPE_FP32;

    HcclReduceOp op = HCCL_REDUCE_SUM;
    //  s32 count = 100;
    s32 count = 10;
    s32 ndev = DEV_NUM_4;

    HcclRootInfo rootInfo;
    ret = hccl::hcclComm::GetUniqueId(&rootInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    /** 初始化输入输出缓存 */
    for (s32 i = 0; i < ndev; i++) {
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

    // sendbuf 赋值
    for (u32 j = 0; j < ndev; j++) {
        for (u32 i = 0; i < count; i++) {
            sendbuf[j][i] = 1.0;
        }
    }

    // resultbuf 赋值

    for (u32 j = 0; j < count; j++) {
        result_buff[0][j] = 4.0;
    }

    for (s32 i = 0; i < ndev; ++i) {
        hrtSetDevice(i);
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
        para_info[i].offline = false;
    }

    // 创建每个Dev的allreduce任务线程
    for (s32 i = 0; i < ndev; ++i) {
        tid[i] = sal_thread_create("thread", inter_reduce_task_0, (void*)&para_info[i]);
        EXPECT_NE(tid[i], (sal_thread_t)NULL);
    }

    for (s32 i = 0; i < ndev; ++i) {
        while (sal_thread_is_running(tid[i])) {
            SaluSleep(SAL_MILLISECOND_USEC * 10);
        }
    }

    // 获取stream的操作的同步信号量

    for (s32 i = 0; i < count; i++) {
        float res = result_buff[0][i];
        float recv = recvbuf[0][i];

        if (abs(res - recv) > 1e-6) {
            HCCL_ERROR(" recvbuf[%f] result_buff[%f] \n", recv, res);
            errors++;
            break;
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
    remove(file_name_t);
}

TEST_F(HcclCommTest, hcclComm_allreduce_external_input)
{
    public_stubs(true);

    s32 ret = HCCL_SUCCESS;
    s32 rt_ret = RT_ERROR_NONE;

    setenv("HCCL_LL_THRESHOLD", "2", 1);
    setenv("HCCL_HB_THRESHOLD", "4", 1);
    setenv("HCCL_NET_NAME", "eth0", 1);

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 申请stream
    rtStream_t stream;
    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    // 申请device memory
    aclrtMallocAttrValue moduleIdValue;
    moduleIdValue.moduleId = HCCL;
    aclrtMallocAttribute attrs{.attr = ACL_RT_MEM_ATTR_MODULE_ID, .value = moduleIdValue};
    aclrtMallocConfig cfg{.attrs = &attrs, .numAttrs = 1};
    void* mem_dev_input = NULL;
    aclError aclRet = aclrtMallocWithCfg(&mem_dev_input, 1024, ACL_MEM_TYPE_HIGH_BAND_WIDTH, &cfg);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    void* mem_dev_output = NULL;
    aclRet = aclrtMallocWithCfg(&mem_dev_output, 1024, ACL_MEM_TYPE_HIGH_BAND_WIDTH, &cfg);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ResetInitState();

    // 初始化通信域
    HcclCommParams comm_params;
    comm_params.rank = 0;
    comm_params.totalRanks = 1;
    comm_params.logicDevId = 0;
    comm_params.deviceType = DevType::DEV_TYPE_910;
    ret = hcclComm::GetUniqueId(&comm_params.id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclComm comm;
    RankTable_t rankTable = get_rank_table_rank_nic_device();
    CommConfig commConfig("hccl_world_group");
    ret = comm.init(comm_params, commConfig, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    if (HCCL_SUCCESS == ret) {
        // 执行all-reduce
        ret = comm.AllReduce(
            "allreduce", mem_dev_input, mem_dev_output, 1, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_MAX, stream);
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }

    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    rt_ret = aclrtFree(mem_dev_input);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    rt_ret = aclrtFree(mem_dev_output);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    rt_ret = aclrtDestroyStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    ResetInitState();
}

void get_ranks(std::vector<RankInfo>& rank_vector)
{
    RankInfo tmp_para_0;

    tmp_para_0.userRank = 0;
    tmp_para_0.devicePhyId = 0;
    tmp_para_0.deviceType = DevType::DEV_TYPE_910;
    tmp_para_0.serverIdx = 0;
    tmp_para_0.serverId = "10.0.0.10";
    tmp_para_0.nicIp.push_back(HcclIpAddress("192.168.0.11"));
    tmp_para_0.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_1;

    tmp_para_1.userRank = 1;
    tmp_para_1.devicePhyId = 1;
    tmp_para_1.deviceType = DevType::DEV_TYPE_910;
    tmp_para_1.serverIdx = 0;
    tmp_para_1.serverId = "10.0.0.10";
    tmp_para_1.nicIp.push_back(HcclIpAddress("192.168.0.12"));
    tmp_para_1.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_2;

    tmp_para_2.userRank = 2;
    tmp_para_2.devicePhyId = 2;
    tmp_para_2.deviceType = DevType::DEV_TYPE_910;
    tmp_para_2.serverIdx = 0;
    tmp_para_2.serverId = "10.0.0.10";
    tmp_para_2.nicIp.push_back(HcclIpAddress("192.168.0.13"));
    tmp_para_2.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_3;

    tmp_para_3.userRank = 3;
    tmp_para_3.devicePhyId = 3;
    tmp_para_3.deviceType = DevType::DEV_TYPE_910;
    tmp_para_3.serverIdx = 0;
    tmp_para_3.serverId = "10.0.0.10";
    tmp_para_3.nicIp.push_back(HcclIpAddress("192.168.0.14"));
    tmp_para_3.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    rank_vector.push_back(tmp_para_0);
    rank_vector.push_back(tmp_para_1);
    rank_vector.push_back(tmp_para_2);
    rank_vector.push_back(tmp_para_3);
    return;
}

void get_ranks_1server_1dev(std::vector<RankInfo>& rank_vector)
{
    RankInfo tmp_para_0;

    tmp_para_0.userRank = 0;
    tmp_para_0.devicePhyId = 0;
    tmp_para_0.deviceType = DevType::DEV_TYPE_910;
    tmp_para_0.serverIdx = 0;
    tmp_para_0.serverId = "10.0.0.10";
    tmp_para_0.nicIp.push_back(HcclIpAddress("192.168.0.11"));
    tmp_para_0.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    rank_vector.push_back(tmp_para_0);

    return;
}

void get_ranks_1server_2dev(std::vector<RankInfo>& rank_vector)
{
    RankInfo tmp_para_0;

    tmp_para_0.userRank = 0;
    tmp_para_0.devicePhyId = 0;
    tmp_para_0.deviceType = DevType::DEV_TYPE_910;
    tmp_para_0.serverIdx = 0;
    tmp_para_0.serverId = "10.0.0.10";
    tmp_para_0.nicIp.push_back(HcclIpAddress("192.168.0.11"));
    tmp_para_0.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_1;

    tmp_para_1.userRank = 1;
    tmp_para_1.devicePhyId = 1;
    tmp_para_1.deviceType = DevType::DEV_TYPE_910;
    tmp_para_1.serverIdx = 0;
    tmp_para_1.serverId = "10.0.0.10";
    tmp_para_1.nicIp.push_back(HcclIpAddress("192.168.0.12"));
    tmp_para_1.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    rank_vector.push_back(tmp_para_0);
    rank_vector.push_back(tmp_para_1);
    return;
}

void get_ranks_1server_3dev(std::vector<RankInfo>& rank_vector)
{
    RankInfo tmp_para_0;

    tmp_para_0.userRank = 0;
    tmp_para_0.devicePhyId = 0;
    tmp_para_0.deviceType = DevType::DEV_TYPE_910;
    tmp_para_0.serverIdx = 0;
    tmp_para_0.serverId = "10.0.0.10";
    tmp_para_0.nicIp.push_back(HcclIpAddress("192.168.0.11"));
    tmp_para_0.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_1;

    tmp_para_1.userRank = 1;
    tmp_para_1.devicePhyId = 1;
    tmp_para_1.deviceType = DevType::DEV_TYPE_910;
    tmp_para_1.serverIdx = 0;
    tmp_para_1.serverId = "10.0.0.10";
    tmp_para_1.nicIp.push_back(HcclIpAddress("192.168.0.12"));
    tmp_para_1.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_2;

    tmp_para_2.userRank = 2;
    tmp_para_2.devicePhyId = 2;
    tmp_para_2.deviceType = DevType::DEV_TYPE_910;
    tmp_para_2.serverIdx = 0;
    tmp_para_2.serverId = "10.0.0.10";
    tmp_para_2.nicIp.push_back(HcclIpAddress("192.168.0.13"));
    tmp_para_2.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    rank_vector.push_back(tmp_para_0);
    rank_vector.push_back(tmp_para_1);
    rank_vector.push_back(tmp_para_2);
    return;
}

void get_ranks_1server_4dev(std::vector<RankInfo>& rank_vector)
{
    RankInfo tmp_para_0;

    tmp_para_0.userRank = 0;
    tmp_para_0.devicePhyId = 0;
    tmp_para_0.deviceType = DevType::DEV_TYPE_910;
    tmp_para_0.serverIdx = 0;
    tmp_para_0.serverId = "10.0.0.10";
    tmp_para_0.nicIp.push_back(HcclIpAddress("192.168.0.11"));
    tmp_para_0.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_1;

    tmp_para_1.userRank = 1;
    tmp_para_1.devicePhyId = 1;
    tmp_para_1.deviceType = DevType::DEV_TYPE_910;
    tmp_para_1.serverIdx = 0;
    tmp_para_1.serverId = "10.0.0.10";
    tmp_para_1.nicIp.push_back(HcclIpAddress("192.168.0.12"));
    tmp_para_1.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_2;

    tmp_para_2.userRank = 2;
    tmp_para_2.devicePhyId = 2;
    tmp_para_2.deviceType = DevType::DEV_TYPE_910;
    tmp_para_2.serverIdx = 0;
    tmp_para_2.serverId = "10.0.0.10";
    tmp_para_2.nicIp.push_back(HcclIpAddress("192.168.0.13"));
    tmp_para_2.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_3;

    tmp_para_3.userRank = 3;
    tmp_para_3.devicePhyId = 3;
    tmp_para_3.deviceType = DevType::DEV_TYPE_910;
    tmp_para_3.serverIdx = 0;
    tmp_para_3.serverId = "10.0.0.10";
    tmp_para_3.nicIp.push_back(HcclIpAddress("192.168.0.14"));
    tmp_para_3.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    rank_vector.push_back(tmp_para_0);
    rank_vector.push_back(tmp_para_1);
    rank_vector.push_back(tmp_para_2);
    rank_vector.push_back(tmp_para_3);
    return;
}

void get_ranks_1server_8dev(std::vector<RankInfo>& rank_vector)
{
    RankInfo tmp_para_0;

    tmp_para_0.userRank = 0;
    tmp_para_0.devicePhyId = 0;
    tmp_para_0.deviceType = DevType::DEV_TYPE_910;
    tmp_para_0.serverIdx = 0;
    tmp_para_0.serverId = "10.0.0.10";
    tmp_para_0.nicIp.push_back(HcclIpAddress("192.168.0.11"));
    tmp_para_0.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_1;

    tmp_para_1.userRank = 1;
    tmp_para_1.devicePhyId = 1;
    tmp_para_1.deviceType = DevType::DEV_TYPE_910;
    tmp_para_1.serverIdx = 0;
    tmp_para_1.serverId = "10.0.0.10";
    tmp_para_1.nicIp.push_back(HcclIpAddress("192.168.0.12"));
    tmp_para_1.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_2;

    tmp_para_2.userRank = 2;
    tmp_para_2.devicePhyId = 2;
    tmp_para_2.deviceType = DevType::DEV_TYPE_910;
    tmp_para_2.serverIdx = 0;
    tmp_para_2.serverId = "10.0.0.10";
    tmp_para_2.nicIp.push_back(HcclIpAddress("192.168.0.13"));
    tmp_para_2.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_3;

    tmp_para_3.userRank = 3;
    tmp_para_3.devicePhyId = 3;
    tmp_para_3.deviceType = DevType::DEV_TYPE_910;
    tmp_para_3.serverIdx = 0;
    tmp_para_3.serverId = "10.0.0.10";
    tmp_para_3.nicIp.push_back(HcclIpAddress("192.168.0.14"));
    tmp_para_3.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_4;

    tmp_para_4.userRank = 4;
    tmp_para_4.devicePhyId = 4;
    tmp_para_4.deviceType = DevType::DEV_TYPE_910;
    tmp_para_4.serverIdx = 0;
    tmp_para_4.serverId = "10.0.0.10";
    tmp_para_4.nicIp.push_back(HcclIpAddress("192.168.0.15"));
    tmp_para_4.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_5;

    tmp_para_5.userRank = 5;
    tmp_para_5.devicePhyId = 5;
    tmp_para_5.deviceType = DevType::DEV_TYPE_910;
    tmp_para_5.serverIdx = 0;
    tmp_para_5.serverId = "10.0.0.10";
    tmp_para_5.nicIp.push_back(HcclIpAddress("192.168.0.16"));
    tmp_para_5.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_6;

    tmp_para_6.userRank = 6;
    tmp_para_6.devicePhyId = 6;
    tmp_para_6.deviceType = DevType::DEV_TYPE_910;
    tmp_para_6.serverIdx = 0;
    tmp_para_6.serverId = "10.0.0.10";
    tmp_para_6.nicIp.push_back(HcclIpAddress("192.168.0.17"));
    tmp_para_6.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_7;

    tmp_para_7.userRank = 7;
    tmp_para_7.devicePhyId = 7;
    tmp_para_7.deviceType = DevType::DEV_TYPE_910;
    tmp_para_7.serverIdx = 0;
    tmp_para_7.serverId = "10.0.0.10";
    tmp_para_7.nicIp.push_back(HcclIpAddress("192.168.0.18"));
    tmp_para_7.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

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

void get_ranks_7server_1dev(std::vector<RankInfo>& rank_vector)
{
    RankInfo tmp_para_0;

    tmp_para_0.userRank = 0;
    tmp_para_0.devicePhyId = 0;
    tmp_para_0.deviceType = DevType::DEV_TYPE_910;
    tmp_para_0.serverIdx = 0;
    tmp_para_0.serverId = "10.0.0.10";
    tmp_para_0.nicIp.push_back(HcclIpAddress("192.168.0.11"));
    tmp_para_0.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_1;

    tmp_para_1.userRank = 1;
    tmp_para_1.devicePhyId = 1;
    tmp_para_1.deviceType = DevType::DEV_TYPE_910;
    tmp_para_1.serverIdx = 1;
    tmp_para_1.serverId = "10.0.1.10";
    tmp_para_1.nicIp.push_back(HcclIpAddress("192.168.0.12"));
    tmp_para_1.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_2;

    tmp_para_2.userRank = 2;
    tmp_para_2.devicePhyId = 2;
    tmp_para_2.deviceType = DevType::DEV_TYPE_910;
    tmp_para_2.serverIdx = 2;
    tmp_para_2.serverId = "10.0.2.10";
    tmp_para_2.nicIp.push_back(HcclIpAddress("192.168.0.13"));
    tmp_para_2.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_3;

    tmp_para_3.userRank = 3;
    tmp_para_3.devicePhyId = 3;
    tmp_para_3.deviceType = DevType::DEV_TYPE_910;
    tmp_para_3.serverIdx = 3;
    tmp_para_3.serverId = "10.0.3.10";
    tmp_para_3.nicIp.push_back(HcclIpAddress("192.168.0.14"));
    tmp_para_3.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_4;

    tmp_para_4.userRank = 4;
    tmp_para_4.devicePhyId = 4;
    tmp_para_4.deviceType = DevType::DEV_TYPE_910;
    tmp_para_4.serverIdx = 4;
    tmp_para_4.serverId = "10.0.4.10";
    tmp_para_4.nicIp.push_back(HcclIpAddress("192.168.0.15"));
    tmp_para_4.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_5;

    tmp_para_5.userRank = 5;
    tmp_para_5.devicePhyId = 5;
    tmp_para_5.deviceType = DevType::DEV_TYPE_910;
    tmp_para_5.serverIdx = 5;
    tmp_para_5.serverId = "10.0.5.10";
    tmp_para_5.nicIp.push_back(HcclIpAddress("192.168.0.16"));
    tmp_para_5.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_6;

    tmp_para_6.userRank = 6;
    tmp_para_6.devicePhyId = 6;
    tmp_para_6.deviceType = DevType::DEV_TYPE_910;
    tmp_para_6.serverIdx = 6;
    tmp_para_6.serverId = "10.0.6.10";
    tmp_para_6.nicIp.push_back(HcclIpAddress("192.168.0.17"));
    tmp_para_6.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    rank_vector.push_back(tmp_para_0);
    rank_vector.push_back(tmp_para_1);
    rank_vector.push_back(tmp_para_2);
    rank_vector.push_back(tmp_para_3);
    rank_vector.push_back(tmp_para_4);
    rank_vector.push_back(tmp_para_5);
    rank_vector.push_back(tmp_para_6);
    return;
}

void get_ranks_8server_1dev(std::vector<RankInfo>& rank_vector)
{
    RankInfo tmp_para_0;

    tmp_para_0.userRank = 0;
    tmp_para_0.devicePhyId = 0;
    tmp_para_0.deviceType = DevType::DEV_TYPE_910;
    tmp_para_0.serverIdx = 0;
    tmp_para_0.serverId = "10.0.0.10";
    tmp_para_0.nicIp.push_back(HcclIpAddress("192.168.0.11"));
    tmp_para_0.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_1;

    tmp_para_1.userRank = 1;
    tmp_para_1.devicePhyId = 1;
    tmp_para_1.deviceType = DevType::DEV_TYPE_910;
    tmp_para_1.serverIdx = 1;
    tmp_para_1.serverId = "10.0.1.10";
    tmp_para_1.nicIp.push_back(HcclIpAddress("192.168.0.12"));
    tmp_para_1.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_2;

    tmp_para_2.userRank = 2;
    tmp_para_2.devicePhyId = 2;
    tmp_para_2.deviceType = DevType::DEV_TYPE_910;
    tmp_para_2.serverIdx = 2;
    tmp_para_2.serverId = "10.0.2.10";
    tmp_para_2.nicIp.push_back(HcclIpAddress("192.168.0.13"));
    tmp_para_2.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_3;

    tmp_para_3.userRank = 3;
    tmp_para_3.devicePhyId = 3;
    tmp_para_3.deviceType = DevType::DEV_TYPE_910;
    tmp_para_3.serverIdx = 3;
    tmp_para_3.serverId = "10.0.3.10";
    tmp_para_3.nicIp.push_back(HcclIpAddress("192.168.0.14"));
    tmp_para_3.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_4;

    tmp_para_4.userRank = 4;
    tmp_para_4.devicePhyId = 4;
    tmp_para_4.deviceType = DevType::DEV_TYPE_910;
    tmp_para_4.serverIdx = 4;
    tmp_para_4.serverId = "10.0.4.10";
    tmp_para_4.nicIp.push_back(HcclIpAddress("192.168.0.15"));
    tmp_para_4.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_5;

    tmp_para_5.userRank = 5;
    tmp_para_5.devicePhyId = 5;
    tmp_para_5.deviceType = DevType::DEV_TYPE_910;
    tmp_para_5.serverIdx = 5;
    tmp_para_5.serverId = "10.0.5.10";
    tmp_para_5.nicIp.push_back(HcclIpAddress("192.168.0.16"));
    tmp_para_5.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_6;

    tmp_para_6.userRank = 6;
    tmp_para_6.devicePhyId = 6;
    tmp_para_6.deviceType = DevType::DEV_TYPE_910;
    tmp_para_6.serverIdx = 6;
    tmp_para_6.serverId = "10.0.6.10";
    tmp_para_6.nicIp.push_back(HcclIpAddress("192.168.0.17"));
    tmp_para_6.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

    RankInfo tmp_para_7;

    tmp_para_7.userRank = 7;
    tmp_para_7.devicePhyId = 7;
    tmp_para_7.deviceType = DevType::DEV_TYPE_910;
    tmp_para_7.serverIdx = 7;
    tmp_para_7.serverId = "10.0.7.10";
    tmp_para_7.nicIp.push_back(HcclIpAddress("192.168.0.18"));
    tmp_para_7.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;

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

TEST_F(HcclCommTest, ut_SetAlgType)
{
    public_stubs(true);
    s32 ret = HCCL_SUCCESS;

    std::vector<RankInfo> ranks;
    get_ranks(ranks);

    HcclCommParams params;
    WorldGroupInfo groupCommonData;
    TestConstructParamsByRankInfo(params, groupCommonData, ranks);

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, ranks, groupCommonData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 serverNum = implBase->serverNum_;
    hcclImpl* impl = implBase->implAlg_->pimpl_.get();
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    std::map<HcclCMDType, AlgType> algType;
    ret = algConfigurator->SelectAlgType(serverNum, DevType::DEV_TYPE_COUNT, algType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase = nullptr;
    GlobalMockObject::verify();
}

TEST_F(HcclCommTest, ut_SetAlgType_1server_1dev_ring_hd)
{
    public_stubs(true);

    HcclResult ret;
    // setenv("HCCL_ALGORITHM", "level0:ring;level1:H-D_R", 1);
    // ResetInitState();
    std::string algo = "level0:ring;level1:H-D_R";
    ret = SetHcclAlgoConfig(algo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> ranks;
    get_ranks_1server_1dev(ranks);
    HcclCommParams params;
    WorldGroupInfo groupCommonData;
    TestConstructParamsByRankInfo(params, groupCommonData, ranks);

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, ranks, groupCommonData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 serverNum = implBase->serverNum_;
    hcclImpl* impl = implBase->implAlg_->pimpl_.get();
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    std::map<HcclCMDType, AlgType> algType;
    ret = algConfigurator->SelectAlgType(serverNum, DevType::DEV_TYPE_COUNT, algType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase = nullptr;
    GlobalMockObject::verify();
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel0, AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING);
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel1, AlgTypeLevel1::ALG_LEVEL1_HD);
    // ResetInitState();
    // unsetenv("HCCL_ALGORITHM");
}

TEST_F(HcclCommTest, ut_SetAlgType_1server_1dev_ring_ring)
{
    public_stubs(true);

    HcclResult ret;
    // setenv("HCCL_ALGORITHM", "level0:ring;level1:ring", 1);
    // ResetInitState();
    std::string algo = "level0:ring;level1:ring";
    ret = SetHcclAlgoConfig(algo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> ranks;
    get_ranks_1server_1dev(ranks);
    HcclCommParams params;
    WorldGroupInfo groupCommonData;
    TestConstructParamsByRankInfo(params, groupCommonData, ranks);

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, ranks, groupCommonData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 serverNum = implBase->serverNum_;
    hcclImpl* impl = implBase->implAlg_->pimpl_.get();
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    std::map<HcclCMDType, AlgType> algType;
    ret = algConfigurator->SelectAlgType(serverNum, DevType::DEV_TYPE_COUNT, algType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase = nullptr;
    GlobalMockObject::verify();
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel0, AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING);
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel1, AlgTypeLevel1::ALG_LEVEL1_RING);
    // ResetInitState();
    // unsetenv("HCCL_ALGORITHM");
}

TEST_F(HcclCommTest, ut_SetAlgType_1server_1dev_mesh_hd)
{
    public_stubs(true);

    HcclResult ret;
    // setenv("HCCL_ALGORITHM", "level0:fullmesh;level1:H-D_R", 1);
    // ResetInitState();
    std::string algo = "level0:fullmesh;level1:H-D_R";
    ret = SetHcclAlgoConfig(algo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> ranks;
    get_ranks_1server_1dev(ranks);
    HcclCommParams params;
    WorldGroupInfo groupCommonData;
    TestConstructParamsByRankInfo(params, groupCommonData, ranks);

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, ranks, groupCommonData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 serverNum = implBase->serverNum_;
    hcclImpl* impl = implBase->implAlg_->pimpl_.get();
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    std::map<HcclCMDType, AlgType> algType;
    ret = algConfigurator->SelectAlgType(serverNum, DevType::DEV_TYPE_COUNT, algType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase = nullptr;
    GlobalMockObject::verify();
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel0, AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING);
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel1, AlgTypeLevel1::ALG_LEVEL1_HD);
    // ResetInitState();
    // unsetenv("HCCL_ALGORITHM");
}

TEST_F(HcclCommTest, ut_SetAlgType_1server_1dev_hd_hd)
{
    public_stubs(true);

    HcclResult ret;
    // setenv("HCCL_ALGORITHM", "level0:H-D_R;level1:H-D_R", 1);
    // ResetInitState();
    std::string algo = "level0:H-D_R;level1:H-D_R";
    ret = SetHcclAlgoConfig(algo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> ranks;
    get_ranks_1server_1dev(ranks);
    HcclCommParams params;
    WorldGroupInfo groupCommonData;
    TestConstructParamsByRankInfo(params, groupCommonData, ranks);

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, ranks, groupCommonData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 serverNum = implBase->serverNum_;
    hcclImpl* impl = implBase->implAlg_->pimpl_.get();
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    std::map<HcclCMDType, AlgType> algType;
    ret = algConfigurator->SelectAlgType(serverNum, DevType::DEV_TYPE_COUNT, algType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase = nullptr;
    GlobalMockObject::verify();
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel0, AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING);
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel1, AlgTypeLevel1::ALG_LEVEL1_HD);
    // ResetInitState();
    // unsetenv("HCCL_ALGORITHM");
}

TEST_F(HcclCommTest, ut_SetAlgType_1server_1dev_ring_mesh)
{
    public_stubs(true);

    HcclResult ret;
    // setenv("HCCL_ALGORITHM", "level0:ring;level1:fullmesh", 1);
    // ResetInitState();
    std::string algo = "level0:ring;level1:fullmesh";
    ret = SetHcclAlgoConfig(algo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> ranks;
    get_ranks_1server_1dev(ranks);
    HcclCommParams params;
    WorldGroupInfo groupCommonData;
    TestConstructParamsByRankInfo(params, groupCommonData, ranks);

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, ranks, groupCommonData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 serverNum = implBase->serverNum_;
    hcclImpl* impl = implBase->implAlg_->pimpl_.get();
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    std::map<HcclCMDType, AlgType> algType;
    ret = algConfigurator->SelectAlgType(serverNum, DevType::DEV_TYPE_COUNT, algType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase = nullptr;
    GlobalMockObject::verify();
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel0, AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING);
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel1, AlgTypeLevel1::ALG_LEVEL1_RING);
    ResetInitState();
    unsetenv("HCCL_ALGORITHM");
}

TEST_F(HcclCommTest, ut_SetAlgType_1server_2dev_ring_hd)
{
    public_stubs(true);

    HcclResult ret;
    // setenv("HCCL_ALGORITHM", "level0:ring;level1:H-D_R", 1);
    // ResetInitState();
    std::string algo = "level0:ring;level1:H-D_R";
    ret = SetHcclAlgoConfig(algo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> ranks;
    get_ranks_1server_2dev(ranks);
    HcclCommParams params;
    WorldGroupInfo groupCommonData;
    TestConstructParamsByRankInfo(params, groupCommonData, ranks);

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, ranks, groupCommonData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 serverNum = implBase->serverNum_;
    hcclImpl* impl = implBase->implAlg_->pimpl_.get();
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    std::map<HcclCMDType, AlgType> algType;
    ret = algConfigurator->SelectAlgType(serverNum, DevType::DEV_TYPE_COUNT, algType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase = nullptr;
    GlobalMockObject::verify();
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel0, AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING);
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel1, AlgTypeLevel1::ALG_LEVEL1_HD);
    // ResetInitState();
    // unsetenv("HCCL_ALGORITHM");
}

TEST_F(HcclCommTest, ut_SetAlgType_1server_2dev_mesh_hd)
{
    public_stubs(true);

    HcclResult ret;
    // setenv("HCCL_ALGORITHM", "level0:fullmesh;level1:H-D_R", 1);
    // ResetInitState();
    std::string algo = "level0:fullmesh;level1:H-D_R";
    ret = SetHcclAlgoConfig(algo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> ranks;
    get_ranks_1server_2dev(ranks);
    HcclCommParams params;
    WorldGroupInfo groupCommonData;
    TestConstructParamsByRankInfo(params, groupCommonData, ranks);

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, ranks, groupCommonData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 serverNum = implBase->serverNum_;
    hcclImpl* impl = implBase->implAlg_->pimpl_.get();
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    std::map<HcclCMDType, AlgType> algType;
    ret = algConfigurator->SelectAlgType(serverNum, DevType::DEV_TYPE_COUNT, algType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase = nullptr;
    GlobalMockObject::verify();
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel0, AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING);
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel1, AlgTypeLevel1::ALG_LEVEL1_HD);
    // ResetInitState();
    // unsetenv("HCCL_ALGORITHM");
}

TEST_F(HcclCommTest, ut_SetAlgType_1server_2dev_hd_ring)
{
    public_stubs(true);

    HcclResult ret;
    // setenv("HCCL_ALGORITHM", "level0:H-D_R;level1:ring", 1);
    // ResetInitState();
    std::string algo = "level0:H-D_R;level1:ring";
    ret = SetHcclAlgoConfig(algo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> ranks;
    get_ranks_1server_2dev(ranks);
    HcclCommParams params;
    WorldGroupInfo groupCommonData;
    TestConstructParamsByRankInfo(params, groupCommonData, ranks);

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, ranks, groupCommonData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 serverNum = implBase->serverNum_;
    hcclImpl* impl = implBase->implAlg_->pimpl_.get();
    impl->isStandardCard_ = false;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    algConfigurator->topoAttr_.isStandardCard = false;

    std::map<HcclCMDType, AlgType> algType;
    ret = algConfigurator->SelectAlgType(serverNum, DevType::DEV_TYPE_COUNT, algType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase = nullptr;
    GlobalMockObject::verify();
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel0, AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING);
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel1, AlgTypeLevel1::ALG_LEVEL1_RING);
    // ResetInitState();
    // unsetenv("HCCL_ALGORITHM");
}

TEST_F(HcclCommTest, ut_SetAlgType_1server_3dev_mesh_hd)
{
    public_stubs(true);

    HcclResult ret;
    // setenv("HCCL_ALGORITHM", "level0:fullmesh;level1:H-D_R", 1);
    // ResetInitState();
    std::string algo = "level0:fullmesh;level1:H-D_R";
    ret = SetHcclAlgoConfig(algo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> ranks;
    get_ranks_1server_3dev(ranks);
    HcclCommParams params;
    WorldGroupInfo groupCommonData;
    TestConstructParamsByRankInfo(params, groupCommonData, ranks);

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicatorAttrs::IsStandardCard).stubs().with(mockcpp::any()).will(returnValue(true));

    MOCKER_CPP_VIRTUAL(*implBase, &HcclCommunicator::IsStandardCard).stubs().will(returnValue(true));

    implBase->isStandardCard_ = true;
    ret = implBase->Init(params, ranks, groupCommonData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 serverNum = implBase->serverNum_;
    hcclImpl* impl = implBase->implAlg_->pimpl_.get();
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    std::map<HcclCMDType, AlgType> algType;
    ret = algConfigurator->SelectAlgType(serverNum, DevType::DEV_TYPE_COUNT, algType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase = nullptr;
    GlobalMockObject::verify();
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel0, AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING);
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel1, AlgTypeLevel1::ALG_LEVEL1_HD);
    // ResetInitState();
    // unsetenv("HCCL_ALGORITHM");
}

TEST_F(HcclCommTest, ut_SetAlgType_1server_3dev_ring_hd)
{
    public_stubs(true);

    HcclResult ret;
    // setenv("HCCL_ALGORITHM", "level0:ring;level1:H-D_R", 1);
    // ResetInitState();
    std::string algo = "level0:ring;level1:H-D_R";
    ret = SetHcclAlgoConfig(algo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> ranks;
    get_ranks_1server_3dev(ranks);
    HcclCommParams params;
    WorldGroupInfo groupCommonData;
    TestConstructParamsByRankInfo(params, groupCommonData, ranks);

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, ranks, groupCommonData);
    EXPECT_NE(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcclCommTest, ut_SetAlgType_1server_3dev_hd_ring)
{
    public_stubs(true);

    HcclResult ret;
    // setenv("HCCL_ALGORITHM", "level0:H-D_R;level1:ring", 1);
    // ResetInitState();
    std::string algo = "level0:H-D_R;level1:ring";
    ret = SetHcclAlgoConfig(algo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> ranks;
    get_ranks_1server_3dev(ranks);
    HcclCommParams params;
    WorldGroupInfo groupCommonData;
    TestConstructParamsByRankInfo(params, groupCommonData, ranks);

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP_VIRTUAL(*implBase, &HcclCommunicator::IsStandardCard).stubs().will(returnValue(true));

    MOCKER_CPP(&HcclCommunicatorAttrs::IsStandardCard).stubs().with(mockcpp::any()).will(returnValue(true));

    ret = implBase->Init(params, ranks, groupCommonData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 serverNum = implBase->serverNum_;
    hcclImpl* impl = implBase->implAlg_->pimpl_.get();
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    std::map<HcclCMDType, AlgType> algType;
    ret = algConfigurator->SelectAlgType(serverNum, DevType::DEV_TYPE_COUNT, algType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase = nullptr;
    GlobalMockObject::verify();
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel0, AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING);
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel1, AlgTypeLevel1::ALG_LEVEL1_RING);
    // ResetInitState();
    // unsetenv("HCCL_ALGORITHM");
}

TEST_F(HcclCommTest, ut_SetAlgType_1server_4dev_mesh_hd)
{
    public_stubs(true);

    HcclResult ret;
    // setenv("HCCL_ALGORITHM", "level0:fullmesh;level1:H-D_R", 1);
    // ResetInitState();
    std::string algo = "level0:fullmesh;level1:H-D_R";
    ret = SetHcclAlgoConfig(algo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> ranks;
    get_ranks_1server_4dev(ranks);

    HcclCommParams params;
    WorldGroupInfo groupCommonData;
    TestConstructParamsByRankInfo(params, groupCommonData, ranks);

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, ranks, groupCommonData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 serverNum = implBase->serverNum_;
    hcclImpl* impl = implBase->implAlg_->pimpl_.get();
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    std::map<HcclCMDType, AlgType> algType;
    ret = algConfigurator->SelectAlgType(serverNum, DevType::DEV_TYPE_COUNT, algType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase = nullptr;
    GlobalMockObject::verify();
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel0, AlgTypeLevel0::ALG_LEVEL0_4P_MESH);
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel1, AlgTypeLevel1::ALG_LEVEL1_HD);
    // ResetInitState();
    // unsetenv("HCCL_ALGORITHM");
}

TEST_F(HcclCommTest, ut_SetAlgType_1server_4dev_ring_hd)
{
    public_stubs(true);

    HcclResult ret;
    // setenv("HCCL_ALGORITHM", "level0:ring;level1:H-D_R", 1);
    // ResetInitState();
    std::string algo = "level0:ring;level1:H-D_R";
    ret = SetHcclAlgoConfig(algo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> ranks;
    get_ranks_1server_4dev(ranks);

    HcclCommParams params;
    WorldGroupInfo groupCommonData;
    TestConstructParamsByRankInfo(params, groupCommonData, ranks);

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, ranks, groupCommonData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 serverNum = implBase->serverNum_;
    hcclImpl* impl = implBase->implAlg_->pimpl_.get();
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    std::map<HcclCMDType, AlgType> algType;
    // 当前将serverId设置为impl的成员变量，之后添加新llt用例需要考虑serverid以获取server内dev数
    ret = algConfigurator->SelectAlgType(serverNum, DevType::DEV_TYPE_COUNT, algType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase = nullptr;
    GlobalMockObject::verify();
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel0, AlgTypeLevel0::ALG_LEVEL0_4P_MESH);
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel1, AlgTypeLevel1::ALG_LEVEL1_HD);
    // ResetInitState();
    // unsetenv("HCCL_ALGORITHM");
}

TEST_F(HcclCommTest, ut_SetAlgType_1server_4dev_hd_ring)
{
    public_stubs(true);

    HcclResult ret;
    // setenv("HCCL_ALGORITHM", "level0:H-D_R;level1:ring", 1);
    // ResetInitState();
    std::string algo = "level0:H-D_R;level1:ring";
    ret = SetHcclAlgoConfig(algo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> ranks;
    get_ranks_1server_4dev(ranks);

    HcclCommParams params;
    WorldGroupInfo groupCommonData;
    TestConstructParamsByRankInfo(params, groupCommonData, ranks);

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, ranks, groupCommonData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 serverNum = implBase->serverNum_;
    hcclImpl* impl = implBase->implAlg_->pimpl_.get();
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::map<HcclCMDType, AlgType> algType;

    ret = algConfigurator->SelectAlgType(serverNum, DevType::DEV_TYPE_COUNT, algType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase = nullptr;
    GlobalMockObject::verify();
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel0, AlgTypeLevel0::ALG_LEVEL0_4P_MESH);
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel1, AlgTypeLevel1::ALG_LEVEL1_RING);
    // ResetInitState();
    // unsetenv("HCCL_ALGORITHM");
}

TEST_F(HcclCommTest, ut_SetAlgType_1server_4dev_default)
{
    public_stubs(true);

    HcclResult ret;
    // unsetenv("HCCL_ALGORITHM");
    // ResetInitState();
    std::string algo;
    ret = SetHcclAlgoConfig(algo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> ranks;
    get_ranks_1server_4dev(ranks);

    HcclCommParams params;
    WorldGroupInfo groupCommonData;
    TestConstructParamsByRankInfo(params, groupCommonData, ranks);

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, ranks, groupCommonData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 serverNum = implBase->serverNum_;
    hcclImpl* impl = implBase->implAlg_->pimpl_.get();
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    std::map<HcclCMDType, AlgType> algType;
    ret = algConfigurator->SelectAlgType(serverNum, DevType::DEV_TYPE_COUNT, algType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase = nullptr;
    GlobalMockObject::verify();
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel0, AlgTypeLevel0::ALG_LEVEL0_4P_MESH);
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel1, AlgTypeLevel1::ALG_LEVEL1_RING);
    // ResetInitState();
    // unsetenv("HCCL_ALGORITHM");
}

TEST_F(HcclCommTest, ut_SetAlgType_1server_8dev_mesh_hd)
{
    public_stubs(true);

    HcclResult ret;
    // setenv("HCCL_ALGORITHM", "level0:fullmesh;level1:H-D_R", 1);
    // ResetInitState();
    std::string algo = "level0:fullmesh;level1:H-D_R";
    ret = SetHcclAlgoConfig(algo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> ranks;
    get_ranks_1server_8dev(ranks);

    HcclCommParams params;
    WorldGroupInfo groupCommonData;
    TestConstructParamsByRankInfo(params, groupCommonData, ranks);

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, ranks, groupCommonData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 serverNum = implBase->serverNum_;
    hcclImpl* impl = implBase->implAlg_->pimpl_.get();
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    std::map<HcclCMDType, AlgType> algType;
    ret = algConfigurator->SelectAlgType(serverNum, DevType::DEV_TYPE_COUNT, algType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase = nullptr;
    GlobalMockObject::verify();
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel0, AlgTypeLevel0::ALG_LEVEL0_8P_RING);
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel1, AlgTypeLevel1::ALG_LEVEL1_HD);
    // ResetInitState();
    // unsetenv("HCCL_ALGORITHM");
}

TEST_F(HcclCommTest, ut_SetAlgType_1server_8dev_ring_hd)
{
    public_stubs(true);

    HcclResult ret;
    // setenv("HCCL_ALGORITHM", "level0:ring;level1:H-D_R", 1);
    // ResetInitState();
    std::string algo = "level0:ring;level1:H-D_R";
    ret = SetHcclAlgoConfig(algo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> ranks;
    get_ranks_1server_8dev(ranks);

    HcclCommParams params;
    WorldGroupInfo groupCommonData;
    TestConstructParamsByRankInfo(params, groupCommonData, ranks);

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, ranks, groupCommonData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 serverNum = implBase->serverNum_;
    hcclImpl* impl = implBase->implAlg_->pimpl_.get();
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    std::map<HcclCMDType, AlgType> algType;
    ret = algConfigurator->SelectAlgType(serverNum, DevType::DEV_TYPE_COUNT, algType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase = nullptr;
    GlobalMockObject::verify();
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel0, AlgTypeLevel0::ALG_LEVEL0_8P_RING);
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel1, AlgTypeLevel1::ALG_LEVEL1_HD);
    // ResetInitState();
    // unsetenv("HCCL_ALGORITHM");
}

TEST_F(HcclCommTest, ut_SetAlgType_1server_8dev_default)
{
    public_stubs(true);

    HcclResult ret;
    // setenv("HCCL_ALGORITHM", "level0:ring;level1:H-D_R", 1);
    // ResetInitState();
    std::string algo = "level0:ring;level1:H-D_R";
    ret = SetHcclAlgoConfig(algo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> ranks;
    get_ranks_1server_8dev(ranks);

    HcclCommParams params;
    WorldGroupInfo groupCommonData;
    TestConstructParamsByRankInfo(params, groupCommonData, ranks);

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, ranks, groupCommonData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 serverNum = implBase->serverNum_;
    hcclImpl* impl = implBase->implAlg_->pimpl_.get();
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    std::map<HcclCMDType, AlgType> algType;
    ret = algConfigurator->SelectAlgType(serverNum, DevType::DEV_TYPE_COUNT, algType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase = nullptr;
    GlobalMockObject::verify();
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel0, AlgTypeLevel0::ALG_LEVEL0_8P_RING);
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel1, AlgTypeLevel1::ALG_LEVEL1_HD);
    // ResetInitState();
    // unsetenv("HCCL_ALGORITHM");
}

TEST_F(HcclCommTest, ut_SetAlgType_module_1server_4dev_mesh_hd)
{
    public_stubs(true);

    HcclResult ret;
    // setenv("HCCL_ALGORITHM", "level0:fullmesh;level1:H-D_R", 1);
    // ResetInitState();
    std::string algo = "level0:fullmesh;level1:H-D_R";
    ret = SetHcclAlgoConfig(algo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> ranks;
    get_ranks_1server_4dev(ranks);

    HcclCommParams params;
    WorldGroupInfo groupCommonData;
    TestConstructParamsByRankInfo(params, groupCommonData, ranks);

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, ranks, groupCommonData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 serverNum = implBase->serverNum_;
    hcclImpl* impl = implBase->implAlg_->pimpl_.get();
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    std::map<HcclCMDType, AlgType> algType;
    ret = algConfigurator->SelectAlgType(serverNum, DevType::DEV_TYPE_COUNT, algType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase = nullptr;
    GlobalMockObject::verify();
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel0, AlgTypeLevel0::ALG_LEVEL0_4P_MESH);
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel1, AlgTypeLevel1::ALG_LEVEL1_HD);
    // ResetInitState();
    // unsetenv("HCCL_ALGORITHM");
}

TEST_F(HcclCommTest, ut_SetAlgType_module_1server_4dev_ring_hd)
{
    public_stubs(true);

    HcclResult ret;
    // setenv("HCCL_ALGORITHM", "level0:ring;level1:H-D_R", 1);
    // ResetInitState();
    std::string algo = "level0:ring;level1:H-D_R";
    ret = SetHcclAlgoConfig(algo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> ranks;
    get_ranks_1server_4dev(ranks);

    HcclCommParams params;
    WorldGroupInfo groupCommonData;
    TestConstructParamsByRankInfo(params, groupCommonData, ranks);

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, ranks, groupCommonData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 serverNum = implBase->serverNum_;
    hcclImpl* impl = implBase->implAlg_->pimpl_.get();
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    std::map<HcclCMDType, AlgType> algType;
    ret = algConfigurator->SelectAlgType(serverNum, DevType::DEV_TYPE_COUNT, algType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase = nullptr;
    GlobalMockObject::verify();
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel0, AlgTypeLevel0::ALG_LEVEL0_4P_MESH);
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel1, AlgTypeLevel1::ALG_LEVEL1_HD);
    // ResetInitState();
    // unsetenv("HCCL_ALGORITHM");
}

TEST_F(HcclCommTest, ut_SetAlgType_module_1server_4dev_hd_ring)
{
    public_stubs(true);

    HcclResult ret;
    // setenv("HCCL_ALGORITHM", "level0:H-D_R;level1:ring", 1);
    // ResetInitState();
    std::string algo = "level0:H-D_R;level1:ring";
    ret = SetHcclAlgoConfig(algo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> ranks;
    get_ranks_1server_4dev(ranks);

    HcclCommParams params;
    WorldGroupInfo groupCommonData;
    TestConstructParamsByRankInfo(params, groupCommonData, ranks);

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, ranks, groupCommonData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 serverNum = implBase->serverNum_;
    hcclImpl* impl = implBase->implAlg_->pimpl_.get();
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    std::map<HcclCMDType, AlgType> algType;
    ret = algConfigurator->SelectAlgType(serverNum, DevType::DEV_TYPE_COUNT, algType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase = nullptr;
    GlobalMockObject::verify();
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel0, AlgTypeLevel0::ALG_LEVEL0_4P_MESH);
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel1, AlgTypeLevel1::ALG_LEVEL1_RING);
    // ResetInitState();
    // unsetenv("HCCL_ALGORITHM");
}

TEST_F(HcclCommTest, ut_SetAlgType_module_1server_4dev_default)
{
    public_stubs(true);

    HcclResult ret;
    // unsetenv("HCCL_ALGORITHM");
    ResetInitState();

    ret = InitExternalInput();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> ranks;
    get_ranks_1server_4dev(ranks);

    HcclCommParams params;
    WorldGroupInfo groupCommonData;
    TestConstructParamsByRankInfo(params, groupCommonData, ranks);

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, ranks, groupCommonData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 serverNum = implBase->serverNum_;
    hcclImpl* impl = implBase->implAlg_->pimpl_.get();
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    std::map<HcclCMDType, AlgType> algType;
    ret = algConfigurator->SelectAlgType(serverNum, DevType::DEV_TYPE_COUNT, algType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase = nullptr;
    GlobalMockObject::verify();
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel0, AlgTypeLevel0::ALG_LEVEL0_4P_MESH);
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel1, AlgTypeLevel1::ALG_LEVEL1_RING);
    ResetInitState();
    // unsetenv("HCCL_ALGORITHM");
}

TEST_F(HcclCommTest, ut_SetAlgType_module_1server_8dev_ring_hd)
{
    public_stubs(true);

    HcclResult ret;
    // setenv("HCCL_ALGORITHM", "level0:ring;level1:H-D_R", 1);
    // ResetInitState();
    std::string algo = "level0:ring;level1:H-D_R";
    ret = SetHcclAlgoConfig(algo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> ranks;
    get_ranks_1server_8dev(ranks);

    HcclCommParams params;
    WorldGroupInfo groupCommonData;
    TestConstructParamsByRankInfo(params, groupCommonData, ranks);

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, ranks, groupCommonData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 serverNum = implBase->serverNum_;
    hcclImpl* impl = implBase->implAlg_->pimpl_.get();
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    std::map<HcclCMDType, AlgType> algType;
    ret = algConfigurator->SelectAlgType(serverNum, DevType::DEV_TYPE_COUNT, algType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase = nullptr;
    GlobalMockObject::verify();
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel0, AlgTypeLevel0::ALG_LEVEL0_8P_RING);
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel1, AlgTypeLevel1::ALG_LEVEL1_HD);
    // ResetInitState();
    // unsetenv("HCCL_ALGORITHM");
}

TEST_F(HcclCommTest, ut_SetAlgType_module_1server_8dev_mesh_hd)
{
    public_stubs(true);

    HcclResult ret;
    // setenv("HCCL_ALGORITHM", "level0:fullmesh;level1:H-D_R", 1);
    // ResetInitState();
    std::string algo = "level0:fullmesh;level1:H-D_R";
    ret = SetHcclAlgoConfig(algo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> ranks;
    get_ranks_1server_8dev(ranks);

    HcclCommParams params;
    WorldGroupInfo groupCommonData;
    TestConstructParamsByRankInfo(params, groupCommonData, ranks);

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, ranks, groupCommonData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 serverNum = implBase->serverNum_;
    hcclImpl* impl = implBase->implAlg_->pimpl_.get();
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    std::map<HcclCMDType, AlgType> algType;
    ret = algConfigurator->SelectAlgType(serverNum, DevType::DEV_TYPE_COUNT, algType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase = nullptr;
    GlobalMockObject::verify();
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel0, AlgTypeLevel0::ALG_LEVEL0_8P_RING);
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel1, AlgTypeLevel1::ALG_LEVEL1_HD);
    // ResetInitState();
    // unsetenv("HCCL_ALGORITHM");
}

TEST_F(HcclCommTest, ut_SetAlgType_module_1server_8dev_hd_ring)
{
    public_stubs(true);

    HcclResult ret;
    // setenv("HCCL_ALGORITHM", "level0:H-D_R;level1:ring", 1);
    // ResetInitState();
    std::string algo = "level0:H-D_R;level1:ring";
    ret = SetHcclAlgoConfig(algo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> ranks;
    get_ranks_1server_8dev(ranks);

    HcclCommParams params;
    WorldGroupInfo groupCommonData;
    TestConstructParamsByRankInfo(params, groupCommonData, ranks);

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, ranks, groupCommonData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 serverNum = implBase->serverNum_;
    hcclImpl* impl = implBase->implAlg_->pimpl_.get();
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    std::map<HcclCMDType, AlgType> algType;
    ret = algConfigurator->SelectAlgType(serverNum, DevType::DEV_TYPE_COUNT, algType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase = nullptr;
    GlobalMockObject::verify();
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel0, AlgTypeLevel0::ALG_LEVEL0_8P_RING);
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel1, AlgTypeLevel1::ALG_LEVEL1_RING);
    // ResetInitState();
    // unsetenv("HCCL_ALGORITHM");
}

TEST_F(HcclCommTest, ut_SetAlgType_module_1server_8dev_default)
{
    public_stubs(true);

    HcclResult ret;
    // unsetenv("HCCL_ALGORITHM");
    ResetInitState();
    ret = InitExternalInput();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> ranks;
    get_ranks_1server_8dev(ranks);

    HcclCommParams params;
    WorldGroupInfo groupCommonData;
    TestConstructParamsByRankInfo(params, groupCommonData, ranks);

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP_VIRTUAL(*implBase, &HcclCommunicator::IsStandardCard).stubs().will(returnValue(false));

    ret = implBase->Init(params, ranks, groupCommonData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 serverNum = implBase->serverNum_;
    hcclImpl* impl = implBase->implAlg_->pimpl_.get();

    // 相当于插入个桩，强制设置 isSingleMeshAggregation_ 的值
    impl->isSingleMeshAggregation_ = true;

    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    algConfigurator->topoAttr_.isSingleMeshAggregation = true;

    std::map<HcclCMDType, AlgType> algType;
    ret = algConfigurator->SelectAlgType(serverNum, DevType::DEV_TYPE_COUNT, algType);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    implBase = nullptr;
    GlobalMockObject::verify();
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel0, AlgTypeLevel0::ALG_LEVEL0_8P_RING);
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel1, AlgTypeLevel1::ALG_LEVEL1_RING);
    ResetInitState();
    // unsetenv("HCCL_ALGORITHM");
}

TEST_F(HcclCommTest, ut_SetAlgType_module_7server_1dev_default)
{
    public_stubs(true);

    HcclResult ret;
    // unsetenv("HCCL_ALGORITHM");
    ResetInitState();
    ret = InitExternalInput();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> ranks;
    get_ranks_7server_1dev(ranks);

    HcclCommParams params;
    WorldGroupInfo groupCommonData;
    TestConstructParamsByRankInfo(params, groupCommonData, ranks);

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, ranks, groupCommonData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 serverNum = implBase->serverNum_;
    hcclImpl* impl = implBase->implAlg_->pimpl_.get();
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    std::map<HcclCMDType, AlgType> algType;
    ret = algConfigurator->SelectAlgType(serverNum, DevType::DEV_TYPE_COUNT, algType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase = nullptr;
    GlobalMockObject::verify();
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel0, AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING);
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel1, AlgTypeLevel1::ALG_LEVEL1_RING);
    ResetInitState();
    // unsetenv("HCCL_ALGORITHM");
}

TEST_F(HcclCommTest, ut_SetAlgType_module_7server_1dev_ring_ring)
{
    public_stubs(true);

    HcclResult ret;
    // setenv("HCCL_ALGORITHM", "level0:ring;level1:ring", 1);
    // ResetInitState();
    std::string algo = "level0:ring;level1:ring";
    ret = SetHcclAlgoConfig(algo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> ranks;
    get_ranks_7server_1dev(ranks);

    HcclCommParams params;
    WorldGroupInfo groupCommonData;
    TestConstructParamsByRankInfo(params, groupCommonData, ranks);

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, ranks, groupCommonData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 serverNum = implBase->serverNum_;
    hcclImpl* impl = implBase->implAlg_->pimpl_.get();
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    std::map<HcclCMDType, AlgType> algType;
    ret = algConfigurator->SelectAlgType(serverNum, DevType::DEV_TYPE_COUNT, algType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase = nullptr;
    GlobalMockObject::verify();
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel0, AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING);
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel1, AlgTypeLevel1::ALG_LEVEL1_RING);
    // ResetInitState();
    // unsetenv("HCCL_ALGORITHM");
}

TEST_F(HcclCommTest, ut_SetAlgType_module_7server_1dev_ring_hd)
{
    public_stubs(true);

    HcclResult ret;
    // setenv("HCCL_ALGORITHM", "level0:ring;level1:H-D_R", 1);
    // ResetInitState();
    std::string algo = "level0:ring;level1:H-D_R";
    ret = SetHcclAlgoConfig(algo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> ranks;
    get_ranks_7server_1dev(ranks);

    HcclCommParams params;
    WorldGroupInfo groupCommonData;
    TestConstructParamsByRankInfo(params, groupCommonData, ranks);

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, ranks, groupCommonData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 serverNum = implBase->serverNum_;
    hcclImpl* impl = implBase->implAlg_->pimpl_.get();
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    std::map<HcclCMDType, AlgType> algType;
    ret = algConfigurator->SelectAlgType(serverNum, DevType::DEV_TYPE_COUNT, algType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase = nullptr;
    GlobalMockObject::verify();
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel0, AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING);
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel1, AlgTypeLevel1::ALG_LEVEL1_HD);
    // ResetInitState();
    // unsetenv("HCCL_ALGORITHM");
}

TEST_F(HcclCommTest, ut_SetAlgType_module_8server_1dev_default)
{
    public_stubs(true);

    HcclResult ret;
    // unsetenv("HCCL_ALGORITHM");
    ResetInitState();
    ret = InitExternalInput();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> ranks;
    get_ranks_8server_1dev(ranks);

    HcclCommParams params;
    WorldGroupInfo groupCommonData;
    TestConstructParamsByRankInfo(params, groupCommonData, ranks);

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, ranks, groupCommonData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 serverNum = implBase->serverNum_;
    hcclImpl* impl = implBase->implAlg_->pimpl_.get();
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    std::map<HcclCMDType, AlgType> algType;
    ret = algConfigurator->SelectAlgType(serverNum, DevType::DEV_TYPE_COUNT, algType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase = nullptr;
    GlobalMockObject::verify();
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel0, AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING);
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel1, AlgTypeLevel1::ALG_LEVEL1_HD);
    ResetInitState();
    // unsetenv("HCCL_ALGORITHM");
}

TEST_F(HcclCommTest, ut_SetAlgType_module_8server_1dev_ring_ring)
{
    public_stubs(true);

    HcclResult ret;
    // setenv("HCCL_ALGORITHM", "level0:ring;level1:ring", 1);
    // ResetInitState();
    std::string algo = "level0:ring;level1:ring";
    ret = SetHcclAlgoConfig(algo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> ranks;
    get_ranks_8server_1dev(ranks);

    HcclCommParams params;
    WorldGroupInfo groupCommonData;
    TestConstructParamsByRankInfo(params, groupCommonData, ranks);

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, ranks, groupCommonData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 serverNum = implBase->serverNum_;
    hcclImpl* impl = implBase->implAlg_->pimpl_.get();
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    std::map<HcclCMDType, AlgType> algType;
    ret = algConfigurator->SelectAlgType(serverNum, DevType::DEV_TYPE_COUNT, algType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase = nullptr;
    GlobalMockObject::verify();
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel0, AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING);
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel1, AlgTypeLevel1::ALG_LEVEL1_RING);
    // ResetInitState();
    // unsetenv("HCCL_ALGORITHM");
}

TEST_F(HcclCommTest, ut_SetAlgType_module_8server_1dev_ring_hd)
{
    public_stubs(true);

    HcclResult ret;
    // setenv("HCCL_ALGORITHM", "level0:ring;level1:H-D_R", 1);
    // ResetInitState();
    std::string algo = "level0:ring;level1:H-D_R";
    ret = SetHcclAlgoConfig(algo);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<RankInfo> ranks;
    get_ranks_8server_1dev(ranks);

    HcclCommParams params;
    WorldGroupInfo groupCommonData;
    TestConstructParamsByRankInfo(params, groupCommonData, ranks);

    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());
    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    ret = implBase->Init(params, ranks, groupCommonData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 serverNum = implBase->serverNum_;
    hcclImpl* impl = implBase->implAlg_->pimpl_.get();
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    std::map<HcclCMDType, AlgType> algType;
    ret = algConfigurator->SelectAlgType(serverNum, DevType::DEV_TYPE_COUNT, algType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase = nullptr;
    GlobalMockObject::verify();
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel0, AlgTypeLevel0::ALG_LEVEL0_NP_SINGLE_RING);
    EXPECT_EQ(algType[HcclCMDType::HCCL_CMD_ALL].algoLevel1, AlgTypeLevel1::ALG_LEVEL1_HD);
    // ResetInitState();
    // unsetenv("HCCL_ALGORITHM");
}

TEST_F(HcclCommTest, ut_gradient_segment)
{
    public_stubs(true);

    s32 ret;

    /* 初始化通信域 */
    HcclCommParams comm_params;
    comm_params.rank = 0;
    comm_params.totalRanks = 1;
    comm_params.logicDevId = 0;
    comm_params.deviceType = DevType::DEV_TYPE_910;
    hrtSetDevice(comm_params.logicDevId);
    ret = hcclComm::GetUniqueId(&comm_params.id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclComm comm;
    RankTable_t rankTable = get_rank_table_rank_nic_device();
    CommConfig commConfig("hccl_world_group");
    ret = comm.init(comm_params, commConfig, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    /* 初始化模型参数 */
    struct model_feature feature;
    std::vector<u32> segment_index;
    char group[] = "";
    char model_name[] = "resnet";
    feature.gradient_num = 2;
    feature.gradient_size = (float*)sal_malloc(2 * sizeof(float));
    sal_memset(feature.gradient_size, 2 * sizeof(float), 0, 2 * sizeof(float));
    feature.gradient_time = (float*)sal_malloc(2 * sizeof(float));
    sal_memset(feature.gradient_time, 2 * sizeof(float), 0, 2 * sizeof(float));
    feature.model_name = model_name;

    segment_index.push_back(feature.gradient_num - 1);
    MOCKER_CPP(&GradientSegment::GetGradientSegmentExecutor).expects(atMost(1)).will(returnValue(0));

    // ret = comm.GetGradientSegment(group, &feature, segment_index);
    bool isConfig = true;
    ret = GetGradientSegment(group, &feature, segment_index, isConfig);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();

    sal_free(feature.gradient_size);
    sal_free(feature.gradient_time);
}

RankTable_t get_rank_table_v71()
{
    RankTable_t rankTable;
    rankTable.deviceNum = 2;

    rankTable.serverNum = 1;
    rankTable.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    rankTable.nicNum = 2;
    rankTable.nicNames.push_back("eth0");
    rankTable.rankNum = 2;

    // rank 信息
    RankInfo_t rank;
    rank.rankId = 0;
    rank.serverIdx = 0;
    rank.serverId = "192.168.1.1";
    rank.deviceInfo.devicePhyId = 0;
    rank.deviceInfo.deviceIp.push_back(HcclIpAddress("172.17.10.1"));
    rankTable.rankList.push_back(rank);

    RankInfo_t rank1;
    rank1.rankId = 0;
    rank1.serverIdx = 0;
    rank1.serverId = "192.168.1.1";
    rank1.deviceInfo.devicePhyId = 8;
    rank1.deviceInfo.deviceIp.push_back(HcclIpAddress("172.17.11.1"));
    rankTable.rankList.push_back(rank1);
    return rankTable;
}

TEST_F(HcclCommTest, ut_SetInnerServerAverageDevice)
{
    public_stubs(true);
    s32 ret = HCCL_SUCCESS;
    u64 inCCLbufferSizeConf = 104857600;
    u64 outCCLbufferSizeConf = 104857600;
    // 初始化通信域
    HcclCommParams comm_params;
    comm_params.rank = 0;
    comm_params.totalRanks = 1;
    comm_params.logicDevId = 0;
    comm_params.deviceType = DevType::DEV_TYPE_910B;
    hrtSetDevice(comm_params.logicDevId);
    ret = hcclComm::GetUniqueId(&comm_params.id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclComm comm(inCCLbufferSizeConf, outCCLbufferSizeConf);
    RankTable_t rankTable = get_rank_table_v71();
    CommConfig commConfig("hccl_world_group");
    ret = comm.init(comm_params, commConfig, rankTable);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcclCommTest, ut_CommCheckErrorCqe)
{
    public_stubs(true);
    HcclResult ret;

    /* 初始化通信域 */
    HcclCommParams comm_params;
    comm_params.rank = 0;
    comm_params.totalRanks = 1;
    comm_params.logicDevId = 0;
    comm_params.deviceType = DevType::DEV_TYPE_910;
    hrtSetDevice(comm_params.logicDevId);
    ret = hcclComm::GetUniqueId(&comm_params.id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclComm comm;
    RankTable_t rankTable = get_rank_table_rank_nic_device();
    CommConfig commConfig("hccl_world_group");
    ret = comm.init(comm_params, commConfig, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    comm.CommCheckErrorCqe(ret);

    GlobalMockObject::verify();
}

RankTable_t get_rank_table_rank_2server_3p_2_1()
{
    RankTable_t rankTable;
    rankTable.deviceNum = 3;

    rankTable.serverNum = 2;
    rankTable.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    rankTable.nicNum = 1;
    rankTable.nicNames.push_back("eth0");
    rankTable.rankNum = 3;

    int server1rankNum = 2;
    int server2rankNum = 1;

    for (int i = 0; i < server1rankNum; ++i) {
        RankInfo_t rank;
        rank.rankId = i;
        rank.serverIdx = 0;
        rank.serverId = "192.168.1.1";
        rank.deviceInfo.devicePhyId = i;
        rank.deviceInfo.deviceIp.push_back(HcclIpAddress("172.17.10.1"));
        rankTable.rankList.push_back(rank);
    }

    for (int i = 0; i < server2rankNum; ++i) {
        RankInfo_t rank;
        rank.rankId = i + server1rankNum;
        rank.serverIdx = 1;
        rank.serverId = "192.168.1.2";
        rank.deviceInfo.devicePhyId = i;
        rank.deviceInfo.deviceIp.push_back(HcclIpAddress("172.17.10.1"));
        rankTable.rankList.push_back(rank);
    }
    return rankTable;
}

TEST_F(HcclCommTest, ut_implbase_hcclalg_nullptr)
{
    std::vector<SendRecvInfo> allMeshAggregationSendRecvInfo;
    u64 memSize;
    std::unique_ptr<HcclCommunicator> communicator(new (std::nothrow) HcclCommunicator());
    HcclResult ret = communicator->GetAlltoAllStagedWorkSpaceMemSize(allMeshAggregationSendRecvInfo, memSize);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

void get_rank_table_rank_1server_2module_7p_4_3(HcclCommParams& params, RankTable_t& rankTable)
{
    string commId = "comm ";
    memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1);
    params.rank = 0;
    params.totalRanks = 7;
    params.isHeterogComm = false;
    params.logicDevId = 0;
    params.commWorkMode = WorkMode::HCCL_MODE_NORMAL;
    params.deviceType = DevType::DEV_TYPE_910B;

    rankTable.deviceNum = 7;
    rankTable.serverNum = 1;
    rankTable.nicDeploy = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    rankTable.nicNum = 1;
    rankTable.nicNames.push_back("eth0");
    rankTable.rankNum = 7;

    int server1rankNum = 4;
    int server2rankNum = 3;

    for (int i = 0; i < server1rankNum; ++i) {
        RankInfo_t rank;
        rank.rankId = i;
        rank.serverIdx = 0;
        rank.serverId = "192.168.1.1";
        rank.deviceInfo.devicePhyId = i;
        rank.deviceInfo.deviceIp.push_back(HcclIpAddress("172.17.10.1"));
        rankTable.rankList.push_back(rank);
    }

    for (int i = 0; i < server2rankNum; ++i) {
        RankInfo_t rank;
        rank.rankId = i + server1rankNum;
        rank.serverIdx = 0;
        rank.serverId = "192.168.1.1";
        rank.deviceInfo.devicePhyId = i + 8;
        rank.deviceInfo.deviceIp.push_back(HcclIpAddress("172.17.10.1"));
        rankTable.rankList.push_back(rank);
    }
    return;
}

TEST_F(HcclCommTest, ut_multiModuleDiffDeviceNumMode_GetModuleInfo)
{
    public_stubs(true);
    // module0:0,1,2,3  module1:8,9,10
    HcclCommParams params;
    RankTable_t rankTable;
    get_rank_table_rank_1server_2module_7p_4_3(params, rankTable);
    std::unique_ptr<HcclCommunicator> impl(new (std::nothrow) HcclCommunicator());
    impl->attrCollector_.deviceType_ = DevType::DEV_TYPE_910B;
    s32 ret = impl->attrCollector_.SetModuleInfo(rankTable.rankList);

    EXPECT_EQ(impl->attrCollector_.isDiffDeviceModule_, true);
    EXPECT_EQ(impl->attrCollector_.moduleNum_, 2);
    EXPECT_EQ(impl->attrCollector_.multiModuleDiffDeviceNumMode_, true);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // module0:0,1,2,3  module1:8,9,10,12
    RankInfo_t rank;
    rank.rankId = 8;
    rank.serverIdx = 0;
    rank.serverId = "192.168.1.1";
    rank.deviceInfo.devicePhyId = 12;
    rank.deviceInfo.deviceIp.push_back(HcclIpAddress("172.17.10.1"));
    rankTable.rankList.push_back(rank);
    params.totalRanks = 8;
    rankTable.deviceNum = 8;
    rankTable.rankNum = 8;

    ret = impl->attrCollector_.SetModuleInfo(rankTable.rankList);
    EXPECT_EQ(impl->attrCollector_.isDiffDeviceModule_, true);
    EXPECT_EQ(impl->attrCollector_.moduleNum_, 2);
    EXPECT_EQ(impl->attrCollector_.multiModuleDiffDeviceNumMode_, false);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclCommTest, hcclComm_printErrIndex)
{
    public_stubs(true);
    s32 ret = HCCL_SUCCESS;
    s32 rt_ret = RT_ERROR_NONE;

    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 申请stream
    rtStream_t stream;
    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    // 申请device memory
    aclrtMallocAttrValue moduleIdValue;
    moduleIdValue.moduleId = HCCL;
    aclrtMallocAttribute attrs{.attr = ACL_RT_MEM_ATTR_MODULE_ID, .value = moduleIdValue};
    aclrtMallocConfig cfg{.attrs = &attrs, .numAttrs = 1};
    void* mem_dev_input = NULL;
    aclError aclRet = aclrtMallocWithCfg(&mem_dev_input, 1024, ACL_MEM_TYPE_HIGH_BAND_WIDTH, &cfg);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    void* mem_dev_output = NULL;
    aclRet = aclrtMallocWithCfg(&mem_dev_output, 1024, ACL_MEM_TYPE_HIGH_BAND_WIDTH, &cfg);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // 初始化通信域
    HcclCommParams comm_params;
    comm_params.rank = 0;
    comm_params.totalRanks = 1;
    comm_params.logicDevId = 0;
    comm_params.deviceType = DevType::DEV_TYPE_910;
    ret = hcclComm::GetUniqueId(&comm_params.id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclCommunicator impl2;
    MOCKER_CPP_VIRTUAL(impl2, &HcclCommunicator::AllGather).stubs().will(returnValue(HCCL_E_PARA));

    MOCKER_CPP_VIRTUAL(impl2, &HcclCommunicator::Broadcast).stubs().will(returnValue(HCCL_E_PARA));

    MOCKER_CPP_VIRTUAL(impl2, &HcclCommunicator::SendOutPlace).stubs().will(returnValue(HCCL_E_PARA));

    MOCKER_CPP_VIRTUAL(impl2, &HcclCommunicator::ReceiveOutPlace).stubs().will(returnValue(HCCL_E_PARA));

    hcclComm comm;
    RankTable_t rankTable = get_rank_table_rank_nic_device();
    CommConfig commConfig("hccl_world_group");
    ret = comm.init(comm_params, commConfig, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 执行all-gather
    ret = comm.AllGather("allgather", mem_dev_input, mem_dev_output, 1, HCCL_DATA_TYPE_INT8, stream);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = comm.Broadcast("broadcast", mem_dev_input, 1, HCCL_DATA_TYPE_INT8, 0, stream);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = comm.ReceiveOutPlace("receiveOutPlace", mem_dev_output, 1, HCCL_DATA_TYPE_INT8, 0, stream);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = comm.SendOutPlace("sendOutPlace", mem_dev_input, 1, HCCL_DATA_TYPE_INT8, 0, stream);
    EXPECT_EQ(ret, HCCL_E_PARA);

    rt_ret = aclrtFree(mem_dev_input);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    rt_ret = aclrtFree(mem_dev_output);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    rt_ret = aclrtDestroyStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    GlobalMockObject::verify();
}

TEST_F(HcclCommTest, hcclComm_SetRanksPort)
{
    HcclCommunicator hcclCommunicator;
    hcclCommunicator.commPortConfig_.devPortSwitchOn = true;
    HcclIpAddress localIp{"10.10.10.10"};
    hcclCommunicator.userRankSize_ = 1;
    std::vector<RankInfo_t> rankLists = {};
    RankInfo_t node;
    node.rankId = 0;
    node.deviceInfo.port = 50000;
    node.deviceInfo.vnicPort = 50001;
    rankLists.push_back(node);
    HcclResult ret;
    ret = hcclCommunicator.SetRanksPort(rankLists);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclCommTest, hcclCommAttr_SetRanksPort)
{
    HcclCommunicatorAttrs hcclCommunicator;
    MOCKER(GetExternalInputNpuPortSwitch).stubs().will(returnValue(true));
    HcclIpAddress localIp{"10.10.10.10"};
    hcclCommunicator.userRankSize_ = 1;
    std::vector<RankInfo_t> rankLists = {};
    RankInfo_t node;
    node.rankId = 0;
    node.deviceInfo.port = 50000;
    node.deviceInfo.vnicPort = 50001;
    rankLists.push_back(node);
    HcclResult ret;
    ret = hcclCommunicator.SetRanksPort(rankLists);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclCommTest, hcclComm_ReleasePreemptSocket)
{
    MOCKER_CPP(&PreemptPortManager::Release).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(HcclNetCloseDev).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(HcclNetDeInit).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetPairDevicePhyId).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceIndexByPhyId).stubs().will(returnValue(HCCL_SUCCESS));

    HcclCommunicator hcclCommunicator;
    HcclIpAddress remoteIp1{"10.10.10.10"};
    std::shared_ptr<HcclSocket> listenSocket1(
        new (std::nothrow) HcclSocket("my tag1", nullptr, remoteIp1, 0, HcclSocketRole::SOCKET_ROLE_SERVER));
    HcclNetDevCtx ctx1;
    hcclCommunicator.commPortConfig_.devNicListen = std::make_pair(listenSocket1, ctx1);

    HcclIpAddress remoteIp2{"10.10.10.11"};
    std::shared_ptr<HcclSocket> listenSocket2(
        new (std::nothrow) HcclSocket("my tag2", nullptr, remoteIp2, 0, HcclSocketRole::SOCKET_ROLE_SERVER));
    HcclNetDevCtx ctx2;
    hcclCommunicator.commPortConfig_.devVnicListen = std::make_pair(listenSocket2, ctx2);

    HcclIpAddress remoteIp3{"10.10.10.12"};
    std::shared_ptr<HcclSocket> listenSocket3(
        new (std::nothrow) HcclSocket("my tag3", nullptr, remoteIp3, 0, HcclSocketRole::SOCKET_ROLE_SERVER));
    HcclNetDevCtx ctx3;
    hcclCommunicator.commPortConfig_.backupDevNicListen = std::make_pair(listenSocket3, ctx3);

    hcclCommunicator.deviceBackUpLogicId_ = 1;

    HcclResult ret;
    ret = hcclCommunicator.ReleasePreemptSocket();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcclCommTest, hcclComm_InitRankInfoSubGroup_devPortSwitchOn_branch)
{
    MOCKER_CPP(&HcclCommunicatorAttrs::GetInlineReduceSwitchOn).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER_CPP(&HcclCommunicatorAttrs::GetMeshAggregationRankSize).stubs().with(mockcpp::any()).will(returnValue(1));
    MOCKER_CPP(&HcclCommunicatorAttrs::GetUsedRdmaLevel0).stubs().with(mockcpp::any()).will(returnValue(false));
    MOCKER_CPP(&HcclCommunicator::SetWorldGroupInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(SetRetryEnable).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(IsHostUseDevNic).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    std::vector<RankInfo> rankLists = {};
    RankInfo node;
    node.worldRank = 0;
    node.userRank = 0;
    rankLists.push_back(node);
    WorldGroupInfo groupCommonData;
    groupCommonData.devPortSwitchOn = true;

    HcclCommunicator hcclCommunicator;
    hcclCommunicator.commPortConfig_.devPortSwitchOn = true;
    hcclCommunicator.vnicRanksPort_.push_back(50000);
    HcclResult ret;
    ret = hcclCommunicator.InitRankInfoSubGroup(groupCommonData);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcclCommTest, ut_InitRaResource_notSupportChangelink)
{
    std::unique_ptr<HcclCommunicator> communicator(new (std::nothrow) HcclCommunicator());
    std::unique_ptr<HcclSocketManager> socketManager(
        new (std::nothrow) HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, 0, 0, 0));
    u32 devicePhyId = 0;
    HcclNetDevCtx vnicPortCtx;
    HcclResult ret
        = HcclNetOpenDev(&vnicPortCtx, NicType::DEVICE_NIC_TYPE, devicePhyId, devicePhyId, HcclIpAddress(devicePhyId));
    EXPECT_EQ(ret, HCCL_SUCCESS);
    communicator->devicePhyId_ = devicePhyId;
    communicator->netDevCtxMap_.insert(make_pair(HcclIpAddress(devicePhyId), vnicPortCtx));
    communicator->socketManager_ = std::move(socketManager);
    communicator->userRankSize_ = 2;
    communicator->nicDeployment_ = NICDeployment::NIC_DEPLOYMENT_DEVICE;
    communicator->isHaveCpuRank_ = false;

    MOCKER(IsHostUseDevNic).stubs().with(outBound(true)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::IsEnableBackupLink).stubs().will(returnValue(true));
    MOCKER(Is310PDevice).stubs().will(returnValue(true));
    MOCKER_CPP(&HcclCommunicator::InitNic).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetPairDevicePhyId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    // rts无法访问备用die（客户自定义） -> 不支持借轨
    MOCKER(hrtGetDeviceIndexByPhyId).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_RUNTIME));
    // 网络资源初始化失败（不初始化备用资源）
    ret = communicator->InitRaResource();
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    HcclNetCloseDev(vnicPortCtx);
    GlobalMockObject::verify();
}

TEST_F(HcclCommTest, hcclComm_InitRaResource_devVnicSocket_branch)
{
    MOCKER(IsHostUseDevNic).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(HcclNetInit).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(Is310PDevice).stubs().with(mockcpp::any()).will(returnValue(false));
    MOCKER_CPP(&HcclCommunicator::IsEnableBackupLink).stubs().with(mockcpp::any()).will(returnValue(false));
    MOCKER_CPP(&HcclCommunicator::InitSocketManager).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicatorAttrs::IsEnableRoce).stubs().with(mockcpp::any()).will(returnValue(false));
    MOCKER_CPP(&HcclSocketManager::ServerInit).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclSocketManager::ServerDeInit, HcclResult(HcclSocketManager::*)(const HcclNetDevCtx, u32))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicatorAttrs::GenSupportRdmaLite).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicatorAttrs::GetSupportRdmaLite).stubs().with(mockcpp::any()).will(returnValue(false));
    MOCKER_CPP(&HcclCommunicator::ReleasePreemptSocket).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclCommunicator hcclCommunicator;
    hcclCommunicator.userRankSize_ = 2;
    hcclCommunicator.devicePhyId_ = 0;
    hcclCommunicator.nicDeployment_ = NICDeployment::NIC_DEPLOYMENT_HOST;
    hcclCommunicator.isHaveCpuRank_ = false;
    hcclCommunicator.socketManager_.reset(new (std::nothrow)
                                              HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_HOST, 0, 0, 0));

    HcclIpAddress remoteIp{"10.10.10.11"};
    std::shared_ptr<HcclSocket> listenSocket(
        new (std::nothrow) HcclSocket("my tag2", nullptr, remoteIp, 0, HcclSocketRole::SOCKET_ROLE_SERVER));
    HcclIpAddress localIp{"127.0.0.1"};
    listenSocket->localIp_ = localIp;
    listenSocket->localPort_ = 50000;

    HcclResult ret;

    HcclNetDevCtx ctx = nullptr;
    ret = HcclNetOpenDev(&ctx, NicType::DEVICE_NIC_TYPE, 0, 0, localIp);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclCommunicator.commPortConfig_.devVnicListen = std::make_pair(listenSocket, ctx);

    ret = hcclCommunicator.InitRaResource();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hcclCommunicator.GetRanksPort();

    HcclNetCloseDev(ctx);

    hcclCommunicator.raResourceInit_ = false;
    hcclCommunicator.nicInitialized_ = 0;
    GlobalMockObject::verify();
}

TEST_F(HcclCommTest, hcclComm_InitNic_IsEnableBackupLink_branch1)
{
    setenv("HCCL_INTRA_ROCE_ENABLE", "1", 1);
    MOCKER_CPP(&HcclCommunicator::IsEnableBackupLink).stubs().with(mockcpp::any()).will(returnValue(true));
    MOCKER_CPP(&HcclSocketManager::ServerInit).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclSocketManager::ServerDeInit, HcclResult(HcclSocketManager::*)(const HcclNetDevCtx, u32))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicatorAttrs::SetNeedInitNicFlag).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(Is310PDevice).stubs().with(mockcpp::any()).will(returnValue(false));
    MOCKER_CPP(&HcclCommunicator::ReleasePreemptSocket).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcclIpAddress remoteIp{"10.10.10.11"};
    std::shared_ptr<HcclSocket> listenSocket(
        new (std::nothrow) HcclSocket("my tag2", nullptr, remoteIp, 0, HcclSocketRole::SOCKET_ROLE_SERVER));
    HcclIpAddress localIp{"127.0.0.1"};
    listenSocket->localIp_ = localIp;
    listenSocket->localPort_ = 50000;

    HcclNetDevCtx ctx1;
    HcclResult ret = HcclNetOpenDev(&ctx1, NicType::DEVICE_NIC_TYPE, 0, 0, HcclIpAddress("1.1.1.1"));

    HcclIpAddress remoteIp2{"10.10.10.12"};
    std::shared_ptr<HcclSocket> listenSocket2(
        new (std::nothrow) HcclSocket("my tag2", nullptr, remoteIp2, 0, HcclSocketRole::SOCKET_ROLE_SERVER));
    HcclIpAddress localIp2{"127.0.0.1"};
    listenSocket2->localIp_ = localIp2;
    listenSocket2->localPort_ = 50001;
    HcclNetDevCtx ctx2;

    HcclCommunicator hcclCommunicator;
    hcclCommunicator.nicDeployment_ = NICDeployment::NIC_DEPLOYMENT_RESERVED;
    hcclCommunicator.commPortConfig_.devNicListen = std::make_pair(listenSocket, ctx1);
    hcclCommunicator.commPortConfig_.backupDevNicListen = std::make_pair(listenSocket2, ctx2);

    ret = hcclCommunicator.InitNic();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_IF_BASE_PORT", "50000", 1);
    InitExternalInput();
    hcclCommunicator.GetHostPort(0);

    unsetenv("HCCL_INTRA_ROCE_ENABLE");
    InitExternalInput();
    hcclCommunicator.GetHostPort(0);

    hcclCommunicator.nicInitialized_ = 0;
    hcclCommunicator.raResourceInit_ = false;
    ResetInitState();

    GlobalMockObject::verify();
}

TEST_F(HcclCommTest, hcclComm_InitRankInfoSubGroup_devicePortSwitchOn)
{
    MOCKER_CPP(&HcclCommunicatorAttrs::SethbRankInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicatorAttrs::TransformRankList).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicatorAttrs::SetServerNum).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicatorAttrs::SetModuleInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicatorAttrs::SetSuperPodInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicatorAttrs::SetLocalRankInfoSubGroup)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicatorAttrs::SetInterModeInSuperPod)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicatorAttrs::UpdateNicList).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicatorAttrs::CheckLocalRankInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicatorAttrs::CalAndSetMeshAggRankSize)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicatorAttrs::IsEnableRoce).stubs().with(mockcpp::any()).will(returnValue(false));
    MOCKER_CPP(&HcclCommunicatorAttrs::SetWorldGroupInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(HcclCheckLogLevel).stubs().with(mockcpp::any()).will(returnValue(false));
    MOCKER(IsHostUseDevNic).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(
        &HcclCommunicatorAttrs::SetInnerServerAverageDevice,
        HcclResult(HcclCommunicatorAttrs::*)(const std::vector<RankInfo>& rankList))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(
        &HcclCommunicatorAttrs::InitTopoInfo,
        HcclResult(HcclCommunicatorAttrs::*)(const std::vector<RankInfo>& rankList))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    HcclCommunicatorAttrs hcomattr;

    std::vector<RankInfo> rankLists;
    RankInfo node;
    node.worldRank = 0;
    node.userRank = 0;
    rankLists.push_back(node);
    WorldGroupInfo groupCommonData;
    groupCommonData.devPortSwitchOn = true;

    hcomattr.vnicRanksPort_.push_back(50000);
    HcclResult ret;
    ret = hcomattr.InitRankInfoSubGroup(rankLists, groupCommonData);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcclCommTest, hcclComm_SetWorldGroupInfo)
{
    HcclCommunicatorAttrs hcclComm;

    std::unordered_map<std::string, std::map<u32, HcclIpAddress>> phyIdNicInfoMap;
    std::vector<RankInfo> worldRankInfoList;
    std::vector<u32> nicRanksPort;
    std::vector<u32> vnicRanksPort;
    vnicRanksPort.push_back(50000);

    HcclResult ret;
    ret = hcclComm.SetWorldGroupInfo(phyIdNicInfoMap, worldRankInfoList, nicRanksPort, vnicRanksPort);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcclCommTest, hcclComm_AllGatherV_call)
{
    MOCKER_CPP(&HcclCommunicator::IsAtomicInit).stubs().will(returnValue(true));
    MOCKER_CPP(&HcclCommunicator::ExecOp).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCallbackTask::CallbackRegStream).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::StreamIsCapture).stubs().with(mockcpp::any()).will(returnValue(false));
    MOCKER_CPP(&hcclImpl::SetHDCModeInfo).stubs().will(returnValue(HCCL_SUCCESS));

    HcclCommunicator hcclCommunicator;
    CCLBufferManager cclBufferManager;
    HcclDispatcher dispatcher;
    HcclDispatcher vDispatcher;
    hcclCommunicator.implAlg_.reset(new (std::nothrow) HcclAlg(cclBufferManager, dispatcher, vDispatcher));
    hcclCommunicator.userRankSize_ = 2;

    string strTag = "allgatherv_tag_test";
    int sendBuff[4] = {0};
    u64 sendCount = 4;
    int recvBuff[8];
    u64 recvCount[4] = {4, 4};
    u64 rdispls[4] = {4, 4};
    aclrtStream stream;
    aclError ret = aclrtCreateStream(&stream);
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = hcclCommunicator.AllGatherV(
        strTag, sendBuff, sendCount, recvBuff, recvCount, rdispls, HcclDataType::HCCL_DATA_TYPE_INT32, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcclCommTest, hcclComm_AllGatherVOutPlace_call)
{
    MOCKER_CPP(&HcclCommunicator::IsAtomicInit).stubs().will(returnValue(true));
    MOCKER_CPP(&HcclCommunicator::ExecOp).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCallbackTask::CallbackRegStream).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::StreamIsCapture).stubs().with(mockcpp::any()).will(returnValue(false));
    MOCKER_CPP(&hcclImpl::SetHDCModeInfo).stubs().will(returnValue(HCCL_SUCCESS));

    HcclCommunicator hcclCommunicator;
    CCLBufferManager cclBufferManager;
    HcclDispatcher dispatcher;
    HcclDispatcher vDispatcher;
    hcclCommunicator.implAlg_.reset(new (std::nothrow) HcclAlg(cclBufferManager, dispatcher, vDispatcher));
    hcclCommunicator.userRankSize_ = 2;

    string strTag = "allgathervoutplace_tag_test";
    int sendBuff[4] = {0};
    u64 sendCount = 4;
    int recvBuff[8];
    u64 recvCount[4] = {4, 4};
    u64 rdispls[4] = {4, 4};
    aclrtStream stream;
    aclError ret = aclrtCreateStream(&stream);
    EXPECT_EQ(ret, ACL_SUCCESS);
    ret = hcclCommunicator.AllGatherVOutPlace(
        strTag, sendBuff, recvBuff, sendCount, recvCount, rdispls, HcclDataType::HCCL_DATA_TYPE_INT32, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

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
#define private public
#define protected public
#include "tbe_vector_reduce/tbe_vector_reduce.h"
#include "tbe_vector_reduce/tbe_crack_cleared.h"
#undef protected
#undef private
#include <stdio.h>
#define private public
#define protected public
#include "hcom_ops_kernel_info_store.h"
#include "hcom_ops_kernel_builder.h"
#include "hcom_graph_optimizer.h"
#include "hvd_ops_kernel_info_store.h"
#include "hccl_communicator.h"
#include "network_manager_pub.h"
#undef protected
#undef private
#include "hccl/base.h"
#include <hccl/hccl_types.h>

#include "stream_pub.h"
#include "mem_host_pub.h"
#include "mem_device_pub.h"
#include "hccl_comm_pub.h"
#include "sal.h"
#include "hccl_impl.h"
#include "llt_hccl_stub_pub.h"
#include "externalinput.h"
#include "config.h"
#include "topoinfo_ranktableParser_pub.h"

#include "plugin_manager.h"
#include "external/ge/ge_api_types.h"  // ge对内options
#include "framework/common/ge_types.h" // ge对外options
#include "hcom_pub.h"
#include "hccl/hcom.h"
#include "hccl/hcom_executor.h"
#include "ranktable/v80_rank_table.h"
#include <iostream>
#include <fstream>
#include "graph/utils/node_utils.h"
#include "llt_hccl_stub_ge.h"
#include "workspace_resource_impl.h"
#include "workspace_mem.h"
#include "../misc/ut_rank_table.h"

using namespace std;
using namespace hccl;

static nlohmann::json allreduce_topo_switch_connect
    = {{"topology type", "switch connection"},
       {"topology desc",
        {{{"node type", "TOR"},
          {"node name", "tor0"},
          {"link info",
           {{{"link id", "0"},
             {"local port name", "port0"},
             {"local ip address", "100.100.83.1"},
             {"opposite type", "SERVER"},
             {"opposite name", "server0"},
             {"opposite port name", "eth8"},
             {"opposite ip address", "100.100.83.178"}}}}}}}};

class HcomKernelInfoTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        nlohmann::json rank_table
            = {{"status", "completed"},
               {"deploy_mode", "lab"},
               {"device_num", "4"},
               {"server_num", "2"},
               {"boardType", "0"},
               {"para_plane_location", "device"},
               {"para_plane_nic_num", "2"},
               {"para_plane_nic_name", {"eth0", "eth1"}},
               {"instance_count", "4"},
               {"device_count", "4"},
               {"instance_list",
                {{{"pod_name", ""},
                  {"rank_id", "0"},
                  {"server_id", "10.0.0.10"},
                  {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.12"}, {"ref_ip", "192.168.10.13"}}}}},
                 {{"pod_name", ""},
                  {"rank_id", "1"},
                  {"server_id", "10.0.0.10"},
                  {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.1.12"}, {"ref_ip", "192.168.11.13"}}}}},
                 {{"pod_name", ""},
                  {"rank_id", "2"},
                  {"server_id", "10.0.0.11"},
                  {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.14"}, {"ref_ip", "192.168.10.15"}}}}},
                 {{"pod_name", ""},
                  {"rank_id", "3"},
                  {"server_id", "10.0.0.11"},
                  {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.1.14"}, {"ref_ip", "192.168.11.15"}}}}}}},
               {"server_list",
                {
                    {{"server_id", "192.168.10.2"},
                     {"para_plane_info",
                      {{{"eth1", "192.168.210.2"}, {"ref_ip", "192.168.210.1"}},
                       {{"eth0", "192.168.200.2"}, {"ref_ip", "192.168.200.1"}}}}

                    },
                    {{"server_id", "192.168.10.3"},
                     {"para_plane_info",
                      {{{"eth0", "192.168.200.3"}, {"ref_ip", "192.168.200.1"}},
                       {{"eth1", "192.168.210.3"}, {"ref_ip", "192.168.210.1"}}}}

                    },

                }}};
        char file_name[] = "./ut_HcomKernelInfoTest.json";

        WriteRankTableFile(file_name, rank_table);

        std::cout << "\033[36m--HcomKernelInfoTest SetUP--\033[0m" << std::endl;
    }
    static void TearDownTestCase()
    {
        char file_name[] = "./ut_HcomKernelInfoTest.json";
        remove(file_name);
        std::cout << "\033[36m--HcomKernelInfoTest TearDown--\033[0m" << std::endl;
    }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }
};

class NodeTest : public ge::Node {
public:
    NodeTest() { ; };
    ~NodeTest() { ; };
};

ge::graphStatus
OffloadGetOption1(ge::GEThreadLocalContext* that, const std::string& optionExec, std::string& dumpDebugValue)
{
    nlohmann::json group_list = {"rank_map1", {{"logic_rank_id", "1"}}};
    dumpDebugValue = group_list.dump();
    return ge::GRAPH_SUCCESS;
}

#if 1
TEST_F(HcomKernelInfoTest, ut_LoadTask)
{
    s8* sendbuf = (s8*)sal_malloc(10 * sizeof(s8));
    sal_memset(sendbuf, 10 * sizeof(s8), 0, 10 * sizeof(s8));
    s8* recv = (s8*)sal_malloc(10 * sizeof(s8));
    sal_memset(recv, 10 * sizeof(s8), 0, 10 * sizeof(s8));

    void* dumpbuf;
    dumpbuf = sal_malloc(10 * sizeof(s8));
    sal_memset(dumpbuf, 10 * sizeof(s8), 0, 10 * sizeof(s8));
std:
    vector<void*> globalWorkSpaceAddr;
    globalWorkSpaceAddr.push_back(dumpbuf);

    nlohmann::json rank_table = rank_table_910_1server_1rank;
    char file_name_t[] = "./ut_LoadTask.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();
    u32 ret1 = hrtSetDevice(0);
    EXPECT_EQ(ret1, HCCL_SUCCESS);

    // 走1910 4pring
    char* rank_table_file = "./ut_LoadTask.json";
    char* rank_ID = "0";

    ret1 = HcomInitByFile(rank_table_file, rank_ID);
    EXPECT_EQ(ret1, HCCL_SUCCESS);
    string groupName = "HCOM_GROUP";
    u32 rankNum1 = 1;
    u32 rankIds = 0;
    ret1 = HcomCreateGroup(groupName.c_str(), rankNum1, &rankIds);
    EXPECT_EQ(ret1, HCCL_SUCCESS);
    rtStream_t stream;
    ge::Status ret;
    HcomOpsKernelInfoStore hcomKernelInfo;

    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    ge::GETaskInfo task;
    ge::GETaskKernelHcclInfo hcclInfo;
    task.kernelHcclInfo.push_back(hcclInfo);
    HCCL_KERNEL_INFO_PRIVATE_DEF privateDefBuf;
    strcpy_s((char*)privateDefBuf.group, 128, "HCOM_GROUP");
    privateDefBuf.nodeNameHash = 123456;
    privateDefBuf.graphId = 1;
    privateDefBuf.srcRank = 1;
    privateDefBuf.destRank = 2;
    privateDefBuf.srTag = 3;
    privateDefBuf.originalGraphShapeType = 0;
    privateDefBuf.dataType = HCCL_DATA_TYPE_FP32;
    privateDefBuf.tensorNum = 3;
    privateDefBuf.privateDefSize = sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF);
    privateDefBuf.needMapRank = true;
    privateDefBuf.isOfflineComp = true;
    privateDefBuf.devType = DevType::DEV_TYPE_910;
    privateDefBuf.aivCoreLimit = 48;
    task.type = RT_MODEL_TASK_HCCL;
    task.stream = stream;

    std::int64_t offset[privateDefBuf.tensorNum] = {33280, 33792, 34304};
    std::int64_t size[privateDefBuf.tensorNum] = {400, 40, 80};
    size_t tensorInfoSize = sizeof(hcclKernelInfoPrivateDef) + privateDefBuf.tensorNum * sizeof(int64_t) * 2;
    void* privateDefPtr = sal_malloc(tensorInfoSize);
    memset(privateDefPtr, 0, tensorInfoSize);
    memcpy(privateDefPtr, (void*)&privateDefBuf.group[0], sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF));
    memcpy(
        reinterpret_cast<int64_t*>(reinterpret_cast<char*>(privateDefPtr) + sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF)),
        offset, sizeof(offset));
    memcpy(
        reinterpret_cast<int64_t*>(
            reinterpret_cast<char*>(privateDefPtr) + sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF) + sizeof(offset)),
        size, sizeof(size));

    task.privateDef = privateDefPtr;
    task.privateDefLen = (uint32_t)tensorInfoSize;
    task.kernelHcclInfo[0].count = 100;
    task.kernelHcclInfo[0].dataType = HCCL_DATA_TYPE_FP32;
    task.kernelHcclInfo[0].hccl_type = HCCL_KERNEL_OP_TYPE_BROADCAST;
    task.kernelHcclInfo[0].inputDataAddr = sendbuf;
    task.kernelHcclInfo[0].outputDataAddr = recv;
    task.kernelHcclInfo[0].opType = HCCL_REDUCE_SUM;
    task.kernelHcclInfo[0].rootId = 0;
    task.kernelHcclInfo[0].hcclQosCfg = INVALID_QOSCFG;
    task.kernelHcclInfo[0].global_workspace_addr = globalWorkSpaceAddr;
    task.needRefresh = false;

    u64 memSize = HCCL_WORKSPACE_MEM_32_KB;
    DeviceMem workSpaceMem = DeviceMem::alloc(memSize);
    task.kernelHcclInfo[0].workSpaceAddr = workSpaceMem.ptr();
    task.kernelHcclInfo[0].workSpaceMemSize = workSpaceMem.size();
    // -------------------common test----------------
    MOCKER(HcomBroadcast).expects(atMost(6)).will(returnValue(HCCL_SUCCESS));

    MOCKER(HcomSetWorkspaceResource).expects(atMost(6)).will(returnValue(HCCL_SUCCESS));

    MOCKER(HcomGetInitStatus).stubs().with(outBound(true)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemAsyncCopy).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemcpyAddrAsync).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(InitGroup).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::OffloadStreamManager::GetSlaves)
        .stubs()
        .will(returnValue(vector<Stream>(1, Stream(StreamType::STREAM_TYPE_OFFLINE))));

    MOCKER_CPP(&TbeReduce::TbeVectorReduce::GetTilingDataDevMem)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TbeReduce::TbeCrackCleared::ExecuteKernelLaunch)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    DevType devType = DevType::DEV_TYPE_910;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));

    // type 非HCCL fail

    u32 taskID = 0;

    task.type = 0;
    task.id = taskID++;
    ret = hcomKernelInfo.LoadTask(task);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    task.type = RT_MODEL_TASK_HCCL;
    // privateDefLen invalid
    task.privateDefLen = 0;
    task.id = taskID++;
    ret = hcomKernelInfo.LoadTask(task);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    task.privateDefLen = (uint32_t)tensorInfoSize;
    // hccl_type invalid
    task.kernelHcclInfo[0].hccl_type = "";
    task.id = taskID++;
    ret = hcomKernelInfo.LoadTask(task);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    task.kernelHcclInfo[0].hccl_type = HCCL_KERNEL_OP_TYPE_BROADCAST;
    // date type invalid
    task.kernelHcclInfo[0].dataType = HCCL_DATA_TYPE_RESERVED;
    privateDefBuf.dataType = HCCL_DATA_TYPE_RESERVED;
    task.id = taskID++;
    ret = hcomKernelInfo.LoadTask(task);
    // EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    task.kernelHcclInfo[0].dataType = HCCL_DATA_TYPE_FP32;
    privateDefBuf.dataType = HCCL_DATA_TYPE_FP32;

    task.kernelHcclInfo[0].workSpaceAddr = NULL;
    task.id = taskID++;
    ret = hcomKernelInfo.LoadTask(task);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    task.kernelHcclInfo[0].workSpaceAddr = workSpaceMem.ptr();

    task.kernelHcclInfo[0].workSpaceMemSize = 0;
    task.id = taskID++;
    ret = hcomKernelInfo.LoadTask(task);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    task.kernelHcclInfo[0].workSpaceMemSize = workSpaceMem.size();
    GlobalMockObject::verify();

    // -------------------HcomBroadcast test----------------
    tensorInfoSize = sizeof(hcclKernelInfoPrivateDef);
    privateDefBuf.originalGraphShapeType = ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE;
    privateDefBuf.tensorNum = 0;
    void* privateDefPtr_broadcast = sal_malloc(tensorInfoSize);
    memset(privateDefPtr_broadcast, 0, tensorInfoSize);
    memcpy(privateDefPtr_broadcast, (void*)&privateDefBuf.group[0], sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF));

    task.privateDef = privateDefPtr_broadcast;
    task.privateDefLen = sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF);

    task.kernelHcclInfo[0].hccl_type = HCCL_KERNEL_OP_TYPE_BROADCAST;

    MOCKER(HcomBroadcast).expects(atMost(3)).will(returnValue(HCCL_E_PARA)).then(returnValue(HCCL_SUCCESS));

    MOCKER(HcomSetWorkspaceResource).expects(atMost(3)).will(returnValue(HCCL_SUCCESS));

    MOCKER(HcomGetInitStatus).stubs().with(outBound(true)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemAsyncCopy).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemcpyAddrAsync).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(InitGroup).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::OffloadStreamManager::GetSlaves)
        .stubs()
        .will(returnValue(vector<Stream>(1, Stream(StreamType::STREAM_TYPE_OFFLINE))));

    MOCKER_CPP(&TbeReduce::TbeVectorReduce::GetTilingDataDevMem)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TbeReduce::TbeCrackCleared::ExecuteKernelLaunch)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    // HcomBroadcast fail
    task.id = taskID++;
    ret = hcomKernelInfo.LoadTask(task);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    // load task success
    task.id = taskID++;
    ret = hcomKernelInfo.LoadTask(task);
    EXPECT_EQ(ret, ge::SUCCESS);
    GlobalMockObject::verify();

    // -------------------HcomAllReduce test----------------
    tensorInfoSize = sizeof(hcclKernelInfoPrivateDef);
    privateDefBuf.originalGraphShapeType = ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE;
    privateDefBuf.tensorNum = 0;
    void* privateDefPtr_allreduce = sal_malloc(tensorInfoSize);
    memset(privateDefPtr_allreduce, 0, tensorInfoSize);
    memcpy(privateDefPtr_allreduce, (void*)&privateDefBuf.group[0], sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF));

    task.privateDef = privateDefPtr_allreduce;
    task.privateDefLen = sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF);

    task.kernelHcclInfo[0].hccl_type = HCCL_KERNEL_OP_TYPE_ALLREDUCE;

    MOCKER(HcomAllReduce).expects(atMost(3)).will(returnValue(HCCL_E_PARA)).then(returnValue(HCCL_SUCCESS));

    MOCKER(HcomSetWorkspaceResource).expects(atMost(3)).will(returnValue(HCCL_SUCCESS));

    MOCKER(HcomGetInitStatus).stubs().with(outBound(true)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemAsyncCopy).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemcpyAddrAsync).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(InitGroup).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::OffloadStreamManager::GetSlaves)
        .stubs()
        .will(returnValue(vector<Stream>(1, Stream(StreamType::STREAM_TYPE_OFFLINE))));

    MOCKER_CPP(&TbeReduce::TbeVectorReduce::GetTilingDataDevMem)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TbeReduce::TbeCrackCleared::ExecuteKernelLaunch)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    // HcomAllReduce fail
    task.id = taskID++;
    ret = hcomKernelInfo.LoadTask(task);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    // reduce type invalid
    task.kernelHcclInfo[0].opType = HCCL_REDUCE_RESERVED;
    task.id = taskID++;
    ret = hcomKernelInfo.LoadTask(task);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    // load task success
    task.kernelHcclInfo[0].opType = HCCL_REDUCE_SUM;
    task.id = taskID++;
    ret = hcomKernelInfo.LoadTask(task);
    EXPECT_EQ(ret, ge::SUCCESS);
    GlobalMockObject::verify();

    // -------------------HcomAllGather test----------------
    tensorInfoSize = sizeof(hcclKernelInfoPrivateDef);
    privateDefBuf.originalGraphShapeType = ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE;
    privateDefBuf.tensorNum = 0;
    void* privateDefPtr_allgather = sal_malloc(tensorInfoSize);
    memset(privateDefPtr_allgather, 0, tensorInfoSize);
    memcpy(privateDefPtr_allgather, (void*)&privateDefBuf.group[0], sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF));

    task.privateDef = privateDefPtr_allgather;
    task.privateDefLen = sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF);

    task.kernelHcclInfo[0].hccl_type = HCCL_KERNEL_OP_TYPE_ALLGATHER;
    MOCKER(HcomAllGather).expects(atMost(2)).will(returnValue(HCCL_E_PARA)).then(returnValue(HCCL_SUCCESS));

    MOCKER(HcomSetWorkspaceResource).expects(atMost(2)).will(returnValue(HCCL_SUCCESS));

    MOCKER(HcomGetInitStatus).stubs().with(outBound(true)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemAsyncCopy).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemcpyAddrAsync).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(InitGroup).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::OffloadStreamManager::GetSlaves)
        .stubs()
        .will(returnValue(vector<Stream>(1, Stream(StreamType::STREAM_TYPE_OFFLINE))));

    MOCKER_CPP(&TbeReduce::TbeVectorReduce::GetTilingDataDevMem)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TbeReduce::TbeCrackCleared::ExecuteKernelLaunch)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    // HcomAllGather fail
    task.id = taskID++;
    ret = hcomKernelInfo.LoadTask(task);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    // load task success
    task.id = taskID++;
    ret = hcomKernelInfo.LoadTask(task);
    EXPECT_EQ(ret, ge::SUCCESS);
    GlobalMockObject::verify();

    // -------------------HcomAllGatherV test----------------
    u64 countAllgatherv = 10;
    memSize = 2 * countAllgatherv * sizeof(int32_t);

    DeviceMem paramDevMem = DeviceMem::alloc(memSize);

    strcpy_s((char*)privateDefBuf.group, 128, "hccl_world_group");
    privateDefBuf.nodeNameHash = 1234567890;
    privateDefBuf.graphId = 1;
    privateDefBuf.destRank = 1;
    privateDefBuf.originalGraphShapeType = 0;
    privateDefBuf.originalGraphShapeType = ORIGINAL_GRAPH_KNOWNSHAPE_TYPE;

    HCCL_ALLGATHERV_KERNEL_INFO_PRIVATE_DEF allgathervPrivateDefBuf(privateDefBuf);
    allgathervPrivateDefBuf.paramsInfo.recvDispls[0] = 10;
    allgathervPrivateDefBuf.paramsInfo.recvCounts[0] = 10;

    hcclInfo.dataType = HCCL_DATA_TYPE_FP32;
    hcclInfo.hccl_type = HCCL_KERNEL_OP_TYPE_ALLGATHERV;
    hcclInfo.inputDataAddr = paramDevMem.ptr();
    hcclInfo.outputDataAddr = paramDevMem.ptr();
    hcclInfo.workSpaceAddr = workSpaceMem.ptr();
    hcclInfo.workSpaceMemSize = workSpaceMem.size();
    ge::GETaskInfo newTask;
    newTask.kernelHcclInfo.push_back(hcclInfo);
    newTask.stream = stream;
    newTask.streamID = 1;
    newTask.type = RT_MODEL_TASK_HCCL;
    newTask.privateDef = reinterpret_cast<void*>(&allgathervPrivateDefBuf);
    newTask.privateDefLen = sizeof(HCCL_ALLGATHERV_KERNEL_INFO_PRIVATE_DEF);
    MOCKER(HcomAllGatherV).expects(atMost(2)).will(returnValue(HCCL_E_PARA)).then(returnValue(HCCL_SUCCESS));

    MOCKER(HcomSetWorkspaceResource).expects(atMost(2)).will(returnValue(HCCL_SUCCESS));

    MOCKER(HcomGetInitStatus).stubs().with(outBound(true)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemAsyncCopy).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemcpyAddrAsync).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(InitGroup).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::OffloadStreamManager::GetSlaves)
        .stubs()
        .will(returnValue(vector<Stream>(1, Stream(StreamType::STREAM_TYPE_OFFLINE))));

    MOCKER_CPP(&TbeReduce::TbeVectorReduce::GetTilingDataDevMem)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TbeReduce::TbeCrackCleared::ExecuteKernelLaunch)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    // HcomAllGather fail
    newTask.id = taskID++;
    ret = hcomKernelInfo.LoadTask(newTask);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    // load task success
    newTask.id = taskID++;
    ret = hcomKernelInfo.LoadTask(newTask);
    EXPECT_EQ(ret, ge::SUCCESS);
    GlobalMockObject::verify();

    // -------------------HcomReduceScatter test----------------
    tensorInfoSize = sizeof(hcclKernelInfoPrivateDef);
    privateDefBuf.originalGraphShapeType = ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE;
    privateDefBuf.tensorNum = 0;
    void* privateDefPtr_reducescatter = sal_malloc(tensorInfoSize);
    memset(privateDefPtr_reducescatter, 0, tensorInfoSize);
    memcpy(privateDefPtr_reducescatter, (void*)&privateDefBuf.group[0], sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF));

    task.privateDef = privateDefPtr_reducescatter;
    task.privateDefLen = sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF);

    task.kernelHcclInfo[0].hccl_type = HCCL_KERNEL_OP_TYPE_REDUCESCATTER;
    MOCKER(HcomReduceScatter).expects(atMost(3)).will(returnValue(HCCL_E_PARA)).then(returnValue(HCCL_SUCCESS));

    MOCKER(HcomSetWorkspaceResource).expects(atMost(3)).will(returnValue(HCCL_SUCCESS));

    MOCKER(HcomGetInitStatus).stubs().with(outBound(true)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemAsyncCopy).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemcpyAddrAsync).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(InitGroup).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::OffloadStreamManager::GetSlaves)
        .stubs()
        .will(returnValue(vector<Stream>(1, Stream(StreamType::STREAM_TYPE_OFFLINE))));

    MOCKER_CPP(&TbeReduce::TbeVectorReduce::GetTilingDataDevMem)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TbeReduce::TbeCrackCleared::ExecuteKernelLaunch)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    // HcomReduceScatter fail
    task.id = taskID++;
    ret = hcomKernelInfo.LoadTask(task);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    // reduce type invalid
    task.kernelHcclInfo[0].opType = HCCL_REDUCE_RESERVED;
    task.id = taskID++;
    ret = hcomKernelInfo.LoadTask(task);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    // load task success
    task.kernelHcclInfo[0].opType = HCCL_REDUCE_SUM;
    task.id = taskID++;
    ret = hcomKernelInfo.LoadTask(task);
    EXPECT_EQ(ret, ge::SUCCESS);
    GlobalMockObject::verify();

    // -------------------HcomReduceScatterV test----------------
    u64 countReduceScatterV = 10;
    memSize = 2 * countReduceScatterV * sizeof(int32_t);

    paramDevMem = DeviceMem::alloc(memSize);

    strcpy_s((char*)privateDefBuf.group, 128, "hccl_world_group");
    privateDefBuf.nodeNameHash = 1234567890;
    privateDefBuf.graphId = 1;
    privateDefBuf.destRank = 1;
    privateDefBuf.originalGraphShapeType = 0;
    privateDefBuf.originalGraphShapeType = ORIGINAL_GRAPH_KNOWNSHAPE_TYPE;

    HCCL_REDUCESCATTERV_KERNEL_INFO_PRIVATE_DEF reducescattervPrivateDefBuf(privateDefBuf);
    reducescattervPrivateDefBuf.paramsInfo.sendCounts[0] = 10;
    reducescattervPrivateDefBuf.paramsInfo.recvCounts[0] = 10;

    hcclInfo.dataType = HCCL_DATA_TYPE_FP32;
    hcclInfo.hccl_type = HCCL_KERNEL_OP_TYPE_REDUCESCATTERV;
    hcclInfo.inputDataAddr = paramDevMem.ptr();
    hcclInfo.outputDataAddr = paramDevMem.ptr();
    hcclInfo.workSpaceAddr = workSpaceMem.ptr();
    hcclInfo.workSpaceMemSize = workSpaceMem.size();
    ge::GETaskInfo newTask_rs;
    newTask_rs.kernelHcclInfo.push_back(hcclInfo);
    newTask_rs.stream = stream;
    newTask_rs.streamID = 1;
    newTask_rs.type = RT_MODEL_TASK_HCCL;
    newTask_rs.privateDef = reinterpret_cast<void*>(&reducescattervPrivateDefBuf);
    newTask_rs.privateDefLen = sizeof(HCCL_REDUCESCATTERV_KERNEL_INFO_PRIVATE_DEF);
    newTask_rs.kernelHcclInfo[0].opType = HCCL_REDUCE_SUM;
    newTask_rs.kernelHcclInfo[0].hccl_type = HCCL_KERNEL_OP_TYPE_REDUCESCATTERV;

    MOCKER(HcomReduceScatterV).expects(atMost(2)).will(returnValue(HCCL_E_PARA)).then(returnValue(HCCL_SUCCESS));

    MOCKER(HcomSetWorkspaceResource).expects(atMost(2)).will(returnValue(HCCL_SUCCESS));

    MOCKER(HcomGetInitStatus).stubs().with(outBound(true)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemAsyncCopy).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemcpyAddrAsync).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(InitGroup).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::OffloadStreamManager::GetSlaves)
        .stubs()
        .will(returnValue(vector<Stream>(1, Stream(StreamType::STREAM_TYPE_OFFLINE))));

    MOCKER_CPP(&TbeReduce::TbeVectorReduce::GetTilingDataDevMem)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TbeReduce::TbeCrackCleared::ExecuteKernelLaunch)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    // HcomReduceScatterv fail
    newTask_rs.id = taskID++;
    ret = hcomKernelInfo.LoadTask(newTask_rs);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    // load task success
    newTask_rs.id = taskID++;
    ret = hcomKernelInfo.LoadTask(newTask_rs);
    EXPECT_EQ(ret, ge::SUCCESS);
    GlobalMockObject::verify();

    // -------------------HcomSend test----------------

    tensorInfoSize = sizeof(hcclKernelInfoPrivateDef);
    privateDefBuf.originalGraphShapeType = ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE;
    privateDefBuf.tensorNum = 0;
    void* privateDefPtr1 = sal_malloc(tensorInfoSize);
    memset(privateDefPtr1, 0, tensorInfoSize);
    memcpy(privateDefPtr1, (void*)&privateDefBuf.group[0], sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF));

    task.privateDef = privateDefPtr1;
    task.privateDefLen = sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF);

    task.kernelHcclInfo[0].hccl_type = HCCL_KERNEL_OP_TYPE_SEND;
    MOCKER(HcomGetRankId).expects(atMost(2)).will(returnValue(HCCL_SUCCESS));
    MOCKER(HcomSend).expects(atMost(2)).will(returnValue(HCCL_SUCCESS));
    MOCKER(HcomSetWorkspaceResource).expects(atMost(2)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemAsyncCopy).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemcpyAddrAsync).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(HcomGetInitStatus).stubs().with(outBound(true)).will(returnValue(HCCL_SUCCESS));

    MOCKER(InitGroup).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::OffloadStreamManager::GetSlaves)
        .stubs()
        .will(returnValue(vector<Stream>(1, Stream(StreamType::STREAM_TYPE_OFFLINE))));

    MOCKER_CPP(&TbeReduce::TbeVectorReduce::GetTilingDataDevMem)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TbeReduce::TbeCrackCleared::ExecuteKernelLaunch)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    // HcomSend fail
    // ret = hcomKernelInfo.LoadTask(task);
    // EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    // load task success
    task.id = taskID++;
    ret = hcomKernelInfo.LoadTask(task);
    EXPECT_EQ(ret, ge::SUCCESS);
    GlobalMockObject::verify();

    // -------------------HcomReceive test----------------
    task.kernelHcclInfo[0].hccl_type = HCCL_KERNEL_OP_TYPE_RECEIVE;
    MOCKER(HcomGetRankId).expects(atMost(2)).will(returnValue(HCCL_SUCCESS));
    MOCKER(HcomReceive).expects(atMost(2)).will(returnValue(HCCL_SUCCESS));
    MOCKER(HcomSetWorkspaceResource).expects(atMost(2)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemAsyncCopy).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemcpyAddrAsync).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(HcomGetInitStatus).stubs().with(outBound(true)).will(returnValue(HCCL_SUCCESS));

    MOCKER(InitGroup).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::OffloadStreamManager::GetSlaves)
        .stubs()
        .will(returnValue(vector<Stream>(1, Stream(StreamType::STREAM_TYPE_OFFLINE))));

    MOCKER_CPP(&TbeReduce::TbeVectorReduce::GetTilingDataDevMem)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TbeReduce::TbeCrackCleared::ExecuteKernelLaunch)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    // HcomReceive fail
    // ret = hcomKernelInfo.LoadTask(task);
    // EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    // load task success
    task.id = taskID++;
    ret = hcomKernelInfo.LoadTask(task);
    EXPECT_EQ(ret, ge::SUCCESS);
    MOCKER_CPP(&ge::GEThreadLocalContext::GetOption).stubs().will(invoke(OffloadGetOption1));
    // ret = hcomKernelInfo.LoadTask(task);
    // EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    GlobalMockObject::verify();

    // -------------------HcomReduce test----------------
    task.kernelHcclInfo[0].hccl_type = HCCL_KERNEL_OP_TYPE_REDUCE;
    MOCKER(HcomReduce).expects(atMost(3)).will(returnValue(HCCL_E_PARA)).then(returnValue(HCCL_SUCCESS));

    MOCKER(HcomSetWorkspaceResource).expects(atMost(3)).will(returnValue(HCCL_SUCCESS));

    MOCKER(HcomGetInitStatus).stubs().with(outBound(true)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemAsyncCopy).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemcpyAddrAsync).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(InitGroup).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::OffloadStreamManager::GetSlaves)
        .stubs()
        .will(returnValue(vector<Stream>(1, Stream(StreamType::STREAM_TYPE_OFFLINE))));

    MOCKER_CPP(&TbeReduce::TbeVectorReduce::GetTilingDataDevMem)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TbeReduce::TbeCrackCleared::ExecuteKernelLaunch)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    // HcomReduce fail
    task.id = taskID++;
    ret = hcomKernelInfo.LoadTask(task);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    // reduce type invalid
    task.kernelHcclInfo[0].opType = HCCL_REDUCE_RESERVED;
    task.id = taskID++;
    ret = hcomKernelInfo.LoadTask(task);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    // load task success
    task.kernelHcclInfo[0].opType = HCCL_REDUCE_SUM;
    task.id = taskID++;
    ret = hcomKernelInfo.LoadTask(task);
    EXPECT_EQ(ret, ge::SUCCESS);
    GlobalMockObject::verify();

    sal_free(sendbuf);
    sal_free(recv);
    sal_free(dumpbuf);
    sal_free(privateDefPtr);
    sal_free(privateDefPtr1);
    sal_free(privateDefPtr_broadcast);
    sal_free(privateDefPtr_allreduce);
    sal_free(privateDefPtr_allgather);
    sal_free(privateDefPtr_reducescatter);
    rt_ret = aclrtDestroyStream(stream);
    HcomDestroy();
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    remove(file_name_t);
}

TEST_F(HcomKernelInfoTest, ut_LoadTask_comm)
{
    s8* sendbuf = (s8*)sal_malloc(10 * sizeof(s8));
    sal_memset(sendbuf, 10 * sizeof(s8), 0, 10 * sizeof(s8));
    s8* recv = (s8*)sal_malloc(10 * sizeof(s8));
    sal_memset(recv, 10 * sizeof(s8), 0, 10 * sizeof(s8));

    void* dumpbuf;
    dumpbuf = sal_malloc(10 * sizeof(s8));
    sal_memset(dumpbuf, 10 * sizeof(s8), 0, 10 * sizeof(s8));
std:
    vector<void*> globalWorkSpaceAddr;
    globalWorkSpaceAddr.push_back(dumpbuf);

    nlohmann::json rank_table = rank_table_910_1server_1rank;
    char file_name_t[] = "./ut_LoadTask_com.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();
    u32 ret1 = hrtSetDevice(0);
    EXPECT_EQ(ret1, HCCL_SUCCESS);

    // 走1910 4pring
    char* rank_table_file = "./ut_LoadTask_com.json";
    char* rank_ID = "0";

    ret1 = HcomInitByFile(rank_table_file, rank_ID);
    EXPECT_EQ(ret1, HCCL_SUCCESS);
    string groupName = "HCOM_GROUP";
    u32 rankNum1 = 1;
    u32 rankIds = 0;
    ret1 = HcomCreateGroup(groupName.c_str(), rankNum1, &rankIds);
    EXPECT_EQ(ret1, HCCL_SUCCESS);
    rtStream_t stream;
    ge::Status ret;
    HcomOpsKernelInfoStore hcomKernelInfo;

    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    int64_t hcomComm = 0;
    std::shared_ptr<hccl::hcclComm> comm;
    ret1 = HcomGetCommByGroup(groupName.c_str(), comm);
    EXPECT_EQ(ret1, HCCL_SUCCESS);
    hcomComm = reinterpret_cast<int64_t>(comm.get());

    ge::GETaskInfo task;
    ge::GETaskKernelHcclInfo hcclInfo;
    task.kernelHcclInfo.push_back(hcclInfo);
    HCCL_KERNEL_INFO_PRIVATE_DEF privateDefBuf;
    strcpy_s((char*)privateDefBuf.group, 128, groupName.c_str());
    privateDefBuf.nodeNameHash = 123456;
    privateDefBuf.graphId = 1;
    privateDefBuf.srcRank = 1;
    privateDefBuf.destRank = 2;
    privateDefBuf.srTag = 3;
    privateDefBuf.comm = hcomComm;
    privateDefBuf.originalGraphShapeType = 0;
    privateDefBuf.dataType = HCCL_DATA_TYPE_FP32;
    privateDefBuf.tensorNum = 3;
    privateDefBuf.privateDefSize = sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF);
    privateDefBuf.needMapRank = true;
    privateDefBuf.isOfflineComp = true;
    privateDefBuf.devType = DevType::DEV_TYPE_910;
    privateDefBuf.aivCoreLimit = 48;
    task.type = RT_MODEL_TASK_HCCL;
    task.stream = stream;

    std::int64_t offset[privateDefBuf.tensorNum] = {33280, 33792, 34304};
    std::int64_t size[privateDefBuf.tensorNum] = {400, 40, 80};
    size_t tensorInfoSize = sizeof(hcclKernelInfoPrivateDef) + privateDefBuf.tensorNum * sizeof(int64_t) * 2;
    void* privateDefPtr = sal_malloc(tensorInfoSize);
    memset(privateDefPtr, 0, tensorInfoSize);
    memcpy(privateDefPtr, (void*)&privateDefBuf.group[0], sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF));
    memcpy(
        reinterpret_cast<int64_t*>(reinterpret_cast<char*>(privateDefPtr) + sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF)),
        offset, sizeof(offset));
    memcpy(
        reinterpret_cast<int64_t*>(
            reinterpret_cast<char*>(privateDefPtr) + sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF) + sizeof(offset)),
        size, sizeof(size));

    task.privateDef = privateDefPtr;
    task.privateDefLen = (uint32_t)tensorInfoSize;
    task.kernelHcclInfo[0].count = 100;
    task.kernelHcclInfo[0].dataType = HCCL_DATA_TYPE_FP32;
    task.kernelHcclInfo[0].hccl_type = HCCL_KERNEL_OP_TYPE_BROADCAST;
    task.kernelHcclInfo[0].inputDataAddr = sendbuf;
    task.kernelHcclInfo[0].outputDataAddr = recv;
    task.kernelHcclInfo[0].opType = HCCL_REDUCE_SUM;
    task.kernelHcclInfo[0].rootId = 0;
    task.kernelHcclInfo[0].hcclQosCfg = INVALID_QOSCFG;
    task.kernelHcclInfo[0].global_workspace_addr = globalWorkSpaceAddr;
    task.needRefresh = false;

    u64 memSize = HCCL_WORKSPACE_MEM_32_KB;
    DeviceMem workSpaceMem = DeviceMem::alloc(memSize);
    task.kernelHcclInfo[0].workSpaceAddr = workSpaceMem.ptr();
    task.kernelHcclInfo[0].workSpaceMemSize = workSpaceMem.size();

    u32 taskID = 0;

    // -------------------HcomBroadcast test----------------
    tensorInfoSize = sizeof(hcclKernelInfoPrivateDef);
    privateDefBuf.originalGraphShapeType = ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE;
    privateDefBuf.tensorNum = 0;
    void* privateDefPtr_broadcast = sal_malloc(tensorInfoSize);
    memset(privateDefPtr_broadcast, 0, tensorInfoSize);
    memcpy(privateDefPtr_broadcast, (void*)&privateDefBuf.group[0], sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF));

    task.privateDef = privateDefPtr_broadcast;
    task.privateDefLen = sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF);

    task.kernelHcclInfo[0].hccl_type = HCCL_KERNEL_OP_TYPE_BROADCAST;

    MOCKER(HcomBroadcast).expects(atMost(3)).will(returnValue(HCCL_E_PARA)).then(returnValue(HCCL_SUCCESS));

    MOCKER(HcomSetWorkspaceResource).expects(atMost(3)).will(returnValue(HCCL_SUCCESS));

    MOCKER(HcomGetInitStatus).stubs().with(outBound(true)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemAsyncCopy).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemcpyAddrAsync).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(InitGroup).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::OffloadStreamManager::GetSlaves)
        .stubs()
        .will(returnValue(vector<Stream>(1, Stream(StreamType::STREAM_TYPE_OFFLINE))));

    MOCKER_CPP(&TbeReduce::TbeVectorReduce::GetTilingDataDevMem)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TbeReduce::TbeCrackCleared::ExecuteKernelLaunch)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    // HcomBroadcast
    // load task success
    task.id = taskID++;
    ret = hcomKernelInfo.LoadTask(task);
    EXPECT_EQ(ret, ge::SUCCESS);
    GlobalMockObject::verify();

    sal_free(sendbuf);
    sal_free(recv);
    sal_free(dumpbuf);
    sal_free(privateDefPtr);
    sal_free(privateDefPtr_broadcast);
    rt_ret = aclrtDestroyStream(stream);
    HcomDestroy();
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    remove(file_name_t);
}

#endif

TEST_F(HcomKernelInfoTest, ut_SetUnkownWorkSpace)
{
    nlohmann::json rank_table = rank_table_910_1server_1rank;
    char file_name_t[] = "./ut_SetUnkownWorkSpace.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();
    u32 ret1 = hrtSetDevice(0);
    EXPECT_EQ(ret1, HCCL_SUCCESS);

    // 走1910 4pring
    char* rank_table_file = "./ut_SetUnkownWorkSpace.json";
    char* rank_ID = "0";

    ret1 = HcomInitByFile(rank_table_file, rank_ID);
    EXPECT_EQ(ret1, HCCL_SUCCESS);
    string groupName = "HCOM_GROUP";
    u32 rankNum1 = 1;
    u32 rankIds = 0;
    ret1 = HcomCreateGroup(groupName.c_str(), rankNum1, &rankIds);
    EXPECT_EQ(ret1, HCCL_SUCCESS);
    ge::GETaskInfo task;
    ge::GETaskKernelHcclInfo hcclInfo;
    task.kernelHcclInfo.push_back(hcclInfo);
    HCCL_KERNEL_INFO_PRIVATE_DEF privateDefBuf;
    strcpy_s((char*)privateDefBuf.group, 128, "HCOM_GROUP");
    privateDefBuf.nodeNameHash = 123456;
    privateDefBuf.graphId = 1;
    privateDefBuf.srcRank = 1;
    privateDefBuf.destRank = 2;
    privateDefBuf.srTag = 3;
    privateDefBuf.originalGraphShapeType = 1;
    privateDefBuf.aivCoreLimit = 48;
    task.type = RT_MODEL_TASK_HCCL;
    task.privateDef = (void*)&privateDefBuf.group[0];
    task.privateDefLen = sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF);
    task.kernelHcclInfo[0].hccl_type = HCCL_KERNEL_OP_TYPE_REDUCESCATTER;
    MOCKER_CPP(&hccl::WorkSpaceMem::SetMemResource).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hccl::OffloadStreamManager::RegisterSlaves).stubs().will(returnValue(HCCL_SUCCESS));
    HcomOpsKernelInfoStore hcomKernelInfo;
    string tag;
    std::vector<std::string> tagVec;
    tagVec.push_back(tag);
    HcclResult ret = hcomKernelInfo.SetUnknownShapeWorkspaceResource(task, HCCL_KERNEL_OP_TYPE_REDUCESCATTER, tagVec);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = hcomKernelInfo.SetUnknownShapeWorkspaceResource(task, HCCL_KERNEL_OP_TYPE_REDUCESCATTER, tagVec);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    privateDefBuf.comm = 1;
    task.privateDef = (void*)&privateDefBuf.group[0];
    MOCKER_CPP(&HcomOpsKernelInfoStore::CommGraphSetWorkspaceResourceFromtagVec)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hcclComm::SupportDeterministicOptim).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = hcomKernelInfo.SetUnknownShapeWorkspaceResource(task, HCCL_KERNEL_OP_TYPE_REDUCESCATTER, tagVec);

    task.kernelHcclInfo[0].hccl_type = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    privateDefBuf.comm = 2;
    task.privateDef = (void*)&privateDefBuf.group[0];
    ret = hcomKernelInfo.SetUnknownShapeWorkspaceResource(task, HCCL_KERNEL_OP_TYPE_ALLREDUCE, tagVec);

    HcomDestroy();
    remove(file_name_t);
    GlobalMockObject::verify();
}

TEST_F(HcomKernelInfoTest, ut_SetUnkownWorkSpace310p)
{
    nlohmann::json rank_table = rank_table_910_1server_1rank;
    char file_name_t[] = "./ut_SetUnkownWorkSpace310p.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();
    u32 ret1 = hrtSetDevice(0);
    EXPECT_EQ(ret1, HCCL_SUCCESS);
    DevType deviceType = DevType::DEV_TYPE_310P3;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));
    // 走1910 4pring
    char* rank_table_file = "./ut_SetUnkownWorkSpace310p.json";
    char* rank_ID = "0";

    ret1 = HcomInitByFile(rank_table_file, rank_ID);
    EXPECT_EQ(ret1, HCCL_SUCCESS);
    string groupName = "HCOM_GROUP";
    u32 rankNum1 = 1;
    u32 rankIds = 0;
    ret1 = HcomCreateGroup(groupName.c_str(), rankNum1, &rankIds);
    EXPECT_EQ(ret1, HCCL_SUCCESS);
    ge::GETaskInfo task;
    ge::GETaskKernelHcclInfo hcclInfo;
    task.kernelHcclInfo.push_back(hcclInfo);
    HCCL_KERNEL_INFO_PRIVATE_DEF privateDefBuf;
    strcpy_s((char*)privateDefBuf.group, 128, "HCOM_GROUP");
    privateDefBuf.nodeNameHash = 123456;
    privateDefBuf.graphId = 1;
    privateDefBuf.srcRank = 1;
    privateDefBuf.destRank = 2;
    privateDefBuf.srTag = 3;
    privateDefBuf.originalGraphShapeType = 1;
    privateDefBuf.aivCoreLimit = 48;
    task.type = RT_MODEL_TASK_HCCL;
    task.privateDef = (void*)&privateDefBuf.group[0];
    task.privateDefLen = sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF);
    task.kernelHcclInfo[0].hccl_type = HCCL_KERNEL_OP_TYPE_REDUCESCATTER;
    MOCKER_CPP(&hccl::WorkSpaceMem::SetMemResource).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hccl::WorkspaceResourceImpl::SetStreamResource).stubs().will(returnValue(HCCL_SUCCESS));
    HcomOpsKernelInfoStore hcomKernelInfo;
    string tag;
    std::vector<std::string> tagVec;
    tagVec.push_back(tag);
    HcclResult ret = hcomKernelInfo.SetUnknownShapeWorkspaceResource(task, HCCL_KERNEL_OP_TYPE_REDUCESCATTER, tagVec);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = hcomKernelInfo.SetUnknownShapeWorkspaceResource(task, HCCL_KERNEL_OP_TYPE_REDUCESCATTER, tagVec);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcomDestroy();
    remove(file_name_t);
}

TEST_F(HcomKernelInfoTest, ut_Initialize_Finalize)
{
    ge ::Status ret;
    std::map<std::string, std::string> options;

    // 未设置 rank table：失败
    ret = Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);
    MOCKER(HcomDestroy).expects(atMost(15)).will(returnValue(HCCL_SUCCESS));
    ret = Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);
    options.insert(pair<string, string>(ge::OPTION_EXEC_RANK_TABLE_FILE, "rank_table.json"));
    ret = Initialize(options);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    ret = Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);
    // 实验室场景 未设置 rank id ：失败
    options.insert(pair<string, string>(ge::OPTION_EXEC_DEPLOY_MODE, "0"));
    ret = Initialize(options);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    ret = Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);
    // 实验室场景 设置 rank id
    options.insert(pair<string, string>(ge::OPTION_EXEC_RANK_ID, "1"));
    options.insert(pair<string, string>(ge::OPTION_EXEC_PROFILING_MODE, "0"));

    MOCKER(HcomInitByFile).expects(atMost(20)).will(returnValue(HCCL_E_PARA)).then(returnValue(HCCL_SUCCESS));
    // 实验室场景 hcom_init失败：失败
    ret = Initialize(options);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    ret = Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);
    ret = Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);
    ret = Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);
    options[ge::OPTION_EXEC_PROFILING_MODE] = "1";
    options.insert(pair<string, string>(ge::OPTION_EXEC_PROFILING_OPTIONS, "training_trace"));
    ret = Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);
    ret = Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);
    options[ge::OPTION_EXEC_PROFILING_OPTIONS] = "training_trace:task_trace";
    ret = Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);
    ret = Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    // 实验室场景 hcom_init成功：成功
    ret = Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);
    ret = Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);
    // 实验室场景 再次初始化：成功
    ret = Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);

    ge ::Status ge_ret;
    bool bRet;
    HcomOpsKernelInfoStore hcomKernelInfo;

    std::map<string, OpsKernelInfoStorePtr> opKernInfos;
    std::map<string, GraphOptimizerPtr> graphOptimizers;
    GetOpsKernelInfoStores(opKernInfos);
    GetGraphOptimizerObjs(graphOptimizers);

    ge_ret = opKernInfos.at(HCCL_OPS_LIB_NAME)->Initialize(options);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    ge::NodePtr nodeptr(new NodeTest);
    // ge::OpDesc op;
    std::string reason;
    std::string type;
    // op type invalid
    type = "";
    nodeptr->GetOpDesc()->SetType(type);
    bRet = opKernInfos.at(HCCL_OPS_LIB_NAME)->CheckSupported(nodeptr->GetOpDesc(), reason);
    EXPECT_EQ(bRet, false);
    // op type broadcast success
    type = HCCL_KERNEL_OP_TYPE_BROADCAST;
    nodeptr->GetOpDesc()->SetType(type);
    bRet = opKernInfos.at(HCCL_OPS_LIB_NAME)->CheckSupported(nodeptr->GetOpDesc(), reason);
    EXPECT_EQ(bRet, true);

    std::map<string, ge::OpInfo> infos;
    opKernInfos.at(HCCL_OPS_LIB_NAME)->GetAllOpsKernelInfo(infos);

    ge_ret = opKernInfos.at(HCCL_OPS_LIB_NAME)->Finalize();
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    ret = Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    options.erase(ge::OPTION_EXEC_RANK_ID);
    options.erase(ge::OPTION_EXEC_DEPLOY_MODE);
    // 上云场景 未设置 pod name：失败
    options.insert(pair<string, string>(ge::OPTION_EXEC_DEPLOY_MODE, "1"));
    ret = Initialize(options);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    ret = Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);
    // 上云场景 设置 pod name： 成功
    options.insert(pair<string, string>(ge::OPTION_EXEC_POD_NAME, "pod_name_0001"));
    ret = Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);
    ret = Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    // 默认场景 rankid 和 podname 都设置：成功
    options.erase(ge::OPTION_EXEC_POD_NAME);
    options.erase(ge::OPTION_EXEC_RANK_ID);
    options.insert(pair<string, string>(ge::OPTION_EXEC_RANK_ID, "1"));
    options.insert(pair<string, string>(ge::OPTION_EXEC_POD_NAME, "1"));
    ret = Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);
    ret = Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    // 只设置pod name：成功
    options.erase(ge::OPTION_EXEC_POD_NAME);
    options.erase(ge::OPTION_EXEC_RANK_ID);
    options.insert(pair<string, string>(ge::OPTION_EXEC_POD_NAME, "1"));
    ret = Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);
    ret = Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    // 只设置rankid: 成功
    options.erase(ge::OPTION_EXEC_POD_NAME);
    options.erase(ge::OPTION_EXEC_RANK_ID);
    options.insert(pair<string, string>(ge::OPTION_EXEC_RANK_ID, "1"));
    ret = Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);
    ret = Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);

    // pod name 和 rank id 都 未设置 ：失败
    options.erase(ge::OPTION_EXEC_POD_NAME);
    options.erase(ge::OPTION_EXEC_RANK_ID);
    ret = Initialize(options);
    EXPECT_EQ(ret, ge::INTERNAL_ERROR);
    ret = Finalize();
    EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(HcomKernelInfoTest, ut_GetHcomOpMemSize)
{
    HcclResult ret;
    u32 shapeType = ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE;
    std::string sCollectiveType = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    const std::string sGroup = HCCL_WORLD_GROUP;
    int64_t hcomComm = 0;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    u64 count = 10;
    u64 inputAddrSize = 0;
    u64 outputAddrSize = 0;
    HcomOpsKernelInfoStore hcomOpsKernelInfoStore_;
    u32 rankSize = 8;

    MOCKER(HcomGetRankSize).stubs().with(mockcpp::any(), outBound(&rankSize)).will(returnValue(HCCL_SUCCESS));
    ret = hcomOpsKernelInfoStore_.GetHcomOpMemSize(
        shapeType, sCollectiveType, hcomComm, sGroup, dataType, count, inputAddrSize, outputAddrSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    sCollectiveType = HCCL_KERNEL_OP_TYPE_ALLGATHER;
    ret = hcomOpsKernelInfoStore_.GetHcomOpMemSize(
        shapeType, sCollectiveType, hcomComm, sGroup, dataType, count, inputAddrSize, outputAddrSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    sCollectiveType = HCCL_KERNEL_OP_TYPE_REDUCESCATTER;
    ret = hcomOpsKernelInfoStore_.GetHcomOpMemSize(
        shapeType, sCollectiveType, hcomComm, sGroup, dataType, count, inputAddrSize, outputAddrSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    sCollectiveType = HCCL_KERNEL_OP_TYPE_SEND;
    ret = hcomOpsKernelInfoStore_.GetHcomOpMemSize(
        shapeType, sCollectiveType, hcomComm, sGroup, dataType, count, inputAddrSize, outputAddrSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    sCollectiveType = HCCL_KERNEL_OP_TYPE_RECEIVE;
    ret = hcomOpsKernelInfoStore_.GetHcomOpMemSize(
        shapeType, sCollectiveType, hcomComm, sGroup, dataType, count, inputAddrSize, outputAddrSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hcomOpsKernelInfoStore_.GetHcomOpMemSize(shapeType, sCollectiveType, dataType, count, inputAddrSize);
    EXPECT_EQ(ret, HCCL_E_PARA);
    sCollectiveType = HCCL_KERNEL_OP_TYPE_BROADCAST;
    ret = hcomOpsKernelInfoStore_.GetHcomOpMemSize(shapeType, sCollectiveType, dataType, count, inputAddrSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcomKernelInfoTest, ut_GetCommCCLBuf)
{
    HcclResult ret;
    u32 shapeType = ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE;
    const std::string sGroup = HCCL_WORLD_GROUP;
    int64_t hcomComm = 0;
    void* commInputPtr;
    void* commOutputPtr;
    std::shared_ptr<hccl::hcclComm> comm;
    comm.reset(new (std::nothrow) hccl::hcclComm());
    HcomOpsKernelInfoStore hcomOpsKernelInfoStore_;

    MOCKER(HcomGetCommByGroup).stubs().with(mockcpp::any(), outBound(comm)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hccl::hcclComm::GetInCCLbuffer).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hccl::hcclComm::GetOutCCLbuffer).stubs().will(returnValue(HCCL_SUCCESS));
    ret = hcomOpsKernelInfoStore_.GetCommCCLBuf(shapeType, hcomComm, sGroup, commInputPtr, commOutputPtr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = hcomOpsKernelInfoStore_.GetCommCCLBuf(shapeType, "HcomBroadcast", hcomComm, sGroup, commInputPtr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcomKernelInfoTest, ut_RefreshInputAddr)
{
    HcclResult ret;
    u32 shapeType = ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE;
    const std::string sGroup = HCCL_WORLD_GROUP;
    int64_t hcomComm = 0;
    u32 addr = 0;
    void* inputAddr = &addr;
    u64 inputAddrSize = sizeof(u32);
    Stream stream(StreamType::STREAM_TYPE_OFFLINE);
    std::shared_ptr<hccl::hcclComm> comm;
    comm.reset(new (std::nothrow) hccl::hcclComm());
    HcomOpsKernelInfoStore hcomOpsKernelInfoStore_;

    MOCKER(HcomGetCommByGroup).stubs().with(mockcpp::any(), outBound(comm)).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::hcclComm::GetInCCLbuffer)
        .stubs()
        .with(outBound(inputAddr), outBound(inputAddrSize))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::hcclComm::GetIndirectInCCLbuf)
        .stubs()
        .with(outBound(inputAddr), outBound(inputAddrSize))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemAsyncCopy).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));

    ret = hcomOpsKernelInfoStore_.RefreshInputAddr(shapeType, hcomComm, sGroup, inputAddr, inputAddrSize, stream.ptr());
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcomKernelInfoTest, ut_RefreshInputAddr2)
{
    HcclResult ret;
    u32 shapeType = ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE;
    const std::string sGroup = HCCL_WORLD_GROUP;
    int64_t hcomComm = 1;
    u32 addr = 0;
    void* inputAddr = &addr;
    u64 inputAddrSize = sizeof(u32);
    Stream stream(StreamType::STREAM_TYPE_OFFLINE);
    std::shared_ptr<hccl::hcclComm> comm;
    comm.reset(new (std::nothrow) hccl::hcclComm());
    HcomOpsKernelInfoStore hcomOpsKernelInfoStore_;

    MOCKER(HcomGetCommByGroup).stubs().with(mockcpp::any(), outBound(comm)).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::hcclComm::GetInCCLbuffer)
        .stubs()
        .with(outBound(inputAddr), outBound(inputAddrSize))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::hcclComm::GetIndirectInCCLbuf)
        .stubs()
        .with(outBound(inputAddr), outBound(inputAddrSize))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemAsyncCopy).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemcpyAddrAsync).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));

    ret = hcomOpsKernelInfoStore_.RefreshInputAddr(
        DevType::DEV_TYPE_910, shapeType, hcomComm, sGroup, inputAddr, inputAddrSize, 4, false, stream.ptr());
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcomKernelInfoTest, ut_RefreshInputAddr3)
{
    HcclResult ret;
    u32 shapeType = ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE;
    const std::string sGroup = HCCL_WORLD_GROUP;
    int64_t hcomComm = 1;
    u32 addr = 0;
    void* inputAddr = &addr;
    u64 inputAddrSize = sizeof(u32);
    Stream stream(StreamType::STREAM_TYPE_OFFLINE);
    std::shared_ptr<hccl::hcclComm> comm;
    comm.reset(new (std::nothrow) hccl::hcclComm());
    HcomOpsKernelInfoStore hcomOpsKernelInfoStore_;

    MOCKER(HcomGetCommByGroup).stubs().with(mockcpp::any(), outBound(comm)).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::hcclComm::GetInCCLbuffer)
        .stubs()
        .with(outBound(inputAddr), outBound(inputAddrSize))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::hcclComm::GetIndirectInCCLbuf)
        .stubs()
        .with(outBound(inputAddr), outBound(inputAddrSize))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemAsyncCopy).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemcpyAddrAsync).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));

    ret = hcomOpsKernelInfoStore_.RefreshReduceScatterInputAddr(
        DevType::DEV_TYPE_910, shapeType, hcomComm, sGroup, inputAddr, inputAddrSize, 4, 1, 1, 1, true, stream.ptr());
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcomKernelInfoTest, ut_RefreshOutputAddr)
{
    HcclResult ret;
    u32 shapeType = ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE;
    const std::string sGroup = HCCL_WORLD_GROUP;
    int64_t hcomComm = 0;
    u32 addr = 0;
    void* outputAddr = &addr;
    u64 OutputAddrSize = sizeof(u32);
    Stream stream(StreamType::STREAM_TYPE_OFFLINE);
    std::shared_ptr<hccl::hcclComm> comm;
    comm.reset(new (std::nothrow) hccl::hcclComm());
    HcomOpsKernelInfoStore hcomOpsKernelInfoStore_;

    MOCKER(HcomGetCommByGroup).stubs().with(mockcpp::any(), outBound(comm)).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::hcclComm::GetOutCCLbuffer)
        .stubs()
        .with(outBound(outputAddr), outBound(OutputAddrSize))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::hcclComm::GetInCCLbuffer)
        .stubs()
        .with(outBound(outputAddr), outBound(OutputAddrSize))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::hcclComm::GetIndirectOutCCLbuf)
        .stubs()
        .with(outBound(outputAddr), outBound(OutputAddrSize))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::hcclComm::GetIndirectInCCLbuf)
        .stubs()
        .with(outBound(outputAddr), outBound(OutputAddrSize))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemAsyncCopy).expects(atMost(2)).will(returnValue(HCCL_SUCCESS));

    ret = hcomOpsKernelInfoStore_.RefreshOutputAddr(
        shapeType, hcomComm, sGroup, outputAddr, OutputAddrSize, stream.ptr());
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomKernelInfoTest, ut_RefreshOutputAddr2)
{
    HcclResult ret;
    u32 shapeType = ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE;
    const std::string sGroup = HCCL_WORLD_GROUP;
    int64_t hcomComm = 1;
    u32 addr = 0;
    void* outputAddr = &addr;
    u64 OutputAddrSize = sizeof(u32);
    Stream stream(StreamType::STREAM_TYPE_OFFLINE);
    std::shared_ptr<hccl::hcclComm> comm;
    comm.reset(new (std::nothrow) hccl::hcclComm());
    HcomOpsKernelInfoStore hcomOpsKernelInfoStore_;

    MOCKER(HcomGetCommByGroup).stubs().with(mockcpp::any(), outBound(comm)).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::hcclComm::GetOutCCLbuffer)
        .stubs()
        .with(outBound(outputAddr), outBound(OutputAddrSize))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::hcclComm::GetInCCLbuffer)
        .stubs()
        .with(outBound(outputAddr), outBound(OutputAddrSize))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::hcclComm::GetIndirectOutCCLbuf)
        .stubs()
        .with(outBound(outputAddr), outBound(OutputAddrSize))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::hcclComm::GetIndirectInCCLbuf)
        .stubs()
        .with(outBound(outputAddr), outBound(OutputAddrSize))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemAsyncCopy).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemcpyAddrAsync).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));

    ret = hcomOpsKernelInfoStore_.RefreshOutputAddr(
        DevType::DEV_TYPE_910, shapeType, hcomComm, sGroup, outputAddr, OutputAddrSize, 4, 1024, false, stream.ptr());
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomKernelInfoTest, ut_GetHcomOutCCLbufferSize)
{
    HcclResult ret;
    u32 shapeType = ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE;
    const std::string sGroup = HCCL_WORLD_GROUP;
    const int64_t hcomComm = 1;
    u32 addr = 0;
    void* outputAddr = &addr;
    u64 OutputAddrSize = sizeof(u32);
    Stream stream(StreamType::STREAM_TYPE_OFFLINE);
    std::shared_ptr<hccl::hcclComm> comm;
    comm.reset(new (std::nothrow) hccl::hcclComm());
    HcomOpsKernelInfoStore hcomOpsKernelInfoStore_;
    u64 commOutputSize = 0;

    MOCKER(HcomGetCommByGroup).stubs().with(mockcpp::any(), outBound(comm)).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::hcclComm::GetOutCCLbuffer)
        .stubs()
        .with(outBound(outputAddr), outBound(OutputAddrSize))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::hcclComm::GetInCCLbuffer)
        .stubs()
        .with(outBound(outputAddr), outBound(OutputAddrSize))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::hcclComm::GetIndirectOutCCLbuf)
        .stubs()
        .with(outBound(outputAddr), outBound(OutputAddrSize))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::hcclComm::GetIndirectInCCLbuf)
        .stubs()
        .with(outBound(outputAddr), outBound(OutputAddrSize))
        .will(returnValue(HCCL_SUCCESS));

    ret = hcomOpsKernelInfoStore_.GetHcomOutCCLbufferSize(commOutputSize, shapeType, hcomComm, sGroup);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hcomOpsKernelInfoStore_.GetHcomInCCLbufferSize(commOutputSize, shapeType, hcomComm, sGroup);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomKernelInfoTest, ut_RefreshOutputAddr3)
{
    HcclResult ret;
    u32 shapeType = ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE;
    const std::string sGroup = HCCL_WORLD_GROUP;
    int64_t hcomComm = 1;
    u32 addr = 0;
    void* outputAddr = &addr;
    u64 OutputAddrSize = sizeof(u32);
    Stream stream(StreamType::STREAM_TYPE_OFFLINE);
    std::shared_ptr<hccl::hcclComm> comm;
    comm.reset(new (std::nothrow) hccl::hcclComm());
    HcomOpsKernelInfoStore hcomOpsKernelInfoStore_;

    MOCKER(HcomGetCommByGroup).stubs().with(mockcpp::any(), outBound(comm)).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::hcclComm::GetOutCCLbuffer)
        .stubs()
        .with(outBound(outputAddr), outBound(OutputAddrSize))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::hcclComm::GetInCCLbuffer)
        .stubs()
        .with(outBound(outputAddr), outBound(OutputAddrSize))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::hcclComm::GetIndirectOutCCLbuf)
        .stubs()
        .with(outBound(outputAddr), outBound(OutputAddrSize))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::hcclComm::GetIndirectInCCLbuf)
        .stubs()
        .with(outBound(outputAddr), outBound(OutputAddrSize))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemAsyncCopy).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemcpyAddrAsync).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));

    ret = hcomOpsKernelInfoStore_.RefreshAllgatherOutputAddr(
        DevType::DEV_TYPE_910, shapeType, hcomComm, sGroup, outputAddr, OutputAddrSize, 4, 1, 1, 1, true, stream.ptr());
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

#define CHK_BBIT_RET(call) \
    do {                   \
        s32 ret = call;    \
        EXPECT_EQ(ret, 0); \
    } while (0)

#define CHK_UT_PTR_NULL(ptr)     \
    do {                         \
        EXPECT_NE(ptr, nullptr); \
    } while (0)

TEST_F(HcomKernelInfoTest, ut_hcom_alltoallv_loadtask)
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
            {{
                {"group_name", ""},
                {"device_num", "1"},
                {"server_num", "1"},
                {"instance_count", "1"},
                {"instance_list",
                 {{{"rank_id", "0"},
                   {"server_id", "172.17.1.120"},
                   {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.1.120"}}}}}}},
            }}}};

    char file_name_t[] = "./ut_hcom_alltoallv_loadtask.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    char* identify = "0";
    CHK_BBIT_RET(HcomInitByFile(file_name_t, identify));
    u32 rankSize = 0;
    CHK_BBIT_RET(HcomGetRankSize(HCCL_WORLD_GROUP, &rankSize));
    u32 rankID = 0;
    CHK_BBIT_RET(HcomGetRankId(HCCL_WORLD_GROUP, &rankID));

    u64 count = 10;
    u64 memSize = rankSize * count * sizeof(int32_t);
    HostMem inputHostMem = HostMem::alloc(memSize);
    CHK_UT_PTR_NULL(inputHostMem.ptr());
    HostMem comparaMem = HostMem::alloc(memSize);
    for (u32 i = 0; i < count * rankSize; i++) {
        *(reinterpret_cast<int32_t*>(inputHostMem.ptr()) + i) = i / count;
        *(reinterpret_cast<int32_t*>(comparaMem.ptr()) + i) = rankID;
    }
    DeviceMem inputDevMem = DeviceMem::alloc(memSize);
    CHK_UT_PTR_NULL(inputDevMem.ptr());
    CHK_BBIT_RET(hrtMemSyncCopy(
        inputDevMem.ptr(), memSize, inputHostMem.ptr(), memSize, HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));

    DeviceMem outputDevMem = DeviceMem::alloc(memSize);
    CHK_UT_PTR_NULL(outputDevMem.ptr());

    vector<u64> sendRecvCounts(rankSize, count);
    vector<u64> sendRecvDispls(rankSize, 0);
    for (u32 index = 0; index < sendRecvDispls.size(); index++) {
        sendRecvDispls[index] = index * count;
    }

    Stream stream(StreamType::STREAM_TYPE_OFFLINE);
    CHK_UT_PTR_NULL(stream.ptr());

    HCCL_KERNEL_INFO_PRIVATE_DEF privateDefBuf;
    strcpy_s((char*)privateDefBuf.group, 128, "hccl_world_group");
    privateDefBuf.nodeNameHash = 1234567890;
    privateDefBuf.graphId = 1;
    privateDefBuf.destRank = 1;
    privateDefBuf.originalGraphShapeType = 0;
    privateDefBuf.originalGraphShapeType = ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE;
    privateDefBuf.aivCoreLimit = 48;

    HCCL_ALLTOALLV_KERNEL_INFO_PRIVATE_DEF alltoallvPrivateDefBuf(privateDefBuf);
    alltoallvPrivateDefBuf.paramsInfo.sendCounts[0] = count;
    alltoallvPrivateDefBuf.paramsInfo.recvCounts[0] = count;
    alltoallvPrivateDefBuf.paramsInfo.sendType = HCCL_DATA_TYPE_FP32;
    alltoallvPrivateDefBuf.paramsInfo.recvType = HCCL_DATA_TYPE_FP32;

    u64 wsMemSize = HCCL_WORKSPACE_MEM_32_KB;
    DeviceMem workSpaceMem = DeviceMem::alloc(wsMemSize);

    ge::GETaskKernelHcclInfo hcclInfo;
    hcclInfo.dataType = HCCL_DATA_TYPE_FP32;
    hcclInfo.hccl_type = HCCL_KERNEL_OP_TYPE_ALLTOALLV;
    hcclInfo.inputDataAddr = inputDevMem.ptr();
    hcclInfo.outputDataAddr = outputDevMem.ptr();
    hcclInfo.workSpaceAddr = workSpaceMem.ptr();
    hcclInfo.workSpaceMemSize = workSpaceMem.size();
    ge::GETaskInfo task;
    task.kernelHcclInfo.push_back(hcclInfo);
    task.stream = stream.ptr();
    task.streamID = 1;
    task.type = RT_MODEL_TASK_HCCL;
    task.privateDef = reinterpret_cast<void*>(&alltoallvPrivateDefBuf);
    task.privateDefLen = sizeof(HCCL_ALLTOALLV_KERNEL_INFO_PRIVATE_DEF);
    HcomOpsKernelInfoStore infoStore;

    MOCKER(HcomGetInitStatus).stubs().with(outBound(true)).will(returnValue(HCCL_SUCCESS));

    MOCKER(InitGroup).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemAsyncCopy).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(HcclMemcpyAsync).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(HcclD2DMemcpyAsync).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    CHK_BBIT_RET(infoStore.LoadTask(task));
    // CHK_BBIT_RET(HcomAlltoAllV(inputDevMem.ptr(), sendRecvCounts.data(), sendRecvDispls.data(), HCCL_DATA_TYPE_INT32,
    //     outputDevMem.ptr(), sendRecvCounts.data(), sendRecvDispls.data(), HCCL_DATA_TYPE_INT32, nullptr,
    //     stream.ptr(), "bbit_alltoallv"));
    HCCL_INFO("TEST000: hcom alltoallv issue task");

    CHK_BBIT_RET(hcclStreamSynchronize(stream.ptr()));
    HCCL_INFO("TEST000: execute hcom alltoallv finished");

    HostMem resultHostMem = HostMem::alloc(memSize);
    CHK_UT_PTR_NULL(resultHostMem.ptr());
    CHK_BBIT_RET(hrtMemSyncCopy(
        resultHostMem.ptr(), memSize, outputDevMem.ptr(), memSize,
        HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST));
    CHK_BBIT_RET(HcomDestroy());
    remove(file_name_t);
    GlobalMockObject::verify();
}

TEST_F(HcomKernelInfoTest, ut_hcom_alltoall_loadtask)
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
            {{
                {"group_name", ""},
                {"device_num", "1"},
                {"server_num", "1"},
                {"instance_count", "1"},
                {"instance_list",
                 {{{"rank_id", "0"},
                   {"server_id", "172.17.1.120"},
                   {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.1.120"}}}}}}},
            }}}};

    char file_name_t[] = "./ut_hcom_alltoall_loadtask.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    char* identify = "0";
    CHK_BBIT_RET(HcomInitByFile(file_name_t, identify));
    u32 rankSize = 0;
    CHK_BBIT_RET(HcomGetRankSize(HCCL_WORLD_GROUP, &rankSize));
    u32 rankID = 0;
    CHK_BBIT_RET(HcomGetRankId(HCCL_WORLD_GROUP, &rankID));

    u64 count = 10;
    u64 memSize = rankSize * count * sizeof(float);
    DeviceMem inputDevMem = DeviceMem::alloc(memSize);
    CHK_UT_PTR_NULL(inputDevMem.ptr());

    DeviceMem outputDevMem = DeviceMem::alloc(memSize);
    CHK_UT_PTR_NULL(outputDevMem.ptr());

    Stream stream(StreamType::STREAM_TYPE_OFFLINE);
    CHK_UT_PTR_NULL(stream.ptr());

    HCCL_KERNEL_INFO_PRIVATE_DEF privateDefBuf;
    strcpy_s((char*)privateDefBuf.group, 128, "hccl_world_group");
    privateDefBuf.nodeNameHash = 1234567890;
    privateDefBuf.graphId = 1;
    privateDefBuf.destRank = 1;
    privateDefBuf.originalGraphShapeType = 0;
    privateDefBuf.dataType = HCCL_DATA_TYPE_FP32;
    privateDefBuf.aivCoreLimit = 48;

    HCCL_ALLTOALLV_KERNEL_INFO_PRIVATE_DEF alltoallvPrivateDefBuf(privateDefBuf);
    alltoallvPrivateDefBuf.paramsInfo.sendType = HCCL_DATA_TYPE_FP32;
    alltoallvPrivateDefBuf.paramsInfo.recvType = HCCL_DATA_TYPE_FP32;

    u64 wsMemSize = HCCL_WORKSPACE_MEM_32_KB;
    DeviceMem workSpaceMem = DeviceMem::alloc(wsMemSize);

    ge::GETaskKernelHcclInfo hcclInfo;
    hcclInfo.dataType = HCCL_DATA_TYPE_FP32;
    hcclInfo.hccl_type = HCCL_KERNEL_OP_TYPE_ALLTOALL;
    hcclInfo.count = count;
    hcclInfo.inputDataAddr = inputDevMem.ptr();
    hcclInfo.outputDataAddr = outputDevMem.ptr();
    hcclInfo.workSpaceAddr = workSpaceMem.ptr();
    hcclInfo.workSpaceMemSize = workSpaceMem.size();
    ge::GETaskInfo task;
    task.kernelHcclInfo.push_back(hcclInfo);
    task.stream = stream.ptr();
    task.streamID = 1;
    task.type = RT_MODEL_TASK_HCCL;
    task.privateDef = reinterpret_cast<void*>(&alltoallvPrivateDefBuf);
    task.privateDefLen = sizeof(HCCL_ALLTOALLV_KERNEL_INFO_PRIVATE_DEF);
    HcomOpsKernelInfoStore infoStore;

    MOCKER(HcomGetInitStatus).stubs().with(outBound(true)).will(returnValue(HCCL_SUCCESS));

    MOCKER(InitGroup).stubs().will(returnValue(HCCL_SUCCESS));

    CHK_BBIT_RET(infoStore.LoadTask(task));
    HCCL_INFO("TEST000: hcom alltoallv issue task");

    CHK_BBIT_RET(hcclStreamSynchronize(stream.ptr()));
    HCCL_INFO("TEST000: execute hcom alltoallv finished");

    CHK_BBIT_RET(HcomDestroy());
    remove(file_name_t);
    GlobalMockObject::verify();
}

ge::graphStatus
FakeGetOption(ge::GEThreadLocalContext* that, const std::string& optionExec, std::string& dumpDebugValue)
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
            {{
                {"group_name", ""},
                {"device_num", "1"},
                {"server_num", "1"},
                {"instance_count", "1"},
                {"instance_list",
                 {{{"rank_id", "0"},
                   {"server_id", "172.17.1.120"},
                   {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.1.120"}}}}}}},
            }}}};

    nlohmann::json group_list = {{{"group_name", "aaa"}, {"group_rank_list", {0, 1}}}};
    if (optionExec == ge::OPTION_EXEC_RANK_TABLE) {
        dumpDebugValue = rank_table.dump();
    } else if (optionExec == ge::OPTION_EXEC_HCOM_GROUPLIST) {
        dumpDebugValue = group_list.dump();
    } else if (optionExec == ge::OPTION_EXEC_RANK_ID) {
        dumpDebugValue.push_back('1');
    } else if (optionExec == ge::OPTION_EXEC_ROLE_TABLE_ADDR) {
        return !ge::GRAPH_SUCCESS;
    } else {
        dumpDebugValue.push_back('1');
    }
    return ge::GRAPH_SUCCESS;
}

TEST_F(HcomKernelInfoTest, ut_init_group)
{
    MOCKER_CPP(&ge::GEThreadLocalContext::GetOption).stubs().will(invoke(FakeGetOption));
    u32 rankId = 0;
    MOCKER(HcomGetRankId).stubs().with(mockcpp::any(), outBound(&rankId)).will(returnValue(HCCL_SUCCESS));
    MOCKER(HcomCreateGroup).stubs().will(returnValue(HCCL_SUCCESS));
    auto ret = InitGroup();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcomKernelInfoTest, ut_hcom_offline_build_init_hcom)
{
    u64 count = 10;
    u64 memSize = count * sizeof(float);
    DeviceMem inputDevMem = DeviceMem::alloc(memSize);
    CHK_UT_PTR_NULL(inputDevMem.ptr());

    DeviceMem outputDevMem = DeviceMem::alloc(memSize);
    CHK_UT_PTR_NULL(outputDevMem.ptr());

    Stream stream(StreamType::STREAM_TYPE_OFFLINE);
    CHK_UT_PTR_NULL(stream.ptr());

    HCCL_KERNEL_INFO_PRIVATE_DEF privateDefBuf;
    strcpy_s((char*)privateDefBuf.group, 128, "hccl_world_group");
    privateDefBuf.nodeNameHash = 1234567890;
    privateDefBuf.graphId = 1;
    privateDefBuf.destRank = 1;
    privateDefBuf.originalGraphShapeType = 0;
    privateDefBuf.dataType = HCCL_DATA_TYPE_FP32;
    privateDefBuf.aivCoreLimit = 48;

    HCCL_ALLTOALLV_KERNEL_INFO_PRIVATE_DEF alltoallvPrivateDefBuf(privateDefBuf);
    alltoallvPrivateDefBuf.paramsInfo.sendType = HCCL_DATA_TYPE_FP32;
    alltoallvPrivateDefBuf.paramsInfo.recvType = HCCL_DATA_TYPE_FP32;

    u64 wsMemSize = HCCL_WORKSPACE_MEM_32_KB;
    DeviceMem workSpaceMem = DeviceMem::alloc(wsMemSize);

    ge::GETaskKernelHcclInfo hcclInfo;
    hcclInfo.dataType = HCCL_DATA_TYPE_FP32;
    hcclInfo.hccl_type = HCCL_KERNEL_OP_TYPE_ALLTOALL;
    hcclInfo.count = count;
    hcclInfo.inputDataAddr = inputDevMem.ptr();
    hcclInfo.outputDataAddr = outputDevMem.ptr();
    hcclInfo.workSpaceAddr = workSpaceMem.ptr();
    hcclInfo.workSpaceMemSize = workSpaceMem.size();
    ge::GETaskInfo task;
    task.kernelHcclInfo.push_back(hcclInfo);
    task.stream = stream.ptr();
    task.streamID = 1;
    task.type = RT_MODEL_TASK_HCCL;
    task.privateDef = reinterpret_cast<void*>(&alltoallvPrivateDefBuf);
    task.privateDefLen = sizeof(HCCL_ALLTOALLV_KERNEL_INFO_PRIVATE_DEF);
    HcomOpsKernelInfoStore infoStore;

    MOCKER_CPP(&ge::GEThreadLocalContext::GetOption).stubs().will(invoke(FakeGetOption));

    string rankTable = "aaa";
    u64 rankTableAddr = reinterpret_cast<uintptr_t>(rankTable.data());
    u32 rankTableLen = rankTable.length();

    MOCKER(SalStrToULonglong)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(rankTableAddr))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER(SalStrToULonglong)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(rankTableLen))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER(HcomInitByString).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(HcomCreateGroup).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(HcomGetRankId).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcomOpsKernelInfoStore::CheckCommunicatorValidity).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(HcomSetWorkspaceResource).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemAsyncCopy).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(HcomGetandClearOverFlowTasks).stubs().will(returnValue(HCCL_SUCCESS));

    infoStore.LoadTask(task);
    HCCL_INFO("TEST000: hcom alltoallv issue task");

    CHK_BBIT_RET(hcclStreamSynchronize(stream.ptr()));
    HCCL_INFO("TEST000: execute hcom alltoallv finished");

    CHK_BBIT_RET(HcomDestroy());
    GlobalMockObject::verify();
}

#define HCCL_COM_DATA_SIZE 1024
TEST_F(HcomKernelInfoTest, ut_hcom_loadtask_checkTaskID)
{
    nlohmann::json rank_table = rank_table_910_1server_1rank;

    char file_name_t[] = "./ut_hcom_loadtask_checkTaskID.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    s8* sendbuf;
    s8* recvbuf;
    void* dumpbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    char* rank_table_file = "./ut_hcom_loadtask_checkTaskID.json";
    char* rank_ID = "0";

    sendbuf = (s8*)sal_malloc(count);
    sal_memset(sendbuf, count, 0, count);
    recvbuf = (s8*)sal_malloc(count);
    sal_memset(recvbuf, count, 0, count);

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    rtStream_t stream;
    // HcomOpsKernelInfoStore hcomKernelInfo;

    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    ge ::Status ge_ret;
    std::map<std::string, std::string> options;

    options.insert(pair<string, string>(ge::OPTION_EXEC_RANK_TABLE_FILE, "./ut_hcom_loadtask_checkTaskID.json"));
    options.insert(pair<string, string>(ge::OPTION_EXEC_RANK_ID, "0"));
    options.insert(pair<string, string>(ge::OPTION_EXEC_DEPLOY_MODE, "0"));
    options.insert(pair<string, string>(ge::OPTION_GRAPH_RUN_MODE, "1"));
    options.insert(pair<string, string>(ge::OPTION_EXEC_ENABLE_DUMP_DEBUG, "1"));
    options.insert(pair<string, string>(ge::OPTION_EXEC_PROFILING_MODE, "0"));
    ret = Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);

    std::map<string, OpsKernelInfoStorePtr> opKernInfos;
    std::map<string, GraphOptimizerPtr> graphOptimizers;
    GetOpsKernelInfoStores(opKernInfos);
    GetGraphOptimizerObjs(graphOptimizers);
    OpsKernelInfoStorePtr opsKernerInfoStorePtr = opKernInfos.at(HCCL_OPS_LIB_NAME);
    ge_ret = opsKernerInfoStorePtr->Initialize(options);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    ge::NodePtr nodeptr(new NodeTest);
    // ge::OpDesc op;
    std::string reason;
    std::string type = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    nodeptr->GetOpDesc()->SetType(type);
    bool bRet = opsKernerInfoStorePtr->CheckSupported(nodeptr->GetOpDesc(), reason);
    EXPECT_EQ(bRet, true);

    HCCL_INFO("test GetAllOpsKernelInfo.");
    std::map<string, ge::OpInfo> infos;
    opsKernerInfoStorePtr->GetAllOpsKernelInfo(infos);

    dumpbuf = sal_malloc(10 * sizeof(s8));
    sal_memset(dumpbuf, 10 * sizeof(s8), 0, 10 * sizeof(s8));
std:
    vector<void*> globalWorkSpaceAddr;
    globalWorkSpaceAddr.push_back(dumpbuf);

    ge::GETaskInfo task;
    ge::GETaskKernelHcclInfo hcclInfo;
    task.kernelHcclInfo.push_back(hcclInfo);
    HCCL_KERNEL_INFO_PRIVATE_DEF privateDefBuf;
    strcpy_s((char*)privateDefBuf.group, 128, "hccl_world_group");
    privateDefBuf.nodeNameHash = 123456;
    privateDefBuf.graphId = 1;
    privateDefBuf.dataType = HCCL_DATA_TYPE_INT8;
    privateDefBuf.tensorNum = 3;
    privateDefBuf.privateDefSize = sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF);
    privateDefBuf.aivCoreLimit = 48;
    task.id = 1;
    task.type = RT_MODEL_TASK_HCCL;
    task.stream = stream;

    std::int64_t offset[privateDefBuf.tensorNum] = {16};
    std::int64_t size[privateDefBuf.tensorNum] = {8};
    size_t tensorInfoSize = sizeof(hcclKernelInfoPrivateDef) + privateDefBuf.tensorNum * sizeof(int64_t) * 2;
    void* privateDefPtr = sal_malloc(tensorInfoSize);
    memset(privateDefPtr, 0, tensorInfoSize);
    memcpy(privateDefPtr, (void*)&privateDefBuf.group[0], sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF));
    memcpy(
        reinterpret_cast<int64_t*>(reinterpret_cast<char*>(privateDefPtr) + sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF)),
        offset, sizeof(offset));
    memcpy(
        reinterpret_cast<int64_t*>(
            reinterpret_cast<char*>(privateDefPtr) + sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF) + sizeof(offset)),
        size, sizeof(size));

    task.privateDef = privateDefPtr;
    task.privateDefLen = (uint32_t)tensorInfoSize;

    // task.privateDef = (void *)&privateDefBuf.group[0];
    // task.privateDefLen = sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF);
    task.kernelHcclInfo[0].count = count;
    task.kernelHcclInfo[0].dataType = HCCL_DATA_TYPE_INT8;
    task.kernelHcclInfo[0].hccl_type = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    task.kernelHcclInfo[0].inputDataAddr = sendbuf;
    task.kernelHcclInfo[0].outputDataAddr = recvbuf;
    task.kernelHcclInfo[0].opType = HCCL_REDUCE_SUM;
    task.kernelHcclInfo[0].rootId = 0;
    task.kernelHcclInfo[0].hcclQosCfg = INVALID_QOSCFG;
    task.kernelHcclInfo[0].global_workspace_addr = globalWorkSpaceAddr;
    task.needRefresh = false;

    u64 memSize = HCCL_WORKSPACE_MEM_32_KB;
    DeviceMem workSpaceMem = DeviceMem::alloc(memSize);
    task.kernelHcclInfo[0].workSpaceAddr = workSpaceMem.ptr();
    task.kernelHcclInfo[0].workSpaceMemSize = workSpaceMem.size();

    MOCKER(HcomGetInitStatus).stubs().with(outBound(true)).will(returnValue(HCCL_SUCCESS));

    MOCKER(InitGroup).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::OffloadStreamManager::GetSlaves).stubs().will(returnValue(vector<Stream>()));

    MOCKER_CPP(&TbeReduce::TbeVectorReduce::GetTilingDataDevMem)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TbeReduce::TbeCrackCleared::ExecuteKernelLaunch)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    ge_ret = opsKernerInfoStorePtr->LoadTask(task);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    ge_ret = opsKernerInfoStorePtr->LoadTask(task);
    EXPECT_EQ(ge_ret, ge::INTERNAL_ERROR);

    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sal_free(sendbuf);
    sal_free(recvbuf);
    sal_free(dumpbuf);
    sal_free(privateDefPtr);
    rt_ret = aclrtDestroyStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    ge_ret = opsKernerInfoStorePtr->Finalize();
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    ge_ret = Finalize();
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    remove(file_name_t);

    GlobalMockObject::verify();
}

TEST_F(HcomKernelInfoTest, ut_unloadtask)
{
    nlohmann::json rank_table = rank_table_910_1server_1rank;

    char file_name_t[] = "./ut_unloadtask.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    s8* sendbuf;
    s8* recvbuf;
    void* dumpbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    char* rank_table_file = "./ut_unloadtask.json";
    char* rank_ID = "0";

    sendbuf = (s8*)sal_malloc(count);
    sal_memset(sendbuf, count, 0, count);
    recvbuf = (s8*)sal_malloc(count);
    sal_memset(recvbuf, count, 0, count);

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    rtStream_t stream;
    // HcomOpsKernelInfoStore hcomKernelInfo;

    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    ge ::Status ge_ret;
    std::map<std::string, std::string> options;

    options.insert(pair<string, string>(ge::OPTION_EXEC_RANK_TABLE_FILE, "./ut_unloadtask.json"));
    options.insert(pair<string, string>(ge::OPTION_EXEC_RANK_ID, "0"));
    options.insert(pair<string, string>(ge::OPTION_EXEC_DEPLOY_MODE, "0"));
    options.insert(pair<string, string>(ge::OPTION_GRAPH_RUN_MODE, "1"));
    options.insert(pair<string, string>(ge::OPTION_EXEC_ENABLE_DUMP_DEBUG, "0"));
    options.insert(pair<string, string>(ge::OPTION_EXEC_PROFILING_MODE, "0"));
    ret = Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);

    std::map<string, OpsKernelInfoStorePtr> opKernInfos;
    std::map<string, GraphOptimizerPtr> graphOptimizers;
    GetOpsKernelInfoStores(opKernInfos);
    GetGraphOptimizerObjs(graphOptimizers);
    OpsKernelInfoStorePtr opsKernerInfoStorePtr = opKernInfos.at(HCCL_OPS_LIB_NAME);
    ge_ret = opsKernerInfoStorePtr->Initialize(options);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    ge::NodePtr nodeptr(new NodeTest);
    // ge::OpDesc op;
    std::string reason;
    std::string type = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    nodeptr->GetOpDesc()->SetType(type);
    bool bRet = opsKernerInfoStorePtr->CheckSupported(nodeptr->GetOpDesc(), reason);
    EXPECT_EQ(bRet, true);

    HCCL_INFO("test GetAllOpsKernelInfo.");
    std::map<string, ge::OpInfo> infos;
    opsKernerInfoStorePtr->GetAllOpsKernelInfo(infos);

    ge::GETaskInfo task;
    ge::GETaskKernelHcclInfo hcclInfo;
    task.kernelHcclInfo.push_back(hcclInfo);
    HCCL_KERNEL_INFO_PRIVATE_DEF privateDefBuf;
    strcpy_s((char*)privateDefBuf.group, 128, "hccl_world_group");
    privateDefBuf.nodeNameHash = 123456;
    privateDefBuf.graphId = 1;
    privateDefBuf.dataType = HCCL_DATA_TYPE_INT8;
    privateDefBuf.tensorNum = 3;
    privateDefBuf.privateDefSize = sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF);
    privateDefBuf.aivCoreLimit = 48;
    task.id = 1;
    task.type = RT_MODEL_TASK_HCCL;
    task.stream = stream;

    std::int64_t offset[privateDefBuf.tensorNum] = {16, 32, 64};
    std::int64_t size[privateDefBuf.tensorNum] = {8, 8, 9};
    size_t tensorInfoSize = sizeof(hcclKernelInfoPrivateDef) + privateDefBuf.tensorNum * sizeof(int64_t) * 2;
    void* privateDefPtr = sal_malloc(tensorInfoSize);
    memset(privateDefPtr, 0, tensorInfoSize);
    memcpy(privateDefPtr, (void*)&privateDefBuf.group[0], sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF));
    memcpy(
        reinterpret_cast<int64_t*>(reinterpret_cast<char*>(privateDefPtr) + sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF)),
        offset, sizeof(offset));
    memcpy(
        reinterpret_cast<int64_t*>(
            reinterpret_cast<char*>(privateDefPtr) + sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF) + sizeof(offset)),
        size, sizeof(size));

    task.privateDef = privateDefPtr;
    task.privateDefLen = (uint32_t)tensorInfoSize;

    // task.privateDef = (void *)&privateDefBuf.group[0];
    // task.privateDefLen = sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF);
    task.kernelHcclInfo[0].count = count;
    task.kernelHcclInfo[0].dataType = HCCL_DATA_TYPE_INT8;
    task.kernelHcclInfo[0].hccl_type = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    task.kernelHcclInfo[0].hccl_type = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    task.kernelHcclInfo[0].inputDataAddr = sendbuf;
    task.kernelHcclInfo[0].outputDataAddr = recvbuf;
    task.kernelHcclInfo[0].opType = HCCL_REDUCE_SUM;
    task.kernelHcclInfo[0].rootId = 0;
    task.kernelHcclInfo[0].hcclQosCfg = INVALID_QOSCFG;

    u64 memSize = HCCL_WORKSPACE_MEM_32_KB;
    DeviceMem workSpaceMem = DeviceMem::alloc(memSize);
    task.kernelHcclInfo[0].workSpaceAddr = workSpaceMem.ptr();
    task.kernelHcclInfo[0].workSpaceMemSize = workSpaceMem.size();
    ;

    MOCKER(HcomGetInitStatus).stubs().with(outBound(true)).will(returnValue(HCCL_SUCCESS));

    MOCKER(InitGroup).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::OffloadStreamManager::GetSlaves).stubs().will(returnValue(vector<Stream>()));

    MOCKER_CPP(&TbeReduce::TbeVectorReduce::GetTilingDataDevMem)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TbeReduce::TbeCrackCleared::ExecuteKernelLaunch)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    ge_ret = opsKernerInfoStorePtr->LoadTask(task);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    for (int j = 0; j < count; j++) {
        if (recvbuf[j] != 2) {
            errors++;
        }
    }

    sal_free(sendbuf);
    sal_free(recvbuf);
    rt_ret = aclrtDestroyStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    ge_ret = opsKernerInfoStorePtr->UnloadTask(task);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    ge_ret = opsKernerInfoStorePtr->Finalize();
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    ge_ret = Finalize();
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    sal_free(privateDefPtr);
    remove(file_name_t);

    HCCL_INFO("error counter:%d", errors);
    EXPECT_EQ(errors, 0);
    GlobalMockObject::verify();
}

TEST_F(HcomKernelInfoTest, ut_task_overflow)
{
    nlohmann::json rank_table = rank_table_910_1server_1rank;

    char file_name_t[] = "./ut_task_overflow.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    char* rank_table_file = "./ut_task_overflow.json";
    char* rank_ID = "0";

    sendbuf = (s8*)sal_malloc(count);
    sal_memset(sendbuf, count, 0, count);
    recvbuf = (s8*)sal_malloc(count);
    sal_memset(recvbuf, count, 0, count);

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    rtStream_t stream;
    // HcomOpsKernelInfoStore hcomKernelInfo;

    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    ge ::Status ge_ret;
    std::map<std::string, std::string> options;
    options.insert(pair<string, string>(ge::OPTION_EXEC_IS_USEHCOM, "1"));
    options.insert(pair<string, string>(ge::OPTION_EXEC_RANK_TABLE_FILE, "./ut_task_overflow.json"));
    options.insert(pair<string, string>(ge::OPTION_EXEC_RANK_ID, "0"));
    options.insert(pair<string, string>(ge::OPTION_EXEC_DEPLOY_MODE, "0"));
    options.insert(pair<string, string>(ge::OPTION_GRAPH_RUN_MODE, "1"));
    options.insert(pair<string, string>(ge::OPTION_EXEC_PROFILING_MODE, "0"));
    options.insert(pair<string, string>(ge::OPTION_EXEC_ENABLE_DUMP_DEBUG, "1"));
    ret = Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);

    std::map<string, OpsKernelInfoStorePtr> opKernInfos;
    std::map<string, GraphOptimizerPtr> graphOptimizers;
    GetOpsKernelInfoStores(opKernInfos);
    GetGraphOptimizerObjs(graphOptimizers);
    OpsKernelInfoStorePtr opsKernerInfoStorePtr = opKernInfos.at(HCCL_OPS_LIB_NAME);
    ge_ret = opsKernerInfoStorePtr->Initialize(options);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    ge::NodePtr nodeptr(new NodeTest);
    // ge::OpDesc op;
    std::string reason;
    std::string type = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    nodeptr->GetOpDesc()->SetType(type);
    bool bRet = opsKernerInfoStorePtr->CheckSupported(nodeptr->GetOpDesc(), reason);
    EXPECT_EQ(bRet, true);

    HCCL_INFO("test GetAllOpsKernelInfo.");
    std::map<string, ge::OpInfo> infos;
    opsKernerInfoStorePtr->GetAllOpsKernelInfo(infos);

    void* dumpbuf = sal_malloc(10 * sizeof(s8));
    sal_memset(dumpbuf, 10 * sizeof(s8), 0, 10 * sizeof(s8));
std:
    vector<void*> globalWorkSpaceAddr;
    globalWorkSpaceAddr.push_back(dumpbuf);

    ge::GETaskInfo task;
    ge::GETaskKernelHcclInfo hcclInfo;
    task.kernelHcclInfo.push_back(hcclInfo);
    HCCL_KERNEL_INFO_PRIVATE_DEF privateDefBuf;
    strcpy_s((char*)privateDefBuf.group, 128, "hccl_world_group");
    privateDefBuf.nodeNameHash = 123456;
    privateDefBuf.graphId = 1;
    privateDefBuf.dataType = HCCL_DATA_TYPE_INT8;
    privateDefBuf.tensorNum = 3;
    privateDefBuf.privateDefSize = sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF);
    privateDefBuf.aivCoreLimit = 48;
    task.id = 1;
    task.type = RT_MODEL_TASK_HCCL;
    task.stream = stream;

    std::int64_t offset[privateDefBuf.tensorNum] = {16, 32, 64};
    std::int64_t size[privateDefBuf.tensorNum] = {8, 8, 9};
    size_t tensorInfoSize = sizeof(hcclKernelInfoPrivateDef) + privateDefBuf.tensorNum * sizeof(int64_t) * 2;
    void* privateDefPtr = sal_malloc(tensorInfoSize);
    memset(privateDefPtr, 0, tensorInfoSize);
    memcpy(privateDefPtr, (void*)&privateDefBuf.group[0], sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF));
    memcpy(
        reinterpret_cast<int64_t*>(reinterpret_cast<char*>(privateDefPtr) + sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF)),
        offset, sizeof(offset));
    memcpy(
        reinterpret_cast<int64_t*>(
            reinterpret_cast<char*>(privateDefPtr) + sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF) + sizeof(offset)),
        size, sizeof(size));

    task.privateDef = privateDefPtr;
    task.privateDefLen = (uint32_t)tensorInfoSize;

    // task.privateDef = (void *)&privateDefBuf.group[0];
    // task.privateDefLen = sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF);
    task.kernelHcclInfo[0].count = count;
    task.kernelHcclInfo[0].dataType = HCCL_DATA_TYPE_INT8;
    task.kernelHcclInfo[0].hccl_type = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    task.kernelHcclInfo[0].inputDataAddr = sendbuf;
    task.kernelHcclInfo[0].outputDataAddr = recvbuf;
    task.kernelHcclInfo[0].opType = HCCL_REDUCE_SUM;
    task.kernelHcclInfo[0].rootId = 0;
    task.kernelHcclInfo[0].hcclQosCfg = INVALID_QOSCFG;
    task.kernelHcclInfo[0].global_workspace_addr = globalWorkSpaceAddr;
    task.needRefresh = false;

    u64 memSize = HCCL_WORKSPACE_MEM_32_KB;
    DeviceMem workSpaceMem = DeviceMem::alloc(memSize);
    task.kernelHcclInfo[0].workSpaceAddr = workSpaceMem.ptr();
    task.kernelHcclInfo[0].workSpaceMemSize = workSpaceMem.size();
    ;

    MOCKER(HcomGetInitStatus).stubs().with(outBound(true)).will(returnValue(HCCL_SUCCESS));

    MOCKER(InitGroup).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::OffloadStreamManager::GetSlaves).stubs().will(returnValue(vector<Stream>()));

    MOCKER_CPP(&TbeReduce::TbeVectorReduce::GetTilingDataDevMem)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TbeReduce::TbeCrackCleared::ExecuteKernelLaunch)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    ge_ret = opsKernerInfoStorePtr->LoadTask(task);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    for (int j = 0; j < count; j++) {
        if (recvbuf[j] != 2) {
            errors++;
        }
    }

    sal_free(sendbuf);
    sal_free(recvbuf);
    sal_free(dumpbuf);
    rt_ret = aclrtDestroyStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    ge_ret = opsKernerInfoStorePtr->UnloadTask(task);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    ge_ret = opsKernerInfoStorePtr->Finalize();
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    ge_ret = Finalize();
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    sal_free(privateDefPtr);
    remove(file_name_t);

    HCCL_INFO("error counter:%d", errors);
    EXPECT_EQ(errors, 0);
    GlobalMockObject::verify();
}

TEST_F(HcomKernelInfoTest, ut_SaveReduceDumpTask)
{
    MOCKER_CPP(&HcomOpsKernelInfoStore::GetCommFromTaskInfo).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));

    MOCKER(HcomGetInitStatus).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));

    HcomOpsKernelInfoStore infoStore;
    ge::GETaskInfo task;

    ge::GETaskKernelHcclInfo hcclInfo;
    task.kernelHcclInfo.push_back(hcclInfo);

    std::vector<hccl::HcclDumpInfo> dumpInfoVec;
    hccl::HcclDumpInfo dumpInfo;
    dumpInfo.task_id = 1;
    dumpInfo.stream_id = 1;
    dumpInfo.sub_task_type = 1;
    dumpInfo.output_addr = (void*)0x05;
    dumpInfo.output_size = 8;
    dumpInfo.input_addr = (void*)0x06;
    dumpInfo.input_size = 8;
    dumpInfoVec.push_back(dumpInfo);
    HcclResult ret = infoStore.SaveReduceDumpTask(task.kernelHcclInfo[0].hccl_dump_info, dumpInfoVec);
    GlobalMockObject::verify();
    EXPECT_EQ(ret, 0);
}

TEST_F(HcomKernelInfoTest, ut_HcclGetRemoteOperationParams)
{
    MOCKER(GetOpDescStrAttr).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(GetOpDescIntAttr)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS))
        .then(returnValue(HCCL_SUCCESS))
        .then(returnValue(HCCL_SUCCESS))
        .then(returnValue(HCCL_E_INTERNAL));

    HcomRemoteOperationParams opParams;
    ge::OpDesc* opdesc;
    opdesc->SetType("HcomCollRemoteLookupPaired");
    s32 ret = HcclGetRemoteOperationParams(*opdesc, &opParams);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    GlobalMockObject::verify();
}

TEST_F(HcomKernelInfoTest, ut_GetOpWorkspaceMemSize)
{
    ge::Node node;
    ge::AttrUtils::HasAttr(node.GetOpDesc(), "DUMMY_SET_TRUE_MAXNUM");
    ge::AttrUtils::HasAttr(node.GetOpDesc(), "DUMMY_SET_TRUE_EMBEDDINGDIM");
    ge::AttrUtils::HasAttr(node.GetOpDesc(), "DUMMY_SET_TRUE_FLAGS");
    ge::AttrUtils::HasAttr(node.GetOpDesc(), "DUMMY_SET_TRUE_TAG");
    ge::AttrUtils::HasAttr(node.GetOpDesc(), "DUMMY_SET_TRUE_MAXEMBEDDINGDIM");

    ge::AttrUtils::SetInt(node.GetOpDesc(), "flags", 1);

    u64 opMemSize;
    std::string sCollectiveType;
    sCollectiveType = HCCL_KERNEL_OP_TYPE_GATHER;
    HcomOpsKernelBuilder kernelbuilder;
    s32 ret = kernelbuilder.GetOpWorkspaceMemSize(node, sCollectiveType, opMemSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    sCollectiveType = HCCL_KERNEL_OP_TYPE_GATHER;

    ret = kernelbuilder.GetOpWorkspaceMemSize(node, sCollectiveType, opMemSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcomKernelInfoTest, ExecuteRemoteOperation)
{
    MOCKER(GetOpDescStrAttr).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(GetOpDescIntAttr).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    HcomRemoteOperationParams opParams;
    ge::OpDesc* opdesc;
    opdesc->SetType("HcomCollRemoteLookupPaired");
    s32 ret = HcclGetRemoteOperationParams(*opdesc, &opParams);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomKernelInfoTest, ut_hvd_GetDataTypeFromTaskInfo_Test)
{
    ge::Status ge_ret;
    ge::GETaskInfo task;
    rtStream_t stream;

    HvdOpsKernelInfoStore hvdOpsKernelInfo;

    s8* sendbuf = (s8*)sal_malloc(10 * sizeof(float));
    HCCL_INFO("sendbuf is [%p]", sendbuf);
    sal_memset(sendbuf, 10 * sizeof(float), 0, 10 * sizeof(float));
    s8* recv = (s8*)sal_malloc(20 * sizeof(float));
    HCCL_INFO("recv is [%p]", recv);
    sal_memset(recv, 20 * sizeof(float), 0, 20 * sizeof(float));
    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    ge::GETaskKernelHcclInfo hcclInfo;
    task.kernelHcclInfo.push_back(hcclInfo);
    task.kernelHcclInfo[0].count = 10;
    task.kernelHcclInfo[0].dataType = HCCL_DATA_TYPE_FP32;
    task.kernelHcclInfo[0].hccl_type = HCCL_KERNEL_OP_TYPE_ALLGATHER;
    task.kernelHcclInfo[0].inputDataAddr = sendbuf;
    task.kernelHcclInfo[0].outputDataAddr = recv;
    task.kernelHcclInfo[0].opType = HCCL_REDUCE_SUM;
    task.kernelHcclInfo[0].rootId = 0;
    task.kernelHcclInfo[0].hcclQosCfg = INVALID_QOSCFG;
    task.type = RT_MODEL_TASK_HCCL;
    task.stream = stream;
    HvdKernelInfoPrivateDef privateDefBuf = {0, HCCL_DATA_TYPE_INT8};
    task.privateDef = (void*)&privateDefBuf;
    task.privateDefLen = sizeof(HvdKernelInfoPrivateDef);

    HcclDataType dataType;
    ge_ret = hvdOpsKernelInfo.GetDataTypeFromTaskInfo(task, dataType);
    EXPECT_EQ(ge_ret, HCCL_SUCCESS);

    privateDefBuf = {0, HCCL_DATA_TYPE_RESERVED};
    ge_ret = hvdOpsKernelInfo.GetDataTypeFromTaskInfo(task, dataType);
    EXPECT_EQ(ge_ret, HCCL_E_PARA);

    rt_ret = aclrtDestroyStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sal_free(sendbuf);
    sal_free(recv);
}

#define HCCL_COM_DATA_SIZE 1024
TEST_F(HcomKernelInfoTest, ut_hcom_loadtask_int64)
{
    nlohmann::json rank_table = rank_table_910_1server_1rank;

    char file_name_t[] = "./ut_hcom_loadtask_int64.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    int ret = HCCL_SUCCESS;
    s8* sendbuf;
    s8* recvbuf;
    void* dumpbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    char* rank_table_file = "./ut_hcom_loadtask_int64.json";
    char* rank_ID = "0";

    sendbuf = (s8*)sal_malloc(count);
    sal_memset(sendbuf, count, 0, count);
    recvbuf = (s8*)sal_malloc(count);
    sal_memset(recvbuf, count, 0, count);

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    rtStream_t stream;
    // HcomOpsKernelInfoStore hcomKernelInfo;

    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    ge ::Status ge_ret;
    std::map<std::string, std::string> options;

    options.insert(pair<string, string>(ge::OPTION_EXEC_RANK_TABLE_FILE, "./ut_hcom_loadtask_int64.json"));
    options.insert(pair<string, string>(ge::OPTION_EXEC_RANK_ID, "0"));
    options.insert(pair<string, string>(ge::OPTION_EXEC_DEPLOY_MODE, "0"));
    options.insert(pair<string, string>(ge::OPTION_GRAPH_RUN_MODE, "1"));
    options.insert(pair<string, string>(ge::OPTION_EXEC_ENABLE_DUMP_DEBUG, "1"));
    options.insert(pair<string, string>(ge::OPTION_EXEC_PROFILING_MODE, "0"));
    ret = Initialize(options);
    EXPECT_EQ(ret, ge::SUCCESS);

    std::map<string, OpsKernelInfoStorePtr> opKernInfos;
    std::map<string, GraphOptimizerPtr> graphOptimizers;
    GetOpsKernelInfoStores(opKernInfos);
    GetGraphOptimizerObjs(graphOptimizers);
    OpsKernelInfoStorePtr opsKernerInfoStorePtr = opKernInfos.at(HCCL_OPS_LIB_NAME);
    ge_ret = opsKernerInfoStorePtr->Initialize(options);
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    ge::NodePtr nodeptr(new NodeTest);
    // ge::OpDesc op;
    std::string reason;
    std::string type = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    nodeptr->GetOpDesc()->SetType(type);
    bool bRet = opsKernerInfoStorePtr->CheckSupported(nodeptr->GetOpDesc(), reason);
    EXPECT_EQ(bRet, true);

    HCCL_INFO("test GetAllOpsKernelInfo.");
    std::map<string, ge::OpInfo> infos;
    opsKernerInfoStorePtr->GetAllOpsKernelInfo(infos);

    dumpbuf = sal_malloc(10 * sizeof(s8));
    sal_memset(dumpbuf, 10 * sizeof(s8), 0, 10 * sizeof(s8));
std:
    vector<void*> globalWorkSpaceAddr;
    globalWorkSpaceAddr.push_back(dumpbuf);

    ge::GETaskInfo task;
    ge::GETaskKernelHcclInfo hcclInfo;
    task.kernelHcclInfo.push_back(hcclInfo);
    HCCL_KERNEL_INFO_PRIVATE_DEF privateDefBuf;
    strcpy_s((char*)privateDefBuf.group, 128, "hccl_world_group");
    privateDefBuf.nodeNameHash = 123456;
    privateDefBuf.graphId = 1;
    privateDefBuf.dataType = HCCL_DATA_TYPE_INT8;
    privateDefBuf.tensorNum = 3;
    privateDefBuf.privateDefSize = sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF);
    privateDefBuf.aivCoreLimit = 48;
    task.id = 1;
    task.type = RT_MODEL_TASK_HCCL;
    task.stream = stream;

    std::int64_t offset[privateDefBuf.tensorNum] = {16, 32, 64};
    std::int64_t size[privateDefBuf.tensorNum] = {8, 8, 9};
    size_t tensorInfoSize = sizeof(hcclKernelInfoPrivateDef) + privateDefBuf.tensorNum * sizeof(int64_t) * 2;
    void* privateDefPtr = sal_malloc(tensorInfoSize);
    memset(privateDefPtr, 0, tensorInfoSize);
    memcpy(privateDefPtr, (void*)&privateDefBuf.group[0], sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF));
    memcpy(
        reinterpret_cast<int64_t*>(reinterpret_cast<char*>(privateDefPtr) + sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF)),
        offset, sizeof(offset));
    memcpy(
        reinterpret_cast<int64_t*>(
            reinterpret_cast<char*>(privateDefPtr) + sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF) + sizeof(offset)),
        size, sizeof(size));

    task.privateDef = privateDefPtr;
    task.privateDefLen = (uint32_t)tensorInfoSize;

    // task.privateDef = (void *)&privateDefBuf.group[0];
    // task.privateDefLen = sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF);
    task.kernelHcclInfo[0].count = count;
    task.kernelHcclInfo[0].dataType = HCCL_DATA_TYPE_INT64;
    task.kernelHcclInfo[0].hccl_type = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    task.kernelHcclInfo[0].inputDataAddr = sendbuf;
    task.kernelHcclInfo[0].outputDataAddr = recvbuf;
    task.kernelHcclInfo[0].opType = HCCL_REDUCE_SUM;
    task.kernelHcclInfo[0].rootId = 0;
    task.kernelHcclInfo[0].hcclQosCfg = INVALID_QOSCFG;
    task.kernelHcclInfo[0].global_workspace_addr = globalWorkSpaceAddr;
    task.needRefresh = false;

    u64 memSize = HCCL_WORKSPACE_MEM_32_KB;
    DeviceMem workSpaceMem = DeviceMem::alloc(memSize);
    task.kernelHcclInfo[0].workSpaceAddr = workSpaceMem.ptr();
    task.kernelHcclInfo[0].workSpaceMemSize = workSpaceMem.size();

    MOCKER(HcomGetInitStatus).stubs().with(outBound(true)).will(returnValue(HCCL_SUCCESS));

    MOCKER(InitGroup).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&hccl::OffloadStreamManager::GetSlaves).stubs().will(returnValue(vector<Stream>()));

    MOCKER_CPP(&TbeReduce::TbeVectorReduce::GetTilingDataDevMem)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TbeReduce::TbeCrackCleared::ExecuteKernelLaunch)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    ge_ret = opsKernerInfoStorePtr->LoadTask(task);
    EXPECT_EQ(ge_ret, ge::SUCCESS);
    ge_ret = opsKernerInfoStorePtr->LoadTask(task);
    EXPECT_EQ(ge_ret, ge::INTERNAL_ERROR);

    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sal_free(sendbuf);
    sal_free(recvbuf);
    sal_free(dumpbuf);
    sal_free(privateDefPtr);
    rt_ret = aclrtDestroyStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    ge_ret = opsKernerInfoStorePtr->Finalize();
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    ge_ret = Finalize();
    EXPECT_EQ(ge_ret, ge::SUCCESS);

    remove(file_name_t);

    GlobalMockObject::verify();
}

TEST_F(HcomKernelInfoTest, ut_CleanIntervalMemory_0)
{
    HcomOpsKernelInfoStore hcomKernelInfo;
    rtStream_t stream = NULL;

    std::vector<std::int64_t> crackAddr = {16};
    std::vector<std::int64_t> crackSize = {16};

    MOCKER(hrtMemAsyncCopy).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemSyncCopy).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtGetDeviceType).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    s32 ret = hcomKernelInfo.CleanIntervalMemory("tag", crackAddr, crackSize, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}
HcclResult stub_HcomGetRankSize(const char* group, u32* rankSize)
{
    *rankSize = 1;
    return HCCL_SUCCESS;
}

HcclResult stub_HcomGetRankId(const char* group, u32* rankId)
{
    *rankId = 0;
    return HCCL_SUCCESS;
}

HcclResult stub_HcclCommGraphGetRankSize(s64 opBaseHcom, u32* rankSize)
{
    *rankSize = 1;
    return HCCL_SUCCESS;
}

HcclResult stub_HcclCommGraphGetRankId(s64 opBaseHcom, u32* rankId)
{
    *rankId = 0;
    return HCCL_SUCCESS;
}

TEST_F(HcomKernelInfoTest, ut_opKernelLoop)
{
    HcclResult ret;
    u32 shapeType = ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE;
    const std::string sGroup = HCCL_WORLD_GROUP;
    int64_t hcomComm = 1;
    u32 addr = 0;
    void* inputAddr = &addr;
    void* outputAddr = &addr;
    u64 OutputAddrSize = sizeof(u32);
    Stream stream(StreamType::STREAM_TYPE_OFFLINE);
    std::shared_ptr<hccl::hcclComm> comm;
    comm.reset(new (std::nothrow) hccl::hcclComm());
    HcomOpsKernelInfoStore hcomOpsKernelInfoStore_;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;

    u64 commOutputSize = 102400;

    MOCKER_CPP(&HcomOpsKernelInfoStore::GetHcomOutCCLbufferSize)
        .stubs()
        .with(outBound(commOutputSize), mockcpp::any(), mockcpp::any(), outBound(sGroup))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcomOpsKernelInfoStore::GetHcomInCCLbufferSize)
        .stubs()
        .with(outBound(commOutputSize), mockcpp::any(), mockcpp::any(), outBound(sGroup))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER(HcomGetRankSize).stubs().will(invoke(stub_HcomGetRankSize));

    MOCKER(HcclCommGraphGetRankSize).stubs().will(invoke(stub_HcclCommGraphGetRankSize));

    MOCKER(HcclCommGraphGetRankId).stubs().will(invoke(stub_HcclCommGraphGetRankId));

    MOCKER(HcomGetRankId).stubs().will(invoke(stub_HcomGetRankId));

    DevType type = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(type)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemcpyAddrAsync).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtMemAsyncCopy).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcomOpsKernelInfoStore::CheckHcomOpMemSize)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcomOpsKernelInfoStore::GetOutputCCLbufPtrAndIndirectOutCCLbufPtr)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcomOpsKernelInfoStore::GetInputCCLbufPtrAndIndirectInCCLbufPtr)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcomOpsKernelInfoStore::CleanIntervalMemoryOpKernel)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcomOpsKernelInfoStore::CheckTensorNumAndTensorSize)
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER(HcomGetCommByGroup).stubs().with(mockcpp::any(), outBound(comm)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hccl::hcclComm::GetInCCLbuffer).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&hccl::hcclComm::GetOutCCLbuffer).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(HcomAllGather).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER(HcomReduceScatter).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ge::GETaskInfo task;
    ge::GETaskKernelHcclInfo hcclInfo;
    task.kernelHcclInfo.push_back(hcclInfo);
    HCCL_KERNEL_INFO_PRIVATE_DEF privateDefBuf;
    strcpy_s((char*)privateDefBuf.group, 128, "hccl_world_group");
    privateDefBuf.nodeNameHash = 123456;
    privateDefBuf.graphId = 1;
    privateDefBuf.dataType = HCCL_DATA_TYPE_INT8;
    privateDefBuf.tensorNum = 3;
    privateDefBuf.privateDefSize = sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF);

    std::int64_t offset[privateDefBuf.tensorNum] = {16, 32, 64};
    std::int64_t size[privateDefBuf.tensorNum] = {8, 8, 9};
    size_t tensorInfoSize = sizeof(hcclKernelInfoPrivateDef) + privateDefBuf.tensorNum * sizeof(int64_t) * 2;
    void* privateDefPtr = sal_malloc(tensorInfoSize);
    memset(privateDefPtr, 0, tensorInfoSize);
    memcpy(privateDefPtr, (void*)&privateDefBuf.group[0], sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF));
    memcpy(
        reinterpret_cast<int64_t*>(reinterpret_cast<char*>(privateDefPtr) + sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF)),
        offset, sizeof(offset));
    memcpy(
        reinterpret_cast<int64_t*>(
            reinterpret_cast<char*>(privateDefPtr) + sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF) + sizeof(offset)),
        size, sizeof(size));

    task.privateDef = privateDefPtr;
    task.privateDefLen = (uint32_t)tensorInfoSize;

    HcclReduceOp reduceType = HCCL_REDUCE_SUM;

    u32 srcTag = 0;

    std::vector<std::string> tagVec;
    tagVec.push_back("test");

    hcomOpsKernelInfoStore_.HcomAllGatherLoop(
        tagVec, shapeType, 0, sGroup, inputAddr, outputAddr, 1, dataType, stream.ptr());
    hcomOpsKernelInfoStore_.HcomAllGatherLoop(
        tagVec, shapeType, 1, sGroup, inputAddr, outputAddr, 1, dataType, stream.ptr());
    hcomOpsKernelInfoStore_.HcomAllReduceLoop(
        task, tagVec, shapeType, 0, sGroup, inputAddr, outputAddr, 1, dataType, reduceType, stream.ptr());
    hcomOpsKernelInfoStore_.HcomAllReduceLoop(
        task, tagVec, shapeType, 1, sGroup, inputAddr, outputAddr, 1, dataType, reduceType, stream.ptr());
    hcomOpsKernelInfoStore_.HcomReduceScatterLoop(
        task, tagVec, shapeType, 0, sGroup, inputAddr, outputAddr, 1, dataType, reduceType, stream.ptr());
    hcomOpsKernelInfoStore_.HcomReduceScatterLoop(
        task, tagVec, shapeType, 1, sGroup, inputAddr, outputAddr, 1, dataType, reduceType, stream.ptr());
    hcomOpsKernelInfoStore_.HcomReduceLoop(
        task, tagVec, shapeType, 0, sGroup, inputAddr, outputAddr, 1, dataType, reduceType, 0, stream.ptr());
    hcomOpsKernelInfoStore_.HcomReduceLoop(
        task, tagVec, shapeType, 1, sGroup, inputAddr, outputAddr, 1, dataType, reduceType, 0, stream.ptr());
    hcomOpsKernelInfoStore_.HcomSendLoop(
        tagVec, srcTag, shapeType, 0, sGroup, inputAddr, 1, dataType, srcTag, stream.ptr());
    hcomOpsKernelInfoStore_.HcomSendLoop(
        tagVec, srcTag, shapeType, 1, sGroup, inputAddr, 1, dataType, srcTag, stream.ptr());
    hcomOpsKernelInfoStore_.HcomReceiveLoop(
        tagVec, srcTag, shapeType, 0, sGroup, outputAddr, 1, dataType, srcTag, stream.ptr());
    hcomOpsKernelInfoStore_.HcomReceiveLoop(
        tagVec, srcTag, shapeType, 1, sGroup, outputAddr, 1, dataType, srcTag, stream.ptr());

    sal_free(privateDefPtr);
    GlobalMockObject::verify();
}

TEST_F(HcomKernelInfoTest, ut_GetOpKernelLoopTime1)
{
    u64 count = 128;
    const std::string sGroup = HCCL_WORLD_GROUP;
    HcomOpsKernelInfoStore hcomKernelInfo;

    ge::GETaskInfo task;
    ge::GETaskKernelHcclInfo hcclInfo;
    task.kernelHcclInfo.push_back(hcclInfo);
    HCCL_KERNEL_INFO_PRIVATE_DEF privateDefBuf;
    strcpy_s((char*)privateDefBuf.group, 128, "hccl_world_group");
    privateDefBuf.nodeNameHash = 123456;
    privateDefBuf.graphId = 1;
    privateDefBuf.dataType = HCCL_DATA_TYPE_INT8;
    privateDefBuf.tensorNum = 0;
    privateDefBuf.privateDefSize = sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF);
    privateDefBuf.devType = DevType::DEV_TYPE_910;
    privateDefBuf.comm = 0;
    privateDefBuf.originalGraphShapeType = ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE;
    task.id = 1;
    task.type = RT_MODEL_TASK_HCCL;
    task.privateDef = (void*)&privateDefBuf.group[0];
    task.privateDefLen = sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF);
    task.kernelHcclInfo[0].count = count;
    task.kernelHcclInfo[0].dataType = HCCL_DATA_TYPE_INT8;
    task.kernelHcclInfo[0].hccl_type = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    task.kernelHcclInfo[0].opType = HCCL_REDUCE_SUM;
    task.kernelHcclInfo[0].rootId = 0;
    task.needRefresh = false;

    u64 commOutputSize = 102400;

    MOCKER_CPP(&HcomOpsKernelInfoStore::GetHcomOutCCLbufferSize)
        .stubs()
        .with(outBound(commOutputSize), mockcpp::any(), mockcpp::any(), outBound(sGroup))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcomOpsKernelInfoStore::GetHcomInCCLbufferSize)
        .stubs()
        .with(outBound(commOutputSize), mockcpp::any(), mockcpp::any(), outBound(sGroup))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER(HcomGetRankSize).stubs().will(invoke(stub_HcomGetRankSize));

    u32 loopMaxTime = 0;
    std::string opType = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    std::string opType1 = HCCL_KERNEL_OP_TYPE_REDUCESCATTER;
    std::string opType2 = HCCL_KERNEL_OP_TYPE_RECEIVE;
    std::string opType3 = HCCL_KERNEL_OP_TYPE_ALLGATHER;

    std::string srctag = "test";

    hcomKernelInfo.GetOpKernelLoopTime(task, opType, srctag, loopMaxTime);
    hcomKernelInfo.GetOpKernelLoopTime(task, opType1, srctag, loopMaxTime);
    hcomKernelInfo.GetOpKernelLoopTime(task, opType2, srctag, loopMaxTime);
    hcomKernelInfo.GetOpKernelLoopTime(task, opType3, srctag, loopMaxTime);

    GlobalMockObject::verify();
}

TEST_F(HcomKernelInfoTest, ut_GetOpKernelLoopTime2)
{
    u64 count = 128;
    const std::string sGroup = HCCL_WORLD_GROUP;
    HcomOpsKernelInfoStore hcomKernelInfo;

    ge::GETaskInfo task;
    ge::GETaskKernelHcclInfo hcclInfo;
    task.kernelHcclInfo.push_back(hcclInfo);
    HCCL_KERNEL_INFO_PRIVATE_DEF privateDefBuf;
    strcpy_s((char*)privateDefBuf.group, 128, "hccl_world_group");
    privateDefBuf.nodeNameHash = 123456;
    privateDefBuf.graphId = 1;
    privateDefBuf.dataType = HCCL_DATA_TYPE_INT8;
    privateDefBuf.tensorNum = 0;
    privateDefBuf.privateDefSize = sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF);
    privateDefBuf.devType = DevType::DEV_TYPE_910;
    privateDefBuf.comm = 1;
    privateDefBuf.originalGraphShapeType = ORIGINAL_GRAPH_UNKNOWNSHAPE_TYPE;
    task.id = 1;
    task.type = RT_MODEL_TASK_HCCL;
    task.privateDef = (void*)&privateDefBuf.group[0];
    task.privateDefLen = sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF);
    task.kernelHcclInfo[0].count = count;
    task.kernelHcclInfo[0].dataType = HCCL_DATA_TYPE_INT8;
    task.kernelHcclInfo[0].hccl_type = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    task.kernelHcclInfo[0].opType = HCCL_REDUCE_SUM;
    task.kernelHcclInfo[0].rootId = 0;
    task.needRefresh = false;

    u64 commOutputSize = 102400;

    MOCKER_CPP(&HcomOpsKernelInfoStore::GetHcomOutCCLbufferSize)
        .stubs()
        .with(outBound(commOutputSize), mockcpp::any(), mockcpp::any(), outBound(sGroup))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcomOpsKernelInfoStore::GetHcomInCCLbufferSize)
        .stubs()
        .with(outBound(commOutputSize), mockcpp::any(), mockcpp::any(), outBound(sGroup))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER(HcomGetRankSize).stubs().will(invoke(stub_HcomGetRankSize));

    MOCKER(HcclCommGraphGetRankSize).stubs().will(invoke(stub_HcclCommGraphGetRankSize));

    MOCKER(HcclCommGraphGetRankId).stubs().will(invoke(stub_HcclCommGraphGetRankId));

    u32 loopMaxTime = 0;
    std::string opType1 = HCCL_KERNEL_OP_TYPE_REDUCESCATTER;
    std::string opType2 = HCCL_KERNEL_OP_TYPE_ALLGATHER;

    std::string srctag = "test";

    hcomKernelInfo.GetOpKernelLoopTime(task, opType1, srctag, loopMaxTime);
    hcomKernelInfo.GetOpKernelLoopTime(task, opType2, srctag, loopMaxTime);

    GlobalMockObject::verify();
}

TEST_F(HcomKernelInfoTest, ut_GetTagVectorInfo)
{
    const std::string sCollectiveType = HCCL_KERNEL_OP_TYPE_ALLREDUCE;
    HcomOpsKernelInfoStore hcomKernelInfo;

    ge::GETaskInfo task;
    ge::GETaskKernelHcclInfo hcclInfo;
    task.kernelHcclInfo.push_back(hcclInfo);

    std::string tag = "test";
    u32 loopMaxTime = 3;
    std::vector<std::string> tagVec;

    MOCKER_CPP(&HcomOpsKernelInfoStore::GenerateOpTagFromTaskInfo)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(tag), outBound(loopMaxTime))
        .will(returnValue(HCCL_SUCCESS));

    s32 ret = hcomKernelInfo.GetTagVectorInfo(task, sCollectiveType, tagVec);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcomKernelInfoTest, ut_HcclCommGraphSetWorkspaceResource)
{
    ge::GETaskInfo task;
    ge::GETaskKernelHcclInfo hcclInfo;
    task.kernelHcclInfo.push_back(hcclInfo);

    std::vector<rtStream_t> all_hccl_stream_list;
    rtStream_t stream;
    HcomOpsKernelInfoStore hcomKernelInfo;

    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    all_hccl_stream_list.push_back(stream);
    task.kernelHcclInfo[0].hcclStreamList = all_hccl_stream_list;

    std::vector<std::string> tagVec;
    std::string tag = "test";
    tagVec.push_back(tag);

    void* memPtr;

    MOCKER(HcclCommGraphSetWorkspaceResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    s32 ret = hcomKernelInfo.CommGraphSetWorkspaceResourceFromtagVec(task, 0, tagVec, memPtr, 10);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    aclrtDestroyStream(stream);
    GlobalMockObject::verify();
}

ge::graphStatus
RoleTableGetOption(ge::GEThreadLocalContext* that, const std::string& optionExec, std::string& dumpDebugValue)
{
    dumpDebugValue = "10";
    return ge::GRAPH_SUCCESS;
}

TEST_F(HcomKernelInfoTest, ut_ReStartVnic)
{
    HcclResult ret;
    HcomOpsKernelInfoStore hcomKernelInfo;

    MOCKER_CPP(&ge::GEThreadLocalContext::GetOption).stubs().will(invoke(RoleTableGetOption));

    MOCKER_CPP(&HcclCommunicator::ReStartVnic).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    std::shared_ptr<hccl::hcclComm> comm;
    comm.reset(new (std::nothrow) hccl::hcclComm());
    comm->communicator_.reset(new (std::nothrow) HcclCommunicator());
    s64 commId = reinterpret_cast<s64>(comm.get());

    ret = hcomKernelInfo.ReStartVnic(commId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcomKernelInfoTest, ut_Communicator_ReStartVnic)
{
    HcclResult ret;
    HcclCommunicator hcclCommunicator;

    MOCKER_CPP(&NetworkManager::StopVnic).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&NetworkManager::StartVnic).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&NetworkManager::IsHasStartVnic).stubs().with(mockcpp::any()).will(returnValue(true));

    HcclCommParams params;
    RankTable_t rankTable;
    RankInfo_t rankInfo;
    params.rank = 0;
    rankInfo.rankId = 0;
    rankInfo.deviceInfo.port = 2000;
    rankTable.rankList.push_back(rankInfo);
    ret = hcclCommunicator.ReStartVnic(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcomKernelInfoTest, ut_Communicator_CheckReduceDataType)
{
    HcclResult ret = HCCL_SUCCESS;
    HcclDataType dataType = HCCL_DATA_TYPE_INT16;
    HcclReduceOp op = HCCL_REDUCE_PROD;
    HcclCommunicator hcclCommunicator;

    hcclCommunicator.deviceType_ = DevType::DEV_TYPE_910;

    HcclCommParams params;
    RankTable_t rankTable;
    RankInfo_t rankInfo;
    params.rank = 0;
    rankInfo.rankId = 0;
    rankInfo.deviceInfo.port = 2000;
    rankTable.rankList.push_back(rankInfo);

    ret = hcclCommunicator.CheckReduceDataType(dataType, op);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);

    hcclCommunicator.deviceType_ = DevType::DEV_TYPE_910B;
    ret = hcclCommunicator.CheckReduceDataType(dataType, op);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);

    hcclCommunicator.deviceType_ = DevType::DEV_TYPE_310P3;
    ret = hcclCommunicator.CheckReduceDataType(dataType, op);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    GlobalMockObject::verify();
}

ge::graphStatus
EsMaxRemoteOpNumGetOption1(ge::GEThreadLocalContext* that, const std::string& optionExec, std::string& dumpDebugValue)
{
    dumpDebugValue = "0.0";
    return ge::GRAPH_SUCCESS;
}

TEST_F(HcomKernelInfoTest, ut_HcclGetEsMaxRemoteOpNumPerStreamFromGe)
{
    MOCKER_CPP(&ge::GEThreadLocalContext::GetOption).stubs().will(invoke(EsMaxRemoteOpNumGetOption1));

    GetEsMaxRemoteOpNumPerStreamFromGe();
}

TEST_F(HcomKernelInfoTest, ut_CheckWorkSpaceNeedSet)
{
    HcclResult ret;
    HcomOpsKernelInfoStore hcomKernelInfo;

    ge::GETaskKernelHcclInfo hcclInfo;
    ge::GETaskInfo task;
    hcclInfo.workSpaceAddr = nullptr;
    hcclInfo.workSpaceMemSize = 0;

    HCCL_KERNEL_INFO_PRIVATE_DEF privateDefBuf;
    strcpy_s((char*)privateDefBuf.group, 128, "hccl_world_group");
    privateDefBuf.nodeNameHash = 123456;
    privateDefBuf.graphId = 1;
    privateDefBuf.dataType = HCCL_DATA_TYPE_INT8;
    privateDefBuf.tensorNum = 0;
    privateDefBuf.privateDefSize = sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF);

    privateDefBuf.needMapRank = true;
    privateDefBuf.isOfflineComp = true;
    privateDefBuf.devType = DevType::DEV_TYPE_910;
    strcpy_s((char*)privateDefBuf.esInfo.uniqueTag, 64, "12345");
    task.id = 1;
    task.type = RT_MODEL_TASK_HCCL;
    task.privateDef = (void*)&privateDefBuf.group[0];
    task.privateDefLen = sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF);

    std::string tag = "test";
    std::vector<std::string> tagVec;
    tagVec.push_back(tag);

    std::string sCollectiveType = HCCL_KERNEL_OP_TYPE_GATHER;
    bool isNeedSet;
    ret = hcomKernelInfo.CheckWorkSpaceNeedSet(task, hcclInfo, sCollectiveType, isNeedSet);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(HcomKernelInfoTest, ut_SetAttachedStream)
{
    HcclResult ret;
    HcomOpsKernelInfoStore hcomKernelInfo;

    ge::GETaskKernelHcclInfo hcclInfo;
    ge::GETaskInfo task;
    hcclInfo.workSpaceAddr = nullptr;
    hcclInfo.workSpaceMemSize = 0;

    HCCL_KERNEL_INFO_PRIVATE_DEF privateDefBuf;
    strcpy_s((char*)privateDefBuf.group, 128, "hccl_world_group");
    privateDefBuf.nodeNameHash = 123456;
    privateDefBuf.graphId = 1;
    privateDefBuf.dataType = HCCL_DATA_TYPE_INT8;
    privateDefBuf.tensorNum = 0;
    privateDefBuf.privateDefSize = sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF);

    privateDefBuf.needMapRank = true;
    privateDefBuf.isOfflineComp = true;
    privateDefBuf.devType = DevType::DEV_TYPE_910;
    strcpy_s((char*)privateDefBuf.esInfo.uniqueTag, 64, "12345");
    task.id = 1;
    task.type = RT_MODEL_TASK_HCCL;
    task.privateDef = (void*)&privateDefBuf.group[0];
    task.privateDefLen = sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF);

    task.rt_attached_streams.push_back((void*)(0x123)); // 配置dummystream
    MOCKER(&HcomSetAttachedStream).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(&HcclCommSetAttachedStream).stubs().will(returnValue(HCCL_SUCCESS));

    privateDefBuf.comm == 0;
    EXPECT_EQ(hcomKernelInfo.SetAttachedStream(task), HCCL_SUCCESS);

    privateDefBuf.comm == 0x123;
    EXPECT_EQ(hcomKernelInfo.SetAttachedStream(task), HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(HcomKernelInfoTest, ut_HcomAicpuStreamRegister)
{
    HcomOpsKernelInfoStore hcomKernelInfo;
    ge::GETaskInfo task;
    aclrtCreateStream(&task.stream, 5);
    std::string group1 = "test1";

    std::shared_ptr<hccl::hcclComm> comm;
    comm.reset(new (std::nothrow) hccl::hcclComm());
    comm->communicator_.reset(new (std::nothrow) HcclCommunicator());
    int64_t commStub = 0;

    MOCKER_CPP(&HcomOpsKernelInfoStore::GetCommFromTaskInfo)
        .stubs()
        .with(mockcpp::any(), outBound(commStub))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcomOpsKernelInfoStore::GetGroupFromTaskInfo)
        .stubs()
        .with(mockcpp::any(), outBound(group1))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(HcomGetCommByGroup).stubs().with(mockcpp::any(), outBound(comm)).will(returnValue(HCCL_SUCCESS));

    // 多次注册、解注册，验证引用计数功能
    EXPECT_EQ(hcomKernelInfo.HcomAicpuStreamRegister(task), HCCL_SUCCESS);
    EXPECT_EQ(hcomKernelInfo.HcomAicpuStreamRegister(task), HCCL_SUCCESS);
    EXPECT_EQ(hcomKernelInfo.HcomAicpuStreamUnRegister(task), HCCL_SUCCESS);
    EXPECT_EQ(hcomKernelInfo.HcomAicpuStreamUnRegister(task), HCCL_SUCCESS);
    EXPECT_EQ(hcomKernelInfo.HcomAicpuStreamUnRegister(task), HCCL_SUCCESS);

    aclrtDestroyStream(task.stream);
    GlobalMockObject::verify();
}

TEST_F(HcomKernelInfoTest, ut_LoadTaskSetAivCoreLimit)
{
    HcclResult ret;
    HcomOpsKernelInfoStore hcomKernelInfo;
    ge::GETaskInfo task;
    HCCL_KERNEL_INFO_PRIVATE_DEF privateDefBuf;

    task.id = 1;
    task.type = RT_MODEL_TASK_HCCL;
    task.privateDef = (void*)&privateDefBuf;
    task.privateDefLen = sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF);

    strcpy_s((char*)privateDefBuf.group, 128, "hccl_world_group");
    privateDefBuf.nodeNameHash = 123456;
    privateDefBuf.graphId = 1;
    privateDefBuf.dataType = HCCL_DATA_TYPE_INT8;
    privateDefBuf.privateDefSize = sizeof(HCCL_KERNEL_INFO_PRIVATE_DEF);
    privateDefBuf.devType = DevType::DEV_TYPE_910;

    privateDefBuf.aivCoreLimit = 0;
    ret = hcomKernelInfo.SetAivCoreLimit(task);
    EXPECT_EQ(ret, HCCL_E_PARA);

    u32 numBlocks1 = 0;
    MOCKER(&HcomSetAivCoreLimit).stubs().with(mockcpp::any(), spy(numBlocks1)).will(returnValue(HCCL_SUCCESS));
    privateDefBuf.comm = 0;
    privateDefBuf.aivCoreLimit = 7;
    ret = hcomKernelInfo.SetAivCoreLimit(task);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(numBlocks1, privateDefBuf.aivCoreLimit);

    u32 numBlocks2 = 0;
    privateDefBuf.comm = 0x1234;
    privateDefBuf.aivCoreLimit = 8;
    ret = hcomKernelInfo.SetAivCoreLimit(task);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(numBlocks2, privateDefBuf.aivCoreLimit);

    GlobalMockObject::verify();
}

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "hccl/hccl_res.h"
#include "../../hccl_api_base_test.h"
#include "hccl_tbe_task.h"
#include "llt_hccl_stub_GenRankTable.h"
#include "llt_hccl_stub_pub.h"
#include "hccl_communicator.h"
#include "hccl_types.h"
#include "hccl_comm_pub.h"
#include "dlra_function.h"
#include "hcom_common.h"

using namespace hccl;
// 每个超节点内包含两个AI Server，每个AI Server内四个Device的资源配置文件为例
using json = nlohmann::json;
static json rank_table_910C_2superPod_2server_4rank_1nic
    = {{"status", "completed"},
       {"version", "1.2"},
       {"server_count", "4"},
       {"server_list",
        {{{"server_id", "node_0"},
          {"host_ip", "172.16.0.100"},
          {"device",
           {{{"device_id", "0"},
             {"super_device_id", "0"},
             {"device_ip", "192.168.1.6"},
             {"device_port", "16666"},
             {"backup_device_ip", "192.168.1.7"},
             {"backup_device_port", "16667"},
             {"host_port", "16665"},
             {"rank_id", "0"}},
            {{"device_id", "1"},
             {"super_device_id", "1"},
             {"device_ip", "192.168.1.7"},
             {"device_port", "16666"},
             {"backup_device_ip", "192.168.1.6"},
             {"backup_device_port", "16667"},
             {"host_port", "16666"},
             {"rank_id", "1"}},
            {{"device_id", "2"},
             {"super_device_id", "2"},
             {"device_ip", "192.168.1.8"},
             {"device_port", "16668"},
             {"backup_device_ip", "192.168.1.9"},
             {"backup_device_port", "16670"},
             {"host_port", "16667"},
             {"rank_id", "2"}},
            {{"device_id", "3"},
             {"super_device_id", "3"},
             {"device_ip", "192.168.1.9"},
             {"device_port", "16669"},
             {"backup_device_ip", "192.168.1.8"},
             {"backup_device_port", "16667"},
             {"host_port", "16668"},
             {"rank_id", "3"}}}}},
         {{"server_id", "node_1"},
          {"host_ip", "172.16.0.101"},
          {"device",
           {{{"device_id", "0"},
             {"super_device_id", "4"},
             {"device_ip", "192.168.2.6"},
             {"device_port", "16666"},
             {"backup_device_ip", "192.168.2.7"},
             {"backup_device_port", "16667"},
             {"host_port", "16665"},
             {"rank_id", "4"}},
            {{"device_id", "1"},
             {"super_device_id", "5"},
             {"device_ip", "192.168.2.7"},
             {"device_port", "16666"},
             {"backup_device_ip", "192.168.2.6"},
             {"backup_device_port", "16667"},
             {"host_port", "16666"},
             {"rank_id", "5"}},
            {{"device_id", "2"},
             {"super_device_id", "6"},
             {"device_ip", "192.168.2.8"},
             {"device_port", "16668"},
             {"backup_device_ip", "192.168.2.9"},
             {"backup_device_port", "16670"},
             {"host_port", "16667"},
             {"rank_id", "6"}},
            {{"device_id", "3"},
             {"super_device_id", "7"},
             {"device_ip", "192.168.2.9"},
             {"device_port", "16669"},
             {"backup_device_ip", "192.168.2.8"},
             {"backup_device_port", "16667"},
             {"host_port", "16668"},
             {"rank_id", "7"}}}}},
         {{"server_id", "node_2"},
          {"host_ip", "172.16.0.102"},
          {"device",
           {{{"device_id", "0"},
             {"super_device_id", "0"},
             {"device_ip", "192.168.3.6"},
             {"device_port", "16666"},
             {"backup_device_ip", "192.168.3.7"},
             {"backup_device_port", "16667"},
             {"host_port", "16665"},
             {"rank_id", "8"}},
            {{"device_id", "1"},
             {"super_device_id", "1"},
             {"device_ip", "192.168.3.7"},
             {"device_port", "16666"},
             {"backup_device_ip", "192.168.3.6"},
             {"backup_device_port", "16667"},
             {"host_port", "16666"},
             {"rank_id", "9"}},
            {{"device_id", "2"},
             {"super_device_id", "2"},
             {"device_ip", "192.168.3.8"},
             {"device_port", "16668"},
             {"backup_device_ip", "192.168.3.9"},
             {"backup_device_port", "16670"},
             {"host_port", "16667"},
             {"rank_id", "10"}},
            {{"device_id", "3"},
             {"super_device_id", "3"},
             {"device_ip", "192.168.3.9"},
             {"device_port", "16669"},
             {"backup_device_ip", "192.168.3.8"},
             {"backup_device_port", "16667"},
             {"host_port", "16668"},
             {"rank_id", "11"}}}}},
         {{"server_id", "node_3"},
          {"host_ip", "172.16.0.103"},
          {"device",
           {{{"device_id", "0"},
             {"super_device_id", "4"},
             {"device_ip", "192.168.4.6"},
             {"device_port", "16666"},
             {"backup_device_ip", "192.168.4.7"},
             {"backup_device_port", "16667"},
             {"host_port", "16665"},
             {"rank_id", "12"}},
            {{"device_id", "1"},
             {"super_device_id", "5"},
             {"device_ip", "192.168.4.7"},
             {"device_port", "16666"},
             {"backup_device_ip", "192.168.4.6"},
             {"backup_device_port", "16667"},
             {"host_port", "16666"},
             {"rank_id", "13"}},
            {{"device_id", "2"},
             {"super_device_id", "6"},
             {"device_ip", "192.168.4.8"},
             {"device_port", "16668"},
             {"backup_device_ip", "192.168.4.9"},
             {"backup_device_port", "16670"},
             {"host_port", "16667"},
             {"rank_id", "14"}},
            {{"device_id", "3"},
             {"super_device_id", "7"},
             {"device_ip", "192.168.4.9"},
             {"device_port", "16669"},
             {"backup_device_ip", "192.168.4.8"},
             {"backup_device_port", "16667"},
             {"host_port", "16668"},
             {"rank_id", "15"}}}}}}},
       {"super_pod_list",
        {{{"super_pod_id", "0"}, {"server_list", {{{"server_id", "node_0"}}, {{"server_id", "node_1"}}}}},
         {{"super_pod_id", "1"}, {"server_list", {{{"server_id", "node_2"}}, {{"server_id", "node_3"}}}}}}}};

static json rank_table_2server_2rank
    = {{"status", "completed"}, // rank table可用标识，completed为可用
       {"version", "1.0"},      // rank table模板版本信息，配置为：1.0
       {"server_count", "2"},   // 参与训练的AI Server数目，此例中，有两个AI Server
       {"server_list",
        {                          // server_list[0]：第一个AI Server
         {{"server_id", "node_0"}, // AI Server标识，String类型，请确保全局唯一
          {"device",
           {// device[0]：第一个设备
            {
                {"device_id", "0"},           // 处理器的物理ID
                {"device_ip", "192.168.1.8"}, // 处理器真实网卡IP
                {"device_port", "16667"},     // 处理器的网卡监听端口
                {"rank_id", "0"}              // rank的标识，从0开始配置，请确保全局唯一
            },
            // device[1]：第二个设备
            {{"device_id", "1"}, {"device_ip", "192.168.1.9"}, {"device_port", "16667"}, {"rank_id", "1"}}}}},
         // server_list[1]：第二个AI Server
         {{"server_id", "node_1"},
          {"device",
           {// device[0]：第一个设备
            {{"device_id", "0"}, {"device_ip", "192.168.2.8"}, {"device_port", "16667"}, {"rank_id", "2"}},
            // device[1]：第二个设备
            {{"device_id", "1"}, {"device_ip", "192.168.2.9"}, {"device_port", "16667"}, {"rank_id", "3"}}}}}}}};

static json rank_table_910B_1server_16rank
    = {{"status", "completed"}, // rank table可用标识，completed为可用
       {"version", "1.0"},      // rank table模板版本信息，配置为：1.0
       {"server_count", "1"},   // 参与训练的AI Server数目：单机部署，固定为1
       {"server_list",
        {                          // server_list[0]：唯一的1台AI Server
         {{"server_id", "node_0"}, // AI Server标识，全局唯一
          {"device",
           {{{"device_id", "0"}, {"device_ip", "192.168.1.10"}, {"device_port", "16667"}, {"rank_id", "0"}},
            {{"device_id", "1"}, {"device_ip", "192.168.1.11"}, {"device_port", "16667"}, {"rank_id", "1"}},
            {{"device_id", "2"}, {"device_ip", "192.168.1.12"}, {"device_port", "16667"}, {"rank_id", "2"}},
            {{"device_id", "3"}, {"device_ip", "192.168.1.13"}, {"device_port", "16667"}, {"rank_id", "3"}},
            {{"device_id", "4"}, {"device_ip", "192.168.1.14"}, {"device_port", "16667"}, {"rank_id", "4"}},
            {{"device_id", "5"}, {"device_ip", "192.168.1.15"}, {"device_port", "16667"}, {"rank_id", "5"}},
            {{"device_id", "6"}, {"device_ip", "192.168.1.16"}, {"device_port", "16667"}, {"rank_id", "6"}},
            {{"device_id", "7"}, {"device_ip", "192.168.1.17"}, {"device_port", "16667"}, {"rank_id", "7"}},
            {{"device_id", "8"}, {"device_ip", "192.168.1.18"}, {"device_port", "16667"}, {"rank_id", "8"}},
            {{"device_id", "9"}, {"device_ip", "192.168.1.19"}, {"device_port", "16667"}, {"rank_id", "9"}},
            {{"device_id", "10"}, {"device_ip", "192.168.1.20"}, {"device_port", "16667"}, {"rank_id", "10"}},
            {{"device_id", "11"}, {"device_ip", "192.168.1.21"}, {"device_port", "16667"}, {"rank_id", "11"}},
            {{"device_id", "12"}, {"device_ip", "192.168.1.22"}, {"device_port", "16667"}, {"rank_id", "12"}},
            {{"device_id", "13"}, {"device_ip", "192.168.1.23"}, {"device_port", "16667"}, {"rank_id", "13"}},
            {{"device_id", "14"}, {"device_ip", "192.168.1.24"}, {"device_port", "16667"}, {"rank_id", "14"}},
            {{"device_id", "15"}, {"device_ip", "192.168.1.25"}, {"device_port", "16667"}, {"rank_id", "15"}}}}}}}};

static json rank_table_2server_4rank
    = {{"status", "completed"},
       {"version", "1.0"},
       {"server_count", "2"},
       {"server_list",
        {{{"server_id", "node_0"},
          {"device",
           {{{"device_id", "0"}, {"device_ip", "192.168.1.8"}, {"device_port", "16667"}, {"rank_id", "0"}},
            {{"device_id", "1"}, {"device_ip", "192.168.1.9"}, {"device_port", "16667"}, {"rank_id", "1"}},
            {{"device_id", "2"}, {"device_ip", "192.168.1.10"}, {"device_port", "16667"}, {"rank_id", "2"}},
            {{"device_id", "3"}, {"device_ip", "192.168.1.11"}, {"device_port", "16667"}, {"rank_id", "3"}}}}},
         {{"server_id", "node_1"},
          {"device",
           {{{"device_id", "0"}, {"device_ip", "192.168.2.8"}, {"device_port", "16667"}, {"rank_id", "4"}},
            {{"device_id", "1"}, {"device_ip", "192.168.2.9"}, {"device_port", "16667"}, {"rank_id", "5"}},
            {{"device_id", "2"}, {"device_ip", "192.168.2.10"}, {"device_port", "16667"}, {"rank_id", "6"}},
            {{"device_id", "3"}, {"device_ip", "192.168.2.11"}, {"device_port", "16667"}, {"rank_id", "7"}}}}}}}};

static const char* RANKTABLE_FILE_NAME = "./ut_independent_op_test.json";
aclError aclrtGetLogicDevIdByPhyDevId_stub(const int32_t phyDevId, int32_t* const logicDevId)
{
    *logicDevId = phyDevId;
    return RT_ERROR_NONE;
}

class HcclIndependentOpRankGraphTest : public BaseInit {
public:
    void SetUp() override
    {
        MOCKER(&HcclCommunicator::InitRaResource).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(aclrtGetLogicDevIdByPhyDevId)
            .stubs()
            .with(mockcpp::any())
            .will(invoke(aclrtGetLogicDevIdByPhyDevId_stub));
        BaseInit::SetUp();
    }
    void TearDown() override
    {
        unsetenv("HCCL_INTRA_ROCE_ENABLE");
        unsetenv("HCCL_INTRA_PCIE_ENABLE");
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }
};

HcclResult DevTypeToCommProtocol_stub(RankGraph* graph, DevType& type, CommProtocol& protocol)
{
    type = DevType::DEV_TYPE_910_93;
    protocol = COMM_PROTOCOL_ROCE;
    return HCCL_SUCCESS;
}

void Create91093Comm(HcclComm* comm)
{
    MOCKER_CPP(&RankGraph::DevTypeToCommProtocol).stubs().with(mockcpp::any()).will(invoke(DevTypeToCommProtocol_stub));
    MOCKER(hrtGetDeviceInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(IsSuperPodMode).stubs().with(outBound(true)).will(returnValue(HCCL_SUCCESS));
    MOCKER(IsSupportAtomicWrite).stubs().will(returnValue(HCCL_SUCCESS));
    Ut_Clusterinfo_File_Create(RANKTABLE_FILE_NAME, rank_table_910C_2superPod_2server_4rank_1nic);
    HcclResult ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclCommInitClusterInfo(RANKTABLE_FILE_NAME, 0, comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(comm != nullptr, true);
}

void DestroyComm(HcclComm comm)
{
    remove(RANKTABLE_FILE_NAME);
    Ut_Comm_Destroy(comm);
    comm = nullptr;
}

// 910C 用例
TEST_F(HcclIndependentOpRankGraphTest, Ut_HcclRankGraphGetLinks_When_Param_Is_Null_Expect_PtrError)
{
    HcclResult ret = HcclRankGraphGetLinks(nullptr, 0, 0, 0, nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
    Create91093Comm(&comm);
    ret = HcclRankGraphGetLinks(comm, 0, 0, 0, nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
    uint32_t listSize = 0;
    CommLink* linkList = nullptr;
    ret = HcclRankGraphGetLinks(comm, 0, 0, 0, &linkList, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
    ret = HcclRankGraphGetLinks(comm, 0, 0, 0, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_E_PARA);
    ret = HcclRankGraphGetLinks(comm, 0, 0, 100, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_E_PARA);
    ret = HcclRankGraphGetLinks(comm, 100, 0, 1, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_E_PARA);
    DestroyComm(comm);
}

/**
A3 同个server情况
两个DIE间：
netLayer = 0， SIO连接
netLayer = 1， HCCS连接
netLayer = 2， 无

A3 非DIE间：
netLayer = 0， HCCS
netLayer = 1， 无 (如果是超节点，可能是HCCS)
netLayer = 2， 无
*/
TEST_F(HcclIndependentOpRankGraphTest, Ut_HcclRankGraphGetLinks_When_In_Same_Server)
{
    CommLink* linkList = nullptr;
    uint32_t listSize = 0;
    set_chip_type_stub(0, static_cast<s32>(DevType::DEV_TYPE_910_93));
    Create91093Comm(&comm);
    HcclResult ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_0, 0, 1, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    for (int i = 0; i < listSize; i++) {
        EXPECT_EQ(linkList[i].srcEndpointDesc.commAddr.id, 0);
        EXPECT_EQ(linkList[i].dstEndpointDesc.commAddr.id, 1);
        EXPECT_EQ(linkList[i].srcEndpointDesc.protocol, COMM_PROTOCOL_SIO);
        EXPECT_EQ(linkList[i].dstEndpointDesc.protocol, COMM_PROTOCOL_SIO);
        EXPECT_EQ(linkList[i].linkAttr.linkProtocol, COMM_PROTOCOL_SIO);
    }
    EXPECT_EQ(listSize, 1);
    ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_1, 0, 1, &linkList, &listSize);
    for (int i = 0; i < listSize; i++) {
        EXPECT_EQ(linkList[i].srcEndpointDesc.commAddr.id, 0);
        EXPECT_EQ(linkList[i].dstEndpointDesc.commAddr.id, 1);
        EXPECT_EQ(linkList[i].srcEndpointDesc.protocol, COMM_PROTOCOL_HCCS);
        EXPECT_EQ(linkList[i].dstEndpointDesc.protocol, COMM_PROTOCOL_HCCS);
        EXPECT_EQ(linkList[i].linkAttr.linkProtocol, COMM_PROTOCOL_HCCS);
    }
    EXPECT_EQ(listSize, 1);
    ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_2, 0, 1, &linkList, &listSize);
    EXPECT_EQ(listSize, 0);
    EXPECT_EQ(linkList, nullptr);

    ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_0, 0, 2, &linkList, &listSize);
    for (int i = 0; i < listSize; i++) {
        EXPECT_EQ(linkList[i].srcEndpointDesc.commAddr.id, 0);
        EXPECT_EQ(linkList[i].dstEndpointDesc.commAddr.id, 2);
        EXPECT_EQ(linkList[i].srcEndpointDesc.protocol, COMM_PROTOCOL_HCCS);
        EXPECT_EQ(linkList[i].dstEndpointDesc.protocol, COMM_PROTOCOL_HCCS);
        EXPECT_EQ(linkList[i].linkAttr.linkProtocol, COMM_PROTOCOL_HCCS);
    }
    EXPECT_EQ(listSize, 1);

    ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_1, 0, 2, &linkList, &listSize);
    for (int i = 0; i < listSize; i++) {
        EXPECT_EQ(linkList[i].srcEndpointDesc.commAddr.id, 0);
        EXPECT_EQ(linkList[i].dstEndpointDesc.commAddr.id, 2);
        EXPECT_EQ(linkList[i].srcEndpointDesc.protocol, COMM_PROTOCOL_HCCS);
        EXPECT_EQ(linkList[i].dstEndpointDesc.protocol, COMM_PROTOCOL_HCCS);
        EXPECT_EQ(linkList[i].linkAttr.linkProtocol, COMM_PROTOCOL_HCCS);
    }
    EXPECT_EQ(listSize, 1);

    ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_2, 0, 2, &linkList, &listSize);
    EXPECT_EQ(listSize, 0);
    EXPECT_EQ(linkList, nullptr);
    DestroyComm(comm);
}

/**
A3超节点：
2个rank在1个超节点内，但不在1个server内
netLayer = 0， 无
netLayer = 1， HCCS
netLayer = 2， 无

2个rank分别在2个A3超节点
netLayer = 0， 无
netLayer = 1， 无
netLayer = 2， RDMA连接
*/
TEST_F(HcclIndependentOpRankGraphTest, Ut_HcclRankGraphGetLinks_When_In_diff_Servers_Same_Supod)
{
    CommLink* linkList = nullptr;
    uint32_t listSize = 0;
    set_chip_type_stub(0, static_cast<s32>(DevType::DEV_TYPE_910_93));
    Create91093Comm(&comm);
    HcclResult ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_1, 0, 4, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    for (int i = 0; i < listSize; i++) {
        EXPECT_EQ(linkList[i].srcEndpointDesc.commAddr.id, 0);
        EXPECT_EQ(linkList[i].dstEndpointDesc.commAddr.id, 4);
        EXPECT_EQ(linkList[i].srcEndpointDesc.protocol, COMM_PROTOCOL_HCCS);
        EXPECT_EQ(linkList[i].dstEndpointDesc.protocol, COMM_PROTOCOL_HCCS);
        EXPECT_EQ(linkList[i].linkAttr.linkProtocol, COMM_PROTOCOL_HCCS);
    }
    EXPECT_EQ(listSize, 1);
    // 从缓存里拿
    ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_1, 0, 4, &linkList, &listSize);
    EXPECT_EQ(listSize, 1);

    ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_0, 0, 4, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(listSize, 0);
    EXPECT_EQ(linkList, nullptr);

    ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_2, 0, 4, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(listSize, 0);
    EXPECT_EQ(linkList, nullptr);
    DestroyComm(comm);
}

TEST_F(HcclIndependentOpRankGraphTest, Ut_HcclRankGraphGetLinks_When_In_diff_Servers_diff_Supod)
{
    CommLink* linkList = nullptr;
    uint32_t listSize = 0;
    set_chip_type_stub(0, static_cast<s32>(DevType::DEV_TYPE_910_93));
    Create91093Comm(&comm);
    HcclResult ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_2, 0, 8, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    for (int i = 0; i < listSize; i++) {
        EXPECT_EQ(linkList[i].srcEndpointDesc.protocol, COMM_PROTOCOL_ROCE);
        EXPECT_EQ(linkList[i].dstEndpointDesc.protocol, COMM_PROTOCOL_ROCE);
        EXPECT_EQ(linkList[i].linkAttr.linkProtocol, COMM_PROTOCOL_ROCE);
    }
    EXPECT_EQ(listSize, 1);

    ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_1, 0, 8, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(listSize, 0);
    EXPECT_EQ(linkList, nullptr);

    ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_0, 0, 8, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(listSize, 0);
    EXPECT_EQ(linkList, nullptr);
    DestroyComm(comm);
}

HcclResult DevTypeToCommProtocol_stub_910B(RankGraph* graph, DevType& type, CommProtocol& protocol)
{
    type = DevType::DEV_TYPE_910B;
    protocol = COMM_PROTOCOL_ROCE;
    return HCCL_SUCCESS;
}

HcclResult hrtGetDeviceTypeStub(DevType& devType)
{
    devType = DevType::DEV_TYPE_910B;
    return HCCL_SUCCESS;
}

void Create910BComm(HcclComm* comm, json jsonName)
{
    MOCKER(hrtGetDeviceType).stubs().with(mockcpp::any()).will(invoke(hrtGetDeviceTypeStub));
    MOCKER_CPP(&RankGraph::DevTypeToCommProtocol)
        .stubs()
        .with(mockcpp::any())
        .will(invoke(DevTypeToCommProtocol_stub_910B));
    MOCKER(hrtGetDeviceInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(IsSuperPodMode).stubs().with(outBound(false)).will(returnValue(HCCL_SUCCESS));
    MOCKER(IsSupportAtomicWrite).stubs().will(returnValue(HCCL_SUCCESS));
    Ut_Clusterinfo_File_Create(RANKTABLE_FILE_NAME, jsonName);
    HcclResult ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclCommInitClusterInfo(RANKTABLE_FILE_NAME, 0, comm);
    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_NE(comm, nullptr);
}

// 910B A + K用例
/**
2个rank在server间：
netLayer = 0， 无
netLayer = 1， RDMA连接
netLayer = 2， 无
2个rank在1个server内：
netLayer = 0， HCCS连接
netLayer = 1， 无
netLayer = 2， 无
*/
TEST_F(HcclIndependentOpRankGraphTest, Ut_HcclRankGraphGetLinks_When_In_Same_Server_910B)
{
    CommLink* linkList = nullptr;
    uint32_t listSize = 0;
    set_chip_type_stub(0, static_cast<s32>(DevType::DEV_TYPE_910B));
    Create910BComm(&comm, rank_table_2server_2rank);
    HcclResult ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_0, 0, 1, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    for (int i = 0; i < listSize; i++) {
        EXPECT_EQ(linkList[i].srcEndpointDesc.commAddr.id, 0);
        EXPECT_EQ(linkList[i].dstEndpointDesc.commAddr.id, 1);
        EXPECT_EQ(linkList[i].srcEndpointDesc.protocol, COMM_PROTOCOL_HCCS);
        EXPECT_EQ(linkList[i].dstEndpointDesc.protocol, COMM_PROTOCOL_HCCS);
        EXPECT_EQ(linkList[i].linkAttr.linkProtocol, COMM_PROTOCOL_HCCS);
    }
    EXPECT_EQ(listSize, 1);

    ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_1, 0, 1, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(listSize, 0);
    EXPECT_EQ(linkList, nullptr);

    ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_2, 0, 1, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(listSize, 0);
    EXPECT_EQ(linkList, nullptr);
    DestroyComm(comm);
}

TEST_F(HcclIndependentOpRankGraphTest, Ut_HcclRankGraphGetLinks_When_In_Diff_Servers_910B)
{
    CommLink* linkList = nullptr;
    uint32_t listSize = 0;
    set_chip_type_stub(0, static_cast<s32>(DevType::DEV_TYPE_910B));
    Create910BComm(&comm, rank_table_2server_2rank);
    HcclResult ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_1, 0, 3, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    for (int i = 0; i < listSize; i++) {
        EXPECT_EQ(linkList[i].srcEndpointDesc.protocol, COMM_PROTOCOL_ROCE);
        EXPECT_EQ(linkList[i].dstEndpointDesc.protocol, COMM_PROTOCOL_ROCE);
        EXPECT_EQ(linkList[i].linkAttr.linkProtocol, COMM_PROTOCOL_ROCE);
    }
    EXPECT_EQ(listSize, 1);

    ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_0, 0, 3, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(listSize, 0);
    EXPECT_EQ(linkList, nullptr);

    ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_2, 0, 3, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(listSize, 0);
    EXPECT_EQ(linkList, nullptr);
    DestroyComm(comm);
}

// 910B A + X用例，同个MESH间走HCCS
/**
2个rank在server间：
netLayer = 0， 无
netLayer = 1， RDMA连接
netLayer = 2， 无

2个rank在1个server内，不在1个mesh：
netLayer = 0， PCIE连接
netLayer = 1， 无（有接交换机，就有rdma连接）
netLayer = 2， 无

2个rank在1个server内， 同个mesh：
netLayer = 0， HCCS连接
netLayer = 1， 无（有接交换机，就有rdma连接）
netLayer = 2， 无
*/

TEST_F(HcclIndependentOpRankGraphTest, Ut_HcclRankGraphGetLinks_When_In_Same_Mesh_AX)
{
    CommLink* linkList = nullptr;
    uint32_t listSize = 0;
    set_chip_type_stub(0, static_cast<s32>(DevType::DEV_TYPE_910B));
    MOCKER(&TopoInfoExtractor::CheckPlaneInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    Create910BComm(&comm, rank_table_910B_1server_16rank);
    HcclResult ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_0, 0, 3, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    for (int i = 0; i < listSize; i++) {
        EXPECT_EQ(linkList[i].srcEndpointDesc.protocol, COMM_PROTOCOL_HCCS);
        EXPECT_EQ(linkList[i].dstEndpointDesc.protocol, COMM_PROTOCOL_HCCS);
        EXPECT_EQ(linkList[i].linkAttr.linkProtocol, COMM_PROTOCOL_HCCS);
    }
    EXPECT_EQ(listSize, 1);

    ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_1, 0, 3, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(listSize, 0);
    EXPECT_EQ(linkList, nullptr);

    ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_2, 0, 3, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(listSize, 0);
    EXPECT_EQ(linkList, nullptr);
    DestroyComm(comm);
}

// 不同个MESH间对称场景走PCIE
TEST_F(HcclIndependentOpRankGraphTest, Ut_HcclRankGraphGetLinks_When_In_diff_Mesh_AX)
{
    CommLink* linkList = nullptr;
    uint32_t listSize = 0;
    set_chip_type_stub(0, static_cast<s32>(DevType::DEV_TYPE_910B));
    MOCKER(&TopoInfoExtractor::CheckPlaneInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    Create910BComm(&comm, rank_table_910B_1server_16rank);
    HcclResult ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_0, 0, 8, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    for (int i = 0; i < listSize; i++) {
        EXPECT_EQ(linkList[i].srcEndpointDesc.protocol, COMM_PROTOCOL_PCIE);
        EXPECT_EQ(linkList[i].dstEndpointDesc.protocol, COMM_PROTOCOL_PCIE);
        EXPECT_EQ(linkList[i].linkAttr.linkProtocol, COMM_PROTOCOL_PCIE);
    }
    EXPECT_EQ(listSize, 1);

    ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_1, 0, 8, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(listSize, 0);
    EXPECT_EQ(linkList, nullptr);

    ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_2, 0, 8, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(listSize, 0);
    EXPECT_EQ(linkList, nullptr);
    DestroyComm(comm);
}

TEST_F(HcclIndependentOpRankGraphTest, Ut_HcclRankGraphGetLinks_When_In_diff_Mesh_AX_Roce)
{
    setenv("HCCL_INTRA_ROCE_ENABLE", "1", 1);
    setenv("HCCL_INTRA_PCIE_ENABLE", "0", 1);
    CommLink* linkList = nullptr;
    uint32_t listSize = 0;
    set_chip_type_stub(0, static_cast<s32>(DevType::DEV_TYPE_910B));
    MOCKER(&TopoInfoExtractor::CheckPlaneInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    Create910BComm(&comm, rank_table_910B_1server_16rank);
    HcclResult ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_0, 0, 8, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    for (int i = 0; i < listSize; i++) {
        EXPECT_EQ(linkList[i].srcEndpointDesc.protocol, COMM_PROTOCOL_PCIE);
        EXPECT_EQ(linkList[i].dstEndpointDesc.protocol, COMM_PROTOCOL_PCIE);
        EXPECT_EQ(linkList[i].linkAttr.linkProtocol, COMM_PROTOCOL_PCIE);
    }
    EXPECT_EQ(listSize, 1);

    ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_1, 0, 8, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    for (int i = 0; i < listSize; i++) {
        EXPECT_EQ(linkList[i].srcEndpointDesc.protocol, COMM_PROTOCOL_ROCE);
        EXPECT_EQ(linkList[i].dstEndpointDesc.protocol, COMM_PROTOCOL_ROCE);
        EXPECT_EQ(linkList[i].linkAttr.linkProtocol, COMM_PROTOCOL_ROCE);
    }
    EXPECT_EQ(listSize, 1);
    DestroyComm(comm);
}

// 不同个MESH间非对称场景不存在链路
TEST_F(HcclIndependentOpRankGraphTest, Ut_HcclRankGraphGetLinks_When_In_diff_Mesh_AX_Unsym)
{
    CommLink* linkList = nullptr;
    uint32_t listSize = 0;
    set_chip_type_stub(0, static_cast<s32>(DevType::DEV_TYPE_910B));
    MOCKER(&TopoInfoExtractor::CheckPlaneInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    Create910BComm(&comm, rank_table_910B_1server_16rank);
    HcclResult ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_0, 0, 9, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(listSize, 0);
    EXPECT_EQ(linkList, nullptr);

    ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_1, 0, 9, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(listSize, 0);
    EXPECT_EQ(linkList, nullptr);

    ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_2, 0, 9, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(listSize, 0);
    EXPECT_EQ(linkList, nullptr);
    DestroyComm(comm);
}

// 不同个MESH间非对称场景不存在链路，除非使能ROCE
TEST_F(HcclIndependentOpRankGraphTest, Ut_HcclRankGraphGetLinks_When_In_diff_Mesh_AX_Unsym_Roce)
{
    setenv("HCCL_INTRA_ROCE_ENABLE", "1", 1);
    setenv("HCCL_INTRA_PCIE_ENABLE", "0", 1);
    CommLink* linkList = nullptr;
    uint32_t listSize = 0;
    set_chip_type_stub(0, static_cast<s32>(DevType::DEV_TYPE_910B));
    MOCKER(&TopoInfoExtractor::CheckPlaneInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    Create910BComm(&comm, rank_table_910B_1server_16rank);
    HcclResult ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_0, 0, 9, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(listSize, 0);
    EXPECT_EQ(linkList, nullptr);

    ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_1, 0, 8, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    for (int i = 0; i < listSize; i++) {
        EXPECT_EQ(linkList[i].srcEndpointDesc.protocol, COMM_PROTOCOL_ROCE);
        EXPECT_EQ(linkList[i].dstEndpointDesc.protocol, COMM_PROTOCOL_ROCE);
        EXPECT_EQ(linkList[i].linkAttr.linkProtocol, COMM_PROTOCOL_ROCE);
    }
    EXPECT_EQ(listSize, 1);

    ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_2, 0, 9, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(listSize, 0);
    EXPECT_EQ(linkList, nullptr);
    DestroyComm(comm);
}

// 310P用例
HcclResult DevTypeToCommProtocol_stub_310P(RankGraph* graph, DevType& type, CommProtocol& protocol)
{
    type = DevType::DEV_TYPE_310P3;
    protocol = COMM_PROTOCOL_ROCE;
    return HCCL_SUCCESS;
}

void Create310PComm(HcclComm* comm, json jsonName)
{
    MOCKER_CPP(&RankGraph::DevTypeToCommProtocol)
        .stubs()
        .with(mockcpp::any())
        .will(invoke(DevTypeToCommProtocol_stub_310P));
    MOCKER(hrtGetDeviceInfo).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER(IsSuperPodMode).stubs().with(outBound(false)).will(returnValue(HCCL_SUCCESS));
    MOCKER(IsSupportAtomicWrite).stubs().will(returnValue(HCCL_SUCCESS));
    Ut_Clusterinfo_File_Create(RANKTABLE_FILE_NAME, jsonName);
    HcclResult ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclCommInitClusterInfo(RANKTABLE_FILE_NAME, 0, comm);
    ASSERT_EQ(ret, HCCL_SUCCESS);
    ASSERT_NE(comm, nullptr);
}

// 310P
/**
310P：
2个rank在server间：
netLayer = 0， 无
netLayer = 1， PCIE连接 （V卡HCCS连接，但无法识别）
netLayer = 2， 无

2个rank在个server内：
netLayer = 0， V卡PCIE连接 DUO卡DIE间走HCCS
netLayer = 1， 无
netLayer = 2， 无
**/
TEST_F(HcclIndependentOpRankGraphTest, Ut_HcclRankGraphGetLinks_When_In_Same_Server_310P)
{
    CommLink* linkList = nullptr;
    uint32_t listSize = 0;
    set_chip_type_stub(0, static_cast<s32>(DevType::DEV_TYPE_310P3));
    Create310PComm(&comm, rank_table_2server_4rank);
    HcclResult ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_0, 0, 1, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    for (int i = 0; i < listSize; i++) {
        EXPECT_EQ(linkList[i].srcEndpointDesc.commAddr.id, 0);
        EXPECT_EQ(linkList[i].dstEndpointDesc.commAddr.id, 1);
        EXPECT_EQ(linkList[i].srcEndpointDesc.protocol, COMM_PROTOCOL_HCCS);
        EXPECT_EQ(linkList[i].dstEndpointDesc.protocol, COMM_PROTOCOL_HCCS);
        EXPECT_EQ(linkList[i].linkAttr.linkProtocol, COMM_PROTOCOL_HCCS);
    }
    EXPECT_EQ(listSize, 1);

    ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_1, 0, 1, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(listSize, 0);
    EXPECT_EQ(linkList, nullptr);

    ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_2, 0, 1, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(listSize, 0);
    EXPECT_EQ(linkList, nullptr);
    // DUO卡两个卡间其实走PCIE
    ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_0, 0, 2, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(listSize, 1);
    DestroyComm(comm);
}

TEST_F(HcclIndependentOpRankGraphTest, Ut_HcclRankGraphGetLinks_When_In_Diff_Servers_310P)
{
    CommLink* linkList = nullptr;
    uint32_t listSize = 0;
    set_chip_type_stub(0, static_cast<s32>(DevType::DEV_TYPE_310P3));
    Create310PComm(&comm, rank_table_2server_4rank);
    HcclResult ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_1, 0, 4, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    for (int i = 0; i < listSize; i++) {
        EXPECT_EQ(linkList[i].srcEndpointDesc.protocol, COMM_PROTOCOL_PCIE);
        EXPECT_EQ(linkList[i].dstEndpointDesc.protocol, COMM_PROTOCOL_PCIE);
        EXPECT_EQ(linkList[i].linkAttr.linkProtocol, COMM_PROTOCOL_PCIE);
    }
    EXPECT_EQ(listSize, 1);

    ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_0, 0, 4, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(listSize, 0);
    EXPECT_EQ(linkList, nullptr);

    ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_2, 0, 4, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(listSize, 0);
    EXPECT_EQ(linkList, nullptr);
    DestroyComm(comm);
}

// 310P V卡走PCIE
TEST_F(HcclIndependentOpRankGraphTest, Ut_HcclRankGraphGetLinks_When_In_Same_Servers_310P_V)
{
    setenv("HCCL_INTRA_ROCE_ENABLE", "1", 1);
    setenv("HCCL_INTRA_PCIE_ENABLE", "0", 1);
    CommLink* linkList = nullptr;
    uint32_t listSize = 0;
    set_chip_type_stub(0, static_cast<s32>(DevType::DEV_TYPE_310P3));
    set_board_id(0x1E); // 设置为V卡(310P标卡场景)
    Create310PComm(&comm, rank_table_2server_4rank);
    HcclResult ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_0, 0, 1, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    for (int i = 0; i < listSize; i++) {
        EXPECT_EQ(linkList[i].srcEndpointDesc.protocol, COMM_PROTOCOL_PCIE);
        EXPECT_EQ(linkList[i].dstEndpointDesc.protocol, COMM_PROTOCOL_PCIE);
        EXPECT_EQ(linkList[i].linkAttr.linkProtocol, COMM_PROTOCOL_PCIE);
    }
    EXPECT_EQ(listSize, 1);

    ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_1, 0, 1, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    for (int i = 0; i < listSize; i++) {
        EXPECT_EQ(linkList[i].srcEndpointDesc.protocol, COMM_PROTOCOL_ROCE);
        EXPECT_EQ(linkList[i].dstEndpointDesc.protocol, COMM_PROTOCOL_ROCE);
        EXPECT_EQ(linkList[i].linkAttr.linkProtocol, COMM_PROTOCOL_ROCE);
    }
    EXPECT_EQ(listSize, 1);

    ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_2, 0, 1, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(listSize, 0);
    EXPECT_EQ(linkList, nullptr);

    ret = HcclRankGraphGetLinks(comm, HCCL_NETLAYER_1, 0, 4, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    for (int i = 0; i < listSize; i++) {
        EXPECT_EQ(linkList[i].srcEndpointDesc.protocol, COMM_PROTOCOL_PCIE);
        EXPECT_EQ(linkList[i].dstEndpointDesc.protocol, COMM_PROTOCOL_PCIE);
        EXPECT_EQ(linkList[i].linkAttr.linkProtocol, COMM_PROTOCOL_PCIE);
    }
    EXPECT_EQ(listSize, 1);
    DestroyComm(comm);
}

// ==================== 新增拓扑查询函数 UT 测试用例 ====================

// GetTopoInstsByLayer: 参数校验 - netLayer无效
TEST_F(HcclIndependentOpRankGraphTest, Ut_GetTopoInstsByLayer_When_InvalidNetLayer_Expect_ParaError)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 获取netLayer信息
    uint32_t netLayerNum = 0;
    uint32_t* netLayers = nullptr;
    ret = HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 使用一个无效的netLayer（大于等于netLayerNum）
    uint32_t invalidNetLayer = netLayerNum + 1;
    uint32_t* topoInsts = nullptr;
    uint32_t topoInstNum = 0;

    // 通过HcclRankGraphGetTopoInstsByLayer接口测试
    ret = HcclRankGraphGetTopoInstsByLayer(comm, invalidNetLayer, &topoInsts, &topoInstNum);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// GetTopoType: 参数校验 - netLayer无效
TEST_F(HcclIndependentOpRankGraphTest, Ut_GetTopoType_When_InvalidNetLayer_Expect_ParaError)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 使用一个无效的netLayer
    uint32_t invalidNetLayer = 100;
    uint32_t topoInstId = 0;
    CommTopo topoType;

    ret = HcclRankGraphGetTopoType(comm, invalidNetLayer, topoInstId, &topoType);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// GetRanksByTopoInst: 非910B设备类型返回NOT_SUPPORT
TEST_F(HcclIndependentOpRankGraphTest, Ut_GetRanksByTopoInst_When_Not910B_Expect_NotSupport)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 910_93设备类型，非910B
    uint32_t netLayer = 0;
    uint32_t topoInstId = 0;
    uint32_t* ranks = nullptr;
    uint32_t rankNum = 0;

    ret = HcclRankGraphGetRanksByTopoInst(comm, netLayer, topoInstId, &ranks, &rankNum);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

// GetRanksByTopoInst: 参数校验 - netLayer无效
TEST_F(HcclIndependentOpRankGraphTest, Ut_GetRanksByTopoInst_When_InvalidNetLayer_Expect_ParaError)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 设置为910B设备类型
    set_chip_type_stub(0, static_cast<s32>(DevType::DEV_TYPE_910B));

    // 使用无效的netLayer
    uint32_t invalidNetLayer = 100;
    uint32_t topoInstId = 0;
    uint32_t* ranks = nullptr;
    uint32_t rankNum = 0;

    ret = HcclRankGraphGetRanksByTopoInst(comm, invalidNetLayer, topoInstId, &ranks, &rankNum);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// GetRanksByTopoInst: 910B设备类型，有效参数
TEST_F(HcclIndependentOpRankGraphTest, Ut_GetRanksByTopoInst_When_910BValidParams_Expect_Success)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 设置为910B设备类型
    set_chip_type_stub(0, static_cast<s32>(DevType::DEV_TYPE_910B));

    uint32_t netLayer = 0;
    uint32_t topoInstId = 0;
    uint32_t* ranks = nullptr;
    uint32_t rankNum = 0;

    ret = HcclRankGraphGetRanksByTopoInst(comm, netLayer, topoInstId, &ranks, &rankNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// GetEndpointNum: 参数校验 - netLayer无效
TEST_F(HcclIndependentOpRankGraphTest, Ut_GetEndpointNum_When_InvalidNetLayer_Expect_ParaError)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    uint32_t invalidNetLayer = 100;
    uint32_t topoInstId = 0;
    uint32_t num = 0;

    ret = HcclRankGraphGetEndpointNum(comm, invalidNetLayer, topoInstId, &num);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// GetEndpointDesc: 参数校验 - nullptr参数
TEST_F(HcclIndependentOpRankGraphTest, Ut_GetEndpointDesc_When_NullPtr_Expect_ParaError)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    uint32_t netLayer = 0;
    uint32_t topoInstId = 0;
    uint32_t descNum = 0;

    // descNum为nullptr
    ret = HcclRankGraphGetEndpointDesc(comm, netLayer, topoInstId, nullptr, nullptr);
    EXPECT_EQ(ret, HCCL_E_PARA);

    // endpointDesc为nullptr
    ret = HcclRankGraphGetEndpointDesc(comm, netLayer, topoInstId, &descNum, nullptr);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// GetEndpointDesc: 参数校验 - netLayer无效
TEST_F(HcclIndependentOpRankGraphTest, Ut_GetEndpointDesc_When_InvalidNetLayer_Expect_ParaError)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    uint32_t invalidNetLayer = 100;
    uint32_t topoInstId = 0;
    uint32_t descNum = 10;
    EndpointDesc endpointDesc[10];

    ret = HcclRankGraphGetEndpointDesc(comm, invalidNetLayer, topoInstId, &descNum, endpointDesc);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// GetDevicePort: rank不存在时返回错误
TEST_F(HcclIndependentOpRankGraphTest, Ut_GetDevicePort_When_RankNotExist_Expect_ParaError)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 使用一个不存在的rank id（9999）
    uint32_t invalidRank = 9999;
    uint32_t devPort = 0;

    // GetDevicePort 是 RankGraphV1 的公开方法，直接通过 rankGraphV2 调用
    ret = rankGraphV2->GetDevicePort(invalidRank, &devPort);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// GetDevicePort: 正常获取
TEST_F(HcclIndependentOpRankGraphTest, Ut_GetDevicePort_When_ValidRank_Expect_Success)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 使用存在的rank id（0）
    uint32_t rank = 0;
    uint32_t devPort = 0;

    // GetDevicePort 是 RankGraphV1 的公开方法，直接通过 rankGraphV2 调用
    ret = rankGraphV2->GetDevicePort(rank, &devPort);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// GetTopoInstsByLayer: 有效参数
TEST_F(HcclIndependentOpRankGraphTest, Ut_GetTopoInstsByLayer_When_ValidParams_Expect_Success)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    uint32_t netLayer = 0;
    uint32_t* topoInsts = nullptr;
    uint32_t topoInstNum = 0;

    ret = HcclRankGraphGetTopoInstsByLayer(comm, netLayer, &topoInsts, &topoInstNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_GE(topoInstNum, 0u);
}

// GetTopoType: 有效参数
TEST_F(HcclIndependentOpRankGraphTest, Ut_GetTopoType_When_ValidParams_Expect_Success)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    uint32_t netLayer = 0;
    uint32_t topoInstId = 0;
    CommTopo topoType;

    ret = HcclRankGraphGetTopoType(comm, netLayer, topoInstId, &topoType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// GetEndpointNum: 有效参数
TEST_F(HcclIndependentOpRankGraphTest, Ut_GetEndpointNum_When_ValidParams_Expect_Success)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    uint32_t netLayer = 0;
    uint32_t topoInstId = 0;
    uint32_t num = 0;

    ret = HcclRankGraphGetEndpointNum(comm, netLayer, topoInstId, &num);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// GetEndpointDesc: 有效参数
TEST_F(HcclIndependentOpRankGraphTest, Ut_GetEndpointDesc_When_ValidParams_Expect_Success)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    uint32_t netLayer = 0;
    uint32_t topoInstId = 0;
    uint32_t descNum = 10;
    EndpointDesc endpointDesc[10];

    ret = HcclRankGraphGetEndpointDesc(comm, netLayer, topoInstId, &descNum, endpointDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

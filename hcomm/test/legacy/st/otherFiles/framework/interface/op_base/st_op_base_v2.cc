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
#include <mockcpp/MockObject.h>
#include <algorithm>
#include <future>
#include <map>
#include <fstream>
#include <string>
#include <nlohmann/json.hpp>
#define private public
#define protected public
#include "root_handle_v2.h"
#include "rank_info_detect.h"
#include "hccl_params_pub.h"
#include "hccl_common_v2.h"
#include "param_check_v2.h"
#include "comm_manager.h"
#include "binary_stream.h"
#include "snap_shot_parse.h"
#include "op_base_v2.h"
#include "orion_adapter_rts.h"
#include "communicator_callback.h"
#include "task_abort_handler_v2.h"
#include "hccl_communicator.h"
#include "communicator_impl.h"
#include "hccl.h"
#undef private
#undef protected

using namespace Hccl;
using namespace std;

class OpbaseTestV2 : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "OpbaseTestV2 tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "OpbaseTestV2 tests tear down." << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in OpbaseTestV2 SetUP" << std::endl; }

    virtual void TearDown()
    {
        std::cout << "A Test case in OpbaseTestV2 TearDown" << std::endl;
        GlobalMockObject::verify();
    }
};

const std::string rankTable_ut_stub_4p = R"(
    {
        "version": "2.0",
        "rank_count" : "4",
        "rank_list": [
            {
                "rank_id": 0,
                "local_id": 0,
                "level_list":  [
                    {
                        "level": 0,
                        "id" : "az0-rack0", 
                        "fabric_type": "INNER", 
                        "rank_addr_type": "",
                        "rank_addrs": []
                    }
                ]
            },
            {
                "rank_id": 1,
                "local_id": 1,
                "level_list":  [
                    {
                        "level": 0,
                        "id" : "az0-rack0", 
                        "fabric_type": "INNER", 
                        "rank_addr_type": "",
                        "rank_addrs": []
                    }
                ]
            },
            {
                "rank_id": 2,
                "local_id": 2,
                "level_list":  [
                    {
                        "level": 0,
                        "id" : "az0-rack0", 
                        "fabric_type": "INNER", 
                        "rank_addr_type": "",
                        "rank_addrs": []
                    }
                ]
            },
            {
                "rank_id": 3,
                "local_id": 3,
                "level_list":  [
                    {
                        "level": 0,
                        "id" : "az0-rack0", 
                        "fabric_type": "INNER", 
                        "rank_addr_type": "",
                        "rank_addrs": []
                    }
                ]
            }
        ],
 
        "replace_count" : 1,
        "replace_list" : [ 
            {"level": 0,  "group_id" : "az0-rack0", "backup_local_id": 64, "target_local_id": 1}
        ]
    }
)";

// ranktable 910 8p
static nlohmann::json rank_table_950_1server_8rank = nlohmann::json::object(
    {{"version", "2.0"},
     {"rank_count", "4"},
     {"rank_list", nlohmann::json::array(
                       {nlohmann::json::object(
                            {{"rank_id", 0},
                             {"local_id", 0},
                             {"level_list", nlohmann::json::array({nlohmann::json::object(
                                                {{"level", 0},
                                                 {"id", "az0-rack0"},
                                                 {"fabric_type", "INNER"},
                                                 {"rank_addr_type", ""},
                                                 {"rank_addrs", nlohmann::json::array()}})})}}),
                        nlohmann::json::object(
                            {{"rank_id", 1},
                             {"local_id", 1},
                             {"level_list", nlohmann::json::array({nlohmann::json::object(
                                                {{"level", 0},
                                                 {"id", "az0-rack0"},
                                                 {"fabric_type", "INNER"},
                                                 {"rank_addr_type", ""},
                                                 {"rank_addrs", nlohmann::json::array()}})})}}),
                        nlohmann::json::object(
                            {{"rank_id", 2},
                             {"local_id", 2},
                             {"level_list", nlohmann::json::array({nlohmann::json::object(
                                                {{"level", 0},
                                                 {"id", "az0-rack0"},
                                                 {"fabric_type", "INNER"},
                                                 {"rank_addr_type", ""},
                                                 {"rank_addrs", nlohmann::json::array()}})})}}),
                        nlohmann::json::object(
                            {{"rank_id", 3},
                             {"local_id", 3},
                             {"level_list", nlohmann::json::array({nlohmann::json::object(
                                                {{"level", 0},
                                                 {"id", "az0-rack0"},
                                                 {"fabric_type", "INNER"},
                                                 {"rank_addr_type", ""},
                                                 {"rank_addrs", nlohmann::json::array()}})})}})})}});

TEST_F(OpbaseTestV2, HcclBatchSendRecvV2)
{
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(communicator.get());
    int a = 0;
    aclrtStream stream = static_cast<aclrtStream>(&a);

    u32 itemNum = 10;
    unique_ptr<HcclSendRecvItem> sendRecvInfo = make_unique<HcclSendRecvItem>();

    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcclBatchSendRecvV2(sendRecvInfo.get(), itemNum, comm, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclAlltoAllV2)
{
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(communicator.get());
    void* sendBuf = nullptr;
    uint64_t sendCount = 10;
    HcclDataType sendType = HCCL_DATA_TYPE_INT32;
    void* recvBuf = nullptr;
    uint64_t recvCount = 10;
    HcclDataType recvType = HCCL_DATA_TYPE_INT32;
    int a = 0;
    aclrtStream stream = static_cast<aclrtStream>(&a);

    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcclAlltoAllV2(sendBuf, sendCount, sendType, recvBuf, recvCount, recvType, comm, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclAlltoAllVV2)
{
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(communicator.get());
    void* sendBuf = nullptr;
    void* sendCounts = (void*)0x1000000;
    HcclDataType sendType = HCCL_DATA_TYPE_INT32;
    void* recvBuf = nullptr;
    void* recvCounts = (void*)0x1000001;
    void* sdispls = nullptr;
    void* rdispls = nullptr;

    HcclDataType recvType = HCCL_DATA_TYPE_INT32;
    int a = 0;
    aclrtStream stream = static_cast<aclrtStream>(&a);

    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result
        = HcclAlltoAllVV2(sendBuf, sendCounts, sdispls, sendType, recvBuf, recvCounts, rdispls, recvType, comm, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclCommInitClusterInfoV2_1)
{
    string clusterInfo = "clusterInfo";
    uint32_t rank = 0;
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(communicator.get());
    MOCKER_CPP(&HcclCommunicator::Init, HcclResult(HcclCommunicator::*)(const std::string&))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap.clear();
    CommManager::GetInstance(0).GetCommInfoV2().pComm = nullptr;
    HcclResult ret = HcclCommInitClusterInfoV2(clusterInfo.c_str(), rank, &comm);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(OpbaseTestV2, HcclCommInitClusterInfoV2)
{
    HcclComm comm;
    // 打桩GetCommInfoV2。
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap.clear();
    CommManager::GetInstance(0).GetCommInfoV2().pComm = nullptr;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));

    nlohmann::json rank_table = rank_table_950_1server_8rank;
    char file_name_t[] = "./st_hcom_test_rank_table_1server_8rank_950.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();
    s32 deviceId = 0;
    char* identify = "0";
    s32 rankSize = 1;
    s32 rank = atoi(identify);
    DevType devType = DevType::DEV_TYPE_950;

    char* clusterInfo = "./st_hcom_test_rank_table_1server_8rank_950.json";
    MOCKER(HrtGetDeviceType).stubs().with(mockcpp::any()).will(returnValue(devType));
    MOCKER_CPP(&HcclCommunicator::Init, HcclResult(HcclCommunicator::*)(const std::string&))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());
    auto ret = HcclCommInitClusterInfoV2(clusterInfo, rank, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

void PrepareCommConfig(
    HcclCommConfig& config, uint32_t hcclBufferSize = 200, string worldgroup = "hccl_world_group",
    uint32_t hcclDeterministic = 1, uint32_t hcclOpExpansionMode = 0)
{
    config.hcclBufferSize = hcclBufferSize;
    std::strncpy(config.hcclCommName, worldgroup.c_str(), worldgroup.size() + 1);
    std::strncpy(config.reserved, worldgroup.c_str(), worldgroup.size() + 1);
    config.hcclDeterministic = hcclDeterministic;
    config.hcclOpExpansionMode = hcclOpExpansionMode;
    std::strncpy(config.hcclUdi, worldgroup.c_str(), worldgroup.size() + 1);
}

TEST_F(OpbaseTestV2, HcclCommInitClusterInfoConfigV2)
{
    nlohmann::json rank_table = rank_table_950_1server_8rank;
    char file_name_t[] = "./st_hcom_test_rank_table_1server_8rank_950.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();
    s32 deviceId = 0;
    char* identify = "0";
    s32 rankSize = 1;
    s32 rank = atoi(identify);

    char* clusterInfo = "./st_hcom_test_rank_table_1server_8rank_950.json";

    HcclCommConfig config;
    string worldgroup = "hccl_world_group";
    PrepareCommConfig(config, 200, worldgroup, 1, 0);
    HcclComm comm;

    // 打桩GetCommInfoV2。
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap.clear();
    CommManager::GetInstance(0).GetCommInfoV2().pComm = nullptr;

    MOCKER_CPP(&HcclCommunicator::Init, HcclResult(HcclCommunicator::*)(const std::string&))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());
    auto ret = HcclCommInitClusterInfoConfigV2(clusterInfo, rank, &config, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclCommInitClusterInfoConfigV2_CONFIGNOTSET)
{
    nlohmann::json rank_table = rank_table_950_1server_8rank;
    char file_name_t[] = "./st_hcom_test_rank_table_1server_8rank_950.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();
    s32 deviceId = 0;
    char* identify = "0";
    s32 rankSize = 1;
    s32 rank = atoi(identify);

    char* clusterInfo = "./st_hcom_test_rank_table_1server_8rank_950.json";

    HcclCommConfig config;
    string worldgroup = "hccl_world_group";
    PrepareCommConfig(config, 0xffffffff, worldgroup, 1, 0);
    HcclComm comm;

    // 打桩GetCommInfoV2。
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap.clear();
    CommManager::GetInstance(0).GetCommInfoV2().pComm = nullptr;

    MOCKER_CPP(&HcclCommunicator::Init, HcclResult(HcclCommunicator::*)(const std::string&))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());
    auto ret = HcclCommInitClusterInfoConfigV2(clusterInfo, rank, &config, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclGetRankIdV2)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    uint32_t rank = 0;
    MOCKER_CPP(&HcclCommunicator::GetRankId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HcclGetRankIdV2(comm, &rank);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclGetCommNameV2)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    char* commName = new char[100];
    HcclResult ret = HcclGetCommNameV2(comm, commName);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    delete[] commName;
}

TEST_F(OpbaseTestV2, HcclGetRankSizeV2)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    uint32_t rankSize;
    MOCKER_CPP(&HcclCommunicator::GetRankSize).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HcclGetRankSizeV2(comm, &rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclAlltoAllVCV2)
{
    // Prepare test data
    void* sendBuf = nullptr;
    void* sendCountMatrix = (void*)0x1000000;
    void* recvBuf = nullptr;
    HcclDataType sendType = HCCL_DATA_TYPE_INT8;
    HcclDataType recvType = HCCL_DATA_TYPE_INT8;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    int a = 0;
    rtStream_t stream = static_cast<rtStream_t>(&a);

    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcclAlltoAllVCV2(sendBuf, sendCountMatrix, sendType, recvBuf, recvType, comm, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclReduceV2_Sum_ShouldPass_WhenValidParams)
{
    // Mock objects and parameters
    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    uint64_t count = 10;
    HcclDataType dataType = HCCL_DATA_TYPE_INT8;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    uint32_t root = 0;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclComm->pimpl = std::make_unique<CommunicatorImpl>();
    hcclComm->pimpl->rankSize = 4;
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    aclrtStream stream = &count;
    DevType devType = DevType::DEV_TYPE_950;

    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcclReduceV2(sendBuf, recvBuf, count, dataType, op, root, comm, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclReduceV2_PROD_ShouldFail_WhenValidParams)
{
    // Mock objects and parameters
    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    uint64_t count = 10;
    HcclDataType dataType = HCCL_DATA_TYPE_INT8;
    HcclReduceOp op = HCCL_REDUCE_PROD;
    uint32_t root = 0;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    aclrtStream stream = &count;
    DevType devType = DevType::DEV_TYPE_950;

    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcclReduceV2(sendBuf, recvBuf, count, dataType, op, root, comm, stream);
    EXPECT_EQ(result, HCCL_E_NOT_SUPPORT);
}

TEST_F(OpbaseTestV2, HcclReduceV2_MAX_ShouldPass_WhenValidParams)
{
    // Mock objects and parameters
    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    uint64_t count = 10;
    HcclDataType dataType = HCCL_DATA_TYPE_INT8;
    HcclReduceOp op = HCCL_REDUCE_MAX;
    uint32_t root = 0;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclComm->pimpl = std::make_unique<CommunicatorImpl>();
    hcclComm->pimpl->rankSize = 4;
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    aclrtStream stream = &count;
    DevType devType = DevType::DEV_TYPE_950;

    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcclReduceV2(sendBuf, recvBuf, count, dataType, op, root, comm, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclReduceV2_MIN_ShouldPass_WhenValidParams)
{
    // Mock objects and parameters
    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    uint64_t count = 10;
    HcclDataType dataType = HCCL_DATA_TYPE_INT8;
    HcclReduceOp op = HCCL_REDUCE_MIN;
    uint32_t root = 0;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclComm->pimpl = std::make_unique<CommunicatorImpl>();
    hcclComm->pimpl->rankSize = 4;
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    aclrtStream stream = &count;
    DevType devType = DevType::DEV_TYPE_950;

    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcclReduceV2(sendBuf, recvBuf, count, dataType, op, root, comm, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclAllReduceV2_Sum_ShouldPass_WhenValidParams_v2)
{
    // Mock objects and parameters
    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    uint64_t count = 10;
    HcclDataType dataType = HCCL_DATA_TYPE_INT8;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    aclrtStream stream = &count;
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcclAllReduceV2(sendBuf, recvBuf, count, dataType, op, comm, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclAllReduceV2_PROD_ShouldFail_WhenValidParams_v2)
{
    // Mock objects and parameters
    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    uint64_t count = 10;
    HcclDataType dataType = HCCL_DATA_TYPE_INT8;
    HcclReduceOp op = HCCL_REDUCE_PROD;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    aclrtStream stream = &count;
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcclAllReduceV2(sendBuf, recvBuf, count, dataType, op, comm, stream);
    EXPECT_EQ(result, HCCL_E_NOT_SUPPORT);
}

TEST_F(OpbaseTestV2, HcclAllReduceV2_MAX_ShouldPass_WhenValidParams_v2)
{
    // Mock objects and parameters
    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    uint64_t count = 10;
    HcclDataType dataType = HCCL_DATA_TYPE_INT8;
    HcclReduceOp op = HCCL_REDUCE_MAX;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    aclrtStream stream = &count;
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcclAllReduceV2(sendBuf, recvBuf, count, dataType, op, comm, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclAllReduceV2_MIN_ShouldPass_WhenValidParams_v2)
{
    // Mock objects and parameters
    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    uint64_t count = 10;
    HcclDataType dataType = HCCL_DATA_TYPE_INT8;
    HcclReduceOp op = HCCL_REDUCE_MIN;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    aclrtStream stream = &count;
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcclAllReduceV2(sendBuf, recvBuf, count, dataType, op, comm, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclBroadcastV2_ShouldReturnSuccess_WhenAllParamsValid)
{
    void* buf = nullptr;
    uint64_t count = 10;
    HcclDataType dataType = HCCL_DATA_TYPE_INT32;
    uint32_t root = 0;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclComm->pimpl = std::make_unique<CommunicatorImpl>();
    hcclComm->pimpl->rankSize = 4;
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    aclrtStream stream = &count;
    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcclBroadcastV2(buf, count, dataType, root, comm, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclAllocComResourceByTilingV2)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    int dd = 0;
    void* stream = static_cast<void*>(&dd);
    void* mc2Tiling = static_cast<void*>(&dd);
    void* commContext = static_cast<void*>(&dd);
    MOCKER_CPP(&HcclCommunicator::AllocCommResource)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HcclAllocComResourceByTilingV2(comm, stream, mc2Tiling, &commContext);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclCommResumeV2)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));
    HcclResult ret = HcclCommResumeV2(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclCommOperationImplV2)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    HcclResult ret = HcclCommResumeV2(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclScatterV2)
{
    // Arrange
    void* sendBuf = (void*)0x1000000;
    void* recvBuf = nullptr;
    uint64_t recvCount = 10;
    HcclDataType dataType = HCCL_DATA_TYPE_INT8;
    uint32_t root = 0;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclComm->pimpl = std::make_unique<CommunicatorImpl>();
    hcclComm->pimpl->rankSize = 4;
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    aclrtStream stream = &recvCount;

    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcclScatterV2(sendBuf, recvBuf, recvCount, dataType, root, comm, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclAllGatherV2)
{
    // Arrange
    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    uint64_t recvCount = 10;
    HcclDataType dataType = HCCL_DATA_TYPE_INT8;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    int a = 10;
    aclrtStream stream = static_cast<aclrtStream>(&a);

    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcclAllGatherV2(sendBuf, recvBuf, recvCount, dataType, comm, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclSendV2)
{
    // Arrange
    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    uint64_t recvCount = 10;
    uint32_t destRank = 1;
    HcclDataType dataType = HCCL_DATA_TYPE_INT8;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    aclrtStream stream = nullptr;
    uint64_t count = 10;

    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcclSendV2(sendBuf, count, dataType, destRank, comm, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclRecvV2)
{
    // Arrange
    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    uint64_t recvCount = 10;
    uint32_t srcRank = 1;
    HcclDataType dataType = HCCL_DATA_TYPE_INT8;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    aclrtStream stream = nullptr;
    uint64_t count = 10;

    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcclRecvV2(recvBuf, count, dataType, srcRank, comm, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclReduceScatterV2)
{
    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    uint64_t recvCount = 10;
    HcclDataType dataType = HCCL_DATA_TYPE_INT8;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    int a = 10;
    aclrtStream stream = static_cast<aclrtStream>(&a);
    DevType devType = DevType::DEV_TYPE_950;
    HcclReduceOp op = HCCL_REDUCE_SUM;

    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HcclReduceScatterV2(sendBuf, recvBuf, recvCount, dataType, op, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    op = HCCL_REDUCE_MAX;
    ret = HcclReduceScatterV2(sendBuf, recvBuf, recvCount, dataType, op, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    op = HCCL_REDUCE_MIN;
    ret = HcclReduceScatterV2(sendBuf, recvBuf, recvCount, dataType, op, comm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    op = HCCL_REDUCE_PROD;
    ret = HcclReduceScatterV2(sendBuf, recvBuf, recvCount, dataType, op, comm, stream);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(OpbaseTestV2, HcclGetRawCommHandle)
{
    string commName = "hccl_world_group";
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm_1 = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm_1;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm_1;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));

    HcclComm commHandle;

    HcclResult ret = HcclGetRawCommHandle(commName.c_str(), &commHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclGetCcuTaskInfoLegacy_OK)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm_1 = std::make_shared<Hccl::HcclCommunicator>(commParams);

    HcclComm comm = static_cast<HcclComm>(hcclComm_1.get());

    int a = 5;
    void* ccuTaskGroup = static_cast<void*>(&a);
    void* fusionArgs = static_cast<void*>(&a);
    MOCKER_CPP(&HcclCommunicator::GetCcuTaskInfo)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HcclGetCcuTaskInfoLegacy(comm, fusionArgs, ccuTaskGroup);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclGetCcuTaskInfoLegacy_ERR)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm_1 = std::make_shared<Hccl::HcclCommunicator>(commParams);

    HcclComm comm = static_cast<HcclComm>(hcclComm_1.get());
    int a = 5;
    void* ccuTaskGroup = static_cast<void*>(&a);
    void* fusionArgs = static_cast<void*>(&a);
    MOCKER_CPP(&HcclCommunicator::GetCcuTaskInfo)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_INTERNAL));
    HcclResult ret = HcclGetCcuTaskInfoLegacy(comm, fusionArgs, ccuTaskGroup);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(OpbaseTestV2, HcclSnapshotSave_err)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm_1 = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm_1;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm_1;
    CommManager::GetInstance(0).GetCommInfoV2().step = 0;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));

    uint32_t step = 0;
    char* snapshot = new char[200];
    void* snapshotBuf = static_cast<void*>(snapshot);
    uint32_t size = 0;
    HcclResult ret = HcclSnapshotSave(snapshotBuf, size, step);
    EXPECT_EQ(ret, HCCL_E_PARA);

    delete[] snapshot;
}

TEST_F(OpbaseTestV2, HcclSnapshotSave_err2)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm_1 = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm_1;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm_1;
    CommManager::GetInstance(0).GetCommInfoV2().step = 1;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));

    uint32_t step = 1;
    char* snapshot = new char[200];
    void* snapshotBuf = static_cast<void*>(snapshot);
    Hccl::BinaryStream& savedSnapshotBuf = Hccl::SnapShotParser::GetInstance().GetSnapShotBuf();
    uint32_t dataLen = savedSnapshotBuf.GetSize();
    uint32_t size = dataLen + sizeof(dataLen) + sizeof(dataLen);
    ;

    HcclResult ret = HcclSnapshotSave(snapshotBuf, size, step);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    delete[] snapshot;
}

TEST_F(OpbaseTestV2, HcclSnapshotRecoverAllComms_OK)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm_1 = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm_1;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm_1;
    CommManager::GetInstance(0).GetCommInfoV2().step = 0;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));

    std::shared_ptr<Hccl::SnapShotBuf> snapshotBuf = std::make_shared<Hccl::SnapShotBuf>();
    snapshotBuf->groupNum = 0;
    void* snapshot = static_cast<void*>(snapshotBuf.get());
    string clusterInfo = "test";
    string changedInfo = "test";
    uint32_t snapshotBufSize = 10;
    MOCKER_CPP(&HcclCommunicator::RecoverComm)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_PARA));

    string worldgroup = "hccl_world_group";
    SnapShotBuf localBuff;
    strncpy(localBuff.snapshot.groupName, worldgroup.c_str(), worldgroup.size() + 1);

    MOCKER_CPP(&HcclCommunicator::IsCommReady).stubs().with().will(returnValue(true));
    MOCKER_CPP(&SnapShotParser::ParseSnapshotToLocalBuff)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(localBuff))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::RecoverSubComm)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_PARA));
    HcclResult ret = HcclSnapshotRecoverAllComms(clusterInfo.c_str(), changedInfo.c_str(), snapshot, snapshotBufSize);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(OpbaseTestV2, SnapshotGenerate)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> pComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    pComm.get()->GetCommImpl()->id = "hccl_world_group";
    pComm.get()->GetCommImpl()->isWorldGroup = true;
    std::map<std::string, std::shared_ptr<Hccl::HcclCommunicator>> hcclGroupMap;
    hcclGroupMap["group0"] = pComm;
    uint32_t step = 0;
    uint32_t size = 0;

    HcclResult ret = SnapshotGenerate(pComm, hcclGroupMap, step, &size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclSnapshotGetBufSize_OK)
{
    DevType devType = DevType::DEV_TYPE_950;

    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm_1 = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm_1;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm_1;
    CommManager::GetInstance(0).GetCommInfoV2().step = 0;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER(HrtGetDeviceType).stubs().with(mockcpp::any()).will(returnValue(devType));

    uint32_t step = 0;
    uint32_t size = 0;
    hcclComm_1.get()->GetCommImpl()->id = "hccl_world_group";
    hcclComm_1.get()->GetCommImpl()->isWorldGroup = true;
    HcclResult ret = HcclSnapshotGetBufSize(step, &size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclSnapshotGetBufSize_HCCL_SUCCESS)
{
    DevType devType = DevType::DEV_TYPE_950;

    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm_1 = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm_1;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;
    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm_1;
    CommManager::GetInstance(0).GetCommInfoV2().step = 0;

    uint32_t step = 0;
    uint32_t size = 0;
    hcclComm_1.get()->GetCommImpl()->id = "hccl_world_group";
    hcclComm_1.get()->GetCommImpl()->isWorldGroup = true;
    MOCKER(HrtGetDeviceType).stubs().with(mockcpp::any()).will(returnValue(devType));
    HcclResult ret = HcclSnapshotGetBufSize(step, &size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap.clear();
    CommManager::GetInstance(0).GetCommInfoV2().pComm = nullptr;
    hcclComm_1 = nullptr;
}
TEST_F(OpbaseTestV2, HcclGetTopoDescV2)
{
    HcclResult ret = HcclGetTopoDescV2();
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(OpbaseTestV2, HcclCreateSubCommConfigV2)
{
    Hccl::CommParams commParams_1;
    std::shared_ptr<Hccl::HcclCommunicator> hcclcomm = std::make_shared<Hccl::HcclCommunicator>(commParams_1);
    HcclComm comm = static_cast<HcclComm>(hcclcomm.get());
    HcclComm subComm;
    uint32_t rankNum = 1;
    uint32_t rankIds = 0;
    uint64_t subCommId = 42;
    uint32_t subCommRankId = 1;
    HcclCommConfig config;
    string worldgroup = "hccl_world_group";
    PrepareCommConfig(config, 200, worldgroup, 1, 0);

    // 打桩 hrtGetDevice
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm_1 = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm_1;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm_1;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));

    HcclGroupParamsV2 groupParamsV2Tem;
    groupParamsV2Tem.groupRank = 0;
    MOCKER(GetHcomRankListV2)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(groupParamsV2Tem))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(static_cast<HcclResult (CommunicatorImpl::*)(
                   const CommParams& subCommParams, const std::vector<u32>& rankIds, CommunicatorImpl* subCommImpl,
                   HcclCommConfig& subConfig)>(&CommunicatorImpl::CreateSubComm))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    hcclcomm.get()->GetCommImpl()->id = "hccl_world_group";
    hcclcomm.get()->GetCommImpl()->isWorldGroup = true;
    HcclResult ret = HcclCreateSubCommConfigV2(&comm, rankNum, &rankIds, subCommId, subCommRankId, &config, &subComm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclCreateSubCommConfigV2_IDEL)
{
    Hccl::CommParams commParams_1;
    std::shared_ptr<Hccl::HcclCommunicator> hcclcomm = std::make_shared<Hccl::HcclCommunicator>(commParams_1);
    HcclComm comm = static_cast<HcclComm>(hcclcomm.get());
    HcclComm subComm;
    uint32_t rankNum = 1;
    uint32_t rankIds = 0;
    uint64_t subCommId = 42;
    uint32_t subCommRankId = 1;
    HcclCommConfig config;
    string worldgroup = "hccl_world_group_1";
    PrepareCommConfig(config, 200, worldgroup, 1, 0);

    // 打桩 hrtGetDevice
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm_1 = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm_1;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm_1;
    CommManager::GetInstance(0).GetCommInfoV2().status = DeviceStatus::DEVICE_IDLE;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));

    HcclGroupParamsV2 groupParamsV2Tem;
    groupParamsV2Tem.groupRank = 0;
    MOCKER(GetHcomRankListV2)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(groupParamsV2Tem))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(static_cast<HcclResult (CommunicatorImpl::*)(
                   const CommParams& subCommParams, const std::vector<u32>& rankIds, CommunicatorImpl* subCommImpl,
                   HcclCommConfig& subConfig)>(&CommunicatorImpl::CreateSubComm))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    hcclcomm.get()->GetCommImpl()->id = "hccl_world_group";
    hcclcomm.get()->GetCommImpl()->isWorldGroup = true;
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::RegisterAcceStateCallBack).stubs().will(ignoreReturnValue());
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(HrtGetDeviceType).stubs().with(mockcpp::any()).will(returnValue(devType));
    HcclResult ret = HcclCreateSubCommConfigV2(&comm, rankNum, &rankIds, subCommId, subCommRankId, &config, &subComm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclCreateSubCommConfigV2_Invalided_Config)
{
    Hccl::CommParams commParams_1;
    std::shared_ptr<Hccl::HcclCommunicator> hcclcomm = std::make_shared<Hccl::HcclCommunicator>(commParams_1);
    HcclComm comm = static_cast<HcclComm>(hcclcomm.get());
    HcclComm subComm;
    uint32_t rankNum = 1;
    uint32_t rankIds = 0;
    uint64_t subCommId = 43;
    uint32_t subCommRankId = 1;
    HcclCommConfig config;
    string worldgroup = "hccl_world_group_1";
    PrepareCommConfig(config, HCCL_COMM_BUFFSIZE_CONFIG_NOT_SET, worldgroup, 1, 0);

    // 打桩 hrtGetDevice
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm_1 = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm_1;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm_1;
    CommManager::GetInstance(0).GetCommInfoV2().status = DeviceStatus::DEVICE_IDLE;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));

    HcclGroupParamsV2 groupParamsV2Tem;
    groupParamsV2Tem.groupRank = 0;
    MOCKER(GetHcomRankListV2)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(groupParamsV2Tem))
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(static_cast<HcclResult (CommunicatorImpl::*)(
                   const CommParams& subCommParams, const std::vector<u32>& rankIds, CommunicatorImpl* subCommImpl,
                   HcclCommConfig& subConfig)>(&CommunicatorImpl::CreateSubComm))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    hcclcomm.get()->GetCommImpl()->id = "hccl_world_group";
    hcclcomm.get()->GetCommImpl()->isWorldGroup = true;
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());
    HcclResult ret = HcclCreateSubCommConfigV2(&comm, rankNum, &rankIds, subCommId, subCommRankId, &config, &subComm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclCommInitClusterInfoMemConfigV2_err)
{
    Hccl::CommParams commParams;
    Hccl::HcclCommunicator communicator(commParams);
    HcclComm hcom = static_cast<HcclComm>(&communicator);

    HcclGroupParamsV2 hcclGroupParamsV2;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    // 打桩GetCommInfoV2。
    CommManager::GetInstance(0).GetCommInfoV2().commParams.commId = "hccl_world_group";

    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));
    std::string rankTableString = "test";
    uint32_t rank = 0;
    HcclCommConfig config;
    string worldgroup = "hccl_world_group_1";
    PrepareCommConfig(config, 200, worldgroup, 1, 0);

    Hccl::CommParams commParams1;
    std::unique_ptr<Hccl::HcclCommunicator> communicator_1 = std::make_unique<Hccl::HcclCommunicator>(commParams1);
    HcclComm comm = static_cast<HcclComm>(communicator_1.get());
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());
    HcclResult ret = HcclCommInitClusterInfoMemConfigV2(rankTableString.c_str(), rank, &config, &comm);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclCommInitClusterInfoMemConfigV2_err_2)
{
    Hccl::CommParams commParams;
    Hccl::HcclCommunicator communicator(commParams);
    HcclComm hcom = static_cast<HcclComm>(&communicator);

    HcclGroupParamsV2 hcclGroupParamsV2;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    // 打桩GetCommInfoV2。
    CommManager::GetInstance(0).GetCommInfoV2().commParams.commId = "hccl_world_group";

    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));

    MOCKER_CPP(&HcclCommunicator::Init, HcclResult(HcclCommunicator::*)(const std::string&))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_E_INTERNAL));
    std::string rankTableString = rankTable_ut_stub_4p;
    uint32_t rank = 0;
    HcclCommConfig config;
    string worldgroup = "hccl_world_group_1";
    PrepareCommConfig(config, 200, worldgroup, 1, 0);

    Hccl::CommParams commParams1;
    std::unique_ptr<Hccl::HcclCommunicator> communicator_1 = std::make_unique<Hccl::HcclCommunicator>(commParams1);
    HcclComm comm = static_cast<HcclComm>(communicator_1.get());
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());
    HcclResult ret = HcclCommInitClusterInfoMemConfigV2(rankTableString.c_str(), rank, &config, &comm);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclCommInitClusterInfoMemConfigV2_err_3)
{
    Hccl::CommParams commParams;
    Hccl::HcclCommunicator communicator(commParams);
    HcclComm hcom = static_cast<HcclComm>(&communicator);

    HcclGroupParamsV2 hcclGroupParamsV2;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    // 打桩GetCommInfoV2。
    CommManager::GetInstance(0).GetCommInfoV2().commParams.commId = "hccl_world_group";

    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));

    MOCKER_CPP(&HcclCommunicator::Init, HcclResult(HcclCommunicator::*)(const std::string&))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_E_INTERNAL));
    std::string rankTableString = rankTable_ut_stub_4p;
    uint32_t rank = 0;
    HcclCommConfig config;
    string worldgroup = "hccl_world_group_1";
    PrepareCommConfig(config, 200, worldgroup, 1, 0);

    Hccl::CommParams commParams1;
    std::unique_ptr<Hccl::HcclCommunicator> communicator_1 = std::make_unique<Hccl::HcclCommunicator>(commParams1);
    HcclComm comm = static_cast<HcclComm>(communicator_1.get());
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());
    HcclResult ret = HcclCommInitClusterInfoMemConfigV2(rankTableString.c_str(), rank, &config, &comm);
    EXPECT_NE(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclCommInitClusterInfoMemConfigV2)
{
    Hccl::CommParams commParams;
    Hccl::HcclCommunicator communicator(commParams);
    HcclComm hcom = static_cast<HcclComm>(&communicator);

    HcclGroupParamsV2 hcclGroupParamsV2;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;
    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    // 打桩GetCommInfoV2。
    CommManager::GetInstance(0).GetCommInfoV2().commParams.commId = "hccl_world_group";

    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));

    MOCKER_CPP(&HcclCommunicator::Init, HcclResult(HcclCommunicator::*)(const std::string&))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    std::string rankTableString = rankTable_ut_stub_4p;
    uint32_t rank = 0;
    HcclCommConfig config;
    string worldgroup = "hccl_world_group_1";
    PrepareCommConfig(config, 200, worldgroup, 1, 0);

    Hccl::CommParams commParams1;
    std::unique_ptr<Hccl::HcclCommunicator> communicator_1 = std::make_unique<Hccl::HcclCommunicator>(commParams1);
    HcclComm comm = static_cast<HcclComm>(communicator_1.get());
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());
    HcclResult ret = HcclCommInitClusterInfoMemConfigV2(rankTableString.c_str(), rank, &config, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclCommDestroyV2) // 放最后
{
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    HcclComm hcom = static_cast<HcclComm>(communicator.get());

    HcclGroupParamsV2 hcclGroupParamsV2;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    // 打桩GetCommInfoV2。
    CommManager::GetInstance(0).GetCommInfoV2().commParams.commId = "hccl_world_group";

    communicator.get()->GetCommImpl()->id = "hccl_world_group";
    communicator.get()->GetCommImpl()->isWorldGroup = true;
    MOCKER_CPP(&HcclCommunicator::IsCommReady).stubs().with().will(returnValue(true));
    HcclResult ret = HcclCommDestroyV2(hcom);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclCommDestroyV2_2)
{
    MOCKER(aclrtGetDevice).stubs().will(returnValue(0));

    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    HcclComm hcom = static_cast<HcclComm>(communicator.get());

    HcclResult ret = HcclCommDestroyV2(hcom);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(OpbaseTestV2, HcclGetCommAsyncErrorV2)
{
    HcclResult ret = HcclGetCommAsyncErrorV2();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, St_HcclAllGatherVV2_When_Normal_Expect_Success)
{
    constexpr u64 FAKE_RANK_SIZE = 2;
    constexpr void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_INT32;
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    communicator->GetCommImpl()->myRank = 0;
    communicator->GetCommImpl()->rankSize = FAKE_RANK_SIZE;
    HcclComm comm = static_cast<HcclComm>(communicator.get());
    void* sendBuf = FAKE_PTR;
    u64 sendCount = 1;
    void* recvBuf = FAKE_PTR;
    u64 recvCounts[FAKE_RANK_SIZE] = {1, 1};
    u64 recvDispls[FAKE_RANK_SIZE] = {1, 1};
    HcclDataType sendType = FAKE_DATA_TYPE;
    int a = 1;
    aclrtStream stream = static_cast<aclrtStream>(&a);

    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcclAllGatherVV2(sendBuf, sendCount, recvBuf, &recvCounts, &recvDispls, sendType, comm, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, St_HcclAllGatherVV2_When_DipHas0_Expect_Success)
{
    constexpr u64 FAKE_RANK_SIZE = 2;
    constexpr void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_INT32;
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    communicator->GetCommImpl()->myRank = 0;
    communicator->GetCommImpl()->rankSize = FAKE_RANK_SIZE;
    HcclComm comm = static_cast<HcclComm>(communicator.get());
    void* sendBuf = FAKE_PTR;
    u64 sendCount = 1;
    void* recvBuf = FAKE_PTR;
    u64 recvCounts[FAKE_RANK_SIZE] = {1, 1};
    u64 recvDispls[FAKE_RANK_SIZE] = {0, 1};
    HcclDataType sendType = FAKE_DATA_TYPE;
    int a = 1;
    aclrtStream stream = static_cast<aclrtStream>(&a);

    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcclAllGatherVV2(sendBuf, sendCount, recvBuf, &recvCounts, &recvDispls, sendType, comm, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, St_HcclAllGatherVV2_When_OutputEq0_Expect_Success)
{
    constexpr u64 FAKE_RANK_SIZE = 2;
    constexpr void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_INT32;
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    communicator->GetCommImpl()->myRank = 0;
    communicator->GetCommImpl()->rankSize = FAKE_RANK_SIZE;
    HcclComm comm = static_cast<HcclComm>(communicator.get());
    void* sendBuf = FAKE_PTR;
    u64 sendCount = 0;
    void* recvBuf = FAKE_PTR;
    u64 recvCounts[FAKE_RANK_SIZE] = {0, 0};
    u64 recvDispls[FAKE_RANK_SIZE] = {1, 1};
    HcclDataType sendType = FAKE_DATA_TYPE;
    int a = 1;
    aclrtStream stream = static_cast<aclrtStream>(&a);

    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcclAllGatherVV2(sendBuf, sendCount, recvBuf, &recvCounts, &recvDispls, sendType, comm, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, St_HcclAllGatherVV2_When_CountNotFixCounts_Expect_Error)
{
    constexpr u64 FAKE_RANK_SIZE = 2;
    constexpr void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_INT32;
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    communicator->GetCommImpl()->myRank = 0;
    communicator->GetCommImpl()->rankSize = FAKE_RANK_SIZE;
    HcclComm comm = static_cast<HcclComm>(communicator.get());
    void* sendBuf = FAKE_PTR;
    u64 sendCount = 1;
    void* recvBuf = FAKE_PTR;
    u64 recvCounts[FAKE_RANK_SIZE] = {0, 1};
    u64 recvDispls[FAKE_RANK_SIZE] = {1, 1};
    HcclDataType sendType = FAKE_DATA_TYPE;
    int a = 1;
    aclrtStream stream = static_cast<aclrtStream>(&a);

    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcclAllGatherVV2(sendBuf, sendCount, recvBuf, &recvCounts, &recvDispls, sendType, comm, stream);
    EXPECT_EQ(result, HCCL_E_PARA);
}

TEST_F(OpbaseTestV2, St_HcclAllGatherVV2_When_CountTooLarge_Expect_Error)
{
    constexpr u64 FAKE_RANK_SIZE = 2;
    constexpr void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_INT32;
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    communicator->GetCommImpl()->myRank = 0;
    communicator->GetCommImpl()->rankSize = FAKE_RANK_SIZE;
    HcclComm comm = static_cast<HcclComm>(communicator.get());
    void* sendBuf = FAKE_PTR;
    u64 sendCount = 0x7ffffffffff;
    void* recvBuf = FAKE_PTR;
    u64 recvCounts[FAKE_RANK_SIZE] = {1, 1};
    u64 recvDispls[FAKE_RANK_SIZE] = {1, 1};
    HcclDataType sendType = FAKE_DATA_TYPE;
    int a = 1;
    aclrtStream stream = static_cast<aclrtStream>(&a);

    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcclAllGatherVV2(sendBuf, sendCount, recvBuf, &recvCounts, &recvDispls, sendType, comm, stream);
    EXPECT_EQ(result, HCCL_E_PARA);
}

TEST_F(OpbaseTestV2, St_HcclAllGatherVV2_When_DatatypeNotSurport_Expect_Error)
{
    constexpr u64 FAKE_RANK_SIZE = 2;
    constexpr void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_RESERVED;
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    communicator->GetCommImpl()->myRank = 0;
    communicator->GetCommImpl()->rankSize = FAKE_RANK_SIZE;
    HcclComm comm = static_cast<HcclComm>(communicator.get());
    void* sendBuf = FAKE_PTR;
    u64 sendCount = 1;
    void* recvBuf = FAKE_PTR;
    u64 recvCounts[FAKE_RANK_SIZE] = {1, 1};
    u64 recvDispls[FAKE_RANK_SIZE] = {1, 1};
    HcclDataType sendType = FAKE_DATA_TYPE;
    int a = 1;
    aclrtStream stream = static_cast<aclrtStream>(&a);

    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcclAllGatherVV2(sendBuf, sendCount, recvBuf, &recvCounts, &recvDispls, sendType, comm, stream);
    EXPECT_EQ(result, HCCL_E_NOT_SUPPORT);
}

TEST_F(OpbaseTestV2, St_HcclReduceScatterVV2_When_Normal_Expect_Success)
{
    constexpr u64 FAKE_RANK_SIZE = 2;
    constexpr void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_INT32;
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    communicator->GetCommImpl()->myRank = 0;
    communicator->GetCommImpl()->rankSize = FAKE_RANK_SIZE;
    HcclComm comm = static_cast<HcclComm>(communicator.get());
    void* sendBuf = FAKE_PTR;
    u64 sendCounts[FAKE_RANK_SIZE] = {1, 1};
    u64 sendDispls[FAKE_RANK_SIZE] = {1, 1};
    void* recvBuf = FAKE_PTR;
    u64 recvCount = 1;
    HcclDataType dataType = FAKE_DATA_TYPE;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    int a = 1;
    aclrtStream stream = static_cast<aclrtStream>(&a);

    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result
        = HcclReduceScatterVV2(sendBuf, &sendCounts, &sendDispls, recvBuf, recvCount, dataType, op, comm, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, St_HcclReduceScatterVV2_When_DipHas0_Expect_Success)
{
    constexpr u64 FAKE_RANK_SIZE = 2;
    constexpr void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_INT32;
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    communicator->GetCommImpl()->myRank = 0;
    communicator->GetCommImpl()->rankSize = FAKE_RANK_SIZE;
    HcclComm comm = static_cast<HcclComm>(communicator.get());
    void* sendBuf = FAKE_PTR;
    u64 sendCounts[FAKE_RANK_SIZE] = {1, 1};
    u64 sendDispls[FAKE_RANK_SIZE] = {0, 1};
    void* recvBuf = FAKE_PTR;
    u64 recvCount = 1;
    HcclDataType dataType = FAKE_DATA_TYPE;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    int a = 1;
    aclrtStream stream = static_cast<aclrtStream>(&a);

    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result
        = HcclReduceScatterVV2(sendBuf, &sendCounts, &sendDispls, recvBuf, recvCount, dataType, op, comm, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, St_HcclReduceScatterVV2_When_InuputCountEq0_Expect_Success)
{
    constexpr u64 FAKE_RANK_SIZE = 2;
    constexpr void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_INT32;
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    communicator->GetCommImpl()->myRank = 0;
    communicator->GetCommImpl()->rankSize = FAKE_RANK_SIZE;
    HcclComm comm = static_cast<HcclComm>(communicator.get());
    void* sendBuf = FAKE_PTR;
    u64 sendCounts[FAKE_RANK_SIZE] = {0, 0};
    u64 sendDispls[FAKE_RANK_SIZE] = {1, 1};
    void* recvBuf = FAKE_PTR;
    u64 recvCount = 0;
    HcclDataType dataType = FAKE_DATA_TYPE;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    int a = 1;
    aclrtStream stream = static_cast<aclrtStream>(&a);

    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result
        = HcclReduceScatterVV2(sendBuf, &sendCounts, &sendDispls, recvBuf, recvCount, dataType, op, comm, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, St_HcclReduceScatterVV2_When_CountNotFixCounts_Expect_Error)
{
    constexpr u64 FAKE_RANK_SIZE = 2;
    constexpr void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_INT32;
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    communicator->GetCommImpl()->myRank = 0;
    communicator->GetCommImpl()->rankSize = FAKE_RANK_SIZE;
    HcclComm comm = static_cast<HcclComm>(communicator.get());
    void* sendBuf = FAKE_PTR;
    u64 sendCounts[FAKE_RANK_SIZE] = {1, 1};
    u64 sendDispls[FAKE_RANK_SIZE] = {1, 1};
    void* recvBuf = FAKE_PTR;
    u64 recvCount = 0;
    HcclDataType dataType = FAKE_DATA_TYPE;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    int a = 1;
    aclrtStream stream = static_cast<aclrtStream>(&a);

    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result
        = HcclReduceScatterVV2(sendBuf, &sendCounts, &sendDispls, recvBuf, recvCount, dataType, op, comm, stream);
    EXPECT_EQ(result, HCCL_E_PARA);
}

TEST_F(OpbaseTestV2, St_HcclReduceScatterVV2_When_CountTooLarge_Expect_Error)
{
    constexpr u64 FAKE_RANK_SIZE = 2;
    constexpr void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_INT32;
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    communicator->GetCommImpl()->myRank = 0;
    communicator->GetCommImpl()->rankSize = FAKE_RANK_SIZE;
    HcclComm comm = static_cast<HcclComm>(communicator.get());
    void* sendBuf = FAKE_PTR;
    u64 sendCounts[FAKE_RANK_SIZE] = {1, 1};
    u64 sendDispls[FAKE_RANK_SIZE] = {1, 1};
    void* recvBuf = FAKE_PTR;
    u64 recvCount = 0xffffffff;
    HcclDataType dataType = FAKE_DATA_TYPE;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    int a = 1;
    aclrtStream stream = static_cast<aclrtStream>(&a);

    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result
        = HcclReduceScatterVV2(sendBuf, &sendCounts, &sendDispls, recvBuf, recvCount, dataType, op, comm, stream);
    EXPECT_EQ(result, HCCL_E_PARA);
}

TEST_F(OpbaseTestV2, St_HcclReduceScatterVV2_When_DatatypeNotSurport_Expect_Error)
{
    constexpr u64 FAKE_RANK_SIZE = 2;
    constexpr void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_RESERVED;
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    communicator->GetCommImpl()->myRank = 0;
    communicator->GetCommImpl()->rankSize = FAKE_RANK_SIZE;
    HcclComm comm = static_cast<HcclComm>(communicator.get());
    void* sendBuf = FAKE_PTR;
    u64 sendCounts[FAKE_RANK_SIZE] = {1, 1};
    u64 sendDispls[FAKE_RANK_SIZE] = {1, 1};
    void* recvBuf = FAKE_PTR;
    u64 recvCount = 1;
    HcclDataType dataType = FAKE_DATA_TYPE;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    int a = 1;
    aclrtStream stream = static_cast<aclrtStream>(&a);

    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result
        = HcclReduceScatterVV2(sendBuf, &sendCounts, &sendDispls, recvBuf, recvCount, dataType, op, comm, stream);
    EXPECT_EQ(result, HCCL_E_NOT_SUPPORT);
}

TEST_F(OpbaseTestV2, St_HcclReduceScatterVV2_When_ReduceOpNotSurport_Expect_Error)
{
    constexpr u64 FAKE_RANK_SIZE = 2;
    constexpr void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_INT32;
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    communicator->GetCommImpl()->myRank = 0;
    communicator->GetCommImpl()->rankSize = FAKE_RANK_SIZE;
    HcclComm comm = static_cast<HcclComm>(communicator.get());
    void* sendBuf = FAKE_PTR;
    u64 sendCounts[FAKE_RANK_SIZE] = {1, 1};
    u64 sendDispls[FAKE_RANK_SIZE] = {1, 1};
    void* recvBuf = FAKE_PTR;
    u64 recvCount = 1;
    HcclDataType dataType = FAKE_DATA_TYPE;
    HcclReduceOp op = HCCL_REDUCE_PROD;
    int a = 1;
    aclrtStream stream = static_cast<aclrtStream>(&a);

    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result
        = HcclReduceScatterVV2(sendBuf, &sendCounts, &sendDispls, recvBuf, recvCount, dataType, op, comm, stream);
    EXPECT_EQ(result, HCCL_E_NOT_SUPPORT);
}

TEST_F(OpbaseTestV2, St_HcclGetRootInfoV2_When_NoNeedInput_Expect_Return_HCCL_SUCCESS)
{
    // when
    HcclRootHandleV2 rootHandle{};
    MOCKER_CPP(&RankInfoDetect::SetupServer).stubs().with(outBound(rootHandle)).will(ignoreReturnValue());
    MOCKER(HrtGetDeviceCount).stubs().will(returnValue(1));
    MOCKER(HrtGetDeviceType).stubs().with(mockcpp::any()).will(returnValue((DevType)DevType::DEV_TYPE_950));

    // then
    HcclRootInfo rootInfo{};
    EXPECT_EQ(HcclGetRootInfoV2(&rootInfo), HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, St_HcclGetRootInfoV2_When_Throw_Expect_Return_HCCL_E_INTERNAL)
{
    // when
    MOCKER_CPP(&RankInfoDetect::SetupServer)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(throws(InternalException("...")));
    MOCKER(HrtGetDeviceCount).stubs().will(returnValue(1));
    MOCKER(HrtGetDeviceType).stubs().with(mockcpp::any()).will(returnValue((DevType)DevType::DEV_TYPE_950));

    // then
    HcclRootInfo rootInfo{};
    EXPECT_EQ(HcclGetRootInfoV2(&rootInfo), HCCL_E_INTERNAL);
}

TEST_F(OpbaseTestV2, St_HcclCommInitRootInfoV2_When_InputValue_Expect_Return_HCCL_SUCCESS)
{
    // when
    HcclRootHandleV2 rootHandle{};
    MOCKER_CPP(&RankInfoDetect::SetupAgent)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(ignoreReturnValue());
    MOCKER_CPP(&RankInfoDetect::WaitComplete).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER(HrtGetDeviceType).stubs().with(mockcpp::any()).will(returnValue((DevType)DevType::DEV_TYPE_950));
    MOCKER(HrtGetDeviceCount).stubs().will(returnValue(1));
    MOCKER_CPP(&HcclCommunicator::Init, HcclResult(HcclCommunicator::*)(const RankTableInfo&))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());

    // then
    uint32_t nRanks = 1;
    HcclRootInfo rootInfo{};
    uint32_t rank = 0;
    HcclComm comm{};
    std::string identifier{};
    EXPECT_EQ(HcclCommInitRootInfoV2(nRanks, &rootInfo, rank, &comm, identifier), HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, St_HcclCommInitRootInfoV2_When_Throw_Expect_Return_HCCL_E_INTERNAL)
{
    // when
    MOCKER_CPP(&RankInfoDetect::SetupAgent)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(throws(InternalException("...")));
    MOCKER_CPP(&RankInfoDetect::WaitComplete).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER(HrtGetDeviceType).stubs().with(mockcpp::any()).will(returnValue((DevType)DevType::DEV_TYPE_950));
    MOCKER(HrtGetDeviceCount).stubs().will(returnValue(1));
    MOCKER_CPP(&HcclCommunicator::Init, HcclResult(HcclCommunicator::*)(const RankTableInfo&))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap.clear();
    CommManager::GetInstance(0).GetCommInfoV2().pComm = nullptr;

    // then
    uint32_t nRanks{};
    HcclRootInfo rootInfo{};
    uint32_t rank{};
    HcclComm comm{};
    std::string identifier{};
    EXPECT_EQ(HcclCommInitRootInfoV2(nRanks, &rootInfo, rank, &comm, identifier), HCCL_E_INTERNAL);
}

TEST_F(OpbaseTestV2, St_HcclCommInitRootInfoConfigV2_When_InputValue_Expect_Return_HCCL_SUCCESS)
{
    // when
    HcclRootHandleV2 rootHandle{};
    MOCKER_CPP(&RankInfoDetect::SetupAgent)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(ignoreReturnValue());
    MOCKER_CPP(&RankInfoDetect::WaitComplete).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER(HrtGetDeviceType).stubs().with(mockcpp::any()).will(returnValue((DevType)DevType::DEV_TYPE_950));
    MOCKER(HrtGetDeviceCount).stubs().will(returnValue(1));
    MOCKER_CPP(&HcclCommunicator::Init, HcclResult(HcclCommunicator::*)(const RankTableInfo&))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap.clear();
    CommManager::GetInstance(0).GetCommInfoV2().pComm = nullptr;

    // then
    uint32_t nRanks{2};
    HcclRootInfo rootInfo{};
    uint32_t rank{};
    HcclComm comm{};
    HcclCommConfig config{};
    string worldgroup = "hccl_world_group";
    PrepareCommConfig(config, 200, worldgroup, 1, 0);
    EXPECT_EQ(HcclCommInitRootInfoConfigV2(nRanks, &rootInfo, rank, &config, &comm), HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, St_HcclCommInitRootInfoConfigV2_When_Throw_Expect_Return_HCCL_E_INTERNAL)
{
    // when
    MOCKER_CPP(&RankInfoDetect::SetupAgent)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(throws(InternalException("...")));
    MOCKER_CPP(&RankInfoDetect::WaitComplete).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER(HrtGetDeviceType).stubs().with(mockcpp::any()).will(returnValue((DevType)DevType::DEV_TYPE_950));
    MOCKER(HrtGetDeviceCount).stubs().will(returnValue(1));
    MOCKER_CPP(&HcclCommunicator::Init, HcclResult(HcclCommunicator::*)(const RankTableInfo&))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap.clear();
    CommManager::GetInstance(0).GetCommInfoV2().pComm = nullptr;

    // then
    uint32_t nRanks{2};
    HcclRootInfo rootInfo{};
    uint32_t rank{};
    HcclComm comm{};
    HcclCommConfig config{};
    HcclCommConfigInit(&config);
    EXPECT_EQ(HcclCommInitRootInfoConfigV2(nRanks, &rootInfo, rank, &config, &comm), HCCL_E_INTERNAL);
}

TEST_F(OpbaseTestV2, St_HcclCommInitAllV2_When_InputValue_Expect_Return_HCCL_SUCCESS)
{
    // when
    HcclRootHandleV2 rootHandle{};
    MOCKER_CPP(&RankInfoDetect::SetupServer).stubs().with(outBound(rootHandle)).will(ignoreReturnValue());
    MOCKER(HrtGetDeviceCount).stubs().will(returnValue(1));
    MOCKER(HrtGetDeviceType).stubs().with(mockcpp::any()).will(returnValue((DevType)DevType::DEV_TYPE_950));
    MOCKER_CPP(&RankInfoDetect::SetupAgent)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(ignoreReturnValue());
    MOCKER_CPP(&RankInfoDetect::WaitComplete).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER(HrtGetDeviceType).stubs().with(mockcpp::any()).will(returnValue((DevType)DevType::DEV_TYPE_950));
    MOCKER(HrtGetDeviceCount).stubs().will(returnValue(1));
    MOCKER_CPP(&HcclCommunicator::Init, HcclResult(HcclCommunicator::*)(const RankTableInfo&))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap.clear();
    CommManager::GetInstance(0).GetCommInfoV2().pComm = nullptr;

    // then
    uint32_t ndev = 1;
    int32_t devices{};
    HcclComm comms{};
    EXPECT_EQ(HcclCommInitAllV2(ndev, &devices, &comms), HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, St_HcclCommInitAllV2_When_Throw_Expect_Return_HCCL_E_INTERNAL)
{
    // when
    MOCKER_CPP(&RankInfoDetect::SetupServer)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(throws(InternalException("...")));
    MOCKER(HrtGetDeviceCount).stubs().will(returnValue(1));
    MOCKER(HrtGetDeviceType).stubs().with(mockcpp::any()).will(returnValue((DevType)DevType::DEV_TYPE_950));
    MOCKER_CPP(&RankInfoDetect::SetupAgent)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(ignoreReturnValue());
    MOCKER_CPP(&RankInfoDetect::WaitComplete).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER(HrtGetDeviceType).stubs().with(mockcpp::any()).will(returnValue((DevType)DevType::DEV_TYPE_950));
    MOCKER(HrtGetDeviceCount).stubs().will(returnValue(1));
    MOCKER_CPP(&HcclCommunicator::Init, HcclResult(HcclCommunicator::*)(const RankTableInfo&))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap.clear();
    CommManager::GetInstance(0).GetCommInfoV2().pComm = nullptr;

    // then
    uint32_t ndev = 1;
    int32_t devices{};
    HcclComm comms{};
    EXPECT_EQ(HcclCommInitAllV2(ndev, &devices, &comms), HCCL_E_INTERNAL);
}

TEST_F(OpbaseTestV2, St_HcclBarrierV2_When_Creat_Memory_Stub_SUCCESS)
{
    // Mock objects and parameters
    uint64_t count = 10;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    aclrtStream stream = &count;
    MOCKER_CPP(&CommunicatorImpl::CreateBarrierMemory)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcclBarrierV2(comm, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, St_HcclBarrierV2_When_Creat_Memory_Fail_Return_HCCL_E_PTR)
{
    // Mock objects and parameters
    uint64_t count = 10;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    aclrtStream stream = &count;
    MOCKER_CPP(&CommunicatorImpl::CreateBarrierMemory)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_PTR));
    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcclBarrierV2(comm, stream);
    EXPECT_EQ(result, HCCL_E_PTR);
}

TEST_F(OpbaseTestV2, St_HcclGetInstRanksByNetLayerV2_When_InputValue_Expect_Return_HCCL_SUCCESS)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    uint32_t* ranks = nullptr;
    uint32_t rankNum = 0;
    MOCKER_CPP(&CommunicatorImpl::GetInstRanksByNetLayer)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HcclGetInstRanksByNetLayerV2(comm, 0, &ranks, &rankNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, St_HcclGetInstRanksByNetLayerV2_When_InValid_Expect_ReturnHCCL_NOT_FOUND)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    uint32_t* ranks = nullptr;
    uint32_t rankNum = 0;
    MOCKER_CPP(&CommunicatorImpl::GetInstRanksByNetLayer)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_PTR));
    HcclResult ret = HcclGetInstRanksByNetLayerV2(comm, 0, &ranks, &rankNum);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(OpbaseTestV2, Ut_HcclGetInstRanksByNetLayerV2_When_InValid_Expect_ReturnHCCL_NOT_FOUND)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    uint32_t* ranks = nullptr;
    uint32_t rankNum = 0;
    MOCKER_CPP(&CommunicatorImpl::GetInstRanksByNetLayer)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_PTR));
    HcclResult ret = HcclGetInstRanksByNetLayerV2(comm, 0, &ranks, &rankNum);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(OpbaseTestV2, Ut_HcclGetInstTopoTypeByNetLayerV2_When_InputValue_Expect_Return_HCCL_SUCCESS)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    uint32_t type = 0;
    MOCKER_CPP(&CommunicatorImpl::GetInstTopoTypeByNetLayer)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HcclGetInstTopoTypeByNetLayerV2(comm, 0, &type);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, St_HcclGetInstSizeListByNetLayerV2_When_InValid_Expect_ReturnHCCL_NOT_FOUND)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    uint32_t* instSizeList = nullptr;
    uint32_t listSize = 0;
    MOCKER_CPP(&CommunicatorImpl::GetInstSizeListByNetLayer)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_NOT_FOUND));
    HcclResult ret = HcclGetInstSizeListByNetLayerV2(comm, 0, &instSizeList, &listSize);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(OpbaseTestV2, St_HcclGetInstSizeListByNetLayerV2_When_Valid_Expect_ReturnHCCL_SUCCESS)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    uint32_t* instSizeList = nullptr;
    uint32_t listSize = 0;
    MOCKER_CPP(&CommunicatorImpl::GetInstSizeListByNetLayer)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HcclGetInstSizeListByNetLayerV2(comm, 0, &instSizeList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, St_HcclGetLinksV2_When_InputValue_Expect_Return_HCCL_SUCCESS)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    CommLink* linkList = nullptr;
    uint32_t listSize = 0;
    MOCKER_CPP(&CommunicatorImpl::GetLinks)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HcclGetLinksV2(comm, 0, 0, 1, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, St_HcclGetLinksV2_When_InValid_Expect_ReturnHCCL_NOT_FOUND)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    CommLink* linkList = nullptr;
    uint32_t listSize = 0;
    MOCKER_CPP(&CommunicatorImpl::GetLinks).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(HCCL_E_PTR));
    HcclResult ret = HcclGetLinksV2(comm, 0, 0, 1, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(OpbaseTestV2, St_HcclGetTopoInstsByLayer_When_InputValue_Expect_Return_HCCL_SUCCESS)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    uint32_t* topoInsts = nullptr;
    uint32_t topoInstNum = 0;
    MOCKER_CPP(&CommunicatorImpl::GetTopoInstsByLayer)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HcclGetTopoInstsByLayerV2(comm, 0, &topoInsts, &topoInstNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, St_HcclGetTopoInstsByLayer_When_inValid_Expect_ReturnHCCL_NOT_FOUND)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    uint32_t* topoInsts = nullptr;
    uint32_t topoInstNum = 0;
    MOCKER_CPP(&CommunicatorImpl::GetTopoInstsByLayer)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_PTR));
    HcclResult ret = HcclGetTopoInstsByLayerV2(comm, 0, &topoInsts, &topoInstNum);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(OpbaseTestV2, St_HcclGetTopoType_When_InputValue_Expect_Return_HCCL_SUCCESS)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    CommTopo topoType = COMM_TOPO_CLOS;
    uint32_t topoInstId = 0;
    MOCKER_CPP(&CommunicatorImpl::GetTopoType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HcclGetTopoTypeV2(comm, 0, topoInstId, &topoType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, St_HcclGetTopoType_When_inValid_Expect_ReturnHCCL_NOT_FOUND)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    CommTopo topoType;
    uint32_t topoInstId = 0;
    MOCKER_CPP(&CommunicatorImpl::GetTopoType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_PTR));
    HcclResult ret = HcclGetTopoTypeV2(comm, 0, topoInstId, &topoType);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(OpbaseTestV2, St_HcclGetRanksByTopoInst_When_InputValue_Expect_Return_HCCL_SUCCESS)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    uint32_t topoInstId = 0;
    uint32_t* ranks = nullptr;
    uint32_t rankNum = 0;
    MOCKER_CPP(&CommunicatorImpl::GetRanksByTopoInst)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HcclGetRanksByTopoInstV2(comm, 0, topoInstId, &ranks, &rankNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, St_HcclGetRanksByTopoInst_When_InValid_Expect_ReturnHCCL_NOT_FOUND)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    uint32_t topoInstId = 0;
    uint32_t* ranks = nullptr;
    uint32_t rankNum = 0;
    MOCKER_CPP(&CommunicatorImpl::GetRanksByTopoInst)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_PTR));
    HcclResult ret = HcclGetRanksByTopoInstV2(comm, 0, topoInstId, &ranks, &rankNum);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(OpbaseTestV2, St_WaitAllCommReady_When_InputValue_Expect_Return_HCCL_SUCCESS)
{
    HcclResult ret = WaitAllCommReady(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, St_HcclCommResPrepareV2_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    char* opname = "allreduce";
    void* opArgs = nullptr;
    HcclResult ret1 = HcclGetOpArgsV2(&opArgs);
    EXPECT_EQ(ret1, HCCL_SUCCESS);
    EXPECT_NE(nullptr, opArgs);
    HcclResult ret2 = HcclSetOpSrcDataTypeV2(opArgs, 2);
    EXPECT_EQ(ret2, HCCL_SUCCESS);
    HcclResult ret3 = HcclSetOpDstDataTypeV2(opArgs, 2);
    EXPECT_EQ(ret3, HCCL_SUCCESS);
    HcclResult ret4 = HcclSetOpReduceTypeV2(opArgs, 1);
    EXPECT_EQ(ret4, HCCL_SUCCESS);
    HcclResult ret5 = HcclSetOpCountV2(opArgs, 1024);
    EXPECT_EQ(ret5, HCCL_SUCCESS);
    HcclResult ret6 = HcclSetOpCommEngineV2(opArgs, 7);
    EXPECT_EQ(ret6, HCCL_SUCCESS);
    char algConfig[128] = "xxxxxxxxxxx";
    HcclResult ret7 = HcclSetOpAlgConfigV2(opArgs, algConfig);
    EXPECT_EQ(ret7, HCCL_SUCCESS);

    void* addr = nullptr;
    MOCKER_CPP(&HcclCommunicator::AllocCollOpResource).stubs().will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HcclCommResPrepareV2(comm, opname, opArgs, &addr);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclResult ret8 = HcclFreeOpArgsV2(opArgs);
    EXPECT_EQ(ret8, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, St_HcclSetOpArgs_When_Param_Error_Expect_ReturnIsHCCL_E_PARA)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    char* opname = "allreduce";
    void* opArgs = nullptr;
    HcclResult ret1 = HcclGetOpArgsV2(&opArgs);
    EXPECT_EQ(ret1, HCCL_SUCCESS);
    EXPECT_NE(nullptr, opArgs);
    HcclResult ret2 = HcclSetOpCountV2(opArgs, 0xFFFFFFFFF);
    EXPECT_EQ(ret2, HCCL_E_PARA);
    char algConfig[128] = "xxxxxxxxxxx";
    MOCKER(strcpy_s).stubs().will(returnValue(1));
    HcclResult ret3 = HcclSetOpAlgConfigV2(opArgs, algConfig);
    EXPECT_EQ(ret3, HCCL_E_PARA);
    HcclResult ret4 = HcclFreeOpArgsV2(opArgs);
    EXPECT_EQ(ret4, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, St_HcclDevMemAcquireV2_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    char memTag[] = "memTag";
    uint64_t size = 1024;
    void* addr = nullptr;
    bool newCreated = false;
    HcclResult ret = HcclDevMemAcquireV2(comm, memTag, &size, &addr, &newCreated);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    char* memTag1 = nullptr;
    HcclResult ret1 = HcclDevMemAcquireV2(comm, memTag, &size, &addr, &newCreated);
    EXPECT_EQ(ret1, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, St_HcclGetHcclBufferV2_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    MOCKER_CPP(&HcclCommunicator::GetLocalCclBuffer).stubs().will(returnValue(HCCL_SUCCESS));
    void* addr = nullptr;
    uint64_t size = 0;
    HcclResult ret = HcclGetHcclBufferV2(comm, &addr, &size);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, St_HcclGetRemoteIpcHcclBufV2_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    uint64_t remoteRank = 1;
    void* addr = nullptr;
    uint64_t size = 0;
    HcclResult ret = HcclGetRemoteIpcHcclBufV2(comm, remoteRank, &addr, &size);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(OpbaseTestV2, St_HcclGetAicpuOpStreamAndNotifyV2_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    int fakeNotify = 0;
    int fakeStream = 0;
    rtStream_t stream = static_cast<rtStream_t>(&fakeStream);
    u8 aicpuNotifyNum = 8;
    void* aicpuNotify = static_cast<void*>(&fakeNotify);
    MOCKER_CPP(&HcclCommunicator::GetAicpuOpStreamNotify)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HcclGetAicpuOpStreamAndNotifyV2(comm, &stream, aicpuNotifyNum, &aicpuNotify);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

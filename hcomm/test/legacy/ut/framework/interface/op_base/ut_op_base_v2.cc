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
#include <chrono>
#include <future>
#include <map>
#include <fstream>
#include <string>
#include <nlohmann/json.hpp>
#include "hccl_params_pub.h"
#include "hccl_common_v2.h"
#include "param_check_v2.h"
#include "comm_manager.h"
#include "binary_stream.h"
#include "snap_shot_parse.h"
#include "op_base_v2.h"
#include "orion_adapter_rts.h"
#include "rank_graph_builder_bridge.h"
#include "root_info_detect_bridge.h"
#include "hccl_comm.h"
#define private public
#include "hccl_communicator.h"
#include "communicator_impl.h"
#include "communicator_callback.h"
#include "task_abort_handler_v2.h"
#include "internal_exception.h"
#include "task_service.h"
#include "hostdpu/dpu_kernel_entrance.h"
#undef private

using namespace Hccl;
using namespace std;

namespace {
HcclResult g_getRootInfoResult = HCCL_SUCCESS;
HcclResult g_detectRankTableResult = HCCL_SUCCESS;

HcclResult GetRootInfoForUt(HcclRootInfo* rootInfo)
{
    if (g_getRootInfoResult != HCCL_SUCCESS) {
        return g_getRootInfoResult;
    }
    if (rootInfo == nullptr) {
        return HCCL_E_PTR;
    }
    *rootInfo = {};
    return HCCL_SUCCESS;
}

HcclResult DetectRankTableForUt(
    u32 nRanks, u32 rank, const HcclRootHandleV2& rootHandle, RankTableInfo& rankTable,
    RootInfoDetectBridge::DetectContext& detectContext)
{
    (void)nRanks;
    (void)rank;
    (void)rootHandle;
    (void)rankTable;
    detectContext.reset();
    return g_detectRankTableResult;
}
} // namespace

class OpbaseTestV2 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "OpbaseTestV2 tests set up." << std::endl;
        const RootInfoDetectBridge bridge = {GetRootInfoForUt, DetectRankTableForUt};
        ASSERT_EQ(RegisterRootInfoDetectBridge(bridge), HCCL_SUCCESS);
    }

    static void TearDownTestCase() { std::cout << "OpbaseTestV2 tests tear down." << std::endl; }

    virtual void SetUp()
    {
        std::cout << "A Test case in OpbaseTestV2 SetUP" << std::endl;
        g_getRootInfoResult = HCCL_SUCCESS;
        g_detectRankTableResult = HCCL_SUCCESS;
    }

    virtual void TearDown()
    {
        std::cout << "A Test case in OpbaseTestV2 TearDown" << std::endl;
        GlobalMockObject::verify();
    }
};

TEST_F(OpbaseTestV2, Ut_RankGraphBuilderBridgeAdoptRankGraph_WhenValid_ExpectSuccess)
{
    const RankGraphBuilderBridge* bridge = GetRankGraphBuilderBridge();
    ASSERT_NE(bridge, nullptr);
    ASSERT_NE(bridge->adoptRankGraph, nullptr);

    std::unique_ptr<RankGraph> uniqueRankGraph = std::make_unique<RankGraph>(0);
    std::shared_ptr<RankGraph> sharedRankGraph;
    EXPECT_EQ(bridge->adoptRankGraph(std::move(uniqueRankGraph), sharedRankGraph), HCCL_SUCCESS);
    EXPECT_EQ(uniqueRankGraph, nullptr);
    ASSERT_NE(sharedRankGraph, nullptr);
    EXPECT_EQ(sharedRankGraph->GetMyRank(), 0);
}

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
    MOCKER_CPP(static_cast<HcclResult (*)(const char*, u64, HcclDataType, const void*)>(&HcomCheckOpParamV2))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(&HcomCheckUserRankV2).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcclBatchSendRecvV2(sendRecvInfo.get(), itemNum, comm, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclBatchSendRecvV2_With_Log)
{
    EnvConfig::GetInstance().logCfg.entryLogEnable = CfgField<bool>({"HCCL_ENTRY_LOG_ENABLE", true, CastBin2Bool});
    EnvConfig::GetInstance().logCfg.entryLogEnable.isParsed = true;
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

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
    MOCKER_CPP(static_cast<HcclResult (*)(const char*, u64, HcclDataType, const void*)>(&HcomCheckOpParamV2))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(&HcomCheckUserRankV2).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcclBatchSendRecvV2(sendRecvInfo.get(), itemNum, comm, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);

    EnvConfig::GetInstance().logCfg.entryLogEnable.value = false;
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

TEST_F(OpbaseTestV2, HcclAlltoAllV2_With_Log)
{
    EnvConfig::GetInstance().logCfg.entryLogEnable = CfgField<bool>({"HCCL_ENTRY_LOG_ENABLE", true, CastBin2Bool});
    EnvConfig::GetInstance().logCfg.entryLogEnable.isParsed = true;
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

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

    EnvConfig::GetInstance().logCfg.entryLogEnable.value = false;
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

TEST_F(OpbaseTestV2, HcclAlltoAllVV2_With_Log)
{
    EnvConfig::GetInstance().logCfg.entryLogEnable = CfgField<bool>({"HCCL_ENTRY_LOG_ENABLE", true, CastBin2Bool});
    EnvConfig::GetInstance().logCfg.entryLogEnable.isParsed = true;
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

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

    EnvConfig::GetInstance().logCfg.entryLogEnable.value = false;
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
    HcclResult ret = HcclCommInitClusterInfoV2(clusterInfo.c_str(), rank, &comm);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(OpbaseTestV2, HcclCommInitClusterInfoV2)
{
    HcclComm comm;
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

static void MockSubCommConfigV2Dependencies(HcclGroupParamsV2& groupParamsV2Tem)
{
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));

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
}

static void PrepareSubCommConfigV2WorldManagerState(bool setIdleStatus = false)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> communicator = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclGroupParamsV2 groupParams;
    groupParams.pComm = communicator;
    std::map<std::string, HcclGroupParamsV2> groupMap = {{"hccl_world_group", groupParams}};
    HcclCommInfoV2& commInfo = CommManager::GetInstance(0).GetCommInfoV2();
    commInfo.hcclGroupMap = groupMap;
    commInfo.commParams = commParams;
    commInfo.isUsed = true;
    commInfo.pComm = communicator;
    if (setIdleStatus) {
        commInfo.status = DeviceStatus::DEVICE_IDLE;
    }
}

struct SubCommConfigV2TestContext {
    std::shared_ptr<Hccl::HcclCommunicator> communicator;
    HcclComm comm;
    HcclComm subComm;
    uint32_t rankNum;
    uint32_t rankIds;
    uint64_t subCommId;
    uint32_t subCommRankId;
    HcclCommConfig config;
};

static SubCommConfigV2TestContext
ConstructSubCommConfigV2TestContext(uint64_t subCommId, const std::string& worldgroup, uint32_t hcclBufferSize = 200)
{
    SubCommConfigV2TestContext context;
    context.communicator = std::make_shared<Hccl::HcclCommunicator>(Hccl::CommParams{});
    context.comm = static_cast<HcclComm>(context.communicator.get());
    context.subComm = nullptr;
    context.rankNum = 1;
    context.rankIds = 0;
    context.subCommId = subCommId;
    context.subCommRankId = 1;
    PrepareCommConfig(context.config, hcclBufferSize, worldgroup, 1, 0);
    return context;
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

TEST_F(OpbaseTestV2, HcclAlltoAllVCV2_With_Log)
{
    EnvConfig::GetInstance().logCfg.entryLogEnable = CfgField<bool>({"HCCL_ENTRY_LOG_ENABLE", true, CastBin2Bool});
    EnvConfig::GetInstance().logCfg.entryLogEnable.isParsed = true;
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

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

    EnvConfig::GetInstance().logCfg.entryLogEnable.value = false;
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

TEST_F(OpbaseTestV2, HcclReduceV2_Sum_ShouldPass_WhenValidParams_With_Log)
{
    EnvConfig::GetInstance().logCfg.entryLogEnable = CfgField<bool>({"HCCL_ENTRY_LOG_ENABLE", true, CastBin2Bool});
    EnvConfig::GetInstance().logCfg.entryLogEnable.isParsed = true;
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

    // Mock objects and parameters
    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    uint64_t count = 10;
    HcclDataType dataType = HCCL_DATA_TYPE_INT8;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    uint32_t root = 0;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
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

    EnvConfig::GetInstance().logCfg.entryLogEnable.value = false;
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

TEST_F(OpbaseTestV2, HcclAllReduceV2_Sum_ShouldPass_WhenValidParams_v2_With_Log)
{
    EnvConfig::GetInstance().logCfg.entryLogEnable = CfgField<bool>({"HCCL_ENTRY_LOG_ENABLE", true, CastBin2Bool});
    EnvConfig::GetInstance().logCfg.entryLogEnable.isParsed = true;
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

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

    EnvConfig::GetInstance().logCfg.entryLogEnable.value = false;
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

TEST_F(OpbaseTestV2, HcclBroadcastV2_ShouldReturnSuccess_WhenAllParamsValid_With_Log)
{
    EnvConfig::GetInstance().logCfg.entryLogEnable = CfgField<bool>({"HCCL_ENTRY_LOG_ENABLE", true, CastBin2Bool});
    EnvConfig::GetInstance().logCfg.entryLogEnable.isParsed = true;
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

    void* buf = nullptr;
    uint64_t count = 10;
    HcclDataType dataType = HCCL_DATA_TYPE_INT32;
    uint32_t root = 0;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclComm->pimpl->rankSize = 4;
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    aclrtStream stream = &count;
    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcclBroadcastV2(buf, count, dataType, root, comm, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);

    EnvConfig::GetInstance().logCfg.entryLogEnable.value = false;
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

TEST_F(OpbaseTestV2, HcclAllocComResourceByTilingV2_With_Log)
{
    EnvConfig::GetInstance().logCfg.entryLogEnable = CfgField<bool>({"HCCL_ENTRY_LOG_ENABLE", true, CastBin2Bool});
    EnvConfig::GetInstance().logCfg.entryLogEnable.isParsed = true;
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

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

    EnvConfig::GetInstance().logCfg.entryLogEnable.value = false;
}

TEST_F(OpbaseTestV2, Ut_HcclCommSuspendV2_When_InputComm_Expect_ReturnHCCL_E_NOT_SUPPORT)
{
    HcclComm comm;
    HcclResult ret = HcclCommSuspendV2(comm);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
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

TEST_F(OpbaseTestV2, HcclScatterV2_1)
{
    EnvConfig::GetInstance().logCfg.entryLogEnable = CfgField<bool>({"HCCL_ENTRY_LOG_ENABLE", true, CastBin2Bool});
    EnvConfig::GetInstance().logCfg.entryLogEnable.isParsed = true;
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

    // Arrange
    void* sendBuf = (void*)0x1000000;
    void* recvBuf = nullptr;
    uint64_t recvCount = 10;
    HcclDataType dataType = HCCL_DATA_TYPE_INT8;
    uint32_t root = 0;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclComm->pimpl->rankSize = 4;
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    aclrtStream stream = &recvCount;

    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcclScatterV2(sendBuf, recvBuf, recvCount, dataType, root, comm, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);

    EnvConfig::GetInstance().logCfg.entryLogEnable.value = false;
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

TEST_F(OpbaseTestV2, HcclAllGatherV2_With_Log)
{
    EnvConfig::GetInstance().logCfg.entryLogEnable = CfgField<bool>({"HCCL_ENTRY_LOG_ENABLE", true, CastBin2Bool});
    EnvConfig::GetInstance().logCfg.entryLogEnable.isParsed = true;
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

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

    EnvConfig::GetInstance().logCfg.entryLogEnable.value = false;
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
    hcclComm->pimpl->rankSize = 2;
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    aclrtStream stream = (void*)0x1000000;
    uint64_t count = 10;

    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcclSendV2(sendBuf, count, dataType, destRank, comm, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclSendV2_With_Log)
{
    EnvConfig::GetInstance().logCfg.entryLogEnable = CfgField<bool>({"HCCL_ENTRY_LOG_ENABLE", true, CastBin2Bool});
    EnvConfig::GetInstance().logCfg.entryLogEnable.isParsed = true;
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

    // Arrange
    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    uint64_t recvCount = 10;
    uint32_t destRank = 1;
    HcclDataType dataType = HCCL_DATA_TYPE_INT8;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclComm->pimpl->rankSize = 2;
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    aclrtStream stream = (void*)0x1000000;
    uint64_t count = 10;

    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcclSendV2(sendBuf, count, dataType, destRank, comm, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);

    EnvConfig::GetInstance().logCfg.entryLogEnable.value = false;
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
    hcclComm->pimpl->rankSize = 2;
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    aclrtStream stream = (void*)0x1000000;
    uint64_t count = 10;

    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcclRecvV2(recvBuf, count, dataType, srcRank, comm, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclRecvV2_With_Log)
{
    EnvConfig::GetInstance().logCfg.entryLogEnable = CfgField<bool>({"HCCL_ENTRY_LOG_ENABLE", true, CastBin2Bool});
    EnvConfig::GetInstance().logCfg.entryLogEnable.isParsed = true;
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

    // Arrange
    void* sendBuf = nullptr;
    void* recvBuf = nullptr;
    uint64_t recvCount = 10;
    uint32_t srcRank = 1;
    HcclDataType dataType = HCCL_DATA_TYPE_INT8;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclComm->pimpl->rankSize = 2;
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    aclrtStream stream = (void*)0x1000000;
    uint64_t count = 10;

    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result = HcclRecvV2(recvBuf, count, dataType, srcRank, comm, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);

    EnvConfig::GetInstance().logCfg.entryLogEnable.value = false;
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

TEST_F(OpbaseTestV2, HcclReduceScatterV2_With_Log)
{
    EnvConfig::GetInstance().logCfg.entryLogEnable = CfgField<bool>({"HCCL_ENTRY_LOG_ENABLE", true, CastBin2Bool});
    EnvConfig::GetInstance().logCfg.entryLogEnable.isParsed = true;
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));

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

    EnvConfig::GetInstance().logCfg.entryLogEnable.value = false;
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

    string worldgroup = "hccl_world_group";
    SnapShotBuf localBuff;
    strncpy(localBuff.snapshot.groupName, worldgroup.c_str(), worldgroup.size() + 1);
    localBuff.groupNum = 1;
    SubSnapshot sub;
    localBuff.subSnapshot.push_back(sub);
    MOCKER_CPP(&HcclCommunicator::IsCommReady).stubs().with().will(returnValue(true));
    MOCKER_CPP(&SnapShotParser::ParseSnapshotToLocalBuff)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(localBuff))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::RecoverSubComm)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::RecoverComm)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HcclSnapshotRecoverAllComms(clusterInfo.c_str(), changedInfo.c_str(), snapshot, snapshotBufSize);
    std::unique_lock<std::mutex> groupParaLock(CommManager::GetInstance(0).GetCommInfoV2().groupParamsLock);
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap.clear();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, SnapshotGenerate)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> pComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    pComm.get()->pimpl.get()->id = "hccl_world_group";
    pComm.get()->pimpl.get()->isWorldGroup = true;
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
    hcclComm_1.get()->pimpl.get()->id = "hccl_world_group";
    hcclComm_1.get()->pimpl.get()->isWorldGroup = true;
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
    hcclComm_1.get()->pimpl.get()->id = "hccl_world_group";
    hcclComm_1.get()->pimpl.get()->isWorldGroup = true;
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
    auto context = ConstructSubCommConfigV2TestContext(42, "hccl_world_group");
    HcclGroupParamsV2 groupParamsV2Tem;

    PrepareSubCommConfigV2WorldManagerState();
    MockSubCommConfigV2Dependencies(groupParamsV2Tem);
    context.communicator.get()->pimpl.get()->id = "hccl_world_group";
    context.communicator.get()->pimpl.get()->isWorldGroup = true;
    HcclResult ret = HcclCreateSubCommConfigV2(
        &context.comm, context.rankNum, &context.rankIds, context.subCommId, context.subCommRankId, &context.config,
        &context.subComm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclCreateSubCommConfigV2_IDEL)
{
    auto context = ConstructSubCommConfigV2TestContext(42, "hccl_world_group_1");
    HcclGroupParamsV2 groupParamsV2Tem;

    PrepareSubCommConfigV2WorldManagerState(true);
    MockSubCommConfigV2Dependencies(groupParamsV2Tem);
    context.communicator.get()->pimpl.get()->id = "hccl_world_group";
    context.communicator.get()->pimpl.get()->isWorldGroup = true;
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::RegisterAcceStateCallBack).stubs().will(ignoreReturnValue());
    DevType devType = DevType::DEV_TYPE_950;
    MOCKER(HrtGetDeviceType).stubs().with(mockcpp::any()).will(returnValue(devType));
    HcclResult ret = HcclCreateSubCommConfigV2(
        &context.comm, context.rankNum, &context.rankIds, context.subCommId, context.subCommRankId, &context.config,
        &context.subComm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclCreateSubCommConfigV2_Invalided_Config)
{
    auto context = ConstructSubCommConfigV2TestContext(43, "hccl_world_group_1", HCCL_COMM_BUFFSIZE_CONFIG_NOT_SET);
    HcclGroupParamsV2 groupParamsV2Tem;

    PrepareSubCommConfigV2WorldManagerState(true);
    MockSubCommConfigV2Dependencies(groupParamsV2Tem);
    context.communicator.get()->pimpl.get()->id = "hccl_world_group";
    context.communicator.get()->pimpl.get()->isWorldGroup = true;
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());
    HcclResult ret = HcclCreateSubCommConfigV2(
        &context.comm, context.rankNum, &context.rankIds, context.subCommId, context.subCommRankId, &context.config,
        &context.subComm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclCreateSubCommConfigV2_WhenParentIsChild_ExpectSuccess)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> parentComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    parentComm->pimpl.get()->id = "child_comm";
    parentComm->pimpl.get()->isWorldGroup = false;
    parentComm->pimpl.get()->commDepth = 1;
    parentComm->pimpl.get()->myRank = 0;
    HcclComm comm = static_cast<HcclComm>(parentComm.get());

    HcclComm subComm = nullptr;
    uint32_t rankNum = 1;
    uint32_t rankIds = 0;
    uint64_t subCommId = 42;
    uint32_t subCommRankId = 0;
    HcclCommConfig config;
    PrepareCommConfig(config, 200, "grandchild_comm", 1, 0);

    HcclGroupParamsV2 hcclGroupParamsV2;
    std::shared_ptr<Hccl::HcclCommunicator> dummyComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = dummyComm;
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap.clear();
    CommManager::GetInstance(0).GetCommInfoV2().pComm = dummyComm;
    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;

    HcclGroupParamsV2 groupParamsV2Tem;
    MockSubCommConfigV2Dependencies(groupParamsV2Tem);

    HcclResult ret = HcclCreateSubCommConfigV2(&comm, rankNum, &rankIds, subCommId, subCommRankId, &config, &subComm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ASSERT_NE(subComm, nullptr);
    HcclCommunicator* grandChildComm = static_cast<HcclCommunicator*>(subComm);
    EXPECT_EQ(grandChildComm->commParams.commDepth, 2);

    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap.clear();
    CommManager::GetInstance(0).GetCommInfoV2().pComm = nullptr;
}

TEST_F(OpbaseTestV2, HcclCreateSubCommConfigV2_WhenParentIsGrandChild_ExpectNotSupport)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> parentComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    parentComm->pimpl.get()->id = "grandchild_comm";
    parentComm->pimpl.get()->isWorldGroup = false;
    parentComm->pimpl.get()->commDepth = 2;
    HcclComm comm = static_cast<HcclComm>(parentComm.get());

    HcclComm subComm = nullptr;
    uint32_t rankNum = 1;
    uint32_t rankIds = 0;
    uint64_t subCommId = 42;
    uint32_t subCommRankId = 0;
    HcclCommConfig config;
    PrepareCommConfig(config, 200, "deeper_comm", 1, 0);

    HcclGroupParamsV2 hcclGroupParamsV2;
    std::shared_ptr<Hccl::HcclCommunicator> dummyComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = dummyComm;
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap.clear();
    CommManager::GetInstance(0).GetCommInfoV2().pComm = dummyComm;
    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;

    HcclResult ret = HcclCreateSubCommConfigV2(&comm, rankNum, &rankIds, subCommId, subCommRankId, &config, &subComm);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    EXPECT_EQ(subComm, nullptr);

    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap.clear();
    CommManager::GetInstance(0).GetCommInfoV2().pComm = nullptr;
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

    communicator.get()->pimpl.get()->id = "hccl_world_group";
    communicator.get()->pimpl.get()->isWorldGroup = true;
    MOCKER_CPP(&HcclCommunicator::IsCommReady).stubs().with().will(returnValue(true));
    HcclResult ret = HcclCommDestroyV2(hcom);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, HcclCommDestroyV2_2)
{
    MOCKER(aclrtGetDevice).stubs().will(returnValue(0));
    DevType devType = DevType::DEV_TYPE_910A;
    MOCKER(HrtGetDeviceType).stubs().will(returnValue(devType));

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

TEST_F(OpbaseTestV2, HcclCommDestroyV2_3)
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

    communicator.get()->pimpl.get()->id = "hccl_world_group";
    communicator.get()->pimpl.get()->isWorldGroup = true;
    communicator.get()->pimpl.get()->SetCommStatus(CommStatus::COMM_INUSE);
    MOCKER_CPP(&HcclCommunicator::IsCommReady).stubs().with().will(returnValue(true));
    HcclResult ret = HcclCommDestroyV2(hcom);
    EXPECT_EQ(ret, HCCL_E_AGAIN);
}

TEST_F(OpbaseTestV2, Ut_HcclCommInitClusterInfoConfigV2_When_InputInvalue_Expect_Return_HCCL_SUCCESS)
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

    // 打桩GetCommInfoV2。
    CommManager::GetInstance(0).GetCommInfoV2().status = DeviceStatus::DEVICE_IDLE;
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap.clear();
    CommManager::GetInstance(0).GetCommInfoV2().pComm = nullptr;

    HcclCommConfig config;
    string worldgroup = "hccl_world_group";
    PrepareCommConfig(config, 200, worldgroup, 1, 0);
    HcclComm comm;

    MOCKER_CPP(&HcclCommunicator::Init, HcclResult(HcclCommunicator::*)(const std::string&))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());
    auto ret = HcclCommInitClusterInfoConfigV2(clusterInfo, rank, &config, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, Ut_HcclCommInitClusterInfoConfigV2_When_InputValue_Expect_Return_HCCL_SUCCESS)
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
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    CommManager::GetInstance(0).GetCommInfoV2().status = DeviceStatus::DEVICE_RECOVERED;
    CommManager::GetInstance(0).GetCommInfoV2().pComm->pimpl.get()->id = worldgroup;

    MOCKER_CPP(&HcclCommunicator::Init, HcclResult(HcclCommunicator::*)(const std::string&))
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());
    auto ret = HcclCommInitClusterInfoConfigV2(clusterInfo, rank, &config, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, Ut_HcclAllGatherVV2_When_Normal_Expect_Success)
{
    constexpr u64 FAKE_RANK_SIZE = 2;
    void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_INT32;
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    communicator->pimpl->myRank = 0;
    communicator->pimpl->rankSize = FAKE_RANK_SIZE;
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

TEST_F(OpbaseTestV2, Ut_HcclAllGatherVV2_When_DipHas0_Expect_Success)
{
    constexpr u64 FAKE_RANK_SIZE = 2;
    void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_INT32;
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    communicator->pimpl->myRank = 0;
    communicator->pimpl->rankSize = FAKE_RANK_SIZE;
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

TEST_F(OpbaseTestV2, Ut_HcclAllGatherVV2_When_OutputEq0_Expect_Success)
{
    constexpr u64 FAKE_RANK_SIZE = 2;
    void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_INT32;
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    communicator->pimpl->myRank = 0;
    communicator->pimpl->rankSize = FAKE_RANK_SIZE;
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

TEST_F(OpbaseTestV2, Ut_HcclAllGatherVV2_When_CountNotFixCounts_Expect_Error)
{
    constexpr u64 FAKE_RANK_SIZE = 2;
    void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_INT32;
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    communicator->pimpl->myRank = 0;
    communicator->pimpl->rankSize = FAKE_RANK_SIZE;
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

TEST_F(OpbaseTestV2, Ut_HcclAllGatherVV2_When_CountTooLarge_Expect_Error)
{
    constexpr u64 FAKE_RANK_SIZE = 2;
    void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_INT32;
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    communicator->pimpl->myRank = 0;
    communicator->pimpl->rankSize = FAKE_RANK_SIZE;
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

TEST_F(OpbaseTestV2, Ut_HcclAllGatherVV2_When_DatatypeNotSurport_Expect_Error)
{
    constexpr u64 FAKE_RANK_SIZE = 2;
    void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_RESERVED;
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    communicator->pimpl->myRank = 0;
    communicator->pimpl->rankSize = FAKE_RANK_SIZE;
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

TEST_F(OpbaseTestV2, Ut_HcclAllGatherVV2_When_SingleRank)
{
    constexpr u64 FAKE_RANK_SIZE = 1;
    void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_INT32;
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    communicator->pimpl->myRank = 0;
    communicator->pimpl->rankSize = FAKE_RANK_SIZE;
    HcclComm comm = static_cast<HcclComm>(communicator.get());
    void* sendBuf = FAKE_PTR;
    u64 sendCount = 1;
    void* recvBuf = FAKE_PTR;
    u64 recvCounts[FAKE_RANK_SIZE] = {1};
    u64 recvDispls[FAKE_RANK_SIZE] = {1};
    HcclDataType sendType = FAKE_DATA_TYPE;
    int a = 1;
    aclrtStream stream = static_cast<aclrtStream>(&a);

    MOCKER_CPP(&HcclCommunicator::LoadOpbasedCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    // 回退： 空count
    sendBuf = FAKE_PTR;
    sendCount = 0;
    recvBuf = FAKE_PTR;
    HcclResult result = HcclAllGatherVV2(sendBuf, sendCount, recvBuf, &recvCounts, &recvDispls, sendType, comm, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);
    // 空sendcount
    sendBuf = nullptr;
    sendCount = 1;
    recvBuf = FAKE_PTR;
    result = HcclAllGatherVV2(sendBuf, sendCount, recvBuf, &recvCounts, &recvDispls, sendType, comm, stream);
    EXPECT_EQ(result, HCCL_E_PTR);
    // 正常回退
    sendBuf = FAKE_PTR;
    sendCount = 1;
    recvBuf = FAKE_PTR;
    result = HcclAllGatherVV2(sendBuf, sendCount, recvBuf, &recvCounts, &recvDispls, sendType, comm, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, Ut_HcclReduceScatterVV2_When_Normal_Expect_Success)
{
    constexpr u64 FAKE_RANK_SIZE = 2;
    void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_INT32;
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    communicator->pimpl->myRank = 0;
    communicator->pimpl->rankSize = FAKE_RANK_SIZE;
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

TEST_F(OpbaseTestV2, Ut_HcclReduceScatterVV2_When_DipHas0_Expect_Success)
{
    constexpr u64 FAKE_RANK_SIZE = 2;
    void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_INT32;
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    communicator->pimpl->myRank = 0;
    communicator->pimpl->rankSize = FAKE_RANK_SIZE;
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

TEST_F(OpbaseTestV2, Ut_HcclReduceScatterVV2_When_InuputCountEq0_Expect_Success)
{
    constexpr u64 FAKE_RANK_SIZE = 2;
    void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_INT32;
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    communicator->pimpl->myRank = 0;
    communicator->pimpl->rankSize = FAKE_RANK_SIZE;
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

TEST_F(OpbaseTestV2, Ut_HcclReduceScatterVV2_When_CountNotFixCounts_Expect_Error)
{
    constexpr u64 FAKE_RANK_SIZE = 2;
    void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_INT32;
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    communicator->pimpl->myRank = 0;
    communicator->pimpl->rankSize = FAKE_RANK_SIZE;
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

TEST_F(OpbaseTestV2, Ut_HcclReduceScatterVV2_When_CountTooLarge_Expect_Error)
{
    constexpr u64 FAKE_RANK_SIZE = 2;
    void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_INT32;
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    communicator->pimpl->myRank = 0;
    communicator->pimpl->rankSize = FAKE_RANK_SIZE;
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

TEST_F(OpbaseTestV2, Ut_HcclReduceScatterVV2_When_DatatypeNotSurport_Expect_Error)
{
    constexpr u64 FAKE_RANK_SIZE = 2;
    void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_RESERVED;
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    communicator->pimpl->myRank = 0;
    communicator->pimpl->rankSize = FAKE_RANK_SIZE;
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

TEST_F(OpbaseTestV2, Ut_HcclReduceScatterVV2_When_ReduceOpNotSurport_Expect_Error)
{
    constexpr u64 FAKE_RANK_SIZE = 2;
    void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_INT32;
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    communicator->pimpl->myRank = 0;
    communicator->pimpl->rankSize = FAKE_RANK_SIZE;
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

TEST_F(OpbaseTestV2, Ut_HcclReduceScatterVV2_When_SingleRank)
{
    constexpr u64 FAKE_RANK_SIZE = 1;
    void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_INT32;
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    communicator->pimpl->myRank = 0;
    communicator->pimpl->rankSize = FAKE_RANK_SIZE;
    HcclComm comm = static_cast<HcclComm>(communicator.get());
    void* sendBuf = FAKE_PTR;
    u64 sendCounts[FAKE_RANK_SIZE] = {1};
    u64 sendDispls[FAKE_RANK_SIZE] = {1};
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
    // 回退： 空count
    sendBuf = FAKE_PTR;
    recvCount = 0;
    recvBuf = FAKE_PTR;
    HcclResult result
        = HcclReduceScatterVV2(sendBuf, &sendCounts, &sendDispls, recvBuf, recvCount, dataType, op, comm, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);
    // 空sendcount
    sendBuf = nullptr;
    recvCount = 1;
    recvBuf = FAKE_PTR;
    result = HcclReduceScatterVV2(sendBuf, &sendCounts, &sendDispls, recvBuf, recvCount, dataType, op, comm, stream);
    EXPECT_EQ(result, HCCL_E_PTR);
    // 正常回退
    sendBuf = FAKE_PTR;
    recvCount = 1;
    recvBuf = FAKE_PTR;
    result = HcclReduceScatterVV2(sendBuf, &sendCounts, &sendDispls, recvBuf, recvCount, dataType, op, comm, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, Ut_HcclGetRootInfoV2_When_NoNeedInput_Expect_Return_HCCL_SUCCESS)
{
    HcclRootInfo rootInfo{};
    EXPECT_EQ(HcclGetRootInfoV2(&rootInfo), HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, Ut_HcclGetRootInfoV2_When_ProviderFails_Expect_Return_HCCL_E_INTERNAL)
{
    g_getRootInfoResult = HCCL_E_INTERNAL;
    HcclRootInfo rootInfo{};
    EXPECT_EQ(HcclGetRootInfoV2(&rootInfo), HCCL_E_INTERNAL);
}

TEST_F(OpbaseTestV2, Ut_HcclCommInitRootInfoV2_When_InputValue_Expect_Return_HCCL_SUCCESS)
{
    // when
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

TEST_F(OpbaseTestV2, Ut_HcclCommInitRootInfoV2_When_RankSizeInvalid_Expect_Return_HCCL_E_PARA)
{
    // when
    MOCKER(HrtGetDeviceType).stubs().with(mockcpp::any()).will(returnValue((DevType)DevType::DEV_TYPE_950));
    MOCKER(HrtGetDeviceCount).stubs().will(returnValue(1));
    MOCKER_CPP(&HcclCommunicator::Init, HcclResult(HcclCommunicator::*)(const RankTableInfo&))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());

    // then
    uint32_t nRanks{};
    HcclRootInfo rootInfo{};
    uint32_t rank{};
    HcclComm comm{};
    std::string identifier{};
    EXPECT_EQ(HcclCommInitRootInfoV2(nRanks, &rootInfo, rank, &comm, identifier), HCCL_E_PARA);
}

TEST_F(OpbaseTestV2, Ut_HcclCommInitRootInfoConfigV2_When_InputValue_Expect_Return_HCCL_SUCCESS)
{
    // when
    MOCKER(HrtGetDeviceType).stubs().with(mockcpp::any()).will(returnValue((DevType)DevType::DEV_TYPE_950));
    MOCKER(HrtGetDeviceCount).stubs().will(returnValue(1));
    MOCKER_CPP(&HcclCommunicator::Init, HcclResult(HcclCommunicator::*)(const RankTableInfo&))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());

    // then
    uint32_t nRanks{2};
    HcclRootInfo rootInfo{};
    uint32_t rank{};
    HcclComm comm{};
    HcclCommConfig config{};
    string worldgroup = "hccl_world_group_1";
    PrepareCommConfig(config, 200, worldgroup, 1, 0);
    EXPECT_EQ(HcclCommInitRootInfoConfigV2(nRanks, &rootInfo, rank, &config, &comm), HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, Ut_HcclCommInitRootInfoConfigV2_When_NotSetBufSize_Expect_Return_HCCL_SUCCESS)
{
    // when
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
    PrepareCommConfig(config, HCCL_COMM_BUFFSIZE_CONFIG_NOT_SET, worldgroup, 1, 0);
    EXPECT_EQ(HcclCommInitRootInfoConfigV2(nRanks, &rootInfo, rank, &config, &comm), HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, Ut_HcclCommInitRootInfoConfigV2_When_SetBufSize_Zero_Expect_Return_HCCL_E_PARA)
{
    uint32_t nRanks{2};
    HcclRootInfo rootInfo{};
    uint32_t rank{};
    HcclComm comm{};
    HcclCommConfig config{};
    string worldgroup = "hccl_world_group";
    PrepareCommConfig(config, 0, worldgroup, 1, 0);
    EXPECT_EQ(HcclCommInitRootInfoConfigV2(nRanks, &rootInfo, rank, &config, &comm), HCCL_E_PARA);
}

TEST_F(OpbaseTestV2, Ut_HcclCommInitRootInfoConfigV2_When_ProviderFails_Expect_Return_HCCL_E_INTERNAL)
{
    // when
    g_detectRankTableResult = HCCL_E_INTERNAL;
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

namespace {
// 捕获 RptInputErr 上报内容, 用于断言通信域重复创建时触发 EI0003 规范报错
std::string g_dupRptErrorCode;
std::vector<std::string> g_dupRptKeys;
std::vector<std::string> g_dupRptValues;

void stub_RptInputErr_capture(std::string error_code, std::vector<std::string> key, std::vector<std::string> value)
{
    g_dupRptErrorCode = error_code;
    g_dupRptKeys = key;
    g_dupRptValues = value;
}
} // namespace

// op_base_v2.cc 内部接口(未在头文件中声明): 其查重分支被 HcclCommInitRootInfoConfigV2 入口查重遮蔽,
// 公共接口无法触达, 此处声明用于直接测试
HcclResult CreateCommConfigRootInfo(
    uint32_t rank, const HcclCommConfig* config, const std::string& identifier, const RankTableInfo& ranktable,
    HcclComm* comm);

// 相同 identifier 重复创建通信域: CommInitRootInfo 查重命中, 上报 EI0003 并返回 HCCL_E_PARA
TEST_F(OpbaseTestV2, Ut_HcclCommInitRootInfoV2_When_DuplicateCreation_Expect_Return_HCCL_E_PARA)
{
    // when
    MOCKER(HrtGetDeviceType).stubs().with(mockcpp::any()).will(returnValue((DevType)DevType::DEV_TYPE_950));
    MOCKER(HrtGetDeviceCount).stubs().will(returnValue(1));
    MOCKER_CPP(&HcclCommunicator::Init, HcclResult(HcclCommunicator::*)(const RankTableInfo&))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());
    MOCKER(RptInputErr).stubs().will(invoke(stub_RptInputErr_capture));
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap.clear();
    CommManager::GetInstance(0).GetCommInfoV2().pComm = nullptr;

    // then: 首次创建成功, 不触发规范报错
    uint32_t nRanks = 1;
    HcclRootInfo rootInfo{};
    uint32_t rank = 0;
    HcclComm commFirst{};
    std::string identifier{};
    g_dupRptErrorCode.clear();
    EXPECT_EQ(HcclCommInitRootInfoV2(nRanks, &rootInfo, rank, &commFirst, identifier), HCCL_SUCCESS);
    EXPECT_TRUE(g_dupRptErrorCode.empty());

    // 相同 rootInfo(即相同 identifier)重复创建: 查重命中, 上报 EI0003 并返回 HCCL_E_PARA
    HcclComm commSecond{};
    EXPECT_EQ(HcclCommInitRootInfoV2(nRanks, &rootInfo, rank, &commSecond, identifier), HCCL_E_PARA);
    EXPECT_EQ(g_dupRptErrorCode, "EI0003");
    ASSERT_EQ(g_dupRptValues.size(), 4); // ccl_op/value/parameter/expect
    EXPECT_EQ(g_dupRptValues[0], "HcclCommInitRootInfo");
    EXPECT_EQ(g_dupRptValues[2], "identifier");

    // 清理现场, 避免残留通信域信息影响后续用例
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap.clear();
    CommManager::GetInstance(0).GetCommInfoV2().pComm = nullptr;
}

// 相同 identifier 重复创建通信域(带 config): HcclCommInitRootInfoConfigV2 查重命中,
// 上报 EI0003 并返回 HCCL_E_PARA
TEST_F(OpbaseTestV2, Ut_HcclCommInitRootInfoConfigV2_When_DuplicateCreation_Expect_Return_HCCL_E_PARA)
{
    // when
    MOCKER(HrtGetDeviceType).stubs().with(mockcpp::any()).will(returnValue((DevType)DevType::DEV_TYPE_950));
    MOCKER(HrtGetDeviceCount).stubs().will(returnValue(1));
    MOCKER_CPP(&HcclCommunicator::Init, HcclResult(HcclCommunicator::*)(const RankTableInfo&))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());
    MOCKER(RptInputErr).stubs().will(invoke(stub_RptInputErr_capture));
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap.clear();
    CommManager::GetInstance(0).GetCommInfoV2().pComm = nullptr;

    // then: 首次创建成功, 不触发规范报错
    uint32_t nRanks{2};
    HcclRootInfo rootInfo{};
    uint32_t rank{};
    HcclComm commFirst{};
    HcclCommConfig config{};
    string worldgroup = "ut_dup_rootinfo_config";
    PrepareCommConfig(config, 200, worldgroup, 1, 0);
    g_dupRptErrorCode.clear();
    EXPECT_EQ(HcclCommInitRootInfoConfigV2(nRanks, &rootInfo, rank, &config, &commFirst), HCCL_SUCCESS);
    EXPECT_TRUE(g_dupRptErrorCode.empty());

    // 相同 config(即相同 identifier)重复创建: 查重命中, 上报 EI0003 并返回 HCCL_E_PARA
    HcclComm commSecond{};
    EXPECT_EQ(HcclCommInitRootInfoConfigV2(nRanks, &rootInfo, rank, &config, &commSecond), HCCL_E_PARA);
    EXPECT_EQ(g_dupRptErrorCode, "EI0003");
    ASSERT_EQ(g_dupRptValues.size(), 4); // ccl_op/value/parameter/expect
    EXPECT_EQ(g_dupRptValues[0], "HcclCommInitRootInfo");
    EXPECT_EQ(g_dupRptValues[1], worldgroup);
    EXPECT_EQ(g_dupRptValues[2], "identifier");

    // 清理现场, 避免残留通信域信息影响后续用例
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap.clear();
    CommManager::GetInstance(0).GetCommInfoV2().pComm = nullptr;
}

// 相同 identifier 重复创建通信域: CreateCommConfigRootInfo 查重命中,
// 上报 EI0003 并返回 HCCL_E_PARA(该分支位于接口内部, 需预置 map 后直接调用)
TEST_F(OpbaseTestV2, Ut_CreateCommConfigRootInfo_When_DuplicateCreation_Expect_Return_HCCL_E_PARA)
{
    // 预置 identifier 已存在的通信域信息
    MOCKER(RptInputErr).stubs().will(invoke(stub_RptInputErr_capture));
    const string dupIdentifier = "ut_dup_createcommconfig_rootinfo";
    HcclGroupParamsV2 params{};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap[dupIdentifier] = params;

    uint32_t rank{};
    HcclCommConfig config{};
    HcclCommConfigInit(&config);
    HcclComm comm{};
    g_dupRptErrorCode.clear();
    EXPECT_EQ(CreateCommConfigRootInfo(rank, &config, dupIdentifier, RankTableInfo{}, &comm), HCCL_E_PARA);
    EXPECT_EQ(g_dupRptErrorCode, "EI0003");
    ASSERT_EQ(g_dupRptValues.size(), 4); // ccl_op/value/parameter/expect
    EXPECT_EQ(g_dupRptValues[0], "HcclCommInitRootInfo");
    EXPECT_EQ(g_dupRptValues[1], dupIdentifier);
    EXPECT_EQ(g_dupRptValues[2], "identifier");

    // 清理现场, 避免残留通信域信息影响后续用例
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap.clear();
}

TEST_F(OpbaseTestV2, Ut_HcclCommInitAllV2_When_InputValue_Expect_Return_HCCL_SUCCESS)
{
    // when
    MOCKER(HrtGetDeviceCount).stubs().will(returnValue(1));
    MOCKER(HrtGetDeviceType).stubs().with(mockcpp::any()).will(returnValue((DevType)DevType::DEV_TYPE_950));
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
    int32_t devices;
    HcclComm comms;
    EXPECT_EQ(HcclCommInitAllV2(ndev, &devices, &comms), HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, Ut_HcclCommInitAllV2_When_ProviderFails_Expect_Return_HCCL_E_INTERNAL)
{
    // when
    g_getRootInfoResult = HCCL_E_INTERNAL;
    MOCKER(HrtGetDeviceCount).stubs().will(returnValue(1));
    MOCKER(HrtGetDeviceType).stubs().with(mockcpp::any()).will(returnValue((DevType)DevType::DEV_TYPE_950));
    MOCKER(HrtGetDeviceType).stubs().with(mockcpp::any()).will(returnValue((DevType)DevType::DEV_TYPE_950));
    MOCKER(HrtGetDeviceCount).stubs().will(returnValue(1));
    MOCKER_CPP(&HcclCommunicator::Init, HcclResult(HcclCommunicator::*)(const RankTableInfo&))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());

    // then
    uint32_t ndev = 1;
    int32_t devices;
    HcclComm comms;
    EXPECT_EQ(HcclCommInitAllV2(ndev, &devices, &comms), HCCL_E_INTERNAL);
}

TEST_F(OpbaseTestV2, Ut_HcclBarrierV2_When_Creat_Memory_Stub_SUCCESS)
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

TEST_F(OpbaseTestV2, Ut_HcclBarrierV2_When_Creat_Memory_Fail_Return_HCCL_E_PTR)
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

TEST_F(OpbaseTestV2, Ut_HcclGetInstRanksByNetLayerV2_When_InputValue_Expect_Return_HCCL_SUCCESS)
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

TEST_F(OpbaseTestV2, Ut_HcclGetInstTopoTypeByNetLayerV2_When_InValid_Expect_ReturnHCCL_NOT_FOUND)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    uint32_t type = 0;
    MOCKER_CPP(&CommunicatorImpl::GetInstTopoTypeByNetLayer)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_PTR));
    HcclResult ret = HcclGetInstTopoTypeByNetLayerV2(comm, 0, &type);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(OpbaseTestV2, Ut_HcclGetInstSizeListByNetLayerV2_When_InValid_Expect_ReturnHCCL_NOT_FOUND)
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

TEST_F(OpbaseTestV2, Ut_HcclGetInstSizeListByNetLayerV2_When_Valid_Expect_ReturnHCCL_SUCCESS)
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

TEST_F(OpbaseTestV2, Ut_HcclGetLinksV2_When_InputValue_Expect_Return_HCCL_SUCCESS)
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

TEST_F(OpbaseTestV2, Ut_HcclGetLinksV2_When_InValid_Expect_ReturnHCCL_NOT_FOUND)
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

TEST_F(OpbaseTestV2, Ut_HcclGetTopoInstsByLayer_When_InputValue_Expect_Return_HCCL_SUCCESS)
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

TEST_F(OpbaseTestV2, Ut_HcclGetTopoInstsByLayer_When_inValid_Expect_ReturnHCCL_NOT_FOUND)
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

TEST_F(OpbaseTestV2, Ut_HcclGetTopoType_When_InputValue_Expect_Return_HCCL_SUCCESS)
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

TEST_F(OpbaseTestV2, Ut_HcclGetTopoType_When_inValid_Expect_ReturnHCCL_NOT_FOUND)
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

TEST_F(OpbaseTestV2, Ut_HcclGetRanksByTopoInst_When_InputValue_Expect_Return_HCCL_SUCCESS)
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

TEST_F(OpbaseTestV2, Ut_HcclGetRanksByTopoInst_When_InValid_Expect_ReturnHCCL_NOT_FOUND)
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

TEST_F(OpbaseTestV2, Ut_HcclCommResPrepareV2_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
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

TEST_F(OpbaseTestV2, Ut_HcclSetOpArgs_When_Param_Error_Expect_ReturnIsHCCL_E_PARA)
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

    HcclResult ret4 = HcclSetOpSrcDataTypeV2(opArgs, 100);
    EXPECT_EQ(ret4, HCCL_E_PARA);

    HcclResult ret5 = HcclSetOpDstDataTypeV2(opArgs, 100);
    EXPECT_EQ(ret5, HCCL_E_PARA);

    HcclResult ret6 = HcclSetOpReduceTypeV2(opArgs, 100);
    EXPECT_EQ(ret6, HCCL_E_PARA);

    HcclResult ret7 = HcclFreeOpArgsV2(opArgs);
    EXPECT_EQ(ret7, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, Ut_HcclDevMemAcquireV2_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
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

TEST_F(OpbaseTestV2, Ut_HcclGetHcclBufferV2_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
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

TEST_F(OpbaseTestV2, Ut_HcclGetRemoteIpcHcclBufV2_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
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

TEST_F(OpbaseTestV2, Ut_HcclGetAicpuOpStreamAndNotifyV2_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
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

TEST_F(OpbaseTestV2, Ut_HcclGetHeterogModeV2_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(communicator.get());

    HcclHeterogMode mode{};
    HcclResult ret = HcclGetHeterogModeV2(comm, &mode);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(mode, HcclHeterogMode::HCCL_HETEROG_MODE_HOMOGENEOUS);
}

TEST_F(OpbaseTestV2, Ut_HcclRankGraphGetEndpointNumV2_When_Valid_Expect_ReturnHCCL_SUCCESS)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());

    uint32_t num = 0;
    uint32_t layer = 0;
    uint32_t topoInstId = 0;
    MOCKER_CPP(&CommunicatorImpl::GetEndpointNum)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HcclRankGraphGetEndpointNumV2(comm, layer, topoInstId, &num);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OpbaseTestV2, Ut_HcclRankGraphGetEndpointNumV2_When_InValid_Expect_ReturnHCCL_E_NOT_FOUND)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());

    uint32_t num = 0;
    uint32_t layer = 0;
    uint32_t topoInstId = 0;
    MOCKER_CPP(&CommunicatorImpl::GetEndpointNum)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_NOT_FOUND));
    HcclResult ret = HcclRankGraphGetEndpointNumV2(comm, layer, topoInstId, &num);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(OpbaseTestV2, Ut_HcclRankGraphGetEndpointDescV2_When_Valid_Expect_ReturnHCCL_SUCCESS)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());

    uint32_t descNum = 1;
    uint32_t layer = 0;
    uint32_t topoInstId = 0;
    EndpointDesc* endPointDesc = new EndpointDesc[descNum];
    MOCKER_CPP(&CommunicatorImpl::GetEndpointDesc)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HcclRankGraphGetEndpointDescV2(comm, layer, topoInstId, &descNum, endPointDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    delete[] endPointDesc;
}

// HcclTaskRegisterV2 Tests

TEST_F(OpbaseTestV2, Ut_HcclTaskRegisterV2_When_CommIdNotFound_Expect_ReturnHCCL_E_NOT_FOUND)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    g_taskServiceMap.clear();

    HcclResult ret = HcclTaskRegisterV2(comm, "test_tag", [](uint64_t, int32_t) -> int32_t {
        return 0;
    });
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(OpbaseTestV2, Ut_HcclTaskRegisterV2_When_DeviceIdNotFound_Expect_ReturnHCCL_E_NOT_FOUND)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    g_taskServiceMap.clear();

    std::string commId = hcclComm->GetId();
    auto taskService = std::make_unique<Hccl::TaskService>();
    g_taskServiceMap[commId][9999] = std::move(taskService);

    HcclResult ret = HcclTaskRegisterV2(comm, "test_tag", [](uint64_t, int32_t) -> int32_t {
        return 0;
    });
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
    g_taskServiceMap.clear();
}

// HcclTaskRegisterProfV2 Tests

TEST_F(OpbaseTestV2, Ut_HcclTaskRegisterProfV2_When_CommIdNotFound_Expect_ReturnHCCL_E_NOT_FOUND)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    g_taskServiceMap.clear();

    Hccl::ProfCallbackTemplate callback = [](const Hccl::TaskParam&, uint64_t) -> HcclResult {
        return HCCL_SUCCESS;
    };
    HcclResult ret = HcclTaskRegisterProfV2(comm, callback);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(OpbaseTestV2, Ut_HcclTaskRegisterProfV2_When_DeviceIdNotFound_Expect_ReturnHCCL_E_NOT_FOUND)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    g_taskServiceMap.clear();

    std::string commId = hcclComm->GetId();
    auto taskService = std::make_unique<Hccl::TaskService>();
    g_taskServiceMap[commId][9999] = std::move(taskService);

    Hccl::ProfCallbackTemplate callback = [](const Hccl::TaskParam&, uint64_t) -> HcclResult {
        return HCCL_SUCCESS;
    };
    HcclResult ret = HcclTaskRegisterProfV2(comm, callback);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
    g_taskServiceMap.clear();
}

// HcclTaskReportRegisterV2 Tests

TEST_F(OpbaseTestV2, Ut_HcclTaskReportRegisterV2_When_CommIsNull_Expect_ReturnHCCL_E_PTR)
{
    Hccl::ReportCallbackTemplate callback = []() -> HcclResult {
        return HCCL_SUCCESS;
    };
    HcclResult ret = HcclTaskReportRegisterV2(nullptr, callback);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(OpbaseTestV2, Ut_HcclTaskReportRegisterV2_When_CommIdNotFound_Expect_ReturnHCCL_E_NOT_FOUND)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    g_taskServiceMap.clear();

    Hccl::ReportCallbackTemplate callback = []() -> HcclResult {
        return HCCL_SUCCESS;
    };
    HcclResult ret = HcclTaskReportRegisterV2(comm, callback);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(OpbaseTestV2, Ut_HcclTaskReportRegisterV2_When_DeviceIdNotFound_Expect_ReturnHCCL_E_NOT_FOUND)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    g_taskServiceMap.clear();

    std::string commId = hcclComm->GetId();
    auto taskService = std::make_unique<Hccl::TaskService>();
    g_taskServiceMap[commId][9999] = std::move(taskService);

    Hccl::ReportCallbackTemplate callback = []() -> HcclResult {
        return HCCL_SUCCESS;
    };
    HcclResult ret = HcclTaskReportRegisterV2(comm, callback);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
    g_taskServiceMap.clear();
}

TEST_F(OpbaseTestV2, Ut_HcclTaskReportRegisterV2_When_Normal_Expect_ReturnHCCL_SUCCESS)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    g_taskServiceMap.clear();

    std::string commId = hcclComm->GetId();
    s32 deviceId = hcclComm->GetDeviceLogicId();
    auto taskService = std::make_unique<Hccl::TaskService>();
    g_taskServiceMap[commId][deviceId] = std::move(taskService);

    Hccl::ReportCallbackTemplate callback = []() -> HcclResult {
        return HCCL_SUCCESS;
    };
    HcclResult ret = HcclTaskReportRegisterV2(comm, callback);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    g_taskServiceMap.clear();
}

// HcclTaskUnRegisterV2 Tests

TEST_F(OpbaseTestV2, Ut_HcclTaskUnRegisterV2_When_CommIdNotFound_Expect_ReturnHCCL_E_NOT_FOUND)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    g_taskServiceMap.clear();

    HcclResult ret = HcclTaskUnRegisterV2(comm, "test_tag");
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(OpbaseTestV2, Ut_HcclTaskUnRegisterV2_When_DeviceIdNotFound_Expect_ReturnHCCL_E_NOT_FOUND)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    g_taskServiceMap.clear();

    std::string commId = hcclComm->GetId();
    auto taskService = std::make_unique<Hccl::TaskService>();
    g_taskServiceMap[commId][9999] = std::move(taskService);

    HcclResult ret = HcclTaskUnRegisterV2(comm, "test_tag");
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
    g_taskServiceMap.clear();
}

TEST_F(OpbaseTestV2, Ut_HcclRankGraphGetEndpointDescV2_When_InValid_Expect_ReturnHCCL_E_NOT_FOUND)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());

    uint32_t descNum = 1;
    uint32_t layer = 0;
    uint32_t topoInstId = 0;
    EndpointDesc* endPointDesc = new EndpointDesc[descNum];
    MOCKER_CPP(&CommunicatorImpl::GetEndpointDesc)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_E_NOT_FOUND));
    HcclResult ret = HcclRankGraphGetEndpointDescV2(comm, layer, topoInstId, &descNum, endPointDesc);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
    delete[] endPointDesc;
}

TEST_F(OpbaseTestV2, Ut_HcclRankGraphGetEndpointInfoV2_When_Valid_Expect_ReturnHCCL_SUCCESS)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());

    uint32_t descNum = 1;
    uint32_t layer = 0;
    uint32_t topoInstId = 0;
    EndpointDesc* endPointDesc = new EndpointDesc[descNum];
    MOCKER_CPP(&CommunicatorImpl::GetEndpointDesc)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HcclRankGraphGetEndpointDescV2(comm, layer, topoInstId, &descNum, endPointDesc);
    MOCKER_CPP(&CommunicatorImpl::GetEndpointInfo)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    uint32_t infoLen = sizeof(EndpointAttrBwCoeff);
    EndpointAttrBwCoeff bwCoeff{};
    ret = HcclRankGraphGetEndpointInfoV2(comm, 0, endPointDesc, ENDPOINT_ATTR_BW_COEFF, infoLen, &bwCoeff);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    delete[] endPointDesc;
}

TEST_F(OpbaseTestV2, Ut_WaitAllCommReady_When_Timeout_Expect_Return_HCCL_E_TIMEOUT)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm_1 = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm_1;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm_1;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;

    MOCKER(HrtSetDevice).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&HcclCommunicator::IsCommReady).stubs().with().will(returnValue(false));
    MOCKER_CPP(&std::chrono::steady_clock::now)
        .stubs()
        .will(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(0))))
        .then(returnValue(std::chrono::time_point<std::chrono::steady_clock>(std::chrono::seconds(11))));

    HcclResult ret = WaitAllCommReady(0);
    EXPECT_EQ(ret, HCCL_E_TIMEOUT);

    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap.clear();
    CommManager::GetInstance(0).GetCommInfoV2().pComm = nullptr;
}

TEST_F(OpbaseTestV2, Ut_HcclAllocComResourceByTilingV2_When_CommIsNull_Expect_ReturnHCCL_E_PTR)
{
    int dd = 0;
    void* stream = static_cast<void*>(&dd);
    void* mc2Tiling = static_cast<void*>(&dd);
    void* commContext = static_cast<void*>(&dd);
    HcclResult ret = HcclAllocComResourceByTilingV2(nullptr, stream, mc2Tiling, &commContext);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(OpbaseTestV2, Ut_HcclAllocComResourceByTilingV2_When_StreamIsNull_Expect_ReturnHCCL_E_PTR)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    int dd = 0;
    void* mc2Tiling = static_cast<void*>(&dd);
    void* commContext = static_cast<void*>(&dd);
    HcclResult ret = HcclAllocComResourceByTilingV2(comm, nullptr, mc2Tiling, &commContext);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(OpbaseTestV2, Ut_HcclAllocComResourceByTilingV2_When_Mc2TilingIsNull_Expect_ReturnHCCL_E_PTR)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    int dd = 0;
    void* stream = static_cast<void*>(&dd);
    void* commContext = static_cast<void*>(&dd);
    HcclResult ret = HcclAllocComResourceByTilingV2(comm, stream, nullptr, &commContext);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(OpbaseTestV2, Ut_HcclAllocComResourceByTilingV2_When_CommContextIsNull_Expect_ReturnHCCL_E_PTR)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    int dd = 0;
    void* stream = static_cast<void*>(&dd);
    void* mc2Tiling = static_cast<void*>(&dd);
    HcclResult ret = HcclAllocComResourceByTilingV2(comm, stream, mc2Tiling, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(OpbaseTestV2, Ut_HcclGetRankGraphV2_When_RankGraphIsValid_Expect_ReturnHCCL_SUCCESS)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    hcclComm->pimpl->rankGraph = std::make_shared<Hccl::RankGraph>(0);
    void* rankGraph = nullptr;
    HcclResult ret = HcclGetRankGraphV2(&comm, &rankGraph);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(rankGraph, nullptr);
}

TEST_F(OpbaseTestV2, Ut_HcclCommWorkingDevNicSetV2_When_Call_Expect_ReturnHCCL_E_NOT_SUPPORT)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    uint32_t ranks = 0;
    bool useBackup = false;
    HcclResult ret = HcclCommWorkingDevNicSetV2(comm, &ranks, &useBackup, 1);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(OpbaseTestV2, Ut_HcclCommSetMemoryRangeV2_When_Call_Expect_ReturnHCCL_E_NOT_SUPPORT)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    int dd = 0;
    void* baseVirPtr = static_cast<void*>(&dd);
    HcclResult ret = HcclCommSetMemoryRangeV2(comm, baseVirPtr, 1024, 64, 0);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(OpbaseTestV2, Ut_HcclCommUnsetMemoryRangeV2_When_Call_Expect_ReturnHCCL_E_NOT_SUPPORT)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    int dd = 0;
    void* baseVirPtr = static_cast<void*>(&dd);
    HcclResult ret = HcclCommUnsetMemoryRangeV2(comm, baseVirPtr);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(OpbaseTestV2, Ut_HcclCommActivateCommMemoryV2_When_Call_Expect_ReturnHCCL_E_NOT_SUPPORT)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    int dd = 0;
    void* virPtr = static_cast<void*>(&dd);
    void* handle = static_cast<void*>(&dd);
    HcclResult ret = HcclCommActivateCommMemoryV2(comm, virPtr, 1024, 0, handle, 0);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(OpbaseTestV2, Ut_HcclCommDeactivateCommMemoryV2_When_Call_Expect_ReturnHCCL_E_NOT_SUPPORT)
{
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    int dd = 0;
    void* virPtr = static_cast<void*>(&dd);
    HcclResult ret = HcclCommDeactivateCommMemoryV2(comm, virPtr);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

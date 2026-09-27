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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#include <stdio.h>
#include <algorithm>
#include <fstream>
#include <list>
#include <vector>
#include <string>
#include <securec.h>
#include <hccl/hccl_types.h>
#include "hccl/base.h"
#include "orion_adapter_rts.h"
#include "hccl_communicator.h"
#include "comm_manager.h"
#include "hcom_v2.h"
#include "param_check_v2.h"
#include "log.h"
#include "hccl_common_v2.h"
#include "communicator_impl.h"
#include "ccu_dev_mgr.h"
#include "mc2_type.h"

using namespace std;
using namespace Hccl;

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

class HcomutCommManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HcomTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "HcomTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in HcomTest SetUp" << std::endl; }

    virtual void TearDown()
    {
        std::cout << "A Test case in HcomTest TearDown" << std::endl;
        GlobalMockObject::verify();
    }
};

TEST_F(HcomutCommManagerTest, ut_V2_gradient_Manage_Split_API)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    commParams.rankSize = 5;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;
    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));

    HcclGroupParamsV2 params;
    params.pComm = hcclComm;
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap["hccl_world_group"] = params;
    int ret;

    u32 ranksize = 0;
    ret = HcomGetRankSizeV2(NULL, &ranksize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 groupRank = 0;
    u32 worldRank = 0;
    ret = HcomGetWorldRankFromGroupRankV2(NULL, groupRank, &worldRank);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcomGetGroupRankFromWorldRankV2(worldRank, NULL, &groupRank);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcomutCommManagerTest, ut_hcomv2_backlog_group)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    commParams.rankSize = 5;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;
    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    CommManager::GetInstance(0).GetCommInfoV2().devId = 0;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));

    HcclGroupParamsV2 params;
    params.pComm = hcclComm;
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap["hccl_world_group"] = params;
    const u32 groupRanksNum = 4;
    std::string strGroup = "group1";
    std::vector<u32> rankIds = {0, 1, 2, 3};
    int ret = HCCL_SUCCESS;
    MOCKER_CPP(static_cast<HcclResult (CommunicatorImpl::*)(
                   const CommParams& subCommParams, const std::vector<u32>& rankIds, CommunicatorImpl* subCommImpl)>(
                   &CommunicatorImpl::CreateSubComm))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());
    ret = HcomCreateGroupImplV2(strGroup, groupRanksNum, rankIds);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcomDestroyGroupImplV2(strGroup);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomCreateGroupImplV2(strGroup, groupRanksNum, rankIds);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // GROUP 已存在
    ret = HcomCreateGroupImplV2(strGroup, groupRanksNum, rankIds);
    EXPECT_EQ(ret, HCCL_E_PARA);

    s32 deviceId = 0;
    char* identify = "0";
    s32 rankSize = 1;
    s32 rank = atoi(identify);
    u32 devLogicId = 0;
    HrtSetDevice(devLogicId);

    nlohmann::json rank_table = rank_table_950_1server_8rank;
    char file_name_t[] = "./st_hcom_test_rank_table_1server_8rank_950.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    // GROUP 已存在
    ret = HcomCreateGroupImplV2(strGroup, groupRanksNum, rankIds);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = HcomDestroyGroupImplV2(strGroup);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomDestroyV2();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcomutCommManagerTest, Ut_GetFileSize_When_InvalidFilePath_Expect_Exception)
{
    char file_name_t[] = "./test/orion/testranktable.json";
    int ret = GetFileSize(file_name_t);
    EXPECT_EQ(ret, 0);
}

TEST_F(HcomutCommManagerTest, Ut_GetFileSize_When_RightFilePath_Expect_Sucess)
{
    std::string file_name_t{HCOMM_CODE_ROOT_DIR "/test/legacy/ut/framework/interface/hcom_comm/testranktable.json"};
    u64 ranktablesize = 739;
    int ret = GetFileSize(file_name_t.c_str());
    EXPECT_EQ(ret, ranktablesize);
}

TEST_F(HcomutCommManagerTest, Ut_HcomInitByFileV2_When_InvalidRanktableSize_Expect_Exception)
{
    char* identify = "0";
    std::string file_name_t{HCOMM_CODE_ROOT_DIR "/test/legacy/ut/framework/interface/hcom_comm/testranktable.json"};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap.clear();
    CommManager::GetInstance(0).GetCommInfoV2().pComm = nullptr;
    MOCKER(GetFileSize).stubs().will(returnValue(RANKTABLE_FILE_MAX_SIZE + 1));
    int ret = HcomInitByFileV2(file_name_t.c_str(), identify);
    EXPECT_EQ(ret, HCCL_E_OPEN_FILE_FAILURE);
}

TEST_F(HcomutCommManagerTest, ut_v2_comm_manager_max_device_id_test)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    commParams.rankSize = 5;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(::MAX_MODULE_DEVICE_NUM + 1);
    MOCKER(aclrtGetDevice).stubs().will(returnValue(1));
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;
    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    EXPECT_NO_THROW(GetCommInfoV2());
}

TEST_F(HcomutCommManagerTest, ut_v2_comm_manager_CallSingletons_Exception_test1)
{
    MOCKER(HrtGetDevice).stubs().will(returnValue(::MAX_MODULE_DEVICE_NUM + 1));
    EXPECT_EQ(HcclResult::HCCL_E_RUNTIME, CallSingletons());
}

TEST_F(HcomutCommManagerTest, ut_v2_comm_manager_CallSingletons_Exception_test2)
{
    MOCKER(aclrtGetDevice).stubs().will(returnValue(1));
    EXPECT_EQ(HcclResult::HCCL_E_RUNTIME, CallSingletons());
}

TEST_F(HcomutCommManagerTest, ut_v2_comm_manager_CallSingletons_Exception_test3)
{
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    char targetChipVer[CHIP_VERSION_MAX_LEN] = "Ascend910B1";
    MOCKER(HrtGetSocVer)
        .stubs()
        .with(outBoundP(&targetChipVer[0], sizeof(targetChipVer)))
        .will(returnValue(RT_ERROR_NONE));
    EXPECT_NO_THROW(CallSingletons());
}

TEST_F(HcomutCommManagerTest, ut_v2_comm_manager_PrintChannelInfo_test1)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    commParams.rankSize = 5;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(::MAX_MODULE_DEVICE_NUM + 1);
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    MOCKER(CcuGetChannelSpecNum, HcclResult(int32_t, uint8_t, uint32_t&)).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::HcclCommunicator::GetUsedChannelCount).stubs().will(returnValue(static_cast<u32>(1)));
    CommManager::GetInstance(0).PrintChannelInfo();
    GlobalMockObject::verify();

    MOCKER(CcuGetChannelSpecNum, HcclResult(int32_t, uint8_t, uint32_t&)).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::HcclCommunicator::GetUsedChannelCount).stubs().will(returnValue(static_cast<u32>(0)));
    CommManager::GetInstance(0).PrintChannelInfo();
    GlobalMockObject::verify();

    MOCKER(CcuGetChannelSpecNum, HcclResult(int32_t, uint8_t, uint32_t&)).stubs().will(returnValue(HCCL_E_PARA));
    MOCKER_CPP(&Hccl::HcclCommunicator::GetUsedChannelCount).stubs().will(returnValue(static_cast<u32>(0)));
    CommManager::GetInstance(0).PrintChannelInfo();
    GlobalMockObject::verify();
}

TEST_F(HcomutCommManagerTest, CcuResAllocAndCtxMgrInitTest)
{
    CcuStatus ccuStatus;
    EXPECT_EQ(ccuStatus.InsertCommId("1", false, false), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(ccuStatus.InsertCommId("1", false, true), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(ccuStatus.InsertCommId("1", false, true), HcclResult::HCCL_SUCCESS);
}

TEST_F(HcomutCommManagerTest, ut_v2_comm_manager_InsertMsCommId)
{
    CcuStatus ccuStatus;
    ccuStatus.useMsCommIds.push_back("comm1");

    EXPECT_EQ(ccuStatus.InsertCommId("comm2", true, false), HcclResult::HCCL_E_INTERNAL);

    ccuStatus.useMsCommIds.clear();
    EXPECT_EQ(ccuStatus.InsertCommId("comm2", true, false), HcclResult::HCCL_SUCCESS);

    EXPECT_EQ(ccuStatus.useMsCommIds.at(0), "comm2");
}

TEST_F(HcomutCommManagerTest, ut_v2_GetCcuTaskInfo_When_Normal_Expect_Success)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    commParams.rankSize = 5;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;
    Mc2InitTilingInner mc2Tiling;
    mc2Tiling.version = UNKNOWN_TILING_V2;
    rtCcuTaskGroup_t ccuTaskGroup;
    MOCKER_CPP(&HcclCommunicator::GetCcuTaskInfo).stubs().will(returnValue(HCCL_SUCCESS));
    EXPECT_EQ(HcomGetCcuTaskInfo("hccl_world_group", (void*)&mc2Tiling, (void*)&ccuTaskGroup), HCCL_SUCCESS);
}

TEST_F(HcomutCommManagerTest, ut_v2_GetCcuTaskInfo_When_GroupNotExist_Expect_PARA)
{
    Mc2InitTilingInner mc2Tiling;
    mc2Tiling.version = UNKNOWN_TILING_V2;
    rtCcuTaskGroup_t ccuTaskGroup;
    EXPECT_EQ(HcomGetCcuTaskInfo("comm1", (void*)&mc2Tiling, (void*)&ccuTaskGroup), HCCL_E_PARA);
}

TEST_F(HcomutCommManagerTest, ut_v2_GetCcuTaskInfo_When_Fail_Expect_Internal)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    commParams.rankSize = 5;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;
    Mc2InitTilingInner mc2Tiling;
    mc2Tiling.version = UNKNOWN_TILING_V2;
    rtCcuTaskGroup_t ccuTaskGroup;
    MOCKER_CPP(&HcclCommunicator::GetCcuTaskInfo).stubs().will(returnValue(HCCL_E_INTERNAL));
    EXPECT_EQ(HcomGetCcuTaskInfo("hccl_world_group", (void*)&mc2Tiling, (void*)&ccuTaskGroup), HCCL_E_INTERNAL);
}

TEST_F(HcomutCommManagerTest, ut_HcomInitByStringV2_expectHCCL_E_INTERNAL)
{
    MOCKER(CallSingletons).stubs().will(returnValue(HCCL_SUCCESS));
    GlobalMockObject::verify();
    nlohmann::json rank_table
        = {{"version", "2.0"},
           {"rank_count", "1"},
           {"rank_list",
            {{{"rank_id", "0"},
              {"device_id", "0"},
              {"local_id", "0"},
              {"level_list",
               {{{"net_layer", "0"},
                 {"net_instance_id", "az0-rack0"},
                 {"net_type", "TOPO_FILE_DESC"},
                 {"net_attr", ""},
                 {"rank_addr_list", {{{"addr_type", "IPV4"}, {"addr", "223.0.0.28"}, {"ports", {{"0/0"}}}}}}}}}}}}};

    std::string rank_table_string = rank_table.dump();
    Hccl::CommParams commParams;
    commParams.rankSize = 1;
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap.clear();
    CommManager::GetInstance(0).GetCommInfoV2().pComm = nullptr;

    MOCKER_CPP(&HcclCommunicator::Init, HcclResult(HcclCommunicator::*)(const std::string&))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER(HrtGetDevice).stubs().will(returnValue(0));
    MOCKER_CPP(&CommManager::SetCommAcceleratorV2).stubs().will(returnValue(HCCL_SUCCESS));
    HcclResult ret;

    ret = HcomInitByStringV2(rank_table_string.c_str(), "0");
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcomutCommManagerTest, ut_HcomGetCommV2_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    HcclCommInfoV2& hcomCommInfoV2 = GetCommInfoV2();
    Hccl::CommParams commParams;
    hcomCommInfoV2.pComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    void* commV2 = nullptr;
    HcclResult ret = HcomGetCommV2(&commV2);
    EXPECT_NE(commV2, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcomutCommManagerTest, ut_HcomGetGroupParamsV2_When_Normal_Expect_ReturnIsHCCL_SUCCESS)
{
    HcclCommInfoV2& hcomCommInfoV2 = GetCommInfoV2();
    HcclGroupParamsV2 groupParamsV2;
    groupParamsV2.worldRank = 1;
    groupParamsV2.groupRank = 2;
    groupParamsV2.serverNum = 3;
    groupParamsV2.totalRanks = 4;
    std::vector<u32> vec{0, 1, 2, 3};
    groupParamsV2.groupRanks = vec;
    groupParamsV2.refCounter = 0;
    groupParamsV2.destroyFlag = true;
    Hccl::CommParams commParams;
    groupParamsV2.pComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    std::string strGroup = "group2";
    hcomCommInfoV2.hcclGroupMap.insert(std::make_pair(strGroup, groupParamsV2));
    HcclGroupParamsV2 groupParams;
    void* commV2 = nullptr;
    HcclResult ret = HcomGetGroupParamsV2(strGroup.c_str(), static_cast<void*>(&groupParams), &commV2);
    EXPECT_EQ(groupParams.worldRank, 1);
    EXPECT_EQ(groupParams.groupRank, 2);
    EXPECT_EQ(groupParams.serverNum, 3);
    EXPECT_EQ(groupParams.totalRanks, 4);
    EXPECT_EQ(groupParams.groupRanks, vec);
    EXPECT_EQ(groupParams.refCounter, 0);
    EXPECT_EQ(groupParams.destroyFlag, true);
    EXPECT_NE(commV2, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

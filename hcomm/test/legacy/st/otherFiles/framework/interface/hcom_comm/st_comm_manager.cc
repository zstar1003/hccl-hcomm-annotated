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
#include <list>
#include <vector>
#include <string>
#include <securec.h>
#include <hccl/hccl_types.h>
#include <memory>
#include "hccl/base.h"
#include "orion_adapter_rts.h"
#define private public
#include "hccl_communicator.h"
#include "comm_manager.h"
#include "hcom_v2.h"
#include "param_check_v2.h"
#include "log.h"
#include "hccl_common_v2.h"
#include "communicator_impl.h"
#include "ccu_dev_mgr.h"
#include "coll_service_device_mode.h"
#include "mc2_type.h"
#undef private
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
    static void SetUpTestCase()
    {
        GlobalMockObject::verify();
        std::cout << "HcomTest SetUP" << std::endl;
    }

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

    // // GROUP 已存在
    ret = HcomCreateGroupImplV2(strGroup, groupRanksNum, rankIds);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = HcomDestroyGroupImplV2(strGroup);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcomDestroyV2();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcomutCommManagerTest, st_v2_comm_manager_PrintChannelInfo_test1)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    commParams.rankSize = 5;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_unique<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    hcclComm->RegisterPrintChannelInfoCallback(CommManager::GetInstance(0).GetPrintChannelInfoCallback());

    MOCKER(CcuGetChannelSpecNum, HcclResult(int32_t, uint8_t, uint32_t&)).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Hccl::CommunicatorImpl::GetUsedChannelCount).stubs().will(returnValue(2));
    EXPECT_EQ(hcclComm->GetUsedChannelCount(0), 2);
    GlobalMockObject::verify();
    MOCKER_CPP(&Hccl::CommunicatorImpl::GetUsedChannelCount).stubs().will(returnValue(0));
    EXPECT_EQ(hcclComm->GetUsedChannelCount(1), 0);
    hcclComm->pimpl->PrintChannelInfoCallback();
    GlobalMockObject::verify();

    MOCKER(CcuGetChannelSpecNum, HcclResult(int32_t, uint8_t, uint32_t&)).stubs().will(returnValue(HCCL_E_PARA));
    hcclComm->pimpl->PrintChannelInfoCallback();
    GlobalMockObject::verify();

    MOCKER(CcuGetChannelSpecNum, HcclResult(int32_t, uint8_t, uint32_t&)).stubs().will(returnValue(HCCL_E_PARA));
    hcclComm->pimpl->PrintChannelInfoCallback();
    GlobalMockObject::verify();
    delete hcclComm->pimpl->collService;
}

TEST_F(HcomutCommManagerTest, CcuResAllocAndCtxMgrInitTest)
{
    CcuStatus ccuStatus;
    EXPECT_EQ(ccuStatus.InsertCommId("1", false, false), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(ccuStatus.InsertCommId("1", false, true), HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(ccuStatus.InsertCommId("1", false, true), HcclResult::HCCL_SUCCESS);
}

TEST_F(HcomutCommManagerTest, St_GetFileSize_When_InvalidFilePath_Expect_Exception)
{
    char file_name_t[] = "./llt/ace/comop/hccl/orion/testranktable.json";
    int ret = GetFileSize(file_name_t);
    EXPECT_EQ(ret, 0);
}

TEST_F(HcomutCommManagerTest, St_GetFileSize_When_RightFilePath_Expect_Sucess)
{
    char file_name_t[] = "./llt/ace/comop/hccl/orion/ut/framework/interface/hcom_comm/testranktable.json";
    u64 ranktablesize = 739;
    int ret = GetFileSize(file_name_t);
    EXPECT_EQ(ret, ranktablesize);
}

TEST_F(HcomutCommManagerTest, St_HcomInitByFileV2_When_InvalidRanktableSize_Expect_Exception)
{
    char* identify = "0";
    char file_name_t[] = "./llt/ace/comop/hccl/orion/ut/framework/interface/hcom_comm/testranktable.json";
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap.clear();
    CommManager::GetInstance(0).GetCommInfoV2().pComm = nullptr;
    MOCKER(GetFileSize).stubs().will(returnValue(RANKTABLE_FILE_MAX_SIZE + 1));
    int ret = HcomInitByFileV2(file_name_t, identify);
    EXPECT_EQ(ret, HCCL_E_OPEN_FILE_FAILURE);
}

TEST_F(HcomutCommManagerTest, St_HcomGetRankSizeV2_When_InvalidGetRankSize_Expect_Exception)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    commParams.rankSize = 5;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"test_1111", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;
    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));

    HcclGroupParamsV2 params;
    params.pComm = hcclComm;
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap["test_1111"] = params;
    int ret;

    u32 ranksize = 0;
    const char* group = "test_1111";
    ret = HcomGetRankSizeV2(group, &ranksize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcomutCommManagerTest, st_HcomInitByStringV2_expectHCCL_E_INTERNAL)
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
    HcclResult ret = HcomInitByStringV2(rank_table_string.c_str(), "0");
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

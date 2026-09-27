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
#include <sys/types.h>
#include <algorithm>
#include <future>
#include <map>
#include <fstream>
#include <string>
#include <nlohmann/json.hpp>
#define private public
#include "hccl_communicator.h"
#include "communicator_impl.h"
#undef private
#include "hccl_params_pub.h"
#include "hccl_common_v2.h"
#include "param_check_v2.h"
#include "comm_manager.h"
#include "binary_stream.h"
#include "snap_shot_parse.h"
#include "op_base_v2.h"
#include "orion_adapter_rts.h"
#include "one_sided_service_adapt_v2.h"

#include "hccl_one_sided_service_v2.h"
using namespace std;
using namespace Hccl;

static nlohmann::json rank_table_910_1server_4rank
    = {{"status", "completed"},
       {"deploy_mode", "lab"},
       {"group_count", "1"},
       {"chip_info", "910"},
       {"board_id", "0x0000"},
       {"para_plane_nic_location", "device"},
       {"para_plane_nic_num", "4"},
       {"para_plane_nic_name", {"eth0", "eth1", "eth2", "eth3"}},
       {"group_list",
        {{{"group_name", ""},
          {"device_num", "4"},
          {"server_num", "1"},
          {"instance_count", "4"},
          {"instance_list",
           {
               {{"rank_id", "0"},
                {"server_id", "10.0.0.10"},
                {"devices", {{{"device_id", "0"}, {"device_ip", "192.168.0.12"}, {"device_port", "16666"}}}}},

               {{"rank_id", "1"},
                {"server_id", "10.0.0.10"},
                {"devices", {{{"device_id", "1"}, {"device_ip", "192.168.0.14"}, {"device_port", "16666"}}}}},
               {{"rank_id", "2"},
                {"server_id", "10.0.0.10"},
                {"devices", {{{"device_id", "2"}, {"device_ip", "192.168.0.16"}, {"device_port", "16666"}}}}},

               {{"rank_id", "3"},
                {"server_id", "10.0.0.10"},
                {"devices", {{{"device_id", "3"}, {"device_ip", "192.168.0.18"}, {"device_port", "16666"}}}}},

           }},
          {"server_list",
           {
               {{"server_id", "10.0.0.10"},
                {"para_plane_info",
                 {{
                      {"eth0", "192.168.210.2"},
                  },
                  {
                      {"eth1", "192.168.200.2"},
                  },
                  {
                      {"eth2", "192.168.210.2"},
                  },
                  {
                      {"eth3", "192.168.200.2"},
                  }}}

               },
           }}}}}};
class OneSidedUtV2 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        unsetenv("HCCL_INTRA_PCIE_ENABLE");
        setenv("HCCL_INTRA_ROCE_ENABLE", "1", 1);
        std::cout << "\033[36m--OneSidedUtV2 SetUP--\033[0m" << std::endl;
    }
    static void TearDownTestCase()
    {
        unsetenv("HCCL_INTRA_ROCE_ENABLE");
        std::cout << "\033[36m--OneSidedUtV2 TearDown--\033[0m" << std::endl;
    }
    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown()
    {
        std::cout << "A Test TearDown" << std::endl;
        GlobalMockObject::verify();
    }
};

TEST_F(OneSidedUtV2, ut_HcclRegisterMemV2_func)
{
    u32 remoteRankId = 1;
    s32 count = 1024;
    s8* localbuf;
    localbuf = (s8*)malloc(count * sizeof(s8));
    memset_s(localbuf, count * sizeof(s8), 0, count * sizeof(s8));
    HcclMemDesc localMemDesc, remoteMemDesc;

    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    HcclComm hcom = static_cast<HcclComm>(communicator.get());
    CommunicatorImpl commImpl;
    communicator.get()->pimpl.get()->oneSidedService = std::make_unique<Hccl::HcclOneSidedService>(commImpl);
    MOCKER_CPP(&HcclCommunicator::GetRankId).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclOneSidedService::RegMem).stubs().will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HcclRegisterMemV2(hcom, remoteRankId, 0, localbuf, 1024, &localMemDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclDeregisterMemV2(hcom, &localMemDesc);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
    free(localbuf);
}

TEST_F(OneSidedUtV2, ut_HcclExchangeMemDescV2_func)
{
    HcclMemDesc localDesc, remoteDesc;
    char str[21] = "aaaabbbbccccddddeeee";
    memcpy_s(localDesc.desc, sizeof(localDesc.desc), str, sizeof(str));
    memcpy_s(remoteDesc.desc, sizeof(remoteDesc.desc), str, sizeof(str));
    HcclMemDescs local;
    local.arrayLength = 1;
    local.array = &localDesc;
    HcclMemDescs remote;
    remote.arrayLength = 1;
    remote.array = &remoteDesc;
    u32 actualNum;
    MOCKER_CPP(&HcclCommunicator::GetRankId).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclOneSidedService::ExchangeMemDesc).stubs().will(returnValue(HCCL_SUCCESS));
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    HcclComm hcom = static_cast<HcclComm>(communicator.get());
    CommunicatorImpl commImpl;
    communicator.get()->pimpl.get()->oneSidedService = std::make_unique<Hccl::HcclOneSidedService>(commImpl);
    HcclResult ret = HcclExchangeMemDescV2(hcom, 1, &local, 120, &remote, &actualNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OneSidedUtV2, ut_HcclEnableMemAccessV2_func)
{
    u32 remoteRankId = 1;
    HcclMemDesc remoteMemDesc;
    HcclMem remoteMem;
    memcpy_s(&(remoteMemDesc.desc[0]), sizeof(remoteMemDesc.desc), &remoteRankId, sizeof(u32));

    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    HcclComm hcom = static_cast<HcclComm>(communicator.get());
    CommunicatorImpl commImpl;
    communicator.get()->pimpl.get()->oneSidedService = std::make_unique<Hccl::HcclOneSidedService>(commImpl);

    MOCKER_CPP(&HcclOneSidedService::EnableMemAccess).stubs().will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HcclEnableMemAccessV2(hcom, &remoteMemDesc, &remoteMem);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    MOCKER_CPP(&HcclOneSidedService::DisableMemAccess).stubs().will(returnValue(HCCL_SUCCESS));
    ret = HcclDisableMemAccessV2(hcom, &remoteMemDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OneSidedUtV2, ut_HcclBatchPutV2_func)
{
    nlohmann::json rank_table = rank_table_910_1server_4rank;

    char file_name_t[] = "./ut_opbase_test.json";
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);

    if (outfile.is_open()) {
        outfile << std::setw(1) << rank_table << std::endl;
        HCCL_INFO("open %s success", file_name_t);
    } else {
        HCCL_ERROR("open %s failed", file_name_t);
    }

    outfile.close();

    void* comm;
    s32 count = 1024;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream = (void*)1;
    u32 itemNum = 1;
    s8* localbuf;
    s8* remotebuf;

    localbuf = (s8*)malloc(count * sizeof(s8));
    memset_s(localbuf, count * sizeof(s8), 0, count * sizeof(s8));
    remotebuf = (s8*)malloc(count * sizeof(s8));
    memset_s(remotebuf, count * sizeof(s8), 0, count * sizeof(s8));
    HcclOneSideOpDesc desc[itemNum];
    desc[0].count = 1024;
    desc[0].dataType = HCCL_DATA_TYPE_INT8;
    desc[0].localAddr = localbuf;
    desc[0].remoteAddr = remotebuf;

    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    char* rank_table_file = "./ut_opbase_test.json";
    MOCKER(RaTlvRequest).stubs().will(returnValue(0));
    MOCKER_CPP(&CommunicatorImpl::TryInitCcuFeature).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(static_cast<HcclResult (HcclCommunicator::*)(const std::string&)>(&HcclCommunicator::Init))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap.clear();
    CommManager::GetInstance(0).GetCommInfoV2().pComm = nullptr;
    HcclResult ret = HcclCommInitClusterInfoV2(rank_table_file, 0, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclBatchPutV2(comm, 1, nullptr, itemNum, stream);

    EXPECT_EQ(ret, HCCL_E_PTR);

    ret = HcclBatchGetV2(comm, 1, nullptr, itemNum, stream);

    EXPECT_EQ(ret, HCCL_E_PTR);

    ret = HcclBatchPutV2(nullptr, 1, desc, itemNum, stream);

    EXPECT_EQ(ret, HCCL_E_PTR);

    ret = HcclBatchGetV2(nullptr, 1, desc, itemNum, stream);

    EXPECT_EQ(ret, HCCL_E_PTR);

    ret = HcclBatchPutV2(comm, 1, desc, itemNum, nullptr);

    EXPECT_EQ(ret, HCCL_E_PTR);

    ret = HcclBatchGetV2(comm, 1, desc, itemNum, nullptr);

    EXPECT_EQ(ret, HCCL_E_PTR);

    MOCKER(&HcomCheckUserRankV2).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    Hccl::CommParams commParams;
    std::unique_ptr<Hccl::HcclCommunicator> communicator = std::make_unique<Hccl::HcclCommunicator>(commParams);
    HcclComm hcom = static_cast<HcclComm>(communicator.get());
    CommunicatorImpl commImpl;
    communicator.get()->pimpl.get()->oneSidedService = std::make_unique<Hccl::HcclOneSidedService>(commImpl);
    u32 a = 10;
    stream = reinterpret_cast<rtStream_t>(&a);
    MOCKER_CPP(&HcclOneSidedService::BatchGet).stubs().will(returnValue(HCCL_SUCCESS));
    ret = HcclBatchGetV2(hcom, 1, desc, itemNum, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    MOCKER_CPP(&HcclOneSidedService::BatchPut).stubs().will(returnValue(HCCL_SUCCESS));
    ret = HcclBatchPutV2(hcom, 1, desc, itemNum, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    free(localbuf);
    free(remotebuf);

    remove(file_name_t);
}

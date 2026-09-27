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
#include "comm_base_pub.h"
#undef protected
#undef private
#include "sal.h"
#include <vector>

#include "llt_hccl_stub_pub.h"

using namespace std;
using namespace hccl;

class CommInnerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--CommInnerTest SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--CommInnerTest TearDown--\033[0m" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = -1;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }
};

typedef struct innerpara_struct_inner {
    std::string collectiveId;
    u32 userRank;
    u32 user_rank_size;
    u32 rank;
    u32 rank_size;
    u32 devicePhyId;
    std::vector<s32> device_ids;
    std::vector<u32> device_ips;
    std::vector<u32> user_ranks;
    std::string tag;
    HcclDispatcher dispatcher;
    IntraExchanger* exchanger;
    std::vector<RankInfo> para_vector;
    DeviceMem inputMem;
    DeviceMem outputMem;
    std::shared_ptr<CommBase> comm_inner;
} innerpara_t_inner;

HcclDispatcher get_inner_dispatcher(s32 devid)
{
    HcclResult ret = HCCL_SUCCESS;

    // 创建dispatcher
    DevType chipType = DevType::DEV_TYPE_910;

    void* dispatcher = nullptr;
    ret = HcclDispatcherInit(DispatcherType::DISPATCHER_NORMAL, devid, &dispatcher);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(dispatcher, nullptr);

    return dispatcher;
}

TEST_F(CommInnerTest, ut_get_rank_by_userrank1)
{
    s32 ret = HCCL_SUCCESS;

    u32 userRank = 1;
    u32 user_rank_size = 5;

    RankInfo tmp_para_1;
    RankInfo tmp_para_2;
    RankInfo tmp_para_3;

    tmp_para_1.userRank = userRank;
    tmp_para_2.userRank = userRank + 1;
    tmp_para_3.userRank = userRank - 1;

    s32 devicePhyId = 0;
    ret = hrtGetDevice(&devicePhyId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    tmp_para_1.devicePhyId = devicePhyId;
    tmp_para_2.devicePhyId = devicePhyId;
    tmp_para_3.devicePhyId = devicePhyId;

    tmp_para_1.serverId = "10.21.78.208";
    tmp_para_2.serverId = "10.21.78.208";
    tmp_para_3.serverId = "10.21.78.209";

    tmp_para_1.nicIp.push_back(HcclIpAddress("10.21.78.208"));
    tmp_para_2.nicIp.push_back(HcclIpAddress("10.21.78.208"));
    tmp_para_3.nicIp.push_back(HcclIpAddress("10.21.78.209"));

    DeviceMem inputMem = DeviceMem::alloc(128 * 3);
    DeviceMem outputMem = DeviceMem::alloc(128 * 3);
    std::vector<RankInfo> para_vector;
    para_vector.push_back(tmp_para_1);
    para_vector.push_back(tmp_para_2);
    para_vector.push_back(tmp_para_3);

    char collectiveId[SAL_UNIQUE_ID_BYTES];
    ret = SalGetUniqueId(collectiveId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string collective_id_tmp = collectiveId;
    IntraExchanger exchanger{};

    TopoType topoFlag = TopoType::TOPO_TYPE_8P_RING;
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;
    CommBase* comm_inner = new CommBase(
        collective_id_tmp, userRank, user_rank_size, 0, 1, para_vector, topoFlag, nullptr, nullptr, netDevCtxMap,
        exchanger, inputMem, outputMem, true);

    ret = comm_inner->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 rank = INVALID_VALUE_RANKID;
    ret = comm_inner->GetRankByUserRank(5, rank);
    EXPECT_EQ(rank, INVALID_VALUE_RANKID);

    ret = comm_inner->GetRankByUserRank(1, rank);
    EXPECT_EQ(rank, 0);

    ret = comm_inner->GetRankByUserRank(3, rank);
    EXPECT_EQ(rank, INVALID_VALUE_RANKID);

    delete comm_inner;
}

TEST_F(CommInnerTest, ut_get_socket_timeout_err_msg)
{
    s32 ret = HCCL_SUCCESS;

    u32 userRank = 1;
    u32 user_rank_size = 5;

    RankInfo tmp_para_1;
    RankInfo tmp_para_2;
    RankInfo tmp_para_3;

    tmp_para_1.userRank = userRank;
    tmp_para_2.userRank = userRank + 1;
    tmp_para_3.userRank = userRank - 1;

    s32 devicePhyId = 0;
    ret = hrtGetDevice(&devicePhyId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    tmp_para_1.devicePhyId = devicePhyId;
    tmp_para_2.devicePhyId = devicePhyId;
    tmp_para_3.devicePhyId = devicePhyId;

    tmp_para_1.serverId = "10.21.78.208";
    tmp_para_2.serverId = "10.21.78.208";
    tmp_para_3.serverId = "10.21.78.209";

    tmp_para_1.nicIp.push_back(HcclIpAddress("10.21.78.208"));
    tmp_para_2.nicIp.push_back(HcclIpAddress("10.21.78.208"));
    tmp_para_3.nicIp.push_back(HcclIpAddress("10.21.78.209"));

    DeviceMem inputMem = DeviceMem::alloc(128 * 3);
    DeviceMem outputMem = DeviceMem::alloc(128 * 3);
    std::vector<RankInfo> para_vector;
    para_vector.push_back(tmp_para_1);
    para_vector.push_back(tmp_para_2);
    para_vector.push_back(tmp_para_3);

    char collectiveId[SAL_UNIQUE_ID_BYTES];
    ret = SalGetUniqueId(collectiveId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string collective_id_tmp = collectiveId;
    IntraExchanger exchanger{};

    TopoType topoFlag = TopoType::TOPO_TYPE_8P_RING;
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;
    CommBase* comm_inner = new CommBase(
        collective_id_tmp, userRank, user_rank_size, 0, 1, para_vector, topoFlag, nullptr, nullptr, netDevCtxMap,
        exchanger, inputMem, outputMem, true);

    delete comm_inner;
}

TEST_F(CommInnerTest, ut_get_userrank_by_rank1)
{
    s32 ret = HCCL_SUCCESS;

    u32 userRank = 0;
    u32 user_rank_size = 5;

    RankInfo tmp_para;

    tmp_para.userRank = userRank;
    // tmp_para.inputMem = DeviceMem::alloc(128*3);
    // tmp_para.outputMem = DeviceMem::alloc(128*3);
    s32 devicePhyId = 0;
    ret = hrtGetDevice(&devicePhyId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    tmp_para.devicePhyId = devicePhyId;

    tmp_para.serverId = "10.21.78.208";
    tmp_para.nicIp.push_back(HcclIpAddress("10.21.78.208"));

    std::vector<RankInfo> para_vector;
    para_vector.push_back(tmp_para);

    char collectiveId[SAL_UNIQUE_ID_BYTES];
    ret = SalGetUniqueId(collectiveId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string collective_id_tmp = collectiveId;
    IntraExchanger exchanger{};

    TopoType topoFlag = TopoType::TOPO_TYPE_8P_RING;
    DeviceMem inputMem = DeviceMem::alloc(128 * 3);
    DeviceMem outputMem = DeviceMem::alloc(128 * 3);
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;
    CommBase* comm_inner = new CommBase(
        collective_id_tmp, userRank, user_rank_size, userRank, 1, para_vector, topoFlag, nullptr, nullptr, netDevCtxMap,
        exchanger, inputMem, outputMem, true);

    ret = comm_inner->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 user_rank = INVALID_VALUE_RANKID;
    ret = comm_inner->GetUserRankByRank(5, user_rank);
    EXPECT_EQ(user_rank, INVALID_VALUE_RANKID);

    ret = comm_inner->GetUserRankByRank(0, user_rank);
    EXPECT_EQ(user_rank, 0);

    delete comm_inner;
}

TEST_F(CommInnerTest, ut_get_userrank_by_rank_with_err1)
{
    s32 ret = HCCL_SUCCESS;

    u32 userRank = 0;
    u32 user_rank_size = 5;

    RankInfo tmp_para;

    tmp_para.userRank = userRank;
    // tmp_para.inputMem = DeviceMem::alloc(128*3);
    // tmp_para.outputMem = DeviceMem::alloc(128*3);

    s32 devicePhyId = 0;
    ret = hrtGetDevice(&devicePhyId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    tmp_para.devicePhyId = devicePhyId;

    tmp_para.serverId = "10.21.78.208";
    tmp_para.nicIp.push_back(HcclIpAddress("10.21.78.208"));

    std::vector<RankInfo> para_vector;
    para_vector.push_back(tmp_para);

    char collectiveId[SAL_UNIQUE_ID_BYTES];
    ret = SalGetUniqueId(collectiveId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string collective_id_tmp = collectiveId;
    IntraExchanger exchanger{};

    TopoType topoFlag = TopoType::TOPO_TYPE_8P_RING;
    DeviceMem inputMem = DeviceMem::alloc(128 * 3);
    DeviceMem outputMem = DeviceMem::alloc(128 * 3);
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap;
    CommBase* comm_inner = new CommBase(
        collective_id_tmp, userRank, user_rank_size, 0, 1, para_vector, topoFlag, nullptr, nullptr, netDevCtxMap,
        exchanger, inputMem, outputMem, true);

    ret = comm_inner->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 user_rank = INVALID_VALUE_RANKID;
    ret = comm_inner->GetUserRankByRank(5, user_rank);
    EXPECT_EQ(user_rank, INVALID_VALUE_RANKID);

    delete comm_inner;
}

TEST_F(CommInnerTest, print_error_connection)
{
    HcclResult ret = HCCL_SUCCESS;

    s32 userRank = 0;

    RankInfo tmp_para;

    tmp_para.userRank = userRank;

    s32 devicePhyId = 0;
    ret = hrtGetDevice(&devicePhyId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    tmp_para.devicePhyId = devicePhyId;

    tmp_para.serverIdx = 0;
    tmp_para.serverId = "10.21.78.208";
    tmp_para.nicIp.push_back(HcclIpAddress("10.21.78.208"));

    std::vector<RankInfo> para_vector;
    para_vector.push_back(tmp_para);

    char collectiveId[SAL_UNIQUE_ID_BYTES];
    ret = SalGetUniqueId(collectiveId);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::string collective_id_tmp = collectiveId;

    std::map<u32, std::vector<HcclIpAddress>> dstInterServerMap;
    std::map<u32, std::vector<HcclIpAddress>> dstInterClientMap;
}

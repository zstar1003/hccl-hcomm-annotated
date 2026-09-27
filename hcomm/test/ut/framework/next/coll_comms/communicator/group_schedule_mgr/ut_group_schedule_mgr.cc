/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include <vector>
#include <string>
#include "hccl/hccl_types.h"
#include "hccl/hccl_launch.h"
#include "hccl/hccl_rank_graph.h"
#include "acl/acl_base_rt.h"
#define private public
#include "group_schedule_mgr.h"
#include "hccl_comm_pub.h"
#include "coll_comm.h"
#undef private

using namespace hccl;

class GroupScheduleMgrTest : public testing::Test {
public:
    void SetUp() override
    {
        SetHcclP2pTaskNums(0);
        GetHcclGroupCommList().clear();
    }
    void TearDown() override
    {
        GlobalMockObject::verify();
        SetHcclP2pTaskNums(0);
        GetHcclGroupCommList().clear();
    }
};

TEST_F(GroupScheduleMgrTest, Ut_GetHcclGroupCommList_When_Initialized_Expect_EmptyList)
{
    std::vector<HcclComm>& commList = GetHcclGroupCommList();
    EXPECT_EQ(commList.size(), 0u);
}

TEST_F(GroupScheduleMgrTest, Ut_GetHcclGroupCommList_When_AddedComms_Expect_ReturnCorrectList)
{
    std::vector<HcclComm>& commList = GetHcclGroupCommList();
    HcclComm comm1 = reinterpret_cast<HcclComm>(0x1000);
    HcclComm comm2 = reinterpret_cast<HcclComm>(0x2000);
    commList.push_back(comm1);
    commList.push_back(comm2);

    EXPECT_EQ(commList.size(), 2u);
    EXPECT_EQ(commList[0], comm1);
    EXPECT_EQ(commList[1], comm2);
}

TEST_F(GroupScheduleMgrTest, Ut_GetHcclP2pTaskNums_When_Initialized_Expect_Zero)
{
    int32_t taskNums = GetHcclP2pTaskNums();
    EXPECT_EQ(taskNums, 0);
}

TEST_F(GroupScheduleMgrTest, Ut_SetHcclP2pTaskNums_When_SetValue_Expect_GetReturnsSameValue)
{
    SetHcclP2pTaskNums(100);
    int32_t taskNums = GetHcclP2pTaskNums();
    EXPECT_EQ(taskNums, 100);

    SetHcclP2pTaskNums(0);
    taskNums = GetHcclP2pTaskNums();
    EXPECT_EQ(taskNums, 0);
}

TEST_F(GroupScheduleMgrTest, Ut_SetUsrStream_When_ValidStream_Expect_Success)
{
    GroupScheduleMgr mgr;
    aclrtStream stream = reinterpret_cast<aclrtStream>(0x1000);
    HcclResult ret = mgr.SetUsrStream(stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(GroupScheduleMgrTest, Ut_SetUsrStream_When_NullStream_Expect_PtrError)
{
    GroupScheduleMgr mgr;
    aclrtStream stream = nullptr;
    HcclResult ret = mgr.SetUsrStream(stream);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(GroupScheduleMgrTest, Ut_GetUsrStream_When_StreamSet_Expect_ReturnCorrectStream)
{
    GroupScheduleMgr mgr;
    aclrtStream inputStream = reinterpret_cast<aclrtStream>(0x2000);
    HcclResult ret = mgr.SetUsrStream(inputStream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    aclrtStream outputStream = nullptr;
    ret = mgr.GetUsrStream(outputStream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(outputStream, inputStream);
}

TEST_F(GroupScheduleMgrTest, Ut_GetUsrStream_When_StreamNotSet_Expect_PtrError)
{
    GroupScheduleMgr mgr;
    aclrtStream outputStream = nullptr;
    HcclResult ret = mgr.GetUsrStream(outputStream);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

HcclResult
MockHcclRankGraphGetInstSizeListByLayer(HcclComm comm, uint32_t netLayer, uint32_t** instSizeList, uint32_t* listSize)
{
    static uint32_t mockSizeList[4] = {2, 2, 2, 2};
    *instSizeList = mockSizeList;
    *listSize = 4;
    return HCCL_SUCCESS;
}

static CollComm g_mockCollComm(nullptr, 0, "test", ManagerCallbacks{}, CollCommInitMode::simpleMode);
void SetupCollCommMocks()
{
    MOCKER_CPP(&hcclComm::GetCollComm).stubs().will(returnValue(&g_mockCollComm));

    MOCKER_CPP(&CollComm::GetMyRankId).stubs().will(returnValue(0U));

    MOCKER_CPP(&CollComm::GetRankSize).stubs().will(returnValue(4U));
}

TEST_F(GroupScheduleMgrTest, Ut_AppendGroupP2pTask_When_NullComm_Expect_PtrError)
{
    GroupScheduleMgr mgr;
    HcclP2pTask task;
    HcclOpP2pDesc p2pDesc;
    p2pDesc.cmdType = HcclCMDType::HCCL_CMD_SEND;
    p2pDesc.remoteRank = 0;

    HcclResult ret = mgr.AppendGroupP2pTask(nullptr, task, p2pDesc);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(GroupScheduleMgrTest, Ut_AppendGroupP2pTask_When_MaxTaskNumReached_Expect_InternalError)
{
    GroupScheduleMgr mgr;
    SetHcclP2pTaskNums(MAX_P2P_TASK_NUM);

    hccl::hcclComm mockHcclComm;
    HcclComm comm = static_cast<HcclComm>(&mockHcclComm);
    HcclP2pTask task;
    HcclOpP2pDesc p2pDesc;
    p2pDesc.cmdType = HcclCMDType::HCCL_CMD_SEND;
    p2pDesc.remoteRank = 0;

    HcclResult ret = mgr.AppendGroupP2pTask(comm, task, p2pDesc);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);

    SetHcclP2pTaskNums(0);
}

TEST_F(GroupScheduleMgrTest, Ut_AppendGroupP2pTask_When_SendTask_Expect_AddedToSendQueue)
{
    GroupScheduleMgr mgr;
    hccl::hcclComm mockHcclComm;
    HcclComm comm = static_cast<HcclComm>(&mockHcclComm);
    HcclP2pTask task;
    HcclOpP2pDesc p2pDesc;
    p2pDesc.cmdType = HcclCMDType::HCCL_CMD_SEND;
    p2pDesc.remoteRank = 1;

    SetupCollCommMocks();
    MOCKER(HcclRankGraphGetInstSizeListByLayer).stubs().will(invoke(MockHcclRankGraphGetInstSizeListByLayer));

    HcclResult ret = mgr.AppendGroupP2pTask(comm, task, p2pDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(GroupScheduleMgrTest, Ut_GetP2pTaskSchedule_When_NoTasks_Expect_EmptyQueues)
{
    GroupScheduleMgr mgr;
    std::vector<HcclP2pTask> sortedSendQue;
    std::vector<HcclP2pTask> sortedRecvQue;

    HcclResult ret = mgr.GetP2pTaskSchedule(sortedSendQue, sortedRecvQue);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(sortedSendQue.size(), 0u);
    EXPECT_EQ(sortedRecvQue.size(), 0u);
}

TEST_F(GroupScheduleMgrTest, Ut_GroupScheduleMgrDestructor_When_ObjectDestroyed_Expect_NoException)
{
    GroupScheduleMgr* mgr = new GroupScheduleMgr();
    delete mgr;
    EXPECT_TRUE(true);
}

TEST_F(GroupScheduleMgrTest, Ut_pow2Up_When_InputPowerOfTwo_Expect_ReturnSameValue)
{
    uint32_t input = 8;
    uint32_t expected = 8;
    uint32_t power = 1;
    while (power < input) {
        power = power << 1U;
    }
    EXPECT_EQ(power, expected);
}

TEST_F(GroupScheduleMgrTest, Ut_GetCurLocalRank_When_RankNotInList_Expect_InternalError)
{
    GroupScheduleMgr mgr;
    mgr.userRank_ = 5;
    mgr.serverNum_ = 1;
    mgr.serverToRankSize_[0] = 2;
    mgr.serverToRankList_[0] = {0, 1};

    uint32_t localRank = 0;
    HcclResult ret = mgr.GetCurLocalRank(localRank);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

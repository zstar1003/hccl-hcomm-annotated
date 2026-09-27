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
#define __HCCL_SAL_GLOBAL_RES_INCLUDE__

#include "sal.h"
#include "llt_hccl_stub_pub.h"
#include "alltoallv_staged_calculator_pub.h"

#include <semaphore.h>
#include <sys/time.h> /* 获取时间 */
#include <sys/mman.h>
#include <fcntl.h>
#include <securec.h>
#include <dirent.h>

using namespace std;
using namespace hccl;

constexpr u32 MESH_AGGREGATION_RANK_SIZE_910 = 4; // 4p mesh
class AlltoAllVStagedCalculatorTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AlltoAllVStagedCalculatorTest SetUP" << std::endl;
        //    (void)rt_stop_sequence_thread();
    }
    static void TearDownTestCase()
    {
        //  (void)rt_start_sequence_thread();
        std::cout << "AlltoAllVStagedCalculatorTest TearDown" << std::endl;
    }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = -1;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};

#if 1
TEST_F(AlltoAllVStagedCalculatorTest, ut_alltoallv_staged_static)
{
    AlltoAllUserRankInfo userRankInfo;
    userRankInfo.userRankSize = 12;
    userRankInfo.userRank = 5;

    u32 userRankSize = 12;
    u32 userRank = 5;

    // 生成数据
    std::vector<SendRecvInfo> allSendRecvInfo;
    for (u32 i = 0; i < userRankSize; i++) {
        SendRecvInfo curSendRecvInfo;
        curSendRecvInfo.sendLength.resize(userRankSize);
        curSendRecvInfo.sendOffset.resize(userRankSize);
        curSendRecvInfo.recvLength.resize(userRankSize);
        curSendRecvInfo.recvOffset.resize(userRankSize);
        u64 sdisp = 0;
        u64 rdisp = 0;
        //        cout << "rank " << i << endl;
        for (u32 j = 0; j < userRankSize; j++) {
            curSendRecvInfo.sendLength[j] = j + 1;
            curSendRecvInfo.sendOffset[j] = sdisp;
            sdisp += curSendRecvInfo.sendLength[j];

            curSendRecvInfo.recvLength[j] = i + 1; // 从每个rank收到的数据是一样的
            curSendRecvInfo.recvOffset[j] = rdisp;
            rdisp += curSendRecvInfo.recvLength[j];
            //            cout << curSendRecvInfo.sendLength[j] << "," << curSendRecvInfo.sendOffset[j] << "," <<
            //                curSendRecvInfo.recvLength[j] << "," << curSendRecvInfo.recvOffset[j] << endl;
        }
        allSendRecvInfo.push_back(curSendRecvInfo);
    }
    u64 workspaceMemSize = 0;
    AlltoAllVStagedCalculator::CalcWorkSpaceMemSize(
        userRankInfo, allSendRecvInfo, workspaceMemSize, MESH_AGGREGATION_RANK_SIZE_910);
    // cout << "workspaceMemSize:  " << workspaceMemSize << endl;
    EXPECT_EQ(workspaceMemSize, 72);
}
#endif

TEST_F(AlltoAllVStagedCalculatorTest, ut_alltoallv_staged_static_int16)
{
    u32 userRankSize = 12;
    for (u32 rankIndex = 0; rankIndex < userRankSize; rankIndex++) {
        u32 userRank = rankIndex;

        AlltoAllUserRankInfo userRankInfo;
        userRankInfo.userRankSize = userRankSize;
        userRankInfo.userRank = userRank;

        u64 unitBytes = 100; // 104857600*2

        // 生成数据
        std::vector<SendRecvInfo> allSendRecvInfo;
        for (u32 i = 0; i < userRankSize; i++) {
            SendRecvInfo curSendRecvInfo;
            curSendRecvInfo.sendLength.resize(userRankSize);
            curSendRecvInfo.sendOffset.resize(userRankSize);
            curSendRecvInfo.recvLength.resize(userRankSize);
            curSendRecvInfo.recvOffset.resize(userRankSize);
            u64 sdisp = 0;
            u64 rdisp = 0;
            //            cout << "rank " << i << endl;

            for (u32 j = 0; j < userRankSize; j++) {
                curSendRecvInfo.sendLength[j] = unitBytes * (j + 1);
                curSendRecvInfo.sendOffset[j] = sdisp;
                sdisp += curSendRecvInfo.sendLength[j];

                curSendRecvInfo.recvLength[j] = unitBytes * (i + 1); // 从每个rank收到的数据是一样的
                curSendRecvInfo.recvOffset[j] = rdisp;
                rdisp += curSendRecvInfo.recvLength[j];
                //                cout << curSendRecvInfo.sendLength[j] << "," << curSendRecvInfo.sendOffset[j] << ","
                //                <<
                //                    curSendRecvInfo.recvLength[j] << "," << curSendRecvInfo.recvOffset[j] << endl;
            }
            allSendRecvInfo.push_back(curSendRecvInfo);
        }

        u64 workspaceMemSize = 0;
        AlltoAllVStagedCalculator::CalcWorkSpaceMemSize(
            userRankInfo, allSendRecvInfo, workspaceMemSize, MESH_AGGREGATION_RANK_SIZE_910);
    }
}

TEST_F(AlltoAllVStagedCalculatorTest, ut_alltoallv_staged_static_size_0)
{
    u32 userRankSize = 8;
    for (u32 rankIndex = 0; rankIndex < userRankSize; rankIndex++) {
        u32 userRank = rankIndex;

        AlltoAllUserRankInfo userRankInfo;
        userRankInfo.userRankSize = userRankSize;
        userRankInfo.userRank = userRank;
        // 生成数据
        std::vector<SendRecvInfo> allSendRecvInfo;
        for (u32 i = 0; i < userRankSize; i++) {
            SendRecvInfo curSendRecvInfo;
            curSendRecvInfo.sendLength.resize(userRankSize);
            curSendRecvInfo.sendOffset.resize(userRankSize);
            curSendRecvInfo.recvLength.resize(userRankSize);
            curSendRecvInfo.recvOffset.resize(userRankSize);

            for (u32 j = 0; j < userRankSize; j++) {
                curSendRecvInfo.sendLength[j] = 0;
                curSendRecvInfo.sendOffset[j] = 0;
                curSendRecvInfo.recvLength[j] = 0;
                curSendRecvInfo.recvOffset[j] = 0;
            }
            allSendRecvInfo.push_back(curSendRecvInfo);
        }

        u64 workspaceMemSize = 0;
        AlltoAllVStagedCalculator::CalcWorkSpaceMemSize(userRankInfo, allSendRecvInfo, workspaceMemSize, 1);
        EXPECT_EQ(workspaceMemSize, 2 * 1024 * 1024);
    }
}

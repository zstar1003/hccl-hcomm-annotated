/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include <set>
#include <string>

#include "invalid_params_exception.h"

#define private public
#include "updater_for_64_plus_1.h"
#undef private

using namespace Hccl;

class UpdaterFor64Plus1Test : public testing::Test {};

TEST_F(UpdaterFor64Plus1Test, Ut_SaveReplaceInfo_When_BackupRank_Expect_RecordReplaceInfo)
{
    UpdaterFor64Plus1 updater;
    NewRankInfo rank;
    rank.rankId = 1;
    rank.localId = BACKUP_LOCAL_ID;
    rank.replacedLocalId = 1;
    RankLevelInfo levelInfo;
    levelInfo.netLayer = 0;
    levelInfo.netInstId = "az0-rack0";
    rank.rankLevelInfos.emplace_back(levelInfo);

    updater.SaveReplaceInfo(rank);

    ASSERT_EQ(1U, updater.replaceInfo.size());
    EXPECT_EQ(std::make_pair(BACKUP_LOCAL_ID, static_cast<LocalId>(1)), updater.replaceInfo["az0-rack0"]);
}

TEST_F(UpdaterFor64Plus1Test, Ut_SaveReplaceInfo_When_NormalRank_Expect_Ignore)
{
    UpdaterFor64Plus1 updater;
    NewRankInfo rank;
    rank.localId = 0;

    updater.SaveReplaceInfo(rank);

    EXPECT_TRUE(updater.replaceInfo.empty());
}

TEST_F(UpdaterFor64Plus1Test, Ut_SaveReplaceInfo_When_BackupWithoutLayer0_Expect_InvalidParamsException)
{
    UpdaterFor64Plus1 updater;
    NewRankInfo rank;
    rank.rankId = 1;
    rank.localId = BACKUP_LOCAL_ID;

    EXPECT_THROW(updater.SaveReplaceInfo(rank), InvalidParamsException);
}

TEST_F(UpdaterFor64Plus1Test, Ut_GetLinkIndex_When_SameAxis_Expect_SelectExpectedBackupPlane)
{
    UpdaterFor64Plus1 updater;

    EXPECT_EQ(std::make_pair(0U, 2U), updater.GetLinkIndex(2, 1));
    EXPECT_EQ(std::make_pair(1U, 1U), updater.GetLinkIndex(5, 1));
    EXPECT_EQ(std::make_pair(2U, 2U), updater.GetLinkIndex(16, 0));
    EXPECT_EQ(std::make_pair(3U, 1U), updater.GetLinkIndex(40, 0));
}

TEST_F(UpdaterFor64Plus1Test, Ut_GetLinkIndex_When_InvalidInput_Expect_InvalidParamsException)
{
    UpdaterFor64Plus1 updater;

    EXPECT_THROW(updater.GetLinkIndex(BACKUP_LOCAL_ID, 0), InvalidParamsException);
    EXPECT_THROW(updater.GetLinkIndex(1, 1), InvalidParamsException);
    EXPECT_THROW(updater.GetLinkIndex(9, 0), InvalidParamsException);
}

TEST_F(UpdaterFor64Plus1Test, Ut_GetPortFromSet_When_IndexValid_Expect_ReturnSelectedPort)
{
    UpdaterFor64Plus1 updater;
    std::set<std::string> ports = {"0/0", "0/1", "0/2"};

    EXPECT_EQ("0/0", updater.GetPortFromSet(ports, 0));
    EXPECT_EQ("0/2", updater.GetPortFromSet(ports, 2));
}

TEST_F(UpdaterFor64Plus1Test, Ut_GetPortFromSet_When_IndexInvalid_Expect_InvalidParamsException)
{
    UpdaterFor64Plus1 updater;
    std::set<std::string> ports = {"0/0"};

    EXPECT_THROW(updater.GetPortFromSet(ports, 1), InvalidParamsException);
}

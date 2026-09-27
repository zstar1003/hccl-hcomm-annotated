/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <unistd.h>

#include "db_sim_runner_db.h"
#include "db_sim_runner_ops.h"
#include "db_sim_sqlite_db.h"

namespace {
const std::string kTestDbPath = "/tmp/test_sim_runner_ops.db";

void CleanUpDb()
{
    std::remove(kTestDbPath.c_str());
    std::remove((kTestDbPath + "-wal").c_str());
    std::remove((kTestDbPath + "-shm").c_str());
}

void SetupTestData()
{
    CleanUpDb();
    sim::SqliteDatabase::SetDbPath(kTestDbPath);
    SimRunnerSqliteDB::Instance().ClearAll();

    sim::Server server{};
    server.pod_id = 100;
    snprintf(server.version, sizeof(server.version), "v1.0");
    RunnerDB::Add<sim::Server>(server);

    sim::Host host{};
    host.server_id = 1;
    snprintf(host.ip_addr, sizeof(host.ip_addr), "192.168.1.100");
    host.arch = 1;
    RunnerDB::Add<sim::Host>(host);

    sim::Device device{};
    device.server_id = 1;
    device.logic_id = 0;
    device.physical_id = 0;
    RunnerDB::Add<sim::Device>(device);

    sim::Context ctx{};
    ctx.device_id = 1;
    ctx.is_default = 1;
    ctx.run_id = 1;
    RunnerDB::Add<sim::Context>(ctx);

    sim::Rank rank{};
    rank.rank_id = 0;
    rank.device_id = 1;
    RunnerDB::Add<sim::Rank>(rank);
}
} // namespace

class SimRunnerOpsTest : public testing::Test {
protected:
    void SetUp() override { SetupTestData(); }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimRunnerOpsTest, SetAndGetLastStreamId)
{
    sim::SetLastStreamIdTls(12345);

    uint64_t streamId = sim::GetLastStreamIdTls();

    EXPECT_EQ(streamId, 12345);
}

TEST_F(SimRunnerOpsTest, SetAndGetLastTaskId)
{
    sim::SetLastTaskIdTls(67890);

    uint64_t taskId = sim::GetLastTaskIdTls();

    EXPECT_EQ(taskId, 67890);
}

TEST_F(SimRunnerOpsTest, SetTsDevice)
{
    sim::SetTsDevice(42);
    sim::SetTsDevice(0);
}

TEST_F(SimRunnerOpsTest, GetRankSize_ReturnsCorrectCount)
{
    uint32_t size = sim::GetRankSize();

    EXPECT_GE(size, 1);
}

TEST_F(SimRunnerOpsTest, GetHostSize_ReturnsCorrectCount)
{
    uint32_t size = sim::GetHostSize();

    EXPECT_GE(size, 1);
}

TEST_F(SimRunnerOpsTest, GetCurrentStreamId_SameKey_ReturnsSameId)
{
    uint64_t streamKey = 1000;

    uint32_t id1 = sim::GetCurrentStreamId(streamKey);
    uint32_t id2 = sim::GetCurrentStreamId(streamKey);

    EXPECT_EQ(id1, id2);
}

TEST_F(SimRunnerOpsTest, GetCurrentStreamId_DifferentKeys_ReturnsDifferentIds)
{
    uint32_t id1 = sim::GetCurrentStreamId(1000);
    uint32_t id2 = sim::GetCurrentStreamId(2000);
    uint32_t id3 = sim::GetCurrentStreamId(3000);

    EXPECT_NE(id1, id2);
    EXPECT_NE(id2, id3);
    EXPECT_EQ(id1, static_cast<uint32_t>(0));
    EXPECT_EQ(id2, static_cast<uint32_t>(1));
    EXPECT_EQ(id3, static_cast<uint32_t>(2));
}

TEST_F(SimRunnerOpsTest, SetCurrCtxTls_UpdatesContext)
{
    bool ret = sim::SetCurrCtxTls(1);

    EXPECT_TRUE(ret);
}

TEST_F(SimRunnerOpsTest, GetCurrDeviceKey_WithValidCtx_ReturnsDeviceKey)
{
    sim::SetCurrCtxTls(1);

    uint64_t deviceKey = sim::GetCurrDeviceKey();

    EXPECT_EQ(deviceKey, 1);
}

TEST_F(SimRunnerOpsTest, GetCurrDeviceKey_WithInvalidCtx_ReturnsZero)
{
    sim::SetCurrCtxTls(99999);

    uint64_t deviceKey = sim::GetCurrDeviceKey();

    EXPECT_EQ(deviceKey, 0);
}

TEST_F(SimRunnerOpsTest, GetRankIdByCtxId_WithValidCtx_ReturnsRankId)
{
    uint64_t rankId = sim::GetRankIdByCtxId(1);

    EXPECT_EQ(rankId, 0);
}

TEST_F(SimRunnerOpsTest, GetRankIdByCtxId_WithInvalidCtx_ReturnsZero)
{
    uint64_t rankId = sim::GetRankIdByCtxId(99999);

    EXPECT_EQ(rankId, 0);
}

TEST_F(SimRunnerOpsTest, GetCurrentStreamId_SequentialIncrease)
{
    uint32_t firstId = sim::GetCurrentStreamId(static_cast<uint64_t>(100));
    EXPECT_GE(firstId, static_cast<uint32_t>(0));
}

class SimRunnerOpsRankTest : public testing::Test {
protected:
    void SetUp() override
    {
        SetupTestData();

        sim::Device device{};
        auto ret = RunnerDB::GetOneByPred<sim::Device>([](const sim::Device& d) {
            return d.physical_id == 0;
        });
        ASSERT_TRUE(ret.second);

        sim::Rank rank1{};
        rank1.rank_id = 1;
        rank1.device_id = ret.first.id;
        RunnerDB::Add<sim::Rank>(rank1);

        sim::Rank rank2{};
        rank2.rank_id = 2;
        rank2.device_id = ret.first.id;
        RunnerDB::Add<sim::Rank>(rank2);
    }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimRunnerOpsRankTest, GetRankSize_AfterAddingRanks)
{
    uint32_t size = sim::GetRankSize();

    EXPECT_GE(size, 1);
}

TEST_F(SimRunnerOpsRankTest, GetCurrRankId_WithValidContext)
{
    sim::SetCurrCtxTls(1);

    uint64_t rankId = sim::GetCurrRankId();

    EXPECT_GE(rankId, 0);
}

// ==================== 新增：GetCurrRankId 错误路径 ====================

class SimRunnerOpsGetCurrRankIdTest : public testing::Test {
protected:
    void SetUp() override
    {
        SetupTestData();
        sim::Runner runner;
        sim::GetCurrRunnerTls(1, runner);
    }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimRunnerOpsGetCurrRankIdTest, GetCurrRankId_InvalidCtx_ReturnsZero)
{
    // 设置一个不存在的 context id
    sim::SetCurrCtxTls(99999);
    uint64_t rankId = sim::GetCurrRankId();
    EXPECT_EQ(rankId, 0);
}

TEST_F(SimRunnerOpsGetCurrRankIdTest, GetCurrRankId_NoRankForDevice_ReturnsZero)
{
    // 添加一个 context，其 device_id 没有对应的 rank
    sim::Device device2{};
    device2.server_id = 1;
    device2.logic_id = 2;
    device2.physical_id = 2;
    RunnerDB::Add<sim::Device>(device2);

    sim::Context ctx2{};
    ctx2.device_id = 2; // device_id=2 没有对应的 rank
    ctx2.is_default = 1;
    ctx2.run_id = 1;
    RunnerDB::Add<sim::Context>(ctx2);

    sim::SetCurrCtxTls(2);
    uint64_t rankId = sim::GetCurrRankId();
    EXPECT_EQ(rankId, 0);
}

// ==================== 新增：GetRankIdByCtxId 错误路径 ====================

class SimRunnerOpsGetRankIdByCtxIdTest : public testing::Test {
protected:
    void SetUp() override { SetupTestData(); }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimRunnerOpsGetRankIdByCtxIdTest, GetRankIdByCtxId_InvalidCtx_ReturnsZero)
{
    uint64_t rankId = sim::GetRankIdByCtxId(99999);
    EXPECT_EQ(rankId, 0);
}

TEST_F(SimRunnerOpsGetRankIdByCtxIdTest, GetRankIdByCtxId_NoDevice_ReturnsZero)
{
    // 添加一个 context，其 device_id 没有对应的 device
    sim::Context ctx2{};
    ctx2.device_id = 999; // device_id=999 不存在
    ctx2.is_default = 1;
    ctx2.run_id = 1;
    RunnerDB::Add<sim::Context>(ctx2);

    uint64_t rankId = sim::GetRankIdByCtxId(2);
    EXPECT_EQ(rankId, 0);
}

// ==================== 新增：GetCurrDeviceKey 错误路径 ====================

TEST_F(SimRunnerOpsGetCurrRankIdTest, GetCurrDeviceKey_InvalidCtx_ReturnsZero)
{
    sim::SetCurrCtxTls(99999);
    uint64_t deviceKey = sim::GetCurrDeviceKey();
    EXPECT_EQ(deviceKey, 0);
}

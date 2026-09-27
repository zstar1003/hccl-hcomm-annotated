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

#include "db_sim_runner_db.h"
#include "db_sim_sqlite_db.h"

namespace {
const std::string kTestDbPath = "/tmp/test_sim_runner_db.db";

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
}
} // namespace

class SimRunnerDbTest : public testing::Test {
protected:
    void SetUp() override { SetupTestData(); }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimRunnerDbTest, GetAllTableName_ReturnsNonEmptyList)
{
    auto tableNames = RunnerDB::GetAllTableName();

    EXPECT_GT(tableNames.size(), 0);
}

TEST_F(SimRunnerDbTest, GetAllTableName_ContainsExpectedTables)
{
    auto tableNames = RunnerDB::GetAllTableName();

    bool foundServer = false;
    bool foundHost = false;
    bool foundDevice = false;

    for (const auto& name : tableNames) {
        if (name == "Server") {
            foundServer = true;
        }
        if (name == "Host") {
            foundHost = true;
        }
        if (name == "Device") {
            foundDevice = true;
        }
    }

    EXPECT_TRUE(foundServer);
    EXPECT_TRUE(foundHost);
    EXPECT_TRUE(foundDevice);
}

TEST_F(SimRunnerDbTest, Add_And_GetById_Server)
{
    sim::Server server{};
    server.pod_id = 200;
    snprintf(server.version, sizeof(server.version), "v2.0");

    uint64_t id = RunnerDB::Add<sim::Server>(server);

    auto result = RunnerDB::GetById<sim::Server>(id);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->pod_id, 200);
    EXPECT_STREQ(result->version, "v2.0");
}

TEST_F(SimRunnerDbTest, Add_And_GetById_Host)
{
    sim::Host host{};
    host.server_id = 2;
    snprintf(host.ip_addr, sizeof(host.ip_addr), "192.168.2.1");
    host.arch = 2;

    uint64_t id = RunnerDB::Add<sim::Host>(host);

    auto result = RunnerDB::GetById<sim::Host>(id);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->server_id, 2);
    EXPECT_STREQ(result->ip_addr, "192.168.2.1");
    EXPECT_EQ(result->arch, 2);
}

TEST_F(SimRunnerDbTest, GetByPred_ReturnsMatchingRecords)
{
    auto results = RunnerDB::GetByPred<sim::Device>([](const sim::Device& d) {
        return d.server_id == 1;
    });

    EXPECT_GE(results.size(), 1);
}

TEST_F(SimRunnerDbTest, GetByPred_ReturnsEmptyForNoMatch)
{
    auto results = RunnerDB::GetByPred<sim::Device>([](const sim::Device& d) {
        return d.server_id == 999;
    });

    EXPECT_EQ(results.size(), 0);
}

TEST_F(SimRunnerDbTest, GetOneByPred_ReturnsFirstMatch)
{
    auto result = RunnerDB::GetOneByPred<sim::Device>([](const sim::Device& d) {
        return d.server_id == 1;
    });

    EXPECT_TRUE(result.second);
    EXPECT_EQ(result.first.server_id, 1);
}

TEST_F(SimRunnerDbTest, GetOneByPred_ReturnsFalseForNoMatch)
{
    auto result = RunnerDB::GetOneByPred<sim::Device>([](const sim::Device& d) {
        return d.server_id == 999;
    });

    EXPECT_FALSE(result.second);
}

TEST_F(SimRunnerDbTest, Update_ModifiesRecord)
{
    auto result = RunnerDB::GetOneByPred<sim::Device>([](const sim::Device& d) {
        return d.physical_id == 0;
    });
    ASSERT_TRUE(result.second);

    uint64_t id = result.first.id;
    RunnerDB::Update<sim::Device>(id, [](sim::Device& d) {
        d.logic_id = 999;
    });

    auto updated = RunnerDB::GetById<sim::Device>(id);
    ASSERT_TRUE(updated.has_value());
    EXPECT_EQ(updated->logic_id, 999);
}

TEST_F(SimRunnerDbTest, Update_ReturnsFalseForInvalidId)
{
    bool ret = RunnerDB::Update<sim::Device>(99999, [](sim::Device& d) {
        d.logic_id = 999;
    });

    EXPECT_FALSE(ret);
}

TEST_F(SimRunnerDbTest, Delete_RemovesRecord)
{
    sim::Device device{};
    device.server_id = 1;
    device.logic_id = 100;
    device.physical_id = 100;
    uint64_t id = RunnerDB::Add<sim::Device>(device);

    bool ret = RunnerDB::Delete<sim::Device>(id);
    EXPECT_TRUE(ret);

    auto result = RunnerDB::GetById<sim::Device>(id);
    EXPECT_FALSE(result.has_value());
}

TEST_F(SimRunnerDbTest, Delete_ReturnsFalseForInvalidId)
{
    bool ret = RunnerDB::Delete<sim::Device>(99999);

    EXPECT_FALSE(ret);
}

TEST_F(SimRunnerDbTest, DeleteAll_RemovesAllRecords)
{
    sim::Server server1{};
    server1.pod_id = 1;
    snprintf(server1.version, sizeof(server1.version), "v1");
    RunnerDB::Add<sim::Server>(server1);

    sim::Server server2{};
    server2.pod_id = 2;
    snprintf(server2.version, sizeof(server2.version), "v2");
    RunnerDB::Add<sim::Server>(server2);

    sim::Server server3{};
    server3.pod_id = 3;
    snprintf(server3.version, sizeof(server3.version), "v3");
    RunnerDB::Add<sim::Server>(server3);

    bool ret = RunnerDB::DeleteAll<sim::Server>();
    EXPECT_TRUE(ret);

    auto results = RunnerDB::GetByPred<sim::Server>([](const sim::Server&) {
        return true;
    });
    EXPECT_EQ(results.size(), 0);
}

TEST_F(SimRunnerDbTest, MultipleAdd_MaintainsDataIntegrity)
{
    for (int i = 0; i < 10; i++) {
        sim::Device device{};
        device.server_id = 1;
        device.logic_id = static_cast<uint32_t>(i);
        device.physical_id = static_cast<uint32_t>(i + 100);
        RunnerDB::Add<sim::Device>(device);
    }

    auto results = RunnerDB::GetByPred<sim::Device>([](const sim::Device&) {
        return true;
    });

    EXPECT_GE(results.size(), 11);
}

class SimRunnerDbRankTest : public testing::Test {
protected:
    void SetUp() override
    {
        SetupTestData();

        sim::Device device{};
        auto ret = RunnerDB::GetOneByPred<sim::Device>([](const sim::Device& d) {
            return d.physical_id == 0;
        });
        ASSERT_TRUE(ret.second);

        sim::Rank rank{};
        rank.rank_id = 0;
        rank.device_id = ret.first.id;
        RunnerDB::Add<sim::Rank>(rank);
    }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimRunnerDbRankTest, Add_And_GetById_Rank)
{
    auto results = RunnerDB::GetByPred<sim::Rank>([](const sim::Rank&) {
        return true;
    });

    EXPECT_GE(results.size(), 1);
    EXPECT_EQ(results[0].rank_id, 0);
}

TEST_F(SimRunnerDbRankTest, QueryRankByDeviceId)
{
    sim::Device device{};
    auto ret = RunnerDB::GetOneByPred<sim::Device>([](const sim::Device& d) {
        return d.physical_id == 0;
    });
    ASSERT_TRUE(ret.second);

    auto rank = RunnerDB::GetOneByPred<sim::Rank>([devKey = ret.first.id](const sim::Rank& r) {
        return r.device_id == devKey;
    });

    EXPECT_TRUE(rank.second);
    EXPECT_EQ(rank.first.rank_id, 0);
}

class SimRunnerDbContextTest : public testing::Test {
protected:
    void SetUp() override
    {
        SetupTestData();

        sim::Device device{};
        auto ret = RunnerDB::GetOneByPred<sim::Device>([](const sim::Device& d) {
            return d.physical_id == 0;
        });
        ASSERT_TRUE(ret.second);

        sim::Context ctx{};
        ctx.device_id = ret.first.id;
        ctx.is_default = 1;
        ctx.run_id = 1;
        RunnerDB::Add<sim::Context>(ctx);
    }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimRunnerDbContextTest, GetById_Context)
{
    auto results = RunnerDB::GetByPred<sim::Context>([](const sim::Context&) {
        return true;
    });

    EXPECT_GE(results.size(), 1);
    EXPECT_EQ(results[0].is_default, 1);
}

TEST_F(SimRunnerDbContextTest, GetDefaultContext)
{
    auto ctx = RunnerDB::GetOneByPred<sim::Context>([](const sim::Context& c) {
        return c.is_default == 1;
    });

    EXPECT_TRUE(ctx.second);
    EXPECT_EQ(ctx.first.is_default, 1);
}

class SimRunnerDbStreamTest : public testing::Test {
protected:
    void SetUp() override { SetupTestData(); }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimRunnerDbStreamTest, Add_Stream)
{
    sim::Stream stream{};
    stream.ctx_id = 1;
    stream.priority = 1;
    stream.schedule_strategy = 1;

    uint64_t id = RunnerDB::Add<sim::Stream>(stream);

    auto result = RunnerDB::GetById<sim::Stream>(id);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->ctx_id, 1);
    EXPECT_EQ(result->priority, 1);
}

TEST_F(SimRunnerDbStreamTest, Update_Stream)
{
    sim::Stream stream{};
    stream.ctx_id = 1;
    uint64_t id = RunnerDB::Add<sim::Stream>(stream);

    RunnerDB::Update<sim::Stream>(id, [](sim::Stream& s) {
        s.priority = 5;
    });

    auto result = RunnerDB::GetById<sim::Stream>(id);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->priority, 5);
}

TEST_F(SimRunnerDbStreamTest, GetStreamsByCtxId)
{
    sim::Stream stream1{};
    stream1.ctx_id = 10;
    RunnerDB::Add<sim::Stream>(stream1);

    sim::Stream stream2{};
    stream2.ctx_id = 10;
    RunnerDB::Add<sim::Stream>(stream2);

    auto results = RunnerDB::GetByPred<sim::Stream>([](const sim::Stream& s) {
        return s.ctx_id == 10;
    });

    EXPECT_EQ(results.size(), 2);
}

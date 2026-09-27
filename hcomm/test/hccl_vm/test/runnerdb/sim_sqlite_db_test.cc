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

#include "db_sim_sqlite_db.h"
#include "sim_sqlite_table.h"

namespace {
const std::string kTestDbPath = "/tmp/test_sqlite_db.db";

void CleanUpDb()
{
    std::remove(kTestDbPath.c_str());
    std::remove((kTestDbPath + "-wal").c_str());
    std::remove((kTestDbPath + "-shm").c_str());
}

void SetupTestDb()
{
    CleanUpDb();
    sim::SqliteDatabase::SetDbPath(kTestDbPath);
}
} // namespace

class SimSqliteDbTest : public testing::Test {
protected:
    void SetUp() override
    {
        SetupTestDb();
        SimRunnerSqliteDB::Instance().ClearAll();
    }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimSqliteDbTest, GetAllTableName_ReturnsNonEmptyList)
{
    auto tableNames = SimRunnerSqliteDB::Instance().GetAllTableName();

    EXPECT_GT(tableNames.size(), 0);
}

TEST_F(SimSqliteDbTest, GetAllTableName_ContainsCoreTables)
{
    auto tableNames = SimRunnerSqliteDB::Instance().GetAllTableName();

    bool foundServer = false;
    bool foundDevice = false;
    bool foundHost = false;
    bool foundContext = false;

    for (const auto& name : tableNames) {
        if (name == "Server") {
            foundServer = true;
        }
        if (name == "Device") {
            foundDevice = true;
        }
        if (name == "Host") {
            foundHost = true;
        }
        if (name == "Context") {
            foundContext = true;
        }
    }

    EXPECT_TRUE(foundServer);
    EXPECT_TRUE(foundDevice);
    EXPECT_TRUE(foundHost);
    EXPECT_TRUE(foundContext);
}

TEST_F(SimSqliteDbTest, AddServer_IncreasesCount)
{
    sim::Server server{};
    server.pod_id = 100;
    snprintf(server.version, sizeof(server.version), "v1.0");

    uint64_t id = SimRunnerSqliteDB::Instance().Add<sim::Server>(server);

    EXPECT_GT(id, 0);
}

TEST_F(SimSqliteDbTest, FindServer_ReturnsInsertedRecord)
{
    sim::Server server{};
    server.pod_id = 200;
    snprintf(server.version, sizeof(server.version), "v2.0");

    uint64_t id = SimRunnerSqliteDB::Instance().Add<sim::Server>(server);
    auto found = SimRunnerSqliteDB::Instance().Find<sim::Server>(id);

    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(found->pod_id, 200);
    EXPECT_STREQ(found->version, "v2.0");
}

TEST_F(SimSqliteDbTest, FindServer_InvalidId_ReturnsNullopt)
{
    auto found = SimRunnerSqliteDB::Instance().Find<sim::Server>(99999);

    EXPECT_FALSE(found.has_value());
}

TEST_F(SimSqliteDbTest, QueryServer_ReturnsMatchingRecords)
{
    sim::Server server1{};
    server1.pod_id = 100;
    snprintf(server1.version, sizeof(server1.version), "v1.0");
    SimRunnerSqliteDB::Instance().Add<sim::Server>(server1);

    sim::Server server2{};
    server2.pod_id = 200;
    snprintf(server2.version, sizeof(server2.version), "v2.0");
    SimRunnerSqliteDB::Instance().Add<sim::Server>(server2);

    auto results = SimRunnerSqliteDB::Instance().QueryList<sim::Server>([](const sim::Server&) {
        return true;
    });

    EXPECT_GE(results.size(), 2);
}

TEST_F(SimSqliteDbTest, QueryServer_WithPredicate_ReturnsFilteredRecords)
{
    sim::Server server1{};
    server1.pod_id = 300;
    snprintf(server1.version, sizeof(server1.version), "v3");
    SimRunnerSqliteDB::Instance().Add<sim::Server>(server1);

    auto results = SimRunnerSqliteDB::Instance().QueryList<sim::Server>([](const sim::Server& s) {
        return s.pod_id == 300;
    });

    EXPECT_GE(results.size(), 1);
    EXPECT_EQ(results[0].pod_id, 300);
}

TEST_F(SimSqliteDbTest, QueryServer_ReturnsFirstMatch)
{
    sim::Server server1{};
    server1.pod_id = 100;
    SimRunnerSqliteDB::Instance().Add<sim::Server>(server1);

    sim::Server server2{};
    server2.pod_id = 200;
    SimRunnerSqliteDB::Instance().Add<sim::Server>(server2);

    auto result = SimRunnerSqliteDB::Instance().Query<sim::Server>([](const sim::Server& s) {
        return s.pod_id > 150;
    });

    EXPECT_TRUE(result.second);
    EXPECT_EQ(result.first.pod_id, 200);
}

TEST_F(SimSqliteDbTest, QueryServer_NoMatch_ReturnsFalse)
{
    auto result = SimRunnerSqliteDB::Instance().Query<sim::Server>([](const sim::Server& s) {
        return s.pod_id > 999999;
    });

    EXPECT_FALSE(result.second);
}

TEST_F(SimSqliteDbTest, UpdateServer_ModifiesRecord)
{
    sim::Server server{};
    server.pod_id = 100;
    snprintf(server.version, sizeof(server.version), "v1.0");
    uint64_t id = SimRunnerSqliteDB::Instance().Add<sim::Server>(server);

    bool ret = SimRunnerSqliteDB::Instance().Update<sim::Server>(id, [](sim::Server& s) {
        s.pod_id = 999;
    });

    EXPECT_TRUE(ret);

    auto found = SimRunnerSqliteDB::Instance().Find<sim::Server>(id);
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(found->pod_id, 999);
}

TEST_F(SimSqliteDbTest, UpdateServer_InvalidId_ReturnsFalse)
{
    bool ret = SimRunnerSqliteDB::Instance().Update<sim::Server>(99999, [](sim::Server& s) {
        s.pod_id = 999;
    });

    EXPECT_FALSE(ret);
}

TEST_F(SimSqliteDbTest, DeleteServer_RemovesRecord)
{
    sim::Server server{};
    server.pod_id = 100;
    uint64_t id = SimRunnerSqliteDB::Instance().Add<sim::Server>(server);

    bool ret = SimRunnerSqliteDB::Instance().Delete<sim::Server>(id);
    EXPECT_TRUE(ret);

    auto found = SimRunnerSqliteDB::Instance().Find<sim::Server>(id);
    EXPECT_FALSE(found.has_value());
}

TEST_F(SimSqliteDbTest, DeleteServer_InvalidId_ReturnsFalse)
{
    bool ret = SimRunnerSqliteDB::Instance().Delete<sim::Server>(99999);

    EXPECT_FALSE(ret);
}

TEST_F(SimSqliteDbTest, DeleteAllServers_RemovesAllRecords)
{
    for (int i = 0; i < 5; i++) {
        sim::Server server{};
        server.pod_id = 100 + i;
        SimRunnerSqliteDB::Instance().Add<sim::Server>(server);
    }

    bool ret = SimRunnerSqliteDB::Instance().DeleteAll<sim::Server>();
    EXPECT_TRUE(ret);

    auto results = SimRunnerSqliteDB::Instance().QueryList<sim::Server>([](const sim::Server&) {
        return true;
    });
    EXPECT_EQ(results.size(), 0);
}

TEST_F(SimSqliteDbTest, AddMultipleDevices_MaintainsDataIntegrity)
{
    for (int i = 0; i < 10; i++) {
        sim::Device device{};
        device.server_id = 1;
        device.logic_id = static_cast<uint32_t>(i);
        device.physical_id = static_cast<uint32_t>(i * 2);
        SimRunnerSqliteDB::Instance().Add<sim::Device>(device);
    }

    auto results = SimRunnerSqliteDB::Instance().QueryList<sim::Device>([](const sim::Device&) {
        return true;
    });

    EXPECT_EQ(results.size(), 10);

    for (const auto& device : results) {
        auto found = SimRunnerSqliteDB::Instance().Find<sim::Device>(device.id);
        ASSERT_TRUE(found.has_value());
        EXPECT_EQ(found->id, device.id);
        EXPECT_EQ(found->server_id, device.server_id);
    }
}

TEST_F(SimSqliteDbTest, ClearAll_ClearsAllTables)
{
    sim::Server server{};
    server.pod_id = 9999;
    snprintf(server.version, sizeof(server.version), "v_clear");
    SimRunnerSqliteDB::Instance().Add<sim::Server>(server);

    auto afterAdd = SimRunnerSqliteDB::Instance().QueryList<sim::Server>([](const sim::Server& s) {
        return s.pod_id == 9999;
    });
    EXPECT_GE(afterAdd.size(), 1);

    EXPECT_NO_THROW(SimRunnerSqliteDB::Instance().ClearAll());
}

class SimSqliteDbHostTest : public testing::Test {
protected:
    void SetUp() override
    {
        SetupTestDb();
        SimRunnerSqliteDB::Instance().ClearAll();
    }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimSqliteDbHostTest, AddHost_InsertsRecord)
{
    sim::Host host{};
    host.server_id = 1;
    snprintf(host.ip_addr, sizeof(host.ip_addr), "192.168.1.100");
    host.arch = 1;

    uint64_t id = SimRunnerSqliteDB::Instance().Add<sim::Host>(host);

    EXPECT_GT(id, 0);

    auto found = SimRunnerSqliteDB::Instance().Find<sim::Host>(id);
    ASSERT_TRUE(found.has_value());
    EXPECT_STREQ(found->ip_addr, "192.168.1.100");
    EXPECT_EQ(found->arch, 1);
}

TEST_F(SimSqliteDbHostTest, QueryHostByIpAddr)
{
    sim::Host host{};
    host.server_id = 1;
    snprintf(host.ip_addr, sizeof(host.ip_addr), "192.168.2.200");
    host.arch = 2;
    SimRunnerSqliteDB::Instance().Add<sim::Host>(host);

    auto result = SimRunnerSqliteDB::Instance().Query<sim::Host>([](const sim::Host& h) {
        return strcmp(h.ip_addr, "192.168.2.200") == 0;
    });

    EXPECT_TRUE(result.second);
    EXPECT_STREQ(result.first.ip_addr, "192.168.2.200");
}

class SimSqliteDbContextTest : public testing::Test {
protected:
    void SetUp() override
    {
        SetupTestDb();
        SimRunnerSqliteDB::Instance().ClearAll();
    }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimSqliteDbContextTest, AddContext_InsertsRecord)
{
    sim::Context ctx{};
    ctx.device_id = 1;
    ctx.is_default = 1;
    ctx.run_id = 100;

    uint64_t id = SimRunnerSqliteDB::Instance().Add<sim::Context>(ctx);

    EXPECT_GT(id, 0);

    auto found = SimRunnerSqliteDB::Instance().Find<sim::Context>(id);
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(found->device_id, 1);
    EXPECT_EQ(found->is_default, 1);
}

TEST_F(SimSqliteDbContextTest, GetDefaultContext)
{
    sim::Context ctx{};
    ctx.device_id = 999;
    ctx.is_default = 1;
    ctx.run_id = 123;
    SimRunnerSqliteDB::Instance().Add<sim::Context>(ctx);

    auto result = SimRunnerSqliteDB::Instance().Query<sim::Context>([](const sim::Context& c) {
        return c.is_default == 1 && c.device_id == 999;
    });

    EXPECT_TRUE(result.second);
    EXPECT_EQ(result.first.device_id, 999);
}

class SimSqliteDbStreamTest : public testing::Test {
protected:
    void SetUp() override
    {
        SetupTestDb();
        SimRunnerSqliteDB::Instance().ClearAll();
    }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimSqliteDbStreamTest, AddStream_InsertsRecord)
{
    sim::Stream stream{};
    stream.ctx_id = 1;
    stream.priority = 5;

    uint64_t id = SimRunnerSqliteDB::Instance().Add<sim::Stream>(stream);

    EXPECT_GT(id, 0);

    auto found = SimRunnerSqliteDB::Instance().Find<sim::Stream>(id);
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(found->priority, 5);
}

TEST_F(SimSqliteDbStreamTest, UpdateStream_ModifiesRecord)
{
    sim::Stream stream{};
    stream.ctx_id = 1;
    uint64_t id = SimRunnerSqliteDB::Instance().Add<sim::Stream>(stream);

    bool ret = SimRunnerSqliteDB::Instance().Update<sim::Stream>(id, [](sim::Stream& s) {
        s.priority = 10;
        s.schedule_strategy = 2;
    });

    EXPECT_TRUE(ret);

    auto found = SimRunnerSqliteDB::Instance().Find<sim::Stream>(id);
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(found->priority, 10);
    EXPECT_EQ(found->schedule_strategy, 2);
}

TEST_F(SimSqliteDbStreamTest, GetStreamsByCtxId)
{
    sim::Stream stream1{};
    stream1.ctx_id = 10;
    SimRunnerSqliteDB::Instance().Add<sim::Stream>(stream1);

    sim::Stream stream2{};
    stream2.ctx_id = 10;
    SimRunnerSqliteDB::Instance().Add<sim::Stream>(stream2);

    sim::Stream stream3{};
    stream3.ctx_id = 20;
    SimRunnerSqliteDB::Instance().Add<sim::Stream>(stream3);

    auto results = SimRunnerSqliteDB::Instance().QueryList<sim::Stream>([](const sim::Stream& s) {
        return s.ctx_id == 10;
    });

    EXPECT_EQ(results.size(), 2);
}

class SimSqliteDbTableTest : public testing::Test {
protected:
    void SetUp() override
    {
        SetupTestDb();
        SimRunnerSqliteDB::Instance().ClearAll();
    }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimSqliteDbTableTest, AddRank_InsertsRecord)
{
    sim::Rank rank{};
    rank.rank_id = 0;
    rank.device_id = 1;

    uint64_t id = SimRunnerSqliteDB::Instance().Add<sim::Rank>(rank);

    EXPECT_GT(id, 0);

    auto found = SimRunnerSqliteDB::Instance().Find<sim::Rank>(id);
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(found->rank_id, 0);
    EXPECT_EQ(found->device_id, 1);
}

TEST_F(SimSqliteDbTableTest, AddPort_InsertsRecord)
{
    sim::Port port{};
    port.device_id = 1;
    snprintf(port.name, sizeof(port.name), "eth0");
    port.status = 1;

    uint64_t id = SimRunnerSqliteDB::Instance().Add<sim::Port>(port);

    EXPECT_GT(id, 0);

    auto found = SimRunnerSqliteDB::Instance().Find<sim::Port>(id);
    ASSERT_TRUE(found.has_value());
    EXPECT_STREQ(found->name, "eth0");
}

TEST_F(SimSqliteDbTableTest, AddCcu_InsertsRecord)
{
    sim::Ccu ccu{};
    ccu.device_id = 1;
    ccu.die_id = 0;
    ccu.status = 1;

    uint64_t id = SimRunnerSqliteDB::Instance().Add<sim::Ccu>(ccu);

    EXPECT_GT(id, 0);

    auto found = SimRunnerSqliteDB::Instance().Find<sim::Ccu>(id);
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(found->die_id, 0);
    EXPECT_EQ(found->status, 1);
}

TEST_F(SimSqliteDbTableTest, AddCcuResource_InsertsRecord)
{
    sim::CcuResource ccuRes{};
    ccuRes.ccu_id = 1;
    ccuRes.state = 0;
    ccuRes.instr_cnt = 0;

    uint64_t id = SimRunnerSqliteDB::Instance().Add<sim::CcuResource>(ccuRes);

    EXPECT_GT(id, 0);

    auto found = SimRunnerSqliteDB::Instance().Find<sim::CcuResource>(id);
    ASSERT_TRUE(found.has_value());
    EXPECT_EQ(found->ccu_id, 1);
}

TEST_F(SimSqliteDbTest, RunModeConfig_WriteThenRead_RoundTrip)
{
    // RunModeConfig 表惰性注册，ClearAll 不一定覆盖到它，写入前先显式清空，避免跨用例残留。
    SimRunnerSqliteDB::Instance().DeleteAll<sim::RunModeConfig>();
    sim::RunModeConfig cfg{};
    cfg.mode = 1;
    SimRunnerSqliteDB::Instance().Add<sim::RunModeConfig>(cfg);

    auto got = SimRunnerSqliteDB::Instance().Query<sim::RunModeConfig>([](const sim::RunModeConfig&) {
        return true;
    });
    EXPECT_TRUE(got.second);
    EXPECT_EQ(got.first.mode, 1);
}

TEST_F(SimSqliteDbTest, RunModeConfig_DeleteAllThenWrite_LatestSingleRowWins)
{
    // RunModeConfig 表惰性注册，ClearAll 不一定覆盖到它，进表前先显式清空，避免跨用例残留。
    SimRunnerSqliteDB::Instance().DeleteAll<sim::RunModeConfig>();
    // 先写仅校验模式行，DeleteAll 清空后再写 clean 行：单行覆盖语义，读到的是清空后的最新值。
    sim::RunModeConfig checkOnly{};
    checkOnly.mode = 1;
    SimRunnerSqliteDB::Instance().Add<sim::RunModeConfig>(checkOnly);

    SimRunnerSqliteDB::Instance().DeleteAll<sim::RunModeConfig>();

    sim::RunModeConfig clean{};
    clean.mode = 0;
    SimRunnerSqliteDB::Instance().Add<sim::RunModeConfig>(clean);

    auto all = SimRunnerSqliteDB::Instance().QueryList<sim::RunModeConfig>([](const sim::RunModeConfig&) {
        return true;
    });
    ASSERT_EQ(all.size(), 1);
    EXPECT_EQ(all[0].mode, 0);
}

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

#include "db_sim_runner_common.h"
#include "db_sim_runner_db.h"
#include "db_sim_sqlite_db.h"

extern uint64_t g_cur_server_key;

namespace sim {
aclError GetServerByKey(uint64_t serverKey, sim::Server& server);
uint32_t GetCubeCoreCount(uint64_t deviceId);
} // namespace sim

namespace {
const std::string kTestDbPath = "/tmp/test_sim_runner_common.db";

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
    device.super_device_id = 0;
    RunnerDB::Add<sim::Device>(device);

    device.logic_id = 1;
    device.physical_id = 1;
    RunnerDB::Add<sim::Device>(device);

    g_cur_server_key = 1;
}
} // namespace

class SimRunnerCommonTest : public testing::Test {
protected:
    void SetUp() override { SetupTestData(); }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimRunnerCommonTest, GetDeviceByLogicId_WhenDeviceExists_ReturnSuccess)
{
    sim::Device device{};
    auto ret = sim::GetDeviceByLogicId(0, device);

    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_EQ(device.logic_id, 0);
    EXPECT_EQ(device.physical_id, 0);
    EXPECT_EQ(device.server_id, 1);
}

TEST_F(SimRunnerCommonTest, GetDeviceByLogicId_WhenDeviceNotExists_ReturnError)
{
    sim::Device device{};
    auto ret = sim::GetDeviceByLogicId(999, device);

    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

TEST_F(SimRunnerCommonTest, GetDeviceByPhysicalId_WhenDeviceExists_ReturnSuccess)
{
    sim::Device device{};
    auto ret = sim::GetDeviceByPhysicalId(0, device);

    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_EQ(device.physical_id, 0);
    EXPECT_EQ(device.logic_id, 0);
}

TEST_F(SimRunnerCommonTest, GetDeviceByPhysicalId_WhenDeviceNotExists_ReturnError)
{
    sim::Device device{};
    auto ret = sim::GetDeviceByPhysicalId(999, device);

    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

TEST_F(SimRunnerCommonTest, UpdateDeviceLogicId_WhenValidParams_UpdateSuccessfully)
{
    auto ret = sim::UpdateDeviceLogicId(1, 0, 100, 0);

    EXPECT_EQ(ret, ACL_SUCCESS);

    sim::Device device{};
    auto getRet = sim::GetDeviceByPhysicalId(0, device);
    EXPECT_EQ(getRet, ACL_SUCCESS);
    EXPECT_EQ(device.logic_id, 100);
}

TEST_F(SimRunnerCommonTest, UpdateDeviceLogicId_WhenServerKeyInvalid_ReturnError)
{
    auto ret = sim::UpdateDeviceLogicId(999, 0, 100, 0);

    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

TEST_F(SimRunnerCommonTest, UpdateDeviceLogicId_WhenPhyDevIdInvalid_ReturnError)
{
    auto ret = sim::UpdateDeviceLogicId(1, 999, 100, 0);

    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

TEST_F(SimRunnerCommonTest, UpdateSuperDeviceId_WhenValidParams_UpdateSuccessfully)
{
    auto ret = sim::UpdateSuperDeviceId(0, 200);

    EXPECT_EQ(ret, ACL_SUCCESS);

    sim::Device device{};
    auto getRet = sim::GetDeviceByLogicId(0, device);
    EXPECT_EQ(getRet, ACL_SUCCESS);
    EXPECT_EQ(device.super_device_id, 200);
}

TEST_F(SimRunnerCommonTest, UpdateSuperDeviceId_WhenLogicIdInvalid_ReturnError)
{
    auto ret = sim::UpdateSuperDeviceId(999, 200);

    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

TEST_F(SimRunnerCommonTest, GetServerByKey_WhenServerExists_ReturnSuccess)
{
    sim::Server server{};
    auto ret = RunnerDB::GetOneByPred<sim::Server>([](const sim::Server& s) {
        return s.id == 1;
    });

    EXPECT_TRUE(ret.second);
    EXPECT_EQ(ret.first.pod_id, 100);
}

TEST_F(SimRunnerCommonTest, GetServerByKey_WhenServerNotExists_ReturnError)
{
    auto ret = RunnerDB::GetOneByPred<sim::Server>([](const sim::Server& s) {
        return s.id == 999;
    });

    EXPECT_FALSE(ret.second);
}

TEST_F(SimRunnerCommonTest, GetDeviceByServerKeyAndPhysicalId_WhenValid_ReturnSuccess)
{
    auto ret = RunnerDB::GetOneByPred<sim::Device>([](const sim::Device& d) {
        return d.server_id == 1 && d.physical_id == 0;
    });

    EXPECT_TRUE(ret.second);
    EXPECT_EQ(ret.first.physical_id, 0);
    EXPECT_EQ(ret.first.server_id, 1);
}

TEST_F(SimRunnerCommonTest, GetDeviceByServerKeyAndPhysicalId_WhenInvalidServerKey_ReturnError)
{
    auto ret = RunnerDB::GetOneByPred<sim::Device>([](const sim::Device& d) {
        return d.server_id == 999 && d.physical_id == 0;
    });

    EXPECT_FALSE(ret.second);
}

TEST_F(SimRunnerCommonTest, GetDeviceByServerKeyAndPhysicalId_WhenInvalidPhysicalId_ReturnError)
{
    auto ret = RunnerDB::GetOneByPred<sim::Device>([](const sim::Device& d) {
        return d.server_id == 1 && d.physical_id == 999;
    });

    EXPECT_FALSE(ret.second);
}

TEST_F(SimRunnerCommonTest, GetServerByKey_Found_ReturnSuccess)
{
    sim::Server server{};
    auto ret = sim::GetServerByKey(1, server);
    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_EQ(server.pod_id, 100);
}

TEST_F(SimRunnerCommonTest, GetServerByKey_NotFound_ReturnError)
{
    sim::Server server{};
    auto ret = sim::GetServerByKey(999, server);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

class SimRunnerCommonCcuTest : public testing::Test {
protected:
    void SetUp() override
    {
        SetupTestData();

        sim::Device device{};
        auto ret = sim::GetDeviceByPhysicalId(0, device);
        ASSERT_EQ(ret, ACL_SUCCESS);

        sim::Ccu ccu{};
        ccu.device_id = device.id;
        ccu.die_id = 0;
        ccu.status = 1;
        RunnerDB::Add<sim::Ccu>(ccu);

        sim::CcuResource ccuRes{};
        ccuRes.ccu_id = 1;
        RunnerDB::Add<sim::CcuResource>(ccuRes);
    }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimRunnerCommonCcuTest, GetCcuFromDeviceByDieId_WhenValid_ReturnSuccess)
{
    sim::Ccu ccu{};
    auto ret = sim::GetCcuFromDeviceByDieId(1, 0, ccu);

    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_EQ(ccu.die_id, 0);
    EXPECT_EQ(ccu.device_id, 1);
}

TEST_F(SimRunnerCommonCcuTest, GetCcuFromDeviceByDieId_WhenNotFound_ReturnError)
{
    sim::Ccu ccu{};
    auto ret = sim::GetCcuFromDeviceByDieId(1, 99, ccu);

    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

TEST_F(SimRunnerCommonCcuTest, GetCcuResourceByCcu_WhenValid_ReturnSuccess)
{
    sim::CcuResource ccuRes{};
    auto ret = sim::GetCcuResourceByCcu(1, ccuRes);

    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_EQ(ccuRes.ccu_id, 1);
}

TEST_F(SimRunnerCommonCcuTest, GetCcuResourceByCcu_WhenNotFound_ReturnError)
{
    sim::CcuResource ccuRes{};
    auto ret = sim::GetCcuResourceByCcu(999, ccuRes);

    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

class SimRunnerCommonContextTest : public testing::Test {
protected:
    void SetUp() override
    {
        SetupTestData();

        sim::Device device{};
        auto ret = sim::GetDeviceByPhysicalId(0, device);
        ASSERT_EQ(ret, ACL_SUCCESS);

        sim::Context ctx{};
        ctx.device_id = device.id;
        ctx.is_default = 1;
        ctx.run_id = 1;
        RunnerDB::Add<sim::Context>(ctx);
    }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimRunnerCommonContextTest, GetContextByDevId_WhenValid_ReturnSuccess)
{
    sim::Device device{};
    auto ret = sim::GetDeviceByPhysicalId(0, device);
    ASSERT_EQ(ret, ACL_SUCCESS);

    sim::Context ctx{};
    auto getRet = sim::GetContextByDevId(device.id, ctx);

    EXPECT_EQ(getRet, ACL_SUCCESS);
    EXPECT_EQ(ctx.device_id, device.id);
    EXPECT_EQ(ctx.is_default, 1);
}

TEST_F(SimRunnerCommonContextTest, GetContextByDevId_WhenNotFound_ReturnError)
{
    sim::Context ctx{};
    auto ret = sim::GetContextByDevId(999, ctx);

    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

class SimRunnerCommonPortTest : public testing::Test {
protected:
    void SetUp() override
    {
        SetupTestData();

        sim::Device device{};
        auto ret = sim::GetDeviceByPhysicalId(0, device);
        ASSERT_EQ(ret, ACL_SUCCESS);

        sim::Port port{};
        port.device_id = device.id;
        snprintf(port.name, sizeof(port.name), "eth0");
        port.status = 1;
        RunnerDB::Add<sim::Port>(port);

        sim::Port port2{};
        port2.device_id = device.id;
        snprintf(port2.name, sizeof(port2.name), "eth1");
        port2.status = 1;
        RunnerDB::Add<sim::Port>(port2);
    }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimRunnerCommonPortTest, GetPortByName_WhenValid_ReturnSuccess)
{
    sim::Port port{};
    auto ret = sim::GetPortByName(1, 0, "eth0", port);

    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_STREQ(port.name, "eth0");
}

TEST_F(SimRunnerCommonPortTest, GetPortByName_WhenNotFound_ReturnError)
{
    sim::Port port{};
    auto ret = sim::GetPortByName(1, 0, "nonexistent", port);

    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

class SimRunnerCommonCountTest : public testing::Test {
protected:
    void SetUp() override
    {
        SetupTestData();

        sim::Device device{};
        auto ret = sim::GetDeviceByLogicId(0, device);
        ASSERT_EQ(ret, ACL_SUCCESS);

        sim::TaskSchedulerDevice tsDev{};
        tsDev.device_id = device.id;
        tsDev.type = static_cast<uint8_t>(sim::TS_DEV_TYPE_CPU);
        RunnerDB::Add<sim::TaskSchedulerDevice>(tsDev);

        tsDev.type = static_cast<uint8_t>(sim::TS_DEV_TYPE_SCALAR);
        RunnerDB::Add<sim::TaskSchedulerDevice>(tsDev);
    }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimRunnerCommonCountTest, GetAICpuCount_WhenDeviceExists_ReturnCorrectCount)
{
    auto count = sim::GetAICpuCount(0);

    EXPECT_EQ(count, 1);
}

TEST_F(SimRunnerCommonCountTest, GetAICpuCount_WhenDeviceNotExists_ReturnZero)
{
    auto count = sim::GetAICpuCount(999);

    EXPECT_EQ(count, 0);
}

TEST_F(SimRunnerCommonCountTest, GetAICoreCount_WhenDeviceExists_ReturnCorrectCount)
{
    auto count = sim::GetAICoreCount(0);

    EXPECT_GE(count, 1);
}

TEST_F(SimRunnerCommonCountTest, GetAICoreCount_WhenDeviceNotExists_ReturnZero)
{
    auto count = sim::GetAICoreCount(999);

    EXPECT_EQ(count, 0);
}

class SimRunnerCommonVectorCubeTest : public testing::Test {
protected:
    void SetUp() override
    {
        SetupTestData();

        sim::Device device{};
        auto ret = sim::GetDeviceByLogicId(0, device);
        ASSERT_EQ(ret, ACL_SUCCESS);

        sim::TaskSchedulerDevice tsDev{};
        tsDev.device_id = device.id;
        tsDev.type = static_cast<uint8_t>(sim::TS_DEV_TYPE_SCALAR);
        uint64_t tsId = RunnerDB::Add<sim::TaskSchedulerDevice>(tsDev);

        sim::ComputeDie die{};
        die.ts_id = tsId;
        die.type = static_cast<uint8_t>(sim::COMPUTE_TYPE_VECTOR);
        RunnerDB::Add<sim::ComputeDie>(die);

        die.type = static_cast<uint8_t>(sim::COMPUTE_TYPE_CUBE);
        RunnerDB::Add<sim::ComputeDie>(die);
    }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimRunnerCommonVectorCubeTest, GetVectorCoreCount_WhenDeviceExists_ReturnCorrectCount)
{
    auto count = sim::GetVectorCoreCount(0);

    EXPECT_GE(count, 1);
}

TEST_F(SimRunnerCommonVectorCubeTest, GetVectorCoreCount_WhenDeviceNotExists_ReturnZero)
{
    auto count = sim::GetVectorCoreCount(999);

    EXPECT_EQ(count, 0);
}

TEST_F(SimRunnerCommonVectorCubeTest, GetCubeCoreCount_WhenDeviceExists_ReturnCorrectCount)
{
    auto count = sim::GetCubeCoreCount(0);
    EXPECT_GE(count, 1);
}

TEST_F(SimRunnerCommonVectorCubeTest, GetCubeCoreCount_WhenDeviceNotExists_ReturnZero)
{
    auto count = sim::GetCubeCoreCount(999);
    EXPECT_EQ(count, 0);
}

class SimRunnerCommonRankEndpointTest : public testing::Test {
protected:
    void SetUp() override
    {
        SetupTestData();

        sim::Device device{};
        auto ret = sim::GetDeviceByLogicId(0, device);
        ASSERT_EQ(ret, ACL_SUCCESS);

        sim::Rank rank{};
        rank.device_id = device.id;
        rank.rank_id = 5;
        RunnerDB::Add<sim::Rank>(rank);
    }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(SimRunnerCommonRankEndpointTest, GetRankIdByDeviceId_WhenExists_ReturnsRankId)
{
    auto ret = RunnerDB::GetOneByPred<sim::Rank>([](const sim::Rank& r) {
        return r.rank_id == 5;
    });
    ASSERT_TRUE(ret.second);

    int rankId = sim::GetRankIdByDeviceId(ret.first.device_id);
    EXPECT_EQ(rankId, 5);
}

TEST_F(SimRunnerCommonRankEndpointTest, GetRankIdByDeviceId_WhenNotExists_ReturnsZero)
{
    int rankId = sim::GetRankIdByDeviceId(999);
    EXPECT_EQ(rankId, 0);
}

TEST_F(SimRunnerCommonRankEndpointTest, GetEndPointByIpAddr_WhenNotExists_ReturnError)
{
    sim::EndPoint ep{};
    auto ret = sim::GetEndPointByIpAddr("192.168.99.99", ep);
    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

class SimRunnerCommonFullPortTest : public testing::Test {
protected:
    void SetUp() override
    {
        SetupTestData();

        sim::Device device{};
        auto ret = sim::GetDeviceByPhysicalId(0, device);
        ASSERT_EQ(ret, ACL_SUCCESS);

        sim::Port port{};
        port.device_id = device.id;
        port.die_id = 0;
        snprintf(port.name, sizeof(port.name), "eth0");
        port.status = 1;
        portId_ = RunnerDB::Add<sim::Port>(port);

        sim::Ccu ccu{};
        ccu.device_id = device.id;
        ccu.die_id = 0;
        ccu.status = 1;
        ccuId_ = RunnerDB::Add<sim::Ccu>(ccu);

        snprintf(port.name, sizeof(port.name), "ccu_eth0");
        RunnerDB::Add<sim::Port>(port);
    }

    void TearDown() override { CleanUpDb(); }

    uint64_t portId_{0};
    uint64_t ccuId_{0};
};

TEST_F(SimRunnerCommonFullPortTest, GetPortById_WhenValid_ReturnSuccess)
{
    sim::Port port{};
    auto ret = sim::GetPortById(portId_, port);

    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_EQ(port.id, portId_);
}

TEST_F(SimRunnerCommonFullPortTest, GetPortById_WhenNotFound_ReturnError)
{
    sim::Port port{};
    auto ret = sim::GetPortById(999, port);

    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

TEST_F(SimRunnerCommonFullPortTest, GetPortByName_DeviceNotFound_ReturnError)
{
    sim::Port port{};
    auto ret = sim::GetPortByName(999, 0, "eth0", port);

    EXPECT_EQ(ret, ACL_ERROR_INVALID_PARAM);
}

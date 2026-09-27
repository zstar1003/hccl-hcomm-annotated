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

#include "ascend_hal.h"
#include "db_sim_runner_common.h"
#include "db_sim_runner_db.h"
#include "db_sim_sqlite_db.h"

extern uint64_t g_cur_server_key;

namespace {
const std::string kTestDbPath = "/tmp/test_ascend_hal_stub.db";

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

    g_cur_server_key = 1;
}
} // namespace

class AscendHalStubTest : public testing::Test {
protected:
    void SetUp() override { SetupTestData(); }

    void TearDown() override { CleanUpDb(); }
};

TEST_F(AscendHalStubTest, halEschedAttachDevice_ReturnsSuccess)
{
    drvError_t result = halEschedAttachDevice(0);
    EXPECT_EQ(result, DRV_ERROR_NONE);
}

TEST_F(AscendHalStubTest, halEschedDettachDevice_ReturnsSuccess)
{
    drvError_t result = halEschedDettachDevice(0);
    EXPECT_EQ(result, DRV_ERROR_NONE);
}

TEST_F(AscendHalStubTest, halEschedCreateGrpEx_ReturnsSuccess)
{
    struct esched_grp_para grpPara;
    unsigned int grpId = 0;

    drvError_t result = halEschedCreateGrpEx(0, &grpPara, &grpId);
    EXPECT_EQ(result, DRV_ERROR_NONE);
}

TEST_F(AscendHalStubTest, halEschedSubscribeEvent_ReturnsSuccess)
{
    drvError_t result = halEschedSubscribeEvent(0, 0, 0, 0);
    EXPECT_EQ(result, DRV_ERROR_NONE);
}

TEST_F(AscendHalStubTest, halEschedQueryInfo_ReturnsSuccess)
{
    struct esched_input_info inPut;
    struct esched_output_info outPut;

    drvError_t result = halEschedQueryInfo(0, ESCHED_QUERY_TYPE::QUERY_TYPE_LOCAL_GRP_ID, &inPut, &outPut);
    EXPECT_EQ(result, DRV_ERROR_NONE);
}

TEST_F(AscendHalStubTest, halEschedWaitEvent_ReturnsSuccess)
{
    struct event_info event;

    drvError_t result = halEschedWaitEvent(0, 0, 0, 0, &event);
    EXPECT_EQ(result, DRV_ERROR_NONE);
}

TEST_F(AscendHalStubTest, halEschedRegisterAckFunc_ReturnsSuccess)
{
    drvError_t result = halEschedRegisterAckFunc(0, EVENT_ID::EVENT_RANDOM_KERNEL, nullptr);
    EXPECT_EQ(result, DRV_ERROR_NONE);
}

TEST_F(AscendHalStubTest, drvGetDevNum_ReturnsOne)
{
    uint32_t num_dev = 0;

    drvError_t result = drvGetDevNum(&num_dev);
    EXPECT_EQ(result, DRV_ERROR_NONE);
    EXPECT_EQ(num_dev, 1);
}

TEST_F(AscendHalStubTest, halGetChipInfo_ReturnsSuccess)
{
    halChipInfo chipInfo;

    drvError_t result = halGetChipInfo(0, &chipInfo);
    EXPECT_EQ(result, DRV_ERROR_NONE);
}

TEST_F(AscendHalStubTest, halHostRegister_ReturnsSuccess)
{
    void* dst_ptr = nullptr;

    drvError_t result = halHostRegister(nullptr, 1024, 0, 0, &dst_ptr);
    EXPECT_EQ(result, DRV_ERROR_NONE);
}

TEST_F(AscendHalStubTest, halHostUnregister_ReturnsSuccess)
{
    drvError_t result = halHostUnregister(nullptr, 0);
    EXPECT_EQ(result, DRV_ERROR_NONE);
}

TEST_F(AscendHalStubTest, halMemCtl_ReturnsSuccess)
{
    size_t out_size_ret = 0;

    drvError_t result = halMemCtl(0, nullptr, 0, nullptr, &out_size_ret);
    EXPECT_EQ(result, DRV_ERROR_NONE);
}

TEST_F(AscendHalStubTest, drvGetPlatformInfo_ReturnsOne)
{
    uint32_t info = 0;

    drvError_t result = drvGetPlatformInfo(&info);
    EXPECT_EQ(result, DRV_ERROR_NONE);
    EXPECT_EQ(info, 1);
}

TEST_F(AscendHalStubTest, drvQueryProcessHostPid_ReturnsSuccess)
{
    unsigned int chip_id = 0;
    unsigned int vfid = 0;
    unsigned int host_pid = 0;
    unsigned int cp_type = 0;

    drvError_t result = drvQueryProcessHostPid(0, &chip_id, &vfid, &host_pid, &cp_type);
    EXPECT_EQ(result, DRV_ERROR_NONE);
}

TEST_F(AscendHalStubTest, halGetDeviceInfo_ModuleTypeSystem)
{
    int64_t value = 0;

    drvError_t result = halGetDeviceInfo(0, MODULE_TYPE_SYSTEM, INFO_TYPE_CORE_NUM, &value);
    EXPECT_EQ(result, DRV_ERROR_NONE);
    EXPECT_EQ(value, 1);
}

TEST_F(AscendHalStubTest, halGetDeviceInfo_OtherModule)
{
    int64_t value = 0;

    drvError_t result = halGetDeviceInfo(0, 100, 100, &value);
    EXPECT_EQ(result, DRV_ERROR_NONE);
    EXPECT_EQ(value, 0);
}

TEST_F(AscendHalStubTest, halEschedSubmitEvent_ReturnsSuccess)
{
    struct event_summary event;

    drvError_t result = halEschedSubmitEvent(0, &event);
    EXPECT_EQ(result, DRV_ERROR_NONE);
}

TEST_F(AscendHalStubTest, halBindCgroup_ReturnsSuccess)
{
    drvError_t result = halBindCgroup(BIND_AICPU_CGROUP);
    EXPECT_EQ(result, DRV_ERROR_NONE);
}

TEST_F(AscendHalStubTest, halGetAPIVersion_ReturnsSuccess)
{
    int halAPIVersion = 0;

    drvError_t result = halGetAPIVersion(&halAPIVersion);
    EXPECT_EQ(result, DRV_ERROR_NONE);
}

TEST_F(AscendHalStubTest, halSensorNodeRegister_ReturnsSuccess)
{
    struct halSensorNodeCfg cfg;
    uint64_t handle = 0;

    drvError_t result = halSensorNodeRegister(0, &cfg, &handle);
    EXPECT_EQ(result, DRV_ERROR_NONE);
}

TEST_F(AscendHalStubTest, halSensorNodeUnregister_ReturnsSuccess)
{
    drvError_t result = halSensorNodeUnregister(0, 0);
    EXPECT_EQ(result, DRV_ERROR_NONE);
}

TEST_F(AscendHalStubTest, halSensorNodeUpdateState_ReturnsSuccess)
{
    drvError_t result = halSensorNodeUpdateState(0, 0, 0, halGeneralEventType_t::GENERAL_EVENT_TYPE_OCCUR);
    EXPECT_EQ(result, DRV_ERROR_NONE);
}

TEST_F(AscendHalStubTest, drvDeviceGetPhyIdByIndex_ReturnsSuccess)
{
    uint32_t phyId = 0;

    drvError_t result = drvDeviceGetPhyIdByIndex(0, &phyId);
    EXPECT_EQ(result, DRV_ERROR_NONE);
}

TEST_F(AscendHalStubTest, drvMemcpy_ReturnsSuccess)
{
    drvError_t result = drvMemcpy(0, 0, 0, 0);
    EXPECT_EQ(result, DRV_ERROR_NONE);
}

TEST_F(AscendHalStubTest, halGrpQuery_ReturnsZero)
{
    unsigned int outLen = 0;

    int result = halGrpQuery(GroupQueryCmdType::GRP_QUERY_GROUP, nullptr, 0, nullptr, &outLen);
    EXPECT_EQ(result, 0);
}

TEST_F(AscendHalStubTest, drvDeviceGetBareTgid_ReturnsPid)
{
    pid_t pid = drvDeviceGetBareTgid();
    EXPECT_GT(pid, 0);
}

TEST_F(AscendHalStubTest, halResourceIdCheck_ReturnsSuccess)
{
    struct drvResIdKey info;

    drvError_t result = halResourceIdCheck(&info);
    EXPECT_EQ(result, DRV_ERROR_NONE);
}

TEST_F(AscendHalStubTest, halSqCqQuery_NullInfo)
{
    drvError_t result = halSqCqQuery(0, nullptr);
    EXPECT_EQ(result, DRV_ERROR_INNER_ERR);
}

TEST_F(AscendHalStubTest, halSqCqQuery_SqHead)
{
    struct halSqCqQueryInfo info;
    info.prop = DRV_SQCQ_PROP_SQ_HEAD;

    drvError_t result = halSqCqQuery(0, &info);
    EXPECT_EQ(result, DRV_ERROR_NONE);
    EXPECT_EQ(info.value[0], 0);
}

TEST_F(AscendHalStubTest, halSqCqQuery_SqDepth)
{
    struct halSqCqQueryInfo info;
    info.prop = DRV_SQCQ_PROP_SQ_DEPTH;

    drvError_t result = halSqCqQuery(0, &info);
    EXPECT_EQ(result, DRV_ERROR_NONE);
    EXPECT_EQ(info.value[0], 2048);
}

TEST_F(AscendHalStubTest, halSqCqQuery_SqTail)
{
    struct halSqCqQueryInfo info;
    info.prop = DRV_SQCQ_PROP_SQ_TAIL;

    drvError_t result = halSqCqQuery(0, &info);
    EXPECT_EQ(result, DRV_ERROR_NONE);
    EXPECT_EQ(info.value[0], 0);
}

TEST_F(AscendHalStubTest, halSqCqConfig_ReturnsSuccess)
{
    struct halSqCqConfigInfo info;
    info.prop = DRV_SQCQ_PROP_SQ_TAIL;

    drvError_t result = halSqCqConfig(0, &info);
    EXPECT_EQ(result, DRV_ERROR_NONE);
}

TEST_F(AscendHalStubTest, halShrIdInfoGet_ReturnsSuccess)
{
    struct shrIdGetInfo info;

    drvError_t result = halShrIdInfoGet("test", &info);
    EXPECT_EQ(result, DRV_ERROR_NONE);
}

TEST_F(AscendHalStubTest, halTsdrvCtl_ReturnsSuccess)
{
    size_t outSize = 0;

    drvError_t result = halTsdrvCtl(0, 0, nullptr, 0, nullptr, &outSize);
    EXPECT_EQ(result, DRV_ERROR_NONE);
}

TEST_F(AscendHalStubTest, halResourceIdRestore_ReturnsSuccess)
{
    struct drvResIdKey info;

    drvError_t result = halResourceIdRestore(&info);
    EXPECT_EQ(result, DRV_ERROR_NONE);
}

TEST_F(AscendHalStubTest, drvGetLocalDevIDByHostDevID_ReturnsSameId)
{
    uint32_t host_dev_id = 5;
    uint32_t local_dev_id = 0;

    drvError_t result = drvGetLocalDevIDByHostDevID(host_dev_id, &local_dev_id);
    EXPECT_EQ(result, DRV_ERROR_NONE);
    EXPECT_EQ(local_dev_id, host_dev_id);
}

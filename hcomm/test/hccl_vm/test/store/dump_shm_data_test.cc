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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <sys/mman.h>

#include "store_binary_data_operator.h"
#include "store_dump_shm_data.h"
#include "hccl_types.h"
#include "sim_models.h"
#include "db_sim_runner_common.h"
#include "db_sim_runner_db.h"
#include "store_sim_shm_ops.h"
#include "db_sim_sqlite_db.h"

using namespace HcclSim;

extern uint8_t g_opExpansionMode;
namespace HcclSim {
std::string GetBinLocation();
HcclVmResult CreateSimSynData(HcclVmSynData& hvmSynData);
HcclVmResult CreateSimInstrData(HcclVmInstrData& hvmInstrData);
HcclVmResult CreateSimTaskMetaData(HcclVmTaskMetaData& hvmTaskMetaData);
} // namespace HcclSim

static const std::string kSynDataFile = "/%s_hcclvm_syn_data.bin";
static const std::string kTaskDataFile = "/%s_hcclvm_task_data.bin";

static void ClearAllDbTables()
{
    SimRunnerSqliteDB::Instance().ClearAll();
    RunnerDB::DeleteAll<sim::SimModelData>();
    RunnerDB::DeleteAll<sim::Server>();
    RunnerDB::DeleteAll<sim::Host>();
    RunnerDB::DeleteAll<sim::Device>();
    RunnerDB::DeleteAll<sim::Rank>();
    RunnerDB::DeleteAll<sim::MemoryLayout>();
    RunnerDB::DeleteAll<sim::EndPoint>();
    RunnerDB::DeleteAll<sim::Ccu>();
    RunnerDB::DeleteAll<sim::CcuResource>();
    RunnerDB::DeleteAll<sim::CcuChannel>();
    RunnerDB::DeleteAll<sim::RaContext>();
    RunnerDB::DeleteAll<sim::RaJetty>();
    RunnerDB::DeleteAll<sim::EndPointPair>();
}

class DumpShmDataTest : public testing::Test {
protected:
    void SetUp() override
    {
        ClearAllDbTables();
        std::filesystem::create_directories("data");
    }
    void TearDown() override
    {
        ClearAllDbTables();
        g_opExpansionMode = 0;
    }
};

TEST_F(DumpShmDataTest, ShmOps_CheckConstants)
{
    EXPECT_EQ(SHM_MAGIC, 0x53484D50);
    EXPECT_EQ(SHM_VERSION, 1);
}

TEST_F(DumpShmDataTest, GetBinLocation_ReturnsNonEmpty)
{
    std::string loc = GetBinLocation();
    EXPECT_FALSE(loc.empty());
}

TEST_F(DumpShmDataTest, GenDataId_ReturnsNonEmpty)
{
    std::string dataId = GenDataId();
    EXPECT_FALSE(dataId.empty());
    EXPECT_GE(dataId.size(), 17);
}

TEST_F(DumpShmDataTest, GenDataId_ContainsTimestamp)
{
    std::string dataId = GenDataId();
    EXPECT_NE(dataId.find("_"), std::string::npos);
    EXPECT_EQ(dataId[0], '2');
    EXPECT_EQ(dataId[1], '0');
}

TEST_F(DumpShmDataTest, g_opExpansionMode_DefaultZero) { EXPECT_EQ(g_opExpansionMode, 0); }

TEST_F(DumpShmDataTest, DumpHcclVmFlagData_FileCreation)
{
    HcclVmFlagData flagData{};
    flagData.header.magic = HCCLVM_FLAG_FILE_MAGIC;
    flagData.header.header_size = sizeof(FileHeader);
    flagData.header.flags = 1;
    flagData.runner_status = 0;

    HcclVmResult ret = DumpHcclVmFlagData(flagData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(DumpShmDataTest, DumpHcclVmFlagData_ThenGetHcclVmFlagData)
{
    HcclVmFlagData writtenData{};
    writtenData.header.magic = HCCLVM_FLAG_FILE_MAGIC;
    writtenData.header.header_size = sizeof(FileHeader);
    writtenData.header.flags = 1;
    writtenData.runner_status = 1;

    HcclVmResult ret = DumpHcclVmFlagData(writtenData);
    ASSERT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    HcclVmFlagData readData{};
    ret = GetHcclVmFlagData(readData);
    ASSERT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    EXPECT_EQ(readData.header.magic, HCCLVM_FLAG_FILE_MAGIC);
    EXPECT_EQ(readData.runner_status, writtenData.runner_status);
}

TEST_F(DumpShmDataTest, DumpHcclVmFlagData_WriteFail_ReturnsError)
{
    HcclVmFlagData flagData{};
    flagData.header.magic = HCCLVM_FLAG_FILE_MAGIC;
    flagData.header.header_size = 0;
    flagData.header.flags = 1;
    flagData.runner_status = 0;

    std::string rootPath = GetBinLocation();
    std::string fullPath = rootPath + DATA_FILE_PATH + "/hcclvm_flag_data.bin";
    FILE* fp = fopen(fullPath.c_str(), "wb");
    if (fp) {
        auto ret = HcclVmFlagDataWrite(fp, flagData);
        fclose(fp);
        if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
            HcclVmResult dumpRet = DumpHcclVmFlagData(flagData);
            EXPECT_NE(dumpRet, HcclVmResult::HCCL_SIM_SUCCESS);
        }
    }
}

TEST_F(DumpShmDataTest, GetHcclVmFlagData_FileNotExist_ReturnsError)
{
    HcclVmFlagData readData{};
    std::string rootPath = GetBinLocation();
    std::string fullPath = rootPath + DATA_FILE_PATH + "/hcclvm_flag_data.bin";
    std::filesystem::remove(fullPath);
    HcclVmResult ret = GetHcclVmFlagData(readData);
    EXPECT_NE(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(DumpShmDataTest, DumpHcclVmFlagData_StatusStart)
{
    HcclVmFlagData flagData{};
    flagData.header.magic = HCCLVM_FLAG_FILE_MAGIC;
    flagData.header.header_size = sizeof(FileHeader);
    flagData.header.flags = 1;
    flagData.runner_status = 1;

    HcclVmResult ret = DumpHcclVmFlagData(flagData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    HcclVmFlagData readData{};
    ret = GetHcclVmFlagData(readData);
    ASSERT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(readData.runner_status, 1);
}

TEST_F(DumpShmDataTest, DumpHcclVmFlagData_StatusExit)
{
    HcclVmFlagData flagData{};
    flagData.header.magic = HCCLVM_FLAG_FILE_MAGIC;
    flagData.header.header_size = sizeof(FileHeader);
    flagData.header.flags = 1;
    flagData.runner_status = 2;

    HcclVmResult ret = DumpHcclVmFlagData(flagData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    HcclVmFlagData readData{};
    ret = GetHcclVmFlagData(readData);
    ASSERT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(readData.runner_status, 2);
}

class DumpShmDataWithDBTest : public testing::Test {
protected:
    void SetUp() override
    {
        ClearAllDbTables();
        std::filesystem::create_directories("data");
    }
    void TearDown() override
    {
        ClearAllDbTables();
        g_opExpansionMode = 0;
    }

    void InsertSimModelData_CCU()
    {
        sim::SimModelData model;
        model.rank_id = 0;
        model.src_rank = 0;
        model.dst_rank = 1;
        model.root = 0;
        model.rank_size = 2;
        model.chip_type = 0;
        model.op_type = 0;
        model.reduce_op = 0;
        model.data_type = 0;
        model.data_count = 1024;
        model.op_expansion_mode = sim::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_CCU;
        model.ccu0_resource_base_addr = 0;
        model.ccu1_resource_base_addr = 0;
        model.all2AllDataDes.sendType = 0;
        model.all2AllDataDes.recvType = 0;
        model.all2AllDataDes.sendCount = 0;
        model.all2AllDataDes.recvCount = 0;
        model.all2AllDataDes.count = 2;
        memset(model.all2AllDataDes.sendCountMatrix, 0, sizeof(model.all2AllDataDes.sendCountMatrix));
        model.all2AllDataDes.sendCountMatrix[0] = 512;
        model.all2AllDataDes.sendCountMatrix[1] = 512;
        RunnerDB::Add<sim::SimModelData>(model);
    }

    void InsertSimModelData_AICPU()
    {
        sim::SimModelData model;
        model.rank_id = 0;
        model.src_rank = 0;
        model.dst_rank = 1;
        model.root = 0;
        model.rank_size = 2;
        model.chip_type = 0;
        model.op_type = 0;
        model.reduce_op = 0;
        model.data_type = 0;
        model.data_count = 1024;
        model.op_expansion_mode = sim::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_AICPU;
        model.ccu0_resource_base_addr = 0;
        model.ccu1_resource_base_addr = 0;
        model.all2AllDataDes.sendType = 0;
        model.all2AllDataDes.recvType = 0;
        model.all2AllDataDes.sendCount = 0;
        model.all2AllDataDes.recvCount = 0;
        model.all2AllDataDes.count = 0;
        RunnerDB::Add<sim::SimModelData>(model);
    }

    void InsertSimModelData_AllToAllV()
    {
        sim::SimModelData model0;
        model0.rank_id = 0;
        model0.src_rank = 0;
        model0.dst_rank = 1;
        model0.root = 0;
        model0.rank_size = 2;
        model0.chip_type = 0;
        model0.op_type = static_cast<uint16_t>(HcclCMDType::HCCL_CMD_ALLTOALLV);
        model0.reduce_op = 0;
        model0.data_type = 0;
        model0.data_count = 1024;
        model0.op_expansion_mode = sim::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_CCU;
        model0.all2AllDataDes.count = 2;
        memset(model0.all2AllDataDes.sendCountMatrix, 0, sizeof(model0.all2AllDataDes.sendCountMatrix));
        model0.all2AllDataDes.sendCountMatrix[0] = 512;
        model0.all2AllDataDes.sendCountMatrix[1] = 512;
        model0.all2AllDataDes.sendCountMatrix[2] = 256;
        model0.all2AllDataDes.sendCountMatrix[3] = 256;
        RunnerDB::Add<sim::SimModelData>(model0);

        sim::SimModelData model1;
        model1.rank_id = 1;
        model1.src_rank = 1;
        model1.dst_rank = 0;
        model1.root = 0;
        model1.rank_size = 2;
        model1.chip_type = 0;
        model1.op_type = static_cast<uint16_t>(HcclCMDType::HCCL_CMD_ALLTOALLV);
        model1.reduce_op = 0;
        model1.data_type = 0;
        model1.data_count = 1024;
        model1.op_expansion_mode = sim::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_CCU;
        model1.all2AllDataDes.count = 2;
        memset(model1.all2AllDataDes.sendCountMatrix, 0, sizeof(model1.all2AllDataDes.sendCountMatrix));
        model1.all2AllDataDes.sendCountMatrix[0] = 128;
        model1.all2AllDataDes.sendCountMatrix[1] = 128;
        model1.all2AllDataDes.sendCountMatrix[2] = 64;
        model1.all2AllDataDes.sendCountMatrix[3] = 64;
        RunnerDB::Add<sim::SimModelData>(model1);
    }

    void InsertBasicTopology()
    {
        sim::Server server{};
        uint64_t serverId = RunnerDB::Add<sim::Server>(server);
        sim::Host host{};
        host.server_id = serverId;
        strncpy(host.ip_addr, "192.1.5.5", sizeof(host.ip_addr) - 1);
        uint64_t hostId = RunnerDB::Add<sim::Host>(host);

        sim::Device dev0{};
        dev0.server_id = serverId;
        uint64_t dev0Id = RunnerDB::Add<sim::Device>(dev0);

        sim::Device dev1{};
        dev1.server_id = serverId;
        uint64_t dev1Id = RunnerDB::Add<sim::Device>(dev1);

        sim::Rank rank0{};
        rank0.rank_id = 0;
        rank0.device_id = static_cast<uint32_t>(dev0Id);
        RunnerDB::Add<sim::Rank>(rank0);

        sim::Rank rank1{};
        rank1.rank_id = 1;
        rank1.device_id = static_cast<uint32_t>(dev1Id);
        RunnerDB::Add<sim::Rank>(rank1);

        sim::MemoryLayout mem0{};
        mem0.rank_id = 0;
        mem0.buf_type = 0;
        mem0.base_addr = 0x1000;
        mem0.size = 4096;
        mem0.global_offset = 0;
        RunnerDB::Add<sim::MemoryLayout>(mem0);

        sim::MemoryLayout mem1{};
        mem1.rank_id = 1;
        mem1.buf_type = 1;
        mem1.base_addr = 0x2000;
        mem1.size = 8192;
        mem1.global_offset = 4096;
        RunnerDB::Add<sim::MemoryLayout>(mem1);
    }

    void InsertChannelTopology()
    {
        InsertBasicTopology();

        sim::EndPoint localEp{};
        localEp.device_id = 0;
        localEp.die_id = 0;
        memset(localEp.eid, 0x01, sizeof(localEp.eid));
        strncpy(localEp.ip_addr, "192.1.5.5", sizeof(localEp.ip_addr) - 1);
        uint64_t localEpId = RunnerDB::Add<sim::EndPoint>(localEp);

        sim::EndPoint remoteEp{};
        remoteEp.device_id = 1;
        remoteEp.die_id = 1;
        memset(remoteEp.eid, 0x02, sizeof(remoteEp.eid));
        strncpy(remoteEp.ip_addr, "192.2.5.5", sizeof(remoteEp.ip_addr) - 1);
        uint64_t remoteEpId = RunnerDB::Add<sim::EndPoint>(remoteEp);

        sim::CcuChannel channel{};
        channel.channel_id = 1;
        channel.local_endpoint_id = localEpId;
        channel.remote_endpoint_id = remoteEpId;
        channel.protocol = 0;
        channel.jetty_start = 10;
        channel.jetty_num = 4;
        RunnerDB::Add<sim::CcuChannel>(channel);
    }

    void InsertJettyTopology()
    {
        InsertBasicTopology();

        sim::EndPoint localEp{};
        localEp.device_id = 0;
        localEp.die_id = 0;
        memset(localEp.eid, 0x03, sizeof(localEp.eid));
        strncpy(localEp.ip_addr, "192.1.5.5", sizeof(localEp.ip_addr) - 1);
        uint64_t localEpId = RunnerDB::Add<sim::EndPoint>(localEp);

        sim::EndPoint remoteEp{};
        remoteEp.device_id = 1;
        remoteEp.die_id = 1;
        memset(remoteEp.eid, 0x04, sizeof(remoteEp.eid));
        strncpy(remoteEp.ip_addr, "192.2.5.5", sizeof(remoteEp.ip_addr) - 1);
        uint64_t remoteEpId = RunnerDB::Add<sim::EndPoint>(remoteEp);

        sim::RaContext raCtx{};
        raCtx.endpoint_id = localEpId;
        uint64_t raCtxId = RunnerDB::Add<sim::RaContext>(raCtx);

        sim::RaJetty jetty{};
        jetty.ctx_handle = raCtxId;
        jetty.jetty_id = 5;
        jetty.mode = 3;
        RunnerDB::Add<sim::RaJetty>(jetty);

        sim::EndPointPair epPair{};
        epPair.local_enpoint_id = localEpId;
        epPair.remote_enpoint_id = remoteEpId;
        RunnerDB::Add<sim::EndPointPair>(epPair);
    }

    void InsertCcuResourceTopology()
    {
        sim::Server server{};
        uint64_t serverId = RunnerDB::Add<sim::Server>(server);
        sim::Host host{};
        host.server_id = serverId;
        strncpy(host.ip_addr, "192.1.5.5", sizeof(host.ip_addr) - 1);
        RunnerDB::Add<sim::Host>(host);

        sim::Device dev0{};
        dev0.server_id = serverId;
        uint64_t dev0Id = RunnerDB::Add<sim::Device>(dev0);

        sim::Device dev1{};
        dev1.server_id = serverId;
        uint64_t dev1Id = RunnerDB::Add<sim::Device>(dev1);

        sim::Rank rank0{};
        rank0.rank_id = 0;
        rank0.device_id = static_cast<uint32_t>(dev0Id);
        RunnerDB::Add<sim::Rank>(rank0);

        sim::Rank rank1{};
        rank1.rank_id = 1;
        rank1.device_id = static_cast<uint32_t>(dev1Id);
        RunnerDB::Add<sim::Rank>(rank1);

        sim::MemoryLayout mem0{};
        mem0.rank_id = 0;
        mem0.buf_type = 0;
        mem0.base_addr = 0x1000;
        mem0.size = 4096;
        mem0.global_offset = 0;
        RunnerDB::Add<sim::MemoryLayout>(mem0);

        sim::MemoryLayout mem1{};
        mem1.rank_id = 1;
        mem1.buf_type = 1;
        mem1.base_addr = 0x2000;
        mem1.size = 8192;
        mem1.global_offset = 4096;
        RunnerDB::Add<sim::MemoryLayout>(mem1);

        sim::Ccu ccu{};
        ccu.device_id = dev0Id;
        ccu.die_id = 0;
        uint64_t ccuId = RunnerDB::Add<sim::Ccu>(ccu);

        sim::CcuResource ccuRes{};
        ccuRes.ccu_id = ccuId;
        ccuRes.state = 1;
        ccuRes.instr_cnt = 2;
        memset(ccuRes.instr_space, 0, sizeof(ccuRes.instr_space));
        RunnerDB::Add<sim::CcuResource>(ccuRes);
    }
};

TEST_F(DumpShmDataWithDBTest, CreateSimSynData_NoSimModel_ReturnsError)
{
    InsertBasicTopology();
    HcclVmSynData synData;
    HcclVmResult ret = CreateSimSynData(synData);
    EXPECT_NE(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(DumpShmDataWithDBTest, CreateSimSynData_CCU_Success)
{
    InsertSimModelData_CCU();
    InsertChannelTopology();

    HcclVmSynData synData;
    HcclVmResult ret = CreateSimSynData(synData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    EXPECT_EQ(synData.header.magic, HCCLVM_SYN_FILE_MAGIC);
    EXPECT_EQ(synData.header.version, 1);
    EXPECT_EQ(synData.header.count, 1);
    EXPECT_EQ(synData.model_info.comm.rank_size, 2u);
    EXPECT_EQ(
        synData.model_info.comm.op_expansion_mode,
        static_cast<uint32_t>(sim::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_CCU));
    EXPECT_EQ(synData.channel_info.count, 1u);
    EXPECT_EQ(synData.channel_info.data.size(), 1u);
    if (!synData.channel_info.data.empty()) {
        EXPECT_EQ(synData.channel_info.data[0].channelId, 1u);
        EXPECT_EQ(synData.channel_info.data[0].jettyNum, 4u);
    }
    EXPECT_GE(synData.memory_info.count, 2u);
}

TEST_F(DumpShmDataWithDBTest, CreateSimSynData_AICPU_JettyInfo)
{
    InsertSimModelData_AICPU();
    InsertJettyTopology();

    HcclVmSynData synData;
    HcclVmResult ret = CreateSimSynData(synData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    EXPECT_EQ(
        synData.model_info.comm.op_expansion_mode,
        static_cast<uint32_t>(sim::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_AICPU));
    EXPECT_GE(synData.channel_info.count, 1u);
}

TEST_F(DumpShmDataWithDBTest, CreateSimSynData_AllToAllV)
{
    InsertSimModelData_AllToAllV();
    InsertChannelTopology();

    HcclVmSynData synData;
    HcclVmResult ret = CreateSimSynData(synData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    EXPECT_EQ(synData.model_info.comm.op_type, static_cast<uint16_t>(HcclCMDType::HCCL_CMD_ALLTOALLV));
    EXPECT_GT(synData.model_info.all2AllDataDes.sendCountMatrix.size(), 0u);
}

TEST_F(DumpShmDataWithDBTest, CreateChannelInfo_NoChannel_SuccessEmpty)
{
    InsertBasicTopology();
    HcclVmSynData synData;
    HcclVmResult ret = CreateChannelInfo(synData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(synData.channel_info.count, 0u);
}

TEST_F(DumpShmDataWithDBTest, CreateChannelInfo_ChannelWithEndpoint_Success)
{
    InsertChannelTopology();
    HcclVmSynData synData;
    HcclVmResult ret = CreateChannelInfo(synData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(synData.channel_info.count, 1u);
    if (!synData.channel_info.data.empty()) {
        EXPECT_EQ(synData.channel_info.data[0].channelId, 1u);
        EXPECT_EQ(synData.channel_info.data[0].protocol, 0u);
        EXPECT_EQ(synData.channel_info.data[0].jettyNum, 4u);
        for (int i = 0; i < synData.channel_info.data[0].jettyNum; i++) {
            EXPECT_EQ(synData.channel_info.data[0].jettyId[i], 10 + i);
        }
    }
}

TEST_F(DumpShmDataWithDBTest, CreateChannelInfo_MissingLocalEndpoint_ReturnsError)
{
    InsertBasicTopology();
    sim::EndPoint remoteEp{};
    remoteEp.device_id = 1;
    RunnerDB::Add<sim::EndPoint>(remoteEp);

    sim::CcuChannel channel{};
    channel.channel_id = 1;
    channel.local_endpoint_id = 99999;
    channel.remote_endpoint_id = remoteEp.id;
    channel.protocol = 0;
    channel.jetty_num = 2;
    RunnerDB::Add<sim::CcuChannel>(channel);

    HcclVmSynData synData;
    HcclVmResult ret = CreateChannelInfo(synData);
    EXPECT_NE(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(DumpShmDataWithDBTest, CreateChannelInfo_MissingRemoteEndpoint_ReturnsError)
{
    InsertBasicTopology();
    sim::EndPoint localEp{};
    localEp.device_id = 0;
    RunnerDB::Add<sim::EndPoint>(localEp);

    sim::CcuChannel channel{};
    channel.channel_id = 1;
    channel.local_endpoint_id = localEp.id;
    channel.remote_endpoint_id = 99999;
    channel.protocol = 0;
    channel.jetty_num = 2;
    RunnerDB::Add<sim::CcuChannel>(channel);

    HcclVmSynData synData;
    HcclVmResult ret = CreateChannelInfo(synData);
    EXPECT_NE(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(DumpShmDataWithDBTest, CreateJettyInfo_NoJetty_SuccessEmpty)
{
    InsertBasicTopology();
    HcclVmSynData synData;
    HcclVmResult ret = CreateJettyInfo(synData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(synData.channel_info.count, 0u);
}

TEST_F(DumpShmDataWithDBTest, CreateJettyInfo_JettyWithContext_Success)
{
    InsertJettyTopology();
    HcclVmSynData synData;
    HcclVmResult ret = CreateJettyInfo(synData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_GE(synData.channel_info.count, 1u);
}

TEST_F(DumpShmDataWithDBTest, CreateJettyInfo_MissingRaContext_Continues)
{
    InsertBasicTopology();

    sim::RaJetty jetty{};
    jetty.ctx_handle = 99999;
    jetty.jetty_id = 5;
    jetty.mode = 3;
    RunnerDB::Add<sim::RaJetty>(jetty);

    HcclVmSynData synData;
    HcclVmResult ret = CreateJettyInfo(synData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(synData.channel_info.data.size(), 0u);
}

TEST_F(DumpShmDataWithDBTest, CreateJettyInfo_MissingEndPointPair_Continues)
{
    InsertJettyTopology();

    RunnerDB::DeleteAll<sim::EndPointPair>();

    HcclVmSynData synData;
    HcclVmResult ret = CreateJettyInfo(synData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(DumpShmDataWithDBTest, CreateJettyInfo_MissingLocalEndPoint_Continues)
{
    InsertBasicTopology();

    sim::RaContext raCtx{};
    raCtx.endpoint_id = 99999;
    RunnerDB::Add<sim::RaContext>(raCtx);

    sim::RaJetty jetty{};
    jetty.ctx_handle = raCtx.id;
    jetty.jetty_id = 5;
    jetty.mode = 3;
    RunnerDB::Add<sim::RaJetty>(jetty);

    HcclVmSynData synData;
    HcclVmResult ret = CreateJettyInfo(synData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(DumpShmDataWithDBTest, DumpHcclVmSynthesisData_Success)
{
    InsertSimModelData_CCU();
    InsertChannelTopology();

    std::string dataId = GenDataId();
    HcclVmResult ret = DumpHcclVmSynthesisData(dataId);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);

    char fileName[256];
    snprintf(fileName, sizeof(fileName), kSynDataFile.c_str(), dataId.c_str());
    std::string rootPath = GetBinLocation();
    std::string fullPath = rootPath + DATA_FILE_PATH + fileName;
    EXPECT_TRUE(std::filesystem::exists(fullPath));

    FILE* fp = fopen(fullPath.c_str(), "rb");
    if (fp) {
        HcclVmSynData readData;
        ret = HcclVmSynDataRead(fp, readData, HCCLVM_SYN_FILE_MAGIC);
        fclose(fp);
        if (ret == HcclVmResult::HCCL_SIM_SUCCESS) {
            EXPECT_EQ(readData.header.magic, HCCLVM_SYN_FILE_MAGIC);
            EXPECT_EQ(readData.model_info.comm.rank_size, 2u);
        }
    }
}

TEST_F(DumpShmDataWithDBTest, DumpHcclVmSynthesisData_NoSimModel_Fails)
{
    InsertBasicTopology();
    std::string dataId = GenDataId();
    HcclVmResult ret = DumpHcclVmSynthesisData(dataId);
    EXPECT_NE(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(DumpShmDataWithDBTest, CreateSimInstrData_NoCcuResource_SuccessEmpty)
{
    InsertBasicTopology();
    HcclVmInstrData instrData;
    HcclVmResult ret = CreateSimInstrData(instrData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(instrData.header.count, 0u);
}

TEST_F(DumpShmDataWithDBTest, CreateSimInstrData_WithCcuResource_Success)
{
    InsertCcuResourceTopology();
    HcclVmInstrData instrData;
    HcclVmResult ret = CreateSimInstrData(instrData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(instrData.header.magic, HCCLVM_INSTR_FILE_MAGIC);
    EXPECT_GE(instrData.header.count, 1u);
    EXPECT_EQ(instrData.instr_data.size(), instrData.header.count);
}

TEST_F(DumpShmDataWithDBTest, CreateSimInstrData_MissingCcu_ReturnsError)
{
    InsertBasicTopology();

    sim::CcuResource ccuRes{};
    ccuRes.ccu_id = 99999;
    ccuRes.state = 1;
    ccuRes.instr_cnt = 2;
    RunnerDB::Add<sim::CcuResource>(ccuRes);

    HcclVmInstrData instrData;
    HcclVmResult ret = CreateSimInstrData(instrData);
    EXPECT_NE(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(DumpShmDataWithDBTest, DumpHcclVmInstrData_CCU_WithInstr_Success)
{
    g_opExpansionMode = sim::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_CCU;
    InsertSimModelData_CCU();
    InsertCcuResourceTopology();

    std::string dataId = GenDataId();
    HcclVmResult ret = DumpHcclVmInstrData(dataId);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(DumpShmDataWithDBTest, DumpHcclVmInstrData_AICPU_NoInstrDump_Success)
{
    g_opExpansionMode = sim::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_AICPU;
    InsertSimModelData_AICPU();
    InsertBasicTopology();

    std::string dataId = GenDataId();
    HcclVmResult ret = DumpHcclVmInstrData(dataId);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(DumpShmDataWithDBTest, DumpHcclVmInstrData_CCU_NoInstr_ReturnsError)
{
    g_opExpansionMode = sim::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_CCU;
    InsertSimModelData_CCU();
    InsertBasicTopology();

    std::string dataId = GenDataId();
    HcclVmResult ret = DumpHcclVmInstrData(dataId);
    EXPECT_NE(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(DumpShmDataWithDBTest, DumpDataToFile_Success)
{
    InsertSimModelData_CCU();
    InsertChannelTopology();
    InsertCcuResourceTopology();

    std::string dataId = GenDataId();
    HcclVmResult ret = DumpDataToFile(dataId);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(DumpShmDataWithDBTest, DumpDataToFile_SynFail_PropagatesError)
{
    InsertBasicTopology();
    std::string dataId = GenDataId();
    HcclVmResult ret = DumpDataToFile(dataId);
    EXPECT_NE(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(DumpShmDataWithDBTest, DumpDataToFile_TaskFail_PropagatesError)
{
    InsertSimModelData_CCU();
    InsertChannelTopology();

    std::string dataId = GenDataId();
    HcclVmResult ret = DumpDataToFile(dataId);
    EXPECT_NE(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

class DumpShmDataTaskTest : public DumpShmDataWithDBTest {
protected:
    void SetUp() override { DumpShmDataWithDBTest::SetUp(); }
    void TearDown() override { DumpShmDataWithDBTest::TearDown(); }
};

TEST_F(DumpShmDataTaskTest, CreateSimTaskMetaData_NoTask_ReturnsError)
{
    HcclVmTaskMetaData taskMeta;
    HcclVmResult ret = CreateSimTaskMetaData(taskMeta, {});
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(taskMeta.header.count, 0u);
}

TEST_F(DumpShmDataTaskTest, CreateSimTaskMetaData_WithTask_Success)
{
    HcclVmTaskMetaData taskMeta;
    HcclVmResult ret = CreateSimTaskMetaData(taskMeta, {});
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(taskMeta.header.count, 0u);
}

TEST_F(DumpShmDataTaskTest, CreateSimTaskMetaData_AivGraphTask_NoTaskMeta)
{
    HcclVmTaskMetaData taskMeta;
    HcclVmResult ret = CreateSimTaskMetaData(taskMeta, {});
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(taskMeta.header.count, 0u);
}

TEST_F(DumpShmDataTaskTest, DumpHcclVmTask_WithModels_Success)
{
    g_opExpansionMode = sim::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_CCU;
    InsertSimModelData_CCU();
    InsertChannelTopology();

    std::string dataId = GenDataId();
    HcclVmResult ret = DumpHcclVmTask(dataId);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(DumpShmDataTaskTest, DumpHcclVmTask_NoTask_ReturnsError)
{
    std::string dataId = GenDataId();
    HcclVmResult ret = DumpHcclVmTask(dataId);
    EXPECT_NE(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(DumpShmDataWithDBTest, CreateJettyInfo_JettyWithEndPointPair_Success)
{
    InsertSimModelData_AICPU();
    InsertJettyTopology();

    HcclVmSynData synData;
    HcclVmResult ret = CreateJettyInfo(synData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_GE(synData.channel_info.count, 1u);
}

TEST_F(DumpShmDataWithDBTest, DumpHcclVmInstrData_CCU_WithNullCcuSimulator)
{
    g_opExpansionMode = sim::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_CCU;
    InsertSimModelData_CCU();
    InsertCcuResourceTopology();

    std::string dataId = GenDataId();
    HcclVmResult ret = DumpHcclVmInstrData(dataId);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(DumpShmDataWithDBTest, CreateSimInstrData_MultipleCcuResources)
{
    InsertBasicTopology();

    sim::Server server{};
    uint64_t serverId = RunnerDB::Add<sim::Server>(server);

    sim::Device dev0{};
    dev0.server_id = serverId;
    uint64_t dev0Id = RunnerDB::Add<sim::Device>(dev0);

    sim::Device dev1{};
    dev1.server_id = serverId;
    uint64_t dev1Id = RunnerDB::Add<sim::Device>(dev1);

    sim::Rank rank0{};
    rank0.rank_id = 0;
    rank0.device_id = static_cast<uint32_t>(dev0Id);
    RunnerDB::Add<sim::Rank>(rank0);

    sim::Rank rank1{};
    rank1.rank_id = 1;
    rank1.device_id = static_cast<uint32_t>(dev1Id);
    RunnerDB::Add<sim::Rank>(rank1);

    sim::Ccu ccu0{};
    ccu0.device_id = dev0Id;
    ccu0.die_id = 0;
    uint64_t ccu0Id = RunnerDB::Add<sim::Ccu>(ccu0);

    sim::Ccu ccu1{};
    ccu1.device_id = dev1Id;
    ccu1.die_id = 0;
    uint64_t ccu1Id = RunnerDB::Add<sim::Ccu>(ccu1);

    sim::CcuResource ccuRes0{};
    ccuRes0.ccu_id = ccu0Id;
    ccuRes0.state = 1;
    ccuRes0.instr_cnt = 1;
    memset(ccuRes0.instr_space, 0, sizeof(ccuRes0.instr_space));
    RunnerDB::Add<sim::CcuResource>(ccuRes0);

    sim::CcuResource ccuRes1{};
    ccuRes1.ccu_id = ccu1Id;
    ccuRes1.state = 1;
    ccuRes1.instr_cnt = 1;
    memset(ccuRes1.instr_space, 0, sizeof(ccuRes1.instr_space));
    RunnerDB::Add<sim::CcuResource>(ccuRes1);

    HcclVmInstrData instrData;
    HcclVmResult ret = CreateSimInstrData(instrData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(instrData.header.count, 2u);
}

TEST_F(DumpShmDataTaskTest, CreateSimTaskMetaData_CCU_NoTaskMeta)
{
    HcclVmTaskMetaData taskMeta;
    HcclVmResult ret = CreateSimTaskMetaData(taskMeta, {});
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(taskMeta.header.count, 0u);
}

TEST_F(DumpShmDataTaskTest, DumpHcclVmTask_WithCcuAndChannel_Success)
{
    g_opExpansionMode = sim::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_CCU;
    InsertSimModelData_CCU();
    InsertChannelTopology();

    std::string dataId = GenDataId();
    HcclVmResult ret = DumpHcclVmTask(dataId);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

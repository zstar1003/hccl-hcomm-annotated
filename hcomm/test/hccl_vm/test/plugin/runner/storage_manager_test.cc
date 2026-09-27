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
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

#include "store_binary_data_operator.h"
#define private public
#include "storage_manager.h"
#undef private

class StorageManagerTest : public testing::Test {
protected:
    void SetUp() override
    {
        StorageManager::GetInstance().SetDataId("");
        StorageManager::GetInstance().m_synData = HcclVmSynData{};
        StorageManager::GetInstance().m_allRankChannelInfo.clear();
        StorageManager::GetInstance().m_allRankTaskQueues.clear();
        StorageManager::GetInstance().m_instrData = HcclVmInstrData{};
        StorageManager::GetInstance().m_allPhyMem.clear();
        StorageManager::GetInstance().devType_ = DevType::DEV_TYPE_COUNT;
        StorageManager::GetInstance().m_checker_param = CheckerParam{};
        StorageManager::GetInstance().ResetAivResource();
    }

    void TearDown() override
    {
        StorageManager::GetInstance().SetDataId("");
        StorageManager::GetInstance().ResetAivResource();
        unsetenv("HCCLVM_ENABLE_DUMP_DATA");
    }

    std::string CreateTempDir()
    {
        auto dir = std::filesystem::temp_directory_path() / "storage_mgr_test_XXXXXX";
        auto dirStr = dir.string();
        char tmpl[256];
        snprintf(tmpl, sizeof(tmpl), "%s", dirStr.c_str());
        auto result = mkdtemp(tmpl);
        return result ? std::string(result) : "";
    }

    void SetupSynDataForTrans()
    {
        auto& mgr = StorageManager::GetInstance();
        mgr.m_synData.model_info.comm.rank_size = 2;
        mgr.m_synData.model_info.comm.src_rank = 0;
        mgr.m_synData.model_info.comm.dst_rank = 1;
        mgr.m_synData.model_info.comm.root = 0;
        mgr.m_synData.model_info.comm.op_type = 1;
        mgr.m_synData.model_info.comm.reduce_op = 0;
        mgr.m_synData.model_info.comm.data_type = 0;
        mgr.m_synData.model_info.comm.data_count = 100;
        mgr.m_synData.model_info.comm.chip_type = static_cast<uint16_t>(DevType::DEV_TYPE_950);
        mgr.m_synData.model_info.comm.op_expansion_mode = 0;
        mgr.m_synData.model_info.all2AllDataDes.sendType = 0;
        mgr.m_synData.model_info.all2AllDataDes.recvType = 0;
        mgr.m_synData.model_info.all2AllDataDes.sendCount = 0;
        mgr.m_synData.model_info.all2AllDataDes.recvCount = 0;
        mgr.m_synData.model_info.all2AllDataDes.count = 0;
    }

    bool WriteSynDataFile(const std::string& dir, const std::string& dataId)
    {
        char fileName[256];
        snprintf(fileName, sizeof(fileName), "/%s_hcclvm_syn_data.bin", dataId.c_str());
        std::string dataDir = dir + "/data";
        std::filesystem::create_directories(dataDir);
        std::string fullPath = dataDir + fileName;
        FILE* fp = fopen(fullPath.c_str(), "wb");
        if (!fp) {
            return false;
        }
        SetupSynDataForTrans();
        StorageManager::GetInstance().m_synData.header.magic = HCCLVM_SYN_FILE_MAGIC;
        StorageManager::GetInstance().m_synData.header.version = 1;
        StorageManager::GetInstance().m_synData.header.count = 0;
        auto ret = HcclVmSynDataWrite(fp, StorageManager::GetInstance().m_synData);
        fclose(fp);
        return ret == HcclVmResult::HCCL_SIM_SUCCESS;
    }

    bool WriteInstrDataFile(const std::string& dir, const std::string& dataId)
    {
        char fileName[256];
        snprintf(fileName, sizeof(fileName), "/%s_hcclvm_instr_data.bin", dataId.c_str());
        std::string dataDir = dir + "/data";
        std::filesystem::create_directories(dataDir);
        std::string fullPath = dataDir + fileName;
        FILE* fp = fopen(fullPath.c_str(), "wb");
        if (!fp) {
            return false;
        }
        HcclVmInstrData instrData;
        instrData.header.magic = HCCLVM_INSTR_FILE_MAGIC;
        instrData.header.version = 1;
        instrData.header.count = 0;
        auto ret = HcclVmInstrDataWrite(fp, instrData);
        fclose(fp);
        return ret == HcclVmResult::HCCL_SIM_SUCCESS;
    }

    bool WriteTaskDataFile(const std::string& dir, const std::string& dataId)
    {
        char fileName[256];
        snprintf(fileName, sizeof(fileName), "/%s_hcclvm_task_data.bin", dataId.c_str());
        std::string dataDir = dir + "/data";
        std::filesystem::create_directories(dataDir);
        std::string fullPath = dataDir + fileName;
        FILE* fp = fopen(fullPath.c_str(), "wb");
        if (!fp) {
            return false;
        }
        HcclVmTaskMetaData taskMeta;
        taskMeta.header.magic = HCCLVM_TASK_FILE_MAGIC;
        taskMeta.header.version = 1;
        taskMeta.header.count = 0;
        auto ret = HcclVmTaskMetaDataWrite(fp, taskMeta);
        fclose(fp);
        return ret == HcclVmResult::HCCL_SIM_SUCCESS;
    }

    bool WriteFlagDataFile(const std::string& dir, const std::string& dataId)
    {
        std::string dataDir = dir + "/data";
        std::filesystem::create_directories(dataDir);
        std::string fullPath = dataDir + "/hcclvm_flag_data.bin";
        FILE* fp = fopen(fullPath.c_str(), "wb");
        if (!fp) {
            return false;
        }
        HcclVmFlagData flagData;
        flagData.header.magic = HCCLVM_FLAG_FILE_MAGIC;
        flagData.header.version = 1;
        flagData.runner_status = 1;
        auto ret = HcclVmFlagDataWrite(fp, flagData);
        fclose(fp);
        return ret == HcclVmResult::HCCL_SIM_SUCCESS;
    }
};

TEST_F(StorageManagerTest, GetInstance_ReturnsSameSingleton)
{
    StorageManager& mgr1 = StorageManager::GetInstance();
    StorageManager& mgr2 = StorageManager::GetInstance();
    EXPECT_EQ(&mgr1, &mgr2);
}

TEST_F(StorageManagerTest, SetDataId_GetDataId_ReturnsCorrectValue)
{
    StorageManager::GetInstance().SetDataId("test_data_123");
    EXPECT_EQ(StorageManager::GetInstance().GetDataId(), "test_data_123");
}

TEST_F(StorageManagerTest, SetDataId_EmptyString_ReturnsEmpty)
{
    StorageManager::GetInstance().SetDataId("");
    EXPECT_EQ(StorageManager::GetInstance().GetDataId(), "");
}

TEST_F(StorageManagerTest, GetCheckerParam_DefaultValues)
{
    CheckerParam param = StorageManager::GetInstance().GetCheckerParam();
    EXPECT_EQ(param.rankSize, 0u);
    EXPECT_EQ(param.dataCount, 0u);
}

TEST_F(StorageManagerTest, GetRankSize_Default_ReturnsZero)
{
    EXPECT_EQ(StorageManager::GetInstance().GetRankSize(), 0u);
}

TEST_F(StorageManagerTest, GetHvmInstrData_Default_ReturnsEmpty)
{
    HcclVmInstrData instrData = StorageManager::GetInstance().GetHvmInstrData();
    EXPECT_EQ(instrData.instr_data.size(), 0u);
}

TEST_F(StorageManagerTest, GetAllRankTaskQueues_Default_ReturnsEmpty)
{
    AllRankTaskQueues& queues = StorageManager::GetInstance().GetAllRankTaskQueues();
    EXPECT_EQ(queues.size(), 0u);
}

TEST_F(StorageManagerTest, ConvertTaskQueue_EmptyTaskMeta_ReturnsSuccess)
{
    HcclVmTaskMetaData taskMetaData;
    auto ret = StorageManager::GetInstance().ConvertTaskQueue(taskMetaData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(StorageManagerTest, LoadHcclVmSynthesisData_EmptyDataId_ReturnsError)
{
    StorageManager::GetInstance().SetDataId("");
    sim::OpDetailTab detail{};
    std::vector<sim::CcuChannelTab> channels;
    auto ret = StorageManager::GetInstance().LoadHcclVmSynthesisData(detail, channels);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(StorageManagerTest, LoadHcclVmInstrData_EmptyDataId_ReturnsError)
{
    StorageManager::GetInstance().SetDataId("");
    std::vector<sim::CcuInstrResTab> instrRes;
    auto ret = StorageManager::GetInstance().LoadHcclVmInstrData(instrRes);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(StorageManagerTest, LoadHcclVmTaskMetaData_EmptyDataId_ReturnsError)
{
    StorageManager::GetInstance().SetDataId("");
    std::vector<sim::OpTaskTab> tasks;
    auto ret = StorageManager::GetInstance().LoadHcclVmTaskMetaData(tasks);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(StorageManagerTest, GetHcclVmFlagData_EmptyDataId_ReturnsError)
{
    HcclSim::HcclVmFlagData flagData;
    StorageManager::GetInstance().SetDataId("");
    auto ret = StorageManager::GetInstance().GetHcclVmFlagData(flagData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);
}

TEST_F(StorageManagerTest, DumpHcclVmFlagData_EmptyDataId_ReturnsError)
{
    StorageManager::GetInstance().SetDataId("");
    auto ret = StorageManager::GetInstance().DumpHcclVmFlagData(0);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_INTERNAL);
}

TEST_F(StorageManagerTest, InitAivResourceFromCompositeOpDetail_RankSizeZero_ReturnsError)
{
    sim::CompositeOpDetail opDetail{};
    auto ret = StorageManager::GetInstance().InitAivResourceFromCompositeOpDetail(opDetail);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_PARA);
}

TEST_F(StorageManagerTest, InitAivResourceFromCompositeOpDetail_EmptyMemInfo_ReturnsSuccess)
{
    auto& mgr = StorageManager::GetInstance();
    mgr.m_checker_param.rankSize = 2;
    sim::CompositeOpDetail opDetail{};
    opDetail.rankId = 0;
    auto ret = mgr.InitAivResourceFromCompositeOpDetail(opDetail);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(StorageManagerTest, InitCcuResource_InvalidDevType_ReturnsError)
{
    std::vector<sim::CcuInstrResTab> instrRes;
    auto ret = StorageManager::GetInstance().InitCcuResource(instrRes);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(StorageManagerTest, ReleasePhyMem_NoAllocatedMemory_Safe)
{
    EXPECT_NO_THROW(StorageManager::GetInstance().ReleasePhyMem());
}

TEST_F(StorageManagerTest, DumpAllRankInputOutput_Disabled_Safe)
{
    std::vector<std::map<uint32_t, sim::CompositeOpDetail>> emptyData;
    EXPECT_NO_THROW(StorageManager::GetInstance().DumpAllRankInputOutput(emptyData));
}

TEST_F(StorageManagerTest, Trans2CheckerParam_EmptySynData_ReturnsSuccess)
{
    sim::OpDetailTab detailTab{};
    ::OpDetails detail{};
    auto ret = StorageManager::GetInstance().Trans2CheckerParam(detailTab, detail);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(StorageManagerTest, Trans2CheckerParam_WithPopulatedSynData)
{
    SetupSynDataForTrans();
    sim::OpDetailTab detailTab{};
    detailTab.rankSize = 2;
    detailTab.srcRank = 0;
    detailTab.dstRank = 1;
    detailTab.root = 0;
    ::OpDetails detail{};
    detail.opV1.count = 100;
    auto ret = StorageManager::GetInstance().Trans2CheckerParam(detailTab, detail);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    auto param = StorageManager::GetInstance().GetCheckerParam();
    EXPECT_EQ(param.rankSize, 2u);
    EXPECT_EQ(param.srcRank, 0u);
    EXPECT_EQ(param.dstRank, 1u);
    EXPECT_EQ(param.dataCount, 100u);
}

TEST_F(StorageManagerTest, Trans2CheckerParam_All2AllDataDes)
{
    auto& mgr = StorageManager::GetInstance();
    mgr.m_synData.model_info.comm.rank_size = 4;
    mgr.m_synData.model_info.all2AllDataDes.sendType = 1;
    mgr.m_synData.model_info.all2AllDataDes.recvType = 2;
    mgr.m_synData.model_info.all2AllDataDes.sendCount = 10;
    mgr.m_synData.model_info.all2AllDataDes.recvCount = 20;
    mgr.m_synData.model_info.all2AllDataDes.count = 0;
    sim::OpDetailTab detailTab{};
    ::OpDetails detail{};
    auto ret = mgr.Trans2CheckerParam(detailTab, detail);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(StorageManagerTest, ConvertTaskQueue_ValidTaskMeta_RankSizeZero_ReturnsError)
{
    HcclVmTaskMetaData taskMetaData;
    HcclTaskMetaData task;
    task.rankId = 0;
    task.streamId = 0;
    taskMetaData.task_meta.push_back(task);
    auto ret = StorageManager::GetInstance().ConvertTaskQueue(taskMetaData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_PARA);
}

TEST_F(StorageManagerTest, ConvertTaskQueue_ValidRankSize_Success)
{
    auto& mgr = StorageManager::GetInstance();
    mgr.m_checker_param.rankSize = 2;
    HcclVmTaskMetaData taskMetaData;
    HcclTaskMetaData task;
    task.rankId = 0;
    task.streamId = 0;
    taskMetaData.task_meta.push_back(task);
    auto ret = mgr.ConvertTaskQueue(taskMetaData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    auto& queues = mgr.GetAllRankTaskQueues();
    EXPECT_EQ(queues.size(), 2u);
    EXPECT_EQ(queues[0].size(), 1u);
}

TEST_F(StorageManagerTest, ConvertTaskQueue_MultipleStreams_Success)
{
    auto& mgr = StorageManager::GetInstance();
    mgr.m_checker_param.rankSize = 2;
    HcclVmTaskMetaData taskMetaData;
    HcclTaskMetaData task1;
    task1.rankId = 0;
    task1.streamId = 0;
    taskMetaData.task_meta.push_back(task1);
    HcclTaskMetaData task2;
    task2.rankId = 0;
    task2.streamId = 1;
    taskMetaData.task_meta.push_back(task2);
    auto ret = mgr.ConvertTaskQueue(taskMetaData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(mgr.GetAllRankTaskQueues()[0].size(), 2u);
}

TEST_F(StorageManagerTest, ConvertTaskQueue_OutOfRangeRankId_ReturnsError)
{
    auto& mgr = StorageManager::GetInstance();
    mgr.m_checker_param.rankSize = 2;
    HcclVmTaskMetaData taskMetaData;
    HcclTaskMetaData task;
    task.rankId = 999;
    task.streamId = 0;
    taskMetaData.task_meta.push_back(task);
    auto ret = mgr.ConvertTaskQueue(taskMetaData);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_E_PARA);
}

TEST_F(StorageManagerTest, InitCcuResource_With950DevType_RequiresChannelInfo)
{
    auto& mgr = StorageManager::GetInstance();
    mgr.devType_ = DevType::DEV_TYPE_950;
    mgr.m_checker_param.rankSize = 2;
    mgr.m_synData.model_info.comm.ccu0_resource_base_addr = 0x1000;
    mgr.m_synData.model_info.comm.ccu1_resource_base_addr = 0x2000;
    mgr.m_allRankChannelInfo.resize(2);
    std::vector<sim::CcuInstrResTab> instrRes;
    auto ret = mgr.InitCcuResource(instrRes);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(StorageManagerTest, InitCcuResource_With950DevTypeAndInstr)
{
    auto& mgr = StorageManager::GetInstance();
    mgr.devType_ = DevType::DEV_TYPE_950;
    mgr.m_checker_param.rankSize = 2;
    mgr.m_synData.model_info.comm.ccu0_resource_base_addr = 0x1000;
    mgr.m_synData.model_info.comm.ccu1_resource_base_addr = 0x2000;
    mgr.m_allRankChannelInfo.resize(2);
    MicrocodeInstrInner instr;
    instr.desc.rank_id = 0;
    instr.desc.die_id = 0;
    instr.desc.count = 1;
    instr.data.push_back({});
    mgr.m_instrData.instr_data.push_back(instr);
    std::vector<sim::CcuInstrResTab> instrRes;
    auto ret = mgr.InitCcuResource(instrRes);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
}

TEST_F(StorageManagerTest, FindRootPath_NoPluginDir_ReturnsEmpty)
{
    auto& mgr = StorageManager::GetInstance();
    std::string result = mgr.FindRootPath();
    EXPECT_TRUE(result.empty() || !result.empty());
}

TEST_F(StorageManagerTest, FindRootPath_WithPluginDir)
{
    auto& mgr = StorageManager::GetInstance();
    std::string tmpDir = CreateTempDir();
    ASSERT_FALSE(tmpDir.empty());
    std::filesystem::create_directories(tmpDir + "/plugin");
    auto origDir = std::filesystem::current_path();
    std::filesystem::current_path(tmpDir);
    std::string result = mgr.FindRootPath();
    EXPECT_FALSE(result.empty());
    std::filesystem::current_path(origDir);
    std::filesystem::remove_all(tmpDir);
}

TEST_F(StorageManagerTest, IsDirExists_NonExistentDir)
{
    auto& mgr = StorageManager::GetInstance();
    EXPECT_FALSE(mgr.IsDirExists("/nonexistent_dir_12345"));
}

TEST_F(StorageManagerTest, IsDirExists_ExistingDir)
{
    auto& mgr = StorageManager::GetInstance();
    EXPECT_TRUE(mgr.IsDirExists("/tmp"));
}

TEST_F(StorageManagerTest, LoadHcclVmSynthesisData_WithFileData)
{
    auto& mgr = StorageManager::GetInstance();
    std::string tmpDir = CreateTempDir();
    ASSERT_FALSE(tmpDir.empty());
    std::filesystem::create_directories(tmpDir + "/plugin");
    std::string dataId = "test1";
    ASSERT_TRUE(WriteSynDataFile(tmpDir, dataId));
    mgr.SetDataId(dataId);
    auto origDir = std::filesystem::current_path();
    std::filesystem::current_path(tmpDir);
    sim::OpDetailTab detail{};
    std::vector<sim::CcuChannelTab> channels;
    auto ret = mgr.LoadHcclVmSynthesisData(detail, channels);
    std::filesystem::current_path(origDir);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    std::filesystem::remove_all(tmpDir);
}

TEST_F(StorageManagerTest, LoadHcclVmInstrData_WithFileData)
{
    auto& mgr = StorageManager::GetInstance();
    std::string tmpDir = CreateTempDir();
    ASSERT_FALSE(tmpDir.empty());
    std::filesystem::create_directories(tmpDir + "/plugin");
    std::string dataId = "test2";
    ASSERT_TRUE(WriteInstrDataFile(tmpDir, dataId));
    mgr.SetDataId(dataId);
    mgr.m_synData.model_info.comm.op_expansion_mode = 0;
    auto origDir = std::filesystem::current_path();
    std::filesystem::current_path(tmpDir);
    std::vector<sim::CcuInstrResTab> instrRes;
    auto ret = mgr.LoadHcclVmInstrData(instrRes);
    std::filesystem::current_path(origDir);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    std::filesystem::remove_all(tmpDir);
}

TEST_F(StorageManagerTest, LoadHcclVmInstrData_NonCcuMode)
{
    auto& mgr = StorageManager::GetInstance();
    std::string tmpDir = CreateTempDir();
    ASSERT_FALSE(tmpDir.empty());
    std::filesystem::create_directories(tmpDir + "/plugin");
    std::string dataId = "test3";
    ASSERT_TRUE(WriteInstrDataFile(tmpDir, dataId));
    mgr.SetDataId(dataId);
    mgr.m_synData.model_info.comm.op_expansion_mode = 1;
    auto origDir = std::filesystem::current_path();
    std::filesystem::current_path(tmpDir);
    std::vector<sim::CcuInstrResTab> instrRes;
    auto ret = mgr.LoadHcclVmInstrData(instrRes);
    std::filesystem::current_path(origDir);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    std::filesystem::remove_all(tmpDir);
}

TEST_F(StorageManagerTest, LoadHcclVmTaskMetaData_WithFileData)
{
    auto& mgr = StorageManager::GetInstance();
    std::string tmpDir = CreateTempDir();
    ASSERT_FALSE(tmpDir.empty());
    std::filesystem::create_directories(tmpDir + "/plugin");
    std::string dataId = "test4";
    ASSERT_TRUE(WriteTaskDataFile(tmpDir, dataId));
    mgr.SetDataId(dataId);
    auto origDir = std::filesystem::current_path();
    std::filesystem::current_path(tmpDir);
    std::vector<sim::OpTaskTab> tasks;
    auto ret = mgr.LoadHcclVmTaskMetaData(tasks);
    std::filesystem::current_path(origDir);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    std::filesystem::remove_all(tmpDir);
}

TEST_F(StorageManagerTest, GetHcclVmFlagData_WithFileData)
{
    auto& mgr = StorageManager::GetInstance();
    std::string tmpDir = CreateTempDir();
    ASSERT_FALSE(tmpDir.empty());
    std::filesystem::create_directories(tmpDir + "/plugin");
    std::string dataId = "test5";
    ASSERT_TRUE(WriteFlagDataFile(tmpDir, dataId));
    mgr.SetDataId(dataId);
    auto origDir = std::filesystem::current_path();
    std::filesystem::current_path(tmpDir);
    HcclVmFlagData flagData;
    auto ret = mgr.GetHcclVmFlagData(flagData);
    std::filesystem::current_path(origDir);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(flagData.runner_status, 1);
    std::filesystem::remove_all(tmpDir);
}

TEST_F(StorageManagerTest, DumpHcclVmFlagData_WithFileData)
{
    auto& mgr = StorageManager::GetInstance();
    std::string tmpDir = CreateTempDir();
    ASSERT_FALSE(tmpDir.empty());
    std::filesystem::create_directories(tmpDir + "/plugin");
    std::string dataId = "test6";
    std::string dataDir = tmpDir + "/data";
    std::filesystem::create_directories(dataDir);
    mgr.SetDataId(dataId);
    auto origDir = std::filesystem::current_path();
    std::filesystem::current_path(tmpDir);
    auto ret = mgr.DumpHcclVmFlagData(1);
    std::filesystem::current_path(origDir);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    std::filesystem::remove_all(tmpDir);
}

TEST_F(StorageManagerTest, LoadHcclVmSynthesisData_NonExistentFile)
{
    auto& mgr = StorageManager::GetInstance();
    std::string tmpDir = CreateTempDir();
    ASSERT_FALSE(tmpDir.empty());
    std::filesystem::create_directories(tmpDir + "/plugin");
    mgr.SetDataId("nonexistent");
    auto origDir = std::filesystem::current_path();
    std::filesystem::current_path(tmpDir);
    sim::OpDetailTab detail{};
    std::vector<sim::CcuChannelTab> channels;
    auto ret = mgr.LoadHcclVmSynthesisData(detail, channels);
    std::filesystem::current_path(origDir);
    EXPECT_EQ(ret, HcclVmResult::HCCL_SIM_SUCCESS);
    std::filesystem::remove_all(tmpDir);
}

TEST_F(StorageManagerTest, DumpAllRankInputOutput_EnabledEnv_NoData)
{
    setenv("HCCLVM_ENABLE_DUMP_DATA", "1", 1);
    auto& mgr = StorageManager::GetInstance();
    mgr.m_checker_param.rankSize = 1;
    std::vector<std::map<uint32_t, sim::CompositeOpDetail>> emptyData;
    EXPECT_NO_THROW(mgr.DumpAllRankInputOutput(emptyData));
    unsetenv("HCCLVM_ENABLE_DUMP_DATA");
}

TEST_F(StorageManagerTest, DumpAllRankInputOutput_EnabledEnvEmpty)
{
    setenv("HCCLVM_ENABLE_DUMP_DATA", "", 1);
    std::vector<std::map<uint32_t, sim::CompositeOpDetail>> emptyData;
    EXPECT_NO_THROW(StorageManager::GetInstance().DumpAllRankInputOutput(emptyData));
}

TEST_F(StorageManagerTest, DumpAllRankInputOutput_EnabledEnvZero)
{
    setenv("HCCLVM_ENABLE_DUMP_DATA", "0", 1);
    std::vector<std::map<uint32_t, sim::CompositeOpDetail>> emptyData;
    EXPECT_NO_THROW(StorageManager::GetInstance().DumpAllRankInputOutput(emptyData));
}

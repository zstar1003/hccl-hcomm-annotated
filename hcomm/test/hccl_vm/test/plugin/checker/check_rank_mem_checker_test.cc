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
#include <gtest/gtest.h>
#include <map>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "binary_data_operator.h"
#include "check_rank_mem.h"
#include "mem_access_info.h"
#include "mem_access_record.h"
#include "sim_common.h"
#include "storage_manager.h"

using namespace HcclSim;

std::map<RankId, std::map<u32, ChannelsPerDie>> g_allRankChannelInfo;

class CheckRankMemTest : public testing::Test {
protected:
    void SetUp() override
    {
        StorageManager::GetInstance().Reset();
        StorageManager::GetInstance().SetDataId("");
    }

    void TearDown() override
    {
        StorageManager::GetInstance().Reset();
        StorageManager::GetInstance().SetDataId("");
    }

    std::string CreateTempDir()
    {
        char tmpl[] = "/tmp/crm_test_XXXXXX";
        std::string dir = mkdtemp(tmpl);
        return dir;
    }

    void RemoveDir(const std::string& dir)
    {
        std::string cmd = "rm -rf " + dir;
        system(cmd.c_str());
    }

    std::string CreateTestDataDir(const std::string& baseDir, const std::string& dataId)
    {
        std::string pluginDir = baseDir + "/plugin";
        std::string dataDir = baseDir + "/data";
        mkdir(pluginDir.c_str(), 0755);
        mkdir(dataDir.c_str(), 0755);
        return dataDir;
    }

    void WriteSynDataFile(const std::string& path, const HcclVmSynData& synData)
    {
        FILE* fp = fopen(path.c_str(), "wb");
        ASSERT_NE(fp, nullptr);
        auto ret = HcclVmSynDataWrite(fp, synData);
        fclose(fp);
        ASSERT_EQ(ret, HcclResult::HCCL_SUCCESS);
    }

    void WriteGzJsonlFile(const std::string& path, const std::vector<std::string>& lines)
    {
        gzFile file = gzopen(path.c_str(), "wb");
        ASSERT_NE(file, nullptr);
        for (const auto& line : lines) {
            std::string l = line + "\n";
            gzwrite(file, l.c_str(), l.size());
        }
        gzclose(file);
    }

    void SetupStorageManagerWithMemLayout(
        const std::string& tmpDir, const std::string& dataId, const std::vector<std::string>& memLayoutLines)
    {
        std::string dataDir = CreateTestDataDir(tmpDir, dataId);

        HcclVmSynData synData{};
        synData.header.magic = HCCLVM_SYN_FILE_MAGIC;
        synData.header.version = 1;
        synData.header.header_size = sizeof(FileHeader);
        synData.model_info.comm.rank_size = 2;
        synData.model_info.comm.op_expansion_mode = 0;

        std::string synFilePath = dataDir + "/" + dataId + "_hcclvm_syn_data.bin";
        WriteSynDataFile(synFilePath, synData);

        std::string memFilePath = dataDir + "/" + dataId + "_mem_layout.jsonl.gz";
        WriteGzJsonlFile(memFilePath, memLayoutLines);

        char savedCwd[4096];
        getcwd(savedCwd, sizeof(savedCwd));
        chdir(tmpDir.c_str());

        StorageManager::GetInstance().SetDataId(dataId);
        StorageManager::GetInstance().LoadHcclVmSynthesisData();
        StorageManager::GetInstance().Trans2CheckerParam();

        chdir(savedCwd);
    }

    void
    SetupStorageManagerWithSynData(const std::string& tmpDir, const std::string& dataId, const HcclVmSynData& synData)
    {
        std::string dataDir = CreateTestDataDir(tmpDir, dataId);
        std::string synFilePath = dataDir + "/" + dataId + "_hcclvm_syn_data.bin";
        WriteSynDataFile(synFilePath, synData);

        char savedCwd[4096];
        getcwd(savedCwd, sizeof(savedCwd));
        chdir(tmpDir.c_str());

        StorageManager::GetInstance().SetDataId(dataId);
        StorageManager::GetInstance().LoadHcclVmSynthesisData();
        StorageManager::GetInstance().Trans2CheckerParam();

        chdir(savedCwd);
    }

    HcclTaskMetaData MakeMemCpyTask(
        uint32_t srcRank, uint64_t srcOffset, uint32_t dstRank, uint64_t dstOffset, uint64_t len, uint32_t commId = 0,
        uint32_t rankId = 0, uint32_t streamId = 0)
    {
        HcclTaskMetaData task{};
        task.taskType = HccLTaskMetaType::MEM_CPY;
        task.commId = commId;
        task.rankId = rankId;
        task.streamId = streamId;
        task.taskData.transMem.srcRankId = srcRank;
        task.taskData.transMem.srcOffset = srcOffset;
        task.taskData.transMem.dstRankId = dstRank;
        task.taskData.transMem.dstOffset = dstOffset;
        task.taskData.transMem.len = len;
        task.taskData.transMem.protocol = 0;
        return task;
    }

    HcclTaskMetaData MakeReduceTask(
        uint32_t srcRank, uint64_t srcOffset, uint32_t dstRank, uint64_t dstOffset, uint64_t dataCount,
        uint32_t commId = 0, uint32_t rankId = 0, uint32_t streamId = 0)
    {
        HcclTaskMetaData task{};
        task.taskType = HccLTaskMetaType::REDUCE;
        task.commId = commId;
        task.rankId = rankId;
        task.streamId = streamId;
        task.taskData.reduce.srcRankId = srcRank;
        task.taskData.reduce.srcOffset = srcOffset;
        task.taskData.reduce.dstRankId = dstRank;
        task.taskData.reduce.dstOffset = dstOffset;
        task.taskData.reduce.dataCount = dataCount;
        task.taskData.reduce.dataType = 0;
        task.taskData.reduce.reduceOp = 0;
        task.taskData.reduce.protocol = 0;
        return task;
    }

    HcclTaskMetaData MakeNotifyRecordTask(
        uint32_t srcRank, uint32_t notifyId, uint32_t dstRank, uint32_t notifyCount = 1, uint32_t commId = 0,
        uint32_t rankId = 0, uint32_t streamId = 0)
    {
        HcclTaskMetaData task{};
        task.taskType = HccLTaskMetaType::NOTIFY_RECORD;
        task.commId = commId;
        task.rankId = rankId;
        task.streamId = streamId;
        task.taskData.notify.srcRankId = srcRank;
        task.taskData.notify.notifyId = notifyId;
        task.taskData.notify.dstRankId = dstRank;
        task.taskData.notify.notifyCount = notifyCount;
        task.taskData.notify.protocol = 0;
        return task;
    }

    HcclTaskMetaData MakeNotifyWaitTask(
        uint32_t srcRank, uint32_t notifyId, uint32_t dstRank, uint32_t notifyCount = 1, uint32_t commId = 0,
        uint32_t rankId = 0, uint32_t streamId = 0)
    {
        HcclTaskMetaData task{};
        task.taskType = HccLTaskMetaType::NOTIFY_WAIT;
        task.commId = commId;
        task.rankId = rankId;
        task.streamId = streamId;
        task.taskData.notify.srcRankId = srcRank;
        task.taskData.notify.notifyId = notifyId;
        task.taskData.notify.dstRankId = dstRank;
        task.taskData.notify.notifyCount = notifyCount;
        task.taskData.notify.protocol = 0;
        return task;
    }

    HcclTaskMetaData MakeCcuGraphTask(uint32_t commId = 0, uint32_t rankId = 0, uint32_t streamId = 0)
    {
        HcclTaskMetaData task{};
        task.taskType = HccLTaskMetaType::CCU_GRAPH;
        task.commId = commId;
        task.rankId = rankId;
        task.streamId = streamId;
        return task;
    }

    HcclTaskMetaData MakeAivGraphTask(uint32_t commId = 0, uint32_t rankId = 0, uint32_t streamId = 0)
    {
        HcclTaskMetaData task{};
        task.taskType = HccLTaskMetaType::AIV_GRAPH;
        task.commId = commId;
        task.rankId = rankId;
        task.streamId = streamId;
        return task;
    }
};

TEST_F(CheckRankMemTest, DefaultConstruction)
{
    CheckRankMem checker(0);
    EXPECT_EQ(checker.GetRankId(), 0);
}

TEST_F(CheckRankMemTest, ConstructionWithRankId)
{
    CheckRankMem checker(3);
    EXPECT_EQ(checker.GetRankId(), 3);
}

TEST_F(CheckRankMemTest, CheckEmptyTaskList)
{
    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());
}

TEST_F(CheckRankMemTest, CheckSingleMemCpyTask)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_single_mcpy";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 4096, 0]");
    memLayoutLines.push_back("[1, 1024, 4096, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 1024));

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckTwoMemCpyTasksNoOverlap)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_two_mcpy_nooverlap";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[1, 1024, 8192, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 1024, 0, 0, 0));
    tasks.push_back(MakeMemCpyTask(0, 2048, 1, 2048, 1024, 0, 0, 0));

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckMemCpySrcDstOverlapConflict)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_mcpy_overlap";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[1, 1024, 8192, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 1024, 0, 0, 0));
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 1024, 0, 0, 0));

    auto result = checker.Check(tasks);
    EXPECT_FALSE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckMemCpyWriteWriteConflict)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_ww_conflict";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[1, 1024, 8192, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 1024, 0, 0, 0));
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 1024, 0, 0, 1));

    auto result = checker.Check(tasks);
    EXPECT_FALSE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckReduceTask)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_reduce";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[1, 1024, 8192, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeReduceTask(0, 0, 1, 0, 256));

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckReduceConflictWithMemCpy)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_reduce_mcpy_conflict";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[1, 1024, 8192, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 1024, 0, 0, 0));
    tasks.push_back(MakeReduceTask(0, 0, 1, 0, 256, 0, 0, 0));

    auto result = checker.Check(tasks);
    EXPECT_FALSE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckNotifyRecordTask)
{
    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeNotifyRecordTask(0, 1, 1));

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());
}

TEST_F(CheckRankMemTest, CheckNotifyWaitTask)
{
    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeNotifyWaitTask(0, 1, 1));

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());
}

TEST_F(CheckRankMemTest, CheckNotifyRecordAndWaitNoConflict)
{
    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeNotifyRecordTask(0, 1, 1, 1, 0, 0, 0));
    tasks.push_back(MakeNotifyWaitTask(1, 1, 0, 1, 0, 0, 1));

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());
}

TEST_F(CheckRankMemTest, CheckCcuGraphTask)
{
    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeCcuGraphTask(0, 0, 0));

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());
}

TEST_F(CheckRankMemTest, CheckAivGraphTask)
{
    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeAivGraphTask(0, 0, 0));

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());
}

TEST_F(CheckRankMemTest, CheckMixedTasksNoConflict)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_mixed_noconflict";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[1, 1024, 8192, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 1024, 0, 0, 0));
    tasks.push_back(MakeNotifyRecordTask(0, 1, 1, 1, 0, 0, 0));
    tasks.push_back(MakeNotifyWaitTask(1, 1, 0, 1, 0, 0, 1));
    tasks.push_back(MakeMemCpyTask(0, 2048, 1, 2048, 1024, 0, 0, 1));

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckMixedTasksWithConflict)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_mixed_conflict";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[1, 1024, 8192, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 1024, 0, 0, 0));
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 1024, 0, 0, 0));

    auto result = checker.Check(tasks);
    EXPECT_FALSE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckDifferentRanksNoConflict)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_diff_ranks";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 4096, 0]");
    memLayoutLines.push_back("[1, 1024, 4096, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker0(0);
    std::vector<HcclTaskMetaData> tasks0;
    tasks0.push_back(MakeMemCpyTask(0, 0, 1, 0, 1024, 0, 0, 0));
    auto result0 = checker0.Check(tasks0);
    EXPECT_TRUE(result0.empty());

    CheckRankMem checker1(1);
    std::vector<HcclTaskMetaData> tasks1;
    tasks1.push_back(MakeMemCpyTask(1, 0, 0, 0, 1024, 0, 1, 0));
    auto result1 = checker1.Check(tasks1);
    EXPECT_TRUE(result1.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckReadWriteConflict)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_rw_conflict";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[1, 1024, 8192, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 1024, 0, 0, 0));
    tasks.push_back(MakeReduceTask(0, 0, 1, 0, 256, 0, 0, 0));

    auto result = checker.Check(tasks);
    EXPECT_FALSE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckMultipleStreamNoConflict)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_multi_stream";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[1, 1024, 8192, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 1024, 0, 0, 0));
    tasks.push_back(MakeMemCpyTask(0, 2048, 1, 2048, 1024, 0, 0, 1));

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckCrossStreamConflict)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_cross_stream_conflict";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[1, 1024, 8192, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 1024, 0, 0, 0));
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 1024, 0, 0, 1));

    auto result = checker.Check(tasks);
    EXPECT_FALSE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckMemCpyDifferentSrcDstNoConflict)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_diff_srcdst";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":4})");
    memLayoutLines.push_back("[0, 1024, 4096, 0]");
    memLayoutLines.push_back("[0, 8192, 4096, 1]");
    memLayoutLines.push_back("[1, 1024, 4096, 0]");
    memLayoutLines.push_back("[1, 8192, 4096, 1]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeMemCpyTask(0, 1024, 1, 8192, 1024, 0, 0, 0));
    tasks.push_back(MakeMemCpyTask(1, 8192, 0, 1024, 1024, 0, 0, 0));

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckNotifySyncBetweenStreams)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_notify_sync";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[1, 1024, 8192, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 1024, 0, 0, 0));
    tasks.push_back(MakeNotifyRecordTask(0, 1, 1, 1, 0, 0, 0));
    tasks.push_back(MakeNotifyWaitTask(0, 1, 1, 1, 0, 0, 1));
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 1024, 0, 0, 1));

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckNotifySyncStillConflicts)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_notify_still_conflict";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[1, 1024, 8192, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 1024, 0, 0, 0));
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 1024, 0, 0, 0));

    auto result = checker.Check(tasks);
    EXPECT_FALSE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckLargeMemCpyTask)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_large_mcpy";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 65536, 0]");
    memLayoutLines.push_back("[1, 1024, 65536, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 32768));

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckMultipleReduceTasksNoConflict)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_multi_reduce";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[1, 1024, 8192, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeReduceTask(0, 0, 1, 0, 256, 0, 0, 0));
    tasks.push_back(MakeReduceTask(0, 2048, 1, 2048, 256, 0, 0, 0));

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckReduceOverlapConflict)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_reduce_overlap";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[1, 1024, 8192, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeReduceTask(0, 0, 1, 0, 256, 0, 0, 0));
    tasks.push_back(MakeReduceTask(0, 0, 1, 0, 256, 0, 0, 0));

    auto result = checker.Check(tasks);
    EXPECT_FALSE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckMemCpyPartialOverlap)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_partial_overlap";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[1, 1024, 8192, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 1024, 0, 0, 0));
    tasks.push_back(MakeMemCpyTask(0, 512, 1, 512, 1024, 0, 0, 0));

    auto result = checker.Check(tasks);
    EXPECT_FALSE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckMemCpyAdjacentNoOverlap)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_adjacent";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[1, 1024, 8192, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 1024, 0, 0, 0));
    tasks.push_back(MakeMemCpyTask(0, 1024, 1, 1024, 1024, 0, 0, 0));

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckMemCpyZeroLen)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_zero_len";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[1, 1024, 8192, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 0));

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckReduceZeroDataCount)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_reduce_zero";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[1, 1024, 8192, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeReduceTask(0, 0, 1, 0, 0));

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckMemCpyOutputBufferConflict)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_output_conflict";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":4})");
    memLayoutLines.push_back("[0, 1024, 4096, 0]");
    memLayoutLines.push_back("[0, 8192, 4096, 1]");
    memLayoutLines.push_back("[1, 1024, 4096, 0]");
    memLayoutLines.push_back("[1, 8192, 4096, 1]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeMemCpyTask(0, 1024, 1, 8192, 1024, 0, 0, 0));
    tasks.push_back(MakeMemCpyTask(0, 1024, 1, 8192, 1024, 0, 0, 0));

    auto result = checker.Check(tasks);
    EXPECT_FALSE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckMemCpyInputOutputNoConflict)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_inout_noconflict";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":4})");
    memLayoutLines.push_back("[0, 1024, 4096, 0]");
    memLayoutLines.push_back("[0, 8192, 4096, 1]");
    memLayoutLines.push_back("[1, 1024, 4096, 0]");
    memLayoutLines.push_back("[1, 8192, 4096, 1]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeMemCpyTask(0, 1024, 1, 8192, 1024, 0, 0, 0));
    tasks.push_back(MakeMemCpyTask(1, 8192, 0, 1024, 1024, 0, 0, 0));

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckNotifyRecordAndWaitSync)
{
    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeNotifyRecordTask(0, 1, 1, 1, 0, 0, 0));
    tasks.push_back(MakeNotifyWaitTask(0, 1, 1, 1, 0, 0, 1));

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());
}

TEST_F(CheckRankMemTest, CheckMultipleNotifyIds)
{
    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeNotifyRecordTask(0, 1, 1, 1, 0, 0, 0));
    tasks.push_back(MakeNotifyRecordTask(0, 2, 1, 1, 0, 0, 0));
    tasks.push_back(MakeNotifyWaitTask(0, 1, 1, 1, 0, 0, 1));
    tasks.push_back(MakeNotifyWaitTask(0, 2, 1, 1, 0, 0, 1));

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());
}

TEST_F(CheckRankMemTest, CheckCcuGraphAndAivGraphMixed)
{
    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeCcuGraphTask(0, 0, 0));
    tasks.push_back(MakeAivGraphTask(0, 0, 1));
    tasks.push_back(MakeCcuGraphTask(0, 0, 2));

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());
}

TEST_F(CheckRankMemTest, CheckMemCpySameStreamSequential)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_same_stream_seq";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[1, 1024, 8192, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 1024, 0, 0, 0));
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 1024, 0, 0, 0));

    auto result = checker.Check(tasks);
    EXPECT_FALSE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckMemCpySelfCopy)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_self_copy";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[0, 8192, 4096, 1]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeMemCpyTask(0, 1024, 0, 8192, 1024));

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckMemCpySelfCopyConflict)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_self_copy_conflict";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[0, 8192, 4096, 1]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeMemCpyTask(0, 1024, 0, 8192, 1024, 0, 0, 0));
    tasks.push_back(MakeMemCpyTask(0, 1024, 0, 8192, 1024, 0, 0, 0));

    auto result = checker.Check(tasks);
    EXPECT_FALSE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckMemCpyDifferentCommId)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_diff_comm";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[1, 1024, 8192, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 1024, 0, 0, 0));
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 1024, 1, 0, 0));

    auto result = checker.Check(tasks);
    EXPECT_FALSE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckReduceThenMemCpyNoConflict)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_reduce_then_mcpy";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[1, 1024, 8192, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeReduceTask(0, 0, 1, 0, 256, 0, 0, 0));
    tasks.push_back(MakeMemCpyTask(0, 2048, 1, 2048, 1024, 0, 0, 0));

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckMemCpyThenReduceConflict)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_mcpy_reduce_conflict";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[1, 1024, 8192, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 1024, 0, 0, 0));
    tasks.push_back(MakeReduceTask(0, 0, 1, 0, 256, 0, 0, 0));

    auto result = checker.Check(tasks);
    EXPECT_FALSE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckNotifyRecordOnly)
{
    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    for (int i = 0; i < 5; i++) {
        tasks.push_back(MakeNotifyRecordTask(0, i, 1, 1, 0, 0, 0));
    }

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());
}

TEST_F(CheckRankMemTest, CheckNotifyWaitOnly)
{
    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    for (int i = 0; i < 5; i++) {
        tasks.push_back(MakeNotifyWaitTask(0, i, 1, 1, 0, 0, 0));
    }

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());
}

TEST_F(CheckRankMemTest, CheckMemCpyWithNotifySyncNoConflict)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_mcpy_notify_sync";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[1, 1024, 8192, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 1024, 0, 0, 0));
    tasks.push_back(MakeNotifyRecordTask(0, 1, 1, 1, 0, 0, 0));
    tasks.push_back(MakeNotifyWaitTask(0, 1, 1, 1, 0, 0, 1));
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 1024, 0, 0, 1));

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckMemCpyWithoutNotifySyncConflict)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_mcpy_no_notify";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[1, 1024, 8192, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 1024, 0, 0, 0));
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 1024, 0, 0, 1));

    auto result = checker.Check(tasks);
    EXPECT_FALSE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckMultipleNotifyRecordsForSameId)
{
    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeNotifyRecordTask(0, 1, 1, 1, 0, 0, 0));
    tasks.push_back(MakeNotifyRecordTask(0, 1, 1, 1, 0, 0, 0));

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());
}

TEST_F(CheckRankMemTest, CheckNotifyWaitBeforeRecord)
{
    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeNotifyWaitTask(0, 1, 1, 1, 0, 0, 0));
    tasks.push_back(MakeNotifyRecordTask(0, 1, 1, 1, 0, 0, 0));

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());
}

TEST_F(CheckRankMemTest, CheckAllTaskTypesMixed)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_all_types";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[1, 1024, 8192, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 1024, 0, 0, 0));
    tasks.push_back(MakeNotifyRecordTask(0, 1, 1, 1, 0, 0, 0));
    tasks.push_back(MakeCcuGraphTask(0, 0, 0));
    tasks.push_back(MakeAivGraphTask(0, 0, 1));
    tasks.push_back(MakeNotifyWaitTask(0, 1, 1, 1, 0, 0, 1));
    tasks.push_back(MakeReduceTask(0, 2048, 1, 2048, 256, 0, 0, 1));

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckRankIdAccessor)
{
    CheckRankMem checker0(0);
    EXPECT_EQ(checker0.GetRankId(), 0);

    CheckRankMem checker7(7);
    EXPECT_EQ(checker7.GetRankId(), 7);
}

TEST_F(CheckRankMemTest, CheckMemCpyDstOverlapWithReduceSrc)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_dst_reduce_src";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":4})");
    memLayoutLines.push_back("[0, 1024, 4096, 0]");
    memLayoutLines.push_back("[0, 8192, 4096, 1]");
    memLayoutLines.push_back("[1, 1024, 4096, 0]");
    memLayoutLines.push_back("[1, 8192, 4096, 1]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeMemCpyTask(0, 1024, 1, 8192, 1024, 0, 0, 0));
    tasks.push_back(MakeReduceTask(0, 1024, 1, 8192, 256, 0, 0, 0));

    auto result = checker.Check(tasks);
    EXPECT_FALSE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckMemCpySrcOnlyConflict)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_src_only_conflict";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[1, 1024, 8192, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeReduceTask(0, 0, 1, 0, 256, 0, 0, 0));
    tasks.push_back(MakeReduceTask(0, 0, 1, 2048, 256, 0, 0, 0));

    auto result = checker.Check(tasks);
    EXPECT_FALSE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckMemCpyDstOnlyConflict)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_dst_only_conflict";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[1, 1024, 8192, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeReduceTask(0, 0, 1, 0, 256, 0, 0, 0));
    tasks.push_back(MakeReduceTask(0, 2048, 1, 0, 256, 0, 0, 0));

    auto result = checker.Check(tasks);
    EXPECT_FALSE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckNotifyWithMultipleNotifyCount)
{
    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeNotifyRecordTask(0, 1, 1, 3, 0, 0, 0));
    tasks.push_back(MakeNotifyWaitTask(0, 1, 1, 3, 0, 0, 1));

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());
}

TEST_F(CheckRankMemTest, CheckReduceWithDifferentDataType)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_reduce_dtype";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[1, 1024, 8192, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    HcclTaskMetaData reduceTask = MakeReduceTask(0, 0, 1, 0, 256);
    reduceTask.taskData.reduce.dataType = 1;
    tasks.push_back(reduceTask);

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckReduceWithDifferentReduceOp)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_reduce_op";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[1, 1024, 8192, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    HcclTaskMetaData reduceTask = MakeReduceTask(0, 0, 1, 0, 256);
    reduceTask.taskData.reduce.reduceOp = 1;
    tasks.push_back(reduceTask);

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckMemCpyWithProtocol)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_mcpy_protocol";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[1, 1024, 8192, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    HcclTaskMetaData mcpyTask = MakeMemCpyTask(0, 0, 1, 0, 1024);
    mcpyTask.taskData.transMem.protocol = 1;
    tasks.push_back(mcpyTask);

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckNotifyWithProtocol)
{
    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    HcclTaskMetaData notifyTask = MakeNotifyRecordTask(0, 1, 1);
    notifyTask.taskData.notify.protocol = 1;
    tasks.push_back(notifyTask);

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());
}

TEST_F(CheckRankMemTest, CheckThreeWayConflict)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_three_way";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[1, 1024, 8192, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 1024, 0, 0, 0));
    tasks.push_back(MakeReduceTask(0, 0, 1, 0, 256, 0, 0, 0));
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 512, 0, 0, 0));

    auto result = checker.Check(tasks);
    EXPECT_FALSE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckMultipleRanksSameChecker)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_multi_ranks";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":4})");
    memLayoutLines.push_back("[0, 1024, 4096, 0]");
    memLayoutLines.push_back("[0, 8192, 4096, 1]");
    memLayoutLines.push_back("[1, 1024, 4096, 0]");
    memLayoutLines.push_back("[1, 8192, 4096, 1]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker0(0);
    std::vector<HcclTaskMetaData> tasks0;
    tasks0.push_back(MakeMemCpyTask(0, 1024, 1, 8192, 1024, 0, 0, 0));
    auto result0 = checker0.Check(tasks0);
    EXPECT_TRUE(result0.empty());

    CheckRankMem checker1(1);
    std::vector<HcclTaskMetaData> tasks1;
    tasks1.push_back(MakeMemCpyTask(1, 8192, 0, 1024, 1024, 0, 1, 0));
    auto result1 = checker1.Check(tasks1);
    EXPECT_TRUE(result1.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckNotifyRecordMultipleWaits)
{
    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeNotifyRecordTask(0, 1, 1, 1, 0, 0, 0));
    tasks.push_back(MakeNotifyWaitTask(0, 1, 1, 1, 0, 0, 1));
    tasks.push_back(MakeNotifyWaitTask(0, 1, 1, 1, 0, 0, 2));

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());
}

TEST_F(CheckRankMemTest, CheckMemCpyDifferentBufferTypesNoConflict)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_diff_buf";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":4})");
    memLayoutLines.push_back("[0, 1024, 4096, 0]");
    memLayoutLines.push_back("[0, 8192, 4096, 1]");
    memLayoutLines.push_back("[1, 1024, 4096, 0]");
    memLayoutLines.push_back("[1, 8192, 4096, 1]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeMemCpyTask(0, 1024, 1, 1024, 1024, 0, 0, 0));
    tasks.push_back(MakeMemCpyTask(0, 8192, 1, 8192, 1024, 0, 0, 0));

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckReduceAndNotifyNoConflict)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_reduce_notify";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[1, 1024, 8192, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeReduceTask(0, 0, 1, 0, 256, 0, 0, 0));
    tasks.push_back(MakeNotifyRecordTask(0, 1, 1, 1, 0, 0, 0));

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckCcuGraphWithMemCpyNoConflict)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_ccu_mcpy";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[1, 1024, 8192, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeCcuGraphTask(0, 0, 0));
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 1024, 0, 0, 0));

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckAivGraphWithMemCpyNoConflict)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_aiv_mcpy";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 8192, 0]");
    memLayoutLines.push_back("[1, 1024, 8192, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    tasks.push_back(MakeAivGraphTask(0, 0, 0));
    tasks.push_back(MakeMemCpyTask(0, 0, 1, 0, 1024, 0, 0, 0));

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckLargeNumberOfTasks)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_large_tasks";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 1048576, 0]");
    memLayoutLines.push_back("[1, 1024, 1048576, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    for (int i = 0; i < 100; i++) {
        tasks.push_back(MakeMemCpyTask(0, i * 1024, 1, i * 1024, 1024, 0, 0, 0));
    }

    auto result = checker.Check(tasks);
    EXPECT_FALSE(result.empty());

    RemoveDir(tmpDir);
}

TEST_F(CheckRankMemTest, CheckLargeNumberOfTasksNoConflict)
{
    std::string tmpDir = CreateTempDir();
    std::string dataId = "test_large_noconflict";

    std::vector<std::string> memLayoutLines;
    memLayoutLines.push_back(R"({"total_count":2})");
    memLayoutLines.push_back("[0, 1024, 1048576, 0]");
    memLayoutLines.push_back("[1, 1024, 1048576, 0]");

    SetupStorageManagerWithMemLayout(tmpDir, dataId, memLayoutLines);

    CheckRankMem checker(0);
    std::vector<HcclTaskMetaData> tasks;
    for (int i = 0; i < 50; i++) {
        tasks.push_back(MakeNotifyRecordTask(0, i, 1, 1, 0, 0, 0));
        tasks.push_back(MakeNotifyWaitTask(0, i, 1, 1, 0, 0, 1));
        tasks.push_back(MakeMemCpyTask(0, i * 1024, 1, i * 1024, 1024, 0, 0, 0));
        tasks.push_back(MakeMemCpyTask(0, i * 1024, 1, i * 1024, 1024, 0, 0, 1));
    }

    auto result = checker.Check(tasks);
    EXPECT_TRUE(result.empty());

    RemoveDir(tmpDir);
}

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <cstring>
#include <gtest/gtest.h>
#include <unistd.h>

#include "binary_data_operator.h"
#include "sim_binary_data_type_pub.h"

namespace HcclSim {
class BinaryDataOperatorCheckerTest : public testing::Test {
protected:
    void SetUp() override
    {
        strcpy(testFilePath, "/tmp/binary_checker_test_XXXXXX");
        int fd = mkstemp(testFilePath);
        ASSERT_NE(fd, -1);
        close(fd);
    }

    void TearDown() override
    {
        if (access(testFilePath, F_OK) == 0) {
            unlink(testFilePath);
        }
    }

    char testFilePath[256];
};

TEST_F(BinaryDataOperatorCheckerTest, TaskMetaWrite_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    HcclTaskMetaData taskData;
    memset(&taskData, 0, sizeof(taskData));
    taskData.taskType = HccLTaskMetaType::MEM_CPY;
    taskData.rankId = 0;
    taskData.streamId = 1;

    HcclResult ret = TaskMetaWrite(fp, taskData);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    fclose(fp);
}

TEST_F(BinaryDataOperatorCheckerTest, TaskMetaRead_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    HcclTaskMetaData taskDataIn;
    memset(&taskDataIn, 0, sizeof(taskDataIn));
    taskDataIn.taskType = HccLTaskMetaType::REDUCE;
    taskDataIn.rankId = 1;
    taskDataIn.streamId = 2;
    taskDataIn.taskData.reduce.srcRankId = 0;
    taskDataIn.taskData.reduce.dstRankId = 1;

    TaskMetaWrite(fp, taskDataIn);
    fclose(fp);

    fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    HcclTaskMetaData taskDataOut;
    HcclResult ret = TaskMetaRead(fp, taskDataOut);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(taskDataOut.taskType, HccLTaskMetaType::REDUCE);
    EXPECT_EQ(taskDataOut.rankId, 1u);

    fclose(fp);
}

TEST_F(BinaryDataOperatorCheckerTest, HcclVmTaskMetaDataWrite_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    HcclVmTaskMetaData taskMeta;
    memset(&taskMeta, 0, sizeof(taskMeta));
    taskMeta.header.magic = HCCLVM_TASK_FILE_MAGIC;
    taskMeta.header.version = 1;
    taskMeta.header.header_size = sizeof(FileHeader);
    taskMeta.header.count = 2;

    HcclTaskMetaData task1;
    memset(&task1, 0, sizeof(task1));
    task1.taskType = HccLTaskMetaType::MEM_CPY;
    task1.rankId = 0;

    HcclTaskMetaData task2;
    memset(&task2, 0, sizeof(task2));
    task2.taskType = HccLTaskMetaType::REDUCE;
    task2.rankId = 1;

    taskMeta.task_meta.push_back(task1);
    taskMeta.task_meta.push_back(task2);

    HcclResult ret = HcclVmTaskMetaDataWrite(fp, taskMeta);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    fclose(fp);
}

TEST_F(BinaryDataOperatorCheckerTest, HcclVmTaskMetaDataRead_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    HcclVmTaskMetaData taskMetaIn;
    memset(&taskMetaIn, 0, sizeof(taskMetaIn));
    taskMetaIn.header.magic = HCCLVM_TASK_FILE_MAGIC;
    taskMetaIn.header.version = 1;
    taskMetaIn.header.header_size = sizeof(FileHeader);
    taskMetaIn.header.count = 1;

    HcclTaskMetaData task;
    memset(&task, 0, sizeof(task));
    task.taskType = HccLTaskMetaType::NOTIFY_RECORD;
    task.rankId = 2;
    taskMetaIn.task_meta.push_back(task);

    HcclVmTaskMetaDataWrite(fp, taskMetaIn);
    fclose(fp);

    fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    HcclVmTaskMetaData taskMetaOut;
    HcclResult ret = HcclVmTaskMetaDataRead(fp, taskMetaOut, HCCLVM_TASK_FILE_MAGIC);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(taskMetaOut.header.magic, HCCLVM_TASK_FILE_MAGIC);
    EXPECT_EQ(taskMetaOut.header.count, 1u);

    fclose(fp);
}

TEST_F(BinaryDataOperatorCheckerTest, HcclVmInstrDataWrite_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    HcclVmInstrData instrData;
    memset(&instrData, 0, sizeof(instrData));
    instrData.header.magic = HCCLVM_INSTR_FILE_MAGIC;
    instrData.header.version = 1;
    instrData.header.header_size = sizeof(FileHeader);
    instrData.header.count = 1;

    MicrocodeInstrInner mcInstr;
    memset(&mcInstr, 0, sizeof(mcInstr));
    mcInstr.desc.rank_id = 0;
    mcInstr.desc.die_id = 0;
    mcInstr.desc.count = 0;

    instrData.instr_data.push_back(mcInstr);

    HcclResult ret = HcclVmInstrDataWrite(fp, instrData);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    fclose(fp);
}

TEST_F(BinaryDataOperatorCheckerTest, HcclVmInstrDataRead_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    HcclVmInstrData instrDataIn;
    memset(&instrDataIn, 0, sizeof(instrDataIn));
    instrDataIn.header.magic = HCCLVM_INSTR_FILE_MAGIC;
    instrDataIn.header.version = 1;
    instrDataIn.header.header_size = sizeof(FileHeader);
    instrDataIn.header.count = 1;

    MicrocodeInstrInner mcInstr;
    memset(&mcInstr, 0, sizeof(mcInstr));
    mcInstr.desc.rank_id = 1;
    mcInstr.desc.die_id = 1;
    mcInstr.desc.count = 0;

    instrDataIn.instr_data.push_back(mcInstr);

    HcclVmInstrDataWrite(fp, instrDataIn);
    fclose(fp);

    fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    HcclVmInstrData instrDataOut;
    HcclResult ret = HcclVmInstrDataRead(fp, instrDataOut, HCCLVM_INSTR_FILE_MAGIC);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(instrDataOut.header.magic, HCCLVM_INSTR_FILE_MAGIC);

    fclose(fp);
}

TEST_F(BinaryDataOperatorCheckerTest, MicrocodeInstrWrite_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    MicrocodeInstrInner mcInstr;
    memset(&mcInstr, 0, sizeof(mcInstr));
    mcInstr.desc.rank_id = 0;
    mcInstr.desc.die_id = 0;
    mcInstr.desc.count = 0;

    HcclResult ret = MicrocodeInstrWrite(fp, mcInstr);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);

    fclose(fp);
}

TEST_F(BinaryDataOperatorCheckerTest, MicrocodeInstrRead_Normal_Success)
{
    FILE* fp = fopen(testFilePath, "wb");
    ASSERT_NE(fp, nullptr);

    MicrocodeInstrInner mcInstrIn;
    memset(&mcInstrIn, 0, sizeof(mcInstrIn));
    mcInstrIn.desc.rank_id = 2;
    mcInstrIn.desc.die_id = 1;
    mcInstrIn.desc.count = 0;

    MicrocodeInstrWrite(fp, mcInstrIn);
    fclose(fp);

    fp = fopen(testFilePath, "rb");
    ASSERT_NE(fp, nullptr);

    MicrocodeInstrInner mcInstrOut;
    HcclResult ret = MicrocodeInstrRead(fp, mcInstrOut);
    EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
    EXPECT_EQ(mcInstrOut.desc.rank_id, 2u);
    EXPECT_EQ(mcInstrOut.desc.die_id, 1u);

    fclose(fp);
}
} // namespace HcclSim

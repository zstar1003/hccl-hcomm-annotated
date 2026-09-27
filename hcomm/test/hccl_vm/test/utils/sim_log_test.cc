/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include "sim_log.h"

class HcclVmLogTest : public testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override { DeInitLogger(); }
};

TEST_F(HcclVmLogTest, LogConfig_DefaultValues)
{
    LogConfig config;
    EXPECT_EQ(config.consoleLevel, 2);
    EXPECT_EQ(config.fileLevel, 1);
    EXPECT_EQ(config.maxFileSize, 10 * 1024 * 1024);
    EXPECT_EQ(config.maxFiles, UINT16_MAX);
    EXPECT_EQ(config.filePath, "logs");
    EXPECT_EQ(config.fileBaseName, "app_log");
    EXPECT_EQ(config.fileSuffix, ".log");
    EXPECT_EQ(config.enableCompress, false);
}

TEST_F(HcclVmLogTest, LogConfig_CustomValues)
{
    LogConfig config;
    config.consoleLevel = 0;
    config.fileLevel = 0;
    config.maxFileSize = 1024 * 1024;
    config.maxFiles = 10;
    config.filePath = "/tmp/test_logs";
    config.fileBaseName = "test_log";
    config.fileSuffix = ".txt";
    config.enableCompress = true;

    EXPECT_EQ(config.consoleLevel, 0);
    EXPECT_EQ(config.fileLevel, 0);
    EXPECT_EQ(config.maxFileSize, 1024 * 1024);
    EXPECT_EQ(config.maxFiles, 10);
    EXPECT_EQ(config.filePath, "/tmp/test_logs");
    EXPECT_EQ(config.fileBaseName, "test_log");
    EXPECT_EQ(config.fileSuffix, ".txt");
    EXPECT_EQ(config.enableCompress, true);
}

TEST_F(HcclVmLogTest, InitLogger_DefaultConfig)
{
    LogConfig config;
    EXPECT_NO_THROW(InitLogger(config));
}

TEST_F(HcclVmLogTest, InitLogger_CustomConfig)
{
    LogConfig config;
    config.consoleLevel = 0;
    config.fileLevel = 0;
    config.maxFileSize = 1024 * 1024;
    config.maxFiles = 5;
    config.filePath = "/tmp";
    EXPECT_NO_THROW(InitLogger(config));
}

TEST_F(HcclVmLogTest, DeInitLogger_Nullptr)
{
    DeInitLogger();
    EXPECT_EQ(g_logger, nullptr);
}

TEST_F(HcclVmLogTest, Logger_MacrosWithNullptr)
{
    DeInitLogger();
    EXPECT_NO_THROW(HCCL_VM_TRACE("test trace"));
    EXPECT_NO_THROW(HCCL_VM_DEBUG("test debug"));
    EXPECT_NO_THROW(HCCL_VM_INFO("test info"));
    EXPECT_NO_THROW(HCCL_VM_WARN("test warn"));
    EXPECT_NO_THROW(HCCL_VM_ERROR("test error"));
    EXPECT_NO_THROW(HCCL_VM_CRITICAL("test critical"));
}

TEST_F(HcclVmLogTest, Logger_MacrosWithValidLogger)
{
    LogConfig config;
    config.filePath = "/tmp";
    InitLogger(config);

    EXPECT_NO_THROW(HCCL_VM_TRACE("test trace {}", 1));
    EXPECT_NO_THROW(HCCL_VM_DEBUG("test debug {}", 2));
    EXPECT_NO_THROW(HCCL_VM_INFO("test info {}", 3));
    EXPECT_NO_THROW(HCCL_VM_WARN("test warn {}", 4));
    EXPECT_NO_THROW(HCCL_VM_ERROR("test error {}", 5));
    EXPECT_NO_THROW(HCCL_VM_CRITICAL("test critical {}", 6));
}

TEST_F(HcclVmLogTest, Logger_MultipleInit)
{
    LogConfig config1;
    config1.filePath = "/tmp";
    InitLogger(config1);
    EXPECT_NE(g_logger, nullptr);

    LogConfig config2;
    config2.filePath = "/tmp";
    InitLogger(config2);
    EXPECT_NE(g_logger, nullptr);
}

TEST_F(HcclVmLogTest, LogConfig_StructSize) { EXPECT_GT(sizeof(LogConfig), 0); }

TEST_F(HcclVmLogTest, InitLogger_LargeFileSize)
{
    LogConfig config;
    config.filePath = "/tmp";
    config.maxFileSize = 100 * 1024 * 1024;
    EXPECT_NO_THROW(InitLogger(config));
}

TEST_F(HcclVmLogTest, InitLogger_MaxFiles)
{
    LogConfig config;
    config.filePath = "/tmp";
    config.maxFiles = UINT16_MAX;
    EXPECT_NO_THROW(InitLogger(config));
}

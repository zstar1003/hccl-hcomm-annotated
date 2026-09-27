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
#include <gtest/gtest.h>
#include <iostream>
#include <sstream>

#include "cmd_table_utils.h"
#include "db_sim_runner_db.h"

using namespace HcclSim;

class CmdTableUtilsTest : public testing::Test {
protected:
    void SetUp() override
    {
        savedCoutBuf_ = std::cout.rdbuf();
        std::cout.rdbuf(ssCout_.rdbuf());
    }
    void TearDown() override { std::cout.rdbuf(savedCoutBuf_); }
    std::stringstream ssCout_;
    std::streambuf* savedCoutBuf_;
};

TEST_F(CmdTableUtilsTest, CmdTableShow_UnknownTable_PrintsUndefine)
{
    std::string tableName = "NonExistentTable";
    CmdTableShow(tableName);
    std::string output = ssCout_.str();
    EXPECT_NE(output.find("undefine table"), std::string::npos);
    EXPECT_NE(output.find("NonExistentTable"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableUpdate_UnknownTable_ReturnsFalse)
{
    bool result = CmdTableUpdate("BogusTable", 42, "some_column", "some_value");
    EXPECT_FALSE(result);
    std::string output = ssCout_.str();
    EXPECT_NE(output.find("undefine update"), std::string::npos);
    EXPECT_NE(output.find("BogusTable"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableUpdate_UnknownColumn_ReturnsFalse)
{
    bool result = CmdTableUpdate("Device", 1, "bogus_column", "val");
    EXPECT_FALSE(result);
    std::string output = ssCout_.str();
    EXPECT_NE(output.find("undefine update"), std::string::npos);
    EXPECT_NE(output.find("Device"), std::string::npos);
    EXPECT_NE(output.find("bogus_column"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_Device)
{
    std::string tableName = "Device";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("id"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_Server)
{
    std::string tableName = "Server";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("id"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_Host)
{
    std::string tableName = "Host";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("id"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_Runner)
{
    std::string tableName = "Runner";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("id"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_TaskSchedulerDevice)
{
    std::string tableName = "TaskSchedulerDevice";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("id"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_ComputeDie)
{
    std::string tableName = "ComputeDie";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("id"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_DeviceStatus)
{
    std::string tableName = "DeviceStatus";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("id"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_Port)
{
    std::string tableName = "Port";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("id"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_Ccu)
{
    std::string tableName = "Ccu";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("id"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_CcuResource)
{
    std::string tableName = "CcuResource";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("id"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_DeviceConnection)
{
    std::string tableName = "DeviceConnection";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("id"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_EndPointPair)
{
    std::string tableName = "EndPointPair";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("id"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_CcuChannel)
{
    std::string tableName = "CcuChannel";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("id"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_Context)
{
    std::string tableName = "Context";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("id"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_Stream)
{
    std::string tableName = "Stream";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("id"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_Task)
{
    std::string tableName = "Task";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("id"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_EventSyncTask)
{
    std::string tableName = "EventSyncTask";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("id"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_Notify)
{
    std::string tableName = "Notify";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("id"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_IpcNotify)
{
    std::string tableName = "IpcNotify";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("id"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_IpcNotifyVistorList)
{
    std::string tableName = "IpcNotifyVistorList";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("ipc_id"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_NotifyRecordTask)
{
    std::string tableName = "NotifyRecordTask";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("notify_id"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_NotifyWaitTask)
{
    std::string tableName = "NotifyWaitTask";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("notify_id"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_Event)
{
    std::string tableName = "Event";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("id"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_PhyMem)
{
    std::string tableName = "PhyMem";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("id"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_VirMem)
{
    std::string tableName = "VirMem";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("id"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_IpcMemRecord)
{
    std::string tableName = "IpcMemRecord";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("id"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_IpcMemWhiteList)
{
    std::string tableName = "IpcMemWhiteList";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("id"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_FdMemRecord)
{
    std::string tableName = "FdMemRecord";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("id"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_FdMemWhiteList)
{
    std::string tableName = "FdMemWhiteList";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("id"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_RaSocket)
{
    std::string tableName = "RaSocket";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("id"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_RaSocketPair)
{
    std::string tableName = "RaSocketPair";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("id"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_MemoryLayout)
{
    std::string tableName = "MemoryLayout";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("id"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_SimModelData)
{
    std::string tableName = "SimModelData";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("id"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_Rank)
{
    std::string tableName = "Rank";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("id"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_RaDevice)
{
    std::string tableName = "RaDevice";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("id"), std::string::npos);
}

TEST_F(CmdTableUtilsTest, CmdTableShow_RaQP)
{
    std::string tableName = "RaQP";
    CmdTableShow(tableName);
    EXPECT_NE(ssCout_.str().find("id"), std::string::npos);
}

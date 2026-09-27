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
#include <cstring>
#include <gtest/gtest.h>
#include <map>
#include <regex>
#include <vector>

#include "sim_binary_data_type_pub.h"
#include "store_dump_shm_data.h"
#include "sim_data_dump.h"
#include "sim_op_db_types.h"

using namespace HcclSim;

namespace {
std::vector<uint8_t> ToBlob(const HcclTaskMetaData& task)
{
    const auto* begin = reinterpret_cast<const uint8_t*>(&task);
    return std::vector<uint8_t>(begin, begin + sizeof(HcclTaskMetaData));
}
} // namespace

TEST(HcclDataDumpTest, DumpData_FillsFinishedStatusAndDataId)
{
    nlohmann::json data;
    EXPECT_EQ(DumpData(data), HcclVmResult::HCCL_SIM_SUCCESS);
    EXPECT_EQ(data["status"], "finish");
    EXPECT_EQ(data["data_id"], "1234");
}

TEST(HcclDataDumpTest, GenDataId_UsesTimestampAndRandomSuffix)
{
    const std::string dataId = GenDataId();
    const std::regex pattern(R"(^[0-9]{8}_[0-9]{6}_[0-9]{4}$)");
    EXPECT_TRUE(std::regex_match(dataId, pattern)) << dataId;
}

TEST(HcclDataDumpTest, CreateMemoryInfo_AddsOnlyNonZeroBuffers)
{
    HcclVmSynData synData{};
    sim::OpMemInfoTab memInfo{};
    memInfo.inputAddr = 0x1000;
    memInfo.inputSize = 128;
    memInfo.outputAddr = 0x2000;
    memInfo.outputSize = 256;
    memInfo.cclAddr = 0;
    memInfo.cclSize = 0;

    EXPECT_EQ(CreateMemoryInfo(synData, memInfo, 3), HcclVmResult::HCCL_SIM_SUCCESS);
    ASSERT_EQ(synData.memory_info.data.size(), 2U);
    EXPECT_EQ(synData.memory_info.count, 2U);
    EXPECT_EQ(synData.memory_info.data[0].rank_id, 3U);
    EXPECT_EQ(synData.memory_info.data[0].buffer_type, static_cast<uint8_t>(BufferType::INPUT));
    EXPECT_EQ(synData.memory_info.data[0].start_addr, 0x1000U);
    EXPECT_EQ(synData.memory_info.data[0].size, 128U);
    EXPECT_EQ(synData.memory_info.data[1].buffer_type, static_cast<uint8_t>(BufferType::OUTPUT));
    EXPECT_EQ(synData.memory_info.data[1].start_addr, 0x2000U);
    EXPECT_EQ(synData.memory_info.data[1].size, 256U);
}

TEST(HcclDataDumpTest, CreateSimTaskMetaData_SkipsShortBlobAndDeserializesValidTasks)
{
    HcclTaskMetaData memCopy{};
    memCopy.rankId = 1;
    memCopy.streamId = 2;
    memCopy.taskType = HccLTaskMetaType::MEM_CPY;
    memCopy.taskData.transMem.len = 4096;

    HcclTaskMetaData aiv{};
    aiv.rankId = 2;
    aiv.taskType = HccLTaskMetaType::AIV_GRAPH;
    aiv.taskData.aiv.launchIdx = 99;

    sim::OpTaskTab shortBlob{};
    shortBlob.optaskMeta = {0x1, 0x2, 0x3};

    sim::OpTaskTab memCopyBlob{};
    memCopyBlob.optaskMeta = ToBlob(memCopy);

    sim::OpTaskTab aivBlob{};
    aivBlob.optaskMeta = ToBlob(aiv);

    sim::CompositeOpDetail composite{};
    composite.tasks = {shortBlob, memCopyBlob, aivBlob};

    std::map<uint32_t, std::vector<sim::CompositeOpDetail>> compositeDataMap;
    compositeDataMap[0].push_back(composite);

    HcclVmTaskMetaData taskMeta{};
    EXPECT_EQ(CreateSimTaskMetaData(taskMeta, compositeDataMap), HcclVmResult::HCCL_SIM_SUCCESS);

    ASSERT_EQ(taskMeta.task_meta.size(), 2U);
    EXPECT_EQ(taskMeta.header.magic, HCCLVM_TASK_FILE_MAGIC);
    EXPECT_EQ(taskMeta.header.version, 1U);
    EXPECT_EQ(taskMeta.header.header_size, 20U);
    EXPECT_EQ(taskMeta.header.count, 2U);
    EXPECT_EQ(taskMeta.task_meta[0].rankId, 1U);
    EXPECT_EQ(taskMeta.task_meta[0].taskType, HccLTaskMetaType::MEM_CPY);
    EXPECT_EQ(taskMeta.task_meta[1].rankId, 2U);
    EXPECT_EQ(taskMeta.task_meta[1].taskType, HccLTaskMetaType::AIV_GRAPH);
    EXPECT_EQ(taskMeta.task_meta[1].taskData.aiv.launchIdx, 1U);
}

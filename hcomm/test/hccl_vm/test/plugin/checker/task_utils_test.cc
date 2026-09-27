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
#include <map>

#include "checker_def.h"
#include "hccl_types.h"
#include "storage_manager.h"
#include "task_ccu.h"
#include "task_meta_defs.h"
#include "task_utils.h"

#define private public
#undef private

using RankId = uint32_t;
using u32 = uint32_t;
std::map<RankId, std::map<u32, HcclSim::ChannelsPerDie>> g_allRankChannelInfo;

using namespace HcclSim;

namespace HcclSim {
LinkProtoStub GetLinkProto(uint8_t commProtocol);
uint64_t CalcDataSize(HcclDataType dataType, uint64_t dataCount);
extern std::map<uint32_t, std::map<uint8_t, std::map<uint8_t, std::shared_ptr<TaskStubCcuGraph>>>> g_missionTask;
} // namespace HcclSim

class TaskUtilsTest : public testing::Test {
protected:
    void SetUp() override { g_missionTask.clear(); }
    void TearDown() override { g_missionTask.clear(); }
};

TEST_F(TaskUtilsTest, GetLinkProto_Roce)
{
    EXPECT_EQ(GetLinkProto(CommProtocol::COMM_PROTOCOL_ROCE), LinkProtoStub::RDMA);
}

TEST_F(TaskUtilsTest, GetLinkProto_Other)
{
    EXPECT_EQ(GetLinkProto(CommProtocol::COMM_PROTOCOL_HCCS), LinkProtoStub::SDMA);
    EXPECT_EQ(GetLinkProto(0), LinkProtoStub::SDMA);
    EXPECT_EQ(GetLinkProto(99), LinkProtoStub::SDMA);
}

TEST_F(TaskUtilsTest, CalcDataSize_Int8)
{
    uint64_t sz = CalcDataSize(HCCL_DATA_TYPE_INT8, 10);
    EXPECT_EQ(sz, 10);
}

TEST_F(TaskUtilsTest, CalcDataSize_Int16)
{
    uint64_t sz = CalcDataSize(HCCL_DATA_TYPE_INT16, 10);
    EXPECT_EQ(sz, 20);
}

TEST_F(TaskUtilsTest, CalcDataSize_Int32)
{
    uint64_t sz = CalcDataSize(HCCL_DATA_TYPE_INT32, 10);
    EXPECT_EQ(sz, 40);
}

TEST_F(TaskUtilsTest, CalcDataSize_Int64)
{
    uint64_t sz = CalcDataSize(HCCL_DATA_TYPE_INT64, 10);
    EXPECT_EQ(sz, 80);
}

TEST_F(TaskUtilsTest, CalcDataSize_Fp32)
{
    uint64_t sz = CalcDataSize(HCCL_DATA_TYPE_FP32, 5);
    EXPECT_EQ(sz, 20);
}

TEST_F(TaskUtilsTest, CalcDataSize_Fp16)
{
    uint64_t sz = CalcDataSize(HCCL_DATA_TYPE_FP16, 8);
    EXPECT_EQ(sz, 16);
}

TEST_F(TaskUtilsTest, CalcDataSize_Bf16)
{
    uint64_t sz = CalcDataSize(HCCL_DATA_TYPE_BFP16, 4);
    EXPECT_EQ(sz, 8);
}

TEST_F(TaskUtilsTest, CalcDataSize_ZeroCount)
{
    uint64_t sz = CalcDataSize(HCCL_DATA_TYPE_FP32, 0);
    EXPECT_EQ(sz, 0);
}

TEST_F(TaskUtilsTest, CalcDataSize_InvalidType)
{
    uint64_t sz = CalcDataSize(HCCL_DATA_TYPE_RESERVED, 10);
    EXPECT_EQ(sz, 0);
}

TEST_F(TaskUtilsTest, ConvertTask_NotifyWait_Local)
{
    HcclSim::StorageManager& storage = HcclSim::StorageManager::GetInstance();
    HcclTaskMetaData task{};
    task.taskType = HccLTaskMetaType::NOTIFY_WAIT;
    task.rankId = 0;
    task.taskData.notify.srcRankId = 0;
    task.taskData.notify.notifyId = 100;
    std::shared_ptr<TaskStub> result;
    ASSERT_EQ(ConvertTask(storage, task, result), HCCL_SUCCESS);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->GetType(), TaskTypeStub::LOCAL_WAIT_FROM);
}

TEST_F(TaskUtilsTest, ConvertTask_NotifyWait_Remote)
{
    HcclSim::StorageManager& storage = HcclSim::StorageManager::GetInstance();
    HcclTaskMetaData task{};
    task.taskType = HccLTaskMetaType::NOTIFY_WAIT;
    task.rankId = 0;
    task.taskData.notify.srcRankId = 1;
    task.taskData.notify.notifyId = 200;
    task.taskData.notify.protocol = CommProtocol::COMM_PROTOCOL_ROCE;
    std::shared_ptr<TaskStub> result;
    ASSERT_EQ(ConvertTask(storage, task, result), HCCL_SUCCESS);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->GetType(), TaskTypeStub::WAIT);
    EXPECT_EQ(result->GetLinkType(), LinkProtoStub::RDMA);
}

TEST_F(TaskUtilsTest, ConvertTask_NotifyWait_Remote_SDMA)
{
    HcclSim::StorageManager& storage = HcclSim::StorageManager::GetInstance();
    HcclTaskMetaData task{};
    task.taskType = HccLTaskMetaType::NOTIFY_WAIT;
    task.rankId = 0;
    task.taskData.notify.srcRankId = 2;
    task.taskData.notify.notifyId = 300;
    task.taskData.notify.protocol = CommProtocol::COMM_PROTOCOL_HCCS;
    std::shared_ptr<TaskStub> result;
    ASSERT_EQ(ConvertTask(storage, task, result), HCCL_SUCCESS);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->GetType(), TaskTypeStub::WAIT);
    EXPECT_EQ(result->GetLinkType(), LinkProtoStub::SDMA);
}

TEST_F(TaskUtilsTest, ConvertTask_NotifyRecord_Local)
{
    HcclSim::StorageManager& storage = HcclSim::StorageManager::GetInstance();
    HcclTaskMetaData task{};
    task.taskType = HccLTaskMetaType::NOTIFY_RECORD;
    task.rankId = 0;
    task.taskData.notify.dstRankId = 0;
    task.taskData.notify.notifyId = 400;
    std::shared_ptr<TaskStub> result;
    ASSERT_EQ(ConvertTask(storage, task, result), HCCL_SUCCESS);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->GetType(), TaskTypeStub::LOCAL_POST_TO);
}

TEST_F(TaskUtilsTest, ConvertTask_NotifyRecord_Remote)
{
    HcclSim::StorageManager& storage = HcclSim::StorageManager::GetInstance();
    HcclTaskMetaData task{};
    task.taskType = HccLTaskMetaType::NOTIFY_RECORD;
    task.rankId = 0;
    task.taskData.notify.dstRankId = 1;
    task.taskData.notify.notifyId = 500;
    task.taskData.notify.protocol = CommProtocol::COMM_PROTOCOL_ROCE;
    std::shared_ptr<TaskStub> result;
    ASSERT_EQ(ConvertTask(storage, task, result), HCCL_SUCCESS);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->GetType(), TaskTypeStub::POST);
    EXPECT_EQ(result->GetLinkType(), LinkProtoStub::RDMA);
}

TEST_F(TaskUtilsTest, ConvertTask_MemCpy_Local)
{
    HcclSim::StorageManager& storage = HcclSim::StorageManager::GetInstance();
    HcclTaskMetaData task{};
    task.taskType = HccLTaskMetaType::MEM_CPY;
    task.rankId = 0;
    task.taskData.transMem.srcRankId = 0;
    task.taskData.transMem.dstRankId = 0;
    task.taskData.transMem.srcOffset = 0x1000;
    task.taskData.transMem.dstOffset = 0x2000;
    task.taskData.transMem.len = 1024;
    std::shared_ptr<TaskStub> result;
    ASSERT_EQ(ConvertTask(storage, task, result), HCCL_SUCCESS);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->GetType(), TaskTypeStub::LOCAL_COPY);
}

TEST_F(TaskUtilsTest, ConvertTask_MemCpy_Write)
{
    HcclSim::StorageManager& storage = HcclSim::StorageManager::GetInstance();
    HcclTaskMetaData task{};
    task.taskType = HccLTaskMetaType::MEM_CPY;
    task.rankId = 0;
    task.taskData.transMem.srcRankId = 0;
    task.taskData.transMem.dstRankId = 1;
    task.taskData.transMem.srcOffset = 0x1000;
    task.taskData.transMem.dstOffset = 0x2000;
    task.taskData.transMem.len = 512;
    task.taskData.transMem.protocol = CommProtocol::COMM_PROTOCOL_ROCE;
    std::shared_ptr<TaskStub> result;
    ASSERT_EQ(ConvertTask(storage, task, result), HCCL_SUCCESS);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->GetType(), TaskTypeStub::WRITE);
    EXPECT_EQ(result->GetLinkType(), LinkProtoStub::RDMA);
}

TEST_F(TaskUtilsTest, ConvertTask_MemCpy_Read)
{
    HcclSim::StorageManager& storage = HcclSim::StorageManager::GetInstance();
    HcclTaskMetaData task{};
    task.taskType = HccLTaskMetaType::MEM_CPY;
    task.rankId = 1;
    task.taskData.transMem.srcRankId = 0;
    task.taskData.transMem.dstRankId = 1;
    task.taskData.transMem.srcOffset = 0x1000;
    task.taskData.transMem.dstOffset = 0x2000;
    task.taskData.transMem.len = 256;
    task.taskData.transMem.protocol = CommProtocol::COMM_PROTOCOL_HCCS;
    std::shared_ptr<TaskStub> result;
    ASSERT_EQ(ConvertTask(storage, task, result), HCCL_SUCCESS);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->GetType(), TaskTypeStub::READ);
    EXPECT_EQ(result->GetLinkType(), LinkProtoStub::SDMA);
}

TEST_F(TaskUtilsTest, ConvertTask_MemCpy_NoMatch)
{
    HcclSim::StorageManager& storage = HcclSim::StorageManager::GetInstance();
    HcclTaskMetaData task{};
    task.taskType = HccLTaskMetaType::MEM_CPY;
    task.rankId = 2;
    task.taskData.transMem.srcRankId = 0;
    task.taskData.transMem.dstRankId = 1;
    task.taskData.transMem.srcOffset = 0x1000;
    task.taskData.transMem.dstOffset = 0x2000;
    task.taskData.transMem.len = 128;
    std::shared_ptr<TaskStub> result;
    ASSERT_EQ(ConvertTask(storage, task, result), HCCL_SUCCESS);
    EXPECT_EQ(result, nullptr);
}

TEST_F(TaskUtilsTest, ConvertTask_Reduce_Local)
{
    HcclSim::StorageManager& storage = HcclSim::StorageManager::GetInstance();
    HcclTaskMetaData task{};
    task.taskType = HccLTaskMetaType::REDUCE;
    task.rankId = 0;
    task.taskData.reduce.srcRankId = 0;
    task.taskData.reduce.dstRankId = 0;
    task.taskData.reduce.srcOffset = 0x1000;
    task.taskData.reduce.dstOffset = 0x2000;
    task.taskData.reduce.dataCount = 100;
    task.taskData.reduce.dataType = HCCL_DATA_TYPE_FP32;
    task.taskData.reduce.reduceOp = HCCL_REDUCE_SUM;
    std::shared_ptr<TaskStub> result;
    ASSERT_EQ(ConvertTask(storage, task, result), HCCL_SUCCESS);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->GetType(), TaskTypeStub::LOCAL_REDUCE);
}

TEST_F(TaskUtilsTest, ConvertTask_Reduce_Write)
{
    HcclSim::StorageManager& storage = HcclSim::StorageManager::GetInstance();
    HcclTaskMetaData task{};
    task.taskType = HccLTaskMetaType::REDUCE;
    task.rankId = 0;
    task.taskData.reduce.srcRankId = 0;
    task.taskData.reduce.dstRankId = 1;
    task.taskData.reduce.srcOffset = 0x1000;
    task.taskData.reduce.dstOffset = 0x2000;
    task.taskData.reduce.dataCount = 50;
    task.taskData.reduce.dataType = HCCL_DATA_TYPE_FP32;
    task.taskData.reduce.reduceOp = HCCL_REDUCE_SUM;
    task.taskData.transMem.protocol = CommProtocol::COMM_PROTOCOL_ROCE;
    std::shared_ptr<TaskStub> result;
    ASSERT_EQ(ConvertTask(storage, task, result), HCCL_SUCCESS);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->GetType(), TaskTypeStub::WRITE_REDUCE);
    EXPECT_EQ(result->GetLinkType(), LinkProtoStub::RDMA);
}

TEST_F(TaskUtilsTest, ConvertTask_Reduce_Read)
{
    HcclSim::StorageManager& storage = HcclSim::StorageManager::GetInstance();
    HcclTaskMetaData task{};
    task.taskType = HccLTaskMetaType::REDUCE;
    task.rankId = 1;
    task.taskData.reduce.srcRankId = 0;
    task.taskData.reduce.dstRankId = 1;
    task.taskData.reduce.srcOffset = 0x1000;
    task.taskData.reduce.dstOffset = 0x2000;
    task.taskData.reduce.dataCount = 25;
    task.taskData.reduce.dataType = HCCL_DATA_TYPE_FP32;
    task.taskData.reduce.reduceOp = HCCL_REDUCE_SUM;
    task.taskData.transMem.protocol = CommProtocol::COMM_PROTOCOL_HCCS;
    std::shared_ptr<TaskStub> result;
    ASSERT_EQ(ConvertTask(storage, task, result), HCCL_SUCCESS);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->GetType(), TaskTypeStub::READ_REDUCE);
    EXPECT_EQ(result->GetLinkType(), LinkProtoStub::SDMA);
}

TEST_F(TaskUtilsTest, ConvertTask_Reduce_NoMatch)
{
    HcclSim::StorageManager& storage = HcclSim::StorageManager::GetInstance();
    HcclTaskMetaData task{};
    task.taskType = HccLTaskMetaType::REDUCE;
    task.rankId = 2;
    task.taskData.reduce.srcRankId = 0;
    task.taskData.reduce.dstRankId = 1;
    task.taskData.reduce.srcOffset = 0x1000;
    task.taskData.reduce.dstOffset = 0x2000;
    task.taskData.reduce.dataCount = 10;
    task.taskData.reduce.dataType = HCCL_DATA_TYPE_FP32;
    task.taskData.reduce.reduceOp = HCCL_REDUCE_SUM;
    std::shared_ptr<TaskStub> result;
    ASSERT_EQ(ConvertTask(storage, task, result), HCCL_SUCCESS);
    EXPECT_EQ(result, nullptr);
}

TEST_F(TaskUtilsTest, ConvertTask_DefaultUnknown)
{
    HcclSim::StorageManager& storage = HcclSim::StorageManager::GetInstance();
    HcclTaskMetaData task{};
    task.taskType = static_cast<HccLTaskMetaType>(99);
    task.rankId = 0;
    std::shared_ptr<TaskStub> result;
    ASSERT_EQ(ConvertTask(storage, task, result), HCCL_SUCCESS);
    EXPECT_EQ(result, nullptr);
}

TEST_F(TaskUtilsTest, ConvertTaskQueue_Empty)
{
    AllRankTaskQueues allRankTaskQueues;
    EXPECT_EQ(ConvertTaskQueue(allRankTaskQueues), HCCL_SUCCESS);
}

TEST_F(TaskUtilsTest, ConvertTask_NotifyWait_Remote_SDMA_MultipleRanks)
{
    HcclSim::StorageManager& storage = HcclSim::StorageManager::GetInstance();
    for (uint32_t rank = 0; rank < 3; rank++) {
        HcclTaskMetaData task{};
        task.taskType = HccLTaskMetaType::NOTIFY_WAIT;
        task.rankId = rank;
        task.taskData.notify.srcRankId = rank + 1;
        task.taskData.notify.notifyId = rank * 100;
        task.taskData.notify.protocol = CommProtocol::COMM_PROTOCOL_HCCS;
        std::shared_ptr<TaskStub> result;
        ASSERT_EQ(ConvertTask(storage, task, result), HCCL_SUCCESS);
        ASSERT_NE(result, nullptr);
        EXPECT_EQ(result->GetType(), TaskTypeStub::WAIT);
    }
}

TEST_F(TaskUtilsTest, ConvertTask_MemCpy_Write_ROCE)
{
    HcclSim::StorageManager& storage = HcclSim::StorageManager::GetInstance();
    HcclTaskMetaData task{};
    task.taskType = HccLTaskMetaType::MEM_CPY;
    task.rankId = 3;
    task.taskData.transMem.srcRankId = 3;
    task.taskData.transMem.dstRankId = 5;
    task.taskData.transMem.srcOffset = 0;
    task.taskData.transMem.dstOffset = 0;
    task.taskData.transMem.len = 4096;
    task.taskData.transMem.protocol = CommProtocol::COMM_PROTOCOL_ROCE;
    std::shared_ptr<TaskStub> result;
    ASSERT_EQ(ConvertTask(storage, task, result), HCCL_SUCCESS);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->GetType(), TaskTypeStub::WRITE);
    EXPECT_EQ(result->GetLinkType(), LinkProtoStub::RDMA);
}

TEST_F(TaskUtilsTest, ConvertTask_MemCpy_Read_ROCE)
{
    HcclSim::StorageManager& storage = HcclSim::StorageManager::GetInstance();
    HcclTaskMetaData task{};
    task.taskType = HccLTaskMetaType::MEM_CPY;
    task.rankId = 5;
    task.taskData.transMem.srcRankId = 3;
    task.taskData.transMem.dstRankId = 5;
    task.taskData.transMem.srcOffset = 0;
    task.taskData.transMem.dstOffset = 0;
    task.taskData.transMem.len = 2048;
    task.taskData.transMem.protocol = CommProtocol::COMM_PROTOCOL_ROCE;
    std::shared_ptr<TaskStub> result;
    ASSERT_EQ(ConvertTask(storage, task, result), HCCL_SUCCESS);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->GetType(), TaskTypeStub::READ);
    EXPECT_EQ(result->GetLinkType(), LinkProtoStub::RDMA);
}

TEST_F(TaskUtilsTest, ConvertTask_NotifyRecord_Remote_SDMA)
{
    HcclSim::StorageManager& storage = HcclSim::StorageManager::GetInstance();
    HcclTaskMetaData task{};
    task.taskType = HccLTaskMetaType::NOTIFY_RECORD;
    task.rankId = 0;
    task.taskData.notify.dstRankId = 3;
    task.taskData.notify.notifyId = 600;
    task.taskData.notify.protocol = CommProtocol::COMM_PROTOCOL_HCCS;
    std::shared_ptr<TaskStub> result;
    ASSERT_EQ(ConvertTask(storage, task, result), HCCL_SUCCESS);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->GetType(), TaskTypeStub::POST);
    EXPECT_EQ(result->GetLinkType(), LinkProtoStub::SDMA);
}

TEST_F(TaskUtilsTest, ConvertTask_CCU_Graph_Basic)
{
    HcclSim::StorageManager& storage = HcclSim::StorageManager::GetInstance();
    HcclTaskMetaData task{};
    task.taskType = HccLTaskMetaType::CCU_GRAPH;
    task.rankId = 0;
    task.taskData.ccu.dieId = 0;
    task.taskData.ccu.missionId = 1;
    task.taskData.ccu.instStartId = 10;
    task.taskData.ccu.instCnt = 1;
    task.taskData.ccu.argSize = 0;
    task.taskData.ccu.key = 1;
    task.taskData.ccu.timeout = 5000;
    std::shared_ptr<TaskStub> result;
    ASSERT_EQ(ConvertTask(storage, task, result), HCCL_SUCCESS);
    EXPECT_EQ(result, nullptr);
}

TEST_F(TaskUtilsTest, ConvertTaskQueue_Empty_AlreadyCovered)
{
    AllRankTaskQueues q;
    EXPECT_EQ(ConvertTaskQueue(q), HCCL_SUCCESS);
}

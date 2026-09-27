/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_task_queue_stub.h"

using namespace hccl;

namespace checker {

void AivSingleBlockTaskQues::AppendAivTask(pipe_t pipeId, std::shared_ptr<TaskStub> task)
{
    taskQueues[pipeId].push_back(task);
    return;
}

std::shared_ptr<TaskStub> AivSingleBlockTaskQues::GetTask(pipe_t pipeId, u32 pos) const
{
    return taskQueues.at(pipeId).at(pos);
}

std::vector<std::shared_ptr<TaskStub>> AivSingleBlockTaskQues::GetQueTasks(pipe_t pipeId) const
{
    return taskQueues.at(pipeId);
}

u32 AivSingleBlockTaskQues::GetPipeTaskNum(pipe_t pipeId) const { return taskQueues.at(pipeId).size(); }

void AllAivTaskQueues::Clear()
{
    for (auto& rankPair : rsb2AivTaskQueues) {
        for (auto& blockPair : rankPair.second) {
            for (auto aivSingle : blockPair.second) {
                delete aivSingle;
                aivSingle = nullptr;
            }
        }
    }

    for (auto& task : headAndTailResource) {
        task.reset();
    }
    headAndTailResource.clear();
    rsb2AivTaskQueues.clear();
    pipeBarrierAllRecord.clear();
    aivNumforRank.clear();
    rank2AivTask.clear();
    for (auto& rankPair : copyRank2AivTask) {
        rankPair.second.clear();
    }
    copyRank2AivTask.clear();

    return;
}

void AllAivTaskQueues::AppendAivTask(RankId rankId, BlockId blockId, pipe_t pipeId, std::shared_ptr<TaskStub> task)
{
    if (rsb2AivTaskQueues[rankId].count(blockId) == 0) {
        rsb2AivTaskQueues[rankId][blockId] = std::vector<AivSingleBlockTaskQues*>();
    }

    if (aivNumforRank.find(rankId) == aivNumforRank.end()) {
        aivNumforRank[rankId] = 0;
    }

    if (rsb2AivTaskQueues[rankId][blockId].size() < aivNumforRank[rankId] + 1) {
        rsb2AivTaskQueues[rankId][blockId].resize(aivNumforRank[rankId] + 1);
        rsb2AivTaskQueues[rankId][blockId][aivNumforRank[rankId]] = new AivSingleBlockTaskQues();
    }

    rsb2AivTaskQueues[rankId][blockId][aivNumforRank[rankId]]->AppendAivTask(pipeId, task);
    return;
}

AivSingleBlockTaskQues* AllAivTaskQueues::GetTaskQueueOfAiv(RankId rankId, BlockId blockId, u32 aivTaskIdx) const
{
    return rsb2AivTaskQueues.at(rankId).at(blockId).at(aivTaskIdx);
}

AivTaskQueueStub* AivTaskQueueStub::Global()
{
    static AivTaskQueueStub* aivTaskQueue = new AivTaskQueueStub();
    return aivTaskQueue;
}

AllAivTaskQueues& AivTaskQueueStub::GetAllAivTasks() { return allAivTaskQueues; }

void AivTaskQueueStub::AppendAivTask(RankId rankId, BlockId blockId, pipe_t pipeId, std::shared_ptr<TaskStub> task)
{
    Global()->GetAllAivTasks().AppendAivTask(rankId, blockId, pipeId, task);
}

void AivTaskQueueStub::Reset()
{
    allAivTaskQueues.Clear();
    return;
}

void AivTaskQueueStub::SetRank2AivStart(RankId rankId, TaskNode* aivStart)
{
    Global()->GetAllAivTasks().rank2AivTask[rankId].push_back(aivStart);
}

void AivTaskQueueStub::SetAllCopyAivStart(RankId rankId, TaskNode* aivStart)
{
    Global()->GetAllAivTasks().copyRank2AivTask[rankId].push_back(aivStart);
}

bool AivTaskQueueStub::HasAivTask(RankId rankId)
{
    if (Global()->GetAllAivTasks().hasAivTask.find(rankId) == Global()->GetAllAivTasks().hasAivTask.end()) {
        return false;
    } else {
        return true;
    }
}

void AivTaskQueueStub::PrintAivTask()
{
    for (auto& rankPair : Global()->GetAllAivTasks().rsb2AivTaskQueues) {
        printf("Rankid %d\n", rankPair.first);
        for (auto& blockPair : rankPair.second) {
            printf("\tBlockID %d\n", blockPair.first);
            for (auto blockPos = 0; blockPos < blockPair.second.size(); blockPos++) {
                printf("\t    RankPos %d\n", blockPos);
                auto taskQueues = blockPair.second[blockPos]->taskQueues;
                for (int i = 0; i < taskQueues.size(); i++) {
                    printf("\t\t");
                    switch (i) {
                        case 0:
                            printf("%-6s", "SCALAR");
                            break;
                        case 1:
                            printf("%-6s", "MTE2");
                            break;
                        case 2:
                            printf("%-6s", "MTE3");
                            break;
                        default:
                            printf("unknown pipe");
                    }
                    printf(" : ");
                    for (auto& task : taskQueues[i]) {
                        printf("%s, ", GetTaskName(task->GetType()).c_str());
                        // printf("%s, ", task->Describe().c_str());
                    }
                    printf("\n");
                }
            }
        }
    }
}

void AivTaskQueueStub::AppendAivTaskStubInMainStream(RankId rankId)
{
    Stream stream;
    stream.streamId_ = 0;
    u32 aivTaskIdx = AivTaskQueueStub::Global()->GetAllAivTasks().aivNumforRank[rankId];
    u32 mainStreamPos = 0;

    if (TaskQueueStub::Global()->GetAllRankTasks().rank2TaskQueues.find(rankId)
        == TaskQueueStub::Global()->GetAllRankTasks().rank2TaskQueues.end()) {
        mainStreamPos = 0;
    } else {
        mainStreamPos = TaskQueueStub::Global()->GetAllRankTasks().rank2TaskQueues[rankId]->taskQueues[0].size();
    }

    TaskQueueStub::Global()->AppendTask(
        rankId, &stream, std::make_shared<AivTaskStub>(rankId, aivTaskIdx, mainStreamPos));

    AivTaskQueueStub::Global()->GetAllAivTasks().aivNumforRank[rankId]++;

    auto hasAivTask = AivTaskQueueStub::Global()->GetAllAivTasks().hasAivTask;
    if (hasAivTask.find(rankId) == hasAivTask.end()) {
        hasAivTask.insert(rankId);
    }
}

} // namespace checker

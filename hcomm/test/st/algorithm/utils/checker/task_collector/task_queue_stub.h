/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV1_TASK_QUEUE_STUB_H
#define HCCLV1_TASK_QUEUE_STUB_H

#include "task_stub.h"
#include "stream_pub.h"

#include <string>
#include <map>
#include <vector>

// 1.0需要通过stream来获取所处的task queue
using namespace hccl;
namespace checker {

struct SingleRankTaskQueues {
    // taskQueues[0]: master queue; taskQueues[1:]: slave queues
    std::vector<std::vector<std::shared_ptr<TaskStub>>> taskQueues;

    void AppendTask(Stream* stream, std::shared_ptr<TaskStub> task);

    std::vector<std::shared_ptr<TaskStub>>& operator[](QId queId);

    std::shared_ptr<TaskStub> GetTask(QId queId, u32 pos) const;
    std::vector<std::shared_ptr<TaskStub>> GetQueTasks(QId queId) const;
    u32 GetQueTaskNum(QId queId) const;
};

struct AllRankTaskQueues {
    std::map<RankId, SingleRankTaskQueues*> rank2TaskQueues;
    u32 rankSize = 0;

    void Clear();
    void AppendTask(RankId rankId, Stream* stream, std::shared_ptr<TaskStub> task);

    SingleRankTaskQueues* operator[](RankId rankId);

    SingleRankTaskQueues* GetRankTaskQues(RankId rankId) const;
};

class TaskQueueStub {
public:
    static TaskQueueStub* Global();
    void Reset();

    static void AppendTask(RankId rankId, Stream* stream, std::shared_ptr<TaskStub> task);

    u32 GetRankSize() const;
    SingleRankTaskQueues* GetTaskQueueOfRank(RankId rankId) const;
    AllRankTaskQueues& GetAllRankTasks();

private:
    AllRankTaskQueues allRankTaskQueues;
};

} // namespace checker
#endif

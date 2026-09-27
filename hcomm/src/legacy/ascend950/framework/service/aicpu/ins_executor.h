/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_AICPU_INSTRUCTION_INS_EXECUTOR_H_
#define HCCL_AICPU_INSTRUCTION_INS_EXECUTOR_H_

#include "ins_queue.h"
#include "sqe_mgr.h"
#include "connected_link_mgr.h"
#include "lite_res_mgr_fetcher.h"
#include "kernel_param_lite.h"
#include "profiling_handler_lite.h"

namespace Hccl {

class InsExecutor {
public:
    explicit InsExecutor(ResMgrFetcher* resMgrFetcher);
    void Execute(const InsQueue& insQueue);
    void ExecuteV82(const InsQueue& insQueue, bool isMc2 = false);

private:
    void ExecuteSingleQue(const InsQueue& insQueue, const StreamLite* streamLite, const bool isMaster = false);
    void ReportMainStreamTask(const StreamLite& stream, MainStreamTaskType type) const;

    void ExecuteAllQueues950(const InsQueue& insQueue, StreamLiteMgr* streamLiteMgr);
    void ExecuteSlaveQueue950(
        list<InsQueue::Iterator>& slaveQueueIters, StreamLiteMgr* streamLiteMgr, bool& isLaunchTask,
        std::set<u32>& slaveStreamIndexSet);
    void ExecuteMasterQueue950(
        InsQueue::Iterator& masterQueueIter, StreamLite* masterStream, bool& isMasterInsIterEnd, bool& isLaunchTask);

    void AddOpCounter(const StreamLite& stream, bool isHead) const;
    void CheckPreStreamSync(StreamLiteMgr* streamLiteMgr, u32 slaveQueuesSize);
    std::unique_ptr<SqeMgr> sqeMgr;
    ResMgrFetcher* resMgrFetcher_{nullptr};

    bool isPreStreamSyncExist_{false};
};

} // namespace Hccl

#endif // HCCL_AICPU_INSTRUCTION_INS_EXECUTOR_H_

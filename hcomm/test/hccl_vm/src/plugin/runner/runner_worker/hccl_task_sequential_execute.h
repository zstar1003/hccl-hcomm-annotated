/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_TASK_SEQUENTIAL_EXECUTE_H
#define HCCL_TASK_SEQUENTIAL_EXECUTE_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

#include "sim_common_defs.h"
#include "hccl_task_thread.h"
#include "storage_manager.h"

using namespace HcclSim;

namespace VirtualRunTime {
using workerFunc = std::function<HcclSim::HcclVmResult(const HcclTaskMetaData&)>;

class SqeuentialExecutor {
public:
    SqeuentialExecutor(AllRankTaskQueues& allRankTaskQueues, const std::string& rootPath);
    ~SqeuentialExecutor() = default;

    HcclVmResult Execute();

private:
    HcclVmResult ExecuteOneTask(HcclTaskMetaData& task);
    bool HasTask();

private:
    const static std::map<HccLTaskMetaType, const std::string> taskNames_;

    HcclSim::AllRankTaskQueues allRankTaskQueues_;
    std::string rootPath_;
};
} // namespace VirtualRunTime
#endif

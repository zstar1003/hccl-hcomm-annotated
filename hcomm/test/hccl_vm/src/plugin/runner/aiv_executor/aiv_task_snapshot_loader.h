/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_TASK_SNAPSHOT_LOADER_H
#define AIV_TASK_SNAPSHOT_LOADER_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "aiv_task.h"

struct AivRuntimeBlockTaskSnapshot {
    uint32_t blockIdx{0};
    std::vector<std::shared_ptr<AivSim::AivTask>> scalarTasks;
    std::vector<std::shared_ptr<AivSim::AivTask>> mte2Tasks;
    std::vector<std::shared_ptr<AivSim::AivTask>> mte3Tasks;
};

struct AivRuntimeTaskSnapshot {
    uint32_t rankId{0};
    uint32_t rankSize{0};
    uint32_t launchIndex{0};
    std::string filePath;
    std::vector<AivRuntimeBlockTaskSnapshot> blocks;
};

class AivTaskSnapshotLoader {
public:
    static bool LoadRuntimeTaskSnapshotByLaunchDirect(
        uint32_t rankId, uint32_t launchIndex, AivRuntimeTaskSnapshot& taskSnapshot,
        std::string* errorMessage = nullptr);

private:
    static void SetError(std::string* errorMessage, const std::string& message);
};

bool IsAivExpansionModeEnabled();

#endif // AIV_TASK_SNAPSHOT_LOADER_H

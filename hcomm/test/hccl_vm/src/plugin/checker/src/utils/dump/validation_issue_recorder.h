/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_VM_VALIDATION_ISSUE_RECORDER_H
#define HCCL_VM_VALIDATION_ISSUE_RECORDER_H

#include <mutex>
#include <nlohmann_json/json.hpp>
#include <string>
#include <vector>

#include "hccl_types.h"
#include "task_def.h"

namespace HcclSim {
nlohmann::json DumpIssueTaskNodeToJson(TaskNode* node, const std::string& nodeId = "");

class ValidationIssueRecorder {
public:
    static ValidationIssueRecorder& GetInstance()
    {
        static ValidationIssueRecorder instance;
        return instance;
    }

    ValidationIssueRecorder(const ValidationIssueRecorder&) = delete;
    ValidationIssueRecorder& operator=(const ValidationIssueRecorder&) = delete;

    void Reset();
    void RecordIssue(
        const std::string& severity, const std::string& stage, const std::string& code, const nlohmann::json& detail);
    HcclResult Flush() const;

private:
    ValidationIssueRecorder() = default;

    mutable std::mutex mutex_;
    u32 nextIssueId_ = 1;
    std::vector<nlohmann::json> issues_;
};
} // namespace HcclSim

#endif // HCCL_VM_VALIDATION_ISSUE_RECORDER_H

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INTERPRETER_H
#define HCCLV2_INTERPRETER_H

#include <functional>
#include <unordered_map>
#include <memory>
#include <vector>
#include <map>
#include "ins_queue.h"
#include "task.h"
#include "op_task_config.h"
#include "notify_timeout_cfg.h"
namespace std {
template <>
struct hash<Hccl::InstructionType> {
    std::size_t operator()(const Hccl::InstructionType& type) const noexcept { return static_cast<std::size_t>(type); }
};
} // namespace std
namespace Hccl {

class CommunicatorImpl;

class Interpreter {
public:
    using InsRule = std::function<void(const Instruction&, CommunicatorImpl&, const Stream&, const OpTaskConfig&)>;

    explicit Interpreter(CommunicatorImpl& communicator);

    void Submit(const InsQueue& insQueue);

private:
    CommunicatorImpl& comm;
    std::unordered_map<InstructionType, InsRule> insRuleMap;
    OpTaskConfig taskConfig{};

    void SubmitSlaveQueueAlternatively(list<InsQueue::Iterator>& slaveQueueIters, std::set<u32>& slaveStreamIndexSet);
    void SubmitMasterQueueAlternatively(InsQueue::Iterator& masterQueueIter);
};
} // namespace Hccl

#endif // !HCCLV2_INTERPRETER_H

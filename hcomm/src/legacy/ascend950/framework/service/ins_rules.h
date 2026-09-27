/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INS_RULES_H
#define HCCLV2_INS_RULES_H

#include "communicator_impl.h"
#include "instruction.h"
#include "task.h"
#include "ccu_ins.h"
#include "ccu_ins_group.h"
#include "aicpu_ins.h"
#include "aiv/aiv_ins/aiv_ins.h"

namespace Hccl {

void Interpret(
    const InsLocalPostTo& insLocalPostTo, CommunicatorImpl& comm, const Stream& stream, const OpTaskConfig& taskConfig);

void Interpret(
    const InsLocalWaitFrom& insLocalWaitFrom, CommunicatorImpl& comm, const Stream& stream,
    const OpTaskConfig& taskConfig);

void Interpret(
    const InsLocalWaitGroup& insLocalWaitGroup, CommunicatorImpl& comm, const Stream& stream,
    const OpTaskConfig& taskConfig);

void Interpret(
    const InsLocalBcastPost& insLocalBcastPost, CommunicatorImpl& comm, const Stream& stream,
    const OpTaskConfig& taskConfig);

void Interpret(
    const InsLocalCopy& insLocalCopy, CommunicatorImpl& comm, const Stream& stream, const OpTaskConfig& taskConfig);

void Interpret(
    const InsLocalReduce& insLocalReduce, CommunicatorImpl& comm, const Stream& stream, const OpTaskConfig& taskConfig);

void Interpret(
    const InsPostReady& insPostReady, CommunicatorImpl& comm, const Stream& stream, const OpTaskConfig& taskConfig);

void Interpret(
    const InsWaitReady& insWaitReady, CommunicatorImpl& comm, const Stream& stream, const OpTaskConfig& taskConfig);

void Interpret(
    const InsPostFin& insPostFin, CommunicatorImpl& comm, const Stream& stream, const OpTaskConfig& taskConfig);

void Interpret(
    const InsWaitFin& insWaitFin, CommunicatorImpl& comm, const Stream& stream, const OpTaskConfig& taskConfig);

void Interpret(
    const InsWaitGroupFin& insWaitGroupFin, CommunicatorImpl& comm, const Stream& stream,
    const OpTaskConfig& taskConfig);

void Interpret(
    const InsPostFinAck& insPostFinAck, CommunicatorImpl& comm, const Stream& stream, const OpTaskConfig& taskConfig);

void Interpret(
    const InsWaitFinAck& insWaitFinAck, CommunicatorImpl& comm, const Stream& stream, const OpTaskConfig& taskConfig);

void Interpret(const InsRead& insRead, CommunicatorImpl& comm, const Stream& stream, const OpTaskConfig& taskConfig);

void Interpret(const InsWrite& insWrite, CommunicatorImpl& comm, const Stream& stream, const OpTaskConfig& taskConfig);

void Interpret(
    const InsReadReduce& insReadReduce, CommunicatorImpl& comm, const Stream& stream, const OpTaskConfig& taskConfig);

void Interpret(
    const InsWriteReduce& insWriteReduce, CommunicatorImpl& comm, const Stream& stream, const OpTaskConfig& taskConfig);

void Interpret(
    const InsWriteWithFin& insWriteWithFin, CommunicatorImpl& comm, const Stream& stream,
    const OpTaskConfig& taskConfig);

void Interpret(
    const InsWriteReduceWithFin& insWriteReduceWithFin, CommunicatorImpl& comm, const Stream& stream,
    const OpTaskConfig& taskConfig);

void Interpret(
    const CcuInstruction& ccuInstruction, CommunicatorImpl& comm, const Stream& stream, const OpTaskConfig& taskConfig);

void Interpret(
    const AicpuInstruction& aicpuInstruction, CommunicatorImpl& comm, const Stream& stream,
    const OpTaskConfig& taskConfig);

void Interpret(
    const AivInstruction& aivInstruction, const CommunicatorImpl& comm, const Stream& stream,
    const OpTaskConfig& taskConfig);

template <typename INS_TYPE>
Interpreter::InsRule GetInsRule()
{
    return [](const Instruction& ins, CommunicatorImpl& comm, const Stream& stream, const OpTaskConfig& taskConfig) {
        Interpret(static_cast<const INS_TYPE&>(ins), comm, stream, taskConfig);
    };
}

} // namespace Hccl

#endif // HCCLV2_INS_RULES_H

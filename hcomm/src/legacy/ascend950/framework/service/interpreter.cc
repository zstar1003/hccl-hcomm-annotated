/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <memory>

#include "interpreter.h"
#include "ins_rules.h"
#include "aiv_ins.h"

namespace Hccl {
Interpreter::Interpreter(CommunicatorImpl& communicator)
    : comm(communicator),
      insRuleMap(
          {{InstructionType::LOCAL_COPY, GetInsRule<InsLocalCopy>()},
           {InstructionType::LOCAL_REDUCE, GetInsRule<InsLocalReduce>()},
           {InstructionType::LOCAL_POST_TO, GetInsRule<InsLocalPostTo>()},
           {InstructionType::LOCAL_WAIT_FROM, GetInsRule<InsLocalWaitFrom>()},
           {InstructionType::LOCAL_WAIT_GROUP, GetInsRule<InsLocalWaitGroup>()},
           {InstructionType::LOCAL_BCAST_POST, GetInsRule<InsLocalBcastPost>()},
           {InstructionType::POST_READY, GetInsRule<InsPostReady>()},
           {InstructionType::WAIT_READY, GetInsRule<InsWaitReady>()},
           {InstructionType::POST_FIN, GetInsRule<InsPostFin>()},
           {InstructionType::WAIT_FIN, GetInsRule<InsWaitFin>()},
           {InstructionType::WAIT_GROUP_FIN, GetInsRule<InsWaitGroupFin>()},
           {InstructionType::POST_FIN_ACK, GetInsRule<InsPostFinAck>()},
           {InstructionType::WAIT_FIN_ACK, GetInsRule<InsWaitFinAck>()},
           {InstructionType::WRITE_WITH_FIN, GetInsRule<InsWriteWithFin>()},
           {InstructionType::WRITE_REDUCE_WITH_FIN, GetInsRule<InsWriteReduceWithFin>()},
           {InstructionType::WRITE, GetInsRule<InsWrite>()},
           {InstructionType::WRITE_REDUCE, GetInsRule<InsWriteReduce>()},
           {InstructionType::READ, GetInsRule<InsRead>()},
           {InstructionType::READ_REDUCE, GetInsRule<InsReadReduce>()},
           {InstructionType::CCU_INS, GetInsRule<CcuInstruction>()},
           {InstructionType::AICPU_INS, GetInsRule<AicpuInstruction>()},
           {InstructionType::AIV_INS, GetInsRule<AivInstruction>()}})
{
    if (communicator.GetCurrentCollOperator()->opType == OpType::BARRIER) {
        taskConfig.SetNotifyWaitTime(communicator.GetNotifyTimeoutCfg().GetBarrierTimeout());
    } else {
        taskConfig.SetNotifyWaitTime(communicator.GetNotifyTimeoutCfg().GetNotifyTimeout());
    }
}

void Interpreter::Submit(const InsQueue& insQueue)
{
    list<InsQueue::Iterator> slaveQueueIters;
    for (auto slaveQueueIter = insQueue.IterSlaves(); slaveQueueIter.HasNext(); ++slaveQueueIter) {
        slaveQueueIters.emplace_back((*slaveQueueIter).Iter());
    }

    std::set<u32> slaveStreamIndexSet;
    for (u32 slaveStreamIndex = 0; slaveStreamIndex < slaveQueueIters.size(); ++slaveStreamIndex) {
        slaveStreamIndexSet.insert(slaveStreamIndex);
    }

    // 获取指令规模，填充桶宽及初始化
    auto& streamMgr = comm.GetStreamManager();
    auto masterStream = streamMgr.GetMaster();
    streamMgr.InitBucket(UINT32_MAX);
    streamMgr.RecordStreamIdToIndex(masterStream->GetId(), UINT32_MAX);
    u32 index = 0;
    for (auto slaveIter = insQueue.IterSlaves(); slaveIter.HasNext(); ++slaveIter) {
        streamMgr.InitBucket(streamMgr.GetSlaveIndex());
        auto slaveStream = streamMgr.GetSlave();
        streamMgr.RecordStreamIdToIndex(slaveStream->GetId(), index);
        streamMgr.CaptureSlaveStream(masterStream, slaveStream);
    }
    InsQueue::Iterator masterQueueIter = insQueue.Iter();

    // 交替流下Task，直至全部下完
    while (!slaveQueueIters.empty() || masterQueueIter.HasNext()) {
        SubmitSlaveQueueAlternatively(slaveQueueIters, slaveStreamIndexSet);
        SubmitMasterQueueAlternatively(masterQueueIter);
    }

    // 销毁桶
    streamMgr.DestroyRecords();
    // 销毁流的占用状态
    streamMgr.ResetSlaveIndex(0);
}

void Interpreter::SubmitSlaveQueueAlternatively(
    list<InsQueue::Iterator>& slaveQueueIters, std::set<u32>& slaveStreamIndexSet)
{
    auto& streamMgr = comm.GetStreamManager();
    auto slaveStreamIndexIter = slaveStreamIndexSet.begin();
    for (auto slaveQueueIter = slaveQueueIters.begin(); slaveQueueIter != slaveQueueIters.end();) {
        if (!slaveQueueIter->HasNext()) {
            // 销毁已经完成下发的流迭代器
            HCCL_INFO(
                "[SubmitSlaveQueueAlternatively] slave stream index(%u) interpret finish", (*slaveStreamIndexIter));
            slaveQueueIter = slaveQueueIters.erase(slaveQueueIter);
            slaveStreamIndexIter = slaveStreamIndexSet.erase(slaveStreamIndexIter);
            continue;
        }
        auto& rule = insRuleMap.at((*slaveQueueIter)->GetType());
        auto stream = streamMgr.GetSlaveByIndex(*slaveStreamIndexIter);
        rule(**slaveQueueIter, comm, *stream, taskConfig);

        HCCL_INFO(
            "[SubmitSlaveQueueAlternatively] slave stream index[%u], stream id[%u]. Instruction start %s",
            (*slaveStreamIndexIter), stream->GetId(), (*slaveQueueIter)->Describe().c_str());
        // 当前Task下载完毕，跳转这条流上的InsQueue里下一个task
        ++(*slaveQueueIter);
        // 切换至下一条流
        ++slaveQueueIter;
        ++slaveStreamIndexIter;
    }
}

void Interpreter::SubmitMasterQueueAlternatively(InsQueue::Iterator& masterQueueIter)
{
    if (!masterQueueIter.HasNext()) {
        HCCL_INFO("[SubmitMasterQueueAlternatively] main stream interpret finish");
        return;
    }
    auto& streamMgr = comm.GetStreamManager();
    auto stream = streamMgr.GetMaster();
    auto& rule = insRuleMap.at(masterQueueIter->GetType());
    rule(*masterQueueIter, comm, *stream, taskConfig);
    HCCL_INFO(
        "[SubmitMasterQueueAlternatively] master stream id[%u]. Instruction start %s", stream->GetId(),
        masterQueueIter->Describe().c_str());
    // 切换至下一个task
    ++masterQueueIter;
}

} // namespace Hccl

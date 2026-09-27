/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TASK_INFO_H
#define TASK_INFO_H
#include <string>
#include <memory>
#include <functional>
#include "task_param.h"
#include "coll_operator.h"

namespace Hccl {

class DfxOpInfo {
public:
    CollOperator op_;
    std::string tag_; // 实际是opType
    std::string algType_;
    u32 commIndex_;
    u64 beginTime_;
    u64 endTime_;
    void* comm_{nullptr};
    bool isIndop_{false};
    std::string algTag_;
    std::string groupName_;
    u64 groupNameHash_{0}; // 通信域 groupName 哈希值，由 HcclCommDfx::SetCurrDfxOpInfo 填入
    u32 rankSize_{0};
    u32 cpuWaitAicpuNotifyId_{0};
    std::string commId_{};
    u32 opIndex_{0};           // 下发算子总计数(单算子/图模式/CCU快速下发)
    u64 headOpCounterAddr_{0}; // 执行算子头计数器地址（执行前加一）
    u64 tailOpCounterAddr_{0}; // 执行算子尾计数器地址（执行后加一）
    CommEngine engine{CommEngine::COMM_ENGINE_RESERVED};

public:
    std::string Describe() const
    {
        return StringFormat(
            "DfxOpInfo: [collOperator:[%s], tag:[%s], algType:[%s], commIndex:[%u], commId[%s], "
            "beginTime:[%llu], endTime:[%llu], opIndex[%u], headOpCounterAddr[%llx], "
            "tailOpCounterAddr[%llx], groupName[%s], rankSize[%u], myRank[%d], dataType[%s]]",
            CollOpToString(op_).c_str(), tag_.c_str(), algType_.c_str(), commIndex_, commId_.c_str(), beginTime_,
            endTime_, opIndex_, headOpCounterAddr_, tailOpCounterAddr_, groupName_.c_str(), rankSize_, op_.myRank,
            DataTypeToSerialString(static_cast<uint32_t>(op_.dataType)).c_str());
    }
};

class TaskInfo {
public:
    u32 streamId_;
    u32 taskId_;
    u32 remoteRank_{0xffffffff};
    TaskParam taskParam_;
    std::shared_ptr<DfxOpInfo> dfxOpInfo_;
    bool isMaster_;
    u64 channelHandle_{INVALID_U64};
    std::function<u32(u64)> getRemoteRankByHandle_{nullptr};

public:
    TaskInfo(
        u32 streamId, u32 taskId, u32 remoteRank, const TaskParam& taskParam,
        const std::shared_ptr<DfxOpInfo>& dfxOpInfo = nullptr, bool isMaster = false);

    std::string Describe() const;

    std::string GetAlgTypeName() const;
    std::string GetBaseInfo() const;
    std::string GetConciseBaseInfo() const;
    std::string GetParaInfo() const;
    std::string GetOpInfo() const;

    std::string GetIndopDataInfo() const;
    std::string GetIndopBaseInfo() const;
    u32 GetRemoteRankId() const;

private:
    std::string GetParaDMA() const;
    std::string GetParaReduce() const;
    std::string GetParaNotify() const;
    std::string GetParaAiv() const;
    std::string GetRemoteRankInfo(bool needConcise = false) const;
    std::string GetTaskConciseName() const;
    std::string GetNotifyInfo() const;
    std::string GetReduceOp() const;
};

void PrintTaskLog(u32 streamId, u32 taskId, const TaskParam& taskParam, u32 remoteRankId);

} // namespace Hccl

#endif

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "local_notify.h"

#include "task_stub.h"
#include "task_queue_stub.h"
#include "rank_info_recorder.h"

using namespace checker;
namespace hccl {

u32 LocalNotify::GetNotifyIdx() { return notifyidx_; }

HcclResult LocalNotify::SetNotifyId(u32 notifyId)
{
    notifyidx_ = notifyId;
    return HCCL_SUCCESS;
}

HcclResult LocalNotify::Wait(
    Stream& stream, HcclDispatcher dispatcherPtr, const std::shared_ptr<LocalNotify>& notify, s32 stage, u32 timeOut)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    u32 notifyId = notify->GetNotifyIdx();
    std::shared_ptr<TaskStub> task(new TaskStubLocalWaitFrom(notifyId));
    TaskQueueStub::AppendTask(curRank, &stream, task);

    return HCCL_SUCCESS;
}

HcclResult LocalNotify::Wait(Stream& stream, HcclDispatcher dispatcher, s32 stage, u32 timeOut)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    u32 notifyId = this->GetNotifyIdx();
    std::shared_ptr<TaskStub> task(new TaskStubLocalWaitFrom(notifyId));
    TaskQueueStub::AppendTask(curRank, &stream, task);

    return HCCL_SUCCESS;
}

HcclResult
LocalNotify::Post(Stream& stream, HcclDispatcher dispatcherPtr, const std::shared_ptr<LocalNotify>& notify, s32 stage)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    u32 notifyId = notify->GetNotifyIdx();
    std::shared_ptr<TaskStub> task(new TaskStubLocalPostTo(notifyId));
    TaskQueueStub::AppendTask(curRank, &stream, task);

    return HCCL_SUCCESS;
}

HcclResult LocalNotify::Post(Stream& stream, HcclDispatcher dispatcherPtr, s32 stage)
{
    RankId curRank = RankInfoRecorder::Global()->GetRankId();
    u32 notifyId = this->GetNotifyIdx();
    std::shared_ptr<TaskStub> task(new TaskStubLocalPostTo(notifyId));
    TaskQueueStub::AppendTask(curRank, &stream, task);

    return HCCL_SUCCESS;
}

HcclResult Destroy() { return HCCL_SUCCESS; }

HcclResult SetIpc() { return HCCL_SUCCESS; }
} // namespace hccl

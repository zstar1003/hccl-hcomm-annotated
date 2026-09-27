/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include "ins_to_sqe_rule.h"
#include "null_ptr_exception.h"
#include "ins_executor.h"
#include "communicator_impl_lite_manager.h"

namespace Hccl {

constexpr u64 FOUR_BYTES = 4;
constexpr u32 LAUNCH_PRINT_INTERVAL = 20;

void InsExecutor::Execute(const InsQueue& insQueue)
{
    StreamLiteMgr* streamLiteMgr = resMgrFetcher_->GetStreamLiteMgr();
    int slaveStreamIndex = 0;
    for (auto slaveIter = insQueue.IterSlaves(); slaveIter.HasNext(); ++slaveIter) {
        ExecuteSingleQue(*slaveIter, streamLiteMgr->GetSlave(slaveStreamIndex++));
    }
    CHECK_NULLPTR(streamLiteMgr->GetMaster(), "[Execute]master stream is nullptr!");
    ExecuteSingleQue(insQueue, streamLiteMgr->GetMaster(), true);
}

void InsExecutor::AddOpCounter(const StreamLite& stream, bool isHead) const
{
    CHECK_NULLPTR(resMgrFetcher_, "[InsExecutor::AddOpCounter] resMgrFetcher_ is nullptr!");
    auto rtsq = stream.GetRtsq();
    if (rtsq == nullptr) {
        HCCL_ERROR("[InsExecutor::%s] stream.GetRtsq() is nullptr", __func__);
        return;
    }
    u64 counterSrcAddr = resMgrFetcher_->GetCounterAddr();
    if (counterSrcAddr == 0) {
        HCCL_ERROR("InsExecutor::%s counter addr is null.", __func__);
        return;
    }
    u64 dstAddr = isHead == true ? counterSrcAddr + FOUR_BYTES : counterSrcAddr + FOUR_BYTES * 2;
    u64 count = FOUR_BYTES;
    HCCL_INFO("%s AddOpCounter start", __func__);
    auto taskId = rtsq->GetTaskId();
    rtsq->SdmaReduce(counterSrcAddr, dstAddr, count, 0, ReduceIn(DataType::FP32, ReduceOp::SUM));

    TaskParam taskParam{};
    taskParam.taskType = TaskParamType::TASK_REDUCE_INLINE;
    taskParam.beginTime = ProfGetCurCpuTimestamp();
    taskParam.taskPara.Reduce.src = reinterpret_cast<void*>(counterSrcAddr);
    taskParam.taskPara.Reduce.dst = reinterpret_cast<void*>(dstAddr);
    taskParam.taskPara.Reduce.size = count;
    taskParam.taskPara.Reduce.notifyID = INVALID_VALUE_NOTIFYID;
    taskParam.taskPara.Reduce.linkType = DfxLinkType::ONCHIP;
    taskParam.taskPara.Reduce.reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    taskParam.taskPara.Reduce.dataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    auto taskInfo = std::make_unique<TaskInfo>(stream.GetSqId(), taskId, INVALID_VALUE_RANKID, taskParam);
    resMgrFetcher_->GetMirrorTaskMgrLite()->AddTaskInfo(std::move(taskInfo));
}

void InsExecutor::ExecuteV82(const InsQueue& insQueue, bool isMc2)
{
    // InsQueue 非空已经在外部进行了校验
    if (resMgrFetcher_ == nullptr) {
        THROW<NullPtrException>(StringFormat("InsExecutor::%s resMgrFetcher is null, isMc2 %d.", __func__, isMc2));
        return;
    }
    StreamLiteMgr* streamLiteMgr = resMgrFetcher_->GetStreamLiteMgr();
    if (streamLiteMgr == nullptr) {
        THROW<NullPtrException>(StringFormat("InsExecutor::%s streamLiteMgr is null, isMc2 %d.", __func__, isMc2));
        return;
    }
    // 先下主流上的notify wait任务，包括和host同步和op计数任务
    StreamLite* masterStream = streamLiteMgr->GetMaster();
    if (masterStream == nullptr) {
        THROW<NullPtrException>(StringFormat("InsExecutor::%s masterStream is null, isMc2 %d.", __func__, isMc2));
        return;
    }
    ReportMainStreamTask(*masterStream, MainStreamTaskType::HEAD);
    auto deviceWaitNotifyId = resMgrFetcher_->GetHostDeviceSyncNotifyLiteMgr()->GetDeviceWaitNotify()->GetId();
    HCCL_INFO("InsExecutor::%s GetDeviceWaitNotify id %u", __func__, deviceWaitNotifyId);
    if (!isMc2) {
        auto taskId = masterStream->GetRtsq()->GetTaskId();
        masterStream->GetRtsq()->NotifyWait(deviceWaitNotifyId);
        TaskParam taskParam{};
        taskParam.taskType = TaskParamType::TASK_NOTIFY_WAIT;
        taskParam.beginTime = ProfGetCurCpuTimestamp();
        taskParam.taskPara.Notify.notifyID = deviceWaitNotifyId;
        taskParam.taskPara.Notify.value = 1;
        auto taskInfo = std::make_unique<TaskInfo>(masterStream->GetSqId(), taskId, INVALID_VALUE_RANKID, taskParam);
        resMgrFetcher_->GetMirrorTaskMgrLite()->AddTaskInfo(std::move(taskInfo));
    }
    AddOpCounter(*masterStream, true);

    // 将主流和从流上的Task分别下发执行
    ExecuteAllQueues950(insQueue, streamLiteMgr);

    // 下主流上的notify record任务，包括和host同步和op计数任务
    AddOpCounter(*masterStream, false);
    ReportMainStreamTask(*masterStream, MainStreamTaskType::TAIL);
    auto hostWaitNotifyId = resMgrFetcher_->GetHostDeviceSyncNotifyLiteMgr()->GetHostWaitNotify()->GetId();
    HCCL_INFO("InsExecutor::%s GetHostWaitNotify id %u", __func__, hostWaitNotifyId);
    if (!isMc2) {
        auto taskId = masterStream->GetRtsq()->GetTaskId();
        masterStream->GetRtsq()->NotifyRecordLoc(hostWaitNotifyId);
        TaskParam taskParam{};
        taskParam.taskType = TaskParamType::TASK_NOTIFY_RECORD;
        taskParam.beginTime = ProfGetCurCpuTimestamp();
        taskParam.taskPara.Notify.notifyID = hostWaitNotifyId;
        taskParam.taskPara.Notify.value = 1;
        auto taskInfo = std::make_unique<TaskInfo>(masterStream->GetSqId(), taskId, INVALID_VALUE_RANKID, taskParam);
        resMgrFetcher_->GetMirrorTaskMgrLite()->AddTaskInfo(std::move(taskInfo));
    }
    masterStream->GetRtsq()->LaunchTask();
}

void InsExecutor::ReportMainStreamTask(const StreamLite& stream, MainStreamTaskType type) const
{
    FlagTaskInfo flagTaskInfo;
    flagTaskInfo.taskId = stream.GetRtsq()->GetTaskId();
    flagTaskInfo.type = type;
    HCCL_INFO("[%s] TaskInfo taskId %u", __func__, flagTaskInfo.taskId);
    ProfilingHandlerLite::GetInstance().ReportMainStreamTask(flagTaskInfo);
}

void InsExecutor::ExecuteAllQueues950(const InsQueue& insQueue, StreamLiteMgr* streamLiteMgr)
{
    HCCL_INFO("InsExecutor::%s start", __func__);
    list<InsQueue::Iterator> slaveQueueIters;
    std::set<u32> slaveStreamIndexSet;

    bool isMasterInsIterEnd = false;
    // 用于判断一轮下发task过程中，是否有成功下发Task，以及开始记时
    bool isLaunchTask = false;
    auto startTime = std::chrono::steady_clock::now();
    auto timeoutValue = CommunicatorImplLiteMgr::GetInstance().GetEnvConfig().hcclExecTimeout + 20;
    auto timeout = std::chrono::seconds(timeoutValue);
    const std::chrono::seconds printInterval(LAUNCH_PRINT_INTERVAL); // 打印间隔30s
    auto lastPrintTime = std::chrono::steady_clock::now() - printInterval;
    InsQueue::Iterator masterQueueIter = insQueue.Iter();
    StreamLite* masterStream = streamLiteMgr->GetMaster();
    CHK_PRT_THROW(
        masterStream == nullptr, HCCL_ERROR("[InsExecutor::%s] masterStream is null.", __func__), InternalException,
        "masterStream is null");
    // 将准备下发到从流的subInsQueue的迭代器都存到迭代器数组内部
    for (auto slaveQueueIter = insQueue.IterSlaves(); slaveQueueIter.HasNext(); ++slaveQueueIter) {
        slaveQueueIters.emplace_back((*slaveQueueIter).Iter());
    }
    u32 maxSlaveQueuesSize = slaveQueueIters.size();
    // 创建和slaveQueueIter相对应的从流索引容器，用于后续下发任务一一对应
    for (u32 slaveStreamIndex = 0; slaveStreamIndex < slaveQueueIters.size(); ++slaveStreamIndex) {
        slaveStreamIndexSet.insert(slaveStreamIndex);
    }
    // 遍历迭代器数组，一个流上的InsQueue去下一个任务
    while (!slaveQueueIters.empty() || !isMasterInsIterEnd) {
        // 遍历从流InsQueue，每一次下发一个Task
        ExecuteSlaveQueue950(slaveQueueIters, streamLiteMgr, isLaunchTask, slaveStreamIndexSet);
        // 每一次循环，下发一次主流Task
        if (!isMasterInsIterEnd) {
            ExecuteMasterQueue950(masterQueueIter, masterStream, isMasterInsIterEnd, isLaunchTask);
        }

        CheckPreStreamSync(streamLiteMgr, maxSlaveQueuesSize);
        // 如果没有下发任务就开始记录超时时间
        if (isLaunchTask) {
            startTime = std::chrono::steady_clock::now();
        } else if (std::chrono::steady_clock::now() - lastPrintTime >= printInterval) {
            HCCL_INFO("[ExecuteAllQueues950]All Rtsq Queues full, wait for executor");
            lastPrintTime = std::chrono::steady_clock::now();
        }
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            auto msg = StringFormat("[ExecuteAllQueues950]All Rtsq Queues full, timeout[%u s]", timeoutValue);
            HCCL_ERROR("%s", msg.c_str());
            THROW<InternalException>(msg);
        }
    }
    HCCL_INFO("InsExecutor::%s success", __func__);
}

void InsExecutor::ExecuteSlaveQueue950(
    list<InsQueue::Iterator>& slaveQueueIters, StreamLiteMgr* streamLiteMgr, bool& isLaunchTask,
    std::set<u32>& slaveStreamIndexSet)
{
    auto slaveStreamIndexIter = slaveStreamIndexSet.begin();
    isLaunchTask = false;
    for (auto slaveQueueIter = slaveQueueIters.begin(); slaveQueueIter != slaveQueueIters.end();) {
        StreamLite* slaveStream = streamLiteMgr->GetSlave(*slaveStreamIndexIter);
        if (UNLIKELY(slaveStream == nullptr)) {
            THROW<NullPtrException>(StringFormat("InsExecutor::%s slaveStream is null,", __func__));
        }
        if (slaveStream->GetRtsq() == nullptr) {
            THROW<NullPtrException>(StringFormat(
                "InsExecutor::%s GetRtsq returned null for slaveStream Id(%u)", __func__, slaveStream->GetId()));
        }
        // 判断rtsq队列中的空间是否充足
        bool isRtsqQueueSpaceSufficient = slaveStream->GetRtsq()->IsRtsqQueueSpaceSufficient();
        // 判断当前从流是否有Int64类型reduce算子，是否需要等其他流任务下发完成
        bool isPreStreamSync = slaveStream->GetRtsq()->GetPreStreamSyncStatus();
        if (isRtsqQueueSpaceSufficient && !isPreStreamSync) {
            if (slaveQueueIter->HasNext()) {
                HCCL_INFO(
                    "[ExecuteAllQueues950]InsExecutor::%s slave stream InsQueue start %s SqId(%u) stream Id(%u)",
                    __func__, (*slaveQueueIter)->Describe().c_str(), slaveStream->GetSqId(), slaveStream->GetId());
                Interpret(**slaveQueueIter, *slaveStream, resMgrFetcher_);
                // 给迭代器内部Iter指向这条流上的InsQueue里下一个task
                ++(*slaveQueueIter);
                // 将迭代器指向下一条流上的InsQueue
                ++slaveQueueIter;
                // 将对应的流的索引执行下一条流
                ++slaveStreamIndexIter;
            } else {
                // 如果这个InsQueue上没有下一个task了，就擦掉容器内的对应迭代器
                HCCL_INFO(
                    "[ExecuteAllQueues950]InsExecutor::%s slave stream Id(%u) Interpret insQueue finish", __func__,
                    slaveStream->GetId());
                slaveQueueIter = slaveQueueIters.erase(slaveQueueIter);
                // 擦掉对应的从流索引，避免下任务下错从流
                slaveStreamIndexIter = slaveStreamIndexSet.erase(slaveStreamIndexIter);
                slaveStream->GetRtsq()->LaunchTask();
                HCCL_INFO(
                    "[ExecuteAllQueues950]InsExecutor::%s slave stream Id(%u) launch task finish", __func__,
                    slaveStream->GetId());
            }
            isLaunchTask = true;
        } else {
            // Rtsq上位置不足，先跳去下一个
            ++slaveQueueIter;
            // 将对应的流的索引执行下一条流
            ++slaveStreamIndexIter;
        }
        isPreStreamSyncExist_ = slaveStream->GetRtsq()->GetPreStreamSyncStatus() || isPreStreamSyncExist_;
    }
}

void InsExecutor::ExecuteMasterQueue950(
    InsQueue::Iterator& masterQueueIter, StreamLite* masterStream, bool& isMasterInsIterEnd, bool& isLaunchTask)
{
    // 判断rtsq队列中的空间是否充足
    bool isRtsqQueueSpaceSufficient = masterStream->GetRtsq()->IsRtsqQueueSpaceSufficient();
    // 判断当前主流是否有Int64类型reduce算子，是否需要等其他流任务下发完成
    bool isPreStreamSync = masterStream->GetRtsq()->GetPreStreamSyncStatus();
    if (isRtsqQueueSpaceSufficient && !isPreStreamSync) {
        if (masterQueueIter.HasNext()) {
            HCCL_INFO(
                "[ExecuteAllQueues950]InsExecutor::%s master stream InsQueue start %s SqId(%u) stream Id(%u)", __func__,
                masterQueueIter->Describe().c_str(), masterStream->GetSqId(), masterStream->GetId());
            Interpret(*masterQueueIter, *masterStream, resMgrFetcher_);
            ++masterQueueIter;
        } else if (!masterQueueIter.HasNext() && !isMasterInsIterEnd) {
            HCCL_INFO(
                "[ExecuteAllQueues950]InsExecutor::%s master stream Id(%u) Interpret insQueue finish", __func__,
                masterStream->GetId());
            isMasterInsIterEnd = true;
            masterStream->GetRtsq()->LaunchTask();
        }
        isLaunchTask = true;
    }
    isPreStreamSyncExist_ = masterStream->GetRtsq()->GetPreStreamSyncStatus() || isPreStreamSyncExist_;
}

void InsExecutor::CheckPreStreamSync(StreamLiteMgr* streamLiteMgr, u32 slaveQueuesSize)
{
    if (!isPreStreamSyncExist_) {
        return;
    }
    u32 preStreamSyncValue = 0;
    for (u32 slaveStreamIndex = 0; slaveStreamIndex < slaveQueuesSize; ++slaveStreamIndex) {
        StreamLite* slaveStream = streamLiteMgr->GetSlave(slaveStreamIndex);
        if (slaveStream == nullptr) {
            THROW<NullPtrException>(StringFormat("InsExecutor::%s slaveStream is null,", __func__));
        }
        if (slaveStream->GetRtsq() == nullptr) {
            THROW<NullPtrException>(StringFormat(
                "InsExecutor::%s GetRtsq returned null for slaveStream Id(%u)", __func__, slaveStream->GetId()));
        }
        if (slaveStream->GetRtsq()->GetPreStreamSyncStatus()) {
            ++preStreamSyncValue;
        }
    }
    StreamLite* masterStream = streamLiteMgr->GetMaster();
    if (masterStream->GetRtsq()->GetPreStreamSyncStatus()) {
        ++preStreamSyncValue;
    }
    if (preStreamSyncValue == slaveQueuesSize + 1) {
        for (u32 slaveStreamIndex = 0; slaveStreamIndex < slaveQueuesSize; ++slaveStreamIndex) {
            StreamLite* slaveStream = streamLiteMgr->GetSlave(slaveStreamIndex);
            CHK_RET_THROW(
                InternalException, StringFormat("[InsExecutor][%s] SetPreStreamSyncFin", __func__),
                slaveStream->GetRtsq()->SetPreStreamSyncFin());
        }
        masterStream->GetRtsq()->SetPreStreamSyncFin();
        CHK_RET_THROW(
            InternalException, StringFormat("[InsExecutor][%s] SetPreStreamSyncFin", __func__),
            masterStream->GetRtsq()->SetPreStreamSyncFin());
        isPreStreamSyncExist_ = false;
    }
}

void InsExecutor::ExecuteSingleQue(const InsQueue& insQueue, const StreamLite* streamLite, const bool isMaster)
{
    sqeMgr->Begin(streamLite->GetSqId());
    if (isMaster) {
        HcclNotifyWaitSqe waitSqe;
        waitSqe.Config(
            streamLite->GetSqId(), 0, resMgrFetcher_->GetHostDeviceSyncNotifyLiteMgr()->GetDeviceWaitNotify()->GetId());
        sqeMgr->Add(streamLite->GetSqId(), &waitSqe);
    }

    for (auto iter = insQueue.Iter(); iter.HasNext(); ++iter) {
        HCCL_INFO("InsExecutor::%s start %s", __func__, iter->Describe().c_str());
        vector<std::unique_ptr<HcclSqe>> sqeItems = Interpret(*iter, streamLite->GetSqId(), resMgrFetcher_);
        std::for_each(sqeItems.begin(), sqeItems.end(), [streamLite, this](std::unique_ptr<HcclSqe>& sqeItem) {
            sqeMgr->Add(streamLite->GetSqId(), sqeItem.get());
        });
    }

    if (isMaster) {
        HcclNotifyRecordSqe recordSqe;
        recordSqe.Config(
            streamLite->GetSqId(), 0, resMgrFetcher_->GetHostDeviceSyncNotifyLiteMgr()->GetHostWaitNotify()->GetId());
        sqeMgr->Add(streamLite->GetSqId(), &recordSqe);
    }

    sqeMgr->Commit(streamLite->GetSqId());
}

InsExecutor::InsExecutor(ResMgrFetcher* resMgrFetcher) : resMgrFetcher_(resMgrFetcher)
{
    CHECK_NULLPTR(resMgrFetcher, "[InsExecutor] resMgrFetcher is nullptr!");
    sqeMgr = make_unique<SqeMgr>(resMgrFetcher->GetDevPhyId());
}
} // namespace Hccl

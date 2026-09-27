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
#include <mutex>
#include <thread>
#include <vector>

#include "hccl_group.h"

using namespace hccl;

thread_local s32 hcclGroupDepth = 0;
thread_local std::deque<std::shared_ptr<struct hcclAsyncJob>> hcclInitJobs;
thread_local std::vector<HcclComm> hcclGroupCommList;

HcclResult HcclLegacyGroupStart()
{
    hcclGroupDepth++;
    HCCL_INFO("[HcclGroupStart] hcclGroupDepth=[%d]", hcclGroupDepth);
    return HCCL_SUCCESS;
}

namespace hccl {

HcclResult initGroupPlanner(HcclComm comm)
{
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    std::shared_ptr<struct hcclKernelPlanner> planner = hcclComm->planner;
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    hcclComm->GetRankSize(rankSize);
    planner->rankSize = rankSize;
    HCCL_DEBUG("[initGroupPlanner] ranksize: %d", rankSize);

    planner->nTasksP2p = 0;
    planner->nTasksColl = 0;
    return HCCL_SUCCESS;
}

HcclResult taskAppend(HcclComm comm, hcclOpInfo& info)
{
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    std::shared_ptr<struct hcclKernelPlanner> planner = hcclComm->planner;
    if (planner->nTasksP2p == -1) {
        initGroupPlanner(comm);
    }

    HcclResult ret = HCCL_SUCCESS;
    if (info.coll == HcclCMDType::HCCL_CMD_SEND || info.coll == HcclCMDType::HCCL_CMD_RECEIVE) {
        hcclComm->SetGroupMode(true);
        bool isSendOp = (info.coll == HcclCMDType::HCCL_CMD_SEND);
        HcclSendRecvItem item;
        item.sendRecvType = isSendOp ? HcclSendRecvType::HCCL_SEND : HcclSendRecvType::HCCL_RECV;
        item.buf = isSendOp ? info.sendbuff : info.recvbuff;
        item.count = isSendOp ? info.sendCount : info.recvCount;
        item.dataType = isSendOp ? info.sendType : info.recvType;
        item.remoteRank = info.root;

        planner->sendRecvInfo.push_back(item);

        if (planner->sendRecvMainStream == nullptr) { // 用第一条用户流作为主流
            planner->sendRecvMainStream = info.stream;
            HCCL_INFO("[TaskAppend] planner->sendRecvMainStream[%p]", planner->sendRecvMainStream);
        }

        planner->nTasksP2p += 1;
    } else {
        hcclOpInfo task = info;
        planner->collTaskQueue.push_back(task);
        planner->nTasksColl += 1;
        /*记录stream到planner*/
        planner->collStreams.insert(info.stream);
    }

    auto itComm = std::find(hcclGroupCommList.begin(), hcclGroupCommList.end(), comm);
    if (itComm == hcclGroupCommList.end()) {
        hcclGroupCommList.push_back(comm);
    }
    return ret;
}

HcclResult
commInitTaskAppend(std::shared_ptr<struct hcclAsyncJob> job, HcclResult (*func)(struct hcclAsyncJob*), HcclComm* comm)
{
    HCCL_INFO("[hcclAsyncJobEnqueue] add item to queue");
    CHK_PRT_RET(!job, HCCL_ERROR("[commInitTaskAppend] job is nullptr"), HCCL_E_INTERNAL);
    /*hcclAsyncLaunch只是将job放入队列，并不等待执行完成。groupLaunch->asyncJobLaunch中给每个job起一个线程去执行*/
    job->func = func;
    job->comm = comm;
    job->state = hcclGroupJobRunning;
    hcclInitJobs.push_back(job);
    return HCCL_SUCCESS;
}
} // namespace hccl

void* hcclAsyncJobMain(void* arg)
{
    struct hcclAsyncJob* job = (struct hcclAsyncJob*)arg;
    job->result = job->func(job); /*func是上层asyncjob里面设置的函数*/
    if (job->result == HCCL_SUCCESS) {
        HCCL_INFO("Function launch success");
    }
    /*加锁修改job->state为hcclGroupJobDone*/
    std::unique_lock<std::mutex> lock(job->mtx);
    job->state = hcclGroupJobDone;
    return arg;
}

static HcclResult asyncJobLaunch()
{
    HCCL_DEBUG("[asyncJobLaunch] entered");
    HcclResult ret = HCCL_SUCCESS;
    bool jobsDone = false;

    if (!hcclInitJobs.empty()) {
        for (auto job : hcclInitJobs) {
            CHK_PRT_RET(!job, HCCL_ERROR("[asyncJobLaunch] job is nullptr"), HCCL_E_INTERNAL);
            job->thread.reset(new (std::nothrow) std::thread(&hcclAsyncJobMain, job.get()));
            CHK_PRT_RET(!job->thread, HCCL_ERROR("[asyncJobLaunch]threads reset failed "), HCCL_E_INTERNAL);
        }

        do { /*主线程轮询阻塞，等待所有线程上的asyncJob执行完成*/
            jobsDone = true;
            for (auto job : hcclInitJobs) {
                /*上面job执行线程可能并发修改state，在主线程里面要通过加线程锁来读取*/
                hcclGroupJobState_t state = hcclGroupJobJoined;
                std::unique_lock<std::mutex> lock(job->mtx);
                state = job->state;

                if (state == hcclGroupJobRunning) {
                    jobsDone = false;
                } else if (state == hcclGroupJobDone) {
                    job->thread->join();
                    job->state = hcclGroupJobJoined;
                    if (job->result != HCCL_SUCCESS && ret == HCCL_SUCCESS) {
                        ret = job->result;
                    }
                } else {
                    /* safety check */
                    CHK_PRT_RET(
                        state != hcclGroupJobJoined, HCCL_ERROR("[asyncJobLaunch] state != hcclGroupJobJoined"),
                        HCCL_E_INTERNAL);
                }
            }
            // Let preconnect threads progress.
            if (jobsDone == false)
                usleep(1);
        } while (jobsDone == false);
        hcclInitJobs.clear();
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[asyncJobLaunch] fail!"), ret);
    }
    return HCCL_SUCCESS;
}

// 计算集合通信算子的数据量（字节），用于按数据量降序排列（仅非V类算子参与排序）
u64 calcOpDataVolume(const hcclOpInfo& info)
{
    switch (info.coll) {
        case HcclCMDType::HCCL_CMD_ALLGATHER:
        case HcclCMDType::HCCL_CMD_ALLTOALL:
        case HcclCMDType::HCCL_CMD_ALLREDUCE:
        case HcclCMDType::HCCL_CMD_BROADCAST:
        case HcclCMDType::HCCL_CMD_REDUCE:
            return info.sendCount * SIZE_TABLE[info.sendType];
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER:
        case HcclCMDType::HCCL_CMD_SCATTER:
            return info.recvCount * SIZE_TABLE[info.recvType];
        default:
            return 0;
    }
}

// 对group任务排序：非V类按数据量降序在前，V类保持原序在后
std::vector<hcclOpInfo> sortGroupTasks(const std::deque<hcclOpInfo>& tasks)
{
    std::vector<hcclOpInfo> sorted(tasks.begin(), tasks.end());
    auto isVariant = [](const hcclOpInfo& op) {
        return op.coll == HcclCMDType::HCCL_CMD_ALLTOALLV || op.coll == HcclCMDType::HCCL_CMD_ALLTOALLVC
               || op.coll == HcclCMDType::HCCL_CMD_ALLGATHER_V || op.coll == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V;
    };
    auto itV = std::stable_partition(sorted.begin(), sorted.end(), [&isVariant](const hcclOpInfo& op) {
        return !isVariant(op);
    });
    std::stable_sort(sorted.begin(), itV, [](const hcclOpInfo& a, const hcclOpInfo& b) {
        return calcOpDataVolume(a) > calcOpDataVolume(b);
    });
    return sorted;
}

static HcclResult doLaunches(HcclComm comm)
{
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    std::shared_ptr<struct hcclKernelPlanner> planner = hcclComm->planner;
    HcclUs startutime = TIME_NOW();
    if (planner->nTasksP2p != 0) {
        // 将所有send/recv的任务打包作为一个集合通信算子来执行
        HCCL_INFO("HcclBatchSendRecvGroup, sendRecvInfo.size()[%u]", static_cast<u32>(planner->sendRecvInfo.size()));
        CHK_RET(HcclBatchSendRecvGroup(
            planner->sendRecvInfo.data(), planner->sendRecvInfo.size(), comm, planner->sendRecvMainStream));
    }
    HCCL_INFO("[doLaunches] take time [%lld]us.", DURATION_US(TIME_NOW() - startutime));
    if (planner->nTasksColl != 0) {
        /* 按数据量降序排列，V类算子各rank数据量可能不同，统一放到最后，保持原序 */
        auto sortedTasks = sortGroupTasks(planner->collTaskQueue);
        HCCL_INFO("Collectives sorted!");
        for (auto& taskColl : sortedTasks) {
            switch (taskColl.coll) {
                case HcclCMDType::HCCL_CMD_ALLGATHER:
                    HCCL_INFO("AllGather, sendCount[%llu]", taskColl.sendCount);
                    CHK_RET(HcclAllGatherInner(
                        taskColl.sendbuff, taskColl.recvbuff, taskColl.sendCount, taskColl.sendType, taskColl.comm,
                        taskColl.stream));
                    break;
                case HcclCMDType::HCCL_CMD_REDUCE_SCATTER:
                    HCCL_INFO("ReduceScatter, recvCount[%llu]", taskColl.recvCount);
                    CHK_RET(HcclReduceScatterInner(
                        taskColl.sendbuff, taskColl.recvbuff, taskColl.recvCount, taskColl.recvType, taskColl.op,
                        taskColl.comm, taskColl.stream));
                    break;
                case HcclCMDType::HCCL_CMD_ALLREDUCE:
                    HCCL_INFO("AllReduce, sendCount[%llu]", taskColl.sendCount);
                    CHK_RET(HcclAllReduceInner(
                        taskColl.sendbuff, taskColl.recvbuff, taskColl.sendCount, taskColl.sendType, taskColl.op,
                        taskColl.comm, taskColl.stream));
                    break;
                case HcclCMDType::HCCL_CMD_BROADCAST:
                    CHK_RET(HcclBroadcastInner(
                        taskColl.sendbuff, taskColl.sendCount, taskColl.sendType, taskColl.root, taskColl.comm,
                        taskColl.stream));
                    break;
                case HcclCMDType::HCCL_CMD_ALLTOALL:
                    CHK_RET(HcclAlltoAllInner(
                        taskColl.sendbuff, taskColl.sendCount, taskColl.sendType, taskColl.recvbuff, taskColl.recvCount,
                        taskColl.recvType, taskColl.comm, taskColl.stream));
                    break;
                case HcclCMDType::HCCL_CMD_ALLTOALLV:
                    CHK_RET(HcclAlltoAllVInner(
                        taskColl.sendbuff, taskColl.sendCounts, taskColl.sdispls, taskColl.sendType, taskColl.recvbuff,
                        taskColl.recvCounts, taskColl.rdispls, taskColl.recvType, taskColl.comm, taskColl.stream));
                    break;
                case HcclCMDType::HCCL_CMD_ALLTOALLVC:
                    CHK_RET(HcclAlltoAllVCInner(
                        taskColl.sendbuff, taskColl.sendCounts, taskColl.sendType, taskColl.recvbuff, taskColl.recvType,
                        taskColl.comm, taskColl.stream));
                    break;
                case HcclCMDType::HCCL_CMD_REDUCE:
                    CHK_RET(HcclReduceInner(
                        taskColl.sendbuff, taskColl.recvbuff, taskColl.recvCount, taskColl.recvType, taskColl.op,
                        taskColl.root, taskColl.comm, taskColl.stream));
                    break;
                case HcclCMDType::HCCL_CMD_SCATTER:
                    CHK_RET(HcclScatterInner(
                        taskColl.sendbuff, taskColl.recvbuff, taskColl.recvCount, taskColl.recvType, taskColl.root,
                        taskColl.comm, taskColl.stream));
                    break;
                case HcclCMDType::HCCL_CMD_ALLGATHER_V:
                    CHK_RET(HcclAllGatherVInner(
                        taskColl.sendbuff, taskColl.sendCount, taskColl.recvbuff, taskColl.recvCounts, taskColl.rdispls,
                        taskColl.recvType, taskColl.comm, taskColl.stream));
                    break;
                case HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V:
                    CHK_RET(HcclReduceScatterVInner(
                        taskColl.sendbuff, taskColl.sendCounts, taskColl.sdispls, taskColl.recvbuff, taskColl.recvCount,
                        taskColl.sendType, taskColl.op, taskColl.comm, taskColl.stream));
                    break;
                default:
                    HCCL_ERROR("[doLaunches] not supported hcclFunc!");
                    return HCCL_E_INTERNAL;
            }
        }
    }
    return HCCL_SUCCESS;
}

static HcclResult groupLaunch()
{ // 将各种通信域初始化/destroy的asyncJobs，在这里触发放到背景线程执行
    HCCL_INFO("[groupLaunch] entered");

    asyncJobLaunch();
    HCCL_DEBUG("[groupLaunch] asyncJobLaunch done");
    for (HcclComm comm : hcclGroupCommList) {
        doLaunches(comm);
    }
    HCCL_INFO("[groupLaunch] doLaunches done");
    // 流同步
    for (HcclComm comm : hcclGroupCommList) {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
        std::shared_ptr<struct hcclKernelPlanner> planner = hcclComm->planner;
        for (auto it : planner->collStreams) {
            CHK_RET(hcclStreamSynchronize(it));
        }
    }
    HCCL_INFO("groupLaunch Done!");
    return HCCL_SUCCESS;
}

inline void groupLocalResetJobState()
{
    // hcclcomm中group相关的变量
    for (HcclComm comm : hcclGroupCommList) {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
        hcclComm->planner = std::make_shared<hcclKernelPlanner>();
        hcclComm->SetGroupMode(false);
    }
    hcclGroupCommList.clear();

    return;
}

HcclResult HcclLegacyGroupEnd()
{
    groupLaunch();
    HCCL_INFO("[GroupEnd] done groupLaunch");
    groupLocalResetJobState();
    HCCL_INFO("[GroupEnd] to the end");
    return HCCL_SUCCESS;
}

HcclResult HcclLegacyAsyncJobLaunch() { return asyncJobLaunch(); }

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_kfc_batchwrite_process.h"

#include "common/aicpu_hccl_common.h"
#include "utils/hccl_aicpu_utils.h"
#include "framework/aicpu_kfc_prof.h"
#include "coll_batch_write_executor.h"

using namespace hccl;

ANONYMOUS_NAMESPACE_BEGIN
class CommonHcclMsgRingBuffer {
public:
    static constexpr uint8_t DEFAULT_CAPACITY = 4;

    CommonHcclMsgRingBuffer() : CommonHcclMsgRingBuffer(DEFAULT_CAPACITY) {}

    CommonHcclMsgRingBuffer(uint8_t capacity) : capacity_(capacity)
    {
        if (capacity > 0) {
            buffer_ = new CommonHcclMsg[capacity_];
        }
    }

    ~CommonHcclMsgRingBuffer()
    {
        if (capacity_ > 0 && buffer_ != nullptr) {
            delete[] buffer_;
            buffer_ = nullptr;
            capacity_ = 0;
        }
    }

    bool Enqueue(const CommonHcclMsg* msg)
    {
        if (capacity_ == 0) {
            HCCL_ERROR("capacity is zero");
            return false;
        }
        uint32_t curTail = tail_.load(std::memory_order_acquire);
        uint32_t nextTail = (curTail + 1) % capacity_;
        if (nextTail == head_.load(std::memory_order_acquire)) {
            HCCL_INFO("CommonHcclMsgRingBuffer queue is full.");
            return false;
        }
        s32 sRet = memcpy_s(&buffer_[curTail], sizeof(CommonHcclMsg), msg, sizeof(CommonHcclMsg));
        if (sRet != EOK) {
            HCCL_ERROR("memcpy_s failed, errorno[%d]", sRet);
            return false;
        }
        tail_.store(nextTail, std::memory_order_release);
        return true;
    }

    bool Peek(CommonHcclMsg* msg)
    {
        uint32_t tempIdx;
        do {
            tempIdx = head_.load(std::memory_order_acquire);
            if (tempIdx == tail_.load(std::memory_order_acquire)) {
                return false;
            }
            s32 sRet = memcpy_s(msg, sizeof(CommonHcclMsg), &buffer_[tempIdx], sizeof(CommonHcclMsg));
            if (sRet != EOK) {
                HCCL_ERROR("memcpy_s failed, errorno[%d]", sRet);
                return false;
            }
        } while (tempIdx != head_.load(std::memory_order_acquire)); // 确保在读取过程中head没被修改
        return true;
    }

    bool Dequeue()
    {
        uint32_t curHead = head_.load(std::memory_order_acquire);
        if (curHead == tail_.load(std::memory_order_acquire) || capacity_ == 0) {
            HCCL_INFO("CommonHcclMsgRingBuffer queue is empty.");
            return false;
        }
        head_.store((curHead + 1) % capacity_, std::memory_order_release);
        return true;
    }

    void Clear()
    {
        head_.store(0, std::memory_order_release);
        tail_.store(0, std::memory_order_release);
    }

private:
    uint8_t capacity_{0};
    std::atomic<uint32_t> head_{0};
    std::atomic<uint32_t> tail_{0};
    CommonHcclMsg* buffer_{nullptr};
};

struct BatchWriteItem {
    uint64_t localBuf;
    uint64_t remoteBuf;
    uint64_t count;
    uint32_t dataType;
    uint32_t remoteRankId;
};
WqeSendSharedContect g_sharedCtx;
CommonHcclMsgRingBuffer g_hcclMsgQueue;
constexpr s32 PREFER_CLUSTER_ID = 0;
constexpr u32 DELAY_TIME_IN_NS = 15U * 1000U;
static constexpr uint64_t WQE_SEND_TIMEOUT = 15;
std::mutex g_mtxForCpuCheck;
#ifdef CCL_LLT
// mock GetCpuId 多个线程需要放回不同的值，mock组件在多线程时不安全，会放回错误。所以在跑llt时加锁。
std::mutex g_mtxForLLT;
#endif

HcclResult ConcurrentPostSendWqe(const CommonHcclMsg& commonHcclMsg, const AicpuComContext* ctx, u8* needSendTotalNum)
{
    const BatchWriteItem* item = reinterpret_cast<BatchWriteItem*>(static_cast<uintptr_t>(commonHcclMsg.sendBuffer));
    std::vector<Transport::Buffer> remoteList = {{}};
    std::vector<Transport::Buffer> local = {{}};
    int32_t cpuId = 0;
    {
#ifdef CCL_LLT
        std::lock_guard<std::mutex> lock(g_mtxForLLT);
#endif
        cpuId = HcclAicpuUtils::GetCpuId();
    }
    u32 threadId = g_sharedCtx.curThreadIdsOnCpu[cpuId];
    u32 sendWqeNum = 0;
    for (u64 i = 0; i < commonHcclMsg.dataCnt; ++i) {
        if (item->remoteRankId != ctx->rankId) {
            (*needSendTotalNum)++;
            if (item->remoteRankId % g_sharedCtx.workedThreadNum == threadId) {
                remoteList[0].addr = reinterpret_cast<void*>(item->remoteBuf);
                local[0].addr = reinterpret_cast<void*>(item->localBuf);
                remoteList[0].size = local[0].size
                    = item->count * DataUnitSize(static_cast<HcclDataType>(item->dataType));
                HCCL_INFO(
                    "Batch write item[%u]: context rankId [%u], remoteRankId[%u], sendThreadId[%ld], remoteBuf[%#llx],"
                    " localBuf[%#llx], dataType[%u], count[%lu]",
                    i, ctx->rankId, item->remoteRankId, threadId, item->remoteBuf, item->localBuf, item->dataType,
                    item->count);
                CHK_RET(HcclAicpuUtils::PostSend(*ctx, item->remoteRankId, remoteList, local, true));
                sendWqeNum++;
            }
        }
        ++item;
    }
    g_sharedCtx.sendWqeNum[threadId] = sendWqeNum;
    HCCL_INFO("thread %u send %u wqe success.", threadId, sendWqeNum);
    return HCCL_SUCCESS;
}

bool CheckTimeOut(u64 startTimeStamp, u64 timeOutTime)
{
    if ((GetCurCpuTimestamp() - startTimeStamp) > static_cast<unsigned long long>(NSEC_PER_SEC * timeOutTime)) {
        HCCL_ERROR("Execution TimeOut %lus...", timeOutTime);
        return true;
    }
    return false;
}

HcclResult WaitForSlaveCompletion(u8 needSendTotalNum)
{
    HCCL_DEBUG("needsendTotalNum is %ld.", needSendTotalNum);
    u64 startTimeStamp = GetCurCpuTimestamp();
    while (true) {
        uint32_t sendNum = 0;
        for (uint32_t i = 0; i < g_sharedCtx.workedThreadNum; ++i) {
            HCCL_DEBUG("wait thread %ld send %ld wqe success.", i, g_sharedCtx.sendWqeNum[i]);
            sendNum += g_sharedCtx.sendWqeNum[i];
        }
        if (needSendTotalNum <= sendNum) {
            for (uint32_t i = 0; i < g_sharedCtx.workedThreadNum; ++i) {
                g_sharedCtx.sendWqeNum[i] = 0U;
            }
            HCCL_INFO("needsendTotalNum is %ld, already send %ld", needSendTotalNum, sendNum);
            return HCCL_SUCCESS;
        }
        if (CheckTimeOut(startTimeStamp, WQE_SEND_TIMEOUT)) {
            g_sharedCtx.taskFinishFlag.store(true, std::memory_order_release);
            HCCL_ERROR("slave thread send wqe timeout.");
            return HCCL_E_TIMEOUT;
        }
    }
}

void InitMultiThreadSharedCtx(int32_t cpuId)
{
    g_sharedCtx.startedThreadNum = 1;
    g_hcclMsgQueue.Clear();
    g_sharedCtx.taskFinishFlag.store(false, std::memory_order_release);
    g_sharedCtx.curThreadIdsOnCpu[cpuId] = 0;
    g_sharedCtx.sendWqeNum[0] = 0;
    for (s32 i = 0; i < AICPU_CNT; ++i) {
        g_sharedCtx.curThreadIdsOnCpu[i] = 0;
    }
}

HcclResult OrchestrateSdmaSqe(const OpParam& param, hccl::HcclCommAicpu& comm)
{
    AicpuKfcProf::SetKfcTimeLine(KfcTimeLine::HCC_EXEC_START_TIME);
    const u32 queueIdx = param.BatchWriteDataDes.queueIdx;
    auto streams = comm.GetSlaveStream();
    CHK_PRT_RET(
        queueIdx >= streams.size(), HCCL_ERROR("Invalid queue idx %u, stream number %u", queueIdx, streams.size()),
        HCCL_E_PARA);
    auto streamInfo = streams[queueIdx];
    u8* newSqAddr = static_cast<u8*>(param.inputPtr);
    auto& sqeBuffer = streamInfo.GetSqeContextPtr()->buffer;
    u16& taskId = sqeBuffer.tailSqeTaskId;
    const u32 sqeCnt = param.BatchWriteDataDes.itemNum;
    const u32 depth = streamInfo.GetHcclStreamInfo().sqDepth;
    CHK_PRT_RET(sqeCnt >= depth, HCCL_ERROR("Sqe count %u reaches the sq depth %u.", sqeCnt, depth), HCCL_E_PARA);
    u8 sqeType;
    for (u32 i = 0U; i < sqeCnt; ++i) {
        const uint8_t* sqe = newSqAddr + i * AC_SQE_SIZE;
        AddOneMemcpySqeV1(
            streamInfo.id(), taskId++, nullptr, 0U, ACL_DT_UNDEFINED, ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM, nullptr, 0U, 0U,
            0U, 0U, static_cast<uint8_t>(LinkType::LINK_RESERVED), sqe, &sqeType, SDMA_QOS_DEFAULT);
    }

    u32& head = sqeBuffer.sqHead;
    u32& tail = sqeBuffer.sqTail;
    u32 newTail = (tail + sqeCnt) % depth;
    HCCL_INFO(
        "Before send sqe:%d cnt:%u head:%u curtail:%u newTail:%u.", streamInfo.sqId(), sqeCnt, head, tail, newTail);
    const u64 startUsec = GetCurCpuTimestamp();
    const u32 devId = comm.GetDevId();
    while ((tail + depth - head) % depth + sqeCnt >= depth) {
        CHK_RET(QuerySqStatusByType(devId, streamInfo.sqId(), DRV_SQCQ_PROP_SQ_HEAD, head));
        if (GetCurCpuTimestamp() - startUsec > NSEC_PER_SEC * dfx::kKfcTimeOut) {
            HCCL_ERROR("Rtsq(%u) full for more than %u seconds, head:%u.", streamInfo.sqId(), dfx::kKfcTimeOut, head);
            return HCCL_E_INTERNAL;
        }
    }

    u8* sqAddr = static_cast<u8*>(streamInfo.GetHcclStreamInfo().sqBaseAddr);
    const u32 left = depth - tail;
    HCCL_INFO(
        "Before copy sqe:%d cnt:%u head:%u curtail:%u newTail:%u left:%u", streamInfo.sqId(), sqeCnt, head, tail,
        newTail, left);
    if (sqeCnt <= left) {
        (void)memcpy_s(sqAddr + tail * AC_SQE_SIZE, left * AC_SQE_SIZE, newSqAddr, sqeCnt * AC_SQE_SIZE);
    } else {
        (void)memcpy_s(sqAddr + tail * AC_SQE_SIZE, left * AC_SQE_SIZE, newSqAddr, left * AC_SQE_SIZE);
        (void)memcpy_s(sqAddr, head * AC_SQE_SIZE, newSqAddr + left * AC_SQE_SIZE, (sqeCnt - left) * AC_SQE_SIZE);
    }
#ifdef __aarch64__
    __asm__ __volatile__("dsb st" : : : "memory");
#endif
    if (UNLIKELY(HcclCheckLogLevel(DLOG_DEBUG))) {
        rtStarsMemcpyAsyncSqe_t* tmp = reinterpret_cast<rtStarsMemcpyAsyncSqe_t*>(sqAddr) + tail;
        for (u32 i = tail; i < newTail; ++i) {
            HCCL_DEBUG(
                "[Sdma-BatchWrite]Orchestrated sq %u, idx %u, stream %u, task %u, data length %u, "
                "src addr %#llx, dst addr %#llx.",
                streamInfo.sqId(), i, tmp->header.rtStreamId, tmp->header.taskId, tmp->length,
                (static_cast<uint64_t>(tmp->src_addr_high) << 32U) | tmp->src_addr_low,
                (static_cast<uint64_t>(tmp->dst_addr_high) << 32U) | tmp->dst_addr_low);
            ++tmp;
        }
    }

    AicpuKfcProf::SetKfcTimeLine(KfcTimeLine::SEND_TASK_START_TIME);
    CHK_RET(ConfigSqStatusByType(devId, streamInfo.sqId(), DRV_SQCQ_PROP_SQ_TAIL, newTail));
    tail = newTail;
    AicpuKfcProf::SetKfcTimeLine(KfcTimeLine::SEND_SQE_FINISH_TIME);
    return HCCL_SUCCESS;
}
ANONYMOUS_NAMESPACE_END

void AicpuKfcBatchwriteProcess::FinishProcess()
{
    HCCL_INFO("master task is finished.");
    g_sharedCtx.taskFinishFlag.store(true, std::memory_order_release);
}

AicpuServerRole AicpuKfcBatchwriteProcess::GetVerifiedServerRole(const AicpuComContext& ctx)
{
    if (!ctx.multiServerFlag) {
        HCCL_INFO("Skip server start check for non-multi server scene.");
        return AicpuServerRole::MASTER;
    }

    static std::atomic<u32> opThreadIdx{0U};
    if (HcclAicpuUtils::GetCurClusterId() != PREFER_CLUSTER_ID) {
        u64 startTimestamp = GetCurCpuTimestamp();
        while (opThreadIdx.load(std::memory_order_acquire) == 0U
               && GetCurCpuTimestamp() - startTimestamp < DELAY_TIME_IN_NS) {
            usleep(1);
        }
    }

    std::lock_guard<std::mutex> lock(g_mtxForCpuCheck);
    int32_t cpuId = HcclAicpuUtils::GetCpuId();
    AicpuServerRole role;
    if (opThreadIdx.fetch_add(1U, std::memory_order_acq_rel) == 0U) {
        InitMultiThreadSharedCtx(cpuId);
        HCCL_INFO("Master thread starts on cpu %d, clusterID %d", cpuId, HcclAicpuUtils::GetCurClusterId());
        role = AicpuServerRole::MASTER;
    } else {
        if (HcclAicpuUtils::GetCurClusterId() != PREFER_CLUSTER_ID
            || g_sharedCtx.startedThreadNum >= MAX_BATCH_WRITE_THREAD_NUM) {
            HCCL_INFO(
                "This is invalid thread, cluster id %d, started thread number %ld.", HcclAicpuUtils::GetCurClusterId(),
                g_sharedCtx.startedThreadNum);
            role = AicpuServerRole::INVALID;
        } else {
            g_sharedCtx.sendWqeNum[g_sharedCtx.startedThreadNum] = 0;
            g_sharedCtx.curThreadIdsOnCpu[cpuId] = g_sharedCtx.startedThreadNum++;
            HCCL_INFO(
                "Slave thread index %u on cpu %d. clusterID %d", g_sharedCtx.curThreadIdsOnCpu[cpuId], cpuId,
                HcclAicpuUtils::GetCurClusterId());
            role = AicpuServerRole::SLAVE;
        }
    }
    // 老驱动包无法获取GetBlockNum，使用默认值6
    const u32 numBlocks = HcclAicpuUtils::GetBlockNum(6U);
    if (opThreadIdx.load(std::memory_order_acquire) == numBlocks) {
        HCCL_INFO("Clear thread index at last with block dim %u.", numBlocks);
        opThreadIdx.store(0U, std::memory_order_relaxed);
    }
    return role;
}

// 真正处理BatchWrite master 从commonHcclMsg中取消息，更新工作线程数，放到队列中。 从队列中取数据进行发送。
HcclResult
AicpuKfcBatchwriteProcess::HandleBatchWriteOperation(const CommonHcclMsg& commonHcclMsg, const AicpuComContext* ctx)
{
    if (commonHcclMsg.dataCnt == 0UL || commonHcclMsg.sendBuffer == 0UL) {
        HCCL_ERROR(
            "Get msg send buffer is nullptr or dataCnt is zero. "
            "Msg[commType %u, opType %u, sendBuffer %p, dataCnt %lu]",
            static_cast<uint32_t>(commonHcclMsg.commType), static_cast<uint32_t>(commonHcclMsg.opType),
            commonHcclMsg.sendBuffer, commonHcclMsg.dataCnt);
        return HCCL_E_PARA;
    }

    g_sharedCtx.workedThreadNum = g_sharedCtx.startedThreadNum;
    if (g_sharedCtx.workedThreadNum > 1) {
        bool success = false;
        while (!success) {
            success = g_hcclMsgQueue.Enqueue(&commonHcclMsg);
        }
    }
    u8 needSendTotalNum = 0;
    CHK_RET(ConcurrentPostSendWqe(commonHcclMsg, ctx, &needSendTotalNum));
    HCCL_DEBUG("total need send wqe num is %u", needSendTotalNum);
    CHK_RET(WaitForSlaveCompletion(needSendTotalNum));
    g_hcclMsgQueue.Dequeue();
    return HCCL_SUCCESS;
}

HcclResult AicpuKfcBatchwriteProcess::RunSlaveRpcServerForApi(AicpuComContext* ctx)
{
    HCCL_INFO("----------start Slave Rpc Server For Api Hccl, ctx:%p ----------", ctx);
    if (ctx->devType != DevType::DEV_TYPE_910B) {
        HCCL_WARNING("Platform not support multi thread handle batch write, please use 910B platform.");
        return HCCL_SUCCESS;
    }
    CommonHcclMsg commonHcclMsg;
    int32_t sendSeqNum = -1;
    u32 threadId = g_sharedCtx.curThreadIdsOnCpu[HcclAicpuUtils::GetCpuId()];
    while (true) {
#if defined(__aarch64__) || defined(__amd64__)
        __asm__ __volatile__("nop");
#endif

        if (g_sharedCtx.taskFinishFlag.load(std::memory_order_acquire)) {
            HCCL_INFO("task is finished, slave process exit");
            break;
        }
        u8 needSendTotalNum = 0;
        if (threadId < g_sharedCtx.workedThreadNum && g_hcclMsgQueue.Peek(&commonHcclMsg)
            && commonHcclMsg.seqNum != sendSeqNum) {
            if (commonHcclMsg.commType == HcclCMDType::HCCL_CMD_BATCH_WRITE) {
                CHK_RET(ConcurrentPostSendWqe(commonHcclMsg, ctx, &needSendTotalNum));
                sendSeqNum = commonHcclMsg.seqNum;
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult
AicpuKfcBatchwriteProcess::BatchWriteProcess(hccl::OpParam& opParam, hccl::HcclCommAicpu& comm, HcclOpResParam& param)
{
    static hccl::AlgResourceResponse* algResResponse = nullptr;
    if (UNLIKELY(algResResponse == nullptr || algResResponse->slaveStreams.empty())) {
        const std::string tag = comm.GetGroupName()
                                + std::to_string(static_cast<uint8_t>(HcclCMDType::HCCL_CMD_BATCH_WRITE))
                                + std::string("_mc2") + std::string(BATCH_WRITE_ALG_NAME) + std::string("_device");
        std::unique_ptr<hccl::CollExecutorBase> executor;
        CHK_RET(comm.GetAlgResponseRes(tag, BATCH_WRITE_ALG_NAME, opParam, &param, executor, algResResponse));
    }
    const u64 ts = GetCurCpuTimestamp();
    while (algResResponse->slaveStreams.empty()) {
        CHK_PRT_RET(
            GetCurCpuTimestamp() - ts > static_cast<u64>(NSEC_PER_SEC),
            HCCL_ERROR("[%s]Timeout during batchwrite initialization.", __func__), HCCL_E_INTERNAL);
    }
    HcclResult ret = OrchestrateSdmaSqe(opParam, comm);
    AicpuKfcProf::GetCurrentAicpuProf()->workCnt++;
    return ret;
}

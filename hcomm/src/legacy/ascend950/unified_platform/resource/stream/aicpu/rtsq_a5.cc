/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include <chrono>
#include <unordered_map>
#include "rtsq_a5.h"
#include "log.h"
#include "exception_util.h"
#include "internal_exception.h"
#include "sqe_build_a5.h"
#include "sqe.h"
#ifdef CCL_KERNEL_AICPU
#include "aicpu_ts_primitives_c_adpt.h"
#endif
#include "aicpu_task_utils.h"

namespace Hccl {
using namespace std;
constexpr u32 RTSQ_A5_PART_ID = 0;
constexpr u32 PRINT_INTERVAL = 30;

RtsqA5::RtsqA5(u32 devPhyId, u32 streamId, u32 sqId) : RtsqBase(devPhyId, streamId, sqId) { SetTaskIdBySqeId(); }

RtsqA5::RtsqA5(u32 devPhyId, u32 streamId, u32 sqId, bool launchFlag) : RtsqBase(devPhyId, streamId, sqId)
{
    SetTaskIdBySqeId();
    launchFlag_ = launchFlag;
}

void RtsqA5::Reset(bool reset)
{
    RtsqBase::Reset(reset);
    pendingSqeCnt = 0;
    s32 sRet = memset_s(locBuf, RTSQ_SQE_SIZE * PER_LAUNCH_SQE_CNT, 0, RTSQ_SQE_SIZE * PER_LAUNCH_SQE_CNT);
    if (UNLIKELY(sRet != EOK)) {
        auto msg = StringFormat("[RtsqA5][Reset] locBuf memset fail. errorno[%d]", sRet);
        THROW<InternalException>(msg);
    }
    HCCL_INFO("[NsRecovery]RtsqA5::%s success", __func__);
}

// 计算head和tail之间的距离
u32 RtsqA5::GetTailToHeadDist() const
{
    if (UNLIKELY(sqHead_ == sqTail_)) { // 头尾相同，则距离大小为sq深度
        return sqDepth_;
    }
    return (sqTail_ < sqHead_) ? (sqHead_ - sqTail_) : (sqDepth_ - (sqTail_ - sqHead_));
}

void RtsqA5::MakeSureAvailableSpace()
{
    u32 availableSpace = GetTailToHeadDist();
    auto startTime = std::chrono::steady_clock::now();
#ifdef CCL_KERNEL_AICPU
    sqFullTimeout_ = GetSqFullTimeOut();
#endif
    const std::chrono::seconds printInterval(PRINT_INTERVAL); // 打印间隔30s
    auto lastPrintTime = std::chrono::steady_clock::now() - printInterval;
    HCCL_INFO(
        "[%s]sqId:%u, sqFullTimeout_: %u s, sqHead:%u, sqTail:%u, pendingSqeCnt:%u", __func__, sqId_, sqFullTimeout_,
        sqHead_, sqTail_, pendingSqeCnt);

    while (availableSpace <= pendingSqeCnt) {
        sqHead_ = QuerySqHead();
        availableSpace = GetTailToHeadDist();
        if (availableSpace > pendingSqeCnt) {
            break; // 避免head没更新导致假反压
        }

        auto curTime = std::chrono::steady_clock::now();
        if (UNLIKELY(curTime - lastPrintTime >= printInterval)) {
            HCCL_RUN_INFO(
                "[%s]while loop, sqId:%u, sqHead:%u, sqTail:%u, availableSpace:%u, pendingSqeCnt:%u, "
                "sqFullTimeout_:%u s",
                __func__, sqId_, sqHead_, sqTail_, availableSpace, pendingSqeCnt, sqFullTimeout_);
            lastPrintTime = curTime;
        }

        CheckLaunchTaskStatus(startTime, curTime);
#ifdef CCL_KERNEL_AICPU
        HcclResult ret = HandleDispatchAllStreams();
        if (UNLIKELY(ret != HCCL_SUCCESS)) {
            auto msg
                = StringFormat("RtsqA5::%s HandleDispatchAllStreams failed, ret = %d, sqId:%u, ", __func__, ret, sqId_);
            HCCL_ERROR("%s", msg.c_str());
            THROW<InternalException>(msg);
        }
#endif
        if (checkOpExecStatusCallback_ != nullptr) {
            checkOpExecStatusCallback_();
        }
    }
}

void RtsqA5::CheckLaunchTaskStatus(
    const std::chrono::steady_clock::time_point& startTime, const std::chrono::steady_clock::time_point& curTime)
{
    bool isTimeout = (sqFullTimeout_ == 0) ? false : ((curTime - startTime) >= std::chrono::seconds(sqFullTimeout_));
    // step1 检测是否launch超时，如果超时打印rtsq full的ERROR日志
    if (UNLIKELY(isTimeout)) {
        HCCL_ERROR(
            "Rtsq full, sqFullTimeout_:[%u s]. sqId:[%u], sqHead:[%u], sqTail:[%u], pendingSqeCnt:[%u]", sqFullTimeout_,
            sqId_, sqHead_, sqTail_, pendingSqeCnt);
    }

    HcclResult checkRet = (checkExecStatusCallback_ != nullptr) ? checkExecStatusCallback_(isTimeout) : HCCL_SUCCESS;
    // step2 通信域状态为HCCL_COMM_STATUS_SUSPENDING状态，则终止launch不抛异
    if (UNLIKELY(checkRet == HCCL_E_SUSPENDING)) {
        pendingSqeCnt = 0;
        return;
    }
    // step3 调用回调检查执行状态：1、如果超时，打印taskException；2、如果通信域不可用，终止launch
    if (UNLIKELY(isTimeout || checkRet != HCCL_SUCCESS)) {
        THROW<InternalException>(
            StringFormat("[%s]stop launch Task, isTimeout[%d], checkRet[%d]", __func__, isTimeout, checkRet));
    }
}

void RtsqA5::CopySqeBufToSq(u8* sqeBuf) const
{
    u8* sqCurrAddr = ReinterpretAs<u8*>(sqBaseAddr_) + sqTail_ * RTSQ_SQE_SIZE;
    if (sqTail_ >= sqHead_) {
        u32 depthLeft = sqDepth_ - sqTail_;
        if (pendingSqeCnt <= depthLeft) { // 没有回绕
            HCCL_INFO(
                "RtsqA5::%s copy sqe from sqe buffer, sqId_: %u, streamId_: %u, cur head: %u, cur tail: %u, size: %u, "
                "depth remain: %u",
                __func__, sqId_, streamId_, sqHead_, sqTail_, pendingSqeCnt, depthLeft);
            int ret = memcpy_sp(sqCurrAddr, pendingSqeCnt * AC_SQE_SIZE, sqeBuf, pendingSqeCnt * RTSQ_SQE_SIZE);
            if (UNLIKELY(ret != 0)) {
                THROW<InternalException>(StringFormat("RtsqA5::%s sqe memcpy_sp failed, ret = %d", __func__, ret));
            }
        } else {
            HCCL_INFO(
                "RtsqA5::%s copy sqe twice, sqId_: %u, streamId_: %u, cur head: %u, cur tail: %u, cnt: %u, depth "
                "remain: %u",
                __func__, sqId_, streamId_, sqHead_, sqTail_, pendingSqeCnt, depthLeft);
            // 先拷贝rtsq里剩余空间大小
            int ret = memcpy_sp(sqCurrAddr, depthLeft * AC_SQE_SIZE, sqeBuf, depthLeft * RTSQ_SQE_SIZE);
            if (ret != 0) {
                THROW<InternalException>(
                    StringFormat("RtsqA5::%s rtsq remaining space memcpy_sp failed, ret = %d", __func__, ret));
            }
            // 拷贝剩余sqe
            ret = memcpy_sp(
                ReinterpretAs<u8*>(sqBaseAddr_), sqHead_ * RTSQ_SQE_SIZE, sqeBuf + depthLeft * RTSQ_SQE_SIZE,
                (pendingSqeCnt - depthLeft) * AC_SQE_SIZE);
            if (UNLIKELY(ret != 0)) {
                THROW<InternalException>(
                    StringFormat("RtsqA5::%s remaining sqe memcpy_sp failed, ret = %d", __func__, ret));
            }
        }
    } else {
        HCCL_INFO(
            "RtsqA5::%s copy sqe from sqe buffer, tail < head, sqId_: %u, streamId_: %u, cur head: %u, cur tail: %u, "
            "size: %u",
            __func__, sqId_, streamId_, sqHead_, sqTail_, pendingSqeCnt);
        int ret = memcpy_sp(sqCurrAddr, pendingSqeCnt * AC_SQE_SIZE, sqeBuf, pendingSqeCnt * RTSQ_SQE_SIZE);
        if (UNLIKELY(ret != 0)) {
            THROW<InternalException>(StringFormat("RtsqA5::%s sqe memcpy_sp failed, ret = %d", __func__, ret));
        }
    }
}

void RtsqA5::PreLaunchSqeForCache(bool& needCacheTask)
{
    // 校验needCacheTaskCallback_
    // 注意: A5新流程下needCacheTaskCallback_一定非空; 但A5老流程下不支持aicpu task cache, needCacheTaskCallback_为空;
    //     为避免A5老流程报错, 这里为空时跳过执行而非报错
    needCacheTask = false;
    if (UNLIKELY(needCacheTaskCallback_ == nullptr)) {
        HCCL_WARNING("[RtsqA5][PreLaunchSqeForCache] needCacheTaskCallback_ is null, keep needCacheTask as false");
    } else {
        needCacheTask = needCacheTaskCallback_();
    }
}

void RtsqA5::PostLaunchSqeForCache()
{
    // 注意: 只有needCacheTask为true时才调用PostLaunchSqeForCache, 此时一定是A5新流程, 因此addSqeArrayCallback_一定非空
    if (UNLIKELY(aicpuTsThreadPtr_ == nullptr)) {
        THROW<InternalException>("[RtsqA5][PostLaunchSqeForCache] aicpuTsThreadPtr_ is null");
    }
    if (UNLIKELY(addSqeArrayCallback_ == nullptr)) {
        THROW<InternalException>("[RtsqA5][PostLaunchSqeForCache] addSqeArrayCallback_ is null");
    }
    HcclResult ret = addSqeArrayCallback_(this, aicpuTsThreadPtr_, pendingSqeCnt, locBuf, streamId_);
    if (UNLIKELY(ret != HCCL_SUCCESS)) {
        THROW<InternalException>("[RtsqA5][PostLaunchSqeForCache] addSqeArrayCallback_ failed, ret %d", ret);
    }
}

// 向芯片RTSQ VA中写入 SQE，并触发芯片执行
void RtsqA5::LaunchTask()
{
    HCCL_INFO("RtsqA5::%s: START, pendingSqeCnt[%u]", __func__, pendingSqeCnt);
    if (pendingSqeCnt == 0) { // 没有SQE ，直接返回
        HCCL_INFO("RtsqA5::%s: pendingSqeCnt is %u, return", __func__, pendingSqeCnt);
        return;
    }
    // 确保 rtsq 有足够空间放pending SQE
    MakeSureAvailableSpace();

    if (pendingSqeCnt == 0) {
        return;
    }

    bool needCacheTask = false;
    PreLaunchSqeForCache(needCacheTask);
    // localBuffer拷贝到 RTSQ
    CopySqeBufToSq(locBuf);

    // 正常展开按需打印SQE
    if ((UNLIKELY(GetPlfDebugConfigValue() & PLF_TASK)) || UNLIKELY(HcclCheckLogLevel(HCCL_LOG_INFO))) {
        PLF_CONFIG_INFO(
            PLF_TASK, "[RtsqA5][LaunchTask] dump %llu generated SQEs in stream[%u]", pendingSqeCnt, streamId_);

        int ret = HCCL_SUCCESS;
        uint8_t* sqePtr = locBuf;
        for (size_t sqeIdx = 0; sqeIdx < pendingSqeCnt; sqeIdx++) {
            PLF_CONFIG_INFO(PLF_TASK, "[RtsqA5][LaunchTask] %uth generated SQE in stream[%u]", sqeIdx, streamId_);
            ret = hcomm::AicpuTaskUtils::DumpSqeContent(sqePtr);
            if (UNLIKELY(ret != HCCL_SUCCESS)) {
                THROW<InternalException>(StringFormat("RtsqA5::%s DumpSqeContent failed, ret = %d", __func__, ret));
            }

            sqePtr += RTSQ_SQE_SIZE;
        }
    }

    // 更新tail，触发芯片执行
    u32 newTail = (sqTail_ + pendingSqeCnt) % sqDepth_;
    ConfigSqTail(newTail);
    sqTail_ = newTail;

    // 缓存sqe
    if (needCacheTask) {
        PostLaunchSqeForCache();
    }
    // 清空本地的locBuffer和sqeCnt数目
    HCCL_INFO(
        "RtsqA5::%s: END, pendingSqeCnt[%u], streamId_[%u] sqHead_[%u] sqTail_[%u]", __func__, pendingSqeCnt, streamId_,
        sqHead_, sqTail_);
    pendingSqeCnt = 0;
    (void)memset_s(locBuf, RTSQ_SQE_SIZE * PER_LAUNCH_SQE_CNT, 0, RTSQ_SQE_SIZE * PER_LAUNCH_SQE_CNT); // locBuffer清零
}

void RtsqA5::RefreshSqeHeaderTaskField(Rt91095StarsSqeHeader* sqeHeaderPtr)
{
    SetSqeHeaderTaskFields(sqeHeaderPtr, taskId_);
    SetTaskIdBySqeId();
}

// 向芯片RTSQ VA中写入aicpu task cache SQE，并触发芯片执行
void RtsqA5::LaunchNewTask(uint8_t* sqeArray, uint32_t sqeCount)
{
    // 注意: cache命中时才会调用LaunchNewTask, 此时一定不存在pending SQE
    if (UNLIKELY(pendingSqeCnt > 0)) {
        THROW<InternalException>(StringFormat(
            "RtsqA5::%s: pendingSqeCnt[%u] should be 0 when aicpu task cache hits!", __func__, pendingSqeCnt));
    }

    // 临时设置pendingSqeCnt, 用于MakeSureAvailableSpace
    pendingSqeCnt = sqeCount;

    // 确保 rtsq 有足够空间放pending SQE
    MakeSureAvailableSpace();

    // sqeArray拷贝到 RTSQ
    CopySqeBufToSq(sqeArray);

    // 更新tail，触发芯片执行
    u32 newTail = (sqTail_ + pendingSqeCnt) % sqDepth_;
    ConfigSqTail(newTail);
    sqTail_ = newTail;

    HCCL_INFO(
        "RtsqA5::%s: END, pendingSqeCnt[%u], streamId_[%u] sqHead_[%u] sqTail_[%u]", __func__, pendingSqeCnt, streamId_,
        sqHead_, sqTail_);

    // 重置pendingSqeCnt
    pendingSqeCnt = 0;
}

void RtsqA5::TryLaunchTask()
{
    if (pendingSqeCnt == 0) {
        return;
    }

    sqHead_ = QuerySqHead();
    u32 availableSpace = GetTailToHeadDist();
    if (availableSpace <= pendingSqeCnt) {
        return;
    }

    bool needCacheTask = false;
    PreLaunchSqeForCache(needCacheTask);

    CopySqeBufToSq(locBuf);

    u32 newTail = (sqTail_ + pendingSqeCnt) % sqDepth_;
    ConfigSqTail(newTail);
    sqTail_ = newTail;

    // 缓存sqe
    if (needCacheTask) {
        PostLaunchSqeForCache();
    }
    pendingSqeCnt = 0;
    (void)memset_s(locBuf, RTSQ_SQE_SIZE * PER_LAUNCH_SQE_CNT, 0, RTSQ_SQE_SIZE * PER_LAUNCH_SQE_CNT);
    HCCL_INFO(
        "RtsqA5::%s: END, pendingSqeCnt[%u], streamId_[%u] sqHead_[%u] sqTail_[%u]", __func__, pendingSqeCnt, streamId_,
        sqHead_, sqTail_);
}

u8* RtsqA5::GetCurrSqeBuffer()
{
    lastSqeAddr_ = sqBaseAddr_ + static_cast<u64>((sqTail_ + pendingSqeCnt) % sqDepth_) * RTSQ_SQE_SIZE;
    return locBuf + pendingSqeCnt * RTSQ_SQE_SIZE;
}

u64 RtsqA5::GetSqeAddr() const { return lastSqeAddr_; }

void RtsqA5::RefreshInfo()
{
    SetTaskIdBySqeId();
    pendingSqeCnt++;

#ifdef CCL_KERNEL_AICPU
    if (launchFlag_ && !IsBatchLaunchMode()) {
        LaunchTask();
        return;
    }
#endif

    if (pendingSqeCnt != PER_LAUNCH_SQE_CNT) {
        return;
    }
    // 挂起的sqe数量为128个，则需要向芯片RTSQ中写入task
    LaunchTask();
}

void RtsqA5::NotifyWait(u32 notifyId) { NotifyWait(notifyId, GetKernelExecTimeoutFromEnvConfig()); }

void RtsqA5::NotifyWait(u32 notifyId, u32 timeout)
{
    BuildA5SqeNotifyWait(streamId_, taskId_, notifyId, timeout, GetCurrSqeBuffer());
    HCCL_INFO(
        "RtsqA5::NotifyWait: streamId %u, taskId %u, notifyId %u, timeout[%u s]", streamId_, taskId_, notifyId,
        timeout);
    RefreshInfo();
}

void RtsqA5::NotifyRecordLoc(u32 notifyId)
{
    BuildA5SqeNotifyRecord(streamId_, taskId_, notifyId, GetCurrSqeBuffer());
    HCCL_INFO("RtsqA5::NotifyRecordLoc: streamId %u, taskId %u, notifyId %u", streamId_, taskId_, notifyId);
    RefreshInfo();
}

void RtsqA5::Cnt1toNNotifyWait(u32 notifyId, u32 value)
{
    BuildA5SqeCnt1toNNotifyWait(streamId_, taskId_, notifyId, value, GetCurrSqeBuffer());
    HCCL_INFO("RtsqA5::Cnt1toNNotifyWait: streamId %u, taskId %u, notifyId %u", streamId_, taskId_, notifyId);
    RefreshInfo();
}

void RtsqA5::Cnt1toNNotifyRecord(u32 notifyId, u32 value)
{
    BuildA5SqeCnt1toNNotifyRecord(streamId_, taskId_, notifyId, value, GetCurrSqeBuffer());
    HCCL_INFO("RtsqA5::Cnt1toNNotifyRecord: streamId %u, taskId %u, notifyId %u", streamId_, taskId_, notifyId);
    RefreshInfo();
}

void RtsqA5::CntNto1NotifyWait(u32 notifyId, u32 value)
{
    BuildA5SqeCntNto1NotifyWait(streamId_, taskId_, notifyId, value, GetCurrSqeBuffer());
    HCCL_INFO("RtsqA5::CntNto1NotifyWait: streamId %u, taskId %u, notifyId %u", streamId_, taskId_, notifyId);
    RefreshInfo();
}

void RtsqA5::CntNto1NotifyRecord(u32 notifyId, u32 value)
{
    BuildA5SqeCntNto1NotifyRecord(streamId_, taskId_, notifyId, value, GetCurrSqeBuffer());
    HCCL_INFO("RtsqA5::CntNto1NotifyRecord: streamId %u, taskId %u, notifyId %u", streamId_, taskId_, notifyId);
    RefreshInfo();
}

void RtsqA5::SdmaCopy(u64 srcAddr, u64 dstAddr, u32 size, u32 partId)
{
    // 不带reduce的拷贝，opcode填0
    (void)partId;
    BuildA5SqeSdmaCopy(streamId_, taskId_, dstAddr, srcAddr, size, RTSQ_A5_PART_ID, 0, GetCurrSqeBuffer());
    HCCL_INFO(
        "RtsqA5::SdmaCopy: streamId %u, taskId %u, srcAddr 0x%llx, dstAddr 0x%llx, size %u", streamId_, taskId_,
        srcAddr, dstAddr, size);
    RefreshInfo();
}

const std::unordered_map<ReduceOp, RtStarsMemcpyAsyncOperationKind, EnumClassHash> ReduceOpToStarsOpKindMap
    = {{ReduceOp::SUM, RtStarsMemcpyAsyncOperationKind::RT_STARS_MEMCPY_ASYNC_OP_KIND_ADD},
       {ReduceOp::MAX, RtStarsMemcpyAsyncOperationKind::RT_STARS_MEMCPY_ASYNC_OP_KIND_MAX},
       {ReduceOp::MIN, RtStarsMemcpyAsyncOperationKind::RT_STARS_MEMCPY_ASYNC_OP_KIND_MIN},
       {ReduceOp::EQUAL, RtStarsMemcpyAsyncOperationKind::RT_STARS_MEMCPY_ASYNC_OP_KIND_EQUAL}};

const std::unordered_map<DataType, RtStarsMemcpyAsyncDataType, EnumClassHash> DataTypeToStarsDataTypeMap
    = {{DataType::INT8, RtStarsMemcpyAsyncDataType::RT_STARS_MEMCPY_ASYNC_DATA_TYPE_INT8},
       {DataType::INT16, RtStarsMemcpyAsyncDataType::RT_STARS_MEMCPY_ASYNC_DATA_TYPE_INT16},
       {DataType::INT32, RtStarsMemcpyAsyncDataType::RT_STARS_MEMCPY_ASYNC_DATA_TYPE_INT32},
       {DataType::FP16, RtStarsMemcpyAsyncDataType::RT_STARS_MEMCPY_ASYNC_DATA_TYPE_FP16},
       {DataType::FP32, RtStarsMemcpyAsyncDataType::RT_STARS_MEMCPY_ASYNC_DATA_TYPE_FP32},
       {DataType::BFP16, RtStarsMemcpyAsyncDataType::RT_STARS_MEMCPY_ASYNC_DATA_TYPE_BFP16}};

void RtsqA5::SdmaReduce(u64 srcAddr, u64 dstAddr, u32 size, u32 partId, const ReduceIn& reduceIn)
{
    (void)partId;
    if (UNLIKELY(
            ReduceOpToStarsOpKindMap.find(reduceIn.reduceOp) == ReduceOpToStarsOpKindMap.end()
            || DataTypeToStarsDataTypeMap.find(reduceIn.dataType) == DataTypeToStarsDataTypeMap.end())) {
        THROW<InternalException>(StringFormat(
            "Sdma does not support reduceOp %s dataType %s", reduceIn.reduceOp.Describe().c_str(),
            reduceIn.dataType.Describe().c_str()));
    }

    u8 op = static_cast<u8>(ReduceOpToStarsOpKindMap.at(reduceIn.reduceOp));
    u8 type = static_cast<u8>(DataTypeToStarsDataTypeMap.at(reduceIn.dataType));

    BuildA5SqeSdmaCopy(streamId_, taskId_, dstAddr, srcAddr, size, RTSQ_A5_PART_ID, (op | type), GetCurrSqeBuffer());
    HCCL_INFO(
        "RtsqA5::SdmaReduce: streamId %u, taskId %u, srcAddr 0x%llx, dstAddr 0x%llx, size %u", streamId_, taskId_,
        srcAddr, dstAddr, size);
    RefreshInfo();
}

bool RtsqA5::IsRtsqQueueSpaceSufficient()
{
    // 判断逻辑与rtsq内部保持一致，rtsq剩余空间需要大于（rtsq挂起的任务数量+本次任务）
    u32 availableSpace = GetTailToHeadDist();
    if (availableSpace > pendingSqeCnt + 1) {
        return true;
    }

    // 否则的话，需要再次查询一次head，确认是否是因为head没有更新导致空间不足，如果查询后空间仍然不足，则返回false
    sqHead_ = QuerySqHead();
    availableSpace = GetTailToHeadDist();

    return (availableSpace > pendingSqeCnt + 1);
}

HcclResult RtsqA5::SetPreStreamSyncReady()
{
    isPreStreamSync = true;
    return HCCL_SUCCESS;
}

HcclResult RtsqA5::SetPreStreamSyncFin()
{
    isPreStreamSync = false;
    return HCCL_SUCCESS;
}

bool RtsqA5::GetPreStreamSyncStatus() { return isPreStreamSync; }

void RtsqA5::UbDbSend(const UbJettyLiteId& jettyLiteId, u16 piValue)
{
    // piValue需要使用u16数据类型，保证自然增长，用于判断是否翻转
    BuildA5SqeUbDbSend(streamId_, taskId_, jettyLiteId, piValue, GetCurrSqeBuffer());
    HCCL_INFO(
        "RtsqA5::UbDbSend: streamId %u, taskId %u, piValue(UbPi):%u, SqTail(Rtsq Pi):%u", streamId_, taskId_, piValue,
        sqTail_);
    RefreshInfo();
}

void RtsqA5::RdmaDbSend(const uint64_t& dbAddr, const uint64_t& dbValue)
{
    BuildA5SqeRdmaDbSend(streamId_, taskId_, dbAddr, dbValue, GetCurrSqeBuffer());
    HCCL_INFO(
        "RtsqA5::RdmaDbSend: RdmaDbSend streamId %u, taskId %u, Sqe: %s, dbAddr:0x%llx, dbValue:0x%llx, SqTail(Rtsq "
        "Pi):%u",
        streamId_, taskId_, Bytes2hex(GetCurrSqeBuffer(), RTSQ_SQE_SIZE).c_str(), dbAddr, dbValue, sqTail_);
    RefreshInfo();
}

void RtsqA5::CCoreNotifyWait(u64 waitAddr, u64 curTurnCntAddr, bool last)
{
    BuildA5SqeCCoreNotifyWait(streamId_, taskId_, waitAddr, curTurnCntAddr, last, GetCurrSqeBuffer());
    HCCL_INFO(
        "RtsqA5::CCoreNotifyWait: streamId %u, taskId %u, waitAddr %llu, curTurnCntAddr %llu, last %d", streamId_,
        taskId_, waitAddr, curTurnCntAddr, last);
    RefreshInfo();
}

void RtsqA5::CCoreNotifyRecord(u64 recordAddr, u64 curTurnCntAddr)
{
    BuildA5SqeCCoreNotifyRecord(streamId_, taskId_, recordAddr, curTurnCntAddr, GetCurrSqeBuffer());
    HCCL_INFO(
        "RtsqA5::CCoreNotifyRecord: streamId %u, taskId %u, recordAddr %llu, curTurnCntAddr %llu", streamId_, taskId_,
        recordAddr, curTurnCntAddr);
    RefreshInfo();
}

void RtsqA5::P2PWriteValue(u64 remoteAddr, u32 writeValue)
{
    BuildA5SqeP2pWriteValue(streamId_, taskId_, remoteAddr, writeValue, GetCurrSqeBuffer());
    HCCL_INFO(
        "RtsqA5::P2PWriteValue: streamId %u, taskId %u, remoteAddr %llu, writeValue %u", streamId_, taskId_, remoteAddr,
        writeValue);
    RefreshInfo();
}

HcclResult RtsqA5::GetLastStreamIdAndTaskId(uint16_t& streamId, uint16_t& taskId) const
{
    if (pendingSqeCnt > 0) {
        const u8* lastSqe = locBuf + (pendingSqeCnt - 1U) * RTSQ_SQE_SIZE;
        auto* sqe = ReinterpretAs<const Rt91095StarsNotifySqe*>(lastSqe);
        streamId = sqe->header.rtStreamId;
        taskId = sqe->header.taskId;
        HCCL_INFO(
            "[%s] from pending, pendingSqeCnt[%u], sqId[%u], streamId[%u], taskId[%u].", __func__, pendingSqeCnt, sqId_,
            streamId, taskId);
        return HCCL_SUCCESS;
    }
    const u32 lastIdx = (sqTail_ + sqDepth_ - 1U) % sqDepth_;
    HCCL_INFO(
        "[%s] from rtsq, sqId[%u], sqTail[%u], sqDepth[%u], lastIdx[%u].", __func__, sqId_, sqTail_, sqDepth_, lastIdx);
    return GetStreamIdAndTaskIdBySqIdx(lastIdx, streamId, taskId);
}
} // namespace Hccl

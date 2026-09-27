/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_RTSQ_A5_H
#define HCCLV2_RTSQ_A5_H
#include "rtsq_base.h"
#include "sqe_v82.h"
#include "log.h"
#include <chrono>

namespace hccl {
class AicpuTsThread;
}

namespace Hccl {

class RtsqA5 : public RtsqBase {
public:
    RtsqA5(u32 devPhyId, u32 streamId, u32 sqId);

    RtsqA5(u32 devPhyId, u32 streamId, u32 sqId, bool launchFlag);

    void Reset(bool reset) override;

    void LaunchTask() override;

    void TryLaunchTask() override;

    void NotifyWait(u32 notifyId) override;

    void NotifyWait(u32 notifyId, u32 timeout);

    void NotifyRecordLoc(u32 notifyId) override;

    void Cnt1toNNotifyWait(u32 notifyId, u32 value) override;

    void Cnt1toNNotifyRecord(u32 notifyId, u32 value) override;

    void CntNto1NotifyWait(u32 notifyId, u32 value) override;

    void CntNto1NotifyRecord(u32 notifyId, u32 value) override;

    void SdmaCopy(u64 srcAddr, u64 dstAddr, u32 size, u32 partId) override;

    void SdmaReduce(u64 srcAddr, u64 dstAddr, u32 size, u32 partId, const ReduceIn& reduceIn) override;

    void P2PWriteValue(u64 remoteAddr, u32 writeValue) override;

    void UbDbSend(const UbJettyLiteId& jettyLiteId, u16 piValue) override;

    void RdmaDbSend(const uint64_t& dbAddr, const uint64_t& dbValue) override;

    void UbDirectSend(const UbJettyLiteId& jettyLiteId, u32 dwqeSize, const u8* wqe) override
    {
        // 构造UBDMA的command，这个里面，SQE可能占用 128Byte 或者 192Byte
        (void)jettyLiteId;
        (void)dwqeSize;
        (void)wqe;
    }

    void UbWriteValue(u64 dbAddr, u32 piValue) override
    {
        (void)dbAddr;
        (void)piValue;
    }

    bool IsRtsqQueueSpaceSufficient() override;

    void CCoreNotifyWait(u64 waitAddr, u64 curTurnCntAddr, bool last) override;

    void CCoreNotifyRecord(u64 recordAddr, u64 curTurnCntAddr) override;

    HcclResult SetPreStreamSyncReady() override;

    HcclResult SetPreStreamSyncFin() override;

    bool GetPreStreamSyncStatus() override;

    HcclResult GetLastStreamIdAndTaskId(uint16_t& streamId, uint16_t& taskId) const override;

    // 用于aicpu task cache
    uint32_t GetPendingSqeCnt() const override { return pendingSqeCnt; }

    inline HcclResult SetAicpuTsThreadPtr(hccl::AicpuTsThread* threadPtr)
    {
        CHK_PTR_NULL(threadPtr);
        aicpuTsThreadPtr_ = threadPtr;
        return HCCL_SUCCESS;
    }

    inline HcclResult SetNeedCacheTaskCallback(std::function<bool()> callback)
    {
        CHK_PTR_NULL(callback);
        needCacheTaskCallback_ = callback;
        return HCCL_SUCCESS;
    }

    inline HcclResult SetAddSqeArrayCallback(
        std::function<HcclResult(Hccl::RtsqA5*, hccl::AicpuTsThread*, uint64_t, const uint8_t*, const uint32_t)>
            callback)
    {
        CHK_PTR_NULL(callback);
        addSqeArrayCallback_ = callback;
        return HCCL_SUCCESS;
    }

    void RefreshSqeHeaderTaskField(Rt91095StarsSqeHeader* sqeHeaderPtr);

    void LaunchNewTask(uint8_t* sqeArray, uint32_t sqeCount);
    u64 GetSqeAddr() const override;

private:
    u32 pendingSqeCnt{0};

    u32 sqFullTimeout_ = RTSQ_FULL_TIMEOUT_DEFAULT;

    bool isPreStreamSync = false;

    bool launchFlag_ = false;

    u64 lastSqeAddr_{0};

    u8 locBuf[RTSQ_SQE_SIZE * PER_LAUNCH_SQE_CNT]{0};

    std::function<bool()> needCacheTaskCallback_{nullptr};
    std::function<HcclResult(Hccl::RtsqA5*, hccl::AicpuTsThread*, uint64_t, const uint8_t*, const uint32_t)>
        addSqeArrayCallback_{nullptr};
    hccl::AicpuTsThread* aicpuTsThreadPtr_{nullptr};

    u8* GetCurrSqeBuffer();

    void RefreshInfo();

    void CopySqeBufToSq(u8* sqeBuf) const;

    void MakeSureAvailableSpace();

    u32 GetTailToHeadDist() const;

    void CheckLaunchTaskStatus(
        const std::chrono::steady_clock::time_point& startTime, const std::chrono::steady_clock::time_point& curTime);

    void PreLaunchSqeForCache(bool& needCacheTask);

    void PostLaunchSqeForCache();
};

} // namespace Hccl

#endif

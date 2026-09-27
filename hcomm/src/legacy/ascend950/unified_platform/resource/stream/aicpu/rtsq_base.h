/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_RTSQ_BASE_H
#define HCCLV2_RTSQ_BASE_H
#include "types.h"
#include "buffer.h"
#include "notify_lite.h"
#include "reduce_op.h"
#include "data_type.h"
#include "reduce_in.h"
#include "not_support_exception.h"
#include "ub_jetty_lite.h"
#include <functional>

#include "ascend_hal.h"
namespace aicpu {
void __attribute__((weak)) __attribute__((visibility("default")))
GetSqeId(const uint32_t num, uint32_t& start, uint32_t& end);
}

namespace Hccl {

constexpr u32 RTSQ_FULL_TIMEOUT_DEFAULT = 1836 + 20;
constexpr u32 RTSQ_SQE_SIZE = 64;
constexpr u32 PER_LAUNCH_SQE_CNT = 128;

class RtsqBase {
public:
    RtsqBase(u32 devPhyId, u32 streamId, u32 sqId);

    virtual ~RtsqBase() = default;

    virtual void Reset(bool reset);

    inline u32 GetStreamId() const { return streamId_; }

    inline u32 GetSqDepth() const { return sqDepth_; }

    inline u32 GetHead() const { return sqHead_; }

    inline u32 GetTail() const { return sqTail_; }

    inline u32 GetTaskId() const { return taskId_; }

    void SetOpExecStatusCallback(std::function<void()> callback) { checkOpExecStatusCallback_ = callback; }

    void
    SetCheckExecStatusCallback(std::function<HcclResult(bool)> callback) // 自定义算子流程注册检查执行状态的回调函数
    {
        checkExecStatusCallback_ = callback;
    }

    virtual void LaunchTask() { MACRO_THROW(NotSupportException, StringFormat("not supported.")); }

    virtual void TryLaunchTask() { MACRO_THROW(NotSupportException, StringFormat("not supported.")); }

    virtual void NotifyWait(u32 notifyId)
    {
        (void)notifyId;
        MACRO_THROW(NotSupportException, StringFormat("not supported."));
    }

    virtual void NotifyWait(u32 notifyId, u32 timeout)
    {
        (void)notifyId;
        (void)timeout;
        MACRO_THROW(NotSupportException, StringFormat("not supported."));
    }

    virtual void Cnt1toNNotifyWait(u32 notifyId, u32 value)
    {
        (void)notifyId;
        (void)value;
        MACRO_THROW(NotSupportException, StringFormat("not supported."));
    }

    virtual void Cnt1toNNotifyRecord(u32 notifyId, u32 value)
    {
        (void)notifyId;
        (void)value;
        MACRO_THROW(NotSupportException, StringFormat("not supported."));
    }

    virtual void CntNto1NotifyWait(u32 notifyId, u32 value)
    {
        (void)notifyId;
        (void)value;
        MACRO_THROW(NotSupportException, StringFormat("not supported."));
    }

    virtual void CntNto1NotifyRecord(u32 notifyId, u32 value)
    {
        (void)notifyId;
        (void)value;
        MACRO_THROW(NotSupportException, StringFormat("not supported."));
    }

    virtual void NotifyRecordLoc(u32 notifyId)
    {
        (void)notifyId;
        MACRO_THROW(NotSupportException, StringFormat("not supported."));
    }

    virtual void NotifyRecordRmt(u32 rmtDevPhyId, u32 notifyId) // 仅 P2P 使用
    {
        (void)rmtDevPhyId;
        (void)notifyId;
        MACRO_THROW(NotSupportException, StringFormat("not supported."));
    }

    virtual void SdmaCopy(u64 srcAddr, u64 dstAddr, u32 size, u32 partId)
    {
        (void)srcAddr;
        (void)dstAddr;
        (void)size;
        (void)partId;
        MACRO_THROW(NotSupportException, StringFormat("not supported."));
    }

    virtual void SdmaReduce(u64 srcAddr, u64 dstAddr, u32 size, u32 partId, const ReduceIn& reduceIn)
    {
        (void)srcAddr;
        (void)dstAddr;
        (void)size;
        (void)partId;
        (void)reduceIn;
        MACRO_THROW(NotSupportException, StringFormat("not supported."));
    }

    virtual void P2PWriteValue(u64 remoteAddr, u32 writeValue)
    {
        (void)remoteAddr;
        (void)writeValue;
        MACRO_THROW(NotSupportException, StringFormat("not supported."));
    }

    virtual void UbDbSend(const UbJettyLiteId& jettyLiteId, u16 piValue)
    {
        (void)jettyLiteId;
        (void)piValue;
        MACRO_THROW(NotSupportException, StringFormat("not supported."));
    }

    virtual void RdmaDbSend(const uint64_t& dbAddr, const uint64_t& dbValue)
    {
        (void)dbAddr;
        (void)dbValue;
        MACRO_THROW(NotSupportException, StringFormat("not supported."));
    }

    virtual void UbDirectSend(const UbJettyLiteId& jettyLiteId, u32 dwqeSize, const u8* wqe)
    {
        (void)jettyLiteId;
        (void)dwqeSize;
        (void)wqe;
        MACRO_THROW(NotSupportException, StringFormat("not supported."));
    }

    virtual void UbWriteValue(u64 dbAddr, u32 piValue)
    {
        (void)dbAddr;
        (void)piValue;
        MACRO_THROW(NotSupportException, StringFormat("not supported."));
    }

    virtual void CCoreNotifyWait(u64 waitAddr, u64 curTurnCntAddr, bool last)
    {
        (void)waitAddr;
        (void)curTurnCntAddr;
        (void)last;
        MACRO_THROW(NotSupportException, StringFormat("not supported."));
    }

    virtual void CCoreNotifyRecord(u64 recordAddr, u64 curTurnCntAddr)
    {
        (void)recordAddr;
        (void)curTurnCntAddr;
        MACRO_THROW(NotSupportException, StringFormat("not supported."));
    }

    virtual u64 GetSqeAddr() const { return 0; }

    virtual uint32_t GetPendingSqeCnt() const { return 0; }

    u32 QuerySqHead() const;
    u32 QuerySqTail() const;

    virtual bool IsRtsqQueueSpaceSufficient() { return true; }

    virtual HcclResult SetPreStreamSyncReady()
    {
        MACRO_THROW(NotSupportException, StringFormat("not supported."));
        return HCCL_SUCCESS;
    }

    virtual HcclResult SetPreStreamSyncFin()
    {
        MACRO_THROW(NotSupportException, StringFormat("not supported."));
        return HCCL_SUCCESS;
    }

    virtual bool GetPreStreamSyncStatus() { return false; }

    HcclResult GetStreamIdAndTaskIdBySqIdx(u32 sqIdx, uint16_t& streamId, uint16_t& taskId) const;

    virtual HcclResult GetLastStreamIdAndTaskId(uint16_t& streamId, uint16_t& taskId) const
    {
        (void)streamId;
        (void)taskId;
        return HCCL_E_NOT_SUPPORT;
    }

protected:
    u32 devPhyId_{0};
    u32 localDevId_{0};
    u32 streamId_{0}; // 填写到SQE中的streamId
    u32 sqId_{0};

    u32 sqHead_{0};
    u32 sqTail_{0};
    u32 sqDepth_{0};
    u64 sqBaseAddr_{0};

    u32 taskId_{0};    // 填写到SQE中的taskId，现改为由AICPU组件提供的sqeId维护
    u32 taskIdEnd_{0}; // 当前流已经申请到的最大taskId

    std::function<void()> checkOpExecStatusCallback_{nullptr};
    std::function<HcclResult(bool)> checkExecStatusCallback_{nullptr}; // 自定义算子流程，检查执行状态

    u32 QuerySqDepth() const;

    std::string GetHwSqDescribe() const;

    void ConfigSqTail(u32 value) const;
    void ConfigDisableToEnable(u32 value) const;

    inline void SetTaskIdBySqeId()
    {
        taskId_++; // taskId_的范围是aicpu::GetSqeId返回的[start, end), taskId累加到end时重新向aicpu申请, 不会翻转
        if (UNLIKELY(taskId_ >= taskIdEnd_)) {                        // taskEnd_视为未申请的taskId，不可使用
            constexpr u32 PER_GET_SQE_ID_NUM = 1024;                  // 一次性申请sqeId数量
            aicpu::GetSqeId(PER_GET_SQE_ID_NUM, taskId_, taskIdEnd_); // aicpu框架保证 taskId_ < taskIdEnd_
        }
        return;
    }

private:
    u64 QuerySqBaseAddr() const;
    u32 QueryCqeStatus() const;

    u32 QuerySqStatusByType(drvSqCqPropType_t givenType) const;
    void ConfigSqStatusByType(drvSqCqPropType_t givenType, u32 value) const;
};

} // namespace Hccl

#endif

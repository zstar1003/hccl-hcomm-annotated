/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ROCE_TRANSPORT_LITE_IMPL_H
#define ROCE_TRANSPORT_LITE_IMPL_H

#include <vector>
#include <memory>
#include "base_transport_lite_impl.h"
#include "notify_lite.h"
#include "rma_buffer_lite.h"
#include "rmt_rma_buffer_lite.h"
#include "rdma_conn_lite_v2.h"

namespace Hccl {

class RoceTransportLiteImpl : public BaseTransportLiteImpl {
public:
    explicit RoceTransportLiteImpl(std::vector<char>& uniqueId);
    RoceTransportLiteImpl() = default;
    ~RoceTransportLiteImpl() override;

    void Init(std::vector<char>& uniqueId);

    std::string Describe() const override;

    // ========== Buffer 构造接口 ==========
    HcclResult BuildLocRmaBufferLite(const uintptr_t addr, const size_t size, RmaBufferLite& rmaBufferLite) override;

    // ========== RMA 数据传输接口 ==========
    void Read(const RmaBufferLite& loc, const Buffer& rmt, const StreamLite& stream) override;
    void Write(const RmaBufferLite& loc, const Buffer& rmt, const StreamLite& stream) override;
    void WriteReduce(
        const RmaBufferLite& loc, const Buffer& rmt, const ReduceIn& reduceIn, const StreamLite& stream) override;
    void WriteWithNotify(
        const RmaBufferLite& loc, const Buffer& rmt, const WithNotifyIn& withNotify, const StreamLite& stream) override;
    void WriteReduceWithNotify(
        const RmaBufferLite& loc, const Buffer& rmt, const ReduceIn& reduceIn, const WithNotifyIn& withNotify,
        const StreamLite& stream) override;

    // ========== 同步 / Notify 接口 ==========
    HcclResult Fence() override;
    void Post(u32 index, const StreamLite& stream) override;
    void WaitWithTimeout(u32 index, const StreamLite& stream, u32 timeout) override;

    // ========== 错误上报 接口 ==========
    HcclResult PollCq(int32_t numEntries, int32_t timeOut, std::vector<int32_t>& errList);

    // 设置TaskException开关
    void SetTaskExceptionEnable(bool flag) { taskExceptionEnable_ = flag; }

private:
    u32 notifyNum_{0};
    u32 bufferNum_{0};
    u32 connNum_{0};

    bool fence_{false};
    bool taskExceptionEnable_{true};

    std::vector<std::unique_ptr<NotifyLite>> localNotifies_{};
    std::vector<RmtRmaBufferLite> remoteNotifies_{};
    std::vector<RmaBufferLite> locBufferVec_{};
    std::vector<RmtRmaBufferLite> rmtBufferVec_{};
    std::vector<std::vector<char>> connUniqueIdVec_{};
    std::vector<std::unique_ptr<RdmaConnLiteV2>> connVec_{};
    std::unique_ptr<RmaBufferLite> notifyValueBuffer_{};

    RmaBufSliceLite GetRmaBufSlicelite(const RmaBufferLite& lite) const;
    RmaBufSliceLite GetNotifySlicelite(u32 index) const;
    RmtRmaBufSliceLite GetRmtRmaBufSliceLite(const Buffer& rmtBuf) const;
    RmtRmaBufSliceLite GetRmtNotifySliceLite(u32 index) const;
    void SetFenceConfig(SqeConfigLite& cfg);

    void ParseLocNotifyVec(std::vector<char>& data);
    void ParseRmtNotifyVec(std::vector<char>& data);
    void ParseNotifyValueBuffer(std::vector<char>& data);
    void ParseLocBufferVec(std::vector<char>& data);
    void ParseRmtBufferVec(std::vector<char>& data);
    void ParseConnVec(std::vector<char>& data);

    // ========== 底层 Task 构造接口(rtsq) ==========
    void BuildRdmaDbSendTask(const StreamLite& stream, u64 remoteAddr, u64 dbValue) const;
    void BuildNotifyWaitTask(u32 notifyId, const StreamLite& stream, u32 timeout) const;

    // ========== Profiling接口 ==========
    void ReportDmaTask(
        const void* src, const void* dst, u64 size, const StreamLite& stream, u32 taskId, TaskParamType taskType,
        DmaOp dmaOp, u64 notifyId, u32 notifyValue, const char* funcName) const;
    void ReportReduceTask(
        const void* src, const void* dst, u64 size, const ReduceIn& reduceIn, const StreamLite& stream, u32 taskId,
        TaskParamType taskType, u64 notifyId, u32 notifyValue, const char* funcName) const;
    void ReportNotifyWaitTask(u64 notifyId, const StreamLite& stream, u32 taskId) const;
    bool IsReportTask() const;
};

} // namespace Hccl
#endif

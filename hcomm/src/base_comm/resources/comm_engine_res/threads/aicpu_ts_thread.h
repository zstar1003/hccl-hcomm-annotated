/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_TS_THREAD_H
#define AICPU_TS_THREAD_H

#include <vector>
#include <functional>
#include "thread.h"
#include "aicpu_ts_thread_interface.h"
#include "dfx_circular_queue.h"

namespace hccl {
class AicpuTsThread : public Thread {
public:
    AicpuTsThread(StreamType streamType, uint32_t notifyNum, const NotifyLoadType notifyLoadType);
    AicpuTsThread(const std::string& uniqueIdStr);

    ~AicpuTsThread();

    HcclResult Init() override;
    HcclResult DeInit() override;
    std::string& GetUniqueId() override;
    uint32_t GetNotifyNum() const override;
    LocalNotify* GetNotify(uint32_t index) const override;
    HcclResult GetNotifyByUniqueId(u32& notifyNum, std::string& notifyDesc);
    HcclResult SupplementNotify(uint32_t notifyNum) override;
    HcclResult SupplementNotify(u32 notifyNum, const std::string& notifyDesc);

    // A3 Stream & A5 Stream
    inline bool IsDeviceA5() const override
    {
        return devType_ == DevType::DEV_TYPE_950 || devType_ == DevType::DEV_TYPE_960;
    }

    Stream* GetStream() const override;

    inline void* GetStreamLitePtr() const override { return pImpl_ ? pImpl_->GetStreamLitePtr() : nullptr; }

    void LaunchTask() const override;
    void TryLaunchTask() const override;

    // Local Data Plane Functions
    HcclResult LocalNotifyWait(uint32_t notifyId) const override;
    HcclResult LocalNotifyRecord(uint32_t notifyId) const override;

    HcclResult LocalNotifyRecord(ThreadHandle dstThread, uint32_t dstNotifyIdx) const override;
    HcclResult LocalNotifyWait(uint32_t notifyId, uint32_t timeout) const override;

    HcclResult LocalCopy(void* dst, const void* src, uint64_t size) const override;
    HcclResult LocalReduce(
        void* dst, const void* src, uint64_t size, HcommDataType dataType, HcommReduceOp reduceOp) const override;

    // Non-override functions
    HcclResult GetSqHeadAndTail(uint32_t& sqHead, uint32_t& sqTail);
    bool GetMaster() const override;
    void SetIsMaster(bool isMaster) override;

    HcclResult SetCheckExecStatusCallback(std::function<HcclResult(bool)> callback) override;

    Hccl::TaskInfoCircularQueue* GetTaskInfos() const;
    HcclResult GetTaskInfoCount(u32& count) const;
    void SetReportStreamTaskCallback(std::function<void(Hccl::TaskInfoCircularQueue*)> callback) const;

    void SetGetLatestDfxOpInfoCallback(std::function<const void*()> callback) const;

private:
    struct HcclStreamInfo {
        s32 streamIds;
        uint32_t sqIds;
        uint32_t cqIds;      // 记录物理cqId
        uint32_t logicCqids; // 记录逻辑cqId
    };

    struct HcclStreamParam {
        HcclStreamInfo streamInfo;
        uint64_t sqCqContextAddr = 0; // 记录sqeContext地址
        uint64_t sqCqContextSize = 0; // 记录sqeContext大小
    };
    HcclResult InitStreamLite(HcclStreamInfo& streamParam, uint32_t hostPhyId);
    HcclResult InitStream(HcclStreamParam& streamParam);
    HcclResult HostInit();
    HcclResult DeviceInit();
    std::string& UpdateUniqueId();
    HcclResult DeInitImpl();
#ifdef CCL_KERNEL_AICPU
    HcclResult BuildComStreamInfo(const HcclStreamInfo& streamInfo, HcclComStreamInfo& comStreamInfo) const;
#endif
    HcclResult LocalCopyReport(uint32_t taskId, Hccl::StreamLite* sl, Hccl::RtsqBase* rtsq) const;
    HcclResult LocalReduceReport(
        void* dst, const void* src, uint64_t size, HcommReduceOp reduceOp, uint32_t taskId, Hccl::StreamLite* sl,
        Hccl::RtsqBase* rtsq) const;

    // 成员变量（适配 AICPU-TS）
    bool isDeviceSide_ = false;
    rtStream_t rtStream_ = nullptr;
    StreamType streamType_ = StreamType::STREAM_TYPE_RESERVED;
    uint32_t notifyNum_ = 0;
    NotifyLoadType notifyLoadType_ = NotifyLoadType::HOST_NOTIFY;
    uint32_t devId_ = INVALID_UINT;
    std::unique_ptr<Stream> stream_ = nullptr;
    std::vector<std::unique_ptr<LocalNotify>> notifys_;
    std::string uniqueIdStr_;
    DeviceMem sqCqeContext_;
    DevType devType_ = DevType::DEV_TYPE_COUNT;
    std::unique_ptr<Hccl::IAicpuTsThread> pImpl_{nullptr};
    bool isMaster_{false};
};

} // namespace hccl
#endif // AICPU_TS_THREAD_H

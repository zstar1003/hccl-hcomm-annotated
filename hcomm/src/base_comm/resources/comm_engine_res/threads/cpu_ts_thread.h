/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CPU_TS_THREAD_H
#define HCCL_CPU_TS_THREAD_H

#include "thread.h"

namespace hccl {

class CpuTsThread : public Thread {
public:
    CpuTsThread(rtStream_t rtStream, uint32_t notifyNum, const NotifyLoadType notifyLoadType);
    CpuTsThread(StreamType streamType, uint32_t notifyNum, const NotifyLoadType notifyLoadType);

    ~CpuTsThread();

    HcclResult Init() override;
    HcclResult DeInit() override;
    std::string& GetUniqueId() override;
    uint32_t GetNotifyNum() const override;
    LocalNotify* GetNotify(uint32_t index) const override;
    HcclResult SupplementNotify(uint32_t notifyNum) override;

    // A3 Stream & A5 Stream
    bool IsDeviceA5() const override;
    Stream* GetStream() const override;
    void* GetStreamLitePtr() const override;
    void LaunchTask() const override;
    void TryLaunchTask() const override;

    // Local Data Plane Functions
    HcclResult LocalNotifyRecord(uint32_t notifyId) const override;
    HcclResult LocalNotifyWait(uint32_t notifyId) const override;

    HcclResult LocalNotifyRecord(ThreadHandle dstThread, uint32_t dstNotifyIdx) const override;
    HcclResult LocalNotifyWait(uint32_t notifyIdx, uint32_t timeOut) const override;

    HcclResult LocalCopy(void* dst, const void* src, uint64_t sizeByte) const override;
    HcclResult LocalReduce(
        void* dst, const void* src, uint64_t sizeByte, HcommDataType dataType, HcommReduceOp reduceOp) const override;
    bool GetMaster() const override;
    void SetIsMaster(bool isMaster) override;

private:
    bool isMaster_{false};
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
    std::string& UpdateUniqueId();
    HcclResult DeInitImpl();
    rtStream_t rtStream_ = nullptr;
    bool isDeviceSide_ = false;
    StreamType streamType_ = StreamType::STREAM_TYPE_RESERVED;
    uint32_t notifyNum_ = 0;
    uint32_t devId_ = INVALID_UINT;
    NotifyLoadType notifyLoadType_ = NotifyLoadType::HOST_NOTIFY;
    std::unique_ptr<Stream> stream_;
    DevType devType_ = DevType::DEV_TYPE_COUNT;
    std::vector<std::unique_ptr<LocalNotify>> notifys_;

    std::unique_ptr<Stream> streamDevice_; // 在把用户的stream生成的thread导出到device时使用
    DeviceMem sqCqeContext_;
    std::string uniqueIdStr_;
};

} // namespace hccl

#endif // HCCL_CPU_TS_THREAD_H

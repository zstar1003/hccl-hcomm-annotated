/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_TS_THREAD_INTERFACE_H
#define AICPU_TS_THREAD_INTERFACE_H

#include "hccl_types.h"

namespace Hccl {

class IAicpuTsThread {
public:
    IAicpuTsThread(uint32_t id, uint32_t sqIds, uint32_t phyId, uint32_t cqIds)
    {
        streamLiteVoidPtr_ = reinterpret_cast<void*>(0x3344);
    }

    ~IAicpuTsThread() {}

    HcclResult NotifyWait(uint32_t notifyId) const { return HCCL_SUCCESS; }

    HcclResult NotifyWait(uint32_t notifyId, uint32_t timeout) const { return HCCL_SUCCESS; }

    HcclResult NotifyRecordLoc(uint32_t notifyId) const { return HCCL_SUCCESS; }

    HcclResult SdmaCopy(uint64_t dstAddr, uint64_t srcAddr, uint64_t sizeByte) const { return HCCL_SUCCESS; }

    HcclResult
    SdmaReduce(uint64_t dstAddr, uint64_t srcAddr, uint64_t sizeByte, uint32_t dataTypeRaw, uint32_t reduceOpRaw) const
    {
        return HCCL_SUCCESS;
    }

    inline void* GetStreamLitePtr() const { return streamLiteVoidPtr_; }

    void LaunchTask() const {}

    void TryLaunchTask() const {}

    inline uint32_t GetSqId() const { return 0; }

private:
    void* streamLiteVoidPtr_ = nullptr;
};

} // namespace Hccl

#endif

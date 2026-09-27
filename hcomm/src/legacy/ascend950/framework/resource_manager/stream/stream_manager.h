/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_STREAM_MANAGER_H
#define HCCLV2_STREAM_MANAGER_H

#include <memory>
#include "stream.h"
#include "opbase_stream_manager.h"
#include "offload_stream_manager.h"

namespace Hccl {
class CommunicatorImpl;

class StreamManager {
public:
    explicit StreamManager(CommunicatorImpl* comm);

    Stream* GetSlave() const;
    Stream* GetMaster() const;

    Stream* GetSlaveByIndex(u32 index) const;

    u32 GetSlaveIndex() const;
    void ResetSlaveIndex(u32 index) const;

    CommunicatorImpl* comm{nullptr};
    std::unique_ptr<OpbaseStreamManager> opbase{nullptr};
    std::unique_ptr<OffloadStreamManager> offload{nullptr};

    void CaptureSlaveStream(const Stream* masterStream, const Stream* slaveStream) const;

    // CCU单Task多流管理使用
    u32 GetStreamIndex(u32 streamId);

    void RecordStreamIdToIndex(u32 streamId, u32 streamIndex);

    void InitBucket(u32 bucket);

    void RegisterBucket(u32 bucket, u32 subStreamIndex);

    std::vector<u32>& GetSubSlaveIndexes(u32 slaveIndex);

    void DestroyRecords();

private:
    // 用于维护CCU流交替下发的桶
    std::unordered_map<u32, std::vector<u32>> streamBucket_{};
    std::unordered_map<u32, u32> streamIdToIndexMap_{};
};

} // namespace Hccl
#endif

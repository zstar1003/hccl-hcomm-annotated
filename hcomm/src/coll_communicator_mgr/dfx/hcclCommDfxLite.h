/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_COMM_DFX_LITE_H
#define HCCL_COMM_DFX_LITE_H
#include "hcclCommProfilingLite.h"
#include "hccl_common.h"
#include "buffer.h"
#include "common.h"
#include "res_pub.h"
#include "hcclCommOp.h"
#include <vector>

namespace hccl {
// NOTE: HcclCommDfxLite is designed for AICPU single-threaded environments.
// channelRemoteRankIdLite_ and its access methods (AddChannelRemoteRankId/GetChannelRemoteRankId)
// are NOT thread-safe. If used in multi-threaded environments, external synchronization is required.
class HcclCommDfxLite {
public:
    explicit HcclCommDfxLite();
    ~HcclCommDfxLite();

    HcclResult Init(u32 deviceId, const std::string& commTag, u32 rankSize, u32 localRank);
    void ReportAllTasks(const std::vector<hccl::Thread*>& threads);
    HcclResult ReportStreamTask(Hccl::TaskInfoCircularQueue* taskQueue);
    HcclResult UpdateProfStat();
    HcclResult SetCurrDfxOpInfo(const Hccl::DfxDfxOpInfo* newDfxOpInfo);
    const void* GetLatestDfxOpInfo() const;
    void AddChannelRemoteRankId(u64 handle, u32 remoteRankId);
    u32 GetChannelRemoteRankId(u64 handle) const;
    Hccl::DfxCommContext GetDfxCommContext() const;

private:
    HcclCommProfilingLite* profilingImpl_{nullptr};
    void* opInfoQueue_{nullptr};
    bool queueInitialized_{false};
    std::unordered_map<u64, u32> channelRemoteRankIdLite_{};
    u64 groupNameHash_{DFX_INVALID_U64};
    std::string commTag_{};
    u32 deviceId_{0};
    u32 rankSize_{0};
    u32 localRank_{0};
    bool initializedFlag_{false};
};
} // namespace hccl
#endif // HCCL_COMM_DFX_LITE_H

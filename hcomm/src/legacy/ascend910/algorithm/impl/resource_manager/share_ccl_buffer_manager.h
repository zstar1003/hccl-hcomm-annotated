/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <mutex>
#include <string>
#include <memory>
#include <unordered_map>
#include "hccl_common.h"
#include "mem_device_pub.h"

namespace hccl {
struct ShareCCLMem {
    DeviceMem cclBuffer = DeviceMem();
    uint64_t refCount{0}; // 引用计数
};

// 进程粒度的内存管理单例
class ShareCCLbufferMgr {
public:
    static ShareCCLbufferMgr& GetInstance(); // 获取单例
    ~ShareCCLbufferMgr() = default;

    HcclResult RecordShareCCLbuffer(const std::string& bufferName);
    HcclResult CreateShareCCLbuffer(const std::string& bufferName, u64 bufferSize, DeviceMem& cclBuffer);
    HcclResult FreeShareCCLbuffer(const std::string& bufferName);
    HcclResult CheckCCLbuffConflict(const std::string& bufferName, s32 streamId);

private:
    HcclResult CreateDevMem(u64 size, DeviceMem& buffer);

    std::mutex lock_;                                        // 锁保证多线程访问安全
    u64 shareBufferSize_ = 0;                                // 共享buffer大小
    std::unordered_map<std::string, s32> streamIdMap_;       // bufferName与streamID映射关系
    std::unordered_map<std::string, ShareCCLMem> memRecord_; // 记录内存的次数
};
} // namespace hccl

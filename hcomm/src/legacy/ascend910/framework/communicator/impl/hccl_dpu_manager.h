/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_DPU_MANAGER_H
#define HCCL_DPU_MANAGER_H

#include "acl/acl_rt.h"
#include "hccl/hccl_types.h"
#include <memory>
#include <string>
#include <unordered_map>
#include "config_log.h"
#include "../../../unified_platform/resource/buffer/dev_buffer.h"

namespace hccl {

constexpr u32 SHARE_HBM_MEMORY_SIZE = 100 * 1024 * 1024; // 100MB
constexpr char DPUTAG[] = "DPUTAG";

class DpuManager {
public:
    DpuManager();
    ~DpuManager();

    HcclResult Init(const std::string& commId, u32 deviceLogicId);
    HcclResult InitDpuKernel();
    HcclResult DeInitDpuKernel();

    bool IsDpuKernelLaunched() const { return isDpuKernelLaunched_; }

    HcclResult CreateWorkspaceBuf(const char* memTag, uint64_t* size, bool* newCreated);
    std::shared_ptr<Hccl::DevBuffer> GetKFCWorkSpace(const char* memTag);
    HcclResult GetDevMemWorkSpace(const std::string& memTag, uint64_t* size, void** addr, bool* newCreated);
    HcclResult GetKFCWorkSpaceVA(const std::string& memTag, uint64_t* size, void** addr, bool* newCreated);
    HcclResult AllocAndRegKFCWorkSpace(uint64_t size);
    HcclResult DestroyKFCWorkSpaceVA();
    HcclResult DestroyDpuKernelResource();

private:
    HcclResult InitAndLaunchDpuKernel();
    HcclResult PrepareDpuKernelResource(aclrtFuncHandle& funcHandle);
    HcclResult LaunchDpuKernel(aclrtFuncHandle& funcHandle);
    HcclResult WaitDpuKernelThreadTerminate();

    std::string commId_;
    u32 devLogicId_{0};
    void* hostShareBuf_{nullptr};
    void* va_{nullptr};
    void* accessVA_{nullptr};
    std::unordered_map<std::string, std::shared_ptr<Hccl::DevBuffer>> tagWorkspaceVAMap_;
    aclrtStream dpuStream_{nullptr};
    aclrtContext dpuContext_{nullptr};
    aclrtContext npuContext_{nullptr};
    bool isDpuKernelLaunched_{false};
    u32 waitTransportReadyTimeoutMs_{200 * 1000}; // 默认200秒

    std::unordered_map<std::string, std::shared_ptr<Hccl::DevBuffer>> tagWorkspaceMap_;
};
} // namespace hccl

#endif // HCCL_DPU_MANAGER_H

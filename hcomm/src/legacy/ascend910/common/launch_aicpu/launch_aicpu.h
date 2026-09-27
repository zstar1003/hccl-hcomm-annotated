/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LAUNCH_AICPU_H
#define LAUNCH_AICPU_H

#include "hccl/base.h"
#include "hccl_types.h"
#include "acl/acl_rt.h"

namespace hccl {
HcclResult AicpuAclKernelLaunch(
    const rtStream_t stm, void* addr, u32 size, aclrtBinHandle binHandle, const std::string& kernelName,
    bool isInitTask, u16 timeOut, void* tilingDataPtr = nullptr, u32 tilingDataSize = 0);
HcclResult AicpuAclKernelLaunchV2(
    const rtStream_t stm, void* addr, u32 size, aclrtBinHandle binHandle, const std::string& kernelName,
    bool isInitTask, u16 timeOut, void* tilingDataPtr, u32 tilingDataSize, const std::string& identify);
HcclResult GetKernelFilePath(std::string& binaryPath);
HcclResult GetCustomKernelFilePath(std::string& binaryPath);
HcclResult CacheTaskOpInfo(aclrtStream stream, const std::string& identify);
} // namespace hccl

#endif // LAUNCH_AICPU_H

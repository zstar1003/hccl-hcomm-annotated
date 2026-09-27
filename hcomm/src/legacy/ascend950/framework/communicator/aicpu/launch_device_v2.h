/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LEGACY_LAUNCH_DEVICE_H
#define LEGACY_LAUNCH_DEVICE_H

#include <array>
#include <unordered_map>
#include "acl/acl_rt.h"
#include "kernel_param_lite.h"
namespace Hccl {

void LoadBinaryFromFile(
    const char* binPath, aclrtBinaryLoadOptionType optionType, uint32_t cpuKernelMode, aclrtBinHandle& binHandle);
void GetKernelFilePath(std::string& binaryPath);
class AicpuBinaryHolder {
public:
    AicpuBinaryHolder();
    ~AicpuBinaryHolder() noexcept;
    void Load();
    void Unload();
    // 调用前请检查是否Load期间有注册对应的kernelName funcHandle
    aclrtFuncHandle GetAicpuKernelFuncHandle(const char* kernelName) const;

private:
    aclrtBinHandle handle_;
    bool loaded_;
    std::unordered_map<std::string, aclrtFuncHandle> aicpuFuncMap_;
};
} // namespace Hccl
#endif // LEGACY_LAUNCH_DEVICE_H

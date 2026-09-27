/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <string>
#include "launch_device_v2.h"
#include "log.h"
#include "mmpa_api.h"
#include "sal.h"
#include "exception_util.h"
#include "invalid_params_exception.h"
#include "runtime_api_exception.h"
#include "hccl/base.h"

using namespace std;

namespace Hccl {
void GetKernelFilePath(std::string& binaryPath)
{
    std::string libPath = SalGetEnv("ASCEND_HOME_PATH");
    if (libPath.empty() || libPath == "EmptyString") {
        HCCL_WARNING("[GetKernelFilePath]ENV:ASCEND_HOME_PATH is not set, use default:/usr/local/Ascend/cann/");
        libPath = "/usr/local/Ascend/cann/";
    }
    libPath += "/opp/built-in/op_impl/aicpu/config/";
    binaryPath = libPath;
    HCCL_DEBUG("[GetKernelFilePath]kernel folder path[%s]", binaryPath.c_str());
}

void LoadBinaryFromFile(
    const char* binPath, aclrtBinaryLoadOptionType optionType, uint32_t cpuKernelMode, aclrtBinHandle& binHandle)
{
    if (binPath == nullptr) {
        THROW<InvalidParamsException>(StringFormat("[LoadBinaryFromFile]binary path is nullptr", binPath));
    }
    char realPath[PATH_MAX] = {0};
    if (realpath(binPath, realPath) == nullptr) {
        THROW<InvalidParamsException>(StringFormat(
            "[LoadBinaryFromFile]binPath:%s is not a valid real path,"
            "err[%d]",
            binPath, errno));
    }
    HCCL_INFO("[LoadBinaryFromFile]realPath: %s", realPath);

    aclrtBinaryLoadOptions loadOptions = {};
    aclrtBinaryLoadOption option;
    loadOptions.numOpt = 1;
    loadOptions.options = &option;
    option.type = optionType;
    option.value.cpuKernelMode = cpuKernelMode;
    // ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE
    aclError aclRet = aclrtBinaryLoadFromFile(realPath, &loadOptions, &binHandle);
    if (aclRet != ACL_SUCCESS) {
        THROW<RuntimeApiException>(
            StringFormat("[LoadBinaryFromFile]:errNo[0x%016llx]load binary from file error.", aclRet));
    }
}

AicpuBinaryHolder::AicpuBinaryHolder() : handle_(nullptr), loaded_(false) {}

AicpuBinaryHolder::~AicpuBinaryHolder() noexcept { Unload(); }

namespace {
    struct LoadCleanupGuard {
        explicit LoadCleanupGuard(AicpuBinaryHolder& holder) : holder_(holder), active_(true) {}

        ~LoadCleanupGuard()
        {
            if (active_) {
                holder_.Unload();
            }
        }

        void Dismiss() { active_ = false; }

        AicpuBinaryHolder& holder_;
        bool active_;
    };
} // namespace

void AicpuBinaryHolder::Load()
{
    if (loaded_) {
        HCCL_WARNING("[AicpuBinaryHolder::%s] has registered aicpu kernel, skip register again.", __func__);
        return;
    }
    HCCL_INFO("[AicpuBinaryHolder::%s] start.", __func__);
    std::string jsonPath;
    GetKernelFilePath(jsonPath);
    jsonPath += "ccl_kernel.json";
    aclrtBinHandle tempHandle = nullptr;
    LoadBinaryFromFile(jsonPath.c_str(), ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE, 0, tempHandle);
    handle_ = tempHandle;
    loaded_ = true; // 提前设置loaded_为true，确保下方触发异常后aicpuKernelGuard里面能正确释放handle_资源
    LoadCleanupGuard aicpuKernelGuard(*this);
    HCCL_INFO("[AicpuBinaryHolder::%s] LoadBinaryFromFile success [%s]", __func__, jsonPath.c_str());
    // register base Func
    constexpr std::array<const char*, 3> kernelFunction{
        "HcclKernelEntrance", "HcclUpdateCommKernelEntrance", "HcclDpuTaskexpShmemRestore"};
    for (const auto& kernelName : kernelFunction) {
        if (strlen(kernelName) == 0 || strlen(kernelName) >= KERNEL_PARAM_NAME_SIZE) {
            HCCL_ERROR("[AicpuBinaryHolder::%s] invalid kernel name", __func__);
            THROW<InvalidParamsException>("kernel name is invalid");
        }
        aclrtFuncHandle funcHandle;
        const aclError aclRet = aclrtBinaryGetFunction(handle_, kernelName, &funcHandle);
        if (aclRet != ACL_SUCCESS) {
            THROW<RuntimeApiException>(StringFormat("Call aclrtBinaryGetFunction failed, with ret[%d]", aclRet));
        }
        HCCL_INFO("[AicpuBinaryHolder::%s] getting funcHandle for kernel[%s]", __func__, kernelName);
        const std::string kernelNameStr = std::string(kernelName);
        aicpuFuncMap_[kernelNameStr] = funcHandle;
    }
    aicpuKernelGuard.Dismiss();
    HCCL_INFO("[AicpuBinaryHolder::%s] end.", __func__);
}
void AicpuBinaryHolder::Unload()
{
    if (loaded_ && handle_ != nullptr) {
        const aclError aclRet = aclrtBinaryUnLoad(handle_);
        if (aclRet != ACL_SUCCESS) {
            HCCL_ERROR("[~AicpuBinaryHolder] failed to unload binary, ret[%d]", aclRet);
        }
        handle_ = nullptr;
        loaded_ = false;
        aicpuFuncMap_.clear();
    }
}
aclrtFuncHandle AicpuBinaryHolder::GetAicpuKernelFuncHandle(const char* kernelName) const
{
    if (kernelName == nullptr || strlen(kernelName) == 0 || strlen(kernelName) >= KERNEL_PARAM_NAME_SIZE) {
        HCCL_ERROR("[AicpuBinaryHolder::%s] invalid kernel name", __func__);
        THROW<InvalidParamsException>("kernel name is invalid");
    }
    if (!loaded_) {
        HCCL_ERROR("[AicpuBinaryHolder::%s] aicpu kernel not registered, kernelName[%s]", __func__, kernelName);
        THROW<RuntimeApiException>("aicpu kernel not registered");
    }
    const auto kernelNameStr = std::string(kernelName);
    const auto it = aicpuFuncMap_.find(kernelNameStr);
    if (it == aicpuFuncMap_.end()) {
        HCCL_ERROR("[AicpuBinaryHolder::%s] function handle of kernelName[%s] is not get before", __func__, kernelName);
        THROW<RuntimeApiException>(StringFormat("function handle of kernelName[%s] is not get before", kernelName));
    }
    return it->second;
}
} // namespace Hccl

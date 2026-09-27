/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <chrono>
#include <climits>
#include <memory>
#include <atomic>
#include "config_log.h"
#include "launch_device.h"
#include "op_task_config.h"
#include "hccl_common_v2.h"
#include "ascend_hal.h"
#include "orion_adapter_rts.h"
#include "orion_adapter_hal.h"
#include "hccl_dpu_manager.h"

namespace hccl {
static std::atomic<u32> g_commDpuManagerNum(0);
constexpr uint8_t DEVICE_SIGNAL_SECOND = 2;
constexpr uint8_t DEVICE_SIGNAL_THIRD = 3;
constexpr uint32_t TEMP_DEV_TYPE_DPU = 0; // 临时适配，后续rts接口上库之后使用rts的定义
struct DpuKernelLaunchParam {             // 需要和RunDpuRpcSrvLaunch入参定义保持一致
    u64 memorySize;
    void* deviceMem;
    void* hostMem;
    int32_t deviceId;
    std::string commId;
};
DpuKernelLaunchParam g_hostArgsTemp;

DpuManager::DpuManager() {}

DpuManager::~DpuManager() {}

HcclResult DpuManager::CreateWorkspaceBuf(const char* memTag, uint64_t* size, bool* newCreated)
{
    std::string tag = memTag != nullptr ? std::string(memTag) : "";
    if (tagWorkspaceMap_.find(tag) == tagWorkspaceMap_.end()) {
        std::shared_ptr<Hccl::DevBuffer> workspace = std::make_shared<Hccl::DevBuffer>(*size);
        tagWorkspaceMap_.insert(make_pair(tag, workspace));
        HCCL_INFO(
            "[DpuManager::%s] Create tagMem[%s] WorkspaceBuf success, WorkspaceBuf = %p", __func__, tag.c_str(),
            workspace.get());
        if (newCreated != nullptr) {
            *newCreated = true;
        }
    }
    return HCCL_SUCCESS;
}

std::shared_ptr<Hccl::DevBuffer> DpuManager::GetKFCWorkSpace(const char* memTag)
{
    std::string tag = memTag != nullptr ? std::string(memTag) : "";
    auto it = tagWorkspaceMap_.find(tag);
    return it != tagWorkspaceMap_.end() ? it->second : nullptr;
}

HcclResult DpuManager::AllocAndRegKFCWorkSpace(uint64_t size)
{
    accessVA_ = nullptr;
    drvError_t ret = DRV_ERROR_NONE;

    va_ = Hccl::HrtMalloc(size, ACL_MEM_TYPE_HIGH_BAND_WIDTH);
    ret = halHostRegister(va_, size, DEV_SVM_MAP_HOST, devLogicId_, &accessVA_);
    if (ret != DRV_ERROR_NONE) {
        HCCL_ERROR("DpuManager halHostRegister failed, ret: %d", ret);
        if (va_ != nullptr) {
            Hccl::HrtFree(va_);
        }
        return HCCL_E_DRV;
    }

    return HCCL_SUCCESS;
}

HcclResult DpuManager::DestroyKFCWorkSpaceVA()
{
    if (accessVA_ == nullptr && va_ == nullptr) {
        HCCL_RUN_WARNING("[%s]accessVA_/va_ is nullptr", __func__);
        return HCCL_E_INTERNAL;
    }

    // 必须先halHostUnregister解除映射，再释放设备内存，否则HrtFree会因内存被pin住而异常
    if (accessVA_ != nullptr) {
        drvError_t drvRet = halHostUnregister(accessVA_, devLogicId_);
        if (drvRet != DRV_ERROR_NONE) {
            HCCL_ERROR("DpuManager halHostUnregister failed, drvRet[%d]", drvRet);
        }
    }

    if (va_ != nullptr) {
        Hccl::HrtFree(va_);
    }

    va_ = nullptr;
    accessVA_ = nullptr;
    tagWorkspaceVAMap_.erase(DPUTAG);
    return HCCL_SUCCESS;
}

HcclResult DpuManager::GetKFCWorkSpaceVA(const std::string& memTag, uint64_t* size, void** addr, bool* newCreated)
{
    if (memTag != DPUTAG) {
        HCCL_ERROR("DpuManager::GetKFCWorkSpaceVA, memTag is invalid, memTag: %s", memTag.c_str());
        return HCCL_E_PARA;
    }
    auto iter = tagWorkspaceVAMap_.find(memTag);
    if (iter != tagWorkspaceVAMap_.end()) {
        std::shared_ptr<Hccl::DevBuffer> oldWorkspace = iter->second;
        if (*size != static_cast<uint64_t>(oldWorkspace.get()->GetSize())) {
            HCCL_ERROR(
                "DpuManager::GetKFCWorkSpaceVA, The size of oldWorkspace %p is non-consistent, target size compare now "
                "size: %llu->%llu",
                *addr, *size, oldWorkspace.get()->GetSize());
            return HCCL_E_PARA;
        }
        *addr = reinterpret_cast<void*>(oldWorkspace.get()->GetAddr());
        if (newCreated != nullptr) {
            *newCreated = false;
        }
        return HcclResult::HCCL_SUCCESS;
    }

    CHK_RET(AllocAndRegKFCWorkSpace(*size));
    std::shared_ptr<Hccl::DevBuffer> newWorkspace
        = Hccl::DevBuffer::Create(reinterpret_cast<uintptr_t>(accessVA_), *size);
    tagWorkspaceVAMap_.insert(make_pair(memTag, newWorkspace));
    if (newCreated != nullptr) {
        *newCreated = true;
    }
    *addr = reinterpret_cast<void*>(newWorkspace.get()->GetAddr());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult DpuManager::GetDevMemWorkSpace(const std::string& memTag, uint64_t* size, void** addr, bool* newCreated)
{
    if (memTag == DPUTAG) {
        return GetKFCWorkSpaceVA(memTag, size, addr, newCreated);
    }
    auto iter = tagWorkspaceMap_.find(memTag);
    if (iter != tagWorkspaceMap_.end()) {
        std::shared_ptr<Hccl::DevBuffer> oldWorkspace = iter->second;
        if (*size != static_cast<uint64_t>(oldWorkspace.get()->GetSize())) {
            HCCL_ERROR(
                "DpuManager::GetDevMemWorkSpace, The size of oldWorkspace %p is non-consistent, "
                "target size compare now size: %llu->%llu",
                *addr, *size, oldWorkspace.get()->GetSize());
            return HCCL_E_PARA;
        }
        *addr = reinterpret_cast<void*>(oldWorkspace.get()->GetAddr());
        if (newCreated != nullptr) {
            *newCreated = false;
        }
        return HcclResult::HCCL_SUCCESS;
    }

    std::shared_ptr<Hccl::DevBuffer> newWorkspace = std::make_shared<Hccl::DevBuffer>(*size);
    tagWorkspaceMap_.insert(make_pair(memTag, newWorkspace));
    HCCL_INFO(
        "Create tagMem[%s] WorkspaceBuf success, WorkspaceBuf: %p -> %p, size[%llu]", memTag.c_str(),
        newWorkspace.get(), newWorkspace.get()->GetAddr(), *size);
    if (newCreated != nullptr) {
        *newCreated = true;
    }
    *addr = reinterpret_cast<void*>(newWorkspace.get()->GetAddr());
    return HcclResult::HCCL_SUCCESS;
}

HcclResult DpuManager::Init(const std::string& commId, u32 deviceLogicId)
{
    commId_ = commId;
    devLogicId_ = deviceLogicId;
    return InitDpuKernel();
}

HcclResult DpuManager::LaunchDpuKernel(aclrtFuncHandle& funcHandle)
{
    HCCL_INFO("[DpuManager::%s] Launch Dpu Kernel", __func__);
    aclrtLaunchKernelCfg cfg;
    aclrtLaunchKernelAttr kernelAttr;
    kernelAttr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    kernelAttr.value.timeout = Hccl::NOTIFY_DEFAULT_WAIT_TIME > std::numeric_limits<uint16_t>::max() ?
                                   std::numeric_limits<uint16_t>::max() :
                                   Hccl::NOTIFY_DEFAULT_WAIT_TIME;
    cfg.numAttrs = 1;
    cfg.attrs = &kernelAttr;
    constexpr u32 numBlocks = 1;

    g_hostArgsTemp.commId = commId_;
    g_hostArgsTemp.memorySize = SHARE_HBM_MEMORY_SIZE;
    g_hostArgsTemp.hostMem = hostShareBuf_;
    g_hostArgsTemp.deviceMem = accessVA_;
    g_hostArgsTemp.deviceId = devLogicId_;

    HCCL_INFO(
        "[DpuManager::%s] DpuKernelLaunchParam{commId:%s; memorySize:%u; deviceMem:%p; hostMem:%p}", __func__,
        g_hostArgsTemp.commId.c_str(), g_hostArgsTemp.memorySize, g_hostArgsTemp.deviceMem, g_hostArgsTemp.hostMem);

    size_t argsSize = sizeof(g_hostArgsTemp);
    aclrtPlaceHolderInfo placeHolderArrays;
    size_t placeHolderNum = 0;
    if (aclrtLaunchKernelWithHostArgs(
            funcHandle, numBlocks, dpuStream_, &cfg, &g_hostArgsTemp, argsSize, &placeHolderArrays, placeHolderNum)
        != ACL_SUCCESS) {
        HCCL_ERROR("[DpuManager::%s] Launch Dpu Kernel Failed", __func__);
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult DpuManager::PrepareDpuKernelResource(aclrtFuncHandle& funcHandle)
{
    // 获取二进制文件路径
    std::string jsonPath;
    const char* envPath = getenv("ASCEND_HOME_PATH");
    if (envPath != nullptr && envPath[0] != '\0') {
        jsonPath = envPath;
    } else {
        jsonPath = "/usr/local/Ascend/cann/";
        HCCL_WARNING("[DpuManager::%s] ENV:ASCEND_HOME_PATH is not set", __func__);
    }

    jsonPath += "/opp/built-in/op_impl/dpu/";
    HCCL_DEBUG("[DpuManager::%s] kernel folder path[%s]", __func__, jsonPath.c_str());

    // cpuKernelMode为1时，json命名需与so命名保持一致， 即libccl_dpu.json与libccl_dpu.so
    jsonPath += "libccl_dpu.json";
    char realPath[PATH_MAX] = {0};
    CHK_PRT_RET(
        realpath(jsonPath.c_str(), realPath) == nullptr,
        HCCL_ERROR("[DpuManager::%s]: %s is not a valid real path, err[%d]", __func__, jsonPath.c_str(), errno),
        HCCL_E_INTERNAL);
    HCCL_INFO("[DpuManager::%s] realPath: %s", __func__, realPath);

    aclrtBinHandle binHandle;
    aclrtBinaryLoadOptions options;
    aclrtBinaryLoadOption option;
    option.type = ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE;
    option.value.cpuKernelMode = 1;
    options.numOpt = 1;
    options.options = &option;
    if (aclrtBinaryLoadFromFile(realPath, &options, &binHandle) != ACL_SUCCESS) {
        HCCL_ERROR("[DpuManager::%s] load binary from file error.", __func__);
        return HCCL_E_OPEN_FILE_FAILURE;
    }

    // 创建dpustream
    if (aclrtCreateStreamWithConfig(&dpuStream_, 0, ACL_STREAM_FAST_LAUNCH) != ACL_SUCCESS) {
        HCCL_ERROR("[DpuManager::%s] Create Local Stream Failed", __func__);
        return HCCL_E_INTERNAL;
    }

    // 查找核函数
    if (aclrtBinaryGetFunction(binHandle, "RunDpuRpcSrvLaunch", &funcHandle) != ACL_SUCCESS) {
        HCCL_ERROR("[DpuManager::%s] Get Function Failed", __func__);
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

HcclResult DpuManager::InitAndLaunchDpuKernel()
{
    HCCL_INFO("[DpuManager::%s] Start to Launch Dpu Kernel", __func__);
    bool newCreate = false;
    uint64_t memSize = static_cast<uint64_t>(SHARE_HBM_MEMORY_SIZE);
    HcclResult memRet = GetKFCWorkSpaceVA(DPUTAG, &memSize, &accessVA_, &newCreate);
    if (memRet != HCCL_SUCCESS) {
        HCCL_ERROR("[DpuManager::InitCommResource] Alloc Share HBM Failed");
        return HCCL_E_RUNTIME;
    }

    // 设置XPU
    HCCL_INFO("[DpuManager::%s] Switch to Dpu Ctx", __func__);
    if (aclrtGetCurrentContext(&npuContext_) != ACL_SUCCESS) {
        HCCL_ERROR("[DpuManager::%s] Get Npu Ctx Failed", __func__);
        return HCCL_E_INTERNAL;
    }
    if (Hccl::HrtSetXpuDevice(TEMP_DEV_TYPE_DPU, 0) != HCCL_SUCCESS) {
        HCCL_ERROR("[DpuManager::%s] Switch to Dpu Ctx Failed", __func__);
        return HCCL_E_INTERNAL;
    }
    if (aclrtGetCurrentContext(&dpuContext_) != ACL_SUCCESS) {
        HCCL_ERROR("[DpuManager::%s] Get Dpu Ctx Failed", __func__);
        return HCCL_E_INTERNAL;
    }

    // 准备资源
    aclrtFuncHandle funcHandle;
    CHK_RET(PrepareDpuKernelResource(funcHandle));

    hostShareBuf_ = malloc(SHARE_HBM_MEMORY_SIZE);
    CHK_PTR_NULL(hostShareBuf_);

    // 下发
    HcclResult ret = LaunchDpuKernel(funcHandle);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[CommunicatorImpl::%s] Launch Dpu Kernel Failed", __func__);
        free(hostShareBuf_);
        hostShareBuf_ = nullptr;
        return ret;
    }

    // 切换回当前Ctx
    HCCL_INFO("[DpuManager::%s] Switch to Npu Ctx", __func__);
    if (ACL_SUCCESS != aclrtSetCurrentContext(npuContext_)) {
        HCCL_ERROR("[DpuManager::%s] Reset Current Ctx Failed", __func__);
        free(hostShareBuf_);
        hostShareBuf_ = nullptr;
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("[DpuManager::%s] Launch Dpu Kernel End", __func__);
    isDpuKernelLaunched_ = true;
    g_commDpuManagerNum++;
    return HCCL_SUCCESS;
}

HcclResult DpuManager::InitDpuKernel()
{
    /* kernel Launch */
    CHK_RET(InitAndLaunchDpuKernel());
    return HCCL_SUCCESS;
}

HcclResult DpuManager::DeInitDpuKernel()
{
    if (!isDpuKernelLaunched_) {
        return HCCL_SUCCESS;
    }
    (void)DestroyDpuKernelResource();
    (void)DestroyKFCWorkSpaceVA();
    if (hostShareBuf_ != nullptr) {
        free(hostShareBuf_);
        hostShareBuf_ = nullptr;
    }
    return HCCL_SUCCESS;
}

HcclResult DpuManager::WaitDpuKernelThreadTerminate()
{
    if (!isDpuKernelLaunched_) {
        return HCCL_SUCCESS;
    }
    if (accessVA_ == nullptr) {
        HCCL_ERROR("[CommunicatorImpl::%s] accessVA_ is nullptr", __func__);
        return HCCL_E_MEMORY;
    }
    uint8_t flag = DEVICE_SIGNAL_SECOND;
    errno_t ret = memcpy_s(accessVA_, sizeof(flag), &flag, sizeof(flag));
    if (ret != EOK) {
        HCCL_ERROR("Terminate TaskRun Fail, return[%d]", ret);
        return HCCL_E_MEMORY;
    }
    do {
        ret = memcpy_s(&flag, sizeof(flag), accessVA_, sizeof(flag));
        if (ret != EOK) {
            HCCL_ERROR("Read Terminate TaskRun Signal Fail, return[%d]", ret);
            return HCCL_E_MEMORY;
        }
    } while (flag != DEVICE_SIGNAL_THIRD);

    return HCCL_SUCCESS;
}

HcclResult DpuManager::DestroyDpuKernelResource()
{
    // 终止Dpu Kernel的TaskRun
    if (!isDpuKernelLaunched_) {
        return HCCL_SUCCESS;
    }

    CHK_RET(WaitDpuKernelThreadTerminate());

    // 切换回 dpu ctx
    aclError aclRet = aclrtSetCurrentContext(dpuContext_);
    if (ACL_SUCCESS != aclRet) {
        HCCL_ERROR("set dpu Ctx Failed, aclReturn[%d]", aclRet);
        return HCCL_E_RUNTIME;
    }
    // 销毁局部流
    aclRet = aclrtDestroyStreamForce(dpuStream_);
    if (ACL_SUCCESS != aclRet) {
        HCCL_ERROR("Destroy Stream Failed, aclReturn[%d]", aclRet);
        aclRet = aclrtSetCurrentContext(npuContext_);
        CHK_PRT_RET(aclRet == ACL_SUCCESS, HCCL_ERROR("set npu Ctx Failed, aclReturn[%d]", aclRet), HCCL_E_RUNTIME);
        return HCCL_E_RUNTIME;
    }
    if (g_commDpuManagerNum > 1) {
        g_commDpuManagerNum--;
    } else {
        // reset DPU kernel 线程
        HcclResult ret = Hccl::HrtResetXpuDevice(TEMP_DEV_TYPE_DPU, 0);
        if (HCCL_SUCCESS != ret) {
            HCCL_ERROR("ResetXpuDevice Failed, return[%d]", ret);
            aclRet = aclrtSetCurrentContext(npuContext_);
            CHK_PRT_RET(aclRet == ACL_SUCCESS, HCCL_ERROR("set npu Ctx Failed, aclReturn[%d]", aclRet), HCCL_E_RUNTIME);
            return HCCL_E_RUNTIME;
        }
    }
    // 切回 npu ctx
    aclRet = aclrtSetCurrentContext(npuContext_);
    if (ACL_SUCCESS != aclRet) {
        HCCL_ERROR("set npu Ctx Failed, aclReturn[%d]", aclRet);
        return HCCL_E_RUNTIME;
    }

    return HCCL_SUCCESS;
}

} // namespace hccl

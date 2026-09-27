/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include "launch_aicpu.h"
#include "log.h"
#include "mmpa_api.h"
#include "mem_host_pub.h"
#include "adapter_rts_common.h"

using namespace std;

namespace hccl {
static thread_local HostMem g_aicpuKernelBinV2;

HcclResult InitKernelArgsPrepare(
    aclrtBinHandle binHandle, const std::string& kernelName, void* initTaskAddr, u32 initTaskSize,
    aclrtFuncHandle& funcHandle, aclrtArgsHandle& argsHandle)
{
    aclError ret = aclrtBinaryGetFunction(binHandle, kernelName.c_str(), &funcHandle);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[aclrtBinaryGetFunction]errNo[0x%016llx] get func handle failed, kernelName[%s]", ret, kernelName.c_str()),
        HCCL_E_RUNTIME);

    ret = aclrtKernelArgsInit(funcHandle, &argsHandle);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR("[aclrtKernelArgsInit]errNo[0x%016llx] args init failed, kernelName[%s]", ret, kernelName.c_str()),
        HCCL_E_RUNTIME);

    aclrtParamHandle paraHandle;
    ret = aclrtKernelArgsAppend(argsHandle, initTaskAddr, initTaskSize, &paraHandle);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[aclrtKernelArgsAppend]errNo[0x%016llx] args append failed, append size %u, kernelName[%s]", ret,
            initTaskSize, kernelName.c_str()),
        HCCL_E_RUNTIME);

    ret = aclrtKernelArgsFinalize(argsHandle);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[aclrtKernelArgsFinalize]errNo[0x%016llx] args finalize failed, kernelName[%s]", ret, kernelName.c_str()),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
}

HcclResult TaskCommKernelArgsPrepare(
    aclrtBinHandle binHandle, const std::string& kernelName, void* contextAddr, u32 contextSize, void* tilingDataPtr,
    u32 tilingDataSize, aclrtFuncHandle& funcHandle, aclrtArgsHandle& argsHandle)
{
    CHK_PRT_RET(
        (tilingDataPtr == nullptr || tilingDataSize == 0),
        HCCL_ERROR(
            "[TaskCommKernelArgsPrepare]param is invalid,tilingDataPtr[%p], tilingDataSize[%u], kernelName[%s]",
            tilingDataPtr, tilingDataSize, kernelName.c_str()),
        HCCL_E_PARA);

    aclError aclRet = aclrtBinaryGetFunction(binHandle, kernelName.c_str(), &funcHandle);
    CHK_PRT_RET(
        aclRet != ACL_SUCCESS,
        HCCL_ERROR(
            "[aclrtBinaryGetFunction]errNo[0x%016llx] get func handle failed, kernelName[%s]", aclRet,
            kernelName.c_str()),
        HCCL_E_RUNTIME);

    aclRet = aclrtKernelArgsInit(funcHandle, &argsHandle);
    CHK_PRT_RET(
        aclRet != ACL_SUCCESS,
        HCCL_ERROR(
            "[aclrtKernelArgsInit]errNo[0x%016llx] args init failed, kernelName[%s]", aclRet, kernelName.c_str()),
        HCCL_E_RUNTIME);
    // 拼凑aicpu侧KFCTaskComm结构体
    // 1、先存放HcclOpResParam的context指针
    aclrtParamHandle paraHandle;
    aclRet = aclrtKernelArgsAppend(argsHandle, contextAddr, contextSize, &paraHandle);
    CHK_PRT_RET(
        aclRet != ACL_SUCCESS,
        HCCL_ERROR(
            "[aclrtKernelArgsAppend]errNo[0x%016llx] args append failed, append size[%u], kernelName[%s]", aclRet,
            contextSize, kernelName.c_str()),
        HCCL_E_RUNTIME);
    // 2、再将OpTilingData的tilingData指针拼凑，并将tilingData的hostMem给rts进行H2D
    void* dataAddr;
    aclRet = aclrtKernelArgsAppendPlaceHolder(argsHandle, &paraHandle);
    CHK_PRT_RET(
        aclRet != ACL_SUCCESS,
        HCCL_ERROR("[aclrtKernelArgsAppendPlaceHolder]errNo[0x%016llx] args append place holder failed", aclRet),
        HCCL_E_RUNTIME);

    aclRet = aclrtKernelArgsGetPlaceHolderBuffer(argsHandle, paraHandle, tilingDataSize, &dataAddr);
    CHK_PRT_RET(
        aclRet != ACL_SUCCESS,
        HCCL_ERROR(
            "[aclrtKernelArgsGetPlaceHolderBuffer]errNo[0x%016llx] args get place holder buffer failed,"
            "tilingDataSize[%u]",
            aclRet, tilingDataSize),
        HCCL_E_RUNTIME);
    CHK_SAFETY_FUNC_RET(memcpy_s(dataAddr, tilingDataSize, tilingDataPtr, tilingDataSize));

    aclRet = aclrtKernelArgsFinalize(argsHandle);
    CHK_PRT_RET(
        aclRet != ACL_SUCCESS,
        HCCL_ERROR(
            "[aclrtKernelArgsFinalize]errNo[0x%016llx] args finalize failed, kernelName[%s]", aclRet,
            kernelName.c_str()),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
}

HcclResult AicpuAclKernelLaunch(
    const rtStream_t stm, void* addr, u32 size, aclrtBinHandle binHandle, const std::string& kernelName,
    bool isInitTask, u16 timeOut, void* tilingDataPtr, u32 tilingDataSize)
{
    if (binHandle == nullptr) {
        HCCL_ERROR("binHandle is nullptr, no need to launch aicpu kernel, binHandle[%p]", binHandle);
        return HCCL_E_PTR;
    }
    CHK_PRT_RET(
        (addr == nullptr || size == 0),
        HCCL_ERROR(
            "[AicpuAclKernelLaunch]param is invalid, contextAddr[%p], size[%u], kernelName[%s]", addr, size,
            kernelName.c_str()),
        HCCL_E_PARA);

    aclrtFuncHandle funcHandle;
    aclrtArgsHandle argsHandle;
    HcclResult ret;
    if (isInitTask) {
        ret = InitKernelArgsPrepare(binHandle, kernelName, addr, size, funcHandle, argsHandle);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[InitKernelArgsPrepare]errNo[0x%016llx]init args prepare failed, kernelName[%s], "
                "contextAddr[%p], size[%u]",
                ret, kernelName.c_str(), addr, size),
            HCCL_E_RUNTIME);
    } else {
        ret = TaskCommKernelArgsPrepare(
            binHandle, kernelName, addr, size, tilingDataPtr, tilingDataSize, funcHandle, argsHandle);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[TaskCommKernelArgsPrepare]errNo[0x%016llx]TaskComm args prepare failed, kernelName[%s], "
                "contextAddr[%p], size[%u]",
                ret, kernelName.c_str(), addr, size),
            HCCL_E_RUNTIME);
    }

    aclrtLaunchKernelCfg cfg;
    aclrtLaunchKernelAttr attr;
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attr.value.timeout = timeOut;
    cfg.numAttrs = 1;
    cfg.attrs = &attr;
    constexpr u32 numBlocks = 1;
    aclError aclRet = aclrtLaunchKernelWithConfig(funcHandle, numBlocks, stm, &cfg, argsHandle, nullptr);
    CHK_PRT_RET(
        aclRet != ACL_SUCCESS, HCCL_ERROR("[aclrtLaunchKernelWithConfig]errNo[0x%016llx] launch kernel failed", ret),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
}

HcclResult CacheTaskOpInfo(aclrtStream stream, const std::string& identify)
{
    aclmdlRI rtModel = nullptr;
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclError aclRet = aclmdlRICaptureGetInfo(stream, &captureStatus, &rtModel);
    if (aclRet == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        HCCL_WARNING("[%s]Stream capture does not support!", __func__);
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        aclRet != ACL_SUCCESS, HCCL_ERROR("[%s]rtGet stream get capture status fail. return[%d]", __func__, aclRet),
        HCCL_E_RUNTIME);

    aclrtStreamAttrValue value;
    aclRet = aclrtGetStreamAttribute(stream, ACL_STREAM_ATTR_CACHE_OP_INFO, &value);
    CHK_PRT_RET(
        aclRet != ACL_SUCCESS, HCCL_ERROR("[%s]stream get attribute fail. return[%d]", __func__, aclRet),
        HCCL_E_RUNTIME);

    HCCL_INFO(
        "[CacheTaskOpInfo] cacheOpInfoSwitch[%u] captureStatus[%d] identify[%s]", value.cacheOpInfoSwitch,
        captureStatus, identify.c_str());
    if (value.cacheOpInfoSwitch == 1 && captureStatus == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE) {
        HcclResult cacheRet = hrtCacheLastTaskExtendInfo(identify.c_str(), strlen(identify.c_str()));
        if (cacheRet == HCCL_E_NOT_SUPPORT) {
            HCCL_INFO("[%s] aclrtCacheLastTaskExtendInfo not supported", __func__);
        } else {
            CHK_PRT_RET(
                cacheRet != HCCL_SUCCESS,
                HCCL_ERROR("[%s] stream cache task op info fail. return[%d]", __func__, cacheRet), HCCL_E_RUNTIME);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuAclKernelLaunchV2(
    const rtStream_t stm, void* addr, u32 size, aclrtBinHandle binHandle, const std::string& kernelName,
    bool isInitTask, u16 timeOut, void* tilingDataPtr, u32 tilingDataSize, const std::string& identify)
{
    if (binHandle == nullptr) {
        HCCL_ERROR("binHandle is nullptr, no need to launch aicpu kernel, binHandle[%p]", binHandle);
        return HCCL_E_PTR;
    }
    CHK_PRT_RET(
        (addr == nullptr || size == 0),
        HCCL_ERROR(
            "[AicpuAclKernelLaunch]param is invalid, contextAddr[%p], "
            "size[%u], kernelName[%s]",
            addr, size, kernelName.c_str()),
        HCCL_E_PARA);
    aclrtFuncHandle funcHandle;
    aclError aclRet = aclrtBinaryGetFunction(binHandle, kernelName.c_str(), &funcHandle);
    CHK_PRT_RET(
        aclRet != ACL_SUCCESS,
        HCCL_ERROR(
            "[aclrtBinaryGetFunction]errNo[0x%016llx] get func handle failed, "
            "kernelName[%s]",
            aclRet, kernelName.c_str()),
        HCCL_E_RUNTIME);
    // !isInitTask LaunchTask {u64 context, TillingData data}
    u64 hostBufferSize = isInitTask ? size : sizeof(u64) + tilingDataSize;
    if (g_aicpuKernelBinV2.size() < hostBufferSize) {
        g_aicpuKernelBinV2.free();
        g_aicpuKernelBinV2 = HostMem::alloc(hostBufferSize, false);
        if (g_aicpuKernelBinV2.ptr() == nullptr) {
            HCCL_ERROR("[AicpuAclKernelLaunchV2] alloc memory failed");
            return HCCL_E_MEMORY;
        }
    }
    if (isInitTask) {
        auto memRet = memcpy_s(reinterpret_cast<void*>(g_aicpuKernelBinV2.ptr()), hostBufferSize, addr, hostBufferSize);
        CHK_PRT_RET(
            memRet != EOK, HCCL_ERROR("[AicpuAclKernelLaunchV2]memcpy_s failed,return[%d]", memRet), HCCL_E_INTERNAL);
    } else {
        auto memRet = memcpy_s(reinterpret_cast<void*>(g_aicpuKernelBinV2.ptr()), sizeof(u64), addr, sizeof(u64));
        CHK_PRT_RET(
            memRet != EOK, HCCL_ERROR("[AicpuAclKernelLaunchV2]memcpy_s failed,return[%d]", memRet), HCCL_E_INTERNAL);
        memRet = memcpy_s(
            reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(g_aicpuKernelBinV2.ptr()) + sizeof(u64)),
            hostBufferSize - sizeof(u64), tilingDataPtr, tilingDataSize);
        CHK_PRT_RET(
            memRet != EOK, HCCL_ERROR("[AicpuAclKernelLaunchV2]memcpy_s failed,return[%d]", memRet), HCCL_E_INTERNAL);
    }
    aclrtLaunchKernelCfg cfg;
    aclrtLaunchKernelAttr attr;
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    attr.value.timeout = timeOut;
    cfg.numAttrs = 1;
    cfg.attrs = &attr;
    constexpr u32 numBlocks = 1;
    aclRet = aclrtLaunchKernelWithHostArgs(
        funcHandle, numBlocks, stm, &cfg, g_aicpuKernelBinV2.ptr(), hostBufferSize, nullptr, 0);
    CHK_PRT_RET(
        aclRet != ACL_SUCCESS,
        HCCL_ERROR("[aclrtLaunchKernelWithHostArgs]errNo[0x%016llx] launch kernel failed", aclRet), HCCL_E_RUNTIME);

    CHK_RET(CacheTaskOpInfo(stm, identify));

    return HCCL_SUCCESS;
}

HcclResult GetKernelFilePath(std::string& binaryPath)
{
    // 获取二进制文件路径
    std::string libPath;
    char* getPath = getenv("ASCEND_HOME_PATH");
    MM_SYS_GET_ENV(MM_ENV_ASCEND_HOME_PATH, getPath);
    if (getPath != nullptr) {
        libPath = getPath;
    } else {
        libPath = "/usr/local/Ascend/cann/";
        HCCL_WARNING("[GetKernelFilePath]ENV:ASCEND_HOME_PATH is not set");
    }

    libPath += "/opp/built-in/op_impl/aicpu/config/";
    binaryPath = libPath;
    HCCL_DEBUG("[GetKernelFilePath]kernel folder path[%s]", binaryPath.c_str());

    return HCCL_SUCCESS;
}

HcclResult GetCustomKernelFilePath(std::string& binaryPath)
{
    // 获取二进制文件路径
    std::string libPath;
    char* getPath = getenv("ASCEND_HOME_PATH");
    MM_SYS_GET_ENV(MM_ENV_ASCEND_HOME_PATH, getPath);
    if (getPath != nullptr) {
        libPath = getPath;
    } else {
        libPath = "/usr/local/Ascend/cann/";
        HCCL_WARNING("[GetCustomKernelFilePath]ENV:ASCEND_HOME_PATH is not set");
    }

    libPath += "/opp/built-in/op_impl/aicpu/kernel/";
    binaryPath = libPath;
    HCCL_DEBUG("[GetCustomKernelFilePath]kernel folder path[%s]", binaryPath.c_str());

    return HCCL_SUCCESS;
}

} // namespace hccl

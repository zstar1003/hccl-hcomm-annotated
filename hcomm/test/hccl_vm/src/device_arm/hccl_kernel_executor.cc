/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_kernel_executor.h"

#include <cstdint>
#include <cstring>
#include <dlfcn.h>

#include "aicpu_args_stub.h"
#include "alg_param.h"
#include "hccl_device_pub.h"
#include "sim_common_api.h"
#include "sim_log.h"
#include "store_sim_memory_manager.h"

using namespace ops_hccl;

bool gLibsLoaded = false;
void* gCsecHandle = nullptr;
void* gSlogHandle = nullptr;
void* gHcommHandle = nullptr;
void* gHcclKerHandle = nullptr;

uint32_t (*runAicpuIndOpCommInitPtr)(void* args) = nullptr;
uint32_t (*runAicpuIndOpThreadInitPtr)(void* args) = nullptr;
uint32_t (*runAicpuIndOpChannelInitV2Ptr)(void* args) = nullptr;
uint32_t (*runAicpuDfxOpInfoInitV2Ptr)(void* args) = nullptr;
uint32_t (*runAicpuThreadSupplementNotifyPtr)(void* args) = nullptr;
uint32_t (*runAicpuNotifyWaitPtr)(void* args) = nullptr;
uint32_t (*runAicpuNotifyRecordPtr)(void* args) = nullptr;
unsigned int (*hcclLaunchAicpuKernelPtr)(OpParam* param) = nullptr;
unsigned int (*hcclLaunchP2pAicpuKernelPtr)(void* args) = nullptr;

uint64_t d2hAddr = 0;

void ExecuteAicpuKernel(uint32_t rankId, const std::string& kernelName, uint64_t args)
{
    HCCL_VM_INFO("rankId[{}] kernel[{}] start run...", rankId, kernelName);
    if (!InitKernelFuncHandle()) {
        HCCL_VM_ERROR("rankId[{}] init func handle failed.", rankId);
        return;
    }

    void* realPtr = GetRealPtrByDevPtr(reinterpret_cast<void*>(args));
    if (kernelName == "RunAicpuCommInit") {
        CommAicpuParam* param = reinterpret_cast<CommAicpuParam*>(realPtr);
        param->kfcControlTransferH2DParams.deviceAddr = reinterpret_cast<uint64_t>(
            GetRealPtrByDevPtr(reinterpret_cast<void*>(param->kfcControlTransferH2DParams.deviceAddr)));
        param->kfcControlTransferH2DParams.readCacheAddr = reinterpret_cast<uint64_t>(
            GetRealPtrByDevPtr(reinterpret_cast<void*>(param->kfcControlTransferH2DParams.readCacheAddr)));
        param->kfcStatusTransferD2HParams.deviceAddr = reinterpret_cast<uint64_t>(
            GetRealPtrByDevPtr(reinterpret_cast<void*>(param->kfcStatusTransferD2HParams.deviceAddr)));
        d2hAddr = param->kfcStatusTransferD2HParams.deviceAddr;
        runAicpuIndOpCommInitPtr(realPtr);
    } else if (kernelName == "RunAicpuIndOpThreadInit" || kernelName == "RunAicpuThreadSupplementNotify") {
        uint64_t value = *reinterpret_cast<uint64_t*>(realPtr);
        void* ptr = GetRealPtrByDevPtr(reinterpret_cast<void*>(value));
        ThreadMgrAicpuParam* param = reinterpret_cast<ThreadMgrAicpuParam*>(ptr);
        param->deviceHandle = GetRealPtrByDevPtr(param->deviceHandle);
        // 更新序列化资源中的devId为rankId(多Server传递的时phyId设备侧无法区分，需转成rankId使用)
        for (auto i = 0; i < param->threadNum; i++) {
            char* threadData = param->threadParam[i];
            AicpuTsThread* thread = reinterpret_cast<AicpuTsThread*>(threadData);
            thread->devId = rankId;
        }

        if (kernelName == "RunAicpuIndOpThreadInit") {
            runAicpuIndOpThreadInitPtr(&ptr);
        } else {
            // 资源补充场景
            runAicpuThreadSupplementNotifyPtr(&ptr);
        }
    } else if (kernelName == "RunAicpuIndOpChannelInitV2") {
        InitTask* task = reinterpret_cast<InitTask*>(realPtr);
        void* ctxPtr = GetRealPtrByDevPtr(reinterpret_cast<void*>(task->context));
        task->context = reinterpret_cast<uint64_t>(ctxPtr);
        HcclChannelUrmaRes* res = reinterpret_cast<HcclChannelUrmaRes*>(ctxPtr);
        uint32_t listNum = res->listNum;
        uint32_t totalListSize = res->uniqueIdSize;
        uint32_t listSize = totalListSize / listNum;
        HCCL_VM_INFO("start packing, listNum:{}, totalListSize:{}, listSize:{}", listNum, totalListSize, listSize);
        res->channelList = GetRealPtrByDevPtr(res->channelList);
        res->channelSizeAddr = GetRealPtrByDevPtr(res->channelSizeAddr);
        res->uniqueIdAddr = GetRealPtrByDevPtr(res->uniqueIdAddr);
        res->remoteRankList
            = reinterpret_cast<uint32_t*>(GetRealPtrByDevPtr(reinterpret_cast<void*>(res->remoteRankList)));
        // 解析UniqueIdV2Header
        uint8_t* startAddr = reinterpret_cast<uint8_t*>(res->uniqueIdAddr);
        for (auto i = 0; i < listNum; i++) {
            uint32_t startOffset = i * listSize;
            UniqueIdV2Header* header
                = reinterpret_cast<UniqueIdV2Header*>(startAddr + startOffset + UNIQUEID_HEADER_OFFSET);
            HCCL_VM_INFO(
                "uniqueId header type:{} notifyNum:{} bufferNum:{} rmtBufferNum:{} connNum:{}.", header->type,
                header->notifyNum, header->bufferNum, header->rmtBufferNum, header->connNum);
            // 根据UniqueIdV2Header计算偏移
            uint32_t offset = COMMON_DATA_SIZE + header->notifyNum * NOTIFY_ID_SIZE + COMMON_DATA_SIZE
                              + header->notifyNum * NOTIFY_BUFFER_SIZE + COMMON_DATA_SIZE
                              + header->bufferNum * LOCAL_BUFFER_SIZE + COMMON_DATA_SIZE
                              + header->rmtBufferNum * REMOTE_BUFFER_SIZE;
            uint64_t* drainUnqueIdSize = reinterpret_cast<uint64_t*>(
                startAddr + startOffset + UNIQUEID_HEADER_OFFSET + UNIQUEID_HEADER_SIZE + offset);
            offset += sizeof(uint64_t) + *drainUnqueIdSize;
            ConnUniqueBlock* block = reinterpret_cast<ConnUniqueBlock*>(
                startAddr + startOffset + UNIQUEID_HEADER_OFFSET + UNIQUEID_HEADER_SIZE + offset);
            for (auto j = 0; j < header->connNum; ++j) {
                block->conn[j].sqBuffVa
                    = reinterpret_cast<uint64_t>(GetRealPtrByDevPtr(reinterpret_cast<void*>(block->conn[j].sqBuffVa)));
            }
        }
        runAicpuIndOpChannelInitV2Ptr(realPtr);
    } else if (kernelName == "RunAicpuDfxInitV2") {
    } else if (kernelName == "HcclLaunchAicpuKernel") {
        OpParam* opParam = reinterpret_cast<OpParam*>(realPtr);
        opParam->resCtx = GetRealPtrByDevPtr(opParam->resCtx);
        hcclLaunchAicpuKernelPtr(opParam);
        UpdataKfcStatus(d2hAddr);
    } else if (kernelName == "RunAicpuNotifyWait") {
        runAicpuNotifyWaitPtr(realPtr);
    } else if (kernelName == "HcclLaunchP2pAicpuKernel") {
        auto* p2pParam = reinterpret_cast<HcclP2pKernelParamStub*>(realPtr);
        OpParam* opParam = reinterpret_cast<OpParam*>(p2pParam->opParams);
        opParam->resCtx = GetRealPtrByDevPtr(opParam->resCtx);
        hcclLaunchP2pAicpuKernelPtr(p2pParam);
        UpdataKfcStatus(d2hAddr);
    } else if (kernelName == "RunAicpuNotifyRecord") {
        runAicpuNotifyRecordPtr(realPtr);
    } else {
        HCCL_VM_ERROR("rankId[{}] kernel[{}] not support.", rankId, kernelName);
    }

    HCCL_VM_INFO("rankId[{}] kernel[{}] finish run...", rankId, kernelName);
}

void* LoadLibrary(const std::string& libDir, const std::string& libName)
{
    void* handle = nullptr;
    std::string libPath = libDir + libName;
    handle = dlopen(libPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle != nullptr) {
        HCCL_VM_INFO("Load so {} from {}", libName, libPath);
        return handle;
    }

    HCCL_VM_ERROR("Failed to load so {} from {}", libName, dlerror());
    return nullptr;
}

bool InitKernelFuncHandle()
{
    if (gLibsLoaded) {
        return true;
    }
    std::string libDir = InstallPath::ResolveToInstallRoot("lib/" + GetArchStr()) + "/";
    gSlogHandle = LoadLibrary(libDir, "libslog.so");
    gHcclKerHandle = LoadLibrary(libDir, "libscatter_aicpu_kernel.so");
    gHcommHandle = LoadLibrary(libDir, "libccl_kernel.so");
    if (gHcclKerHandle == nullptr || gHcommHandle == nullptr || gSlogHandle == nullptr) {
        HCCL_VM_ERROR("Failed to load kernel libs.");
        return false;
    }
    runAicpuIndOpCommInitPtr = reinterpret_cast<uint32_t (*)(void*)>(dlsym(gHcommHandle, "RunAicpuCommInit"));
    runAicpuIndOpThreadInitPtr = reinterpret_cast<uint32_t (*)(void*)>(dlsym(gHcommHandle, "RunAicpuIndOpThreadInit"));
    runAicpuIndOpChannelInitV2Ptr
        = reinterpret_cast<uint32_t (*)(void*)>(dlsym(gHcommHandle, "RunAicpuIndOpChannelInitV2"));
    runAicpuDfxOpInfoInitV2Ptr = reinterpret_cast<uint32_t (*)(void*)>(dlsym(gHcommHandle, "RunAicpuDfxInitV2"));
    runAicpuThreadSupplementNotifyPtr
        = reinterpret_cast<uint32_t (*)(void*)>(dlsym(gHcommHandle, "RunAicpuThreadSupplementNotify"));
    runAicpuNotifyWaitPtr = reinterpret_cast<uint32_t (*)(void*)>(dlsym(gHcommHandle, "RunAicpuNotifyWait"));
    runAicpuNotifyRecordPtr = reinterpret_cast<uint32_t (*)(void*)>(dlsym(gHcommHandle, "RunAicpuNotifyRecord"));
    hcclLaunchAicpuKernelPtr
        = reinterpret_cast<unsigned int (*)(OpParam*)>(dlsym(gHcclKerHandle, "HcclLaunchAicpuKernel"));
    hcclLaunchP2pAicpuKernelPtr
        = reinterpret_cast<unsigned int (*)(void*)>(dlsym(gHcclKerHandle, "HcclLaunchP2pAicpuKernel"));
    if (runAicpuIndOpCommInitPtr == nullptr || runAicpuIndOpThreadInitPtr == nullptr
        || runAicpuIndOpChannelInitV2Ptr == nullptr || runAicpuDfxOpInfoInitV2Ptr == nullptr
        || runAicpuThreadSupplementNotifyPtr == nullptr || runAicpuNotifyWaitPtr == nullptr
        || runAicpuNotifyRecordPtr == nullptr) {
        HCCL_VM_ERROR("Failed to get kernel func handle in gHcommHandle.");
        return false;
    }

    if (hcclLaunchAicpuKernelPtr == nullptr || hcclLaunchP2pAicpuKernelPtr == nullptr) {
        HCCL_VM_ERROR("[InitKernelFuncHandle] Failed to get kernel func handle in gHcclKerHandle.");
        return false;
    }
    HCCL_VM_INFO("Load kernel libs and get func handles successfully.");
    gLibsLoaded = true;
    return true;
}

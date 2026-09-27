/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "local_rdma_rma_buffer_impl.h"
#include "adapter_hccp.h"
#include "hccl_network.h"
#include "network_manager_pub.h"
#include "mem_mapping_manager.h"

namespace hccl {
namespace {
    void* GetAliasDevAddr(void* addr, const LocalRdmaRmaBufferImpl& parent)
    {
        const uintptr_t childAddr = reinterpret_cast<uintptr_t>(addr);
        const uintptr_t parentAddr = reinterpret_cast<uintptr_t>(parent.GetAddr());
        const uintptr_t parentDevAddr = reinterpret_cast<uintptr_t>(parent.GetDevAddr());
        return reinterpret_cast<void*>(parentDevAddr + (childAddr - parentAddr));
    }
} // namespace

LocalRdmaRmaBufferImpl::LocalRdmaRmaBufferImpl(
    const HcclNetDevCtx netDevCtx, void* addr, u64 size, const RmaMemType memType)
    : RmaBuffer(netDevCtx, addr, size, memType, RmaType::RDMA_RMA)
{}

LocalRdmaRmaBufferImpl::LocalRdmaRmaBufferImpl(
    const HcclNetDevCtx netDevCtx, void* addr, u64 size, const RmaMemType memType, const LocalRdmaRmaBufferImpl& parent)
    : RmaBuffer(netDevCtx, addr, size, memType, RmaType::RDMA_RMA, true),
      deviceLogicId(parent.deviceLogicId),
      rdmaHandle(parent.rdmaHandle),
      mrHandle(parent.mrHandle),
      lkey(parent.lkey),
      devAddrID(parent.devAddrID),
      initialized_(true)
{
    devAddr = (memType == RmaMemType::DEVICE) ? addr : GetAliasDevAddr(addr, parent);
    HCCL_INFO("[LocalRdmaRmaBufferImpl] alias constructor, lkey[%u] mrHandle[%p]", lkey, mrHandle);
}

LocalRdmaRmaBufferImpl::~LocalRdmaRmaBufferImpl()
{
    if (!isAlias_) {
        HcclResult res = Destroy();
        if (res != HCCL_SUCCESS) {
            HCCL_ERROR("[LocalRdmaRmaBufferImpl][~LocalRdmaRmaBufferImpl]failed, ret[%d]", res);
        }
    }
}

std::unordered_map<s32, std::unordered_map<std::string, u32>> g_devAddrIdentifierMap;
std::mutex g_devAddrIdentifierMutex;

bool IsDevAddrExistInDevAddrIdentifierMap(s32 deviceLogicId, const std::string& devAddrID)
{
    std::lock_guard<std::mutex> lock(g_devAddrIdentifierMutex);
    if (g_devAddrIdentifierMap.find(deviceLogicId) != g_devAddrIdentifierMap.end()) {
        return (g_devAddrIdentifierMap[deviceLogicId].find(devAddrID) != g_devAddrIdentifierMap[deviceLogicId].end());
    }
    return false;
}

HcclResult AddDevAddrIdentifierMap(s32 deviceLogicId, const std::string& devAddrID)
{
    CHK_PRT_RET(
        deviceLogicId == INVALID_INT, HCCL_ERROR("[AddDevAddrIdentifierMap] deviceLogicId is error."), HCCL_E_PARA);
    CHK_PRT_RET(devAddrID.empty(), HCCL_ERROR("[AddDevAddrIdentifierMap] devAddrID is error."), HCCL_E_PARA);
    // devAddrID exit
    bool isDevAddrExist = IsDevAddrExistInDevAddrIdentifierMap(deviceLogicId, devAddrID);
    std::lock_guard<std::mutex> lock(g_devAddrIdentifierMutex);
    if (isDevAddrExist) {
        g_devAddrIdentifierMap[deviceLogicId][devAddrID] += 1;
        return HCCL_SUCCESS;
    }
    // 确保 deviceLogicId 和 devAddrID 的 map 已经被初始化
    if (g_devAddrIdentifierMap.find(deviceLogicId) == g_devAddrIdentifierMap.end()) {
        g_devAddrIdentifierMap[deviceLogicId] = {};
    }
    g_devAddrIdentifierMap[deviceLogicId][devAddrID] = 1;
    return HCCL_SUCCESS;
}

HcclResult DeDevAddrIdentifierMap(s32 deviceLogicId, const std::string& devAddrID)
{
    bool isDevAddrExist = IsDevAddrExistInDevAddrIdentifierMap(deviceLogicId, devAddrID);
    CHK_PRT_RET(
        !isDevAddrExist, HCCL_ERROR("[LocalRdmaRmaBufferImpl][DeDevAddrIdentifierMap]devAddrID is not existed."),
        HCCL_E_PARA);
    std::lock_guard<std::mutex> lock(g_devAddrIdentifierMutex);
    if (g_devAddrIdentifierMap[deviceLogicId][devAddrID] > 0) {
        g_devAddrIdentifierMap[deviceLogicId][devAddrID]--;
        if (g_devAddrIdentifierMap[deviceLogicId][devAddrID] == 0) {
            g_devAddrIdentifierMap[deviceLogicId].erase(devAddrID);
            HCCL_RUN_INFO("Entry-%s: deviceLogicId[%d] erased.", __func__, deviceLogicId);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult LocalRdmaRmaBufferImpl::Init()
{
    CHK_PTR_NULL(netDevCtx);
    deviceLogicId = (static_cast<NetDevContext*>(netDevCtx))->GetLogicId();
    HcclIpAddress localIp = (static_cast<NetDevContext*>(netDevCtx))->GetLocalIp();
    bool isBackupIpValid = !(static_cast<NetDevContext*>(netDevCtx))->GetBackupIp().IsInvalid();
    RaResourceInfo raResourceInfo;
    CHK_RET(NetworkManager::GetInstance(deviceLogicId).GetRaResourceInfo(raResourceInfo));
    rdmaHandle = raResourceInfo.nicSocketMap[localIp].nicRdmaHandle;
    CHK_PTR_NULL(rdmaHandle);
    if (isBackupIpValid) {
        HCCL_INFO(
            "[%s] before hrtGetDevice deviceLogicId[%d], isBackupIpValid[%d]", __func__, deviceLogicId,
            isBackupIpValid);
        CHK_RET(hrtGetDevice(&deviceLogicId));
        HCCL_INFO("[%s] after hrtGetDevice deviceLogicId[%d]", __func__, deviceLogicId);
    }
    // host内存地址映射
    devAddr = addr;
    if (memType == RmaMemType::HOST) {
        CHK_RET(MemMappingManager::GetInstance(deviceLogicId).GetDevVA(deviceLogicId, addr, size, devAddr));
    }
    HCCL_DEBUG("[Init]addr[%p], size[%llu], devAddr[%p], memType[%d]", addr, size, devAddr, memType);

    // 内存注册
    MrInfoT info = {};
    info.size = size;
    info.access = RA_ACCESS_REMOTE_WRITE | RA_ACCESS_LOCAL_WRITE | RA_ACCESS_REMOTE_READ;
    info.addr = devAddr;

    std::ostringstream oss;
    oss.write(reinterpret_cast<const char_t*>(&rdmaHandle), sizeof(rdmaHandle));
    oss.write(reinterpret_cast<const char_t*>(&addr), sizeof(addr));
    oss.write(reinterpret_cast<const char_t*>(&size), sizeof(size));
    devAddrID = oss.str();

    CHK_RET(hrtRaRegGlobalMr(rdmaHandle, info, mrHandle));
    HCCL_DEBUG("[Init][RegMr] LocalRdmaRmaBuffer rdmaHandle[%p], mrHandle[%p].", rdmaHandle, mrHandle);
    // 信息保存
    CHK_RET(AddDevAddrIdentifierMap(deviceLogicId, devAddrID));
    this->lkey = info.lkey;
    initialized_ = true;
    return HCCL_SUCCESS;
}

std::string& LocalRdmaRmaBufferImpl::Serialize()
{
    if (!serializeStr_.empty()) {
        return serializeStr_;
    }
    // 序列化信息
    std::ostringstream oss;
    u8 type{static_cast<u8>(rmaType)};
    oss.write(reinterpret_cast<const char_t*>(&type), sizeof(type));
    oss.write(reinterpret_cast<const char_t*>(&addr), sizeof(addr));
    oss.write(reinterpret_cast<const char_t*>(&size), sizeof(size));
    oss.write(reinterpret_cast<const char_t*>(&devAddr), sizeof(devAddr));
    oss.write(reinterpret_cast<const char_t*>(&memType), sizeof(memType));
    oss.write(reinterpret_cast<const char_t*>(&lkey), sizeof(lkey));

    serializeStr_ = oss.str();
    return serializeStr_;
}

HcclResult LocalRdmaRmaBufferImpl::Destroy()
{
    if (addr != nullptr && initialized_) {
        // 内存解注册
        HcclResult ret = HCCL_SUCCESS;
        if (mrHandle != nullptr) {
            HCCL_DEBUG("[Destroy][DeRegMr] LocalRdmaRmaBuffer rdmaHandle[%p], mrHandle[%p].", rdmaHandle, mrHandle);

            // 防止重复释放内存，仅在内存使用个数 = 0 时，释放内存
            HcclResult retDe = HCCL_SUCCESS;
            retDe = DeDevAddrIdentifierMap(deviceLogicId, devAddrID);
            if (retDe != HCCL_SUCCESS) {
                HCCL_WARNING("[Destroy][DeRegMr][DeDevAddrIdentifierMap]err[%d] deDevAddrIdentifierMap failed.", retDe);
            }
            if (!IsDevAddrExistInDevAddrIdentifierMap(deviceLogicId, devAddrID)) {
                ret = hrtRaDeRegGlobalMr(rdmaHandle, mrHandle);
            }

            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[LocalRdmaRmaBufferImpl][Destroy]deReg Global Mr failed, "
                    "ret[%d], dev[%d], ptr[%p], size[%llu]",
                    ret, deviceLogicId, addr, size);
            }
        }

        // host内存解映射
        if (memType == RmaMemType::HOST) {
            ret = MemMappingManager::GetInstance(deviceLogicId).ReleaseDevVA(deviceLogicId, addr, size);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[LocalRdmaRmaBufferImpl][Destroy]release dev va failed, "
                    "ret[%d], dev[%d], ptr[%p], size[%llu]",
                    ret, deviceLogicId, addr, size);
            }
        }

        addr = nullptr;
        size = 0;
        mrHandle = nullptr;
        devAddrID = std::string();
        initialized_ = false;
        return ret;
    }

    return HCCL_SUCCESS;
}

HcclResult LocalRdmaRmaBufferImpl::Remap(void* addr, u64 length)
{
    CHK_PTR_NULL(addr);
    CHK_PRT_RET(length == 0, HCCL_ERROR("[Remap]memorySize[%llu] must be greater than 0.", length), HCCL_E_PARA);

    struct MemRemapInfo info = {};
    info.addr = addr;
    info.size = length;
    unsigned int num = 1;
    return HrtRaRemapMr(rdmaHandle, &info, num);
}

} // namespace hccl

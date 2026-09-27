/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mr_manager.h"
#include "adapter_hal.h"
#include "adapter_hccp.h"
#include "network_manager_pub.h"
#include "../resource/socket/hccl_network.h"

namespace hccl {
using namespace std;
u64 MrManager::g_devAddr = 0;
map<HostMappingKey, HostMappingInfo> MrManager::mappedHostToDevMap_ = {};
std::mutex MrManager::mappedHostToDevMutex_;

MrManager& MrManager::GetInstance()
{
    static MrManager hcclMrManager;
    return hcclMrManager;
}

MrManager::MrManager() : rdmaHandle_(nullptr), count_(0) {}

MrManager::MrManager(HcclNetDevCtx netDevCtx) : rdmaHandle_(nullptr), count_(0), netDevCtx_(netDevCtx) {}

MrManager::~MrManager() {}

HcclResult MrManager::Init(QpHandle qpHandle, u32 devId, bool isHostMem, map<MrMapKey, MrInfo>& unRegMrMap)
{
    CHK_PTR_NULL(qpHandle);
    unique_lock<std::mutex> lockUnMrMap(unMrMapSpinMutex_);
    unRegMrMap_ = unRegMrMap;
    lockUnMrMap.unlock();
    SetHdcPara(devId, isHostMem, true);
    CHK_RET(InitMrManager(qpHandle));
    return HCCL_SUCCESS;
}

HcclResult MrManager::Init(RdmaHandle rdmaHandle, u32 devId, bool isHostMem)
{
    CHK_PTR_NULL(rdmaHandle);
    SetHdcPara(devId, isHostMem, false);
    CHK_RET(InitMrManager(rdmaHandle));
    return HCCL_SUCCESS;
}

HcclResult MrManager::Init(RdmaHandle rdmaHandle)
{
    CHK_PTR_NULL(rdmaHandle);
    return InitMrManager(rdmaHandle);
}

HcclResult MrManager::Init()
{
    CHK_PTR_NULL(netDevCtx_);
    RaResourceInfo raResourceInfo;
    s32 deviceLogicId = (static_cast<hccl::NetDevContext*>(netDevCtx_))->GetLogicId();
    HcclIpAddress localIp = (static_cast<hccl::NetDevContext*>(netDevCtx_))->GetLocalIp();
    CHK_RET(NetworkManager::GetInstance(deviceLogicId).GetRaResourceInfo(raResourceInfo));
    void* nicRdmaHandle = raResourceInfo.nicSocketMap[localIp].nicRdmaHandle;
    return InitMrManager(nicRdmaHandle);
}

HcclResult MrManager::InitUnRegMrMap()
{
    unique_lock<std::mutex> lockUnMrMap(unMrMapSpinMutex_);
    for (auto& iter : unRegMrMap_) {
        // 目前全局内存由于地址非法注册失败返回成功，需要driver修复进程退出不通知通信库解注册内存问题
        CHK_RET(RegMr(iter.second.addr, iter.second.size));
        // 内存注册失败，mrHandl为空，不用记录
        MrMapKey mrMapKey(reinterpret_cast<u64>(iter.second.addr), iter.second.size);
        unique_lock<std::mutex> lockMrMap(mrMapSpinMutex_);
        if (regedMrMap_.find(mrMapKey) != regedMrMap_.end()) {
            auto iterator = regedMrMap_.find(iter.first);
            if (iterator != regedMrMap_.end()) {
                iterator->second.gloMemRef = iter.second.gloMemRef;
            }
        }
        lockMrMap.unlock();
    }

    unRegMrMap_.clear();
    lockUnMrMap.unlock();
    return HCCL_SUCCESS;
}

HcclResult MrManager::InitUnRegMrMap(map<MrMapKey, MrInfo>& unRegMrMap)
{
    unique_lock<std::mutex> lockUnMrMap(unMrMapSpinMutex_);
    unRegMrMap_ = unRegMrMap;
    lockUnMrMap.unlock();
    CHK_RET(InitUnRegMrMap());
    return HCCL_SUCCESS;
}

HcclResult MrManager::InitMrManager(void* handle)
{
    CHK_PTR_NULL(handle);
    if (++count_ == COUNT_ONE) {
        if (isUseQPHandle_) {
            qpHandle_ = handle;
        } else {
            rdmaHandle_ = handle;
        }
        CHK_RET(InitUnRegMrMap());
    } else if (count_ > COUNT_ONE) {
        if (rdmaHandle_ != handle && qpHandle_ != handle) {
            HCCL_ERROR("[MrManager][Init]mr manager init failed, count[%d].", count_.load());
            return HCCL_E_PARA;
        }
    }
    HCCL_INFO("[MrManager][Init]mr manager init success, count[%d]", count_.load());
    return HCCL_SUCCESS;
}

HcclResult MrManager::DeInit()
{
    RaResourceInfo raResourceInfo;
    s32 deviceLogicId = (static_cast<hccl::NetDevContext*>(netDevCtx_))->GetLogicId();
    HcclIpAddress localIp = (static_cast<hccl::NetDevContext*>(netDevCtx_))->GetLocalIp();
    CHK_RET(NetworkManager::GetInstance(deviceLogicId).GetRaResourceInfo(raResourceInfo));
    void* nicRdmaHandle = raResourceInfo.nicSocketMap[localIp].nicRdmaHandle;
    return DeInit(nicRdmaHandle);
}

HcclResult MrManager::DeInit(const void* handle)
{
    CHK_PTR_NULL(handle);
    if (rdmaHandle_ == handle || qpHandle_ == handle) {
        --count_;
        if (count_ > 0) {
            HCCL_INFO("[MrManager][DeInit]mr manager deinit success, count[%d].", count_.load());
            return HCCL_SUCCESS;
        } else if (count_ == 0) {
            ReleaseMrResource();
            if (isUseQPHandle_) {
                qpHandle_ = nullptr;
            } else {
                rdmaHandle_ = nullptr;
            }
        }
    } else {
        HCCL_ERROR("[MrManager][DeInit]count[%d]", count_.load());
        return HCCL_E_PARA;
    }
    HCCL_INFO("[MrManager][DeInit]mr manager deinit success, count[%d].", count_.load());
    return HCCL_SUCCESS;
}

bool MrManager::IsRequireMapping(void* addr, u64 size, void*& devVirAddr)
{
    u64 userAddr = reinterpret_cast<u64>(addr);
    u64 userSize = size;
    if (mappedHostToDevMap_.size() == 0) {
        return true;
    }

    auto iter = SearchMappingMap(userAddr, userSize);
    if (iter != mappedHostToDevMap_.end()) {
        u64 tmpDva = reinterpret_cast<u64>(iter->second.devVirAddr) + userAddr - iter->first.addr;
        devVirAddr = reinterpret_cast<void*>(static_cast<uintptr_t>(tmpDva));
        iter->second.mappingRef++;
        return false;
    }

    return true;
}

map<MrMapKey, MrInfo> MrManager::GetUnregMap()
{
    unique_lock<std::mutex> lockUnMrMap(unMrMapSpinMutex_);
    return unRegMrMap_;
}

std::map<HostMappingKey, HostMappingInfo>::iterator MrManager::SearchMappingMap(u64 userAddr, u64 userSize)
{
    for (auto iter = mappedHostToDevMap_.begin(); iter != mappedHostToDevMap_.end(); ++iter) {
        if ((userAddr >= iter->first.addr) && (userAddr + userSize <= iter->first.size + iter->first.addr)
            && (iter->first.devId == curDevId_)) {
            return iter;
        }
    }
    return mappedHostToDevMap_.end();
}

HcclResult MrManager::RegMrImpl(void* addr, u64 size, HcclMrInfo& mrInfo, MrHandle& mrHandle, void*& devVirAddr)
{
    MrInfoT info = {};
    info.addr = mrInfo.addr;
    info.size = mrInfo.size;
    info.access = mrInfo.access;

    if (IsHostMem_) {
        unique_lock<std::mutex> lockMapping(mappedHostToDevMutex_);
        CHK_RET(MapMem(addr, size, devVirAddr));
        lockMapping.unlock();
        info.addr = devVirAddr;
    }

    if (isUseQPHandle_) {
        CHK_RET(HrtRaMrReg(qpHandle_, &info));
    } else {
        CHK_RET(hrtRaRegGlobalMr(rdmaHandle_, info, mrHandle));
    }

    mrInfo.addr = addr;
    mrInfo.lkey = info.lkey;
    return HCCL_SUCCESS;
}

HcclResult MrManager::MapMem(void* addr, u64 size, void*& devVirAddr)
{
    CHK_PTR_NULL(addr);
    if (IsRequireMapping(addr, size, devVirAddr)) {
        DevType devType;
        CHK_RET(hrtHalGetDeviceType(curDevId_, devType));
        if ((devType == DevType::DEV_TYPE_910B) || (devType == DevType::DEV_TYPE_910_93)) {
            // 910B环境传参要特殊处理
            HCCL_INFO(
                "[MrManager][MapMem]hrtHalHostRegister addr[%p], size[%llu Byte], flag[%u], devId[%u]", addr, size,
                HOST_MEM_MAP_DEV_PCIE_TH, curDevId_);
            CHK_RET(hrtHalHostRegister(addr, size, HOST_MEM_MAP_DEV_PCIE_TH, curDevId_, devVirAddr));
        } else {
            CHK_RET(hrtHalHostRegister(addr, size, HOST_MEM_MAP_DEV, curDevId_, devVirAddr));
        }
        HostMappingKey hostMappingKey(reinterpret_cast<u64>(addr), size, curDevId_);
        mappedHostToDevMap_[hostMappingKey].devVirAddr = devVirAddr;
    }
    return HCCL_SUCCESS;
}

HcclResult MrManager::DeRegMrImpl(MrInfo mrInfo)
{
    HcclMrInfo mrInfoTmp;
    if (isUseQPHandle_) {
        // 注销MR
        TransMrInfo((IsHostMem_) ? mrInfo.devVirAddr : mrInfo.addr, mrInfo.size, mrInfoTmp);
        MrInfoT hccpMrInfoTmp = {};
        hccpMrInfoTmp.addr = mrInfoTmp.addr;
        hccpMrInfoTmp.size = mrInfoTmp.size;
        hccpMrInfoTmp.access = mrInfoTmp.access;
        hccpMrInfoTmp.lkey = mrInfoTmp.lkey;
        CHK_RET(HrtRaMrDereg(qpHandle_, &hccpMrInfoTmp));
    } else {
        CHK_RET(hrtRaDeRegGlobalMr(rdmaHandle_, mrInfo.mrHandle));
    }
    if (IsHostMem_) {
        CHK_RET(UnmapMem(mrInfo));
    }
    return HCCL_SUCCESS;
}

HcclResult MrManager::DelayedReg(void* addr, u64 size)
{
    CHK_PTR_NULL(addr);
    unique_lock<std::mutex> lockUnMrMap(unMrMapSpinMutex_);
    MrMapKey key(reinterpret_cast<u64>(addr), size);
    MrInfo info(addr, size);
    auto iter = unRegMrMap_.find(key);
    if (iter == unRegMrMap_.end()) {
        info.gloMemRef++;
        unRegMrMap_.emplace(key, info);
    } else {
        iter->second.gloMemRef++;
    }

    unique_lock<std::mutex> lock(addrSizeMutex_);
    globalAddrSizeMap_[addr] = size;
    lock.unlock();

    HCCL_INFO(
        "[MrManager][RecordMr]record mr info success, size[%llu Byte], unRegMrMap size[%u].", size, unRegMrMap_.size());
    return HCCL_SUCCESS;
}

HcclResult MrManager::RegGlobalMr(void* addr, u64 size)
{
    CHK_PTR_NULL(addr);

    // count = 0时表示没有初始化通信域,只需将内存信息记录到未注册内存unRegMrMap_中，无需注册MR等动作
    if (count_ == 0) {
        CHK_RET(DelayedReg(addr, size));
    } else {
        CHK_RET(RegMr(addr, size));
    }

    return HCCL_SUCCESS;
}

HcclResult MrManager::RegMr(void* addr, u64 size)
{
    CHK_PTR_NULL(addr);
    CHK_PRT_RET(
        (size == 0), HCCL_ERROR("[MrManager][RegTmpMr]memory size[%llu Byte] should be greater than 0.", size),
        HCCL_E_PARA);
    HcclMrInfo mrInfo;
    mrInfo.addr = addr;
    mrInfo.size = size;
    mrInfo.access = RA_ACCESS_REMOTE_WRITE | RA_ACCESS_LOCAL_WRITE | RA_ACCESS_REMOTE_READ;

    unique_lock<std::mutex> lockMrMap(mrMapSpinMutex_);
    MrMapKey mrMapKey(reinterpret_cast<u64>(addr), size);
    auto iter = regedMrMap_.find(mrMapKey);
    // 防止重复注册
    if (iter != regedMrMap_.end()) {
        HCCL_WARNING("[MrManager][RegMr]mr map addr is already exists, size[%llu Byte].", iter->second.size);
        iter->second.gloMemRef++;
        unique_lock<std::mutex> lock(addrSizeMutex_);
        globalAddrSizeMap_[addr] = size;
        lock.unlock();
        return HCCL_SUCCESS;
    }

    lockMrMap.unlock();
    MrHandle mrHandle = nullptr;
    void* devVirAddr = nullptr;
    CHK_RET(RegMrImpl(addr, size, mrInfo, mrHandle, devVirAddr));
    if (!isUseQPHandle_ && mrHandle == nullptr) {
        HCCL_WARNING("[MrManager][RegMr]global mr register not success, addr[%p], size[%u Byte]", addr, size);
        return HCCL_SUCCESS;
    }

    MrInfo tmpMrInfo{};
    tmpMrInfo = mrInfo;
    if (!isUseQPHandle_) {
        tmpMrInfo.mrHandle = mrHandle;
    }

    tmpMrInfo.gloMemRef++;
    tmpMrInfo.devVirAddr = devVirAddr;

    lockMrMap.lock();
    regedMrMap_.emplace(mrMapKey, tmpMrInfo);
    lockMrMap.unlock();

    unique_lock<std::mutex> lock(addrSizeMutex_);
    globalAddrSizeMap_[addr] = size;
    lock.unlock();

    HCCL_INFO(
        "[MrManager][RegGlobalMr]global mr register success, size[%llu Byte], regMrMap size[%u].", size,
        regedMrMap_.size());
    return HCCL_SUCCESS;
}

HcclResult MrManager::RegTmpMr(void* addr, u64 size, u32& lkey) // 注册临时MR
{
    CHK_PTR_NULL(addr);
    CHK_PRT_RET(
        (size == 0), HCCL_ERROR("[MrManager][RegTmpMr]memory size[%llu Byte] should be greater than 0.", size),
        HCCL_E_PARA);

    HcclMrInfo mrInfo;
    mrInfo.addr = addr;
    mrInfo.size = size;
    mrInfo.access = RA_ACCESS_REMOTE_WRITE | RA_ACCESS_LOCAL_WRITE | RA_ACCESS_REMOTE_READ;
    u64 uAddr = reinterpret_cast<u64>(addr);
    MrMapKey tmpMrMapKey(uAddr, size);

    unique_lock<std::mutex> lockMrMap(mrMapSpinMutex_);
    auto iter = regedMrMap_.find(tmpMrMapKey);
    if (iter != regedMrMap_.end()) {
        iter->second.tmpMemRef++;
        lkey = iter->second.lkey;
        HCCL_INFO(
            "[MrManager][RegTmpMr]temp mr find success, size[%llu Byte], temp mr map size[%u], "
            "glo count[%d].",
            size, regedMrMap_.size(), iter->second.gloMemRef);
        return HCCL_SUCCESS;
    }

    lockMrMap.unlock();
    MrHandle mrHandle = nullptr;
    void* devVirAddr = nullptr;
    CHK_RET(RegMrImpl(addr, size, mrInfo, mrHandle, devVirAddr));
    if (!isUseQPHandle_ && mrHandle == nullptr) {
        HCCL_ERROR("[MrManager][RegTmpMr]temp mr register failed, size[%u Byte]", size);
        return HCCL_E_NETWORK;
    }

    MrInfo tmpMrInfo{};
    tmpMrInfo = mrInfo;
    if (!isUseQPHandle_) {
        tmpMrInfo.mrHandle = mrHandle;
    }
    tmpMrInfo.devVirAddr = devVirAddr;
    // 目前这个全局地址只有hdc模式下用，而hdc模式可能以qpHandle与rdmaHandle两种粒度去注册MR
    g_devAddr = (u64)devVirAddr;
    tmpMrInfo.tmpMemRef++;

    lockMrMap.lock();
    regedMrMap_.emplace(tmpMrMapKey, tmpMrInfo);
    lockMrMap.unlock();

    lkey = mrInfo.lkey;
    HCCL_INFO(
        "[MrManager][RegTmpMr]temp mr register success, size[%llu Byte], temp mr map size[%u]", size,
        regedMrMap_.size());
    return HCCL_SUCCESS;
}

HcclResult MrManager::DeRegGlobalMr(void* addr)
{
    CHK_PTR_NULL(addr);
    HCCL_INFO("[MrManager][DeRegGlobalMr] addr[%p]", hash<void*>{}(addr));
    unique_lock<std::mutex> lock(addrSizeMutex_);
    if (globalAddrSizeMap_.find(addr) == globalAddrSizeMap_.end()) {
        HCCL_ERROR("[MrManager][DeRegGlobalMr] is not found");
        return HCCL_E_PARA;
    }

    MrMapKey key(reinterpret_cast<u64>(addr), globalAddrSizeMap_[addr]);
    lock.unlock();

    unique_lock<std::mutex> lockUnMrMap(unMrMapSpinMutex_);
    auto aiter = unRegMrMap_.find(key);
    if (aiter != unRegMrMap_.end()) {
        aiter->second.gloMemRef--;
        if (aiter->second.gloMemRef == 0) {
            unRegMrMap_.erase(key);
        }

        HCCL_INFO("[MrManager][DeRecordMr]derecord global mr info success, unRegMrMap size[%u]", unRegMrMap_.size());
        return HCCL_SUCCESS;
    }

    lockUnMrMap.unlock();
    unique_lock<std::mutex> lockMrMap(mrMapSpinMutex_);
    auto iter = regedMrMap_.find(key);
    if (iter != regedMrMap_.end()) {
        iter->second.gloMemRef--;
        if (iter->second.gloMemRef > 0 || iter->second.tmpMemRef > 0) {
            HCCL_INFO(
                "[MrManager][DeRegGlobalMr] minus count[%d] tmp count[%d] success, regMrMap size[%u].",
                iter->second.gloMemRef, iter->second.tmpMemRef, regedMrMap_.size());
            return HCCL_SUCCESS;
        }

        if (iter->second.size > 0) {
            CHK_RET(DeRegMrImpl(iter->second));
        }

        regedMrMap_.erase(key);
        lockMrMap.unlock();
        HCCL_INFO("[MrManager][DeRegGlobalMr]addr deregister success, regMrMap size[%u].", regedMrMap_.size());
    } else {
        HCCL_ERROR(
            "[MrManager][DeRegGlobalMr]addr was not found, unRegMrMap size[%u], regMrMap size[%u].", unRegMrMap_.size(),
            regedMrMap_.size());
        return HCCL_E_MEMORY;
    }
    HCCL_INFO("[MrManager][DeRegGlobalMr] DeReg GlobalMr end");
    return HCCL_SUCCESS;
}

HcclResult MrManager::UnmapMem(MrInfo mrInfo)
{
    unique_lock<std::mutex> lockMapping(mappedHostToDevMutex_);
    u64 userAddr = reinterpret_cast<u64>(mrInfo.addr);
    auto iter = SearchMappingMap(userAddr, mrInfo.size);
    CHK_PRT_RET(
        (iter == mappedHostToDevMap_.end()),
        HCCL_ERROR("[MrManager][UnmapMem]the memory to be deregistered has not been registered"), HCCL_E_PARA);
    if (iter->second.mappingRef == 0) {
        // 解除内存映射
        CHK_RET(hrtHalHostUnregister(mrInfo.addr, curDevId_));
        mappedHostToDevMap_.erase(iter->first);
    } else {
        iter->second.mappingRef--;
    }
    return HCCL_SUCCESS;
}

HcclResult MrManager::GetKey(void* addr, u64 size, u32& lkey) // 获取内存的lkey
{
    CHK_PTR_NULL(addr);
    CHK_PRT_RET(
        (size == 0), HCCL_ERROR("[MrManager][GetKey]memory size[%llu Byte] should be greater than 0.", size),
        HCCL_E_PARA);

    MrInfo mrInfo(addr, size);
    unique_lock<std::mutex> lockMrMap(mrMapSpinMutex_);
    bool isEmpty = regedMrMap_.empty();
    lockMrMap.unlock();
    if (isEmpty) {
        CHK_PRT_RET(
            (RegTmpMr(addr, size, lkey) != HCCL_SUCCESS),
            HCCL_ERROR("[MrManager][GetKey]register temp memory error, size[%llu Byte].", size), HCCL_E_INTERNAL);
    } else {
        bool isInfoNotFound = false;
        CHK_PRT_RET(
            (GetMrInfo(mrInfo, isInfoNotFound) != HCCL_SUCCESS),
            HCCL_ERROR("[MrManager][GetKey]get memory info error, size[%llu Byte].", size), HCCL_E_INTERNAL);
        if (isInfoNotFound) {
            CHK_PRT_RET(
                (RegTmpMr(addr, size, lkey) != HCCL_SUCCESS),
                HCCL_ERROR("[MrManager][GetKey]register temp memory error, size[%llu Byte].", size), HCCL_E_INTERNAL);
        } else {
            lockMrMap.lock();
            MrMapKey key(reinterpret_cast<u64>(mrInfo.addr), mrInfo.size);
            auto iter = regedMrMap_.find(key);
            if (iter != regedMrMap_.end()) {
                iter->second.tmpMemRef++;
                lkey = mrInfo.lkey;
                HCCL_INFO(
                    "[MrManager][GetKey]get memory lkey success, size[%llu Byte], regMrMap size[%u], "
                    "temp mr map size[%u].",
                    size, regedMrMap_.size(), regedMrMap_.size());
            } else {
                // GetMrInfo释放锁到重新加锁期间，该MR可能已被并发释放，退回临时注册路径
                lockMrMap.unlock();
                CHK_PRT_RET(
                    (RegTmpMr(addr, size, lkey) != HCCL_SUCCESS),
                    HCCL_ERROR("[MrManager][GetKey]register temp memory error, size[%llu Byte].", size),
                    HCCL_E_INTERNAL);
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult MrManager::ReleaseKey(void* addr, u64 size) // 释放临时MR
{
    CHK_PTR_NULL(addr);
    CHK_PRT_RET(
        (size == 0), HCCL_ERROR("[MrManager][ReleaseKey]memory size[%llu Byte] should be greater than 0.", size),
        HCCL_E_PARA);

    HcclResult ret = HCCL_SUCCESS;
    MrInfo mrInfo;
    mrInfo.addr = addr;
    mrInfo.size = size;
    bool isInfoNotFound = false;
    ret = GetMrInfo(mrInfo, isInfoNotFound);
    if (ret != HCCL_SUCCESS || isInfoNotFound) {
        HCCL_ERROR("[MrManager][ReleaseKey]get memory info error, size[%llu Byte].", size);
        return HCCL_E_INTERNAL;
    }

    MrMapKey tmpMrMapKey;
    tmpMrMapKey.addr = reinterpret_cast<u64>(mrInfo.addr);
    tmpMrMapKey.size = mrInfo.size;

    unique_lock<std::mutex> lockMrMap(mrMapSpinMutex_);
    auto iter = regedMrMap_.find(tmpMrMapKey);
    CHK_PRT_RET(
        (iter == regedMrMap_.end()),
        HCCL_ERROR(
            "[MrManager][ReleaseKey] release key failed, size[%llu Byte], "
            "mrInfo.size[%llu], regMrMap size[%u].",
            size, mrInfo.size, regedMrMap_.size()),
        HCCL_E_INTERNAL);

    --iter->second.tmpMemRef;
    if (iter->second.tmpMemRef > 0 || iter->second.gloMemRef > 0) {
        HCCL_INFO(
            "[MrManager][ReleaseKey]release key success, size[%llu Byte], tmpMrMap size[%u], count[%d] "
            "tmp count[%d].",
            size, regedMrMap_.size(), iter->second.gloMemRef, iter->second.tmpMemRef);
        return HCCL_SUCCESS;
    } else if (iter->second.tmpMemRef < 0) {
        HCCL_ERROR(
            "[MrManager][ReleaseKey]release key error, size[%llu Byte], count[%d].", size, iter->second.tmpMemRef);
        return HCCL_E_MEMORY;
    }

    CHK_RET(DeRegMrImpl(iter->second));
    HCCL_INFO(
        "[MrManager][ReleaseKey] deregister success, size[%llu Byte], "
        "temp mr map size[%u].",
        size, regedMrMap_.size());
    regedMrMap_.erase(iter);
    lockMrMap.unlock();
    return HCCL_SUCCESS;
}

HcclResult MrManager::GetMrInfo(MrInfo& mrInfo, bool& isInfoNotFound)
{
    CHK_PRT_RET(
        regedMrMap_.empty(), HCCL_ERROR("[MrManager][GetMrInfo]get mr info failed, mr map is empty"), HCCL_E_PARA);

    isInfoNotFound = false;
    unique_lock<std::mutex> lockMrMap(mrMapSpinMutex_);

    u64 uAddr = reinterpret_cast<u64>(mrInfo.addr);
    u64 size = mrInfo.size;
    MrMapKey key(uAddr, size);
    auto iter = regedMrMap_.find(key);
    if (iter != regedMrMap_.end()) {
        if (iter->second.size >= size) {
            mrInfo = iter->second;
            HCCL_DEBUG("[MrManager][GetMrInfo]get memory info success, size[%llu].", iter->second.size);
        } else {
            isInfoNotFound = true;
            HCCL_WARNING(
                "[MrManager][GetMrInfo]mr addr size[%llu], but required addr size[%llu].", iter->second.size,
                mrInfo.size);
        }

        return HCCL_SUCCESS;
    }

    iter = regedMrMap_.upper_bound(key);
    if (iter != regedMrMap_.begin()
        && !(iter != regedMrMap_.end() && iter->first.addr == uAddr && iter->first.size >= size)) {
        iter--;
    }

    u64 uTmpAddr = iter->first.addr;
    u64 tmpSize = iter->second.size;
    if (((uTmpAddr <= uAddr) && (uAddr < (uTmpAddr + tmpSize)))
        && ((uTmpAddr < (uAddr + size)) && ((uAddr + size) <= (uTmpAddr + tmpSize)))) {
        mrInfo = iter->second;
    } else {
        HCCL_WARNING("[MrManager][GetMrInfo] size[%llu] was not found.", mrInfo.size);
        isInfoNotFound = true;
        return HCCL_SUCCESS;
    }
    HCCL_DEBUG("[MrManager][GetMrInfo]get memory info success, size[%llu]", mrInfo.size);
    return HCCL_SUCCESS;
}

HcclResult MrManager::GetDevVirAddr(void* addr, u64 size, u64& devVirAddr)
{
    CHK_PTR_NULL(addr);
    CHK_PRT_RET(
        (size == 0), HCCL_ERROR("[MrManager][GetDevVirAddr]memory size[%llu Byte] should be greater than 0.", size),
        HCCL_E_PARA);
    MrInfo mrInfo(addr, size);
    bool isInfoNotFound = false;
    CHK_PRT_RET(
        (GetMrInfo(mrInfo, isInfoNotFound) != HCCL_SUCCESS),
        HCCL_ERROR("[MrManager][GetDevVirAddr]get memory info error, size[%llu Byte].", size), HCCL_E_INTERNAL);
    CHK_PRT_RET(
        isInfoNotFound,
        HCCL_ERROR("[MrManager][GetDevVirAddr]get memory info fail, addr[%p], size[%llu Byte].", addr, size),
        HCCL_E_PARA);
    devVirAddr
        = reinterpret_cast<u64>(mrInfo.devVirAddr) + reinterpret_cast<u64>(addr) - reinterpret_cast<u64>(mrInfo.addr);

    return HCCL_SUCCESS;
}

void MrManager::TransMrInfo(void* addr, u64 size, HcclMrInfo& mrInfo)
{
    mrInfo.addr = addr;
    mrInfo.size = size;
    mrInfo.access = RA_ACCESS_REMOTE_WRITE | RA_ACCESS_LOCAL_WRITE | RA_ACCESS_REMOTE_READ;
}

HcclResult MrManager::ReleaseMrResource()
{
    HCCL_INFO("[MrManager][ReleaseMrResource]start release mr resource");
    unique_lock<std::mutex> lockMrMap(mrMapSpinMutex_);
    if (!regedMrMap_.empty()) {
        unique_lock<std::mutex> lockUnMrMap(unMrMapSpinMutex_);
        unRegMrMap_ = regedMrMap_;
        lockUnMrMap.unlock();
        u64 bound = regedMrMap_.begin()->first.addr;
        for (auto& iter : regedMrMap_) {
            if (iter.first.addr >= bound && iter.second.size != 0) {
                HCCL_DEBUG("deinit addr[%llu], size[%llu]", hash<void*>{}(iter.second.addr), iter.second.size);
                CHK_RET(DeRegMrImpl(iter.second));
                bound = iter.first.addr + iter.first.size;
            }
        }

        regedMrMap_.clear();
    }

    HCCL_INFO("[MrManager][ReleaseMrResource]release memory resource success.");
    return HCCL_SUCCESS;
}

void MrManager::SetHdcPara(u32 devId, bool isHostMem, bool isUseQPHandle)
{
    isUseQPHandle_ = isUseQPHandle;
    curDevId_ = devId;
    IsHostMem_ = isHostMem;
}

} // namespace hccl

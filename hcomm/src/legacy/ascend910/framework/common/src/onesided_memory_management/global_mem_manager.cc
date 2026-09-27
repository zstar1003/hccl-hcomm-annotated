/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "global_mem_manager.h"

#include <string>
#include "hccl_mem.h"

namespace hccl {
GlobalMemRegMgr::~GlobalMemRegMgr() {}

GlobalMemRegMgr& GlobalMemRegMgr::GetInstance()
{
    // reserve 1 instance for invalid deviceid and host
    static GlobalMemRegMgr instance[MAX_MODULE_DEVICE_NUM + 1];
    s32 deviceLogicID = 0;

    HcclResult hcclRet = hrtGetDeviceRefresh(&deviceLogicID);
    if (hcclRet != HCCL_SUCCESS) {
        HCCL_RUN_WARNING(
            "GlobalMemRegMgr::GetInstance hrtGetDeviceRefresh failed, ret[%d], "
            "return reserve instance",
            hcclRet);
        return instance[MAX_MODULE_DEVICE_NUM];
    }

    if (static_cast<u32>(deviceLogicID) >= MAX_MODULE_DEVICE_NUM || deviceLogicID <= HOST_DEVICE_ID) {
        HCCL_RUN_WARNING("[Get][Instance]deviceLogicID[%d] is invalid, return reserve instance", deviceLogicID);
        return instance[MAX_MODULE_DEVICE_NUM];
    }

    HCCL_INFO("GlobalMemRegMgr::GetInstance deviceLogicID[%d].", deviceLogicID);
    return instance[deviceLogicID];
}

HcclResult GlobalMemRegMgr::Destroy()
{
    HCCL_INFO("[GlobalMemRegMgr][%s] start.", __func__);
    std::unique_lock<std::mutex> lock(netDevCtxMtx_);
    for (auto& pair : netDevCtxMap_) {
        if (pair.second.first == NicType::DEVICE_NIC_TYPE) {
            socketManager_->ServerDeInit(pair.first.ip, pair.first.listenPort);
        }
        HcclNetCloseDev(pair.second.second);
        HCCL_INFO("[GlobalMemRegMgr][%s] Close netdev[%p].", __func__, pair.second.second);
    }
    netDevCtxMap_.clear();
    lock.unlock();
    CHK_RET(DeInitNic());
    HCCL_INFO("[GlobalMemRegMgr][%s] end.", __func__);
    return HCCL_SUCCESS;
}

HcclResult GlobalMemRegMgr::CheckOverlapAndInsert(GlobalMemRecord& memRecord, void** memRecordHandle)
{
    // 由于每次插入都会保证不产生重叠，所以只需要检查最接近的两条记录是否有重叠即可
    const auto memInfo = memRecord.PrintInfo();

    auto it = memRecordSet_.lower_bound(memRecord);
    if (it != memRecordSet_.cend()) {
        if (memRecord == *it) {
            // 已经存在相同的记录，取出地址作为handle
            *memRecordHandle = const_cast<GlobalMemRecord*>(&(*it));
            HCCL_INFO(
                "[GlobalMemRegMgr][CheckOverlapAndInsert] The memory[%s] has been registered already.",
                memInfo.c_str());
            return HCCL_SUCCESS;
        }

        // 检查后一个记录
        if (memRecord.HasOverlap(*it)) {
            // 后一个记录有重叠，报错
            HCCL_ERROR(
                "[GlobalMemRegMgr][CheckOverlapAndInsert] The new memory[%s] overlaps with an existing memory[%s].",
                memInfo.c_str(), (*it).PrintInfo().c_str());
            return HCCL_E_PARA;
        }
    }

    // 检查前一个记录
    if (it != memRecordSet_.cbegin()) {
        auto prevIt = std::prev(it);
        if (memRecord.HasOverlap(*prevIt)) {
            // 前一个记录有重叠，报错
            HCCL_ERROR(
                "[GlobalMemRegMgr][CheckOverlapAndInsert] The new memory[%s] overlaps with an existing memory[%s].",
                memInfo.c_str(), (*prevIt).PrintInfo().c_str());
            return HCCL_E_PARA;
        }
    }

    // 没有重叠，插入在当前it附近的位置
    auto insertIt = memRecordSet_.insert(it, std::move(memRecord));

    // 取出地址作为handle
    *memRecordHandle = const_cast<GlobalMemRecord*>(&(*insertIt));

    return HCCL_SUCCESS;
}

HcclResult GlobalMemRegMgr::Reg(const HcclMem* mem, void** memRecordHandle)
{
    // 不允许注册空内存，报错退出
    CHK_PTR_NULL(mem);
    CHK_PRT_RET(
        mem->addr == nullptr, HCCL_ERROR("[GlobalMemRegMgr][Reg] The address of mem[%p] to register is null.", mem),
        HCCL_E_PARA);
    CHK_PRT_RET(
        mem->size == 0, HCCL_ERROR("[GlobalMemRegMgr][Reg] The size of mem[%p] to register is 0.", mem), HCCL_E_PARA);

    GlobalMemRecord newRecord(mem);
    const auto memInfo = newRecord.PrintInfo();
    std::unique_lock<std::mutex> lock(lock_);
    CHK_RET(CheckOverlapAndInsert(newRecord, memRecordHandle));
    HCCL_INFO("[GlobalMemRegMgr][Reg] Added a new memory record[%s], handle[%p].", memInfo.c_str(), *memRecordHandle);

    // 记录地址，便于其他接口进行入参handle合法性校验
    validHandlePtrSet.emplace(*memRecordHandle);

    return HCCL_SUCCESS;
}

HcclResult GlobalMemRegMgr::DeReg(void* memRecordHandle)
{
    const auto* memRecordPtr = static_cast<GlobalMemRecord*>(memRecordHandle);
    const auto memInfo = memRecordPtr->PrintInfo();
    std::unique_lock<std::mutex> lock(lock_);

    // 先找到指向这个记录的迭代器
    const auto it = memRecordSet_.find(*memRecordPtr);
    if (it == memRecordSet_.cend()) {
        // 找不到记录报错退出
        HCCL_ERROR("[GlobalMemRegMgr][DeReg] Cannot find the corresponding record of memory[%s].", memInfo.c_str());
        return HCCL_E_NOT_FOUND;
    }

    // 检查内存记录是否还与通信域绑定
    if (memRecordPtr->IsBeingBound()) {
        // 该内存还与一个或多个通信域绑定，报错并打印绑定的信息
        const auto boundComm = memRecordPtr->GetBoundComm();
        HCCL_ERROR(
            "[GlobalMemRegMgr][DeReg] Cannot deregister memory[%s] since it is still bound to comm(s) listed below:",
            memInfo.c_str());

        for (const auto& commIdentifier : boundComm) {
            HCCL_ERROR("[GlobalMemRegMgr][DeReg][bound comm] %s", commIdentifier.c_str());
        }

        HCCL_ERROR("[GlobalMemRegMgr][DeReg] Please unbind from all bound comm first.");
        return HCCL_E_PARA;
    }
    HcclResult ret = HCCL_SUCCESS;
    auto regBufInfo = memRecordPtr->GetAllRegBufInfo();
    for (auto& pair : regBufInfo) {
        do {
            ret = HcclMemDereg(&pair.second); // 需循环调用DeregMem解注册注册内存(一块内存多次Reg的情况，内部有计数)
            CHK_PRT_CONT(
                ret != HCCL_SUCCESS && ret != HCCL_E_AGAIN,
                HCCL_ERROR(
                    "[GlobalMemRegMgr][DeReg] Dereg global mem failed, addr[%p] size[%lu].", pair.second.addr,
                    pair.second.len));
        } while (ret == HCCL_E_AGAIN);
    }

    // 清除记录，析构时会触发网络设备的解注册
    memRecordSet_.erase(it);
    HCCL_INFO("[GlobalMemRegMgr][DeReg] Memory[%s] has been deregistered.", memInfo.c_str());

    // 当内存全部解注册后，主动释放网络资源
    if (memRecordSet_.empty()) {
        CHK_RET(Destroy());
    }

    validHandlePtrSet.erase(memRecordHandle);
    return HCCL_SUCCESS;
}

HcclResult GlobalMemRegMgr::InitNic()
{
    std::lock_guard<std::mutex> lock(netDevCtxMtx_);
    if (nicInited_) {
        HCCL_INFO("[InitNic] Nic has been inited. devicePhyId[%u], deviceLogicId[%d]", devicePhyId_, deviceLogicId_);
        return HCCL_SUCCESS;
    }

    if (devicePhyId_ == INVALID_UINT || deviceLogicId_ == INVALID_INT) {
        CHK_RET(hrtGetDevice(&deviceLogicId_));
        CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(deviceLogicId_), devicePhyId_));
    }
    CHK_RET(HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, devicePhyId_, static_cast<u32>(deviceLogicId_), false));
    nicInited_ = true;
    socketManager_.reset(new (std::nothrow)
                             HcclSocketManager(NICDeployment::NIC_DEPLOYMENT_DEVICE, deviceLogicId_, devicePhyId_, 0));
    CHK_PTR_NULL(socketManager_);
    HCCL_INFO("[InitNic] Nic init success, devicePhyId[%u], deviceLogicId[%d]", devicePhyId_, deviceLogicId_);
    return HCCL_SUCCESS;
}

HcclResult GlobalMemRegMgr::DeInitNic()
{
    if (!nicInited_) {
        HCCL_INFO(
            "[DeInitNic] Nic has been deinited. devicePhyId[%u], deviceLogicId[%d]", devicePhyId_, deviceLogicId_);
        return HCCL_SUCCESS;
    }

    if (devicePhyId_ == INVALID_UINT || deviceLogicId_ == INVALID_INT) {
        CHK_RET(hrtGetDevice(&deviceLogicId_));
        CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(deviceLogicId_), devicePhyId_));
    }
    CHK_RET(HcclNetDeInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, devicePhyId_, static_cast<u32>(deviceLogicId_)));
    nicInited_ = false;
    HCCL_INFO("[DeInitNic] Nic deinit success. devicePhyId[%u], deviceLogicId[%d]", devicePhyId_, deviceLogicId_);
    return HCCL_SUCCESS;
}

HcclResult GlobalMemRegMgr::CheckOneSidedBackupAndSetDevId(
    const HcclIpAddress& ipAddr, u32& backupDevPhyId, u32& backupDevLogicId, std::vector<HcclIpAddress>& localIpList,
    bool& isOneSidedTaskAndBackupInitA3) const
{
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (deviceType != DevType::DEV_TYPE_910_93) {
        isOneSidedTaskAndBackupInitA3 = false;
        HCCL_INFO(
            "[GlobalMemRegMgr::CheckOneSidedBackupAndSetDevId] deviceType[%d] is not 910_93, One sided backup not "
            "support",
            static_cast<u32>(deviceType));
        return HCCL_SUCCESS;
    }
    CHK_RET(hrtGetPairDevicePhyId(devicePhyId_, backupDevPhyId));
    CHK_RET(hrtRaGetDeviceIP(devicePhyId_, localIpList));
    std::vector<HcclIpAddress> backupIpList;
    std::vector<std::vector<HcclIpAddress>> chipDeviceIPs;
    CHK_RET(hrtRaGetDeviceAllNicIP(chipDeviceIPs));
    if (chipDeviceIPs.empty()) {
        HCCL_RUN_WARNING(
            "[GlobalMemRegMgr::CheckOneSidedBackupAndSetDevId] chipDeviceIPs is empty, system nic ip may not set.");
        isOneSidedTaskAndBackupInitA3 = false;
        return HCCL_SUCCESS;
    }
    u32 ipIdex = 1U - (devicePhyId_ % 2U);
    std::copy_if(
        chipDeviceIPs[ipIdex].begin(), chipDeviceIPs[ipIdex].end(), std::back_inserter(backupIpList),
        [](const HcclIpAddress& ip) {
            return !ip.IsIPv6();
        });
    auto equalToLocal = [&ipAddr](const HcclIpAddress& entry) {
        return entry == ipAddr;
    };
    isOneSidedTaskAndBackupInitA3 = !std::any_of(localIpList.begin(), localIpList.end(), equalToLocal)
                                    && std::any_of(backupIpList.begin(), backupIpList.end(), equalToLocal);
    if (isOneSidedTaskAndBackupInitA3) {
        CHK_RET(hrtGetDeviceIndexByPhyId(backupDevPhyId, backupDevLogicId));
    }
    HCCL_INFO(
        "[GlobalMemRegMgr::CheckOneSidedBackupAndSetDevI]devicePhysicID[%u], localIpList[%s], backupDevPhyId[%d], "
        "backupDeviceIP[0]:[%s],"
        "isOneSidedTaskAndBackupInitA3[%s]",
        devicePhyId_, localIpList[0].GetReadableAddress(), backupDevPhyId, backupIpList[0].GetReadableAddress(),
        isOneSidedTaskAndBackupInitA3 ? "true" : "false");
    return HCCL_SUCCESS;
}

HcclResult
GlobalMemRegMgr::GetNetDevCtx(NicType nicType, const HcclIpAddress& ipAddr, u32 port, HcclNetDevCtx& netDevCtx)
{
    if (devicePhyId_ == INVALID_UINT || deviceLogicId_ == INVALID_INT) {
        CHK_RET(hrtGetDevice(&deviceLogicId_));
        CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(deviceLogicId_), devicePhyId_));
    }
    HCCL_INFO("[GlobalMemRegMgr][GetNetDevCtx] nicType[%d], ip[%s]", nicType, ipAddr.GetReadableAddress());

    u32 backupDevPhyId = INVALID_INT;
    u32 backupDevLogicId = INVALID_INT;
    bool isOneSidedTaskAndBackupInitA3 = false;
    std::vector<HcclIpAddress> localIpList;
    CHK_RET(CheckOneSidedBackupAndSetDevId(
        ipAddr, backupDevPhyId, backupDevLogicId, localIpList, isOneSidedTaskAndBackupInitA3));
    HCCL_INFO(
        "[GlobalMemRegMgr][GetNetDevCtx] nicType[%d], ip[%s], port[%u]", nicType, ipAddr.GetReadableAddress(), port);

    std::lock_guard<std::mutex> lock(netDevCtxMtx_);
    // 进程粒度open dev，如果已open，直接复用
    PortInfo portInfo(ipAddr, port);
    if (netDevCtxMap_.find(portInfo) != netDevCtxMap_.end()) {
        netDevCtx = netDevCtxMap_[portInfo].second;
        CHK_PTR_NULL(netDevCtx);
        return HCCL_SUCCESS;
    }
    HcclNetDevCtx tempNetDevCtx;
    if (isOneSidedTaskAndBackupInitA3) {
        HCCL_INFO(
            "[GlobalMemRegMgr::GetNetDevCtx] OneSidedService backupInit: backupDevPhyId[%d], backupDevLogicId[%d], "
            "localIp[%s], backupIp[%s]",
            backupDevPhyId, backupDevLogicId, localIpList[0].GetReadableAddress(), ipAddr.GetReadableAddress());
        CHK_RET(HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, backupDevPhyId, backupDevLogicId, false, true));
        CHK_RET(
            HcclNetInit(NICDeployment::NIC_DEPLOYMENT_DEVICE, devicePhyId_, static_cast<u32>(deviceLogicId_), false));
        CHK_RET(HcclNetOpenDev(&tempNetDevCtx, nicType, backupDevPhyId, backupDevLogicId, ipAddr, localIpList[0]));
    } else {
        CHK_RET(HcclNetOpenDev(&tempNetDevCtx, nicType, devicePhyId_, deviceLogicId_, ipAddr));
    }
    CHK_PTR_NULL(tempNetDevCtx);
    netDevCtxMap_.insert(std::make_pair(portInfo, std::make_pair(nicType, tempNetDevCtx)));
    netDevCtx = tempNetDevCtx;
    if (nicType == NicType::DEVICE_NIC_TYPE) {
        CHK_RET(socketManager_->ServerInit(netDevCtx, port));
    }
    HCCL_INFO(
        "[GlobalMemRegMgr][GetNetDevCtx] nicType[%d] ip[%s] has been Init.", nicType, ipAddr.GetReadableAddress());
    return HCCL_SUCCESS;
}

} // namespace hccl

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_var_event_res_mgr.h"

#include <mutex>

#include "ccu_log.h"
#include "hccl_common.h"
#include "ccu_common.h"

#include "ccu_dev_mgr_imp.h"
#include "../ccu_device/ccu_res_specs.h"

#include "rt_external.h"

namespace hcomm {

CcuVarEventResMgr& CcuVarEventResMgr::GetInstance(const int32_t userDevId)
{
    static CcuVarEventResMgr resMgrs[MAX_MODULE_DEVICE_NUM + 1];

    int32_t validUserDevId = userDevId;
    if (validUserDevId < 0 || static_cast<uint32_t>(validUserDevId) >= MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING(
            "[CcuVarEventResMgr][%s] use the backup device, userDevId[%d] should be "
            "less than %u.",
            __func__, validUserDevId, MAX_MODULE_DEVICE_NUM);
        validUserDevId = MAX_MODULE_DEVICE_NUM;
    }

    resMgrs[validUserDevId].userDevId_ = validUserDevId;
    return resMgrs[validUserDevId];
}

CcuResult CcuVarEventResMgr::AllocFromPool(std::vector<ResInfo>& pool, uint32_t num, std::vector<ResInfo>& out)
{
    for (auto it = pool.begin(); it != pool.end(); ++it) {
        if (it->num < num) {
            continue;
        }

        out.clear();
        out.emplace_back(it->startId, num);

        if (it->num == num) {
            pool.erase(it);
        } else {
            it->startId += num;
            it->num -= num;
        }
        return CCU_SUCCESS;
    }

    return CCU_E_UNAVAIL;
}

void CcuVarEventResMgr::ReturnToPool(std::vector<ResInfo>& pool, const std::vector<ResInfo>& res)
{
    for (const auto& info : res) {
        if (info.num == 0) {
            continue;
        }
        pool.emplace_back(info.startId, info.num);
    }
}

static std::vector<ResInfo>* SelectPool(CcuResRepository& resRepo, CcuVarEventType type, uint8_t dieId)
{
    switch (type) {
        case CcuVarEventType::VARIABLE:
            return &resRepo.blockXn[dieId];
        case CcuVarEventType::EVENT:
            return &resRepo.blockCke[dieId];
        default:
            return nullptr;
    }
}

// 由预约资源类型推导runtime资源类型，不支持的类型返回false
static bool GetRtResType(CcuVarEventType type, rtDevResType_t& resType)
{
    switch (type) {
        case CcuVarEventType::VARIABLE:
            resType = RT_RES_TYPE_CCU_XN;
            return true;
        case CcuVarEventType::EVENT:
            resType = RT_RES_TYPE_CCU_CKE;
            return true;
        default:
            return false;
    }
}

static CcuResult MapDevResAddress(uint8_t dieId, rtDevResType_t resType, uint32_t resId, uint64_t& va)
{
    rtDevResInfo resInfo{};
    resInfo.dieId = dieId;
    resInfo.procType = RT_PROCESS_CP1;
    resInfo.resType = resType;
    resInfo.resId = resId;
    resInfo.flag = 0;

    uint64_t mappedAddr = 0;
    uint32_t mappedLen = 0;
    rtDevResAddrInfo addrInfo{};
    addrInfo.resAddress = &mappedAddr;
    addrInfo.len = &mappedLen;

    rtError_t ret = rtGetDevResAddress(&resInfo, &addrInfo);
    if (ret != RT_ERROR_NONE) {
        HCCL_ERROR(
            "[CcuVarEventResMgr][%s] rtGetDevResAddress failed[%d], dieId[%u] resType[%d] "
            "resId[%u].",
            __func__, ret, dieId, static_cast<int32_t>(resType), resId);
        return CCU_E_RUNTIME;
    }

    va = mappedAddr;
    return CCU_SUCCESS;
}

// 解除MapDevResAddress映射的进程可访问VA，与映射一一对应
static CcuResult UnmapDevResAddress(uint8_t dieId, rtDevResType_t resType, uint32_t resId)
{
    rtDevResInfo resInfo{};
    resInfo.dieId = dieId;
    resInfo.procType = RT_PROCESS_CP1;
    resInfo.resType = resType;
    resInfo.resId = resId;
    resInfo.flag = 0;

    rtError_t ret = rtReleaseDevResAddress(&resInfo);
    if (ret != RT_ERROR_NONE) {
        HCCL_ERROR(
            "[CcuVarEventResMgr][%s] rtReleaseDevResAddress failed[%d], dieId[%u] "
            "resType[%d] resId[%u].",
            __func__, ret, dieId, static_cast<int32_t>(resType), resId);
        return CCU_E_RUNTIME;
    }

    return CCU_SUCCESS;
}

namespace {
    // RegisterAddrs 中记录本次已成功映射的资源，供失败回滚逆序解除
    struct MappedRes {
        uint8_t dieId;
        uint32_t resId;
    };
} // namespace

// 逆序解除本次已完成的映射，与 MapDevResAddress 一一对应
static void
UnmapMappedRes(const std::vector<MappedRes>& mapped, rtDevResType_t resType, uint64_t handle, CcuVarEventType type)
{
    HCCL_RUN_WARNING(
        "[CcuVarEventResMgr][%s] rollback, unmap [%zu] mapped res of "
        "handle[0x%llx] type[%d].",
        __func__, mapped.size(), handle, static_cast<int32_t>(type));
    for (auto it = mapped.rbegin(); it != mapped.rend(); ++it) {
        (void)UnmapDevResAddress(it->dieId, resType, it->resId);
    }
}

CcuResult CcuVarEventResMgr::RegisterAddrs(CcuVarEventType type, uint64_t handle, uint32_t num)
{
    rtDevResType_t resType = RT_RES_TYPE_CCU_XN;
    if (!GetRtResType(type, resType)) {
        HCCL_ERROR(
            "[CcuVarEventResMgr][%s] failed, unsupported type[%d], handle[0x%llx].", __func__,
            static_cast<int32_t>(type), handle);
        return CCU_E_PARA;
    }

    std::vector<uint64_t> vaList{};
    std::vector<MappedRes> mapped{};
    vaList.reserve(num);
    mapped.reserve(num);

    // 归一本函数内三处失败回滚路径：均为“逆序解除已完成映射”，收敛成一个闭包，
    // 避免 UnmapMappedRes 调用点重复三份；(void) 消歧义式丢弃返回值，规避静态检查误报
    auto rollback = [&mapped, resType, handle, type]() {
        (void)UnmapMappedRes(mapped, resType, handle, type);
    };

    for (uint32_t index = 0; index < num; index++) {
        uint8_t dieId = 0;
        uint32_t resId = 0;
        CcuResult idRet = (type == CcuVarEventType::VARIABLE) ? GetVariableXnId(handle, index, dieId, resId) :
                                                                GetEventCkeId(handle, index, dieId, resId);
        if (idRet != CCU_SUCCESS) {
            rollback();
            return idRet;
        }

        uint64_t va = 0;
        CcuResult mapRet = MapDevResAddress(dieId, resType, resId, va);
        if (mapRet != CCU_SUCCESS) {
            rollback();
            return mapRet;
        }
        vaList.push_back(va);
        mapped.push_back({dieId, resId});
    }

    CcuResult saveRet = SaveAddrs(type, handle, vaList);
    if (saveRet != CCU_SUCCESS) {
        rollback();
        return saveRet;
    }
    return CCU_SUCCESS;
}

CcuResult CcuVarEventResMgr::AllocAndRecord(
    CcuInsHandle insHandle, CcuResRepository& resRepo, CcuVarEventType type, uint8_t dieId, uint32_t num,
    uint64_t& newHandle)
{
    if (dieId >= CCU_MAX_IODIE_NUM) {
        HCCL_ERROR(
            "[CcuVarEventResMgr][%s] failed, dieId[%u] should be less than %u.", __func__, dieId, CCU_MAX_IODIE_NUM);
        return CCU_E_PARA;
    }
    if (num == 0) {
        HCCL_ERROR("[CcuVarEventResMgr][%s] failed, num should not be 0.", __func__);
        return CCU_E_PARA;
    }

    std::vector<ResInfo>* pool = SelectPool(resRepo, type, dieId);
    if (pool == nullptr) {
        HCCL_ERROR("[CcuVarEventResMgr][%s] failed, unsupported type[%d].", __func__, static_cast<int32_t>(type));
        return CCU_E_PARA;
    }

    std::vector<ResInfo> resInfos{};
    CcuResult ret = AllocFromPool(*pool, num, resInfos);
    if (ret != CCU_SUCCESS) {
        HCCL_ERROR(
            "[CcuVarEventResMgr][%s] failed, no consecutive block of num[%u] in insHandle[0x%llx] "
            "resource pool, dieId[%u] type[%d].",
            __func__, num, insHandle, dieId, static_cast<int32_t>(type));
        return ret;
    }

    CcuVarEventRes res{};
    res.insHandle = insHandle;
    res.userDevId = userDevId_;
    res.dieId = dieId;
    res.type = type;
    res.resInfos = std::move(resInfos);
    res.resRepo = &resRepo;

    std::unique_lock<std::shared_timed_mutex> lock(mapMutex_);
    handleSeed_ += 1;
    newHandle = handleSeed_;
    resMap_.emplace(newHandle, std::move(res));
    return CCU_SUCCESS;
}

CcuResult CcuVarEventResMgr::Acquire(
    CcuInsHandle insHandle, CcuResRepository& resRepo, CcuVarEventType type, uint8_t dieId, uint32_t num,
    uint64_t& handle)
{
    uint64_t newHandle = 0;
    CcuResult allocRet = AllocAndRecord(insHandle, resRepo, type, dieId, num, newHandle);
    if (allocRet != CCU_SUCCESS) {
        return allocRet;
    }

    // 申请期即完成地址映射；失败与切池动作配对，整笔撤销后不写出参
    CcuResult regRet = RegisterAddrs(type, newHandle, num);
    if (regRet != CCU_SUCCESS) {
        HCCL_RUN_WARNING(
            "[CcuVarEventResMgr][%s] register addrs failed[%d], release acquired "
            "handle[0x%llx] insHandle[0x%llx] dieId[%u] type[%d] num[%u].",
            __func__, static_cast<int32_t>(regRet), newHandle, insHandle, dieId, static_cast<int32_t>(type), num);
        (void)ReleaseByHandle(newHandle);
        return regRet;
    }

    handle = newHandle;
    HCCL_RUN_INFO(
        "[CcuVarEventResMgr][%s] success, userDevId[%d] insHandle[0x%llx] dieId[%u] "
        "type[%d] num[%u] handle[0x%llx].",
        __func__, userDevId_, insHandle, dieId, static_cast<int32_t>(type), num, handle);
    return CCU_SUCCESS;
}

CcuResult CcuVarEventResMgr::GetVariableXnId(uint64_t handle, uint32_t index, uint8_t& dieId, uint32_t& xnId) const
{
    std::shared_lock<std::shared_timed_mutex> lock(mapMutex_);
    auto it = resMap_.find(handle);
    if (it == resMap_.end()) {
        HCCL_ERROR("[CcuVarEventResMgr][%s] handle[0x%llx] is not existed.", __func__, handle);
        return CCU_E_NOT_FOUND;
    }

    const auto& res = it->second;
    if (res.type != CcuVarEventType::VARIABLE) {
        HCCL_ERROR("[CcuVarEventResMgr][%s] failed, handle[0x%llx] is not a variable(xn) resource.", __func__, handle);
        return CCU_E_PARA;
    }
    if (res.resInfos.size() != 1) {
        HCCL_ERROR(
            "[CcuVarEventResMgr][%s] get variable resource id failed, variable resource is fragmented into %zu blocks.",
            __func__, res.resInfos.size());
        return CCU_E_NOT_SUPPORT;
    }

    const ResInfo& info = res.resInfos[0];
    if (index >= info.num) {
        HCCL_ERROR(
            "[CcuVarEventResMgr][%s] get variable resource id failed, index[%u] out of range, block num[%u].", __func__,
            index, info.num);
        return CCU_E_PARA;
    }

    dieId = res.dieId;
    xnId = info.startId + index;
    return CCU_SUCCESS;
}

CcuResult CcuVarEventResMgr::GetEventCkeId(uint64_t handle, uint32_t index, uint8_t& dieId, uint32_t& ckeId) const
{
    std::shared_lock<std::shared_timed_mutex> lock(mapMutex_);
    auto it = resMap_.find(handle);
    if (it == resMap_.end()) {
        HCCL_ERROR("[CcuVarEventResMgr][%s] handle[0x%llx] is not existed.", __func__, handle);
        return CCU_E_NOT_FOUND;
    }

    const auto& res = it->second;
    if (res.type != CcuVarEventType::EVENT) {
        HCCL_ERROR(
            "[CcuVarEventResMgr][%s] get event resource id failed, handle[0x%llx] is not an event(cke) resource.",
            __func__, handle);
        return CCU_E_PARA;
    }
    if (res.resInfos.size() != 1) {
        HCCL_ERROR(
            "[CcuVarEventResMgr][%s] get event resource id failed, event resource is fragmented into %zu blocks.",
            __func__, res.resInfos.size());
        return CCU_E_NOT_SUPPORT;
    }

    const ResInfo& info = res.resInfos[0];
    if (index >= info.num) {
        HCCL_ERROR("[CcuVarEventResMgr][%s] failed, index[%u] out of range, block num[%u].", __func__, index, info.num);
        return CCU_E_PARA;
    }

    dieId = res.dieId;
    ckeId = info.startId + index;
    return CCU_SUCCESS;
}

CcuResult CcuVarEventResMgr::SaveAddrs(CcuVarEventType type, uint64_t handle, const std::vector<uint64_t>& vaList)
{
    std::unique_lock<std::shared_timed_mutex> lock(mapMutex_);
    auto it = resMap_.find(handle);
    if (it == resMap_.end()) {
        HCCL_ERROR("[CcuVarEventResMgr][%s] handle[0x%llx] is not existed.", __func__, handle);
        return CCU_E_NOT_FOUND;
    }

    auto& res = it->second;
    if (res.type != type) {
        HCCL_ERROR(
            "[CcuVarEventResMgr][%s] failed, handle[0x%llx] type mismatch, expect[%d] "
            "actual[%d].",
            __func__, handle, static_cast<int32_t>(type), static_cast<int32_t>(res.type));
        return CCU_E_PARA;
    }
    if (res.resInfos.size() != 1) {
        HCCL_ERROR(
            "[CcuVarEventResMgr][%s] failed, resource is fragmented into %zu blocks.", __func__, res.resInfos.size());
        return CCU_E_NOT_SUPPORT;
    }
    if (vaList.size() != res.resInfos[0].num) {
        HCCL_ERROR(
            "[CcuVarEventResMgr][%s] failed, va count[%zu] mismatch resource num[%u].", __func__, vaList.size(),
            res.resInfos[0].num);
        return CCU_E_PARA;
    }

    res.vaList = vaList;
    return CCU_SUCCESS;
}

CcuResult CcuVarEventResMgr::GetSavedAddr(CcuVarEventType type, uint64_t handle, uint32_t index, uint64_t& va) const
{
    std::shared_lock<std::shared_timed_mutex> lock(mapMutex_);
    auto it = resMap_.find(handle);
    if (it == resMap_.end()) {
        HCCL_ERROR("[CcuVarEventResMgr][%s] handle[0x%llx] is not existed.", __func__, handle);
        return CCU_E_NOT_FOUND;
    }

    const auto& res = it->second;
    if (res.type != type) {
        HCCL_ERROR(
            "[CcuVarEventResMgr][%s] failed, handle[0x%llx] type mismatch, expect[%d] "
            "actual[%d].",
            __func__, handle, static_cast<int32_t>(type), static_cast<int32_t>(res.type));
        return CCU_E_PARA;
    }
    if (index >= res.vaList.size()) {
        HCCL_ERROR(
            "[CcuVarEventResMgr][%s] failed, index[%u] out of range, registered num[%zu].", __func__, index,
            res.vaList.size());
        return CCU_E_PARA;
    }

    va = res.vaList[index];
    return CCU_SUCCESS;
}

CcuResult CcuVarEventResMgr::UnmapSavedAddrs(const CcuVarEventRes& res)
{
    // 仅 unmap Alloc 阶段已成功映射并保存 VA 的资源；错误回滚路径中 vaList 为空，天然跳过。
    // 不变量：vaList 非空 <=> SaveAddrs 已成功，而 SaveAddrs 强校验 resInfos 为单个连续块且
    // vaList.size() == resInfos[0].num，故下面用 resInfos[0].startId + index 反推 resId 恒成立
    if (res.vaList.empty() || res.resInfos.empty()) {
        return CCU_SUCCESS;
    }
    // 上述不变量当前由 SaveAddrs 保证，此处再作一次防御校验：一旦将来分配策略改为可返回多块，
    // 用首块 startId 反推 resId 会越出块边界，宁可跳过 unmap 也不能解除错误资源的映射
    if (res.resInfos.size() != 1) {
        HCCL_ERROR(
            "[CcuVarEventResMgr][%s] unexpected fragmented resInfos size[%zu], skip unmap.", __func__,
            res.resInfos.size());
        return CCU_E_NOT_SUPPORT;
    }

    rtDevResType_t resType = RT_RES_TYPE_CCU_XN;
    if (!GetRtResType(res.type, resType)) {
        HCCL_ERROR("[CcuVarEventResMgr][%s] failed, unsupported type[%d].", __func__, static_cast<int32_t>(res.type));
        return CCU_E_PARA;
    }

    CcuResult firstErr = CCU_SUCCESS;
    const uint32_t startId = res.resInfos[0].startId;
    for (uint32_t index = 0; index < res.vaList.size(); index++) {
        CcuResult ret = UnmapDevResAddress(res.dieId, resType, startId + index);
        if (ret != CCU_SUCCESS && firstErr == CCU_SUCCESS) {
            firstErr = ret;
        }
    }
    return firstErr;
}

CcuResult CcuVarEventResMgr::ReleaseByHandle(uint64_t handle)
{
    CcuVarEventRes res{};
    {
        std::unique_lock<std::shared_timed_mutex> lock(mapMutex_);
        auto it = resMap_.find(handle);
        if (it == resMap_.end()) {
            HCCL_ERROR("[CcuVarEventResMgr][%s] handle[0x%llx] is not existed.", __func__, handle);
            return CCU_E_NOT_FOUND;
        }
        res = std::move(it->second);
        resMap_.erase(it);
    }

    // 归还资源池前，先解除该 handle 已映射的进程可访问 VA；
    // unmap 失败不阻断归还，错误码上抛由调用方决定是否处理
    CcuResult unmapRet = UnmapSavedAddrs(res);
    if (unmapRet != CCU_SUCCESS) {
        HCCL_RUN_WARNING(
            "[CcuVarEventResMgr][%s] unmap failed[%d], continue to return resources, "
            "handle[0x%llx].",
            __func__, static_cast<int32_t>(unmapRet), handle);
    }

    if (res.resRepo == nullptr) {
        return unmapRet;
    }
    std::vector<ResInfo>* pool = SelectPool(*res.resRepo, res.type, res.dieId);
    if (pool == nullptr) {
        HCCL_ERROR(
            "[CcuVarEventResMgr][%s] failed to return, handle[0x%llx] type[%d].", __func__, handle,
            static_cast<int32_t>(res.type));
        return CCU_E_INTERNAL;
    }
    ReturnToPool(*pool, res.resInfos);
    return unmapRet;
}

CcuResult CcuVarEventResMgr::ReleaseByInstance(CcuInsHandle insHandle)
{
    std::vector<CcuVarEventRes> toRelease{};
    {
        std::unique_lock<std::shared_timed_mutex> lock(mapMutex_);
        for (auto it = resMap_.begin(); it != resMap_.end();) {
            if (it->second.insHandle == insHandle) {
                toRelease.push_back(std::move(it->second));
                it = resMap_.erase(it);
            } else {
                ++it;
            }
        }
    }

    // 通信域正在销毁，单条失败不中断，记录首个错误码后继续清干净其余记录
    CcuResult firstErr = CCU_SUCCESS;
    for (auto& res : toRelease) {
        // ccu_instance 析构释放资源前，先解除 Alloc 阶段映射的进程可访问 VA
        CcuResult unmapRet = UnmapSavedAddrs(res);
        if (unmapRet != CCU_SUCCESS && firstErr == CCU_SUCCESS) {
            firstErr = unmapRet;
        }

        if (res.resRepo == nullptr) {
            continue;
        }
        std::vector<ResInfo>* pool = SelectPool(*res.resRepo, res.type, res.dieId);
        if (pool == nullptr) {
            HCCL_ERROR(
                "[CcuVarEventResMgr][%s] failed to return, insHandle[0x%llx] type[%d].", __func__, insHandle,
                static_cast<int32_t>(res.type));
            if (firstErr == CCU_SUCCESS) {
                firstErr = CCU_E_INTERNAL;
            }
            continue;
        }
        ReturnToPool(*pool, res.resInfos);
    }
    return firstErr;
}

// 从空闲块列表 pool 中扣除区间 [start, start+num)必要时把命中的空闲块拆分成左右两段。
static void RemoveRangeFromPool(std::vector<ResInfo>& pool, uint32_t start, uint32_t num)
{
    if (num == 0) {
        return;
    }
    const uint32_t end = start + num;
    std::vector<ResInfo> result{};
    result.reserve(pool.size() + 1);
    for (const auto& block : pool) {
        const uint32_t blockStart = block.startId;
        const uint32_t blockEnd = block.startId + block.num;
        if (end <= blockStart || start >= blockEnd) {
            result.push_back(block);
            continue;
        }
        if (blockStart < start) {
            result.emplace_back(blockStart, start - blockStart);
        }
        if (end < blockEnd) {
            result.emplace_back(end, blockEnd - end);
        }
    }
    pool.swap(result);
}

CcuResult CcuVarEventResMgr::ExcludeAllocatedFromRepo(CcuInsHandle insHandle) const
{
    // shared_lock 用于只读遍历 resMap_；被修改的 *pool 属于该 insHandle 自己的 CcuResPack，
    // 其并发安全由“同一 instance 单线程串行访问”契约保证，详见头文件线程安全契约说明
    std::shared_lock<std::shared_timed_mutex> lock(mapMutex_);
    for (const auto& kv : resMap_) {
        const CcuVarEventRes& res = kv.second;
        if (res.insHandle != insHandle || res.resRepo == nullptr) {
            continue;
        }
        std::vector<ResInfo>* pool = SelectPool(*res.resRepo, res.type, res.dieId);
        if (pool == nullptr) {
            HCCL_ERROR(
                "[CcuVarEventResMgr][%s] failed, insHandle[0x%llx] type[%d].", __func__, insHandle,
                static_cast<int32_t>(res.type));
            continue;
        }
        for (const auto& info : res.resInfos) {
            RemoveRangeFromPool(*pool, info.startId, info.num);
            HCCL_INFO(
                "[CcuVarEventResMgr][%s] exclude acquired res, insHandle[0x%llx] type[%d] "
                "dieId[%u] startId[%u] num[%u].",
                __func__, insHandle, static_cast<int32_t>(res.type), res.dieId, info.startId, info.num);
        }
    }
    return CCU_SUCCESS;
}

} // namespace hcomm

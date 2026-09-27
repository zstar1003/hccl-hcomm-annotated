/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_instance_mgr.h"

#include "ccu_log.h"
#include "hccl_common.h"

namespace hcomm {

CcuInstanceMgr::~CcuInstanceMgr()
{
    if (!initializedFlag_) {
        return;
    }

    (void)Deinit();
}

CcuInstanceMgr& CcuInstanceMgr::GetInstance(const int32_t userDevId)
{
    static CcuInstanceMgr instanceMgrs[MAX_MODULE_DEVICE_NUM + 1];

    int32_t validUserDevId = userDevId;
    if (validUserDevId < 0 || static_cast<uint32_t>(validUserDevId) >= MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING(
            "[CcuInstanceMgr][%s] use the backup device, userDevId[%d] should be "
            "less than %u.",
            __func__, validUserDevId, MAX_MODULE_DEVICE_NUM);
        validUserDevId = MAX_MODULE_DEVICE_NUM; // 使用备份设备
    }

    instanceMgrs[validUserDevId].userDevId_ = validUserDevId;
    return instanceMgrs[validUserDevId];
}

CcuResult CcuInstanceMgr::Init()
{
    std::unique_lock<std::shared_timed_mutex> lock(insMapMutex_);
    if (initializedFlag_) {
        return CcuResult::CCU_SUCCESS;
    }

    initializedFlag_ = true;
    insMap_.clear();
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuInstanceMgr::Deinit()
{
    std::unique_lock<std::shared_timed_mutex> lock(insMapMutex_);
    insMap_.clear();
    (void)resDescMgr_.Deinit();
    initializedFlag_ = false;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuInstanceMgr::CreateByInsType(const CcuInstanceType insType, CcuInsHandle& insHandle)
{
    std::unique_lock<std::shared_timed_mutex> lock(insMapMutex_);

    std::unique_ptr<CcuInstance> instance{nullptr};
    EXCEPTION_CATCH(instance = std::make_unique<CcuInstance>(), return CcuResult::CCU_E_INTERNAL);

    CCU_CHK_RET(instance->InitByInsType(insType));

    instanceId_ += 1;
    instance->SetHandle(instanceId_);
    EXCEPTION_CATCH(insMap_.emplace(instanceId_, std::move(instance)), return CcuResult::CCU_E_INTERNAL);
    insHandle = instanceId_;
    return CcuResult::CCU_SUCCESS;
}

CcuInstance* CcuInstanceMgr::Get(CcuInsHandle insHandle) const
{
    std::shared_lock<std::shared_timed_mutex> lock(insMapMutex_);
    auto it = insMap_.find(insHandle);
    if (it == insMap_.end()) {
        HCCL_ERROR("[CcuInstanceMgr][%s] handle[%llx] is not existed.", __func__, insHandle);
        return nullptr;
    }

    return it->second.get();
}

CcuResult CcuInstanceMgr::Destroy(CcuInsHandle insHandle)
{
    std::unique_lock<std::shared_timed_mutex> lock(insMapMutex_);
    auto it = insMap_.find(insHandle);
    if (it == insMap_.end()) {
        HCCL_ERROR("[CcuInstanceMgr][%s] handle[%llx] is not existed.", __func__, insHandle);
        return CcuResult::CCU_E_NOT_FOUND;
    }

    // 修改 cascCntMap_ 需持写锁; 边遍历边 erase 会使迭代器失效, 用 erase 返回值推进避免 coredump
    std::unique_lock<std::shared_timed_mutex> cascCntLock(cascCntMapMutex_);
    for (auto it = cascCntMap_.begin(); it != cascCntMap_.end();) {
        if (it->second == insHandle) {
            it = cascCntMap_.erase(it);
        } else {
            ++it;
        }
    }

    insMap_.erase(it);
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuInstanceMgr::CreateByResDescs(const CcuResDesc* descs[], uint32_t descNum, CcuInsHandle& insHandle)
{
    std::unique_lock<std::shared_timed_mutex> lock(insMapMutex_);

    std::unique_ptr<CcuInstance> instance{nullptr};
    EXCEPTION_CATCH(instance = std::make_unique<CcuInstance>(), return CcuResult::CCU_E_INTERNAL);

    CCU_CHK_RET(instance->InitByResDescs(descs, descNum));

    instanceId_ += 1;
    instance->SetHandle(instanceId_);
    EXCEPTION_CATCH(insMap_.emplace(instanceId_, std::move(instance)), return CcuResult::CCU_E_INTERNAL);
    insHandle = instanceId_;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuInstanceMgr::CreateByAllRes(CcuInsHandle& insHandle)
{
    std::unique_lock<std::shared_timed_mutex> lock(insMapMutex_);

    std::unique_ptr<CcuInstance> instance{nullptr};
    EXCEPTION_CATCH(instance = std::make_unique<CcuInstance>(), return CcuResult::CCU_E_INTERNAL);

    CCU_CHK_RET(instance->InitByAllRes());

    instanceId_ += 1;
    instance->SetHandle(instanceId_);
    EXCEPTION_CATCH(insMap_.emplace(instanceId_, std::move(instance)), return CcuResult::CCU_E_INTERNAL);
    insHandle = instanceId_;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuInstanceMgr::QueryInsResDesc(CcuInsHandle& ccuInsHandle, uint8_t dieId, HcommCcuResDescHandle& resDesc)
{
    std::shared_lock<std::shared_timed_mutex> lock(insMapMutex_);

    auto it = insMap_.find(ccuInsHandle);
    if (it == insMap_.end()) {
        HCCL_ERROR("[CcuInstanceMgr][%s] handle[%llx] is not existed.", __func__, ccuInsHandle);
        return CcuResult::CCU_E_NOT_FOUND;
    }

    // 从 ccuIns 持有的 totalResDescs_ 取该 die 的资源描述符，逐项写入入参 resDesc
    const auto& totalDesc = it->second->GetTotalResDescs(dieId);
    uint32_t num = 0;
    CCU_CHK_RET(totalDesc.QueryResNum(ResType::LOOP, num));
    CCU_CHK_RET(resDescMgr_.SetResNum(resDesc, ResType::LOOP, num));
    CCU_CHK_RET(totalDesc.QueryResNum(ResType::MS, num));
    CCU_CHK_RET(resDescMgr_.SetResNum(resDesc, ResType::MS, num));
    CCU_CHK_RET(totalDesc.QueryResNum(ResType::CKE, num));
    CCU_CHK_RET(resDescMgr_.SetResNum(resDesc, ResType::CKE, num));
    CCU_CHK_RET(totalDesc.QueryResNum(ResType::XN, num));
    CCU_CHK_RET(resDescMgr_.SetResNum(resDesc, ResType::XN, num));
    CCU_CHK_RET(totalDesc.QueryResNum(ResType::GSA, num));
    CCU_CHK_RET(resDescMgr_.SetResNum(resDesc, ResType::GSA, num));
    CCU_CHK_RET(totalDesc.QueryResNum(ResType::MISSION, num));
    CCU_CHK_RET(resDescMgr_.SetResNum(resDesc, ResType::MISSION, num));
    CCU_CHK_RET(totalDesc.QueryResNum(ResType::INS, num));
    CCU_CHK_RET(resDescMgr_.SetResNum(resDesc, ResType::INS, num));

    return CcuResult::CCU_SUCCESS;
}

CcuResDescMgr& CcuInstanceMgr::GetResDescMgr() { return resDescMgr_; }

CcuResult CcuInstanceMgr::CascCntHandleAlloc(CcuInsHandle ccuInsHandle, uint8_t dieId, HcommCcuCascCntHandle& handle)
{
    std::shared_lock<std::shared_timed_mutex> lock(insMapMutex_);

    auto it = insMap_.find(ccuInsHandle);
    if (it == insMap_.end()) {
        HCCL_ERROR("[CcuInstanceMgr][%s] ins handle[%llx] is not existed.", __func__, ccuInsHandle);
        return CcuResult::CCU_E_NOT_FOUND;
    }

    // cascCnt 句柄由 Mgr 统一分配(per-device 全局唯一, 单调递增从 1 起), 从根本上消除各实例
    // resPack 独立自增导致的跨实例句柄冲突 -> cascCntMap_ 静默错路由(致命)。
    HcommCcuCascCntHandle allocHandle = nextCascCntHandle_.fetch_add(1) + 1;

    // 先登记 handle->ins 路由(写操作必须持写锁; 原实现误用 shared_lock 存在并发写竞态), 再向下
    // 申请块。全局唯一句柄理论上不会冲突, 仍防御性检查插入结果, 一旦冲突即显式失败而非静默错路由。
    {
        std::unique_lock<std::shared_timed_mutex> cascCntLock(cascCntMapMutex_);
        auto emplaceRet = cascCntMap_.emplace(allocHandle, ccuInsHandle);
        if (!emplaceRet.second) {
            HCCL_ERROR(
                "[CcuInstanceMgr][%s] failed, cascCnt handle[%llx] already routed to ins[%llx].", __func__, allocHandle,
                emplaceRet.first->second);
            return CcuResult::CCU_E_INTERNAL;
        }
    }

    // 向下申请块失败(如无空闲块)时回滚路由登记, 避免 cascCntMap_ 残留指向未成功分配的悬挂项。
    CcuResult ret = it->second->CascCntHandleAlloc(dieId, allocHandle);
    if (ret != CcuResult::CCU_SUCCESS) {
        std::unique_lock<std::shared_timed_mutex> cascCntLock(cascCntMapMutex_);
        cascCntMap_.erase(allocHandle);
        return ret;
    }

    handle = allocHandle;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuInstanceMgr::GetCascCntBlock(HcommCcuCascCntHandle cntHandle, CntXnBlock& cascCntBlock)
{
    // 锁序与 Destroy/CascCntHandleAlloc 全仓统一(先 insMapMutex_ 后 cascCntMapMutex_), 避免 AB-BA 死锁。
    // insMap_ 的 find 与解引用调用必须持 insMapMutex_ 读锁: 否则与 CreateByResDescs 的 emplace(可能 rehash
    // 释放桶数组)及 Destroy 的 erase 并发时为数据竞争(UB); 持读锁亦保证解引用期间实例不被 Destroy 析构。
    std::shared_lock<std::shared_timed_mutex> insLock(insMapMutex_);
    std::shared_lock<std::shared_timed_mutex> cascCntLock(cascCntMapMutex_);

    auto cascCntIter = cascCntMap_.find(cntHandle);
    if (cascCntIter == cascCntMap_.end()) {
        HCCL_ERROR("[CcuInstanceMgr][%s] cascCnt handle[%llx] is not existed.", __func__, cntHandle);
        return CcuResult::CCU_E_NOT_FOUND;
    }

    auto insIter = insMap_.find(cascCntIter->second);
    if (insIter == insMap_.end()) {
        HCCL_ERROR("[CcuInstanceMgr][%s] ins handle[%llx] is not existed.", __func__, cascCntIter->second);
        return CcuResult::CCU_E_NOT_FOUND;
    }

    CCU_CHK_RET(insIter->second->GetCascCntBlock(cntHandle, cascCntBlock));
    return CcuResult::CCU_SUCCESS;
}

} // namespace hcomm

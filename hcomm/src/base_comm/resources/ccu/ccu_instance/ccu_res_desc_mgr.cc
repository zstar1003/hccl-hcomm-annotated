/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_res_desc_mgr.h"

#include <algorithm>
#include <mutex>
#include <vector>

#include "ccu_dev_mgr_imp.h"
#include "ccu_log.h"
#include "exception_handler.h"

namespace hcomm {

CcuResult CcuResDescMgr::Create(uint32_t dieId, HcommCcuResDescHandle& handle)
{
    std::unique_lock<std::shared_timed_mutex> lock(descMapMutex_);

    std::unique_ptr<CcuResDesc> desc{nullptr};
    EXCEPTION_CATCH(desc = std::make_unique<CcuResDesc>(), return CcuResult::CCU_E_INTERNAL);

    desc->dieId = dieId;
    nextHandle_ += 1;
    EXCEPTION_CATCH(descMap_.emplace(nextHandle_, std::move(desc)), return CcuResult::CCU_E_INTERNAL);
    handle = nextHandle_;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuResDescMgr::FindDesc(HcommCcuResDescHandle handle, const char* funcName, ConstDescIterator& it) const
{
    it = descMap_.find(handle);
    if (it == descMap_.cend()) {
        HCCL_ERROR("[CcuResDescMgr][%s] handle[%llx] is not existed.", funcName, handle);
        return CcuResult::CCU_E_NOT_FOUND;
    }

    return CcuResult::CCU_SUCCESS;
}

const CcuResDesc* CcuResDescMgr::Get(HcommCcuResDescHandle handle) const
{
    std::shared_lock<std::shared_timed_mutex> lock(descMapMutex_);
    auto it = descMap_.cend();
    if (FindDesc(handle, __func__, it) != CcuResult::CCU_SUCCESS) {
        return nullptr;
    }

    return it->second.get();
}

CcuResult CcuResDescMgr::Destroy(HcommCcuResDescHandle handle)
{
    std::unique_lock<std::shared_timed_mutex> lock(descMapMutex_);
    auto it = descMap_.cend();
    CCU_CHK_RET(FindDesc(handle, __func__, it));

    descMap_.erase(it);
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuResDescMgr::SetResNum(HcommCcuResDescHandle handle, ResType resType, uint32_t resNum)
{
    std::unique_lock<std::shared_timed_mutex> lock(descMapMutex_);
    auto it = descMap_.cend();
    CCU_CHK_RET(FindDesc(handle, __func__, it));

    return it->second->SetResNum(resType, resNum);
}

CcuResult CcuResDescMgr::QueryResNum(HcommCcuResDescHandle handle, ResType resType, uint32_t& resNum) const
{
    std::shared_lock<std::shared_timed_mutex> lock(descMapMutex_);
    auto it = descMap_.cend();
    CCU_CHK_RET(FindDesc(handle, __func__, it));

    return it->second->QueryResNum(resType, resNum);
}

CcuResult CcuResDescMgr::QueryDieId(HcommCcuResDescHandle handle, uint32_t& dieId) const
{
    std::shared_lock<std::shared_timed_mutex> lock(descMapMutex_);
    auto it = descMap_.cend();
    CCU_CHK_RET(FindDesc(handle, __func__, it));

    dieId = it->second->dieId;
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuResDescMgr::QueryRemainRes(HcommCcuResDescHandle handle, int32_t userDevId) const
{
    std::unique_lock<std::shared_timed_mutex> lock(descMapMutex_);
    auto it = descMap_.cend();
    CCU_CHK_RET(FindDesc(handle, __func__, it));

    CcuResDesc& desc = *it->second;
    const uint8_t dieId = static_cast<uint8_t>(desc.dieId);

    // 遍历全部 ResType (LOOP..MISSION, 不含 INS)
    constexpr ResType kResTypes[]
        = {ResType::LOOP, ResType::MS, ResType::CKE, ResType::XN, ResType::GSA, ResType::MISSION};

    uint32_t remainNum = 0;
    for (auto internalType : kResTypes) {
        // 查询最大连续剩余
        if (CcuDevMgrImp::QueryRemainRes(userDevId, dieId, internalType, remainNum) != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[CcuResDescMgr][%s] userDevId[%d] dieId[%u] resType[%s] query failed", __func__, userDevId, dieId,
                internalType.Describe().c_str());
            return CcuResult::CCU_E_INTERNAL;
        }
        HCCL_INFO(
            "[CcuResDescMgr][%s] userDevId[%d] dieId[%u] resType[%s] remainNum[%u]", __func__, userDevId, dieId,
            internalType.Describe().c_str(), remainNum);
        // 写入最大连续剩余
        CCU_CHK_RET(desc.SetResNum(internalType, remainNum));
    }

    // 单独处理 INSTRUCTION: 从 CcuComponent 查询实际剩余量
    uint32_t insFreeSize = CcuDevMgrImp::GetInsConsecutiveRemainSize(userDevId, dieId);
    HCCL_INFO(
        "[CcuResDescMgr][%s] userDevId[%d] dieId[%u] resType[ResType::INS] remainNum[%u]", __func__, userDevId, dieId,
        insFreeSize);
    CCU_CHK_RET(desc.SetResNum(ResType::INS, insFreeSize));
    // 单独处理 CASC_CNT: 从 CcuComponent 查询实际剩余量
    uint32_t cascCntFreeSize = CcuDevMgrImp::GetCascCntBlockRemainSize(userDevId, dieId);
    HCCL_INFO(
        "[CcuResDescMgr][%s] userDevId[%d] dieId[%u] resType[ResType::CASC_CNT] remainNum[%u]", __func__, userDevId,
        dieId, cascCntFreeSize);
    CCU_CHK_RET(desc.SetResNum(ResType::CASC_CNT, cascCntFreeSize));

    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuResDescMgr::Deinit()
{
    std::unique_lock<std::shared_timed_mutex> lock(descMapMutex_);
    descMap_.clear();
    nextHandle_ = 0;
    return CcuResult::CCU_SUCCESS;
}

} // namespace hcomm

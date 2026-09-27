/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "typical_mr_manager.h"
#include "adapter_rts_common.h"
#include "rdma_resource_manager.h"

namespace hccl {

TypicalMrManager& TypicalMrManager::GetInstance()
{
    static TypicalMrManager typicalMrManager[MAX_MODULE_DEVICE_NUM + 1];
    s32 deviceLogicId = INVALID_INT;
    HcclResult ret = hrtGetDevice(&deviceLogicId);
    if (ret == HCCL_SUCCESS && (static_cast<u32>(deviceLogicId) < MAX_MODULE_DEVICE_NUM)) {
        HCCL_INFO("[TypicalMrManager::GetInstance]deviceLogicID[%d]", deviceLogicId);
        return typicalMrManager[deviceLogicId];
    }
    HCCL_WARNING("[TypicalWindowMem::GetInstance]deviceLogicID[%d] is invalid, ret[%d]", deviceLogicId, ret);
    return typicalMrManager[MAX_MODULE_DEVICE_NUM];
}

TypicalMrManager::TypicalMrManager() : rdmaHandle_(nullptr) {}

TypicalMrManager::~TypicalMrManager()
{
    if (rdmaHandle_ != nullptr) {
        ReleaseMrResource();
    }
}

HcclResult TypicalMrManager::RegisterMem(struct MrInfoT& mrInfo)
{
    HCCL_DEBUG(
        "[TypicalMrManager][RegisterMem]MR register start, addr[%llu], MR size[%llu].", mrInfo.addr, mrInfo.size);
    CHK_RET(RdmaResourceManager::GetInstance().GetRdmaHandle(rdmaHandle_));
    CHK_PTR_NULL(rdmaHandle_);
    // Check whether the key is the default value or already registered in map
    std::unique_lock<std::mutex> lockMrMap(mrMapMutex_);
    if (mrInfo.lkey != DEFAULT_MR_KEY) {
        auto mrIter = regedMrMap_.find(mrInfo.lkey);
        if (mrIter != regedMrMap_.end()) {
            HCCL_WARNING(
                "[TypicalMrManager][RegisterMem]MR key[%lu] already registered, "
                "MR size[%llu], regedMrMap size[%u].",
                mrInfo.lkey, mrIter->second.first.size, regedMrMap_.size());
            return HCCL_E_PARA;
        }
        HCCL_ERROR(
            "[TypicalMrManager][RegisterMem]invalid MR info for register, addr[%llu], key[%lu].", mrInfo.addr,
            mrInfo.lkey);
        return HCCL_E_PARA;
    }
    MrHandle mrHandle = nullptr;
    // Register MR
    CHK_RET(hrtRaRegGlobalMr(rdmaHandle_, mrInfo, mrHandle));
    if (mrHandle == nullptr) {
        HCCL_WARNING(
            "[TypicalMrManager][RegisterMem]MR register not success, addr[%llu], MR size[%llu].", mrInfo.addr,
            mrInfo.size);
        return HCCL_E_INTERNAL;
    }
    regedMrMap_[mrInfo.lkey].first = mrInfo;
    regedMrMap_[mrInfo.lkey].second = mrHandle;
    HCCL_INFO(
        "[TypicalMrManager][RegisterMem]MR register success, "
        "MR key[%llu], addr[%llu], size[%llu], mrHandle[%p], regedMrMap size[%u].",
        mrInfo.lkey, mrInfo.addr, mrInfo.size, mrHandle, regedMrMap_.size());
    return HCCL_SUCCESS;
}

HcclResult TypicalMrManager::DeRegisterMem(struct MrInfoT& mrInfo)
{
    HCCL_DEBUG(
        "[TypicalMrManager][DeRegisterMem]MR deregister start, addr[%llu], size[%llu], key[%lu].", mrInfo.addr,
        mrInfo.size, mrInfo.lkey);
    CHK_RET(RdmaResourceManager::GetInstance().GetRdmaHandle(rdmaHandle_));
    CHK_PTR_NULL(rdmaHandle_);
    // Check whether the mem key exists. If exists，remove from MR map; else return error.
    std::unique_lock<std::mutex> lockMrMap(mrMapMutex_);
    auto mrIter = regedMrMap_.find(mrInfo.lkey);
    if (mrIter == regedMrMap_.end()) {
        HCCL_ERROR(
            "[TypicalMrManager][DeRegisterMem]no match MR info in MR map, "
            "MR key[%llu], addr[%llu], size[%llu], regedMrMap size[%u].",
            mrInfo.lkey, mrInfo.addr, mrInfo.size, regedMrMap_.size());
        return HCCL_E_PARA;
    }
    // Unregister MR
    MrHandle mrHandle = mrIter->second.second;
    HCCL_DEBUG("[TypicalMrManager][DeRegisterMem]rdma handle[%p], mr handle [%p].", rdmaHandle_, mrHandle);
    CHK_RET(hrtRaDeRegGlobalMr(rdmaHandle_, mrHandle));
    regedMrMap_.erase(mrIter);
    HCCL_INFO("[TypicalMrManager][DeRegisterMem]MR unregister success, regedMrMap size[%u].", regedMrMap_.size());
    mrInfo.lkey = DEFAULT_MR_KEY;
    return HCCL_SUCCESS;
}

HcclResult TypicalMrManager::ReleaseMrResource()
{
    CHK_PTR_NULL(rdmaHandle_);
    std::unique_lock<std::mutex> lockMrMap(mrMapMutex_);
    if (!regedMrMap_.empty()) {
        for (auto& mrIter : regedMrMap_) {
            MrHandle mrHandle = mrIter.second.second;
            if (mrHandle != nullptr) {
                CHK_RET(hrtRaDeRegGlobalMr(rdmaHandle_, mrHandle));
            }
        }
        regedMrMap_.clear();
    }
    HCCL_INFO("[TypicalMrManager][ReleaseMrResource]release mr resources success.");
    return HCCL_SUCCESS;
}
} // namespace hccl

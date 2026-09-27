/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_res_pack.h"

#include "ccu_device_pub.h"
#include "ccu_log.h"
#include "ccu_res_repo.h"
#include "ccu_types.h"
#include "hcom_common.h"

namespace hcomm {

CcuResPack::~CcuResPack() { CcuReleaseRes(); }

void CcuResPack::CcuReleaseRes()
{
    if (resHandle_ != 0) {
        auto ret = CcuReleaseResHandle(userDevId_, resHandle_);
        if (ret != HcclResult::HCCL_SUCCESS) {
            HCCL_ERROR("[CcuResPack][%s] failed, resHandle[%p] userDevId[%d].", __func__, resHandle_, userDevId_);
        }
        resHandle_ = 0;
    }

    // 释放countXn block
    std::lock_guard<std::mutex> lock(cascCntBlockMutex_);
    for (uint8_t dieId = 0; dieId < CCU_MAX_IODIE_NUM; dieId++) {
        if (cascCntBlocks_[dieId].empty()) {
            continue;
        }
        for (auto& cascCntBlock : cascCntBlocks_[dieId]) {
            auto ret = CcuReleaseCntXnBlock(userDevId_, dieId, *(cascCntBlock.first.get()));
            if (ret != CcuResult::CCU_SUCCESS) {
                HCCL_ERROR(
                    "[CcuResPack][%s] release cntXn block failed, userDevId[%d], dieId[%d], blockIdx[%d], "
                    "cascCntBlock[%p], ret[%d].",
                    __func__, userDevId_, dieId, cascCntBlock.first->blockIdx, cascCntBlock.first.get(), ret);
            }
        }
        cascCntBlocks_[dieId].clear();
    }

    return;
}

CcuResult CcuResPack::Reset()
{
    if (!resHandle_) {
        return CcuResult::CCU_SUCCESS;
    }

    CCU_CHK_RET(CcuCheckResource(userDevId_, resHandle_, resRepo_));
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuResPack::InitByInsType(const CcuInstanceType insType)
{
    userDevId_ = HcclGetThreadDeviceId();
    if (insType == CcuInstanceType::CCU_UNUSED) {
        HCCL_ERROR("[CcuResPack][%s] failed, error ccu instance type[%d].", __func__, static_cast<int32_t>(insType));
        return CcuResult::CCU_E_PARA;
    }

    // 根据通信域算子展开模式申请资源
    // 如果资源不足，返回HCCL_E_UNAVAIL，表示需要回退
    auto ret = CcuAllocResHandleByInsType(userDevId_, insType, resHandle_);
    if (ret == CcuResult::CCU_E_UNAVAIL) {
        HCCL_RUN_WARNING(
            "[%s] failed but passed, resource is not enough, "
            "userDevId[%d], ccuInsType[%d].",
            __func__, userDevId_, insType);
        return ret;
    }
    CCU_CHK_RET(ret);
    CCU_CHK_RET(Reset());
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuResPack::AllocCascCntBlock(const CcuResDesc* descs[], uint32_t descNum)
{
    uint32_t maxCascCntReq[CCU_MAX_IODIE_NUM] = {0};
    for (uint32_t i = 0; i < descNum; i++) {
        if (descs[i] != nullptr) {
            uint32_t cascCntNum = 0;
            CCU_CHK_RET(descs[i]->QueryResNum(ResType::CASC_CNT, cascCntNum));
            uint8_t dieId = static_cast<uint8_t>(descs[i]->dieId);
            maxCascCntReq[dieId] += cascCntNum;
        }
    }

    std::lock_guard<std::mutex> lock(cascCntBlockMutex_);
    for (uint8_t dieId = 0; dieId < CCU_MAX_IODIE_NUM; dieId++) {
        if (maxCascCntReq[dieId] > CCU_V2_RESOURCE_TOTAL_CNT_XNS_NUM) {
            HCCL_ERROR(
                "[CcuResPack][%s] failed, die[%u] no available maxCascCntReq[%u] > "
                "CCU_V2_RESOURCE_TOTAL_CNT_XNS_NUM[%u].",
                __func__, dieId, maxCascCntReq[dieId], CCU_V2_RESOURCE_TOTAL_CNT_XNS_NUM);
            return CcuResult::CCU_E_UNAVAIL;
        }
        if (maxCascCntReq[dieId] == 0) {
            continue;
        }
        for (uint32_t i = 0; i < maxCascCntReq[dieId]; i++) {
            std::unique_ptr<CntXnBlock> cntXnBlock{nullptr};
            EXCEPTION_CATCH(cntXnBlock = std::make_unique<CntXnBlock>(), return CcuResult::CCU_E_INTERNAL);
            CCU_CHK_RET(CcuAllocCntXnBlock(userDevId_, dieId, *cntXnBlock));
            cascCntBlocks_[dieId].emplace(std::move(cntXnBlock), false);
        }
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResult CcuResPack::InitByResDescs(const CcuResDesc* descs[], uint32_t descNum)
{
    userDevId_ = HcclGetThreadDeviceId();

    // 基于 resDesc 驱动的资源数量申请资源
    // 如果资源不足，返回CCU_E_UNAVAIL，表示需要回退
    auto ret = CcuAllocResHandleByResDescs(userDevId_, descs, descNum, resHandle_);
    if (ret == CcuResult::CCU_E_UNAVAIL) {
        HCCL_RUN_WARNING(
            "[%s] failed but passed, resource is not enough, "
            "userDevId[%d], descNum[%u].",
            __func__, userDevId_, descNum);
        return ret;
    }
    CCU_CHK_RET(ret);
    CCU_CHK_RET(Reset());
    auto cascCntRet = AllocCascCntBlock(descs, descNum);
    if (cascCntRet != CcuResult::CCU_SUCCESS) {
        // 透传 AllocCascCntBlock 的原始返回码，并打日志补充 devLogicId/descNum
        // 上下文，便于区分是块数超上限还是底层申请失败
        HCCL_ERROR(
            "[CcuResPack][%s] failed, alloc cascCnt block failed, ret[%d], userDevId[%d], descNum[%u], "
            "release allocated resource.",
            __func__, cascCntRet, userDevId_, descNum);
        CcuReleaseRes();
        return cascCntRet;
    }
    return CcuResult::CCU_SUCCESS;
}

CcuResRepository& CcuResPack::GetCcuResRepo() { return resRepo_; }

CcuResult CcuResPack::AcquireCascCntBlock(uint8_t dieId, HcommCcuCascCntHandle handle)
{
    if (dieId >= CCU_MAX_IODIE_NUM) {
        HCCL_ERROR(
            "[CcuResPack][%s] failed, userDevId[%d], dieId[%u] is out of range[0, %u).", __func__, userDevId_, dieId,
            CCU_MAX_IODIE_NUM);
        return CcuResult::CCU_E_PARA;
    }
    // handle 由上层 CcuInstanceMgr 全局唯一分配后传入, 本类仅按其登记, 不再自增;
    // 0 恒为无效句柄(HcommCcuCascCntAlloc/GetMem 均以 0 表示无效)。
    if (handle == 0) {
        HCCL_ERROR("[CcuResPack][%s] failed, userDevId[%d], invalid cascCnt handle[0].", __func__, userDevId_);
        return CcuResult::CCU_E_PARA;
    }

    std::lock_guard<std::mutex> lock(cascCntBlockMutex_);
    auto& blocks = cascCntBlocks_[dieId];
    if (blocks.empty()) {
        HCCL_ERROR(
            "[CcuResPack][%s] failed, userDevId[%d], dieId[%u], cascCntBlocks is empty.", __func__, userDevId_, dieId);
        return CcuResult::CCU_E_UNAVAIL;
    }

    for (auto& cascCntBlock : blocks) {
        if (cascCntBlock.second) {
            continue;
        }
        cascCntBlock.second = true;
        std::unique_lock<std::shared_timed_mutex> lock(handleMapMutex_);
        EXCEPTION_CATCH(handleMap_.emplace(handle, *cascCntBlock.first.get()), return CcuResult::CCU_E_INTERNAL);
        return CcuResult::CCU_SUCCESS;
    }
    return CcuResult::CCU_E_UNAVAIL;
}

CcuResult CcuResPack::GetCascCntBlock(HcommCcuCascCntHandle handle, CntXnBlock& cascCntBlock)
{
    std::shared_lock<std::shared_timed_mutex> lock(handleMapMutex_);
    auto it = handleMap_.find(handle);
    if (it == handleMap_.end()) {
        HCCL_ERROR("[CcuResPack][%s] failed, handle[%llx] is not existed.", __func__, handle);
        return CcuResult::CCU_E_NOT_FOUND;
    }

    cascCntBlock = it->second;
    return CcuResult::CCU_SUCCESS;
}
} // namespace hcomm

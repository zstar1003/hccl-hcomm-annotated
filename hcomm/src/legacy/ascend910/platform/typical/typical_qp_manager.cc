/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "typical_qp_manager.h"
#include <map>
#include <memory>
#include "externalinput.h"
#include "adapter_rts_common.h"
#include "rdma_resource_manager.h"

namespace hccl {
TypicalQpManager::TypicalQpManager() {}

TypicalQpManager::~TypicalQpManager()
{
    std::unique_lock<std::mutex> cqLock(cqMutex_);
    for (auto& item : cqMap_) {
        if (item.second.second == nullptr) {
            continue;
        }
        DestroyTypicalCq(rdmaHandle_, item.first, item.second.second);
    }
    cqMap_.clear();
    cqLock.unlock();

    std::unique_lock<std::mutex> lock(qpMutex_);
    for (auto& item : qpMap_) {
        if (item.second.second == nullptr) {
            continue;
        }
        HrtRaQpDestroy(item.second.second);
    }
    qpMap_.clear();
    for (auto& item : verbsQpMap_) {
        if (item.second.second == nullptr) {
            continue;
        }
        HrtRaQpDestroyWithoutCQ(item.second.second);
    }
    verbsQpMap_.clear();
    lock.unlock();
}

TypicalQpManager& TypicalQpManager::GetInstance()
{
    static TypicalQpManager qpInstance[MAX_MODULE_DEVICE_NUM + 1];
    s32 deviceLogicId = INVALID_INT;
    HcclResult ret = hrtGetDevice(&deviceLogicId);
    if (ret == HCCL_SUCCESS && (static_cast<u32>(deviceLogicId) < MAX_MODULE_DEVICE_NUM)) {
        HCCL_INFO("[TypicalQpManager::GetInstance]deviceLogicID[%d]", deviceLogicId);
        return qpInstance[deviceLogicId];
    }
    HCCL_WARNING("[TypicalQpManager::GetInstance]deviceLogicID[%d] is invalid, ret[%d]", deviceLogicId, ret);
    return qpInstance[MAX_MODULE_DEVICE_NUM];
}

HcclResult TypicalQpManager::CreateQp(struct TypicalQp& qpInfo)
{
    HcclResult ret = HCCL_SUCCESS;
    QpHandle qpHandle = nullptr;
    CHK_RET(RdmaResourceManager::GetInstance().GetRdmaHandle(rdmaHandle_));
    CHK_PTR_NULL(rdmaHandle_);
    std::unique_lock<std::mutex> lock(qpMutex_);
    ret = hrtRaTypicalQpCreate(rdmaHandle_, QP_FLAG_RC, OPBASE_QP_MODE, &qpInfo, qpHandle);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[TypicalQpManager][CreateQp] Create qp failed."), HCCL_E_INTERNAL);
    qpMap_.insert(std::make_pair(qpInfo.qpn, std::make_pair(qpInfo, qpHandle)));
    return HCCL_SUCCESS;
}

HcclResult TypicalQpManager::CreateQp(struct TypicalQp& qpInfo, const QpConfigInfo& qpConfig)
{
    HcclResult ret = HCCL_SUCCESS;
    QpHandle qpHandle = nullptr;
    CHK_RET(RdmaResourceManager::GetInstance().GetRdmaHandle(rdmaHandle_));
    CHK_PTR_NULL(rdmaHandle_);
    std::unique_lock<std::mutex> lock(qpMutex_);
    ret = CreateQpWithDepthConfig(rdmaHandle_, OPBASE_QP_MODE, qpConfig, qpHandle, qpInfo);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[TypicalQpManager][CreateQp] Create qp failed."), HCCL_E_INTERNAL);
    qpMap_.insert(std::make_pair(qpInfo.qpn, std::make_pair(qpInfo, qpHandle)));
    return HCCL_SUCCESS;
}

HcclResult TypicalQpManager::CreateCq(AscendCQInfo& cqInfo)
{
    HcclResult ret = HCCL_SUCCESS;
    void* cqHandle = nullptr;
    CHK_RET(RdmaResourceManager::GetInstance().GetRdmaHandle(rdmaHandle_));
    CHK_PTR_NULL(rdmaHandle_);
    std::unique_lock<std::mutex> lock(cqMutex_);
    ret = CreateTypicalCq(rdmaHandle_, cqInfo.cqDepth, cqInfo.cqn, &cqHandle);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[TypicalQpManager][CreateCq] Create cq failed."), HCCL_E_INTERNAL);
    cqMap_.insert(std::make_pair(cqInfo.cqn, std::make_pair(cqInfo, cqHandle)));
    return HCCL_SUCCESS;
}

HcclResult TypicalQpManager::DestroyCq(uint32_t cqn)
{
    HcclResult ret = HCCL_SUCCESS;
    CHK_RET(RdmaResourceManager::GetInstance().GetRdmaHandle(rdmaHandle_));
    CHK_PTR_NULL(rdmaHandle_);
    std::unique_lock<std::mutex> lock(cqMutex_);
    auto it = cqMap_.find(cqn);
    CHK_PRT_RET((it == cqMap_.end()), HCCL_ERROR("[TypicalQpManager][DestroyCq] cqn[%u] not found.", cqn), HCCL_E_PARA);
    ret = DestroyTypicalCq(rdmaHandle_, cqn, it->second.second);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[TypicalQpManager][DestroyCq] Destroy cq failed."), HCCL_E_INTERNAL);
    cqMap_.erase(it);
    return HCCL_SUCCESS;
}

HcclResult TypicalQpManager::ValidateCq(uint32_t cqn)
{
    std::unique_lock<std::mutex> lock(cqMutex_);
    auto it = cqMap_.find(cqn);
    CHK_PRT_RET(
        (it == cqMap_.end()), HCCL_ERROR("[TypicalQpManager][ValidateCq] cqn[%u] not found.", cqn), HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult TypicalQpManager::GetCqDepth(uint32_t cqn, uint32_t& cqDepth)
{
    std::unique_lock<std::mutex> lock(cqMutex_);
    auto it = cqMap_.find(cqn);
    CHK_PRT_RET(
        (it == cqMap_.end()), HCCL_ERROR("[TypicalQpManager][GetCqDepth] cqn[%u] not found.", cqn), HCCL_E_PARA);
    cqDepth = it->second.first.cqDepth;
    return HCCL_SUCCESS;
}

HcclResult TypicalQpManager::GetCqHandle(uint32_t cqn, void*& cqHandle)
{
    std::unique_lock<std::mutex> lock(cqMutex_);
    auto it = cqMap_.find(cqn);
    CHK_PRT_RET(
        (it == cqMap_.end()), HCCL_ERROR("[TypicalQpManager][GetCqHandle] cqn[%u] not found.", cqn), HCCL_E_PARA);
    cqHandle = it->second.second;
    return HCCL_SUCCESS;
}

HcclResult TypicalQpManager::CreateQpWithCQ(struct TypicalQp& qpInfo, const QpConfigWithCQInfo& qpConfig)
{
    HcclResult ret = HCCL_SUCCESS;
    QpHandle qpHandle = nullptr;
    CHK_RET(RdmaResourceManager::GetInstance().GetRdmaHandle(rdmaHandle_));
    CHK_PTR_NULL(rdmaHandle_);
    std::unique_lock<std::mutex> lock(qpMutex_);
    ret = CreateQpWithCQConfig(rdmaHandle_, OPBASE_QP_MODE, qpConfig, qpHandle, qpInfo);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[TypicalQpManager][CreateQpWithCQ] Create qp with cq failed."),
        HCCL_E_INTERNAL);
    verbsQpMap_.insert(std::make_pair(qpInfo.qpn, std::make_pair(qpInfo, qpHandle)));
    return HCCL_SUCCESS;
}

HcclResult TypicalQpManager::ModifyQp(struct TypicalQp& localQpInfo, struct TypicalQp& remoteQpInfo)
{
    CHK_PRT_RET(
        (localQpInfo.qpn == 0 || remoteQpInfo.qpn == 0),
        HCCL_ERROR("[TypicalQpManager][ModifyQp] the qpinfo is wrong, qpn is 0."), HCCL_E_PARA);
    QpHandle qpHandle = nullptr;
    CHK_RET(GetQpHandleByQpn(localQpInfo.qpn, qpHandle));
    CHK_PTR_NULL(qpHandle);
    CHK_RET(SetQpRdmaRetryCfg(localQpInfo));
    std::unique_lock<std::mutex> lock(qpMutex_);
    HcclResult ret = hrtRaTypicalQpModify(qpHandle, &localQpInfo, &remoteQpInfo);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[TypicalQpManager][ModifyQp] Modify qp failed."), HCCL_E_INTERNAL);

    return HCCL_SUCCESS;
}

HcclResult TypicalQpManager::ModifyVerbsQp(struct TypicalQp& localQpInfo, struct TypicalQp& remoteQpInfo)
{
    CHK_PRT_RET(
        (localQpInfo.qpn == 0 || remoteQpInfo.qpn == 0),
        HCCL_ERROR("[TypicalQpManager][ModifyVerbsQp] the qpinfo is wrong, qpn is 0."), HCCL_E_PARA);
    QpHandle qpHandle = nullptr;
    CHK_RET(GetVerbsQpHandleByQpn(localQpInfo.qpn, qpHandle));
    CHK_PTR_NULL(qpHandle);
    CHK_RET(SetQpRdmaRetryCfg(localQpInfo));
    std::unique_lock<std::mutex> lock(qpMutex_);
    HcclResult ret = hrtRaTypicalQpModify(qpHandle, &localQpInfo, &remoteQpInfo);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[TypicalQpManager][ModifyVerbsQp] Modify qp failed."), HCCL_E_INTERNAL);

    return HCCL_SUCCESS;
}

HcclResult TypicalQpManager::DestroyQp(struct TypicalQp& qpInfo)
{
    CHK_PRT_RET(
        (qpInfo.qpn == 0), HCCL_ERROR("[TypicalQpManager][DestroyQp] The qpinfo is wrong, qpn is 0."), HCCL_E_PARA);
    QpHandle qpHandle;
    CHK_RET(GetQpHandleByQpn(qpInfo.qpn, qpHandle));
    CHK_PTR_NULL(qpHandle);
    std::unique_lock<std::mutex> lock(qpMutex_);
    HcclResult ret = HrtRaQpDestroy(qpHandle);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[TypicalQpManager][DestroyQp] Destroy qp failed."), HCCL_E_INTERNAL);
    qpMap_.erase(qpInfo.qpn);
    return HCCL_SUCCESS;
}

HcclResult TypicalQpManager::DestroyQpWithoutCQ(struct TypicalQp& qpInfo)
{
    CHK_PRT_RET(
        (qpInfo.qpn == 0), HCCL_ERROR("[TypicalQpManager][DestroyQpWithoutCQ] The qpinfo is wrong, qpn is 0."),
        HCCL_E_PARA);
    QpHandle qpHandle;
    CHK_RET(GetVerbsQpHandleByQpn(qpInfo.qpn, qpHandle));
    CHK_PTR_NULL(qpHandle);
    std::unique_lock<std::mutex> lock(qpMutex_);
    HcclResult ret = HrtRaQpDestroyWithoutCQ(qpHandle);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[TypicalQpManager][DestroyQpWithoutCQ] Destroy qp without cq failed."),
        HCCL_E_INTERNAL);
    verbsQpMap_.erase(qpInfo.qpn);
    return HCCL_SUCCESS;
}

HcclResult TypicalQpManager::SetQpRdmaRetryCfg(struct TypicalQp& qpInfo)
{
    qpInfo.retryCnt = GetExternalInputRdmaRetryCnt();
    qpInfo.retryTime = GetExternalInputRdmaTimeOut();
    HCCL_INFO(
        "[TypicalQpManager][SetQpCreateBaseInfo] Qpinfo is set, tc is %u, sl is %u, retry cnt is %u, "
        "retry time is %u",
        qpInfo.tc, qpInfo.sl, qpInfo.retryCnt, qpInfo.retryTime);
    return HCCL_SUCCESS;
}

HcclResult TypicalQpManager::GetQpHandleByQpn(u32 qpn, QpHandle& qpHandle)
{
    HCCL_DEBUG("[TypicalQpManager][GetQpHandleByQpn] Get qpHandle by qpn[%u]", qpn);
    CHK_RET(RdmaResourceManager::GetInstance().GetRdmaHandle(rdmaHandle_));
    CHK_PTR_NULL(rdmaHandle_);
    std::unique_lock<std::mutex> lock(qpMutex_);
    auto it = qpMap_.find(qpn);
    CHK_PRT_RET(
        (it == qpMap_.end()), HCCL_ERROR("[TypicalQpManager][GetQpHandleByQpn] Qpn is not found"), HCCL_E_NOT_FOUND);
    qpHandle = it->second.second;
    CHK_PRT_RET(
        (qpHandle == nullptr),
        HCCL_ERROR("[TypicalQpManager][GetQpHandleByQpn] Get Qphandle failed, qphandle is nullptr. qpn is %u", qpn),
        HCCL_E_NOT_FOUND);
    return HCCL_SUCCESS;
}

HcclResult TypicalQpManager::GetVerbsQpHandleByQpn(u32 qpn, QpHandle& qpHandle)
{
    HCCL_DEBUG("[TypicalQpManager][GetVerbsQpHandleByQpn] Get qpHandle by qpn[%u]", qpn);
    CHK_RET(RdmaResourceManager::GetInstance().GetRdmaHandle(rdmaHandle_));
    CHK_PTR_NULL(rdmaHandle_);
    std::unique_lock<std::mutex> lock(qpMutex_);
    auto it = verbsQpMap_.find(qpn);
    CHK_PRT_RET(
        (it == verbsQpMap_.end()), HCCL_ERROR("[TypicalQpManager][GetVerbsQpHandleByQpn] Qpn is not found"),
        HCCL_E_NOT_FOUND);
    qpHandle = it->second.second;
    CHK_PRT_RET(
        (qpHandle == nullptr),
        HCCL_ERROR(
            "[TypicalQpManager][GetVerbsQpHandleByQpn] Get Qphandle failed, qphandle is nullptr. qpn is %u", qpn),
        HCCL_E_NOT_FOUND);
    return HCCL_SUCCESS;
}
} // namespace hccl

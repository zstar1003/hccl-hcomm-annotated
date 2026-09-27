/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclgraph_callback.h"
#include "stream_utils.h"

namespace hccl {

void AclgraphDestroyCallback(void* fnData)
{
    AclgraphDestroyCallbackParam* callbackParam = static_cast<AclgraphDestroyCallbackParam*>(fnData);
    if (callbackParam == nullptr) {
        HCCL_ERROR("[%s] callbackParam ptr is NULL", __func__);
        return;
    }

    HCCL_INFO("[%s] Entry modelID[%llu] CleanCaptureRes", __func__, callbackParam->modelId);
    HcclResult ret = AclgraphCallback::GetInstance().CleanCaptureRes(callbackParam->modelId);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] modelID[%llu] CleanCaptureRes failed", __func__, callbackParam->modelId);
    }
}

AclgraphCallback& AclgraphCallback::GetInstance()
{
    static AclgraphCallback aclgraphCallback;
    return aclgraphCallback;
}

AclgraphCallback::~AclgraphCallback()
{
    std::lock_guard<std::mutex> lock(resMutex_);
    captureResMap_.clear();
    captureCallbackParamMap_.clear();
}

HcclResult AclgraphCallback::CleanCaptureRes(u64 modelId)
{
    HcclResult ret;

    std::lock_guard<std::mutex> lock(resMutex_);
    auto modelIt = captureResMap_.find(modelId);
    if (modelIt == captureResMap_.end()) {
        HCCL_ERROR("[%s] modelID[%llu] is not recorded", __func__, modelId);
        return HCCL_E_NOT_FOUND;
    }

    bool isResourceReleaseFailed = false;
    for (auto& commIt : modelIt->second) {
        // 1. 整批 RPC sync aicpu 端：一次 launch erase 所有 tag 的 7 map entry，内含 sync，返回后 aicpu 不再访问这些
        // tag
        HcclResult aicpuRet = commIt.first->AicpuKfcClearOpResLaunch(commIt.second);
        if (aicpuRet != HCCL_SUCCESS) {
            HCCL_RUN_WARNING(
                "[%s] modelID[%llu] aicpu batch sync fail, tagCount[%zu] ret[%d]; "
                "skip host link surgery this batch, tagsRequiringHostCleanup_ entries retained",
                __func__, modelId, commIt.second.size(), aicpuRet);
            isResourceReleaseFailed = true;
        }

        // 2. host 端逐 tag 清自己 resMap_/tagStreamInfo_ 等 host 进程内状态，与 aicpu sync 独立；aicpu 失败也要清，否则
        // resMap_ 残留
        for (auto& newTag : commIt.second) {
            ret = commIt.first->ClearOpResource(newTag, true);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[%s] modelID[%llu] tag[%s] host resource release fail, ret[%d]", __func__, modelId, newTag.c_str(),
                    ret);
                isResourceReleaseFailed = true;
            }
            HCCL_DEBUG("[%s] modelID[%llu] tag[%s] host resource release finish", __func__, modelId, newTag.c_str());
        }
        // 3. aicpu 已 sync，host 端整批 ListCommonRemove + 三容器 erase race-free；aicpu 失败时跳过，tag 保留至析构
        if (aicpuRet == HCCL_SUCCESS) {
            (void)commIt.first->ClearAclgraphHostLinks(commIt.second);
        }
    }

    // 清理通信域上的AIV modelId, aclgraph上modelId是可以复用的
    for (auto& commIt : modelIt->second) {
        commIt.first->EraseCaptureModelId(modelId);
    }

    captureResMap_.erase(modelId);
    captureCallbackParamMap_.erase(modelId);
    if (isResourceReleaseFailed) {
        HCCL_RUN_WARNING("[%s] modelID[%llu] resource release partially failed", __func__, modelId);
    } else {
        HCCL_INFO("[%s] modelID[%llu] resource release success", __func__, modelId);
    }

    return isResourceReleaseFailed ? HCCL_E_INTERNAL : HCCL_SUCCESS;
}

void AclgraphCallback::CleanCaptureRes(HcclCommunicator* communicator)
{
    if (communicator == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(resMutex_);
    for (auto& modelIt : captureResMap_) {
        if (modelIt.second.find(communicator) != modelIt.second.end()) {
            modelIt.second.erase(communicator);
        }
    }

    HCCL_INFO("[%s] communicator[%p] resource release success", __func__, communicator);
}

// 记录aclgraph下发的所有tag, 首次记录时注册aclgraph销毁回调
HcclResult AclgraphCallback::InsertNewTagToCaptureResMap(
    HcclCommunicator* communicator, const std::string& newTag, const OpParam& opParam)
{
    CHK_PTR_NULL(communicator);
    aclmdlRI rtModel = nullptr;
    bool isCapture = false;
    u64 modelId = 0;

    CHK_RET(GetStreamCaptureInfo(opParam.stream.ptr(), rtModel, isCapture));
    CHK_PTR_NULL(rtModel);
    CHK_RET(GetModelId(rtModel, modelId));

    std::lock_guard<std::mutex> lock(resMutex_);
    CHK_RET(RegisterDestroyCallbackLocked(rtModel, modelId));
    captureResMap_[modelId][communicator].insert(newTag);
    HCCL_DEBUG("[%s] captureResMap insert tag[%s] to modelID[%llu]", __func__, newTag.c_str(), modelId);

    return HCCL_SUCCESS;
}

HcclResult AclgraphCallback::RegisterModelId(HcclCommunicator* communicator, aclmdlRI rtModel, u64 modelId)
{
    CHK_PTR_NULL(communicator);
    CHK_PTR_NULL(rtModel);

    std::lock_guard<std::mutex> lock(resMutex_);
    CHK_RET(RegisterDestroyCallbackLocked(rtModel, modelId));
    captureResMap_[modelId][communicator];
    HCCL_DEBUG("[%s] register communicator[%p] to modelID[%llu]", __func__, communicator, modelId);

    return HCCL_SUCCESS;
}

HcclResult AclgraphCallback::RegisterDestroyCallbackLocked(aclmdlRI rtModel, u64 modelId)
{
    if (captureResMap_.find(modelId) != captureResMap_.end()) {
        return HCCL_SUCCESS;
    }
    captureCallbackParamMap_[modelId].modelId = modelId;
    aclError aclRet = aclmdlRIDestroyRegisterCallback(
        rtModel, AclgraphDestroyCallback, static_cast<void*>(&captureCallbackParamMap_[modelId]));
    CHK_PRT_RET(
        aclRet != ACL_SUCCESS,
        HCCL_ERROR("[%s] aclmdlRIDestroyRegisterCallback fail, modelId[%llu]", __func__, modelId), HCCL_E_RUNTIME);
    HCCL_INFO("[%s] aclmdlRIDestroyRegisterCallback success modelID[%llu]", __func__, modelId);

    return HCCL_SUCCESS;
}
} // namespace hccl

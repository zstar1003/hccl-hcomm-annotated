/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "stream_utils.h"
#include <unordered_map>
#include <functional>
#include "log.h"
#include "rt_external.h"
#include "error_codes/rt_error_codes.h"
#include "workflow_pub.h"

static const std::unordered_map<int, std::function<void(bool&)>> captureStatusHandlers = {
    // ACL Graph 获取capture状态处理
    {aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE,
     [](bool& isCapture) {
         isCapture = true;
     }},
    {aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE,
     []([[maybe_unused]] bool& isCapture) {
         HCCL_DEBUG("[GetStreamCaptureInfo]Stream capture status NONE.");
     }},
    {aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_INVALIDATED, []([[maybe_unused]] bool& isCapture) {
         HCCL_ERROR("[GetStreamCaptureInfo]Stream capture status invalidated.");
     }}};

HcclResult GetStreamCaptureInfo(aclrtStream stream, aclmdlRI& rtModel, bool& isCapture)
{
    isCapture = false;
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclError ret = aclmdlRICaptureGetInfo(stream, &captureStatus, &rtModel);
    if (ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        HCCL_WARNING("[%s]Stream capture not support.", __func__);
        return HCCL_SUCCESS;
    } else {
        CHK_PRT_RET(
            ret != ACL_SUCCESS, HCCL_ERROR("[%s]aclmdlRICaptureGetInfo fail. return[%d].", __func__, ret),
            HCCL_E_RUNTIME);
    }
    auto it = captureStatusHandlers.find(captureStatus);
    if (it != captureStatusHandlers.end()) {
        it->second(isCapture);
    } else {
        HCCL_ERROR("[%s]Unsupported stream capture status.", __func__);
    }
    return HCCL_SUCCESS;
}

HcclResult AddStreamToModel(rtStream_t stream, rtModel_t& rtModel)
{
    rtError_t ret = rtStreamAddToModel(stream, rtModel);
    if (ret != RT_ERROR_NONE) {
        HCCL_ERROR("[%s]rtStreamAddToModel failed. ret[%d].", __func__, ret);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult GetModelId(aclmdlRI& rtModel, u64& modelId)
{
    uint32_t mdlId;
    rtError_t rtRet = rtModelGetId(rtModel, &mdlId);
    CHK_PRT_RET(
        rtRet != RT_ERROR_NONE, HCCL_ERROR("[%s]rtGet stream get model id fail. return[%d]", __func__, rtRet),
        HCCL_E_RUNTIME);
    modelId = static_cast<uint64_t>(mdlId);
    return HCCL_SUCCESS;
}

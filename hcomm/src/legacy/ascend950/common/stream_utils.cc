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
#include "acl/acl_rt.h"
#include "rt_external.h"

namespace Hccl {
#ifdef CCL_FWK_LLT
#define ACL_ERROR_RT_FEATURE_NOT_SUPPORT 207000 // feature not support
#endif

static const std::unordered_map<int, std::function<void(bool&)>> captureStatusHandlers = {
    // ACL Graph 获取capture状态处理
    {aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE,
     [](bool& isCapture) {
         isCapture = true;
     }},
    {aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE,
     [](bool& isCapture) {
         HCCL_DEBUG("[GetStreamCaptureInfo]Stream capture status NONE, isCapture is %d", isCapture);
     }},
    {aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_INVALIDATED, [](bool& isCapture) {
         HCCL_ERROR("[GetStreamCaptureInfo]Stream capture status invalidated, isCapture is %d", isCapture);
     }}};

HcclResult GetStreamCaptureInfo(rtStream_t stream, rtModel_t& rtModel, bool& isCapture)
{
    isCapture = false;
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    rtError_t ret = aclmdlRICaptureGetInfo(stream, &captureStatus, &rtModel);
    if (ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        HCCL_WARNING("[%s]Stream capture not support.", __func__);
        return HCCL_SUCCESS;
    } else {
        CHK_PRT_RET(
            ret != RT_ERROR_NONE, HCCL_ERROR("[%s]rtStreamGetCaptureInfo fail. return[%d].", __func__, ret),
            HCCL_E_RUNTIME);
    }
    auto it = captureStatusHandlers.find(captureStatus);
    if (it != captureStatusHandlers.end()) {
        it->second(isCapture);
    } else {
        HCCL_ERROR("[%s]Unsupported stream capture status.", __func__);
        return HCCL_E_NOT_SUPPORT;
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

HcclResult GetModelId(rtModel_t& rtModel, u32& modelId)
{
    rtError_t ret = rtModelGetId(rtModel, &modelId);
    if (ret != RT_ERROR_NONE) {
        HCCL_ERROR("[%s]rtModelGetId failed. ret[%d].", __func__, ret);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

} // namespace Hccl

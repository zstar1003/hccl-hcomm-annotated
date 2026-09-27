/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "plugin_runner.h"
#include "adapter_rts_common.h"
#include "externalinput_pub.h"
#include "acl/error_codes/rt_error_codes.h"

using namespace hccl;
HcclResult PluginRunner::isStreamCapture(rtStream_t stream, bool& isCapture) const
{
    isCapture = false;
    DevType devType;
    CHK_RET(hrtGetDeviceType(devType));
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        HCCL_WARNING("[PluginRunner][isStreamCapture]Stream capture only supports opbase mode!");
        return HCCL_SUCCESS;
    }
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclmdlRI rtModel = nullptr;
    aclError ret = aclmdlRICaptureGetInfo(stream, &captureStatus, &rtModel);
    if (ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        HCCL_WARNING("[PluginRunner][isStreamCapture]Stream capture is not supported!");
        return HCCL_SUCCESS;
    } else {
        CHK_PRT_RET(
            ret != ACL_SUCCESS,
            HCCL_ERROR("[PluginRunner][isStreamCapture]rtGet stream get capture status fail. return[%d]", ret),
            HCCL_E_RUNTIME);
    }

    switch (captureStatus) {
        case aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE: {
            isCapture = true;
            break;
        }
        case aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_NONE: {
            isCapture = false;
            break;
        }
        case aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_INVALIDATED: {
            HCCL_ERROR("[PluginRunner][isStreamCapture]rtGet stream capture status invalidated.");
            break;
        }
        default: {
            HCCL_ERROR("[PluginRunner][isStreamCapture]rtGet does not support stream capture status.");
            break;
        }
    }
    return HCCL_SUCCESS;
}

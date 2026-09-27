/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_STREAM_UTILS_H
#define HCOMM_STREAM_UTILS_H

#include "hccl/base.h"
#include "acl/acl_rt.h"

HcclResult GetStreamCaptureInfo(aclrtStream stream, aclmdlRI& rtModel, bool& isCapture);
HcclResult AddStreamToModel(rtStream_t stream, rtModel_t& rtModel);
HcclResult GetModelId(aclmdlRI& rtModel, u64& modelId);

#endif

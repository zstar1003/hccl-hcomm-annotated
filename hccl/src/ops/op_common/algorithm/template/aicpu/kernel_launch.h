/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_AICPU_KERNEL_LAUNCH_H
#define OPS_HCCL_AICPU_KERNEL_LAUNCH_H

#include "alg_param.h"

namespace ops_hccl {

HcclResult RestoreVarDataBatchSendRecv(OpParam& param);

HcclResult RestoreVarDataAlltoAllV(OpParam& param, const AlgResourceCtxSerializable& resCtx);

HcclResult RestoreVarDataReduceScatterV(OpParam& param, const AlgResourceCtxSerializable& resCtx);

HcclResult RestoreVarDataAllGatherV(OpParam& param, const AlgResourceCtxSerializable& resCtx);

// [中文导读] [AllReduce逐行 S26] 声明IsResCtxCacheReusable接口：资源cacheValid成立且缓存记录的通信域地址等于本次域地址才可复用。
inline bool IsResCtxCacheReusable(const AlgResourceCtxSerializable& cachedResCtx, const OpParam& param)
// [中文导读] [AllReduce逐行 S27] 开始IsResCtxCacheReusable的函数体。
{
    // [中文导读] [AllReduce逐行 S28] 只有Host标记资源已复用且缓存通信域地址仍等于本次通信域时返回true。
    return param.cacheValid && cachedResCtx.commInfoPtr == param.hcclComm;
// [中文导读] [AllReduce逐行 S29] 结束IsResCtxCacheReusable函数体。
}

} // namespace ops_hccl
#endif

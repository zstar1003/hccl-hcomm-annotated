/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_ONE_SIDED_SERVICE_ADAPT_V2_H
#define HCCL_ONE_SIDED_SERVICE_ADAPT_V2_H

#include <hccl/hccl_types.h>
#include <hccl/base.h>
#include "hccl_common_v2.h"
#include "hccl_one_sided_data.h"

using HcclBatchData = struct {
    HcclComm comm;
    HcclCMDType cmdType;
    u32 remoteRank;
    HcclOneSideOpDesc* desc;
    u32 descNum;
    rtStream_t stream;
};
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
HcclResult HcclRegisterMemV2(HcclComm comm, u32 remoteRank, int type, void* addr, u64 size, HcclMemDesc* desc);

HcclResult HcclDeregisterMemV2(HcclComm comm, HcclMemDesc* desc);

HcclResult HcclExchangeMemDescV2(
    HcclComm comm, u32 remoteRank, HcclMemDescs* local, int timeout, HcclMemDescs* remote, u32* actualNum);

HcclResult HcclEnableMemAccessV2(HcclComm comm, HcclMemDesc* remoteMemDesc, HcclMem* remoteMem);

HcclResult HcclDisableMemAccessV2(HcclComm comm, HcclMemDesc* remoteMemDesc);

HcclResult HcclBatchPutV2(HcclComm comm, u32 remoteRank, HcclOneSideOpDesc* desc, u32 descNum, const rtStream_t stream);

HcclResult HcclBatchGetV2(HcclComm comm, u32 remoteRank, HcclOneSideOpDesc* desc, u32 descNum, const rtStream_t stream);
#ifdef __cplusplus
}
#endif // __cplusplus
#endif // HCCL_ONE_SIDED_SERVICE_ADAPT_V2_H

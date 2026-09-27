/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CTRL_PLANE_H_
#define HCCL_CTRL_PLANE_H_

#include <hccl/hccl_types.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

extern HcclResult CommGetLocalCCLBuf(HcclComm comm, void** addr, uint64_t* size);
extern HcclResult CommGetRemoteCCLBuf(HcclComm comm, uint32_t remoteRank, void** addr, uint64_t* size);
extern HcclResult CommGetKFCWorkSpace(HcclComm comm, void** addr, uint64_t* size);
extern HcclResult CommGetCCLBufSizeCfg(HcclComm comm, uint64_t* cclBufSize);
#ifdef __cplusplus
}
#endif // __cplusplus
#endif //

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_AICPU_SQEMGR_SQE_MGR_H
#define HCCLV2_AICPU_SQEMGR_SQE_MGR_H

#include <unordered_map>

#include "ascend_hal.h"
#include "hccl_sqe.h"
#include "types.h"

namespace Hccl {

struct SqInfo {
    u32 sqeCnt;
    u32 sqDepth;
    u32 sqTail;
    u32 sqHead;
    u64 sqBaseAddr;
    u8 sqeBuffer[AC_SQE_SIZE * AC_SQE_MAX_CNT];
};

class SqeMgr {
public:
    explicit SqeMgr(u32 devPhysicalId);

    HcclResult Begin(u32 sqId);

    HcclResult Add(u32 sqId, HcclSqe* sqe);

    HcclResult Commit(u32 sqId);

private:
    u32 QuerySqHead(u32 sqId);

    u32 QuerySqTail(u32 sqId);

    u32 QuerySqDepth(u32 sqId);

    u64 QuerySqBaseAddr(u32 sqId);

    u32 QuerySqStatusByType(u32 sqId, drvSqCqPropType_t type) const;

    void ConfigSqTail(u32 sqId, u32 value);

    void ConfigSqStatusByType(u32 sqId, drvSqCqPropType_t type, u32 value) const;

    u32 GetTailToHeadDist(u32 sqId, u32 head, u32 tail);

    void AddSqeToBuffer(void* bufferAddr, void* sqeAddr) const;

    u32 devPhyId;

    std::unordered_map<u32, std::unique_ptr<SqInfo>> sqInfos;
};

} // namespace Hccl

#endif

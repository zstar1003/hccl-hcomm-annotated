/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTOALL_V_STAGED_BASE_PUB_H
#define ALLTOALL_V_STAGED_BASE_PUB_H

#include "alg_template_base_pub.h"
#include "alltoallv_staged_calculator_pub.h"

namespace hccl {
class AlltoAllVStagedBase : public AlgTemplateBase {
public:
    explicit AlltoAllVStagedBase(const HcclDispatcher dispatcher);
    ~AlltoAllVStagedBase() override;

    HcclResult Prepare(
        DeviceMem& sendMem, DeviceMem& recvMem, StageAlltoAllVAddrInfo& sendAddrInfo,
        StageAlltoAllVAddrInfo& recvAddrInfo, bool isAlltoAllZCopyMode, Stream& mainStream) override;

protected:
    HcclResult LocalCopy(u32 rank);

    DeviceMem sendMem_;
    DeviceMem recvMem_;

    StageAlltoAllVAddrInfo sendAddrInfo_;
    StageAlltoAllVAddrInfo recvAddrInfo_;
    bool isAlltoAllZCopyMode_ = false;
    Stream* mainStreamPtr_{nullptr};
};
} // namespace hccl
#endif /* ALLTOALL_V_STAGED_BASE_PUB_H */

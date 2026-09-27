/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_ONE_SIDED_COMPONENT_LITE
#define HCCLV2_ONE_SIDED_COMPONENT_LITE

#include "dev_type.h"
#include "connected_link_mgr.h"
#include "base_config_legacy.h"
#include "coll_operator.h"
#include "coll_alg_component.h"
#include "rmt_data_buffer_mgr.h"

namespace Hccl {

class OneSidedComponentLite {
public:
    OneSidedComponentLite(
        RankId myRank, u32 rankSize, DevType devType, u64 scratchBufferSize, ConnectedLinkMgr* linkMgr,
        RmtDataBufferMgr* rmaDataBufferMgr)
        : myRank_(myRank),
          rankSize_(rankSize),
          devType_(devType),
          scratchBufferSize_(scratchBufferSize),
          linkMgr_(linkMgr),
          rmaDataBufferMgr_(rmaDataBufferMgr)
    {}
    virtual ~OneSidedComponentLite() = default;

    virtual HcclResult Orchestrate(const HcclAicpuOpLite& op, InsQuePtr queue);

protected:
    u32 myRank_ = INVALID_RANKID;
    u32 rankSize_ = 0;
    DevType devType_ = DevType::DEV_TYPE_NOSOC;
    u64 scratchBufferSize_ = 0;
    ConnectedLinkMgr* linkMgr_ = nullptr;
    RmtDataBufferMgr* rmaDataBufferMgr_{nullptr};
};

using OneSidedComponentLitePtr = std::shared_ptr<OneSidedComponentLite>;
} // namespace Hccl
#endif

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEM_TRANSPORT_COMMON_H
#define MEM_TRANSPORT_COMMON_H

#include "string_util.h"
#include "reduce_in.h"

namespace Hccl {
MAKE_ENUM(TransportType, P2P, UB, ROCE, UBoE)
MAKE_ENUM(TransportNotifyType, NORMAL, COUNT)

struct WithNotifyIn {
    TransportNotifyType notifyType_{TransportNotifyType::NORMAL};
    u32 index_{0};
    u32 userData_{0};
    WithNotifyIn(TransportNotifyType notifyType, u32 index, u32 userData = 0)
        : notifyType_(notifyType),
          index_(index),
          userData_(userData)
    {}
    std::string Describe() const
    {
        return StringFormat(
            "WithNotifyIn[notifyType=%s, index=%u, userData=%u]", notifyType_.Describe().c_str(), index_, userData_);
    }
};
} // namespace Hccl

#endif

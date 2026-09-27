/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_UB_JETTY_ID_LITE_H
#define HCCLV2_UB_JETTY_ID_LITE_H

#include "hccl/base.h"

namespace Hccl {

struct UbJettyLiteId {
    u32 dieId_;
    u32 funcId_;
    u32 jettyId_;

    u32 GetDieId() const { return dieId_; }

    u32 GetFuncId() const { return funcId_; }

    u32 GetJettyId() const { return jettyId_; }

    UbJettyLiteId(u32 dieId, u32 funcId, u32 jettyId) : dieId_(dieId), funcId_(funcId), jettyId_(jettyId) {}
};

struct UbJettyLiteAttr {
    const u64 dbAddr_;
    const u64 sqVa_;
    const u32 sqDepth_;
    const u32 tpn_;
    const bool dwqeCacheLocked_;
    const u32 jfcPollMode_; // 0代表STARS POLL， 1代表软件Poll
    const u64 sqCiAddr_;    // 预留给 软件poll CQ 的Jetty使用
    UbJettyLiteAttr(
        u64 dbAddr, u64 sqVa, u32 sqDepth, u32 tpn, bool dwqeCacheLocked = false, u32 jfcPollMode = 0, u64 sqCiAddr = 0)
        : dbAddr_(dbAddr),
          sqVa_(sqVa),
          sqDepth_(sqDepth),
          tpn_(tpn),
          dwqeCacheLocked_(dwqeCacheLocked),
          jfcPollMode_(jfcPollMode),
          sqCiAddr_(sqCiAddr)
    {}
};

} // namespace Hccl
#endif

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __MC2_AICPU_UTILS_H__
#define __MC2_AICPU_UTILS_H__

#include <string>
#include "aicpu_schedule/aicpu_context.h"
#include "common/aicpu_hccl_def.h"
#include "hdc_pub.h"

class HcclAicpuUtils {
public:
    static int32_t GetCpuId();
    static int32_t GetCurClusterId();
    static void PrintHcclCombinOpParam(const HccCommResParamTask& commParam);
    static void PrintHcclOpResParam(const HcclOpResParam* resParam);
    static HcclResult Getkey(
        const AicpuComContext& ctx, u32 remoteRankId, const void* userAddr, u64 length, u32& outKey, int32_t keyType);
    static HcclResult PostSend(
        const AicpuComContext& ctx, u32 remoteRankId, struct std::vector<hccl::Transport::Buffer>& remoteBuf,
        struct std::vector<hccl::Transport::Buffer>& localBuf, bool isWrite);
    static HcclResult PostSend(
        const u32 lKey, const u32 rKey, const struct HcclQpInfoV2& qpInfo,
        const struct hccl::Transport::Buffer& remoteBuf, const struct hccl::Transport::Buffer& localBuf,
        const bool isWrite);
    static u32 GetBlockNum(u32 defaultVal = 1U);
    static u32 GetBlockIdx();
};
#endif // __MC2_AICPU_UTILS_HPP__

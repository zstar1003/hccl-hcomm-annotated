/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_CONTEXT_H
#define OP_CONTEXT_H

#include "hccl/base.h"

namespace hccl {
// nonuniform_hierarchical_ring_base_pub.h
constexpr u64 NHR_ALLREDUCE_SMALL_SIZE = 256 * 1024;  // server间allreduce数据大小256k及以下不切片
constexpr u64 NHR_BCAST_SMALL_SIZE = 2 * 1024 * 1024; // server间broadcast数据大小2M及以下不切片
// hccl_impl_pub.h
constexpr u32 LEVEL0_BRIDGE_RANK_ID = 0;

// reduce_scatter_pipeline_pub.h
constexpr u32 PIPELINE_DEPTH = 3;

// coll_alg_param.h
// InplaceSupportRetry算法枚举
enum class InplaceSupportRetryStatus {
    AG_BD_CASE = 0,
    RETRY_1_ALLOW_NO_DMA_REDUCE_CASE1 = 1, // executor需要成非DMA削减模式
    RETRY_0_NOT_ALLOW_NO_DMA_REDUCE_CASE1 = 2,
    ALWAYS_NO_DMA_REDUCE = 3,
    RETRY_1_ALLOW_NO_DMA_REDUCE_CASE2 = 4, // executor需要成非DMA削减模式
    RETRY_0_NOT_ALLOW_NO_DMA_REDUCE_CASE2 = 5,
    UNKONWN_EXECUTOR = 6,
    USER_LARGER_THAN_CCL = 7,
    NOT_BASIC_OP_CASE = 8,
    INPLACE_STATUS_END
};

struct OpRetryHandler {
    bool inplaceSupportRetry = false;
    bool retryEnable = false;
    InplaceSupportRetryStatus inPlaceSupportRetryStatus = InplaceSupportRetryStatus::INPLACE_STATUS_END;
    bool isInplacePreSync = false;
    bool isPostSync = false;
};

struct Mc2Handler {
    u64 version = 0;    // Mc2Handler 版本标记
    u64 commitAddr = 0; // mc2 条件算子的监听地址
    u64 finishAddr = 0; // mc2 写任务的地址
    u64 valueAddr = 0;
    u32 rankSize = 0;            // mc2 作用的卡数
    u32 repeatCnt = 0;           // 一次通信消息可下发多轮通信，标记为通信的轮数
    u8 stepSize = 0;             // 细粒度通信下的通信步长
    u8 skipLocalRankCopy = 0;    // 跳过本卡拷贝
    u8 skipBufferWindowCopy = 0; // 跳过user in到 cclbuffer 的拷贝
};

struct AlgOpContext {
    OpRetryHandler opRetryHandler;
    Mc2Handler mc2Handler;
};
} // namespace hccl

#endif /* * __OP_CONTEXT_H__ */

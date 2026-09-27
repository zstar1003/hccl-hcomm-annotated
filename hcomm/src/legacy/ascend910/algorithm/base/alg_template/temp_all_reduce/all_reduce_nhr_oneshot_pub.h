/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_REDUCE_NHR_ONESHOT_PUB_H
#define ALL_REDUCE_NHR_ONESHOT_PUB_H

#include "nonuniform_hierarchical_ring_base_pub.h"
#include "mem_host_pub.h"
#include "mem_device_pub.h"
#include "stream_pub.h"

#include "reducer_pub.h"
#include "sender_pub.h"

namespace hccl {
class AllReduceNHROneshot : public NHRBase {
public:
    using AlgTemplateBase::Prepare;
    explicit AllReduceNHROneshot(const HcclDispatcher dispatcher);
    ~AllReduceNHROneshot() override;

    // 新增的两段式构造函数，获取实例后要无脑调用实现构造函数功能，后续还要调用其它的基类Prepare函数实现其它成员变量初始化
    HcclResult Prepare(u64 reduceAttrBitMap, HcomCollOpInfo* opInfo = nullptr) override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links) override;

protected:
private:
    HcclResult RunReduceOneshot(u32 rank, u32 rankSize, const std::vector<LINK>& links);
    HcclResult RunBroadcastOneshot(u32 rank, u32 rankSize, const std::vector<LINK>& links);

    HcclResult SimpleCheck(const u32 rank, const u32 rankSize, const std::vector<LINK>& links);

    u64 reduceAttr_; /* 0x1:表示data_type + reduce_type支持inlinereduce  */
};
} // namespace hccl
#endif /* ALL_REDUCE_NHR_ONESHOT_PUB_H */

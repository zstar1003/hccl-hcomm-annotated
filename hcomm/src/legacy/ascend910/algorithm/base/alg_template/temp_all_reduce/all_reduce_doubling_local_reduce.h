/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLREDUCE_DOUBLING_LOCAL_REDUCE_EXECUTOR_H
#define ALLREDUCE_DOUBLING_LOCAL_REDUCE_EXECUTOR_H

#include "alg_template_base_pub.h"

namespace hccl {
class AllReduceDoublingLocalReduce : public AlgTemplateBase {
public:
    using AlgTemplateBase::Prepare;
    explicit AllReduceDoublingLocalReduce(const HcclDispatcher dispatcher);
    ~AllReduceDoublingLocalReduce() override;

    // 新增的两段式构造函数，获取实例后要无脑调用实现构造函数功能，后续还要调用其它的基类Prepare函数实现其它成员变量初始化
    HcclResult Prepare(u64 reduceAttrBitMap, HcomCollOpInfo* opInfo = nullptr) override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK>& links) override;

protected:
private:
    u64 reduceAttr_ = 0;

    HcclResult SimpleCheck(const u32 rank, const u32 rankSize, const std::vector<LINK>& links);
    HcclResult RunAllReduce(const u32 rank, const u32 rankSize, const std::vector<LINK>& links);
    HcclResult RunAllReduceSDMA(const LINK& link, DeviceMem& localCclOutMem, u64 totalSize);
    HcclResult RunAllReduceRDMA(const LINK& link, DeviceMem& localCclInMem, DeviceMem& localCclOutMem);
};
} // namespace hccl

#endif /* ALLREDUCE_DOUBLING_LOCAL_REDUCE_EXECUTOR_H */

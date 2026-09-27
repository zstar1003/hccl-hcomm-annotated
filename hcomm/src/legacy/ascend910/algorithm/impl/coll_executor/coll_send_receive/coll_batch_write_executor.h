/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_BATCH_WRITE_EXECUTOR_H
#define COLL_BATCH_WRITE_EXECUTOR_H
#include "coll_comm_executor.h"
#include "alg_template_base_pub.h"

namespace hccl {
constexpr char BATCH_WRITE_ALG_NAME[] = "BatchWriteBySdma";
class CollBatchWriteExecutor : public CollExecutorBase {
public:
    CollBatchWriteExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher)
        : CollExecutorBase(dispatcher, topoMatcher)
    {}
    ~CollBatchWriteExecutor() override = default;
    HcclResult Orchestrate(OpParam& param, AlgResourceResponse& algRes) override;
    HcclResult CalcResRequest(const OpParam& param, AlgResourceRequest& resourceRequest) override;
};
} // namespace hccl

#endif

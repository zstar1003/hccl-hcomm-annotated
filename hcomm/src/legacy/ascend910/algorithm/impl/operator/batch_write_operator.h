/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BATCH_WRITE_OPERATOR_H
#define BATCH_WRITE_OPERATOR_H

#include <set>
#include "coll_alg_operator.h"
#include "coll_alg_op_registry.h"

namespace hccl {
class BatchWriteOperator : public CollAlgOperator {
public:
    BatchWriteOperator(
        AlgConfigurator* algConfigurator, CCLBufferManager& cclBufferManager, HcclDispatcher dispatcher,
        std::unique_ptr<TopoMatcher>& topoMatcher)
        : CollAlgOperator(algConfigurator, cclBufferManager, dispatcher, topoMatcher, HcclCMDType::HCCL_CMD_BATCH_WRITE)
    {}
    ~BatchWriteOperator() override = default;
};
} // namespace hccl

#endif

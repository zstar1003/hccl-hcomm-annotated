/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_SCATTER_SINGLE_RANK_EXECUTOR_H
#define COLL_SCATTER_SINGLE_RANK_EXECUTOR_H

#include "coll_scatter_executor.h"
#include "coll_alg_exec_registry.h"

namespace hccl {
class CollScatterSingleRankExecutor : public CollScatterExecutor {
public:
    CollScatterSingleRankExecutor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher>& topoMatcher);
    ~CollScatterSingleRankExecutor() override = default;

private:
    HcclResult KernelRun(const OpParam& param, ExecMem& execMem) override;
};
} // namespace hccl

#endif

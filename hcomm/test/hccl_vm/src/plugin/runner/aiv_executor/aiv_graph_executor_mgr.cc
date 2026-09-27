/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_graph_executor_mgr.h"

#include <cstdint>

AivGraphExecutorMgr& AivGraphExecutorMgr::GetInstance()
{
    static AivGraphExecutorMgr instance;
    return instance;
}

std::shared_ptr<AivGraphExecutor> AivGraphExecutorMgr::GetAivGraphExecutor(uint32_t rankId, uint64_t launchIdx)
{
    const auto key = std::make_pair(rankId, launchIdx);
    auto iter = executorMap_.find(key);
    if (iter == executorMap_.end()) {
        iter = executorMap_.emplace(key, std::make_shared<AivGraphExecutor>(launchIdx)).first;
    }
    return iter->second;
}

void AivGraphExecutorMgr::Reset() { executorMap_.clear(); }

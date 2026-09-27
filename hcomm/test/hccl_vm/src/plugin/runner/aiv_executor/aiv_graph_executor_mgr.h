/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_AIVGRAPHEXECUTORMGR_H
#define AIV_AIVGRAPHEXECUTORMGR_H

#include <cstdint>
#include <map>
#include <memory>
#include <utility>

#include "aiv_graph_executor.h"

class AivGraphExecutorMgr {
public:
    ~AivGraphExecutorMgr() = default;
    AivGraphExecutorMgr(const AivGraphExecutorMgr&) = delete;
    AivGraphExecutorMgr& operator=(const AivGraphExecutorMgr&) = delete;

    static AivGraphExecutorMgr& GetInstance();

    std::shared_ptr<AivGraphExecutor> GetAivGraphExecutor(uint32_t rankId, uint64_t launchIdx);
    void Reset();

private:
    AivGraphExecutorMgr() = default;

private:
    std::map<std::pair<uint32_t, uint64_t>, std::shared_ptr<AivGraphExecutor>>
        executorMap_{}; // key: <rankId, launchIdx>
};

#endif // AIV_AIVGRAPHEXECUTORMGR_H

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMM_ENGINE_RES_MGR_H
#define COMM_ENGINE_RES_MGR_H

#include <memory>
#include <unordered_map>
#include <vector>
#include <mutex>
#include "comm_engine_res.h"

namespace hcomm {
/**
 * @note 职责：管理不同的通信引擎的资源
 */
class CommEngineResMgr {
public:
    CommEngineResMgr() = default;
    ~CommEngineResMgr() = default;

    // 获取指定引擎类型的资源
    HcclResult GetEngineRes(CommEngineType engineType, std::shared_ptr<CommEngineRes>* engineRes);

private:
    std::unordered_map<CommEngineType, std::shared_ptr<CommEngineRes>> engineResources_{};
    std::mutex mutex_{};
};
} // namespace hcomm
#endif // COMM_ENGINE_RES_MGR_H

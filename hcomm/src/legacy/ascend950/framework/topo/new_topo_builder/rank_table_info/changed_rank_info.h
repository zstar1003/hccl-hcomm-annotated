/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHANGED_RANK_INFO_H
#define CHANGED_RANK_INFO_H

#include <vector>
#include <string>
#include <unordered_map>
#include "nlohmann/json.hpp"
#include "new_rank_info.h"
#include "orion_adapter_rts.h"

namespace Hccl {
class ChangedRankInfo {
public:
    ChangedRankInfo() {};
    std::string version;
    u32 rankCount{0};
    std::vector<NewRankInfo> ranks;
    void Dump() const;
    std::string Describe() const;
    void Deserialize(const nlohmann::json& changedRankInfoJson);
};

} // namespace Hccl

#endif // CHANGED_RANK_INFO_H

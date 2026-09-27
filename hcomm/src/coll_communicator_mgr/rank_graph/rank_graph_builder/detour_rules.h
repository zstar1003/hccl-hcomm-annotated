/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DETOUR_RULES_H
#define DETOUR_RULES_H

#include <unordered_map>
#include <vector>
#include "topo_common_types.h"

namespace Hccl {

const std::unordered_map<LocalId, std::unordered_map<LocalId, std::vector<LocalId>>>& GetDetour2PTable01();

const std::unordered_map<LocalId, std::unordered_map<LocalId, std::vector<LocalId>>>& GetDetour2PTable04();

const std::unordered_map<LocalId, std::unordered_map<LocalId, std::vector<LocalId>>>& GetDetour4PTable0123();

const std::unordered_map<LocalId, std::unordered_map<LocalId, std::vector<LocalId>>>& GetDetour4PTable4567();

const std::unordered_map<LocalId, std::unordered_map<LocalId, std::vector<LocalId>>>& GetDetour4PTable0246();

const std::unordered_map<LocalId, std::unordered_map<LocalId, std::vector<LocalId>>>& GetDetour4PTable1357();

} // namespace Hccl

#endif // DETOUR_RULES_H

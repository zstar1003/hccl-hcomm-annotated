/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_TRANSFORM_TASK_COMMON_V3_H
#define HCCLV2_CCU_TRANSFORM_TASK_COMMON_V3_H

#include <cstdint>
#include <map>
#include <set>
#include <queue>
#include <vector>
#include <memory>
#include <unordered_map>
#include <hccl_types.h>

#include "base.h"
#include "log.h"
#include "data_slice.h"
#include "ccu_microcode_v1.h"
#include "task_ccu.h"
#include "ccu_instr_info.h"
#include "sim_task.h"
#include "../task_def_v3.h"
#include "storage_manager.h"
#include "data_type.h"

using namespace hcomm;
namespace HcclSim {
namespace TaskGraphGeneratorV3 {

    using AllRankChannelInfoV3 = std::map<RankId, std::map<u32, ChannelsPerDie>>;
    extern AllRankChannelInfoV3& g_allRankChannelInfo;

    std::string ParseMSList(const CcuRep::CcuInstr* instr);
} // namespace TaskGraphGeneratorV3
} // namespace HcclSim

#endif

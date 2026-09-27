/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_task_transform_instruct_common_v3.h"
#include "storage_manager.h"

extern HcclSim::TaskGraphGeneratorV3::AllRankChannelInfoV3 g_allRankChannelInfo;

namespace HcclSim {
namespace TaskGraphGeneratorV3 {

    AllRankChannelInfoV3& g_allRankChannelInfo = ::g_allRankChannelInfo;

    std::string ParseMSList(const CcuRep::CcuInstr* instr)
    {
        uint16_t msId[CcuRep::CCU_REDUCE_MAX_MS];
        uint16_t count = instr->v1.add.count;
        for (uint16_t index = 0; index < CcuRep::CCU_REDUCE_MAX_MS; index++) {
            msId[index] = instr->v1.add.msId[index];
        }

        std::string res = "MS[";
        for (uint16_t i = 0; i < count + 2; i++) { // 循环范围 0~count + 2
            if (i == count + 1) {
                res += std::to_string(msId[i] / 0x8000) + ":" + std::to_string(msId[i] % 0x8000) + "]";
            } else {
                res += std::to_string(msId[i] / 0x8000) + ":" + std::to_string(msId[i] % 0x8000) + ", ";
            }
        }
        return res;
    }
} // namespace TaskGraphGeneratorV3
} // namespace HcclSim

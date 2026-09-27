/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SIM_CCU_RESOURCE_COMMON_H
#define HCCL_SIM_CCU_RESOURCE_COMMON_H

#include <array>
#include <cstdint>
#include <vector>

#include "ccu_microcode_v1.h"
#include "ccu_microcode_common_v1.h"
#include "ccu_microcode_common_v2.h"
#include "sim_common_defs.h"

enum class RunnerCcuVersion : uint16_t { CCU_INVALID, CCU_V1, CCU_V2 };

inline const char* RunnerCcuVersionToString(RunnerCcuVersion version)
{
    switch (version) {
        case RunnerCcuVersion::CCU_INVALID:
            return "CCU_INVALID";
        case RunnerCcuVersion::CCU_V1:
            return "CCU_V1";
        case RunnerCcuVersion::CCU_V2:
            return "CCU_V2";
        default:
            return "UNKNOWN";
    }
}
struct CcuInfo {
    int rankId{INT32_MAX};
    int dieId{INT32_MAX};
    // 默认构造：初始化值，保证安全
    CcuInfo() : rankId(INT32_MAX), dieId(INT32_MAX) {}

    // 带参构造：方便赋值
    CcuInfo(int r, int d) : rankId(r), dieId(d) {}

    // = default 让编译器自动生成：
    // 拷贝构造、拷贝赋值、移动构造、移动赋值 —— 全部自动支持
    CcuInfo(const CcuInfo&) = default;
    CcuInfo& operator=(const CcuInfo&) = default;
    ~CcuInfo() = default;
};

struct CcuInstrData {
    std::vector<hcomm::CcuRep::CcuInstr> instrData;
    uint16_t instrCnt{0};
};

using RankChannelInfo = std::array<std::array<CcuInfo, SimCcuV1::MAX_CCU_CHANNEL_NUM>, HcclSim::DIE_NUM>;

#endif // HCCL_SIM_CCU_RESOURCE_COMMON_H

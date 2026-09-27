/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SIM_CCU_RESOURCE_V1_H
#define HCCL_SIM_CCU_RESOURCE_V1_H

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "ccu_resource_common.h"
#include "ccu_simulator.h"
#include "sim_common_defs.h"

class CcuResourceManager;

class CcuResourceV1 {
public:
    CcuResourceV1(int rankId, uint32_t rankSize);
    ~CcuResourceV1() = default;
    void Reset();

private:
    int rankId_{0};
    int rankSize_{0};
    friend CcuResourceManager; // 声明资源管理类为友元类
    std::array<CcuTask, HcclSim::DIE_NUM> ccuTaskInfos_;

    std::array<uint64_t, SimCcuV1::CCU_RESOURCE_XN_MAX> xn_[HcclSim::DIE_NUM];
    std::array<uint64_t, SimCcuV1::CCU_RESOURCE_GSA_MAX> gsa_[HcclSim::DIE_NUM];
    std::array<uint64_t, SimCcuV1::CCU_RESOURCE_CKE_MAX> cke_[HcclSim::DIE_NUM];
    std::array<std::vector<char>, HcclSim::DIE_NUM> ms_;
    std::array<CcuInstrData, HcclSim::DIE_NUM> instrSpace_;

    // channelId映射表：rank进程独立
    RankChannelInfo channelId2RmtRankMap_;

    // ccu模拟执行器
    std::array<std::shared_ptr<CcuSimulator>, HcclSim::DIE_NUM> simulators_;
};

#endif // HCCL_SIM_CCU_RESOURCE_V1_H

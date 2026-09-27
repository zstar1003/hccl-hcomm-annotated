/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_pfe_cfg_generator.h"

#include "hccl_common_v2.h"

#include "ccu_eid_info.h"
#include "ccu_res_specs_legacy.h"
#include <unordered_set>

namespace Hccl {

CcuPfeCfgGenerator& CcuPfeCfgGenerator::GetInstance(const int32_t deviceLogicId)
{
    static CcuPfeCfgGenerator ccuPfeCfgGenerator[MAX_MODULE_DEVICE_NUM + 1];

    if (deviceLogicId < 0 || static_cast<uint32_t>(deviceLogicId) > MAX_MODULE_DEVICE_NUM) {
        THROW<InvalidParamsException>(
            "[CcuPfeCfgGenerator][%s] failed to get instance, devLogicId[%d] "
            "should be less than %u.",
            __func__, deviceLogicId, MAX_MODULE_DEVICE_NUM);
    }

    ccuPfeCfgGenerator[deviceLogicId].Init(deviceLogicId);

    return ccuPfeCfgGenerator[deviceLogicId];
}

void CcuPfeCfgGenerator::Init(const int32_t deviceLogicId)
{
    if (initFlag) {
        return;
    }

    devLogicId = deviceLogicId;

    std::vector<HrtDevEidInfo> eidInfoList;
    (void)CcuEidInfo::GetInstance(devLogicId).GetEidInfo(devLogicId, eidInfoList);
    if (eidInfoList.empty()) {
        HCCL_RUN_INFO("[CcuPfeCfgGenerator][%s] eid infos are empty, devLogicId[%d]", __func__, devLogicId);
        initFlag = true;
        return;
    }

    bool dieEnableFlags[MAX_CCU_IODIE_NUM] = {false, false};
    for (uint8_t i = 0; i < MAX_CCU_IODIE_NUM; i++) {
        const auto& ccuResSpecs = CcuResSpecifications::GetInstance(devLogicId);
        (void)ccuResSpecs.GetDieEnableFlag(i, dieEnableFlags[i]);
    }

    // 不同die的feId独立分配，可能一致，需要die粒度去重
    std::array<std::unordered_set<uint32_t>, MAX_CCU_IODIE_NUM> dieFuncIdSet;
    for (auto& param : eidInfoList) {
        const uint32_t dieId = param.dieId;
        if (dieId >= MAX_CCU_IODIE_NUM) {
            continue; // 跳过HCCL不使用的dieId
        }

        if (!dieEnableFlags[dieId]) {
            continue; // die如果未使能认为无需分配
        }

        const uint32_t feId = param.funcId;
        if (dieFuncIdSet[dieId].find(feId) != dieFuncIdSet[dieId].end()) {
            continue; // 跳过已配置的feId
        }

        uint32_t startJettyCtxId = 0;
        uint32_t startTaJettyId = CCU_START_TA_JETTY_ID;
        uint8_t pfeJettyNum = CCU_PER_DIE_JETTY_RESERVED_NUM;

        PfeJettyCtxCfg cfg{feId, startJettyCtxId, startTaJettyId, pfeJettyNum};
        pfeJettyCtxCfgs[dieId].emplace_back(std::move(cfg));
        dieFuncIdSet[dieId].insert(feId);

        HCCL_RUN_INFO(
            "[CcuPfeCfgGenerator] new pfe cfg set: dieId[%u] feId[%u] startJettyCtxId[%u] "
            "startTaJettyId[%u] pfeJettyNum[%u].",
            dieId, feId, startJettyCtxId, startTaJettyId, pfeJettyNum);
    }

    initFlag = true;
}

std::vector<PfeJettyCtxCfg> CcuPfeCfgGenerator::GetPfeJettyCtxCfg(const uint8_t dieId)
{
    if (dieId >= MAX_CCU_IODIE_NUM) {
        HCCL_WARNING("[CcuPfeCfgGenerator][PfeJettyCtxCfg] invalid dieId[%u]", dieId);
        return std::vector<PfeJettyCtxCfg>();
    }

    if (pfeJettyCtxCfgs[dieId].empty()) {
        HCCL_WARNING("[CcuPfeCfgGenerator][PfeJettyCtxCfg] pfeJettyCtxCfgMap is empty, dieId[%u]", dieId);
        return std::vector<PfeJettyCtxCfg>();
    }

    return pfeJettyCtxCfgs[dieId];
}

}; // namespace Hccl

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_pfe_cfg_mgr.h"

#include <unordered_set>

#include "hccl_common.h"
#include "eid_info_mgr.h"
#include "ccu_res_specs.h"

namespace hcomm {

CcuPfeCfgMgr& CcuPfeCfgMgr::GetInstance(const int32_t userDevId)
{
    static CcuPfeCfgMgr ccuPfeCfgMgr[MAX_MODULE_DEVICE_NUM + 1];

    int32_t validUserDevId = userDevId;
    if (validUserDevId < 0 || static_cast<uint32_t>(validUserDevId) >= MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING(
            "[CcuPfeCfgMgr][%s] use the backup device, userDevId[%d] should be "
            "less than %u.",
            __func__, validUserDevId, MAX_MODULE_DEVICE_NUM);
        validUserDevId = MAX_MODULE_DEVICE_NUM; // 使用备份设备
    }

    ccuPfeCfgMgr[validUserDevId].userDevId_ = userDevId;

    return ccuPfeCfgMgr[validUserDevId];
}

HcclResult CcuPfeCfgMgr::Init()
{
    if (initFlag_) {
        return HcclResult::HCCL_SUCCESS;
    }

    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<uint32_t>(userDevId_), devPhyId_));

    std::vector<DevEidInfo> eidInfos;
    CHK_RET(EidInfoMgr::GetInstance(devPhyId_).GetEidInfos(eidInfos));

    bool dieEnableFlags[CCU_MAX_IODIE_NUM] = {false, false};
    for (uint8_t i = 0; i < CCU_MAX_IODIE_NUM; i++) {
        const auto& ccuResSpecs = CcuResSpecifications::GetInstance(userDevId_);
        (void)ccuResSpecs.GetDieEnableFlag(i, dieEnableFlags[i]);
    }

    // 不同die的feId独立分配，可能一致，需要die粒度去重
    std::array<std::unordered_set<uint32_t>, CCU_MAX_IODIE_NUM> dieFuncIdSet;
    for (auto& param : eidInfos) {
        const uint32_t dieId = param.dieId;
        if (dieId >= CCU_MAX_IODIE_NUM) {
            continue; // 跳过HCCL不使用的dieId
        }

        if (!dieEnableFlags[dieId]) {
            continue; // die如果未使能认为无需分配
        }

        const uint32_t feId = param.funcId;
        if (dieFuncIdSet[dieId].find(feId) != dieFuncIdSet[dieId].end()) {
            continue; // 跳过已配置的feId
        }

        constexpr uint32_t startJettyCtxId = 0;
        constexpr uint32_t startTaJettyId = CCU_START_TA_JETTY_ID;
        constexpr uint8_t pfeJettyNum = CCU_PER_DIE_JETTY_RESERVED_NUM;

        PfeJettyCtxCfg cfg{feId, startJettyCtxId, startTaJettyId, pfeJettyNum};
        pfeJettyCtxCfgs_[dieId].emplace_back(std::move(cfg));
        dieFuncIdSet[dieId].insert(feId);

        HCCL_RUN_INFO(
            "[CcuPfeCfgMgr] new pfe cfg set: dieId[%u] feId[%u] startJettyCtxId[%u] "
            "startTaJettyId[%u] pfeJettyNum[%u].",
            dieId, feId, startJettyCtxId, startTaJettyId, pfeJettyNum);
    }

    initFlag_ = true;
    return HcclResult::HCCL_SUCCESS;
}

HcclResult CcuPfeCfgMgr::Deinit()
{
    for (auto& cfg : pfeJettyCtxCfgs_) {
        cfg.clear();
    }
    initFlag_ = false;
    return HcclResult::HCCL_SUCCESS;
}

std::vector<PfeJettyCtxCfg> CcuPfeCfgMgr::GetPfeJettyCtxCfg(const uint8_t dieId)
{
    if (dieId >= CCU_MAX_IODIE_NUM) {
        HCCL_WARNING("[CcuPfeCfgMgr][PfeJettyCtxCfg] invalid dieId[%u]", dieId);
        return std::vector<PfeJettyCtxCfg>();
    }

    if (pfeJettyCtxCfgs_[dieId].empty()) {
        HCCL_WARNING("[CcuPfeCfgMgr][PfeJettyCtxCfg] pfeJettyCtxCfgMap is empty, dieId[%u]", dieId);
        return std::vector<PfeJettyCtxCfg>();
    }

    return pfeJettyCtxCfgs_[dieId];
}

} // namespace hcomm

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_PFE_CFG_MGR_H
#define CCU_PFE_CFG_MGR_H

#include <array>
#include <cstdint>
#include <vector>
#include "hccl_types.h"
#include "ccu_dev_mgr_imp.h"

namespace hcomm {

struct PfeJettyCtxCfg {
    uint32_t feId{0};
    uint32_t startJettyCtxId{0};
    uint32_t startTaJettyId{0};
    uint8_t size{0};

    PfeJettyCtxCfg() = default;
    PfeJettyCtxCfg(uint32_t feId, uint32_t startJettyCtxId, uint32_t startTaJettyId, uint8_t size)
        : feId(feId),
          startJettyCtxId(startJettyCtxId),
          startTaJettyId(startTaJettyId),
          size(size)
    {}
};

class CcuPfeCfgMgr {
public:
    static CcuPfeCfgMgr& GetInstance(const int32_t userDevId);
    HcclResult Init();
    HcclResult Deinit();

    std::vector<PfeJettyCtxCfg> GetPfeJettyCtxCfg(const uint8_t dieId);

private:
    explicit CcuPfeCfgMgr() = default;
    ~CcuPfeCfgMgr() = default;
    CcuPfeCfgMgr(const CcuPfeCfgMgr& that) = delete;
    CcuPfeCfgMgr& operator=(const CcuPfeCfgMgr& that) = delete;

    HcclResult SetPfeJettyCtxCfgMap(const int32_t userDevId);

private:
    bool initFlag_{false};
    int32_t userDevId_{0};
    uint32_t devPhyId_{0};
    // 每个iodie上PfeJettyCtxCfg的映射关系
    std::array<std::vector<PfeJettyCtxCfg>, CCU_MAX_IODIE_NUM> pfeJettyCtxCfgs_{};
};

}; // namespace hcomm

#endif // CCU_PFE_CFG_MGR_H

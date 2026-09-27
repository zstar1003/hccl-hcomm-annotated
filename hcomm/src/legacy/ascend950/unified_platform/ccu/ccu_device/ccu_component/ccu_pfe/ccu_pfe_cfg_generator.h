/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_PFE_CFG_GENERATOR_H
#define CCU_PFE_CFG_GENERATOR_H

#include <array>
#include <vector>
#include "hccl/hccl_types.h"
#include "ccu_device_manager.h"

namespace Hccl {

constexpr uint8_t MAX_CCU_INNER_FE_IDX = 7;  // 框内最大FE Index
constexpr uint8_t INNER_MAX_CCU_PFE_NUM = 6; // 2D fullmesh组网框内x/y轴最大FE个数
constexpr uint8_t ONE_CCU_PFE_USE_JETTY_NUM
    = 23; // 框内每个FE使用的Jetty个数
          // 优先分配框内FE，每个FE按最大数量分配，剩余Jetty分配个出框，每个die仅支持1个出框FE
constexpr uint8_t INNER_MAX_TA_JETTY_ID_OFFSET = (INNER_MAX_CCU_PFE_NUM - 2) * ONE_CCU_PFE_USE_JETTY_NUM;

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

class CcuPfeCfgGenerator {
public:
    CcuPfeCfgGenerator(const CcuPfeCfgGenerator& that) = delete;

    CcuPfeCfgGenerator& operator=(const CcuPfeCfgGenerator& that) = delete;

    static CcuPfeCfgGenerator& GetInstance(const int32_t deviceLogicId);
    std::vector<PfeJettyCtxCfg> GetPfeJettyCtxCfg(const uint8_t dieId);

private:
    bool initFlag{false};
    int32_t devLogicId{0};
    std::array<std::vector<PfeJettyCtxCfg>, MAX_CCU_IODIE_NUM> pfeJettyCtxCfgs; // 每个iodie上PfeJettyCtxCfg的映射关系

    explicit CcuPfeCfgGenerator() = default;
    ~CcuPfeCfgGenerator() = default;

    void Init(const int32_t deviceLogicId);
    HcclResult SetPfeJettyCtxCfgMap(const int32_t logicDeviceId);
};

}; // namespace Hccl

#endif // CCU_PFE_CFG_GENERATOR_H

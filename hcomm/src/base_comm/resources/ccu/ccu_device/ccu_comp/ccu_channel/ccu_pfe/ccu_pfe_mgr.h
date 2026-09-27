/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_PFE_MGR_H
#define CCU_PFE_MGR_H

#include <vector>
#include <unordered_map>

#include "ccu_dev_mgr_imp.h"
#include "ccu_pfe_cfg_mgr.h"

namespace hcomm {

struct PfeJettyStrategy {
    uint32_t feId;                 // FE index.
    uint32_t pfeId;                // PFE ctx index.
    uint32_t startTaJettyId;       // PFE assigned to CCU starting jetty index.
    uint8_t size;                  // PFE assigned to CCU jetty num.
    uint32_t startLocalJettyCtxId; // PFE assigned to CCU starting local jetty index.
};

#pragma pack(push, 1)
struct PfeCtx {
    uint16_t startJettyId; // PFE assigned to CCU starting jetty index.
    /********2 Bytes**********/

    uint16_t jettyNum : 7;             // PFE assigned to CCU total jetty num.
    uint16_t startLocalJettyCtxId : 7; // CCU maintained starting jetty context index.
    uint16_t rsvBit : 2;
    /********4 Bytes**********/

    uint16_t rsv[2];
    /********8 Bytes**********/
};
#pragma pack(pop)

class CcuPfeMgr {
public:
    CcuPfeMgr(const int32_t userDevId, const uint8_t dieId, const uint32_t devPhyId)
        : userDevId_(userDevId),
          dieId_(dieId),
          devPhyId_(devPhyId) {};
    CcuPfeMgr() = default;
    ~CcuPfeMgr() = default;

    HcclResult Init();
    HcclResult GetPfeStrategy(uint32_t feId, PfeJettyStrategy& pfeJettyStrategy) const;

private:
    int32_t userDevId_{0};
    uint8_t dieId_{0};
    uint32_t devPhyId_{0};

    std::unordered_map<uint32_t, struct PfeJettyStrategy> pfeJettyMap_;
};

}; // namespace hcomm

#endif // CCU_PFE_MGR_H

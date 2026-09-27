/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_JETTY_CTX_MGR_H
#define CCU_JETTY_CTX_MGR_H

#include <vector>

#include "ccu_device_pub.h"
#include "ccu_pfe_mgr.h"
#include "ccu_wqebb_mgr.h"

namespace hcomm {

constexpr uint8_t DB_ADDR_TYPE = 1;
constexpr uint8_t TOKEN_VALUE_IS_VALIDE = 1;

constexpr uint32_t MASK_TK_ID_LOW = 0x000000FF;
constexpr uint32_t MASK_TK_ID_HIGH = 0x00000FFF;

constexpr uint32_t MASK_TK_VALUE_LOW = 0x0000000F;
constexpr uint32_t MASK_TK_VALUE_MID = 0x0000FFFF;
constexpr uint32_t MASK_TK_VALUE_HIGH = 0x00000FFF;

constexpr uint32_t MASK_WQEBB_IDX_LOW = 0x0000000F;
constexpr uint32_t MASK_WQEBB_IDX_HIGH = 0x000000FF;

constexpr uint16_t CCU_HARDWARE_DEFAULT_VALUE = 0x0;

#pragma pack(push, 1)
struct LocalJettyCtxData {
    uint16_t doorbellAddr[4] = {0}; // jetty doorbell addr
    /********8 Bytes**********/

    uint16_t pfeIdx : 4;            // jetty relegation use PFE num.
    uint16_t ioDieId : 1;           // 0: local jetty use IODIE0, 1: local jetty use IODIE1.
    uint16_t doorbellAddrType : 1;  // doorbell addr type: 0:PA, 1:VA.
    uint16_t tokenValueIsValid : 1; // doorbell addr relate token value valid type: invalid(0), valid(1).
    uint16_t cqeErrValue : 1;       // v1 not used
    uint16_t tokenIdLow : 8;
    /********10 Bytes**********/

    uint16_t tokenIdHigh : 12;
    uint16_t tokenValueLow : 4;
    /********12 Bytes**********/

    uint16_t tokenValueMiddle{0};
    /********14 Bytes**********/

    uint16_t tokenValueHigh : 12;
    // JFS/jetty SQE basic block  left shifts bit num, 1: SQBuffDepth = 2 ^ sqeBasicBlockLeftShifts, 4: SQ->16 WQEBB
    uint16_t sqeBasicBlockLeftShifts : 4;
    /********16 Bytes**********/

    uint16_t pi{0};    // ccu hardware maintain
    uint16_t ci{0};    // ccu hardware maintain
    uint16_t maxCi{0}; // ccu hardware maintain
    /********22 Bytes**********/

    uint16_t oooCqeCnt : 12; // ccu hardware maintain
    uint16_t startWqeBasicBlockIdxLow : 4;
    /********24 Bytes**********/

    uint16_t startWqeBasicBlockIdxHigh : 8;
    uint16_t doorbellSendState : 2; // ccu hardware maintain
    uint16_t rsvSixBits : 6;
    /********26 Bytes**********/

    uint16_t rsvs[3]{0};
    /********32 Bytes**********/

    LocalJettyCtxData()
        : pfeIdx(0),
          ioDieId{0},
          doorbellAddrType{0},
          tokenValueIsValid{0},
          cqeErrValue{0},
          tokenIdLow{0},
          tokenIdHigh{0},
          tokenValueLow(0),
          tokenValueHigh{0},
          sqeBasicBlockLeftShifts{0},
          oooCqeCnt{0},
          startWqeBasicBlockIdxLow{0},
          startWqeBasicBlockIdxHigh{0},
          doorbellSendState{0},
          rsvSixBits{0}
    {}
};
#pragma pack(pop)

LocalJettyCtxData
BuildJettyCtxData(const uint8_t dieId, const uint32_t pfeId, const JettyInfo& jettyInfo, const JettyCfg& jettyCfg);

HcclResult ConfigJettyCtxData(
    const int32_t userDevId, const uint8_t dieId, const uint32_t devPhyId, const uint16_t startJettyCtxId,
    std::vector<LocalJettyCtxData>& jettyCtxData);

void DumpJettyCtxData(const LocalJettyCtxData& tmp);

class CcuJettyCtxMgr {
public:
    CcuJettyCtxMgr(const int32_t userDevId, const uint8_t dieId, const uint32_t devPhyId);
    CcuJettyCtxMgr() = default;
    virtual ~CcuJettyCtxMgr() = default;
    virtual HcclResult Init() = 0;

    virtual HcclResult
    Alloc(const uint32_t feId, const uint32_t jettyNum, const uint32_t sqSize, std::vector<JettyInfo>& jettyInfos)
        = 0;
    virtual HcclResult
    Config(const uint32_t feId, const std::vector<JettyInfo>& jettyInfos, const std::vector<JettyCfg>& jettyCfgs)
        = 0;
    virtual HcclResult Release(const uint32_t feId, const std::vector<JettyInfo>& jettyInfos) = 0;

protected:
    int32_t userDevId_{0};
    uint8_t dieId_{0};
    uint32_t devPhyId_{0};

    uint32_t jettySpecNum_{0};
    uint64_t ccuResBaseVa_{0};

    std::unique_ptr<CcuWqeBBMgr> wqeBBMgr_{nullptr};
    CcuPfeMgr pfeMgr_{};

    HcclResult TryAllocWqeBBResource(
        const uint32_t sqSize, const uint32_t jettyCtxStartId, const uint32_t taJettyStartId,
        const CcuJettyType jettyType, std::vector<JettyInfo>& jettyInfos);
    HcclResult ReleaseWqeBBResource(const std::vector<JettyInfo>& jettyInfos);
    HcclResult
    CheckIfJettyCfgsValid(const std::vector<JettyInfo>& jettyInfos, const std::vector<JettyCfg>& jettyCfgs) const;
};

}; // namespace hcomm

#endif // CCU_JETTY_CTX_MGR_H

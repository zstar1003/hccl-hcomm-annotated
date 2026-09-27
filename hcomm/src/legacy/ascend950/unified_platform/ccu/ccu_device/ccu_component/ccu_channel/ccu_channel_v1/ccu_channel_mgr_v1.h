/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_CHANNEL_MGR_V1_H
#define HCCL_CCU_CHANNEL_MGR_V1_H

#include "ccu_channel_mgr.h"
#include "ccu_jetty_ctx_mgr_v1_legacy.h"

namespace Hccl {

constexpr uint16_t TOKEN_VALUE_VALID = 1;
constexpr uint16_t REMOTE_CCU_VA_RIGHT_SHIFT_NUM = 23;

constexpr uint16_t MASK_START_JETTY_ID_LOW = 0x000F;
constexpr uint16_t MASK_START_JETTY_ID_HIGH = 0x0FFF;

constexpr uint8_t MASK_JETTY_NUM_LOW = 0x000F;
constexpr uint8_t MASK_JETTY_NUM_HIGH = 0x0007;

constexpr uint32_t MASK_TOKEN_ID_LOW = 0x00000FFF;
constexpr uint32_t MASK_TOKEN_ID_HIGH = 0x000000FF;

constexpr uint32_t MASK_TOKEN_VALUE_LOW = 0x000000FF;
constexpr uint32_t MASK_TOKEN_VALUE_MID = 0x0000FFFF;
constexpr uint32_t MASK_TOKEN_VALUE_HIGH = 0x000000FF;

constexpr uint64_t MASK_VA_LOW = 0x00000000000000FF;
constexpr uint64_t MASK_VA_MID = 0x000000000000FFFF;
constexpr uint64_t MASK_VA_HIGH = 0x000000000000FFFF;
constexpr uint64_t MASK_VA_HIGHER = 0x0000000000000001;

#pragma pack(push, 1)
struct ChannelDataV1 {
    uint8_t eidRaw[URMA_EID_LEN] = {0};
    /********16 Bytes**********/

    uint16_t vtpLow{0};
    /********18 Bytes**********/

    uint16_t vtpHigh : 8;
    uint16_t srcPfeId : 4;
    uint16_t startJettyIdLow : 4;
    /********20 Bytes**********/

    uint16_t startJettyIdHigh : 12;
    uint16_t jettyNumLow : 4;
    /********22 Bytes**********/

    uint16_t jettyNumHigh : 3;
    uint16_t ioDieId : 1;
    uint16_t dstTokenIdLow : 12;
    /********24 Bytes**********/

    uint16_t dstTokenIdHigh : 8;
    uint16_t dstTokenValueLow : 8;
    /********26 Bytes**********/

    uint16_t dstTokenValueMiddle{0};
    /********28 Bytes**********/

    uint16_t dstTokenValueHigh : 8;
    uint16_t dstVaLow : 8;
    /********30 Bytes**********/

    uint16_t dstVaMiddle{0};
    /********32 Bytes**********/

    uint16_t dstVaHigh{0};
    /********34 Bytes**********/

    uint16_t dstVaHigher : 1;
    uint16_t dstTokenValueValid : 1;
    uint16_t rsv14Bits : 14;
    /********36 Bytes**********/

    uint16_t rsvs[14] = {0};
    /********64 Bytes**********/

    ChannelDataV1()
        : vtpHigh(0),
          srcPfeId{0},
          startJettyIdLow{0},
          startJettyIdHigh{0},
          jettyNumLow{0},
          jettyNumHigh{0},
          ioDieId{0},
          dstTokenIdLow(0),
          dstTokenIdHigh{0},
          dstTokenValueLow{0},
          dstTokenValueHigh{0},
          dstVaLow{0},
          dstVaHigher{0},
          dstTokenValueValid{0},
          rsv14Bits{0}
    {}
};
#pragma pack(pop)

class CcuChannelMgrV1 : public CcuChannelMgr {
public:
    CcuChannelMgrV1(const int32_t devLogicId, const uint8_t dieId, const uint32_t devPhyId);

    CcuChannelMgrV1() = default;
    ~CcuChannelMgrV1() override = default;

    HcclResult Alloc(const ChannelPara& channelPara, std::vector<ChannelInfo>& channelInfos) override;
    HcclResult Config(const ChannelCfg& channelCfg) override;
    HcclResult Release(const uint32_t channelId) override;

private:
    uint32_t strategy{0};

    CcuJettyCtxMgrV1 jettyCtxMgr{};
};

}; // namespace Hccl

#endif

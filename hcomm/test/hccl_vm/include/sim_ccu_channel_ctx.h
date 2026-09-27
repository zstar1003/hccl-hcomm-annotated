/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_CCU_CHANNEL_CTX_H
#define SIM_CCU_CHANNEL_CTX_H

#include <vector>

#include "hccl_types.h"
#include "sim_common_defs.h"
namespace HcclSim {
constexpr uint32_t SHIFT_2BITS = 2;
constexpr uint32_t SHIFT_4BITS = 4;
constexpr uint32_t SHIFT_8BITS = 8;
constexpr uint32_t SHIFT_12BITS = 12;
constexpr uint32_t SHIFT_16BITS = 16;
constexpr uint32_t SHIFT_20BITS = 20;
constexpr uint32_t SHIFT_24BITS = 24;
constexpr uint32_t SHIFT_40BITS = 40;

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
struct ChannelCtxDataV1 {
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

    ChannelCtxDataV1()
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

struct ChannelDataV2 {
    uint8_t eidRaw[URMA_EID_LEN] = {0};
    uint16_t vtpLow{0};
    uint16_t vtpHigh : 8;
    uint16_t ioDieId : 1;
    uint16_t rsv3Bits : 7;

    uint16_t resv[6] = {0};
    ChannelDataV2() : vtpHigh(0), ioDieId(0), rsv3Bits(0) {}
};

#pragma pack(pop)
} // namespace HcclSim
#endif

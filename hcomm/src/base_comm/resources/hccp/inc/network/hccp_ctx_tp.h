/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCP_CTX_TP_H
#define HCCP_CTX_TP_H

#include "hccp_common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TP_ATTR_SIP_MASK 0x4
#define TP_ATTR_DIP_MASK 0x8
#define TP_ATTR_SMAC_MASK 0x10
#define TP_ATTR_DMAC_MASK 0x20

union GetTpCfgFlag {
    struct {
        uint32_t ctp : 1;
        uint32_t rtp : 1;
        uint32_t utp : 1;
        uint32_t uboe : 1;
        uint32_t preDefined : 1;
        uint32_t dynamicDefined : 1;
        uint32_t udp : 5;
        uint32_t groupId : 15;
        uint32_t reserved : 6;
    } bs;
    uint32_t value;
};

struct GetTpCfg {
    union GetTpCfgFlag flag;
    enum TransportModeT transMode;
    union HccpEid localEid;
    union HccpEid peerEid;
};

#define HCCP_MAX_TPID_INFO_NUM 128

struct HccpTpInfo {
    uint64_t tpHandle;
    uint32_t resv;
};

#pragma pack(1)
struct TpAttr {
    uint8_t retryTimesInit : 3;  // corresponding bitmap bit: 0
    uint8_t at : 5;              // corresponding bitmap bit: 1
    uint8_t sip[16U];            // corresponding bitmap bit: 2
    uint8_t dip[16U];            // corresponding bitmap bit: 3
    uint8_t sma[6U];             // corresponding bitmap bit: 4
    uint8_t dma[6U];             // corresponding bitmap bit: 5
    uint16_t vlanId : 12;        // corresponding bitmap bit: 6
    uint8_t vlanEn : 1;          // corresponding bitmap bit: 7
    uint8_t dscp : 6;            // corresponding bitmap bit: 8
    uint8_t atTimes : 5;         // corresponding bitmap bit: 9
    uint8_t sl : 4;              // corresponding bitmap bit: 10
    uint8_t ttl;                 // corresponding bitmap bit: 11
    uint16_t ackUdpSrcport;      // corresponding bitmap bit: 12
    uint16_t dataUdpSrcport;     // corresponding bitmap bit: 13
    uint8_t udpSrcportRange : 4; // corresponding bitmap bit: 14
    uint8_t sprayEn : 1;         // corresponding bitmap bit: 15
    uint8_t udpGlobalEn : 1;     // corresponding bitmap bit: 16
    uint8_t reserve0 : 2;
    uint16_t slBitmap;
    uint8_t dscpConfigMode : 1;
    uint8_t reserve1 : 7;
    uint8_t reserved[70];
};
#pragma pack()

#ifdef __cplusplus
}
#endif

#endif // HCCP_CTX_TP_H

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UDMA_DATA_STRUCT_STUB_H
#define UDMA_DATA_STRUCT_STUB_H

#include <cstdint>
#include <string>

const uint32_t UDMA_SQE_RMT_EID_SIZE = 4;

enum UdmaSqOpcode {
    UDMA_OPC_SEND = 0x0,
    UDMA_OPC_SEND_WITH_IMM = 0x1,
    UDMA_OPC_SEND_WITH_INVALID = 0x2,
    UDMA_OPC_WRITE = 0x3,
    UDMA_OPC_WRITE_WITH_IMM = 0x4,
    UDMA_OPC_WRITE_WITH_NOTIFY = 0x5,
    UDMA_OPC_READ = 0x6,
    UDMA_OPC_CAS = 0x7,
    UDMA_OPC_FAA = 0xb,
    UDMA_OPC_NOP = 0x11,
    UDMA_OPC_INVALID = 0x12
};

struct UdfExtDate { // UDF扩展数据
    uint32_t udfType : 8;
    uint32_t reduceType : 4;
    uint32_t reduceOp : 4;
    uint32_t rsv : 16;
};

struct UdmaSqeCommon {
    uint32_t sqeBbIdx : 16;
    uint32_t placeOdr : 2;
    uint32_t compOrder : 1;
    uint32_t fence : 1;
    uint32_t se : 1;
    uint32_t cqe : 1;
    uint32_t inlineEn : 1;
    uint32_t udfFlag : 1;
    uint32_t rsv : 4;
    uint32_t tokenEn : 1;
    uint32_t rmtJettyType : 2;
    uint32_t owner : 1;
    uint32_t targetHint : 8;
    uint32_t opcode : 8;
    uint32_t rsv1 : 6;
    uint32_t inlineMsgLen : 10;
    uint32_t tpn : 24;
    uint32_t sgeNum : 8;
    uint32_t rmtObjId : 20;
    uint32_t rsv2 : 12;
    uint32_t rmtEid[UDMA_SQE_RMT_EID_SIZE];
    uint32_t rmtTokenValue;
    union {
        uint32_t rsv3;
        UdfExtDate udfData;
    } inlinedata;

    uint32_t rmtAddrLow;
    uint32_t rmtAddrHigh;
};

struct UdmaNormalSge {
    uint32_t length;
    uint32_t tokenId;
    uint32_t dataAddrLow;
    uint32_t dataAddrHigh;

    std::string Desc() const
    {
        return "length = " + std::to_string(length) + " dataAddrLow = " + std::to_string(dataAddrLow)
               + " dataAddrHigh = " + std::to_string(dataAddrHigh);
    }
};

struct UdmaInlineData {
    uint8_t data[16];
};

union LocalValueU {
    UdmaNormalSge sge;
    UdmaInlineData inlineData;
};

struct UdmaSqeWrite {
    struct UdmaSqeCommon comm;
    union LocalValueU u;
};

struct UdmaSqeNotify {
    uint32_t notifyTokenId : 20;
    uint32_t rsv : 12;
    uint32_t notifyTokenValue;
    uint32_t notifyAddrLow;
    uint32_t notifyAddrHigh;
    uint32_t notifyDataLow;
    uint32_t notifyDataHigh;
    std::string Desc() const
    {
        return "notifyAddrLow = " + std::to_string(notifyAddrLow) + " notifyAddrHigh = "
               + std::to_string(notifyAddrHigh) + " notifyDataLow = " + std::to_string(notifyDataLow)
               + " notifyDataHigh = " + std::to_string(notifyDataHigh);
    }
};

struct UdmaSqeWriteWithNotify {
    struct UdmaSqeCommon comm;
    struct UdmaSqeNotify notify;
    uint32_t rsv1;
    uint32_t rsv2;
    union LocalValueU localU;
};

#endif

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RA_HDC_TLV_H
#define RA_HDC_TLV_H

#include "ra_tlv.h"
#include "ra_rs_comm.h"
#include "ra_hdc.h"

union OpTlvInitDataV1 {
    struct {
        unsigned int phyId;
        unsigned int moduleType;
        uint32_t reserved[RA_RSVD_NUM_61];
    } txData;

    struct {
        unsigned int bufferSize;
        uint32_t reserved[RA_RSVD_NUM_63];
    } rxData;
};

union OpTlvInitData {
    struct {
        unsigned int phyId;
        uint32_t reserved[RA_RSVD_NUM_62];
    } txData;

    struct {
        unsigned int bufferSize;
        uint32_t reserved[RA_RSVD_NUM_63];
    } rxData;
};

union OpTlvDeinitData {
    struct {
        unsigned int phyId;
        uint32_t reserved[RA_RSVD_NUM_61];
    } txData;

    struct {
        uint32_t reserved[RA_RSVD_NUM_64];
    } rxData;
};

union OpTlvRequestData {
    struct {
        struct TlvRequestMsgHead head;
        char data[MAX_TLV_MSG_DATA_LEN];
    } txData;

    struct {
        unsigned int recvBytes;
        char recvData[MAX_TLV_MSG_DATA_LEN];
    } rxData;
};

union OpTlvRequestDataV2 {
    struct {
        struct TlvRequestMsgHead head;
        char data[MAX_TLV_MSG_DATA_LEN_V2];
    } txData;

    struct {
        unsigned int recvBytes;
        char recvData[MAX_TLV_MSG_DATA_LEN_V2];
    } rxData;
};

union OpTlvRequestDataCommon {
    struct {
        void *tlvData;
        void *head;
        char *data;
        unsigned int dataLength;
        int size;
        int opcode;
    } txData;

    struct {
        void *tlvData;
        unsigned int *recvBytes;
        char *recvData;
        unsigned int recvDataLength;
        int size;
        int opcode;
    } rxData;
};

union OpCustomChannelData {
    struct {
        unsigned int phyId;
        struct CustomChanInfoIn info;
        unsigned int rsvd[64U];
    } txData;

    struct {
        struct CustomChanInfoOut info;
        unsigned int rsvd[64U];
    } rxData;
};

#define TLV_SETUP_TX(isV1, tlvData, commonData)                                                                        \
    do {                                                                                                               \
        unsigned int headoff = (isV1) ? offsetof(union OpTlvRequestData, txData.head)                                  \
                                      : offsetof(union OpTlvRequestDataV2, txData.head);                               \
        unsigned int dataoff = (isV1) ? offsetof(union OpTlvRequestData, txData.data)                                  \
                                      : offsetof(union OpTlvRequestDataV2, txData.data);                               \
        (commonData).txData.tlvData = (tlvData);                                                                       \
        (commonData).txData.head = (tlvData) + headoff;                                                                \
        (commonData).txData.data = (tlvData) + dataoff;                                                                \
        (commonData).txData.dataLength = (isV1) ? MAX_TLV_MSG_DATA_LEN : MAX_TLV_MSG_DATA_LEN_V2;                      \
        (commonData).txData.size = (isV1) ? sizeof(union OpTlvRequestData) : sizeof(union OpTlvRequestDataV2);         \
        (commonData).txData.opcode = (isV1) ? RA_RS_TLV_REQUEST : RA_RS_TLV_REQUEST_V2;                                \
    } while (0)

#define TLV_SETUP_RX(isV1, tlvData, commonData)                                                                        \
    do {                                                                                                               \
        unsigned int recvBytesoff = (isV1) ? offsetof(union OpTlvRequestData, rxData.recvBytes)                        \
                                           : offsetof(union OpTlvRequestDataV2, rxData.recvBytes);                     \
        unsigned int recvDataoff = (isV1) ? offsetof(union OpTlvRequestData, rxData.recvData)                          \
                                          : offsetof(union OpTlvRequestDataV2, rxData.recvData);                       \
        (commonData).rxData.tlvData = (tlvData);                                                                       \
        (commonData).rxData.recvBytes = (tlvData) + recvBytesoff;                                                      \
        (commonData).rxData.recvData = (tlvData) + recvDataoff;                                                        \
        (commonData).rxData.recvDataLength = (isV1) ? MAX_TLV_MSG_DATA_LEN : MAX_TLV_MSG_DATA_LEN_V2;                  \
        (commonData).rxData.size = (isV1) ? sizeof(union OpTlvRequestData) : sizeof(union OpTlvRequestDataV2);         \
        (commonData).rxData.opcode = (isV1) ? RA_RS_TLV_REQUEST : RA_RS_TLV_REQUEST_V2;                                \
    } while (0)

int RaHdcTlvInit(struct RaTlvHandle *tlvHandle);
int RaHdcTlvDeinit(struct RaTlvHandle *tlvHandle);
int RaHdcTlvRequest(struct RaTlvHandle *tlvHandle, unsigned int moduleType, struct TlvMsg *sendMsg,
    struct TlvMsg *recvMsg);
#endif // RA_HDC_TLV_H
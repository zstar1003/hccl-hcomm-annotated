/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdlib.h>
#include <errno.h>
#include "config_log.h"
#include "ra_hdc_tlv.h"
#include "ra_rs_err.h"
#include "rs_tlv.h"
#include "ra_adp.h"
#include "ra_adp_tlv.h"

struct RsTlvOps {
    int (*tlvInit)(unsigned int phyId, unsigned int *bufferSize);
    int (*tlvDeinit)(unsigned int phyId);
    int (*tlvRequest)(struct TlvRequestMsgHead *head, char *dataIn, char *dataOut, unsigned int *bufferSize,
        unsigned int dataMaxLength);
    int (*ccuCustomChannel)(const struct CustomChanInfoIn *in, struct CustomChanInfoOut *out);
};

struct RsTlvOps gRaRsTlvOps = {
    .tlvInit = RsTlvInit,
    .tlvDeinit = RsTlvDeinit,
    .tlvRequest = RsTlvRequest,
    .ccuCustomChannel = RsCtxCustomChannel,
};

int RaRsTlvInitV1(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)inBuf;
    (void)outBuf;
    (void)outLen;
    (void)rcvBufLen;
    hccp_warn_init("Tlv init is not supported in this version.");
    *opResult = -ENOTSUPP;
    return 0;
}

int RaRsTlvInit(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpTlvInitData *dataOut = (union OpTlvInitData *)(outBuf + sizeof(struct MsgHead));
    union OpTlvInitData *dataIn = (union OpTlvInitData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpTlvInitData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsTlvOps.tlvInit(dataIn->txData.phyId, &dataOut->rxData.bufferSize);
    CHK_PRT_RETURN(*opResult == -ENOTSUPP, hccp_warn_init("tlv_init unsuccessful ret[%d]", *opResult), 0);
    if (*opResult != 0) {
        hccp_err("tlv_init failed ret[%d]", *opResult);
    }

    return 0;
}

int RaRsTlvDeinit(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outBuf;
    (void)outLen;
    union OpTlvDeinitData *dataIn = (union OpTlvDeinitData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpTlvDeinitData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsTlvOps.tlvDeinit(dataIn->txData.phyId);
    if (*opResult != 0) {
        hccp_err("tlv_deinit failed ret[%d]", *opResult);
    }

    return 0;
}

int RaRsTlvRequest(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpTlvRequestData *dataOut = (union OpTlvRequestData *)(outBuf + sizeof(struct MsgHead));
    union OpTlvRequestData *dataIn = (union OpTlvRequestData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpTlvRequestData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsTlvOps.tlvRequest(&dataIn->txData.head, dataIn->txData.data, dataOut->rxData.recvData,
        &dataOut->rxData.recvBytes, MAX_TLV_MSG_DATA_LEN);

    CHK_PRT_RETURN(*opResult == -EUSERS || *opResult == -ENOTSUPP, hccp_warn_others("tlv request unsuccessful"), 0);
    if (*opResult != 0) {
        hccp_err("tlv_request failed ret[%d]", *opResult);
    }

    return 0;
}

int RaRsTlvRequestV2(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpTlvRequestDataV2 *dataOut = (union OpTlvRequestDataV2 *)(outBuf + sizeof(struct MsgHead));
    union OpTlvRequestDataV2 *dataIn = (union OpTlvRequestDataV2 *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpTlvRequestDataV2), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsTlvOps.tlvRequest(&dataIn->txData.head, dataIn->txData.data, dataOut->rxData.recvData,
        &dataOut->rxData.recvBytes, MAX_TLV_MSG_DATA_LEN_V2);

    CHK_PRT_RETURN(*opResult == -EUSERS || *opResult == -ENOTSUPP, hccp_warn_others("tlv request v2 unsuccessful"), 0);
    if (*opResult != 0) {
        hccp_err("tlv_request_v2 failed ret[%d]", *opResult);
    }

    return 0;
}

int RaRsCustomChannel(char *inBuf, char *outBuf, int *outLen, int *opResult, int rcvBufLen)
{
    (void)outLen;
    union OpCustomChannelData *opDataOut = (union OpCustomChannelData *)(outBuf + sizeof(struct MsgHead));
    union OpCustomChannelData *opData = (union OpCustomChannelData *)(inBuf + sizeof(struct MsgHead));

    HCCP_CHECK_PARAM_LEN_RET_HOST(sizeof(union OpCustomChannelData), sizeof(struct MsgHead), rcvBufLen, opResult);

    *opResult = gRaRsTlvOps.ccuCustomChannel(&opData->txData.info, &opDataOut->rxData.info);
    if (*opResult != 0) {
        hccp_err("[ccu]custom channel failed, ret[%d], phyId[%u]", *opResult, opData->txData.phyId);
    }

    return 0;
}
